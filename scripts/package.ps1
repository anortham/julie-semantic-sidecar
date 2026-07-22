param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "apple-arm64-metal-portable",
        "linux-x64-vulkan-portable",
        "windows-x64-vulkan-portable",
        "linux-x64-cuda-vendor",
        "windows-x64-cuda-vendor"
    )]
    [string]$Profile,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$settings = switch ($Profile) {
    "apple-arm64-metal-portable" { @{ Target = "aarch64-apple-darwin"; Backend = "metal"; Tier = "portable"; Features = "metal" } }
    "linux-x64-vulkan-portable" { @{ Target = "x86_64-unknown-linux-gnu"; Backend = "vulkan"; Tier = "portable"; Features = "vulkan,dynamic-backends" } }
    "windows-x64-vulkan-portable" { @{ Target = "x86_64-pc-windows-msvc"; Backend = "vulkan"; Tier = "portable"; Features = "vulkan,dynamic-backends" } }
    "linux-x64-cuda-vendor" { @{ Target = "x86_64-unknown-linux-gnu"; Backend = "cuda"; Tier = "vendor"; Features = "cuda,dynamic-backends" } }
    "windows-x64-cuda-vendor" { @{ Target = "x86_64-pc-windows-msvc"; Backend = "cuda"; Tier = "vendor"; Features = "cuda,dynamic-backends" } }
}

$hostTriple = (& rustc -vV | Select-String '^host: ').Line.Substring(6)
if ($hostTriple -ne $settings.Target) {
    throw "profile $Profile must run on $($settings.Target), current host is $hostTriple"
}

$effectiveRustFlags = "$($env:RUSTFLAGS) $($env:CARGO_ENCODED_RUSTFLAGS)" -replace '\s', ''
if ($effectiveRustFlags.Contains("target-cpu=native")) {
    throw "effective Rust flags must not contain -Ctarget-cpu=native"
}

$version = (Select-String -Path Cargo.toml -Pattern '^version = "([^"]+)"').Matches[0].Groups[1].Value
$isWindows = $settings.Target.Contains("windows")
$exe = if ($isWindows) { "julie-semantic-sidecar.exe" } else { "julie-semantic-sidecar" }
$helper = if ($isWindows) { "julie-package-manifest.exe" } else { "julie-package-manifest" }
$archiveKind = if ($isWindows) { "zip" } else { "tar.gz" }

$cargoArguments = @(
    "build", "--release", "--target", $settings.Target,
    "--features", $settings.Features, "--bins", "--message-format=json"
)
$messages = & cargo @cargoArguments
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
$nativeOut = @(
    $messages |
        ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $_.reason -eq "build-script-executed" -and $_.package_id -like "*llama-cpp-sys-2*" } |
        ForEach-Object { $_.out_dir }
)
if ($nativeOut.Count -ne 1) { throw "expected one llama-cpp-sys out_dir, found $($nativeOut.Count)" }

$buildDir = Join-Path $repoRoot "target/$($settings.Target)/release"
$stageRoot = Join-Path $repoRoot "dist"
$stage = Join-Path $stageRoot $Profile
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item (Join-Path $buildDir $exe) (Join-Path $stage $exe)
Copy-Item LICENSE (Join-Path $stage "LICENSE")
Copy-Item README.md (Join-Path $stage "README.md")

function Copy-NativeFile([System.IO.FileInfo]$Source) {
    if ($Source.Name -match '\.dll$|\.so($|\.)|\.dylib($|\.)') {
        Copy-Item -Force $Source.FullName (Join-Path $stage $Source.Name)
    }
}

if ($settings.Features.Contains("dynamic-backends")) {
    Get-ChildItem -File (Join-Path $nativeOut[0] "lib") | ForEach-Object { Copy-NativeFile $_ }
    Get-ChildItem -File (Join-Path $nativeOut[0] "backends") | ForEach-Object {
        if ($_.Name -match '^(lib)?ggml-cpu' -or $_.Name -match "^(lib)?ggml-$($settings.Backend)\.") {
            Copy-NativeFile $_
        }
    }
}

$helperPath = Join-Path $buildDir $helper
$helperRunDir = $null
if ($settings.Features.Contains("dynamic-backends")) {
    $helperRunDir = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $helperRunDir | Out-Null
    Copy-Item $helperPath (Join-Path $helperRunDir $helper)
    Get-ChildItem -File $stage | Where-Object { $_.Name -match '\.dll$|\.so($|\.)|\.dylib($|\.)' } | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $helperRunDir $_.Name)
    }
    $helperPath = Join-Path $helperRunDir $helper
}
& $helperPath create --root $stage --target $settings.Target --tier $settings.Tier --backend $settings.Backend
if ($LASTEXITCODE -ne 0) { throw "package manifest creation failed" }
& $helperPath verify --root $stage
if ($LASTEXITCODE -ne 0) { throw "package manifest verification failed" }

if ($settings.Target.Contains("linux") -and $settings.Features.Contains("dynamic-backends")) {
    $dynamic = & readelf -d (Join-Path $stage $exe)
    if (($dynamic -join "`n") -notmatch '(RPATH|RUNPATH).*\$ORIGIN') {
        throw "Linux dynamic executable lacks an `$ORIGIN runpath"
    }
}

if ($Smoke) {
    & (Join-Path $stage $exe) --version
    if ($LASTEXITCODE -ne 0) { throw "package version smoke failed" }
    $smokeCache = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $smokeCache | Out-Null
    $oldCache = $env:JULIE_EMBEDDING_CACHE_DIR
    try {
        $env:JULIE_EMBEDDING_CACHE_DIR = $smokeCache
        $requests = @(
            '{"schema":"julie.embedding.sidecar","version":1,"request_id":"health","method":"health","params":{}}',
            '{"schema":"julie.embedding.sidecar","version":1,"request_id":"stop","method":"shutdown","params":{}}'
        )
        $smokeOutput = $requests | & (Join-Path $stage $exe) serve
        if (($smokeOutput -join "`n") -notmatch '"ready":false' -or ($smokeOutput -join "`n") -notmatch '"stopping":true') {
            throw "package protocol smoke failed"
        }
    }
    finally {
        $env:JULIE_EMBEDDING_CACHE_DIR = $oldCache
        Remove-Item -Recurse -Force $smokeCache
    }
}

$archiveBase = "julie-semantic-sidecar-$version-$($settings.Target)-$($settings.Backend)-$($settings.Tier)"
$archive = Join-Path $stageRoot "$archiveBase.$archiveKind"
Remove-Item -Force -ErrorAction SilentlyContinue $archive, "$archive.sha256"
$archiveProgram = @'
import gzip, io, pathlib, stat, sys, tarfile, zipfile
root, output, kind = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
files = sorted(path for path in root.iterdir() if path.is_file())
if kind == "zip":
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.name, (1980, 1, 1, 0, 0, 0))
            mode = 0o755 if path.name.endswith(".exe") else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
else:
    with output.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for path in files:
                data = path.read_bytes()
                info = tarfile.TarInfo(path.name)
                info.size = len(data); info.mtime = 0; info.uid = 0; info.gid = 0
                info.uname = ""; info.gname = ""; info.mode = 0o755 if path.name == "julie-semantic-sidecar" else 0o644
                archive.addfile(info, io.BytesIO(data))
'@
$archiveProgram | python - $stage $archive $archiveKind
if ($LASTEXITCODE -ne 0) { throw "archive creation failed" }
$hash = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToLowerInvariant()
"$hash  $([System.IO.Path]::GetFileName($archive))" | Set-Content -NoNewline "$archive.sha256"
Write-Host "package: manifest $(Join-Path $stage 'package-manifest.json')"
Write-Host "package: archive  $archive"
Write-Host "package: sha256   $hash"
if ($null -ne $helperRunDir) { Remove-Item -Recurse -Force $helperRunDir }
