param(
    [Parameter(Mandatory = $true)][string]$Archive,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$Sha256,
    [Parameter(Mandatory = $true)][ValidateSet('metal', 'vulkan', 'cuda')][string]$Backend,
    [Parameter(Mandatory = $true)][string]$Lane,
    [string]$CacheDir,
    [string]$FixturesDir = $(if ($env:FIXTURES_DIR) { $env:FIXTURES_DIR } else { '/Users/murphy/source/miller/eval/sidecar-conformance' }),
    [string]$EvidenceDir,
    [switch]$ArtifactValidation
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$pythonCommand = Get-Command python3 -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
}
if (-not $pythonCommand) {
    throw 'python3 or python is required'
}
$python = $pythonCommand.Source

if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) {
    throw "archive does not exist: $Archive"
}
$expectedSha256 = $Sha256.ToLowerInvariant()
$actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant()
if ($actualSha256 -ne $expectedSha256) {
    throw "archive checksum $actualSha256 does not match $expectedSha256"
}

$unpackDir = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString('N'))
$emptyCache = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $unpackDir, $emptyCache | Out-Null
$oldCache = $env:JULIE_EMBEDDING_CACHE_DIR
$oldBackend = $env:JULIE_SIDECAR_FORCE_BACKEND

$protocolProgram = @'
import json, os, subprocess, sys
binary, expectation, advertised = sys.argv[1:]
requests = [{"schema":"julie.embedding.sidecar","version":1,"request_id":"health","method":"health","params":{}}]
if expectation != "absent":
    requests += [
        {"schema":"julie.embedding.sidecar","version":1,"request_id":"query","method":"embed_query","params":{"text":"archive query smoke"}},
        {"schema":"julie.embedding.sidecar","version":1,"request_id":"batch","method":"embed_batch","params":{"texts":["archive batch one","archive batch two"]}},
    ]
requests.append({"schema":"julie.embedding.sidecar","version":1,"request_id":"shutdown","method":"shutdown","params":{}})
process = subprocess.Popen([binary, "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=os.environ.copy())
payload = "\n".join(json.dumps(request, separators=(",", ":")) for request in requests) + "\n"
try:
    stdout, stderr = process.communicate(payload, timeout=900)
except subprocess.TimeoutExpired:
    process.kill()
    stdout, stderr = process.communicate()
    sys.stderr.write(stderr)
    raise SystemExit("sidecar protocol smoke timed out after 900s")
sys.stderr.write(stderr)
if process.returncode != 0:
    raise SystemExit(f"sidecar exited {process.returncode}")
lines = [line for line in stdout.splitlines() if line]
if len(lines) != len(requests):
    raise SystemExit(f"stdout purity/count failure: expected {len(requests)} protocol lines, got {len(lines)}: {stdout!r}")
responses = [json.loads(line) for line in lines]
health = responses[0].get("result", {})
if expectation == "absent":
    if health.get("ready") is not False or health.get("degraded_reason") != "model_not_prepared":
        raise SystemExit(f"absent-model health mismatch: {responses[0]}")
else:
    if health.get("ready") is not True:
        raise SystemExit(f"prepared health mismatch: {responses[0]}")
    if expectation == "accelerated" and (health.get("resolved_backend") != advertised or health.get("accelerated") is not True):
        raise SystemExit(f"accelerator health mismatch: {responses[0]}")
    if expectation == "accelerated" and any(marker in str(health.get("device", "")).lower() for marker in ("llvmpipe", "lavapipe", "swiftshader", "software rasterizer", "microsoft basic render")):
        raise SystemExit(f"software device was selected: {responses[0]}")
    if expectation == "cpu" and (health.get("resolved_backend") != "cpu" or health.get("accelerated") is not False):
        raise SystemExit(f"forced CPU health mismatch: {responses[0]}")
    if expectation == "fallback" and (health.get("resolved_backend") != "cpu" or health.get("accelerated") is not False or not health.get("degraded_reason")):
        raise SystemExit(f"fallback health mismatch: {responses[0]}")
    dims = health.get("dims")
    if len(responses[1].get("result", {}).get("vector", [])) != dims:
        raise SystemExit(f"query dimensions mismatch: {responses[1]}")
    vectors = responses[2].get("result", {}).get("vectors", [])
    if len(vectors) != 2 or any(len(vector) != dims for vector in vectors):
        raise SystemExit(f"batch shape mismatch: {responses[2]}")
if responses[-1].get("result") != {"stopping": True}:
    raise SystemExit(f"shutdown mismatch: {responses[-1]}")
sys.stdout.write(stdout)
'@

$extractProgram = @'
import pathlib, sys, tarfile, zipfile
archive = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
def safe_name(name):
    path = pathlib.PurePosixPath(name)
    if "\\" in name or path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise SystemExit(f"archive member is not flat and safe: {name!r}")
    return path.name
if archive.name.endswith(".zip"):
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            name = safe_name(member.filename)
            if member.is_dir():
                raise SystemExit(f"archive contains a directory: {member.filename!r}")
            destination = root / name
            destination.write_bytes(source.read(member))
            mode = (member.external_attr >> 16) & 0o777
            destination.chmod(mode or 0o644)
else:
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            name = safe_name(member.name)
            if not member.isfile():
                raise SystemExit(f"archive member is not a regular file: {member.name!r}")
            stream = source.extractfile(member)
            if stream is None:
                raise SystemExit(f"archive member cannot be read: {member.name!r}")
            destination = root / name
            destination.write_bytes(stream.read())
            destination.chmod(member.mode & 0o777)
'@

function Invoke-ProtocolSmoke([string]$Label, [string]$SmokeCache, [string]$ForcedBackend, [string]$Expectation) {
    $env:JULIE_EMBEDDING_CACHE_DIR = $SmokeCache
    if ($ForcedBackend) {
        $env:JULIE_SIDECAR_FORCE_BACKEND = $ForcedBackend
    }
    else {
        Remove-Item Env:JULIE_SIDECAR_FORCE_BACKEND -ErrorAction SilentlyContinue
    }
    $stdoutPath = Join-Path $EvidenceDir "raw-logs/$Label.stdout.jsonl"
    $stderrPath = Join-Path $EvidenceDir "raw-logs/$Label.stderr.log"
    & $script:python -c $protocolProgram $script:binary $Expectation $Backend 1> $stdoutPath 2> $stderrPath
    if ($LASTEXITCODE -ne 0) { throw "protocol smoke failed: $Label" }
}

function Invoke-Conformance([string]$RequestedBackend) {
    $env:FIXTURES_DIR = $FixturesDir
    $env:JULIE_CONFORMANCE_BIN = $script:binary
    $env:JULIE_EMBEDDING_CACHE_DIR = $CacheDir
    $env:JULIE_CONFORMANCE_UNAVAILABLE_BACKEND = $script:fallbackBackend
    $env:JULIE_SIDECAR_FORCE_BACKEND = $RequestedBackend
    $log = Join-Path $EvidenceDir "raw-logs/conformance-$RequestedBackend.log"
    $arguments = @('test', '--release', '--test', 'conformance', '--', '--ignored', '--test-threads=1', '--nocapture')
    @(
        "conformance: binary   $script:binary"
        "conformance: fixtures $FixturesDir"
        "conformance: backend  $RequestedBackend"
    ) | Set-Content -LiteralPath $log
    & cargo @arguments *>> $log
    if ($LASTEXITCODE -ne 0) { throw "conformance failed: $RequestedBackend" }
}

function Invoke-Prepare([string]$Model) {
    $env:JULIE_EMBEDDING_CACHE_DIR = $CacheDir
    $prepareLog = Join-Path $EvidenceDir "raw-logs/prepare-$Model.log"
    Set-Content -LiteralPath $prepareLog -Value '' -NoNewline
    foreach ($attempt in 1..3) {
        & $script:binary prepare --model $Model *>> $prepareLog
        if ($LASTEXITCODE -eq 0) { return }
        "hardware-smoke: prepare attempt $attempt failed for $Model" | Add-Content -LiteralPath $prepareLog
        if ($attempt -ne 3) { Start-Sleep -Seconds ($attempt * 30) }
    }
    throw "prepare failed after 3 attempts: $Model"
}

try {
    & $script:python -c $extractProgram $Archive $unpackDir
    if ($LASTEXITCODE -ne 0) { throw 'archive extraction failed' }

    $binary = Join-Path $unpackDir 'julie-semantic-sidecar.exe'
    if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) {
        $binary = Join-Path $unpackDir 'julie-semantic-sidecar'
    }
    if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) {
        throw 'unpacked archive has no sidecar executable'
    }
    $manifestPath = Join-Path $unpackDir 'package-manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw 'unpacked archive has no package-manifest.json'
    }
    $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    if ($manifest.advertised_backend -ne $Backend) {
        throw "manifest backend $($manifest.advertised_backend) does not match $Backend"
    }
    $declared = @('package-manifest.json')
    foreach ($item in $manifest.files) {
        if ([System.IO.Path]::GetFileName($item.path) -ne $item.path) { throw "manifest path is not flat: $($item.path)" }
        $path = Join-Path $unpackDir $item.path
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "manifest payload is missing: $($item.path)" }
        if ((Get-Item -LiteralPath $path).Length -ne $item.size) { throw "manifest size mismatch: $($item.path)" }
        $digest = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
        if ($digest -ne $item.sha256) { throw "manifest checksum mismatch: $($item.path)" }
        if ($item.role -eq 'model_weight') { throw "model weight declared in archive: $($item.path)" }
        $declared += $item.path
    }
    $actual = @(Get-ChildItem -LiteralPath $unpackDir -File | ForEach-Object Name | Sort-Object)
    $declared = @($declared | Sort-Object)
    if (($actual | ConvertTo-Json -Compress) -ne ($declared | ConvertTo-Json -Compress)) {
        throw 'archive inventory differs from package manifest'
    }

    if (-not $EvidenceDir) {
        $EvidenceDir = Join-Path $repoRoot "hardware-evidence/$Lane-$($actualSha256.Substring(0, 12))"
    }
    New-Item -ItemType Directory -Force -Path (Join-Path $EvidenceDir 'raw-logs') | Out-Null
    Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $EvidenceDir 'package-manifest.json')
    "$actualSha256  $([System.IO.Path]::GetFileName($Archive))" | Set-Content (Join-Path $EvidenceDir 'archive.sha256')
    @(
        "hardware_lane=$Lane"
        "advertised_backend=$Backend"
        "archive=$([System.IO.Path]::GetFileName($Archive))"
        "archive_sha256=$actualSha256"
        "host=$([System.Environment]::OSVersion.VersionString)"
        "recorded_utc=$([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'))"
    ) | Set-Content (Join-Path $EvidenceDir 'identity.txt')
    $versionLog = Join-Path $EvidenceDir 'raw-logs/version.log'
    & $binary --version *> $versionLog
    if ($LASTEXITCODE -ne 0) { throw 'version smoke failed' }

    Invoke-ProtocolSmoke 'absent-model' $emptyCache 'cpu' 'absent'
    if ($ArtifactValidation) {
        @('artifact_validation=passed', 'support_evidence=false') | Add-Content (Join-Path $EvidenceDir 'identity.txt')
        Write-Host 'hardware-smoke: artifact validation passed; this is not support evidence'
        Write-Host "hardware-smoke: evidence $EvidenceDir"
        return
    }
    if (-not $CacheDir) { throw '-CacheDir is required for real-device proof' }
    New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
    if (-not (Test-Path -LiteralPath (Join-Path $FixturesDir 'corpus.jsonl') -PathType Leaf)) {
        throw "fixtures do not hold corpus.jsonl: $FixturesDir"
    }

    $deviceLog = Join-Path $EvidenceDir 'raw-logs/device.txt'
    $runtimeLog = Join-Path $EvidenceDir 'raw-logs/runtime.txt'
    switch ($Backend) {
        'metal' {
            if (-not $IsMacOS) { throw 'Metal requires macOS' }
            & system_profiler SPDisplaysDataType *> $deviceLog
            & sw_vers *> $runtimeLog
            $fallbackBackend = 'vulkan'
        }
        'vulkan' {
            if (-not (Get-Command vulkaninfo -ErrorAction SilentlyContinue)) { throw 'vulkaninfo is required' }
            & vulkaninfo --summary *> $deviceLog
            & vulkaninfo --summary *> $runtimeLog
            $fallbackBackend = 'metal'
        }
        'cuda' {
            if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw 'nvidia-smi is required' }
            & nvidia-smi -q *> $deviceLog
            & nvidia-smi *> $runtimeLog
            $fallbackBackend = 'metal'
        }
    }
    $deviceReport = Get-Content -Raw $deviceLog
    if ($deviceReport -match '(?i)llvmpipe|lavapipe|swiftshader|software rasterizer|microsoft basic render' -and
        ($Backend -ne 'vulkan' -or $deviceReport -notmatch 'deviceType.*PHYSICAL_DEVICE_TYPE_(INTEGRATED|DISCRETE)_GPU')) {
        throw 'software renderer is not real-device evidence'
    }

    foreach ($model in @('bge-small-en-v1.5-f32', 'qwen3-0.6b-f16')) {
        Invoke-Prepare $model
    }

    $selectionCache = Join-Path $CacheDir 'backend-selection.json'
    Remove-Item -Force -ErrorAction SilentlyContinue $selectionCache
    Invoke-ProtocolSmoke 'selection-rebuild' $CacheDir '' 'accelerated'
    $selectionLog = Get-Content -Raw (Join-Path $EvidenceDir 'raw-logs/selection-rebuild.stderr.log')
    if ($selectionLog -match '(?i)using device .*\b(llvmpipe|lavapipe|swiftshader|software rasterizer|microsoft basic render)\b') {
        throw 'software device was selected'
    }
    if (-not (Test-Path -LiteralPath $selectionCache -PathType Leaf)) { throw 'selection cache was not rebuilt' }
    $selectionBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $selectionCache).Hash
    Invoke-ProtocolSmoke 'selection-reuse' $CacheDir '' 'accelerated'
    $selectionAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $selectionCache).Hash
    if ($selectionBefore -ne $selectionAfter) { throw 'cached selection changed during reuse' }

    Invoke-ProtocolSmoke 'forced-cpu' $CacheDir 'cpu' 'cpu'
    Invoke-ProtocolSmoke 'fallback' $CacheDir $fallbackBackend 'fallback'
    Invoke-Conformance 'cpu'
    Invoke-Conformance $Backend

    foreach ($measuredBackend in @('cpu', $Backend)) {
        $env:JULIE_EMBEDDING_CACHE_DIR = $CacheDir
        $env:JULIE_SIDECAR_FORCE_BACKEND = $measuredBackend
        $batchOneLog = Join-Path $EvidenceDir "raw-logs/bench-$measuredBackend-batch-1.json"
        & $script:python scripts/bench-throughput.py --binary $binary --batch 1 --rounds 4 --floor 0 --expect-backend $measuredBackend --json > $batchOneLog
        if ($LASTEXITCODE -ne 0) { throw "batch-1 measurement failed: $measuredBackend" }
        $batchSixteenLog = Join-Path $EvidenceDir "raw-logs/bench-$measuredBackend-batch-16.json"
        & $script:python scripts/bench-throughput.py --binary $binary --batch 16 --rounds 4 --floor 0 --expect-backend $measuredBackend --json > $batchSixteenLog
        if ($LASTEXITCODE -ne 0) { throw "batch-16 measurement failed: $measuredBackend" }
    }

    Copy-Item -LiteralPath $selectionCache -Destination (Join-Path $EvidenceDir 'backend-selection.json')
    @('artifact_validation=passed', 'support_evidence=real-device-pending-review') | Add-Content (Join-Path $EvidenceDir 'identity.txt')
    Write-Host 'hardware-smoke: real-device evidence captured for manual review'
    Write-Host "hardware-smoke: evidence $EvidenceDir"
}
finally {
    $env:JULIE_EMBEDDING_CACHE_DIR = $oldCache
    $env:JULIE_SIDECAR_FORCE_BACKEND = $oldBackend
    Remove-Item Env:JULIE_CONFORMANCE_BIN -ErrorAction SilentlyContinue
    Remove-Item Env:JULIE_CONFORMANCE_UNAVAILABLE_BACKEND -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $unpackDir, $emptyCache
}
