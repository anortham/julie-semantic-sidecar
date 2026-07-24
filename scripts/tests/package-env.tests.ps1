$ErrorActionPreference = "Stop"
. (Join-Path (Split-Path -Parent $PSScriptRoot) "package-env.ps1")

function Assert-Equal {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Expected,
        [AllowEmptyString()]
        [string]$Actual,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    if ($Expected -cne $Actual) {
        throw "$Name expected '$Expected', got '$Actual'"
    }
}

function Clear-LinkingEnvironment {
    foreach ($name in @(
        "LINK",
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "CMAKE_EXE_LINKER_FLAGS",
        "CMAKE_SHARED_LINKER_FLAGS",
        "CMAKE_MODULE_LINKER_FLAGS"
    )) {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
}

Clear-LinkingEnvironment
Enable-ReproducibleWindowsLinking
Assert-Equal "/Brepro" $env:LINK "empty LINK"
Assert-Equal "-C link-arg=/Brepro" $env:RUSTFLAGS "empty RUSTFLAGS"
Assert-Equal "/Brepro" $env:CMAKE_EXE_LINKER_FLAGS "empty CMAKE_EXE_LINKER_FLAGS"
Assert-Equal "/Brepro" $env:CMAKE_SHARED_LINKER_FLAGS "empty CMAKE_SHARED_LINKER_FLAGS"
Assert-Equal "/Brepro" $env:CMAKE_MODULE_LINKER_FLAGS "empty CMAKE_MODULE_LINKER_FLAGS"

Clear-LinkingEnvironment
$env:LINK = "/DEBUG"
$env:RUSTFLAGS = "-C opt-level=3"
$env:CMAKE_EXE_LINKER_FLAGS = "/DEBUG"
$env:CMAKE_SHARED_LINKER_FLAGS = "/DEBUG"
$env:CMAKE_MODULE_LINKER_FLAGS = "/DEBUG"
Enable-ReproducibleWindowsLinking
Assert-Equal "/DEBUG /Brepro" $env:LINK "seeded LINK"
Assert-Equal "-C opt-level=3 -C link-arg=/Brepro" $env:RUSTFLAGS "seeded RUSTFLAGS"
Assert-Equal "/DEBUG /Brepro" $env:CMAKE_EXE_LINKER_FLAGS "seeded CMAKE_EXE_LINKER_FLAGS"
Assert-Equal "/DEBUG /Brepro" $env:CMAKE_SHARED_LINKER_FLAGS "seeded CMAKE_SHARED_LINKER_FLAGS"
Assert-Equal "/DEBUG /Brepro" $env:CMAKE_MODULE_LINKER_FLAGS "seeded CMAKE_MODULE_LINKER_FLAGS"

Clear-LinkingEnvironment
$separator = [char]0x1f
$env:LINK = "/BREPRO"
$env:CARGO_ENCODED_RUSTFLAGS = "-C$($separator)opt-level=3"
$env:CMAKE_EXE_LINKER_FLAGS = "/brepro"
$env:CMAKE_SHARED_LINKER_FLAGS = "/BREPRO"
$env:CMAKE_MODULE_LINKER_FLAGS = "/Brepro"
Enable-ReproducibleWindowsLinking
Assert-Equal "/BREPRO" $env:LINK "case-insensitive LINK"
Assert-Equal "-C$($separator)opt-level=3$($separator)-C$($separator)link-arg=/Brepro" $env:CARGO_ENCODED_RUSTFLAGS "encoded RUSTFLAGS"
Assert-Equal "/brepro" $env:CMAKE_EXE_LINKER_FLAGS "case-insensitive CMAKE_EXE_LINKER_FLAGS"
Assert-Equal "/BREPRO" $env:CMAKE_SHARED_LINKER_FLAGS "case-insensitive CMAKE_SHARED_LINKER_FLAGS"
Assert-Equal "/Brepro" $env:CMAKE_MODULE_LINKER_FLAGS "case-insensitive CMAKE_MODULE_LINKER_FLAGS"

Write-Host "package-env.tests.ps1: passed"
