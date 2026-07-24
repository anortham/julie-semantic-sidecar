function Add-CommandLineFlag {
    param(
        [AllowEmptyString()]
        [string]$Value,
        [Parameter(Mandatory = $true)]
        [string]$Flag
    )

    $tokens = @($Value -split '\s+' | Where-Object { $_ })
    if ($tokens.Where({ $_.Equals($Flag, [StringComparison]::OrdinalIgnoreCase) }).Count -gt 0) {
        return $Value
    }
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $Flag
    }
    return "$Value $Flag"
}

function Enable-ReproducibleWindowsBuild {
    $brepro = "/Brepro"
    $compilerOptions = [Environment]::GetEnvironmentVariable("CL")
    [Environment]::SetEnvironmentVariable(
        "CL",
        (Add-CommandLineFlag -Value $compilerOptions -Flag $brepro)
    )

    $linkOptions = [Environment]::GetEnvironmentVariable("LINK")
    [Environment]::SetEnvironmentVariable(
        "LINK",
        (Add-CommandLineFlag -Value $linkOptions -Flag $brepro)
    )

    $breproRustFlag = "link-arg=/Brepro"
    $effectiveRustFlags = "$($env:RUSTFLAGS) $($env:CARGO_ENCODED_RUSTFLAGS)"
    if ($effectiveRustFlags.IndexOf($breproRustFlag, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
        if ([string]::IsNullOrWhiteSpace($env:CARGO_ENCODED_RUSTFLAGS)) {
            $env:RUSTFLAGS = "$($env:RUSTFLAGS) -C $breproRustFlag".Trim()
        }
        else {
            $separator = [char]0x1f
            $env:CARGO_ENCODED_RUSTFLAGS = "$($env:CARGO_ENCODED_RUSTFLAGS)$($separator)-C$($separator)$breproRustFlag"
        }
    }

    foreach ($name in @(
        "CMAKE_C_FLAGS",
        "CMAKE_CXX_FLAGS",
        "CMAKE_EXE_LINKER_FLAGS",
        "CMAKE_SHARED_LINKER_FLAGS",
        "CMAKE_MODULE_LINKER_FLAGS"
    )) {
        $current = [Environment]::GetEnvironmentVariable($name)
        [Environment]::SetEnvironmentVariable(
            $name,
            (Add-CommandLineFlag -Value $current -Flag $brepro)
        )
    }
}
