use std::env;

fn main() {
    for name in [
        "TARGET",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_CFG_TARGET_OS",
        "CARGO_CFG_TARGET_FEATURE",
        "CARGO_FEATURE_METAL",
        "CARGO_FEATURE_VULKAN",
        "CARGO_FEATURE_CUDA",
        "CARGO_FEATURE_ROCM",
        "CARGO_FEATURE_DYNAMIC_BACKENDS",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }

    let target = env::var("TARGET").unwrap_or_else(|_| "unknown-target".to_string());
    let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    let rustflags = rustflags
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let package_features = ["metal", "vulkan", "cuda", "rocm", "dynamic-backends"]
        .into_iter()
        .filter(|feature| {
            let env_name = format!("CARGO_FEATURE_{}", feature.replace('-', "_").to_uppercase());
            env::var_os(env_name).is_some()
        })
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "cargo:rustc-env=JULIE_NATIVE_BUILD_IDENTITY=target={target};target_features={target_features};package_features={package_features};rustflags={rustflags}"
    );

    if env::var_os("CARGO_FEATURE_DYNAMIC_BACKENDS").is_some()
        && env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux")
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    }
}
