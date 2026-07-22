use julie_semantic_sidecar::backend_select::NATIVE_BUILD_IDENTITY;
use julie_semantic_sidecar::package_manifest::{
    self, AdvertisedBackend, PackageProfile, PackageTier,
};
use std::path::PathBuf;

const USAGE: &str = "usage: julie-package-manifest create --root <dir> --target <triple> --tier <portable|vendor> --backend <metal|vulkan|cuda>\n       julie-package-manifest verify --root <dir>";

fn main() {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => {}
        Err((code, message)) => {
            eprintln!("julie-package-manifest: {message}");
            if code == 2 {
                eprintln!("{USAGE}");
            }
            std::process::exit(code);
        }
    }
}

fn run(arguments: Vec<String>) -> Result<(), (i32, String)> {
    let Some(command) = arguments.first().map(String::as_str) else {
        return Err((2, "missing command".to_string()));
    };
    let options = parse_options(&arguments[1..])?;
    let root = required(&options, "--root").map(PathBuf::from)?;
    match command {
        "create" => {
            let target = required(&options, "--target")?;
            let tier = match required(&options, "--tier")?.as_str() {
                "portable" => PackageTier::Portable,
                "vendor" => PackageTier::Vendor,
                value => return Err((2, format!("unknown tier {value}"))),
            };
            let backend = match required(&options, "--backend")?.as_str() {
                "metal" => AdvertisedBackend::Metal,
                "vulkan" => AdvertisedBackend::Vulkan,
                "cuda" => AdvertisedBackend::Cuda,
                value => return Err((2, format!("unknown backend {value}"))),
            };
            package_manifest::write(
                &root,
                &PackageProfile {
                    rust_target: target,
                    tier,
                    advertised_backend: backend,
                    native_build_identity: NATIVE_BUILD_IDENTITY.to_string(),
                },
            )
            .map_err(|error| (1, error.to_string()))?;
        }
        "verify" => {
            package_manifest::verify(&root).map_err(|error| (1, error.to_string()))?;
        }
        value => return Err((2, format!("unknown command {value}"))),
    }
    Ok(())
}

fn parse_options(
    arguments: &[String],
) -> Result<std::collections::BTreeMap<String, String>, (i32, String)> {
    let mut options = std::collections::BTreeMap::new();
    let mut index = 0;
    while index < arguments.len() {
        let name = &arguments[index];
        if !name.starts_with("--") || index + 1 == arguments.len() {
            return Err((2, format!("invalid option {name}")));
        }
        if options
            .insert(name.clone(), arguments[index + 1].clone())
            .is_some()
        {
            return Err((2, format!("duplicate option {name}")));
        }
        index += 2;
    }
    Ok(options)
}

fn required(
    options: &std::collections::BTreeMap<String, String>,
    name: &str,
) -> Result<String, (i32, String)> {
    options
        .get(name)
        .cloned()
        .ok_or_else(|| (2, format!("missing {name}")))
}
