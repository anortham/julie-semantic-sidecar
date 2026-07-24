use julie_semantic_sidecar::backend_select;
use julie_semantic_sidecar::package_manifest::{
    self, AdvertisedBackend, PackageFileRole, PackageManifest, PackageProfile, PackageTier,
    MANIFEST_FILE,
};
use std::path::Path;
use std::process::Command;

const HELPER: &str = env!("CARGO_BIN_EXE_julie-package-manifest");
const PORTABLE_PROFILES: [&str; 4] = [
    "apple-arm64-metal-portable",
    "apple-x64-metal-portable",
    "linux-x64-vulkan-portable",
    "windows-x64-vulkan-portable",
];
const VENDOR_PROFILES: [&str; 2] = ["linux-x64-cuda-vendor", "windows-x64-cuda-vendor"];

fn write(root: &Path, name: &str, bytes: &[u8]) {
    std::fs::write(root.join(name), bytes).expect("write payload");
}

fn dynamic_stage(root: &Path) {
    write(root, "julie-semantic-sidecar", b"executable");
    write(root, "libggml.so", b"ggml");
    write(root, "libllama.so", b"llama");
    write(root, "libggml-cpu-x86_64.so", b"cpu");
    write(root, "libggml-vulkan.so", b"vulkan");
    write(root, "LICENSE", b"license");
    write(root, "README.md", b"readme");
}

fn vulkan_profile() -> PackageProfile {
    PackageProfile {
        rust_target: "x86_64-unknown-linux-gnu".to_string(),
        tier: PackageTier::Portable,
        advertised_backend: AdvertisedBackend::Vulkan,
        native_build_identity: "native-build-a".to_string(),
    }
}

fn rewrite(root: &Path, mutate: impl FnOnce(&mut PackageManifest)) {
    let path = root.join(MANIFEST_FILE);
    let mut manifest: PackageManifest =
        serde_json::from_slice(&std::fs::read(&path).expect("read manifest")).expect("manifest");
    mutate(&mut manifest);
    std::fs::write(path, serde_json::to_vec_pretty(&manifest).expect("json")).expect("rewrite");
}

#[test]
fn create_is_deterministic_and_records_sorted_verified_payloads_and_model_policy() {
    let first = tempfile::tempdir().expect("tempdir");
    let second = tempfile::tempdir().expect("tempdir");
    for root in [first.path(), second.path()] {
        dynamic_stage(root);
        package_manifest::write(root, &vulkan_profile()).expect("write manifest");
    }

    assert_eq!(
        std::fs::read(first.path().join(MANIFEST_FILE)).expect("first"),
        std::fs::read(second.path().join(MANIFEST_FILE)).expect("second")
    );
    let manifest = package_manifest::verify(first.path()).expect("verify");
    let paths = manifest
        .files
        .iter()
        .map(|file| file.path.as_str())
        .collect::<Vec<_>>();
    assert!(paths.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(!paths.contains(&MANIFEST_FILE));
    assert_eq!(manifest.schema_version, 1);
    assert_eq!(manifest.rust_target, vulkan_profile().rust_target);
    assert_eq!(manifest.advertised_backend, AdvertisedBackend::Vulkan);
    assert_eq!(
        manifest.model_policy.default_id,
        julie_semantic_sidecar::DEFAULT_MODEL_ID
    );
    assert_eq!(
        manifest.model_policy.ids,
        vec!["bge-small-en-v1.5-f32", "qwen3-0.6b-f16"]
    );
    assert!(manifest
        .files
        .iter()
        .all(|file| file.sha256.len() == 64 && file.size > 0));
}

#[test]
fn packaged_backend_identity_includes_core_runtime_replacements() {
    let dir = tempfile::tempdir().expect("tempdir");
    dynamic_stage(dir.path());
    package_manifest::write(dir.path(), &vulkan_profile()).expect("write manifest");
    let executable = dir.path().join("julie-semantic-sidecar");
    let before = backend_select::packaged_backend_identity(&executable);

    write(dir.path(), "libllama.so", b"replacement-llama-runtime");

    assert_ne!(
        backend_select::packaged_backend_identity(&executable),
        before
    );
}

#[test]
fn verify_rejects_every_undeclared_file_except_the_manifest_itself() {
    let dir = tempfile::tempdir().expect("tempdir");
    dynamic_stage(dir.path());
    package_manifest::write(dir.path(), &vulkan_profile()).expect("write manifest");
    write(dir.path(), "surprise.txt", b"undeclared");

    let error = package_manifest::verify(dir.path()).expect_err("undeclared file");
    assert!(error.to_string().contains("undeclared file surprise.txt"));
}

#[test]
fn verify_rejects_checksum_and_size_mismatches() {
    for replacement in [b"changed".as_slice(), b"longer changed payload".as_slice()] {
        let dir = tempfile::tempdir().expect("tempdir");
        dynamic_stage(dir.path());
        package_manifest::write(dir.path(), &vulkan_profile()).expect("write manifest");
        write(dir.path(), "README.md", replacement);

        let error = package_manifest::verify(dir.path()).expect_err("payload mismatch");
        assert!(
            error.to_string().contains("README.md")
                && (error.to_string().contains("checksum") || error.to_string().contains("size"))
        );
    }
}

#[test]
fn verify_rejects_absolute_traversal_and_development_paths() {
    for bad in [
        "/tmp/libggml.so",
        "../libggml.so",
        "target/release/libggml.so",
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        dynamic_stage(dir.path());
        package_manifest::write(dir.path(), &vulkan_profile()).expect("write manifest");
        rewrite(dir.path(), |manifest| {
            manifest.files[0].path = bad.to_string()
        });

        let error = package_manifest::verify(dir.path()).expect_err("unsafe path");
        assert!(
            error.to_string().contains("invalid package path"),
            "{bad}: {error}"
        );
    }
}

#[test]
fn create_rejects_model_weights_and_misplaced_native_libraries() {
    for bad in ["model.gguf", "lib/libggml.so"] {
        let dir = tempfile::tempdir().expect("tempdir");
        dynamic_stage(dir.path());
        let path = dir.path().join(bad);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("directory");
        }
        std::fs::write(path, b"bad").expect("bad payload");

        let error = package_manifest::write(dir.path(), &vulkan_profile()).expect_err("bad file");
        assert!(
            error.to_string().contains("model weight")
                || error.to_string().contains("flat package root")
        );
    }
}

#[test]
fn dynamic_profiles_require_executable_core_cpu_and_advertised_accelerator_roles() {
    for missing in [
        &["julie-semantic-sidecar"][..],
        &["libggml.so", "libllama.so"][..],
        &["libggml-cpu-x86_64.so"][..],
        &["libggml-vulkan.so"][..],
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        dynamic_stage(dir.path());
        for name in missing {
            std::fs::remove_file(dir.path().join(name)).expect("remove");
        }

        let error =
            package_manifest::write(dir.path(), &vulkan_profile()).expect_err("missing role");
        assert!(error.to_string().contains("missing required"), "{error}");
    }
}

#[test]
fn backend_file_disagreement_and_extra_accelerator_modules_are_rejected() {
    for extra in ["libggml-cuda.so", "libggml-hip.so", "libggml-sycl.so"] {
        let dir = tempfile::tempdir().expect("tempdir");
        dynamic_stage(dir.path());
        write(dir.path(), extra, b"extra accelerator");

        let error =
            package_manifest::write(dir.path(), &vulkan_profile()).expect_err("extra backend");
        assert!(error.to_string().contains("accelerator"), "{error}");
    }
}

#[test]
fn apple_metal_is_built_in_and_rejects_a_fake_plugin() {
    for rust_target in ["aarch64-apple-darwin", "x86_64-apple-darwin"] {
        let dir = tempfile::tempdir().expect("tempdir");
        write(dir.path(), "julie-semantic-sidecar", b"executable");
        write(dir.path(), "LICENSE", b"license");
        write(dir.path(), "README.md", b"readme");
        let profile = PackageProfile {
            rust_target: rust_target.to_string(),
            tier: PackageTier::Portable,
            advertised_backend: AdvertisedBackend::Metal,
            native_build_identity: "native-metal".to_string(),
        };

        let manifest = package_manifest::write(dir.path(), &profile).expect("built in metal");
        assert_eq!(manifest.files.len(), 3);
        assert!(manifest
            .files
            .iter()
            .all(|file| file.role != PackageFileRole::AcceleratorBackend));
        write(dir.path(), "libggml-metal.so", b"fake plugin");
        assert!(package_manifest::write(dir.path(), &profile).is_err());
    }
}

#[test]
fn rust_helper_creates_and_verifies_with_the_shared_validator() {
    let dir = tempfile::tempdir().expect("tempdir");
    write(dir.path(), "julie-semantic-sidecar", b"executable");
    write(dir.path(), "LICENSE", b"license");
    write(dir.path(), "README.md", b"readme");

    let create = Command::new(HELPER)
        .args([
            "create",
            "--root",
            dir.path().to_str().expect("root"),
            "--target",
            "aarch64-apple-darwin",
            "--tier",
            "portable",
            "--backend",
            "metal",
        ])
        .output()
        .expect("create helper");
    assert!(
        create.status.success(),
        "{}",
        String::from_utf8_lossy(&create.stderr)
    );

    let verify = Command::new(HELPER)
        .args(["verify", "--root", dir.path().to_str().expect("root")])
        .output()
        .expect("verify helper");
    assert!(
        verify.status.success(),
        "{}",
        String::from_utf8_lossy(&verify.stderr)
    );
    assert!(dir.path().join(MANIFEST_FILE).is_file());
}

fn packaging_scripts() -> [String; 2] {
    [
        std::fs::read_to_string("scripts/package.sh").expect("bash package script"),
        std::fs::read_to_string("scripts/package.ps1").expect("PowerShell package script"),
    ]
}

fn bash_declared_profiles(script: &str) -> Vec<&str> {
    let profile_case = script
        .split_once("case \"$profile\" in")
        .expect("Bash profile case")
        .1
        .split_once("esac")
        .expect("Bash profile case terminator")
        .0;
    profile_case
        .lines()
        .filter_map(|line| {
            let (profile, _) = line.trim().split_once(')')?;
            (profile != "*").then_some(profile)
        })
        .collect()
}

fn powershell_declared_profiles(script: &str) -> Vec<&str> {
    let validate_set = script
        .split_once("[ValidateSet(")
        .expect("PowerShell profile ValidateSet")
        .1
        .split_once(")]")
        .expect("PowerShell profile ValidateSet terminator")
        .0;
    validate_set
        .lines()
        .filter_map(|line| {
            line.trim()
                .trim_end_matches(',')
                .strip_prefix('"')?
                .strip_suffix('"')
        })
        .collect()
}

#[test]
fn packaging_scripts_define_only_the_explicit_portable_and_cuda_candidate_profiles() {
    let expected = PORTABLE_PROFILES
        .into_iter()
        .chain(VENDOR_PROFILES)
        .collect::<Vec<_>>();
    let [bash, powershell] = packaging_scripts();

    assert_eq!(bash_declared_profiles(&bash), expected);
    assert_eq!(powershell_declared_profiles(&powershell), expected);
    for profile in &expected {
        assert_eq!(
            bash.matches(profile).count(),
            1,
            "Bash mapping for {profile}"
        );
        assert_eq!(
            powershell.matches(profile).count(),
            2,
            "PowerShell declaration and mapping for {profile}"
        );
    }

    let bash_drift = bash.replacen(
        "  *) echo \"package: unknown profile: $profile\"",
        "  macos-x64)\n    target=\"x86_64-apple-darwin\"; backend=\"cpu\"; tier=\"portable\"; features=\"\" ;;\n  *) echo \"package: unknown profile: $profile\"",
        1,
    );
    assert_ne!(bash_drift, bash);
    assert_ne!(bash_declared_profiles(&bash_drift), expected);

    let powershell_drift = powershell.replacen(
        "        \"apple-x64-metal-portable\",",
        "        \"apple-x64-metal-portable\",\n        \"macos-x64\",",
        1,
    );
    assert_ne!(powershell_drift, powershell);
    assert_ne!(powershell_declared_profiles(&powershell_drift), expected);
}

#[test]
fn windows_packaging_routes_deterministic_linking_through_a_native_ci_test() {
    let [_, powershell] = packaging_scripts();
    let helper = std::fs::read_to_string("scripts/package-env.ps1").expect("package environment");
    let native_test =
        std::fs::read_to_string("scripts/tests/package-env.tests.ps1").expect("native test");
    let workflow = std::fs::read_to_string(".github/workflows/release.yml").expect("workflow");

    assert!(powershell.contains("Enable-ReproducibleWindowsBuild"));
    assert!(helper.contains("[StringComparison]::OrdinalIgnoreCase"));
    assert!(helper.contains("\"CL\""));
    assert!(helper.contains("link-arg=/Brepro"));
    assert!(helper.contains("\"CMAKE_CXX_FLAGS\""));
    assert!(native_test.contains("Enable-ReproducibleWindowsBuild"));
    assert!(workflow.contains("scripts/tests/package-env.tests.ps1"));
}

#[test]
fn packaging_patches_the_pinned_native_shader_before_building() {
    let [bash, powershell] = packaging_scripts();
    let build_script = std::fs::read_to_string("build.rs").expect("build script");

    for script in [&bash, &powershell] {
        assert!(script.contains("cargo vendor"));
        assert!(script.contains("patch-native-source.py"));
        assert!(script.contains("JULIE_NATIVE_PATCH_IDENTITY"));
        assert!(script.contains("--config"));
        assert!(script.contains("package-vendor"));
    }
    assert!(bash.find("patch-native-source.py") < bash.find("cargo --config"));
    assert!(powershell.find("patch-native-source.py") < powershell.find("$messages = & cargo"));
    assert!(bash.contains("vulkan-infinity-v3:[0-9a-f]{64}"));
    assert!(powershell.contains("vulkan-infinity-v3:[0-9a-f]{64}"));
    assert!(powershell.contains("-cnotmatch"));
    assert!(bash.contains("verify-patched"));
    assert!(powershell.contains("verify-patched"));
    assert!(build_script.contains("JULIE_NATIVE_PATCH_IDENTITY"));
}

#[test]
fn patched_package_verification_rejects_every_invalid_native_patch_identity_shape() {
    for identity in [
        "cargo=release".to_string(),
        "cargo=release;native_patch=none".to_string(),
        format!(
            "cargo=release;native_patch=llama-cpp-sys-2-0.1.151:vulkan-infinity-v3:{}",
            "0".repeat(63)
        ),
        format!(
            "cargo=release;native_patch=llama-cpp-sys-2-0.1.151:vulkan-infinity-v3:{}",
            "A".repeat(64)
        ),
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        write(dir.path(), "julie-semantic-sidecar", b"executable");
        write(dir.path(), "LICENSE", b"license");
        write(dir.path(), "README.md", b"readme");

        let create = Command::new(HELPER)
            .args([
                "create",
                "--root",
                dir.path().to_str().expect("root"),
                "--target",
                "aarch64-apple-darwin",
                "--tier",
                "portable",
                "--backend",
                "metal",
            ])
            .output()
            .expect("create helper");
        assert!(create.status.success());

        rewrite(dir.path(), |manifest| {
            manifest.native_build_identity = identity;
        });
        let verify = Command::new(HELPER)
            .args([
                "verify-patched",
                "--root",
                dir.path().to_str().expect("root"),
            ])
            .output()
            .expect("verify helper");

        assert!(!verify.status.success());
        assert!(String::from_utf8_lossy(&verify.stderr).contains("native patch identity"));
    }
}

#[test]
fn failed_checksum_proofs_retain_the_candidate_for_diagnosis() {
    let workflow = std::fs::read_to_string(".github/workflows/release.yml").expect("workflow");
    let upload = workflow
        .split("- name: Upload private archive, manifest, checksum, and raw logs")
        .nth(1)
        .expect("candidate upload step");

    assert!(upload.trim_start().starts_with("if: always()"));
}

#[test]
fn public_docs_and_promotion_gate_name_every_portable_profile() {
    for path in ["README.md", "docs/rc-promotion-gate.md"] {
        let document = std::fs::read_to_string(path).expect("package documentation");
        for profile in PORTABLE_PROFILES {
            assert!(document.contains(profile), "{path} is missing {profile}");
        }
    }
}

#[test]
fn current_release_candidate_has_matching_public_notes_and_status() {
    let version = env!("CARGO_PKG_VERSION");
    assert_eq!(version, "0.1.0-rc.4");

    let tag = format!("v{version}");
    let release_notes_path = format!("docs/release-notes/{tag}.md");
    let readme = std::fs::read_to_string("README.md").expect("README");
    let release_notes =
        std::fs::read_to_string(&release_notes_path).expect("current release notes");

    assert!(
        readme.contains(&format!(
            "**Current candidate: [`{tag}`]({release_notes_path}).**"
        )),
        "README current-candidate pointer does not match {tag}"
    );
    assert!(
        release_notes.starts_with(&format!("# {tag}\n")),
        "release-note title does not match {tag}"
    );
    assert!(release_notes.contains("package candidate"));
    assert!(release_notes.contains("physical Intel Mac"));
    assert!(
        readme.contains("python3 -B -m unittest discover -s scripts/tests -p 'test_*.py'"),
        "README must list the Python release-harness gate"
    );
}

#[test]
fn packaging_scripts_use_the_shared_helper_before_backend_tier_archives() {
    for script in packaging_scripts() {
        assert!(script.contains("julie-package-manifest"));
        assert!(script.contains("create"));
        assert!(script.contains("verify"));
        assert!(script.contains("package-manifest.json"));
        assert!(script.contains("backend"));
        assert!(script.contains("tier"));
    }
}

#[test]
fn packaging_scripts_stage_native_core_libraries_from_lib_or_lib64() {
    for script in packaging_scripts() {
        assert!(script.contains("lib64"));
    }
}

#[test]
fn packaging_scripts_reject_native_cpu_flags_and_contain_no_publication_behavior() {
    let [bash, powershell] = packaging_scripts();
    for script in [&bash, &powershell] {
        assert!(script.contains("target-cpu=native"));
        for forbidden in ["cargo publish", "gh release", "git push", "Publish-Module"] {
            assert!(!script.contains(forbidden), "forbidden {forbidden}");
        }
    }
    assert!(bash.contains("sha256sum \"$archive_name\""));
    assert!(!bash.contains("sha256sum \"$archive\""));
    assert!(powershell.contains("GetFileName($archive)"));
    assert!(!powershell.contains("$env:PATH"));
}
