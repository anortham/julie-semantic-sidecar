use sha2::{Digest, Sha256};

fn packaging_scripts() -> [String; 2] {
    [
        std::fs::read_to_string("scripts/package.sh").expect("bash package script"),
        std::fs::read_to_string("scripts/package.ps1").expect("PowerShell package script"),
    ]
}

#[test]
fn every_packaging_path_stages_the_generated_third_party_license_report() {
    let [bash, powershell] = packaging_scripts();
    for (script, staging_command) in [
        (
            bash,
            r#"cp THIRD_PARTY-LICENSES.html "$stage/THIRD_PARTY-LICENSES.html""#,
        ),
        (
            powershell,
            r#"Copy-Item THIRD_PARTY-LICENSES.html (Join-Path $stage "THIRD_PARTY-LICENSES.html")"#,
        ),
    ] {
        assert!(
            script.contains(staging_command),
            "packaging script must stage the generated report"
        );
    }
}

#[test]
fn generated_report_records_its_locked_graph_provenance_and_full_license_texts() {
    let report = std::fs::read_to_string("THIRD_PARTY-LICENSES.html")
        .expect("generated third-party license report");

    assert!(report.contains("cargo-about 0.9.1"));
    assert!(report.contains("cargo about generate --locked --all-features --fail"));
    assert!(report.contains("<pre class=\"license-text\">"));
    let lockfile = std::fs::read("Cargo.lock").expect("Cargo.lock");
    let lockfile_sha256 = format!("{:x}", Sha256::digest(lockfile));
    assert!(report.contains(&format!("Cargo.lock SHA-256: {lockfile_sha256}")));
}
