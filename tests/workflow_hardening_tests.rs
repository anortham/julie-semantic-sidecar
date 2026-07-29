use std::collections::BTreeMap;

const ACTION_PINS: [(&str, &str, &str); 9] = [
    (
        "actions/checkout",
        "11d5960a326750d5838078e36cf38b85af677262",
        "v4",
    ),
    (
        "Swatinem/rust-cache",
        "e18b497796c12c097a38f9edb9d0641fb99eee32",
        "v2",
    ),
    (
        "ilammy/msvc-dev-cmd",
        "0b201ec74fa43914dc39ae48a89fd1d8cb592756",
        "v1",
    ),
    (
        "actions/cache",
        "0057852bfaa89a56745cba8c7296529d2fc39830",
        "v4",
    ),
    (
        "humbletim/install-vulkan-sdk",
        "30ba978f977e81b72d091fc8888feb1fb26f9aff",
        "v1.2",
    ),
    (
        "humbletim/setup-vulkan-sdk",
        "c25f41106918cde0bf347e6f201277392b3a9e9d",
        "v1.2.1",
    ),
    (
        "Jimver/cuda-toolkit",
        "3d45d157f327c09c04b50ee6ccdea2d9d017ec76",
        "v0.2.35",
    ),
    (
        "actions/upload-artifact",
        "ea165f8d65b6e75b540449e92b4886f43607fa02",
        "v4",
    ),
    (
        "rustsec/audit-check",
        "69366f33c96575abad1ee0dba8212993eecbe998",
        "v2.0.0",
    ),
];

fn workflow_sources() -> [(&'static str, String); 2] {
    [
        (
            ".github/workflows/ci.yml",
            std::fs::read_to_string(".github/workflows/ci.yml").expect("ci workflow"),
        ),
        (
            ".github/workflows/release.yml",
            std::fs::read_to_string(".github/workflows/release.yml").expect("release workflow"),
        ),
    ]
}

fn action_uses(source: &str) -> impl Iterator<Item = &str> {
    source.lines().filter_map(|line| {
        let line = line.trim_start();
        let line = line
            .strip_prefix("- uses: ")
            .or_else(|| line.strip_prefix("uses: "))?;
        Some(line.trim())
    })
}

#[test]
fn every_external_action_is_pinned_to_a_full_commit_with_a_tag_comment() {
    for (path, source) in workflow_sources() {
        for action in action_uses(&source) {
            if action.starts_with("./") {
                continue;
            }
            let (reference, comment) = action
                .split_once(" # ")
                .unwrap_or_else(|| panic!("{path}: action lacks a readable tag comment: {action}"));
            let (repository, revision) = reference
                .rsplit_once('@')
                .unwrap_or_else(|| panic!("{path}: malformed action reference: {action}"));
            assert!(
                repository.contains('/'),
                "{path}: malformed action repository: {action}"
            );
            assert!(
                revision.len() == 40
                    && revision
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
                "{path}: action is not pinned to a full commit SHA: {action}"
            );
            assert!(
                comment.starts_with('v'),
                "{path}: action comment is not a readable release tag: {action}"
            );
        }
    }
}

#[test]
fn workflows_use_the_verified_action_release_commits() {
    let references = workflow_sources()
        .into_iter()
        .flat_map(|(_, source)| {
            action_uses(&source)
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let expected = ACTION_PINS
        .into_iter()
        .map(|(repository, revision, tag)| (repository, format!("{repository}@{revision} # {tag}")))
        .collect::<BTreeMap<_, _>>();

    for reference in &references {
        let repository = reference
            .split_once('@')
            .map(|(repository, _)| repository)
            .expect("external action reference");
        let verified = expected
            .get(repository)
            .unwrap_or_else(|| panic!("unverified external action: {reference}"));
        assert_eq!(
            reference, verified,
            "action use does not match its verified release commit"
        );
    }

    for (repository, reference) in &expected {
        assert!(
            references.iter().any(|candidate| candidate == reference),
            "missing verified action pin for {repository}: {reference}"
        );
    }
}

#[test]
fn ci_runs_the_rustsec_dependency_audit() {
    let ci = std::fs::read_to_string(".github/workflows/ci.yml").expect("ci workflow");
    let audit_start = ci.find("  security-audit:\n").expect("security audit job");
    let audit_end = ci[audit_start + 1..]
        .find("\n  fast:\n")
        .map(|offset| audit_start + 1 + offset)
        .expect("job after security audit");
    let audit = &ci[audit_start..audit_end];

    assert!(audit.contains("name: dependency vulnerability audit"));
    assert!(audit.contains("runs-on: ubuntu-latest"));
    assert!(audit.contains("checks: write"));
    assert!(audit
        .contains("uses: rustsec/audit-check@69366f33c96575abad1ee0dba8212993eecbe998 # v2.0.0"));
    assert!(audit.contains("token: ${{ secrets.GITHUB_TOKEN }}"));
    assert!(!audit.lines().any(|line| {
        let line = line.trim_start();
        line.strip_prefix('-')
            .unwrap_or(line)
            .trim_start()
            .starts_with("if:")
    }));
    assert!(!audit.contains("continue-on-error:"));
}

#[test]
fn dependabot_checks_cargo_and_github_actions_weekly() {
    let dependabot =
        std::fs::read_to_string(".github/dependabot.yml").expect("dependabot configuration");

    assert!(dependabot.contains("version: 2"));
    for ecosystem in ["cargo", "github-actions"] {
        let start = dependabot
            .find(&format!("package-ecosystem: \"{ecosystem}\""))
            .unwrap_or_else(|| panic!("missing {ecosystem} update configuration"));
        let entry = &dependabot[start..];
        let end = entry[1..]
            .find("package-ecosystem:")
            .map(|offset| offset + 1)
            .unwrap_or(entry.len());
        let entry = &entry[..end];
        assert!(
            entry.contains("directory: \"/\""),
            "{ecosystem} updates must target the repository root"
        );
        assert!(
            entry.contains("interval: \"weekly\""),
            "{ecosystem} updates must run weekly"
        );
    }
}
