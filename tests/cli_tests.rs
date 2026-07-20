#[allow(unused_imports)]
use julie_semantic_sidecar::{
    engine_trait, health, manifest, prepare, protocol, sanitize, truncate,
};
use std::io::Write;
use std::process::{Command, Stdio};

const BIN: &str = env!("CARGO_BIN_EXE_julie-semantic-sidecar");

#[test]
fn protocol_constants_come_from_the_library_target() {
    assert_eq!(protocol::SCHEMA, "julie.embedding.sidecar");
    assert_eq!(protocol::VERSION, 1);
}

#[test]
fn declared_cache_lock_api_is_available_for_prepare() {
    use fs4::FileExt;
    let dir = tempfile::tempdir().expect("tempdir");
    let file = std::fs::File::create(dir.path().join("cache.lock")).expect("create");
    // std::fs::File gained inherent lock/unlock in Rust 1.89, which shadows fs4's
    // same-named trait methods; call fs4 fully qualified so the crate is actually used.
    FileExt::lock(&file).expect("lock");
    FileExt::unlock(&file).expect("unlock");
}

#[test]
fn version_flag_prints_name_and_semver() {
    let out = Command::new(BIN).arg("--version").output().expect("spawn");
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).expect("utf8");
    let printed = stdout.trim();
    let semver = printed
        .strip_prefix("julie-semantic-sidecar ")
        .unwrap_or_else(|| panic!("unexpected version line: {printed}"));
    assert_eq!(semver, env!("CARGO_PKG_VERSION"));
    let core = semver.split(['-', '+']).next().expect("semver core");
    assert_eq!(core.split('.').count(), 3);
}

#[test]
fn unknown_verb_exits_two_with_usage_on_stderr() {
    let out = Command::new(BIN).arg("embed").output().expect("spawn");
    assert_eq!(out.status.code(), Some(2));
    assert!(out.stdout.is_empty());
    let stderr = String::from_utf8(out.stderr).expect("utf8");
    assert!(stderr.contains("unknown verb: embed"), "{stderr}");
    assert!(stderr.contains("usage:"), "{stderr}");
}

#[test]
fn serve_reads_stdin_and_exits_cleanly_on_eof() {
    // An empty cache dir keeps this about the stdin/EOF lifecycle: no multi-gigabyte load,
    // and the not-prepared health payload is the same one line either way.
    let cache = tempfile::tempdir().expect("tempdir");
    let mut child = Command::new(BIN)
        .arg("serve")
        .env("JULIE_EMBEDDING_CACHE_DIR", cache.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut stdin = child.stdin.take().expect("stdin");
    stdin
        .write_all(b"{\"method\":\"health\"}\n")
        .expect("write");
    drop(stdin);
    let out = child.wait_with_output().expect("wait");
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).expect("utf8");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 1, "one protocol line per request: {stdout}");
    let envelope: serde_json::Value = serde_json::from_str(lines[0]).expect("protocol json");
    assert_eq!(envelope["schema"], "julie.embedding.sidecar");
    assert_eq!(envelope["version"], 1);
    assert_eq!(envelope["result"]["ready"], false);
}

#[test]
fn prepare_rejects_unknown_model_offline_with_exit_two() {
    let out = Command::new(BIN)
        .args(["prepare", "--model", "not-a-model"])
        .output()
        .expect("spawn");
    assert_eq!(out.status.code(), Some(2));
    let stdout = String::from_utf8(out.stdout).expect("utf8");
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(lines.len(), 1, "one ndjson event: {stdout}");
    let event: serde_json::Value = serde_json::from_str(lines[0]).expect("ndjson event");
    assert_eq!(event["event"], "error");
    assert_eq!(event["model_id"], "not-a-model");
    assert!(
        event["message"]
            .as_str()
            .expect("message")
            .contains("qwen3-0.6b-f16"),
        "{event}"
    );
}
