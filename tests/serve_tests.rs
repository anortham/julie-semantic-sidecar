use julie_semantic_sidecar::manifest;
use julie_semantic_sidecar::prepare::{PARTIAL_PREFIX, PARTIAL_SUFFIX};
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

const BIN: &str = env!("CARGO_BIN_EXE_julie-semantic-sidecar");

struct Served {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Served {
    fn spawn(cache_dir: Option<&Path>) -> Self {
        let mut command = Command::new(BIN);
        command
            .arg("serve")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(dir) = cache_dir {
            command.env("JULIE_EMBEDDING_CACHE_DIR", dir);
        }
        let mut child = command.spawn().expect("spawn serve");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = BufReader::new(child.stdout.take().expect("stdout"));
        Self {
            child,
            stdin,
            stdout,
        }
    }

    fn request(&mut self, request_id: &str, method: &str, params: Value) -> Value {
        let line = serde_json::json!({
            "schema": "julie.embedding.sidecar",
            "version": 1,
            "request_id": request_id,
            "method": method,
            "params": params,
        });
        writeln!(self.stdin, "{line}").expect("write request");
        self.stdin.flush().expect("flush request");
        self.read_envelope()
    }

    fn read_envelope(&mut self) -> Value {
        let mut line = String::new();
        let read = self.stdout.read_line(&mut line).expect("read response");
        assert!(read > 0, "sidecar closed stdout before responding");
        serde_json::from_str(&line)
            .unwrap_or_else(|err| panic!("non-protocol stdout line {line:?}: {err}"))
    }

    fn is_alive(&mut self) -> bool {
        self.child.try_wait().expect("try_wait").is_none()
    }

    fn shutdown(mut self) -> (String, String) {
        let response = self.request("bye", "shutdown", serde_json::json!({}));
        assert_eq!(response["result"]["stopping"], true, "{response}");
        drop(self.stdin);
        let mut remaining = String::new();
        std::io::Read::read_to_string(self.stdout.get_mut(), &mut remaining).expect("drain stdout");
        let output = self.child.wait_with_output().expect("wait");
        assert!(output.status.success(), "clean shutdown exits zero");
        let trailing = String::from_utf8(output.stdout).expect("utf8 stdout");
        (
            format!("{remaining}{trailing}"),
            String::from_utf8_lossy(&output.stderr).to_string(),
        )
    }
}

fn assert_envelope(value: &Value, request_id: &str) {
    assert_eq!(value["schema"], "julie.embedding.sidecar", "{value}");
    assert_eq!(value["version"], 1, "{value}");
    assert_eq!(value["request_id"], request_id, "{value}");
}

#[test]
fn a_missing_model_serves_a_wire_conformant_not_ready_health() {
    let cache = tempfile::tempdir().expect("tempdir");
    let mut served = Served::spawn(Some(cache.path()));

    let health = served.request("h1", "health", serde_json::json!({}));
    assert_envelope(&health, "h1");
    assert_eq!(health["result"]["ready"], false, "{health}");
    assert_eq!(
        health["result"]["degraded_reason"], "model_not_prepared",
        "{health}"
    );

    let embed = served.request("e1", "embed_query", serde_json::json!({"text": "hello"}));
    assert_envelope(&embed, "e1");
    assert_eq!(embed["error"]["code"], "internal_error", "{embed}");
    assert!(embed["result"].is_null(), "{embed}");

    assert!(served.is_alive(), "an unprepared model must not kill serve");
    let (trailing_stdout, _stderr) = served.shutdown();
    assert!(
        trailing_stdout.trim().is_empty(),
        "nothing follows the shutdown envelope: {trailing_stdout:?}"
    );
}

#[test]
fn an_unprepared_serve_answers_every_request_with_an_envelope() {
    let cache = tempfile::tempdir().expect("tempdir");
    let mut served = Served::spawn(Some(cache.path()));

    let batch = served.request(
        "b1",
        "embed_batch",
        serde_json::json!({"texts": ["one", "two"]}),
    );
    assert_envelope(&batch, "b1");
    assert_eq!(batch["error"]["code"], "internal_error", "{batch}");

    let unknown = served.request("u1", "teleport", serde_json::json!({}));
    assert_envelope(&unknown, "u1");
    assert_eq!(unknown["error"]["code"], "unknown_method", "{unknown}");

    served.shutdown();
}

#[test]
fn startup_removes_stale_partial_downloads() {
    let cache = tempfile::tempdir().expect("tempdir");
    let stale = cache
        .path()
        .join(format!("{PARTIAL_PREFIX}abc123{PARTIAL_SUFFIX}"));
    let keeper = cache.path().join("already-prepared.gguf");
    std::fs::write(&stale, b"half a download").expect("seed partial");
    std::fs::write(&keeper, b"not a partial").expect("seed keeper");

    let mut served = Served::spawn(Some(cache.path()));
    served.request("h1", "health", serde_json::json!({}));
    served.shutdown();

    assert!(
        !stale.exists(),
        "a stale partial must be removed at startup"
    );
    assert!(keeper.exists(), "a non-partial file must be left alone");
}

#[test]
#[ignore = "requires the prepared model in the shared cache"]
fn discovery_benchmark_model_load_and_inference_write_only_protocol_lines_to_stdout() {
    let mut served = Served::spawn(None);

    let health = served.request("h1", "health", serde_json::json!({}));
    assert_envelope(&health, "h1");
    assert_eq!(health["result"]["ready"], true, "{health}");
    let dims = health["result"]["dims"].as_u64().expect("dims") as usize;

    let query = served.request(
        "q1",
        "embed_query",
        serde_json::json!({"text": "fn main() {}"}),
    );
    assert_envelope(&query, "q1");
    assert_eq!(
        query["result"]["dims"].as_u64(),
        Some(dims as u64),
        "{query}"
    );
    assert_eq!(
        query["result"]["vector"].as_array().expect("vector").len(),
        dims
    );

    let batch = served.request(
        "b1",
        "embed_batch",
        serde_json::json!({"texts": ["alpha", "beta"]}),
    );
    assert_envelope(&batch, "b1");
    assert_eq!(
        batch["result"]["vectors"]
            .as_array()
            .expect("vectors")
            .len(),
        2
    );

    let (trailing_stdout, _stderr) = served.shutdown();
    assert!(
        trailing_stdout.trim().is_empty(),
        "native discovery, benchmark, model load, and inference chatter must never reach stdout: {trailing_stdout:?}"
    );
}

#[test]
#[ignore = "requires the prepared model in the shared cache"]
fn a_ready_health_reports_the_selected_backend() {
    let mut served = Served::spawn(None);
    let health = served.request("h1", "health", serde_json::json!({}));
    let result = &health["result"];
    assert_eq!(result["ready"], true, "{health}");
    assert_eq!(result["resolved_backend"], "cpu", "{health}");
    assert_eq!(result["accelerated"], false, "{health}");
    assert_eq!(result["degraded_reason"], Value::Null, "{health}");
    assert_eq!(result["capabilities"]["cpu"]["available"], true, "{health}");
    assert_eq!(
        result["load_policy"]["requested_device_backend"], "cpu",
        "{health}"
    );
    served.shutdown();
}

/// llama.cpp's own placement report, e.g. `load_tensors: offloaded 29/29 layers to GPU`.
///
/// It is the only evidence of where the weights actually went; `health` reports the
/// *decision*, and before the load applied `n_gpu_layers` the two disagreed silently.
fn offloaded_layers(stderr: &str) -> Option<u32> {
    let line = stderr.lines().find(|l| l.contains("offloaded "))?;
    let tail = line.split("offloaded ").nth(1)?;
    tail.split('/').next()?.trim().parse().ok()
}

#[test]
#[ignore = "requires the prepared model in the shared cache"]
fn a_forced_cpu_backend_offloads_no_layers_to_the_gpu() {
    let mut command = Command::new(BIN);
    command
        .arg("serve")
        .env("JULIE_SIDECAR_FORCE_BACKEND", "cpu")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().expect("spawn serve");
    let mut stdin = child.stdin.take().expect("stdin");
    let mut stdout = BufReader::new(child.stdout.take().expect("stdout"));
    writeln!(
        stdin,
        "{}",
        serde_json::json!({"request_id": "h1", "method": "health"})
    )
    .expect("write");
    stdin.flush().expect("flush");
    let mut line = String::new();
    stdout.read_line(&mut line).expect("read");
    let health: Value = serde_json::from_str(&line).expect("protocol json");
    assert_eq!(health["result"]["ready"], true, "{health}");
    assert_eq!(health["result"]["resolved_backend"], "cpu", "{health}");

    drop(stdin);
    let output = child.wait_with_output().expect("wait");
    let stderr = String::from_utf8_lossy(&output.stderr);

    let offload_lines: Vec<_> = stderr.lines().filter(|l| l.contains("offload")).collect();
    match offloaded_layers(&stderr) {
        Some(layers) => assert_eq!(
            layers, 0,
            "a cpu selection must load with zero layers on the GPU; llama.cpp reported: {offload_lines:?}"
        ),
        // A GPU-less build (Linux CI) prints no placement line at all: nothing was
        // offloaded because there is nowhere to offload to. Any surviving offload
        // mention we failed to parse is format drift and must still fail.
        None => assert_eq!(
            offload_lines,
            Vec::<&str>::new(),
            "offload lines present but none parseable as a placement report"
        ),
    }
    assert!(
        !stderr.contains("offloading output layer to GPU"),
        "the output layer must stay on the cpu"
    );
}

#[test]
#[ignore = "requires the prepared model in the shared cache"]
fn a_corrupt_cached_model_serves_not_ready_rather_than_loading_it() {
    let cache = tempfile::tempdir().expect("tempdir");
    let model_file = manifest::default_model().file;
    let real = dirs_cache().join(model_file);
    assert!(
        real.is_file(),
        "the pinned model must be prepared: {real:?}"
    );
    let corrupt = cache.path().join(model_file);
    let mut bytes = std::fs::read(&real).expect("read model")[..4096].to_vec();
    bytes[4095] ^= 0xff;
    std::fs::write(&corrupt, &bytes).expect("seed corrupt model");

    let mut served = Served::spawn(Some(cache.path()));
    let health = served.request("h1", "health", serde_json::json!({}));
    assert_eq!(health["result"]["ready"], false, "{health}");
    assert_eq!(
        health["result"]["degraded_reason"], "model_not_prepared",
        "{health}"
    );

    let embed = served.request("e1", "embed_query", serde_json::json!({"text": "hello"}));
    assert_eq!(embed["error"]["code"], "internal_error", "{embed}");
    assert!(served.is_alive(), "a corrupt model must not kill serve");

    let (_stdout, stderr) = served.shutdown();
    assert!(
        stderr.contains("does not match its pinned digest"),
        "the diagnostic must name the mismatch: {stderr:?}"
    );
    assert!(
        stderr.contains("expected sha256") && stderr.contains("found"),
        "the diagnostic must name expected and actual: {stderr:?}"
    );
}

/// The shared cache `prepare` writes to, resolved the same way the binary resolves it.
fn dirs_cache() -> std::path::PathBuf {
    std::env::var_os("JULIE_EMBEDDING_CACHE_DIR")
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            #[cfg(windows)]
            {
                std::path::PathBuf::from(std::env::var_os("LOCALAPPDATA").expect("LOCALAPPDATA"))
                    .join("julie-semantic")
            }
            #[cfg(not(windows))]
            {
                std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME"))
                    .join(".cache")
                    .join("julie-semantic")
            }
        })
}

#[test]
#[ignore = "requires the prepared model in the shared cache"]
fn a_forced_unavailable_backend_stays_ready_and_degraded() {
    let mut command = Command::new(BIN);
    command
        .arg("serve")
        .env("JULIE_SIDECAR_FORCE_BACKEND", "vulkan")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().expect("spawn serve");
    let mut stdin = child.stdin.take().expect("stdin");
    let mut stdout = BufReader::new(child.stdout.take().expect("stdout"));
    writeln!(
        stdin,
        "{}",
        serde_json::json!({"request_id": "h1", "method": "health"})
    )
    .expect("write");
    stdin.flush().expect("flush");
    let mut line = String::new();
    stdout.read_line(&mut line).expect("read");
    let health: Value = serde_json::from_str(&line).expect("protocol json");
    let result = &health["result"];
    assert_eq!(result["ready"], true, "{health}");
    assert_eq!(result["accelerated"], false, "{health}");
    assert_eq!(result["load_policy"]["requested_device_backend"], "vulkan");
    assert_eq!(result["resolved_backend"], "cpu", "{health}");
    assert!(result["degraded_reason"].is_string(), "{health}");
    drop(stdin);
    child.wait().expect("wait");
}
