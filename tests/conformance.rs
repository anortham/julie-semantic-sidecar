//! The P2a conformance gate: every row of `semantic-sidecar-protocol-v1.md` § Conformance
//! executed against the real release binary spawned as a child over stdio.
//!
//! Groups A and B are wire/lifecycle assertions. Group C is the numeric gate, bound to the
//! frozen fixture set in `eval/sidecar-conformance/` and the frozen tolerance policy:
//! dims exactly equal to the served lane, `|norm - 1| <= 1e-3` per emitted vector, and
//! cosine `>= 0.999` against the committed golden lane vector for every corpus text.
//!
//! The heavy rows are `#[ignore]`-gated because they load multi-gigabyte models; run them
//! through `scripts/conformance.sh`. The comparison helpers themselves are pure and are
//! tested un-ignored, including the negative direction — a gate that cannot fail is not a
//! gate.

use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const BIN: &str = env!("CARGO_BIN_EXE_julie-semantic-sidecar");

/// Frozen tolerance policy — these three numbers are the bar.
const NORM_TOLERANCE: f64 = 1e-3;
const COSINE_BAR: f64 = 0.999;

/// llama.cpp build the committed goldens were generated with, printed as the drift
/// diagnostic whenever a Group C comparison fails.
const GOLDEN_LLAMA_CPP_BUILD: &str = "b10068";

/// Hard budgets from group B.
const INIT_BUDGET_MS: u128 = 120_000;
const REQUEST_BUDGET_MS: u128 = 30_000;

/// Upper bound on any single child's lifetime, so a hung sidecar fails the run instead of
/// hanging it. Far above every budget this file asserts.
const WATCHDOG_SECONDS: u64 = 900;

const ERROR_CODES: [&str; 4] = [
    "invalid_request",
    "invalid_json",
    "unknown_method",
    "internal_error",
];

// ---------------------------------------------------------------------------
// Comparison helpers (pure — covered by the un-ignored tests at the bottom)
// ---------------------------------------------------------------------------

fn l2_norm(vector: &[f32]) -> f64 {
    vector
        .iter()
        .map(|component| f64::from(*component) * f64::from(*component))
        .sum::<f64>()
        .sqrt()
}

fn cosine(actual: &[f32], golden: &[f64]) -> f64 {
    let dot: f64 = actual
        .iter()
        .zip(golden)
        .map(|(a, g)| f64::from(*a) * g)
        .sum();
    let actual_norm = l2_norm(actual);
    let golden_norm = golden.iter().map(|g| g * g).sum::<f64>().sqrt();
    if actual_norm == 0.0 || golden_norm == 0.0 {
        return 0.0;
    }
    dot / (actual_norm * golden_norm)
}

/// Reconstructs a golden lane vector from its committed symmetric int8 codes.
///
/// Cosine is scale-invariant, so the reconstruction needs no renormalization.
fn reconstruct_lane(codes: &[i32], scale: f64) -> Vec<f64> {
    codes.iter().map(|code| f64::from(*code) * scale).collect()
}

/// Applies the frozen tolerance policy to one emitted vector.
///
/// Returns every violation found, so one failing text names all of its failures at once.
fn check_vector(actual: &[f32], golden_lane: &[f64], lane_dims: usize) -> Vec<String> {
    let mut violations = Vec::new();
    if actual.len() != lane_dims {
        violations.push(format!("dims {} != lane dims {lane_dims}", actual.len()));
        return violations;
    }
    if golden_lane.len() != lane_dims {
        violations.push(format!(
            "golden lane dims {} != declared lane dims {lane_dims}",
            golden_lane.len()
        ));
        return violations;
    }
    if let Some(index) = actual.iter().position(|c| !c.is_finite()) {
        violations.push(format!("non-finite component at index {index}"));
        return violations;
    }
    let norm = l2_norm(actual);
    if (norm - 1.0).abs() > NORM_TOLERANCE {
        violations.push(format!(
            "L2 norm {norm:.6} outside 1.0 +/- {NORM_TOLERANCE}"
        ));
    }
    let similarity = cosine(actual, golden_lane);
    if similarity < COSINE_BAR {
        violations.push(format!("cosine {similarity:.6} < {COSINE_BAR}"));
    }
    violations
}

/// Cosine of an emitted vector against another emitted vector, for the batch-position probe.
fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let promoted: Vec<f64> = b.iter().map(|c| f64::from(*c)).collect();
    cosine(a, &promoted)
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct CorpusRow {
    text_id: String,
    class: String,
    role: String,
    text: String,
    #[serde(default)]
    batch_expand: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct GoldenRow {
    text_id: String,
    role: String,
    lane_dims: usize,
    vector_lane_int8: Vec<i32>,
    lane_int8_scale: f64,
    #[serde(default)]
    batch_group_positions_checked: Option<usize>,
    generator: GoldenGenerator,
}

#[derive(Debug, Deserialize)]
struct GoldenGenerator {
    llama_cpp: String,
}

fn fixtures_dir() -> PathBuf {
    let dir = std::env::var("FIXTURES_DIR")
        .unwrap_or_else(|_| "/Users/murphy/source/miller/eval/sidecar-conformance".to_string());
    let path = PathBuf::from(dir);
    assert!(
        path.join("corpus.jsonl").is_file(),
        "FIXTURES_DIR does not hold corpus.jsonl: {}",
        path.display()
    );
    path
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Vec<T> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str(line)
                .unwrap_or_else(|err| panic!("{} line {}: {err}", path.display(), index + 1))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Spawned sidecar harness
// ---------------------------------------------------------------------------

struct Sidecar {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    /// Every stdout line read during the session, for the A22 purity assertion.
    stdout_lines: Vec<String>,
    watchdog_done: Arc<AtomicBool>,
}

#[derive(Default)]
struct Spawn<'a> {
    model: Option<&'a str>,
    cache_dir: Option<&'a Path>,
    force_backend: Option<&'a str>,
}

impl Sidecar {
    fn spawn(options: Spawn<'_>) -> Self {
        let mut command = Command::new(BIN);
        command.arg("serve");
        if let Some(model) = options.model {
            command.arg("--model").arg(model);
        }
        command
            .env(
                "JULIE_SIDECAR_FORCE_BACKEND",
                options.force_backend.unwrap_or("cpu"),
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Discarded rather than piped: a long group C session logs enough to fill a
            // pipe buffer, and a full stderr pipe with no reader deadlocks the child.
            .stderr(Stdio::null());
        if let Some(dir) = options.cache_dir {
            command.env("JULIE_EMBEDDING_CACHE_DIR", dir);
        }
        let mut child = command.spawn().expect("spawn the release sidecar");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = BufReader::new(child.stdout.take().expect("stdout"));
        let watchdog_done = arm_watchdog(child.id());
        Self {
            child,
            stdin: Some(stdin),
            stdout,
            stdout_lines: Vec::new(),
            watchdog_done,
        }
    }

    fn stdin_mut(&mut self) -> &mut ChildStdin {
        self.stdin.as_mut().expect("stdin is still open")
    }

    fn send_raw(&mut self, raw: &str) {
        let stdin = self.stdin_mut();
        stdin.write_all(raw.as_bytes()).expect("write raw line");
        stdin.write_all(b"\n").expect("write newline");
        stdin.flush().expect("flush");
    }

    fn send_value(&mut self, request: &Value) {
        self.send_raw(&serde_json::to_string(request).expect("serialize request"));
    }

    fn request(&mut self, request_id: &str, method: &str, params: Value) -> Value {
        self.request_timed(request_id, method, params).0
    }

    fn request_timed(
        &mut self,
        request_id: &str,
        method: &str,
        params: Value,
    ) -> (Value, Duration) {
        let request = json!({
            "schema": "julie.embedding.sidecar",
            "version": 1,
            "request_id": request_id,
            "method": method,
            "params": params,
        });
        let started = Instant::now();
        self.send_value(&request);
        let response = self.read_envelope();
        (response, started.elapsed())
    }

    fn read_envelope(&mut self) -> Value {
        let mut line = String::new();
        let read = self
            .stdout
            .read_line(&mut line)
            .expect("read a response line");
        assert!(
            read > 0,
            "sidecar closed stdout before responding (watchdog kill or crash)"
        );
        let value: Value = serde_json::from_str(&line)
            .unwrap_or_else(|err| panic!("A22: non-protocol stdout line {line:?}: {err}"));
        self.stdout_lines.push(line);
        assert_envelope_shape(&value);
        value
    }

    fn is_alive(&mut self) -> bool {
        self.child.try_wait().expect("try_wait").is_none()
    }

    /// Sends `shutdown`, asserts row A19, and returns anything that followed on stdout.
    fn shutdown(mut self) -> String {
        let response = self.request("shutdown", "shutdown", json!({}));
        assert_eq!(
            response["result"],
            json!({"stopping": true}),
            "A19 shutdown result: {response}"
        );
        drop(self.stdin.take());
        let mut trailing = String::new();
        self.stdout
            .get_mut()
            .read_to_string(&mut trailing)
            .expect("drain stdout");
        let status = self.child.wait().expect("wait");
        self.watchdog_done.store(true, Ordering::SeqCst);
        assert!(
            status.success(),
            "A19 the process exits cleanly after shutdown: {status:?}"
        );
        trailing
    }
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        self.watchdog_done.store(true, Ordering::SeqCst);
        drop(self.stdin.take());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Kills `pid` if it outlives [`WATCHDOG_SECONDS`], so a hung sidecar surfaces as a failed
/// read rather than a hung test run.
fn arm_watchdog(pid: u32) -> Arc<AtomicBool> {
    let done = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&done);
    std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(WATCHDOG_SECONDS);
        while Instant::now() < deadline {
            if flag.load(Ordering::SeqCst) {
                return;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        if !flag.load(Ordering::SeqCst) {
            unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
        }
    });
    done
}

/// Rows A1 and A6, plus the closed error-code set — asserted on every response read.
fn assert_envelope_shape(value: &Value) {
    assert_eq!(
        value["schema"], "julie.embedding.sidecar",
        "A1 schema: {value}"
    );
    assert_eq!(value["version"], 1, "A1 version: {value}");
    let has_result = value.get("result").is_some_and(|v| !v.is_null());
    let has_error = value.get("error").is_some_and(|v| !v.is_null());
    assert!(
        has_result ^ has_error,
        "A6 exactly one of result/error: {value}"
    );
    if has_error {
        let code = value["error"]["code"].as_str().expect("error code string");
        assert!(
            ERROR_CODES.contains(&code),
            "error code {code:?} is outside the frozen closed set {ERROR_CODES:?}: {value}"
        );
    }
}

fn vector_of(value: &Value, pointer: &str) -> Vec<f32> {
    value
        .pointer(pointer)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("no vector at {pointer}: {value}"))
        .iter()
        .map(|component| component.as_f64().expect("numeric component") as f32)
        .collect()
}

/// Renders a response's error envelope, when it carries one.
fn error_of(value: &Value) -> Option<String> {
    let error = value.get("error").filter(|e| !e.is_null())?;
    Some(format!(
        "{} error: {}",
        error["code"].as_str().unwrap_or("<no code>"),
        error["message"].as_str().unwrap_or("<no message>")
    ))
}

fn pass(row: &str, detail: &str) {
    println!("PASS {row:<4} {detail}");
}

// ---------------------------------------------------------------------------
// Group A — envelope and method conformance
// ---------------------------------------------------------------------------

#[test]
#[ignore = "spawns the real sidecar with the prepared qwen3 model"]
fn group_a_envelope_and_method_rows() {
    let mut sidecar = Sidecar::spawn(Spawn::default());

    // A7 — health before any embed.
    let health = sidecar.request("h1", "health", json!({}));
    let result = &health["result"];
    let ready = result["ready"].as_bool().expect("A7 ready boolean");
    assert!(ready, "A7 the prepared model must serve ready: {health}");
    let dims = result["dims"].as_u64().expect("A7 dims when ready") as usize;
    pass("A7", &format!("ready=true dims={dims}"));

    // A1/A2 — schema, version, request_id echo (A1 asserted on every read).
    assert_eq!(health["request_id"], "h1", "A2 echo: {health}");
    pass(
        "A1",
        "schema/version present on every response read this session",
    );
    pass("A2", "request_id echoed");

    // A3 — `id` alias.
    sidecar.send_value(&json!({"id": "alias-1", "method": "health"}));
    let aliased = sidecar.read_envelope();
    assert_eq!(aliased["request_id"], "alias-1", "A3 id alias: {aliased}");
    pass("A3", "id alias echoed as request_id");

    // A4 — schema and version omitted.
    sidecar.send_value(&json!({"request_id": "a4", "method": "health"}));
    let bare = sidecar.read_envelope();
    assert!(bare["result"].is_object(), "A4 handled normally: {bare}");
    pass("A4", "request without schema/version handled normally");

    // A5 — wrong schema and wrong version.
    for (request_id, request) in [
        (
            "a5-schema",
            json!({"schema": "other", "version": 1, "request_id": "a5-schema", "method": "health"}),
        ),
        (
            "a5-version",
            json!({"schema": "julie.embedding.sidecar", "version": 2, "request_id": "a5-version", "method": "health"}),
        ),
    ] {
        sidecar.send_value(&request);
        let rejected = sidecar.read_envelope();
        assert_eq!(
            rejected["error"]["code"], "invalid_request",
            "A5 code: {rejected}"
        );
        assert_eq!(rejected["request_id"], request_id, "A5 echo: {rejected}");
    }
    pass(
        "A5",
        "schema/version mismatch -> invalid_request, id echoed",
    );

    pass(
        "A6",
        "exactly one of result/error on every response read this session",
    );

    // A8 — capability shape.
    for backend in ["cpu", "cuda", "directml", "mps"] {
        let entry = &result["capabilities"][backend];
        assert!(entry.is_object(), "A8 {backend} is an object: {health}");
        assert!(
            entry["available"].is_boolean(),
            "A8 {backend}.available is boolean: {health}"
        );
    }
    pass(
        "A8",
        "capabilities cpu/cuda/directml/mps carry boolean available",
    );

    // A10 — embed_query dims agreement.
    let query = sidecar.request("a10", "embed_query", json!({"text": "fn main() {}"}));
    assert_eq!(
        query["result"]["dims"].as_u64(),
        Some(dims as u64),
        "A10 dims: {query}"
    );
    assert_eq!(
        vector_of(&query, "/result/vector").len(),
        dims,
        "A10 vector length equals dims"
    );
    pass("A10", &format!("embed_query vector.len == dims == {dims}"));

    // A11 — embed_batch count and per-element dims.
    let batch = sidecar.request(
        "a11",
        "embed_batch",
        json!({"texts": ["alpha", "beta", "gamma"]}),
    );
    let vectors = batch["result"]["vectors"]
        .as_array()
        .expect("A11 vectors array");
    assert_eq!(vectors.len(), 3, "A11 count: {batch}");
    for (index, vector) in vectors.iter().enumerate() {
        assert_eq!(
            vector.as_array().expect("vector").len(),
            dims,
            "A11 element {index} length"
        );
    }
    pass("A11", "embed_batch returns N vectors, each of dims");

    // A12 — empty batch.
    let empty = sidecar.request("a12", "embed_batch", json!({"texts": []}));
    assert_eq!(empty["result"]["vectors"], json!([]), "A12: {empty}");
    assert!(empty["error"].is_null(), "A12 no error: {empty}");
    pass("A12", "empty texts -> empty vectors, no error");

    // A13 — empty string is embedded, not rejected.
    let empty_text = sidecar.request("a13", "embed_query", json!({"text": ""}));
    assert_eq!(
        vector_of(&empty_text, "/result/vector").len(),
        dims,
        "A13 empty input embeds: {empty_text}"
    );
    pass("A13", "empty string embeds to a full vector");

    // A14 — non-string text.
    let bad_query = sidecar.request("a14", "embed_query", json!({"text": 42}));
    assert_eq!(
        bad_query["error"]["code"], "invalid_request",
        "A14: {bad_query}"
    );
    pass("A14", "non-string embed_query text -> invalid_request");

    // A15 — non-string batch element.
    let bad_batch = sidecar.request("a15", "embed_batch", json!({"texts": ["ok", 7]}));
    assert_eq!(
        bad_batch["error"]["code"], "invalid_request",
        "A15: {bad_batch}"
    );
    pass("A15", "non-string batch element -> invalid_request");

    // A16 — unparseable line.
    sidecar.send_raw("{not json at all");
    let unparseable = sidecar.read_envelope();
    assert_eq!(
        unparseable["error"]["code"], "invalid_json",
        "A16: {unparseable}"
    );
    assert_eq!(unparseable["request_id"], "", "A16 empty request_id");
    pass(
        "A16",
        "unparseable line -> invalid_json with empty request_id",
    );

    // A17 — unknown method.
    let unknown = sidecar.request("a17", "nope", json!({}));
    assert_eq!(unknown["error"]["code"], "unknown_method", "A17: {unknown}");
    pass("A17", "unknown method -> unknown_method");

    // A18 — batch shape holds across inputs that stress the encode path.
    //
    // The engine sanitizes every degenerate input into an embeddable string, so this build
    // has no natural poison text; the forced-failure path (encode returns None -> zero
    // vector, count preserved) is proven by the pure `engine::isolate` unit seam. What is
    // asserted here is the wire half of the row: N in, N out, every vector at dims, and no
    // process exit, across the degenerate inputs that come closest to failing.
    let degenerate = json!({"texts": ["", "   ", "\u{0}\u{0}\u{0}", "\u{1}\u{2}", "𝕏🧬", "a"]});
    let isolated = sidecar.request("a18", "embed_batch", degenerate);
    let isolated_vectors = isolated["result"]["vectors"]
        .as_array()
        .expect("A18 vectors");
    assert_eq!(isolated_vectors.len(), 6, "A18 count preserved: {isolated}");
    for (index, vector) in isolated_vectors.iter().enumerate() {
        assert_eq!(
            vector.as_array().expect("vector").len(),
            dims,
            "A18 element {index} length"
        );
    }
    assert!(sidecar.is_alive(), "A18 no process exit");
    pass(
        "A18",
        "degenerate batch keeps count and dims, process alive (forced-failure path: engine::isolate unit seam)",
    );

    // A20 — a blank line draws no response.
    sidecar.send_raw("");
    sidecar.send_raw("   ");
    let after_blank = sidecar.request("a20", "health", json!({}));
    assert_eq!(
        after_blank["request_id"], "a20",
        "A20 blank lines drew a response: {after_blank}"
    );
    pass("A20", "blank lines skipped, loop continues");

    // A21 — unrecognized top-level field.
    sidecar.send_value(&json!({
        "schema": "julie.embedding.sidecar",
        "version": 1,
        "request_id": "a21",
        "method": "health",
        "telemetry_hint": {"anything": [1, 2, 3]},
    }));
    let extra = sidecar.read_envelope();
    assert!(extra["result"].is_object(), "A21: {extra}");
    pass("A21", "unrecognized top-level field ignored");

    // A23 — every error condition in one stream, then a served health.
    sidecar.send_value(&json!({"schema": "other", "request_id": "a23-1", "method": "health"}));
    let e1 = sidecar.read_envelope();
    sidecar.send_value(&json!({"version": 2, "request_id": "a23-2", "method": "health"}));
    let e2 = sidecar.read_envelope();
    let e3 = sidecar.request("a23-3", "embed_query", json!({"text": null}));
    let e4 = sidecar.request("a23-4", "embed_batch", json!({"texts": [{}]}));
    sidecar.send_raw("]]not json[[");
    let e5 = sidecar.read_envelope();
    let e6 = sidecar.request("a23-6", "teleport", json!({}));
    for (label, response, code) in [
        ("A5 schema", &e1, "invalid_request"),
        ("A5 version", &e2, "invalid_request"),
        ("A14", &e3, "invalid_request"),
        ("A15", &e4, "invalid_request"),
        ("A16", &e5, "invalid_json"),
        ("A17", &e6, "unknown_method"),
    ] {
        assert_eq!(response["error"]["code"], code, "A23 {label}: {response}");
    }
    let survivor = sidecar.request("a23-alive", "health", json!({}));
    assert_eq!(survivor["result"]["ready"], true, "A23 still serving");
    pass(
        "A23",
        "six error conditions in one stream, health still served",
    );

    // A19 + A22 — clean shutdown, then the whole session's stdout.
    let session_lines = sidecar.stdout_lines.len();
    let trailing = sidecar.shutdown();
    assert!(
        trailing.trim().is_empty(),
        "A22 stdout carried non-protocol bytes after shutdown: {trailing:?}"
    );
    pass("A19", "shutdown -> {\"stopping\": true} then exit 0");
    pass(
        "A22",
        &format!(
            "all {} stdout lines parsed as protocol envelopes, nothing trailing",
            session_lines + 1
        ),
    );
}

// ---------------------------------------------------------------------------
// Group B — lifecycle conformance
// ---------------------------------------------------------------------------

#[test]
#[ignore = "spawns the real sidecar with the prepared qwen3 model"]
fn group_b1_stdin_eof_exits_the_process() {
    let mut sidecar = Sidecar::spawn(Spawn::default());
    sidecar.request("h1", "health", json!({}));
    drop(sidecar.stdin.take());
    let status = sidecar.child.wait().expect("wait after EOF");
    sidecar.watchdog_done.store(true, Ordering::SeqCst);
    assert!(status.success(), "B1 EOF exits cleanly: {status:?}");
    pass("B1", "stdin EOF exits the process with status 0");
}

#[test]
#[ignore = "spawns the real sidecar with the prepared qwen3 model"]
fn group_b2_sigkill_leaves_no_orphan_or_lock_residue() {
    let cache = default_cache_dir();
    let before = cache_entries(&cache);

    let mut sidecar = Sidecar::spawn(Spawn::default());
    sidecar.request("h1", "health", json!({}));
    let pid = sidecar.child.id();
    sidecar.child.kill().expect("SIGKILL");
    let status = sidecar.child.wait().expect("reap");
    sidecar.watchdog_done.store(true, Ordering::SeqCst);
    assert!(!status.success(), "B2 a killed process does not exit clean");

    let alive = unsafe { libc::kill(pid as libc::pid_t, 0) } == 0;
    assert!(!alive, "B2 pid {pid} survived SIGKILL as an orphan");

    let after = cache_entries(&cache);
    let added: Vec<&String> = after
        .iter()
        .filter(|name| !before.contains(*name))
        .collect();
    assert!(
        added.is_empty(),
        "B2 a killed session left residue in the cache: {added:?}"
    );

    let mut respawned = Sidecar::spawn(Spawn::default());
    let health = respawned.request("h1", "health", json!({}));
    assert_eq!(
        health["result"]["ready"], true,
        "B2 a respawn after SIGKILL needs no consumer cleanup: {health}"
    );
    respawned.shutdown();
    pass(
        "B2",
        "SIGKILL leaves no orphan pid, no new cache residue, and the next spawn serves ready",
    );
}

#[test]
#[ignore = "spawns the real sidecar"]
fn group_b3_an_empty_cache_reports_exactly_model_not_prepared() {
    let cache = tempfile::tempdir().expect("tempdir");
    let mut sidecar = Sidecar::spawn(Spawn {
        cache_dir: Some(cache.path()),
        ..Spawn::default()
    });
    let health = sidecar.request("h1", "health", json!({}));
    assert_eq!(health["result"]["ready"], false, "B3: {health}");
    assert_eq!(
        health["result"]["degraded_reason"], "model_not_prepared",
        "B3 exact reason: {health}"
    );
    sidecar.shutdown();
    pass(
        "B3",
        "empty cache -> ready:false, degraded_reason:model_not_prepared",
    );
}

#[test]
#[ignore = "spawns the real sidecar with the prepared qwen3 model"]
fn group_b4_cold_start_answers_health_inside_the_init_budget() {
    let started = Instant::now();
    let mut sidecar = Sidecar::spawn(Spawn::default());
    let health = sidecar.request("h1", "health", json!({}));
    let elapsed = started.elapsed();
    assert_eq!(health["result"]["ready"], true, "B4 must serve ready");
    println!(
        "MEASURED B4 spawn -> first health answer: {} ms",
        elapsed.as_millis()
    );
    assert!(
        elapsed.as_millis() < INIT_BUDGET_MS,
        "B4 first health took {} ms, budget is {INIT_BUDGET_MS} ms",
        elapsed.as_millis()
    );
    sidecar.shutdown();
    pass(
        "B4",
        &format!(
            "cold start first health in {} ms (< {INIT_BUDGET_MS} ms)",
            elapsed.as_millis()
        ),
    );
}

#[test]
#[ignore = "spawns the real sidecar with the prepared qwen3 model"]
fn group_b5_a_two_hundred_fifty_text_batch_answers_inside_the_request_budget() {
    let corpus: Vec<CorpusRow> = read_jsonl(&fixtures_dir().join("corpus.jsonl"));
    let probe = corpus
        .iter()
        .find(|row| row.batch_expand.is_some())
        .expect("the corpus defines a batch-expansion row");
    let positions = probe.batch_expand.expect("batch_expand");
    let texts: Vec<&str> = std::iter::repeat_n(probe.text.as_str(), positions).collect();

    let mut sidecar = Sidecar::spawn(Spawn::default());
    sidecar.request("h1", "health", json!({}));
    let (response, elapsed) = sidecar.request_timed("b5", "embed_batch", json!({"texts": texts}));
    let vectors = response["result"]["vectors"]
        .as_array()
        .unwrap_or_else(|| panic!("B5 vectors: {response}"));
    assert_eq!(vectors.len(), positions, "B5 count");
    println!(
        "MEASURED B5 embed_batch of {positions} texts: {} ms",
        elapsed.as_millis()
    );
    assert!(
        elapsed.as_millis() < REQUEST_BUDGET_MS,
        "B5 took {} ms, budget is {REQUEST_BUDGET_MS} ms",
        elapsed.as_millis()
    );
    sidecar.shutdown();
    pass(
        "B5",
        &format!(
            "{positions}-text batch answered in {} ms (< {REQUEST_BUDGET_MS} ms)",
            elapsed.as_millis()
        ),
    );
}

#[test]
#[ignore = "spawns the real sidecar with the prepared qwen3 model"]
fn group_b6_a_forced_unavailable_backend_stays_ready_and_degraded() {
    let mut sidecar = Sidecar::spawn(Spawn {
        force_backend: Some("vulkan"),
        ..Spawn::default()
    });
    let health = sidecar.request("h1", "health", json!({}));
    let result = &health["result"];
    assert_eq!(result["ready"], true, "B6 ready: {health}");
    assert_eq!(result["accelerated"], false, "B6 accelerated: {health}");
    assert!(
        result["degraded_reason"].is_string(),
        "B6 non-null reason: {health}"
    );

    // A9 rides here: this is the only configuration where requested != resolved.
    let requested = result["load_policy"]["requested_device_backend"]
        .as_str()
        .expect("requested backend");
    let resolved = result["load_policy"]["resolved_device_backend"]
        .as_str()
        .expect("resolved backend");
    assert_ne!(requested, resolved, "A9 precondition: {health}");
    assert_eq!(
        result["load_policy"]["accelerated"], result["accelerated"],
        "A9 load_policy.accelerated mirrors top level: {health}"
    );
    assert_eq!(
        result["load_policy"]["degraded_reason"], result["degraded_reason"],
        "A9 load_policy.degraded_reason mirrors top level: {health}"
    );
    pass(
        "A9",
        "requested != resolved -> non-null reason mirrored into load_policy",
    );

    let trailing = sidecar.shutdown();
    assert!(trailing.trim().is_empty(), "B6 clean exit");
    pass(
        "B6",
        &format!("forced {requested} resolved {resolved}: ready:true, accelerated:false, reason non-null, exit 0"),
    );
}

fn default_cache_dir() -> PathBuf {
    julie_semantic_sidecar::prepare::cache_dir().expect("resolve the shared model cache")
}

fn cache_entries(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().to_string())
        .collect();
    names.sort();
    names
}

// ---------------------------------------------------------------------------
// Group C — numeric conformance, both pinned models
// ---------------------------------------------------------------------------

#[test]
#[ignore = "spawns the real sidecar with the prepared qwen3 model"]
fn group_c_qwen3_reproduces_every_golden_lane_vector() {
    run_group_c("qwen3-0.6b-f16", "golden-qwen3-0.6b-f16.jsonl", 512);
}

#[test]
#[ignore = "spawns the real sidecar with the prepared bge model"]
fn group_c_bge_reproduces_every_golden_lane_vector() {
    run_group_c("bge-small-en-v1.5-f32", "golden-bge-small-f32.jsonl", 384);
}

fn run_group_c(model_id: &str, golden_file: &str, expected_lane_dims: usize) {
    let fixtures = fixtures_dir();
    let corpus: Vec<CorpusRow> = read_jsonl(&fixtures.join("corpus.jsonl"));
    let goldens: Vec<GoldenRow> = read_jsonl(&fixtures.join(golden_file));
    assert_eq!(
        corpus.len(),
        goldens.len(),
        "the corpus and {golden_file} must line up row for row"
    );

    let mut sidecar = Sidecar::spawn(Spawn {
        model: Some(model_id),
        ..Spawn::default()
    });
    let health = sidecar.request("h1", "health", json!({}));
    assert_eq!(
        health["result"]["ready"], true,
        "group C needs a ready model: {health}"
    );
    let served_dims = health["result"]["dims"].as_u64().expect("dims") as usize;
    let sidecar_build = health["result"]["llama_cpp_build"]
        .as_str()
        .unwrap_or("<absent>")
        .to_string();
    assert_eq!(
        served_dims, expected_lane_dims,
        "{model_id} must serve the lane the goldens commit"
    );

    println!(
        "--- group C {model_id}: {} texts, lane {served_dims} ---",
        corpus.len()
    );
    let mut failures: Vec<String> = Vec::new();
    let started = Instant::now();

    for (index, row) in corpus.iter().enumerate() {
        let golden = &goldens[index];
        assert_eq!(
            golden.text_id, row.text_id,
            "fixture order mismatch at index {index}"
        );
        assert_eq!(golden.role, row.role, "role mismatch for {}", row.text_id);
        assert_eq!(
            golden.lane_dims, expected_lane_dims,
            "golden lane dims for {} disagree with the served lane",
            row.text_id
        );
        let golden_lane = reconstruct_lane(&golden.vector_lane_int8, golden.lane_int8_scale);

        let embedded = match row.role.as_str() {
            "query" => {
                let value = sidecar.request(&row.text_id, "embed_query", json!({"text": row.text}));
                error_of(&value).map_or_else(|| Ok(vector_of(&value, "/result/vector")), Err)
            }
            "document" => {
                let value =
                    sidecar.request(&row.text_id, "embed_batch", json!({"texts": [row.text]}));
                error_of(&value).map_or_else(
                    || {
                        let vectors = value["result"]["vectors"]
                            .as_array()
                            .unwrap_or_else(|| panic!("{}: {value}", row.text_id));
                        assert_eq!(vectors.len(), 1, "{}: one text, one vector", row.text_id);
                        Ok(vector_of(&value, "/result/vectors/0"))
                    },
                    Err,
                )
            }
            other => panic!("unknown role {other:?} for {}", row.text_id),
        };

        // An error envelope is a group C failure for this text, not a harness crash: it is
        // recorded and the run continues so every failing text is named in one pass.
        let actual = match embedded {
            Ok(vector) => vector,
            Err(message) => {
                failures.push(format!("{} ({}): {message}", row.text_id, row.class));
                println!("FAIL C    {model_id} {}: {message}", row.text_id);
                continue;
            }
        };
        let actual = &actual;
        let violations = check_vector(actual, &golden_lane, golden.lane_dims);
        if violations.is_empty() {
            println!(
                "PASS C    {model_id} {:<20} role={:<8} class={:<18} dims={} norm={:.6} cos={:.6}",
                row.text_id,
                row.role,
                row.class,
                actual.len(),
                l2_norm(actual),
                cosine(actual, &golden_lane),
            );
        } else {
            for violation in &violations {
                failures.push(format!("{} ({}): {violation}", row.text_id, row.class));
            }
            println!(
                "FAIL C    {model_id} {}: {}",
                row.text_id,
                violations.join("; ")
            );
        }

        // The batch-position probe: the same text at every position of a full batch.
        if let Some(positions) = row.batch_expand {
            assert_eq!(
                golden.batch_group_positions_checked,
                Some(positions),
                "{} declares {positions} positions; the golden records {:?}",
                row.text_id,
                golden.batch_group_positions_checked
            );
            let texts: Vec<&str> = std::iter::repeat_n(row.text.as_str(), positions).collect();
            let value = sidecar.request(
                &format!("{}-probe", row.text_id),
                "embed_batch",
                json!({"texts": texts}),
            );
            let vectors = value["result"]["vectors"]
                .as_array()
                .unwrap_or_else(|| panic!("{} probe: {value}", row.text_id));
            assert_eq!(vectors.len(), positions, "{} probe count", row.text_id);
            let mut probe_failures = 0usize;
            let first = vector_of(&value, "/result/vectors/0");
            for position in 0..positions {
                let vector = vector_of(&value, &format!("/result/vectors/{position}"));
                let mut violations = check_vector(&vector, &golden_lane, golden.lane_dims);
                let invariance = cosine_f32(&vector, &first);
                if invariance < COSINE_BAR {
                    violations.push(format!(
                        "batch position {position} cosine {invariance:.6} vs position 0 < {COSINE_BAR}"
                    ));
                }
                if !violations.is_empty() {
                    probe_failures += 1;
                    failures.push(format!(
                        "{} batch position {position}: {}",
                        row.text_id,
                        violations.join("; ")
                    ));
                }
            }
            if probe_failures == 0 {
                println!(
                    "PASS C    {model_id} {:<20} batch-position probe: all {positions} positions",
                    row.text_id
                );
            } else {
                println!(
                    "FAIL C    {model_id} {} batch-position probe: {probe_failures}/{positions} positions",
                    row.text_id
                );
            }
        }
    }

    let elapsed = started.elapsed();
    sidecar.shutdown();

    if !failures.is_empty() {
        let goldens_build = &goldens[0].generator.llama_cpp;
        assert_eq!(
            goldens_build, GOLDEN_LLAMA_CPP_BUILD,
            "the goldens' generator build changed; the diagnostic constant is stale"
        );
        println!(
            "DRIFT DIAGNOSTIC: sidecar llama_cpp_build ={sidecar_build} vs goldens {goldens_build}"
        );
        panic!(
            "group C failed for {model_id}: {} violation(s)\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
    println!(
        "GROUP C {model_id}: {} texts + batch probe passed in {} ms",
        corpus.len(),
        elapsed.as_millis()
    );
}

// ---------------------------------------------------------------------------
// Negative self-tests — the gate must be able to fail
// ---------------------------------------------------------------------------

fn unit_vector(dims: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..dims).map(|i| ((i % 17) as f32) - 8.0).collect();
    let norm = l2_norm(&raw);
    raw.iter().map(|c| (f64::from(*c) / norm) as f32).collect()
}

#[test]
fn an_exact_match_passes_every_bar() {
    let vector = unit_vector(512);
    let golden: Vec<f64> = vector.iter().map(|c| f64::from(*c)).collect();
    assert!(
        check_vector(&vector, &golden, 512).is_empty(),
        "an exact match must pass"
    );
}

#[test]
fn a_scaled_vector_fails_the_norm_bar() {
    let vector = unit_vector(512);
    let golden: Vec<f64> = vector.iter().map(|c| f64::from(*c)).collect();
    let scaled: Vec<f32> = vector.iter().map(|c| c * 1.01).collect();

    let violations = check_vector(&scaled, &golden, 512);
    assert!(
        violations.iter().any(|v| v.contains("L2 norm")),
        "a 1.01-scaled vector must fail the norm bar: {violations:?}"
    );
    // Cosine is scale-invariant by construction, so a pure rescale cannot move it — the
    // norm bar is the check that catches this class of defect.
    assert!(
        (cosine(&scaled, &golden) - 1.0).abs() < 1e-9,
        "a rescale must not move cosine"
    );
}

#[test]
fn a_direction_perturbed_vector_fails_the_cosine_bar() {
    let vector = unit_vector(512);
    let golden: Vec<f64> = vector.iter().map(|c| f64::from(*c)).collect();
    let mut perturbed = vector.clone();
    for (index, component) in perturbed.iter_mut().enumerate() {
        if index % 2 == 0 {
            *component *= 1.10;
        }
    }
    let norm = l2_norm(&perturbed);
    let perturbed: Vec<f32> = perturbed
        .iter()
        .map(|c| (f64::from(*c) / norm) as f32)
        .collect();

    let violations = check_vector(&perturbed, &golden, 512);
    assert!(
        violations.iter().any(|v| v.contains("cosine")),
        "a direction-perturbed vector must fail the cosine bar: {violations:?}"
    );
    assert!(
        !violations.iter().any(|v| v.contains("L2 norm")),
        "the perturbation was renormalized, so only cosine may fail: {violations:?}"
    );
}

#[test]
fn a_wrong_dimensionality_vector_fails_the_dims_bar() {
    let golden: Vec<f64> = unit_vector(512).iter().map(|c| f64::from(*c)).collect();
    let short = unit_vector(384);

    let violations = check_vector(&short, &golden, 512);
    assert!(
        violations.iter().any(|v| v.contains("dims")),
        "a 384-dim vector against a 512 lane must fail dims: {violations:?}"
    );
}

#[test]
fn a_non_finite_component_fails_before_the_numeric_bars() {
    let mut vector = unit_vector(512);
    vector[7] = f32::NAN;
    let golden: Vec<f64> = unit_vector(512).iter().map(|c| f64::from(*c)).collect();

    let violations = check_vector(&vector, &golden, 512);
    assert_eq!(violations.len(), 1, "{violations:?}");
    assert!(violations[0].contains("non-finite"), "{violations:?}");
}

#[test]
fn the_lane_reconstruction_recovers_the_golden_direction() {
    let vector = unit_vector(512);
    let scale = vector
        .iter()
        .map(|c| f64::from(c.abs()))
        .fold(0.0f64, f64::max)
        / 127.0;
    let codes: Vec<i32> = vector
        .iter()
        .map(|c| (f64::from(*c) / scale).round() as i32)
        .collect();
    assert!(codes.iter().all(|code| code.abs() <= 127), "code range");

    let reconstructed = reconstruct_lane(&codes, scale);
    assert!(
        check_vector(&vector, &reconstructed, 512).is_empty(),
        "int8 reconstruction must stay inside the cosine bar"
    );
}
