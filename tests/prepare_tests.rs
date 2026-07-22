use julie_semantic_sidecar::manifest;
use julie_semantic_sidecar::prepare::{self, PrepareRequest, PARTIAL_PREFIX, PARTIAL_SUFFIX};
use sha2::{Digest, Sha256};
use std::net::TcpListener;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

const BIN: &str = env!("CARGO_BIN_EXE_julie-semantic-sidecar");

struct Fixture {
    server: Arc<tiny_http::Server>,
    hits: Arc<AtomicUsize>,
    url: String,
    handle: Option<thread::JoinHandle<()>>,
}

impl Fixture {
    fn start(body: Vec<u8>, delay: Duration) -> Self {
        let server = Arc::new(tiny_http::Server::http("127.0.0.1:0").expect("bind fixture server"));
        let url = format!("http://{}/model.gguf", server.server_addr());
        let hits = Arc::new(AtomicUsize::new(0));
        let handle = {
            let server = Arc::clone(&server);
            let hits = Arc::clone(&hits);
            thread::spawn(move || {
                for request in server.incoming_requests() {
                    hits.fetch_add(1, Ordering::SeqCst);
                    thread::sleep(delay);
                    let _ = request.respond(tiny_http::Response::from_data(body.clone()));
                }
            })
        };
        Fixture {
            server,
            hits,
            url,
            handle: Some(handle),
        }
    }

    /// Serves `body` with no `Content-Length`, so tiny_http streams it chunked and the client
    /// learns the real size only by reading it.
    fn start_chunked(body: Vec<u8>) -> Self {
        let server = Arc::new(tiny_http::Server::http("127.0.0.1:0").expect("bind fixture server"));
        let url = format!("http://{}/model.gguf", server.server_addr());
        let hits = Arc::new(AtomicUsize::new(0));
        let handle = {
            let server = Arc::clone(&server);
            let hits = Arc::clone(&hits);
            thread::spawn(move || {
                for request in server.incoming_requests() {
                    hits.fetch_add(1, Ordering::SeqCst);
                    let response = tiny_http::Response::new(
                        tiny_http::StatusCode(200),
                        Vec::new(),
                        std::io::Cursor::new(body.clone()),
                        None,
                        None,
                    );
                    let _ = request.respond(response);
                }
            })
        };
        Fixture {
            server,
            hits,
            url,
            handle: Some(handle),
        }
    }

    fn hits(&self) -> usize {
        self.hits.load(Ordering::SeqCst)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.unblock();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn request_for(fixture: &Fixture, body: &[u8]) -> PrepareRequest {
    request_for_url(&fixture.url, body)
}

fn request_for_url(url: &str, body: &[u8]) -> PrepareRequest {
    PrepareRequest {
        model_id: "test-model".to_string(),
        file_name: "test-model.gguf".to_string(),
        source_url: url.to_string(),
        sha256: sha256_hex(body),
        size_bytes: body.len() as u64,
    }
}

fn events(raw: &[u8]) -> Vec<serde_json::Value> {
    String::from_utf8(raw.to_vec())
        .expect("utf8 events")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("ndjson event"))
        .collect()
}

fn kinds(events: &[serde_json::Value]) -> Vec<String> {
    events
        .iter()
        .map(|event| event["event"].as_str().expect("event key").to_string())
        .collect()
}

fn partials(dir: &Path) -> Vec<String> {
    let mut found: Vec<String> = std::fs::read_dir(dir)
        .expect("read cache dir")
        .map(|entry| {
            entry
                .expect("dir entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|name| name.starts_with(PARTIAL_PREFIX) && name.ends_with(PARTIAL_SUFFIX))
        .collect();
    found.sort();
    found
}

fn partial_prefix() -> String {
    prepare::partial_prefix("test-model")
}

fn hold_lock(path: &Path) -> std::fs::File {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .expect("open lock");
    fs4::FileExt::try_lock(&file).expect("take lock");
    file
}

fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
fn successful_download_emits_progress_then_done_and_leaves_no_partials() {
    let body = b"gguf-payload-for-the-happy-path".to_vec();
    let fixture = Fixture::start(body.clone(), Duration::ZERO);
    let cache = tempfile::tempdir().expect("tempdir");
    let request = request_for(&fixture, &body);
    let mut out = Vec::new();

    let prepared = prepare::acquire(&request, cache.path(), &mut out).expect("prepare succeeds");

    let events = events(&out);
    let kinds = kinds(&events);
    assert!(kinds.contains(&"progress".to_string()), "{kinds:?}");
    assert_eq!(kinds.last().map(String::as_str), Some("done"), "{kinds:?}");

    let done = events.last().expect("done event");
    assert_eq!(done["model_id"], "test-model");
    assert_eq!(done["sha256"], request.sha256);
    assert_eq!(done["path"], prepared.path.to_string_lossy().as_ref());

    let progress = events
        .iter()
        .find(|event| event["event"] == "progress")
        .expect("progress event");
    assert_eq!(progress["model_id"], "test-model");
    assert_eq!(progress["total_bytes"], body.len() as u64);

    assert_eq!(prepared.path, cache.path().join("test-model.gguf"));
    assert_eq!(std::fs::read(&prepared.path).expect("read model"), body);
    assert!(!prepared.already_cached);
    assert!(partials(cache.path()).is_empty());
    assert_eq!(fixture.hits(), 1);
}

#[test]
fn sha256_mismatch_deletes_the_partial_and_reports_an_error_event() {
    let body = b"payload-that-will-not-match".to_vec();
    let fixture = Fixture::start(body.clone(), Duration::ZERO);
    let cache = tempfile::tempdir().expect("tempdir");
    let mut request = request_for(&fixture, &body);
    request.sha256 = "0".repeat(64);
    let mut out = Vec::new();

    let error = prepare::acquire(&request, cache.path(), &mut out).expect_err("digest mismatch");

    assert!(error.to_string().contains("sha256"), "{error}");
    let events = events(&out);
    let last = events.last().expect("error event");
    assert_eq!(last["event"], "error");
    assert_eq!(last["model_id"], "test-model");
    assert_eq!(last["source_url"], request.source_url);
    assert_eq!(
        last["expected_path"],
        cache
            .path()
            .join("test-model.gguf")
            .to_string_lossy()
            .as_ref()
    );
    assert!(
        last["message"]
            .as_str()
            .expect("message")
            .contains("sha256"),
        "{last}"
    );

    assert!(!cache.path().join("test-model.gguf").exists());
    assert!(partials(cache.path()).is_empty());
}

#[test]
fn disk_preflight_fails_before_any_request_when_the_pin_exceeds_free_space() {
    let body = b"never-served".to_vec();
    let fixture = Fixture::start(body.clone(), Duration::ZERO);
    let cache = tempfile::tempdir().expect("tempdir");
    let mut request = request_for(&fixture, &body);
    request.size_bytes = u64::MAX / 2;
    let mut out = Vec::new();

    let error = prepare::acquire(&request, cache.path(), &mut out).expect_err("preflight fails");

    assert!(error.to_string().contains("space"), "{error}");
    let events = events(&out);
    assert_eq!(kinds(&events), vec!["error".to_string()]);
    assert_eq!(events[0]["source_url"], request.source_url);
    assert_eq!(fixture.hits(), 0);
    assert!(!cache.path().join("test-model.gguf").exists());
}

#[test]
fn concurrent_prepares_download_once_and_the_waiter_reports_waiting() {
    let body = b"payload-downloaded-exactly-once".to_vec();
    let fixture = Fixture::start(body.clone(), Duration::from_millis(400));
    let cache = tempfile::tempdir().expect("tempdir");
    let request = request_for(&fixture, &body);

    let first = {
        let request = request.clone();
        let dir = cache.path().to_path_buf();
        thread::spawn(move || {
            let mut out = Vec::new();
            let result = prepare::acquire(&request, &dir, &mut out);
            (result.map(|p| p.path), out)
        })
    };
    thread::sleep(Duration::from_millis(150));
    let second = {
        let request = request.clone();
        let dir = cache.path().to_path_buf();
        thread::spawn(move || {
            let mut out = Vec::new();
            let result = prepare::acquire(&request, &dir, &mut out);
            (result.map(|p| p.path), out)
        })
    };

    let (first_result, first_out) = first.join().expect("first thread");
    let (second_result, second_out) = second.join().expect("second thread");
    let model = cache.path().join("test-model.gguf");
    assert_eq!(first_result.expect("first prepared"), model);
    assert_eq!(second_result.expect("second prepared"), model);

    assert_eq!(fixture.hits(), 1);
    assert_eq!(std::fs::read(&model).expect("read model"), body);
    assert!(partials(cache.path()).is_empty());

    let waited: Vec<bool> = [&first_out, &second_out]
        .iter()
        .map(|out| kinds(&events(out)).contains(&"waiting".to_string()))
        .collect();
    assert_eq!(
        waited.iter().filter(|w| **w).count(),
        1,
        "exactly one waiter"
    );
    for out in [&first_out, &second_out] {
        assert_eq!(kinds(&events(out)).last().map(String::as_str), Some("done"));
    }
}

#[test]
fn already_cached_model_is_served_without_touching_the_network() {
    let body = b"already-on-disk".to_vec();
    let fixture = Fixture::start(body.clone(), Duration::ZERO);
    let cache = tempfile::tempdir().expect("tempdir");
    let request = request_for(&fixture, &body);
    std::fs::write(cache.path().join("test-model.gguf"), &body).expect("seed cache");
    let mut out = Vec::new();

    let prepared = prepare::acquire(&request, cache.path(), &mut out).expect("cached hit");

    assert!(prepared.already_cached);
    assert_eq!(fixture.hits(), 0);
    let events = events(&out);
    assert_eq!(kinds(&events), vec!["done".to_string()]);
    assert_eq!(events[0]["sha256"], request.sha256);
}

#[test]
fn cached_file_with_a_wrong_digest_is_treated_as_stale_and_redownloaded() {
    let body = b"the-authentic-payload".to_vec();
    let fixture = Fixture::start(body.clone(), Duration::ZERO);
    let cache = tempfile::tempdir().expect("tempdir");
    let request = request_for(&fixture, &body);
    std::fs::write(cache.path().join("test-model.gguf"), b"stale-garbage").expect("seed cache");
    let mut out = Vec::new();

    let prepared = prepare::acquire(&request, cache.path(), &mut out).expect("stale replaced");

    assert!(!prepared.already_cached);
    assert_eq!(fixture.hits(), 1);
    assert_eq!(std::fs::read(&prepared.path).expect("read model"), body);
    assert!(partials(cache.path()).is_empty());
}

#[test]
fn unreachable_source_reports_the_model_id_expected_path_and_source_url() {
    let closed = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = closed.local_addr().expect("addr").port();
    drop(closed);
    let cache = tempfile::tempdir().expect("tempdir");
    let request = PrepareRequest {
        model_id: "test-model".to_string(),
        file_name: "test-model.gguf".to_string(),
        source_url: format!("http://127.0.0.1:{port}/model.gguf"),
        sha256: "0".repeat(64),
        size_bytes: 16,
    };
    let mut out = Vec::new();

    prepare::acquire(&request, cache.path(), &mut out).expect_err("unreachable source");

    let events = events(&out);
    let error = events.last().expect("error event");
    assert_eq!(error["event"], "error");
    assert_eq!(error["model_id"], "test-model");
    assert_eq!(error["source_url"], request.source_url);
    assert_eq!(
        error["expected_path"],
        cache
            .path()
            .join("test-model.gguf")
            .to_string_lossy()
            .as_ref()
    );
    assert!(!cache.path().join("test-model.gguf").exists());
    assert!(partials(cache.path()).is_empty());
}

#[test]
fn clean_stale_partials_removes_partials_and_keeps_the_model_file() {
    let cache = tempfile::tempdir().expect("tempdir");
    let keep = cache.path().join("test-model.gguf");
    let lock = cache.path().join("test-model.lock");
    let stale_one = cache
        .path()
        .join(format!("{PARTIAL_PREFIX}abc123{PARTIAL_SUFFIX}"));
    let stale_two = cache
        .path()
        .join(format!("{}def456{PARTIAL_SUFFIX}", partial_prefix()));
    for path in [&keep, &lock, &stale_one, &stale_two] {
        std::fs::write(path, b"x").expect("seed file");
    }

    let removed = prepare::clean_stale_partials(cache.path()).expect("cleanup");

    assert_eq!(removed.len(), 2, "{removed:?}");
    assert!(!stale_one.exists());
    assert!(!stale_two.exists());
    assert!(keep.exists());
    assert!(
        lock.exists(),
        "free lock files are permanent: unlinking one races an open-before-lock prepare into a second download"
    );
    assert!(partials(cache.path()).is_empty());
}

#[test]
fn a_partial_whose_model_lock_is_held_survives_cleanup() {
    let cache = tempfile::tempdir().expect("tempdir");
    let lock_path = cache.path().join("test-model.lock");
    let in_flight = cache
        .path()
        .join(format!("{}live99{PARTIAL_SUFFIX}", partial_prefix()));
    std::fs::write(&in_flight, b"half a download").expect("seed partial");
    let _held = hold_lock(&lock_path);

    let removed = prepare::clean_stale_partials(cache.path()).expect("cleanup");

    assert!(removed.is_empty(), "{removed:?}");
    assert!(
        in_flight.exists(),
        "an active download's partial must survive"
    );
    assert!(lock_path.exists(), "a held lock file must survive");
}

#[test]
fn a_held_lock_protects_only_its_own_model_and_unattributable_partials() {
    let cache = tempfile::tempdir().expect("tempdir");
    let live_lock = cache.path().join("busy-model.lock");
    let idle_lock = cache.path().join("idle-model.lock");
    std::fs::write(&idle_lock, b"").expect("seed lock");
    let live = cache
        .path()
        .join(format!("{PARTIAL_PREFIX}busy-model.aaa{PARTIAL_SUFFIX}"));
    let idle = cache
        .path()
        .join(format!("{PARTIAL_PREFIX}idle-model.bbb{PARTIAL_SUFFIX}"));
    let legacy = cache
        .path()
        .join(format!("{PARTIAL_PREFIX}nomodel{PARTIAL_SUFFIX}"));
    for path in [&live, &idle, &legacy] {
        std::fs::write(path, b"x").expect("seed partial");
    }
    let _held = hold_lock(&live_lock);

    let removed = prepare::clean_stale_partials(cache.path()).expect("cleanup");

    assert_eq!(removed, vec![idle.clone()], "{removed:?}");
    assert!(live.exists(), "the busy model's partial must survive");
    assert!(!idle.exists(), "the idle model's partial must be removed");
    assert!(
        legacy.exists(),
        "an unattributable partial must survive while any download is live"
    );
    assert!(
        idle_lock.exists(),
        "free lock files are permanent: unlinking one races an open-before-lock prepare into a second download"
    );
}

#[test]
fn an_interrupted_download_leaves_a_partial_that_later_cleanup_removes() {
    let body = b"payload-for-the-partial-naming-check".to_vec();
    let fixture = Fixture::start(body.clone(), Duration::ZERO);
    let cache = tempfile::tempdir().expect("tempdir");
    let mut request = request_for(&fixture, &body);
    request.model_id = "named-model".to_string();
    request.sha256 = "0".repeat(64);
    let mut out = Vec::new();

    prepare::acquire(&request, cache.path(), &mut out).expect_err("digest mismatch");
    let orphan = cache.path().join(format!(
        "{PARTIAL_PREFIX}named-model.orphaned{PARTIAL_SUFFIX}"
    ));
    std::fs::write(&orphan, b"left behind").expect("seed orphan");

    let removed = prepare::clean_stale_partials(cache.path()).expect("cleanup");

    assert_eq!(removed, vec![orphan.clone()], "{removed:?}");
    assert!(cache.path().join("named-model.lock").exists());
}

struct TimeoutEnvGuard;

impl TimeoutEnvGuard {
    fn set(secs: &str) -> Self {
        std::env::set_var("JULIE_SIDECAR_DOWNLOAD_TIMEOUT_SECS", secs);
        TimeoutEnvGuard
    }
}

impl Drop for TimeoutEnvGuard {
    fn drop(&mut self) {
        std::env::remove_var("JULIE_SIDECAR_DOWNLOAD_TIMEOUT_SECS");
    }
}

#[test]
fn a_stalled_server_fails_the_download_and_releases_the_model_lock() {
    let body = b"stalled-then-served".to_vec();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let stalled_url = format!("http://{}/model.gguf", listener.local_addr().expect("addr"));
    // Detached on purpose: accepts one connection and never writes a byte. Joining a
    // sleeping handler (the Fixture pattern) would hang the test's own teardown.
    thread::spawn(move || {
        if let Ok((stream, _)) = listener.accept() {
            thread::sleep(Duration::from_secs(30));
            drop(stream);
        }
    });
    let cache = tempfile::tempdir().expect("tempdir");
    let mut request = request_for_url(&stalled_url, &body);
    request.model_id = "stall-model".to_string();
    request.file_name = "stall-model.gguf".to_string();
    let _env = env_guard();
    let _guard = TimeoutEnvGuard::set("5");
    let mut out = Vec::new();

    let started = std::time::Instant::now();
    let err = prepare::acquire(&request, cache.path(), &mut out).expect_err("stall must not hang");
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "the deadline must fire long before the 30 s stall"
    );
    assert!(
        err.message()
            .contains("JULIE_SIDECAR_DOWNLOAD_TIMEOUT_SECS"),
        "the error must name the slow-link escape hatch: {}",
        err.message()
    );

    let healthy = Fixture::start(body.clone(), Duration::ZERO);
    let mut request = request_for(&healthy, &body);
    request.model_id = "stall-model".to_string();
    request.file_name = "stall-model.gguf".to_string();
    let mut out = Vec::new();
    prepare::acquire(&request, cache.path(), &mut out)
        .expect("the stalled attempt must release the model lock");
}

#[test]
fn a_content_length_disagreeing_with_the_pin_fails_before_the_body_is_read() {
    let body = vec![b'x'; 4096];
    let fixture = Fixture::start(body.clone(), Duration::ZERO);
    let cache = tempfile::tempdir().expect("tempdir");
    let mut request = request_for(&fixture, &body);
    request.size_bytes = 64;
    let mut out = Vec::new();

    let error = prepare::acquire(&request, cache.path(), &mut out).expect_err("size mismatch");

    assert!(error.to_string().contains("declared size"), "{error}");
    let events = events(&out);
    assert_eq!(kinds(&events), vec!["error".to_string()]);
    assert!(!cache.path().join("test-model.gguf").exists());
    assert!(partials(cache.path()).is_empty());
}

#[test]
fn an_oversized_chunked_response_is_aborted_at_the_cap_and_the_partial_is_deleted() {
    let body = vec![b'x'; 512 * 1024];
    let fixture = Fixture::start_chunked(body.clone());
    let cache = tempfile::tempdir().expect("tempdir");
    let mut request = request_for(&fixture, &body);
    request.size_bytes = 1024;
    let mut out = Vec::new();

    let error = prepare::acquire(&request, cache.path(), &mut out).expect_err("oversized body");

    assert!(error.to_string().contains("oversized"), "{error}");
    let events = events(&out);
    assert_eq!(kinds(&events).last().map(String::as_str), Some("error"));
    assert!(!cache.path().join("test-model.gguf").exists());
    assert!(partials(cache.path()).is_empty());
}

#[test]
fn clean_stale_partials_on_a_missing_directory_is_not_an_error() {
    let cache = tempfile::tempdir().expect("tempdir");
    let missing = cache.path().join("never-created");

    let removed = prepare::clean_stale_partials(&missing).expect("cleanup tolerates absence");

    assert!(removed.is_empty());
}

#[test]
fn resolve_defaults_to_the_manifest_default_tier() {
    let pin = prepare::resolve(None).expect("default pin");
    assert_eq!(pin.id, "bge-small-en-v1.5-f32");
    assert_eq!(pin.id, manifest::default_model().id);
}

#[test]
fn resolve_accepts_the_explicit_qwen_comparison_model() {
    let pin = prepare::resolve(Some("qwen3-0.6b-f16")).expect("qwen3 pin");
    assert_eq!(pin.id, "qwen3-0.6b-f16");
}

#[test]
fn resolve_rejects_an_unknown_id_with_a_message_listing_known_ids() {
    let message = prepare::resolve(Some("not-a-model")).expect_err("unknown id");
    assert!(message.contains("not-a-model"), "{message}");
    for pin in manifest::manifest() {
        assert!(message.contains(pin.id), "{message}");
    }
}

#[test]
fn cache_dir_honors_the_environment_override() {
    let _guard = env_guard();
    let cache = tempfile::tempdir().expect("tempdir");
    let previous = std::env::var_os("JULIE_EMBEDDING_CACHE_DIR");
    std::env::set_var("JULIE_EMBEDDING_CACHE_DIR", cache.path());

    let resolved = prepare::cache_dir().expect("cache dir");

    match previous {
        Some(value) => std::env::set_var("JULIE_EMBEDDING_CACHE_DIR", value),
        None => std::env::remove_var("JULIE_EMBEDDING_CACHE_DIR"),
    }
    assert_eq!(resolved, cache.path());
}

#[test]
fn cache_dir_without_an_override_lands_under_the_home_cache() {
    let _guard = env_guard();
    let previous = std::env::var_os("JULIE_EMBEDDING_CACHE_DIR");
    std::env::remove_var("JULIE_EMBEDDING_CACHE_DIR");

    let resolved = prepare::cache_dir().expect("cache dir");

    if let Some(value) = previous {
        std::env::set_var("JULIE_EMBEDDING_CACHE_DIR", value);
    }
    assert!(resolved.ends_with("julie-semantic"), "{resolved:?}");
    #[cfg(not(windows))]
    assert!(
        resolved.starts_with(dirs::home_dir().expect("home").join(".cache")),
        "{resolved:?}"
    );
}

#[test]
fn unknown_model_exits_two_with_a_single_error_event_on_stdout() {
    let out = std::process::Command::new(BIN)
        .args(["prepare", "--model", "not-a-model"])
        .output()
        .expect("spawn");

    assert_eq!(out.status.code(), Some(2));
    let events = events(&out.stdout);
    assert_eq!(kinds(&events), vec!["error".to_string()]);
    assert_eq!(events[0]["model_id"], "not-a-model");
    for pin in manifest::manifest() {
        assert!(
            events[0]["message"]
                .as_str()
                .expect("message")
                .contains(pin.id),
            "{}",
            events[0]
        );
    }
}
