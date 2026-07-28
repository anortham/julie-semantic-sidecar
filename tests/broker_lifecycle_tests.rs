#![cfg(unix)]

use fs4::FileExt;
use julie_semantic_sidecar::broker::engine::BrokerEngine;
use julie_semantic_sidecar::broker::{serve_with_loader, BrokerConfig, BrokerEndpoint};
use julie_semantic_sidecar::engine_trait::{EmbedEngine, EmbedOutput, EngineError, Role};
use serde_json::{json, Value};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

const HELPER_ENV: &str = "JULIE_BROKER_TEST_HELPER";
const HELPER_CONFIG_ENV: &str = "JULIE_BROKER_TEST_CONFIG";
const HELPER_MODE_ENV: &str = "JULIE_BROKER_TEST_MODE";
const HELPER_LOAD_LOG_ENV: &str = "JULIE_BROKER_TEST_LOAD_LOG";

#[derive(Debug)]
struct FakeEngine;

impl EmbedEngine for FakeEngine {
    fn health_facts(&self) -> Result<Value, EngineError> {
        Ok(json!({
            "ready": true,
            "model": "test-model",
            "dims": 2,
            "backend": "cpu",
            "capabilities": {
                "protocol": "julie.embedding.sidecar",
                "version": 1
            }
        }))
    }

    fn embed(&self, texts: &[String], _role: Role) -> Result<EmbedOutput, EngineError> {
        Ok(EmbedOutput {
            dims: 2,
            vectors: texts.iter().map(|_| vec![1.0, 0.0]).collect(),
        })
    }
}

#[test]
fn broker_process_helper() {
    if std::env::var_os(HELPER_ENV).is_none() {
        return;
    }

    let root = PathBuf::from(std::env::var_os(HELPER_CONFIG_ENV).unwrap());
    let config = config(&root);
    let mode = std::env::var(HELPER_MODE_ENV).unwrap_or_else(|_| "ready".to_string());
    let load_log = PathBuf::from(std::env::var_os(HELPER_LOAD_LOG_ENV).unwrap());
    serve_with_loader(config, move |_config, accelerator_lease| {
        assert!(accelerator_lease.is_some());
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&load_log)?
            .write_all(b"load\n")?;
        if mode == "blocking" {
            loop {
                thread::park();
            }
        }
        Ok(BrokerEngine::new(FakeEngine, accelerator_lease))
    })
    .unwrap();
}

#[test]
fn concurrent_broker_starts_load_one_engine_and_losers_exit() {
    let temp = TempDir::new().unwrap();
    let load_log = temp.path().join("loads");
    let mut children: Vec<Child> = (0..8)
        .map(|_| spawn_helper(temp.path(), &load_log, "ready"))
        .collect();
    wait_for_path(&config(temp.path()).endpoint_path(), Duration::from_secs(5));
    wait_for_load_count(&load_log, 1, Duration::from_secs(5));

    let live = wait_for_one_live_child(&mut children, Duration::from_secs(5));
    assert_eq!(load_count(&load_log), 1);
    for (index, child) in children.iter_mut().enumerate() {
        if index != live {
            assert!(child.try_wait().unwrap().is_some());
        }
    }

    close_owner_and_wait(&mut children[live]);
}

#[test]
fn owner_stdin_eof_removes_socket_and_releases_lock() {
    let temp = TempDir::new().unwrap();
    let load_log = temp.path().join("loads");
    let mut child = spawn_helper(temp.path(), &load_log, "ready");
    let config = config(temp.path());
    wait_for_path(&config.endpoint_path(), Duration::from_secs(5));
    assert_eq!(
        std::fs::metadata(temp.path()).unwrap().permissions().mode() & 0o777,
        0o700
    );
    assert_eq!(
        std::fs::metadata(config.endpoint_path())
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );

    close_owner_and_wait(&mut child);

    assert!(!config.endpoint_path().exists());
    assert_lock_is_free(&config.service_lock);
    assert_lock_is_free(&config.accelerator_lock);
}

#[test]
fn stale_socket_is_unlinked_only_after_service_lock_acquisition() {
    let temp = TempDir::new().unwrap();
    let load_log = temp.path().join("loads");
    let config = config(temp.path());
    let mut first_owner = spawn_helper(temp.path(), &load_log, "ready");
    wait_for_path(&config.endpoint_path(), Duration::from_secs(5));
    wait_for_load_count(&load_log, 1, Duration::from_secs(5));
    first_owner.kill().unwrap();
    wait_for_any_exit(&mut first_owner, Duration::from_secs(5));
    assert!(config.endpoint_path().exists());
    assert_lock_is_free(&config.service_lock);
    assert_lock_is_free(&config.accelerator_lock);

    let held_lock = lock_exclusive(&config.service_lock);

    let mut loser = spawn_helper(temp.path(), &load_log, "ready");
    wait_for_exit(&mut loser, Duration::from_secs(5));
    assert!(config.endpoint_path().exists());
    assert_eq!(load_count(&load_log), 1);

    drop(held_lock);
    let mut winner = spawn_helper(temp.path(), &load_log, "ready");
    wait_for_load_count(&load_log, 2, Duration::from_secs(5));
    wait_for_health(&config.endpoint_path(), Duration::from_secs(5));
    close_owner_and_wait(&mut winner);
}

#[test]
fn shutdown_response_closes_only_its_connection() {
    let temp = TempDir::new().unwrap();
    let load_log = temp.path().join("loads");
    let mut child = spawn_helper(temp.path(), &load_log, "ready");
    let config = config(temp.path());
    let endpoint = config.endpoint_path();
    wait_for_path(&endpoint, Duration::from_secs(5));
    let mut first = UnixStream::connect(&endpoint).unwrap();
    let mut second = UnixStream::connect(&endpoint).unwrap();

    let shutdown = request(&mut first, "one", "shutdown", json!({}));
    assert_eq!(shutdown["request_id"], "one");
    let mut closed = String::new();
    assert_eq!(BufReader::new(first).read_line(&mut closed).unwrap(), 0);

    let health = request(&mut second, "two", "health", json!({}));
    assert_eq!(health["request_id"], "two");
    assert_eq!(health["result"]["ready"], true);
    assert!(child.try_wait().unwrap().is_none());
    assert_lock_is_held(&config.service_lock);
    assert_lock_is_held(&config.accelerator_lock);

    close_owner_and_wait(&mut child);
}

#[test]
fn owner_eof_during_model_load_terminates_before_endpoint_bind() {
    let temp = TempDir::new().unwrap();
    let load_log = temp.path().join("loads");
    let config = config(temp.path());
    let mut child = spawn_helper(temp.path(), &load_log, "blocking");
    wait_for_load_count(&load_log, 1, Duration::from_secs(5));

    close_owner_and_wait(&mut child);

    assert!(!config.endpoint_path().exists());
    assert_lock_is_free(&config.service_lock);
    assert_lock_is_free(&config.accelerator_lock);
}

fn config(root: &Path) -> BrokerConfig {
    BrokerConfig {
        model_id: "test-model".to_string(),
        endpoint: BrokerEndpoint::Unix(root.join("broker.sock")),
        service_lock: root.join("broker.lock"),
        accelerator_lock: root.join("accelerator.lock"),
    }
}

trait EndpointPath {
    fn endpoint_path(&self) -> PathBuf;
}

impl EndpointPath for BrokerConfig {
    fn endpoint_path(&self) -> PathBuf {
        match &self.endpoint {
            BrokerEndpoint::Unix(path) => path.clone(),
        }
    }
}

fn spawn_helper(root: &Path, load_log: &Path, mode: &str) -> Child {
    Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "broker_process_helper", "--nocapture"])
        .env(HELPER_ENV, "1")
        .env(HELPER_CONFIG_ENV, root)
        .env(HELPER_MODE_ENV, mode)
        .env(HELPER_LOAD_LOG_ENV, load_log)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

fn wait_for_one_live_child(children: &mut [Child], timeout: Duration) -> usize {
    let deadline = Instant::now() + timeout;
    loop {
        let live: Vec<_> = children
            .iter_mut()
            .enumerate()
            .filter_map(|(index, child)| child.try_wait().unwrap().is_none().then_some(index))
            .collect();
        if live.len() == 1 {
            return live[0];
        }
        assert!(
            Instant::now() < deadline,
            "expected exactly one live broker"
        );
        thread::sleep(Duration::from_millis(20));
    }
}

fn close_owner_and_wait(child: &mut Child) {
    drop(child.stdin.take());
    wait_for_exit(child, Duration::from_secs(5));
}

fn wait_for_exit(child: &mut Child, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "broker exited with {status}");
            return;
        }
        assert!(Instant::now() < deadline, "broker did not exit");
        thread::sleep(Duration::from_millis(20));
    }
}

fn wait_for_any_exit(child: &mut Child, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        if child.try_wait().unwrap().is_some() {
            return;
        }
        assert!(Instant::now() < deadline, "broker did not exit");
        thread::sleep(Duration::from_millis(20));
    }
}

fn wait_for_path(path: &Path, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while !path.exists() {
        assert!(
            Instant::now() < deadline,
            "{} was not created",
            path.display()
        );
        thread::sleep(Duration::from_millis(20));
    }
}

fn wait_for_health(path: &Path, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(mut stream) = UnixStream::connect(path) {
            let response = request(&mut stream, "health", "health", json!({}));
            assert_eq!(response["result"]["ready"], true);
            return;
        }
        assert!(Instant::now() < deadline, "broker never became healthy");
        thread::sleep(Duration::from_millis(20));
    }
}

fn request(stream: &mut UnixStream, request_id: &str, method: &str, params: Value) -> Value {
    let line = json!({
        "schema": "julie.embedding.sidecar",
        "version": 1,
        "request_id": request_id,
        "method": method,
        "params": params
    });
    writeln!(stream, "{line}").unwrap();
    stream.flush().unwrap();
    let mut response = String::new();
    BufReader::new(stream.try_clone().unwrap())
        .read_line(&mut response)
        .unwrap();
    serde_json::from_str(&response).unwrap()
}

fn wait_for_load_count(path: &Path, expected: usize, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while load_count(path) != expected {
        assert!(
            Instant::now() < deadline,
            "expected {expected} loads, got {}",
            load_count(path)
        );
        thread::sleep(Duration::from_millis(20));
    }
}

fn load_count(path: &Path) -> usize {
    std::fs::read_to_string(path)
        .map(|contents| contents.lines().count())
        .unwrap_or(0)
}

fn lock_exclusive(path: &Path) -> File {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .unwrap();
    FileExt::lock(&file).unwrap();
    file
}

fn assert_lock_is_free(path: &Path) {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .unwrap();
    FileExt::try_lock(&file).unwrap();
}

fn assert_lock_is_held(path: &Path) {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .unwrap();
    assert!(matches!(
        FileExt::try_lock(&file),
        Err(fs4::TryLockError::WouldBlock)
    ));
}
