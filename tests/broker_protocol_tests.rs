#![cfg(unix)]

use julie_semantic_sidecar::broker::engine::BrokerEngine;
use julie_semantic_sidecar::broker::{serve_with_loader, BrokerConfig, BrokerEndpoint};
use julie_semantic_sidecar::engine_trait::{EmbedEngine, EmbedOutput, EngineError, Role};
use julie_semantic_sidecar::health::Limits;
use julie_semantic_sidecar::protocol::internal_error_reply;
use serde_json::{json, Value};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

const HELPER_ENV: &str = "JULIE_BROKER_PROTOCOL_TEST_HELPER";
const HELPER_ROOT_ENV: &str = "JULIE_BROKER_PROTOCOL_TEST_ROOT";
const HELPER_RELEASE_ENV: &str = "JULIE_BROKER_PROTOCOL_TEST_RELEASE";

struct BlockingEngine {
    release: PathBuf,
}

impl EmbedEngine for BlockingEngine {
    fn health_facts(&self) -> Result<Value, EngineError> {
        Ok(json!({"ready": true}))
    }

    fn embed(&self, texts: &[String], _role: Role) -> Result<EmbedOutput, EngineError> {
        while !self.release.exists() {
            thread::sleep(Duration::from_millis(5));
        }
        Ok(EmbedOutput {
            dims: 2,
            vectors: texts.iter().map(|_| vec![1.0, 0.0]).collect(),
        })
    }
}

#[test]
fn broker_protocol_process_helper() {
    if std::env::var_os(HELPER_ENV).is_none() {
        return;
    }
    let root = PathBuf::from(std::env::var_os(HELPER_ROOT_ENV).unwrap());
    let release = PathBuf::from(std::env::var_os(HELPER_RELEASE_ENV).unwrap());
    serve_with_loader(config(&root), move |_config, accelerator_lease| {
        assert!(accelerator_lease.is_some());
        Ok(BrokerEngine::new(
            BlockingEngine { release },
            accelerator_lease,
        ))
    })
    .unwrap();
}

#[test]
fn queue_full_uses_the_existing_internal_error_envelope() {
    let reply = internal_error_reply("full", "BrokerQueueFull", "broker queue is full").unwrap();
    let reply: Value = serde_json::from_str(&reply.line).unwrap();
    assert_eq!(reply["schema"], "julie.embedding.sidecar");
    assert_eq!(reply["version"], 1);
    assert_eq!(reply["request_id"], "full");
    assert_eq!(reply["error"]["code"], "internal_error");
    assert_eq!(
        reply["error"]["message"],
        "BrokerQueueFull: broker queue is full"
    );
}

#[test]
fn expired_work_uses_the_existing_internal_error_envelope() {
    let reply =
        internal_error_reply("expired", "BrokerRequestExpired", "broker request expired").unwrap();
    let reply: Value = serde_json::from_str(&reply.line).unwrap();
    assert_eq!(reply["request_id"], "expired");
    assert_eq!(reply["error"]["code"], "internal_error");
}

#[test]
fn saturated_shutdown_is_an_internal_error_without_stopping_the_connection() {
    let reply =
        internal_error_reply("shutdown", "BrokerQueueFull", "broker queue is full").unwrap();
    assert!(!reply.stop_connection);
    let reply: Value = serde_json::from_str(&reply.line).unwrap();
    assert_eq!(reply["request_id"], "shutdown");
    assert_eq!(reply["error"]["code"], "internal_error");
}

#[test]
fn queue_full_error_leaves_the_connection_usable() {
    let temp = TempDir::new().unwrap();
    let release = temp.path().join("release");
    let config = config(temp.path());
    let endpoint = endpoint_path(&config);
    let mut child = spawn_helper(temp.path(), &release);
    wait_for_path(&endpoint, Duration::from_secs(5));

    let mut streams: Vec<_> = (0..66)
        .map(|request_id| {
            let mut stream = UnixStream::connect(&endpoint).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_millis(20)))
                .unwrap();
            writeln!(
                stream,
                "{}",
                json!({
                    "schema": "julie.embedding.sidecar",
                    "version": 1,
                    "request_id": request_id.to_string(),
                    "method": "embed_query",
                    "params": {"text": "query"}
                })
            )
            .unwrap();
            stream.flush().unwrap();
            stream
        })
        .collect();

    let rejected = wait_for_queue_full(&streams, Duration::from_secs(5));
    OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&release)
        .unwrap();

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut attempt = 0;
    loop {
        streams[rejected]
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let response = request(
            &mut streams[rejected],
            &format!("after-full-{attempt}"),
            "health",
            json!({}),
        );
        if response["result"]["ready"] == true {
            break;
        }
        assert_eq!(response["error"]["code"], "internal_error");
        assert!(
            Instant::now() < deadline,
            "connection did not recover after saturation"
        );
        attempt += 1;
        thread::sleep(Duration::from_millis(10));
    }

    drop(child.stdin.take());
    wait_for_exit(&mut child, Duration::from_secs(5));
}

#[test]
fn oversized_line_then_health_uses_frozen_framing_and_keeps_connection_usable() {
    let temp = TempDir::new().unwrap();
    let release = temp.path().join("release");
    OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&release)
        .unwrap();
    let config = config(temp.path());
    let endpoint = endpoint_path(&config);
    let mut child = spawn_helper(temp.path(), &release);
    wait_for_path(&endpoint, Duration::from_secs(5));
    let mut stream = UnixStream::connect(&endpoint).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();

    let mut oversized = vec![b'x'; Limits::default().max_request_bytes + 1];
    oversized.push(b'\n');
    stream.write_all(&oversized).unwrap();
    stream.flush().unwrap();
    let mut line = String::new();
    BufReader::new(stream.try_clone().unwrap())
        .read_line(&mut line)
        .unwrap();
    let rejected: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(rejected["request_id"], "");
    assert_eq!(rejected["error"]["code"], "invalid_request");
    assert!(rejected["error"]["message"]
        .as_str()
        .unwrap()
        .contains("max_request_bytes"));

    let health = request(&mut stream, "after-oversized", "health", json!({}));
    assert_eq!(health["request_id"], "after-oversized");
    assert_eq!(health["result"]["ready"], true);

    drop(child.stdin.take());
    wait_for_exit(&mut child, Duration::from_secs(5));
}

fn config(root: &Path) -> BrokerConfig {
    BrokerConfig {
        model_id: "test-model".to_string(),
        endpoint: BrokerEndpoint::Unix(root.join("broker.sock")),
        service_lock: root.join("broker.lock"),
        accelerator_lock: root.join("accelerator.lock"),
    }
}

fn endpoint_path(config: &BrokerConfig) -> PathBuf {
    match &config.endpoint {
        BrokerEndpoint::Unix(path) => path.clone(),
    }
}

fn spawn_helper(root: &Path, release: &Path) -> Child {
    Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "broker_protocol_process_helper", "--nocapture"])
        .env(HELPER_ENV, "1")
        .env(HELPER_ROOT_ENV, root)
        .env(HELPER_RELEASE_ENV, release)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

fn wait_for_queue_full(streams: &[UnixStream], timeout: Duration) -> usize {
    let deadline = Instant::now() + timeout;
    loop {
        for (index, stream) in streams.iter().enumerate() {
            let mut line = String::new();
            match BufReader::new(stream.try_clone().unwrap()).read_line(&mut line) {
                Ok(0) => {}
                Ok(_) => {
                    let response: Value = serde_json::from_str(&line).unwrap();
                    if response["error"]["code"] == "internal_error"
                        && response["error"]["message"]
                            .as_str()
                            .unwrap()
                            .contains("broker queue is full")
                    {
                        return index;
                    }
                }
                Err(err) if matches!(err.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
                Err(err) => panic!("could not read broker response: {err}"),
            }
        }
        assert!(Instant::now() < deadline, "queue never saturated");
    }
}

fn request(stream: &mut UnixStream, request_id: &str, method: &str, params: Value) -> Value {
    writeln!(
        stream,
        "{}",
        json!({
            "schema": "julie.embedding.sidecar",
            "version": 1,
            "request_id": request_id,
            "method": method,
            "params": params
        })
    )
    .unwrap();
    stream.flush().unwrap();
    let mut line = String::new();
    BufReader::new(stream.try_clone().unwrap())
        .read_line(&mut line)
        .unwrap();
    serde_json::from_str(&line).unwrap()
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
