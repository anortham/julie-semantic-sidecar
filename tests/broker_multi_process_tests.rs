use julie_semantic_sidecar::broker::engine::BrokerEngine;
use julie_semantic_sidecar::broker::{serve_with_loader, BrokerConfig, BrokerEndpoint};
use julie_semantic_sidecar::engine_trait::{EmbedEngine, EmbedOutput, EngineError, Role};
use serde_json::{json, Value};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

const HELPER_ENV: &str = "JULIE_MULTI_PROCESS_HELPER";
const HELPER_ROLE_ENV: &str = "JULIE_MULTI_PROCESS_ROLE";
const HELPER_ROOT_ENV: &str = "JULIE_MULTI_PROCESS_ROOT";
const HELPER_MODEL_ENV: &str = "JULIE_MULTI_PROCESS_MODEL";
const HELPER_ENDPOINT_ENV: &str = "JULIE_MULTI_PROCESS_ENDPOINT";
const HELPER_SERVICE_LOCK_ENV: &str = "JULIE_MULTI_PROCESS_SERVICE_LOCK";
const HELPER_ACCELERATOR_LOCK_ENV: &str = "JULIE_MULTI_PROCESS_ACCELERATOR_LOCK";
const HELPER_LOAD_LOG_ENV: &str = "JULIE_MULTI_PROCESS_LOAD_LOG";
const HELPER_READY_FILE_ENV: &str = "JULIE_MULTI_PROCESS_READY_FILE";
const HELPER_TRIGGER_FILE_ENV: &str = "JULIE_MULTI_PROCESS_TRIGGER_FILE";
const HELPER_RESULT_FILE_ENV: &str = "JULIE_MULTI_PROCESS_RESULT_FILE";
const HELPER_METHOD_ENV: &str = "JULIE_MULTI_PROCESS_METHOD";
const HELPER_BLOCK_CLAIM_ENV: &str = "JULIE_MULTI_PROCESS_BLOCK_CLAIM";
const HELPER_REQUEST_STARTED_ENV: &str = "JULIE_MULTI_PROCESS_REQUEST_STARTED";
const HELPER_UNBLOCKED_FILE_ENV: &str = "JULIE_MULTI_PROCESS_UNBLOCKED_FILE";

#[derive(Debug)]
struct FakeEngine {
    model_id: String,
    accelerated: bool,
    block_claim: Option<PathBuf>,
    request_started: Option<PathBuf>,
}

impl EmbedEngine for FakeEngine {
    fn health_facts(&self) -> Result<Value, EngineError> {
        Ok(json!({
            "ready": true,
            "model": self.model_id,
            "dims": 2,
            "backend": if self.accelerated { "test-accelerator" } else { "cpu" },
            "accelerated": self.accelerated,
            "capabilities": {
                "protocol": "julie.embedding.sidecar",
                "version": 1
            }
        }))
    }

    fn is_accelerated(&self) -> bool {
        self.accelerated
    }

    fn embed(&self, texts: &[String], _role: Role) -> Result<EmbedOutput, EngineError> {
        if let (Some(claim), Some(started)) = (&self.block_claim, &self.request_started) {
            if OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(claim)
                .is_ok()
            {
                File::create(started).unwrap();
                loop {
                    thread::park();
                }
            }
        }
        Ok(EmbedOutput {
            dims: 2,
            vectors: texts.iter().map(|_| vec![1.0, 0.0]).collect(),
        })
    }
}

#[test]
fn broker_multi_process_helper() {
    if std::env::var_os(HELPER_ENV).is_none() {
        return;
    }

    let endpoint = std::env::var(HELPER_ENDPOINT_ENV).unwrap();
    match std::env::var(HELPER_ROLE_ENV).unwrap().as_str() {
        "broker" => run_broker_helper(endpoint),
        "client" => {
            let method = std::env::var(HELPER_METHOD_ENV).unwrap();
            request_embed(&endpoint, &method, Duration::from_secs(10)).unwrap();
        }
        "survivor" => run_survivor_helper(&endpoint),
        role => panic!("unknown helper role {role}"),
    }
}

#[test]
fn eight_same_model_clients_share_exactly_one_model_loaded_broker() {
    let result = run_same_model_scenario(8);

    assert_eq!(result.model_load_count, 1);
    assert_eq!(result.live_broker_count, 1);
    assert_eq!(result.loser_exit_count, 7);
    assert_eq!(result.concurrent_query_count, 4);
    assert_eq!(result.concurrent_batch_count, 4);
    assert_eq!(result.live_broker_count_after_cleanup, 0);
    assert_eq!(result.hung_requests, 0);
    assert_eq!(result.failed_requests, 0);
}

#[test]
fn old_and_new_models_use_distinct_endpoints_and_one_accelerator_lease() {
    let result = run_multi_model_scenario();

    assert_ne!(result.old_endpoint, result.new_endpoint);
    assert_eq!(result.model_load_count, 2);
    assert!(result.accelerated_broker_count <= 1);
    assert_eq!(result.old_reported_model, "old-model");
    assert_eq!(result.new_reported_model, "new-model");
    assert_eq!(result.live_broker_count_after_cleanup, 0);
}

#[test]
fn killed_owner_during_in_flight_embed_is_replaced_by_client_simulation_within_thirty_seconds() {
    let result = run_recovery_scenario(Duration::from_secs(30));

    assert!(result.recovery_time <= Duration::from_secs(30));
    assert_eq!(result.model_load_count, 2);
    assert!(result.in_flight_request_unblocked);
    assert!(result.replacement_spawned_by_client);
    assert_eq!(result.live_broker_count_after_cleanup, 0);
    assert_eq!(result.hung_requests, 0);
    assert_eq!(result.failed_requests, 0);
}

fn run_broker_helper(endpoint: String) {
    let model_id = std::env::var(HELPER_MODEL_ENV).unwrap();
    let config = BrokerConfig {
        model_id: model_id.clone(),
        endpoint: broker_endpoint(endpoint),
        service_lock: PathBuf::from(std::env::var_os(HELPER_SERVICE_LOCK_ENV).unwrap()),
        accelerator_lock: PathBuf::from(std::env::var_os(HELPER_ACCELERATOR_LOCK_ENV).unwrap()),
    };
    let load_log = PathBuf::from(std::env::var_os(HELPER_LOAD_LOG_ENV).unwrap());
    let block_claim = std::env::var_os(HELPER_BLOCK_CLAIM_ENV).map(PathBuf::from);
    let request_started = std::env::var_os(HELPER_REQUEST_STARTED_ENV).map(PathBuf::from);
    serve_with_loader(config, move |_config, accelerator_lease| {
        let accelerated = accelerator_lease.is_some();
        writeln!(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&load_log)?,
            "{model_id},{},{accelerated}",
            std::process::id()
        )?;
        Ok(BrokerEngine::new(
            FakeEngine {
                model_id,
                accelerated,
                block_claim,
                request_started,
            },
            accelerator_lease,
        ))
    })
    .unwrap();
}

fn run_survivor_helper(endpoint: &str) {
    request_health(endpoint, Duration::from_secs(10)).unwrap();
    let ready = PathBuf::from(std::env::var_os(HELPER_READY_FILE_ENV).unwrap());
    let trigger = PathBuf::from(std::env::var_os(HELPER_TRIGGER_FILE_ENV).unwrap());
    let result = PathBuf::from(std::env::var_os(HELPER_RESULT_FILE_ENV).unwrap());
    let unblocked = PathBuf::from(std::env::var_os(HELPER_UNBLOCKED_FILE_ENV).unwrap());
    File::create(ready).unwrap();
    wait_for_path(&trigger, Duration::from_secs(10));
    assert!(request_embed(endpoint, "embed_batch", Duration::from_secs(30)).is_err());
    File::create(unblocked).unwrap();
    request_embed(endpoint, "embed_query", Duration::from_secs(30)).unwrap();
    File::create(result).unwrap();
}

fn run_same_model_scenario(client_count: usize) -> SameModelResult {
    let temp = TempDir::new().unwrap();
    let endpoint = unique_endpoint(temp.path(), "same-model");
    let service_lock = temp.path().join("same-model.lock");
    let accelerator_lock = temp.path().join("accelerator.lock");
    let load_log = temp.path().join("loads");
    let mut brokers: Vec<_> = (0..client_count)
        .map(|_| {
            spawn_broker(
                temp.path(),
                "same-model",
                &endpoint,
                &service_lock,
                &accelerator_lock,
                &load_log,
                None,
            )
        })
        .collect();
    wait_for_load_count(&load_log, 1, Duration::from_secs(10));
    wait_for_health(&endpoint, Duration::from_secs(10));
    let live_broker_count = wait_for_live_count(&mut brokers, 1, Duration::from_secs(10));
    let loser_exit_count = brokers
        .iter_mut()
        .filter_map(|broker| broker.try_wait().unwrap())
        .count();
    let mut query_clients: Vec<_> = (0..client_count.div_ceil(2))
        .map(|_| spawn_client(temp.path(), &endpoint, "embed_query"))
        .collect();
    let mut batch_clients: Vec<_> = (0..client_count / 2)
        .map(|_| spawn_client(temp.path(), &endpoint, "embed_batch"))
        .collect();
    let query_outcome = wait_for_clients(&mut query_clients, Duration::from_secs(15));
    let batch_outcome = wait_for_clients(&mut batch_clients, Duration::from_secs(15));
    let model_load_count = load_lines(&load_log).len();
    close_live_brokers(&mut brokers);
    let live_broker_count_after_cleanup = current_live_count(&mut brokers);
    SameModelResult {
        model_load_count,
        live_broker_count,
        loser_exit_count,
        concurrent_query_count: query_outcome.successful,
        concurrent_batch_count: batch_outcome.successful,
        live_broker_count_after_cleanup,
        hung_requests: query_outcome.hung + batch_outcome.hung,
        failed_requests: query_outcome.failed + batch_outcome.failed,
    }
}

fn run_multi_model_scenario() -> MultiModelResult {
    let temp = TempDir::new().unwrap();
    let old_endpoint = unique_endpoint(temp.path(), "old");
    let new_endpoint = unique_endpoint(temp.path(), "new");
    let accelerator_lock = temp.path().join("accelerator.lock");
    let load_log = temp.path().join("loads");
    let mut old = spawn_broker(
        temp.path(),
        "old-model",
        &old_endpoint,
        &temp.path().join("old.lock"),
        &accelerator_lock,
        &load_log,
        None,
    );
    let mut new = spawn_broker(
        temp.path(),
        "new-model",
        &new_endpoint,
        &temp.path().join("new.lock"),
        &accelerator_lock,
        &load_log,
        None,
    );
    wait_for_load_count(&load_log, 2, Duration::from_secs(10));
    let old_health = request_health(&old_endpoint, Duration::from_secs(10)).unwrap();
    let new_health = request_health(&new_endpoint, Duration::from_secs(10)).unwrap();
    let lines = load_lines(&load_log);
    let accelerated_broker_count = lines.iter().filter(|line| line.ends_with(",true")).count();
    close_owner_and_wait(&mut old);
    close_owner_and_wait(&mut new);
    let live_broker_count_after_cleanup = current_live_count(std::slice::from_mut(&mut old))
        + current_live_count(std::slice::from_mut(&mut new));
    MultiModelResult {
        old_endpoint,
        new_endpoint,
        model_load_count: lines.len(),
        accelerated_broker_count,
        old_reported_model: old_health["result"]["model"].as_str().unwrap().to_string(),
        new_reported_model: new_health["result"]["model"].as_str().unwrap().to_string(),
        live_broker_count_after_cleanup,
    }
}

fn run_recovery_scenario(timeout: Duration) -> RecoveryResult {
    let temp = TempDir::new().unwrap();
    let endpoint = unique_endpoint(temp.path(), "recovery");
    let service_lock = temp.path().join("recovery.lock");
    let accelerator_lock = temp.path().join("accelerator.lock");
    let load_log = temp.path().join("loads");
    let ready = temp.path().join("survivor.ready");
    let trigger = temp.path().join("survivor.trigger");
    let result = temp.path().join("survivor.result");
    let unblocked = temp.path().join("survivor.unblocked");
    let block_claim = temp.path().join("block.claimed");
    let request_started = temp.path().join("request.started");
    let mut owner = spawn_broker(
        temp.path(),
        "recovery-model",
        &endpoint,
        &service_lock,
        &accelerator_lock,
        &load_log,
        Some((&block_claim, &request_started)),
    );
    wait_for_health(&endpoint, Duration::from_secs(10));
    let mut survivor = spawn_survivor(
        temp.path(),
        &endpoint,
        &ready,
        &trigger,
        &result,
        &unblocked,
    );
    wait_for_path(&ready, Duration::from_secs(10));

    File::create(&trigger).unwrap();
    wait_for_path(&request_started, Duration::from_secs(10));
    let started = Instant::now();
    owner.kill().unwrap();
    wait_for_any_exit(&mut owner, Duration::from_secs(10));
    let mut replacement = spawn_broker(
        temp.path(),
        "recovery-model",
        &endpoint,
        &service_lock,
        &accelerator_lock,
        &load_log,
        Some((&block_claim, &request_started)),
    );
    wait_for_path(&result, timeout);
    let survivor_outcome = wait_for_child(&mut survivor, timeout);
    let recovery_time = started.elapsed();
    wait_for_load_count(&load_log, 2, Duration::from_secs(10));
    close_owner_and_wait(&mut replacement);
    let live_broker_count_after_cleanup =
        current_live_count(std::slice::from_mut(&mut replacement));
    let model_load_count = load_lines(&load_log).len();
    RecoveryResult {
        recovery_time,
        model_load_count,
        in_flight_request_unblocked: unblocked.exists(),
        replacement_spawned_by_client: model_load_count == 2,
        live_broker_count_after_cleanup,
        hung_requests: survivor_outcome.hung,
        failed_requests: survivor_outcome.failed,
    }
}

#[derive(Debug)]
struct SameModelResult {
    model_load_count: usize,
    live_broker_count: usize,
    loser_exit_count: usize,
    concurrent_query_count: usize,
    concurrent_batch_count: usize,
    live_broker_count_after_cleanup: usize,
    hung_requests: usize,
    failed_requests: usize,
}

#[derive(Debug)]
struct MultiModelResult {
    old_endpoint: String,
    new_endpoint: String,
    model_load_count: usize,
    accelerated_broker_count: usize,
    old_reported_model: String,
    new_reported_model: String,
    live_broker_count_after_cleanup: usize,
}

#[derive(Debug)]
struct RecoveryResult {
    recovery_time: Duration,
    model_load_count: usize,
    in_flight_request_unblocked: bool,
    replacement_spawned_by_client: bool,
    live_broker_count_after_cleanup: usize,
    hung_requests: usize,
    failed_requests: usize,
}

#[derive(Debug, Default)]
struct ClientOutcome {
    successful: usize,
    failed: usize,
    hung: usize,
}

fn spawn_broker(
    root: &Path,
    model: &str,
    endpoint: &str,
    service_lock: &Path,
    accelerator_lock: &Path,
    load_log: &Path,
    blocked_request: Option<(&Path, &Path)>,
) -> Child {
    let mut command = helper_command(root, "broker", endpoint);
    command
        .env(HELPER_MODEL_ENV, model)
        .env(HELPER_SERVICE_LOCK_ENV, service_lock)
        .env(HELPER_ACCELERATOR_LOCK_ENV, accelerator_lock)
        .env(HELPER_LOAD_LOG_ENV, load_log)
        .stdin(Stdio::piped());
    if let Some((block_claim, request_started)) = blocked_request {
        command
            .env(HELPER_BLOCK_CLAIM_ENV, block_claim)
            .env(HELPER_REQUEST_STARTED_ENV, request_started);
    }
    command.spawn().unwrap()
}

fn spawn_client(root: &Path, endpoint: &str, method: &str) -> Child {
    helper_command(root, "client", endpoint)
        .env(HELPER_METHOD_ENV, method)
        .spawn()
        .unwrap()
}

fn spawn_survivor(
    root: &Path,
    endpoint: &str,
    ready: &Path,
    trigger: &Path,
    result: &Path,
    unblocked: &Path,
) -> Child {
    helper_command(root, "survivor", endpoint)
        .env(HELPER_READY_FILE_ENV, ready)
        .env(HELPER_TRIGGER_FILE_ENV, trigger)
        .env(HELPER_RESULT_FILE_ENV, result)
        .env(HELPER_UNBLOCKED_FILE_ENV, unblocked)
        .spawn()
        .unwrap()
}

fn helper_command(root: &Path, role: &str, endpoint: &str) -> Command {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", "broker_multi_process_helper", "--nocapture"])
        .env(HELPER_ENV, "1")
        .env(HELPER_ROLE_ENV, role)
        .env(HELPER_ROOT_ENV, root)
        .env(HELPER_ENDPOINT_ENV, endpoint)
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    command
}

fn wait_for_clients(clients: &mut [Child], timeout: Duration) -> ClientOutcome {
    let deadline = Instant::now() + timeout;
    let mut outcome = ClientOutcome::default();
    for child in clients {
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                if status.success() {
                    outcome.successful += 1;
                } else {
                    outcome.failed += 1;
                }
                break;
            }
            if Instant::now() >= deadline {
                outcome.hung += 1;
                child.kill().unwrap();
                wait_for_any_exit(child, Duration::from_secs(5));
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
    }
    outcome
}

fn wait_for_child(child: &mut Child, timeout: Duration) -> ClientOutcome {
    wait_for_clients(std::slice::from_mut(child), timeout)
}

fn wait_for_live_count(children: &mut [Child], count: usize, timeout: Duration) -> usize {
    let deadline = Instant::now() + timeout;
    loop {
        let mut live = 0;
        for child in children.iter_mut() {
            if child.try_wait().unwrap().is_none() {
                live += 1;
            }
        }
        if live == count {
            return live;
        }
        assert!(
            Instant::now() < deadline,
            "live broker count did not converge"
        );
        thread::sleep(Duration::from_millis(20));
    }
}

fn current_live_count(children: &mut [Child]) -> usize {
    children
        .iter_mut()
        .map(|child| usize::from(child.try_wait().unwrap().is_none()))
        .sum()
}

fn close_live_brokers(children: &mut [Child]) {
    for child in children {
        if child.try_wait().unwrap().is_none() {
            close_owner_and_wait(child);
        }
    }
}

fn close_owner_and_wait(child: &mut Child) {
    drop(child.stdin.take());
    wait_for_success(child, Duration::from_secs(10));
}

fn wait_for_success(child: &mut Child, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            if !status.success() {
                let stderr = child
                    .stderr
                    .take()
                    .map(|mut stderr| {
                        let mut text = String::new();
                        stderr.read_to_string(&mut text).unwrap();
                        text
                    })
                    .unwrap_or_default();
                panic!("helper exited with {status}: {stderr}");
            }
            return;
        }
        assert!(Instant::now() < deadline, "helper did not exit");
        thread::sleep(Duration::from_millis(20));
    }
}

fn wait_for_any_exit(child: &mut Child, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        if child.try_wait().unwrap().is_some() {
            return;
        }
        assert!(Instant::now() < deadline, "killed helper did not exit");
        thread::sleep(Duration::from_millis(20));
    }
}

fn wait_for_health(endpoint: &str, timeout: Duration) {
    request_health(endpoint, timeout).unwrap();
}

fn request_health(endpoint: &str, timeout: Duration) -> std::io::Result<Value> {
    request(
        endpoint,
        "health",
        json!({}),
        "multi-process-health",
        timeout,
    )
}

fn request_embed(endpoint: &str, method: &str, timeout: Duration) -> std::io::Result<Value> {
    let params = match method {
        "embed_query" => json!({ "text": "concurrent query" }),
        "embed_batch" => json!({ "texts": ["concurrent document one", "concurrent document two"] }),
        _ => return Err(std::io::Error::other("unsupported test embed method")),
    };
    request(endpoint, method, params, "multi-process-embed", timeout)
}

fn request(
    endpoint: &str,
    method: &str,
    params: Value,
    request_id: &str,
    timeout: Duration,
) -> std::io::Result<Value> {
    let mut stream = open_client(endpoint, timeout)?;
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
    )?;
    stream.flush()?;
    let mut line = String::new();
    BufReader::new(&mut stream).read_line(&mut line)?;
    serde_json::from_str(&line).map_err(std::io::Error::other)
}

fn wait_for_load_count(path: &Path, count: usize, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        if load_lines(path).len() == count {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "model load count did not reach {count}"
        );
        thread::sleep(Duration::from_millis(20));
    }
}

fn load_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(str::to_string)
        .collect()
}

fn wait_for_path(path: &Path, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while !path.exists() {
        assert!(
            Instant::now() < deadline,
            "path did not appear: {}",
            path.display()
        );
        thread::sleep(Duration::from_millis(20));
    }
}

trait ClientStream: Read + Write {}
impl<T: Read + Write> ClientStream for T {}

#[cfg(unix)]
fn broker_endpoint(endpoint: String) -> BrokerEndpoint {
    BrokerEndpoint::Unix(PathBuf::from(endpoint))
}

#[cfg(windows)]
fn broker_endpoint(endpoint: String) -> BrokerEndpoint {
    BrokerEndpoint::Windows(endpoint)
}

#[cfg(unix)]
fn unique_endpoint(root: &Path, label: &str) -> String {
    root.join(format!("{label}.sock"))
        .to_string_lossy()
        .into_owned()
}

#[cfg(windows)]
fn unique_endpoint(_root: &Path, label: &str) -> String {
    format!(
        r"\\.\pipe\julie-semantic-sidecar-multi-{label}-{}",
        std::process::id()
    )
}

#[cfg(unix)]
fn open_client(endpoint: &str, timeout: Duration) -> std::io::Result<Box<dyn ClientStream>> {
    let deadline = Instant::now() + timeout;
    loop {
        match UnixStream::connect(endpoint) {
            Ok(stream) => {
                stream.set_read_timeout(Some(timeout))?;
                stream.set_write_timeout(Some(timeout))?;
                return Ok(Box::new(stream));
            }
            Err(error) if Instant::now() < deadline => {
                let _ = error;
                thread::sleep(Duration::from_millis(20));
            }
            Err(error) => return Err(error),
        }
    }
}

#[cfg(windows)]
fn open_client(endpoint: &str, timeout: Duration) -> std::io::Result<Box<dyn ClientStream>> {
    let deadline = Instant::now() + timeout;
    loop {
        match OpenOptions::new().read(true).write(true).open(endpoint) {
            Ok(file) => return Ok(Box::new(file)),
            Err(error) if Instant::now() < deadline => {
                let _ = error;
                thread::sleep(Duration::from_millis(20));
            }
            Err(error) => return Err(error),
        }
    }
}
