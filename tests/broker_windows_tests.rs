#![cfg(windows)]

use julie_semantic_sidecar::broker::engine::BrokerEngine;
use julie_semantic_sidecar::broker::transport::windows::Listener;
use julie_semantic_sidecar::broker::{serve_with_loader, BrokerConfig, BrokerEndpoint};
use julie_semantic_sidecar::engine_trait::{EmbedEngine, EmbedOutput, EngineError, Role};
use serde_json::{json, Value};
use std::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::mem::size_of;
use std::os::windows::io::AsRawHandle;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tempfile::TempDir;
use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_BROKEN_PIPE, ERROR_PIPE_NOT_CONNECTED, HANDLE,
};
use windows_sys::Win32::Security::{
    EqualSid, GetAce, GetKernelObjectSecurity, GetSecurityDescriptorDacl, GetTokenInformation,
    TokenUser, ACCESS_ALLOWED_ACE, ACL, DACL_SECURITY_INFORMATION, TOKEN_QUERY, TOKEN_USER,
};
use windows_sys::Win32::Storage::FileSystem::FILE_ALL_ACCESS;
use windows_sys::Win32::System::Pipes::WaitNamedPipeW;
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

const HELPER_ENV: &str = "JULIE_WINDOWS_BROKER_TEST_HELPER";
const HELPER_ROOT_ENV: &str = "JULIE_WINDOWS_BROKER_TEST_ROOT";
const HELPER_ENDPOINT_ENV: &str = "JULIE_WINDOWS_BROKER_TEST_ENDPOINT";

static ENDPOINT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

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
fn windows_broker_process_helper() {
    if std::env::var_os(HELPER_ENV).is_none() {
        return;
    }

    let root = PathBuf::from(std::env::var_os(HELPER_ROOT_ENV).unwrap());
    let endpoint = std::env::var(HELPER_ENDPOINT_ENV).unwrap();
    serve_with_loader(config(&root, endpoint), |_config, accelerator_lease| {
        assert!(accelerator_lease.is_some());
        Ok(BrokerEngine::new(FakeEngine, accelerator_lease))
    })
    .unwrap();
}

#[test]
fn cancelled_connect_releases_the_pipe_instance_within_one_second() {
    let endpoint = unique_endpoint("cancel-connect");
    let bound = AtomicBool::new(false);
    let listener = Listener::bind(&endpoint, &bound).unwrap();
    let accepter = listener.clone();
    let started = Instant::now();
    let join = thread::spawn(move || match accepter.accept() {
        Ok(_) => panic!("cancelled accept unexpectedly connected"),
        Err(err) => err,
    });
    thread::sleep(Duration::from_millis(50));

    listener.cancel_pending().unwrap();

    assert!(join.join().unwrap().kind() == std::io::ErrorKind::Interrupted);
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[test]
fn cancelled_read_releases_the_pipe_instance_within_one_second() {
    let (_listener, connection, _client) = connected_pair("cancel-read");
    let mut reader = connection.clone();
    let started = Instant::now();
    let join = thread::spawn(move || {
        let mut buffer = [0_u8; 64];
        reader.read(&mut buffer).unwrap_err()
    });
    thread::sleep(Duration::from_millis(50));

    connection.cancel_io().unwrap();

    assert!(join.join().unwrap().kind() == std::io::ErrorKind::Interrupted);
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[test]
fn cancelled_write_releases_the_pipe_instance_within_one_second() {
    let (_listener, connection, _client) = connected_pair("cancel-write");
    let mut writer = connection.clone();
    let started = Instant::now();
    let join = thread::spawn(move || {
        let buffer = vec![b'x'; 8 * 1024 * 1024];
        writer.write(&buffer).unwrap_err()
    });
    thread::sleep(Duration::from_millis(50));

    connection.cancel_io().unwrap();

    assert!(join.join().unwrap().kind() == std::io::ErrorKind::Interrupted);
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[test]
fn buffered_read_and_write_return_the_exact_immediate_byte_counts() {
    let (_listener, mut connection, mut client) = connected_pair("immediate-bytes");
    client.write_all(b"ready").unwrap();
    client.flush().unwrap();

    let mut request = [0_u8; 5];
    connection.read_exact(&mut request).unwrap();
    assert_eq!(&request, b"ready");

    connection.write_all(b"reply").unwrap();
    connection.flush().unwrap();
    let mut response = [0_u8; 5];
    client.read_exact(&mut response).unwrap();
    assert_eq!(&response, b"reply");
}

#[test]
fn pipe_acl_contains_only_the_current_process_user() {
    let (_listener, connection, _client) = connected_pair("current-user-acl");
    let handle = connection.as_raw_handle() as HANDLE;
    let mut needed = 0_u32;
    unsafe {
        GetKernelObjectSecurity(
            handle,
            DACL_SECURITY_INFORMATION,
            null_mut(),
            0,
            &mut needed,
        );
    }
    assert!(needed > 0);
    let mut descriptor = vec![0_u8; needed as usize];
    assert_ne!(
        unsafe {
            GetKernelObjectSecurity(
                handle,
                DACL_SECURITY_INFORMATION,
                descriptor.as_mut_ptr().cast(),
                needed,
                &mut needed,
            )
        },
        0
    );
    let mut present = 0;
    let mut defaulted = 0;
    let mut acl: *mut ACL = null_mut();
    assert_ne!(
        unsafe {
            GetSecurityDescriptorDacl(
                descriptor.as_mut_ptr().cast(),
                &mut present,
                &mut acl,
                &mut defaulted,
            )
        },
        0
    );
    assert_ne!(present, 0);
    assert!(!acl.is_null());
    assert_eq!(unsafe { (*acl).AceCount }, 1);
    let mut ace: *mut c_void = null_mut();
    assert_ne!(unsafe { GetAce(acl, 0, &mut ace) }, 0);
    let allowed = ace.cast::<ACCESS_ALLOWED_ACE>();
    assert_eq!(unsafe { (*allowed).Mask }, FILE_ALL_ACCESS);
    let ace_sid = unsafe { std::ptr::addr_of_mut!((*allowed).SidStart).cast() };
    let token_user = current_token_user();
    assert_ne!(unsafe { EqualSid(ace_sid, token_user.sid()) }, 0);
}

#[test]
fn remote_clients_are_rejected_while_local_current_user_connects() {
    let endpoint = unique_endpoint("reject-remote");
    let bound = AtomicBool::new(false);
    let listener = Listener::bind(&endpoint, &bound).unwrap();
    let accepter = listener.clone();
    let join = thread::spawn(move || accepter.accept().unwrap());

    let remote = endpoint.replacen(r"\\.\pipe\", r"\\localhost\pipe\", 1);
    assert!(OpenOptions::new()
        .read(true)
        .write(true)
        .open(remote)
        .is_err());

    let client = open_client(&endpoint, Duration::from_secs(1));
    let _connection = join.join().unwrap();
    drop(client);
}

#[test]
fn killed_mid_line_client_does_not_wedge_other_clients() {
    let temp = TempDir::new().unwrap();
    let endpoint = unique_endpoint("three-clients");
    let mut child = spawn_helper(temp.path(), &endpoint);
    let mut first = open_client(&endpoint, Duration::from_secs(5));
    let mut second = open_client(&endpoint, Duration::from_secs(5));
    let mut killed = open_client(&endpoint, Duration::from_secs(5));

    killed
        .write_all(br#"{"schema":"julie.embedding.sidecar""#)
        .unwrap();
    killed.flush().unwrap();
    drop(killed);

    assert_eq!(
        request(&mut first, "one", "health", json!({}))["result"]["ready"],
        true
    );
    assert_eq!(
        request(&mut second, "two", "health", json!({}))["result"]["ready"],
        true
    );

    close_owner_and_wait(&mut child);
}

#[test]
fn owner_stdin_eof_stops_the_broker_without_windows_cleanup_files() {
    let temp = TempDir::new().unwrap();
    let endpoint = unique_endpoint("owner-eof");
    let mut child = spawn_helper(temp.path(), &endpoint);
    let mut client = open_client(&endpoint, Duration::from_secs(5));
    assert_eq!(
        request(&mut client, "ready", "health", json!({}))["result"]["ready"],
        true
    );

    close_owner_and_wait(&mut child);

    assert!(OpenOptions::new()
        .read(true)
        .write(true)
        .open(&endpoint)
        .is_err());
    assert_eq!(
        std::fs::read_dir(temp.path()).unwrap().count(),
        2,
        "Windows broker must create only the two lock files"
    );
}

#[test]
fn shutdown_closes_only_the_requesting_connection() {
    let temp = TempDir::new().unwrap();
    let endpoint = unique_endpoint("shutdown");
    let mut child = spawn_helper(temp.path(), &endpoint);
    let mut first = open_client(&endpoint, Duration::from_secs(5));
    let mut second = open_client(&endpoint, Duration::from_secs(5));

    assert_eq!(
        request(&mut first, "shutdown", "shutdown", json!({}))["request_id"],
        "shutdown"
    );
    assert_pipe_closed(first);
    assert_eq!(
        request(&mut second, "health", "health", json!({}))["result"]["ready"],
        true
    );
    assert!(child.try_wait().unwrap().is_none());

    close_owner_and_wait(&mut child);
}

fn connected_pair(
    label: &str,
) -> (
    Listener,
    julie_semantic_sidecar::broker::transport::Connection,
    File,
) {
    let endpoint = unique_endpoint(label);
    let bound = AtomicBool::new(false);
    let listener = Listener::bind(&endpoint, &bound).unwrap();
    let accepter = listener.clone();
    let join = thread::spawn(move || accepter.accept().unwrap());
    let client = open_client(&endpoint, Duration::from_secs(1));
    let connection = join.join().unwrap();
    (listener, connection, client)
}

fn unique_endpoint(label: &str) -> String {
    let sequence = ENDPOINT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!(
        r"\\.\pipe\julie-semantic-sidecar-{label}-{}-{nanos}-{sequence}",
        std::process::id()
    )
}

fn open_client(endpoint: &str, timeout: Duration) -> File {
    let deadline = Instant::now() + timeout;
    loop {
        match OpenOptions::new().read(true).write(true).open(endpoint) {
            Ok(file) => return file,
            Err(err) => {
                assert!(
                    Instant::now() < deadline,
                    "could not connect to {endpoint}: {err}"
                );
                let wide: Vec<u16> = endpoint.encode_utf16().chain([0]).collect();
                unsafe {
                    WaitNamedPipeW(wide.as_ptr(), 20);
                }
            }
        }
    }
}

fn request(stream: &mut File, request_id: &str, method: &str, params: Value) -> Value {
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

fn assert_pipe_closed(stream: File) {
    let mut line = String::new();
    match BufReader::new(stream).read_line(&mut line) {
        Ok(0) => {}
        Err(error)
            if matches!(
                error.raw_os_error(),
                Some(code)
                    if code == ERROR_BROKEN_PIPE as i32
                        || code == ERROR_PIPE_NOT_CONNECTED as i32
            ) => {}
        result => panic!("expected a closed named pipe, got {result:?}"),
    }
}

fn config(root: &Path, endpoint: String) -> BrokerConfig {
    BrokerConfig {
        model_id: "test-model".to_string(),
        endpoint: BrokerEndpoint::Windows(endpoint),
        service_lock: root.join("broker.lock"),
        accelerator_lock: root.join("accelerator.lock"),
    }
}

fn spawn_helper(root: &Path, endpoint: &str) -> Child {
    Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "windows_broker_process_helper", "--nocapture"])
        .env(HELPER_ENV, "1")
        .env(HELPER_ROOT_ENV, root)
        .env(HELPER_ENDPOINT_ENV, endpoint)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

fn close_owner_and_wait(child: &mut Child) {
    drop(child.stdin.take());
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "broker exited with {status}");
            return;
        }
        assert!(Instant::now() < deadline, "broker did not exit");
        thread::sleep(Duration::from_millis(20));
    }
}

struct TokenUserBuffer {
    token: HANDLE,
    words: Vec<usize>,
}

impl TokenUserBuffer {
    fn sid(&self) -> *mut c_void {
        unsafe { (*(self.words.as_ptr().cast::<TOKEN_USER>())).User.Sid }
    }
}

impl Drop for TokenUserBuffer {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.token);
        }
    }
}

fn current_token_user() -> TokenUserBuffer {
    let mut token = null_mut();
    assert_ne!(
        unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) },
        0
    );
    let mut needed = 0;
    unsafe {
        GetTokenInformation(token, TokenUser, null_mut(), 0, &mut needed);
    }
    assert!(needed > 0);
    let mut words = vec![0_usize; (needed as usize).div_ceil(size_of::<usize>())];
    assert_ne!(
        unsafe {
            GetTokenInformation(
                token,
                TokenUser,
                words.as_mut_ptr().cast(),
                needed,
                &mut needed,
            )
        },
        0
    );
    TokenUserBuffer { token, words }
}
