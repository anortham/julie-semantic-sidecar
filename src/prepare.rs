//! `prepare` subcommand: model acquisition into the shared cache.
//!
//! Acquisition is the sidecar binary's exclusive job — no consumer parses a model URL,
//! computes a model path, or downloads a weight file. This module owns the disk
//! preflight, the streaming download, sha256 verification, the atomic rename, and the
//! cache lock that makes concurrent invocations safe.
//!
//! # Seam
//!
//! [`run`] resolves the manifest pin and the cache directory from the environment, then
//! delegates to [`acquire`], which takes its source URL, digest, and cache directory as
//! plain arguments in a [`PrepareRequest`]. Tests drive [`acquire`] against a local
//! fixture server and a temporary cache, so no test path reaches the pinned URLs.
//!
//! # Events
//!
//! Machine-readable NDJSON is written to the event sink — stdout under [`run`]. Free-form
//! diagnostics go to stderr and are never contractual.
//!
//! ```text
//! {"event":"waiting","model_id":"qwen3-0.6b-f16"}
//! {"event":"progress","model_id":"qwen3-0.6b-f16","received_bytes":65536,"total_bytes":1197629632}
//! {"event":"done","model_id":"qwen3-0.6b-f16","path":"/…/Qwen3-Embedding-0.6B-f16.gguf","sha256":"421a…"}
//! {"event":"error","model_id":"qwen3-0.6b-f16","message":"…","expected_path":"/…","source_url":"https://…"}
//! ```

use crate::manifest::{self, ModelPin};
use fs4::FileExt;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

/// Leading component of a partial-download file name, matched by [`clean_stale_partials`].
pub const PARTIAL_PREFIX: &str = ".julie-prepare-";

/// Trailing component of a partial-download file name, matched by [`clean_stale_partials`].
pub const PARTIAL_SUFFIX: &str = ".partial";

/// File-name prefix a live download uses for `model_id`, so [`clean_stale_partials`] can tell
/// which cache lock owns a partial instead of deleting a file a running `prepare` still holds.
///
/// The full name is `<prefix><random>.partial`; the model id is everything between
/// [`PARTIAL_PREFIX`] and the last `.` before the random component, so ids containing dots
/// (`qwen3-0.6b-f16`) round-trip.
pub fn partial_prefix(model_id: &str) -> String {
    format!("{PARTIAL_PREFIX}{model_id}.")
}

fn model_of_partial(name: &str) -> Option<&str> {
    let inner = name
        .strip_prefix(PARTIAL_PREFIX)?
        .strip_suffix(PARTIAL_SUFFIX)?;
    let (model_id, random) = inner.rsplit_once('.')?;
    (!model_id.is_empty() && !random.is_empty()).then_some(model_id)
}

const PROGRESS_INTERVAL: Duration = Duration::from_secs(1);
const COPY_CHUNK_BYTES: usize = 64 * 1024;
const UNKNOWN_MODEL_EXIT: u8 = 2;

/// What [`acquire`] needs to fetch one model, decoupled from the embedded manifest so
/// tests can point it at a local fixture server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrepareRequest {
    /// Manifest id reported in every event.
    pub model_id: String,
    /// File name the verified download is renamed to inside the cache directory.
    pub file_name: String,
    /// URL the weight file is streamed from.
    pub source_url: String,
    /// Lowercase hex sha256 the completed file must match before it becomes visible.
    pub sha256: String,
    /// Declared size, checked against free space before the download starts.
    pub size_bytes: u64,
}

impl PrepareRequest {
    /// Builds the request for a pinned model from the embedded manifest.
    pub fn from_pin(pin: &ModelPin) -> Self {
        PrepareRequest {
            model_id: pin.id.to_string(),
            file_name: pin.file.to_string(),
            source_url: pin.url.to_string(),
            sha256: pin.sha256.to_string(),
            size_bytes: pin.size_bytes,
        }
    }
}

/// A model present in the cache under its final name with a verified digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prepared {
    /// Absolute path of the verified model file.
    pub path: PathBuf,
    /// True when the file was already present and verified without any network access.
    pub already_cached: bool,
}

/// A `prepare` failure that has already been reported as an `error` event.
#[derive(Debug)]
pub struct PrepareError {
    message: String,
}

impl PrepareError {
    /// The actionable message carried by the emitted `error` event.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for PrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PrepareError {}

/// Downloads and verifies the manifest model, defaulting to the tier in
/// [`crate::DEFAULT_MODEL_ID`] when `model_id` is `None`.
///
/// Exit codes: `0` on success including an already-cached model, `2` for an unknown
/// manifest id, `1` for every other failure.
pub fn run(model_id: Option<&str>) -> ExitCode {
    let mut events = std::io::stdout().lock();
    let pin = match resolve(model_id) {
        Ok(pin) => pin,
        Err(message) => {
            emit_error(
                &mut events,
                model_id.unwrap_or(crate::DEFAULT_MODEL_ID),
                &message,
                None,
                None,
            );
            return ExitCode::from(UNKNOWN_MODEL_EXIT);
        }
    };
    let cache_dir = match cache_dir() {
        Ok(dir) => dir,
        Err(err) => {
            emit_error(&mut events, pin.id, err.message(), None, Some(pin.url));
            return ExitCode::FAILURE;
        }
    };
    match acquire(&PrepareRequest::from_pin(pin), &cache_dir, &mut events) {
        Ok(_) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("julie-semantic-sidecar: prepare failed: {err}");
            ExitCode::FAILURE
        }
    }
}

/// Resolves a manifest id, or the default tier when `model_id` is `None`.
///
/// The error message names the rejected id and lists every id the binary knows, because
/// the manifest is embedded and a caller has no other way to enumerate it.
pub fn resolve(model_id: Option<&str>) -> Result<&'static ModelPin, String> {
    let Some(id) = model_id else {
        return Ok(manifest::default_model());
    };
    manifest::by_id(id).ok_or_else(|| {
        let known: Vec<&str> = manifest::manifest().iter().map(|pin| pin.id).collect();
        format!("unknown model id '{id}'; known ids: {}", known.join(", "))
    })
}

/// Resolves the shared model cache directory without creating it.
///
/// `JULIE_EMBEDDING_CACHE_DIR` wins when set to a non-empty value. Otherwise the location
/// is `~/.cache/julie-semantic` on Unix — deliberately home-rooted rather than the
/// platform cache dir, so macOS shares one path with Linux instead of diverging into
/// `~/Library/Caches` — and `%LOCALAPPDATA%\julie-semantic` on Windows.
pub fn cache_dir() -> Result<PathBuf, PrepareError> {
    if let Some(configured) = std::env::var_os("JULIE_EMBEDDING_CACHE_DIR") {
        if !configured.is_empty() {
            return Ok(PathBuf::from(configured));
        }
    }
    #[cfg(windows)]
    {
        dirs::data_local_dir()
            .map(|base| base.join("julie-semantic"))
            .ok_or_else(|| PrepareError {
                message: "cannot resolve %LOCALAPPDATA%; set JULIE_EMBEDDING_CACHE_DIR".to_string(),
            })
    }
    #[cfg(not(windows))]
    {
        dirs::home_dir()
            .map(|home| home.join(".cache").join("julie-semantic"))
            .ok_or_else(|| PrepareError {
                message: "cannot resolve the home directory; set JULIE_EMBEDDING_CACHE_DIR"
                    .to_string(),
            })
    }
}

/// Removes partial downloads left behind by a killed `prepare`, returning what it deleted.
///
/// Ownership is decided by the cache lock, never by the file's presence alone: `serve` starts
/// while another process may be mid-download, and deleting that open temp file breaks the
/// download (the final `persist` fails on Unix, the unlink itself fails on Windows). A model's
/// partials are deleted only when this call can take `<model_id>.lock` itself — a lock it cannot
/// take means an active `prepare` owns them.
///
/// Partials that predate model-qualified names (see [`partial_prefix`]) carry no model, so they
/// are deleted only when no cache lock in the directory is held at all — i.e. when no download
/// anywhere could own them.
///
/// Lock files are permanent cache residents — one tiny file per pinned model, never deleted.
/// Unlinking a "free" lock races a concurrent `prepare` that has opened the file but not yet
/// locked it: the loser locks the unlinked inode, a third process re-creates the path, and two
/// processes hold "exclusive" locks on different inodes — both download, violating `acquire`'s
/// downloads-or-waits guarantee. Cleanup therefore probes locks only to attribute partials.
///
/// A missing cache directory is not an error — nothing has been prepared yet.
pub fn clean_stale_partials(cache_dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let entries = match std::fs::read_dir(cache_dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };

    let mut attributed: Vec<(String, PathBuf)> = Vec::new();
    let mut unattributed: Vec<PathBuf> = Vec::new();
    let mut lock_paths: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".lock") {
            lock_paths.push(entry.path());
        } else if name.starts_with(PARTIAL_PREFIX) && name.ends_with(PARTIAL_SUFFIX) {
            match model_of_partial(&name) {
                Some(model_id) => attributed.push((model_id.to_string(), entry.path())),
                None => unattributed.push(entry.path()),
            }
        }
    }

    let mut free_locks: Vec<(PathBuf, File)> = Vec::new();
    let mut held_models: Vec<String> = Vec::new();
    let mut any_lock_held = false;
    for path in lock_paths {
        match try_hold(&path) {
            Some(file) => free_locks.push((path, file)),
            None => {
                any_lock_held = true;
                if let Some(model_id) = path.file_stem().and_then(|stem| stem.to_str()) {
                    held_models.push(model_id.to_string());
                }
            }
        }
    }

    let mut removed = Vec::new();
    for (model_id, path) in attributed {
        if held_models.contains(&model_id) {
            continue;
        }
        remove_partial(&path, &mut removed)?;
    }
    if !any_lock_held {
        for path in unattributed {
            remove_partial(&path, &mut removed)?;
        }
    }

    for (_path, file) in free_locks {
        let _ = FileExt::unlock(&file);
        drop(file);
    }

    removed.sort();
    Ok(removed)
}

fn remove_partial(path: &Path, removed: &mut Vec<PathBuf>) -> std::io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => removed.push(path.to_path_buf()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(err),
    }
    Ok(())
}

/// Takes `path`'s exclusive lock without blocking, yielding `None` when another process holds it
/// or the file cannot be opened at all.
fn try_hold(path: &Path) -> Option<File> {
    let file = OpenOptions::new().read(true).write(true).open(path).ok()?;
    // std::fs::File gained inherent lock/try_lock in Rust 1.89 and shadows fs4's same-named
    // trait methods; call fs4 fully qualified.
    FileExt::try_lock(&file).ok()?;
    Some(file)
}

/// Makes the requested model present and verified in `cache_dir`, emitting NDJSON events.
///
/// Holds a per-model cache lock for the whole check-download-verify-rename sequence, so a
/// concurrent invocation either downloads or waits — never both. The digest is verified
/// before the rename, so no path exists on which an unverified file appears under its
/// final name.
pub fn acquire(
    request: &PrepareRequest,
    cache_dir: &Path,
    events: &mut dyn Write,
) -> Result<Prepared, PrepareError> {
    acquire_with_timeouts(
        request,
        cache_dir,
        events,
        DownloadTimeouts {
            total: download_total_timeout(request.size_bytes),
            ..DOWNLOAD_TIMEOUTS
        },
    )
}

fn acquire_with_timeouts(
    request: &PrepareRequest,
    cache_dir: &Path,
    events: &mut dyn Write,
    timeouts: DownloadTimeouts,
) -> Result<Prepared, PrepareError> {
    let final_path = cache_dir.join(&request.file_name);

    if let Err(err) = std::fs::create_dir_all(cache_dir) {
        return Err(report(
            events,
            request,
            &final_path,
            format!(
                "cannot create cache directory {}: {err}",
                cache_dir.display()
            ),
        ));
    }

    let lock = match hold_cache_lock(request, cache_dir, events) {
        Ok(lock) => lock,
        Err(message) => return Err(report(events, request, &final_path, message)),
    };
    let outcome = acquire_locked(request, &final_path, cache_dir, events, timeouts);
    let _ = FileExt::unlock(&lock);
    match outcome {
        Ok(prepared) => Ok(prepared),
        Err(message) => Err(report(events, request, &final_path, message)),
    }
}

fn report(
    events: &mut dyn Write,
    request: &PrepareRequest,
    final_path: &Path,
    message: String,
) -> PrepareError {
    emit_error(
        events,
        &request.model_id,
        &message,
        Some(final_path),
        Some(&request.source_url),
    );
    PrepareError { message }
}

fn acquire_locked(
    request: &PrepareRequest,
    final_path: &Path,
    cache_dir: &Path,
    events: &mut dyn Write,
    timeouts: DownloadTimeouts,
) -> Result<Prepared, String> {
    if final_path.exists() {
        match file_digest(final_path) {
            Ok(digest) if digest.eq_ignore_ascii_case(&request.sha256) => {
                emit_done(events, &request.model_id, final_path, &request.sha256);
                return Ok(Prepared {
                    path: final_path.to_path_buf(),
                    already_cached: true,
                });
            }
            Ok(_) => {
                eprintln!(
                    "julie-semantic-sidecar: cached {} failed verification; re-downloading",
                    final_path.display()
                );
                std::fs::remove_file(final_path).map_err(|err| {
                    format!("cannot remove stale {}: {err}", final_path.display())
                })?;
            }
            Err(err) => {
                return Err(format!(
                    "cannot read cached {}: {err}",
                    final_path.display()
                ));
            }
        }
    }

    preflight_disk(cache_dir, request.size_bytes)?;
    download_verified(request, final_path, cache_dir, events, timeouts)?;
    emit_done(events, &request.model_id, final_path, &request.sha256);
    Ok(Prepared {
        path: final_path.to_path_buf(),
        already_cached: false,
    })
}

fn preflight_disk(cache_dir: &Path, size_bytes: u64) -> Result<(), String> {
    let available = fs4::available_space(cache_dir).map_err(|err| {
        format!(
            "cannot measure free space in {}: {err}",
            cache_dir.display()
        )
    })?;
    if available < size_bytes {
        return Err(format!(
            "insufficient disk space in {}: need {size_bytes} bytes, {available} available",
            cache_dir.display()
        ));
    }
    Ok(())
}

/// Connection-phase deadlines. ureq 3.3 defaults every timeout to `None`, so without
/// these a server that accepts and then stalls would block forever while `acquire`
/// holds the per-model lock — hanging every concurrent prepare instead of failing loud.
const DOWNLOAD_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const DOWNLOAD_RESPONSE_TIMEOUT: Duration = Duration::from_secs(300);
const DOWNLOAD_TOTAL_TIMEOUT: Duration = Duration::from_secs(2_700);
const DOWNLOAD_MIN_BYTES_PER_SECOND: u64 = 64 * 1_024;

fn download_total_timeout(size_bytes: u64) -> Duration {
    Duration::from_secs(
        size_bytes
            .div_ceil(DOWNLOAD_MIN_BYTES_PER_SECOND)
            .max(DOWNLOAD_TOTAL_TIMEOUT.as_secs()),
    )
}

#[derive(Clone, Copy)]
struct DownloadTimeouts {
    connect: Duration,
    response: Duration,
    total: Duration,
}

const DOWNLOAD_TIMEOUTS: DownloadTimeouts = DownloadTimeouts {
    connect: DOWNLOAD_CONNECT_TIMEOUT,
    response: DOWNLOAD_RESPONSE_TIMEOUT,
    total: DOWNLOAD_TOTAL_TIMEOUT,
};

fn download_verified(
    request: &PrepareRequest,
    final_path: &Path,
    cache_dir: &Path,
    events: &mut dyn Write,
    timeouts: DownloadTimeouts,
) -> Result<(), String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(timeouts.connect))
        .timeout_recv_response(Some(timeouts.response))
        .timeout_global(Some(timeouts.total))
        .build()
        .into();
    let mut response = agent
        .get(&request.source_url)
        .call()
        .map_err(|err| format!("cannot fetch {}: {err}; retry prepare", request.source_url))?;
    let total_bytes = request.size_bytes;
    if let Some(offered) = response.body_mut().content_length() {
        if offered != total_bytes {
            return Err(format!(
                "declared size mismatch for {}: the pin declares {total_bytes} bytes, {} offers {offered}",
                request.model_id, request.source_url
            ));
        }
    }
    let mut reader = response.body_mut().with_config().limit(u64::MAX).reader();

    let mut temp = tempfile::Builder::new()
        .prefix(partial_prefix(&request.model_id).as_str())
        .suffix(PARTIAL_SUFFIX)
        .tempfile_in(cache_dir)
        .map_err(|err| {
            format!(
                "cannot create a partial file in {}: {err}",
                cache_dir.display()
            )
        })?;

    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; COPY_CHUNK_BYTES];
    let mut received: u64 = 0;
    let mut last_progress = Instant::now();
    emit_progress(events, &request.model_id, received, total_bytes);
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|err| format!("cannot read {}: {err}; retry prepare", request.source_url))?;
        if read == 0 {
            break;
        }
        // A lying or malicious server would otherwise fill the disk before the digest check.
        if received + read as u64 > total_bytes {
            drop(temp);
            return Err(format!(
                "oversized response for {}: {} sent more than the pinned {total_bytes} bytes",
                request.model_id, request.source_url
            ));
        }
        hasher.update(&buffer[..read]);
        temp.write_all(&buffer[..read])
            .map_err(|err| format!("cannot write the partial download: {err}"))?;
        received += read as u64;
        if last_progress.elapsed() >= PROGRESS_INTERVAL {
            emit_progress(events, &request.model_id, received, total_bytes);
            last_progress = Instant::now();
        }
    }
    temp.flush()
        .map_err(|err| format!("cannot flush the partial download: {err}"))?;
    emit_progress(events, &request.model_id, received, total_bytes);

    let digest = hex(hasher.finalize().as_slice());
    if !digest.eq_ignore_ascii_case(&request.sha256) {
        drop(temp);
        return Err(format!(
            "sha256 mismatch for {}: expected {}, downloaded {digest}",
            request.model_id, request.sha256
        ));
    }
    temp.persist(final_path)
        .map_err(|err| format!("cannot place {}: {err}", final_path.display()))?;
    Ok(())
}

fn hold_cache_lock(
    request: &PrepareRequest,
    cache_dir: &Path,
    events: &mut dyn Write,
) -> Result<File, String> {
    let lock_path = cache_dir.join(format!("{}.lock", request.model_id));
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|err| format!("cannot open the cache lock {}: {err}", lock_path.display()))?;
    // std::fs::File gained inherent lock/try_lock in Rust 1.89 and shadows fs4's
    // same-named trait methods; call fs4 fully qualified.
    if FileExt::try_lock(&lock).is_err() {
        emit(
            events,
            json!({"event": "waiting", "model_id": request.model_id}),
        );
        FileExt::lock(&lock)
            .map_err(|err| format!("cannot take the cache lock {}: {err}", lock_path.display()))?;
    }
    Ok(lock)
}

fn file_digest(path: &Path) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; COPY_CHUNK_BYTES];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(hasher.finalize().as_slice()))
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn emit(events: &mut dyn Write, value: serde_json::Value) {
    let _ = writeln!(events, "{value}");
    let _ = events.flush();
}

fn emit_progress(events: &mut dyn Write, model_id: &str, received: u64, total: u64) {
    emit(
        events,
        json!({
            "event": "progress",
            "model_id": model_id,
            "received_bytes": received,
            "total_bytes": total,
        }),
    );
}

fn emit_done(events: &mut dyn Write, model_id: &str, path: &Path, sha256: &str) {
    emit(
        events,
        json!({
            "event": "done",
            "model_id": model_id,
            "path": path.to_string_lossy(),
            "sha256": sha256,
        }),
    );
}

fn emit_error(
    events: &mut dyn Write,
    model_id: &str,
    message: &str,
    expected_path: Option<&Path>,
    source_url: Option<&str>,
) {
    emit(
        events,
        json!({
            "event": "error",
            "model_id": model_id,
            "message": message,
            "expected_path": expected_path.map(|path| path.to_string_lossy().into_owned()),
            "source_url": source_url,
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn a_stalled_download_times_out_and_releases_the_model_lock() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let url = format!("http://{}/model.gguf", listener.local_addr().expect("addr"));
        thread::spawn(move || {
            if let Ok((stream, _)) = listener.accept() {
                thread::sleep(Duration::from_secs(2));
                drop(stream);
            }
        });
        let cache = tempfile::tempdir().expect("tempdir");
        let request = PrepareRequest {
            model_id: "stall-model".to_string(),
            file_name: "stall-model.gguf".to_string(),
            source_url: url,
            sha256: "0".repeat(64),
            size_bytes: 1,
        };
        let timeouts = DownloadTimeouts {
            connect: Duration::from_millis(100),
            response: Duration::from_millis(100),
            total: Duration::from_millis(200),
        };
        let mut output = Vec::new();

        let error = acquire_with_timeouts(&request, cache.path(), &mut output, timeouts)
            .expect_err("stall must time out");

        assert!(error.message().contains("retry prepare"), "{error}");
        assert!(!error.message().contains("JULIE_SIDECAR_"), "{error}");
        let lock = cache.path().join("stall-model.lock");
        assert!(try_hold(&lock).is_some(), "model lock remained held");
    }

    #[test]
    fn total_download_timeout_scales_with_the_declared_model_size() {
        assert_eq!(
            download_total_timeout(133_609_568),
            Duration::from_secs(2_700)
        );
        assert_eq!(
            download_total_timeout(1_197_629_632),
            Duration::from_secs(18_275)
        );
    }
}
