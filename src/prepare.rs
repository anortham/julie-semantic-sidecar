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
/// A missing cache directory is not an error — nothing has been prepared yet.
pub fn clean_stale_partials(cache_dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let entries = match std::fs::read_dir(cache_dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };
    let mut removed = Vec::new();
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(PARTIAL_PREFIX) && name.ends_with(PARTIAL_SUFFIX) {
            let path = entry.path();
            match std::fs::remove_file(&path) {
                Ok(()) => removed.push(path),
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                Err(err) => return Err(err),
            }
        }
    }
    removed.sort();
    Ok(removed)
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
    let outcome = acquire_locked(request, &final_path, cache_dir, events);
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
    download_verified(request, final_path, cache_dir, events)?;
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

fn download_verified(
    request: &PrepareRequest,
    final_path: &Path,
    cache_dir: &Path,
    events: &mut dyn Write,
) -> Result<(), String> {
    let mut response = ureq::get(&request.source_url)
        .call()
        .map_err(|err| format!("cannot fetch {}: {err}", request.source_url))?;
    let total_bytes = response
        .body_mut()
        .content_length()
        .unwrap_or(request.size_bytes);
    let mut reader = response.body_mut().with_config().limit(u64::MAX).reader();

    let mut temp = tempfile::Builder::new()
        .prefix(PARTIAL_PREFIX)
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
            .map_err(|err| format!("cannot read {}: {err}", request.source_url))?;
        if read == 0 {
            break;
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
