//! Backend selection: which llama.cpp device the sidecar loads on, and why.
//!
//! CPU is the floor in every build. Accelerators become candidates only when the sidecar
//! declares them and llama.cpp enumerates a matching device after packaged modules load.
//! The first start times fixed single-item and indexing-batch probes against CPU; only a
//! faster accelerator wins. Verdicts are cached per complete build, package, model, GPU,
//! and driver identity, while forced CPU bypasses discovery, timing, and cache entirely.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::health::BackendCapabilities;

/// Environment variable that pins the backend, bypassing benchmark and cache entirely.
pub const FORCE_BACKEND_ENV: &str = "JULIE_SIDECAR_FORCE_BACKEND";

/// Canonical name of the backend every build can serve.
pub const CPU: &str = "cpu";

/// Canonical name of the backend the `metal` feature compiles in.
pub const METAL: &str = "metal";

/// Canonical name of the Vulkan backend.
pub const VULKAN: &str = "vulkan";

/// Canonical name of the CUDA backend.
pub const CUDA: &str = "cuda";

/// File the cached benchmark choice is written to, beside the model cache.
pub const SELECTION_CACHE_FILE: &str = "backend-selection.json";

/// Reason reported when an accelerated backend was asked for and this build has none.
pub const NO_ACCELERATED_BACKEND: &str = "cpu (no accelerated backend compiled)";

/// Reproducible identity emitted by this crate's build script.
pub const NATIVE_BUILD_IDENTITY: &str = env!("JULIE_NATIVE_BUILD_IDENTITY");

/// Backends the sidecar package declares, independent of upstream transitive features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeclaredBackends {
    pub metal: bool,
    pub vulkan: bool,
    pub cuda: bool,
    pub rocm: bool,
    pub dynamic_backends: bool,
}

impl DeclaredBackends {
    /// The features explicitly selected on this package.
    pub fn build() -> Self {
        Self {
            metal: cfg!(feature = "metal"),
            vulkan: cfg!(feature = "vulkan"),
            cuda: cfg!(feature = "cuda"),
            rocm: cfg!(feature = "rocm"),
            dynamic_backends: cfg!(feature = "dynamic-backends"),
        }
    }

    /// Stable cache identity for the package's native backend surface.
    pub fn identity(self) -> String {
        format!(
            "metal={};vulkan={};cuda={};rocm={};dynamic={}",
            self.metal, self.vulkan, self.cuda, self.rocm, self.dynamic_backends
        )
    }

    fn supports(self, backend: &str) -> bool {
        match backend {
            METAL => self.metal,
            VULKAN => self.vulkan,
            CUDA => self.cuda,
            _ => backend == CPU,
        }
    }

    fn stable_accelerators(self) -> impl Iterator<Item = &'static str> {
        [
            (METAL, self.metal),
            (VULKAN, self.vulkan),
            (CUDA, self.cuda),
        ]
        .into_iter()
        .filter_map(|(backend, enabled)| enabled.then_some(backend))
    }
}

/// Runtime device facts returned by llama.cpp after backend modules are loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeDevice {
    pub backend: String,
    pub index: usize,
    pub name: String,
    pub description: String,
    pub memory_total: usize,
    pub driver: String,
}

/// A backend that can be placed explicitly for a probe or final model load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendCandidate {
    pub backend: String,
    pub device_index: Option<usize>,
}

/// Runtime candidates and the machine identity that makes their verdict cacheable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Discovery {
    pub candidates: Vec<BackendCandidate>,
    pub capabilities: BackendCapabilities,
    pub machine: Machine,
    declared_accelerators: Vec<String>,
}

/// Cache-key components known before runtime device discovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectionContext<'a> {
    pub sidecar_version: &'a str,
    pub model_sha256: &'a str,
    pub native_build_identity: &'a str,
    pub packaged_backend_identity: &'a str,
}

/// Comparable inference measurements for the fixed single-item and indexing probes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbeTiming {
    pub batch_1: Duration,
    pub batch_16: Duration,
}

impl ProbeTiming {
    fn total(self) -> Duration {
        self.batch_1.saturating_add(self.batch_16)
    }
}

/// Selection plus the exact placement and capabilities proven by runtime probing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeSelection {
    pub selection: Selection,
    pub capabilities: BackendCapabilities,
    pub device_index: Option<usize>,
}

/// Identity of the machine's accelerator, as two cache-key components.
///
/// Both are opaque strings: only equality matters, because a change in either invalidates
/// the cached benchmark result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Machine {
    /// GPU identity, or `none` when the build sees no accelerator.
    pub gpu: String,
    /// Driver identity, or `none` when the build sees no accelerator.
    pub driver: String,
}

impl Machine {
    /// The identity a build with no accelerated backend reports.
    pub fn none() -> Self {
        Self {
            gpu: "none".to_string(),
            driver: "none".to_string(),
        }
    }
}

/// The backend decision, in the shape `health` reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    /// Backend the cached choice asked for.
    pub requested: String,
    /// Backend that actually loaded.
    pub resolved: String,
    /// Whether the resolved backend is something other than CPU.
    pub accelerated: bool,
    /// Why the resolved backend differs from the requested one, or `None`.
    pub degraded_reason: Option<String>,
}

impl Selection {
    /// Builds a selection, deriving `accelerated` from the resolved backend.
    pub fn new(
        requested: impl Into<String>,
        resolved: impl Into<String>,
        degraded_reason: Option<String>,
    ) -> Self {
        let resolved = resolved.into();
        Self {
            requested: requested.into(),
            accelerated: resolved != CPU,
            resolved,
            degraded_reason,
        }
    }

    /// The undegraded floor: CPU asked for, CPU loaded.
    pub fn cpu() -> Self {
        Self::new(CPU, CPU, None)
    }

    /// An accelerated backend was asked for and CPU answered instead.
    pub fn degraded_to_cpu(requested: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::new(requested, CPU, Some(reason.into()))
    }
}

/// Intersects explicitly declared package backends with enumerated llama.cpp devices.
pub fn discover_candidates(
    declared: &DeclaredBackends,
    runtime_devices: &[RuntimeDevice],
) -> Discovery {
    let mut candidates = vec![BackendCandidate {
        backend: CPU.to_string(),
        device_index: None,
    }];
    let mut capabilities = BackendCapabilities::cpu_only();
    let mut identity_parts = Vec::new();
    let mut driver_parts = Vec::new();

    for backend in [METAL, VULKAN, CUDA] {
        if !declared.supports(backend) {
            continue;
        }
        let Some(device) = runtime_devices
            .iter()
            .find(|device| normalize_backend(&device.backend) == Some(backend))
        else {
            continue;
        };
        candidates.push(BackendCandidate {
            backend: backend.to_string(),
            device_index: Some(device.index),
        });
        set_capability(&mut capabilities, backend, true);
        identity_parts.push(format!(
            "{backend}:{}:{}:{}",
            device.name, device.description, device.memory_total
        ));
        driver_parts.push(format!("{backend}:{}", device.driver));
    }

    Discovery {
        candidates,
        capabilities,
        machine: if identity_parts.is_empty() {
            Machine::none()
        } else {
            Machine {
                gpu: identity_parts.join(","),
                driver: driver_parts.join(","),
            }
        },
        declared_accelerators: declared.stable_accelerators().map(str::to_string).collect(),
    }
}

/// Resolves a backend from injected discovery and inference probes.
pub fn select_runtime_with<D, B>(
    cache_dir: &Path,
    context: SelectionContext<'_>,
    forced: Option<&str>,
    discover: D,
    mut benchmark: B,
) -> Result<RuntimeSelection, String>
where
    D: FnOnce() -> Result<Discovery, String>,
    B: FnMut(&BackendCandidate) -> Result<ProbeTiming, String>,
{
    let forced = forced.map(|value| value.trim().to_ascii_lowercase());
    if forced.as_deref() == Some(CPU) {
        return Ok(RuntimeSelection {
            selection: Selection::cpu(),
            capabilities: BackendCapabilities::cpu_only(),
            device_index: None,
        });
    }

    let discovery = match discover() {
        Ok(discovery) => discovery,
        Err(reason) => {
            return Ok(RuntimeSelection {
                selection: Selection::new(forced.as_deref().unwrap_or(CPU), CPU, Some(reason)),
                capabilities: BackendCapabilities::cpu_only(),
                device_index: None,
            });
        }
    };

    if let Some(requested) = forced.as_deref() {
        if !matches!(requested, METAL | VULKAN | CUDA) {
            return Ok(degraded_runtime(requested, "unknown forced backend"));
        }
        if !discovery
            .candidates
            .iter()
            .any(|candidate| candidate.backend == requested)
        {
            return Ok(degraded_runtime(
                requested,
                "requested backend is unavailable",
            ));
        }
    }

    let key = runtime_cache_key(context, &discovery.machine);
    if forced.is_none() {
        if let Some(cached) = read_runtime_cache(cache_dir, &key) {
            let device_index = discovery
                .candidates
                .iter()
                .find(|candidate| candidate.backend == cached.selection.resolved)
                .and_then(|candidate| candidate.device_index);
            return Ok(RuntimeSelection {
                selection: cached.selection,
                capabilities: cached.capabilities,
                device_index,
            });
        }
    }

    let cpu = &discovery.candidates[0];
    let cpu_timing = match benchmark(cpu) {
        Ok(timing) => timing,
        Err(reason) => return Err(format!("cpu probe failed: {reason}")),
    };

    let mut capabilities = discovery.capabilities;
    let accelerated_candidates: Vec<&BackendCandidate> = discovery
        .candidates
        .iter()
        .skip(1)
        .filter(|candidate| {
            forced
                .as_deref()
                .is_none_or(|requested| candidate.backend == requested)
        })
        .collect();

    let requested = forced
        .clone()
        .or_else(|| {
            accelerated_candidates
                .first()
                .map(|candidate| candidate.backend.clone())
        })
        .or_else(|| discovery.declared_accelerators.first().cloned());

    let mut fastest: Option<(&BackendCandidate, ProbeTiming)> = None;
    let mut failures = Vec::new();
    for candidate in accelerated_candidates {
        match benchmark(candidate) {
            Ok(timing) => {
                if fastest
                    .as_ref()
                    .is_none_or(|(_, current)| timing.total() < current.total())
                {
                    fastest = Some((candidate, timing));
                }
            }
            Err(reason) => {
                set_capability(&mut capabilities, &candidate.backend, false);
                failures.push(format!("{} probe failed: {reason}", candidate.backend));
            }
        }
    }

    let result = if let Some((winner, timing)) = fastest {
        if timing.total() < cpu_timing.total() {
            RuntimeSelection {
                selection: Selection::new(&winner.backend, &winner.backend, None),
                capabilities,
                device_index: winner.device_index,
            }
        } else {
            RuntimeSelection {
                selection: Selection::degraded_to_cpu(
                    requested.as_deref().unwrap_or(CPU),
                    "accelerated backend did not beat cpu",
                ),
                capabilities,
                device_index: None,
            }
        }
    } else if let Some(requested) = requested {
        RuntimeSelection {
            selection: Selection::degraded_to_cpu(
                requested,
                if failures.is_empty() {
                    "requested backend is unavailable".to_string()
                } else {
                    failures.join("; ")
                },
            ),
            capabilities,
            device_index: None,
        }
    } else {
        RuntimeSelection {
            selection: Selection::cpu(),
            capabilities,
            device_index: None,
        }
    };

    if forced.is_none() {
        write_runtime_cache(cache_dir, &key, &result);
    }
    Ok(result)
}

fn degraded_runtime(requested: &str, reason: &str) -> RuntimeSelection {
    RuntimeSelection {
        selection: Selection::degraded_to_cpu(requested, reason),
        capabilities: BackendCapabilities::cpu_only(),
        device_index: None,
    }
}

fn normalize_backend(backend: &str) -> Option<&'static str> {
    if backend.eq_ignore_ascii_case(CPU) {
        Some(CPU)
    } else if backend.eq_ignore_ascii_case(METAL) || backend.eq_ignore_ascii_case("MTL") {
        Some(METAL)
    } else if backend.eq_ignore_ascii_case(VULKAN) {
        Some(VULKAN)
    } else if backend.eq_ignore_ascii_case(CUDA) {
        Some(CUDA)
    } else {
        None
    }
}

/// Hashes the exact declared sibling modules into the package cache identity.
pub fn packaged_backend_identity(executable: &Path) -> String {
    let declared = DeclaredBackends::build();
    let mut hash = Sha256::new();
    hash.update(declared.identity());
    if let Some(directory) = executable.parent() {
        if let Ok(paths) = packaged_module_paths(directory, &declared) {
            for path in paths {
                hash.update(path.file_name().unwrap_or_default().as_encoded_bytes());
                if let Ok(bytes) = std::fs::read(path) {
                    hash.update(bytes);
                }
            }
        }
    }
    format!("{:x}", hash.finalize())
}

fn set_capability(capabilities: &mut BackendCapabilities, backend: &str, available: bool) {
    match backend {
        METAL => capabilities.metal = available,
        VULKAN => capabilities.vulkan = available,
        CUDA => capabilities.cuda = available,
        _ => {}
    }
}

/// Reads `JULIE_SIDECAR_FORCE_BACKEND`, treating an empty value as unset.
pub fn forced_backend() -> Option<String> {
    let value = std::env::var(FORCE_BACKEND_ENV).ok()?;
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn cache_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join(SELECTION_CACHE_FILE)
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct RuntimeCache {
    entries: BTreeMap<String, RuntimeCachedChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RuntimeCachedChoice {
    requested: String,
    resolved: String,
    degraded_reason: Option<String>,
    cuda: bool,
    metal: bool,
    vulkan: bool,
}

impl RuntimeCachedChoice {
    fn from_runtime(selection: &RuntimeSelection) -> Self {
        Self {
            requested: selection.selection.requested.clone(),
            resolved: selection.selection.resolved.clone(),
            degraded_reason: selection.selection.degraded_reason.clone(),
            cuda: selection.capabilities.cuda,
            metal: selection.capabilities.metal,
            vulkan: selection.capabilities.vulkan,
        }
    }

    fn into_runtime(self) -> RuntimeSelection {
        RuntimeSelection {
            selection: Selection::new(self.requested, self.resolved, self.degraded_reason),
            capabilities: BackendCapabilities {
                cpu: true,
                cuda: self.cuda,
                metal: self.metal,
                vulkan: self.vulkan,
                ..Default::default()
            },
            device_index: None,
        }
    }
}

fn runtime_cache_key(context: SelectionContext<'_>, machine: &Machine) -> String {
    let mut hash = Sha256::new();
    for component in [
        context.sidecar_version,
        context.model_sha256,
        context.native_build_identity,
        context.packaged_backend_identity,
        &machine.gpu,
        &machine.driver,
    ] {
        hash.update(component.len().to_le_bytes());
        hash.update(component.as_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn read_runtime_store(cache_dir: &Path) -> RuntimeCache {
    std::fs::read(cache_path(cache_dir))
        .ok()
        .and_then(|raw| serde_json::from_slice(&raw).ok())
        .unwrap_or_default()
}

fn read_runtime_cache(cache_dir: &Path, key: &str) -> Option<RuntimeSelection> {
    read_runtime_store(cache_dir)
        .entries
        .remove(key)
        .map(RuntimeCachedChoice::into_runtime)
}

fn write_runtime_cache(cache_dir: &Path, key: &str, selection: &RuntimeSelection) {
    if std::fs::create_dir_all(cache_dir).is_err() {
        return;
    }
    let Ok(lock) = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(cache_dir.join("backend-selection.lock"))
    else {
        return;
    };
    if fs4::FileExt::lock(&lock).is_err() {
        return;
    }
    let mut store = read_runtime_store(cache_dir);
    store.entries.insert(
        key.to_string(),
        RuntimeCachedChoice::from_runtime(selection),
    );
    let Ok(encoded) = serde_json::to_vec(&store) else {
        return;
    };
    let Ok(mut temporary) = tempfile::NamedTempFile::new_in(cache_dir) else {
        return;
    };
    if temporary.write_all(&encoded).is_err() || temporary.as_file().sync_all().is_err() {
        return;
    }
    let _ = temporary.persist(cache_path(cache_dir));
    let _ = fs4::FileExt::unlock(&lock);
}

/// Validates executable-relative dynamic module loading before invoking native code.
pub fn load_modules_from_executable_with<F>(executable: &Path, mut loader: F) -> Result<(), String>
where
    F: FnMut(&Path) -> Result<(), String>,
{
    let parent = executable
        .parent()
        .ok_or_else(|| "executable has no parent directory".to_string())?;
    let parent_text = parent
        .to_str()
        .ok_or_else(|| "executable parent path is not valid UTF-8".to_string())?;
    if parent_text.as_bytes().contains(&0) {
        return Err("executable parent path contains a NUL byte".to_string());
    }
    if !parent.is_dir() {
        return Err("executable parent directory is not readable".to_string());
    }
    loader(parent)
}

fn module_file_name(backend: &str) -> String {
    format!("{}ggml-{backend}{}", module_prefix(), module_suffix())
}

fn module_prefix() -> &'static str {
    if cfg!(target_os = "windows") {
        ""
    } else {
        "lib"
    }
}

fn module_suffix() -> &'static str {
    if cfg!(target_os = "windows") {
        ".dll"
    } else {
        ".so"
    }
}

fn packaged_module_paths(
    directory: &Path,
    declared: &DeclaredBackends,
) -> Result<Vec<PathBuf>, String> {
    packaged_module_paths_for_scope(directory, declared, true)
}

fn packaged_module_paths_for_scope(
    directory: &Path,
    declared: &DeclaredBackends,
    include_accelerators: bool,
) -> Result<Vec<PathBuf>, String> {
    if !directory.is_dir() {
        return Err("backend module directory is not readable".to_string());
    }
    let declared_names: Vec<String> = if include_accelerators {
        [
            (METAL, declared.metal),
            (VULKAN, declared.vulkan),
            (CUDA, declared.cuda),
        ]
        .into_iter()
        .filter(|(_, enabled)| *enabled)
        .map(|(backend, _)| module_file_name(backend))
        .collect()
    } else {
        Vec::new()
    };
    let cpu_prefix = format!("{}ggml-cpu", module_prefix());
    let mut paths = std::fs::read_dir(directory)
        .map_err(|err| format!("cannot read backend module directory: {err}"))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter(|path| {
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                return false;
            };
            (name.starts_with(&cpu_prefix) && name.ends_with(module_suffix()))
                || declared_names.iter().any(|declared| name == declared)
        })
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

fn load_packaged_modules_with<S, L>(
    executable: &Path,
    declared: &DeclaredBackends,
    include_accelerators: bool,
    mut score_cpu_variant: S,
    mut loader: L,
) -> Result<(), String>
where
    S: FnMut(&Path) -> Result<i32, String>,
    L: FnMut(&Path) -> Result<(), String>,
{
    load_modules_from_executable_with(executable, |directory| {
        let paths = packaged_module_paths_for_scope(directory, declared, include_accelerators)?;
        let cpu_base_name = module_file_name(CPU);
        let cpu_variant_prefix = format!("{}ggml-cpu-", module_prefix());
        let cpu_base = paths.iter().find(|path| {
            path.file_name().and_then(|name| name.to_str()) == Some(cpu_base_name.as_str())
        });
        let mut best: Option<(&PathBuf, i32)> = None;
        let mut scoring_failures = Vec::new();
        for path in paths.iter().filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(&cpu_variant_prefix))
        }) {
            match score_cpu_variant(path) {
                Ok(score) if score > 0 && best.is_none_or(|(_, current)| score > current) => {
                    best = Some((path, score));
                }
                Ok(_) => {}
                Err(reason) => scoring_failures.push(format!("{}: {reason}", path.display())),
            }
        }
        let cpu = best.map(|(path, _)| path).or(cpu_base).ok_or_else(|| {
            if scoring_failures.is_empty() {
                "no supported packaged CPU backend module found beside the executable".to_string()
            } else {
                format!(
                    "cannot score a packaged CPU backend module: {}",
                    scoring_failures.join(", ")
                )
            }
        })?;
        loader(cpu).map_err(|reason| {
            format!(
                "llama.cpp could not load packaged CPU backend module {}: {reason}",
                cpu.display()
            )
        })?;
        if include_accelerators {
            for path in paths.iter().filter(|path| {
                let name = path.file_name().and_then(|name| name.to_str());
                name != Some(cpu_base_name.as_str())
                    && !name.is_some_and(|name| name.starts_with(&cpu_variant_prefix))
            }) {
                let _ = loader(path);
            }
        }
        Ok(())
    })
}

/// Loads only packaged CPU sibling modules through an injected native loader.
pub fn load_packaged_cpu_modules_from_executable_with<S, L>(
    executable: &Path,
    score_cpu_variant: S,
    loader: L,
) -> Result<(), String>
where
    S: FnMut(&Path) -> Result<i32, String>,
    L: FnMut(&Path) -> Result<(), String>,
{
    load_packaged_modules_with(
        executable,
        &DeclaredBackends::build(),
        false,
        score_cpu_variant,
        loader,
    )
}

#[cfg(feature = "dynamic-backends")]
fn score_native_cpu_variant(path: &Path) -> Result<i32, String> {
    // Pinned CPU variant modules export this no-argument C symbol before registration.
    unsafe {
        let library = libloading::Library::new(path).map_err(|err| err.to_string())?;
        let score = library
            .get::<unsafe extern "C" fn() -> std::ffi::c_int>(b"ggml_backend_score\0")
            .map_err(|err| err.to_string())?;
        Ok(score())
    }
}

#[cfg(feature = "dynamic-backends")]
fn load_native_module(path: &Path) -> Result<(), String> {
    let path_text = path
        .to_str()
        .ok_or_else(|| "backend module path is not valid UTF-8".to_string())?;
    let path_text = std::ffi::CString::new(path_text)
        .map_err(|_| "backend module path contains a NUL byte".to_string())?;
    let loaded = unsafe { llama_cpp_sys_2::ggml_backend_load(path_text.as_ptr()) };
    if loaded.is_null() {
        Err("native registry load returned null".to_string())
    } else {
        Ok(())
    }
}

/// Loads only explicitly declared sibling modules and never consults loader search paths.
#[cfg(feature = "dynamic-backends")]
pub fn load_packaged_modules_from_executable(executable: &Path) -> Result<(), String> {
    let declared = DeclaredBackends::build();
    load_packaged_modules_with(
        executable,
        &declared,
        true,
        score_native_cpu_variant,
        load_native_module,
    )
}

/// Loads only packaged CPU sibling modules for a forced-CPU run.
#[cfg(feature = "dynamic-backends")]
pub fn load_packaged_cpu_modules_from_executable(executable: &Path) -> Result<(), String> {
    load_packaged_cpu_modules_from_executable_with(
        executable,
        score_native_cpu_variant,
        load_native_module,
    )
}

/// Static builds have no sibling modules to load.
#[cfg(not(feature = "dynamic-backends"))]
pub fn load_packaged_modules_from_executable(_executable: &Path) -> Result<(), String> {
    Ok(())
}

/// Static builds link their CPU backend and have no sibling module to load.
#[cfg(not(feature = "dynamic-backends"))]
pub fn load_packaged_cpu_modules_from_executable(_executable: &Path) -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
mod task3_tests {
    use super::*;
    use std::cell::Cell;
    use std::time::Duration;

    fn declared() -> DeclaredBackends {
        DeclaredBackends {
            metal: true,
            vulkan: true,
            cuda: true,
            rocm: false,
            dynamic_backends: true,
        }
    }

    fn context<'a>() -> SelectionContext<'a> {
        SelectionContext {
            sidecar_version: "0.1.0",
            model_sha256: "model-a",
            native_build_identity: "native-a",
            packaged_backend_identity: "metal+vulkan+cuda",
        }
    }

    fn device(backend: &str, index: usize, description: &str) -> RuntimeDevice {
        RuntimeDevice {
            backend: backend.to_string(),
            index,
            name: format!("{backend}{index}"),
            description: description.to_string(),
            memory_total: 1024,
            driver: "driver-a".to_string(),
        }
    }

    fn discovery(devices: &[RuntimeDevice]) -> Discovery {
        discover_candidates(&declared(), devices)
    }

    fn timing(batch_1_ms: u64, batch_16_ms: u64) -> ProbeTiming {
        ProbeTiming {
            batch_1: Duration::from_millis(batch_1_ms),
            batch_16: Duration::from_millis(batch_16_ms),
        }
    }

    #[test]
    fn declared_runtime_intersection_filters_transitive_metal_and_unknown_backends() {
        let package = DeclaredBackends {
            metal: false,
            vulkan: true,
            cuda: false,
            rocm: true,
            dynamic_backends: true,
        };
        let found = discover_candidates(
            &package,
            &[
                device("Metal", 1, "Apple GPU"),
                device("Vulkan", 2, "Vulkan GPU"),
                device("CUDA", 3, "CUDA GPU"),
                device("ROCm", 4, "AMD GPU"),
                device("SYCL", 5, "Intel GPU"),
            ],
        );

        assert_eq!(
            found
                .candidates
                .iter()
                .map(|candidate| candidate.backend.as_str())
                .collect::<Vec<_>>(),
            vec![CPU, VULKAN]
        );
        assert_eq!(
            found.capabilities,
            BackendCapabilities {
                cpu: true,
                vulkan: true,
                ..Default::default()
            }
        );
    }

    #[test]
    fn forced_cpu_loads_only_cpu_runtime_and_skips_discovery_benchmark_and_cache() {
        let dir = tempfile::tempdir().expect("tempdir");
        let executable = dir.path().join("julie-semantic-sidecar");
        std::fs::write(&executable, b"").expect("executable");
        for backend in ["cpu-unsupported", "cpu-good", "cpu-best", METAL, VULKAN] {
            std::fs::write(dir.path().join(module_file_name(backend)), b"module").expect("module");
        }
        let scored = std::cell::RefCell::new(Vec::new());
        let loaded = std::cell::RefCell::new(Vec::new());
        load_packaged_cpu_modules_from_executable_with(
            &executable,
            |path| {
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .expect("module name");
                scored.borrow_mut().push(name.to_string());
                Ok(if name.contains("unsupported") {
                    0
                } else if name.contains("best") {
                    20
                } else {
                    10
                })
            },
            |path| {
                loaded.borrow_mut().push(
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .expect("module name")
                        .to_string(),
                );
                Ok(())
            },
        )
        .expect("cpu runtime");
        let selected = select_runtime_with(
            dir.path(),
            context(),
            Some("CPU"),
            || panic!("cpu force must skip discovery"),
            |_| panic!("cpu force must skip probes"),
        )
        .expect("selection");

        assert_eq!(scored.borrow().len(), 3);
        assert_eq!(loaded.into_inner(), vec![module_file_name("cpu-best")]);
        assert_eq!(selected.selection, Selection::cpu());
        assert_eq!(selected.capabilities, BackendCapabilities::cpu_only());
        assert!(!dir.path().join(SELECTION_CACHE_FILE).exists());
    }

    #[test]
    fn forced_backend_probes_cpu_and_only_the_requested_accelerator() {
        let dir = tempfile::tempdir().expect("tempdir");
        let seen = std::cell::RefCell::new(Vec::new());
        let selected = select_runtime_with(
            dir.path(),
            context(),
            Some("VULKAN"),
            || {
                Ok(discovery(&[
                    device("Metal", 1, "Apple GPU"),
                    device("Vulkan", 2, "Vulkan GPU"),
                ]))
            },
            |candidate| {
                seen.borrow_mut().push(candidate.backend.clone());
                Ok(if candidate.backend == CPU {
                    timing(20, 200)
                } else {
                    timing(10, 100)
                })
            },
        )
        .expect("selection");

        assert_eq!(seen.into_inner(), vec![CPU, VULKAN]);
        assert_eq!(selected.selection.resolved, VULKAN);
        assert_eq!(selected.device_index, Some(2));
    }

    #[test]
    fn unknown_and_unavailable_forced_values_stay_ready_on_cpu_with_a_reason() {
        for forced in ["sycl", "cuda"] {
            let dir = tempfile::tempdir().expect("tempdir");
            let selected = select_runtime_with(
                dir.path(),
                context(),
                Some(forced),
                || Ok(discovery(&[device("Vulkan", 2, "Vulkan GPU")])),
                |candidate| {
                    assert_eq!(candidate.backend, CPU);
                    Ok(timing(20, 200))
                },
            )
            .expect("selection");
            assert_eq!(selected.selection.requested, forced);
            assert_eq!(selected.selection.resolved, CPU);
            assert!(selected.selection.degraded_reason.is_some());
        }
    }

    #[test]
    fn failed_slower_and_tied_accelerators_fall_back_to_cpu() {
        for accelerator in [
            Err("probe failed"),
            Ok(timing(21, 200)),
            Ok(timing(20, 200)),
        ] {
            let dir = tempfile::tempdir().expect("tempdir");
            let selected = select_runtime_with(
                dir.path(),
                context(),
                None,
                || Ok(discovery(&[device("Vulkan", 2, "Vulkan GPU")])),
                |candidate| {
                    if candidate.backend == CPU {
                        Ok(timing(20, 200))
                    } else {
                        accelerator.map_err(str::to_string)
                    }
                },
            )
            .expect("selection");
            assert_eq!(selected.selection.resolved, CPU);
            assert!(!selected.selection.accelerated);
            assert!(selected.selection.degraded_reason.is_some());
            assert!(!selected.capabilities.vulkan || accelerator.is_ok());
        }
    }

    #[test]
    fn cpu_probe_failure_is_fatal() {
        let dir = tempfile::tempdir().expect("tempdir");
        let error = select_runtime_with(
            dir.path(),
            context(),
            None,
            || Ok(discovery(&[device("Vulkan", 2, "Vulkan GPU")])),
            |candidate| {
                if candidate.backend == CPU {
                    Err("cpu inference failed".to_string())
                } else {
                    Ok(timing(10, 100))
                }
            },
        )
        .expect_err("cpu probe failure");

        assert!(error.contains("cpu probe failed"));
        assert!(!dir.path().join(SELECTION_CACHE_FILE).exists());
    }

    #[test]
    fn every_cache_identity_component_invalidates_independently() {
        let variants = [
            SelectionContext {
                sidecar_version: "0.2.0",
                ..context()
            },
            SelectionContext {
                model_sha256: "model-b",
                ..context()
            },
            SelectionContext {
                native_build_identity: "native-b",
                ..context()
            },
            SelectionContext {
                packaged_backend_identity: "vulkan",
                ..context()
            },
        ];
        for variant in variants {
            let dir = tempfile::tempdir().expect("tempdir");
            let probes = Cell::new(0);
            for current in [context(), variant] {
                let selected = select_runtime_with(
                    dir.path(),
                    current,
                    None,
                    || Ok(discovery(&[device("Vulkan", 2, "Vulkan GPU")])),
                    |candidate| {
                        probes.set(probes.get() + 1);
                        Ok(if candidate.backend == CPU {
                            timing(20, 200)
                        } else {
                            timing(10, 100)
                        })
                    },
                )
                .expect("selection");
                assert_eq!(selected.selection.resolved, VULKAN);
            }
            assert_eq!(probes.get(), 4);
        }
    }

    #[test]
    fn gpu_and_driver_identity_each_invalidate_the_cache() {
        for changed in [
            device("Vulkan", 2, "Other GPU"),
            RuntimeDevice {
                driver: "driver-b".to_string(),
                ..device("Vulkan", 2, "Vulkan GPU")
            },
        ] {
            let dir = tempfile::tempdir().expect("tempdir");
            let probes = Cell::new(0);
            for gpu in [device("Vulkan", 2, "Vulkan GPU"), changed] {
                select_runtime_with(
                    dir.path(),
                    context(),
                    None,
                    || Ok(discovery(&[gpu])),
                    |candidate| {
                        probes.set(probes.get() + 1);
                        Ok(if candidate.backend == CPU {
                            timing(20, 200)
                        } else {
                            timing(10, 100)
                        })
                    },
                )
                .expect("selection");
            }
            assert_eq!(probes.get(), 4);
        }
    }

    #[test]
    fn alternating_models_keep_both_cached_verdicts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let probes = Cell::new(0);
        for model_sha256 in ["model-a", "model-b", "model-a", "model-b"] {
            select_runtime_with(
                dir.path(),
                SelectionContext {
                    model_sha256,
                    ..context()
                },
                None,
                || Ok(discovery(&[device("Vulkan", 2, "Vulkan GPU")])),
                |candidate| {
                    probes.set(probes.get() + 1);
                    Ok(if candidate.backend == CPU {
                        timing(20, 200)
                    } else {
                        timing(10, 100)
                    })
                },
            )
            .expect("selection");
        }
        assert_eq!(probes.get(), 4);
    }

    #[test]
    fn corrupt_cache_recovers_by_probing_and_rewriting() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join(SELECTION_CACHE_FILE), "not json").expect("seed");
        let probes = Cell::new(0);
        let selected = select_runtime_with(
            dir.path(),
            context(),
            None,
            || Ok(discovery(&[device("Vulkan", 2, "Vulkan GPU")])),
            |candidate| {
                probes.set(probes.get() + 1);
                Ok(if candidate.backend == CPU {
                    timing(20, 200)
                } else {
                    timing(10, 100)
                })
            },
        )
        .expect("selection");
        assert_eq!(selected.selection.resolved, VULKAN);
        assert_eq!(probes.get(), 2);
        assert!(serde_json::from_slice::<serde_json::Value>(
            &std::fs::read(dir.path().join(SELECTION_CACHE_FILE)).expect("cache")
        )
        .is_ok());
    }

    #[test]
    fn loader_uses_the_executable_parent_and_reports_missing_modules() {
        let dir = tempfile::tempdir().expect("tempdir");
        let exe = dir.path().join("julie-semantic-sidecar");
        std::fs::write(&exe, b"").expect("exe");
        let loaded = std::cell::RefCell::new(None);
        load_modules_from_executable_with(&exe, |path| {
            *loaded.borrow_mut() = Some(path.to_path_buf());
            Err("module missing".to_string())
        })
        .expect_err("missing module");
        assert_eq!(loaded.into_inner().as_deref(), Some(dir.path()));
    }

    #[test]
    fn packaged_module_discovery_ignores_arbitrary_environment_and_undeclared_modules() {
        let dir = tempfile::tempdir().expect("tempdir");
        let outside = tempfile::tempdir().expect("outside");
        let vulkan = dir.path().join(module_file_name(VULKAN));
        let cuda = dir.path().join(module_file_name(CUDA));
        let arbitrary = outside.path().join(module_file_name(VULKAN));
        for path in [&vulkan, &cuda, &arbitrary] {
            std::fs::write(path, b"module").expect("module");
        }
        std::env::set_var("GGML_BACKEND_PATH", outside.path());
        let package = DeclaredBackends {
            metal: false,
            vulkan: true,
            cuda: false,
            rocm: false,
            dynamic_backends: true,
        };
        let found = packaged_module_paths(dir.path(), &package).expect("paths");
        std::env::remove_var("GGML_BACKEND_PATH");

        assert_eq!(found, vec![vulkan]);
    }

    #[test]
    fn rocm_feature_does_not_load_hip_or_invent_a_rocm_module() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cpu = dir.path().join(module_file_name("cpu-test"));
        let hip = dir.path().join(module_file_name("hip"));
        let rocm = dir.path().join(module_file_name("rocm"));
        for path in [&cpu, &hip, &rocm] {
            std::fs::write(path, b"module").expect("module");
        }
        let package = DeclaredBackends {
            metal: false,
            vulkan: false,
            cuda: false,
            rocm: true,
            dynamic_backends: true,
        };

        assert_eq!(
            packaged_module_paths(dir.path(), &package).expect("paths"),
            vec![cpu]
        );
    }

    #[test]
    fn zero_scoring_cpu_variants_fall_back_to_the_base_cpu_module() {
        let dir = tempfile::tempdir().expect("tempdir");
        let executable = dir.path().join("julie-semantic-sidecar");
        std::fs::write(&executable, b"").expect("executable");
        for backend in [CPU, "cpu-unsupported-a", "cpu-unsupported-b"] {
            std::fs::write(dir.path().join(module_file_name(backend)), b"module").expect("module");
        }
        let loaded = std::cell::RefCell::new(Vec::new());

        load_packaged_cpu_modules_from_executable_with(
            &executable,
            |_| Ok(0),
            |path| {
                loaded.borrow_mut().push(path.to_path_buf());
                Ok(())
            },
        )
        .expect("base cpu fallback");

        assert_eq!(
            loaded.into_inner(),
            vec![dir.path().join(module_file_name(CPU))]
        );
    }

    #[test]
    fn full_loader_registers_one_cpu_winner_and_each_declared_stable_accelerator() {
        let dir = tempfile::tempdir().expect("tempdir");
        let executable = dir.path().join("julie-semantic-sidecar");
        std::fs::write(&executable, b"").expect("executable");
        for backend in ["cpu-good", "cpu-best", METAL, VULKAN, CUDA, "hip"] {
            std::fs::write(dir.path().join(module_file_name(backend)), b"module").expect("module");
        }
        let declared = DeclaredBackends {
            metal: true,
            vulkan: false,
            cuda: true,
            rocm: true,
            dynamic_backends: true,
        };
        let loaded = std::cell::RefCell::new(Vec::new());

        load_packaged_modules_with(
            &executable,
            &declared,
            true,
            |path| {
                Ok(if path.ends_with(module_file_name("cpu-best")) {
                    20
                } else {
                    10
                })
            },
            |path| {
                loaded.borrow_mut().push(
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .expect("module name")
                        .to_string(),
                );
                Ok(())
            },
        )
        .expect("modules");

        assert_eq!(
            loaded.into_inner(),
            vec![
                module_file_name("cpu-best"),
                module_file_name(CUDA),
                module_file_name(METAL),
            ]
        );
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn native_backend_modules_use_the_cmake_module_suffix() {
        assert_eq!(module_file_name(METAL), "libggml-metal.so");
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_executable_parent_is_rejected_without_calling_the_loader() {
        use std::os::unix::ffi::OsStringExt;
        let path = PathBuf::from(std::ffi::OsString::from_vec(vec![
            b'/', b't', b'm', b'p', b'/', 0xff, b'/', b'x',
        ]));
        let called = Cell::new(false);
        let error = load_modules_from_executable_with(&path, |_| {
            called.set(true);
            Ok(())
        })
        .expect_err("non utf8 path");
        assert!(error.contains("UTF-8"));
        assert!(!called.get());
    }
}
