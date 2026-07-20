//! Backend selection: which llama.cpp device the sidecar loads on, and why.
//!
//! `semantic-sidecar-protocol-v1.md` § Backend selection. CPU is the floor in every build
//! and is never unavailable. On the first start for a given cache key the sidecar
//! micro-benchmarks the available backends and caches the winner, keyed by shim version +
//! model sha256 + GPU identity + driver identity; any component changing re-runs the
//! benchmark. An accelerated backend losing to CPU is a normal outcome — `ready: true`,
//! `accelerated: false`, and a `degraded_reason` naming the result.
//!
//! The CPU-only build (`llama-cpp-2` with `default-features = false` and no `metal`
//! feature) compiles no accelerated backend, so its benchmark has exactly one candidate and
//! every selection resolves to CPU. The `metal` feature is the accelerated macOS build: it
//! supplies a real [`MachineIdentity`] (GPU brand + OS build) and resolves `metal` without
//! probing — on Apple Silicon the unified-memory Metal path has no known CPU-wins case for
//! the pinned encoders, so a micro-benchmark would only slow the first start. If a real
//! CPU-wins case appears, wire a probe into [`select_with`]'s benchmark closure; the cache
//! mechanics, the key, and the degradation shape already support it.
//! `JULIE_SIDECAR_FORCE_BACKEND=cpu` remains the operator escape hatch in every build.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Environment variable that pins the backend, bypassing benchmark and cache entirely.
pub const FORCE_BACKEND_ENV: &str = "JULIE_SIDECAR_FORCE_BACKEND";

/// Canonical name of the backend every build can serve.
pub const CPU: &str = "cpu";

/// Canonical name of the backend the `metal` feature compiles in.
pub const METAL: &str = "metal";

/// File the cached benchmark choice is written to, beside the model cache.
pub const SELECTION_CACHE_FILE: &str = "backend-selection.json";

/// Reason reported when an accelerated backend was asked for and this build has none.
pub const NO_ACCELERATED_BACKEND: &str = "cpu (no accelerated backend compiled)";

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

/// Supplies the GPU and driver identity that key the selection cache.
///
/// The CPU-only build answers `none`/`none`; an accelerated build queries its backend.
pub trait MachineIdentity {
    /// Reports the current machine's accelerator identity.
    fn identify(&self) -> Machine;
}

/// [`MachineIdentity`] for a build with no accelerated backend compiled in.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuOnlyMachine;

impl MachineIdentity for CpuOnlyMachine {
    fn identify(&self) -> Machine {
        Machine::none()
    }
}

/// [`MachineIdentity`] for the `metal` build: GPU brand plus OS build as the driver proxy.
///
/// On Apple platforms the Metal "driver" ships with the OS, so the kernel build string is
/// the component whose change should invalidate a cached selection. Both lookups fall back
/// to `unknown` rather than failing a start — an unidentifiable machine still embeds; it
/// just re-benchmarks when the identity becomes readable again.
#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalMachine;

#[cfg(feature = "metal")]
impl MachineIdentity for MetalMachine {
    fn identify(&self) -> Machine {
        Machine {
            gpu: sysctl_string("machdep.cpu.brand_string"),
            driver: sysctl_string("kern.osversion"),
        }
    }
}

#[cfg(feature = "metal")]
fn sysctl_string(name: &str) -> String {
    std::process::Command::new("sysctl")
        .args(["-n", name])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
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

/// Resolves the backend for `model_sha256` using this build's identity and benchmark.
///
/// This is the production entry point. The CPU-only build's benchmark has one candidate,
/// so the first start writes a `cpu` choice that every later start reads back; the `metal`
/// build resolves `metal` (see the module docs for why no probe runs today).
pub fn select(cache_dir: &Path, sidecar_version: &str, model_sha256: &str) -> Selection {
    select_with(
        cache_dir,
        sidecar_version,
        model_sha256,
        &build_machine(),
        forced_backend().as_deref(),
        |_machine| build_selection(),
    )
}

#[cfg(feature = "metal")]
fn build_machine() -> impl MachineIdentity {
    MetalMachine
}

#[cfg(not(feature = "metal"))]
fn build_machine() -> impl MachineIdentity {
    CpuOnlyMachine
}

/// The backend this build serves when nothing is forced and nothing is cached.
fn build_selection() -> Selection {
    if cfg!(feature = "metal") {
        Selection::new(METAL, METAL, None)
    } else {
        Selection::cpu()
    }
}

/// Resolves the backend from injected identity, override, and benchmark.
///
/// Order of authority: a forced backend short-circuits everything; otherwise a cache entry
/// whose key matches every component is returned without probing; otherwise `benchmark`
/// runs and its verdict is cached.
///
/// A forced choice is never cached and never reads the cache — it is an operator override,
/// not a benchmark result, and must not outlive the environment variable that set it.
pub fn select_with<M, B>(
    cache_dir: &Path,
    sidecar_version: &str,
    model_sha256: &str,
    machine: &M,
    forced: Option<&str>,
    benchmark: B,
) -> Selection
where
    M: MachineIdentity,
    B: FnOnce(&Machine) -> Selection,
{
    if let Some(forced) = forced {
        return force(forced);
    }

    let machine = machine.identify();
    let key = cache_key(sidecar_version, model_sha256, &machine);
    if let Some(cached) = read_cached(cache_dir) {
        if cached.key == key {
            return cached.into_selection();
        }
    }

    let selection = benchmark(&machine);
    write_cached(cache_dir, &key, &selection);
    selection
}

/// Applies `JULIE_SIDECAR_FORCE_BACKEND`.
///
/// `cpu` is always honourable, and so is a backend this build compiled. Anything else
/// names a backend this build did not compile, so CPU answers and the mismatch is
/// reported as a degradation rather than a failure.
fn force(requested: &str) -> Selection {
    if requested.eq_ignore_ascii_case(CPU) {
        return Selection::cpu();
    }
    if cfg!(feature = "metal") && requested.eq_ignore_ascii_case(METAL) {
        return Selection::new(METAL, METAL, None);
    }
    Selection::degraded_to_cpu(requested.to_ascii_lowercase(), NO_ACCELERATED_BACKEND)
}

/// Reads `JULIE_SIDECAR_FORCE_BACKEND`, treating an empty value as unset.
pub fn forced_backend() -> Option<String> {
    let value = std::env::var(FORCE_BACKEND_ENV).ok()?;
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Builds the cache key: shim version + model sha256 + GPU identity + driver identity.
///
/// Components are joined with a character none of them contains, so a change in any one
/// component always produces a different key.
pub fn cache_key(sidecar_version: &str, model_sha256: &str, machine: &Machine) -> String {
    format!(
        "{sidecar_version}|{model_sha256}|{}|{}",
        machine.gpu, machine.driver
    )
}

/// Directory the binary lives in, where an accelerated build's backend plugins sit.
///
/// llama.cpp's split-backend builds load `ggml-<backend>` shared libraries at runtime and
/// resolve them relative to the executable, not the working directory — so an accelerated
/// build (Task 8) points llama.cpp's backend loader here before the first probe. This
/// build compiles no accelerated backend and loads nothing dynamically; the helper exists
/// so the path rule is decided and testable now.
pub fn plugin_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()?
        .parent()
        .map(Path::to_path_buf)
}

/// The cached benchmark verdict, as written beside the model cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedChoice {
    key: String,
    requested: String,
    resolved: String,
    degraded_reason: Option<String>,
}

impl CachedChoice {
    fn into_selection(self) -> Selection {
        Selection::new(self.requested, self.resolved, self.degraded_reason)
    }
}

fn cache_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join(SELECTION_CACHE_FILE)
}

/// Reads the cached choice, treating a missing or unreadable file as no cache.
///
/// A corrupt file is not an error worth failing a start over: re-running the benchmark
/// costs a moment and rewrites the file correctly.
fn read_cached(cache_dir: &Path) -> Option<CachedChoice> {
    let raw = std::fs::read_to_string(cache_path(cache_dir)).ok()?;
    serde_json::from_str(&raw).ok()
}

/// Writes the choice, ignoring failures — a cache that cannot be written costs a benchmark
/// next start, which is not worth refusing to serve over.
fn write_cached(cache_dir: &Path, key: &str, selection: &Selection) {
    let record = CachedChoice {
        key: key.to_string(),
        requested: selection.requested.clone(),
        resolved: selection.resolved.clone(),
        degraded_reason: selection.degraded_reason.clone(),
    };
    let Ok(encoded) = serde_json::to_string(&record) else {
        return;
    };
    if std::fs::create_dir_all(cache_dir).is_err() {
        return;
    }
    let _ = std::fs::write(cache_path(cache_dir), encoded);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::{self, Limits, ModelState};
    use crate::manifest;
    use std::cell::Cell;

    /// Counts benchmark runs so a test can prove a cache hit skipped the probe.
    struct ProbeCounter(Cell<u32>);

    struct FixedMachine(Machine);

    impl MachineIdentity for FixedMachine {
        fn identify(&self) -> Machine {
            self.0.clone()
        }
    }

    fn machine(gpu: &str, driver: &str) -> FixedMachine {
        FixedMachine(Machine {
            gpu: gpu.to_string(),
            driver: driver.to_string(),
        })
    }

    fn counted<'a>(
        probes: &'a ProbeCounter,
        verdict: Selection,
    ) -> impl FnOnce(&Machine) -> Selection + 'a {
        move |_| {
            probes.0.set(probes.0.get() + 1);
            verdict
        }
    }

    fn probes() -> ProbeCounter {
        ProbeCounter(Cell::new(0))
    }

    #[test]
    fn a_forced_cpu_backend_short_circuits_the_benchmark_and_the_cache() {
        let dir = tempfile::tempdir().expect("tempdir");
        let selection = select_with(
            dir.path(),
            "0.1.0",
            "abc",
            &machine("gpu", "driver"),
            Some("cpu"),
            |_| panic!("a forced backend must not benchmark"),
        );
        assert_eq!(selection, Selection::cpu());
        assert!(!cache_path(dir.path()).exists());
    }

    #[cfg(not(feature = "metal"))]
    #[test]
    fn the_cpu_only_build_defaults_to_cpu() {
        assert_eq!(build_selection(), Selection::cpu());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn the_metal_build_defaults_to_metal_with_no_degradation() {
        let selection = build_selection();
        assert_eq!(selection.requested, METAL);
        assert_eq!(selection.resolved, METAL);
        assert!(selection.accelerated);
        assert_eq!(selection.degraded_reason, None);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn a_forced_metal_backend_is_honoured_in_the_metal_build() {
        let dir = tempfile::tempdir().expect("tempdir");
        let selection = select_with(
            dir.path(),
            "0.1.0",
            "abc",
            &machine("gpu", "driver"),
            Some("Metal"),
            |_| panic!("a forced backend must not benchmark"),
        );
        assert_eq!(selection.resolved, METAL);
        assert!(selection.accelerated);
        assert_eq!(selection.degraded_reason, None);
        assert!(!cache_path(dir.path()).exists());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn the_metal_machine_reports_a_non_none_identity() {
        let identity = MetalMachine.identify();
        assert_ne!(identity, Machine::none());
        assert!(!identity.gpu.is_empty());
        assert!(!identity.driver.is_empty());
    }

    #[cfg(not(feature = "metal"))]
    #[test]
    fn a_forced_accelerated_backend_degrades_to_cpu_in_this_build() {
        let dir = tempfile::tempdir().expect("tempdir");
        let selection = select_with(
            dir.path(),
            "0.1.0",
            "abc",
            &machine("gpu", "driver"),
            Some("Metal"),
            |_| panic!("a forced backend must not benchmark"),
        );
        assert_eq!(selection.requested, "metal");
        assert_eq!(selection.resolved, CPU);
        assert!(!selection.accelerated);
        assert_eq!(
            selection.degraded_reason.as_deref(),
            Some(NO_ACCELERATED_BACKEND)
        );
        assert!(!cache_path(dir.path()).exists());
    }

    #[test]
    fn the_first_start_benchmarks_and_the_second_reads_the_cache() {
        let dir = tempfile::tempdir().expect("tempdir");
        let runs = probes();
        let first = select_with(
            dir.path(),
            "0.1.0",
            "abc",
            &machine("gpu", "driver"),
            None,
            counted(&runs, Selection::cpu()),
        );
        let second = select_with(
            dir.path(),
            "0.1.0",
            "abc",
            &machine("gpu", "driver"),
            None,
            counted(&runs, Selection::cpu()),
        );
        assert_eq!(first, Selection::cpu());
        assert_eq!(second, first);
        assert_eq!(runs.0.get(), 1);
    }

    #[test]
    fn a_cached_degraded_choice_round_trips_every_field() {
        let dir = tempfile::tempdir().expect("tempdir");
        let runs = probes();
        let verdict =
            Selection::degraded_to_cpu("vulkan", "vulkan lost the micro-benchmark to cpu");
        let written = select_with(
            dir.path(),
            "0.1.0",
            "abc",
            &machine("gpu", "driver"),
            None,
            counted(&runs, verdict.clone()),
        );
        let read_back = select_with(
            dir.path(),
            "0.1.0",
            "abc",
            &machine("gpu", "driver"),
            None,
            |_| panic!("a cached choice must skip the probe"),
        );
        assert_eq!(written, verdict);
        assert_eq!(read_back, verdict);
        assert_eq!(runs.0.get(), 1);
    }

    #[test]
    fn every_key_component_invalidates_the_cached_choice() {
        let components: [(&str, &str, &str, &str); 4] = [
            ("0.2.0", "abc", "gpu", "driver"),
            ("0.1.0", "def", "gpu", "driver"),
            ("0.1.0", "abc", "other-gpu", "driver"),
            ("0.1.0", "abc", "gpu", "other-driver"),
        ];
        for (version, sha, gpu, driver) in components {
            let dir = tempfile::tempdir().expect("tempdir");
            let runs = probes();
            select_with(
                dir.path(),
                "0.1.0",
                "abc",
                &machine("gpu", "driver"),
                None,
                counted(&runs, Selection::cpu()),
            );
            select_with(
                dir.path(),
                version,
                sha,
                &machine(gpu, driver),
                None,
                counted(&runs, Selection::cpu()),
            );
            assert_eq!(
                runs.0.get(),
                2,
                "changing ({version}, {sha}, {gpu}, {driver}) must re-benchmark"
            );
        }
    }

    #[test]
    fn a_corrupt_cache_file_re_runs_the_benchmark() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(cache_path(dir.path()), "not json").expect("seed");
        let runs = probes();
        let selection = select_with(
            dir.path(),
            "0.1.0",
            "abc",
            &machine("gpu", "driver"),
            None,
            counted(&runs, Selection::cpu()),
        );
        assert_eq!(selection, Selection::cpu());
        assert_eq!(runs.0.get(), 1);
    }

    #[test]
    fn cache_key_changes_with_every_component() {
        let base = cache_key("0.1.0", "abc", &Machine::none());
        assert_ne!(base, cache_key("0.2.0", "abc", &Machine::none()));
        assert_ne!(base, cache_key("0.1.0", "def", &Machine::none()));
        assert_ne!(
            base,
            cache_key(
                "0.1.0",
                "abc",
                &Machine {
                    gpu: "other".to_string(),
                    driver: "none".to_string()
                }
            )
        );
        assert_ne!(
            base,
            cache_key(
                "0.1.0",
                "abc",
                &Machine {
                    gpu: "none".to_string(),
                    driver: "other".to_string()
                }
            )
        );
    }

    #[test]
    fn a_cpu_only_build_reports_no_accelerator_identity() {
        assert_eq!(CpuOnlyMachine.identify(), Machine::none());
    }

    #[test]
    fn the_plugin_dir_is_the_directory_holding_the_running_binary() {
        let exe = std::env::current_exe().expect("current exe");
        assert_eq!(plugin_dir().as_deref(), exe.parent());
    }

    #[test]
    fn a_requested_but_unavailable_backend_stays_ready_and_degraded_in_health() {
        let selection = Selection::degraded_to_cpu("vulkan", NO_ACCELERATED_BACKEND);
        let pin = manifest::default_model();
        let facts = crate::health::EngineFacts {
            runtime: "llama.cpp".to_string(),
            device: selection.resolved.clone(),
            requested_backend: selection.requested.clone(),
            resolved_backend: selection.resolved.clone(),
            accelerated: selection.accelerated,
            degraded_reason: selection.degraded_reason.clone(),
            capabilities: crate::health::BackendCapabilities {
                cpu: true,
                ..Default::default()
            },
            llama_cpp_build: "test".to_string(),
        };
        let value = health::build(
            &ModelState::Ready {
                pin,
                dims: pin.serve_dims,
            },
            &facts,
            Limits::default(),
            "0.1.0",
        );
        assert_eq!(value["ready"], true);
        assert_eq!(value["accelerated"], false);
        assert_eq!(value["degraded_reason"], NO_ACCELERATED_BACKEND);
        assert_eq!(value["load_policy"]["requested_device_backend"], "vulkan");
        assert_eq!(value["load_policy"]["resolved_device_backend"], "cpu");
        assert_eq!(value["load_policy"]["accelerated"], false);
        assert_eq!(
            value["load_policy"]["degraded_reason"],
            NO_ACCELERATED_BACKEND
        );
    }
}
