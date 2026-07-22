//! Assembly of the `health` result: readiness, dims, capabilities, and load policy.
//!
//! The field list is the one frozen in `semantic-sidecar-protocol-v1.md` § Methods
//! (`health`) and § Health metadata. Assembly is deliberately self-contained: it takes
//! plain facts and returns a [`serde_json::Value`], so the protocol loop and the engine
//! can be tested independently of each other.

use serde_json::{json, Map, Value};

use crate::manifest::ModelPin;

/// Largest `texts` array `embed_batch` accepts.
///
/// The contract bounds converge batches at 250 texts per RPC and conformance row B5
/// asserts a 250-text batch answers within the request budget.
pub const MAX_BATCH_ITEMS: usize = 250;

/// Largest single request line accepted, in bytes.
///
/// The contract names the field but fixes no value; 32 MiB comfortably admits a
/// full 250-item batch of budget-length inputs while still bounding a hostile line.
pub const MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;

/// Version of the prompt-template set applied to inputs.
pub const INSTRUCTION_POLICY_VERSION: u64 = 1;

/// The only normalization v1 emits.
pub const NORMALIZATION: &str = "l2";

/// Reason reported when a backend other than the requested one loaded and the engine
/// supplied no more specific explanation.
const BACKEND_FALLBACK_REASON: &str = "requested_backend_unavailable";

/// Exact reason string a `health` call reports when the model is absent from the cache.
pub const MODEL_NOT_PREPARED: &str = "model_not_prepared";

/// Backend availability, spanning both the torch-compatible reference keys and the
/// llama.cpp backends v1 adds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BackendCapabilities {
    /// Always available — CPU is the floor in every build.
    pub cpu: bool,
    /// NVIDIA CUDA.
    pub cuda: bool,
    /// Windows DirectML; false for a llama.cpp shim, never omitted.
    pub directml: bool,
    /// Apple GPU under the torch spelling.
    pub mps: bool,
    /// Apple GPU under the llama.cpp spelling.
    pub metal: bool,
    /// Vulkan.
    pub vulkan: bool,
}

impl BackendCapabilities {
    /// The guaranteed floor before any accelerator proves usable.
    pub fn cpu_only() -> Self {
        Self {
            cpu: true,
            ..Self::default()
        }
    }

    fn to_value(self) -> Value {
        json!({
            "cpu": { "available": self.cpu },
            "cuda": { "available": self.cuda },
            "directml": { "available": self.directml },
            "mps": { "available": self.mps },
            "metal": { "available": self.metal },
            "vulkan": { "available": self.vulkan },
        })
    }
}

/// Runtime and backend facts the engine reports about the process it actually built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineFacts {
    /// Runtime family name.
    pub runtime: String,
    /// Resolved device identity.
    pub device: String,
    /// Backend the cached benchmark choice asked for.
    pub requested_backend: String,
    /// Backend that actually loaded.
    pub resolved_backend: String,
    /// Whether the resolved device is not CPU.
    pub accelerated: bool,
    /// Engine-supplied degradation explanation, when it has one.
    pub degraded_reason: Option<String>,
    /// Backends this build can use.
    pub capabilities: BackendCapabilities,
    /// Vendored llama.cpp build tag.
    pub llama_cpp_build: String,
}

/// Whether the model is loadable, and at what served dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelState {
    /// The model is present and can embed.
    Ready {
        /// Pin the loaded weights came from.
        pin: &'static ModelPin,
        /// Lane actually served; becomes the `dims` every response must match.
        dims: usize,
    },
    /// The model is absent from the cache; `prepare` has not run.
    NotPrepared {
        /// Pin that would be served once prepared.
        pin: &'static ModelPin,
    },
}

impl ModelState {
    fn pin(&self) -> &'static ModelPin {
        match self {
            ModelState::Ready { pin, .. } | ModelState::NotPrepared { pin } => pin,
        }
    }

    fn dims(&self) -> Option<usize> {
        match self {
            ModelState::Ready { dims, .. } => Some(*dims),
            ModelState::NotPrepared { .. } => None,
        }
    }
}

/// Request-shape limits the sidecar advertises so callers can size their batches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Largest `texts` array accepted.
    pub max_batch_items: usize,
    /// Largest single request line in bytes.
    pub max_request_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_batch_items: MAX_BATCH_ITEMS,
            max_request_bytes: MAX_REQUEST_BYTES,
        }
    }
}

/// Assembles the `health` result.
///
/// `dims` is emitted only for a ready model; a consumer requires it only there. The
/// degradation invariant is enforced here rather than trusted from the engine: a
/// resolved backend differing from the requested one always carries a non-null reason,
/// mirrored into `load_policy` alongside `accelerated`.
pub fn build(
    model: &ModelState,
    engine: &EngineFacts,
    limits: Limits,
    sidecar_version: &str,
) -> Value {
    let pin = model.pin();
    let ready = model.dims().is_some();
    let degraded_reason = resolve_degraded_reason(model, engine);

    let mut result = Map::new();
    result.insert("ready".to_string(), json!(ready));
    if let Some(dims) = model.dims() {
        result.insert("dims".to_string(), json!(dims));
    }
    result.insert("model_id".to_string(), json!(pin.id));
    result.insert("model_sha256".to_string(), json!(pin.sha256));
    result.insert("model_revision".to_string(), json!(pin.model_revision));
    result.insert("runtime".to_string(), json!(engine.runtime));
    result.insert("device".to_string(), json!(engine.device));
    result.insert(
        "resolved_backend".to_string(),
        json!(engine.resolved_backend),
    );
    result.insert("accelerated".to_string(), json!(engine.accelerated));
    result.insert("degraded_reason".to_string(), json!(degraded_reason));
    result.insert("capabilities".to_string(), engine.capabilities.to_value());
    result.insert(
        "load_policy".to_string(),
        json!({
            "requested_device_backend": engine.requested_backend,
            "resolved_device_backend": engine.resolved_backend,
            "accelerated": engine.accelerated,
            "degraded_reason": degraded_reason,
        }),
    );
    result.insert("pooling".to_string(), json!(pin.pooling.as_str()));
    result.insert("normalization".to_string(), json!(NORMALIZATION));
    result.insert(
        "instruction_policy_version".to_string(),
        json!(INSTRUCTION_POLICY_VERSION),
    );
    result.insert("max_text_tokens".to_string(), json!(pin.max_text_tokens));
    result.insert("max_batch_items".to_string(), json!(limits.max_batch_items));
    result.insert(
        "max_request_bytes".to_string(),
        json!(limits.max_request_bytes),
    );
    result.insert("native_dims".to_string(), json!(pin.native_dims));
    result.insert("mrl_lanes".to_string(), json!(pin.mrl_lanes));
    result.insert("llama_cpp_build".to_string(), json!(engine.llama_cpp_build));
    result.insert("sidecar_version".to_string(), json!(sidecar_version));

    Value::Object(result)
}

fn resolve_degraded_reason(model: &ModelState, engine: &EngineFacts) -> Option<String> {
    if matches!(model, ModelState::NotPrepared { .. }) {
        return Some(MODEL_NOT_PREPARED.to_string());
    }
    if let Some(reason) = &engine.degraded_reason {
        return Some(reason.clone());
    }
    if engine.requested_backend != engine.resolved_backend {
        return Some(BACKEND_FALLBACK_REASON.to_string());
    }
    None
}
