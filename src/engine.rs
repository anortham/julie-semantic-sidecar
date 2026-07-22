//! The llama.cpp-backed [`EmbedEngine`] implementation.
//!
//! Everything model-specific is read from the [`ModelPin`]: pooling mode, instruction
//! templates, EOS marker, token budget, and the served MRL lane. The pin is the single
//! source of truth, so adding a model is a manifest change rather than an engine change.
//!
//! The batch-isolation algorithm of `semantic-sidecar-protocol-v1.md` § Per-item failure
//! isolation lives in [`isolate`], which is generic over the encode operation so it is
//! testable without a model — the engine supplies the real closure.

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::{list_llama_ggml_backend_devices, DecodeError, EncodeError, TokenToStringError};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::backend_select::{self, Selection};
use crate::engine_trait::{EmbedEngine, EmbedOutput, EngineError, Role};
use crate::health::{self, BackendCapabilities, EngineFacts, Limits, ModelState};
use crate::manifest::{ModelPin, Pooling};
use crate::{sanitize, truncate, VERSION};

/// Identity of the llama.cpp the binary is linked against.
///
/// The vendored tree ships without git metadata, so no upstream build number is
/// recoverable at compile time; the crate pin plus the `llama-cpp-rs` revision it was
/// published from is the reproducible identity, and both are exact-pinned in `Cargo.toml`.
pub const LLAMA_CPP_BUILD: &str = "llama-cpp-2 0.1.151 (llama-cpp-rs 7f0a0d95, vendored llama.cpp)";

/// Runtime family reported through `health`.
const RUNTIME: &str = "llama.cpp";

/// Probe used to measure the tokenizer's per-input special-token overhead.
const OVERHEAD_PROBE: &str = "probe";

/// Ceiling on tokens encoded in a single llama context.
///
/// Bounds peak memory: a 250-item batch of budget-length Qwen3 inputs would otherwise ask
/// for millions of context tokens at once. Inputs are grouped under this ceiling, and a
/// single input longer than it still gets a context sized to fit it exactly.
const MAX_TOKENS_PER_ENCODE: usize = 16_384;

/// Bytes read per digest chunk when the cached model file is verified at load.
const DIGEST_CHUNK_BYTES: usize = 1024 * 1024;

/// Ceiling on tokens grouped into one `cls`-pooled encode.
///
/// A non-causal encoder must hold its whole group in a single micro-batch, so this bounds
/// the micro-batch too. bge's entire 512-token budget fits well inside it.
const MAX_CLS_GROUP_TOKENS: usize = 2_048;

/// Ceiling on the physical micro-batch of a `last`-pooled (causal) decode.
///
/// Bounds the compute buffer for a worst-case Qwen3 input (`max_text_tokens` = 32768
/// tokens), measured on the f16 pin:
///
/// | ubatch | compute buffer | weights + KV + compute |
/// |--------|---------------|------------------------|
/// | 2048   | 1649 MiB      | 6369 MiB               |
/// | 1024   |  973 MiB      | 5693 MiB               |
/// | 512    |  634 MiB      | 5354 MiB               |
/// | 256    |  465 MiB      | 5185 MiB               |
///
/// The weights (1136 MiB) and KV cache (3584 MiB: 28 layers x 8 kv heads x 256 f16 K+V
/// bytes x 32768 cells) are fixed by the frozen contract; the compute buffer is
/// `~296 MiB + 0.58 MiB * ubatch`.
///
/// History: the 512 -> 256 step-downs were chasing CI OOMs that this lever never caused.
/// The real term was the output buffer — `logits_all=true` in `encode_group` reserved a
/// vocab-width row per token (19 GiB at 32768 tokens), which macOS overcommits silently
/// but a 16 GiB Linux runner refuses. With outputs fixed to one row per sequence, every
/// row above fits CI with >9 GiB of headroom. The micro-batch width is a pure chunking
/// choice for a causal model — conformance cosines were bit-identical across the
/// 2048 -> 512 change — but it is NOT throughput-neutral on an accelerated backend: at
/// 256, a symbol-card batch decomposes into dozens of micro-dispatches and Metal spends
/// more time on launch overhead and CPU<->GPU sync than on math (measured 2026-07-20 on an
/// M2 Ultra: raising 256 -> 2048 was a large share of recovering the 8x gap between the
/// shipped sidecar and the P0 llama-server floor). 2048 keeps the worst-case total at
/// 6369 MiB, inside the 16 GiB CI runner with headroom.
const MAX_DECODE_UBATCH_TOKENS: usize = 2_048;

const BACKEND_PROBE_TEXTS: [&str; 16] = [
    "FullRebuildPromotion",
    "WorkspaceIndexProvider.OpenReadOnly",
    "l2_normalize_vector",
    "IHostedService::StartAsync",
    "public sealed record LeadershipEligibility(bool CanClaim, string? Reason)",
    "pub fn cosine(a: &[f32], b: &[f32]) -> f32",
    "def slice_renormalize(vec, dims): return vec[:dims]",
    "CREATE VIRTUAL TABLE vectors USING vec0(embedding int8[512]);",
    "how does the indexer decide which process holds the write lock",
    "where do we atomically swap the rebuilt database over the live artifact",
    "A force scan extracts into a rebuild database and atomically promotes it.",
    "Semantic retrieval is optional, local-first, and off by default.",
    "## Restore the cache\n\nRun the download and verify stages.",
    "# Tolerance policy\n\n| check | bar |\n|---|---|\n| cosine | >= 0.999 |",
    "```rust\nfn main() {}\n```",
    "该函数以原子方式将重建后的索引数据库替换为线上的工件文件。",
];

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelPlacement {
    n_gpu_layers: Option<u32>,
    device_indexes: Vec<usize>,
}

fn placement_for_candidate(candidate: &backend_select::BackendCandidate) -> ModelPlacement {
    if candidate.backend == backend_select::CPU {
        ModelPlacement {
            n_gpu_layers: Some(0),
            device_indexes: Vec::new(),
        }
    } else {
        ModelPlacement {
            n_gpu_layers: None,
            device_indexes: candidate.device_index.into_iter().collect(),
        }
    }
}

fn run_fixed_probes_with<F>(
    candidate: &backend_select::BackendCandidate,
    mut probe: F,
) -> Result<backend_select::ProbeTiming, String>
where
    F: FnMut(&backend_select::BackendCandidate, &[String]) -> Result<std::time::Duration, String>,
{
    let batch_1 = [BACKEND_PROBE_TEXTS[0].to_string()];
    let batch_16 = BACKEND_PROBE_TEXTS.map(str::to_string);
    probe(candidate, &batch_1)?;
    Ok(backend_select::ProbeTiming {
        batch_1: probe(candidate, &batch_1)?,
        batch_16: probe(candidate, &batch_16)?,
    })
}

fn forced_cpu_runtime_with<M, D, B>(
    cache_dir: &Path,
    executable: &Path,
    context: backend_select::SelectionContext<'_>,
    load_modules: M,
    discover: D,
    benchmark: B,
) -> Result<backend_select::RuntimeSelection, EngineError>
where
    M: FnOnce(&Path) -> Result<(), String>,
    D: FnOnce() -> backend_select::Discovery,
    B: FnMut(&backend_select::BackendCandidate) -> Result<backend_select::ProbeTiming, String>,
{
    load_modules(executable).map_err(|reason| EngineError::new("BackendDiscovery", reason))?;
    backend_select::select_runtime_with(
        cache_dir,
        context,
        Some(backend_select::CPU),
        discover,
        benchmark,
    )
    .map_err(|reason| EngineError::new("BackendProbe", reason))
}

/// A loaded llama.cpp embedding model serving one manifest pin.
pub struct LlamaEngine {
    pin: &'static ModelPin,
    backend: Rc<LlamaBackend>,
    model: LlamaModel,
    eos_reserve: usize,
    special_token_overhead: usize,
    selection: Selection,
    capabilities: BackendCapabilities,
}

impl LlamaEngine {
    /// Loads `pin`'s weights from `cache_dir`.
    ///
    /// The cached file is verified against the pin's sha256 before it is loaded: existence
    /// alone would let `serve` hand a truncated, swapped, or corrupt GGUF straight to
    /// llama.cpp. A streaming hash runs at roughly a second per gigabyte in a release
    /// build, which the contract's 120s cold-start budget absorbs. A digest mismatch is
    /// reported as `ModelNotPrepared` so `serve` answers the contract's unready health
    /// rather than dying.
    ///
    /// Backend initialisation, weight load, and the tokenizer probe all run inside a
    /// [`stdio_guard`](crate::stdio_guard) so llama.cpp's native chatter can never reach fd
    /// 1 — the contract's § Stdout purity obligation. The guard covers the backend
    /// selection benchmark for the same reason.
    pub fn load(pin: &'static ModelPin, cache_dir: &Path) -> Result<Self, EngineError> {
        let path = cache_dir.join(pin.file);
        if !path.is_file() {
            return Err(EngineError::new(
                "ModelNotPrepared",
                format!("{} is not in the cache; run `prepare`", path.display()),
            ));
        }
        verify_cached_digest(&path, pin.sha256)?;

        crate::stdio_guard::guarded(|| Self::load_guarded(pin, cache_dir, &path))
    }

    fn load_guarded(
        pin: &'static ModelPin,
        cache_dir: &Path,
        path: &Path,
    ) -> Result<Self, EngineError> {
        let forced = backend_select::forced_backend();
        if forced
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case(backend_select::CPU))
        {
            let executable = std::env::current_exe().map_err(|err| {
                EngineError::new(
                    "BackendDiscovery",
                    format!("cannot resolve executable: {err}"),
                )
            })?;
            let runtime = forced_cpu_runtime_with(
                cache_dir,
                &executable,
                backend_select::SelectionContext {
                    sidecar_version: VERSION,
                    model_sha256: pin.sha256,
                    native_build_identity: backend_select::NATIVE_BUILD_IDENTITY,
                    packaged_backend_identity: "forced-cpu",
                },
                backend_select::load_packaged_cpu_modules_from_executable,
                || panic!("forced cpu must skip discovery"),
                |_| panic!("forced cpu must skip benchmarks"),
            )?;
            let backend = Rc::new(
                LlamaBackend::init()
                    .map_err(|err| EngineError::new("BackendInit", err.to_string()))?,
            );
            return Self::load_final(pin, cache_dir, path, backend, runtime);
        }

        let executable = std::env::current_exe().map_err(|err| {
            EngineError::new(
                "BackendDiscovery",
                format!("cannot resolve executable: {err}"),
            )
        })?;
        let package_identity = backend_select::packaged_backend_identity(&executable);
        let context = backend_select::SelectionContext {
            sidecar_version: VERSION,
            model_sha256: pin.sha256,
            native_build_identity: backend_select::NATIVE_BUILD_IDENTITY,
            packaged_backend_identity: &package_identity,
        };
        let module_report = backend_select::load_packaged_modules_from_executable(&executable)
            .map_err(|reason| EngineError::new("BackendDiscovery", reason))?;
        let backend = Rc::new(
            LlamaBackend::init().map_err(|err| EngineError::new("BackendInit", err.to_string()))?,
        );
        let mut discovery = backend_select::discover_candidates(
            &backend_select::DeclaredBackends::build(),
            &runtime_devices(),
        );
        discovery.accelerator_failures = module_report.accelerator_failures;
        let runtime = backend_select::select_runtime_with(
            cache_dir,
            context,
            forced.as_deref(),
            || discovery,
            |candidate| Self::benchmark_candidate(pin, path, &backend, candidate),
        )
        .map_err(|reason| EngineError::new("BackendProbe", reason))?;
        Self::load_final(pin, cache_dir, path, backend, runtime)
    }

    fn load_final(
        pin: &'static ModelPin,
        cache_dir: &Path,
        path: &Path,
        backend: Rc<LlamaBackend>,
        runtime: backend_select::RuntimeSelection,
    ) -> Result<Self, EngineError> {
        let candidate = backend_select::BackendCandidate {
            backend: runtime.selection.resolved.clone(),
            device_index: runtime.device_index,
        };
        match Self::load_candidate(
            pin,
            path,
            &backend,
            &candidate,
            runtime.selection.clone(),
            runtime.capabilities,
        ) {
            Ok(engine) => Ok(engine),
            Err(error) if candidate.backend != backend_select::CPU => {
                backend_select::invalidate_cached_selection(cache_dir, &runtime);
                let mut capabilities = runtime.capabilities;
                disable_capability(&mut capabilities, &candidate.backend);
                let fallback = Selection::degraded_to_cpu(
                    &candidate.backend,
                    format!("{} final load failed: {}", candidate.backend, error.message),
                );
                Self::load_candidate(
                    pin,
                    path,
                    &backend,
                    &backend_select::BackendCandidate {
                        backend: backend_select::CPU.to_string(),
                        device_index: None,
                    },
                    fallback,
                    capabilities,
                )
            }
            Err(error) => Err(error),
        }
    }

    fn load_candidate(
        pin: &'static ModelPin,
        path: &Path,
        backend: &Rc<LlamaBackend>,
        candidate: &backend_select::BackendCandidate,
        selection: Selection,
        capabilities: BackendCapabilities,
    ) -> Result<Self, EngineError> {
        let params = model_params_for_candidate(candidate)?;
        let model = LlamaModel::load_from_file(backend, path, &params)
            .map_err(|err| EngineError::new("ModelLoad", err.to_string()))?;
        let eos_reserve = match pin.eos_marker {
            Some(marker) => model
                .str_to_token(marker, AddBos::Never)
                .map_err(|err| EngineError::new("Tokenize", err.to_string()))?
                .len(),
            None => 0,
        };
        let special_token_overhead = measure_special_token_overhead(&model)?;
        Ok(Self {
            pin,
            backend: Rc::clone(backend),
            model,
            eos_reserve,
            special_token_overhead,
            selection,
            capabilities,
        })
    }

    fn benchmark_candidate(
        pin: &'static ModelPin,
        path: &Path,
        backend: &Rc<LlamaBackend>,
        candidate: &backend_select::BackendCandidate,
    ) -> Result<backend_select::ProbeTiming, String> {
        let selection = Selection::new(&candidate.backend, &candidate.backend, None);
        let engine = Self::load_candidate(
            pin,
            path,
            backend,
            candidate,
            selection,
            BackendCapabilities::cpu_only(),
        )
        .map_err(|err| err.to_string())?;
        run_fixed_probes_with(candidate, |_, texts| {
            let started = std::time::Instant::now();
            engine
                .embed(texts, Role::Document)
                .map_err(|err| err.to_string())?;
            Ok(started.elapsed())
        })
    }

    /// Loads the pin from the shared cache directory `prepare` writes to.
    pub fn load_from_default_cache(pin: &'static ModelPin) -> Result<Self, EngineError> {
        let cache_dir = default_cache_dir()?;
        Self::load(pin, &cache_dir)
    }

    /// The pin this engine serves.
    pub fn pin(&self) -> &'static ModelPin {
        self.pin
    }

    /// Tokens reserved for the EOS marker, measured under the model's own tokenizer.
    pub fn eos_reserve(&self) -> usize {
        self.eos_reserve
    }

    /// Tokens the tokenizer adds around every input, measured once at load.
    pub fn special_token_overhead(&self) -> usize {
        self.special_token_overhead
    }

    /// Renders a token sequence back to text, special tokens included.
    ///
    /// Pieces are concatenated as bytes before the UTF-8 decode so a multi-byte character
    /// split across two tokens still round-trips.
    fn detokenize(&self, tokens: &[i32]) -> Result<String, String> {
        let mut bytes: Vec<u8> = Vec::with_capacity(tokens.len() * 4);
        for token in tokens {
            bytes.extend_from_slice(&self.token_piece(LlamaToken(*token))?);
        }
        String::from_utf8(bytes).map_err(|err| err.to_string())
    }

    /// Renders one token's bytes, growing the buffer when the first guess is too small.
    ///
    /// `token_to_piece_bytes` does not retry: it reports the space it needed as a negative
    /// size and leaves the caller to allocate. `str_to_token` in the same crate handles its
    /// own overflow this way, and this mirrors that idiom — without it, any piece longer
    /// than the initial guess is a hard error, which bge's WordPiece vocabulary hits with
    /// its 9- and 11-byte pieces.
    fn token_piece(&self, token: LlamaToken) -> Result<Vec<u8>, String> {
        const FIRST_GUESS: usize = 8;
        match self
            .model
            .token_to_piece_bytes(token, FIRST_GUESS, true, None)
        {
            Err(TokenToStringError::InsufficientBufferSpace(needed)) => {
                let needed = usize::try_from(-needed).map_err(|err| err.to_string())?;
                self.model
                    .token_to_piece_bytes(token, needed, true, None)
                    .map_err(|err| err.to_string())
            }
            other => other.map_err(|err| err.to_string()),
        }
    }

    /// Tokens `text` will actually present to the model under `role`, after truncation.
    ///
    /// This is the number that sizes the context, and through it the KV cache — the single
    /// input to the peak-memory math documented on [`MAX_DECODE_UBATCH_TOKENS`]. Exposed so
    /// a test can pin the worst case without paying for a full embed.
    pub fn input_token_count(&self, text: &str, role: Role) -> Result<usize, EngineError> {
        self.build_input(text, role).map(|tokens| tokens.len())
    }

    /// Applies the contract's per-input pipeline: sanitize, prefix, fit, append EOS.
    fn build_input(&self, text: &str, role: Role) -> Result<Vec<LlamaToken>, EngineError> {
        let instruction = match role {
            Role::Query => self.pin.query_instruction,
            Role::Document => self.pin.document_instruction,
        };
        let prefixed = format!("{instruction}{}", sanitize::sanitize(text));

        let tokenize_error: RefCell<Option<String>> = RefCell::new(None);
        let record = |err: String| {
            let mut slot = tokenize_error.borrow_mut();
            slot.get_or_insert(err);
        };
        let fitted = truncate::fit(
            &prefixed,
            self.pin.max_text_tokens,
            self.eos_reserve,
            self.special_token_overhead,
            |s| match self.model.str_to_token(s, AddBos::Never) {
                Ok(tokens) => tokens.into_iter().map(|t| t.0).collect(),
                Err(err) => {
                    record(err.to_string());
                    Vec::new()
                }
            },
            |tokens| match self.detokenize(tokens) {
                Ok(text) => text,
                Err(err) => {
                    record(err);
                    String::new()
                }
            },
        );
        if let Some(err) = tokenize_error.into_inner() {
            return Err(EngineError::new("Tokenize", err));
        }

        let with_eos = match self.pin.eos_marker {
            Some(marker) => format!("{fitted}{marker}"),
            None => fitted,
        };
        self.model
            .str_to_token(&with_eos, AddBos::Always)
            .map_err(|err| EngineError::new("Tokenize", err.to_string()))
    }

    /// Encodes one group of pre-tokenized inputs in a single fresh context.
    ///
    /// Failures are typed so [`isolate`] can tell a poison input from a broken backend: an
    /// [`EncodeFailure::Item`] is bisected down to the offending input, while an
    /// [`EncodeFailure::Systemic`] aborts the whole request. A fresh context per call is
    /// both the crate's requirement and the batch memory hygiene the contract implies.
    ///
    /// The two pinned models take different llama.cpp entry points. A `cls`-pooled model
    /// is a non-causal encoder: `llama_encode` asserts the whole batch fits one micro-batch
    /// (`llama-context.cpp:1369`), which its 512-token budget always satisfies. A
    /// `last`-pooled model is a causal LM, so `llama_decode` accepts a micro-batch smaller
    /// than the sequence (`llama-context.cpp:1714`) — the only way a 32k-token Qwen3 input
    /// is embeddable without asking the backend for tens of gigabytes.
    fn encode_group(&self, group: &[&Vec<LlamaToken>]) -> Result<Vec<Vec<f32>>, EncodeFailure> {
        let total_tokens: usize = group.iter().map(|t| t.len()).sum();
        let max_tokens: usize = group.iter().map(|t| t.len()).max().unwrap_or(0);
        // llama.cpp partitions the context evenly across sequences (`n_ctx_seq =
        // n_ctx / n_seq_max`), so the context must be sized to the LONGEST member times
        // the member count — and `n_seq_max` must actually be set. Leaving it at its
        // default of 1 makes every multi-sequence group fail its KV placement, which
        // [`isolate`] then silently bisects down to one context per input: correct
        // output, catastrophic throughput (the 2026-07-20 8x regression).
        let kv_cells = max_tokens * group.len();
        let (Ok(n_tokens), Ok(n_seq), Ok(n_seq_max), Ok(n_ctx)) = (
            u32::try_from(total_tokens),
            i32::try_from(group.len()),
            u32::try_from(group.len()),
            u32::try_from(kv_cells),
        ) else {
            return Err(EncodeFailure::Item);
        };
        if total_tokens == 0 {
            return Err(EncodeFailure::Item);
        }
        let n_ubatch = match self.pin.pooling {
            Pooling::Cls => n_tokens,
            Pooling::Last => u32::try_from(total_tokens.min(MAX_DECODE_UBATCH_TOKENS))
                .map_err(|_| EncodeFailure::Item)?,
        };
        // Named in every failure message: a Decode/Encode error is almost always a
        // consequence of these numbers, and reconstructing them from a CI log is guesswork.
        let shape = EncodeShape {
            total_tokens,
            n_ctx,
            n_batch: n_tokens,
            n_ubatch,
            n_seq,
        };

        let params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(n_ctx))
            .with_n_batch(n_tokens)
            .with_n_ubatch(n_ubatch)
            .with_n_seq_max(n_seq_max)
            .with_embeddings(true)
            .with_pooling_type(match self.pin.pooling {
                Pooling::Last => LlamaPoolingType::Last,
                Pooling::Cls => LlamaPoolingType::Cls,
            });
        // A context that cannot be created is an allocation failure, not a bad input:
        // bisecting it would zero-vector an entire healthy batch.
        let mut context = self
            .model
            .new_context(&self.backend, params)
            .map_err(|err| EncodeFailure::systemic("ContextAlloc", format!("{err} ({shape})")))?;

        // `logits_all=true` on the causal decode path makes llama.cpp reserve a
        // vocab-width output row for EVERY token — 19 GiB for a 32768-token Qwen3 input
        // (152k vocab x f32), which is what actually OOM'd the 16 GiB CI runner. Last
        // pooling reads only each sequence's final token, and `add_sequence(.., false)`
        // still flags that one. The non-causal encoder ignores per-token output flags,
        // so `false` is correct for both paths.
        let mut batch = LlamaBatch::new(total_tokens, n_seq);
        for (seq_id, tokens) in group.iter().enumerate() {
            let seq_id = i32::try_from(seq_id).map_err(|_| EncodeFailure::Item)?;
            batch
                .add_sequence(tokens, seq_id, false)
                .map_err(|_| EncodeFailure::Item)?;
        }
        match self.pin.pooling {
            Pooling::Cls => context
                .encode(&mut batch)
                .map_err(|err| encode_failure(err, &shape))?,
            Pooling::Last => context
                .decode(&mut batch)
                .map_err(|err| decode_failure(err, &shape))?,
        }

        let mut vectors = Vec::with_capacity(group.len());
        for seq_id in 0..group.len() {
            let seq_id = i32::try_from(seq_id).map_err(|_| EncodeFailure::Item)?;
            // Every variant of this error names a context that was configured wrong, which
            // is true for every item in the batch equally.
            let raw = context
                .embeddings_seq_ith(seq_id)
                .map_err(|err| EncodeFailure::systemic("Embeddings", err.to_string()))?;
            vectors.push(self.serve_lane(raw).ok_or(EncodeFailure::Item)?);
        }
        Ok(vectors)
    }

    /// Normalizes, slices to the served MRL lane, and renormalizes.
    ///
    /// Slicing an L2-normalized vector denormalizes it, so the second normalization is
    /// what makes the served lane a unit vector — the contract freezes this order. For a
    /// model whose served lane is its native width the slice is the identity.
    fn serve_lane(&self, raw: &[f32]) -> Option<Vec<f32>> {
        if raw.len() < self.pin.serve_dims {
            return None;
        }
        let mut vector = raw.to_vec();
        l2_normalize(&mut vector)?;
        vector.truncate(self.pin.serve_dims);
        l2_normalize(&mut vector)?;
        Some(vector)
    }

    /// The backend decision this engine loaded under.
    pub fn selection(&self) -> &Selection {
        &self.selection
    }

    /// Reports the real backend outcome, not a fixed one.
    ///
    /// `capabilities` advertises what this build compiled — CPU always, Metal under the
    /// `metal` feature; the requested/resolved pair and the degradation reason come from
    /// the cached [`Selection`], so a forced or benchmarked degradation is visible on the
    /// wire.
    fn engine_facts(&self) -> EngineFacts {
        EngineFacts {
            runtime: RUNTIME.to_string(),
            device: self.selection.resolved.clone(),
            requested_backend: self.selection.requested.clone(),
            resolved_backend: self.selection.resolved.clone(),
            accelerated: self.selection.accelerated,
            degraded_reason: self.selection.degraded_reason.clone(),
            capabilities: self.capabilities,
            llama_cpp_build: LLAMA_CPP_BUILD.to_string(),
        }
    }
}

impl EmbedEngine for LlamaEngine {
    fn health_facts(&self) -> Result<Value, EngineError> {
        let state = ModelState::Ready {
            pin: self.pin,
            dims: self.pin.serve_dims,
        };
        Ok(health::build(
            &state,
            &self.engine_facts(),
            Limits::default(),
            VERSION,
        ))
    }

    fn embed(&self, texts: &[String], role: Role) -> Result<EmbedOutput, EngineError> {
        let dims = self.pin.serve_dims;
        if texts.is_empty() {
            return Ok(EmbedOutput {
                dims,
                vectors: Vec::new(),
            });
        }

        let inputs: Vec<Vec<LlamaToken>> = texts
            .iter()
            .map(|text| self.build_input(text, role))
            .collect::<Result<_, _>>()?;

        // A cls-pooled encoder must hold its whole group in one micro-batch, so its groups
        // are bounded by the micro-batch ceiling rather than the larger encode ceiling.
        let group_budget = match self.pin.pooling {
            Pooling::Cls => MAX_CLS_GROUP_TOKENS,
            Pooling::Last => MAX_TOKENS_PER_ENCODE,
        };
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());
        for group in group_by_cell_budget(&inputs, group_budget) {
            vectors.extend(isolate(&group, dims, &|items| self.encode_group(items))?);
        }

        if vectors.len() != texts.len() {
            return Err(EngineError::new(
                "CountMismatch",
                format!(
                    "produced {} vectors for {} texts",
                    vectors.len(),
                    texts.len()
                ),
            ));
        }
        if let Some(bad) = vectors.iter().find(|v| v.len() != dims) {
            return Err(EngineError::new(
                "DimsMismatch",
                format!("produced a {}-dim vector, declared {dims}", bad.len()),
            ));
        }
        Ok(EmbedOutput { dims, vectors })
    }
}

/// Groups inputs into runs whose CONTEXT CELLS stay under `budget`.
///
/// The context a group needs is `longest member x member count`, because llama.cpp
/// partitions `n_ctx` evenly across sequences — so the budget is applied to that product,
/// not to the token sum. The product is always >= the sum, so a group under this budget
/// also fits a same-sized batch/ubatch ceiling. An input longer than the budget forms a
/// group of its own rather than being dropped or split — it is already fitted to the
/// model's own limit.
pub fn group_by_cell_budget<T>(inputs: &[T], budget: usize) -> Vec<Vec<&Vec<LlamaToken>>>
where
    T: std::borrow::Borrow<Vec<LlamaToken>>,
{
    let mut groups: Vec<Vec<&Vec<LlamaToken>>> = Vec::new();
    let mut current: Vec<&Vec<LlamaToken>> = Vec::new();
    let mut current_max = 0usize;

    for input in inputs {
        let tokens = input.borrow();
        let next_max = current_max.max(tokens.len());
        if !current.is_empty() && next_max * (current.len() + 1) > budget {
            groups.push(std::mem::take(&mut current));
            current_max = 0;
        }
        current_max = current_max.max(tokens.len());
        current.push(tokens);
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

/// Why an encode attempt failed, and therefore whether bisecting it means anything.
///
/// The distinction is the difference between a degraded batch and a silent lie: bisecting
/// a backend-wide failure reaches one item at a time, fails at every level, and returns a
/// full set of zero vectors as a successful embedding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeFailure {
    /// A single input in the group can explain the failure, so [`isolate`] bisects.
    Item,
    /// The backend itself failed — allocation, a fatal encode, a misconfigured context.
    /// Every item would fail identically, so the request errors instead.
    Systemic(EngineError),
}

impl EncodeFailure {
    /// Builds a systemic failure carrying the error the request will fail with.
    pub fn systemic(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Systemic(EngineError::new(kind, message))
    }
}

/// The context geometry an encode ran under, reported with every failure.
///
/// A `Decode Error -2` is an allocation failure, and its cause is always some combination
/// of these five numbers against the machine's memory ceiling. Carrying them in the message
/// is the difference between a one-line diagnosis and a log-spelunking session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeShape {
    /// Tokens submitted across the whole group.
    pub total_tokens: usize,
    /// Context size requested, which is the token total.
    pub n_ctx: u32,
    /// Logical batch size requested.
    pub n_batch: u32,
    /// Physical micro-batch width requested.
    pub n_ubatch: u32,
    /// Sequences in the group.
    pub n_seq: i32,
}

impl std::fmt::Display for EncodeShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "tokens={} n_ctx={} n_batch={} n_ubatch={} n_seq={}",
            self.total_tokens, self.n_ctx, self.n_batch, self.n_ubatch, self.n_seq
        )
    }
}

/// Classifies a `llama_decode` result per `llama.h`'s documented return codes.
///
/// `1` (no KV slot) explicitly suggests reducing the batch size, and `-1` is an invalid
/// input batch — both are exactly what bisection isolates. Everything else is `2` (aborted)
/// or `< -1` (fatal error), where llama.h notes the context's memory state is left dirty.
fn decode_failure(err: DecodeError, shape: &EncodeShape) -> EncodeFailure {
    match err {
        DecodeError::NoKvCacheSlot | DecodeError::NTokensZero => EncodeFailure::Item,
        other => EncodeFailure::systemic("Decode", format!("{other} ({shape})")),
    }
}

/// Classifies a `llama_encode` result; `llama.h` documents only `0` and `< 0` here.
fn encode_failure(err: EncodeError, shape: &EncodeShape) -> EncodeFailure {
    match err {
        EncodeError::NoKvCacheSlot | EncodeError::NTokensZero => EncodeFailure::Item,
        other => EncodeFailure::systemic("Encode", format!("{other} ({shape})")),
    }
}

/// Binary-search failure isolation over an encode operation.
///
/// Encodes the whole slice; on an item-shaped failure splits in half and recurses, so a
/// single poison item costs about `log2(n)` extra encodes rather than `n`. When recursion
/// reaches one item that still fails, that item receives a zero vector of `dims` and the
/// batch succeeds — `vectors.len()` always equals `items.len()`.
///
/// A [`EncodeFailure::Systemic`] propagates immediately from whatever depth raised it: the
/// backend, not the input, is broken, and answering with zero vectors would report a
/// working embedding for a request that produced none. Generic over `encode` so the
/// algorithm is testable without a model.
pub fn isolate<T, F>(items: &[T], dims: usize, encode: &F) -> Result<Vec<Vec<f32>>, EngineError>
where
    F: Fn(&[T]) -> Result<Vec<Vec<f32>>, EncodeFailure>,
    T: Clone,
{
    if items.is_empty() {
        return Ok(Vec::new());
    }
    match encode(items) {
        Ok(vectors) if vectors.len() == items.len() => return Ok(vectors),
        Ok(_) | Err(EncodeFailure::Item) => {}
        Err(EncodeFailure::Systemic(err)) => return Err(err),
    }
    if items.len() == 1 {
        eprintln!("julie-semantic-sidecar: substituting a zero vector for an unencodable input");
        return Ok(vec![vec![0.0; dims]]);
    }
    let mid = items.len() / 2;
    let mut vectors = isolate(&items[..mid], dims, encode)?;
    vectors.extend(isolate(&items[mid..], dims, encode)?);
    Ok(vectors)
}

/// Scales `vector` to unit length in place, returning `None` if it has no direction.
fn l2_normalize(vector: &mut [f32]) -> Option<()> {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return None;
    }
    for value in vector.iter_mut() {
        *value /= norm;
    }
    Some(())
}

/// Measures how many tokens the tokenizer adds around every input beyond the text itself.
fn measure_special_token_overhead(model: &LlamaModel) -> Result<usize, EngineError> {
    let with_special = model
        .str_to_token(OVERHEAD_PROBE, AddBos::Always)
        .map_err(|err| EngineError::new("Tokenize", err.to_string()))?;
    let without_special = model
        .str_to_token(OVERHEAD_PROBE, AddBos::Never)
        .map_err(|err| EngineError::new("Tokenize", err.to_string()))?;
    Ok(with_special.len().saturating_sub(without_special.len()))
}

/// Model load parameters that actually apply the resolved backend placement.
///
/// `LlamaModelParams::default()` leaves `n_gpu_layers` at llama.cpp's `-1` — offload every
/// layer — so on a machine with an accelerator a `cpu` resolution would load onto the GPU
/// anyway while `health` reported `cpu`. Pinning zero offloaded layers makes the reported
/// device the applied one, which is what `JULIE_SIDECAR_FORCE_BACKEND=cpu` and the
/// CPU-generated conformance goldens both depend on.
fn model_params_for_candidate(
    candidate: &backend_select::BackendCandidate,
) -> Result<LlamaModelParams, EngineError> {
    let placement = placement_for_candidate(candidate);
    let mut params = LlamaModelParams::default();
    if let Some(layers) = placement.n_gpu_layers {
        params = params.with_n_gpu_layers(layers);
    }
    if !placement.device_indexes.is_empty() {
        params = params
            .with_devices(&placement.device_indexes)
            .map_err(|err| EngineError::new("BackendPlacement", err.to_string()))?;
    }
    Ok(params)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DriverIdentity {
    value: String,
    cacheable: bool,
}

fn runtime_devices() -> Vec<backend_select::RuntimeDevice> {
    let mut identities = HashMap::new();
    list_llama_ggml_backend_devices()
        .into_iter()
        .map(|device| {
            let identity = cached_driver_identity_with(
                &mut identities,
                &device.backend,
                &device.name,
                &device.description,
                &mut command_output,
            );
            backend_select::RuntimeDevice {
                driver: identity.value,
                driver_cacheable: identity.cacheable,
                backend: device.backend,
                index: device.index,
                name: device.name,
                description: device.description,
                memory_total: device.memory_total,
            }
        })
        .collect()
}

fn cached_driver_identity_with<F>(
    identities: &mut HashMap<String, DriverIdentity>,
    backend: &str,
    name: &str,
    description: &str,
    output: &mut F,
) -> DriverIdentity
where
    F: FnMut(&str, &[&str]) -> Option<String>,
{
    if backend.eq_ignore_ascii_case(backend_select::CPU) {
        return DriverIdentity {
            value: backend_select::CPU.to_string(),
            cacheable: true,
        };
    }
    let key = backend.to_ascii_lowercase();
    if let Some(identity) = identities.get(&key) {
        return identity.clone();
    }
    let identity = driver_identity_with(backend, name, description, output);
    identities.insert(key, identity.clone());
    identity
}

fn driver_identity_with<F>(
    backend: &str,
    name: &str,
    description: &str,
    mut output: F,
) -> DriverIdentity
where
    F: FnMut(&str, &[&str]) -> Option<String>,
{
    if backend.eq_ignore_ascii_case(backend_select::CUDA) {
        if let Some(value) = output(
            "nvidia-smi",
            &["--query-gpu=driver_version", "--format=csv,noheader"],
        ) {
            return DriverIdentity {
                value: format!("cuda:{value}"),
                cacheable: true,
            };
        }
    }
    if backend.eq_ignore_ascii_case(backend_select::VULKAN) {
        if let Some(value) = output("vulkaninfo", &["--summary"]) {
            return DriverIdentity {
                value: format!("vulkan:{value}"),
                cacheable: true,
            };
        }
    }
    if cfg!(target_os = "windows") {
        if let Some(value) = output(
            "powershell.exe",
            &[
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress",
            ],
        ) {
            return DriverIdentity {
                value: format!("windows-gpu:{value}"),
                cacheable: true,
            };
        }
    }
    if cfg!(target_os = "macos")
        && (backend.eq_ignore_ascii_case(backend_select::METAL)
            || backend.eq_ignore_ascii_case("MTL"))
    {
        if let Some(value) = output("sysctl", &["-n", "kern.osversion"]) {
            return DriverIdentity {
                value: format!("metal:{value}"),
                cacheable: true,
            };
        }
    }
    let platform = if cfg!(target_os = "macos") {
        output("sysctl", &["-n", "kern.osversion"])
    } else if cfg!(unix) {
        output("uname", &["-r"])
    } else if cfg!(target_os = "windows") {
        output("cmd", &["/C", "ver"])
    } else {
        None
    }
    .unwrap_or_else(|| "unknown".to_string());
    DriverIdentity {
        value: format!(
            "backend={backend};name={name};description={description};platform={platform};driver=unverified"
        ),
        cacheable: false,
    }
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.trim().to_string())
        .filter(|output| !output.is_empty())
}

fn disable_capability(capabilities: &mut BackendCapabilities, backend: &str) {
    match backend {
        backend_select::METAL => capabilities.metal = false,
        backend_select::VULKAN => capabilities.vulkan = false,
        backend_select::CUDA => capabilities.cuda = false,
        _ => {}
    }
}

/// Verifies the cached model file against the pin's digest before it is loaded.
///
/// A mismatch is reported as `ModelNotPrepared` — the cached bytes are not the pinned
/// model, so the honest state is "not prepared" rather than a hard start failure. The
/// message names both digests so the operator can tell corruption from a swapped file.
fn verify_cached_digest(path: &Path, expected: &str) -> Result<(), EngineError> {
    let actual = file_digest(path).map_err(|err| {
        EngineError::new(
            "ModelNotPrepared",
            format!("cannot read {} to verify its digest: {err}", path.display()),
        )
    })?;
    if actual.eq_ignore_ascii_case(expected) {
        return Ok(());
    }
    Err(EngineError::new(
        "ModelNotPrepared",
        format!(
            "{} does not match its pinned digest: expected sha256 {expected}, found {actual}; \
             delete it and re-run `prepare`",
            path.display()
        ),
    ))
}

/// Streams `path` through sha256, returning the lowercase hex digest.
fn file_digest(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; DIGEST_CHUNK_BYTES];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let mut hex = String::with_capacity(Sha256::output_size() * 2);
    for byte in hasher.finalize() {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    Ok(hex)
}

/// Cache directory the model file is read from, matching `prepare`'s resolution rule.
fn default_cache_dir() -> Result<PathBuf, EngineError> {
    crate::prepare::cache_dir()
        .map_err(|err| EngineError::new("CacheDir", err.message().to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(seed: f32, dims: usize) -> Vec<f32> {
        let mut v = vec![seed; dims];
        l2_normalize(&mut v).expect("seed is non-zero");
        v
    }

    #[test]
    fn l2_normalize_scales_to_unit_length() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v).expect("vector has direction");
        assert!((v.iter().map(|x| x * x).sum::<f32>().sqrt() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_refuses_a_zero_vector() {
        let mut v = vec![0.0, 0.0];
        assert!(l2_normalize(&mut v).is_none());
    }

    fn encoded(items: &[usize]) -> Result<Vec<Vec<f32>>, EncodeFailure> {
        Ok(items.iter().map(|i| unit(*i as f32 + 1.0, 4)).collect())
    }

    #[test]
    fn isolate_returns_encoder_output_when_the_batch_succeeds() {
        let vectors = isolate(&[0usize, 1, 2], 4, &encoded).expect("no failure");
        assert_eq!(vectors.len(), 3);
        assert!(vectors.iter().all(|v| v.len() == 4));
    }

    #[test]
    fn isolate_zero_vectors_only_the_poison_item() {
        let poison = 2usize;
        let encode = |items: &[usize]| {
            if items.contains(&poison) {
                Err(EncodeFailure::Item)
            } else {
                encoded(items)
            }
        };
        let vectors = isolate(&[0usize, 1, 2, 3, 4], 4, &encode).expect("item failures isolate");
        assert_eq!(vectors.len(), 5);
        assert_eq!(vectors[poison], vec![0.0; 4]);
        for (i, v) in vectors.iter().enumerate() {
            if i != poison {
                assert!((v.iter().map(|x| x * x).sum::<f32>().sqrt() - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn isolate_handles_every_item_failing() {
        let encode = |_: &[usize]| Err(EncodeFailure::Item);
        let vectors = isolate(&[0usize, 1, 2], 4, &encode).expect("item failures isolate");
        assert_eq!(vectors, vec![vec![0.0; 4]; 3]);
    }

    #[test]
    fn isolate_propagates_a_systemic_failure_instead_of_zero_vectoring_the_batch() {
        let encode = |_: &[usize]| Err(EncodeFailure::systemic("ContextAlloc", "out of memory"));
        let err = isolate(&[0usize, 1, 2, 3], 4, &encode).expect_err("systemic failures error");
        assert_eq!(err, EngineError::new("ContextAlloc", "out of memory"));
    }

    #[test]
    fn isolate_propagates_a_systemic_failure_raised_deep_in_the_bisection() {
        let encode = |items: &[usize]| {
            if items.len() == 1 {
                Err(EncodeFailure::systemic("Decode", "fatal error"))
            } else {
                Err(EncodeFailure::Item)
            }
        };
        let err = isolate(&[0usize, 1, 2, 3], 4, &encode).expect_err("systemic failures error");
        assert_eq!(err.kind, "Decode");
    }

    #[test]
    fn isolate_rejects_a_count_mismatch_from_the_encoder() {
        let encode = |items: &[usize]| {
            if items.len() == 1 {
                Ok(vec![unit(1.0, 4)])
            } else {
                Ok(Vec::new())
            }
        };
        assert_eq!(isolate(&[0usize, 1], 4, &encode).expect("bisects").len(), 2);
    }

    #[test]
    fn isolate_of_an_empty_batch_is_empty() {
        let encode = |_: &[usize]| Err(EncodeFailure::Item);
        assert!(isolate::<usize, _>(&[], 4, &encode)
            .expect("empty is not a failure")
            .is_empty());
    }

    fn shape() -> EncodeShape {
        EncodeShape {
            total_tokens: 32768,
            n_ctx: 32768,
            n_batch: 32768,
            n_ubatch: 512,
            n_seq: 1,
        }
    }

    #[test]
    fn a_no_kv_slot_decode_is_an_item_failure_and_a_fatal_one_is_systemic() {
        assert_eq!(
            decode_failure(DecodeError::NoKvCacheSlot, &shape()),
            EncodeFailure::Item
        );
        assert_eq!(
            decode_failure(DecodeError::NTokensZero, &shape()),
            EncodeFailure::Item
        );
        assert!(matches!(
            decode_failure(DecodeError::Unknown(-2), &shape()),
            EncodeFailure::Systemic(_)
        ));
        assert!(matches!(
            encode_failure(EncodeError::Unknown(-2), &shape()),
            EncodeFailure::Systemic(_)
        ));
    }

    #[test]
    fn a_fatal_decode_names_the_context_geometry_that_caused_it() {
        let EncodeFailure::Systemic(err) = decode_failure(DecodeError::Unknown(-2), &shape())
        else {
            panic!("a fatal decode is systemic");
        };
        assert_eq!(err.kind, "Decode");
        for fact in [
            "tokens=32768",
            "n_ctx=32768",
            "n_batch=32768",
            "n_ubatch=512",
            "n_seq=1",
        ] {
            assert!(
                err.message.contains(fact),
                "{fact} missing: {}",
                err.message
            );
        }
    }

    #[test]
    fn a_cpu_resolution_pins_zero_offloaded_layers() {
        let params = model_params_for_candidate(&backend_select::BackendCandidate {
            backend: backend_select::CPU.to_string(),
            device_index: None,
        })
        .expect("cpu placement");
        assert_eq!(params.n_gpu_layers(), 0);
    }

    #[test]
    fn a_file_whose_bytes_do_not_match_the_pin_is_not_prepared() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"not the pinned weights").expect("seed");
        let expected = "0".repeat(64);

        let err = verify_cached_digest(&path, &expected).expect_err("a mismatch is rejected");
        assert_eq!(err.kind, "ModelNotPrepared");
        assert!(err.message.contains(&expected), "{}", err.message);
        assert!(
            err.message.contains(&file_digest(&path).expect("digest")),
            "{}",
            err.message
        );
    }

    #[test]
    fn a_file_matching_the_pin_verifies() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"the pinned weights").expect("seed");
        let digest = file_digest(&path).expect("digest");
        assert!(verify_cached_digest(&path, &digest).is_ok());
        assert!(verify_cached_digest(&path, &digest.to_uppercase()).is_ok());
    }

    #[test]
    fn the_digest_of_a_known_input_is_the_standard_sha256() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("abc");
        std::fs::write(&path, b"abc").expect("seed");
        assert_eq!(
            file_digest(&path).expect("digest"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn group_by_cell_budget_packs_inputs_under_the_ceiling() {
        let inputs: Vec<Vec<LlamaToken>> = vec![
            vec![LlamaToken(1); 4],
            vec![LlamaToken(1); 4],
            vec![LlamaToken(1); 4],
        ];
        let groups = group_by_cell_budget(&inputs, 8);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].len(), 2);
        assert_eq!(groups[1].len(), 1);
    }

    #[test]
    fn group_by_cell_budget_gives_an_oversized_input_its_own_group() {
        let inputs: Vec<Vec<LlamaToken>> = vec![vec![LlamaToken(1); 2], vec![LlamaToken(1); 50]];
        let groups = group_by_cell_budget(&inputs, 8);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[1][0].len(), 50);
    }

    #[test]
    fn group_by_cell_budget_charges_the_longest_member_for_every_seat() {
        // Sum-budgeting would pack all four (2+2+2+6 = 12 <= 16); cell-budgeting must not,
        // because the partitioned context costs longest x count = 6 x 4 = 24 cells.
        let inputs: Vec<Vec<LlamaToken>> = vec![
            vec![LlamaToken(1); 2],
            vec![LlamaToken(1); 2],
            vec![LlamaToken(1); 2],
            vec![LlamaToken(1); 6],
        ];
        let groups = group_by_cell_budget(&inputs, 16);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].len(), 3);
        assert_eq!(groups[1].len(), 1);
    }

    #[test]
    fn candidate_placement_pins_cpu_and_uses_the_enumerated_accelerator_index() {
        let cpu = placement_for_candidate(&backend_select::BackendCandidate {
            backend: backend_select::CPU.to_string(),
            device_index: None,
        });
        let vulkan = placement_for_candidate(&backend_select::BackendCandidate {
            backend: backend_select::VULKAN.to_string(),
            device_index: Some(7),
        });

        assert_eq!(cpu.n_gpu_layers, Some(0));
        assert!(cpu.device_indexes.is_empty());
        assert_eq!(vulkan.n_gpu_layers, None);
        assert_eq!(vulkan.device_indexes, vec![7]);
    }

    #[test]
    fn forced_cpu_loads_runtime_modules_before_short_circuiting_selection() {
        let dir = tempfile::tempdir().expect("tempdir");
        let executable = dir.path().join("julie-semantic-sidecar");
        let module_loads = std::cell::Cell::new(0);
        let selected = forced_cpu_runtime_with(
            dir.path(),
            &executable,
            backend_select::SelectionContext {
                sidecar_version: VERSION,
                model_sha256: "model-a",
                native_build_identity: "native-a",
                packaged_backend_identity: "forced-cpu",
            },
            |path| {
                assert_eq!(path, executable);
                module_loads.set(module_loads.get() + 1);
                Ok(())
            },
            || panic!("forced cpu must skip discovery"),
            |_| panic!("forced cpu must skip benchmarks"),
        )
        .expect("forced cpu runtime");

        assert_eq!(module_loads.get(), 1);
        assert_eq!(selected.selection, Selection::cpu());
        assert!(!dir
            .path()
            .join(backend_select::SELECTION_CACHE_FILE)
            .exists());
    }

    #[test]
    fn fixed_probes_time_batch_one_and_the_sixteen_text_indexing_batch() {
        let candidate = backend_select::BackendCandidate {
            backend: backend_select::METAL.to_string(),
            device_index: Some(3),
        };
        let calls = std::cell::RefCell::new(Vec::new());
        let timing = run_fixed_probes_with(&candidate, |placed, texts| {
            calls.borrow_mut().push((placed.clone(), texts.len()));
            Ok(std::time::Duration::from_millis(texts.len() as u64))
        })
        .expect("probes");

        assert_eq!(
            calls.into_inner(),
            vec![
                (candidate.clone(), 1),
                (candidate.clone(), 1),
                (candidate, 16),
            ]
        );
        assert_eq!(timing.batch_1, std::time::Duration::from_millis(1));
        assert_eq!(timing.batch_16, std::time::Duration::from_millis(16));
    }

    #[test]
    fn vulkan_driver_output_changes_runtime_machine_identity() {
        let identity = |driver: &str| {
            driver_identity_with("Vulkan", "GPU 0", "Discrete GPU", |program, _| {
                (program == "vulkaninfo").then(|| driver.to_string())
            })
            .value
        };
        let discovery = |driver: String| {
            backend_select::discover_candidates(
                &backend_select::DeclaredBackends {
                    metal: false,
                    vulkan: true,
                    cuda: false,
                    rocm: false,
                    dynamic_backends: true,
                },
                &[backend_select::RuntimeDevice {
                    backend: "Vulkan".to_string(),
                    index: 1,
                    name: "GPU 0".to_string(),
                    description: "Discrete GPU".to_string(),
                    memory_total: 1024,
                    driver,
                    driver_cacheable: true,
                }],
            )
        };

        assert_ne!(
            discovery(identity("driverVersion = 1")).machine,
            discovery(identity("driverVersion = 2")).machine
        );
    }

    #[test]
    fn driver_lookup_is_cached_per_backend_and_skips_cpu() {
        let calls = std::cell::Cell::new(0);
        let mut cache = std::collections::HashMap::new();
        let mut output = |program: &str, _: &[&str]| {
            calls.set(calls.get() + 1);
            (program == "vulkaninfo").then(|| "driverVersion = 1".to_string())
        };

        let cpu = cached_driver_identity_with(&mut cache, "CPU", "CPU", "Host CPU", &mut output);
        let first = cached_driver_identity_with(
            &mut cache,
            "Vulkan",
            "GPU 0",
            "Integrated GPU",
            &mut output,
        );
        let second =
            cached_driver_identity_with(&mut cache, "Vulkan", "GPU 1", "Discrete GPU", &mut output);

        assert_eq!(cpu.value, "cpu");
        assert_eq!(first, second);
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn missing_vulkan_driver_tool_marks_the_identity_uncacheable() {
        let identity = driver_identity_with("Vulkan", "GPU 0", "Discrete GPU", |_, _| None);

        assert!(!identity.cacheable);
        assert!(identity.value.contains("driver=unverified"));
    }

    #[test]
    fn failed_cpu_inference_probe_aborts_backend_selection() {
        let dir = tempfile::tempdir().expect("tempdir");
        let declared = backend_select::DeclaredBackends {
            metal: false,
            vulkan: false,
            cuda: false,
            rocm: false,
            dynamic_backends: false,
        };
        let error = backend_select::select_runtime_with(
            dir.path(),
            backend_select::SelectionContext {
                sidecar_version: VERSION,
                model_sha256: "model-a",
                native_build_identity: "native-a",
                packaged_backend_identity: "package-a",
            },
            None,
            || backend_select::discover_candidates(&declared, &[]),
            |candidate| {
                run_fixed_probes_with(candidate, |_, _| Err("inference failed".to_string()))
            },
        )
        .expect_err("cpu probe failure");

        assert!(error.contains("cpu probe failed"));
    }

    #[test]
    fn a_failed_fixed_probe_rejects_the_candidate() {
        let candidate = backend_select::BackendCandidate {
            backend: backend_select::CUDA.to_string(),
            device_index: Some(4),
        };
        let calls = std::cell::Cell::new(0);
        let error = run_fixed_probes_with(&candidate, |_, _| {
            calls.set(calls.get() + 1);
            Err("encode failed".to_string())
        })
        .expect_err("failed probe");

        assert_eq!(error, "encode failed");
        assert_eq!(calls.get(), 1);
    }
}
