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
use std::io::Read;
use std::path::{Path, PathBuf};

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::{DecodeError, EncodeError, TokenToStringError};
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

/// Ceiling on the physical micro-batch a single encode step processes.
///
/// The attention compute buffer grows with the square of the micro-batch, so sizing it to
/// a whole 32k-token Qwen3 input asks the backend for tens of gigabytes and crashes it.
/// Both pinned models' sequences are pooled correctly when chunked at this width, and
/// bge's entire 512-token budget still fits in one micro-batch as its non-causal attention
/// requires.
const MAX_UBATCH_TOKENS: usize = 2_048;

/// A loaded llama.cpp embedding model serving one manifest pin.
pub struct LlamaEngine {
    pin: &'static ModelPin,
    backend: LlamaBackend,
    model: LlamaModel,
    eos_reserve: usize,
    special_token_overhead: usize,
    selection: Selection,
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

        crate::stdio_guard::guarded(|| {
            let selection = backend_select::select(cache_dir, VERSION, pin.sha256);
            let backend = LlamaBackend::init()
                .map_err(|err| EngineError::new("BackendInit", err.to_string()))?;
            let model = LlamaModel::load_from_file(&backend, &path, &model_params(&selection))
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
                backend,
                model,
                eos_reserve,
                special_token_overhead,
                selection,
            })
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
        let (Ok(n_tokens), Ok(n_seq)) = (u32::try_from(total_tokens), i32::try_from(group.len()))
        else {
            return Err(EncodeFailure::Item);
        };
        if total_tokens == 0 {
            return Err(EncodeFailure::Item);
        }
        let n_ubatch = match self.pin.pooling {
            Pooling::Cls => n_tokens,
            Pooling::Last => u32::try_from(total_tokens.min(MAX_UBATCH_TOKENS))
                .map_err(|_| EncodeFailure::Item)?,
        };

        let params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(n_tokens))
            .with_n_batch(n_tokens)
            .with_n_ubatch(n_ubatch)
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
            .map_err(|err| EncodeFailure::systemic("ContextAlloc", err.to_string()))?;

        let mut batch = LlamaBatch::new(total_tokens, n_seq);
        for (seq_id, tokens) in group.iter().enumerate() {
            let seq_id = i32::try_from(seq_id).map_err(|_| EncodeFailure::Item)?;
            batch
                .add_sequence(tokens, seq_id, true)
                .map_err(|_| EncodeFailure::Item)?;
        }
        match self.pin.pooling {
            Pooling::Cls => context.encode(&mut batch).map_err(encode_failure)?,
            Pooling::Last => context.decode(&mut batch).map_err(decode_failure)?,
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
    /// `capabilities` advertises only CPU because this build compiles no accelerated
    /// backend; the requested/resolved pair and the degradation reason come from the
    /// cached [`Selection`], so a forced or benchmarked degradation is visible on the wire.
    fn engine_facts(&self) -> EngineFacts {
        EngineFacts {
            runtime: RUNTIME.to_string(),
            device: self.selection.resolved.clone(),
            requested_backend: self.selection.requested.clone(),
            resolved_backend: self.selection.resolved.clone(),
            accelerated: self.selection.accelerated,
            degraded_reason: self.selection.degraded_reason.clone(),
            capabilities: BackendCapabilities {
                cpu: true,
                ..BackendCapabilities::default()
            },
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
            Pooling::Cls => MAX_UBATCH_TOKENS,
            Pooling::Last => MAX_TOKENS_PER_ENCODE,
        };
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());
        for group in group_by_token_budget(&inputs, group_budget) {
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

/// Groups inputs into runs whose token count stays under `budget`.
///
/// An input longer than the budget forms a group of its own rather than being dropped or
/// split — it is already fitted to the model's own limit.
pub fn group_by_token_budget<T>(inputs: &[T], budget: usize) -> Vec<Vec<&Vec<LlamaToken>>>
where
    T: std::borrow::Borrow<Vec<LlamaToken>>,
{
    let mut groups: Vec<Vec<&Vec<LlamaToken>>> = Vec::new();
    let mut current: Vec<&Vec<LlamaToken>> = Vec::new();
    let mut current_tokens = 0usize;

    for input in inputs {
        let tokens = input.borrow();
        if !current.is_empty() && current_tokens + tokens.len() > budget {
            groups.push(std::mem::take(&mut current));
            current_tokens = 0;
        }
        current_tokens += tokens.len();
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

/// Classifies a `llama_decode` result per `llama.h`'s documented return codes.
///
/// `1` (no KV slot) explicitly suggests reducing the batch size, and `-1` is an invalid
/// input batch — both are exactly what bisection isolates. Everything else is `2` (aborted)
/// or `< -1` (fatal error), where llama.h notes the context's memory state is left dirty.
fn decode_failure(err: DecodeError) -> EncodeFailure {
    match err {
        DecodeError::NoKvCacheSlot | DecodeError::NTokensZero => EncodeFailure::Item,
        other => EncodeFailure::systemic("Decode", other.to_string()),
    }
}

/// Classifies a `llama_encode` result; `llama.h` documents only `0` and `< 0` here.
fn encode_failure(err: EncodeError) -> EncodeFailure {
    match err {
        EncodeError::NoKvCacheSlot | EncodeError::NTokensZero => EncodeFailure::Item,
        other => EncodeFailure::systemic("Encode", other.to_string()),
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
fn model_params(selection: &Selection) -> LlamaModelParams {
    let params = LlamaModelParams::default();
    if selection.resolved == backend_select::CPU {
        params.with_n_gpu_layers(0)
    } else {
        params
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

    #[test]
    fn a_no_kv_slot_decode_is_an_item_failure_and_a_fatal_one_is_systemic() {
        assert_eq!(
            decode_failure(DecodeError::NoKvCacheSlot),
            EncodeFailure::Item
        );
        assert_eq!(
            decode_failure(DecodeError::NTokensZero),
            EncodeFailure::Item
        );
        assert!(matches!(
            decode_failure(DecodeError::Unknown(-2)),
            EncodeFailure::Systemic(_)
        ));
        assert!(matches!(
            encode_failure(EncodeError::Unknown(-2)),
            EncodeFailure::Systemic(_)
        ));
    }

    #[test]
    fn a_cpu_resolution_pins_zero_offloaded_layers() {
        assert_eq!(model_params(&Selection::cpu()).n_gpu_layers(), 0);
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
    fn group_by_token_budget_packs_inputs_under_the_ceiling() {
        let inputs: Vec<Vec<LlamaToken>> = vec![
            vec![LlamaToken(1); 4],
            vec![LlamaToken(1); 4],
            vec![LlamaToken(1); 4],
        ];
        let groups = group_by_token_budget(&inputs, 8);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].len(), 2);
        assert_eq!(groups[1].len(), 1);
    }

    #[test]
    fn group_by_token_budget_gives_an_oversized_input_its_own_group() {
        let inputs: Vec<Vec<LlamaToken>> = vec![vec![LlamaToken(1); 2], vec![LlamaToken(1); 50]];
        let groups = group_by_token_budget(&inputs, 8);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[1][0].len(), 50);
    }
}
