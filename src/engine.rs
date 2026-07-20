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
use std::path::{Path, PathBuf};

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use serde_json::Value;

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
}

impl LlamaEngine {
    /// Loads `pin`'s weights from `cache_dir`.
    ///
    /// The model file is checked for existence only — never re-hashed. `prepare` owns
    /// digest verification, and re-hashing a multi-gigabyte GGUF on every start would cost
    /// seconds to minutes.
    pub fn load(pin: &'static ModelPin, cache_dir: &Path) -> Result<Self, EngineError> {
        let path = cache_dir.join(pin.file);
        if !path.is_file() {
            return Err(EngineError::new(
                "ModelNotPrepared",
                format!("{} is not in the cache; run `prepare`", path.display()),
            ));
        }

        let backend =
            LlamaBackend::init().map_err(|err| EngineError::new("BackendInit", err.to_string()))?;
        let model = LlamaModel::load_from_file(&backend, &path, &LlamaModelParams::default())
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
            let piece = self
                .model
                .token_to_piece_bytes(LlamaToken(*token), 8, true, None)
                .map_err(|err| err.to_string())?;
            bytes.extend_from_slice(&piece);
        }
        String::from_utf8(bytes).map_err(|err| err.to_string())
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
    /// Returns `None` on any failure so [`isolate`] can bisect the group; a fresh context
    /// per call is both the crate's requirement and the batch memory hygiene the contract
    /// implies.
    ///
    /// The two pinned models take different llama.cpp entry points. A `cls`-pooled model
    /// is a non-causal encoder: `llama_encode` asserts the whole batch fits one micro-batch
    /// (`llama-context.cpp:1369`), which its 512-token budget always satisfies. A
    /// `last`-pooled model is a causal LM, so `llama_decode` accepts a micro-batch smaller
    /// than the sequence (`llama-context.cpp:1714`) — the only way a 32k-token Qwen3 input
    /// is embeddable without asking the backend for tens of gigabytes.
    fn encode_group(&self, group: &[&Vec<LlamaToken>]) -> Option<Vec<Vec<f32>>> {
        let total_tokens: usize = group.iter().map(|t| t.len()).sum();
        if total_tokens == 0 {
            return None;
        }
        let n_tokens = u32::try_from(total_tokens).ok()?;
        let n_seq = i32::try_from(group.len()).ok()?;
        let n_ubatch = match self.pin.pooling {
            Pooling::Cls => n_tokens,
            Pooling::Last => u32::try_from(total_tokens.min(MAX_UBATCH_TOKENS)).ok()?,
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
        let mut context = self.model.new_context(&self.backend, params).ok()?;

        let mut batch = LlamaBatch::new(total_tokens, n_seq);
        for (seq_id, tokens) in group.iter().enumerate() {
            batch
                .add_sequence(tokens, i32::try_from(seq_id).ok()?, true)
                .ok()?;
        }
        match self.pin.pooling {
            Pooling::Cls => context.encode(&mut batch).ok()?,
            Pooling::Last => context.decode(&mut batch).ok()?,
        }

        let mut vectors = Vec::with_capacity(group.len());
        for seq_id in 0..group.len() {
            let raw = context
                .embeddings_seq_ith(i32::try_from(seq_id).ok()?)
                .ok()?;
            vectors.push(self.serve_lane(raw)?);
        }
        Some(vectors)
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

    fn engine_facts(&self) -> EngineFacts {
        EngineFacts {
            runtime: RUNTIME.to_string(),
            device: "cpu".to_string(),
            requested_backend: "cpu".to_string(),
            resolved_backend: "cpu".to_string(),
            accelerated: false,
            degraded_reason: None,
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
            vectors.extend(isolate(&group, dims, &|items| self.encode_group(items)));
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

/// Binary-search failure isolation over an encode operation.
///
/// Encodes the whole slice; on failure splits in half and recurses, so a single poison
/// item costs about `log2(n)` extra encodes rather than `n`. When recursion reaches one
/// item that still fails, that item receives a zero vector of `dims` and the batch
/// succeeds — `vectors.len()` always equals `items.len()`, and the caller never sees an
/// error. Generic over `encode` so the algorithm is testable without a model.
pub fn isolate<T, F>(items: &[T], dims: usize, encode: &F) -> Vec<Vec<f32>>
where
    F: Fn(&[T]) -> Option<Vec<Vec<f32>>>,
    T: Clone,
{
    if items.is_empty() {
        return Vec::new();
    }
    if let Some(vectors) = encode(items) {
        if vectors.len() == items.len() {
            return vectors;
        }
    }
    if items.len() == 1 {
        eprintln!("julie-semantic-sidecar: substituting a zero vector for an unencodable input");
        return vec![vec![0.0; dims]];
    }
    let mid = items.len() / 2;
    let mut vectors = isolate(&items[..mid], dims, encode);
    vectors.extend(isolate(&items[mid..], dims, encode));
    vectors
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

    #[test]
    fn isolate_returns_encoder_output_when_the_batch_succeeds() {
        let encode =
            |items: &[usize]| Some(items.iter().map(|i| unit(*i as f32 + 1.0, 4)).collect());
        let vectors = isolate(&[0usize, 1, 2], 4, &encode);
        assert_eq!(vectors.len(), 3);
        assert!(vectors.iter().all(|v| v.len() == 4));
    }

    #[test]
    fn isolate_zero_vectors_only_the_poison_item() {
        let poison = 2usize;
        let encode = |items: &[usize]| {
            if items.contains(&poison) {
                None
            } else {
                Some(items.iter().map(|i| unit(*i as f32 + 1.0, 4)).collect())
            }
        };
        let vectors = isolate(&[0usize, 1, 2, 3, 4], 4, &encode);
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
        let encode = |_: &[usize]| None;
        let vectors = isolate(&[0usize, 1, 2], 4, &encode);
        assert_eq!(vectors, vec![vec![0.0; 4]; 3]);
    }

    #[test]
    fn isolate_rejects_a_count_mismatch_from_the_encoder() {
        let encode = |items: &[usize]| {
            if items.len() == 1 {
                Some(vec![unit(1.0, 4)])
            } else {
                Some(Vec::new())
            }
        };
        assert_eq!(isolate(&[0usize, 1], 4, &encode).len(), 2);
    }

    #[test]
    fn isolate_of_an_empty_batch_is_empty() {
        let encode = |_: &[usize]| None;
        assert!(isolate::<usize, _>(&[], 4, &encode).is_empty());
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
