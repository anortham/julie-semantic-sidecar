//! Embedded model manifest: model id → sha256, size, source URL, and serving knobs.
//!
//! This module is the single source of pin truth for the binary. Every value below is
//! transcribed from the frozen § Model knob table of `semantic-sidecar-protocol-v1.md`
//! (itself sourced from `eval/model-bench/bench-pins.json`) and may not be changed
//! without a contract amendment — an embedding model is sensitive to every byte of the
//! instruction strings, and the sha256 values gate `prepare`'s download verification.

/// Pooling mode the model's representation is read with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// Final-token pooling — the EOS marker must survive truncation.
    Last,
    /// Leading classification-token pooling.
    Cls,
}

impl Pooling {
    /// Wire spelling reported through the `health` result.
    pub fn as_str(self) -> &'static str {
        match self {
            Pooling::Last => "last",
            Pooling::Cls => "cls",
        }
    }
}

/// Selection tier: which pin `prepare` and the server resolve to without an explicit id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// The pin served when no `--model` argument is given.
    Default,
    /// A smaller pin available by explicit id.
    Fallback,
}

impl Tier {
    /// Wire spelling of the tier.
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Default => "default",
            Tier::Fallback => "fallback",
        }
    }
}

/// One pinned model: acquisition identity plus the knobs that determine its vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelPin {
    /// Manifest id accepted by `prepare --model`.
    pub id: &'static str,
    /// Upstream model name.
    pub model: &'static str,
    /// GGUF file name inside the cache directory.
    pub file: &'static str,
    /// Source URL the weight file is downloaded from.
    pub url: &'static str,
    /// Lowercase hex sha256 the downloaded file must match before it is renamed into place.
    pub sha256: &'static str,
    /// Declared file size, checked against free space before download begins.
    pub size_bytes: u64,
    /// The model's native output dimensionality.
    pub native_dims: usize,
    /// Lanes the model supports; a single-element slice for non-MRL models.
    pub mrl_lanes: &'static [usize],
    /// Lane actually served, and therefore the `dims` a ready `health` declares.
    pub serve_dims: usize,
    /// Pooling mode applied to the model's token states.
    pub pooling: Pooling,
    /// Marker appended to every input before tokenization, when the model declares one.
    pub eos_marker: Option<&'static str>,
    /// Instruction prefixed by `embed_query`.
    pub query_instruction: &'static str,
    /// Instruction prefixed by `embed_batch`.
    pub document_instruction: &'static str,
    /// Token budget covering instruction prefix, text, and EOS marker together.
    pub max_text_tokens: usize,
    /// Manifest revision identifier reported through `health`.
    pub model_revision: &'static str,
    /// Selection tier.
    pub tier: Tier,
}

const PINS: &[ModelPin] = &[
    ModelPin {
        id: "qwen3-0.6b-f16",
        model: "Qwen3-Embedding-0.6B",
        file: "Qwen3-Embedding-0.6B-f16.gguf",
        url: "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-f16.gguf",
        sha256: "421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340",
        size_bytes: 1_197_629_632,
        native_dims: 1024,
        mrl_lanes: &[256, 512, 1024],
        serve_dims: 512,
        pooling: Pooling::Last,
        eos_marker: Some("<|endoftext|>"),
        query_instruction: "Instruct: Given a code search query, retrieve the code or documentation that answers it\nQuery: ",
        document_instruction: "",
        max_text_tokens: 32768,
        model_revision: "main",
        tier: Tier::Default,
    },
    ModelPin {
        id: "bge-small-en-v1.5-f32",
        model: "bge-small-en-v1.5",
        file: "bge-small-en-v1.5-f32.gguf",
        url: "https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-f32.gguf",
        sha256: "bf40c42ad7d89382e9ba7376d5c4b73f6b556cb541fab37aaa1da9c320149b65",
        size_bytes: 133_609_568,
        native_dims: 384,
        mrl_lanes: &[384],
        serve_dims: 384,
        pooling: Pooling::Cls,
        eos_marker: None,
        query_instruction: "Represent this sentence for searching relevant passages: ",
        document_instruction: "",
        max_text_tokens: 512,
        model_revision: "main",
        tier: Tier::Fallback,
    },
];

/// Every pinned model, in tier order.
pub fn manifest() -> &'static [ModelPin] {
    PINS
}

/// Resolves an exact manifest id, or `None` when the id is not pinned.
pub fn by_id(id: &str) -> Option<&'static ModelPin> {
    PINS.iter().find(|pin| pin.id == id)
}

/// The pin served when no `--model` argument is given.
pub fn default_model() -> &'static ModelPin {
    PINS.iter()
        .find(|pin| pin.tier == Tier::Default)
        .expect("manifest carries a default-tier pin")
}
