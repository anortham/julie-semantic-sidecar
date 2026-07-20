//! Engine abstraction keeping [`crate::protocol`] pure and testable without a model.
//!
//! [`EmbedEngine`] is the entire surface the wire loop needs: a health payload it passes
//! through verbatim, and one embedding entry point parameterised by role. Task 5 supplies
//! the llama.cpp implementation; Task 3 assembles the health payload behind
//! [`EmbedEngine::health_facts`].

use serde_json::Value;
use std::fmt;

/// Instruction-template role selected by the method name, per the v1 contract.
///
/// `embed_query` selects [`Role::Query`], `embed_batch` selects [`Role::Document`]; callers
/// never pass a role on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// The model's `query_instruction` is prefixed to each input.
    Query,
    /// The model's `document_instruction` is prefixed to each input.
    Document,
}

/// One homogeneous batch of embeddings: a single `dims` plus one vector per input.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbedOutput {
    /// Output dimensionality shared by every vector in `vectors`.
    pub dims: usize,
    /// One vector per input text, in input order.
    pub vectors: Vec<Vec<f32>>,
}

/// A failure raised while handling a well-formed request.
///
/// The wire loop renders it as an `internal_error` envelope with the message
/// `"{kind}: {message}"`, mirroring the reference's `"{ExceptionType}: {message}"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineError {
    /// Short failure class, rendered as the message prefix.
    pub kind: String,
    /// Human-readable detail.
    pub message: String,
}

impl EngineError {
    /// Builds an error from a failure class and a detail message.
    pub fn new(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl std::error::Error for EngineError {}

/// The embedding backend behind the wire protocol.
pub trait EmbedEngine {
    /// Returns the complete `health` result object, including `ready`.
    ///
    /// The wire loop passes the value through verbatim as `result` and only requires it to
    /// be a JSON object; every health invariant in the contract (`capabilities`,
    /// `load_policy`, the degradation rule, `dims` when ready) is the engine's obligation.
    fn health_facts(&self) -> Result<Value, EngineError>;

    /// Embeds `texts` under `role`, returning one vector per input in input order.
    fn embed(&self, texts: &[String], role: Role) -> Result<EmbedOutput, EngineError>;
}

/// Placeholder engine that reports `ready: false` and refuses to embed.
///
/// [`crate::protocol::serve`] uses it so the binary speaks a conformant protocol before
/// Task 5 lands the real llama.cpp engine.
#[derive(Debug, Clone)]
pub struct UnreadyEngine {
    reason: String,
}

impl UnreadyEngine {
    /// Builds an unready engine reporting `reason` as its `degraded_reason`.
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl EmbedEngine for UnreadyEngine {
    fn health_facts(&self) -> Result<Value, EngineError> {
        Ok(serde_json::json!({
            "ready": false,
            "degraded_reason": self.reason,
        }))
    }

    fn embed(&self, _texts: &[String], _role: Role) -> Result<EmbedOutput, EngineError> {
        Err(EngineError::new("EngineNotReady", self.reason.clone()))
    }
}
