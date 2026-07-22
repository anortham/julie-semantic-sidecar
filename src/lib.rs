//! `julie-semantic-sidecar` — a stdio embedding sidecar speaking the frozen
//! `julie.embedding.sidecar` v1 protocol.
//!
//! The library target exists so integration tests exercise the same code paths the
//! binary runs; `src/main.rs` is only a verb dispatcher.

pub mod backend_select;
pub mod engine;
pub mod engine_trait;
pub mod health;
pub mod manifest;
pub mod prepare;
pub mod protocol;
pub mod sanitize;
pub mod stdio_guard;
pub mod truncate;

/// Manifest id of the model served when no `--model` argument is given.
pub const DEFAULT_MODEL_ID: &str = "bge-small-en-v1.5-f32";

/// Semantic version of this binary, single-sourced from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
