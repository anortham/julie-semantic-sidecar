//! `prepare` subcommand: model acquisition into the shared cache.
//!
//! Stub: Task 4 owns atomic download, sha256 verification, cache locking, disk
//! preflight, and offline failure messaging.

use std::process::ExitCode;

/// Downloads and verifies the manifest model, defaulting to the tier in
/// [`crate::DEFAULT_MODEL_ID`] when `model_id` is `None`.
pub fn run(model_id: Option<&str>) -> ExitCode {
    let model = model_id.unwrap_or(crate::DEFAULT_MODEL_ID);
    eprintln!("julie-semantic-sidecar: prepare is not implemented yet (model {model})");
    ExitCode::FAILURE
}
