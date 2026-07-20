//! NDJSON wire loop for the `julie.embedding.sidecar` v1 protocol.
//!
//! Stub: Task 2 owns the envelope parsing, method dispatch, and error codes.

use std::io::{BufRead, Write};

/// Schema identifier carried by every request and response envelope.
pub const SCHEMA: &str = "julie.embedding.sidecar";

/// Protocol version carried by every request and response envelope.
pub const VERSION: u32 = 1;

/// Serves the NDJSON protocol on stdin/stdout for `model_id` until EOF.
///
/// Stub: Task 5 loads the engine for `model_id` and hands it to [`run_loop`].
pub fn serve(_model_id: &str) -> std::io::Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    run_loop(stdin.lock(), stdout.lock())
}

/// Reads NDJSON requests until EOF, writing one response line per request.
///
/// The stub consumes stdin without answering; stdout purity is preserved because it
/// emits nothing.
pub fn run_loop<R: BufRead, W: Write>(input: R, _output: W) -> std::io::Result<()> {
    for line in input.lines() {
        line?;
    }
    Ok(())
}
