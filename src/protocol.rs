//! NDJSON wire loop for the frozen `julie.embedding.sidecar` v1 protocol.
//!
//! One JSON object per line on the reader, exactly one compact response line per answered
//! request on the writer, flushed after each. Blank lines are skipped without a response;
//! every other failure is answered with an error envelope so the loop — and the process —
//! survives. See `docs/contracts/semantic-sidecar-protocol-v1.md` in the Miller repository.

use crate::engine_trait::{EmbedEngine, EngineError, Role};
use serde::Serialize;
use serde_json::{Map, Value};
use std::io::{BufRead, Write};

/// Schema identifier carried by every request and response envelope.
pub const SCHEMA: &str = "julie.embedding.sidecar";

/// Protocol version carried by every request and response envelope.
pub const VERSION: u32 = 1;

const INVALID_REQUEST: &str = "invalid_request";
const INVALID_JSON: &str = "invalid_json";
const UNKNOWN_METHOD: &str = "unknown_method";
const INTERNAL_ERROR: &str = "internal_error";

/// Serves the NDJSON protocol on stdin/stdout for `model_id` until EOF or `shutdown`.
///
/// The model is loaded eagerly: a Qwen3 load costs about a second, and paying that inside
/// the first `embed` would blow that request's budget.
///
/// A model absent from the cache is **not** a startup failure. The contract's row B3 wants
/// a wire-conformant `health` reporting `ready: false` and the exact reason
/// `model_not_prepared`, which [`UnreadyEngine`](crate::engine_trait::UnreadyEngine)
/// renders; `embed` calls answer `internal_error` and the process stays alive. Any other
/// load failure — a present but unloadable model — fails loud: `serve` returns the error,
/// `main` prints it to stderr, and the process exits nonzero.
pub fn serve(model_id: &str) -> std::io::Result<()> {
    let pin = crate::manifest::by_id(model_id).ok_or_else(|| {
        std::io::Error::other(format!(
            "unknown model id '{model_id}'; run `prepare --model`"
        ))
    })?;
    let cache_dir = crate::prepare::cache_dir().map_err(|err| {
        std::io::Error::other(format!("cannot resolve the model cache: {}", err.message()))
    })?;
    if let Err(err) = crate::prepare::clean_stale_partials(&cache_dir) {
        eprintln!("julie-semantic-sidecar: could not clean stale partial downloads: {err}");
    }

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    match crate::engine::LlamaEngine::load(pin, &cache_dir) {
        Ok(engine) => run_loop(stdin.lock(), stdout.lock(), &engine),
        Err(err) if err.kind == "ModelNotPrepared" => {
            eprintln!("julie-semantic-sidecar: {err}");
            let engine = crate::engine_trait::UnreadyEngine::new(crate::health::MODEL_NOT_PREPARED);
            run_loop(stdin.lock(), stdout.lock(), &engine)
        }
        Err(err) => Err(std::io::Error::other(format!(
            "cannot load model '{model_id}' from {}: {err}",
            cache_dir.display()
        ))),
    }
}

/// Reads NDJSON requests from `input` until EOF or `shutdown`, writing responses to `output`.
///
/// Returns `Err` only on a writer I/O failure; malformed input is answered, never propagated.
pub fn run_loop<R: BufRead, W: Write, E: EmbedEngine>(
    mut input: R,
    mut output: W,
    engine: &E,
) -> std::io::Result<()> {
    let mut buffer: Vec<u8> = Vec::new();
    loop {
        buffer.clear();
        if input.read_until(b'\n', &mut buffer)? == 0 {
            return Ok(());
        }
        let Some(outcome) = handle_line(&buffer, engine) else {
            continue;
        };
        let line = serde_json::to_string(&outcome.response).map_err(std::io::Error::other)?;
        output.write_all(line.as_bytes())?;
        output.write_all(b"\n")?;
        output.flush()?;
        if outcome.stop {
            return Ok(());
        }
    }
}

struct Outcome {
    response: Response,
    stop: bool,
}

impl Outcome {
    fn reply(response: Response) -> Option<Self> {
        Some(Self {
            response,
            stop: false,
        })
    }
}

#[derive(Serialize)]
struct Response {
    schema: &'static str,
    version: u32,
    request_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<ErrorBody>,
}

#[derive(Serialize)]
struct ErrorBody {
    code: &'static str,
    message: String,
}

fn success(request_id: impl Into<String>, result: Value) -> Response {
    Response {
        schema: SCHEMA,
        version: VERSION,
        request_id: request_id.into(),
        result: Some(result),
        error: None,
    }
}

fn failure(
    request_id: impl Into<String>,
    code: &'static str,
    message: impl Into<String>,
) -> Response {
    Response {
        schema: SCHEMA,
        version: VERSION,
        request_id: request_id.into(),
        result: None,
        error: Some(ErrorBody {
            code,
            message: message.into(),
        }),
    }
}

fn handle_line<E: EmbedEngine>(raw: &[u8], engine: &E) -> Option<Outcome> {
    let Ok(text) = std::str::from_utf8(raw) else {
        return Outcome::reply(failure(
            "",
            INVALID_JSON,
            "invalid json: line is not valid UTF-8",
        ));
    };
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let request: Value = match serde_json::from_str(trimmed) {
        Ok(value) => value,
        Err(err) => {
            return Outcome::reply(failure("", INVALID_JSON, format!("invalid json: {err}")))
        }
    };
    let Value::Object(request) = request else {
        return Outcome::reply(failure("", INVALID_REQUEST, "request must be an object"));
    };
    Some(dispatch(&request, engine))
}

fn dispatch<E: EmbedEngine>(request: &Map<String, Value>, engine: &E) -> Outcome {
    let request_id = match extract_request_id(request) {
        Ok(id) => id,
        Err(message) => {
            return Outcome {
                response: failure("", INVALID_REQUEST, message),
                stop: false,
            }
        }
    };

    if let Err(message) = validate_schema(request).and_then(|()| validate_version(request)) {
        return Outcome {
            response: failure(request_id, INVALID_REQUEST, message),
            stop: false,
        };
    }

    let Some(Value::String(method)) = request.get("method") else {
        return Outcome {
            response: failure(request_id, INVALID_REQUEST, "method must be a string"),
            stop: false,
        };
    };

    let empty_params = Map::new();
    let params = match request.get("params") {
        None => &empty_params,
        Some(Value::Object(params)) => params,
        Some(_) => {
            return Outcome {
                response: failure(request_id, INVALID_REQUEST, "params must be an object"),
                stop: false,
            }
        }
    };

    match method.as_str() {
        "health" => Outcome {
            response: match engine.health_facts() {
                Ok(facts) => success(request_id, facts),
                Err(err) => internal_error(request_id, &err),
            },
            stop: false,
        },
        "embed_query" => Outcome {
            response: embed_query(request_id, params, engine),
            stop: false,
        },
        "embed_batch" => Outcome {
            response: embed_batch(request_id, params, engine),
            stop: false,
        },
        "shutdown" => Outcome {
            response: success(request_id, serde_json::json!({"stopping": true})),
            stop: true,
        },
        other => Outcome {
            response: failure(
                request_id,
                UNKNOWN_METHOD,
                format!("unknown method: {other}"),
            ),
            stop: false,
        },
    }
}

fn embed_query<E: EmbedEngine>(
    request_id: String,
    params: &Map<String, Value>,
    engine: &E,
) -> Response {
    let Some(Value::String(text)) = params.get("text") else {
        return failure(
            request_id,
            INVALID_REQUEST,
            "embed_query params.text must be a string",
        );
    };
    let texts = [text.clone()];
    let output = match engine.embed(&texts, Role::Query) {
        Ok(output) => output,
        Err(err) => return internal_error(request_id, &err),
    };
    if output.vectors.len() != 1 {
        return internal_error(
            request_id,
            &EngineError::new(
                "EmbeddingCountError",
                format!("expected 1 vector, got {}", output.vectors.len()),
            ),
        );
    }
    let vector = &output.vectors[0];
    if vector.len() != output.dims {
        return internal_error(
            request_id,
            &EngineError::new(
                "EmbeddingDimsError",
                format!("expected {} dims, got {}", output.dims, vector.len()),
            ),
        );
    }
    success(
        request_id,
        serde_json::json!({"dims": output.dims, "vector": vector}),
    )
}

fn embed_batch<E: EmbedEngine>(
    request_id: String,
    params: &Map<String, Value>,
    engine: &E,
) -> Response {
    let Some(Value::Array(raw_texts)) = params.get("texts") else {
        return failure(
            request_id,
            INVALID_REQUEST,
            "embed_batch params.texts must be an array",
        );
    };
    let mut texts: Vec<String> = Vec::with_capacity(raw_texts.len());
    for value in raw_texts {
        match value {
            Value::String(text) => texts.push(text.clone()),
            _ => {
                return failure(
                    request_id,
                    INVALID_REQUEST,
                    "embed_batch params.texts must contain only strings",
                )
            }
        }
    }
    let output = match engine.embed(&texts, Role::Document) {
        Ok(output) => output,
        Err(err) => return internal_error(request_id, &err),
    };
    if output.vectors.len() != texts.len() {
        return internal_error(
            request_id,
            &EngineError::new(
                "EmbeddingCountError",
                format!(
                    "expected {} vectors, got {}",
                    texts.len(),
                    output.vectors.len()
                ),
            ),
        );
    }
    if let Some((index, actual)) = output
        .vectors
        .iter()
        .enumerate()
        .find_map(|(index, vector)| (vector.len() != output.dims).then_some((index, vector.len())))
    {
        return internal_error(
            request_id,
            &EngineError::new(
                "EmbeddingDimsError",
                format!(
                    "expected {} dims at index {index}, got {actual}",
                    output.dims
                ),
            ),
        );
    }
    success(
        request_id,
        serde_json::json!({"dims": output.dims, "vectors": output.vectors}),
    )
}

fn internal_error(request_id: String, error: &EngineError) -> Response {
    failure(request_id, INTERNAL_ERROR, error.to_string())
}

fn extract_request_id(request: &Map<String, Value>) -> Result<String, &'static str> {
    for (key, message) in [
        ("request_id", "request_id must be a string"),
        ("id", "id must be a string"),
    ] {
        if let Some(value) = request.get(key) {
            return match value {
                Value::String(id) => Ok(id.clone()),
                _ => Err(message),
            };
        }
    }
    Ok(String::new())
}

fn validate_schema(request: &Map<String, Value>) -> Result<(), String> {
    match request.get("schema") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(schema)) if schema == SCHEMA => Ok(()),
        Some(other) => Err(format!("schema mismatch: expected '{SCHEMA}', got {other}")),
    }
}

fn validate_version(request: &Map<String, Value>) -> Result<(), String> {
    match request.get("version") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Number(version)) if version.as_f64() == Some(f64::from(VERSION)) => Ok(()),
        Some(other) => Err(format!(
            "unsupported version: expected {VERSION}, got {other}"
        )),
    }
}
