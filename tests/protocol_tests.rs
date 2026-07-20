use julie_semantic_sidecar::engine_trait::{EmbedEngine, EmbedOutput, EngineError, Role};
use julie_semantic_sidecar::health::{Limits, MAX_BATCH_ITEMS, MAX_REQUEST_BYTES};
use julie_semantic_sidecar::protocol;
use serde_json::{json, Value};
use std::cell::Cell;

const DIMS: usize = 4;

struct FakeEngine {
    health: Value,
    fail_with: Option<EngineError>,
    vectors_override: Option<Vec<Vec<f32>>>,
    dims_override: Option<usize>,
    embed_calls: Cell<usize>,
}

impl FakeEngine {
    fn ready() -> Self {
        Self {
            health: json!({
                "ready": true,
                "dims": DIMS,
                "model_id": "qwen3-0.6b-f16",
                "runtime": "llama.cpp",
                "device": "cpu",
                "resolved_backend": "cpu",
                "accelerated": false,
                "degraded_reason": Value::Null,
                "capabilities": {
                    "cpu": {"available": true},
                    "cuda": {"available": false},
                    "directml": {"available": false},
                    "mps": {"available": false},
                    "metal": {"available": false},
                    "vulkan": {"available": false}
                },
                "load_policy": {
                    "requested_device_backend": "cpu",
                    "resolved_device_backend": "cpu",
                    "accelerated": false,
                    "degraded_reason": Value::Null
                }
            }),
            fail_with: None,
            vectors_override: None,
            dims_override: None,
            embed_calls: Cell::new(0),
        }
    }

    fn with_health(health: Value) -> Self {
        Self {
            health,
            ..Self::ready()
        }
    }

    fn failing(error: EngineError) -> Self {
        Self {
            fail_with: Some(error),
            ..Self::ready()
        }
    }

    fn returning(vectors: Vec<Vec<f32>>) -> Self {
        Self {
            vectors_override: Some(vectors),
            ..Self::ready()
        }
    }

    fn with_dims(dims: usize) -> Self {
        Self {
            dims_override: Some(dims),
            ..Self::ready()
        }
    }
}

impl EmbedEngine for FakeEngine {
    fn health_facts(&self) -> Result<Value, EngineError> {
        match &self.fail_with {
            Some(error) => Err(error.clone()),
            None => Ok(self.health.clone()),
        }
    }

    fn embed(&self, texts: &[String], role: Role) -> Result<EmbedOutput, EngineError> {
        self.embed_calls.set(self.embed_calls.get() + 1);
        if let Some(error) = &self.fail_with {
            return Err(error.clone());
        }
        let dims = self.dims_override.unwrap_or(DIMS);
        let vectors = match &self.vectors_override {
            Some(vectors) => vectors.clone(),
            None => texts
                .iter()
                .enumerate()
                .map(|(index, text)| {
                    let seed = index as f32 + text.len() as f32 + role_seed(role);
                    (0..dims).map(|d| seed + d as f32).collect()
                })
                .collect(),
        };
        Ok(EmbedOutput { dims, vectors })
    }
}

fn role_seed(role: Role) -> f32 {
    match role {
        Role::Query => 0.5,
        Role::Document => 0.0,
    }
}

fn responses<E: EmbedEngine>(engine: &E, input: &str) -> Vec<Value> {
    let mut output: Vec<u8> = Vec::new();
    protocol::run_loop(input.as_bytes(), &mut output, engine).expect("run_loop io");
    let text = String::from_utf8(output).expect("utf-8 output");
    text.lines()
        .map(|line| serde_json::from_str(line).expect("response line is json"))
        .collect()
}

fn one<E: EmbedEngine>(engine: &E, request: Value) -> Value {
    let mut all = responses(engine, &format!("{request}\n"));
    assert_eq!(all.len(), 1, "expected exactly one response");
    all.remove(0)
}

fn error_code(response: &Value) -> &str {
    response["error"]["code"].as_str().expect("error.code")
}

#[test]
fn a1_every_response_carries_schema_and_version() {
    let engine = FakeEngine::ready();
    let requests = [
        json!({"method": "health"}),
        json!({"method": "nope"}),
        json!({"method": "embed_query", "params": {"text": "x"}}),
    ];
    for request in requests {
        let response = one(&engine, request);
        assert_eq!(response["schema"], "julie.embedding.sidecar");
        assert_eq!(response["version"], 1);
    }
}

#[test]
fn a2_request_id_is_echoed() {
    let response = one(
        &FakeEngine::ready(),
        json!({"request_id": "x", "method": "health"}),
    );
    assert_eq!(response["request_id"], "x");
}

#[test]
fn a3_id_alias_is_echoed_as_request_id() {
    let response = one(&FakeEngine::ready(), json!({"id": "x", "method": "health"}));
    assert_eq!(response["request_id"], "x");
}

#[test]
fn a4_request_omitting_schema_and_version_is_handled_normally() {
    let response = one(&FakeEngine::ready(), json!({"method": "health"}));
    assert_eq!(response["result"]["ready"], true);
    assert!(response.get("error").is_none());
}

#[test]
fn a5_schema_mismatch_is_invalid_request_with_request_id_echo() {
    let response = one(
        &FakeEngine::ready(),
        json!({"schema": "other", "request_id": "r1", "method": "health"}),
    );
    assert_eq!(error_code(&response), "invalid_request");
    assert_eq!(response["request_id"], "r1");
}

#[test]
fn a5_version_mismatch_is_invalid_request_with_request_id_echo() {
    let response = one(
        &FakeEngine::ready(),
        json!({"version": 2, "request_id": "r2", "method": "health"}),
    );
    assert_eq!(error_code(&response), "invalid_request");
    assert_eq!(response["request_id"], "r2");
}

#[test]
fn a6_response_carries_exactly_one_of_result_or_error() {
    let engine = FakeEngine::ready();
    let cases = [
        json!({"method": "health"}),
        json!({"method": "embed_batch", "params": {"texts": ["a"]}}),
        json!({"method": "nope"}),
        json!({"method": 7}),
    ];
    for request in cases {
        let response = one(&engine, request);
        let has_result = response.get("result").is_some();
        let has_error = response.get("error").is_some();
        assert!(has_result ^ has_error, "exactly one of result/error");
    }
}

#[test]
fn a7_health_reports_ready_boolean_and_dims_when_ready() {
    let response = one(&FakeEngine::ready(), json!({"method": "health"}));
    assert_eq!(response["result"]["ready"], true);
    assert_eq!(response["result"]["dims"], DIMS);
}

#[test]
fn a7_health_may_report_not_ready_without_dims() {
    let engine = FakeEngine::with_health(json!({
        "ready": false,
        "degraded_reason": "model_not_prepared"
    }));
    let response = one(&engine, json!({"method": "health"}));
    assert_eq!(response["result"]["ready"], false);
    assert_eq!(response["result"]["degraded_reason"], "model_not_prepared");
    assert!(response["result"].get("dims").is_none());
}

#[test]
fn a8_health_capabilities_carry_the_four_reference_backends() {
    let response = one(&FakeEngine::ready(), json!({"method": "health"}));
    let capabilities = &response["result"]["capabilities"];
    for backend in ["cpu", "cuda", "directml", "mps"] {
        assert!(
            capabilities[backend]["available"].is_boolean(),
            "capabilities.{backend}.available must be boolean"
        );
    }
}

#[test]
fn a9_health_degradation_fields_reach_the_wire_unchanged() {
    let engine = FakeEngine::with_health(json!({
        "ready": true,
        "dims": DIMS,
        "accelerated": false,
        "degraded_reason": "vulkan slower than cpu in benchmark",
        "capabilities": {
            "cpu": {"available": true},
            "cuda": {"available": false},
            "directml": {"available": false},
            "mps": {"available": false}
        },
        "load_policy": {
            "requested_device_backend": "vulkan",
            "resolved_device_backend": "cpu",
            "accelerated": false,
            "degraded_reason": "vulkan slower than cpu in benchmark"
        }
    }));
    let result = one(&engine, json!({"method": "health"}))["result"].clone();
    assert!(!result["degraded_reason"].is_null());
    assert_eq!(result["load_policy"]["accelerated"], result["accelerated"]);
    assert_eq!(
        result["load_policy"]["degraded_reason"],
        result["degraded_reason"]
    );
}

#[test]
fn a10_embed_query_dims_equals_vector_length() {
    let response = one(
        &FakeEngine::ready(),
        json!({"method": "embed_query", "params": {"text": "hello"}}),
    );
    let result = &response["result"];
    assert_eq!(result["dims"], DIMS);
    assert_eq!(result["vector"].as_array().expect("vector").len(), DIMS);
}

#[test]
fn a11_embed_batch_returns_one_vector_of_dims_per_text() {
    let response = one(
        &FakeEngine::ready(),
        json!({"method": "embed_batch", "params": {"texts": ["a", "b", "c"]}}),
    );
    let result = &response["result"];
    assert_eq!(result["dims"], DIMS);
    let vectors = result["vectors"].as_array().expect("vectors");
    assert_eq!(vectors.len(), 3);
    for vector in vectors {
        assert_eq!(vector.as_array().expect("vector").len(), DIMS);
    }
}

#[test]
fn a12_empty_batch_returns_empty_vectors_without_error() {
    let response = one(
        &FakeEngine::ready(),
        json!({"method": "embed_batch", "params": {"texts": []}}),
    );
    assert_eq!(response["result"]["dims"], DIMS);
    assert_eq!(response["result"]["vectors"], json!([]));
    assert!(response.get("error").is_none());
}

#[test]
fn a13_embed_query_with_empty_text_succeeds_with_a_vector() {
    let response = one(
        &FakeEngine::ready(),
        json!({"method": "embed_query", "params": {"text": ""}}),
    );
    assert_eq!(
        response["result"]["vector"]
            .as_array()
            .expect("vector")
            .len(),
        DIMS
    );
}

#[test]
fn a14_embed_query_with_non_string_text_is_invalid_request() {
    let response = one(
        &FakeEngine::ready(),
        json!({"method": "embed_query", "params": {"text": 7}}),
    );
    assert_eq!(error_code(&response), "invalid_request");
}

#[test]
fn a15_embed_batch_with_a_non_string_element_is_invalid_request() {
    let response = one(
        &FakeEngine::ready(),
        json!({"method": "embed_batch", "params": {"texts": ["a", 7]}}),
    );
    assert_eq!(error_code(&response), "invalid_request");
}

#[test]
fn a15_embed_batch_with_non_array_texts_is_invalid_request() {
    let response = one(
        &FakeEngine::ready(),
        json!({"method": "embed_batch", "params": {"texts": "a"}}),
    );
    assert_eq!(error_code(&response), "invalid_request");
}

#[test]
fn a16_unparseable_line_is_invalid_json_with_empty_request_id() {
    let all = responses(&FakeEngine::ready(), "{not json\n");
    assert_eq!(all.len(), 1);
    assert_eq!(error_code(&all[0]), "invalid_json");
    assert_eq!(all[0]["request_id"], "");
}

#[test]
fn a16_non_object_json_line_is_invalid_request_with_empty_request_id() {
    let all = responses(&FakeEngine::ready(), "[1,2]\n");
    assert_eq!(error_code(&all[0]), "invalid_request");
    assert_eq!(all[0]["request_id"], "");
}

#[test]
fn a17_unknown_method_is_unknown_method() {
    let response = one(&FakeEngine::ready(), json!({"method": "nope"}));
    assert_eq!(error_code(&response), "unknown_method");
}

#[test]
fn a19_shutdown_returns_stopping_and_breaks_the_loop() {
    let input = format!(
        "{}\n{}\n",
        json!({"request_id": "s", "method": "shutdown"}),
        json!({"request_id": "after", "method": "health"})
    );
    let all = responses(&FakeEngine::ready(), &input);
    assert_eq!(all.len(), 1, "no response after shutdown");
    assert_eq!(all[0]["result"], json!({"stopping": true}));
    assert_eq!(all[0]["request_id"], "s");
}

#[test]
fn a19_error_response_to_shutdown_does_not_break_the_loop() {
    let input = format!(
        "{}\n{}\n",
        json!({"version": 2, "method": "shutdown"}),
        json!({"request_id": "after", "method": "health"})
    );
    let all = responses(&FakeEngine::ready(), &input);
    assert_eq!(all.len(), 2);
    assert_eq!(error_code(&all[0]), "invalid_request");
    assert_eq!(all[1]["request_id"], "after");
}

#[test]
fn a20_blank_line_produces_no_response_and_the_loop_continues() {
    let input = format!(
        "\n   \n{}\n",
        json!({"request_id": "q", "method": "health"})
    );
    let all = responses(&FakeEngine::ready(), &input);
    assert_eq!(all.len(), 1);
    assert_eq!(all[0]["request_id"], "q");
}

#[test]
fn a21_unrecognized_top_level_field_is_ignored() {
    let response = one(
        &FakeEngine::ready(),
        json!({"method": "health", "trace_context": {"span": 1}, "future": "x"}),
    );
    assert_eq!(response["result"]["ready"], true);
}

#[test]
fn a23_process_survives_every_error_condition_and_answers_the_next_request() {
    let input = format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n",
        json!({"schema": "other", "method": "health"}),
        json!({"version": 2, "method": "health"}),
        json!({"method": "embed_query", "params": {"text": 7}}),
        json!({"method": "embed_batch", "params": {"texts": [7]}}),
        "{not json",
        json!({"method": "nope"})
    );
    let input = format!(
        "{input}{}\n",
        json!({"request_id": "alive", "method": "health"})
    );
    let all = responses(&FakeEngine::ready(), &input);
    assert_eq!(all.len(), 7);
    let codes: Vec<&str> = all[..6].iter().map(error_code).collect();
    assert_eq!(
        codes,
        vec![
            "invalid_request",
            "invalid_request",
            "invalid_request",
            "invalid_request",
            "invalid_json",
            "unknown_method"
        ]
    );
    assert_eq!(all[6]["request_id"], "alive");
    assert_eq!(all[6]["result"]["ready"], true);
}

#[test]
fn request_id_takes_precedence_over_the_id_alias() {
    let response = one(
        &FakeEngine::ready(),
        json!({"request_id": "primary", "id": "alias", "method": "health"}),
    );
    assert_eq!(response["request_id"], "primary");
}

#[test]
fn invalid_request_id_is_rejected_even_when_a_valid_id_alias_is_present() {
    let response = one(
        &FakeEngine::ready(),
        json!({"request_id": 7, "id": "alias", "method": "health"}),
    );
    assert_eq!(error_code(&response), "invalid_request");
    assert_eq!(response["request_id"], "");
}

#[test]
fn non_string_id_alias_is_invalid_request() {
    let response = one(&FakeEngine::ready(), json!({"id": 7, "method": "health"}));
    assert_eq!(error_code(&response), "invalid_request");
    assert_eq!(response["request_id"], "");
}

#[test]
fn non_string_method_is_invalid_request() {
    let response = one(&FakeEngine::ready(), json!({"method": 7}));
    assert_eq!(error_code(&response), "invalid_request");
}

#[test]
fn missing_method_is_invalid_request() {
    let response = one(&FakeEngine::ready(), json!({"request_id": "m"}));
    assert_eq!(error_code(&response), "invalid_request");
}

#[test]
fn non_object_params_is_invalid_request() {
    let response = one(
        &FakeEngine::ready(),
        json!({"method": "health", "params": 7}),
    );
    assert_eq!(error_code(&response), "invalid_request");
}

#[test]
fn engine_failure_becomes_internal_error_and_the_loop_continues() {
    let engine = FakeEngine::failing(EngineError::new("RuntimeError", "backend exploded"));
    let input = format!(
        "{}\n{}\n",
        json!({"request_id": "boom", "method": "embed_query", "params": {"text": "a"}}),
        json!({"request_id": "next", "method": "embed_query", "params": {"text": "b"}})
    );
    let all = responses(&engine, &input);
    assert_eq!(all.len(), 2);
    assert_eq!(error_code(&all[0]), "internal_error");
    assert_eq!(all[0]["error"]["message"], "RuntimeError: backend exploded");
    assert_eq!(all[0]["request_id"], "boom");
    assert_eq!(error_code(&all[1]), "internal_error");
}

#[test]
fn batch_count_mismatch_from_the_engine_becomes_internal_error() {
    let engine = FakeEngine::returning(vec![vec![0.0; DIMS]]);
    let response = one(
        &engine,
        json!({"method": "embed_batch", "params": {"texts": ["a", "b"]}}),
    );
    assert_eq!(error_code(&response), "internal_error");
}

#[test]
fn vector_length_mismatch_from_the_engine_becomes_internal_error() {
    let engine = FakeEngine::returning(vec![vec![0.0; DIMS - 1]]);
    let response = one(
        &engine,
        json!({"method": "embed_query", "params": {"text": "a"}}),
    );
    assert_eq!(error_code(&response), "internal_error");
}

#[test]
fn embed_query_dims_echo_follows_the_engine() {
    let engine = FakeEngine::with_dims(512);
    let response = one(
        &engine,
        json!({"method": "embed_query", "params": {"text": "a"}}),
    );
    assert_eq!(response["result"]["dims"], 512);
    assert_eq!(
        response["result"]["vector"]
            .as_array()
            .expect("vector")
            .len(),
        512
    );
}

#[test]
fn responses_are_compact_single_line_json() {
    let mut output: Vec<u8> = Vec::new();
    protocol::run_loop(
        "{\"method\":\"health\"}\n".as_bytes(),
        &mut output,
        &FakeEngine::ready(),
    )
    .expect("run_loop io");
    let text = String::from_utf8(output).expect("utf-8");
    assert!(text.ends_with('\n'));
    assert_eq!(text.matches('\n').count(), 1);
    assert!(!text.contains(": "), "compact separators");
    assert!(text.starts_with("{\"schema\":\"julie.embedding.sidecar\",\"version\":1,"));
}

fn responses_with_limits<E: EmbedEngine>(engine: &E, input: &[u8], limits: Limits) -> Vec<Value> {
    let mut output: Vec<u8> = Vec::new();
    protocol::run_loop_with_limits(input, &mut output, engine, limits).expect("run_loop io");
    String::from_utf8(output)
        .expect("utf-8 output")
        .lines()
        .map(|line| serde_json::from_str(line).expect("response line is json"))
        .collect()
}

fn small_line_limits(max_request_bytes: usize) -> Limits {
    Limits {
        max_request_bytes,
        ..Limits::default()
    }
}

/// An `embed_query` request line padded to exactly `total_bytes`, terminator excluded.
fn request_line_of(total_bytes: usize) -> String {
    let build = |text: String| {
        json!({"request_id": "pad", "method": "embed_query", "params": {"text": text}}).to_string()
    };
    let padding = total_bytes - build(String::new()).len();
    let line = build("x".repeat(padding));
    assert_eq!(line.len(), total_bytes);
    line
}

#[test]
fn a_line_of_exactly_max_request_bytes_is_accepted() {
    let limits = small_line_limits(256);
    let input = format!("{}\n", request_line_of(256));
    let all = responses_with_limits(&FakeEngine::ready(), input.as_bytes(), limits);
    assert_eq!(all.len(), 1);
    assert_eq!(all[0]["request_id"], "pad");
    assert_eq!(all[0]["result"]["dims"], DIMS);
}

#[test]
fn a_line_over_max_request_bytes_is_invalid_request_and_the_loop_continues() {
    let limits = small_line_limits(256);
    let input = format!(
        "{}\n{}\n",
        request_line_of(257),
        json!({"request_id": "after", "method": "health"})
    );
    let all = responses_with_limits(&FakeEngine::ready(), input.as_bytes(), limits);
    assert_eq!(all.len(), 2);
    assert_eq!(error_code(&all[0]), "invalid_request");
    assert_eq!(all[0]["request_id"], "");
    assert!(
        all[0]["error"]["message"]
            .as_str()
            .expect("message")
            .contains("max_request_bytes"),
        "message names the limit"
    );
    assert_eq!(all[1]["request_id"], "after");
    assert_eq!(all[1]["result"]["ready"], true);
}

#[test]
fn an_unterminated_oversized_stream_ends_without_buffering_it() {
    let limits = small_line_limits(1024);
    let input = vec![b'x'; 4 * 1024 * 1024];
    let all = responses_with_limits(&FakeEngine::ready(), input.as_slice(), limits);
    assert_eq!(all.len(), 1);
    assert_eq!(error_code(&all[0]), "invalid_request");
}

#[test]
fn a_batch_of_exactly_max_batch_items_is_accepted() {
    let engine = FakeEngine::ready();
    let texts: Vec<String> = (0..MAX_BATCH_ITEMS)
        .map(|index| index.to_string())
        .collect();
    let response = one(
        &engine,
        json!({"method": "embed_batch", "params": {"texts": texts}}),
    );
    assert_eq!(
        response["result"]["vectors"]
            .as_array()
            .expect("vectors")
            .len(),
        MAX_BATCH_ITEMS
    );
    assert_eq!(engine.embed_calls.get(), 1);
}

#[test]
fn a_batch_over_max_batch_items_is_invalid_request_without_invoking_the_engine() {
    let engine = FakeEngine::ready();
    let texts: Vec<String> = (0..=MAX_BATCH_ITEMS)
        .map(|index| index.to_string())
        .collect();
    let response = one(
        &engine,
        json!({"method": "embed_batch", "params": {"texts": texts}}),
    );
    assert_eq!(error_code(&response), "invalid_request");
    assert!(
        response["error"]["message"]
            .as_str()
            .expect("message")
            .contains("max_batch_items"),
        "message names the limit"
    );
    assert_eq!(engine.embed_calls.get(), 0);
}

#[test]
fn default_limits_are_the_ones_health_advertises() {
    let limits = Limits::default();
    assert_eq!(limits.max_batch_items, MAX_BATCH_ITEMS);
    assert_eq!(limits.max_request_bytes, MAX_REQUEST_BYTES);
}

#[test]
fn invalid_utf8_line_is_invalid_json_and_the_loop_continues() {
    let mut input: Vec<u8> = vec![0xff, 0xfe, b'\n'];
    input.extend_from_slice(b"{\"request_id\":\"ok\",\"method\":\"health\"}\n");
    let mut output: Vec<u8> = Vec::new();
    protocol::run_loop(input.as_slice(), &mut output, &FakeEngine::ready()).expect("run_loop io");
    let text = String::from_utf8(output).expect("utf-8 output");
    let all: Vec<Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).expect("json"))
        .collect();
    assert_eq!(all.len(), 2);
    assert_eq!(error_code(&all[0]), "invalid_json");
    assert_eq!(all[1]["request_id"], "ok");
}
