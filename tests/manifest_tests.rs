use julie_semantic_sidecar::health::{self, BackendCapabilities, EngineFacts, Limits, ModelState};
use julie_semantic_sidecar::manifest::{self, Pooling, Tier};
use serde_json::Value;

const QWEN_QUERY_INSTRUCTION: &str = "Instruct: Given a code search query, retrieve the code or documentation that answers it\nQuery: ";
const BGE_QUERY_INSTRUCTION: &str = "Represent this sentence for searching relevant passages: ";

#[test]
fn manifest_holds_exactly_the_two_pinned_models_in_stable_order() {
    let pins = manifest::manifest();
    assert_eq!(pins.len(), 2);
    assert_eq!(pins[0].id, "qwen3-0.6b-f16");
    assert_eq!(pins[1].id, "bge-small-en-v1.5-f32");
}

#[test]
fn qwen3_pin_matches_the_frozen_knob_table_byte_for_byte() {
    let pin = manifest::by_id("qwen3-0.6b-f16").expect("qwen3 pin present");

    assert_eq!(pin.id, "qwen3-0.6b-f16");
    assert_eq!(pin.model, "Qwen3-Embedding-0.6B");
    assert_eq!(pin.file, "Qwen3-Embedding-0.6B-f16.gguf");
    assert_eq!(
        pin.url,
        "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-f16.gguf"
    );
    assert_eq!(
        pin.sha256,
        "421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340"
    );
    assert_eq!(pin.size_bytes, 1_197_629_632);
    assert_eq!(pin.native_dims, 1024);
    assert_eq!(pin.mrl_lanes, &[256, 512, 1024]);
    assert_eq!(pin.serve_dims, 512);
    assert_eq!(pin.pooling, Pooling::Last);
    assert_eq!(pin.eos_marker, Some("<|endoftext|>"));
    assert_eq!(pin.query_instruction, QWEN_QUERY_INSTRUCTION);
    assert_eq!(pin.document_instruction, "");
    assert_eq!(pin.max_text_tokens, 32768);
    assert_eq!(pin.model_revision, "main");
    assert_eq!(pin.tier, Tier::Fallback);
}

#[test]
fn qwen3_query_instruction_carries_a_real_newline_and_a_trailing_space() {
    let pin = manifest::by_id("qwen3-0.6b-f16").expect("qwen3 pin present");
    assert!(pin.query_instruction.contains('\n'));
    assert!(!pin.query_instruction.contains("\\n"));
    assert!(pin.query_instruction.ends_with("Query: "));
    assert_eq!(pin.query_instruction.matches('\n').count(), 1);
}

#[test]
fn bge_pin_matches_the_frozen_knob_table_byte_for_byte() {
    let pin = manifest::by_id("bge-small-en-v1.5-f32").expect("bge pin present");

    assert_eq!(pin.id, "bge-small-en-v1.5-f32");
    assert_eq!(pin.model, "bge-small-en-v1.5");
    assert_eq!(pin.file, "bge-small-en-v1.5-f32.gguf");
    assert_eq!(
        pin.url,
        "https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-f32.gguf"
    );
    assert_eq!(
        pin.sha256,
        "bf40c42ad7d89382e9ba7376d5c4b73f6b556cb541fab37aaa1da9c320149b65"
    );
    assert_eq!(pin.size_bytes, 133_609_568);
    assert_eq!(pin.native_dims, 384);
    assert_eq!(pin.mrl_lanes, &[384]);
    assert_eq!(pin.serve_dims, 384);
    assert_eq!(pin.pooling, Pooling::Cls);
    assert_eq!(pin.eos_marker, None);
    assert_eq!(pin.query_instruction, BGE_QUERY_INSTRUCTION);
    assert_eq!(pin.document_instruction, "");
    assert_eq!(pin.max_text_tokens, 512);
    assert_eq!(pin.model_revision, "main");
    assert_eq!(pin.tier, Tier::Default);
}

#[test]
fn bge_query_instruction_keeps_its_trailing_space() {
    let pin = manifest::by_id("bge-small-en-v1.5-f32").expect("bge pin present");
    assert!(pin.query_instruction.ends_with("passages: "));
    assert!(!pin.query_instruction.contains('\n'));
}

#[test]
fn lookup_is_exact_and_rejects_unknown_ids() {
    assert!(manifest::by_id("qwen3-0.6b-f16").is_some());
    assert!(manifest::by_id("bge-small-en-v1.5-f32").is_some());
    assert!(manifest::by_id("bge-small-f32").is_none());
    assert!(manifest::by_id("arctic-embed-s-f16").is_none());
    assert!(manifest::by_id("QWEN3-0.6B-F16").is_none());
    assert!(manifest::by_id("").is_none());
}

#[test]
fn default_tier_resolution_selects_the_bge_pin() {
    let pin = manifest::default_model();
    assert_eq!(pin.id, "bge-small-en-v1.5-f32");
    assert_eq!(pin.tier, Tier::Default);
    assert_eq!(pin.id, julie_semantic_sidecar::DEFAULT_MODEL_ID);
}

#[test]
fn exactly_one_pin_carries_the_default_tier() {
    let defaults = manifest::manifest()
        .iter()
        .filter(|pin| pin.tier == Tier::Default)
        .count();
    assert_eq!(defaults, 1);
}

#[test]
fn pooling_and_tier_render_as_the_contract_wire_strings() {
    assert_eq!(Pooling::Last.as_str(), "last");
    assert_eq!(Pooling::Cls.as_str(), "cls");
    assert_eq!(Tier::Default.as_str(), "default");
    assert_eq!(Tier::Fallback.as_str(), "fallback");
}

fn all_capabilities() -> BackendCapabilities {
    BackendCapabilities {
        cpu: true,
        cuda: false,
        directml: false,
        mps: true,
        metal: true,
        vulkan: false,
    }
}

fn engine_on(requested: &str, resolved: &str, degraded_reason: Option<&str>) -> EngineFacts {
    EngineFacts {
        runtime: "llama.cpp".to_string(),
        device: resolved.to_string(),
        requested_backend: requested.to_string(),
        resolved_backend: resolved.to_string(),
        accelerated: resolved != "cpu",
        degraded_reason: degraded_reason.map(str::to_string),
        capabilities: all_capabilities(),
        llama_cpp_build: "b10068".to_string(),
    }
}

fn ready_health() -> Value {
    let pin = manifest::default_model();
    health::build(
        &ModelState::Ready {
            pin,
            dims: pin.serve_dims,
        },
        &engine_on("metal", "metal", None),
        Limits::default(),
        "0.1.0",
    )
}

#[test]
fn ready_health_reports_dims_and_the_full_contract_field_list() {
    let value = ready_health();

    assert_eq!(value["ready"], true);
    assert_eq!(value["dims"], 384);
    assert!(value["degraded_reason"].is_null());
    assert_eq!(value["model_id"], "bge-small-en-v1.5-f32");
    assert_eq!(value["model_sha256"], manifest::default_model().sha256);
    assert_eq!(value["model_revision"], "main");
    assert_eq!(value["runtime"], "llama.cpp");
    assert_eq!(value["device"], "metal");
    assert_eq!(value["resolved_backend"], "metal");
    assert_eq!(value["accelerated"], true);
    assert_eq!(value["pooling"], "cls");
    assert_eq!(value["normalization"], "l2");
    assert_eq!(value["instruction_policy_version"], 1);
    assert_eq!(value["max_text_tokens"], 512);
    assert_eq!(value["max_batch_items"], health::MAX_BATCH_ITEMS);
    assert_eq!(value["max_request_bytes"], health::MAX_REQUEST_BYTES);
    assert_eq!(value["native_dims"], 384);
    assert_eq!(value["mrl_lanes"], serde_json::json!([384]));
    assert_eq!(value["llama_cpp_build"], "b10068");
    assert_eq!(value["sidecar_version"], "0.1.0");
}

#[test]
fn max_batch_items_matches_the_contracts_250_text_converge_bound() {
    assert_eq!(health::MAX_BATCH_ITEMS, 250);
}

#[test]
fn missing_model_reports_not_ready_with_the_exact_model_not_prepared_reason() {
    let value = health::build(
        &ModelState::NotPrepared {
            pin: manifest::default_model(),
        },
        &engine_on("metal", "metal", None),
        Limits::default(),
        "0.1.0",
    );

    assert_eq!(value["ready"], false);
    assert_eq!(value["degraded_reason"], "model_not_prepared");
    assert_eq!(
        value["load_policy"]["degraded_reason"],
        "model_not_prepared"
    );
    assert!(value.get("dims").is_none());
    assert_eq!(value["model_id"], "bge-small-en-v1.5-f32");
    assert_eq!(value["native_dims"], 384);
}

#[test]
fn degraded_backend_mirrors_accelerated_and_reason_into_load_policy() {
    let value = health::build(
        &ModelState::Ready {
            pin: manifest::default_model(),
            dims: manifest::default_model().serve_dims,
        },
        &engine_on("vulkan", "cpu", Some("benchmark_cpu_faster")),
        Limits::default(),
        "0.1.0",
    );

    assert_eq!(value["ready"], true);
    assert_eq!(value["accelerated"], false);
    assert_eq!(value["degraded_reason"], "benchmark_cpu_faster");
    assert_eq!(value["load_policy"]["requested_device_backend"], "vulkan");
    assert_eq!(value["load_policy"]["resolved_device_backend"], "cpu");
    assert_eq!(value["load_policy"]["accelerated"], value["accelerated"]);
    assert_eq!(
        value["load_policy"]["degraded_reason"],
        value["degraded_reason"]
    );
}

#[test]
fn backend_mismatch_without_a_supplied_reason_still_yields_a_non_null_reason() {
    let value = health::build(
        &ModelState::Ready {
            pin: manifest::default_model(),
            dims: manifest::default_model().serve_dims,
        },
        &engine_on("vulkan", "cpu", None),
        Limits::default(),
        "0.1.0",
    );

    assert!(!value["degraded_reason"].is_null());
    assert_eq!(
        value["load_policy"]["degraded_reason"],
        value["degraded_reason"]
    );
}

#[test]
fn matching_backends_leave_both_degraded_reasons_null() {
    let value = ready_health();
    assert!(value["degraded_reason"].is_null());
    assert!(value["load_policy"]["degraded_reason"].is_null());
    assert_eq!(value["load_policy"]["accelerated"], value["accelerated"]);
}

#[test]
fn the_four_torch_compat_capability_keys_are_always_present_as_boolean_objects() {
    for value in [
        ready_health(),
        health::build(
            &ModelState::NotPrepared {
                pin: manifest::by_id("bge-small-en-v1.5-f32").expect("bge pin present"),
            },
            &engine_on("cpu", "cpu", None),
            Limits::default(),
            "0.1.0",
        ),
    ] {
        let capabilities = value["capabilities"]
            .as_object()
            .expect("capabilities is an object");
        for key in ["cpu", "cuda", "directml", "mps"] {
            let entry = capabilities
                .get(key)
                .unwrap_or_else(|| panic!("capability {key} present"));
            assert!(
                entry["available"].is_boolean(),
                "capability {key} has boolean available"
            );
        }
    }
}

#[test]
fn additive_metal_and_vulkan_capabilities_ride_alongside_the_torch_keys() {
    let value = ready_health();
    assert_eq!(value["capabilities"]["metal"]["available"], true);
    assert_eq!(value["capabilities"]["vulkan"]["available"], false);
    assert_eq!(value["capabilities"]["cpu"]["available"], true);
    assert_eq!(value["capabilities"]["directml"]["available"], false);
}

#[test]
fn comparison_model_health_reports_qwen_identity_and_dimensions() {
    let pin = manifest::by_id("qwen3-0.6b-f16").expect("qwen3 pin present");
    let value = health::build(
        &ModelState::Ready { pin, dims: 512 },
        &engine_on("cpu", "cpu", None),
        Limits::default(),
        "0.1.0",
    );

    assert_eq!(value["dims"], 512);
    assert_eq!(value["native_dims"], 1024);
    assert_eq!(value["mrl_lanes"], serde_json::json!([256, 512, 1024]));
    assert_eq!(value["pooling"], "last");
    assert_eq!(value["max_text_tokens"], 32768);
    assert_eq!(value["model_id"], "qwen3-0.6b-f16");
}
