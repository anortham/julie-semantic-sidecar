//! Engine tests.
//!
//! Pure tests run always. Model-backed tests are `#[ignore]`d because they load real GGUF
//! weights from the shared cache: run them with
//! `cargo test --test engine_tests -- --ignored --test-threads=1`. Single-threaded is
//! required — `LlamaBackend::init` is a process-global initialisation and two engines
//! loading concurrently race on it.

use julie_semantic_sidecar::engine::{isolate, EncodeFailure, LlamaEngine, LLAMA_CPP_BUILD};
use julie_semantic_sidecar::engine_trait::{EmbedEngine, Role};
use julie_semantic_sidecar::manifest;
use julie_semantic_sidecar::sanitize::sanitize;
use julie_semantic_sidecar::truncate::{body_budget, fit};

const QWEN3: &str = "qwen3-0.6b-f16";
const BGE: &str = "bge-small-en-v1.5-f32";

fn pin(id: &str) -> &'static manifest::ModelPin {
    manifest::by_id(id).expect("id is pinned in the manifest")
}

fn load(id: &str) -> LlamaEngine {
    LlamaEngine::load_from_default_cache(pin(id))
        .unwrap_or_else(|err| panic!("{id} must be prepared into the shared cache: {err}"))
}

fn norm(vector: &[f32]) -> f32 {
    vector.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() / (norm(a) * norm(b))
}

#[test]
fn qwen3_budget_reserves_one_token_for_eos_and_one_for_special_tokens() {
    assert_eq!(body_budget(pin(QWEN3).max_text_tokens, 1, 1), 32766);
}

#[test]
fn bge_budget_reserves_two_tokens_for_special_tokens_and_none_for_eos() {
    assert_eq!(body_budget(pin(BGE).max_text_tokens, 0, 2), 510);
}

#[test]
fn sanitized_blank_input_becomes_the_placeholder_before_prefixing() {
    assert_eq!(sanitize("\0  \0"), "[empty]");
}

#[test]
fn fit_keeps_the_instruction_prefix_and_cuts_the_tail() {
    let tokenize = |s: &str| s.chars().map(|c| c as i32).collect::<Vec<_>>();
    let detokenize = |t: &[i32]| {
        t.iter()
            .map(|c| char::from_u32(*c as u32).expect("char code"))
            .collect::<String>()
    };
    let prefixed = format!("{}{}", pin(BGE).query_instruction, "x".repeat(1000));
    let fitted = fit(&prefixed, 512, 0, 2, tokenize, detokenize);
    assert!(fitted.starts_with(pin(BGE).query_instruction));
    assert_eq!(tokenize(&fitted).len(), 510);
}

#[test]
fn isolation_substitutes_a_zero_vector_only_for_the_failing_item() {
    let encode = |items: &[usize]| {
        if items.contains(&3) {
            Err(EncodeFailure::Item)
        } else {
            Ok(items.iter().map(|_| vec![0.5f32; 4]).collect())
        }
    };
    let vectors = isolate(&[0usize, 1, 2, 3, 4, 5], 4, &encode).expect("an item failure isolates");
    assert_eq!(vectors.len(), 6);
    assert_eq!(vectors[3], vec![0.0; 4]);
    assert!(vectors
        .iter()
        .enumerate()
        .all(|(i, v)| i == 3 || v[0] == 0.5));
}

#[test]
fn a_systemic_encoder_failure_errors_the_request_instead_of_returning_zero_vectors() {
    let encode = |_: &[usize]| Err(EncodeFailure::systemic("ContextAlloc", "null reference"));
    let err = isolate(&[0usize, 1, 2, 3], 4, &encode).expect_err("a broken backend must not pass");
    assert_eq!(err.kind, "ContextAlloc");
}

#[test]
fn a_systemic_failure_surfacing_only_at_the_leaf_still_errors_the_request() {
    let encode = |items: &[usize]| {
        if items.len() == 1 {
            Err(EncodeFailure::systemic(
                "Decode",
                "Decode Error -2: unknown",
            ))
        } else {
            Err(EncodeFailure::Item)
        }
    };
    let err = isolate(&[0usize, 1, 2, 3], 4, &encode).expect_err("a broken backend must not pass");
    assert_eq!(err.kind, "Decode");
}

#[test]
fn the_llama_build_string_names_the_exact_crate_pin() {
    assert!(LLAMA_CPP_BUILD.contains("0.1.151"));
}

#[test]
#[ignore = "loads real GGUF weights from the shared cache"]
fn qwen3_serves_five_hundred_twelve_unit_norm_dimensions() {
    let engine = load(QWEN3);
    let texts = vec!["fn main() {}".to_string(), "a database index".to_string()];
    let out = engine
        .embed(&texts, Role::Document)
        .expect("embed succeeds");

    assert_eq!(out.dims, 512);
    assert_eq!(out.vectors.len(), 2);
    for vector in &out.vectors {
        assert_eq!(vector.len(), 512);
        assert!(
            (norm(vector) - 1.0).abs() <= 1e-3,
            "norm was {}",
            norm(vector)
        );
    }
}

#[test]
#[ignore = "loads real GGUF weights from the shared cache"]
fn qwen3_measures_one_eos_token_and_one_special_token() {
    let engine = load(QWEN3);
    assert_eq!(engine.eos_reserve(), 1);
    assert_eq!(engine.special_token_overhead(), 1);
}

#[test]
#[ignore = "loads real GGUF weights from the shared cache"]
fn qwen3_embeds_a_budget_length_input_without_error() {
    let engine = load(QWEN3);
    let text = "let value = compute(input); ".repeat(4000);
    let out = engine
        .embed(&[text], Role::Document)
        .expect("a budget-length input truncates rather than failing");

    assert_eq!(out.vectors.len(), 1);
    assert!((norm(&out.vectors[0]) - 1.0).abs() <= 1e-3);
    assert_ne!(
        out.vectors[0],
        vec![0.0; 512],
        "must not be the zero-vector fallback"
    );
}

#[test]
#[ignore = "loads real GGUF weights from the shared cache"]
fn qwen3_health_declares_the_served_lane() {
    let engine = load(QWEN3);
    let health = engine.health_facts().expect("health builds");
    assert_eq!(health["ready"], true);
    assert_eq!(health["dims"], 512);
    assert_eq!(health["native_dims"], 1024);
    assert_eq!(health["pooling"], "last");
}

#[test]
#[ignore = "loads real GGUF weights from the shared cache"]
fn bge_serves_three_hundred_eighty_four_unit_norm_dimensions() {
    let engine = load(BGE);
    let out = engine
        .embed(&["a small sentence".to_string()], Role::Query)
        .expect("embed succeeds");

    assert_eq!(out.dims, 384);
    assert_eq!(out.vectors[0].len(), 384);
    assert!((norm(&out.vectors[0]) - 1.0).abs() <= 1e-3);
}

#[test]
#[ignore = "loads real GGUF weights from the shared cache"]
fn bge_measures_two_special_tokens_and_no_eos() {
    let engine = load(BGE);
    assert_eq!(engine.eos_reserve(), 0);
    assert_eq!(engine.special_token_overhead(), 2);
}

#[test]
#[ignore = "loads real GGUF weights from the shared cache"]
fn bge_over_budget_text_embeds_as_its_own_truncation() {
    let engine = load(BGE);
    let long = "the quick brown fox jumps over the lazy dog ".repeat(70);
    assert!(long.len() >= 3000);

    let full = engine
        .embed(std::slice::from_ref(&long), Role::Document)
        .expect("embed");
    // Everything past the budget is dropped, so a longer input embeds identically.
    let longer = format!("{long}{long}");
    let extended = engine.embed(&[longer], Role::Document).expect("embed");

    assert!(
        cosine(&full.vectors[0], &extended.vectors[0]) >= 0.9999,
        "truncation must make the tail irrelevant, cosine was {}",
        cosine(&full.vectors[0], &extended.vectors[0])
    );
}

/// Prose whose bge WordPiece tokenization contains 9- and 11-byte pieces.
///
/// Verified against the cached vocabulary: tokens 20785, 27059, 7809, 27425 and 14067
/// need 9 bytes and token 26520 needs 11. Truncation is the only path that detokenizes,
/// so a piece longer than the initial buffer guess only surfaces on an over-budget input.
const MULTI_BYTE_PIECE_PROSE: &str = "The full rebuild path never merges into the live \
artifact. It extracts into a rebuild database, verifies the artifact metadata, and then \
atomically promotes the rebuilt file over the served one. Readers keep the write-ahead log \
from checkpointing. ";

#[test]
#[ignore = "loads real GGUF weights from the shared cache"]
fn bge_truncates_prose_whose_pieces_exceed_the_initial_buffer_guess() {
    let engine = load(BGE);
    // Long enough to force the truncation detokenize path, which is where a piece
    // larger than the first buffer guess used to fail the whole request.
    let text = MULTI_BYTE_PIECE_PROSE.repeat(40);

    let out = engine
        .embed(std::slice::from_ref(&text), Role::Document)
        .expect("a multi-byte WordPiece must not fail detokenization during truncation");

    assert_eq!(out.vectors[0].len(), 384);
    assert!((norm(&out.vectors[0]) - 1.0).abs() <= 1e-3);
    assert_ne!(
        out.vectors[0],
        vec![0.0; 384],
        "must embed for real, not fall back to a zero vector"
    );
}

#[test]
#[ignore = "loads real GGUF weights from the shared cache"]
fn qwen3_truncates_prose_whose_pieces_exceed_the_initial_buffer_guess() {
    let engine = load(QWEN3);
    let text = MULTI_BYTE_PIECE_PROSE.repeat(1200);

    let out = engine
        .embed(std::slice::from_ref(&text), Role::Query)
        .expect("a multi-byte piece must not fail detokenization during truncation");

    assert_eq!(out.vectors[0].len(), 512);
    assert!((norm(&out.vectors[0]) - 1.0).abs() <= 1e-3);
}

#[test]
#[ignore = "loads real GGUF weights from the shared cache"]
fn a_batch_keeps_count_and_stays_alive_across_degenerate_inputs() {
    let engine = load(BGE);
    let texts = vec![
        String::new(),
        "\0\0".to_string(),
        "   ".to_string(),
        "a real sentence".to_string(),
    ];
    let out = engine
        .embed(&texts, Role::Document)
        .expect("embed succeeds");

    assert_eq!(out.vectors.len(), texts.len());
    for vector in &out.vectors {
        assert_eq!(vector.len(), 384);
        assert!(
            (norm(vector) - 1.0).abs() <= 1e-3,
            "blank inputs embed [empty], not zeros"
        );
    }
}
