//! Token-budget truncation of prefixed input before EOS append.
//!
//! Implements the frozen six-step algorithm of `semantic-sidecar-protocol-v1.md`
//! § Truncation semantics. Steps 1–2 (sanitize, prefix the role instruction) and step 6
//! (append the EOS marker) belong to the caller; this module owns steps 3–5 — the budget
//! arithmetic, the tail cut, and the round-trip stability rule.
//!
//! Token operations are injected rather than imported so the algorithm is testable without
//! loading a model.

/// Token budget available to the text body itself.
///
/// The model's `max_text_tokens` covers the entire input, so the EOS marker and the
/// tokenizer's own special tokens are reserved out of it before the body is measured.
/// Saturating: a budget smaller than its own reserves yields zero rather than wrapping.
pub fn body_budget(
    max_text_tokens: usize,
    eos_reserve: usize,
    special_token_overhead: usize,
) -> usize {
    max_text_tokens.saturating_sub(eos_reserve + special_token_overhead)
}

/// Fits `prefixed` to the model's token budget, returning the text body to embed.
///
/// `prefixed` must already be sanitized and carry its role instruction; the EOS marker is
/// appended by the caller afterwards, which is what makes it unconditionally survive.
/// `tokenize` must tokenize exactly as embedding input is tokenized but **without** the
/// tokenizer-added special tokens — those are accounted for by `special_token_overhead`.
///
/// Input at or below budget is returned unchanged, byte for byte. Over-budget input is
/// tail-truncated to the budget and then shrunk further, one token at a time, until it
/// detokenizes to text that retokenizes to itself — so a string-level and a token-level
/// implementation embed the same tokens.
pub fn fit(
    prefixed: &str,
    max_text_tokens: usize,
    eos_reserve: usize,
    special_token_overhead: usize,
    tokenize: impl Fn(&str) -> Vec<i32>,
    detokenize: impl Fn(&[i32]) -> String,
) -> String {
    let budget = body_budget(max_text_tokens, eos_reserve, special_token_overhead);
    let tokens = tokenize(prefixed);
    if tokens.len() <= budget {
        return prefixed.to_string();
    }

    let mut kept = budget;
    while kept > 0 {
        let candidate = detokenize(&tokens[..kept]);
        if tokenize(&candidate) == tokens[..kept] {
            return candidate;
        }
        kept -= 1;
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAGMENT: i32 = 0;
    const SECTION_LEAD: i32 = 1000;

    fn tokenize(text: &str) -> Vec<i32> {
        text.chars()
            .flat_map(|c| match c {
                '§' => vec![SECTION_LEAD, FRAGMENT],
                other => vec![other as i32],
            })
            .collect()
    }

    fn detokenize(tokens: &[i32]) -> String {
        tokens
            .iter()
            .map(|t| match *t {
                FRAGMENT => String::new(),
                SECTION_LEAD => "§".to_string(),
                code => char::from_u32(code as u32)
                    .expect("test tokens are char codes")
                    .to_string(),
            })
            .collect()
    }

    fn fit_text(text: &str, max_text_tokens: usize, eos: usize, overhead: usize) -> String {
        fit(text, max_text_tokens, eos, overhead, tokenize, detokenize)
    }

    #[test]
    fn body_budget_subtracts_both_reserves() {
        assert_eq!(body_budget(32768, 1, 1), 32766);
        assert_eq!(body_budget(512, 0, 2), 510);
    }

    #[test]
    fn body_budget_saturates_instead_of_wrapping() {
        assert_eq!(body_budget(1, 1, 2), 0);
    }

    #[test]
    fn below_budget_text_is_returned_unchanged() {
        assert_eq!(fit_text("hello", 512, 0, 2), "hello");
    }

    #[test]
    fn text_at_exactly_the_budget_is_returned_unchanged() {
        let text = "a".repeat(510);
        assert_eq!(fit_text(&text, 512, 0, 2), text);
    }

    #[test]
    fn text_one_token_over_the_budget_is_cut_to_the_budget() {
        let text = "a".repeat(511);
        assert_eq!(fit_text(&text, 512, 0, 2), "a".repeat(510));
    }

    #[test]
    fn bge_over_budget_text_cuts_to_five_hundred_ten_tokens() {
        let text = "b".repeat(3000);
        let fitted = fit_text(&text, 512, 0, 2);
        assert_eq!(tokenize(&fitted).len(), 510);
    }

    #[test]
    fn qwen3_over_budget_text_cuts_to_thirty_two_thousand_seven_hundred_sixty_six_tokens() {
        let text = "q".repeat(40_000);
        let fitted = fit_text(&text, 32768, 1, 1);
        assert_eq!(tokenize(&fitted).len(), 32766);
    }

    #[test]
    fn truncation_is_tail_only_so_the_instruction_prefix_survives() {
        let text = format!("Query: {}", "z".repeat(1000));
        let fitted = fit_text(&text, 512, 0, 2);
        assert!(fitted.starts_with("Query: "));
    }

    #[test]
    fn unstable_round_trip_drops_one_more_trailing_token() {
        let text = "ab§cd";
        assert_eq!(tokenize(text).len(), 6);
        assert_eq!(fit_text(text, 3, 0, 0), "ab");
    }

    #[test]
    fn stable_round_trip_keeps_the_full_budget() {
        let text = "abcdef";
        assert_eq!(fit_text(text, 3, 0, 0), "abc");
    }

    #[test]
    fn a_budget_consumed_entirely_by_reserves_yields_empty_text() {
        assert_eq!(fit_text("abcdef", 2, 1, 1), "");
    }
}
