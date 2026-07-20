//! Engine-layer input sanitization.
//!
//! Scope note that later tasks must preserve — sanitization is engine-layer only: a
//! non-string wire value is `invalid_request` at the protocol layer and never reaches
//! here. What does reach here is every well-formed string, including the empty one, which
//! the contract requires be embedded rather than rejected
//! (`semantic-sidecar-protocol-v1.md` § Per-item failure isolation).

/// Text substituted for an input that carries no embeddable content.
pub const EMPTY_PLACEHOLDER: &str = "[empty]";

/// Strips NUL bytes and substitutes [`EMPTY_PLACEHOLDER`] for blank input.
///
/// Never fails: every input string maps to a non-empty output string. A string that is
/// empty, whitespace-only, or blank once its NUL bytes are removed becomes the literal
/// `[empty]`.
pub fn sanitize(text: &str) -> String {
    let stripped: String = text.chars().filter(|c| *c != '\0').collect();
    if stripped.trim().is_empty() {
        EMPTY_PLACEHOLDER.to_string()
    } else {
        stripped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_text_passes_through_unchanged() {
        assert_eq!(sanitize("fn main() {}"), "fn main() {}");
    }

    #[test]
    fn nul_bytes_are_stripped_from_text_that_survives() {
        assert_eq!(sanitize("ab\0cd"), "abcd");
    }

    #[test]
    fn empty_string_becomes_the_placeholder() {
        assert_eq!(sanitize(""), EMPTY_PLACEHOLDER);
    }

    #[test]
    fn whitespace_only_string_becomes_the_placeholder() {
        assert_eq!(sanitize("  \t\n "), EMPTY_PLACEHOLDER);
    }

    #[test]
    fn string_blank_after_nul_strip_becomes_the_placeholder() {
        assert_eq!(sanitize("\0\0"), EMPTY_PLACEHOLDER);
    }

    #[test]
    fn string_of_nuls_and_whitespace_becomes_the_placeholder() {
        assert_eq!(sanitize(" \0\t\0 "), EMPTY_PLACEHOLDER);
    }

    #[test]
    fn interior_whitespace_is_preserved() {
        assert_eq!(sanitize("  a  b  "), "  a  b  ");
    }
}
