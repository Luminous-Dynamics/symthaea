// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Token Codebook — bidirectional mapping between Rust tokens and BinaryHVs.
//!
//! Enables decoding manifold centroids back to Rust code expressions.
//! Each token is pre-encoded as a deterministic BinaryHV from its name.
//! Decoding = find the top-k most similar tokens to a query HV.

use std::sync::OnceLock;

use symthaea_core::hdc::binary_hv::BinaryHV;

// ═══════════════════════════════════════════════════════════════════════════════
// TOKEN CATEGORIES
// ═══════════════════════════════════════════════════════════════════════════════

/// Category of a Rust token in the codebook.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenCategory {
    /// Language keywords: fn, let, mut, if, else, ...
    Keyword,
    /// Operators: +, -, *, /, ==, !=, &&, ||, ...
    Operator,
    /// Built-in and standard library types: i32, Vec, Option, ...
    Type,
    /// Common method-call patterns: .iter(), .map(), .collect(), ...
    Pattern,
    /// Literal values: 0, 1, true, false, ...
    Literal,
    /// Delimiters and punctuation: (, ), {, }, ::, ->, ...
    Delimiter,
    /// Control flow: break, continue, return, yield
    Control,
    /// Common compound patterns: .iter().map(), .to_string(), ...
    CommonPattern,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TOKEN ENTRY
// ═══════════════════════════════════════════════════════════════════════════════

/// A single entry in the codebook: token string + category + pre-encoded HV.
#[derive(Debug, Clone)]
pub struct TokenEntry {
    /// The token text (e.g., "fn", ".iter()", "Vec").
    pub token: String,
    /// Semantic category.
    pub category: TokenCategory,
    /// Deterministic BinaryHV encoding of this token.
    pub hv: BinaryHV,
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEED COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute a deterministic seed from a token string.
///
/// Uses the same byte-folding approach as `SkeletonSlot::new()` in
/// skeleton_synthesis.rs, ensuring consistency across the geodesic crate.
fn token_seed(token: &str) -> u64 {
    token
        .bytes()
        .enumerate()
        .fold(0u64, |acc, (i, b)| {
            acc.wrapping_add((b as u64) << (i % 8))
        })
}

// ═══════════════════════════════════════════════════════════════════════════════
// TOKEN CODEBOOK
// ═══════════════════════════════════════════════════════════════════════════════

/// Pre-encoded vocabulary of Rust tokens for manifold centroid decoding.
///
/// The codebook is built once from a fixed vocabulary of ~120 Rust tokens.
/// Each token is encoded to a deterministic [`BinaryHV`] via `BinaryHV::random(seed)`.
///
/// **Decoding** finds the most similar tokens to a query HV (brute-force
/// cosine/Hamming search over all entries).
///
/// **Encoding** looks up a token by name and returns its HV.
pub struct TokenCodebook {
    entries: Vec<TokenEntry>,
}

impl TokenCodebook {
    /// Create a codebook with the default Rust vocabulary.
    pub fn new() -> Self {
        let mut entries = Vec::with_capacity(140);

        // --- Keywords (20) ---
        for tok in KEYWORDS {
            entries.push(TokenEntry {
                token: tok.to_string(),
                category: TokenCategory::Keyword,
                hv: BinaryHV::random(token_seed(tok)),
            });
        }

        // --- Operators (17) ---
        for tok in OPERATORS {
            entries.push(TokenEntry {
                token: tok.to_string(),
                category: TokenCategory::Operator,
                hv: BinaryHV::random(token_seed(tok)),
            });
        }

        // --- Types (20) ---
        for tok in TYPES {
            entries.push(TokenEntry {
                token: tok.to_string(),
                category: TokenCategory::Type,
                hv: BinaryHV::random(token_seed(tok)),
            });
        }

        // --- Patterns (20) ---
        for tok in PATTERNS {
            entries.push(TokenEntry {
                token: tok.to_string(),
                category: TokenCategory::Pattern,
                hv: BinaryHV::random(token_seed(tok)),
            });
        }

        // --- Literals (6) ---
        for tok in LITERALS {
            entries.push(TokenEntry {
                token: tok.to_string(),
                category: TokenCategory::Literal,
                hv: BinaryHV::random(token_seed(tok)),
            });
        }

        // --- Delimiters (12) ---
        for tok in DELIMITERS {
            entries.push(TokenEntry {
                token: tok.to_string(),
                category: TokenCategory::Delimiter,
                hv: BinaryHV::random(token_seed(tok)),
            });
        }

        // --- Control (4) ---
        for tok in CONTROL {
            entries.push(TokenEntry {
                token: tok.to_string(),
                category: TokenCategory::Control,
                hv: BinaryHV::random(token_seed(tok)),
            });
        }

        // --- Common patterns (20) ---
        for tok in COMMON_PATTERNS {
            entries.push(TokenEntry {
                token: tok.to_string(),
                category: TokenCategory::CommonPattern,
                hv: BinaryHV::random(token_seed(tok)),
            });
        }

        Self { entries }
    }

    /// Decode: find the top-k tokens most similar to a query HV.
    ///
    /// Returns `(token_str, similarity)` pairs sorted by descending similarity.
    pub fn decode(&self, query: &BinaryHV, top_k: usize) -> Vec<(&str, f32)> {
        let mut scored: Vec<(&str, f32)> = self
            .entries
            .iter()
            .map(|e| (e.token.as_str(), query.similarity(&e.hv)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Decode with category filter — only considers tokens from the given category.
    pub fn decode_category(
        &self,
        query: &BinaryHV,
        category: TokenCategory,
        top_k: usize,
    ) -> Vec<(&str, f32)> {
        let mut scored: Vec<(&str, f32)> = self
            .entries
            .iter()
            .filter(|e| e.category == category)
            .map(|e| (e.token.as_str(), query.similarity(&e.hv)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Encode a token string to its HV (lookup by name).
    ///
    /// Returns `None` if the token is not in the vocabulary.
    pub fn encode(&self, token: &str) -> Option<BinaryHV> {
        self.entries
            .iter()
            .find(|e| e.token == token)
            .map(|e| e.hv)
    }

    /// Compose an expression from a sequence of decoded tokens.
    ///
    /// Takes a query HV, decodes the top matching tokens, and assembles them
    /// into a plausible Rust expression string. Uses category alternation
    /// heuristics to produce syntactically reasonable output.
    pub fn decode_expression(&self, query: &BinaryHV, max_tokens: usize) -> String {
        let candidates = self.decode(query, max_tokens * 2);
        if candidates.is_empty() {
            return "todo!()".to_string();
        }

        // Collect top tokens, preferring category diversity.
        let mut selected: Vec<&str> = Vec::with_capacity(max_tokens);
        let mut seen_categories = std::collections::HashSet::new();

        // First pass: pick the best token per category for diversity.
        for &(tok, _sim) in &candidates {
            if selected.len() >= max_tokens {
                break;
            }
            let cat = self
                .entries
                .iter()
                .find(|e| e.token == tok)
                .map(|e| e.category);
            if let Some(c) = cat {
                if seen_categories.insert(c) {
                    selected.push(tok);
                }
            }
        }

        // Second pass: fill remaining slots with highest-similarity tokens.
        if selected.len() < max_tokens {
            for &(tok, _sim) in &candidates {
                if selected.len() >= max_tokens {
                    break;
                }
                if !selected.contains(&tok) {
                    selected.push(tok);
                }
            }
        }

        if selected.is_empty() {
            return "todo!()".to_string();
        }

        // Assemble: join tokens, using context-aware spacing.
        let mut result = String::with_capacity(selected.len() * 8);
        for (i, tok) in selected.iter().enumerate() {
            if i > 0 && !is_prefix_delimiter(tok) && !is_suffix_delimiter(selected[i - 1]) {
                result.push(' ');
            }
            result.push_str(tok);
        }
        result
    }

    /// Number of entries in the codebook.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the codebook is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all entries.
    pub fn entries(&self) -> &[TokenEntry] {
        &self.entries
    }
}

impl Default for TokenCodebook {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GLOBAL LAZY CODEBOOK
// ═══════════════════════════════════════════════════════════════════════════════

/// Global lazy-initialized codebook (built once on first access).
///
/// Uses the same `OnceLock` pattern as other singletons in the geodesic crate.
pub fn codebook() -> &'static TokenCodebook {
    static CODEBOOK: OnceLock<TokenCodebook> = OnceLock::new();
    CODEBOOK.get_or_init(TokenCodebook::new)
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

/// Tokens that should not have a space before them.
fn is_prefix_delimiter(tok: &str) -> bool {
    matches!(tok, "(" | "{" | "[" | "." | "::" | "," | ";")
}

/// Tokens that should not have a space after them.
fn is_suffix_delimiter(tok: &str) -> bool {
    matches!(tok, "(" | "{" | "[" | "." | "::")
}

// ═══════════════════════════════════════════════════════════════════════════════
// VOCABULARY CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

const KEYWORDS: &[&str] = &[
    "fn", "let", "mut", "if", "else", "for", "while", "return", "match", "loop", "pub", "struct",
    "enum", "impl", "trait", "use", "mod", "where", "const", "type",
];

const OPERATORS: &[&str] = &[
    "+", "-", "*", "/", "%", "==", "!=", "<", ">", "<=", ">=", "&&", "||", "!", "&", "|", "=",
];

const TYPES: &[&str] = &[
    "i32", "i64", "u32", "u64", "f32", "f64", "usize", "isize", "bool", "char", "String", "str",
    "Vec", "Option", "Result", "Box", "Arc", "HashMap", "BTreeMap", "HashSet",
];

const PATTERNS: &[&str] = &[
    ".iter()",
    ".map()",
    ".filter()",
    ".collect()",
    ".unwrap()",
    ".unwrap_or()",
    ".expect()",
    ".is_some()",
    ".is_none()",
    ".ok()",
    ".err()",
    "Some()",
    "None",
    "Ok()",
    "Err()",
    ".len()",
    ".push()",
    ".pop()",
    ".contains()",
    ".clone()",
];

const LITERALS: &[&str] = &["0", "1", "true", "false", "0.0", "\"\""];

const DELIMITERS: &[&str] = &["(", ")", "{", "}", "[", "]", ";", ",", "::", "->", "=>", "."];

const CONTROL: &[&str] = &["break", "continue", "return", "yield"];

const COMMON_PATTERNS: &[&str] = &[
    ".to_string()",
    ".as_str()",
    ".into()",
    ".from()",
    ".new()",
    ".default()",
    ".iter().map()",
    ".iter().filter()",
    ".iter().collect()",
    ".iter().fold()",
    ".iter().enumerate()",
    ".iter().zip()",
    ".iter().any()",
    ".iter().all()",
    ".iter().find()",
    ".iter().position()",
    ".iter().count()",
    ".iter().sum()",
    ".iter().max()",
    ".iter().min()",
];

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_size() {
        let cb = TokenCodebook::new();
        // 20 + 17 + 20 + 20 + 6 + 12 + 4 + 20 = 119
        assert!(
            cb.len() >= 119,
            "codebook should have at least 119 entries, got {}",
            cb.len()
        );
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let cb = TokenCodebook::new();
        let fn_hv = cb.encode("fn").expect("'fn' should be in codebook");
        let decoded = cb.decode(&fn_hv, 1);
        assert_eq!(
            decoded[0].0, "fn",
            "encoding then decoding 'fn' should yield 'fn' as top-1"
        );
        assert!(
            (decoded[0].1 - 1.0).abs() < f32::EPSILON,
            "self-similarity should be 1.0"
        );
    }

    #[test]
    fn test_decode_top_k() {
        let cb = TokenCodebook::new();
        let if_hv = cb.encode("if").expect("'if' should be in codebook");
        let top5 = cb.decode(&if_hv, 5);
        assert_eq!(top5.len(), 5, "should return exactly 5 results");
        // Top-1 must be "if" itself
        assert_eq!(top5[0].0, "if");
        // Similarities should be descending
        for w in top5.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "results should be sorted descending by similarity"
            );
        }
    }

    #[test]
    fn test_decode_expression() {
        let cb = TokenCodebook::new();
        let let_hv = cb.encode("let").expect("'let' should be in codebook");
        let expr = cb.decode_expression(&let_hv, 3);
        assert!(
            !expr.is_empty(),
            "decoded expression should not be empty"
        );
        assert!(
            expr.contains("let"),
            "expression from 'let' HV should contain 'let', got: {expr}"
        );
    }

    #[test]
    fn test_category_filter() {
        let cb = TokenCodebook::new();
        let query = BinaryHV::random(0xDEAD);
        let keywords = cb.decode_category(&query, TokenCategory::Keyword, 10);
        assert!(
            !keywords.is_empty(),
            "should find at least one keyword match"
        );
        // Verify all returned tokens are actually keywords
        for (tok, _sim) in &keywords {
            let entry = cb.entries.iter().find(|e| e.token == *tok).unwrap();
            assert_eq!(
                entry.category,
                TokenCategory::Keyword,
                "category filter should only return keywords, got: {tok}"
            );
        }
    }

    #[test]
    fn test_deterministic_encoding() {
        let cb1 = TokenCodebook::new();
        let cb2 = TokenCodebook::new();
        let hv1 = cb1.encode("Vec").unwrap();
        let hv2 = cb2.encode("Vec").unwrap();
        assert_eq!(
            hv1.similarity(&hv2),
            1.0,
            "same token should produce identical HVs across codebook instances"
        );
    }

    #[test]
    fn test_global_codebook() {
        let cb = codebook();
        assert!(cb.len() >= 119);
        // Second access should return the same instance
        let cb2 = codebook();
        assert_eq!(cb.len(), cb2.len());
    }

    #[test]
    fn test_token_seed_determinism() {
        assert_eq!(token_seed("fn"), token_seed("fn"));
        assert_ne!(token_seed("fn"), token_seed("let"));
    }
}
