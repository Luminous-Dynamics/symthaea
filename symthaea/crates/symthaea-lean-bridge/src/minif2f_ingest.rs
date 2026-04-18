// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # miniF2F-v2 automated ingestion pipeline
//!
//! Parses miniF2F-v2 Lean 4 `theorem NAME BINDERS : STMT := by sorry` files
//! and translates them to `FolFormulaExt` for the Phase 2 W4 proof cascade.
//! Replaces the hand-translation step that produced the 32-fixture 78.1%
//! baseline (see `docs/phase3-findings.md`) with a programmatic pipeline.
//!
//! ## Three-tier scorecard
//!
//! A miniF2F source file goes through three gates. We record which gate it
//! passes to localize failures:
//!
//! 1. **Parsed** — tokenizer + parser produced a well-formed `LeanTheorem`.
//! 2. **Translated** — `translate_theorem()` emitted a valid `FolFormulaExt`.
//! 3. **Accepted** — the downstream emitter → `lake env lean` closed the
//!    goal (handled by the existing `render_fol_ext_file` cascade).
//!
//! Failure at stage 1 is a parser gap (Lean 4 surface syntax we don't yet
//! recognize). Failure at stage 2 is a translator gap (a shape the parser
//! captured but `FolFormulaExt` doesn't have a constructor for). Failure at
//! stage 3 is a tactic-cascade gap (the shape ingests cleanly but the
//! proof search doesn't close it). Each gap class has a distinct next
//! move, so the distinction is operationally important.
//!
//! ## Scope (this commit)
//!
//! Tokenizer + AST types. Parser and translator stubs land in follow-up
//! commits to keep each step reviewable. The tokenizer targets the
//! operator set used by the 178-file candidate pool from
//! `scripts/filter_minif2f.sh`: arithmetic `+ - * / ^`, relations
//! `= ≠ < ≤ > ≥`, connectives `∧ ∨ → ↔ ¬`, universal binder `∀`,
//! base types `ℝ ℤ ℕ`, integer + rational literals, identifiers with
//! unicode subscripts (`h₀`, `x₁`).

use std::fmt;

// ════════════════════════════════════════════════════════════════════════
// AST — what the parser produces
// ════════════════════════════════════════════════════════════════════════

/// Numeric base type as declared in a binder: `(x : ℝ)`, `(n : ℕ)`, …
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeanType {
    Real,
    Int,
    Nat,
}

/// Arithmetic expression. Intentionally narrow: only the shapes that
/// `FolFormulaExt::Term` can represent. Anything outside (e.g.
/// `Real.sqrt`, `abs`, `Finset`) should fail at parse time with a
/// specific `IngestError` variant so the scorecard can attribute.
#[derive(Debug, Clone, PartialEq)]
pub enum LeanTerm {
    /// Named variable — `a`, `x`, `h₀` (hypothesis names stay in this
    /// form until the parser resolves them).
    Var(String),
    /// Integer literal (may be negative; the parser eagerly folds
    /// unary-minus over integer literals into the sign).
    IntLit(i64),
    /// Binary arithmetic. The translator maps these to `Term::Add/Sub/Mul/Div/Pow`.
    Bin(BinOp, Box<LeanTerm>, Box<LeanTerm>),
    /// Unary minus. The translator maps this to `Term::neg()`.
    Neg(Box<LeanTerm>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

/// Relation between two arithmetic expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelOp {
    Eq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
}

/// A logical statement — what appears after the `:` in a binder or as
/// the theorem goal. Built atomically from `LeanTerm` relations, then
/// composed via logical connectives.
#[derive(Debug, Clone, PartialEq)]
pub enum LeanStatement {
    /// `lhs OP rhs` — the atomic propositional shape.
    Rel(RelOp, LeanTerm, LeanTerm),
    /// `p ∧ q`
    And(Box<LeanStatement>, Box<LeanStatement>),
    /// `p ∨ q`
    Or(Box<LeanStatement>, Box<LeanStatement>),
    /// `p → q`
    Implies(Box<LeanStatement>, Box<LeanStatement>),
    /// `p ↔ q` — translated as `(p → q) ∧ (q → p)` downstream.
    Iff(Box<LeanStatement>, Box<LeanStatement>),
    /// `¬ p`
    Not(Box<LeanStatement>),
}

/// A parameter of a theorem. Two distinct shapes:
///
/// - `TypedVar` — `(x : ℝ)` declares `x` at type `ℝ`. The translator
///   emits a `∀ x : ℝ, …` binder around the theorem body.
/// - `Hyp` — `(h₀ : 3a + 2b = 12)` is a hypothesis. The translator
///   chains these as `hyp → goal` implications.
#[derive(Debug, Clone, PartialEq)]
pub enum LeanBinder {
    TypedVar { name: String, ty: LeanType },
    Hyp { name: String, stmt: LeanStatement },
}

/// A parsed miniF2F theorem:
///   `theorem NAME (binder)* : goal := by sorry`
#[derive(Debug, Clone, PartialEq)]
pub struct LeanTheorem {
    pub name: String,
    pub binders: Vec<LeanBinder>,
    pub goal: LeanStatement,
}

// ════════════════════════════════════════════════════════════════════════
// Error taxonomy — every failure has a named bucket for the scorecard
// ════════════════════════════════════════════════════════════════════════

/// Errors surfaced by the ingestion pipeline. Variants are intentionally
/// fine-grained so the three-tier scorecard can attribute failures. Use
/// `IngestError::category()` when tallying.
#[derive(Debug, Clone, PartialEq)]
pub enum IngestError {
    /// Tokenizer hit a character it doesn't recognize.
    UnknownChar { ch: char, offset: usize },
    /// Integer literal overflowed `i64`.
    IntLitOverflow { lit: String, offset: usize },
    /// Parser expected one token, found another.
    Unexpected {
        expected: &'static str,
        found: String,
        offset: usize,
    },
    /// Reached EOF mid-construct.
    UnexpectedEof { expected: &'static str },
    /// Source doesn't start with `theorem`.
    NotATheorem,
    /// Construct is outside the supported fragment (e.g. `Real.sqrt`,
    /// `Finset`, `Nat.Prime`). The upstream filter in
    /// `scripts/filter_minif2f.sh` should reject these before they
    /// reach us, so hitting this variant indicates a filter gap.
    OutOfScope { reason: String },
    /// Translator couldn't map a parsed construct to `FolFormulaExt`.
    UnsupportedTranslation { reason: String },
}

impl IngestError {
    /// Coarse category for scorecard tallies: "parse" vs "translate".
    pub fn category(&self) -> &'static str {
        match self {
            IngestError::UnknownChar { .. }
            | IngestError::IntLitOverflow { .. }
            | IngestError::Unexpected { .. }
            | IngestError::UnexpectedEof { .. }
            | IngestError::NotATheorem => "parse",
            IngestError::OutOfScope { .. } => "parse",
            IngestError::UnsupportedTranslation { .. } => "translate",
        }
    }
}

impl fmt::Display for IngestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IngestError::UnknownChar { ch, offset } => {
                write!(f, "unknown character {ch:?} at offset {offset}")
            }
            IngestError::IntLitOverflow { lit, offset } => {
                write!(
                    f,
                    "integer literal {lit:?} overflows i64 at offset {offset}"
                )
            }
            IngestError::Unexpected {
                expected,
                found,
                offset,
            } => {
                write!(f, "expected {expected}, found {found:?} at offset {offset}")
            }
            IngestError::UnexpectedEof { expected } => {
                write!(f, "unexpected EOF; expected {expected}")
            }
            IngestError::NotATheorem => {
                write!(f, "source does not begin with `theorem`")
            }
            IngestError::OutOfScope { reason } => {
                write!(f, "out of scope: {reason}")
            }
            IngestError::UnsupportedTranslation { reason } => {
                write!(f, "unsupported translation: {reason}")
            }
        }
    }
}

impl std::error::Error for IngestError {}

// ════════════════════════════════════════════════════════════════════════
// Tokenizer
// ════════════════════════════════════════════════════════════════════════

/// A lexed token. Spans are tracked as byte offsets into the original
/// source so error messages can pinpoint locations.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    // Keywords
    Theorem,
    By,
    Sorry,
    Forall, // ∀

    // Numeric base types
    TyReal, // ℝ
    TyInt,  // ℤ
    TyNat,  // ℕ

    // Identifiers + literals
    /// Identifier — may contain unicode subscripts like `h₀`, `x₁`.
    Ident(String),
    /// Non-negative integer literal. Sign is handled by the parser as
    /// unary minus.
    IntLit(i64),

    // Punctuation
    LParen,
    RParen,
    Colon,
    Comma,
    /// `:=`
    Assign,

    // Arithmetic
    Plus,
    Minus,
    Star,
    Slash,
    Caret,

    // Relations (unicode and ASCII)
    Eq,
    NotEq, // ≠
    Lt,
    Le, // ≤
    Gt,
    Ge, // ≥

    // Logical connectives (unicode)
    And,     // ∧
    Or,      // ∨
    Implies, // →
    Iff,     // ↔
    Not,     // ¬

    /// End-of-input sentinel — simplifies parser lookahead.
    Eof,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use TokenKind::*;
        match self {
            Theorem => write!(f, "theorem"),
            By => write!(f, "by"),
            Sorry => write!(f, "sorry"),
            Forall => write!(f, "∀"),
            TyReal => write!(f, "ℝ"),
            TyInt => write!(f, "ℤ"),
            TyNat => write!(f, "ℕ"),
            Ident(s) => write!(f, "{s}"),
            IntLit(n) => write!(f, "{n}"),
            LParen => write!(f, "("),
            RParen => write!(f, ")"),
            Colon => write!(f, ":"),
            Comma => write!(f, ","),
            Assign => write!(f, ":="),
            Plus => write!(f, "+"),
            Minus => write!(f, "-"),
            Star => write!(f, "*"),
            Slash => write!(f, "/"),
            Caret => write!(f, "^"),
            Eq => write!(f, "="),
            NotEq => write!(f, "≠"),
            Lt => write!(f, "<"),
            Le => write!(f, "≤"),
            Gt => write!(f, ">"),
            Ge => write!(f, "≥"),
            And => write!(f, "∧"),
            Or => write!(f, "∨"),
            Implies => write!(f, "→"),
            Iff => write!(f, "↔"),
            Not => write!(f, "¬"),
            Eof => write!(f, "<eof>"),
        }
    }
}

/// Tokenize a Lean 4 source snippet covering the miniF2F-v2 arithmetic
/// subset. Skips whitespace + line comments (`-- …`). Block comments
/// `/- … -/` are also elided.
///
/// Returns tokens in order with a trailing `Eof`. Caller holds the
/// `Vec<Token>` for parser lookahead.
pub fn tokenize(src: &str) -> Result<Vec<Token>, IngestError> {
    let mut out = Vec::new();
    let bytes = src.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        // Whitespace — ASCII only; Lean files don't use non-ASCII
        // whitespace beyond ordinary spaces/newlines.
        let b = bytes[i];
        if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
            i += 1;
            continue;
        }

        // Line comment: `-- …` to end of line.
        if i + 1 < bytes.len() && bytes[i] == b'-' && bytes[i + 1] == b'-' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // Block comment: `/- … -/`, possibly nested.
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'-' {
            let mut depth = 1;
            i += 2;
            while depth > 0 && i + 1 < bytes.len() {
                if bytes[i] == b'/' && bytes[i + 1] == b'-' {
                    depth += 1;
                    i += 2;
                } else if bytes[i] == b'-' && bytes[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            continue;
        }

        let start = i;

        // Integer literal.
        if b.is_ascii_digit() {
            let mut end = i;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            let lit = &src[i..end];
            let n = lit
                .parse::<i64>()
                .map_err(|_| IngestError::IntLitOverflow {
                    lit: lit.to_string(),
                    offset: start,
                })?;
            out.push(Token {
                kind: TokenKind::IntLit(n),
                offset: start,
            });
            i = end;
            continue;
        }

        // ASCII identifier start: letter or underscore.
        if b.is_ascii_alphabetic() || b == b'_' {
            let mut end = i;
            while end < bytes.len() {
                let c = bytes[end];
                if c.is_ascii_alphanumeric() || c == b'_' || c == b'\'' {
                    end += 1;
                } else if c >= 0x80 {
                    // Unicode continuation — might be a subscript digit
                    // like `₀` (U+2080..U+2089). Advance by full char.
                    if let Some((ch, ch_len)) = next_char(&src[end..]) {
                        if is_ident_continue(ch) {
                            end += ch_len;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            let ident = src[i..end].to_string();
            let kind = match ident.as_str() {
                "theorem" => TokenKind::Theorem,
                "by" => TokenKind::By,
                "sorry" => TokenKind::Sorry,
                _ => TokenKind::Ident(ident),
            };
            out.push(Token {
                kind,
                offset: start,
            });
            i = end;
            continue;
        }

        // ASCII single-char punctuation / operators.
        let single = match b {
            b'(' => Some(TokenKind::LParen),
            b')' => Some(TokenKind::RParen),
            b',' => Some(TokenKind::Comma),
            b'+' => Some(TokenKind::Plus),
            b'-' => Some(TokenKind::Minus),
            b'*' => Some(TokenKind::Star),
            b'/' => Some(TokenKind::Slash),
            b'^' => Some(TokenKind::Caret),
            b'=' => Some(TokenKind::Eq),
            b'<' => Some(TokenKind::Lt),
            b'>' => Some(TokenKind::Gt),
            _ => None,
        };
        if let Some(kind) = single {
            out.push(Token {
                kind,
                offset: start,
            });
            i += 1;
            continue;
        }

        // Two-char ASCII: `:=` vs bare `:`.
        if b == b':' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                out.push(Token {
                    kind: TokenKind::Assign,
                    offset: start,
                });
                i += 2;
            } else {
                out.push(Token {
                    kind: TokenKind::Colon,
                    offset: start,
                });
                i += 1;
            }
            continue;
        }

        // Unicode operator / keyword. Dispatch by first char.
        if b >= 0x80 {
            let (ch, ch_len) = next_char(&src[i..]).ok_or(IngestError::UnknownChar {
                ch: '\u{FFFD}',
                offset: start,
            })?;
            let kind = match ch {
                '∀' => TokenKind::Forall,
                'ℝ' => TokenKind::TyReal,
                'ℤ' => TokenKind::TyInt,
                'ℕ' => TokenKind::TyNat,
                '≠' => TokenKind::NotEq,
                '≤' => TokenKind::Le,
                '≥' => TokenKind::Ge,
                '∧' => TokenKind::And,
                '∨' => TokenKind::Or,
                '→' => TokenKind::Implies,
                '↔' => TokenKind::Iff,
                '¬' => TokenKind::Not,
                _ => return Err(IngestError::UnknownChar { ch, offset: start }),
            };
            out.push(Token {
                kind,
                offset: start,
            });
            i += ch_len;
            continue;
        }

        return Err(IngestError::UnknownChar {
            ch: b as char,
            offset: start,
        });
    }
    out.push(Token {
        kind: TokenKind::Eof,
        offset: src.len(),
    });
    Ok(out)
}

/// UTF-8 decode the first code point of `s`, returning `(char, byte_len)`.
fn next_char(s: &str) -> Option<(char, usize)> {
    let mut it = s.char_indices();
    let (_, ch) = it.next()?;
    let len = match it.next() {
        Some((n, _)) => n,
        None => s.len(),
    };
    Some((ch, len))
}

/// Is `ch` a valid continuation character in an identifier? Permits
/// the Unicode subscript digits `₀..₉` (U+2080..U+2089) used in
/// miniF2F hypothesis names like `h₀`, `h₁`. Not a general Lean
/// identifier-continue predicate — only as broad as the in-scope
/// miniF2F surface requires.
fn is_ident_continue(ch: char) -> bool {
    matches!(ch, '₀'..='₉')
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<TokenKind> {
        tokenize(src)
            .unwrap()
            .into_iter()
            .map(|t| t.kind)
            .filter(|k| !matches!(k, TokenKind::Eof))
            .collect()
    }

    #[test]
    fn tokenize_empty_gives_only_eof() {
        let toks = tokenize("").unwrap();
        assert_eq!(toks.len(), 1);
        assert_eq!(toks[0].kind, TokenKind::Eof);
    }

    #[test]
    fn tokenize_theorem_keyword() {
        assert_eq!(kinds("theorem"), vec![TokenKind::Theorem]);
    }

    #[test]
    fn tokenize_unicode_types() {
        assert_eq!(
            kinds("ℝ ℤ ℕ"),
            vec![TokenKind::TyReal, TokenKind::TyInt, TokenKind::TyNat]
        );
    }

    #[test]
    fn tokenize_unicode_operators() {
        assert_eq!(
            kinds("≤ ≥ ≠ ∧ ∨ → ↔ ¬ ∀"),
            vec![
                TokenKind::Le,
                TokenKind::Ge,
                TokenKind::NotEq,
                TokenKind::And,
                TokenKind::Or,
                TokenKind::Implies,
                TokenKind::Iff,
                TokenKind::Not,
                TokenKind::Forall,
            ]
        );
    }

    #[test]
    fn tokenize_identifier_with_subscript_digit() {
        // The hypothesis-name pattern from every miniF2F file.
        assert_eq!(kinds("h₀"), vec![TokenKind::Ident("h₀".to_string())]);
        assert_eq!(
            kinds("h₁ x₂"),
            vec![
                TokenKind::Ident("h₁".to_string()),
                TokenKind::Ident("x₂".to_string()),
            ]
        );
    }

    #[test]
    fn tokenize_assign_vs_colon() {
        assert_eq!(kinds(":="), vec![TokenKind::Assign]);
        assert_eq!(kinds(":"), vec![TokenKind::Colon]);
        assert_eq!(kinds(": ="), vec![TokenKind::Colon, TokenKind::Eq]);
    }

    #[test]
    fn tokenize_arithmetic_and_parens() {
        assert_eq!(
            kinds("(3 * x + 2)"),
            vec![
                TokenKind::LParen,
                TokenKind::IntLit(3),
                TokenKind::Star,
                TokenKind::Ident("x".to_string()),
                TokenKind::Plus,
                TokenKind::IntLit(2),
                TokenKind::RParen,
            ]
        );
    }

    #[test]
    fn tokenize_line_comment_skipped() {
        assert_eq!(
            kinds("x -- this is ignored\ny"),
            vec![
                TokenKind::Ident("x".to_string()),
                TokenKind::Ident("y".to_string()),
            ]
        );
    }

    #[test]
    fn tokenize_block_comment_skipped() {
        assert_eq!(
            kinds("x /- block -/ y"),
            vec![
                TokenKind::Ident("x".to_string()),
                TokenKind::Ident("y".to_string()),
            ]
        );
    }

    #[test]
    fn tokenize_mathd_algebra_109_signature() {
        // Spot-check the shape that the 100%-accepted linear_real
        // category takes: typed binders + equation hypotheses + goal.
        let src = "theorem mathd_algebra_109 (a b : ℝ) (h₀ : 3 * a + 2 * b = 12) (h₁ : a = 4) : b = 0 := by sorry";
        let toks = kinds(src);
        assert!(toks.contains(&TokenKind::Theorem));
        assert!(toks.contains(&TokenKind::TyReal));
        assert!(toks.contains(&TokenKind::Assign));
        assert!(toks.contains(&TokenKind::Sorry));
        assert!(toks.contains(&TokenKind::By));
    }

    #[test]
    fn tokenize_unknown_char_errors() {
        // `%` is not in the in-scope operator set; the filter should
        // reject such files before they reach us, but we want a clean
        // error if one slips through.
        let e = tokenize("3 % 2").unwrap_err();
        match e {
            IngestError::UnknownChar { ch, .. } => assert_eq!(ch, '%'),
            other => panic!("expected UnknownChar, got {other:?}"),
        }
    }

    #[test]
    fn error_categories() {
        assert_eq!(
            IngestError::UnknownChar { ch: '%', offset: 0 }.category(),
            "parse"
        );
        assert_eq!(
            IngestError::UnsupportedTranslation { reason: "x".into() }.category(),
            "translate"
        );
    }
}
