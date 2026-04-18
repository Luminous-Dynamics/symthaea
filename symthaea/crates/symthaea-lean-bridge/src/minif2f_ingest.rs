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

use symthaea_core::hdc::fol_formula_ext::{FolFormulaExt, NumericType, Term};

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
    /// `a ∣ b` — divisibility. Not a first-order FolFormulaExt
    /// construct; the translator lowers it to `∃ c : ℤ, b = a * c`
    /// at translate time with a fresh synthetic witness name.
    Divides,
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

    /// `⁻¹` — postfix reciprocal operator. Lexed as the two-codepoint
    /// pair U+207B + U+00B9 (superscript minus + superscript one).
    /// Lean semantics: `x⁻¹` = `HPow.hPow x (-1 : ℤ)` = `1 / x` for
    /// fields; the parser lowers it to the latter at parse time.
    Reciprocal,

    /// `∣` (U+2223 DIVIDES) — divisibility relation. Parsed as a
    /// `RelOp` at the statement level; lowered to an existential
    /// `∃ c : ℤ, b = a * c` by the translator.
    Divides,

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
            Reciprocal => write!(f, "⁻¹"),
            Divides => write!(f, "∣"),
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
            // Two-codepoint postfix: `⁻¹` = U+207B U+00B9. Handle
            // before the single-char match so a U+207B with anything
            // other than U+00B9 after it still errors cleanly.
            if ch == '\u{207B}' {
                if let Some((next_ch, next_len)) = next_char(&src[i + ch_len..]) {
                    if next_ch == '\u{00B9}' {
                        out.push(Token {
                            kind: TokenKind::Reciprocal,
                            offset: start,
                        });
                        i += ch_len + next_len;
                        continue;
                    }
                }
                return Err(IngestError::UnknownChar { ch, offset: start });
            }
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
                '∣' => TokenKind::Divides,
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
// Parser
// ════════════════════════════════════════════════════════════════════════

/// Top-level entry point. Takes a full miniF2F `.lean` source, skips the
/// `import Mathlib` preamble, locates the `theorem` declaration, and
/// parses it into a `LeanTheorem`. Non-theorem lines (imports, `open`,
/// `set_option`) are ignored; the parser is not a full Lean front-end.
pub fn parse_theorem(src: &str) -> Result<LeanTheorem, IngestError> {
    let toks = tokenize(src)?;
    // Find the `theorem` keyword, skipping any preceding tokens. This
    // lets callers pass a full `.lean` file without pre-stripping the
    // `import Mathlib\nset_option …\nopen …` preamble.
    let start = toks
        .iter()
        .position(|t| matches!(t.kind, TokenKind::Theorem))
        .ok_or(IngestError::NotATheorem)?;
    let mut p = Parser {
        toks: &toks,
        pos: start,
    };
    p.parse_theorem()
}

/// Stateful cursor over the token stream. The grammar is small enough
/// that a plain recursive-descent / Pratt combo fits in ~300 LOC.
struct Parser<'a> {
    toks: &'a [Token],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> &TokenKind {
        &self.toks[self.pos].kind
    }
    fn peek_at(&self, k: usize) -> Option<&TokenKind> {
        self.toks.get(self.pos + k).map(|t| &t.kind)
    }
    fn offset(&self) -> usize {
        self.toks[self.pos].offset
    }
    fn advance(&mut self) -> &Token {
        let t = &self.toks[self.pos];
        if !matches!(t.kind, TokenKind::Eof) {
            self.pos += 1;
        }
        t
    }
    /// Consume the next token iff it matches `kind`; return whether we
    /// consumed. Useful for optional grammar positions.
    fn eat(&mut self, kind: &TokenKind) -> bool {
        if std::mem::discriminant(self.peek()) == std::mem::discriminant(kind) {
            self.advance();
            return true;
        }
        false
    }
    /// Consume the next token and assert it matches `kind`; otherwise
    /// produce a positioned `IngestError::Unexpected`.
    fn expect(&mut self, kind: &TokenKind, expected: &'static str) -> Result<(), IngestError> {
        if std::mem::discriminant(self.peek()) == std::mem::discriminant(kind) {
            self.advance();
            Ok(())
        } else if matches!(self.peek(), TokenKind::Eof) {
            Err(IngestError::UnexpectedEof { expected })
        } else {
            Err(IngestError::Unexpected {
                expected,
                found: format!("{}", self.peek()),
                offset: self.offset(),
            })
        }
    }

    fn parse_theorem(&mut self) -> Result<LeanTheorem, IngestError> {
        self.expect(&TokenKind::Theorem, "`theorem`")?;
        // Theorem name.
        let name = match self.advance().kind.clone() {
            TokenKind::Ident(s) => s,
            other => {
                return Err(IngestError::Unexpected {
                    expected: "theorem name",
                    found: format!("{other}"),
                    offset: self.offset(),
                })
            }
        };
        // Binders: zero or more `(IDENT+ : BODY)` groups.
        let mut binders = Vec::new();
        while matches!(self.peek(), TokenKind::LParen) {
            let group = self.parse_binder_group()?;
            binders.extend(group);
        }
        // `: GOAL`
        self.expect(&TokenKind::Colon, "`:` before goal")?;
        let goal = self.parse_statement()?;
        // `:= by sorry` — the Phase 4 ingestion path only cares about
        // the statement; `sorry` is what we're replacing.
        self.expect(&TokenKind::Assign, "`:=`")?;
        self.expect(&TokenKind::By, "`by`")?;
        self.expect(&TokenKind::Sorry, "`sorry`")?;
        Ok(LeanTheorem {
            name,
            binders,
            goal,
        })
    }

    /// Parse `(name₁ name₂ … : BODY)`. One group can declare multiple
    /// bound names sharing the same body; we splat them into separate
    /// `LeanBinder` values in the enclosing list.
    fn parse_binder_group(&mut self) -> Result<Vec<LeanBinder>, IngestError> {
        self.expect(&TokenKind::LParen, "`(` to start binder")?;
        let mut names = Vec::new();
        loop {
            match self.peek().clone() {
                TokenKind::Ident(s) => {
                    self.advance();
                    names.push(s);
                }
                _ => break,
            }
        }
        if names.is_empty() {
            return Err(IngestError::Unexpected {
                expected: "binder identifier",
                found: format!("{}", self.peek()),
                offset: self.offset(),
            });
        }
        self.expect(&TokenKind::Colon, "`:` inside binder")?;
        // Decide TypedVar vs Hyp by peeking: a bare type token that's
        // immediately followed by `)` is a `LeanType` annotation; any
        // other shape is a statement body.
        let is_type_annotation = matches!(
            self.peek(),
            TokenKind::TyReal | TokenKind::TyInt | TokenKind::TyNat
        ) && matches!(self.peek_at(1), Some(TokenKind::RParen));
        let out = if is_type_annotation {
            let ty = match self.advance().kind {
                TokenKind::TyReal => LeanType::Real,
                TokenKind::TyInt => LeanType::Int,
                TokenKind::TyNat => LeanType::Nat,
                _ => unreachable!(),
            };
            names
                .into_iter()
                .map(|n| LeanBinder::TypedVar { name: n, ty })
                .collect()
        } else {
            let stmt = self.parse_statement()?;
            names
                .into_iter()
                .map(|n| LeanBinder::Hyp {
                    name: n,
                    stmt: stmt.clone(),
                })
                .collect()
        };
        self.expect(&TokenKind::RParen, "`)` closing binder")?;
        Ok(out)
    }

    // Statement precedence ladder (loosest first).
    //
    //   Statement → Implies
    //   Implies   → Iff ('→' Implies)?            right-assoc
    //   Iff       → Or ('↔' Or)?                  non-assoc; rare
    //   Or        → And ('∨' And)*                left-assoc
    //   And       → NotExpr ('∧' NotExpr)*        left-assoc
    //   NotExpr   → '¬' NotExpr | Atom
    //   Atom      → '(' Statement ')' | Relation
    //   Relation  → Term RelOp Term                non-assoc
    fn parse_statement(&mut self) -> Result<LeanStatement, IngestError> {
        self.parse_implies()
    }

    fn parse_implies(&mut self) -> Result<LeanStatement, IngestError> {
        let lhs = self.parse_iff()?;
        if self.eat(&TokenKind::Implies) {
            let rhs = self.parse_implies()?; // right-assoc
            return Ok(LeanStatement::Implies(Box::new(lhs), Box::new(rhs)));
        }
        Ok(lhs)
    }

    fn parse_iff(&mut self) -> Result<LeanStatement, IngestError> {
        let lhs = self.parse_or()?;
        if self.eat(&TokenKind::Iff) {
            let rhs = self.parse_or()?;
            return Ok(LeanStatement::Iff(Box::new(lhs), Box::new(rhs)));
        }
        Ok(lhs)
    }

    fn parse_or(&mut self) -> Result<LeanStatement, IngestError> {
        let mut lhs = self.parse_and()?;
        while self.eat(&TokenKind::Or) {
            let rhs = self.parse_and()?;
            lhs = LeanStatement::Or(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<LeanStatement, IngestError> {
        let mut lhs = self.parse_not()?;
        while self.eat(&TokenKind::And) {
            let rhs = self.parse_not()?;
            lhs = LeanStatement::And(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_not(&mut self) -> Result<LeanStatement, IngestError> {
        if self.eat(&TokenKind::Not) {
            let inner = self.parse_not()?;
            return Ok(LeanStatement::Not(Box::new(inner)));
        }
        self.parse_atom_stmt()
    }

    /// Either a parenthesized statement or a relation between terms.
    /// We need one-token lookahead to decide: `(x + 1 = 2)` vs
    /// `(x + 1) = 2`. We try-parse parens, then if the result isn't
    /// followed by a relation, we treat it as a parenthesized
    /// statement; otherwise we commit the inner content as a term's
    /// LHS. To avoid backtracking, we parse a `Term` first, and if a
    /// relation operator follows we finish the relation; if instead we
    /// see a propositional connective, this was a nested statement
    /// (and we reject — the grammar above doesn't generate this).
    fn parse_atom_stmt(&mut self) -> Result<LeanStatement, IngestError> {
        // Parenthesized statement: `(P)`. Distinguished from `(a + b)`
        // by peeking for a statement-level token mid-body. The cheap
        // rule: if the first inner token is `¬` or if a connective
        // appears at depth 0, treat it as a nested statement.
        if matches!(self.peek(), TokenKind::LParen) && self.looks_like_paren_stmt() {
            self.advance(); // consume `(`
            let inner = self.parse_statement()?;
            self.expect(&TokenKind::RParen, "`)` closing parenthesized statement")?;
            return Ok(inner);
        }
        self.parse_relation()
    }

    /// Peek ahead to decide whether the parens contain a statement
    /// (has a connective at depth 0) or a term (doesn't). Cheap scan;
    /// stops at a matching `)`.
    fn looks_like_paren_stmt(&self) -> bool {
        let mut depth = 0i32;
        for i in self.pos.. {
            let Some(tk) = self.toks.get(i).map(|t| &t.kind) else {
                return false;
            };
            match tk {
                TokenKind::LParen => depth += 1,
                TokenKind::RParen => {
                    depth -= 1;
                    if depth == 0 {
                        return false;
                    }
                }
                TokenKind::And
                | TokenKind::Or
                | TokenKind::Implies
                | TokenKind::Iff
                | TokenKind::Not
                    if depth == 1 =>
                {
                    return true;
                }
                TokenKind::Eof => return false,
                _ => {}
            }
        }
        false
    }

    fn parse_relation(&mut self) -> Result<LeanStatement, IngestError> {
        let lhs = self.parse_term()?;
        let op = match self.peek() {
            TokenKind::Eq => RelOp::Eq,
            TokenKind::NotEq => RelOp::NotEq,
            TokenKind::Lt => RelOp::Lt,
            TokenKind::Le => RelOp::Le,
            TokenKind::Gt => RelOp::Gt,
            TokenKind::Ge => RelOp::Ge,
            TokenKind::Divides => RelOp::Divides,
            other => {
                return Err(IngestError::Unexpected {
                    expected: "relation operator (= ≠ < ≤ > ≥ ∣)",
                    found: format!("{other}"),
                    offset: self.offset(),
                })
            }
        };
        self.advance();
        let rhs = self.parse_term()?;
        Ok(LeanStatement::Rel(op, lhs, rhs))
    }

    // Term precedence: + − (lowest), * /, ^, unary −, atom (highest).
    // Implicit multiplication lives at the Mul level: `3x^2` = `3 *
    // (x^2)`, `3(a+b)` = `3 * (a+b)`.
    fn parse_term(&mut self) -> Result<LeanTerm, IngestError> {
        self.parse_addsub()
    }

    fn parse_addsub(&mut self) -> Result<LeanTerm, IngestError> {
        let mut lhs = self.parse_muldiv()?;
        loop {
            let op = match self.peek() {
                TokenKind::Plus => BinOp::Add,
                TokenKind::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_muldiv()?;
            lhs = LeanTerm::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_muldiv(&mut self) -> Result<LeanTerm, IngestError> {
        let mut lhs = self.parse_pow()?;
        loop {
            // Explicit * or /.
            let op = match self.peek() {
                TokenKind::Star => Some(BinOp::Mul),
                TokenKind::Slash => Some(BinOp::Div),
                _ => None,
            };
            if let Some(op) = op {
                self.advance();
                let rhs = self.parse_pow()?;
                lhs = LeanTerm::Bin(op, Box::new(lhs), Box::new(rhs));
                continue;
            }
            // Implicit multiplication. Valid only when the previous
            // atom is a numeric literal and the next starts a primary
            // that could be factor (Ident or LParen). `3x` and `3(x+1)`
            // are accepted; we don't recognize `(x)(y)` or `xy` to
            // avoid ambiguity with function application shapes we
            // don't support.
            let implicit_after_literal = matches!(lhs, LeanTerm::IntLit(_))
                && matches!(self.peek(), TokenKind::Ident(_) | TokenKind::LParen);
            if implicit_after_literal {
                let rhs = self.parse_pow()?;
                lhs = LeanTerm::Bin(BinOp::Mul, Box::new(lhs), Box::new(rhs));
                continue;
            }
            break;
        }
        Ok(lhs)
    }

    fn parse_pow(&mut self) -> Result<LeanTerm, IngestError> {
        let base = self.parse_unary()?;
        if self.eat(&TokenKind::Caret) {
            let exp = self.parse_pow()?; // right-assoc
            return Ok(LeanTerm::Bin(BinOp::Pow, Box::new(base), Box::new(exp)));
        }
        Ok(base)
    }

    fn parse_unary(&mut self) -> Result<LeanTerm, IngestError> {
        if self.eat(&TokenKind::Minus) {
            let inner = self.parse_unary()?;
            // Eager fold of `-IntLit(n)` → `IntLit(-n)` to simplify
            // downstream matching (the hand-translator writes literal
            // negative integers directly).
            if let LeanTerm::IntLit(n) = inner {
                return Ok(LeanTerm::IntLit(-n));
            }
            return Ok(LeanTerm::Neg(Box::new(inner)));
        }
        self.parse_postfix()
    }

    /// Primary then any number of trailing `⁻¹` postfix operators.
    /// Lean precedence: `⁻¹` binds tighter than `^`, `*`, `/`, unary
    /// `-`, and `+`. `-x⁻¹` = `-(x⁻¹)`; `x⁻¹^2` = `(x⁻¹)^2`. We
    /// lower `x⁻¹` to `1 / x` at parse time so the translator never
    /// needs to know about the reciprocal operator.
    fn parse_postfix(&mut self) -> Result<LeanTerm, IngestError> {
        let mut base = self.parse_primary()?;
        while self.eat(&TokenKind::Reciprocal) {
            base = LeanTerm::Bin(BinOp::Div, Box::new(LeanTerm::IntLit(1)), Box::new(base));
        }
        Ok(base)
    }

    fn parse_primary(&mut self) -> Result<LeanTerm, IngestError> {
        match self.peek().clone() {
            TokenKind::IntLit(n) => {
                self.advance();
                Ok(LeanTerm::IntLit(n))
            }
            TokenKind::Ident(s) => {
                self.advance();
                Ok(LeanTerm::Var(s))
            }
            TokenKind::LParen => {
                self.advance();
                let t = self.parse_term()?;
                self.expect(&TokenKind::RParen, "`)` closing parenthesized term")?;
                Ok(t)
            }
            TokenKind::Eof => Err(IngestError::UnexpectedEof { expected: "term" }),
            other => Err(IngestError::Unexpected {
                expected: "term (identifier, integer, or `(`)",
                found: format!("{other}"),
                offset: self.offset(),
            }),
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Translator — LeanTheorem → FolFormulaExt
// ════════════════════════════════════════════════════════════════════════

/// Translate a parsed `LeanTheorem` into the `FolFormulaExt` the Phase 2
/// W4 cascade consumes. The shape mirrors the hand-translator idiom
/// in `prove_minif2f_curated.rs`:
///
///   `∀ x₁ : T₁, … ∀ xₙ : Tₙ, (hyp₁ → hyp₂ → … → goal)`
///
/// Typed binders become `Forall`s wrapping the body (outermost first,
/// matching the hand-translator's `forall_all(&binders, body)` loop).
/// Hypothesis binders become a right-associated implication chain
/// whose final consequent is the translated goal.
pub fn translate_theorem(t: &LeanTheorem) -> Result<FolFormulaExt, IngestError> {
    // Translator context threaded through recursive helpers. Currently
    // just tracks a counter for synthetic `_div_witness_N` names used
    // to lower divisibility to existentials.
    let mut ctx = TranslationCtx::default();
    // Body: hyp₁ → hyp₂ → … → goal (right-assoc).
    let mut body = translate_statement(&t.goal, &mut ctx)?;
    for b in t.binders.iter().rev() {
        if let LeanBinder::Hyp { stmt, .. } = b {
            let hyp = translate_statement(stmt, &mut ctx)?;
            body = hyp.implies(body);
        }
    }
    // Wrap with Forall for each typed binder (outer quantifier first
    // matches `forall_all`'s iter.rev() loop in the hand-translator).
    for b in t.binders.iter().rev() {
        if let LeanBinder::TypedVar { name, ty } = b {
            body = FolFormulaExt::forall(name, translate_type(*ty), body);
        }
    }
    Ok(body)
}

/// Stateful counters threaded through the recursive translator.
/// Currently just the divisibility-witness counter; expected to grow
/// as other non-first-order constructs land.
#[derive(Default)]
struct TranslationCtx {
    /// Next numeric suffix for synthetic `_div_witness_N` names emitted
    /// when lowering `a ∣ b` to `∃ c, b = a * c`. Incremented per
    /// occurrence so nested divisibility doesn't capture.
    div_witness_counter: usize,
}

fn translate_type(t: LeanType) -> NumericType {
    match t {
        LeanType::Real => NumericType::Real,
        LeanType::Int => NumericType::Int,
        LeanType::Nat => NumericType::Nat,
    }
}

fn translate_statement(
    s: &LeanStatement,
    ctx: &mut TranslationCtx,
) -> Result<FolFormulaExt, IngestError> {
    match s {
        LeanStatement::Rel(op, a, b) => {
            let a = translate_term(a)?;
            let b = translate_term(b)?;
            Ok(match op {
                RelOp::Eq => FolFormulaExt::eq(a, b),
                // `a ≠ b` → `¬ (a = b)`; FolFormulaExt has no NotEq
                // constructor.
                RelOp::NotEq => FolFormulaExt::eq(a, b).neg(),
                RelOp::Lt => FolFormulaExt::lt(a, b),
                RelOp::Le => FolFormulaExt::le(a, b),
                // `a > b` ≡ `b < a`; flip to match the available constructor.
                RelOp::Gt => FolFormulaExt::lt(b, a),
                // `a ≥ b` ≡ `b ≤ a`; flip.
                RelOp::Ge => FolFormulaExt::le(b, a),
                // `a ∣ b` ≡ `∃ c : ℤ, b = a * c`. Synthetic witness
                // name `_div_witness_N` — underscore prefix keeps it
                // out of Lean's default display; N-suffix counter
                // prevents capture when the same formula contains
                // multiple `∣`. ℤ is the broadest common type that
                // works for both ℕ- and ℤ-valued operands; Lean's
                // elaborator handles any sub-type coercion.
                RelOp::Divides => {
                    ctx.div_witness_counter += 1;
                    let fresh = format!("_div_witness_{}", ctx.div_witness_counter);
                    let body = FolFormulaExt::eq(b, a.mul(Term::var(&fresh)));
                    FolFormulaExt::exists(&fresh, NumericType::Int, body)
                }
            })
        }
        LeanStatement::And(p, q) => {
            Ok(translate_statement(p, ctx)?.and(translate_statement(q, ctx)?))
        }
        LeanStatement::Or(p, q) => {
            Ok(translate_statement(p, ctx)?.or(translate_statement(q, ctx)?))
        }
        LeanStatement::Implies(p, q) => {
            Ok(translate_statement(p, ctx)?.implies(translate_statement(q, ctx)?))
        }
        LeanStatement::Iff(p, q) => {
            // `p ↔ q` ≡ `(p → q) ∧ (q → p)`; no direct constructor.
            let p = translate_statement(p, ctx)?;
            let q = translate_statement(q, ctx)?;
            Ok(p.clone().implies(q.clone()).and(q.implies(p)))
        }
        LeanStatement::Not(p) => Ok(translate_statement(p, ctx)?.neg()),
    }
}

fn translate_term(t: &LeanTerm) -> Result<Term, IngestError> {
    match t {
        LeanTerm::Var(n) => Ok(Term::var(n)),
        LeanTerm::IntLit(n) => Ok(Term::int(*n)),
        LeanTerm::Neg(inner) => Ok(translate_term(inner)?.neg()),
        LeanTerm::Bin(op, a, b) => {
            match op {
                BinOp::Add => Ok(translate_term(a)?.add(translate_term(b)?)),
                BinOp::Sub => Ok(translate_term(a)?.sub(translate_term(b)?)),
                BinOp::Mul => Ok(translate_term(a)?.mul(translate_term(b)?)),
                BinOp::Div => {
                    // Common miniF2F shape: `p / q` with both sides
                    // integer literals → exact rational. This matches
                    // the hand-translator's `rat(p, q)` and keeps the
                    // SMT fragment detector (LRA vs NRA) honest.
                    if let (LeanTerm::IntLit(p), LeanTerm::IntLit(q)) = (a.as_ref(), b.as_ref()) {
                        if *q == 0 {
                            return Err(IngestError::UnsupportedTranslation {
                                reason: "division by literal 0".into(),
                            });
                        }
                        return Ok(Term::rat(*p, *q));
                    }
                    Ok(translate_term(a)?.div(translate_term(b)?))
                }
                BinOp::Pow => {
                    // `FolFormulaExt::Term::Pow` stores the exponent as
                    // `u32`, so we only accept non-negative integer
                    // literal exponents. Variable exponents (`x^y`) and
                    // negative / fractional exponents (`x^(-1)`) fall
                    // outside the supported fragment and produce a
                    // named failure for the scorecard.
                    let base = translate_term(a)?;
                    match b.as_ref() {
                        LeanTerm::IntLit(n) if *n >= 0 => {
                            let exp: u32 = (*n).try_into().map_err(|_| {
                                IngestError::UnsupportedTranslation {
                                    reason: format!("exponent {n} exceeds u32::MAX"),
                                }
                            })?;
                            Ok(base.pow(exp))
                        }
                        LeanTerm::IntLit(n) => Err(IngestError::UnsupportedTranslation {
                            reason: format!("negative exponent {n}"),
                        }),
                        _ => Err(IngestError::UnsupportedTranslation {
                            reason: "non-literal exponent".into(),
                        }),
                    }
                }
            }
        }
    }
}

/// One-shot helper: parse a `.lean` source string and translate. Returns
/// a `(parsed_ok, translated_ok, formula)` tuple where `parsed_ok` and
/// `translated_ok` feed the three-tier scorecard; the third stage
/// (Lake accept) is handled downstream by `render_fol_ext_file`.
///
/// Returns `Err` with the categorized `IngestError` when either stage
/// fails. Callers that need the "Parsed but not Translated" middle
/// state can call `parse_theorem` and `translate_theorem` separately
/// and dispatch on the individual `Result`s.
pub fn ingest(src: &str) -> Result<FolFormulaExt, IngestError> {
    let theorem = parse_theorem(src)?;
    translate_theorem(&theorem)
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
    fn tokenize_reciprocal_is_a_single_token() {
        // `⁻¹` is U+207B + U+00B9; must coalesce into one `Reciprocal`.
        assert_eq!(
            kinds("x⁻¹"),
            vec![TokenKind::Ident("x".into()), TokenKind::Reciprocal,]
        );
    }

    #[test]
    fn tokenize_bare_superscript_minus_still_errors() {
        // U+207B alone (without the following U+00B9) is not a valid
        // token in our surface. Keeps the error crisp rather than
        // swallowing partial postfix shapes.
        let e = tokenize("\u{207B}").unwrap_err();
        match e {
            IngestError::UnknownChar { ch, .. } => assert_eq!(ch, '\u{207B}'),
            other => panic!("expected UnknownChar, got {other:?}"),
        }
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

    // ─── parser ─────────────────────────────────────────────────────

    fn v(n: &str) -> LeanTerm {
        LeanTerm::Var(n.into())
    }
    fn i(n: i64) -> LeanTerm {
        LeanTerm::IntLit(n)
    }
    fn add(a: LeanTerm, b: LeanTerm) -> LeanTerm {
        LeanTerm::Bin(BinOp::Add, Box::new(a), Box::new(b))
    }
    fn mul(a: LeanTerm, b: LeanTerm) -> LeanTerm {
        LeanTerm::Bin(BinOp::Mul, Box::new(a), Box::new(b))
    }
    fn eq_stmt(a: LeanTerm, b: LeanTerm) -> LeanStatement {
        LeanStatement::Rel(RelOp::Eq, a, b)
    }

    #[test]
    fn parse_bare_theorem_no_binders() {
        let src = "theorem trivial_eq : 1 = 1 := by sorry";
        let t = parse_theorem(src).unwrap();
        assert_eq!(t.name, "trivial_eq");
        assert!(t.binders.is_empty());
        assert_eq!(t.goal, eq_stmt(i(1), i(1)));
    }

    #[test]
    fn parse_mathd_algebra_109_shape() {
        // The canonical linear_real signature.
        let src = "theorem mathd_algebra_109 (a b : ℝ) (h₀ : 3 * a + 2 * b = 12) \
                   (h₁ : a = 4) : b = 0 := by sorry";
        let t = parse_theorem(src).unwrap();
        assert_eq!(t.name, "mathd_algebra_109");
        assert_eq!(t.binders.len(), 4);
        assert!(matches!(
            t.binders[0],
            LeanBinder::TypedVar { ref name, ty: LeanType::Real } if name == "a"
        ));
        assert!(matches!(
            t.binders[1],
            LeanBinder::TypedVar { ref name, ty: LeanType::Real } if name == "b"
        ));
        // h₀: 3 * a + 2 * b = 12
        match &t.binders[2] {
            LeanBinder::Hyp { name, stmt } => {
                assert_eq!(name, "h₀");
                assert_eq!(
                    *stmt,
                    eq_stmt(add(mul(i(3), v("a")), mul(i(2), v("b"))), i(12))
                );
            }
            _ => panic!("expected Hyp, got {:?}", t.binders[2]),
        }
        assert_eq!(t.goal, eq_stmt(v("b"), i(0)));
    }

    #[test]
    fn parse_implicit_multiplication_on_literal() {
        // `3a` without explicit `*` — common in miniF2F.
        let src = "theorem t (a : ℝ) : 3a = 3 * a := by sorry";
        let t = parse_theorem(src).unwrap();
        assert_eq!(t.goal, eq_stmt(mul(i(3), v("a")), mul(i(3), v("a"))));
    }

    #[test]
    fn parse_conjunction_goal() {
        let src = "theorem t (x y : ℝ) : x = 1 ∧ y = 2 := by sorry";
        let t = parse_theorem(src).unwrap();
        assert_eq!(
            t.goal,
            LeanStatement::And(
                Box::new(eq_stmt(v("x"), i(1))),
                Box::new(eq_stmt(v("y"), i(2)))
            )
        );
    }

    #[test]
    fn parse_negative_literal_folds_into_intlit() {
        // `-11` should lex as Minus+IntLit(11) and fold to IntLit(-11).
        let src = "theorem t : -11 = -11 := by sorry";
        let t = parse_theorem(src).unwrap();
        assert_eq!(t.goal, eq_stmt(i(-11), i(-11)));
    }

    #[test]
    fn parse_right_assoc_implication() {
        // `a → b → c` parses as `a → (b → c)`.
        let src = "theorem t (a b c : ℝ) : a = 0 → b = 0 → c = 0 := by sorry";
        let t = parse_theorem(src).unwrap();
        // Outer Implies lhs is `a = 0`; outer rhs is Implies(b=0, c=0).
        match t.goal {
            LeanStatement::Implies(lhs, rhs) => {
                assert_eq!(*lhs, eq_stmt(v("a"), i(0)));
                match *rhs {
                    LeanStatement::Implies(b_eq, c_eq) => {
                        assert_eq!(*b_eq, eq_stmt(v("b"), i(0)));
                        assert_eq!(*c_eq, eq_stmt(v("c"), i(0)));
                    }
                    other => panic!("expected nested Implies, got {other:?}"),
                }
            }
            other => panic!("expected Implies, got {other:?}"),
        }
    }

    #[test]
    fn parse_left_assoc_and() {
        // `a ∧ b ∧ c` = `(a ∧ b) ∧ c`.
        let src = "theorem t (x : ℝ) : x = 1 ∧ x = 1 ∧ x = 1 := by sorry";
        let t = parse_theorem(src).unwrap();
        match t.goal {
            LeanStatement::And(outer_l, _outer_r) => {
                assert!(matches!(*outer_l, LeanStatement::And(_, _)));
            }
            other => panic!("expected And, got {other:?}"),
        }
    }

    #[test]
    fn parse_multi_name_binder_group() {
        // `(a b : ℝ)` declares two TypedVars sharing the same type.
        let src = "theorem t (a b : ℝ) : a = b := by sorry";
        let t = parse_theorem(src).unwrap();
        assert_eq!(t.binders.len(), 2);
        assert!(matches!(&t.binders[0], LeanBinder::TypedVar { .. }));
        assert!(matches!(&t.binders[1], LeanBinder::TypedVar { .. }));
    }

    #[test]
    fn parse_skips_full_file_preamble() {
        // Real miniF2F files have `import Mathlib\nset_option …\nopen …`
        // before the theorem. The parser skips to `theorem`.
        let src = "import Mathlib\n\
                   set_option maxHeartbeats 0\n\
                   open BigOperators Real Nat Topology Rat\n\
                   \n\
                   theorem mathd_trivial : 1 = 1 := by sorry";
        let t = parse_theorem(src).unwrap();
        assert_eq!(t.name, "mathd_trivial");
    }

    #[test]
    fn parse_reciprocal_lowers_to_div_one_by_x() {
        // `x⁻¹` should parse as `LeanTerm::Bin(Div, IntLit(1), Var(x))`.
        let src = "theorem t (x : ℝ) : x⁻¹ = 1 / x := by sorry";
        let t = parse_theorem(src).unwrap();
        let one_over_x = LeanTerm::Bin(
            BinOp::Div,
            Box::new(LeanTerm::IntLit(1)),
            Box::new(LeanTerm::Var("x".into())),
        );
        let expected_rhs = LeanTerm::Bin(
            BinOp::Div,
            Box::new(LeanTerm::IntLit(1)),
            Box::new(LeanTerm::Var("x".into())),
        );
        assert_eq!(
            t.goal,
            LeanStatement::Rel(RelOp::Eq, one_over_x, expected_rhs)
        );
    }

    #[test]
    fn parse_reciprocal_on_parenthesized_expression() {
        // `(4 / x)⁻¹` — the canonical shape from mathd_algebra_245.
        let src = "theorem t (x : ℝ) : (4 / x)⁻¹ = 0 := by sorry";
        let t = parse_theorem(src).unwrap();
        match t.goal {
            LeanStatement::Rel(RelOp::Eq, lhs, _rhs) => {
                // Expect Div(1, Div(4, x))
                match lhs {
                    LeanTerm::Bin(BinOp::Div, num, _den) => {
                        assert_eq!(*num, LeanTerm::IntLit(1));
                    }
                    other => panic!("expected Div(1, _), got {other:?}"),
                }
            }
            other => panic!("expected Rel(Eq,..), got {other:?}"),
        }
    }

    #[test]
    fn parse_reciprocal_higher_precedence_than_pow() {
        // `x⁻¹^2` parses as `(x⁻¹)^2`, i.e. Pow(Div(1,x), 2).
        let src = "theorem t (x : ℝ) : x⁻¹^2 = 0 := by sorry";
        let t = parse_theorem(src).unwrap();
        match t.goal {
            LeanStatement::Rel(RelOp::Eq, lhs, _) => match lhs {
                LeanTerm::Bin(BinOp::Pow, base, exp) => {
                    // base should be Div(1, x), exp should be IntLit(2)
                    assert!(
                        matches!(*base, LeanTerm::Bin(BinOp::Div, _, _)),
                        "expected Pow base to be Div(1,x), got {base:?}"
                    );
                    assert_eq!(*exp, LeanTerm::IntLit(2));
                }
                other => panic!("expected Pow, got {other:?}"),
            },
            other => panic!("expected Rel, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_non_theorem_source() {
        assert_eq!(
            parse_theorem("example : 1 = 1 := by rfl").unwrap_err(),
            IngestError::NotATheorem
        );
    }

    // ─── translator ─────────────────────────────────────────────────

    /// Convenience: build the same `FolFormulaExt` shape the
    /// hand-translator writes for `mathd_algebra_109`. Used by the
    /// round-trip test below.
    fn expected_algebra_109() -> FolFormulaExt {
        use symthaea_core::hdc::fol_formula_ext::NumericType as N;
        let r = N::Real;
        // body: h₀ → h₁ → goal (right-assoc)
        let h0 = FolFormulaExt::eq(
            Term::int(3)
                .mul(Term::var("a"))
                .add(Term::int(2).mul(Term::var("b"))),
            Term::int(12),
        );
        let h1 = FolFormulaExt::eq(Term::var("a"), Term::int(4));
        let goal = FolFormulaExt::eq(Term::var("b"), Term::int(0));
        let body = h0.implies(h1.implies(goal));
        // ∀ a : ℝ, ∀ b : ℝ, body (outermost = a)
        FolFormulaExt::forall("a", r, FolFormulaExt::forall("b", r, body))
    }

    #[test]
    fn translate_mathd_algebra_109_matches_hand_translator() {
        let src = "theorem mathd_algebra_109 (a b : ℝ) (h₀ : 3 * a + 2 * b = 12) \
                   (h₁ : a = 4) : b = 0 := by sorry";
        let t = parse_theorem(src).unwrap();
        let formula = translate_theorem(&t).unwrap();
        assert_eq!(formula, expected_algebra_109());
    }

    #[test]
    fn translate_rel_operators_cover_all_six() {
        let cases = [
            ("x = y", RelOp::Eq),
            ("x ≠ y", RelOp::NotEq),
            ("x < y", RelOp::Lt),
            ("x ≤ y", RelOp::Le),
            ("x > y", RelOp::Gt),
            ("x ≥ y", RelOp::Ge),
        ];
        for (expr, _op) in cases {
            let src = format!("theorem t (x y : ℝ) : {expr} := by sorry");
            let t = parse_theorem(&src).expect(expr);
            let f = translate_theorem(&t).expect(expr);
            // A shallow assertion: translator must not have returned
            // an unsupported-translation error for any of the six
            // relations.
            match f {
                FolFormulaExt::Forall(_, _, _) => { /* ok */ }
                other => panic!("expected ∀-wrapped formula for {expr}, got {other:?}"),
            }
        }
    }

    #[test]
    fn translate_rational_literal_folds_to_ratlit() {
        // `q/p = 2/3` is Phase 5 Pattern B's canonical shape; the
        // translator should fold `2 / 3` into an exact `RatLit(2, 3)`
        // rather than a BinOp(Div) — the SMT fragment detector uses
        // this distinction (QF_LRA vs QF_NRA).
        let src = "theorem t : 1 = 2 / 3 := by sorry";
        let t = parse_theorem(src).unwrap();
        let f = translate_theorem(&t).unwrap();
        match f {
            FolFormulaExt::Eq(_, Term::RatLit(p, q)) => {
                assert_eq!((p, q), (2, 3));
            }
            other => panic!("expected Eq(_, RatLit(2,3)), got {other:?}"),
        }
    }

    #[test]
    fn translate_pow_accepts_nonneg_literal_exponent() {
        let src = "theorem t (x : ℝ) : x^2 = x*x := by sorry";
        let t = parse_theorem(src).unwrap();
        translate_theorem(&t).unwrap();
    }

    #[test]
    fn translate_pow_rejects_variable_exponent() {
        let src = "theorem t (x y : ℝ) : x^y = 0 := by sorry";
        let t = parse_theorem(src).unwrap();
        let err = translate_theorem(&t).unwrap_err();
        assert_eq!(err.category(), "translate");
    }

    #[test]
    fn translate_pow_rejects_negative_literal_exponent() {
        // `-1` is pre-folded to IntLit(-1) by the parser; translator
        // sees it as a negative literal and rejects.
        let src = "theorem t (x : ℝ) : x^(-1) = 1 := by sorry";
        let t = parse_theorem(src).unwrap();
        let err = translate_theorem(&t).unwrap_err();
        match err {
            IngestError::UnsupportedTranslation { reason } => {
                assert!(reason.contains("negative exponent"), "got {reason}");
            }
            other => panic!("expected UnsupportedTranslation, got {other:?}"),
        }
    }

    #[test]
    fn translate_iff_desugars_to_bidirectional_implies_conjunction() {
        let src = "theorem t (x : ℝ) : x = 0 ↔ 2 * x = 0 := by sorry";
        let t = parse_theorem(src).unwrap();
        let f = translate_theorem(&t).unwrap();
        // Outer: ∀ x, (p → q) ∧ (q → p)
        match f {
            FolFormulaExt::Forall(_, _, body) => match *body {
                FolFormulaExt::And(l, r) => {
                    assert!(matches!(*l, FolFormulaExt::Implies(_, _)));
                    assert!(matches!(*r, FolFormulaExt::Implies(_, _)));
                }
                other => panic!("expected And for iff desugar, got {other:?}"),
            },
            other => panic!("expected Forall, got {other:?}"),
        }
    }

    #[test]
    fn ingest_one_shot_returns_formula() {
        let src = "theorem t (a b : ℝ) (h : a = b) : b = a := by sorry";
        let f = ingest(src).unwrap();
        // Result shape: Forall a, Forall b, Implies(Eq(a,b), Eq(b,a))
        match f {
            FolFormulaExt::Forall(_, _, _) => { /* ok */ }
            other => panic!("expected Forall, got {other:?}"),
        }
    }

    #[test]
    fn ingest_division_by_zero_literal_rejected() {
        let src = "theorem t : 1 = 1 / 0 := by sorry";
        let err = ingest(src).unwrap_err();
        assert_eq!(err.category(), "translate");
    }

    // ─── divisibility (∣) ────────────────────────────────────────────

    #[test]
    fn tokenize_divides_is_distinct_from_ascii_pipe() {
        // U+2223 ∣ lexes as `Divides`; ASCII `|` would currently be
        // `UnknownChar` (not in our operator set).
        assert_eq!(
            kinds("a ∣ b"),
            vec![
                TokenKind::Ident("a".into()),
                TokenKind::Divides,
                TokenKind::Ident("b".into()),
            ]
        );
    }

    #[test]
    fn parse_divides_produces_rel_divides_statement() {
        let src = "theorem t (n : ℕ) (h : 7 ∣ n) : n = 0 := by sorry";
        let t = parse_theorem(src).unwrap();
        match &t.binders[1] {
            LeanBinder::Hyp { stmt, .. } => match stmt {
                LeanStatement::Rel(RelOp::Divides, l, r) => {
                    assert_eq!(*l, LeanTerm::IntLit(7));
                    assert_eq!(*r, LeanTerm::Var("n".into()));
                }
                other => panic!("expected Rel(Divides,..), got {other:?}"),
            },
            other => panic!("expected Hyp, got {other:?}"),
        }
    }

    #[test]
    fn translate_divides_lowers_to_existential_multiplication() {
        // `7 ∣ n` → `∃ _div_witness_1 : ℤ, n = 7 * _div_witness_1`
        let src = "theorem t (n : ℕ) (h : 7 ∣ n) : n = 0 := by sorry";
        let f = ingest(src).unwrap();
        // Skip the two outer ∀ (for n) + Forall hypothesis chain to
        // find the Exists introduced by the ∣ translation.
        fn contains_div_witness_exists(f: &FolFormulaExt) -> bool {
            match f {
                FolFormulaExt::Exists(name, _ty, body) => {
                    name.starts_with("_div_witness_") || contains_div_witness_exists(body)
                }
                FolFormulaExt::Forall(_, _, body) => contains_div_witness_exists(body),
                FolFormulaExt::Implies(p, q)
                | FolFormulaExt::And(p, q)
                | FolFormulaExt::Or(p, q) => {
                    contains_div_witness_exists(p) || contains_div_witness_exists(q)
                }
                FolFormulaExt::Not(inner) => contains_div_witness_exists(inner),
                _ => false,
            }
        }
        assert!(
            contains_div_witness_exists(&f),
            "expected _div_witness existential in {f:?}"
        );
    }

    #[test]
    fn translate_multiple_divides_use_distinct_witness_names() {
        // Two ∣ in the same formula — the counter should give each a
        // fresh name so the existentials don't capture.
        let src = "theorem t (n m : ℕ) (h : 3 ∣ n ∧ 5 ∣ m) : n + m = 0 := by sorry";
        let f = ingest(src).unwrap();
        fn collect_exists_names(f: &FolFormulaExt, out: &mut Vec<String>) {
            match f {
                FolFormulaExt::Exists(name, _, body) => {
                    out.push(name.clone());
                    collect_exists_names(body, out);
                }
                FolFormulaExt::Forall(_, _, body) => collect_exists_names(body, out),
                FolFormulaExt::Implies(p, q)
                | FolFormulaExt::And(p, q)
                | FolFormulaExt::Or(p, q) => {
                    collect_exists_names(p, out);
                    collect_exists_names(q, out);
                }
                FolFormulaExt::Not(inner) => collect_exists_names(inner, out),
                _ => {}
            }
        }
        let mut names = Vec::new();
        collect_exists_names(&f, &mut names);
        let witnesses: Vec<&String> = names
            .iter()
            .filter(|n| n.starts_with("_div_witness_"))
            .collect();
        assert_eq!(
            witnesses.len(),
            2,
            "expected 2 witnesses, got {witnesses:?}"
        );
        assert_ne!(witnesses[0], witnesses[1], "witness names must be distinct");
    }
}
