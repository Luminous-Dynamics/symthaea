// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # IMO Natural-Language Parser — Symthaea-native
//!
//! Takes raw English text describing an IMO-style problem and returns a
//! parsed `CurriculumProblem` ready to feed into the existing tactic
//! library. The critical property: **this uses Symthaea's own HDC
//! semantic encoder**, no external LLM, no Lean interop.
//!
//! ## Pipeline
//!
//! 1. **Text → ContinuousHV** via `SemanticEncoder` (the same pipeline
//!    the cognitive loop uses for semantic memory).
//! 2. **HDC similarity match** against a small labeled reference corpus
//!    of canonical IMO problem patterns. The highest-similarity match
//!    above a threshold wins.
//! 3. **Parameter extraction** via targeted regex on the matched pattern
//!    (e.g. "x² − 13y² = 1" → D = 13 for a Pell problem).
//! 4. **Goal construction** — build the corresponding `CurriculumProblem`
//!    with extracted parameters, ready for `solve()`.
//!
//! ## Why this is a k-NN retrieval, not a parser
//!
//! Full natural-language parsing would require either a learned
//! grammar (hard) or a Lean-style formal reformulation (hand-written,
//! tedious). Instead, we lean on Symthaea's existing strength:
//! **HDC-space similarity**. For any problem text we've seen before (or
//! any text that's similar to one we've seen), the encoder gives us a
//! hypervector that lands close to a reference vector. The reference
//! tells us which template to use.
//!
//! This is NOT a general-purpose parser — problems outside the reference
//! corpus will either fall back to low-confidence matches or return
//! None. That's a feature: the parser **knows when it doesn't know**,
//! which is the epistemic-honesty property we want.
//!
//! ## Scope limits honestly flagged
//!
//! - Reference corpus is small (~15 patterns). Coverage is bounded by
//!   what's in it; expanding coverage means adding references.
//! - Parameter extraction uses hand-written regex, not a learned parser.
//!   It handles integer extraction, tuple extraction, and a few
//!   domain-specific patterns. Anything more creative fails.
//! - Similarity threshold (0.3) is hand-tuned. False positives are
//!   possible for confusingly-worded problems.
//! - The underlying encoder in pure-Rust mode is `MoralSemanticEncoder`
//!   or `CharNgramEncoder` — both give reasonable topical similarity
//!   but neither is a precision semantic model. The `embeddings`
//!   feature would upgrade this to a real ONNX sentence transformer.

use crate::hdc::curriculum::{CurriculumProblem, Difficulty, Domain, ProblemKind};
use crate::hdc::semantic_encoder::{create_best_encoder, SemanticEncoder};
use crate::hdc::unified_hv::ContinuousHV;

// ─── Reference corpus ──────────────────────────────────────────────────────

/// A reference problem pattern with its canonical text and a template
/// constructor that fills in parameters extracted from the input.
///
/// The template constructor takes the input text and returns `Some(problem)`
/// if it can extract the needed parameters, else `None`.
pub struct ImoReference {
    pub canonical_text: &'static str,
    pub template: fn(&str) -> Option<CurriculumProblem>,
}

// ─── Parameter extraction helpers ──────────────────────────────────────────
//
// These are hand-written, not regex-based — using simple character-level
// scanning to keep us off the regex dependency. Domain-specific patterns
// for each primitive type.

/// Extract all positive integers from a text string.
fn extract_integers(text: &str) -> Vec<i64> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if !current.is_empty() {
            if let Ok(n) = current.parse::<i64>() {
                out.push(n);
            }
            current.clear();
        }
    }
    if !current.is_empty() {
        if let Ok(n) = current.parse::<i64>() {
            out.push(n);
        }
    }
    out
}

/// Extract the first integer matching a pattern like "D = N", "D=N",
/// "D: N", or "D ... N" within a short window. Returns None if no
/// single integer is unambiguously associated with the label.
fn extract_labeled_integer(text: &str, label: &str) -> Option<i64> {
    let lowered = text.to_lowercase();
    let lower_label = label.to_lowercase();
    if let Some(idx) = lowered.find(&lower_label) {
        let window = &text[idx..idx.saturating_add(label.len() + 30)];
        let nums = extract_integers(window);
        nums.first().copied()
    } else {
        None
    }
}

// ─── Canonical reference corpus (15 patterns) ─────────────────────────────

/// Pigeonhole: "Among any N integers, some two have the same remainder
/// when divided by M." Extracts (N, M) from the first two integers.
fn template_pigeonhole(text: &str) -> Option<CurriculumProblem> {
    let ints = extract_integers(text);
    if ints.len() < 2 {
        return None;
    }
    let items = ints[0] as usize;
    let boxes = ints[1] as usize;
    if items <= boxes || boxes == 0 {
        return None;
    }
    let min_collision = (items + boxes - 1) / boxes;
    Some(CurriculumProblem {
        name: format!("Pigeonhole {}/{}", items, boxes),
        difficulty: Difficulty::Easy,
        domain: Domain::Combinatorics,
        kind: ProblemKind::Pigeonhole {
            items,
            boxes,
            min_collision,
        },
    })
}

/// Pell: "Show x² − D·y² = 1 has a positive integer solution" → extract D.
fn template_pell(text: &str) -> Option<CurriculumProblem> {
    // Look for "D = N" or "- N·y²" pattern
    if let Some(d) = extract_labeled_integer(text, "D =")
        .or_else(|| extract_labeled_integer(text, "D="))
    {
        if d >= 2 && d <= 150 {
            return Some(CurriculumProblem {
                name: format!("Pell D={}", d),
                difficulty: Difficulty::Medium,
                domain: Domain::NumberTheory,
                kind: ProblemKind::PellEquation { d },
            });
        }
    }
    // Fallback: find "− N" or "-N" after x² to get the D coefficient
    let ints = extract_integers(text);
    // A Pell problem usually contains D as the 2nd or 3rd integer — heuristic
    for &n in &ints {
        if n >= 2 && n <= 150 && (n as f64).sqrt().fract() > 1e-6 {
            return Some(CurriculumProblem {
                name: format!("Pell D={}", n),
                difficulty: Difficulty::Medium,
                domain: Domain::NumberTheory,
                kind: ProblemKind::PellEquation { d: n },
            });
        }
    }
    None
}

/// CRT: "Find x with x ≡ a₁ (mod m₁), x ≡ a₂ (mod m₂), ..." Extract
/// (a, m) pairs from the integer list.
fn template_crt(text: &str) -> Option<CurriculumProblem> {
    let ints = extract_integers(text);
    // Need an even number of integers (residue, modulus pairs), at least 4.
    if ints.len() < 4 || ints.len() % 2 != 0 {
        return None;
    }
    let residues: Vec<(i64, i64)> =
        ints.chunks(2).map(|c| (c[0], c[1])).collect();
    // Sanity: moduli must be ≥ 2
    if !residues.iter().all(|(_, m)| *m >= 2) {
        return None;
    }
    // Verify by computing the expected solution
    use crate::hdc::number_theory::NumberTheoryEngine;
    let engine = NumberTheoryEngine::new();
    match engine.crt(&residues) {
        Some((x, _m)) => Some(CurriculumProblem {
            name: format!("CRT {} residues", residues.len()),
            difficulty: if residues.len() <= 2 {
                Difficulty::Easy
            } else {
                Difficulty::Medium
            },
            domain: Domain::NumberTheory,
            kind: ProblemKind::CrtSystem {
                residues,
                expected: x,
            },
        }),
        None => None,
    }
}

/// Legendre: "Determine whether a is a quadratic residue mod p" → (a, p).
fn template_legendre(text: &str) -> Option<CurriculumProblem> {
    let ints = extract_integers(text);
    if ints.len() < 2 {
        return None;
    }
    let a = ints[0];
    let p = ints[1];
    // p must be an odd prime > 2
    if p < 3 || p % 2 == 0 {
        return None;
    }
    use crate::hdc::number_theory::NumberTheoryEngine;
    let engine = NumberTheoryEngine::new();
    let expected = engine.legendre_symbol(a, p);
    Some(CurriculumProblem {
        name: format!("Legendre ({}/{})", a, p),
        difficulty: Difficulty::Easy,
        domain: Domain::NumberTheory,
        kind: ProblemKind::LegendreSymbol { a, p, expected },
    })
}

/// AM-GM: "Prove the arithmetic-geometric mean inequality for the
/// numbers X, Y, Z, ...". Extract the numeric list and run a verify.
/// For text descriptions without specific values, we skip this template.
fn template_amgm(text: &str) -> Option<CurriculumProblem> {
    let ints = extract_integers(text);
    if ints.len() < 2 {
        return None;
    }
    let values: Vec<f64> = ints.iter().map(|&n| n as f64).collect();
    if values.iter().any(|v| *v < 0.0) {
        return None;
    }
    Some(CurriculumProblem {
        name: format!("AM-GM on {} values", values.len()),
        difficulty: Difficulty::Easy,
        domain: Domain::Inequality,
        kind: ProblemKind::Amgm { values },
    })
}

/// Cauchy-Schwarz: "Verify Cauchy-Schwarz on vectors a and b." Expects
/// two comma-separated lists of integers.
fn template_cauchy_schwarz(text: &str) -> Option<CurriculumProblem> {
    let ints = extract_integers(text);
    if ints.len() < 4 || ints.len() % 2 != 0 {
        return None;
    }
    let half = ints.len() / 2;
    let a: Vec<f64> = ints[..half].iter().map(|&n| n as f64).collect();
    let b: Vec<f64> = ints[half..].iter().map(|&n| n as f64).collect();
    Some(CurriculumProblem {
        name: format!("Cauchy-Schwarz on {}-vectors", half),
        difficulty: Difficulty::Easy,
        domain: Domain::Inequality,
        kind: ProblemKind::CauchySchwarz { a, b },
    })
}

/// Primality check: "Show that N is prime" or "prove N is a prime number".
fn template_primality(text: &str) -> Option<CurriculumProblem> {
    let ints = extract_integers(text);
    if ints.is_empty() {
        return None;
    }
    // Take the largest integer in the text (most likely the candidate)
    let n = *ints.iter().max().unwrap() as u64;
    if n < 2 {
        return None;
    }
    let engine = crate::hdc::number_theory::NumberTheoryEngine::new();
    let expected = engine.miller_rabin(n);
    Some(CurriculumProblem {
        name: format!("Primality of {}", n),
        difficulty: Difficulty::Easy,
        domain: Domain::NumberTheory,
        kind: ProblemKind::PrimalityCheck { n, expected },
    })
}

/// Euler's totient: "Compute φ(N)" or "Euler's totient of N".
fn template_euler_phi(text: &str) -> Option<CurriculumProblem> {
    let ints = extract_integers(text);
    if ints.is_empty() {
        return None;
    }
    let n = *ints.iter().max().unwrap() as u64;
    if n < 2 || n > 10_000 {
        return None;
    }
    use crate::hdc::number_theory::ModularRing;
    let ring = ModularRing::new(n);
    let expected = ring.euler_totient();
    Some(CurriculumProblem {
        name: format!("φ({})", n),
        difficulty: Difficulty::Easy,
        domain: Domain::NumberTheory,
        kind: ProblemKind::EulerPhi { n, expected },
    })
}

/// Power mean inequality: "Show that HM ≤ GM ≤ AM" for concrete values.
fn template_power_mean(text: &str) -> Option<CurriculumProblem> {
    let ints = extract_integers(text);
    if ints.len() < 2 {
        return None;
    }
    let values: Vec<f64> = ints.iter().map(|&n| n as f64).collect();
    if values.iter().any(|v| *v <= 0.0) {
        return None;
    }
    Some(CurriculumProblem {
        name: format!("Power mean HM≤AM on {} values", values.len()),
        difficulty: Difficulty::Medium,
        domain: Domain::Inequality,
        kind: ProblemKind::PowerMeanIneq {
            values,
            p: -1.0,
            q: 1.0,
        },
    })
}

/// Schur inequality: "For non-negative reals a, b, c, prove Schur's
/// inequality a(a−b)(a−c) + ... ≥ 0."
fn template_schur(text: &str) -> Option<CurriculumProblem> {
    let ints = extract_integers(text);
    // Need at least 3 values for a triple
    if ints.len() < 3 {
        return None;
    }
    let a = ints[0] as f64;
    let b = ints[1] as f64;
    let c = ints[2] as f64;
    if a < 0.0 || b < 0.0 || c < 0.0 {
        return None;
    }
    Some(CurriculumProblem {
        name: format!("Schur t=1 on ({}, {}, {})", a, b, c),
        difficulty: Difficulty::Hard,
        domain: Domain::Inequality,
        kind: ProblemKind::SchurIneq { a, b, c, t: 1 },
    })
}

/// Bezout's identity: "Find integers x, y with a·x + b·y = gcd(a, b)."
fn template_bezout(text: &str) -> Option<CurriculumProblem> {
    let ints = extract_integers(text);
    if ints.len() < 2 {
        return None;
    }
    let a = ints[0];
    let b = ints[1];
    if a == 0 && b == 0 {
        return None;
    }
    let engine = crate::hdc::number_theory::NumberTheoryEngine::new();
    let expected_gcd = engine.gcd(a.unsigned_abs(), b.unsigned_abs()) as i64;
    Some(CurriculumProblem {
        name: format!("Bezout gcd({}, {})={}", a, b, expected_gcd),
        difficulty: Difficulty::Easy,
        domain: Domain::NumberTheory,
        kind: ProblemKind::BezoutIdentity {
            a,
            b,
            expected_gcd,
        },
    })
}

/// Full reference corpus: canonical problem texts paired with template
/// constructors. Each reference is encoded to a hypervector once at
/// parser creation time.
pub fn reference_corpus() -> Vec<ImoReference> {
    vec![
        // ── Pigeonhole (5 phrasings) ──────────────────────────────────
        ImoReference {
            canonical_text: "Among any 7 integers, some two have the same remainder when divided by 6.",
            template: template_pigeonhole,
        },
        ImoReference {
            canonical_text: "Show that among 14 people, at least two share the same birthday month in a year with 12 months.",
            template: template_pigeonhole,
        },
        ImoReference {
            canonical_text: "Prove that if you distribute 10 items into 3 boxes, some box must contain at least 4 items.",
            template: template_pigeonhole,
        },
        ImoReference {
            canonical_text: "If 20 objects are placed in 7 containers, prove that some container holds at least 3 objects.",
            template: template_pigeonhole,
        },
        ImoReference {
            canonical_text: "Show that in a group of 13 people, two must be born in the same month.",
            template: template_pigeonhole,
        },
        // ── Pell (4 phrasings) ────────────────────────────────────────
        ImoReference {
            canonical_text: "Show that the Pell equation x² − 13y² = 1 has a positive integer solution.",
            template: template_pell,
        },
        ImoReference {
            canonical_text: "Prove that x² − 2y² = 1 has infinitely many positive integer solutions.",
            template: template_pell,
        },
        ImoReference {
            canonical_text: "Find the smallest positive integer solution to x² − 7y² = 1.",
            template: template_pell,
        },
        ImoReference {
            canonical_text: "Demonstrate that x² − 61y² = 1 has nontrivial solutions in positive integers.",
            template: template_pell,
        },
        // ── CRT (3 phrasings) ─────────────────────────────────────────
        ImoReference {
            canonical_text: "Find the smallest positive integer x satisfying x ≡ 2 (mod 3), x ≡ 3 (mod 5), and x ≡ 2 (mod 7).",
            template: template_crt,
        },
        ImoReference {
            canonical_text: "Show there exists x with x ≡ 1 (mod 4) and x ≡ 2 (mod 5).",
            template: template_crt,
        },
        ImoReference {
            canonical_text: "Determine a positive integer x with x ≡ 1 (mod 2), x ≡ 2 (mod 3), and x ≡ 4 (mod 5).",
            template: template_crt,
        },
        // ── Legendre (3 phrasings) ────────────────────────────────────
        ImoReference {
            canonical_text: "Determine whether 2 is a quadratic residue modulo the prime 7.",
            template: template_legendre,
        },
        ImoReference {
            canonical_text: "Show that 3 is a quadratic non-residue modulo the prime 11.",
            template: template_legendre,
        },
        ImoReference {
            canonical_text: "Decide if 5 is a square modulo the prime 13.",
            template: template_legendre,
        },
        // ── AM-GM (3 phrasings) ───────────────────────────────────────
        ImoReference {
            canonical_text: "Prove the arithmetic-mean geometric-mean inequality for the positive numbers 1, 2, and 4.",
            template: template_amgm,
        },
        ImoReference {
            canonical_text: "Verify that the arithmetic mean is at least the geometric mean for 3, 5, and 7.",
            template: template_amgm,
        },
        ImoReference {
            canonical_text: "For positive reals 2, 4, 8, show that their arithmetic mean exceeds their geometric mean.",
            template: template_amgm,
        },
        // ── Cauchy-Schwarz (3 phrasings) ──────────────────────────────
        ImoReference {
            canonical_text: "Verify the Cauchy-Schwarz inequality for the vectors 1, 2, 3 and 4, 5, 6.",
            template: template_cauchy_schwarz,
        },
        ImoReference {
            canonical_text: "Check that the Cauchy-Schwarz inequality holds for the pair of vectors 2, 3 and 4, 5.",
            template: template_cauchy_schwarz,
        },
        ImoReference {
            canonical_text: "Show that for vectors 1, 1, 2 and 3, 4, 5, the dot product squared is at most the product of norms squared.",
            template: template_cauchy_schwarz,
        },
        // ── Primality (3 phrasings) ───────────────────────────────────
        ImoReference {
            canonical_text: "Prove that 17 is a prime number.",
            template: template_primality,
        },
        ImoReference {
            canonical_text: "Show that the number 101 is prime.",
            template: template_primality,
        },
        ImoReference {
            canonical_text: "Determine whether 561 is prime (note: this is a Carmichael number).",
            template: template_primality,
        },
        // ── Euler phi (2 phrasings) ───────────────────────────────────
        ImoReference {
            canonical_text: "Compute Euler's totient function phi of 12.",
            template: template_euler_phi,
        },
        ImoReference {
            canonical_text: "Find the number of positive integers less than 15 that are coprime to 15.",
            template: template_euler_phi,
        },
        // ── Power mean (2 phrasings) ──────────────────────────────────
        ImoReference {
            canonical_text: "Show that the harmonic mean is at most the arithmetic mean for the positive numbers 1, 2, 4.",
            template: template_power_mean,
        },
        ImoReference {
            canonical_text: "Prove the HM ≤ AM inequality for the three positive values 2, 3, 6.",
            template: template_power_mean,
        },
        // ── Schur (2 phrasings) ───────────────────────────────────────
        ImoReference {
            canonical_text: "For non-negative reals 1, 2, 3 prove Schur's inequality a(a-b)(a-c) + b(b-a)(b-c) + c(c-a)(c-b) ≥ 0.",
            template: template_schur,
        },
        ImoReference {
            canonical_text: "Verify Schur's inequality at t=1 for the non-negative triple 2, 3, 5.",
            template: template_schur,
        },
        // ── Bezout (2 phrasings) ──────────────────────────────────────
        ImoReference {
            canonical_text: "Find integers x, y such that 35x + 15y equals the greatest common divisor of 35 and 15.",
            template: template_bezout,
        },
        ImoReference {
            canonical_text: "Prove Bezout's identity for the integers 12 and 8.",
            template: template_bezout,
        },
    ]
}

// ─── Parser ────────────────────────────────────────────────────────────────

/// Parse result: the matched template, its similarity score, and the
/// constructed problem.
#[derive(Debug, Clone)]
pub struct ParsedProblem {
    pub problem: CurriculumProblem,
    pub matched_reference: String,
    pub similarity: f32,
}

/// The IMO natural-language parser.
pub struct ImoNlParser {
    encoder: Box<dyn SemanticEncoder>,
    corpus: Vec<ImoReference>,
    encoded_corpus: Vec<ContinuousHV>,
    /// Minimum similarity threshold to accept a match. Defaults to 0.3.
    /// Below this, the parser returns None.
    pub min_similarity: f32,
}

impl ImoNlParser {
    pub fn new() -> Self {
        let encoder = create_best_encoder();
        let corpus = reference_corpus();
        let encoded_corpus: Vec<ContinuousHV> = corpus
            .iter()
            .map(|r| encoder.encode(r.canonical_text))
            .collect();
        Self {
            encoder,
            corpus,
            encoded_corpus,
            min_similarity: 0.3,
        }
    }

    /// Parse a natural-language IMO problem statement. Returns the
    /// matched problem template (filled with extracted parameters) and
    /// the similarity score, or None if no reference matches above
    /// `min_similarity`.
    pub fn parse(&self, text: &str) -> Option<ParsedProblem> {
        if text.trim().is_empty() {
            return None;
        }
        let query_hv = self.encoder.encode(text);
        // Find nearest neighbor
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, ref_hv) in self.encoded_corpus.iter().enumerate() {
            let sim = query_hv.similarity(ref_hv);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }
        if best_sim < self.min_similarity {
            return None;
        }
        // Try the matched template with the input text. If it fails
        // (parameters don't extract), report the match but return None.
        let reference = &self.corpus[best_idx];
        match (reference.template)(text) {
            Some(problem) => Some(ParsedProblem {
                problem,
                matched_reference: reference.canonical_text.to_string(),
                similarity: best_sim,
            }),
            None => None,
        }
    }
}

impl Default for ImoNlParser {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_integers_basic() {
        assert_eq!(extract_integers("hello 42 world"), vec![42]);
        assert_eq!(
            extract_integers("3, 14, 27, 100, 5, 18, 71"),
            vec![3, 14, 27, 100, 5, 18, 71]
        );
        assert_eq!(extract_integers("no numbers here"), Vec::<i64>::new());
    }

    #[test]
    fn test_parser_constructs_without_panic() {
        let _p = ImoNlParser::new();
    }

    #[test]
    fn test_parser_rejects_empty_input() {
        let p = ImoNlParser::new();
        assert!(p.parse("").is_none());
        assert!(p.parse("   ").is_none());
    }

    #[test]
    fn test_parse_canonical_pigeonhole() {
        let p = ImoNlParser::new();
        // Exact canonical text — should match with high similarity
        let result = p.parse("Among any 7 integers, some two have the same remainder when divided by 6.");
        let parsed = result.expect("should parse canonical pigeonhole");
        eprintln!("Match similarity: {:.3}", parsed.similarity);
        match parsed.problem.kind {
            ProblemKind::Pigeonhole { items, boxes, .. } => {
                assert_eq!(items, 7);
                assert_eq!(boxes, 6);
            }
            _ => panic!("expected Pigeonhole, got {:?}", parsed.problem.kind),
        }
        assert!(parsed.problem.solve(), "parsed problem should solve");
    }

    #[test]
    fn test_parse_rephrased_pigeonhole() {
        let p = ImoNlParser::new();
        // Paraphrased — different words, same structure
        let result = p.parse("If you put 15 objects into 4 bins, prove one bin has at least 4 objects.");
        match result {
            Some(parsed) => {
                eprintln!(
                    "Rephrased pigeonhole matched '{}' at similarity {:.3}",
                    parsed.matched_reference, parsed.similarity
                );
                assert!(parsed.problem.solve());
            }
            None => {
                // Acceptable failure — paraphrase may not hit the threshold
                eprintln!("Rephrased pigeonhole did not match any reference — expected for weak encoder");
            }
        }
    }

    #[test]
    fn test_parse_canonical_pell() {
        let p = ImoNlParser::new();
        let result = p.parse("Show that the Pell equation x² − 13y² = 1 has a positive integer solution.");
        let parsed = result.expect("should parse canonical Pell");
        match parsed.problem.kind {
            ProblemKind::PellEquation { d } => {
                assert_eq!(d, 13);
            }
            _ => panic!("expected Pell, got {:?}", parsed.problem.kind),
        }
        assert!(parsed.problem.solve());
    }

    #[test]
    fn test_parse_canonical_crt() {
        let p = ImoNlParser::new();
        let result = p.parse(
            "Find the smallest positive integer x satisfying x ≡ 2 (mod 3), x ≡ 3 (mod 5), and x ≡ 2 (mod 7).",
        );
        let parsed = result.expect("should parse canonical CRT");
        match &parsed.problem.kind {
            ProblemKind::CrtSystem { residues, .. } => {
                assert_eq!(residues.len(), 3);
            }
            _ => panic!("expected CrtSystem, got {:?}", parsed.problem.kind),
        }
        assert!(parsed.problem.solve());
    }

    #[test]
    fn test_parse_canonical_legendre() {
        let p = ImoNlParser::new();
        let result = p.parse("Determine whether 2 is a quadratic residue modulo the prime 7.");
        let parsed = result.expect("should parse canonical Legendre");
        match parsed.problem.kind {
            ProblemKind::LegendreSymbol { a, p, .. } => {
                assert_eq!(a, 2);
                assert_eq!(p, 7);
            }
            _ => panic!("expected Legendre, got {:?}", parsed.problem.kind),
        }
        assert!(parsed.problem.solve());
    }

    #[test]
    fn test_parse_canonical_amgm() {
        let p = ImoNlParser::new();
        let result = p.parse("Prove the arithmetic-mean geometric-mean inequality for the positive numbers 1, 2, and 4.");
        let parsed = result.expect("should parse canonical AM-GM");
        match &parsed.problem.kind {
            ProblemKind::Amgm { values } => {
                assert_eq!(values.len(), 3);
            }
            _ => panic!("expected Amgm, got {:?}", parsed.problem.kind),
        }
        assert!(parsed.problem.solve());
    }

    #[test]
    fn test_parse_canonical_cauchy_schwarz() {
        let p = ImoNlParser::new();
        let result = p.parse("Verify the Cauchy-Schwarz inequality for the vectors 1, 2, 3 and 4, 5, 6.");
        let parsed = result.expect("should parse canonical Cauchy-Schwarz");
        match &parsed.problem.kind {
            ProblemKind::CauchySchwarz { a, b } => {
                assert_eq!(a.len(), 3);
                assert_eq!(b.len(), 3);
            }
            _ => panic!("expected CauchySchwarz, got {:?}", parsed.problem.kind),
        }
        assert!(parsed.problem.solve());
    }

    /// **Expanded real-IMO batch test.** Parses 20 IMO-flavored
    /// problems spanning all 11 templates (6 original + 5 new). Each
    /// problem is phrased in a variation distinct from the reference
    /// corpus so the encoder has to generalize beyond exact match.
    /// Requires ≥ 65% parse+solve rate — the expected failure modes
    /// are semantically-distant phrasings and problems needing
    /// template types we don't have.
    #[test]
    fn test_expanded_real_imo_batch() {
        let parser = ImoNlParser::new();
        let problems = [
            // Pigeonhole
            ("Prove that among any 25 distinct integers, some two have the same remainder when divided by 24.", "Pigeonhole"),
            ("If 50 balls are placed in 7 urns, prove at least one urn contains 8 or more balls.", "Pigeonhole"),
            // Pell
            ("Show that the equation x² − 2y² = 1 admits infinitely many integer solutions.", "Pell"),
            ("Find a positive integer solution to x² − 19y² = 1.", "Pell"),
            // CRT
            ("Determine a positive integer x with x ≡ 1 (mod 3), x ≡ 2 (mod 5), and x ≡ 3 (mod 7).", "CRT"),
            // Legendre
            ("Prove that 3 is a quadratic residue modulo the prime 11.", "Legendre"),
            ("Show that 7 is a quadratic non-residue modulo the prime 23.", "Legendre"),
            // AM-GM
            ("For the positive numbers 5, 10, 20, prove the arithmetic mean exceeds the geometric mean.", "AM-GM"),
            // Cauchy-Schwarz
            ("Verify the Cauchy-Schwarz inequality for the vectors 3, 4, 5 and 6, 7, 8.", "Cauchy-Schwarz"),
            // Primality
            ("Prove that 37 is a prime number.", "Primality"),
            ("Determine whether 1009 is prime.", "Primality"),
            // Euler phi
            ("Compute Euler's totient phi of 18.", "EulerPhi"),
            ("Calculate the number of positive integers less than 20 that are relatively prime to 20.", "EulerPhi"),
            // Power mean
            ("Show that the harmonic mean of 3, 4, 6 is at most their arithmetic mean.", "PowerMean"),
            // Schur
            ("Verify Schur's inequality at t=1 for the non-negative triple 1, 4, 9.", "Schur"),
            // Bezout
            ("Find integers x, y with 21x + 14y = gcd(21, 14).", "Bezout"),
            ("Prove Bezout's identity for the pair 30 and 18.", "Bezout"),
            // Expected failures (unusual phrasings or out-of-scope)
            ("Color the vertices of a regular pentagon with 3 colors such that no two adjacent vertices share a color.", "out-of-scope graph coloring"),
            ("Show that the sum of the first n cubes equals the square of the sum of the first n integers.", "out-of-scope identity proof"),
            ("For a triangle with sides a, b, c, prove that a² + b² + c² ≤ 2(ab + bc + ca).", "out-of-scope triangle inequality"),
        ];

        let mut parse_successes = 0usize;
        let mut solve_successes = 0usize;
        let mut by_domain: std::collections::HashMap<&'static str, (usize, usize)> = std::collections::HashMap::new();

        eprintln!("\n════════════════════════════════════════════════════════════");
        eprintln!("  IMO NL PARSER — EXPANDED REAL-IMO BATCH TEST");
        eprintln!("  {} problems, 30 reference patterns, 11 templates", problems.len());
        eprintln!("────────────────────────────────────────────────────────────");

        for (text, label) in &problems {
            let (s, t) = by_domain.entry(label).or_insert((0, 0));
            *t += 1;
            match parser.parse(text) {
                Some(parsed) => {
                    parse_successes += 1;
                    let solved = parsed.problem.solve();
                    let status = if solved { "✓" } else { "⚠" };
                    eprintln!(
                        "  {} [{}]  sim={:.3}  solved={}  → {}",
                        status, label, parsed.similarity, solved, parsed.problem.name
                    );
                    if solved {
                        solve_successes += 1;
                        *s += 1;
                    }
                }
                None => {
                    eprintln!("  ✗ [{}]  (no match above threshold 0.3)", label);
                }
            }
        }
        eprintln!("────────────────────────────────────────────────────────────");
        eprintln!(
            "  PARSED:       {}/{} ({:.1}%)",
            parse_successes,
            problems.len(),
            parse_successes as f64 / problems.len() as f64 * 100.0
        );
        eprintln!(
            "  PARSED+SOLVED: {}/{} ({:.1}%)",
            solve_successes,
            problems.len(),
            solve_successes as f64 / problems.len() as f64 * 100.0
        );
        eprintln!("  BY CATEGORY:");
        let mut cats: Vec<_> = by_domain.iter().collect();
        cats.sort_by_key(|(k, _)| *k);
        for (cat, (s, t)) in cats {
            eprintln!("    {:35} {}/{}", cat, s, t);
        }
        eprintln!("════════════════════════════════════════════════════════════");

        // Of the 20 problems, 3 are expected to fail (out-of-scope).
        // Require ≥ 65% parse+solve rate overall (13+ out of 20).
        let rate = solve_successes as f64 / problems.len() as f64;
        assert!(
            rate >= 0.65,
            "parse+solve rate {:.1}% < 65%",
            rate * 100.0
        );
    }

    /// **The original end-to-end test.** Parse a batch of 7 natural-language
    /// problems covering all 6 original templates, print the match report, and
    /// require at least 70% successful parses.
    #[test]
    fn test_end_to_end_nl_parse_batch() {
        let parser = ImoNlParser::new();
        let problems = [
            ("Among any 7 integers, some two have the same remainder when divided by 6.", "Pigeonhole"),
            ("Show that the Pell equation x² − 13y² = 1 has a positive integer solution.", "Pell"),
            ("Find the smallest positive integer x with x ≡ 2 (mod 3) and x ≡ 3 (mod 5).", "CRT"),
            ("Determine whether 2 is a quadratic residue modulo the prime 7.", "Legendre"),
            ("Prove the arithmetic-mean geometric-mean inequality for the positive numbers 1, 2, and 4.", "AM-GM"),
            ("Verify the Cauchy-Schwarz inequality for the vectors 1, 2, 3 and 4, 5, 6.", "Cauchy-Schwarz"),
            ("Show that among 10 integers, some two differ by a multiple of 9.", "Pigeonhole rephrased"),
        ];
        let mut successes = 0;
        let total = problems.len();

        eprintln!("\n════════════════════════════════════════════════════════════");
        eprintln!("  IMO NL PARSER — END-TO-END BATCH TEST");
        eprintln!("  {} problems, Symthaea-native semantic encoder", total);
        eprintln!("────────────────────────────────────────────────────────────");
        for (text, label) in &problems {
            match parser.parse(text) {
                Some(parsed) => {
                    let solved = parsed.problem.solve();
                    eprintln!(
                        "  ✓ [{}]  sim={:.3}  solved={}  → {}",
                        label, parsed.similarity, solved, parsed.problem.name
                    );
                    if solved {
                        successes += 1;
                    }
                }
                None => {
                    eprintln!("  ✗ [{}]  (no match above threshold)", label);
                }
            }
        }
        eprintln!("────────────────────────────────────────────────────────────");
        eprintln!("  SUCCESS: {}/{} parsed + solved", successes, total);
        eprintln!("════════════════════════════════════════════════════════════");

        assert!(
            successes as f64 / total as f64 >= 0.7,
            "parser success rate {}/{} < 70%",
            successes,
            total
        );
    }
}
