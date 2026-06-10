// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Programming Understanding — Spec-Structure Learning
//!
//! Maps natural language specifications to program structures via Case-Based
//! Reasoning (Kolodner 1993) implemented in Hyperdimensional Computing.
//!
//! ## The Three Layers
//!
//! 1. **Knowledge** (Concrete): `(spec, program)` pairs — "find maximum" → Iterate+Branch+Track
//! 2. **Generalization** (Pattern): Abstracted from multiple examples — "iterate-compare-track"
//! 3. **Reasoning** (Adaptation): Rules to modify retrieved structures — "second" → AddTracker
//!
//! ## The CBR Cycle
//!
//! - **Retrieve**: Find similar specs via HDC similarity
//! - **Reuse**: Return the associated program structure
//! - **Revise**: Apply adaptation rules based on spec difference
//! - **Retain**: Store successful generations as new cases
//!
//! ## Why This Matters
//!
//! Without understanding, Symthaea matches shapes blindly. With understanding,
//! she retrieves "find maximum" when asked for "find second largest" and ADAPTS
//! it by adding a second tracking variable. This is how humans learn to program.

use std::collections::HashSet;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::program_algebra::ProgramNode;

use crate::periodic_table::encode_structural;

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Abstraction level of a spec-structure pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbstractionLevel {
    /// A specific implementation (find_max with exact code)
    Concrete,
    /// An abstracted pattern (iterate-compare-track)
    Pattern,
}

/// A spec-structure pair: what a program should do + how it's structured.
#[derive(Debug, Clone)]
pub struct SpecStructurePair {
    pub spec: String,
    pub spec_hv: BinaryHV,
    pub program: ProgramNode,
    pub structural_hv: BinaryHV,
    pub keywords: Vec<String>,
    pub level: AbstractionLevel,
}

/// An adaptation: a specific structural change to apply.
#[derive(Debug, Clone)]
pub enum Adaptation {
    /// Add a tracking variable ("find second X" → add second tracker)
    AddTracker { name: String, init: ProgramNode },
    /// Change comparison operator ("maximum" → "minimum" = GT→LT)
    ChangeOperator { from: String, to: String },
    /// Change accumulation operation ("sum" → "product" = ADD→MUL)
    ChangeAccumulator {
        from: String,
        to: String,
        new_identity: String,
    },
    /// Add early return ("find first" → add return in branch)
    AddEarlyReturn,
    /// Wrap in recursion ("iterative" → "recursive")
    WrapRecursive,
}

/// Result of understanding a spec.
#[derive(Debug, Clone)]
pub enum UnderstandingResult {
    /// Exact match found in knowledge base
    Known(ProgramNode, f32),
    /// Found similar spec, adapted structure
    Adapted(ProgramNode, f32, Vec<Adaptation>),
    /// Composed from multiple partial matches
    Composed(ProgramNode, f32),
    /// Best guess (low confidence)
    BestGuess(ProgramNode, f32),
    /// No knowledge available
    Unknown,
}

impl UnderstandingResult {
    pub fn program(&self) -> Option<&ProgramNode> {
        match self {
            Self::Known(p, _)
            | Self::Adapted(p, _, _)
            | Self::Composed(p, _)
            | Self::BestGuess(p, _) => Some(p),
            Self::Unknown => None,
        }
    }

    pub fn score(&self) -> f32 {
        match self {
            Self::Known(_, s)
            | Self::Adapted(_, s, _)
            | Self::Composed(_, s)
            | Self::BestGuess(_, s) => *s,
            Self::Unknown => 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SPEC ENCODING
// ═══════════════════════════════════════════════════════════════════════════════

/// Encode a natural language spec as a BinaryHV.
/// Uses word-level encoding: each word gets a position-permuted HV, bundled together.
fn encode_spec(spec: &str) -> BinaryHV {
    let words: Vec<&str> = spec
        .split_whitespace()
        .filter(|w| w.len() > 2) // skip small words
        .collect();

    if words.is_empty() {
        return BinaryHV::random(0xDEAD_59EC_0000_0001);
    }

    let word_hvs: Vec<BinaryHV> = words
        .iter()
        .enumerate()
        .map(|(i, word)| {
            let seed = word
                .bytes()
                .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            BinaryHV::random(seed).permute(i)
        })
        .collect();

    BinaryHV::bundle(&word_hvs)
}

/// Extract keywords from a spec string.
fn extract_keywords(spec: &str) -> Vec<String> {
    spec.split_whitespace()
        .filter(|w| w.len() > 2)
        .map(|w| w.to_lowercase())
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNDERSTANDING ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

/// The understanding engine: learns and applies spec→structure mappings.
pub struct UnderstandingEngine {
    pairs: Vec<SpecStructurePair>,
    patterns: Vec<SpecStructurePair>,
}

impl UnderstandingEngine {
    pub fn new() -> Self {
        Self {
            pairs: Vec::new(),
            patterns: Vec::new(),
        }
    }

    /// Create with basic programming knowledge pre-seeded.
    pub fn with_basic_knowledge() -> Self {
        let mut engine = Self::new();
        engine.seed_basic_knowledge();
        engine
    }

    /// Number of known spec-structure pairs.
    pub fn knowledge_count(&self) -> usize {
        self.pairs.len()
    }

    /// Number of abstracted patterns.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    // ── Learning ────────────────────────────────────────────────────────

    /// Learn a new spec-structure pair.
    pub fn learn(&mut self, spec: &str, program: &ProgramNode) {
        let spec_hv = encode_spec(spec);
        let structural_hv = encode_structural(program);
        let keywords = extract_keywords(spec);

        self.pairs.push(SpecStructurePair {
            spec: spec.to_string(),
            spec_hv,
            program: program.clone(),
            structural_hv,
            keywords,
            level: AbstractionLevel::Concrete,
        });
    }

    /// Seed with basic programming knowledge (~30 pairs).
    pub fn seed_basic_knowledge(&mut self) {
        // ── Arithmetic ──
        self.learn(
            "add two numbers",
            &ProgramNode::apply(
                ProgramNode::op("ADD"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        );
        self.learn(
            "subtract two numbers",
            &ProgramNode::apply(
                ProgramNode::op("SUB"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        );
        self.learn(
            "multiply two numbers",
            &ProgramNode::apply(
                ProgramNode::op("MUL"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        );
        self.learn(
            "divide two numbers",
            &ProgramNode::apply(
                ProgramNode::op("DIV"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        );
        self.learn(
            "check equality",
            &ProgramNode::apply(
                ProgramNode::op("EQ"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        );
        self.learn(
            "compare less than",
            &ProgramNode::apply(
                ProgramNode::op("LT"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
        );

        // ── Searching ──
        self.learn(
            "find maximum in array",
            &ProgramNode::Sequence(vec![
                ProgramNode::atom("max = arr[0]"),
                ProgramNode::iterate(
                    ProgramNode::atom("i = 1"),
                    ProgramNode::branch(
                        ProgramNode::apply(
                            ProgramNode::op("GT"),
                            vec![ProgramNode::atom("arr[i]"), ProgramNode::atom("max")],
                        ),
                        ProgramNode::atom("max = arr[i]"),
                        ProgramNode::atom("skip"),
                    ),
                    ProgramNode::apply(
                        ProgramNode::op("LT"),
                        vec![ProgramNode::atom("i"), ProgramNode::atom("len")],
                    ),
                ),
                ProgramNode::atom("max"),
            ]),
        );

        self.learn(
            "find minimum in array",
            &ProgramNode::Sequence(vec![
                ProgramNode::atom("min = arr[0]"),
                ProgramNode::iterate(
                    ProgramNode::atom("i = 1"),
                    ProgramNode::branch(
                        ProgramNode::apply(
                            ProgramNode::op("LT"),
                            vec![ProgramNode::atom("arr[i]"), ProgramNode::atom("min")],
                        ),
                        ProgramNode::atom("min = arr[i]"),
                        ProgramNode::atom("skip"),
                    ),
                    ProgramNode::apply(
                        ProgramNode::op("LT"),
                        vec![ProgramNode::atom("i"), ProgramNode::atom("len")],
                    ),
                ),
                ProgramNode::atom("min"),
            ]),
        );

        self.learn(
            "find element in array",
            &ProgramNode::Sequence(vec![
                ProgramNode::iterate(
                    ProgramNode::atom("i = 0"),
                    ProgramNode::branch(
                        ProgramNode::apply(
                            ProgramNode::op("EQ"),
                            vec![ProgramNode::atom("arr[i]"), ProgramNode::atom("target")],
                        ),
                        ProgramNode::atom("return true"),
                        ProgramNode::atom("i += 1"),
                    ),
                    ProgramNode::apply(
                        ProgramNode::op("LT"),
                        vec![ProgramNode::atom("i"), ProgramNode::atom("len")],
                    ),
                ),
                ProgramNode::atom("false"),
            ]),
        );

        // ── Accumulation ──
        self.learn(
            "sum of array",
            &ProgramNode::reduce(
                ProgramNode::op("ADD"),
                ProgramNode::typed("0", "INT"),
                ProgramNode::atom("arr"),
            ),
        );
        self.learn(
            "product of array",
            &ProgramNode::reduce(
                ProgramNode::op("MUL"),
                ProgramNode::typed("1", "INT"),
                ProgramNode::atom("arr"),
            ),
        );
        self.learn(
            "count elements in array",
            &ProgramNode::Sequence(vec![
                ProgramNode::atom("count = 0"),
                ProgramNode::iterate(
                    ProgramNode::atom("i = 0"),
                    ProgramNode::atom("count += 1"),
                    ProgramNode::apply(
                        ProgramNode::op("LT"),
                        vec![ProgramNode::atom("i"), ProgramNode::atom("len")],
                    ),
                ),
                ProgramNode::atom("count"),
            ]),
        );

        // ── Transformation ──
        self.learn(
            "transform each element",
            &ProgramNode::map(ProgramNode::atom("transform"), ProgramNode::atom("items")),
        );
        self.learn(
            "filter elements matching condition",
            &ProgramNode::filter(ProgramNode::atom("predicate"), ProgramNode::atom("items")),
        );

        // ── Recursion ──
        self.learn(
            "factorial of number",
            &ProgramNode::recurse(
                ProgramNode::branch(
                    ProgramNode::apply(
                        ProgramNode::op("EQ"),
                        vec![ProgramNode::atom("n"), ProgramNode::typed("0", "INT")],
                    ),
                    ProgramNode::typed("1", "INT"),
                    ProgramNode::apply(
                        ProgramNode::op("MUL"),
                        vec![
                            ProgramNode::atom("n"),
                            ProgramNode::apply(
                                ProgramNode::atom("factorial"),
                                vec![ProgramNode::apply(
                                    ProgramNode::op("SUB"),
                                    vec![ProgramNode::atom("n"), ProgramNode::typed("1", "INT")],
                                )],
                            ),
                        ],
                    ),
                ),
                ProgramNode::atom("factorial"),
            ),
        );

        self.learn(
            "fibonacci number",
            &ProgramNode::recurse(
                ProgramNode::branch(
                    ProgramNode::apply(
                        ProgramNode::op("LT"),
                        vec![ProgramNode::atom("n"), ProgramNode::typed("2", "INT")],
                    ),
                    ProgramNode::atom("n"),
                    ProgramNode::apply(
                        ProgramNode::op("ADD"),
                        vec![
                            ProgramNode::apply(
                                ProgramNode::atom("fib"),
                                vec![ProgramNode::apply(
                                    ProgramNode::op("SUB"),
                                    vec![ProgramNode::atom("n"), ProgramNode::typed("1", "INT")],
                                )],
                            ),
                            ProgramNode::apply(
                                ProgramNode::atom("fib"),
                                vec![ProgramNode::apply(
                                    ProgramNode::op("SUB"),
                                    vec![ProgramNode::atom("n"), ProgramNode::typed("2", "INT")],
                                )],
                            ),
                        ],
                    ),
                ),
                ProgramNode::atom("fib"),
            ),
        );

        // Auto-abstract after seeding
        self.auto_abstract();
    }

    // ── Understanding ───────────────────────────────────────────────────

    /// Given a spec, find the best matching structure.
    pub fn understand(&self, spec: &str) -> UnderstandingResult {
        if self.pairs.is_empty() && self.patterns.is_empty() {
            return UnderstandingResult::Unknown;
        }

        let spec_hv = encode_spec(spec);

        // Find nearest pairs by spec similarity
        let mut matches: Vec<(&SpecStructurePair, f32)> = self
            .pairs
            .iter()
            .chain(self.patterns.iter())
            .map(|p| (p, p.spec_hv.similarity(&spec_hv)))
            .collect();
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if matches.is_empty() {
            return UnderstandingResult::Unknown;
        }

        let best = &matches[0];

        // High similarity → known pattern
        if best.1 > 0.75 {
            return UnderstandingResult::Known(best.0.program.clone(), best.1);
        }

        // Moderate similarity → try adaptation
        if best.1 > 0.45 {
            let adaptations = self.compute_adaptations(&best.0.spec, spec);
            if !adaptations.is_empty() {
                let adapted = apply_adaptations(&best.0.program, &adaptations);
                // Score the adapted version: boost by adaptation count
                let boost = (adaptations.len() as f32 * 0.05).min(0.15);
                return UnderstandingResult::Adapted(adapted, best.1 + boost, adaptations);
            }
        }

        // Low similarity → best guess
        UnderstandingResult::BestGuess(best.0.program.clone(), best.1)
    }

    // ── Adaptation ──────────────────────────────────────────────────────

    /// Compute adaptations needed to go from base_spec to target_spec.
    fn compute_adaptations(&self, base_spec: &str, target_spec: &str) -> Vec<Adaptation> {
        let mut adaptations = Vec::new();

        let base_words: HashSet<&str> = base_spec
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| w.len() > 1)
            .collect();
        let target_words: HashSet<&str> = target_spec
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| w.len() > 1)
            .collect();
        let new_words: HashSet<&&str> = target_words.difference(&base_words).collect();

        for word in &new_words {
            match word.to_lowercase().as_str() {
                // Quantifier adaptations
                "second" | "2nd" | "penultimate" => {
                    adaptations.push(Adaptation::AddTracker {
                        name: "second_best".into(),
                        init: ProgramNode::atom("arr[1]"),
                    });
                }

                // Polarity adaptations (invert comparison)
                "minimum" | "smallest" | "least" | "lowest" => {
                    if base_spec.to_lowercase().contains("maximum")
                        || base_spec.to_lowercase().contains("largest")
                        || base_spec.to_lowercase().contains("greatest")
                    {
                        adaptations.push(Adaptation::ChangeOperator {
                            from: "GT".into(),
                            to: "LT".into(),
                        });
                    }
                }
                "maximum" | "largest" | "greatest" | "biggest" => {
                    if base_spec.to_lowercase().contains("minimum")
                        || base_spec.to_lowercase().contains("smallest")
                    {
                        adaptations.push(Adaptation::ChangeOperator {
                            from: "LT".into(),
                            to: "GT".into(),
                        });
                    }
                }

                // Accumulation adaptations
                "product" | "multiply" => {
                    if base_spec.to_lowercase().contains("sum") {
                        adaptations.push(Adaptation::ChangeAccumulator {
                            from: "ADD".into(),
                            to: "MUL".into(),
                            new_identity: "1".into(),
                        });
                    }
                }
                "sum" | "total" => {
                    if base_spec.to_lowercase().contains("product") {
                        adaptations.push(Adaptation::ChangeAccumulator {
                            from: "MUL".into(),
                            to: "ADD".into(),
                            new_identity: "0".into(),
                        });
                    }
                }

                // Control flow adaptations
                "first" | "earliest" => {
                    adaptations.push(Adaptation::AddEarlyReturn);
                }
                "recursive" | "recursion" | "recursively" => {
                    adaptations.push(Adaptation::WrapRecursive);
                }

                _ => {}
            }
        }

        adaptations
    }

    // ── Abstraction ─────────────────────────────────────────────────────

    /// Auto-discover patterns from clusters of similar specs.
    fn auto_abstract(&mut self) {
        if self.pairs.len() < 3 {
            return;
        }

        // Simple clustering: group pairs whose specs share >50% keywords
        let mut used = vec![false; self.pairs.len()];

        for i in 0..self.pairs.len() {
            if used[i] {
                continue;
            }

            let mut cluster = vec![i];
            for j in (i + 1)..self.pairs.len() {
                if used[j] {
                    continue;
                }
                let shared = keyword_overlap(&self.pairs[i].keywords, &self.pairs[j].keywords);
                if shared > 0.3 {
                    cluster.push(j);
                    used[j] = true;
                }
            }

            if cluster.len() >= 2 {
                // Extract common structure
                let programs: Vec<ProgramNode> = cluster
                    .iter()
                    .map(|&idx| self.pairs[idx].program.clone())
                    .collect();
                let abstract_node = ProgramNode::Abstract(programs);
                let abstract_hv = encode_structural(&abstract_node);

                // Common keywords
                let common: Vec<String> = common_keywords(
                    &cluster
                        .iter()
                        .map(|&i| &self.pairs[i].keywords)
                        .collect::<Vec<_>>(),
                );
                let abstract_spec = if common.is_empty() {
                    format!("pattern_{}", self.patterns.len())
                } else {
                    common.join(" ")
                };

                self.patterns.push(SpecStructurePair {
                    spec: abstract_spec.clone(),
                    spec_hv: encode_spec(&abstract_spec),
                    program: abstract_node,
                    structural_hv: abstract_hv,
                    keywords: common,
                    level: AbstractionLevel::Pattern,
                });
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADAPTATION APPLICATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Apply a list of adaptations to a ProgramNode tree.
fn apply_adaptations(base: &ProgramNode, adaptations: &[Adaptation]) -> ProgramNode {
    let mut result = base.clone();

    for adaptation in adaptations {
        result = apply_one(&result, adaptation);
    }

    result
}

fn apply_one(node: &ProgramNode, adaptation: &Adaptation) -> ProgramNode {
    match adaptation {
        Adaptation::AddTracker { name, init } => {
            // Wrap in a sequence that initializes the tracker before the main logic
            ProgramNode::Sequence(vec![
                ProgramNode::atom(&format!("{} = {}", name, expr_text(init))),
                node.clone(),
                ProgramNode::atom(name),
            ])
        }

        Adaptation::ChangeOperator { from, to } => replace_op(node, from, to),

        Adaptation::ChangeAccumulator {
            from,
            to,
            new_identity,
        } => {
            let mut result = replace_op(node, from, to);
            // Also replace the identity element
            result = replace_atom(&result, "0", new_identity);
            result = replace_atom(&result, "1", new_identity);
            result
        }

        Adaptation::AddEarlyReturn => {
            // Add "return" to the first branch in the tree
            add_early_return(node)
        }

        Adaptation::WrapRecursive => ProgramNode::recurse(
            ProgramNode::branch(
                ProgramNode::apply(
                    ProgramNode::op("LT"),
                    vec![ProgramNode::atom("n"), ProgramNode::typed("2", "INT")],
                ),
                ProgramNode::atom("base_case"),
                node.clone(),
            ),
            ProgramNode::atom("recurse"),
        ),
    }
}

/// Replace all occurrences of operation `from` with `to` in a tree.
fn replace_op(node: &ProgramNode, from: &str, to: &str) -> ProgramNode {
    match node {
        ProgramNode::Atom(name) if name.to_uppercase() == from.to_uppercase() => {
            ProgramNode::op(to)
        }
        ProgramNode::Apply { func, args } => ProgramNode::apply(
            replace_op(func, from, to),
            args.iter().map(|a| replace_op(a, from, to)).collect(),
        ),
        ProgramNode::Sequence(steps) => {
            ProgramNode::Sequence(steps.iter().map(|s| replace_op(s, from, to)).collect())
        }
        ProgramNode::Branch {
            condition,
            then_branch,
            else_branch,
        } => ProgramNode::branch(
            replace_op(condition, from, to),
            replace_op(then_branch, from, to),
            replace_op(else_branch, from, to),
        ),
        ProgramNode::Iterate {
            init,
            step,
            condition,
        } => ProgramNode::iterate(
            replace_op(init, from, to),
            replace_op(step, from, to),
            replace_op(condition, from, to),
        ),
        ProgramNode::Recurse {
            base_case,
            recursive_step,
        } => ProgramNode::recurse(
            replace_op(base_case, from, to),
            replace_op(recursive_step, from, to),
        ),
        ProgramNode::Reduce {
            func,
            initial,
            collection,
        } => ProgramNode::reduce(
            replace_op(func, from, to),
            replace_op(initial, from, to),
            replace_op(collection, from, to),
        ),
        _ => node.clone(),
    }
}

/// Replace atom text in a tree.
fn replace_atom(node: &ProgramNode, from: &str, to: &str) -> ProgramNode {
    match node {
        ProgramNode::Atom(name) if name == from => ProgramNode::atom(to),
        ProgramNode::Typed(name, ty) if name == from => ProgramNode::typed(to, ty),
        ProgramNode::Apply { func, args } => ProgramNode::apply(
            replace_atom(func, from, to),
            args.iter().map(|a| replace_atom(a, from, to)).collect(),
        ),
        ProgramNode::Sequence(steps) => {
            ProgramNode::Sequence(steps.iter().map(|s| replace_atom(s, from, to)).collect())
        }
        ProgramNode::Branch {
            condition,
            then_branch,
            else_branch,
        } => ProgramNode::branch(
            replace_atom(condition, from, to),
            replace_atom(then_branch, from, to),
            replace_atom(else_branch, from, to),
        ),
        _ => node.clone(),
    }
}

fn add_early_return(node: &ProgramNode) -> ProgramNode {
    match node {
        ProgramNode::Branch {
            condition,
            then_branch,
            else_branch,
        } => ProgramNode::branch(
            condition.as_ref().clone(),
            ProgramNode::Sequence(vec![
                then_branch.as_ref().clone(),
                ProgramNode::atom("return"),
            ]),
            else_branch.as_ref().clone(),
        ),
        ProgramNode::Sequence(steps) => {
            ProgramNode::Sequence(steps.iter().map(|s| add_early_return(s)).collect())
        }
        ProgramNode::Iterate {
            init,
            step,
            condition,
        } => ProgramNode::iterate(
            init.as_ref().clone(),
            add_early_return(step),
            condition.as_ref().clone(),
        ),
        _ => node.clone(),
    }
}

fn expr_text(node: &ProgramNode) -> String {
    match node {
        ProgramNode::Atom(s) => s.clone(),
        ProgramNode::Typed(s, _) => s.clone(),
        _ => "?".to_string(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

fn keyword_overlap(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let set_a: HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let set_b: HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

fn common_keywords(groups: &[&Vec<String>]) -> Vec<String> {
    if groups.is_empty() {
        return Vec::new();
    }
    let mut common: HashSet<String> = groups[0].iter().cloned().collect();
    for group in &groups[1..] {
        let set: HashSet<String> = group.iter().cloned().collect();
        common = common.intersection(&set).cloned().collect();
    }
    common.into_iter().collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program_emitter::emit_expression;

    #[test]
    fn test_seed_knowledge() {
        let engine = UnderstandingEngine::with_basic_knowledge();
        assert!(
            engine.knowledge_count() >= 15,
            "should have 15+ pairs, got {}",
            engine.knowledge_count()
        );
        assert!(
            engine.pattern_count() >= 1,
            "should have at least 1 pattern"
        );
    }

    #[test]
    fn test_understand_known_spec() {
        let engine = UnderstandingEngine::with_basic_knowledge();
        let result = engine.understand("add two numbers");
        assert!(
            result.score() > 0.7,
            "should find 'add' with high score, got {:.4}",
            result.score()
        );
        if let Some(prog) = result.program() {
            let code = emit_expression(prog);
            println!("'add two numbers' → {}", code);
        }
    }

    #[test]
    fn test_understand_similar_spec() {
        let engine = UnderstandingEngine::with_basic_knowledge();
        let result = engine.understand("find the largest element in the array");
        assert!(
            result.score() > 0.4,
            "should match 'find maximum', got {:.4}",
            result.score()
        );
    }

    #[test]
    fn test_adaptation_minimum_from_maximum() {
        let engine = UnderstandingEngine::with_basic_knowledge();
        let adaptations =
            engine.compute_adaptations("find maximum in array", "find minimum in array");
        assert!(!adaptations.is_empty(), "should produce adaptations");
        assert!(adaptations.iter().any(|a| matches!(a, Adaptation::ChangeOperator { from, to } if from == "GT" && to == "LT")),
            "should change GT to LT");
    }

    #[test]
    fn test_adaptation_second_largest() {
        let engine = UnderstandingEngine::with_basic_knowledge();
        let adaptations =
            engine.compute_adaptations("find maximum in array", "find second largest in array");
        assert!(
            adaptations
                .iter()
                .any(|a| matches!(a, Adaptation::AddTracker { .. })),
            "should add tracker for 'second'"
        );
    }

    #[test]
    fn test_adaptation_product_from_sum() {
        let engine = UnderstandingEngine::with_basic_knowledge();
        let adaptations = engine.compute_adaptations("sum of array", "product of array");
        assert!(adaptations.iter().any(|a| matches!(a, Adaptation::ChangeAccumulator { from, to, .. } if from == "ADD" && to == "MUL")),
            "should change ADD to MUL");
    }

    #[test]
    fn test_apply_change_operator() {
        let find_max = ProgramNode::branch(
            ProgramNode::apply(
                ProgramNode::op("GT"),
                vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
            ),
            ProgramNode::atom("a"),
            ProgramNode::atom("b"),
        );
        let adapted = apply_adaptations(
            &find_max,
            &[Adaptation::ChangeOperator {
                from: "GT".into(),
                to: "LT".into(),
            }],
        );
        let code = emit_expression(&adapted);
        println!("Adapted GT→LT: {}", code);
        assert!(
            code.contains("< ") || code.contains("LT"),
            "should contain LT, got: {}",
            code
        );
    }

    #[test]
    fn test_understand_then_adapt() {
        let engine = UnderstandingEngine::with_basic_knowledge();
        let result = engine.understand("find second largest element in array");
        println!("Score: {:.4}", result.score());
        if let UnderstandingResult::Adapted(prog, score, adaptations) = &result {
            let code = emit_expression(prog);
            println!("Adapted program: {}", code);
            println!("Adaptations: {:?}", adaptations);
            assert!(*score > 0.4, "adapted score should be meaningful");
        } else if let Some(prog) = result.program() {
            println!("Found: {}", emit_expression(prog));
        }
    }

    #[test]
    fn test_learning_improves_score() {
        let mut engine = UnderstandingEngine::with_basic_knowledge();
        let score_before = engine.understand("reverse array elements").score();

        // Learn a new pattern
        engine.learn(
            "reverse array elements",
            &ProgramNode::Sequence(vec![
                ProgramNode::atom("left = 0"),
                ProgramNode::atom("right = len - 1"),
                ProgramNode::iterate(
                    ProgramNode::atom("init"),
                    ProgramNode::atom("swap(arr[left], arr[right]); left += 1; right -= 1"),
                    ProgramNode::apply(
                        ProgramNode::op("LT"),
                        vec![ProgramNode::atom("left"), ProgramNode::atom("right")],
                    ),
                ),
            ]),
        );

        let score_after = engine.understand("reverse array elements").score();
        println!("Before: {:.4}, After: {:.4}", score_before, score_after);
        assert!(score_after > score_before, "learning should improve score");
    }

    #[test]
    fn test_auto_abstraction() {
        let mut engine = UnderstandingEngine::new();
        engine.learn("find maximum in array", &ProgramNode::atom("max_impl"));
        engine.learn("find minimum in array", &ProgramNode::atom("min_impl"));
        engine.learn("find median in array", &ProgramNode::atom("median_impl"));
        engine.auto_abstract();
        assert!(
            engine.pattern_count() >= 1,
            "should abstract 'find X in array' pattern"
        );
    }
}
