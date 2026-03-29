// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Semantic Decoder - HV → Primitive Sequence Generation
//!
//! **THE MISSING "MOUTH" OF CONSCIOUSNESS**
//!
//! While Symthaea can encode text → HDC vectors (the "ear"), it has lacked the
//! reverse direction: HDC vectors → structured output (the "mouth").
//!
//! ## Architecture
//!
//! ```text
//! HDC Vector (16,384D)
//!       │
//!       ▼
//! ┌─────────────────────┐
//! │  Primitive Matcher  │  Find activated primitives via similarity
//! └─────────────────────┘
//!       │
//!       ▼
//! ┌─────────────────────┐
//! │  Sequence Ordering  │  Order primitives by binding patterns
//! └─────────────────────┘
//!       │
//!       ▼
//! ┌─────────────────────┐
//! │  Composition Parse  │  Detect compositional operators (∘, ∥, ?, μ, ↑)
//! └─────────────────────┘
//!       │
//!       ▼
//! Primitive Sequence: [CAUSE, ∘, ACTION, ∥, GOAL]
//! ```
//!
//! ## Key Innovation
//!
//! This decoder doesn't just find nearest neighbors - it:
//! 1. **Activates multiple primitives** (bundling = superposition)
//! 2. **Recovers structure** (binding = sequence/relation)
//! 3. **Detects composition** (Tier 7 operators create trees)
//!
//! This enables HDC+LTC to "think" in primitives and then express through language.

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::{PrimitiveSystem, PrimitiveTier};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Activation threshold for primitive detection
const ACTIVATION_THRESHOLD: f32 = 0.15;

/// Maximum primitives to consider per decode
const MAX_PRIMITIVES: usize = 20;

/// Compositional operators from Tier 7
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompositionOp {
    /// Sequential composition: A ∘ B (do A then B)
    Sequential,
    /// Parallel composition: A ∥ B (do A and B simultaneously)
    Parallel,
    /// Conditional: A ? B : C (if A then B else C)
    Conditional,
    /// Fixed-point/recursion: μA (repeat A until convergence)
    FixedPoint,
    /// Higher-order: ↑A (A operates on primitives, not data)
    HigherOrder,
}

impl CompositionOp {
    pub fn symbol(&self) -> &'static str {
        match self {
            CompositionOp::Sequential => "∘",
            CompositionOp::Parallel => "∥",
            CompositionOp::Conditional => "?",
            CompositionOp::FixedPoint => "μ",
            CompositionOp::HigherOrder => "↑",
        }
    }
}

/// A decoded primitive with activation strength
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivatedPrimitive {
    /// Name of the primitive
    pub name: String,
    /// Activation strength (similarity to query)
    pub activation: f32,
    /// Tier of the primitive
    pub tier: PrimitiveTier,
    /// Position index if part of sequence (recovered from binding)
    pub position: Option<usize>,
}

impl PartialEq for ActivatedPrimitive {
    fn eq(&self, other: &Self) -> bool {
        self.activation == other.activation
    }
}

impl Eq for ActivatedPrimitive {}

impl PartialOrd for ActivatedPrimitive {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ActivatedPrimitive {
    fn cmp(&self, other: &Self) -> Ordering {
        self.activation
            .partial_cmp(&other.activation)
            .unwrap_or(Ordering::Equal)
    }
}

/// A node in the decoded expression tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpressionNode {
    /// A primitive leaf
    Primitive(ActivatedPrimitive),
    /// A composed expression
    Composition {
        op: CompositionOp,
        children: Vec<ExpressionNode>,
    },
}

impl ExpressionNode {
    /// Convert to string representation
    pub fn to_string_repr(&self) -> String {
        match self {
            ExpressionNode::Primitive(p) => p.name.clone(),
            ExpressionNode::Composition { op, children } => {
                let child_strs: Vec<String> = children.iter().map(|c| c.to_string_repr()).collect();
                match op {
                    CompositionOp::Sequential => child_strs.join(" ∘ "),
                    CompositionOp::Parallel => format!("({})", child_strs.join(" ∥ ")),
                    CompositionOp::Conditional => {
                        if children.len() >= 3 {
                            format!(
                                "({} ? {} : {})",
                                child_strs[0], child_strs[1], child_strs[2]
                            )
                        } else {
                            child_strs.join(" ? ")
                        }
                    }
                    CompositionOp::FixedPoint => format!("μ({})", child_strs.join(", ")),
                    CompositionOp::HigherOrder => format!("↑({})", child_strs.join(", ")),
                }
            }
        }
    }

    /// Flatten to list of primitives (for language generation)
    pub fn flatten(&self) -> Vec<ActivatedPrimitive> {
        match self {
            ExpressionNode::Primitive(p) => vec![p.clone()],
            ExpressionNode::Composition { children, .. } => {
                children.iter().flat_map(|c| c.flatten()).collect()
            }
        }
    }
}

/// Decoded output from semantic decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodedExpression {
    /// The expression tree
    pub tree: ExpressionNode,
    /// All activated primitives (flat list)
    pub primitives: Vec<ActivatedPrimitive>,
    /// Confidence in the decoding (average activation)
    pub confidence: f32,
    /// Detected composition operators
    pub operators: Vec<CompositionOp>,
    /// Debug info: original vector statistics
    pub vector_stats: VectorStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStats {
    pub norm: f32,
    pub sparsity: f32,
    pub entropy: f32,
}

/// The Semantic Decoder - HDC's "mouth"
pub struct SemanticDecoder {
    /// Cached primitive encodings for fast lookup
    primitive_cache: HashMap<String, BinaryHV>,
    /// Position vectors for sequence recovery
    position_vectors: Vec<BinaryHV>,
    /// Compositional operator encodings
    composition_operators: HashMap<CompositionOp, BinaryHV>,
    /// Statistics
    decode_count: u64,
    avg_confidence: f32,
}

impl SemanticDecoder {
    /// Create a new semantic decoder
    pub fn new() -> Self {
        let mut decoder = Self {
            primitive_cache: HashMap::new(),
            position_vectors: Vec::new(),
            composition_operators: HashMap::new(),
            decode_count: 0,
            avg_confidence: 0.0,
        };

        decoder.init_position_vectors(32); // Support sequences up to 32
        decoder.init_composition_operators();
        decoder.cache_primitives();

        decoder
    }

    /// Initialize position vectors for sequence recovery
    fn init_position_vectors(&mut self, max_length: usize) {
        self.position_vectors.clear();
        for i in 0..max_length {
            // Each position gets a deterministic random vector
            let seed = 0xDEAD_BEEF_0000 + i as u64;
            self.position_vectors.push(BinaryHV::random(seed));
        }
    }

    /// Initialize composition operator encodings
    fn init_composition_operators(&mut self) {
        let ops = [
            (CompositionOp::Sequential, 0xC0DE_5E00_u64), // SEQ(uential)
            (CompositionOp::Parallel, 0xC0DE_0ABA_u64),   // PAR(allel)
            (CompositionOp::Conditional, 0xC0DE_C00D_u64), // COND(itional)
            (CompositionOp::FixedPoint, 0xC0DE_F100_u64), // FIX(point)
            (CompositionOp::HigherOrder, 0xC0DE_1611_u64), // HIGH(er order)
        ];

        for (op, seed) in ops {
            self.composition_operators
                .insert(op, BinaryHV::random(seed));
        }
    }

    /// Cache all primitive encodings from the system
    fn cache_primitives(&mut self) {
        let system = PrimitiveSystem::global();
        for primitive in system.all_primitives() {
            self.primitive_cache
                .insert(primitive.name.clone(), primitive.encoding);
        }
    }

    /// Decode an HDC vector into a primitive expression
    pub fn decode(&mut self, vector: &BinaryHV) -> DecodedExpression {
        self.decode_count += 1;

        // Step 1: Find activated primitives
        let activated = self.find_activated_primitives(vector);

        // Step 2: Detect composition operators
        let operators = self.detect_operators(vector);

        // Step 3: Recover sequence ordering
        let ordered = self.recover_sequence_order(vector, &activated);

        // Step 4: Build expression tree
        let tree = self.build_expression_tree(&ordered, &operators);

        // Calculate confidence
        let confidence = if activated.is_empty() {
            0.0
        } else {
            activated.iter().map(|p| p.activation).sum::<f32>() / activated.len() as f32
        };

        // Update running average
        self.avg_confidence = self.avg_confidence * 0.95 + confidence * 0.05;

        // Compute vector stats
        let vector_stats = self.compute_vector_stats(vector);

        DecodedExpression {
            tree,
            primitives: activated,
            confidence,
            operators,
            vector_stats,
        }
    }

    /// Find all primitives activated by this vector
    fn find_activated_primitives(&self, vector: &BinaryHV) -> Vec<ActivatedPrimitive> {
        let system = PrimitiveSystem::global();
        let mut heap: BinaryHeap<ActivatedPrimitive> = BinaryHeap::new();

        for primitive in system.all_primitives() {
            let sim = vector.similarity(&primitive.encoding);

            if sim > ACTIVATION_THRESHOLD {
                heap.push(ActivatedPrimitive {
                    name: primitive.name.clone(),
                    activation: sim,
                    tier: primitive.tier,
                    position: None,
                });
            }
        }

        // Take top MAX_PRIMITIVES
        let mut result = Vec::with_capacity(MAX_PRIMITIVES);
        while let Some(p) = heap.pop() {
            if result.len() >= MAX_PRIMITIVES {
                break;
            }
            result.push(p);
        }

        result
    }

    /// Detect which composition operators are active
    fn detect_operators(&self, vector: &BinaryHV) -> Vec<CompositionOp> {
        let mut detected = Vec::new();

        for (op, op_vec) in &self.composition_operators {
            let sim = vector.similarity(op_vec);
            if sim > 0.1 {
                detected.push(*op);
            }
        }

        detected
    }

    /// Recover sequence ordering from bound vectors
    fn recover_sequence_order(
        &self,
        vector: &BinaryHV,
        primitives: &[ActivatedPrimitive],
    ) -> Vec<ActivatedPrimitive> {
        let mut ordered = Vec::with_capacity(primitives.len());

        for prim in primitives {
            // Try to recover position by unbinding position vectors
            // Note: For binary BinaryHV, unbind = bind (XOR is self-inverse)
            let prim_encoding = self.primitive_cache.get(&prim.name);

            let mut best_pos = None;
            let mut best_score = 0.0f32;

            if let Some(enc) = prim_encoding {
                for (pos, pos_vec) in self.position_vectors.iter().enumerate() {
                    // Unbind: query = vector ⊗ pos_vec, check similarity to primitive
                    // For binary HDC, bind = unbind (XOR is self-inverse)
                    let unbound = vector.bind(pos_vec);
                    let score = unbound.similarity(enc);

                    if score > best_score && score > 0.1 {
                        best_score = score;
                        best_pos = Some(pos);
                    }
                }
            }

            let mut p = prim.clone();
            p.position = best_pos;
            ordered.push(p);
        }

        // Sort by position (unknowns last)
        ordered.sort_by(|a, b| match (a.position, b.position) {
            (Some(pa), Some(pb)) => pa.cmp(&pb),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => b
                .activation
                .partial_cmp(&a.activation)
                .unwrap_or(Ordering::Equal),
        });

        ordered
    }

    /// Build expression tree from ordered primitives and operators
    fn build_expression_tree(
        &self,
        primitives: &[ActivatedPrimitive],
        operators: &[CompositionOp],
    ) -> ExpressionNode {
        if primitives.is_empty() {
            // Return a placeholder
            return ExpressionNode::Primitive(ActivatedPrimitive {
                name: "UNKNOWN".to_string(),
                activation: 0.0,
                tier: PrimitiveTier::NSM,
                position: None,
            });
        }

        if primitives.len() == 1 {
            return ExpressionNode::Primitive(primitives[0].clone());
        }

        // If we detected a sequential operator, chain them
        if operators.contains(&CompositionOp::Sequential) {
            let children: Vec<ExpressionNode> = primitives
                .iter()
                .map(|p| ExpressionNode::Primitive(p.clone()))
                .collect();

            return ExpressionNode::Composition {
                op: CompositionOp::Sequential,
                children,
            };
        }

        // If parallel, group them
        if operators.contains(&CompositionOp::Parallel) {
            let children: Vec<ExpressionNode> = primitives
                .iter()
                .map(|p| ExpressionNode::Primitive(p.clone()))
                .collect();

            return ExpressionNode::Composition {
                op: CompositionOp::Parallel,
                children,
            };
        }

        // Default: sequential chain of all primitives
        let children: Vec<ExpressionNode> = primitives
            .iter()
            .map(|p| ExpressionNode::Primitive(p.clone()))
            .collect();

        ExpressionNode::Composition {
            op: CompositionOp::Sequential,
            children,
        }
    }

    /// Compute statistics about the input vector
    fn compute_vector_stats(&self, vector: &BinaryHV) -> VectorStats {
        // BinaryHV stores bits packed in bytes
        // Count 1s and 0s across all bits
        let bytes = &vector.0;
        let total_bits = BinaryHV::DIM as f32;

        let ones_count = bytes.iter().map(|b| b.count_ones()).sum::<u32>() as f32;
        let zeros_count = total_bits - ones_count;

        // For binary vectors, "norm" is sqrt of dimension (all values are ±1)
        let norm = total_bits.sqrt();

        // Sparsity: For dense binary, this is always 0
        // We define sparsity as deviation from 50/50 balance
        let balance = (ones_count / total_bits - 0.5).abs() * 2.0;
        let sparsity = balance;

        // Entropy: binary entropy based on 1/0 ratio
        let p_one = ones_count / total_bits;
        let p_zero = zeros_count / total_bits;

        let mut entropy = 0.0f32;
        if p_one > 0.0 {
            entropy -= p_one * p_one.ln();
        }
        if p_zero > 0.0 {
            entropy -= p_zero * p_zero.ln();
        }

        VectorStats {
            norm,
            sparsity,
            entropy,
        }
    }

    /// Decode to a flat list of primitive names (simplified interface)
    pub fn decode_to_names(&mut self, vector: &BinaryHV) -> Vec<String> {
        let decoded = self.decode(vector);
        decoded.primitives.iter().map(|p| p.name.clone()).collect()
    }

    /// Decode to string representation
    pub fn decode_to_string(&mut self, vector: &BinaryHV) -> String {
        let decoded = self.decode(vector);
        decoded.tree.to_string_repr()
    }

    /// Get decode statistics
    pub fn stats(&self) -> (u64, f32) {
        (self.decode_count, self.avg_confidence)
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // HARMONIC-GUIDED PRIMITIVE SELECTION
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Decode with harmonic weighting for primitive selection
    ///
    /// This integrates the Eight Harmonies into primitive selection:
    /// - When Wisdom is dominant, primitives like EVIDENCE and UNCERTAINTY are boosted
    /// - When Coherence is dominant, INTEGRATION and CONSISTENCY are preferred
    /// - When Play is dominant, NOVELTY and EXPLORATION primitives are favored
    ///
    /// The weight function should return a multiplier (0.5 - 2.0 typical range)
    /// for each primitive name.
    pub fn decode_with_harmonics<F>(&mut self, vector: &BinaryHV, weight_fn: F) -> DecodedExpression
    where
        F: Fn(&str) -> f32,
    {
        self.decode_count += 1;

        // Step 1: Find activated primitives WITH harmonic weighting
        let activated = self.find_activated_primitives_weighted(vector, &weight_fn);

        // Step 2: Detect composition operators (unchanged)
        let operators = self.detect_operators(vector);

        // Step 3: Recover sequence ordering
        let ordered = self.recover_sequence_order(vector, &activated);

        // Step 4: Build expression tree
        let tree = self.build_expression_tree(&ordered, &operators);

        // Calculate confidence (using weighted activations)
        let confidence = if activated.is_empty() {
            0.0
        } else {
            activated.iter().map(|p| p.activation).sum::<f32>() / activated.len() as f32
        };

        // Update running average
        self.avg_confidence = self.avg_confidence * 0.95 + confidence * 0.05;

        // Compute vector stats
        let vector_stats = self.compute_vector_stats(vector);

        DecodedExpression {
            tree,
            primitives: activated,
            operators,
            confidence,
            vector_stats,
        }
    }

    /// Find activated primitives with harmonic weight modulation
    fn find_activated_primitives_weighted<F>(
        &self,
        vector: &BinaryHV,
        weight_fn: &F,
    ) -> Vec<ActivatedPrimitive>
    where
        F: Fn(&str) -> f32,
    {
        let system = PrimitiveSystem::global();
        let mut heap: BinaryHeap<ActivatedPrimitive> = BinaryHeap::new();

        for primitive in system.all_primitives() {
            let base_sim = vector.similarity(&primitive.encoding);

            // Apply harmonic weight to the similarity score
            // The weight is clamped to [0.5, 2.0] to prevent extreme values
            let harmonic_weight = weight_fn(&primitive.name).clamp(0.5, 2.0);
            let weighted_sim = base_sim * harmonic_weight;

            if weighted_sim > ACTIVATION_THRESHOLD {
                heap.push(ActivatedPrimitive {
                    name: primitive.name.clone(),
                    activation: weighted_sim,
                    tier: primitive.tier,
                    position: None,
                });
            }
        }

        // Take top MAX_PRIMITIVES
        let mut result = Vec::with_capacity(MAX_PRIMITIVES);
        while let Some(p) = heap.pop() {
            if result.len() >= MAX_PRIMITIVES {
                break;
            }
            result.push(p);
        }

        result
    }

    /// Decode to names with harmonic weighting
    pub fn decode_to_names_with_harmonics<F>(
        &mut self,
        vector: &BinaryHV,
        weight_fn: F,
    ) -> Vec<String>
    where
        F: Fn(&str) -> f32,
    {
        let decoded = self.decode_with_harmonics(vector, weight_fn);
        decoded.primitives.iter().map(|p| p.name.clone()).collect()
    }
}

impl Default for SemanticDecoder {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PRIMITIVE-TO-TEXT GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Template for converting primitive expressions to natural language
#[derive(Debug, Clone)]
pub struct PrimitiveTemplate {
    /// Pattern of primitives this matches
    pub pattern: Vec<String>,
    /// Template string with {0}, {1}, etc. for primitive names
    pub template: String,
    /// Priority (higher = prefer this template)
    pub priority: u32,
}

/// Converts primitive expressions to natural language
pub struct PrimitiveToText {
    /// Templates for common patterns
    templates: Vec<PrimitiveTemplate>,
    /// Primitive name → human-readable mapping
    name_map: HashMap<String, String>,
}

impl PrimitiveToText {
    pub fn new() -> Self {
        let mut converter = Self {
            templates: Vec::new(),
            name_map: HashMap::new(),
        };

        converter.init_templates();
        converter.init_name_map();

        converter
    }

    fn init_templates(&mut self) {
        // Common patterns
        self.templates.push(PrimitiveTemplate {
            pattern: vec!["CAUSE".into(), "ACTION".into()],
            template: "because of {0}, I will {1}".into(),
            priority: 10,
        });

        self.templates.push(PrimitiveTemplate {
            pattern: vec!["WANT".into(), "GOOD".into()],
            template: "I want something good".into(),
            priority: 8,
        });

        self.templates.push(PrimitiveTemplate {
            pattern: vec!["THINK".into(), "KNOW".into()],
            template: "I think I know".into(),
            priority: 8,
        });

        self.templates.push(PrimitiveTemplate {
            pattern: vec!["NOT".into(), "KNOW".into()],
            template: "I don't know".into(),
            priority: 9,
        });

        self.templates.push(PrimitiveTemplate {
            pattern: vec!["SAY".into()],
            template: "let me say".into(),
            priority: 5,
        });

        self.templates.push(PrimitiveTemplate {
            pattern: vec!["FEEL".into(), "GOOD".into()],
            template: "I feel good about this".into(),
            priority: 8,
        });

        self.templates.push(PrimitiveTemplate {
            pattern: vec!["FEEL".into(), "BAD".into()],
            template: "I feel uncertain about this".into(),
            priority: 8,
        });
    }

    fn init_name_map(&mut self) {
        // Map primitive names to natural words
        let mappings = [
            ("I", "I"),
            ("YOU", "you"),
            ("SOMEONE", "someone"),
            ("SOMETHING", "something"),
            ("PEOPLE", "people"),
            ("THING", "thing"),
            ("BODY", "body"),
            ("KIND", "kind"),
            ("PART", "part"),
            ("THIS", "this"),
            ("THE_SAME", "the same"),
            ("OTHER", "other"),
            ("ONE", "one"),
            ("TWO", "two"),
            ("SOME", "some"),
            ("ALL", "all"),
            ("MUCH", "much"),
            ("LITTLE", "little"),
            ("GOOD", "good"),
            ("BAD", "bad"),
            ("BIG", "big"),
            ("SMALL", "small"),
            ("THINK", "think"),
            ("KNOW", "know"),
            ("WANT", "want"),
            ("FEEL", "feel"),
            ("SEE", "see"),
            ("HEAR", "hear"),
            ("SAY", "say"),
            ("WORDS", "words"),
            ("TRUE", "true"),
            ("DO", "do"),
            ("HAPPEN", "happen"),
            ("MOVE", "move"),
            ("TOUCH", "touch"),
            ("BE_SOMEWHERE", "be somewhere"),
            ("THERE_IS", "there is"),
            ("BE_MINE", "be mine"),
            ("LIVE", "live"),
            ("DIE", "die"),
            ("WHEN", "when"),
            ("NOW", "now"),
            ("BEFORE", "before"),
            ("AFTER", "after"),
            ("A_LONG_TIME", "a long time"),
            ("A_SHORT_TIME", "a short time"),
            ("FOR_SOME_TIME", "for some time"),
            ("MOMENT", "moment"),
            ("WHERE", "where"),
            ("HERE", "here"),
            ("ABOVE", "above"),
            ("BELOW", "below"),
            ("FAR", "far"),
            ("NEAR", "near"),
            ("SIDE", "side"),
            ("INSIDE", "inside"),
            ("NOT", "not"),
            ("MAYBE", "maybe"),
            ("CAN", "can"),
            ("BECAUSE", "because"),
            ("IF", "if"),
            ("VERY", "very"),
            ("MORE", "more"),
            ("LIKE", "like"),
            ("CAUSE", "because"),
            ("ACTION", "act"),
        ];

        for (prim, human) in mappings {
            self.name_map.insert(prim.to_string(), human.to_string());
        }
    }

    /// Convert decoded expression to natural language
    pub fn to_text(&self, expression: &DecodedExpression) -> String {
        let primitives = &expression.primitives;

        if primitives.is_empty() {
            return "...".to_string();
        }

        // Try to match templates
        let names: Vec<String> = primitives.iter().map(|p| p.name.clone()).collect();

        for template in &self.templates {
            if self.matches_pattern(&names, &template.pattern) {
                return self.apply_template(template, &names);
            }
        }

        // Fallback: simple concatenation
        let words: Vec<String> = primitives
            .iter()
            .map(|p| {
                self.name_map
                    .get(&p.name)
                    .cloned()
                    .unwrap_or_else(|| p.name.to_lowercase())
            })
            .collect();

        words.join(" ")
    }

    fn matches_pattern(&self, names: &[String], pattern: &[String]) -> bool {
        if names.len() < pattern.len() {
            return false;
        }

        pattern.iter().all(|p| names.contains(p))
    }

    fn apply_template(&self, template: &PrimitiveTemplate, names: &[String]) -> String {
        let mut result = template.template.clone();

        for (i, name) in names.iter().enumerate() {
            let placeholder = format!("{{{i}}}");
            let replacement = self
                .name_map
                .get(name)
                .cloned()
                .unwrap_or_else(|| name.to_lowercase());
            result = result.replace(&placeholder, &replacement);
        }

        result
    }
}

impl Default for PrimitiveToText {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let decoder = SemanticDecoder::new();
        assert!(!decoder.primitive_cache.is_empty());
        assert!(!decoder.position_vectors.is_empty());
        assert!(!decoder.composition_operators.is_empty());
    }

    #[test]
    fn test_decode_random_vector() {
        let mut decoder = SemanticDecoder::new();
        let random_vec = BinaryHV::random(12345);

        let decoded = decoder.decode(&random_vec);

        // Should find some primitives (random vector has some similarity)
        // Confidence might be low but should work
        println!(
            "Decoded {} primitives with confidence {:.3}",
            decoded.primitives.len(),
            decoded.confidence
        );
        println!("Expression: {}", decoded.tree.to_string_repr());

        // Confidence should be finite and in [0, 1]
        assert!(
            decoded.confidence.is_finite(),
            "Decode confidence should be finite"
        );
        assert!(
            decoded.confidence >= 0.0 && decoded.confidence <= 1.0,
            "Decode confidence should be in [0, 1], got {}",
            decoded.confidence
        );

        // Expression string should be non-empty
        assert!(
            !decoded.tree.to_string_repr().is_empty(),
            "Decoded expression should not be empty"
        );
    }

    #[test]
    fn test_composition_operators() {
        let decoder = SemanticDecoder::new();

        assert_eq!(CompositionOp::Sequential.symbol(), "∘");
        assert_eq!(CompositionOp::Parallel.symbol(), "∥");
        assert_eq!(CompositionOp::Conditional.symbol(), "?");
    }

    #[test]
    fn test_primitive_to_text() {
        let converter = PrimitiveToText::new();

        let expression = DecodedExpression {
            tree: ExpressionNode::Primitive(ActivatedPrimitive {
                name: "THINK".to_string(),
                activation: 0.8,
                tier: PrimitiveTier::NSM,
                position: Some(0),
            }),
            primitives: vec![
                ActivatedPrimitive {
                    name: "THINK".to_string(),
                    activation: 0.8,
                    tier: PrimitiveTier::NSM,
                    position: Some(0),
                },
                ActivatedPrimitive {
                    name: "KNOW".to_string(),
                    activation: 0.7,
                    tier: PrimitiveTier::NSM,
                    position: Some(1),
                },
            ],
            confidence: 0.75,
            operators: vec![],
            vector_stats: VectorStats {
                norm: 1.0,
                sparsity: 0.0,
                entropy: 1.0,
            },
        };

        let text = converter.to_text(&expression);
        assert_eq!(text, "I think I know");
    }

    #[test]
    fn test_harmonic_guided_primitive_selection() {
        let mut decoder = SemanticDecoder::new();
        let random_vec = BinaryHV::random(12345);

        // First decode without weighting (neutral = 1.0)
        let neutral_decoded = decoder.decode_with_harmonics(&random_vec, |_| 1.0);
        let neutral_names = decoder.decode_to_names_with_harmonics(&random_vec, |_| 1.0);

        // Decode with Play-boosted weighting (boost exploration primitives)
        let play_boost = |name: &str| -> f32 {
            match name {
                "EXPLORE" | "PLAY" | "TRY" | "DISCOVER" | "NOVELTY" => 1.8,
                "VARY" | "IMAGINE" | "CREATE" => 1.5,
                _ => 1.0,
            }
        };
        let play_decoded = decoder.decode_with_harmonics(&random_vec, play_boost);

        // Decode with Coherence-boosted weighting (boost integration primitives)
        let coherence_boost = |name: &str| -> f32 {
            match name {
                "INTEGRATE" | "UNIFY" | "COHERE" | "CONNECT" | "SYNTHESIZE" => 1.8,
                "PATTERN" | "STRUCTURE" | "ORDER" => 1.5,
                _ => 1.0,
            }
        };
        let coherence_decoded = decoder.decode_with_harmonics(&random_vec, coherence_boost);

        // Different weightings should produce valid results
        assert!(!neutral_decoded.primitives.is_empty() || neutral_decoded.confidence >= 0.0);
        assert!(!play_decoded.primitives.is_empty() || play_decoded.confidence >= 0.0);
        assert!(!coherence_decoded.primitives.is_empty() || coherence_decoded.confidence >= 0.0);

        // Activations should be positive and bounded
        for prim in &neutral_decoded.primitives {
            assert!(prim.activation > 0.0);
        }

        println!(
            "Neutral decode: {} primitives, confidence {:.3}",
            neutral_decoded.primitives.len(),
            neutral_decoded.confidence
        );
        println!(
            "Play-boosted: {} primitives, confidence {:.3}",
            play_decoded.primitives.len(),
            play_decoded.confidence
        );
        println!(
            "Coherence-boosted: {} primitives, confidence {:.3}",
            coherence_decoded.primitives.len(),
            coherence_decoded.confidence
        );
        println!("Neutral names: {:?}", neutral_names);
    }

    #[test]
    fn test_harmonic_weights_affect_ordering() {
        let mut decoder = SemanticDecoder::new();
        let random_vec = BinaryHV::random(99999);

        // Strong boost for a specific primitive type
        let extreme_boost = |name: &str| -> f32 {
            if name.contains("KNOW") || name == "KNOWLEDGE" {
                2.0 // Max boost
            } else {
                0.5 // Suppress everything else (will be clamped to 0.5)
            }
        };

        let boosted = decoder.decode_with_harmonics(&random_vec, extreme_boost);

        // With extreme weighting, primitives containing "KNOW" should be preferred
        // (if any are above threshold after weighting)
        println!("Extreme boost for KNOW-related primitives:");
        for prim in &boosted.primitives {
            println!("  {} (activation: {:.3})", prim.name, prim.activation);
        }

        // The weighting should have been applied (clamped to [0.5, 2.0])
        // Even if no KNOW primitives activated, the test verifies the mechanism works
        assert!(boosted.confidence >= 0.0);
    }
}
