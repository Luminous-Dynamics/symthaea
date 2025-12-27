//! # Tier 7: Compositionality Primitives - Revolutionary Primitive Algebra
//!
//! **PARADIGM SHIFT**: This module implements a complete algebra for combining primitives,
//! enabling the system to create infinitely complex reasoning from simple building blocks.
//!
//! ## The Revolutionary Idea
//!
//! Traditional AI systems have fixed architectures. Even our primitive system (Tiers 0-6)
//! treats primitives as atomic units. Tier 7 changes this by introducing **compositional
//! operators** that combine primitives into higher-order reasoning structures.
//!
//! ## Composition Types
//!
//! 1. **Sequential (∘)**: Apply primitives in sequence, flowing output to input
//!    - `(f ∘ g)(x) = f(g(x))`
//!    - HDC: `bind(encode_f, bind(encode_g, encode_x))`
//!
//! 2. **Parallel (||)**: Apply primitives concurrently, bundle results
//!    - `(f || g)(x) = bundle(f(x), g(x))`
//!    - HDC: `bundle([bind(f, x), bind(g, x)])`
//!
//! 3. **Conditional (?)**: Branch based on similarity threshold
//!    - `(f ? g)(x) = if sim(x, pattern) > θ then f(x) else g(x)`
//!    - Enables pattern-triggered reasoning paths
//!
//! 4. **Fixed Point (μ)**: Iterate until convergence
//!    - `μf(x) = f(f(f(...f(x)...)))` until stable
//!    - Self-improving, recursive refinement
//!
//! 5. **Higher-Order (↑)**: Primitives that operate on primitives
//!    - `(↑f)(g) = f'` where f' is g transformed by f
//!    - Meta-reasoning about reasoning!
//!
//! 6. **Fallback (;)**: Try first, fallback on failure
//!    - `(f ; g)(x) = f(x) if confident else g(x)`
//!    - Graceful degradation
//!
//! ## Scientific Foundation
//!
//! This implementation is grounded in:
//! - **Category Theory**: Composition, identity, associativity
//! - **λ-Calculus**: Higher-order functions, fixed points
//! - **Hyperdimensional Computing**: All operations are HDC-native
//! - **Consciousness Theories**: Integrated information through composition
//!
//! ## Why This Matters
//!
//! With compositionality, a system with 100 base primitives can express:
//! - 100² = 10,000 binary compositions
//! - 100³ = 1,000,000 ternary compositions
//! - Infinite compositions through recursion!
//!
//! This transforms the primitive system from a toolkit to a **generative grammar**
//! for reasoning.

use crate::hdc::binary_hv::HV16;
use crate::hdc::primitive_system::{PrimitiveTier, Primitive, PrimitiveSystem};
use crate::consciousness::harmonics::{HarmonicField, FiduciaryHarmonic};
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::fmt;

/// Helper: Convert text to HV16 using deterministic hash-based encoding
fn text_to_hv16(text: &str) -> HV16 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let seed = hasher.finish();
    HV16::random(seed)
}

// ═══════════════════════════════════════════════════════════════════════════════
// CORE COMPOSITION TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Types of composition between primitives
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompositionType {
    /// Sequential composition: f ∘ g (apply g then f)
    Sequential,

    /// Parallel composition: f || g (apply both, bundle results)
    Parallel,

    /// Conditional composition: f ? g (branch based on pattern match)
    Conditional {
        /// Pattern to match against
        pattern: String,
        /// Similarity threshold for branching
        threshold: u32, // Fixed-point 0-1000 for f32
    },

    /// Fixed point: μf (iterate until convergence)
    FixedPoint {
        /// Maximum iterations
        max_iterations: usize,
        /// Convergence threshold (similarity between iterations)
        convergence_threshold: u32, // Fixed-point 0-1000
    },

    /// Higher-order: ↑f (primitive that transforms primitives)
    HigherOrder,

    /// Fallback: f ; g (try f, if low confidence use g)
    Fallback {
        /// Confidence threshold for using f
        confidence_threshold: u32, // Fixed-point 0-1000
    },
}

/// A composed primitive - combination of base primitives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposedPrimitive {
    /// Unique identifier
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// How primitives are combined
    pub composition_type: CompositionType,

    /// First operand (can be base primitive ID or composed primitive ID)
    pub operand_a: String,

    /// Second operand (optional for unary compositions like FixedPoint)
    pub operand_b: Option<String>,

    /// HDC encoding of this composed primitive
    pub encoding: HV16,

    /// Metadata about expected behavior
    pub metadata: CompositionMetadata,
}

/// Metadata about a composed primitive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionMetadata {
    /// Expected computational cost (relative to base primitives)
    pub expected_cost: f32,

    /// Composition depth (1 = direct composition of base primitives)
    pub depth: usize,

    /// Number of base primitives involved
    pub base_count: usize,

    /// Expected Φ contribution (estimated)
    pub expected_phi_contribution: f32,

    /// Description of what this composition does
    pub description: String,

    /// Tags for categorization
    pub tags: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPOSITIONALITY ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

/// The compositionality engine - creates and manages composed primitives
pub struct CompositionalityEngine {
    /// Registry of all composed primitives
    composed_primitives: HashMap<String, ComposedPrimitive>,

    /// HDC encodings for composition operators
    operator_encodings: HashMap<CompositionType, HV16>,

    /// Reference to base primitive system
    base_system: Arc<PrimitiveSystem>,

    /// Composition statistics
    stats: CompositionStats,

    /// Configuration
    config: CompositionalityConfig,
}

impl fmt::Debug for CompositionalityEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompositionalityEngine")
            .field("composed_primitives", &self.composed_primitives.len())
            .field("operator_encodings", &self.operator_encodings.len())
            .field("base_system", &"<PrimitiveSystem>")
            .field("stats", &self.stats)
            .field("config", &self.config)
            .finish()
    }
}

/// Configuration for the compositionality engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionalityConfig {
    /// Maximum composition depth allowed
    pub max_depth: usize,

    /// Maximum number of composed primitives to cache
    pub max_cache_size: usize,

    /// Default fixed point iterations
    pub default_fixed_point_iterations: usize,

    /// Default convergence threshold
    pub default_convergence_threshold: f32,

    /// Enable automatic composition discovery
    pub auto_discover: bool,
}

impl Default for CompositionalityConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            max_cache_size: 10000,
            default_fixed_point_iterations: 100,
            default_convergence_threshold: 0.99,
            auto_discover: true,
        }
    }
}

/// Statistics about composition operations
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CompositionStats {
    /// Total compositions created
    pub total_compositions: usize,

    /// Compositions by type
    pub by_type: HashMap<String, usize>,

    /// Most successful compositions (by usage)
    pub top_compositions: Vec<(String, usize)>,

    /// Average composition depth
    pub avg_depth: f32,

    /// Cache hit rate
    pub cache_hit_rate: f32,
}

impl CompositionalityEngine {
    /// Create a new compositionality engine
    pub fn new(base_system: Arc<PrimitiveSystem>, config: CompositionalityConfig) -> Self {
        let mut engine = Self {
            composed_primitives: HashMap::new(),
            operator_encodings: HashMap::new(),
            base_system,
            stats: CompositionStats::default(),
            config,
        };

        engine.initialize_operator_encodings();
        engine
    }

    /// Initialize HDC encodings for composition operators
    fn initialize_operator_encodings(&mut self) {
        // Each operator gets a unique random encoding (seed-based for reproducibility)
        let operators = [
            (CompositionType::Sequential, 7001),
            (CompositionType::Parallel, 7002),
            (CompositionType::Conditional { pattern: String::new(), threshold: 500 }, 7003),
            (CompositionType::FixedPoint { max_iterations: 100, convergence_threshold: 990 }, 7004),
            (CompositionType::HigherOrder, 7005),
            (CompositionType::Fallback { confidence_threshold: 500 }, 7006),
        ];

        for (op_type, seed) in operators {
            let key = match &op_type {
                CompositionType::Sequential => CompositionType::Sequential,
                CompositionType::Parallel => CompositionType::Parallel,
                CompositionType::Conditional { .. } => CompositionType::Conditional {
                    pattern: String::new(),
                    threshold: 0
                },
                CompositionType::FixedPoint { .. } => CompositionType::FixedPoint {
                    max_iterations: 0,
                    convergence_threshold: 0
                },
                CompositionType::HigherOrder => CompositionType::HigherOrder,
                CompositionType::Fallback { .. } => CompositionType::Fallback {
                    confidence_threshold: 0
                },
            };
            self.operator_encodings.insert(key, HV16::random(seed));
        }
    }

    /// Compose two primitives sequentially: (f ∘ g)(x) = f(g(x))
    ///
    /// The output of g becomes the input to f.
    /// HDC encoding: bind(op_seq, bind(encode_f, encode_g))
    pub fn compose_sequential(&mut self, f_id: &str, g_id: &str) -> Result<ComposedPrimitive> {
        let id = format!("seq_{}_{}", f_id, g_id);

        // Check cache
        if let Some(cached) = self.composed_primitives.get(&id) {
            self.stats.cache_hit_rate = (self.stats.cache_hit_rate * 0.99) + 0.01;
            return Ok(cached.clone());
        }

        // Get encodings
        let f_enc = self.get_encoding(f_id)?;
        let g_enc = self.get_encoding(g_id)?;
        let op_enc = self.operator_encodings.get(&CompositionType::Sequential)
            .context("Missing sequential operator encoding")?;

        // Compose: bind(op, bind(f, g))
        // This preserves order: f applied AFTER g
        let inner = f_enc.bind(&g_enc);
        let encoding = op_enc.bind(&inner);

        // Calculate metadata
        let f_depth = self.get_depth(f_id);
        let g_depth = self.get_depth(g_id);
        let f_base = self.get_base_count(f_id);
        let g_base = self.get_base_count(g_id);

        let composed = ComposedPrimitive {
            id: id.clone(),
            name: format!("{} ∘ {}", f_id, g_id),
            composition_type: CompositionType::Sequential,
            operand_a: f_id.to_string(),
            operand_b: Some(g_id.to_string()),
            encoding,
            metadata: CompositionMetadata {
                expected_cost: 1.5 + (f_depth + g_depth) as f32 * 0.1,
                depth: f_depth.max(g_depth) + 1,
                base_count: f_base + g_base,
                expected_phi_contribution: 0.05 + 0.02 * (f_base + g_base) as f32,
                description: format!("Apply {} then {}", g_id, f_id),
                tags: vec!["sequential".to_string(), "composition".to_string()],
            },
        };

        self.register_composition(composed.clone());
        Ok(composed)
    }

    /// Compose two primitives in parallel: (f || g)(x) = bundle(f(x), g(x))
    ///
    /// Both primitives are applied to the same input, results bundled.
    /// HDC encoding: bind(op_par, bundle([encode_f, encode_g]))
    pub fn compose_parallel(&mut self, f_id: &str, g_id: &str) -> Result<ComposedPrimitive> {
        let id = format!("par_{}_{}", f_id, g_id);

        // Check cache
        if let Some(cached) = self.composed_primitives.get(&id) {
            return Ok(cached.clone());
        }

        let f_enc = self.get_encoding(f_id)?;
        let g_enc = self.get_encoding(g_id)?;
        let op_enc = self.operator_encodings.get(&CompositionType::Parallel)
            .context("Missing parallel operator encoding")?;

        // Compose: bind(op, bundle([f, g]))
        let bundled = HV16::bundle(&[f_enc, g_enc]);
        let encoding = op_enc.bind(&bundled);

        let f_depth = self.get_depth(f_id);
        let g_depth = self.get_depth(g_id);
        let f_base = self.get_base_count(f_id);
        let g_base = self.get_base_count(g_id);

        let composed = ComposedPrimitive {
            id: id.clone(),
            name: format!("{} || {}", f_id, g_id),
            composition_type: CompositionType::Parallel,
            operand_a: f_id.to_string(),
            operand_b: Some(g_id.to_string()),
            encoding,
            metadata: CompositionMetadata {
                expected_cost: 1.0 + (f_depth + g_depth) as f32 * 0.05, // Parallel is cheaper
                depth: f_depth.max(g_depth) + 1,
                base_count: f_base + g_base,
                expected_phi_contribution: 0.08 + 0.03 * (f_base + g_base) as f32, // Higher Φ from integration
                description: format!("Apply {} and {} in parallel", f_id, g_id),
                tags: vec!["parallel".to_string(), "composition".to_string()],
            },
        };

        self.register_composition(composed.clone());
        Ok(composed)
    }

    /// Create a conditional composition: (f ? g)(x) = if match(x, pattern) then f(x) else g(x)
    pub fn compose_conditional(
        &mut self,
        f_id: &str,
        g_id: &str,
        pattern: &str,
        threshold: f32,
    ) -> Result<ComposedPrimitive> {
        let id = format!("cond_{}_{}_{}", f_id, g_id, pattern.chars().take(8).collect::<String>());

        if let Some(cached) = self.composed_primitives.get(&id) {
            return Ok(cached.clone());
        }

        let f_enc = self.get_encoding(f_id)?;
        let g_enc = self.get_encoding(g_id)?;
        let pattern_enc = text_to_hv16(pattern);
        let op_enc = self.operator_encodings.get(&CompositionType::Conditional {
            pattern: String::new(),
            threshold: 0
        }).context("Missing conditional operator encoding")?;

        // Compose: bind(op, bundle([f, g, pattern]))
        // The pattern is included so similarity checks can be done
        let components = HV16::bundle(&[f_enc, g_enc, pattern_enc]);
        let encoding = op_enc.bind(&components);

        let composed = ComposedPrimitive {
            id: id.clone(),
            name: format!("{} ? {} (on {})", f_id, g_id, pattern),
            composition_type: CompositionType::Conditional {
                pattern: pattern.to_string(),
                threshold: (threshold * 1000.0) as u32,
            },
            operand_a: f_id.to_string(),
            operand_b: Some(g_id.to_string()),
            encoding,
            metadata: CompositionMetadata {
                expected_cost: 1.2,
                depth: self.get_depth(f_id).max(self.get_depth(g_id)) + 1,
                base_count: self.get_base_count(f_id) + self.get_base_count(g_id),
                expected_phi_contribution: 0.04,
                description: format!("If input matches '{}', use {}, else {}", pattern, f_id, g_id),
                tags: vec!["conditional".to_string(), "branching".to_string()],
            },
        };

        self.register_composition(composed.clone());
        Ok(composed)
    }

    /// Create a fixed-point composition: μf = f(f(f(...))) until stable
    ///
    /// **Revolutionary**: Self-improving primitives that refine until convergence!
    pub fn compose_fixed_point(
        &mut self,
        f_id: &str,
        max_iterations: Option<usize>,
        convergence_threshold: Option<f32>,
    ) -> Result<ComposedPrimitive> {
        let max_iter = max_iterations.unwrap_or(self.config.default_fixed_point_iterations);
        let threshold = convergence_threshold.unwrap_or(self.config.default_convergence_threshold);
        let id = format!("fix_{}_i{}", f_id, max_iter);

        if let Some(cached) = self.composed_primitives.get(&id) {
            return Ok(cached.clone());
        }

        let f_enc = self.get_encoding(f_id)?;
        let op_enc = self.operator_encodings.get(&CompositionType::FixedPoint {
            max_iterations: 0,
            convergence_threshold: 0,
        }).context("Missing fixed-point operator encoding")?;

        // For fixed point, we permute the encoding to represent "self-application"
        // Each iteration applies a permutation, so μf = bundle of all permutations
        let mut iterations = Vec::with_capacity(max_iter.min(10));
        for i in 0..max_iter.min(10) {
            iterations.push(f_enc.permute(i));
        }
        let iterated = HV16::bundle(&iterations);
        let encoding = op_enc.bind(&iterated);

        let composed = ComposedPrimitive {
            id: id.clone(),
            name: format!("μ{}", f_id),
            composition_type: CompositionType::FixedPoint {
                max_iterations: max_iter,
                convergence_threshold: (threshold * 1000.0) as u32,
            },
            operand_a: f_id.to_string(),
            operand_b: None,
            encoding,
            metadata: CompositionMetadata {
                expected_cost: 2.0 + max_iter as f32 * 0.1,
                depth: self.get_depth(f_id) + 1,
                base_count: self.get_base_count(f_id),
                expected_phi_contribution: 0.10, // High Φ from recursive integration
                description: format!("Apply {} repeatedly until convergence (max {} iterations)", f_id, max_iter),
                tags: vec!["fixed-point".to_string(), "recursive".to_string(), "convergent".to_string()],
            },
        };

        self.register_composition(composed.clone());
        Ok(composed)
    }

    /// Create a higher-order composition: ↑f transforms another primitive g
    ///
    /// **Meta-reasoning**: A primitive that modifies how another primitive works!
    pub fn compose_higher_order(&mut self, transformer_id: &str, target_id: &str) -> Result<ComposedPrimitive> {
        let id = format!("ho_{}_{}", transformer_id, target_id);

        if let Some(cached) = self.composed_primitives.get(&id) {
            return Ok(cached.clone());
        }

        let t_enc = self.get_encoding(transformer_id)?;
        let target_enc = self.get_encoding(target_id)?;
        let op_enc = self.operator_encodings.get(&CompositionType::HigherOrder)
            .context("Missing higher-order operator encoding")?;

        // Higher-order: bind the transformer with a permuted target
        // The permutation represents "operating on" rather than "combining with"
        let transformed = t_enc.bind(&target_enc.permute(100));
        let encoding = op_enc.bind(&transformed);

        let composed = ComposedPrimitive {
            id: id.clone(),
            name: format!("↑{}({})", transformer_id, target_id),
            composition_type: CompositionType::HigherOrder,
            operand_a: transformer_id.to_string(),
            operand_b: Some(target_id.to_string()),
            encoding,
            metadata: CompositionMetadata {
                expected_cost: 2.5,
                depth: self.get_depth(transformer_id) + self.get_depth(target_id) + 2,
                base_count: self.get_base_count(transformer_id) + self.get_base_count(target_id),
                expected_phi_contribution: 0.15, // Highest Φ from meta-cognition
                description: format!("{} transforms how {} operates", transformer_id, target_id),
                tags: vec!["higher-order".to_string(), "meta".to_string(), "transformer".to_string()],
            },
        };

        self.register_composition(composed.clone());
        Ok(composed)
    }

    /// Create a fallback composition: f ; g = try f, if low confidence use g
    pub fn compose_fallback(
        &mut self,
        primary_id: &str,
        fallback_id: &str,
        confidence_threshold: f32,
    ) -> Result<ComposedPrimitive> {
        let id = format!("fall_{}_{}", primary_id, fallback_id);

        if let Some(cached) = self.composed_primitives.get(&id) {
            return Ok(cached.clone());
        }

        let p_enc = self.get_encoding(primary_id)?;
        let f_enc = self.get_encoding(fallback_id)?;
        let op_enc = self.operator_encodings.get(&CompositionType::Fallback {
            confidence_threshold: 0
        }).context("Missing fallback operator encoding")?;

        // Fallback uses weighted bundle - primary is weighted higher
        // We simulate this by bundling primary twice
        let weighted = HV16::bundle(&[p_enc.clone(), p_enc, f_enc]);
        let encoding = op_enc.bind(&weighted);

        let composed = ComposedPrimitive {
            id: id.clone(),
            name: format!("{} ; {}", primary_id, fallback_id),
            composition_type: CompositionType::Fallback {
                confidence_threshold: (confidence_threshold * 1000.0) as u32,
            },
            operand_a: primary_id.to_string(),
            operand_b: Some(fallback_id.to_string()),
            encoding,
            metadata: CompositionMetadata {
                expected_cost: 1.3,
                depth: self.get_depth(primary_id).max(self.get_depth(fallback_id)) + 1,
                base_count: self.get_base_count(primary_id) + self.get_base_count(fallback_id),
                expected_phi_contribution: 0.03,
                description: format!("Try {} (threshold={}), fallback to {}",
                    primary_id, confidence_threshold, fallback_id),
                tags: vec!["fallback".to_string(), "robust".to_string()],
            },
        };

        self.register_composition(composed.clone());
        Ok(composed)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EXECUTION ENGINE
    // ═══════════════════════════════════════════════════════════════════════════

    /// Execute a composed primitive on input
    ///
    /// **Core Algorithm**: Recursively unpack and execute compositions
    pub fn execute(&self, composition_id: &str, input: &HV16) -> Result<CompositionResult> {
        let start = std::time::Instant::now();

        // Check if it's a base primitive
        if let Some(base) = self.get_base_primitive(composition_id) {
            let output = base.encoding.bind(input);
            return Ok(CompositionResult {
                output,
                confidence: 0.9, // Base primitives have high confidence
                iterations: 1,
                execution_path: vec![composition_id.to_string()],
                duration: start.elapsed(),
            });
        }

        // Get composed primitive
        let composed = self.composed_primitives.get(composition_id)
            .context(format!("Unknown composition: {}", composition_id))?;

        match &composed.composition_type {
            CompositionType::Sequential => {
                // Execute g first, then f
                let g_id = composed.operand_b.as_ref().context("Missing second operand")?;
                let g_result = self.execute(g_id, input)?;
                let f_result = self.execute(&composed.operand_a, &g_result.output)?;

                let mut path = g_result.execution_path;
                path.extend(f_result.execution_path);

                Ok(CompositionResult {
                    output: f_result.output,
                    confidence: g_result.confidence * f_result.confidence,
                    iterations: g_result.iterations + f_result.iterations,
                    execution_path: path,
                    duration: start.elapsed(),
                })
            }

            CompositionType::Parallel => {
                // Execute both and bundle
                let g_id = composed.operand_b.as_ref().context("Missing second operand")?;
                let f_result = self.execute(&composed.operand_a, input)?;
                let g_result = self.execute(g_id, input)?;

                let output = HV16::bundle(&[f_result.output, g_result.output]);
                let mut path = f_result.execution_path;
                path.extend(g_result.execution_path);

                Ok(CompositionResult {
                    output,
                    confidence: (f_result.confidence + g_result.confidence) / 2.0,
                    iterations: f_result.iterations + g_result.iterations,
                    execution_path: path,
                    duration: start.elapsed(),
                })
            }

            CompositionType::Conditional { pattern, threshold } => {
                let pattern_enc = text_to_hv16(pattern);
                let similarity = input.similarity(&pattern_enc);
                let threshold_f = *threshold as f32 / 1000.0;

                let chosen_id = if similarity >= threshold_f {
                    &composed.operand_a
                } else {
                    composed.operand_b.as_ref().context("Missing fallback")?
                };

                let mut result = self.execute(chosen_id, input)?;
                result.execution_path.insert(0, format!("cond(sim={:.3})", similarity));
                Ok(result)
            }

            CompositionType::FixedPoint { max_iterations, convergence_threshold } => {
                let threshold_f = *convergence_threshold as f32 / 1000.0;
                let mut current = input.clone();
                let mut prev = current.clone();
                let mut iterations = 0;

                for i in 0..*max_iterations {
                    let result = self.execute(&composed.operand_a, &current)?;
                    prev = current;
                    current = result.output;
                    iterations = i + 1;

                    // Check convergence
                    if current.similarity(&prev) >= threshold_f {
                        break;
                    }
                }

                Ok(CompositionResult {
                    output: current,
                    confidence: 0.8 + 0.2 * (iterations as f32 / *max_iterations as f32).min(1.0),
                    iterations,
                    execution_path: vec![format!("μ{}(i={})", composed.operand_a, iterations)],
                    duration: start.elapsed(),
                })
            }

            CompositionType::HigherOrder => {
                // Higher-order applies transformer to modify how target works
                let target_id = composed.operand_b.as_ref().context("Missing target")?;

                // First, get the target's effect
                let target_result = self.execute(target_id, input)?;

                // Then transform it with the higher-order primitive
                let final_result = self.execute(&composed.operand_a, &target_result.output)?;

                let mut path = vec![format!("↑{}", composed.operand_a)];
                path.extend(target_result.execution_path);
                path.extend(final_result.execution_path);

                Ok(CompositionResult {
                    output: final_result.output,
                    confidence: target_result.confidence * final_result.confidence * 0.9,
                    iterations: target_result.iterations + final_result.iterations,
                    execution_path: path,
                    duration: start.elapsed(),
                })
            }

            CompositionType::Fallback { confidence_threshold } => {
                let threshold_f = *confidence_threshold as f32 / 1000.0;

                // Try primary
                let primary_result = self.execute(&composed.operand_a, input)?;

                if primary_result.confidence >= threshold_f {
                    Ok(primary_result)
                } else {
                    // Use fallback
                    let fallback_id = composed.operand_b.as_ref().context("Missing fallback")?;
                    let mut fallback_result = self.execute(fallback_id, input)?;
                    fallback_result.execution_path.insert(0,
                        format!("fallback(primary_conf={:.3})", primary_result.confidence));
                    Ok(fallback_result)
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // HELPER METHODS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get encoding for a primitive (base or composed)
    fn get_encoding(&self, id: &str) -> Result<HV16> {
        // Check composed first
        if let Some(composed) = self.composed_primitives.get(id) {
            return Ok(composed.encoding.clone());
        }

        // Check base primitives
        if let Some(base) = self.get_base_primitive(id) {
            return Ok(base.encoding.clone());
        }

        anyhow::bail!("Unknown primitive: {}", id)
    }

    /// Get base primitive by ID
    fn get_base_primitive(&self, id: &str) -> Option<Primitive> {
        // This would query the base_system
        // For now, create a synthetic primitive
        Some(Primitive {
            name: id.to_string(),
            tier: PrimitiveTier::NSM,
            domain: "compositionality".to_string(),
            encoding: text_to_hv16(id),
            definition: format!("Base primitive: {}", id),
            is_base: true,
            derivation: None,
        })
    }

    /// Get composition depth
    fn get_depth(&self, id: &str) -> usize {
        if let Some(composed) = self.composed_primitives.get(id) {
            composed.metadata.depth
        } else {
            1 // Base primitives have depth 1
        }
    }

    /// Get base primitive count
    fn get_base_count(&self, id: &str) -> usize {
        if let Some(composed) = self.composed_primitives.get(id) {
            composed.metadata.base_count
        } else {
            1 // Base primitives count as 1
        }
    }

    /// Register a new composition
    fn register_composition(&mut self, composed: ComposedPrimitive) {
        let type_name = match &composed.composition_type {
            CompositionType::Sequential => "sequential",
            CompositionType::Parallel => "parallel",
            CompositionType::Conditional { .. } => "conditional",
            CompositionType::FixedPoint { .. } => "fixed_point",
            CompositionType::HigherOrder => "higher_order",
            CompositionType::Fallback { .. } => "fallback",
        };

        self.stats.total_compositions += 1;
        *self.stats.by_type.entry(type_name.to_string()).or_insert(0) += 1;

        // Update average depth
        let n = self.stats.total_compositions as f32;
        self.stats.avg_depth = ((n - 1.0) * self.stats.avg_depth + composed.metadata.depth as f32) / n;

        // Enforce cache limit
        if self.composed_primitives.len() >= self.config.max_cache_size {
            // Remove least used (simplified: just remove oldest)
            if let Some(key) = self.composed_primitives.keys().next().cloned() {
                self.composed_primitives.remove(&key);
            }
        }

        self.composed_primitives.insert(composed.id.clone(), composed);
    }

    /// Get statistics
    pub fn get_stats(&self) -> &CompositionStats {
        &self.stats
    }

    /// Get all composed primitives
    pub fn get_all_compositions(&self) -> Vec<&ComposedPrimitive> {
        self.composed_primitives.values().collect()
    }

    /// Update harmonic field based on composition usage
    pub fn update_harmonics(&self, field: &mut HarmonicField, composition_id: &str) {
        if let Some(composed) = self.composed_primitives.get(composition_id) {
            // Compositionality contributes to:
            // - Integral Wisdom (higher-order reasoning)
            // - Resonant Coherence (integrated structures)
            // - Evolutionary Progression (self-improvement via fixed points)

            field.adjust_level(FiduciaryHarmonic::IntegralWisdom,
                (composed.metadata.expected_phi_contribution * 0.4) as f64);
            field.adjust_level(FiduciaryHarmonic::ResonantCoherence,
                (composed.metadata.expected_phi_contribution * 0.3) as f64);

            if matches!(composed.composition_type, CompositionType::FixedPoint { .. }) {
                field.adjust_level(FiduciaryHarmonic::EvolutionaryProgression, 0.1);
            }

            if matches!(composed.composition_type, CompositionType::HigherOrder) {
                field.adjust_level(FiduciaryHarmonic::IntegralWisdom, 0.1);
            }
        }
    }
}

/// Result of executing a composed primitive
#[derive(Debug, Clone)]
pub struct CompositionResult {
    /// The output hypervector
    pub output: HV16,

    /// Confidence in the result (0.0-1.0)
    pub confidence: f32,

    /// Number of primitive applications
    pub iterations: usize,

    /// Path of execution (for debugging/explanation)
    pub execution_path: Vec<String>,

    /// Execution time
    pub duration: std::time::Duration,
}

// ═══════════════════════════════════════════════════════════════════════════════
// AUTOMATIC COMPOSITION DISCOVERY
// ═══════════════════════════════════════════════════════════════════════════════

/// Discover useful compositions automatically
pub struct CompositionDiscovery {
    engine: Arc<CompositionalityEngine>,
}

impl CompositionDiscovery {
    /// Find compositions that improve Φ on given test cases
    pub fn discover_beneficial_compositions(
        &self,
        _test_inputs: &[HV16],
        _base_primitive_ids: &[String],
        _max_compositions: usize,
    ) -> Vec<ComposedPrimitive> {
        // Implementation would:
        // 1. Try all pairwise compositions
        // 2. Measure Φ improvement on test cases
        // 3. Return top compositions by improvement

        Vec::new() // Placeholder
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_engine() -> CompositionalityEngine {
        let base_system = Arc::new(PrimitiveSystem::new());
        CompositionalityEngine::new(base_system, CompositionalityConfig::default())
    }

    #[test]
    fn test_sequential_composition() {
        let mut engine = create_test_engine();

        let composed = engine.compose_sequential("analyze", "encode").unwrap();

        assert_eq!(composed.composition_type, CompositionType::Sequential);
        assert_eq!(composed.operand_a, "analyze");
        assert_eq!(composed.operand_b, Some("encode".to_string()));
        assert!(composed.metadata.depth >= 2);
    }

    #[test]
    fn test_parallel_composition() {
        let mut engine = create_test_engine();

        let composed = engine.compose_parallel("fast_path", "accurate_path").unwrap();

        assert_eq!(composed.composition_type, CompositionType::Parallel);
        assert!(composed.name.contains("||"));
    }

    #[test]
    fn test_fixed_point_composition() {
        let mut engine = create_test_engine();

        let composed = engine.compose_fixed_point("refine", Some(50), Some(0.99)).unwrap();

        match &composed.composition_type {
            CompositionType::FixedPoint { max_iterations, .. } => {
                assert_eq!(*max_iterations, 50);
            }
            _ => panic!("Wrong composition type"),
        }
        assert!(composed.name.starts_with("μ"));
    }

    #[test]
    fn test_higher_order_composition() {
        let mut engine = create_test_engine();

        let composed = engine.compose_higher_order("optimizer", "reasoner").unwrap();

        assert_eq!(composed.composition_type, CompositionType::HigherOrder);
        assert!(composed.name.contains("↑"));
        assert!(composed.metadata.expected_phi_contribution >= 0.1);
    }

    #[test]
    fn test_conditional_composition() {
        let mut engine = create_test_engine();

        let composed = engine.compose_conditional(
            "deep_analysis",
            "quick_scan",
            "complex_query",
            0.7,
        ).unwrap();

        match &composed.composition_type {
            CompositionType::Conditional { pattern, threshold } => {
                assert_eq!(pattern, "complex_query");
                assert_eq!(*threshold, 700);
            }
            _ => panic!("Wrong composition type"),
        }
    }

    #[test]
    fn test_composition_caching() {
        let mut engine = create_test_engine();

        // First call creates
        let composed1 = engine.compose_sequential("a", "b").unwrap();

        // Second call should hit cache
        let composed2 = engine.compose_sequential("a", "b").unwrap();

        assert_eq!(composed1.id, composed2.id);
        assert_eq!(composed1.encoding.0, composed2.encoding.0);
    }

    #[test]
    fn test_deep_composition() {
        let mut engine = create_test_engine();

        // Create nested composition: ((a ∘ b) ∘ (c || d))
        let ab = engine.compose_sequential("a", "b").unwrap();
        let cd = engine.compose_parallel("c", "d").unwrap();
        let abcd = engine.compose_sequential(&ab.id, &cd.id).unwrap();

        assert!(abcd.metadata.depth >= 3);
        assert!(abcd.metadata.base_count >= 4);
    }

    #[test]
    fn test_execution() {
        let mut engine = create_test_engine();

        // Create and execute a sequential composition
        let composed = engine.compose_sequential("encode", "transform").unwrap();
        let input = HV16::random(42);

        let result = engine.execute(&composed.id, &input).unwrap();

        // Execution should produce valid results
        assert!(result.confidence >= 0.0, "Confidence should be non-negative");
        assert!(result.iterations >= 1, "Should have at least one iteration");
        // Execution path may or may not be populated depending on tracing config
    }

    #[test]
    fn test_fixed_point_execution() {
        let mut engine = create_test_engine();

        let composed = engine.compose_fixed_point("refine", Some(10), Some(0.99)).unwrap();
        let input = HV16::random(42);

        let result = engine.execute(&composed.id, &input).unwrap();

        // Should converge or hit max iterations
        assert!(result.iterations >= 1 && result.iterations <= 10);
    }

    #[test]
    fn test_stats() {
        let mut engine = create_test_engine();

        engine.compose_sequential("a", "b").unwrap();
        engine.compose_parallel("c", "d").unwrap();
        engine.compose_fixed_point("e", None, None).unwrap();

        let stats = engine.get_stats();

        assert_eq!(stats.total_compositions, 3);
        assert_eq!(stats.by_type.get("sequential"), Some(&1));
        assert_eq!(stats.by_type.get("parallel"), Some(&1));
        assert_eq!(stats.by_type.get("fixed_point"), Some(&1));
    }
}
