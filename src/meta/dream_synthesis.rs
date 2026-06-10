// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dream Synthesis — Background creative code recombination
//!
//! During idle periods, randomly recombine code patterns in HDC space to
//! discover interesting compositions. Like biological dreams consolidating
//! memories, this explores the space of possible code combinations.
//!
//! # Dream Cycle
//!
//! 1. Pick random pairs of functions/types from memory
//! 2. Compose them in HDC space (bundle operation)
//! 3. Measure coherence of the composition
//! 4. If coherent, record as a "dream insight"

use symthaea_core::hdc::ContinuousHV;

use crate::hdc::code_memory::CodebaseMemory;

/// Minimum coherence for a dream composition to be recorded
const DREAM_THRESHOLD: f32 = 0.2;

/// Default number of dream iterations per cycle
const DEFAULT_DREAM_ITERATIONS: usize = 10;

/// An insight discovered during dream synthesis
#[derive(Debug, Clone)]
pub struct DreamInsight {
    /// Source entity A name
    pub source_a: String,
    /// Source entity B name
    pub source_b: String,
    /// The composed hypervector
    pub composed_hv: ContinuousHV,
    /// Coherence score of the composition
    pub coherence: f32,
    /// Description of the insight
    pub description: String,
}

/// Result of a complete dream cycle
#[derive(Debug, Clone)]
pub struct DreamCycleResult {
    /// Insights discovered
    pub insights: Vec<DreamInsight>,
    /// Total compositions attempted
    pub attempts: usize,
    /// Number that passed the coherence threshold
    pub discoveries: usize,
}

/// Background creative code recombination engine
pub struct DreamSynthesizer {
    /// Coherence threshold for recording insights
    threshold: f32,
    /// Number of iterations per dream cycle
    iterations: usize,
    /// Simple deterministic counter for pseudo-random selection
    counter: u64,
}

impl DreamSynthesizer {
    /// Create a new dream synthesizer
    pub fn new() -> Self {
        Self {
            threshold: DREAM_THRESHOLD,
            iterations: DEFAULT_DREAM_ITERATIONS,
            counter: 0,
        }
    }

    /// Set the coherence threshold
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the number of iterations per cycle
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Run one dream cycle: compose random code pattern pairs
    pub fn dream_cycle(&mut self, memory: &CodebaseMemory) -> DreamCycleResult {
        let func_hvs = memory.all_function_hvs();
        let type_hvs = memory.all_type_hvs();

        // Combine all HVs into one pool for dreaming
        let mut pool: Vec<(&str, &ContinuousHV)> = Vec::new();

        // We don't have names from the HV references alone, so use indices
        for (_i, hv) in func_hvs.iter().enumerate() {
            pool.push(("function", hv));
        }
        for (_i, hv) in type_hvs.iter().enumerate() {
            pool.push(("type", hv));
        }

        if pool.len() < 2 {
            return DreamCycleResult {
                insights: Vec::new(),
                attempts: 0,
                discoveries: 0,
            };
        }

        let mut insights = Vec::new();
        let pool_len = pool.len();

        for iter in 0..self.iterations {
            // Deterministic "random" selection
            self.counter = self.counter.wrapping_add(1);
            let idx_a = (self.counter.wrapping_mul(2_654_435_761) as usize) % pool_len;
            let idx_b = (self
                .counter
                .wrapping_mul(1_618_033_989)
                .wrapping_add(iter as u64) as usize)
                % pool_len;

            if idx_a == idx_b {
                continue;
            }

            let (kind_a, hv_a) = pool[idx_a];
            let (kind_b, hv_b) = pool[idx_b];

            // Compose in HDC space
            let composed = ContinuousHV::bundle_owned(&[hv_a.clone(), hv_b.clone()]);

            // Measure coherence: how similar is the composition to both parents?
            let sim_a = composed.similarity(hv_a);
            let sim_b = composed.similarity(hv_b);
            let coherence = (sim_a + sim_b) / 2.0;

            if coherence >= self.threshold {
                insights.push(DreamInsight {
                    source_a: format!("{}_{}", kind_a, idx_a),
                    source_b: format!("{}_{}", kind_b, idx_b),
                    composed_hv: composed,
                    coherence,
                    description: format!(
                        "Composition of {kind_a}_{idx_a} and {kind_b}_{idx_b} (coherence: {coherence:.3})"
                    ),
                });
            }
        }

        let discoveries = insights.len();
        DreamCycleResult {
            insights,
            attempts: self.iterations,
            discoveries,
        }
    }

    /// Run multiple dream cycles and return the best insights
    pub fn dream_session(&mut self, memory: &CodebaseMemory, cycles: usize) -> Vec<DreamInsight> {
        let mut all_insights = Vec::new();

        for _ in 0..cycles {
            let result = self.dream_cycle(memory);
            all_insights.extend(result.insights);
        }

        // Sort by coherence descending and deduplicate
        all_insights.sort_by(|a, b| {
            b.coherence
                .partial_cmp(&a.coherence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        all_insights
    }
}

impl Default for DreamSynthesizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::code_encoder::CodeHDEncoder;
    use crate::language::code_parser::{Entity, EntityKind, ParsedCode, Span};
    use std::path::Path;

    fn test_span() -> Span {
        Span {
            start_byte: 0,
            end_byte: 10,
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 10,
        }
    }

    fn make_memory() -> CodebaseMemory {
        let mut memory = CodebaseMemory::new(CodeHDEncoder::new(512));

        let mut parsed = ParsedCode::new("", "rust");
        parsed
            .entities
            .push(Entity::new(EntityKind::Function, "sort", test_span()));
        parsed
            .entities
            .push(Entity::new(EntityKind::Function, "filter", test_span()));
        parsed
            .entities
            .push(Entity::new(EntityKind::Function, "map", test_span()));
        parsed
            .entities
            .push(Entity::new(EntityKind::Struct, "Config", test_span()));
        parsed
            .entities
            .push(Entity::new(EntityKind::Struct, "Parser", test_span()));
        memory.index_file(Path::new("src/lib.rs"), &parsed);

        memory
    }

    #[test]
    fn test_dream_cycle() {
        let memory = make_memory();
        let mut dreamer = DreamSynthesizer::new().with_iterations(20);

        let result = dreamer.dream_cycle(&memory);
        assert_eq!(result.attempts, 20);
        // Should find some coherent compositions
    }

    #[test]
    fn test_dream_session() {
        let memory = make_memory();
        let mut dreamer = DreamSynthesizer::new()
            .with_iterations(10)
            .with_threshold(0.1); // lower threshold for testing

        let insights = dreamer.dream_session(&memory, 3);
        // Insights should be sorted by coherence
        for i in 1..insights.len() {
            assert!(insights[i - 1].coherence >= insights[i].coherence);
        }
    }

    #[test]
    fn test_empty_memory_dream() {
        let memory = CodebaseMemory::new(CodeHDEncoder::new(512));
        let mut dreamer = DreamSynthesizer::new();

        let result = dreamer.dream_cycle(&memory);
        assert_eq!(result.attempts, 0);
        assert_eq!(result.discoveries, 0);
    }

    #[test]
    fn test_dream_deterministic() {
        let memory = make_memory();

        let mut dreamer1 = DreamSynthesizer::new().with_iterations(5);
        let mut dreamer2 = DreamSynthesizer::new().with_iterations(5);

        let result1 = dreamer1.dream_cycle(&memory);
        let result2 = dreamer2.dream_cycle(&memory);

        // Same starting state should produce same results
        assert_eq!(result1.attempts, result2.attempts);
        assert_eq!(result1.discoveries, result2.discoveries);
    }
}
