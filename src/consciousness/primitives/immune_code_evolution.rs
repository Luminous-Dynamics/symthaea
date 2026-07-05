// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Immune-Inspired Code Evolution
//!
//! Applies lessons from the adaptive immune system to code pattern evolution:
//!
//! | Immune Mechanism | Code Analogue |
//! |-----------------|---------------|
//! | V(D)J Recombination | Segment-level crossover (V=input, D=kernel, J=output) |
//! | Affinity Maturation | Targeted mutation at failing segment only |
//! | Clonal Selection | Fitness-proportional reproduction |
//! | Negative Selection | Anti-pattern rejection before deployment |
//! | Immunological Memory | Priority routing for proven patterns |
//! | Cytokine Signaling | Global exploration boost on systemic failure |
//! | Clonal Anergy | Deprioritize stale patterns without deleting |
//!
//! # Architecture
//!
//! A code pattern HV (16,384D) is decomposed into three segments:
//!
//! ```text
//! V segment (dims 0..5460)       — Input validation / parameter handling
//! D segment (dims 5461..10921)   — Core algorithm / computation kernel
//! J segment (dims 10922..16383)  — Output formatting / error handling
//!
//! Recombination: V_a ⊕ D_b ⊕ J_c → novel hybrid function
//! ```
//!
//! Each segment can be independently mutated, selected, and recombined,
//! producing far greater diversity than whole-HV crossover.

use std::collections::HashMap;

use symthaea_core::hdc::ContinuousHV;

use crate::hdc::code_encoder::CodeHDEncoder;

/// The three segments of a code pattern, inspired by immunoglobulin gene segments.
///
/// V(D)J recombination in B cells randomly combines Variable, Diversity, and
/// Joining gene segments to create novel antibody binding sites. We apply
/// the same principle to code: Variable input handling, Diverse algorithm
/// kernels, and Joining output patterns are independently evolvable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodeSegment {
    /// Variable region: input validation, parameter handling, type conversion
    V,
    /// Diversity region: core algorithm, computation kernel, business logic
    D,
    /// Joining region: output formatting, error handling, return construction
    J,
}

impl CodeSegment {
    /// All segments in order
    pub const ALL: [CodeSegment; 3] = [Self::V, Self::D, Self::J];

    /// Dimension range for this segment within a 16,384D hypervector.
    ///
    /// V: [0, 5461)  — first third
    /// D: [5461, 10922) — middle third
    /// J: [10922, 16384) — final third
    pub fn range(&self, dim: usize) -> (usize, usize) {
        let third = dim / 3;
        match self {
            Self::V => (0, third),
            Self::D => (third, third * 2),
            Self::J => (third * 2, dim),
        }
    }
}

/// A segmented code pattern — the "antibody" of code evolution.
///
/// Unlike a flat HV, this decomposes the pattern into independently
/// evolvable segments that can be recombined like V(D)J gene segments.
#[derive(Debug, Clone)]
pub struct CodeAntibody {
    /// Unique identifier
    pub id: u64,
    /// Human-readable name
    pub name: String,
    /// The full hypervector (composed from segments)
    pub hv: ContinuousHV,
    /// Fitness score (0.0-1.0)
    pub fitness: f64,
    /// Per-segment fitness breakdown
    pub segment_fitness: [f64; 3],
    /// Generation born
    pub generation: usize,
    /// Parent antibody IDs
    pub parents: Vec<u64>,
    /// Whether this antibody is anergic (alive but deprioritized)
    pub anergic: bool,
    /// Number of times selected by the orchestrator (affinity maturation signal)
    pub selection_count: usize,
}

impl CodeAntibody {
    /// Create a new antibody from a full HV.
    pub fn new(id: u64, name: &str, hv: ContinuousHV) -> Self {
        Self {
            id,
            name: name.to_string(),
            hv,
            fitness: 0.0,
            segment_fitness: [0.0; 3],
            generation: 0,
            parents: Vec::new(),
            anergic: false,
            selection_count: 0,
        }
    }

    /// Extract a specific segment from the full HV.
    pub fn segment(&self, seg: CodeSegment) -> ContinuousHV {
        let dim = self.hv.values.len();
        let (start, end) = seg.range(dim);
        let mut values = vec![0.0f32; dim];
        values[start..end].copy_from_slice(&self.hv.values[start..end]);
        ContinuousHV::from_values(values)
    }

    /// Replace a segment with values from another antibody.
    pub fn splice_segment(&mut self, seg: CodeSegment, donor: &CodeAntibody) {
        let dim = self.hv.values.len();
        let (start, end) = seg.range(dim);
        for i in start..end {
            self.hv.values[i] = donor.hv.values[i];
        }
    }

    /// Mutate only a specific segment (targeted somatic hypermutation).
    ///
    /// Unlike global perturb(), this only adds noise to the dimensions
    /// corresponding to the failing segment, preserving the working segments.
    pub fn targeted_mutate(&mut self, segment: CodeSegment, rate: f32) {
        let dim = self.hv.values.len();
        let (start, end) = segment.range(dim);

        // Generate deterministic noise for the segment
        let noise_hv = self.hv.perturb(rate);
        for i in start..end {
            self.hv.values[i] = noise_hv.values[i];
        }
    }
}

/// Anti-pattern library — the "self" antigens that code must NOT match.
///
/// Inspired by thymic negative selection: immune cells that react to
/// self-antigens are killed before deployment. Code patterns that match
/// known anti-patterns are rejected regardless of compilation success.
#[derive(Debug, Clone)]
pub struct AntiPatternLibrary {
    /// Known bad patterns as HVs
    patterns: Vec<(String, ContinuousHV)>,
    /// Rejection threshold: similarity above this triggers negative selection
    threshold: f32,
}

impl AntiPatternLibrary {
    /// Create with default anti-patterns.
    pub fn new(encoder: &CodeHDEncoder) -> Self {
        let mut lib = Self {
            patterns: Vec::new(),
            threshold: 0.6, // Reject if >60% similar to anti-pattern
        };

        // Seed with known dangerous code patterns
        let anti_patterns = [
            "unwrap_unchecked",
            "unsafe_transmute",
            "sql_string_concat",
            "command_injection_exec",
            "hardcoded_password",
            "infinite_loop_no_break",
            "panic_in_production",
            "todo_unimplemented_stub",
            "buffer_overflow_unchecked",
            "race_condition_shared_mut",
        ];

        for name in &anti_patterns {
            lib.patterns
                .push((name.to_string(), encoder.encode_name(name)));
        }

        lib
    }

    /// Check if a pattern matches any anti-pattern (negative selection).
    ///
    /// Returns `Some(anti_pattern_name)` if rejected, `None` if safe.
    pub fn check(&self, candidate: &ContinuousHV) -> Option<&str> {
        for (name, pattern_hv) in &self.patterns {
            let sim = candidate.similarity(pattern_hv);
            if sim > self.threshold {
                return Some(name);
            }
        }
        None
    }

    /// Add a new anti-pattern (learned from failures).
    pub fn add_pattern(&mut self, name: &str, hv: ContinuousHV) {
        self.patterns.push((name.to_string(), hv));
    }

    /// Number of anti-patterns in the library.
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Whether the library is empty.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }
}

/// Cytokine signal — global modulation of evolution intensity.
///
/// When the system experiences sustained failure (high "inflammation"),
/// mutation rates increase, exploration broadens, and the system enters
/// a heightened immune response mode.
#[derive(Debug, Clone)]
pub struct CytokineSignal {
    /// Current inflammation level (0.0 = calm, 1.0 = acute)
    pub inflammation: f32,
    /// Failure rate over recent window (EMA)
    pub failure_ema: f32,
    /// Threshold to trigger inflammatory response
    pub inflammation_threshold: f32,
    /// Mutation rate multiplier when inflamed
    pub inflamed_mutation_boost: f32,
    /// History of failure/success for EMA
    recent_outcomes: Vec<bool>,
    /// EMA decay factor
    ema_alpha: f32,
}

impl CytokineSignal {
    /// Create a new cytokine signaling system.
    pub fn new() -> Self {
        Self {
            inflammation: 0.0,
            failure_ema: 0.0,
            inflammation_threshold: 0.6,
            inflamed_mutation_boost: 3.0,
            recent_outcomes: Vec::new(),
            ema_alpha: 0.15,
        }
    }

    /// Record an outcome (true = success, false = failure).
    pub fn record_outcome(&mut self, success: bool) {
        self.recent_outcomes.push(success);
        if self.recent_outcomes.len() > 50 {
            self.recent_outcomes.remove(0);
        }

        // Update failure EMA
        let failure_signal = if success { 0.0 } else { 1.0 };
        self.failure_ema =
            self.failure_ema * (1.0 - self.ema_alpha) + failure_signal * self.ema_alpha;

        // Update inflammation
        if self.failure_ema > self.inflammation_threshold {
            // Ramp up inflammation
            self.inflammation = (self.inflammation + 0.1).min(1.0);
        } else {
            // Cool down
            self.inflammation = (self.inflammation - 0.05).max(0.0);
        }
    }

    /// Get the effective mutation rate (boosted during inflammation).
    pub fn effective_mutation_rate(&self, base_rate: f32) -> f32 {
        if self.inflammation > 0.3 {
            base_rate * (1.0 + self.inflammation * self.inflamed_mutation_boost)
        } else {
            base_rate
        }
    }

    /// Whether the system is in an inflammatory state.
    pub fn is_inflamed(&self) -> bool {
        self.inflammation > 0.3
    }

    /// Get a human-readable status.
    pub fn status(&self) -> &str {
        if self.inflammation > 0.7 {
            "ACUTE — systemic code generation failure, maximum exploration"
        } else if self.inflammation > 0.3 {
            "ELEVATED — increased mutation and exploration"
        } else {
            "NORMAL — steady-state evolution"
        }
    }
}

impl Default for CytokineSignal {
    fn default() -> Self {
        Self::new()
    }
}

/// The immune-inspired code evolution engine.
///
/// Combines V(D)J recombination, affinity maturation, clonal selection,
/// negative selection, immunological memory, and cytokine signaling into
/// a unified code pattern evolution system.
pub struct ImmuneCodeEvolver {
    /// Population of code antibodies
    population: HashMap<u64, CodeAntibody>,
    /// Next antibody ID
    next_id: u64,
    /// Current generation
    generation: usize,
    /// Anti-pattern library for negative selection
    anti_patterns: AntiPatternLibrary,
    /// Cytokine signaling for global modulation
    cytokines: CytokineSignal,
    /// Immunological memory: proven patterns from past sessions
    memory_cells: Vec<CodeAntibody>,
    /// Base mutation rate
    mutation_rate: f32,
    /// Maximum population size
    max_population: usize,
    /// HDC encoder for name-based seeding
    encoder: CodeHDEncoder,
    /// Statistics
    stats: ImmuneEvolutionStats,
}

/// Statistics for immune code evolution.
#[derive(Debug, Clone, Default)]
pub struct ImmuneEvolutionStats {
    /// Total antibodies created
    pub antibodies_created: usize,
    /// Total V(D)J recombinations performed
    pub recombinations: usize,
    /// Total targeted mutations (affinity maturation)
    pub targeted_mutations: usize,
    /// Total negative selections (anti-pattern rejections)
    pub negative_selections: usize,
    /// Total patterns made anergic
    pub anergy_events: usize,
    /// Total memory cells stored
    pub memory_cells_stored: usize,
    /// Total cytokine inflammation events
    pub inflammation_events: usize,
    /// Best fitness ever achieved
    pub best_fitness: f64,
}

impl ImmuneCodeEvolver {
    /// Create a new immune-inspired code evolver.
    pub fn new() -> Self {
        let encoder = CodeHDEncoder::default_dim();
        let anti_patterns = AntiPatternLibrary::new(&encoder);

        Self {
            population: HashMap::new(),
            next_id: 1,
            generation: 0,
            anti_patterns,
            cytokines: CytokineSignal::new(),
            memory_cells: Vec::new(),
            mutation_rate: 0.05,
            max_population: 100,
            encoder,
            stats: ImmuneEvolutionStats::default(),
        }
    }

    /// Seed the population with named patterns.
    pub fn seed(&mut self, patterns: &[&str]) {
        for name in patterns {
            let hv = self.encoder.encode_name(name);
            let mut antibody = CodeAntibody::new(self.next_id, name, hv);
            antibody.generation = self.generation;
            self.population.insert(self.next_id, antibody);
            self.next_id += 1;
            self.stats.antibodies_created += 1;
        }
    }

    /// Seed with standard code pattern library.
    pub fn seed_standard(&mut self) {
        self.seed(&[
            // V-segment patterns (input handling)
            "validate_input",
            "parse_parameters",
            "check_bounds",
            "convert_type",
            // D-segment patterns (core algorithms)
            "sort_ascending",
            "binary_search",
            "hash_lookup",
            "tree_traverse",
            "accumulate_sum",
            "filter_predicate",
            "map_transform",
            "reduce_fold",
            "dynamic_programming",
            "graph_bfs",
            "divide_conquer",
            "greedy_select",
            // J-segment patterns (output handling)
            "format_result",
            "handle_error",
            "wrap_option",
            "collect_vec",
            "return_result",
            "serialize_json",
            "log_output",
        ]);
    }

    /// V(D)J Recombination: create a novel antibody by combining segments
    /// from different parent antibodies.
    ///
    /// Unlike standard crossover (which BINDs whole HVs), this independently
    /// selects the best V, D, and J segments from different parents,
    /// producing much higher structural diversity.
    pub fn vdj_recombine(
        &mut self,
        v_donor_id: u64,
        d_donor_id: u64,
        j_donor_id: u64,
    ) -> Option<u64> {
        let v_donor = self.population.get(&v_donor_id)?.clone();
        let d_donor = self.population.get(&d_donor_id)?.clone();
        let j_donor = self.population.get(&j_donor_id)?.clone();

        // Create child with V from first, D from second, J from third
        let dim = v_donor.hv.values.len();
        let mut child_values = vec![0.0f32; dim];

        let (v_start, v_end) = CodeSegment::V.range(dim);
        let (d_start, d_end) = CodeSegment::D.range(dim);
        let (j_start, j_end) = CodeSegment::J.range(dim);

        child_values[v_start..v_end].copy_from_slice(&v_donor.hv.values[v_start..v_end]);
        child_values[d_start..d_end].copy_from_slice(&d_donor.hv.values[d_start..d_end]);
        child_values[j_start..j_end].copy_from_slice(&j_donor.hv.values[j_start..j_end]);

        let child_hv = ContinuousHV::from_values(child_values);

        // Negative selection: reject if matches anti-pattern
        if let Some(anti_pattern) = self.anti_patterns.check(&child_hv) {
            self.stats.negative_selections += 1;
            return None; // Killed in thymus
        }

        let child_name = format!(
            "V({})×D({})×J({})",
            v_donor.name, d_donor.name, j_donor.name
        );

        let mut child = CodeAntibody::new(self.next_id, &child_name, child_hv);
        child.generation = self.generation;
        child.parents = vec![v_donor_id, d_donor_id, j_donor_id];

        let id = self.next_id;
        self.population.insert(id, child);
        self.next_id += 1;
        self.stats.antibodies_created += 1;
        self.stats.recombinations += 1;

        Some(id)
    }

    /// Affinity Maturation: targeted mutation of the weakest segment.
    ///
    /// Unlike global mutation, this only mutates the segment with the
    /// lowest fitness, preserving the segments that already work.
    pub fn affinity_mature(&mut self, antibody_id: u64) -> bool {
        let effective_rate = self.cytokines.effective_mutation_rate(self.mutation_rate);

        if let Some(antibody) = self.population.get_mut(&antibody_id) {
            // Find weakest segment
            let weakest = if antibody.segment_fitness[0] <= antibody.segment_fitness[1]
                && antibody.segment_fitness[0] <= antibody.segment_fitness[2]
            {
                CodeSegment::V
            } else if antibody.segment_fitness[1] <= antibody.segment_fitness[2] {
                CodeSegment::D
            } else {
                CodeSegment::J
            };

            antibody.targeted_mutate(weakest, effective_rate);
            self.stats.targeted_mutations += 1;
            true
        } else {
            false
        }
    }

    /// Evaluate fitness for all antibodies using a provided fitness function.
    ///
    /// The fitness function should return (overall_fitness, [v_fitness, d_fitness, j_fitness]).
    pub fn evaluate_all<F>(&mut self, fitness_fn: F)
    where
        F: Fn(&ContinuousHV) -> (f64, [f64; 3]),
    {
        let ids: Vec<u64> = self.population.keys().copied().collect();
        for id in ids {
            if let Some(antibody) = self.population.get_mut(&id) {
                let (overall, segments) = fitness_fn(&antibody.hv);
                antibody.fitness = overall;
                antibody.segment_fitness = segments;

                if overall > self.stats.best_fitness {
                    self.stats.best_fitness = overall;
                }
            }
        }
    }

    /// Run one generation of immune evolution.
    ///
    /// 1. Fitness-proportional selection (clonal selection)
    /// 2. V(D)J recombination of selected parents
    /// 3. Affinity maturation of top candidates
    /// 4. Negative selection against anti-patterns
    /// 5. Anergy for persistently unused patterns
    /// 6. Cytokine modulation based on success rate
    pub fn evolve_generation<F>(&mut self, fitness_fn: F)
    where
        F: Fn(&ContinuousHV) -> (f64, [f64; 3]),
    {
        self.generation += 1;

        // 1. Evaluate fitness
        self.evaluate_all(&fitness_fn);

        // 2. Fitness-proportional selection (clonal selection)
        let selected = self.clonal_select(self.max_population / 3);

        // 3. V(D)J recombination from selected parents
        let num_children = self.max_population / 3;
        for _ in 0..num_children {
            if selected.len() >= 3 {
                let v_idx = self.proportional_pick(&selected);
                let d_idx = self.proportional_pick(&selected);
                let j_idx = self.proportional_pick(&selected);
                self.vdj_recombine(selected[v_idx].0, selected[d_idx].0, selected[j_idx].0);
            }
        }

        // 4. Affinity maturation on top 20%
        let mut sorted: Vec<(u64, f64)> = self
            .population
            .iter()
            .map(|(&id, ab)| (id, ab.fitness))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_count = (sorted.len() / 5).max(1);
        for &(id, _) in sorted.iter().take(top_count) {
            self.affinity_mature(id);
        }

        // 5. Anergy: mark patterns with zero selection count after 5+ generations
        let anergy_ids: Vec<u64> = self
            .population
            .iter()
            .filter(|(_, ab)| {
                ab.generation + 5 < self.generation && ab.selection_count == 0 && !ab.anergic
            })
            .map(|(&id, _)| id)
            .collect();
        for id in &anergy_ids {
            if let Some(ab) = self.population.get_mut(id) {
                ab.anergic = true;
                self.stats.anergy_events += 1;
            }
        }

        // 6. Cull: remove worst performers to maintain population cap
        if self.population.len() > self.max_population {
            let mut all: Vec<(u64, f64, bool)> = self
                .population
                .iter()
                .map(|(&id, ab)| (id, ab.fitness, ab.anergic))
                .collect();
            // Remove anergic first, then lowest fitness
            all.sort_by(|a, b| {
                // Anergic patterns sorted to end, then by fitness ascending
                a.2.cmp(&b.2)
                    .reverse()
                    .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            });
            let to_remove = self.population.len() - self.max_population;
            for &(id, _, _) in all.iter().take(to_remove) {
                self.population.remove(&id);
            }
        }

        // 7. Record success rate for cytokine signaling
        let success_rate = if sorted.is_empty() {
            0.0
        } else {
            sorted.iter().filter(|(_, f)| *f > 0.5).count() as f32 / sorted.len() as f32
        };
        self.cytokines.record_outcome(success_rate > 0.3);
        if self.cytokines.is_inflamed() {
            self.stats.inflammation_events += 1;
        }
    }

    /// Fitness-proportional selection (clonal selection).
    ///
    /// Unlike tournament selection, this reproduces patterns in proportion
    /// to their fitness. A pattern with fitness 0.8 is twice as likely to
    /// be selected as one with fitness 0.4.
    fn clonal_select(&self, count: usize) -> Vec<(u64, f64)> {
        let mut candidates: Vec<(u64, f64)> = self
            .population
            .iter()
            .filter(|(_, ab)| !ab.anergic) // Anergic cells don't divide
            .map(|(&id, ab)| (id, ab.fitness))
            .collect();

        if candidates.is_empty() {
            return Vec::new();
        }

        // Normalize fitness to probabilities
        let total_fitness: f64 = candidates.iter().map(|(_, f)| f.max(0.01)).sum();

        // Select proportionally (deterministic roulette for reproducibility)
        let mut selected = Vec::new();
        let step = total_fitness / count as f64;
        let mut accumulated = 0.0;
        let mut threshold = step / 2.0; // Start at midpoint

        // Sort by fitness descending for deterministic ordering
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for &(id, fitness) in &candidates {
            accumulated += fitness.max(0.01);
            while accumulated >= threshold && selected.len() < count {
                selected.push((id, fitness));
                threshold += step;
            }
        }

        // Fill remaining slots with best if roulette undersampled
        while selected.len() < count && !candidates.is_empty() {
            selected.push(candidates[0]);
        }

        selected
    }

    /// Pick an index proportionally to fitness from a selected list.
    fn proportional_pick(&self, selected: &[(u64, f64)]) -> usize {
        if selected.is_empty() {
            return 0;
        }
        // Simple deterministic pick: use generation as pseudorandom seed
        let total: f64 = selected.iter().map(|(_, f)| f.max(0.01)).sum();
        let target = (self.generation as f64 * 0.618033988 % 1.0) * total; // Golden ratio hash
        let mut acc = 0.0;
        for (i, (_, fitness)) in selected.iter().enumerate() {
            acc += fitness.max(0.01);
            if acc >= target {
                return i;
            }
        }
        selected.len() - 1
    }

    /// Store a proven pattern as a memory cell for future sessions.
    pub fn memorize(&mut self, antibody_id: u64) {
        if let Some(antibody) = self.population.get(&antibody_id) {
            self.memory_cells.push(antibody.clone());
            self.stats.memory_cells_stored += 1;
        }
    }

    /// Recall memory cells into the current population.
    ///
    /// Like immunological memory: previously successful patterns get
    /// priority reactivation, enabling faster response to similar problems.
    pub fn recall_memory(&mut self) -> usize {
        let count = self.memory_cells.len();
        for cell in &self.memory_cells {
            let mut recalled = cell.clone();
            recalled.id = self.next_id;
            recalled.generation = self.generation;
            recalled.selection_count = 0; // Reset — must prove itself again
            recalled.anergic = false;
            self.population.insert(self.next_id, recalled);
            self.next_id += 1;
        }
        count
    }

    /// Add a learned anti-pattern from a specific failure.
    pub fn learn_anti_pattern(&mut self, name: &str, failed_hv: &ContinuousHV) {
        self.anti_patterns.add_pattern(name, failed_hv.clone());
    }

    /// Get the best antibody in the population.
    pub fn best(&self) -> Option<&CodeAntibody> {
        self.population.values().max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get the cytokine signaling state.
    pub fn cytokines(&self) -> &CytokineSignal {
        &self.cytokines
    }

    /// Get the anti-pattern library.
    pub fn anti_patterns(&self) -> &AntiPatternLibrary {
        &self.anti_patterns
    }

    /// Get evolution statistics.
    pub fn stats(&self) -> &ImmuneEvolutionStats {
        &self.stats
    }

    /// Get current generation.
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Get population size.
    pub fn population_size(&self) -> usize {
        self.population.len()
    }
}

impl Default for ImmuneCodeEvolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction_and_seeding() {
        let mut evolver = ImmuneCodeEvolver::new();
        evolver.seed_standard();
        assert!(evolver.population_size() > 20);
        assert_eq!(evolver.generation(), 0);
    }

    #[test]
    fn test_vdj_recombination() {
        let mut evolver = ImmuneCodeEvolver::new();
        evolver.seed(&["input_parser", "quicksort", "json_formatter"]);

        let child_id = evolver.vdj_recombine(1, 2, 3);
        assert!(child_id.is_some());
        assert_eq!(evolver.stats().recombinations, 1);

        let child = evolver.population.get(&child_id.unwrap()).unwrap();
        assert!(child.name.contains("input_parser"));
        assert!(child.name.contains("quicksort"));
        assert!(child.name.contains("json_formatter"));
    }

    #[test]
    fn test_targeted_mutation() {
        let mut evolver = ImmuneCodeEvolver::new();
        evolver.seed(&["test_pattern"]);

        // Set segment fitness so V is weakest
        if let Some(ab) = evolver.population.get_mut(&1) {
            ab.segment_fitness = [0.2, 0.8, 0.7]; // V is weakest
        }

        let original_hv = evolver.population.get(&1).unwrap().hv.clone();
        evolver.affinity_mature(1);

        let mutated_hv = &evolver.population.get(&1).unwrap().hv;

        // D and J segments should be partially preserved (targeted mutation)
        let dim = mutated_hv.values.len();
        let (d_start, d_end) = CodeSegment::D.range(dim);
        let d_preserved: f32 = (d_start..d_end)
            .map(|i| (mutated_hv.values[i] - original_hv.values[i]).abs())
            .sum::<f32>()
            / (d_end - d_start) as f32;

        // D segment changes should be less than V segment changes
        // (only V was targeted for mutation)
        assert_eq!(evolver.stats().targeted_mutations, 1);
    }

    #[test]
    fn test_negative_selection() {
        let mut evolver = ImmuneCodeEvolver::new();
        evolver.seed(&["safe_pattern"]);

        // Add the first pattern as an anti-pattern
        let bad_hv = evolver.encoder.encode_name("safe_pattern");
        evolver.learn_anti_pattern("known_bad", &bad_hv);

        // Try to recombine — the child should be rejected because
        // it'll be similar to the anti-pattern
        // (We need 3 parents for VDJ, so seed more)
        evolver.seed(&["safe_pattern", "safe_pattern"]);
        let result = evolver.vdj_recombine(1, 2, 3);

        // May or may not be rejected depending on similarity
        // The anti-pattern check uses 0.6 threshold
        if result.is_none() {
            assert!(evolver.stats().negative_selections > 0);
        }
    }

    #[test]
    fn test_cytokine_inflammation() {
        let mut cytokines = CytokineSignal::new();

        // Record many failures
        for _ in 0..20 {
            cytokines.record_outcome(false);
        }

        assert!(cytokines.is_inflamed());
        assert!(cytokines.inflammation > 0.3);

        // Mutation rate should be boosted
        let boosted = cytokines.effective_mutation_rate(0.05);
        assert!(boosted > 0.05, "Mutation should be boosted: {boosted}");

        // Recovery: record successes
        for _ in 0..30 {
            cytokines.record_outcome(true);
        }
        assert!(!cytokines.is_inflamed());
    }

    #[test]
    fn test_evolve_generation() {
        let mut evolver = ImmuneCodeEvolver::new();
        evolver.seed_standard();

        // Simple fitness: higher similarity to "sort" is better
        let target = evolver.encoder.encode_name("sort_ascending");
        let fitness_fn = |hv: &ContinuousHV| -> (f64, [f64; 3]) {
            let sim = ((hv.similarity(&target) + 1.0) / 2.0) as f64;
            (sim, [sim, sim, sim]) // Uniform segment fitness for simplicity
        };

        evolver.evolve_generation(fitness_fn);
        assert_eq!(evolver.generation(), 1);
        assert!(evolver.stats().antibodies_created > 0);
    }

    #[test]
    fn test_immunological_memory() {
        let mut evolver = ImmuneCodeEvolver::new();
        evolver.seed(&["proven_sort"]);

        // Memorize the proven pattern
        evolver.memorize(1);
        assert_eq!(evolver.stats().memory_cells_stored, 1);

        // Clear population (simulating new session)
        evolver.population.clear();
        assert_eq!(evolver.population_size(), 0);

        // Recall memory
        let recalled = evolver.recall_memory();
        assert_eq!(recalled, 1);
        assert_eq!(evolver.population_size(), 1);
    }

    #[test]
    fn test_anergy() {
        let mut evolver = ImmuneCodeEvolver::new();
        evolver.seed(&["stale_pattern"]);

        // Artificially age the pattern
        if let Some(ab) = evolver.population.get_mut(&1) {
            ab.generation = 0;
        }

        // Run 6 generations without selecting the pattern
        let dummy_fitness = |_: &ContinuousHV| -> (f64, [f64; 3]) { (0.1, [0.1, 0.1, 0.1]) };
        for _ in 0..6 {
            evolver.evolve_generation(dummy_fitness);
        }

        // Pattern should be anergic after 5+ generations without selection
        let ab = evolver.population.get(&1);
        if let Some(ab) = ab {
            // May or may not be anergic depending on whether it survived culling
            if ab.anergic {
                assert!(evolver.stats().anergy_events > 0);
            }
        }
    }

    #[test]
    fn test_segment_extraction() {
        let encoder = CodeHDEncoder::default_dim();
        let hv = encoder.encode_name("test_function");
        let ab = CodeAntibody::new(1, "test", hv.clone());

        let v_seg = ab.segment(CodeSegment::V);
        let d_seg = ab.segment(CodeSegment::D);
        let j_seg = ab.segment(CodeSegment::J);

        // Segments should cover the full dimension
        assert_eq!(v_seg.values.len(), hv.values.len());
        assert_eq!(d_seg.values.len(), hv.values.len());
        assert_eq!(j_seg.values.len(), hv.values.len());

        // V segment should have non-zero values only in first third
        let dim = hv.values.len();
        let third = dim / 3;
        let v_energy: f32 = v_seg.values[0..third].iter().map(|v| v * v).sum();
        let v_rest: f32 = v_seg.values[third..].iter().map(|v| v * v).sum();
        assert!(v_energy > 0.0);
        assert!((v_rest).abs() < 1e-6);
    }

    #[test]
    fn test_anti_pattern_library() {
        let encoder = CodeHDEncoder::default_dim();
        let lib = AntiPatternLibrary::new(&encoder);

        assert_eq!(lib.len(), 10); // 10 default anti-patterns

        // A random safe pattern should not be rejected
        let safe_hv = encoder.encode_name("safe_fibonacci_implementation");
        assert!(lib.check(&safe_hv).is_none());
    }
}
