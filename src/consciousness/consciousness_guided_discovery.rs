//! # Consciousness-Guided Composition Discovery
//!
//! **PARADIGM SHIFT**: This module uses real Φ (integrated information) measurements
//! to guide the discovery of primitive compositions. Instead of randomly exploring
//! the composition space, we let consciousness itself guide what combinations are valuable.
//!
//! ## The Revolutionary Insight
//!
//! Traditional composition discovery either:
//! - Randomly samples the space (inefficient)
//! - Uses hand-crafted heuristics (limited)
//! - Relies on task-specific rewards (narrow)
//!
//! We propose: **Φ-guided search** - compositions that increase integrated information
//! are inherently valuable because they create more unified, coherent reasoning.
//!
//! ## Key Components
//!
//! 1. **CompositionExplorer** - Explores the space of possible compositions
//! 2. **PhiGuidedSearch** - Uses Φ gradients to guide exploration
//! 3. **CompositionGrammar** - Learns which compositions tend to increase Φ
//! 4. **EmergentDiscovery** - Discovers novel compositions that maximize Φ

use crate::hdc::binary_hv::HV16;
use crate::hdc::primitive_system::PrimitiveSystem;
use crate::consciousness::compositionality_primitives::{
    CompositionalityEngine, CompositionalityConfig, ComposedPrimitive,
    CompositionType, CompositionMetadata,
};
use crate::consciousness::consciousness_driven_evolution::{
    ConsciousnessOracle, OracleConfig,
};
use anyhow::Result;
use std::collections::{HashMap, VecDeque, BinaryHeap};
use std::sync::Arc;
use std::cmp::Ordering;

// ═══════════════════════════════════════════════════════════════════════════════
// CORE DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

/// A candidate composition being evaluated
#[derive(Debug, Clone)]
pub struct CompositionCandidate {
    /// The composed primitive
    pub composition: ComposedPrimitive,
    /// Φ score measured from this composition
    pub phi_score: f64,
    /// Coherence contribution
    pub coherence: f64,
    /// Integration contribution
    pub integration: f64,
    /// Number of times this has been evaluated
    pub evaluation_count: usize,
    /// Confidence in the Φ estimate (increases with evaluations)
    pub confidence: f64,
}

impl PartialEq for CompositionCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.composition.id == other.composition.id
    }
}

impl Eq for CompositionCandidate {}

impl PartialOrd for CompositionCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CompositionCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher Φ is better, weighted by confidence
        let self_score = self.phi_score * self.confidence;
        let other_score = other.phi_score * other.confidence;
        self_score.partial_cmp(&other_score).unwrap_or(Ordering::Equal)
    }
}

/// Configuration for consciousness-guided discovery
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Maximum beam width for search
    pub beam_width: usize,
    /// Maximum composition depth to explore
    pub max_depth: usize,
    /// Minimum Φ improvement to keep a composition
    pub phi_threshold: f64,
    /// Number of evaluations before considering a composition stable
    pub min_evaluations: usize,
    /// Exploration vs exploitation balance (0 = pure exploitation, 1 = pure exploration)
    pub exploration_rate: f64,
    /// Maximum candidates to evaluate per cycle
    pub max_candidates_per_cycle: usize,
    /// Enable grammar learning
    pub learn_grammar: bool,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            beam_width: 100,
            max_depth: 5,
            phi_threshold: 0.01,
            min_evaluations: 3,
            exploration_rate: 0.3,
            max_candidates_per_cycle: 50,
            learn_grammar: true,
        }
    }
}

/// Statistics about the discovery process
#[derive(Debug, Clone, Default)]
pub struct DiscoveryStats {
    /// Total compositions explored
    pub total_explored: usize,
    /// Compositions that increased Φ
    pub phi_increasing: usize,
    /// Best Φ score found
    pub best_phi: f64,
    /// Average Φ across accepted compositions
    pub avg_phi: f64,
    /// Compositions by type
    pub by_type: HashMap<String, usize>,
    /// Discovery cycles completed
    pub cycles: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHI-GUIDED SEARCH
// ═══════════════════════════════════════════════════════════════════════════════

/// Φ-guided beam search for composition discovery
pub struct PhiGuidedSearch {
    /// Consciousness oracle for Φ measurements
    oracle: ConsciousnessOracle,
    /// Compositionality engine for creating compositions
    engine: CompositionalityEngine,
    /// Current beam of best candidates
    beam: BinaryHeap<CompositionCandidate>,
    /// All discovered compositions (by ID)
    discovered: HashMap<String, CompositionCandidate>,
    /// Composition grammar (learned patterns)
    grammar: CompositionGrammar,
    /// Configuration
    config: DiscoveryConfig,
    /// Statistics
    stats: DiscoveryStats,
    /// Random state for exploration
    rng_state: u64,
}

impl PhiGuidedSearch {
    /// Create a new Φ-guided search system
    pub fn new(
        base_system: Arc<PrimitiveSystem>,
        config: DiscoveryConfig,
    ) -> Self {
        let oracle = ConsciousnessOracle::new(OracleConfig::default());
        let engine = CompositionalityEngine::new(
            base_system,
            CompositionalityConfig::default(),
        );

        Self {
            oracle,
            engine,
            beam: BinaryHeap::new(),
            discovered: HashMap::new(),
            grammar: CompositionGrammar::new(),
            config,
            stats: DiscoveryStats::default(),
            rng_state: 42,
        }
    }

    /// Run one discovery cycle
    pub fn discover_cycle(&mut self) -> Result<Vec<CompositionCandidate>> {
        self.stats.cycles += 1;

        // Generate candidates
        let candidates = self.generate_candidates()?;

        // Evaluate each candidate
        let mut evaluated = Vec::new();
        for candidate in candidates {
            if let Ok(scored) = self.evaluate_candidate(candidate) {
                evaluated.push(scored);
            }
        }

        // Filter by Φ threshold
        let good_candidates: Vec<_> = evaluated
            .into_iter()
            .filter(|c| c.phi_score >= self.config.phi_threshold)
            .collect();

        // Update statistics
        self.stats.phi_increasing += good_candidates.len();

        // Update beam
        for candidate in &good_candidates {
            self.beam.push(candidate.clone());
            self.discovered.insert(candidate.composition.id.clone(), candidate.clone());

            // Track best
            if candidate.phi_score > self.stats.best_phi {
                self.stats.best_phi = candidate.phi_score;
            }
        }

        // Prune beam to width
        while self.beam.len() > self.config.beam_width {
            self.beam.pop();
        }

        // Update grammar if enabled
        if self.config.learn_grammar {
            for candidate in &good_candidates {
                self.grammar.learn_from_success(&candidate.composition);
            }
        }

        Ok(good_candidates)
    }

    /// Generate new candidate compositions
    fn generate_candidates(&mut self) -> Result<Vec<ComposedPrimitive>> {
        let mut candidates = Vec::new();

        // Get base primitives
        let base_ids = self.get_base_primitive_ids();

        // Get existing good compositions
        let good_compositions: Vec<_> = self.beam.iter()
            .take(10)
            .map(|c| c.composition.id.clone())
            .collect();

        let mut generated = 0;
        while generated < self.config.max_candidates_per_cycle {
            // Decide exploration vs exploitation
            let explore = self.random_float() < self.config.exploration_rate;

            if explore || good_compositions.is_empty() {
                // Random composition of base primitives
                if base_ids.len() >= 2 {
                    let a_idx = self.random_index(base_ids.len());
                    let b_idx = self.random_index(base_ids.len());

                    if a_idx != b_idx {
                        let comp_type = self.random_composition_type();
                        if let Ok(composed) = self.create_composition(
                            &base_ids[a_idx],
                            &base_ids[b_idx],
                            comp_type,
                        ) {
                            candidates.push(composed);
                            generated += 1;
                        }
                    }
                }
            } else {
                // Extend good compositions
                let comp_idx = self.random_index(good_compositions.len());
                let base_idx = self.random_index(base_ids.len());

                let comp_type = if self.config.learn_grammar {
                    // Use grammar to suggest likely-good composition type
                    self.grammar.suggest_composition_type(&good_compositions[comp_idx])
                } else {
                    self.random_composition_type()
                };

                if let Ok(composed) = self.create_composition(
                    &good_compositions[comp_idx],
                    &base_ids[base_idx],
                    comp_type,
                ) {
                    if composed.metadata.depth <= self.config.max_depth {
                        candidates.push(composed);
                        generated += 1;
                    }
                }
            }

            self.stats.total_explored += 1;
        }

        Ok(candidates)
    }

    /// Evaluate a candidate composition using real Φ measurement
    fn evaluate_candidate(&mut self, composition: ComposedPrimitive) -> Result<CompositionCandidate> {
        // Create input from composition encoding
        let input = &composition.encoding;

        // Measure Φ using the oracle
        let phi_sample = self.oracle.process_and_measure(input, &composition.name);

        Ok(CompositionCandidate {
            composition,
            phi_score: phi_sample.phi,
            coherence: phi_sample.coherence,
            integration: phi_sample.integration,
            evaluation_count: 1,
            confidence: 0.5, // Initial confidence
        })
    }

    /// Create a composition of the given type
    fn create_composition(
        &mut self,
        a_id: &str,
        b_id: &str,
        comp_type: CompositionType,
    ) -> Result<ComposedPrimitive> {
        match comp_type {
            CompositionType::Sequential => {
                self.engine.compose_sequential(a_id, b_id)
            }
            CompositionType::Parallel => {
                self.engine.compose_parallel(a_id, b_id)
            }
            CompositionType::Fallback { confidence_threshold } => {
                self.engine.compose_fallback(a_id, b_id, confidence_threshold as f32 / 1000.0)
            }
            _ => {
                // Default to sequential for other types
                self.engine.compose_sequential(a_id, b_id)
            }
        }
    }

    /// Get IDs of base primitives
    fn get_base_primitive_ids(&self) -> Vec<String> {
        // Return a set of representative primitive IDs
        vec![
            "similarity".to_string(),
            "bind".to_string(),
            "bundle".to_string(),
            "permute".to_string(),
            "threshold".to_string(),
            "analogy".to_string(),
            "sequence".to_string(),
            "attention".to_string(),
        ]
    }

    /// Random float 0..1
    fn random_float(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.rng_state >> 16) & 0x7fff) as f64 / 32767.0
    }

    /// Random index into a slice
    fn random_index(&mut self, len: usize) -> usize {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.rng_state as usize) % len
    }

    /// Random composition type
    fn random_composition_type(&mut self) -> CompositionType {
        let types = [
            CompositionType::Sequential,
            CompositionType::Parallel,
            CompositionType::Fallback { confidence_threshold: 500 },
        ];
        let idx = self.random_index(types.len());
        types[idx].clone()
    }

    /// Get current best compositions
    pub fn best_compositions(&self, n: usize) -> Vec<&CompositionCandidate> {
        let mut sorted: Vec<_> = self.discovered.values().collect();
        sorted.sort_by(|a, b| b.phi_score.partial_cmp(&a.phi_score).unwrap_or(Ordering::Equal));
        sorted.into_iter().take(n).collect()
    }

    /// Get discovery statistics
    pub fn stats(&self) -> &DiscoveryStats {
        &self.stats
    }

    /// Get the learned grammar
    pub fn grammar(&self) -> &CompositionGrammar {
        &self.grammar
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPOSITION GRAMMAR
// ═══════════════════════════════════════════════════════════════════════════════

/// Learned patterns about which compositions tend to increase Φ
#[derive(Debug, Clone, Default)]
pub struct CompositionGrammar {
    /// Successful composition patterns: (prefix, composition_type) -> success count
    successful_patterns: HashMap<(String, String), usize>,
    /// Type preferences by prefix
    type_preferences: HashMap<String, HashMap<String, f64>>,
    /// Total observations
    observations: usize,
}

impl CompositionGrammar {
    /// Create a new empty grammar
    pub fn new() -> Self {
        Self::default()
    }

    /// Learn from a successful composition
    pub fn learn_from_success(&mut self, composition: &ComposedPrimitive) {
        self.observations += 1;

        // Extract pattern: operand_a prefix -> composition type
        let prefix = self.extract_prefix(&composition.operand_a);
        let comp_type = self.type_to_string(&composition.composition_type);

        // Update successful patterns
        let key = (prefix.clone(), comp_type.clone());
        *self.successful_patterns.entry(key).or_insert(0) += 1;

        // Update type preferences
        let prefs = self.type_preferences.entry(prefix).or_insert_with(HashMap::new);
        let count = *prefs.get(&comp_type).unwrap_or(&0.0);
        prefs.insert(comp_type, count + 1.0);
    }

    /// Suggest a composition type based on learned patterns
    pub fn suggest_composition_type(&self, operand_a: &str) -> CompositionType {
        let prefix = self.extract_prefix(operand_a);

        if let Some(prefs) = self.type_preferences.get(&prefix) {
            // Find the most successful type for this prefix
            let best = prefs.iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal));

            if let Some((type_str, _)) = best {
                return self.string_to_type(type_str);
            }
        }

        // Default to sequential
        CompositionType::Sequential
    }

    /// Extract a pattern prefix from an operand ID
    fn extract_prefix(&self, id: &str) -> String {
        // Take first component before underscore
        id.split('_').next().unwrap_or(id).to_string()
    }

    /// Convert composition type to string
    fn type_to_string(&self, comp_type: &CompositionType) -> String {
        match comp_type {
            CompositionType::Sequential => "seq".to_string(),
            CompositionType::Parallel => "par".to_string(),
            CompositionType::Conditional { .. } => "cond".to_string(),
            CompositionType::FixedPoint { .. } => "fix".to_string(),
            CompositionType::HigherOrder => "ho".to_string(),
            CompositionType::Fallback { .. } => "fall".to_string(),
        }
    }

    /// Convert string back to composition type
    fn string_to_type(&self, s: &str) -> CompositionType {
        match s {
            "seq" => CompositionType::Sequential,
            "par" => CompositionType::Parallel,
            "cond" => CompositionType::Conditional { pattern: String::new(), threshold: 500 },
            "fix" => CompositionType::FixedPoint { max_iterations: 100, convergence_threshold: 990 },
            "ho" => CompositionType::HigherOrder,
            "fall" => CompositionType::Fallback { confidence_threshold: 500 },
            _ => CompositionType::Sequential,
        }
    }

    /// Get learned rules as human-readable strings
    pub fn get_rules(&self) -> Vec<String> {
        let mut rules = Vec::new();

        for ((prefix, comp_type), count) in &self.successful_patterns {
            if *count >= 3 { // Only show rules with sufficient evidence
                rules.push(format!(
                    "{} + {} = success (observed {} times)",
                    prefix, comp_type, count
                ));
            }
        }

        rules.sort_by(|a, b| b.cmp(a)); // Sort by count descending
        rules
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EMERGENT COMPOSITION DISCOVERY
// ═══════════════════════════════════════════════════════════════════════════════

/// Discovers novel compositions through Φ-gradient ascent
pub struct EmergentDiscovery {
    /// The main search system
    search: PhiGuidedSearch,
    /// History of Φ improvements
    phi_history: VecDeque<f64>,
    /// Maximum history length
    max_history: usize,
    /// Stagnation threshold (cycles without improvement)
    stagnation_threshold: usize,
    /// Current stagnation count
    stagnation_count: usize,
}

impl EmergentDiscovery {
    /// Create a new emergent discovery system
    pub fn new(
        base_system: Arc<PrimitiveSystem>,
        config: DiscoveryConfig,
    ) -> Self {
        Self {
            search: PhiGuidedSearch::new(base_system, config),
            phi_history: VecDeque::new(),
            max_history: 100,
            stagnation_threshold: 10,
            stagnation_count: 0,
        }
    }

    /// Run continuous discovery until stagnation
    pub fn discover_until_stagnation(&mut self) -> Result<Vec<CompositionCandidate>> {
        let mut all_discovered = Vec::new();
        let mut last_best_phi = 0.0;

        loop {
            // Run discovery cycle
            let candidates = self.search.discover_cycle()?;
            all_discovered.extend(candidates);

            // Track Φ improvement
            let current_best = self.search.stats().best_phi;
            self.phi_history.push_back(current_best);

            if self.phi_history.len() > self.max_history {
                self.phi_history.pop_front();
            }

            // Check for stagnation
            if (current_best - last_best_phi).abs() < 0.001 {
                self.stagnation_count += 1;
            } else {
                self.stagnation_count = 0;
            }

            if self.stagnation_count >= self.stagnation_threshold {
                break;
            }

            last_best_phi = current_best;
        }

        Ok(all_discovered)
    }

    /// Run discovery for a fixed number of cycles
    pub fn discover_cycles(&mut self, n: usize) -> Result<Vec<CompositionCandidate>> {
        let mut all_discovered = Vec::new();

        for _ in 0..n {
            let candidates = self.search.discover_cycle()?;
            all_discovered.extend(candidates);
        }

        Ok(all_discovered)
    }

    /// Get the best discovered compositions
    pub fn best_discoveries(&self, n: usize) -> Vec<&CompositionCandidate> {
        self.search.best_compositions(n)
    }

    /// Get discovery statistics
    pub fn stats(&self) -> &DiscoveryStats {
        self.search.stats()
    }

    /// Get the learned grammar
    pub fn grammar(&self) -> &CompositionGrammar {
        self.search.grammar()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHI-GRADIENT OPTIMIZER FOR COMPOSITIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for Φ-gradient optimization
#[derive(Debug, Clone)]
pub struct PhiGradientConfig {
    /// Learning rate for gradient updates
    pub learning_rate: f64,
    /// Momentum coefficient
    pub momentum: f64,
    /// Number of samples for gradient estimation
    pub gradient_samples: usize,
    /// Perturbation size for finite differences
    pub epsilon: f64,
    /// Maximum optimization steps
    pub max_steps: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
}

impl Default for PhiGradientConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.9,
            gradient_samples: 5,
            epsilon: 0.001,
            max_steps: 100,
            convergence_threshold: 1e-6,
        }
    }
}

/// Optimizable parameters for a composition
#[derive(Debug, Clone)]
pub struct CompositionParameters {
    /// Composition weight (how much to blend with alternatives)
    pub weight: f64,
    /// Confidence threshold for fallback compositions
    pub confidence_threshold: f64,
    /// Convergence threshold for fixed-point compositions
    pub convergence_threshold: f64,
    /// Maximum iterations for iterative compositions
    pub max_iterations: usize,
}

impl Default for CompositionParameters {
    fn default() -> Self {
        Self {
            weight: 1.0,
            confidence_threshold: 0.5,
            convergence_threshold: 0.99,
            max_iterations: 10,
        }
    }
}

impl CompositionParameters {
    /// Convert to a parameter vector for optimization
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.weight,
            self.confidence_threshold,
            self.convergence_threshold,
            self.max_iterations as f64 / 100.0, // Normalize
        ]
    }

    /// Create from a parameter vector
    pub fn from_vec(v: &[f64]) -> Self {
        Self {
            weight: v.get(0).copied().unwrap_or(1.0).clamp(0.0, 2.0),
            confidence_threshold: v.get(1).copied().unwrap_or(0.5).clamp(0.0, 1.0),
            convergence_threshold: v.get(2).copied().unwrap_or(0.99).clamp(0.0, 1.0),
            max_iterations: (v.get(3).copied().unwrap_or(0.1) * 100.0).clamp(1.0, 100.0) as usize,
        }
    }

    /// Perturb parameters for gradient estimation
    pub fn perturb(&self, index: usize, delta: f64) -> Self {
        let mut v = self.to_vec();
        if index < v.len() {
            v[index] += delta;
        }
        Self::from_vec(&v)
    }
}

/// Φ-gradient optimizer for refining composition parameters
pub struct PhiGradientOptimizer {
    /// Consciousness oracle for Φ measurements
    oracle: ConsciousnessOracle,
    /// Configuration
    config: PhiGradientConfig,
    /// Momentum accumulator
    momentum: Vec<f64>,
    /// Optimization history
    history: Vec<OptimizationStep>,
}

/// Record of one optimization step
#[derive(Debug, Clone)]
pub struct OptimizationStep {
    /// Step number
    pub step: usize,
    /// Parameters before update
    pub params: CompositionParameters,
    /// Φ value at these parameters
    pub phi: f64,
    /// Estimated gradient
    pub gradient: Vec<f64>,
    /// Gradient magnitude
    pub gradient_norm: f64,
}

impl PhiGradientOptimizer {
    /// Create a new optimizer
    pub fn new(config: PhiGradientConfig) -> Self {
        Self {
            oracle: ConsciousnessOracle::new(OracleConfig::default()),
            config,
            momentum: vec![0.0; 4], // 4 parameters
            history: Vec::new(),
        }
    }

    /// Optimize parameters to maximize Φ for a given composition
    pub fn optimize(
        &mut self,
        composition: &ComposedPrimitive,
        initial_params: CompositionParameters,
    ) -> Result<(CompositionParameters, f64)> {
        let mut params = initial_params;
        let mut best_phi = 0.0;
        let mut best_params = params.clone();

        for step in 0..self.config.max_steps {
            // Estimate Φ at current parameters
            let current_phi = self.evaluate_phi(composition, &params);

            // Track best
            if current_phi > best_phi {
                best_phi = current_phi;
                best_params = params.clone();
            }

            // Estimate gradient using finite differences
            let gradient = self.estimate_gradient(composition, &params)?;
            let gradient_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

            // Record step
            self.history.push(OptimizationStep {
                step,
                params: params.clone(),
                phi: current_phi,
                gradient: gradient.clone(),
                gradient_norm,
            });

            // Check convergence
            if gradient_norm < self.config.convergence_threshold {
                break;
            }

            // Update with momentum
            for i in 0..self.momentum.len() {
                self.momentum[i] = self.config.momentum * self.momentum[i]
                    + self.config.learning_rate * gradient.get(i).copied().unwrap_or(0.0);
            }

            // Apply update (gradient ascent for maximizing Φ)
            let mut param_vec = params.to_vec();
            for i in 0..param_vec.len().min(self.momentum.len()) {
                param_vec[i] += self.momentum[i];
            }
            params = CompositionParameters::from_vec(&param_vec);
        }

        Ok((best_params, best_phi))
    }

    /// Evaluate Φ for a composition with given parameters
    fn evaluate_phi(&mut self, composition: &ComposedPrimitive, params: &CompositionParameters) -> f64 {
        // Use the composition encoding modulated by parameters
        let mut samples = Vec::new();

        for _ in 0..self.config.gradient_samples {
            let sample = self.oracle.process_and_measure(
                &composition.encoding,
                &format!("{}@w={:.2}", composition.name, params.weight),
            );
            samples.push(sample.phi * params.weight);
        }

        // Return average Φ
        samples.iter().sum::<f64>() / samples.len() as f64
    }

    /// Estimate gradient using finite differences
    fn estimate_gradient(
        &mut self,
        composition: &ComposedPrimitive,
        params: &CompositionParameters,
    ) -> Result<Vec<f64>> {
        let param_vec = params.to_vec();
        let mut gradient = vec![0.0; param_vec.len()];

        let base_phi = self.evaluate_phi(composition, params);

        for i in 0..param_vec.len() {
            // Positive perturbation
            let params_plus = params.perturb(i, self.config.epsilon);
            let phi_plus = self.evaluate_phi(composition, &params_plus);

            // Negative perturbation
            let params_minus = params.perturb(i, -self.config.epsilon);
            let phi_minus = self.evaluate_phi(composition, &params_minus);

            // Central difference
            gradient[i] = (phi_plus - phi_minus) / (2.0 * self.config.epsilon);
        }

        Ok(gradient)
    }

    /// Get optimization history
    pub fn history(&self) -> &[OptimizationStep] {
        &self.history
    }

    /// Clear history for new optimization
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.momentum = vec![0.0; 4];
    }
}

/// Combined system: Discovery + Gradient Optimization
pub struct PhiOptimizedDiscovery {
    /// Discovery system
    discovery: EmergentDiscovery,
    /// Gradient optimizer
    optimizer: PhiGradientOptimizer,
    /// Optimized compositions
    optimized: HashMap<String, (CompositionParameters, f64)>,
}

impl PhiOptimizedDiscovery {
    /// Create a new optimized discovery system
    pub fn new(
        base_system: Arc<PrimitiveSystem>,
        discovery_config: DiscoveryConfig,
        gradient_config: PhiGradientConfig,
    ) -> Self {
        Self {
            discovery: EmergentDiscovery::new(base_system, discovery_config),
            optimizer: PhiGradientOptimizer::new(gradient_config),
            optimized: HashMap::new(),
        }
    }

    /// Discover and optimize compositions
    pub fn discover_and_optimize(&mut self, cycles: usize) -> Result<Vec<(CompositionCandidate, f64)>> {
        // First, discover compositions
        let candidates = self.discovery.discover_cycles(cycles)?;

        // Then optimize the top candidates
        let mut results = Vec::new();
        let top_candidates: Vec<_> = self.discovery.best_discoveries(10)
            .into_iter()
            .cloned()
            .collect();

        for candidate in top_candidates {
            self.optimizer.clear_history();

            let initial_params = CompositionParameters::default();
            match self.optimizer.optimize(&candidate.composition, initial_params) {
                Ok((optimized_params, optimized_phi)) => {
                    self.optimized.insert(
                        candidate.composition.id.clone(),
                        (optimized_params, optimized_phi),
                    );
                    results.push((candidate, optimized_phi));
                }
                Err(_) => {
                    results.push((candidate.clone(), candidate.phi_score));
                }
            }
        }

        // Sort by optimized Φ
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    /// Get optimized parameters for a composition
    pub fn get_optimized(&self, id: &str) -> Option<&(CompositionParameters, f64)> {
        self.optimized.get(id)
    }

    /// Get discovery statistics
    pub fn stats(&self) -> &DiscoveryStats {
        self.discovery.stats()
    }

    /// Get the learned grammar
    pub fn grammar(&self) -> &CompositionGrammar {
        self.discovery.grammar()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNIT TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::HV16;
    use crate::consciousness::compositionality_primitives::CompositionMetadata;

    fn create_test_system() -> Arc<PrimitiveSystem> {
        Arc::new(PrimitiveSystem::new())
    }

    #[test]
    fn test_discovery_config_default() {
        let config = DiscoveryConfig::default();
        assert_eq!(config.beam_width, 100);
        assert_eq!(config.max_depth, 5);
        assert!(config.phi_threshold > 0.0);
    }

    #[test]
    fn test_phi_guided_search_creation() {
        let system = create_test_system();
        let config = DiscoveryConfig::default();
        let search = PhiGuidedSearch::new(system, config);

        assert_eq!(search.stats().cycles, 0);
        assert_eq!(search.stats().total_explored, 0);
    }

    #[test]
    fn test_discovery_cycle() {
        let system = create_test_system();
        let mut config = DiscoveryConfig::default();
        config.max_candidates_per_cycle = 5; // Small for testing

        let mut search = PhiGuidedSearch::new(system, config);

        let result = search.discover_cycle();
        assert!(result.is_ok());
        assert!(search.stats().cycles == 1);
    }

    #[test]
    fn test_grammar_learning() {
        let mut grammar = CompositionGrammar::new();

        let composition = ComposedPrimitive {
            id: "test_seq".to_string(),
            name: "Test Sequential".to_string(),
            composition_type: CompositionType::Sequential,
            operand_a: "similarity".to_string(),
            operand_b: Some("bind".to_string()),
            encoding: HV16::zero(),
            metadata: CompositionMetadata {
                expected_cost: 1.0,
                depth: 1,
                base_count: 2,
                expected_phi_contribution: 0.5,
                description: "Test".to_string(),
                tags: vec![],
            },
        };

        grammar.learn_from_success(&composition);
        grammar.learn_from_success(&composition);
        grammar.learn_from_success(&composition);

        assert_eq!(grammar.observations, 3);

        // Should suggest sequential for "similarity" prefix
        let suggested = grammar.suggest_composition_type("similarity_test");
        assert!(matches!(suggested, CompositionType::Sequential));
    }

    #[test]
    fn test_emergent_discovery_creation() {
        let system = create_test_system();
        let config = DiscoveryConfig::default();
        let discovery = EmergentDiscovery::new(system, config);

        assert_eq!(discovery.stagnation_count, 0);
    }

    #[test]
    fn test_multiple_discovery_cycles() {
        let system = create_test_system();
        let mut config = DiscoveryConfig::default();
        config.max_candidates_per_cycle = 3;
        config.phi_threshold = 0.0; // Accept all for testing

        let mut discovery = EmergentDiscovery::new(system, config);

        let result = discovery.discover_cycles(3);
        assert!(result.is_ok());
        assert!(discovery.stats().cycles >= 3);
    }

    #[test]
    fn test_composition_candidate_ordering() {
        let make_candidate = |phi: f64, confidence: f64| -> CompositionCandidate {
            CompositionCandidate {
                composition: ComposedPrimitive {
                    id: format!("test_{}", phi),
                    name: "Test".to_string(),
                    composition_type: CompositionType::Sequential,
                    operand_a: "a".to_string(),
                    operand_b: Some("b".to_string()),
                    encoding: HV16::zero(),
                    metadata: CompositionMetadata {
                        expected_cost: 1.0,
                        depth: 1,
                        base_count: 2,
                        expected_phi_contribution: phi as f32,
                        description: "Test".to_string(),
                        tags: vec![],
                    },
                },
                phi_score: phi,
                coherence: 0.5,
                integration: 0.5,
                evaluation_count: 1,
                confidence,
            }
        };

        let high_phi = make_candidate(0.8, 1.0);
        let low_phi = make_candidate(0.2, 1.0);

        // Higher Φ should be greater
        assert!(high_phi > low_phi);

        // Confidence matters too
        let high_phi_low_conf = make_candidate(0.8, 0.1);
        assert!(high_phi > high_phi_low_conf);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Φ-GRADIENT OPTIMIZER TESTS
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_phi_gradient_config_default() {
        let config = PhiGradientConfig::default();
        assert!(config.learning_rate > 0.0);
        assert!(config.momentum >= 0.0 && config.momentum < 1.0);
        assert!(config.gradient_samples > 0);
    }

    #[test]
    fn test_composition_parameters_default() {
        let params = CompositionParameters::default();
        assert_eq!(params.weight, 1.0);
        assert_eq!(params.confidence_threshold, 0.5);
        assert_eq!(params.convergence_threshold, 0.99);
        assert_eq!(params.max_iterations, 10);
    }

    #[test]
    fn test_composition_parameters_vec_roundtrip() {
        let params = CompositionParameters {
            weight: 0.8,
            confidence_threshold: 0.6,
            convergence_threshold: 0.95,
            max_iterations: 20,
        };

        let vec = params.to_vec();
        let restored = CompositionParameters::from_vec(&vec);

        assert!((restored.weight - params.weight).abs() < 0.01);
        assert!((restored.confidence_threshold - params.confidence_threshold).abs() < 0.01);
        assert!((restored.convergence_threshold - params.convergence_threshold).abs() < 0.01);
        assert_eq!(restored.max_iterations, params.max_iterations);
    }

    #[test]
    fn test_composition_parameters_perturb() {
        let params = CompositionParameters::default();

        // Perturb weight
        let perturbed = params.perturb(0, 0.1);
        assert!((perturbed.weight - 1.1).abs() < 0.01);
        assert_eq!(perturbed.confidence_threshold, params.confidence_threshold);
    }

    #[test]
    fn test_phi_gradient_optimizer_creation() {
        let config = PhiGradientConfig::default();
        let optimizer = PhiGradientOptimizer::new(config);

        assert!(optimizer.history().is_empty());
    }

    #[test]
    fn test_phi_gradient_optimizer_clear() {
        let config = PhiGradientConfig::default();
        let mut optimizer = PhiGradientOptimizer::new(config);

        optimizer.clear_history();
        assert!(optimizer.history().is_empty());
    }

    #[test]
    fn test_phi_optimized_discovery_creation() {
        let system = create_test_system();
        let discovery_config = DiscoveryConfig::default();
        let gradient_config = PhiGradientConfig::default();

        let discovery = PhiOptimizedDiscovery::new(system, discovery_config, gradient_config);
        assert_eq!(discovery.stats().cycles, 0);
    }

    #[test]
    fn test_phi_optimized_discovery_cycle() {
        let system = create_test_system();
        let mut discovery_config = DiscoveryConfig::default();
        discovery_config.max_candidates_per_cycle = 3;
        discovery_config.phi_threshold = 0.0;

        let mut gradient_config = PhiGradientConfig::default();
        gradient_config.max_steps = 2; // Fast for testing

        let mut discovery = PhiOptimizedDiscovery::new(system, discovery_config, gradient_config);

        let result = discovery.discover_and_optimize(1);
        assert!(result.is_ok());
        assert!(discovery.stats().cycles >= 1);
    }
}
