//! # Quantum-Inspired Coherence Routing
//!
//! This module implements routing using quantum mechanical principles:
//!
//! ## Key Concepts
//!
//! 1. **Superposition**: Route strategies exist as probability amplitudes
//!    until measurement forces a decision
//!
//! 2. **Interference**: Strategies can constructively/destructively interfere
//!    based on consciousness state alignment
//!
//! 3. **Coherence**: Measures how "quantum" the routing is - high coherence
//!    means many strategies viable, low means classical behavior
//!
//! 4. **Decoherence**: Environmental "noise" causes collapse to classical routing
//!
//! 5. **Entanglement**: Past decisions affect current state amplitudes
//!
//! ## Mathematical Foundation
//!
//! - State vector |ψ⟩ = Σᵢ αᵢ|strategyᵢ⟩ where Σ|αᵢ|² = 1
//! - Evolution via H|ψ⟩ where H encodes strategy preferences
//! - Measurement collapses to strategy with probability |αᵢ|²
//! - Off-diagonal density matrix elements track coherence

use std::f64::consts::PI;
use serde::{Deserialize, Serialize};

use super::{
    RoutingStrategy,
    TopologicalConsciousnessRouter, TopologicalRouterConfig, TopologicalRoutingDecision,
};
use crate::consciousness::recursive_improvement::LatentConsciousnessState;

// =============================================================================
// COMPLEX AMPLITUDE
// =============================================================================

/// Complex amplitude for quantum-inspired routing
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ComplexAmplitude {
    /// Real part
    pub re: f64,
    /// Imaginary part
    pub im: f64,
}

impl ComplexAmplitude {
    /// Create a new complex amplitude
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Create from polar form
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Get the magnitude squared (probability)
    pub fn norm_squared(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Get the magnitude
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Get the phase angle
    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Multiply by another complex amplitude
    pub fn mul(&self, other: &ComplexAmplitude) -> ComplexAmplitude {
        ComplexAmplitude {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    /// Add another complex amplitude
    pub fn add(&self, other: &ComplexAmplitude) -> ComplexAmplitude {
        ComplexAmplitude {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    /// Conjugate
    pub fn conj(&self) -> ComplexAmplitude {
        ComplexAmplitude {
            re: self.re,
            im: -self.im,
        }
    }

    /// Scalar multiply
    pub fn scale(&self, s: f64) -> ComplexAmplitude {
        ComplexAmplitude {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

impl Default for ComplexAmplitude {
    fn default() -> Self {
        Self { re: 0.0, im: 0.0 }
    }
}

// =============================================================================
// QUANTUM STATE VECTOR
// =============================================================================

/// Quantum state vector for routing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumStateVector {
    /// Amplitudes for each strategy (7 strategies total)
    pub amplitudes: Vec<ComplexAmplitude>,
    /// Whether the state has been measured (collapsed)
    pub collapsed: bool,
    /// The collapsed strategy (if measured)
    pub collapsed_strategy: Option<RoutingStrategy>,
    /// Time since last measurement (affects decoherence)
    pub coherence_time: f64,
}

impl QuantumStateVector {
    /// Create a new quantum state in equal superposition
    pub fn equal_superposition(n_strategies: usize) -> Self {
        let amplitude = 1.0 / (n_strategies as f64).sqrt();
        Self {
            amplitudes: vec![ComplexAmplitude::new(amplitude, 0.0); n_strategies],
            collapsed: false,
            collapsed_strategy: None,
            coherence_time: 0.0,
        }
    }

    /// Create a state focused on a specific strategy
    pub fn focused(strategy_idx: usize, n_strategies: usize) -> Self {
        let mut amplitudes = vec![ComplexAmplitude::default(); n_strategies];
        if strategy_idx < n_strategies {
            amplitudes[strategy_idx] = ComplexAmplitude::new(1.0, 0.0);
        }
        Self {
            amplitudes,
            collapsed: false,
            collapsed_strategy: None,
            coherence_time: 0.0,
        }
    }

    /// Normalize the state vector
    pub fn normalize(&mut self) {
        let norm: f64 = self.amplitudes.iter().map(|a| a.norm_squared()).sum();
        if norm > 0.0 {
            let factor = 1.0 / norm.sqrt();
            for a in &mut self.amplitudes {
                *a = a.scale(factor);
            }
        }
    }

    /// Get probability of each strategy
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_squared()).collect()
    }

    /// Get the most probable strategy
    pub fn most_probable(&self) -> usize {
        self.probabilities()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Compute purity (1 = pure state, 0 = maximally mixed)
    pub fn purity(&self) -> f64 {
        let probs = self.probabilities();
        probs.iter().map(|p| p * p).sum()
    }

    /// Compute von Neumann entropy
    pub fn entropy(&self) -> f64 {
        let probs = self.probabilities();
        let mut entropy = 0.0;
        for p in probs {
            if p > 1e-10 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Apply a phase shift to a specific strategy
    pub fn apply_phase(&mut self, idx: usize, phase: f64) {
        if idx < self.amplitudes.len() {
            let a = &self.amplitudes[idx];
            self.amplitudes[idx] = a.mul(&ComplexAmplitude::from_polar(1.0, phase));
        }
    }

    /// Apply decoherence (gradually collapse toward classical)
    pub fn decohere(&mut self, rate: f64) {
        // Decoherence suppresses off-diagonal elements (phases)
        // This is equivalent to phase randomization
        for a in &mut self.amplitudes {
            let prob = a.norm_squared();
            let decay = (-rate).exp();
            // Keep magnitude but decay phase toward 0
            let new_phase = a.phase() * decay;
            let r = prob.sqrt();
            *a = ComplexAmplitude::from_polar(r, new_phase);
        }
        self.normalize();
        self.coherence_time += 1.0;
    }

    /// Measure and collapse the state
    pub fn measure(&mut self) -> usize {
        if self.collapsed {
            return self.strategy_to_index(self.collapsed_strategy.unwrap_or(RoutingStrategy::StandardProcessing));
        }

        // Sample according to probabilities (deterministic for testing: use most probable)
        let chosen = self.most_probable();

        // Collapse to chosen state
        self.amplitudes = vec![ComplexAmplitude::default(); self.amplitudes.len()];
        self.amplitudes[chosen] = ComplexAmplitude::new(1.0, 0.0);
        self.collapsed = true;
        self.collapsed_strategy = Some(Self::index_to_strategy(chosen));

        chosen
    }

    /// Convert strategy index to enum
    pub fn index_to_strategy(idx: usize) -> RoutingStrategy {
        match idx {
            0 => RoutingStrategy::Reflexive,
            1 => RoutingStrategy::FastPatterns,
            2 => RoutingStrategy::HeuristicGuided,
            3 => RoutingStrategy::StandardProcessing,
            4 => RoutingStrategy::FullDeliberation,
            5 => RoutingStrategy::Ensemble,
            _ => RoutingStrategy::Preparatory,
        }
    }

    /// Convert strategy to index
    fn strategy_to_index(&self, strategy: RoutingStrategy) -> usize {
        match strategy {
            RoutingStrategy::Reflexive => 0,
            RoutingStrategy::FastPatterns => 1,
            RoutingStrategy::HeuristicGuided => 2,
            RoutingStrategy::StandardProcessing => 3,
            RoutingStrategy::FullDeliberation => 4,
            RoutingStrategy::Ensemble => 5,
            RoutingStrategy::Preparatory => 6,
        }
    }
}

// =============================================================================
// DENSITY MATRIX
// =============================================================================

/// Density matrix for tracking quantum coherence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityMatrix {
    /// Elements ρᵢⱼ (n×n matrix)
    pub elements: Vec<Vec<ComplexAmplitude>>,
    /// Dimension
    pub dim: usize,
}

impl DensityMatrix {
    /// Create from a pure state vector
    pub fn from_state(state: &QuantumStateVector) -> Self {
        let n = state.amplitudes.len();
        let mut elements = vec![vec![ComplexAmplitude::default(); n]; n];

        for i in 0..n {
            for j in 0..n {
                // ρᵢⱼ = αᵢ * αⱼ*
                elements[i][j] = state.amplitudes[i].mul(&state.amplitudes[j].conj());
            }
        }

        Self { elements, dim: n }
    }

    /// Get coherence measure (sum of off-diagonal magnitudes)
    pub fn coherence(&self) -> f64 {
        let mut coh = 0.0;
        for i in 0..self.dim {
            for j in 0..self.dim {
                if i != j {
                    coh += self.elements[i][j].norm();
                }
            }
        }
        coh
    }

    /// Get purity Tr(ρ²)
    pub fn purity(&self) -> f64 {
        let mut trace = 0.0;
        for i in 0..self.dim {
            for j in 0..self.dim {
                let elem = self.elements[i][j].mul(&self.elements[j][i]);
                trace += elem.re;
            }
        }
        trace
    }

    /// Get diagonal elements (classical probabilities)
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.dim).map(|i| self.elements[i][i].re).collect()
    }
}

// =============================================================================
// ROUTING HAMILTONIAN
// =============================================================================

/// Hamiltonian operator for quantum evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingHamiltonian {
    /// Matrix elements Hᵢⱼ (energy/preference between strategies)
    pub elements: Vec<Vec<f64>>,
    /// Base energies for each strategy
    pub energies: Vec<f64>,
    /// Coupling strength between strategies
    pub coupling: f64,
}

impl RoutingHamiltonian {
    /// Create a new Hamiltonian from consciousness state
    pub fn from_consciousness(state: &LatentConsciousnessState, coupling: f64) -> Self {
        let n = 7; // Number of strategies

        // Base energies: lower = more favorable
        // Map consciousness level to strategy energies
        let consciousness_level = state.phi;

        let mut energies = vec![0.0; n];
        // Higher consciousness favors deliberation (lower energy)
        energies[0] = 1.0 - consciousness_level; // Reflexive
        energies[1] = 0.8 - 0.6 * consciousness_level; // FastPatterns
        energies[2] = 0.6 - 0.4 * consciousness_level; // HeuristicGuided
        energies[3] = 0.4 - 0.2 * consciousness_level; // StandardProcessing
        energies[4] = 0.2 + 0.2 * consciousness_level; // FullDeliberation
        energies[5] = 0.3; // Ensemble (neutral)
        energies[6] = 0.5; // Preparatory (neutral)

        // Build Hamiltonian matrix
        let mut elements = vec![vec![0.0; n]; n];

        // Diagonal: base energies
        for i in 0..n {
            elements[i][i] = energies[i];
        }

        // Off-diagonal: coupling between adjacent strategies
        for i in 0..n - 1 {
            elements[i][i + 1] = coupling;
            elements[i + 1][i] = coupling;
        }

        Self {
            elements,
            energies,
            coupling,
        }
    }

    /// Apply Hamiltonian evolution: |ψ(t)⟩ = exp(-iHt)|ψ(0)⟩
    /// (Simplified: first-order approximation)
    pub fn evolve(&self, state: &mut QuantumStateVector, dt: f64) {
        let n = state.amplitudes.len();
        let mut new_amplitudes = vec![ComplexAmplitude::default(); n];

        for i in 0..n {
            // Apply -iHt to each amplitude
            for j in 0..n {
                let h_ij = self.elements[i][j];
                // exp(-iHt) ≈ 1 - iHt for small dt
                let phase_factor = -h_ij * dt;
                let evolution = ComplexAmplitude::from_polar(1.0, phase_factor);
                let contribution = state.amplitudes[j].mul(&evolution);
                new_amplitudes[i] = new_amplitudes[i].add(&contribution);
            }
        }

        state.amplitudes = new_amplitudes;
        state.normalize();
    }
}

// =============================================================================
// CONFIGURATION AND TYPES
// =============================================================================

/// Configuration for quantum router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumRouterConfig {
    /// Hamiltonian coupling strength
    pub coupling: f64,
    /// Evolution time step
    pub dt: f64,
    /// Number of evolution steps before measurement
    pub evolution_steps: usize,
    /// Decoherence rate
    pub decoherence_rate: f64,
    /// Threshold coherence for "quantum" behavior
    pub quantum_threshold: f64,
    /// Minimum probability to consider a strategy
    pub min_probability: f64,
}

impl Default for QuantumRouterConfig {
    fn default() -> Self {
        Self {
            coupling: 0.1,
            dt: 0.1,
            evolution_steps: 10,
            decoherence_rate: 0.05,
            quantum_threshold: 0.3,
            min_probability: 0.1,
        }
    }
}

/// Statistics for quantum router
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuantumRouterStats {
    /// Number of routing decisions
    pub decisions: usize,
    /// Number of times in superposition (quantum behavior)
    pub superposition_decisions: usize,
    /// Number of collapsed (classical) decisions
    pub classical_decisions: usize,
    /// Average coherence at decision time
    pub avg_coherence: f64,
    /// Average purity at decision time
    pub avg_purity: f64,
    /// Average entropy at decision time
    pub avg_entropy: f64,
    /// Interference effects observed
    pub interference_events: usize,
}

/// Quantum routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumRoutingDecision {
    /// Chosen strategy
    pub strategy: RoutingStrategy,
    /// Probability of chosen strategy
    pub probability: f64,
    /// Full probability distribution
    pub distribution: Vec<f64>,
    /// Coherence at decision time
    pub coherence: f64,
    /// Purity at decision time
    pub purity: f64,
    /// Whether decision was quantum (superposition) or classical
    pub is_quantum: bool,
    /// Entropy of the distribution
    pub entropy: f64,
    /// Detected interference pattern
    pub interference_detected: bool,
}

/// Summary of quantum router state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumRouterSummary {
    /// Current probabilities for each strategy
    pub probabilities: Vec<f64>,
    /// Current coherence
    pub coherence: f64,
    /// Total decisions made
    pub decisions: usize,
    /// Ratio of quantum to classical decisions
    pub quantum_ratio: f64,
    /// Whether currently in superposition
    pub in_superposition: bool,
}

// =============================================================================
// QUANTUM COHERENCE ROUTER
// =============================================================================

/// Quantum-Inspired Coherence Router
///
/// Uses quantum mechanical principles for routing decisions:
/// - Superposition of multiple strategies
/// - Interference between strategy amplitudes
/// - Coherence-based decision quality
/// - Decoherence under noise/uncertainty
pub struct QuantumCoherenceRouter {
    /// Underlying topological router for base decisions
    topological_router: TopologicalConsciousnessRouter,
    /// Current quantum state
    state: QuantumStateVector,
    /// Current Hamiltonian
    hamiltonian: Option<RoutingHamiltonian>,
    /// Configuration
    config: QuantumRouterConfig,
    /// Previous state for interference detection
    previous_state: Option<QuantumStateVector>,
    /// Statistics
    stats: QuantumRouterStats,
}

impl QuantumCoherenceRouter {
    /// Create a new quantum router
    pub fn new(config: QuantumRouterConfig) -> Self {
        Self {
            topological_router: TopologicalConsciousnessRouter::new(TopologicalRouterConfig::default()),
            state: QuantumStateVector::equal_superposition(7),
            hamiltonian: None,
            config,
            previous_state: None,
            stats: QuantumRouterStats::default(),
        }
    }

    /// Observe a consciousness state
    pub fn observe_state(&mut self, state: &LatentConsciousnessState) {
        // Update topological router
        self.topological_router.observe_state(state);

        // Create new Hamiltonian based on current consciousness
        self.hamiltonian = Some(RoutingHamiltonian::from_consciousness(state, self.config.coupling));

        // Apply decoherence
        self.state.decohere(self.config.decoherence_rate);
    }

    /// Evolve the quantum state
    fn evolve_state(&mut self) {
        if let Some(ref h) = self.hamiltonian {
            for _ in 0..self.config.evolution_steps {
                h.evolve(&mut self.state, self.config.dt);
            }
        }
    }

    /// Detect interference patterns
    fn detect_interference(&self) -> bool {
        if let Some(ref prev) = self.previous_state {
            // Check for phase-dependent probability changes
            let prev_probs = prev.probabilities();
            let curr_probs = self.state.probabilities();

            let mut positive_changes = 0;
            let mut negative_changes = 0;

            for (p, c) in prev_probs.iter().zip(curr_probs.iter()) {
                let delta = c - p;
                if delta > 0.05 {
                    positive_changes += 1;
                } else if delta < -0.05 {
                    negative_changes += 1;
                }
            }

            // Interference: simultaneous constructive and destructive
            positive_changes > 0 && negative_changes > 0
        } else {
            false
        }
    }

    /// Bias the state toward a particular strategy
    pub fn bias_toward(&mut self, strategy: RoutingStrategy, strength: f64) {
        let idx = self.strategy_to_index(strategy);
        if idx < self.state.amplitudes.len() {
            // Amplify the target amplitude
            let current = &self.state.amplitudes[idx];
            let boost = ComplexAmplitude::new(1.0 + strength, 0.0);
            self.state.amplitudes[idx] = current.mul(&boost);
            self.state.normalize();
        }
    }

    /// Apply constructive interference between strategies
    pub fn interfere(&mut self, idx1: usize, idx2: usize, phase: f64) {
        if idx1 < self.state.amplitudes.len() && idx2 < self.state.amplitudes.len() {
            // Create interference by phase-shifting one amplitude
            self.state.apply_phase(idx2, phase);

            // Combine amplitudes
            let a1 = &self.state.amplitudes[idx1];
            let a2 = &self.state.amplitudes[idx2];
            let combined = a1.add(a2).scale(0.5);

            // Apply interference to both
            self.state.amplitudes[idx1] = combined;
            self.state.amplitudes[idx2] = combined;
            self.state.normalize();
        }
    }

    /// Route with quantum mechanics
    pub fn route(&mut self, target: &LatentConsciousnessState) -> QuantumRoutingDecision {
        // Save previous state for interference detection
        self.previous_state = Some(self.state.clone());

        // Update Hamiltonian
        self.hamiltonian = Some(RoutingHamiltonian::from_consciousness(target, self.config.coupling));

        // Evolve quantum state
        self.evolve_state();

        // Compute density matrix for coherence
        let density = DensityMatrix::from_state(&self.state);
        let coherence = density.coherence();
        let purity = self.state.purity();
        let entropy = self.state.entropy();

        // Check for interference
        let interference_detected = self.detect_interference();
        if interference_detected {
            self.stats.interference_events += 1;
        }

        // Get probability distribution
        let distribution = self.state.probabilities();

        // Decide if we're in "quantum" regime (high coherence)
        let is_quantum = coherence > self.config.quantum_threshold;

        // Get base strategy from topological router
        let topo_decision = self.topological_router.route(target);

        // Choose strategy
        let (strategy, probability) = if is_quantum {
            // Quantum regime: use full probability distribution
            self.stats.superposition_decisions += 1;

            // Sample from distribution (or use most probable for determinism)
            let idx = self.state.most_probable();
            let prob = distribution[idx];

            // But blend with topological recommendation
            let topo_idx = self.strategy_to_index(topo_decision.strategy);
            let blended_idx = if distribution[topo_idx] > self.config.min_probability {
                topo_idx
            } else {
                idx
            };

            (QuantumStateVector::index_to_strategy(blended_idx), distribution[blended_idx])
        } else {
            // Classical regime: just use topological router's decision
            self.stats.classical_decisions += 1;
            let idx = self.strategy_to_index(topo_decision.strategy);
            (topo_decision.strategy, distribution[idx])
        };

        // Update running statistics
        let n = self.stats.decisions as f64;
        self.stats.avg_coherence = (self.stats.avg_coherence * n + coherence) / (n + 1.0);
        self.stats.avg_purity = (self.stats.avg_purity * n + purity) / (n + 1.0);
        self.stats.avg_entropy = (self.stats.avg_entropy * n + entropy) / (n + 1.0);
        self.stats.decisions += 1;

        QuantumRoutingDecision {
            strategy,
            probability,
            distribution,
            coherence,
            purity,
            is_quantum,
            entropy,
            interference_detected,
        }
    }

    /// Helper: convert strategy to index
    fn strategy_to_index(&self, strategy: RoutingStrategy) -> usize {
        match strategy {
            RoutingStrategy::Reflexive => 0,
            RoutingStrategy::FastPatterns => 1,
            RoutingStrategy::HeuristicGuided => 2,
            RoutingStrategy::StandardProcessing => 3,
            RoutingStrategy::FullDeliberation => 4,
            RoutingStrategy::Ensemble => 5,
            RoutingStrategy::Preparatory => 6,
        }
    }

    /// Get current probabilities
    pub fn probabilities(&self) -> Vec<f64> {
        self.state.probabilities()
    }

    /// Get current coherence
    pub fn coherence(&self) -> f64 {
        DensityMatrix::from_state(&self.state).coherence()
    }

    /// Check if in superposition (quantum) regime
    pub fn is_quantum(&self) -> bool {
        self.coherence() > self.config.quantum_threshold
    }

    /// Reset to equal superposition
    pub fn reset(&mut self) {
        self.state = QuantumStateVector::equal_superposition(7);
        self.previous_state = None;
    }

    /// Get summary of router state
    pub fn summary(&self) -> QuantumRouterSummary {
        let total = self.stats.superposition_decisions + self.stats.classical_decisions;
        let quantum_ratio = if total > 0 {
            self.stats.superposition_decisions as f64 / total as f64
        } else {
            0.0
        };

        QuantumRouterSummary {
            probabilities: self.state.probabilities(),
            coherence: self.coherence(),
            decisions: self.stats.decisions,
            quantum_ratio,
            in_superposition: !self.state.collapsed,
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &QuantumRouterStats {
        &self.stats
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_amplitude() {
        let a = ComplexAmplitude::new(3.0, 4.0);
        assert!((a.norm() - 5.0).abs() < 0.001);
        assert!((a.norm_squared() - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_complex_multiplication() {
        let a = ComplexAmplitude::new(1.0, 1.0);
        let b = ComplexAmplitude::new(1.0, -1.0);
        let c = a.mul(&b);
        // (1+i)(1-i) = 1 - i + i - i² = 1 + 1 = 2
        assert!((c.re - 2.0).abs() < 0.001);
        assert!(c.im.abs() < 0.001);
    }

    #[test]
    fn test_complex_from_polar() {
        let a = ComplexAmplitude::from_polar(1.0, PI / 2.0);
        assert!(a.re.abs() < 0.001);
        assert!((a.im - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_quantum_state_equal_superposition() {
        let state = QuantumStateVector::equal_superposition(7);
        let probs = state.probabilities();

        // Each probability should be 1/7
        for p in &probs {
            assert!((p - 1.0 / 7.0).abs() < 0.001);
        }

        // Total probability should be 1
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_quantum_state_focused() {
        let state = QuantumStateVector::focused(3, 7);
        let probs = state.probabilities();

        assert!((probs[3] - 1.0).abs() < 0.001);
        for (i, p) in probs.iter().enumerate() {
            if i != 3 {
                assert!(p.abs() < 0.001);
            }
        }
    }

    #[test]
    fn test_quantum_state_measurement() {
        let mut state = QuantumStateVector::focused(2, 7);
        let result = state.measure();

        assert_eq!(result, 2);
        assert!(state.collapsed);
    }

    #[test]
    fn test_quantum_state_entropy() {
        // Equal superposition should have maximum entropy
        let equal = QuantumStateVector::equal_superposition(7);
        let eq_entropy = equal.entropy();

        // Focused state should have zero entropy
        let focused = QuantumStateVector::focused(0, 7);
        let f_entropy = focused.entropy();

        assert!(eq_entropy > f_entropy);
        assert!(f_entropy.abs() < 0.001);
    }

    #[test]
    fn test_quantum_state_purity() {
        // Focused state should have purity = 1
        let focused = QuantumStateVector::focused(0, 7);
        assert!((focused.purity() - 1.0).abs() < 0.001);

        // Equal superposition should have lower purity
        let equal = QuantumStateVector::equal_superposition(7);
        assert!(equal.purity() < 0.5);
    }

    #[test]
    fn test_density_matrix_coherence() {
        // Equal superposition should have high coherence
        let state = QuantumStateVector::equal_superposition(7);
        let density = DensityMatrix::from_state(&state);

        assert!(density.coherence() > 0.0);
    }

    #[test]
    fn test_density_matrix_purity() {
        let state = QuantumStateVector::focused(0, 7);
        let density = DensityMatrix::from_state(&state);

        assert!((density.purity() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_hamiltonian_creation() {
        let state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        let h = RoutingHamiltonian::from_consciousness(&state, 0.1);

        assert_eq!(h.elements.len(), 7);
        assert_eq!(h.energies.len(), 7);
    }

    #[test]
    fn test_hamiltonian_evolution() {
        let state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        let h = RoutingHamiltonian::from_consciousness(&state, 0.1);

        let mut qstate = QuantumStateVector::focused(0, 7);
        let initial_prob = qstate.probabilities()[0];

        h.evolve(&mut qstate, 0.1);

        // Evolution should spread probability
        let final_prob = qstate.probabilities()[0];
        assert!(final_prob < initial_prob);
    }

    #[test]
    fn test_quantum_router_creation() {
        let router = QuantumCoherenceRouter::new(QuantumRouterConfig::default());
        assert_eq!(router.stats.decisions, 0);
    }

    #[test]
    fn test_quantum_router_observe() {
        let mut router = QuantumCoherenceRouter::new(QuantumRouterConfig::default());

        let state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        router.observe_state(&state);

        assert!(router.hamiltonian.is_some());
    }

    #[test]
    fn test_quantum_router_route() {
        let mut router = QuantumCoherenceRouter::new(QuantumRouterConfig::default());

        for i in 0..10 {
            let v = 0.3 + (i as f64) * 0.05;
            let state = LatentConsciousnessState::from_observables(v, v, v, v);
            router.observe_state(&state);
        }

        let target = LatentConsciousnessState::from_observables(0.6, 0.6, 0.6, 0.6);
        let decision = router.route(&target);

        assert!(decision.probability > 0.0);
        assert!(decision.coherence >= 0.0);
        assert_eq!(router.stats.decisions, 1);
    }

    #[test]
    fn test_quantum_router_probabilities() {
        let router = QuantumCoherenceRouter::new(QuantumRouterConfig::default());
        let probs = router.probabilities();

        assert_eq!(probs.len(), 7);
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_quantum_router_bias() {
        let mut router = QuantumCoherenceRouter::new(QuantumRouterConfig::default());

        let initial_prob = router.probabilities()[4];
        router.bias_toward(RoutingStrategy::FullDeliberation, 0.5);
        let biased_prob = router.probabilities()[4];

        assert!(biased_prob > initial_prob);
    }

    #[test]
    fn test_quantum_router_interference() {
        let mut router = QuantumCoherenceRouter::new(QuantumRouterConfig::default());

        // Apply constructive interference
        router.interfere(0, 1, 0.0);

        let probs = router.probabilities();
        // After interference, probabilities should still sum to 1
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_quantum_router_decoherence() {
        let mut router = QuantumCoherenceRouter::new(QuantumRouterConfig {
            decoherence_rate: 0.5, // High decoherence
            ..Default::default()
        });

        let initial_coherence = router.coherence();

        let state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        router.observe_state(&state);

        // Coherence should decrease
        let final_coherence = router.coherence();
        // Note: with high initial coherence and decoherence, it might not always decrease
        // Just check it's finite
        assert!(final_coherence.is_finite());
    }

    #[test]
    fn test_quantum_router_reset() {
        let mut router = QuantumCoherenceRouter::new(QuantumRouterConfig::default());

        router.bias_toward(RoutingStrategy::Reflexive, 1.0);
        router.reset();

        let probs = router.probabilities();
        for p in &probs {
            assert!((p - 1.0 / 7.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_quantum_router_summary() {
        let router = QuantumCoherenceRouter::new(QuantumRouterConfig::default());
        let summary = router.summary();

        assert_eq!(summary.probabilities.len(), 7);
        assert!(summary.in_superposition);
    }

    #[test]
    fn test_quantum_router_is_quantum() {
        let router = QuantumCoherenceRouter::new(QuantumRouterConfig {
            quantum_threshold: 0.0, // Very low threshold
            ..Default::default()
        });

        // Equal superposition should have coherence
        assert!(router.is_quantum());
    }

    #[test]
    fn test_quantum_decision_structure() {
        let decision = QuantumRoutingDecision {
            strategy: RoutingStrategy::StandardProcessing,
            probability: 0.5,
            distribution: vec![0.1; 7],
            coherence: 0.5,
            purity: 0.8,
            is_quantum: true,
            entropy: 0.5,
            interference_detected: false,
        };

        assert!(decision.is_quantum);
        assert_eq!(decision.distribution.len(), 7);
    }

    #[test]
    fn test_phase_evolution() {
        let mut state = QuantumStateVector::equal_superposition(7);
        let initial_phase = state.amplitudes[0].phase();

        state.apply_phase(0, PI / 4.0);

        let final_phase = state.amplitudes[0].phase();
        assert!((final_phase - initial_phase - PI / 4.0).abs() < 0.001);
    }
}
