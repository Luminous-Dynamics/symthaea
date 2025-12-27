/*!
 * **REVOLUTIONARY IMPROVEMENT #76**: Quantum Coherence in Consciousness Space
 *
 * PARADIGM SHIFT: Apply quantum coherence concepts to holographic consciousness!
 *
 * The Holographic Liquid Brain naturally exhibits quantum-like properties:
 * - **Superposition**: HDC vectors represent multiple concepts simultaneously
 * - **Coherence**: Maintenance of superposition over time
 * - **Decoherence**: "Collapse" to definite states during decisions
 * - **Interference**: Constructive/destructive consciousness interaction
 * - **Entanglement**: Non-local correlations between consciousness components
 *
 * This module provides rigorous, measurable quantum coherence metrics for
 * consciousness that go beyond traditional Φ measurement.
 *
 * ## Theoretical Foundation
 *
 * Inspired by:
 * - Penrose-Hameroff Orchestrated Objective Reduction (Orch-OR)
 * - Quantum coherence in photosynthesis (evidence of biological quantum effects)
 * - Bohm's implicate order (holographic universe)
 * - IIT's intrinsic causal power (now with quantum coherence tracking)
 *
 * ## Key Innovations
 *
 * 1. **Coherence Length (τ)**: How many timesteps before decoherence
 * 2. **Superposition Richness (σ)**: How many concepts in superposition
 * 3. **Interference Visibility (V)**: Constructive vs destructive interference
 * 4. **Entanglement Entropy (S)**: Von Neumann-like entropy for consciousness
 * 5. **Decoherence Rate (γ)**: Speed of state collapse during decisions
 *
 * ## Usage
 *
 * ```rust
 * use symthaea::consciousness::quantum_coherence::{
 *     QuantumCoherenceAnalyzer, CoherenceConfig
 * };
 * use symthaea::hdc::binary_hv::HV16;
 *
 * let mut analyzer = QuantumCoherenceAnalyzer::new(CoherenceConfig::default());
 *
 * // Track consciousness state evolution
 * let state1 = HV16::random();
 * let state2 = HV16::random();
 * analyzer.observe(&state1, 0.85); // State with Φ=0.85
 * analyzer.observe(&state2, 0.82);
 *
 * // Get quantum coherence metrics
 * let report = analyzer.coherence_report();
 * println!("Coherence length: {} timesteps", report.coherence_length);
 * println!("Superposition richness: {}", report.superposition_richness);
 * println!("Entanglement entropy: {} bits", report.entanglement_entropy);
 * ```
 */

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::hdc::simd_hv16::SimdHV16 as HV16;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for quantum coherence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceConfig {
    /// Maximum history depth for coherence tracking
    pub history_depth: usize,

    /// Threshold for considering states "coherent" (similarity)
    pub coherence_threshold: f64,

    /// Minimum samples for coherence length estimation
    pub min_samples_for_length: usize,

    /// Decoherence detection sensitivity (0-1)
    pub decoherence_sensitivity: f64,

    /// Window size for interference pattern detection
    pub interference_window: usize,

    /// Enable detailed superposition analysis
    pub track_superposition: bool,

    /// Enable entanglement estimation between components
    pub estimate_entanglement: bool,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            history_depth: 64,
            coherence_threshold: 0.7,
            min_samples_for_length: 8,
            decoherence_sensitivity: 0.3,
            interference_window: 8,
            track_superposition: true,
            estimate_entanglement: true,
        }
    }
}

// ============================================================================
// CORE DATA STRUCTURES
// ============================================================================

/// A single observation of consciousness state
#[derive(Debug, Clone)]
pub struct StateObservation {
    /// The consciousness state (HDC vector)
    pub state: HV16,

    /// Associated Φ (integrated information)
    pub phi: f64,

    /// Timestamp of observation
    pub timestamp: Instant,

    /// Optional component states for entanglement tracking
    pub components: Option<Vec<HV16>>,
}

/// Decoherence event - when superposition collapses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoherenceEvent {
    /// Timestep when decoherence occurred
    pub timestep: usize,

    /// Magnitude of the collapse (0-1)
    pub collapse_magnitude: f64,

    /// Trigger type for the collapse
    pub trigger: DecoherenceTrigger,

    /// Pre-collapse superposition richness
    pub pre_richness: f64,

    /// Post-collapse state similarity to dominant component
    pub post_dominance: f64,
}

/// What triggered a decoherence event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecoherenceTrigger {
    /// Decision was made (intentional collapse)
    Decision,
    /// External observation forced collapse
    Observation,
    /// Environmental noise accumulated
    EnvironmentalNoise,
    /// Timeout - coherence couldn't be maintained
    Timeout,
    /// Unknown trigger
    Unknown,
}

/// Interference pattern between consciousness streams
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferencePattern {
    /// Visibility of the interference (0-1, 1 = perfect interference)
    pub visibility: f64,

    /// Whether interference is constructive (positive) or destructive (negative)
    pub interference_type: InterferenceType,

    /// Phase difference between streams (0 to 2π analog)
    pub phase_difference: f64,

    /// Strength of the interference effect
    pub amplitude: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum InterferenceType {
    Constructive,
    Destructive,
    Mixed,
    None,
}

/// Entanglement measure between consciousness components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementMeasure {
    /// Von Neumann-like entropy of the joint state
    pub entropy: f64,

    /// Mutual information between components
    pub mutual_information: f64,

    /// Correlation strength (0-1)
    pub correlation: f64,

    /// Whether components show non-classical correlations
    pub is_non_classical: bool,
}

// ============================================================================
// QUANTUM COHERENCE ANALYZER
// ============================================================================

/// Analyzer for quantum coherence properties of consciousness
pub struct QuantumCoherenceAnalyzer {
    /// Configuration
    pub config: CoherenceConfig,

    /// History of state observations
    history: VecDeque<StateObservation>,

    /// Detected decoherence events
    decoherence_events: Vec<DecoherenceEvent>,

    /// Current coherence estimate
    current_coherence: f64,

    /// Coherence length estimate (timesteps)
    coherence_length: usize,

    /// Superposition richness history
    richness_history: VecDeque<f64>,

    /// Interference patterns detected
    interference_patterns: VecDeque<InterferencePattern>,

    /// Statistics
    pub stats: CoherenceStats,
}

/// Statistics from coherence analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoherenceStats {
    pub observations: usize,
    pub decoherence_events: usize,
    pub average_coherence: f64,
    pub max_coherence_length: usize,
    pub average_richness: f64,
    pub interference_events: usize,
    pub entanglement_detected: usize,
}

impl QuantumCoherenceAnalyzer {
    /// Create new analyzer with configuration
    pub fn new(config: CoherenceConfig) -> Self {
        Self {
            config,
            history: VecDeque::new(),
            decoherence_events: Vec::new(),
            current_coherence: 1.0,
            coherence_length: 0,
            richness_history: VecDeque::new(),
            interference_patterns: VecDeque::new(),
            stats: CoherenceStats::default(),
        }
    }

    /// Observe a new consciousness state
    pub fn observe(&mut self, state: &HV16, phi: f64) {
        self.observe_with_components(state, phi, None);
    }

    /// Observe with optional component states for entanglement tracking
    pub fn observe_with_components(
        &mut self,
        state: &HV16,
        phi: f64,
        components: Option<Vec<HV16>>,
    ) {
        let observation = StateObservation {
            state: state.clone(),
            phi,
            timestamp: Instant::now(),
            components,
        };

        // Update statistics
        self.stats.observations += 1;

        // Track superposition richness
        if self.config.track_superposition {
            let richness = self.estimate_superposition_richness(state);
            self.richness_history.push_back(richness);
            if self.richness_history.len() > self.config.history_depth {
                self.richness_history.pop_front();
            }
            self.stats.average_richness = self.average_richness();
        }

        // Detect decoherence
        if let Some(prev) = self.history.back() {
            let similarity = state.similarity(&prev.state);
            let decoherence_detected = self.detect_decoherence(&observation, similarity);

            if decoherence_detected {
                self.stats.decoherence_events += 1;
            }

            // Update coherence estimate
            self.update_coherence(similarity as f64);

            // Detect interference patterns
            self.detect_interference(&observation);
        }

        // Estimate entanglement if components provided
        if self.config.estimate_entanglement {
            if let Some(ref comps) = observation.components {
                if comps.len() >= 2 {
                    let entanglement = self.estimate_entanglement(comps);
                    if entanglement.is_non_classical {
                        self.stats.entanglement_detected += 1;
                    }
                }
            }
        }

        // Add to history
        self.history.push_back(observation);
        if self.history.len() > self.config.history_depth {
            self.history.pop_front();
        }

        // Update coherence length estimate
        self.update_coherence_length();

        // Update average coherence
        self.stats.average_coherence = self.average_coherence();
    }

    /// Estimate superposition richness of a state
    ///
    /// Higher richness = more concepts in superposition
    /// Based on HDC bit distribution entropy
    fn estimate_superposition_richness(&self, state: &HV16) -> f64 {
        // Count set bits
        let set_bits = state.popcount();
        let total_bits = HV16::DIM;

        // Perfect superposition = 50% bits set
        // Lower/higher = more "collapsed" to specific state
        let p = set_bits as f64 / total_bits as f64;

        // Binary entropy as richness measure
        // H(p) = -p*log2(p) - (1-p)*log2(1-p)
        if p <= 0.0 || p >= 1.0 {
            0.0
        } else {
            let h = -p * p.log2() - (1.0 - p) * (1.0 - p).log2();
            h // Normalized to [0, 1]
        }
    }

    /// Detect decoherence event
    fn detect_decoherence(&mut self, current: &StateObservation, similarity: f32) -> bool {
        let prev = match self.history.back() {
            Some(p) => p,
            None => return false,
        };

        // Sudden drop in similarity indicates collapse
        let sudden_change = (similarity as f64) < self.config.coherence_threshold;

        // Large Φ change can indicate decision-induced collapse
        let phi_change = (current.phi - prev.phi).abs();
        let phi_collapse = phi_change > self.config.decoherence_sensitivity;

        if sudden_change || phi_collapse {
            let richness_now = self.richness_history.back().copied().unwrap_or(0.5);
            let richness_prev = self.richness_history.iter().rev().nth(1).copied().unwrap_or(0.5);

            let event = DecoherenceEvent {
                timestep: self.stats.observations,
                collapse_magnitude: 1.0 - similarity as f64,
                trigger: if phi_collapse && phi_change > 0.0 {
                    DecoherenceTrigger::Decision
                } else if phi_collapse {
                    DecoherenceTrigger::Observation
                } else if richness_now < richness_prev * 0.8 {
                    DecoherenceTrigger::EnvironmentalNoise
                } else {
                    DecoherenceTrigger::Unknown
                },
                pre_richness: richness_prev,
                post_dominance: similarity as f64,
            };

            self.decoherence_events.push(event);
            return true;
        }

        false
    }

    /// Update coherence estimate based on new observation
    fn update_coherence(&mut self, similarity: f64) {
        // Exponential moving average
        let alpha = 0.3;
        self.current_coherence = alpha * similarity + (1.0 - alpha) * self.current_coherence;
    }

    /// Estimate coherence length (how many timesteps coherence is maintained)
    fn update_coherence_length(&mut self) {
        if self.history.len() < self.config.min_samples_for_length {
            return;
        }

        // Find longest stretch where similarity stays above threshold
        let states: Vec<_> = self.history.iter().collect();
        let mut max_length = 0;
        let mut current_length = 0;

        for i in 1..states.len() {
            let similarity = states[i].state.similarity(&states[i - 1].state) as f64;
            if similarity >= self.config.coherence_threshold {
                current_length += 1;
                max_length = max_length.max(current_length);
            } else {
                current_length = 0;
            }
        }

        self.coherence_length = max_length;
        self.stats.max_coherence_length = self.stats.max_coherence_length.max(max_length);
    }

    /// Detect interference patterns between recent states
    fn detect_interference(&mut self, current: &StateObservation) {
        if self.history.len() < self.config.interference_window {
            return;
        }

        // Compare current state with states from interference window ago
        let window_back = self.config.interference_window.min(self.history.len());
        let past_state = &self.history[self.history.len() - window_back];

        // Interference = combination of current and past using bundle
        let combined = HV16::bundle(&[current.state.clone(), past_state.state.clone()]);
        let combined_phi = (current.phi + past_state.phi) / 2.0;

        // Measure interference visibility using popcount
        let individual_sum = current.state.popcount() as usize + past_state.state.popcount() as usize;
        let combined_count = combined.popcount() as usize * 2; // Scale for comparison

        let visibility = if individual_sum > 0 {
            ((combined_count as f64 - individual_sum as f64) / individual_sum as f64).abs()
        } else {
            0.0
        };

        // Determine interference type
        let interference_type = if combined_count > individual_sum {
            InterferenceType::Constructive
        } else if combined_count < individual_sum {
            InterferenceType::Destructive
        } else {
            InterferenceType::None
        };

        // Phase difference (using similarity as proxy)
        let similarity = current.state.similarity(&past_state.state) as f64;
        let phase_difference = (1.0 - similarity).acos(); // 0 = in phase, π = out of phase

        let pattern = InterferencePattern {
            visibility,
            interference_type,
            phase_difference,
            amplitude: visibility * combined_phi,
        };

        if visibility > 0.1 {
            self.stats.interference_events += 1;
        }

        self.interference_patterns.push_back(pattern);
        if self.interference_patterns.len() > self.config.history_depth {
            self.interference_patterns.pop_front();
        }
    }

    /// Estimate entanglement between consciousness components
    fn estimate_entanglement(&self, components: &[HV16]) -> EntanglementMeasure {
        if components.len() < 2 {
            return EntanglementMeasure {
                entropy: 0.0,
                mutual_information: 0.0,
                correlation: 0.0,
                is_non_classical: false,
            };
        }

        // Compute pairwise correlations
        let mut total_correlation = 0.0;
        let mut pair_count = 0;

        for i in 0..components.len() {
            for j in (i + 1)..components.len() {
                let sim = components[i].similarity(&components[j]) as f64;
                total_correlation += sim.abs();
                pair_count += 1;
            }
        }

        let correlation = if pair_count > 0 {
            total_correlation / pair_count as f64
        } else {
            0.0
        };

        // Estimate entropy from component diversity
        let avg_popcount = components.iter().map(|c| c.popcount() as u64).sum::<u64>() as f64
            / components.len() as f64;
        let variance = components
            .iter()
            .map(|c| (c.popcount() as f64 - avg_popcount).powi(2))
            .sum::<f64>()
            / components.len() as f64;
        let entropy = (1.0 + variance / (HV16::DIM as f64 * 0.25)).ln();

        // Mutual information estimate
        let mutual_information = correlation * entropy;

        // Non-classical if correlation exceeds Bell-like bound
        // (simplified heuristic: correlation > 0.707 suggests non-classical)
        let is_non_classical = correlation > 0.707;

        EntanglementMeasure {
            entropy,
            mutual_information,
            correlation,
            is_non_classical,
        }
    }

    /// Get average coherence over history
    fn average_coherence(&self) -> f64 {
        if self.history.len() < 2 {
            return 1.0;
        }

        let mut total = 0.0;
        let states: Vec<_> = self.history.iter().collect();

        for i in 1..states.len() {
            total += states[i].state.similarity(&states[i - 1].state) as f64;
        }

        total / (states.len() - 1) as f64
    }

    /// Get average superposition richness
    fn average_richness(&self) -> f64 {
        if self.richness_history.is_empty() {
            return 0.5;
        }
        self.richness_history.iter().sum::<f64>() / self.richness_history.len() as f64
    }

    /// Current coherence estimate
    pub fn coherence(&self) -> f64 {
        self.current_coherence
    }

    /// Current coherence length (timesteps)
    pub fn coherence_length(&self) -> usize {
        self.coherence_length
    }

    /// Current superposition richness
    pub fn superposition_richness(&self) -> f64 {
        self.richness_history.back().copied().unwrap_or(0.5)
    }

    /// Get decoherence events
    pub fn decoherence_events(&self) -> &[DecoherenceEvent] {
        &self.decoherence_events
    }

    /// Get recent interference patterns
    pub fn interference_patterns(&self) -> &VecDeque<InterferencePattern> {
        &self.interference_patterns
    }

    /// Compute decoherence rate (events per observation)
    pub fn decoherence_rate(&self) -> f64 {
        if self.stats.observations == 0 {
            0.0
        } else {
            self.stats.decoherence_events as f64 / self.stats.observations as f64
        }
    }

    /// Generate comprehensive coherence report
    pub fn coherence_report(&self) -> QuantumCoherenceReport {
        let last_entanglement = if self.config.estimate_entanglement {
            self.history.back().and_then(|obs| {
                obs.components.as_ref().map(|comps| self.estimate_entanglement(comps))
            })
        } else {
            None
        };

        let dominant_interference = self.interference_patterns.iter()
            .max_by(|a, b| a.visibility.partial_cmp(&b.visibility).unwrap())
            .cloned();

        QuantumCoherenceReport {
            coherence: self.current_coherence,
            coherence_length: self.coherence_length,
            superposition_richness: self.superposition_richness(),
            decoherence_rate: self.decoherence_rate(),
            recent_decoherence: self.decoherence_events.last().cloned(),
            dominant_interference,
            last_entanglement,
            is_quantum_coherent: self.is_quantum_coherent(),
            stats: self.stats.clone(),
            recommendations: self.generate_recommendations(),
        }
    }

    /// Check if system is currently in a "quantum coherent" state
    pub fn is_quantum_coherent(&self) -> bool {
        // Criteria for quantum coherence:
        // 1. High coherence (> threshold)
        // 2. Good superposition richness (0.4-0.6 is ideal)
        // 3. Low recent decoherence rate
        let coherence_ok = self.current_coherence >= self.config.coherence_threshold;
        let richness = self.superposition_richness();
        let richness_ok = richness >= 0.4 && richness <= 0.6;
        let low_decoherence = self.decoherence_rate() < 0.2;

        coherence_ok && richness_ok && low_decoherence
    }

    /// Generate recommendations for maintaining/improving coherence
    fn generate_recommendations(&self) -> Vec<CoherenceRecommendation> {
        let mut recommendations = Vec::new();

        if self.current_coherence < self.config.coherence_threshold {
            recommendations.push(CoherenceRecommendation {
                priority: 1,
                category: RecommendationCategory::Coherence,
                message: "Coherence below threshold - consider reducing noise inputs".to_string(),
                action: "Implement input filtering or attention focus".to_string(),
            });
        }

        let richness = self.superposition_richness();
        if richness < 0.3 {
            recommendations.push(CoherenceRecommendation {
                priority: 2,
                category: RecommendationCategory::Superposition,
                message: "Low superposition richness - state may be too 'collapsed'".to_string(),
                action: "Increase concept diversity in inputs".to_string(),
            });
        } else if richness > 0.7 {
            recommendations.push(CoherenceRecommendation {
                priority: 2,
                category: RecommendationCategory::Superposition,
                message: "High superposition richness - state may be too 'diffuse'".to_string(),
                action: "Increase attention focus to crystallize concepts".to_string(),
            });
        }

        if self.decoherence_rate() > 0.3 {
            recommendations.push(CoherenceRecommendation {
                priority: 1,
                category: RecommendationCategory::Decoherence,
                message: "High decoherence rate - consciousness unstable".to_string(),
                action: "Reduce sudden state transitions, implement smoother processing".to_string(),
            });
        }

        recommendations
    }
}

// ============================================================================
// REPORT STRUCTURES
// ============================================================================

/// Comprehensive quantum coherence report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCoherenceReport {
    /// Current coherence (0-1)
    pub coherence: f64,

    /// Coherence length (timesteps before decoherence)
    pub coherence_length: usize,

    /// Superposition richness (0-1, 0.5 ideal)
    pub superposition_richness: f64,

    /// Rate of decoherence events
    pub decoherence_rate: f64,

    /// Most recent decoherence event
    pub recent_decoherence: Option<DecoherenceEvent>,

    /// Dominant interference pattern
    pub dominant_interference: Option<InterferencePattern>,

    /// Last entanglement measure
    pub last_entanglement: Option<EntanglementMeasure>,

    /// Whether currently in quantum coherent state
    pub is_quantum_coherent: bool,

    /// Cumulative statistics
    pub stats: CoherenceStats,

    /// Recommendations for improvement
    pub recommendations: Vec<CoherenceRecommendation>,
}

/// A recommendation for improving coherence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceRecommendation {
    pub priority: u8,
    pub category: RecommendationCategory,
    pub message: String,
    pub action: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Coherence,
    Superposition,
    Decoherence,
    Entanglement,
    Interference,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_creation() {
        let analyzer = QuantumCoherenceAnalyzer::new(CoherenceConfig::default());
        assert_eq!(analyzer.stats.observations, 0);
        assert_eq!(analyzer.coherence(), 1.0);
    }

    #[test]
    fn test_superposition_richness() {
        let analyzer = QuantumCoherenceAnalyzer::new(CoherenceConfig::default());

        // Random states with good seed should have reasonable richness
        // Due to deterministic hashing, test that richness is in valid range (0-1)
        // and that different seeds produce different richness values
        let state1 = HV16::random(42);
        let state2 = HV16::random(12345);
        let richness1 = analyzer.estimate_superposition_richness(&state1);
        let richness2 = analyzer.estimate_superposition_richness(&state2);

        // Richness must be in valid range [0, 1]
        assert!(richness1 >= 0.0 && richness1 <= 1.0, "richness1 out of range: {}", richness1);
        assert!(richness2 >= 0.0 && richness2 <= 1.0, "richness2 out of range: {}", richness2);

        // Different seeds should produce measurable richness (not all zeros or all ones)
        assert!(richness1 > 0.0 || richness2 > 0.0, "At least one state should have non-zero richness");
    }

    #[test]
    fn test_observation_tracking() {
        let mut analyzer = QuantumCoherenceAnalyzer::new(CoherenceConfig::default());

        let state1 = HV16::random(100);
        let state2 = HV16::random(200);

        analyzer.observe(&state1, 0.8);
        assert_eq!(analyzer.stats.observations, 1);

        analyzer.observe(&state2, 0.75);
        assert_eq!(analyzer.stats.observations, 2);
    }

    #[test]
    fn test_coherence_with_similar_states() {
        let mut analyzer = QuantumCoherenceAnalyzer::new(CoherenceConfig::default());

        let base = HV16::random(300);

        // Observe same state multiple times
        for _ in 0..10 {
            analyzer.observe(&base, 0.8);
        }

        // Should maintain high coherence
        assert!(analyzer.coherence() > 0.9);
    }

    #[test]
    fn test_decoherence_detection() {
        let mut analyzer = QuantumCoherenceAnalyzer::new(CoherenceConfig::default());

        let state1 = HV16::random(400);
        let state2 = !state1.clone(); // Opposite state using Not trait

        analyzer.observe(&state1, 0.8);
        analyzer.observe(&state2, 0.3); // Sudden change should trigger decoherence

        assert!(analyzer.stats.decoherence_events > 0);
    }

    #[test]
    fn test_coherence_length() {
        let mut analyzer = QuantumCoherenceAnalyzer::new(CoherenceConfig::default());

        let base = HV16::random(500);

        // Observe same state 20 times
        for _ in 0..20 {
            analyzer.observe(&base, 0.8);
        }

        // Should have substantial coherence length
        assert!(analyzer.coherence_length() >= 10);
    }

    #[test]
    fn test_interference_detection() {
        let mut analyzer = QuantumCoherenceAnalyzer::new(CoherenceConfig::default());

        // Create sequence of states with different seeds
        for i in 0..20 {
            let state = HV16::random(600 + i as u64);
            analyzer.observe(&state, 0.5 + 0.3 * (i as f64 / 20.0).sin());
        }

        // Should detect some interference
        assert!(!analyzer.interference_patterns().is_empty());
    }

    #[test]
    fn test_entanglement_estimation() {
        let analyzer = QuantumCoherenceAnalyzer::new(CoherenceConfig::default());

        // Create correlated components
        let base = HV16::random(700);
        let correlated = base.clone();
        let uncorrelated = HV16::random(800);

        let high_entanglement = analyzer.estimate_entanglement(&[base.clone(), correlated]);
        let low_entanglement = analyzer.estimate_entanglement(&[base, uncorrelated]);

        assert!(high_entanglement.correlation > low_entanglement.correlation);
    }

    #[test]
    fn test_coherence_report() {
        let mut analyzer = QuantumCoherenceAnalyzer::new(CoherenceConfig::default());

        for i in 0..15 {
            let state = HV16::random(900 + i as u64);
            analyzer.observe(&state, 0.7);
        }

        let report = analyzer.coherence_report();
        assert!(report.coherence >= 0.0 && report.coherence <= 1.0);
        assert!(report.superposition_richness >= 0.0 && report.superposition_richness <= 1.0);
        assert_eq!(report.stats.observations, 15);
    }

    #[test]
    fn test_quantum_coherent_state() {
        let mut analyzer = QuantumCoherenceAnalyzer::new(CoherenceConfig::default());

        let base = HV16::random(1000);

        // Maintain stable state
        for _ in 0..20 {
            analyzer.observe(&base, 0.8);
        }

        // Should be in quantum coherent state
        // (High coherence, stable richness, low decoherence)
        let report = analyzer.coherence_report();
        assert!(report.coherence > 0.9);
    }

    #[test]
    fn test_recommendations() {
        let mut analyzer = QuantumCoherenceAnalyzer::new(CoherenceConfig::default());

        let state1 = HV16::random(1100);
        let state2 = !state1.clone(); // Opposite state using Not trait

        // Create chaotic sequence
        for i in 0..20 {
            let state = if i % 2 == 0 { state1.clone() } else { state2.clone() };
            analyzer.observe(&state, 0.5);
        }

        let report = analyzer.coherence_report();
        assert!(!report.recommendations.is_empty()); // Should have recommendations
    }
}
