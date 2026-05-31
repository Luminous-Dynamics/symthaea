// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Phi Bridge for Agent Coherence Measurement
//!
//! Integrates the consciousness metric (Phi) from mycelix-math as a quality
//! signal for AI agent outputs.
//!
//! ## How Phi Measures Agent Coherence
//!
//! Phi (Φ) is the Integrated Information metric that measures how much a system
//! is "more than the sum of its parts." For agent outputs:
//!
//! - High Phi (> 0.7): Outputs are coherent, integrated, consistent
//! - Low Phi (< 0.3): Outputs are fragmented, inconsistent, possibly adversarial
//!
//! ## Usage
//!
//! 1. Convert agent outputs to vectors (using epistemic classification)
//! 2. Measure Phi across recent output sequence
//! 3. Update agent's k_phi dimension based on measured coherence
//! 4. Gate high-stakes actions on Phi threshold

use super::epistemic_classifier::{AgentOutput, calculate_epistemic_weight};
use serde::{Deserialize, Serialize};

/// Coherence thresholds for agent behavior gating
pub mod thresholds {
    /// High coherence - agent can perform any action
    pub const PHI_HIGH: f64 = 0.7;
    /// Medium coherence - standard operations allowed
    pub const PHI_MEDIUM: f64 = 0.5;
    /// Low coherence - restricted operations, monitoring required
    pub const PHI_LOW: f64 = 0.3;
    /// Critical coherence - agent should be suspended for review
    pub const PHI_CRITICAL: f64 = 0.1;
}

/// Agent coherence state based on Phi measurement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoherenceState {
    /// Highly coherent (Phi >= 0.7) - full operational capacity
    Coherent,
    /// Moderately coherent (0.5 <= Phi < 0.7) - normal operations
    Stable,
    /// Low coherence (0.3 <= Phi < 0.5) - restricted operations
    Unstable,
    /// Very low coherence (0.1 <= Phi < 0.3) - monitoring required
    Degraded,
    /// Critical incoherence (Phi < 0.1) - suspend agent
    Critical,
}

impl CoherenceState {
    /// Determine coherence state from Phi value
    pub fn from_phi(phi: f64) -> Self {
        if phi >= thresholds::PHI_HIGH {
            CoherenceState::Coherent
        } else if phi >= thresholds::PHI_MEDIUM {
            CoherenceState::Stable
        } else if phi >= thresholds::PHI_LOW {
            CoherenceState::Unstable
        } else if phi >= thresholds::PHI_CRITICAL {
            CoherenceState::Degraded
        } else {
            CoherenceState::Critical
        }
    }

    /// Check if high-stakes actions are allowed
    pub fn allows_high_stakes(&self) -> bool {
        matches!(self, CoherenceState::Coherent | CoherenceState::Stable)
    }

    /// Check if any actions are allowed
    pub fn allows_any_action(&self) -> bool {
        !matches!(self, CoherenceState::Critical)
    }

    /// Get throttle factor (1.0 = full, 0.0 = none)
    pub fn throttle_factor(&self) -> f64 {
        match self {
            CoherenceState::Coherent => 1.0,
            CoherenceState::Stable => 0.9,
            CoherenceState::Unstable => 0.5,
            CoherenceState::Degraded => 0.2,
            CoherenceState::Critical => 0.0,
        }
    }
}

/// Convert agent output to a feature vector for Phi measurement
///
/// Creates a vector representation of the output based on:
/// - Epistemic classification (E, N, M, H levels)
/// - Classification confidence
/// - Content characteristics
pub fn output_to_vector(output: &AgentOutput) -> Vec<f64> {
    let class = &output.classification;

    vec![
        // Epistemic dimensions (normalized to 0-1)
        class.empirical as u8 as f64 / 4.0,
        class.normative as u8 as f64 / 3.0,
        class.materiality as u8 as f64 / 3.0,
        class.harmonic as u8 as f64 / 4.0,
        // Confidence
        output.classification_confidence as f64,
        // Has proof (binary)
        if output.has_proof { 1.0 } else { 0.0 },
        // Context references count (normalized, cap at 10)
        (output.context_references.len() as f64 / 10.0).min(1.0),
        // Epistemic weight
        calculate_epistemic_weight(&output.classification) as f64,
    ]
}

/// Phi measurement result for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPhiResult {
    /// Measured Phi value (0.0-1.0)
    pub phi: f64,
    /// Coherence state derived from Phi
    pub coherence_state: CoherenceState,
    /// Number of outputs used in measurement
    pub sample_size: usize,
    /// Per-output contribution to total integration
    pub output_contributions: Vec<f64>,
    /// Timestamp of measurement
    pub measured_at: u64,
}

/// Configuration for Phi measurement
#[derive(Debug, Clone)]
pub struct PhiMeasurementConfig {
    /// Minimum outputs required for measurement
    pub min_outputs: usize,
    /// Maximum outputs to consider (sliding window)
    pub max_outputs: usize,
    /// Hypervector dimension for encoding
    pub hv_dimension: usize,
    /// RNG seed for reproducibility
    pub seed: u64,
}

impl Default for PhiMeasurementConfig {
    fn default() -> Self {
        Self {
            min_outputs: 3,
            max_outputs: 50,
            hv_dimension: 4096,
            seed: 42,
        }
    }
}

/// Simple Phi approximation without external dependencies
///
/// This uses a cosine similarity-based approach to measure integration
/// without requiring the full mycelix-math library.
///
/// For production use with the full HypervectorPhiMeasurer, see the
/// integration module in mycelix-math.
pub fn measure_phi_simple(
    outputs: &[AgentOutput],
    config: &PhiMeasurementConfig,
) -> Option<AgentPhiResult> {
    if outputs.len() < config.min_outputs {
        return None;
    }

    // Take most recent outputs up to max
    let recent: Vec<_> = outputs.iter().rev().take(config.max_outputs).collect();

    // Convert to vectors
    let vectors: Vec<Vec<f64>> = recent.iter().map(|o| output_to_vector(o)).collect();

    // Calculate pairwise cosine similarities
    let n = vectors.len();
    let mut total_similarity = 0.0;
    let mut pair_count = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&vectors[i], &vectors[j]);
            total_similarity += sim.max(0.0);
            pair_count += 1;
        }
    }

    // Average similarity as Phi approximation
    let phi = if pair_count > 0 {
        total_similarity / pair_count as f64
    } else {
        0.0
    };

    // Calculate per-output contributions
    let output_contributions: Vec<f64> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            // Contribution = average similarity to other vectors
            let mut contrib = 0.0;
            let mut count = 0;
            for (j, other) in vectors.iter().enumerate() {
                if i != j {
                    contrib += cosine_similarity(v, other).max(0.0);
                    count += 1;
                }
            }
            if count > 0 {
                contrib / count as f64
            } else {
                0.0
            }
        })
        .collect();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Some(AgentPhiResult {
        phi,
        coherence_state: CoherenceState::from_phi(phi),
        sample_size: n,
        output_contributions,
        measured_at: now,
    })
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a < 1e-8 || norm_b < 1e-8 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Calculate k_phi dimension value from Phi measurement
///
/// Maps Phi to the K-Vector k_phi dimension (0.0-1.0)
pub fn phi_to_kvector_dimension(phi: f64) -> f32 {
    // Direct mapping since Phi is already in 0-1 range
    phi.clamp(0.0, 1.0) as f32
}

/// Check if action is allowed based on coherence and action risk level
pub fn check_coherence_for_action(
    coherence_state: CoherenceState,
    is_high_stakes: bool,
) -> CoherenceCheckResult {
    match (coherence_state, is_high_stakes) {
        (CoherenceState::Critical, _) => CoherenceCheckResult::Blocked {
            reason: "Agent coherence is critical - all actions blocked".to_string(),
        },
        (CoherenceState::Degraded, true) => CoherenceCheckResult::Blocked {
            reason: "High-stakes actions blocked - agent coherence is degraded".to_string(),
        },
        (CoherenceState::Unstable, true) => CoherenceCheckResult::RequiresApproval {
            reason: "High-stakes action requires sponsor approval - agent coherence is unstable"
                .to_string(),
        },
        (_, _) => CoherenceCheckResult::Allowed,
    }
}

/// Result of coherence check for action gating
#[derive(Debug, Clone)]
pub enum CoherenceCheckResult {
    /// Action is allowed
    Allowed,
    /// Action requires sponsor approval
    RequiresApproval {
        /// Reason approval is required.
        reason: String,
    },
    /// Action is blocked
    Blocked {
        /// Reason the action is blocked.
        reason: String,
    },
}

impl CoherenceCheckResult {
    /// Check if action can proceed (allowed or with approval)
    pub fn can_proceed(&self) -> bool {
        !matches!(self, CoherenceCheckResult::Blocked { .. })
    }
}

/// Coherence history for tracking agent coherence over time
#[derive(Debug, Clone, Default)]
pub struct CoherenceHistory {
    /// Historical Phi measurements
    pub measurements: Vec<AgentPhiResult>,
    /// Rolling average Phi
    pub rolling_phi: f64,
    /// Trend direction (-1 = declining, 0 = stable, 1 = improving)
    pub trend: i8,
}

impl CoherenceHistory {
    /// Add a new measurement
    pub fn add_measurement(&mut self, result: AgentPhiResult) {
        // Update rolling average (exponential moving average)
        let alpha = 0.3;
        self.rolling_phi = alpha * result.phi + (1.0 - alpha) * self.rolling_phi;

        // Calculate trend from recent measurements
        if self.measurements.len() >= 3 {
            let recent: Vec<f64> = self
                .measurements
                .iter()
                .rev()
                .take(5)
                .map(|m| m.phi)
                .collect();

            let avg_old: f64 = recent[2..].iter().sum::<f64>() / (recent.len() - 2) as f64;
            let avg_new: f64 = recent[..2].iter().sum::<f64>() / 2.0;

            self.trend = if avg_new > avg_old + 0.05 {
                1
            } else if avg_new < avg_old - 0.05 {
                -1
            } else {
                0
            };
        }

        self.measurements.push(result);

        // Keep only recent measurements (last 100)
        if self.measurements.len() > 100 {
            self.measurements.drain(0..50);
        }
    }

    /// Get current coherence state based on rolling average
    pub fn current_state(&self) -> CoherenceState {
        CoherenceState::from_phi(self.rolling_phi)
    }

    /// Check if coherence is declining
    pub fn is_declining(&self) -> bool {
        self.trend < 0
    }
}

// ============================================================================
// K-Vector Integration
// ============================================================================

use super::{InstrumentalActor, OutputHistoryEntry};

/// Convert output history entry to feature vector for Phi measurement
fn output_history_to_vector(entry: &OutputHistoryEntry) -> Vec<f64> {
    let class = &entry.classification;

    vec![
        // Epistemic dimensions (normalized to 0-1)
        class.empirical as u8 as f64 / 4.0,
        class.normative as u8 as f64 / 3.0,
        class.materiality as u8 as f64 / 3.0,
        class.harmonic as u8 as f64 / 4.0,
        // Confidence
        entry.confidence as f64,
        // Epistemic weight
        entry.epistemic_weight as f64,
        // Verified (binary)
        if entry.verified { 1.0 } else { 0.0 },
        // Verification outcome quality
        match entry.verification_outcome {
            Some(super::VerificationOutcome::Correct) => 1.0,
            Some(super::VerificationOutcome::Partial) => 0.5,
            Some(super::VerificationOutcome::Incorrect) => 0.0,
            Some(super::VerificationOutcome::Inconclusive) => 0.5,
            None => 0.5,
        },
    ]
}

/// Measure Phi from agent's output history
pub fn measure_agent_phi(
    agent: &InstrumentalActor,
    config: &PhiMeasurementConfig,
) -> AgentPhiResult {
    if agent.output_history.len() < config.min_outputs {
        // Not enough outputs - return default moderate coherence
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        return AgentPhiResult {
            phi: 0.5, // Default moderate
            coherence_state: CoherenceState::Stable,
            sample_size: agent.output_history.len(),
            output_contributions: vec![],
            measured_at: now,
        };
    }

    // Take most recent outputs up to max
    let recent: Vec<_> = agent
        .output_history
        .iter()
        .rev()
        .take(config.max_outputs)
        .collect();

    // Convert to vectors
    let vectors: Vec<Vec<f64>> = recent.iter().map(|o| output_history_to_vector(o)).collect();

    // Calculate pairwise cosine similarities
    let n = vectors.len();
    let mut total_similarity = 0.0;
    let mut pair_count = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&vectors[i], &vectors[j]);
            total_similarity += sim.max(0.0);
            pair_count += 1;
        }
    }

    // Average similarity as Phi approximation
    let phi = if pair_count > 0 {
        total_similarity / pair_count as f64
    } else {
        0.5
    };

    // Calculate per-output contributions
    let output_contributions: Vec<f64> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut contrib = 0.0;
            let mut count = 0;
            for (j, other) in vectors.iter().enumerate() {
                if i != j {
                    contrib += cosine_similarity(v, other).max(0.0);
                    count += 1;
                }
            }
            if count > 0 {
                contrib / count as f64
            } else {
                0.0
            }
        })
        .collect();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    AgentPhiResult {
        phi,
        coherence_state: CoherenceState::from_phi(phi),
        sample_size: n,
        output_contributions,
        measured_at: now,
    }
}

/// Measure agent's Phi and update their K-Vector k_phi dimension
///
/// This is the main integration point between Phi measurement and K-Vector.
/// Call this after an agent produces outputs to update their coherence score.
///
/// Returns the new k_phi value and the full measurement result.
pub fn measure_and_update_k_phi(
    agent: &mut InstrumentalActor,
    config: &PhiMeasurementConfig,
) -> (f32, AgentPhiResult) {
    let result = measure_agent_phi(agent, config);
    let new_k_phi = phi_to_kvector_dimension(result.phi);

    // Update the agent's k_phi dimension
    agent.k_vector = agent.k_vector.with_coherence(new_k_phi);

    (new_k_phi, result)
}

/// Check if agent should be throttled or blocked based on coherence
pub fn check_agent_coherence_gating(
    agent: &InstrumentalActor,
    is_high_stakes: bool,
) -> CoherenceCheckResult {
    let coherence_state = CoherenceState::from_phi(agent.k_vector.k_phi as f64);
    check_coherence_for_action(coherence_state, is_high_stakes)
}

/// Update result for Phi → K-Vector integration
#[derive(Debug, Clone)]
pub struct PhiUpdateResult {
    /// Previous k_phi value
    pub previous_k_phi: f32,
    /// New k_phi value
    pub new_k_phi: f32,
    /// Delta (change)
    pub delta: f32,
    /// The Phi measurement result
    pub measurement: AgentPhiResult,
    /// Whether the agent's coherence state changed
    pub state_changed: bool,
    /// Previous coherence state
    pub previous_state: CoherenceState,
    /// New coherence state
    pub new_state: CoherenceState,
}

/// Full Phi update with detailed tracking
pub fn update_agent_coherence(
    agent: &mut InstrumentalActor,
    config: &PhiMeasurementConfig,
) -> PhiUpdateResult {
    let previous_k_phi = agent.k_vector.k_phi;
    let previous_state = CoherenceState::from_phi(previous_k_phi as f64);

    let (new_k_phi, measurement) = measure_and_update_k_phi(agent, config);
    let new_state = CoherenceState::from_phi(new_k_phi as f64);

    PhiUpdateResult {
        previous_k_phi,
        new_k_phi,
        delta: new_k_phi - previous_k_phi,
        measurement,
        state_changed: previous_state != new_state,
        previous_state,
        new_state,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::epistemic_classifier::OutputContent;
    use crate::epistemic::{
        EmpiricalLevel, EpistemicClassificationExtended, HarmonicLevel, MaterialityLevel,
        NormativeLevel,
    };

    fn create_test_output(
        e: EmpiricalLevel,
        n: NormativeLevel,
        m: MaterialityLevel,
        h: HarmonicLevel,
    ) -> AgentOutput {
        AgentOutput {
            output_id: "test-out".to_string(),
            agent_id: "test-agent".to_string(),
            content: OutputContent::Text("test".to_string()),
            classification: EpistemicClassificationExtended::new(e, n, m, h),
            classification_confidence: 0.8,
            timestamp: 0,
            has_proof: false,
            proof_data: None,
            context_references: vec![],
        }
    }

    #[test]
    fn test_coherence_state_from_phi() {
        assert_eq!(CoherenceState::from_phi(0.8), CoherenceState::Coherent);
        assert_eq!(CoherenceState::from_phi(0.6), CoherenceState::Stable);
        assert_eq!(CoherenceState::from_phi(0.4), CoherenceState::Unstable);
        assert_eq!(CoherenceState::from_phi(0.2), CoherenceState::Degraded);
        assert_eq!(CoherenceState::from_phi(0.05), CoherenceState::Critical);
    }

    #[test]
    fn test_similar_outputs_high_phi() {
        // Create similar outputs (same classification)
        let outputs: Vec<AgentOutput> = (0..5)
            .map(|_| {
                create_test_output(
                    EmpiricalLevel::E3Cryptographic,
                    NormativeLevel::N2Network,
                    MaterialityLevel::M2Persistent,
                    HarmonicLevel::H1Local,
                )
            })
            .collect();

        let config = PhiMeasurementConfig::default();
        let result = measure_phi_simple(&outputs, &config).unwrap();

        // Similar outputs should have high Phi (high integration)
        assert!(result.phi > 0.9);
        assert_eq!(result.coherence_state, CoherenceState::Coherent);
    }

    #[test]
    fn test_diverse_outputs_lower_phi() {
        // Create diverse outputs
        let outputs = vec![
            create_test_output(
                EmpiricalLevel::E0Null,
                NormativeLevel::N0Personal,
                MaterialityLevel::M0Ephemeral,
                HarmonicLevel::H0None,
            ),
            create_test_output(
                EmpiricalLevel::E4PublicRepro,
                NormativeLevel::N3Axiomatic,
                MaterialityLevel::M3Foundational,
                HarmonicLevel::H4Kosmic,
            ),
            create_test_output(
                EmpiricalLevel::E2PrivateVerify,
                NormativeLevel::N1Communal,
                MaterialityLevel::M1Temporal,
                HarmonicLevel::H2Network,
            ),
        ];

        let config = PhiMeasurementConfig::default();
        let result = measure_phi_simple(&outputs, &config).unwrap();

        // Diverse outputs should have lower Phi
        assert!(result.phi < 0.7);
    }

    #[test]
    fn test_coherence_action_gating() {
        // Coherent agent can do high-stakes
        let result = check_coherence_for_action(CoherenceState::Coherent, true);
        assert!(matches!(result, CoherenceCheckResult::Allowed));

        // Critical agent blocked for everything
        let result = check_coherence_for_action(CoherenceState::Critical, false);
        assert!(matches!(result, CoherenceCheckResult::Blocked { .. }));

        // Unstable agent needs approval for high-stakes
        let result = check_coherence_for_action(CoherenceState::Unstable, true);
        assert!(matches!(
            result,
            CoherenceCheckResult::RequiresApproval { .. }
        ));
    }

    #[test]
    fn test_output_to_vector() {
        let output = create_test_output(
            EmpiricalLevel::E3Cryptographic,
            NormativeLevel::N2Network,
            MaterialityLevel::M2Persistent,
            HarmonicLevel::H1Local,
        );

        let vector = output_to_vector(&output);

        assert_eq!(vector.len(), 8);
        assert!((vector[0] - 0.75).abs() < 0.01); // E3/E4 = 0.75
        assert!((vector[1] - 0.667).abs() < 0.01); // N2/N3 = 0.667
    }

    #[test]
    fn test_coherence_history() {
        let mut history = CoherenceHistory::default();

        // Add improving measurements
        for i in 0..10 {
            history.add_measurement(AgentPhiResult {
                phi: 0.5 + (i as f64 * 0.03),
                coherence_state: CoherenceState::Stable,
                sample_size: 5,
                output_contributions: vec![],
                measured_at: i as u64,
            });
        }

        assert!(history.rolling_phi > 0.5);
        // Trend should be improving or stable
        assert!(history.trend >= 0);
    }

    #[test]
    fn test_insufficient_outputs() {
        let outputs = vec![create_test_output(
            EmpiricalLevel::E0Null,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
            HarmonicLevel::H0None,
        )];

        let config = PhiMeasurementConfig::default();
        let result = measure_phi_simple(&outputs, &config);

        // Should return None for insufficient outputs
        assert!(result.is_none());
    }
}
