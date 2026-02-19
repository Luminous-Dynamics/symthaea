//! # Coherence Bridge for Agent Coherence Gating
//!
//! **THIS IS THE PRIMARY LEGITIMATE USE OF COHERENCE IN THE CODEBASE.**
//!
//! This module provides coherence gating for high-stakes operations. Coherence measures
//! output consistency - an agent producing wildly inconsistent outputs should be
//! prevented from performing critical ZK operations.
//!
//! ## When to Use Coherence
//!
//! - **Gating ZK operations** - prevent incoherent agents from generating proofs
//! - **Byzantine consensus participation** - only coherent agents should vote
//! - **One-time coherence checks** - before critical decisions
//!
//! ## When NOT to Use Coherence
//!
//! - **Trust scoring** - k_coherence has 0% weight, use trust_score() instead
//! - **Query routing** - use trust score, not coherence
//! - **Dashboards** - collective coherence is informational only, don't gate on it
//! - **Emergent behavior detection** - use simple clustering instead
//!
//! ## Coherence States
//!
//! - `Coherent` (>= 0.7): All operations allowed
//! - `Stable` (0.5-0.7): Normal operations allowed
//! - `Unstable` (0.3-0.5): Restricted operations
//! - `Degraded` (0.1-0.3): Monitoring required
//! - `Critical` (< 0.1): Agent suspended

use super::epistemic_classifier::{calculate_epistemic_weight, AgentOutput};
use serde::{Deserialize, Serialize};

/// Coherence thresholds for agent behavior gating
pub mod thresholds {
    /// High coherence - agent can perform any action
    pub const COHERENCE_HIGH: f64 = 0.7;
    /// Medium coherence - standard operations allowed
    pub const COHERENCE_MEDIUM: f64 = 0.5;
    /// Low coherence - restricted operations, monitoring required
    pub const COHERENCE_LOW: f64 = 0.3;
    /// Critical coherence - agent should be suspended for review
    pub const COHERENCE_CRITICAL: f64 = 0.1;
}

/// Agent coherence state based on coherence measurement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoherenceState {
    /// Highly coherent (>= 0.7) - full operational capacity
    Coherent,
    /// Moderately coherent (0.5 <= value < 0.7) - normal operations
    Stable,
    /// Low coherence (0.3 <= value < 0.5) - restricted operations
    Unstable,
    /// Very low coherence (0.1 <= value < 0.3) - monitoring required
    Degraded,
    /// Critical incoherence (< 0.1) - suspend agent
    Critical,
}

impl CoherenceState {
    /// Determine coherence state from coherence value
    pub fn from_phi(phi: f64) -> Self {
        if phi >= thresholds::COHERENCE_HIGH {
            CoherenceState::Coherent
        } else if phi >= thresholds::COHERENCE_MEDIUM {
            CoherenceState::Stable
        } else if phi >= thresholds::COHERENCE_LOW {
            CoherenceState::Unstable
        } else if phi >= thresholds::COHERENCE_CRITICAL {
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

/// Convert agent output to a feature vector for coherence measurement
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

/// Coherence measurement result for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCoherenceResult {
    /// Measured coherence value (0.0-1.0)
    pub coherence: f64,
    /// Coherence state derived from measurement
    pub coherence_state: CoherenceState,
    /// Number of outputs used in measurement
    pub sample_size: usize,
    /// Per-output contribution to total integration
    pub output_contributions: Vec<f64>,
    /// Timestamp of measurement
    pub measured_at: u64,
}

/// Configuration for coherence measurement
#[derive(Debug, Clone)]
pub struct CoherenceMeasurementConfig {
    /// Minimum outputs required for measurement
    pub min_outputs: usize,
    /// Maximum outputs to consider (sliding window)
    pub max_outputs: usize,
    /// Hypervector dimension for encoding
    pub hv_dimension: usize,
    /// RNG seed for reproducibility
    pub seed: u64,
}

impl Default for CoherenceMeasurementConfig {
    fn default() -> Self {
        Self {
            min_outputs: 3,
            max_outputs: 50,
            hv_dimension: 4096,
            seed: 42,
        }
    }
}

/// Simple coherence approximation without external dependencies
///
/// This uses a cosine similarity-based approach to measure integration
/// without requiring the full mycelix-math library.
///
/// For production use with the full HypervectorPhiMeasurer, see the
/// integration module in mycelix-math.
pub fn measure_coherence(
    outputs: &[AgentOutput],
    config: &CoherenceMeasurementConfig,
) -> Option<AgentCoherenceResult> {
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

    // Average similarity as coherence approximation
    let coherence = if pair_count > 0 {
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

    Some(AgentCoherenceResult {
        coherence,
        coherence_state: CoherenceState::from_phi(coherence),
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

/// Calculate k_coherence dimension value from coherence measurement
///
/// Maps coherence to the K-Vector k_coherence dimension (0.0-1.0)
pub fn coherence_to_kvector_dimension(coherence: f64) -> f32 {
    // Direct mapping since coherence is already in 0-1 range
    coherence.clamp(0.0, 1.0) as f32
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
    /// Historical coherence measurements
    pub measurements: Vec<AgentCoherenceResult>,
    /// Rolling average coherence
    pub rolling_coherence: f64,
    /// Trend direction (-1 = declining, 0 = stable, 1 = improving)
    pub trend: i8,
}

impl CoherenceHistory {
    /// Add a new measurement
    pub fn add_measurement(&mut self, result: AgentCoherenceResult) {
        // Update rolling average (exponential moving average)
        let alpha = 0.3;
        self.rolling_coherence = alpha * result.coherence + (1.0 - alpha) * self.rolling_coherence;

        // Calculate trend from recent measurements
        if self.measurements.len() >= 3 {
            let recent: Vec<f64> = self
                .measurements
                .iter()
                .rev()
                .take(5)
                .map(|m| m.coherence)
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
        CoherenceState::from_phi(self.rolling_coherence)
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

/// Convert output history entry to feature vector for coherence measurement
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

/// Measure coherence from agent's output history
pub fn measure_agent_coherence(
    agent: &InstrumentalActor,
    config: &CoherenceMeasurementConfig,
) -> AgentCoherenceResult {
    if agent.output_history.len() < config.min_outputs {
        // Not enough outputs - return default moderate coherence
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        return AgentCoherenceResult {
            coherence: 0.5, // Default moderate
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

    // Average similarity as coherence approximation
    let coherence = if pair_count > 0 {
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

    AgentCoherenceResult {
        coherence,
        coherence_state: CoherenceState::from_phi(coherence),
        sample_size: n,
        output_contributions,
        measured_at: now,
    }
}

/// Measure agent's coherence and update their K-Vector k_coherence dimension
///
/// This is the main integration point between coherence measurement and K-Vector.
/// Call this after an agent produces outputs to update their coherence score.
///
/// Returns the new k_coherence value and the full measurement result.
pub fn measure_and_update_k_coherence(
    agent: &mut InstrumentalActor,
    config: &CoherenceMeasurementConfig,
) -> (f32, AgentCoherenceResult) {
    let result = measure_agent_coherence(agent, config);
    let new_k_coherence = coherence_to_kvector_dimension(result.coherence);

    // Update the agent's k_coherence dimension
    agent.k_vector = agent.k_vector.with_coherence(new_k_coherence);

    (new_k_coherence, result)
}

/// Check if agent should be throttled or blocked based on coherence
pub fn check_agent_coherence_gating(
    agent: &InstrumentalActor,
    is_high_stakes: bool,
) -> CoherenceCheckResult {
    let coherence_state = CoherenceState::from_phi(agent.k_vector.k_coherence as f64);
    check_coherence_for_action(coherence_state, is_high_stakes)
}

/// Update result for coherence -> K-Vector integration
#[derive(Debug, Clone)]
pub struct CoherenceUpdateResult {
    /// Previous k_coherence value
    pub previous_k_coherence: f32,
    /// New k_coherence value
    pub new_k_coherence: f32,
    /// Delta (change)
    pub delta: f32,
    /// The coherence measurement result
    pub measurement: AgentCoherenceResult,
    /// Whether the agent's coherence state changed
    pub state_changed: bool,
    /// Previous coherence state
    pub previous_state: CoherenceState,
    /// New coherence state
    pub new_state: CoherenceState,
}

/// Full coherence update with detailed tracking
pub fn update_agent_coherence(
    agent: &mut InstrumentalActor,
    config: &CoherenceMeasurementConfig,
) -> CoherenceUpdateResult {
    let previous_k_coherence = agent.k_vector.k_coherence;
    let previous_state = CoherenceState::from_phi(previous_k_coherence as f64);

    let (new_k_coherence, measurement) = measure_and_update_k_coherence(agent, config);
    let new_state = CoherenceState::from_phi(new_k_coherence as f64);

    CoherenceUpdateResult {
        previous_k_coherence,
        new_k_coherence,
        delta: new_k_coherence - previous_k_coherence,
        measurement,
        state_changed: previous_state != new_state,
        previous_state,
        new_state,
    }
}

// ============================================================================
// ZK Operation Coherence Gating
// ============================================================================

/// ZK operation types that can be gated by coherence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZKOperationType {
    /// Generate a trust proof
    GenerateProof,
    /// Verify a trust proof
    VerifyProof,
    /// Create K-Vector commitment
    CreateCommitment,
    /// Prove trust improvement
    ProveImprovement,
    /// Aggregate proofs (high stakes)
    AggregateProofs,
    /// Participate in Byzantine consensus
    ByzantineConsensus,
}

impl ZKOperationType {
    /// Get the minimum coherence state required for this operation
    pub fn required_coherence(&self) -> CoherenceState {
        match self {
            // Low-stakes: available to most agents
            Self::VerifyProof => CoherenceState::Degraded,
            Self::CreateCommitment => CoherenceState::Unstable,
            // Medium-stakes: require stable coherence
            Self::GenerateProof => CoherenceState::Stable,
            Self::ProveImprovement => CoherenceState::Stable,
            // High-stakes: require good coherence
            Self::AggregateProofs => CoherenceState::Stable,
            Self::ByzantineConsensus => CoherenceState::Coherent,
        }
    }

    /// Check if this is a high-stakes operation
    pub fn is_high_stakes(&self) -> bool {
        matches!(self, Self::AggregateProofs | Self::ByzantineConsensus)
    }
}

/// Result of ZK operation coherence gating check
#[derive(Debug, Clone)]
pub struct ZKCoherenceGatingResult {
    /// Whether the operation is permitted
    pub permitted: bool,
    /// Current coherence state
    pub current_state: CoherenceState,
    /// Required coherence state
    pub required_state: CoherenceState,
    /// Current coherence value
    pub current_coherence: f64,
    /// Recommendation for the agent
    pub recommendation: ZKGatingRecommendation,
}

/// Recommendations for ZK gating decisions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZKGatingRecommendation {
    /// Proceed with operation
    Proceed,
    /// Wait for coherence to improve
    WaitForCoherence,
    /// Use a lower-stakes alternative
    UseLowerStakesAlternative,
    /// Escalate to sponsor for approval
    EscalateToSponsor,
    /// Operation denied
    Deny,
}

/// Check if a ZK operation is permitted based on agent's coherence
pub fn check_zk_operation_coherence(
    agent: &InstrumentalActor,
    operation: ZKOperationType,
) -> ZKCoherenceGatingResult {
    let current_coherence = agent.k_vector.k_coherence as f64;
    let current_state = CoherenceState::from_phi(current_coherence);
    let required_state = operation.required_coherence();

    // Compare states (higher state index = more restrictive)
    let state_order = |s: &CoherenceState| -> u8 {
        match s {
            CoherenceState::Coherent => 4,
            CoherenceState::Stable => 3,
            CoherenceState::Unstable => 2,
            CoherenceState::Degraded => 1,
            CoherenceState::Critical => 0,
        }
    };

    let permitted = state_order(&current_state) >= state_order(&required_state);

    let recommendation = if permitted {
        ZKGatingRecommendation::Proceed
    } else if current_state == CoherenceState::Critical {
        ZKGatingRecommendation::Deny
    } else if operation.is_high_stakes() {
        ZKGatingRecommendation::EscalateToSponsor
    } else if state_order(&current_state) + 1 >= state_order(&required_state) {
        // Close to threshold - might improve soon
        ZKGatingRecommendation::WaitForCoherence
    } else {
        ZKGatingRecommendation::UseLowerStakesAlternative
    };

    ZKCoherenceGatingResult {
        permitted,
        current_state,
        required_state,
        current_coherence,
        recommendation,
    }
}

/// Check if agent can participate in Byzantine consensus
pub fn check_byzantine_participation(agent: &InstrumentalActor) -> ZKCoherenceGatingResult {
    check_zk_operation_coherence(agent, ZKOperationType::ByzantineConsensus)
}

/// Check if agent can generate trust proofs
pub fn check_proof_generation(agent: &InstrumentalActor) -> ZKCoherenceGatingResult {
    check_zk_operation_coherence(agent, ZKOperationType::GenerateProof)
}

/// Check if agent can aggregate proofs
pub fn check_proof_aggregation(agent: &InstrumentalActor) -> ZKCoherenceGatingResult {
    check_zk_operation_coherence(agent, ZKOperationType::AggregateProofs)
}

// ============================================================================
// Coherence-Weighted Trust Updates
// ============================================================================

/// Calculate a coherence-weighted multiplier for trust updates
///
/// Higher coherence = updates have more impact
/// Lower coherence = updates are dampened
pub fn coherence_weight_multiplier(coherence: f64) -> f64 {
    // Non-linear scaling: sqrt gives more weight to moderate coherence
    // Range: 0.0 (coherence=0) to 1.0 (coherence=1)
    coherence.sqrt().clamp(0.0, 1.0)
}

/// Apply coherence-weighting to a K-Vector trust delta
pub fn apply_coherence_weighting_to_delta(base_delta: f32, coherence: f64) -> f32 {
    let multiplier = coherence_weight_multiplier(coherence);
    (base_delta * multiplier as f32).clamp(-0.5, 0.5)
}

/// Coherence-weighted attestation value
///
/// Attestations from high-coherence agents are weighted more heavily
/// in Byzantine aggregation and consensus.
pub fn coherence_weighted_attestation(attestation_value: f32, attester_coherence: f64) -> f32 {
    let weight = coherence_weight_multiplier(attester_coherence);
    attestation_value * weight as f32
}

// ============================================================================
// Observability Export
// ============================================================================

/// Export coherence metrics for observability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceExport {
    /// Agent ID
    pub agent_id: String,
    /// Current coherence value
    pub coherence: f64,
    /// Coherence state
    pub coherence_state: String,
    /// K-Vector k_coherence dimension
    pub k_coherence: f32,
    /// Sample size used for measurement
    pub sample_size: usize,
    /// Trend direction
    pub trend: i8,
    /// Rolling average
    pub rolling_coherence: f64,
    /// Can perform high-stakes actions
    pub can_high_stakes: bool,
    /// Is critical (blocked)
    pub is_critical: bool,
}

/// Export agent's coherence metrics
pub fn export_coherence_metrics(
    agent_id: &str,
    agent: &InstrumentalActor,
    history: &CoherenceHistory,
) -> CoherenceExport {
    let coherence = agent.k_vector.k_coherence as f64;
    let state = CoherenceState::from_phi(coherence);

    CoherenceExport {
        agent_id: agent_id.to_string(),
        coherence,
        coherence_state: format!("{:?}", state),
        k_coherence: agent.k_vector.k_coherence,
        sample_size: agent.output_history.len(),
        trend: history.trend,
        rolling_coherence: history.rolling_coherence,
        can_high_stakes: state.allows_high_stakes(),
        is_critical: matches!(state, CoherenceState::Critical),
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
    fn test_similar_outputs_high_coherence() {
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

        let config = CoherenceMeasurementConfig::default();
        let result = measure_coherence(&outputs, &config).unwrap();

        // Similar outputs should have high coherence (high integration)
        assert!(result.coherence > 0.9);
        assert_eq!(result.coherence_state, CoherenceState::Coherent);
    }

    #[test]
    fn test_diverse_outputs_lower_coherence() {
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

        let config = CoherenceMeasurementConfig::default();
        let result = measure_coherence(&outputs, &config).unwrap();

        // Diverse outputs should have lower coherence
        assert!(result.coherence < 0.7);
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
            history.add_measurement(AgentCoherenceResult {
                coherence: 0.5 + (i as f64 * 0.03),
                coherence_state: CoherenceState::Stable,
                sample_size: 5,
                output_contributions: vec![],
                measured_at: i as u64,
            });
        }

        assert!(history.rolling_coherence > 0.5);
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

        let config = CoherenceMeasurementConfig::default();
        let result = measure_coherence(&outputs, &config);

        // Should return None for insufficient outputs
        assert!(result.is_none());
    }

    // =========================================================================
    // ZK Coherence Gating Tests
    // =========================================================================

    use crate::agentic::epistemic_classifier::EpistemicStats;
    use crate::agentic::{
        AgentClass, AgentConstraints, AgentId, AgentStatus, UncertaintyCalibration,
    };
    use crate::matl::KVector;

    fn create_test_agent_with_coherence(coherence: f32) -> InstrumentalActor {
        InstrumentalActor {
            agent_id: AgentId::from_string("test-agent".to_string()),
            sponsor_did: "did:example:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: 5000,
            kredit_cap: 10000,
            constraints: AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: 1000,
            last_activity: 2000,
            actions_this_hour: 5,
            k_vector: KVector::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, coherence),
            epistemic_stats: EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        }
    }

    #[test]
    fn test_zk_operation_type_required_coherence() {
        // Low-stakes operations have lower requirements
        assert_eq!(
            ZKOperationType::VerifyProof.required_coherence(),
            CoherenceState::Degraded
        );

        // High-stakes operations require high coherence
        assert_eq!(
            ZKOperationType::ByzantineConsensus.required_coherence(),
            CoherenceState::Coherent
        );
    }

    #[test]
    fn test_zk_gating_coherent_agent() {
        let agent = create_test_agent_with_coherence(0.8); // Coherent

        // Should be able to do all operations
        let result = check_zk_operation_coherence(&agent, ZKOperationType::ByzantineConsensus);
        assert!(result.permitted);
        assert_eq!(result.recommendation, ZKGatingRecommendation::Proceed);

        let result = check_zk_operation_coherence(&agent, ZKOperationType::GenerateProof);
        assert!(result.permitted);
    }

    #[test]
    fn test_zk_gating_degraded_agent() {
        let agent = create_test_agent_with_coherence(0.2); // Degraded

        // Can verify proofs
        let result = check_zk_operation_coherence(&agent, ZKOperationType::VerifyProof);
        assert!(result.permitted);

        // Cannot participate in Byzantine consensus
        let result = check_zk_operation_coherence(&agent, ZKOperationType::ByzantineConsensus);
        assert!(!result.permitted);
        assert_eq!(
            result.recommendation,
            ZKGatingRecommendation::EscalateToSponsor
        );
    }

    #[test]
    fn test_zk_gating_critical_agent() {
        let agent = create_test_agent_with_coherence(0.05); // Critical

        // Cannot do anything
        let result = check_zk_operation_coherence(&agent, ZKOperationType::VerifyProof);
        assert!(!result.permitted);
        assert_eq!(result.recommendation, ZKGatingRecommendation::Deny);
    }

    #[test]
    fn test_coherence_weight_multiplier() {
        // High coherence = high weight
        assert!((coherence_weight_multiplier(1.0) - 1.0).abs() < 0.001);

        // Low coherence = low weight
        assert!(coherence_weight_multiplier(0.1) < 0.4);

        // Zero coherence = zero weight
        assert!(coherence_weight_multiplier(0.0).abs() < 0.001);
    }

    #[test]
    fn test_apply_coherence_weighting_to_delta() {
        // High coherence: delta passes through mostly unchanged
        let high_delta = apply_coherence_weighting_to_delta(0.1, 0.9);
        assert!(high_delta > 0.08);

        // Low coherence: delta is dampened
        let low_delta = apply_coherence_weighting_to_delta(0.1, 0.25);
        assert!(low_delta < 0.06);

        // Zero coherence: delta is zero
        let zero_delta = apply_coherence_weighting_to_delta(0.1, 0.0);
        assert!(zero_delta.abs() < 0.001);
    }

    #[test]
    fn test_coherence_weighted_attestation() {
        // High-coherence attester's attestation weighted more
        let high_attestation = coherence_weighted_attestation(1.0, 0.9);
        let low_attestation = coherence_weighted_attestation(1.0, 0.2);

        assert!(high_attestation > low_attestation);
        assert!(high_attestation > 0.9);
        assert!(low_attestation < 0.5);
    }

    #[test]
    fn test_export_coherence_metrics() {
        let agent = create_test_agent_with_coherence(0.65);
        let history = CoherenceHistory::default();

        let export = export_coherence_metrics("agent-123", &agent, &history);

        assert_eq!(export.agent_id, "agent-123");
        assert!((export.coherence - 0.65).abs() < 0.01);
        assert_eq!(export.coherence_state, "Stable");
        assert!(export.can_high_stakes);
        assert!(!export.is_critical);
    }

    #[test]
    fn test_byzantine_participation_check() {
        let coherent_agent = create_test_agent_with_coherence(0.75);
        let result = check_byzantine_participation(&coherent_agent);
        assert!(result.permitted);

        let unstable_agent = create_test_agent_with_coherence(0.4);
        let result = check_byzantine_participation(&unstable_agent);
        assert!(!result.permitted);
    }

    #[test]
    fn test_proof_generation_check() {
        let stable_agent = create_test_agent_with_coherence(0.55);
        let result = check_proof_generation(&stable_agent);
        assert!(result.permitted);

        let degraded_agent = create_test_agent_with_coherence(0.15);
        let result = check_proof_generation(&degraded_agent);
        assert!(!result.permitted);
    }
}
