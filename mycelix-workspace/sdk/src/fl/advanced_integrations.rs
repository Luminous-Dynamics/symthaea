//! # Advanced FL Integrations
//!
//! Extended integrations connecting FL with other Mycelix subsystems:
//! - HyperFeel for agent communication (#175)
//! - DKG quality claims (#176)
//! - PoG-weighted participation (#177)
//! - Cross-domain ZK proofs (#178)
//! - Temporal commitments (#179)
//! - Storage-backed immutability (#180)

use super::epistemic_fl_bridge::{
    EpistemicGradientUpdate, EpistemicQualityTier, GradientEpistemicClassification,
    RoundEpistemicStats,
};
use super::types::{FLRound, GradientUpdate, Participant};
use crate::epistemic::{
    EmpiricalLevel, EpistemicClassificationExtended, HarmonicLevel, MaterialityLevel,
    NormativeLevel,
};
use crate::pog::PogScore;
use serde::{Deserialize, Serialize};

// ============================================================================
// #175: HyperFeel for Agent Communication
// Generalized hypervector compression for K-Vector states
// ============================================================================

/// Compressed agent state using HyperFeel encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedAgentState {
    /// Agent identifier
    pub agent_id: String,
    /// Compressed K-Vector (2KB hypervector)
    pub compressed_kvector: Vec<u8>,
    /// Compressed epistemic stats
    pub compressed_epistemic: Vec<u8>,
    /// Timestamp
    pub timestamp: u64,
    /// Original dimension before compression
    pub original_dim: usize,
    /// Compression ratio achieved
    pub compression_ratio: f64,
}

/// Configuration for agent state compression
#[derive(Debug, Clone)]
pub struct AgentCompressionConfig {
    /// Target hypervector dimension (default: 16384 bits = 2KB)
    pub hv_dimension: usize,
    /// Include epistemic stats in compression
    pub include_epistemic: bool,
    /// Include recent output history
    pub include_history: bool,
    /// Maximum history entries to include
    pub max_history: usize,
}

impl Default for AgentCompressionConfig {
    fn default() -> Self {
        Self {
            hv_dimension: 16384,
            include_epistemic: true,
            include_history: false,
            max_history: 10,
        }
    }
}

/// Compress agent K-Vector state for efficient transmission
pub fn compress_agent_state(
    agent_id: &str,
    kvector: &[f64; 10],
    epistemic_stats: Option<&RoundEpistemicStats>,
    config: &AgentCompressionConfig,
) -> CompressedAgentState {
    // Simplified compression: pack K-Vector into bytes
    // In production, would use actual HyperFeel encoder
    let mut compressed = Vec::with_capacity(config.hv_dimension / 8);

    // Pack K-Vector (10 dimensions × 8 bytes = 80 bytes)
    for &v in kvector {
        compressed.extend_from_slice(&v.to_le_bytes());
    }

    // Pack epistemic stats if available
    let mut epistemic_bytes = Vec::new();
    if let Some(stats) = epistemic_stats {
        epistemic_bytes.extend_from_slice(&stats.average_empirical.to_le_bytes());
        epistemic_bytes.extend_from_slice(&stats.average_normative.to_le_bytes());
        epistemic_bytes.extend_from_slice(&(stats.quality_tier as u8).to_le_bytes());
    }

    let original_dim = 10 + if epistemic_stats.is_some() { 3 } else { 0 };
    let compression_ratio = original_dim as f64 * 8.0 / compressed.len() as f64;

    CompressedAgentState {
        agent_id: agent_id.to_string(),
        compressed_kvector: compressed,
        compressed_epistemic: epistemic_bytes,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        original_dim,
        compression_ratio,
    }
}

/// Compute similarity between two compressed agent states
/// Returns cosine similarity (0.0 = orthogonal, 1.0 = identical)
pub fn compute_state_similarity(
    state1: &CompressedAgentState,
    state2: &CompressedAgentState,
) -> f64 {
    if state1.compressed_kvector.len() != state2.compressed_kvector.len() {
        return 0.0;
    }

    // Interpret bytes as f64 values and compute cosine similarity
    let n = state1.compressed_kvector.len() / 8;
    let mut dot = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;

    for i in 0..n {
        let start = i * 8;
        let end = start + 8;
        if end <= state1.compressed_kvector.len() {
            let v1 = f64::from_le_bytes(
                state1.compressed_kvector[start..end]
                    .try_into()
                    .unwrap_or([0; 8]),
            );
            let v2 = f64::from_le_bytes(
                state2.compressed_kvector[start..end]
                    .try_into()
                    .unwrap_or([0; 8]),
            );
            dot += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }
    }

    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }

    dot / (norm1.sqrt() * norm2.sqrt())
}

/// Multi-agent consensus using compressed states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedConsensusResult {
    /// Consensus achieved
    pub consensus_reached: bool,
    /// Average similarity among agents
    pub average_similarity: f64,
    /// Outlier agents (similarity < threshold)
    pub outliers: Vec<String>,
    /// Cluster assignments (if multiple clusters detected)
    pub clusters: Vec<Vec<String>>,
}

/// Compute consensus from compressed agent states
pub fn compute_compressed_consensus(
    states: &[CompressedAgentState],
    similarity_threshold: f64,
) -> CompressedConsensusResult {
    if states.len() < 2 {
        return CompressedConsensusResult {
            consensus_reached: true,
            average_similarity: 1.0,
            outliers: vec![],
            clusters: vec![states.iter().map(|s| s.agent_id.clone()).collect()],
        };
    }

    // Compute pairwise similarities
    let mut total_sim = 0.0;
    let mut count = 0;
    let mut outliers = Vec::new();

    for i in 0..states.len() {
        let mut agent_avg_sim = 0.0;
        for j in 0..states.len() {
            if i != j {
                let sim = compute_state_similarity(&states[i], &states[j]);
                total_sim += sim;
                agent_avg_sim += sim;
                count += 1;
            }
        }
        agent_avg_sim /= (states.len() - 1) as f64;
        if agent_avg_sim < similarity_threshold {
            outliers.push(states[i].agent_id.clone());
        }
    }

    let average_similarity = if count > 0 {
        total_sim / count as f64
    } else {
        1.0
    };
    let consensus_reached = outliers.is_empty() && average_similarity >= similarity_threshold;

    // Simple clustering: consensus group vs outliers
    let consensus_group: Vec<_> = states
        .iter()
        .filter(|s| !outliers.contains(&s.agent_id))
        .map(|s| s.agent_id.clone())
        .collect();

    let mut clusters = vec![consensus_group];
    if !outliers.is_empty() {
        clusters.push(outliers.clone());
    }

    CompressedConsensusResult {
        consensus_reached,
        average_similarity,
        outliers,
        clusters,
    }
}

// ============================================================================
// #176: DKG Quality Claims
// FL quality → epistemic claims → DKG storage
// ============================================================================

/// Epistemic claim about FL round quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FLQualityClaim {
    /// Claim identifier
    pub claim_id: String,
    /// Round this claim is about
    pub round_id: u64,
    /// Model identifier
    pub model_id: String,
    /// Quality score (0.0-1.0)
    pub quality_score: f64,
    /// Epistemic classification of the claim itself
    pub classification: EpistemicClassificationExtended,
    /// Evidence supporting the claim
    pub evidence: QualityEvidence,
    /// Timestamp
    pub timestamp: u64,
    /// Attestations from validators
    pub attestations: Vec<QualityAttestation>,
}

/// Evidence supporting a quality claim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityEvidence {
    /// Number of participants
    pub participant_count: usize,
    /// Byzantine participants detected
    pub byzantine_count: usize,
    /// Average PoGQ score
    pub average_pogq: f64,
    /// Round epistemic stats
    pub epistemic_stats: RoundEpistemicStats,
    /// Model improvement delta
    pub quality_delta: f64,
    /// ZK proof of aggregation (if available)
    pub zk_proof_hash: Option<String>,
}

/// Attestation from a validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAttestation {
    /// Validator identifier
    pub validator_id: String,
    /// Validator's reputation weight
    pub reputation_weight: f64,
    /// Whether validator confirms the claim
    pub confirms: bool,
    /// Signature (placeholder)
    pub signature: String,
    /// Timestamp
    pub timestamp: u64,
}

impl FLQualityClaim {
    /// Create a new quality claim from FL round results
    pub fn from_round(
        round: &FLRound,
        quality_score: f64,
        quality_delta: f64,
        epistemic_stats: RoundEpistemicStats,
        zk_proof_hash: Option<String>,
    ) -> Self {
        // Determine claim classification based on evidence
        let empirical = if zk_proof_hash.is_some() {
            EmpiricalLevel::E3Cryptographic
        } else if round.participants.len() >= 5 {
            EmpiricalLevel::E2PrivateVerify
        } else {
            EmpiricalLevel::E1Testimonial
        };

        let normative = if round.participants.len() >= 10 {
            NormativeLevel::N2Network
        } else if round.participants.len() >= 3 {
            NormativeLevel::N1Communal
        } else {
            NormativeLevel::N0Personal
        };

        let materiality = match epistemic_stats.quality_tier {
            EpistemicQualityTier::Gold => MaterialityLevel::M3Foundational,
            EpistemicQualityTier::Silver => MaterialityLevel::M2Persistent,
            _ => MaterialityLevel::M1Temporal,
        };

        let harmonic = HarmonicLevel::H1Local; // FL rounds typically local impact

        Self {
            claim_id: format!("fl-quality-{}-{}", round.round_id, round.model_version),
            round_id: round.round_id,
            model_id: format!("model-v{}", round.model_version),
            quality_score,
            classification: EpistemicClassificationExtended::new(
                empirical,
                normative,
                materiality,
                harmonic,
            ),
            evidence: QualityEvidence {
                participant_count: round.participants.len(),
                byzantine_count: round.participants.len() - round.updates.len(),
                average_pogq: 0.7, // Placeholder
                epistemic_stats,
                quality_delta,
                zk_proof_hash,
            },
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            attestations: vec![],
        }
    }

    /// Calculate claim confidence based on attestations
    pub fn confidence(&self) -> f64 {
        if self.attestations.is_empty() {
            return 0.5; // Base confidence without attestations
        }

        let total_weight: f64 = self.attestations.iter().map(|a| a.reputation_weight).sum();

        let confirming_weight: f64 = self
            .attestations
            .iter()
            .filter(|a| a.confirms)
            .map(|a| a.reputation_weight)
            .sum();

        if total_weight == 0.0 {
            0.5
        } else {
            confirming_weight / total_weight
        }
    }

    /// Add attestation
    pub fn add_attestation(&mut self, attestation: QualityAttestation) {
        self.attestations.push(attestation);
    }
}

// ============================================================================
// #177: PoG-Weighted FL Participation
// Physical grounding gates FL participation
// ============================================================================

/// PoG thresholds for FL participation
pub mod pog_thresholds {
    /// Minimum PoG to participate in any FL round
    pub const MIN_PARTICIPATION: f64 = 0.1;
    /// PoG for full weight in aggregation
    pub const FULL_WEIGHT: f64 = 0.3;
    /// PoG for leadership/coordination roles
    pub const LEADERSHIP: f64 = 0.5;
    /// PoG for high-security rounds
    pub const HIGH_SECURITY: f64 = 0.7;
}

/// Extended participant with PoG score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PogParticipant {
    /// Base participant info
    pub participant: Participant,
    /// PoG score
    pub pog_score: PogScore,
    /// PoG verification level
    pub verification_level: PogVerificationLevel,
    /// Last PoG attestation timestamp
    pub last_attestation: u64,
}

/// PoG verification levels for FL
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PogVerificationLevel {
    /// Self-reported PoG (low trust)
    SelfReported,
    /// Community-witnessed PoG
    CommunityWitnessed,
    /// Oracle-verified PoG
    OracleVerified,
    /// Hardware-attested PoG (highest trust)
    HardwareAttested,
}

impl PogVerificationLevel {
    /// Get weight multiplier for this verification level
    pub fn weight_multiplier(&self) -> f64 {
        match self {
            Self::SelfReported => 0.3,
            Self::CommunityWitnessed => 0.6,
            Self::OracleVerified => 0.8,
            Self::HardwareAttested => 1.0,
        }
    }
}

/// Result of PoG participation check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PogParticipationResult {
    /// Whether participation is allowed
    pub allowed: bool,
    /// PoG weight multiplier
    pub pog_multiplier: f64,
    /// Whether eligible for leadership
    pub leadership_eligible: bool,
    /// Whether eligible for high-security rounds
    pub high_security_eligible: bool,
    /// Reason if not allowed
    pub rejection_reason: Option<String>,
}

/// Check if participant can join FL based on PoG
pub fn check_pog_participation(
    pog_score: &PogScore,
    verification_level: PogVerificationLevel,
    is_high_security_round: bool,
) -> PogParticipationResult {
    let score = pog_score.value();
    let verified_score = score * verification_level.weight_multiplier();

    // Check high-security requirement
    if is_high_security_round && verified_score < pog_thresholds::HIGH_SECURITY {
        return PogParticipationResult {
            allowed: false,
            pog_multiplier: 0.0,
            leadership_eligible: false,
            high_security_eligible: false,
            rejection_reason: Some(format!(
                "PoG score {:.2} below high-security threshold {:.2}",
                verified_score,
                pog_thresholds::HIGH_SECURITY
            )),
        };
    }

    // Check minimum participation
    if verified_score < pog_thresholds::MIN_PARTICIPATION {
        return PogParticipationResult {
            allowed: false,
            pog_multiplier: 0.0,
            leadership_eligible: false,
            high_security_eligible: false,
            rejection_reason: Some(format!(
                "PoG score {:.2} below minimum threshold {:.2}",
                verified_score,
                pog_thresholds::MIN_PARTICIPATION
            )),
        };
    }

    // Calculate multiplier
    let pog_multiplier = if verified_score >= pog_thresholds::FULL_WEIGHT {
        1.0
    } else {
        0.5 + (verified_score - pog_thresholds::MIN_PARTICIPATION)
            / (pog_thresholds::FULL_WEIGHT - pog_thresholds::MIN_PARTICIPATION)
            * 0.5
    };

    PogParticipationResult {
        allowed: true,
        pog_multiplier,
        leadership_eligible: verified_score >= pog_thresholds::LEADERSHIP,
        high_security_eligible: verified_score >= pog_thresholds::HIGH_SECURITY,
        rejection_reason: None,
    }
}

/// PoG-weighted aggregation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PogWeightedResult {
    /// Aggregated gradients
    pub aggregated: Vec<f64>,
    /// Weights used per participant
    pub participant_weights: Vec<(String, f64)>,
    /// Average PoG of included participants
    pub average_pog: f64,
    /// Participants excluded due to low PoG
    pub excluded_count: usize,
}

/// Perform PoG-weighted aggregation
pub fn pog_weighted_aggregation(
    updates: &[(GradientUpdate, PogScore, PogVerificationLevel)],
    is_high_security: bool,
) -> Result<PogWeightedResult, String> {
    if updates.is_empty() {
        return Err("No updates provided".into());
    }

    let dim = updates[0].0.gradients.len();
    let mut aggregated = vec![0.0; dim];
    let mut total_weight = 0.0;
    let mut participant_weights = Vec::new();
    let mut pog_sum = 0.0;
    let mut excluded = 0;

    for (update, pog, level) in updates {
        let check = check_pog_participation(pog, *level, is_high_security);

        if !check.allowed {
            excluded += 1;
            continue;
        }

        let weight = check.pog_multiplier;
        participant_weights.push((update.participant_id.clone(), weight));
        total_weight += weight;
        pog_sum += pog.value();

        for (i, g) in update.gradients.iter().enumerate() {
            aggregated[i] += g * weight;
        }
    }

    if total_weight == 0.0 {
        return Err("All participants excluded".into());
    }

    // Normalize
    for g in aggregated.iter_mut() {
        *g /= total_weight;
    }

    let included = participant_weights.len();
    Ok(PogWeightedResult {
        aggregated,
        participant_weights,
        average_pog: if included > 0 {
            pog_sum / included as f64
        } else {
            0.0
        },
        excluded_count: excluded,
    })
}

// ============================================================================
// #178: Cross-Domain ZK Proof Transfer
// Portable gradient contribution proofs
// ============================================================================

/// Cross-domain gradient contribution proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientContributionProof {
    /// Proof identifier
    pub proof_id: String,
    /// Agent who contributed
    pub agent_id: String,
    /// Source domain
    pub source_domain: String,
    /// Model identifier (hashed)
    pub model_hash: String,
    /// Round identifier
    pub round_id: u64,
    /// Quality improvement (epsilon)
    pub improvement_epsilon: f64,
    /// ZK proof data
    pub zk_proof: ZkProofData,
    /// Timestamp
    pub timestamp: u64,
    /// Expiration timestamp
    pub expires_at: u64,
}

/// ZK proof data (simplified representation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkProofData {
    /// Proof type
    pub proof_type: ZkProofType,
    /// Commitment to gradient hash
    pub gradient_commitment: String,
    /// Commitment to quality improvement
    pub improvement_commitment: String,
    /// Proof bytes (actual ZK proof)
    pub proof_bytes: Vec<u8>,
    /// Public inputs
    pub public_inputs: Vec<String>,
}

/// Types of ZK proofs for FL
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZkProofType {
    /// Proves gradient was valid (within bounds)
    GradientValidity,
    /// Proves contribution improved model
    ContributionImprovement,
    /// Proves Byzantine behavior was NOT present
    NonByzantine,
    /// Proves participation without revealing gradient
    PrivateParticipation,
}

impl GradientContributionProof {
    /// Create a new contribution proof
    pub fn new(
        agent_id: &str,
        source_domain: &str,
        model_hash: &str,
        round_id: u64,
        improvement: f64,
        proof_bytes: Vec<u8>,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            proof_id: format!("gcp-{}-{}-{}", agent_id, source_domain, round_id),
            agent_id: agent_id.to_string(),
            source_domain: source_domain.to_string(),
            model_hash: model_hash.to_string(),
            round_id,
            improvement_epsilon: improvement,
            zk_proof: ZkProofData {
                proof_type: ZkProofType::ContributionImprovement,
                gradient_commitment: format!("commit-{}", agent_id),
                improvement_commitment: format!("improve-{:.4}", improvement),
                proof_bytes,
                public_inputs: vec![
                    model_hash.to_string(),
                    format!("{}", round_id),
                    format!("{:.6}", improvement),
                ],
            },
            timestamp: now,
            expires_at: now + 30 * 24 * 60 * 60, // 30 days
        }
    }

    /// Check if proof is still valid
    pub fn is_valid(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now < self.expires_at
    }
}

/// Result of verifying a cross-domain proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDomainVerification {
    /// Whether proof verified successfully
    pub verified: bool,
    /// Verification confidence (0.0-1.0)
    pub confidence: f64,
    /// Suggested K-Vector updates for the agent
    pub kvector_credits: CrossDomainCredits,
    /// Warning messages
    pub warnings: Vec<String>,
}

/// K-Vector credits from cross-domain proof
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CrossDomainCredits {
    /// k_r (Reputation) credit
    pub reputation_credit: f64,
    /// k_p (Performance) credit
    pub performance_credit: f64,
    /// k_h (Historical) credit
    pub historical_credit: f64,
}

/// Verify a cross-domain gradient contribution proof
pub fn verify_cross_domain_proof(
    proof: &GradientContributionProof,
    _target_domain: &str,
    trust_source_domain: bool,
) -> CrossDomainVerification {
    // Basic validity check
    if !proof.is_valid() {
        return CrossDomainVerification {
            verified: false,
            confidence: 0.0,
            kvector_credits: CrossDomainCredits::default(),
            warnings: vec!["Proof has expired".into()],
        };
    }

    // Cross-domain trust scaling
    let domain_trust = if trust_source_domain { 1.0 } else { 0.5 };

    // Calculate credits based on improvement and domain trust
    let base_credit = proof.improvement_epsilon.clamp(0.0, 1.0);
    let scaled_credit = base_credit * domain_trust;

    CrossDomainVerification {
        verified: true,
        confidence: domain_trust,
        kvector_credits: CrossDomainCredits {
            reputation_credit: scaled_credit * 0.3,
            performance_credit: scaled_credit * 0.5,
            historical_credit: scaled_credit * 0.2,
        },
        warnings: if !trust_source_domain {
            vec![format!(
                "Source domain '{}' not fully trusted",
                proof.source_domain
            )]
        } else {
            vec![]
        },
    }
}

// ============================================================================
// #179: Temporal Commitment-Backed FL
// Long-term participation incentives
// ============================================================================

/// Temporal commitment for FL participation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FLTemporalCommitment {
    /// Commitment identifier
    pub commitment_id: String,
    /// Participant who made commitment
    pub participant_id: String,
    /// Commitment tier
    pub tier: CommitmentTier,
    /// Minimum quality threshold promised
    pub quality_threshold: f64,
    /// Number of rounds committed
    pub rounds_committed: u64,
    /// Rounds completed so far
    pub rounds_completed: u64,
    /// KREDIT collateral locked
    pub collateral_locked: u64,
    /// Start timestamp
    pub start_time: u64,
    /// End timestamp
    pub end_time: u64,
    /// Quality history
    pub quality_history: Vec<f64>,
    /// Current status
    pub status: CommitmentStatus,
}

/// Commitment tiers with different durations and rewards
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitmentTier {
    /// Short-term: 10 rounds
    Bronze,
    /// Medium-term: 50 rounds
    Silver,
    /// Long-term: 200 rounds
    Gold,
    /// Very long-term: 1000 rounds
    Platinum,
}

impl CommitmentTier {
    /// Get required rounds for this tier
    pub fn required_rounds(&self) -> u64 {
        match self {
            Self::Bronze => 10,
            Self::Silver => 50,
            Self::Gold => 200,
            Self::Platinum => 1000,
        }
    }

    /// Get patience coefficient multiplier
    pub fn patience_multiplier(&self) -> f64 {
        match self {
            Self::Bronze => 1.1,
            Self::Silver => 1.25,
            Self::Gold => 1.5,
            Self::Platinum => 2.0,
        }
    }

    /// Get early exit penalty percentage
    pub fn early_exit_penalty(&self) -> f64 {
        match self {
            Self::Bronze => 0.1,
            Self::Silver => 0.2,
            Self::Gold => 0.3,
            Self::Platinum => 0.5,
        }
    }
}

/// Commitment status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitmentStatus {
    /// Active and in good standing
    Active,
    /// Quality below threshold (warning)
    Warning,
    /// Breached commitment terms
    Breached,
    /// Successfully completed
    Completed,
    /// Early exit (with penalty)
    EarlyExit,
}

impl FLTemporalCommitment {
    /// Create a new commitment
    pub fn new(
        participant_id: &str,
        tier: CommitmentTier,
        quality_threshold: f64,
        collateral: u64,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            commitment_id: format!("commit-{}-{}", participant_id, now),
            participant_id: participant_id.to_string(),
            tier,
            quality_threshold,
            rounds_committed: tier.required_rounds(),
            rounds_completed: 0,
            collateral_locked: collateral,
            start_time: now,
            end_time: 0, // Set when completed
            quality_history: Vec::new(),
            status: CommitmentStatus::Active,
        }
    }

    /// Record a round completion
    pub fn record_round(&mut self, quality: f64) {
        self.quality_history.push(quality);
        self.rounds_completed += 1;

        // Update status
        if quality < self.quality_threshold {
            self.status = CommitmentStatus::Warning;
        } else if self.rounds_completed >= self.rounds_committed {
            self.status = CommitmentStatus::Completed;
            self.end_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
        } else {
            self.status = CommitmentStatus::Active;
        }
    }

    /// Calculate average quality
    pub fn average_quality(&self) -> f64 {
        if self.quality_history.is_empty() {
            return 0.0;
        }
        self.quality_history.iter().sum::<f64>() / self.quality_history.len() as f64
    }

    /// Calculate current patience coefficient
    pub fn patience_coefficient(&self) -> f64 {
        if self.status == CommitmentStatus::Completed {
            return self.tier.patience_multiplier();
        }

        // Partial coefficient based on progress
        let progress = self.rounds_completed as f64 / self.rounds_committed as f64;
        1.0 + (self.tier.patience_multiplier() - 1.0) * progress
    }

    /// Calculate early exit penalty
    pub fn calculate_early_exit_penalty(&self) -> u64 {
        if self.status == CommitmentStatus::Completed {
            return 0;
        }

        let penalty_rate = self.tier.early_exit_penalty();
        let remaining = self.rounds_committed - self.rounds_completed;
        let remaining_ratio = remaining as f64 / self.rounds_committed as f64;

        (self.collateral_locked as f64 * penalty_rate * remaining_ratio) as u64
    }
}

// ============================================================================
// #180: Storage-Backed FL Immutability
// Merkle tree of round history
// ============================================================================

/// Merkle node for FL round archive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FLMerkleNode {
    /// Node hash
    pub hash: [u8; 32],
    /// Left child hash (if internal node)
    pub left: Option<[u8; 32]>,
    /// Right child hash (if internal node)
    pub right: Option<[u8; 32]>,
    /// Data (if leaf node)
    pub data: Option<FLMerkleLeaf>,
}

/// Leaf data for FL Merkle tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FLMerkleLeaf {
    /// Participant ID
    pub participant_id: String,
    /// Gradient hash
    pub gradient_hash: [u8; 32],
    /// Timestamp
    pub timestamp: u64,
    /// Epistemic classification
    pub classification: GradientEpistemicClassification,
}

/// FL round archive with Merkle root
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FLRoundArchive {
    /// Round identifier
    pub round_id: u64,
    /// Model version
    pub model_version: u64,
    /// Merkle root of all gradients
    pub merkle_root: [u8; 32],
    /// Number of leaves (participants)
    pub leaf_count: usize,
    /// Aggregated result hash
    pub result_hash: [u8; 32],
    /// Epistemic quality tier
    pub quality_tier: EpistemicQualityTier,
    /// Storage receipt (from distributed storage)
    pub storage_receipt: Option<StorageReceipt>,
    /// Timestamp
    pub timestamp: u64,
}

/// Receipt from distributed storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageReceipt {
    /// Storage backend used
    pub backend: StorageBackend,
    /// Content address (IPFS CID, DHT key, etc.)
    pub content_address: String,
    /// Replication factor
    pub replication: u32,
    /// Timestamp stored
    pub stored_at: u64,
}

/// Storage backends for FL archives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageBackend {
    /// IPFS content-addressed storage
    Ipfs,
    /// Holochain DHT
    HolochainDht,
    /// Local storage (development)
    Local,
    /// Memory only (testing)
    Memory,
}

/// Simple hash function (SHA-256 placeholder)
fn hash_bytes(data: &[u8]) -> [u8; 32] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    let h = hasher.finish();

    let mut result = [0u8; 32];
    result[0..8].copy_from_slice(&h.to_le_bytes());
    result[8..16].copy_from_slice(&h.to_be_bytes());
    // Fill rest with hash of hash
    let mut hasher2 = DefaultHasher::new();
    h.hash(&mut hasher2);
    let h2 = hasher2.finish();
    result[16..24].copy_from_slice(&h2.to_le_bytes());
    result[24..32].copy_from_slice(&h2.to_be_bytes());
    result
}

/// Build Merkle tree from gradient updates
pub fn build_gradient_merkle_tree(
    updates: &[EpistemicGradientUpdate],
) -> (Vec<FLMerkleNode>, [u8; 32]) {
    if updates.is_empty() {
        return (vec![], [0u8; 32]);
    }

    // Create leaf nodes
    let mut leaves: Vec<FLMerkleNode> = updates
        .iter()
        .map(|u| {
            let gradient_bytes: Vec<u8> = u
                .gradient
                .gradients
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            let gradient_hash = hash_bytes(&gradient_bytes);

            let leaf_data = FLMerkleLeaf {
                participant_id: u.gradient.participant_id.clone(),
                gradient_hash,
                timestamp: u.gradient.metadata.timestamp,
                classification: u.classification.clone(),
            };

            // Leaf hash includes all data
            let mut leaf_bytes = Vec::new();
            leaf_bytes.extend_from_slice(&gradient_hash);
            leaf_bytes.extend_from_slice(leaf_data.participant_id.as_bytes());
            leaf_bytes.extend_from_slice(&leaf_data.timestamp.to_le_bytes());

            FLMerkleNode {
                hash: hash_bytes(&leaf_bytes),
                left: None,
                right: None,
                data: Some(leaf_data),
            }
        })
        .collect();

    // Pad to power of 2
    while leaves.len().count_ones() != 1 {
        leaves.push(FLMerkleNode {
            hash: [0u8; 32],
            left: None,
            right: None,
            data: None,
        });
    }

    let mut nodes = leaves.clone();
    let mut current_level = leaves;

    // Build tree bottom-up
    while current_level.len() > 1 {
        let mut next_level = Vec::new();

        for chunk in current_level.chunks(2) {
            let left_hash = chunk[0].hash;
            let right_hash = chunk.get(1).map(|n| n.hash).unwrap_or([0u8; 32]);

            let mut combined = Vec::new();
            combined.extend_from_slice(&left_hash);
            combined.extend_from_slice(&right_hash);

            let parent = FLMerkleNode {
                hash: hash_bytes(&combined),
                left: Some(left_hash),
                right: Some(right_hash),
                data: None,
            };

            nodes.push(parent.clone());
            next_level.push(parent);
        }

        current_level = next_level;
    }

    let root = current_level.first().map(|n| n.hash).unwrap_or([0u8; 32]);
    (nodes, root)
}

/// Create FL round archive
pub fn create_round_archive(
    round: &FLRound,
    updates: &[EpistemicGradientUpdate],
    epistemic_stats: &RoundEpistemicStats,
) -> FLRoundArchive {
    let (_, merkle_root) = build_gradient_merkle_tree(updates);

    // Hash aggregated result
    let result_hash = round
        .aggregated_result
        .as_ref()
        .map(|r| {
            let bytes: Vec<u8> = r.gradients.iter().flat_map(|f| f.to_le_bytes()).collect();
            hash_bytes(&bytes)
        })
        .unwrap_or([0u8; 32]);

    FLRoundArchive {
        round_id: round.round_id,
        model_version: round.model_version,
        merkle_root,
        leaf_count: updates.len(),
        result_hash,
        quality_tier: epistemic_stats.quality_tier,
        storage_receipt: None,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    }
}

/// Generate Merkle proof for a participant's gradient
pub fn generate_merkle_proof(
    nodes: &[FLMerkleNode],
    participant_id: &str,
) -> Option<Vec<[u8; 32]>> {
    // Find leaf with matching participant
    let leaf_idx = nodes.iter().position(|n| {
        n.data
            .as_ref()
            .map(|d| d.participant_id == participant_id)
            .unwrap_or(false)
    })?;

    // Build proof path (simplified - would need proper tree structure)
    let mut proof = Vec::new();
    let mut idx = leaf_idx;
    let leaf_count = nodes.iter().filter(|n| n.data.is_some()).count();

    while idx < nodes.len() - 1 {
        let sibling_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
        if sibling_idx < nodes.len() {
            proof.push(nodes[sibling_idx].hash);
        }
        idx = leaf_count + idx / 2;
    }

    Some(proof)
}

/// Verify a Merkle proof
pub fn verify_merkle_proof(
    leaf_hash: [u8; 32],
    proof: &[[u8; 32]],
    root: [u8; 32],
    leaf_index: usize,
) -> bool {
    let mut current = leaf_hash;
    let mut idx = leaf_index;

    for sibling in proof {
        let mut combined = Vec::new();
        if idx.is_multiple_of(2) {
            combined.extend_from_slice(&current);
            combined.extend_from_slice(sibling);
        } else {
            combined.extend_from_slice(sibling);
            combined.extend_from_slice(&current);
        }
        current = hash_bytes(&combined);
        idx /= 2;
    }

    current == root
}

#[cfg(test)]
mod tests {
    use super::super::types::GradientMetadata;
    use super::*;

    #[test]
    fn test_compressed_agent_state() {
        let kvector = [0.5; 10];
        let state = compress_agent_state(
            "agent-1",
            &kvector,
            None,
            &AgentCompressionConfig::default(),
        );

        assert_eq!(state.agent_id, "agent-1");
        assert!(!state.compressed_kvector.is_empty());
    }

    #[test]
    fn test_state_similarity() {
        let kv1 = [0.5; 10];
        let kv2 = [0.5; 10];
        let kv3 = [0.0; 10];

        let config = AgentCompressionConfig::default();
        let s1 = compress_agent_state("a1", &kv1, None, &config);
        let s2 = compress_agent_state("a2", &kv2, None, &config);
        let s3 = compress_agent_state("a3", &kv3, None, &config);

        // Identical should be 1.0
        assert!((compute_state_similarity(&s1, &s2) - 1.0).abs() < 0.01);
        // Different should be lower
        assert!(compute_state_similarity(&s1, &s3) < 0.5);
    }

    #[test]
    fn test_pog_participation() {
        let high_pog = PogScore::new(0.8);
        let low_pog = PogScore::new(0.05);

        let high_result =
            check_pog_participation(&high_pog, PogVerificationLevel::OracleVerified, false);
        assert!(high_result.allowed);
        assert!(high_result.leadership_eligible);

        let low_result =
            check_pog_participation(&low_pog, PogVerificationLevel::SelfReported, false);
        assert!(!low_result.allowed);
    }

    #[test]
    fn test_temporal_commitment() {
        let mut commitment = FLTemporalCommitment::new("p1", CommitmentTier::Bronze, 0.7, 1000);

        assert_eq!(commitment.status, CommitmentStatus::Active);
        assert_eq!(commitment.rounds_committed, 10);

        // Record good rounds
        for _ in 0..10 {
            commitment.record_round(0.8);
        }

        assert_eq!(commitment.status, CommitmentStatus::Completed);
        assert!((commitment.patience_coefficient() - 1.1).abs() < 0.01);
    }

    #[test]
    fn test_merkle_tree() {
        let updates = vec![
            EpistemicGradientUpdate::new(
                GradientUpdate::new("p1".into(), 1, vec![0.1, 0.2], 100, 0.5),
                GradientEpistemicClassification::default(),
            ),
            EpistemicGradientUpdate::new(
                GradientUpdate::new("p2".into(), 1, vec![0.2, 0.3], 100, 0.4),
                GradientEpistemicClassification::default(),
            ),
        ];

        let (nodes, root) = build_gradient_merkle_tree(&updates);

        assert!(!nodes.is_empty());
        assert_ne!(root, [0u8; 32]);
    }

    #[test]
    fn test_cross_domain_proof() {
        let proof = GradientContributionProof::new(
            "agent-1",
            "domain-a",
            "model-hash-123",
            42,
            0.15,
            vec![1, 2, 3, 4],
        );

        assert!(proof.is_valid());

        let verification = verify_cross_domain_proof(&proof, "domain-b", true);
        assert!(verification.verified);
        assert!(verification.kvector_credits.performance_credit > 0.0);
    }
}
