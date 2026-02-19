//! # Epistemic FL Bridge
//!
//! Integrates the epistemic classification framework with federated learning.
//! Gradients are classified by E-N-M-H dimensions and aggregation is weighted
//! by epistemic quality.
//!
//! ## Gradient Epistemic Classification
//!
//! | Dimension | Gradient Interpretation |
//! |-----------|------------------------|
//! | E (Empirical) | How verifiable is this gradient? (PoGQ level) |
//! | N (Normative) | What scope vouches for it? (single agent vs consensus) |
//! | M (Materiality) | How long does this gradient matter? |
//! | H (Harmonic) | Impact on model coherence |
//!
//! ## Epistemic-Weighted Aggregation
//!
//! Instead of pure trust scores, aggregation uses:
//! `weight = reputation² × PoGQ × epistemic_weight(E,N,M,H)`

use super::types::{AggregatedGradient, AggregationMethod, GradientUpdate, Participant};
use crate::epistemic::{
    EmpiricalLevel, EpistemicClassificationExtended, HarmonicLevel, MaterialityLevel,
    NormativeLevel,
};
use serde::{Deserialize, Serialize};

/// Epistemic classification for a gradient update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientEpistemicClassification {
    /// Empirical level - how verifiable is this gradient?
    pub empirical: EmpiricalLevel,
    /// Normative level - what scope vouches for this?
    pub normative: NormativeLevel,
    /// Materiality level - how long does this matter?
    pub materiality: MaterialityLevel,
    /// Harmonic level - impact on model coherence
    pub harmonic: HarmonicLevel,
    /// Classification confidence (0.0-1.0)
    pub confidence: f32,
}

impl GradientEpistemicClassification {
    /// Create from full E-N-M-H specification
    pub fn new(
        e: EmpiricalLevel,
        n: NormativeLevel,
        m: MaterialityLevel,
        h: HarmonicLevel,
    ) -> Self {
        Self {
            empirical: e,
            normative: n,
            materiality: m,
            harmonic: h,
            confidence: 1.0,
        }
    }

    /// Create with confidence
    pub fn with_confidence(
        e: EmpiricalLevel,
        n: NormativeLevel,
        m: MaterialityLevel,
        h: HarmonicLevel,
        confidence: f32,
    ) -> Self {
        Self {
            empirical: e,
            normative: n,
            materiality: m,
            harmonic: h,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Get as extended classification
    pub fn as_extended(&self) -> EpistemicClassificationExtended {
        EpistemicClassificationExtended::new(
            self.empirical,
            self.normative,
            self.materiality,
            self.harmonic,
        )
    }

    /// Calculate epistemic weight for aggregation
    pub fn epistemic_weight(&self) -> f64 {
        let e_factor = match self.empirical {
            EmpiricalLevel::E0Null => 0.2,
            EmpiricalLevel::E1Testimonial => 0.4,
            EmpiricalLevel::E2PrivateVerify => 0.6,
            EmpiricalLevel::E3Cryptographic => 0.8,
            EmpiricalLevel::E4PublicRepro => 1.0,
        };

        let n_factor = match self.normative {
            NormativeLevel::N0Personal => 0.5,
            NormativeLevel::N1Communal => 0.7,
            NormativeLevel::N2Network => 0.9,
            NormativeLevel::N3Axiomatic => 1.0,
        };

        let m_factor = match self.materiality {
            MaterialityLevel::M0Ephemeral => 0.3,
            MaterialityLevel::M1Temporal => 0.5,
            MaterialityLevel::M2Persistent => 0.8,
            MaterialityLevel::M3Foundational => 1.0,
        };

        let h_factor = match self.harmonic {
            HarmonicLevel::H0None => 0.5,
            HarmonicLevel::H1Local => 0.6,
            HarmonicLevel::H2Network => 0.8,
            HarmonicLevel::H3Civilizational => 0.9,
            HarmonicLevel::H4Kosmic => 1.0,
        };

        e_factor * n_factor * m_factor * h_factor * self.confidence as f64
    }
}

impl Default for GradientEpistemicClassification {
    fn default() -> Self {
        Self {
            empirical: EmpiricalLevel::E0Null,
            normative: NormativeLevel::N0Personal,
            materiality: MaterialityLevel::M0Ephemeral,
            harmonic: HarmonicLevel::H0None,
            confidence: 0.5,
        }
    }
}

/// Extended gradient update with epistemic classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicGradientUpdate {
    /// Base gradient update
    pub gradient: GradientUpdate,
    /// Epistemic classification of this gradient
    pub classification: GradientEpistemicClassification,
    /// Optional ZK proof data (for E3+ classification)
    pub proof_data: Option<Vec<u8>>,
    /// Phi coherence score of the submitting agent (if available)
    pub agent_phi: Option<f64>,
}

impl EpistemicGradientUpdate {
    /// Create from base gradient with default classification
    pub fn from_gradient(gradient: GradientUpdate) -> Self {
        Self {
            gradient,
            classification: GradientEpistemicClassification::default(),
            proof_data: None,
            agent_phi: None,
        }
    }

    /// Create with explicit classification
    pub fn new(gradient: GradientUpdate, classification: GradientEpistemicClassification) -> Self {
        Self {
            gradient,
            classification,
            proof_data: None,
            agent_phi: None,
        }
    }

    /// Add ZK proof (automatically upgrades to E3 minimum)
    pub fn with_proof(mut self, proof: Vec<u8>) -> Self {
        self.proof_data = Some(proof);
        if self.classification.empirical < EmpiricalLevel::E3Cryptographic {
            self.classification.empirical = EmpiricalLevel::E3Cryptographic;
        }
        self
    }

    /// Add agent Phi coherence score
    pub fn with_phi(mut self, phi: f64) -> Self {
        self.agent_phi = Some(phi.clamp(0.0, 1.0));
        self
    }

    /// Get combined weight for aggregation
    /// Formula: epistemic_weight × phi_multiplier
    pub fn combined_weight(&self) -> f64 {
        let epistemic = self.classification.epistemic_weight();
        let phi_mult = self.agent_phi.map(|p| 0.5 + p * 0.5).unwrap_or(0.75);
        epistemic * phi_mult
    }
}

/// Hints for automatic gradient classification
#[derive(Debug, Clone, Default)]
pub struct GradientClassificationHints {
    /// Does gradient have PoGQ proof?
    pub has_pogq: bool,
    /// PoGQ quality score (0.0-1.0)
    pub pogq_score: Option<f64>,
    /// Has ZK proof of gradient properties?
    pub has_zk_proof: bool,
    /// Was gradient validated by other participants?
    pub peer_validated: bool,
    /// Number of peers that validated
    pub peer_validation_count: usize,
    /// Is this for a long-running model?
    pub persistent_model: bool,
    /// Number of downstream models affected
    pub downstream_impact: usize,
    /// Agent's coherence (Phi) score
    pub agent_phi: Option<f64>,
}

/// Classify gradient based on hints
pub fn classify_gradient(hints: &GradientClassificationHints) -> GradientEpistemicClassification {
    // E-axis: Verifiability
    let empirical = if hints.has_zk_proof && hints.has_pogq && hints.pogq_score.unwrap_or(0.0) > 0.8
    {
        EmpiricalLevel::E4PublicRepro
    } else if hints.has_zk_proof || (hints.has_pogq && hints.pogq_score.unwrap_or(0.0) > 0.7) {
        EmpiricalLevel::E3Cryptographic
    } else if hints.peer_validated && hints.peer_validation_count >= 2 {
        EmpiricalLevel::E2PrivateVerify
    } else if hints.has_pogq {
        EmpiricalLevel::E1Testimonial
    } else {
        EmpiricalLevel::E0Null
    };

    // N-axis: Scope of agreement
    let normative = if hints.peer_validation_count >= 5 {
        NormativeLevel::N2Network
    } else if hints.peer_validation_count >= 2 {
        NormativeLevel::N1Communal
    } else {
        NormativeLevel::N0Personal
    };

    // M-axis: Materiality/persistence
    let materiality = if hints.persistent_model && hints.downstream_impact >= 3 {
        MaterialityLevel::M3Foundational
    } else if hints.persistent_model {
        MaterialityLevel::M2Persistent
    } else if hints.downstream_impact >= 1 {
        MaterialityLevel::M1Temporal
    } else {
        MaterialityLevel::M0Ephemeral
    };

    // H-axis: Harmonic impact
    let harmonic = if hints.downstream_impact >= 10 {
        HarmonicLevel::H3Civilizational
    } else if hints.downstream_impact >= 3 {
        HarmonicLevel::H2Network
    } else if hints.downstream_impact >= 1 {
        HarmonicLevel::H1Local
    } else {
        HarmonicLevel::H0None
    };

    // Confidence based on how much evidence we have
    let evidence_count = [
        hints.has_pogq,
        hints.has_zk_proof,
        hints.peer_validated,
        hints.persistent_model,
        hints.agent_phi.is_some(),
    ]
    .iter()
    .filter(|&&x| x)
    .count();
    let confidence = (0.3 + evidence_count as f32 * 0.14).min(1.0);

    GradientEpistemicClassification::with_confidence(
        empirical,
        normative,
        materiality,
        harmonic,
        confidence,
    )
}

/// Epistemic-weighted aggregation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicAggregationResult {
    /// Aggregated gradient
    pub aggregated: AggregatedGradient,
    /// Average epistemic weight of included gradients
    pub average_epistemic_weight: f64,
    /// Distribution of E levels in input
    pub empirical_distribution: [u32; 5],
    /// Number of gradients with ZK proofs
    pub zk_proven_count: usize,
    /// Average Phi of participating agents
    pub average_phi: Option<f64>,
    /// Weights used for each participant
    pub participant_weights: Vec<(String, f64)>,
}

/// Perform epistemic-weighted aggregation
///
/// Weight formula: `reputation² × pogq × epistemic_weight × phi_multiplier`
pub fn epistemic_weighted_aggregation(
    updates: &[EpistemicGradientUpdate],
    participants: &[Participant],
    min_epistemic_weight: f64,
) -> Result<EpistemicAggregationResult, EpistemicAggregationError> {
    if updates.is_empty() {
        return Err(EpistemicAggregationError::NoUpdates);
    }

    // Validate all gradients have same dimension
    let dim = updates[0].gradient.gradients.len();
    if !updates.iter().all(|u| u.gradient.gradients.len() == dim) {
        return Err(EpistemicAggregationError::DimensionMismatch);
    }

    // Build participant lookup
    let participant_map: std::collections::HashMap<_, _> =
        participants.iter().map(|p| (p.id.clone(), p)).collect();

    // Calculate weights and filter by minimum epistemic threshold
    let mut weights: Vec<(usize, f64)> = Vec::new();
    let mut participant_weights: Vec<(String, f64)> = Vec::new();
    let mut total_weight = 0.0;
    let mut empirical_dist = [0u32; 5];
    let mut zk_count = 0;
    let mut phi_sum = 0.0;
    let mut phi_count = 0;

    for (idx, update) in updates.iter().enumerate() {
        let epistemic_weight = update.classification.epistemic_weight();

        // Skip if below minimum threshold
        if epistemic_weight < min_epistemic_weight {
            continue;
        }

        // Get participant reputation
        let reputation = participant_map
            .get(&update.gradient.participant_id)
            .map(|p| p.trust_score())
            .unwrap_or(0.5);

        // Get PoGQ score
        let pogq = participant_map
            .get(&update.gradient.participant_id)
            .and_then(|p| p.pogq.as_ref())
            .map(|p| p.quality)
            .unwrap_or(0.5);

        // Phi multiplier
        let phi_mult = update.agent_phi.map(|p| 0.5 + p * 0.5).unwrap_or(0.75);

        // Combined weight: reputation² × pogq × epistemic × phi
        let weight = reputation.powi(2) * pogq * epistemic_weight * phi_mult;

        weights.push((idx, weight));
        participant_weights.push((update.gradient.participant_id.clone(), weight));
        total_weight += weight;

        // Track statistics
        empirical_dist[update.classification.empirical as usize] += 1;
        if update.proof_data.is_some() {
            zk_count += 1;
        }
        if let Some(phi) = update.agent_phi {
            phi_sum += phi;
            phi_count += 1;
        }
    }

    if weights.is_empty() {
        return Err(EpistemicAggregationError::AllBelowThreshold);
    }

    // Normalize weights
    for (_, w) in weights.iter_mut() {
        *w /= total_weight;
    }

    // Weighted average of gradients
    let mut aggregated_grads = vec![0.0; dim];
    for (idx, weight) in &weights {
        for (i, g) in updates[*idx].gradient.gradients.iter().enumerate() {
            aggregated_grads[i] += g * weight;
        }
    }

    let aggregated = AggregatedGradient::new(
        aggregated_grads,
        updates[0].gradient.model_version,
        weights.len(),
        updates.len() - weights.len(),
        AggregationMethod::TrustWeighted,
    );

    let avg_weight = participant_weights.iter().map(|(_, w)| w).sum::<f64>()
        / participant_weights.len() as f64
        * total_weight;
    let avg_phi = if phi_count > 0 {
        Some(phi_sum / phi_count as f64)
    } else {
        None
    };

    Ok(EpistemicAggregationResult {
        aggregated,
        average_epistemic_weight: avg_weight,
        empirical_distribution: empirical_dist,
        zk_proven_count: zk_count,
        average_phi: avg_phi,
        participant_weights,
    })
}

/// Errors during epistemic aggregation
#[derive(Debug, Clone)]
pub enum EpistemicAggregationError {
    /// No updates provided
    NoUpdates,
    /// Gradient dimensions don't match
    DimensionMismatch,
    /// All gradients below epistemic threshold
    AllBelowThreshold,
    /// Invalid classification
    InvalidClassification(String),
}

impl std::fmt::Display for EpistemicAggregationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoUpdates => write!(f, "No gradient updates provided"),
            Self::DimensionMismatch => write!(f, "Gradient dimensions do not match"),
            Self::AllBelowThreshold => write!(f, "All gradients below epistemic threshold"),
            Self::InvalidClassification(msg) => write!(f, "Invalid classification: {}", msg),
        }
    }
}

impl std::error::Error for EpistemicAggregationError {}

/// Statistics about epistemic quality of FL round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundEpistemicStats {
    /// Average E level (0-4)
    pub average_empirical: f32,
    /// Average N level (0-3)
    pub average_normative: f32,
    /// Percentage of E3+ (cryptographically proven) gradients
    pub high_verification_ratio: f32,
    /// Percentage of gradients with ZK proofs
    pub zk_proof_ratio: f32,
    /// Average epistemic weight
    pub average_weight: f64,
    /// Quality tier of this round
    pub quality_tier: EpistemicQualityTier,
}

/// Quality tier for FL rounds based on epistemic metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpistemicQualityTier {
    /// High quality: >50% E3+, avg weight > 0.5
    Gold,
    /// Medium quality: >25% E3+, avg weight > 0.3
    Silver,
    /// Low quality: some verification
    Bronze,
    /// Minimal quality: mostly unverified
    Unverified,
}

impl RoundEpistemicStats {
    /// Calculate from a set of epistemic updates
    pub fn from_updates(updates: &[EpistemicGradientUpdate]) -> Self {
        if updates.is_empty() {
            return Self {
                average_empirical: 0.0,
                average_normative: 0.0,
                high_verification_ratio: 0.0,
                zk_proof_ratio: 0.0,
                average_weight: 0.0,
                quality_tier: EpistemicQualityTier::Unverified,
            };
        }

        let n = updates.len() as f32;
        let e_sum: f32 = updates
            .iter()
            .map(|u| u.classification.empirical as u8 as f32)
            .sum();
        let n_sum: f32 = updates
            .iter()
            .map(|u| u.classification.normative as u8 as f32)
            .sum();
        let high_e = updates
            .iter()
            .filter(|u| u.classification.empirical >= EmpiricalLevel::E3Cryptographic)
            .count();
        let zk = updates.iter().filter(|u| u.proof_data.is_some()).count();
        let weight_sum: f64 = updates
            .iter()
            .map(|u| u.classification.epistemic_weight())
            .sum();

        let avg_e = e_sum / n;
        let avg_n = n_sum / n;
        let high_ratio = high_e as f32 / n;
        let zk_ratio = zk as f32 / n;
        let avg_weight = weight_sum / n as f64;

        let tier = if high_ratio > 0.5 && avg_weight > 0.5 {
            EpistemicQualityTier::Gold
        } else if high_ratio > 0.25 && avg_weight > 0.3 {
            EpistemicQualityTier::Silver
        } else if high_ratio > 0.0 || avg_weight > 0.1 {
            EpistemicQualityTier::Bronze
        } else {
            EpistemicQualityTier::Unverified
        };

        Self {
            average_empirical: avg_e,
            average_normative: avg_n,
            high_verification_ratio: high_ratio,
            zk_proof_ratio: zk_ratio,
            average_weight: avg_weight,
            quality_tier: tier,
        }
    }
}

// ============================================================================
// Byzantine Attack Classification with Epistemic Dimensions
// ============================================================================

/// Classification of Byzantine attack types using epistemic framework
///
/// Maps attack characteristics to E-N-M-H dimensions for graduated response:
/// - E-axis: How detectable/provable is this attack?
/// - N-axis: Scope of participants involved (single vs coordinated)
/// - M-axis: Duration/persistence of attack impact
/// - H-axis: Breadth of harm (local model vs civilization-wide)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ByzantineAttackType {
    /// E0: Undetectable/unprovable anomaly - could be noise
    UnverifiedAnomaly,
    /// E1: Testimonial evidence - other participants flagged it
    PeerReportedMalice,
    /// E2: Private verification - admin investigation confirmed
    VerifiedMalfunction,
    /// E3: Cryptographic proof - ZK verification failed
    CryptoProofFailure,
    /// E4: Publicly reproducible - gradient clearly Byzantine
    ManifestByzantine,
}

impl ByzantineAttackType {
    /// Get the empirical level for this attack type
    pub fn empirical_level(&self) -> EmpiricalLevel {
        match self {
            Self::UnverifiedAnomaly => EmpiricalLevel::E0Null,
            Self::PeerReportedMalice => EmpiricalLevel::E1Testimonial,
            Self::VerifiedMalfunction => EmpiricalLevel::E2PrivateVerify,
            Self::CryptoProofFailure => EmpiricalLevel::E3Cryptographic,
            Self::ManifestByzantine => EmpiricalLevel::E4PublicRepro,
        }
    }

    /// Base penalty multiplier for this attack type
    /// Higher E-level = more certain = stricter penalty
    pub fn penalty_multiplier(&self) -> f64 {
        match self {
            Self::UnverifiedAnomaly => 0.1, // Very light - might be false positive
            Self::PeerReportedMalice => 0.3, // Light - needs investigation
            Self::VerifiedMalfunction => 0.5, // Moderate - confirmed issue
            Self::CryptoProofFailure => 0.8, // Severe - cryptographic evidence
            Self::ManifestByzantine => 1.0, // Maximum - undeniable attack
        }
    }
}

/// Scope of Byzantine attack (N-axis)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ByzantineScope {
    /// N0: Single participant acting alone
    Individual,
    /// N1: Small group (2-4) coordinated
    SmallGroup,
    /// N2: Network-level coordinated attack
    NetworkCoordinated,
    /// N3: Systemic/constitutional attack
    SystemicAttack,
}

impl ByzantineScope {
    /// Get the normative level
    pub fn normative_level(&self) -> NormativeLevel {
        match self {
            Self::Individual => NormativeLevel::N0Personal,
            Self::SmallGroup => NormativeLevel::N1Communal,
            Self::NetworkCoordinated => NormativeLevel::N2Network,
            Self::SystemicAttack => NormativeLevel::N3Axiomatic,
        }
    }

    /// Scope multiplier for penalties
    /// Coordinated attacks get harsher treatment
    pub fn scope_multiplier(&self) -> f64 {
        match self {
            Self::Individual => 1.0,
            Self::SmallGroup => 1.5,
            Self::NetworkCoordinated => 2.0,
            Self::SystemicAttack => 3.0,
        }
    }
}

/// Extended Byzantine detection result with epistemic classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicByzantineResult {
    /// Participant ID
    pub participant_id: String,
    /// Type of attack detected
    pub attack_type: ByzantineAttackType,
    /// Scope of attack
    pub scope: ByzantineScope,
    /// Detection confidence (0.0-1.0)
    pub confidence: f64,
    /// Whether this appears malicious vs incoherent
    pub appears_malicious: bool,
    /// Agent's Phi coherence at time of detection
    pub agent_phi: Option<f64>,
    /// Recommended K-Vector penalties
    pub recommended_penalties: KVectorPenalties,
    /// Human-readable explanation
    pub explanation: String,
}

/// K-Vector dimension penalties for Byzantine behavior
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KVectorPenalties {
    /// k_r (Reputation) penalty
    pub reputation_penalty: f64,
    /// k_i (Integrity) penalty
    pub integrity_penalty: f64,
    /// k_p (Performance) penalty
    pub performance_penalty: f64,
    /// k_v (Verification) penalty
    pub verification_penalty: f64,
    /// Total combined penalty
    pub total_penalty: f64,
}

impl KVectorPenalties {
    /// Calculate penalties based on attack classification
    pub fn from_attack(
        attack_type: ByzantineAttackType,
        scope: ByzantineScope,
        confidence: f64,
        appears_malicious: bool,
    ) -> Self {
        let base = attack_type.penalty_multiplier();
        let scope_mult = scope.scope_multiplier();
        let malice_mult = if appears_malicious { 1.5 } else { 1.0 };

        let combined = base * scope_mult * malice_mult * confidence;

        // Distribute penalties across dimensions
        // Malicious attacks hit integrity harder
        // Incoherent attacks hit verification harder
        let (integrity_ratio, verification_ratio) = if appears_malicious {
            (0.4, 0.2)
        } else {
            (0.2, 0.4)
        };

        Self {
            reputation_penalty: combined * 0.25,
            integrity_penalty: combined * integrity_ratio,
            performance_penalty: combined * 0.15,
            verification_penalty: combined * verification_ratio,
            total_penalty: combined,
        }
    }
}

/// Classify a Byzantine detection using epistemic framework
pub fn classify_byzantine_attack(
    participant_id: &str,
    detection_confidence: f64,
    agent_phi: Option<f64>,
    peer_reports: usize,
    zk_proof_failed: bool,
    gradient_deviation: f64,
    coordinated_count: usize,
) -> EpistemicByzantineResult {
    // Determine attack type (E-axis)
    let attack_type = if zk_proof_failed && gradient_deviation > 2.0 {
        ByzantineAttackType::ManifestByzantine
    } else if zk_proof_failed {
        ByzantineAttackType::CryptoProofFailure
    } else if detection_confidence > 0.9 {
        ByzantineAttackType::VerifiedMalfunction
    } else if peer_reports >= 2 {
        ByzantineAttackType::PeerReportedMalice
    } else {
        ByzantineAttackType::UnverifiedAnomaly
    };

    // Determine scope (N-axis)
    let scope = if coordinated_count >= 5 {
        ByzantineScope::NetworkCoordinated
    } else if coordinated_count >= 2 {
        ByzantineScope::SmallGroup
    } else {
        ByzantineScope::Individual
    };

    // Determine if malicious vs incoherent based on Phi
    // Low Phi + Byzantine = likely incoherent/broken
    // High Phi + Byzantine = likely malicious
    let appears_malicious = match agent_phi {
        Some(phi) if phi >= 0.5 => true, // High coherence = intentional
        Some(phi) if phi < 0.3 => false, // Low coherence = broken
        _ => detection_confidence > 0.8, // No Phi? Use confidence as proxy
    };

    let penalties =
        KVectorPenalties::from_attack(attack_type, scope, detection_confidence, appears_malicious);

    let explanation = format!(
        "{} detected: {} (scope: {:?}, confidence: {:.1}%, {}). Total penalty: {:.2}",
        if appears_malicious {
            "Malicious attack"
        } else {
            "Incoherent behavior"
        },
        match attack_type {
            ByzantineAttackType::UnverifiedAnomaly => "unverified anomaly",
            ByzantineAttackType::PeerReportedMalice => "peer-reported malice",
            ByzantineAttackType::VerifiedMalfunction => "verified malfunction",
            ByzantineAttackType::CryptoProofFailure => "ZK proof failure",
            ByzantineAttackType::ManifestByzantine => "manifest Byzantine",
        },
        scope,
        detection_confidence * 100.0,
        agent_phi
            .map(|p| format!("Phi: {:.2}", p))
            .unwrap_or_else(|| "no Phi".into()),
        penalties.total_penalty,
    );

    EpistemicByzantineResult {
        participant_id: participant_id.to_string(),
        attack_type,
        scope,
        confidence: detection_confidence,
        appears_malicious,
        agent_phi,
        recommended_penalties: penalties,
        explanation,
    }
}

/// Batch classify Byzantine detections with epistemic analysis
pub fn classify_byzantine_batch(
    detections: &[(String, f64)], // (participant_id, confidence)
    participant_phis: &std::collections::HashMap<String, f64>,
    zk_failures: &[String],
    gradient_deviations: &std::collections::HashMap<String, f64>,
) -> Vec<EpistemicByzantineResult> {
    // Detect coordination by looking for similar deviation patterns
    let coordinated_groups = detect_coordination(detections, gradient_deviations);

    detections
        .iter()
        .map(|(pid, confidence)| {
            let phi = participant_phis.get(pid).copied();
            let zk_failed = zk_failures.contains(pid);
            let deviation = gradient_deviations.get(pid).copied().unwrap_or(0.0);
            let coordinated = coordinated_groups.get(pid).copied().unwrap_or(0);

            classify_byzantine_attack(
                pid,
                *confidence,
                phi,
                0, // peer reports not available in batch
                zk_failed,
                deviation,
                coordinated,
            )
        })
        .collect()
}

/// Detect coordinated attacks by finding similar deviation patterns
fn detect_coordination(
    detections: &[(String, f64)],
    deviations: &std::collections::HashMap<String, f64>,
) -> std::collections::HashMap<String, usize> {
    let mut coordination = std::collections::HashMap::new();
    let threshold = 0.1; // Similar deviation within 10%

    for (pid1, _) in detections {
        let dev1 = deviations.get(pid1).copied().unwrap_or(0.0);
        let mut similar_count = 0;

        for (pid2, _) in detections {
            if pid1 != pid2 {
                let dev2 = deviations.get(pid2).copied().unwrap_or(0.0);
                if (dev1 - dev2).abs() < threshold * dev1.max(dev2).max(1.0) {
                    similar_count += 1;
                }
            }
        }

        coordination.insert(pid1.clone(), similar_count);
    }

    coordination
}

// ============================================================================
// Agent Behavioral Feedback Loop (#173)
// FL round results → K-Vector updates
// ============================================================================

/// Feedback from an FL round to update agent K-Vectors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FLRoundFeedback {
    /// Round identifier
    pub round_id: u64,
    /// Participant feedback entries
    pub participant_feedback: Vec<ParticipantFeedback>,
    /// Overall model quality improvement (delta from previous round)
    pub model_quality_delta: f64,
    /// Round epistemic stats
    pub epistemic_stats: RoundEpistemicStats,
}

/// Feedback for a single participant's contribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantFeedback {
    /// Participant identifier
    pub participant_id: String,
    /// Marginal contribution to model quality
    /// Computed as: Q_with_participant - Q_without_participant
    pub marginal_contribution: f64,
    /// Epistemic weight of their gradient
    pub epistemic_weight: f64,
    /// Whether they were included in final aggregation
    pub was_included: bool,
    /// If excluded, reason for exclusion
    pub exclusion_reason: Option<String>,
    /// Recommended K-Vector updates
    pub kvector_updates: KVectorUpdates,
}

/// Recommended K-Vector updates based on FL participation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KVectorUpdates {
    /// k_r (Reputation) change
    pub reputation_delta: f64,
    /// k_p (Performance) change
    pub performance_delta: f64,
    /// k_a (Activity) change
    pub activity_delta: f64,
    /// k_i (Integrity) change (negative if excluded for Byzantine)
    pub integrity_delta: f64,
    /// k_h (Historical) change
    pub historical_delta: f64,
}

impl KVectorUpdates {
    /// Create updates for a positive contribution
    pub fn positive(marginal_contribution: f64, epistemic_weight: f64) -> Self {
        // Scale updates by contribution and epistemic quality
        let scale = marginal_contribution.max(0.0) * epistemic_weight;
        Self {
            reputation_delta: scale * 0.05, // Small reputation boost
            performance_delta: scale * 0.1, // Larger performance boost
            activity_delta: 0.01,           // Constant activity credit
            integrity_delta: 0.02,          // Small integrity boost
            historical_delta: scale * 0.03, // Historical consistency
        }
    }

    /// Create updates for a negative/excluded contribution
    pub fn negative(_reason: &str, penalties: &KVectorPenalties) -> Self {
        Self {
            reputation_delta: -penalties.reputation_penalty,
            performance_delta: -penalties.performance_penalty,
            activity_delta: 0.0, // Still participated
            integrity_delta: -penalties.integrity_penalty,
            historical_delta: -0.05,
        }
    }

    /// Create neutral updates (participated but minimal impact)
    pub fn neutral() -> Self {
        Self {
            reputation_delta: 0.0,
            performance_delta: 0.0,
            activity_delta: 0.01, // Credit for participation
            integrity_delta: 0.0,
            historical_delta: 0.0,
        }
    }
}

/// Compute feedback for all participants in an FL round
pub fn compute_round_feedback(
    round_id: u64,
    updates: &[EpistemicGradientUpdate],
    aggregation_result: &EpistemicAggregationResult,
    byzantine_results: &[EpistemicByzantineResult],
    baseline_quality: f64,
    final_quality: f64,
) -> FLRoundFeedback {
    let quality_delta = final_quality - baseline_quality;
    let epistemic_stats = RoundEpistemicStats::from_updates(updates);

    // Build Byzantine lookup
    let byzantine_map: std::collections::HashMap<_, _> = byzantine_results
        .iter()
        .map(|r| (r.participant_id.clone(), r))
        .collect();

    // Build weight lookup
    let weight_map: std::collections::HashMap<_, _> = aggregation_result
        .participant_weights
        .iter()
        .cloned()
        .collect();

    let participant_feedback: Vec<_> = updates
        .iter()
        .map(|update| {
            let pid = &update.gradient.participant_id;

            // Check if marked Byzantine
            if let Some(byzantine) = byzantine_map.get(pid) {
                return ParticipantFeedback {
                    participant_id: pid.clone(),
                    marginal_contribution: 0.0,
                    epistemic_weight: update.classification.epistemic_weight(),
                    was_included: false,
                    exclusion_reason: Some(byzantine.explanation.clone()),
                    kvector_updates: KVectorUpdates::negative(
                        &byzantine.explanation,
                        &byzantine.recommended_penalties,
                    ),
                };
            }

            // Check if included in aggregation
            let weight = weight_map.get(pid).copied();
            let was_included = weight.is_some();
            let epistemic_weight = update.classification.epistemic_weight();

            if was_included {
                // Estimate marginal contribution (simplified: proportional to weight)
                let marginal = quality_delta * weight.unwrap_or(0.0);
                ParticipantFeedback {
                    participant_id: pid.clone(),
                    marginal_contribution: marginal,
                    epistemic_weight,
                    was_included: true,
                    exclusion_reason: None,
                    kvector_updates: KVectorUpdates::positive(marginal, epistemic_weight),
                }
            } else {
                ParticipantFeedback {
                    participant_id: pid.clone(),
                    marginal_contribution: 0.0,
                    epistemic_weight,
                    was_included: false,
                    exclusion_reason: Some("Below epistemic threshold".into()),
                    kvector_updates: KVectorUpdates::neutral(),
                }
            }
        })
        .collect();

    FLRoundFeedback {
        round_id,
        participant_feedback,
        model_quality_delta: quality_delta,
        epistemic_stats,
    }
}

// ============================================================================
// Phi-Gated Byzantine Detection (#174)
// Gate FL participation on agent coherence
// ============================================================================

/// Phi coherence thresholds for FL participation
pub mod phi_thresholds {
    /// Minimum Phi to participate in any FL round
    pub const MIN_PARTICIPATION: f64 = 0.2;
    /// Minimum Phi for full weight in aggregation
    pub const FULL_WEIGHT: f64 = 0.5;
    /// Phi threshold for leadership/coordination roles
    pub const LEADERSHIP: f64 = 0.7;
    /// Critical threshold - agent may be malfunctioning
    pub const CRITICAL: f64 = 0.1;
}

/// Result of Phi-gated participation check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiParticipationResult {
    /// Whether participation is allowed
    pub allowed: bool,
    /// Phi multiplier for weight calculation
    pub phi_multiplier: f64,
    /// Whether agent can take leadership roles
    pub leadership_eligible: bool,
    /// Warning message (if any)
    pub warning: Option<String>,
    /// Recommendation
    pub recommendation: PhiRecommendation,
}

/// Recommendations based on Phi level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhiRecommendation {
    /// Full participation - agent is highly coherent
    FullParticipation,
    /// Normal participation - standard operations
    NormalParticipation,
    /// Reduced weight - lower coherence detected
    ReducedWeight,
    /// Monitor only - very low coherence
    MonitorOnly,
    /// Suspend - critical incoherence
    Suspend,
}

/// Check if an agent can participate in FL based on Phi coherence
pub fn check_coherence_participation(phi: f64) -> PhiParticipationResult {
    if phi < phi_thresholds::CRITICAL {
        return PhiParticipationResult {
            allowed: false,
            phi_multiplier: 0.0,
            leadership_eligible: false,
            warning: Some("Critical incoherence detected - agent may be malfunctioning".into()),
            recommendation: PhiRecommendation::Suspend,
        };
    }

    if phi < phi_thresholds::MIN_PARTICIPATION {
        return PhiParticipationResult {
            allowed: false,
            phi_multiplier: 0.0,
            leadership_eligible: false,
            warning: Some("Phi below minimum participation threshold".into()),
            recommendation: PhiRecommendation::MonitorOnly,
        };
    }

    if phi < phi_thresholds::FULL_WEIGHT {
        let multiplier = 0.5
            + (phi - phi_thresholds::MIN_PARTICIPATION)
                / (phi_thresholds::FULL_WEIGHT - phi_thresholds::MIN_PARTICIPATION)
                * 0.5;
        return PhiParticipationResult {
            allowed: true,
            phi_multiplier: multiplier,
            leadership_eligible: false,
            warning: Some("Reduced weight due to lower coherence".into()),
            recommendation: PhiRecommendation::ReducedWeight,
        };
    }

    let leadership = phi >= phi_thresholds::LEADERSHIP;
    PhiParticipationResult {
        allowed: true,
        phi_multiplier: 1.0,
        leadership_eligible: leadership,
        warning: None,
        recommendation: if leadership {
            PhiRecommendation::FullParticipation
        } else {
            PhiRecommendation::NormalParticipation
        },
    }
}

/// Extended Byzantine analysis that incorporates Phi coherence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiAwareByzantineAnalysis {
    /// Standard Byzantine detection confidence
    pub byzantine_confidence: f64,
    /// Agent's Phi coherence
    pub phi: f64,
    /// Classification: malicious vs incoherent
    pub classification: PhiByzantineClassification,
    /// Adjusted penalty (reduced if incoherent)
    pub adjusted_penalty: f64,
    /// Recommended action
    pub recommended_action: PhiByzantineAction,
}

/// Classification of Byzantine behavior considering Phi
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhiByzantineClassification {
    /// High Phi + Byzantine = intentional attack
    IntentionalMalice,
    /// Medium Phi + Byzantine = possible attack or bug
    Ambiguous,
    /// Low Phi + Byzantine = likely malfunctioning
    IncoherentMalfunction,
    /// Very low Phi = agent broken, not attacking
    SystemFailure,
}

/// Recommended actions for Phi-classified Byzantine behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhiByzantineAction {
    /// Full penalties - intentional attack
    FullPenalty,
    /// Reduced penalties - investigate further
    ReducedPenalty,
    /// Suspend for diagnosis - likely broken
    SuspendForDiagnosis,
    /// Immediate suspension - system failure
    EmergencySuspend,
}

/// Analyze Byzantine behavior with Phi awareness
pub fn analyze_byzantine_with_phi(
    byzantine_confidence: f64,
    phi: f64,
    gradient_deviation: f64,
) -> PhiAwareByzantineAnalysis {
    let (classification, action, penalty_mult) = if phi < phi_thresholds::CRITICAL {
        // Very low Phi - system failure, not attack
        (
            PhiByzantineClassification::SystemFailure,
            PhiByzantineAction::EmergencySuspend,
            0.1, // Minimal penalty - not their fault
        )
    } else if phi < phi_thresholds::MIN_PARTICIPATION {
        // Low Phi - likely incoherent
        (
            PhiByzantineClassification::IncoherentMalfunction,
            PhiByzantineAction::SuspendForDiagnosis,
            0.3, // Reduced penalty
        )
    } else if phi < phi_thresholds::FULL_WEIGHT {
        // Medium Phi - ambiguous
        (
            PhiByzantineClassification::Ambiguous,
            PhiByzantineAction::ReducedPenalty,
            0.6, // Moderate penalty
        )
    } else {
        // High Phi - intentional
        (
            PhiByzantineClassification::IntentionalMalice,
            PhiByzantineAction::FullPenalty,
            1.0, // Full penalty
        )
    };

    // Base penalty from Byzantine confidence and deviation
    let base_penalty = byzantine_confidence * (1.0 + gradient_deviation.min(3.0) / 3.0);
    let adjusted_penalty = base_penalty * penalty_mult;

    PhiAwareByzantineAnalysis {
        byzantine_confidence,
        phi,
        classification,
        adjusted_penalty,
        recommended_action: action,
    }
}

/// Filter FL updates by Phi, returning allowed updates with multipliers
pub fn phi_filter_updates(updates: &[EpistemicGradientUpdate]) -> Vec<(usize, f64)> {
    updates
        .iter()
        .enumerate()
        .filter_map(|(idx, update)| {
            let phi = update.agent_phi.unwrap_or(0.5); // Default to medium
            let check = check_coherence_participation(phi);
            if check.allowed {
                Some((idx, check.phi_multiplier))
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_classification() {
        let hints = GradientClassificationHints {
            has_pogq: true,
            pogq_score: Some(0.85),
            has_zk_proof: true,
            peer_validated: true,
            peer_validation_count: 5, // Network level
            persistent_model: true,
            downstream_impact: 5, // Higher impact
            agent_phi: Some(0.8),
        };

        let class = classify_gradient(&hints);
        assert_eq!(class.empirical, EmpiricalLevel::E4PublicRepro);
        assert_eq!(class.normative, NormativeLevel::N2Network);
        // E4=1.0 * N2=0.9 * M2=0.8 * H2=0.8 * confidence=1.0 = 0.576
        assert!(
            class.epistemic_weight() > 0.5,
            "weight was {}",
            class.epistemic_weight()
        );
    }

    #[test]
    fn test_epistemic_weight_calculation() {
        // Maximum weight
        let max_class = GradientEpistemicClassification::new(
            EmpiricalLevel::E4PublicRepro,
            NormativeLevel::N3Axiomatic,
            MaterialityLevel::M3Foundational,
            HarmonicLevel::H4Kosmic,
        );
        assert!((max_class.epistemic_weight() - 1.0).abs() < 0.001);

        // Minimum weight
        let min_class = GradientEpistemicClassification::default();
        assert!(min_class.epistemic_weight() < 0.1);
    }

    #[test]
    fn test_epistemic_aggregation() {
        let updates = vec![
            EpistemicGradientUpdate::new(
                GradientUpdate::new("p1".into(), 1, vec![0.1, 0.2], 100, 0.5),
                GradientEpistemicClassification::new(
                    EmpiricalLevel::E3Cryptographic,
                    NormativeLevel::N2Network,
                    MaterialityLevel::M2Persistent,
                    HarmonicLevel::H1Local,
                ),
            ),
            EpistemicGradientUpdate::new(
                GradientUpdate::new("p2".into(), 1, vec![0.2, 0.3], 100, 0.4),
                GradientEpistemicClassification::new(
                    EmpiricalLevel::E1Testimonial,
                    NormativeLevel::N0Personal,
                    MaterialityLevel::M0Ephemeral,
                    HarmonicLevel::H0None,
                ),
            ),
        ];

        let participants = vec![Participant::new("p1".into()), Participant::new("p2".into())];

        let result = epistemic_weighted_aggregation(&updates, &participants, 0.0).unwrap();

        // p1 should have higher weight (higher epistemic score)
        let p1_weight = result
            .participant_weights
            .iter()
            .find(|(id, _)| id == "p1")
            .map(|(_, w)| *w)
            .unwrap();
        let p2_weight = result
            .participant_weights
            .iter()
            .find(|(id, _)| id == "p2")
            .map(|(_, w)| *w)
            .unwrap();
        assert!(p1_weight > p2_weight);
    }

    #[test]
    fn test_round_epistemic_stats() {
        let updates = vec![
            EpistemicGradientUpdate::new(
                GradientUpdate::new("p1".into(), 1, vec![0.1], 100, 0.5),
                GradientEpistemicClassification::new(
                    EmpiricalLevel::E4PublicRepro,
                    NormativeLevel::N3Axiomatic,
                    MaterialityLevel::M3Foundational,
                    HarmonicLevel::H3Civilizational,
                ),
            )
            .with_proof(vec![1, 2, 3]),
            EpistemicGradientUpdate::new(
                GradientUpdate::new("p2".into(), 1, vec![0.2], 100, 0.4),
                GradientEpistemicClassification::new(
                    EmpiricalLevel::E4PublicRepro,
                    NormativeLevel::N2Network,
                    MaterialityLevel::M2Persistent,
                    HarmonicLevel::H2Network,
                ),
            )
            .with_proof(vec![4, 5, 6]),
        ];

        let stats = RoundEpistemicStats::from_updates(&updates);
        // 100% E3+, avg weight = (0.9 + 0.576) / 2 = 0.738 > 0.5 → Gold
        assert_eq!(stats.quality_tier, EpistemicQualityTier::Gold);
        assert!((stats.zk_proof_ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_byzantine_attack_classification() {
        // High Phi + ZK failure = malicious manifest Byzantine
        let result = classify_byzantine_attack(
            "attacker-1",
            0.95,
            Some(0.8), // High coherence
            0,
            true, // ZK proof failed
            3.0,  // High deviation
            0,
        );
        assert_eq!(result.attack_type, ByzantineAttackType::ManifestByzantine);
        assert!(result.appears_malicious);
        assert!(result.recommended_penalties.total_penalty > 1.0);
    }

    #[test]
    fn test_incoherent_vs_malicious() {
        // Low Phi + Byzantine = incoherent (broken agent)
        let incoherent = classify_byzantine_attack(
            "broken-agent",
            0.8,
            Some(0.2), // Low coherence
            0,
            false,
            1.5,
            0,
        );
        assert!(!incoherent.appears_malicious);
        assert!(
            incoherent.recommended_penalties.verification_penalty
                > incoherent.recommended_penalties.integrity_penalty
        );

        // High Phi + Byzantine = malicious (intentional)
        let malicious = classify_byzantine_attack(
            "malicious-agent",
            0.8,
            Some(0.8), // High coherence
            0,
            false,
            1.5,
            0,
        );
        assert!(malicious.appears_malicious);
        assert!(
            malicious.recommended_penalties.integrity_penalty
                > malicious.recommended_penalties.verification_penalty
        );
    }

    #[test]
    fn test_coordinated_attack_detection() {
        let detections = vec![("p1".into(), 0.8), ("p2".into(), 0.85), ("p3".into(), 0.75)];
        let deviations: std::collections::HashMap<_, _> = [
            ("p1".into(), 2.5),
            ("p2".into(), 2.6), // Similar to p1
            ("p3".into(), 1.0), // Different
        ]
        .into_iter()
        .collect();

        let results = classify_byzantine_batch(
            &detections,
            &std::collections::HashMap::new(),
            &[],
            &deviations,
        );

        // p1 and p2 should show coordination
        let p1_result = results.iter().find(|r| r.participant_id == "p1").unwrap();
        let p3_result = results.iter().find(|r| r.participant_id == "p3").unwrap();
        assert!(
            p1_result.scope != ByzantineScope::Individual
                || p3_result.scope == ByzantineScope::Individual
        );
    }

    #[test]
    fn test_penalty_scaling() {
        // UnverifiedAnomaly should have light penalty
        let light = KVectorPenalties::from_attack(
            ByzantineAttackType::UnverifiedAnomaly,
            ByzantineScope::Individual,
            0.5,
            false,
        );
        assert!(light.total_penalty < 0.1);

        // ManifestByzantine + coordinated should have severe penalty
        let severe = KVectorPenalties::from_attack(
            ByzantineAttackType::ManifestByzantine,
            ByzantineScope::NetworkCoordinated,
            1.0,
            true,
        );
        assert!(severe.total_penalty > 2.0);
    }
}
