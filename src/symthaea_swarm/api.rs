// symthaea_swarm/api.rs - Core types and traits for Symthaea ↔ Mycelix integration
//
// This module defines the interface between Symthaea's Holographic Liquid Brain
// and the Mycelix Protocol (DKG, MATL, MFDI).

use serde::{Deserialize, Serialize};
use std::time::Duration;
use uuid::Uuid;

// ============================================================================
// Core Types
// ============================================================================

pub type Did = String;
pub type ClaimId = Uuid;
pub type Hypervector = Vec<f32>;

// ============================================================================
// Epistemic Cube Types (LEM v2.0)
// ============================================================================

/// E-Axis: Empirical verification level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EpistemicTierE {
    E0, // Null (no claim)
    E1, // Testimonial (personal observation)
    E2, // Privately Verifiable (guild/audit verified)
    E3, // Cryptographic Proof (zk-STARK, signatures)
    E4, // Publicly Reproducible (anyone can verify)
}

/// N-Axis: Normative/subjective level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum NormativeTierN {
    N0, // Personal (individual preference)
    N1, // Communal (local consensus)
    N2, // Network Consensus (DAO vote)
    N3, // Axiomatic (constitutional principle)
}

/// M-Axis: Materiality/persistence level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MaterialityTierM {
    M0, // Ephemeral (cache, session)
    M1, // Temporal (valid for period)
    M2, // Persistent (archive)
    M3, // Foundational (immutable)
}

// ============================================================================
// Symthaea Pattern Types
// ============================================================================

/// A learned pattern from Symthaea's experience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymthaeaPattern {
    pub pattern_id: Uuid,
    pub problem_vector: Hypervector,
    pub solution_vector: Hypervector,
    pub success_rate: f64,
    pub context: String,
    pub tested_on_nixos: String,

    // Epistemic classification
    pub e_tier: EpistemicTierE,
    pub n_tier: NormativeTierN,
    pub m_tier: MaterialityTierM,
}

/// Query for searching patterns in DKG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternQuery {
    pub query_vector: Hypervector,
    pub min_similarity: f64,
    pub min_e_tier: Option<EpistemicTierE>,
    pub min_n_tier: Option<NormativeTierN>,
    pub min_m_tier: Option<MaterialityTierM>,
}

// ============================================================================
// Epistemic Claim (DKG Format)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicClaim {
    pub claim_id: ClaimId,
    pub claim_hash: String,
    pub submitted_by_did: Did,
    pub submitter_type: SubmitterType,
    pub epistemic_tier_e: String,
    pub epistemic_tier_n: String,
    pub epistemic_tier_m: String,
    pub claim_type: ClaimType,
    pub content: ClaimContent,
    pub verifiability: Verifiability,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubmitterType {
    Human,
    InstrumentalActor,
    DAO,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClaimType {
    Testimony,
    Measurement,
    Computation,
    Attestation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimContent {
    pub format: String,
    pub body: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verifiability {
    pub method: String,
    pub status: String,
    pub proof_cid: Option<String>,
}

/// Epistemic Claim with MATL trust score attached
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluatedClaim {
    pub claim: EpistemicClaim,
    pub trust_score: CompositeTrustScore,
    pub similarity: f64,
}

// ============================================================================
// MATL Trust Scoring
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeTrustScore {
    pub composite: f64,
    pub pogq_score: f64,
    pub tcdm_score: f64,
    pub entropy_score: f64,
    pub reputation_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CartelRisk {
    pub risk_score: f64,
    pub tcdm_score: f64,
    pub temporal_anomaly: bool,
    pub community_clustering: bool,
}

// ============================================================================
// MFDI Identity
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MycelixIdentity {
    pub did: Did,
    pub identity_type: IdentityType,
    pub operator_did: Option<Did>,
    pub humanity_score: Option<f64>,
    pub reputation: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IdentityType {
    Human,
    InstrumentalActor,
    DAO,
}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum SwarmError {
    #[error("Network error: {0}")]
    Network(String),

    #[error("Authentication error: {0}")]
    Auth(String),

    #[error("Invalid claim: {0}")]
    InvalidClaim(String),

    #[error("Trust score too low: {0}")]
    TrustTooLow(f64),

    #[error("Cartel detected: {0}")]
    CartelDetected(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(String),
}

// ============================================================================
// Swarm Client Traits
// ============================================================================

/// DKG (Decentralized Knowledge Graph) client
#[async_trait::async_trait]
pub trait DkgClient: Send + Sync {
    /// Publish a learned pattern as an Epistemic Claim
    async fn publish_pattern_claim(&self, pattern: SymthaeaPattern) -> Result<ClaimId, SwarmError>;

    /// Retrieve a specific claim by ID
    async fn get_claim(&self, claim_id: ClaimId) -> Result<Option<EpistemicClaim>, SwarmError>;

    /// Query for patterns matching criteria
    async fn query_claims(&self, query: PatternQuery, limit: usize) -> Result<Vec<EvaluatedClaim>, SwarmError>;
}

/// MATL (Mycelix Adaptive Trust Layer) client
#[async_trait::async_trait]
pub trait MatlClient: Send + Sync {
    /// Get trust score for a specific claim
    async fn trust_for_claim(&self, claim_id: ClaimId) -> Result<CompositeTrustScore, SwarmError>;

    /// Get trust score for an agent (by DID)
    async fn trust_for_agent(&self, did: &Did) -> Result<CompositeTrustScore, SwarmError>;

    /// Check cartel risk for an agent
    async fn cartel_risk_for_agent(&self, did: &Did, window: Duration) -> Result<CartelRisk, SwarmError>;
}

/// MFDI (Multi-Factor Decentralized Identity) client
#[async_trait::async_trait]
pub trait MfdiClient: Send + Sync {
    /// Ensure Symthaea instance is registered as Instrumental Actor
    async fn ensure_instrumental_identity(
        &self,
        model_type: &str,
        model_version: &str,
        operator_did: &Did,
    ) -> Result<MycelixIdentity, SwarmError>;
}

/// Combined Symthaea ↔ Mycelix swarm client
pub trait SymthaeaSwarmClient: DkgClient + MatlClient + MfdiClient + Send + Sync {}

// Auto-implement SymthaeaSwarmClient for any type implementing all three traits
impl<T> SymthaeaSwarmClient for T where T: DkgClient + MatlClient + MfdiClient + Send + Sync {}

// ============================================================================
// Helper Functions
// ============================================================================

impl SymthaeaPattern {
    /// Convert to Epistemic Claim for DKG submission
    pub fn to_epistemic_claim(&self, did: &Did) -> EpistemicClaim {
        use sha2::{Digest, Sha256};

        let content_json = serde_json::json!({
            "problem_vector": self.problem_vector,
            "solution_vector": self.solution_vector,
            "success_rate": self.success_rate,
            "tested_on": self.tested_on_nixos,
            "context": self.context,
        });

        let content_str = serde_json::to_string(&content_json).unwrap();
        let claim_hash = format!("{:x}", Sha256::digest(content_str.as_bytes()));

        EpistemicClaim {
            claim_id: self.pattern_id,
            claim_hash,
            submitted_by_did: did.clone(),
            submitter_type: SubmitterType::InstrumentalActor,
            epistemic_tier_e: format!("{:?}", self.e_tier),
            epistemic_tier_n: format!("{:?}", self.n_tier),
            epistemic_tier_m: format!("{:?}", self.m_tier),
            claim_type: ClaimType::Computation,
            content: ClaimContent {
                format: "application/json".to_string(),
                body: content_json,
            },
            verifiability: Verifiability {
                method: "SymthaeaLocalValidation".to_string(),
                status: "SelfReported".to_string(),
                proof_cid: None,
            },
        }
    }
}

impl CompositeTrustScore {
    /// Calculate composite score from components
    /// Formula: (PoGQ × 0.4) + (TCDM × 0.3) + (Entropy × 0.3)
    pub fn calculate(pogq: f64, tcdm: f64, entropy: f64, reputation: f64) -> Self {
        let composite = (pogq * 0.4) + (tcdm * 0.3) + (entropy * 0.3);

        Self {
            composite,
            pogq_score: pogq,
            tcdm_score: tcdm,
            entropy_score: entropy,
            reputation_weight: reputation,
        }
    }

    /// Is this score above threshold for auto-apply?
    pub fn is_auto_apply(&self, threshold: f64) -> bool {
        self.composite >= threshold
    }
}
