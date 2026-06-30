use serde::{Deserialize, Serialize};
use std::fmt;

// ==============================================================================
// E-AXIS: EMPIRICAL VERIFIABILITY
// ==============================================================================

/// E-Axis: How do we VERIFY this knowledge claim?
///
/// Ranges from unverifiable (E0) to publicly reproducible (E4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum EmpiricalLevel {
    /// E0: Inferred from theory, no empirical evidence
    E0Null = 0,
    /// E1: Single observation (testimonial)
    E1Testimonial = 1,
    /// E2: Multiple observations, verified internally
    E2PrivatelyVerifiable = 2,
    /// E3: Cryptographically proven (counterfactual proof, ZKP)
    E3CryptographicallyProven = 3,
    /// E4: Open data + code, anyone can reproduce
    E4PubliclyReproducible = 4,
}

impl EmpiricalLevel {
    pub fn level(&self) -> u8 {
        *self as u8
    }

    pub fn name(&self) -> &str {
        match self {
            Self::E0Null => "Null",
            Self::E1Testimonial => "Testimonial",
            Self::E2PrivatelyVerifiable => "Privately Verifiable",
            Self::E3CryptographicallyProven => "Cryptographically Proven",
            Self::E4PubliclyReproducible => "Publicly Reproducible",
        }
    }

    pub fn abbreviation(&self) -> &str {
        match self {
            Self::E0Null => "E0",
            Self::E1Testimonial => "E1",
            Self::E2PrivatelyVerifiable => "E2",
            Self::E3CryptographicallyProven => "E3",
            Self::E4PubliclyReproducible => "E4",
        }
    }
}

impl fmt::Display for EmpiricalLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.abbreviation())
    }
}

// ==============================================================================
// N-AXIS: NORMATIVE AUTHORITY
// ==============================================================================

/// N-Axis: WHO agrees this knowledge claim is valid?
///
/// Ranges from personal (N0) to axiomatic truth (N3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum NormativeLevel {
    /// N0: Only this system instance
    N0Personal = 0,
    /// N1: Local agent community consensus
    N1Communal = 1,
    /// N2: Global network consensus
    N2Network = 2,
    /// N3: Mathematical/constitutional truth
    N3Axiomatic = 3,
}

impl NormativeLevel {
    pub fn level(&self) -> u8 {
        *self as u8
    }

    pub fn name(&self) -> &str {
        match self {
            Self::N0Personal => "Personal",
            Self::N1Communal => "Communal",
            Self::N2Network => "Network",
            Self::N3Axiomatic => "Axiomatic",
        }
    }

    pub fn abbreviation(&self) -> &str {
        match self {
            Self::N0Personal => "N0",
            Self::N1Communal => "N1",
            Self::N2Network => "N2",
            Self::N3Axiomatic => "N3",
        }
    }
}

impl fmt::Display for NormativeLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.abbreviation())
    }
}

// ==============================================================================
// M-AXIS: MATERIALITY / PERMANENCE
// ==============================================================================

/// M-Axis: How PERMANENT is this knowledge?
///
/// Ranges from ephemeral (M0) to foundational (M3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum MaterialityLevel {
    /// M0: Valid only for this reasoning session
    M0Ephemeral = 0,
    /// M1: Valid until model updates
    M1Temporal = 1,
    /// M2: Long-term archived knowledge
    M2Persistent = 2,
    /// M3: Core principle, effectively permanent
    M3Foundational = 3,
}

impl MaterialityLevel {
    pub fn level(&self) -> u8 {
        *self as u8
    }

    pub fn name(&self) -> &str {
        match self {
            Self::M0Ephemeral => "Ephemeral",
            Self::M1Temporal => "Temporal",
            Self::M2Persistent => "Persistent",
            Self::M3Foundational => "Foundational",
        }
    }

    pub fn abbreviation(&self) -> &str {
        match self {
            Self::M0Ephemeral => "M0",
            Self::M1Temporal => "M1",
            Self::M2Persistent => "M2",
            Self::M3Foundational => "M3",
        }
    }
}

impl fmt::Display for MaterialityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.abbreviation())
    }
}

// ==============================================================================
// PRISMATIC EPISTEMIC CONTEXT
// ==============================================================================

/// Prismatic epistemic context — different knowledge traditions apply
/// legitimately different weights to E/N/M axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpistemicContext {
    Scientific,
    Governance,
    Personal,
    Indigenous,
    Contemplative,
    Emergency,
    Standard,
}

impl EpistemicContext {
    pub fn weights(&self) -> (f64, f64, f64) {
        match self {
            Self::Scientific => (0.50, 0.30, 0.20),
            Self::Governance => (0.30, 0.45, 0.25),
            Self::Personal => (0.33, 0.33, 0.34),
            Self::Indigenous => (0.25, 0.40, 0.35),
            Self::Contemplative => (0.20, 0.30, 0.50),
            Self::Emergency => (0.60, 0.25, 0.15),
            Self::Standard => (0.40, 0.35, 0.25),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Scientific => "Scientific",
            Self::Governance => "Governance",
            Self::Personal => "Personal",
            Self::Indigenous => "Indigenous",
            Self::Contemplative => "Contemplative",
            Self::Emergency => "Emergency",
            Self::Standard => "Standard",
        }
    }
}

impl Default for EpistemicContext {
    fn default() -> Self {
        Self::Standard
    }
}

// ==============================================================================
// EPISTEMIC COORDINATE (the full 3D+context position)
// ==============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EpistemicCoordinate {
    pub empirical: EmpiricalLevel,
    pub normative: NormativeLevel,
    pub materiality: MaterialityLevel,
    pub context: EpistemicContext,
}

// ==============================================================================
// EPISTEMIC PROVENANCE (where knowledge comes from)
// ==============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GroundingLevel {
    Sensorimotor,
    Temporal,
    Social,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EpistemicProvenance {
    Embodied {
        level: GroundingLevel,
        platform: String,
        observation_confidence: f32,
    },
    Inferred {
        method: String,
        confidence: f32,
    },
    Collective {
        corroboration_count: u32,
        network_coverage: f32,
    },
    Unknown,
}

impl Default for EpistemicProvenance {
    fn default() -> Self {
        Self::Unknown
    }
}

// ==============================================================================
// FACT CHECK TYPES
// ==============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FactCheckVerdict {
    True,
    False,
    Mixed,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactCheckResult {
    pub claim: String,
    pub verdict: FactCheckVerdict,
    pub confidence: f32,
    pub source_ids: Vec<String>,
    pub epistemic_position: Option<EpistemicCoordinate>,
}

// ==============================================================================
// GLOBAL LEDGER TYPE
// ==============================================================================

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone, Copy)]
pub enum GlobalClaimStatus {
    Heuristic,
    Formalized,
    Proven,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GlobalClaim {
    pub domain: String,
    pub name: String,
    pub status: GlobalClaimStatus,
    pub formal_proof_path: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct GlobalEpistemicLedger {
    pub claims: Vec<GlobalClaim>,
}

impl GlobalEpistemicLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn audit_all(&self) -> bool {
        self.claims.iter().all(|c| match c.status {
            GlobalClaimStatus::Proven => c.formal_proof_path.is_some(),
            _ => true,
        })
    }
}
impl EpistemicCoordinate {
    pub fn new(
        empirical: EmpiricalLevel,
        normative: NormativeLevel,
        materiality: MaterialityLevel,
        context: EpistemicContext,
    ) -> Self {
        Self {
            empirical,
            normative,
            materiality,
            context,
        }
    }
}
