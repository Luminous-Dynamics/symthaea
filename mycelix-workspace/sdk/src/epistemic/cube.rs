//! Epistemic Cube - 3D Classification System
//!
//! Every claim in Mycelix is classified along three orthogonal dimensions:
//! - Empirical: How can this be verified?
//! - Normative: Who agrees with this?
//! - Materiality: How long does this matter?

// Epistemic cube types — all variants documented via doc comments

use serde::{Deserialize, Serialize};

#[cfg(feature = "ts-export")]
use ts_rs::TS;

/// Empirical Axis (E): How to verify the claim
///
/// From least to most verifiable:
/// - E0: Unverifiable belief
/// - E1: Personal testimony
/// - E2: Private verification (audit)
/// - E3: Cryptographic proof (ZKP)
/// - E4: Publicly reproducible
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/epistemic/"))]
#[repr(u8)]
pub enum EmpiricalLevel {
    /// E0: Null - Unverifiable belief, personal opinion
    E0Null = 0,

    /// E1: Testimonial - Personal attestation, witness statement
    E1Testimonial = 1,

    /// E2: Privately Verifiable - Audit guild, trusted third party
    E2PrivateVerify = 2,

    /// E3: Cryptographically Proven - ZKP, digital signature
    E3Cryptographic = 3,

    /// E4: Publicly Reproducible - Open data, verifiable computation
    E4PublicRepro = 4,
}

impl EmpiricalLevel {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::E0Null => "Null (Unverifiable)",
            Self::E1Testimonial => "Testimonial",
            Self::E2PrivateVerify => "Privately Verifiable",
            Self::E3Cryptographic => "Cryptographically Proven",
            Self::E4PublicRepro => "Publicly Reproducible",
        }
    }

    /// Get short code
    pub fn code(&self) -> &'static str {
        match self {
            Self::E0Null => "E0",
            Self::E1Testimonial => "E1",
            Self::E2PrivateVerify => "E2",
            Self::E3Cryptographic => "E3",
            Self::E4PublicRepro => "E4",
        }
    }

    /// Check if this level meets or exceeds a minimum requirement
    pub fn meets_minimum(&self, minimum: Self) -> bool {
        *self >= minimum
    }

    /// Get numeric index (0-4) for array indexing
    pub fn as_index(&self) -> usize {
        match self {
            Self::E0Null => 0,
            Self::E1Testimonial => 1,
            Self::E2PrivateVerify => 2,
            Self::E3Cryptographic => 3,
            Self::E4PublicRepro => 4,
        }
    }

    /// Create from numeric index
    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Self::E0Null),
            1 => Some(Self::E1Testimonial),
            2 => Some(Self::E2PrivateVerify),
            3 => Some(Self::E3Cryptographic),
            4 => Some(Self::E4PublicRepro),
            _ => None,
        }
    }
}

/// Normative Axis (N): Who agrees with the claim
///
/// From narrowest to broadest agreement:
/// - N0: Personal (self only)
/// - N1: Communal (local DAO)
/// - N2: Network (global consensus)
/// - N3: Axiomatic (constitutional/mathematical)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/epistemic/"))]
#[repr(u8)]
pub enum NormativeLevel {
    /// N0: Personal - Only the claimant agrees
    N0Personal = 0,

    /// N1: Communal - Local DAO or community agreement
    N1Communal = 1,

    /// N2: Network - Global network consensus
    N2Network = 2,

    /// N3: Axiomatic - Constitutional law, mathematical truth
    N3Axiomatic = 3,
}

impl NormativeLevel {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::N0Personal => "Personal",
            Self::N1Communal => "Communal",
            Self::N2Network => "Network",
            Self::N3Axiomatic => "Axiomatic",
        }
    }

    /// Get short code
    pub fn code(&self) -> &'static str {
        match self {
            Self::N0Personal => "N0",
            Self::N1Communal => "N1",
            Self::N2Network => "N2",
            Self::N3Axiomatic => "N3",
        }
    }

    /// Check if this level meets or exceeds a minimum requirement
    pub fn meets_minimum(&self, minimum: Self) -> bool {
        *self >= minimum
    }
}

/// Materiality Axis (M): How long the claim matters
///
/// From shortest to longest lifespan:
/// - M0: Ephemeral (discard immediately)
/// - M1: Temporal (prune after state change)
/// - M2: Persistent (archive after time)
/// - M3: Foundational (preserve forever)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/epistemic/"))]
#[repr(u8)]
pub enum MaterialityLevel {
    /// M0: Ephemeral - Discard immediately after use
    M0Ephemeral = 0,

    /// M1: Temporal - Prune after state change
    M1Temporal = 1,

    /// M2: Persistent - Archive after retention period
    M2Persistent = 2,

    /// M3: Foundational - Preserve forever
    M3Foundational = 3,
}

impl MaterialityLevel {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::M0Ephemeral => "Ephemeral",
            Self::M1Temporal => "Temporal",
            Self::M2Persistent => "Persistent",
            Self::M3Foundational => "Foundational",
        }
    }

    /// Get short code
    pub fn code(&self) -> &'static str {
        match self {
            Self::M0Ephemeral => "M0",
            Self::M1Temporal => "M1",
            Self::M2Persistent => "M2",
            Self::M3Foundational => "M3",
        }
    }

    /// Suggested retention in days (0 = forever)
    pub fn suggested_retention_days(&self) -> Option<u64> {
        match self {
            Self::M0Ephemeral => Some(0),
            Self::M1Temporal => Some(7),
            Self::M2Persistent => Some(365),
            Self::M3Foundational => None, // Forever
        }
    }

    /// Check if this level meets or exceeds a minimum requirement
    pub fn meets_minimum(&self, minimum: Self) -> bool {
        *self >= minimum
    }
}

/// Harmonic Axis (H): GIS v4.0 integration - Resonance with Kosmic Song
///
/// The fourth dimension connects epistemic claims to the broader consciousness
/// metrics and harmonic patterns of the Symthaea framework.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/epistemic/"))]
#[repr(u8)]
pub enum HarmonicLevel {
    /// H0: No Harmonic - Claim has no relevance to consciousness metrics
    H0None = 0,

    /// H1: Local Resonance - Affects individual or small group coherence
    H1Local = 1,

    /// H2: Network Resonance - Influences network-wide patterns
    H2Network = 2,

    /// H3: Civilizational - Shapes civilizational intention alignment
    H3Civilizational = 3,

    /// H4: Kosmic - Integrates with universal harmonic patterns
    H4Kosmic = 4,
}

impl HarmonicLevel {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::H0None => "No Harmonic",
            Self::H1Local => "Local Resonance",
            Self::H2Network => "Network Resonance",
            Self::H3Civilizational => "Civilizational",
            Self::H4Kosmic => "Kosmic",
        }
    }

    /// Get short code
    pub fn code(&self) -> &'static str {
        match self {
            Self::H0None => "H0",
            Self::H1Local => "H1",
            Self::H2Network => "H2",
            Self::H3Civilizational => "H3",
            Self::H4Kosmic => "H4",
        }
    }

    /// Check if this level meets or exceeds a minimum requirement
    pub fn meets_minimum(&self, minimum: Self) -> bool {
        *self >= minimum
    }
}

/// Combined epistemic classification (3D: E/N/M)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/epistemic/"))]
pub struct EpistemicClassification {
    /// E-Axis: How to verify the claim (E0-E4)
    pub empirical: EmpiricalLevel,
    /// N-Axis: Who agrees with the claim (N0-N3)
    pub normative: NormativeLevel,
    /// M-Axis: How long the claim matters (M0-M3)
    pub materiality: MaterialityLevel,
}

/// Extended epistemic classification with Harmonic dimension (4D: E/N/M/H)
///
/// GIS v4.0 integration adds the Harmonic axis for consciousness metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/epistemic/"))]
pub struct EpistemicClassificationExtended {
    /// E-Axis: How to verify the claim (E0-E4)
    pub empirical: EmpiricalLevel,
    /// N-Axis: Who agrees with the claim (N0-N3)
    pub normative: NormativeLevel,
    /// M-Axis: How long the claim matters (M0-M3)
    pub materiality: MaterialityLevel,
    /// H-Axis: Resonance with consciousness metrics (H0-H4)
    pub harmonic: HarmonicLevel,
}

impl EpistemicClassificationExtended {
    /// Create a new extended classification
    pub fn new(
        empirical: EmpiricalLevel,
        normative: NormativeLevel,
        materiality: MaterialityLevel,
        harmonic: HarmonicLevel,
    ) -> Self {
        Self {
            empirical,
            normative,
            materiality,
            harmonic,
        }
    }

    /// Create from a basic classification with default harmonic level
    pub fn from_basic(basic: EpistemicClassification, harmonic: HarmonicLevel) -> Self {
        Self {
            empirical: basic.empirical,
            normative: basic.normative,
            materiality: basic.materiality,
            harmonic,
        }
    }

    /// Get the 3D classification (without harmonic)
    pub fn to_basic(&self) -> EpistemicClassification {
        EpistemicClassification {
            empirical: self.empirical,
            normative: self.normative,
            materiality: self.materiality,
        }
    }

    /// Get a string representation like "E3-N2-M2-H1"
    pub fn to_code(&self) -> String {
        format!(
            "{}-{}-{}-{}",
            self.empirical.code(),
            self.normative.code(),
            self.materiality.code(),
            self.harmonic.code()
        )
    }

    /// Check if this classification meets minimum requirements
    pub fn meets_requirements(
        &self,
        min_e: EmpiricalLevel,
        min_n: NormativeLevel,
        min_m: MaterialityLevel,
        min_h: HarmonicLevel,
    ) -> bool {
        self.empirical.meets_minimum(min_e)
            && self.normative.meets_minimum(min_n)
            && self.materiality.meets_minimum(min_m)
            && self.harmonic.meets_minimum(min_h)
    }
}

impl std::fmt::Display for EpistemicClassificationExtended {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_code())
    }
}

impl EpistemicClassification {
    /// Create a new classification
    pub fn new(
        empirical: EmpiricalLevel,
        normative: NormativeLevel,
        materiality: MaterialityLevel,
    ) -> Self {
        Self {
            empirical,
            normative,
            materiality,
        }
    }

    /// Get a string representation like "E3-N2-M2"
    pub fn to_code(&self) -> String {
        format!(
            "{}-{}-{}",
            self.empirical.code(),
            self.normative.code(),
            self.materiality.code()
        )
    }

    /// Check if this classification meets minimum requirements
    pub fn meets_requirements(
        &self,
        min_e: EmpiricalLevel,
        min_n: NormativeLevel,
        min_m: MaterialityLevel,
    ) -> bool {
        self.empirical.meets_minimum(min_e)
            && self.normative.meets_minimum(min_n)
            && self.materiality.meets_minimum(min_m)
    }
}

impl std::fmt::Display for EpistemicClassification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_code())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empirical_ordering() {
        assert!(EmpiricalLevel::E4PublicRepro > EmpiricalLevel::E0Null);
        assert!(EmpiricalLevel::E3Cryptographic > EmpiricalLevel::E2PrivateVerify);
    }

    #[test]
    fn test_meets_minimum() {
        assert!(EmpiricalLevel::E3Cryptographic.meets_minimum(EmpiricalLevel::E2PrivateVerify));
        assert!(!EmpiricalLevel::E1Testimonial.meets_minimum(EmpiricalLevel::E3Cryptographic));
    }

    #[test]
    fn test_classification_code() {
        let class = EpistemicClassification::new(
            EmpiricalLevel::E3Cryptographic,
            NormativeLevel::N2Network,
            MaterialityLevel::M2Persistent,
        );
        assert_eq!(class.to_code(), "E3-N2-M2");
    }

    #[test]
    fn test_meets_requirements() {
        let class = EpistemicClassification::new(
            EmpiricalLevel::E3Cryptographic,
            NormativeLevel::N2Network,
            MaterialityLevel::M2Persistent,
        );

        assert!(class.meets_requirements(
            EmpiricalLevel::E2PrivateVerify,
            NormativeLevel::N1Communal,
            MaterialityLevel::M1Temporal,
        ));

        assert!(!class.meets_requirements(
            EmpiricalLevel::E4PublicRepro,
            NormativeLevel::N1Communal,
            MaterialityLevel::M1Temporal,
        ));
    }
}
