//! Epistemic Charter stub

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EmpiricalLevel {
    E0Null,
    E1Testimonial,
    E2PrivateVerify,
    E3Cryptographic,
    E4PublicRepro,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NormativeLevel {
    N0Personal,
    N1Communal,
    N2Network,
    N3Axiomatic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MaterialityLevel {
    M0Ephemeral,
    M1Temporal,
    M2Persistent,
    M3Foundational,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EpistemicClassification {
    pub empirical: EmpiricalLevel,
    pub normative: NormativeLevel,
    pub materiality: MaterialityLevel,
}

impl EpistemicClassification {
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicClaim {
    pub content: String,
    pub empirical: EmpiricalLevel,
    pub normative: NormativeLevel,
    pub materiality: MaterialityLevel,
}

impl EpistemicClaim {
    pub fn new(
        content: impl Into<String>,
        empirical: EmpiricalLevel,
        normative: NormativeLevel,
        materiality: MaterialityLevel,
    ) -> Self {
        Self {
            content: content.into(),
            empirical,
            normative,
            materiality,
        }
    }

    pub fn meets_standard(
        &self,
        min_empirical: EmpiricalLevel,
        min_normative: NormativeLevel,
    ) -> bool {
        self.empirical >= min_empirical && self.normative >= min_normative
    }
}
