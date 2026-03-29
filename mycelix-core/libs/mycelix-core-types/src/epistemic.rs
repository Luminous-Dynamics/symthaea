// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemic Classification System (E/N/M/H)
//!
//! The E/N/M/H classification provides a multi-dimensional framework for
//! categorizing knowledge and claims within the Mycelix ecosystem.
//!
//! - **E** (Empirical): How verifiable is the claim? (E0-E4)
//! - **N** (Normative): What scope of values/norms apply? (N0-N3)
//! - **M** (Materiality): What is the practical impact? (M0-M3)
//! - **H** (Harmonic): Which harmonies are affected? (GIS v4.0)

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::harmonic::HarmonicImpact;

// ==============================================================================
// EPISTEMIC CONTEXT - The Prismatic Architecture
// ==============================================================================

/// Context-Adaptive Epistemic Weights (Prismatic Architecture)
///
/// Different epistemic contexts require different weight balances.
/// "Fixed Weights = Colonialism; Adaptive Weights = Pluralism"
///
/// See: EPISTEMICS_FAIRNESS_ANALYSIS.md for rationale
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EpistemicContext {
    /// Scientific context: Empirical verification matters most
    /// Weights: E=0.50, N=0.30, M=0.20
    Scientific,

    /// Governance context: Normative consensus matters most
    /// Weights: E=0.30, N=0.45, M=0.25
    Governance,

    /// Personal/Contemplative context: Balanced weights
    /// Weights: E=0.33, N=0.33, M=0.34
    Personal,

    /// Indigenous/Cultural context: Community validation + persistence
    /// Weights: E=0.25, N=0.40, M=0.35
    Indigenous,

    /// Contemplative/Experiential context: Deep integration matters
    /// Weights: E=0.20, N=0.30, M=0.50
    Contemplative,

    /// Emergency context: Speed and verification paramount
    /// Weights: E=0.60, N=0.25, M=0.15
    Emergency,

    /// Standard context: Original fixed weights (for backwards compatibility)
    /// Weights: E=0.40, N=0.35, M=0.25
    Standard,
}

impl EpistemicContext {
    /// Get the E/N/M weights for this context
    pub fn weights(&self) -> (f32, f32, f32) {
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

    /// Human-readable description of this context
    pub fn description(&self) -> &'static str {
        match self {
            Self::Scientific => "Scientific inquiry prioritizing reproducible verification",
            Self::Governance => "Governance decisions prioritizing broad consensus",
            Self::Personal => "Personal knowledge with balanced considerations",
            Self::Indigenous => "Indigenous/cultural knowledge honoring community and tradition",
            Self::Contemplative => "Contemplative/experiential knowledge valuing depth",
            Self::Emergency => "Emergency situations requiring rapid verification",
            Self::Standard => "Standard context with original fixed weights",
        }
    }
}

impl Default for EpistemicContext {
    fn default() -> Self {
        Self::Standard
    }
}

// ==============================================================================
// TESTIMONIAL QUALITY - E1 Sub-levels (Testimonial Rehabilitation)
// ==============================================================================

/// Testimonial quality sub-levels for E1 tier
///
/// Addresses the "Testimonial Collapse" problem where diverse forms of
/// witness-based knowledge are collapsed into a single low-scoring tier.
///
/// This rehabilitates oral traditions, elder wisdom, and collective witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TestimonialQuality {
    /// E1.0: Single anonymous witness
    /// Score modifier: 0.20
    SingleAnonymous,

    /// E1.1: Single identified witness with known reputation
    /// Score modifier: 0.25
    SingleIdentified,

    /// E1.2: Multiple independent oral accounts (corroborated)
    /// Score modifier: 0.35
    CorroboratedOral,

    /// E1.3: Elder council or traditional authority attestation
    /// Score modifier: 0.45
    ElderCouncil,

    /// E1.4: Collective community witness (ceremony, ritual, shared experience)
    /// Score modifier: 0.50
    CollectiveWitness,

    /// E1.5: Corroborated Witness - multiple independent testimonies that agree
    /// This bridges E1 (Testimonial) and E2 (Private/Observable)
    /// Score modifier: 0.55 (approaches E2's 0.50 base)
    CorroboratedWitness,
}

impl TestimonialQuality {
    /// Get the E-axis score for this testimonial quality (0.0-1.0 scale)
    pub fn score(&self) -> f32 {
        match self {
            Self::SingleAnonymous => 0.20,
            Self::SingleIdentified => 0.25,
            Self::CorroboratedOral => 0.35,
            Self::ElderCouncil => 0.45,
            Self::CollectiveWitness => 0.50,
            Self::CorroboratedWitness => 0.55,
        }
    }

    /// Get the numeric sub-level (1.0, 1.1, 1.2, etc.)
    pub fn sub_level(&self) -> f32 {
        match self {
            Self::SingleAnonymous => 1.0,
            Self::SingleIdentified => 1.1,
            Self::CorroboratedOral => 1.2,
            Self::ElderCouncil => 1.3,
            Self::CollectiveWitness => 1.4,
            Self::CorroboratedWitness => 1.5,
        }
    }

    /// Short code for display (e.g., "E1.5")
    pub fn code(&self) -> &'static str {
        match self {
            Self::SingleAnonymous => "E1.0",
            Self::SingleIdentified => "E1.1",
            Self::CorroboratedOral => "E1.2",
            Self::ElderCouncil => "E1.3",
            Self::CollectiveWitness => "E1.4",
            Self::CorroboratedWitness => "E1.5",
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::SingleAnonymous => "Single anonymous witness testimony",
            Self::SingleIdentified => "Single identified witness with reputation",
            Self::CorroboratedOral => "Multiple independent oral accounts",
            Self::ElderCouncil => "Elder council or traditional authority",
            Self::CollectiveWitness => "Collective community witness (ceremony/ritual)",
            Self::CorroboratedWitness => "Corroborated witness (multiple agreeing testimonies)",
        }
    }
}

impl Default for TestimonialQuality {
    fn default() -> Self {
        Self::SingleAnonymous
    }
}

// ==============================================================================
// VERIFICATION STATUS - Including Contested Reproducibility
// ==============================================================================

/// Verification status including contested/forked states
///
/// Handles edge cases like "some labs can reproduce, some cannot"
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VerificationStatus {
    /// Standard verification - claim is verified to stated E-level
    Verified,

    /// Unverified - no verification attempts yet
    Unverified,

    /// Contested - some verify, some do not (forked epistemics)
    /// Contains ratio of successful verifications (0.0-1.0)
    Contested,

    /// Refuted - claim has been actively disproven
    Refuted,

    /// Superseded - claim replaced by better understanding
    Superseded,
}

impl VerificationStatus {
    /// Whether this status indicates the claim should be trusted
    pub fn is_trustworthy(&self) -> bool {
        matches!(self, Self::Verified)
    }

    /// Whether this status requires special handling
    pub fn requires_nuance(&self) -> bool {
        matches!(self, Self::Contested | Self::Superseded)
    }
}

impl Default for VerificationStatus {
    fn default() -> Self {
        Self::Unverified
    }
}

/// Detailed contested verification state
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ContestedVerification {
    /// Ratio of successful verification attempts (0.0-1.0)
    pub success_ratio: f32,

    /// Number of verification attempts
    pub attempt_count: u32,

    /// Notes on why verification is contested
    pub notes: Option<String>,
}

impl ContestedVerification {
    /// Create new contested verification record
    pub fn new(successes: u32, failures: u32) -> Self {
        let total = successes + failures;
        let ratio = if total > 0 {
            successes as f32 / total as f32
        } else {
            0.0
        };
        Self {
            success_ratio: ratio,
            attempt_count: total,
            notes: None,
        }
    }

    /// Whether this leans toward verified (>50% success)
    pub fn leans_verified(&self) -> bool {
        self.success_ratio > 0.5
    }

    /// Epistemic adjustment factor for contested claims
    /// Returns value between 0.5 (heavily contested) and 1.0 (barely contested)
    pub fn adjustment_factor(&self) -> f32 {
        // The more even the split, the lower the factor
        let deviation_from_50 = (self.success_ratio - 0.5).abs();
        0.5 + deviation_from_50
    }
}

// ==============================================================================
// EMPIRICAL LEVEL (Extended with Testimonial Quality support)
// ==============================================================================

/// Empirical verification levels (E0-E4) with optional testimonial quality
///
/// Measures the degree to which a claim can be empirically verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u8)]
pub enum EmpiricalLevel {
    /// E0: Unverifiable - cannot be empirically tested
    Unverifiable = 0,
    /// E1: Anecdotal - based on individual testimony
    Anecdotal = 1,
    /// E2: Observable - can be directly observed but not measured
    Observable = 2,
    /// E3: Measurable - can be quantitatively measured
    Measurable = 3,
    /// E4: Cryptographically Verifiable - provable via cryptographic means
    CryptographicallyVerifiable = 4,
}

impl EmpiricalLevel {
    /// Get the numeric value of this level
    pub fn value(&self) -> u8 {
        *self as u8
    }

    /// Create from numeric value
    pub fn from_value(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Unverifiable),
            1 => Some(Self::Anecdotal),
            2 => Some(Self::Observable),
            3 => Some(Self::Measurable),
            4 => Some(Self::CryptographicallyVerifiable),
            _ => None,
        }
    }

    /// Short code for display (e.g., "E3")
    pub fn code(&self) -> &'static str {
        match self {
            Self::Unverifiable => "E0",
            Self::Anecdotal => "E1",
            Self::Observable => "E2",
            Self::Measurable => "E3",
            Self::CryptographicallyVerifiable => "E4",
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Unverifiable => "Cannot be empirically verified",
            Self::Anecdotal => "Based on individual testimony",
            Self::Observable => "Can be directly observed",
            Self::Measurable => "Can be quantitatively measured",
            Self::CryptographicallyVerifiable => "Provable via cryptographic proof",
        }
    }
}

impl Default for EmpiricalLevel {
    fn default() -> Self {
        Self::Unverifiable
    }
}

/// Normative scope levels (N0-N3)
///
/// Measures the scope of values, norms, or stakeholders affected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u8)]
pub enum NormativeLevel {
    /// N0: Individual - affects only the individual agent
    Individual = 0,
    /// N1: Group - affects a defined group or community
    Group = 1,
    /// N2: Network - affects the entire network
    Network = 2,
    /// N3: Sentient - affects sentient beings (hard boundary for harm)
    Sentient = 3,
}

impl NormativeLevel {
    /// Get the numeric value of this level
    pub fn value(&self) -> u8 {
        *self as u8
    }

    /// Create from numeric value
    pub fn from_value(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Individual),
            1 => Some(Self::Group),
            2 => Some(Self::Network),
            3 => Some(Self::Sentient),
            _ => None,
        }
    }

    /// Short code for display (e.g., "N2")
    pub fn code(&self) -> &'static str {
        match self {
            Self::Individual => "N0",
            Self::Group => "N1",
            Self::Network => "N2",
            Self::Sentient => "N3",
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Individual => "Affects only the individual",
            Self::Group => "Affects a group or community",
            Self::Network => "Affects the entire network",
            Self::Sentient => "Affects sentient beings (harm boundary)",
        }
    }

    /// Check if this level crosses the harm boundary (N3)
    pub fn crosses_harm_boundary(&self) -> bool {
        matches!(self, Self::Sentient)
    }
}

impl Default for NormativeLevel {
    fn default() -> Self {
        Self::Individual
    }
}

/// Materiality levels (M0-M3)
///
/// Measures the practical impact and persistence of a claim or action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u8)]
pub enum MaterialityLevel {
    /// M0: Ephemeral - no lasting impact
    Ephemeral = 0,
    /// M1: Temporary - short-term impact
    Temporary = 1,
    /// M2: Persistent - medium-term storage/impact
    Persistent = 2,
    /// M3: Permanent - long-term or irreversible impact
    Permanent = 3,
}

impl MaterialityLevel {
    /// Get the numeric value of this level
    pub fn value(&self) -> u8 {
        *self as u8
    }

    /// Create from numeric value
    pub fn from_value(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Ephemeral),
            1 => Some(Self::Temporary),
            2 => Some(Self::Persistent),
            3 => Some(Self::Permanent),
            _ => None,
        }
    }

    /// Short code for display (e.g., "M2")
    pub fn code(&self) -> &'static str {
        match self {
            Self::Ephemeral => "M0",
            Self::Temporary => "M1",
            Self::Persistent => "M2",
            Self::Permanent => "M3",
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Ephemeral => "No lasting impact",
            Self::Temporary => "Short-term impact",
            Self::Persistent => "Medium-term storage/impact",
            Self::Permanent => "Long-term or irreversible impact",
        }
    }

    /// Check if this level is irreversible
    pub fn is_irreversible(&self) -> bool {
        matches!(self, Self::Permanent)
    }
}

impl Default for MaterialityLevel {
    fn default() -> Self {
        Self::Ephemeral
    }
}

/// Complete epistemic classification (E/N/M/H) with context-adaptive scoring
///
/// Combines all four dimensions of epistemic classification.
/// Supports the "Prismatic Architecture" for context-adaptive weights.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EpistemicClassification {
    /// Empirical verification level (E0-E4)
    pub empirical: EmpiricalLevel,
    /// Normative scope level (N0-N3)
    pub normative: NormativeLevel,
    /// Materiality level (M0-M3)
    pub materiality: MaterialityLevel,
    /// Harmonic impact (optional, for GIS v4.0)
    pub harmonic: Option<HarmonicImpact>,
    /// Testimonial quality (optional, for E1 sub-levels)
    pub testimonial_quality: Option<TestimonialQuality>,
    /// Verification status (optional, for contested claims)
    pub verification_status: VerificationStatus,
    /// Contested verification details (optional)
    pub contested_details: Option<ContestedVerification>,
}

impl EpistemicClassification {
    /// Create a new classification with just E/N/M
    pub fn new(
        empirical: EmpiricalLevel,
        normative: NormativeLevel,
        materiality: MaterialityLevel,
    ) -> Self {
        Self {
            empirical,
            normative,
            materiality,
            harmonic: None,
            testimonial_quality: None,
            verification_status: VerificationStatus::default(),
            contested_details: None,
        }
    }

    /// Create a new classification with full E/N/M/H
    pub fn with_harmonic(
        empirical: EmpiricalLevel,
        normative: NormativeLevel,
        materiality: MaterialityLevel,
        harmonic: HarmonicImpact,
    ) -> Self {
        Self {
            empirical,
            normative,
            materiality,
            harmonic: Some(harmonic),
            testimonial_quality: None,
            verification_status: VerificationStatus::default(),
            contested_details: None,
        }
    }

    /// Create a testimonial classification with quality sub-level
    pub fn testimonial(
        quality: TestimonialQuality,
        normative: NormativeLevel,
        materiality: MaterialityLevel,
    ) -> Self {
        Self {
            empirical: EmpiricalLevel::Anecdotal,
            normative,
            materiality,
            harmonic: None,
            testimonial_quality: Some(quality),
            verification_status: VerificationStatus::default(),
            contested_details: None,
        }
    }

    /// Mark this classification as having contested verification
    pub fn with_contested(mut self, successes: u32, failures: u32) -> Self {
        self.verification_status = VerificationStatus::Contested;
        self.contested_details = Some(ContestedVerification::new(successes, failures));
        self
    }

    /// Format as short code (e.g., "E3/N2/M1" or "E1.5/N1/M3")
    pub fn code(&self) -> String {
        let e_code = if let Some(ref tq) = self.testimonial_quality {
            tq.code().to_string()
        } else {
            self.empirical.code().to_string()
        };

        let base = format!(
            "{}/{}/{}",
            e_code,
            self.normative.code(),
            self.materiality.code()
        );

        if let Some(ref h) = self.harmonic {
            format!("{}/H{{{}}}", base, h.short_format())
        } else {
            base
        }
    }

    /// Calculate quality score with STANDARD (fixed) weights
    /// This is the original scoring for backwards compatibility.
    ///
    /// WARNING: Fixed weights may encode epistemic bias.
    /// Consider using `quality_score_contextual()` instead.
    pub fn quality_score(&self) -> f32 {
        self.quality_score_contextual(EpistemicContext::Standard)
    }

    /// Calculate quality score with CONTEXT-ADAPTIVE weights (Prismatic Architecture)
    ///
    /// This is the recommended method for fair epistemic scoring.
    /// Different contexts have different weight balances.
    pub fn quality_score_contextual(&self, context: EpistemicContext) -> f32 {
        let (e_weight, n_weight, m_weight) = context.weights();

        // Calculate E score - use testimonial quality if available
        let e_score = if let Some(ref tq) = self.testimonial_quality {
            // Testimonial sub-level score (0.20-0.55)
            tq.score()
        } else {
            // Standard E-level score (0.0-1.0)
            self.empirical.value() as f32 / 4.0
        };

        let n_score = self.normative.value() as f32 / 3.0;
        let m_score = self.materiality.value() as f32 / 3.0;

        let mut score = e_score * e_weight + n_score * n_weight + m_score * m_weight;

        // Apply contested verification adjustment
        if let Some(ref contested) = self.contested_details {
            score *= contested.adjustment_factor();
        }

        score
    }

    /// Calculate a risk score based on classification
    ///
    /// Higher scores indicate higher risk/importance.
    /// Range: 0.0 to 1.0
    pub fn risk_score(&self) -> f32 {
        // Weight factors
        const E_WEIGHT: f32 = 0.2; // Lower E = higher risk (less verifiable)
        const N_WEIGHT: f32 = 0.4; // Higher N = higher risk (more stakeholders)
        const M_WEIGHT: f32 = 0.4; // Higher M = higher risk (more permanent)

        // Invert empirical (E0=high risk, E4=low risk)
        let e_risk = 1.0 - (self.empirical.value() as f32 / 4.0);
        let n_risk = self.normative.value() as f32 / 3.0;
        let m_risk = self.materiality.value() as f32 / 3.0;

        E_WEIGHT * e_risk + N_WEIGHT * n_risk + M_WEIGHT * m_risk
    }

    /// Check if this classification requires elevated scrutiny
    ///
    /// Returns true if:
    /// - Normative level is N3 (sentient harm boundary)
    /// - Materiality is M3 (permanent/irreversible)
    /// - Empirical is E0/E1 with high normative/materiality
    /// - Verification is contested
    pub fn requires_scrutiny(&self) -> bool {
        self.normative.crosses_harm_boundary()
            || self.materiality.is_irreversible()
            || (self.empirical.value() <= 1 && self.normative.value() >= 2)
            || self.verification_status.requires_nuance()
    }

    /// Check if this is a high-stakes classification
    pub fn is_high_stakes(&self) -> bool {
        self.risk_score() > 0.6
    }

    /// Check if this claim has contested reproducibility
    pub fn is_contested(&self) -> bool {
        matches!(self.verification_status, VerificationStatus::Contested)
    }
}

impl Default for EpistemicClassification {
    fn default() -> Self {
        Self {
            empirical: EmpiricalLevel::default(),
            normative: NormativeLevel::default(),
            materiality: MaterialityLevel::default(),
            harmonic: None,
            testimonial_quality: None,
            verification_status: VerificationStatus::default(),
            contested_details: None,
        }
    }
}

/// Parse an E/N/M code string into classification
///
/// Accepts formats like "E3/N2/M1" or "E3N2M1"
pub fn parse_enm_code(code: &str) -> Option<EpistemicClassification> {
    let normalized = code.to_uppercase().replace('/', "");

    let mut e: Option<EmpiricalLevel> = None;
    let mut n: Option<NormativeLevel> = None;
    let mut m: Option<MaterialityLevel> = None;

    let mut chars = normalized.chars().peekable();
    while let Some(c) = chars.next() {
        let digit = chars.next()?.to_digit(10)? as u8;

        match c {
            'E' => e = EmpiricalLevel::from_value(digit),
            'N' => n = NormativeLevel::from_value(digit),
            'M' => m = MaterialityLevel::from_value(digit),
            _ => return None,
        }
    }

    Some(EpistemicClassification::new(e?, n?, m?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_formatting() {
        let c = EpistemicClassification::new(
            EmpiricalLevel::Measurable,
            NormativeLevel::Network,
            MaterialityLevel::Persistent,
        );
        assert_eq!(c.code(), "E3/N2/M2");
    }

    #[test]
    fn test_risk_score() {
        let low_risk = EpistemicClassification::new(
            EmpiricalLevel::CryptographicallyVerifiable,
            NormativeLevel::Individual,
            MaterialityLevel::Ephemeral,
        );
        assert!(low_risk.risk_score() < 0.3);

        let high_risk = EpistemicClassification::new(
            EmpiricalLevel::Unverifiable,
            NormativeLevel::Sentient,
            MaterialityLevel::Permanent,
        );
        assert!(high_risk.risk_score() > 0.7);
    }

    #[test]
    fn test_parse_code() {
        let c = parse_enm_code("E3/N2/M1").unwrap();
        assert_eq!(c.empirical, EmpiricalLevel::Measurable);
        assert_eq!(c.normative, NormativeLevel::Network);
        assert_eq!(c.materiality, MaterialityLevel::Temporary);

        let c2 = parse_enm_code("E3N2M1").unwrap();
        assert_eq!(c2.code(), "E3/N2/M1");
    }

    #[test]
    fn test_scrutiny_requirements() {
        let needs_scrutiny = EpistemicClassification::new(
            EmpiricalLevel::Measurable,
            NormativeLevel::Sentient,
            MaterialityLevel::Persistent,
        );
        assert!(needs_scrutiny.requires_scrutiny());

        let normal = EpistemicClassification::new(
            EmpiricalLevel::Measurable,
            NormativeLevel::Individual,
            MaterialityLevel::Temporary,
        );
        assert!(!normal.requires_scrutiny());
    }

    // ==============================================================================
    // PRISMATIC ARCHITECTURE TESTS
    // ==============================================================================

    #[test]
    fn test_context_adaptive_weights() {
        // Scientific context should weight E highest
        let (e, n, m) = EpistemicContext::Scientific.weights();
        assert!(e > n && e > m, "Scientific should prioritize empirical");
        assert!((e + n + m - 1.0).abs() < 0.01, "Weights should sum to 1.0");

        // Indigenous context should weight N highest
        let (e, n, m) = EpistemicContext::Indigenous.weights();
        assert!(
            n > e,
            "Indigenous should prioritize normative over empirical"
        );
        assert!((e + n + m - 1.0).abs() < 0.01, "Weights should sum to 1.0");

        // Contemplative context should weight M highest
        let (e, n, m) = EpistemicContext::Contemplative.weights();
        assert!(
            m > e && m > n,
            "Contemplative should prioritize materiality"
        );
        assert!((e + n + m - 1.0).abs() < 0.01, "Weights should sum to 1.0");
    }

    #[test]
    fn test_testimonial_quality_scores() {
        // E1.0 < E1.1 < E1.2 < E1.3 < E1.4 < E1.5
        assert!(
            TestimonialQuality::SingleAnonymous.score()
                < TestimonialQuality::SingleIdentified.score()
        );
        assert!(
            TestimonialQuality::SingleIdentified.score()
                < TestimonialQuality::CorroboratedOral.score()
        );
        assert!(
            TestimonialQuality::CorroboratedOral.score() < TestimonialQuality::ElderCouncil.score()
        );
        assert!(
            TestimonialQuality::ElderCouncil.score()
                < TestimonialQuality::CollectiveWitness.score()
        );
        assert!(
            TestimonialQuality::CollectiveWitness.score()
                < TestimonialQuality::CorroboratedWitness.score()
        );
    }

    #[test]
    fn test_testimonial_classification_code() {
        let grandmother = EpistemicClassification::testimonial(
            TestimonialQuality::CorroboratedWitness,
            NormativeLevel::Group,
            MaterialityLevel::Permanent,
        );
        assert_eq!(grandmother.code(), "E1.5/N1/M3");
    }

    #[test]
    fn test_context_affects_quality_score() {
        // Grandmother's remedy: E1.5/N1/M3
        let grandmother = EpistemicClassification::testimonial(
            TestimonialQuality::CorroboratedWitness,
            NormativeLevel::Group,
            MaterialityLevel::Permanent,
        );

        // In standard context, empirical is weighted heavily (40%)
        let standard_score = grandmother.quality_score_contextual(EpistemicContext::Standard);

        // In indigenous context, normative (community) is weighted more (40%)
        let indigenous_score = grandmother.quality_score_contextual(EpistemicContext::Indigenous);

        // Indigenous context should give a higher score for community-validated knowledge
        assert!(
            indigenous_score > standard_score,
            "Indigenous context should honor grandmother's knowledge more: {} > {}",
            indigenous_score,
            standard_score
        );
    }

    #[test]
    fn test_galileo_stress_case() {
        // Galileo: E4 (reproducible observation) but N0 (only he believes it)
        let galileo = EpistemicClassification::new(
            EmpiricalLevel::CryptographicallyVerifiable, // E4 - publicly verifiable
            NormativeLevel::Individual,                  // N0 - personal (nobody agrees)
            MaterialityLevel::Permanent,                 // M3 - fundamental truth
        );

        // In scientific context, empirical matters most - Galileo should score well
        let science_score = galileo.quality_score_contextual(EpistemicContext::Scientific);

        // In standard context, the N0 hurts him
        let standard_score = galileo.quality_score_contextual(EpistemicContext::Standard);

        assert!(
            science_score > standard_score,
            "Scientific context should vindicate Galileo: {} > {}",
            science_score,
            standard_score
        );
    }

    #[test]
    fn test_contested_verification() {
        // A claim where 6 labs replicate, 4 labs fail to replicate
        let contested = EpistemicClassification::new(
            EmpiricalLevel::Measurable,
            NormativeLevel::Network,
            MaterialityLevel::Persistent,
        )
        .with_contested(6, 4);

        assert!(contested.is_contested());
        assert!(contested.requires_scrutiny());

        // Score should be reduced due to contested nature
        let uncontested = EpistemicClassification::new(
            EmpiricalLevel::Measurable,
            NormativeLevel::Network,
            MaterialityLevel::Persistent,
        );

        let contested_score = contested.quality_score();
        let uncontested_score = uncontested.quality_score();

        assert!(
            contested_score < uncontested_score,
            "Contested claims should score lower: {} < {}",
            contested_score,
            uncontested_score
        );
    }

    #[test]
    fn test_contested_adjustment_factor() {
        // 50/50 split should give lowest adjustment (0.5)
        let even_split = ContestedVerification::new(5, 5);
        assert!((even_split.adjustment_factor() - 0.5).abs() < 0.01);

        // 90/10 split should give higher adjustment (~0.9)
        let clear_majority = ContestedVerification::new(9, 1);
        assert!(clear_majority.adjustment_factor() > 0.8);
    }

    #[test]
    fn test_flat_earth_veto() {
        // Flat Earth: E0 (unverifiable belief), N1 (some community agrees)
        let flat_earth = EpistemicClassification::new(
            EmpiricalLevel::Unverifiable,
            NormativeLevel::Group,
            MaterialityLevel::Persistent,
        );

        // Even in governance context (N-heavy), E0 should score poorly
        let governance_score = flat_earth.quality_score_contextual(EpistemicContext::Governance);

        // E0 contributes 0.0 to the E component, so score is limited
        assert!(
            governance_score < 0.5,
            "E0 claims should score low even in N-heavy contexts: {}",
            governance_score
        );
    }
}
