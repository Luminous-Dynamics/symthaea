// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Symthaea Epistemic Types
//!
//! Canonical epistemic type definitions shared between Symthaea (consciousness engine),
//! Mycelix (decentralized knowledge graph), and the bridge crate.
//!
//! ## The Epistemic Cube (E/N/M)
//!
//! Three orthogonal axes classify knowledge quality:
//! - **E-Axis (Empirical)**: How is this verified? (E0-E4)
//! - **N-Axis (Normative)**: Who agrees? (N0-N3)
//! - **M-Axis (Materiality)**: How permanent? (M0-M3)
//!
//! ## Prismatic Architecture
//!
//! Seven epistemic contexts apply different weights to the E/N/M axes,
//! recognizing that scientific, indigenous, contemplative, and other
//! knowledge traditions have legitimately different validation criteria.
//!
//! ## Epistemic Provenance
//!
//! Claims carry provenance tags indicating whether they originate from
//! embodied observation (sensorimotor), temporal pattern recognition,
//! social corroboration, or pure inference.
//!
//! ## References
//!
//! - Mycelix Epistemic Charter v2.0
//! - Wierzbicka, A. (1996). Semantics: Primes and Universals.
//! - Putnam, H. (1967). Psychological predicates (Multiple Realizability).
//! - Goldberg, S. (2023). Epistemic Health, Immunity, and Inoculation.

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
///
/// Inspired by:
/// - Two-Eyed Seeing (Etuaptmumk, Mi'kmaw epistemology)
/// - Wierzbicka's NSM universality
/// - Mycelix Epistemic Charter v2.0 Prismatic Architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpistemicContext {
    /// Scientific: empirical evidence weighted highest
    /// Weights: E=0.50, N=0.30, M=0.20
    Scientific,
    /// Governance: normative consensus weighted highest
    /// Weights: E=0.30, N=0.45, M=0.25
    Governance,
    /// Personal: balanced across all axes
    /// Weights: E=0.33, N=0.33, M=0.34
    Personal,
    /// Indigenous: relational knowledge, materiality (permanence/place) weighted highest
    /// Weights: E=0.25, N=0.40, M=0.35
    Indigenous,
    /// Contemplative: materiality (depth/permanence) weighted highest
    /// Weights: E=0.20, N=0.30, M=0.50
    Contemplative,
    /// Emergency: empirical evidence heavily weighted (act on best data NOW)
    /// Weights: E=0.60, N=0.25, M=0.15
    Emergency,
    /// Standard: default balanced weights (legacy compatibility)
    /// Weights: E=0.40, N=0.35, M=0.25
    Standard,
}

impl EpistemicContext {
    /// Returns (e_weight, n_weight, m_weight) for this context.
    /// Weights always sum to 1.0.
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

impl fmt::Display for EpistemicContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ==============================================================================
// EPISTEMIC COORDINATE (the full 3D+context position)
// ==============================================================================

/// Complete epistemic coordinate in 3D space with optional context.
///
/// The core type shared across Symthaea, Mycelix, and the bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EpistemicCoordinate {
    /// E-Axis: Empirical verifiability
    pub empirical: EmpiricalLevel,
    /// N-Axis: Normative authority (who agrees)
    pub normative: NormativeLevel,
    /// M-Axis: Materiality (how permanent)
    pub materiality: MaterialityLevel,
    /// Prismatic context (determines axis weights)
    pub context: EpistemicContext,
}

impl EpistemicCoordinate {
    /// Create new epistemic coordinate with explicit context.
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

    /// Create coordinate with Standard context (backward compatible).
    pub fn standard(
        empirical: EmpiricalLevel,
        normative: NormativeLevel,
        materiality: MaterialityLevel,
    ) -> Self {
        Self::new(
            empirical,
            normative,
            materiality,
            EpistemicContext::Standard,
        )
    }

    /// Weakest possible coordinate (E0/N0/M0/Standard).
    pub fn null() -> Self {
        Self::standard(
            EmpiricalLevel::E0Null,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
        )
    }

    /// Strongest possible coordinate (E4/N3/M3/Scientific).
    pub fn axiom() -> Self {
        Self::new(
            EmpiricalLevel::E4PubliclyReproducible,
            NormativeLevel::N3Axiomatic,
            MaterialityLevel::M3Foundational,
            EpistemicContext::Scientific,
        )
    }

    /// Quality score using Standard weights (E=0.40, N=0.35, M=0.25).
    /// For backward compatibility with existing code.
    pub fn quality_score(&self) -> f64 {
        self.contextual_quality_score(EpistemicContext::Standard)
    }

    /// Quality score using this coordinate's context weights.
    pub fn quality_score_contextual(&self) -> f64 {
        self.contextual_quality_score(self.context)
    }

    /// Quality score using specified context weights.
    pub fn contextual_quality_score(&self, context: EpistemicContext) -> f64 {
        let e_score = self.empirical.level() as f64 / 4.0;
        let n_score = self.normative.level() as f64 / 3.0;
        let m_score = self.materiality.level() as f64 / 3.0;
        let (ew, nw, mw) = context.weights();
        e_score * ew + n_score * nw + m_score * mw
    }

    /// Short notation: "E2/N0/M1"
    pub fn notation(&self) -> String {
        format!(
            "{}/{}/{}",
            self.empirical.abbreviation(),
            self.normative.abbreviation(),
            self.materiality.abbreviation()
        )
    }

    /// Full notation with context: "E2/N0/M1 [Indigenous]"
    pub fn full_notation(&self) -> String {
        format!("{} [{}]", self.notation(), self.context.name())
    }
}

impl Default for EpistemicCoordinate {
    fn default() -> Self {
        Self::null()
    }
}

impl fmt::Display for EpistemicCoordinate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.notation())
    }
}

// ==============================================================================
// EPISTEMIC PROVENANCE (where knowledge comes from)
// ==============================================================================

/// Grounding level for embodied epistemic provenance.
///
/// Based on the three-level grounding hierarchy:
/// - Level 1 (Sensorimotor): direct physical observation
/// - Level 2 (Temporal): persistent experience over time
/// - Level 3 (Social): corroboration with other agents
///
/// Reference: Roadmap for Embodied and Social Grounding in LLMs (arXiv 2409.16900)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GroundingLevel {
    /// Direct physical sensor observation (highest epistemic authority)
    Sensorimotor,
    /// Pattern recognized over temporal experience
    Temporal,
    /// Knowledge corroborated by other agents
    Social,
}

impl GroundingLevel {
    /// Suggested epistemic level for this grounding.
    pub fn suggested_empirical(&self) -> EmpiricalLevel {
        match self {
            Self::Sensorimotor => EmpiricalLevel::E2PrivatelyVerifiable,
            Self::Temporal => EmpiricalLevel::E1Testimonial,
            Self::Social => EmpiricalLevel::E1Testimonial,
        }
    }

    /// Confidence multiplier (higher grounding = more trust).
    pub fn confidence_multiplier(&self) -> f64 {
        match self {
            Self::Sensorimotor => 1.0,
            Self::Temporal => 0.8,
            Self::Social => 0.6,
        }
    }
}

impl fmt::Display for GroundingLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sensorimotor => write!(f, "sensorimotor"),
            Self::Temporal => write!(f, "temporal"),
            Self::Social => write!(f, "social"),
        }
    }
}

/// Provenance of an epistemic claim — how was this knowledge obtained?
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EpistemicProvenance {
    /// Directly observed via embodied sensors
    Embodied {
        level: GroundingLevel,
        platform: String,
        observation_confidence: f32,
    },
    /// Inferred from reasoning (no direct observation)
    Inferred { method: String, confidence: f32 },
    /// Validated by collective consensus
    Collective {
        corroboration_count: u32,
        network_coverage: f32,
    },
    /// Unknown or untracked provenance
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

/// Verdict from epistemic fact-checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FactCheckVerdict {
    /// Supported by knowledge graph
    True,
    /// Contradicted by knowledge graph
    False,
    /// Insufficient or conflicting evidence
    Mixed,
    /// No matching claims found
    Unknown,
}

impl FactCheckVerdict {
    /// Whether this verdict indicates the claim should be suppressed.
    pub fn should_suppress(&self, confidence: f32) -> bool {
        matches!(self, Self::False) && confidence >= 0.85
    }

    /// Cadence penalty for Broca generation (0.0 = no penalty).
    pub fn cadence_penalty(&self) -> f32 {
        match self {
            Self::True => 0.0,
            Self::False => 0.30,
            Self::Mixed => 0.15,
            Self::Unknown => 0.15,
        }
    }
}

impl fmt::Display for FactCheckVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::True => write!(f, "TRUE"),
            Self::False => write!(f, "FALSE"),
            Self::Mixed => write!(f, "MIXED"),
            Self::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

/// Result of a fact-check operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactCheckResult {
    /// The claim that was checked
    pub claim: String,
    /// Verdict from the knowledge graph
    pub verdict: FactCheckVerdict,
    /// Confidence in the verdict (0.0-1.0)
    pub confidence: f32,
    /// IDs of supporting/contradicting sources
    pub source_ids: Vec<String>,
    /// Epistemic coordinate of the aggregate evidence
    pub epistemic_position: Option<EpistemicCoordinate>,
}

impl FactCheckResult {
    pub fn unknown(claim: String) -> Self {
        Self {
            claim,
            verdict: FactCheckVerdict::Unknown,
            confidence: 0.0,
            source_ids: Vec::new(),
            epistemic_position: None,
        }
    }
}

// ==============================================================================
// TESTS
// ==============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empirical_level_ordering() {
        assert!(EmpiricalLevel::E0Null < EmpiricalLevel::E4PubliclyReproducible);
        assert_eq!(EmpiricalLevel::E3CryptographicallyProven.level(), 3);
    }

    #[test]
    fn test_normative_level_ordering() {
        assert!(NormativeLevel::N0Personal < NormativeLevel::N3Axiomatic);
        assert_eq!(NormativeLevel::N2Network.level(), 2);
    }

    #[test]
    fn test_materiality_level_ordering() {
        assert!(MaterialityLevel::M0Ephemeral < MaterialityLevel::M3Foundational);
        assert_eq!(MaterialityLevel::M1Temporal.level(), 1);
    }

    #[test]
    fn test_epistemic_context_weights_sum_to_one() {
        for ctx in [
            EpistemicContext::Scientific,
            EpistemicContext::Governance,
            EpistemicContext::Personal,
            EpistemicContext::Indigenous,
            EpistemicContext::Contemplative,
            EpistemicContext::Emergency,
            EpistemicContext::Standard,
        ] {
            let (e, n, m) = ctx.weights();
            let sum = e + n + m;
            assert!(
                (sum - 1.0).abs() < 0.01,
                "{:?} weights sum to {sum}, expected 1.0",
                ctx
            );
        }
    }

    #[test]
    fn test_coordinate_null_is_weakest() {
        let null = EpistemicCoordinate::null();
        assert_eq!(null.quality_score(), 0.0);
    }

    #[test]
    fn test_coordinate_axiom_is_strongest() {
        let axiom = EpistemicCoordinate::axiom();
        assert!(axiom.quality_score() > 0.95);
    }

    #[test]
    fn test_contextual_quality_varies_by_context() {
        let coord = EpistemicCoordinate::standard(
            EmpiricalLevel::E4PubliclyReproducible,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
        );
        let scientific = coord.contextual_quality_score(EpistemicContext::Scientific);
        let contemplative = coord.contextual_quality_score(EpistemicContext::Contemplative);
        // High empirical, low normative/materiality:
        // Scientific weights empirical heavily, so score should be higher
        assert!(
            scientific > contemplative,
            "scientific={scientific}, contemplative={contemplative}"
        );
    }

    #[test]
    fn test_indigenous_context_weights_normative_higher() {
        let coord = EpistemicCoordinate::standard(
            EmpiricalLevel::E0Null,
            NormativeLevel::N3Axiomatic,
            MaterialityLevel::M0Ephemeral,
        );
        let indigenous = coord.contextual_quality_score(EpistemicContext::Indigenous);
        let scientific = coord.contextual_quality_score(EpistemicContext::Scientific);
        // High normative only: Indigenous weights normative higher
        assert!(
            indigenous > scientific,
            "indigenous={indigenous}, scientific={scientific}"
        );
    }

    #[test]
    fn test_notation_format() {
        let coord = EpistemicCoordinate::standard(
            EmpiricalLevel::E2PrivatelyVerifiable,
            NormativeLevel::N1Communal,
            MaterialityLevel::M2Persistent,
        );
        assert_eq!(coord.notation(), "E2/N1/M2");
        assert_eq!(coord.full_notation(), "E2/N1/M2 [Standard]");
    }

    #[test]
    fn test_grounding_level_suggested_empirical() {
        assert_eq!(
            GroundingLevel::Sensorimotor.suggested_empirical(),
            EmpiricalLevel::E2PrivatelyVerifiable
        );
        assert_eq!(
            GroundingLevel::Temporal.suggested_empirical(),
            EmpiricalLevel::E1Testimonial
        );
    }

    #[test]
    fn test_fact_check_verdict_suppression() {
        assert!(FactCheckVerdict::False.should_suppress(0.90));
        assert!(!FactCheckVerdict::False.should_suppress(0.50));
        assert!(!FactCheckVerdict::True.should_suppress(0.99));
    }

    #[test]
    fn test_fact_check_verdict_cadence_penalty() {
        assert_eq!(FactCheckVerdict::True.cadence_penalty(), 0.0);
        assert_eq!(FactCheckVerdict::False.cadence_penalty(), 0.30);
        assert_eq!(FactCheckVerdict::Mixed.cadence_penalty(), 0.15);
    }

    #[test]
    fn test_provenance_default_is_unknown() {
        assert_eq!(EpistemicProvenance::default(), EpistemicProvenance::Unknown);
    }

    #[test]
    fn test_serde_roundtrip() {
        let coord = EpistemicCoordinate::new(
            EmpiricalLevel::E3CryptographicallyProven,
            NormativeLevel::N2Network,
            MaterialityLevel::M2Persistent,
            EpistemicContext::Indigenous,
        );
        let json = serde_json::to_string(&coord).unwrap();
        let roundtrip: EpistemicCoordinate = serde_json::from_str(&json).unwrap();
        assert_eq!(coord, roundtrip);
    }

    #[test]
    fn test_emergency_context_weights_empirical_highest() {
        let (e, n, m) = EpistemicContext::Emergency.weights();
        assert!(e > n && e > m, "Emergency should weight empirical highest");
    }
}
