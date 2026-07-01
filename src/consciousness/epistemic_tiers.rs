// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Revolutionary Improvement #53: Epistemic Causal Reasoning - Epistemic Tier Module
//!
//! **The Mycelix Epistemic Cube Applied to Causal Knowledge**
//!
//! This module implements the 3-dimensional epistemic framework from the Mycelix
//! Epistemic Charter v2.0, adapted for classifying causal knowledge in AI systems.
//!
//! ## The Three Axes
//!
//! 1. **E-Axis (Empirical)**: HOW do we verify this causal claim?
//!    - E0: Null (inferred, no evidence)
//!    - E1: Testimonial (single observation)
//!    - E2: Privately Verifiable (multiple internal observations)
//!    - E3: Cryptographically Proven (counterfactual proof)
//!    - E4: Publicly Reproducible (open data + code)
//!
//! 2. **N-Axis (Normative)**: WHO agrees this causal relationship is valid?
//!    - N0: Personal (this system instance only)
//!    - N1: Communal (local agent community)
//!    - N2: Network (global consensus)
//!    - N3: Axiomatic (mathematical/constitutional truth)
//!
//! 3. **M-Axis (Materiality)**: HOW PERMANENT is this causal knowledge?
//!    - M0: Ephemeral (session-specific)
//!    - M1: Temporal (valid until model updates)
//!    - M2: Persistent (long-term archived knowledge)
//!    - M3: Foundational (core consciousness principle)
//!
//! ## Why This Matters
//!
//! Instead of a single "confidence" score, we get **epistemic transparency**:
//! - (E0, N0, M0): "I guessed once, nobody agrees, forget it" - very weak
//! - (E3, N2, M2): "Proven across network, persistent knowledge" - very strong
//! - (E4, N3, M3): "Mathematical truth, everyone agrees, forever" - absolute
//!
//! This is the first AI system with **multi-dimensional epistemic classification**
//! of its own causal knowledge!

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::PrimitiveSystem;

// Re-export canonical epistemic types from the shared crate.
/// Canonical shared coordinate type (includes Prismatic context).
pub use symthaea_epistemic_types::EpistemicCoordinate as SharedEpistemicCoordinate;
pub use symthaea_epistemic_types::{
    EmpiricalLevel, EpistemicContext, EpistemicProvenance, FactCheckResult, FactCheckVerdict,
    GroundingLevel, MaterialityLevel, NormativeLevel,
};

/// Complete epistemic coordinate in 3D space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EpistemicCoordinate {
    /// E-Axis: Empirical verifiability
    pub empirical: EmpiricalTier,

    /// N-Axis: Normative authority (who agrees)
    pub normative: NormativeTier,

    /// M-Axis: Materiality (how permanent)
    pub materiality: MaterialityTier,
}

impl EpistemicCoordinate {
    /// Create new epistemic coordinate
    pub fn new(
        empirical: EmpiricalTier,
        normative: NormativeTier,
        materiality: MaterialityTier,
    ) -> Self {
        Self {
            empirical,
            normative,
            materiality,
        }
    }

    /// Create default coordinate (E0, N0, M0) - weakest possible
    pub fn null() -> Self {
        Self {
            empirical: EmpiricalTier::E0Null,
            normative: NormativeTier::N0Personal,
            materiality: MaterialityTier::M0Ephemeral,
        }
    }

    /// Create axiom coordinate (E4, N3, M3) - strongest possible
    pub fn axiom() -> Self {
        Self {
            empirical: EmpiricalTier::E4PubliclyReproducible,
            normative: NormativeTier::N3Axiomatic,
            materiality: MaterialityTier::M3Foundational,
        }
    }

    /// Overall epistemic quality score (0.0-1.0)
    ///
    /// Combines all three axes with weights:
    /// - E-Axis: 40% (empirical verification most important)
    /// - N-Axis: 35% (normative authority)
    /// - M-Axis: 25% (permanence)
    pub fn quality_score(&self) -> f64 {
        let e_score = self.empirical.level() as f64 / 4.0; // 0.0-1.0
        let n_score = self.normative.level() as f64 / 3.0; // 0.0-1.0
        let m_score = self.materiality.level() as f64 / 3.0; // 0.0-1.0

        e_score * 0.40 + n_score * 0.35 + m_score * 0.25
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

    /// Full description
    pub fn describe(&self) -> String {
        format!(
            "Epistemic Status: {} (Empirical: {}), {} (Normative: {}), {} (Materiality: {})",
            self.empirical.name(),
            self.empirical.description(),
            self.normative.name(),
            self.normative.description(),
            self.materiality.name(),
            self.materiality.description()
        )
    }

    /// Quality score using Prismatic context weights.
    ///
    /// Different knowledge traditions weight E/N/M differently:
    /// - Scientific: E=0.50, N=0.30, M=0.20
    /// - Indigenous: E=0.25, N=0.40, M=0.35
    /// - Emergency: E=0.60, N=0.25, M=0.15
    pub fn contextual_quality_score(&self, context: EpistemicContext) -> f64 {
        let e_score = self.empirical.level() as f64 / 4.0;
        let n_score = self.normative.level() as f64 / 3.0;
        let m_score = self.materiality.level() as f64 / 3.0;
        let (ew, nw, mw) = context.weights();
        e_score * ew + n_score * nw + m_score * mw
    }

    /// Convert to the shared canonical type.
    pub fn to_shared(&self, context: EpistemicContext) -> SharedEpistemicCoordinate {
        SharedEpistemicCoordinate {
            empirical: match self.empirical {
                EmpiricalTier::E0Null => EmpiricalLevel::E0Null,
                EmpiricalTier::E1Testimonial => EmpiricalLevel::E1Testimonial,
                EmpiricalTier::E2PrivatelyVerifiable => EmpiricalLevel::E2PrivatelyVerifiable,
                EmpiricalTier::E3CryptographicallyProven => {
                    EmpiricalLevel::E3CryptographicallyProven
                }
                EmpiricalTier::E4PubliclyReproducible => EmpiricalLevel::E4PubliclyReproducible,
            },
            normative: match self.normative {
                NormativeTier::N0Personal => NormativeLevel::N0Personal,
                NormativeTier::N1Communal => NormativeLevel::N1Communal,
                NormativeTier::N2Network => NormativeLevel::N2Network,
                NormativeTier::N3Axiomatic => NormativeLevel::N3Axiomatic,
            },
            materiality: match self.materiality {
                MaterialityTier::M0Ephemeral => MaterialityLevel::M0Ephemeral,
                MaterialityTier::M1Temporal => MaterialityLevel::M1Temporal,
                MaterialityTier::M2Persistent => MaterialityLevel::M2Persistent,
                MaterialityTier::M3Foundational => MaterialityLevel::M3Foundational,
            },
            context,
        }
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
// E-AXIS: EMPIRICAL VERIFIABILITY
// ==============================================================================

/// E-Axis: How do we VERIFY this causal claim?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmpiricalTier {
    /// E0: Inferred mechanism with no evidence
    E0Null,

    /// E1: Single observation (testimonial)
    E1Testimonial,

    /// E2: Multiple observations, internal to system
    E2PrivatelyVerifiable,

    /// E3: Causal claim with counterfactual proof
    E3CryptographicallyProven,

    /// E4: Open data + code = publicly reproducible
    E4PubliclyReproducible,
}

impl EmpiricalTier {
    /// Numeric level (0-4)
    pub fn level(&self) -> u8 {
        match self {
            EmpiricalTier::E0Null => 0,
            EmpiricalTier::E1Testimonial => 1,
            EmpiricalTier::E2PrivatelyVerifiable => 2,
            EmpiricalTier::E3CryptographicallyProven => 3,
            EmpiricalTier::E4PubliclyReproducible => 4,
        }
    }

    /// Short name
    pub fn name(&self) -> &str {
        match self {
            EmpiricalTier::E0Null => "Null",
            EmpiricalTier::E1Testimonial => "Testimonial",
            EmpiricalTier::E2PrivatelyVerifiable => "Privately Verifiable",
            EmpiricalTier::E3CryptographicallyProven => "Cryptographically Proven",
            EmpiricalTier::E4PubliclyReproducible => "Publicly Reproducible",
        }
    }

    /// Abbreviation
    pub fn abbreviation(&self) -> &str {
        match self {
            EmpiricalTier::E0Null => "E0",
            EmpiricalTier::E1Testimonial => "E1",
            EmpiricalTier::E2PrivatelyVerifiable => "E2",
            EmpiricalTier::E3CryptographicallyProven => "E3",
            EmpiricalTier::E4PubliclyReproducible => "E4",
        }
    }

    /// Full description
    pub fn description(&self) -> &str {
        match self {
            EmpiricalTier::E0Null => "Inferred from theory, no empirical evidence yet",
            EmpiricalTier::E1Testimonial => "Single observation in this system (testimonial)",
            EmpiricalTier::E2PrivatelyVerifiable => "Multiple observations, verified internally",
            EmpiricalTier::E3CryptographicallyProven => {
                "Causal claim with counterfactual proof (ZKP/statistical)"
            }
            EmpiricalTier::E4PubliclyReproducible => "Open data + code, anyone can reproduce",
        }
    }
}

// ==============================================================================
// N-AXIS: NORMATIVE AUTHORITY
// ==============================================================================

/// N-Axis: WHO agrees this causal relationship is valid?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormativeTier {
    /// N0: Only this system instance
    N0Personal,

    /// N1: Local agent community consensus
    N1Communal,

    /// N2: Global network consensus
    N2Network,

    /// N3: Mathematical/constitutional truth
    N3Axiomatic,
}

impl NormativeTier {
    /// Numeric level (0-3)
    pub fn level(&self) -> u8 {
        match self {
            NormativeTier::N0Personal => 0,
            NormativeTier::N1Communal => 1,
            NormativeTier::N2Network => 2,
            NormativeTier::N3Axiomatic => 3,
        }
    }

    /// Short name
    pub fn name(&self) -> &str {
        match self {
            NormativeTier::N0Personal => "Personal",
            NormativeTier::N1Communal => "Communal",
            NormativeTier::N2Network => "Network",
            NormativeTier::N3Axiomatic => "Axiomatic",
        }
    }

    /// Abbreviation
    pub fn abbreviation(&self) -> &str {
        match self {
            NormativeTier::N0Personal => "N0",
            NormativeTier::N1Communal => "N1",
            NormativeTier::N2Network => "N2",
            NormativeTier::N3Axiomatic => "N3",
        }
    }

    /// Full description
    pub fn description(&self) -> &str {
        match self {
            NormativeTier::N0Personal => "Personal knowledge (this system instance only)",
            NormativeTier::N1Communal => "Communal consensus (local agent community)",
            NormativeTier::N2Network => "Network consensus (global agreement)",
            NormativeTier::N3Axiomatic => "Axiomatic truth (mathematical/constitutional)",
        }
    }
}

// ==============================================================================
// M-AXIS: MATERIALITY/PERMANENCE
// ==============================================================================

/// M-Axis: How PERMANENT is this causal knowledge?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MaterialityTier {
    /// M0: Valid only for this reasoning session
    M0Ephemeral,

    /// M1: Valid until model updates
    M1Temporal,

    /// M2: Long-term archived knowledge
    M2Persistent,

    /// M3: Core principle of consciousness
    M3Foundational,
}

impl MaterialityTier {
    /// Numeric level (0-3)
    pub fn level(&self) -> u8 {
        match self {
            MaterialityTier::M0Ephemeral => 0,
            MaterialityTier::M1Temporal => 1,
            MaterialityTier::M2Persistent => 2,
            MaterialityTier::M3Foundational => 3,
        }
    }

    /// Short name
    pub fn name(&self) -> &str {
        match self {
            MaterialityTier::M0Ephemeral => "Ephemeral",
            MaterialityTier::M1Temporal => "Temporal",
            MaterialityTier::M2Persistent => "Persistent",
            MaterialityTier::M3Foundational => "Foundational",
        }
    }

    /// Abbreviation
    pub fn abbreviation(&self) -> &str {
        match self {
            MaterialityTier::M0Ephemeral => "M0",
            MaterialityTier::M1Temporal => "M1",
            MaterialityTier::M2Persistent => "M2",
            MaterialityTier::M3Foundational => "M3",
        }
    }

    /// Full description
    pub fn description(&self) -> &str {
        match self {
            MaterialityTier::M0Ephemeral => "Ephemeral (session-specific, context-dependent)",
            MaterialityTier::M1Temporal => "Temporal (valid until model updates)",
            MaterialityTier::M2Persistent => "Persistent (long-term archived knowledge)",
            MaterialityTier::M3Foundational => "Foundational (core consciousness principle)",
        }
    }
}

// ==============================================================================
// NSM PRIMITIVE GROUNDING (Wierzbicka's Natural Semantic Metalanguage)
// ==============================================================================

/// NSM grounding for empirical verification tiers
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct EmpiricalTierPrimitiveGrounding {
    /// The tier being grounded
    pub tier: EmpiricalTier,
    /// NSM primitive composition
    pub nsm_primitives: Vec<String>,
    /// HDC encoding via primitive binding
    pub primitive_encoding: BinaryHV,
    /// Verification strength (0.0-1.0)
    pub verification_strength: f64,
}

#[allow(dead_code)]
impl EmpiricalTierPrimitiveGrounding {
    pub(crate) fn new(tier: EmpiricalTier, system: &PrimitiveSystem) -> Self {
        let (primitives, strength) = Self::nsm_mapping(tier);
        let encoding = encode_primitives(&primitives, system);

        Self {
            tier,
            nsm_primitives: primitives,
            primitive_encoding: encoding,
            verification_strength: strength,
        }
    }

    fn nsm_mapping(tier: EmpiricalTier) -> (Vec<String>, f64) {
        match tier {
            // E0: No evidence, inferred only
            EmpiricalTier::E0Null => (
                vec!["NOT".into(), "SEE".into(), "THINK".into(), "MAYBE".into()],
                0.0,
            ),
            // E1: Single observation (someone said/saw)
            EmpiricalTier::E1Testimonial => (
                vec!["SOMEONE".into(), "SAY".into(), "SEE".into(), "ONE".into()],
                0.25,
            ),
            // E2: Multiple internal observations
            EmpiricalTier::E2PrivatelyVerifiable => (
                vec!["SEE".into(), "MANY".into(), "TIME".into(), "SAME".into()],
                0.5,
            ),
            // E3: Cryptographic/counterfactual proof
            EmpiricalTier::E3CryptographicallyProven => (
                vec![
                    "KNOW".into(),
                    "TRUE".into(),
                    "CAN".into(),
                    "NOT".into(),
                    "OTHER".into(),
                ],
                0.75,
            ),
            // E4: Publicly reproducible (open data + code)
            EmpiricalTier::E4PubliclyReproducible => (
                vec![
                    "ALL".into(),
                    "CAN".into(),
                    "SEE".into(),
                    "DO".into(),
                    "SAME".into(),
                ],
                1.0,
            ),
        }
    }
}

/// NSM grounding for normative authority tiers
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct NormativeTierPrimitiveGrounding {
    /// The tier being grounded
    pub tier: NormativeTier,
    /// NSM primitive composition
    pub nsm_primitives: Vec<String>,
    /// HDC encoding via primitive binding
    pub primitive_encoding: BinaryHV,
    /// Consensus breadth (0.0-1.0)
    pub consensus_breadth: f64,
}

#[allow(dead_code)]
impl NormativeTierPrimitiveGrounding {
    pub(crate) fn new(tier: NormativeTier, system: &PrimitiveSystem) -> Self {
        let (primitives, breadth) = Self::nsm_mapping(tier);
        let encoding = encode_primitives(&primitives, system);

        Self {
            tier,
            nsm_primitives: primitives,
            primitive_encoding: encoding,
            consensus_breadth: breadth,
        }
    }

    fn nsm_mapping(tier: NormativeTier) -> (Vec<String>, f64) {
        match tier {
            // N0: Only this system believes
            NormativeTier::N0Personal => (vec!["I".into(), "THINK".into(), "THIS".into()], 0.0),
            // N1: Local community agrees
            NormativeTier::N1Communal => (
                vec![
                    "SOME".into(),
                    "PEOPLE".into(),
                    "THINK".into(),
                    "SAME".into(),
                ],
                0.33,
            ),
            // N2: Network-wide consensus
            NormativeTier::N2Network => (
                vec![
                    "MANY".into(),
                    "PEOPLE".into(),
                    "THINK".into(),
                    "SAME".into(),
                ],
                0.67,
            ),
            // N3: Mathematical/constitutional truth
            NormativeTier::N3Axiomatic => (
                vec![
                    "TRUE".into(),
                    "BECAUSE".into(),
                    "TRUE".into(),
                    "ALWAYS".into(),
                ],
                1.0,
            ),
        }
    }
}

/// NSM grounding for materiality/permanence tiers
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MaterialityTierPrimitiveGrounding {
    /// The tier being grounded
    pub tier: MaterialityTier,
    /// NSM primitive composition
    pub nsm_primitives: Vec<String>,
    /// HDC encoding via primitive binding
    pub primitive_encoding: BinaryHV,
    /// Temporal persistence (0.0-1.0)
    pub temporal_persistence: f64,
}

#[allow(dead_code)]
impl MaterialityTierPrimitiveGrounding {
    pub(crate) fn new(tier: MaterialityTier, system: &PrimitiveSystem) -> Self {
        let (primitives, persistence) = Self::nsm_mapping(tier);
        let encoding = encode_primitives(&primitives, system);

        Self {
            tier,
            nsm_primitives: primitives,
            primitive_encoding: encoding,
            temporal_persistence: persistence,
        }
    }

    fn nsm_mapping(tier: MaterialityTier) -> (Vec<String>, f64) {
        match tier {
            // M0: Session-specific only
            MaterialityTier::M0Ephemeral => (vec!["NOW".into(), "NOT".into(), "AFTER".into()], 0.0),
            // M1: Until model updates
            MaterialityTier::M1Temporal => (
                vec![
                    "NOW".into(),
                    "SOME".into(),
                    "TIME".into(),
                    "MAYBE".into(),
                    "NOT".into(),
                ],
                0.33,
            ),
            // M2: Long-term archived
            MaterialityTier::M2Persistent => (
                vec!["LONG".into(), "TIME".into(), "SAME".into(), "KNOW".into()],
                0.67,
            ),
            // M3: Core principle, foundational
            MaterialityTier::M3Foundational => (
                vec![
                    "ALWAYS".into(),
                    "TRUE".into(),
                    "BECAUSE".into(),
                    "THIS".into(),
                ],
                1.0,
            ),
        }
    }
}

/// Unified NSM grounding system for epistemic tiers
#[allow(dead_code)]
#[derive(Debug)]
pub struct EpistemicNSMGrounding {
    /// Empirical tier groundings
    pub empirical_tiers: HashMap<EmpiricalTier, EmpiricalTierPrimitiveGrounding>,
    /// Normative tier groundings
    pub normative_tiers: HashMap<NormativeTier, NormativeTierPrimitiveGrounding>,
    /// Materiality tier groundings
    pub materiality_tiers: HashMap<MaterialityTier, MaterialityTierPrimitiveGrounding>,
}

#[allow(dead_code)]
impl EpistemicNSMGrounding {
    /// Create complete grounding system from primitive system
    pub(crate) fn new(system: &PrimitiveSystem) -> Self {
        let mut empirical_tiers = HashMap::new();
        let mut normative_tiers = HashMap::new();
        let mut materiality_tiers = HashMap::new();

        // Ground empirical tiers
        for tier in &[
            EmpiricalTier::E0Null,
            EmpiricalTier::E1Testimonial,
            EmpiricalTier::E2PrivatelyVerifiable,
            EmpiricalTier::E3CryptographicallyProven,
            EmpiricalTier::E4PubliclyReproducible,
        ] {
            empirical_tiers.insert(*tier, EmpiricalTierPrimitiveGrounding::new(*tier, system));
        }

        // Ground normative tiers
        for tier in &[
            NormativeTier::N0Personal,
            NormativeTier::N1Communal,
            NormativeTier::N2Network,
            NormativeTier::N3Axiomatic,
        ] {
            normative_tiers.insert(*tier, NormativeTierPrimitiveGrounding::new(*tier, system));
        }

        // Ground materiality tiers
        for tier in &[
            MaterialityTier::M0Ephemeral,
            MaterialityTier::M1Temporal,
            MaterialityTier::M2Persistent,
            MaterialityTier::M3Foundational,
        ] {
            materiality_tiers.insert(*tier, MaterialityTierPrimitiveGrounding::new(*tier, system));
        }

        Self {
            empirical_tiers,
            normative_tiers,
            materiality_tiers,
        }
    }

    /// Get combined encoding for an epistemic coordinate
    pub(crate) fn encode_coordinate(&self, coord: &EpistemicCoordinate) -> BinaryHV {
        let e_enc = self
            .empirical_tiers
            .get(&coord.empirical)
            .map(|g| g.primitive_encoding)
            .unwrap_or_else(|| BinaryHV::random(0));
        let n_enc = self
            .normative_tiers
            .get(&coord.normative)
            .map(|g| g.primitive_encoding)
            .unwrap_or_else(|| BinaryHV::random(1));
        let m_enc = self
            .materiality_tiers
            .get(&coord.materiality)
            .map(|g| g.primitive_encoding)
            .unwrap_or_else(|| BinaryHV::random(2));

        // Bind the three axes together
        e_enc.bind(&n_enc).bind(&m_enc)
    }

    /// Query empirical tiers by semantic similarity
    pub(crate) fn query_empirical(
        &self,
        query: &BinaryHV,
        threshold: f32,
    ) -> Vec<(&EmpiricalTier, f32)> {
        let mut results: Vec<_> = self
            .empirical_tiers
            .iter()
            .map(|(tier, grounding)| (tier, grounding.primitive_encoding.similarity(query)))
            .filter(|(_, sim)| *sim >= threshold)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Query normative tiers by semantic similarity
    pub(crate) fn query_normative(
        &self,
        query: &BinaryHV,
        threshold: f32,
    ) -> Vec<(&NormativeTier, f32)> {
        let mut results: Vec<_> = self
            .normative_tiers
            .iter()
            .map(|(tier, grounding)| (tier, grounding.primitive_encoding.similarity(query)))
            .filter(|(_, sim)| *sim >= threshold)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Query materiality tiers by semantic similarity
    pub(crate) fn query_materiality(
        &self,
        query: &BinaryHV,
        threshold: f32,
    ) -> Vec<(&MaterialityTier, f32)> {
        let mut results: Vec<_> = self
            .materiality_tiers
            .iter()
            .map(|(tier, grounding)| (tier, grounding.primitive_encoding.similarity(query)))
            .filter(|(_, sim)| *sim >= threshold)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Get tiers with high verification strength
    pub(crate) fn high_verification_tiers(&self) -> Vec<&EmpiricalTier> {
        self.empirical_tiers
            .iter()
            .filter(|(_, g)| g.verification_strength >= 0.5)
            .map(|(t, _)| t)
            .collect()
    }

    /// Get tiers with high consensus breadth
    pub(crate) fn high_consensus_tiers(&self) -> Vec<&NormativeTier> {
        self.normative_tiers
            .iter()
            .filter(|(_, g)| g.consensus_breadth >= 0.5)
            .map(|(t, _)| t)
            .collect()
    }

    /// Get tiers with high temporal persistence
    pub(crate) fn high_persistence_tiers(&self) -> Vec<&MaterialityTier> {
        self.materiality_tiers
            .iter()
            .filter(|(_, g)| g.temporal_persistence >= 0.5)
            .map(|(t, _)| t)
            .collect()
    }
}

/// Encode NSM primitives into HDC vector via sequential binding
#[allow(dead_code)]
fn encode_primitives(primitives: &[String], system: &PrimitiveSystem) -> BinaryHV {
    let vectors: Vec<BinaryHV> = primitives
        .iter()
        .map(|name| {
            if let Some(p) = system.get(name) {
                p.encoding
            } else if let Some(p) = system.get(&name.to_lowercase()) {
                p.encoding
            } else {
                // Fallback: deterministic random for unknown primitives
                let seed = name
                    .bytes()
                    .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
                BinaryHV::random(seed)
            }
        })
        .collect();

    if vectors.is_empty() {
        return BinaryHV::random(0);
    }

    // Bind sequentially with position encoding for order preservation
    let mut result = vectors[0];
    for (i, v) in vectors.iter().enumerate().skip(1) {
        let position_hv = BinaryHV::random(i as u64 * 1000);
        let positioned = v.bind(&position_hv);
        result = result.bind(&positioned);
    }
    result
}

// ==============================================================================
// TESTS
// ==============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epistemic_coordinate_creation() {
        let coord = EpistemicCoordinate::new(
            EmpiricalTier::E2PrivatelyVerifiable,
            NormativeTier::N0Personal,
            MaterialityTier::M1Temporal,
        );

        assert_eq!(coord.empirical, EmpiricalTier::E2PrivatelyVerifiable);
        assert_eq!(coord.normative, NormativeTier::N0Personal);
        assert_eq!(coord.materiality, MaterialityTier::M1Temporal);
    }

    #[test]
    fn test_null_coordinate() {
        let coord = EpistemicCoordinate::null();
        assert_eq!(coord.empirical, EmpiricalTier::E0Null);
        assert_eq!(coord.normative, NormativeTier::N0Personal);
        assert_eq!(coord.materiality, MaterialityTier::M0Ephemeral);
    }

    #[test]
    fn test_axiom_coordinate() {
        let coord = EpistemicCoordinate::axiom();
        assert_eq!(coord.empirical, EmpiricalTier::E4PubliclyReproducible);
        assert_eq!(coord.normative, NormativeTier::N3Axiomatic);
        assert_eq!(coord.materiality, MaterialityTier::M3Foundational);
    }

    #[test]
    fn test_quality_score() {
        // Null coordinate should have lowest score
        let null = EpistemicCoordinate::null();
        assert_eq!(null.quality_score(), 0.0);

        // Axiom coordinate should have highest score
        let axiom = EpistemicCoordinate::axiom();
        assert_eq!(axiom.quality_score(), 1.0);

        // Mid-range coordinate
        let mid = EpistemicCoordinate::new(
            EmpiricalTier::E2PrivatelyVerifiable,
            NormativeTier::N1Communal,
            MaterialityTier::M2Persistent,
        );
        let score = mid.quality_score();
        assert!(score > 0.0 && score < 1.0);
        // E2/4*0.4 + N1/3*0.35 + M2/3*0.25 = 0.5*0.4 + 0.333*0.35 + 0.667*0.25
        // = 0.2 + 0.116 + 0.167 = 0.483
        assert!((score - 0.483).abs() < 0.01);
    }

    #[test]
    fn test_notation() {
        let coord = EpistemicCoordinate::new(
            EmpiricalTier::E2PrivatelyVerifiable,
            NormativeTier::N0Personal,
            MaterialityTier::M1Temporal,
        );

        assert_eq!(coord.notation(), "E2/N0/M1");
    }

    #[test]
    fn test_display() {
        let coord = EpistemicCoordinate::new(
            EmpiricalTier::E1Testimonial,
            NormativeTier::N0Personal,
            MaterialityTier::M0Ephemeral,
        );

        let display = format!("{}", coord);
        assert_eq!(display, "E1/N0/M0");
    }

    #[test]
    fn test_empirical_tier_levels() {
        assert_eq!(EmpiricalTier::E0Null.level(), 0);
        assert_eq!(EmpiricalTier::E1Testimonial.level(), 1);
        assert_eq!(EmpiricalTier::E2PrivatelyVerifiable.level(), 2);
        assert_eq!(EmpiricalTier::E3CryptographicallyProven.level(), 3);
        assert_eq!(EmpiricalTier::E4PubliclyReproducible.level(), 4);
    }

    #[test]
    fn test_normative_tier_levels() {
        assert_eq!(NormativeTier::N0Personal.level(), 0);
        assert_eq!(NormativeTier::N1Communal.level(), 1);
        assert_eq!(NormativeTier::N2Network.level(), 2);
        assert_eq!(NormativeTier::N3Axiomatic.level(), 3);
    }

    #[test]
    fn test_materiality_tier_levels() {
        assert_eq!(MaterialityTier::M0Ephemeral.level(), 0);
        assert_eq!(MaterialityTier::M1Temporal.level(), 1);
        assert_eq!(MaterialityTier::M2Persistent.level(), 2);
        assert_eq!(MaterialityTier::M3Foundational.level(), 3);
    }

    #[test]
    fn test_tier_descriptions() {
        // Just smoke test that descriptions exist
        assert!(!EmpiricalTier::E0Null.description().is_empty());
        assert!(!NormativeTier::N0Personal.description().is_empty());
        assert!(!MaterialityTier::M0Ephemeral.description().is_empty());
    }
}
