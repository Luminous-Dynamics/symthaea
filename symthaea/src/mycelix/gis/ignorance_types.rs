// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Ignorance Type Taxonomy
//!
//! Implements the 5-type ignorance taxonomy from GIS v1.0, extended in v4.0 with
//! Harmonic Ignorance for Eight Harmonies integration.
//!
//! | Symbol | Name | Description |
//! |--------|------|-------------|
//! | κ | Known | Information exists and is accessible |
//! | ι₁ | Known Unknown | We know we don't know this specific thing |
//! | ι₂ | Unknown Unknown | We don't know what we don't know |
//! | ι₃ | Impossible | Fundamentally unknowable |
//! | ι∞ | None | No ignorance (complete knowledge) |
//!
//! ## GIS v4.0 Extension: Harmonic Ignorance
//!
//! Ignorance is not uniform across the Eight Harmonies. A gap in Care-Knowing
//! (Pan-Sentient Flourishing) differs fundamentally from a gap in Truth-Knowing
//! (Integral Wisdom).
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::mycelix::gis::{IgnoranceType, HarmonicIgnorance, Harmony};
//!
//! let mut harmonic_ig = HarmonicIgnorance::new(IgnoranceType::KnownUnknown);
//! harmonic_ig.add_affected_harmony(Harmony::PanSentientFlourishing, 0.8);
//! harmonic_ig.add_affected_harmony(Harmony::IntegralWisdom, 0.6);
//!
//! println!("Primary gap: {:?}", harmonic_ig.primary_harmony_gap());
//! println!("Total impact: {:.2}", harmonic_ig.total_harmonic_impact());
//! ```

use std::time::SystemTime;

/// The 5-type ignorance taxonomy
///
/// Based on epistemological analysis of what kinds of "not knowing" exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IgnoranceType {
    /// No ignorance - we have complete knowledge (κ)
    None,

    /// Known - Information exists somewhere, we know what it is but don't have it (κ temporally limited)
    Known,

    /// Known Unknown (ι₁) - We know exactly what we don't know
    /// "I don't know the population of Tokyo, but I know the question is well-formed"
    KnownUnknown,

    /// Unknown Unknown (ι₂) - We don't know what we don't know
    /// "There may be relevant factors I'm not even aware of"
    Unknown,

    /// Impossible (ι₃) - Fundamentally unknowable
    /// "What is north of the North Pole?" - the question is malformed
    Impossible,
}

impl IgnoranceType {
    /// Can this type of ignorance be resolved in principle?
    pub fn is_resolvable(&self) -> bool {
        match self {
            Self::None => true,         // Already resolved
            Self::Known => true,        // Just need to fetch it
            Self::KnownUnknown => true, // Can research it
            Self::Unknown => false,     // Can't target what we don't know
            Self::Impossible => false,  // Fundamentally impossible
        }
    }

    /// Get the resolution strategy
    pub fn resolution_strategy(&self) -> ResolutionStrategy {
        match self {
            Self::None => ResolutionStrategy::None,
            Self::Known => ResolutionStrategy::Fetch,
            Self::KnownUnknown => ResolutionStrategy::Research,
            Self::Unknown => ResolutionStrategy::Explore,
            Self::Impossible => ResolutionStrategy::Reframe,
        }
    }

    /// Get confidence ceiling for this ignorance type
    ///
    /// Even if we answer, this is the max confidence we can claim.
    pub fn confidence_ceiling(&self) -> f32 {
        match self {
            Self::None => 1.0,
            Self::Known => 0.95,
            Self::KnownUnknown => 0.85,
            Self::Unknown => 0.50,
            Self::Impossible => 0.10,
        }
    }

    /// Get Greek symbol
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::None => "κ",
            Self::Known => "κ-t",
            Self::KnownUnknown => "ι₁",
            Self::Unknown => "ι₂",
            Self::Impossible => "ι₃",
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::None => "Complete knowledge - no ignorance",
            Self::Known => "Information exists but is not currently accessible",
            Self::KnownUnknown => "We know what we don't know",
            Self::Unknown => "We don't know what we don't know",
            Self::Impossible => "Fundamentally unknowable",
        }
    }
}

/// Strategy for resolving ignorance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionStrategy {
    /// No resolution needed
    None,
    /// Fetch from existing source
    Fetch,
    /// Active research required
    Research,
    /// Broad exploration needed
    Explore,
    /// Question needs reframing
    Reframe,
}

impl ResolutionStrategy {
    /// Get the expected effort level (0.0 - 1.0)
    pub fn effort_level(&self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::Fetch => 0.1,
            Self::Research => 0.5,
            Self::Explore => 0.8,
            Self::Reframe => 0.3,
        }
    }
}

/// Domain of knowledge
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Domain {
    /// Mathematics - formal proofs possible
    Mathematics,
    /// Physics - empirical verification possible
    Physics,
    /// History - documentary evidence possible
    History,
    /// Subjective - personal experience
    Subjective,
    /// General knowledge
    General,
    /// Undefined/malformed domain
    Undefined,
}

impl Domain {
    /// Is this domain amenable to formal proof?
    pub fn is_formal(&self) -> bool {
        matches!(self, Domain::Mathematics)
    }

    /// Is this domain empirically verifiable?
    pub fn is_empirical(&self) -> bool {
        matches!(self, Domain::Physics | Domain::History)
    }

    /// Is this domain inherently subjective?
    pub fn is_subjective(&self) -> bool {
        matches!(self, Domain::Subjective)
    }
}

impl std::fmt::Display for Domain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Domain::Mathematics => write!(f, "mathematics"),
            Domain::Physics => write!(f, "physics"),
            Domain::History => write!(f, "history"),
            Domain::Subjective => write!(f, "subjective"),
            Domain::General => write!(f, "general"),
            Domain::Undefined => write!(f, "undefined"),
        }
    }
}

/// Ignorance record for tracking
#[derive(Debug, Clone)]
pub struct IgnoranceRecord {
    /// Unique identifier
    pub id: String,

    /// The detection result
    pub detection: super::IgnoranceDetection,

    /// Current status
    pub status: IgnoranceStatus,

    /// Resolution (if resolved)
    pub resolution: Option<IgnoranceResolution>,

    /// When created
    pub created_at: SystemTime,

    /// When last updated
    pub updated_at: SystemTime,
}

/// Status of an ignorance record
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IgnoranceStatus {
    /// Active - still ignorant
    Active,
    /// Resolution has been requested
    ResolutionRequested,
    /// Successfully resolved
    Resolved,
    /// Expired without resolution
    Expired,
}

/// Resolution of an ignorance
#[derive(Debug, Clone)]
pub struct IgnoranceResolution {
    /// How was it resolved?
    pub method: ResolutionMethod,

    /// The answer (if applicable)
    pub answer: Option<String>,

    /// Confidence in the resolution
    pub confidence: f32,

    /// Source of resolution
    pub source: String,

    /// When resolved
    pub resolved_at: SystemTime,
}

/// Method of resolution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionMethod {
    /// Found in local knowledge
    LocalKnowledge,
    /// Retrieved from network
    NetworkRetrieval,
    /// Derived through reasoning
    Derivation,
    /// Received from Dark Spot DHT
    DarkSpotMatch,
    /// User provided answer
    UserProvided,
    /// Question was reframed
    Reframed,
}

// =============================================================================
// GIS v4.0: Eight Harmonies Integration
// =============================================================================

// Canonical Harmony type from shared types crate
pub use symthaea_types::Harmony;

/// Harmonic Ignorance: Ignorance weighted by affected harmonies
///
/// This is the core GIS v4.0 extension. Ignorance is not uniform -
/// a gap in Care-Knowing differs fundamentally from a gap in Truth-Knowing.
#[derive(Debug, Clone)]
pub struct HarmonicIgnorance {
    /// The base ignorance type
    pub base_ignorance: IgnoranceType,

    /// Which harmonies are affected and by how much (0.0 - 1.0)
    pub affected_harmonies: Vec<(Harmony, f32)>,

    /// When this harmonic ignorance was detected
    pub detected_at: SystemTime,
}

impl HarmonicIgnorance {
    /// Create a new harmonic ignorance with base type
    pub fn new(base_ignorance: IgnoranceType) -> Self {
        Self {
            base_ignorance,
            affected_harmonies: Vec::new(),
            detected_at: SystemTime::now(),
        }
    }

    /// Add an affected harmony with its impact level
    pub fn add_affected_harmony(&mut self, harmony: Harmony, impact: f32) {
        let impact = impact.clamp(0.0, 1.0);
        // Update if already present, otherwise add
        if let Some(entry) = self
            .affected_harmonies
            .iter_mut()
            .find(|(h, _)| *h == harmony)
        {
            entry.1 = impact;
        } else {
            self.affected_harmonies.push((harmony, impact));
        }
    }

    /// Calculate total harmonic impact (weighted sum)
    pub fn total_harmonic_impact(&self) -> f32 {
        self.affected_harmonies
            .iter()
            .map(|(harmony, impact)| harmony.base_weight() * impact)
            .sum()
    }

    /// Get the primary (most affected) harmony gap
    pub fn primary_harmony_gap(&self) -> Option<Harmony> {
        self.affected_harmonies
            .iter()
            .max_by(|a, b| {
                let weight_a = a.0.base_weight() * a.1;
                let weight_b = b.0.base_weight() * b.1;
                weight_a
                    .partial_cmp(&weight_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(h, _)| *h)
    }

    /// Get harmonies above a threshold impact
    pub fn significant_gaps(&self, threshold: f32) -> Vec<Harmony> {
        self.affected_harmonies
            .iter()
            .filter(|(_, impact)| *impact >= threshold)
            .map(|(h, _)| *h)
            .collect()
    }

    /// Generate resolution paths based on affected harmonies
    pub fn resolution_paths(&self) -> Vec<HarmonicResolution> {
        self.affected_harmonies
            .iter()
            .filter(|(_, impact)| *impact > 0.3) // Only significant gaps
            .map(|(harmony, impact)| HarmonicResolution {
                harmony: *harmony,
                impact: *impact,
                strategy: harmony_resolution_strategy(*harmony, self.base_ignorance),
                estimated_effort: harmony_resolution_effort(*harmony, *impact),
            })
            .collect()
    }

    /// Generate the H-dimension code for E/N/M/H notation
    pub fn h_code(&self) -> String {
        if self.affected_harmonies.is_empty() {
            return String::from("H{}");
        }

        let parts: Vec<String> = self
            .affected_harmonies
            .iter()
            .filter(|(_, impact)| *impact > 0.1) // Only show significant impacts
            .map(|(h, impact)| format!("{}:{:.1}", h.code(), impact))
            .collect();

        format!("H{{{}}}", parts.join(","))
    }

    /// Calculate confidence ceiling based on harmonic impacts
    ///
    /// High impact on PSF or IW lowers confidence ceiling.
    pub fn confidence_ceiling(&self) -> f32 {
        let base_ceiling = self.base_ignorance.confidence_ceiling();

        // Critical harmonies reduce ceiling more
        let critical_penalty: f32 = self
            .affected_harmonies
            .iter()
            .filter(|(h, _)| matches!(h, Harmony::PanSentientFlourishing | Harmony::IntegralWisdom))
            .map(|(_, impact)| impact * 0.2)
            .sum();

        (base_ceiling - critical_penalty).max(0.1)
    }
}

/// Resolution path for a specific harmony gap
#[derive(Debug, Clone)]
pub struct HarmonicResolution {
    /// The harmony to address
    pub harmony: Harmony,
    /// Current impact level
    pub impact: f32,
    /// Recommended resolution strategy
    pub strategy: HarmonicResolutionStrategy,
    /// Estimated effort (0.0 - 1.0)
    pub estimated_effort: f32,
}

/// Harmony-specific resolution strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarmonicResolutionStrategy {
    /// Seek integration, find connecting patterns
    SeekIntegration,
    /// Consult affected parties, gather perspectives
    ConsultAffected,
    /// Verify facts, seek evidence
    VerifyEvidence,
    /// Explore possibilities, brainstorm
    ExplorePossibilities,
    /// Map relationships, trace connections
    MapRelationships,
    /// Consider exchange dynamics, reciprocity
    AnalyzeReciprocity,
    /// Study development patterns, emergence
    StudyEmergence,
    /// General research
    Research,
}

/// Determine resolution strategy based on harmony and ignorance type
fn harmony_resolution_strategy(
    harmony: Harmony,
    ignorance: IgnoranceType,
) -> HarmonicResolutionStrategy {
    match (harmony, ignorance) {
        (Harmony::ResonantCoherence, _) => HarmonicResolutionStrategy::SeekIntegration,
        (Harmony::PanSentientFlourishing, _) => HarmonicResolutionStrategy::ConsultAffected,
        (Harmony::IntegralWisdom, _) => HarmonicResolutionStrategy::VerifyEvidence,
        (Harmony::InfinitePlay, _) => HarmonicResolutionStrategy::ExplorePossibilities,
        (Harmony::UniversalInterconnectedness, _) => HarmonicResolutionStrategy::MapRelationships,
        (Harmony::SacredReciprocity, _) => HarmonicResolutionStrategy::AnalyzeReciprocity,
        (Harmony::EvolutionaryProgression, _) => HarmonicResolutionStrategy::StudyEmergence,
        (Harmony::SacredStillness, _) => HarmonicResolutionStrategy::SeekIntegration,
    }
}

/// Estimate resolution effort based on harmony and impact
fn harmony_resolution_effort(harmony: Harmony, impact: f32) -> f32 {
    let base_effort = match harmony {
        // Social harmonies require more effort (stakeholder engagement)
        Harmony::PanSentientFlourishing => 0.7,
        Harmony::UniversalInterconnectedness => 0.6,
        // Verification takes moderate effort
        Harmony::IntegralWisdom => 0.5,
        // Integration can be quick if data exists
        Harmony::ResonantCoherence => 0.4,
        // Creative exploration is variable
        Harmony::InfinitePlay => 0.3,
        // Pattern recognition can be efficient
        Harmony::SacredReciprocity => 0.4,
        Harmony::EvolutionaryProgression => 0.5,
        Harmony::SacredStillness => 0.3,
    };

    (base_effort * impact).clamp(0.1, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ignorance_resolvability() {
        assert!(IgnoranceType::None.is_resolvable());
        assert!(IgnoranceType::Known.is_resolvable());
        assert!(IgnoranceType::KnownUnknown.is_resolvable());
        assert!(!IgnoranceType::Unknown.is_resolvable());
        assert!(!IgnoranceType::Impossible.is_resolvable());
    }

    #[test]
    fn test_confidence_ceilings() {
        assert!(
            IgnoranceType::None.confidence_ceiling()
                > IgnoranceType::KnownUnknown.confidence_ceiling()
        );
        assert!(
            IgnoranceType::KnownUnknown.confidence_ceiling()
                > IgnoranceType::Unknown.confidence_ceiling()
        );
        assert!(
            IgnoranceType::Unknown.confidence_ceiling()
                > IgnoranceType::Impossible.confidence_ceiling()
        );
    }

    #[test]
    fn test_domain_classification() {
        assert!(Domain::Mathematics.is_formal());
        assert!(!Domain::Mathematics.is_empirical());
        assert!(Domain::Physics.is_empirical());
        assert!(Domain::Subjective.is_subjective());
    }

    #[test]
    fn test_symbols() {
        assert_eq!(IgnoranceType::None.symbol(), "κ");
        assert_eq!(IgnoranceType::KnownUnknown.symbol(), "ι₁");
        assert_eq!(IgnoranceType::Unknown.symbol(), "ι₂");
        assert_eq!(IgnoranceType::Impossible.symbol(), "ι₃");
    }

    // === GIS v4.0 Harmony Tests ===

    #[test]
    fn test_harmony_weights_sum_to_one() {
        let total: f32 = Harmony::all().iter().map(|h| h.base_weight()).sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "Harmony weights should sum to ~1.0, got {}",
            total
        );
    }

    #[test]
    fn test_harmony_codes() {
        assert_eq!(Harmony::ResonantCoherence.code(), "RC");
        assert_eq!(Harmony::PanSentientFlourishing.code(), "PSF");
        assert_eq!(Harmony::IntegralWisdom.code(), "IW");
    }

    #[test]
    fn test_harmonic_ignorance_creation() {
        let mut hi = HarmonicIgnorance::new(IgnoranceType::KnownUnknown);
        hi.add_affected_harmony(Harmony::PanSentientFlourishing, 0.8);
        hi.add_affected_harmony(Harmony::IntegralWisdom, 0.6);

        assert_eq!(hi.affected_harmonies.len(), 2);
        assert_eq!(
            hi.primary_harmony_gap(),
            Some(Harmony::PanSentientFlourishing)
        );
    }

    #[test]
    fn test_harmonic_impact_calculation() {
        let mut hi = HarmonicIgnorance::new(IgnoranceType::KnownUnknown);
        hi.add_affected_harmony(Harmony::PanSentientFlourishing, 1.0); // weight 0.17

        let impact = hi.total_harmonic_impact();
        assert!(
            (impact - 0.17).abs() < 0.01,
            "Impact should be ~0.17, got {}",
            impact
        );
    }

    #[test]
    fn test_h_code_generation() {
        let mut hi = HarmonicIgnorance::new(IgnoranceType::KnownUnknown);
        hi.add_affected_harmony(Harmony::ResonantCoherence, 0.8);
        hi.add_affected_harmony(Harmony::InfinitePlay, 0.3);

        let code = hi.h_code();
        assert!(
            code.contains("RC:0.8"),
            "Should contain RC:0.8, got {}",
            code
        );
        assert!(
            code.contains("IP:0.3"),
            "Should contain IP:0.3, got {}",
            code
        );
    }

    #[test]
    fn test_resolution_paths() {
        let mut hi = HarmonicIgnorance::new(IgnoranceType::KnownUnknown);
        hi.add_affected_harmony(Harmony::PanSentientFlourishing, 0.8);
        hi.add_affected_harmony(Harmony::IntegralWisdom, 0.5);
        hi.add_affected_harmony(Harmony::InfinitePlay, 0.1); // Too low, should be excluded

        let paths = hi.resolution_paths();
        assert_eq!(paths.len(), 2); // Only PSF and IW
        assert!(
            paths
                .iter()
                .any(|p| p.harmony == Harmony::PanSentientFlourishing)
        );
        assert!(paths.iter().any(|p| p.harmony == Harmony::IntegralWisdom));
    }

    #[test]
    fn test_harmonic_confidence_ceiling() {
        let mut hi = HarmonicIgnorance::new(IgnoranceType::KnownUnknown);
        let base_ceiling = hi.base_ignorance.confidence_ceiling();

        // Add critical harmony impact
        hi.add_affected_harmony(Harmony::PanSentientFlourishing, 0.8);

        let adjusted_ceiling = hi.confidence_ceiling();
        assert!(
            adjusted_ceiling < base_ceiling,
            "Critical harmony should reduce ceiling"
        );
    }
}
