// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! 8-Dimensional Sovereign Profile for consciousness-gated governance.
//!
//! Duplicated from `crates/sovereign-profile` per Phase 2a survey decision:
//! sim-side duplication with ~monthly manual sync avoids cross-crate coupling
//! (the sim would otherwise break every time `mycelix-bridge-common` changes).
//!
//! When you reduce a citizen to a single Phi score, you create a system that
//! is infinitely gamifiable. The 8D profile maps identity as a multi-faceted
//! geometry — a reclusive solar engineer and a social mediator can both pass
//! Citizen through completely different dimensional routes.
//!
//! ## Mapping to the sim
//!
//! Axes are populated from existing agent state where a direct mapping exists,
//! and sampled from culture-conditioned distributions otherwise:
//!
//! | Dimension | Source in sim |
//! |---|---|
//! | EpistemicIntegrity | `coordination_understanding` + `education_level` |
//! | ThermodynamicYield | Monte Carlo (culture-conditioned) |
//! | NetworkResilience | Monte Carlo (culture-conditioned) |
//! | EconomicVelocity | SAP-turnover proxy (1 − hoard_ratio) |
//! | CivicParticipation | `consciousness.phi()` (legacy bridge) |
//! | StewardshipCare | `tend_balance` normalized + `ethics.virtue_care` |
//! | SemanticResonance | `mycel_score` |
//! | DomainCompetence | max of `skills` vector |

use serde::{Deserialize, Serialize};

use crate::stochastic::StochasticEngine;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// The 8 axes of civic identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SovereignDimension {
    EpistemicIntegrity,
    ThermodynamicYield,
    NetworkResilience,
    EconomicVelocity,
    CivicParticipation,
    StewardshipCare,
    SemanticResonance,
    DomainCompetence,
}

impl SovereignDimension {
    pub const ALL: [Self; 8] = [
        Self::EpistemicIntegrity,
        Self::ThermodynamicYield,
        Self::NetworkResilience,
        Self::EconomicVelocity,
        Self::CivicParticipation,
        Self::StewardshipCare,
        Self::SemanticResonance,
        Self::DomainCompetence,
    ];

    pub fn index(&self) -> usize {
        match self {
            Self::EpistemicIntegrity => 0,
            Self::ThermodynamicYield => 1,
            Self::NetworkResilience => 2,
            Self::EconomicVelocity => 3,
            Self::CivicParticipation => 4,
            Self::StewardshipCare => 5,
            Self::SemanticResonance => 6,
            Self::DomainCompetence => 7,
        }
    }
}

/// 8-dimensional sovereign civic profile. Each dimension ∈ [0, 1].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SovereignProfile {
    pub epistemic_integrity: f64,
    pub thermodynamic_yield: f64,
    pub network_resilience: f64,
    pub economic_velocity: f64,
    pub civic_participation: f64,
    pub stewardship_care: f64,
    pub semantic_resonance: f64,
    pub domain_competence: f64,
}

impl SovereignProfile {
    pub fn zero() -> Self {
        Self {
            epistemic_integrity: 0.0,
            thermodynamic_yield: 0.0,
            network_resilience: 0.0,
            economic_velocity: 0.0,
            civic_participation: 0.0,
            stewardship_care: 0.0,
            semantic_resonance: 0.0,
            domain_competence: 0.0,
        }
    }

    pub fn get(&self, dim: SovereignDimension) -> f64 {
        match dim {
            SovereignDimension::EpistemicIntegrity => self.epistemic_integrity,
            SovereignDimension::ThermodynamicYield => self.thermodynamic_yield,
            SovereignDimension::NetworkResilience => self.network_resilience,
            SovereignDimension::EconomicVelocity => self.economic_velocity,
            SovereignDimension::CivicParticipation => self.civic_participation,
            SovereignDimension::StewardshipCare => self.stewardship_care,
            SovereignDimension::SemanticResonance => self.semantic_resonance,
            SovereignDimension::DomainCompetence => self.domain_competence,
        }
    }

    pub fn set(&mut self, dim: SovereignDimension, value: f64) {
        let slot = match dim {
            SovereignDimension::EpistemicIntegrity => &mut self.epistemic_integrity,
            SovereignDimension::ThermodynamicYield => &mut self.thermodynamic_yield,
            SovereignDimension::NetworkResilience => &mut self.network_resilience,
            SovereignDimension::EconomicVelocity => &mut self.economic_velocity,
            SovereignDimension::CivicParticipation => &mut self.civic_participation,
            SovereignDimension::StewardshipCare => &mut self.stewardship_care,
            SovereignDimension::SemanticResonance => &mut self.semantic_resonance,
            SovereignDimension::DomainCompetence => &mut self.domain_competence,
        };
        *slot = value;
    }

    pub fn as_array(&self) -> [f64; 8] {
        [
            self.epistemic_integrity,
            self.thermodynamic_yield,
            self.network_resilience,
            self.economic_velocity,
            self.civic_participation,
            self.stewardship_care,
            self.semantic_resonance,
            self.domain_competence,
        ]
    }

    pub fn from_array(values: [f64; 8]) -> Self {
        Self {
            epistemic_integrity: values[0],
            thermodynamic_yield: values[1],
            network_resilience: values[2],
            economic_velocity: values[3],
            civic_participation: values[4],
            stewardship_care: values[5],
            semantic_resonance: values[6],
            domain_competence: values[7],
        }
    }

    fn sanitize(v: f64) -> f64 {
        if v.is_finite() {
            v.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Weighted combined score, clamped to [0, 1].
    pub fn combined_score(&self, weights: &DimensionWeights) -> f64 {
        let dims = self.as_array();
        let mut score = 0.0;
        for i in 0..8 {
            score += Self::sanitize(dims[i]) * weights.weights[i];
        }
        score.clamp(0.0, 1.0)
    }

    pub fn tier(&self, weights: &DimensionWeights) -> CivicTier {
        CivicTier::from_score(self.combined_score(weights))
    }

    pub fn meets_requirement(
        &self,
        requirement: &CivicRequirement,
        weights: &DimensionWeights,
    ) -> bool {
        if self.tier(weights) < requirement.min_tier {
            return false;
        }
        for &(dim, min_value) in &requirement.min_dimensions {
            if Self::sanitize(self.get(dim)) < min_value {
                return false;
            }
        }
        true
    }

    // -------------------- Monte Carlo initialization --------------------

    /// Sample a profile from culture-conditioned Beta-like distributions.
    ///
    /// `individualism` ∈ [0, 1] biases the distribution — individualist
    /// cultures sample higher EpistemicIntegrity and EconomicVelocity,
    /// collectivist cultures sample higher StewardshipCare and SemanticResonance.
    /// Each dimension is `clamp(Normal(μ(culture), 0.18), 0, 1)` — cheap,
    /// deterministic from the StochasticEngine, and produces realistic
    /// asymmetric profiles without the heavy machinery of true Beta sampling.
    pub fn sample(individualism: f64, rng: &mut StochasticEngine) -> Self {
        let ind = individualism.clamp(0.0, 1.0);
        let col = 1.0 - ind;
        let draw = |mean: f64, rng: &mut StochasticEngine| -> f64 {
            rng.next_gaussian(mean, 0.18).clamp(0.0, 1.0)
        };
        Self {
            epistemic_integrity: draw(0.30 + 0.20 * ind, rng),
            thermodynamic_yield: draw(0.30, rng),
            network_resilience: draw(0.30, rng),
            economic_velocity: draw(0.35 + 0.15 * ind, rng),
            civic_participation: draw(0.25, rng),
            stewardship_care: draw(0.25 + 0.20 * col, rng),
            semantic_resonance: draw(0.25 + 0.20 * col, rng),
            domain_competence: draw(0.30, rng),
        }
    }
}

impl Default for SovereignProfile {
    fn default() -> Self {
        Self::zero()
    }
}

// ---------------------------------------------------------------------------
// CivicTier
// ---------------------------------------------------------------------------

/// Progressive civic tier derived from the combined 8D score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CivicTier {
    Observer,
    Participant,
    Citizen,
    Steward,
    Guardian,
}

impl CivicTier {
    pub fn min_score(&self) -> f64 {
        match self {
            Self::Observer => 0.0,
            Self::Participant => 0.3,
            Self::Citizen => 0.4,
            Self::Steward => 0.6,
            Self::Guardian => 0.8,
        }
    }

    pub fn from_score(score: f64) -> Self {
        if score >= 0.8 {
            Self::Guardian
        } else if score >= 0.6 {
            Self::Steward
        } else if score >= 0.4 {
            Self::Citizen
        } else if score >= 0.3 {
            Self::Participant
        } else {
            Self::Observer
        }
    }

    /// Index (0-4) matching the legacy `tier_distribution` layout.
    pub fn index(&self) -> usize {
        match self {
            Self::Observer => 0,
            Self::Participant => 1,
            Self::Citizen => 2,
            Self::Steward => 3,
            Self::Guardian => 4,
        }
    }

    /// Vote weight in basis points (1 bp = 0.01%).
    pub fn vote_weight_bp(&self) -> u32 {
        match self {
            Self::Observer => 0,
            Self::Participant => 5_000,
            Self::Citizen => 7_500,
            Self::Steward => 10_000,
            Self::Guardian => 10_000,
        }
    }
}

// ---------------------------------------------------------------------------
// DimensionWeights
// ---------------------------------------------------------------------------

/// Community-configurable dimension weights. Must sum to 1.0 ± 0.01.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimensionWeights {
    pub weights: [f64; 8],
}

impl DimensionWeights {
    pub fn equal() -> Self {
        Self {
            weights: [0.125; 8],
        }
    }

    /// Default governance weights — slight emphasis on epistemic integrity
    /// and civic participation. Matches `crates/sovereign-profile`.
    pub fn governance() -> Self {
        Self {
            weights: [0.15, 0.10, 0.10, 0.12, 0.18, 0.13, 0.12, 0.10],
        }
    }

    pub fn energy_cooperative() -> Self {
        Self {
            weights: [0.10, 0.22, 0.18, 0.10, 0.10, 0.12, 0.10, 0.08],
        }
    }

    pub fn knowledge_commons() -> Self {
        Self {
            weights: [0.22, 0.06, 0.08, 0.08, 0.12, 0.10, 0.14, 0.20],
        }
    }

    pub fn care_community() -> Self {
        Self {
            weights: [0.08, 0.08, 0.08, 0.10, 0.14, 0.22, 0.20, 0.10],
        }
    }

    pub fn is_normalized(&self) -> bool {
        let sum: f64 = self.weights.iter().sum();
        (sum - 1.0).abs() < 0.01
    }

    pub fn normalize(&mut self) {
        let sum: f64 = self.weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
    }
}

impl Default for DimensionWeights {
    fn default() -> Self {
        Self::governance()
    }
}

// ---------------------------------------------------------------------------
// CivicRequirement
// ---------------------------------------------------------------------------

/// Governance requirement — minimum tier plus optional per-dimension minimums.
///
/// Passes when (1) derived tier ≥ `min_tier` AND (2) every listed dimension
/// meets its own threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CivicRequirement {
    pub min_tier: CivicTier,
    pub min_dimensions: Vec<(SovereignDimension, f64)>,
}

pub fn civic_requirement_basic() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Participant,
        min_dimensions: vec![],
    }
}

pub fn civic_requirement_proposal() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Participant,
        min_dimensions: vec![(SovereignDimension::EpistemicIntegrity, 0.25)],
    }
}

pub fn civic_requirement_voting() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Citizen,
        min_dimensions: vec![(SovereignDimension::EpistemicIntegrity, 0.25)],
    }
}

pub fn civic_requirement_constitutional() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Steward,
        min_dimensions: vec![
            (SovereignDimension::EpistemicIntegrity, 0.5),
            (SovereignDimension::CivicParticipation, 0.3),
        ],
    }
}

pub fn civic_requirement_guardian() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Guardian,
        min_dimensions: vec![
            (SovereignDimension::EpistemicIntegrity, 0.7),
            (SovereignDimension::CivicParticipation, 0.5),
        ],
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_profile_is_observer() {
        let p = SovereignProfile::zero();
        let w = DimensionWeights::governance();
        assert_eq!(p.tier(&w), CivicTier::Observer);
        assert_eq!(p.combined_score(&w), 0.0);
    }

    #[test]
    fn full_profile_is_guardian() {
        let p = SovereignProfile::from_array([1.0; 8]);
        let w = DimensionWeights::governance();
        assert_eq!(p.tier(&w), CivicTier::Guardian);
    }

    #[test]
    fn governance_weights_sum_to_one() {
        assert!(DimensionWeights::governance().is_normalized());
        assert!(DimensionWeights::energy_cooperative().is_normalized());
        assert!(DimensionWeights::knowledge_commons().is_normalized());
        assert!(DimensionWeights::care_community().is_normalized());
        assert!(DimensionWeights::equal().is_normalized());
    }

    #[test]
    fn sanitize_handles_nan_and_out_of_range() {
        let mut p = SovereignProfile::zero();
        p.epistemic_integrity = f64::NAN;
        p.civic_participation = 2.0;
        p.stewardship_care = -0.5;
        let w = DimensionWeights::equal();
        let score = p.combined_score(&w);
        // NaN → 0, 2.0 → 1.0, -0.5 → 0.0, rest 0 → score = 1/8 = 0.125
        assert!((score - 0.125).abs() < 1e-9);
    }

    #[test]
    fn asymmetric_profile_gated_by_dimension_min() {
        // High overall score but EpistemicIntegrity below voting minimum.
        let p = SovereignProfile {
            epistemic_integrity: 0.10,
            thermodynamic_yield: 1.0,
            network_resilience: 1.0,
            economic_velocity: 1.0,
            civic_participation: 1.0,
            stewardship_care: 1.0,
            semantic_resonance: 1.0,
            domain_competence: 1.0,
        };
        let w = DimensionWeights::governance();
        let voting = civic_requirement_voting();
        // Tier is Guardian by score but EI < 0.25 blocks voting.
        assert!(p.tier(&w) >= CivicTier::Citizen);
        assert!(!p.meets_requirement(&voting, &w));
    }

    #[test]
    fn different_weights_produce_different_tiers() {
        // Energy-cooperative profile: high thermo + network, low epistemic.
        let p = SovereignProfile {
            epistemic_integrity: 0.10,
            thermodynamic_yield: 0.85,
            network_resilience: 0.85,
            economic_velocity: 0.40,
            civic_participation: 0.40,
            stewardship_care: 0.40,
            semantic_resonance: 0.40,
            domain_competence: 0.40,
        };
        let energy = DimensionWeights::energy_cooperative();
        let knowledge = DimensionWeights::knowledge_commons();
        assert!(p.combined_score(&energy) > p.combined_score(&knowledge));
    }

    #[test]
    fn monte_carlo_sampling_bounded() {
        let mut rng = StochasticEngine::new(42);
        for _ in 0..100 {
            let p = SovereignProfile::sample(0.5, &mut rng);
            for v in p.as_array() {
                assert!((0.0..=1.0).contains(&v), "sampled dim out of range: {}", v);
            }
        }
    }

    #[test]
    fn individualism_biases_sampling() {
        let mut rng_i = StochasticEngine::new(7);
        let mut rng_c = StochasticEngine::new(7);
        let mut sum_ei_i = 0.0;
        let mut sum_sc_c = 0.0;
        for _ in 0..500 {
            sum_ei_i += SovereignProfile::sample(0.95, &mut rng_i).epistemic_integrity;
            sum_sc_c += SovereignProfile::sample(0.05, &mut rng_c).stewardship_care;
        }
        let mean_ei = sum_ei_i / 500.0;
        let mean_sc = sum_sc_c / 500.0;
        // Individualist mean EI ≈ 0.5, collectivist mean SC ≈ 0.45 — both > 0.35.
        assert!(mean_ei > 0.35, "individualist EI mean too low: {}", mean_ei);
        assert!(mean_sc > 0.35, "collectivist SC mean too low: {}", mean_sc);
    }
}
