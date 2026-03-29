// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NIMH Research Domain Criteria (RDoC) dimensional framework.
//!
//! RDoC replaces categorical diagnosis with 6 continuous dimensions.
//! Each dimension maps naturally to Symthaea's neuromodulator bath:
//!
//! | RDoC Domain         | Neuromod Target      | Direction            |
//! |---------------------|----------------------|----------------------|
//! | NegativeValence     | 5-HT, GABA           | Boost (anxiolytic)   |
//! | PositiveValence     | DA                    | Boost (motivation)   |
//! | CognitiveSystems    | ACh                   | Boost (attention)    |
//! | SocialProcesses     | Oxytocin              | Boost (bonding)      |
//! | ArousalRegulatory   | NE, Adenosine         | Modulate (calm/activate) |
//! | Sensorimotor        | Glutamate             | Modulate (excitatory)|
//!
//! Science: Insel et al. (2010), Cuthbert & Insel (2013), Sanislow et al. (2019).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

// ── RDoC Domains ───────────────────────────────────────────────────────────

/// NIMH RDoC domains — dimensional axes of psychopathology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RDocDomain {
    /// Threat, loss, frustrative nonreward (anxiety/depression spectrum)
    NegativeValence,
    /// Reward, motivation, habit (anhedonia/addiction spectrum)
    PositiveValence,
    /// Attention, perception, working memory, cognitive control
    CognitiveSystems,
    /// Affiliation, communication, self/other perception (social cognition)
    SocialProcesses,
    /// Arousal, circadian rhythms, sleep-wake (regulatory systems)
    ArousalRegulatory,
    /// Motor actions, agency, sensorimotor dynamics
    Sensorimotor,
}

impl RDocDomain {
    /// All domain variants for iteration.
    pub const ALL: [Self; 6] = [
        Self::NegativeValence,
        Self::PositiveValence,
        Self::CognitiveSystems,
        Self::SocialProcesses,
        Self::ArousalRegulatory,
        Self::Sensorimotor,
    ];

    /// Short display name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::NegativeValence => "NegativeValence",
            Self::PositiveValence => "PositiveValence",
            Self::CognitiveSystems => "CognitiveSystems",
            Self::SocialProcesses => "SocialProcesses",
            Self::ArousalRegulatory => "ArousalRegulatory",
            Self::Sensorimotor => "Sensorimotor",
        }
    }

    /// Primary neuromodulator target for this domain.
    pub fn primary_neuromodulator(&self) -> &'static str {
        match self {
            Self::NegativeValence => "serotonin",
            Self::PositiveValence => "dopamine",
            Self::CognitiveSystems => "acetylcholine",
            Self::SocialProcesses => "oxytocin",
            Self::ArousalRegulatory => "noradrenaline",
            Self::Sensorimotor => "glutamate",
        }
    }

    /// Secondary neuromodulator target (if any).
    pub fn secondary_neuromodulator(&self) -> Option<&'static str> {
        match self {
            Self::NegativeValence => Some("gaba"),
            Self::ArousalRegulatory => Some("adenosine"),
            _ => None,
        }
    }

    /// Whether higher values in this domain indicate dysfunction.
    ///
    /// NegativeValence: high = more threat reactivity (dysfunction).
    /// PositiveValence: low = anhedonia (dysfunction is at low end).
    pub fn high_is_dysfunction(&self) -> bool {
        matches!(self, Self::NegativeValence)
    }
}

// ── RDoC Profile ───────────────────────────────────────────────────────────

/// Dimensional RDoC profile: 6 continuous scores (0.0–1.0).
///
/// Unlike categorical diagnosis, RDoC profiles are continuous and can encode
/// subthreshold presentations, comorbidity patterns, and treatment targets
/// in a single representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RDocProfile {
    /// Dimensional scores per domain (0.0 = floor, 1.0 = ceiling).
    pub scores: HashMap<RDocDomain, f32>,
}

impl Default for RDocProfile {
    fn default() -> Self {
        let mut scores = HashMap::new();
        // Healthy baseline: low negative valence, moderate positive, moderate cognitive
        scores.insert(RDocDomain::NegativeValence, 0.2);
        scores.insert(RDocDomain::PositiveValence, 0.6);
        scores.insert(RDocDomain::CognitiveSystems, 0.5);
        scores.insert(RDocDomain::SocialProcesses, 0.5);
        scores.insert(RDocDomain::ArousalRegulatory, 0.4);
        scores.insert(RDocDomain::Sensorimotor, 0.5);
        Self { scores }
    }
}

impl RDocProfile {
    /// Create from explicit domain scores.
    pub fn new(scores: HashMap<RDocDomain, f32>) -> Self {
        let mut profile = Self { scores };
        profile.clamp_all();
        profile
    }

    /// Get score for a domain (defaults to 0.5 if missing).
    pub fn score(&self, domain: RDocDomain) -> f32 {
        *self.scores.get(&domain).unwrap_or(&0.5)
    }

    /// Set score for a domain.
    pub fn set_score(&mut self, domain: RDocDomain, value: f32) {
        self.scores.insert(domain, value.clamp(0.0, 1.0));
    }

    /// Encode as a ContinuousHV by generating a domain-seeded base vector
    /// per domain, scaling by score, and bundling.
    pub fn to_hv(&self) -> ContinuousHV {
        let mut result = ContinuousHV::zero(HDC_DIMENSION);
        for domain in RDocDomain::ALL {
            let score = self.score(domain);
            let seed = domain_seed(domain);
            let base = ContinuousHV::random(HDC_DIMENSION, seed);
            // Scale base vector by domain score and accumulate
            let scaled = base.scale(score);
            result = result.add(&scaled);
        }
        result.normalize()
    }

    /// Clinical severity: mean of dysfunction-indicating scores.
    ///
    /// Higher = more clinical concern. Weights NegativeValence positively
    /// and PositiveValence inversely (low positive = anhedonia).
    pub fn clinical_severity(&self) -> f32 {
        let neg_val = self.score(RDocDomain::NegativeValence);
        let pos_val = 1.0 - self.score(RDocDomain::PositiveValence); // inverted
        let cognitive = 1.0 - self.score(RDocDomain::CognitiveSystems); // low = impaired
        let social = 1.0 - self.score(RDocDomain::SocialProcesses); // low = impaired
        let arousal = (self.score(RDocDomain::ArousalRegulatory) - 0.5).abs() * 2.0; // extremes bad
        let sensorimotor = (self.score(RDocDomain::Sensorimotor) - 0.5).abs() * 2.0;

        (neg_val + pos_val + cognitive + social + arousal + sensorimotor) / 6.0
    }

    /// Similarity to another RDoC profile (cosine of encoded HVs).
    pub fn similarity(&self, other: &Self) -> f32 {
        self.to_hv().similarity(&other.to_hv())
    }

    /// Clamp all scores to [0.0, 1.0].
    fn clamp_all(&mut self) {
        for value in self.scores.values_mut() {
            *value = value.clamp(0.0, 1.0);
        }
    }
}

/// Deterministic seed for a domain base vector.
fn domain_seed(domain: RDocDomain) -> u64 {
    let label = format!("rdoc:{}", domain.name());
    let hash = blake3::hash(label.as_bytes());
    u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap())
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_profile_healthy_baseline() {
        let profile = RDocProfile::default();
        assert!(profile.score(RDocDomain::NegativeValence) < 0.3);
        assert!(profile.score(RDocDomain::PositiveValence) > 0.5);
    }

    #[test]
    fn test_clinical_severity_healthy_low() {
        let healthy = RDocProfile::default();
        assert!(
            healthy.clinical_severity() < 0.5,
            "healthy severity {:.3} should be < 0.5",
            healthy.clinical_severity()
        );
    }

    #[test]
    fn test_clinical_severity_depressed_high() {
        let mut depressed = RDocProfile::default();
        depressed.set_score(RDocDomain::NegativeValence, 0.9);
        depressed.set_score(RDocDomain::PositiveValence, 0.1);
        depressed.set_score(RDocDomain::CognitiveSystems, 0.2);
        assert!(
            depressed.clinical_severity() > 0.5,
            "depressed severity {:.3} should be > 0.5",
            depressed.clinical_severity()
        );
    }

    #[test]
    fn test_encoding_deterministic() {
        let profile = RDocProfile::default();
        let hv1 = profile.to_hv();
        let hv2 = profile.to_hv();
        assert!(hv1.similarity(&hv2) > 0.99);
    }

    #[test]
    fn test_similar_profiles_high_similarity() {
        let mut a = RDocProfile::default();
        let mut b = RDocProfile::default();
        // Slightly different
        a.set_score(RDocDomain::NegativeValence, 0.3);
        b.set_score(RDocDomain::NegativeValence, 0.35);
        let sim = a.similarity(&b);
        assert!(sim > 0.9, "similar profiles sim {:.3} should be > 0.9", sim);
    }

    #[test]
    fn test_dissimilar_profiles_low_similarity() {
        let healthy = RDocProfile::default();
        let mut severe = RDocProfile::default();
        severe.set_score(RDocDomain::NegativeValence, 1.0);
        severe.set_score(RDocDomain::PositiveValence, 0.0);
        severe.set_score(RDocDomain::CognitiveSystems, 0.0);
        severe.set_score(RDocDomain::SocialProcesses, 0.0);
        let sim = healthy.similarity(&severe);
        assert!(
            sim < 0.8,
            "dissimilar profiles sim {:.3} should be < 0.8",
            sim
        );
    }

    #[test]
    fn test_domain_neuromodulator_mapping() {
        assert_eq!(
            RDocDomain::NegativeValence.primary_neuromodulator(),
            "serotonin"
        );
        assert_eq!(
            RDocDomain::PositiveValence.primary_neuromodulator(),
            "dopamine"
        );
        assert_eq!(
            RDocDomain::CognitiveSystems.primary_neuromodulator(),
            "acetylcholine"
        );
        assert_eq!(
            RDocDomain::SocialProcesses.primary_neuromodulator(),
            "oxytocin"
        );
    }

    #[test]
    fn test_score_clamping() {
        let mut profile = RDocProfile::default();
        profile.set_score(RDocDomain::NegativeValence, 2.0);
        assert_eq!(profile.score(RDocDomain::NegativeValence), 1.0);
        profile.set_score(RDocDomain::NegativeValence, -1.0);
        assert_eq!(profile.score(RDocDomain::NegativeValence), 0.0);
    }

    #[test]
    fn test_all_domains_count() {
        assert_eq!(RDocDomain::ALL.len(), 6);
    }

    #[test]
    fn test_high_is_dysfunction() {
        assert!(RDocDomain::NegativeValence.high_is_dysfunction());
        assert!(!RDocDomain::PositiveValence.high_is_dysfunction());
        assert!(!RDocDomain::CognitiveSystems.high_is_dysfunction());
    }
}
