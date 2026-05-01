// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Civic Dimensions — Derive 8D Governance Profile from Agent State
//!
//! Maps a `CivAgent`'s existing state (consciousness, skills, needs, TEND
//! balance, etc.) into the 8 dimensions used by the Sovereign scoring model
//! and the 4 dimensions used by the Canonical model.
//!
//! These derivations are the bridge between the simulation engine and the
//! pluggable governance scoring infrastructure.

use crate::agent::CivAgent;

/// The 8 sovereign civic dimensions derived from agent state.
#[derive(Debug, Clone, Copy)]
pub struct CivicDimensions {
    /// D0: Epistemic Integrity — truthfulness and knowledge contribution.
    pub epistemic_integrity: f64,
    /// D1: Thermodynamic Yield — physical energy/resource contribution.
    pub thermodynamic_yield: f64,
    /// D2: Network Resilience — infrastructure and uptime contribution.
    pub network_resilience: f64,
    /// D3: Economic Velocity — trade activity and demurrage compliance.
    pub economic_velocity: f64,
    /// D4: Civic Participation — voting, dispute resolution, governance.
    pub civic_participation: f64,
    /// D5: Stewardship Care — care labor, teaching, mutual aid.
    pub stewardship_care: f64,
    /// D6: Semantic Resonance — alignment with community consensus.
    pub semantic_resonance: f64,
    /// D7: Domain Competence — peer-verified specialization depth.
    pub domain_competence: f64,
}

impl CivicDimensions {
    /// Derive civic dimensions from a CivAgent's current state.
    ///
    /// Each dimension is computed from available agent fields, clamped
    /// to [0.0, 1.0].  These are intentionally approximate — the
    /// simulation doesn't model every aspect perfectly, but provides
    /// enough signal for governance model comparison.
    pub fn from_agent(agent: &CivAgent, current_tick: u32) -> Self {
        Self {
            epistemic_integrity: derive_epistemic_integrity(agent),
            thermodynamic_yield: derive_thermodynamic_yield(agent),
            network_resilience: derive_network_resilience(agent, current_tick),
            economic_velocity: derive_economic_velocity(agent),
            civic_participation: derive_civic_participation(agent),
            stewardship_care: derive_stewardship_care(agent),
            semantic_resonance: derive_semantic_resonance(agent),
            domain_competence: derive_domain_competence(agent),
        }
    }

    /// Return as a fixed-size array (for ScoringModel::compute_score).
    pub fn as_8d_array(&self) -> [f64; 8] {
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

    /// Collapse to canonical 4D profile (I/R/C/E).
    ///
    /// Maps 8D → 4D by averaging related dimensions:
    /// - Identity ← epistemic_integrity (verified knowledge = identity signal)
    /// - Reputation ← (economic_velocity + stewardship_care) / 2
    /// - Community ← (civic_participation + semantic_resonance) / 2
    /// - Engagement ← (thermodynamic_yield + domain_competence) / 2
    pub fn as_4d_array(&self) -> [f64; 4] {
        [
            self.epistemic_integrity,
            (self.economic_velocity + self.stewardship_care) / 2.0,
            (self.civic_participation + self.semantic_resonance) / 2.0,
            (self.thermodynamic_yield + self.domain_competence) / 2.0,
        ]
    }

    /// Collapse to minimal 3D profile (I/R/E).
    pub fn as_3d_array(&self) -> [f64; 3] {
        let four = self.as_4d_array();
        [four[0], four[1], four[3]] // Identity, Reputation, Engagement
    }
}

// ============================================================================
// Dimension derivation functions
// ============================================================================

/// D0: Epistemic Integrity — from consciousness epistemic_confidence
/// + education level + science skill.
fn derive_epistemic_integrity(agent: &CivAgent) -> f64 {
    let c = agent.consciousness.epistemic_confidence;
    let e = agent.education_level;
    let s = agent.skills.science;
    ((c * 0.4 + e * 0.3 + s * 0.3) as f64).clamp(0.0, 1.0)
}

/// D1: Thermodynamic Yield — from engineering skill + health.
/// Physically fit engineers contribute more to energy production.
fn derive_thermodynamic_yield(agent: &CivAgent) -> f64 {
    let eng = agent.skills.engineering;
    let h = agent.health;
    ((eng * 0.6 + h * 0.4) as f64).clamp(0.0, 1.0)
}

/// D2: Network Resilience — from logistics skill + coordination understanding.
/// Agents who understand systems keep infrastructure running.
fn derive_network_resilience(agent: &CivAgent, current_tick: u32) -> f64 {
    let log = agent.skills.logistics;
    let coord = agent.coordination_understanding;
    // Bonus for being alive longer (experienced operators are more reliable)
    let experience = (agent.age_years(current_tick) / 50.0).clamp(0.0, 1.0) * 0.2;
    ((log * 0.4 + coord * 0.4 + experience) as f64).clamp(0.0, 1.0)
}

/// D3: Economic Velocity — from TEND balance activity + low allostatic load.
/// Active traders with low stress contribute to economic health.
fn derive_economic_velocity(agent: &CivAgent) -> f64 {
    // TEND balance normalized (assume max useful balance ~100)
    let tend_norm = (agent.tend_balance / 100.0).clamp(0.0, 1.0);
    // Low stress = better economic participation
    let stress_factor = 1.0 - agent.needs.allostatic_load;
    ((tend_norm * 0.5 + stress_factor * 0.5) as f64).clamp(0.0, 1.0)
}

/// D4: Civic Participation — from governance skill + engagement + consciousness level.
fn derive_civic_participation(agent: &CivAgent) -> f64 {
    let gov = agent.skills.governance;
    let eng = agent.needs.engagement;
    let cl = agent.consciousness.level;
    ((gov * 0.3 + eng * 0.4 + cl * 0.3) as f64).clamp(0.0, 1.0)
}

/// D5: Stewardship Care — from care_activation + education skill + TEND.
/// Teaching, caregiving, and mutual aid.
fn derive_stewardship_care(agent: &CivAgent) -> f64 {
    let care = agent.consciousness.care_activation;
    let edu = agent.skills.education;
    let med = agent.skills.medicine;
    ((care * 0.4 + edu * 0.3 + med * 0.3) as f64).clamp(0.0, 1.0)
}

/// D6: Semantic Resonance — from harmonic_alignment + social_satiation.
/// Alignment with community values and active social engagement.
fn derive_semantic_resonance(agent: &CivAgent) -> f64 {
    let ha = agent.consciousness.harmonic_alignment;
    let social = agent.needs.social_satiation;
    let coherence = agent.consciousness.coherence;
    ((ha * 0.4 + social * 0.3 + coherence * 0.3) as f64).clamp(0.0, 1.0)
}

/// D7: Domain Competence — peak skill depth.
/// The highest single skill indicates domain expertise.
fn derive_domain_competence(agent: &CivAgent) -> f64 {
    agent
        .skills
        .as_slice()
        .iter()
        .cloned()
        .fold(0.0f64, f64::max)
        .clamp(0.0, 1.0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{BiologicalSex, ConsciousnessState, SkillVector};
    use crate::needs::PsychologicalNeeds;

    fn sample_agent() -> CivAgent {
        CivAgent {
            id: 1,
            birth_tick: 0,
            death_tick: None,
            sex: BiologicalSex::Female,
            world_id: 0,
            health: 0.8,
            skills: SkillVector {
                engineering: 0.6,
                agriculture: 0.3,
                medicine: 0.5,
                governance: 0.7,
                science: 0.8,
                education: 0.6,
                art_culture: 0.4,
                logistics: 0.5,
            },
            education_level: 0.7,
            consciousness: ConsciousnessState {
                level: 0.6,
                meta_awareness: 0.5,
                coherence: 0.7,
                care_activation: 0.8,
                harmonic_alignment: 0.6,
                epistemic_confidence: 0.7,
            },
            partner_id: None,
            children_ids: vec![],
            is_immigrant: false,
            needs: PsychologicalNeeds::new(),
            tend_balance: 50.0,
            parent_ids: None,
            faction_id: None,
            generation: 0,
            trauma_level: 0.1,
            cumulative_dose_sv: 0.01,
            adversarial: None,
            coordination_understanding: 0.5,
            mycel_score: 0.1,
            sap_balance: 100.0,
            is_biological: true,
            wounds: Vec::new(),
            ethics: crate::agent::EthicalOrientation::default(),
            sovereign_profile: crate::sovereign_profile::SovereignProfile::zero(),
            justice: crate::sub_passport::RestorativeJustice::new(),
        }
    }

    #[test]
    fn civic_dimensions_all_in_range() {
        let agent = sample_agent();
        let dims = CivicDimensions::from_agent(&agent, 120); // 10 years old

        let arr = dims.as_8d_array();
        for (i, &v) in arr.iter().enumerate() {
            assert!(
                v >= 0.0 && v <= 1.0,
                "Dimension {} = {} out of [0, 1] range",
                i,
                v
            );
        }
    }

    #[test]
    fn canonical_4d_in_range() {
        let agent = sample_agent();
        let dims = CivicDimensions::from_agent(&agent, 120);
        let arr = dims.as_4d_array();
        for (i, &v) in arr.iter().enumerate() {
            assert!(v >= 0.0 && v <= 1.0, "4D dim {} = {} out of range", i, v);
        }
    }

    #[test]
    fn minimal_3d_in_range() {
        let agent = sample_agent();
        let dims = CivicDimensions::from_agent(&agent, 120);
        let arr = dims.as_3d_array();
        assert_eq!(arr.len(), 3);
        for &v in &arr {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn high_skill_agent_has_high_domain_competence() {
        let mut agent = sample_agent();
        agent.skills.science = 0.95;
        let dims = CivicDimensions::from_agent(&agent, 120);
        assert!(
            dims.domain_competence >= 0.9,
            "Agent with science=0.95 should have high domain competence: {}",
            dims.domain_competence
        );
    }

    #[test]
    fn high_care_agent_has_high_stewardship() {
        let mut agent = sample_agent();
        agent.consciousness.care_activation = 0.95;
        agent.skills.education = 0.9;
        agent.skills.medicine = 0.9;
        let dims = CivicDimensions::from_agent(&agent, 120);
        assert!(
            dims.stewardship_care >= 0.8,
            "Caring educated agent should have high stewardship: {}",
            dims.stewardship_care
        );
    }

    #[test]
    fn stressed_agent_has_lower_economic_velocity() {
        let mut healthy = sample_agent();
        healthy.needs.allostatic_load = 0.1;

        let mut stressed = sample_agent();
        stressed.needs.allostatic_load = 0.9;

        let d_healthy = CivicDimensions::from_agent(&healthy, 120);
        let d_stressed = CivicDimensions::from_agent(&stressed, 120);

        assert!(
            d_healthy.economic_velocity > d_stressed.economic_velocity,
            "Stressed agent should have lower economic velocity: {} vs {}",
            d_healthy.economic_velocity,
            d_stressed.economic_velocity
        );
    }

    #[test]
    fn experienced_agent_has_higher_network_resilience() {
        let agent = sample_agent();
        let young = CivicDimensions::from_agent(&agent, 60); // 5 years
        let old = CivicDimensions::from_agent(&agent, 360); // 30 years

        assert!(
            old.network_resilience >= young.network_resilience,
            "Experienced agent should have >= network resilience: {} vs {}",
            old.network_resilience,
            young.network_resilience
        );
    }

    #[test]
    fn eight_d_and_four_d_are_consistent() {
        let agent = sample_agent();
        let dims = CivicDimensions::from_agent(&agent, 120);
        let eight = dims.as_8d_array();
        let four = dims.as_4d_array();

        // Identity should be epistemic_integrity
        assert_eq!(four[0], eight[0]);

        // Engagement should be mean of thermodynamic + domain
        let expected_e = (eight[1] + eight[7]) / 2.0;
        assert!((four[3] - expected_e).abs() < 1e-10);
    }
}
