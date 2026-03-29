// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Individual agent model: demographics, skills, consciousness, mortality.

use crate::needs::PsychologicalNeeds;

use serde::{Deserialize, Serialize};

/// Biological sex for reproductive modeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BiologicalSex {
    Male,
    Female,
}

/// Life stage derived from age.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifeStage {
    /// 0-15 years
    Child,
    /// 15-25 years
    Youth,
    /// 25-65 years
    Adult,
    /// 65+ years
    Elder,
}

impl LifeStage {
    /// Determine life stage from age in months.
    pub fn from_age_months(age_months: u32) -> Self {
        let years = age_months / 12;
        match years {
            0..=14 => Self::Child,
            15..=24 => Self::Youth,
            25..=64 => Self::Adult,
            _ => Self::Elder,
        }
    }

    /// Can this agent contribute labor?
    pub fn can_work(&self) -> bool {
        matches!(self, Self::Youth | Self::Adult | Self::Elder)
    }

    /// Can this agent participate in governance votes?
    pub fn can_vote(&self) -> bool {
        matches!(self, Self::Adult | Self::Elder)
    }

    /// Can this agent reproduce? (Youth and Adult up to ~50yr handled separately.)
    pub fn can_reproduce(&self) -> bool {
        matches!(self, Self::Youth | Self::Adult)
    }
}

/// Skill sectors for economic contribution.
pub const SKILL_SECTORS: [&str; 8] = [
    "engineering",
    "agriculture",
    "medicine",
    "governance",
    "science",
    "education",
    "art_culture",
    "logistics",
];

/// 8-dimensional skill vector for agent capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillVector {
    pub engineering: f64,
    pub agriculture: f64,
    pub medicine: f64,
    pub governance: f64,
    pub science: f64,
    pub education: f64,
    pub art_culture: f64,
    pub logistics: f64,
}

impl SkillVector {
    /// Base skill vector (all 0.1).
    pub fn new() -> Self {
        Self {
            engineering: 0.1,
            agriculture: 0.1,
            medicine: 0.1,
            governance: 0.1,
            science: 0.1,
            education: 0.1,
            art_culture: 0.1,
            logistics: 0.1,
        }
    }

    /// Sum of all skills.
    pub fn total(&self) -> f64 {
        self.as_slice().iter().sum()
    }

    /// Name of the strongest skill sector.
    pub fn strongest(&self) -> &str {
        SKILL_SECTORS[self.strongest_index()]
    }

    /// Index of the strongest skill sector (0-7).
    pub fn strongest_index(&self) -> usize {
        let slice = self.as_slice();
        slice.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Learn: increase skill at `sector` index by `amount`, capped at 1.0.
    pub fn learn(&mut self, sector: usize, amount: f64) {
        let mut refs = self.as_mut_slice();
        if let Some(v) = refs.get_mut(sector) {
            **v = (**v + amount).min(1.0);
        }
    }

    /// Decay all skills by `rate`, with minimum 0.05.
    pub fn decay(&mut self, rate: f64) {
        for v in self.as_mut_slice().iter_mut() {
            **v = (**v - rate).max(0.05);
        }
    }

    pub fn as_slice(&self) -> [f64; 8] {
        [
            self.engineering,
            self.agriculture,
            self.medicine,
            self.governance,
            self.science,
            self.education,
            self.art_culture,
            self.logistics,
        ]
    }

    fn as_mut_slice(&mut self) -> [&mut f64; 8] {
        [
            &mut self.engineering,
            &mut self.agriculture,
            &mut self.medicine,
            &mut self.governance,
            &mut self.science,
            &mut self.education,
            &mut self.art_culture,
            &mut self.logistics,
        ]
    }
}

impl Default for SkillVector {
    fn default() -> Self {
        Self::new()
    }
}

/// Individual consciousness state (lightweight sim version).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessState {
    /// Overall consciousness level [0, 1].
    pub level: f64,
    /// Meta-awareness capacity [0, 1].
    pub meta_awareness: f64,
    /// Internal coherence [0, 1].
    pub coherence: f64,
    /// Care/compassion activation [0, 1].
    pub care_activation: f64,
    /// Alignment with Eight Harmonies [0, 1].
    pub harmonic_alignment: f64,
    /// Epistemic confidence calibration [0, 1].
    pub epistemic_confidence: f64,
}

impl ConsciousnessState {
    /// Default nascent consciousness.
    pub fn nascent() -> Self {
        Self {
            level: 0.1,
            meta_awareness: 0.05,
            coherence: 0.2,
            care_activation: 0.3,
            harmonic_alignment: 0.1,
            epistemic_confidence: 0.1,
        }
    }

    /// Weighted average representing individual Phi.
    pub fn phi(&self) -> f64 {
        0.25 * self.level
            + 0.20 * self.meta_awareness
            + 0.15 * self.coherence
            + 0.15 * self.care_activation
            + 0.15 * self.harmonic_alignment
            + 0.10 * self.epistemic_confidence
    }

    /// Consciousness tier (0-4) based on phi thresholds.
    /// 0: Observer (<0.2), 1: Participant (0.2-0.4), 2: Contributor (0.4-0.6),
    /// 3: Steward (0.6-0.8), 4: Guardian (>=0.8)
    pub fn tier(&self) -> u8 {
        let p = self.phi();
        if p < 0.2 {
            0
        } else if p < 0.4 {
            1
        } else if p < 0.6 {
            2
        } else if p < 0.8 {
            3
        } else {
            4
        }
    }
}

/// A single agent in the civilization simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CivAgent {
    pub id: u64,
    pub birth_tick: u32,
    pub death_tick: Option<u32>,
    pub sex: BiologicalSex,
    pub world_id: u32,
    pub health: f64,
    pub skills: SkillVector,
    pub education_level: f64,
    pub consciousness: ConsciousnessState,
    pub partner_id: Option<u64>,
    pub children_ids: Vec<u64>,
    pub is_immigrant: bool,
    /// Psychological needs state (allostatic load, social satiation, engagement).
    pub needs: PsychologicalNeeds,
    /// TEND balance earned through care work (teaching, mutual aid).
    pub tend_balance: f64,
    /// Parent agent IDs (mother, father). None for founding colonists.
    pub parent_ids: Option<(u64, u64)>,
    /// Faction membership. None if unaffiliated.
    pub faction_id: Option<u32>,
    /// Generation number (0 for founders, incremented for births).
    pub generation: u16,
    /// Intergenerational trauma level [0.0, 1.0].
    pub trauma_level: f64,
    /// Realism E: Cumulative radiation dose (Sieverts lifetime).
    /// Linear no-threshold model: 5% cancer risk per Sv (ICRP 103).
    /// Earth background: ~0.002 Sv/year. Europa: ~54 Sv/year unshielded.
    pub cumulative_dose_sv: f64,
}

impl CivAgent {
    /// Age in months at a given tick.
    ///
    /// Uses wrapping subtraction to handle agents born "before" tick 0
    /// (initial colonists whose birth_tick is set via `0u32.wrapping_sub(age)`).
    pub fn age_months(&self, current_tick: u32) -> u32 {
        current_tick.wrapping_sub(self.birth_tick)
    }

    /// Age in fractional years.
    pub fn age_years(&self, current_tick: u32) -> f64 {
        self.age_months(current_tick) as f64 / 12.0
    }

    /// Current life stage.
    pub fn life_stage(&self, current_tick: u32) -> LifeStage {
        LifeStage::from_age_months(self.age_months(current_tick))
    }

    /// Whether the agent is still alive.
    pub fn is_alive(&self) -> bool {
        self.death_tick.is_none()
    }

    /// Age-dependent fertility (monthly probability component).
    /// Peaks at 25-35, zero below 15 or above 50. Female only.
    pub fn fertility(&self, current_tick: u32) -> f64 {
        if self.sex == BiologicalSex::Male {
            // Males contribute to pair fertility but don't have an independent curve
            return 1.0;
        }
        let age = self.age_years(current_tick);
        if age < 15.0 || age > 50.0 {
            return 0.0;
        }
        // Bell curve peaking at 30
        let peak = 30.0;
        let sigma = 8.0;
        let base = 0.08; // max monthly probability component
        base * (-0.5 * ((age - peak) / sigma).powi(2)).exp()
    }

    /// Gompertz-Makeham monthly mortality rate, modified by health and tech era.
    ///
    /// M(x) = alpha * exp(beta * x) + lambda
    ///
    /// Parameters evolve with medical technology (lifespan research calibration):
    /// - `alpha_mult`: reduces initial mortality (senolytics, public health)
    /// - `beta_mult`: reduces aging rate itself (reprogramming, negligible senescence)
    /// - `lambda_mult`: reduces background mortality (medical infrastructure)
    ///
    /// Space health penalties:
    /// - Radiation: cumulative_dose_sv increases cancer risk (linear no-threshold, ICRP 103)
    /// - Low gravity: accelerates bone/cardiovascular aging
    /// - Isolation: amplifies background mortality for small populations
    ///
    /// Sources: Pyrkov et al. (2021) Nature Comms (resilience wall 120-150yr);
    /// NASA-STD-3001 (radiation limits); Frankham (1995) (Ne/N ratios).
    pub fn mortality_rate(&self, current_tick: u32) -> f64 {
        self.mortality_rate_with_modifiers(current_tick, 1.0, 1.0, 1.0)
    }

    /// Mortality rate with tech-era modifiers for Gompertz parameters.
    pub fn mortality_rate_with_modifiers(
        &self,
        current_tick: u32,
        alpha_mult: f64,
        beta_mult: f64,
        lambda_mult: f64,
    ) -> f64 {
        let age = self.age_years(current_tick);
        let alpha = 0.00003 * alpha_mult;
        let beta = 0.085 * beta_mult;
        let lambda = 0.0001 * lambda_mult;
        let base_annual = alpha * (beta * age).exp() + lambda;

        // Health modifier (existing)
        let health_modifier = 1.0 + 2.0 * (1.0 - self.health);

        // Radiation cancer risk: 5% excess risk per Sv (ICRP 103)
        let radiation_modifier = 1.0 + self.cumulative_dose_sv * 0.05;

        (base_annual / 12.0) * health_modifier * radiation_modifier
    }

    /// Effective labor output: skill total * health * life-stage factor * engagement.
    ///
    /// Engagement modulates productivity: disengaged agents (digital escapism) produce
    /// less. Floor at 0.5 ensures even disengaged agents contribute minimally.
    pub fn effective_labor(&self, current_tick: u32) -> f64 {
        let stage_factor = match self.life_stage(current_tick) {
            LifeStage::Child => 0.0,
            LifeStage::Youth => 0.6,
            LifeStage::Adult => 1.0,
            LifeStage::Elder => 0.5,
        };
        let engagement_factor = 0.5 + 0.5 * self.needs.engagement;
        self.skills.total() * self.health * stage_factor * engagement_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_agent(birth_tick: u32, sex: BiologicalSex) -> CivAgent {
        CivAgent {
            id: 1,
            birth_tick,
            death_tick: None,
            sex,
            world_id: 0,
            health: 1.0,
            skills: SkillVector::new(),
            education_level: 0.0,
            consciousness: ConsciousnessState::nascent(),
            partner_id: None,
            children_ids: vec![],
            is_immigrant: false,
            needs: PsychologicalNeeds::new(),
            tend_balance: 0.0,
            parent_ids: None,
            faction_id: None,
            generation: 0,
            trauma_level: 0.0,
                    cumulative_dose_sv: 0.0,
        }
    }

    #[test]
    fn test_age_calculation() {
        let a = make_agent(10, BiologicalSex::Female);
        assert_eq!(a.age_months(22), 12);
        assert!((a.age_years(22) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_life_stage_transitions() {
        assert_eq!(LifeStage::from_age_months(0), LifeStage::Child);
        assert_eq!(LifeStage::from_age_months(14 * 12), LifeStage::Child);
        assert_eq!(LifeStage::from_age_months(15 * 12), LifeStage::Youth);
        assert_eq!(LifeStage::from_age_months(25 * 12), LifeStage::Adult);
        assert_eq!(LifeStage::from_age_months(65 * 12), LifeStage::Elder);
    }

    #[test]
    fn test_fertility_curve_shape() {
        let a = make_agent(0, BiologicalSex::Female);
        // Zero for children
        assert_eq!(a.fertility(10 * 12), 0.0);
        // Zero above 50
        assert_eq!(a.fertility(51 * 12), 0.0);
        // Peak around 30
        let f25 = a.fertility(25 * 12);
        let f30 = a.fertility(30 * 12);
        let f40 = a.fertility(40 * 12);
        assert!(f30 > f25);
        assert!(f30 > f40);
    }

    #[test]
    fn test_mortality_increases_with_age() {
        let a = make_agent(0, BiologicalSex::Male);
        let m20 = a.mortality_rate(20 * 12);
        let m60 = a.mortality_rate(60 * 12);
        let m80 = a.mortality_rate(80 * 12);
        assert!(m60 > m20);
        assert!(m80 > m60);
    }

    #[test]
    fn test_skill_vector_learn_and_decay() {
        let mut s = SkillVector::new();
        s.learn(0, 0.5); // engineering
        assert!((s.engineering - 0.6).abs() < 0.01);
        s.learn(0, 0.8); // should cap at 1.0
        assert!((s.engineering - 1.0).abs() < 0.01);
        s.decay(0.2);
        assert!(s.engineering >= 0.05);
        assert!(s.agriculture >= 0.05);
    }

    #[test]
    fn test_strongest_skill() {
        let mut s = SkillVector::new();
        s.learn(2, 0.8); // medicine
        assert_eq!(s.strongest(), "medicine");
    }

    #[test]
    fn test_consciousness_tier_boundaries() {
        let mut c = ConsciousnessState::nascent();
        assert_eq!(c.tier(), 0); // nascent phi < 0.2

        // Push to tier 2
        c.level = 0.6;
        c.meta_awareness = 0.5;
        c.coherence = 0.5;
        c.care_activation = 0.5;
        c.harmonic_alignment = 0.5;
        c.epistemic_confidence = 0.5;
        let p = c.phi();
        assert!(p >= 0.4 && p < 0.6, "phi={p}");
        assert_eq!(c.tier(), 2);

        // Push to tier 4
        c.level = 1.0;
        c.meta_awareness = 1.0;
        c.coherence = 1.0;
        c.care_activation = 1.0;
        c.harmonic_alignment = 1.0;
        c.epistemic_confidence = 1.0;
        assert_eq!(c.tier(), 4);
    }

    #[test]
    fn test_effective_labor_child_is_zero() {
        let a = make_agent(0, BiologicalSex::Male);
        assert_eq!(a.effective_labor(5 * 12), 0.0);
    }
}
