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
        slice
            .iter()
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

/// An agent's ethical framework — how they weigh moral considerations.
///
/// Each value [0,1] represents emphasis on that ethical tradition. Values don't
/// need to sum to 1 — agents can be pluralistic (high on multiple frameworks).
/// Initialized from cultural profile + random variation, evolves through
/// education, trauma, care work, and aging.
///
/// # References
/// - Kant (1785) *Groundwork* — deontological duty
/// - Mill (1863) *Utilitarianism* — consequentialist welfare
/// - Aristotle *Nicomachean Ethics*, Gilligan (1982) *In a Different Voice* — virtue/care
/// - Tutu (1999) *No Future Without Forgiveness*, Metz (2007) — Ubuntu/relational
/// - Kohlberg (1981) *Essays on Moral Development* — stage theory (evolution over lifespan)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthicalOrientation {
    /// Deontological: duty, rules, rights. "What is my obligation?"
    pub deontological: f64,
    /// Consequentialist: outcomes, welfare, utility. "What produces the best result?"
    pub consequentialist: f64,
    /// Virtue/Care: character, relationships, compassion. "What would a good person do?"
    pub virtue_care: f64,
    /// Relational/Ubuntu: community, reciprocity, interconnection. "I am because we are."
    pub relational: f64,
}

impl Default for EthicalOrientation {
    fn default() -> Self {
        Self {
            deontological: 0.4,
            consequentialist: 0.4,
            virtue_care: 0.4,
            relational: 0.4,
        }
    }
}

impl EthicalOrientation {
    /// Initialize from cultural profile with random variation.
    /// Individualist cultures lean deontological/consequentialist.
    /// Collectivist cultures lean relational/virtue_care.
    pub fn from_culture(individualism: f64, rng: &mut crate::stochastic::StochasticEngine) -> Self {
        Self {
            deontological: (0.3 + individualism * 0.4 + rng.next_gaussian(0.0, 0.1))
                .clamp(0.05, 1.0),
            consequentialist: (0.3 + individualism * 0.3 + rng.next_gaussian(0.0, 0.1))
                .clamp(0.05, 1.0),
            virtue_care: (0.3 + (1.0 - individualism) * 0.3 + rng.next_gaussian(0.0, 0.1))
                .clamp(0.05, 1.0),
            relational: (0.3 + (1.0 - individualism) * 0.4 + rng.next_gaussian(0.0, 0.1))
                .clamp(0.05, 1.0),
        }
    }

    /// Dominant ethical framework (highest weight).
    pub fn dominant(&self) -> &'static str {
        let vals = [
            self.deontological,
            self.consequentialist,
            self.virtue_care,
            self.relational,
        ];
        let names = [
            "deontological",
            "consequentialist",
            "virtue_care",
            "relational",
        ];
        let max_idx = vals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        names[max_idx]
    }

    /// Sacred dimension: the agent's highest ethical commitment.
    /// This dimension resists modification by 80% — it is non-negotiable.
    /// Violating a sacred value triggers moral outrage (Tetlock 2003).
    /// Returns (dimension_index, value).
    pub fn sacred_dimension(&self) -> (usize, f64) {
        let vals = [
            self.deontological,
            self.consequentialist,
            self.virtue_care,
            self.relational,
        ];
        let (idx, &val) = vals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        (idx, val)
    }

    /// Apply a modification to a specific dimension, respecting sacred resistance.
    /// Sacred dimension resists 80% of the change. Others accept fully.
    pub fn modify_with_sacred_resistance(&mut self, dim: usize, delta: f64) {
        let (sacred_idx, _) = self.sacred_dimension();
        let effective = if dim == sacred_idx {
            delta * 0.2
        } else {
            delta
        };
        match dim {
            0 => self.deontological = (self.deontological + effective).clamp(0.05, 1.0),
            1 => self.consequentialist = (self.consequentialist + effective).clamp(0.05, 1.0),
            2 => self.virtue_care = (self.virtue_care + effective).clamp(0.05, 1.0),
            _ => self.relational = (self.relational + effective).clamp(0.05, 1.0),
        }
    }

    /// Ethical affinity for a sector (0..7).
    /// Sectors: 0=engineering, 1=agriculture, 2=medicine, 3=governance,
    ///          4=science, 5=education, 6=art_culture, 7=logistics
    ///
    /// Virtue/care agents are drawn to medicine (2) and education (5).
    /// Deontological agents are drawn to governance (3).
    /// Relational agents prefer art/culture (6) and agriculture (1).
    /// Consequentialist agents are sector-agnostic (outcome-focused).
    pub fn sector_affinity(&self, sector: usize) -> f64 {
        match sector {
            2 => self.virtue_care * 0.15,   // medicine
            5 => self.virtue_care * 0.12,   // education
            3 => self.deontological * 0.10, // governance (duty)
            6 => self.relational * 0.08,    // art/culture (community)
            1 => self.relational * 0.06,    // agriculture (sustenance)
            _ => 0.0,
        }
    }

    /// As a 4D vector for distance/similarity computations.
    pub fn as_vec(&self) -> [f64; 4] {
        [
            self.deontological,
            self.consequentialist,
            self.virtue_care,
            self.relational,
        ]
    }

    /// Inherit from two parents with noise and occasional rebellion.
    ///
    /// Base: midpoint of parents + Gaussian noise (Harris 1998).
    /// Rebellion: with probability 15%, child INVERTS one dimension
    /// (1.0 - midpoint), modeling Mannheim's generational theory.
    /// Baby boomers rejected parents' deontological conformity;
    /// their children reacted with relational communitarianism.
    /// This creates oscillating ethical dynamics across generations.
    pub fn inherit(
        parent_a: &EthicalOrientation,
        parent_b: &EthicalOrientation,
        rng: &mut crate::stochastic::StochasticEngine,
    ) -> Self {
        let mut child = Self {
            deontological: ((parent_a.deontological + parent_b.deontological) * 0.5
                + rng.next_gaussian(0.0, 0.08))
            .clamp(0.05, 1.0),
            consequentialist: ((parent_a.consequentialist + parent_b.consequentialist) * 0.5
                + rng.next_gaussian(0.0, 0.08))
            .clamp(0.05, 1.0),
            virtue_care: ((parent_a.virtue_care + parent_b.virtue_care) * 0.5
                + rng.next_gaussian(0.0, 0.08))
            .clamp(0.05, 1.0),
            relational: ((parent_a.relational + parent_b.relational) * 0.5
                + rng.next_gaussian(0.0, 0.08))
            .clamp(0.05, 1.0),
        };
        // Generational rebellion: 15% chance to invert ONE dimension.
        // The child defines themselves in opposition to their parents'
        // strongest ethical commitment — a universal pattern across cultures.
        if rng.next_f64() < 0.15 {
            let rebel_dim = (rng.next_u64() % 4) as usize;
            let vals = [
                child.deontological,
                child.consequentialist,
                child.virtue_care,
                child.relational,
            ];
            let inverted = (1.0 - vals[rebel_dim]).clamp(0.05, 1.0);
            match rebel_dim {
                0 => child.deontological = inverted,
                1 => child.consequentialist = inverted,
                2 => child.virtue_care = inverted,
                _ => child.relational = inverted,
            }
        }
        child
    }

    /// Revealed ethics: what the agent ACTUALLY does under stress.
    /// Under low stress, revealed ≈ stated. Under high stress, all agents
    /// drift toward consequentialism (survival calculus overrides principles).
    /// Gap between stated and revealed = moral hypocrisy (Batson 2008).
    ///
    /// This should be used for actual decision points (mortality, fertility,
    /// migration) instead of raw `self` when stress-sensitivity matters.
    pub fn revealed(&self, allostatic_load: f64) -> Self {
        let stress = allostatic_load.clamp(0.0, 1.0);
        // Below 0.3 stress: revealed = stated (principles hold)
        // Above 0.6 stress: strong consequentialist drift (survival mode)
        let drift = ((stress - 0.5) / 0.3).clamp(0.0, 1.0) * 0.10;
        Self {
            deontological: (self.deontological - drift * 0.1).max(0.05),
            consequentialist: (self.consequentialist + drift).min(1.0),
            virtue_care: (self.virtue_care - drift * 0.08).max(0.05),
            relational: (self.relational - drift * 0.06).max(0.05),
        }
    }

    /// Compute mean ethical orientation across a population of agents.
    pub fn mean_of(agents: &[CivAgent]) -> Self {
        let living: Vec<_> = agents.iter().filter(|a| a.is_alive()).collect();
        let n = living.len().max(1) as f64;
        Self {
            deontological: living.iter().map(|a| a.ethics.deontological).sum::<f64>() / n,
            consequentialist: living
                .iter()
                .map(|a| a.ethics.consequentialist)
                .sum::<f64>()
                / n,
            virtue_care: living.iter().map(|a| a.ethics.virtue_care).sum::<f64>() / n,
            relational: living.iter().map(|a| a.ethics.relational).sum::<f64>() / n,
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
    /// Red team: adversarial strategy (None = normal agent).
    #[serde(default)]
    pub adversarial: Option<crate::red_team::AdversarialStrategy>,
    /// Coordination science understanding [0.0, 1.0].
    /// Orthogonal to both education_level and consciousness.level.
    /// Measures literacy in game theory, systems thinking, reciprocity
    /// dynamics, theory of mind, and thermodynamic reasoning.
    /// An agent can be highly educated but defect in prisoner's dilemmas (low),
    /// or be compassionate but cooperate naively without strategic reasoning (low).
    #[serde(default)]
    pub coordination_understanding: f64,
    /// MYCEL soulbound reputation score [0.0, 1.0].
    /// Computed from Participation(40%) + Recognition(20%) + Quality(20%) + Longevity(20%).
    #[serde(default = "default_mycel")]
    pub mycel_score: f64,
    /// SAP transferable balance (demurrage-bearing currency).
    #[serde(default = "default_sap")]
    pub sap_balance: f64,
    /// Whether this agent is biological (true) or AI (false).
    /// AI agents are capped at Steward tier per Constitution Article XI.
    #[serde(default = "default_biological")]
    pub is_biological: bool,
    /// Active wounds (compartmental healing model).
    #[serde(default)]
    pub wounds: Vec<crate::wound_healing::WoundState>,
    /// Ethical orientation: how this agent weighs moral considerations.
    /// Influences governance voting, sector preference, faction joining, migration.
    #[serde(default)]
    pub ethics: EthicalOrientation,
    /// 8D Sovereign Profile for consciousness-gated governance.
    /// Production Mycelix uses this instead of scalar Phi. Initialized via
    /// Monte Carlo sampling at agent creation and refreshed from agent state
    /// each governance tick via `refresh_sovereign_profile`.
    #[serde(default)]
    pub sovereign_profile: crate::sovereign_profile::SovereignProfile,
    /// Restorative justice state — tracks violations, corrections, and
    /// effective civic tier degradation. Survey Gap 2 (Phase 2b).
    #[serde(default)]
    pub justice: crate::sub_passport::RestorativeJustice,
}

fn default_mycel() -> f64 {
    0.1
}
fn default_sap() -> f64 {
    100.0
}
fn default_biological() -> bool {
    true
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
        // Dead Loop #6 fix: Affect modulates labor output.
        // Joy boosts productivity, sadness suppresses it.
        // Clamped to [0.5, 1.2] to prevent runaway effects.
        let affect_factor =
            (0.7 + 0.3 * (self.needs.affect.joy - self.needs.affect.sadness * 0.5)).clamp(0.5, 1.2);
        self.skills.total() * self.health * stage_factor * engagement_factor * affect_factor
    }

    /// Refresh the 8D sovereign profile from current agent state.
    ///
    /// Maps observable state (skills, mycel, tend, consciousness, ethics)
    /// to the 6 dimensions that can be derived; leaves the other 2
    /// (ThermodynamicYield, NetworkResilience) at their sampled values —
    /// those require infrastructure-telemetry inputs not modeled in the sim.
    pub fn refresh_sovereign_profile(&mut self) {
        let max_skill = self
            .skills
            .as_slice()
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);

        // SAP hoard proxy: low balance = high velocity. SAP_EXEMPT_FLOOR = 200.
        let hoard = (self.sap_balance / 1_000.0).clamp(0.0, 1.0);

        self.sovereign_profile.epistemic_integrity = ((self.coordination_understanding * 0.6
            + self.education_level * 0.4)
            + self.consciousness.epistemic_confidence * 0.1)
            .clamp(0.0, 1.0);
        self.sovereign_profile.economic_velocity = (1.0 - hoard).clamp(0.0, 1.0);
        self.sovereign_profile.civic_participation = self.consciousness.phi().clamp(0.0, 1.0);
        self.sovereign_profile.stewardship_care =
            ((self.tend_balance / 40.0).clamp(0.0, 1.0) * 0.6 + self.ethics.virtue_care * 0.4)
                .clamp(0.0, 1.0);
        self.sovereign_profile.semantic_resonance = self.mycel_score.clamp(0.0, 1.0);
        self.sovereign_profile.domain_competence = max_skill.clamp(0.0, 1.0);
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
            adversarial: None,
            coordination_understanding: 0.0,
            mycel_score: 0.1,
            sap_balance: 100.0,
            is_biological: true,
            wounds: Vec::new(),
            ethics: EthicalOrientation::default(),
            sovereign_profile: crate::sovereign_profile::SovereignProfile::zero(),
            justice: crate::sub_passport::RestorativeJustice::new(),
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

    #[test]
    fn test_ethics_from_culture_bounds() {
        let mut rng = crate::stochastic::StochasticEngine::new(42);
        for individualism in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for _ in 0..100 {
                let e = EthicalOrientation::from_culture(individualism, &mut rng);
                assert!(
                    e.deontological >= 0.05 && e.deontological <= 1.0,
                    "deont={}",
                    e.deontological
                );
                assert!(
                    e.consequentialist >= 0.05 && e.consequentialist <= 1.0,
                    "conseq={}",
                    e.consequentialist
                );
                assert!(
                    e.virtue_care >= 0.05 && e.virtue_care <= 1.0,
                    "virtue={}",
                    e.virtue_care
                );
                assert!(
                    e.relational >= 0.05 && e.relational <= 1.0,
                    "relat={}",
                    e.relational
                );
            }
        }
    }

    #[test]
    fn test_ethics_inherit_bounds() {
        let mut rng = crate::stochastic::StochasticEngine::new(123);
        let extremes = [
            EthicalOrientation {
                deontological: 0.05,
                consequentialist: 0.05,
                virtue_care: 0.05,
                relational: 0.05,
            },
            EthicalOrientation {
                deontological: 1.0,
                consequentialist: 1.0,
                virtue_care: 1.0,
                relational: 1.0,
            },
            EthicalOrientation {
                deontological: 0.9,
                consequentialist: 0.1,
                virtue_care: 0.5,
                relational: 0.5,
            },
        ];
        for a in &extremes {
            for b in &extremes {
                for _ in 0..50 {
                    let child = EthicalOrientation::inherit(a, b, &mut rng);
                    assert!(child.deontological >= 0.05 && child.deontological <= 1.0);
                    assert!(child.consequentialist >= 0.05 && child.consequentialist <= 1.0);
                    assert!(child.virtue_care >= 0.05 && child.virtue_care <= 1.0);
                    assert!(child.relational >= 0.05 && child.relational <= 1.0);
                }
            }
        }
    }

    #[test]
    fn test_ethics_sector_affinity_non_negative() {
        let e = EthicalOrientation {
            deontological: 1.0,
            consequentialist: 1.0,
            virtue_care: 1.0,
            relational: 1.0,
        };
        for sector in 0..8 {
            assert!(
                e.sector_affinity(sector) >= 0.0,
                "sector {} has negative affinity",
                sector
            );
        }
    }

    #[test]
    fn test_ethics_dominant_correctness() {
        let deont = EthicalOrientation {
            deontological: 0.9,
            consequentialist: 0.1,
            virtue_care: 0.1,
            relational: 0.1,
        };
        assert_eq!(deont.dominant(), "deontological");
        let relat = EthicalOrientation {
            deontological: 0.1,
            consequentialist: 0.1,
            virtue_care: 0.1,
            relational: 0.9,
        };
        assert_eq!(relat.dominant(), "relational");
    }

    #[test]
    fn test_ethics_trauma_shift_stays_bounded() {
        let mut e = EthicalOrientation {
            deontological: 0.95,
            consequentialist: 0.08,
            virtue_care: 0.5,
            relational: 0.5,
        };
        // Simulate 100 severe trauma events
        for _ in 0..100 {
            let trauma_inc = 0.1;
            e.deontological = (e.deontological + trauma_inc * 0.01).min(1.0);
            e.consequentialist = (e.consequentialist - trauma_inc * 0.005).max(0.05);
        }
        assert!(e.deontological <= 1.0);
        assert!(e.consequentialist >= 0.05);
    }

    #[test]
    fn test_ethics_age_drift_stays_bounded() {
        let mut e = EthicalOrientation {
            deontological: 0.5,
            consequentialist: 0.5,
            virtue_care: 0.95,
            relational: 0.95,
        };
        // Simulate 600 ticks of elder drift (50 years)
        for _ in 0..600 {
            e.virtue_care = (e.virtue_care + 0.0001).min(1.0);
            e.relational = (e.relational + 0.0001).min(1.0);
        }
        assert!(e.virtue_care <= 1.0);
        assert!(e.relational <= 1.0);
    }
}
