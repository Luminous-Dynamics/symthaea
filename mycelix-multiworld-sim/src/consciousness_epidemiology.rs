// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consciousness epidemiology: modeling the spread and evolution of
//! consciousness states through populations using epidemic dynamics.
//!
//! Treats consciousness tiers as "infectious" states that spread via
//! social contact, mentorship, education, and cultural transmission.
//! Uses SIR-like compartmental models adapted for consciousness.
//!
//! Key insight: consciousness is not a disease, but its dynamics
//! resemble epidemic spread — contact with higher-consciousness
//! individuals raises one's own consciousness probability.
//!
//! References:
//! - Christakis, N. A. & Fowler, J. H. (2007). Spread of happiness in social networks.
//! - Centola, D. (2010). Spread of behavior in complex contagions.
//! - Dunbar, R. I. M. (2010). Social brain hypothesis.
//! - Tononi, G. (2004). IIT.
//! - Henrich, J. (2016). Cultural evolution.

use serde::{Deserialize, Serialize};

/// Dunbar's number: typical social network size.
const DUNBAR_NUMBER: usize = 150;

/// Default fraction of the population seeded as Awakening.
const DEFAULT_SEED_FRACTION: f64 = 0.03;

/// Maximum number of mentor assignments per agent.
const MAX_MENTORS: usize = 3;

/// Transition history ring buffer capacity.
const INCIDENCE_HISTORY_CAP: usize = 2000;

/// Number of tiers in the consciousness model.
const NUM_TIERS: usize = 5;

// ---------------------------------------------------------------------------
// CivicTier
// ---------------------------------------------------------------------------

/// Consciousness tier — epidemic compartment analogues.
///
/// Ordered from least to most conscious. The spread dynamics model
/// how agents transition upward (and sometimes regress downward).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CivicTier {
    /// No active consciousness practice; baseline.
    Dormant,
    /// Beginning to question; first exposure.
    Awakening,
    /// Sustained practice; stable awareness.
    Aware,
    /// Deep integration of consciousness into daily life.
    Integrated,
    /// Rare; sustained meta-awareness and service orientation.
    Transcendent,
}

impl CivicTier {
    /// Numeric index (0-4).
    pub fn as_index(&self) -> usize {
        match self {
            Self::Dormant => 0,
            Self::Awakening => 1,
            Self::Aware => 2,
            Self::Integrated => 3,
            Self::Transcendent => 4,
        }
    }

    /// Construct from index (clamped to 0..4).
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Self::Dormant,
            1 => Self::Awakening,
            2 => Self::Aware,
            3 => Self::Integrated,
            _ => Self::Transcendent,
        }
    }

    /// Phi range characteristic of this tier.
    pub fn phi_range(&self) -> (f64, f64) {
        match self {
            Self::Dormant => (0.0, 0.1),
            Self::Awakening => (0.1, 0.3),
            Self::Aware => (0.3, 0.5),
            Self::Integrated => (0.5, 0.7),
            Self::Transcendent => (0.7, 1.0),
        }
    }

    /// All tiers in ascending order.
    pub const ALL: [CivicTier; NUM_TIERS] = [
        Self::Dormant,
        Self::Awakening,
        Self::Aware,
        Self::Integrated,
        Self::Transcendent,
    ];
}

// ---------------------------------------------------------------------------
// EpiAgent
// ---------------------------------------------------------------------------

/// An agent in the consciousness epidemic model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpiAgent {
    pub id: usize,
    pub tier: CivicTier,
    /// Individual phi value within the tier's range.
    pub phi: f64,
    /// Openness to consciousness shift [0, 1].
    pub susceptibility: f64,
    /// How strongly this agent influences neighbours [0, 1].
    pub influence_radius: f64,
    /// Indices of social contacts (small-world network).
    pub contacts: Vec<usize>,
    /// Number of ticks spent at the current tier.
    pub tier_duration: u32,
    /// Indices of higher-tier contacts acting as mentors.
    pub mentors: Vec<usize>,
    /// High-water mark of consciousness tier ever reached.
    pub ever_reached_max: CivicTier,
}

// ---------------------------------------------------------------------------
// EpiState — aggregate compartment tracking
// ---------------------------------------------------------------------------

/// Aggregate epidemic state: compartment counts, R0, transition rates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpiState {
    /// Population count per tier.
    pub compartments: [usize; NUM_TIERS],
    /// Effective reproduction number per tier (how many agents one
    /// agent at tier *i* advances to tier *i+1* on average).
    pub r0_per_tier: [f64; NUM_TIERS],
    /// Tier-to-tier transition matrix: `transition_rates[from][to]`
    /// counts the number of transitions observed this tick.
    pub transition_rates: [[f64; NUM_TIERS]; NUM_TIERS],
    /// Fraction of total population that has ever reached Aware or above.
    pub attack_rate: f64,
    /// Tier penetration fraction required for collective stability.
    pub herd_threshold: f64,
    /// Per-tick compartment history (ring buffer).
    pub incidence_curve: Vec<[usize; NUM_TIERS]>,
}

impl EpiState {
    fn new(num_agents: usize) -> Self {
        let mut compartments = [0usize; NUM_TIERS];
        compartments[0] = num_agents;
        Self {
            compartments,
            r0_per_tier: [0.0; NUM_TIERS],
            transition_rates: [[0.0; NUM_TIERS]; NUM_TIERS],
            attack_rate: 0.0,
            herd_threshold: 0.6,
            incidence_curve: Vec::with_capacity(256),
        }
    }
}

// ---------------------------------------------------------------------------
// TransmissionModel
// ---------------------------------------------------------------------------

/// Parameters governing consciousness transmission dynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransmissionModel {
    /// Mean social contacts evaluated per tick (Dunbar-scaled).
    pub contact_rate: f64,
    /// Per-contact probability of tier advancement.
    pub transmission_probability: f64,
    /// Minimum higher-tier contacts required for advancement (Centola 2010).
    pub complex_contagion_threshold: usize,
    /// Multiplier for education / media amplification.
    pub cultural_transmission_weight: f64,
    /// Per-tick probability of regressing one tier (isolation, trauma).
    pub regression_rate: f64,
    /// Mentor contacts boost transmission by this factor.
    pub mentorship_multiplier: f64,
}

impl Default for TransmissionModel {
    fn default() -> Self {
        Self {
            contact_rate: 10.0,
            transmission_probability: 0.05,
            complex_contagion_threshold: 2,
            cultural_transmission_weight: 1.0,
            regression_rate: 0.01,
            mentorship_multiplier: 3.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Intervention
// ---------------------------------------------------------------------------

/// Interventions that modify epidemic dynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Intervention {
    /// Boost cultural transmission weight (education programs).
    Education { boost: f64 },
    /// Assign mentor pairs (higher-tier → lower-tier).
    Mentorship { pairs: usize },
    /// Temporary contact rate boost (media campaign).
    MediaCampaign { duration: u32, boost: f64 },
    /// External shock causing regression spike.
    CrisisEvent { regression_spike: f64 },
}

// ---------------------------------------------------------------------------
// EpidemicReport
// ---------------------------------------------------------------------------

/// Summary report produced after an epidemic simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpidemicReport {
    /// Effective R0 at the end of the run.
    pub r0_effective: f64,
    /// Tier with the highest population at the peak of advancement.
    pub peak_tier: CivicTier,
    /// Tick at which the highest mean phi was observed.
    pub peak_tick: u32,
    /// Fraction of population that ever reached Aware+.
    pub final_attack_rate: f64,
    /// Whether herd threshold was reached.
    pub herd_immunity_reached: bool,
    /// Mean phi across all agents at the end.
    pub mean_phi: f64,
    /// Gini coefficient of the phi distribution [0, 1].
    pub consciousness_gini: f64,
}

// ---------------------------------------------------------------------------
// ConsciousnessEpidemic — main engine
// ---------------------------------------------------------------------------

/// Main engine for consciousness epidemiology simulation.
///
/// Models the spread of consciousness tiers through a population
/// using SIR-like compartmental dynamics with complex contagion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessEpidemic {
    pub agents: Vec<EpiAgent>,
    pub model: TransmissionModel,
    pub state: EpiState,
    pub tick: u32,
    rng_seed: u64,
    rng_state: u64,
    /// Active media campaign: (remaining_ticks, boost).
    media_campaign_active: Option<(u32, f64)>,
    /// Tracks peak mean_phi and the tick it occurred at.
    peak_mean_phi: f64,
    peak_mean_phi_tick: u32,
}

impl ConsciousnessEpidemic {
    /// Create a new epidemic simulation.
    ///
    /// - `num_agents`: total population
    /// - `seed`: RNG seed for reproducibility
    /// - `model`: transmission parameters
    ///
    /// Initialises all agents as Dormant, generates a small-world
    /// social network (~Dunbar contacts), and seeds 1-5% as Awakening.
    pub fn new(num_agents: usize, seed: u64, model: TransmissionModel) -> Self {
        let mut epidemic = Self {
            agents: Vec::with_capacity(num_agents),
            model,
            state: EpiState::new(num_agents),
            tick: 0,
            rng_seed: seed,
            rng_state: seed,
            media_campaign_active: None,
            peak_mean_phi: 0.0,
            peak_mean_phi_tick: 0,
        };

        // Create agents
        for i in 0..num_agents {
            let susceptibility = 0.3 + 0.7 * epidemic.rand_f64();
            let influence_radius = 0.2 + 0.8 * epidemic.rand_f64();
            epidemic.agents.push(EpiAgent {
                id: i,
                tier: CivicTier::Dormant,
                phi: 0.05,
                susceptibility,
                influence_radius,
                contacts: Vec::new(),
                tier_duration: 0,
                mentors: Vec::new(),
                ever_reached_max: CivicTier::Dormant,
            });
        }

        // Build small-world network: each agent connects to ~min(DUNBAR, n-1) others
        let max_contacts = DUNBAR_NUMBER.min(num_agents.saturating_sub(1));
        // Pre-generate all contact indices to avoid borrow issues
        let mut all_contacts: Vec<Vec<usize>> = Vec::with_capacity(num_agents);
        for i in 0..num_agents {
            let mut contacts = Vec::with_capacity(max_contacts);
            for _ in 0..max_contacts {
                let j = epidemic.rand_usize(num_agents);
                if j != i && !contacts.contains(&j) {
                    contacts.push(j);
                }
            }
            all_contacts.push(contacts);
        }
        for (i, contacts) in all_contacts.into_iter().enumerate() {
            epidemic.agents[i].contacts = contacts;
        }

        // Seed initial Awakening agents (DEFAULT_SEED_FRACTION)
        let num_seeds = ((num_agents as f64 * DEFAULT_SEED_FRACTION).ceil() as usize).max(1);
        let seed_indices: Vec<usize> = (0..num_seeds)
            .map(|_| epidemic.rand_usize(num_agents))
            .collect();
        for idx in seed_indices {
            epidemic.agents[idx].tier = CivicTier::Awakening;
            epidemic.agents[idx].phi = 0.15;
            epidemic.agents[idx].ever_reached_max = CivicTier::Awakening;
        }

        // Recount compartments after seeding
        epidemic.recount_compartments();
        epidemic
    }

    /// Advance the simulation by one tick.
    pub fn tick(&mut self) {
        self.tick += 1;

        let n = self.agents.len();
        if n == 0 {
            return;
        }

        // Snapshot current tiers for read-during-write safety
        let tiers: Vec<CivicTier> = self.agents.iter().map(|a| a.tier).collect();

        // Effective contact rate (media campaign boost)
        let effective_contact_rate =
            if let Some((remaining, boost)) = &mut self.media_campaign_active {
                let rate = self.model.contact_rate + *boost;
                *remaining = remaining.saturating_sub(1);
                if *remaining == 0 {
                    self.media_campaign_active = None;
                }
                rate
            } else {
                self.model.contact_rate
            };

        // Reset per-tick transition counts
        self.state.transition_rates = [[0.0; NUM_TIERS]; NUM_TIERS];

        // --- Transmission phase ---
        let mut new_tiers = tiers.clone();

        for i in 0..n {
            let current_idx = tiers[i].as_index();

            // Can't advance beyond Transcendent
            if current_idx >= NUM_TIERS - 1 {
                self.agents[i].tier_duration += 1;
                continue;
            }

            let next_tier_idx = current_idx + 1;

            // Count contacts at tier >= next_tier_idx
            let num_contacts_to_check =
                (effective_contact_rate as usize).min(self.agents[i].contacts.len());
            let mut higher_contacts = 0usize;
            let mut mentor_contacts = 0usize;

            for c_pos in 0..num_contacts_to_check {
                let c_idx = self.agents[i].contacts[c_pos];
                if tiers[c_idx].as_index() >= next_tier_idx {
                    higher_contacts += 1;
                }
                if self.agents[i].mentors.contains(&c_idx)
                    && tiers[c_idx].as_index() >= next_tier_idx
                {
                    mentor_contacts += 1;
                }
            }

            // Complex contagion: need >= threshold higher-tier contacts
            if higher_contacts >= self.model.complex_contagion_threshold {
                let base_prob = self.model.transmission_probability
                    * self.agents[i].susceptibility
                    * self.model.cultural_transmission_weight;
                let mentor_boost = mentor_contacts as f64 * self.model.mentorship_multiplier;
                let total_prob = (base_prob * (1.0 + mentor_boost)).min(1.0);

                if self.rand_f64() < total_prob {
                    new_tiers[i] = CivicTier::from_index(next_tier_idx);
                    self.state.transition_rates[current_idx][next_tier_idx] += 1.0;
                }
            }

            // --- Regression phase ---
            // Isolated agents (few higher-tier contacts) may regress
            if current_idx > 0 && higher_contacts == 0 {
                if self.rand_f64() < self.model.regression_rate {
                    new_tiers[i] = CivicTier::from_index(current_idx - 1);
                    self.state.transition_rates[current_idx][current_idx - 1] += 1.0;
                }
            }
        }

        // Apply tier changes and update phi / duration
        for i in 0..n {
            let old_tier = self.agents[i].tier;
            let new_tier = new_tiers[i];

            if new_tier != old_tier {
                self.agents[i].tier = new_tier;
                self.agents[i].tier_duration = 0;

                if new_tier > self.agents[i].ever_reached_max {
                    self.agents[i].ever_reached_max = new_tier;
                }
            } else {
                self.agents[i].tier_duration += 1;
            }

            // Update phi based on tier range + small noise
            let (lo, hi) = self.agents[i].tier.phi_range();
            let target = lo + (hi - lo) * 0.5;
            // Smooth phi toward tier midpoint
            self.agents[i].phi += (target - self.agents[i].phi) * 0.1;
            // Small noise
            self.agents[i].phi += (self.rand_f64() - 0.5) * 0.01;
            self.agents[i].phi = self.agents[i].phi.clamp(0.0, 1.0);
        }

        // Update aggregate state
        self.recount_compartments();
        self.update_attack_rate();
        self.update_r0(effective_contact_rate);

        // Track peak
        let mean_phi = self.mean_phi();
        if mean_phi > self.peak_mean_phi {
            self.peak_mean_phi = mean_phi;
            self.peak_mean_phi_tick = self.tick;
        }

        // Record incidence
        if self.state.incidence_curve.len() >= INCIDENCE_HISTORY_CAP {
            self.state.incidence_curve.remove(0);
        }
        self.state.incidence_curve.push(self.state.compartments);
    }

    /// Run the simulation for `ticks` steps and return a report.
    pub fn run(&mut self, ticks: u32) -> EpidemicReport {
        for _ in 0..ticks {
            self.tick();
        }
        self.report()
    }

    /// Apply an intervention to the simulation.
    pub fn inject_intervention(&mut self, intervention: Intervention) {
        match intervention {
            Intervention::Education { boost } => {
                self.model.cultural_transmission_weight += boost;
            }
            Intervention::Mentorship { pairs } => {
                self.assign_mentors(pairs);
            }
            Intervention::MediaCampaign { duration, boost } => {
                self.media_campaign_active = Some((duration, boost));
            }
            Intervention::CrisisEvent { regression_spike } => {
                let n = self.agents.len();
                for i in 0..n {
                    if self.agents[i].tier.as_index() > 0 && self.rand_f64() < regression_spike {
                        let old_idx = self.agents[i].tier.as_index();
                        self.agents[i].tier = CivicTier::from_index(old_idx - 1);
                        self.agents[i].tier_duration = 0;
                    }
                }
                self.recount_compartments();
            }
        }
    }

    /// Compute effective R0: expected number of agents one Awakening agent
    /// advances to the next tier.
    pub fn compute_r0(&self) -> f64 {
        let mean_susceptibility =
            self.agents.iter().map(|a| a.susceptibility).sum::<f64>() / self.agents.len() as f64;
        self.model.contact_rate
            * self.model.transmission_probability
            * mean_susceptibility
            * self.model.cultural_transmission_weight
    }

    /// Compute Gini coefficient of the phi distribution.
    pub fn compute_gini(&self) -> f64 {
        let n = self.agents.len();
        if n == 0 {
            return 0.0;
        }
        let mut phis: Vec<f64> = self.agents.iter().map(|a| a.phi).collect();
        phis.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = phis.iter().sum::<f64>() / n as f64;
        if mean < 1e-15 {
            return 0.0;
        }

        let mut sum_abs_diff = 0.0;
        for i in 0..n {
            for j in 0..n {
                sum_abs_diff += (phis[i] - phis[j]).abs();
            }
        }
        sum_abs_diff / (2.0 * n as f64 * n as f64 * mean)
    }

    /// Mean phi across all agents.
    pub fn mean_phi(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        self.agents.iter().map(|a| a.phi).sum::<f64>() / self.agents.len() as f64
    }

    /// Generate a summary report at the current state.
    pub fn report(&self) -> EpidemicReport {
        let aware_plus =
            self.state.compartments[2] + self.state.compartments[3] + self.state.compartments[4];
        let n = self.agents.len();
        let herd_reached = n > 0 && (aware_plus as f64 / n as f64) >= self.state.herd_threshold;

        // Peak tier = the highest tier with nonzero population at peak
        let peak_tier = self
            .state
            .compartments
            .iter()
            .enumerate()
            .rev()
            .find(|(_, &count)| count > 0)
            .map(|(i, _)| CivicTier::from_index(i))
            .unwrap_or(CivicTier::Dormant);

        EpidemicReport {
            r0_effective: self.compute_r0(),
            peak_tier,
            peak_tick: self.peak_mean_phi_tick,
            final_attack_rate: self.state.attack_rate,
            herd_immunity_reached: herd_reached,
            mean_phi: self.mean_phi(),
            consciousness_gini: self.compute_gini(),
        }
    }

    // --- Internal helpers ---

    fn recount_compartments(&mut self) {
        self.state.compartments = [0; NUM_TIERS];
        for a in &self.agents {
            self.state.compartments[a.tier.as_index()] += 1;
        }
    }

    fn update_attack_rate(&mut self) {
        let n = self.agents.len();
        if n == 0 {
            self.state.attack_rate = 0.0;
            return;
        }
        let ever_aware = self
            .agents
            .iter()
            .filter(|a| a.ever_reached_max.as_index() >= CivicTier::Aware.as_index())
            .count();
        self.state.attack_rate = ever_aware as f64 / n as f64;
    }

    fn update_r0(&mut self, effective_contact_rate: f64) {
        let n = self.agents.len();
        if n == 0 {
            return;
        }
        let mean_susceptibility =
            self.agents.iter().map(|a| a.susceptibility).sum::<f64>() / n as f64;

        for tier_idx in 0..NUM_TIERS {
            let tier_pop = self.state.compartments[tier_idx] as f64;
            if tier_pop > 0.0 && tier_idx < NUM_TIERS - 1 {
                self.state.r0_per_tier[tier_idx] = effective_contact_rate
                    * self.model.transmission_probability
                    * mean_susceptibility
                    * self.model.cultural_transmission_weight
                    * (self.state.compartments[tier_idx + 1..]
                        .iter()
                        .sum::<usize>() as f64
                        / n as f64);
            } else {
                self.state.r0_per_tier[tier_idx] = 0.0;
            }
        }
    }

    fn assign_mentors(&mut self, pairs: usize) {
        let n = self.agents.len();
        if n < 2 {
            return;
        }

        // Collect agents by tier for mentor assignment
        let tiers: Vec<CivicTier> = self.agents.iter().map(|a| a.tier).collect();

        let mut assigned = 0;
        let mut attempts = 0;
        while assigned < pairs && attempts < pairs * 10 {
            attempts += 1;
            let mentee_idx = self.rand_usize(n);
            let mentor_idx = self.rand_usize(n);

            if mentee_idx == mentor_idx {
                continue;
            }
            if tiers[mentor_idx].as_index() <= tiers[mentee_idx].as_index() {
                continue;
            }
            if self.agents[mentee_idx].mentors.len() >= MAX_MENTORS {
                continue;
            }
            if self.agents[mentee_idx].mentors.contains(&mentor_idx) {
                continue;
            }

            self.agents[mentee_idx].mentors.push(mentor_idx);
            // Ensure mentor is in contacts
            if !self.agents[mentee_idx].contacts.contains(&mentor_idx) {
                self.agents[mentee_idx].contacts.push(mentor_idx);
            }
            assigned += 1;
        }
    }

    // --- Simple xoshiro128+ PRNG (deterministic, no external dep) ---

    fn rand_u64(&mut self) -> u64 {
        // SplitMix64
        self.rng_state = self.rng_state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn rand_f64(&mut self) -> f64 {
        (self.rand_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn rand_usize(&mut self, upper: usize) -> usize {
        if upper == 0 {
            return 0;
        }
        (self.rand_u64() as usize) % upper
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_model() -> TransmissionModel {
        TransmissionModel::default()
    }

    #[test]
    fn test_initial_population_mostly_dormant() {
        let epi = ConsciousnessEpidemic::new(200, 42, default_model());
        let dormant = epi.state.compartments[0];
        let total: usize = epi.state.compartments.iter().sum();
        assert_eq!(total, 200);
        // At least 90% should be Dormant after seeding ~3%
        assert!(
            dormant as f64 / total as f64 > 0.90,
            "Expected >90% Dormant, got {}/{}",
            dormant,
            total
        );
    }

    #[test]
    fn test_tick_advances_some_agents() {
        // Use high transmission to guarantee some advancement
        let model = TransmissionModel {
            contact_rate: 50.0,
            transmission_probability: 0.9,
            complex_contagion_threshold: 1, // easy threshold
            cultural_transmission_weight: 2.0,
            regression_rate: 0.0,
            mentorship_multiplier: 1.0,
        };
        let mut epi = ConsciousnessEpidemic::new(500, 123, model);

        // Seed more awakening agents for reliable contact
        for i in 0..50 {
            epi.agents[i].tier = CivicTier::Awakening;
            epi.agents[i].phi = 0.2;
            epi.agents[i].ever_reached_max = CivicTier::Awakening;
        }
        epi.recount_compartments();

        let initial_awakening = epi.state.compartments[1];

        for _ in 0..10 {
            epi.tick();
        }

        // Some Dormant agents should have become Awakening
        let current_dormant = epi.state.compartments[0];
        assert!(
            current_dormant < 500 - initial_awakening,
            "Some Dormant agents should have advanced; dormant={}",
            current_dormant
        );
    }

    #[test]
    fn test_r0_gt_1_leads_to_growth() {
        let model = TransmissionModel {
            contact_rate: 20.0,
            transmission_probability: 0.3,
            complex_contagion_threshold: 1,
            cultural_transmission_weight: 1.5,
            regression_rate: 0.0,
            mentorship_multiplier: 1.0,
        };
        let mut epi = ConsciousnessEpidemic::new(300, 7, model);

        // Seed 10% Awakening
        for i in 0..30 {
            epi.agents[i].tier = CivicTier::Awakening;
            epi.agents[i].phi = 0.2;
            epi.agents[i].ever_reached_max = CivicTier::Awakening;
        }
        epi.recount_compartments();

        assert!(epi.compute_r0() > 1.0, "R0 should be > 1");

        let initial_dormant = epi.state.compartments[0];
        epi.run(50);
        assert!(
            epi.state.compartments[0] < initial_dormant,
            "Dormant should decrease with R0 > 1"
        );
    }

    #[test]
    fn test_r0_lt_1_leads_to_decline() {
        let model = TransmissionModel {
            contact_rate: 1.0,
            transmission_probability: 0.01,
            complex_contagion_threshold: 5, // hard threshold
            cultural_transmission_weight: 0.5,
            regression_rate: 0.1, // high regression
            mentorship_multiplier: 1.0,
        };
        let mut epi = ConsciousnessEpidemic::new(200, 99, model);

        // Seed a few Awakening
        for i in 0..6 {
            epi.agents[i].tier = CivicTier::Awakening;
            epi.agents[i].phi = 0.2;
            epi.agents[i].ever_reached_max = CivicTier::Awakening;
        }
        epi.recount_compartments();

        let initial_awakening = epi.state.compartments[1];
        epi.run(50);

        // With high regression and low transmission, Awakening should decline
        assert!(
            epi.state.compartments[1] <= initial_awakening,
            "Awakening should not grow with R0 << 1; was {} now {}",
            initial_awakening,
            epi.state.compartments[1]
        );
    }

    #[test]
    fn test_complex_contagion_requires_multiple_contacts() {
        // High threshold, only 1 Awakening agent — should NOT spread
        let model = TransmissionModel {
            contact_rate: 50.0,
            transmission_probability: 1.0,
            complex_contagion_threshold: 10, // need 10 higher-tier contacts
            cultural_transmission_weight: 1.0,
            regression_rate: 0.0,
            mentorship_multiplier: 1.0,
        };
        let mut epi = ConsciousnessEpidemic::new(100, 55, model);

        // Only 1 Awakening agent — can't satisfy threshold=10
        for a in &mut epi.agents {
            a.tier = CivicTier::Dormant;
            a.ever_reached_max = CivicTier::Dormant;
        }
        epi.agents[0].tier = CivicTier::Awakening;
        epi.agents[0].ever_reached_max = CivicTier::Awakening;
        epi.recount_compartments();

        epi.run(20);

        // No Dormant agents should have advanced (can't reach threshold)
        let non_dormant: usize = epi.state.compartments[1..].iter().sum();
        // Allow <=1 because the original seeded agent might remain
        assert!(
            non_dormant <= 1,
            "With threshold=10 and 1 Awakening, spreading should be blocked; non_dormant={}",
            non_dormant
        );
    }

    #[test]
    fn test_mentorship_accelerates_transmission() {
        // Run multiple trials to confirm mentorship helps on average
        // (single trials can be noisy due to RNG divergence after intervention)
        let mut mentored_better_count = 0u32;
        let trials = 10;

        for trial in 0..trials {
            let model = TransmissionModel {
                contact_rate: 10.0,
                transmission_probability: 0.05,
                complex_contagion_threshold: 1,
                cultural_transmission_weight: 1.0,
                regression_rate: 0.0,
                mentorship_multiplier: 5.0,
            };

            let seed = 100 + trial as u64;
            let mut control = ConsciousnessEpidemic::new(200, seed, model.clone());
            let mut mentored = ConsciousnessEpidemic::new(200, seed, model);

            // Seed both identically
            for i in 0..20 {
                control.agents[i].tier = CivicTier::Awakening;
                control.agents[i].phi = 0.2;
                control.agents[i].ever_reached_max = CivicTier::Awakening;
                mentored.agents[i].tier = CivicTier::Awakening;
                mentored.agents[i].phi = 0.2;
                mentored.agents[i].ever_reached_max = CivicTier::Awakening;
            }
            control.recount_compartments();
            mentored.recount_compartments();

            mentored.inject_intervention(Intervention::Mentorship { pairs: 50 });

            control.run(30);
            mentored.run(30);

            if mentored.state.compartments[0] <= control.state.compartments[0] {
                mentored_better_count += 1;
            }
        }

        // Mentored group should win at least 60% of trials
        assert!(
            mentored_better_count >= 6,
            "Mentorship should help in majority of trials: won {}/{}",
            mentored_better_count,
            trials
        );
    }

    #[test]
    fn test_regression_for_isolated_agents() {
        let model = TransmissionModel {
            contact_rate: 0.0, // no contacts evaluated
            transmission_probability: 0.0,
            complex_contagion_threshold: 1,
            cultural_transmission_weight: 1.0,
            regression_rate: 0.5, // high regression
            mentorship_multiplier: 1.0,
        };
        let mut epi = ConsciousnessEpidemic::new(100, 77, model);

        // Start all agents at Awakening
        for a in &mut epi.agents {
            a.tier = CivicTier::Awakening;
            a.phi = 0.2;
            a.ever_reached_max = CivicTier::Awakening;
        }
        epi.recount_compartments();

        epi.run(20);

        // Some should have regressed to Dormant
        assert!(
            epi.state.compartments[0] > 0,
            "Isolated agents with high regression should regress to Dormant"
        );
    }

    #[test]
    fn test_gini_between_0_and_1() {
        let epi = ConsciousnessEpidemic::new(100, 42, default_model());
        let gini = epi.compute_gini();
        assert!(
            (0.0..=1.0).contains(&gini),
            "Gini should be in [0, 1], got {}",
            gini
        );
    }

    #[test]
    fn test_intervention_education_boosts_attack_rate() {
        let model = TransmissionModel {
            contact_rate: 15.0,
            transmission_probability: 0.1,
            complex_contagion_threshold: 1,
            cultural_transmission_weight: 1.0,
            regression_rate: 0.0,
            mentorship_multiplier: 1.0,
        };

        // Control group
        let mut control = ConsciousnessEpidemic::new(200, 42, model.clone());
        for i in 0..20 {
            control.agents[i].tier = CivicTier::Awakening;
            control.agents[i].phi = 0.2;
            control.agents[i].ever_reached_max = CivicTier::Awakening;
        }
        control.recount_compartments();
        control.run(50);

        // Education group
        let mut educated = ConsciousnessEpidemic::new(200, 42, model);
        for i in 0..20 {
            educated.agents[i].tier = CivicTier::Awakening;
            educated.agents[i].phi = 0.2;
            educated.agents[i].ever_reached_max = CivicTier::Awakening;
        }
        educated.recount_compartments();
        educated.inject_intervention(Intervention::Education { boost: 3.0 });
        educated.run(50);

        assert!(
            educated.state.attack_rate >= control.state.attack_rate,
            "Education should boost attack rate: educated={:.3} vs control={:.3}",
            educated.state.attack_rate,
            control.state.attack_rate
        );
    }

    #[test]
    fn test_crisis_event_causes_regression() {
        let model = TransmissionModel {
            contact_rate: 10.0,
            transmission_probability: 0.3,
            complex_contagion_threshold: 1,
            cultural_transmission_weight: 1.0,
            regression_rate: 0.0,
            mentorship_multiplier: 1.0,
        };
        let mut epi = ConsciousnessEpidemic::new(200, 42, model);

        // Start most agents at Awakening
        for a in &mut epi.agents {
            a.tier = CivicTier::Awakening;
            a.phi = 0.2;
            a.ever_reached_max = CivicTier::Awakening;
        }
        epi.recount_compartments();
        let pre_dormant = epi.state.compartments[0];

        epi.inject_intervention(Intervention::CrisisEvent {
            regression_spike: 0.8,
        });

        assert!(
            epi.state.compartments[0] > pre_dormant,
            "Crisis should cause regression: pre={} post={}",
            pre_dormant,
            epi.state.compartments[0]
        );
    }

    #[test]
    fn test_epidemic_report_captures_peak() {
        let model = TransmissionModel {
            contact_rate: 20.0,
            transmission_probability: 0.2,
            complex_contagion_threshold: 1,
            cultural_transmission_weight: 1.5,
            regression_rate: 0.0,
            mentorship_multiplier: 1.0,
        };
        let mut epi = ConsciousnessEpidemic::new(200, 42, model);

        // Seed generously
        for i in 0..40 {
            epi.agents[i].tier = CivicTier::Awakening;
            epi.agents[i].phi = 0.2;
            epi.agents[i].ever_reached_max = CivicTier::Awakening;
        }
        epi.recount_compartments();

        let report = epi.run(100);

        assert!(report.peak_tick > 0, "Peak tick should be > 0");
        assert!(report.mean_phi > 0.0, "Mean phi should be > 0");
        assert!(
            report.consciousness_gini >= 0.0 && report.consciousness_gini <= 1.0,
            "Gini should be in [0, 1]"
        );
        assert!(
            report.final_attack_rate >= 0.0 && report.final_attack_rate <= 1.0,
            "Attack rate should be in [0, 1]"
        );
    }

    #[test]
    fn test_full_100_tick_run_produces_valid_curve() {
        let mut epi = ConsciousnessEpidemic::new(150, 42, default_model());

        // Seed some Awakening
        for i in 0..10 {
            epi.agents[i].tier = CivicTier::Awakening;
            epi.agents[i].phi = 0.2;
            epi.agents[i].ever_reached_max = CivicTier::Awakening;
        }
        epi.recount_compartments();

        let report = epi.run(100);

        // Incidence curve should have 100 entries
        assert_eq!(
            epi.state.incidence_curve.len(),
            100,
            "Should have 100 incidence records"
        );

        // Each incidence entry should sum to total population
        for (tick_idx, entry) in epi.state.incidence_curve.iter().enumerate() {
            let total: usize = entry.iter().sum();
            assert_eq!(
                total, 150,
                "Tick {} compartment sum should be 150, got {}",
                tick_idx, total
            );
        }

        // Report should have valid R0
        assert!(report.r0_effective >= 0.0, "R0 should be non-negative");
    }

    #[test]
    fn test_tier_phi_ranges_are_contiguous() {
        let mut prev_hi = 0.0;
        for tier in &CivicTier::ALL {
            let (lo, hi) = tier.phi_range();
            assert!(
                (lo - prev_hi).abs() < 1e-10,
                "Tier {:?} lo={} should equal previous hi={}",
                tier,
                lo,
                prev_hi
            );
            assert!(hi > lo, "Tier {:?} range should be non-empty", tier);
            prev_hi = hi;
        }
        assert!((prev_hi - 1.0).abs() < 1e-10, "Last tier should end at 1.0");
    }

    #[test]
    fn test_media_campaign_temporary_boost() {
        let model = TransmissionModel {
            contact_rate: 5.0,
            transmission_probability: 0.1,
            complex_contagion_threshold: 1,
            cultural_transmission_weight: 1.0,
            regression_rate: 0.0,
            mentorship_multiplier: 1.0,
        };
        let mut epi = ConsciousnessEpidemic::new(100, 42, model);

        epi.inject_intervention(Intervention::MediaCampaign {
            duration: 3,
            boost: 20.0,
        });
        assert!(epi.media_campaign_active.is_some());

        // Run 3 ticks — campaign should expire
        for _ in 0..3 {
            epi.tick();
        }
        assert!(
            epi.media_campaign_active.is_none(),
            "Media campaign should expire after duration"
        );
    }
}
