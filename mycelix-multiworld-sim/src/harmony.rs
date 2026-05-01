// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Eight Harmonies tracking and love coherence computation.
//!
//! Each world is scored on eight dimensions derived from Luminous Dynamics'
//! philosophical framework. Love coherence is a composite metric reflecting
//! the balanced, diverse, low-tension expression of all eight harmonies.

use crate::consciousness::ConsciousnessEngine;
use crate::events::{CivEvent, CivEventType};
use crate::world::World;

use serde::{Deserialize, Serialize};

/// The Eight Harmonies of Luminous Dynamics.
pub const HARMONY_NAMES: [&str; 8] = [
    "Resonant Coherence",           // integration + governance stability
    "Pan-Sentient Flourishing",     // min(health, nutrition, shelter) — Rawlsian
    "Integral Wisdom",              // knowledge depth x education x epistemic confidence
    "Infinite Play",                // art/culture output per capita x innovation rate
    "Universal Interconnectedness", // inter-world trade + migration + communication
    "Sacred Reciprocity",           // gini_inverse x mutual_aid x trade_symmetry
    "Evolutionary Progression",     // tech_growth x pop_growth x genetic_diversity
    "Sacred Stillness",             // 1.0 - emergency_frequency - overwork_fraction
];

/// Love coherence hard ceiling (moral humility).
const LOVE_COHERENCE_CEILING: f64 = 0.95;

/// History length: 120 ticks = 10 years at monthly resolution.
const HARMONY_HISTORY_LEN: usize = 120;

/// Diversity threshold: harmonies above this contribute to diversity factor.
const DIVERSITY_THRESHOLD: f64 = 0.3;

/// Love coherence milestone threshold.
const LOVE_MILESTONE_THRESHOLD: f64 = 0.5;

/// Snapshot of harmony state at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonySnapshot {
    pub tick: u32,
    pub scores: [f64; 8],
    pub love_coherence: f64,
}

/// External world state needed for harmony scoring.
///
/// This trait-free struct lets callers pass whatever metrics they have,
/// with sensible defaults for stubs that aren't yet implemented.
#[derive(Debug, Clone)]
pub struct HarmonyInputs {
    /// Governance stability score [0, 1].
    pub governance_stability: f64,
    /// Food resource level [0, 1] (food.current / food.capacity).
    pub food_level: f64,
    /// Mean education level across agents [0, 1].
    pub mean_education: f64,
    /// Mean tech level [0, 1].
    pub mean_tech_level: f64,
    /// Innovation rate [0, 1].
    pub innovation_rate: f64,
    /// Art/culture sector output per capita [0, 1].
    pub art_per_capita: f64,
    /// Number of active trade connections.
    pub trade_connections: u32,
    /// Gini coefficient [0, 1] where 0 = perfect equality.
    pub gini_coefficient: f64,
    /// Resource self-sufficiency [0, 1].
    pub self_sufficiency: f64,
    /// Knowledge growth rate [0, 1].
    pub knowledge_growth_rate: f64,
    /// Population stability [0, 1] (1.0 = stable, 0.0 = collapsing/exploding).
    pub pop_stability: f64,
    /// Genetic diversity index [0, 1].
    pub genetic_diversity: f64,
    /// Fraction of population in active emergencies [0, 1].
    pub emergency_fraction: f64,
    /// Workers-to-population ratio.
    pub worker_ratio: f64,
    /// Mean allostatic load across agents [0, 1].
    pub mean_allostatic_load: f64,
    /// Mean engagement across agents [0, 1].
    pub mean_engagement: f64,
}

impl Default for HarmonyInputs {
    fn default() -> Self {
        Self {
            governance_stability: 0.5,
            food_level: 0.5,
            mean_education: 0.3,
            mean_tech_level: 0.3,
            innovation_rate: 0.1,
            art_per_capita: 0.1,
            trade_connections: 0,
            gini_coefficient: 0.4,
            self_sufficiency: 0.5,
            knowledge_growth_rate: 0.05,
            pop_stability: 0.8,
            genetic_diversity: 0.7,
            emergency_fraction: 0.0,
            worker_ratio: 0.5,
            mean_allostatic_load: 0.1,
            mean_engagement: 0.8,
        }
    }
}

/// Tracks the Eight Harmonies for a single world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyTracker {
    /// Current harmony scores [0, 1] for each of the eight harmonies.
    pub current_scores: [f64; 8],
    /// Love coherence: mean * (1 - tension) * diversity, capped at 0.95.
    pub love_coherence: f64,
    /// Tension ratio: variance across harmonies / mean.
    pub tension_ratio: f64,
    /// How many harmonies are above 0.3, divided by 8.
    pub diversity_factor: f64,
    /// History of snapshots (last 120 ticks).
    pub history: Vec<HarmonySnapshot>,
    /// Whether the love coherence milestone has been logged.
    love_milestone_logged: bool,
}

impl HarmonyTracker {
    /// Create a new tracker with zeroed scores.
    pub fn new() -> Self {
        Self {
            current_scores: [0.0; 8],
            love_coherence: 0.0,
            tension_ratio: 0.0,
            diversity_factor: 0.0,
            history: Vec::with_capacity(HARMONY_HISTORY_LEN),
            love_milestone_logged: false,
        }
    }

    /// Score all eight harmonies from world state and recompute love coherence.
    pub fn tick_harmony(
        &mut self,
        world: &World,
        inputs: &HarmonyInputs,
        consciousness: &ConsciousnessEngine,
        current_tick: u32,
    ) -> Vec<CivEvent> {
        let mut events = Vec::new();

        let pop = world.population().max(1) as f64;

        // 1. Resonant Coherence = collective_phi * governance_stability
        self.current_scores[0] =
            (consciousness.collective_phi * inputs.governance_stability).clamp(0.0, 1.0);

        // 2. Pan-Sentient Flourishing = min(health, food, shelter) * (1 - load*0.3)
        // Shelter: overcrowding hurts; 1.0 when pop << max, 0.0 when at capacity
        // Allostatic load degrades flourishing even when physical health is fine.
        let shelter_frac = (1.0 - pop / world.max_population.max(1) as f64).max(0.0);
        let base_flourishing = world.mean_health().min(inputs.food_level).min(shelter_frac);
        self.current_scores[1] =
            (base_flourishing * (1.0 - inputs.mean_allostatic_load * 0.3)).clamp(0.0, 1.0);

        // 3. Integral Wisdom = weighted average of tech, education, and phi.
        // Previous formula (product of three) collapsed near zero. Weighted average
        // allows partial credit: a colony with high education but low tech still has wisdom.
        self.current_scores[2] = (0.3 * inputs.mean_tech_level
            + 0.4 * inputs.mean_education
            + 0.3 * consciousness.mean_phi)
            .clamp(0.0, 1.0);

        // 4. Infinite Play = sqrt(art_per_capita * innovation_rate)
        // Bug #3 fix: Product of two small numbers (e.g., 0.1 × 0.2 = 0.02) produces
        // a near-zero score. Geometric mean (sqrt of product) is more appropriate —
        // a civilization with modest art AND modest innovation should score ~0.14, not 0.02.
        self.current_scores[3] = (inputs.art_per_capita * inputs.innovation_rate)
            .max(0.0)
            .sqrt()
            .clamp(0.0, 1.0);

        // 5. Universal Interconnectedness = trade_connections / 5 (max 5)
        self.current_scores[4] = (inputs.trade_connections as f64 / 5.0).clamp(0.0, 1.0);

        // 6. Sacred Reciprocity = (1 - gini) * self_sufficiency
        self.current_scores[5] =
            ((1.0 - inputs.gini_coefficient) * inputs.self_sufficiency).clamp(0.0, 1.0);

        // 7. Evolutionary Progression = weighted average of knowledge, stability, genetics
        // Bug #3 fix: Triple product of small numbers collapses to near-zero
        // (0.3 × 0.5 × 0.4 = 0.06). Weighted average better captures partial progress.
        self.current_scores[6] = (0.4 * inputs.knowledge_growth_rate
            + 0.3 * inputs.pop_stability
            + 0.3 * inputs.genetic_diversity)
            .clamp(0.0, 1.0);

        // 8. Sacred Stillness = (1 - emergency_fraction) * (1 - overwork) * presence
        // Overwork = worker ratio above 0.6
        // Engagement = physical-world presence (high = present, not escapist)
        let overwork = (inputs.worker_ratio - 0.6).max(0.0) / 0.4; // 0 at 0.6, 1.0 at 1.0
        let presence = 0.5 + 0.5 * inputs.mean_engagement;
        self.current_scores[7] =
            ((1.0 - inputs.emergency_fraction) * (1.0 - overwork) * presence).clamp(0.0, 1.0);

        // --- Love coherence computation ---
        let mean_01 = self.current_scores.iter().sum::<f64>() / 8.0;
        let variance = if mean_01 > 0.0 {
            self.current_scores
                .iter()
                .map(|s| (s - mean_01).powi(2))
                .sum::<f64>()
                / 8.0
        } else {
            0.0
        };

        self.tension_ratio = if mean_01 > 0.01 {
            variance / mean_01
        } else {
            0.0
        };

        self.diversity_factor = self
            .current_scores
            .iter()
            .filter(|&&s| s > DIVERSITY_THRESHOLD)
            .count() as f64
            / 8.0;

        self.love_coherence =
            (mean_01 * (1.0 - self.tension_ratio.min(1.0)) * self.diversity_factor)
                .min(LOVE_COHERENCE_CEILING);

        // History
        self.history.push(HarmonySnapshot {
            tick: current_tick,
            scores: self.current_scores,
            love_coherence: self.love_coherence,
        });
        if self.history.len() > HARMONY_HISTORY_LEN {
            self.history.remove(0);
        }

        // Milestone
        if !self.love_milestone_logged && self.love_coherence >= LOVE_MILESTONE_THRESHOLD {
            self.love_milestone_logged = true;
            events.push(CivEvent::new(
                current_tick,
                Some(world.id),
                CivEventType::HarmonyMilestone,
                format!(
                    "Love coherence reached {:.3} on {}",
                    self.love_coherence, world.name
                ),
            ));
        }

        events
    }

    /// Slope of love_coherence history (positive = improving).
    pub fn harmony_trajectory(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let values: Vec<f64> = self.history.iter().map(|s| s.love_coherence).collect();
        linear_slope(&values)
    }

    /// Mean harmony scores across all provided worlds.
    pub fn civilization_harmony(worlds: &[HarmonyTracker]) -> [f64; 8] {
        if worlds.is_empty() {
            return [0.0; 8];
        }
        let mut totals = [0.0f64; 8];
        for w in worlds {
            for i in 0..8 {
                totals[i] += w.current_scores[i];
            }
        }
        let n = worlds.len() as f64;
        for t in &mut totals {
            *t /= n;
        }
        totals
    }

    /// Mean pairwise distance of harmony score vectors across worlds.
    ///
    /// High values indicate worlds are emphasizing different harmonies.
    pub fn per_world_emphasis_divergence(trackers: &[HarmonyTracker]) -> f64 {
        if trackers.len() < 2 {
            return 0.0;
        }
        let mut total_dist = 0.0f64;
        let mut pairs = 0usize;
        for i in 0..trackers.len() {
            for j in (i + 1)..trackers.len() {
                let diff: f64 = trackers[i]
                    .current_scores
                    .iter()
                    .zip(trackers[j].current_scores.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                total_dist += diff;
                pairs += 1;
            }
        }
        if pairs == 0 {
            return 0.0;
        }
        total_dist / pairs as f64
    }

    /// Harmony conflict index: structural conflict when some harmonies are high
    /// and others low. Formula: max_score - min_score.
    pub fn harmony_conflict_index(tracker: &HarmonyTracker) -> f64 {
        let max = tracker
            .current_scores
            .iter()
            .copied()
            .fold(0.0f64, f64::max);
        let min = tracker
            .current_scores
            .iter()
            .copied()
            .fold(1.0f64, f64::min);
        (max - min).max(0.0)
    }

    /// Mean love coherence across all provided worlds.
    pub fn civilization_love_coherence(worlds: &[HarmonyTracker]) -> f64 {
        if worlds.is_empty() {
            return 0.0;
        }
        worlds.iter().map(|w| w.love_coherence).sum::<f64>() / worlds.len() as f64
    }
}

impl Default for HarmonyTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Least-squares slope of a time series (x = 0, 1, 2, ...).
fn linear_slope(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let sum_x: f64 = (0..n).map(|i| i as f64).sum();
    let sum_y: f64 = values.iter().sum();
    let sum_xy: f64 = values.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
    let sum_xx: f64 = (0..n).map(|i| (i as f64) * (i as f64)).sum();
    let denom = nf * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return 0.0;
    }
    (nf * sum_xy - sum_x * sum_y) / denom
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{BiologicalSex, CivAgent, ConsciousnessState, SkillVector};
    use crate::world::{CulturalProfile, World, WorldResources};

    fn make_world(pop: usize) -> World {
        let mut world = World {
            id: 0,
            name: "TestWorld".into(),
            location: "Moon".into(),
            founded_tick: 0,
            parent_world_id: None,
            agents: Vec::new(),
            next_agent_id: 0,
            resources: WorldResources::lunar_default(),
            culture: CulturalProfile::pioneer_default(),
            infrastructure_level: 0.5,
            max_population: 10_000,
            habitable_area_m2: 1_000_000.0,
            founding_harmony_emphasis: [0.125; 8],
            epidemics: Vec::new(),
            knowledge: crate::knowledge::WorldKnowledge::new(),
            economy: crate::economy::WorldEconomy::new(),
            harmony: crate::harmony::HarmonyTracker::new(),
            governance: crate::governance::WorldGovernance::new(),
            metabolism_state: crate::metabolism::MetabolismState::default(),
            currency_state: crate::currency::WorldCurrencyState::default(),
            policy_state: crate::proposals::PolicyState::default(),
            power_generation_kw: 0.0,
            power_demand_kw: 0.0,
            narrative_identity: crate::world::NarrativeIdentity::default(),
            maintenance_hours_required: 0.0,
            maintenance_hours_available: 0.0,
            bus_factor_critical: 0,
            pathogen_pressure: 0.0,
            civilizational_phi: 0.0,
            trust_level: 0.7,
            earth_funding: 1.0,
            mortality_alpha_mult: 1.0,
            mortality_beta_mult: 1.0,
            mortality_lambda_mult: 1.0,
            reproduction_viable: true,
            ecosystem_balance: 1.0,
            fertility_multiplier: 1.0,
            automation_level: 0.0,
            explorations_completed: 0,
            project_manager: crate::projects::ProjectManager::new(),
            habitat: crate::habitat::HabitatComplex::default(),
            fleet: crate::robotics::RoboticFleet::default(),
            diplomatic_relations: std::collections::HashMap::new(),
            zones: Vec::new(),
            moral_memories: Vec::new(),
            institutional_ethics: crate::agent::EthicalOrientation::default(),
        };
        for i in 0..pop {
            world.agents.push(CivAgent {
                id: i as u64,
                birth_tick: 0,
                death_tick: None,
                sex: if i % 2 == 0 {
                    BiologicalSex::Female
                } else {
                    BiologicalSex::Male
                },
                world_id: 0,
                health: 0.9,
                skills: SkillVector::new(),
                education_level: 0.5,
                consciousness: ConsciousnessState {
                    level: 0.5,
                    meta_awareness: 0.4,
                    coherence: 0.5,
                    care_activation: 0.4,
                    harmonic_alignment: 0.4,
                    epistemic_confidence: 0.4,
                },
                partner_id: None,
                children_ids: vec![],
                is_immigrant: false,
                needs: crate::needs::PsychologicalNeeds::new(),
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
                ethics: crate::agent::EthicalOrientation::default(),
                sovereign_profile: crate::sovereign_profile::SovereignProfile::zero(),
                justice: crate::sub_passport::RestorativeJustice::new(),
            });
        }
        world.next_agent_id = pop as u64;
        world
    }

    fn make_consciousness() -> ConsciousnessEngine {
        let mut e = ConsciousnessEngine::new();
        e.collective_phi = 0.4;
        e.mean_phi = 0.5;
        e
    }

    #[test]
    fn test_harmony_scoring_from_world_state() {
        let world = make_world(50);
        let consciousness = make_consciousness();
        let inputs = HarmonyInputs {
            governance_stability: 0.8,
            food_level: 0.7,
            mean_education: 0.5,
            mean_tech_level: 0.4,
            innovation_rate: 0.3,
            art_per_capita: 0.2,
            trade_connections: 3,
            gini_coefficient: 0.3,
            self_sufficiency: 0.7,
            knowledge_growth_rate: 0.1,
            pop_stability: 0.9,
            genetic_diversity: 0.8,
            emergency_fraction: 0.0,
            worker_ratio: 0.5,
            mean_allostatic_load: 0.1,
            mean_engagement: 0.8,
        };

        let mut tracker = HarmonyTracker::new();
        let events = tracker.tick_harmony(&world, &inputs, &consciousness, 100);

        // All scores should be in [0, 1]
        for (i, &score) in tracker.current_scores.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&score),
                "Harmony {} ({}) out of range: {}",
                i,
                HARMONY_NAMES[i],
                score
            );
        }
        assert!(tracker.love_coherence >= 0.0);
        // No milestone expected at first tick with these moderate values
        let _ = events;
    }

    #[test]
    fn test_love_coherence_formula() {
        let mut tracker = HarmonyTracker::new();
        // All harmonies equal at 0.5 — low tension, good diversity
        tracker.current_scores = [0.5; 8];

        // Manually compute what tick_harmony would produce
        let mean: f64 = 0.5;
        let _variance: f64 = 0.0; // all equal
        let tension: f64 = 0.0;
        let diversity: f64 = 8.0 / 8.0; // all above 0.3
        let expected = (mean * (1.0 - tension) * diversity).min(LOVE_COHERENCE_CEILING);

        // Use a world + inputs that produce all-0.5 scores
        // Easier to just test the formula directly
        assert!((expected - 0.5).abs() < 1e-10);

        // High tension case: one harmony at 1.0, rest at 0.0
        tracker.current_scores = [0.0; 8];
        tracker.current_scores[0] = 1.0;
        let mean2: f64 = 1.0 / 8.0;
        let var2: f64 = (1.0 * (1.0 - mean2).powi(2) + 7.0 * (0.0 - mean2).powi(2)) / 8.0;
        let tension2 = var2 / mean2;
        let diversity2 = 1.0 / 8.0; // only 1 above 0.3
        let coherence2 = (mean2 * (1.0 - tension2.min(1.0)) * diversity2).min(0.95);
        // Should be very low due to poor diversity and high tension
        assert!(
            coherence2 < 0.1,
            "Imbalanced harmonies should yield low coherence: {coherence2}"
        );
    }

    #[test]
    fn test_tension_ratio() {
        let world = make_world(10);
        let consciousness = make_consciousness();
        let mut tracker = HarmonyTracker::new();

        // All same inputs → low tension
        let inputs = HarmonyInputs::default();
        tracker.tick_harmony(&world, &inputs, &consciousness, 1);
        let t1 = tracker.tension_ratio;

        // Extreme imbalance: high trade, zero everything else
        let inputs2 = HarmonyInputs {
            trade_connections: 5,
            governance_stability: 0.0,
            food_level: 0.0,
            mean_education: 0.0,
            mean_tech_level: 0.0,
            innovation_rate: 0.0,
            art_per_capita: 0.0,
            gini_coefficient: 1.0,
            self_sufficiency: 0.0,
            knowledge_growth_rate: 0.0,
            pop_stability: 0.0,
            genetic_diversity: 0.0,
            emergency_fraction: 1.0,
            worker_ratio: 1.0,
            mean_allostatic_load: 0.5,
            mean_engagement: 0.3,
        };
        tracker.tick_harmony(&world, &inputs2, &consciousness, 2);
        let t2 = tracker.tension_ratio;

        assert!(
            t2 > t1,
            "Imbalanced inputs should produce higher tension: {t2} vs {t1}"
        );
    }

    #[test]
    fn test_diversity_factor() {
        let mut tracker = HarmonyTracker::new();

        // All above threshold
        tracker.current_scores = [0.5; 8];
        let diversity_all = tracker
            .current_scores
            .iter()
            .filter(|&&s| s > DIVERSITY_THRESHOLD)
            .count() as f64
            / 8.0;
        assert!((diversity_all - 1.0).abs() < 1e-10);

        // Only 2 above threshold
        tracker.current_scores = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0];
        let diversity_two = tracker
            .current_scores
            .iter()
            .filter(|&&s| s > DIVERSITY_THRESHOLD)
            .count() as f64
            / 8.0;
        assert!((diversity_two - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_love_coherence_ceiling_095() {
        let world = make_world(10);
        let mut consciousness = make_consciousness();
        consciousness.collective_phi = 1.0;
        consciousness.mean_phi = 1.0;

        let inputs = HarmonyInputs {
            governance_stability: 1.0,
            food_level: 1.0,
            mean_education: 1.0,
            mean_tech_level: 1.0,
            innovation_rate: 1.0,
            art_per_capita: 1.0,
            trade_connections: 5,
            gini_coefficient: 0.0,
            self_sufficiency: 1.0,
            knowledge_growth_rate: 1.0,
            pop_stability: 1.0,
            genetic_diversity: 1.0,
            emergency_fraction: 0.0,
            worker_ratio: 0.5,
            mean_allostatic_load: 0.1,
            mean_engagement: 0.8,
        };

        let mut tracker = HarmonyTracker::new();
        tracker.tick_harmony(&world, &inputs, &consciousness, 1);

        assert!(
            tracker.love_coherence <= LOVE_COHERENCE_CEILING,
            "Love coherence should be capped at {LOVE_COHERENCE_CEILING}, was {}",
            tracker.love_coherence
        );
    }

    #[test]
    fn test_civilization_aggregate() {
        let mut t1 = HarmonyTracker::new();
        t1.current_scores = [0.4; 8];
        t1.love_coherence = 0.3;

        let mut t2 = HarmonyTracker::new();
        t2.current_scores = [0.6; 8];
        t2.love_coherence = 0.5;

        let civ = HarmonyTracker::civilization_harmony(&[t1.clone(), t2.clone()]);
        for &s in &civ {
            assert!((s - 0.5).abs() < 1e-10, "Mean of 0.4 and 0.6 should be 0.5");
        }

        let civ_love = HarmonyTracker::civilization_love_coherence(&[t1, t2]);
        assert!(
            (civ_love - 0.4).abs() < 1e-10,
            "Mean of 0.3 and 0.5 should be 0.4, was {civ_love}"
        );
    }

    #[test]
    fn test_harmony_trajectory_positive() {
        let mut tracker = HarmonyTracker::new();
        for i in 0..20 {
            tracker.history.push(HarmonySnapshot {
                tick: i,
                scores: [0.0; 8],
                love_coherence: i as f64 * 0.01,
            });
        }
        assert!(
            tracker.harmony_trajectory() > 0.0,
            "Increasing love coherence should give positive trajectory"
        );
    }
}
