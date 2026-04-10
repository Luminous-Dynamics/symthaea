// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consciousness evolution: individual growth and collective Phi computation.
//!
//! Each tick evolves individual consciousness dimensions (level, meta_awareness,
//! coherence, care_activation, harmonic_alignment, epistemic_confidence) based on
//! education, age, social role, and cultural alignment. Collective Phi aggregates
//! individual Phi weighted by inter-agent coherence.

use crate::agent::CivAgent;
use crate::events::{CivEvent, CivEventType};
use crate::stochastic::StochasticEngine;
use crate::world::CulturalProfile;

use serde::{Deserialize, Serialize};

/// Consciousness decay per tick — consciousness is earned, not permanent.
const DECAY_RATE: f64 = 0.0015;

/// Maximum number of agents to sample for pairwise coherence.
/// Bug #2 fix: increased from 100 to 200 for better statistical estimate.
const COHERENCE_SAMPLE_AGENTS: usize = 200;

/// Maximum number of pairwise comparisons for inter-agent coherence.
/// Bug #2 fix: increased from 50 to 200 for more stable coherence estimate.
const COHERENCE_SAMPLE_PAIRS: usize = 200;

/// History length: 120 ticks = 10 years at monthly resolution.
const PHI_HISTORY_LEN: usize = 120;

/// Collective Phi threshold for first HarmonyMilestone event.
const PHI_MILESTONE_THRESHOLD: f64 = 0.5;

/// Collective Phi threshold for consciousness viability.
const PHI_VIABLE_THRESHOLD: f64 = 0.1;

/// Fragile consensus: tier concentration above this fraction is suspicious.
const FRAGILE_TIER_THRESHOLD: f64 = 0.7;

/// Fragile consensus: collective Phi below this value means the consensus is fragile.
const FRAGILE_PHI_THRESHOLD: f64 = 0.3;

/// Individual and collective consciousness evolution engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessEngine {
    /// Collective Phi for this world (mean_phi * inter_agent_coherence).
    pub collective_phi: f64,
    /// Inter-agent coherence (pairwise cosine similarity of 6D vectors, sampled).
    pub inter_agent_coherence: f64,
    /// Mean individual Phi across sampled agents.
    pub mean_phi: f64,
    /// Tier distribution [Observer, Participant, Contributor, Steward, Guardian].
    pub tier_distribution: [f64; 5],
    /// Fragile consensus detected (high tier agreement, low collective Phi).
    pub fragile_consensus: bool,
    /// History of collective Phi (last 120 ticks = 10 years).
    pub phi_history: Vec<f64>,
    /// Whether the Phi > 0.5 milestone has already been logged.
    phi_milestone_logged: bool,
}

impl ConsciousnessEngine {
    /// Create a new engine with default state.
    pub fn new() -> Self {
        Self {
            collective_phi: 0.0,
            inter_agent_coherence: 0.0,
            mean_phi: 0.0,
            tier_distribution: [0.0; 5],
            fragile_consensus: false,
            phi_history: Vec::with_capacity(PHI_HISTORY_LEN),
            phi_milestone_logged: false,
        }
    }

    /// Evolve individual consciousness and recompute collective metrics.
    ///
    /// Returns any milestone events generated this tick.
    pub fn tick_consciousness(
        &mut self,
        agents: &mut [CivAgent],
        culture: &CulturalProfile,
        current_tick: u32,
        rng: &mut StochasticEngine,
    ) -> Vec<CivEvent> {
        let mut events = Vec::new();

        // --- Individual evolution ---
        for agent in agents.iter_mut().filter(|a| a.is_alive()) {
            // Decay all dimensions (consciousness is earned)
            agent.consciousness.level = (agent.consciousness.level - DECAY_RATE).max(0.0);
            agent.consciousness.meta_awareness =
                (agent.consciousness.meta_awareness - DECAY_RATE).max(0.0);
            agent.consciousness.coherence =
                (agent.consciousness.coherence - DECAY_RATE).max(0.0);
            agent.consciousness.care_activation =
                (agent.consciousness.care_activation - DECAY_RATE).max(0.0);
            agent.consciousness.harmonic_alignment =
                (agent.consciousness.harmonic_alignment - DECAY_RATE).max(0.0);
            agent.consciousness.epistemic_confidence =
                (agent.consciousness.epistemic_confidence - DECAY_RATE).max(0.0);

            // Level: grows with education (must exceed DECAY_RATE for educated agents)
            agent.consciousness.level =
                (agent.consciousness.level + agent.education_level * 0.003).min(1.0);

            // Level: governance participation bonus (tier >= 2 = Contributor)
            if agent.consciousness.tier() >= 2 {
                agent.consciousness.level = (agent.consciousness.level + 0.001).min(1.0);
            }

            // Meta-awareness: grows with age (wisdom of elders, > 40 years)
            let age_years = agent.age_years(current_tick);
            if age_years > 40.0 {
                agent.consciousness.meta_awareness =
                    (agent.consciousness.meta_awareness + 0.0005).min(1.0);
            }

            // Coherence: emergent from education, trauma, experience, and cultural alignment.
            // Previously a static cultural-weight lookup; now multi-factor so coherence
            // is genuinely earned through education and resilience, not assigned.
            let skill_slice = agent.skills.as_slice();
            let strongest_idx = skill_slice
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let edu_factor = agent.education_level;
            let trauma_factor = 1.0 - agent.trauma_level;
            let experience_factor = (agent.age_years(current_tick) / 80.0).min(1.0);
            let cultural_factor = culture.harmony_weights[strongest_idx].min(1.0);
            agent.consciousness.coherence = (
                edu_factor * 0.4
                + trauma_factor * 0.2
                + experience_factor * 0.2
                + cultural_factor * 0.2
            ).clamp(0.0, 1.0);

            // Care activation: grows if agent has children or works in medicine/education
            if !agent.children_ids.is_empty() {
                agent.consciousness.care_activation =
                    (agent.consciousness.care_activation + 0.001).min(1.0);
            }
            let strongest = agent.skills.strongest();
            if strongest == "medicine" || strongest == "education" {
                agent.consciousness.care_activation =
                    (agent.consciousness.care_activation + 0.001).min(1.0);
            }

            // Harmonic alignment: cosine similarity between skill vector and cultural weights
            agent.consciousness.harmonic_alignment =
                cosine_similarity_8(&skill_slice, &culture.harmony_weights).max(0.0).min(1.0);

            // Epistemic confidence: grows with education
            agent.consciousness.epistemic_confidence =
                (agent.consciousness.epistemic_confidence + agent.education_level * 0.001)
                    .min(1.0);
        }

        // --- Collective computation ---
        let living: Vec<&CivAgent> = agents.iter().filter(|a| a.is_alive()).collect();
        let n = living.len();

        if n == 0 {
            self.mean_phi = 0.0;
            self.inter_agent_coherence = 0.0;
            self.collective_phi = 0.0;
            self.tier_distribution = [0.0; 5];
            self.fragile_consensus = false;
            self.push_phi_history();
            return events;
        }

        // Sample up to COHERENCE_SAMPLE_AGENTS agents (stride-based)
        let stride = if n > COHERENCE_SAMPLE_AGENTS {
            n / COHERENCE_SAMPLE_AGENTS
        } else {
            1
        };
        let sampled: Vec<&CivAgent> = living.iter().step_by(stride).copied().collect();

        // Mean Phi from sampled agents
        self.mean_phi = sampled.iter().map(|a| a.consciousness.phi()).sum::<f64>()
            / sampled.len() as f64;

        // Inter-agent coherence: pairwise cosine similarity of 6D consciousness vectors
        if sampled.len() >= 2 {
            let mut coherence_sum = 0.0;
            let pairs = COHERENCE_SAMPLE_PAIRS.min(sampled.len() * (sampled.len() - 1) / 2);
            let mut pair_count = 0u32;
            for _ in 0..pairs {
                let i = (rng.next_u64() as usize) % sampled.len();
                let mut j = (rng.next_u64() as usize) % sampled.len();
                if j == i {
                    j = (i + 1) % sampled.len();
                }
                coherence_sum +=
                    consciousness_cosine_similarity(&sampled[i].consciousness, &sampled[j].consciousness);
                pair_count += 1;
            }
            self.inter_agent_coherence = if pair_count > 0 {
                coherence_sum / pair_count as f64
            } else {
                0.0
            };

            // Bug #2 fix: Add institutional coherence floor.
            // Even a diverse population has shared institutions (language, governance,
            // infrastructure) that maintain collective coherence. Without this floor,
            // collective_phi collapses toward zero for large populations because
            // pairwise cosine similarity drops with diversity — physically wrong.
            // Floor: 0.2 base + 0.1 dampened by population size.
            let n = living.len() as f64;
            let institutional_floor = 0.2 + 0.1 / (1.0 + (n / 500.0).max(0.0).ln().max(0.0));
            self.inter_agent_coherence = self.inter_agent_coherence.max(institutional_floor);
        } else {
            self.inter_agent_coherence = 1.0; // single agent is perfectly coherent with itself
        }

        // No finite civilization achieves perfect consciousness — cap at 0.85.
        self.collective_phi = (self.mean_phi * self.inter_agent_coherence).min(0.85);

        // Tier distribution from ALL living agents
        let mut counts = [0usize; 5];
        for a in &living {
            counts[a.consciousness.tier() as usize] += 1;
        }
        for i in 0..5 {
            self.tier_distribution[i] = counts[i] as f64 / n as f64;
        }

        // Fragile consensus: >70% at one tier but collective Phi < 0.3
        let max_tier_frac = self
            .tier_distribution
            .iter()
            .copied()
            .fold(0.0f64, f64::max);
        self.fragile_consensus =
            max_tier_frac > FRAGILE_TIER_THRESHOLD && self.collective_phi < FRAGILE_PHI_THRESHOLD;

        // History
        self.push_phi_history();

        // Milestone: collective Phi crosses 0.5 for the first time
        if !self.phi_milestone_logged && self.collective_phi >= PHI_MILESTONE_THRESHOLD {
            self.phi_milestone_logged = true;
            events.push(CivEvent::new(
                current_tick,
                None,
                CivEventType::HarmonyMilestone,
                format!(
                    "Collective Phi reached {:.3} (threshold {PHI_MILESTONE_THRESHOLD})",
                    self.collective_phi
                ),
            ));
        }

        events
    }

    /// Slope of phi_history (positive = improving consciousness).
    pub fn phi_trend(&self) -> f64 {
        linear_slope(&self.phi_history)
    }

    /// Whether civilization consciousness is viable (collective Phi > 0.1).
    pub fn consciousness_viable(&self) -> bool {
        self.collective_phi > PHI_VIABLE_THRESHOLD
    }

    fn push_phi_history(&mut self) {
        self.phi_history.push(self.collective_phi);
        if self.phi_history.len() > PHI_HISTORY_LEN {
            self.phi_history.remove(0);
        }
    }
}

impl Default for ConsciousnessEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Fix 5: Consciousness development with growth model.
///
/// Growth rate depends on education, mentoring, trauma, and age-gating.
/// Plateau: growth halves every 0.2 above current level (diminishing returns).
///
/// Ref: Kegan (1982) stages of adult development; Fischer & Bidell (2006)
/// dynamic skill theory.
pub fn tick_consciousness_development(agent: &mut CivAgent, has_mentor: bool) {
    let _age = if agent.birth_tick == 0 { 30.0 } else { 0.0 }; // approximate; caller should pass tick
    // We use consciousness.level as a proxy for development stage

    let base_rate = 0.001;
    let education_bonus = if agent.education_level > 0.3 { 0.002 } else { 0.0 };
    let mentor_bonus = if has_mentor { 0.003 } else { 0.0 };
    let trauma_penalty = if agent.trauma_level > 0.3 { -0.005 * agent.trauma_level } else { 0.0 };

    let raw_rate = base_rate + education_bonus + mentor_bonus + trauma_penalty;
    if raw_rate <= 0.0 {
        return;
    }

    // Plateau: growth halves every 0.2 above current level
    let current = agent.consciousness.level;
    let plateau_factor = 0.5_f64.powf(current / 0.2);
    let effective_rate = raw_rate * plateau_factor;

    agent.consciousness.level = (agent.consciousness.level + effective_rate).min(1.0);
}

/// Fix 5 variant with explicit age gating via current_tick.
pub fn tick_consciousness_development_with_age(
    agent: &mut CivAgent,
    has_mentor: bool,
    current_tick: u32,
) {
    let age_years = agent.age_years(current_tick);
    if age_years < 15.0 {
        return; // No consciousness growth below 15 years
    }

    let base_rate = 0.001;
    let education_bonus = if agent.education_level > 0.3 { 0.002 } else { 0.0 };
    let mentor_bonus = if has_mentor { 0.003 } else { 0.0 };
    let trauma_penalty = if agent.trauma_level > 0.3 { -0.005 * agent.trauma_level } else { 0.0 };

    let raw_rate = base_rate + education_bonus + mentor_bonus + trauma_penalty;
    if raw_rate <= 0.0 {
        return;
    }

    // Plateau: growth halves every 0.2 above current level
    let current = agent.consciousness.level;
    let plateau_factor = 0.5_f64.powf(current / 0.2);
    let effective_rate = raw_rate * plateau_factor;

    agent.consciousness.level = (agent.consciousness.level + effective_rate).min(1.0);
}

// --- Utility functions ---

/// 6D consciousness vector for an agent.
fn consciousness_vec(c: &crate::agent::ConsciousnessState) -> [f64; 6] {
    [
        c.level,
        c.meta_awareness,
        c.coherence,
        c.care_activation,
        c.harmonic_alignment,
        c.epistemic_confidence,
    ]
}

/// Cosine similarity between two 6D consciousness vectors.
fn consciousness_cosine_similarity(
    a: &crate::agent::ConsciousnessState,
    b: &crate::agent::ConsciousnessState,
) -> f64 {
    let va = consciousness_vec(a);
    let vb = consciousness_vec(b);
    let dot: f64 = va.iter().zip(vb.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = va.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = vb.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denom = mag_a * mag_b;
    if denom < 1e-15 {
        return 0.0;
    }
    (dot / denom).clamp(0.0, 1.0)
}

/// Cosine similarity between two 8-element vectors.
fn cosine_similarity_8(a: &[f64; 8], b: &[f64; 8]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denom = mag_a * mag_b;
    if denom < 1e-15 {
        return 0.0;
    }
    dot / denom
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

/// Tick consciousness growth across all worlds.
///
/// Extracted from `MultiWorldSimulator::tick_consciousness()`. Handles:
/// - Red team adversarial recruitment (maintain ~5% adversarial population)
/// - Per-agent consciousness decay (earned, not permanent; load-amplified)
/// - Growth with diminishing returns, burnout penalty, pharma boost
/// - Trust-weighted governance gating
/// - Phi ceiling (moral humility at 0.95)
pub fn tick_consciousness_all_worlds(
    worlds: &mut [crate::world::World],
    current_tick: u32,
    pharma_boost: f64,
    trust_weighted_governance: bool,
) {
    use crate::agent;
    use crate::red_team;

    // Red team: maintain adversarial population at ~5% if any adversaries exist.
    for world in worlds.iter_mut() {
        let has_adversaries = world.agents.iter().any(|a| a.adversarial.is_some());
        if has_adversaries {
            let living = world.agents.iter().filter(|a| a.is_alive()).count();
            let adv_count = world.agents.iter()
                .filter(|a| a.is_alive() && a.adversarial.is_some()).count();
            let target = (living as f64 * 0.05).ceil() as usize;
            if adv_count < target {
                let deficit = target - adv_count;
                let mut recruited = 0;
                for agent in world.agents.iter_mut()
                    .filter(|a| a.is_alive() && a.adversarial.is_none())
                {
                    if recruited >= deficit { break; }
                    agent.adversarial = Some(red_team::AdversarialStrategy::ProfileMaximizer);
                    recruited += 1;
                }
            }
        }
    }

    // Phase 7: Gradual consciousness growth for agents.
    for world in worlds.iter_mut() {
        let mean_edu: f64 = {
            let living: Vec<f64> = world.agents.iter()
                .filter(|a| a.is_alive())
                .map(|a| a.education_level)
                .collect();
            if living.is_empty() { 0.0 }
            else { living.iter().sum::<f64>() / living.len() as f64 }
        };

        for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
            let stage = agent.life_stage(current_tick);
            let growth_rate = match stage {
                agent::LifeStage::Child => 0.001,
                agent::LifeStage::Youth => 0.003,
                agent::LifeStage::Adult => 0.002,
                agent::LifeStage::Elder => 0.001,
            };

            let base_decay = if stage == agent::LifeStage::Adult || stage == agent::LifeStage::Elder {
                0.0015
            } else {
                0.0
            };
            let decay = base_decay * (1.0 + agent.needs.allostatic_load);
            let c = &mut agent.consciousness;
            c.level = (c.level - decay).max(0.0);
            c.meta_awareness = (c.meta_awareness - decay).max(0.0);
            c.coherence = (c.coherence - decay).max(0.0);
            c.care_activation = (c.care_activation - decay).max(0.0);
            c.harmonic_alignment = (c.harmonic_alignment - decay).max(0.0);
            c.epistemic_confidence = (c.epistemic_confidence - decay).max(0.0);

            let pharma = if world.infrastructure_level > 0.3 { pharma_boost } else { 0.0 };
            let gating_factor = if trust_weighted_governance {
                let phi = c.phi();
                let bonus = 0.5 / (1.0 + (-10.0 * (phi - 0.3)).exp());
                0.5 + bonus
            } else {
                0.5
            };
            let amplifier = (1.0 + mean_edu * 0.5 + pharma) * gating_factor;
            let burnout_penalty = if agent.needs.is_burnout() { 0.35 } else { 1.0 };
            let amplifier = amplifier * burnout_penalty;

            let amplifier = if let Some(strategy) = &agent.adversarial {
                let modifier = red_team::AdversarialModifier::for_strategy(*strategy, 0.01);
                amplifier * modifier.phi_growth_mult
            } else {
                amplifier
            };

            if agent.needs.social_satiation > 0.5 {
                c.care_activation = (c.care_activation + 0.0005).min(1.0);
            }
            let cap = 0.85;
            let dr = |dim: f64, rate: f64| -> f64 {
                let headroom = (1.0 - dim / cap).max(0.0);
                rate * amplifier * headroom
            };
            c.level = (c.level + dr(c.level, growth_rate)).min(cap);
            c.meta_awareness = (c.meta_awareness + dr(c.meta_awareness, growth_rate * 0.8)).min(cap);
            c.coherence = (c.coherence + dr(c.coherence, growth_rate * 0.6)).min(cap);
            c.care_activation = (c.care_activation + dr(c.care_activation, growth_rate * 0.7)).min(cap);
            c.harmonic_alignment =
                (c.harmonic_alignment + dr(c.harmonic_alignment, growth_rate * 0.5)).min(cap);
            c.epistemic_confidence =
                (c.epistemic_confidence + dr(c.epistemic_confidence, growth_rate * 0.4)).min(cap);

            let phi = c.phi();
            if phi > 0.95 {
                let scale = 0.95 / phi;
                c.level *= scale;
                c.meta_awareness *= scale;
                c.coherence *= scale;
                c.care_activation *= scale;
                c.harmonic_alignment *= scale;
                c.epistemic_confidence *= scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{BiologicalSex, CivAgent, ConsciousnessState, SkillVector};
    use crate::stochastic::StochasticEngine;
    use crate::world::CulturalProfile;

    fn make_agent(id: u64, birth_tick: u32) -> CivAgent {
        CivAgent {
            id,
            birth_tick,
            death_tick: None,
            sex: BiologicalSex::Female,
            world_id: 0,
            health: 1.0,
            skills: SkillVector::new(),
            education_level: 0.5,
            consciousness: ConsciousnessState::nascent(),
            partner_id: None,
            children_ids: vec![],
            is_immigrant: false,
            needs: crate::needs::PsychologicalNeeds::new(),
            tend_balance: 0.0,
                    parent_ids: None,
                    faction_id: None,
                    generation: 0,
                    trauma_level: 0.0,
                    cumulative_dose_sv: 0.0, adversarial: None, coordination_understanding: 0.0, mycel_score: 0.1, sap_balance: 100.0, is_biological: true, wounds: Vec::new(),
        }
    }

    #[test]
    fn test_individual_consciousness_growth() {
        let mut engine = ConsciousnessEngine::new();
        let mut rng = StochasticEngine::new(42);
        let culture = CulturalProfile::earth_default();
        let mut agents = vec![make_agent(0, 0)];
        agents[0].education_level = 0.8;

        let initial_level = agents[0].consciousness.level;
        for tick in 1..=120 {
            engine.tick_consciousness(&mut agents, &culture, tick, &mut rng);
        }
        // Education drives consciousness level up over time
        assert!(
            agents[0].consciousness.level > initial_level,
            "Level should grow with education: {} vs {}",
            agents[0].consciousness.level,
            initial_level
        );
    }

    #[test]
    fn test_collective_phi_from_agents() {
        let mut engine = ConsciousnessEngine::new();
        let mut rng = StochasticEngine::new(42);
        let culture = CulturalProfile::earth_default();

        // Create agents with elevated consciousness
        let mut agents: Vec<CivAgent> = (0..20).map(|i| {
            let mut a = make_agent(i, 0);
            a.consciousness.level = 0.7;
            a.consciousness.meta_awareness = 0.6;
            a.consciousness.coherence = 0.5;
            a.consciousness.care_activation = 0.5;
            a.consciousness.harmonic_alignment = 0.5;
            a.consciousness.epistemic_confidence = 0.5;
            a
        }).collect();

        engine.tick_consciousness(&mut agents, &culture, 1, &mut rng);

        assert!(engine.mean_phi > 0.0, "mean_phi should be positive");
        assert!(engine.collective_phi > 0.0, "collective_phi should be positive");
        assert!(
            engine.inter_agent_coherence > 0.0,
            "coherence should be positive for similar agents"
        );
    }

    #[test]
    fn test_tier_distribution_sums_to_one() {
        let mut engine = ConsciousnessEngine::new();
        let mut rng = StochasticEngine::new(42);
        let culture = CulturalProfile::earth_default();

        let mut agents: Vec<CivAgent> = (0..50).map(|i| make_agent(i, 0)).collect();
        engine.tick_consciousness(&mut agents, &culture, 100, &mut rng);

        let sum: f64 = engine.tier_distribution.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "tier distribution should sum to 1.0, was {sum}"
        );
    }

    #[test]
    fn test_fragile_consensus_detection() {
        let mut engine = ConsciousnessEngine::new();
        let mut rng = StochasticEngine::new(42);
        let culture = CulturalProfile::earth_default();

        // All agents at tier 0 (nascent) with low phi — classic fragile consensus.
        // Low education ensures emergent coherence stays low (phi < 0.3).
        let mut agents: Vec<CivAgent> = (0..30).map(|i| {
            let mut a = make_agent(i, 0);
            a.consciousness = ConsciousnessState::nascent();
            a.education_level = 0.05;
            a
        }).collect();

        engine.tick_consciousness(&mut agents, &culture, 1, &mut rng);

        // Nascent agents are all tier 0 (>70% at one level) and collective phi < 0.3
        assert!(
            engine.fragile_consensus,
            "Should detect fragile consensus when all agents are nascent"
        );
    }

    #[test]
    fn test_phi_history_tracking() {
        let mut engine = ConsciousnessEngine::new();
        let mut rng = StochasticEngine::new(42);
        let culture = CulturalProfile::earth_default();
        let mut agents = vec![make_agent(0, 0)];

        for tick in 1..=10 {
            engine.tick_consciousness(&mut agents, &culture, tick, &mut rng);
        }

        assert_eq!(
            engine.phi_history.len(),
            10,
            "Should have 10 history entries after 10 ticks"
        );

        // Run 200 more to test truncation
        for tick in 11..=200 {
            engine.tick_consciousness(&mut agents, &culture, tick, &mut rng);
        }
        assert!(
            engine.phi_history.len() <= PHI_HISTORY_LEN,
            "History should be capped at {PHI_HISTORY_LEN}, was {}",
            engine.phi_history.len()
        );
    }

    #[test]
    fn test_consciousness_viable_threshold() {
        let mut engine = ConsciousnessEngine::new();
        assert!(
            !engine.consciousness_viable(),
            "Zero collective phi should not be viable"
        );

        engine.collective_phi = 0.05;
        assert!(!engine.consciousness_viable());

        engine.collective_phi = 0.15;
        assert!(engine.consciousness_viable());
    }

    #[test]
    fn test_phi_trend_positive_for_growing_series() {
        let mut engine = ConsciousnessEngine::new();
        engine.phi_history = (0..20).map(|i| i as f64 * 0.01).collect();
        assert!(
            engine.phi_trend() > 0.0,
            "Trend should be positive for increasing series"
        );
    }

    #[test]
    fn test_consciousness_grows_over_time() {
        // Fix 5: Consciousness Development
        let mut agent = make_agent(0, 0);
        agent.education_level = 0.5;
        let initial_level = agent.consciousness.level;

        // 100 ticks of development with mentor
        for _ in 0..100 {
            tick_consciousness_development(&mut agent, true);
        }

        assert!(
            agent.consciousness.level > initial_level,
            "Consciousness should grow: {} vs {}",
            agent.consciousness.level, initial_level
        );
    }

    #[test]
    fn test_consciousness_age_gate() {
        // Fix 5: No growth below 15 years
        let mut child = make_agent(0, 100); // born at tick 100
        child.education_level = 0.8;
        let initial_level = child.consciousness.level;

        // Tick at age 10 (tick 220)
        tick_consciousness_development_with_age(&mut child, true, 220);
        assert!(
            (child.consciousness.level - initial_level).abs() < 1e-10,
            "Child should not grow consciousness"
        );

        // Tick at age 20 (tick 340)
        tick_consciousness_development_with_age(&mut child, true, 340);
        assert!(
            child.consciousness.level > initial_level,
            "Adult should grow consciousness"
        );
    }

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let c = ConsciousnessState {
            level: 0.5,
            meta_awareness: 0.5,
            coherence: 0.5,
            care_activation: 0.5,
            harmonic_alignment: 0.5,
            epistemic_confidence: 0.5,
        };
        let sim = consciousness_cosine_similarity(&c, &c);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "Identical vectors should have similarity 1.0, was {sim}"
        );
    }
}
