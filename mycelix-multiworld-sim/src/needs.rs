// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Psychological needs engine: allostatic load, social satiation, and engagement.
//!
//! Models human psychological needs as thermodynamic state variables with natural
//! accumulation/decay dynamics. The system reflects costs back through existing
//! mechanisms (mortality, consciousness, harmony, economy) rather than punitive
//! enforcement. "Vices" emerge as thermodynamic consequences of unmet needs.
//!
//! # Key Variables
//!
//! - **Allostatic load**: Cumulative stress from isolation, overwork, unmet needs.
//!   Decays via care work, intimacy, and stillness. Burnout at > 0.8.
//! - **Social satiation**: Relationship fulfillment. Decays monthly, replenished
//!   by partner/children/care interactions. Below 0.3 = isolation stress.
//! - **Engagement**: Physical-world participation. Decays under high load + low
//!   social (digital escapism proxy). Directly modulates labor productivity.
//!
//! # References
//!
//! - McEwen (1998) "Protective and Damaging Effects of Stress Mediators" NEJM 338(3):171-179
//! - Dunbar (2010) "How Many Friends Does One Person Need?" — social brain hypothesis
//! - Karasek (1979) "Job demands, job decision latitude" — demand-control model
//! - Cacioppo & Patrick (2008) "Loneliness: Human Nature and the Need for Social Connection"
//! - NASA HI-SEAS isolation studies — communication latency amplifies loneliness

use crate::events::{CivEvent, CivEventType};
use crate::population::SirEpidemic;
use crate::stochastic::StochasticEngine;
use crate::world::World;

use serde::{Deserialize, Serialize};

// =============================================================================
// Named constants with scientific citations
// =============================================================================

/// Allostatic load accumulation rate from social isolation (social_satiation < 0.3).
/// Ref: McEwen (1998) — chronic stress mediator accumulation.
const ISOLATION_LOAD_RATE: f64 = 0.015;

/// Allostatic load accumulation from overwork (worker_ratio > 0.6).
/// Ref: Karasek (1979) — demand-control model of occupational stress.
const OVERWORK_LOAD_RATE: f64 = 0.01;

/// Allostatic load natural decay per tick (monthly resolution).
/// Ref: Adapted from symthaea-psych-bench allostatic_stress.rs (-0.005 sleep / -0.002 wake).
const LOAD_DECAY_RATE: f64 = 0.008;

/// Care worker load reduction per worker per 100 recipients.
/// Models the TEND ServiceCategory::CareWork effect from mycelix-finance.
const CARE_LOAD_REDUCTION: f64 = 0.02;

/// Burnout threshold: allostatic_load above this caps consciousness growth.
/// Ref: symthaea-psych-bench burnout regime at load > 0.8.
const BURNOUT_THRESHOLD: f64 = 0.8;

/// Social satiation monthly decay rate.
/// Ref: Dunbar (2010) — relationships require active maintenance.
const SOCIAL_DECAY_RATE: f64 = 0.04;

/// Social satiation replenishment from a partner bond per tick.
const PARTNER_SOCIAL_BONUS: f64 = 0.03;

/// Social satiation replenishment per child relationship per tick.
const CHILD_SOCIAL_BONUS: f64 = 0.01;

/// Isolation threshold: social_satiation below this triggers stress accumulation.
/// Ref: Cacioppo & Patrick (2008) — chronic loneliness onset.
const ISOLATION_THRESHOLD: f64 = 0.3;

/// Engagement decay rate when allostatic_load > 0.5 AND social_satiation < 0.3.
/// Models digital escapism / VR withdrawal.
const ESCAPISM_DECAY_RATE: f64 = 0.02;

/// Engagement natural recovery rate when needs are met.
const ENGAGEMENT_RECOVERY_RATE: f64 = 0.005;

/// Thrill-seeking health risk probability per tick for eligible agents.
/// Eligibility: engagement > 0.7 AND social_satiation < 0.3 AND age 15-45.
const THRILL_RISK_PROBABILITY: f64 = 0.005;

/// Thrill-seeking health cost per incident.
const THRILL_HEALTH_COST: f64 = 0.1;

/// Rogue bio-hacking epidemic trigger probability per tick.
/// Requires: tech_level > 0.6 AND governance_stability < 0.5 AND mean_load > 0.5.
const BIOHACK_EPIDEMIC_PROBABILITY: f64 = 0.001;

/// Deep-space isolation multiplier for social decay (non-Earth, Epochs 2-3).
/// Ref: NASA HI-SEAS — communication latency (3-24 min) amplifies loneliness.
const DEEP_SPACE_SOCIAL_DECAY_MULT: f64 = 1.5;

/// Epoch range for deep-space isolation amplification (Branches through Canopy).
const DEEP_SPACE_EPOCH_START: u8 = 2;
const DEEP_SPACE_EPOCH_END: u8 = 3;

// =============================================================================
// Types
// =============================================================================

/// Per-agent psychological needs state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsychologicalNeeds {
    /// Cumulative stress from isolation, overwork, unmet needs [0, 1].
    pub allostatic_load: f64,
    /// Social connection satiation [0, 1]. Decays monthly, replenished by relationships.
    pub social_satiation: f64,
    /// Physical-world participation / engagement [0, 1].
    /// Decays under high load + low social (digital escapism proxy).
    pub engagement: f64,
}

impl PsychologicalNeeds {
    /// Default state for adult colonists: slight baseline stress, moderate social reserve.
    pub fn new() -> Self {
        Self {
            allostatic_load: 0.1,
            social_satiation: 0.7,
            engagement: 0.8,
        }
    }

    /// Newborn/child state: protected by parental bond.
    pub fn nascent() -> Self {
        Self {
            allostatic_load: 0.0,
            social_satiation: 0.9,
            engagement: 0.9,
        }
    }

    /// Whether this agent is in burnout.
    pub fn is_burnout(&self) -> bool {
        self.allostatic_load > BURNOUT_THRESHOLD
    }

    /// Whether this agent is socially isolated.
    pub fn is_isolated(&self) -> bool {
        self.social_satiation < ISOLATION_THRESHOLD
    }
}

impl Default for PsychologicalNeeds {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregate psychological needs summary for a world (computed each tick).
#[derive(Debug, Clone, Default)]
pub struct NeedsWorldSummary {
    pub mean_allostatic_load: f64,
    pub mean_social_satiation: f64,
    pub mean_engagement: f64,
    pub thrill_incidents: usize,
    /// Agents with engagement < 0.3 (digital escapism).
    pub escapism_count: usize,
    /// Agents in burnout (allostatic_load > 0.8).
    pub burnout_count: usize,
}

/// Psychological needs engine — stateless, operates on world + agents each tick.
pub struct PsychNeedsEngine;

impl PsychNeedsEngine {
    /// Tick psychological needs for all living agents in a world.
    ///
    /// Returns events (thrill incidents, bio-hack epidemics) and a world summary.
    pub fn tick_needs(
        world: &mut World,
        current_tick: u32,
        current_epoch: u8,
        care_worker_count: usize,
        mean_tech_level: f64,
        governance_stability: f64,
        worker_ratio: f64,
        rng: &mut StochasticEngine,
    ) -> (Vec<CivEvent>, NeedsWorldSummary) {
        let mut events = Vec::new();
        let is_off_earth = world.location != "Earth";
        let is_deep_space_epoch =
            current_epoch >= DEEP_SPACE_EPOCH_START && current_epoch <= DEEP_SPACE_EPOCH_END;

        let pop = world.population().max(1) as f64;
        let care_ratio = care_worker_count as f64 / (pop / 100.0).max(1.0);

        let mut total_load = 0.0;
        let mut total_social = 0.0;
        let mut total_engagement = 0.0;
        let mut thrill_incidents = 0usize;
        let mut escapism_count = 0usize;
        let mut burnout_count = 0usize;
        let mut living_count = 0usize;

        for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
            living_count += 1;
            // Extract needed fields before mutable borrow of agent.needs
            let has_partner = agent.partner_id.is_some();
            let num_children = agent.children_ids.len().min(5);
            let agent_age = agent.age_years(current_tick);
            let n = &mut agent.needs;

            // --- 1. Social satiation decay ---
            let mut decay = SOCIAL_DECAY_RATE;
            if is_off_earth && is_deep_space_epoch {
                decay *= DEEP_SPACE_SOCIAL_DECAY_MULT;
            }
            n.social_satiation = (n.social_satiation - decay).max(0.0);

            // --- 2. Social replenishment from relationships ---
            if has_partner {
                n.social_satiation = (n.social_satiation + PARTNER_SOCIAL_BONUS).min(1.0);
            }
            let child_bonus = CHILD_SOCIAL_BONUS * num_children as f64;
            n.social_satiation = (n.social_satiation + child_bonus).min(1.0);

            // --- 3. Allostatic load accumulation ---
            if n.social_satiation < ISOLATION_THRESHOLD {
                n.allostatic_load = (n.allostatic_load + ISOLATION_LOAD_RATE).min(1.0);
            }
            // Overwork: when worker_ratio > 0.6, working agents accumulate stress.
            if worker_ratio > 0.6 && agent_age >= 15.0 {
                n.allostatic_load =
                    (n.allostatic_load + OVERWORK_LOAD_RATE * (worker_ratio - 0.6) / 0.4).min(1.0);
            }

            // --- 4. Allostatic load decay (care workers help) ---
            let care_decay = LOAD_DECAY_RATE + CARE_LOAD_REDUCTION * care_ratio;
            n.allostatic_load = (n.allostatic_load - care_decay).max(0.0);

            // --- 5. Engagement dynamics (digital escapism) ---
            if n.allostatic_load > 0.5 && n.social_satiation < ISOLATION_THRESHOLD {
                n.engagement = (n.engagement - ESCAPISM_DECAY_RATE).max(0.0);
            } else {
                n.engagement = (n.engagement + ENGAGEMENT_RECOVERY_RATE).min(1.0);
            }

            // --- 6. Thrill-seeking eligibility (checked after needs borrow ends) ---
            // Thrill-seeking occurs when agents are physically active (engagement > 0.5)
            // but socially unfulfilled (social < 0.5) — seeking intensity through risk.
            // Previous condition (engagement > 0.7 AND social < 0.3) was nearly unreachable
            // because engagement decays under the same conditions that trigger isolation.
            let thrill_eligible = n.engagement > 0.5
                && n.social_satiation < 0.5
                && agent_age >= 15.0
                && agent_age <= 45.0;

            // --- Aggregate ---
            total_load += n.allostatic_load;
            total_social += n.social_satiation;
            total_engagement += n.engagement;
            if n.engagement < 0.3 {
                escapism_count += 1;
            }
            if n.is_burnout() {
                burnout_count += 1;
            }

            // End mutable borrow of agent.needs before accessing agent.health.
            let _ = n;

            // Thrill-seeking stochastic health event
            if thrill_eligible && rng.bernoulli(THRILL_RISK_PROBABILITY) {
                agent.health = (agent.health - THRILL_HEALTH_COST).max(0.1);
                thrill_incidents += 1;
                events.push(CivEvent::new(
                    current_tick,
                    Some(world.id),
                    CivEventType::ThrillIncident,
                    format!(
                        "Agent {} thrill-seeking incident on {} (health: {:.2})",
                        agent.id, world.name, agent.health
                    ),
                ));
            }
        }

        let count = living_count.max(1) as f64;
        let mean_load = total_load / count;

        // --- 7. Rogue bio-hacking epidemic trigger (world-level) ---
        if mean_tech_level > 0.6
            && governance_stability < 0.5
            && mean_load > 0.5
            && rng.bernoulli(BIOHACK_EPIDEMIC_PROBABILITY)
        {
            let pop_now = world.population();
            if pop_now > 20 {
                let epi = SirEpidemic {
                    name: "Novel Biohack Pathogen".into(),
                    beta: 0.25,
                    gamma: 0.08,
                    mortality: 0.03,
                    susceptible: pop_now.saturating_sub(1),
                    infected: 1,
                    recovered: 0,
                    dead: 0,
                    tick_started: current_tick,
                };
                world.epidemics.push(epi);
                events.push(CivEvent::new(
                    current_tick,
                    Some(world.id),
                    CivEventType::BiohackIncident,
                    format!(
                        "Rogue bio-hacking triggered novel pathogen on {} (load: {:.2}, gov: {:.2})",
                        world.name, mean_load, governance_stability
                    ),
                ));
            }
        }

        let summary = NeedsWorldSummary {
            mean_allostatic_load: mean_load,
            mean_social_satiation: total_social / count,
            mean_engagement: total_engagement / count,
            thrill_incidents,
            escapism_count,
            burnout_count,
        };

        (events, summary)
    }
}

// =============================================================================
// Public constant accessors (for integration in other modules)
// =============================================================================

/// Returns the burnout threshold for use in consciousness/mortality integration.
pub fn burnout_threshold() -> f64 {
    BURNOUT_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{BiologicalSex, CivAgent, ConsciousnessState, SkillVector};
    use crate::knowledge::WorldKnowledge;
    use crate::world::{CulturalProfile, WorldResources};

    fn make_agent(id: u64, age_years: u32, tick: u32) -> CivAgent {
        let birth_tick = tick.wrapping_sub(age_years * 12);
        CivAgent {
            id,
            birth_tick,
            death_tick: None,
            sex: if id % 2 == 0 {
                BiologicalSex::Female
            } else {
                BiologicalSex::Male
            },
            world_id: 0,
            health: 0.9,
            skills: SkillVector::new(),
            education_level: 0.5,
            consciousness: ConsciousnessState::nascent(),
            partner_id: None,
            children_ids: vec![],
            is_immigrant: false,
            needs: PsychologicalNeeds::new(),
            tend_balance: 0.0,
        }
    }

    fn make_world(agents: Vec<CivAgent>) -> World {
        let n = agents.len();
        World {
            id: 0,
            name: "TestWorld".into(),
            location: "Moon".into(),
            founded_tick: 0,
            parent_world_id: None,
            agents,
            next_agent_id: n as u64,
            resources: WorldResources::lunar_default(),
            culture: CulturalProfile::pioneer_default(),
            infrastructure_level: 0.5,
            max_population: 10_000,
            habitable_area_m2: 1_000_000.0,
            founding_harmony_emphasis: [0.125; 8],
            epidemics: Vec::new(),
            knowledge: WorldKnowledge::new(),
            economy: crate::economy::WorldEconomy::new(),
            harmony: crate::harmony::HarmonyTracker::new(),
        }
    }

    #[test]
    fn test_social_satiation_decays_monthly() {
        let mut agent = make_agent(0, 30, 360);
        let initial = agent.needs.social_satiation;
        agent.needs.social_satiation -= SOCIAL_DECAY_RATE;
        assert!(
            agent.needs.social_satiation < initial,
            "Social satiation should decay: {} -> {}",
            initial,
            agent.needs.social_satiation
        );
    }

    #[test]
    fn test_partner_replenishes_social() {
        let tick = 360;
        let mut agents = vec![make_agent(0, 30, tick), make_agent(1, 30, tick)];
        agents[0].partner_id = Some(1);
        agents[0].needs.social_satiation = 0.5;
        agents[1].needs.social_satiation = 0.5;

        let mut world = make_world(agents);
        let mut rng = StochasticEngine::new(42);

        PsychNeedsEngine::tick_needs(&mut world, tick, 1, 0, 0.3, 0.8, 0.5, &mut rng);

        let with_partner = world.agents[0].needs.social_satiation;
        let without_partner = world.agents[1].needs.social_satiation;

        assert!(
            with_partner > without_partner,
            "Agent with partner ({with_partner:.3}) should have higher social than without ({without_partner:.3})"
        );
    }

    #[test]
    fn test_isolation_increases_allostatic_load() {
        let tick = 360;
        let mut agents = vec![make_agent(0, 30, tick)];
        agents[0].needs.social_satiation = 0.1; // well below threshold
        agents[0].needs.allostatic_load = 0.2;

        let mut world = make_world(agents);
        let mut rng = StochasticEngine::new(42);

        PsychNeedsEngine::tick_needs(&mut world, tick, 1, 0, 0.3, 0.8, 0.5, &mut rng);

        // Load should increase due to isolation, minus care decay
        // Net: +ISOLATION_LOAD_RATE - LOAD_DECAY_RATE = 0.015 - 0.008 = +0.007
        assert!(
            world.agents[0].needs.allostatic_load > 0.2,
            "Isolation should increase load: {:.4}",
            world.agents[0].needs.allostatic_load
        );
    }

    #[test]
    fn test_care_workers_reduce_load() {
        let tick = 360;

        // World with care workers
        let mut agents_with = vec![make_agent(0, 30, tick)];
        agents_with[0].needs.allostatic_load = 0.5;
        agents_with[0].needs.social_satiation = 0.5; // above isolation threshold
        let mut world_with = make_world(agents_with);
        let mut rng1 = StochasticEngine::new(42);
        PsychNeedsEngine::tick_needs(&mut world_with, tick, 1, 5, 0.3, 0.8, 0.5, &mut rng1);
        let load_with_care = world_with.agents[0].needs.allostatic_load;

        // World without care workers
        let mut agents_without = vec![make_agent(0, 30, tick)];
        agents_without[0].needs.allostatic_load = 0.5;
        agents_without[0].needs.social_satiation = 0.5;
        let mut world_without = make_world(agents_without);
        let mut rng2 = StochasticEngine::new(42);
        PsychNeedsEngine::tick_needs(&mut world_without, tick, 1, 0, 0.3, 0.8, 0.5, &mut rng2);
        let load_without_care = world_without.agents[0].needs.allostatic_load;

        assert!(
            load_with_care < load_without_care,
            "Care workers should reduce load: {load_with_care:.4} vs {load_without_care:.4}"
        );
    }

    #[test]
    fn test_engagement_decays_under_stress() {
        let tick = 360;
        let mut agents = vec![make_agent(0, 30, tick)];
        agents[0].needs.allostatic_load = 0.7; // above 0.5
        agents[0].needs.social_satiation = 0.1; // below 0.3
        agents[0].needs.engagement = 0.6;

        let mut world = make_world(agents);
        let mut rng = StochasticEngine::new(42);

        PsychNeedsEngine::tick_needs(&mut world, tick, 1, 0, 0.3, 0.8, 0.5, &mut rng);

        assert!(
            world.agents[0].needs.engagement < 0.6,
            "Engagement should decay under stress: {:.3}",
            world.agents[0].needs.engagement
        );
    }

    #[test]
    fn test_engagement_recovers_when_needs_met() {
        let tick = 360;
        let mut agents = vec![make_agent(0, 30, tick)];
        agents[0].needs.allostatic_load = 0.1;
        agents[0].needs.social_satiation = 0.7;
        agents[0].needs.engagement = 0.5;

        let mut world = make_world(agents);
        let mut rng = StochasticEngine::new(42);

        PsychNeedsEngine::tick_needs(&mut world, tick, 1, 0, 0.3, 0.8, 0.5, &mut rng);

        assert!(
            world.agents[0].needs.engagement > 0.5,
            "Engagement should recover: {:.3}",
            world.agents[0].needs.engagement
        );
    }

    #[test]
    fn test_burnout_threshold() {
        let mut needs = PsychologicalNeeds::new();
        assert!(!needs.is_burnout());
        needs.allostatic_load = 0.85;
        assert!(needs.is_burnout());
    }

    #[test]
    fn test_thrill_seeking_only_working_age() {
        let tick = 360;
        // Child (age 5) — should NOT have thrill incidents
        let mut agents = vec![make_agent(0, 5, tick)];
        agents[0].needs.engagement = 0.9;
        agents[0].needs.social_satiation = 0.1;

        let mut world = make_world(agents);
        let mut rng = StochasticEngine::new(42);

        // Run many ticks to give probability a chance
        for t in tick..tick + 100 {
            PsychNeedsEngine::tick_needs(&mut world, t, 1, 0, 0.3, 0.8, 0.5, &mut rng);
        }
        // Child health should be untouched by thrill-seeking
        assert!(
            world.agents[0].health >= 0.85,
            "Children should not have thrill incidents: health={:.2}",
            world.agents[0].health
        );
    }

    #[test]
    fn test_biohack_requires_conditions() {
        let tick = 360;
        let agents: Vec<CivAgent> = (0..50).map(|i| make_agent(i, 30, tick)).collect();
        let mut world = make_world(agents);
        let mut rng = StochasticEngine::new(42);

        // Good governance, low tech — should NOT trigger
        for t in tick..tick + 500 {
            PsychNeedsEngine::tick_needs(&mut world, t, 1, 0, 0.3, 0.8, 0.5, &mut rng);
        }
        assert!(
            world.epidemics.is_empty(),
            "Biohack should not trigger with good governance"
        );
    }

    #[test]
    fn test_deep_space_amplifies_social_decay() {
        let tick = 360;

        // Epoch 2 (Branches), off-Earth
        let mut agents_deep = vec![make_agent(0, 30, tick)];
        agents_deep[0].needs.social_satiation = 0.7;
        let mut world_deep = make_world(agents_deep);
        let mut rng1 = StochasticEngine::new(42);
        PsychNeedsEngine::tick_needs(&mut world_deep, tick, 2, 0, 0.3, 0.8, 0.5, &mut rng1);
        let social_deep = world_deep.agents[0].needs.social_satiation;

        // Epoch 1 (Roots), same world
        let mut agents_normal = vec![make_agent(0, 30, tick)];
        agents_normal[0].needs.social_satiation = 0.7;
        let mut world_normal = make_world(agents_normal);
        let mut rng2 = StochasticEngine::new(42);
        PsychNeedsEngine::tick_needs(&mut world_normal, tick, 1, 0, 0.3, 0.8, 0.5, &mut rng2);
        let social_normal = world_normal.agents[0].needs.social_satiation;

        assert!(
            social_deep < social_normal,
            "Deep space should amplify social decay: {social_deep:.3} vs {social_normal:.3}"
        );
    }

    #[test]
    fn test_needs_world_summary() {
        let tick = 360;
        let agents: Vec<CivAgent> = (0..20).map(|i| make_agent(i, 30, tick)).collect();
        let mut world = make_world(agents);
        let mut rng = StochasticEngine::new(42);

        let (_, summary) = PsychNeedsEngine::tick_needs(&mut world, tick, 1, 0, 0.3, 0.8, 0.5, &mut rng);

        assert!(summary.mean_allostatic_load >= 0.0 && summary.mean_allostatic_load <= 1.0);
        assert!(summary.mean_social_satiation >= 0.0 && summary.mean_social_satiation <= 1.0);
        assert!(summary.mean_engagement >= 0.0 && summary.mean_engagement <= 1.0);
    }

    #[test]
    fn test_all_needs_bounded() {
        let tick = 360;
        let mut agents: Vec<CivAgent> = (0..10).map(|i| make_agent(i, 30, tick)).collect();
        // Set extreme values
        for a in &mut agents {
            a.needs.allostatic_load = 1.0;
            a.needs.social_satiation = 0.0;
            a.needs.engagement = 0.0;
        }
        let mut world = make_world(agents);
        let mut rng = StochasticEngine::new(42);

        for t in tick..tick + 50 {
            PsychNeedsEngine::tick_needs(&mut world, t, 1, 0, 0.3, 0.8, 0.5, &mut rng);
        }

        for a in &world.agents {
            assert!(a.needs.allostatic_load >= 0.0 && a.needs.allostatic_load <= 1.0);
            assert!(a.needs.social_satiation >= 0.0 && a.needs.social_satiation <= 1.0);
            assert!(a.needs.engagement >= 0.0 && a.needs.engagement <= 1.0);
        }
    }
}
