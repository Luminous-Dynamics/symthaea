// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Decentralized peer-to-peer education engine: curiosity-driven agents teach
//! each other, earn TEND for knowledge sharing, and learning emerges from
//! community needs.
//!
//! The 1602 education system is a batch-processing facility optimized for
//! compliance. The human brain is an Active Inference engine that naturally
//! minimizes expected free energy through curiosity. This tick models what
//! happens when you stop crushing that default and start leveraging it.
//!
//! # Phases
//!
//! A. **Allostatic gating** — classify each agent's learning mode based on stress.
//!    You cannot force a highly stressed neural network to update priors.
//! B. **Epistemic foraging** — agents seek knowledge in weakest sectors (curiosity).
//! C. **Peer-to-peer teaching** — matched interactions with TEND rewards.
//! D. **Community skill crisis** — real stakes when collective skill drops.
//!
//! # References
//!
//! - McEwen (2004) "Allostatic Load and Allostatic Overload" — stress gating
//! - Nestojko et al. (2014) "Expecting to teach enhances learning" — protégé effect
//! - Friston et al. (2017) "Active Inference, Curiosity and Insight" — epistemic foraging

use crate::events::{CivEvent, CivEventType};
use crate::stochastic::StochasticEngine;
use crate::world::World;

// =============================================================================
// Named constants with scientific citations
// =============================================================================

/// Allostatic gating thresholds.
/// You cannot force a highly stressed neural network to update priors.
/// (McEwen, 2004: Allostatic Load and Allostatic Overload)
const HIGH_STRESS_THRESHOLD: f64 = 0.6;
const MEDIUM_STRESS_THRESHOLD: f64 = 0.3;

/// Peer teaching parameters.
/// Teaching consolidates the teacher's knowledge (Nestojko et al., 2014).
/// "The protégé effect" — expecting to teach enhances learning.
const TEACHING_SKILL_GAP: f64 = 0.2; // Minimum gap for teaching to be useful
const LEARNER_SKILL_GAIN: f64 = 0.015; // Per interaction
const TEACHER_CONSOLIDATION: f64 = 0.003; // Protégé effect
const TEACHING_SOCIAL_BOOST: f64 = 0.02; // Both parties (per interaction)
const TEACHING_TEND_REWARD: f64 = 5.0; // Teaching is care work

/// Epistemic friction — the thermodynamic cost of updating priors.
/// Rewiring neural pathways is biologically stressful (Lupien et al., 2009).
/// Learning MUST cost energy, or the model collapses to a Utopian Attractor.
const LEARNER_EPISTEMIC_COST: f64 = 0.015; // Allostatic load spike from learning
const FORAGING_EPISTEMIC_COST: f64 = 0.008; // Self-directed is less stressful than peer

/// Teacher cognitive fatigue — projecting knowledge outward costs bandwidth.
/// The teacher's own FEP loop takes a hit from externalizing their model
/// (Kalyuga, 2007: expertise reversal effect applied to teaching load).
const TEACHER_FATIGUE_COST: f64 = 0.01; // Allostatic load per teaching event

/// Social satiation cap from education per tick.
/// The 5th teaching interaction cannot provide the same dopamine hit as the 1st.
/// Logarithmic diminishing returns on social bonding from repeated interactions.
const MAX_EDUCATION_SOCIAL_BOOST_PER_TICK: f64 = 0.04; // Cap per agent per tick

/// Epistemic foraging parameters.
/// Curiosity = expected information gain (Friston et al., 2017).
const CURIOSITY_THRESHOLD: f64 = 0.3; // Min uncertainty to trigger foraging
const FORAGING_SKILL_GAIN: f64 = 0.008; // Self-directed learning (slower than peer)
const REVIEW_SKILL_GAIN: f64 = 0.004; // Light consolidation under medium stress
const EDUCATION_LEVEL_GAIN: f64 = 0.003; // Per foraging tick

/// Community stakes — collective skill thresholds.
/// Below these, the community suffers (real thermodynamic consequences).
const COMMUNITY_CRISIS_THRESHOLD: f64 = 0.20;
const CRISIS_LOAD_PENALTY: f64 = 0.01; // Allostatic load increase per tick during crisis

/// Capacity limits.
const MAX_TEACHINGS_PER_TICK: usize = 3;

// =============================================================================
// Types
// =============================================================================

/// Learning mode for an agent this tick, determined by allostatic state.
#[derive(Debug, Clone, Copy, PartialEq)]
enum LearningMode {
    /// High stress: rest, no learning. Healing the Markov blanket.
    Rest,
    /// Medium stress: light review only. Consolidation, not exploration.
    Review,
    /// Low stress: full epistemic foraging. Curiosity-driven exploration.
    Foraging,
}

/// A matched teaching interaction.
#[derive(Debug, Clone)]
struct TeachingMatch {
    teacher_idx: usize,
    learner_idx: usize,
    sector: usize,
}

/// Summary of education activity for a world this tick.
#[derive(Debug, Clone, Default)]
pub struct EducationTickSummary {
    pub agents_foraging: usize,
    pub agents_reviewing: usize,
    pub agents_resting: usize,
    pub teaching_interactions: usize,
    pub tend_distributed: f64,
    pub skill_crises: Vec<usize>, // sector indices in crisis
    pub mean_education_level: f64,
}

// =============================================================================
// Engine
// =============================================================================

/// Stateless education engine — operates on a World each tick.
pub struct EducationEngine;

impl EducationEngine {
    /// Run the education tick for a world.
    ///
    /// Phases:
    /// A. Allostatic gating — classify each agent's learning mode
    /// B. Epistemic foraging — agents seek knowledge in weakest sectors
    /// C. Peer-to-peer teaching — matched interactions, TEND rewards
    /// D. Community skill crisis — real stakes when collective skill drops
    pub fn tick(
        world: &mut World,
        current_tick: u32,
        rng: &mut StochasticEngine,
    ) -> (Vec<CivEvent>, EducationTickSummary) {
        let mut events = Vec::new();
        let mut summary = EducationTickSummary::default();

        // Phase A: Classify learning modes
        let modes: Vec<LearningMode> = world
            .agents
            .iter()
            .map(|a| {
                if !a.is_alive() {
                    return LearningMode::Rest;
                }
                if a.needs.allostatic_load > HIGH_STRESS_THRESHOLD {
                    LearningMode::Rest
                } else if a.needs.allostatic_load > MEDIUM_STRESS_THRESHOLD {
                    LearningMode::Review
                } else {
                    LearningMode::Foraging
                }
            })
            .collect();

        // Count modes
        for &m in &modes {
            match m {
                LearningMode::Foraging => summary.agents_foraging += 1,
                LearningMode::Review => summary.agents_reviewing += 1,
                LearningMode::Rest => summary.agents_resting += 1,
            }
        }

        // Phase B: Epistemic foraging + review
        // Each foraging agent identifies their weakest sector (highest curiosity)
        let mut learning_targets: Vec<Option<usize>> = vec![None; world.agents.len()];

        for (i, agent) in world.agents.iter_mut().enumerate() {
            if !agent.is_alive() {
                continue;
            }
            match modes[i] {
                LearningMode::Foraging => {
                    // Find weakest sector where curiosity exceeds threshold
                    let skills = agent.skills.as_slice();
                    let mut weakest_sector = None;
                    let mut max_curiosity: f64 = 0.0;
                    for (s, &skill) in skills.iter().enumerate() {
                        let curiosity = 1.0 - skill;
                        if curiosity > CURIOSITY_THRESHOLD && curiosity > max_curiosity {
                            max_curiosity = curiosity;
                            weakest_sector = Some(s);
                        }
                    }
                    learning_targets[i] = weakest_sector;

                    // Self-directed learning (slower than peer teaching)
                    // Epistemic friction: learning costs energy (Lupien et al., 2009)
                    if weakest_sector.is_some() {
                        let sector = weakest_sector.unwrap();
                        agent.skills.learn(sector, FORAGING_SKILL_GAIN);
                        agent.education_level =
                            (agent.education_level + EDUCATION_LEVEL_GAIN).min(1.0);
                        // Learning is work — updating priors costs allostatic energy
                        agent.needs.allostatic_load =
                            (agent.needs.allostatic_load + FORAGING_EPISTEMIC_COST).min(1.0);
                    }
                }
                LearningMode::Review => {
                    // Light consolidation of strongest skill
                    let skills = agent.skills.as_slice();
                    let strongest = skills
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    agent.skills.learn(strongest, REVIEW_SKILL_GAIN);
                }
                LearningMode::Rest => {
                    // Rest: no learning. Allostatic load recovery handled by needs tick.
                }
            }
        }

        // Phase C: Peer-to-peer teaching matchmaking
        // Build teacher pool: foraging agents with skill > 0.5 in any sector.
        // Sector-indexed lookup for O(n) matching instead of O(n^2).
        let num_sectors = 8;
        let mut teacher_pool: Vec<Vec<usize>> = vec![Vec::new(); num_sectors];
        let mut teach_count: Vec<usize> = vec![0; world.agents.len()];

        for (i, agent) in world.agents.iter().enumerate() {
            if !agent.is_alive() || modes[i] != LearningMode::Foraging {
                continue;
            }
            let skills = agent.skills.as_slice();
            for (s, &skill) in skills.iter().enumerate() {
                if skill > 0.5 {
                    teacher_pool[s].push(i);
                }
            }
        }

        // Collect learner indices (shuffled for fairness)
        let mut learner_indices: Vec<usize> = (0..world.agents.len())
            .filter(|&i| learning_targets[i].is_some())
            .collect();
        // Fisher-Yates shuffle using StochasticEngine
        for i in (1..learner_indices.len()).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            learner_indices.swap(i, j);
        }

        // Match learners to teachers
        let mut matches: Vec<TeachingMatch> = Vec::new();

        for &learner_idx in &learner_indices {
            let sector = match learning_targets[learner_idx] {
                Some(s) => s,
                None => continue,
            };

            let learner_skill = world.agents[learner_idx].skills.as_slice()[sector];

            // Find a teacher in this sector
            if let Some(teachers) = teacher_pool.get(sector) {
                for &teacher_idx in teachers {
                    if teacher_idx == learner_idx {
                        continue;
                    }
                    if teach_count[teacher_idx] >= MAX_TEACHINGS_PER_TICK {
                        continue;
                    }

                    let teacher_skill = world.agents[teacher_idx].skills.as_slice()[sector];
                    if teacher_skill - learner_skill >= TEACHING_SKILL_GAP {
                        matches.push(TeachingMatch {
                            teacher_idx,
                            learner_idx,
                            sector,
                        });
                        teach_count[teacher_idx] += 1;
                        break; // One teacher per learner per tick
                    }
                }
            }
        }

        // Track cumulative social boost per agent this tick (for diminishing returns)
        let mut social_boost_this_tick: Vec<f64> = vec![0.0; world.agents.len()];

        // Apply teaching interactions
        for m in &matches {
            // --- Learner ---
            // Skill gain, boosted by teacher's coordination understanding.
            // Teachers with high cu (perspective taking) adapt to the learner's
            // zone of proximal development, producing up to 30% more effective transfer.
            let teacher_cu = world.agents[m.teacher_idx].coordination_understanding;
            let cu_teaching_boost = 1.0 + teacher_cu * 0.3;
            world.agents[m.learner_idx]
                .skills
                .learn(m.sector, LEARNER_SKILL_GAIN * cu_teaching_boost);
            world.agents[m.learner_idx].education_level =
                (world.agents[m.learner_idx].education_level + 0.002 * cu_teaching_boost).min(1.0);

            // Epistemic cost: updating priors is biologically stressful.
            // Learning is exhausting. A guild must balance education with rest.
            world.agents[m.learner_idx].needs.allostatic_load =
                (world.agents[m.learner_idx].needs.allostatic_load + LEARNER_EPISTEMIC_COST)
                    .min(1.0);

            // Social boost with diminishing returns (capped per tick)
            let learner_remaining =
                MAX_EDUCATION_SOCIAL_BOOST_PER_TICK - social_boost_this_tick[m.learner_idx];
            let learner_boost = TEACHING_SOCIAL_BOOST.min(learner_remaining.max(0.0));
            world.agents[m.learner_idx].needs.social_satiation =
                (world.agents[m.learner_idx].needs.social_satiation + learner_boost).min(1.0);
            social_boost_this_tick[m.learner_idx] += learner_boost;

            // --- Teacher ---
            // Consolidation (protégé effect)
            world.agents[m.teacher_idx]
                .skills
                .learn(m.sector, TEACHER_CONSOLIDATION);

            // Teacher cognitive fatigue: projecting knowledge outward costs bandwidth.
            // Teaching is care work, and care work is real work.
            world.agents[m.teacher_idx].needs.allostatic_load =
                (world.agents[m.teacher_idx].needs.allostatic_load + TEACHER_FATIGUE_COST).min(1.0);

            // Social boost with diminishing returns
            let teacher_remaining =
                MAX_EDUCATION_SOCIAL_BOOST_PER_TICK - social_boost_this_tick[m.teacher_idx];
            let teacher_boost = TEACHING_SOCIAL_BOOST.min(teacher_remaining.max(0.0));
            world.agents[m.teacher_idx].needs.social_satiation =
                (world.agents[m.teacher_idx].needs.social_satiation + teacher_boost).min(1.0);
            social_boost_this_tick[m.teacher_idx] += teacher_boost;

            // TEND reward
            world.agents[m.teacher_idx].tend_balance += TEACHING_TEND_REWARD;

            // Ethical transmission (Phase 3a): teacher→student orientation blending.
            // Mentorship subtly shapes ethical worldview (Kohlberg 1981; Bandura 1977).
            // Rate 0.03/tick — slow enough that it takes ~33 teaching interactions
            // to shift a dimension by one full unit. Students aren't blank slates.
            let teacher_ethics = world.agents[m.teacher_idx].ethics.clone();
            let student = &mut world.agents[m.learner_idx];
            student.ethics.deontological +=
                (teacher_ethics.deontological - student.ethics.deontological) * 0.03;
            student.ethics.consequentialist +=
                (teacher_ethics.consequentialist - student.ethics.consequentialist) * 0.03;
            student.ethics.virtue_care +=
                (teacher_ethics.virtue_care - student.ethics.virtue_care) * 0.03;
            student.ethics.relational +=
                (teacher_ethics.relational - student.ethics.relational) * 0.03;

            summary.teaching_interactions += 1;
            summary.tend_distributed += TEACHING_TEND_REWARD;
        }

        // Emit teaching events
        if summary.teaching_interactions > 0 {
            events.push(CivEvent::new(
                current_tick,
                Some(world.id),
                CivEventType::TeachingInteraction,
                format!(
                    "{} peer teaching interactions, {:.0} TEND distributed",
                    summary.teaching_interactions, summary.tend_distributed,
                ),
            ));
        }

        // Phase D: Community skill crisis — real stakes
        // First pass: compute mean skills and education (immutable borrow)
        let n = world.agents.iter().filter(|a| a.is_alive()).count() as f64;
        if n > 0.0 {
            let critical_sectors: [(usize, &str); 3] =
                [(1, "agriculture"), (2, "medicine"), (0, "engineering")];
            let mut crisis_sectors: Vec<(usize, &str, f64)> = Vec::new();

            for &(sector, name) in &critical_sectors {
                let mean_skill: f64 = world
                    .agents
                    .iter()
                    .filter(|a| a.is_alive())
                    .map(|a| a.skills.as_slice()[sector])
                    .sum::<f64>()
                    / n;

                if mean_skill < COMMUNITY_CRISIS_THRESHOLD {
                    crisis_sectors.push((sector, name, mean_skill));
                }
            }

            summary.mean_education_level = world
                .agents
                .iter()
                .filter(|a| a.is_alive())
                .map(|a| a.education_level)
                .sum::<f64>()
                / n;

            // Second pass: apply penalties (mutable borrow)
            for &(sector, sector_name, mean_skill) in &crisis_sectors {
                summary.skill_crises.push(sector);

                for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                    agent.needs.allostatic_load =
                        (agent.needs.allostatic_load + CRISIS_LOAD_PENALTY).min(1.0);
                }

                events.push(CivEvent::new(
                    current_tick,
                    Some(world.id),
                    CivEventType::SkillCrisis,
                    format!(
                        "Community {} skill crisis (mean {:.3} < {:.3}). Allostatic load rising.",
                        sector_name, mean_skill, COMMUNITY_CRISIS_THRESHOLD,
                    ),
                ));
            }
        }

        (events, summary)
    }
}

/// Run education tick across all worlds, collecting events.
///
/// Extracted from `MultiWorldSimulator::tick_education()`.
/// Takes `&mut` of each world independently — no `std::mem::take` needed.
pub fn tick_education_all_worlds(
    worlds: &mut [World],
    current_tick: u32,
    rng: &mut StochasticEngine,
) -> Vec<CivEvent> {
    let mut all_events = Vec::new();
    for world in worlds.iter_mut() {
        let (events, _summary) = EducationEngine::tick(world, current_tick, rng);
        all_events.extend(events);
    }
    all_events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{BiologicalSex, CivAgent, ConsciousnessState, SkillVector};
    use crate::knowledge::WorldKnowledge;
    use crate::needs::PsychologicalNeeds;
    use crate::stochastic::StochasticEngine;
    use crate::world::{CulturalProfile, World, WorldResources};

    /// Reference tick used by all education tests.
    const TEST_TICK: u32 = 1000;

    /// Make a learner whose weakest sector is `weak_sector` (kept at 0.1).
    /// All other sectors raised to 0.65 so the weak sector is the clear
    /// curiosity target for epistemic foraging.
    fn make_learner_weak_in(id: u64, age_years: u32, weak_sector: usize) -> CivAgent {
        let mut a = make_agent(id, age_years);
        a.needs.allostatic_load = 0.1;
        for s in 0..8 {
            if s != weak_sector {
                a.skills.learn(s, 0.55); // becomes 0.65
            }
        }
        a
    }

    fn make_agent(id: u64, age_years: u32) -> CivAgent {
        let birth_tick = TEST_TICK.wrapping_sub(age_years * 12);
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
        }
    }

    // -------------------------------------------------------------------------
    // Phase A: Allostatic gating
    // -------------------------------------------------------------------------

    #[test]
    fn test_high_stress_agent_rests() {
        let mut agent = make_agent(0, 30);
        agent.needs.allostatic_load = 0.9;
        let initial_education = agent.education_level;
        let initial_skills = agent.skills.as_slice();

        let mut world = make_world(vec![agent]);
        let mut rng = StochasticEngine::new(42);

        let (_, summary) = EducationEngine::tick(&mut world, TEST_TICK, &mut rng);

        assert_eq!(summary.agents_resting, 1);
        assert_eq!(summary.agents_foraging, 0);
        assert_eq!(summary.agents_reviewing, 0);
        // Skills and education should be unchanged
        assert!(
            (world.agents[0].education_level - initial_education).abs() < 1e-9,
            "High-stress agent should not learn"
        );
        assert_eq!(world.agents[0].skills.as_slice(), initial_skills);
    }

    #[test]
    fn test_medium_stress_agent_reviews() {
        let mut agent = make_agent(0, 30);
        agent.needs.allostatic_load = 0.4; // between 0.3 and 0.6
                                           // Give a clear strongest skill
        agent.skills.learn(2, 0.5); // medicine = 0.6

        let initial_medicine = agent.skills.as_slice()[2];
        let mut world = make_world(vec![agent]);
        let mut rng = StochasticEngine::new(42);

        let (_, summary) = EducationEngine::tick(&mut world, TEST_TICK, &mut rng);

        assert_eq!(summary.agents_reviewing, 1);
        assert_eq!(summary.agents_foraging, 0);
        // Strongest skill should have gained REVIEW_SKILL_GAIN
        let new_medicine = world.agents[0].skills.as_slice()[2];
        assert!(
            (new_medicine - initial_medicine - REVIEW_SKILL_GAIN).abs() < 1e-9,
            "Review should consolidate strongest: {new_medicine} vs {initial_medicine}"
        );
    }

    #[test]
    fn test_low_stress_agent_forages() {
        let mut agent = make_agent(0, 30);
        agent.needs.allostatic_load = 0.1; // below 0.3

        let mut world = make_world(vec![agent]);
        let mut rng = StochasticEngine::new(42);

        let (_, summary) = EducationEngine::tick(&mut world, TEST_TICK, &mut rng);

        assert_eq!(summary.agents_foraging, 1);
        assert_eq!(summary.agents_resting, 0);
        assert_eq!(summary.agents_reviewing, 0);
    }

    // -------------------------------------------------------------------------
    // Phase B: Epistemic foraging
    // -------------------------------------------------------------------------

    #[test]
    fn test_foraging_learns_weakest_sector() {
        let mut agent = make_agent(0, 30);
        agent.needs.allostatic_load = 0.1;
        // Make one sector strong, rest at default 0.1
        agent.skills.learn(0, 0.8); // engineering = 0.9

        let mut world = make_world(vec![agent]);
        let mut rng = StochasticEngine::new(42);

        // All non-engineering sectors are at 0.1 (curiosity = 0.9 > threshold)
        let initial_skills = world.agents[0].skills.as_slice();
        EducationEngine::tick(&mut world, TEST_TICK, &mut rng);

        // At least one weak sector should have gained skill
        let new_skills = world.agents[0].skills.as_slice();
        let any_gained = (0..8).any(|s| new_skills[s] > initial_skills[s] + 1e-9 && s != 0);
        assert!(any_gained, "Foraging should improve a weak sector");
    }

    #[test]
    fn test_no_foraging_when_all_skilled() {
        let mut agent = make_agent(0, 30);
        agent.needs.allostatic_load = 0.1;
        // Set all skills above (1.0 - CURIOSITY_THRESHOLD) = 0.7
        for s in 0..8 {
            agent.skills.learn(s, 0.65); // all become 0.75
        }
        let initial_education = agent.education_level;

        let mut world = make_world(vec![agent]);
        let mut rng = StochasticEngine::new(42);

        EducationEngine::tick(&mut world, TEST_TICK, &mut rng);

        // Education level should NOT increase (no foraging target found)
        assert!(
            (world.agents[0].education_level - initial_education).abs() < 1e-9,
            "Agent with all skills > 0.7 should not forage: {} vs {}",
            world.agents[0].education_level,
            initial_education
        );
    }

    // -------------------------------------------------------------------------
    // Phase C: Peer teaching
    // -------------------------------------------------------------------------

    #[test]
    fn test_teaching_match_when_gap_sufficient() {
        // Teacher: high medicine, low stress
        let mut teacher = make_agent(0, 30);
        teacher.needs.allostatic_load = 0.1;
        teacher.skills.learn(2, 0.7); // medicine = 0.8

        // Learner: low medicine, low stress. Other skills raised so medicine is
        // clearly the weakest sector (highest curiosity target).
        let mut learner = make_agent(1, 25);
        learner.needs.allostatic_load = 0.1;
        // Raise all non-medicine sectors above default so medicine (0.1) is weakest
        for s in [0, 1, 3, 4, 5, 6, 7] {
            learner.skills.learn(s, 0.55); // each becomes 0.65
        }
        // medicine stays at 0.1 (gap = 0.7 > TEACHING_SKILL_GAP)

        let initial_learner_med = learner.skills.as_slice()[2];
        let mut world = make_world(vec![teacher, learner]);
        let mut rng = StochasticEngine::new(42);

        let (_, summary) = EducationEngine::tick(&mut world, TEST_TICK, &mut rng);

        assert!(
            summary.teaching_interactions > 0,
            "Should have at least one teaching interaction"
        );
        // Learner's medicine should have increased
        let new_learner_med = world.agents[1].skills.as_slice()[2];
        assert!(
            new_learner_med > initial_learner_med,
            "Learner should gain skill: {new_learner_med} vs {initial_learner_med}"
        );
    }

    #[test]
    fn test_no_teaching_when_gap_too_small() {
        // Both agents with similar skills
        let mut a1 = make_agent(0, 30);
        a1.needs.allostatic_load = 0.1;
        a1.skills.learn(2, 0.55); // medicine = 0.65

        let mut a2 = make_agent(1, 25);
        a2.needs.allostatic_load = 0.1;
        a2.skills.learn(2, 0.45); // medicine = 0.55
                                  // Gap = 0.10 < TEACHING_SKILL_GAP (0.2)

        let mut world = make_world(vec![a1, a2]);
        let mut rng = StochasticEngine::new(42);

        let (_, summary) = EducationEngine::tick(&mut world, TEST_TICK, &mut rng);

        assert_eq!(
            summary.teaching_interactions, 0,
            "No teaching when gap < {TEACHING_SKILL_GAP}"
        );
    }

    #[test]
    fn test_teacher_earns_tend() {
        let mut teacher = make_agent(0, 30);
        teacher.needs.allostatic_load = 0.1;
        teacher.skills.learn(2, 0.7); // medicine = 0.8

        let learner = make_learner_weak_in(1, 25, 2); // weak in medicine

        let mut world = make_world(vec![teacher, learner]);
        let mut rng = StochasticEngine::new(42);

        EducationEngine::tick(&mut world, TEST_TICK, &mut rng);

        assert!(
            world.agents[0].tend_balance > 0.0,
            "Teacher should earn TEND: {}",
            world.agents[0].tend_balance
        );
    }

    #[test]
    fn test_teaching_social_boost() {
        let mut teacher = make_agent(0, 30);
        teacher.needs.allostatic_load = 0.1;
        teacher.needs.social_satiation = 0.5;
        teacher.skills.learn(2, 0.7); // medicine = 0.8

        let mut learner = make_learner_weak_in(1, 25, 2);
        learner.needs.social_satiation = 0.5;

        let mut world = make_world(vec![teacher, learner]);
        let mut rng = StochasticEngine::new(42);

        let (_, summary) = EducationEngine::tick(&mut world, TEST_TICK, &mut rng);

        assert!(
            summary.teaching_interactions > 0,
            "Should have teaching interaction"
        );
        assert!(
            world.agents[0].needs.social_satiation > 0.5,
            "Teacher social should increase: {}",
            world.agents[0].needs.social_satiation
        );
        assert!(
            world.agents[1].needs.social_satiation > 0.5,
            "Learner social should increase: {}",
            world.agents[1].needs.social_satiation
        );
    }

    #[test]
    fn test_teacher_consolidation_protege_effect() {
        let mut teacher = make_agent(0, 30);
        teacher.needs.allostatic_load = 0.1;
        teacher.skills.learn(2, 0.7); // medicine = 0.8
        let initial_teacher_med = teacher.skills.as_slice()[2];

        let learner = make_learner_weak_in(1, 25, 2);

        let mut world = make_world(vec![teacher, learner]);
        let mut rng = StochasticEngine::new(42);

        let (_, summary) = EducationEngine::tick(&mut world, TEST_TICK, &mut rng);

        assert!(
            summary.teaching_interactions > 0,
            "Should have teaching interaction"
        );
        let new_teacher_med = world.agents[0].skills.as_slice()[2];
        assert!(
            new_teacher_med > initial_teacher_med,
            "Teacher should consolidate via protégé effect: {new_teacher_med} vs {initial_teacher_med}"
        );
    }

    #[test]
    fn test_max_teachings_per_tick() {
        // One strong teacher (medicine=0.8, others=0.65), many weak-in-medicine learners.
        // The teacher is also strong in other sectors so learners can't reverse-teach.
        let mut teacher = make_agent(0, 30);
        teacher.needs.allostatic_load = 0.1;
        teacher.skills.learn(2, 0.7); // medicine = 0.8
                                      // Raise teacher's other skills so learners can't teach back
        for s in [0, 1, 3, 4, 5, 6, 7] {
            teacher.skills.learn(s, 0.55); // becomes 0.65
        }

        let mut agents = vec![teacher];
        for i in 1..=10u64 {
            let learner = make_learner_weak_in(i, 25, 2);
            agents.push(learner);
        }

        let mut world = make_world(agents);
        let mut rng = StochasticEngine::new(42);

        let (_, summary) = EducationEngine::tick(&mut world, TEST_TICK, &mut rng);

        // Total interactions should be bounded by teacher capacity since there's only
        // one teacher who can teach in medicine (the only sector with gap >= 0.2).
        assert!(
            summary.teaching_interactions <= MAX_TEACHINGS_PER_TICK,
            "Teacher limited to {} interactions, got {}",
            MAX_TEACHINGS_PER_TICK,
            summary.teaching_interactions
        );
    }

    // -------------------------------------------------------------------------
    // Phase D: Community skill crisis
    // -------------------------------------------------------------------------

    #[test]
    fn test_skill_crisis_raises_load() {
        // All agents with very low agriculture (sector 1)
        let agents: Vec<CivAgent> = (0..20)
            .map(|i| {
                let mut a = make_agent(i, 30);
                a.needs.allostatic_load = 0.1;
                // agriculture stays at default 0.1, which is < COMMUNITY_CRISIS_THRESHOLD (0.20)
                a
            })
            .collect();

        let initial_load = agents[0].needs.allostatic_load;
        let mut world = make_world(agents);
        let mut rng = StochasticEngine::new(42);

        let (events, summary) = EducationEngine::tick(&mut world, TEST_TICK, &mut rng);

        assert!(
            !summary.skill_crises.is_empty(),
            "Should detect skill crisis"
        );
        // All agents should have increased load from crisis
        assert!(
            world.agents[0].needs.allostatic_load > initial_load,
            "Crisis should raise allostatic load: {} vs {initial_load}",
            world.agents[0].needs.allostatic_load
        );
        // Should emit crisis events
        let crisis_events: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == CivEventType::SkillCrisis)
            .collect();
        assert!(!crisis_events.is_empty(), "Should emit SkillCrisis event");
    }

    #[test]
    fn test_no_crisis_when_skills_adequate() {
        let agents: Vec<CivAgent> = (0..20)
            .map(|i| {
                let mut a = make_agent(i, 30);
                a.needs.allostatic_load = 0.1;
                // Raise agriculture, medicine, engineering above crisis threshold
                a.skills.learn(0, 0.2); // engineering = 0.3
                a.skills.learn(1, 0.2); // agriculture = 0.3
                a.skills.learn(2, 0.2); // medicine = 0.3
                a
            })
            .collect();

        let mut world = make_world(agents);
        let mut rng = StochasticEngine::new(42);

        let (_, summary) = EducationEngine::tick(&mut world, TEST_TICK, &mut rng);

        assert!(
            summary.skill_crises.is_empty(),
            "No crisis when critical sectors above threshold"
        );
    }

    // -------------------------------------------------------------------------
    // Education level tracking
    // -------------------------------------------------------------------------

    #[test]
    fn test_education_level_increases_for_foraging() {
        let mut agent = make_agent(0, 30);
        agent.needs.allostatic_load = 0.1;
        let initial = agent.education_level;

        let mut world = make_world(vec![agent]);
        let mut rng = StochasticEngine::new(42);

        EducationEngine::tick(&mut world, TEST_TICK, &mut rng);

        assert!(
            world.agents[0].education_level > initial,
            "Foraging should increase education_level: {} vs {initial}",
            world.agents[0].education_level
        );
    }

    // -------------------------------------------------------------------------
    // Dead agent exclusion
    // -------------------------------------------------------------------------

    #[test]
    fn test_dead_agents_excluded() {
        let mut alive = make_agent(0, 30);
        alive.needs.allostatic_load = 0.1;

        let mut dead = make_agent(1, 30);
        dead.death_tick = Some(500); // dead
        dead.needs.allostatic_load = 0.1;
        let dead_education = dead.education_level;
        let dead_skills = dead.skills.as_slice();

        let mut world = make_world(vec![alive, dead]);
        let mut rng = StochasticEngine::new(42);

        EducationEngine::tick(&mut world, TEST_TICK, &mut rng);

        // Dead agent unchanged
        assert!(
            (world.agents[1].education_level - dead_education).abs() < 1e-9,
            "Dead agent should not learn"
        );
        assert_eq!(world.agents[1].skills.as_slice(), dead_skills);
    }

    // -------------------------------------------------------------------------
    // TEND distribution tracking
    // -------------------------------------------------------------------------

    #[test]
    fn test_tend_distributed_tracks_total() {
        let mut teacher = make_agent(0, 30);
        teacher.needs.allostatic_load = 0.1;
        teacher.skills.learn(2, 0.7); // medicine = 0.8

        let learner = make_learner_weak_in(1, 25, 2);

        let mut world = make_world(vec![teacher, learner]);
        let mut rng = StochasticEngine::new(42);

        let (_, summary) = EducationEngine::tick(&mut world, TEST_TICK, &mut rng);

        assert!(
            (summary.tend_distributed
                - summary.teaching_interactions as f64 * TEACHING_TEND_REWARD)
                .abs()
                < 1e-9,
            "TEND distributed should equal interactions * reward"
        );
    }

    // -------------------------------------------------------------------------
    // Event emission
    // -------------------------------------------------------------------------

    #[test]
    fn test_teaching_event_emitted() {
        let mut teacher = make_agent(0, 30);
        teacher.needs.allostatic_load = 0.1;
        teacher.skills.learn(2, 0.7);

        let learner = make_learner_weak_in(1, 25, 2);

        let mut world = make_world(vec![teacher, learner]);
        let mut rng = StochasticEngine::new(42);

        let (events, summary) = EducationEngine::tick(&mut world, TEST_TICK, &mut rng);

        assert!(
            summary.teaching_interactions > 0,
            "Should have teaching interaction"
        );
        let teaching_events: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == CivEventType::TeachingInteraction)
            .collect();
        assert!(
            !teaching_events.is_empty(),
            "Should emit TeachingInteraction event"
        );
        assert_eq!(teaching_events[0].tick, TEST_TICK);
    }

    #[test]
    fn test_empty_world_no_crash() {
        let mut world = make_world(vec![]);
        let mut rng = StochasticEngine::new(42);

        let (events, summary) = EducationEngine::tick(&mut world, TEST_TICK, &mut rng);

        assert!(events.is_empty());
        assert_eq!(summary.agents_foraging, 0);
        assert_eq!(summary.teaching_interactions, 0);
        assert!(summary.skill_crises.is_empty());
    }
}
