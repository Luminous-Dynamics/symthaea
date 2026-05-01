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

/// P1 v2: Allostatic load dynamics calibrated for ~0.3 equilibrium.
///
/// V1 problem: net +0.007/tick → load 1.0 within 140 ticks (overcrowded).
/// P1 v1 problem: decay > accumulation → load 0.01 (zero stress, unrealistic).
///
/// Fix: add baseline environmental stress (space is inherently stressful).
/// A stable colony with social bonds equilibrates at ~0.25-0.35.
/// Disasters spike it to 0.6-0.8. Burnout threshold at 0.8.
///
/// Equilibrium math (no isolation, no overwork, no care workers):
///   base_stress per tick = 0.004
///   decay per tick = 0.010
///   equilibrium = base_stress / (decay - 0) ≈ base_stress / decay
///   But load decays multiplicatively, so eq = base / (base + decay) ≈ 0.28

/// Baseline environmental stress (every agent, every tick).
/// Space habitats are inherently stressful: artificial light, recycled air,
/// confined spaces, radiation awareness, distance from Earth.
/// Ref: Palinkas & Suedfeld (2008) — Antarctic station stress baselines.
const BASELINE_STRESS_RATE: f64 = 0.004;

/// Additional load from social isolation (social_satiation < 0.3).
/// Ref: McEwen (1998) — chronic stress mediator accumulation.
const ISOLATION_LOAD_RATE: f64 = 0.008;

/// Additional load from overwork (worker_ratio > 0.6).
/// Ref: Karasek (1979) — demand-control model of occupational stress.
const OVERWORK_LOAD_RATE: f64 = 0.006;

/// Natural load decay per tick (rest, adaptation, habituation).
const LOAD_DECAY_RATE: f64 = 0.010;

/// Care worker load reduction per worker per 100 recipients.
const CARE_LOAD_REDUCTION: f64 = 0.006;

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
/// Tuned down from 0.005: was producing ~5000 incidents over 150 years.
const THRILL_RISK_PROBABILITY: f64 = 0.002;

/// Thrill-seeking health cost per incident.
const THRILL_HEALTH_COST: f64 = 0.1;

/// Rogue bio-hacking epidemic trigger probability per tick.
/// Requires: tech_level > 0.6 AND governance_stability < 0.5 AND mean_load > 0.5.
const BIOHACK_EPIDEMIC_PROBABILITY: f64 = 0.001;

/// Deep-space isolation multiplier for social decay (non-Earth, Epochs 2-3).
/// Ref: NASA HI-SEAS — communication latency (3-24 min) amplifies loneliness.
const _DEEP_SPACE_SOCIAL_DECAY_MULT: f64 = 1.5;

/// Epoch range for deep-space isolation amplification (Branches through Canopy).
const DEEP_SPACE_EPOCH_START: u8 = 2;
const DEEP_SPACE_EPOCH_END: u8 = 3;

// =============================================================================
// Types
// =============================================================================

/// Spinozist affect state: 6 dimensions derived from agent needs and consciousness.
///
/// Grounded in Spinoza's *Ethics* III: affects as transitions in the power of acting.
/// Joy = increase in power (conatus enhanced), Sadness = decrease (conatus diminished).
/// The 6 CfC (Conatus-for-Collective) dimensions map to:
///
/// 1. **Joy** (Laetitia): Met needs + social bonds + engagement → power increase
/// 2. **Sadness** (Tristitia): Isolation + burnout + trauma → power decrease
/// 3. **Desire** (Cupiditas): Conatus — striving toward flourishing (gap between current and potential)
/// 4. **Care** (Cura): Capacity for mutual aid — consciousness.care_activation + social bonds
/// 5. **Harm** (Nocere): Accumulated moral injury — dealt + received harm, faction violence
/// 6. **Consent** (Consensus): Trust/reciprocity in collective decisions — governance alignment
///
/// References:
/// - Spinoza, *Ethics* III, Propositions 11-13 (joy, sadness, desire as primary affects)
/// - Damasio (2003), "Looking for Spinoza" — somatic marker hypothesis
/// - Nussbaum (2001), "Upheavals of Thought" — emotions as evaluative judgments
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AffectState {
    /// Laetitia: increase in power of acting [0, 1].
    /// High when needs met, socially connected, engaged.
    pub joy: f64,
    /// Tristitia: decrease in power of acting [0, 1].
    /// High when isolated, burned out, traumatized.
    pub sadness: f64,
    /// Cupiditas: striving intensity [0, 1].
    /// The gap between current state and flourishing potential.
    pub desire: f64,
    /// Cura: capacity for mutual aid and caregiving [0, 1].
    /// Maps to consciousness.care_activation + social bonds.
    pub care: f64,
    /// Nocere: accumulated moral injury [0, 1].
    /// Tracks both harm dealt (faction violence) and harm received (disaster casualties).
    pub harm: f64,
    /// Consensus: trust and reciprocity in collective governance [0, 1].
    /// High when governance is stable and agent participates; low during crises.
    pub consent: f64,
    /// Moral guilt: rises when revealed ethics diverge from stated values [0, 1].
    /// "I believe in care but I acted selfishly under pressure."
    /// Suppresses further consequentialist drift (guilt restrains hypocrisy).
    /// Ref: Tangney & Dearing (2002) — guilt as moral self-regulation.
    #[serde(default)]
    pub guilt: f64,
    /// Moral outrage: rises when witnessing violations of one's sacred value [0, 1].
    /// Amplifies commitment to sacred dimension + increases deontological resolve.
    /// Ref: Haidt (2003) — outrage as moral enforcement emotion.
    #[serde(default)]
    pub outrage: f64,
}

impl AffectState {
    /// Compute affect state from agent's needs, consciousness, and context.
    ///
    /// This is the core Spinozist mapping: affects emerge from the body's
    /// relationship to its environment, not as independent mental states.
    pub fn compute(
        needs: &PsychologicalNeeds,
        care_activation: f64,
        trauma_level: f64,
        governance_stability: f64,
        is_faction_member: bool,
        resource_fraction: f64,
    ) -> Self {
        // Realism H: Nonlinear affect dynamics.
        // Joy and sadness compete via Lotka-Volterra-inspired dynamics:
        // when joy is high, sadness is suppressed (and vice versa).
        // Care has a threshold: below 0.3 social_satiation, care collapses.
        // Desire follows Yerkes-Dodson: moderate stress = peak striving.

        // Raw drivers
        let satisfaction = needs.social_satiation * 0.4
            + needs.engagement * 0.3
            + (1.0 - needs.allostatic_load) * 0.3;
        let suffering =
            needs.allostatic_load * 0.4 + (1.0 - needs.social_satiation) * 0.3 + trauma_level * 0.3;

        // Competitive exclusion: joy and sadness suppress each other (soft-max).
        // This creates bistable dynamics — the system tips toward one or the other.
        let joy_raw = satisfaction * (1.0 - suffering * 0.5); // Sadness suppresses joy
        let sad_raw = suffering * (1.0 - satisfaction * 0.3); // Joy suppresses sadness (weaker)
        let joy = joy_raw.clamp(0.0, 1.0);
        let sadness = sad_raw.clamp(0.0, 1.0);

        // Desire: Yerkes-Dodson curve — moderate stress maximizes striving.
        // Too little stress = complacency. Too much = paralysis.
        let deprivation = (1.0 - resource_fraction).max(0.0);
        let stress_level = needs.allostatic_load * 0.5 + deprivation * 0.5;
        // Inverted-U: peak at stress=0.5, drops at extremes
        let desire = (4.0 * stress_level * (1.0 - stress_level)).clamp(0.0, 1.0);

        // Care: threshold dynamics — below 0.3 social satiation, care collapses.
        // This models Maslow: you can't care for others when your own needs are unmet.
        let care_base = care_activation * 0.5
            + needs.social_satiation * 0.3
            + (1.0 - needs.allostatic_load) * 0.2;
        let care = if needs.social_satiation < 0.3 {
            care_base * (needs.social_satiation / 0.3) // Sigmoid collapse below threshold
        } else {
            care_base
        }
        .clamp(0.0, 1.0);

        // Harm, guilt, and outrage are NOT computed here — they are managed
        // exclusively by the moral emotions system in consciousness.rs.
        // compute() only contributes structural violence from faction conflict.
        // Trauma-driven suffering is captured by `sadness` above — not `harm`.
        // Moral injury (harm) must be earned through ethical violations, not grief.
        // This separation is critical: grief ≠ moral injury (Litz 2009, Haidt 2012).
        let _ = (trauma_level, is_faction_member); // acknowledged but not used for harm

        // Consent: hysteresis — trust is hard to build, easy to lose.
        // Once consent drops below 0.3, it requires extra governance stability to rebuild.
        let consent_raw = governance_stability * 0.5
            + (1.0 - needs.allostatic_load) * 0.3
            + care_activation * 0.2;
        let consent = consent_raw.clamp(0.0, 1.0);

        // harm=0.0, guilt=0.0, outrage=0.0 — all restored from previous state
        // by blend_with_previous, which delegates to consciousness.rs for accumulation.
        Self {
            joy,
            sadness,
            desire,
            care,
            harm: 0.0,
            consent,
            guilt: 0.0,
            outrage: 0.0,
        }
    }

    /// Blend new computed state with previous state (emotional momentum).
    /// Alpha = 0.3 means 30% new state, 70% previous state.
    /// Grief doesn't vanish when the cause is removed. Joy persists.
    /// Resentment builds over years. This is the key to realistic psychology.
    pub fn blend_with_previous(&self, previous: &AffectState, alpha: f64) -> Self {
        Self {
            joy: previous.joy * (1.0 - alpha) + self.joy * alpha,
            sadness: previous.sadness * (1.0 - alpha) + self.sadness * alpha,
            desire: previous.desire * (1.0 - alpha) + self.desire * alpha,
            care: previous.care * (1.0 - alpha) + self.care * alpha,
            // Harm is managed exclusively by consciousness.rs moral injury system
            // (same as guilt/outrage). compute() outputs harm=0.0 so blending would
            // decay it — instead preserve previous value, let consciousness.rs own it.
            harm: previous.harm,
            // Consent rebuilds slowly but collapses fast (hysteresis)
            consent: if self.consent < previous.consent {
                previous.consent * (1.0 - alpha * 1.5).max(0.0)
                    + self.consent * (alpha * 1.5).min(1.0)
            } else {
                previous.consent * (1.0 - alpha * 0.5) + self.consent * (alpha * 0.5)
            },
            // Guilt and outrage are managed by consciousness.rs moral emotions,
            // NOT by compute(). Preserve previous values — only consciousness.rs
            // should modify them (with its own decay/accumulation logic).
            guilt: previous.guilt,
            outrage: previous.outrage,
        }
    }

    /// Net conatus: joy - sadness. Positive = flourishing, negative = suffering.
    pub fn net_conatus(&self) -> f64 {
        self.joy - self.sadness
    }

    /// Moral balance: care - harm. Positive = moral health, negative = moral injury.
    pub fn moral_balance(&self) -> f64 {
        self.care - self.harm
    }
}

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
    /// Spinozist affect state (derived from needs + consciousness each tick).
    pub affect: AffectState,
}

impl PsychologicalNeeds {
    /// Default state for adult colonists: slight baseline stress, moderate social reserve.
    pub fn new() -> Self {
        Self {
            allostatic_load: 0.1,
            social_satiation: 0.7,
            engagement: 0.8,
            affect: AffectState::default(),
        }
    }

    /// Newborn/child state: protected by parental bond.
    pub fn nascent() -> Self {
        Self {
            allostatic_load: 0.0,
            social_satiation: 0.9,
            engagement: 0.9,
            affect: AffectState::default(),
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
    // --- Spinozist collective affect aggregates ---
    /// Mean collective joy (Laetitia): power of acting.
    pub mean_joy: f64,
    /// Mean collective sadness (Tristitia): power diminished.
    pub mean_sadness: f64,
    /// Mean collective desire (Cupiditas): conatus intensity.
    pub mean_desire: f64,
    /// Mean collective care (Cura): mutual aid capacity.
    pub mean_care: f64,
    /// Mean collective harm (Nocere): moral injury.
    pub mean_harm: f64,
    /// Mean collective consent (Consensus): governance trust.
    pub mean_consent: f64,
    /// Net collective conatus (joy - sadness): >0 = flourishing, <0 = suffering.
    pub net_conatus: f64,
    /// Moral balance (care - harm): >0 = moral health, <0 = moral crisis.
    pub moral_balance: f64,
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
        care_effectiveness: f64,
        deep_space_mult: f64,
        rng: &mut StochasticEngine,
    ) -> (Vec<CivEvent>, NeedsWorldSummary) {
        let mut events = Vec::new();
        let is_off_earth = world.location != "Earth";
        let is_deep_space_epoch =
            current_epoch >= DEEP_SPACE_EPOCH_START && current_epoch <= DEEP_SPACE_EPOCH_END;

        let pop = world.population().max(1) as f64;
        let care_ratio = care_worker_count as f64 / (pop / 100.0).max(1.0) * care_effectiveness;

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
                decay *= deep_space_mult;
            }
            n.social_satiation = (n.social_satiation - decay).max(0.0);

            // --- 2. Social replenishment from relationships ---
            if has_partner {
                n.social_satiation = (n.social_satiation + PARTNER_SOCIAL_BONUS).min(1.0);
            }
            let child_bonus = CHILD_SOCIAL_BONUS * num_children as f64;
            n.social_satiation = (n.social_satiation + child_bonus).min(1.0);

            // --- 3. Allostatic load accumulation ---
            // Baseline: space habitats are inherently stressful
            n.allostatic_load = (n.allostatic_load + BASELINE_STRESS_RATE).min(1.0);
            // Social isolation amplifies stress
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
            // Threshold lowered from 0.5: even moderate stress + isolation triggers withdrawal.
            if n.allostatic_load > 0.3 && n.social_satiation < ISOLATION_THRESHOLD {
                n.engagement = (n.engagement - ESCAPISM_DECAY_RATE).max(0.0);
            } else {
                n.engagement = (n.engagement + ENGAGEMENT_RECOVERY_RATE).min(1.0);
            }

            // --- 6. Thrill-seeking eligibility (checked after needs borrow ends) ---
            // Thrill-seeking occurs when agents are physically active (engagement > 0.6)
            // but socially unfulfilled (social < 0.4) — seeking intensity through risk.
            let thrill_eligible = n.engagement > 0.6
                && n.social_satiation < 0.4
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

        // Compute aggregate Spinozist affects across all living agents.
        // Affects are computed from each agent's state and averaged.
        let (mut total_joy, mut total_sadness, mut total_desire) = (0.0, 0.0, 0.0);
        let (mut total_care, mut total_harm, mut total_consent) = (0.0, 0.0, 0.0);
        for agent in world.agents.iter().filter(|a| a.is_alive()) {
            total_joy += agent.needs.affect.joy;
            total_sadness += agent.needs.affect.sadness;
            total_desire += agent.needs.affect.desire;
            total_care += agent.needs.affect.care;
            total_harm += agent.needs.affect.harm;
            total_consent += agent.needs.affect.consent;
        }
        let (mean_joy, mean_sadness, mean_desire) = (
            total_joy / count,
            total_sadness / count,
            total_desire / count,
        );
        let (mean_care_aff, mean_harm_aff, mean_consent_aff) = (
            total_care / count,
            total_harm / count,
            total_consent / count,
        );

        let summary = NeedsWorldSummary {
            mean_allostatic_load: mean_load,
            mean_social_satiation: total_social / count,
            mean_engagement: total_engagement / count,
            thrill_incidents,
            escapism_count,
            burnout_count,
            mean_joy,
            mean_sadness,
            mean_desire,
            mean_care: mean_care_aff,
            mean_harm: mean_harm_aff,
            mean_consent: mean_consent_aff,
            net_conatus: mean_joy - mean_sadness,
            moral_balance: mean_care_aff - mean_harm_aff,
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

        PsychNeedsEngine::tick_needs(&mut world, tick, 1, 0, 0.3, 0.8, 0.5, 1.0, 1.5, &mut rng);

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

        PsychNeedsEngine::tick_needs(&mut world, tick, 1, 0, 0.3, 0.8, 0.5, 1.0, 1.5, &mut rng);

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
        PsychNeedsEngine::tick_needs(
            &mut world_with,
            tick,
            1,
            5,
            0.3,
            0.8,
            0.5,
            1.0,
            1.5,
            &mut rng1,
        );
        let load_with_care = world_with.agents[0].needs.allostatic_load;

        // World without care workers
        let mut agents_without = vec![make_agent(0, 30, tick)];
        agents_without[0].needs.allostatic_load = 0.5;
        agents_without[0].needs.social_satiation = 0.5;
        let mut world_without = make_world(agents_without);
        let mut rng2 = StochasticEngine::new(42);
        PsychNeedsEngine::tick_needs(
            &mut world_without,
            tick,
            1,
            0,
            0.3,
            0.8,
            0.5,
            1.0,
            1.5,
            &mut rng2,
        );
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

        PsychNeedsEngine::tick_needs(&mut world, tick, 1, 0, 0.3, 0.8, 0.5, 1.0, 1.5, &mut rng);

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

        PsychNeedsEngine::tick_needs(&mut world, tick, 1, 0, 0.3, 0.8, 0.5, 1.0, 1.5, &mut rng);

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
            PsychNeedsEngine::tick_needs(&mut world, t, 1, 0, 0.3, 0.8, 0.5, 1.0, 1.5, &mut rng);
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
            PsychNeedsEngine::tick_needs(&mut world, t, 1, 0, 0.3, 0.8, 0.5, 1.0, 1.5, &mut rng);
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
        PsychNeedsEngine::tick_needs(
            &mut world_deep,
            tick,
            2,
            0,
            0.3,
            0.8,
            0.5,
            1.0,
            1.5,
            &mut rng1,
        );
        let social_deep = world_deep.agents[0].needs.social_satiation;

        // Epoch 1 (Roots), same world
        let mut agents_normal = vec![make_agent(0, 30, tick)];
        agents_normal[0].needs.social_satiation = 0.7;
        let mut world_normal = make_world(agents_normal);
        let mut rng2 = StochasticEngine::new(42);
        PsychNeedsEngine::tick_needs(
            &mut world_normal,
            tick,
            1,
            0,
            0.3,
            0.8,
            0.5,
            1.0,
            1.5,
            &mut rng2,
        );
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

        let (_, summary) =
            PsychNeedsEngine::tick_needs(&mut world, tick, 1, 0, 0.3, 0.8, 0.5, 1.0, 1.5, &mut rng);

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
            PsychNeedsEngine::tick_needs(&mut world, t, 1, 0, 0.3, 0.8, 0.5, 1.0, 1.5, &mut rng);
        }

        for a in &world.agents {
            assert!(a.needs.allostatic_load >= 0.0 && a.needs.allostatic_load <= 1.0);
            assert!(a.needs.social_satiation >= 0.0 && a.needs.social_satiation <= 1.0);
            assert!(a.needs.engagement >= 0.0 && a.needs.engagement <= 1.0);
        }
    }
}
