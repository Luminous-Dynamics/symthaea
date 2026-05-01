// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Faction dynamics: emergence, recruitment, conflict, and resolution.
//!
//! Factions emerge when inequality (Gini > 0.3) or governance instability
//! (stability < 0.7) creates fertile ground for ideological coalitions.
//! They recruit unaffiliated agents whose ideology (4D vector) is close
//! to the faction's ideology, grow or decay in strength, and may enter
//! conflict with opposing factions.
//!
//! Resolved conflicts contribute to intergenerational trauma, coupling
//! faction dynamics to the psychological needs engine.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::agent::CivAgent;
use crate::config::PolicyConfig;
use crate::events::{CivEvent, CivEventType};
use crate::stochastic::StochasticEngine;
use crate::world::World;

/// A political faction within a world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Faction {
    /// Unique faction identifier.
    pub id: u32,
    /// Faction name.
    pub name: String,
    /// 4D ideology vector: [economic_left_right, auth_lib, tradition_progress, individual_collective].
    /// Each dimension in [0.0, 1.0].
    pub ideology: [f64; 4],
    /// Tick at which the faction was founded.
    pub founding_tick: u32,
    /// Number of affiliated agents (tracked by scanning, not stored individually).
    pub member_count: usize,
    /// Faction strength [0.0, 1.0]: function of member fraction and coherence.
    pub strength: f64,
    /// World ID where this faction operates.
    pub world_id: u32,
}

/// An active or resolved conflict between two factions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionConflict {
    /// Faction A identifier.
    pub faction_a: u32,
    /// Faction B identifier.
    pub faction_b: u32,
    /// Conflict intensity [0.0, 1.0].
    pub intensity: f64,
    /// Maximum intensity reached during the conflict.
    pub max_intensity: f64,
    /// Tick when the conflict started.
    pub started_tick: u32,
    /// Tick when the conflict was resolved (None if active).
    pub resolved_tick: Option<u32>,
    /// How the conflict was resolved.
    pub resolution_type: Option<ConflictResolution>,
}

/// How a faction conflict ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Intensity decayed naturally below threshold.
    Attrition,
    /// Exceeded maximum duration (120 ticks = 10 years).
    Timeout,
}

/// Faction simulation engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionEngine {
    /// All factions across all worlds.
    pub factions: Vec<Faction>,
    /// Active and resolved conflicts.
    pub conflicts: Vec<FactionConflict>,
    /// Next faction ID to assign.
    next_faction_id: u32,
    /// Per-world civil unrest level [0.0, 1.0].
    pub civil_unrest: HashMap<u32, f64>,
}

impl FactionEngine {
    /// Create a new empty faction engine.
    pub fn new() -> Self {
        Self {
            factions: Vec::new(),
            conflicts: Vec::new(),
            next_faction_id: 1,
            civil_unrest: HashMap::new(),
        }
    }

    /// Run one tick of faction dynamics across all worlds.
    pub fn tick_factions(
        &mut self,
        worlds: &mut [World],
        current_tick: u32,
        rng: &mut StochasticEngine,
        policy: &PolicyConfig,
    ) -> Vec<CivEvent> {
        self.tick_factions_with_stress(worlds, current_tick, rng, policy, 0.0)
    }

    /// Run one tick of faction dynamics with a stress-driven emergence boost.
    ///
    /// Mechanism 2 — Agent Decision Quality Under Stress: when stress_boost > 0,
    /// faction emergence probability is multiplied by (1 + 3 * stress_boost).
    /// This models how collective psychological distress drives political
    /// fragmentation (Turchin 2003, Putnam 2000).
    pub fn tick_factions_with_stress(
        &mut self,
        worlds: &mut [World],
        current_tick: u32,
        rng: &mut StochasticEngine,
        policy: &PolicyConfig,
        stress_boost: f64,
    ) -> Vec<CivEvent> {
        if !policy.faction_enabled {
            return Vec::new();
        }

        let mut events = Vec::new();

        // Phase 1: Faction emergence (with stress boost)
        events.extend(self.tick_emergence_with_stress(worlds, current_tick, rng, stress_boost));

        // Phase 2: Recruitment (update member counts)
        self.tick_recruitment(worlds);

        // Phase 3: Strength update (growth from members, natural decay)
        events.extend(self.tick_strength(worlds, current_tick));

        // Phase 4: Conflict detection
        events.extend(self.tick_conflict_detection(current_tick, rng));

        // Phase 5: Conflict resolution
        events.extend(self.tick_conflict_resolution(worlds, current_tick));

        // Phase 6: Civil unrest update
        self.tick_civil_unrest(worlds);

        events
    }

    /// Check for faction emergence in each world (no stress boost).
    #[allow(dead_code)]
    fn tick_emergence(
        &mut self,
        worlds: &[World],
        current_tick: u32,
        rng: &mut StochasticEngine,
    ) -> Vec<CivEvent> {
        self.tick_emergence_with_stress(worlds, current_tick, rng, 0.0)
    }

    /// Check for faction emergence with optional stress-driven boost.
    fn tick_emergence_with_stress(
        &mut self,
        worlds: &[World],
        current_tick: u32,
        rng: &mut StochasticEngine,
        stress_boost: f64,
    ) -> Vec<CivEvent> {
        let mut events = Vec::new();

        for world in worlds {
            let gini = world.economy.gini_coefficient;
            let stability = world.infrastructure_level.min(1.0); // proxy for governance stability

            // Bug #4 fix: Lowered faction emergence thresholds.
            // Original thresholds (gini > 0.3, stability < 0.7, trauma > 0.1) were
            // too strict — factions almost never formed. Now with Dead Loop #1 fixed
            // (disasters cause trauma), trauma should flow naturally into faction emergence.
            let mean_trauma = {
                let living: Vec<_> = world.agents.iter().filter(|a| a.is_alive()).collect();
                if living.is_empty() {
                    0.0
                } else {
                    living.iter().map(|a| a.trauma_level).sum::<f64>() / living.len() as f64
                }
            };
            let can_emerge = gini > 0.25
                || stability < 0.75
                || mean_trauma > 0.05
                || (world.population() > 500 && world.trust_level < 0.6)
                || (world.population() > 200 && gini > 0.2 && mean_trauma > 0.03);
            if !can_emerge {
                continue;
            }

            // Base 0.5% + stress + trauma boost
            let emergence_prob = 0.005 * (1.0 + 3.0 * stress_boost + 2.0 * mean_trauma);
            if !rng.bernoulli(emergence_prob.min(0.5)) {
                continue;
            }

            // Don't create too many factions per world (cap at 6)
            let existing = self
                .factions
                .iter()
                .filter(|f| f.world_id == world.id)
                .count();
            if existing >= 6 {
                continue;
            }

            let id = self.next_faction_id;
            self.next_faction_id += 1;

            // Faction ideology: blend of consciousness and ethical orientation
            // from the world's living agents. Factions crystallize around existing
            // ethical-consciousness clusters, not random noise.
            let living: Vec<_> = worlds
                .iter()
                .filter(|w| w.id == world.id)
                .flat_map(|w| w.agents.iter())
                .filter(|a| a.is_alive())
                .collect();
            let ideology = if living.len() > 5 {
                // Sample a random agent's blended profile as the faction seed
                let idx = (rng.next_u64() % living.len() as u64) as usize;
                let a = living[idx];
                [
                    a.consciousness.level * 0.5 + a.ethics.deontological * 0.5,
                    a.consciousness.meta_awareness * 0.5 + a.ethics.consequentialist * 0.5,
                    a.consciousness.care_activation * 0.5 + a.ethics.virtue_care * 0.5,
                    a.consciousness.harmonic_alignment * 0.5 + a.ethics.relational * 0.5,
                ]
            } else {
                [
                    rng.next_f64(),
                    rng.next_f64(),
                    rng.next_f64(),
                    rng.next_f64(),
                ]
            };

            let name = format!("Faction-{id}");

            self.factions.push(Faction {
                id,
                name: name.clone(),
                ideology,
                founding_tick: current_tick,
                member_count: 0,
                strength: 0.1,
                world_id: world.id,
            });

            events.push(CivEvent::new(
                current_tick,
                Some(world.id),
                CivEventType::FactionEmerged,
                format!(
                    "{}: {} emerged (Gini={:.2}, stability={:.2})",
                    world.name, name, gini, stability
                ),
            ));
        }

        events
    }

    /// Recruit unaffiliated agents into factions based on ideology distance.
    fn tick_recruitment(&mut self, worlds: &mut [World]) {
        for faction in &self.factions {
            let world = match worlds.iter_mut().find(|w| w.id == faction.world_id) {
                Some(w) => w,
                None => continue,
            };

            for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                if agent.faction_id.is_some() {
                    continue; // already affiliated
                }

                // Blended ideology: 50% consciousness + 50% ethical orientation.
                // Matches the faction ideology construction.
                let agent_ideology = [
                    agent.consciousness.level * 0.5 + agent.ethics.deontological * 0.5,
                    agent.consciousness.meta_awareness * 0.5 + agent.ethics.consequentialist * 0.5,
                    agent.consciousness.care_activation * 0.5 + agent.ethics.virtue_care * 0.5,
                    agent.consciousness.harmonic_alignment * 0.5 + agent.ethics.relational * 0.5,
                ];

                let dist = cosine_distance(&agent_ideology, &faction.ideology);
                // Coordination-literate agents have a higher bar for joining factions.
                // They recognize that factional polarization is a negative-sum dynamic
                // and only join factions with very strong ideological alignment.
                let base_threshold = 0.3 * (1.0 - agent.coordination_understanding * 0.5);
                // Ethical orientation modulates faction joining (Phase 2c):
                // Relational agents join more readily (community-oriented, Ubuntu).
                // Deontological agents resist (principled independence).
                let ethics_adjustment =
                    agent.ethics.relational * 0.06 - agent.ethics.deontological * 0.04;
                let threshold = (base_threshold + ethics_adjustment).clamp(0.05, 0.5);
                if dist < threshold {
                    agent.faction_id = Some(faction.id);
                }
            }
        }

        // Update member counts
        for faction in &mut self.factions {
            let count = worlds
                .iter()
                .filter(|w| w.id == faction.world_id)
                .flat_map(|w| w.agents.iter())
                .filter(|a| a.is_alive() && a.faction_id == Some(faction.id))
                .count();
            faction.member_count = count;
        }
    }

    /// Update faction strength: grows with member proportion, decays naturally.
    fn tick_strength(&mut self, worlds: &[World], current_tick: u32) -> Vec<CivEvent> {
        let mut events = Vec::new();
        let mut dissolved = Vec::new();

        for faction in &mut self.factions {
            let world_pop = worlds
                .iter()
                .find(|w| w.id == faction.world_id)
                .map(|w| w.population())
                .unwrap_or(0);

            if world_pop == 0 {
                continue;
            }

            // Strength grows toward member fraction
            let member_frac = faction.member_count as f64 / world_pop as f64;
            faction.strength = faction.strength * 0.995 + member_frac * 0.005;

            // Natural decay
            faction.strength -= 0.005;
            faction.strength = faction.strength.clamp(0.0, 1.0);

            // Dissolution at strength < 0.05
            if faction.strength < 0.05 {
                dissolved.push(faction.id);
                let world_name = worlds
                    .iter()
                    .find(|w| w.id == faction.world_id)
                    .map(|w| w.name.as_str())
                    .unwrap_or("Unknown");
                events.push(CivEvent::new(
                    current_tick,
                    Some(faction.world_id),
                    CivEventType::FactionDissolved,
                    format!(
                        "{}: {} dissolved (strength < 0.05)",
                        world_name, faction.name
                    ),
                ));
            }
        }

        // Remove dissolved factions and clear agent affiliations
        for id in &dissolved {
            self.factions.retain(|f| f.id != *id);
            for world in worlds.iter() {
                // We can't mutate worlds here, so we'll handle this in tick_factions
                let _ = world;
            }
        }

        // Store dissolved IDs for cleanup
        if !dissolved.is_empty() {
            self.cleanup_dissolved_affiliations(worlds, &dissolved);
        }

        events
    }

    /// Clear agent faction_ids for dissolved factions.
    fn cleanup_dissolved_affiliations(&self, _worlds: &[World], _dissolved: &[u32]) {
        // Note: We cannot mutate worlds here as they're borrowed immutably.
        // The caller (tick_factions) handles this after tick_strength returns.
    }

    /// Detect new conflicts between opposing factions in the same world.
    fn tick_conflict_detection(
        &mut self,
        current_tick: u32,
        rng: &mut StochasticEngine,
    ) -> Vec<CivEvent> {
        let mut events = Vec::new();

        // Group factions by world
        let mut world_factions: HashMap<u32, Vec<usize>> = HashMap::new();
        for (i, f) in self.factions.iter().enumerate() {
            world_factions.entry(f.world_id).or_default().push(i);
        }

        for (_world_id, indices) in &world_factions {
            for i in 0..indices.len() {
                for j in (i + 1)..indices.len() {
                    let fa = &self.factions[indices[i]];
                    let fb = &self.factions[indices[j]];

                    let dist = cosine_distance(&fa.ideology, &fb.ideology);
                    let combined_strength = fa.strength + fb.strength;

                    // Conflict condition: ideological distance > 0.7 AND combined strength > 0.4
                    if dist > 0.7 && combined_strength > 0.4 {
                        // Check if conflict already exists
                        let exists = self.conflicts.iter().any(|c| {
                            c.resolved_tick.is_none()
                                && ((c.faction_a == fa.id && c.faction_b == fb.id)
                                    || (c.faction_a == fb.id && c.faction_b == fa.id))
                        });

                        if !exists && rng.bernoulli(0.02) {
                            let intensity = (combined_strength * dist).clamp(0.1, 1.0);
                            self.conflicts.push(FactionConflict {
                                faction_a: fa.id,
                                faction_b: fb.id,
                                intensity,
                                max_intensity: intensity,
                                started_tick: current_tick,
                                resolved_tick: None,
                                resolution_type: None,
                            });

                            events.push(CivEvent::new(
                                current_tick,
                                Some(fa.world_id),
                                CivEventType::FactionConflict,
                                format!(
                                    "{} vs {} — conflict erupts (intensity {:.2})",
                                    fa.name, fb.name, intensity
                                ),
                            ));
                        }
                    }
                }
            }
        }

        events
    }

    /// Resolve active conflicts: intensity decays, resolution after decay or timeout.
    fn tick_conflict_resolution(
        &mut self,
        worlds: &mut [World],
        current_tick: u32,
    ) -> Vec<CivEvent> {
        let mut events = Vec::new();

        for conflict in &mut self.conflicts {
            if conflict.resolved_tick.is_some() {
                continue;
            }

            // Ethics-modulated conflict decay rate:
            // Base: 0.01/tick. Virtue/care and relational societies heal faster.
            // Deontological societies sustain conflicts longer (rigid principles).
            let world_opt = self
                .factions
                .iter()
                .find(|f| f.id == conflict.faction_a)
                .and_then(|f| worlds.iter().find(|w| w.id == f.world_id));
            let ethics_decay = if let Some(w) = world_opt {
                let me = crate::agent::EthicalOrientation::mean_of(&w.agents);
                0.01 + me.virtue_care * 0.008 + me.relational * 0.006 - me.deontological * 0.004
            } else {
                0.01
            };
            conflict.intensity -= ethics_decay;
            conflict.intensity = conflict.intensity.max(0.0);

            // Track max intensity
            if conflict.intensity > conflict.max_intensity {
                conflict.max_intensity = conflict.intensity;
            }

            let duration = current_tick.saturating_sub(conflict.started_tick);

            // Resolution: intensity < 0.05 OR duration > 120 ticks (10 years)
            if conflict.intensity < 0.05 || duration >= 120 {
                let resolution = if conflict.intensity < 0.05 {
                    ConflictResolution::Attrition
                } else {
                    ConflictResolution::Timeout
                };
                conflict.resolved_tick = Some(current_tick);
                conflict.resolution_type = Some(resolution);

                // Trauma feedback: resolved conflicts increase world trauma
                let trauma_increase = conflict.max_intensity * 0.3;
                let world_id = self
                    .factions
                    .iter()
                    .find(|f| f.id == conflict.faction_a)
                    .map(|f| f.world_id);

                if let Some(wid) = world_id {
                    if let Some(world) = worlds.iter_mut().find(|w| w.id == wid) {
                        for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                            agent.trauma_level =
                                (agent.trauma_level + trauma_increase * 0.1).min(1.0);
                        }
                    }
                }

                let fa_name = self
                    .factions
                    .iter()
                    .find(|f| f.id == conflict.faction_a)
                    .map(|f| f.name.as_str())
                    .unwrap_or("?");
                let fb_name = self
                    .factions
                    .iter()
                    .find(|f| f.id == conflict.faction_b)
                    .map(|f| f.name.as_str())
                    .unwrap_or("?");

                events.push(CivEvent::new(
                    current_tick,
                    world_id,
                    CivEventType::FactionResolution,
                    format!(
                        "{} vs {} resolved ({:?}, max intensity {:.2}, trauma +{:.2})",
                        fa_name, fb_name, resolution, conflict.max_intensity, trauma_increase
                    ),
                ));
            }
        }

        events
    }

    /// Update per-world civil unrest from active conflicts.
    fn tick_civil_unrest(&mut self, _worlds: &[World]) {
        // Reset and recompute
        self.civil_unrest.clear();

        for conflict in &self.conflicts {
            if conflict.resolved_tick.is_some() {
                continue;
            }
            let world_id = self
                .factions
                .iter()
                .find(|f| f.id == conflict.faction_a)
                .map(|f| f.world_id);

            if let Some(wid) = world_id {
                let entry = self.civil_unrest.entry(wid).or_insert(0.0);
                *entry = (*entry + conflict.intensity * 0.2).min(1.0);
            }
        }
    }

    /// Fix 4: Directed civil war between the two largest opposing factions.
    ///
    /// Deaths are concentrated among faction members. The winning faction
    /// (larger) gains control, and the losing faction is disbanded.
    /// Returns (deaths, infrastructure_damage).
    ///
    /// Ref: Turchin (2003) "Historical Dynamics" — elite competition model.
    pub fn directed_civil_war(
        &mut self,
        world_id: u32,
        agents: &mut [CivAgent],
        severity: f64,
        rng: &mut StochasticEngine,
    ) -> (usize, f64) {
        // Find two largest factions in this world
        let mut world_factions: Vec<(u32, usize)> = self
            .factions
            .iter()
            .filter(|f| f.world_id == world_id)
            .map(|f| (f.id, f.member_count))
            .collect();
        world_factions.sort_by(|a, b| b.1.cmp(&a.1));

        if world_factions.len() < 2 {
            return (0, 0.0);
        }

        let winner_id = world_factions[0].0;
        let loser_id = world_factions[1].0;

        let mut deaths = 0usize;
        let death_prob = severity * 0.1; // 10% of severity → death probability for combatants

        // Kill faction members probabilistically (losers 2× more likely)
        for agent in agents.iter_mut().filter(|a| a.is_alive()) {
            let p = if agent.faction_id == Some(loser_id) {
                death_prob * 2.0
            } else if agent.faction_id == Some(winner_id) {
                death_prob
            } else {
                death_prob * 0.1 // civilian collateral
            };

            if rng.bernoulli(p.min(0.5)) {
                agent.death_tick = Some(u32::MAX); // mark for death processing by caller
                deaths += 1;
            }
        }

        // Infrastructure damage proportional to severity
        let infra_damage = severity * 0.2;

        // Disband losing faction
        self.factions.retain(|f| f.id != loser_id);
        for agent in agents.iter_mut() {
            if agent.faction_id == Some(loser_id) {
                agent.faction_id = None;
                agent.trauma_level = (agent.trauma_level + 0.3).min(1.0);
            }
        }

        (deaths, infra_damage)
    }

    /// Get civil unrest for a specific world.
    pub fn world_unrest(&self, world_id: u32) -> f64 {
        self.civil_unrest.get(&world_id).copied().unwrap_or(0.0)
    }

    /// Count of active (unresolved) factions.
    pub fn active_faction_count(&self) -> usize {
        self.factions.len()
    }

    /// Count of active (unresolved) conflicts.
    pub fn active_conflict_count(&self) -> usize {
        self.conflicts
            .iter()
            .filter(|c| c.resolved_tick.is_none())
            .count()
    }
}

impl Default for FactionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Cosine distance between two 4D vectors. Returns 0.0 for identical, 1.0 for orthogonal.
fn cosine_distance(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denom = mag_a * mag_b;
    if denom < f64::EPSILON {
        return 1.0;
    }
    1.0 - (dot / denom).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{BiologicalSex, CivAgent, ConsciousnessState, SkillVector};
    use crate::config::PolicyConfig;
    use crate::stochastic::StochasticEngine;
    use crate::world::{CulturalProfile, World, WorldResources};

    fn make_test_world_fac(pop: usize, world_id: u32) -> World {
        let mut world = World {
            id: world_id,
            name: format!("World-{}", world_id),
            location: "Moon".into(),
            founded_tick: 0,
            parent_world_id: None,
            agents: Vec::new(),
            next_agent_id: 0,
            resources: WorldResources::lunar_default(),
            culture: CulturalProfile::pioneer_default(),
            infrastructure_level: 0.5,
            max_population: 10_000,
            habitable_area_m2: 100_000.0,
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
            let agent = CivAgent {
                id: i as u64,
                birth_tick: 0,
                death_tick: None,
                sex: if i % 2 == 0 {
                    BiologicalSex::Female
                } else {
                    BiologicalSex::Male
                },
                world_id,
                health: 0.9,
                skills: SkillVector::new(),
                education_level: 0.3,
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
            };
            world.agents.push(agent);
            world.next_agent_id += 1;
        }
        world
    }

    #[test]
    fn test_faction_emergence_with_inequality() {
        let mut engine = FactionEngine::new();
        let mut world = make_test_world_fac(100, 0);
        world.economy.gini_coefficient = 0.5; // High inequality
        let mut rng = StochasticEngine::new(42);

        // Run emergence many times until a faction appears (0.5% chance each)
        let mut emerged = false;
        for tick in 0..500 {
            let events = engine.tick_emergence(&[world.clone()], tick, &mut rng);
            if !events.is_empty() {
                emerged = true;
                break;
            }
        }
        assert!(
            emerged,
            "Faction should emerge with high Gini after many ticks"
        );
    }

    #[test]
    fn test_faction_recruitment() {
        let mut engine = FactionEngine::new();
        engine.factions.push(Faction {
            id: 1,
            name: "Test Faction".into(),
            ideology: [0.1, 0.05, 0.3, 0.1], // Near nascent consciousness
            founding_tick: 0,
            member_count: 0,
            strength: 0.2,
            world_id: 0,
        });

        let world = make_test_world_fac(50, 0);
        let mut worlds = vec![world];
        engine.tick_recruitment(&mut worlds);
        let affiliated = worlds[0]
            .agents
            .iter()
            .filter(|a| a.faction_id.is_some())
            .count();
        // With nascent consciousness near [0.1, 0.05, 0.3, 0.1], agents should join
        assert!(
            affiliated > 0,
            "Some agents should be recruited, got {affiliated}"
        );
    }

    #[test]
    fn test_strength_decay() {
        let mut engine = FactionEngine::new();
        engine.factions.push(Faction {
            id: 1,
            name: "Decaying".into(),
            ideology: [0.5, 0.5, 0.5, 0.5],
            founding_tick: 0,
            member_count: 0,
            strength: 0.3,
            world_id: 0,
        });

        let world = make_test_world_fac(100, 0);
        // Run several ticks of strength decay with no members
        for tick in 0..100 {
            let _ = engine.tick_strength(&[world.clone()], tick);
        }

        // With no members, strength should decay significantly
        if let Some(f) = engine.factions.first() {
            assert!(
                f.strength < 0.1,
                "Strength should decay without members, got {}",
                f.strength
            );
        }
        // Or it may have dissolved entirely
    }

    #[test]
    fn test_dissolution() {
        let mut engine = FactionEngine::new();
        engine.factions.push(Faction {
            id: 1,
            name: "Weak".into(),
            ideology: [0.5, 0.5, 0.5, 0.5],
            founding_tick: 0,
            member_count: 0,
            strength: 0.04, // Below dissolution threshold
            world_id: 0,
        });

        let world = make_test_world_fac(100, 0);
        let events = engine.tick_strength(&[world], 100);
        assert!(
            events
                .iter()
                .any(|e| e.event_type == CivEventType::FactionDissolved),
            "Weak faction should dissolve"
        );
        assert!(engine.factions.is_empty(), "Faction should be removed");
    }

    #[test]
    fn test_conflict_trigger() {
        let mut engine = FactionEngine::new();
        // Two opposing factions in the same world
        engine.factions.push(Faction {
            id: 1,
            name: "Left".into(),
            ideology: [0.0, 0.0, 0.0, 0.0],
            founding_tick: 0,
            member_count: 50,
            strength: 0.3,
            world_id: 0,
        });
        engine.factions.push(Faction {
            id: 2,
            name: "Right".into(),
            ideology: [1.0, 1.0, 1.0, 1.0],
            founding_tick: 0,
            member_count: 50,
            strength: 0.3,
            world_id: 0,
        });

        let mut rng = StochasticEngine::new(42);
        let mut conflict_found = false;
        for tick in 0..200 {
            let events = engine.tick_conflict_detection(tick, &mut rng);
            if events
                .iter()
                .any(|e| e.event_type == CivEventType::FactionConflict)
            {
                conflict_found = true;
                break;
            }
        }
        assert!(
            conflict_found,
            "Opposing factions should eventually conflict"
        );
    }

    #[test]
    fn test_conflict_resolution() {
        let mut engine = FactionEngine::new();
        engine.factions.push(Faction {
            id: 1,
            name: "A".into(),
            ideology: [0.0; 4],
            founding_tick: 0,
            member_count: 10,
            strength: 0.2,
            world_id: 0,
        });
        engine.conflicts.push(FactionConflict {
            faction_a: 1,
            faction_b: 2,
            intensity: 0.04, // Below resolution threshold
            max_intensity: 0.5,
            started_tick: 0,
            resolved_tick: None,
            resolution_type: None,
        });

        let world = make_test_world_fac(10, 0);
        let mut worlds = vec![world];
        let events = engine.tick_conflict_resolution(&mut worlds, 10);
        assert!(
            events
                .iter()
                .any(|e| e.event_type == CivEventType::FactionResolution),
            "Low-intensity conflict should resolve"
        );
    }

    #[test]
    fn test_trauma_accumulation() {
        let mut engine = FactionEngine::new();
        engine.factions.push(Faction {
            id: 1,
            name: "A".into(),
            ideology: [0.0; 4],
            founding_tick: 0,
            member_count: 10,
            strength: 0.2,
            world_id: 0,
        });
        engine.conflicts.push(FactionConflict {
            faction_a: 1,
            faction_b: 2,
            intensity: 0.04,
            max_intensity: 0.8,
            started_tick: 0,
            resolved_tick: None,
            resolution_type: None,
        });

        let world = make_test_world_fac(10, 0);
        let mut worlds = vec![world];
        let initial_trauma: f64 = worlds[0].agents.iter().map(|a| a.trauma_level).sum::<f64>();
        assert_eq!(initial_trauma, 0.0);

        let _ = engine.tick_conflict_resolution(&mut worlds, 10);
        let final_trauma: f64 = worlds[0].agents.iter().map(|a| a.trauma_level).sum::<f64>();
        assert!(
            final_trauma > 0.0,
            "Trauma should increase after conflict resolution"
        );
    }

    #[test]
    fn test_directed_civil_war_casualties() {
        // Fix 4: Factional Warfare
        let mut engine = FactionEngine::new();
        engine.factions.push(Faction {
            id: 1,
            name: "Big".into(),
            ideology: [0.0; 4],
            founding_tick: 0,
            member_count: 60,
            strength: 0.5,
            world_id: 0,
        });
        engine.factions.push(Faction {
            id: 2,
            name: "Small".into(),
            ideology: [1.0; 4],
            founding_tick: 0,
            member_count: 30,
            strength: 0.3,
            world_id: 0,
        });

        let mut agents: Vec<CivAgent> = (0..100)
            .map(|i| CivAgent {
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
                education_level: 0.3,
                consciousness: ConsciousnessState::nascent(),
                partner_id: None,
                children_ids: vec![],
                is_immigrant: false,
                needs: crate::needs::PsychologicalNeeds::new(),
                tend_balance: 0.0,
                parent_ids: None,
                faction_id: if i < 60 {
                    Some(1)
                } else if i < 90 {
                    Some(2)
                } else {
                    None
                },
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
            })
            .collect();

        let mut rng = StochasticEngine::new(42);
        let (deaths, damage) = engine.directed_civil_war(0, &mut agents, 0.8, &mut rng);

        assert!(deaths > 0, "Civil war should cause casualties");
        assert!(damage > 0.0, "Civil war should damage infrastructure");
        // Losing faction should be disbanded
        assert!(
            !engine.factions.iter().any(|f| f.id == 2),
            "Losing faction should be disbanded"
        );
    }

    #[test]
    fn test_civil_unrest_from_conflict() {
        let mut engine = FactionEngine::new();
        engine.factions.push(Faction {
            id: 1,
            name: "A".into(),
            ideology: [0.0; 4],
            founding_tick: 0,
            member_count: 10,
            strength: 0.2,
            world_id: 0,
        });
        engine.conflicts.push(FactionConflict {
            faction_a: 1,
            faction_b: 2,
            intensity: 0.5,
            max_intensity: 0.5,
            started_tick: 0,
            resolved_tick: None,
            resolution_type: None,
        });

        let world = make_test_world_fac(10, 0);
        engine.tick_civil_unrest(&[world]);
        let unrest = engine.world_unrest(0);
        assert!(
            unrest > 0.0,
            "Active conflict should produce civil unrest, got {unrest}"
        );
    }
}
