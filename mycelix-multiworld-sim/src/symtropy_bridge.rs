// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Symtropy Game Bridge: connect the simulation engine to the game.
//!
//! The Symtropy game is set in "Algorithmic Collapse" — centralized AIs
//! consumed Earth's thermodynamic budget. Players rebuild civilization
//! using TEND economics, trust-weighted governance, and the Mycelix
//! fractal architecture.
//!
//! This bridge converts simulation state into game state and player
//! decisions into simulation inputs. The simulation runs in the background;
//! the game presents decisions, narratives, and consequences.
//!
//! # Decision Points
//!
//! The AI governor's automatic decisions become player choices:
//! - What to build next (project selection)
//! - Birth policy (pro-natal, control, replacement)
//! - Trade openness (autarky vs free trade)
//! - Defense vs exploration spending
//! - Independence: declare or negotiate?
//!
//! # Narrative Output
//!
//! The chronicle becomes the game's story. Events are formatted for
//! the game UI with affect-colored text and named character portraits.

use serde::{Deserialize, Serialize};

/// A decision point presented to the player.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerDecision {
    /// When this decision is available (game tick).
    pub tick: u32,
    /// Which world this affects.
    pub world_name: String,
    /// The decision category.
    pub category: DecisionCategory,
    /// Available choices.
    pub choices: Vec<Choice>,
    /// Context: what prompted this decision.
    pub context: String,
    /// Urgency: 0 = routine, 1 = important, 2 = crisis.
    pub urgency: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionCategory {
    /// What to build next.
    ProjectSelection,
    /// Colony birth rate policy.
    BirthPolicy,
    /// Trade terms with another world.
    TradeNegotiation,
    /// Resource allocation between sectors.
    ResourcePriority,
    /// Declare independence or negotiate.
    IndependenceChoice,
    /// Respond to a disaster.
    CrisisResponse,
    /// Exploration mission approval.
    ExplorationMission,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub id: String,
    pub label: String,
    pub description: String,
    /// Expected consequence (shown to player as forecast).
    pub forecast: String,
    /// Risk level: 0 = safe, 1 = moderate, 2 = risky.
    pub risk: u8,
}

/// Game state snapshot for the Symtropy UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameSnapshot {
    pub year: f64,
    pub worlds: Vec<WorldSnapshot>,
    pub pending_decisions: Vec<PlayerDecision>,
    pub recent_events: Vec<NarrativeSnippet>,
    pub cvs: f64,
    pub total_population: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldSnapshot {
    pub name: String,
    pub population: usize,
    pub conatus: f64,
    pub load: f64,
    pub self_sufficiency: f64,
    pub active_projects: Vec<String>,
    pub mood: String, // Derived from collective affect
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeSnippet {
    pub year: f64,
    pub text: String,
    pub severity: u8,
    pub character: Option<String>,
    /// Affect color: "warm" (joy/care), "cold" (sadness), "hot" (desire/harm).
    pub tone: String,
}

/// Generate player decisions from current simulation state.
pub fn generate_decisions(sim: &crate::MultiWorldSimulator) -> Vec<PlayerDecision> {
    let mut decisions = Vec::new();
    let tick = sim.current_tick;
    let _year = tick as f64 / 12.0;

    for world in &sim.worlds {
        if world.population() == 0 {
            continue;
        }

        // Project selection (when queue is empty)
        if world.project_manager.active.is_empty() && world.project_manager.queue.is_empty() {
            let mut choices = Vec::new();
            let priorities = crate::projects::prioritize_projects(
                world.population(),
                world.power_demand_kw - world.power_generation_kw,
                world.resources.fraction_of_capacity("food"),
                world.max_population,
                world.population() > (world.max_population as f64 * 0.8) as usize,
                world
                    .project_manager
                    .has_completed(crate::projects::ProjectBlueprint::MedicalFacility),
                world
                    .project_manager
                    .has_completed(crate::projects::ProjectBlueprint::FabricationWorkshop),
                &world.location,
                &world.project_manager.completed,
            );
            for bp in &priorities {
                choices.push(Choice {
                    id: format!("{:?}", bp),
                    label: bp.name().to_string(),
                    description: format!(
                        "{} months, {} labor/tick",
                        bp.duration(),
                        bp.labor_per_tick()
                    ),
                    forecast: match bp {
                        crate::projects::ProjectBlueprint::FissionReactor => {
                            "Energy independence — everything else becomes possible".into()
                        }
                        crate::projects::ProjectBlueprint::GreenhouseModule => {
                            "Feeds 50 more people — reduces food pressure".into()
                        }
                        crate::projects::ProjectBlueprint::CentrifugeHabitat => {
                            "Enables reproduction — the colony can grow naturally".into()
                        }
                        _ => format!("Expands {} capability", bp.name()),
                    },
                    risk: 0,
                });
            }
            if !choices.is_empty() {
                decisions.push(PlayerDecision {
                    tick,
                    world_name: world.name.clone(),
                    category: DecisionCategory::ProjectSelection,
                    choices,
                    context: format!(
                        "{} has no active construction projects. What should we build?",
                        world.name
                    ),
                    urgency: 1,
                });
            }
        }

        // Independence choice (when pressure > 0.6)
        if world.location != "Earth"
            && world.population() > 5000
            && world.resources.self_sufficiency() > 0.7
        {
            decisions.push(PlayerDecision {
                tick,
                world_name: world.name.clone(),
                category: DecisionCategory::IndependenceChoice,
                choices: vec![
                    Choice {
                        id: "declare".into(),
                        label: "Declare Independence".into(),
                        description: "Sever political ties with Earth. Full sovereignty.".into(),
                        forecast: "Trade disruption likely. Earth may reduce funding. \
                            But self-determination and cultural identity strengthen."
                            .into(),
                        risk: 2,
                    },
                    Choice {
                        id: "negotiate".into(),
                        label: "Negotiate Federation".into(),
                        description: "Propose equal partnership with Earth. Shared governance."
                            .into(),
                        forecast: "Maintains trade routes. Slower cultural independence. \
                            Less risk of conflict."
                            .into(),
                        risk: 1,
                    },
                    Choice {
                        id: "defer".into(),
                        label: "Defer Decision".into(),
                        description: "Not yet. Build more strength first.".into(),
                        forecast: "Status quo continues. Independence pressure may grow.".into(),
                        risk: 0,
                    },
                ],
                context: format!(
                    "{} has {} people and {:.0}% self-sufficiency. \
                    \"We can govern ourselves.\" The question is: should we?",
                    world.name,
                    world.population(),
                    world.resources.self_sufficiency() * 100.0
                ),
                urgency: 2,
            });
        }
    }

    decisions
}

/// Convert simulation narrative events to game snippets.
pub fn narrative_to_snippets(
    engine: &crate::narrative::NarrativeEngine,
    recent_ticks: u32,
    current_tick: u32,
) -> Vec<NarrativeSnippet> {
    engine
        .events
        .iter()
        .filter(|e| e.tick + recent_ticks >= current_tick)
        .map(|e| {
            let tone = if e.joy > 0.5 || e.care > 0.5 {
                "warm"
            } else if e.sadness > 0.5 {
                "cold"
            } else if e.desire > 0.6 {
                "hot"
            } else {
                "neutral"
            };
            NarrativeSnippet {
                year: e.year,
                text: format!("{} — {}", e.crisis, e.outcome),
                severity: e.severity,
                character: e.character.clone(),
                tone: tone.to_string(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_categories() {
        // Verify all categories are distinct
        let cats = [
            DecisionCategory::ProjectSelection,
            DecisionCategory::BirthPolicy,
            DecisionCategory::IndependenceChoice,
        ];
        assert_ne!(cats[0], cats[1]);
        assert_ne!(cats[1], cats[2]);
    }
}
