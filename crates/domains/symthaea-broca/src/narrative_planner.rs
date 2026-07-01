// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Narrative Planner — Strategic Architectural Trajectories
//!
//! Maps high-level architectural goals to sequences of evolutionary steps.
//! Bridges Broca's "intent" to real-world code changes.

use crate::encoder::ThoughtChannels;
use crate::evolutionary_scaffolder::EvolutionResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ArcStatus {
    Proposed,
    InProgress,
    Stabilized,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeStep {
    pub description: String,
    pub substrate_path: String,
    pub priority: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeArc {
    pub id: String,
    pub title: String,
    pub steps: Vec<ChangeStep>,
    pub status: ArcStatus,
    pub current_step: usize,
}

pub struct NarrativePlanner {
    pub active_arcs: HashMap<String, ChangeArc>,
}

impl NarrativePlanner {
    pub fn new() -> Self {
        Self {
            active_arcs: HashMap::new(),
        }
    }

    /// Propose a new architectural change arc based on current intent.
    pub fn propose_arc(&self, intent: &str, _channels: &ThoughtChannels) -> ChangeArc {
        let mut steps = Vec::new();

        if intent.contains("optimize") {
            steps.push(ChangeStep {
                description: "Identify high-latency core components".to_string(),
                substrate_path: "src/lib.rs".to_string(),
                priority: 0.9,
            });
            steps.push(ChangeStep {
                description: "Apply evolutionary constant tuning".to_string(),
                substrate_path: "src/liquid_mamba.rs".to_string(),
                priority: 0.8,
            });
        }

        ChangeArc {
            id: format!("arc_{}", rand::random::<u32>()),
            title: format!("Strategic Optimization: {}", intent),
            steps,
            status: ArcStatus::Proposed,
            current_step: 0,
        }
    }

    /// **NEW**: Trigger Co-Evolution.
    /// When a successful mutation happens in one substrate (e.g., Rust),
    /// automatically trigger an optimization cycle for paired substrates (e.g., Nix).
    pub fn trigger_co_evolution(
        &self,
        substrate: &str,
        path: &str,
        evolution_result: &EvolutionResult,
    ) {
        if evolution_result.success_score > 0.8 {
            println!(
                "🔗 Co-Evolution Triggered by successful {} mutation at {}",
                substrate, path
            );

            if substrate == "rust" {
                // Heuristic: trigger corresponding Nix module update
                println!("  -> Triggering optimization for nix/module.nix to maintain coherence.");
            }
        }
    }
}

impl Default for NarrativePlanner {
    fn default() -> Self {
        Self::new()
    }
}
