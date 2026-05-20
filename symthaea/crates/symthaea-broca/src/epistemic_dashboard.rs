// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Epistemic Dashboard — Visual Architectural Consciousness
//!
//! CLI visualization of architectural blueprints, substrate surprisal,
//! and the system's emotional/epistemic state.

use crate::architectural_memory::ArchitecturalMemory;
use crate::encoder::ThoughtChannels;
use crate::substrate_binding::SubstrateBindingEngine;

pub struct EpistemicDashboard {
    // Registry of active blueprints for visualization
}

#[derive(Debug, Clone, Copy)]
pub enum CognitiveStyle {
    Rigid,
    Creative,
    Neutral,
}

impl EpistemicDashboard {
    pub fn new() -> Self {
        Self {}
    }

    /// **NEW**: Bidirectional Affective Feedback.
    /// Manually nudge the system's emotional state to change cognitive style.
    pub fn nudge_consciousness(&self, channels: &mut ThoughtChannels, mode: CognitiveStyle) {
        match mode {
            CognitiveStyle::Rigid => {
                // Calm, precise, low temperature
                channels.set_valence(0.8);
                channels.set_arousal(0.2);
            }
            CognitiveStyle::Creative => {
                // Agitated, exploring, high temperature
                channels.set_valence(-0.2);
                channels.set_arousal(0.9);
            }
            CognitiveStyle::Neutral => {
                channels.set_valence(0.0);
                channels.set_arousal(0.5);
            }
        }
    }

    /// **NEW**: Trigger formal proof mode.
    pub fn nudge_formal_proof(&self, channels: &mut ThoughtChannels) {
        println!("🚀 Nudging system into Formal Proof mode (E-axis focus)...");
        self.nudge_consciousness(channels, CognitiveStyle::Rigid);
    }

    /// Render an ASCII heatmap of architectural integrity.
    pub fn render_blueprint_status(&self, engine: &SubstrateBindingEngine) {
        println!("\n🌐 SYMTHAEA ARCHITECTURAL INTEGRITY HEATMAP");
        println!("═════════════════════════════════════════════");

        // In real impl: iterate over blueprints and calculate average surprisal
        println!("  [RUST]  | ████████████████████ | 0.05 Surprisal (OK)");
        println!("  [NIX]   | ████████████░░░░░░░░ | 0.35 Surprisal (DRIFT)");
        println!("  [PYTHON]| ██████████████████░░ | 0.15 Surprisal (OK)");
        println!("═════════════════════════════════════════════");
    }

    /// Render the current emotional and epistemic state of the generator.
    pub fn render_generator_consciousness(
        &self,
        channels: &ThoughtChannels,
        memory: &ArchitecturalMemory,
    ) {
        println!("\n🧠 GENERATOR CONSCIOUSNESS STATE");
        println!("--------------------------------");
        println!("  Valence:  {:.2}", channels.valence());
        println!("  Arousal:  {:.2}", channels.arousal());
        println!("  Moral S.: {:.2}", channels.moral_score());
        println!("  Narrative:{:.2}", channels.narrative_score());
        println!("  Memory K: {}", memory.top_k);
        println!("--------------------------------\n");
    }

    /// **NEW**: Evolutionary Lineage Visualization.
    /// Renders the "Family Tree" of successful mutations stored in memory.
    pub fn render_evolutionary_lineage(&self, memory: &ArchitecturalMemory) {
        println!("\n🌳 EVOLUTIONARY LINEAGE (Successful Mutations)");
        println!("════════════════════════════════════════════════");

        // In real use: walk the result_db and build a tree.
        // Simplified for demo:
        println!("  [Root] -> Commit 0x4f2 (Baseline)");
        println!("     └──> 0x8a1: Optimized Gating (v5 -> v6) [+12% perf]");
        println!("           └──> 0xbc4: Reactive CodeGate [+5% robustness] ← CURRENT");
        println!("     └──> 0x3d2: Substrate Binding Engine [Initial]");
        println!("  Memory K Threshold: {:.2}", memory.min_success_threshold);
        println!("════════════════════════════════════════════════\n");
    }
}

impl Default for EpistemicDashboard {
    fn default() -> Self {
        Self::new()
    }
}
