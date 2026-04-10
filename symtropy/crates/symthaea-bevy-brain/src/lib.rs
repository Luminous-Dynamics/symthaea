// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-bevy-brain
//!
//! Drop-in Bevy plugin that gives any entity a cognitive architecture.
//! Not a behavior tree — a real neural system with HDC perception,
//! CfC temporal dynamics, predictive processing, and Phi metrics.
//!
//! ```ignore
//! use symthaea_bevy_brain::{SymthaeaBrainPlugin, CognitiveBrain};
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(DefaultPlugins)
//!         .add_plugins(SymthaeaBrainPlugin::default())
//!         .add_systems(Startup, spawn_npc)
//!         .run();
//! }
//!
//! fn spawn_npc(mut commands: Commands) {
//!     commands.spawn((
//!         Transform::default(),
//!         CognitiveBrain::new(64, "npc_guard"),
//!     ));
//! }
//! ```

use bevy::prelude::*;
use symthaea::cognitive_loop::{
    CognitiveLoopBuilder,
    CognitiveLoopService,
    CycleResult,
};

/// Bevy component: attach to any entity to give it a cognitive loop.
///
/// Each brain has its own HDC state (16,384D), CfC neural network,
/// episodic memory, prediction history, and Phi computation.
///
/// Feed perception via `brain.perception_input` each frame.
/// Read outputs from `brain.last_result`.
#[derive(Component)]
pub struct CognitiveBrain {
    /// The cognitive loop (perception → dynamics → feedback → output).
    loop_service: CognitiveLoopService,
    /// Result from the last cognitive cycle.
    pub last_result: Option<CycleResult>,
    /// What this brain perceives this tick. Set by your perception system.
    pub perception_input: String,
    /// Ticks since last cognitive cycle (for amortized scheduling).
    ticks_since_cycle: u32,
    /// How many physics ticks between cognitive cycles.
    pub cycle_interval: u32,
}

impl CognitiveBrain {
    /// Create a brain with the given CfC network size and a deterministic seed.
    ///
    /// `cfc_neurons`: network capacity (32 = lightweight, 128 = full cognition)
    /// `genesis_phrase`: deterministic seed for reproducible behavior
    pub fn new(cfc_neurons: usize, genesis_phrase: &str) -> Self {
        let service = CognitiveLoopBuilder::new()
            .with_cfc_neurons(cfc_neurons)
            .with_genesis_phrase(genesis_phrase)
            .build()
            .expect("failed to build cognitive loop");

        Self {
            loop_service: service,
            last_result: None,
            perception_input: String::new(),
            ticks_since_cycle: 0,
            cycle_interval: 3, // cognitive cycle every 3 physics ticks (~20Hz at 64Hz physics)
        }
    }

    /// Create from a pre-built CognitiveLoopService.
    pub fn from_service(service: CognitiveLoopService) -> Self {
        Self {
            loop_service: service,
            last_result: None,
            perception_input: String::new(),
            ticks_since_cycle: 0,
            cycle_interval: 3,
        }
    }

    /// Current Phi (integrated information) estimate. 0.0 if not yet computed.
    pub fn phi(&self) -> f64 {
        self.loop_service.consciousness_level() as f64
    }

    /// Prediction error from the last cycle. Higher = more surprise.
    pub fn prediction_error(&self) -> f32 {
        self.last_result
            .as_ref()
            .map(|r| r.prediction_error)
            .unwrap_or(0.0)
    }

    /// Language output from the last cycle (if Broca generated text).
    pub fn language(&self) -> Option<&str> {
        self.last_result
            .as_ref()
            .and_then(|r| r.language_output.as_deref())
    }

    /// Whether learning occurred in the last cycle.
    pub fn learned(&self) -> bool {
        self.last_result
            .as_ref()
            .map(|r| r.learning_occurred)
            .unwrap_or(false)
    }

    /// Run one cognitive cycle with the current perception input.
    fn cycle(&mut self) {
        let result = self.loop_service.cycle(&self.perception_input);
        self.last_result = Some(result);
    }
}

/// Plugin configuration.
pub struct SymthaeaBrainPlugin {
    /// Default CfC neurons for brains created without explicit size.
    pub default_neurons: usize,
    /// Whether to log consciousness telemetry.
    pub telemetry: bool,
}

impl Default for SymthaeaBrainPlugin {
    fn default() -> Self {
        Self {
            default_neurons: 64,
            telemetry: false,
        }
    }
}

impl Plugin for SymthaeaBrainPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(FixedUpdate, cognitive_cycle_system);
        if self.telemetry {
            app.add_systems(Update, telemetry_system);
        }
    }
}

/// Core system: runs the cognitive cycle for every entity with a CognitiveBrain.
///
/// Uses amortized scheduling: each brain cycles every `cycle_interval` ticks,
/// not every frame. At cycle_interval=3 and 64Hz physics, cognition runs at ~21Hz.
fn cognitive_cycle_system(mut brains: Query<&mut CognitiveBrain>) {
    for mut brain in &mut brains {
        brain.ticks_since_cycle += 1;
        if brain.ticks_since_cycle >= brain.cycle_interval {
            brain.ticks_since_cycle = 0;
            brain.cycle();
        }
    }
}

/// Optional telemetry: logs Phi and prediction error for each brain.
fn telemetry_system(brains: Query<(Entity, &CognitiveBrain)>) {
    for (entity, brain) in &brains {
        if let Some(ref result) = brain.last_result {
            debug!(
                "Brain {:?}: Phi={:.4}, PE={:.4}, learned={}, cycle_time={}us",
                entity,
                brain.phi(),
                result.prediction_error,
                result.learning_occurred,
                result.cycle_time_us,
            );
            if let Some(ref text) = result.language_output {
                info!("Brain {:?} says: {}", entity, text);
            }
        }
    }
}

// Physics coupling: users wire brain.phi() into their own physics system.
// See the symtropy-bevy crate for SymtropyPhysics<D> and PhysicsBody.
//
// Example:
// ```ignore
// fn sync_brain_to_physics(
//     brains: Query<(&CognitiveBrain, &PhysicsBody)>,
//     mut physics: ResMut<SymtropyPhysics<2>>,
// ) {
//     for (brain, body) in &brains {
//         physics.field.set_metric(body.handle, brain.phi());
//     }
// }
// ```
