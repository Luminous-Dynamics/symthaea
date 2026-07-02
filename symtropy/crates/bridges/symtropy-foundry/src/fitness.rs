// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::orchestrator::{BlueprintConfig, OrchestratorObservatory};
use crate::profiler::SimulationLoad;
use bevy::prelude::*;

#[derive(Resource, Default)]
pub struct EpochFitness {
    pub score: f32,
    pub epoch: u32,
}

pub struct FitnessPlugin;

impl Plugin for FitnessPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EpochFitness>()
            .add_systems(Update, calculate_fitness);
    }
}

/// Evaluates the success of the current epoch.
/// Fitness = (Avg Φ * Stability) / (Entropy Load)
fn calculate_fitness(
    observatory: Res<OrchestratorObservatory>,
    load: Res<SimulationLoad>,
    mut fitness: ResMut<EpochFitness>,
) {
    let stability = if load.is_overloaded { 0.1 } else { 1.0 };
    let score = (observatory.avg_phi * stability) / (observatory.thermodynamic_load + 0.1);

    fitness.score = score;
    // Log fitness for the evolutionary engine to pick up later
    debug!("Epoch Fitness: {:.2}", fitness.score);
}
