// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::orchestrator::{BlueprintConfig, OrchestratorObservatory};
use crate::profiler::SimulationLoad;
use bevy::prelude::*;

pub struct CognitionManagerPlugin;

impl Plugin for CognitionManagerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, run_cognitive_introspection);
    }
}

/// System that analyzes simulation state and outputs "Evolutionary Directives"
/// for the orchestrator to adapt the world-blueprint.
fn run_cognitive_introspection(
    observatory: Res<OrchestratorObservatory>,
    load: Res<SimulationLoad>,
    _config: Res<BlueprintConfig>,
) {
    if observatory.avg_phi > 0.8 && !load.is_overloaded {
        info!(
            "Cognition: Φ is high and resources stable. Recommending expansion of complex nodes."
        );
        // Logic would call an internal blueprint modifier here to increase density
    } else if load.is_overloaded {
        info!("Cognition: Overload detected. Recommending structural pruning.");
    }
}
