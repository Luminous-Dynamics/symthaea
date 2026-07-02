// SPDX-License-Identifier: AGPL-3.0-or-later

// symtropy-mycelix-village/src/headless.rs
// M3 Scenario Harness — deterministic governance validation (headless)

use bevy::app::AppExit;
use bevy::ecs::event::EventWriter;
use bevy::prelude::*; // Explicitly import this!

use symthaea_bevy_brain::{CognitiveBrain, SymthaeaBrainPlugin};
use symtropy_bevy_core::BevyPhysicsCorePlugin;

#[derive(Resource)]
struct ScenarioStats {
    ticks: u32,
    max_ticks: u32,
}

fn main() {
    App::new()
        .add_plugins(MinimalPlugins)
        .add_plugins(bevy::log::LogPlugin::default())
        .add_plugins(bevy::transform::TransformPlugin)
        .add_plugins(bevy::hierarchy::HierarchyPlugin)
        .add_plugins(bevy::diagnostic::DiagnosticsPlugin)
        .add_plugins(BevyPhysicsCorePlugin::<3> {
            gravity: nalgebra::SVector::from([0.0, -9.81, 0.0]),
        })
        .add_plugins(SymthaeaBrainPlugin {
            default_neurons: 32,
            telemetry: true,
        })
        .insert_resource(ScenarioStats {
            ticks: 0,
            max_ticks: 1000,
        })
        .add_systems(Startup, spawn_scenario)
        .add_systems(Update, tick_scenario)
        .add_systems(Update, wastefulness_monitor_system)
        .add_systems(Update, validate_governance_invariants)
        .run();
}

fn spawn_scenario(mut commands: Commands) {
    info!("M3 Scenario: Spawning NPCs for moral resonance validation...");

    // NPC-0: Virtuous
    let mut brain_0 = CognitiveBrain::new(32, "Reciprocity is the core of being.");
    brain_0.profile.stewardship_care = 1.0;
    brain_0.profile.epistemic_integrity = 1.0;
    commands.spawn((
        brain_0,
        symtropy_bevy_core::PhysicsBody {
            handle: symtropy_physics::body::BodyHandle(0),
            visual_radius: 0.5,
        },
        Transform::from_xyz(0.0, 1.0, 0.0),
        GlobalTransform::default(),
    ));

    // NPC-1: Wasteful
    let mut brain_1 = CognitiveBrain::new(32, "Consumption is expansion.");
    brain_1.profile.stewardship_care = 1.0;
    brain_1.profile.epistemic_integrity = 1.0;
    commands.spawn((
        brain_1,
        symtropy_bevy_core::PhysicsBody {
            handle: symtropy_physics::body::BodyHandle(1),
            visual_radius: 0.5,
        },
        Transform::from_xyz(5.0, 1.0, 0.0),
        GlobalTransform::default(),
    ));
}

fn wastefulness_monitor_system(
    mut query: Query<(&symtropy_bevy_core::PhysicsBody, &mut CognitiveBrain)>,
) {
    for (body, mut brain) in &mut query {
        let power = if body.handle.0 == 1 { 500.0 } else { 0.0 };
        let phi = brain.phi();

        if power > 0.0 {
            let wastefulness = (power / (phi + 0.1)).min(10.0);
            let penalty = (wastefulness * 0.001) as f64;
            brain.profile.stewardship_care = (brain.profile.stewardship_care - penalty).max(0.0);
        }
    }
}

fn tick_scenario(mut stats: ResMut<ScenarioStats>, mut app_exit: EventWriter<AppExit>) {
    stats.ticks += 1;
    if stats.ticks >= stats.max_ticks {
        info!("Scenario complete after {} ticks.", stats.ticks);
        app_exit.send(AppExit::Success);
    }
}

fn validate_governance_invariants(
    query: Query<(&symtropy_bevy_core::PhysicsBody, &CognitiveBrain)>,
    stats: Res<ScenarioStats>,
) {
    if stats.ticks % 100 != 0 {
        return;
    }

    for (body, brain) in &query {
        let profile = &brain.profile;

        if body.handle.0 == 1 && stats.ticks > 500 {
            if profile.stewardship_care > 0.95 {
                error!(
                    "NPC-1: Stewardship Care should be dropping! Current: {:.4}",
                    profile.stewardship_care
                );
            } else {
                info!(
                    "NPC-1: Stewardship Care is correctly dropping: {:.4}",
                    profile.stewardship_care
                );
            }
        }

        if body.handle.0 == 0 {
            if profile.stewardship_care < 0.8 {
                error!(
                    "NPC-0: Stewardship Care dropped too low: {:.4}",
                    profile.stewardship_care
                );
            }
        }
    }
}
