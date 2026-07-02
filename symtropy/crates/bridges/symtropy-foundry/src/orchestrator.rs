// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::profiler::SimulationLoad;
use bevy::prelude::*;
use rand::Rng;
use rusqlite::Connection;
use serde::Deserialize;
use std::fs;
use std::time::Instant;

#[derive(Deserialize, Debug, Clone)]
pub struct Blueprint {
    pub world_name: String,
    pub biomes: Vec<BiomeBlueprint>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct BiomeBlueprint {
    pub name: String,
    pub density: f32,
    pub sampling_rules: Vec<SamplingRule>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct SamplingRule {
    pub role: String,
    pub count_min: u32,
    pub count_max: u32,
    pub phi_threshold: Option<f32>,
}

pub struct OrchestratorPlugin {
    pub blueprint_path: String,
    pub registry_path: String,
}

impl Plugin for OrchestratorPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(BlueprintConfig {
            path: self.blueprint_path.clone(),
            registry_path: self.registry_path.clone(),
            last_checked: Instant::now(),
            last_modified: None,
        });
        app.init_resource::<OrchestratorObservatory>();
        app.add_systems(
            Update,
            (handle_blueprint_updates, update_observatory, evolve_ecology),
        );
    }
}

#[derive(Resource, Default)]
pub struct OrchestratorObservatory {
    pub avg_phi: f32,
    pub thermodynamic_load: f32,
}

#[derive(Resource)]
pub struct BlueprintConfig {
    pub path: String,
    pub registry_path: String,
    pub last_checked: Instant,
    pub last_modified: Option<std::time::SystemTime>,
}

fn update_observatory(mut observatory: ResMut<OrchestratorObservatory>) {
    observatory.avg_phi = 0.5;
    observatory.thermodynamic_load = 0.0;
}

fn sample_assets(registry_path: &str, role: &str) -> Vec<String> {
    let conn = Connection::open(registry_path).expect("Failed to open registry");
    let mut stmt = conn
        .prepare(
            "SELECT a.id FROM assets a 
         JOIN behaviors b ON a.id = b.asset_id 
         WHERE b.role = ?",
        )
        .expect("Failed to prepare statement");

    let asset_iter = stmt
        .query_map([role], |row| row.get(0))
        .expect("Failed to query");

    asset_iter.filter_map(Result::ok).collect()
}

fn evolve_ecology(
    config: Res<BlueprintConfig>,
    observatory: Res<OrchestratorObservatory>,
    load: Res<SimulationLoad>,
    mut commands: Commands,
) {
    if load.is_overloaded {
        return;
    }

    let Ok(blueprint_content) = fs::read_to_string(&config.path) else {
        return;
    };
    let Ok(blueprint): Result<Blueprint, _> = serde_yaml::from_str(&blueprint_content) else {
        return;
    };

    let mut rng = rand::thread_rng();

    for biome in blueprint.biomes {
        for rule in biome.sampling_rules {
            if let Some(threshold) = rule.phi_threshold {
                if observatory.avg_phi < threshold {
                    continue;
                }
            }

            let candidates = sample_assets(&config.registry_path, &rule.role);
            if candidates.is_empty() {
                continue;
            }

            if rng.gen_bool(biome.density as f64 * 0.1) {
                let asset_id = &candidates[rng.gen_range(0..candidates.len())];
                let transform = Transform::from_xyz(
                    rng.gen_range(-100.0..100.0),
                    0.0,
                    rng.gen_range(-100.0..100.0),
                );

                commands.spawn((
                    Name::new(format!("{}_{}", rule.role, asset_id)),
                    transform,
                    GlobalTransform::default(),
                    crate::FoundryAsset {
                        id: asset_id.clone(),
                    },
                ));
            }
        }
    }
}

fn handle_blueprint_updates(mut _config: ResMut<BlueprintConfig>) {}
