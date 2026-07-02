// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `symtropy-foundry` — Engine-side smart ingestion for Asset Foundry assets.

use bevy::prelude::*;
use bevy_camera::visibility::VisibilityRange;
use serde::Deserialize;
use std::collections::HashMap;
use symtropy_bevy_core::{BevyPhysics, PhysicsBody};
use symtropy_math::{Capsule, HyperBox, Point, Shape, Sphere};

/// Plugin that enables automatic processing of Foundry asset tags (`_COLLISION`, `_LOD`).
pub struct FoundryPlugin {
    pub index_path: Option<String>,
}

impl Default for FoundryPlugin {
    fn default() -> Self {
        Self { index_path: None }
    }
}

impl Plugin for FoundryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FoundryRegistry>();

        if let Some(_path) = &self.index_path {
            // Placeholder for registry loading
        }

        app.add_systems(Update, process_foundry_assets);
    }
}

/// Resource containing the asset index from Foundry.
#[derive(Resource, Default)]
pub struct FoundryRegistry {
    pub assets_by_path: HashMap<String, AssetMetadata>,
    pub assets_by_id: HashMap<String, AssetMetadata>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct AssetMetadata {
    pub id: String,
    pub path: String,
    pub title: String,
    pub material_family: Option<String>,
    pub biome: Option<String>,
    pub mass_kg: Option<f64>,
    pub com_x: Option<f64>,
    pub com_y: Option<f64>,
    pub com_z: Option<f64>,
    pub inertia_ixx: Option<f64>,
    pub inertia_iyy: Option<f64>,
    pub inertia_izz: Option<f64>,
}

/// Component added to a root entity to mark it for Foundry post-processing.
#[derive(Component)]
pub struct FoundryAsset {
    pub id: String,
}

/// Marker to avoid double-processing.
#[derive(Component)]
struct FoundryProcessed;

/// System that scans for newly spawned entities with Foundry tags and wires them up.
fn process_foundry_assets(
    mut commands: Commands,
    // We look for Added<Name> to catch them as soon as they enter the world (e.g. from a Scene spawn)
    new_entities: Query<
        (Entity, &Name, &GlobalTransform),
        (Without<FoundryProcessed>, Added<Name>),
    >,
    mut physics: Option<ResMut<BevyPhysics<3>>>,
) {
    for (entity, name, global_transform) in &new_entities {
        let name_str = name.as_str();

        // 1. Handle Collision Proxies
        if let Some(collider_type) = parse_collider_type(name_str) {
            if let Some(ref mut py) = physics {
                let (scale, _rotation, _) = global_transform.to_scale_rotation_translation();

                let collider: Box<dyn Shape<3>> = match collider_type {
                    "BOX" => {
                        let half_extents = scale * 0.5;
                        Box::new(HyperBox::new([
                            half_extents.x as f64,
                            half_extents.y as f64,
                            half_extents.z as f64,
                        ]))
                    }
                    "SPHERE" => {
                        let radius = scale.max_element() * 0.5;
                        Box::new(Sphere::new(Point::origin(), radius as f64))
                    }
                    "CAPSULE" => {
                        let radius = (scale.x.max(scale.z)) * 0.5;
                        let half_height = scale.y * 0.5;
                        Box::new(Capsule::new(half_height as f64, radius as f64, 1)) // Y-axis
                    }
                    "CONVEX" | _ => {
                        // Fallback to sphere for now
                        let radius = scale.max_element() * 0.5;
                        Box::new(Sphere::new(Point::origin(), radius as f64))
                    }
                };

                let pos = global_transform.translation();
                let handle = py.world.add_body(symtropy_physics::RigidBody::static_body(
                    symtropy_physics::BodyHandle(0), // Will be allocated
                    Point::new([pos.x as f64, pos.y as f64, pos.z as f64]),
                    collider,
                ));

                if let Some(body) = py.world.body_mut(handle) {
                    body.collision_mask = 0xFFFF;
                }

                commands.entity(entity).insert((
                    PhysicsBody::new(handle, scale.max_element() * 0.5),
                    Visibility::Hidden,
                    FoundryProcessed,
                ));
                continue;
            }
        }

        // 2. Handle LODs
        if let Some(lod_index) = parse_lod_index(name_str) {
            // Distances from Symtropy_Bevy_Rendering_Strategy_v0.2.md
            // LOD0: 0-20m, LOD1: 20-60m, LOD2: 60-150m, LOD3+: 150m+
            let range = match lod_index {
                0 => VisibilityRange::abrupt(0.0, 20.0),
                1 => VisibilityRange::abrupt(20.0, 60.0),
                2 => VisibilityRange::abrupt(60.0, 150.0),
                _ => VisibilityRange::abrupt(150.0, 1000.0),
            };

            commands.entity(entity).insert((range, FoundryProcessed));
            continue;
        }

        // If it didn't match any special tags, mark it as processed so we don't check it again.
        commands.entity(entity).insert(FoundryProcessed);
    }
}

fn parse_lod_index(name: &str) -> Option<u32> {
    if let Some(idx) = name.find("_LOD") {
        let suffix = &name[idx + 4..];
        let digits: String = suffix.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse::<u32>().ok()
    } else {
        None
    }
}

fn parse_collider_type(name: &str) -> Option<&'static str> {
    if name.contains("_COLLISION_BOX") {
        Some("BOX")
    } else if name.contains("_COLLISION_SPHERE") {
        Some("SPHERE")
    } else if name.contains("_COLLISION_CAPSULE") {
        Some("CAPSULE")
    } else if name.contains("_COLLISION_CONVEX") {
        Some("CONVEX")
    } else if name.contains("_COLLISION") {
        Some("CONVEX")
    }
    // Default
    else {
        None
    }
}

pub mod ledger_ui;
pub mod narrative;
pub mod nexus_telemetry;
pub mod orchestrator;
pub mod profiler;
pub mod seedworks_bootstrap;
pub mod snapshot;

pub use ledger_ui::*;
pub use narrative::*;
pub use nexus_telemetry::*;
pub use orchestrator::*;
pub use profiler::*;
pub use seedworks_bootstrap::*;
pub use snapshot::*;

// --- Experimental / Quarantined Modules ---
// The following modules are research prototypes and are not yet
// part of the stable Seedworks v0.1 pipeline.
#[cfg(feature = "experimental")]
pub mod bio_feedback;
#[cfg(feature = "experimental")]
pub mod cognition;
#[cfg(feature = "experimental")]
pub mod evolution;
#[cfg(feature = "experimental")]
pub mod fitness;
#[cfg(feature = "experimental")]
pub mod ghost_memory;
#[cfg(feature = "experimental")]
pub mod mesh_synthesis;

#[cfg(test)]
mod tests;
