use bevy::prelude::*;
use rapier3d::prelude::*;
use symtropy_rapier3d_bridge::{RapierColliderSet, RapierRigidBodySet};
use symtropy_robotics_bridge::RoboticAgentTag;

/// Marker for objects that can be scavenged for materials.
#[derive(Component, Default)]
pub struct Reclaimable {
    pub integrity: f32, // 1.0 = solid, 0.0 = fully scavenged
    pub material_type: String,
}

/// System to handle material recovery via fracture physics.
pub fn scavenger_fracture_system(
    mut commands: Commands,
    _rb_set: ResMut<RapierRigidBodySet>,
    _col_set: ResMut<RapierColliderSet>,
    scavengers: Query<(&RoboticAgentTag, &Transform)>,
    mut targets: Query<(Entity, &mut Reclaimable, &Transform)>,
) {
    for (_agent, agent_transform) in scavengers.iter() {
        for (entity, mut reclaimable, target_transform) in targets.iter_mut() {
            let dist = agent_transform
                .translation
                .distance(target_transform.translation);

            // Basic proximity-based scavenging for the demo
            if dist < 1.5 && reclaimable.integrity > 0.0 {
                reclaimable.integrity -= 0.01;

                info!(
                    "Scavenging target {:?}: Integrity {:.2}",
                    entity, reclaimable.integrity
                );

                if reclaimable.integrity <= 0.0 {
                    info!("Target fully reclaimed! Despawning...");
                    commands.entity(entity).despawn();
                    // In a full implementation, this would spawn "scrap" debris
                }
            }
        }
    }
}

pub struct ScavengerPhysicsPlugin;

impl Plugin for ScavengerPhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(FixedUpdate, scavenger_fracture_system);
    }
}
