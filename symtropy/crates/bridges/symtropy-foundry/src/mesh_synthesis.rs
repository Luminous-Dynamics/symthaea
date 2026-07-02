// SPDX-License-Identifier: AGPL-3.0-or-later

use bevy::prelude::*;

pub struct HoloMeshPlugin;

impl Plugin for HoloMeshPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, update_holo_meshes);
    }
}

#[derive(Component)]
pub struct HoloMeshTarget {
    pub source: Entity,
    pub target: Entity,
}

fn update_holo_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    query: Query<(Entity, &HoloMeshTarget, &Transform, &Transform)>,
) {
    for (entity, _target, _t1, _t2) in &query {
        // Fallback to simpler mesh creation using available Bevy types
        let mesh = Mesh::from(Sphere::default());

        commands.entity(entity).insert((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(materials.add(StandardMaterial {
                emissive: LinearRgba::from(Color::srgba(0.0, 1.0, 0.5, 1.0)),
                ..default()
            })),
        ));
    }
}
