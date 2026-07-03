use crate::physics::{Reclaimable, ScavengerPhysicsPlugin};
use bevy::prelude::*;
use symthaea_bevy_brain::CognitiveBrain;
use symtropy_physics::body::BodyHandle;
use symtropy_robotics_bridge::{RoboticBrainPlugin, spawn_robot};

pub struct ScavengerDemoPlugin;

impl Plugin for ScavengerDemoPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((RoboticBrainPlugin, ScavengerPhysicsPlugin))
            .add_systems(Startup, setup_scavenger_scene);
    }
}

fn setup_scavenger_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    info!("Initializing Scavenger agent scene...");

    // 1. Spawn a Scavenger agent via the bridge
    let brain = CognitiveBrain::new(128, "scavenger_genesis");
    let agent_id = spawn_robot(
        &mut commands,
        "Scavenger Mk1",
        Vec3::new(0.0, 1.0, 0.0),
        brain,
        BodyHandle(0),
    );

    // Add visual components to the spawned robot
    commands.entity(agent_id).insert((
        Mesh3d(meshes.add(Sphere::new(0.8).mesh())),
        MeshMaterial3d(materials.add(Color::srgb(0.2, 0.8, 0.2))),
    ));

    // 2. Spawn some reclaimable targets (scrap metal piles)
    for i in 1..=3 {
        commands.spawn((
            Reclaimable {
                integrity: 1.0,
                material_type: "Scrap Metal".into(),
            },
            Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0).mesh())),
            MeshMaterial3d(materials.add(Color::srgb(0.5, 0.5, 0.5))),
            Transform::from_xyz(i as f32 * 3.0, 0.5, 0.0),
        ));
    }

    // 3. Add environment (ground plane)
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(50.0, 50.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.3, 0.3))),
    ));

    // 4. Light and Camera
    commands.spawn((
        DirectionalLight {
            illuminance: 5000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-10.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}
