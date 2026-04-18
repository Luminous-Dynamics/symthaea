//! `{{project_name}}` — Symtropy 3D research scene.
//!
//! One swinging pendulum, PBR rendering, dev console toggleable on F1.
//! Generated from `symtropy new --template 3d-research`.

use symtropy::prelude::*;

const ARM_LENGTH: f64 = 1.0;
const BOB_RADIUS: f32 = 0.10;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "{{project_name}}".into(),
                resolution: bevy::window::WindowResolution::from((1280u32, 720u32)),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(SymtropyScenePlugin::default())
        .add_plugins(SymtropyPhysicsPlugin::<3>::with_gravity([0.0, -9.81, 0.0]))
        .add_plugins(SymtropyDevConsolePlugin)
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut physics: ResMut<SymtropyPhysics<3>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera (light + clear color come from SymtropyScenePlugin).
    commands.spawn(fixed_camera(
        Vec3::new(0.0, 1.5, 4.0),
        Vec3::new(0.0, -0.5, 0.0),
    ));

    // Pivot: a static body at origin (collision off — bob just hangs from it).
    let pivot = physics.world.add_body(RigidBody::<3>::static_body(
        BodyHandle(0),
        Point::new([0.0, 0.0, 0.0]),
        Box::new(PhysicsSphere::new(Point::origin(), 0.01)),
    ));
    if let Some(p) = physics.world.body_mut(pivot) {
        p.collision_mask = 0;
    }

    // Bob: dynamic sphere released from horizontal so it swings under gravity.
    let bob = physics
        .world
        .add_sphere(Point::new([ARM_LENGTH, 0.0, 0.0]), BOB_RADIUS as f64, 1.0);
    if let Some(b) = physics.world.body_mut(bob) {
        b.collision_mask = 0;
        b.linear_damping = 0.05;
    }

    physics
        .world
        .add_constraint(Box::new(DistanceConstraint::<3> {
            body_a: pivot,
            body_b: bob,
            rest_length: ARM_LENGTH,
            stiffness: 1.0,
        }));

    // Register with the consciousness field so the F1 dev console's Φ
    // Inspector has something to show.
    physics.field.register(bob, 100.0, 0.5);

    let bob_mesh = meshes.add(Sphere::new(BOB_RADIUS).mesh().uv(16, 16));
    let bob_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.7, 0.85, 1.0),
        perceptual_roughness: 0.6,
        metallic: 0.1,
        ..default()
    });

    commands.spawn((
        PhysicsBody::new(bob, BOB_RADIUS),
        Mesh3d(bob_mesh),
        MeshMaterial3d(bob_material),
        Transform::from_xyz(ARM_LENGTH as f32, 0.0, 0.0),
    ));
}
