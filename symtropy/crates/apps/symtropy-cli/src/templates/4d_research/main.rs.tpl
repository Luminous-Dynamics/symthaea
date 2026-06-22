//! `{{project_name}}` — Symtropy 4D research scene.
//!
//! Three bobs at three different W-coordinates. Press `[` / `]` to move the
//! visualization slice along W; bobs near the slice fade in, others fade out.
//! The full 4D simulation always runs — only the rendered cross-section moves.
//! Press F1 for the dev console.

use symtropy::prelude::*;

const ARM: f64 = 1.0;
const BOB_RADIUS: f32 = 0.10;
const W_LAYERS: [f64; 3] = [-1.0, 0.0, 1.0];
const SLICE_THICKNESS: f64 = 0.45;
const W_STEP: f64 = 0.1;

#[derive(Component)]
struct Bob {
    handle: BodyHandle,
}

#[derive(Resource)]
struct WSlice(f64);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "{{project_name}} — 4D research".into(),
                resolution: bevy::window::WindowResolution::from((1280u32, 720u32)),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(WSlice(0.0))
        .add_plugins(SymtropyScenePlugin::default())
        .add_plugins(SymtropyPhysicsPlugin::<4>::with_gravity([
            0.0, -9.81, 0.0, 0.0,
        ]))
        .add_plugins(SymtropyDevConsolePlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (handle_w_input, fade_by_w_slice))
        .run();
}

fn setup(
    mut commands: Commands,
    mut physics: ResMut<SymtropyPhysics<4>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn(fixed_camera(
        Vec3::new(0.0, 1.5, 4.0),
        Vec3::new(0.0, -0.5, 0.0),
    ));

    let mesh = meshes.add(Sphere::new(BOB_RADIUS).mesh().uv(16, 16));

    for (i, &w) in W_LAYERS.iter().enumerate() {
        let pivot_x = (i as f64 - 1.0) * 0.4;
        let pivot = physics.world.add_body(RigidBody::<4>::static_body(
            BodyHandle(0),
            Point::new([pivot_x, 0.0, 0.0, w]),
            Box::new(PhysicsSphere::new(Point::origin(), 0.01)),
        ));
        if let Some(p) = physics.world.body_mut(pivot) {
            p.collision_mask = 0;
        }

        let bob = physics
            .world
            .add_sphere(Point::new([pivot_x + ARM, 0.0, 0.0, w]), BOB_RADIUS as f64, 1.0);
        if let Some(b) = physics.world.body_mut(bob) {
            b.collision_mask = 0;
            b.linear_damping = 0.05;
        }

        physics
            .world
            .add_constraint(Box::new(DistanceConstraint::<4> {
                body_a: pivot,
                body_b: bob,
                rest_length: ARM,
                stiffness: 1.0,
            }));

        physics.field.register(bob, 100.0, 0.5);

        let mat = materials.add(StandardMaterial {
            base_color: Color::srgba(0.7, 0.85, 1.0, 1.0),
            perceptual_roughness: 0.6,
            metallic: 0.1,
            alpha_mode: AlphaMode::Blend,
            ..default()
        });

        commands.spawn((
            Bob { handle: bob },
            PhysicsBody::new(bob, BOB_RADIUS),
            Mesh3d(mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(pivot_x as f32 + ARM as f32, 0.0, 0.0),
        ));
    }
}

fn handle_w_input(keys: Res<ButtonInput<KeyCode>>, mut slice: ResMut<WSlice>) {
    if keys.just_pressed(KeyCode::BracketLeft) {
        slice.0 -= W_STEP;
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        slice.0 += W_STEP;
    }
}

fn fade_by_w_slice(
    physics: Res<SymtropyPhysics<4>>,
    slice: Res<WSlice>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    query: Query<(&Bob, &MeshMaterial3d<StandardMaterial>, &mut Visibility)>,
) {
    for (bob, mat_handle, mut vis) in query {
        let Some(body) = physics.world.body(bob.handle) else {
            continue;
        };
        let w = body.position().coord(3);
        let dist = (w - slice.0).abs();
        let alpha = if dist >= SLICE_THICKNESS {
            0.0
        } else {
            (1.0 - dist / SLICE_THICKNESS) as f32
        };
        if alpha <= 0.001 {
            *vis = Visibility::Hidden;
        } else {
            *vis = Visibility::Visible;
            if let Some(mat) = materials.get_mut(&mat_handle.0) {
                let c = mat.base_color.to_srgba();
                mat.base_color = Color::srgba(c.red, c.green, c.blue, alpha);
            }
        }
    }
}
