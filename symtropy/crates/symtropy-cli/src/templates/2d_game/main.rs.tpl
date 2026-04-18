//! `{{project_name}}` — Symtropy 2D scene.
//!
//! A single bouncing physics-bodied sprite. Click anywhere to give it a kick.
//! Press F1 for the dev console (Φ Inspector + Scene controls).

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use symtropy_bevy::{PhysicsBody, SymtropyPhysics, SymtropyPhysicsPlugin};
use symtropy_devconsole::SymtropyDevConsolePlugin;
use symtropy_math::Point;

const BOB_RADIUS: f32 = 20.0;
const SHOCK_VELOCITY: f64 = 400.0;

#[derive(Component)]
struct Bob;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "{{project_name}} — 2D".into(),
                resolution: bevy::window::WindowResolution::from((1280u32, 720u32)),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.04, 0.04, 0.06)))
        .add_plugins(SymtropyPhysicsPlugin::<2>::with_gravity([0.0, -981.0]))
        .add_plugins(SymtropyDevConsolePlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, click_to_kick)
        .run();
}

fn setup(mut commands: Commands, mut physics: ResMut<SymtropyPhysics<2>>) {
    commands.spawn(Camera2d);

    let handle = physics
        .world
        .add_sphere(Point::new([0.0, 200.0]), BOB_RADIUS as f64, 1.0);
    if let Some(b) = physics.world.body_mut(handle) {
        b.collision_mask = 0;
        b.linear_damping = 0.05;
    }
    physics.field.register(handle, 100.0, 32.0);

    commands.spawn((
        Bob,
        PhysicsBody::new(handle, BOB_RADIUS),
        Sprite::from_color(Color::srgb(0.7, 0.85, 1.0), Vec2::splat(BOB_RADIUS * 2.0)),
        Transform::from_xyz(0.0, 200.0, 0.0),
    ));
}

fn click_to_kick(
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    bobs: Query<&PhysicsBody, With<Bob>>,
    mut physics: ResMut<SymtropyPhysics<2>>,
) {
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    let Ok(window) = windows.single() else { return };
    let Some(cursor) = window.cursor_position() else { return };
    let Ok((camera, xform)) = cameras.single() else { return };
    let Ok(world) = camera.viewport_to_world_2d(xform, cursor) else { return };
    for body in &bobs {
        if let Some(b) = physics.world.body_mut(body.handle) {
            let pos = b.position();
            let dx = world.x as f64 - pos.coord(0);
            let dy = world.y as f64 - pos.coord(1);
            let len = (dx * dx + dy * dy).sqrt().max(1.0);
            b.linear_velocity[0] += SHOCK_VELOCITY * dx / len;
            b.linear_velocity[1] += SHOCK_VELOCITY * dy / len;
        }
    }
}
