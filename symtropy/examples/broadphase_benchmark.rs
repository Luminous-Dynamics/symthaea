// symtropy/examples/broadphase_benchmark.rs
use bevy::prelude::*;
use symtropy_bevy_core::PhysicsBody;
use symtropy_physics_gpu::{BroadphaseResults, GpuBroadphaseManager, GpuPhysicsPlugin};

fn main() {
    App::new()
        .add_plugins(MinimalPlugins)
        .add_plugins(bevy::log::LogPlugin::default())
        .add_plugins(bevy::asset::AssetPlugin::default())
        .add_plugins(bevy::render::RenderPlugin::default())
        .add_plugins(GpuPhysicsPlugin)
        .add_systems(Startup, spawn_test_bodies)
        .add_systems(Update, benchmark_stats)
        .run();
}

fn spawn_test_bodies(mut commands: Commands) {
    // Spawn 100k test bodies in a grid
    for i in 0..100_000 {
        commands.spawn((
            PhysicsBody {
                handle: symtropy_physics::body::BodyHandle(i),
                visual_radius: 0.5,
            },
            Transform::from_translation(Vec3::new(
                (i % 400) as f32 * 2.5 - 500.0,
                ((i / 400) % 400) as f32 * 2.5 - 500.0,
                (i / 160000) as f32 * 2.5,
            )),
            GlobalTransform::default(),
        ));
    }
}

fn benchmark_stats(
    time: Res<Time>,
    results: Res<BroadphaseResults>,
    manager: Res<GpuBroadphaseManager>,
) {
    let now = time.elapsed_secs_f64();
    static mut LAST_PRINT: f64 = 0.0;

    unsafe {
        if now - LAST_PRINT > 1.0 {
            println!(
                "Bodies: {} | Pairs found: {} | Frame: {:.1}ms",
                manager.input_aabbs.len(),
                results.pair_count,
                time.delta_secs() * 1000.0
            );
            LAST_PRINT = now;
        }
    }
}
