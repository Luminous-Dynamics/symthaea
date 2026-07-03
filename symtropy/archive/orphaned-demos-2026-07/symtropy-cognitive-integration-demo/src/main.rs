use bevy::prelude::*;
use nalgebra::SVector;
use symthaea_bevy_brain::{CognitiveBrain, SymthaeaBrainPlugin};
use symtropy_cognitive_bridge::integration::cognitive_physics_bridge_system;
use symtropy_physics::{BodyHandle, PhysicsWorld, RigidBody};

fn main() {
    App::new()
        .add_plugins(MinimalPlugins)
        .add_plugins(SymthaeaBrainPlugin::default())
        .insert_resource(PhysicsWorld::<2>::new(SVector::from([0.0, 0.0])))
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, cognitive_physics_bridge_system)
        .run();
}

fn setup(mut world: ResMut<PhysicsWorld<2>>, mut commands: Commands) {
    let handle = world.add_body(RigidBody::new_dynamic(SVector::from([0.0, 0.0]), 1.0));
    commands.spawn((handle, CognitiveBrain::new(32, "Integration Test")));
}
