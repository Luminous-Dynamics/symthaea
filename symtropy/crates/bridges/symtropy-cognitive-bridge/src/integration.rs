use bevy::prelude::*;
use nalgebra::SVector;
use symthaea_bevy_brain::CognitiveBrain;
use symtropy_physics::BodyHandle;

pub fn cognitive_physics_bridge_system(
    mut query: Query<(&BodyHandle, &mut CognitiveBrain)>,
    mut world: ResMut<PhysicsWorld<2>>,
) {
    for (handle, mut brain) in query.iter_mut() {
        if let Some(body) = world.body(*handle) {
            brain.perception_input = format!(
                "pos: [{:.2}, {:.2}], vel: [{:.2}, {:.2}]",
                body.position()[0],
                body.position()[1],
                body.linear_velocity[0],
                body.linear_velocity[1]
            );
        }

        if !brain.motor_output.is_empty() {
            if let Some(body) = world.body_mut(*handle) {
                body.force_accumulator +=
                    SVector::from([brain.motor_output[0] as f64, brain.motor_output[1] as f64]);
            }
        }
    }
}
