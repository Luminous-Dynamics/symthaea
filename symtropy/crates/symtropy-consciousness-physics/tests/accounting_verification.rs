// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::coupling::ConsciousnessField;
use symtropy_math::Point;
use symtropy_physics::world::PhysicsWorld;

fn test_inputs(phi: f64) -> ConsciousnessInputs {
    ConsciousnessInputs {
        phi,
        broadcast: 1.0,
        working_memory: 1.0,
        attention: 1.0,
        recurrence: 1.0,
        embodiment: 1.0,
        knowledge: 1.0,
        synchrony: 1.0,
    }
}

#[test]
fn test_sanctuary_absorption_hits_ledger() {
    let gravity = SVector::zeros();
    let mut world = PhysicsWorld::<2>::new(gravity);
    let mut field = ConsciousnessField::<2>::new();

    let h1 = world.add_sphere(Point::new([-0.9, 0.0]), 1.0, 1.0);
    let h2 = world.add_sphere(Point::new([0.9, 0.0]), 1.0, 1.0);

    // Disable damping
    world.body_mut(h1).unwrap().linear_damping = 0.0;
    world.body_mut(h2).unwrap().linear_damping = 0.0;

    field.register(h1, 100.0, 10.0);
    field.register(h2, 100.0, 10.0);

    // Now we can update config on registered entities
    for entity in field.entities.values_mut() {
        let mut config = entity.equation.config().clone();
        config.enable_embodiment_factor = false;
        config.enable_narrative_factor = false;
        config.enable_social_factor = false;
        entity.equation.update_config(config);
    }

    // Activate sanctuary on h1
    // Requires: stillness > 0.6, total_harmony > 2.0, phi > 0.3
    let entity = field.entities.get_mut(&h1).unwrap();
    entity.harmony_activations[7] = 1.0; // Sacred Stillness
    entity.harmony_activations[0] = 0.8;
    entity.harmony_activations[1] = 0.8; // total = 2.6

    field.update_entity(h1, &test_inputs(0.9), Point::new([-0.9, 0.0]));

    let entity = field.entities.get(&h1).unwrap();
    println!("Entity phi: {}", entity.phi());
    println!("Stillness: {}", entity.stillness());
    println!("Total harmony: {}", entity.total_harmony_energy());

    // Ensure sanctuary is active
    assert!(
        field.sanctuaries.get(&h1).unwrap().active,
        "Sanctuary should be active. Phi={}, Stillness={}, TotalHarmony={}",
        entity.phi(),
        entity.stillness(),
        entity.total_harmony_energy()
    );

    // Set velocities towards each other
    world.body_mut(h1).unwrap().linear_velocity = SVector::from([1.0, 0.0]);
    world.body_mut(h2).unwrap().linear_velocity = SVector::from([-1.0, 0.0]);

    // Step 1: Force a collision inside sanctuary
    world.step_with_callback(0.1, &mut field);

    // Step 2: Finalize thermodynamics
    let balance = field.tick_thermodynamics();

    // Sanctuary should have absorbed some energy
    assert!(
        balance.energy_out > 0.0,
        "Sanctuary absorption should be recorded as energy_out, got {}",
        balance.energy_out
    );
}
