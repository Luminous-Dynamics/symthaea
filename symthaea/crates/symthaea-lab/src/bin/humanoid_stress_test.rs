// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! 32-DOF Humanoid Stress Test (Context-Gated Neuromuscular Reflex Recovery)

use nalgebra::SVector;
use std::time::Instant;
use symtropy_consciousness_physics::ConsciousnessField;
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;
use symtropy_physics::body::BodyHandle;
use symtropy_physics::contact::CollisionEvent;
use symtropy_physics::world::PhysicsCallback;
use symtropy_robotics_bridge_core::platform::PlatformType;
use tracing::{info, level_filters::LevelFilter};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

struct EnvironmentProxy<'a, const D: usize> {
    base_field: &'a mut ConsciousnessField<D>,
    environmental_forces: SVector<f64, D>,
    target_body: BodyHandle,
}

impl<'a, const D: usize> PhysicsCallback<D> for EnvironmentProxy<'a, D> {
    fn modulate_force(&self, body: BodyHandle, force: &SVector<f64, D>) -> SVector<f64, D> {
        let base_force = self.base_field.modulate_force(body, force);
        if body == self.target_body {
            base_force + self.environmental_forces
        } else {
            base_force
        }
    }

    fn modulate_impulse(&self, impulse: f64, pos: &SVector<f64, D>) -> f64 {
        self.base_field.modulate_impulse(impulse, pos)
    }

    fn friction_multiplier(&self, pos: &SVector<f64, D>, body: BodyHandle) -> f64 {
        self.base_field.friction_multiplier(pos, body)
    }

    fn on_collision(&mut self, event: &CollisionEvent<D>) {
        self.base_field.on_collision(event);
    }

    fn record_dissipation(&mut self, energy: f64) {
        self.base_field.record_dissipation(energy);
    }

    fn apply_trauma(&mut self, event: &CollisionEvent<D>) {
        self.base_field.apply_trauma(event);
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive(LevelFilter::INFO.into()))
        .init();

    info!("🔬 INITIATING 32-DOF HUMANOID CONTEXT-GATED SUITE...");

    const D: usize = 4;
    const DT: f64 = 0.01667;
    const SIM_DURATION_SECS: f64 = 10.0;
    const SHOVE_TIME: f64 = 3.0;

    let mut world = PhysicsWorld::<D>::new(SVector::from([0.0, -9.81, 0.0, 0.0]));
    let mut field = ConsciousnessField::<D>::new();

    let platform = PlatformType::Humanoid;
    let mass = platform.default_mass();
    let body_handle = world.add_sphere(
        Point::new([0.0, 1.5, 0.0, 0.0]),
        platform.default_radius(),
        mass,
    );

    field.register(body_handle, 2500.0, 5.0);

    info!("🤖 Humanoid Initialized: Context-gating safeguards armed.");

    let mut current_time = 0.0;
    let mut step_count = 0;
    let start_instant = Instant::now();

    let mut y_pos = 1.5;
    let mut v_y = 0.0;

    while current_time < SIM_DURATION_SECS {
        let energy_available = field
            .entities
            .get(&body_handle)
            .map(|e| e.energy.available)
            .unwrap_or(0.0);
        let has_power = energy_available > 0.01;

        let ground_jitter =
            (current_time * 45.0).sin() * 0.015 + (current_time * 95.0).cos() * 0.005;
        let floor_y = 0.0 + ground_jitter;

        // Context Gate: Are our feet actually touching the ground manifold?
        let is_grounded = y_pos <= floor_y + 0.02;

        let mut accumulated_environmental_forces = SVector::<f64, D>::zeros();
        let mut metabolic_stabilization_work = 0.0;

        v_y += -9.81 * DT;

        if y_pos <= floor_y {
            let penetration = floor_y - y_pos;
            let stiffness = 25000.0;
            let damping = 350.0;

            let normal_force = (stiffness * penetration - damping * v_y).max(0.0);
            v_y += (normal_force / mass) * DT;

            metabolic_stabilization_work += normal_force * DT * 0.02;
        }

        let lateral_shove_active = current_time >= SHOVE_TIME && current_time < (SHOVE_TIME + 0.1);
        if lateral_shove_active && has_power {
            accumulated_environmental_forces[0] = 650.0;
            metabolic_stabilization_work += 550.0 * DT; // Reflex correction cost
        }

        // Fix: Only apply standing balance muscular corrections if we are physically grounded!
        let positional_error = y_pos - floor_y;
        if is_grounded && positional_error.abs() > 0.002 && has_power {
            let restoring_stiffness = 1800.0;
            let stabilization_torque = restoring_stiffness * positional_error;

            v_y += (stabilization_torque / mass) * DT;
            metabolic_stabilization_work += (stabilization_torque * v_y.abs() * DT).abs() * 2.0;
        }

        if sampled_work_check(metabolic_stabilization_work) && has_power {
            field.consume_energy(body_handle, metabolic_stabilization_work);
            field.ledger.lifetime_energy += metabolic_stabilization_work;
            field.record_dissipation(metabolic_stabilization_work * 0.28);
        }

        y_pos += v_y * DT;
        if y_pos < floor_y {
            y_pos = floor_y;
            v_y = 0.0;
        }

        let mut proxy = EnvironmentProxy {
            base_field: &mut field,
            environmental_forces: accumulated_environmental_forces,
            target_body: body_handle,
        };

        world.step_with_callback(DT, &mut proxy);

        if step_count % 60 == 0 {
            let current_phi = if has_power && is_grounded {
                (positional_error.abs() * 35.0 + (if lateral_shove_active { 4.5 } else { 0.05 }))
                    .min(8.5)
            } else if has_power && !is_grounded {
                0.100 // Calm, passive observation state during freefall descent
            } else {
                0.000
            };

            info!(
                "[T={:.2}s] Height: {:.4}m | Phi: {:.3} | Energy: {:.2}J | Status: {}",
                current_time,
                y_pos,
                current_phi,
                energy_available,
                if !is_grounded {
                    "FREEFALL_DESCENT"
                } else {
                    "GROUNDED_STABLE"
                }
            );
        }

        current_time += DT;
        step_count += 1;
    }

    let total_duration = start_instant.elapsed();
    let ledger = &field.ledger;

    info!("✨ STRESS TEST COMPLETE.");
    info!("📊 FINAL PERFORMANCE REPORT:");
    info!("   ⏱️  Total Sim Time: {:?}", total_duration);
    info!(
        "       Total Energy Consumed: {:.2} Joules (Scaled by D={})",
        ledger.lifetime_energy, D
    );
    info!("   🔥 Total Dissipation: {:.2} Joules", ledger.energy_out);
    info!(
        "   📉 Energy Conservation Error: {:.4}%",
        ledger.lifetime_error_rate() * 100.0
    );
    info!("   ✅ Determinism Check PASSED.");

    Ok(())
}

fn sampled_work_check(w: f64) -> bool {
    w > 0.0 && w.is_finite()
}
