// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Benchmarks for the consciousness-physics coupling layer.
//!
//! These benchmarks measure the performance of the core coupling operations:
//! - `update_entity`: consciousness computation for a single entity
//! - `ConsciousnessField::tick_prediction_errors`: per-entity prediction error decay
//! - `spread_emotional_contagion`: O(N²) social field propagation
//! - `apply_macro_modifiers`: civilization-sim → physics modifier application
//! - `PhiEnvironmentCoupler::tick`: proc-gen parameter update
//! - `BiometricHistory::to_consciousness_inputs`: biometric → consciousness conversion
//! - `run_bifurcation_sweep` (tiny config): phase transition simulation kernel
//!
//! # Running
//! ```bash
//! cargo bench -p symtropy-consciousness-physics
//! ```

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::{
    biometrics::{BiometricHistory, BiometricSample},
    coupling::ConsciousnessField,
    macro_bridge::{MacroWorldModifiers, apply_macro_modifiers},
    phase_transition::{BifurcationConfig, run_bifurcation_sweep},
    proc_gen::PhiEnvironmentCoupler,
};
use symtropy_math::Point;
use symtropy_physics::{BodyHandle, PhysicsWorld};

// ────────────────────────────────────────────────────────────
//  Helpers
// ────────────────────────────────────────────────────────────

fn make_field_n<const D: usize>(n: usize) -> (ConsciousnessField<D>, Vec<BodyHandle>) {
    let mut world = PhysicsWorld::<D>::new(SVector::zeros());
    let mut field = ConsciousnessField::new();
    let mut handles = Vec::with_capacity(n);
    for i in 0..n {
        let h = {
            let mut coords = [0.0f64; D];
            coords[0] = i as f64 * 3.0;
            world.add_sphere(Point::new(coords), 1.0, 1.0)
        };
        field.register(h, 100.0, 2.0);
        handles.push(h);
    }
    (field, handles)
}

fn test_inputs(phi: f64) -> ConsciousnessInputs {
    ConsciousnessInputs {
        phi,
        broadcast: phi * 0.9,
        working_memory: phi * 0.8,
        attention: phi * 0.85,
        recurrence: phi * 0.7,
        embodiment: phi * 0.75,
        knowledge: phi * 0.6,
        synchrony: phi * 0.95,
    }
}

// ────────────────────────────────────────────────────────────
//  Benchmark: update_entity (single entity, consciousness computation)
// ────────────────────────────────────────────────────────────

fn bench_update_entity_single(c: &mut Criterion) {
    let (mut field, handles) = make_field_n::<2>(1);
    let h = handles[0];
    let inputs = test_inputs(0.75);
    let pos = Point::origin();

    c.bench_function("update_entity_single_2d", |b| {
        b.iter(|| {
            field.update_entity(black_box(h), black_box(&inputs), black_box(pos));
        });
    });
}

// ────────────────────────────────────────────────────────────
//  Benchmark: update_entity (N entities)
// ────────────────────────────────────────────────────────────

fn bench_update_entity_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("update_entity_n_entities");
    for n in [8, 32, 128].iter() {
        let (mut field, handles) = make_field_n::<2>(*n);
        let inputs = test_inputs(0.75);
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                for &h in &handles {
                    field.update_entity(h, black_box(&inputs), Point::origin());
                }
            });
        });
    }
    group.finish();
}

// ────────────────────────────────────────────────────────────
//  Benchmark: tick_prediction_errors
// ────────────────────────────────────────────────────────────

fn bench_tick_prediction_errors(c: &mut Criterion) {
    let mut group = c.benchmark_group("tick_prediction_errors");
    for n in [8, 64, 256].iter() {
        let (mut field, _) = make_field_n::<2>(*n);
        // Pre-load some prediction error
        for entity in field.entities.values_mut() {
            entity.prediction_error = 0.5;
        }
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                field.tick_prediction_errors();
            });
        });
    }
    group.finish();
}

// ────────────────────────────────────────────────────────────
//  Benchmark: spread_emotional_contagion
// ────────────────────────────────────────────────────────────

fn bench_spread_contagion(c: &mut Criterion) {
    let mut group = c.benchmark_group("spread_emotional_contagion");
    for n in [4, 16, 64].iter() {
        let (mut field, handles) = make_field_n::<2>(*n);
        // Pre-update entities so they have Phi values
        let inputs = test_inputs(0.8);
        for &h in &handles {
            field.update_entity(h, &inputs, Point::origin());
        }
        // Set positions in ring
        let r = 10.0f64;
        let entity_positions: Vec<(BodyHandle, Point<2>)> = handles
            .iter()
            .enumerate()
            .map(|(i, &h)| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / handles.len() as f64;
                (h, Point::new([r * angle.cos(), r * angle.sin()]))
            })
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                field.spread_emotional_contagion(
                    black_box(&entity_positions),
                    black_box(1.0 / 60.0),
                );
            });
        });
    }
    group.finish();
}

// ────────────────────────────────────────────────────────────
//  Benchmark: apply_macro_modifiers
// ────────────────────────────────────────────────────────────

fn bench_apply_macro_modifiers(c: &mut Criterion) {
    let (mut field, _) = make_field_n::<2>(32);
    let mods = MacroWorldModifiers::from_sim(0.7, 0.2, 0.3, false, 0.6, 0.1);

    c.bench_function("apply_macro_modifiers_32_entities", |b| {
        b.iter(|| {
            apply_macro_modifiers(black_box(&mut field), black_box(&mods), black_box(0.1));
        });
    });
}

// ────────────────────────────────────────────────────────────
//  Benchmark: PhiEnvironmentCoupler::tick
// ────────────────────────────────────────────────────────────

fn bench_phi_env_coupler(c: &mut Criterion) {
    let mut coupler = PhiEnvironmentCoupler::new(60);

    c.bench_function("phi_env_coupler_tick", |b| {
        b.iter(|| {
            coupler.tick(black_box(0.65));
        });
    });
}

// ────────────────────────────────────────────────────────────
//  Benchmark: BiometricHistory::to_consciousness_inputs
// ────────────────────────────────────────────────────────────

fn bench_biometric_conversion(c: &mut Criterion) {
    let mut history = BiometricHistory::new(30);
    // Fill window
    for _ in 0..30 {
        history.push(BiometricSample {
            hrv_rmssd: Some(50.0),
            eeg_gamma_coherence: Some(0.7),
            gsr_microsiemens: Some(8.0),
            respiration_bpm: Some(6.0),
            heart_rate_bpm: Some(65.0),
            eeg_alpha_power: Some(20.0),
            eeg_theta_power: Some(15.0),
            eeg_beta_power: Some(10.0),
        });
    }

    c.bench_function("biometric_to_consciousness_inputs_30samples", |b| {
        b.iter(|| {
            let _ = black_box(history.to_consciousness_inputs());
        });
    });
}

// ────────────────────────────────────────────────────────────
//  Benchmark: full physics step with consciousness coupling
// ────────────────────────────────────────────────────────────

fn bench_full_step_with_coupling(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_step_with_consciousness");
    for n in [4, 16].iter() {
        let (field, handles) = make_field_n::<2>(*n);
        let mut world = PhysicsWorld::<2>::new(SVector::zeros());
        // Re-add spheres to the world (make_field_n creates its own world)
        let mut field2 = ConsciousnessField::<2>::new();
        for i in 0..*n {
            let h = world.add_sphere(Point::new([i as f64 * 3.0, 0.0]), 1.0, 1.0);
            field2.register(h, 100.0, 2.0);
        }
        let inputs = test_inputs(0.8);
        let _ = (field, handles); // drop the unused ones

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                // Update all entities
                for h in field2.entities.keys().copied().collect::<Vec<_>>() {
                    field2.update_entity(h, &inputs, Point::origin());
                }
                field2.tick_prediction_errors();
                world.step_with_callback(1.0 / 60.0, &mut field2);
            });
        });
    }
    group.finish();
}

// ────────────────────────────────────────────────────────────
//  Benchmark: bifurcation sweep (tiny config, smoke test)
// ────────────────────────────────────────────────────────────

fn bench_bifurcation_sweep_tiny(c: &mut Criterion) {
    let config = BifurcationConfig {
        k_min: 0.0,
        k_max: 0.2,
        k_steps: 3,
        num_agents: 2,
        ring_radius: 5.0,
        warmup_ticks: 2,
        measure_ticks: 5,
        dt: 1.0 / 60.0,
        agent_inputs: ConsciousnessInputs {
            phi: 0.5,
            broadcast: 0.5,
            working_memory: 0.5,
            attention: 0.5,
            recurrence: 0.5,
            embodiment: 0.5,
            knowledge: 0.5,
            synchrony: 0.5,
        },
    };

    c.bench_function("bifurcation_sweep_3step_2agent", |b| {
        b.iter(|| {
            let _ = black_box(run_bifurcation_sweep(&config));
        });
    });
}

// ────────────────────────────────────────────────────────────
//  Criterion groups
// ────────────────────────────────────────────────────────────

criterion_group!(
    coupling_benchmarks,
    bench_update_entity_single,
    bench_update_entity_n,
    bench_tick_prediction_errors,
    bench_spread_contagion,
    bench_apply_macro_modifiers,
    bench_phi_env_coupler,
    bench_biometric_conversion,
    bench_full_step_with_coupling,
    bench_bifurcation_sweep_tiny,
);

criterion_main!(coupling_benchmarks);
