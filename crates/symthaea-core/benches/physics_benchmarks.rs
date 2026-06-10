// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Physics Hot-Path Benchmarks
//!
//! Run with: `cargo bench --bench physics_benchmarks -p symthaea-core`
//!
//! ## Round 5 baselines (4 benchmarks):
//! 1. Lindblad single RK4 step
//! 2. Lindblad adaptive integration (t=0 to t=3)
//! 3. WKB Eckart barrier (200-point integration)
//! 4. Exact Eckart (Kemble formula)
//!
//! ## Round 6 additions (4 benchmarks):
//! 5. Acoustics Doppler frequency calculation
//! 6. True Phi entropy on 16384-dim ContinuousHV
//! 7. Tensor determinant (Schwarzschild metric)
//! 8. Jarzynski free energy difference (10k samples)

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_core::physics::{
    AcousticsEncoder, Complex64, DensityMatrix, JarzynskiEstimator, LindbladEvolution,
    MetricTensor, TruePhiCalculator, TunnelingCalculator,
};

fn bench_lindblad_single_rk4_step(c: &mut Criterion) {
    // 2×2 dephasing channel, γ=0.5
    let hamiltonian = DensityMatrix::new(2);
    let mut evolution = LindbladEvolution::new(hamiltonian);
    let mut sigma_z = DensityMatrix::new(2);
    sigma_z.set(0, 0, Complex64::from_real(1.0));
    sigma_z.set(1, 1, Complex64::from_real(-1.0));
    evolution.add_lindblad(sigma_z, 0.5);

    let s = 1.0 / 2.0_f64.sqrt();
    let psi = vec![Complex64::new(s, 0.0), Complex64::new(s, 0.0)];
    let rho = DensityMatrix::pure_state(&psi);

    c.bench_function("lindblad_single_rk4_step", |b| {
        b.iter(|| {
            let result = evolution.evolve(black_box(&rho), black_box(0.001));
            black_box(result)
        });
    });
}

fn bench_lindblad_adaptive_t0_to_3(c: &mut Criterion) {
    let hamiltonian = DensityMatrix::new(2);
    let mut evolution = LindbladEvolution::new(hamiltonian);
    let mut sigma_z = DensityMatrix::new(2);
    sigma_z.set(0, 0, Complex64::from_real(1.0));
    sigma_z.set(1, 1, Complex64::from_real(-1.0));
    evolution.add_lindblad(sigma_z, 0.5);

    let s = 1.0 / 2.0_f64.sqrt();
    let psi = vec![Complex64::new(s, 0.0), Complex64::new(s, 0.0)];
    let rho = DensityMatrix::pure_state(&psi);

    c.bench_function("lindblad_adaptive_t0_to_3", |b| {
        b.iter(|| {
            let result = evolution.evolve_adaptive(black_box(&rho), black_box(3.0), 1e-8, 1e-10);
            black_box(result)
        });
    });
}

fn bench_wkb_eckart_200pt(c: &mut Criterion) {
    let calc = TunnelingCalculator::electron();
    // Electron, 0.6 eV barrier
    let v0 = 0.6 * 1.6e-19; // barrier height in J
    let a = 1.0e-10; // barrier width in m
    let energy = 0.3 * 1.6e-19; // sub-barrier energy

    c.bench_function("wkb_eckart_200pt", |b| {
        b.iter(|| {
            let result = calc.eckart_barrier(black_box(energy), black_box(v0), black_box(a));
            black_box(result)
        });
    });
}

fn bench_eckart_exact_kemble(c: &mut Criterion) {
    let calc = TunnelingCalculator::electron();
    let v0 = 0.6 * 1.6e-19;
    let a = 1.0e-10;
    let energy = 0.3 * 1.6e-19;

    c.bench_function("eckart_exact_kemble", |b| {
        b.iter(|| {
            let result = calc.eckart_barrier_exact(black_box(energy), black_box(v0), black_box(a));
            black_box(result)
        });
    });
}

fn bench_acoustics_doppler(c: &mut Criterion) {
    let genesis = GenesisSeed::from_phrase("bench acoustics doppler");
    let encoder = AcousticsEncoder::from_genesis(&genesis);

    c.bench_function("acoustics_doppler", |b| {
        b.iter(|| {
            let result = encoder.doppler_frequency(
                black_box(440.0),
                black_box(343.0),
                black_box(34.3),
                black_box(0.0),
            );
            black_box(result)
        });
    });
}

fn bench_true_phi_entropy(c: &mut Criterion) {
    let calc = TruePhiCalculator::new();
    let hv = ContinuousHV::random(16384, 42);

    c.bench_function("true_phi_entropy_16384d", |b| {
        b.iter(|| {
            let result = calc.entropy(black_box(&hv));
            black_box(result)
        });
    });
}

fn bench_tensor_determinant(c: &mut Criterion) {
    let m_sun = 1.989e30;
    let r_s = 2.0 * 6.674e-11 * m_sun / (299_792_458.0 * 299_792_458.0);
    let r = 5.0 * r_s;
    let theta = std::f64::consts::FRAC_PI_2;
    let g = MetricTensor::schwarzschild(m_sun, r, theta);

    c.bench_function("tensor_determinant_schwarzschild", |b| {
        b.iter(|| {
            let result = black_box(&g).determinant();
            black_box(result)
        });
    });
}

fn bench_jarzynski_10k(c: &mut Criterion) {
    let jarzynski = JarzynskiEstimator::new();
    let work_samples: Vec<f64> = (0..10_000)
        .map(|i| 4.14e-21 * (1.0 + 0.3 * ((i as f64 * 0.1).sin())))
        .collect();

    c.bench_function("jarzynski_10k_samples", |b| {
        b.iter(|| {
            let result =
                jarzynski.free_energy_difference(black_box(&work_samples), black_box(300.0));
            black_box(result)
        });
    });
}

// =========================================================================
// Round 8 benchmarks: Particle physics hot paths
// =========================================================================

fn bench_standard_model_construction(c: &mut Criterion) {
    use symthaea_core::physics::StandardModel;
    let genesis = GenesisSeed::from_phrase("bench standard model");

    c.bench_function("standard_model_construction", |b| {
        b.iter(|| {
            let model = StandardModel::from_genesis(black_box(&genesis));
            black_box(model)
        });
    });
}

fn bench_hadron_composition(c: &mut Criterion) {
    use symthaea_core::physics::{Hadrons, StandardModel};
    let genesis = GenesisSeed::from_phrase("bench hadrons");
    let model = StandardModel::from_genesis(&genesis);

    c.bench_function("hadron_composition", |b| {
        b.iter(|| {
            let hadrons = Hadrons::from_model(black_box(&model), black_box(&genesis));
            black_box(hadrons)
        });
    });
}

fn bench_antimatter_annihilation(c: &mut Criterion) {
    use symthaea_core::physics::{Antimatter, Hadrons, StandardModel};
    let genesis = GenesisSeed::from_phrase("bench antimatter");
    let model = StandardModel::from_genesis(&genesis);
    let hadrons = Hadrons::from_model(&model, &genesis);
    let antimatter = Antimatter::from_model(&model, &hadrons, &genesis);

    c.bench_function("antimatter_annihilation", |b| {
        b.iter(|| {
            let event = antimatter.electron_positron_annihilation(black_box(&model));
            black_box(event)
        });
    });
}

criterion_group!(
    benches,
    bench_lindblad_single_rk4_step,
    bench_lindblad_adaptive_t0_to_3,
    bench_wkb_eckart_200pt,
    bench_eckart_exact_kemble,
    bench_acoustics_doppler,
    bench_true_phi_entropy,
    bench_tensor_determinant,
    bench_jarzynski_10k,
    bench_standard_model_construction,
    bench_hadron_composition,
    bench_antimatter_annihilation,
);
criterion_main!(benches);
