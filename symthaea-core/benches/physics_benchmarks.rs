//! # Physics Hot-Path Benchmarks
//!
//! Run with: `cargo bench --bench physics_benchmarks -p symthaea-core`
//!
//! Benchmarks for performance-critical physics functions:
//! 1. Lindblad single RK4 step
//! 2. Lindblad adaptive integration (t=0 to t=3)
//! 3. WKB Eckart barrier (200-point integration)
//! 4. Exact Eckart (Kemble formula)

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use symthaea_core::physics::{Complex64, DensityMatrix, LindbladEvolution, TunnelingCalculator};

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
    let a = 1.0e-10;         // barrier width in m
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

criterion_group!(
    benches,
    bench_lindblad_single_rk4_step,
    bench_lindblad_adaptive_t0_to_3,
    bench_wkb_eckart_200pt,
    bench_eckart_exact_kemble,
);
criterion_main!(benches);
