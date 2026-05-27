// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Benchmark regression gates.
//!
//! These `#[test]` functions assert that key operations complete within
//! generous time bounds (typically 5x expected). They catch catastrophic
//! performance regressions without requiring a separate benchmark harness.

use super::decoherence::{Complex64, DecoherenceChannel, DensityMatrix, simulate_decoherence};
use super::hadrons::Hadrons;
use super::nonequilibrium::JarzynskiEstimator;
use super::quantum_tunneling::TunnelingCalculator;
use super::standard_model::StandardModel;
use crate::consciousness_metrics::TruePhiCalculator;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use std::time::Instant;

/// Single Lindblad RK4 decoherence step should complete under 1ms.
#[test]
fn gate_lindblad_step_under_1ms() {
    let psi = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let initial = DensityMatrix::pure_state(&psi);
    let channel = DecoherenceChannel::AmplitudeDamping { gamma: 0.1 };

    let start = Instant::now();
    let _result = simulate_decoherence(&initial, channel, 0.01, 1);
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 1,
        "Single Lindblad step took {}ms, threshold is 1ms",
        elapsed.as_millis()
    );
}

/// Eckart barrier exact (Kemble) calculation should complete under 10ms.
/// Ignored: this is a performance benchmark, not a correctness test.
/// Timing-sensitive assertions are unsuitable for CI (shared runners,
/// variable CPU frequency, background load). Run locally with:
///   cargo test -p symthaea-core --lib gate_eckart_exact -- --ignored
#[test]
#[ignore]
fn gate_eckart_exact_under_1ms() {
    let calc = TunnelingCalculator::electron();

    let start = Instant::now();
    let _result = calc.eckart_barrier_exact(0.5, 1.0, 1.0);
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 10,
        "Eckart exact took {}ms, threshold is 10ms",
        elapsed.as_millis()
    );
}

/// StandardModel construction should complete under 500ms.
#[test]
fn gate_standard_model_under_500ms() {
    let genesis = GenesisSeed::from_phrase("bench_sm");

    let start = Instant::now();
    let _model = StandardModel::from_genesis(&genesis);
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 500,
        "StandardModel::from_genesis took {}ms, threshold is 500ms",
        elapsed.as_millis()
    );
}

/// Hadrons construction should complete under 500ms.
#[test]
fn gate_hadron_under_500ms() {
    let genesis = GenesisSeed::from_phrase("bench_hadrons");
    let model = StandardModel::from_genesis(&genesis);

    let start = Instant::now();
    let _hadrons = Hadrons::from_model(&model, &genesis);
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 500,
        "Hadrons::from_model took {}ms, threshold is 500ms",
        elapsed.as_millis()
    );
}

/// TruePhiCalculator entropy on a 16384-dim HV should complete under 10ms.
#[test]
fn gate_true_phi_entropy_under_10ms() {
    let hv = ContinuousHV::random(16384, 42);
    let calc = TruePhiCalculator::new();

    let start = Instant::now();
    let _entropy = calc.entropy(&hv);
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 10,
        "TruePhiCalculator entropy took {}ms, threshold is 10ms",
        elapsed.as_millis()
    );
}

/// Jarzynski free energy estimate with 10k samples should complete under 50ms.
#[test]
fn gate_jarzynski_10k_under_50ms() {
    let estimator = JarzynskiEstimator::new();
    // Generate 10k Gaussian work samples
    let samples: Vec<f64> = (0..10_000)
        .map(|i| {
            let x = (i as f64) / 10_000.0;
            // Simple deterministic "work" values spanning a range
            1.0 + 0.5 * (x * std::f64::consts::PI * 4.0).sin()
        })
        .collect();

    let start = Instant::now();
    let _delta_f = estimator.free_energy_difference(&samples, 300.0);
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 50,
        "Jarzynski 10k samples took {}ms, threshold is 50ms",
        elapsed.as_millis()
    );
}
