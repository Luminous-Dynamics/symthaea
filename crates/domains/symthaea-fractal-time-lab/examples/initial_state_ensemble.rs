// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use std::f64::consts::PI;
use rustfft::num_complex::Complex;
use symthaea_fractal_time_lab::exact_floquet::{
    KickedIsingFloquet,
    analyze_subharmonic_response,
};

const SPINS: usize = 8;
const PERIODS: usize = 200;
const ANALYSIS_START: usize = 40;
const PULSE_ERROR: f64 = 0.10;
const ENSEMBLE_SEEDS: u64 = 8;

/// Small deterministic generator so the emitted initial-state corpus is reproducible.
#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(f64::MIN_POSITIVE);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

fn basis_state(dimension: usize, basis: usize) -> Vec<Complex<f64>> {
    let mut state = vec![Complex::new(0.0, 0.0); dimension];
    state[basis] = Complex::new(1.0, 0.0);
    state
}

/// Product of independently sampled pure single-spin Bloch-sphere states.
fn random_product_state(spins: usize, seed: u64) -> Vec<Complex<f64>> {
    let mut rng = SplitMix64(seed);
    let mut local: Vec<(f64, f64, f64)> = Vec::with_capacity(spins);
    for _ in 0..spins {
        let z = 2.0 * rng.unit() - 1.0;
        let phase = 2.0 * PI * rng.unit();
        let alpha = ((1.0 + z) * 0.5).sqrt();
        let beta = ((1.0 - z) * 0.5).sqrt();
        local.push((alpha, beta, phase));
    }

    let dimension = 1_usize << spins;
    (0..dimension)
        .map(|basis| {
            let mut amplitude = Complex::new(1.0, 0.0);
            for (spin, (alpha, beta, phase)) in local.iter().enumerate() {
                if basis & (1 << spin) == 0 {
                    amplitude *= *alpha;
                } else {
                    amplitude *= Complex::from_polar(*beta, *phase);
                }
            }
            amplitude
        })
        .collect()
}

/// Complex Gaussian vector normalized by the solver's state setter: a finite-N
/// Haar-distributed pure-state reference ensemble (up to PRNG/Box-Muller quality).
fn haar_like_state(dimension: usize, seed: u64) -> Vec<Complex<f64>> {
    let mut rng = SplitMix64(seed);
    (0..dimension)
        .map(|_| Complex::new(rng.normal(), rng.normal()))
        .collect()
}

fn report_case(
    state_class: &str,
    seed: u64,
    coupling: f64,
    initial_state: &[Complex<f64>],
) -> Result<(), Box<dyn Error>> {
    let mut model = KickedIsingFloquet::new(
        SPINS,
        vec![coupling; SPINS - 1],
        vec![0.0; SPINS],
        PI + PULSE_ERROR,
    )?;
    model.set_initial_state(initial_state)?;
    let signal = model.stroboscopic_magnetization(PERIODS);
    let response = analyze_subharmonic_response(&signal, ANALYSIS_START)
        .ok_or("subharmonic response window was invalid")?;

    println!(
        "{state_class},{seed},{coupling:.6},{:.12},{:.12},{:.12},{},{}",
        response.mean_demodulated_order,
        response.mean_absolute_order,
        response.sign_coherence,
        response.samples,
        (model.norm_squared() - 1.0).abs(),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let dimension = 1_usize << SPINS;
    println!("epistemic_status=FINITE_SYSTEM_INITIAL_STATE_SENSITIVITY_SCREEN_NOT_DTC_PHASE_CLAIM");
    println!("model=kicked_ising_open_chain");
    println!("spins={SPINS}");
    println!("hilbert_dimension={dimension}");
    println!("periods={PERIODS}");
    println!("analysis_start={ANALYSIS_START}");
    println!("pulse_angle_rad={:.12}", PI + PULSE_ERROR);
    println!("fields=zero");
    println!("random_product_seeds={ENSEMBLE_SEEDS}");
    println!("haar_like_seeds={ENSEMBLE_SEEDS}");
    println!("state_class,seed,zz_coupling,mean_demodulated_order,mean_absolute_order,sign_coherence,samples,norm_error");

    for coupling in [0.0, 0.5] {
        report_case("all_up", 0, coupling, &basis_state(dimension, 0))?;
        report_case(
            "all_down",
            0,
            coupling,
            &basis_state(dimension, dimension - 1),
        )?;
        report_case(
            "single_defect",
            0,
            coupling,
            &basis_state(dimension, 1 << (SPINS - 1)),
        )?;
        report_case(
            "alternating_defects",
            0,
            coupling,
            &basis_state(dimension, 0b0101_0101),
        )?;

        for seed in 0..ENSEMBLE_SEEDS {
            report_case(
                "random_product",
                seed,
                coupling,
                &random_product_state(SPINS, 0x5052_4F44 + seed),
            )?;
            report_case(
                "haar_like",
                seed,
                coupling,
                &haar_like_state(dimension, 0x4841_4152 + seed),
            )?;
        }
    }

    println!("The J=0 control can show a period-two response from the pulse alone.");
    println!("Compare full state-class distributions at J=0 and J=0.5; do not infer a DTC phase from this finite chain.");
    Ok(())
}
