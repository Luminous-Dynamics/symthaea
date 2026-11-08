// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use std::f64::consts::PI;
use symthaea_fractal_time_lab::exact_floquet::KickedIsingFloquet;
use symthaea_fractal_time_lab::frequency_locking::{
    FrequencyLockingConfig, PulseSweepTrace, assess_frequency_locking,
};

const SPINS: usize = 8;
const PERIODS: usize = 512;
const ANALYSIS_START: usize = 64;

fn evaluate_sweep(
    state_class: &str,
    initial_basis: usize,
    coupling: f64,
    detunings: &[f64],
) -> Result<(), Box<dyn Error>> {
    let mut signals = Vec::with_capacity(detunings.len());
    for detuning in detunings {
        let mut model = KickedIsingFloquet::new(
            SPINS,
            vec![coupling; SPINS - 1],
            vec![0.0; SPINS],
            PI + detuning,
        )?;
        model.set_initial_basis(initial_basis)?;
        signals.push(model.stroboscopic_magnetization(PERIODS));
    }

    let traces: Vec<PulseSweepTrace<'_>> = detunings
        .iter()
        .zip(signals.iter())
        .map(|(pulse_detuning_rad, signal)| PulseSweepTrace {
            pulse_detuning_rad: *pulse_detuning_rad,
            signal,
        })
        .collect();

    let config = FrequencyLockingConfig {
        analysis_start: ANALYSIS_START,
        target_frequency: 0.5,
        search_half_width: 0.12,
        frequency_tolerance: 0.005,
        minimum_peak_amplitude: 0.05,
        minimum_spectral_prominence: 2.0,
        minimum_sign_coherence: 0.80,
        minimum_contiguous_points: 3,
    };
    let report = assess_frequency_locking(&traces, config)?;

    println!(
        "{state_class},{coupling:.6},{},{:.9},{:.9},{:.9},{:.9}",
        report.locked_points,
        report.longest_contiguous_points,
        report.observed_locking_span_rad,
        report
            .points
            .iter()
            .map(|point| point.peak.frequency)
            .fold(f64::INFINITY, f64::min),
        report
            .points
            .iter()
            .map(|point| point.peak.frequency)
            .fold(f64::NEG_INFINITY, f64::max),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let detunings = [
        -0.40_f64, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20,
        0.25, 0.30, 0.35, 0.40,
    ];

    println!("Symthaea exact kicked-Ising finite-sweep frequency-locking experiment");
    println!("epistemic_status=FINITE_SYSTEM_FREQUENCY_LOCKING_NOT_DTC_PHASE_QUALIFICATION");
    println!("model=kicked_ising_open_chain");
    println!("spins={SPINS}");
    println!("periods={PERIODS}");
    println!("analysis_start={ANALYSIS_START}");
    println!("detuning_points={}", detunings.len());
    println!("target_frequency_cycles_per_drive=0.5");
    println!("frequency_tolerance=0.005");
    println!("minimum_contiguous_points=3");
    println!(
        "state_class,Jzz,locked_points,longest_contiguous_points,observed_locking_span_rad,min_peak_frequency,max_peak_frequency"
    );

    for state in [("all_up", 0_usize), ("single_defect", 1 << (SPINS - 1))] {
        for coupling in [0.0_f64, 0.5] {
            evaluate_sweep(state.0, state.1, coupling, &detunings)?;
        }
    }

    println!("J=0 is the noninteracting pulse-control baseline.");
    println!(
        "The report is a finite-system frequency-locking screen; it is not a thermodynamic DTC phase proof."
    );
    Ok(())
}
