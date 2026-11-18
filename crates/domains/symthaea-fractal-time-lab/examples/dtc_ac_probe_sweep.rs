// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use std::f64::consts::PI;
use symthaea_fractal_time_lab::{DtcProbeConfig, SubharmonicLifetime, evaluate_dtc_probe_sensing};

fn lifetime_label(lifetime: SubharmonicLifetime) -> String {
    match lifetime {
        SubharmonicLifetime::Observed {
            first_crossing_period,
            ..
        } => format!("observed:{first_crossing_period}"),
        SubharmonicLifetime::RightCensored { horizon_period, .. } => {
            format!("right_censored:{horizon_period}")
        }
        SubharmonicLifetime::NoInitialOrder => "no_initial_order".to_string(),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    const N_SPINS: usize = 8;
    const PERIODS: usize = 160;
    const ANALYSIS_START: usize = 24;
    const PROBE_AMPLITUDE: f64 = 0.035;
    const DERIVATIVE_STEP: f64 = 0.002;
    let couplings = [0.0_f64, 0.25, 0.50, 1.0];
    let frequencies = [0.0_f64, 0.125, 0.25, 0.375, 0.5];

    println!("Symthaea finite-Floquet AC-probe sensing sweep");
    println!("epistemic_status=SMALL_CLOSED_SYSTEM_SAMPLED_PROBE_NOT_DIAMOND_SENSOR_REPRODUCTION");
    println!("readout=single_final_collective_Z_measurement_with_N_plus_1_Hamming_weight_outcomes");
    println!("initial_state=all_up_product");
    println!("n_spins={N_SPINS},periods={PERIODS},analysis_start={ANALYSIS_START}");
    println!(
        "pulse_angle_rad={:.9},probe_amplitude={PROBE_AMPLITUDE:.9}",
        PI + 0.10
    );
    println!("information_metric=classical_Fisher_information_from_exact_total_Z_distribution");
    println!(
        "cost_metric=Fisher_information_per_simulated_Floquet_period_not_full_experiment_cost"
    );
    println!(
        "coupling,probe_frequency,baseline_order,probed_order,baseline_sign_coherence,probed_sign_coherence,baseline_lifetime,probed_lifetime,CFI_per_shot,QFI_per_shot,CFI_AA,CFI_Aphi,CFI_phiphi,QFI_AA,QFI_Aphi,QFI_phiphi,CFI_A_given_phase,QFI_A_given_phase,phase_time_offset_drive_periods_per_radian,CFI_per_period,QFI_per_period,CFI_over_QFI,CFI_within_QFI,joint_CFI_matrix_within_QFIM,norm_error"
    );

    for coupling in couplings {
        for frequency in frequencies {
            let config = DtcProbeConfig {
                n_spins: N_SPINS,
                nearest_neighbor_coupling: coupling,
                longitudinal_fields: vec![0.0; N_SPINS],
                pulse_angle_rad: PI + 0.10,
                periods: PERIODS,
                analysis_start: ANALYSIS_START,
                probe_frequency_cycles_per_drive_period: frequency,
                probe_initial_phase_rad: 0.0,
                probe_amplitude: PROBE_AMPLITUDE,
                amplitude_derivative_step: DERIVATIVE_STEP,
                phase_derivative_step_rad: 0.002,
                lifetime_window: 12,
                lifetime_threshold_fraction: 0.5,
            };

            match evaluate_dtc_probe_sensing(&config) {
                Ok(report) => println!(
                    "{coupling:.6},{frequency:.6},{:.9},{:.9},{:.9},{:.9},{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{},{},{},{:.12e},{:.12e},{},{},{},{:.3e}",
                    report.baseline_subharmonic_response.mean_demodulated_order,
                    report.probed_subharmonic_response.mean_demodulated_order,
                    report.baseline_subharmonic_response.sign_coherence,
                    report.probed_subharmonic_response.sign_coherence,
                    lifetime_label(report.baseline_lifetime),
                    lifetime_label(report.probed_lifetime),
                    report.classical_fisher_information_per_shot,
                    report.quantum_fisher_information_per_shot,
                    report.classical_fisher_matrix_per_shot[0][0],
                    report.classical_fisher_matrix_per_shot[0][1],
                    report.classical_fisher_matrix_per_shot[1][1],
                    report.quantum_fisher_matrix_per_shot[0][0],
                    report.quantum_fisher_matrix_per_shot[0][1],
                    report.quantum_fisher_matrix_per_shot[1][1],
                    report
                        .classical_effective_amplitude_fisher_with_phase_nuisance
                        .map(|value| format!("{value:.12e}"))
                        .unwrap_or_else(|| "undefined".to_string()),
                    report
                        .quantum_effective_amplitude_fisher_with_phase_nuisance
                        .map(|value| format!("{value:.12e}"))
                        .unwrap_or_else(|| "undefined".to_string()),
                    report
                        .phase_time_offset_drive_periods_per_radian
                        .map(|value| format!("{value:.12e}"))
                        .unwrap_or_else(|| "undefined".to_string()),
                    report.fisher_information_per_drive_period,
                    report.quantum_fisher_information_per_drive_period,
                    report
                        .classical_to_quantum_fisher_ratio
                        .map(|value| format!("{value:.9}"))
                        .unwrap_or_else(|| "undefined".to_string()),
                    report.classical_fisher_within_quantum_bound,
                    report.joint_classical_fisher_within_quantum_bound,
                    report.norm_error,
                ),
                Err(error) => println!(
                    "{coupling:.6},{frequency:.6},unavailable,unavailable,unavailable,unavailable,unavailable,unavailable,undefined,undefined,undefined,undefined,undefined,undefined,undefined,undefined,undefined,undefined,undefined,undefined,undefined,undefined,undefined,undefined,error={error}"
                ),
            }
        }
    }

    println!(
        "Interpretation boundary: a high Fisher value is conditional on the exact simulator, final total-Z readout, ideal calibration and repeated identical preparations."
    );
    println!(
        "Not included: bath/noise, spatial dipolar disorder, sensor back-action, NV optical readout, calibration drift, dead time or device sensitivity."
    );
    println!(
        "No result here establishes a thermodynamic DTC phase or reproduces the experimental diamond sensing protocol."
    );
    Ok(())
}
