// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use std::f64::consts::PI;
use symthaea_fractal_time_lab::exact_floquet::{
    KickedIsingFloquet,
    analyze_subharmonic_response,
};

fn main() -> Result<(), Box<dyn Error>> {
    const PERIODS: usize = 100;
    const ANALYSIS_START: usize = 20;
    let sizes = [4_usize, 6, 8, 10, 12];
    let couplings = [0.0_f64, 0.25, 0.5, 0.75, 1.5];
    let pulse_errors = [0.05_f64, 0.10, 0.20];

    println!("Symthaea exact Floquet finite-window response sweep");
    println!("epistemic_status=SUBHARMONIC_RESPONSE_DIAGNOSTIC_NOT_DTC_PHASE_PROOF");
    println!("periods={PERIODS}");
    println!("analysis_start={ANALYSIS_START}");
    println!("initial_state=all_up_product");
    println!("fields=zero");
    println!("N,J,theta_error,mean_demodulated_order,mean_absolute_order,sign_coherence");
    for n_spins in sizes {
        for coupling in couplings {
            for epsilon in pulse_errors {
                let mut model = KickedIsingFloquet::new(
                    n_spins,
                    vec![coupling; n_spins - 1],
                    vec![0.0; n_spins],
                    PI + epsilon,
                )?;
                let signal = model.stroboscopic_magnetization(PERIODS);
                let response = analyze_subharmonic_response(&signal, ANALYSIS_START)
                    .expect("finite, sufficiently long stroboscopic signal");
                println!(
                    "{n_spins},{coupling:.6},{epsilon:.6},{:.9},{:.9},{:.9}",
                    response.mean_demodulated_order,
                    response.mean_absolute_order,
                    response.sign_coherence,
                );
            }
        }
    }
    println!("The zero-coupling rows are essential negative controls.");
    println!("A finite-window subharmonic response is not by itself evidence of a thermodynamic DTC phase.");
    Ok(())
}
