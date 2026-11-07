// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use std::f64::consts::PI;
use symthaea_fractal_time_lab::exact_floquet::{
    KickedIsingFloquet,
    analyze_subharmonic_response,
};

fn main() -> Result<(), Box<dyn Error>> {
    const PERIODS: usize = 300;
    const ANALYSIS_START: usize = 20;
    let lattices = [(2_usize, 2_usize), (2, 3), (3, 3)];
    let couplings = [0.0_f64, 0.5, 1.5];
    let pulse_errors = [0.05_f64, 0.10, 0.20];

    println!("Symthaea exact square-lattice kicked-Ising sweep");
    println!("epistemic_status=FINITE_GRAPH_SUBHARMONIC_RESPONSE_NOT_THERMODYNAMIC_DTC_PROOF");
    println!("periods={PERIODS}");
    println!("analysis_start={ANALYSIS_START}");
    println!("initial_state=all_up_product");
    println!("longitudinal_fields=zero");
    println!("boundary=open");
    println!("width,height,J_h,J_v,theta_error,mean_demodulated_order,mean_absolute_order,sign_coherence");
    for (width, height) in lattices {
        for coupling in couplings {
            for epsilon in pulse_errors {
                let mut model = KickedIsingFloquet::new_square_lattice(
                    width,
                    height,
                    coupling,
                    coupling,
                    vec![0.0; width * height],
                    PI + epsilon,
                )?;
                let signal = model.stroboscopic_magnetization(PERIODS);
                let response = analyze_subharmonic_response(&signal, ANALYSIS_START)
                    .expect("finite, sufficiently long stroboscopic signal");
                println!(
                    "{width},{height},{coupling:.6},{coupling:.6},{epsilon:.6},{:.9},{:.9},{:.9}",
                    response.mean_demodulated_order,
                    response.mean_absolute_order,
                    response.sign_coherence,
                );
            }
        }
    }
    println!("Zero-coupling rows are required negative controls.");
    println!("Do not infer a phase from finite-size subharmonic order alone.");
    Ok(())
}
