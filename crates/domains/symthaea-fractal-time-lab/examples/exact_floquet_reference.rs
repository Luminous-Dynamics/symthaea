// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use std::f64::consts::PI;
use symthaea_fractal_time_lab::exact_floquet::KickedIsingFloquet;

fn main() -> Result<(), Box<dyn Error>> {
    let couplings = vec![0.8, 0.6, 0.9, 0.7, 0.5];
    let fields = vec![-0.31, 0.17, 0.24, -0.22, 0.11, 0.29];
    let pulse_angle = PI + 0.08;

    let mut interacting =
        KickedIsingFloquet::new(6, couplings, fields.clone(), pulse_angle)?;
    let mut noninteracting =
        KickedIsingFloquet::new(6, vec![0.0; 5], fields, pulse_angle)?;

    let periods = 32;
    let interacting_signal = interacting.stroboscopic_magnetization(periods);
    let noninteracting_signal = noninteracting.stroboscopic_magnetization(periods);

    println!("Symthaea exact Floquet reference — kicked Ising chain");
    println!("epistemic_status=EXACT_SMALL_SYSTEM_REFERENCE_NOT_DTC_PHASE_PROOF");
    println!("spins=6");
    println!("hilbert_dimension={}", interacting.dimension());
    println!("pulse_angle_rad={pulse_angle:.9}");
    println!("periods={periods}");
    println!("period,interacting_mz,noninteracting_mz");
    for t in 0..=periods {
        println!(
            "{t},{:.12},{:.12}",
            interacting_signal[t], noninteracting_signal[t]
        );
    }
    println!("interacting_norm_squared={:.15}", interacting.norm_squared());
    println!(
        "noninteracting_norm_squared={:.15}",
        noninteracting.norm_squared()
    );
    println!("A period-two trace is not sufficient to establish a DTC phase.");
    println!("Test phase rigidity, interaction dependence, finite-size scaling, and heating.");
    Ok(())
}
