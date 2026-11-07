// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use std::f64::consts::PI;
use symthaea_fractal_time_lab::exact_floquet::analyze_subharmonic_response;
use symthaea_fractal_time_lab::exact_heisenberg::KickedHeisenbergFloquet;

fn final_state_distance(left: &KickedHeisenbergFloquet, right: &KickedHeisenbergFloquet) -> f64 {
    left.amplitudes()
        .iter()
        .zip(right.amplitudes())
        .map(|(a, b)| (*a - *b).norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn main() -> Result<(), Box<dyn Error>> {
    const PERIODS: usize = 80;
    const ANALYSIS_START: usize = 20;
    const TROTTER_SLICES: usize = 16;
    const CONVERGENCE_SLICES: usize = 32;

    let lattices = [(2_usize, 2_usize), (2, 3), (3, 3)];
    let pulse_errors = [0.05_f64, 0.10, 0.20];
    let horizontal_xyz = [0.25_f64, 0.10, 0.45];
    let vertical_xyz = [0.18_f64, 0.28, 0.35];

    println!("Symthaea anisotropic-Heisenberg 2D Floquet exploration");
    println!("epistemic_status=FINITE_GRAPH_SECOND_ORDER_TROTTER_NOT_DTC_PHASE_PROOF");
    println!("hamiltonian=Jx*XX+Jy*YY+Jz*ZZ+sum_i(h_i*Z_i)");
    println!("boundary=open");
    println!("initial_state=all_up_product");
    println!("fields=zero");
    println!("periods={PERIODS}");
    println!("analysis_start={ANALYSIS_START}");
    println!("trotter_slices={TROTTER_SLICES}");
    println!("convergence_reference_slices={CONVERGENCE_SLICES}");
    println!(
        "width,height,case,epsilon,mean_demodulated_order,mean_absolute_order,sign_coherence,norm_error,trotter_final_state_distance"
    );

    for (width, height) in lattices {
        let n_spins = width * height;
        for epsilon in pulse_errors {
            for (case, horizontal, vertical) in [
                ("noninteracting-control", [0.0; 3], [0.0; 3]),
                ("anisotropic-xyz", horizontal_xyz, vertical_xyz),
            ] {
                let mut model = KickedHeisenbergFloquet::new_square_lattice(
                    width,
                    height,
                    horizontal,
                    vertical,
                    vec![0.0; n_spins],
                    PI + epsilon,
                    TROTTER_SLICES,
                )?;
                let signal = model.stroboscopic_magnetization(PERIODS);
                let response = analyze_subharmonic_response(&signal, ANALYSIS_START)
                    .expect("finite stroboscopic signal");
                let norm_error = (model.norm_squared() - 1.0).abs();

                let mut convergence_reference = KickedHeisenbergFloquet::new_square_lattice(
                    width,
                    height,
                    horizontal,
                    vertical,
                    vec![0.0; n_spins],
                    PI + epsilon,
                    CONVERGENCE_SLICES,
                )?;
                for _ in 0..PERIODS {
                    convergence_reference.step();
                }
                let distance = final_state_distance(&model, &convergence_reference);

                println!(
                    "{width},{height},{case},{epsilon:.6},{:.9},{:.9},{:.9},{norm_error:.3e},{distance:.6e}",
                    response.mean_demodulated_order,
                    response.mean_absolute_order,
                    response.sign_coherence,
                );
            }
        }
    }

    // Initial-state control: an all-up ordered state versus a single-spin
    // defect under exactly the same 3x3 anisotropic-Hamiltonian protocol.
    let mut all_up = KickedHeisenbergFloquet::new_square_lattice(
        3,
        3,
        horizontal_xyz,
        vertical_xyz,
        vec![0.0; 9],
        PI + 0.10,
        TROTTER_SLICES,
    )?;
    let mut one_spin_defect = KickedHeisenbergFloquet::new_square_lattice(
        3,
        3,
        horizontal_xyz,
        vertical_xyz,
        vec![0.0; 9],
        PI + 0.10,
        TROTTER_SLICES,
    )?;
    one_spin_defect.set_initial_basis(1)?;
    let up_signal = all_up.stroboscopic_magnetization(PERIODS);
    let defect_signal = one_spin_defect.stroboscopic_magnetization(PERIODS);
    let up = analyze_subharmonic_response(&up_signal, ANALYSIS_START).unwrap();
    let defect = analyze_subharmonic_response(&defect_signal, ANALYSIS_START).unwrap();

    println!("initial_state_control=3x3,epsilon=0.10");
    println!(
        "all_up_order={:.9},single_spin_defect_order={:.9}",
        up.mean_demodulated_order, defect.mean_demodulated_order
    );
    println!("A finite-window response is not evidence of thermodynamic time-crystalline order.");
    println!(
        "Trotter distances must be checked under further slice refinement before quantitative claims."
    );
    Ok(())
}
