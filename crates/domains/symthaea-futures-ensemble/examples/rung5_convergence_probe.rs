// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Diagnostic probe, not a gate or a fix: `population_census_collapse_verification.rs` found
//! rung 5 stabilizes at p_true~=0.74 once truly extinct, instead of converging toward 1.0. This
//! directly inspects the underlying `ActiveInferenceAgent`'s belief and precision state while
//! replaying a long constant "fully extinct" observation sequence, to confirm (not just guess at
//! from reading the code) exactly what it converges to and why.
//!
//! Run: `cargo run --example rung5_convergence_probe -p symthaea-futures-ensemble`

use symthaea_futures_ensemble::ecological::FepDrivenGenerator;
use symthaea_futures_state::{ActiveInferenceAgent, Observation, mask_observation};

fn main() {
    let config = FepDrivenGenerator::default().agent_config;
    let mut agent = ActiveInferenceAgent::new(config);
    let mut prev_belief = agent.belief.clone();

    println!("tick   belief_mean   sensory_precision   prior_precision   prediction_error");

    for tick in 0..2000u64 {
        // Constant "fully extinct" observation: raw value 0.0, fully visible.
        let raw_obs = Observation::new(vec![0.0], 1.0, "cohort_survival_fraction");
        let masked = mask_observation(&raw_obs, &agent.belief, &[1.0]);

        let perception = agent.perceive(&masked);
        let new_belief = agent.belief.clone();
        agent.observe_transition(&prev_belief, 0, &new_belief, &masked);
        prev_belief = new_belief;

        if tick % 100 == 0 || tick < 10 {
            println!(
                "{tick:5}  {:.6}      {:.4}              {:.4}            {:.6}",
                agent.belief.mean[0],
                agent.precision.sensory_precision,
                agent.precision.prior_precision,
                perception.free_energy.prediction_error,
            );
        }
    }

    println!(
        "\nGenerativeModel.prior_mean = {:?}",
        agent.model.prior_mean
    );
    println!(
        "GenerativeModel.prior_precision = {:?}",
        agent.model.prior_precision
    );

    // Now project forward with predict_next_state, matching FepDrivenGenerator's extrapolation.
    let mut projected = agent.belief.clone();
    for h in [1u64, 10, 50, 100] {
        let mut p = agent.belief.clone();
        for _ in 0..h {
            p = agent.model.predict_next_state(&p, 0);
        }
        println!("projected fraction at horizon={h}: {:.6}", p.mean[0]);
    }
    let _ = &mut projected;
}
