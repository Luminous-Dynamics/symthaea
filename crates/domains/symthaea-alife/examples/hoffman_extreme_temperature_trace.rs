// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Diagnostic, not a test: traces why `action_temperature=0.0001` produced a qualitative
//! reversal (fine's energy exceeded coarse's; decision-correctness gap collapsed) in
//! `examples/hoffman_temperature_sweep.rs`, left explicitly unresolved in
//! `HOFFMAN_INTERFACE_THEORY_PLAN_2026-07-22.md`'s "Sharper decisions" section.
//!
//! `ActiveInferenceAgent::efe_computer`/`::model`/`::belief` are all public, and
//! `ExpectedFreeEnergyComputer::compute(action, state, model) -> ExpectedFreeEnergyResult` is
//! public too -- so both actions' full pragmatic/epistemic/novelty breakdown can be computed
//! directly, without going through `select_action()`'s stochastic sampling at all. This checks
//! the leading hypothesis from the plan doc: at extreme low temperature, the softmax approaches
//! a hard argmax over TOTAL EFE (pragmatic + epistemic - novelty) -- if epistemic/novelty
//! differences between actions become large relative to the (now hugely amplified by 1/temperature
//! -- no, temperature only scales the SOFTMAX exponent, not the raw EFE values themselves, so this
//! isn't about relative magnitude growing, it's about which raw term's SIGN decides the argmax)
//! pragmatic difference, the winning action may no longer track resource belief at all.
//!
//! Method: clone the agent (never touches the real organism's state), call
//! `efe_computer.clone().compute(action, &belief, &model)` for both actions directly, log the
//! full breakdown alongside true resource level and current energy, for both fine and coarse,
//! at temperature=0.0001, over a representative window.

use symthaea_alife::{Environment, Organism, OrganismConfig};

const TICKS: u64 = 4_000;
const GRAIN_FINE: f64 = 0.02;
const GRAIN_COARSE: f64 = 0.4;
const EXTREME_TEMPERATURE: f64 = 0.0001;
const SEED: u64 = 1;
const FORAGE: usize = 0;
const REST: usize = 1;

fn calibrated_environment() -> Environment {
    Environment {
        mean: 0.20,
        amplitude: 0.20,
        period: 200.0,
        noise_seed: 0xA5A5_1234_DEAD_BEEF,
        noise_amplitude: 0.02,
    }
}

fn calibrated_config(grain: f64) -> OrganismConfig {
    OrganismConfig {
        forage_efficiency: 0.6,
        perceptual_grain: Some(grain),
        resource_prior: 0.0,
        action_temperature: EXTREME_TEMPERATURE,
        ..OrganismConfig::default()
    }
}

fn trace(label: &str, grain: f64) {
    let mut organism = Organism::new(calibrated_config(grain), SEED);
    let env = calibrated_environment();

    println!("=== {label} (grain={grain}) ===");
    println!(
        "{:>5} {:>8} {:>8} {:>8} | {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10} | {:>6}",
        "t",
        "r_true",
        "belief_r",
        "energy",
        "prag_F",
        "epis_F",
        "total_F",
        "prag_R",
        "epis_R",
        "total_R",
        "action"
    );

    for t in 0..TICKS {
        let r_true = env.resource_at(t);

        // Diagnostic: compute BOTH actions' full EFE breakdown directly, bypassing
        // select_action()'s stochastic sampling. Clone efe_computer so `.compute()`'s internal
        // action_history mutation never touches the real organism.
        let belief_before = organism.agent.belief.clone();
        let mut efe_probe = organism.agent.efe_computer.clone();
        let efe_forage = efe_probe.compute(FORAGE, &belief_before, &organism.agent.model);
        let mut efe_probe2 = organism.agent.efe_computer.clone();
        let efe_rest = efe_probe2.compute(REST, &belief_before, &organism.agent.model);

        let tick = organism.tick(r_true, None);

        // Sample every 200th tick after burn-in, plus the first 20 raw ticks, to see both the
        // early transient and the settled regime without an unreadable flood of output.
        if (t < 20) || (t >= TICKS / 4 && t % 200 == 0) {
            println!(
                "{:>5} {:>8.4} {:>8.4} {:>8.4} | {:>10.4} {:>10.4} {:>10.4} | {:>10.4} {:>10.4} \
                 {:>10.4} | {:>6}",
                t,
                r_true,
                belief_before.mean[0],
                organism.energy,
                efe_forage.pragmatic,
                efe_forage.epistemic,
                efe_forage.total,
                efe_rest.pragmatic,
                efe_rest.epistemic,
                efe_rest.total,
                tick.action,
            );
        }
    }
    println!();
}

fn main() {
    println!("Hoffman extreme-temperature trace (temperature={EXTREME_TEMPERATURE}, seed={SEED})");
    println!(
        "Columns: prag/epis/total are pragmatic/epistemic/total EFE for each action (lower total \
         wins). action: 0=Forage, 1=Rest."
    );
    println!();
    trace("FINE", GRAIN_FINE);
    trace("COARSE", GRAIN_COARSE);
}
