// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MA-001R — Social→Physical Coupling Probe: ablations, controls, reversal, interpretation
//! ladder, per `ALIFE_MA001R_SOCIAL_PHYSICAL_COUPLING_PLAN_2026-07-26.md` §7-9.
//!
//! Run: `cargo run -p symthaea-alife --example ma001r_run`
//!
//! Deliberately single-seed (seed 1): this is a mechanism probe on one focal organism under a
//! maximally friendly, fully scripted protocol (plan §1: "deliberately smaller than a population
//! experiment... no claim about emergence"), not a calibrated multi-seed confirmatory test like
//! MA-001A. The plan's exit criteria (§11) do not require multiple seeds for this probe.

use symthaea_alife::ma001r::{CounterfactualReading, Ma001rConfig, Ma001rProbe, Schedule};
use symthaea_alife::organism::OrganismConfig;

const SEED: u64 = 1;

fn social_cfg() -> OrganismConfig {
    OrganismConfig {
        social_enabled: true,
        ..OrganismConfig::default()
    }
}

fn print_reading(label: &str, reading: CounterfactualReading) {
    println!(
        "  {label}: predicted_resource(A={:.4}, B={:.4}) predicted_energy(A={:.4}, B={:.4}) delta_predicted={:.4}",
        reading.predicted_resource_a,
        reading.predicted_resource_b,
        reading.predicted_energy_a,
        reading.predicted_energy_b,
        reading.delta_predicted
    );
}

fn print_coefficients(label: &str, coeffs: [[f64; 2]; 4]) {
    let names = [
        "partner_present",
        "given_to_partner",
        "received_from_partner",
        "encounter_count",
    ];
    println!("  {label} (transition_matrices[Transfer][social_dim][physical_dim]):");
    for (row, name) in coeffs.iter().zip(names.iter()) {
        println!(
            "    {name:22} -> resource={:.5} energy={:.5}",
            row[0], row[1]
        );
    }
}

/// Run one learning-pathway arm (plan §7): baseline reading, train `training_ticks` under
/// `Schedule::Bound`, post-training reading + coefficients + held-out check. Returns
/// `(baseline, post_training, held_out_a, held_out_b)`.
fn run_arm(
    label: &str,
    enable_model_learning: bool,
    enable_td_learning: bool,
    cfg: Ma001rConfig,
) -> (CounterfactualReading, CounterfactualReading, f64, f64) {
    println!(
        "--- Arm: {label} (enable_model_learning={enable_model_learning}, enable_td_learning={enable_td_learning}) ---"
    );
    let mut probe = Ma001rProbe::new(social_cfg(), SEED, cfg);
    probe.set_learning_pathway(enable_model_learning, enable_td_learning);

    let baseline = probe.counterfactual_reading();
    print_reading("baseline (pre-training)", baseline);
    print_coefficients(
        "baseline coefficients",
        probe.raw_social_to_physical_coefficients(),
    );

    probe.run(0, cfg.training_ticks, Schedule::Bound);

    let post_training = probe.counterfactual_reading();
    print_reading("post-training", post_training);
    print_coefficients(
        "post-training coefficients",
        probe.raw_social_to_physical_coefficients(),
    );

    let (held_out_a, held_out_b) = probe.held_out_check(cfg.training_ticks, cfg.held_out_ticks);
    println!(
        "  held-out ({} ticks, no further updates): mean_abs_error(context A)={held_out_a:.4} mean_abs_error(context B)={held_out_b:.4}",
        cfg.held_out_ticks
    );
    println!();

    (baseline, post_training, held_out_a, held_out_b)
}

fn main() {
    let cfg = Ma001rConfig::default();
    println!(
        "MA-001R -- resource_level={} outcome_a={} outcome_b={} training_ticks={} held_out_ticks={} reversal_ticks={}\n",
        cfg.resource_level,
        cfg.outcome_a,
        cfg.outcome_b,
        cfg.training_ticks,
        cfg.held_out_ticks,
        cfg.reversal_ticks
    );

    println!(
        "Gate 0 (crates/core/symthaea-fep/tests/ma001r_gate0_td_learning_cross_dimension.rs): both tests pass, see plan doc sec 2.\n"
    );

    println!("=== Learning-pathway ablations (plan sec 7, corrected) ===\n");
    let (hebbian_base, hebbian_post, hebbian_ho_a, hebbian_ho_b) =
        run_arm("Hebbian-only", true, false, cfg);
    let (td_base, td_post, td_ho_a, td_ho_b) = run_arm("TD-only", false, true, cfg);
    let (both_base, both_post, both_ho_a, both_ho_b) =
        run_arm("Both (crate default)", true, true, cfg);
    let (neither_base, neither_post, neither_ho_a, neither_ho_b) =
        run_arm("Neither (learning-disabled control)", false, false, cfg);

    println!(
        "=== Essential controls (plan sec 6), run under the 'Both' pathway (crate default) ===\n"
    );

    println!("--- Equal-outcome control ---");
    let mut equal_probe = Ma001rProbe::new(social_cfg(), SEED, cfg);
    let equal_base = equal_probe.counterfactual_reading();
    print_reading("baseline", equal_base);
    equal_probe.run(0, cfg.training_ticks, Schedule::EqualOutcome);
    let equal_post = equal_probe.counterfactual_reading();
    print_reading(
        "post-training (outcome constant regardless of context)",
        equal_post,
    );
    println!();

    println!("--- Shuffled-context control ---");
    let mut shuffled_probe = Ma001rProbe::new(social_cfg(), SEED, cfg);
    let shuffled_base = shuffled_probe.counterfactual_reading();
    print_reading("baseline", shuffled_base);
    let mut rng_state = 0x9E3779B97F4A7C15u64 ^ SEED;
    shuffled_probe.run_shuffled(0, cfg.training_ticks, &mut rng_state);
    let shuffled_post = shuffled_probe.counterfactual_reading();
    print_reading(
        "post-training (context independently re-randomized each tick)",
        shuffled_post,
    );
    println!();

    println!(
        "Learning-disabled control == the 'Neither' arm above: baseline delta={:.4}, post-training delta={:.4} (expect == baseline, no learning occurred)\n",
        neither_base.delta_predicted, neither_post.delta_predicted
    );

    println!(
        "=== Reversal condition (plan sec 6), continuing the 'Both' arm's trained organism ===\n"
    );
    let mut reversal_probe = Ma001rProbe::new(social_cfg(), SEED, cfg);
    reversal_probe.run(0, cfg.training_ticks, Schedule::Bound);
    let pre_reversal = reversal_probe.counterfactual_reading();
    print_reading("pre-reversal (end of main training)", pre_reversal);
    print_coefficients(
        "pre-reversal coefficients",
        reversal_probe.raw_social_to_physical_coefficients(),
    );

    reversal_probe.run(
        cfg.training_ticks,
        cfg.reversal_ticks / 2,
        Schedule::Reversed,
    );
    let mid_reversal = reversal_probe.counterfactual_reading();
    print_reading(
        "mid-reversal (halfway through reversal_ticks)",
        mid_reversal,
    );

    reversal_probe.run(
        cfg.training_ticks + cfg.reversal_ticks / 2,
        cfg.reversal_ticks / 2,
        Schedule::Reversed,
    );
    let post_reversal = reversal_probe.counterfactual_reading();
    print_reading("post-reversal (end of reversal_ticks)", post_reversal);
    print_coefficients(
        "post-reversal coefficients",
        reversal_probe.raw_social_to_physical_coefficients(),
    );
    println!();

    // --- Interpretation ladder (plan sec 8) ---
    println!("=== Interpretation (plan sec 8) ===\n");

    let hebbian_moved = (hebbian_post.delta_predicted - hebbian_base.delta_predicted).abs();
    let td_moved = (td_post.delta_predicted - td_base.delta_predicted).abs();
    let both_moved = (both_post.delta_predicted - both_base.delta_predicted).abs();
    let neither_moved = (neither_post.delta_predicted - neither_base.delta_predicted).abs();
    let equal_moved = (equal_post.delta_predicted - equal_base.delta_predicted).abs();
    let shuffled_moved = (shuffled_post.delta_predicted - shuffled_base.delta_predicted).abs();

    println!("delta_predicted shift (|post - baseline|):");
    println!("  Hebbian-only: {hebbian_moved:.4}");
    println!("  TD-only:      {td_moved:.4}");
    println!("  Both:         {both_moved:.4}");
    println!("  Neither:      {neither_moved:.4}");
    println!("  Equal-outcome control: {equal_moved:.4}");
    println!("  Shuffled-context control: {shuffled_moved:.4}");
    println!(
        "  Reversal: pre={:.4} mid={:.4} post={:.4} (direction should invert if genuinely tracking current evidence)",
        pre_reversal.delta_predicted, mid_reversal.delta_predicted, post_reversal.delta_predicted
    );
    println!();
    println!("held-out mean abs error (context A / context B):");
    println!("  Hebbian-only: {hebbian_ho_a:.4} / {hebbian_ho_b:.4}");
    println!("  TD-only:      {td_ho_a:.4} / {td_ho_b:.4}");
    println!("  Both:         {both_ho_a:.4} / {both_ho_b:.4}");
    println!("  Neither:      {neither_ho_a:.4} / {neither_ho_b:.4}");
    println!();

    // A real, if informal (single-seed mechanism probe, not a calibrated multi-seed test),
    // sensitivity floor: the noise/baseline level is whatever Neither + equal-outcome + Hebbian
    // together show (none of them should ever learn anything), so a "real effect" bar is set at
    // 2x the largest of those three -- same calibrated-margin discipline as MA-001A's own noise
    // floor, computed from data actually measured this run, not assumed in advance.
    let noise_floor = [neither_moved, equal_moved, hebbian_moved]
        .into_iter()
        .fold(0.0_f64, f64::max);
    let real_effect_bar = 2.0 * noise_floor.max(1e-6);
    println!(
        "Calibrated real-effect bar = 2 x max(Neither, Equal-outcome, Hebbian-only) = {real_effect_bar:.4}\n"
    );

    let td_real = td_moved >= real_effect_bar;
    let both_real = both_moved >= real_effect_bar;
    let shuffled_collapses = shuffled_moved < real_effect_bar;
    let reversal_inverts = (pre_reversal.predicted_energy_a - pre_reversal.predicted_energy_b)
        .signum()
        != (post_reversal.predicted_energy_a - post_reversal.predicted_energy_b).signum()
        && post_reversal.delta_predicted >= real_effect_bar * 0.5;

    let verdict = if td_real && both_real && shuffled_collapses && reversal_inverts {
        "EXISTING ARCHITECTURE SUCCEEDS -- proceed to re-run MA-001A with a mechanistic reason to expect a different result."
    } else if td_real && both_real && !shuffled_collapses {
        "AMBIGUOUS -- coefficients moved but did not collapse under the shuffled-context control; the apparent coupling may not be tracking the context-outcome correspondence specifically. Needs further investigation before trusting the effect."
    } else if td_real && !reversal_inverts {
        "LEARNING-RULE LIMITATION (partial) -- TD-only shows real coefficient movement under this protocol, but it does not reverse under the reversal condition; the update may be acquiring something closer to an irreversible correlation than genuinely tracking current evidence."
    } else if !td_real {
        "LEARNING-RULE LIMITATION -- TD-only shows no coefficient movement above the calibrated noise floor despite Gate 0 passing in isolation: the linear model could represent the relationship, but the update rule as applied here (competing dimensions, real belief dynamics) cannot reliably acquire it under this protocol."
    } else {
        "FULL NULL -- no arm, including TD-only, shows delta_predicted separating from the calibrated noise floor. A new contextual-learning mechanism would be genuinely justified, not before this rung."
    };
    println!("VERDICT: {verdict}");
}
