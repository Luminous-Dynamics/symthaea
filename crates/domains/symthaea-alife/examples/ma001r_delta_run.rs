// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MA-001R-delta — repeat of MA-001R using the new `DeltaRuleLearner` (MA-001L) in place of
//! TD/Hebbian, per `ALIFE_MA001L_CONTEXTUAL_TRANSITION_LEARNING_PLAN_2026-07-26.md` §9 step 5 and
//! §12's diagnosed fix. Runs **both** mechanisms side by side: the original belief-based
//! `old_state` (diagnosed as failing — the organism's belief is a smeared, inertial running
//! estimate, not an instantaneous per-tick signal) and the gated-observation-based `old_state`
//! (the fix — a clean, single-tick signal matching MA-001L's own synthetic placeholder).
//!
//! Run: `cargo run -p symthaea-alife --example ma001r_delta_run --release`
//!
//! Structured gate results computed before any prose verdict, per the MA-001L/MA-001R lesson.

use symthaea_alife::ma001l::{DeltaRuleConfig, DeltaRuleLearner};
use symthaea_alife::ma001r::{Ma001rConfig, Ma001rProbe, Schedule};
use symthaea_alife::organism::OrganismConfig;

const SEED: u64 = 1;

fn social_cfg() -> OrganismConfig {
    OrganismConfig {
        social_enabled: true,
        ..OrganismConfig::default()
    }
}

/// Structured gate results, mapping onto MA-001R's original §8 interpretation ladder criteria --
/// computed before any prose verdict (the MA-001L lesson applied here too).
#[derive(Debug, Clone, Copy, Default)]
struct Ma001rDeltaGateResults {
    direction_correct: bool,
    separates_from_equal_outcome: bool,
    shuffled_collapses: bool,
    held_out_confirms: bool,
    reversal_flips_and_holds: bool,
}

impl Ma001rDeltaGateResults {
    fn all_pass(&self) -> bool {
        self.direction_correct
            && self.separates_from_equal_outcome
            && self.shuffled_collapses
            && self.held_out_confirms
            && self.reversal_flips_and_holds
    }
}

type RunFn = fn(&mut Ma001rProbe, &DeltaRuleLearner, u64, u64, Schedule);
type RunShuffledFn = fn(&mut Ma001rProbe, &DeltaRuleLearner, u64, u64, &mut u64);

/// Runs the full MA-001R protocol (baseline, Bound training, held-out check, equal-outcome
/// control, shuffled-context control, reversal) using whichever `old_state` mechanism `run`/
/// `run_shuffled` implement, and computes the structured gate results before any prose verdict.
fn run_protocol(
    label: &str,
    cfg: Ma001rConfig,
    run: RunFn,
    run_shuffled: RunShuffledFn,
) -> Ma001rDeltaGateResults {
    println!("############################################");
    println!("### Mechanism: {label}");
    println!("############################################\n");

    let mut untrained_probe = Ma001rProbe::new(social_cfg(), SEED, cfg);
    untrained_probe.set_learning_pathway(false, false);
    let (untrained_ho_a, untrained_ho_b) = untrained_probe.held_out_check(0, cfg.held_out_ticks);
    println!(
        "Untrained baseline held-out error: context A={untrained_ho_a:.4} context B={untrained_ho_b:.4}\n"
    );

    println!("=== Main Bound training run ===");
    let mut probe = Ma001rProbe::new(social_cfg(), SEED, cfg);
    probe.set_learning_pathway(false, false);
    let delta_rule = DeltaRuleLearner::new(DeltaRuleConfig::default(), &probe.organism.agent.model);

    let baseline = probe.counterfactual_reading();
    println!(
        "  baseline (pre-training): predicted_energy(A={:.4}, B={:.4}) delta_predicted={:.4}",
        baseline.predicted_energy_a, baseline.predicted_energy_b, baseline.delta_predicted
    );

    run(
        &mut probe,
        &delta_rule,
        0,
        cfg.training_ticks,
        Schedule::Bound,
    );

    let post_training = probe.counterfactual_reading();
    println!(
        "  post-training: predicted_energy(A={:.4}, B={:.4}) delta_predicted={:.4}",
        post_training.predicted_energy_a,
        post_training.predicted_energy_b,
        post_training.delta_predicted
    );
    let coeffs = probe.raw_social_to_physical_coefficients();
    let names = [
        "partner_present",
        "given_to_partner",
        "received_from_partner",
        "encounter_count",
    ];
    println!(
        "  post-training coefficients (transition_matrices[Transfer][social_dim][physical_dim]):"
    );
    for (row, name) in coeffs.iter().zip(names.iter()) {
        println!(
            "    {name:22} -> resource={:.5} energy={:.5}",
            row[0], row[1]
        );
    }

    let (bound_ho_a, bound_ho_b) = probe.held_out_check(cfg.training_ticks, cfg.held_out_ticks);
    println!(
        "  held-out ({} ticks, no further updates): mean_abs_error(context A)={bound_ho_a:.4} mean_abs_error(context B)={bound_ho_b:.4}\n",
        cfg.held_out_ticks
    );

    println!("=== Equal-outcome control ===");
    let mut equal_probe = Ma001rProbe::new(social_cfg(), SEED, cfg);
    equal_probe.set_learning_pathway(false, false);
    let equal_delta_rule = DeltaRuleLearner::new(
        DeltaRuleConfig::default(),
        &equal_probe.organism.agent.model,
    );
    let equal_baseline = equal_probe.counterfactual_reading();
    run(
        &mut equal_probe,
        &equal_delta_rule,
        0,
        cfg.training_ticks,
        Schedule::EqualOutcome,
    );
    let equal_post = equal_probe.counterfactual_reading();
    println!(
        "  baseline delta={:.4} post-training delta={:.4}\n",
        equal_baseline.delta_predicted, equal_post.delta_predicted
    );

    println!("=== Shuffled-context control ===");
    let mut shuffled_probe = Ma001rProbe::new(social_cfg(), SEED, cfg);
    shuffled_probe.set_learning_pathway(false, false);
    let shuffled_delta_rule = DeltaRuleLearner::new(
        DeltaRuleConfig::default(),
        &shuffled_probe.organism.agent.model,
    );
    let shuffled_baseline = shuffled_probe.counterfactual_reading();
    let mut rng_state = 0x9E3779B97F4A7C15u64 ^ SEED;
    run_shuffled(
        &mut shuffled_probe,
        &shuffled_delta_rule,
        0,
        cfg.training_ticks,
        &mut rng_state,
    );
    let shuffled_post = shuffled_probe.counterfactual_reading();
    println!(
        "  baseline delta={:.4} post-training delta={:.4}\n",
        shuffled_baseline.delta_predicted, shuffled_post.delta_predicted
    );

    println!("=== Reversal condition ===");
    run(
        &mut probe,
        &delta_rule,
        cfg.training_ticks,
        cfg.reversal_ticks / 2,
        Schedule::Reversed,
    );
    let mid_reversal = probe.counterfactual_reading();
    println!(
        "  mid-reversal: predicted_energy(A={:.4}, B={:.4}) delta_predicted={:.4}",
        mid_reversal.predicted_energy_a,
        mid_reversal.predicted_energy_b,
        mid_reversal.delta_predicted
    );
    run(
        &mut probe,
        &delta_rule,
        cfg.training_ticks + cfg.reversal_ticks / 2,
        cfg.reversal_ticks / 2,
        Schedule::Reversed,
    );
    let post_reversal = probe.counterfactual_reading();
    println!(
        "  post-reversal: predicted_energy(A={:.4}, B={:.4}) delta_predicted={:.4}\n",
        post_reversal.predicted_energy_a,
        post_reversal.predicted_energy_b,
        post_reversal.delta_predicted
    );

    let direction_correct = post_training.predicted_energy_a > post_training.predicted_energy_b;
    let separates_from_equal_outcome = post_training.delta_predicted > equal_post.delta_predicted;
    let shuffled_collapses = shuffled_post.delta_predicted <= equal_post.delta_predicted * 1.10;
    let held_out_confirms = bound_ho_a < untrained_ho_a && bound_ho_b < untrained_ho_b;
    let flipped_at_mid = mid_reversal.predicted_energy_b > mid_reversal.predicted_energy_a;
    let flipped_at_end = post_reversal.predicted_energy_b > post_reversal.predicted_energy_a;
    let reversal_flips_and_holds = flipped_at_mid && flipped_at_end;

    let results = Ma001rDeltaGateResults {
        direction_correct,
        separates_from_equal_outcome,
        shuffled_collapses,
        held_out_confirms,
        reversal_flips_and_holds,
    };

    println!("=== Structured gate results ({label}) ===");
    println!("  direction_correct (post-training A>B): {direction_correct}");
    println!(
        "  separates_from_equal_outcome (bound {:.4} > equal {:.4}): {separates_from_equal_outcome}",
        post_training.delta_predicted, equal_post.delta_predicted
    );
    println!(
        "  shuffled_collapses (shuffled {:.4} <= 1.10x equal {:.4}): {shuffled_collapses}",
        shuffled_post.delta_predicted, equal_post.delta_predicted
    );
    println!(
        "  held_out_confirms (bound A {bound_ho_a:.4} < untrained A {untrained_ho_a:.4} AND bound B {bound_ho_b:.4} < untrained B {untrained_ho_b:.4}): {held_out_confirms}"
    );
    println!(
        "  reversal_flips_and_holds (mid={flipped_at_mid} AND end={flipped_at_end}): {reversal_flips_and_holds}"
    );
    println!("  ALL GATES PASS: {}\n", results.all_pass());

    results
}

fn main() {
    let cfg = Ma001rConfig::default();
    println!(
        "MA-001R-delta -- resource_level={} outcome_a={} outcome_b={} training_ticks={} held_out_ticks={} reversal_ticks={}\n",
        cfg.resource_level,
        cfg.outcome_a,
        cfg.outcome_b,
        cfg.training_ticks,
        cfg.held_out_ticks,
        cfg.reversal_ticks
    );

    let belief_results = run_protocol(
        "Belief-based old_state (original, diagnosed as failing)",
        cfg,
        Ma001rProbe::run_with_delta_rule,
        Ma001rProbe::run_shuffled_with_delta_rule,
    );

    let observation_results = run_protocol(
        "Gated-observation-based old_state (sec 12's diagnosed fix)",
        cfg,
        Ma001rProbe::run_with_delta_rule_from_observation,
        Ma001rProbe::run_shuffled_with_delta_rule_from_observation,
    );

    println!("############################################");
    println!("### Summary");
    println!("############################################\n");
    println!(
        "Belief-based:      ALL PASS = {}",
        belief_results.all_pass()
    );
    println!(
        "Observation-based: ALL PASS = {}",
        observation_results.all_pass()
    );
    println!(
        "\nVERDICT: {}",
        if observation_results.all_pass() && !belief_results.all_pass() {
            "The diagnosed fix works: feeding the delta rule the gated observation instead of the belief resolves the sign-flip failure. The delta rule succeeds on a real Organism when given a clean, instantaneous input signal."
        } else if observation_results.all_pass() {
            "Both mechanisms pass -- the diagnosis may not have been the whole story, or belief-based old_state was not as broken as the earlier single-seed run suggested."
        } else {
            "The diagnosed fix does not resolve the failure either -- report exactly which gates still fail and reconsider the root-cause hypothesis, honestly."
        }
    );
}
