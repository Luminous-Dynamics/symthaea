// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MA-001R-delta — raw-observation variant, per
//! `ALIFE_MA001L_CONTEXTUAL_TRANSITION_LEARNING_PLAN_2026-07-26.md` §13's disclosed, untested
//! hypothesis: that MA-001R-delta v2's partial-success shortfall (shuffled control doesn't
//! collapse, held-out improves for only one of two contexts, reversal barely moves) is caused by
//! `MarkovBoundaryOperator::gate_observation`'s permeability-based attenuation (a function of the
//! organism's physiological deficit, not present in MA-001L's idealized synthetic tuples) rather
//! than belief inertia. This example runs the identical MA-001R protocol
//! (`ma001r_delta_run.rs`'s pattern) using ONLY the new `*_from_raw_observation` mechanism, which
//! feeds `DeltaRuleLearner` the observation *before* blanket-permeability gating is applied.
//!
//! Run: `cargo run -p symthaea-alife --example ma001r_delta_raw_observation --release`
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

fn main() {
    let cfg = Ma001rConfig::default();
    println!(
        "MA-001R-delta (raw observation) -- resource_level={} outcome_a={} outcome_b={} training_ticks={} held_out_ticks={} reversal_ticks={}\n",
        cfg.resource_level,
        cfg.outcome_a,
        cfg.outcome_b,
        cfg.training_ticks,
        cfg.held_out_ticks,
        cfg.reversal_ticks
    );

    println!("############################################");
    println!("### Mechanism: Raw-observation-based old_state (sec 13's disclosed hypothesis)");
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

    probe.run_with_delta_rule_from_raw_observation(
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
    equal_probe.run_with_delta_rule_from_raw_observation(
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
    shuffled_probe.run_shuffled_with_delta_rule_from_raw_observation(
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
    probe.run_with_delta_rule_from_raw_observation(
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
    probe.run_with_delta_rule_from_raw_observation(
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

    println!("=== Structured gate results (Raw-observation-based old_state) ===");
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

    println!("############################################");
    println!("### Summary vs. MA-001R-delta v2 (gated observation)");
    println!("############################################\n");
    println!("Raw-observation-based: ALL PASS = {}", results.all_pass());
    println!(
        "\nFor comparison, MA-001R-delta v2 (gated observation) reported: direction_correct=true \
        (predicted_energy A=0.6648 > B=0.5196), delta_predicted=0.1955 (vs. MA-001L's clean \
        0.847/0.423), shuffled did NOT collapse (0.1971 > bound's own 0.1955), held-out improved \
        only for context B (0.3388 < 0.3969) and got WORSE for context A (0.2118 > 0.1891), and \
        reversal barely moved by mid/end.\n"
    );
    println!(
        "VERDICT: {}",
        if results.all_pass() {
            "The raw-observation hypothesis is CONFIRMED as (at least sufficient for) full replication: removing permeability-gating attenuation resolves the remaining gate failures from v2."
        } else {
            "The raw-observation hypothesis does NOT fully resolve v2's remaining failures -- report exactly which gates still fail and which numbers moved (or didn't) relative to v2, honestly. Permeability attenuation may be a contributing factor but not the whole story, or may not be the bottleneck at all."
        }
    );
}
