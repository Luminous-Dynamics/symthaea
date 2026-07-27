// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MA-001R-delta methodology check — tests whether `shuffled_collapses`'s own comparison baseline
//! (the equal-outcome control) is fair to judge the shuffled-context control against, given that
//! `shuffled_collapses` has now failed under every mechanism tried (belief, gated observation,
//! raw observation) despite the raw-observation variant otherwise passing 4/5 gates.
//!
//! Hypothesis: equal-outcome's target is *constant* (0.6 every tick) -- trivial to converge to,
//! near-zero residual coefficient movement. Shuffled's target *alternates* (0.9/0.2 by tick
//! parity, identical to Bound) -- a structurally harder, higher-variance target regardless of
//! whether context correlates with it. Comparing shuffled against equal-outcome may therefore be
//! comparing two controls that differ in target variance, not just context-outcome correlation.
//!
//! This example adds a third control,
//! `run_with_delta_rule_from_raw_observation_balanced_decorrelated`: outcome alternates by tick
//! parity exactly like Bound/Shuffled, but context alternates on a period-4 block schedule (A,A,
//! B,B, repeat) instead of tick parity or per-tick randomization -- over each 4-tick cycle both
//! contexts see an *exactly balanced* mean outcome ({0.9,0.2}, mean 0.55 each), a genuinely
//! decorrelated-but-fully-varying exposure. An earlier draft of this control (fixed always-
//! Context-B) was found, before being reported, to be mechanistically degenerate: Context B's own
//! social fields are all zero except `partner_present`, so three of four coefficients' gradients
//! were exactly zero every tick, and the one coefficient that could move canceled out
//! symmetrically in `delta_predicted` by construction -- that version tested nothing, and is
//! disclosed here rather than silently discarded.
//!
//! Run: `cargo run -p symthaea-alife --example ma001r_delta_methodology_check --release`

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

fn main() {
    let cfg = Ma001rConfig::default();
    println!(
        "MA-001R-delta methodology check -- resource_level={} outcome_a={} outcome_b={} training_ticks={}\n",
        cfg.resource_level, cfg.outcome_a, cfg.outcome_b, cfg.training_ticks
    );

    println!("=== Equal-outcome control (constant target 0.6) ===");
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

    println!("=== Shuffled-context control (alternating target, per-tick randomized context) ===");
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

    println!(
        "=== NEW (corrected): balanced-decorrelated control (alternating target, period-4 block context, exactly balanced mean outcome per context) ==="
    );
    let mut balanced_probe = Ma001rProbe::new(social_cfg(), SEED, cfg);
    balanced_probe.set_learning_pathway(false, false);
    let balanced_delta_rule = DeltaRuleLearner::new(
        DeltaRuleConfig::default(),
        &balanced_probe.organism.agent.model,
    );
    let balanced_baseline = balanced_probe.counterfactual_reading();
    balanced_probe.run_with_delta_rule_from_raw_observation_balanced_decorrelated(
        &balanced_delta_rule,
        0,
        cfg.training_ticks,
    );
    let balanced_post = balanced_probe.counterfactual_reading();
    println!(
        "  baseline delta={:.4} post-training delta={:.4}\n",
        balanced_baseline.delta_predicted, balanced_post.delta_predicted
    );

    println!("=== Analysis ===");
    println!(
        "  equal-outcome (constant target):                         post-training delta = {:.4}",
        equal_post.delta_predicted
    );
    println!(
        "  balanced-decorrelated (alternating target, real context variation, balanced mean outcome): post-training delta = {:.4}",
        balanced_post.delta_predicted
    );
    println!(
        "  shuffled (alternating target, per-tick randomized context):        post-training delta = {:.4}",
        shuffled_post.delta_predicted
    );
    println!(
        "  bound (alternating target, correctly-bound context):               post-training delta = 0.7656 (known from prior run)\n"
    );

    let balanced_close_to_equal = (balanced_post.delta_predicted - equal_post.delta_predicted)
        .abs()
        <= equal_post.delta_predicted.max(1e-9) * 0.20;
    let balanced_close_to_shuffled =
        (balanced_post.delta_predicted - shuffled_post.delta_predicted).abs()
            <= shuffled_post.delta_predicted.max(1e-9) * 0.20;

    println!(
        "VERDICT: {}",
        if balanced_close_to_shuffled && !balanced_close_to_equal {
            "CONFIRMED: the balanced-decorrelated control (alternating target, real context \
            variation, zero true context-outcome correlation) produces a delta_predicted close to \
            Shuffled's own, and far from Equal-outcome's. This means shuffled_collapses's failure \
            is a CONTROL-DESIGN ARTIFACT -- Equal-outcome (constant target) is not a fair \"no \
            coupling\" baseline for Shuffled (alternating target); the alternating target ALONE, \
            with zero true context-outcome correlation, already produces comparable coefficient \
            movement. shuffled_collapses should be re-baselined against this control, not \
            equal-outcome."
        } else if balanced_close_to_equal && !balanced_close_to_shuffled {
            "REFUTED: the balanced-decorrelated control's movement is close to Equal-outcome's own \
            low value, not Shuffled's larger one -- meaning Shuffled's own extra movement beyond \
            this control IS attributable to something about the per-tick-randomized (vs. \
            fixed-schedule) context presentation specifically, not simply the alternating target. \
            shuffled_collapses's comparison against equal-outcome may be more defensible than \
            hypothesized."
        } else {
            "AMBIGUOUS: the balanced-decorrelated control's value is not clearly close to either \
            equal-outcome's or shuffled's -- report the exact numbers and reconsider the \
            hypothesis rather than forcing a verdict either way."
        }
    );
}
