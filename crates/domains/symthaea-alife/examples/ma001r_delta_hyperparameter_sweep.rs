// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MA-001R-delta hyperparameter sweep — tests whether tuning `DeltaRuleConfig` on the
//! ALREADY-WORKING gated-observation mechanism
//! (`Ma001rProbe::run_with_delta_rule_from_observation`, `ALIFE_MA001L_CONTEXTUAL_TRANSITION_LEARNING_PLAN_2026-07-26.md`
//! §12's diagnosed fix) can close the three gates that mechanism still fails at default settings
//! (`shuffled_collapses`, `held_out_confirms`, `reversal_flips_and_holds`), without changing the
//! signal source itself.
//!
//! Runs the identical protocol as `ma001r_delta_run.rs`'s `run_protocol` (baseline, 2000-tick
//! Bound training, held-out check, equal-outcome control, shuffled-context control, reversal at
//! 2000 ticks checked at mid/end) once per `DeltaRuleConfig` variant, and reports structured gate
//! results plus explicit instability diagnostics (coefficient saturation at the clip bound,
//! non-finite predictions) before any prose verdict — per this research arc's standing discipline.
//!
//! Run: `cargo run -p symthaea-alife --example ma001r_delta_hyperparameter_sweep --release`

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

/// Structured gate results, identical criteria to `ma001r_delta_run.rs` — computed before any
/// prose verdict.
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

    fn pass_count(&self) -> usize {
        [
            self.direction_correct,
            self.separates_from_equal_outcome,
            self.shuffled_collapses,
            self.held_out_confirms,
            self.reversal_flips_and_holds,
        ]
        .iter()
        .filter(|&&b| b)
        .count()
    }
}

/// Explicit instability diagnostics — a config that "passes" gates only because it collapsed to a
/// degenerate state (coefficients pinned at the clip bound, non-finite predictions) is not a real
/// success and must be flagged as such, not silently reported as a pass.
#[derive(Debug, Clone, Copy, Default)]
struct StabilityDiagnostics {
    /// True iff every prediction value inspected across the whole protocol is finite (no NaN/inf).
    all_finite: bool,
    /// True iff at least one post-training transition-matrix coefficient sits within 1e-6 of
    /// `+clip_bound` or `-clip_bound` — i.e. it saturated against the clamp rather than converging.
    coeff_saturated: bool,
}

impl StabilityDiagnostics {
    fn is_stable(&self) -> bool {
        self.all_finite && !self.coeff_saturated
    }
}

/// One sweep configuration's full result: gates, stability diagnostics, and the key raw numbers
/// needed to report exact figures (not vague summaries) in the final table.
struct SweepResult {
    label: &'static str,
    /// The config that produced this row. Retained as provenance so a result can never be read
    /// apart from the hyperparameters that generated it; the current printed table identifies
    /// rows by `label` alone, so nothing reads it back yet.
    #[allow(dead_code)]
    delta_cfg: DeltaRuleConfig,
    gates: Ma001rDeltaGateResults,
    stability: StabilityDiagnostics,
    post_training_delta: f64,
    equal_post_delta: f64,
    shuffled_post_delta: f64,
    untrained_ho_a: f64,
    untrained_ho_b: f64,
    bound_ho_a: f64,
    bound_ho_b: f64,
    mid_reversal_a: f64,
    mid_reversal_b: f64,
    post_reversal_a: f64,
    post_reversal_b: f64,
}

fn coeffs_saturated(coeffs: &[[f64; 2]; 4], clip_bound: f64) -> bool {
    const EPS: f64 = 1e-6;
    coeffs
        .iter()
        .any(|row| row.iter().any(|&v| (v.abs() - clip_bound).abs() < EPS))
}

fn all_finite(vals: &[f64]) -> bool {
    vals.iter().all(|v| v.is_finite())
}

/// Runs the full MA-001R protocol using the gated-observation delta-rule mechanism
/// (`run_with_delta_rule_from_observation` / `run_shuffled_with_delta_rule_from_observation`) under
/// `delta_cfg`, and computes structured gate + stability results before any prose verdict. Mirrors
/// `ma001r_delta_run.rs`'s `run_protocol` exactly, parameterized over `DeltaRuleConfig`.
fn run_protocol(label: &'static str, cfg: Ma001rConfig, delta_cfg: DeltaRuleConfig) -> SweepResult {
    println!("############################################");
    println!("### Config: {label}");
    println!(
        "### eta={} decay={} clip_bound={} bias_learning={}",
        delta_cfg.eta, delta_cfg.decay, delta_cfg.clip_bound, delta_cfg.bias_learning
    );
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
    let delta_rule = DeltaRuleLearner::new(delta_cfg, &probe.organism.agent.model);

    let baseline = probe.counterfactual_reading();
    println!(
        "  baseline (pre-training): predicted_energy(A={:.4}, B={:.4}) delta_predicted={:.4}",
        baseline.predicted_energy_a, baseline.predicted_energy_b, baseline.delta_predicted
    );

    probe.run_with_delta_rule_from_observation(&delta_rule, 0, cfg.training_ticks, Schedule::Bound);

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
    let saturated = coeffs_saturated(&coeffs, delta_cfg.clip_bound);
    if saturated {
        println!(
            "  *** WARNING: at least one coefficient saturated at +/-clip_bound ({:.1}) -- this config may be collapsing against the clamp, not converging. ***",
            delta_cfg.clip_bound
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
    let equal_delta_rule = DeltaRuleLearner::new(delta_cfg, &equal_probe.organism.agent.model);
    let equal_baseline = equal_probe.counterfactual_reading();
    equal_probe.run_with_delta_rule_from_observation(
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
    let shuffled_delta_rule =
        DeltaRuleLearner::new(delta_cfg, &shuffled_probe.organism.agent.model);
    let shuffled_baseline = shuffled_probe.counterfactual_reading();
    let mut rng_state = 0x9E3779B97F4A7C15u64 ^ SEED;
    shuffled_probe.run_shuffled_with_delta_rule_from_observation(
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
    probe.run_with_delta_rule_from_observation(
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
    probe.run_with_delta_rule_from_observation(
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

    let gates = Ma001rDeltaGateResults {
        direction_correct,
        separates_from_equal_outcome,
        shuffled_collapses,
        held_out_confirms,
        reversal_flips_and_holds,
    };

    let finite_check_values = [
        post_training.predicted_energy_a,
        post_training.predicted_energy_b,
        post_training.delta_predicted,
        equal_post.delta_predicted,
        shuffled_post.delta_predicted,
        bound_ho_a,
        bound_ho_b,
        mid_reversal.predicted_energy_a,
        mid_reversal.predicted_energy_b,
        post_reversal.predicted_energy_a,
        post_reversal.predicted_energy_b,
    ];
    let stability = StabilityDiagnostics {
        all_finite: all_finite(&finite_check_values),
        coeff_saturated: saturated,
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
    println!("  ALL GATES PASS: {}", gates.all_pass());
    println!(
        "  STABILITY: all_finite={} coeff_saturated={} -> stable={}\n",
        stability.all_finite,
        stability.coeff_saturated,
        stability.is_stable()
    );

    if gates.all_pass() && !stability.is_stable() {
        println!(
            "  *** CAUTION: this config passes all 5 gates but is flagged UNSTABLE (coefficient saturation or non-finite values detected). A pass under saturation/divergence is a degenerate result, not a real success. ***\n"
        );
    }

    SweepResult {
        label,
        delta_cfg,
        gates,
        stability,
        post_training_delta: post_training.delta_predicted,
        equal_post_delta: equal_post.delta_predicted,
        shuffled_post_delta: shuffled_post.delta_predicted,
        untrained_ho_a,
        untrained_ho_b,
        bound_ho_a,
        bound_ho_b,
        mid_reversal_a: mid_reversal.predicted_energy_a,
        mid_reversal_b: mid_reversal.predicted_energy_b,
        post_reversal_a: post_reversal.predicted_energy_a,
        post_reversal_b: post_reversal.predicted_energy_b,
    }
}

fn main() {
    let cfg = Ma001rConfig::default();
    println!(
        "MA-001R-delta hyperparameter sweep -- resource_level={} outcome_a={} outcome_b={} training_ticks={} held_out_ticks={} reversal_ticks={}\n",
        cfg.resource_level,
        cfg.outcome_a,
        cfg.outcome_b,
        cfg.training_ticks,
        cfg.held_out_ticks,
        cfg.reversal_ticks
    );

    let sweep_configs: Vec<(&'static str, DeltaRuleConfig)> = vec![
        (
            "1. Default (eta=0.01, decay=0.001, clip=5.0)",
            DeltaRuleConfig::default(),
        ),
        (
            "2. Higher eta (eta=0.05, decay=0.001, clip=5.0)",
            DeltaRuleConfig {
                eta: 0.05,
                decay: 0.001,
                clip_bound: 5.0,
                bias_learning: false,
            },
        ),
        (
            "3. Much higher eta (eta=0.1, decay=0.001, clip=5.0)",
            DeltaRuleConfig {
                eta: 0.1,
                decay: 0.001,
                clip_bound: 5.0,
                bias_learning: false,
            },
        ),
        (
            "4. No weight decay (eta=0.01, decay=0.0, clip=5.0)",
            DeltaRuleConfig {
                eta: 0.01,
                decay: 0.0,
                clip_bound: 5.0,
                bias_learning: false,
            },
        ),
        (
            "5. Higher eta + no decay (eta=0.05, decay=0.0, clip=5.0)",
            DeltaRuleConfig {
                eta: 0.05,
                decay: 0.0,
                clip_bound: 5.0,
                bias_learning: false,
            },
        ),
        (
            "6. Looser clip bound (eta=0.01, decay=0.001, clip=20.0)",
            DeltaRuleConfig {
                eta: 0.01,
                decay: 0.001,
                clip_bound: 20.0,
                bias_learning: false,
            },
        ),
    ];

    let mut results: Vec<SweepResult> = Vec::new();
    for (label, delta_cfg) in sweep_configs {
        let r = run_protocol(label, cfg, delta_cfg);
        results.push(r);
    }

    println!("############################################");
    println!("### Summary: gate pass/fail across all configs");
    println!("############################################\n");
    println!(
        "{:<58} {:>6} {:>7} {:>7} {:>7} {:>7} {:>5} {:>8}",
        "Config", "Dir", "SepEq", "ShufCol", "HeldOut", "RevFlip", "N/5", "Stable"
    );
    for r in &results {
        println!(
            "{:<58} {:>6} {:>7} {:>7} {:>7} {:>7} {:>5} {:>8}",
            r.label,
            r.gates.direction_correct,
            r.gates.separates_from_equal_outcome,
            r.gates.shuffled_collapses,
            r.gates.held_out_confirms,
            r.gates.reversal_flips_and_holds,
            r.gates.pass_count(),
            r.stability.is_stable(),
        );
    }

    println!("\n############################################");
    println!("### Summary: key raw numbers per config");
    println!("############################################\n");
    for r in &results {
        println!("--- {} ---", r.label);
        println!(
            "  post_training_delta={:.4} equal_post_delta={:.4} shuffled_post_delta={:.4}",
            r.post_training_delta, r.equal_post_delta, r.shuffled_post_delta
        );
        println!(
            "  held_out: untrained(A={:.4}, B={:.4}) bound(A={:.4}, B={:.4})",
            r.untrained_ho_a, r.untrained_ho_b, r.bound_ho_a, r.bound_ho_b
        );
        println!(
            "  reversal: mid(A={:.4}, B={:.4}) post(A={:.4}, B={:.4})",
            r.mid_reversal_a, r.mid_reversal_b, r.post_reversal_a, r.post_reversal_b
        );
        println!();
    }

    println!("############################################");
    println!("### Verdict");
    println!("############################################\n");
    let clean_passes: Vec<&SweepResult> = results
        .iter()
        .filter(|r| r.gates.all_pass() && r.stability.is_stable())
        .collect();
    let unstable_passes: Vec<&SweepResult> = results
        .iter()
        .filter(|r| r.gates.all_pass() && !r.stability.is_stable())
        .collect();

    if !clean_passes.is_empty() {
        println!(
            "PASS (stable): {} config(s) pass all 5 gates without saturation/non-finite values: {}",
            clean_passes.len(),
            clean_passes
                .iter()
                .map(|r| r.label)
                .collect::<Vec<_>>()
                .join(", ")
        );
    } else {
        println!("No config passes all 5 gates cleanly (stably).");
    }
    if !unstable_passes.is_empty() {
        println!(
            "CAUTION: {} config(s) pass all 5 gates but are flagged UNSTABLE (degenerate, not a real success): {}",
            unstable_passes.len(),
            unstable_passes
                .iter()
                .map(|r| r.label)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!(
        "\nBest partial result by gate count: {}",
        results
            .iter()
            .max_by_key(|r| r.gates.pass_count())
            .map(|r| format!(
                "{} ({}/5 gates, stable={})",
                r.label,
                r.gates.pass_count(),
                r.stability.is_stable()
            ))
            .unwrap_or_default()
    );
}
