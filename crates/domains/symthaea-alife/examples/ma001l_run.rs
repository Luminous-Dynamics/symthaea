// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MA-001L — Contextual Transition Learning: 5 arms (7 configs with the bias ablation), 7 frozen
//! gates, per `ALIFE_MA001L_CONTEXTUAL_TRANSITION_LEARNING_PLAN_2026-07-26.md` §6-9.
//!
//! Run: `cargo run -p symthaea-alife --example ma001l_run --release`
//!
//! Per plan §8: a structured, machine-checkable `Ma001lGateResults` is computed for every config
//! *before* any prose verdict is derived — the direct fix for MA-001R's own lesson, where the
//! driver's ad hoc printed heuristic was more lenient than the plan's literal criteria.

use symthaea_alife::ma001l::{
    ACTION_FORAGE, ACTION_REST, ACTION_TRANSFER, ContextLabel, DeltaRuleConfig, DeltaRuleLearner,
    NUM_ACTIONS, OBS_DIM, OUTCOME_A, OUTCOME_B, STATE_DIM, StreamSchedule, Tuple,
    generate_bound_stream, generate_held_out_stream, generate_shuffled_stream,
    held_out_energy_error, predicted_energy_for,
};
use symthaea_fep::{GenerativeModel, TemporalDifferenceLearner, TemporalDifferenceLearningConfig};

const TRAINING_TOTAL: u64 = 4000; // 2000 Transfer + 2000 negative-control, interleaved (plan sec 3)
const HELD_OUT_COUNT: u64 = 200;
const REVERSAL_TOTAL: u64 = 4000; // matches training's own interleaving convention

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LearnerKind {
    HebbianOnly,
    TdOnly,
    Neither,
    DeltaRule,
    DeltaRulePlusTd,
}

struct ArmSpec {
    name: &'static str,
    kind: LearnerKind,
    bias_learning: bool,
}

const ARMS: &[ArmSpec] = &[
    ArmSpec {
        name: "Hebbian-only",
        kind: LearnerKind::HebbianOnly,
        bias_learning: false,
    },
    ArmSpec {
        name: "TD-only",
        kind: LearnerKind::TdOnly,
        bias_learning: false,
    },
    ArmSpec {
        name: "Neither (no-learning control)",
        kind: LearnerKind::Neither,
        bias_learning: false,
    },
    ArmSpec {
        name: "Delta-rule (no bias)",
        kind: LearnerKind::DeltaRule,
        bias_learning: false,
    },
    ArmSpec {
        name: "Delta-rule (with bias)",
        kind: LearnerKind::DeltaRule,
        bias_learning: true,
    },
    ArmSpec {
        name: "Delta-rule+TD (no bias)",
        kind: LearnerKind::DeltaRulePlusTd,
        bias_learning: false,
    },
    ArmSpec {
        name: "Delta-rule+TD (with bias)",
        kind: LearnerKind::DeltaRulePlusTd,
        bias_learning: true,
    },
];

/// Structured, machine-checkable gate results (plan sec 8) -- computed before any prose verdict.
#[derive(Debug, Clone, Copy, Default)]
struct Ma001lGateResults {
    bound_beats_shuffled: bool,             // Gate A
    heldout_beats_baseline: bool,           // Gate B
    counterfactual_direction_correct: bool, // Gate C
    reversal_passes: bool,                  // Gate D
    no_catastrophic_drift: bool,            // Gate E
    action_specificity_passes: bool,        // Gate F
    shuffled_genuinely_fails: bool,         // Gate G
}

impl Ma001lGateResults {
    fn all_pass(&self) -> bool {
        self.bound_beats_shuffled
            && self.heldout_beats_baseline
            && self.counterfactual_direction_correct
            && self.reversal_passes
            && self.no_catastrophic_drift
            && self.action_specificity_passes
            && self.shuffled_genuinely_fails
    }

    fn failed_gates(&self) -> Vec<&'static str> {
        let mut failed = Vec::new();
        if !self.bound_beats_shuffled {
            failed.push("A:bound_beats_shuffled");
        }
        if !self.heldout_beats_baseline {
            failed.push("B:heldout_beats_baseline");
        }
        if !self.counterfactual_direction_correct {
            failed.push("C:counterfactual_direction_correct");
        }
        if !self.reversal_passes {
            failed.push("D:reversal_passes");
        }
        if !self.no_catastrophic_drift {
            failed.push("E:no_catastrophic_drift");
        }
        if !self.action_specificity_passes {
            failed.push("F:action_specificity_passes");
        }
        if !self.shuffled_genuinely_fails {
            failed.push("G:shuffled_genuinely_fails");
        }
        failed
    }
}

struct Learners {
    td: Option<TemporalDifferenceLearner>,
    delta: Option<DeltaRuleLearner>,
}

fn build_learners(kind: LearnerKind, bias_learning: bool, model: &GenerativeModel) -> Learners {
    match kind {
        LearnerKind::HebbianOnly | LearnerKind::Neither => Learners {
            td: None,
            delta: None,
        },
        LearnerKind::TdOnly => Learners {
            td: Some(TemporalDifferenceLearner::new(
                TemporalDifferenceLearningConfig::default(),
                NUM_ACTIONS,
                STATE_DIM,
                OBS_DIM,
            )),
            delta: None,
        },
        LearnerKind::DeltaRule => Learners {
            td: None,
            delta: Some(DeltaRuleLearner::new(
                DeltaRuleConfig {
                    bias_learning,
                    ..Default::default()
                },
                model,
            )),
        },
        LearnerKind::DeltaRulePlusTd => Learners {
            td: Some(TemporalDifferenceLearner::new(
                TemporalDifferenceLearningConfig::default(),
                NUM_ACTIONS,
                STATE_DIM,
                OBS_DIM,
            )),
            delta: Some(DeltaRuleLearner::new(
                DeltaRuleConfig {
                    bias_learning,
                    ..Default::default()
                },
                model,
            )),
        },
    }
}

/// Apply one tuple's update per `kind` (plan sec 2). Frozen order for the composed arm: delta
/// rule first, then TD.
fn apply_tuple(
    kind: LearnerKind,
    learners: &mut Learners,
    model: &mut GenerativeModel,
    tuple: &Tuple,
    timestamp: u64,
) {
    match kind {
        LearnerKind::Neither => {}
        LearnerKind::HebbianOnly => {
            model.learn(&tuple.old_state, &tuple.observation, Some(tuple.action));
        }
        LearnerKind::TdOnly => {
            let td = learners
                .td
                .as_mut()
                .expect("TdOnly arm must have a td learner");
            let td_error = td.observe_transition(
                &tuple.old_state,
                tuple.action,
                &tuple.new_state,
                &tuple.observation,
                model,
                timestamp,
            );
            td.update_model(
                model,
                &tuple.old_state,
                tuple.action,
                &tuple.new_state,
                &tuple.observation,
                td_error,
            );
        }
        LearnerKind::DeltaRule => {
            let delta = learners
                .delta
                .as_ref()
                .expect("DeltaRule arm must have a delta learner");
            delta.update(model, &tuple.old_state, tuple.action, &tuple.new_state);
        }
        LearnerKind::DeltaRulePlusTd => {
            let delta = learners
                .delta
                .as_ref()
                .expect("DeltaRulePlusTd arm must have a delta learner");
            delta.update(model, &tuple.old_state, tuple.action, &tuple.new_state);
            let td = learners
                .td
                .as_mut()
                .expect("DeltaRulePlusTd arm must have a td learner");
            let td_error = td.observe_transition(
                &tuple.old_state,
                tuple.action,
                &tuple.new_state,
                &tuple.observation,
                model,
                timestamp,
            );
            td.update_model(
                model,
                &tuple.old_state,
                tuple.action,
                &tuple.new_state,
                &tuple.observation,
                td_error,
            );
        }
    }
}

fn train(kind: LearnerKind, bias_learning: bool, stream: &[Tuple], model: &mut GenerativeModel) {
    let mut learners = build_learners(kind, bias_learning, model);
    for (i, tuple) in stream.iter().enumerate() {
        apply_tuple(kind, &mut learners, model, tuple, i as u64);
    }
}

fn unconditional_baseline_error(tuples: &[Tuple]) -> f64 {
    let mean_outcome = (OUTCOME_A + OUTCOME_B) / 2.0;
    let errors: Vec<f64> = tuples
        .iter()
        .map(|t| (mean_outcome - t.new_state.mean[1]).abs())
        .collect();
    errors.iter().sum::<f64>() / errors.len() as f64
}

fn counterfactual_direction_correct(model: &GenerativeModel, action: usize) -> bool {
    predicted_energy_for(model, action, ContextLabel::A)
        > predicted_energy_for(model, action, ContextLabel::B)
}

fn counterfactual_magnitude(model: &GenerativeModel, action: usize) -> f64 {
    (predicted_energy_for(model, action, ContextLabel::A)
        - predicted_energy_for(model, action, ContextLabel::B))
    .abs()
}

fn run_arm(arm: &ArmSpec) -> Ma001lGateResults {
    println!("--- Arm: {} ---", arm.name);

    // Fresh, untouched reference model -- doubles as the "Neither" arm's own result and as the
    // pre-training snapshot for gate E's drift check.
    let untouched = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);

    let bound_stream = generate_bound_stream(TRAINING_TOTAL, StreamSchedule::Bound);
    let mut bound_rng = 0x9E3779B97F4A7C15u64;
    let shuffled_stream = generate_shuffled_stream(TRAINING_TOTAL, &mut bound_rng);
    let held_out = generate_held_out_stream(HELD_OUT_COUNT, StreamSchedule::Bound, 2000);

    let mut model_bound = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
    train(arm.kind, arm.bias_learning, &bound_stream, &mut model_bound);

    let mut model_shuffled = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
    train(
        arm.kind,
        arm.bias_learning,
        &shuffled_stream,
        &mut model_shuffled,
    );

    let bound_error = held_out_energy_error(&model_bound, &held_out);
    let shuffled_error = held_out_energy_error(&model_shuffled, &held_out);
    let neither_error = held_out_energy_error(&untouched, &held_out);
    let unconditional_error = unconditional_baseline_error(&held_out);

    println!(
        "  held-out error: bound={bound_error:.4} shuffled={shuffled_error:.4} neither={neither_error:.4} unconditional_baseline={unconditional_error:.4}"
    );

    // Gate A: bound beats shuffled.
    let gate_a = bound_error < shuffled_error;

    // Gate B: bound beats every null baseline.
    let gate_b = bound_error < neither_error
        && bound_error < shuffled_error
        && bound_error < unconditional_error;

    // Gate C: direction-correct counterfactual sensitivity under Transfer.
    let gate_c = counterfactual_direction_correct(&model_bound, ACTION_TRANSFER);
    println!(
        "  counterfactual (Transfer): A={:.4} B={:.4} direction_correct={}",
        predicted_energy_for(&model_bound, ACTION_TRANSFER, ContextLabel::A),
        predicted_energy_for(&model_bound, ACTION_TRANSFER, ContextLabel::B),
        gate_c
    );

    // Gate D: reversal -- continue training model_bound with the reversed contingency, check the
    // sign flips and holds for the final 200 Transfer sub-tuples (400 interleaved tuples).
    let reversal_stream = generate_bound_stream(REVERSAL_TOTAL, StreamSchedule::Reversed);
    let checkpoint_index = (REVERSAL_TOTAL - 400) as usize;
    let mut model_reversal = clone_model(&model_bound);
    {
        let mut learners = build_learners(arm.kind, arm.bias_learning, &model_reversal);
        for (i, tuple) in reversal_stream.iter().enumerate() {
            apply_tuple(
                arm.kind,
                &mut learners,
                &mut model_reversal,
                tuple,
                i as u64,
            );
            if i + 1 == checkpoint_index {
                // no-op marker tick; snapshot taken just after this index below
            }
        }
    }
    // Re-run up to the checkpoint on a fresh copy to get the "held for the final window" reading
    // at the earlier checkpoint too (plan sec 6 Gate D: "flip and hold for the final 200 tuples").
    let mut model_reversal_checkpoint = clone_model(&model_bound);
    {
        let mut learners = build_learners(arm.kind, arm.bias_learning, &model_reversal_checkpoint);
        for (i, tuple) in reversal_stream[..checkpoint_index].iter().enumerate() {
            apply_tuple(
                arm.kind,
                &mut learners,
                &mut model_reversal_checkpoint,
                tuple,
                i as u64,
            );
        }
    }
    let flipped_at_checkpoint =
        counterfactual_direction_correct(&model_reversal_checkpoint, ACTION_TRANSFER);
    let flipped_at_end = counterfactual_direction_correct(&model_reversal, ACTION_TRANSFER);
    // After reversal, the correct (flipped) sign is B > A -- i.e. counterfactual_direction_correct
    // (which checks A > B) must now be FALSE at both checkpoints for a genuine, held flip.
    let gate_d = !flipped_at_checkpoint && !flipped_at_end;
    println!(
        "  reversal: pre-reversal(A>B)={} at_checkpoint(A>B)={} at_end(A>B)={} reversal_passes={}",
        gate_c, flipped_at_checkpoint, flipped_at_end, gate_d
    );

    // Gate E: no catastrophic drift on Forage/Rest self-transition coefficients (physical dims).
    let mut max_drift = 0.0f64;
    for action in [ACTION_FORAGE, ACTION_REST] {
        for dim in 0..2 {
            let pre = untouched.transition_matrices[action][dim][dim];
            let post = model_bound.transition_matrices[action][dim][dim];
            let drift = if pre.abs() > 1e-9 {
                (post - pre).abs() / pre.abs()
            } else {
                (post - pre).abs()
            };
            max_drift = max_drift.max(drift);
        }
    }
    let gate_e = max_drift <= 0.20;
    println!(
        "  max Forage/Rest self-transition drift: {max_drift:.4} (threshold 0.20) no_catastrophic_drift={gate_e}"
    );

    // Gate F: action specificity -- Transfer's coupling must be >= 2x Forage/Rest's.
    let transfer_mag = counterfactual_magnitude(&model_bound, ACTION_TRANSFER);
    let forage_mag = counterfactual_magnitude(&model_bound, ACTION_FORAGE);
    let rest_mag = counterfactual_magnitude(&model_bound, ACTION_REST);
    let gate_f =
        transfer_mag >= 2.0 * forage_mag.max(1e-9) && transfer_mag >= 2.0 * rest_mag.max(1e-9);
    println!(
        "  counterfactual magnitude: Transfer={transfer_mag:.4} Forage={forage_mag:.4} Rest={rest_mag:.4} action_specificity_passes={gate_f}"
    );

    // Gate G: shuffled genuinely fails (within 10% of the neither baseline), not just weaker.
    let gate_g = neither_error.max(1e-9) > 0.0
        && ((shuffled_error - neither_error).abs() / neither_error.max(1e-9)) <= 0.10;
    println!(
        "  shuffled vs neither held-out error: shuffled={shuffled_error:.4} neither={neither_error:.4} shuffled_genuinely_fails={gate_g}"
    );

    let results = Ma001lGateResults {
        bound_beats_shuffled: gate_a,
        heldout_beats_baseline: gate_b,
        counterfactual_direction_correct: gate_c,
        reversal_passes: gate_d,
        no_catastrophic_drift: gate_e,
        action_specificity_passes: gate_f,
        shuffled_genuinely_fails: gate_g,
    };
    println!(
        "  ALL 7 GATES PASS: {} (failed: {:?})\n",
        results.all_pass(),
        results.failed_gates()
    );
    results
}

/// `GenerativeModel` derives `Clone` -- explicit helper only for readability at call sites above.
fn clone_model(model: &GenerativeModel) -> GenerativeModel {
    model.clone()
}

fn main() {
    println!(
        "MA-001L -- training_total={TRAINING_TOTAL} held_out_count={HELD_OUT_COUNT} reversal_total={REVERSAL_TOTAL}\n"
    );
    println!(
        "Per plan sec 8: gates are computed from the structured Ma001lGateResults BEFORE any prose verdict.\n"
    );

    let mut all_results: Vec<(&'static str, Ma001lGateResults)> = Vec::new();
    for arm in ARMS {
        let results = run_arm(arm);
        all_results.push((arm.name, results));
    }

    println!("=== Summary (plan sec 6/8) ===\n");
    println!(
        "{:32} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8}",
        "arm", "A", "B", "C", "D", "E", "F", "G", "ALL"
    );
    let mut any_all_pass = false;
    for (name, r) in &all_results {
        if r.all_pass() {
            any_all_pass = true;
        }
        println!(
            "{:32} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8}",
            name,
            r.bound_beats_shuffled,
            r.heldout_beats_baseline,
            r.counterfactual_direction_correct,
            r.reversal_passes,
            r.no_catastrophic_drift,
            r.action_specificity_passes,
            r.shuffled_genuinely_fails,
            r.all_pass()
        );
    }

    println!(
        "\nVERDICT: {}",
        if any_all_pass {
            "At least one learning-rule config passes all 7 gates -- see plan sec 9 next steps (integrate into a live Organism, repeat MA-001R)."
        } else {
            "No config passes all 7 gates -- per plan exit criteria (sec 10), this is itself a genuinely informative finding, reported honestly regardless."
        }
    );
}
