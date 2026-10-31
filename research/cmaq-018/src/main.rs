use std::fmt::Write as _;

const SCHEMA: &str = "cmaq-018-counterfactual-branch-fidelity-v0.1";
const EPISODES: usize = 64;
const BASE_SEED: u64 = 0xC0A9_0018_2026_1031;
const SIGN_EPSILON: f64 = 1e-9;
const ACTION_COUNT: usize = 7;
const PAIR_COUNT: usize = ACTION_COUNT * (ACTION_COUNT - 1) / 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    NullEffect,
    Monotone,
    NonMonotone,
    StateDependentReversal,
    Interaction,
    TerminationSeam,
    HiddenRegime,
}

impl Scenario {
    const ALL: [Self; 7] = [
        Self::NullEffect,
        Self::Monotone,
        Self::NonMonotone,
        Self::StateDependentReversal,
        Self::Interaction,
        Self::TerminationSeam,
        Self::HiddenRegime,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::NullEffect => "null_effect",
            Self::Monotone => "monotone",
            Self::NonMonotone => "non_monotone",
            Self::StateDependentReversal => "state_dependent_reversal",
            Self::Interaction => "action_interaction",
            Self::TerminationSeam => "termination_seam",
            Self::HiddenRegime => "hidden_regime",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Model {
    PointOutcome,
    ExplicitEffectHead,
    ShuffledActionLabels,
    SupportAwarePoint,
    Oracle,
}

impl Model {
    const ALL: [Self; 5] = [
        Self::PointOutcome,
        Self::ExplicitEffectHead,
        Self::ShuffledActionLabels,
        Self::SupportAwarePoint,
        Self::Oracle,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::PointOutcome => "point_outcome",
            Self::ExplicitEffectHead => "explicit_effect_head",
            Self::ShuffledActionLabels => "shuffled_action_labels",
            Self::SupportAwarePoint => "support_aware_point",
            Self::Oracle => "oracle",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Intervention {
    id: u8,
    force: f64,
    grip: f64,
}

const ACTIONS: [Intervention; ACTION_COUNT] = [
    Intervention { id: 0, force: 0.00, grip: 0.0 },
    Intervention { id: 1, force: 0.25, grip: 0.0 },
    Intervention { id: 2, force: 0.50, grip: 0.0 },
    Intervention { id: 3, force: 0.75, grip: 0.0 },
    Intervention { id: 4, force: 1.00, grip: 0.0 },
    Intervention { id: 5, force: 0.50, grip: 1.0 },
    Intervention { id: 6, force: 1.00, grip: 1.0 },
];

#[derive(Clone, Copy, Debug)]
struct ObservedHistory {
    id: u64,
    x: f64,
    v: f64,
}

#[derive(Clone, Copy, Debug)]
struct EpisodeCase {
    history: ObservedHistory,
    hidden_regime_positive: bool,
    common_disturbance: f64,
}

#[derive(Clone, Copy, Debug)]
struct Outcome {
    utility: f64,
    next_x: f64,
    next_v: f64,
    terminal: bool,
}

#[derive(Clone, Copy, Debug)]
struct BranchObservation {
    history_id: u64,
    action_id: u8,
    outcome: Outcome,
}

#[derive(Clone, Copy, Debug)]
struct BranchPrediction {
    utility: f64,
    next_x: f64,
    next_v: f64,
    terminal: bool,
}

#[derive(Clone, Copy, Debug)]
struct PairMetrics {
    pair_coverage: f64,
    sign_accuracy: Option<f64>,
    effect_magnitude_mae: Option<f64>,
    rank_tau_b: Option<f64>,
    top1_regret: Option<f64>,
    null_false_effect_rate: Option<f64>,
    branch_coverage: f64,
    branch_outcome_mae: Option<f64>,
    branch_state_mae: Option<f64>,
    terminal_false_safe_rate: Option<f64>,
    branch_prediction_evaluations: usize,
    pair_effect_evaluations: usize,
    oracle_branch_evaluations: usize,
}

#[derive(Clone, Copy, Debug)]
struct Summary {
    scenario: &'static str,
    model: &'static str,
    mean_pair_coverage: f64,
    mean_effect_sign_accuracy: Option<f64>,
    mean_effect_magnitude_mae: Option<f64>,
    mean_rank_tau_b: Option<f64>,
    mean_top1_regret: Option<f64>,
    mean_null_false_effect_rate: Option<f64>,
    mean_branch_coverage: f64,
    mean_branch_outcome_mae: Option<f64>,
    mean_branch_state_mae: Option<f64>,
    mean_terminal_false_safe_rate: Option<f64>,
    mean_branch_prediction_evaluations: f64,
    mean_pair_effect_evaluations: f64,
    mean_oracle_branch_evaluations: f64,
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

fn make_episode_case(seed: u64) -> EpisodeCase {
    let mut rng = SplitMix64::new(seed ^ 0x018B_12A7);
    EpisodeCase {
        history: ObservedHistory {
            id: seed,
            x: -0.35 + 0.70 * rng.unit(),
            v: -0.25 + 0.50 * rng.unit(),
        },
        hidden_regime_positive: rng.unit() >= 0.5,
        common_disturbance: -0.01 + 0.02 * rng.unit(),
    }
}

fn supported_intervention(action: Intervention) -> bool {
    action.force <= 0.75 && action.grip == 0.0
}

fn true_effect(
    scenario: Scenario,
    history: ObservedHistory,
    hidden_regime_positive: bool,
    action: Intervention,
) -> f64 {
    let f = action.force;
    let g = action.grip;

    match scenario {
        Scenario::NullEffect => 0.0,
        Scenario::Monotone => 0.70 * f - 0.25 * f * f + 0.10 * g,
        Scenario::NonMonotone => 0.80 * f - 1.00 * f * f + 0.05 * g,
        Scenario::StateDependentReversal => {
            let slope = if history.x < 0.0 { 0.70 } else { -0.70 };
            slope * f - 0.10 * f * f + 0.05 * g
        }
        Scenario::Interaction => {
            0.20 * f + 0.10 * g + 0.80 * f * g - 0.20 * f * f - 0.15 * g * g
        }
        Scenario::TerminationSeam => 0.25 * f + 0.05 * g,
        Scenario::HiddenRegime => {
            let slope = if hidden_regime_positive { 0.65 } else { -0.65 };
            slope * f - 0.12 * f * f + 0.08 * g
        }
    }
}

fn true_outcome(case: EpisodeCase, scenario: Scenario, action: Intervention) -> Outcome {
    let terminal = scenario == Scenario::TerminationSeam
        && action.force >= 0.75
        && action.grip > 0.50;
    let base = 1.0 - 0.08 * case.history.x.powi(2) - 0.03 * case.history.v.powi(2);
    let utility = base
        + true_effect(
            scenario,
            case.history,
            case.hidden_regime_positive,
            action,
        )
        + case.common_disturbance
        - if terminal { 2.0 } else { 0.0 };

    Outcome {
        utility,
        next_x: case.history.x + 0.12 * action.force + 0.03 * action.grip
            + 0.10 * case.common_disturbance,
        next_v: 0.75 * case.history.v + 0.08 * action.force - 0.04 * action.grip
            + case.common_disturbance,
        terminal,
    }
}

fn point_outcome_prediction(history: ObservedHistory, action: Intervention) -> BranchPrediction {
    // A deliberately simple misspecified predictor with a common-mode offset.
    // The common offset should cancel when taking same-history branch differences.
    let common_bias = 0.20 * history.x + 0.10 * history.v;
    BranchPrediction {
        utility: 1.0 - 0.08 * history.x.powi(2) - 0.03 * history.v.powi(2)
            + common_bias
            + 0.45 * action.force
            + 0.10 * action.grip,
        next_x: history.x + 0.12 * action.force + 0.03 * action.grip,
        next_v: 0.75 * history.v + 0.08 * action.force - 0.04 * action.grip,
        terminal: false,
    }
}

fn shuffled_action(action: Intervention) -> Intervention {
    ACTIONS[ACTION_COUNT - 1 - usize::from(action.id)]
}

fn branch_predictions(
    model: Model,
    scenario: Scenario,
    case: EpisodeCase,
    actions: &[Intervention],
) -> Vec<Option<BranchPrediction>> {
    actions
        .iter()
        .map(|action| match model {
            Model::PointOutcome => Some(point_outcome_prediction(case.history, *action)),
            Model::ExplicitEffectHead => None,
            Model::ShuffledActionLabels => {
                Some(point_outcome_prediction(case.history, shuffled_action(*action)))
            }
            Model::SupportAwarePoint => supported_intervention(*action)
                .then(|| point_outcome_prediction(case.history, *action)),
            Model::Oracle => {
                let truth = true_outcome(case, scenario, *action);
                Some(BranchPrediction {
                    utility: truth.utility,
                    next_x: truth.next_x,
                    next_v: truth.next_v,
                    terminal: truth.terminal,
                })
            }
        })
        .collect()
}

fn predicted_effect(
    model: Model,
    case: EpisodeCase,
    left: Intervention,
    right: Intervention,
    predictions: &[Option<BranchPrediction>],
) -> Option<f64> {
    if model == Model::ExplicitEffectHead {
        let slope = if case.history.x < 0.0 { 0.62 } else { -0.62 };
        return Some(
            slope * (left.force - right.force) + 0.12 * (left.grip - right.grip),
        );
    }

    let left_prediction = predictions[usize::from(left.id)]?;
    let right_prediction = predictions[usize::from(right.id)]?;
    Some(left_prediction.utility - right_prediction.utility)
}

fn rank_tau_b(
    concordant: usize,
    discordant: usize,
    true_ties: usize,
    predicted_ties: usize,
) -> Option<f64> {
    let numerator = concordant as f64 - discordant as f64;
    let left = (concordant + discordant + true_ties) as f64;
    let right = (concordant + discordant + predicted_ties) as f64;
    let denominator = (left * right).sqrt();
    (denominator > 0.0).then_some(numerator / denominator)
}

fn canonical_actions(actions: &[Intervention]) -> Vec<Intervention> {
    let mut canonical = actions.to_vec();
    canonical.sort_by_key(|action| action.id);
    canonical
}

fn evaluate_episode(
    scenario: Scenario,
    model: Model,
    case: EpisodeCase,
    actions: &[Intervention],
) -> PairMetrics {
    let canonical = canonical_actions(actions);
    let branches: Vec<BranchObservation> = canonical
        .iter()
        .map(|action| BranchObservation {
            history_id: case.history.id,
            action_id: action.id,
            outcome: true_outcome(case, scenario, *action),
        })
        .collect();

    let predictions = branch_predictions(model, scenario, case, &canonical);
    let branch_prediction_evaluations = predictions
        .iter()
        .filter(|prediction| prediction.is_some())
        .count();
    let oracle_branch_evaluations = if model == Model::Oracle { ACTION_COUNT } else { 0 };
    let pair_effect_evaluations = if model == Model::ExplicitEffectHead { PAIR_COUNT } else { 0 };

    let total_pairs = canonical.len() * (canonical.len() - 1) / 2;
    let mut covered_pairs = 0usize;
    let mut sign_count = 0usize;
    let mut sign_correct = 0usize;
    let mut effect_error_sum = 0.0;
    let mut effect_error_count = 0usize;
    let mut null_nonzero_count = 0usize;
    let mut concordant = 0usize;
    let mut discordant = 0usize;
    let mut true_ties = 0usize;
    let mut predicted_ties = 0usize;

    for left_index in 0..canonical.len() {
        for right_index in (left_index + 1)..canonical.len() {
            let left_action = canonical[left_index];
            let right_action = canonical[right_index];
            let true_delta = branches[left_index].outcome.utility
                - branches[right_index].outcome.utility;

            if let Some(predicted_delta) = predicted_effect(
                model,
                case,
                left_action,
                right_action,
                &predictions,
            ) {
                covered_pairs += 1;
                effect_error_sum += (predicted_delta - true_delta).abs();
                effect_error_count += 1;

                if scenario == Scenario::NullEffect && predicted_delta.abs() > SIGN_EPSILON {
                    null_nonzero_count += 1;
                }

                if true_delta.abs() > SIGN_EPSILON {
                    sign_count += 1;
                    if predicted_delta * true_delta > 0.0 {
                        sign_correct += 1;
                    }
                }

                if true_delta.abs() <= SIGN_EPSILON {
                    if predicted_delta.abs() > SIGN_EPSILON {
                        true_ties += 1;
                    }
                } else if predicted_delta.abs() <= SIGN_EPSILON {
                    predicted_ties += 1;
                } else if true_delta * predicted_delta > 0.0 {
                    concordant += 1;
                } else {
                    discordant += 1;
                }
            }
        }
    }

    let best_true_utility = branches
        .iter()
        .map(|branch| branch.outcome.utility)
        .fold(f64::NEG_INFINITY, f64::max);

    let baseline = ACTIONS[0];
    let mut selected_action: Option<Intervention> = None;
    let mut selected_score = f64::NEG_INFINITY;
    for action in &canonical {
        if let Some(score) = predicted_effect(model, case, *action, baseline, &predictions) {
            if score > selected_score {
                selected_score = score;
                selected_action = Some(*action);
            }
        }
    }
    let top1_regret = selected_action.map(|action| {
        let selected = branches
            .iter()
            .find(|branch| branch.action_id == action.id)
            .expect("canonical branch must exist");
        (best_true_utility - selected.outcome.utility).max(0.0)
    });

    let branch_coverage = branch_prediction_evaluations as f64 / canonical.len() as f64;
    let branch_outcome_errors: Vec<f64> = branches
        .iter()
        .zip(&predictions)
        .filter_map(|(branch, prediction)| {
            prediction.map(|predicted| (predicted.utility - branch.outcome.utility).abs())
        })
        .collect();
    let branch_state_errors: Vec<f64> = branches
        .iter()
        .zip(&predictions)
        .filter_map(|(branch, prediction)| {
            prediction.map(|predicted| {
                ((predicted.next_x - branch.outcome.next_x).abs()
                    + (predicted.next_v - branch.outcome.next_v).abs())
                    / 2.0
            })
        })
        .collect();
    let terminal_pairs: Vec<(bool, bool)> = branches
        .iter()
        .zip(&predictions)
        .filter_map(|(branch, prediction)| {
            prediction.map(|predicted| (branch.outcome.terminal, predicted.terminal))
        })
        .filter(|(actual_terminal, _)| *actual_terminal)
        .collect();
    let terminal_false_safe_rate = (!terminal_pairs.is_empty()).then(|| {
        terminal_pairs
            .iter()
            .filter(|(_, predicted_terminal)| !predicted_terminal)
            .count() as f64
            / terminal_pairs.len() as f64
    });

    // These assertions make same-history identity and action identity evaluator-owned invariants.
    debug_assert!(branches.iter().all(|branch| branch.history_id == case.history.id));
    debug_assert_eq!(
        branches
            .iter()
            .map(|branch| branch.action_id)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        canonical.len()
    );

    PairMetrics {
        pair_coverage: covered_pairs as f64 / total_pairs.max(1) as f64,
        sign_accuracy: (sign_count > 0).then_some(sign_correct as f64 / sign_count as f64),
        effect_magnitude_mae: (effect_error_count > 0)
            .then_some(effect_error_sum / effect_error_count as f64),
        rank_tau_b: rank_tau_b(concordant, discordant, true_ties, predicted_ties),
        top1_regret,
        null_false_effect_rate: (scenario == Scenario::NullEffect && covered_pairs > 0)
            .then_some(null_nonzero_count as f64 / covered_pairs as f64),
        branch_coverage,
        branch_outcome_mae: (!branch_outcome_errors.is_empty()).then(|| {
            branch_outcome_errors.iter().sum::<f64>() / branch_outcome_errors.len() as f64
        }),
        branch_state_mae: (!branch_state_errors.is_empty()).then(|| {
            branch_state_errors.iter().sum::<f64>() / branch_state_errors.len() as f64
        }),
        terminal_false_safe_rate,
        branch_prediction_evaluations,
        pair_effect_evaluations,
        oracle_branch_evaluations,
    }
}

fn mean_optional(values: &[Option<f64>]) -> Option<f64> {
    let present: Vec<f64> = values.iter().filter_map(|value| *value).collect();
    (!present.is_empty()).then(|| present.iter().sum::<f64>() / present.len() as f64)
}

fn summarize(scenario: Scenario, model: Model) -> Summary {
    let mut rows = Vec::with_capacity(EPISODES);
    for episode in 0..EPISODES {
        let seed = BASE_SEED ^ (episode as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let case = make_episode_case(seed);
        rows.push(evaluate_episode(scenario, model, case, &ACTIONS));
    }

    let denominator = EPISODES as f64;
    Summary {
        scenario: scenario.name(),
        model: model.name(),
        mean_pair_coverage: rows.iter().map(|row| row.pair_coverage).sum::<f64>() / denominator,
        mean_effect_sign_accuracy: mean_optional(
            &rows.iter().map(|row| row.sign_accuracy).collect::<Vec<_>>(),
        ),
        mean_effect_magnitude_mae: mean_optional(
            &rows.iter().map(|row| row.effect_magnitude_mae).collect::<Vec<_>>(),
        ),
        mean_rank_tau_b: mean_optional(
            &rows.iter().map(|row| row.rank_tau_b).collect::<Vec<_>>(),
        ),
        mean_top1_regret: mean_optional(
            &rows.iter().map(|row| row.top1_regret).collect::<Vec<_>>(),
        ),
        mean_null_false_effect_rate: mean_optional(
            &rows.iter().map(|row| row.null_false_effect_rate).collect::<Vec<_>>(),
        ),
        mean_branch_coverage: rows.iter().map(|row| row.branch_coverage).sum::<f64>() / denominator,
        mean_branch_outcome_mae: mean_optional(
            &rows.iter().map(|row| row.branch_outcome_mae).collect::<Vec<_>>(),
        ),
        mean_branch_state_mae: mean_optional(
            &rows.iter().map(|row| row.branch_state_mae).collect::<Vec<_>>(),
        ),
        mean_terminal_false_safe_rate: mean_optional(
            &rows.iter().map(|row| row.terminal_false_safe_rate).collect::<Vec<_>>(),
        ),
        mean_branch_prediction_evaluations: rows
            .iter()
            .map(|row| row.branch_prediction_evaluations as f64)
            .sum::<f64>()
            / denominator,
        mean_pair_effect_evaluations: rows
            .iter()
            .map(|row| row.pair_effect_evaluations as f64)
            .sum::<f64>()
            / denominator,
        mean_oracle_branch_evaluations: rows
            .iter()
            .map(|row| row.oracle_branch_evaluations as f64)
            .sum::<f64>()
            / denominator,
    }
}

fn write_optional(output: &mut String, value: Option<f64>) {
    match value {
        Some(number) if number.is_finite() => {
            write!(output, "{number:.9}").expect("writing to String cannot fail");
        }
        _ => output.push_str("null"),
    }
}

fn render_report() -> String {
    let mut output = String::new();
    write!(
        &mut output,
        "{{\"schema_version\":\"{}\",\"episode_count\":{},\"action_count\":{},\"pair_count_per_episode\":{},\"seed\":{},",
        SCHEMA, EPISODES, ACTION_COUNT, PAIR_COUNT, BASE_SEED
    )
    .expect("writing to String cannot fail");
    output.push_str(
        "\"scenarios\":[\"null_effect\",\"monotone\",\"non_monotone\",\"state_dependent_reversal\",\"action_interaction\",\"termination_seam\",\"hidden_regime\"],\"models\":[\"point_outcome\",\"explicit_effect_head\",\"shuffled_action_labels\",\"support_aware_point\",\"oracle\"],\"actions\":[",
    );
    for (index, action) in ACTIONS.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        write!(
            &mut output,
            "{{\"id\":{},\"force\":{:.3},\"grip\":{:.3},\"training_supported\":{}}}",
            action.id,
            action.force,
            action.grip,
            supported_intervention(*action)
        )
        .expect("writing to String cannot fail");
    }
    output.push_str("],\"summary\":[");

    let mut first = true;
    for scenario in Scenario::ALL {
        for model in Model::ALL {
            let row = summarize(scenario, model);
            if !first {
                output.push(',');
            }
            first = false;

            write!(
                &mut output,
                "{{\"scenario\":\"{}\",\"model\":\"{}\",\"episode_count\":{},\"pair_count_per_episode\":{},\"mean_pair_coverage\":{:.9},",
                row.scenario,
                row.model,
                EPISODES,
                PAIR_COUNT,
                row.mean_pair_coverage
            )
            .expect("writing to String cannot fail");

            output.push_str("\"mean_effect_sign_accuracy\":");
            write_optional(&mut output, row.mean_effect_sign_accuracy);
            output.push_str(",\"mean_effect_magnitude_mae\":");
            write_optional(&mut output, row.mean_effect_magnitude_mae);
            output.push_str(",\"mean_rank_tau_b\":");
            write_optional(&mut output, row.mean_rank_tau_b);
            output.push_str(",\"mean_top1_regret\":");
            write_optional(&mut output, row.mean_top1_regret);
            output.push_str(",\"mean_null_false_effect_rate\":");
            write_optional(&mut output, row.mean_null_false_effect_rate);
            write!(
                &mut output,
                ",\"mean_branch_coverage\":{:.9},\"mean_branch_outcome_mae\":",
                row.mean_branch_coverage
            )
            .expect("writing to String cannot fail");
            write_optional(&mut output, row.mean_branch_outcome_mae);
            output.push_str(",\"mean_branch_state_mae\":");
            write_optional(&mut output, row.mean_branch_state_mae);
            output.push_str(",\"mean_terminal_false_safe_rate\":");
            write_optional(&mut output, row.mean_terminal_false_safe_rate);
            write!(
                &mut output,
                ",\"mean_branch_prediction_evaluations\":{:.3},\"mean_pair_effect_evaluations\":{:.3},\"mean_oracle_branch_evaluations\":{:.3}}}",
                row.mean_branch_prediction_evaluations,
                row.mean_pair_effect_evaluations,
                row.mean_oracle_branch_evaluations
            )
            .expect("writing to String cannot fail");
        }
    }

    output.push_str("]}");
    output
}

fn main() {
    println!("{}", render_report());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_effect_branches_share_exactly_zero_causal_difference() {
        let case = make_episode_case(BASE_SEED);
        let left = true_outcome(case, Scenario::NullEffect, ACTIONS[0]);
        let right = true_outcome(case, Scenario::NullEffect, ACTIONS[6]);
        assert!((left.utility - right.utility).abs() < SIGN_EPSILON);
    }

    #[test]
    fn common_disturbance_cancels_in_same_history_differences() {
        let mut first = make_episode_case(BASE_SEED);
        let mut second = first;
        first.common_disturbance = -0.009;
        second.common_disturbance = 0.009;
        let first_delta = true_outcome(first, Scenario::Monotone, ACTIONS[4]).utility
            - true_outcome(first, Scenario::Monotone, ACTIONS[1]).utility;
        let second_delta = true_outcome(second, Scenario::Monotone, ACTIONS[4]).utility
            - true_outcome(second, Scenario::Monotone, ACTIONS[1]).utility;
        assert!((first_delta - second_delta).abs() < SIGN_EPSILON);
    }

    #[test]
    fn monotone_effect_fixture_has_known_sign() {
        let case = make_episode_case(BASE_SEED);
        let delta = true_outcome(case, Scenario::Monotone, ACTIONS[4]).utility
            - true_outcome(case, Scenario::Monotone, ACTIONS[0]).utility;
        assert!(delta > 0.0);
    }

    #[test]
    fn non_monotone_fixture_has_an_interior_optimum() {
        let case = make_episode_case(BASE_SEED);
        let at_half = true_outcome(case, Scenario::NonMonotone, ACTIONS[2]).utility;
        let at_full = true_outcome(case, Scenario::NonMonotone, ACTIONS[4]).utility;
        assert!(at_half > at_full);
    }

    #[test]
    fn state_dependent_action_effect_reverses_across_histories() {
        let case = make_episode_case(BASE_SEED);
        let negative = EpisodeCase {
            history: ObservedHistory { x: -0.2, ..case.history },
            ..case
        };
        let positive = EpisodeCase {
            history: ObservedHistory { x: 0.2, ..case.history },
            ..case
        };
        let negative_delta = true_outcome(
            negative,
            Scenario::StateDependentReversal,
            ACTIONS[3],
        )
        .utility
            - true_outcome(negative, Scenario::StateDependentReversal, ACTIONS[0]).utility;
        let positive_delta = true_outcome(
            positive,
            Scenario::StateDependentReversal,
            ACTIONS[3],
        )
        .utility
            - true_outcome(positive, Scenario::StateDependentReversal, ACTIONS[0]).utility;
        assert!(negative_delta > 0.0);
        assert!(positive_delta < 0.0);
    }

    #[test]
    fn interaction_fixture_is_not_additively_separable() {
        let case = make_episode_case(BASE_SEED);
        let combined = true_outcome(case, Scenario::Interaction, ACTIONS[6]).utility;
        let force_only = true_outcome(case, Scenario::Interaction, ACTIONS[4]).utility;
        let grip_only = true_outcome(case, Scenario::Interaction, ACTIONS[5]).utility;
        let baseline = true_outcome(case, Scenario::Interaction, ACTIONS[0]).utility;
        let interaction_residual = combined - force_only - grip_only + baseline;
        assert!(interaction_residual > 0.20);
    }

    #[test]
    fn terminal_seam_marks_the_combined_high_effort_branch_unsafe() {
        let case = make_episode_case(BASE_SEED);
        assert!(!true_outcome(case, Scenario::TerminationSeam, ACTIONS[4]).terminal);
        assert!(true_outcome(case, Scenario::TerminationSeam, ACTIONS[6]).terminal);
    }

    #[test]
    fn hidden_regime_changes_action_effect_without_changing_observed_history() {
        let case = make_episode_case(BASE_SEED);
        let positive = EpisodeCase { hidden_regime_positive: true, ..case };
        let negative = EpisodeCase { hidden_regime_positive: false, ..case };
        let positive_delta = true_outcome(positive, Scenario::HiddenRegime, ACTIONS[4]).utility
            - true_outcome(positive, Scenario::HiddenRegime, ACTIONS[0]).utility;
        let negative_delta = true_outcome(negative, Scenario::HiddenRegime, ACTIONS[4]).utility
            - true_outcome(negative, Scenario::HiddenRegime, ACTIONS[0]).utility;
        assert!(positive_delta > 0.0);
        assert!(negative_delta < 0.0);
        assert_eq!(positive.history.id, negative.history.id);
        assert_eq!(positive.history.x, negative.history.x);
    }

    #[test]
    fn same_history_branches_retain_unique_action_identities() {
        let case = make_episode_case(BASE_SEED);
        let branches: Vec<_> = ACTIONS
            .iter()
            .map(|action| BranchObservation {
                history_id: case.history.id,
                action_id: action.id,
                outcome: true_outcome(case, Scenario::Monotone, *action),
            })
            .collect();
        assert!(branches.iter().all(|branch| branch.history_id == case.history.id));
        assert_eq!(
            branches.iter().map(|branch| branch.action_id).collect::<std::collections::BTreeSet<_>>().len(),
            ACTION_COUNT
        );
    }

    #[test]
    fn branch_list_permutation_does_not_change_pairwise_effect_metrics() {
        let case = make_episode_case(BASE_SEED);
        let forward = evaluate_episode(
            Scenario::StateDependentReversal,
            Model::PointOutcome,
            case,
            &ACTIONS,
        );
        let reversed: Vec<_> = ACTIONS.iter().rev().copied().collect();
        let backward = evaluate_episode(
            Scenario::StateDependentReversal,
            Model::PointOutcome,
            case,
            &reversed,
        );
        assert_eq!(forward.pair_coverage.to_bits(), backward.pair_coverage.to_bits());
        assert_eq!(
            forward.effect_magnitude_mae.unwrap().to_bits(),
            backward.effect_magnitude_mae.unwrap().to_bits()
        );
        assert_eq!(forward.top1_regret.unwrap().to_bits(), backward.top1_regret.unwrap().to_bits());
    }

    #[test]
    fn common_mode_prediction_bias_cancels_from_point_model_effects() {
        let case = make_episode_case(BASE_SEED);
        let mut other = case;
        other.history.x += 0.15;
        other.history.v -= 0.12;
        let prediction_first =
            branch_predictions(Model::PointOutcome, Scenario::Monotone, case, &ACTIONS);
        let prediction_second =
            branch_predictions(Model::PointOutcome, Scenario::Monotone, other, &ACTIONS);
        let first = predicted_effect(
            Model::PointOutcome,
            case,
            ACTIONS[4],
            ACTIONS[1],
            &prediction_first,
        )
        .unwrap();
        let second = predicted_effect(
            Model::PointOutcome,
            other,
            ACTIONS[4],
            ACTIONS[1],
            &prediction_second,
        )
        .unwrap();
        assert!((first - second).abs() < SIGN_EPSILON);
    }

    #[test]
    fn support_aware_model_abstains_on_unsupported_interventions() {
        let case = make_episode_case(BASE_SEED);
        let predictions =
            branch_predictions(Model::SupportAwarePoint, Scenario::Monotone, case, &ACTIONS);
        assert!(predictions[0].is_some());
        assert!(predictions[3].is_some());
        assert!(predictions[4].is_none());
        assert!(predictions[5].is_none());
        assert!(predictions[6].is_none());
    }

    #[test]
    fn oracle_has_exact_effects_and_zero_top1_regret() {
        let case = make_episode_case(BASE_SEED);
        let metrics = evaluate_episode(Scenario::HiddenRegime, Model::Oracle, case, &ACTIONS);
        assert_eq!(metrics.effect_magnitude_mae.unwrap(), 0.0);
        assert_eq!(metrics.branch_outcome_mae.unwrap(), 0.0);
        assert_eq!(metrics.branch_state_mae.unwrap(), 0.0);
        assert_eq!(metrics.top1_regret.unwrap(), 0.0);
        assert_eq!(metrics.pair_coverage, 1.0);
    }

    #[test]
    fn null_effect_exposes_false_effect_predictions() {
        let case = make_episode_case(BASE_SEED);
        let metrics = evaluate_episode(Scenario::NullEffect, Model::PointOutcome, case, &ACTIONS);
        assert_eq!(metrics.null_false_effect_rate.unwrap(), 1.0);
    }
}
