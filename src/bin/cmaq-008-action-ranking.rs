//! CMAQ-008 v0.3: decision-ranking probe using the real Symthaea CfC model.
//!
//! The model predicts next state through HierarchicalCfCWorldModel::predict.
//! A task reward is evaluated on predicted and evaluator-owned true next states.
//! The no-model baseline is a direct reactive policy selected from the same logged
//! training outcomes. The evaluator oracle is never available to candidate selection.
//!
//! This is a public synthetic diagnostic, not a sealed benchmark or SOTA claim.

use std::process;
use symthaea::dynamics::crate_world_model::{HierarchicalCfCWorldModel, WorldModelConfig};

const DIM: usize = 16;
const HIDDEN: usize = 24;
const TRAIN_SAMPLES: usize = 192;
const EVAL_STATES: usize = 96;
const EPOCHS: usize = 12;
const DATA_SEED: u64 = 0xC0A8_2026_1030;
const MODEL_SEEDS: [u64; 10] = [
    0xA008_0001, 0xA008_0002, 0xA008_0003, 0xA008_0004, 0xA008_0005,
    0xA008_0006, 0xA008_0007, 0xA008_0008, 0xA008_0009, 0xA008_000A,
];
const MODEL_FORWARD_CALLS_PER_DECISION: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1_u32 << 24) as f32)
    }

    fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Law {
    Source,
    ActionEffectReversal,
    InteractionSignReversal,
}

impl Law {
    fn name(self) -> &'static str {
        match self {
            Self::Source => "source_holdout",
            Self::ActionEffectReversal => "action_effect_reversal",
            Self::InteractionSignReversal => "interaction_sign_reversal",
        }
    }

    fn action_coefficient(self) -> f32 {
        match self {
            Self::Source | Self::InteractionSignReversal => 0.25,
            Self::ActionEffectReversal => -0.25,
        }
    }

    fn interaction_coefficient(self) -> f32 {
        match self {
            Self::Source | Self::ActionEffectReversal => 0.06,
            Self::InteractionSignReversal => -0.12,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActionMode {
    Zero,
    CancelSource,
    ReverseSource,
}

impl ActionMode {
    const ALL: [Self; 3] = [Self::Zero, Self::CancelSource, Self::ReverseSource];

    fn index(self) -> usize {
        match self {
            Self::Zero => 0,
            Self::CancelSource => 1,
            Self::ReverseSource => 2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::CancelSource => "cancel_source",
            Self::ReverseSource => "reverse_source",
        }
    }

    fn action(self, state: &[f32]) -> Vec<f32> {
        state.iter().map(|value| {
            if value.abs() <= f32::EPSILON {
                return 0.0;
            }
            match self {
                Self::Zero => 0.0,
                Self::CancelSource => -0.75 * value.signum(),
                Self::ReverseSource => 0.75 * value.signum(),
            }
        }).collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Experience {
    state: Vec<f32>,
    action: Vec<f32>,
    next_state: Vec<f32>,
    action_mode: ActionMode,
}

fn transition(state: &[f32], action: &[f32], law: Law) -> Vec<f32> {
    (0..DIM)
        .map(|i| {
            let neighbour = state[(i + 1) % DIM];
            0.55 * state[i]
                + law.action_coefficient() * action[i]
                + law.interaction_coefficient() * neighbour * action[i]
        })
        .collect()
}

/// The task is to move the next state toward the zero target.
/// Higher (less negative) reward is better.
fn reward(next_state: &[f32]) -> f64 {
    -next_state.iter().map(|value| (*value as f64).powi(2)).sum::<f64>() / DIM as f64
}

fn training_data(seed: u64) -> Vec<Experience> {
    let mut rng = SplitMix64::new(seed);
    (0..TRAIN_SAMPLES)
        .map(|_| {
            let state = (0..DIM).map(|_| rng.signed()).collect::<Vec<_>>();
            let mode = ActionMode::ALL[(rng.next_u64() % ActionMode::ALL.len() as u64) as usize];
            let action = mode.action(&state);
            let next_state = transition(&state, &action, Law::Source);
            Experience { state, action, next_state, action_mode: mode }
        })
        .collect()
}

fn evaluation_states(seed: u64, count: usize) -> Vec<Vec<f32>> {
    let mut rng = SplitMix64::new(seed);
    (0..count).map(|_| (0..DIM).map(|_| rng.signed()).collect()).collect()
}

/// A no-model contextual policy: choose the single action transform with the
/// highest average observed reward in the logged source-law training data.
fn fit_direct_policy(data: &[Experience]) -> ActionMode {
    let mut reward_sums = [0.0_f64; 3];
    let mut counts = [0_usize; 3];
    for row in data {
        let index = row.action_mode.index();
        reward_sums[index] += reward(&row.next_state);
        counts[index] += 1;
    }
    ActionMode::ALL
        .into_iter()
        .max_by(|a, b| {
            let score_a = reward_sums[a.index()] / counts[a.index()].max(1) as f64;
            let score_b = reward_sums[b.index()] / counts[b.index()].max(1) as f64;
            score_a.total_cmp(&score_b)
        })
        .unwrap_or(ActionMode::Zero)
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct RegimeMetrics {
    model_reward: f64,
    direct_reward: f64,
    oracle_reward: f64,
    model_lift_vs_direct: f64,
    model_regret: f64,
    direct_regret: f64,
    model_beats_direct_rate: f64,
    model_harm_rate: f64,
    oracle_action_agreement: f64,
    pairwise_ranking_accuracy: f64,
    reward_prediction_mae: f64,
    next_state_prediction_mse: f64,
    model_action_override_rate: f64,
    sample_count: usize,
}

fn argmax(values: &[f64; 3]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn evaluate_regime(
    model: &HierarchicalCfCWorldModel,
    direct_mode: ActionMode,
    states: &[Vec<f32>],
    law: Law,
) -> Result<RegimeMetrics, String> {
    if states.is_empty() {
        return Err("evaluation states must not be empty".to_owned());
    }

    let mut sums = RegimeMetrics::default();
    let mut pairwise_correct = 0.0_f64;
    let mut pairwise_total = 0.0_f64;

    for state in states {
        let mut predicted_rewards = [0.0_f64; 3];
        let mut actual_rewards = [0.0_f64; 3];
        let mut reward_error = 0.0_f64;
        let mut state_error = 0.0_f64;

        for mode in ActionMode::ALL {
            let index = mode.index();
            let action = mode.action(state);
            let predicted_next = model.predict(state, &action, 1.0)
                .map_err(|error| format!("CfC action prediction failed: {error}"))?;
            if predicted_next.len() != DIM {
                return Err(format!("predicted state dimension {} != {}", predicted_next.len(), DIM));
            }
            let actual_next = transition(state, &action, law);
            predicted_rewards[index] = reward(&predicted_next);
            actual_rewards[index] = reward(&actual_next);
            reward_error += (predicted_rewards[index] - actual_rewards[index]).abs();
            state_error += predicted_next.iter().zip(&actual_next)
                .map(|(predicted, actual)| (*predicted as f64 - *actual as f64).powi(2))
                .sum::<f64>() / DIM as f64;
        }

        let model_choice = argmax(&predicted_rewards);
        let oracle_choice = argmax(&actual_rewards);
        let direct_choice = direct_mode.index();
        let model_reward = actual_rewards[model_choice];
        let direct_reward = actual_rewards[direct_choice];
        let oracle_reward = actual_rewards[oracle_choice];

        for left in 0..3 {
            for right in (left + 1)..3 {
                let predicted_order = predicted_rewards[left].total_cmp(&predicted_rewards[right]);
                let actual_order = actual_rewards[left].total_cmp(&actual_rewards[right]);
                if predicted_order == actual_order || predicted_order == std::cmp::Ordering::Equal {
                    pairwise_correct += 1.0;
                }
                pairwise_total += 1.0;
            }
        }

        sums.model_reward += model_reward;
        sums.direct_reward += direct_reward;
        sums.oracle_reward += oracle_reward;
        sums.model_lift_vs_direct += model_reward - direct_reward;
        sums.model_regret += (oracle_reward - model_reward).max(0.0);
        sums.direct_regret += (oracle_reward - direct_reward).max(0.0);
        sums.model_beats_direct_rate += (model_reward > direct_reward + 1e-12) as u8 as f64;
        sums.model_harm_rate += (model_reward < direct_reward - 1e-12) as u8 as f64;
        sums.oracle_action_agreement += (model_choice == oracle_choice) as u8 as f64;
        sums.reward_prediction_mae += reward_error / ActionMode::ALL.len() as f64;
        sums.next_state_prediction_mse += state_error / ActionMode::ALL.len() as f64;
        sums.model_action_override_rate += (model_choice != direct_choice) as u8 as f64;
        sums.sample_count += 1;
    }

    let denominator = states.len() as f64;
    sums.model_reward /= denominator;
    sums.direct_reward /= denominator;
    sums.oracle_reward /= denominator;
    sums.model_lift_vs_direct /= denominator;
    sums.model_regret /= denominator;
    sums.direct_regret /= denominator;
    sums.model_beats_direct_rate /= denominator;
    sums.model_harm_rate /= denominator;
    sums.oracle_action_agreement /= denominator;
    sums.reward_prediction_mae /= denominator;
    sums.next_state_prediction_mse /= denominator;
    sums.model_action_override_rate /= denominator;
    sums.pairwise_ranking_accuracy = pairwise_correct / pairwise_total.max(1.0);
    Ok(sums)
}

#[derive(Clone, Debug, PartialEq)]
struct SeedResult {
    seed: u64,
    direct_mode: &'static str,
    training_mse: f64,
    training_steps: u64,
    source: RegimeMetrics,
    reversal: RegimeMetrics,
    interaction_shift: RegimeMetrics,
}

fn run_seed(seed: u64) -> Result<SeedResult, String> {
    let training = training_data(DATA_SEED);
    let direct_mode = fit_direct_policy(&training);
    let config = WorldModelConfig {
        hdc_dim: DIM,
        hidden_dim: HIDDEN,
        learning_rate: 0.001,
        momentum: 0.9,
        consciousness_decay: 0.99,
    };
    let mut model = HierarchicalCfCWorldModel::new_with_seed(config, seed)
        .map_err(|error| format!("seeded CfC construction failed: {error}"))?;

    let mut epoch_loss = 0.0_f64;
    for _ in 0..EPOCHS {
        epoch_loss = 0.0;
        for row in &training {
            let loss = model.train_step(&row.state, &row.action, &row.next_state, 1.0)
                .map_err(|error| format!("CfC training failed: {error}"))?;
            epoch_loss += loss as f64;
        }
        epoch_loss /= TRAIN_SAMPLES as f64;
    }

    let source_states = evaluation_states(DATA_SEED ^ 0x1111_2222_3333_4444, EVAL_STATES);
    let reversal_states = evaluation_states(DATA_SEED ^ 0xAAAA_BBBB_CCCC_DDDD, EVAL_STATES);
    let interaction_states = evaluation_states(DATA_SEED ^ 0x1357_9BDF_2468_ACE0, EVAL_STATES);
    let source = evaluate_regime(&model, direct_mode, &source_states, Law::Source)?;
    let reversal = evaluate_regime(&model, direct_mode, &reversal_states, Law::ActionEffectReversal)?;
    let interaction_shift = evaluate_regime(&model, direct_mode, &interaction_states, Law::InteractionSignReversal)?;

    Ok(SeedResult {
        seed,
        direct_mode: direct_mode.name(),
        training_mse: epoch_loss,
        training_steps: model.training_steps,
        source,
        reversal,
        interaction_shift,
    })
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 }
}

fn mean_ci95(values: &[f64]) -> (f64, f64, f64) {
    let average = mean(values);
    if values.len() < 2 { return (average, average, average); }
    let variance = values.iter().map(|value| (value - average).powi(2)).sum::<f64>()
        / (values.len() - 1) as f64;
    let half = 1.96 * (variance / values.len() as f64).sqrt();
    (average, average - half, average + half)
}

#[derive(Clone, Debug, PartialEq)]
struct Aggregate {
    model_reward: f64,
    direct_reward: f64,
    oracle_reward: f64,
    model_lift_vs_direct: f64,
    lift_ci95_low: f64,
    lift_ci95_high: f64,
    model_regret: f64,
    direct_regret: f64,
    model_beats_direct_rate: f64,
    model_harm_rate: f64,
    oracle_action_agreement: f64,
    pairwise_ranking_accuracy: f64,
    reward_prediction_mae: f64,
    next_state_prediction_mse: f64,
    model_action_override_rate: f64,
}

fn aggregate(results: &[SeedResult], law: Law) -> Aggregate {
    let metrics = results.iter().map(|row| match law {
        Law::Source => row.source,
        Law::ActionEffectReversal => row.reversal,
        Law::InteractionSignReversal => row.interaction_shift,
    }).collect::<Vec<_>>();
    let lift = metrics.iter().map(|row| row.model_lift_vs_direct).collect::<Vec<_>>();
    let (mean_lift, low, high) = mean_ci95(&lift);
    let avg = |select: fn(RegimeMetrics) -> f64| mean(&metrics.iter().map(|row| select(*row)).collect::<Vec<_>>());
    Aggregate {
        model_reward: avg(|row| row.model_reward),
        direct_reward: avg(|row| row.direct_reward),
        oracle_reward: avg(|row| row.oracle_reward),
        model_lift_vs_direct: mean_lift,
        lift_ci95_low: low,
        lift_ci95_high: high,
        model_regret: avg(|row| row.model_regret),
        direct_regret: avg(|row| row.direct_regret),
        model_beats_direct_rate: avg(|row| row.model_beats_direct_rate),
        model_harm_rate: avg(|row| row.model_harm_rate),
        oracle_action_agreement: avg(|row| row.oracle_action_agreement),
        pairwise_ranking_accuracy: avg(|row| row.pairwise_ranking_accuracy),
        reward_prediction_mae: avg(|row| row.reward_prediction_mae),
        next_state_prediction_mse: avg(|row| row.next_state_prediction_mse),
        model_action_override_rate: avg(|row| row.model_action_override_rate),
    }
}

fn render(results: &[SeedResult]) -> String {
    let source = aggregate(results, Law::Source);
    let reversal = aggregate(results, Law::ActionEffectReversal);
    let interaction_shift = aggregate(results, Law::InteractionSignReversal);
    let mut out = format!(
        concat!(
            "{{\n",
            "  \"schema_version\": \"cmaq-008-action-ranking-v0.2\",\n",
            "  \"model\": \"HierarchicalCfCWorldModel\",\n",
            "  \"dimensions\": {{\"hdc\": {}, \"hidden\": {}}},\n",
            "  \"training\": {{\"samples\": {}, \"epochs\": {}, \"seeds\": {}, \"learning_rate\": 0.001}},\n",
            "  \"evaluation_states_per_regime_per_seed\": {},\n",
            "  \"candidate_actions\": [\"zero\", \"cancel_source\", \"reverse_source\"],\n",
            "  \"model_forward_calls_per_decision\": {},\n",
            "  \"direct_policy\": \"best constant action transform selected from logged training outcomes; zero model forward calls at evaluation\",\n",
            "  \"reward\": \"negative mean squared next-state distance to zero target\",\n",
            "  \"regimes\": {{\n",
            "    \"source_holdout\": {},\n",
            "    \"action_effect_reversal\": {},\n",
            "    \"interaction_sign_reversal\": {}\n",
            "  }},\n",
            "  \"per_seed\": [\n"
        ),
        DIM, HIDDEN, TRAIN_SAMPLES, EPOCHS, results.len(), EVAL_STATES,
        MODEL_FORWARD_CALLS_PER_DECISION,
        render_aggregate(&source), render_aggregate(&reversal), render_aggregate(&interaction_shift)
    );
    for (index, row) in results.iter().enumerate() {
        out.push_str(&format!(
            "    {{\"seed\":\"{:016x}\",\"direct_mode\":\"{}\",\"training_mse\":{:.8},\"training_steps\":{},\"source_lift\":{:.8},\"source_regret\":{:.8},\"source_pairwise_accuracy\":{:.8},\"reversal_lift\":{:.8},\"reversal_regret\":{:.8},\"reversal_pairwise_accuracy\":{:.8},\"interaction_lift\":{:.8},\"interaction_regret\":{:.8},\"interaction_pairwise_accuracy\":{:.8}}}",
            row.seed, row.direct_mode, row.training_mse, row.training_steps,
            row.source.model_lift_vs_direct, row.source.model_regret, row.source.pairwise_ranking_accuracy,
            row.reversal.model_lift_vs_direct, row.reversal.model_regret, row.reversal.pairwise_ranking_accuracy,
            row.interaction_shift.model_lift_vs_direct, row.interaction_shift.model_regret, row.interaction_shift.pairwise_ranking_accuracy
        ));
        if index + 1 != results.len() { out.push(','); }
        out.push('\n');
    }
    out.push_str("  ]\n}\n");
    out
}

fn render_aggregate(row: &Aggregate) -> String {
    format!(
        concat!(
            "{{\"model_reward\":{:.8},\"direct_reward\":{:.8},\"oracle_reward\":{:.8},",
            "\"model_lift_vs_direct\":{:.8},\"lift_ci95\":[{:.8},{:.8}],",
            "\"model_regret\":{:.8},\"direct_regret\":{:.8},",
            "\"model_beats_direct_rate\":{:.8},\"model_harm_rate\":{:.8},",
            "\"oracle_action_agreement\":{:.8},\"pairwise_ranking_accuracy\":{:.8},",
            "\"reward_prediction_mae\":{:.8},\"next_state_prediction_mse\":{:.8},",
            "\"model_action_override_rate\":{:.8}}}"
        ),
        row.model_reward, row.direct_reward, row.oracle_reward,
        row.model_lift_vs_direct, row.lift_ci95_low, row.lift_ci95_high,
        row.model_regret, row.direct_regret,
        row.model_beats_direct_rate, row.model_harm_rate,
        row.oracle_action_agreement, row.pairwise_ranking_accuracy,
        row.reward_prediction_mae, row.next_state_prediction_mse,
        row.model_action_override_rate
    )
}

fn run() -> Result<Vec<SeedResult>, String> {
    let results = MODEL_SEEDS.iter().map(|seed| run_seed(*seed)).collect::<Result<Vec<_>, _>>()?;
    if results.iter().any(|row| {
        !row.training_mse.is_finite()
            || row.training_steps != (TRAIN_SAMPLES * EPOCHS) as u64
            || !row.source.model_lift_vs_direct.is_finite()
            || !row.reversal.model_lift_vs_direct.is_finite()
            || !row.interaction_shift.model_lift_vs_direct.is_finite()
    }) {
        return Err("non-finite metric or unexpected training-step count".to_owned());
    }
    Ok(results)
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "CMAQ-008 real CfC action-ranking diagnostic: {} model seeds; {} train samples x {} epochs; {} evaluation states x three regimes.",
            MODEL_SEEDS.len(), TRAIN_SAMPLES, EPOCHS, EVAL_STATES
        );
        return;
    }
    if !args.is_empty() {
        eprintln!("unexpected argument {}; use --help", args[0]);
        process::exit(2);
    }
    match run() {
        Ok(results) => print!("{}", render(&results)),
        Err(error) => {
            eprintln!("{error}");
            process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_modes_are_distinct_and_dimension_preserving() {
        let state = vec![0.5, -0.25, 0.0];
        let zero = ActionMode::Zero.action(&state);
        let cancel = ActionMode::CancelSource.action(&state);
        let reverse = ActionMode::ReverseSource.action(&state);
        assert_eq!(zero, vec![0.0, 0.0, 0.0]);
        assert_eq!(cancel, vec![-0.75, 0.75, 0.0]);
        assert_eq!(reverse, vec![0.75, -0.75, 0.0]);
    }

    #[test]
    fn source_and_reversal_change_action_effect_sign() {
        let state = vec![0.0; DIM];
        let action = vec![1.0; DIM];
        let source = transition(&state, &action, Law::Source);
        let reversal = transition(&state, &action, Law::ActionEffectReversal);
        assert!(source.iter().all(|value| (*value - 0.25).abs() < 1e-6));
        assert!(reversal.iter().all(|value| (*value + 0.25).abs() < 1e-6));
    }

    #[test]
    fn interaction_shift_changes_state_action_coupling() {
        let mut state = vec![0.0; DIM];
        let mut action = vec![0.0; DIM];
        state[0] = 0.2;
        state[1] = 0.8;
        action[0] = 1.0;
        let source = transition(&state, &action, Law::Source);
        let shifted = transition(&state, &action, Law::InteractionSignReversal);
        assert!((source[0] - (0.55 * 0.2 + 0.25 + 0.06 * 0.8)).abs() < 1e-6);
        assert!((shifted[0] - (0.55 * 0.2 + 0.25 - 0.12 * 0.8)).abs() < 1e-6);
        assert!((source[0] - shifted[0]).abs() > 0.1);
    }

    #[test]
    fn direct_policy_is_selected_only_from_observed_training_outcomes() {
        let rows = training_data(DATA_SEED);
        let chosen = fit_direct_policy(&rows);
        assert!(ActionMode::ALL.contains(&chosen));
        assert!(rows.iter().all(|row| row.action.len() == DIM && row.next_state.len() == DIM));
    }

    #[test]
    fn identical_model_seed_replays_the_complete_action_ranking_result() {
        let first = run_seed(MODEL_SEEDS[0]).unwrap();
        let second = run_seed(MODEL_SEEDS[0]).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn full_report_has_finite_decision_metrics_and_valid_uncertainty() {
        let results = run().unwrap();
        assert_eq!(results.len(), MODEL_SEEDS.len());
        for row in &results {
            for metrics in [row.source, row.reversal, row.interaction_shift] {
                for value in [
                    metrics.model_reward, metrics.direct_reward, metrics.oracle_reward,
                    metrics.model_lift_vs_direct, metrics.model_regret, metrics.direct_regret,
                    metrics.model_beats_direct_rate, metrics.model_harm_rate,
                    metrics.oracle_action_agreement, metrics.pairwise_ranking_accuracy,
                    metrics.reward_prediction_mae, metrics.next_state_prediction_mse,
                    metrics.model_action_override_rate,
                ] {
                    assert!(value.is_finite());
                }
                for value in [
                    metrics.model_beats_direct_rate, metrics.model_harm_rate,
                    metrics.oracle_action_agreement, metrics.pairwise_ranking_accuracy,
                    metrics.model_action_override_rate,
                ] {
                    assert!((0.0..=1.0).contains(&value));
                }
                assert!(metrics.model_regret >= -1e-10);
                assert!(metrics.direct_regret >= -1e-10);
            }
        }

        let parsed: serde_json::Value = serde_json::from_str(&render(&results)).unwrap();
        assert_eq!(parsed["schema_version"], "cmaq-008-action-ranking-v0.1");
        assert_eq!(parsed["model"], "HierarchicalCfCWorldModel");
        assert_eq!(parsed["regimes"].as_object().unwrap().len(), 3);
        assert_eq!(parsed["per_seed"].as_array().unwrap().len(), MODEL_SEEDS.len());
    }
}
