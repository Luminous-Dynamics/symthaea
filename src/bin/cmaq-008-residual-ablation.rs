//! CMAQ-008 residual-path ablation using the real CfC world-model implementation.
//!
//! Compares the unchanged absolute-next-state objective against an identity-preserving
//! residual objective: train the CfC output on delta-state and add the current state
//! through an explicit persistence skip. This is a synthetic diagnostic, not a SOTA claim.

use std::process;
use symthaea::dynamics::crate_world_model::{HierarchicalCfCWorldModel, WorldModelConfig};

const REPORT_SCHEMA: &str = "cmaq-008-residual-ablation-v0.2";
const DIM: usize = 16;
const HIDDEN: usize = 24;
const TRAIN_SAMPLES: usize = 128;
const EVAL_SAMPLES: usize = 64;
const EPOCHS: usize = 12;
const LEARNING_RATE: f32 = 0.001;
const DATA_SEED: u64 = 0xC0A8_2026_1029;
const MODEL_SEEDS: [u64; 5] = [
    0x5EED_0001,
    0x5EED_0002,
    0x5EED_0003,
    0x5EED_0004,
    0x5EED_0005,
];

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
}

impl Law {
    const ALL: [Self; 2] = [Self::Source, Self::ActionEffectReversal];

    fn name(self) -> &'static str {
        match self {
            Self::Source => "source_holdout",
            Self::ActionEffectReversal => "action_effect_reversal",
        }
    }

    fn action_coefficient(self) -> f32 {
        match self {
            Self::Source => 0.25,
            Self::ActionEffectReversal => -0.25,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Experience {
    state: Vec<f32>,
    action: Vec<f32>,
    next_state: Vec<f32>,
}

fn transition(state: &[f32], action: &[f32], law: Law) -> Vec<f32> {
    (0..DIM)
        .map(|i| {
            let neighbour = state[(i + 1) % DIM];
            0.55 * state[i]
                + law.action_coefficient() * action[i]
                + 0.06 * neighbour * action[i]
        })
        .collect()
}

fn dataset(seed: u64, count: usize, law: Law) -> Vec<Experience> {
    let mut rng = SplitMix64::new(seed);
    (0..count)
        .map(|_| {
            let state = (0..DIM).map(|_| rng.signed()).collect::<Vec<_>>();
            let action = (0..DIM).map(|_| rng.signed()).collect::<Vec<_>>();
            let next_state = transition(&state, &action, law);
            Experience {
                state,
                action,
                next_state,
            }
        })
        .collect()
}

const LINEAR_FEATURES: usize = DIM * 2 + 1;
const LINEAR_RIDGE: f64 = 1.0e-3;

/// A low-compute ARX baseline fitted by ridge least squares on the same support tape.
/// Features are current state, action and intercept; predictions are state + delta.
#[derive(Clone, Debug, PartialEq)]
struct LinearArxBaseline {
    weights: Vec<Vec<f64>>, // output dimension × feature dimension
}

impl LinearArxBaseline {
    fn features(state: &[f32], action: &[f32]) -> [f64; LINEAR_FEATURES] {
        let mut features = [0.0_f64; LINEAR_FEATURES];
        for index in 0..DIM {
            features[index] = state[index] as f64;
            features[DIM + index] = action[index] as f64;
        }
        features[LINEAR_FEATURES - 1] = 1.0;
        features
    }

    fn fit(data: &[Experience], ridge: f64) -> Result<Self, String> {
        if data.is_empty() {
            return Err("linear ARX training data must not be empty".to_owned());
        }
        if !ridge.is_finite() || ridge < 0.0 {
            return Err("linear ARX ridge penalty must be finite and nonnegative".to_owned());
        }

        // Normal equations for all outputs at once:
        // (X'X + λI) B = X' ΔY. Keep the intercept unpenalized.
        let mut gram = vec![vec![0.0_f64; LINEAR_FEATURES]; LINEAR_FEATURES];
        let mut rhs = vec![vec![0.0_f64; DIM]; LINEAR_FEATURES];
        for row in data {
            if row.state.len() != DIM || row.action.len() != DIM || row.next_state.len() != DIM {
                return Err("linear ARX row dimension mismatch".to_owned());
            }
            let features = Self::features(&row.state, &row.action);
            for left in 0..LINEAR_FEATURES {
                for right in 0..LINEAR_FEATURES {
                    gram[left][right] += features[left] * features[right];
                }
                for output in 0..DIM {
                    let delta = row.next_state[output] as f64 - row.state[output] as f64;
                    rhs[left][output] += features[left] * delta;
                }
            }
        }
        for feature in 0..LINEAR_FEATURES - 1 {
            gram[feature][feature] += ridge;
        }

        let mut augmented = vec![vec![0.0_f64; LINEAR_FEATURES + DIM]; LINEAR_FEATURES];
        for row in 0..LINEAR_FEATURES {
            for column in 0..LINEAR_FEATURES {
                augmented[row][column] = gram[row][column];
            }
            for output in 0..DIM {
                augmented[row][LINEAR_FEATURES + output] = rhs[row][output];
            }
        }

        // Gauss-Jordan elimination with partial pivoting. No matrix inversion
        // dependency is needed for this small, deterministic baseline.
        for pivot_column in 0..LINEAR_FEATURES {
            let pivot_row = (pivot_column..LINEAR_FEATURES)
                .max_by(|left, right| {
                    augmented[*left][pivot_column]
                        .abs()
                        .total_cmp(&augmented[*right][pivot_column].abs())
                })
                .ok_or_else(|| "linear ARX pivot selection failed".to_owned())?;
            let pivot_value = augmented[pivot_row][pivot_column];
            if !pivot_value.is_finite() || pivot_value.abs() < 1.0e-12 {
                return Err(format!("linear ARX singular normal matrix at column {pivot_column}"));
            }
            augmented.swap(pivot_column, pivot_row);

            let scale = augmented[pivot_column][pivot_column];
            for column in pivot_column..(LINEAR_FEATURES + DIM) {
                augmented[pivot_column][column] /= scale;
            }

            for row in 0..LINEAR_FEATURES {
                if row == pivot_column {
                    continue;
                }
                let factor = augmented[row][pivot_column];
                if factor == 0.0 {
                    continue;
                }
                for column in pivot_column..(LINEAR_FEATURES + DIM) {
                    augmented[row][column] -= factor * augmented[pivot_column][column];
                }
            }
        }

        let mut weights = vec![vec![0.0_f64; LINEAR_FEATURES]; DIM];
        for feature in 0..LINEAR_FEATURES {
            for output in 0..DIM {
                weights[output][feature] = augmented[feature][LINEAR_FEATURES + output];
            }
        }
        if weights.iter().flatten().any(|weight| !weight.is_finite()) {
            return Err("linear ARX fit produced non-finite weights".to_owned());
        }
        Ok(Self { weights })
    }

    fn predict(&self, state: &[f32], action: &[f32]) -> Result<Vec<f32>, String> {
        if state.len() != DIM || action.len() != DIM {
            return Err(format!(
                "linear ARX expects state/action dimensions {DIM}, got {}/{}",
                state.len(), action.len()
            ));
        }
        let features = Self::features(state, action);
        Ok((0..DIM)
            .map(|output| {
                let delta = self.weights[output]
                    .iter()
                    .zip(features.iter())
                    .map(|(weight, feature)| weight * feature)
                    .sum::<f64>();
                state[output] + delta as f32
            })
            .collect())
    }

    fn evaluate(&self, data: &[Experience]) -> Result<Metrics, String> {
        let mut squared_error = 0.0_f64;
        let mut absolute_error = 0.0_f64;
        let mut persistence_squared_error = 0.0_f64;
        let mut zero_squared_error = 0.0_f64;
        let mut count = 0_usize;
        for row in data {
            let prediction = self.predict(&row.state, &row.action)?;
            for index in 0..DIM {
                let error = prediction[index] as f64 - row.next_state[index] as f64;
                let persistence_error = row.state[index] as f64 - row.next_state[index] as f64;
                let zero_error = row.next_state[index] as f64;
                squared_error += error * error;
                absolute_error += error.abs();
                persistence_squared_error += persistence_error * persistence_error;
                zero_squared_error += zero_error * zero_error;
                count += 1;
            }
        }
        if count == 0 {
            return Err("linear ARX evaluation data must not be empty".to_owned());
        }
        let denominator = count as f64;
        Ok(Metrics {
            mse: squared_error / denominator,
            mae: absolute_error / denominator,
            persistence_mse: persistence_squared_error / denominator,
            zero_mse: zero_squared_error / denominator,
            sample_count: data.len(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Objective {
    AbsoluteNextState,
    ResidualDelta,
}

impl Objective {
    const ALL: [Self; 2] = [Self::AbsoluteNextState, Self::ResidualDelta];

    fn name(self) -> &'static str {
        match self {
            Self::AbsoluteNextState => "absolute_next_state",
            Self::ResidualDelta => "identity_residual_delta",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Metrics {
    mse: f64,
    mae: f64,
    persistence_mse: f64,
    zero_mse: f64,
    sample_count: usize,
}

impl Metrics {
    fn evaluate(
        model: &HierarchicalCfCWorldModel,
        data: &[Experience],
        objective: Objective,
    ) -> Result<Self, String> {
        let mut squared_error = 0.0_f64;
        let mut absolute_error = 0.0_f64;
        let mut persistence_squared_error = 0.0_f64;
        let mut zero_squared_error = 0.0_f64;
        let mut count = 0_usize;

        for row in data {
            let prediction = match objective {
                Objective::AbsoluteNextState => model.predict(&row.state, &row.action, 1.0),
                Objective::ResidualDelta => {
                    model.predict_next_state_residual(&row.state, &row.action, 1.0)
                }
            }
            .map_err(|error| format!("world-model prediction failed: {error}"))?;

            if prediction.len() != row.next_state.len() {
                return Err(format!(
                    "prediction dimension mismatch: got {}, expected {}",
                    prediction.len(),
                    row.next_state.len()
                ));
            }
            for i in 0..DIM {
                let error = prediction[i] as f64 - row.next_state[i] as f64;
                let persistence_error = row.state[i] as f64 - row.next_state[i] as f64;
                let zero_error = row.next_state[i] as f64;
                squared_error += error * error;
                absolute_error += error.abs();
                persistence_squared_error += persistence_error * persistence_error;
                zero_squared_error += zero_error * zero_error;
                count += 1;
            }
        }

        if count == 0 {
            return Err("evaluation dataset must not be empty".to_owned());
        }
        let denominator = count as f64;
        Ok(Self {
            mse: squared_error / denominator,
            mae: absolute_error / denominator,
            persistence_mse: persistence_squared_error / denominator,
            zero_mse: zero_squared_error / denominator,
            sample_count: data.len(),
        })
    }
}

const ACTION_MENU_SIZE: usize = 9;
const TASK_GOAL_SEED: u64 = 0xC0A8_7A5C_2026_1030;

#[derive(Clone, Debug, PartialEq)]
struct DecisionTask {
    state: Vec<f32>,
    goal: Vec<f32>,
    actions: Vec<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct DecisionMetrics {
    model_utility: f64,
    arx_utility: f64,
    direct_utility: f64,
    oracle_utility: f64,
    model_regret: f64,
    arx_regret: f64,
    model_oracle_action_agreement: f64,
    arx_oracle_action_agreement: f64,
}

fn candidate_action_menu() -> Vec<Vec<f32>> {
    let mut rng = SplitMix64::new(0xA671_0A11_2026_1030);
    let mut actions = Vec::with_capacity(ACTION_MENU_SIZE);
    actions.push(vec![0.0; DIM]);
    for _ in 0..4 {
        let action = (0..DIM).map(|_| 0.6 * rng.signed()).collect::<Vec<_>>();
        actions.push(action.clone());
        actions.push(action.iter().map(|value| -*value).collect());
    }
    actions
}

fn make_decision_tasks(data: &[Experience], goal_seed: u64) -> Vec<DecisionTask> {
    let actions = candidate_action_menu();
    let mut rng = SplitMix64::new(goal_seed);
    data.iter()
        .map(|row| DecisionTask {
            state: row.state.clone(),
            goal: (0..DIM).map(|_| 0.45 * rng.signed()).collect(),
            actions: actions.clone(),
        })
        .collect()
}

fn vector_mse(left: &[f32], right: &[f32]) -> f64 {
    if left.is_empty() || left.len() != right.len() {
        return f64::INFINITY;
    }
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            let delta = *a as f64 - *b as f64;
            delta * delta
        })
        .sum::<f64>()
        / left.len() as f64
}

fn argmin_by_score<F>(actions: &[Vec<f32>], mut score: F) -> usize
where
    F: FnMut(&[f32]) -> f64,
{
    actions
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| score(left).total_cmp(&score(right)))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

/// Candidate sees state, public goal and candidate action vectors. The transition law
/// is not an argument to the candidate's scoring path.
fn select_model_action(
    model: &HierarchicalCfCWorldModel,
    objective: Objective,
    task: &DecisionTask,
) -> Result<usize, String> {
    let predictions = task.actions.iter()
        .map(|action| match objective {
            Objective::AbsoluteNextState => model.predict(&task.state, action, 1.0),
            Objective::ResidualDelta => model.predict_next_state_residual(&task.state, action, 1.0),
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("CfC candidate-action prediction failed: {error}"))?;
    Ok(argmin_by_score(&task.actions, |action| {
        let index = task.actions.iter().position(|candidate| candidate == action).unwrap_or(0);
        vector_mse(&predictions[index], &task.goal)
    }))
}

fn select_arx_action(
    model: &LinearArxBaseline,
    task: &DecisionTask,
) -> Result<usize, String> {
    let predictions = task.actions.iter()
        .map(|action| model.predict(&task.state, action))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(argmin_by_score(&task.actions, |action| {
        let index = task.actions.iter().position(|candidate| candidate == action).unwrap_or(0);
        vector_mse(&predictions[index], &task.goal)
    }))
}

fn task_utility(task: &DecisionTask, action_index: usize, law: Law) -> f64 {
    let predicted_truth = transition(&task.state, &task.actions[action_index], law);
    -vector_mse(&predicted_truth, &task.goal)
}

fn evaluate_decision_tasks(
    model: &HierarchicalCfCWorldModel,
    linear: &LinearArxBaseline,
    objective: Objective,
    data: &[Experience],
    law: Law,
    goal_seed: u64,
) -> Result<DecisionMetrics, String> {
    let tasks = make_decision_tasks(data, goal_seed);
    if tasks.is_empty() {
        return Err("decision tasks must not be empty".to_owned());
    }
    let mut model_utility = 0.0;
    let mut arx_utility = 0.0;
    let mut direct_utility = 0.0;
    let mut oracle_utility = 0.0;
    let mut model_agreement = 0_usize;
    let mut arx_agreement = 0_usize;

    for task in &tasks {
        let model_choice = select_model_action(model, objective, task)?;
        let arx_choice = select_arx_action(linear, task)?;
        let direct_choice = 0; // Public no-op/direct baseline.
        let oracle_choice = argmin_by_score(&task.actions, |action| {
            let next_state = transition(&task.state, action, law);
            vector_mse(&next_state, &task.goal)
        });

        model_utility += task_utility(task, model_choice, law);
        arx_utility += task_utility(task, arx_choice, law);
        direct_utility += task_utility(task, direct_choice, law);
        oracle_utility += task_utility(task, oracle_choice, law);
        model_agreement += (model_choice == oracle_choice) as usize;
        arx_agreement += (arx_choice == oracle_choice) as usize;
    }

    let denominator = tasks.len() as f64;
    let model_utility = model_utility / denominator;
    let arx_utility = arx_utility / denominator;
    let direct_utility = direct_utility / denominator;
    let oracle_utility = oracle_utility / denominator;
    Ok(DecisionMetrics {
        model_utility,
        arx_utility,
        direct_utility,
        oracle_utility,
        model_regret: (oracle_utility - model_utility).max(0.0),
        arx_regret: (oracle_utility - arx_utility).max(0.0),
        model_oracle_action_agreement: model_agreement as f64 / denominator,
        arx_oracle_action_agreement: arx_agreement as f64 / denominator,
    })
}

#[derive(Clone, Debug, PartialEq)]
struct SeedResult {
    seed: u64,
    final_epoch_train_mse: f64,
    training_steps: u64,
    regimes: [Metrics; 2],
}

fn run_seed(seed: u64, objective: Objective) -> Result<SeedResult, String> {
    let training = dataset(DATA_SEED, TRAIN_SAMPLES, Law::Source);
    let evaluations = [
        dataset(DATA_SEED ^ 0x1111_2222_3333_4444, EVAL_SAMPLES, Law::Source),
        dataset(
            DATA_SEED ^ 0xAAAA_BBBB_CCCC_DDDD,
            EVAL_SAMPLES,
            Law::ActionEffectReversal,
        ),
    ];

    let config = WorldModelConfig {
        hdc_dim: DIM,
        hidden_dim: HIDDEN,
        learning_rate: LEARNING_RATE,
        momentum: 0.9,
        consciousness_decay: 0.99,
    };
    let mut model = HierarchicalCfCWorldModel::new_with_seed(config, seed)
        .map_err(|error| format!("seeded model construction failed: {error}"))?;

    let mut final_epoch_loss = 0.0_f64;
    for _ in 0..EPOCHS {
        final_epoch_loss = 0.0;
        for row in &training {
            let loss = match objective {
                Objective::AbsoluteNextState => {
                    model.train_step(&row.state, &row.action, &row.next_state, 1.0)
                }
                Objective::ResidualDelta => {
                    model.train_step_residual(&row.state, &row.action, &row.next_state, 1.0)
                }
            }
            .map_err(|error| format!("world-model training failed: {error}"))?;
            final_epoch_loss += loss as f64;
        }
        final_epoch_loss /= TRAIN_SAMPLES as f64;
    }

    let mut regimes = [Metrics::default(); 2];
    for (index, law) in Law::ALL.into_iter().enumerate() {
        regimes[index] = Metrics::evaluate(&model, &evaluations[index], objective)
            .map_err(|error| format!("{} evaluation failed: {error}", law.name()))?;
    }
    Ok(SeedResult {
        seed,
        final_epoch_train_mse: final_epoch_loss,
        training_steps: model.training_steps,
        regimes,
    })
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn mean_ci95(values: &[f64]) -> (f64, f64, f64) {
    let average = mean(values);
    if values.len() < 2 {
        return (average, average, average);
    }
    let variance = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let half_width = 1.96 * (variance / values.len() as f64).sqrt();
    (average, average - half_width, average + half_width)
}

#[derive(Clone, Debug, PartialEq)]
struct Aggregate {
    mse: f64,
    mae: f64,
    persistence_mse: f64,
    zero_mse: f64,
    mse_to_persistence_ratio: f64,
    mse_delta_vs_persistence: f64,
    mse_delta_vs_persistence_ci95_low: f64,
    mse_delta_vs_persistence_ci95_high: f64,
}

fn aggregate(results: &[SeedResult], law_index: usize) -> Aggregate {
    let rows = results.iter().map(|row| row.regimes[law_index]).collect::<Vec<_>>();
    let mses = rows.iter().map(|row| row.mse).collect::<Vec<_>>();
    let persistence = rows.iter().map(|row| row.persistence_mse).collect::<Vec<_>>();
    let (delta, low, high) = mean_ci95(
        &rows
            .iter()
            .map(|row| row.mse - row.persistence_mse)
            .collect::<Vec<_>>(),
    );
    let mse = mean(&mses);
    let persistence_mse = mean(&persistence);
    Aggregate {
        mse,
        mae: mean(&rows.iter().map(|row| row.mae).collect::<Vec<_>>()),
        persistence_mse,
        zero_mse: mean(&rows.iter().map(|row| row.zero_mse).collect::<Vec<_>>()),
        mse_to_persistence_ratio: mse / persistence_mse.max(f64::MIN_POSITIVE),
        mse_delta_vs_persistence: delta,
        mse_delta_vs_persistence_ci95_low: low,
        mse_delta_vs_persistence_ci95_high: high,
    }
}

#[derive(Clone, Debug)]
struct ModeResults {
    objective: Objective,
    seeds: Vec<SeedResult>,
}

struct RunReport {
    linear_baseline: [Metrics; 2],
    modes: Vec<ModeResults>,
}

fn run() -> Result<RunReport, String> {
    let training = dataset(DATA_SEED, TRAIN_SAMPLES, Law::Source);
    let linear = LinearArxBaseline::fit(&training, LINEAR_RIDGE)?;
    let source_eval = dataset(DATA_SEED ^ 0x1111_2222_3333_4444, EVAL_SAMPLES, Law::Source);
    let reversal_eval = dataset(
        DATA_SEED ^ 0xAAAA_BBBB_CCCC_DDDD,
        EVAL_SAMPLES,
        Law::ActionEffectReversal,
    );
    let linear_baseline = [
        linear.evaluate(&source_eval)?,
        linear.evaluate(&reversal_eval)?,
    ];

    let mut modes = Vec::with_capacity(Objective::ALL.len());
    for objective in Objective::ALL {
        let results = MODEL_SEEDS
            .iter()
            .map(|seed| run_seed(*seed, objective))
            .collect::<Result<Vec<_>, _>>()?;
        if results.iter().any(|row| {
            !row.final_epoch_train_mse.is_finite()
                || row.training_steps != (TRAIN_SAMPLES * EPOCHS) as u64
                || row.regimes.iter().any(|metric| {
                    !metric.mse.is_finite()
                        || !metric.mae.is_finite()
                        || !metric.persistence_mse.is_finite()
                        || metric.mse < 0.0
                })
        }) {
            return Err(format!(
                "non-finite metrics or unexpected steps for {}",
                objective.name()
            ));
        }
        modes.push(ModeResults {
            objective,
            seeds: results,
        });
    }
    Ok(RunReport { linear_baseline, modes })
}

fn render(report: &RunReport) -> String {
    let mut out = format!(
        concat!(
            "{{\n",
            "  \"schema_version\": \"{}\",\n",
            "  \"model\": \"HierarchicalCfCWorldModel\",\n",
            "  \"dimensions\": {{\"hdc\": {}, \"hidden\": {}}},\n",
            "  \"training\": {{\"samples\": {}, \"epochs\": {}, \"seeds\": {}, \"learning_rate\": {:.6}}},\n",
            "  \"evaluation_samples_per_regime_per_seed\": {},\n",
            "  \"regimes\": [\"source_holdout\", \"action_effect_reversal\"],\n"
        ),
        REPORT_SCHEMA, DIM, HIDDEN, TRAIN_SAMPLES, EPOCHS, MODEL_SEEDS.len(), LEARNING_RATE, EVAL_SAMPLES
    );
    out.push_str(&format!(
        concat!(
            "  \"linear_arx_baseline\": {{",
            "\"features\": {}, \"ridge\": {:.8},",
            "\"source\": {{\"mse\":{:.8},\"mae\":{:.8},\"persistence_mse\":{:.8},\"zero_mse\":{:.8}}},",
            "\"action_effect_reversal\": {{\"mse\":{:.8},\"mae\":{:.8},\"persistence_mse\":{:.8},\"zero_mse\":{:.8}}}",
            "}},\n",
            "  \"objectives\": [\n"
        ),
        LINEAR_FEATURES, LINEAR_RIDGE,
        report.linear_baseline[0].mse, report.linear_baseline[0].mae,
        report.linear_baseline[0].persistence_mse, report.linear_baseline[0].zero_mse,
        report.linear_baseline[1].mse, report.linear_baseline[1].mae,
        report.linear_baseline[1].persistence_mse, report.linear_baseline[1].zero_mse
    ));

    for (mode_index, mode) in report.modes.iter().enumerate() {
        let source = aggregate(&mode.seeds, 0);
        let reversal = aggregate(&mode.seeds, 1);
        let paired_delta = |regime_index: usize| {
            if mode.objective == Objective::AbsoluteNextState {
                (0.0, 0.0, 0.0)
            } else {
                let deltas = mode.seeds.iter().zip(&report.modes[0].seeds)
                    .map(|(candidate, absolute)| {
                        candidate.regimes[regime_index].mse - absolute.regimes[regime_index].mse
                    })
                    .collect::<Vec<_>>();
                mean_ci95(&deltas)
            }
        };
        let paired_linear_delta = |regime_index: usize| {
            let baseline = report.linear_baseline[regime_index].mse;
            let deltas = mode.seeds.iter()
                .map(|candidate| candidate.regimes[regime_index].mse - baseline)
                .collect::<Vec<_>>();
            mean_ci95(&deltas)
        };
        let source_pair = paired_delta(0);
        let reversal_pair = paired_delta(1);
        let source_linear_pair = paired_linear_delta(0);
        let reversal_linear_pair = paired_linear_delta(1);
        out.push_str(&format!(
            concat!(
                "    {{\"objective\":\"{}\",",
                "\"training_steps_per_seed\":{},",
                "\"source\":{{\"mse\":{:.8},\"mae\":{:.8},\"persistence_mse\":{:.8},\"zero_mse\":{:.8},",
                "\"mse_to_persistence_ratio\":{:.8},\"mse_delta_vs_persistence\":{:.8},",
                "\"mse_delta_vs_persistence_ci95_low\":{:.8},\"mse_delta_vs_persistence_ci95_high\":{:.8},",
                "\"paired_mse_delta_vs_absolute\":{:.8},\"paired_mse_delta_ci95_low\":{:.8},\"paired_mse_delta_ci95_high\":{:.8},",
                "\"paired_mse_delta_vs_linear_arx\":{:.8},\"paired_linear_ci95_low\":{:.8},\"paired_linear_ci95_high\":{:.8}}},",
                "\"action_effect_reversal\":{{\"mse\":{:.8},\"mae\":{:.8},\"persistence_mse\":{:.8},\"zero_mse\":{:.8},",
                "\"mse_to_persistence_ratio\":{:.8},\"mse_delta_vs_persistence\":{:.8},",
                "\"mse_delta_vs_persistence_ci95_low\":{:.8},\"mse_delta_vs_persistence_ci95_high\":{:.8},",
                "\"paired_mse_delta_vs_absolute\":{:.8},\"paired_mse_delta_ci95_low\":{:.8},\"paired_mse_delta_ci95_high\":{:.8},",
                "\"paired_mse_delta_vs_linear_arx\":{:.8},\"paired_linear_ci95_low\":{:.8},\"paired_linear_ci95_high\":{:.8}}},",
                "\"per_seed\":["
            ),
            mode.objective.name(),
            (TRAIN_SAMPLES * EPOCHS),
            source.mse, source.mae, source.persistence_mse, source.zero_mse,
            source.mse_to_persistence_ratio, source.mse_delta_vs_persistence,
            source.mse_delta_vs_persistence_ci95_low, source.mse_delta_vs_persistence_ci95_high,
            source_pair.0, source_pair.1, source_pair.2,
            source_linear_pair.0, source_linear_pair.1, source_linear_pair.2,
            reversal.mse, reversal.mae, reversal.persistence_mse, reversal.zero_mse,
            reversal.mse_to_persistence_ratio, reversal.mse_delta_vs_persistence,
            reversal.mse_delta_vs_persistence_ci95_low, reversal.mse_delta_vs_persistence_ci95_high,
            reversal_pair.0, reversal_pair.1, reversal_pair.2,
            reversal_linear_pair.0, reversal_linear_pair.1, reversal_linear_pair.2
        ));
        for (seed_index, row) in mode.seeds.iter().enumerate() {
            out.push_str(&format!(
                "{{\"seed\":\"{:016x}\",\"final_epoch_train_mse\":{:.8},\"source_mse\":{:.8},\"reversal_mse\":{:.8}}}",
                row.seed, row.final_epoch_train_mse, row.regimes[0].mse, row.regimes[1].mse
            ));
            if seed_index + 1 != mode.seeds.len() {
                out.push(',');
            }
        }
        out.push_str("]}");
        if mode_index + 1 != reports.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ]\n}\n");
    out
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("CMAQ-008 residual-path ablation: real CfC model; compare absolute next-state learning with identity-preserving residual delta learning.");
        return;
    }
    if !args.is_empty() {
        eprintln!("unexpected argument {}; use --help", args[0]);
        process::exit(2);
    }
    match run() {
        Ok(report) => print!("{}", render(&report)),
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
    fn data_tape_is_paired_across_learning_objectives() {
        for law in Law::ALL {
            assert_eq!(
                dataset(DATA_SEED, 12, law),
                dataset(DATA_SEED, 12, law)
            );
        }
    }

    #[test]
    fn ridge_arx_learns_source_dynamics_better_than_persistence() {
        let training = dataset(DATA_SEED, TRAIN_SAMPLES, Law::Source);
        let model = LinearArxBaseline::fit(&training, LINEAR_RIDGE).unwrap();
        let evaluation = dataset(DATA_SEED ^ 0x1111_2222_3333_4444, EVAL_SAMPLES, Law::Source);
        let metrics = model.evaluate(&evaluation).unwrap();
        assert!(metrics.mse.is_finite() && metrics.persistence_mse.is_finite());
        assert!(
            metrics.mse < metrics.persistence_mse,
            "ridge ARX MSE {} should beat persistence {} on source holdout",
            metrics.mse,
            metrics.persistence_mse
        );
    }

    #[test]
    fn both_objectives_replay_same_seed() {
        for objective in Objective::ALL {
            let first = run_seed(MODEL_SEEDS[0], objective).unwrap();
            let second = run_seed(MODEL_SEEDS[0], objective).unwrap();
            assert_eq!(first, second);
        }
    }

    #[test]
    fn report_contains_paired_objectives_and_finite_metrics() {
        let report = run().unwrap();
        assert_eq!(report.modes.len(), 2);
        let output = render(&report);
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(
            parsed["schema_version"],
            "cmaq-008-residual-ablation-v0.2"
        );
        assert_eq!(parsed["objectives"].as_array().unwrap().len(), 2);
        let linear_source_mse = parsed["linear_arx_baseline"]["source"]["mse"]
            .as_f64().expect("linear ARX source MSE must be numeric");
        let linear_source_persistence = parsed["linear_arx_baseline"]["source"]["persistence_mse"]
            .as_f64().expect("persistence source MSE must be numeric");
        assert!(linear_source_mse.is_finite());
        assert!(linear_source_persistence.is_finite());
        assert!(linear_source_mse < linear_source_persistence);

        for objective in parsed["objectives"].as_array().unwrap() {
            for regime in ["source", "action_effect_reversal"] {
                for metric in [
                    "mse",
                    "mae",
                    "persistence_mse",
                    "zero_mse",
                    "mse_to_persistence_ratio",
                    "mse_delta_vs_persistence",
                    "mse_delta_vs_persistence_ci95_low",
                    "mse_delta_vs_persistence_ci95_high",
                    "paired_mse_delta_vs_absolute",
                    "paired_mse_delta_ci95_low",
                    "paired_mse_delta_ci95_high",
                    "paired_mse_delta_vs_linear_arx",
                    "paired_linear_ci95_low",
                    "paired_linear_ci95_high",
                ] {
                    let value = objective[regime][metric]
                        .as_f64()
                        .unwrap_or_else(|| panic!("{} {} must be numeric", regime, metric));
                    assert!(value.is_finite());
                }
            }
        }
    }
}
