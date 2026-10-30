use std::fmt::Write as _;

const REPORT_SCHEMA: &str = "cmaq-012-world-model-advantage-v0.1";
const EPISODES: usize = 512;
const HORIZON: usize = 64;
const SHIFT_STEP: usize = HORIZON / 2;
const PRIOR_HAZARD: f64 = 0.25;
const TRAINED_SENSITIVITY: f64 = 0.80;
const TRAINED_SPECIFICITY: f64 = 0.85;
const SAFE_ACTION_REWARD: f64 = 0.15;
const FAST_ACTION_SAFE_REWARD: f64 = 0.45;
const FAST_ACTION_HAZARD_REWARD: f64 = -0.60;
const REWARD_NOISE: f64 = 0.05;
const BASE_SEED: u64 = 0xC0A9_0012_2026_1030;

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
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    fn bernoulli(&mut self, probability: f64) -> bool {
        self.unit() < probability
    }

    fn centered(&mut self, radius: f64) -> f64 {
        (self.unit() * 2.0 - 1.0) * radius
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    PredictiveSignal,
    ActionInvariant,
    LowInformation,
    SensorShift,
}

impl Scenario {
    fn all() -> [Self; 4] {
        [
            Self::PredictiveSignal,
            Self::ActionInvariant,
            Self::LowInformation,
            Self::SensorShift,
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::PredictiveSignal => "predictive_signal",
            Self::ActionInvariant => "action_invariant",
            Self::LowInformation => "low_information",
            Self::SensorShift => "sensor_shift",
        }
    }

    fn sensor_quality(self, step: usize) -> (f64, f64) {
        match self {
            Self::PredictiveSignal | Self::ActionInvariant => {
                (TRAINED_SENSITIVITY, TRAINED_SPECIFICITY)
            }
            Self::LowInformation => (0.52, 0.52),
            Self::SensorShift if step >= SHIFT_STEP => (0.15, 0.20),
            Self::SensorShift => (TRAINED_SENSITIVITY, TRAINED_SPECIFICITY),
        }
    }

    fn action_invariant(self) -> bool {
        matches!(self, Self::ActionInvariant)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    DirectController,
    ActionBlindPlacebo,
    FrozenActionModel,
    CalibratedActionModel,
    ShuffledActionLabels,
    HiddenStateOracle,
}

impl Policy {
    fn all() -> [Self; 6] {
        [
            Self::DirectController,
            Self::ActionBlindPlacebo,
            Self::FrozenActionModel,
            Self::CalibratedActionModel,
            Self::ShuffledActionLabels,
            Self::HiddenStateOracle,
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::DirectController => "direct_controller",
            Self::ActionBlindPlacebo => "action_blind_placebo",
            Self::FrozenActionModel => "frozen_calibration_action_model",
            Self::CalibratedActionModel => "calibration_oracle_action_model",
            Self::ShuffledActionLabels => "shuffled_action_labels",
            Self::HiddenStateOracle => "hidden_state_oracle",
        }
    }

    fn model_queries_per_decision(self) -> usize {
        match self {
            Self::DirectController | Self::HiddenStateOracle => 0,
            _ => 2,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct StepTape {
    hazard: bool,
    warning: bool,
    reward_fast: f64,
    reward_safe: f64,
    sensitivity: f64,
    specificity: f64,
}

type EpisodeTape = [StepTape; HORIZON];

#[derive(Clone, Copy, Debug)]
struct EpisodeOutcome {
    total_return: f64,
    conditional_regret: f64,
    prediction_abs_error: f64,
    rank_correct: f64,
    action_disagreement: f64,
    model_queries: usize,
}

#[derive(Clone, Debug)]
struct Summary {
    scenario: &'static str,
    policy: &'static str,
    mean_return: f64,
    paired_delta_vs_direct: f64,
    delta_ci95_low: f64,
    delta_ci95_high: f64,
    cvar95_return: f64,
    success_rate: f64,
    mean_conditional_regret_per_step: f64,
    action_disagreement_rate: f64,
    mean_prediction_mae: f64,
    action_rank_accuracy: f64,
    mean_model_queries_per_episode: f64,
    model_prediction_evaluable: bool,
}

fn posterior_hazard(warning: bool, sensitivity: f64, specificity: f64) -> f64 {
    let (hazard_likelihood, safe_likelihood) = if warning {
        (sensitivity, 1.0 - specificity)
    } else {
        (1.0 - sensitivity, specificity)
    };
    let numerator = PRIOR_HAZARD * hazard_likelihood;
    let denominator = numerator + (1.0 - PRIOR_HAZARD) * safe_likelihood;
    if denominator <= f64::EPSILON {
        PRIOR_HAZARD
    } else {
        numerator / denominator
    }
}

fn conditional_fast_value(scenario: Scenario, p_hazard: f64) -> f64 {
    if scenario.action_invariant() {
        0.22
    } else {
        p_hazard * FAST_ACTION_HAZARD_REWARD
            + (1.0 - p_hazard) * FAST_ACTION_SAFE_REWARD
    }
}

fn conditional_safe_value(scenario: Scenario) -> f64 {
    if scenario.action_invariant() {
        0.20
    } else {
        SAFE_ACTION_REWARD
    }
}

fn make_episode_tape(scenario: Scenario, episode: usize) -> EpisodeTape {
    let seed = BASE_SEED
        ^ ((scenario as u64) << 40)
        ^ (episode as u64).wrapping_mul(0xD134_2543_DE82_EF95);
    let mut rng = SplitMix64::new(seed);
    std::array::from_fn(|step| {
        let hazard = rng.bernoulli(PRIOR_HAZARD);
        let (sensitivity, specificity) = scenario.sensor_quality(step);
        let warning_probability = if hazard {
            sensitivity
        } else {
            1.0 - specificity
        };
        let warning = rng.bernoulli(warning_probability);

        let (fast_hazard, fast_safe) = if scenario.action_invariant() {
            (0.22, 0.22)
        } else {
            (FAST_ACTION_HAZARD_REWARD, FAST_ACTION_SAFE_REWARD)
        };
        let reward_fast = (if hazard { fast_hazard } else { fast_safe })
            + rng.centered(REWARD_NOISE);
        let reward_safe = conditional_safe_value(scenario)
            + rng.centered(REWARD_NOISE);

        StepTape {
            hazard,
            warning,
            reward_fast,
            reward_safe,
            sensitivity,
            specificity,
        }
    })
}

fn choose_fast(predicted_fast: f64, predicted_safe: f64) -> bool {
    // Stable tie-break preserves the direct controller.
    predicted_fast >= predicted_safe
}

fn run_episode(
    scenario: Scenario,
    policy: Policy,
    tape: &EpisodeTape,
) -> EpisodeOutcome {
    let mut total_return = 0.0;
    let mut conditional_regret = 0.0;
    let mut prediction_abs_error = 0.0;
    let mut rank_correct = 0.0;
    let mut action_disagreement = 0.0;
    let mut model_queries = 0;

    for step in 0..HORIZON {
        let item = tape[step];
        let p_true = posterior_hazard(item.warning, item.sensitivity, item.specificity);
        let true_fast = conditional_fast_value(scenario, p_true);
        let true_safe = conditional_safe_value(scenario);

        let (action_fast, evaluated, predicted_fast, predicted_safe) = match policy {
            Policy::DirectController => (true, false, 0.0, 0.0),
            Policy::HiddenStateOracle => {
                let fast_value = if scenario.action_invariant() {
                    0.22
                } else if item.hazard {
                    FAST_ACTION_HAZARD_REWARD
                } else {
                    FAST_ACTION_SAFE_REWARD
                };
                (fast_value >= conditional_safe_value(scenario), false, 0.0, 0.0)
            }
            Policy::ActionBlindPlacebo
            | Policy::FrozenActionModel
            | Policy::CalibratedActionModel
            | Policy::ShuffledActionLabels => {
                model_queries += policy.model_queries_per_decision();
                let (assumed_sensitivity, assumed_specificity) = match policy {
                    Policy::CalibratedActionModel => (item.sensitivity, item.specificity),
                    _ => (TRAINED_SENSITIVITY, TRAINED_SPECIFICITY),
                };
                let p_model = posterior_hazard(
                    item.warning,
                    assumed_sensitivity,
                    assumed_specificity,
                );
                let model_fast = conditional_fast_value(scenario, p_model);
                let model_safe = conditional_safe_value(scenario);
                let (pred_a, pred_b) = if policy == Policy::ActionBlindPlacebo {
                    let shared = 0.5 * (model_fast + model_safe);
                    (shared, shared)
                } else if policy == Policy::ShuffledActionLabels {
                    (model_safe, model_fast)
                } else {
                    (model_fast, model_safe)
                };
                (
                    choose_fast(pred_a, pred_b),
                    true,
                    pred_a,
                    pred_b,
                )
            }
        };

        let chosen_value = if action_fast { true_fast } else { true_safe };
        conditional_regret += (true_fast.max(true_safe) - chosen_value).max(0.0);
        if !action_fast {
            action_disagreement += 1.0;
        }

        if evaluated {
            prediction_abs_error += 0.5
                * ((predicted_fast - true_fast).abs() + (predicted_safe - true_safe).abs());
            let predicted_prefers_fast = predicted_fast >= predicted_safe;
            if predicted_prefers_fast == true_prefers_fast {
                rank_correct += 1.0;
            }
        }

        total_return += if action_fast {
            item.reward_fast
        } else {
            item.reward_safe
        };
    }

    EpisodeOutcome {
        total_return,
        conditional_regret,
        prediction_abs_error,
        rank_correct,
        action_disagreement,
        model_queries,
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn sample_sd(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let average = mean(values);
    let sum_sq = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>();
    (sum_sq / (values.len() - 1) as f64).sqrt()
}

fn cvar95_return(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let count = ((sorted.len() as f64 * 0.05).ceil() as usize).max(1);
    mean(&sorted[..count])
}

fn summarize(
    scenario: Scenario,
    policy: Policy,
    outcomes: &[EpisodeOutcome],
    direct: &[EpisodeOutcome],
) -> Summary {
    let returns: Vec<f64> = outcomes.iter().map(|item| item.total_return).collect();
    let deltas: Vec<f64> = outcomes
        .iter()
        .zip(direct)
        .map(|(item, baseline)| item.total_return - baseline.total_return)
        .collect();
    let delta = mean(&deltas);
    let standard_error = sample_sd(&deltas) / (deltas.len() as f64).sqrt();
    let evaluable = policy.model_queries_per_decision() > 0;

    Summary {
        scenario: scenario.name(),
        policy: policy.name(),
        mean_return: mean(&returns),
        paired_delta_vs_direct: delta,
        delta_ci95_low: delta - 1.96 * standard_error,
        delta_ci95_high: delta + 1.96 * standard_error,
        cvar95_return: cvar95_return(&returns),
        success_rate: returns.iter().filter(|value| **value > 0.0).count() as f64
            / returns.len() as f64,
        mean_conditional_regret_per_step: outcomes
            .iter()
            .map(|item| item.conditional_regret)
            .sum::<f64>()
            / (outcomes.len() * HORIZON) as f64,
        action_disagreement_rate: outcomes
            .iter()
            .map(|item| item.action_disagreement)
            .sum::<f64>()
            / (outcomes.len() * HORIZON) as f64,
        mean_prediction_mae: if evaluable {
            outcomes
                .iter()
                .map(|item| item.prediction_abs_error)
                .sum::<f64>()
                / (outcomes.len() * HORIZON) as f64
        } else {
            -1.0
        },
        action_rank_accuracy: if evaluable {
            outcomes
                .iter()
                .map(|item| item.rank_correct)
                .sum::<f64>()
                / (outcomes.len() * HORIZON) as f64
        } else {
            -1.0
        },
        mean_model_queries_per_episode: outcomes
            .iter()
            .map(|item| item.model_queries as f64)
            .sum::<f64>()
            / outcomes.len() as f64,
        model_prediction_evaluable: evaluable,
    }
}

fn run_benchmark() -> Vec<Summary> {
    let mut rows = Vec::with_capacity(Scenario::all().len() * Policy::all().len());
    for scenario in Scenario::all() {
        let tapes: Vec<EpisodeTape> = (0..EPISODES)
            .map(|episode| make_episode_tape(scenario, episode))
            .collect();
        let direct: Vec<EpisodeOutcome> = tapes
            .iter()
            .map(|tape| run_episode(scenario, Policy::DirectController, tape))
            .collect();

        for policy in Policy::all() {
            let outcomes: Vec<EpisodeOutcome> = tapes
                .iter()
                .map(|tape| run_episode(scenario, policy, tape))
                .collect();
            rows.push(summarize(scenario, policy, &outcomes, &direct));
        }
    }
    rows
}

fn report_json(rows: &[Summary]) -> String {
    let scenarios = Scenario::all()
        .iter()
        .map(|scenario| format!("\"{}\"", scenario.name()))
        .collect::<Vec<_>>()
        .join(",");
    let policies = Policy::all()
        .iter()
        .map(|policy| format!("\"{}\"", policy.name()))
        .collect::<Vec<_>>()
        .join(",");
    let mut output = String::new();
    writeln!(&mut output, "{{").unwrap();
    writeln!(&mut output, "  \"schema_version\": \"{REPORT_SCHEMA}\",").unwrap();
    writeln!(&mut output, "  \"episode_count\": {EPISODES},").unwrap();
    writeln!(&mut output, "  \"horizon\": {HORIZON},").unwrap();
    writeln!(&mut output, "  \"shift_step\": {SHIFT_STEP},").unwrap();
    writeln!(&mut output, "  \"prior_hazard\": {PRIOR_HAZARD:.6},").unwrap();
    writeln!(&mut output, "  \"trained_sensitivity\": {TRAINED_SENSITIVITY:.6},").unwrap();
    writeln!(&mut output, "  \"trained_specificity\": {TRAINED_SPECIFICITY:.6},").unwrap();
    writeln!(&mut output, "  \"scenarios\": [{scenarios}],").unwrap();
    writeln!(&mut output, "  \"policies\": [{policies}],").unwrap();
    writeln!(&mut output, "  \"summary\": [").unwrap();
    for (index, row) in rows.iter().enumerate() {
        let comma = if index + 1 == rows.len() { "" } else { "," };
        writeln!(
            &mut output,
            concat!(
                "    {{\"scenario\":\"{}\",\"policy\":\"{}\",",
                "\"mean_return\":{:.8},\"paired_delta_vs_direct\":{:.8},",
                "\"delta_ci95_low\":{:.8},\"delta_ci95_high\":{:.8},",
                "\"cvar95_return\":{:.8},\"success_rate\":{:.8},",
                "\"mean_conditional_regret_per_step\":{:.8},",
                "\"action_disagreement_rate\":{:.8},\"mean_prediction_mae\":{:.8},",
                "\"action_rank_accuracy\":{:.8},\"mean_model_queries_per_episode\":{:.4},",
                "\"model_prediction_evaluable\":{}}}{}"
            ),
            row.scenario,
            row.policy,
            row.mean_return,
            row.paired_delta_vs_direct,
            row.delta_ci95_low,
            row.delta_ci95_high,
            row.cvar95_return,
            row.success_rate,
            row.mean_conditional_regret_per_step,
            row.action_disagreement_rate,
            row.mean_prediction_mae,
            row.action_rank_accuracy,
            row.mean_model_queries_per_episode,
            row.model_prediction_evaluable,
            comma
        )
        .unwrap();
    }
    writeln!(&mut output, "  ]").unwrap();
    writeln!(&mut output, "}}").unwrap();
    output
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("CMAQ-012 causal world-model advantage benchmark\nRuns 512 paired episodes per scenario, 64 decisions per episode.\nEmits deterministic JSON to stdout.");
        return;
    }
    if !args.is_empty() {
        eprintln!("unexpected argument: {}; use --help", args[0]);
        std::process::exit(2);
    }
    print!("{}", report_json(&run_benchmark()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayesian_posterior_matches_known_positive_observation() {
        let posterior = posterior_hazard(true, 0.80, 0.85);
        assert!((posterior - 0.64).abs() < 1e-10);
    }

    #[test]
    fn action_blind_placebo_is_the_direct_controller() {
        let tape = make_episode_tape(Scenario::PredictiveSignal, 7);
        let direct = run_episode(Scenario::PredictiveSignal, Policy::DirectController, &tape);
        let placebo = run_episode(Scenario::PredictiveSignal, Policy::ActionBlindPlacebo, &tape);
        assert!((direct.total_return - placebo.total_return).abs() < 1e-10);
        assert_eq!(direct.action_disagreement, placebo.action_disagreement);
        assert_eq!(placebo.model_queries, HORIZON * 2);
    }

    #[test]
    fn predictive_signal_creates_positive_world_model_advantage() {
        let rows = run_benchmark();
        let row = rows.iter().find(|row| {
            row.scenario == "predictive_signal"
                && row.policy == "frozen_calibration_action_model"
        }).unwrap();
        assert!(row.paired_delta_vs_direct > 0.0);
        assert!(row.delta_ci95_low > 0.0);
    }

    #[test]
    fn action_invariant_lane_has_no_artificial_prediction_lift() {
        let rows = run_benchmark();
        let direct = rows.iter().find(|row| {
            row.scenario == "action_invariant" && row.policy == "direct_controller"
        }).unwrap();
        let model = rows.iter().find(|row| {
            row.scenario == "action_invariant"
                && row.policy == "frozen_calibration_action_model"
        }).unwrap();
        assert!((model.paired_delta_vs_direct).abs() < 1e-10);
        assert!((model.mean_return - direct.mean_return).abs() < 1e-10);
    }

    #[test]
    fn low_information_and_sensor_shift_expose_stale_calibration() {
        let rows = run_benchmark();
        let low_info = rows.iter().find(|row| {
            row.scenario == "low_information"
                && row.policy == "frozen_calibration_action_model"
        }).unwrap();
        let low_info_calibrated = rows.iter().find(|row| {
            row.scenario == "low_information"
                && row.policy == "calibration_oracle_action_model"
        }).unwrap();
        assert!(low_info.paired_delta_vs_direct < 0.0);
        assert!(low_info_calibrated.paired_delta_vs_direct.abs() < 1e-10);

        let shifted = rows.iter().find(|row| {
            row.scenario == "sensor_shift"
                && row.policy == "frozen_calibration_action_model"
        }).unwrap();
        let shifted_calibrated = rows.iter().find(|row| {
            row.scenario == "sensor_shift"
                && row.policy == "calibration_oracle_action_model"
        }).unwrap();
        assert!(shifted.paired_delta_vs_direct < 0.0);
        assert!(shifted_calibrated.paired_delta_vs_direct > 0.0);
    }

    #[test]
    fn shuffled_action_labels_are_a_negative_control() {
        let rows = run_benchmark();
        let useful = rows.iter().find(|row| {
            row.scenario == "predictive_signal"
                && row.policy == "frozen_calibration_action_model"
        }).unwrap();
        let shuffled = rows.iter().find(|row| {
            row.scenario == "predictive_signal"
                && row.policy == "shuffled_action_labels"
        }).unwrap();
        assert!(shuffled.paired_delta_vs_direct < useful.paired_delta_vs_direct);
    }

    #[test]
    fn report_has_exact_scenario_policy_census() {
        let rows = run_benchmark();
        assert_eq!(rows.len(), Scenario::all().len() * Policy::all().len());
        let report = report_json(&rows);
        assert!(report.contains("\"schema_version\": \"cmaq-012-world-model-advantage-v0.1\""));
        assert!(report.contains("\"episode_count\": 512"));
        assert!(report.contains("\"horizon\": 64"));
    }
}
