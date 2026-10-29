use std::process;

const REPORT_SCHEMA: &str = "cmaq-004-value-add-report-v0.3";
const TRAIN_CONTEXTS: [usize; 3] = [0, 1, 2];
const HELDOUT_CONTEXT: usize = 3;
const SUPPORT_SAMPLES: usize = 1_024;
const EVALUATION_EPISODES: usize = 4_096;
const SEED: u64 = 4_204_2026;
const MODEL_QUERY_COST_LEVELS: [f64; 6] = [0.0, 0.02, 0.04, 0.08, 0.12, 0.16];
const MODEL_QUERIES_PER_DECISION: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

    fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorldFamily {
    AdditivePositive,
    InteractionOnly,
    AdditiveReversed,
}

impl WorldFamily {
    const ALL: [Self; 3] = [
        Self::AdditivePositive,
        Self::InteractionOnly,
        Self::AdditiveReversed,
    ];

    fn id(self) -> u64 {
        match self {
            Self::AdditivePositive => 0,
            Self::InteractionOnly => 1,
            Self::AdditiveReversed => 2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::AdditivePositive => "additive_positive",
            Self::InteractionOnly => "interaction_only",
            Self::AdditiveReversed => "additive_reversed",
        }
    }
}

fn context_factors(context: usize) -> (f64, f64) {
    ((context / 2) as f64, (context % 2) as f64)
}

fn reward_probability(world: WorldFamily, context: usize, action: usize) -> f64 {
    if action == 0 {
        return 0.5;
    }
    let (x, y) = context_factors(context);
    match world {
        WorldFamily::AdditivePositive => 0.15 + 0.30 * x + 0.30 * y,
        WorldFamily::InteractionOnly => 0.15 + 0.60 * x * y,
        WorldFamily::AdditiveReversed => 0.15 + 0.30 * (1.0 - x) + 0.30 * (1.0 - y),
    }
}

fn clamp_probability(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SupportMeans {
    by_context_action: [[f64; 2]; 3],
}

impl SupportMeans {
    fn fit(world: WorldFamily, seed: u64) -> Self {
        let mut means = [[0.0_f64; 2]; 3];
        for (row, context) in TRAIN_CONTEXTS.iter().copied().enumerate() {
            for action in 0..2 {
                let probability = reward_probability(world, context, action);
                let stream_seed = seed
                    ^ ((world.id() + 1).wrapping_mul(0x9E37_79B9))
                    ^ ((context as u64 + 1).wrapping_mul(0xBF58_476D))
                    ^ ((action as u64 + 1).wrapping_mul(0x94D0_49BB));
                let mut rng = SplitMix64::new(stream_seed);
                let mut successes = 0usize;
                for _ in 0..SUPPORT_SAMPLES {
                    if rng.unit_f64() < probability {
                        successes += 1;
                    }
                }
                means[row][action] = successes as f64 / SUPPORT_SAMPLES as f64;
            }
        }
        Self {
            by_context_action: means,
        }
    }

    fn nearest_context_action_values(self) -> [f64; 2] {
        // Held-out 11 is one factor flip away from the observed contexts 01 and 10.
        // This is a strong non-model baseline: nearest-context fallback without fitting
        // a compositional reward law.
        [
            (self.by_context_action[1][0] + self.by_context_action[2][0]) / 2.0,
            (self.by_context_action[1][1] + self.by_context_action[2][1]) / 2.0,
        ]
    }

    // An additive model has p(x,y) = intercept + x_effect*x + y_effect*y.
    // With support at (0,0), (0,1), and (1,0), its extrapolation to unseen (1,1)
    // is p(0,1) + p(1,0) - p(0,0). This exposes the model's inductive bias.
    fn additive_prediction_at_heldout(self) -> [f64; 2] {
        [
            clamp_probability(
                self.by_context_action[1][0] + self.by_context_action[2][0]
                    - self.by_context_action[0][0],
            ),
            clamp_probability(
                self.by_context_action[1][1] + self.by_context_action[2][1]
                    - self.by_context_action[0][1],
            ),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Strategy {
    DirectTabular,
    LearnedAdditiveModel,
    ShuffledModel,
    StaleAdditiveModel,
    Oracle,
}

impl Strategy {
    const ALL: [Self; 5] = [
        Self::DirectTabular,
        Self::LearnedAdditiveModel,
        Self::ShuffledModel,
        Self::StaleAdditiveModel,
        Self::Oracle,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::DirectTabular => "direct_nearest_context",
            Self::LearnedAdditiveModel => "learned_additive_model",
            Self::ShuffledModel => "shuffled_model_control",
            Self::StaleAdditiveModel => "stale_model_control",
            Self::Oracle => "oracle_evaluator_only",
        }
    }

    fn uses_model(self) -> bool {
        matches!(
            self,
            Self::LearnedAdditiveModel | Self::ShuffledModel | Self::StaleAdditiveModel
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Decision {
    action: usize,
    predictions: Option<[f64; 2]>,
    model_query_count: usize,
    query_cost: f64,
}

fn choose_action(values: [f64; 2]) -> usize {
    if values[1] > values[0] { 1 } else { 0 }
}

fn decide(world: WorldFamily, support: SupportMeans, strategy: Strategy, cost_per_query: f64) -> Decision {
    let model_prediction = support.additive_prediction_at_heldout();
    let (action, predictions) = match strategy {
        Strategy::DirectTabular => (choose_action(support.nearest_context_action_values()), None),
        Strategy::LearnedAdditiveModel => (choose_action(model_prediction), Some(model_prediction)),
        Strategy::ShuffledModel => {
            let shuffled = [model_prediction[1], model_prediction[0]];
            (choose_action(shuffled), Some(shuffled))
        }
        Strategy::StaleAdditiveModel => {
            // Negative control: a plausible model from a prior regime with positive factor effects.
            let stale = [0.5, 0.75];
            (choose_action(stale), Some(stale))
        }
        Strategy::Oracle => {
            let values = [
                reward_probability(world, HELDOUT_CONTEXT, 0),
                reward_probability(world, HELDOUT_CONTEXT, 1),
            ];
            (choose_action(values), None)
        }
    };
    let model_query_count = if strategy.uses_model() {
        MODEL_QUERIES_PER_DECISION
    } else {
        0
    };
    Decision {
        action,
        predictions,
        model_query_count,
        query_cost: model_query_count as f64 * cost_per_query,
    }
}

#[derive(Clone, Debug, PartialEq)]
struct StrategyResult {
    name: &'static str,
    chosen_action: usize,
    model_query_count: usize,
    model_query_cost: f64,
    model_prediction_mae: Option<f64>,
    mean_reward: f64,
    mean_net_utility: f64,
    oracle_mean_reward: f64,
    mean_regret: f64,
    mean_net_regret: f64,
    net_lift_vs_direct: f64,
    net_lift_standard_error: f64,
    net_lift_ci95_low: f64,
    net_lift_ci95_high: f64,
    model_query_cost_per_query: f64,
}

fn nearest_model_mae(predictions: Option<[f64; 2]>, world: WorldFamily) -> Option<f64> {
    predictions.map(|values| {
        let truth = [
            reward_probability(world, HELDOUT_CONTEXT, 0),
            reward_probability(world, HELDOUT_CONTEXT, 1),
        ];
        ((values[0] - truth[0]).abs() + (values[1] - truth[1]).abs()) / 2.0
    })
}

fn evaluate(world: WorldFamily, strategy: Strategy, support: SupportMeans, seed: u64, cost_per_query: f64) -> StrategyResult {
    let decision = decide(world, support, strategy, cost_per_query);
    let direct_action = decide(world, support, Strategy::DirectTabular, cost_per_query).action;
    let direct_probability = reward_probability(world, HELDOUT_CONTEXT, direct_action);
    let chosen_probability = reward_probability(world, HELDOUT_CONTEXT, decision.action);
    let oracle_probability = reward_probability(world, HELDOUT_CONTEXT, 0)
        .max(reward_probability(world, HELDOUT_CONTEXT, 1));
    let mut reward_total = 0_u64;
    let mut oracle_reward_total = 0_u64;
    let mut regret_total = 0_u64;
    let mut paired_lift_sum = 0.0_f64;
    let mut paired_lift_square_sum = 0.0_f64;

    for episode in 0..EVALUATION_EPISODES {
        let stream_seed = seed
            ^ ((world.id() + 1).wrapping_mul(0x9E37_79B9))
            ^ ((episode as u64 + 1).wrapping_mul(0xBF58_476D));
        let mut rng = SplitMix64::new(stream_seed);
        let shared_draw = rng.unit_f64();
        let reward = if shared_draw < chosen_probability { 1_u64 } else { 0_u64 };
        let oracle_reward = if shared_draw < oracle_probability { 1_u64 } else { 0_u64 };
        let direct_reward = if shared_draw < direct_probability { 1_u64 } else { 0_u64 };
        let paired_lift = reward as f64 - decision.query_cost - direct_reward as f64;
        paired_lift_sum += paired_lift;
        paired_lift_square_sum += paired_lift * paired_lift;
        reward_total += reward;
        oracle_reward_total += oracle_reward;
        regret_total += oracle_reward.saturating_sub(reward);
    }

    let mean_reward = reward_total as f64 / EVALUATION_EPISODES as f64;
    let oracle_mean_reward = oracle_reward_total as f64 / EVALUATION_EPISODES as f64;
    let mean_net_utility = mean_reward - decision.query_cost;
    let mean_regret = regret_total as f64 / EVALUATION_EPISODES as f64;
    let mean_net_regret = oracle_mean_reward - mean_net_utility;
    let net_lift_vs_direct = paired_lift_sum / EVALUATION_EPISODES as f64;
    let variance_numerator = (paired_lift_square_sum
        - paired_lift_sum * paired_lift_sum / EVALUATION_EPISODES as f64)
        .max(0.0);
    let sample_variance = variance_numerator / (EVALUATION_EPISODES - 1) as f64;
    let net_lift_standard_error = (sample_variance / EVALUATION_EPISODES as f64).sqrt();

    StrategyResult {
        name: strategy.name(),
        chosen_action: decision.action,
        model_query_count: decision.model_query_count,
        model_query_cost: decision.query_cost,
        model_prediction_mae: nearest_model_mae(decision.predictions, world),
        mean_reward,
        mean_net_utility,
        oracle_mean_reward,
        mean_regret,
        mean_net_regret,
        net_lift_vs_direct,
        net_lift_standard_error,
        net_lift_ci95_low: net_lift_vs_direct - 1.96 * net_lift_standard_error,
        net_lift_ci95_high: net_lift_vs_direct + 1.96 * net_lift_standard_error,
        model_query_cost_per_query: cost_per_query,
    }
}

fn evaluate_family(world: WorldFamily) -> Vec<StrategyResult> {
    let support = SupportMeans::fit(world, SEED);
    let mut rows = Vec::with_capacity(Strategy::ALL.len() * MODEL_QUERY_COST_LEVELS.len());
    for cost_per_query in MODEL_QUERY_COST_LEVELS {
        rows.extend(Strategy::ALL.iter().copied().map(|strategy| {
            evaluate(world, strategy, support, SEED, cost_per_query)
        }));
    }
    rows
}

fn emit_optional_float(value: Option<f64>) {
    match value {
        Some(number) => print!("{number:.6}"),
        None => print!("null"),
    }
}

fn emit_report() {
    print!(
        "{{\"schema_version\":\"{}\",\"seed\":{},\"support_contexts\":[0,1,2],\"heldout_context\":{},\"support_samples_per_context_action\":{},\"evaluation_episodes\":{},\"model_query_cost_levels\":[",
        REPORT_SCHEMA, SEED, HELDOUT_CONTEXT, SUPPORT_SAMPLES, EVALUATION_EPISODES
    );
    for (index, cost) in MODEL_QUERY_COST_LEVELS.iter().copied().enumerate() {
        if index > 0 {
            print!(",");
        }
        print!("{cost:.2}");
    }
    print!("],\"families\":[");
    for (world_index, world) in WorldFamily::ALL.iter().copied().enumerate() {
        if world_index > 0 {
            print!(",");
        }
        print!("{{\"world_family\":\"{}\",\"results\":[", world.name());
        let results = evaluate_family(world);
        for (index, row) in results.iter().enumerate() {
            if index > 0 {
                print!(",");
            }
            print!(
                "{{\"strategy\":\"{}\",\"chosen_action\":{},\"model_query_count\":{},\"model_query_cost_per_query\":{:.4},\"model_query_cost\":{:.4},\"model_prediction_mae\":",
                row.name, row.chosen_action, row.model_query_count, row.model_query_cost_per_query, row.model_query_cost
            );
            emit_optional_float(row.model_prediction_mae);
            print!(
                ",\"mean_reward\":{:.6},\"mean_net_utility\":{:.6},\"oracle_mean_reward\":{:.6},\"mean_regret\":{:.6},\"mean_net_regret\":{:.6},\"net_lift_vs_direct\":{:.6},\"net_lift_standard_error\":{:.6},\"net_lift_ci95_low\":{:.6},\"net_lift_ci95_high\":{:.6}}}",
                row.mean_reward, row.mean_net_utility, row.oracle_mean_reward,
                row.mean_regret, row.mean_net_regret, row.net_lift_vs_direct,
                row.net_lift_standard_error, row.net_lift_ci95_low, row.net_lift_ci95_high
            );
        }
        print!("]}}");
    }
    println!("]}}");
}

fn main() {
    let result = std::panic::catch_unwind(emit_report);
    if result.is_err() {
        eprintln!("CMAQ-004 value-add smoke harness failed");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result_for(world: WorldFamily, strategy: Strategy, cost_per_query: f64) -> StrategyResult {
        let support = SupportMeans::fit(world, SEED);
        evaluate(world, strategy, support, SEED, cost_per_query)
    }

    #[test]
    fn fitted_model_lifts_net_utility_on_heldout_additive_composition() {
        let result = result_for(WorldFamily::AdditivePositive, Strategy::LearnedAdditiveModel, 0.04);
        assert_eq!(result.chosen_action, 1);
        assert!(result.net_lift_vs_direct > 0.10);
        assert!(result.net_lift_ci95_low > 0.0);
        assert!(result.model_query_cost > 0.0);
        assert!(result.model_prediction_mae.unwrap() < 0.05);
    }

    #[test]
    fn interaction_only_negative_control_exposes_additive_model_misspecification() {
        let result = result_for(WorldFamily::InteractionOnly, Strategy::LearnedAdditiveModel, 0.04);
        assert_eq!(result.chosen_action, 0);
        assert!(result.net_lift_vs_direct < -0.05);
        assert!(result.net_lift_ci95_high < 0.0);
        assert!(result.model_prediction_mae.unwrap() > 0.20);
    }

    #[test]
    fn stale_model_can_be_worse_than_direct_policy_after_regime_reversal() {
        let direct = result_for(WorldFamily::AdditiveReversed, Strategy::DirectTabular, 0.04);
        let stale = result_for(WorldFamily::AdditiveReversed, Strategy::StaleAdditiveModel, 0.04);
        assert_eq!(direct.chosen_action, 0);
        assert_eq!(stale.chosen_action, 1);
        assert!(stale.mean_net_utility < direct.mean_net_utility - 0.30);
        assert!(stale.mean_net_regret > 0.30);
    }

    #[test]
    fn shuffled_model_control_breaks_action_prediction_alignment() {
        let direct = result_for(WorldFamily::AdditivePositive, Strategy::DirectTabular, 0.04);
        let shuffled = result_for(WorldFamily::AdditivePositive, Strategy::ShuffledModel, 0.04);
        let fitted = result_for(WorldFamily::AdditivePositive, Strategy::LearnedAdditiveModel, 0.04);
        assert_eq!(direct.chosen_action, 0);
        assert_eq!(shuffled.chosen_action, 0);
        assert!(shuffled.mean_net_utility < direct.mean_net_utility);
        assert!(shuffled.net_lift_vs_direct < 0.0);
        assert!(fitted.mean_net_utility > shuffled.mean_net_utility);
    }

    #[test]
    fn cost_frontier_exposes_model_break_even_point() {
        let free = result_for(WorldFamily::AdditivePositive, Strategy::LearnedAdditiveModel, 0.0);
        let normal_cost = result_for(WorldFamily::AdditivePositive, Strategy::LearnedAdditiveModel, 0.04);
        let expensive = result_for(WorldFamily::AdditivePositive, Strategy::LearnedAdditiveModel, 0.16);
        assert!(free.net_lift_vs_direct > normal_cost.net_lift_vs_direct);
        assert!(normal_cost.net_lift_vs_direct > expensive.net_lift_vs_direct);
        assert!(free.net_lift_vs_direct > 0.10);
        assert!(normal_cost.net_lift_ci95_low > 0.0);
        assert!(expensive.net_lift_vs_direct < 0.0);
        assert!(expensive.net_lift_ci95_high < 0.0);
        assert_eq!(expensive.model_query_count, 2);
        assert!((expensive.model_query_cost - 0.32).abs() < 1e-9);
    }

    #[test]
    fn oracle_upper_bound_dominates_net_utility() {
        for world in WorldFamily::ALL {
            let oracle = result_for(world, Strategy::Oracle, 0.04);
            for strategy in Strategy::ALL {
                let result = result_for(world, strategy, 0.04);
                assert!(
                    result.mean_net_utility <= oracle.oracle_mean_reward + 1e-12,
                    "{} exceeds evaluator-only oracle in {}",
                    result.name,
                    world.name()
                );
            }
        }
    }

    #[test]
    fn evaluation_is_deterministic_for_fixed_seed() {
        let first = result_for(WorldFamily::AdditivePositive, Strategy::LearnedAdditiveModel, 0.04);
        let second = result_for(WorldFamily::AdditivePositive, Strategy::LearnedAdditiveModel, 0.04);
        assert_eq!(first.mean_reward, second.mean_reward);
        assert_eq!(first.mean_net_utility, second.mean_net_utility);
        assert_eq!(first.mean_regret, second.mean_regret);
    }
}
