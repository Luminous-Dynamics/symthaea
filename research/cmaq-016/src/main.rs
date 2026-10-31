use std::fmt::Write as _;

const SCHEMA: &str = "cmaq-016-search-exploitability-v0.1";
const EPISODES: usize = 256;
const EVALUATOR_DIAGNOSTIC_GRID_POINTS: usize = 201;
const BASE_SEED: u64 = 0xC0A9_0016_2026_1031;
const POCKET_CENTER: f64 = 0.72;
const POCKET_WIDTH: f64 = 0.018;
const POCKET_AMPLITUDE: f64 = 0.65;
// The five synthetic heads have separate error pockets. This is an explicit
// mechanism ablation, not a claim that learned ensemble heads are independent.
const ENSEMBLE_CENTERS: [f64; 5] = [0.56, 0.64, 0.72, 0.80, 0.88];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    BenignIidNoise,
    NarrowOptimismPocket,
}

impl Scenario {
    const ALL: [Self; 2] = [Self::BenignIidNoise, Self::NarrowOptimismPocket];

    fn name(self) -> &'static str {
        match self {
            Self::BenignIidNoise => "benign_iid_noise",
            Self::NarrowOptimismPocket => "narrow_optimism_pocket",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    DirectPolicy,
    RandomSearch,
    AdaptiveCem,
    PessimisticEnsembleCem,
    Oracle,
}

impl Policy {
    const ALL: [Self; 5] = [
        Self::DirectPolicy,
        Self::RandomSearch,
        Self::AdaptiveCem,
        Self::PessimisticEnsembleCem,
        Self::Oracle,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::DirectPolicy => "direct_policy",
            Self::RandomSearch => "random_search",
            Self::AdaptiveCem => "adaptive_cem",
            Self::PessimisticEnsembleCem => "pessimistic_ensemble_cem",
            Self::Oracle => "oracle",
        }
    }

    fn uses_model(self) -> bool {
        matches!(
            self,
            Self::RandomSearch | Self::AdaptiveCem | Self::PessimisticEnsembleCem
        )
    }

    fn model_heads_per_candidate(self) -> usize {
        match self {
            Self::PessimisticEnsembleCem => ENSEMBLE_CENTERS.len(),
            Self::RandomSearch | Self::AdaptiveCem => 1,
            Self::DirectPolicy | Self::Oracle => 0,
        }
    }
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

    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(f64::MIN_POSITIVE);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[derive(Clone, Copy, Debug)]
struct EpisodeResult {
    action: f64,
    true_return: f64,
    selected_model_score: Option<f64>,
    optimizer_optimism_gap: Option<f64>,
    candidate_queries: usize,
    model_head_evaluations: usize,
}

#[derive(Clone, Copy, Debug)]
struct Summary {
    scenario: &'static str,
    candidate_budget: usize,
    policy: &'static str,
    mean_true_return: f64,
    mean_selected_model_score: Option<f64>,
    mean_candidate_queries_per_episode: f64,
    mean_model_head_evaluations_per_episode: f64,
    mean_optimizer_optimism_gap: Option<f64>,
    mean_selected_action_regret: f64,
    selected_pocket_rate: f64,
    uniform_grid_point_model_mse: f64,
}

fn true_return(action: f64) -> f64 {
    1.0 - (action - 0.20).powi(2)
}

fn clamp_action(action: f64) -> f64 {
    action.clamp(-1.0, 1.0)
}

fn deterministic_noise(seed: u64, action: f64, head: usize) -> f64 {
    let action_bits = action.to_bits();
    let mixed = seed
        ^ action_bits.rotate_left(19)
        ^ (head as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    let mut rng = SplitMix64::new(mixed);
    (rng.unit() * 2.0 - 1.0) * 0.004
}

fn gaussian_bump(action: f64, center: f64, width: f64) -> f64 {
    let z = (action - center) / width;
    (-0.5 * z * z).exp()
}

fn model_head_prediction(
    scenario: Scenario,
    action: f64,
    episode_seed: u64,
    head: usize,
) -> f64 {
    let base = true_return(action);
    let noise = deterministic_noise(episode_seed, action, head);

    match scenario {
        Scenario::BenignIidNoise => base + noise,
        Scenario::NarrowOptimismPocket => {
            let center = ENSEMBLE_CENTERS[head.min(ENSEMBLE_CENTERS.len() - 1)];
            let bump = gaussian_bump(action, center, POCKET_WIDTH);
            base + POCKET_AMPLITUDE * bump + noise
        }
    }
}

fn reference_model_prediction(scenario: Scenario, action: f64, seed: u64) -> f64 {
    // The point-predictor baseline uses the central synthetic head. Other
    // heads exist solely for the pessimistic-ensemble comparator.
    model_head_prediction(scenario, action, seed, 2)
}

fn pessimistic_model_prediction(scenario: Scenario, action: f64, seed: u64) -> f64 {
    (0..ENSEMBLE_CENTERS.len())
        .map(|head| model_head_prediction(scenario, action, seed, head))
        .fold(f64::INFINITY, f64::min)
}

fn policy_score(scenario: Scenario, policy: Policy, action: f64, seed: u64) -> f64 {
    match policy {
        Policy::PessimisticEnsembleCem => pessimistic_model_prediction(scenario, action, seed),
        Policy::RandomSearch | Policy::AdaptiveCem => {
            reference_model_prediction(scenario, action, seed)
        }
        Policy::DirectPolicy | Policy::Oracle => true_return(action),
    }
}

fn sample_uniform(rng: &mut SplitMix64) -> f64 {
    rng.unit() * 2.0 - 1.0
}

fn optimize(
    scenario: Scenario,
    policy: Policy,
    candidate_budget: usize,
    episode_seed: u64,
) -> EpisodeResult {
    let mut rng = SplitMix64::new(
        episode_seed
            ^ (candidate_budget as u64).rotate_left(7)
            ^ (policy as u64).wrapping_mul(0xA076_1D64_78BD_642F),
    );

    let (action, selected_model_score) = match policy {
        Policy::DirectPolicy => (-0.20, None),
        Policy::Oracle => (0.20, None),
        Policy::RandomSearch => {
            let mut best_action = 0.0;
            let mut best_score = f64::NEG_INFINITY;
            for _ in 0..candidate_budget {
                let candidate = sample_uniform(&mut rng);
                let score = policy_score(scenario, policy, candidate, episode_seed);
                if score > best_score {
                    best_score = score;
                    best_action = candidate;
                }
            }
            (best_action, Some(best_score))
        }
        Policy::AdaptiveCem | Policy::PessimisticEnsembleCem => {
            cem_search(scenario, policy, candidate_budget, episode_seed, &mut rng)
        }
    };

    let actual = true_return(action);
    EpisodeResult {
        action,
        true_return: actual,
        selected_model_score,
        optimizer_optimism_gap: selected_model_score.map(|score| score - actual),
        candidate_queries: if policy.uses_model() { candidate_budget } else { 0 },
        model_head_evaluations: candidate_budget * policy.model_heads_per_candidate(),
    }
}

fn cem_search(
    scenario: Scenario,
    policy: Policy,
    candidate_budget: usize,
    episode_seed: u64,
    rng: &mut SplitMix64,
) -> (f64, Option<f64>) {
    let rounds = 3usize;
    let base_per_round = candidate_budget / rounds;
    let remainder = candidate_budget % rounds;
    let mut mean = -0.15;
    let mut std_dev = 0.72;
    // No uncounted model query is made at initialization: every score below
    // corresponds to one declared candidate sequence evaluation.
    let mut best_action = mean;
    let mut best_score = f64::NEG_INFINITY;

    for round in 0..rounds {
        let count = base_per_round + (round < remainder) as usize;
        if count == 0 {
            continue;
        }

        let mut candidates = Vec::with_capacity(count);
        for _ in 0..count {
            let action = clamp_action(mean + std_dev * rng.normal());
            let score = policy_score(scenario, policy, action, episode_seed);
            candidates.push((action, score));
            if score > best_score {
                best_score = score;
                best_action = action;
            }
        }

        candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
        let elite_count = (count / 5).max(1).min(candidates.len());
        let elite = &candidates[..elite_count];
        let elite_mean =
            elite.iter().map(|(action, _)| *action).sum::<f64>() / elite_count as f64;
        let variance = elite
            .iter()
            .map(|(action, _)| (*action - elite_mean).powi(2))
            .sum::<f64>()
            / elite_count as f64;
        mean = elite_mean;
        std_dev = (variance.sqrt() * 1.25).clamp(0.02, 0.72);
    }

    (best_action, Some(best_score))
}

fn uniform_grid_point_model_mse(scenario: Scenario, seed: u64) -> f64 {
    let points = EVALUATOR_DIAGNOSTIC_GRID_POINTS;
    (0..points)
        .map(|index| {
            let action = -1.0 + 2.0 * index as f64 / (points - 1) as f64;
            let residual = reference_model_prediction(scenario, action, seed) - true_return(action);
            residual * residual
        })
        .sum::<f64>()
        / points as f64
}

fn summarize(scenario: Scenario, candidate_budget: usize, policy: Policy) -> Summary {
    let mut actual_sum = 0.0;
    let mut model_score_sum = 0.0;
    let mut candidate_query_sum = 0usize;
    let mut head_evaluation_sum = 0usize;
    let mut optimism_gap_sum = 0.0;
    let mut model_score_count = 0usize;
    let mut regret_sum = 0.0;
    let mut pocket_count = 0usize;
    let mut mse_sum = 0.0;

    for episode in 0..EPISODES {
        let episode_seed = BASE_SEED ^ (episode as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let result = optimize(scenario, policy, candidate_budget, episode_seed);
        actual_sum += result.true_return;
        candidate_query_sum += result.candidate_queries;
        head_evaluation_sum += result.model_head_evaluations;
        regret_sum += 1.0 - result.true_return;
        mse_sum += uniform_grid_point_model_mse(scenario, episode_seed);

        if let Some(score) = result.selected_model_score {
            model_score_sum += score;
            model_score_count += 1;
        }
        if let Some(gap) = result.optimizer_optimism_gap {
            optimism_gap_sum += gap;
        }

        let pocket_hit = if policy == Policy::PessimisticEnsembleCem {
            ENSEMBLE_CENTERS
                .iter()
                .any(|center| (result.action - center).abs() <= 0.045)
        } else {
            (result.action - POCKET_CENTER).abs() <= 0.045
        };
        if scenario == Scenario::NarrowOptimismPocket && pocket_hit {
            pocket_count += 1;
        }

        debug_assert!(result.candidate_queries <= candidate_budget);
        debug_assert_eq!(
            result.model_head_evaluations,
            candidate_budget * policy.model_heads_per_candidate()
        );
    }

    let denominator = EPISODES as f64;
    Summary {
        scenario: scenario.name(),
        candidate_budget,
        policy: policy.name(),
        mean_true_return: actual_sum / denominator,
        mean_selected_model_score: (model_score_count > 0)
            .then(|| model_score_sum / model_score_count as f64),
        mean_candidate_queries_per_episode: candidate_query_sum as f64 / denominator,
        mean_model_head_evaluations_per_episode: head_evaluation_sum as f64 / denominator,
        mean_optimizer_optimism_gap: (model_score_count > 0)
            .then(|| optimism_gap_sum / model_score_count as f64),
        mean_selected_action_regret: regret_sum / denominator,
        selected_pocket_rate: pocket_count as f64 / denominator,
        uniform_grid_point_model_mse: mse_sum / denominator,
    }
}

fn write_optional_number(output: &mut String, value: Option<f64>) {
    match value {
        Some(number) => write!(output, "{number:.9}").expect("writing to String cannot fail"),
        None => output.push_str("null"),
    }
}

fn render_report() -> String {
    let candidate_budgets = [8usize, 32, 128, 512];
    let mut summaries = Vec::new();

    for scenario in Scenario::ALL {
        for candidate_budget in candidate_budgets {
            for policy in Policy::ALL {
                summaries.push(summarize(scenario, candidate_budget, policy));
            }
        }
    }

    let mut output = String::new();
    write!(
        &mut output,
        "{{\"schema_version\":\"{}\",\"episode_count\":{},\"seed\":{},",
        SCHEMA, EPISODES, BASE_SEED
    )
    .expect("writing to String cannot fail");
    output.push_str(
        "\"evaluator_diagnostic_grid_points\":201,\"candidate_budgets\":[8,32,128,512],\"scenarios\":[\"benign_iid_noise\",\"narrow_optimism_pocket\"],\"policies\":[\"direct_policy\",\"random_search\",\"adaptive_cem\",\"pessimistic_ensemble_cem\",\"oracle\"],\"summary\":[",
    );

    for (index, row) in summaries.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        write!(
            &mut output,
            "{{\"scenario\":\"{}\",\"candidate_budget\":{},\"policy\":\"{}\",\"mean_true_return\":{:.9},\"mean_selected_model_score\":",
            row.scenario, row.candidate_budget, row.policy, row.mean_true_return
        )
        .expect("writing to String cannot fail");
        write_optional_number(&mut output, row.mean_selected_model_score);
        output.push_str(",\"mean_candidate_queries_per_episode\":");
        write!(
            &mut output,
            "{:.3},\"mean_model_head_evaluations_per_episode\":{:.3},\"mean_optimizer_optimism_gap\":",
            row.mean_candidate_queries_per_episode, row.mean_model_head_evaluations_per_episode
        )
        .expect("writing to String cannot fail");
        write_optional_number(&mut output, row.mean_optimizer_optimism_gap);
        write!(
            &mut output,
            ",\"mean_selected_action_regret\":{:.9},\"selected_pocket_rate\":{:.9},\"uniform_grid_point_model_mse\":{:.9}}}",
            row.mean_selected_action_regret,
            row.selected_pocket_rate,
            row.uniform_grid_point_model_mse
        )
        .expect("writing to String cannot fail");
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
    fn true_reward_has_known_global_optimum() {
        assert!((true_return(0.20) - 1.0).abs() < 1e-12);
        assert!(true_return(0.72) < true_return(0.20));
    }

    #[test]
    fn localized_optimism_is_narrow_and_positive() {
        let center = model_head_prediction(
            Scenario::NarrowOptimismPocket,
            POCKET_CENTER,
            BASE_SEED,
            2,
        );
        let far = model_head_prediction(
            Scenario::NarrowOptimismPocket,
            -0.50,
            BASE_SEED,
            2,
        );
        assert!(center - true_return(POCKET_CENTER) > 0.60);
        assert!((far - true_return(-0.50)).abs() < 0.01);
    }

    #[test]
    fn pessimistic_minimum_suppresses_a_single_head_pocket() {
        let optimistic =
            reference_model_prediction(Scenario::NarrowOptimismPocket, POCKET_CENTER, BASE_SEED);
        let pessimistic =
            pessimistic_model_prediction(Scenario::NarrowOptimismPocket, POCKET_CENTER, BASE_SEED);
        assert!(pessimistic < optimistic);
        assert!(pessimistic - true_return(POCKET_CENTER) < 0.02);
    }

    #[test]
    fn query_and_head_evaluation_budgets_are_explicit() {
        for policy in Policy::ALL {
            let result = optimize(Scenario::NarrowOptimismPocket, policy, 32, BASE_SEED);
            assert_eq!(result.candidate_queries, if policy.uses_model() { 32 } else { 0 });
            assert_eq!(
                result.model_head_evaluations,
                32 * policy.model_heads_per_candidate()
            );
        }
    }

    #[test]
    fn adaptive_search_exposes_the_narrow_optimism_pocket_as_budget_grows() {
        let low = summarize(Scenario::NarrowOptimismPocket, 8, Policy::AdaptiveCem);
        let high = summarize(Scenario::NarrowOptimismPocket, 512, Policy::AdaptiveCem);
        assert!(high.selected_pocket_rate > low.selected_pocket_rate);
        assert!(high.mean_true_return < low.mean_true_return);
    }

    #[test]
    fn pessimistic_ensemble_reduces_the_synthetic_exploit_gap() {
        let greedy = summarize(Scenario::NarrowOptimismPocket, 512, Policy::AdaptiveCem);
        let robust = summarize(
            Scenario::NarrowOptimismPocket,
            512,
            Policy::PessimisticEnsembleCem,
        );
        assert!(robust.selected_pocket_rate < 0.10);
        assert!(robust.mean_true_return > greedy.mean_true_return + 0.10);
    }

    #[test]
    fn deterministic_report_replays_byte_for_byte() {
        assert_eq!(render_report(), render_report());
    }

    #[test]
    fn oracle_regret_is_exactly_zero() {
        for scenario in Scenario::ALL {
            for budget in [8usize, 32, 128, 512] {
                let result = optimize(scenario, Policy::Oracle, budget, BASE_SEED);
                assert!((1.0 - result.true_return).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn direct_policy_is_independent_of_model_error_lane() {
        let benign = optimize(Scenario::BenignIidNoise, Policy::DirectPolicy, 128, BASE_SEED);
        let exploit =
            optimize(Scenario::NarrowOptimismPocket, Policy::DirectPolicy, 128, BASE_SEED);
        assert_eq!(benign.action, exploit.action);
        assert_eq!(benign.true_return, exploit.true_return);
    }
}
