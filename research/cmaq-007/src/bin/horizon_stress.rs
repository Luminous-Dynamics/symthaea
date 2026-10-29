//! CMAQ-007 v0.5 companion panel: persistent model error across rollout depth.
//!
//! This is a synthetic stress model, not a learned world model or a physical simulator.
//! The rollout_growth parameter controls geometric amplification of action-specific model
//! error. It is intentionally exposed as an experimental factor, not fitted to real data.

use std::process;

const REPORT_SCHEMA: &str = "cmaq-007-horizon-stress-v0.1";
const SEEDS: usize = 256;
const BASE_SEED: u64 = 0xC0A7_0075_5EED_2026;
const SEARCH_SIZES: [usize; 4] = [8, 32, 128, 512];
const HORIZONS: [usize; 3] = [1, 4, 16];
const ROLLOUT_GROWTH: [f64; 2] = [0.0, 0.20];
const FALLBACK_UTILITY: f64 = 0.36;
const UNCERTAINTY_PENALTY: f64 = 2.0;

#[derive(Clone, Debug, PartialEq)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1_u64 << 53) as f64)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Action {
    true_utility: f64,
    persistent_error: f64,
    stated_uncertainty: f64,
    support_score: f64,
    supported: bool,
    ensemble_noise: [f64; 5],
}

#[derive(Clone, Debug, PartialEq)]
struct Candidate {
    true_utility: f64,
    predicted_utility: f64,
    ensemble_min_utility: f64,
    stated_uncertainty: f64,
    support_score: f64,
    supported: bool,
}

fn make_catalog(seed: u64) -> Vec<Action> {
    let mut rng = Rng::new(seed);
    (0..512)
        .map(|_| {
            let supported = rng.unit() < 0.84;
            let true_utility = if supported {
                0.18 + 0.52 * rng.unit()
            } else {
                0.05 + 0.40 * rng.unit()
            };
            let persistent_error = if supported {
                (rng.unit() - 0.5) * 0.08
            } else {
                0.035 + 0.10 * rng.unit()
            };
            // The synthetic uncertainty estimate is deliberately under-responsive for
            // unsupported actions: it is an input to test, not an oracle support label.
            let stated_uncertainty = if supported {
                0.020 + 0.025 * rng.unit()
            } else {
                0.004 + 0.010 * rng.unit()
            };
            let support_score = if supported {
                0.35 + 0.65 * rng.unit()
            } else {
                0.15 + 0.85 * rng.unit()
            };
            // Five synthetic model heads share the structural error above but have
            // independent small residuals. Their minimum is a robustness comparator,
            // not a trained ensemble and not a substitute for support evidence.
            let head_noise_scale = if supported { 0.035 } else { 0.012 };
            let ensemble_noise: [f64; 5] =
                std::array::from_fn(|_| (rng.unit() - 0.5) * 2.0 * head_noise_scale);
            Action {
                true_utility,
                persistent_error,
                stated_uncertainty,
                support_score,
                supported,
                ensemble_noise,
            }
        })
        .collect()
}

/// Sum of geometric propagation weights for a persistent transition/model error.
/// growth = 0 means one equal-sized error at every step; positive growth amplifies it.
fn amplification_ratio(horizon: usize, rollout_growth: f64) -> f64 {
    let mut total = 0.0;
    let mut weight = 1.0;
    for _ in 0..horizon {
        total += weight;
        weight *= 1.0 + rollout_growth;
    }
    total / horizon as f64
}

fn project(actions: &[Action], horizon: usize, rollout_growth: f64) -> Vec<Candidate> {
    let amplification = amplification_ratio(horizon, rollout_growth);
    actions
        .iter()
        .map(|action| {
            let ensemble_min_utility = action.ensemble_noise
                .iter()
                .map(|noise| {
                    action.true_utility
                        + (action.persistent_error + noise) * amplification
                })
                .fold(f64::INFINITY, f64::min);
            Candidate {
                true_utility: action.true_utility,
                predicted_utility: action.true_utility + action.persistent_error * amplification,
                ensemble_min_utility,
                stated_uncertainty: action.stated_uncertainty,
                support_score: action.support_score,
                supported: action.supported,
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Strategy {
    DirectFallback,
    PredictedGreedy,
    UncertaintyPenalty,
    EnsembleMin,
    SupportGate060,
    SupportGate090,
    OracleUpperBound,
}

impl Strategy {
    const ALL: [Self; 7] = [
        Self::DirectFallback,
        Self::PredictedGreedy,
        Self::UncertaintyPenalty,
        Self::EnsembleMin,
        Self::SupportGate060,
        Self::SupportGate090,
        Self::OracleUpperBound,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::DirectFallback => "direct_fallback",
            Self::PredictedGreedy => "predicted_greedy",
            Self::UncertaintyPenalty => "uncertainty_penalty",
            Self::EnsembleMin => "ensemble_min",
            Self::SupportGate060 => "support_gate_060",
            Self::SupportGate090 => "support_gate_090",
            Self::OracleUpperBound => "oracle_upper_bound",
        }
    }

    fn support_floor(self) -> Option<f64> {
        match self {
            Self::SupportGate060 => Some(0.60),
            Self::SupportGate090 => Some(0.90),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct EpisodeMetrics {
    prediction_mae: f64,
    realized_utility: f64,
    selected_optimism_gap: f64,
    selected_decision_score_gap: f64,
    regret: f64,
    unsupported_selected: f64,
    coverage: f64,
}

fn prediction_mae(actions: &[Candidate]) -> f64 {
    actions
        .iter()
        .map(|action| (action.predicted_utility - action.true_utility).abs())
        .sum::<f64>()
        / actions.len().max(1) as f64
}

fn evaluate_episode(actions: &[Candidate], strategy: Strategy) -> EpisodeMetrics {
    let mut best_index: Option<usize> = None;
    let mut best_policy_score = FALLBACK_UTILITY;

    if strategy != Strategy::DirectFallback {
        for (index, action) in actions.iter().enumerate() {
            if let Some(floor) = strategy.support_floor() {
                if action.support_score < floor {
                    continue;
                }
            }
            let policy_score = match strategy {
                Strategy::PredictedGreedy
                | Strategy::SupportGate060
                | Strategy::SupportGate090 => action.predicted_utility,
                Strategy::UncertaintyPenalty => {
                    action.predicted_utility - UNCERTAINTY_PENALTY * action.stated_uncertainty
                }
                Strategy::EnsembleMin => action.ensemble_min_utility,
                Strategy::OracleUpperBound => action.true_utility,
                Strategy::DirectFallback => FALLBACK_UTILITY,
            };
            if policy_score > best_policy_score {
                best_policy_score = policy_score;
                best_index = Some(index);
            }
        }
    }

    let (realized, optimism_gap, decision_score_gap, unsupported, coverage) = match best_index {
        Some(index) => {
            let action = &actions[index];
            (
                action.true_utility,
                action.predicted_utility - action.true_utility,
                best_policy_score - action.true_utility,
                (!action.supported) as u8 as f64,
                1.0,
            )
        }
        None => (FALLBACK_UTILITY, 0.0, 0.0, 0.0, 0.0),
    };

    let oracle_utility = actions
        .iter()
        .map(|action| action.true_utility)
        .fold(FALLBACK_UTILITY, f64::max);

    EpisodeMetrics {
        prediction_mae: prediction_mae(actions),
        realized_utility: realized,
        selected_optimism_gap: optimism_gap,
        selected_decision_score_gap: decision_score_gap,
        regret: (oracle_utility - realized).max(0.0),
        unsupported_selected: unsupported,
        coverage,
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Exploratory paired normal-approximation interval; episodes are the sampling unit.
/// Empirical CVaR95: mean of the worst ceil(5% * n) episode regrets.
/// Descriptive finite-panel tail risk; not a population-level confidence guarantee.
fn cvar95(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let tail_count = ((sorted.len() as f64 * 0.05).ceil() as usize).max(1);
    mean(&sorted[sorted.len() - tail_count..])
}

fn mean_ci95(values: &[f64]) -> (f64, f64, f64) {
    let average = mean(values);
    if values.len() < 2 {
        return (average, average, average);
    }
    let variance = values
        .iter()
        .map(|value| (value - average) * (value - average))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let half_width = 1.96 * (variance / values.len() as f64).sqrt();
    (average, average - half_width, average + half_width)
}

#[derive(Clone, Debug)]
struct Summary {
    search_size: usize,
    horizon: usize,
    rollout_growth: f64,
    strategy: &'static str,
    mean_prediction_mae: f64,
    mean_realized_utility: f64,
    mean_selected_optimism_gap: f64,
    mean_selected_decision_score_gap: f64,
    mean_regret: f64,
    cvar95_regret: f64,
    max_regret: f64,
    unsupported_selection_rate: f64,
    unsupported_rate_given_model_action: f64,
    model_action_coverage: f64,
    paired_regret_reduction_vs_greedy: f64,
    paired_regret_reduction_ci95_low: f64,
    paired_regret_reduction_ci95_high: f64,
}

fn summarize(
    search_size: usize,
    horizon: usize,
    rollout_growth: f64,
    strategy: Strategy,
    samples: &[EpisodeMetrics],
    greedy_samples: &[EpisodeMetrics],
) -> Summary {
    let regret_values = samples.iter().map(|row| row.regret).collect::<Vec<_>>();
    let optimism_values = samples
        .iter()
        .map(|row| row.selected_optimism_gap)
        .collect::<Vec<_>>();
    let unsupported_values = samples
        .iter()
        .map(|row| row.unsupported_selected)
        .collect::<Vec<_>>();
    let coverage_values = samples.iter().map(|row| row.coverage).collect::<Vec<_>>();
    let regret_reductions = greedy_samples
        .iter()
        .zip(samples)
        .map(|(greedy, candidate)| greedy.regret - candidate.regret)
        .collect::<Vec<_>>();
    let (paired_regret_reduction, paired_low, paired_high) = mean_ci95(&regret_reductions);
    let unsupported_count = unsupported_values.iter().sum::<f64>();
    let covered_count = coverage_values.iter().sum::<f64>();

    Summary {
        search_size,
        horizon,
        rollout_growth,
        strategy: strategy.name(),
        mean_prediction_mae: mean(&samples.iter().map(|row| row.prediction_mae).collect::<Vec<_>>()),
        mean_realized_utility: mean(&samples.iter().map(|row| row.realized_utility).collect::<Vec<_>>()),
        mean_selected_optimism_gap: mean(&optimism_values),
        mean_selected_decision_score_gap: mean(&samples.iter().map(|row| row.selected_decision_score_gap).collect::<Vec<_>>()),
        mean_regret: mean(&regret_values),
        cvar95_regret: cvar95(&regret_values),
        max_regret: regret_values.iter().copied().fold(0.0, f64::max),
        unsupported_selection_rate: mean(&unsupported_values),
        unsupported_rate_given_model_action: if covered_count == 0.0 {
            0.0
        } else {
            unsupported_count / covered_count
        },
        model_action_coverage: mean(&coverage_values),
        paired_regret_reduction_vs_greedy: paired_regret_reduction,
        paired_regret_reduction_ci95_low: paired_low,
        paired_regret_reduction_ci95_high: paired_high,
    }
}

fn run_benchmark() -> Vec<Summary> {
    let mut output = Vec::with_capacity(
        SEARCH_SIZES.len() * HORIZONS.len() * ROLLOUT_GROWTH.len() * Strategy::ALL.len(),
    );

    for search_size in SEARCH_SIZES {
        for horizon in HORIZONS {
            for rollout_growth in ROLLOUT_GROWTH {
                let mut by_strategy: Vec<Vec<EpisodeMetrics>> = Strategy::ALL
                    .iter()
                    .map(|_| Vec::with_capacity(SEEDS))
                    .collect();

                for episode in 0..SEEDS {
                    let seed = BASE_SEED
                        ^ (episode as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    let catalog = make_catalog(seed);
                    let candidates = project(&catalog[..search_size], horizon, rollout_growth);
                    for (index, strategy) in Strategy::ALL.iter().enumerate() {
                        by_strategy[index].push(evaluate_episode(&candidates, *strategy));
                    }
                }

                // PredictedGreedy is index 1 in Strategy::ALL; all methods share each
                // episode's catalog, truth and seed, so the regret comparison is paired.
                let greedy_samples = &by_strategy[1];
                for (index, strategy) in Strategy::ALL.iter().enumerate() {
                    output.push(summarize(
                        search_size,
                        horizon,
                        rollout_growth,
                        *strategy,
                        &by_strategy[index],
                        greedy_samples,
                    ));
                }
            }
        }
    }
    output
}

fn render_report(rows: &[Summary]) -> String {
    let mut output = format!(
        concat!(
            "{{\n",
            "  \"schema_version\": \"{}\",\n",
            "  \"seed_count\": {},\n",
            "  \"base_seed\": \"{:016x}\",\n",
            "  \"search_sizes\": [8, 32, 128, 512],\n",
            "  \"horizons\": [1, 4, 16],\n",
            "  \"rollout_growth_values\": [0.00, 0.20],\n",
            "  \"fallback_utility\": {:.8},\n",
            "  \"uncertainty_penalty\": {:.8},\n",
            "  \"summary\": [\n"
        ),
        REPORT_SCHEMA, SEEDS, BASE_SEED, FALLBACK_UTILITY, UNCERTAINTY_PENALTY
    );

    for (index, row) in rows.iter().enumerate() {
        output.push_str(&format!(
            concat!(
                "    {{",
                "\"search_size\": {}, ",
                "\"horizon\": {}, ",
                "\"rollout_growth\": {:.2}, ",
                "\"strategy\": \"{}\", ",
                "\"mean_prediction_mae\": {:.8}, ",
                "\"mean_realized_utility\": {:.8}, ",
                "\"mean_selected_optimism_gap\": {:.8}, ",
                "\"mean_selected_decision_score_gap\": {:.8}, ",
                "\"mean_regret\": {:.8}, ",
                "\"cvar95_regret\": {:.8}, ",
                "\"max_regret\": {:.8}, ",
                "\"unsupported_selection_rate\": {:.8}, "
                "\"unsupported_rate_given_model_action\": {:.8}, ",
                "\"model_action_coverage\": {:.8}, ",
                "\"paired_regret_reduction_vs_greedy\": {:.8}, ",
                "\"paired_regret_reduction_ci95_low\": {:.8}, ",
                "\"paired_regret_reduction_ci95_high\": {:.8}",
                "}}"
            ),
            row.search_size,
            row.horizon,
            row.rollout_growth,
            row.strategy,
            row.mean_prediction_mae,
            row.mean_realized_utility,
            row.mean_selected_optimism_gap,
            row.mean_selected_decision_score_gap,
            row.mean_regret,
            row.cvar95_regret,
            row.max_regret,
            row.unsupported_selection_rate,
            row.unsupported_rate_given_model_action,
            row.model_action_coverage,
            row.paired_regret_reduction_vs_greedy,
            row.paired_regret_reduction_ci95_low,
            row.paired_regret_reduction_ci95_high
        ));
        if index + 1 != rows.len() {
            output.push(',');
        }
        output.push('\n');
    }
    output.push_str("  ]\n}\n");
    output
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "CMAQ-007 horizon stress companion\n\nRuns a deterministic {}-seed grid over search sizes {:?}, horizons {:?}, and rollout growth {:?}. Prints JSON.",
            SEEDS, SEARCH_SIZES, HORIZONS, ROLLOUT_GROWTH
        );
        return;
    }
    if !args.is_empty() {
        eprintln!("unexpected argument {}; use --help", args[0]);
        process::exit(2);
    }
    print!("{}", render_report(&run_benchmark()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_and_projection_are_reproducible() {
        let first = make_catalog(7);
        let second = make_catalog(7);
        assert_eq!(first, second);
        assert_eq!(project(&first, 16, 0.20), project(&second, 16, 0.20));
    }

    #[test]
    fn zero_growth_preserves_persistent_per_step_error_at_every_horizon() {
        let catalog = make_catalog(BASE_SEED);
        for horizon in HORIZONS {
            let projected = project(&catalog, horizon, 0.0);
            for (truth, candidate) in catalog.iter().zip(projected) {
                assert!((candidate.predicted_utility - truth.true_utility - truth.persistent_error).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn positive_growth_amplifies_persistent_errors_with_horizon() {
        assert!(amplification_ratio(16, 0.20) > amplification_ratio(4, 0.20));
        assert!(amplification_ratio(4, 0.20) > amplification_ratio(1, 0.20));
        assert_eq!(amplification_ratio(16, 0.0), 1.0);
    }

    #[test]
    fn oracle_has_zero_regret_and_direct_fallback_has_zero_coverage() {
        for horizon in HORIZONS {
            for growth in ROLLOUT_GROWTH {
                let catalog = project(&make_catalog(BASE_SEED), horizon, growth);
                let oracle = evaluate_episode(&catalog[..128], Strategy::OracleUpperBound);
                let fallback = evaluate_episode(&catalog[..128], Strategy::DirectFallback);
                assert!(oracle.regret.abs() < 1e-12);
                assert_eq!(fallback.coverage, 0.0);
                assert_eq!(fallback.unsupported_selected, 0.0);
                assert_eq!(fallback.selected_optimism_gap, 0.0);
            }
        }
    }

    #[test]
    fn strict_support_gate_never_has_more_coverage_than_lenient_gate() {
        for seed in 1..25 {
            for horizon in HORIZONS {
                let catalog = project(&make_catalog(seed), horizon, 0.20);
                let lenient = evaluate_episode(&catalog[..512], Strategy::SupportGate060);
                let strict = evaluate_episode(&catalog[..512], Strategy::SupportGate090);
                assert!(strict.coverage <= lenient.coverage);
            }
        }
    }

    #[test]
    fn selected_policy_error_rises_with_rollout_growth_and_breadth() {
        let report = run_benchmark();
        let get = |size: usize, horizon: usize, growth: f64, strategy: &str| {
            report.iter().find(|row| {
                row.search_size == size
                    && row.horizon == horizon
                    && (row.rollout_growth - growth).abs() < 1e-12
                    && row.strategy == strategy
            }).expect("registered summary row")
        };
        assert!(
            get(512, 16, 0.20, "predicted_greedy").unsupported_selection_rate
                > get(8, 16, 0.20, "predicted_greedy").unsupported_selection_rate
        );
        assert!(
            get(512, 16, 0.20, "predicted_greedy").mean_selected_optimism_gap
                > get(512, 1, 0.20, "predicted_greedy").mean_selected_optimism_gap
        );
        assert!(
            get(512, 16, 0.20, "predicted_greedy").mean_prediction_mae
                > get(512, 16, 0.0, "predicted_greedy").mean_prediction_mae
        );
    }

    #[test]
    fn report_shape_and_metrics_are_finite_and_bounded_where_required() {
        let report = run_benchmark();
        assert_eq!(report.len(), 4 * 3 * 2 * Strategy::ALL.len());
        for row in report {
            assert!(row.mean_prediction_mae.is_finite() && row.mean_prediction_mae >= 0.0);
            assert!(row.mean_realized_utility.is_finite());
            assert!(row.mean_selected_optimism_gap.is_finite());
            assert!(row.mean_selected_decision_score_gap.is_finite());
            assert!(row.mean_regret.is_finite() && row.mean_regret >= 0.0);
            assert!(row.cvar95_regret.is_finite() && row.cvar95_regret >= row.mean_regret - 1e-12);
            assert!(row.max_regret.is_finite() && row.max_regret >= row.cvar95_regret - 1e-12);
            assert!((0.0..=1.0).contains(&row.unsupported_selection_rate));
            assert!((0.0..=1.0).contains(&row.unsupported_rate_given_model_action));
            assert!((0.0..=1.0).contains(&row.model_action_coverage));
            assert!(row.paired_regret_reduction_ci95_low <= row.paired_regret_reduction_ci95_high);
        }
    }

    #[test]
    fn empirical_cvar95_uses_worst_five_percent_and_dominates_mean() {
        let values = (0..100).map(|value| value as f64).collect::<Vec<_>>();
        assert!((cvar95(&values) - 97.0).abs() < 1e-12);
        assert!(cvar95(&values) >= mean(&values));
        assert_eq!(cvar95(&[]), 0.0);
    }

    #[test]
    fn identical_policy_samples_have_zero_paired_interval() {
        let values = [0.0_f64; 32];
        assert_eq!(mean_ci95(&values), (0.0, 0.0, 0.0));
    }
}
