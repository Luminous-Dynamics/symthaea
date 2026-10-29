use std::process;

const REPORT_SCHEMA: &str = "cmaq-007-exploitability-report-v0.1";
const EPISODES: usize = 256;
const SEARCH_SIZES: [usize; 4] = [8, 32, 128, 512];
const MAX_ACTIONS: usize = 512;
const BASELINE_UTILITY: f64 = 0.35;
const BASE_SEED: u64 = 0xC0A7_0007_2026_1029;

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
}

#[derive(Clone, Debug, PartialEq)]
struct CandidateAction {
    true_utility: f64,
    predicted_utility: f64,
    predicted_uncertainty: f64,
    support_score: f64,
    supported: bool,
}

fn candidate_catalog(seed: u64) -> Vec<CandidateAction> {
    let mut rng = SplitMix64::new(seed);
    (0..MAX_ACTIONS)
        .map(|index| {
            // Four-percent structural blind spots in expectation. Their model
            // predictions are systematically optimistic and understate uncertainty.
            let unsupported = index > 0 && rng.next_u64() % 25 == 0;
            let utility_draw = rng.unit();
            let true_utility = if unsupported {
                0.05 + 0.25 * utility_draw
            } else {
                0.10 + 0.45 * utility_draw
            };
            let model_noise = (rng.unit() - 0.5) * 0.12;
            let structural_bias = if unsupported { 0.32 } else { 0.0 };
            let predicted_utility = true_utility + structural_bias + model_noise;
            let predicted_uncertainty = if unsupported {
                0.01 + 0.01 * rng.unit()
            } else {
                0.04 + 0.025 * rng.unit()
            };
            // Intentionally imperfect support confidence: supported and
            // unsupported score distributions overlap, so thresholds trade
            // off coverage against selected-action risk.
            let support_score = if unsupported {
                0.15 + 0.85 * rng.unit()
            } else {
                0.55 + 0.45 * rng.unit()
            };
            CandidateAction {
                true_utility,
                predicted_utility,
                predicted_uncertainty,
                support_score,
                supported: !unsupported,
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Strategy {
    DirectFallback,
    PredictedGreedy,
    UncertaintyPenalty,
    SupportGate(f64),
    Oracle,
}

impl Strategy {
    fn all() -> [Self; 9] {
        [
            Self::DirectFallback,
            Self::PredictedGreedy,
            Self::UncertaintyPenalty,
            Self::SupportGate(0.40),
            Self::SupportGate(0.55),
            Self::SupportGate(0.75),
            Self::SupportGate(0.90),
            Self::SupportGate(0.97),
            Self::Oracle,
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::DirectFallback => "direct_fallback",
            Self::PredictedGreedy => "predicted_greedy",
            Self::UncertaintyPenalty => "uncertainty_penalty",
            Self::SupportGate(threshold) if threshold == 0.40 => "support_gate_040",
            Self::SupportGate(threshold) if threshold == 0.55 => "support_gate_055",
            Self::SupportGate(threshold) if threshold == 0.75 => "support_gate_075",
            Self::SupportGate(threshold) if threshold == 0.90 => "support_gate_090",
            Self::SupportGate(threshold) if threshold == 0.97 => "support_gate_097",
            Self::SupportGate(_) => "support_gate_unknown",
            Self::Oracle => "oracle_upper_bound",
        }
    }
}

fn select_candidate(actions: &[CandidateAction], strategy: Strategy) -> Option<usize> {
    match strategy {
        Strategy::DirectFallback => None,
        Strategy::Oracle => {
            let best = actions
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.true_utility.total_cmp(&right.true_utility));
            best.and_then(|(index, action)| {
                (action.true_utility > BASELINE_UTILITY).then_some(index)
            })
        }
        Strategy::PredictedGreedy => best_index(actions, |action| action.predicted_utility),
        Strategy::UncertaintyPenalty => best_index(actions, |action| {
            action.predicted_utility - 2.0 * action.predicted_uncertainty
        }),
        Strategy::SupportGate(threshold) => {
            let eligible: Vec<(usize, &CandidateAction)> = actions
                .iter()
                .enumerate()
                .filter(|(_, action)| action.support_score >= threshold)
                .collect();
            eligible
                .into_iter()
                .max_by(|(_, left), (_, right)| {
                    left.predicted_utility.total_cmp(&right.predicted_utility)
                })
                .map(|(index, _)| index)
        }
    }
}

fn best_index<F>(actions: &[CandidateAction], score: F) -> Option<usize>
where
    F: Fn(&CandidateAction) -> f64,
{
    let mut best: Option<(usize, f64)> = None;
    for (index, action) in actions.iter().enumerate() {
        let candidate_score = score(action);
        if best
            .map(|(_, best_score)| candidate_score > best_score)
            .unwrap_or(true)
        {
            best = Some((index, candidate_score));
        }
    }
    best.map(|(index, _)| index)
}

#[derive(Clone, Copy, Debug, Default)]
struct EpisodeMetrics {
    prediction_mae: f64,
    realized_utility: f64,
    selected_optimism_gap: f64,
    regret: f64,
    unsupported_selected: f64,
    coverage: f64,
}

fn evaluate_episode(actions: &[CandidateAction], strategy: Strategy) -> EpisodeMetrics {
    let prediction_mae = actions
        .iter()
        .map(|action| (action.predicted_utility - action.true_utility).abs())
        .sum::<f64>()
        / actions.len() as f64;
    let oracle_utility = actions
        .iter()
        .map(|action| action.true_utility)
        .fold(BASELINE_UTILITY, f64::max);

    let selected = select_candidate(actions, strategy);
    let (realized_utility, selected_optimism_gap, unsupported_selected, coverage) =
        if let Some(index) = selected {
            let action = &actions[index];
            (
                action.true_utility,
                action.predicted_utility - action.true_utility,
                f64::from(!action.supported),
                1.0,
            )
        } else {
            (BASELINE_UTILITY, 0.0, 0.0, 0.0)
        };

    EpisodeMetrics {
        prediction_mae,
        realized_utility,
        selected_optimism_gap,
        regret: (oracle_utility - realized_utility).max(0.0),
        unsupported_selected,
        coverage,
    }
}

#[derive(Clone, Debug)]
struct Summary {
    search_size: usize,
    strategy: &'static str,
    mean_prediction_mae: f64,
    mean_realized_utility: f64,
    mean_selected_optimism_gap: f64,
    mean_regret: f64,
    unsupported_selection_rate: f64,
    coverage_rate: f64,
    paired_regret_reduction_vs_greedy: f64,
    paired_regret_reduction_ci95_low: f64,
    paired_regret_reduction_ci95_high: f64,
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn mean_ci95(values: &[f64]) -> (f64, f64, f64) {
    let average = mean(values);
    if values.len() < 2 {
        return (average, average, average);
    }
    let sum_sq = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>();
    let sample_variance = sum_sq / (values.len() - 1) as f64;
    let standard_error = (sample_variance / values.len() as f64).sqrt();
    (average, average - 1.96 * standard_error, average + 1.96 * standard_error)
}

fn summarize(
    search_size: usize,
    strategy: Strategy,
    samples: &[EpisodeMetrics],
    greedy_samples: &[EpisodeMetrics],
) -> Summary {
    let maes: Vec<f64> = samples.iter().map(|sample| sample.prediction_mae).collect();
    let utilities: Vec<f64> = samples.iter().map(|sample| sample.realized_utility).collect();
    let optimism: Vec<f64> = samples.iter().map(|sample| sample.selected_optimism_gap).collect();
    let regrets: Vec<f64> = samples.iter().map(|sample| sample.regret).collect();
    let unsupported: Vec<f64> = samples.iter().map(|sample| sample.unsupported_selected).collect();
    let coverage: Vec<f64> = samples.iter().map(|sample| sample.coverage).collect();
    let paired_reductions: Vec<f64> = greedy_samples
        .iter()
        .zip(samples)
        .map(|(greedy, current)| greedy.regret - current.regret)
        .collect();
    let (paired_mean, ci_low, ci_high) = mean_ci95(&paired_reductions);

    Summary {
        search_size,
        strategy: strategy.name(),
        mean_prediction_mae: mean(&maes),
        mean_realized_utility: mean(&utilities),
        mean_selected_optimism_gap: mean(&optimism),
        mean_regret: mean(&regrets),
        unsupported_selection_rate: mean(&unsupported),
        coverage_rate: mean(&coverage),
        paired_regret_reduction_vs_greedy: paired_mean,
        paired_regret_reduction_ci95_low: ci_low,
        paired_regret_reduction_ci95_high: ci_high,
    }
}

fn run_benchmark() -> Vec<Summary> {
    let strategies = Strategy::all();
    let mut summaries = Vec::with_capacity(SEARCH_SIZES.len() * strategies.len());

    for search_size in SEARCH_SIZES {
        let mut samples_by_strategy: Vec<(Strategy, Vec<EpisodeMetrics>)> = strategies
            .iter()
            .copied()
            .map(|strategy| (strategy, Vec::with_capacity(EPISODES)))
            .collect();

        for episode in 0..EPISODES {
            // All search sizes use prefixes of the same per-episode 512-action
            // catalogue; methods are paired on the same candidates and truths.
            let seed = BASE_SEED
                ^ (episode as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let catalog = candidate_catalog(seed);
            let actions = &catalog[..search_size];
            for (strategy, samples) in &mut samples_by_strategy {
                samples.push(evaluate_episode(actions, *strategy));
            }
        }

        let greedy_samples = &samples_by_strategy
            .iter()
            .find(|(strategy, _)| *strategy == Strategy::PredictedGreedy)
            .expect("greedy baseline is registered")
            .1;

        for (strategy, samples) in &samples_by_strategy {
            summaries.push(summarize(search_size, *strategy, samples, greedy_samples));
        }
    }

    summaries
}

fn render_report(summaries: &[Summary]) -> String {
    let mut output = String::new();
    output.push_str("{\n");
    output.push_str(&format!("  \"schema_version\": \"{REPORT_SCHEMA}\",\n"));
    output.push_str(&format!("  \"seed_count\": {EPISODES},\n"));
    output.push_str(&format!("  \"base_seed\": \"{BASE_SEED:016x}\",\n"));
    output.push_str("  \"search_sizes\": [8, 32, 128, 512],\n");
    output.push_str("  \"summary\": [\n");
    for (index, row) in summaries.iter().enumerate() {
        output.push_str(&format!(
            concat!(
                "    {{\"search_size\": {}, \"strategy\": \"{}\", ",
                "\"mean_prediction_mae\": {:.8}, ",
                "\"mean_realized_utility\": {:.8}, ",
                "\"mean_selected_optimism_gap\": {:.8}, ",
                "\"mean_regret\": {:.8}, ",
                "\"unsupported_selection_rate\": {:.8}, ",
                "\"coverage_rate\": {:.8}, ",
                "\"paired_regret_reduction_vs_greedy\": {:.8}, ",
                "\"paired_regret_reduction_ci95_low\": {:.8}, ",
                "\"paired_regret_reduction_ci95_high\": {:.8}}}"
            ),
            row.search_size,
            row.strategy,
            row.mean_prediction_mae,
            row.mean_realized_utility,
            row.mean_selected_optimism_gap,
            row.mean_regret,
            row.unsupported_selection_rate,
            row.coverage_rate,
            row.paired_regret_reduction_vs_greedy,
            row.paired_regret_reduction_ci95_low,
            row.paired_regret_reduction_ci95_high
        ));
        if index + 1 != summaries.len() {
            output.push(',');
        }
        output.push('\n');
    }
    output.push_str("  ]\n}\n");
    output
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|argument| argument == "--help" || argument == "-h") {
        println!("CMAQ-007 planner exploitability smoke benchmark\n\nRuns a deterministic 256-seed panel over search sizes 8, 32, 128 and 512.\nPrints one JSON report to stdout.");
        return;
    }
    if !args.is_empty() {
        eprintln!("unexpected argument: {}; use --help", args[0]);
        process::exit(2);
    }
    let report = render_report(&run_benchmark());
    print!("{report}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_seed_catalog_is_reproducible() {
        assert_eq!(candidate_catalog(7), candidate_catalog(7));
    }

    #[test]
    fn catalog_contains_structural_blind_spots_and_supported_actions() {
        let catalog = candidate_catalog(BASE_SEED);
        assert!(catalog.iter().any(|action| action.supported));
        assert!(catalog.iter().any(|action| !action.supported));
    }

    #[test]
    fn oracle_regret_is_zero_across_all_smoke_sizes() {
        for episode in 0..32 {
            let seed = BASE_SEED ^ (episode as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let catalog = candidate_catalog(seed);
            for size in SEARCH_SIZES {
                let metrics = evaluate_episode(&catalog[..size], Strategy::Oracle);
                assert!(metrics.regret.abs() < 1e-12);
            }
        }
    }

    #[test]
    fn direct_fallback_never_claims_model_coverage_or_unsupported_selection() {
        let catalog = candidate_catalog(BASE_SEED);
        let metrics = evaluate_episode(&catalog[..128], Strategy::DirectFallback);
        assert_eq!(metrics.coverage, 0.0);
        assert_eq!(metrics.unsupported_selected, 0.0);
        assert_eq!(metrics.selected_optimism_gap, 0.0);
    }

    #[test]
    fn support_confidence_is_not_an_oracle_label() {
        let catalog = candidate_catalog(BASE_SEED);
        assert!(catalog.iter().any(|action| action.supported && action.support_score < 0.8));
        assert!(catalog.iter().any(|action| !action.supported && action.support_score > 0.8));
    }

    #[test]
    fn regret_and_optimism_are_finite_for_all_baselines() {
        let catalog = candidate_catalog(BASE_SEED);
        for strategy in Strategy::all() {
            let metrics = evaluate_episode(&catalog[..128], strategy);
            assert!(metrics.regret.is_finite() && metrics.regret >= 0.0);
            assert!(metrics.selected_optimism_gap.is_finite());
            assert!((0.0..=1.0).contains(&metrics.coverage));
        }
    }

    #[test]
    fn paired_interval_collapses_for_identical_strategy() {
        let values = [0.0_f64; 16];
        let (average, low, high) = mean_ci95(&values);
        assert_eq!((average, low, high), (0.0, 0.0, 0.0));
    }
}
