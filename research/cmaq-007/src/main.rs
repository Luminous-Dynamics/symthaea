use std::process;

const REPORT_SCHEMA: &str = "cmaq-007-exploitability-report-v0.3";
const EPISODES: usize = 256;
const SEARCH_SIZES: [usize; 4] = [8, 32, 128, 512];
const MAX_ACTIONS: usize = 512;
const BASELINE_UTILITY: f64 = 0.35;
const BASE_SEED: u64 = 0xC0A7_0007_2026_1029;
const CALIBRATION_SEED: u64 = 0xCA11_BA7E_2026_1029;
const QUALIFICATION_SEED: u64 = 0x0A11_F1ED_2026_1029;
const RISK_THRESHOLDS: [f64; 6] = [0.40, 0.55, 0.75, 0.90, 0.97, 1.01];
const UNSUPPORTED_RISK_TARGET: f64 = 0.05;
const FAMILYWISE_ALPHA: f64 = 0.05;


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
    shuffled_support_score: f64,
    supported: bool,
}

fn candidate_catalog(seed: u64) -> Vec<CandidateAction> {
    let mut rng = SplitMix64::new(seed);
    let mut shuffled_score_rng = SplitMix64::new(seed ^ 0xC07A_7EED_55AA_0011);
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
                shuffled_support_score: shuffled_score_rng.unit(),
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
    ShuffledSupportGate(f64),
    RiskControlledGate(f64),
    Oracle,
}

impl Strategy {
    fn all() -> [Self; 11] {
        [
            Self::DirectFallback,
            Self::PredictedGreedy,
            Self::UncertaintyPenalty,
            Self::SupportGate(0.40),
            Self::SupportGate(0.55),
            Self::SupportGate(0.75),
            Self::SupportGate(0.90),
            Self::SupportGate(0.97),
            Self::ShuffledSupportGate(0.55),
            Self::ShuffledSupportGate(0.97),
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
            Self::ShuffledSupportGate(threshold) if threshold == 0.55 => "shuffled_support_gate_055",
            Self::ShuffledSupportGate(threshold) if threshold == 0.97 => "shuffled_support_gate_097",
            Self::ShuffledSupportGate(_) => "shuffled_support_gate_unknown",
            Self::RiskControlledGate(_) => "risk_controlled_gate",
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
        Strategy::SupportGate(threshold) => best_gate(actions, threshold, false),
        Strategy::ShuffledSupportGate(threshold) => best_gate(actions, threshold, true),
        Strategy::RiskControlledGate(threshold) => best_risk_controlled_gate(actions, threshold),
    }
}

fn best_risk_controlled_gate(actions: &[CandidateAction], threshold: f64) -> Option<usize> {
    actions
        .iter()
        .enumerate()
        .filter(|(_, action)| {
            action.support_score >= threshold && action.predicted_utility > BASELINE_UTILITY
        })
        .max_by(|(_, left), (_, right)| left.predicted_utility.total_cmp(&right.predicted_utility))
        .map(|(index, _)| index)
}

fn best_gate(actions: &[CandidateAction], threshold: f64, shuffled: bool) -> Option<usize> {
    actions
        .iter()
        .enumerate()
        .filter(|(_, action)| {
            let score = if shuffled { action.shuffled_support_score } else { action.support_score };
            score >= threshold
        })
        .max_by(|(_, left), (_, right)| left.predicted_utility.total_cmp(&right.predicted_utility))
        .map(|(index, _)| index)
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
                if action.supported { 0.0 } else { 1.0 },
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

#[derive(Clone, Debug)]
struct RiskControlSummary {
    search_size: usize,
    target_unsupported_risk: f64,
    familywise_alpha: f64,
    alpha_per_bound: f64,
    threshold_count: usize,
    bound_count: usize,
    selected_threshold: f64,
    calibration_episode_count: usize,
    calibration_unsupported_count: usize,
    calibration_unsupported_rate: f64,
    calibration_unsupported_upper_bound: f64,
    calibration_downside_count: usize,
    calibration_downside_rate: f64,
    calibration_downside_upper_bound: f64,
    calibration_coverage_rate: f64,
    qualification_episode_count: usize,
    qualification_unsupported_count: usize,
    qualification_unsupported_rate: f64,
    qualification_downside_count: usize,
    qualification_downside_rate: f64,
    qualification_coverage_rate: f64,
    qualification_mean_regret: f64,
    direct_fallback_mean_regret: f64,
}

fn log_factorials(n: usize) -> Vec<f64> {
    let mut values = vec![0.0; n + 1];
    for value in 1..=n {
        values[value] = values[value - 1] + (value as f64).ln();
    }
    values
}

fn binomial_cdf_at_most(k: usize, n: usize, p: f64, log_factorial: &[f64]) -> f64 {
    if k >= n {
        return 1.0;
    }
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }

    let log_p = p.ln();
    let log_q = (1.0 - p).ln();
    let log_weights: Vec<f64> = (0..=k)
        .map(|i| {
            log_factorial[n] - log_factorial[i] - log_factorial[n - i]
                + (i as f64) * log_p
                + ((n - i) as f64) * log_q
        })
        .collect();
    let largest = log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scaled_sum = log_weights
        .iter()
        .map(|value| (value - largest).exp())
        .sum::<f64>();
    (largest + scaled_sum.ln()).exp().clamp(0.0, 1.0)
}

// One-sided Clopper-Pearson upper confidence bound for a binomial event rate.
// The caller applies Bonferroni correction across the preregistered threshold grid.
fn clopper_pearson_upper(k: usize, n: usize, alpha: f64) -> f64 {
    assert!(n > 0, "binomial interval requires nonzero sample size");
    assert!(alpha > 0.0 && alpha < 1.0, "alpha must lie in (0, 1)");
    if k >= n {
        return 1.0;
    }
    if k == 0 {
        return 1.0 - alpha.powf(1.0 / n as f64);
    }

    let log_factorial = log_factorials(n);
    let mut lower = k as f64 / n as f64;
    let mut upper = 1.0;
    for _ in 0..64 {
        let midpoint = 0.5 * (lower + upper);
        if binomial_cdf_at_most(k, n, midpoint, &log_factorial) > alpha {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    upper
}

fn risk_control_summaries() -> Vec<RiskControlSummary> {
    // Bonferroni covers both risk constraints across all thresholds.
    let alpha_per_bound = FAMILYWISE_ALPHA / (RISK_THRESHOLDS.len() * 2) as f64;
    let mut output = Vec::with_capacity(SEARCH_SIZES.len());

    for search_size in SEARCH_SIZES {
        let mut calibration_by_threshold: Vec<Vec<EpisodeMetrics>> = RISK_THRESHOLDS
            .iter()
            .map(|_| Vec::with_capacity(EPISODES))
            .collect();

        // This seed stream is separate from both the descriptive panel and
        // the held-out qualification stream.
        for episode in 0..EPISODES {
            let seed = CALIBRATION_SEED
                ^ (episode as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let catalog = candidate_catalog(seed);
            let actions = &catalog[..search_size];
            for (threshold_index, threshold) in RISK_THRESHOLDS.iter().enumerate() {
                calibration_by_threshold[threshold_index]
                    .push(evaluate_episode(actions, Strategy::RiskControlledGate(*threshold)));
            }
        }

        let threshold_stats: Vec<(usize, usize, f64, f64, usize, f64, f64, f64)> = calibration_by_threshold
            .iter()
            .enumerate()
            .map(|(index, samples)| {
                let unsupported_count = samples
                    .iter()
                    .filter(|sample| sample.unsupported_selected > 0.0)
                    .count();
                let unsupported_rate = unsupported_count as f64 / EPISODES as f64;
                let unsupported_upper_bound = clopper_pearson_upper(
                    unsupported_count,
                    EPISODES,
                    alpha_per_bound,
                );
                let downside_count = samples
                    .iter()
                    .filter(|sample| sample.coverage > 0.0 && sample.realized_utility <= BASELINE_UTILITY)
                    .count();
                let downside_rate = downside_count as f64 / EPISODES as f64;
                let downside_upper_bound = clopper_pearson_upper(
                    downside_count,
                    EPISODES,
                    alpha_per_bound,
                );
                let coverage_rate = samples.iter().map(|sample| sample.coverage).sum::<f64>()
                    / EPISODES as f64;
                (index, unsupported_count, unsupported_rate, unsupported_upper_bound,
                    downside_count, downside_rate, downside_upper_bound, coverage_rate)
            })
            .collect();

        // Select the highest-coverage threshold whose simultaneous upper
        // confidence bound clears the target; ties prefer the stricter gate.
        let mut selected: Option<(usize, f64)> = None;
        for (index, _, _, unsupported_upper_bound, _, _, downside_upper_bound, coverage_rate) in &threshold_stats {
            if *unsupported_upper_bound > UNSUPPORTED_RISK_TARGET
                || *downside_upper_bound > UNSUPPORTED_RISK_TARGET
            {
                continue;
            }
            let prefer = selected
                .map(|(best_index, best_coverage)| {
                    *coverage_rate > best_coverage
                        || (*coverage_rate == best_coverage
                            && RISK_THRESHOLDS[*index] > RISK_THRESHOLDS[best_index])
                })
                .unwrap_or(true);
            if prefer {
                selected = Some((*index, *coverage_rate));
            }
        }
        let selected_index = selected
            .expect("the 1.01 threshold must force fallback and meet both risk targets")
            .0;
        let selected_threshold = RISK_THRESHOLDS[selected_index];
        let (_, calibration_unsupported_count, calibration_unsupported_rate,
            calibration_unsupported_upper_bound, calibration_downside_count,
            calibration_downside_rate, calibration_downside_upper_bound,
            calibration_coverage_rate) = threshold_stats[selected_index];

        let mut qualification_samples = Vec::with_capacity(EPISODES);
        let mut fallback_samples = Vec::with_capacity(EPISODES);
        for episode in 0..EPISODES {
            let seed = QUALIFICATION_SEED
                ^ (episode as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let catalog = candidate_catalog(seed);
            let actions = &catalog[..search_size];
            qualification_samples.push(evaluate_episode(
                actions,
                Strategy::RiskControlledGate(selected_threshold),
            ));
            fallback_samples.push(evaluate_episode(actions, Strategy::DirectFallback));
        }

        let qualification_unsupported_count = qualification_samples
            .iter()
            .filter(|sample| sample.unsupported_selected > 0.0)
            .count();
        let qualification_unsupported_rate =
            qualification_unsupported_count as f64 / EPISODES as f64;
        let qualification_downside_count = qualification_samples
            .iter()
            .filter(|sample| sample.coverage > 0.0 && sample.realized_utility <= BASELINE_UTILITY)
            .count();
        let qualification_downside_rate = qualification_downside_count as f64 / EPISODES as f64;
        let qualification_coverage_rate = qualification_samples
            .iter()
            .map(|sample| sample.coverage)
            .sum::<f64>()
            / EPISODES as f64;
        let qualification_mean_regret = qualification_samples
            .iter()
            .map(|sample| sample.regret)
            .sum::<f64>()
            / EPISODES as f64;
        let direct_fallback_mean_regret = fallback_samples
            .iter()
            .map(|sample| sample.regret)
            .sum::<f64>()
            / EPISODES as f64;

        output.push(RiskControlSummary {
            search_size,
            target_unsupported_risk: UNSUPPORTED_RISK_TARGET,
            familywise_alpha: FAMILYWISE_ALPHA,
            alpha_per_bound,
            threshold_count: RISK_THRESHOLDS.len(),
            bound_count: RISK_THRESHOLDS.len() * 2,
            selected_threshold,
            calibration_episode_count: EPISODES,
            calibration_unsupported_count,
            calibration_unsupported_rate,
            calibration_unsupported_upper_bound,
            calibration_downside_count,
            calibration_downside_rate,
            calibration_downside_upper_bound,
            calibration_coverage_rate,
            qualification_episode_count: EPISODES,
            qualification_unsupported_count,
            qualification_unsupported_rate,
            qualification_downside_count,
            qualification_downside_rate,
            qualification_coverage_rate,
            qualification_mean_regret,
            direct_fallback_mean_regret,
        });
    }

    output
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

fn render_report(summaries: &[Summary], risk_control: &[RiskControlSummary]) -> String {
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
    output.push_str("  ],\n");
    output.push_str("  \"risk_control\": [\n");
    for (index, row) in risk_control.iter().enumerate() {
        output.push_str(&format!(
            concat!(
                "    {{\"search_size\": {}, ",
                "\"target_unsupported_risk\": {:.6}, ",
                "\"familywise_alpha\": {:.6}, ",
                "\"alpha_per_bound\": {:.8}, ",
                "\"threshold_count\": {}, ",
                "\"bound_count\": {}, ",
                "\"selected_threshold\": {:.4}, ",
                "\"calibration_episode_count\": {}, ",
                "\"calibration_unsupported_count\": {}, ",
                "\"calibration_unsupported_rate\": {:.8}, ",
                "\"calibration_unsupported_upper_bound\": {:.8}, ",
                "\"calibration_downside_count\": {}, ",
                "\"calibration_downside_rate\": {:.8}, ",
                "\"calibration_downside_upper_bound\": {:.8}, ",
                "\"calibration_coverage_rate\": {:.8}, ",
                "\"qualification_episode_count\": {}, ",
                "\"qualification_unsupported_count\": {}, ",
                "\"qualification_unsupported_rate\": {:.8}, ",
                "\"qualification_downside_count\": {}, ",
                "\"qualification_downside_rate\": {:.8}, ",
                "\"qualification_coverage_rate\": {:.8}, ",
                "\"qualification_mean_regret\": {:.8}, ",
                "\"direct_fallback_mean_regret\": {:.8}}}"
            ),
            row.search_size,
            row.target_unsupported_risk,
            row.familywise_alpha,
            row.alpha_per_bound,
            row.threshold_count,
            row.bound_count,
            row.selected_threshold,
            row.calibration_episode_count,
            row.calibration_unsupported_count,
            row.calibration_unsupported_rate,
            row.calibration_unsupported_upper_bound,
            row.calibration_downside_count,
            row.calibration_downside_rate,
            row.calibration_downside_upper_bound,
            row.calibration_coverage_rate,
            row.qualification_episode_count,
            row.qualification_unsupported_count,
            row.qualification_unsupported_rate,
            row.qualification_downside_count,
            row.qualification_downside_rate,
            row.qualification_coverage_rate,
            row.qualification_mean_regret,
            row.direct_fallback_mean_regret
        ));
        if index + 1 != risk_control.len() {
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
    let summaries = run_benchmark();
    let risk_control = risk_control_summaries();
    let report = render_report(&summaries, &risk_control);
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
    fn shuffled_support_scores_are_not_identical_to_true_support() {
        let catalog = candidate_catalog(BASE_SEED);
        assert!(catalog.iter().any(|action| !action.supported && action.shuffled_support_score >= 0.55));
        assert!(catalog.iter().any(|action| action.supported && action.shuffled_support_score < 0.55));
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
    fn clopper_pearson_bound_handles_zero_and_all_failures() {
        let alpha = FAMILYWISE_ALPHA / RISK_THRESHOLDS.len() as f64;
        let zero_upper = clopper_pearson_upper(0, EPISODES, alpha);
        assert!(zero_upper > 0.0 && zero_upper < UNSUPPORTED_RISK_TARGET);
        assert!((zero_upper - (1.0 - alpha.powf(1.0 / EPISODES as f64))).abs() < 1e-12);
        assert_eq!(clopper_pearson_upper(EPISODES, EPISODES, alpha), 1.0);
        assert!(clopper_pearson_upper(12, EPISODES, alpha) > 12.0 / EPISODES as f64);
    }

    #[test]
    fn calibrated_gate_uses_disjoint_panel_and_respects_calibration_risk_target() {
        let rows = risk_control_summaries();
        assert_eq!(rows.len(), SEARCH_SIZES.len());
        for row in rows {
            assert_eq!(row.calibration_episode_count, EPISODES);
            assert_eq!(row.qualification_episode_count, EPISODES);
            assert!(RISK_THRESHOLDS.contains(&row.selected_threshold));
            assert!(row.calibration_unsupported_upper_bound <= UNSUPPORTED_RISK_TARGET);
            assert!(row.calibration_downside_upper_bound <= UNSUPPORTED_RISK_TARGET);
            assert!((0.0..=1.0).contains(&row.qualification_unsupported_rate));
            assert!((0.0..=1.0).contains(&row.qualification_downside_rate));
            assert!((0.0..=1.0).contains(&row.qualification_coverage_rate));
        }
    }

    #[test]
    fn paired_interval_collapses_for_identical_strategy() {
        let values = [0.0_f64; 16];
        let (average, low, high) = mean_ci95(&values);
        assert_eq!((average, low, high), (0.0, 0.0, 0.0));
    }
}
