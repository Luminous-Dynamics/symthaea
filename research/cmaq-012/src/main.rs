use std::fmt::Write as _;

const SCHEMA: &str = "cmaq-012-joint-decision-calibration-v0.1";
const CALIBRATION_EPISODES: usize = 2_048;
const QUALIFICATION_EPISODES: usize = 2_048;
const CANDIDATE_COUNTS: [usize; 4] = [8, 16, 32, 64];
const TARGET_COVERAGE: f64 = 0.90;
const FALLBACK_VALUE: f64 = 0.42;
const TOTAL_VARIANCE_Z: f64 = 1.644_853_626_951_472_2;
const REPORTED_EPISTEMIC_STD: f64 = 0.04;
const BASE_SEED: u64 = 0xC0A9_0012_2026_1030;
const CALIBRATION_SEED: u64 = 0xCA11_BA7E_0120_0001;
const QUALIFICATION_SEED: u64 = 0x51A7_E000_0120_0002;
const ENSEMBLE_HEADS: usize = 5;

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

    fn centered_normal_approx(&mut self) -> f64 {
        (self.unit() - self.unit()) * 6.0_f64.sqrt()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    InDistribution,
    EpistemicScaleShift,
    AleatoricNoiseShift,
    OptimisticFailureCandidates,
}

impl Scenario {
    fn all() -> [Self; 4] {
        [
            Self::InDistribution,
            Self::EpistemicScaleShift,
            Self::AleatoricNoiseShift,
            Self::OptimisticFailureCandidates,
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::InDistribution => "in_distribution",
            Self::EpistemicScaleShift => "epistemic_scale_shift",
            Self::AleatoricNoiseShift => "aleatoric_noise_shift",
            Self::OptimisticFailureCandidates => "optimistic_failure_candidates",
        }
    }

    fn actual_epistemic_std(self) -> f64 {
        match self {
            Self::EpistemicScaleShift => 0.12,
            _ => REPORTED_EPISTEMIC_STD,
        }
    }

    fn aleatoric_std(self) -> f64 {
        match self {
            Self::AleatoricNoiseShift => 0.18,
            _ => 0.04,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    DirectFallback,
    GreedyMean,
    TotalVarianceLcb,
    PessimisticEnsembleMin,
    MarginalConformalLcb,
    FixedKJointConformalLcb,
    MatchedKJointConformalLcb,
    Oracle,
}

impl Policy {
    fn all() -> [Self; 8] {
        [
            Self::DirectFallback,
            Self::GreedyMean,
            Self::TotalVarianceLcb,
            Self::PessimisticEnsembleMin,
            Self::MarginalConformalLcb,
            Self::FixedKJointConformalLcb,
            Self::MatchedKJointConformalLcb,
            Self::Oracle,
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::DirectFallback => "direct_fallback",
            Self::GreedyMean => "greedy_predicted_mean",
            Self::TotalVarianceLcb => "total_variance_lcb",
            Self::PessimisticEnsembleMin => "pessimistic_ensemble_min_value",
            Self::MarginalConformalLcb => "marginal_conformal_lcb",
            Self::FixedKJointConformalLcb => "fixed_k8_joint_conformal_lcb",
            Self::MatchedKJointConformalLcb => "matched_k_joint_conformal_lcb",
            Self::Oracle => "oracle_action",
        }
    }

    fn uses_interval(self) -> bool {
        matches!(
            self,
            Self::TotalVarianceLcb
                | Self::MarginalConformalLcb
                | Self::FixedKJointConformalLcb
                | Self::MatchedKJointConformalLcb
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct Action {
    true_value: f64,
    predicted_mean: f64,
    epistemic_std: f64,
    aleatoric_std: f64,
    realized_reward: f64,
    ensemble_values: [f64; ENSEMBLE_HEADS],
}

#[derive(Clone, Copy, Debug)]
struct Calibration {
    marginal_q: f64,
    joint_q: f64,
}

#[derive(Clone, Copy, Debug)]
struct EpisodeOutcome {
    decision_regret: f64,
    realized_reward: f64,
    model_used: bool,
    joint_covered: Option<bool>,
    selected_action_covered: Option<bool>,
}

#[derive(Clone, Debug)]
struct Summary {
    scenario: &'static str,
    candidate_count: usize,
    policy: &'static str,
    mean_decision_regret: f64,
    p95_decision_regret: f64,
    mean_realized_reward: f64,
    cvar05_realized_reward: f64,
    model_use_rate: f64,
    joint_candidate_set_coverage: Option<f64>,
    selected_action_error_coverage: Option<f64>,
    regret_delta_vs_greedy: f64,
    regret_delta_ci95_low: f64,
    regret_delta_ci95_high: f64,
    calibration_marginal_q: f64,
    calibration_joint_q: f64,
    fixed_k8_joint_q: f64,
}

fn episode_seed(base: u64, scenario: Scenario, candidate_count: usize, episode: usize) -> u64 {
    base
        ^ ((scenario as u64) << 48)
        ^ (candidate_count as u64).wrapping_mul(0xD134_2543_DE82_EF95)
        ^ (episode as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn build_actions(rng: &mut SplitMix64, scenario: Scenario, count: usize) -> Vec<Action> {
    (0..count)
        .map(|_| {
            let optimistic_trap = scenario == Scenario::OptimisticFailureCandidates
                && rng.unit() < 0.12;
            let true_value = if optimistic_trap {
                0.22 + 0.12 * rng.unit()
            } else {
                0.35 + 0.35 * rng.unit()
            };
            let epistemic_error =
                rng.centered_normal_approx() * scenario.actual_epistemic_std();
            let optimism_bias = if optimistic_trap { 0.16 } else { 0.0 };
            let predicted_mean =
                (true_value + epistemic_error + optimism_bias).clamp(0.0, 1.0);
            let epistemic_std = REPORTED_EPISTEMIC_STD;
            let aleatoric_std = scenario.aleatoric_std();
            let reward_noise = rng.centered_normal_approx() * aleatoric_std;
            let realized_reward = (true_value + reward_noise).clamp(0.0, 1.0);
            let ensemble_values = std::array::from_fn(|_| {
                (true_value
                    + rng.centered_normal_approx() * scenario.actual_epistemic_std()
                    + optimism_bias)
                    .clamp(0.0, 1.0)
            });

            Action {
                true_value,
                predicted_mean,
                epistemic_std,
                aleatoric_std,
                realized_reward,
                ensemble_values,
            }
        })
        .collect()
}

fn conformal_quantile(scores: &mut [f64], target_coverage: f64) -> f64 {
    assert!(!scores.is_empty());
    scores.sort_by(f64::total_cmp);
    let n = scores.len();
    let rank = (((n + 1) as f64) * target_coverage).ceil() as usize;
    let index = rank.saturating_sub(1).min(n - 1);
    scores[index]
}

fn calibrate(candidate_count: usize) -> Calibration {
    let mut rng = SplitMix64::new(
        CALIBRATION_SEED ^ (candidate_count as u64).wrapping_mul(0xA076_1D64_78BD_642F),
    );
    let mut marginal_scores = Vec::with_capacity(CALIBRATION_EPISODES * candidate_count);
    let mut joint_scores = Vec::with_capacity(CALIBRATION_EPISODES);

    for _ in 0..CALIBRATION_EPISODES {
        let actions = build_actions(&mut rng, Scenario::InDistribution, candidate_count);
        let mut maximum = 0.0_f64;
        for action in &actions {
            let score =
                (action.predicted_mean - action.true_value).abs() / action.epistemic_std;
            marginal_scores.push(score);
            maximum = maximum.max(score);
        }
        joint_scores.push(maximum);
    }

    Calibration {
        marginal_q: conformal_quantile(&mut marginal_scores, TARGET_COVERAGE),
        joint_q: conformal_quantile(&mut joint_scores, TARGET_COVERAGE),
    }
}

fn interval_for_action(
    action: &Action,
    policy: Policy,
    calibration: Calibration,
    fixed_k8: Calibration,
) -> Option<f64> {
    match policy {
        Policy::TotalVarianceLcb => Some(
            TOTAL_VARIANCE_Z
                * (action.epistemic_std.powi(2) + action.aleatoric_std.powi(2)).sqrt(),
        ),
        Policy::MarginalConformalLcb => Some(calibration.marginal_q * action.epistemic_std),
        Policy::FixedKJointConformalLcb => Some(fixed_k8.joint_q * action.epistemic_std),
        Policy::MatchedKJointConformalLcb => Some(calibration.joint_q * action.epistemic_std),
        _ => None,
    }
}

fn policy_score(
    action: &Action,
    policy: Policy,
    calibration: Calibration,
    fixed_k8: Calibration,
) -> f64 {
    match policy {
        Policy::DirectFallback => f64::NEG_INFINITY,
        Policy::GreedyMean => action.predicted_mean,
        Policy::PessimisticEnsembleMin => action.ensemble_values.iter().copied().fold(f64::INFINITY, f64::min),
        Policy::TotalVarianceLcb
        | Policy::MarginalConformalLcb
        | Policy::FixedKJointConformalLcb
        | Policy::MatchedKJointConformalLcb => {
            action.predicted_mean
                - interval_for_action(action, policy, calibration, fixed_k8).unwrap()
        }
        Policy::Oracle => action.true_value,
    }
}

fn simulate_episode(
    actions: &[Action],
    policy: Policy,
    calibration: Calibration,
    fixed_k8: Calibration,
) -> EpisodeOutcome {
    let best_true_value = actions
        .iter()
        .map(|action| action.true_value)
        .fold(f64::NEG_INFINITY, f64::max);

    if policy == Policy::DirectFallback {
        return EpisodeOutcome {
            decision_regret: (best_true_value - FALLBACK_VALUE).max(0.0),
            realized_reward: FALLBACK_VALUE,
            model_used: false,
            joint_covered: None,
            selected_action_covered: None,
        };
    }

    let selected_index = actions
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            policy_score(left, policy, calibration, fixed_k8)
                .total_cmp(&policy_score(right, policy, calibration, fixed_k8))
        })
        .map(|(index, _)| index)
        .expect("candidate set is non-empty");

    let selected = &actions[selected_index];
    let selected_bound = interval_for_action(selected, policy, calibration, fixed_k8);
    let used_fallback = match policy {
        Policy::PessimisticEnsembleMin => {
            selected.ensemble_values.iter().copied().fold(f64::INFINITY, f64::min)
                < FALLBACK_VALUE
        }
        _ => selected_bound.is_some()
            && selected.predicted_mean - selected_bound.unwrap() < FALLBACK_VALUE,
    };

    if used_fallback {
        let bound_coverage = joint_coverage(actions, policy, calibration, fixed_k8);
        return EpisodeOutcome {
            decision_regret: (best_true_value - FALLBACK_VALUE).max(0.0),
            realized_reward: FALLBACK_VALUE,
            model_used: false,
            joint_covered: bound_coverage,
            selected_action_covered: None,
        };
    }

    let selected_covered = selected_bound.map(|bound| {
        (selected.predicted_mean - selected.true_value).abs() <= bound
    });

    EpisodeOutcome {
        decision_regret: (best_true_value - selected.true_value).max(0.0),
        realized_reward: selected.realized_reward,
        model_used: true,
        joint_covered: joint_coverage(actions, policy, calibration, fixed_k8),
        selected_action_covered: selected_covered,
    }
}

fn joint_coverage(
    actions: &[Action],
    policy: Policy,
    calibration: Calibration,
    fixed_k8: Calibration,
) -> Option<bool> {
    if !policy.uses_interval() {
        return None;
    }
    Some(actions.iter().all(|action| {
        let bound = interval_for_action(action, policy, calibration, fixed_k8).unwrap();
        (action.predicted_mean - action.true_value).abs() <= bound
    }))
}

fn percentile(values: &mut [f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let index = ((values.len() as f64 * quantile).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    values[index]
}

fn summarize(
    scenario: Scenario,
    candidate_count: usize,
    policy: Policy,
    outcomes: &[EpisodeOutcome],
    greedy: &[EpisodeOutcome],
    calibration: Calibration,
    fixed_k8: Calibration,
) -> Summary {
    let n = outcomes.len() as f64;
    let mean_regret = outcomes.iter().map(|x| x.decision_regret).sum::<f64>() / n;
    let mean_reward = outcomes.iter().map(|x| x.realized_reward).sum::<f64>() / n;
    let model_used = outcomes.iter().filter(|x| x.model_used).count() as f64;
    let mut regrets: Vec<f64> = outcomes.iter().map(|x| x.decision_regret).collect();
    let p95_regret = percentile(&mut regrets, 0.95);
    let mut rewards: Vec<f64> = outcomes.iter().map(|x| x.realized_reward).collect();
    let tail_count = ((rewards.len() as f64) * 0.05).ceil() as usize;
    rewards.sort_by(f64::total_cmp);
    let cvar05 = rewards.iter().take(tail_count).sum::<f64>() / tail_count.max(1) as f64;

    let joint_values: Vec<bool> = outcomes.iter().filter_map(|x| x.joint_covered).collect();
    let selected_values: Vec<bool> = outcomes
        .iter()
        .filter_map(|x| x.selected_action_covered)
        .collect();
    let joint_coverage = if joint_values.is_empty() {
        None
    } else {
        Some(
            joint_values.iter().filter(|x| **x).count() as f64 / joint_values.len() as f64,
        )
    };
    let selected_coverage = if selected_values.is_empty() {
        None
    } else {
        Some(
            selected_values.iter().filter(|x| **x).count() as f64
                / selected_values.len() as f64,
        )
    };

    let deltas: Vec<f64> = outcomes
        .iter()
        .zip(greedy)
        .map(|(candidate, baseline)| candidate.decision_regret - baseline.decision_regret)
        .collect();
    let delta_mean = deltas.iter().sum::<f64>() / n;
    let variance = deltas
        .iter()
        .map(|value| (value - delta_mean).powi(2))
        .sum::<f64>()
        / (n - 1.0).max(1.0);
    let margin = 1.96 * (variance / n).sqrt();

    Summary {
        scenario: scenario.name(),
        candidate_count,
        policy: policy.name(),
        mean_decision_regret: mean_regret,
        p95_decision_regret: p95_regret,
        mean_realized_reward: mean_reward,
        cvar05_realized_reward: cvar05,
        model_use_rate: model_used / n,
        joint_candidate_set_coverage: joint_coverage,
        selected_action_error_coverage: selected_coverage,
        regret_delta_vs_greedy: delta_mean,
        regret_delta_ci95_low: delta_mean - margin,
        regret_delta_ci95_high: delta_mean + margin,
        calibration_marginal_q: calibration.marginal_q,
        calibration_joint_q: calibration.joint_q,
        fixed_k8_joint_q: fixed_k8.joint_q,
    }
}

fn run_benchmark() -> Vec<Summary> {
    let mut rows = Vec::with_capacity(
        Scenario::all().len() * CANDIDATE_COUNTS.len() * Policy::all().len(),
    );
    let fixed_k8 = calibrate(8);

    for scenario in Scenario::all() {
        for candidate_count in CANDIDATE_COUNTS {
            let calibration = calibrate(candidate_count);
            let mut outcomes_by_policy: Vec<Vec<EpisodeOutcome>> =
                Policy::all().iter().map(|_| Vec::with_capacity(QUALIFICATION_EPISODES)).collect();

            for episode in 0..QUALIFICATION_EPISODES {
                let mut rng = SplitMix64::new(episode_seed(
                    QUALIFICATION_SEED ^ BASE_SEED,
                    scenario,
                    candidate_count,
                    episode,
                ));
                let actions = build_actions(&mut rng, scenario, candidate_count);

                for (index, policy) in Policy::all().iter().enumerate() {
                    outcomes_by_policy[index].push(simulate_episode(
                        &actions,
                        *policy,
                        calibration,
                        fixed_k8,
                    ));
                }
            }

            let greedy_index = Policy::all()
                .iter()
                .position(|policy| *policy == Policy::GreedyMean)
                .unwrap();
            let greedy = outcomes_by_policy[greedy_index].clone();

            for (index, policy) in Policy::all().iter().enumerate() {
                rows.push(summarize(
                    scenario,
                    candidate_count,
                    *policy,
                    &outcomes_by_policy[index],
                    &greedy,
                    calibration,
                    fixed_k8,
                ));
            }
        }
    }
    rows
}

fn optional_number(value: Option<f64>) -> String {
    match value {
        Some(number) if number.is_finite() => format!("{number:.8}"),
        _ => "null".to_string(),
    }
}

fn render_report(rows: &[Summary]) -> String {
    let mut output = String::new();
    writeln!(&mut output, "{{").unwrap();
    writeln!(&mut output, "  \"schema_version\": \"{SCHEMA}\",").unwrap();
    writeln!(&mut output, "  \"calibration_episode_count\": {CALIBRATION_EPISODES},").unwrap();
    writeln!(&mut output, "  \"qualification_episode_count\": {QUALIFICATION_EPISODES},").unwrap();
    writeln!(&mut output, "  \"target_coverage\": {TARGET_COVERAGE:.4},").unwrap();
    writeln!(&mut output, "  \"reported_epistemic_std\": {REPORTED_EPISTEMIC_STD:.4},").unwrap();
    writeln!(&mut output, "  \"fallback_value\": {FALLBACK_VALUE:.4},").unwrap();
    writeln!(&mut output, "  \"ensemble_heads\": {ENSEMBLE_HEADS},").unwrap();
    writeln!(&mut output, "  \"calibration_seed_family\": \"{CALIBRATION_SEED:016x}\",").unwrap();
    writeln!(&mut output, "  \"qualification_seed_family\": \"{QUALIFICATION_SEED:016x}\",").unwrap();
    writeln!(&mut output, "  \"candidate_counts\": [8,16,32,64],").unwrap();
    writeln!(&mut output, "  \"scenarios\": [").unwrap();
    for (index, scenario) in Scenario::all().iter().enumerate() {
        let comma = if index + 1 == Scenario::all().len() { "" } else { "," };
        writeln!(&mut output, "    \"{}\"{comma}", scenario.name()).unwrap();
    }
    writeln!(&mut output, "  ],").unwrap();
    writeln!(&mut output, "  \"policies\": [").unwrap();
    for (index, policy) in Policy::all().iter().enumerate() {
        let comma = if index + 1 == Policy::all().len() { "" } else { "," };
        writeln!(&mut output, "    \"{}\"{comma}", policy.name()).unwrap();
    }
    writeln!(&mut output, "  ],").unwrap();
    writeln!(&mut output, "  \"summary\": [").unwrap();

    for (index, row) in rows.iter().enumerate() {
        let comma = if index + 1 == rows.len() { "" } else { "," };
        writeln!(
            &mut output,
            concat!(
                "    {{\"scenario\":\"{}\",\"candidate_count\":{},\"policy\":\"{}\",",
                "\"mean_decision_regret\":{},\"p95_decision_regret\":{},",
                "\"mean_realized_reward\":{},\"cvar05_realized_reward\":{},",
                "\"model_use_rate\":{},\"joint_candidate_set_coverage\":{},",
                "\"selected_action_error_coverage\":{},\"regret_delta_vs_greedy\":{},",
                "\"regret_delta_ci95_low\":{},\"regret_delta_ci95_high\":{},",
                "\"calibration_marginal_q\":{},\"calibration_joint_q\":{},",
                "\"fixed_k8_joint_q\":{}}}{}"
            ),
            row.scenario,
            row.candidate_count,
            row.policy,
            row.mean_decision_regret,
            row.p95_decision_regret,
            row.mean_realized_reward,
            row.cvar05_realized_reward,
            row.model_use_rate,
            optional_number(row.joint_candidate_set_coverage),
            optional_number(row.selected_action_error_coverage),
            row.regret_delta_vs_greedy,
            row.regret_delta_ci95_low,
            row.regret_delta_ci95_high,
            row.calibration_marginal_q,
            row.calibration_joint_q,
            row.fixed_k8_joint_q,
            comma
        )
        .unwrap();
    }
    writeln!(&mut output, "  ]").unwrap();
    writeln!(&mut output, "}}").unwrap();
    output
}

fn main() {
    print!("{}", render_report(&run_benchmark()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_conformal_quantile_uses_finite_sample_rank() {
        let mut scores = (1..=10).map(|x| x as f64).collect::<Vec<_>>();
        assert_eq!(conformal_quantile(&mut scores, 0.90), 10.0);
    }

    #[test]
    fn calibration_quantiles_are_finite_and_positive() {
        let calibration = calibrate(16);
        assert!(calibration.marginal_q.is_finite() && calibration.marginal_q > 0.0);
        assert!(calibration.joint_q.is_finite() && calibration.joint_q > 0.0);
    }

    #[test]
    fn ensemble_minimum_is_no_greater_than_any_head() {
        let action = Action {
            true_value: 0.5,
            predicted_mean: 0.6,
            epistemic_std: 0.1,
            aleatoric_std: 0.1,
            realized_reward: 0.5,
            ensemble_values: [0.7, 0.6, 0.8, 0.4, 0.9],
        };
        let minimum = action.ensemble_values.iter().copied().fold(f64::INFINITY, f64::min);
        assert_eq!(minimum, 0.4);
        assert!(action.ensemble_values.iter().all(|value| minimum <= *value));
    }

    #[test]
    fn calibration_and_qualification_seed_families_are_disjoint() {
        assert_ne!(CALIBRATION_SEED, QUALIFICATION_SEED);
    }

    #[test]
    fn scenario_changes_separate_epistemic_and_aleatoric_uncertainty() {
        assert!(Scenario::EpistemicScaleShift.actual_epistemic_std()
            > Scenario::InDistribution.actual_epistemic_std());
        assert_eq!(
            Scenario::AleatoricNoiseShift.actual_epistemic_std(),
            Scenario::InDistribution.actual_epistemic_std()
        );
        assert!(Scenario::AleatoricNoiseShift.aleatoric_std()
            > Scenario::InDistribution.aleatoric_std());
    }

    #[test]
    fn joint_conformal_bound_uses_all_candidates() {
        let calibration = Calibration { marginal_q: 1.0, joint_q: 2.0 };
        let fixed = Calibration { marginal_q: 1.0, joint_q: 3.0 };
        let action = Action {
            true_value: 0.5,
            predicted_mean: 0.7,
            epistemic_std: 0.1,
            aleatoric_std: 0.1,
            realized_reward: 0.5,
            ensemble_values: [0.5; ENSEMBLE_HEADS],
        };
        let marginal = interval_for_action(
            &action, Policy::MarginalConformalLcb, calibration, fixed
        ).unwrap();
        let matched = interval_for_action(
            &action, Policy::MatchedKJointConformalLcb, calibration, fixed
        ).unwrap();
        let fixed_k8 = interval_for_action(
            &action, Policy::FixedKJointConformalLcb, calibration, fixed
        ).unwrap();
        assert!((marginal - 0.1).abs() < 1e-12);
        assert!((matched - 0.2).abs() < 1e-12);
        assert!((fixed_k8 - 0.3).abs() < 1e-12);
    }

    #[test]
    fn report_contains_all_scenario_count_policy_rows() {
        let expected = Scenario::all().len() * CANDIDATE_COUNTS.len() * Policy::all().len();
        let rows = run_benchmark();
        assert_eq!(rows.len(), expected);
        let report = render_report(&rows);
        assert!(report.contains(&format!("\"schema_version\": \"{SCHEMA}\"")));
        assert!(report.contains("\"candidate_count\":8"));
        assert!(report.contains("\"candidate_count\":64"));
    }
}
