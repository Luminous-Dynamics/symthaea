use std::process;

const REPORT_SCHEMA: &str = "cmaq-008-sequential-support-drift-v0.1";
const EPISODES: usize = 512;
const HORIZON: usize = 128;
const CHANGE_DETECTION_THRESHOLD: f64 = 5.0;
const NULL_UNSUPPORTED_RATE: f64 = 0.01;
const SHIFTED_UNSUPPORTED_RATE: f64 = 0.25;
const SUPPORTED_UTILITY: f64 = 0.03;
const UNSUPPORTED_UTILITY: f64 = -0.20;
const BASE_SEED: u64 = 0xC0A8_0008_2026_1030;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    Stationary,
    AbruptSupportShift,
    GradualSupportShift,
}

impl Scenario {
    fn all() -> [Self; 3] {
        [Self::Stationary, Self::AbruptSupportShift, Self::GradualSupportShift]
    }

    fn name(self) -> &'static str {
        match self {
            Self::Stationary => "stationary",
            Self::AbruptSupportShift => "abrupt_support_shift",
            Self::GradualSupportShift => "gradual_support_shift",
        }
    }

    fn shift_onset(self) -> Option<usize> {
        match self {
            Self::Stationary => None,
            Self::AbruptSupportShift => Some(HORIZON / 2),
            Self::GradualSupportShift => Some(HORIZON / 4),
        }
    }

    fn unsupported_probability(self, step: usize) -> f64 {
        match self {
            Self::Stationary => NULL_UNSUPPORTED_RATE,
            Self::AbruptSupportShift => {
                if step < HORIZON / 2 {
                    NULL_UNSUPPORTED_RATE
                } else {
                    SHIFTED_UNSUPPORTED_RATE
                }
            }
            Self::GradualSupportShift => {
                let start = HORIZON / 4;
                let end = (3 * HORIZON) / 4;
                if step < start {
                    NULL_UNSUPPORTED_RATE
                } else if step >= end {
                    SHIFTED_UNSUPPORTED_RATE
                } else {
                    let fraction = (step - start) as f64 / (end - start) as f64;
                    NULL_UNSUPPORTED_RATE
                        + fraction * (SHIFTED_UNSUPPORTED_RATE - NULL_UNSUPPORTED_RATE)
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    DirectFallback,
    FixedGate,
    RevokeOnFirstUnsupported,
    CusumRevocableGate,
    OracleChangepoint,
}

impl Policy {
    fn all() -> [Self; 5] {
        [
            Self::DirectFallback,
            Self::FixedGate,
            Self::RevokeOnFirstUnsupported,
            Self::CusumRevocableGate,
            Self::OracleChangepoint,
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::DirectFallback => "direct_fallback",
            Self::FixedGate => "fixed_gate",
            Self::RevokeOnFirstUnsupported => "first_unsupported_revoke",
            Self::CusumRevocableGate => "cusum_revocable_gate",
            Self::OracleChangepoint => "oracle_changepoint",
        }
    }

    fn is_online_monitor(self) -> bool {
        matches!(self, Self::RevokeOnFirstUnsupported | Self::CusumRevocableGate)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct EpisodeOutcome {
    cumulative_utility: f64,
    model_action_count: usize,
    unsupported_count: usize,
    revocation_step: Option<usize>,
}

fn likelihood_ratio_increment(unsupported_observed: bool) -> f64 {
    if unsupported_observed {
        (SHIFTED_UNSUPPORTED_RATE / NULL_UNSUPPORTED_RATE).ln()
    } else {
        ((1.0 - SHIFTED_UNSUPPORTED_RATE) / (1.0 - NULL_UNSUPPORTED_RATE)).ln()
    }
}

fn simulate_episode(scenario: Scenario, policy: Policy, seed: u64) -> EpisodeOutcome {
    let mut rng = SplitMix64::new(seed);
    let onset = scenario.shift_onset();
    let mut outcome = EpisodeOutcome::default();
    let mut cusum = 0.0_f64;
    let mut revoked = false;

    for step in 0..HORIZON {
        // Consume the same random tape for every paired policy, even after
        // fallback, so an intervention never changes exogenous randomness.
        let draw = rng.unit();
        if policy == Policy::OracleChangepoint && onset == Some(step) {
            outcome.revocation_step = Some(step);
            revoked = true;
        }

        let active = match policy {
            Policy::DirectFallback => false,
            Policy::FixedGate => true,
            Policy::RevokeOnFirstUnsupported | Policy::CusumRevocableGate => !revoked,
            Policy::OracleChangepoint => !revoked,
        };

        if !active {
            continue;
        }

        outcome.model_action_count += 1;
        let unsupported_observed = draw < scenario.unsupported_probability(step);
        if unsupported_observed {
            outcome.unsupported_count += 1;
            outcome.cumulative_utility += UNSUPPORTED_UTILITY;
        } else {
            outcome.cumulative_utility += SUPPORTED_UTILITY;
        }

        match policy {
            Policy::RevokeOnFirstUnsupported if unsupported_observed => {
                revoked = true;
                outcome.revocation_step = Some(step);
            }
            Policy::CusumRevocableGate => {
                // A one-sided sequential log-likelihood CUSUM contrasts a
                // stationary 1% unsupported-feedback null with a 25% shift.
                // The threshold is frozen before the scenario panel is run.
                cusum = (cusum + likelihood_ratio_increment(unsupported_observed)).max(0.0);
                if cusum >= CHANGE_DETECTION_THRESHOLD {
                    revoked = true;
                    outcome.revocation_step = Some(step);
                }
            }
            _ => {}
        }
    }

    outcome
}

#[derive(Clone, Debug)]
struct Summary {
    scenario: &'static str,
    policy: &'static str,
    episode_count: usize,
    mean_utility: f64,
    mean_utility_delta_vs_fixed: f64,
    utility_delta_ci95_low: f64,
    utility_delta_ci95_high: f64,
    model_action_coverage: f64,
    unsupported_rate_per_model_action: f64,
    unsupported_actions_per_1000_decisions: f64,
    revocation_episode_rate: f64,
    false_revocation_episode_rate: f64,
    shift_detection_coverage: f64,
    p90_detection_delay_steps: f64,
    mean_regret_vs_oracle: f64,
    cvar95_regret_vs_oracle: f64,
    max_regret_vs_oracle: f64,
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
    let standard_error = (variance / values.len() as f64).sqrt();
    (average, average - 1.96 * standard_error, average + 1.96 * standard_error)
}

fn cvar95(regrets: &[f64]) -> f64 {
    if regrets.is_empty() {
        return 0.0;
    }
    let mut sorted = regrets.to_vec();
    sorted.sort_by(|left, right| right.total_cmp(left));
    let tail_count = ((sorted.len() as f64 * 0.05).ceil() as usize).max(1);
    mean(&sorted[..tail_count])
}

fn summarize(
    scenario: Scenario,
    policy: Policy,
    outcomes: &[EpisodeOutcome],
    fixed_outcomes: &[EpisodeOutcome],
    oracle_outcomes: &[EpisodeOutcome],
) -> Summary {
    let utilities: Vec<f64> = outcomes.iter().map(|outcome| outcome.cumulative_utility).collect();
    let utility_deltas: Vec<f64> = outcomes
        .iter()
        .zip(fixed_outcomes)
        .map(|(current, fixed)| current.cumulative_utility - fixed.cumulative_utility)
        .collect();
    let regrets: Vec<f64> = outcomes
        .iter()
        .zip(oracle_outcomes)
        .map(|(current, oracle)| (oracle.cumulative_utility - current.cumulative_utility).max(0.0))
        .collect();
    let (delta_mean, delta_low, delta_high) = mean_ci95(&utility_deltas);

    let model_actions: usize = outcomes.iter().map(|outcome| outcome.model_action_count).sum();
    let unsupported: usize = outcomes.iter().map(|outcome| outcome.unsupported_count).sum();
    let total_decisions = outcomes.len() * HORIZON;
    let revocations = outcomes.iter().filter(|outcome| outcome.revocation_step.is_some()).count();

    let onset = scenario.shift_onset();
    let false_revocations = outcomes
        .iter()
        .filter(|outcome| match (onset, outcome.revocation_step) {
            (None, Some(_)) => policy.is_online_monitor(),
            (Some(shift_at), Some(revoked_at)) => policy.is_online_monitor() && revoked_at < shift_at,
            _ => false,
        })
        .count();

    let detected = outcomes
        .iter()
        .filter(|outcome| matches!((onset, outcome.revocation_step), (Some(shift_at), Some(revoked_at)) if revoked_at >= shift_at))
        .count();

    let detection_delays: Vec<f64> = outcomes
        .iter()
        .filter_map(|outcome| match (onset, outcome.revocation_step) {
            (Some(shift_at), Some(revoked_at)) if revoked_at >= shift_at => {
                Some((revoked_at - shift_at + 1) as f64)
            }
            _ => None,
        })
        .collect();
    let p90_delay = if detection_delays.is_empty() {
        -1.0
    } else {
        let mut sorted = detection_delays;
        sorted.sort_by(f64::total_cmp);
        let index = ((sorted.len() as f64 * 0.90).ceil() as usize).saturating_sub(1);
        sorted[index]
    };

    Summary {
        scenario: scenario.name(),
        policy: policy.name(),
        episode_count: outcomes.len(),
        mean_utility: mean(&utilities),
        mean_utility_delta_vs_fixed: delta_mean,
        utility_delta_ci95_low: delta_low,
        utility_delta_ci95_high: delta_high,
        model_action_coverage: model_actions as f64 / total_decisions as f64,
        unsupported_rate_per_model_action: if model_actions == 0 {
            0.0
        } else {
            unsupported as f64 / model_actions as f64
        },
        unsupported_actions_per_1000_decisions: unsupported as f64 / total_decisions as f64 * 1000.0,
        revocation_episode_rate: revocations as f64 / outcomes.len() as f64,
        false_revocation_episode_rate: false_revocations as f64 / outcomes.len() as f64,
        shift_detection_coverage: if onset.is_some() {
            detected as f64 / outcomes.len() as f64
        } else {
            0.0
        },
        p90_detection_delay_steps: p90_delay,
        mean_regret_vs_oracle: mean(&regrets),
        cvar95_regret_vs_oracle: cvar95(&regrets),
        max_regret_vs_oracle: regrets.iter().copied().fold(0.0_f64, f64::max),
    }
}

fn run_benchmark() -> Vec<Summary> {
    let mut output = Vec::with_capacity(Scenario::all().len() * Policy::all().len());
    for scenario in Scenario::all() {
        let fixed_outcomes: Vec<EpisodeOutcome> = (0..EPISODES)
            .map(|episode| {
                let seed = BASE_SEED ^ (episode as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                simulate_episode(scenario, Policy::FixedGate, seed)
            })
            .collect();
        let oracle_outcomes: Vec<EpisodeOutcome> = (0..EPISODES)
            .map(|episode| {
                let seed = BASE_SEED ^ (episode as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                simulate_episode(scenario, Policy::OracleChangepoint, seed)
            })
            .collect();

        for policy in Policy::all() {
            let outcomes: Vec<EpisodeOutcome> = (0..EPISODES)
                .map(|episode| {
                    let seed = BASE_SEED ^ (episode as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    simulate_episode(scenario, policy, seed)
                })
                .collect();
            output.push(summarize(
                scenario,
                policy,
                &outcomes,
                &fixed_outcomes,
                &oracle_outcomes,
            ));
        }
    }
    output
}

fn render_report(rows: &[Summary]) -> String {
    let mut output = String::new();
    output.push_str("{\n");
    output.push_str(&format!("  \"schema_version\": \"{REPORT_SCHEMA}\",\n"));
    output.push_str(&format!("  \"episode_count\": {EPISODES},\n"));
    output.push_str(&format!("  \"horizon\": {HORIZON},\n"));
    output.push_str(&format!("  \"base_seed\": \"{BASE_SEED:016x}\",\n"));
    output.push_str(&format!("  \"cusum_threshold\": {CHANGE_DETECTION_THRESHOLD:.4},\n"));
    output.push_str(&format!("  \"null_unsupported_rate\": {NULL_UNSUPPORTED_RATE:.4},\n"));
    output.push_str(&format!("  \"shifted_unsupported_rate\": {SHIFTED_UNSUPPORTED_RATE:.4},\n"));
    output.push_str(&format!("  \"supported_utility\": {SUPPORTED_UTILITY:.4},\n"));
    output.push_str(&format!("  \"unsupported_utility\": {UNSUPPORTED_UTILITY:.4},\n"));
    output.push_str("  \"scenarios\": [\"stationary\", \"abrupt_support_shift\", \"gradual_support_shift\"],\n");
    output.push_str("  \"policies\": [\"direct_fallback\", \"fixed_gate\", \"first_unsupported_revoke\", \"cusum_revocable_gate\", \"oracle_changepoint\"],\n");
    output.push_str("  \"summary\": [\n");
    for (index, row) in rows.iter().enumerate() {
        output.push_str(&format!(
            concat!(
                "    {{\"scenario\": \"{}\", \"policy\": \"{}\", \"episode_count\": {}, ",
                "\"mean_utility\": {:.8}, \"mean_utility_delta_vs_fixed\": {:.8}, ",
                "\"utility_delta_ci95_low\": {:.8}, \"utility_delta_ci95_high\": {:.8}, ",
                "\"model_action_coverage\": {:.8}, \"unsupported_rate_per_model_action\": {:.8}, ",
                "\"unsupported_actions_per_1000_decisions\": {:.8}, ",
                "\"revocation_episode_rate\": {:.8}, \"false_revocation_episode_rate\": {:.8}, ",
                "\"shift_detection_coverage\": {:.8}, \"p90_detection_delay_steps\": {:.4}, ",
                "\"mean_regret_vs_oracle\": {:.8}, \"cvar95_regret_vs_oracle\": {:.8}, ",
                "\"max_regret_vs_oracle\": {:.8}}}"
            ),
            row.scenario,
            row.policy,
            row.episode_count,
            row.mean_utility,
            row.mean_utility_delta_vs_fixed,
            row.utility_delta_ci95_low,
            row.utility_delta_ci95_high,
            row.model_action_coverage,
            row.unsupported_rate_per_model_action,
            row.unsupported_actions_per_1000_decisions,
            row.revocation_episode_rate,
            row.false_revocation_episode_rate,
            row.shift_detection_coverage,
            row.p90_detection_delay_steps,
            row.mean_regret_vs_oracle,
            row.cvar95_regret_vs_oracle,
            row.max_regret_vs_oracle
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
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|argument| argument == "--help" || argument == "-h") {
        println!("CMAQ-008 sequential support-drift smoke benchmark\n\nRuns 512 paired seeds per scenario at horizon 128.\nPrints a deterministic JSON report to stdout.");
        return;
    }
    if !args.is_empty() {
        eprintln!("unexpected argument: {}; use --help", args[0]);
        process::exit(2);
    }
    print!("{}", render_report(&run_benchmark()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenario_tapes_define_the_preregistered_shift() {
        assert_eq!(Scenario::Stationary.unsupported_probability(127), 0.01);
        assert_eq!(Scenario::AbruptSupportShift.unsupported_probability(63), 0.01);
        assert_eq!(Scenario::AbruptSupportShift.unsupported_probability(64), 0.25);
        assert_eq!(Scenario::GradualSupportShift.unsupported_probability(32), 0.01);
        assert_eq!(Scenario::GradualSupportShift.unsupported_probability(96), 0.25);
    }

    #[test]
    fn likelihood_ratio_moves_up_on_evidence_and_down_on_null_feedback() {
        assert!(likelihood_ratio_increment(true) > 0.0);
        assert!(likelihood_ratio_increment(false) < 0.0);
    }

    #[test]
    fn fixed_seed_replay_is_deterministic() {
        let a = simulate_episode(Scenario::AbruptSupportShift, Policy::CusumRevocableGate, 7);
        let b = simulate_episode(Scenario::AbruptSupportShift, Policy::CusumRevocableGate, 7);
        assert_eq!(a.cumulative_utility, b.cumulative_utility);
        assert_eq!(a.model_action_count, b.model_action_count);
        assert_eq!(a.unsupported_count, b.unsupported_count);
        assert_eq!(a.revocation_step, b.revocation_step);
    }

    #[test]
    fn direct_fallback_never_uses_the_model_or_records_model_failures() {
        let outcome = simulate_episode(Scenario::AbruptSupportShift, Policy::DirectFallback, 9);
        assert_eq!(outcome.model_action_count, 0);
        assert_eq!(outcome.unsupported_count, 0);
        assert_eq!(outcome.cumulative_utility, 0.0);
    }

    #[test]
    fn fixed_gate_uses_the_model_for_the_entire_horizon() {
        let outcome = simulate_episode(Scenario::Stationary, Policy::FixedGate, 10);
        assert_eq!(outcome.model_action_count, HORIZON);
        assert_eq!(outcome.revocation_step, None);
    }

    #[test]
    fn cusum_revokes_after_evidence_crosses_the_frozen_threshold() {
        let scenario = Scenario::AbruptSupportShift;
        let mut found = false;
        for seed in 0..EPISODES as u64 {
            let outcome = simulate_episode(scenario, Policy::CusumRevocableGate, seed);
            if let Some(step) = outcome.revocation_step {
                if step >= HORIZON / 2 {
                    assert!(step < HORIZON);
                    assert!(outcome.model_action_count <= HORIZON);
                    found = true;
                    break;
                }
            }
        }
        assert!(found, "CUSUM should revoke after post-shift evidence on at least one tape");
    }

    #[test]
    fn first_failure_baseline_is_more_reactive_than_cusum() {
        let first = (0..EPISODES as u64)
            .filter(|seed| simulate_episode(Scenario::Stationary, Policy::RevokeOnFirstUnsupported, *seed).revocation_step.is_some())
            .count();
        let cusum = (0..EPISODES as u64)
            .filter(|seed| simulate_episode(Scenario::Stationary, Policy::CusumRevocableGate, *seed).revocation_step.is_some())
            .count();
        assert!(first > cusum);
    }

    #[test]
    fn empirical_tail_regret_is_not_below_mean_regret() {
        let outcomes: Vec<EpisodeOutcome> = (0..EPISODES as u64)
            .map(|seed| simulate_episode(Scenario::AbruptSupportShift, Policy::FixedGate, seed))
            .collect();
        let fixed = outcomes.clone();
        let oracle: Vec<EpisodeOutcome> = (0..EPISODES as u64)
            .map(|seed| simulate_episode(Scenario::AbruptSupportShift, Policy::OracleChangepoint, seed))
            .collect();
        let rows = summarize(Scenario::AbruptSupportShift, Policy::FixedGate, &outcomes, &fixed, &oracle);
        assert!(rows.cvar95_regret_vs_oracle >= rows.mean_regret_vs_oracle - 1e-9);
        assert!(rows.max_regret_vs_oracle >= rows.cvar95_regret_vs_oracle - 1e-9);
    }

    #[test]
    fn benchmark_report_has_one_row_per_scenario_policy_pair() {
        let rows = run_benchmark();
        assert_eq!(rows.len(), Scenario::all().len() * Policy::all().len());
        let json = render_report(&rows);
        assert!(json.contains("\"schema_version\": \"cmaq-008-sequential-support-drift-v0.1\""));
        assert!(json.contains("\"episode_count\": 512"));
        assert!(json.contains("\"horizon\": 128"));
    }
}
