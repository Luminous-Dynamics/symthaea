use std::process;

const REPORT_SCHEMA: &str = "cmaq-009-active-diagnostic-probes-v0.1";
const EPISODES: usize = 512;
const HORIZON: usize = 128;
const CHANGE_DETECTION_THRESHOLD: f64 = 5.0;
const NULL_ACTION_UNSUPPORTED_RATE: f64 = 0.01;
const SHIFTED_ACTION_UNSUPPORTED_RATE: f64 = 0.25;
const NULL_PROBE_FAILURE_RATE: f64 = 0.01;
const SHIFTED_PROBE_FAILURE_RATE: f64 = 0.60;
const SUPPORTED_UTILITY: f64 = 0.03;
const UNSUPPORTED_UTILITY: f64 = -0.20;
const PROBE_COST: f64 = 0.003;
const ADAPTIVE_PROBE_CAP: usize = 32;
const BASE_SEED: u64 = 0xC0A9_0009_2026_1030;

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

    fn action_unsupported_probability(self, step: usize) -> f64 {
        match self {
            Self::Stationary => NULL_ACTION_UNSUPPORTED_RATE,
            Self::AbruptSupportShift => {
                if step < HORIZON / 2 { NULL_ACTION_UNSUPPORTED_RATE } else { SHIFTED_ACTION_UNSUPPORTED_RATE }
            }
            Self::GradualSupportShift => {
                let start = HORIZON / 4;
                let end = (3 * HORIZON) / 4;
                if step < start {
                    NULL_ACTION_UNSUPPORTED_RATE
                } else if step >= end {
                    SHIFTED_ACTION_UNSUPPORTED_RATE
                } else {
                    let fraction = (step - start) as f64 / (end - start) as f64;
                    NULL_ACTION_UNSUPPORTED_RATE
                        + fraction * (SHIFTED_ACTION_UNSUPPORTED_RATE - NULL_ACTION_UNSUPPORTED_RATE)
                }
            }
        }
    }

    fn probe_failure_probability(self, step: usize) -> f64 {
        match self {
            Self::Stationary => NULL_PROBE_FAILURE_RATE,
            Self::AbruptSupportShift => {
                if step < HORIZON / 2 { NULL_PROBE_FAILURE_RATE } else { SHIFTED_PROBE_FAILURE_RATE }
            }
            Self::GradualSupportShift => {
                let start = HORIZON / 4;
                let end = (3 * HORIZON) / 4;
                if step < start {
                    NULL_PROBE_FAILURE_RATE
                } else if step >= end {
                    SHIFTED_PROBE_FAILURE_RATE
                } else {
                    let fraction = (step - start) as f64 / (end - start) as f64;
                    NULL_PROBE_FAILURE_RATE
                        + fraction * (SHIFTED_PROBE_FAILURE_RATE - NULL_PROBE_FAILURE_RATE)
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    DirectFallback,
    FixedGate,
    PassiveCusum,
    ProbeEvery2,
    ProbeEvery4,
    AdaptiveProbe,
    ZeroInformationProbeEvery2,
    OracleChangepoint,
}

impl Policy {
    fn all() -> [Self; 8] {
        [
            Self::DirectFallback,
            Self::FixedGate,
            Self::PassiveCusum,
            Self::ProbeEvery2,
            Self::ProbeEvery4,
            Self::AdaptiveProbe,
            Self::ZeroInformationProbeEvery2,
            Self::OracleChangepoint,
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::DirectFallback => "direct_fallback",
            Self::FixedGate => "fixed_gate",
            Self::PassiveCusum => "passive_cusum",
            Self::ProbeEvery2 => "probe_every_2",
            Self::ProbeEvery4 => "probe_every_4",
            Self::AdaptiveProbe => "adaptive_probe",
            Self::ZeroInformationProbeEvery2 => "zero_info_probe_every_2",
            Self::OracleChangepoint => "oracle_changepoint",
        }
    }

    fn probe_cap(self) -> usize {
        match self {
            Self::ProbeEvery2 | Self::ZeroInformationProbeEvery2 => HORIZON.div_ceil(2),
            Self::ProbeEvery4 => HORIZON.div_ceil(4),
            Self::AdaptiveProbe => ADAPTIVE_PROBE_CAP,
            _ => 0,
        }
    }

    fn has_detector(self) -> bool {
        matches!(
            self,
            Self::PassiveCusum
                | Self::ProbeEvery2
                | Self::ProbeEvery4
                | Self::AdaptiveProbe
                | Self::ZeroInformationProbeEvery2
        )
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct EpisodeOutcome {
    cumulative_utility: f64,
    model_action_count: usize,
    unsupported_count: usize,
    probe_count: usize,
    revocation_step: Option<usize>,
}

fn action_likelihood_increment(unsupported: bool) -> f64 {
    if unsupported {
        (SHIFTED_ACTION_UNSUPPORTED_RATE / NULL_ACTION_UNSUPPORTED_RATE).ln()
    } else {
        ((1.0 - SHIFTED_ACTION_UNSUPPORTED_RATE) / (1.0 - NULL_ACTION_UNSUPPORTED_RATE)).ln()
    }
}

fn probe_likelihood_increment(failure: bool) -> f64 {
    if failure {
        (SHIFTED_PROBE_FAILURE_RATE / NULL_PROBE_FAILURE_RATE).ln()
    } else {
        ((1.0 - SHIFTED_PROBE_FAILURE_RATE) / (1.0 - NULL_PROBE_FAILURE_RATE)).ln()
    }
}

fn probe_requested(policy: Policy, step: usize, adaptive_armed: bool, used: usize) -> bool {
    if used >= policy.probe_cap() {
        return false;
    }
    match policy {
        Policy::ProbeEvery2 => step % 2 == 0,
        Policy::ProbeEvery4 => step % 4 == 0,
        Policy::AdaptiveProbe => adaptive_armed,
        Policy::ZeroInformationProbeEvery2 => step % 2 == 0,
        _ => false,
    }
}

fn simulate_episode(scenario: Scenario, policy: Policy, seed: u64) -> EpisodeOutcome {
    let mut action_rng = SplitMix64::new(seed);
    let mut probe_rng = SplitMix64::new(seed ^ 0xA8A8_0099_2026_1030);
    let onset = scenario.shift_onset();
    let mut outcome = EpisodeOutcome::default();
    let mut evidence_score = 0.0_f64;
    let mut adaptive_armed = false;
    let mut revoked = false;

    for step in 0..HORIZON {
        // Consume both tapes on every step, even after revocation, so policy
        // interventions don't change exogenous randomness on paired episodes.
        let action_draw = action_rng.unit();
        let probe_draw = probe_rng.unit();

        if policy == Policy::OracleChangepoint && onset == Some(step) {
            revoked = true;
            outcome.revocation_step = Some(step);
        }

        let active = match policy {
            Policy::DirectFallback => false,
            Policy::FixedGate => true,
            Policy::OracleChangepoint => !revoked,
            _ => !revoked,
        };
        if !active {
            continue;
        }

        outcome.model_action_count += 1;
        let unsupported = action_draw < scenario.action_unsupported_probability(step);
        if unsupported {
            outcome.unsupported_count += 1;
            outcome.cumulative_utility += UNSUPPORTED_UTILITY;
        } else {
            outcome.cumulative_utility += SUPPORTED_UTILITY;
        }

        if policy.has_detector() {
            evidence_score = (evidence_score + action_likelihood_increment(unsupported)).max(0.0);
        }

        let wants_probe = probe_requested(policy, step, adaptive_armed, outcome.probe_count);
        if wants_probe {
            outcome.probe_count += 1;
            outcome.cumulative_utility -= PROBE_COST;
            if policy != Policy::ZeroInformationProbeEvery2 {
                let probe_failure = probe_draw < scenario.probe_failure_probability(step);
                // Action and probe evidence are independent under the declared
                // synthetic mechanism, so their log-likelihood increments add.
                evidence_score = (evidence_score + probe_likelihood_increment(probe_failure)).max(0.0);
            }
        }

        adaptive_armed = policy == Policy::AdaptiveProbe
            && evidence_score >= 1.0
            && outcome.probe_count < policy.probe_cap();

        if policy.has_detector() && evidence_score >= CHANGE_DETECTION_THRESHOLD {
            // The triggering action/probe has already incurred its utility/cost.
            revoked = true;
            outcome.revocation_step = Some(step);
        }
    }

    outcome
}

#[derive(Clone, Debug)]
struct Summary {
    scenario: &'static str,
    policy: &'static str,
    episode_count: usize,
    probe_cap: usize,
    mean_utility: f64,
    mean_utility_delta_vs_passive: f64,
    utility_delta_ci95_low: f64,
    utility_delta_ci95_high: f64,
    mean_model_action_coverage: f64,
    unsupported_rate_per_model_action: f64,
    mean_probe_count: f64,
    max_probe_count: usize,
    mean_probe_cost: f64,
    revocation_episode_rate: f64,
    false_revocation_episode_rate: f64,
    shift_detection_coverage: f64,
    p90_detection_delay_steps: f64,
    mean_regret_vs_oracle: f64,
    cvar95_regret_vs_oracle: f64,
    max_regret_vs_oracle: f64,
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 }
}

fn mean_ci95(values: &[f64]) -> (f64, f64, f64) {
    let average = mean(values);
    if values.len() < 2 {
        return (average, average, average);
    }
    let variance = values.iter().map(|value| (value - average).powi(2)).sum::<f64>()
        / (values.len() - 1) as f64;
    let standard_error = (variance / values.len() as f64).sqrt();
    (average, average - 1.96 * standard_error, average + 1.96 * standard_error)
}

fn cvar95(values: &[f64]) -> f64 {
    if values.is_empty() { return 0.0; }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| right.total_cmp(left));
    let tail = ((sorted.len() as f64 * 0.05).ceil() as usize).max(1);
    mean(&sorted[..tail])
}

fn summarize(
    scenario: Scenario,
    policy: Policy,
    outcomes: &[EpisodeOutcome],
    passive_outcomes: &[EpisodeOutcome],
    oracle_outcomes: &[EpisodeOutcome],
) -> Summary {
    let utilities: Vec<f64> = outcomes.iter().map(|item| item.cumulative_utility).collect();
    let paired_deltas: Vec<f64> = outcomes.iter().zip(passive_outcomes)
        .map(|(current, passive)| current.cumulative_utility - passive.cumulative_utility).collect();
    let regrets: Vec<f64> = outcomes.iter().zip(oracle_outcomes)
        .map(|(current, oracle)| (oracle.cumulative_utility - current.cumulative_utility).max(0.0))
        .collect();
    let (delta, delta_low, delta_high) = mean_ci95(&paired_deltas);

    let model_actions: usize = outcomes.iter().map(|item| item.model_action_count).sum();
    let unsupported: usize = outcomes.iter().map(|item| item.unsupported_count).sum();
    let probes: usize = outcomes.iter().map(|item| item.probe_count).sum();
    let total_decisions = outcomes.len() * HORIZON;
    let onset = scenario.shift_onset();
    let revocations = outcomes.iter().filter(|item| item.revocation_step.is_some()).count();
    let false_revocations = outcomes.iter().filter(|item| match (onset, item.revocation_step) {
        (None, Some(_)) => policy.has_detector(),
        (Some(change_at), Some(revoked_at)) => policy.has_detector() && revoked_at < change_at,
        _ => false,
    }).count();
    let detected = outcomes.iter().filter(|item| matches!((onset, item.revocation_step),
        (Some(change_at), Some(revoked_at)) if revoked_at >= change_at && revoked_at < HORIZON)).count();
    let delays: Vec<f64> = outcomes.iter().filter_map(|item| match (onset, item.revocation_step) {
        (Some(change_at), Some(revoked_at)) if revoked_at >= change_at && revoked_at < HORIZON =>
            Some((revoked_at - change_at + 1) as f64),
        _ => None,
    }).collect();
    let p90_delay = if delays.is_empty() {
        -1.0
    } else {
        let mut sorted = delays;
        sorted.sort_by(f64::total_cmp);
        sorted[((sorted.len() as f64 * 0.90).ceil() as usize).saturating_sub(1)]
    };

    Summary {
        scenario: scenario.name(),
        policy: policy.name(),
        episode_count: outcomes.len(),
        probe_cap: policy.probe_cap(),
        mean_utility: mean(&utilities),
        mean_utility_delta_vs_passive: delta,
        utility_delta_ci95_low: delta_low,
        utility_delta_ci95_high: delta_high,
        mean_model_action_coverage: model_actions as f64 / total_decisions as f64,
        unsupported_rate_per_model_action: if model_actions == 0 { 0.0 } else { unsupported as f64 / model_actions as f64 },
        mean_probe_count: probes as f64 / outcomes.len() as f64,
        max_probe_count: outcomes.iter().map(|item| item.probe_count).max().unwrap_or(0),
        mean_probe_cost: probes as f64 / outcomes.len() as f64 * PROBE_COST,
        revocation_episode_rate: revocations as f64 / outcomes.len() as f64,
        false_revocation_episode_rate: false_revocations as f64 / outcomes.len() as f64,
        shift_detection_coverage: if onset.is_some() { detected as f64 / outcomes.len() as f64 } else { 0.0 },
        p90_detection_delay_steps: p90_delay,
        mean_regret_vs_oracle: mean(&regrets),
        cvar95_regret_vs_oracle: cvar95(&regrets),
        max_regret_vs_oracle: regrets.iter().copied().fold(0.0_f64, f64::max),
    }
}

fn run_benchmark() -> Vec<Summary> {
    let mut rows = Vec::with_capacity(Scenario::all().len() * Policy::all().len());
    for scenario in Scenario::all() {
        let seeds: Vec<u64> = (0..EPISODES)
            .map(|episode| BASE_SEED ^ (episode as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect();
        let passive: Vec<EpisodeOutcome> = seeds.iter()
            .map(|seed| simulate_episode(scenario, Policy::PassiveCusum, *seed)).collect();
        let oracle: Vec<EpisodeOutcome> = seeds.iter()
            .map(|seed| simulate_episode(scenario, Policy::OracleChangepoint, *seed)).collect();

        for policy in Policy::all() {
            let outcomes: Vec<EpisodeOutcome> = seeds.iter()
                .map(|seed| simulate_episode(scenario, policy, *seed)).collect();
            rows.push(summarize(scenario, policy, &outcomes, &passive, &oracle));
        }
    }
    rows
}

fn render_report(rows: &[Summary]) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"schema_version\": \"{REPORT_SCHEMA}\",\n"));
    out.push_str(&format!("  \"episode_count\": {EPISODES},\n"));
    out.push_str(&format!("  \"horizon\": {HORIZON},\n"));
    out.push_str(&format!("  \"change_detection_threshold\": {CHANGE_DETECTION_THRESHOLD:.4},\n"));
    out.push_str(&format!("  \"probe_cost\": {PROBE_COST:.6},\n"));
    out.push_str(&format!("  \"probe_signal_null_rate\": {NULL_PROBE_FAILURE_RATE:.4},\n"));
    out.push_str(&format!("  \"probe_signal_shift_rate\": {SHIFTED_PROBE_FAILURE_RATE:.4},\n"));
    out.push_str(&format!("  \"adaptive_probe_cap\": {ADAPTIVE_PROBE_CAP},\n"));
    out.push_str("  \"scenarios\": [\"stationary\", \"abrupt_support_shift\", \"gradual_support_shift\"],\n");
    out.push_str("  \"policies\": [\"direct_fallback\", \"fixed_gate\", \"passive_cusum\", \"probe_every_2\", \"probe_every_4\", \"adaptive_probe\", \"zero_info_probe_every_2\", \"oracle_changepoint\"],\n");
    out.push_str("  \"summary\": [\n");
    for (index, row) in rows.iter().enumerate() {
        out.push_str(&format!(
            concat!(
                "    {{\"scenario\": \"{}\", \"policy\": \"{}\", \"episode_count\": {}, \"probe_cap\": {}, ",
                "\"mean_utility\": {:.8}, \"mean_utility_delta_vs_passive\": {:.8}, ",
                "\"utility_delta_ci95_low\": {:.8}, \"utility_delta_ci95_high\": {:.8}, ",
                "\"mean_model_action_coverage\": {:.8}, \"unsupported_rate_per_model_action\": {:.8}, ",
                "\"mean_probe_count\": {:.8}, \"max_probe_count\": {}, \"mean_probe_cost\": {:.8}, ",
                "\"revocation_episode_rate\": {:.8}, \"false_revocation_episode_rate\": {:.8}, ",
                "\"shift_detection_coverage\": {:.8}, \"p90_detection_delay_steps\": {:.4}, ",
                "\"mean_regret_vs_oracle\": {:.8}, \"cvar95_regret_vs_oracle\": {:.8}, \"max_regret_vs_oracle\": {:.8}}}"
            ),
            row.scenario, row.policy, row.episode_count, row.probe_cap, row.mean_utility,
            row.mean_utility_delta_vs_passive, row.utility_delta_ci95_low, row.utility_delta_ci95_high,
            row.mean_model_action_coverage, row.unsupported_rate_per_model_action, row.mean_probe_count,
            row.max_probe_count, row.mean_probe_cost, row.revocation_episode_rate,
            row.false_revocation_episode_rate, row.shift_detection_coverage,
            row.p90_detection_delay_steps, row.mean_regret_vs_oracle, row.cvar95_regret_vs_oracle,
            row.max_regret_vs_oracle
        ));
        if index + 1 != rows.len() { out.push(','); }
        out.push('\n');
    }
    out.push_str("  ]\n}\n");
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("CMAQ-009 active diagnostic-probe benchmark\n\nRuns 512 paired episodes per scenario, 128 decisions per episode.\nPrints deterministic JSON to stdout.");
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
    fn scenario_probabilities_are_preregistered() {
        assert_eq!(Scenario::Stationary.action_unsupported_probability(127), 0.01);
        assert_eq!(Scenario::AbruptSupportShift.action_unsupported_probability(63), 0.01);
        assert_eq!(Scenario::AbruptSupportShift.action_unsupported_probability(64), 0.25);
        assert_eq!(Scenario::AbruptSupportShift.probe_failure_probability(64), 0.60);
        assert_eq!(Scenario::GradualSupportShift.probe_failure_probability(32), 0.01);
        assert_eq!(Scenario::GradualSupportShift.probe_failure_probability(96), 0.60);
    }

    #[test]
    fn likelihood_increments_have_the_expected_sign() {
        assert!(action_likelihood_increment(true) > 0.0);
        assert!(action_likelihood_increment(false) < 0.0);
        assert!(probe_likelihood_increment(true) > 0.0);
        assert!(probe_likelihood_increment(false) < 0.0);
    }

    #[test]
    fn paired_seed_replay_is_deterministic() {
        let left = simulate_episode(Scenario::AbruptSupportShift, Policy::AdaptiveProbe, 123);
        let right = simulate_episode(Scenario::AbruptSupportShift, Policy::AdaptiveProbe, 123);
        assert_eq!(left.cumulative_utility, right.cumulative_utility);
        assert_eq!(left.probe_count, right.probe_count);
        assert_eq!(left.revocation_step, right.revocation_step);
    }

    #[test]
    fn fallback_never_uses_the_model_or_probes() {
        let result = simulate_episode(Scenario::AbruptSupportShift, Policy::DirectFallback, 7);
        assert_eq!(result.model_action_count, 0);
        assert_eq!(result.probe_count, 0);
        assert_eq!(result.cumulative_utility, 0.0);
    }

    #[test]
    fn zero_information_probe_does_not_change_the_passive_decisions() {
        let passive = simulate_episode(Scenario::AbruptSupportShift, Policy::PassiveCusum, 999);
        let control = simulate_episode(Scenario::AbruptSupportShift, Policy::ZeroInformationProbeEvery2, 999);
        assert_eq!(passive.model_action_count, control.model_action_count);
        assert_eq!(passive.unsupported_count, control.unsupported_count);
        assert_eq!(passive.revocation_step, control.revocation_step);
        assert!(control.cumulative_utility < passive.cumulative_utility);
    }

    #[test]
    fn probe_policies_respect_hard_resource_caps() {
        for scenario in Scenario::all() {
            for policy in Policy::all() {
                let result = simulate_episode(scenario, policy, 77);
                assert!(result.probe_count <= policy.probe_cap());
            }
        }
    }

    #[test]
    fn empirical_tail_regret_is_not_below_mean_regret() {
        let seeds: Vec<u64> = (0..EPISODES as u64).collect();
        let passive: Vec<EpisodeOutcome> = seeds.iter()
            .map(|seed| simulate_episode(Scenario::AbruptSupportShift, Policy::PassiveCusum, *seed)).collect();
        let oracle: Vec<EpisodeOutcome> = seeds.iter()
            .map(|seed| simulate_episode(Scenario::AbruptSupportShift, Policy::OracleChangepoint, *seed)).collect();
        let row = summarize(Scenario::AbruptSupportShift, Policy::PassiveCusum, &passive, &passive, &oracle);
        assert!(row.cvar95_regret_vs_oracle >= row.mean_regret_vs_oracle - 1e-9);
        assert!(row.max_regret_vs_oracle >= row.cvar95_regret_vs_oracle - 1e-9);
    }

    #[test]
    fn benchmark_report_has_one_row_per_scenario_policy_pair() {
        let rows = run_benchmark();
        assert_eq!(rows.len(), Scenario::all().len() * Policy::all().len());
        let rendered = render_report(&rows);
        assert!(rendered.contains("\"schema_version\": \"cmaq-009-active-diagnostic-probes-v0.1\""));
        assert!(rendered.contains("\"episode_count\": 512"));
        assert!(rendered.contains("\"horizon\": 128"));
    }
}
