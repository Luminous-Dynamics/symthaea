use std::collections::VecDeque;
use std::fmt::Write as _;

const SCHEMA: &str = "cmaq-013-selection-conditioned-calibration-v0.1";
const EPISODES: usize = 512;
const HORIZON: usize = 96;
const SEARCH_WIDTHS: [usize; 3] = [8, 32, 128];
const CALIBRATION_EPISODES: usize = 2048;
const TARGET_COVERAGE: f64 = 0.90;
const SAFE_UTILITY_THRESHOLD: f64 = 0.52;
const FALLBACK_UTILITY: f64 = 0.48;
const FEEDBACK_DELAY: usize = 4;
const FEEDBACK_DROP_RATE: f64 = 0.05;
const AUDIT_PERIOD: usize = 8;
const AUDIT_COST: f64 = 0.002;
const ACI_STEP_SIZE: f64 = 0.05;
const ROLLING_WINDOW: usize = 128;
const ROLLING_BURN_IN: usize = 16;
const BOOTSTRAP_REPLICATES: usize = 2000;
const BOOTSTRAP_SEED: u64 = 0xC0A9_0013_2026_1030;
const BASE_SEED: u64 = 0xC0A9_0013_2026_1031;

#[derive(Clone, Copy, Debug)]
struct SplitMix64 { state: u64 }

impl SplitMix64 {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64) }
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario { Stationary, AbruptErrorShift, GradualErrorShift }

impl Scenario {
    fn all() -> [Self; 3] { [Self::Stationary, Self::AbruptErrorShift, Self::GradualErrorShift] }
    fn name(self) -> &'static str {
        match self {
            Self::Stationary => "stationary",
            Self::AbruptErrorShift => "abrupt_model_error_shift",
            Self::GradualErrorShift => "gradual_model_error_shift",
        }
    }
    fn factor(self, step: usize) -> f64 {
        match self {
            Self::Stationary => 1.0,
            Self::AbruptErrorShift => if step < HORIZON / 2 { 1.0 } else { 2.5 },
            Self::GradualErrorShift => {
                let start = HORIZON / 4;
                let end = (3 * HORIZON) / 4;
                if step < start { 1.0 }
                else if step >= end { 2.5 }
                else { 1.0 + (step - start) as f64 / (end - start) as f64 * 1.5 }
            }
        }
    }
    fn shift_onset(self) -> Option<usize> {
        match self {
            Self::Stationary => None,
            Self::AbruptErrorShift => Some(HORIZON / 2),
            Self::GradualErrorShift => Some(HORIZON / 4),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    PointNoGuard,
    GaussianFixed,
    BehaviorSplitConformal,
    SelectionSplitConformal,
    AciSelectiveFeedback,
    AciWithAudit,
    RollingSelectiveFeedback,
    OracleSelectedQuantile,
}

impl Policy {
    fn all() -> [Self; 8] {
        [
            Self::PointNoGuard,
            Self::GaussianFixed,
            Self::BehaviorSplitConformal,
            Self::SelectionSplitConformal,
            Self::AciSelectiveFeedback,
            Self::AciWithAudit,
            Self::RollingSelectiveFeedback,
            Self::OracleSelectedQuantile,
        ]
    }
    fn name(self) -> &'static str {
        match self {
            Self::PointNoGuard => "point_no_guard",
            Self::GaussianFixed => "gaussian_fixed_bound",
            Self::BehaviorSplitConformal => "behavior_split_conformal",
            Self::SelectionSplitConformal => "selection_conditioned_split_conformal",
            Self::AciSelectiveFeedback => "aci_style_selective_feedback",
            Self::AciWithAudit => "aci_style_with_counterfactual_audit",
            Self::RollingSelectiveFeedback => "rolling_quantile_selective_feedback",
            Self::OracleSelectedQuantile => "oracle_selected_quantile",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    truth: f64,
    predicted: f64,
    sigma_reported: f64,
    normalized_error: f64,
}

#[derive(Clone, Debug)]
struct Calibration {
    behavior_q: f64,
    selected_q: f64,
    oracle_q_by_factor: Vec<(f64, f64)>,
}

impl Calibration {
    fn oracle_q(&self, factor: f64) -> f64 {
        let mut best_q = self.selected_q;
        let mut best_distance = f64::INFINITY;
        for (candidate_factor, candidate_q) in &self.oracle_q_by_factor {
            let distance = (*candidate_factor - factor).abs();
            if distance < best_distance {
                best_distance = distance;
                best_q = *candidate_q;
            }
        }
        best_q
    }
}

#[derive(Clone, Copy, Debug)]
struct Feedback {
    release_step: usize,
    normalized_error: f64,
    bound_q_at_prediction: f64,
}

#[derive(Clone, Debug, Default)]
struct EpisodeStats {
    utility: f64,
    regret: f64,
    decisions: usize,
    accepted: usize,
    fallbacks: usize,
    covered: usize,
    pre_shift_decisions: usize,
    pre_shift_covered: usize,
    post_shift_decisions: usize,
    post_shift_covered: usize,
    false_safe: usize,
    audit_count: usize,
    feedback_count: usize,
    feedback_dropped: usize,
    bound_width_sum: f64,
    q_sum: f64,
    selected_overprediction_sum: f64,
    coverage_tape: Vec<bool>,
}

#[derive(Clone, Debug)]
struct Summary {
    scenario: &'static str,
    search_width: usize,
    policy: &'static str,
    episode_count: usize,
    calibration_sample_count: usize,
    behavior_q: f64,
    selected_q: f64,
    mean_utility: f64,
    utility_delta_vs_point: f64,
    utility_ci95_low: f64,
    utility_ci95_high: f64,
    mean_regret: f64,
    selected_action_coverage: f64,
    pre_shift_coverage: f64,
    post_shift_coverage: f64,
    mean_bound_width: f64,
    false_safe_rate: f64,
    fallback_rate: f64,
    mean_audits: f64,
    feedback_observations_per_episode: f64,
    feedback_drop_rate: f64,
    calibration_recovery_delay: f64,
    mean_selected_overprediction: f64,
}

fn make_candidate(rng: &mut SplitMix64, factor: f64) -> Candidate {
    let skill = rng.unit();
    let complexity = rng.unit();
    let truth = 0.46 + 0.15 * skill - 0.012 * complexity;
    let sigma_true = (0.010 + 0.040 * complexity) * factor;
    let sigma_reported = 0.012 + 0.020 * complexity;
    let error = sigma_true * rng.normal();
    Candidate {
        truth,
        predicted: truth + error,
        sigma_reported,
        normalized_error: error / sigma_reported,
    }
}

fn make_pool(rng: &mut SplitMix64, width: usize, factor: f64) -> Vec<Candidate> {
    (0..width).map(|_| make_candidate(rng, factor)).collect()
}

#[derive(Clone, Debug)]
struct EpisodeTape {
    pools: Vec<Vec<Candidate>>,
    feedback_dropped: Vec<bool>,
}

fn make_episode_tape(scenario: Scenario, width: usize, episode: usize) -> EpisodeTape {
    let seed = BASE_SEED
        ^ ((scenario as u64) << 48)
        ^ ((width as u64) << 24)
        ^ (episode as u64).wrapping_mul(0xD134_2543_DE82_EF95);
    let mut rng = SplitMix64::new(seed);
    let mut feedback_rng = SplitMix64::new(seed ^ 0xFEE1_DEAD);
    let mut pools = Vec::with_capacity(HORIZON);
    let mut feedback_dropped = Vec::with_capacity(HORIZON);
    for step in 0..HORIZON {
        pools.push(make_pool(&mut rng, width, scenario.factor(step)));
        feedback_dropped.push(feedback_rng.unit() < FEEDBACK_DROP_RATE);
    }
    EpisodeTape { pools, feedback_dropped }
}

fn selected_candidate(pool: &[Candidate]) -> Candidate {
    *pool.iter().max_by(|a, b| a.predicted.total_cmp(&b.predicted)).expect("candidate pool is non-empty")
}

fn finite_sample_quantile(values: &mut [f64], coverage: f64) -> f64 {
    if values.is_empty() { return 0.0; }
    values.sort_by(f64::total_cmp);
    let rank = (((values.len() + 1) as f64) * coverage).ceil() as usize;
    values[rank.saturating_sub(1).min(values.len() - 1)]
}

fn calibrate(width: usize) -> Calibration {
    let mut behavior = Vec::with_capacity(CALIBRATION_EPISODES);
    let mut selected = Vec::with_capacity(CALIBRATION_EPISODES);
    let mut behavior_rng = SplitMix64::new(BASE_SEED ^ (width as u64) ^ 0xBEEF);
    let mut selected_rng = SplitMix64::new(BASE_SEED ^ (width as u64) ^ 0xCAFE);

    for _ in 0..CALIBRATION_EPISODES {
        let random_candidate = make_candidate(&mut behavior_rng, 1.0);
        behavior.push(random_candidate.normalized_error);

        let pool = make_pool(&mut selected_rng, width, 1.0);
        selected.push(selected_candidate(&pool).normalized_error);
    }

    let behavior_q = finite_sample_quantile(&mut behavior, TARGET_COVERAGE);
    let selected_q = finite_sample_quantile(&mut selected, TARGET_COVERAGE);

    // Evaluator-only oracle calibration for the selected-action distribution at
    // each error scale. This is a diagnostic ceiling, not a deployable policy.
    let mut oracle_q_by_factor = Vec::new();
    for bucket in 0..=15 {
        let factor = 1.0 + bucket as f64 * 0.1;
        let mut ratios = Vec::with_capacity(CALIBRATION_EPISODES);
        let mut rng = SplitMix64::new(BASE_SEED ^ (width as u64) ^ ((bucket as u64) << 32) ^ 0x0A11CE);
        for _ in 0..CALIBRATION_EPISODES {
            let pool = make_pool(&mut rng, width, factor);
            ratios.push(selected_candidate(&pool).normalized_error);
        }
        oracle_q_by_factor.push((factor, finite_sample_quantile(&mut ratios, TARGET_COVERAGE)));
    }

    Calibration { behavior_q, selected_q, oracle_q_by_factor }
}

fn initial_q(policy: Policy, calibration: &Calibration) -> f64 {
    match policy {
        Policy::PointNoGuard => 0.0,
        Policy::GaussianFixed => 1.2815515655446004,
        Policy::BehaviorSplitConformal | Policy::AciSelectiveFeedback
        | Policy::AciWithAudit | Policy::RollingSelectiveFeedback => calibration.behavior_q,
        Policy::SelectionSplitConformal => calibration.selected_q,
        Policy::OracleSelectedQuantile => calibration.selected_q,
    }
}

fn update_aci(q: f64, normalized_error: f64, q_used: f64) -> f64 {
    let miscovered = if normalized_error > q_used { 1.0 } else { 0.0 };
    (q + ACI_STEP_SIZE * (miscovered - (1.0 - TARGET_COVERAGE))).clamp(0.0, 20.0)
}

fn rolling_quantile(history: &VecDeque<f64>, fallback: f64) -> f64 {
    if history.len() < ROLLING_BURN_IN { return fallback; }
    let mut values: Vec<f64> = history.iter().copied().collect();
    finite_sample_quantile(&mut values, TARGET_COVERAGE)
}

fn recovery_delay(coverage_tape: &[bool], shift_onset: Option<usize>) -> f64 {
    let onset = match shift_onset { Some(value) => value, None => return -1.0 };
    for end in (onset + 15)..coverage_tape.len() {
        let window = &coverage_tape[(end + 1 - 16)..=end];
        let count = window.iter().filter(|covered| **covered).count();
        if count as f64 / 16.0 >= TARGET_COVERAGE {
            return (end - onset) as f64;
        }
    }
    -1.0
}

fn simulate_episode(
    scenario: Scenario,
    policy: Policy,
    tape: &EpisodeTape,
    calibration: &Calibration,
) -> EpisodeStats {
    let mut q = initial_q(policy, calibration);
    let mut pending = VecDeque::<Feedback>::new();
    let mut rolling = VecDeque::<f64>::new();
    let mut stats = EpisodeStats::default();

    for step in 0..HORIZON {
        while let Some(feedback) = pending.front().copied() {
            if feedback.release_step > step { break; }
            pending.pop_front();
            if policy == Policy::AciSelectiveFeedback || policy == Policy::AciWithAudit {
                q = update_aci(q, feedback.normalized_error, feedback.bound_q_at_prediction);
            } else if policy == Policy::RollingSelectiveFeedback {
                rolling.push_back(feedback.normalized_error);
                while rolling.len() > ROLLING_WINDOW { rolling.pop_front(); }
                q = rolling_quantile(&rolling, calibration.behavior_q);
            }
        }

        let pool = &tape.pools[step];
        let selected = selected_candidate(pool);
        let q_used = match policy {
            Policy::OracleSelectedQuantile => calibration.oracle_q(factor),
            _ => q,
        };
        let lower_bound = selected.predicted - q_used * selected.sigma_reported;
        let accepted = policy == Policy::PointNoGuard || lower_bound >= SAFE_UTILITY_THRESHOLD;
        let shift = scenario.shift_onset();
        let covered = selected.truth >= lower_bound;
        let realized = if accepted { selected.truth } else { FALLBACK_UTILITY };
        let best_truth = pool.iter().map(|candidate| candidate.truth).max_by(f64::total_cmp).unwrap_or(selected.truth);

        stats.utility += realized;
        stats.regret += (best_truth - realized).max(0.0);
        stats.decisions += 1;
        if accepted { stats.accepted += 1; } else { stats.fallbacks += 1; }
        if covered { stats.covered += 1; }
        if accepted && selected.truth < SAFE_UTILITY_THRESHOLD { stats.false_safe += 1; }
        if let Some(onset) = shift {
            if step < onset {
                stats.pre_shift_decisions += 1;
                if covered { stats.pre_shift_covered += 1; }
            } else {
                stats.post_shift_decisions += 1;
                if covered { stats.post_shift_covered += 1; }
            }
        } else {
            stats.pre_shift_decisions += 1;
            if covered { stats.pre_shift_covered += 1; }
        }
        stats.bound_width_sum += q_used * selected.sigma_reported;
        stats.q_sum += q_used;
        stats.selected_overprediction_sum += selected.predicted - selected.truth;
        stats.coverage_tape.push(covered);

        // Labels from accepted actions arrive after a fixed delay and may be lost.
        // ACI/rolling updates cannot see rejected actions unless the explicit audit
        // arm purchases a counterfactual evaluator query.
        let accepted_feedback = accepted;
        let audit_feedback = policy == Policy::AciWithAudit && !accepted && step % AUDIT_PERIOD == 0;
        if audit_feedback {
            stats.audit_count += 1;
        }
        if accepted_feedback || audit_feedback {
            if !tape.feedback_dropped[step] {
                pending.push_back(Feedback {
                    release_step: step + FEEDBACK_DELAY,
                    normalized_error: selected.normalized_error,
                    bound_q_at_prediction: q_used,
                });
                stats.feedback_count += 1;
            } else {
                stats.feedback_dropped += 1;
            }
        }
        if audit_feedback {
            stats.utility -= AUDIT_COST;
        }
    }

    stats
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 }
}

fn percentile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() { return 0.0; }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * q).ceil() as usize).saturating_sub(1).min(sorted.len() - 1);
    sorted[index]
}

fn bootstrap_interval(values: &[f64], seed: u64) -> (f64, f64) {
    if values.is_empty() { return (0.0, 0.0); }
    let mut rng = SplitMix64::new(seed);
    let mut boot = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    for _ in 0..BOOTSTRAP_REPLICATES {
        let mut total = 0.0;
        for _ in 0..values.len() {
            total += values[rng.next_u64() as usize % values.len()];
        }
        boot.push(total / values.len() as f64);
    }
    (percentile(&boot, 0.025), percentile(&boot, 0.975))
}

fn summarize(
    scenario: Scenario,
    width: usize,
    policy: Policy,
    calibration: &Calibration,
    outcomes: &[EpisodeStats],
    point_outcomes: &[EpisodeStats],
) -> Summary {
    let n = outcomes.len();
    let utilities: Vec<f64> = outcomes.iter().map(|item| item.utility / HORIZON as f64).collect();
    let point_utilities: Vec<f64> = point_outcomes.iter().map(|item| item.utility / HORIZON as f64).collect();
    let paired_delta: Vec<f64> = utilities.iter().zip(point_utilities.iter()).map(|(candidate, baseline)| candidate - baseline).collect();
    let (ci_low, ci_high) = bootstrap_interval(&paired_delta, BOOTSTRAP_SEED ^ ((scenario as u64) << 32) ^ width as u64 ^ ((policy as u64) << 16));

    let total_decisions = outcomes.iter().map(|item| item.decisions).sum::<usize>() as f64;
    let total_accepted = outcomes.iter().map(|item| item.accepted).sum::<usize>() as f64;
    let total_false_safe = outcomes.iter().map(|item| item.false_safe).sum::<usize>() as f64;
    let total_covered = outcomes.iter().map(|item| item.covered).sum::<usize>() as f64;
    let pre_decisions = outcomes.iter().map(|item| item.pre_shift_decisions).sum::<usize>() as f64;
    let pre_covered = outcomes.iter().map(|item| item.pre_shift_covered).sum::<usize>() as f64;
    let post_decisions = outcomes.iter().map(|item| item.post_shift_decisions).sum::<usize>() as f64;
    let post_covered = outcomes.iter().map(|item| item.post_shift_covered).sum::<usize>() as f64;
    let total_audits = outcomes.iter().map(|item| item.audit_count).sum::<usize>() as f64;
    let total_feedback = outcomes.iter().map(|item| item.feedback_count).sum::<usize>() as f64;
    let total_dropped = outcomes.iter().map(|item| item.feedback_dropped).sum::<usize>() as f64;
    let total_width = outcomes.iter().map(|item| item.bound_width_sum).sum::<f64>();
    let total_q = outcomes.iter().map(|item| item.q_sum).sum::<f64>();
    let total_regret = outcomes.iter().map(|item| item.regret).sum::<f64>();
    let total_overprediction = outcomes.iter().map(|item| item.selected_overprediction_sum).sum::<f64>();
    let total_recovery = outcomes.iter().map(|item| recovery_delay(&item.coverage_tape, scenario.shift_onset())).filter(|delay| *delay >= 0.0).collect::<Vec<_>>();

    Summary {
        scenario: scenario.name(),
        search_width: width,
        policy: policy.name(),
        episode_count: n,
        calibration_sample_count: CALIBRATION_EPISODES,
        behavior_q: calibration.behavior_q,
        selected_q: calibration.selected_q,
        mean_utility: mean(&utilities),
        utility_delta_vs_point: mean(&paired_delta),
        utility_ci95_low: ci_low,
        utility_ci95_high: ci_high,
        mean_regret: if total_decisions > 0.0 { total_regret / total_decisions } else { 0.0 },
        selected_action_coverage: if total_decisions > 0.0 { total_covered / total_decisions } else { 0.0 },
        pre_shift_coverage: if pre_decisions > 0.0 { pre_covered / pre_decisions } else { -1.0 },
        post_shift_coverage: if post_decisions > 0.0 { post_covered / post_decisions } else { -1.0 },
        mean_bound_width: if total_decisions > 0.0 { total_width / total_decisions } else { 0.0 },
        false_safe_rate: if total_accepted > 0.0 { total_false_safe / total_accepted } else { 0.0 },
        fallback_rate: if total_decisions > 0.0 { 1.0 - total_accepted / total_decisions } else { 0.0 },
        mean_audits: if n > 0 { total_audits / n as f64 } else { 0.0 },
        feedback_observations_per_episode: if n > 0 { total_feedback / n as f64 } else { 0.0 },
        feedback_drop_rate: if total_feedback + total_dropped > 0.0 { total_dropped / (total_feedback + total_dropped) } else { 0.0 },
        calibration_recovery_delay: if total_recovery.is_empty() { -1.0 } else { mean(&total_recovery) },
        mean_selected_overprediction: if total_decisions > 0.0 { total_overprediction / total_decisions } else { 0.0 },
    }
}

fn run_benchmark() -> Vec<Summary> {
    let calibrations: Vec<(usize, Calibration)> = SEARCH_WIDTHS
        .iter()
        .map(|width| (*width, calibrate(*width)))
        .collect();
    let mut rows = Vec::new();
    for scenario in Scenario::all() {
        for (width, calibration) in &calibrations {
            // Generate each candidate/feedback tape once and share it across
            // all policies. This guarantees exact pairing and avoids wasting
            // compute regenerating identical simulator outcomes.
            let mut outcomes_by_policy: Vec<Vec<EpisodeStats>> = Policy::all().iter()
                .map(|_| Vec::with_capacity(EPISODES))
                .collect();
            for episode in 0..EPISODES {
                let tape = make_episode_tape(scenario, *width, episode);
                for (index, policy) in Policy::all().iter().enumerate() {
                    outcomes_by_policy[index].push(
                        simulate_episode(scenario, *policy, &tape, calibration)
                    );
                }
            }
            let point = &outcomes_by_policy[0];
            for (index, policy) in Policy::all().iter().enumerate() {
                rows.push(summarize(
                    scenario, *width, *policy, calibration,
                    &outcomes_by_policy[index], point
                ));
            }
        }
    }
    rows
}

fn number(value: f64) -> String {
    if value.is_finite() { format!("{value:.8}") } else { "null".to_string() }
}

fn report_json(rows: &[Summary]) -> String {
    let scenarios = Scenario::all().iter().map(|s| format!("\"{}\"", s.name())).collect::<Vec<_>>().join(",");
    let policies = Policy::all().iter().map(|p| format!("\"{}\"", p.name())).collect::<Vec<_>>().join(",");
    let mut out = String::new();
    writeln!(&mut out, "{{").unwrap();
    writeln!(&mut out, "  \"schema_version\":\"{SCHEMA}\",").unwrap();
    writeln!(&mut out, "  \"episode_count\":{EPISODES},").unwrap();
    writeln!(&mut out, "  \"horizon\":{HORIZON},").unwrap();
    writeln!(&mut out, "  \"calibration_episodes\":{CALIBRATION_EPISODES},").unwrap();
    writeln!(&mut out, "  \"target_coverage\":{},", number(TARGET_COVERAGE)).unwrap();
    writeln!(&mut out, "  \"feedback_delay_steps\":{FEEDBACK_DELAY},").unwrap();
    writeln!(&mut out, "  \"feedback_drop_rate\":{},", number(FEEDBACK_DROP_RATE)).unwrap();
    writeln!(&mut out, "  \"audit_period_steps\":{AUDIT_PERIOD},").unwrap();
    writeln!(&mut out, "  \"audit_cost\":{},", number(AUDIT_COST)).unwrap();
    writeln!(&mut out, "  \"scenarios\":[{scenarios}],").unwrap();
    writeln!(&mut out, "  \"search_widths\":[8,32,128],").unwrap();
    writeln!(&mut out, "  \"policies\":[{policies}],").unwrap();
    writeln!(&mut out, "  \"summary\":[").unwrap();
    for (index, row) in rows.iter().enumerate() {
        let comma = if index + 1 == rows.len() { "" } else { "," };
        writeln!(&mut out,
            concat!(
                "    {{\"scenario\":\"{}\",\"search_width\":{},\"policy\":\"{}\",",
                "\"episode_count\":{},\"calibration_sample_count\":{},",
                "\"behavior_q\":{},\"selection_q\":{},",
                "\"mean_utility\":{},\"utility_delta_vs_point\":{},\"utility_ci95_low\":{},\"utility_ci95_high\":{},",
                "\"mean_regret\":{},\"selected_action_coverage\":{},\"pre_shift_coverage\":{},\"post_shift_coverage\":{},",
                "\"mean_bound_width\":{},\"false_safe_rate\":{},\"fallback_rate\":{},",
                "\"mean_audits\":{},\"feedback_observations_per_episode\":{},\"feedback_drop_rate\":{},",
                "\"calibration_recovery_delay\":{},\"mean_selected_overprediction\":{}}}{}"
            ),
            row.scenario, row.search_width, row.policy, row.episode_count, row.calibration_sample_count,
            number(row.behavior_q), number(row.selected_q),
            number(row.mean_utility), number(row.utility_delta_vs_point), number(row.utility_ci95_low), number(row.utility_ci95_high),
            number(row.mean_regret), number(row.selected_action_coverage), number(row.pre_shift_coverage), number(row.post_shift_coverage),
            number(row.mean_bound_width), number(row.false_safe_rate), number(row.fallback_rate),
            number(row.mean_audits), number(row.feedback_observations_per_episode), number(row.feedback_drop_rate),
            number(row.calibration_recovery_delay), number(row.mean_selected_overprediction), comma
        ).unwrap();
    }
    writeln!(&mut out, "  ]").unwrap();
    writeln!(&mut out, "}}").unwrap();
    out
}

fn main() {
    print!("{}", report_json(&run_benchmark()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_rng_and_candidate_generation() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        assert_eq!(a.next_u64(), b.next_u64());
        assert_eq!(make_candidate(&mut a, 1.0).truth.to_bits(), make_candidate(&mut b, 1.0).truth.to_bits());
    }

    #[test]
    fn policy_induced_selection_increases_upper_tail_error() {
        let calibration = calibrate(128);
        assert!(calibration.selected_q > calibration.behavior_q);
    }

    #[test]
    fn abrupt_and_gradual_shift_factors_are_preregistered() {
        assert_eq!(Scenario::Stationary.factor(95), 1.0);
        assert_eq!(Scenario::AbruptErrorShift.factor(47), 1.0);
        assert_eq!(Scenario::AbruptErrorShift.factor(48), 2.5);
        assert_eq!(Scenario::GradualErrorShift.factor(24), 1.0);
        assert_eq!(Scenario::GradualErrorShift.factor(72), 2.5);
    }

    #[test]
    fn aci_update_increases_on_miscoverage_and_decreases_on_coverage() {
        let q = 2.0;
        assert!(update_aci(q, 3.0, q) > q);
        assert!(update_aci(q, 1.0, q) < q);
    }

    #[test]
    fn rolling_quantile_waits_for_burn_in() {
        let history = VecDeque::from([1.0, 2.0, 3.0]);
        assert_eq!(rolling_quantile(&history, 4.0), 4.0);
    }

    #[test]
    fn audit_arm_is_explicitly_costed_and_periodic() {
        assert_eq!(AUDIT_PERIOD, 8);
        assert!(AUDIT_COST > 0.0);
        let mut stats = EpisodeStats::default();
        stats.audit_count += 1;
        stats.utility -= AUDIT_COST;
        assert!(stats.utility < 0.0);
    }

    #[test]
    fn recovery_delay_is_undefined_without_a_shift() {
        assert_eq!(recovery_delay(&vec![true; HORIZON], None), -1.0);
    }

    #[test]
    fn report_has_72_scenario_width_policy_rows() {
        assert_eq!(Scenario::all().len() * SEARCH_WIDTHS.len() * Policy::all().len(), 72);
    }
}
