use std::fmt::Write as _;

const SCHEMA: &str = "cmaq-012-safe-dual-control-v0.1";
const EPISODES: usize = 256;
const HORIZON: usize = 64;
const TARGET_SWITCH: usize = 24;
const PARAMETER_SHIFT: usize = 44;
const A: f64 = 0.75;
const PROCESS_NOISE_BOUND: f64 = 0.02;
const RESPONSE_NOISE_STD: f64 = 0.15;
const PROBE_NOISE_STD: f64 = 0.08;
const CONTROL_EFFORT: f64 = 0.04;
const STATE_LIMIT: f64 = 0.60;
const MISSION_DEGRADATION_BUDGET: f64 = 0.35;
const INFO_WEIGHT: f64 = 0.12;
const FORGETTING: f64 = 0.995;
const PROBE_COST: f64 = 0.002;
const PROBE_PERIOD: usize = 8;
const BASE_SEED: u64 = 0xC0A0_0012_2026_1030;
const GAINS: [f64; 5] = [0.50, 0.75, 1.00, 1.25, 1.50];
const ACTIONS: [f64; 11] = [-0.50, -0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50];

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self { Self { state: seed } }

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

    fn index(&mut self, upper: usize) -> usize {
        ((self.next_u64() as u128 * upper as u128) >> 64) as usize
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    StationaryUnknownGain,
    AbruptGainShift,
}

impl Scenario {
    fn all() -> [Self; 2] { [Self::StationaryUnknownGain, Self::AbruptGainShift] }

    fn name(self) -> &'static str {
        match self {
            Self::StationaryUnknownGain => "stationary_unknown_gain",
            Self::AbruptGainShift => "abrupt_gain_shift",
        }
    }

    fn gain_at(self, tape: &EpisodeTape, step: usize) -> f64 {
        match self {
            Self::StationaryUnknownGain => tape.initial_gain,
            Self::AbruptGainShift => {
                if step < PARAMETER_SHIFT { tape.initial_gain } else { tape.shifted_gain }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    CertaintyEquivalent,
    ShieldedCertaintyEquivalent,
    WorstCaseRobust,
    UnconstrainedInfoBonus,
    BudgetedDualControl,
    SeparatedSensorProbe,
    OracleKnownGain,
}

impl Policy {
    fn all() -> [Self; 7] {
        [
            Self::CertaintyEquivalent,
            Self::ShieldedCertaintyEquivalent,
            Self::WorstCaseRobust,
            Self::UnconstrainedInfoBonus,
            Self::BudgetedDualControl,
            Self::SeparatedSensorProbe,
            Self::OracleKnownGain,
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::CertaintyEquivalent => "certainty_equivalent",
            Self::ShieldedCertaintyEquivalent => "shielded_certainty_equivalent",
            Self::WorstCaseRobust => "worst_case_robust",
            Self::UnconstrainedInfoBonus => "unconstrained_information_bonus",
            Self::BudgetedDualControl => "budgeted_dual_control",
            Self::SeparatedSensorProbe => "separated_sensor_probe_reference",
            Self::OracleKnownGain => "oracle_known_gain",
        }
    }
}

#[derive(Clone, Debug)]
struct EpisodeTape {
    initial_gain: f64,
    shifted_gain: f64,
    process_noise: Vec<f64>,
    response_noise: Vec<f64>,
    probe_noise: Vec<f64>,
}

impl EpisodeTape {
    fn new(scenario: Scenario, episode: usize) -> Self {
        let mut rng = SplitMix64::new(
            BASE_SEED
                ^ ((scenario as u64) << 48)
                ^ (episode as u64).wrapping_mul(0xD134_2543_DE82_EF95),
        );
        let initial_gain = GAINS[rng.index(GAINS.len())];
        let mut shifted_gain = GAINS[rng.index(GAINS.len())];
        while shifted_gain == initial_gain {
            shifted_gain = GAINS[rng.index(GAINS.len())];
        }
        let process_noise = (0..HORIZON)
            .map(|_| (2.0 * rng.unit() - 1.0) * PROCESS_NOISE_BOUND)
            .collect();
        let response_noise = (0..HORIZON).map(|_| rng.normal()).collect();
        let probe_noise = (0..HORIZON).map(|_| rng.normal()).collect();
        Self { initial_gain, shifted_gain, process_noise, response_noise, probe_noise }
    }
}

#[derive(Clone, Debug)]
struct Outcome {
    total_cost: f64,
    violation_steps: usize,
    max_abs_state: f64,
    final_entropy: f64,
    budget_spent: f64,
    probe_count: usize,
    probe_cost: f64,
    action_effort: f64,
}

#[derive(Clone, Debug)]
struct Summary {
    scenario: &'static str,
    policy: &'static str,
    mean_cost: f64,
    mean_cost_delta_vs_shielded_ce: f64,
    ci95_low: f64,
    ci95_high: f64,
    p95_cost: f64,
    cvar95_cost: f64,
    violation_episode_rate: f64,
    violation_step_rate: f64,
    mean_max_abs_state: f64,
    mean_information_gain: f64,
    mean_budget_spent: f64,
    max_budget_spent: f64,
    mean_probe_count: f64,
    max_probe_count: usize,
    mean_probe_cost: f64,
    mean_action_effort: f64,
}

fn target_at(step: usize) -> f64 {
    if step < TARGET_SWITCH { 0.0 } else { 0.40 }
}

fn entropy(belief: &[f64; 5]) -> f64 {
    belief.iter().filter(|p| **p > 0.0).map(|p| -p * p.ln()).sum()
}

fn normalize(belief: &mut [f64; 5]) {
    let total: f64 = belief.iter().sum();
    if total > 1e-300 {
        for p in belief { *p /= total; }
    } else {
        *belief = [0.20; 5];
    }
}

fn temper_prior(belief: &[f64; 5]) -> [f64; 5] {
    let mut prior = [0.0; 5];
    for i in 0..5 { prior[i] = belief[i].max(1e-12).powf(FORGETTING); }
    normalize(&mut prior);
    prior
}

fn update_transition_belief(belief: &mut [f64; 5], action: f64, observation: f64) {
    let mut posterior = temper_prior(belief);
    if action.abs() < 1e-12 {
        *belief = posterior;
        return;
    }
    for i in 0..5 {
        let residual = observation - GAINS[i] * action;
        let exponent = -0.5 * residual * residual / (RESPONSE_NOISE_STD * RESPONSE_NOISE_STD);
        posterior[i] *= exponent.exp();
    }
    normalize(&mut posterior);
    *belief = posterior;
}

fn update_sensor_belief(belief: &mut [f64; 5], observation: f64) {
    let mut posterior = temper_prior(belief);
    for i in 0..5 {
        let residual = observation - GAINS[i];
        let exponent = -0.5 * residual * residual / (PROBE_NOISE_STD * PROBE_NOISE_STD);
        posterior[i] *= exponent.exp();
    }
    normalize(&mut posterior);
    *belief = posterior;
}

fn belief_moments(belief: &[f64; 5]) -> (f64, f64) {
    let mean = (0..5).map(|i| belief[i] * GAINS[i]).sum::<f64>();
    let variance = (0..5).map(|i| belief[i] * (GAINS[i] - mean).powi(2)).sum::<f64>();
    (mean, variance)
}

fn expected_stage_cost(state: f64, action: f64, belief: &[f64; 5], target: f64) -> f64 {
    let mut cost = 0.0;
    for i in 0..5 {
        let next = A * state + GAINS[i] * action;
        cost += belief[i] * ((next - target).powi(2) + CONTROL_EFFORT * action * action);
    }
    cost
}

fn worst_case_stage_cost(state: f64, action: f64, target: f64) -> f64 {
    GAINS.iter().map(|gain| {
        let next = A * state + gain * action;
        (next - target).powi(2) + CONTROL_EFFORT * action * action
    }).fold(0.0, f64::max)
}

fn robustly_safe(state: f64, action: f64) -> bool {
    GAINS.iter().all(|gain| {
        (A * state + gain * action).abs() + PROCESS_NOISE_BOUND <= STATE_LIMIT
    })
}

fn certainty_equivalent_action(state: f64, belief: &[f64; 5], target: f64) -> f64 {
    let mut best_action = ACTIONS[0];
    let mut best_cost = f64::INFINITY;
    for action in ACTIONS {
        let cost = expected_stage_cost(state, action, belief, target);
        if cost < best_cost {
            best_cost = cost;
            best_action = action;
        }
    }
    best_action
}

fn information_proxy(belief: &[f64; 5], action: f64) -> f64 {
    let (_, variance) = belief_moments(belief);
    0.5 * (1.0 + variance * action * action
        / (RESPONSE_NOISE_STD * RESPONSE_NOISE_STD)).ln()
}

fn safe_ce_action(state: f64, belief: &[f64; 5], target: f64) -> f64 {
    ACTIONS.iter().copied()
        .filter(|action| robustly_safe(state, *action))
        .min_by(|a, b| expected_stage_cost(state, *a, belief, target)
            .total_cmp(&expected_stage_cost(state, *b, belief, target)))
        .unwrap_or(0.0)
}

fn select_action(
    policy: Policy,
    state: f64,
    belief: &[f64; 5],
    target: f64,
    true_gain: f64,
    remaining_budget: f64,
) -> (f64, f64) {
    let ce = certainty_equivalent_action(state, belief, target);
    let ce_cost = expected_stage_cost(state, ce, belief, target);

    match policy {
        Policy::CertaintyEquivalent => (ce, 0.0),
        Policy::ShieldedCertaintyEquivalent => (safe_ce_action(state, belief, target), 0.0),
        Policy::WorstCaseRobust => {
            let action = ACTIONS.iter().copied()
                .filter(|action| robustly_safe(state, *action))
                .min_by(|a, b| worst_case_stage_cost(state, *a, target)
                    .total_cmp(&worst_case_stage_cost(state, *b, target)))
                .unwrap_or(0.0);
            (action, 0.0)
        }
        Policy::UnconstrainedInfoBonus => {
            let action = ACTIONS.iter().copied()
                .min_by(|a, b| {
                    let score_a = expected_stage_cost(state, *a, belief, target)
                        - INFO_WEIGHT * information_proxy(belief, *a);
                    let score_b = expected_stage_cost(state, *b, belief, target)
                        - INFO_WEIGHT * information_proxy(belief, *b);
                    score_a.total_cmp(&score_b)
                })
                .unwrap_or(0.0);
            (action, 0.0)
        }
        Policy::BudgetedDualControl => {
            let mut best_action = safe_ce_action(state, belief, target);
            let mut best_score = f64::INFINITY;
            let mut best_spend = 0.0;
            for action in ACTIONS {
                if !robustly_safe(state, action) { continue; }
                let task_cost = expected_stage_cost(state, action, belief, target);
                let degradation = (task_cost - ce_cost).max(0.0);
                if degradation > remaining_budget + 1e-12 { continue; }
                let score = task_cost - INFO_WEIGHT * information_proxy(belief, action);
                if score < best_score {
                    best_score = score;
                    best_action = action;
                    best_spend = degradation;
                }
            }
            (best_action, best_spend)
        }
        Policy::SeparatedSensorProbe => (safe_ce_action(state, belief, target), 0.0),
        Policy::OracleKnownGain => {
            ACTIONS.iter().copied()
                .filter(|action| (A * state + true_gain * *action).abs()
                    + PROCESS_NOISE_BOUND <= STATE_LIMIT)
                .min_by(|a, b| {
                    let cost_a = (A * state + true_gain * *a - target).powi(2)
                        + CONTROL_EFFORT * a * a;
                    let cost_b = (A * state + true_gain * *b - target).powi(2)
                        + CONTROL_EFFORT * b * b;
                    cost_a.total_cmp(&cost_b)
                })
                .map(|action| (action, 0.0))
                .unwrap_or((0.0, 0.0))
        }
    }
}

fn run_episode(scenario: Scenario, policy: Policy, tape: &EpisodeTape) -> Outcome {
    let mut state = 0.0;
    let mut belief = [0.20; 5];
    let initial_entropy = entropy(&belief);
    let mut total_cost = 0.0;
    let mut violation_steps = 0;
    let mut max_abs_state: f64 = 0.0;
    let mut budget_remaining = MISSION_DEGRADATION_BUDGET;
    let mut budget_spent = 0.0;
    let mut probe_count = 0;
    let mut probe_cost = 0.0;
    let mut action_effort = 0.0;

    for step in 0..HORIZON {
        let true_gain = scenario.gain_at(tape, step);
        let target = target_at(step);

        if policy == Policy::SeparatedSensorProbe && step % PROBE_PERIOD == 0 {
            let reading = true_gain + PROBE_NOISE_STD * tape.probe_noise[step];
            update_sensor_belief(&mut belief, reading);
            probe_count += 1;
            probe_cost += PROBE_COST;
            total_cost += PROBE_COST;
        }

        let (action, spend) = select_action(
            policy, state, &belief, target, true_gain, budget_remaining,
        );
        if policy == Policy::BudgetedDualControl {
            budget_remaining = (budget_remaining - spend).max(0.0);
            budget_spent += spend;
        }

        let process_noise = tape.process_noise[step];
        let next_state = A * state + true_gain * action + process_noise;
        let response_observation = true_gain * action
            + RESPONSE_NOISE_STD * tape.response_noise[step];

        if next_state.abs() > STATE_LIMIT { violation_steps += 1; }
        max_abs_state = max_abs_state.max(next_state.abs());
        action_effort += action * action;
        total_cost += (next_state - target).powi(2)
            + CONTROL_EFFORT * action * action;

        // The action is both the control intervention and the system-identification
        // stimulus. There is no independent/free observation in this update.
        update_transition_belief(&mut belief, action, response_observation);
        state = next_state;
    }

    Outcome {
        total_cost,
        violation_steps,
        max_abs_state,
        final_entropy: entropy(&belief),
        budget_spent,
        probe_count,
        probe_cost,
        action_effort,
    }
}

fn percentile(values: &mut [f64], fraction: f64) -> f64 {
    if values.is_empty() { return 0.0; }
    values.sort_by(|a, b| a.total_cmp(b));
    let index = ((values.len() as f64 * fraction).ceil() as usize)
        .saturating_sub(1).min(values.len() - 1);
    values[index]
}

fn summarize(
    scenario: Scenario,
    policy: Policy,
    outcomes: &[Outcome],
    baseline: &[Outcome],
) -> Summary {
    let n = outcomes.len() as f64;
    let mean_cost = outcomes.iter().map(|o| o.total_cost).sum::<f64>() / n;
    let baseline_mean = baseline.iter().map(|o| o.total_cost).sum::<f64>() / n;
    let paired_deltas = outcomes.iter().zip(baseline.iter())
        .map(|(candidate, reference)| candidate.total_cost - reference.total_cost)
        .collect::<Vec<_>>();
    let delta_mean = paired_deltas.iter().sum::<f64>() / n;
    let variance = paired_deltas.iter().map(|d| (d - delta_mean).powi(2)).sum::<f64>()
        / (n - 1.0).max(1.0);
    let standard_error = (variance / n).sqrt();
    let mut costs = outcomes.iter().map(|o| o.total_cost).collect::<Vec<_>>();
    let p95_cost = percentile(&mut costs.clone(), 0.95);
    costs.sort_by(|a, b| b.total_cmp(a));
    let tail_count = ((costs.len() as f64 * 0.05).ceil() as usize).max(1);
    let cvar95_cost = costs.iter().take(tail_count).sum::<f64>() / tail_count as f64;
    let violation_episodes = outcomes.iter().filter(|o| o.violation_steps > 0).count();
    let violation_steps = outcomes.iter().map(|o| o.violation_steps).sum::<usize>();
    let mean_information_gain = outcomes.iter()
        .map(|o| initial_entropy() - o.final_entropy).sum::<f64>() / n;

    Summary {
        scenario: scenario.name(),
        policy: policy.name(),
        mean_cost,
        mean_cost_delta_vs_shielded_ce: mean_cost - baseline_mean,
        ci95_low: delta_mean - 1.96 * standard_error,
        ci95_high: delta_mean + 1.96 * standard_error,
        p95_cost,
        cvar95_cost,
        violation_episode_rate: violation_episodes as f64 / n,
        violation_step_rate: violation_steps as f64 / (outcomes.len() * HORIZON) as f64,
        mean_max_abs_state: outcomes.iter().map(|o| o.max_abs_state).sum::<f64>() / n,
        mean_information_gain,
        mean_budget_spent: outcomes.iter().map(|o| o.budget_spent).sum::<f64>() / n,
        max_budget_spent: outcomes.iter().map(|o| o.budget_spent).fold(0.0, f64::max),
        mean_probe_count: outcomes.iter().map(|o| o.probe_count as f64).sum::<f64>() / n,
        max_probe_count: outcomes.iter().map(|o| o.probe_count).max().unwrap_or(0),
        mean_probe_cost: outcomes.iter().map(|o| o.probe_cost).sum::<f64>() / n,
        mean_action_effort: outcomes.iter().map(|o| o.action_effort).sum::<f64>() / n,
    }
}

fn initial_entropy() -> f64 {
    let uniform = [0.20; 5];
    entropy(&uniform)
}

fn make_outcomes(scenario: Scenario, policy: Policy) -> Vec<Outcome> {
    (0..EPISODES).map(|episode| {
        let tape = EpisodeTape::new(scenario, episode);
        run_episode(scenario, policy, &tape)
    }).collect()
}

fn render_number(value: f64) -> String {
    if value.is_finite() { format!("{value:.8}") } else { "null".to_string() }
}

fn render_report(rows: &[Summary]) -> String {
    let scenarios = Scenario::all().iter().map(|s| format!("\"{}\"", s.name()))
        .collect::<Vec<_>>().join(",");
    let policies = Policy::all().iter().map(|p| format!("\"{}\"", p.name()))
        .collect::<Vec<_>>().join(",");
    let mut out = String::new();
    writeln!(&mut out, "{{").unwrap();
    writeln!(&mut out, "  \"schema_version\": \"{SCHEMA}\",").unwrap();
    writeln!(&mut out, "  \"episode_count\": {EPISODES},").unwrap();
    writeln!(&mut out, "  \"horizon\": {HORIZON},").unwrap();
    writeln!(&mut out, "  \"target_switch_step\": {TARGET_SWITCH},").unwrap();
    writeln!(&mut out, "  \"parameter_shift_step\": {PARAMETER_SHIFT},").unwrap();
    writeln!(&mut out, "  \"response_noise_std\": {},", render_number(RESPONSE_NOISE_STD)).unwrap();
    writeln!(&mut out, "  \"state_limit\": {},", render_number(STATE_LIMIT)).unwrap();
    writeln!(&mut out, "  \"mission_degradation_budget\": {},", render_number(MISSION_DEGRADATION_BUDGET)).unwrap();
    writeln!(&mut out, "  \"info_weight\": {},", render_number(INFO_WEIGHT)).unwrap();
    writeln!(&mut out, "  \"probe_period\": {PROBE_PERIOD},").unwrap();
    writeln!(&mut out, "  \"probe_cost\": {},", render_number(PROBE_COST)).unwrap();
    writeln!(&mut out, "  \"scenarios\": [{scenarios}],").unwrap();
    writeln!(&mut out, "  \"policies\": [{policies}],").unwrap();
    writeln!(&mut out, "  \"summary\": [").unwrap();

    for (index, row) in rows.iter().enumerate() {
        let comma = if index + 1 == rows.len() { "" } else { "," };
        writeln!(&mut out,
            concat!(
                "    {{\"scenario\":\"{}\",\"policy\":\"{}\",",
                "\"mean_cost\":{},\"mean_cost_delta_vs_shielded_ce\":{},",
                "\"paired_ci95_low\":{},\"paired_ci95_high\":{},",
                "\"p95_cost\":{},\"cvar95_cost\":{},",
                "\"violation_episode_rate\":{},\"violation_step_rate\":{},",
                "\"mean_max_abs_state\":{},\"mean_information_gain\":{},",
                "\"mean_budget_spent\":{},\"max_budget_spent\":{},",
                "\"mean_probe_count\":{},\"max_probe_count\":{},",
                "\"mean_probe_cost\":{},\"mean_action_effort\":{}}}{}"
            ),
            row.scenario, row.policy,
            render_number(row.mean_cost), render_number(row.mean_cost_delta_vs_shielded_ce),
            render_number(row.ci95_low), render_number(row.ci95_high),
            render_number(row.p95_cost), render_number(row.cvar95_cost),
            render_number(row.violation_episode_rate), render_number(row.violation_step_rate),
            render_number(row.mean_max_abs_state), render_number(row.mean_information_gain),
            render_number(row.mean_budget_spent), render_number(row.max_budget_spent),
            render_number(row.mean_probe_count), row.max_probe_count,
            render_number(row.mean_probe_cost), render_number(row.mean_action_effort), comma
        ).unwrap();
    }
    writeln!(&mut out, "  ]").unwrap();
    writeln!(&mut out, "}}").unwrap();
    out
}

fn run_benchmark() -> Vec<Summary> {
    let mut rows = Vec::new();
    for scenario in Scenario::all() {
        let baseline = (0..EPISODES).map(|episode| {
            let tape = EpisodeTape::new(scenario, episode);
            run_episode(scenario, Policy::ShieldedCertaintyEquivalent, &tape)
        }).collect::<Vec<_>>();
        for policy in Policy::all() {
            let outcomes = (0..EPISODES).map(|episode| {
                let tape = EpisodeTape::new(scenario, episode);
                run_episode(scenario, policy, &tape)
            }).collect::<Vec<_>>();
            rows.push(summarize(scenario, policy, &outcomes, &baseline));
        }
    }
    rows
}

fn main() {
    println!("{}", render_report(&run_benchmark()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posterior_remains_normalized_and_finite() {
        let mut belief = [0.20; 5];
        update_transition_belief(&mut belief, 0.3, 0.28);
        let sum = belief.iter().sum::<f64>();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(belief.iter().all(|p| p.is_finite() && *p >= 0.0));
    }

    #[test]
    fn zero_control_produces_no_transition_information() {
        let mut belief = [0.20; 5];
        update_transition_belief(&mut belief, 0.0, 0.4);
        assert!(belief.iter().all(|p| (*p - 0.20).abs() < 1e-10));
    }

    #[test]
    fn budgeted_dual_controller_respects_robust_state_envelope() {
        for state in [-0.55, -0.2, 0.0, 0.2, 0.55] {
            let belief = [0.20; 5];
            let (action, _) = select_action(
                Policy::BudgetedDualControl, state, &belief, 0.4, 1.0,
                MISSION_DEGRADATION_BUDGET,
            );
            assert!(robustly_safe(state, action));
        }
    }

    #[test]
    fn zero_mission_budget_reverts_to_safe_certainty_equivalent() {
        let belief = [0.20; 5];
        for state in [-0.4, -0.1, 0.0, 0.1, 0.4] {
            let expected = safe_ce_action(state, &belief, 0.4);
            let (actual, spend) = select_action(
                Policy::BudgetedDualControl, state, &belief, 0.4, 1.0, 0.0,
            );
            assert!((actual - expected).abs() < 1e-12);
            assert_eq!(spend, 0.0);
        }
    }

    #[test]
    fn deterministic_tape_is_paired_across_policies() {
        let tape_a = EpisodeTape::new(Scenario::AbruptGainShift, 17);
        let tape_b = EpisodeTape::new(Scenario::AbruptGainShift, 17);
        assert_eq!(tape_a.initial_gain, tape_b.initial_gain);
        assert_eq!(tape_a.shifted_gain, tape_b.shifted_gain);
        assert_eq!(tape_a.process_noise, tape_b.process_noise);
        assert_eq!(tape_a.response_noise, tape_b.response_noise);
        assert_eq!(tape_a.probe_noise, tape_b.probe_noise);
    }

    #[test]
    fn dual_control_has_a_hard_episode_budget() {
        for scenario in Scenario::all() {
            for episode in 0..32 {
                let tape = EpisodeTape::new(scenario, episode);
                let outcome = run_episode(scenario, Policy::BudgetedDualControl, &tape);
                assert!(outcome.budget_spent <= MISSION_DEGRADATION_BUDGET + 1e-9);
                assert_eq!(outcome.probe_count, 0);
            }
        }
    }

    #[test]
    fn separated_probe_is_capped_and_charged() {
        let tape = EpisodeTape::new(Scenario::StationaryUnknownGain, 5);
        let outcome = run_episode(Scenario::StationaryUnknownGain, Policy::SeparatedSensorProbe, &tape);
        assert_eq!(outcome.probe_count, HORIZON.div_ceil(PROBE_PERIOD));
        assert!((outcome.probe_cost - outcome.probe_count as f64 * PROBE_COST).abs() < 1e-10);
    }

    #[test]
    fn report_has_one_row_per_scenario_policy_pair() {
        let rows = run_benchmark();
        assert_eq!(rows.len(), Scenario::all().len() * Policy::all().len());
        let report = render_report(&rows);
        assert!(report.contains("\"schema_version\": \"cmaq-012-safe-dual-control-v0.1\""));
        assert!(report.contains("\"episode_count\": 256"));
        assert!(report.contains("\"horizon\": 64"));
    }
}
