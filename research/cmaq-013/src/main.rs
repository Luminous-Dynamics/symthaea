use std::fmt::Write as _;

const SCHEMA: &str = "cmaq-013-advantage-of-prediction-v0.1";
const EPISODES: usize = 256;
const STATES: usize = 5;
const ACTIONS: usize = 3;
const HORIZON: usize = 3;
const DISCOUNT: f64 = 0.95;
const ERROR_LEVELS: [f64; 4] = [0.0, 0.25, 0.50, 0.75];
const BASE_SEED: u64 = 0xC0A0_0013_2026_1030;

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

    fn index(&mut self, upper: usize) -> usize {
        ((self.next_u64() as u128 * upper as u128) >> 64) as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario { DelayedReward, TrapAvoidance, ActionCoupling }

impl Scenario {
    fn all() -> [Self; 3] { [Self::DelayedReward, Self::TrapAvoidance, Self::ActionCoupling] }

    fn name(self) -> &'static str {
        match self {
            Self::DelayedReward => "delayed_reward",
            Self::TrapAvoidance => "trap_avoidance",
            Self::ActionCoupling => "action_coupling",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy { DirectGreedy, OneStepModel, LearnedModelPlanner, ShamModelPlanner, OracleModelPlanner }

impl Policy {
    fn all() -> [Self; 5] {
        [Self::DirectGreedy, Self::OneStepModel, Self::LearnedModelPlanner,
         Self::ShamModelPlanner, Self::OracleModelPlanner]
    }

    fn name(self) -> &'static str {
        match self {
            Self::DirectGreedy => "direct_greedy_no_model",
            Self::OneStepModel => "one_step_model_negative_control",
            Self::LearnedModelPlanner => "learned_model_planner",
            Self::ShamModelPlanner => "sham_model_planner",
            Self::OracleModelPlanner => "oracle_model_planner",
        }
    }
}

#[derive(Clone, Debug)]
struct TransitionModel {
    next: [[usize; ACTIONS]; STATES],
}

#[derive(Clone, Debug)]
struct ModelPack {
    truth: TransitionModel,
    learned: TransitionModel,
    sham: TransitionModel,
    transition_error_rate: f64,
}

#[derive(Clone, Debug)]
struct Outcome {
    total_return: f64,
    root_action_regret: f64,
    oracle_return: f64,
    optimism_gap: f64,
    action_change_rate: f64,
    model_queries: usize,
    decision_count: usize,
}

#[derive(Clone, Debug)]
struct Summary {
    scenario: &'static str,
    policy: &'static str,
    model_error_rate: f64,
    mean_return: f64,
    return_delta_vs_direct: f64,
    paired_ci95_low: f64,
    paired_ci95_high: f64,
    mean_regret_vs_oracle: f64,
    cvar95_regret_vs_oracle: f64,
    max_regret_vs_oracle: f64,
    mean_root_action_regret: f64,
    mean_prediction_optimism_gap: f64,
    action_change_rate_vs_direct: f64,
    transition_queries_per_decision: f64,
    max_transition_queries_per_episode: usize,
}

fn transition_reward(scenario: Scenario, state: usize, action: usize) -> (usize, f64) {
    if state == 4 { return (4, 0.0); }
    match scenario {
        Scenario::DelayedReward => match (state, action) {
            (0, 0) => (4, 0.80),
            (0, 1) => (1, 0.10),
            (0, 2) => (2, 0.00),
            (1, 0) => (3, 1.25),
            (1, 1) => (4, 0.20),
            (1, 2) => (4, 0.00),
            (2, 0) => (4, -0.30),
            (2, 1) => (4, 0.30),
            (2, 2) => (4, 0.00),
            (3, 0) => (4, 0.40),
            (3, 1) => (4, 0.60),
            (3, 2) => (4, 0.10),
            _ => (4, 0.0),
        },
        Scenario::TrapAvoidance => match (state, action) {
            (0, 0) => (2, 1.00),
            (0, 1) => (1, 0.20),
            (0, 2) => (3, 0.10),
            (1, 0) => (3, 1.00),
            (1, 1) => (4, 0.10),
            (1, 2) => (3, 0.50),
            (2, 0) => (4, -0.80),
            (2, 1) => (4, -0.50),
            (2, 2) => (4, -0.30),
            (3, 0) => (4, 0.50),
            (3, 1) => (4, 0.20),
            (3, 2) => (4, 0.10),
            _ => (4, 0.0),
        },
        Scenario::ActionCoupling => match (state, action) {
            (0, 0) => (1, 0.40),
            (0, 1) => (2, 0.25),
            (0, 2) => (3, 0.10),
            (1, 0) => (4, 0.20),
            (1, 1) => (2, 1.00),
            (1, 2) => (3, 0.40),
            (2, 0) => (4, 0.10),
            (2, 1) => (4, 1.25),
            (2, 2) => (1, 0.00),
            (3, 0) => (4, 0.80),
            (3, 1) => (4, 0.50),
            (3, 2) => (4, 0.20),
            _ => (4, 0.0),
        },
    }
}

fn true_model(scenario: Scenario) -> TransitionModel {
    let mut next = [[4usize; ACTIONS]; STATES];
    for state in 0..STATES {
        for action in 0..ACTIONS {
            next[state][action] = transition_reward(scenario, state, action).0;
        }
    }
    TransitionModel { next }
}

fn model_pack(scenario: Scenario, error_rate: f64, episode: usize) -> ModelPack {
    let truth = true_model(scenario);
    let mut learned = truth.clone();
    let mut rng = SplitMix64::new(
        BASE_SEED
            ^ ((scenario as u64) << 48)
            ^ (episode as u64).wrapping_mul(0xD134_2543_DE82_EF95)
            ^ ((error_rate.to_bits()).rotate_left(17)),
    );

    // The twelve nonterminal state/action transitions are the model's support.
    // Corrupt exactly round(error_rate * 12) unique cells for a controlled
    // and comparable transition-prediction error rate.
    let mut cells = (0..(STATES - 1) * ACTIONS).collect::<Vec<_>>();
    for i in (1..cells.len()).rev() {
        let j = rng.index(i + 1);
        cells.swap(i, j);
    }
    let corrupt_count = (error_rate * cells.len() as f64).round() as usize;
    for cell in cells.into_iter().take(corrupt_count) {
        let state = cell / ACTIONS;
        let action = cell % ACTIONS;
        let correct = truth.next[state][action];
        let mut wrong = rng.index(STATES - 1);
        if wrong >= correct { wrong += 1; }
        learned.next[state][action] = wrong;
    }

    // Sham model: same state/action/reward interface, but action labels are
    // cyclically permuted in the transition table. It is intentionally not
    // treated as a valid predictor.
    let mut sham = truth.clone();
    for state in 0..STATES - 1 {
        for action in 0..ACTIONS {
            sham.next[state][action] = truth.next[state][(action + 1) % ACTIONS];
        }
    }
    let changed = (0..STATES - 1).flat_map(|s| (0..ACTIONS).map(move |a| (s, a)))
        .filter(|(s, a)| learned.next[*s][*a] != truth.next[*s][*a]).count();

    ModelPack {
        truth,
        learned,
        sham,
        transition_error_rate: changed as f64 / ((STATES - 1) * ACTIONS) as f64,
    }
}

fn value(
    scenario: Scenario,
    model: &TransitionModel,
    state: usize,
    action: usize,
    depth: usize,
    queries: &mut usize,
) -> f64 {
    let (true_next, reward) = transition_reward(scenario, state, action);
    if depth <= 1 || state == 4 { return reward; }
    *queries += 1;
    let predicted_next = model.next[state][action];
    let future = (0..ACTIONS)
        .map(|next_action| value(scenario, model, predicted_next, next_action, depth - 1, queries))
        .fold(f64::NEG_INFINITY, f64::max);
    let _ = true_next;
    reward + DISCOUNT * future
}

fn best_action(
    scenario: Scenario,
    model: &TransitionModel,
    state: usize,
    depth: usize,
    queries: &mut usize,
) -> (usize, f64) {
    let mut best = 0;
    let mut best_value = f64::NEG_INFINITY;
    for action in 0..ACTIONS {
        let q = value(scenario, model, state, action, depth, queries);
        if q > best_value {
            best = action;
            best_value = q;
        }
    }
    (best, best_value)
}

fn immediate_action(scenario: Scenario, state: usize) -> usize {
    (0..ACTIONS).max_by(|a, b| {
        transition_reward(scenario, state, *a).1.total_cmp(&transition_reward(scenario, state, *b).1)
    }).unwrap_or(0)
}

fn choose_action(
    scenario: Scenario,
    policy: Policy,
    pack: &ModelPack,
    state: usize,
    remaining: usize,
    query_counter: &mut usize,
) -> (usize, f64, f64) {
    let direct = immediate_action(scenario, state);
    let true_depth = remaining.min(HORIZON);
    let (_, oracle_best_value) = best_action(
        scenario, &pack.truth, state, true_depth, &mut 0usize,
    );

    let (action, predicted_value) = match policy {
        Policy::DirectGreedy => (direct, transition_reward(scenario, state, direct).1),
        Policy::OneStepModel => {
            best_action(scenario, &pack.learned, state, 1, query_counter)
        }
        Policy::LearnedModelPlanner => {
            best_action(scenario, &pack.learned, state, true_depth, query_counter)
        }
        Policy::ShamModelPlanner => {
            best_action(scenario, &pack.sham, state, true_depth, query_counter)
        }
        Policy::OracleModelPlanner => {
            best_action(scenario, &pack.truth, state, true_depth, query_counter)
        }
    };
    let _ = oracle_best_value;
    (action, predicted_value, oracle_best_value)
}

fn run_episode(scenario: Scenario, policy: Policy, pack: &ModelPack) -> Outcome {
    let mut state = 0usize;
    let mut total_return = 0.0;
    let mut discount = 1.0;
    let mut root_action_regret = 0.0;
    let mut optimism_gap = 0.0;
    let mut action_changes = 0usize;
    let mut decisions = 0usize;
    let mut queries = 0usize;
    let mut oracle_return = 0.0;
    let mut oracle_state = 0usize;
    let mut oracle_discount = 1.0;
    let mut root_done = false;

    for step in 0..HORIZON {
        let remaining = HORIZON - step;

        // Candidate policy and oracle advance on separate, paired trajectories.
        // Do not break when the candidate terminates: the oracle may continue.
        if state != 4 {
            let (action, predicted_value, oracle_best_value) =
                choose_action(scenario, policy, pack, state, remaining, &mut queries);
            let (true_next, reward) = transition_reward(scenario, state, action);
            let mut true_action_queries = 0;
            let true_selected_value = value(
                scenario, &pack.truth, state, action, remaining, &mut true_action_queries,
            );
            if !root_done {
                root_action_regret = (oracle_best_value - true_selected_value).max(0.0);
                if matches!(policy, Policy::LearnedModelPlanner | Policy::ShamModelPlanner) {
                    optimism_gap = predicted_value - true_selected_value;
                }
                root_done = true;
            }
            let direct = immediate_action(scenario, state);
            if action != direct { action_changes += 1; }
            decisions += 1;
            total_return += discount * reward;
            discount *= DISCOUNT;
            state = true_next;
        }

        // Oracle retains its own state and receives the full remaining horizon.
        if oracle_state != 4 {
            let mut oracle_queries = 0;
            let (oracle_action, _, _) = best_action(
                scenario, &pack.truth, oracle_state, remaining, &mut oracle_queries,
            );
            let (next, oracle_reward) = transition_reward(scenario, oracle_state, oracle_action);
            oracle_return += oracle_discount * oracle_reward;
            oracle_discount *= DISCOUNT;
            oracle_state = next;
        }
    }

    Outcome {
        total_return,
        root_action_regret,
        oracle_return,
        optimism_gap,
        action_change_rate: if decisions == 0 { 0.0 } else { action_changes as f64 / decisions as f64 },
        model_queries: queries,
        decision_count: decisions,
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
    direct: &[Outcome],
    oracle: &[Outcome],
    measured_model_error: f64,
) -> Summary {
    let n = outcomes.len() as f64;
    let mean_return = outcomes.iter().map(|o| o.total_return).sum::<f64>() / n;
    let direct_mean = direct.iter().map(|o| o.total_return).sum::<f64>() / n;
    let paired = outcomes.iter().zip(direct.iter())
        .map(|(candidate, baseline)| candidate.total_return - baseline.total_return)
        .collect::<Vec<_>>();
    let delta = paired.iter().sum::<f64>() / n;
    let variance = paired.iter().map(|d| (d - delta).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let standard_error = (variance / n).sqrt();
    let oracle_regrets = outcomes.iter().zip(oracle.iter())
        .map(|(candidate, ideal)| (ideal.total_return - candidate.total_return).max(0.0))
        .collect::<Vec<_>>();
    let mean_regret = oracle_regrets.iter().sum::<f64>() / n;
    let mut tail = oracle_regrets.clone();
    tail.sort_by(|a, b| b.total_cmp(a));
    let tail_count = ((tail.len() as f64 * 0.05).ceil() as usize).max(1);
    let cvar95 = tail.iter().take(tail_count).sum::<f64>() / tail_count as f64;
    let maximum = oracle_regrets.iter().copied().fold(0.0, f64::max);

    Summary {
        scenario: scenario.name(),
        policy: policy.name(),
        model_error_rate: measured_model_error,
        mean_return,
        return_delta_vs_direct: mean_return - direct_mean,
        paired_ci95_low: delta - 1.96 * standard_error,
        paired_ci95_high: delta + 1.96 * standard_error,
        mean_regret_vs_oracle: mean_regret,
        cvar95_regret_vs_oracle: cvar95,
        max_regret_vs_oracle: maximum,
        mean_root_action_regret: outcomes.iter().map(|o| o.root_action_regret).sum::<f64>() / n,
        mean_prediction_optimism_gap: outcomes.iter().map(|o| o.optimism_gap).sum::<f64>() / n,
        action_change_rate_vs_direct: outcomes.iter().map(|o| o.action_change_rate).sum::<f64>() / n,
        transition_queries_per_decision: outcomes.iter().map(|o| {
            if o.decision_count == 0 { 0.0 } else { o.model_queries as f64 / o.decision_count as f64 }
        }).sum::<f64>() / n,
        max_transition_queries_per_episode: outcomes.iter().map(|o| o.model_queries).max().unwrap_or(0),
    }
}

fn run_benchmark() -> Vec<Summary> {
    let mut rows = Vec::new();
    for scenario in Scenario::all() {
        for error_rate in ERROR_LEVELS {
            let packs = (0..EPISODES).map(|episode| model_pack(scenario, error_rate, episode))
                .collect::<Vec<_>>();
            let direct = packs.iter().map(|pack| run_episode(scenario, Policy::DirectGreedy, pack))
                .collect::<Vec<_>>();
            let oracle = packs.iter().map(|pack| run_episode(scenario, Policy::OracleModelPlanner, pack))
                .collect::<Vec<_>>();
            let measured_error = packs.iter().map(|p| p.transition_error_rate).sum::<f64>() / EPISODES as f64;

            for policy in Policy::all() {
                let outcomes = packs.iter().map(|pack| run_episode(scenario, policy, pack))
                    .collect::<Vec<_>>();
                rows.push(summarize(
                    scenario, policy, &outcomes, &direct, &oracle, measured_error,
                ));
            }
        }
    }
    rows
}

fn num(x: f64) -> String {
    if x.is_finite() { format!("{x:.8}") } else { "null".to_string() }
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
    writeln!(&mut out, "  \"discount\": {},", num(DISCOUNT)).unwrap();
    writeln!(&mut out, "  \"error_levels\": [{}],",
        ERROR_LEVELS.iter().map(|x| num(*x)).collect::<Vec<_>>().join(",")).unwrap();
    writeln!(&mut out, "  \"scenarios\": [{scenarios}],").unwrap();
    writeln!(&mut out, "  \"policies\": [{policies}],").unwrap();
    writeln!(&mut out, "  \"summary\": [").unwrap();
    for (i, row) in rows.iter().enumerate() {
        let comma = if i + 1 == rows.len() { "" } else { "," };
        writeln!(&mut out,
            concat!(
                "    {{\"scenario\":\"{}\",\"policy\":\"{}\",\"model_error_rate\":{},",
                "\"mean_return\":{},\"return_delta_vs_direct\":{},",
                "\"paired_ci95_low\":{},\"paired_ci95_high\":{},",
                "\"mean_regret_vs_oracle\":{},\"cvar95_regret_vs_oracle\":{},",
                "\"max_regret_vs_oracle\":{},\"mean_root_action_regret\":{},",
                "\"mean_prediction_optimism_gap\":{},\"action_change_rate_vs_direct\":{},",
                "\"transition_queries_per_decision\":{},\"max_transition_queries_per_episode\":{}}}{}"
            ),
            row.scenario, row.policy, num(row.model_error_rate),
            num(row.mean_return), num(row.return_delta_vs_direct),
            num(row.paired_ci95_low), num(row.paired_ci95_high),
            num(row.mean_regret_vs_oracle), num(row.cvar95_regret_vs_oracle),
            num(row.max_regret_vs_oracle), num(row.mean_root_action_regret),
            num(row.mean_prediction_optimism_gap), num(row.action_change_rate_vs_direct),
            num(row.transition_queries_per_decision), row.max_transition_queries_per_episode, comma
        ).unwrap();
    }
    writeln!(&mut out, "  ]").unwrap();
    writeln!(&mut out, "}}").unwrap();
    out
}

fn main() {
    println!("{}", render_report(&run_benchmark()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_step_model_is_the_same_as_direct_immediate_reward_policy() {
        for scenario in Scenario::all() {
            for episode in 0..32 {
                let pack = model_pack(scenario, 0.75, episode);
                for state in 0..4 {
                    let mut queries = 0;
                    let direct = immediate_action(scenario, state);
                    let (one_step, _, _) = choose_action(
                        scenario, Policy::OneStepModel, &pack, state, 1, &mut queries,
                    );
                    assert_eq!(one_step, direct);
                }
            }
        }
    }

    #[test]
    fn perfect_model_planner_matches_oracle_planner() {
        for scenario in Scenario::all() {
            for episode in 0..32 {
                let pack = model_pack(scenario, 0.0, episode);
                for state in 0..4 {
                    let mut q1 = 0;
                    let mut q2 = 0;
                    let learned = choose_action(
                        scenario, Policy::LearnedModelPlanner, &pack, state, HORIZON, &mut q1,
                    );
                    let oracle = choose_action(
                        scenario, Policy::OracleModelPlanner, &pack, state, HORIZON, &mut q2,
                    );
                    assert_eq!(learned.0, oracle.0);
                    assert!((learned.1 - oracle.1).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn model_error_fraction_is_exact_on_twelve_transition_cells() {
        for error in ERROR_LEVELS {
            let pack = model_pack(Scenario::DelayedReward, error, 7);
            assert!((pack.transition_error_rate - error).abs() < 1e-12);
        }
    }

    #[test]
    fn deterministic_paired_models_repeat_for_same_seed() {
        let first = model_pack(Scenario::ActionCoupling, 0.5, 19);
        let second = model_pack(Scenario::ActionCoupling, 0.5, 19);
        assert_eq!(first.learned.next, second.learned.next);
        assert_eq!(first.sham.next, second.sham.next);
    }

    #[test]
    fn direct_and_one_step_negative_control_match_episode_return() {
        for scenario in Scenario::all() {
            let pack = model_pack(scenario, 0.50, 3);
            let direct = run_episode(scenario, Policy::DirectGreedy, &pack);
            let one_step = run_episode(scenario, Policy::OneStepModel, &pack);
            assert!((direct.total_return - one_step.total_return).abs() < 1e-12);
        }
    }

    #[test]
    fn delayed_reward_perfect_model_improves_over_direct_greedy() {
        let pack = model_pack(Scenario::DelayedReward, 0.0, 0);
        let direct = run_episode(Scenario::DelayedReward, Policy::DirectGreedy, &pack);
        let oracle = run_episode(Scenario::DelayedReward, Policy::OracleModelPlanner, &pack);
        assert!(oracle.total_return > direct.total_return);
    }

    #[test]
    fn report_has_one_row_per_scenario_error_policy() {
        let rows = run_benchmark();
        assert_eq!(rows.len(), Scenario::all().len() * ERROR_LEVELS.len() * Policy::all().len());
        let report = render_report(&rows);
        assert!(report.contains("\"schema_version\": \"cmaq-013-advantage-of-prediction-v0.1\""));
        assert!(report.contains("\"episode_count\": 256"));
    }
}
