use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::process;

const PPM: f64 = 1_000_000.0;
const MAX_HORIZON: usize = 1_000;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Key {
    task: String,
    state: String,
    action: u32,
}

#[derive(Clone, Debug)]
struct TruthRow {
    next_state: String,
    reward_milli: i64,
    safe: bool,
    done: bool,
}

#[derive(Clone, Debug)]
struct PredictionRow {
    next_state: String,
    reward_milli: i64,
    reward_radius_milli: u64,
    safe_ppm: u32,
    safe_radius_ppm: u32,
    done: bool,
}

#[derive(Clone, Debug)]
struct StartConfig {
    state: String,
    horizon: usize,
    gamma_ppm: u32,
    safety_threshold_ppm: u32,
    min_support_count: u64,
    reward_risk_beta_ppm: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Policy {
    Point,
    TrustRegion,
    Oracle,
}

impl Policy {
    fn name(self) -> &'static str {
        match self {
            Self::Point => "point",
            Self::TrustRegion => "trust_region",
            Self::Oracle => "oracle",
        }
    }
}

struct Evaluator {
    truth: HashMap<Key, TruthRow>,
    predictions: HashMap<Key, PredictionRow>,
    support: HashMap<Key, u64>,
    configs: HashMap<String, StartConfig>,
    actions: HashMap<(String, String), Vec<u32>>,
    states: HashMap<String, HashSet<String>>,
}

#[derive(Clone, Debug)]
struct ScoreRow {
    task: String,
    policy: String,
    actual_return_milli: f64,
    constrained_return_milli: f64,
    oracle_safe_return_milli: f64,
    regret_milli: f64,
    risk_adjusted_plan_value_milli: f64,
    model_to_rollout_gap_milli: f64,
    executed_actions: usize,
    unsafe_actions: usize,
    unsupported_actions: usize,
    unsupported_rate_ppm: u64,
    executed_reward_mae_milli: f64,
    executed_safety_brier_micro: f64,
    transition_mismatches: usize,
    reward_interval_coverage_ppm: u64,
    mean_interval_width_milli: f64,
    false_safe_predictions: usize,
    abstained: bool,
    hard_gate_pass: bool,
}

fn valid_id(value: &str, field: &str) -> Result<String, String> {
    if value.is_empty() || value.chars().any(|c| c.is_control() || c.is_whitespace()) {
        return Err(format!("{field} must be a non-empty token without whitespace"));
    }
    Ok(value.to_owned())
}

fn parse_u32(value: &str, field: &str) -> Result<u32, String> {
    value.parse::<u32>().map_err(|_| format!("invalid {field}: {value}"))
}

fn parse_u64(value: &str, field: &str) -> Result<u64, String> {
    value.parse::<u64>().map_err(|_| format!("invalid {field}: {value}"))
}

fn parse_i64(value: &str, field: &str) -> Result<i64, String> {
    value.parse::<i64>().map_err(|_| format!("invalid {field}: {value}"))
}

fn parse_bool(value: &str, field: &str) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("{field} must be exactly true or false, got {value}")),
    }
}

fn read_tsv(path: &str, expected_header: &[&str]) -> Result<Vec<Vec<String>>, String> {
    let contents = fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut lines = contents.lines();
    let header = lines.next().ok_or_else(|| format!("{path} is empty"))?;
    let actual_header: Vec<&str> = header.split('\t').collect();
    if actual_header.as_slice() != expected_header {
        return Err(format!(
            "{path} header mismatch: expected {}, got {}",
            expected_header.join("\t"),
            actual_header.join("\t")
        ));
    }
    let mut rows = Vec::new();
    for (zero_based, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<String> = line.split('\t').map(str::to_owned).collect();
        if fields.len() != expected_header.len() {
            return Err(format!(
                "{path}:{} has {} fields; expected {}",
                zero_based + 2,
                fields.len(),
                expected_header.len()
            ));
        }
        rows.push(fields);
    }
    Ok(rows)
}

impl Evaluator {
    fn load(truth_path: &str, prediction_path: &str, support_path: &str, starts_path: &str) -> Result<Self, String> {
        let truth_rows = read_tsv(
            truth_path,
            &["task_id", "state_id", "action_id", "next_state_id", "reward_milli", "safe", "done"],
        )?;
        let prediction_rows = read_tsv(
            prediction_path,
            &["task_id", "state_id", "action_id", "predicted_next_state_id", "predicted_reward_milli", "reward_radius_milli", "predicted_safe_ppm", "safe_radius_ppm", "predicted_done"],
        )?;
        let support_rows = read_tsv(support_path, &["task_id", "state_id", "action_id", "support_count"])?;
        let start_rows = read_tsv(
            starts_path,
            &["task_id", "start_state_id", "horizon", "gamma_ppm", "safety_threshold_ppm", "min_support_count", "reward_risk_beta_ppm"],
        )?;

        let mut truth = HashMap::new();
        let mut states: HashMap<String, HashSet<String>> = HashMap::new();
        let mut actions: HashMap<(String, String), Vec<u32>> = HashMap::new();

        for row in truth_rows {
            let task = valid_id(&row[0], "task_id")?;
            let state = valid_id(&row[1], "state_id")?;
            let action = parse_u32(&row[2], "action_id")?;
            let next_state = valid_id(&row[3], "next_state_id")?;
            let reward_milli = parse_i64(&row[4], "reward_milli")?;
            let safe = parse_bool(&row[5], "safe")?;
            let done = parse_bool(&row[6], "done")?;
            states.entry(task.clone()).or_default().insert(state.clone());
            states.entry(task.clone()).or_default().insert(next_state.clone());
            let key = Key { task: task.clone(), state: state.clone(), action };
            if truth.insert(key.clone(), TruthRow { next_state, reward_milli, safe, done }).is_some() {
                return Err(format!("duplicate truth key: {task}/{state}/{action}"));
            }
            actions.entry((task, state)).or_default().push(action);
        }
        if truth.is_empty() {
            return Err("truth file has no transitions".to_owned());
        }
        for action_list in actions.values_mut() {
            action_list.sort_unstable();
            action_list.dedup();
        }

        let mut predictions = HashMap::new();
        for row in prediction_rows {
            let task = valid_id(&row[0], "task_id")?;
            let state = valid_id(&row[1], "state_id")?;
            let action = parse_u32(&row[2], "action_id")?;
            let next_state = valid_id(&row[3], "predicted_next_state_id")?;
            let reward_milli = parse_i64(&row[4], "predicted_reward_milli")?;
            let reward_radius_milli = parse_u64(&row[5], "reward_radius_milli")?;
            let safe_ppm = parse_u32(&row[6], "predicted_safe_ppm")?;
            let safe_radius_ppm = parse_u32(&row[7], "safe_radius_ppm")?;
            let done = parse_bool(&row[8], "predicted_done")?;
            if safe_ppm > 1_000_000 || safe_radius_ppm > 1_000_000 {
                return Err(format!("safety prediction/radius exceeds 1000000 for {task}/{state}/{action}"));
            }
            let key = Key { task: task.clone(), state: state.clone(), action };
            if predictions.insert(key, PredictionRow {
                next_state, reward_milli, reward_radius_milli, safe_ppm, safe_radius_ppm, done,
            }).is_some() {
                return Err(format!("duplicate prediction key: {task}/{state}/{action}"));
            }
        }

        let mut support = HashMap::new();
        for row in support_rows {
            let task = valid_id(&row[0], "task_id")?;
            let state = valid_id(&row[1], "state_id")?;
            let action = parse_u32(&row[2], "action_id")?;
            let count = parse_u64(&row[3], "support_count")?;
            let key = Key { task: task.clone(), state: state.clone(), action };
            if support.insert(key, count).is_some() {
                return Err(format!("duplicate support key: {task}/{state}/{action}"));
            }
        }

        if predictions.len() != truth.len() || truth.keys().any(|key| !predictions.contains_key(key)) {
            return Err("candidate predictions must contain exactly one row per truth state-action key".to_owned());
        }
        if support.len() != truth.len() || truth.keys().any(|key| !support.contains_key(key)) {
            return Err("evaluator support file must contain exactly one row per truth state-action key".to_owned());
        }
        for (key, pred) in &predictions {
            if !states.get(&key.task).map(|known| known.contains(&pred.next_state)).unwrap_or(false) {
                return Err(format!("prediction references unknown state {} for {}/{}/{}", pred.next_state, key.task, key.state, key.action));
            }
        }

        let mut configs = HashMap::new();
        for row in start_rows {
            let task = valid_id(&row[0], "task_id")?;
            let state = valid_id(&row[1], "start_state_id")?;
            let horizon = row[2].parse::<usize>().map_err(|_| format!("invalid horizon: {}", row[2]))?;
            let gamma_ppm = parse_u32(&row[3], "gamma_ppm")?;
            let safety_threshold_ppm = parse_u32(&row[4], "safety_threshold_ppm")?;
            let min_support_count = parse_u64(&row[5], "min_support_count")?;
            let reward_risk_beta_ppm = parse_u32(&row[6], "reward_risk_beta_ppm")?;
            if horizon == 0 || horizon > MAX_HORIZON {
                return Err(format!("horizon must be in 1..={MAX_HORIZON} for {task}"));
            }
            if gamma_ppm > 1_000_000 || safety_threshold_ppm > 1_000_000 || reward_risk_beta_ppm > 1_000_000 {
                return Err(format!("probability/discount/risk multiplier outside 0..=1000000 for {task}"));
            }
            if !states.get(&task).map(|known| known.contains(&state)).unwrap_or(false) {
                return Err(format!("start state {state} is not part of task {task}"));
            }
            if configs.insert(task.clone(), StartConfig {
                state, horizon, gamma_ppm, safety_threshold_ppm, min_support_count, reward_risk_beta_ppm,
            }).is_some() {
                return Err(format!("duplicate start configuration for task: {task}"));
            }
        }
        if configs.is_empty() {
            return Err("starts file has no task configurations".to_owned());
        }
        for key in truth.keys() {
            if !configs.contains_key(&key.task) {
                return Err(format!("environment task {} has no start configuration", key.task));
            }
        }

        Ok(Self { truth, predictions, support, configs, actions, states })
    }

    fn action_is_eligible(&self, task: &str, key: &Key, policy: Policy) -> bool {
        let config = &self.configs[task];
        match policy {
            Policy::Oracle => self.truth[key].safe,
            Policy::Point => self.predictions[key].safe_ppm >= config.safety_threshold_ppm,
            Policy::TrustRegion => {
                let pred = &self.predictions[key];
                let lower_safe_ppm = pred.safe_ppm.saturating_sub(pred.safe_radius_ppm);
                lower_safe_ppm >= config.safety_threshold_ppm
                    && self.support[key] >= config.min_support_count
            }
        }
    }

    fn action_reward(&self, task: &str, key: &Key, policy: Policy) -> f64 {
        let config = &self.configs[task];
        match policy {
            Policy::Oracle => self.truth[key].reward_milli as f64,
            Policy::Point => self.predictions[key].reward_milli as f64,
            Policy::TrustRegion => {
                let pred = &self.predictions[key];
                pred.reward_milli as f64
                    - (config.reward_risk_beta_ppm as f64 / PPM) * pred.reward_radius_milli as f64
            }
        }
    }

    fn action_transition<'a>(&'a self, key: &Key, policy: Policy) -> (&'a str, bool) {
        match policy {
            Policy::Oracle => (&self.truth[key].next_state, self.truth[key].done),
            Policy::Point | Policy::TrustRegion => (&self.predictions[key].next_state, self.predictions[key].done),
        }
    }

    fn best_value(
        &self,
        task: &str,
        state: &str,
        depth: usize,
        policy: Policy,
        memo: &mut HashMap<(String, String, usize, Policy), f64>,
    ) -> f64 {
        if depth == 0 {
            return 0.0;
        }
        let memo_key = (task.to_owned(), state.to_owned(), depth, policy);
        if let Some(value) = memo.get(&memo_key) {
            return *value;
        }
        let value = self.best_action(task, state, depth, policy, memo).0;
        memo.insert(memo_key, value);
        value
    }

    fn best_action(
        &self,
        task: &str,
        state: &str,
        depth: usize,
        policy: Policy,
        memo: &mut HashMap<(String, String, usize, Policy), f64>,
    ) -> (f64, Option<u32>) {
        if depth == 0 || !self.states.get(task).map(|known| known.contains(state)).unwrap_or(false) {
            return (0.0, None);
        }
        let state_key = (task.to_owned(), state.to_owned());
        let Some(action_list) = self.actions.get(&state_key) else {
            return (0.0, None);
        };
        let gamma = self.configs[task].gamma_ppm as f64 / PPM;
        let mut best_value = 0.0_f64; // explicit abstention has utility zero
        let mut best_action = None;

        for action in action_list {
            let key = Key { task: task.to_owned(), state: state.to_owned(), action: *action };
            if !self.action_is_eligible(task, &key, policy) {
                continue;
            }
            let immediate = self.action_reward(task, &key, policy);
            let (next_state, done) = self.action_transition(&key, policy);
            let future = if !done && depth > 1 {
                gamma * self.best_value(task, next_state, depth - 1, policy, memo)
            } else {
                0.0
            };
            let value = immediate + future;
            if value > best_value + 1e-9 {
                best_value = value;
                best_action = Some(*action);
            }
        }
        (best_value, best_action)
    }

    fn diagnostics(&self, task: &str, policy: Policy) -> (u64, f64, f64, usize) {
        let keys: Vec<&Key> = self.truth.keys().filter(|key| key.task == task).collect();
        if keys.is_empty() {
            return (0, 0.0, 0.0, 0);
        }
        let mut covered = 0_usize;
        let mut width_sum = 0.0;
        let mut brier_sum = 0.0;
        let mut false_safe = 0_usize;
        for key in &keys {
            let truth = &self.truth[*key];
            let pred = &self.predictions[*key];
            let error = (truth.reward_milli as i128 - pred.reward_milli as i128).unsigned_abs();
            if error <= pred.reward_radius_milli as u128 {
                covered += 1;
            }
            width_sum += 2.0 * pred.reward_radius_milli as f64;
            let p = pred.safe_ppm as f64 / PPM;
            let y = if truth.safe { 1.0 } else { 0.0 };
            brier_sum += (p - y) * (p - y) * PPM;
            if self.action_is_eligible(task, key, policy) && !truth.safe {
                false_safe += 1;
            }
        }
        (
            ((covered as f64 / keys.len() as f64) * PPM).round() as u64,
            width_sum / keys.len() as f64,
            brier_sum / keys.len() as f64,
            false_safe,
        )
    }

    fn score_task(&self, task: &str, policy: Policy) -> ScoreRow {
        let config = &self.configs[task];
        let mut memo: HashMap<(String, String, usize, Policy), f64> = HashMap::new();
        let oracle = self.best_value(task, &config.state, config.horizon, Policy::Oracle, &mut memo);
        let plan_value = self.best_value(task, &config.state, config.horizon, policy, &mut memo);

        let (coverage, width, brier, false_safe) = self.diagnostics(task, policy);
        let mut state = config.state.clone();
        let mut actual_return = 0.0;
        let mut constrained_return = 0.0;
        let mut discount = 1.0;
        let gamma = config.gamma_ppm as f64 / PPM;
        let mut executed = 0_usize;
        let mut unsafe_actions = 0_usize;
        let mut unsupported_actions = 0_usize;
        let mut reward_abs_error = 0.0;
        let mut transition_mismatches = 0_usize;
        let mut abstained = false;

        for step in 0..config.horizon {
            let remaining = config.horizon - step;
            let (_, action) = self.best_action(task, &state, remaining, policy, &mut memo);
            let Some(action) = action else {
                abstained = true;
                break;
            };
            let key = Key { task: task.to_owned(), state: state.clone(), action };
            let truth = &self.truth[&key];
            let pred = &self.predictions[&key];
            executed += 1;
            if !truth.safe {
                unsafe_actions += 1;
            }
            if self.support[&key] < config.min_support_count {
                unsupported_actions += 1;
            }
            reward_abs_error += (pred.reward_milli as f64 - truth.reward_milli as f64).abs();
            if pred.next_state != truth.next_state || pred.done != truth.done {
                transition_mismatches += 1;
            }
            actual_return += discount * truth.reward_milli as f64;
            if truth.safe {
                constrained_return += discount * truth.reward_milli as f64;
            }
            discount *= gamma;
            if truth.done {
                break;
            }
            state = truth.next_state.clone();
        }

        let regret = (oracle - constrained_return).max(0.0);
        let unsupported_rate_ppm = if executed == 0 {
            0
        } else {
            ((unsupported_actions as f64 / executed as f64) * PPM).round() as u64
        };
        ScoreRow {
            task: task.to_owned(),
            policy: policy.name().to_owned(),
            actual_return_milli: actual_return,
            constrained_return_milli: constrained_return,
            oracle_safe_return_milli: oracle,
            regret_milli: regret,
            risk_adjusted_plan_value_milli: plan_value,
            model_to_rollout_gap_milli: plan_value - constrained_return,
            executed_actions: executed,
            unsafe_actions,
            unsupported_actions,
            unsupported_rate_ppm,
            executed_reward_mae_milli: if executed == 0 { 0.0 } else { reward_abs_error / executed as f64 },
            executed_safety_brier_micro: brier,
            transition_mismatches,
            reward_interval_coverage_ppm: coverage,
            mean_interval_width_milli: width,
            false_safe_predictions: false_safe,
            abstained,
            hard_gate_pass: unsafe_actions == 0,
        }
    }
}

impl ScoreRow {
    fn tsv(&self) -> String {
        format!(
            "{}\t{}\t{:.0}\t{:.0}\t{:.0}\t{:.0}\t{:.0}\t{:.0}\t{}\t{}\t{}\t{}\t{:.0}\t{:.0}\t{}\t{}\t{:.0}\t{}\t{}\t{}",
            self.task,
            self.policy,
            self.actual_return_milli.round(),
            self.constrained_return_milli.round(),
            self.oracle_safe_return_milli.round(),
            self.regret_milli.round(),
            self.risk_adjusted_plan_value_milli.round(),
            self.model_to_rollout_gap_milli.round(),
            self.executed_actions,
            self.unsafe_actions,
            self.unsupported_actions,
            self.unsupported_rate_ppm,
            self.executed_reward_mae_milli.round(),
            self.executed_safety_brier_micro.round(),
            self.transition_mismatches,
            self.reward_interval_coverage_ppm,
            self.mean_interval_width_milli.round(),
            self.false_safe_predictions,
            self.abstained,
            self.hard_gate_pass,
        )
    }
}

fn parse_args() -> Result<HashMap<String, String>, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut parsed = HashMap::new();
    let mut index = 0;
    while index < args.len() {
        let flag = &args[index];
        if !flag.starts_with("--") {
            return Err(format!("expected a flag, got {flag}"));
        }
        if index + 1 >= args.len() {
            return Err(format!("missing value for {flag}"));
        }
        if parsed.insert(flag.clone(), args[index + 1].clone()).is_some() {
            return Err(format!("duplicate flag {flag}"));
        }
        index += 2;
    }
    for flag in ["--truth", "--predictions", "--support", "--starts"] {
        if !parsed.contains_key(flag) {
            return Err(format!("missing required flag {flag}"));
        }
    }
    if parsed.keys().any(|flag| !["--truth", "--predictions", "--support", "--starts"].contains(&flag.as_str())) {
        return Err("unknown flag provided".to_owned());
    }
    Ok(parsed)
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let evaluator = Evaluator::load(&args["--truth"], &args["--predictions"], &args["--support"], &args["--starts"])?;
    let mut tasks: Vec<&String> = evaluator.configs.keys().collect();
    tasks.sort();
    println!("task_id\tpolicy\tactual_return_milli\tconstrained_return_milli\toracle_safe_return_milli\tregret_milli\trisk_adjusted_plan_value_milli\tmodel_to_rollout_gap_milli\texecuted_actions\tunsafe_actions\tunsupported_actions\tunsupported_rate_ppm\texecuted_reward_mae_milli\texecuted_safety_brier_micro\ttransition_mismatches\treward_interval_coverage_ppm\tmean_interval_width_milli\tfalse_safe_predictions\tabstained\thard_gate_pass");
    for task in tasks {
        println!("{}", evaluator.score_task(task, Policy::Point).tsv());
        println!("{}", evaluator.score_task(task, Policy::TrustRegion).tsv());
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("cmaq-003: {error}");
        eprintln!("usage: cmaq-003 --truth FILE --predictions FILE --support FILE --starts FILE");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insert(e: &mut Evaluator, state: &str, action: u32, next: &str, reward: i64, safe: bool, done: bool, radius: u64, support: u64) {
        let key = Key { task: "t".to_owned(), state: state.to_owned(), action };
        e.truth.insert(key.clone(), TruthRow { next_state: next.to_owned(), reward_milli: reward, safe, done });
        e.predictions.insert(key.clone(), PredictionRow {
            next_state: next.to_owned(),
            reward_milli: reward,
            reward_radius_milli: radius,
            safe_ppm: if safe { 1_000_000 } else { 0 },
            safe_radius_ppm: 0,
            done,
        });
        e.support.insert(key, support);
        e.states.entry("t".to_owned()).or_default().insert(state.to_owned());
        e.states.entry("t".to_owned()).or_default().insert(next.to_owned());
        e.actions.entry(("t".to_owned(), state.to_owned())).or_default().push(action);
    }

    fn fixture() -> Evaluator {
        let mut e = Evaluator {
            truth: HashMap::new(),
            predictions: HashMap::new(),
            support: HashMap::new(),
            configs: HashMap::new(),
            actions: HashMap::new(),
            states: HashMap::new(),
        };
        insert(&mut e, "s0", 0, "s1", 1_000, true, false, 0, 5);
        insert(&mut e, "s0", 1, "s2", 2_000, true, false, 0, 0);
        insert(&mut e, "s1", 0, "goal", 5_000, true, true, 0, 5);
        insert(&mut e, "s1", 1, "goal", 1_000, true, true, 0, 0);
        insert(&mut e, "s2", 0, "goal", 500, true, true, 0, 0);
        insert(&mut e, "s2", 1, "goal", 600, true, true, 0, 0);
        for list in e.actions.values_mut() {
            list.sort_unstable();
        }
        e.configs.insert("t".to_owned(), StartConfig {
            state: "s0".to_owned(),
            horizon: 2,
            gamma_ppm: 1_000_000,
            safety_threshold_ppm: 500_000,
            min_support_count: 1,
            reward_risk_beta_ppm: 1_000_000,
        });
        e
    }

    #[test]
    fn exact_model_has_zero_regret_for_both_planners() {
        let e = fixture();
        for policy in [Policy::Point, Policy::TrustRegion] {
            let score = e.score_task("t", policy);
            assert_eq!(score.constrained_return_milli, 6_000.0);
            assert_eq!(score.oracle_safe_return_milli, 6_000.0);
            assert_eq!(score.regret_milli, 0.0);
            assert!(score.hard_gate_pass);
            assert_eq!(score.reward_interval_coverage_ppm, 1_000_000);
        }
    }

    #[test]
    fn point_planner_exploits_overvalued_unsupported_branch() {
        let mut e = fixture();
        let key = Key { task: "t".to_owned(), state: "s0".to_owned(), action: 1 };
        let pred = e.predictions.get_mut(&key).unwrap();
        pred.reward_milli = 9_000;
        pred.reward_radius_milli = 8_000;
        let score = e.score_task("t", Policy::Point);
        assert_eq!(score.constrained_return_milli, 2_600.0);
        assert_eq!(score.regret_milli, 3_400.0);
        assert_eq!(score.model_to_rollout_gap_milli, 7_000.0);
        assert_eq!(score.unsupported_actions, 2);
    }

    #[test]
    fn trust_region_blocks_unsupported_optimism_and_recovers_oracle_return() {
        let mut e = fixture();
        let key = Key { task: "t".to_owned(), state: "s0".to_owned(), action: 1 };
        let pred = e.predictions.get_mut(&key).unwrap();
        pred.reward_milli = 9_000;
        pred.reward_radius_milli = 8_000;
        let score = e.score_task("t", Policy::TrustRegion);
        assert_eq!(score.constrained_return_milli, 6_000.0);
        assert_eq!(score.regret_milli, 0.0);
        assert_eq!(score.unsupported_actions, 0);
    }

    #[test]
    fn uncertainty_penalty_rejects_supported_but_overoptimistic_branch() {
        let mut e = fixture();
        let key = Key { task: "t".to_owned(), state: "s0".to_owned(), action: 1 };
        e.support.insert(key.clone(), 10);
        e.predictions.get_mut(&key).unwrap().reward_milli = 9_000;
        e.predictions.get_mut(&key).unwrap().reward_radius_milli = 8_000;
        let point = e.score_task("t", Policy::Point);
        let robust = e.score_task("t", Policy::TrustRegion);
        assert_eq!(point.constrained_return_milli, 2_600.0);
        assert_eq!(robust.constrained_return_milli, 6_000.0);
    }

    #[test]
    fn conservative_safety_bound_can_force_abstention() {
        let mut e = fixture();
        for pred in e.predictions.values_mut() {
            pred.safe_radius_ppm = 600_000;
        }
        let score = e.score_task("t", Policy::TrustRegion);
        assert_eq!(score.executed_actions, 0);
        assert!(score.abstained);
        assert!(score.hard_gate_pass);
    }

    #[test]
    fn reward_interval_coverage_detects_understated_uncertainty() {
        let mut e = fixture();
        let key = Key { task: "t".to_owned(), state: "s0".to_owned(), action: 0 };
        e.predictions.get_mut(&key).unwrap().reward_milli = 5_000;
        e.predictions.get_mut(&key).unwrap().reward_radius_milli = 100;
        let score = e.score_task("t", Policy::Point);
        assert_eq!(score.reward_interval_coverage_ppm, 833_333);
    }

    #[test]
    fn unsafe_false_safe_prediction_is_a_hard_gate_on_execution() {
        let mut e = fixture();
        let key = Key { task: "t".to_owned(), state: "s0".to_owned(), action: 1 };
        e.truth.get_mut(&key).unwrap().safe = false;
        e.predictions.get_mut(&key).unwrap().safe_ppm = 1_000_000;
        e.predictions.get_mut(&key).unwrap().reward_milli = 9_000;
        let score = e.score_task("t", Policy::Point);
        assert_eq!(score.unsafe_actions, 1);
        assert!(!score.hard_gate_pass);
    }
}
