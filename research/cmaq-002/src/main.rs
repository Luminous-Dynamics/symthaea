use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::process;

const ACTIONS: usize = 3;
const CONTEXTS: usize = 4;
const DEFAULT_PROBE_BUDGET: usize = 2;
const DEFAULT_DECISION_HORIZON: usize = 1;
const REPORT_SCHEMA: &str = "cmaq-002-evoi-report-v0.2";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Family {
    Additive,
    Interaction,
    Threshold,
    Parity,
}

impl Family {
    fn all() -> [Self; 3] {
        [Self::Additive, Self::Interaction, Self::Threshold]
    }

    fn index(self) -> usize {
        match self {
            Self::Additive => 0,
            Self::Interaction => 1,
            Self::Threshold => 2,
            Self::Parity => 3,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Additive => "additive",
            Self::Interaction => "interaction",
            Self::Threshold => "threshold",
            Self::Parity => "parity-heldout",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Outcome {
    reward: i32,
    safe: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Signal {
    positive_reward: bool,
    safe: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Query {
    context: usize,
    action: usize,
}

#[derive(Clone, Debug)]
struct Mechanism {
    family: Family,
    variant: usize,
    base: [i32; ACTIONS],
    x_effect: [i32; ACTIONS],
    y_effect: [i32; ACTIONS],
    interaction: [i32; ACTIONS],
    threshold_gain: [i32; ACTIONS],
    threshold: i32,
    unsafe_action: usize,
    unsafe_context: usize,
}

impl Mechanism {
    fn new(family: Family, variant: usize) -> Self {
        let bases = [[8, 4, 1], [4, 8, 2], [3, 2, 9]];
        let x_effects = [[2, 0, -1], [-1, 2, 0], [0, -1, 2]];
        let y_effects = [[0, 2, -1], [2, -1, 0], [-1, 0, 2]];
        let interactions = [[4, -1, -2], [-2, 4, -1], [-1, -2, 4]];
        let threshold_gains = [[3, -1, 0], [-1, 3, 0], [0, -1, 3]];
        let threshold_values = [-1, 1, 2];

        Self {
            family,
            variant,
            base: bases[variant],
            x_effect: x_effects[variant],
            y_effect: y_effects[variant],
            interaction: interactions[variant],
            threshold_gain: threshold_gains[variant],
            threshold: threshold_values[variant],
            unsafe_action: (variant + family.index()) % ACTIONS,
            unsafe_context: (variant + family.index() + 1) % CONTEXTS,
        }
    }

    fn id(&self) -> String {
        format!("{}-v{}", self.family.name(), self.variant)
    }

    fn observe(&self, query: Query) -> Outcome {
        let (x, y) = context_coordinates(query.context);
        let action = query.action;
        let additive = self.base[action]
            + self.x_effect[action] * x
            + self.y_effect[action] * y;

        let reward = match self.family {
            Family::Additive => additive,
            Family::Interaction => additive + self.interaction[action] * x * y,
            Family::Threshold => {
                let gain = if x + y >= self.threshold {
                    self.threshold_gain[action]
                } else {
                    -self.threshold_gain[action]
                };
                additive + gain
            }
            // Evaluator-only held-out family: categorical context/action parity,
            // intentionally excluded from Family::all() and the candidate library.
            Family::Parity => {
                if (query.context + 2 * action + self.variant) % 4 < 2 {
                    11
                } else {
                    -4
                }
            }
        };

        Outcome {
            reward,
            safe: !(query.action == self.unsafe_action && query.context == self.unsafe_context),
        }
    }

    fn sense(&self, query: Query) -> Signal {
        let outcome = self.observe(query);
        Signal {
            positive_reward: outcome.reward > 0,
            safe: outcome.safe,
        }
    }
}

fn context_coordinates(context: usize) -> (i32, i32) {
    match context {
        0 => (-1, -1),
        1 => (-1, 1),
        2 => (1, -1),
        3 => (1, 1),
        _ => unreachable!("context index is bounded by the benchmark"),
    }
}

fn mechanisms() -> Vec<Mechanism> {
    Family::all()
        .into_iter()
        .flat_map(|family| (0..ACTIONS).map(move |variant| Mechanism::new(family, variant)))
        .collect()
}

fn heldout_mechanisms() -> Vec<Mechanism> {
    (0..ACTIONS)
        .map(|variant| Mechanism::new(Family::Parity, variant))
        .collect()
}

fn all_queries() -> Vec<Query> {
    (0..CONTEXTS)
        .flat_map(|context| (0..ACTIONS).map(move |action| Query { context, action }))
        .collect()
}

fn update_posterior(
    worlds: &[Mechanism],
    posterior: &[usize],
    query: Query,
    observation: Signal,
) -> Result<Vec<usize>, String> {
    let next: Vec<usize> = posterior
        .iter()
        .copied()
        .filter(|index| worlds[*index].sense(query) == observation)
        .collect();
    if next.is_empty() {
        return Err(format!(
            "observation at context {} action {} contradicts every remaining hypothesis; possible regime shift or model misspecification",
            query.context, query.action
        ));
    }
    Ok(next)
}

fn posterior_mean_reward(worlds: &[Mechanism], posterior: &[usize], query: Query) -> f64 {
    posterior
        .iter()
        .map(|index| f64::from(worlds[*index].observe(query).reward))
        .sum::<f64>()
        / posterior.len() as f64
}

fn action_is_robustly_safe(worlds: &[Mechanism], posterior: &[usize], context: usize, action: usize) -> bool {
    posterior.iter().all(|index| {
        worlds[*index]
            .observe(Query { context, action })
            .safe
    })
}

fn candidate_action(worlds: &[Mechanism], posterior: &[usize], context: usize) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for action in 0..ACTIONS {
        if !action_is_robustly_safe(worlds, posterior, context, action) {
            continue;
        }
        let expected_reward = posterior_mean_reward(
            worlds,
            posterior,
            Query { context, action },
        );
        if expected_reward <= 0.0 {
            continue;
        }
        if best.map(|(_, value)| expected_reward > value).unwrap_or(true) {
            best = Some((action, expected_reward));
        }
    }
    best.map(|(action, _)| action)
}

fn posterior_decision_value(worlds: &[Mechanism], posterior: &[usize]) -> f64 {
    let total: f64 = (0..CONTEXTS)
        .map(|context| {
            candidate_action(worlds, posterior, context)
                .map(|action| {
                    posterior_mean_reward(worlds, posterior, Query { context, action })
                })
                .unwrap_or(0.0)
        })
        .sum();
    total / CONTEXTS as f64
}

fn actual_decision_value(world: &Mechanism, worlds: &[Mechanism], posterior: &[usize]) -> (f64, usize) {
    let mut total = 0.0;
    let mut unsafe_actions = 0usize;
    for context in 0..CONTEXTS {
        if let Some(action) = candidate_action(worlds, posterior, context) {
            let outcome = world.observe(Query { context, action });
            if outcome.safe {
                total += f64::from(outcome.reward);
            } else {
                unsafe_actions += 1;
                total -= 100.0;
            }
        }
    }
    (total / CONTEXTS as f64, unsafe_actions)
}

fn oracle_decision_value(world: &Mechanism) -> f64 {
    let total: i32 = (0..CONTEXTS)
        .map(|context| {
            (0..ACTIONS)
                .map(|action| world.observe(Query { context, action }))
                .filter(|outcome| outcome.safe)
                .map(|outcome| outcome.reward)
                .max()
                .unwrap_or(0)
                .max(0)
        })
        .sum();
    f64::from(total) / CONTEXTS as f64
}

fn expected_value_of_probe(worlds: &[Mechanism], posterior: &[usize], query: Query) -> f64 {
    if posterior.len() <= 1 {
        return 0.0;
    }
    let before = posterior_decision_value(worlds, posterior);
    let mut groups: BTreeMap<Signal, Vec<usize>> = BTreeMap::new();
    for index in posterior {
        groups
            .entry(worlds[*index].sense(query))
            .or_default()
            .push(*index);
    }
    let after = groups
        .values()
        .map(|group| {
            group.len() as f64 / posterior.len() as f64
                * posterior_decision_value(worlds, group)
        })
        .sum::<f64>();
    after - before
}

fn entropy_from_count(count: usize) -> f64 {
    if count <= 1 {
        0.0
    } else {
        (count as f64).ln()
    }
}

fn expected_information_gain(worlds: &[Mechanism], posterior: &[usize], query: Query) -> f64 {
    if posterior.len() <= 1 {
        return 0.0;
    }
    let mut groups: BTreeMap<Signal, usize> = BTreeMap::new();
    for index in posterior {
        *groups.entry(worlds[*index].sense(query)).or_default() += 1;
    }
    let before = entropy_from_count(posterior.len());
    let expected_after = groups
        .values()
        .map(|count| {
            *count as f64 / posterior.len() as f64 * entropy_from_count(*count)
        })
        .sum::<f64>();
    before - expected_after
}

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
}

#[derive(Clone, Copy, Debug)]
enum Strategy {
    NoProbe,
    Fixed,
    Random,
    MaxInformation,
    Evoi,
    Oracle,
}

impl Strategy {
    fn all() -> [Self; 6] {
        [
            Self::NoProbe,
            Self::Fixed,
            Self::Random,
            Self::MaxInformation,
            Self::Evoi,
            Self::Oracle,
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::NoProbe => "no_probe",
            Self::Fixed => "fixed",
            Self::Random => "random",
            Self::MaxInformation => "max_information",
            Self::Evoi => "evoi",
            Self::Oracle => "oracle",
        }
    }
}

fn select_probe(
    strategy: Strategy,
    worlds: &[Mechanism],
    posterior: &[usize],
    used: &BTreeSet<Query>,
    rng: &mut SplitMix64,
    decision_horizon: usize,
    probe_cost: f64,
) -> Option<Query> {
    let available: Vec<Query> = all_queries()
        .into_iter()
        .filter(|query| !used.contains(query))
        .collect();
    if available.is_empty() || posterior.len() <= 1 {
        return None;
    }

    match strategy {
        Strategy::NoProbe | Strategy::Oracle => None,
        Strategy::Fixed => {
            let preferred = [
                Query { context: 0, action: 0 },
                Query { context: 3, action: 2 },
                Query { context: 1, action: 1 },
                Query { context: 2, action: 0 },
            ];
            preferred
                .into_iter()
                .find(|query| available.contains(query))
                .or_else(|| available.first().copied())
        }
        Strategy::Random => {
            let index = (rng.next_u64() % available.len() as u64) as usize;
            available.get(index).copied()
        }
        Strategy::MaxInformation => available.into_iter().max_by(|left, right| {
            expected_information_gain(worlds, posterior, *left)
                .total_cmp(&expected_information_gain(worlds, posterior, *right))
                .then_with(|| right.cmp(left))
        }),
        Strategy::Evoi => {
            let best = available.into_iter().max_by(|left, right| {
                expected_value_of_probe(worlds, posterior, *left)
                    .total_cmp(&expected_value_of_probe(worlds, posterior, *right))
                    .then_with(|| right.cmp(left))
            })?;
            let gain = expected_value_of_probe(worlds, posterior, best);
            if gain * decision_horizon as f64 > probe_cost { Some(best) } else { None }
        }
    }
}

#[derive(Clone, Debug)]
struct OpenClassControl {
    tested_mechanisms: usize,
    detected: usize,
    mean_queries_to_detection: f64,
    max_queries_to_detection: usize,
}

fn evaluate_open_class_control() -> OpenClassControl {
    let candidates = mechanisms();
    let mut detected = 0usize;
    let mut query_count_total = 0usize;
    let mut max_queries_to_detection = 0usize;
    let heldout = heldout_mechanisms();

    for hidden_world in &heldout {
        let mut posterior: Vec<usize> = (0..candidates.len()).collect();
        for (position, query) in all_queries().into_iter().enumerate() {
            match update_posterior(
                &candidates,
                &posterior,
                query,
                hidden_world.sense(query),
            ) {
                Ok(next) => posterior = next,
                Err(_) => {
                    let queries_used = position + 1;
                    detected += 1;
                    query_count_total += queries_used;
                    max_queries_to_detection = max_queries_to_detection.max(queries_used);
                    break;
                }
            }
        }
    }

    OpenClassControl {
        tested_mechanisms: heldout.len(),
        detected,
        mean_queries_to_detection: if detected == 0 {
            0.0
        } else {
            query_count_total as f64 / detected as f64
        },
        max_queries_to_detection,
    }
}

#[derive(Clone, Debug)]
struct Summary {
    strategy: &'static str,
    mean_probes: f64,
    mean_posterior_size: f64,
    mean_decision_utility: f64,
    mean_net_utility: f64,
    mean_regret: f64,
    mean_net_regret: f64,
    unsafe_actions: usize,
    contradictions: usize,
}

fn evaluate_strategy(strategy: Strategy, seed: u64, probe_budget: usize, decision_horizon: usize, probe_cost: f64) -> Summary {
    let worlds = mechanisms();
    let all_indices: Vec<usize> = (0..worlds.len()).collect();
    let mut probe_count = 0usize;
    let mut posterior_size_total = 0usize;
    let mut utility_total = 0.0;
    let mut net_utility_total = 0.0;
    let mut regret_total = 0.0;
    let mut net_regret_total = 0.0;
    let mut unsafe_actions = 0usize;
    let mut contradictions = 0usize;

    for true_index in 0..worlds.len() {
        let true_world = &worlds[true_index];
        let mut posterior = if matches!(strategy, Strategy::Oracle) {
            vec![true_index]
        } else {
            all_indices.clone()
        };
        let mut used = BTreeSet::new();
        let mut rng = SplitMix64::new(seed ^ ((true_index as u64 + 1) * 0x9E37_79B9));

        if !matches!(strategy, Strategy::Oracle) {
            for _ in 0..probe_budget {
                let Some(query) = select_probe(
                    strategy,
                    &worlds,
                    &posterior,
                    &used,
                    &mut rng,
                    decision_horizon,
                    probe_cost,
                ) else {
                    break;
                };
                used.insert(query);
                let observed = true_world.sense(query);
                match update_posterior(&worlds, &posterior, query, observed) {
                    Ok(next) => posterior = next,
                    Err(_) => {
                        contradictions += 1;
                        break;
                    }
                }
                probe_count += 1;
            }
        }

        posterior_size_total += posterior.len();
        let (utility, unsafe_count) = actual_decision_value(true_world, &worlds, &posterior);
        unsafe_actions += unsafe_count;
        let oracle_utility = oracle_decision_value(true_world);
        let net_utility = utility - used.len() as f64 * probe_cost / decision_horizon as f64;
        utility_total += utility;
        net_utility_total += net_utility;
        regret_total += (oracle_utility - utility).max(0.0);
        net_regret_total += (oracle_utility - net_utility).max(0.0);
    }

    let count = worlds.len() as f64;
    Summary {
        strategy: strategy.name(),
        mean_probes: probe_count as f64 / count,
        mean_posterior_size: posterior_size_total as f64 / count,
        mean_decision_utility: utility_total / count,
        mean_net_utility: net_utility_total / count,
        mean_regret: regret_total / count,
        mean_net_regret: net_regret_total / count,
        unsafe_actions,
        contradictions,
    }
}

fn parse_args() -> Result<(u64, usize, usize, f64, bool), String> {
    let mut seed = 7u64;
    let mut budget = DEFAULT_PROBE_BUDGET;
    let mut cost = 0.25f64;
    let mut decision_horizon = DEFAULT_DECISION_HORIZON;
    let mut help = false;
    let args: Vec<String> = env::args().skip(1).collect();
    let mut position = 0usize;
    while position < args.len() {
        match args[position].as_str() {
            "--help" | "-h" => help = true,
            "--seed" => {
                position += 1;
                seed = args.get(position).ok_or("--seed requires a value")?
                    .parse().map_err(|_| "--seed must be an unsigned integer".to_owned())?;
            }
            "--probe-budget" => {
                position += 1;
                budget = args.get(position).ok_or("--probe-budget requires a value")?
                    .parse().map_err(|_| "--probe-budget must be an integer".to_owned())?;
                if budget > 8 {
                    return Err("--probe-budget must be in 0..=8".to_owned());
                }
            }
            "--probe-cost" => {
                position += 1;
                cost = args.get(position).ok_or("--probe-cost requires a value")?
                    .parse().map_err(|_| "--probe-cost must be a number".to_owned())?;
                if !cost.is_finite() || cost < 0.0 || cost > 100.0 {
                    return Err("--probe-cost must be finite and in 0..=100".to_owned());
                }
            }
            "--decision-horizon" => {
                position += 1;
                decision_horizon = args.get(position).ok_or("--decision-horizon requires a value")?
                    .parse().map_err(|_| "--decision-horizon must be an integer in 1..=10000".to_owned())?;
                if decision_horizon == 0 || decision_horizon > 10_000 {
                    return Err("--decision-horizon must be in 1..=10000".to_owned());
                }
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
        position += 1;
    }
    Ok((seed, budget, decision_horizon, cost, help))
}

fn print_help() {
    println!(
        "CMAQ-002 finite-hypothesis active causal learning benchmark\n\n\
         Usage: cmaq-002-evoi [--seed N] [--probe-budget N] [--decision-horizon N] [--probe-cost X]\n\n\
         Options:\n\
           --seed N          random-probe seed (default 7)\n\
           --probe-budget N  maximum probes per mechanism, 0..=8 (default 2)\n\
           --probe-cost X    utility units charged per probe, 0..=100 (default 0.25)\n\
           --decision-horizon N  downstream decision rounds, 1..=10000 (default 1)\n\\
           --help            print this help"
    );
}

fn run() -> Result<(), String> {
    let (seed, probe_budget, decision_horizon, probe_cost, help) = parse_args()?;
    if help {
        print_help();
        return Ok(());
    }
    let worlds = mechanisms();
    print!(
        "{{\"schema_version\":\"{REPORT_SCHEMA}\",\"seed\":{},\"mechanisms\":{},\"families\":3,\"contexts\":{},\"actions\":{},\"probe_budget\":{},\"decision_horizon\":{},\"probe_cost\":{},\"results\":[",
        seed, worlds.len(), CONTEXTS, ACTIONS, probe_budget, decision_horizon, probe_cost
    );
    for (position, strategy) in Strategy::all().into_iter().enumerate() {
        let summary = evaluate_strategy(strategy, seed, probe_budget, decision_horizon, probe_cost);
        if position > 0 {
            print!(",");
        }
        print!(
            "{{\"strategy\":\"{}\",\"mean_probes\":{:.4},\"mean_posterior_size\":{:.4},\"mean_decision_utility\":{:.4},\"mean_net_utility\":{:.4},\"mean_regret\":{:.4},\"mean_net_regret\":{:.4},\"unsafe_actions\":{},\"contradictions\":{}}}",
            summary.strategy,
            summary.mean_probes,
            summary.mean_posterior_size,
            summary.mean_decision_utility,
            summary.mean_net_utility,
            summary.mean_regret,
            summary.mean_net_regret,
            summary.unsafe_actions,
            summary.contradictions
        );
    }
    let open_class = evaluate_open_class_control();
    print!(
        "],\"open_class_control\":{{\"tested_mechanisms\":{},\"detected\":{},\"mean_queries_to_detection\":{:.4},\"max_queries_to_detection\":{}}}",
        open_class.tested_mechanisms,
        open_class.detected,
        open_class.mean_queries_to_detection,
        open_class.max_queries_to_detection
    );
    println!("}}");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hypothesis_library_covers_three_families_and_nine_mechanisms() {
        let worlds = mechanisms();
        assert_eq!(worlds.len(), 9);
        for family in Family::all() {
            assert_eq!(worlds.iter().filter(|world| world.family == family).count(), 3);
        }
        assert_eq!(worlds.iter().map(Mechanism::id).collect::<BTreeSet<_>>().len(), 9);
    }

    #[test]
    fn family_laws_are_structurally_distinct() {
        let worlds = mechanisms();
        let queries = all_queries();
        for (left, right) in [
            (&worlds[0], &worlds[3]),
            (&worlds[0], &worlds[6]),
            (&worlds[3], &worlds[6]),
        ] {
            assert!(queries.iter().any(|query| left.observe(*query) != right.observe(*query)));
        }
    }

    #[test]
    fn candidate_observation_hides_reward_magnitude() {
        let worlds = mechanisms();
        let mut found = false;
        for left in 0..worlds.len() {
            for right in (left + 1)..worlds.len() {
                for query in all_queries() {
                    let left_outcome = worlds[left].observe(query);
                    let right_outcome = worlds[right].observe(query);
                    if left_outcome.reward != right_outcome.reward
                        && worlds[left].sense(query) == worlds[right].sense(query)
                    {
                        found = true;
                    }
                }
            }
        }
        assert!(found, "some worlds must share a binary observation despite differing reward magnitudes");
    }

    #[test]
    fn posterior_update_keeps_true_mechanism_for_every_probe() {
        let worlds = mechanisms();
        let initial: Vec<usize> = (0..worlds.len()).collect();
        for true_index in 0..worlds.len() {
            for query in all_queries() {
                let observation = worlds[true_index].sense(query);
                let posterior = update_posterior(&worlds, &initial, query, observation)
                    .expect("truth remains in the candidate set");
                assert!(posterior.contains(&true_index));
                assert!(!posterior.is_empty());
            }
        }
    }

    #[test]
    fn contradictory_observation_signals_possible_regime_shift() {
        let worlds = mechanisms();
        let query = all_queries()
            .into_iter()
            .find(|query| worlds[0].observe(*query) != worlds[1].observe(*query))
            .expect("two mechanisms differ on at least one query");
        let error = update_posterior(
            &worlds,
            &[0],
            query,
            worlds[1].sense(query),
        )
        .expect_err("incompatible outcome must not silently empty posterior");
        assert!(error.contains("possible regime shift"));
    }

    #[test]
    fn heldout_mechanism_family_triggers_model_misspecification_signal() {
        let candidates = mechanisms();
        for hidden_world in heldout_mechanisms() {
            let mut posterior: Vec<usize> = (0..candidates.len()).collect();
            let mut detected = false;
            for query in all_queries() {
                match update_posterior(
                    &candidates,
                    &posterior,
                    query,
                    hidden_world.sense(query),
                ) {
                    Ok(next) => posterior = next,
                    Err(error) => {
                        assert!(error.contains("model misspecification"));
                        detected = true;
                        break;
                    }
                }
            }
            assert!(
                detected,
                "held-out family must not be silently represented as an in-class mechanism"
            );
        }
    }

    #[test]
    fn expected_value_of_information_is_nonnegative_with_zero_cost() {
        let worlds = mechanisms();
        let posterior: Vec<usize> = (0..worlds.len()).collect();
        for query in all_queries() {
            assert!(
                expected_value_of_probe(&worlds, &posterior, query) >= -1e-9,
                "expected information should not reduce the optimal posterior decision value"
            );
        }
    }

    #[test]
    fn evoi_selects_best_probe_or_stops_when_net_value_does_not_cover_cost() {
        let worlds = mechanisms();
        let posterior: Vec<usize> = (0..worlds.len()).collect();
        let used = BTreeSet::new();
        let mut rng = SplitMix64::new(11);
        let chosen = select_probe(
            Strategy::Evoi,
            &worlds,
            &posterior,
            &used,
            &mut rng,
            1,
            0.0,
        ).expect("at least one probe should have positive decision value");
        let chosen_gain = expected_value_of_probe(&worlds, &posterior, chosen);
        for query in all_queries() {
            assert!(chosen_gain + 1e-9 >= expected_value_of_probe(&worlds, &posterior, query));
        }

        let stop = select_probe(
            Strategy::Evoi,
            &worlds,
            &[0],
            &used,
            &mut rng,
            1,
            0.0,
        );
        assert!(stop.is_none());
    }

    #[test]
    fn robust_policy_never_selects_an_action_unsafe_in_the_true_hypothesis() {
        let worlds = mechanisms();
        for strategy in Strategy::all() {
            let summary = evaluate_strategy(strategy, 4_242, 2, 1, 0.25);
            assert_eq!(summary.unsafe_actions, 0, "strategy {}", strategy.name());
            assert_eq!(summary.contradictions, 0, "strategy {}", strategy.name());
        }
    }

    #[test]
    fn open_class_control_detects_every_heldout_mechanism() {
        let report = evaluate_open_class_control();
        assert_eq!(report.tested_mechanisms, 3);
        assert_eq!(report.detected, report.tested_mechanisms);
        assert!(report.mean_queries_to_detection > 0.0);
        assert!(report.max_queries_to_detection <= all_queries().len());
    }

    #[test]
    fn evoi_beats_information_only_on_the_calibration_panel() {
        let evoi = evaluate_strategy(Strategy::Evoi, 4_242, 2, 1, 0.25);
        let max_information = evaluate_strategy(Strategy::MaxInformation, 4_242, 2, 1, 0.25);
        let no_probe = evaluate_strategy(Strategy::NoProbe, 4_242, 2, 1, 0.25);
        assert!(
            evoi.mean_net_regret < max_information.mean_net_regret,
            "calibration panel must distinguish decision-value probing from information-only probing"
        );
        assert!(
            evoi.mean_net_regret < no_probe.mean_net_regret,
            "calibration panel must show value over no-probe baseline"
        );
    }

    #[test]
    fn evoi_scales_probe_value_with_downstream_decision_horizon() {
        let worlds = mechanisms();
        let posterior: Vec<usize> = (0..worlds.len()).collect();
        let max_gain = all_queries()
            .iter()
            .map(|query| expected_value_of_probe(&worlds, &posterior, *query))
            .fold(0.0_f64, f64::max);
        assert!(max_gain > 0.0);

        let mut rng = SplitMix64::new(17);
        let used = BTreeSet::new();
        let cost = max_gain * 2.0;
        let one_shot = select_probe(
            Strategy::Evoi, &worlds, &posterior, &used, &mut rng, 1, cost
        );
        let reused = select_probe(
            Strategy::Evoi, &worlds, &posterior, &used, &mut rng, 3, cost
        );
        assert!(one_shot.is_none(), "one decision does not repay twice the maximum per-decision value");
        assert!(reused.is_some(), "three downstream decisions can repay the same probe");
    }

    #[test]
    fn oracle_has_zero_regret_and_probe_cost_is_accounted_for() {
        let oracle = evaluate_strategy(Strategy::Oracle, 7, 2, 1, 0.25);
        assert!(oracle.mean_regret.abs() < 1e-9);
        assert!(oracle.mean_net_regret.abs() < 1e-9);
        let no_probe = evaluate_strategy(Strategy::NoProbe, 7, 2, 1, 0.25);
        assert_eq!(no_probe.mean_probes, 0.0);
        let fixed = evaluate_strategy(Strategy::Fixed, 7, 2, 1, 0.25);
        assert!(fixed.mean_net_regret + 1e-9 >= fixed.mean_regret);
    }

    #[test]
    fn seeded_report_is_reproducible() {
        let first = evaluate_strategy(Strategy::Random, 12_345, 2, 1, 0.25);
        let second = evaluate_strategy(Strategy::Random, 12_345, 2, 1, 0.25);
        assert_eq!(first.strategy, second.strategy);
        assert_eq!(first.mean_probes, second.mean_probes);
        assert_eq!(first.mean_regret, second.mean_regret);
        assert_eq!(first.mean_net_regret, second.mean_net_regret);
    }
}
