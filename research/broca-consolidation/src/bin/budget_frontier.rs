// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Budget-frontier extension for the governed-consolidation harness.
// This is a synthetic decision instrument, not a reproduction of ConsolidationBench.

use std::env;
use std::process::{self, Command};

const PROTOCOL: &str = "GMC-BUDGET-FRONTIER-V1";
const COST_PER_TOKEN: f64 = 0.02;
const DEFAULT_BUDGETS: &[usize] = &[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Action {
    Raw,
    Provisional,
    Summary,
    Merge,
    Abstract,
    Rewrite,
    Retire,
}

impl Action {
    const ALL: [Self; 7] = [
        Self::Raw, Self::Provisional, Self::Summary, Self::Merge,
        Self::Abstract, Self::Rewrite, Self::Retire,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Provisional => "provisional",
            Self::Summary => "summary",
            Self::Merge => "merge",
            Self::Abstract => "abstract",
            Self::Rewrite => "rewrite",
            Self::Retire => "retire",
        }
    }

    fn cost(self, raw_tokens: usize) -> usize {
        match self {
            Self::Raw => raw_tokens,
            Self::Provisional => raw_tokens.div_ceil(4) * 3,
            Self::Summary => raw_tokens.div_ceil(5) * 2,
            Self::Merge => raw_tokens.div_ceil(4),
            Self::Abstract => raw_tokens.div_ceil(3),
            Self::Rewrite => raw_tokens.div_ceil(5) * 3,
            Self::Retire => 0,
        }
    }

    fn is_promotion(self) -> bool {
        matches!(self, Self::Summary | Self::Merge | Self::Abstract | Self::Rewrite)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    ExactFact,
    RepeatedCluster,
    Procedure,
    Invariant,
    PoisonedRule,
    ObsoleteFact,
}

impl Kind {
    const ALL: [Self; 6] = [
        Self::ExactFact,
        Self::RepeatedCluster,
        Self::Procedure,
        Self::Invariant,
        Self::PoisonedRule,
        Self::ObsoleteFact,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::ExactFact => "exact_fact",
            Self::RepeatedCluster => "repeated_cluster",
            Self::Procedure => "procedure",
            Self::Invariant => "invariant",
            Self::PoisonedRule => "poisoned_rule",
            Self::ObsoleteFact => "obsolete_fact",
        }
    }

    fn load_bearing(self) -> bool {
        matches!(self, Self::ExactFact | Self::RepeatedCluster | Self::Procedure | Self::Invariant)
    }

    fn unsafe_or_stale(self) -> bool {
        matches!(self, Self::PoisonedRule | Self::ObsoleteFact)
    }

    // Synthetic per-future-query utility. Negative values represent harmful use.
    // These values are an explicit toy scoring contract, not empirical estimates.
    fn query_utility(self, action: Action) -> f64 {
        match (self, action) {
            (Self::ExactFact, Action::Raw) => 1.00,
            (Self::ExactFact, Action::Provisional) => 0.65,
            (Self::ExactFact, Action::Summary) => 0.40,
            (Self::ExactFact, Action::Merge) => 0.20,
            (Self::ExactFact, Action::Abstract) => 0.15,
            (Self::ExactFact, Action::Rewrite) => 0.45,
            (Self::RepeatedCluster, Action::Raw) => 0.68,
            (Self::RepeatedCluster, Action::Provisional) => 0.45,
            (Self::RepeatedCluster, Action::Summary) => 0.82,
            (Self::RepeatedCluster, Action::Merge) => 0.90,
            (Self::RepeatedCluster, Action::Abstract) => 0.55,
            (Self::RepeatedCluster, Action::Rewrite) => 0.60,
            (Self::Procedure, Action::Raw) => 0.62,
            (Self::Procedure, Action::Provisional) => 0.45,
            (Self::Procedure, Action::Summary) => 0.55,
            (Self::Procedure, Action::Merge) => 0.60,
            (Self::Procedure, Action::Abstract) => 0.92,
            (Self::Procedure, Action::Rewrite) => 0.50,
            (Self::Invariant, Action::Raw) => 0.65,
            (Self::Invariant, Action::Provisional) => 0.50,
            (Self::Invariant, Action::Summary) => 0.70,
            (Self::Invariant, Action::Merge) => 0.62,
            (Self::Invariant, Action::Abstract) => 0.95,
            (Self::Invariant, Action::Rewrite) => 0.55,
            (Self::PoisonedRule, Action::Raw) => -0.15,
            (Self::PoisonedRule, Action::Provisional) => -0.05,
            (Self::PoisonedRule, Action::Summary) => -0.25,
            (Self::PoisonedRule, Action::Merge) => -0.35,
            (Self::PoisonedRule, Action::Abstract) => -0.50,
            (Self::PoisonedRule, Action::Rewrite) => -0.20,
            (Self::ObsoleteFact, Action::Raw) => -0.15,
            (Self::ObsoleteFact, Action::Provisional) => -0.05,
            (Self::ObsoleteFact, Action::Summary) => -0.20,
            (Self::ObsoleteFact, Action::Merge) => -0.25,
            (Self::ObsoleteFact, Action::Abstract) => -0.30,
            (Self::ObsoleteFact, Action::Rewrite) => -0.12,
            (_, Action::Retire) => 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Observed {
    repeats: usize,
    independent_sources: usize,
    contradictions: usize,
    recency: usize,
    correction_seen: bool,
    procedure_successes: usize,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    id: usize,
    kind: Kind, // evaluator-only; never passed to a policy
    observed: Observed,
    raw_tokens: usize,
    future_queries: usize, // evaluator-only future reuse count
}

#[derive(Clone, Copy, Debug)]
struct Config {
    source_sha: [u8; 40],
    seed: u64,
    candidates: usize,
}

#[derive(Clone, Copy, Debug)]
struct Decision {
    action: Action,
    tokens: usize,
}

#[derive(Debug)]
struct Summary {
    policy: &'static str,
    budget: usize,
    raw_demand: usize,
    used_tokens: usize,
    stored_candidates: usize,
    future_queries: usize,
    synthetic_query_score: f64,
    task_utility: f64,
    net_utility: f64,
    oracle_net_utility: f64,
    regret: f64,
    load_bearing_coverage: f64,
    bad_retained: usize,
    bad_promoted: usize,
    compression_ratio: f64,
    utility_per_token: f64,
}

#[derive(Clone)]
struct Rng { state: u64 }

impl Rng {
    fn new(seed: u64) -> Self { Self { state: seed } }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        assert!(bound > 0);
        let bound = bound as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let product = u128::from(self.next()) * u128::from(bound);
            if product as u64 >= threshold {
                return (product >> 64) as usize;
            }
        }
    }
}

fn perturb(value: usize, amount: usize, rng: &mut Rng) -> usize {
    let delta = rng.below(amount * 2 + 1) as isize - amount as isize;
    (value as isize + delta).max(0) as usize
}

fn base_observed(kind: Kind, rng: &mut Rng) -> Observed {
    let (repeats, sources, contradictions, recency, correction, successes) = match kind {
        Kind::ExactFact => (1, 1, 0, 9, false, 1),
        Kind::RepeatedCluster => (6, 5, 0, 8, false, 3),
        Kind::Procedure => (5, 4, 0, 7, false, 4),
        Kind::Invariant => (7, 6, 0, 8, false, 6),
        Kind::PoisonedRule => (5, 2, 1, 9, true, 4),
        Kind::ObsoleteFact => (2, 2, 3, 2, true, 1),
    };
    Observed {
        repeats: perturb(repeats, 2, rng),
        independent_sources: perturb(sources, 2, rng),
        contradictions: perturb(contradictions, 1, rng),
        recency: perturb(recency, 2, rng).min(10),
        correction_seen: correction ^ (rng.below(7) == 0),
        procedure_successes: perturb(successes, 2, rng),
    }
}

fn build_fixture(seed: u64, count: usize) -> Vec<Candidate> {
    let mut rng = Rng::new(seed);
    (0..count)
        .map(|id| {
            let kind = Kind::ALL[(id + rng.below(Kind::ALL.len())) % Kind::ALL.len()];
            Candidate {
                id,
                kind,
                observed: base_observed(kind, &mut rng),
                raw_tokens: 8 + rng.below(33),
                future_queries: 1 + rng.below(6),
            }
        })
        .collect()
}

fn valid_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut sha: Option<String> = None;
    let mut seed = 17u64;
    let mut candidates = 120usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--source-sha" | "--seed" | "--candidates" => {
                let name = args[i].clone();
                let value = args.get(i + 1).ok_or_else(|| format!("{name} needs a value"))?;
                match name.as_str() {
                    "--source-sha" => sha = Some(value.clone()),
                    "--seed" => seed = value.parse().map_err(|_| "seed must be an unsigned integer".to_string())?,
                    "--candidates" => candidates = value.parse().map_err(|_| "candidates must be an integer".to_string())?,
                    _ => unreachable!(),
                }
                i += 2;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    let sha = sha.ok_or_else(|| "--source-sha is required".to_string())?;
    if !valid_sha(&sha) { return Err("--source-sha must be 40 hexadecimal characters".to_string()); }
    if !(30..=1000).contains(&candidates) { return Err("--candidates must be in 30..=1000".to_string()); }

    let actual = Command::new("git").args(["rev-parse", "HEAD"]).output()
        .map_err(|e| format!("cannot run git rev-parse HEAD: {e}"))?;
    if !actual.status.success() { return Err("git rev-parse HEAD failed".to_string()); }
    let actual_sha = String::from_utf8_lossy(&actual.stdout).trim().to_owned();
    if actual_sha != sha { return Err(format!("source SHA mismatch: requested {sha}, HEAD is {actual_sha}")); }

    let mut bytes = [0u8; 40];
    bytes.copy_from_slice(sha.as_bytes());
    Ok(Config { source_sha: bytes, seed, candidates })
}

fn select_action(policy: &str, observed: Observed, id: usize, raw_pressure: bool) -> Action {
    match policy {
        "no_memory" | "retire_everything" => Action::Retire,
        "always_raw" => Action::Raw,
        "always_summary" => Action::Summary,
        "fixed_interval" => if id % 4 == 3 { Action::Summary } else { Action::Raw },
        "dedup_merge" => {
            if observed.repeats >= 3 && observed.independent_sources >= 2 { Action::Merge } else { Action::Raw }
        }
        "feature_threshold" => {
            if observed.contradictions >= 2 || observed.correction_seen { Action::Provisional }
            else if observed.repeats >= 3 && observed.independent_sources >= 2 { Action::Summary }
            else { Action::Raw }
        }
        "budget_adaptive" => {
            if !raw_pressure { return Action::Raw; }
            if observed.correction_seen || observed.contradictions >= 2 {
                Action::Provisional
            } else if observed.procedure_successes >= 4 && observed.independent_sources >= 3 {
                Action::Abstract
            } else if observed.repeats >= 4 && observed.independent_sources >= 3 {
                Action::Merge
            } else if observed.repeats >= 2 {
                Action::Summary
            } else {
                Action::Raw
            }
        }
        "promote_everything" => Action::Abstract,
        _ => Action::Retire,
    }
}

// Heuristic only; no candidate kind, future queries or oracle value enters ranking.
fn estimated_priority(observed: Observed, action: Action, raw_tokens: usize) -> f64 {
    let support = observed.repeats as f64 * 0.12
        + observed.independent_sources as f64 * 0.25
        + observed.procedure_successes as f64 * 0.18
        + observed.recency as f64 * 0.03
        - observed.contradictions as f64 * 0.40
        - if observed.correction_seen { 0.45 } else { 0.0 };
    let cost = action.cost(raw_tokens).max(1) as f64;
    support.max(0.01) / cost
}

fn choose_policy(candidates: &[Candidate], budget: usize, policy: &'static str) -> Vec<Decision> {
    let raw_demand: usize = candidates.iter().map(|c| c.raw_tokens).sum();
    let raw_pressure = raw_demand > budget;
    let mut proposed: Vec<(usize, Action, f64)> = candidates.iter().map(|candidate| {
        let action = select_action(policy, candidate.observed, candidate.id, raw_pressure);
        let priority = if action == Action::Retire { 0.0 }
            else { estimated_priority(candidate.observed, action, candidate.raw_tokens) };
        (candidate.id, action, priority)
    }).collect();
    proposed.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));

    let mut result = vec![Decision { action: Action::Retire, tokens: 0 }; candidates.len()];
    let mut remaining = budget;
    for (id, action, _) in proposed {
        if action == Action::Retire { continue; }
        let cost = action.cost(candidates[id].raw_tokens);
        if cost <= remaining {
            result[id] = Decision { action, tokens: cost };
            remaining -= cost;
        }
    }
    result
}

fn candidate_net(candidate: Candidate, action: Action) -> f64 {
    let token_cost = action.cost(candidate.raw_tokens);
    candidate.kind.query_utility(action) * candidate.future_queries as f64
        - token_cost as f64 * COST_PER_TOKEN
}

fn oracle_select(candidates: &[Candidate], budget: usize) -> Vec<Decision> {
    let n = candidates.len();
    let mut dp = vec![vec![0.0f64; budget + 1]; n + 1];
    let mut pick = vec![vec![Action::Retire; budget + 1]; n + 1];

    for i in 1..=n {
        let candidate = candidates[i - 1];
        for cap in 0..=budget {
            let mut best = dp[i - 1][cap]; // retiring the candidate
            let mut chosen = Action::Retire;
            for action in Action::ALL.into_iter().filter(|a| *a != Action::Retire) {
                let cost = action.cost(candidate.raw_tokens);
                if cost <= cap {
                    let value = dp[i - 1][cap - cost] + candidate_net(candidate, action);
                    if value > best {
                        best = value;
                        chosen = action;
                    }
                }
            }
            dp[i][cap] = best;
            pick[i][cap] = chosen;
        }
    }

    let mut cap = budget;
    let mut result = vec![Decision { action: Action::Retire, tokens: 0 }; n];
    for i in (1..=n).rev() {
        let action = pick[i][cap];
        if action != Action::Retire {
            let cost = action.cost(candidates[i - 1].raw_tokens);
            result[i - 1] = Decision { action, tokens: cost };
            cap -= cost;
        }
    }
    result
}

fn evaluate(candidates: &[Candidate], decisions: &[Decision], budget: usize, policy: &'static str,
            oracle_net: f64, raw_demand: usize) -> Summary {
    let mut used_tokens = 0usize;
    let mut stored_candidates = 0usize;
    let mut future_queries = 0usize;
    let mut successful_query_mass = 0.0;
    let mut task_utility = 0.0;
    let mut load_bearing_total = 0usize;
    let mut load_bearing_stored = 0usize;
    let mut bad_retained = 0usize;
    let mut bad_promoted = 0usize;
    let mut raw_tokens_selected = 0usize;

    for (candidate, decision) in candidates.iter().zip(decisions) {
        if candidate.kind.load_bearing() { load_bearing_total += 1; }
        if decision.action != Action::Retire {
            stored_candidates += 1;
            used_tokens += decision.tokens;
            raw_tokens_selected += candidate.raw_tokens;
            if candidate.kind.load_bearing() { load_bearing_stored += 1; }
            if candidate.kind.unsafe_or_stale() {
                bad_retained += 1;
                if decision.action.is_promotion() { bad_promoted += 1; }
            }
        }
        future_queries += candidate.future_queries;
        let value = candidate.kind.query_utility(decision.action);
        task_utility += value * candidate.future_queries as f64;
        successful_query_mass += value.max(0.0).min(1.0) * candidate.future_queries as f64;
    }

    let net_utility = task_utility - used_tokens as f64 * COST_PER_TOKEN;
    Summary {
        policy,
        budget,
        raw_demand,
        used_tokens,
        stored_candidates,
        future_queries,
        synthetic_query_score: successful_query_mass / future_queries.max(1) as f64,
        task_utility,
        net_utility,
        oracle_net_utility: oracle_net,
        regret: (oracle_net - net_utility).max(0.0),
        load_bearing_coverage: load_bearing_stored as f64 / load_bearing_total.max(1) as f64,
        bad_retained,
        bad_promoted,
        compression_ratio: raw_tokens_selected as f64 / used_tokens.max(1) as f64,
        utility_per_token: net_utility / used_tokens.max(1) as f64,
    }
}

fn print_summary(sha: &str, seed: u64, candidate_count: usize, summary: &Summary) {
    println!(
        "{{\"record\":\"budget_summary\",\"protocol\":\"{PROTOCOL}\",\"source_sha\":\"{sha}\",\"seed\":{seed},\"candidates\":{candidate_count},\"policy\":\"{}\",\"budget_tokens\":{},\"raw_token_demand\":{},\"used_tokens\":{},\"stored_candidates\":{},\"future_queries\":{},\"synthetic_query_score\":{:.6},\"task_utility\":{:.4},\"net_utility\":{:.4},\"oracle_net_utility\":{:.4},\"regret\":{:.4},\"load_bearing_coverage\":{:.6},\"bad_retained\":{},\"bad_promoted\":{},\"compression_ratio\":{:.6},\"net_utility_per_token\":{:.6},\"storage_cost_per_token\":{:.4}}}",
        summary.policy, summary.budget, summary.raw_demand, summary.used_tokens,
        summary.stored_candidates, summary.future_queries, summary.synthetic_query_score,
        summary.task_utility, summary.net_utility, summary.oracle_net_utility,
        summary.regret, summary.load_bearing_coverage, summary.bad_retained,
        summary.bad_promoted, summary.compression_ratio, summary.utility_per_token,
        COST_PER_TOKEN
    );
}

fn run(config: Config) -> Result<(), String> {
    let sha = std::str::from_utf8(&config.source_sha).map_err(|_| "invalid source SHA")?;
    let candidates = build_fixture(config.seed, config.candidates);
    let raw_demand: usize = candidates.iter().map(|c| c.raw_tokens).sum();
    let policies: [&str; 8] = [
        "no_memory", "always_raw", "always_summary", "fixed_interval",
        "dedup_merge", "feature_threshold", "budget_adaptive", "promote_everything",
    ];

    println!(
        "{{\"record\":\"run\",\"protocol\":\"{PROTOCOL}\",\"source_sha\":\"{sha}\",\"seed\":{},\"candidate_count\":{},\"raw_token_demand\":{},\"feature_contract\":\"observed-only-v1\",\"utility_contract\":\"synthetic-v1\",\"storage_cost_per_token\":{:.4},\"budget_units\":\"abstract memory-token units; not real provider billing\"}}",
        config.seed, config.candidates, raw_demand, COST_PER_TOKEN
    );

    for &budget in DEFAULT_BUDGETS {
        let oracle = oracle_select(&candidates, budget);
        let oracle_net: f64 = candidates.iter().zip(&oracle)
            .map(|(candidate, decision)| candidate_net(*candidate, decision.action)).sum();
        let oracle_summary = evaluate(&candidates, &oracle, budget, "oracle_diagnostic", oracle_net, raw_demand);
        print_summary(sha, config.seed, config.candidates, &oracle_summary);

        for &policy in &policies {
            let decisions = choose_policy(&candidates, budget, policy);
            let summary = evaluate(&candidates, &decisions, budget, policy, oracle_net, raw_demand);
            if summary.used_tokens > budget {
                return Err(format!("policy {policy} exceeded budget {budget}"));
            }
            print_summary(sha, config.seed, config.candidates, &summary);
        }
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() == 1 && args[0] == "--help" {
        println!("Budget-matched consolidation frontier (synthetic only)\n\
Usage: cargo run --manifest-path research/broca-consolidation/Cargo.toml --release -- \\\n\
  --source-sha <git rev-parse HEAD> [--seed 17] [--candidates 120]\n\
Emits GMC-BUDGET-FRONTIER-V1 JSONL. The source SHA must match HEAD.");
        return;
    }

    let config = match parse_args(&args) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("error: {error}");
            process::exit(2);
        }
    };
    if let Err(error) = run(config) {
        eprintln!("FAIL: {error}");
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_is_seeded_and_varies_observed_features() {
        let first = build_fixture(17, 120);
        assert_eq!(first, build_fixture(17, 120));
        let second = build_fixture(18, 120);
        assert!(first.iter().zip(&second).any(|(a, b)| a.observed.repeats != b.observed.repeats));
    }

    #[test]
    fn policy_interface_receives_only_observed_features() {
        let observed = base_observed(Kind::PoisonedRule, &mut Rng::new(7));
        let _ = select_action("budget_adaptive", observed, 1, true);
    }

    #[test]
    fn no_memory_has_zero_storage_and_zero_utility() {
        let candidates = build_fixture(17, 120);
        let decisions = choose_policy(&candidates, 64, "no_memory");
        assert!(decisions.iter().all(|d| d.action == Action::Retire && d.tokens == 0));
        let evaluated = evaluate(&candidates, &decisions, 64, "no_memory", 0.0, 0);
        assert_eq!(evaluated.used_tokens, 0);
        assert!((evaluated.task_utility).abs() < 1e-9);
    }

    #[test]
    fn every_policy_respects_each_budget() {
        let candidates = build_fixture(42, 120);
        for budget in DEFAULT_BUDGETS {
            for policy in [
                "no_memory", "always_raw", "always_summary", "fixed_interval",
                "dedup_merge", "feature_threshold", "budget_adaptive", "promote_everything",
            ] {
                let decisions = choose_policy(&candidates, *budget, policy);
                assert!(decisions.iter().map(|d| d.tokens).sum::<usize>() <= *budget);
            }
        }
    }

    #[test]
    fn oracle_budget_is_never_exceeded_and_regret_is_nonnegative() {
        let candidates = build_fixture(91, 120);
        for budget in DEFAULT_BUDGETS {
            let oracle = oracle_select(&candidates, *budget);
            let oracle_net: f64 = candidates.iter().zip(&oracle)
                .map(|(candidate, decision)| candidate_net(*candidate, decision.action)).sum();
            assert!(oracle.iter().map(|d| d.tokens).sum::<usize>() <= *budget);
            let summary = evaluate(&candidates, &oracle, *budget, "oracle_diagnostic", oracle_net, 0);
            assert!(summary.regret.abs() < 1e-9);
        }
    }

    #[test]
    fn anti_control_exposes_unsafe_promotion() {
        let candidates = build_fixture(51, 120);
        let decisions = choose_policy(&candidates, 8192, "promote_everything");
        let summary = evaluate(&candidates, &decisions, 8192, "promote_everything", 0.0, 0);
        assert!(summary.bad_promoted > 0);
    }
}
