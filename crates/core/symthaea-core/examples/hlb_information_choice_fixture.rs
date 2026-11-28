// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic environment fixture for the HLB information-choice task.
//!
//! This is a task/environment qualification harness, not an HLB/FEP implementation.
//! Its reference policies establish that the environment separates worthwhile and
//! non-worthwhile observations before Symthaea adapters are connected.

use std::collections::BTreeSet;
use std::env;

use serde::{Deserialize, Serialize};
use serde_json::json;

const TASK_SPEC: &str =
    include_str!("../../../../docs/research/hlb-information-choice-task-v1.json");
const REWARD_BPS: i32 = 10_000;
const INVALID_ACTION_PENALTY_BPS: i32 = -10_000;

#[derive(Debug, Clone, Deserialize)]
struct TaskSpec {
    schema_version: String,
    status: String,
    claim_ceiling: String,
    environment: EnvironmentSpec,
    scenarios: Vec<Scenario>,
    reference_policies: Vec<PolicySpec>,
    metrics: Vec<String>,
    integrity_requirements: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct EnvironmentSpec {
    episodes_per_scenario: usize,
    correct_reward_bps: i32,
    incorrect_reward_bps: i32,
    one_probe_maximum: usize,
    random_generator: String,
    episode_seed_rule: String,
}

#[derive(Debug, Clone, Deserialize)]
struct Scenario {
    id: String,
    cue_accuracy_bps: u16,
    probe_accuracy_bps: u16,
    probe_cost_bps: u16,
    expected_probe_decision: String,
}

#[derive(Debug, Clone, Deserialize)]
struct PolicySpec {
    id: String,
    description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Side {
    Left,
    Right,
}

impl Side {
    fn opposite(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Observation {
    cue: Side,
    cue_accuracy_bps: u16,
    probe_accuracy_bps: u16,
    probe_cost_bps: u16,
    probe_available: bool,
    // None until a valid Inspect action causes the environment to reveal the probe.
    probe_signal: Option<Side>,
}

#[derive(Debug, Clone, Copy)]
enum Action {
    Inspect,
    Choose(Side),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Policy {
    CueGreedy,
    AlwaysInspect,
    BayesValueOfInformation,
    Random,
}

impl Policy {
    const ALL: [Self; 4] = [
        Self::CueGreedy,
        Self::AlwaysInspect,
        Self::BayesValueOfInformation,
        Self::Random,
    ];

    fn id(self) -> &'static str {
        match self {
            Self::CueGreedy => "cue_greedy",
            Self::AlwaysInspect => "always_inspect",
            Self::BayesValueOfInformation => "bayes_value_of_information",
            Self::Random => "random",
        }
    }

    fn choose(self, observation: &Observation, rng: &mut SplitMix64) -> Action {
        match self {
            Self::CueGreedy => Action::Choose(best_side(
                observation.cue,
                None,
                observation.cue_accuracy_bps,
                observation.probe_accuracy_bps,
            )),
            Self::AlwaysInspect => {
                if observation.probe_available && observation.probe_signal.is_none() {
                    Action::Inspect
                } else {
                    Action::Choose(best_side(
                        observation.cue,
                        observation.probe_signal,
                        observation.cue_accuracy_bps,
                        observation.probe_accuracy_bps,
                    ))
                }
            }
            Self::BayesValueOfInformation => {
                if observation.probe_available && observation.probe_signal.is_none() {
                    let expected_gain_bps = (expected_accuracy_after_probe(
                        observation.cue_accuracy_bps,
                        observation.probe_accuracy_bps,
                    ) - best_accuracy(observation.cue_accuracy_bps))
                        * f64::from(REWARD_BPS);
                    if expected_gain_bps > f64::from(observation.probe_cost_bps) {
                        Action::Inspect
                    } else {
                        Action::Choose(best_side(
                            observation.cue,
                            None,
                            observation.cue_accuracy_bps,
                            observation.probe_accuracy_bps,
                        ))
                    }
                } else {
                    Action::Choose(best_side(
                        observation.cue,
                        observation.probe_signal,
                        observation.cue_accuracy_bps,
                        observation.probe_accuracy_bps,
                    ))
                }
            }
            Self::Random => {
                if observation.probe_available
                    && observation.probe_signal.is_none()
                    && rng.next_u64() & 1 == 0
                {
                    Action::Inspect
                } else if rng.next_u64() & 1 == 0 {
                    Action::Choose(Side::Left)
                } else {
                    Action::Choose(Side::Right)
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrivateWorld {
    goal: Side,
    initial_cue: Side,
    probe_signal: Side,
}

impl PrivateWorld {
    fn new(scenario: &Scenario, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let goal = if rng.next_u64() & 1 == 0 {
            Side::Left
        } else {
            Side::Right
        };
        let initial_cue = sample_signal(goal, scenario.cue_accuracy_bps, &mut rng);
        let probe_signal = sample_signal(goal, scenario.probe_accuracy_bps, &mut rng);
        Self {
            goal,
            initial_cue,
            probe_signal,
        }
    }

    fn initial_observation(self, scenario: &Scenario) -> Observation {
        Observation {
            cue: self.initial_cue,
            cue_accuracy_bps: scenario.cue_accuracy_bps,
            probe_accuracy_bps: scenario.probe_accuracy_bps,
            probe_cost_bps: scenario.probe_cost_bps,
            probe_available: true,
            probe_signal: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EpisodeResult {
    score_bps: i32,
    correct: bool,
    probes: usize,
    invalid_actions: usize,
}

fn run_episode(scenario: &Scenario, seed: u64, policy: Policy) -> EpisodeResult {
    let world = PrivateWorld::new(scenario, seed);
    let mut observation = world.initial_observation(scenario);
    let policy_salt = match policy {
        Policy::CueGreedy => 0x11,
        Policy::AlwaysInspect => 0x22,
        Policy::BayesValueOfInformation => 0x33,
        Policy::Random => 0x44,
    };
    let mut policy_rng = SplitMix64::new(seed ^ policy_salt);
    let mut score_bps = 0;
    let mut probes = 0;
    let mut invalid_actions = 0;

    // At most one probe followed by one commit. Two decisions are enough.
    for _ in 0..2 {
        match policy.choose(&observation, &mut policy_rng) {
            Action::Inspect => {
                if !apply_inspection(
                    &mut observation,
                    world.probe_signal,
                    &mut score_bps,
                    &mut probes,
                ) {
                    invalid_actions += 1;
                    return EpisodeResult {
                        score_bps: INVALID_ACTION_PENALTY_BPS,
                        correct: false,
                        probes,
                        invalid_actions,
                    };
                }
            }
            Action::Choose(side) => {
                let correct = side == world.goal;
                score_bps += if correct {
                    REWARD_BPS
                } else {
                    0
                };
                return EpisodeResult {
                    score_bps,
                    correct,
                    probes,
                    invalid_actions,
                };
            }
        }
    }

    // A policy that never commits within the bounded decision budget fails closed.
    invalid_actions += 1;
    EpisodeResult {
        score_bps: INVALID_ACTION_PENALTY_BPS,
        correct: false,
        probes,
        invalid_actions,
    }
}

#[derive(Debug, Clone, Serialize)]
struct PolicySummary {
    scenario_id: String,
    policy_id: String,
    episodes: usize,
    accuracy: f64,
    mean_reward_bps: f64,
    probe_rate: f64,
    invalid_action_count: usize,
    mean_regret_to_perfect_information_oracle_bps: f64,
}

fn summarize(scenario: &Scenario, episodes: usize) -> Vec<PolicySummary> {
    Policy::ALL
        .iter()
        .copied()
        .map(|policy| {
            let mut correct = 0usize;
            let mut total_score = 0i64;
            let mut probes = 0usize;
            let mut invalid_actions = 0usize;

            for episode_index in 0..episodes {
                let seed = (scenario_index(&scenario.id) as u64)
                    .wrapping_mul(1_000_003)
                    .wrapping_add(episode_index as u64)
                    .wrapping_add(1);
                let result = run_episode(scenario, seed, policy);
                if result.correct { correct += 1; }
                total_score += i64::from(result.score_bps);
                probes += result.probes;
                invalid_actions += result.invalid_actions;
            }

            let n = episodes.max(1) as f64;
            let mean_reward_bps = total_score as f64 / n;
            vec![PolicySummary {
                scenario_id: scenario.id.clone(),
                policy_id: policy.id().to_owned(),
                episodes,
                accuracy: correct as f64 / n,
                mean_reward_bps,
                probe_rate: probes as f64 / n,
                invalid_action_count: invalid_actions,
                mean_regret_to_perfect_information_oracle_bps:
                    f64::from(REWARD_BPS) - mean_reward_bps,
            }]
        })
        .flatten()
        .collect()
}

fn scenario_index(id: &str) -> usize {
    match id {
        "clear_cue_probe_not_worth_cost" => 0,
        "ambiguous_cue_cheap_probe" => 1,
        "ambiguous_cue_expensive_probe" => 2,
        "weak_probe_not_worth_cost" => 3,
        _ => panic!("unregistered scenario id: {id}"),
    }
}

fn apply_inspection(
    observation: &mut Observation,
    private_probe_signal: Side,
    score_bps: &mut i32,
    probes: &mut usize,
) -> bool {
    if !observation.probe_available || observation.probe_signal.is_some() {
        return false;
    }
    *probes += 1;
    *score_bps -= i32::from(observation.probe_cost_bps);
    observation.probe_signal = Some(private_probe_signal);
    observation.probe_available = false;
    true
}

fn best_accuracy(cue_accuracy_bps: u16) -> f64 {
    let q = f64::from(cue_accuracy_bps) / f64::from(REWARD_BPS);
    q.max(1.0 - q)
}

fn likelihood(signal: Side, goal: Side, accuracy_bps: u16) -> f64 {
    let accuracy = f64::from(accuracy_bps) / f64::from(REWARD_BPS);
    if signal == goal {
        accuracy
    } else {
        1.0 - accuracy
    }
}

fn best_side(
    cue: Side,
    probe: Option<Side>,
    cue_accuracy_bps: u16,
    probe_accuracy_bps: u16,
) -> Side {
    let mut left_likelihood = likelihood(cue, Side::Left, cue_accuracy_bps);
    let mut right_likelihood = likelihood(cue, Side::Right, cue_accuracy_bps);
    if let Some(signal) = probe {
        left_likelihood *= likelihood(signal, Side::Left, probe_accuracy_bps);
        right_likelihood *= likelihood(signal, Side::Right, probe_accuracy_bps);
    }
    if left_likelihood > right_likelihood {
        Side::Left
    } else if right_likelihood > left_likelihood {
        Side::Right
    } else {
        // Deterministic tie-break is disclosed and uses only observed information.
        cue
    }
}

fn expected_accuracy_after_probe(cue_accuracy_bps: u16, probe_accuracy_bps: u16) -> f64 {
    let sides = [Side::Left, Side::Right];
    let mut probability_of_correct_decision = 0.0;

    for goal in sides {
        for cue in sides {
            for probe in sides {
                let joint_probability = 0.5
                    * likelihood(cue, goal, cue_accuracy_bps)
                    * likelihood(probe, goal, probe_accuracy_bps);
                let decision =
                    best_side(cue, Some(probe), cue_accuracy_bps, probe_accuracy_bps);
                if decision == goal {
                    probability_of_correct_decision += joint_probability;
                }
            }
        }
    }

    probability_of_correct_decision.clamp(0.0, 1.0)
}

fn sample_signal(goal: Side, accuracy_bps: u16, rng: &mut SplitMix64) -> Side {
    let sample = (rng.next_u64() % (REWARD_BPS as u64)) as u16;
    if sample < accuracy_bps {
        goal
    } else {
        goal.opposite()
    }
}

fn validate_spec(spec: &TaskSpec) -> Result<(), String> {
    if spec.schema_version != "hlb-information-choice-fixture-v1" {
        return Err("unexpected task-spec schema version".into());
    }
    if spec.status != "environment_fixture_only" {
        return Err("task spec must remain environment_fixture_only".into());
    }
    if !spec.claim_ceiling.contains("does not execute Symthaea HDC") {
        return Err("claim ceiling must state that the HLB is not executed".into());
    }
    if spec.environment.episodes_per_scenario == 0
        || spec.environment.correct_reward_bps != REWARD_BPS
        || spec.environment.incorrect_reward_bps != 0
        || spec.environment.one_probe_maximum != 1
        || !spec.environment.random_generator.contains("SplitMix64")
        || !spec.environment.episode_seed_rule.contains("episode_index")
    {
        return Err("environment contract is incomplete or inconsistent".into());
    }
    if spec.scenarios.len() < 4 {
        return Err("expected at least four scenarios to separate observation regimes".into());
    }

    let mut scenario_ids = BTreeSet::new();
    let mut policies = BTreeSet::new();
    for scenario in &spec.scenarios {
        if !scenario_ids.insert(scenario.id.clone()) {
            return Err(format!("duplicate scenario id: {}", scenario.id));
        }
        for (name, value) in [
            ("cue_accuracy_bps", scenario.cue_accuracy_bps),
            ("probe_accuracy_bps", scenario.probe_accuracy_bps),
            ("probe_cost_bps", scenario.probe_cost_bps),
        ] {
            if value > REWARD_BPS as u16 {
                return Err(format!("{} exceeds 10000 in {}", name, scenario.id));
            }
        }
        let expected = if (expected_accuracy_after_probe(
            scenario.cue_accuracy_bps,
            scenario.probe_accuracy_bps,
        ) - best_accuracy(scenario.cue_accuracy_bps))
            * f64::from(REWARD_BPS)
            > f64::from(scenario.probe_cost_bps)
        {
            "inspect"
        } else {
            "decline"
        };
        if scenario.expected_probe_decision != expected {
            return Err(format!(
                "scenario {} declares expected probe decision {}, Bayes calculation implies {}",
                scenario.id, scenario.expected_probe_decision, expected
            ));
        }
        scenario_index(&scenario.id);
    }

    for policy in &spec.reference_policies {
        if !policies.insert(policy.id.clone()) {
            return Err(format!("duplicate policy id: {}", policy.id));
        }
        if !Policy::ALL.iter().any(|known| known.id() == policy.id) {
            return Err(format!("unknown policy id: {}", policy.id));
        }
        if policy.description.trim().is_empty() {
            return Err(format!("policy {} has no description", policy.id));
        }
    }
    let expected_policy_ids = Policy::ALL
        .iter()
        .map(|policy| policy.id().to_owned())
        .collect::<BTreeSet<_>>();
    if policies != expected_policy_ids {
        return Err("reference policy inventory mismatch".into());
    }
    if spec.metrics.is_empty() || spec.integrity_requirements.len() < 5 {
        return Err("required metrics or integrity rules are missing".into());
    }
    Ok(())
}

fn valid_source_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn main() {
    let source_sha = env::var("SYMTHAEA_SOURCE_SHA")
        .expect("SYMTHAEA_SOURCE_SHA must bind the receipt to the exact Git commit");
    assert!(
        valid_source_sha(&source_sha),
        "SYMTHAEA_SOURCE_SHA must be a 40-character hexadecimal Git SHA"
    );

    let spec: TaskSpec = serde_json::from_str(TASK_SPEC).expect("task spec must be valid JSON");
    validate_spec(&spec).unwrap_or_else(|error| panic!("task spec contract failed: {error}"));

    let mut summaries = Vec::new();
    for scenario in &spec.scenarios {
        summaries.extend(summarize(
            scenario,
            spec.environment.episodes_per_scenario,
        ));
    }

    let result = json!({
        "receipt_version": "hlb-information-choice-fixture-receipt-v1",
        "source_sha": source_sha,
        "task_spec_blake3": blake3::hash(TASK_SPEC.as_bytes()).to_hex().to_string(),
        "status": "environment_fixture_executed",
        "integrated_hlb_adapter_executed": false,
        "task_spec_status": spec.status,
        "scenarios": spec.scenarios.len(),
        "episodes_per_scenario": spec.environment.episodes_per_scenario,
        "policy_rows": summaries,
        "sota_claim_supported": false,
        "interpretation": "Reference environment and policy sanity only; no Symthaea cognitive component was executed."
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&result).expect("result receipt must serialize")
    );
}

// A tiny stable PRNG keeps this reference fixture deterministic and dependency-free.
#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> TaskSpec {
        serde_json::from_str(TASK_SPEC).expect("valid task fixture")
    }

    fn scenario(id: &str) -> Scenario {
        spec().scenarios.into_iter().find(|s| s.id == id).expect("scenario")
    }

    #[test]
    fn checked_in_task_contract_is_valid() {
        validate_spec(&spec()).expect("valid environment fixture contract");
    }

    #[test]
    fn inspection_is_chosen_when_its_expected_value_exceeds_cost() {
        let s = scenario("ambiguous_cue_cheap_probe");
        assert!(expected_accuracy_after_probe(s.cue_accuracy_bps, s.probe_accuracy_bps)
            - best_accuracy(s.cue_accuracy_bps)
            > f64::from(s.probe_cost_bps) / f64::from(REWARD_BPS));
        let result = run_episode(&s, 42, Policy::BayesValueOfInformation);
        assert_eq!(result.probes, 1);
        assert_eq!(result.invalid_actions, 0);
    }

    #[test]
    fn inspection_is_declined_when_cost_exceeds_expected_gain() {
        let s = scenario("ambiguous_cue_expensive_probe");
        let result = run_episode(&s, 42, Policy::BayesValueOfInformation);
        assert_eq!(result.probes, 0);
        assert_eq!(result.invalid_actions, 0);
    }

    #[test]
    fn weak_probe_is_not_used() {
        let s = scenario("weak_probe_not_worth_cost");
        let result = run_episode(&s, 42, Policy::BayesValueOfInformation);
        assert_eq!(result.probes, 0);
    }

    #[test]
    fn clear_cue_does_not_justify_the_more_expensive_probe() {
        let s = scenario("clear_cue_probe_not_worth_cost");
        let result = run_episode(&s, 42, Policy::BayesValueOfInformation);
        assert_eq!(result.probes, 0);
    }

    #[test]
    fn hidden_goal_and_probe_are_not_in_initial_observation() {
        let s = scenario("ambiguous_cue_cheap_probe");
        let world = PrivateWorld::new(&s, 123);
        let observation = world.initial_observation(&s);
        assert_eq!(observation.probe_signal, None);
        assert!(observation.probe_available);
    }

    #[test]
    fn same_seed_reproduces_world_and_policy_result() {
        let s = scenario("ambiguous_cue_cheap_probe");
        let a = run_episode(&s, 999, Policy::BayesValueOfInformation);
        let b = run_episode(&s, 999, Policy::BayesValueOfInformation);
        assert_eq!(a, b);
    }

    #[test]
    fn policy_comparisons_reuse_identical_hidden_world_seed() {
        let s = scenario("ambiguous_cue_cheap_probe");
        let a = PrivateWorld::new(&s, 77);
        let b = PrivateWorld::new(&s, 77);
        assert_eq!(a, b);
        assert_eq!(a.initial_cue, b.initial_cue);
        assert_eq!(a.probe_signal, b.probe_signal);
    }

    #[test]
    fn repeated_inspection_is_rejected_without_leaking_another_signal() {
        let s = scenario("ambiguous_cue_cheap_probe");
        let world = PrivateWorld::new(&s, 1234);
        let mut observation = world.initial_observation(&s);
        let mut score = 0;
        let mut probes = 0;
        assert!(apply_inspection(
            &mut observation,
            world.probe_signal,
            &mut score,
            &mut probes
        ));
        let revealed = observation.probe_signal;
        assert!(!apply_inspection(
            &mut observation,
            world.probe_signal.opposite(),
            &mut score,
            &mut probes
        ));
        assert_eq!(observation.probe_signal, revealed);
        assert_eq!(probes, 1);
    }

    #[test]
    fn value_of_information_beats_greedy_on_the_affordable_ambiguous_case() {
        let s = scenario("ambiguous_cue_cheap_probe");
        let rows = summarize(&s, 1000);
        let greedy = rows.iter().find(|row| row.policy_id == "cue_greedy").unwrap();
        let voi = rows
            .iter()
            .find(|row| row.policy_id == "bayes_value_of_information")
            .unwrap();
        assert!(voi.mean_reward_bps > greedy.mean_reward_bps);
    }

    #[test]
    fn value_of_information_avoids_the_expensive_probe() {
        let s = scenario("ambiguous_cue_expensive_probe");
        let rows = summarize(&s, 1000);
        let greedy = rows.iter().find(|row| row.policy_id == "cue_greedy").unwrap();
        let always = rows.iter().find(|row| row.policy_id == "always_inspect").unwrap();
        let voi = rows
            .iter()
            .find(|row| row.policy_id == "bayes_value_of_information")
            .unwrap();
        assert_eq!(voi.probe_rate, 0.0);
        assert!(voi.mean_reward_bps > always.mean_reward_bps);
        assert!(voi.mean_reward_bps >= greedy.mean_reward_bps - 100.0);
    }

    #[test]
    fn source_sha_shape_is_strict() {
        assert!(valid_source_sha("a3ba137d4c3900e2ef6731a4ca5065dc2c8288c3"));
        assert!(!valid_source_sha("queued"));
        assert!(!valid_source_sha("not-a-git-sha"));
    }
}
