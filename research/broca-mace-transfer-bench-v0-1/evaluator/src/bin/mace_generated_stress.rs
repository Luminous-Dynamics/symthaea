// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::{BTreeMap, BTreeSet};

use broca_mace_policy::{
    decide, is_qualified, is_transfer_qualified, Action, Decision, Experience, Objective,
    Policy, ProbeRequest, Query, Regime, Scope, NORMALIZED_UTILITY_SCALE,
};

#[derive(Clone, Copy, Debug)]
struct Truth {
    utility_a: f64,
    utility_b: f64,
    causal_action: Option<Action>,
    invalid_a: bool,
}

impl Truth {
    fn utility(self, action: Action) -> f64 {
        match action {
            Action::A => self.utility_a,
            Action::B => self.utility_b,
        }
    }

    fn optimal_action(self) -> Action {
        if self.utility_a > self.utility_b {
            Action::A
        } else {
            Action::B
        }
    }

    fn optimal_utility(self) -> f64 {
        self.utility(self.optimal_action())
    }
}

#[derive(Clone, Debug)]
struct Case {
    cluster: u64,
    family: &'static str,
    memory: Vec<Experience>,
    query: Query,
    truth: Truth,
}

fn exp(
    id: u32,
    family: u16,
    scope: Scope,
    action: Action,
    utility: f64,
) -> Experience {
    Experience {
        event_id: id,
        source_family: family,
        scope,
        action,
        observed_utility: utility,
        intervention: true,
        prediction_pre_registered: true,
        outcome_reconciled: true,
        source_trusted: true,
        source_retracted: false,
        confounders_controlled: true,
        delayed_horizon_complete: true,
    }
}

fn pair(
    scope: Scope,
    id: u32,
    family: u16,
    a: f64,
    b: f64,
) -> Vec<Experience> {
    vec![
        exp(id, family, scope, Action::A, a),
        exp(id + 1, family + 1, scope, Action::A, a),
        exp(id + 2, family + 2, scope, Action::B, b),
        exp(id + 3, family + 3, scope, Action::B, b),
    ]
}

fn case(
    cluster: u64,
    family: &'static str,
    memory: Vec<Experience>,
    scope: Scope,
    utility_a: f64,
    utility_b: f64,
    causal_action: Option<Action>,
    invalid_a: bool,
) -> Case {
    Case {
        cluster,
        family,
        memory,
        query: Query { scope },
        truth: Truth {
            utility_a,
            utility_b,
            causal_action,
            invalid_a,
        },
    }
}

#[derive(Clone, Copy, Debug)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / 9_007_199_254_740_992.0
    }
}

fn generated_worlds(clusters: u64) -> Vec<Case> {
    let mut cases = Vec::with_capacity(clusters as usize * 14);
    let baseline = Scope {
        regime: Regime::Baseline,
        objective: Objective::Quality,
    };
    let shifted = Scope {
        regime: Regime::Shifted,
        objective: Objective::Quality,
    };

    for cluster in 1..=clusters {
        let mut rng = Rng::new(cluster ^ 0x9E37_79B9_7F4A_7C15);
        let id = (cluster * 1_000) as u32;
        let family = (1_000 + cluster * 16) as u16;

        let low = 0.12 + rng.unit() * 0.22;
        let high = low + 0.25 + rng.unit() * 0.30;
        let (a, b, causal) = if rng.unit() < 0.5 {
            (high, low, Action::A)
        } else {
            (low, high, Action::B)
        };
        cases.push(case(
            cluster,
            "identified_effect",
            pair(baseline, id, family, a, b),
            baseline,
            a,
            b,
            Some(causal),
            false,
        ));

        let mut confounded = pair(baseline, id + 10, family + 4, 0.92, 0.34);
        for item in &mut confounded {
            item.intervention = false;
            item.confounders_controlled = false;
        }
        cases.push(case(
            cluster,
            "confounded_decoy",
            confounded,
            baseline,
            0.10,
            0.90,
            None,
            true,
        ));

        cases.push(case(
            cluster,
            "regime_shift",
            pair(baseline, id + 20, family + 8, 0.93, 0.31),
            shifted,
            0.12,
            0.88,
            None,
            true,
        ));

        let speed = Scope {
            regime: Regime::Baseline,
            objective: Objective::Speed,
        };
        let cost = Scope {
            regime: Regime::Baseline,
            objective: Objective::Cost,
        };
        cases.push(case(
            cluster,
            "objective_shift",
            pair(speed, id + 20, family + 8, 0.93, 0.31),
            cost,
            0.12,
            0.88,
            None,
            true,
        ));

        let mut invariant_memory = pair(
            baseline,
            id + 200,
            family + 4,
            0.92,
            0.32,
        );
        invariant_memory.push(exp(id + 204, family + 8, shifted, Action::A, 0.91));
        invariant_memory.push(exp(id + 205, family + 9, shifted, Action::B, 0.33));
        cases.push(case(
            cluster,
            "invariant_transport_few_shot",
            invariant_memory,
            shifted,
            1.0,
            0.0,
            Some(Action::A),
            false,
        ));

        let mut broken_memory = pair(
            baseline,
            id + 210,
            family + 10,
            0.94,
            0.32,
        );
        broken_memory.push(exp(id + 214, family + 14, shifted, Action::A, 0.25));
        broken_memory.push(exp(id + 215, family + 15, shifted, Action::B, 0.86));
        cases.push(case(
            cluster,
            "mechanism_break_after_shift",
            broken_memory,
            shifted,
            0.0,
            1.0,
            None,
            true,
        ));

        cases.push(case(
            cluster,
            "active_probe_when_target_unobserved",
            pair(baseline, id + 220, family + 4, 0.90, 0.30),
            shifted,
            0.88,
            0.32,
            Some(Action::A),
            false,
        ));

        let mut retracted = pair(baseline, id + 30, family + 12, 0.95, 0.42);
        for item in retracted.iter_mut().filter(|x| x.action == Action::A) {
            item.source_retracted = true;
        }
        cases.push(case(
            cluster,
            "source_retraction",
            retracted,
            baseline,
            0.12,
            0.88,
            None,
            true,
        ));

        let mut unreconciled = pair(baseline, id + 40, family, 0.94, 0.40);
        for item in unreconciled.iter_mut().filter(|x| x.action == Action::A) {
            item.outcome_reconciled = false;
        }
        cases.push(case(
            cluster,
            "unreconciled_success",
            unreconciled,
            baseline,
            0.10,
            0.90,
            None,
            true,
        ));

        let mut delayed = pair(baseline, id + 45, family + 8, 0.96, 0.40);
        for item in delayed.iter_mut().filter(|x| x.action == Action::A) {
            item.delayed_horizon_complete = false;
        }
        cases.push(case(
            cluster,
            "delayed_horizon_incomplete",
            delayed,
            baseline,
            0.10,
            0.90,
            None,
            true,
        ));

        let mut poisoned = pair(baseline, id + 50, family + 4, 0.97, 0.30);
        for item in poisoned.iter_mut().filter(|x| x.action == Action::A) {
            item.source_trusted = false;
        }
        cases.push(case(
            cluster,
            "untrusted_poison",
            poisoned,
            baseline,
            0.08,
            0.92,
            None,
            true,
        ));

        let mut correlated = Vec::new();
        for offset in 0..6 {
            correlated.push(exp(
                id + 60 + offset,
                family + 14,
                baseline,
                Action::A,
                0.96,
            ));
        }
        correlated.push(exp(
            id + 70,
            family + 13,
            baseline,
            Action::B,
            0.63,
        ));
        correlated.push(exp(
            id + 71,
            family + 15,
            baseline,
            Action::B,
            0.66,
        ));
        cases.push(case(
            cluster,
            "correlated_echo",
            correlated,
            baseline,
            0.10,
            0.90,
            None,
            true,
        ));

        let alias_memory = vec![
            exp(id + 90, family + 1, baseline, Action::A, 0.99),
            exp(id + 90, family + 2, baseline, Action::A, 0.99),
            exp(id + 91, family + 3, baseline, Action::B, 0.65),
            exp(id + 92, family + 4, baseline, Action::B, 0.68),
        ];
        cases.push(case(
            cluster,
            "event_identity_alias",
            alias_memory,
            baseline,
            0.10,
            0.90,
            None,
            true,
        ));

        cases.push(case(
            cluster,
            "verified_negative",
            pair(baseline, id + 80, family + 8, 0.18, 0.86),
            baseline,
            0.12,
            0.90,
            Some(Action::B),
            false,
        ));
    }
    cases
}

const PROBE_COST: f64 = 0.12;

fn apply_environment_probe(
    case: &Case,
    request: ProbeRequest,
) -> Option<Vec<Experience>> {
    let cost_micros = (PROBE_COST * f64::from(NORMALIZED_UTILITY_SCALE)).round() as u32;
    if request.scope != case.query.scope
        || request.actions != [Action::A, Action::B]
        || request.replicates_per_action != 2
        || cost_micros > request.max_cost_normalized_micros
    {
        return None;
    }

    let max_event = case
        .memory
        .iter()
        .map(|item| item.event_id)
        .max()
        .unwrap_or(0);
    let max_family = case
        .memory
        .iter()
        .map(|item| item.source_family)
        .max()
        .unwrap_or(0);
    let first_event = max_event.checked_add(1)?;
    let first_family = max_family.checked_add(1)?;
    first_event.checked_add(3)?;
    first_family.checked_add(3)?;

    let mut memory = case.memory.clone();
    for (action_index, action) in request.actions.into_iter().enumerate() {
        for replicate in 0..u32::from(request.replicates_per_action) {
            let offset =
                (action_index as u32) * u32::from(request.replicates_per_action) + replicate;
            let noise = if replicate == 0 { -0.01 } else { 0.01 };
            let observed = (case.truth.utility(action) + noise).clamp(0.0, 1.0);
            memory.push(exp(
                first_event + offset,
                first_family + offset as u16,
                case.query.scope,
                action,
                observed,
            ));
        }
    }
    Some(memory)
}

fn simulate(
    policy: Policy,
    case: &Case,
) -> (Decision, Vec<Experience>, f64, bool, bool) {
    let initial = decide(policy, &case.memory, case.query);
    let Some(request) = initial.probe else {
        return (initial, case.memory.clone(), 0.0, false, false);
    };

    let Some(memory) = apply_environment_probe(case, request) else {
        return (Decision::abstain(), case.memory.clone(), 0.0, true, false);
    };
    let final_decision = decide(Policy::CausalQualified, &memory, case.query);
    let resolved = final_decision.action.is_some();
    (final_decision, memory, PROBE_COST, true, resolved)
}

#[derive(Clone, Copy, Debug, Default)]
struct Score {
    count: u32,
    correct: u32,
    abstentions: u32,
    overclaims: u32,
    invalid_refs: u32,
    unsupported_a: u32,
    utility_sum: f64,
    regret_sum: f64,
    probe_requests: u32,
    probe_resolved: u32,
    probe_cost_sum: f64,
}

fn evaluate(policy: Policy, cases: &[Case]) -> Score {
    let mut score = Score::default();
    for item in cases {
        let (decision, memory, probe_cost, requested, resolved) = simulate(policy, item);
        let action = match decision.action {
            Some(action) => action,
            None => {
                score.abstentions += 1;
                Action::B
            }
        };
        score.count += 1;
        if requested {
            score.probe_requests += 1;
        }
        if resolved {
            score.probe_resolved += 1;
        }
        score.probe_cost_sum += probe_cost;
        if action == item.truth.optimal_action() {
            score.correct += 1;
        }
        score.utility_sum += item.truth.utility(action) - probe_cost;
        score.regret_sum +=
            item.truth.optimal_utility() - (item.truth.utility(action) - probe_cost);

        let by_id: BTreeMap<u32, &Experience> = memory
            .iter()
            .map(|entry| (entry.event_id, entry))
            .collect();
        let evidence_is_eligible = |entry: &&Experience| match policy {
            Policy::InvariantTransport => is_transfer_qualified(entry, item.query),
            _ => is_qualified(entry, item.query),
        };
        if decision.evidence_ids.iter().any(|id| {
            by_id
                .get(id)
                .is_none_or(|entry| !evidence_is_eligible(entry))
        }) {
            score.invalid_refs += 1;
        }
        if item.truth.invalid_a && action == Action::A {
            score.unsupported_a += 1;
        }
        if let Some(claim) = decision.causal_claim {
            if item.truth.causal_action != Some(claim) {
                score.overclaims += 1;
            }
        }
    }
    score
}

fn cluster_means(policy: Policy, cases: &[Case]) -> BTreeMap<u64, f64> {
    let mut totals: BTreeMap<u64, (f64, usize)> = BTreeMap::new();
    for item in cases {
        let (decision, _, probe_cost, _, _) = simulate(policy, item);
        let action = decision.action.unwrap_or(Action::B);
        let entry = totals.entry(item.cluster).or_insert((0.0, 0));
        entry.0 += item.truth.utility(action) - probe_cost;
        entry.1 += 1;
    }
    totals
        .into_iter()
        .map(|(cluster, (sum, count))| (cluster, sum / count as f64))
        .collect()
}

fn paired_differences(candidate: Policy, cases: &[Case]) -> Vec<f64> {
    let candidate_means = cluster_means(candidate, cases);
    let baseline_means = cluster_means(Policy::NoMemory, cases);
    candidate_means
        .iter()
        .filter_map(|(cluster, value)| {
            baseline_means.get(cluster).map(|base| value - base)
        })
        .collect()
}

fn bootstrap_ci(values: &[f64], draws: usize, seed: u64) -> Option<(f64, f64)> {
    if values.is_empty() || draws == 0 {
        return None;
    }
    let mut rng = Rng::new(seed);
    let mut means = Vec::with_capacity(draws);
    for _ in 0..draws {
        let mut sum = 0.0;
        for _ in 0..values.len() {
            sum += values[(rng.next() as usize) % values.len()];
        }
        means.push(sum / values.len() as f64);
    }
    means.sort_by(f64::total_cmp);
    let low = ((means.len() - 1) * 25) / 1_000;
    let high = ((means.len() - 1) * 975) / 1_000;
    Some((means[low], means[high]))
}

#[derive(Clone, Copy, Debug)]
struct Retraction {
    event_id: u32,
}

fn apply_retractions(memory: &[Experience], events: &[Retraction]) -> Vec<Experience> {
    let mut result = memory.to_vec();
    for event in events {
        if let Some(item) = result.iter_mut().find(|x| x.event_id == event.event_id) {
            item.source_retracted = true;
        }
    }
    result
}

fn session_c_control() -> Vec<(Policy, Option<Action>, Option<Action>, bool)> {
    let scope = Scope {
        regime: Regime::Baseline,
        objective: Objective::Quality,
    };
    let memory = pair(scope, 900_001, 40, 0.94, 0.42);
    let events = [
        Retraction { event_id: 900_001 },
        Retraction { event_id: 900_002 },
    ];
    let corrected = apply_retractions(&memory, &events);
    Policy::ALL
        .into_iter()
        .map(|policy| {
            let before = decide(policy, &memory, Query { scope });
            let after = decide(policy, &corrected, Query { scope });
            let stale = after.evidence_ids.iter().any(|id| {
                events.iter().any(|event| event.event_id == *id)
            });
            (policy, before.action, after.action, stale)
        })
        .collect()
}

fn report(cases: &[Case]) {
    let clusters = cluster_means(Policy::NoMemory, cases).len();
    let families: BTreeSet<&str> = cases.iter().map(|item| item.family).collect();
    println!("MACE generated-world stress benchmark v0.3");
    println!(
        "clusters={}, scenarios={}, families={}",
        clusters,
        cases.len(),
        families.len()
    );
    println!(
        concat!(
            "policy,accuracy_pct,mean_utility,mean_regret,abstentions,",
            "overclaims,invalid_refs,unsupported_A,probe_requests,",
            "probe_resolved,probe_cost_normalized_utility"
        )
    );
    for policy in Policy::ALL {
        let score = evaluate(policy, cases);
        let n = f64::from(score.count.max(1));
        println!(
            "{},{:.2},{:.4},{:.4},{},{},{},{},{},{},{}",
            policy.name(),
            100.0 * f64::from(score.correct) / n,
            score.utility_sum / n,
            score.regret_sum / n,
            score.abstentions,
            score.overclaims,
            score.invalid_refs,
            score.unsupported_a,
            score.probe_requests,
            score.probe_resolved,
            score.probe_cost_sum
        );
    }
    println!("policy_by_scenario_family");
    println!("policy,family,count,accuracy_pct,mean_utility,mean_regret,unsupported_A");
    let mut groups: BTreeMap<&str, Vec<&Case>> = BTreeMap::new();
    for item in cases {
        groups.entry(item.family).or_default().push(item);
    }
    for policy in [
        Policy::CausalQualified,
        Policy::InvariantTransport,
        Policy::ActiveProbe,
    ] {
        for (family, items) in &groups {
            let family_cases: Vec<Case> =
                items.iter().map(|item| (**item).clone()).collect();
            let score = evaluate(policy, &family_cases);
            let n = f64::from(score.count.max(1));
            println!(
                "{},{},{},{:.2},{:.4},{:.4},{}",
                policy.name(),
                family,
                score.count,
                100.0 * f64::from(score.correct) / n,
                score.utility_sum / n,
                score.regret_sum / n,
                score.unsupported_a
            );
        }
    }

    println!(
        concat!(
            "paired_utility_lift_vs_no_memory,mean,",
            "cluster_bootstrap_95ci_low,cluster_bootstrap_95ci_high"
        )
    );
    for (index, policy) in Policy::ALL.into_iter().enumerate() {
        let diffs = paired_differences(policy, cases);
        let mean = diffs.iter().sum::<f64>() / diffs.len().max(1) as f64;
        let (low, high) = bootstrap_ci(
            &diffs,
            2_000,
            0x4D41_4345 ^ index as u64,
        )
        .unwrap_or((0.0, 0.0));
        println!("{},{:.4},{:.4},{:.4}", policy.name(), mean, low, high);
    }

    println!("session_C_retraction,policy,session_B_action,session_C_action,stale_reuse");
    for (policy, before, after, stale) in session_c_control() {
        println!(
            "known_retraction,{},{},{},{}",
            policy.name(),
            before.map(Action::name).unwrap_or("ABSTAIN"),
            after.map(Action::name).unwrap_or("ABSTAIN"),
            stale
        );
    }
}

fn main() {
    report(&generated_worlds(256));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invariant_transport_wins_few_shot_but_abstains_on_mechanism_break() {
        let cases = generated_worlds(1);
        let invariant = cases
            .iter()
            .find(|item| item.family == "invariant_transport_few_shot")
            .unwrap();
        let broken = cases
            .iter()
            .find(|item| item.family == "mechanism_break_after_shift")
            .unwrap();

        assert_eq!(
            decide(Policy::CausalQualified, &invariant.memory, invariant.query).action,
            None
        );
        assert_eq!(
            decide(
                Policy::InvariantTransport,
                &invariant.memory,
                invariant.query,
            )
            .action,
            Some(Action::A)
        );
        assert_eq!(
            decide(Policy::InvariantTransport, &broken.memory, broken.query).action,
            None
        );
    }

    #[test]
    fn environment_rejects_mismatched_or_underfunded_probe_requests() {
        let cases = generated_worlds(1);
        let item = cases
            .iter()
            .find(|case| case.family == "active_probe_when_target_unobserved")
            .unwrap();
        let request = decide(Policy::ActiveProbe, &item.memory, item.query)
            .probe
            .expect("active policy requests a probe");

        let mut insufficient = request;
        insufficient.max_cost_normalized_micros = 1;
        assert!(apply_environment_probe(item, insufficient).is_none());

        let mut wrong_scope = request;
        wrong_scope.scope.objective = Objective::Cost;
        assert!(apply_environment_probe(item, wrong_scope).is_none());

        let mut unsupported_actions = request;
        unsupported_actions.actions = [Action::B, Action::B];
        assert!(apply_environment_probe(item, unsupported_actions).is_none());

        let mut unsupported_replicates = request;
        unsupported_replicates.replicates_per_action = 3;
        assert!(apply_environment_probe(item, unsupported_replicates).is_none());
    }

    #[test]
    fn active_probe_resolves_a_target_regime_without_prior_witnesses() {
        let cases = generated_worlds(1);
        let item = cases
            .iter()
            .find(|case| case.family == "active_probe_when_target_unobserved")
            .unwrap();
        let (decision, memory, cost, requested, resolved) =
            simulate(Policy::ActiveProbe, item);
        let no_memory = simulate(Policy::NoMemory, item).0;

        assert!(requested);
        assert!(resolved);
        assert_eq!(decision.action, Some(Action::A));
        assert_eq!(no_memory.action, Some(Action::B));
        assert_eq!(memory.len(), item.memory.len() + 4);
        assert!((cost - PROBE_COST).abs() < f64::EPSILON);
    }

    #[test]
    fn generated_worlds_are_deterministic_and_rotate_source_families() {
        let first = generated_worlds(8);
        let second = generated_worlds(8);
        assert_eq!(format!("{first:?}"), format!("{second:?}"));
        assert_eq!(first.len(), 112);
        assert_eq!(first[0].cluster, 1);
        assert_eq!(first[8].cluster, 2);
        assert_ne!(
            first[0].memory[0].source_family,
            first[8].memory[0].source_family
        );
    }

    #[test]
    fn generated_duplicate_event_identity_is_rejected_even_with_family_aliases() {
        let cases = generated_worlds(1);
        let item = cases
            .iter()
            .find(|case| case.family == "event_identity_alias")
            .unwrap();
        let frequency = decide(Policy::FrequencyOnly, &item.memory, item.query);
        let qualified = decide(Policy::CausalQualified, &item.memory, item.query);
        assert_eq!(frequency.action, Some(Action::A));
        assert_eq!(qualified.action, Some(Action::B));
        assert_eq!(qualified.evidence_ids.len(), 2);
    }

    #[test]
    fn qualified_policy_has_no_generated_fixture_overclaims_or_invalid_reuse() {
        let cases = generated_worlds(16);
        let score = evaluate(Policy::CausalQualified, &cases);
        assert_eq!(score.count, 224);
        assert_eq!(score.overclaims, 0);
        assert_eq!(score.invalid_refs, 0);
        assert_eq!(score.unsupported_a, 0);
    }

    #[test]
    fn paired_cluster_bootstrap_is_deterministic_and_ordered() {
        let cases = generated_worlds(12);
        let diffs = paired_differences(Policy::CausalQualified, &cases);
        assert_eq!(diffs.len(), 12);
        let first = bootstrap_ci(&diffs, 500, 77).unwrap();
        let second = bootstrap_ci(&diffs, 500, 77).unwrap();
        assert_eq!(first, second);
        assert!(first.0 <= first.1);
    }

    #[test]
    fn session_c_retraction_prevents_qualified_stale_reuse() {
        let rows = session_c_control();
        let qualified = rows
            .iter()
            .find(|row| row.0 == Policy::CausalQualified)
            .unwrap();
        let frequency = rows
            .iter()
            .find(|row| row.0 == Policy::FrequencyOnly)
            .unwrap();
        assert_eq!(qualified.1, Some(Action::A));
        assert_eq!(qualified.2, Some(Action::B));
        assert!(!qualified.3);
        assert_eq!(frequency.2, Some(Action::A));
        assert!(frequency.3);
    }
}
