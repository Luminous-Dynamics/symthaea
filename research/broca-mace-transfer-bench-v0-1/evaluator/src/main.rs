#[cfg(test)]
mod raw_ingest;

#[cfg(test)]
mod hdc_retrieval_integration;


// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeMap;

use broca_mace_policy::{
    decide, is_qualified, is_transfer_qualified, Action, Decision, Experience, Objective,
    Policy, ProbeRequest, Query, Regime, Scope, NORMALIZED_UTILITY_SCALE,
    IdentificationMethodV1,
};

#[derive(Clone, Copy, Debug)]
struct OracleTruth {
    utility_a: f64,
    utility_b: f64,
    causal_action: Option<Action>,
    action_a_supported_only_by_invalid_evidence: bool,
}

impl OracleTruth {
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
struct Scenario {
    name: &'static str,
    session_a: Vec<Experience>,
    session_b: Query,
    oracle: OracleTruth,
}

fn fixture_root_digest(seed: u32) -> broca_mace_policy::SourceRootDigest {
    let mut bytes = [0_u8; 32];
    bytes[..4].copy_from_slice(&seed.to_be_bytes());
    broca_mace_policy::SourceRootDigest::from_bytes(bytes)
}

fn fixture_semantic_estimand() -> broca_mace_policy::SemanticEstimandV1 {
    broca_mace_policy::SemanticEstimandV1::new(
        1,
        broca_mace_policy::EvidenceDigest256::from_bytes([1; 32]),
        broca_mace_policy::EvidenceDigest256::from_bytes([2; 32]),
        broca_mace_policy::EvidenceDigest256::from_bytes([3; 32]),
        broca_mace_policy::EvidenceDigest256::from_bytes([4; 32]),
        broca_mace_policy::EvidenceDigest256::from_bytes([5; 32]),
        30_000,
        broca_mace_policy::EvidenceDigest256::from_bytes([6; 32]),
        broca_mace_policy::EvidenceDigest256::from_bytes([7; 32]),
        broca_mace_policy::EvidenceDigest256::from_bytes([8; 32]),
    )
}

fn fixture_causal_identification(
    seed: u32,
) -> broca_mace_policy::CausalIdentificationV1 {
    let mut design = [0_u8; 32];
    design[..4].copy_from_slice(&seed.to_be_bytes());
    broca_mace_policy::CausalIdentificationV1::new(
        broca_mace_policy::IdentificationMethodV1::RandomizedControlledTrial,
        broca_mace_policy::EvidenceDigest256::from_bytes(design),
        fixture_semantic_estimand().canonical_digest(),
        5_000,
        5_000,
        50_000,
        true,
    )
    .with_semantic_estimand(fixture_semantic_estimand())
}

fn experience(
    event_id: u32,
    source_family: u16,
    scope: Scope,
    action: Action,
    observed_utility: f64,
) -> Experience {
    Experience {
        event_id,
        source_family,
        source_root_id: u32::from(source_family),
        source_root_digest: fixture_root_digest(u32::from(source_family)),
        scope,
        action,
        observed_utility,
        measurement_error_bound: 0.02,
        causal_identification: fixture_causal_identification(event_id),
        intervention: true,
        prediction_pre_registered: true,
        outcome_reconciled: true,
        source_trusted: true,
        source_retracted: false,
        confounders_controlled: true,
        delayed_horizon_complete: true,
    }
}

fn valid_pair(scope: Scope, start_id: u32, a: f64, b: f64) -> Vec<Experience> {
    vec![
        experience(start_id, 10, scope, Action::A, a),
        experience(start_id + 1, 11, scope, Action::A, a),
        experience(start_id + 2, 20, scope, Action::B, b),
        experience(start_id + 3, 21, scope, Action::B, b),
    ]
}

fn scenario(
    name: &'static str,
    session_a: Vec<Experience>,
    session_b: Query,
    utility_a: f64,
    utility_b: f64,
    causal_action: Option<Action>,
    invalid_a: bool,
) -> Scenario {
    Scenario {
        name,
        session_a,
        session_b,
        oracle: OracleTruth {
            utility_a,
            utility_b,
            causal_action,
            action_a_supported_only_by_invalid_evidence: invalid_a,
        },
    }
}

fn cases() -> Vec<Scenario> {
    let baseline_quality = Scope {
        regime: Regime::Baseline,
        objective: Objective::Quality,
    };
    let shifted_quality = Scope {
        regime: Regime::Shifted,
        objective: Objective::Quality,
    };
    let baseline_cost = Scope {
        regime: Regime::Baseline,
        objective: Objective::Cost,
    };
    let baseline_speed = Scope {
        regime: Regime::Baseline,
        objective: Objective::Speed,
    };

    let identified = valid_pair(baseline_quality, 1, 0.92, 0.38);

    let mut observational_decoy = valid_pair(baseline_quality, 190, 0.96, 0.65);
    for item in observational_decoy
        .iter_mut()
        .filter(|item| item.action == Action::A)
    {
        item.causal_identification.method = IdentificationMethodV1::Observational;
    }

    let mut quasi_decoy = valid_pair(baseline_quality, 194, 0.94, 0.62);
    for item in quasi_decoy
        .iter_mut()
        .filter(|item| item.action == Action::A)
    {
        item.causal_identification.method = IdentificationMethodV1::QuasiExperimental;
    }

    let mut confounded = valid_pair(baseline_quality, 10, 0.88, 0.42);
    for item in &mut confounded {
        item.intervention = false;
        item.confounders_controlled = false;
    }

    let regime_shift = valid_pair(baseline_quality, 20, 0.91, 0.35);
    let objective_shift = valid_pair(baseline_speed, 30, 0.90, 0.40);

    let mut retracted = valid_pair(baseline_quality, 40, 0.93, 0.35);
    for item in retracted.iter_mut().filter(|item| item.action == Action::A) {
        item.source_retracted = true;
    }

    let mut unreconciled = valid_pair(baseline_quality, 50, 0.94, 0.32);
    for item in unreconciled.iter_mut().filter(|item| item.action == Action::A) {
        item.outcome_reconciled = false;
    }

    let mut poisoned = valid_pair(baseline_quality, 60, 0.97, 0.30);
    for item in poisoned.iter_mut().filter(|item| item.action == Action::A) {
        item.source_trusted = false;
    }

    let mut delayed = valid_pair(baseline_quality, 70, 0.95, 0.45);
    for item in delayed.iter_mut().filter(|item| item.action == Action::A) {
        item.delayed_horizon_complete = false;
    }

    let mut correlated = Vec::new();
    for offset in 0..6 {
        correlated.push(experience(
            80 + offset,
            77,
            baseline_quality,
            Action::A,
            0.96,
        ));
    }
    correlated.push(experience(90, 20, baseline_quality, Action::B, 0.55));
    correlated.push(experience(91, 21, baseline_quality, Action::B, 0.58));

    let verified_negative = valid_pair(baseline_quality, 100, 0.18, 0.86);
    let irrelevant_update = valid_pair(baseline_cost, 110, 0.99, 0.20);
    let ambiguous = vec![
        experience(120, 1, baseline_quality, Action::A, 0.80),
        experience(121, 2, baseline_quality, Action::B, 0.70),
    ];
    let one_sided = vec![
        experience(126, 30, baseline_quality, Action::B, 0.40),
        experience(127, 31, baseline_quality, Action::B, 0.42),
    ];
    let shifted_identified = valid_pair(shifted_quality, 130, 0.89, 0.30);

    let mut common_cause = valid_pair(baseline_quality, 160, 0.96, 0.65);
    for item in common_cause.iter_mut().filter(|item| item.action == Action::A) {
        item.source_root_id = 700;
        item.source_root_digest = fixture_root_digest(700);
    }

    let common_mode_shift = vec![
        experience(180, 330, baseline_quality, Action::A, 0.80),
        experience(181, 331, baseline_quality, Action::A, 0.80),
        experience(182, 340, baseline_quality, Action::B, 0.40),
        experience(183, 341, baseline_quality, Action::B, 0.40),
        experience(184, 350, shifted_quality, Action::A, 0.95),
        experience(185, 351, shifted_quality, Action::B, 0.55),
    ];

    let mut estimand_shift = valid_pair(baseline_quality, 200, 0.92, 0.32);
    estimand_shift.push(experience(204, 304, shifted_quality, Action::A, 0.91));
    estimand_shift.push(experience(205, 305, shifted_quality, Action::B, 0.33));
    for item in estimand_shift
        .iter_mut()
        .filter(|item| item.scope.regime == Regime::Shifted)
    {
        item.causal_identification.estimand_digest =
            broca_mace_policy::EvidenceDigest256::from_bytes([0xCC; 32]);
    }

    let mut wide_intervals = valid_pair(baseline_quality, 170, 0.92, 0.32);
    wide_intervals.push(experience(174, 304, shifted_quality, Action::A, 0.91));
    wide_intervals.push(experience(175, 305, shifted_quality, Action::B, 0.33));
    for item in &mut wide_intervals {
        item.measurement_error_bound = 0.07;
    }

    vec![
        scenario(
            "identified_effect_transfer",
            identified,
            Query { scope: baseline_quality },
            1.0,
            0.0,
            Some(Action::A),
            false,
        ),
        scenario(
            "observational_high_reward_causal_decoy",
            observational_decoy,
            Query { scope: baseline_quality },
            0.10,
            0.90,
            None,
            true,
        ),
        scenario(
            "unsupported_quasi_experimental_decoy",
            quasi_decoy,
            Query { scope: baseline_quality },
            0.10,
            0.90,
            None,
            true,
        ),
        scenario(
            "confounded_association_decoy",
            confounded,
            Query { scope: baseline_quality },
            0.0,
            1.0,
            None,
            true,
        ),
        scenario(
            "regime_shift",
            regime_shift,
            Query { scope: shifted_quality },
            0.0,
            1.0,
            None,
            true,
        ),
        scenario(
            "objective_shift",
            objective_shift,
            Query { scope: baseline_cost },
            0.0,
            1.0,
            None,
            true,
        ),
        scenario(
            "source_retraction",
            retracted,
            Query { scope: baseline_quality },
            0.0,
            1.0,
            None,
            true,
        ),
        scenario(
            "unreconciled_tool_success",
            unreconciled,
            Query { scope: baseline_quality },
            0.0,
            1.0,
            None,
            true,
        ),
        scenario(
            "untrusted_poison",
            poisoned,
            Query { scope: baseline_quality },
            0.0,
            1.0,
            None,
            true,
        ),
        scenario(
            "delayed_consequence_unobserved",
            delayed,
            Query { scope: baseline_quality },
            0.0,
            1.0,
            None,
            true,
        ),
        scenario(
            "correlated_duplicate_echo",
            correlated,
            Query { scope: baseline_quality },
            0.0,
            1.0,
            None,
            true,
        ),
        scenario(
            "verified_negative_experience",
            verified_negative,
            Query { scope: baseline_quality },
            0.0,
            1.0,
            Some(Action::B),
            false,
        ),
        scenario(
            "irrelevant_update_control",
            irrelevant_update,
            Query { scope: baseline_quality },
            0.0,
            1.0,
            None,
            false,
        ),
        scenario(
            "under_supported_candidate",
            ambiguous,
            Query { scope: baseline_quality },
            0.0,
            1.0,
            None,
            false,
        ),
        scenario(
            "one_sided_qualified_evidence",
            one_sided,
            Query { scope: baseline_quality },
            0.90,
            0.40,
            Some(Action::A),
            false,
        ),
        scenario(
            "invariant_transport_few_shot",
            {
                let mut memory = valid_pair(baseline_quality, 140, 300, 0.92, 0.32);
                memory.push(experience(144, 304, shifted_quality, Action::A, 0.91));
                memory.push(experience(145, 305, shifted_quality, Action::B, 0.33));
                memory
            },
            Query { scope: shifted_quality },
            1.0,
            0.0,
            Some(Action::A),
            false,
        ),
        scenario(
            "mechanism_break_after_regime_shift",
            {
                let mut memory = valid_pair(baseline_quality, 150, 310, 0.94, 0.32);
                memory.push(experience(154, 314, shifted_quality, Action::A, 0.25));
                memory.push(experience(155, 315, shifted_quality, Action::B, 0.86));
                memory
            },
            Query { scope: shifted_quality },
            0.0,
            1.0,
            None,
            true,
        ),
        scenario(
            "common_cause_root_attack",
            common_cause,
            Query { scope: baseline_quality },
            0.10,
            0.90,
            None,
            true,
        ),
        scenario(
            "common_mode_baseline_shift_same_effect",
            common_mode_shift,
            Query { scope: shifted_quality },
            1.0,
            0.0,
            Some(Action::A),
            false,
        ),
        scenario(
            "estimand_mismatch_across_regimes",
            estimand_shift,
            Query { scope: shifted_quality },
            1.0,
            0.0,
            Some(Action::A),
            false,
        ),
        scenario(
            "invariance_interval_too_wide",
            wide_intervals,
            Query { scope: shifted_quality },
            1.0,
            0.0,
            Some(Action::A),
            false,
        ),
        scenario(
            "shifted_regime_positive_control",
            shifted_identified,
            Query { scope: shifted_quality },
            1.0,
            0.0,
            Some(Action::A),
            false,
        ),
    ]
}

const PROBE_COST: f64 = 0.12;

fn apply_environment_probe(
    case: &Scenario,
    request: ProbeRequest,
) -> Option<Vec<Experience>> {
    let cost_micros = (PROBE_COST * f64::from(NORMALIZED_UTILITY_SCALE)).round() as u32;
    if request.scope != case.session_b.scope
        || request.actions != [Action::A, Action::B]
        || request.replicates_per_action != 2
        || cost_micros > request.max_cost_normalized_micros
    {
        return None;
    }

    let max_event = case
        .session_a
        .iter()
        .map(|item| item.event_id)
        .max()
        .unwrap_or(0);
    let max_family = case
        .session_a
        .iter()
        .map(|item| item.source_family)
        .max()
        .unwrap_or(0);
    let first_event = max_event.checked_add(1)?;
    let first_family = max_family.checked_add(1)?;
    first_event.checked_add(3)?;
    first_family.checked_add(3)?;

    let mut memory = case.session_a.clone();
    for (action_index, action) in request.actions.into_iter().enumerate() {
        for replicate in 0..u32::from(request.replicates_per_action) {
            let offset =
                (action_index as u32) * u32::from(request.replicates_per_action) + replicate;
            let noise = if replicate == 0 { -0.01 } else { 0.01 };
            let observed = (case.oracle.utility(action) + noise).clamp(0.0, 1.0);
            memory.push(experience(
                first_event + offset,
                first_family + offset as u16,
                case.session_b.scope,
                action,
                observed,
            ));
        }
    }
    Some(memory)
}

// Simulate a candidate-requested experiment as environment interaction. The
// candidate sees only the returned observations on its second decision; oracle
// utilities remain evaluator-only.
fn simulate(
    policy: Policy,
    case: &Scenario,
) -> (Decision, Vec<Experience>, f64, bool, bool) {
    let initial = decide(policy, &case.session_a, case.session_b);
    let Some(request) = initial.probe else {
        return (initial, case.session_a.clone(), 0.0, false, false);
    };

    let Some(memory) = apply_environment_probe(case, request) else {
        return (Decision::abstain(), case.session_a.clone(), 0.0, true, false);
    };
    let final_decision = decide(Policy::CausalQualified, &memory, case.session_b);
    let resolved = final_decision.action.is_some();
    (final_decision, memory, PROBE_COST, true, resolved)
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Metrics {
    decisions: u32,
    correct_actions: u32,
    abstentions: u32,
    causal_claims: u32,
    causal_overclaims: u32,
    invalid_evidence_reuse: u32,
    unsupported_a_reuse: u32,
    evidence_bound_decisions: u32,
    total_evidence_refs: u32,
    probe_requests: u32,
    probe_resolved: u32,
    probe_cost_normalized_micros: u32,
}

fn evaluate(policy: Policy, scenarios: &[Scenario]) -> Metrics {
    let mut metrics = Metrics::default();

    for case in scenarios {
        let (decision, memory, probe_cost, probe_requested, probe_resolved) =
            simulate(policy, case);
        metrics.decisions += 1;
        if probe_requested {
            metrics.probe_requests += 1;
        }
        if probe_resolved {
            metrics.probe_resolved += 1;
        }
        metrics.probe_cost_normalized_micros += (probe_cost * 1_000_000.0).round() as u32;

        let action = match decision.action {
            Some(action) => action,
            None => {
                metrics.abstentions += 1;
                Action::B
            }
        };

        if action == case.oracle.optimal_action() {
            metrics.correct_actions += 1;
        }

        if !decision.evidence_ids.is_empty() {
            metrics.evidence_bound_decisions += 1;
        }
        metrics.total_evidence_refs += decision.evidence_ids.len() as u32;

        let by_id: BTreeMap<u32, &Experience> = memory
            .iter()
            .map(|item| (item.event_id, item))
            .collect();

        let evidence_is_eligible = |item: &&Experience| match policy {
            Policy::InvariantTransport => is_transfer_qualified(item, case.session_b),
            _ => is_qualified(item, case.session_b),
        };
        if decision.evidence_ids.iter().any(|id| {
            by_id
                .get(id)
                .is_none_or(|item| !evidence_is_eligible(item))
        }) {
            metrics.invalid_evidence_reuse += 1;
        }

        if case.oracle.action_a_supported_only_by_invalid_evidence && action == Action::A {
            metrics.unsupported_a_reuse += 1;
        }

        if let Some(claimed_action) = decision.causal_claim {
            metrics.causal_claims += 1;
            if case.oracle.causal_action != Some(claimed_action) {
                metrics.causal_overclaims += 1;
            }
        }
    }

    metrics
}

fn report(scenarios: &[Scenario]) {
    println!("Broca MACE-lite synthetic transfer benchmark v0.1");
    println!(
        "cases={}, policies={}, oracle=separate-evaluator-type",
        scenarios.len(),
        Policy::ALL.len()
    );
    println!();
    println!(
        concat!(
            "policy,accuracy_pct,mean_utility,mean_regret,abstentions,",
            "causal_claims,causal_overclaims,invalid_evidence_reuse,",
            "unsupported_A_reuse,evidence_bound_decisions,total_evidence_refs,",
            "probe_requests,probe_resolved,probe_cost_normalized_micros"
        )
    );

    for policy in Policy::ALL {
        let metrics = evaluate(policy, scenarios);
        let count = f64::from(metrics.decisions.max(1));
        let total_utility: f64 = scenarios
            .iter()
            .map(|case| {
                let (decision, _, probe_cost, _, _) = simulate(policy, case);
                case.oracle.utility(decision.action.unwrap_or(Action::B)) - probe_cost
            })
            .sum();
        let total_regret: f64 = scenarios
            .iter()
            .map(|case| {
                let (decision, _, probe_cost, _, _) = simulate(policy, case);
                let action = decision.action.unwrap_or(Action::B);
                case.oracle.optimal_utility()
                    - (case.oracle.utility(action) - probe_cost)
            })
            .sum();

        println!(
            "{},{:.1},{:.3},{:.3},{},{},{},{},{},{},{},{},{},{}",
            policy.name(),
            100.0 * f64::from(metrics.correct_actions) / count,
            total_utility / count,
            total_regret / count,
            metrics.abstentions,
            metrics.causal_claims,
            metrics.causal_overclaims,
            metrics.invalid_evidence_reuse,
            metrics.unsupported_a_reuse,
            metrics.evidence_bound_decisions,
            metrics.total_evidence_refs,
            metrics.probe_requests,
            metrics.probe_resolved,
            metrics.probe_cost_normalized_micros
        );
    }

    println!();
    println!("scenario,qualified_policy_action,oracle_action,causal_claim");
    for case in scenarios {
        let decision = decide(Policy::CausalQualified, &case.session_a, case.session_b);
        println!(
            "{},{},{},{}",
            case.name,
            decision.action.map(Action::name).unwrap_or("ABSTAIN"),
            case.oracle.optimal_action().name(),
            decision.causal_claim.map(Action::name).unwrap_or("NONE")
        );
    }

    println!();
    println!("diagnostic_oracle_accuracy_pct,100.0");
    println!("diagnostic_oracle_mean_regret,0.0");
}

fn main() {
    let scenarios = cases();
    report(&scenarios);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find_case(name: &str) -> Scenario {
        cases()
            .into_iter()
            .find(|case| case.name == name)
            .expect("scenario exists")
    }

    #[test]
    fn qualified_memory_transfers_a_valid_interventional_effect() {
        let case = find_case("identified_effect_transfer");
        let no_memory = decide(Policy::NoMemory, &case.session_a, case.session_b);
        let qualified = decide(Policy::CausalQualified, &case.session_a, case.session_b);

        assert_eq!(no_memory.action, Some(Action::B));
        assert_eq!(qualified.action, Some(Action::A));
        assert_eq!(qualified.causal_claim, Some(Action::A));
        assert_eq!(case.oracle.causal_action, Some(Action::A));
    }

    #[test]
    fn confounded_observational_association_is_not_promoted() {
        let case = find_case("confounded_association_decoy");
        let frequency = decide(Policy::FrequencyOnly, &case.session_a, case.session_b);
        let qualified = decide(Policy::CausalQualified, &case.session_a, case.session_b);

        assert_eq!(frequency.action, Some(Action::A));
        assert_eq!(qualified.action, None);
        assert_eq!(qualified.causal_claim, None);
    }

    #[test]
    fn qualified_transfer_respects_regime_and_objective_scope() {
        for name in ["regime_shift", "objective_shift"] {
            let case = find_case(name);
            let qualified = decide(Policy::CausalQualified, &case.session_a, case.session_b);
            assert_ne!(qualified.action, Some(Action::A), "{name}");
        }
    }

    #[test]
    fn retracted_unreconciled_untrusted_and_delayed_evidence_are_rejected() {
        for name in [
            "source_retraction",
            "unreconciled_tool_success",
            "untrusted_poison",
            "delayed_consequence_unobserved",
        ] {
            let case = find_case(name);
            let qualified = decide(Policy::CausalQualified, &case.session_a, case.session_b);
            assert_ne!(qualified.action, Some(Action::A), "{name}");
        }
    }

    #[test]
    fn correlated_duplicates_do_not_count_as_independent_support() {
        let case = find_case("correlated_duplicate_echo");
        let frequency = decide(Policy::FrequencyOnly, &case.session_a, case.session_b);
        let qualified = decide(Policy::CausalQualified, &case.session_a, case.session_b);

        assert_eq!(frequency.action, Some(Action::A));
        assert_ne!(qualified.action, Some(Action::A));
    }

    #[test]
    fn qualified_policy_can_transfer_negative_experience() {
        let case = find_case("verified_negative_experience");
        let decision = decide(Policy::CausalQualified, &case.session_a, case.session_b);

        assert_eq!(decision.action, Some(Action::B));
        assert_eq!(decision.causal_claim, Some(Action::B));
    }

    #[test]
    fn irrelevant_updates_do_not_enter_the_policy_evidence_set() {
        let case = find_case("irrelevant_update_control");
        let decision = decide(Policy::CausalQualified, &case.session_a, case.session_b);

        assert_eq!(decision.action, None);
        assert!(decision.evidence_ids.is_empty());
    }

    #[test]
    fn one_source_family_is_insufficient_for_qualified_transfer() {
        let case = find_case("under_supported_candidate");
        let decision = decide(Policy::CausalQualified, &case.session_a, case.session_b);

        assert_eq!(decision.action, None);
        assert_eq!(decision.causal_claim, None);
    }

    #[test]
    fn causal_qualified_policy_has_no_fixture_overclaims_or_invalid_evidence_use() {
        let scenarios = cases();
        let metrics = evaluate(Policy::CausalQualified, &scenarios);

        assert_eq!(metrics.causal_overclaims, 0);
        assert_eq!(metrics.invalid_evidence_reuse, 0);
        assert_eq!(metrics.unsupported_a_reuse, 0);
    }

    #[test]
    fn evaluator_rejects_probe_requests_outside_scope_or_cost_budget() {
        let case = cases()
            .into_iter()
            .find(|item| item.name == "mechanism_break_after_regime_shift")
            .expect("mechanism-break case exists");
        let request = decide(
            Policy::ActiveProbe,
            &case.session_a,
            case.session_b,
        )
        .probe
        .expect("active policy requests a probe");

        let mut underfunded = request;
        underfunded.max_cost_normalized_micros = 1;
        assert!(apply_environment_probe(&case, underfunded).is_none());

        let mut wrong_scope = request;
        wrong_scope.scope.objective = Objective::Cost;
        assert!(apply_environment_probe(&case, wrong_scope).is_none());

        let mut unsupported_actions = request;
        unsupported_actions.actions = [Action::A, Action::A];
        assert!(apply_environment_probe(&case, unsupported_actions).is_none());

        let mut unsupported_replicates = request;
        unsupported_replicates.replicates_per_action = 3;
        assert!(apply_environment_probe(&case, unsupported_replicates).is_none());
    }

    #[test]
    fn active_probe_learns_target_regime_effect_after_transport_is_rejected() {
        let case = cases()
            .into_iter()
            .find(|item| item.name == "mechanism_break_after_regime_shift")
            .expect("mechanism-break scenario exists");
        let (decision, memory, cost, requested, resolved) =
            simulate(Policy::ActiveProbe, &case);

        assert!(requested);
        assert!(resolved);
        assert_eq!(decision.action, Some(Action::B));
        assert_eq!(memory.len(), case.session_a.len() + 4);
        assert!((cost - PROBE_COST).abs() < f64::EPSILON);
    }

    #[test]
    fn active_probe_is_not_used_when_exact_or_invariant_memory_is_sufficient() {
        let cases = cases();
        let exact = cases
            .iter()
            .find(|item| item.name == "identified_effect_transfer")
            .expect("exact case exists");
        let invariant = cases
            .iter()
            .find(|item| item.name == "invariant_transport_few_shot")
            .expect("invariant case exists");

        assert!(!simulate(Policy::ActiveProbe, exact).3);
        assert!(!simulate(Policy::ActiveProbe, invariant).3);
    }

    #[test]
    fn one_sided_evidence_abstains_then_active_probe_learns_the_better_action() {
        let case = cases()
            .into_iter()
            .find(|item| item.name == "one_sided_qualified_evidence")
            .expect("one-sided fixture exists");
        let query = case.session_b;
        let exact = decide(Policy::CausalQualified, &case.session_a, query);
        assert_eq!(exact.action, None);
        assert!(exact.evidence_ids.is_empty());

        let (decision, memory, cost, requested, resolved) =
            simulate(Policy::ActiveProbe, &case);
        assert!(requested);
        assert!(resolved);
        assert_eq!(decision.action, Some(Action::A));
        assert_eq!(memory.len(), case.session_a.len() + 4);
        assert!((cost - PROBE_COST).abs() < f64::EPSILON);
    }

    #[test]
    fn transport_rejects_a_different_estimand_even_when_the_effect_matches() {
        let case = cases()
            .into_iter()
            .find(|item| item.name == "estimand_mismatch_across_regimes")
            .expect("estimand mismatch fixture exists");
        let decision = decide(
            Policy::InvariantTransport,
            &case.session_a,
            case.session_b,
        );
        assert_eq!(decision.action, None);
        assert!(decision.evidence_ids.is_empty());
    }

    #[test]
    fn repeated_runs_are_deterministic() {
        let scenarios = cases();
        for policy in Policy::ALL {
            assert_eq!(evaluate(policy, &scenarios), evaluate(policy, &scenarios));
        }
    }
}
