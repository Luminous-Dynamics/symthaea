use serde_json::Value;
use symthaea_manufacturing_active_learning_bench::{
    AcquisitionObjective, ApplianceAcquisitionDecision, ApplianceAbstentionReason,
    ApplianceObservationQuality, ApplianceProtectionState, DecisionAlternative,
    FiniteHypothesisModel, HypothesisUtility, PosteriorState, condition_posterior,
    expected_information_gain, initial_posterior_state, rank_appliance_measurements,
    rank_feasible_actions_for_state,
};

const APPLIANCE_FIXTURE: &str =
    include_str!("../fixtures/appliance_fan_active_diagnostics_v0_1.json");

fn fixture_document() -> Value {
    serde_json::from_str(APPLIANCE_FIXTURE).expect("appliance fixture JSON must parse")
}

fn fixture_model() -> FiniteHypothesisModel {
    let document = fixture_document();
    assert_eq!(
        document["fixture_id"].as_str(),
        Some("APPLIANCE-FAN-ACTIVE-DIAGNOSTICS-001")
    );
    assert_eq!(
        document["status"].as_str(),
        Some("synthetic_mathematical_fixture_only")
    );
    serde_json::from_value(document["model"].clone())
        .expect("fixture model must deserialize into the shared kernel model")
}

fn ranked(decision: ApplianceAcquisitionDecision) -> symthaea_manufacturing_active_learning_bench::AcquisitionRanking {
    match decision {
        ApplianceAcquisitionDecision::Ranked { ranking } => ranking,
        other => panic!("expected ranking, got {other:?}"),
    }
}

#[test]
fn fixture_uses_shared_finite_hypothesis_model_and_stable_digest() {
    let model = fixture_model();
    model.validate().expect("fixture model must validate");
    let digest_a = model.digest().expect("model digest");
    let digest_b = model.digest().expect("model digest replay");
    assert_eq!(digest_a, digest_b);
    assert_eq!(digest_a.len(), 64);

    for action_id in ["rpm_check", "airflow_check", "vibration_check"] {
        let metrics = expected_information_gain(&model, action_id)
            .expect("shared kernel must score the appliance action");
        assert_eq!(metrics.model_digest, digest_a);
    }
}

#[test]
fn decision_value_per_cost_can_choose_a_different_measurement_than_eig_per_cost() {
    let model = fixture_model();

    let eig = ranked(
        rank_appliance_measurements(
            &model,
            ApplianceProtectionState::Clear,
            ApplianceObservationQuality::Complete,
            AcquisitionObjective::InformationGainPerCost,
        )
        .expect("EIG ranking"),
    );
    let decision_value = ranked(
        rank_appliance_measurements(
            &model,
            ApplianceProtectionState::Clear,
            ApplianceObservationQuality::Complete,
            AcquisitionObjective::DecisionValuePerCost,
        )
        .expect("decision-value ranking"),
    );

    assert_eq!(
        eig.ranked_actions.first().map(|a| a.action_id.as_str()),
        Some("rpm_check")
    );
    assert_eq!(
        decision_value.ranked_actions.first().map(|a| a.action_id.as_str()),
        Some("airflow_check")
    );

    let top = &decision_value.ranked_actions[0].metrics;
    assert!(top.expected_sample_information_value_milli > 1_600.0);
    assert!(top.decision_value_per_cost > 200.0);
    assert_eq!(top.model_digest, decision_value.model_digest);
}

#[test]
fn protection_and_observation_gates_abstain_before_any_measurement_is_ranked() {
    let model = fixture_model();

    let tripped = rank_appliance_measurements(
        &model,
        ApplianceProtectionState::Tripped,
        ApplianceObservationQuality::Stale,
        AcquisitionObjective::DecisionValuePerCost,
    )
    .expect("tripped protection is an abstention, not a model error");
    assert_eq!(
        tripped,
        ApplianceAcquisitionDecision::Abstain {
            reason: ApplianceAbstentionReason::ProtectionTripped
        }
    );

    let protection_unknown = rank_appliance_measurements(
        &model,
        ApplianceProtectionState::Unknown,
        ApplianceObservationQuality::Complete,
        AcquisitionObjective::DecisionValuePerCost,
    )
    .expect("unknown protection must abstain");
    assert_eq!(
        protection_unknown,
        ApplianceAcquisitionDecision::Abstain {
            reason: ApplianceAbstentionReason::ProtectionUnknown
        }
    );

    for (quality, reason) in [
        (
            ApplianceObservationQuality::Missing,
            ApplianceAbstentionReason::ObservationsMissing,
        ),
        (
            ApplianceObservationQuality::Stale,
            ApplianceAbstentionReason::ObservationsStale,
        ),
        (
            ApplianceObservationQuality::Conflicting,
            ApplianceAbstentionReason::ObservationsConflicting,
        ),
        (
            ApplianceObservationQuality::InvalidUnits,
            ApplianceAbstentionReason::ObservationUnitsInvalid,
        ),
    ] {
        let decision = rank_appliance_measurements(
            &model,
            ApplianceProtectionState::Clear,
            quality,
            AcquisitionObjective::DecisionValuePerCost,
        )
        .expect("unreliable observations must abstain");
        assert_eq!(
            decision,
            ApplianceAcquisitionDecision::Abstain { reason },
            "quality={quality:?}"
        );
    }
}

#[test]
fn blocked_or_unknown_test_is_never_scored_as_a_ranked_candidate() {
    let mut model = fixture_model();
    let airflow = model
        .actions
        .iter_mut()
        .find(|action| action.action_id == "airflow_check")
        .expect("airflow action");
    airflow.feasibility = symthaea_manufacturing_active_learning_bench::Feasibility::Blocked {
        reason: "the declared reference instrument is unavailable".into(),
    };

    let ranking = ranked(
        rank_appliance_measurements(
            &model,
            ApplianceProtectionState::Clear,
            ApplianceObservationQuality::Complete,
            AcquisitionObjective::DecisionValuePerCost,
        )
        .expect("remaining feasible tests may be ranked"),
    );
    assert!(ranking.ranked_actions.iter().all(|a| a.action_id != "airflow_check"));
    assert!(ranking.rejected_actions.iter().any(|a| {
        a.action_id == "airflow_check"
            && a.disposition == "Blocked"
            && a.reason == "the declared reference instrument is unavailable"
    }));
}

#[test]
fn informative_tests_with_zero_decision_value_return_no_decision_value() {
    let mut model = fixture_model();
    let only_decision = DecisionAlternative {
        decision_id: "retain_current_action".into(),
        utility_by_hypothesis: model
            .prior
            .iter()
            .map(|hypothesis| HypothesisUtility {
                hypothesis_id: hypothesis.hypothesis_id.clone(),
                utility_milli: -1_000,
            })
            .collect(),
    };
    model.decisions = vec![only_decision];

    let decision = rank_appliance_measurements(
        &model,
        ApplianceProtectionState::Clear,
        ApplianceObservationQuality::Complete,
        AcquisitionObjective::DecisionValuePerCost,
    )
    .expect("zero-value actions are valid but not useful");

    let ranking = match decision {
        ApplianceAcquisitionDecision::NoDecisionValue { ranking } => ranking,
        other => panic!("expected NoDecisionValue, got {other:?}"),
    };
    assert!(ranking.ranked_actions.iter().all(|a| {
        a.metrics.expected_sample_information_value_milli.abs() <= 1e-9
    }));
    // At least one test still reduces entropy, but the result cannot change the decision.
    assert!(ranking
        .ranked_actions
        .iter()
        .any(|a| a.metrics.expected_information_gain_nats > 0.1));
}

#[test]
fn posterior_state_is_reused_for_subsequent_appliance_measurement_ranking() {
    let model = fixture_model();
    let root: PosteriorState =
        initial_posterior_state(&model).expect("root posterior state");
    let update = condition_posterior(
        &model,
        &root,
        "appliance-observation-0001",
        "rpm_check",
        "low",
    )
    .expect("known synthetic outcome conditions posterior");

    let ranking = rank_feasible_actions_for_state(
        &model,
        &update.next_state,
        AcquisitionObjective::DecisionValuePerCost,
    )
    .expect("subsequent tests must use conditioned posterior");
    assert_eq!(
        ranking.posterior_state_digest.as_deref(),
        Some(update.next_state.state_digest.as_str())
    );
    assert!(ranking
        .ranked_actions
        .iter()
        .all(|a| a.metrics.posterior_state_digest.as_deref()
            == Some(update.next_state.state_digest.as_str())));
    assert_ne!(ranking.model_digest, root.root_model_digest);
}
