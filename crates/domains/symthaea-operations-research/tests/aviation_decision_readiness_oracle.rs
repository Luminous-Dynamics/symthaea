// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Public API integration fixtures for the composed decision readiness gate.

use symthaea_operations_research::aviation_calibration_protocol::{
    SplitAuditStatus, SplitIntegrityReport,
};
use symthaea_operations_research::aviation_data_contract::{
    AuditStatus, FeatureAuditReport,
};
use symthaea_operations_research::aviation_decision_readiness::{
    assess_decision_readiness, DecisionReadinessBlock as Block, DecisionReadinessInput,
    DecisionReadinessStatus as Status,
};
use symthaea_operations_research::aviation_distributional_robustness::{
    DistributionalRobustAssessment, DistributionalExclusion,
};
use symthaea_operations_research::aviation_enterprise::{
    VerificationReport, VerificationStatus,
};
use symthaea_operations_research::aviation_network_robustness::{
    ActorWorstRegression, RobustCandidateAssessment, RobustExclusion, RobustNetworkReport,
    ScenarioRisk,
};
use symthaea_operations_research::aviation_online_drift::{
    DriftAlternative, DriftEvidence, DriftMonitorStatus, DriftMonitorReport,
    DriftScenarioProbability, OnlineDriftMonitor,
};

fn feature(status: AuditStatus) -> FeatureAuditReport {
    FeatureAuditReport { status, findings: Vec::new() }
}

fn split(status: SplitAuditStatus) -> SplitIntegrityReport {
    SplitIntegrityReport { status, findings: Vec::new(), partitions: Vec::new() }
}

fn verification(status: VerificationStatus) -> VerificationReport {
    VerificationReport { status, violations: Vec::new(), recomputed_cost: Some(10) }
}

fn drift(status: DriftMonitorStatus) -> DriftMonitorReport {
    let mut monitor = OnlineDriftMonitor::new(
        vec![
            DriftScenarioProbability { scenario_id: "routine".into(), probability_ppm: 900_000 },
            DriftScenarioProbability { scenario_id: "disruption".into(), probability_ppm: 100_000 },
        ],
        vec![DriftAlternative {
            alternative_id: "shift".into(),
            mixture_weight_ppm: 1_000_000,
            probabilities: vec![
                DriftScenarioProbability { scenario_id: "routine".into(), probability_ppm: 500_000 },
                DriftScenarioProbability { scenario_id: "disruption".into(), probability_ppm: 500_000 },
            ],
        }],
        50_000,
    ).unwrap();
    match status {
        DriftMonitorStatus::InControl => monitor.report(),
        DriftMonitorStatus::DriftAlarm => {
            monitor.observe("e1", 1, "disruption").unwrap();
            monitor.observe("e2", 2, "disruption").unwrap()
        }
        DriftMonitorStatus::SupportShiftAlarm => monitor.observe("e1", 1, "unmodeled").unwrap(),
    }
}

fn distributional(eligible: bool) -> DistributionalRobustAssessment {
    DistributionalRobustAssessment {
        eligible,
        exclusions: if eligible {
            Vec::new()
        } else {
            vec![DistributionalExclusion::WorstCaseExpectedDeltaExceeded {
                actual_numerator_ppm: 1,
                maximum_units: 0,
            }]
        },
    }
}

fn network(eligible: bool) -> RobustCandidateAssessment {
    RobustCandidateAssessment {
        eligible,
        exclusions: if eligible {
            Vec::new()
        } else {
            vec![RobustExclusion::MeanBenefitBelowMinimum { actual: -1.0, minimum: 0.0 }]
        },
        report: RobustNetworkReport {
            scenario_count: 1,
            tail_fraction: 1.0,
            objective_id: "fixture".into(),
            mean_net_system_benefit_units: 10.0,
            worst_case_net_system_benefit_units: 10,
            upper_tail_system_regression_units: 0.0,
            worst_actor_regression_units: 0,
            scenarios_with_system_regression: 0,
            scenarios_with_system_improvement: 1,
            scenarios_unchanged: 0,
            actor_worst_regressions: Vec::<ActorWorstRegression>::new(),
            scenario_risks: Vec::<ScenarioRisk>::new(),
        },
    }
}

fn assess(
    feature_status: AuditStatus,
    split_status: SplitAuditStatus,
    plan_status: VerificationStatus,
    drift_status: DriftMonitorStatus,
    distributional_ok: bool,
    network_ok: bool,
    candidate_receipt: &str,
    active_receipt: &str,
) -> symthaea_operations_research::aviation_decision_readiness::DecisionReadinessReport {
    let feature = feature(feature_status);
    let split = split(split_status);
    let plan = verification(plan_status);
    let drift = drift(drift_status);
    let distributional = distributional(distributional_ok);
    let network = network(network_ok);
    assess_decision_readiness(DecisionReadinessInput {
        feature_audit: &feature,
        split_audit: &split,
        plan_verification: &plan,
        drift_monitor: &drift,
        distributional_assessment: &distributional,
        network_assessment: &network,
        candidate_calibration_receipt_id: candidate_receipt,
        active_calibration_receipt_id: active_receipt,
        calibration_created_at_epoch_seconds: 100,
        decision_time_epoch_seconds: 200,
        max_calibration_age_seconds: 1_000,
        max_monitor_observation_age_seconds: 30,
    })
}

#[test]
fn public_api_returns_only_human_review_eligibility_when_every_gate_accepts() {
    let report = assess(
        AuditStatus::Accept,
        SplitAuditStatus::Accept,
        VerificationStatus::Verified,
        DriftMonitorStatus::InControl,
        true,
        true,
        "calibration-v3",
        "calibration-v3",
    );
    assert_eq!(report.status, Status::EligibleForHumanReview);
    assert!(report.blockers.is_empty());
}

#[test]
fn public_api_blocks_stale_calibration_and_preserves_receipt_ids() {
    let report = assess(
        AuditStatus::Accept,
        SplitAuditStatus::Accept,
        VerificationStatus::Verified,
        DriftMonitorStatus::InControl,
        true,
        true,
        "calibration-v2",
        "calibration-v3",
    );
    assert_eq!(report.status, Status::Blocked);
    assert_eq!(report.blockers, vec![Block::StaleCalibrationReceipt]);
    assert_eq!(report.candidate_calibration_receipt_id, "calibration-v2");
    assert_eq!(report.active_calibration_receipt_id, "calibration-v3");
}

#[test]
fn public_api_blocks_drift_and_out_of_support_alarms_separately() {
    let drifted = assess(
        AuditStatus::Accept, SplitAuditStatus::Accept, VerificationStatus::Verified,
        DriftMonitorStatus::DriftAlarm, true, true, "c3", "c3",
    );
    assert_eq!(drifted.blockers, vec![Block::DriftAlarm]);
    assert_eq!(drifted.drift_evidence, DriftEvidence::MixtureLikelihoodRatio);

    let outside = assess(
        AuditStatus::Accept, SplitAuditStatus::Accept, VerificationStatus::Verified,
        DriftMonitorStatus::SupportShiftAlarm, true, true, "c3", "c3",
    );
    assert_eq!(outside.blockers, vec![Block::SupportShiftAlarm]);
    assert_eq!(outside.drift_evidence, DriftEvidence::ScenarioOutsideNominalSupport);
}

#[test]
fn public_api_collects_multiple_ineligible_gates_without_promoting_candidate() {
    let report = assess(
        AuditStatus::Indeterminate,
        SplitAuditStatus::Reject,
        VerificationStatus::Indeterminate,
        DriftMonitorStatus::DriftAlarm,
        false,
        false,
        "old",
        "current",
    );
    assert_eq!(report.status, Status::Blocked);
    assert_eq!(report.blockers, vec![
        Block::FeatureAuditIndeterminate,
        Block::SplitAuditRejected,
        Block::PlanVerificationIndeterminate,
        Block::StaleCalibrationReceipt,
        Block::DriftAlarm,
        Block::DistributionalRiskIneligible,
        Block::NetworkRiskIneligible,
    ]);
}
