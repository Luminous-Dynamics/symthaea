// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Composed fail-closed readiness gate for the synthetic aviation enterprise path.
//!
//! The gate composes outputs of specialist verifiers and never authorizes dispatch.
//! Its strongest positive result is EligibleForHumanReview. Receipt IDs are
//! comparison bindings only; the caller must cryptographically bind them to
//! source snapshots/configuration in a future run-manifest layer.

use crate::aviation_calibration_protocol::{SplitAuditStatus, SplitIntegrityReport};
use crate::aviation_data_contract::{AuditStatus, FeatureAuditReport};
use crate::aviation_distributional_robustness::DistributionalRobustAssessment;
use crate::aviation_enterprise::{VerificationReport, VerificationStatus};
use crate::aviation_network_robustness::RobustCandidateAssessment;
use crate::aviation_online_drift::{DriftEvidence, DriftMonitorReport, DriftMonitorStatus};

/// Final result of the composition gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionReadinessStatus {
    Blocked,
    EligibleForHumanReview,
}

/// Every reason the candidate must not be promoted to human review.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionReadinessBlock {
    FeatureAuditRejected,
    FeatureAuditIndeterminate,
    SplitAuditRejected,
    SplitAuditIndeterminate,
    PlanVerificationRejected,
    PlanVerificationIndeterminate,
    MissingCalibrationReceipt,
    StaleCalibrationReceipt,
    InvalidFreshnessPolicy,
    InvalidCalibrationTimestamp,
    FutureCalibrationReceipt,
    CalibrationExpired,
    MonitoringObservationMissing,
    InvalidMonitoringObservationTimestamp,
    FutureMonitoringObservation,
    MonitoringObservationStale,
    DriftAlarm,
    SupportShiftAlarm,
    DistributionalRiskIneligible,
    NetworkRiskIneligible,
}

/// Outputs that must agree before a candidate can be surfaced for human review.
pub struct DecisionReadinessInput<'a> {
    pub feature_audit: &'a FeatureAuditReport,
    pub split_audit: &'a SplitIntegrityReport,
    pub plan_verification: &'a VerificationReport,
    pub drift_monitor: &'a DriftMonitorReport,
    pub distributional_assessment: &'a DistributionalRobustAssessment,
    pub network_assessment: &'a RobustCandidateAssessment,
    /// Identifier of the calibration receipt bound to the candidate plan/config.
    pub candidate_calibration_receipt_id: &'a str,
    /// Identifier of the active, most recently accepted calibration receipt.
    pub active_calibration_receipt_id: &'a str,
    /// Timestamp of the calibration run bound to this candidate (not merely its ID).
    pub calibration_created_at_epoch_seconds: i64,
    /// Time at which this decision is being assessed.
    pub decision_time_epoch_seconds: i64,
    /// Preregistered maximum age for the calibration receipt, in seconds.
    pub max_calibration_age_seconds: u64,
    /// Timestamp of the latest observation successfully ingested by the monitor.
    pub latest_monitor_observation_epoch_seconds: Option<i64>,
    /// Preregistered maximum age for online monitoring evidence, in seconds.
    pub max_monitor_observation_age_seconds: u64,
}

/// Auditable decision-readiness result. This is not operational authorization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionReadinessReport {
    pub status: DecisionReadinessStatus,
    pub blockers: Vec<DecisionReadinessBlock>,
    pub candidate_calibration_receipt_id: String,
    pub active_calibration_receipt_id: String,
    pub drift_evidence: DriftEvidence,
}

/// Compose the current evidence gates. All failures are returned in deterministic
/// order; the function never silently treats Indeterminate as Accept.
pub fn assess_decision_readiness(input: DecisionReadinessInput<'_>) -> DecisionReadinessReport {
    let mut blockers = Vec::new();

    match input.feature_audit.status {
        AuditStatus::Accept => {}
        AuditStatus::Reject => blockers.push(DecisionReadinessBlock::FeatureAuditRejected),
        AuditStatus::Indeterminate => blockers.push(DecisionReadinessBlock::FeatureAuditIndeterminate),
    }
    match input.split_audit.status {
        SplitAuditStatus::Accept => {}
        SplitAuditStatus::Reject => blockers.push(DecisionReadinessBlock::SplitAuditRejected),
        SplitAuditStatus::Indeterminate => blockers.push(DecisionReadinessBlock::SplitAuditIndeterminate),
    }
    match input.plan_verification.status {
        VerificationStatus::Verified => {}
        VerificationStatus::Rejected => blockers.push(DecisionReadinessBlock::PlanVerificationRejected),
        VerificationStatus::Indeterminate => blockers.push(DecisionReadinessBlock::PlanVerificationIndeterminate),
    }

    if input.candidate_calibration_receipt_id.trim().is_empty()
        || input.active_calibration_receipt_id.trim().is_empty()
    {
        blockers.push(DecisionReadinessBlock::MissingCalibrationReceipt);
    } else if input.candidate_calibration_receipt_id != input.active_calibration_receipt_id {
        blockers.push(DecisionReadinessBlock::StaleCalibrationReceipt);
    }

    if input.max_calibration_age_seconds == 0
        || input.max_monitor_observation_age_seconds == 0
    {
        blockers.push(DecisionReadinessBlock::InvalidFreshnessPolicy);
    }
    if input.decision_time_epoch_seconds < 0 || input.calibration_created_at_epoch_seconds < 0 {
        blockers.push(DecisionReadinessBlock::InvalidCalibrationTimestamp);
    } else if input.calibration_created_at_epoch_seconds > input.decision_time_epoch_seconds {
        blockers.push(DecisionReadinessBlock::FutureCalibrationReceipt);
    } else if (input.decision_time_epoch_seconds - input.calibration_created_at_epoch_seconds) as u64
        > input.max_calibration_age_seconds
    {
        blockers.push(DecisionReadinessBlock::CalibrationExpired);
    }

    match input.latest_monitor_observation_epoch_seconds {
        None => blockers.push(DecisionReadinessBlock::MonitoringObservationMissing),
        Some(timestamp) if timestamp < 0 || input.decision_time_epoch_seconds < 0 => {
            blockers.push(DecisionReadinessBlock::InvalidMonitoringObservationTimestamp);
        }
        Some(timestamp) if timestamp > input.decision_time_epoch_seconds => {
            blockers.push(DecisionReadinessBlock::FutureMonitoringObservation);
        }
        Some(timestamp)
            if (input.decision_time_epoch_seconds - timestamp) as u64
                > input.max_monitor_observation_age_seconds =>
        {
            blockers.push(DecisionReadinessBlock::MonitoringObservationStale);
        }
        Some(_) => {}
    }

    match input.drift_monitor.status {
        DriftMonitorStatus::InControl => {}
        DriftMonitorStatus::DriftAlarm => blockers.push(DecisionReadinessBlock::DriftAlarm),
        DriftMonitorStatus::SupportShiftAlarm => blockers.push(DecisionReadinessBlock::SupportShiftAlarm),
    }

    if !input.distributional_assessment.eligible {
        blockers.push(DecisionReadinessBlock::DistributionalRiskIneligible);
    }
    if !input.network_assessment.eligible {
        blockers.push(DecisionReadinessBlock::NetworkRiskIneligible);
    }

    DecisionReadinessReport {
        status: if blockers.is_empty() {
            DecisionReadinessStatus::EligibleForHumanReview
        } else {
            DecisionReadinessStatus::Blocked
        },
        blockers,
        candidate_calibration_receipt_id: input.candidate_calibration_receipt_id.into(),
        active_calibration_receipt_id: input.active_calibration_receipt_id.into(),
        drift_evidence: input.drift_monitor.evidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aviation_calibration_protocol::{
        EvaluationPartition, PartitionStats, SplitIntegrityReport,
    };
    use crate::aviation_data_contract::{FeatureAuditReport, AuditFinding};
    use crate::aviation_distributional_robustness::{
        DistributionalExclusion,
    };
    use crate::aviation_enterprise::VerificationReport;
    use crate::aviation_network_robustness::{
        ActorWorstRegression, RobustNetworkReport, RobustCandidateAssessment, ScenarioRisk,
        RobustExclusion,
    };
    use crate::aviation_online_drift::{
        DriftAlternative, DriftScenarioProbability, OnlineDriftMonitor,
    };

    fn feature(status: AuditStatus) -> FeatureAuditReport {
        FeatureAuditReport { status, findings: Vec::<AuditFinding>::new() }
    }

    fn split(status: SplitAuditStatus) -> SplitIntegrityReport {
        SplitIntegrityReport {
            status,
            findings: Vec::new(),
            partitions: [
                EvaluationPartition::Training,
                EvaluationPartition::Calibration,
                EvaluationPartition::Validation,
                EvaluationPartition::Test,
            ].into_iter().map(|partition| PartitionStats {
                partition,
                row_count: 3,
                distinct_event_clusters: 3,
                earliest_epoch_seconds: Some(10),
                latest_epoch_seconds: Some(12),
            }).collect(),
        }
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
            DriftMonitorStatus::SupportShiftAlarm => {
                monitor.observe("e1", 1, "unmodeled").unwrap()
            }
        }
    }

    fn distributional(eligible: bool) -> DistributionalRobustAssessment {
        DistributionalRobustAssessment {
            eligible,
            exclusions: if eligible { Vec::<DistributionalExclusion>::new() } else {
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
            exclusions: if eligible { Vec::<RobustExclusion>::new() } else {
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
        verification_status: VerificationStatus,
        drift_status: DriftMonitorStatus,
        robust_ok: bool,
        network_ok: bool,
        candidate_receipt: &str,
        active_receipt: &str,
    ) -> DecisionReadinessReport {
        let feature = feature(feature_status);
        let split = split(split_status);
        let verification = verification(verification_status);
        let drift = drift(drift_status);
        let distributional = distributional(robust_ok);
        let network = network(network_ok);
        assess_decision_readiness(DecisionReadinessInput {
            feature_audit: &feature,
            split_audit: &split,
            plan_verification: &verification,
            drift_monitor: &drift,
            distributional_assessment: &distributional,
            network_assessment: &network,
            candidate_calibration_receipt_id: candidate_receipt,
            active_calibration_receipt_id: active_receipt,
            calibration_created_at_epoch_seconds: 100,
            decision_time_epoch_seconds: 200,
            max_calibration_age_seconds: 1_000,
            latest_monitor_observation_epoch_seconds: Some(190),
            max_monitor_observation_age_seconds: 30,
        })
    }

    #[test]
    fn all_gates_pass_yields_only_human_review_eligibility() {
        let report = assess(
            AuditStatus::Accept, SplitAuditStatus::Accept, VerificationStatus::Verified,
            DriftMonitorStatus::InControl, true, true, "calib-v2", "calib-v2",
        );
        assert_eq!(report.status, DecisionReadinessStatus::EligibleForHumanReview);
        assert!(report.blockers.is_empty());
    }

    #[test]
    fn feature_audit_reject_blocks_readiness() {
        let report = assess(
            AuditStatus::Reject, SplitAuditStatus::Accept, VerificationStatus::Verified,
            DriftMonitorStatus::InControl, true, true, "calib-v2", "calib-v2",
        );
        assert_eq!(report.status, DecisionReadinessStatus::Blocked);
        assert_eq!(report.blockers, vec![DecisionReadinessBlock::FeatureAuditRejected]);
    }

    #[test]
    fn indeterminate_split_is_not_treated_as_acceptance() {
        let report = assess(
            AuditStatus::Accept, SplitAuditStatus::Indeterminate, VerificationStatus::Verified,
            DriftMonitorStatus::InControl, true, true, "calib-v2", "calib-v2",
        );
        assert_eq!(report.blockers, vec![DecisionReadinessBlock::SplitAuditIndeterminate]);
    }

    #[test]
    fn plan_verifier_rejection_blocks_readiness() {
        let report = assess(
            AuditStatus::Accept, SplitAuditStatus::Accept, VerificationStatus::Rejected,
            DriftMonitorStatus::InControl, true, true, "calib-v2", "calib-v2",
        );
        assert_eq!(report.blockers, vec![DecisionReadinessBlock::PlanVerificationRejected]);
    }

    #[test]
    fn stale_or_missing_calibration_receipt_blocks_readiness() {
        let stale = assess(
            AuditStatus::Accept, SplitAuditStatus::Accept, VerificationStatus::Verified,
            DriftMonitorStatus::InControl, true, true, "calib-v1", "calib-v2",
        );
        assert_eq!(stale.blockers, vec![DecisionReadinessBlock::StaleCalibrationReceipt]);
        let missing = assess(
            AuditStatus::Accept, SplitAuditStatus::Accept, VerificationStatus::Verified,
            DriftMonitorStatus::InControl, true, true, "", "calib-v2",
        );
        assert_eq!(missing.blockers, vec![DecisionReadinessBlock::MissingCalibrationReceipt]);
    }

    #[test]
    fn drift_alarm_and_support_shift_each_block_readiness() {
        for status in [DriftMonitorStatus::DriftAlarm, DriftMonitorStatus::SupportShiftAlarm] {
            let report = assess(
                AuditStatus::Accept, SplitAuditStatus::Accept, VerificationStatus::Verified,
                status, true, true, "calib-v2", "calib-v2",
            );
            assert_eq!(report.status, DecisionReadinessStatus::Blocked);
        }
    }

    #[test]
    fn either_robust_risk_gate_can_block_readiness() {
        let distributional = assess(
            AuditStatus::Accept, SplitAuditStatus::Accept, VerificationStatus::Verified,
            DriftMonitorStatus::InControl, false, true, "calib-v2", "calib-v2",
        );
        assert_eq!(distributional.blockers, vec![DecisionReadinessBlock::DistributionalRiskIneligible]);
        let network = assess(
            AuditStatus::Accept, SplitAuditStatus::Accept, VerificationStatus::Verified,
            DriftMonitorStatus::InControl, true, false, "calib-v2", "calib-v2",
        );
        assert_eq!(network.blockers, vec![DecisionReadinessBlock::NetworkRiskIneligible]);
    }

    #[test]
    fn blockers_accumulate_in_deterministic_order() {
        let report = assess(
            AuditStatus::Indeterminate, SplitAuditStatus::Reject, VerificationStatus::Indeterminate,
            DriftMonitorStatus::DriftAlarm, false, false, "old", "new",
        );
        assert_eq!(report.status, DecisionReadinessStatus::Blocked);
        assert_eq!(report.blockers, vec![
            DecisionReadinessBlock::FeatureAuditIndeterminate,
            DecisionReadinessBlock::SplitAuditRejected,
            DecisionReadinessBlock::PlanVerificationIndeterminate,
            DecisionReadinessBlock::StaleCalibrationReceipt,
            DecisionReadinessBlock::DriftAlarm,
            DecisionReadinessBlock::DistributionalRiskIneligible,
            DecisionReadinessBlock::NetworkRiskIneligible,
        ]);
    }

    #[test]
    fn support_shift_block_is_a_distinct_reason() {
        let report = assess(
            AuditStatus::Accept, SplitAuditStatus::Accept, VerificationStatus::Verified,
            DriftMonitorStatus::SupportShiftAlarm, true, true, "calib-v2", "calib-v2",
        );
        assert_eq!(report.blockers, vec![DecisionReadinessBlock::SupportShiftAlarm]);
    }

    #[test]
    fn never_returns_dispatch_authorization() {
        assert_eq!(
            assess(
                AuditStatus::Accept, SplitAuditStatus::Accept, VerificationStatus::Verified,
                DriftMonitorStatus::InControl, true, true, "calib-v2", "calib-v2",
            ).status,
            DecisionReadinessStatus::EligibleForHumanReview
        );
    }

    #[test]
    fn old_calibration_is_blocked_even_when_receipt_ids_match() {
        let feature = feature(AuditStatus::Accept);
        let split = split(SplitAuditStatus::Accept);
        let plan = verification(VerificationStatus::Verified);
        let drift = drift(DriftMonitorStatus::InControl);
        let distributional = distributional(true);
        let network = network(true);
        let report = assess_decision_readiness(DecisionReadinessInput {
            feature_audit: &feature,
            split_audit: &split,
            plan_verification: &plan,
            drift_monitor: &drift,
            distributional_assessment: &distributional,
            network_assessment: &network,
            candidate_calibration_receipt_id: "same",
            active_calibration_receipt_id: "same",
            calibration_created_at_epoch_seconds: 100,
            decision_time_epoch_seconds: 1_000,
            max_calibration_age_seconds: 100,
            latest_monitor_observation_epoch_seconds: Some(990),
            max_monitor_observation_age_seconds: 20,
        });
        assert_eq!(report.blockers, vec![DecisionReadinessBlock::CalibrationExpired]);
    }

    #[test]
    fn missing_or_stale_monitor_feed_blocks_readiness() {
        let missing = assess_with_feed_time(None);
        assert_eq!(missing.blockers, vec![DecisionReadinessBlock::MonitoringObservationMissing]);

        let stale = assess_with_feed_time(Some(100));
        assert_eq!(stale.blockers, vec![DecisionReadinessBlock::MonitoringObservationStale]);
    }

    fn assess_with_feed_time(latest: Option<i64>) -> DecisionReadinessReport {
        let feature = feature(AuditStatus::Accept);
        let split = split(SplitAuditStatus::Accept);
        let plan = verification(VerificationStatus::Verified);
        let drift = drift(DriftMonitorStatus::InControl);
        let distributional = distributional(true);
        let network = network(true);
        assess_decision_readiness(DecisionReadinessInput {
            feature_audit: &feature,
            split_audit: &split,
            plan_verification: &plan,
            drift_monitor: &drift,
            distributional_assessment: &distributional,
            network_assessment: &network,
            candidate_calibration_receipt_id: "same",
            active_calibration_receipt_id: "same",
            calibration_created_at_epoch_seconds: 100,
            decision_time_epoch_seconds: 200,
            max_calibration_age_seconds: 1_000,
            latest_monitor_observation_epoch_seconds: latest,
            max_monitor_observation_age_seconds: 10,
        })
    }

    #[test]
    fn future_timestamps_and_zero_age_limits_fail_closed() {
        let feature = feature(AuditStatus::Accept);
        let split = split(SplitAuditStatus::Accept);
        let plan = verification(VerificationStatus::Verified);
        let drift = drift(DriftMonitorStatus::InControl);
        let distributional = distributional(true);
        let network = network(true);
        let report = assess_decision_readiness(DecisionReadinessInput {
            feature_audit: &feature,
            split_audit: &split,
            plan_verification: &plan,
            drift_monitor: &drift,
            distributional_assessment: &distributional,
            network_assessment: &network,
            candidate_calibration_receipt_id: "same",
            active_calibration_receipt_id: "same",
            calibration_created_at_epoch_seconds: 300,
            decision_time_epoch_seconds: 200,
            max_calibration_age_seconds: 0,
            latest_monitor_observation_epoch_seconds: Some(250),
            max_monitor_observation_age_seconds: 0,
        });
        assert!(report.blockers.contains(&DecisionReadinessBlock::InvalidFreshnessPolicy));
        assert!(report.blockers.contains(&DecisionReadinessBlock::FutureCalibrationReceipt));
        assert!(report.blockers.contains(&DecisionReadinessBlock::FutureMonitoringObservation));
    }

}
