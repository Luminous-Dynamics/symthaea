use crate::realization_contract::RealizationControlV1;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const REALIZATION_EFFECT_SCHEMA_VERSION_V1: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlEffectMetricV1 {
    TempoMilliBpm,
    BeatPhaseMilliBeats,
    OnsetDensityMilliPerBeat,
    PitchClassEntropyMilliBits,
    SpectralCentroidMilliHz,
    IntegratedLoudnessMilliLufs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlEffectDirectionV1 {
    Increase,
    Decrease,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlProbeArmKindV1 {
    Baseline,
    Sham,
    Active,
    Reversal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlApplicationV1 {
    NoIntervention,
    ShamNoop,
    Applied,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlObservationEligibilityV1 {
    StructurallyEligible,
    Incomplete,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollateralMetricV1 {
    pub metric: ControlEffectMetricV1,
    pub value_milli: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlEffectProbeSpecV1 {
    pub schema_version: u32,
    pub probe_id: String,
    pub control: RealizationControlV1,
    pub target_metric: ControlEffectMetricV1,
    pub expected_direction: ControlEffectDirectionV1,
    pub baseline_control_value_milli: i64,
    pub treatment_control_value_milli: i64,
    pub minimum_effect_milli: u64,
    pub sham_equivalence_margin_milli: u64,
    pub reversal_return_margin_milli: u64,
    pub maximum_collateral_drift_milli: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlEffectArmReceiptV1 {
    pub arm_id: String,
    pub probe_session_id: String,
    pub arm: ControlProbeArmKindV1,
    pub application: ControlApplicationV1,
    pub backend_id: String,
    pub backend_version: String,
    pub adapter_version: String,
    pub capability_manifest_sha256: String,
    pub source_artifact_sha256: String,
    pub session_seed: u64,
    pub requested_control_value_milli: Option<i64>,
    pub reported_effective_control_value_milli: Option<i64>,
    pub accepted_by_backend: bool,
    pub ignored_by_backend: bool,
    pub target_metric: ControlEffectMetricV1,
    pub observed_target_value_milli: i64,
    pub collateral_metrics: Vec<CollateralMetricV1>,
    pub output_artifact_sha256: String,
    pub realization_receipt_sha256: String,
    pub capture_payload_sha256: Option<String>,
    pub capture_closure_receipt_sha256: Option<String>,
    pub event_observer_manifest_sha256: Option<String>,
    pub capture_observer_id: String,
    pub event_observer_id: String,
    pub producer_id: String,
    pub observation_eligibility: ControlObservationEligibilityV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlEffectIssueV1 {
    InvalidProbeSpec,
    WrongArmCount {
        expected: usize,
        found: usize,
    },
    DuplicateArm {
        arm: ControlProbeArmKindV1,
    },
    MissingArm {
        arm: ControlProbeArmKindV1,
    },
    InvalidDigest {
        arm: ControlProbeArmKindV1,
        field: String,
    },
    IdentityMismatch {
        arm: ControlProbeArmKindV1,
        field: String,
    },
    SeedMismatch {
        arm: ControlProbeArmKindV1,
    },
    SourceMismatch {
        arm: ControlProbeArmKindV1,
    },
    ObserverIsProducer {
        arm: ControlProbeArmKindV1,
    },
    MissingObservationReceipt {
        arm: ControlProbeArmKindV1,
    },
    InvalidApplication {
        arm: ControlProbeArmKindV1,
    },
    TargetMetricMismatch {
        arm: ControlProbeArmKindV1,
    },
    DuplicateCollateralMetric {
        arm: ControlProbeArmKindV1,
    },
    CollateralMetricSetMismatch {
        arm: ControlProbeArmKindV1,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlEffectVerdictV1 {
    QualifiedObservedEffect,
    ControlNotApplied,
    NoObservedEffect,
    ShamConfounded,
    ReversalFailed,
    CollateralDrift,
    Unobserved,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlEffectReportV1 {
    pub schema_version: u32,
    pub probe_id: String,
    pub verdict: ControlEffectVerdictV1,
    pub baseline_value_milli: Option<i64>,
    pub sham_value_milli: Option<i64>,
    pub active_value_milli: Option<i64>,
    pub reversal_value_milli: Option<i64>,
    pub sham_delta_milli: Option<i128>,
    pub active_delta_milli: Option<i128>,
    pub reversal_residual_milli: Option<u128>,
    pub maximum_collateral_drift_milli: Option<u128>,
    pub findings: Vec<String>,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn abs_diff(a: i64, b: i64) -> u128 {
    (i128::from(a) - i128::from(b)).unsigned_abs()
}

fn application_valid(arm: &ControlEffectArmReceiptV1, spec: &ControlEffectProbeSpecV1) -> bool {
    match arm.arm {
        ControlProbeArmKindV1::Baseline => {
            arm.application == ControlApplicationV1::NoIntervention
                && !arm.accepted_by_backend
                && !arm.ignored_by_backend
                && arm.requested_control_value_milli.is_none()
                && arm.reported_effective_control_value_milli
                    == Some(spec.baseline_control_value_milli)
        }
        ControlProbeArmKindV1::Sham => {
            arm.application == ControlApplicationV1::ShamNoop
                && !arm.accepted_by_backend
                && !arm.ignored_by_backend
                && arm.requested_control_value_milli.is_none()
                && arm.reported_effective_control_value_milli
                    == Some(spec.baseline_control_value_milli)
        }
        ControlProbeArmKindV1::Active => {
            arm.application == ControlApplicationV1::Applied
                && arm.requested_control_value_milli == Some(spec.treatment_control_value_milli)
        }
        ControlProbeArmKindV1::Reversal => {
            arm.application == ControlApplicationV1::Applied
                && arm.requested_control_value_milli == Some(spec.baseline_control_value_milli)
        }
    }
}

/// A provider acknowledgement is not an observed effect. This evaluator
/// requires baseline, sham, active and reversal arms with bound independent
/// capture/observer receipts. It does not authenticate those receipts or
/// decode audio; that is the responsibility of the independent verifier.
pub fn evaluate_control_effect_probe_v1(
    spec: &ControlEffectProbeSpecV1,
    arms: &[ControlEffectArmReceiptV1],
) -> Result<ControlEffectReportV1, Vec<ControlEffectIssueV1>> {
    if spec.schema_version != REALIZATION_EFFECT_SCHEMA_VERSION_V1
        || spec.probe_id.trim().is_empty()
        || spec.minimum_effect_milli == 0
        || spec.baseline_control_value_milli == spec.treatment_control_value_milli
        || spec.sham_equivalence_margin_milli >= spec.minimum_effect_milli
        || spec.reversal_return_margin_milli >= spec.minimum_effect_milli
    {
        return Err(vec![ControlEffectIssueV1::InvalidProbeSpec]);
    }
    if arms.len() != 4 {
        return Err(vec![ControlEffectIssueV1::WrongArmCount {
            expected: 4,
            found: arms.len(),
        }]);
    }
    let mut by_arm = BTreeMap::new();
    let mut issues = Vec::new();
    for arm in arms {
        if by_arm.insert(arm.arm, arm).is_some() {
            issues.push(ControlEffectIssueV1::DuplicateArm { arm: arm.arm });
        }
        if !application_valid(arm, spec) {
            issues.push(ControlEffectIssueV1::InvalidApplication { arm: arm.arm });
        }
        if arm.target_metric != spec.target_metric {
            issues.push(ControlEffectIssueV1::TargetMetricMismatch { arm: arm.arm });
        }
        for (field, digest) in [
            (
                "capability_manifest_sha256",
                Some(arm.capability_manifest_sha256.as_str()),
            ),
            (
                "source_artifact_sha256",
                Some(arm.source_artifact_sha256.as_str()),
            ),
            (
                "output_artifact_sha256",
                Some(arm.output_artifact_sha256.as_str()),
            ),
            (
                "realization_receipt_sha256",
                Some(arm.realization_receipt_sha256.as_str()),
            ),
            (
                "capture_payload_sha256",
                arm.capture_payload_sha256.as_deref(),
            ),
            (
                "capture_closure_receipt_sha256",
                arm.capture_closure_receipt_sha256.as_deref(),
            ),
            (
                "event_observer_manifest_sha256",
                arm.event_observer_manifest_sha256.as_deref(),
            ),
        ] {
            if let Some(value) = digest
                && !valid_digest(value)
            {
                issues.push(ControlEffectIssueV1::InvalidDigest {
                    arm: arm.arm,
                    field: field.into(),
                });
            }
        }
        if arm.observation_eligibility == ControlObservationEligibilityV1::StructurallyEligible {
            if arm.capture_payload_sha256.is_none()
                || arm.capture_closure_receipt_sha256.is_none()
                || arm.event_observer_manifest_sha256.is_none()
                || arm.capture_observer_id.trim().is_empty()
                || arm.event_observer_id.trim().is_empty()
                || arm.capture_observer_id == arm.producer_id
                || arm.event_observer_id == arm.producer_id
            {
                issues.push(ControlEffectIssueV1::MissingObservationReceipt { arm: arm.arm });
            }
            if arm.capture_observer_id == arm.producer_id
                || arm.event_observer_id == arm.producer_id
            {
                issues.push(ControlEffectIssueV1::ObserverIsProducer { arm: arm.arm });
            }
        }
        let mut collateral = BTreeSet::new();
        for metric in &arm.collateral_metrics {
            if !collateral.insert(metric.metric) {
                issues.push(ControlEffectIssueV1::DuplicateCollateralMetric { arm: arm.arm });
            }
            if metric.metric == spec.target_metric {
                issues.push(ControlEffectIssueV1::TargetMetricMismatch { arm: arm.arm });
            }
        }
    }
    for kind in [
        ControlProbeArmKindV1::Baseline,
        ControlProbeArmKindV1::Sham,
        ControlProbeArmKindV1::Active,
        ControlProbeArmKindV1::Reversal,
    ] {
        if !by_arm.contains_key(&kind) {
            issues.push(ControlEffectIssueV1::MissingArm { arm: kind });
        }
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let baseline = by_arm[&ControlProbeArmKindV1::Baseline];
    for arm in [
        by_arm[&ControlProbeArmKindV1::Sham],
        by_arm[&ControlProbeArmKindV1::Active],
        by_arm[&ControlProbeArmKindV1::Reversal],
    ] {
        for (field, same) in [
            (
                "probe_session_id",
                arm.probe_session_id == baseline.probe_session_id,
            ),
            ("backend_id", arm.backend_id == baseline.backend_id),
            (
                "backend_version",
                arm.backend_version == baseline.backend_version,
            ),
            (
                "adapter_version",
                arm.adapter_version == baseline.adapter_version,
            ),
            (
                "capability_manifest_sha256",
                arm.capability_manifest_sha256 == baseline.capability_manifest_sha256,
            ),
        ] {
            if !same {
                issues.push(ControlEffectIssueV1::IdentityMismatch {
                    arm: arm.arm,
                    field: field.into(),
                });
            }
        }
        if arm.session_seed != baseline.session_seed {
            issues.push(ControlEffectIssueV1::SeedMismatch { arm: arm.arm });
        }
        if arm.source_artifact_sha256 != baseline.source_artifact_sha256 {
            issues.push(ControlEffectIssueV1::SourceMismatch { arm: arm.arm });
        }
    }
    let baseline_metrics: BTreeMap<_, _> = baseline
        .collateral_metrics
        .iter()
        .map(|m| (m.metric, m.value_milli))
        .collect();
    for arm in [
        by_arm[&ControlProbeArmKindV1::Sham],
        by_arm[&ControlProbeArmKindV1::Active],
        by_arm[&ControlProbeArmKindV1::Reversal],
    ] {
        let keys: BTreeSet<_> = arm.collateral_metrics.iter().map(|m| m.metric).collect();
        if keys != baseline_metrics.keys().copied().collect() {
            issues.push(ControlEffectIssueV1::CollateralMetricSetMismatch { arm: arm.arm });
        }
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let sham = by_arm[&ControlProbeArmKindV1::Sham];
    let active = by_arm[&ControlProbeArmKindV1::Active];
    let reversal = by_arm[&ControlProbeArmKindV1::Reversal];
    let values = [
        baseline.observed_target_value_milli,
        sham.observed_target_value_milli,
        active.observed_target_value_milli,
        reversal.observed_target_value_milli,
    ];
    let common = |verdict, findings, active_delta, sham_delta, reversal_residual, collateral| {
        ControlEffectReportV1 {
            schema_version: REALIZATION_EFFECT_SCHEMA_VERSION_V1,
            probe_id: spec.probe_id.clone(),
            verdict,
            baseline_value_milli: Some(values[0]),
            sham_value_milli: Some(values[1]),
            active_value_milli: Some(values[2]),
            reversal_value_milli: Some(values[3]),
            sham_delta_milli: sham_delta,
            active_delta_milli: active_delta,
            reversal_residual_milli: reversal_residual,
            maximum_collateral_drift_milli: collateral,
            findings,
        }
    };
    if !arms.iter().all(|arm| {
        arm.observation_eligibility == ControlObservationEligibilityV1::StructurallyEligible
    }) {
        return Ok(common(
            ControlEffectVerdictV1::Unobserved,
            vec!["one or more output observation windows are incomplete or unknown".into()],
            None,
            None,
            None,
            None,
        ));
    }

    let active_not_applied = !active.accepted_by_backend
        || active.ignored_by_backend
        || active.reported_effective_control_value_milli
            != Some(spec.treatment_control_value_milli);
    let reversal_not_applied = !reversal.accepted_by_backend
        || reversal.ignored_by_backend
        || reversal.reported_effective_control_value_milli
            != Some(spec.baseline_control_value_milli);
    if active_not_applied || reversal_not_applied {
        return Ok(common(
            ControlEffectVerdictV1::ControlNotApplied,
            vec!["the adapter did not accept and report the preregistered active and reversal settings".into()],
            None,
            None,
            None,
            None,
        ));
    }

    let sham_delta = i128::from(values[1]) - i128::from(values[0]);
    let active_delta = i128::from(values[2]) - i128::from(values[0]);
    let reversal_residual = abs_diff(values[3], values[0]);
    let sham_confounded =
        sham_delta.unsigned_abs() > u128::from(spec.sham_equivalence_margin_milli);
    let effect = match spec.expected_direction {
        ControlEffectDirectionV1::Increase => active_delta >= i128::from(spec.minimum_effect_milli),
        ControlEffectDirectionV1::Decrease => {
            -active_delta >= i128::from(spec.minimum_effect_milli)
        }
    };
    let reversal_failed = reversal_residual > u128::from(spec.reversal_return_margin_milli);
    let mut max_collateral = 0u128;
    for arm in [sham, active, reversal] {
        for metric in &arm.collateral_metrics {
            max_collateral = max_collateral.max(abs_diff(
                metric.value_milli,
                baseline_metrics[&metric.metric],
            ));
        }
    }
    let collateral_failed = max_collateral > u128::from(spec.maximum_collateral_drift_milli);
    let (verdict, finding) = if sham_confounded {
        (
            ControlEffectVerdictV1::ShamConfounded,
            "sham/no-op changed the measured target beyond its equivalence margin",
        )
    } else if !effect {
        (
            ControlEffectVerdictV1::NoObservedEffect,
            "accepted control did not produce the preregistered measured effect",
        )
    } else if reversal_failed {
        (
            ControlEffectVerdictV1::ReversalFailed,
            "reversal did not return the measured target to baseline",
        )
    } else if collateral_failed {
        (
            ControlEffectVerdictV1::CollateralDrift,
            "a collateral metric exceeded its preregistered drift bound",
        )
    } else {
        (
            ControlEffectVerdictV1::QualifiedObservedEffect,
            "active effect, sham equivalence, reversal and collateral bounds all qualified",
        )
    };
    Ok(common(
        verdict,
        vec![finding.into()],
        Some(active_delta),
        Some(sham_delta),
        Some(reversal_residual),
        Some(max_collateral),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(byte: u8) -> String {
        format!("{:02x}", byte).repeat(32)
    }
    fn spec() -> ControlEffectProbeSpecV1 {
        ControlEffectProbeSpecV1 {
            schema_version: 1,
            probe_id: "tempo-probe-1".into(),
            control: RealizationControlV1::Tempo,
            target_metric: ControlEffectMetricV1::TempoMilliBpm,
            expected_direction: ControlEffectDirectionV1::Increase,
            baseline_control_value_milli: 100_000,
            treatment_control_value_milli: 120_000,
            minimum_effect_milli: 10_000,
            sham_equivalence_margin_milli: 2_000,
            reversal_return_margin_milli: 2_000,
            maximum_collateral_drift_milli: 5_000,
        }
    }
    fn arm(
        kind: ControlProbeArmKindV1,
        observed: i64,
        loudness: i64,
        byte: u8,
    ) -> ControlEffectArmReceiptV1 {
        let (application, requested, accepted, effective) = match kind {
            ControlProbeArmKindV1::Baseline => {
                (ControlApplicationV1::NoIntervention, None, false, 100_000)
            }
            ControlProbeArmKindV1::Sham => (ControlApplicationV1::ShamNoop, None, false, 100_000),
            ControlProbeArmKindV1::Active => {
                (ControlApplicationV1::Applied, Some(120_000), true, 120_000)
            }
            ControlProbeArmKindV1::Reversal => {
                (ControlApplicationV1::Applied, Some(100_000), true, 100_000)
            }
        };
        ControlEffectArmReceiptV1 {
            arm_id: format!("arm-{kind:?}"),
            probe_session_id: "session-1".into(),
            arm: kind,
            application,
            backend_id: "stream-backend".into(),
            backend_version: "v1".into(),
            adapter_version: "a1".into(),
            capability_manifest_sha256: d(1),
            source_artifact_sha256: d(2),
            session_seed: 42,
            requested_control_value_milli: requested,
            reported_effective_control_value_milli: Some(effective),
            accepted_by_backend: accepted,
            ignored_by_backend: false,
            target_metric: ControlEffectMetricV1::TempoMilliBpm,
            observed_target_value_milli: observed,
            collateral_metrics: vec![CollateralMetricV1 {
                metric: ControlEffectMetricV1::IntegratedLoudnessMilliLufs,
                value_milli: loudness,
            }],
            output_artifact_sha256: d(byte),
            realization_receipt_sha256: d(3),
            capture_payload_sha256: Some(d(4)),
            capture_closure_receipt_sha256: Some(d(5)),
            event_observer_manifest_sha256: Some(d(6)),
            capture_observer_id: "loopback".into(),
            event_observer_id: "detector".into(),
            producer_id: "backend".into(),
            observation_eligibility: ControlObservationEligibilityV1::StructurallyEligible,
        }
    }
    fn arms() -> Vec<ControlEffectArmReceiptV1> {
        vec![
            arm(ControlProbeArmKindV1::Baseline, 100_000, -16_000, 7),
            arm(ControlProbeArmKindV1::Sham, 101_000, -16_100, 8),
            arm(ControlProbeArmKindV1::Active, 120_000, -17_000, 9),
            arm(ControlProbeArmKindV1::Reversal, 101_000, -16_200, 10),
        ]
    }

    #[test]
    fn qualifies_only_when_active_sham_reversal_and_collateral_agree() {
        let report = evaluate_control_effect_probe_v1(&spec(), &arms()).unwrap();
        assert_eq!(
            report.verdict,
            ControlEffectVerdictV1::QualifiedObservedEffect
        );
        assert_eq!(report.active_delta_milli, Some(20_000));
        assert_eq!(report.sham_delta_milli, Some(1_000));
        assert_eq!(report.reversal_residual_milli, Some(1_000));
    }

    #[test]
    fn accepted_control_without_observed_effect_is_not_qualified() {
        let mut values = arms();
        values[2].observed_target_value_milli = 101_000;
        let report = evaluate_control_effect_probe_v1(&spec(), &values).unwrap();
        assert_eq!(report.verdict, ControlEffectVerdictV1::NoObservedEffect);
    }

    #[test]
    fn sham_effect_is_confounded() {
        let mut values = arms();
        values[1].observed_target_value_milli = 118_000;
        let report = evaluate_control_effect_probe_v1(&spec(), &values).unwrap();
        assert_eq!(report.verdict, ControlEffectVerdictV1::ShamConfounded);
    }

    #[test]
    fn incomplete_capture_is_unobserved_not_a_failed_effect() {
        let mut values = arms();
        values[2].observation_eligibility = ControlObservationEligibilityV1::Incomplete;
        let report = evaluate_control_effect_probe_v1(&spec(), &values).unwrap();
        assert_eq!(report.verdict, ControlEffectVerdictV1::Unobserved);
        assert_eq!(report.active_delta_milli, None);
    }

    #[test]
    fn backend_identity_must_match_across_arms() {
        let mut values = arms();
        values[2].backend_version = "v2".into();
        let issues = evaluate_control_effect_probe_v1(&spec(), &values).unwrap_err();
        assert!(issues.iter().any(|x| matches!(x, ControlEffectIssueV1::IdentityMismatch { arm: ControlProbeArmKindV1::Active, field } if field == "backend_version")));
    }

    #[test]
    fn observer_must_be_independent_from_producer() {
        let mut values = arms();
        values[2].event_observer_id = values[2].producer_id.clone();
        let issues = evaluate_control_effect_probe_v1(&spec(), &values).unwrap_err();
        assert!(issues.contains(&ControlEffectIssueV1::ObserverIsProducer {
            arm: ControlProbeArmKindV1::Active
        }));
    }

    #[test]
    fn missing_arm_is_rejected() {
        let mut values = arms();
        values.pop();
        assert!(evaluate_control_effect_probe_v1(&spec(), &values).is_err());
    }

    #[test]
    fn reversal_must_restore_baseline() {
        let mut values = arms();
        values[3].observed_target_value_milli = 110_000;
        let report = evaluate_control_effect_probe_v1(&spec(), &values).unwrap();
        assert_eq!(report.verdict, ControlEffectVerdictV1::ReversalFailed);
    }

    #[test]
    fn collateral_drift_is_not_hidden_by_target_success() {
        let mut values = arms();
        values[2].collateral_metrics[0].value_milli = -5_000;
        let report = evaluate_control_effect_probe_v1(&spec(), &values).unwrap();
        assert_eq!(report.verdict, ControlEffectVerdictV1::CollateralDrift);
    }

    #[test]
    fn invalid_observer_receipt_is_rejected() {
        let mut values = arms();
        values[2].capture_closure_receipt_sha256 = Some("bad".into());
        let issues = evaluate_control_effect_probe_v1(&spec(), &values).unwrap_err();
        assert!(issues.contains(&ControlEffectIssueV1::InvalidDigest {
            arm: ControlProbeArmKindV1::Active,
            field: "capture_closure_receipt_sha256".into()
        }));
    }

    #[test]
    fn ignored_active_control_is_control_not_applied_not_observed_effect() {
        let mut values = arms();
        values[2].accepted_by_backend = false;
        values[2].ignored_by_backend = true;
        let report = evaluate_control_effect_probe_v1(&spec(), &values).unwrap();
        assert_eq!(report.verdict, ControlEffectVerdictV1::ControlNotApplied);
        assert_eq!(report.active_delta_milli, None);
    }
}
