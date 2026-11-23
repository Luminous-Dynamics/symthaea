// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability-to-observed-effect qualification for musical renderer controls.
//!
//! A declared capability or accepted API request is not evidence that the
//! rendered music followed the requested control. This module binds a
//! versioned capability manifest, intervention request, execution receipt,
//! and independently observed output. It deliberately does not authenticate
//! receipts or replace the capture-closure and detector-qualification owners.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const CONTROL_EFFECT_SCHEMA_VERSION_V1: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicalControlV1 {
    Tempo,
    TonalCenter,
    NoteDensity,
    Brightness,
    ChordRoot,
    MelodyPitchClass,
    GrooveAccent,
    InstrumentRoleEntry,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlSupportLevelV1 {
    /// The backend contract says the requested value must be applied exactly.
    Hard,
    /// The backend attempts the control but may approximate or ignore it.
    BestEffort,
    /// The value is advisory context, not an execution guarantee.
    Advisory,
    /// This backend/version does not expose the control.
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlEffectDirectionV1 {
    Increase,
    Decrease,
    NoChange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CounterfactualConditionV1 {
    Active,
    Sham,
    Reversal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlExecutionStatusV1 {
    /// The backend claims the effective control was applied.
    Applied,
    /// The API accepted the request but did not attest application.
    Accepted,
    /// The backend explicitly reports that it ignored the control.
    Ignored,
    Rejected,
    Unsupported,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationClosureStateV1 {
    /// Supplied by the output-observation closure evaluator. This module
    /// checks linkage, not the authenticity of that evaluator's receipt.
    StructurallyEligible,
    Incomplete,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlEffectVerdictV1 {
    CapabilityUnsupported,
    NotExecuted,
    Unobserved,
    ObservedEffectVerified,
    ObservedEffectInconclusive,
    ObservedEffectContradictory,
    ShamNoChangeVerified,
    ShamLeakage,
    ObservedEffectUnattributed,
    CollateralDriftExceeded,
    HardControlValueMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendControlCapabilityV1 {
    pub capability_id: String,
    pub control: MusicalControlV1,
    pub support: ControlSupportLevelV1,
    /// Requested control values are fixed-point thousandths in the
    /// capability's declared unit (for example BPM * 1000).
    pub minimum_value_milli: i64,
    pub maximum_value_milli: i64,
    pub unit: String,
    /// Independent observation metric expected to move when this control is
    /// applied (for example median beat period in microseconds).
    pub target_metric_id: String,
    /// Digest of the versioned metric definition, unit, aggregation and windowing.
    pub target_metric_contract_sha256: String,
    /// Minimum absolute effect required for a qualified active intervention.
    pub minimum_effect_milli: u64,
    /// Maximum effect tolerated for a sham/no-op intervention, including
    /// measurement uncertainty.
    pub sham_equivalence_tolerance_milli: u64,
    pub maximum_collateral_drift_milli: u64,
    pub source_references: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendCapabilityManifestPayloadV1 {
    pub schema_version: u32,
    pub backend_id: String,
    pub backend_version: String,
    pub adapter_version: String,
    /// Must match the immutable backend/runtime identity used by the paired
    /// evaluator (for example the RMI immutable identity receipt).
    pub backend_identity_sha256: String,
    pub capabilities: Vec<BackendControlCapabilityV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendCapabilityManifestV1 {
    pub payload: BackendCapabilityManifestPayloadV1,
    pub payload_sha256: String,
}

impl BackendCapabilityManifestV1 {
    pub fn from_payload(payload: BackendCapabilityManifestPayloadV1) -> Result<Self, String> {
        validate_manifest_payload(&payload)?;
        let payload_sha256 = digest_json(&payload)?;
        Ok(Self {
            payload,
            payload_sha256,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_manifest_payload(&self.payload)?;
        if self.payload_sha256 != digest_json(&self.payload)? {
            return Err("capability manifest digest does not match payload".to_owned());
        }
        Ok(())
    }

    pub fn capability(&self, capability_id: &str) -> Option<&BackendControlCapabilityV1> {
        self.payload
            .capabilities
            .iter()
            .find(|capability| capability.capability_id == capability_id)
    }
}

fn validate_manifest_payload(payload: &BackendCapabilityManifestPayloadV1) -> Result<(), String> {
    if payload.schema_version != CONTROL_EFFECT_SCHEMA_VERSION_V1 {
        return Err("unsupported capability manifest schema version".to_owned());
    }
    require_text(&payload.backend_id, "backend_id")?;
    require_text(&payload.backend_version, "backend_version")?;
    require_text(&payload.adapter_version, "adapter_version")?;
    require_digest(&payload.backend_identity_sha256, "backend identity digest")?;
    if payload.capabilities.is_empty() {
        return Err("capability manifest must contain at least one capability".to_owned());
    }

    let mut ids = BTreeSet::new();
    for capability in &payload.capabilities {
        require_text(&capability.capability_id, "capability_id")?;
        require_text(&capability.unit, "capability unit")?;
        require_text(&capability.target_metric_id, "target metric ID")?;
        require_digest(
            &capability.target_metric_contract_sha256,
            "target metric contract digest",
        )?;
        if !ids.insert(capability.capability_id.as_str()) {
            return Err(format!(
                "duplicate capability_id: {}",
                capability.capability_id
            ));
        }
        if capability.minimum_value_milli > capability.maximum_value_milli {
            return Err(format!(
                "capability {} has an inverted supported range",
                capability.capability_id
            ));
        }
        if capability.minimum_effect_milli == 0 {
            return Err(format!(
                "capability {} must declare a positive minimum effect",
                capability.capability_id
            ));
        }
        if capability.source_references.is_empty()
            || capability
                .source_references
                .iter()
                .any(|reference| reference.trim().is_empty())
        {
            return Err(format!(
                "capability {} is missing source references",
                capability.capability_id
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlInterventionRequestPayloadV1 {
    pub schema_version: u32,
    pub request_id: String,
    pub counterfactual_pair_id: String,
    pub condition: CounterfactualConditionV1,
    pub manifest_sha256: String,
    pub backend_id: String,
    pub backend_identity_sha256: String,
    pub capability_id: String,
    pub baseline_control_value_milli: i64,
    pub requested_control_value_milli: i64,
    pub expected_effect_direction: ControlEffectDirectionV1,
    pub baseline_capture_sha256: String,
    pub baseline_observation_closure_sha256: String,
    pub baseline_observation_manifest_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlInterventionRequestV1 {
    pub payload: ControlInterventionRequestPayloadV1,
    pub payload_sha256: String,
}

impl ControlInterventionRequestV1 {
    pub fn from_payload(payload: ControlInterventionRequestPayloadV1) -> Result<Self, String> {
        validate_request_shape(&payload)?;
        let payload_sha256 = digest_json(&payload)?;
        Ok(Self {
            payload,
            payload_sha256,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_request_shape(&self.payload)?;
        if self.payload_sha256 != digest_json(&self.payload)? {
            return Err("control request digest does not match payload".to_owned());
        }
        Ok(())
    }
}

fn validate_request_shape(payload: &ControlInterventionRequestPayloadV1) -> Result<(), String> {
    if payload.schema_version != CONTROL_EFFECT_SCHEMA_VERSION_V1 {
        return Err("unsupported control request schema version".to_owned());
    }
    for (value, name) in [
        (&payload.request_id, "request_id"),
        (&payload.counterfactual_pair_id, "counterfactual_pair_id"),
        (&payload.backend_id, "request backend_id"),
        (&payload.capability_id, "capability_id"),
    ] {
        require_text(value, name)?;
    }
    for (digest, name) in [
        (&payload.manifest_sha256, "request manifest digest"),
        (
            &payload.backend_identity_sha256,
            "request backend identity digest",
        ),
        (&payload.baseline_capture_sha256, "baseline capture digest"),
        (
            &payload.baseline_observation_closure_sha256,
            "baseline observation closure digest",
        ),
        (
            &payload.baseline_observation_manifest_sha256,
            "baseline observation manifest digest",
        ),
    ] {
        require_digest(digest, name)?;
    }
    match payload.condition {
        CounterfactualConditionV1::Sham => {
            if payload.expected_effect_direction != ControlEffectDirectionV1::NoChange
                || payload.requested_control_value_milli != payload.baseline_control_value_milli
            {
                return Err(
                    "sham request must preserve the baseline control value and expect no change"
                        .to_owned(),
                );
            }
        }
        CounterfactualConditionV1::Active | CounterfactualConditionV1::Reversal => {
            if payload.expected_effect_direction == ControlEffectDirectionV1::NoChange
                || payload.requested_control_value_milli == payload.baseline_control_value_milli
            {
                return Err("active/reversal request must change the control value and declare an effect direction".to_owned());
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlExecutionReceiptPayloadV1 {
    pub schema_version: u32,
    pub request_sha256: String,
    pub backend_id: String,
    pub backend_identity_sha256: String,
    pub execution_status: ControlExecutionStatusV1,
    pub effective_control_value_milli: Option<i64>,
    pub output_capture_sha256: Option<String>,
    pub observation_closure_sha256: Option<String>,
    pub observation_manifest_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlExecutionReceiptV1 {
    pub payload: ControlExecutionReceiptPayloadV1,
    pub payload_sha256: String,
}

impl ControlExecutionReceiptV1 {
    pub fn from_payload(payload: ControlExecutionReceiptPayloadV1) -> Result<Self, String> {
        validate_execution_receipt_shape(&payload)?;
        let payload_sha256 = digest_json(&payload)?;
        Ok(Self {
            payload,
            payload_sha256,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_execution_receipt_shape(&self.payload)?;
        if self.payload_sha256 != digest_json(&self.payload)? {
            return Err("execution receipt digest does not match payload".to_owned());
        }
        Ok(())
    }
}

fn validate_execution_receipt_shape(
    payload: &ControlExecutionReceiptPayloadV1,
) -> Result<(), String> {
    if payload.schema_version != CONTROL_EFFECT_SCHEMA_VERSION_V1 {
        return Err("unsupported execution receipt schema version".to_owned());
    }
    require_digest(&payload.request_sha256, "receipt request digest")?;
    require_text(&payload.backend_id, "receipt backend_id")?;
    require_digest(
        &payload.backend_identity_sha256,
        "receipt backend identity digest",
    )?;

    if payload.execution_status == ControlExecutionStatusV1::Applied
        && payload.effective_control_value_milli.is_none()
    {
        return Err("applied receipt must report the effective control value".to_owned());
    }
    for (digest, name) in [
        (
            payload.output_capture_sha256.as_ref(),
            "output capture digest",
        ),
        (
            payload.observation_closure_sha256.as_ref(),
            "observation closure digest",
        ),
        (
            payload.observation_manifest_sha256.as_ref(),
            "observation manifest digest",
        ),
    ] {
        if let Some(digest) = digest {
            require_digest(digest, name)?;
        }
    }
    let has_capture = payload.output_capture_sha256.is_some();
    let has_closure = payload.observation_closure_sha256.is_some();
    let has_manifest = payload.observation_manifest_sha256.is_some();
    if has_capture != has_closure || has_capture != has_manifest {
        return Err("capture, observation-closure and observer-manifest references must be supplied together".to_owned());
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlOutputObservationV1 {
    pub request_sha256: String,
    pub counterfactual_pair_id: String,
    pub capture_observer_id: String,
    pub event_observer_id: String,
    pub output_producer_id: String,
    pub baseline_capture_sha256: String,
    pub baseline_observation_closure_sha256: String,
    pub baseline_observation_manifest_sha256: String,
    pub baseline_closure_state: ObservationClosureStateV1,
    pub output_capture_sha256: String,
    pub observation_closure_sha256: String,
    pub observation_manifest_sha256: String,
    pub closure_state: ObservationClosureStateV1,
    pub observer_method_version: String,
    pub target_metric_id: String,
    pub target_metric_contract_sha256: String,
    pub observer_qualification_receipt_sha256: String,
    pub metric_calibration_receipt_sha256: String,
    /// Independently measured fixed-point baseline metric value.
    pub baseline_metric_value_milli: i64,
    /// Independently measured fixed-point output metric value.
    pub output_metric_value_milli: i64,
    /// Separate uncertainty radii; the evaluator adds them conservatively.
    pub baseline_metric_uncertainty_milli: u64,
    pub output_metric_uncertainty_milli: u64,
    /// Worst observed non-target disturbance in the predeclared invariant set.
    pub collateral_drift_milli: u64,
    pub collateral_uncertainty_milli: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlEffectFindingV1 {
    pub request_id: String,
    pub counterfactual_pair_id: String,
    pub capability_id: String,
    pub backend_identity_sha256: String,
    pub baseline_capture_sha256: String,
    pub baseline_observation_closure_sha256: String,
    pub baseline_observation_manifest_sha256: String,
    pub baseline_control_value_milli: i64,
    pub requested_control_value_milli: i64,
    pub expected_effect_direction: ControlEffectDirectionV1,
    pub support_level: ControlSupportLevelV1,
    pub condition: CounterfactualConditionV1,
    pub verdict: ControlEffectVerdictV1,
    pub execution_status: ControlExecutionStatusV1,
    pub effective_value_matches_request: Option<bool>,
    pub observed_target_effect_milli: Option<i128>,
    pub target_effect_uncertainty_milli: Option<u128>,
    pub collateral_drift_milli: Option<u64>,
    pub evidence_refs: Vec<String>,
}

/// Evaluate one intervention condition. The backend's acknowledgement is not
/// accepted as proof of success: an independently identified observer must
/// bind the output capture, closure record and observed-event manifest.
///
/// The function only checks structural linkage and measured effect contracts.
/// It does not authenticate receipts, validate detector sensitivity, or
/// independently redo #13348's capture-closure arithmetic.
pub fn evaluate_control_effect_v1(
    manifest: &BackendCapabilityManifestV1,
    request: &ControlInterventionRequestV1,
    receipt: &ControlExecutionReceiptV1,
    observation: Option<&ControlOutputObservationV1>,
) -> Result<ControlEffectFindingV1, String> {
    manifest.validate()?;
    request.validate()?;
    receipt.validate()?;

    let capability = manifest
        .capability(&request.payload.capability_id)
        .ok_or_else(|| "request names a capability absent from the manifest".to_owned())?;

    if request.payload.manifest_sha256 != manifest.payload_sha256
        || request.payload.backend_id != manifest.payload.backend_id
        || request.payload.backend_identity_sha256 != manifest.payload.backend_identity_sha256
    {
        return Err(
            "request is not bound to the selected capability manifest/backend identity".to_owned(),
        );
    }
    if request.payload.requested_control_value_milli < capability.minimum_value_milli
        || request.payload.requested_control_value_milli > capability.maximum_value_milli
    {
        return Err("requested control value is outside the declared capability range".to_owned());
    }
    if receipt.payload.request_sha256 != request.payload_sha256
        || receipt.payload.backend_id != manifest.payload.backend_id
        || receipt.payload.backend_identity_sha256 != manifest.payload.backend_identity_sha256
    {
        return Err(
            "execution receipt is not bound to the exact request/backend identity".to_owned(),
        );
    }

    let effective_value_matches_request = receipt
        .payload
        .effective_control_value_milli
        .map(|value| value == request.payload.requested_control_value_milli);

    let base_finding =
        |verdict, observation: Option<&ControlOutputObservationV1>| ControlEffectFindingV1 {
            request_id: request.payload.request_id.clone(),
            counterfactual_pair_id: request.payload.counterfactual_pair_id.clone(),
            capability_id: capability.capability_id.clone(),
            backend_identity_sha256: manifest.payload.backend_identity_sha256.clone(),
            baseline_capture_sha256: request.payload.baseline_capture_sha256.clone(),
            baseline_observation_closure_sha256: request
                .payload
                .baseline_observation_closure_sha256
                .clone(),
            baseline_observation_manifest_sha256: request
                .payload
                .baseline_observation_manifest_sha256
                .clone(),
            baseline_control_value_milli: request.payload.baseline_control_value_milli,
            requested_control_value_milli: request.payload.requested_control_value_milli,
            expected_effect_direction: request.payload.expected_effect_direction,
            support_level: capability.support,
            condition: request.payload.condition,
            verdict,
            execution_status: receipt.payload.execution_status,
            effective_value_matches_request,
            observed_target_effect_milli: observation.map(|value| {
                i128::from(value.output_metric_value_milli)
                    - i128::from(value.baseline_metric_value_milli)
            }),
            target_effect_uncertainty_milli: observation.map(|value| {
                u128::from(value.baseline_metric_uncertainty_milli)
                    + u128::from(value.output_metric_uncertainty_milli)
            }),
            collateral_drift_milli: observation.map(|value| value.collateral_drift_milli),
            evidence_refs: [
                Some(request.payload.baseline_capture_sha256.clone()),
                Some(request.payload.baseline_observation_closure_sha256.clone()),
                Some(request.payload.baseline_observation_manifest_sha256.clone()),
                receipt.payload.output_capture_sha256.clone(),
                receipt.payload.observation_closure_sha256.clone(),
                receipt.payload.observation_manifest_sha256.clone(),
                observation.map(|value| value.target_metric_contract_sha256.clone()),
                observation.map(|value| value.observer_qualification_receipt_sha256.clone()),
                observation.map(|value| value.metric_calibration_receipt_sha256.clone()),
            ]
            .into_iter()
            .flatten()
            .collect(),
        };

    if capability.support == ControlSupportLevelV1::Unsupported
        || receipt.payload.execution_status == ControlExecutionStatusV1::Unsupported
    {
        return Ok(base_finding(
            ControlEffectVerdictV1::CapabilityUnsupported,
            None,
        ));
    }
    if matches!(
        receipt.payload.execution_status,
        ControlExecutionStatusV1::Rejected | ControlExecutionStatusV1::Failed
    ) {
        return Ok(base_finding(ControlEffectVerdictV1::NotExecuted, None));
    }
    if capability.support == ControlSupportLevelV1::Hard
        && receipt.payload.execution_status == ControlExecutionStatusV1::Applied
        && effective_value_matches_request == Some(false)
    {
        return Ok(base_finding(
            ControlEffectVerdictV1::HardControlValueMismatch,
            None,
        ));
    }

    let Some(observation) = observation else {
        return Ok(base_finding(ControlEffectVerdictV1::Unobserved, None));
    };

    validate_observation_linkage(manifest, request, receipt, capability, observation)?;

    if observation.baseline_closure_state != ObservationClosureStateV1::StructurallyEligible
        || observation.closure_state != ObservationClosureStateV1::StructurallyEligible
    {
        return Ok(base_finding(
            ControlEffectVerdictV1::Unobserved,
            Some(observation),
        ));
    }

    let collateral_upper = u128::from(observation.collateral_drift_milli)
        + u128::from(observation.collateral_uncertainty_milli);
    if collateral_upper > u128::from(capability.maximum_collateral_drift_milli) {
        return Ok(base_finding(
            ControlEffectVerdictV1::CollateralDriftExceeded,
            Some(observation),
        ));
    }

    let observed_effect = i128::from(observation.output_metric_value_milli)
        - i128::from(observation.baseline_metric_value_milli);
    let combined_uncertainty = i128::from(observation.baseline_metric_uncertainty_milli)
        + i128::from(observation.output_metric_uncertainty_milli);
    let lower = observed_effect - combined_uncertainty;
    let upper = observed_effect + combined_uncertainty;
    let min_effect = i128::from(capability.minimum_effect_milli);

    let measured_verdict = match request.payload.expected_effect_direction {
        ControlEffectDirectionV1::Increase => {
            if lower >= min_effect {
                ControlEffectVerdictV1::ObservedEffectVerified
            } else if upper <= -min_effect {
                ControlEffectVerdictV1::ObservedEffectContradictory
            } else {
                ControlEffectVerdictV1::ObservedEffectInconclusive
            }
        }
        ControlEffectDirectionV1::Decrease => {
            if upper <= -min_effect {
                ControlEffectVerdictV1::ObservedEffectVerified
            } else if lower >= min_effect {
                ControlEffectVerdictV1::ObservedEffectContradictory
            } else {
                ControlEffectVerdictV1::ObservedEffectInconclusive
            }
        }
        ControlEffectDirectionV1::NoChange => {
            let worst_case = lower.unsigned_abs().max(upper.unsigned_abs());
            if worst_case <= u128::from(capability.sham_equivalence_tolerance_milli) {
                ControlEffectVerdictV1::ShamNoChangeVerified
            } else {
                ControlEffectVerdictV1::ShamLeakage
            }
        }
    };

    let verdict = if measured_verdict == ControlEffectVerdictV1::ObservedEffectVerified
        && receipt.payload.execution_status != ControlExecutionStatusV1::Applied
    {
        ControlEffectVerdictV1::ObservedEffectUnattributed
    } else {
        measured_verdict
    };

    Ok(base_finding(verdict, Some(observation)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CounterfactualControlVerdictV1 {
    Qualified,
    ActiveEffectNotVerified,
    ShamInvarianceNotVerified,
    ReversalEffectNotVerified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualControlQualificationV1 {
    pub counterfactual_pair_id: String,
    pub capability_id: String,
    pub backend_identity_sha256: String,
    pub baseline_capture_sha256: String,
    pub baseline_observation_closure_sha256: String,
    pub baseline_observation_manifest_sha256: String,
    pub verdict: CounterfactualControlVerdictV1,
    pub active: ControlEffectFindingV1,
    pub sham: ControlEffectFindingV1,
    pub reversal: ControlEffectFindingV1,
}

/// Qualify a control only when the active intervention moves the target,
/// the sham remains within the predeclared equivalence band, and a reversed
/// intervention produces the opposite effect. The three arms must share the
/// exact backend identity, capability, pair ID and baseline capture.
pub fn qualify_counterfactual_control_triplet_v1(
    active: &ControlEffectFindingV1,
    sham: &ControlEffectFindingV1,
    reversal: &ControlEffectFindingV1,
) -> Result<CounterfactualControlQualificationV1, String> {
    if active.condition != CounterfactualConditionV1::Active
        || sham.condition != CounterfactualConditionV1::Sham
        || reversal.condition != CounterfactualConditionV1::Reversal
    {
        return Err("counterfactual triplet has incorrect active/sham/reversal roles".to_owned());
    }
    for finding in [sham, reversal] {
        if finding.counterfactual_pair_id != active.counterfactual_pair_id
            || finding.capability_id != active.capability_id
            || finding.backend_identity_sha256 != active.backend_identity_sha256
            || finding.baseline_capture_sha256 != active.baseline_capture_sha256
            || finding.baseline_observation_closure_sha256
                != active.baseline_observation_closure_sha256
            || finding.baseline_observation_manifest_sha256
                != active.baseline_observation_manifest_sha256
        {
            return Err("counterfactual triplet is not matched on pair, capability, backend identity and baseline".to_owned());
        }
    }
    let directions_are_reversed = matches!(
        (
            active.expected_effect_direction,
            reversal.expected_effect_direction
        ),
        (
            ControlEffectDirectionV1::Increase,
            ControlEffectDirectionV1::Decrease
        ) | (
            ControlEffectDirectionV1::Decrease,
            ControlEffectDirectionV1::Increase
        )
    );
    let active_control_delta = i128::from(active.requested_control_value_milli)
        - i128::from(active.baseline_control_value_milli);
    let reversal_control_delta = i128::from(reversal.requested_control_value_milli)
        - i128::from(reversal.baseline_control_value_milli);
    let control_interventions_are_opposite = active_control_delta != 0
        && reversal_control_delta != 0
        && active_control_delta.signum() != reversal_control_delta.signum();
    if !directions_are_reversed
        || !control_interventions_are_opposite
        || sham.expected_effect_direction != ControlEffectDirectionV1::NoChange
        || sham.requested_control_value_milli != sham.baseline_control_value_milli
    {
        return Err(
            "counterfactual triplet does not declare active, sham and opposite reversal directions"
                .to_owned(),
        );
    }

    let verdict = if active.verdict != ControlEffectVerdictV1::ObservedEffectVerified {
        CounterfactualControlVerdictV1::ActiveEffectNotVerified
    } else if sham.verdict != ControlEffectVerdictV1::ShamNoChangeVerified {
        CounterfactualControlVerdictV1::ShamInvarianceNotVerified
    } else if reversal.verdict != ControlEffectVerdictV1::ObservedEffectVerified {
        CounterfactualControlVerdictV1::ReversalEffectNotVerified
    } else {
        CounterfactualControlVerdictV1::Qualified
    };

    Ok(CounterfactualControlQualificationV1 {
        counterfactual_pair_id: active.counterfactual_pair_id.clone(),
        capability_id: active.capability_id.clone(),
        backend_identity_sha256: active.backend_identity_sha256.clone(),
        baseline_capture_sha256: active.baseline_capture_sha256.clone(),
        baseline_observation_closure_sha256: active.baseline_observation_closure_sha256.clone(),
        baseline_observation_manifest_sha256: active.baseline_observation_manifest_sha256.clone(),
        verdict,
        active: active.clone(),
        sham: sham.clone(),
        reversal: reversal.clone(),
    })
}

fn validate_observation_linkage(
    manifest: &BackendCapabilityManifestV1,
    request: &ControlInterventionRequestV1,
    receipt: &ControlExecutionReceiptV1,
    capability: &BackendControlCapabilityV1,
    observation: &ControlOutputObservationV1,
) -> Result<(), String> {
    for (value, name) in [
        (&observation.counterfactual_pair_id, "observation pair ID"),
        (&observation.capture_observer_id, "capture observer ID"),
        (&observation.event_observer_id, "event observer ID"),
        (&observation.output_producer_id, "output producer ID"),
        (
            &observation.observer_method_version,
            "observer method version",
        ),
        (&observation.target_metric_id, "observed target metric ID"),
    ] {
        require_text(value, name)?;
    }
    for (digest, name) in [
        (&observation.request_sha256, "observation request digest"),
        (
            &observation.baseline_capture_sha256,
            "observed baseline capture digest",
        ),
        (
            &observation.baseline_observation_closure_sha256,
            "observed baseline closure digest",
        ),
        (
            &observation.baseline_observation_manifest_sha256,
            "observed baseline manifest digest",
        ),
        (
            &observation.output_capture_sha256,
            "observed output capture digest",
        ),
        (
            &observation.observation_closure_sha256,
            "observed closure digest",
        ),
        (
            &observation.observation_manifest_sha256,
            "observed manifest digest",
        ),
        (
            &observation.target_metric_contract_sha256,
            "target metric contract digest",
        ),
        (
            &observation.observer_qualification_receipt_sha256,
            "observer qualification receipt digest",
        ),
        (
            &observation.metric_calibration_receipt_sha256,
            "metric calibration receipt digest",
        ),
    ] {
        require_digest(digest, name)?;
    }
    if observation.request_sha256 != request.payload_sha256
        || observation.counterfactual_pair_id != request.payload.counterfactual_pair_id
        || observation.baseline_capture_sha256 != request.payload.baseline_capture_sha256
        || observation.baseline_observation_closure_sha256
            != request.payload.baseline_observation_closure_sha256
        || observation.baseline_observation_manifest_sha256
            != request.payload.baseline_observation_manifest_sha256
        || observation.output_producer_id != manifest.payload.backend_id
    {
        return Err(
            "output observation is not bound to the exact request, baseline or producer".to_owned(),
        );
    }
    if observation.capture_observer_id == observation.output_producer_id
        || observation.event_observer_id == observation.output_producer_id
    {
        return Err(
            "capture/event observer identity must be independent of the output producer".to_owned(),
        );
    }
    if observation.target_metric_id != capability.target_metric_id
        || observation.target_metric_contract_sha256 != capability.target_metric_contract_sha256
    {
        return Err(
            "observation metric ID or metric contract does not match the capability declaration"
                .to_owned(),
        );
    }
    if receipt.payload.output_capture_sha256.as_deref()
        != Some(observation.output_capture_sha256.as_str())
        || receipt.payload.observation_closure_sha256.as_deref()
            != Some(observation.observation_closure_sha256.as_str())
        || receipt.payload.observation_manifest_sha256.as_deref()
            != Some(observation.observation_manifest_sha256.as_str())
    {
        return Err(
            "observation capture/closure/manifest does not match the execution receipt".to_owned(),
        );
    }
    Ok(())
}

fn digest_json<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    Ok(sha256_hex(&bytes))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn require_text(value: &str, name: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{name} must not be empty"))
    } else {
        Ok(())
    }
}

fn require_digest(value: &str, name: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Err(format!(
            "{name} must be a 64-character lowercase SHA-256 digest"
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BACKEND_ID: &str = "synthetic-audio-renderer";
    const BACKEND_DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const BASELINE_CAPTURE: &str =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const OUTPUT_CAPTURE: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const CLOSURE_DIGEST: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const OBSERVER_MANIFEST: &str =
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn render_synthetic_click_track(bpm_milli: i64, duration_seconds: u32) -> Vec<f32> {
        let sample_rate = 48_000_u32;
        let bpm = bpm_milli as f64 / 1_000.0;
        let interval_frames = (f64::from(sample_rate) * 60.0 / bpm).round() as usize;
        let total_frames = sample_rate as usize * duration_seconds as usize;
        let mut audio = vec![0.0_f32; total_frames];
        let pulse_frames = (sample_rate as usize / 100).max(1);

        for onset in (0..total_frames).step_by(interval_frames) {
            for offset in 0..pulse_frames.min(total_frames - onset) {
                let envelope = 1.0 - offset as f32 / pulse_frames as f32;
                audio[onset + offset] = 0.8 * envelope;
            }
        }
        audio
    }

    fn estimate_synthetic_click_tempo_milli(audio: &[f32]) -> i64 {
        let sample_rate = 48_000_usize;
        let refractory_frames = sample_rate / 10;
        let mut onsets = Vec::new();
        let mut last_onset = None;

        for (index, sample) in audio.iter().enumerate() {
            if sample.abs() >= 0.25
                && last_onset
                    .is_none_or(|previous| index.saturating_sub(previous) >= refractory_frames)
            {
                onsets.push(index);
                last_onset = Some(index);
            }
        }

        let intervals = onsets
            .windows(2)
            .map(|pair| (pair[1] - pair[0]) as f64)
            .collect::<Vec<_>>();
        assert!(
            intervals.len() >= 2,
            "synthetic click track must expose at least three onsets"
        );
        let mean_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
        (60_000.0 * sample_rate as f64 / mean_interval).round() as i64
    }

    fn synthetic_capture_digest(audio: &[f32]) -> String {
        let mut bytes = Vec::with_capacity(audio.len() * 4);
        for sample in audio {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        sha256_hex(&bytes)
    }

    fn request_with_baseline(
        manifest: &BackendCapabilityManifestV1,
        condition: CounterfactualConditionV1,
        baseline_capture_sha256: &str,
    ) -> ControlInterventionRequestV1 {
        let mut payload = request(manifest, condition).payload;
        payload.baseline_capture_sha256 = baseline_capture_sha256.to_owned();
        ControlInterventionRequestV1::from_payload(payload).unwrap()
    }

    fn receipt_with_capture(
        request: &ControlInterventionRequestV1,
        capture_sha256: &str,
    ) -> ControlExecutionReceiptV1 {
        ControlExecutionReceiptV1::from_payload(ControlExecutionReceiptPayloadV1 {
            schema_version: CONTROL_EFFECT_SCHEMA_VERSION_V1,
            request_sha256: request.payload_sha256.clone(),
            backend_id: BACKEND_ID.to_owned(),
            backend_identity_sha256: BACKEND_DIGEST.to_owned(),
            execution_status: ControlExecutionStatusV1::Applied,
            effective_control_value_milli: Some(request.payload.requested_control_value_milli),
            output_capture_sha256: Some(capture_sha256.to_owned()),
            observation_closure_sha256: Some(CLOSURE_DIGEST.to_owned()),
            observation_manifest_sha256: Some(OBSERVER_MANIFEST.to_owned()),
        })
        .unwrap()
    }

    fn observation_from_click_track(
        request: &ControlInterventionRequestV1,
        baseline_capture_sha256: &str,
        output_audio: &[f32],
        baseline_tempo_milli: i64,
    ) -> ControlOutputObservationV1 {
        let output_tempo_milli = estimate_synthetic_click_tempo_milli(output_audio);
        let mut result = observation(
            request,
            output_tempo_milli - baseline_tempo_milli,
            25,
            0,
            ObservationClosureStateV1::StructurallyEligible,
        );
        result.baseline_capture_sha256 = baseline_capture_sha256.to_owned();
        result.output_capture_sha256 = synthetic_capture_digest(output_audio);
        result.baseline_metric_value_milli = baseline_tempo_milli;
        result.output_metric_value_milli = output_tempo_milli;
        result.baseline_metric_uncertainty_milli = 12;
        result.output_metric_uncertainty_milli = 13;
        result
    }

    fn manifest() -> BackendCapabilityManifestV1 {
        BackendCapabilityManifestV1::from_payload(BackendCapabilityManifestPayloadV1 {
            schema_version: CONTROL_EFFECT_SCHEMA_VERSION_V1,
            backend_id: BACKEND_ID.to_owned(),
            backend_version: "synthetic-renderer-v1".to_owned(),
            adapter_version: "adapter-v1".to_owned(),
            backend_identity_sha256: BACKEND_DIGEST.to_owned(),
            capabilities: vec![BackendControlCapabilityV1 {
                capability_id: "tempo-control-v1".to_owned(),
                control: MusicalControlV1::Tempo,
                support: ControlSupportLevelV1::BestEffort,
                minimum_value_milli: 40_000,
                maximum_value_milli: 240_000,
                unit: "beats_per_minute_milli".to_owned(),
                target_metric_id: "observed_tempo_bpm_milli".to_owned(),
                target_metric_contract_sha256: "1".repeat(64),
                minimum_effect_milli: 500,
                sham_equivalence_tolerance_milli: 75,
                maximum_collateral_drift_milli: 250,
                source_references: vec!["synthetic-control-fixture-v1".to_owned()],
            }],
        })
        .unwrap()
    }

    fn request(
        manifest: &BackendCapabilityManifestV1,
        condition: CounterfactualConditionV1,
    ) -> ControlInterventionRequestV1 {
        let (requested, direction) = match condition {
            CounterfactualConditionV1::Active => (120_000, ControlEffectDirectionV1::Increase),
            CounterfactualConditionV1::Sham => (100_000, ControlEffectDirectionV1::NoChange),
            CounterfactualConditionV1::Reversal => (80_000, ControlEffectDirectionV1::Decrease),
        };
        ControlInterventionRequestV1::from_payload(ControlInterventionRequestPayloadV1 {
            schema_version: CONTROL_EFFECT_SCHEMA_VERSION_V1,
            request_id: format!("request-{condition:?}"),
            counterfactual_pair_id: "paired-tempo-001".to_owned(),
            condition,
            manifest_sha256: manifest.payload_sha256.clone(),
            backend_id: BACKEND_ID.to_owned(),
            backend_identity_sha256: BACKEND_DIGEST.to_owned(),
            capability_id: "tempo-control-v1".to_owned(),
            baseline_control_value_milli: 100_000,
            requested_control_value_milli: requested,
            expected_effect_direction: direction,
            baseline_capture_sha256: BASELINE_CAPTURE.to_owned(),
            baseline_observation_closure_sha256: BASELINE_CLOSURE.to_owned(),
            baseline_observation_manifest_sha256: BASELINE_MANIFEST.to_owned(),
        })
        .unwrap()
    }

    fn receipt(
        request: &ControlInterventionRequestV1,
        status: ControlExecutionStatusV1,
        effective: Option<i64>,
    ) -> ControlExecutionReceiptV1 {
        ControlExecutionReceiptV1::from_payload(ControlExecutionReceiptPayloadV1 {
            schema_version: CONTROL_EFFECT_SCHEMA_VERSION_V1,
            request_sha256: request.payload_sha256.clone(),
            backend_id: BACKEND_ID.to_owned(),
            backend_identity_sha256: BACKEND_DIGEST.to_owned(),
            execution_status: status,
            effective_control_value_milli: effective,
            output_capture_sha256: Some(OUTPUT_CAPTURE.to_owned()),
            observation_closure_sha256: Some(CLOSURE_DIGEST.to_owned()),
            observation_manifest_sha256: Some(OBSERVER_MANIFEST.to_owned()),
        })
        .unwrap()
    }

    fn observation(
        request: &ControlInterventionRequestV1,
        effect: i64,
        uncertainty: u64,
        collateral: u64,
        closure_state: ObservationClosureStateV1,
    ) -> ControlOutputObservationV1 {
        ControlOutputObservationV1 {
            request_sha256: request.payload_sha256.clone(),
            counterfactual_pair_id: request.payload.counterfactual_pair_id.clone(),
            capture_observer_id: "independent-loopback-recorder".to_owned(),
            event_observer_id: "independent-event-detector".to_owned(),
            output_producer_id: BACKEND_ID.to_owned(),
            baseline_capture_sha256: request.payload.baseline_capture_sha256.clone(),
            baseline_observation_closure_sha256: request
                .payload
                .baseline_observation_closure_sha256
                .clone(),
            baseline_observation_manifest_sha256: request
                .payload
                .baseline_observation_manifest_sha256
                .clone(),
            baseline_closure_state: ObservationClosureStateV1::StructurallyEligible,
            output_capture_sha256: OUTPUT_CAPTURE.to_owned(),
            observation_closure_sha256: CLOSURE_DIGEST.to_owned(),
            observation_manifest_sha256: OBSERVER_MANIFEST.to_owned(),
            closure_state,
            observer_method_version: "synthetic-onset-metric-v1".to_owned(),
            target_metric_id: "observed_tempo_bpm_milli".to_owned(),
            target_metric_contract_sha256: "1".repeat(64),
            observer_qualification_receipt_sha256: "2".repeat(64),
            metric_calibration_receipt_sha256: "3".repeat(64),
            baseline_metric_value_milli: 100_000,
            output_metric_value_milli: 100_000_i64.checked_add(effect).unwrap(),
            baseline_metric_uncertainty_milli: uncertainty / 2,
            output_metric_uncertainty_milli: uncertainty - (uncertainty / 2),
            collateral_drift_milli: collateral,
            collateral_uncertainty_milli: 0,
        }
    }

    #[test]
    fn independently_observed_active_effect_is_verified() {
        let manifest = manifest();
        let request = request(&manifest, CounterfactualConditionV1::Active);
        let receipt = receipt(
            &request,
            ControlExecutionStatusV1::Applied,
            Some(request.payload.requested_control_value_milli),
        );
        let observation = observation(
            &request,
            650,
            50,
            30,
            ObservationClosureStateV1::StructurallyEligible,
        );
        let finding =
            evaluate_control_effect_v1(&manifest, &request, &receipt, Some(&observation)).unwrap();
        assert_eq!(
            finding.verdict,
            ControlEffectVerdictV1::ObservedEffectVerified
        );
        assert_eq!(finding.observed_target_effect_milli, Some(650));
        assert_eq!(finding.target_effect_uncertainty_milli, Some(50));
        assert_eq!(finding.effective_value_matches_request, Some(true));
        assert_eq!(finding.evidence_refs.len(), 9);
    }

    #[test]
    fn api_acceptance_without_independent_observation_is_unobserved() {
        let manifest = manifest();
        let request = request(&manifest, CounterfactualConditionV1::Active);
        let receipt = receipt(&request, ControlExecutionStatusV1::Accepted, None);
        let finding = evaluate_control_effect_v1(&manifest, &request, &receipt, None).unwrap();
        assert_eq!(finding.verdict, ControlEffectVerdictV1::Unobserved);
    }

    #[test]
    fn ignored_control_with_audio_change_is_not_attributed_to_the_control() {
        let manifest = manifest();
        let request = request(&manifest, CounterfactualConditionV1::Active);
        let receipt = receipt(&request, ControlExecutionStatusV1::Ignored, None);
        let observation = observation(
            &request,
            900,
            50,
            10,
            ObservationClosureStateV1::StructurallyEligible,
        );
        let finding =
            evaluate_control_effect_v1(&manifest, &request, &receipt, Some(&observation)).unwrap();
        assert_eq!(
            finding.verdict,
            ControlEffectVerdictV1::ObservedEffectUnattributed
        );
    }

    #[test]
    fn sham_detects_effect_leakage() {
        let manifest = manifest();
        let request = request(&manifest, CounterfactualConditionV1::Sham);
        let receipt = receipt(
            &request,
            ControlExecutionStatusV1::Applied,
            Some(request.payload.requested_control_value_milli),
        );
        let observation = observation(
            &request,
            120,
            10,
            5,
            ObservationClosureStateV1::StructurallyEligible,
        );
        let finding =
            evaluate_control_effect_v1(&manifest, &request, &receipt, Some(&observation)).unwrap();
        assert_eq!(finding.verdict, ControlEffectVerdictV1::ShamLeakage);
    }

    #[test]
    fn incomplete_observation_window_never_becomes_a_negative_or_positive_label() {
        let manifest = manifest();
        let request = request(&manifest, CounterfactualConditionV1::Active);
        let receipt = receipt(
            &request,
            ControlExecutionStatusV1::Applied,
            Some(request.payload.requested_control_value_milli),
        );
        let observation = observation(&request, 900, 10, 0, ObservationClosureStateV1::Incomplete);
        let finding =
            evaluate_control_effect_v1(&manifest, &request, &receipt, Some(&observation)).unwrap();
        assert_eq!(finding.verdict, ControlEffectVerdictV1::Unobserved);
    }

    #[test]
    fn observer_identity_must_be_independent_of_renderer() {
        let manifest = manifest();
        let request = request(&manifest, CounterfactualConditionV1::Active);
        let receipt = receipt(
            &request,
            ControlExecutionStatusV1::Applied,
            Some(request.payload.requested_control_value_milli),
        );
        let mut observation = observation(
            &request,
            700,
            20,
            0,
            ObservationClosureStateV1::StructurallyEligible,
        );
        observation.event_observer_id = BACKEND_ID.to_owned();
        let error = evaluate_control_effect_v1(&manifest, &request, &receipt, Some(&observation))
            .unwrap_err();
        assert!(error.contains("independent"));
    }

    #[test]
    fn receipt_capture_and_manifest_must_match_observer_output() {
        let manifest = manifest();
        let request = request(&manifest, CounterfactualConditionV1::Active);
        let receipt = receipt(
            &request,
            ControlExecutionStatusV1::Applied,
            Some(request.payload.requested_control_value_milli),
        );
        let mut observation = observation(
            &request,
            700,
            20,
            0,
            ObservationClosureStateV1::StructurallyEligible,
        );
        observation.output_capture_sha256 = "9".repeat(64);
        let error = evaluate_control_effect_v1(&manifest, &request, &receipt, Some(&observation))
            .unwrap_err();
        assert!(error.contains("does not match the execution receipt"));
    }

    #[test]
    fn collateral_drift_can_fail_an_otherwise_correct_control() {
        let manifest = manifest();
        let request = request(&manifest, CounterfactualConditionV1::Active);
        let receipt = receipt(
            &request,
            ControlExecutionStatusV1::Applied,
            Some(request.payload.requested_control_value_milli),
        );
        let observation = observation(
            &request,
            700,
            20,
            400,
            ObservationClosureStateV1::StructurallyEligible,
        );
        let finding =
            evaluate_control_effect_v1(&manifest, &request, &receipt, Some(&observation)).unwrap();
        assert_eq!(
            finding.verdict,
            ControlEffectVerdictV1::CollateralDriftExceeded
        );
    }

    #[test]
    fn tampered_manifest_payload_is_rejected() {
        let original_manifest = manifest();
        let request = request(&original_manifest, CounterfactualConditionV1::Active);
        let mut manifest = original_manifest;
        manifest.payload.backend_version = "different-version".to_owned();
        let receipt = receipt(
            &request,
            ControlExecutionStatusV1::Applied,
            Some(request.payload.requested_control_value_milli),
        );
        let error = evaluate_control_effect_v1(&manifest, &request, &receipt, None).unwrap_err();
        assert!(error.contains("manifest digest does not match"));
    }

    #[test]
    fn hard_capability_cannot_silently_report_a_different_effective_value() {
        let mut manifest = manifest();
        manifest.payload.capabilities[0].support = ControlSupportLevelV1::Hard;
        manifest.payload_sha256 = digest_json(&manifest.payload).unwrap();
        let request = request(&manifest, CounterfactualConditionV1::Active);
        let receipt = receipt(
            &request,
            ControlExecutionStatusV1::Applied,
            Some(request.payload.requested_control_value_milli - 5_000),
        );
        let finding = evaluate_control_effect_v1(&manifest, &request, &receipt, None).unwrap();
        assert_eq!(
            finding.verdict,
            ControlEffectVerdictV1::HardControlValueMismatch
        );
    }

    #[test]
    fn contradictory_observed_effect_is_not_counted_as_success() {
        let manifest = manifest();
        let request = request(&manifest, CounterfactualConditionV1::Active);
        let receipt = receipt(
            &request,
            ControlExecutionStatusV1::Applied,
            Some(request.payload.requested_control_value_milli),
        );
        let observation = observation(
            &request,
            -800,
            50,
            0,
            ObservationClosureStateV1::StructurallyEligible,
        );
        let finding =
            evaluate_control_effect_v1(&manifest, &request, &receipt, Some(&observation)).unwrap();
        assert_eq!(
            finding.verdict,
            ControlEffectVerdictV1::ObservedEffectContradictory
        );
    }

    #[test]
    fn active_sham_reversal_triplet_qualifies_only_with_directional_reversal() {
        let manifest = manifest();
        let active_request = request(&manifest, CounterfactualConditionV1::Active);
        let sham_request = request(&manifest, CounterfactualConditionV1::Sham);
        let reversal_request = request(&manifest, CounterfactualConditionV1::Reversal);

        let active_receipt = receipt(
            &active_request,
            ControlExecutionStatusV1::Applied,
            Some(active_request.payload.requested_control_value_milli),
        );
        let sham_receipt = receipt(
            &sham_request,
            ControlExecutionStatusV1::Applied,
            Some(sham_request.payload.requested_control_value_milli),
        );
        let reversal_receipt = receipt(
            &reversal_request,
            ControlExecutionStatusV1::Applied,
            Some(reversal_request.payload.requested_control_value_milli),
        );

        let active_observation = observation(
            &active_request,
            700,
            30,
            10,
            ObservationClosureStateV1::StructurallyEligible,
        );
        let sham_observation = observation(
            &sham_request,
            10,
            10,
            5,
            ObservationClosureStateV1::StructurallyEligible,
        );
        let reversal_observation = observation(
            &reversal_request,
            -680,
            40,
            15,
            ObservationClosureStateV1::StructurallyEligible,
        );

        let active = evaluate_control_effect_v1(
            &manifest,
            &active_request,
            &active_receipt,
            Some(&active_observation),
        )
        .unwrap();
        let sham = evaluate_control_effect_v1(
            &manifest,
            &sham_request,
            &sham_receipt,
            Some(&sham_observation),
        )
        .unwrap();
        let reversal = evaluate_control_effect_v1(
            &manifest,
            &reversal_request,
            &reversal_receipt,
            Some(&reversal_observation),
        )
        .unwrap();

        let qualification =
            qualify_counterfactual_control_triplet_v1(&active, &sham, &reversal).unwrap();
        assert_eq!(
            qualification.verdict,
            CounterfactualControlVerdictV1::Qualified
        );
    }

    #[test]
    fn active_effect_without_opposite_reversal_does_not_qualify_control() {
        let manifest = manifest();
        let make_finding = |condition, effect, direction| {
            let mut finding = {
                let request = request(&manifest, condition);
                let receipt = receipt(
                    &request,
                    ControlExecutionStatusV1::Applied,
                    Some(request.payload.requested_control_value_milli),
                );
                let observation = observation(
                    &request,
                    effect,
                    10,
                    0,
                    ObservationClosureStateV1::StructurallyEligible,
                );
                evaluate_control_effect_v1(&manifest, &request, &receipt, Some(&observation))
                    .unwrap()
            };
            finding.expected_effect_direction = direction;
            finding
        };
        let active = make_finding(
            CounterfactualConditionV1::Active,
            700,
            ControlEffectDirectionV1::Increase,
        );
        let sham = make_finding(
            CounterfactualConditionV1::Sham,
            0,
            ControlEffectDirectionV1::NoChange,
        );
        let reversal = make_finding(
            CounterfactualConditionV1::Reversal,
            650,
            ControlEffectDirectionV1::Decrease,
        );
        let qualification =
            qualify_counterfactual_control_triplet_v1(&active, &sham, &reversal).unwrap();
        assert_eq!(
            qualification.verdict,
            CounterfactualControlVerdictV1::ReversalEffectNotVerified
        );
    }

    #[test]
    fn counterfactual_triplet_rejects_different_baseline_capture() {
        let manifest = manifest();
        let make_finding = |condition, effect| {
            let request = request(&manifest, condition);
            let receipt = receipt(
                &request,
                ControlExecutionStatusV1::Applied,
                Some(request.payload.requested_control_value_milli),
            );
            let observation = observation(
                &request,
                effect,
                10,
                0,
                ObservationClosureStateV1::StructurallyEligible,
            );
            evaluate_control_effect_v1(&manifest, &request, &receipt, Some(&observation)).unwrap()
        };
        let active = make_finding(CounterfactualConditionV1::Active, 700);
        let sham = make_finding(CounterfactualConditionV1::Sham, 0);
        let mut reversal = make_finding(CounterfactualConditionV1::Reversal, -700);
        reversal.baseline_capture_sha256 = "9".repeat(64);
        let error =
            qualify_counterfactual_control_triplet_v1(&active, &sham, &reversal).unwrap_err();
        assert!(error.contains("baseline"));
    }

    #[test]
    fn synthetic_waveform_active_sham_reversal_qualifies_from_measured_output() {
        let manifest = manifest();
        let baseline_audio = render_synthetic_click_track(100_000, 5);
        let baseline_digest = synthetic_capture_digest(&baseline_audio);
        let baseline_tempo = estimate_synthetic_click_tempo_milli(&baseline_audio);
        assert!((baseline_tempo - 100_000).abs() <= 1);

        let mut findings = Vec::new();
        for (condition, output_tempo) in [
            (CounterfactualConditionV1::Active, 120_000),
            (CounterfactualConditionV1::Sham, 100_000),
            (CounterfactualConditionV1::Reversal, 80_000),
        ] {
            let request = request_with_baseline(&manifest, condition, &baseline_digest);
            let output_audio = render_synthetic_click_track(output_tempo, 5);
            let receipt = receipt_with_capture(&request, &synthetic_capture_digest(&output_audio));
            let observation = observation_from_click_track(
                &request,
                &baseline_digest,
                &output_audio,
                baseline_tempo,
            );
            findings.push(
                evaluate_control_effect_v1(&manifest, &request, &receipt, Some(&observation))
                    .unwrap(),
            );
        }

        let qualification =
            qualify_counterfactual_control_triplet_v1(&findings[0], &findings[1], &findings[2])
                .unwrap();
        assert_eq!(
            qualification.verdict,
            CounterfactualControlVerdictV1::Qualified
        );
        assert!(findings[0].observed_target_effect_milli.unwrap() > 19_000);
        assert!(findings[1].observed_target_effect_milli.unwrap().abs() <= 1);
        assert!(findings[2].observed_target_effect_milli.unwrap() < -19_000);
    }

    #[test]
    fn synthetic_waveform_sham_detects_session_carryover() {
        let manifest = manifest();
        let baseline_audio = render_synthetic_click_track(100_000, 5);
        let baseline_digest = synthetic_capture_digest(&baseline_audio);
        let baseline_tempo = estimate_synthetic_click_tempo_milli(&baseline_audio);

        let active_request = request_with_baseline(
            &manifest,
            CounterfactualConditionV1::Active,
            &baseline_digest,
        );
        let active_audio = render_synthetic_click_track(120_000, 5);
        let active_receipt =
            receipt_with_capture(&active_request, &synthetic_capture_digest(&active_audio));
        let active_observation = observation_from_click_track(
            &active_request,
            &baseline_digest,
            &active_audio,
            baseline_tempo,
        );
        let active = evaluate_control_effect_v1(
            &manifest,
            &active_request,
            &active_receipt,
            Some(&active_observation),
        )
        .unwrap();

        // The sham request preserves 100 BPM, but the simulated renderer
        // retains the previous active setting, exposing session carryover.
        let sham_request =
            request_with_baseline(&manifest, CounterfactualConditionV1::Sham, &baseline_digest);
        let carried_audio = render_synthetic_click_track(120_000, 5);
        let sham_receipt =
            receipt_with_capture(&sham_request, &synthetic_capture_digest(&carried_audio));
        let sham_observation = observation_from_click_track(
            &sham_request,
            &baseline_digest,
            &carried_audio,
            baseline_tempo,
        );
        let sham = evaluate_control_effect_v1(
            &manifest,
            &sham_request,
            &sham_receipt,
            Some(&sham_observation),
        )
        .unwrap();
        assert_eq!(sham.verdict, ControlEffectVerdictV1::ShamLeakage);

        let reversal_request = request_with_baseline(
            &manifest,
            CounterfactualConditionV1::Reversal,
            &baseline_digest,
        );
        let reversal_audio = render_synthetic_click_track(80_000, 5);
        let reversal_receipt = receipt_with_capture(
            &reversal_request,
            &synthetic_capture_digest(&reversal_audio),
        );
        let reversal_observation = observation_from_click_track(
            &reversal_request,
            &baseline_digest,
            &reversal_audio,
            baseline_tempo,
        );
        let reversal = evaluate_control_effect_v1(
            &manifest,
            &reversal_request,
            &reversal_receipt,
            Some(&reversal_observation),
        )
        .unwrap();

        let qualification =
            qualify_counterfactual_control_triplet_v1(&active, &sham, &reversal).unwrap();
        assert_eq!(
            qualification.verdict,
            CounterfactualControlVerdictV1::ShamInvarianceNotVerified
        );
    }

    #[test]
    fn baseline_and_output_metric_uncertainties_are_added_conservatively() {
        let manifest = manifest();
        let request = request(&manifest, CounterfactualConditionV1::Active);
        let receipt = receipt(
            &request,
            ControlExecutionStatusV1::Applied,
            Some(request.payload.requested_control_value_milli),
        );
        let mut observation = observation(
            &request,
            550,
            0,
            0,
            ObservationClosureStateV1::StructurallyEligible,
        );
        observation.baseline_metric_uncertainty_milli = 300;
        observation.output_metric_uncertainty_milli = 300;
        let finding =
            evaluate_control_effect_v1(&manifest, &request, &receipt, Some(&observation)).unwrap();
        assert_eq!(
            finding.verdict,
            ControlEffectVerdictV1::ObservedEffectInconclusive
        );
        assert_eq!(finding.observed_target_effect_milli, Some(550));
        assert_eq!(finding.target_effect_uncertainty_milli, Some(600));
    }

    #[test]
    fn output_metric_contract_mismatch_is_rejected() {
        let manifest = manifest();
        let request = request(&manifest, CounterfactualConditionV1::Active);
        let receipt = receipt(
            &request,
            ControlExecutionStatusV1::Applied,
            Some(request.payload.requested_control_value_milli),
        );
        let mut observation = observation(
            &request,
            700,
            20,
            0,
            ObservationClosureStateV1::StructurallyEligible,
        );
        observation.target_metric_contract_sha256 = "9".repeat(64);
        let error = evaluate_control_effect_v1(&manifest, &request, &receipt, Some(&observation))
            .unwrap_err();
        assert!(error.contains("metric ID or metric contract"));
    }

    #[test]
    fn missing_metric_calibration_receipt_is_rejected() {
        let manifest = manifest();
        let request = request(&manifest, CounterfactualConditionV1::Active);
        let receipt = receipt(
            &request,
            ControlExecutionStatusV1::Applied,
            Some(request.payload.requested_control_value_milli),
        );
        let mut observation = observation(
            &request,
            700,
            20,
            0,
            ObservationClosureStateV1::StructurallyEligible,
        );
        observation.metric_calibration_receipt_sha256.clear();
        let error = evaluate_control_effect_v1(&manifest, &request, &receipt, Some(&observation))
            .unwrap_err();
        assert!(error.contains("metric calibration receipt digest"));
    }

    #[test]
    fn incomplete_baseline_capture_cannot_verify_output_effect() {
        let manifest = manifest();
        let request = request(&manifest, CounterfactualConditionV1::Active);
        let receipt = receipt(
            &request,
            ControlExecutionStatusV1::Applied,
            Some(request.payload.requested_control_value_milli),
        );
        let mut observation = observation(
            &request,
            700,
            20,
            0,
            ObservationClosureStateV1::StructurallyEligible,
        );
        observation.baseline_closure_state = ObservationClosureStateV1::Incomplete;
        let finding =
            evaluate_control_effect_v1(&manifest, &request, &receipt, Some(&observation)).unwrap();
        assert_eq!(finding.verdict, ControlEffectVerdictV1::Unobserved);
        assert_eq!(finding.observed_target_effect_milli, Some(700));
    }

    #[test]
    fn baseline_observation_closure_mismatch_is_rejected() {
        let manifest = manifest();
        let request = request(&manifest, CounterfactualConditionV1::Active);
        let receipt = receipt(
            &request,
            ControlExecutionStatusV1::Applied,
            Some(request.payload.requested_control_value_milli),
        );
        let mut observation = observation(
            &request,
            700,
            20,
            0,
            ObservationClosureStateV1::StructurallyEligible,
        );
        observation.baseline_observation_closure_sha256 = "9".repeat(64);
        let error = evaluate_control_effect_v1(&manifest, &request, &receipt, Some(&observation))
            .unwrap_err();
        assert!(error.contains("not bound to the exact request"));
    }

    #[test]
    fn counterfactual_triplet_rejects_different_baseline_manifest() {
        let manifest = manifest();
        let make_finding = |condition, effect| {
            let request = request(&manifest, condition);
            let receipt = receipt(
                &request,
                ControlExecutionStatusV1::Applied,
                Some(request.payload.requested_control_value_milli),
            );
            let observation = observation(
                &request,
                effect,
                10,
                0,
                ObservationClosureStateV1::StructurallyEligible,
            );
            evaluate_control_effect_v1(&manifest, &request, &receipt, Some(&observation)).unwrap()
        };
        let active = make_finding(CounterfactualConditionV1::Active, 700);
        let sham = make_finding(CounterfactualConditionV1::Sham, 0);
        let mut reversal = make_finding(CounterfactualConditionV1::Reversal, -700);
        reversal.baseline_observation_manifest_sha256 = "9".repeat(64);
        let error =
            qualify_counterfactual_control_triplet_v1(&active, &sham, &reversal).unwrap_err();
        assert!(error.contains("baseline"));
    }

    #[test]
    fn control_request_requires_baseline_closure_and_manifest_receipts() {
        let manifest = manifest();
        let mut payload = request(&manifest, CounterfactualConditionV1::Active).payload;
        payload.baseline_observation_manifest_sha256.clear();
        let error = ControlInterventionRequestV1::from_payload(payload).unwrap_err();
        assert!(error.contains("baseline observation manifest digest"));
    }
}
