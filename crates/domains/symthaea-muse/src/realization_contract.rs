// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Backend-neutral contracts for capability-qualified musical realization.
//!
//! A provider's documented control, accepted request, or echoed configuration
//! is not proof that the rendered output changed. This module separates
//! capability declarations, admission, execution receipts, and independent
//! observation so a caller cannot silently promote best-effort steering into
//! an exact musical guarantee.
//!
//! This is a contract and deterministic conformance kernel. It does not call
//! a commercial backend or claim that a real renderer has been integrated.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const REALIZATION_CONTRACT_SCHEMA_VERSION_V1: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealizationTaskV1 {
    FullGeneration,
    Continuation,
    LocalizedEdit,
    ScoreMidiRealization,
    LiveStreaming,
    AudioEditing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactModalityV1 {
    MusicalIntent,
    Score,
    Midi,
    Stems,
    Audio,
    StreamEvents,
    Analysis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlGuaranteeV1 {
    Hard,
    BestEffort,
    Advisory,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityEvidenceLevelV1 {
    ProviderDocumentation,
    InterfaceAcknowledgement,
    IndependentConformance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlCapabilityV1 {
    pub control_id: String,
    pub semantics: String,
    pub guarantee: ControlGuaranteeV1,
    pub evidence_level: CapabilityEvidenceLevelV1,
    pub conformance_test_id: Option<String>,
    pub evidence_references: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClockContractV1 {
    pub clock_domain: String,
    pub sample_rate_hz: u32,
    pub buffer_frames: Option<u32>,
    pub measured_p50_latency_us: Option<u64>,
    pub measured_p95_latency_us: Option<u64>,
    pub measured_p99_latency_us: Option<u64>,
    pub measurement_evidence_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendCapabilityManifestPayloadV1 {
    pub schema_version: u32,
    pub backend_id: String,
    pub renderer_version: String,
    pub adapter_version: String,
    pub environment_sha256: String,
    pub tasks: Vec<RealizationTaskV1>,
    pub input_modalities: Vec<ArtifactModalityV1>,
    pub output_modalities: Vec<ArtifactModalityV1>,
    pub controls: Vec<ControlCapabilityV1>,
    pub seed_supported: bool,
    pub deterministic_for_fixed_seed: bool,
    pub stateful_session: bool,
    pub reset_supported: bool,
    pub version_pinned: bool,
    pub clock_contract: Option<ClockContractV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendCapabilityManifestV1 {
    pub payload: BackendCapabilityManifestPayloadV1,
    pub payload_sha256: String,
}

impl BackendCapabilityManifestV1 {
    pub fn from_payload(payload: BackendCapabilityManifestPayloadV1) -> Result<Self, String> {
        validate_manifest_payload_v1(&payload)?;
        let bytes = serde_json::to_vec(&payload).map_err(|error| error.to_string())?;
        Ok(Self {
            payload,
            payload_sha256: sha256_hex(&bytes),
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_manifest_payload_v1(&self.payload)?;
        let bytes = serde_json::to_vec(&self.payload).map_err(|error| error.to_string())?;
        if self.payload_sha256 != sha256_hex(&bytes) {
            return Err("capability manifest digest does not match its payload".to_owned());
        }
        Ok(())
    }

    pub fn control(&self, control_id: &str) -> Option<&ControlCapabilityV1> {
        self.payload
            .controls
            .iter()
            .find(|control| control.control_id == control_id)
    }
}

fn validate_manifest_payload_v1(payload: &BackendCapabilityManifestPayloadV1) -> Result<(), String> {
    if payload.schema_version != REALIZATION_CONTRACT_SCHEMA_VERSION_V1 {
        return Err("unsupported realization capability schema version".to_owned());
    }
    for (value, label) in [
        (&payload.backend_id, "backend ID"),
        (&payload.renderer_version, "renderer version"),
        (&payload.adapter_version, "adapter version"),
    ] {
        if value.trim().is_empty() {
            return Err(format!("{label} must not be empty"));
        }
    }
    require_sha256(&payload.environment_sha256, "backend environment digest")?;
    if payload.tasks.is_empty() || payload.input_modalities.is_empty() || payload.output_modalities.is_empty() {
        return Err("backend must declare at least one task, input modality and output modality".to_owned());
    }
    ensure_unique(&payload.tasks, "task")?;
    ensure_unique(&payload.input_modalities, "input modality")?;
    ensure_unique(&payload.output_modalities, "output modality")?;
    let mut control_ids = BTreeSet::new();
    for control in &payload.controls {
        if control.control_id.trim().is_empty() || control.semantics.trim().is_empty() {
            return Err("control ID and semantics must not be empty".to_owned());
        }
        if !control_ids.insert(control.control_id.as_str()) {
            return Err(format!("duplicate control ID: {}", control.control_id));
        }
        if control.evidence_references.is_empty()
            || control.evidence_references.iter().any(|reference| reference.trim().is_empty())
        {
            return Err(format!("control {} has no evidence references", control.control_id));
        }
        match control.guarantee {
            ControlGuaranteeV1::Hard => {
                if control.evidence_level != CapabilityEvidenceLevelV1::IndependentConformance
                    || control.conformance_test_id.as_deref().unwrap_or("").trim().is_empty()
                {
                    return Err(format!(
                        "hard control {} requires independent conformance evidence and a conformance test ID",
                        control.control_id
                    ));
                }
            }
            ControlGuaranteeV1::Unsupported => {
                if control.conformance_test_id.is_some() {
                    return Err(format!(
                        "unsupported control {} must not claim a passing conformance test",
                        control.control_id
                    ));
                }
            }
            ControlGuaranteeV1::BestEffort | ControlGuaranteeV1::Advisory => {}
        }
    }
    if payload.reset_supported && !payload.stateful_session {
        return Err("stateless backend cannot declare a session reset operation".to_owned());
    }
    if payload.deterministic_for_fixed_seed && !payload.seed_supported {
        return Err("fixed-seed determinism requires seed support".to_owned());
    }
    if let Some(clock) = &payload.clock_contract {
        if clock.clock_domain.trim().is_empty() || clock.sample_rate_hz == 0 {
            return Err("clock contract requires a clock domain and non-zero sample rate".to_owned());
        }
        for digest in clock.measurement_evidence_sha256.iter() {
            require_sha256(digest, "clock measurement evidence digest")?;
        }
        if let (Some(p50), Some(p95)) = (clock.measured_p50_latency_us, clock.measured_p95_latency_us) {
            if p50 > p95 {
                return Err("clock p50 latency cannot exceed p95 latency".to_owned());
            }
        }
        if let (Some(p95), Some(p99)) = (clock.measured_p95_latency_us, clock.measured_p99_latency_us) {
            if p95 > p99 {
                return Err("clock p95 latency cannot exceed p99 latency".to_owned());
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuaranteeRequirementV1 {
    Exact,
    BestEffort,
    Advisory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpectedEffectDirectionV1 {
    Increase,
    Decrease,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlEffectContractV1 {
    pub target_metric_id: String,
    pub expected_delta_min: f64,
    pub expected_delta_max: f64,
    pub sham_abs_delta_max: f64,
    pub require_direction_reversal: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestedControlV1 {
    pub control_id: String,
    pub requested_value: f64,
    pub guarantee_requirement: GuaranteeRequirementV1,
    pub expected_direction: ExpectedEffectDirectionV1,
    pub effect_contract: ControlEffectContractV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicalRealizationRequestPayloadV1 {
    pub schema_version: u32,
    pub request_id: String,
    pub task: RealizationTaskV1,
    pub source_artifact_sha256: String,
    pub input_modality: ArtifactModalityV1,
    pub required_output_modality: ArtifactModalityV1,
    pub controls: Vec<RequestedControlV1>,
    pub target_region_id: Option<String>,
    pub protected_region_ids: Vec<String>,
    pub candidate_budget: u16,
    pub consent_purpose_reference: String,
    pub verification_policy_id: String,
    pub require_independent_capture: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicalRealizationRequestV1 {
    pub payload: MusicalRealizationRequestPayloadV1,
    pub payload_sha256: String,
}

impl MusicalRealizationRequestV1 {
    pub fn from_payload(payload: MusicalRealizationRequestPayloadV1) -> Result<Self, String> {
        validate_request_payload_v1(&payload)?;
        let bytes = serde_json::to_vec(&payload).map_err(|error| error.to_string())?;
        Ok(Self {
            payload,
            payload_sha256: sha256_hex(&bytes),
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_request_payload_v1(&self.payload)?;
        let bytes = serde_json::to_vec(&self.payload).map_err(|error| error.to_string())?;
        if self.payload_sha256 != sha256_hex(&bytes) {
            return Err("realization request digest does not match its payload".to_owned());
        }
        Ok(())
    }
}

fn validate_request_payload_v1(payload: &MusicalRealizationRequestPayloadV1) -> Result<(), String> {
    if payload.schema_version != REALIZATION_CONTRACT_SCHEMA_VERSION_V1 {
        return Err("unsupported realization request schema version".to_owned());
    }
    for (value, label) in [
        (&payload.request_id, "request ID"),
        (&payload.consent_purpose_reference, "consent/purpose reference"),
        (&payload.verification_policy_id, "verification policy ID"),
    ] {
        if value.trim().is_empty() {
            return Err(format!("{label} must not be empty"));
        }
    }
    require_sha256(&payload.source_artifact_sha256, "source artifact digest")?;
    if payload.candidate_budget == 0 {
        return Err("candidate budget must be non-zero".to_owned());
    }
    ensure_unique(&payload.protected_region_ids, "protected region ID")?;
    for region in &payload.protected_region_ids {
        if region.trim().is_empty() {
            return Err("protected region IDs must not be empty".to_owned());
        }
    }
    let mut control_ids = BTreeSet::new();
    for control in &payload.controls {
        if control.control_id.trim().is_empty() || !control_ids.insert(control.control_id.as_str()) {
            return Err("requested control IDs must be non-empty and unique".to_owned());
        }
        if !control.requested_value.is_finite() {
            return Err(format!("requested value for {} must be finite", control.control_id));
        }
        let effect = &control.effect_contract;
        if effect.target_metric_id.trim().is_empty()
            || !effect.expected_delta_min.is_finite()
            || !effect.expected_delta_max.is_finite()
            || !effect.sham_abs_delta_max.is_finite()
            || effect.expected_delta_min > effect.expected_delta_max
            || effect.sham_abs_delta_max < 0.0
        {
            return Err(format!("invalid independent effect contract for {}", control.control_id));
        }
        match control.expected_direction {
            ExpectedEffectDirectionV1::Increase if effect.expected_delta_min <= 0.0 => {
                return Err(format!("increase contract for {} must require a positive effect", control.control_id));
            }
            ExpectedEffectDirectionV1::Decrease if effect.expected_delta_max >= 0.0 => {
                return Err(format!("decrease contract for {} must require a negative effect", control.control_id));
            }
            _ => {}
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionKindV1 {
    Admitted,
    ExplicitDowngradeRequired,
    Abstain,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionReasonCodeV1 {
    TaskUnsupported,
    InputModalityUnsupported,
    OutputModalityUnsupported,
    ControlUnavailable,
    GuaranteeDowngrade,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionReasonV1 {
    pub code: AdmissionReasonCodeV1,
    pub control_id: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionDecisionV1 {
    pub kind: AdmissionKindV1,
    pub backend_id: String,
    pub manifest_sha256: String,
    pub reasons: Vec<AdmissionReasonV1>,
}

impl AdmissionDecisionV1 {
    pub fn may_execute_without_user_downgrade(&self) -> bool {
        self.kind == AdmissionKindV1::Admitted
    }
}

pub fn admit_realization_v1(
    manifest: &BackendCapabilityManifestV1,
    request: &MusicalRealizationRequestV1,
) -> Result<AdmissionDecisionV1, String> {
    manifest.validate()?;
    request.validate()?;
    let mut reasons = Vec::new();
    if !manifest.payload.tasks.contains(&request.payload.task) {
        reasons.push(AdmissionReasonV1 {
            code: AdmissionReasonCodeV1::TaskUnsupported,
            control_id: None,
            detail: "backend does not declare the requested task".to_owned(),
        });
    }
    if !manifest.payload.input_modalities.contains(&request.payload.input_modality) {
        reasons.push(AdmissionReasonV1 {
            code: AdmissionReasonCodeV1::InputModalityUnsupported,
            control_id: None,
            detail: "backend does not accept the requested input modality".to_owned(),
        });
    }
    if !manifest.payload.output_modalities.contains(&request.payload.required_output_modality) {
        reasons.push(AdmissionReasonV1 {
            code: AdmissionReasonCodeV1::OutputModalityUnsupported,
            control_id: None,
            detail: "backend cannot produce the required output modality".to_owned(),
        });
    }
    for requested in &request.payload.controls {
        let Some(capability) = manifest.control(&requested.control_id) else {
            reasons.push(AdmissionReasonV1 {
                code: AdmissionReasonCodeV1::ControlUnavailable,
                control_id: Some(requested.control_id.clone()),
                detail: "backend has no declaration for this control".to_owned(),
            });
            continue;
        };
        if capability.guarantee == ControlGuaranteeV1::Unsupported {
            reasons.push(AdmissionReasonV1 {
                code: AdmissionReasonCodeV1::ControlUnavailable,
                control_id: Some(requested.control_id.clone()),
                detail: "backend explicitly marks this control unsupported".to_owned(),
            });
            continue;
        }
        let downgrade = match requested.guarantee_requirement {
            GuaranteeRequirementV1::Exact => capability.guarantee != ControlGuaranteeV1::Hard,
            GuaranteeRequirementV1::BestEffort => capability.guarantee == ControlGuaranteeV1::Advisory,
            GuaranteeRequirementV1::Advisory => false,
        };
        if downgrade {
            reasons.push(AdmissionReasonV1 {
                code: AdmissionReasonCodeV1::GuaranteeDowngrade,
                control_id: Some(requested.control_id.clone()),
                detail: format!(
                    "request requires {:?}, backend only declares {:?}; explicit approval is required",
                    requested.guarantee_requirement, capability.guarantee
                ),
            });
        }
    }
    let kind = if reasons.iter().any(|reason| matches!(
        reason.code,
        AdmissionReasonCodeV1::TaskUnsupported
            | AdmissionReasonCodeV1::InputModalityUnsupported
            | AdmissionReasonCodeV1::OutputModalityUnsupported
            | AdmissionReasonCodeV1::ControlUnavailable
    )) {
        AdmissionKindV1::Abstain
    } else if !reasons.is_empty() {
        AdmissionKindV1::ExplicitDowngradeRequired
    } else {
        AdmissionKindV1::Admitted
    };
    Ok(AdmissionDecisionV1 {
        kind,
        backend_id: manifest.payload.backend_id.clone(),
        manifest_sha256: manifest.payload_sha256.clone(),
        reasons,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderControlStatusV1 {
    Applied,
    Ignored,
    Rejected,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollateralMetricV1 {
    pub metric_id: String,
    pub observed_delta: f64,
    pub allowed_abs_delta: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndependentObservationV1 {
    pub metric_id: String,
    pub baseline_value: f64,
    pub treatment_value: f64,
    pub sham_delta: f64,
    pub reversal_delta: Option<f64>,
    pub capture_sha256: String,
    pub observer_manifest_sha256: String,
    pub observer_identity_sha256: String,
    pub observer_independent: bool,
    pub complete_window: bool,
    pub collateral_metrics: Vec<CollateralMetricV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlExecutionReceiptV1 {
    pub control_id: String,
    pub requested_value: f64,
    pub provider_accepted_value: Option<f64>,
    pub provider_effective_value: Option<f64>,
    pub provider_status: ProviderControlStatusV1,
    pub independent_observation: Option<IndependentObservationV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicalRealizationReceiptPayloadV1 {
    pub schema_version: u32,
    pub request_sha256: String,
    pub capability_manifest_sha256: String,
    pub backend_id: String,
    pub renderer_version: String,
    pub adapter_version: String,
    pub input_artifact_sha256: String,
    pub output_artifact_sha256: String,
    pub control_outcomes: Vec<ControlExecutionReceiptV1>,
    pub measured_latency_us: Option<u64>,
    pub verification_policy_id: String,
    pub execution_complete: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicalRealizationReceiptV1 {
    pub payload: MusicalRealizationReceiptPayloadV1,
    pub payload_sha256: String,
}

impl MusicalRealizationReceiptV1 {
    pub fn from_payload(payload: MusicalRealizationReceiptPayloadV1) -> Result<Self, String> {
        validate_receipt_payload_v1(&payload)?;
        let bytes = serde_json::to_vec(&payload).map_err(|error| error.to_string())?;
        Ok(Self {
            payload,
            payload_sha256: sha256_hex(&bytes),
        })
    }

    pub fn validate_against(
        &self,
        manifest: &BackendCapabilityManifestV1,
        request: &MusicalRealizationRequestV1,
    ) -> Result<(), String> {
        manifest.validate()?;
        request.validate()?;
        validate_receipt_payload_v1(&self.payload)?;
        let bytes = serde_json::to_vec(&self.payload).map_err(|error| error.to_string())?;
        if self.payload_sha256 != sha256_hex(&bytes) {
            return Err("realization receipt digest does not match its payload".to_owned());
        }
        if self.payload.request_sha256 != request.payload_sha256
            || self.payload.capability_manifest_sha256 != manifest.payload_sha256
        {
            return Err("receipt is not bound to the exact request and capability manifest".to_owned());
        }
        if self.payload.backend_id != manifest.payload.backend_id
            || self.payload.renderer_version != manifest.payload.renderer_version
            || self.payload.adapter_version != manifest.payload.adapter_version
        {
            return Err("receipt backend/renderer/adapter identity mismatch".to_owned());
        }
        if self.payload.input_artifact_sha256 != request.payload.source_artifact_sha256
            || self.payload.verification_policy_id != request.payload.verification_policy_id
        {
            return Err("receipt input artifact or verification policy mismatch".to_owned());
        }
        let expected: BTreeSet<&str> = request.payload.controls.iter().map(|control| control.control_id.as_str()).collect();
        let actual: BTreeSet<&str> = self.payload.control_outcomes.iter().map(|control| control.control_id.as_str()).collect();
        if expected != actual || actual.len() != self.payload.control_outcomes.len() {
            return Err("receipt control outcomes are missing, unexpected, or duplicated".to_owned());
        }
        for outcome in &self.payload.control_outcomes {
            let requested = request.payload.controls.iter().find(|control| control.control_id == outcome.control_id)
                .expect("control outcome set equality was checked");
            if outcome.requested_value.to_bits() != requested.requested_value.to_bits() {
                return Err(format!("receipt requested value mismatch for {}", outcome.control_id));
            }
            if let Some(observation) = &outcome.independent_observation {
                if observation.capture_sha256 != self.payload.output_artifact_sha256 {
                    return Err(format!("observation capture is not the exact realization output for {}", outcome.control_id));
                }
                if observation.metric_id != requested.effect_contract.target_metric_id {
                    return Err(format!("observed metric mismatch for {}", outcome.control_id));
                }
                if requested.effect_contract.require_direction_reversal && observation.reversal_delta.is_none() {
                    return Err(format!("required direction reversal is missing for {}", outcome.control_id));
                }
            }
        }
        Ok(())
    }
}

fn validate_receipt_payload_v1(payload: &MusicalRealizationReceiptPayloadV1) -> Result<(), String> {
    if payload.schema_version != REALIZATION_CONTRACT_SCHEMA_VERSION_V1 {
        return Err("unsupported realization receipt schema version".to_owned());
    }
    for (value, label) in [
        (&payload.request_sha256, "request digest"),
        (&payload.capability_manifest_sha256, "capability manifest digest"),
        (&payload.input_artifact_sha256, "receipt input digest"),
        (&payload.output_artifact_sha256, "receipt output digest"),
    ] {
        require_sha256(value, label)?;
    }
    for (value, label) in [
        (&payload.backend_id, "backend ID"),
        (&payload.renderer_version, "renderer version"),
        (&payload.adapter_version, "adapter version"),
        (&payload.verification_policy_id, "verification policy ID"),
    ] {
        if value.trim().is_empty() {
            return Err(format!("{label} must not be empty"));
        }
    }
    if !payload.execution_complete {
        return Err("incomplete execution cannot produce a scored realization receipt".to_owned());
    }
    let mut ids = BTreeSet::new();
    for outcome in &payload.control_outcomes {
        if outcome.control_id.trim().is_empty() || !ids.insert(outcome.control_id.as_str()) {
            return Err("receipt control IDs must be non-empty and unique".to_owned());
        }
        if !outcome.requested_value.is_finite()
            || outcome.provider_accepted_value.is_some_and(|value| !value.is_finite())
            || outcome.provider_effective_value.is_some_and(|value| !value.is_finite())
        {
            return Err(format!("non-finite control receipt value for {}", outcome.control_id));
        }
        if let Some(observation) = &outcome.independent_observation {
            if !observation.baseline_value.is_finite()
                || !observation.treatment_value.is_finite()
                || !observation.sham_delta.is_finite()
                || observation.reversal_delta.is_some_and(|value| !value.is_finite())
            {
                return Err(format!("non-finite independent observation for {}", outcome.control_id));
            }
            require_sha256(&observation.capture_sha256, "captured output digest")?;
            require_sha256(&observation.observer_manifest_sha256, "observer manifest digest")?;
            require_sha256(&observation.observer_identity_sha256, "observer identity digest")?;
            let mut collateral_ids = BTreeSet::new();
            for metric in &observation.collateral_metrics {
                if metric.metric_id.trim().is_empty()
                    || !collateral_ids.insert(metric.metric_id.as_str())
                    || !metric.observed_delta.is_finite()
                    || !metric.allowed_abs_delta.is_finite()
                    || metric.allowed_abs_delta < 0.0
                {
                    return Err(format!("invalid collateral metric for {}", outcome.control_id));
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlEffectAssessmentV1 {
    ObservedAsExpected,
    NoObservedEffect,
    ShamConfounded,
    DirectionReversalFailed,
    CollateralContractViolated,
    ProviderIgnored,
    ProviderRejected,
    EvidenceInsufficient,
}

pub fn assess_control_effect_v1(
    requested: &RequestedControlV1,
    outcome: &ControlExecutionReceiptV1,
) -> ControlEffectAssessmentV1 {
    if requested.control_id != outcome.control_id
        || requested.requested_value.to_bits() != outcome.requested_value.to_bits()
    {
        return ControlEffectAssessmentV1::EvidenceInsufficient;
    }
    match outcome.provider_status {
        ProviderControlStatusV1::Ignored => return ControlEffectAssessmentV1::ProviderIgnored,
        ProviderControlStatusV1::Rejected => return ControlEffectAssessmentV1::ProviderRejected,
        ProviderControlStatusV1::Unknown => return ControlEffectAssessmentV1::EvidenceInsufficient,
        ProviderControlStatusV1::Applied => {}
    }
    let Some(observation) = &outcome.independent_observation else {
        return ControlEffectAssessmentV1::EvidenceInsufficient;
    };
    if !observation.observer_independent || !observation.complete_window {
        return ControlEffectAssessmentV1::EvidenceInsufficient;
    }
    let delta = observation.treatment_value - observation.baseline_value;
    if !delta.is_finite() || observation.sham_delta.abs() > requested.effect_contract.sham_abs_delta_max {
        return ControlEffectAssessmentV1::ShamConfounded;
    }
    let expected = delta >= requested.effect_contract.expected_delta_min
        && delta <= requested.effect_contract.expected_delta_max;
    if !expected {
        return ControlEffectAssessmentV1::NoObservedEffect;
    }
    if requested.effect_contract.require_direction_reversal {
        let Some(reversal) = observation.reversal_delta else {
            return ControlEffectAssessmentV1::EvidenceInsufficient;
        };
        let reversed = match requested.expected_direction {
            ExpectedEffectDirectionV1::Increase => reversal < 0.0,
            ExpectedEffectDirectionV1::Decrease => reversal > 0.0,
        };
        if !reversed {
            return ControlEffectAssessmentV1::DirectionReversalFailed;
        }
    }
    if observation.collateral_metrics.iter().any(|metric| {
        metric.observed_delta.abs() > metric.allowed_abs_delta
    }) {
        return ControlEffectAssessmentV1::CollateralContractViolated;
    }
    ControlEffectAssessmentV1::ObservedAsExpected
}

fn ensure_unique<T: Ord>(values: &[T], label: &str) -> Result<(), String> {
    let mut unique = BTreeSet::new();
    for value in values {
        if !unique.insert(value) {
            return Err(format!("duplicate {label}"));
        }
    }
    Ok(())
}

fn require_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
        || value.bytes().any(|byte| byte.is_ascii_uppercase())
    {
        return Err(format!("{label} must be a 64-character lowercase SHA-256 digest"));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn control_capability(guarantee: ControlGuaranteeV1) -> ControlCapabilityV1 {
        ControlCapabilityV1 {
            control_id: "bpm".to_owned(),
            semantics: "target tempo in beats per minute".to_owned(),
            guarantee,
            evidence_level: if guarantee == ControlGuaranteeV1::Hard {
                CapabilityEvidenceLevelV1::IndependentConformance
            } else {
                CapabilityEvidenceLevelV1::ProviderDocumentation
            },
            conformance_test_id: (guarantee == ControlGuaranteeV1::Hard).then(|| "bpm-loopback-v1".to_owned()),
            evidence_references: vec!["fixture://bpm-control".to_owned()],
        }
    }

    fn manifest(guarantee: ControlGuaranteeV1) -> BackendCapabilityManifestV1 {
        BackendCapabilityManifestV1::from_payload(BackendCapabilityManifestPayloadV1 {
            schema_version: 1,
            backend_id: "fixture-renderer".to_owned(),
            renderer_version: "1.2.3".to_owned(),
            adapter_version: "adapter-1".to_owned(),
            environment_sha256: digest('a'),
            tasks: vec![RealizationTaskV1::LiveStreaming, RealizationTaskV1::AudioEditing],
            input_modalities: vec![ArtifactModalityV1::MusicalIntent],
            output_modalities: vec![ArtifactModalityV1::Audio],
            controls: vec![control_capability(guarantee)],
            seed_supported: true,
            deterministic_for_fixed_seed: true,
            stateful_session: true,
            reset_supported: true,
            version_pinned: true,
            clock_contract: Some(ClockContractV1 {
                clock_domain: "fixture-monotonic".to_owned(),
                sample_rate_hz: 48_000,
                buffer_frames: Some(256),
                measured_p50_latency_us: Some(80_000),
                measured_p95_latency_us: Some(95_000),
                measured_p99_latency_us: Some(110_000),
                measurement_evidence_sha256: Some(digest('b')),
            }),
        }).unwrap()
    }

    fn request(requirement: GuaranteeRequirementV1) -> MusicalRealizationRequestV1 {
        MusicalRealizationRequestV1::from_payload(MusicalRealizationRequestPayloadV1 {
            schema_version: 1,
            request_id: "request-0001".to_owned(),
            task: RealizationTaskV1::LiveStreaming,
            source_artifact_sha256: digest('c'),
            input_modality: ArtifactModalityV1::MusicalIntent,
            required_output_modality: ArtifactModalityV1::Audio,
            controls: vec![RequestedControlV1 {
                control_id: "bpm".to_owned(),
                requested_value: 120.0,
                guarantee_requirement: requirement,
                expected_direction: ExpectedEffectDirectionV1::Increase,
                effect_contract: ControlEffectContractV1 {
                    target_metric_id: "audio.tempo_bpm".to_owned(),
                    expected_delta_min: 8.0,
                    expected_delta_max: 12.0,
                    sham_abs_delta_max: 1.0,
                    require_direction_reversal: true,
                },
            }],
            target_region_id: None,
            protected_region_ids: vec!["approved-intro".to_owned()],
            candidate_budget: 1,
            consent_purpose_reference: "consent://session-1/live-collaboration".to_owned(),
            verification_policy_id: "verify-audio-control-v1".to_owned(),
            require_independent_capture: true,
        }).unwrap()
    }

    fn observation(
        sham_delta: f64,
        reversal_delta: Option<f64>,
        observer_independent: bool,
    ) -> IndependentObservationV1 {
        IndependentObservationV1 {
            metric_id: "audio.tempo_bpm".to_owned(),
            baseline_value: 100.0,
            treatment_value: 110.0,
            sham_delta,
            reversal_delta,
            capture_sha256: digest('d'),
            observer_manifest_sha256: digest('e'),
            observer_identity_sha256: digest('f'),
            observer_independent,
            complete_window: true,
            collateral_metrics: vec![CollateralMetricV1 {
                metric_id: "audio.loudness_lufs".to_owned(),
                observed_delta: 0.2,
                allowed_abs_delta: 1.0,
            }],
        }
    }

    #[test]
    fn exact_request_does_not_silently_route_to_best_effort_backend() {
        let decision = admit_realization_v1(
            &manifest(ControlGuaranteeV1::BestEffort),
            &request(GuaranteeRequirementV1::Exact),
        ).unwrap();
        assert_eq!(decision.kind, AdmissionKindV1::ExplicitDowngradeRequired);
        assert!(!decision.may_execute_without_user_downgrade());
    }

    #[test]
    fn unsupported_control_causes_pre_execution_abstention() {
        let decision = admit_realization_v1(
            &manifest(ControlGuaranteeV1::Unsupported),
            &request(GuaranteeRequirementV1::Advisory),
        ).unwrap();
        assert_eq!(decision.kind, AdmissionKindV1::Abstain);
    }

    #[test]
    fn independent_observation_confirms_effect_only_when_sham_and_reversal_pass() {
        let requested = request(GuaranteeRequirementV1::Exact).payload.controls[0].clone();
        let outcome = ControlExecutionReceiptV1 {
            control_id: "bpm".to_owned(),
            requested_value: 120.0,
            provider_accepted_value: Some(120.0),
            provider_effective_value: Some(120.0),
            provider_status: ProviderControlStatusV1::Applied,
            independent_observation: Some(observation(0.1, Some(-9.0), true)),
        };
        assert_eq!(
            assess_control_effect_v1(&requested, &outcome),
            ControlEffectAssessmentV1::ObservedAsExpected
        );
    }

    #[test]
    fn accepted_request_is_not_audible_effect_when_independent_output_did_not_change() {
        let requested = request(GuaranteeRequirementV1::Exact).payload.controls[0].clone();
        let mut obs = observation(0.0, Some(0.0), true);
        obs.treatment_value = 101.0;
        let outcome = ControlExecutionReceiptV1 {
            control_id: "bpm".to_owned(),
            requested_value: 120.0,
            provider_accepted_value: Some(120.0),
            provider_effective_value: Some(120.0),
            provider_status: ProviderControlStatusV1::Applied,
            independent_observation: Some(obs),
        };
        assert_eq!(
            assess_control_effect_v1(&requested, &outcome),
            ControlEffectAssessmentV1::NoObservedEffect
        );
    }

    #[test]
    fn sham_effect_invalidates_causal_control_claim() {
        let requested = request(GuaranteeRequirementV1::Exact).payload.controls[0].clone();
        let outcome = ControlExecutionReceiptV1 {
            control_id: "bpm".to_owned(),
            requested_value: 120.0,
            provider_accepted_value: Some(120.0),
            provider_effective_value: Some(120.0),
            provider_status: ProviderControlStatusV1::Applied,
            independent_observation: Some(observation(7.0, Some(-9.0), true)),
        };
        assert_eq!(
            assess_control_effect_v1(&requested, &outcome),
            ControlEffectAssessmentV1::ShamConfounded
        );
    }

    #[test]
    fn self_reported_or_incomplete_observation_cannot_confirm_a_control() {
        let requested = request(GuaranteeRequirementV1::Exact).payload.controls[0].clone();
        let outcome = ControlExecutionReceiptV1 {
            control_id: "bpm".to_owned(),
            requested_value: 120.0,
            provider_accepted_value: Some(120.0),
            provider_effective_value: Some(120.0),
            provider_status: ProviderControlStatusV1::Applied,
            independent_observation: Some(observation(0.0, Some(-9.0), false)),
        };
        assert_eq!(
            assess_control_effect_v1(&requested, &outcome),
            ControlEffectAssessmentV1::EvidenceInsufficient
        );
    }

    #[test]
    fn receipt_must_bind_exact_manifest_request_output_and_all_control_results() {
        let manifest = manifest(ControlGuaranteeV1::Hard);
        let request = request(GuaranteeRequirementV1::Exact);
        let payload = MusicalRealizationReceiptPayloadV1 {
            schema_version: 1,
            request_sha256: request.payload_sha256.clone(),
            capability_manifest_sha256: manifest.payload_sha256.clone(),
            backend_id: "fixture-renderer".to_owned(),
            renderer_version: "1.2.3".to_owned(),
            adapter_version: "adapter-1".to_owned(),
            input_artifact_sha256: digest('c'),
            output_artifact_sha256: digest('d'),
            control_outcomes: vec![ControlExecutionReceiptV1 {
                control_id: "bpm".to_owned(),
                requested_value: 120.0,
                provider_accepted_value: Some(120.0),
                provider_effective_value: Some(120.0),
                provider_status: ProviderControlStatusV1::Applied,
                independent_observation: Some(observation(0.1, Some(-9.0), true)),
            }],
            measured_latency_us: Some(82_000),
            verification_policy_id: "verify-audio-control-v1".to_owned(),
            execution_complete: true,
        };
        let receipt = MusicalRealizationReceiptV1::from_payload(payload).unwrap();
        receipt.validate_against(&manifest, &request).unwrap();

        let mut tampered = receipt;
        tampered.payload.output_artifact_sha256 = digest('9');
        assert!(tampered.validate_against(&manifest, &request).is_err());
    }
}
