// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Capability-qualified, backend-neutral contracts for musical realization.
//!
//! Cognition produces typed intent; a realizer executes only controls it
//! exposes; an independent verifier evaluates the returned artifact. Provider
//! documentation or a successful API response is not proof of task success.
//! This first slice is schema + offline dispatch qualification only: no SDKs,
//! network calls, credentials, rendering code or live backend are included.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write as FmtWrite;

pub const REALIZATION_CONTRACT_SCHEMA_VERSION_V1: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealizationTaskV1 {
    FullGeneration,
    Continuation,
    LocalizedEdit,
    ScoreRendering,
    LiveStreaming,
    AudioEditing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealizationModalityV1 {
    Prompt,
    Score,
    Midi,
    LiveMidi,
    Audio,
    LiveAudio,
    Stems,
    Text,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealizationControlV1 {
    ExactNotes,
    Motif,
    Meter,
    Tempo,
    Key,
    MusicalRole,
    Density,
    Timbre,
    InstrumentIdentity,
    StemSelection,
    FinePitch,
    PromptSteering,
    LiveTiming,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlGuaranteeV1 {
    Hard,
    BestEffort,
    Advisory,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlRequirementStrengthV1 {
    HardRequired,
    SupportedRequired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealizationObservationV1 {
    ExactScoreArtifact,
    ExactMidiArtifact,
    EventLineage,
    StemIdentity,
    AudioArtifactBytes,
    StreamEventTiming,
    EffectiveControlReport,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityEvidenceV1 {
    pub evidence_ref: String,
    pub evidence_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlCapabilityV1 {
    pub control: RealizationControlV1,
    pub guarantee: ControlGuaranteeV1,
    /// None is permitted only when the capability is explicitly unavailable.
    pub evidence: Option<CapabilityEvidenceV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapabilityManifestV1 {
    pub schema_version: u32,
    pub backend_id: String,
    pub backend_version: String,
    pub adapter_version: String,
    pub supported_tasks: Vec<RealizationTaskV1>,
    pub accepted_inputs: Vec<RealizationModalityV1>,
    pub produced_outputs: Vec<RealizationModalityV1>,
    pub controls: Vec<ControlCapabilityV1>,
    pub observations: Vec<RealizationObservationV1>,
    pub deterministic_replay: bool,
    pub seed_control: bool,
    pub stateful_sessions: bool,
    pub reset_supported: bool,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityManifestIssueV1 {
    UnsupportedSchemaVersion { found: u32 },
    EmptyField { field: String },
    EmptyTaskSet,
    EmptyInputSet,
    EmptyOutputSet,
    DuplicateTask { task: RealizationTaskV1 },
    DuplicateInput { modality: RealizationModalityV1 },
    DuplicateOutput { modality: RealizationModalityV1 },
    DuplicateControl { control: RealizationControlV1 },
    DuplicateObservation { observation: RealizationObservationV1 },
    MissingCapabilityEvidence { control: RealizationControlV1 },
    InvalidCapabilityEvidenceDigest { control: RealizationControlV1 },
    EmptyCapabilityEvidenceRef { control: RealizationControlV1 },
    EmptyLimitation,
}

impl BackendCapabilityManifestV1 {
    pub fn validate(&self) -> Vec<CapabilityManifestIssueV1> {
        let mut issues = Vec::new();
        if self.schema_version != REALIZATION_CONTRACT_SCHEMA_VERSION_V1 {
            issues.push(CapabilityManifestIssueV1::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }
        for (field, value) in [
            ("backend_id", self.backend_id.as_str()),
            ("backend_version", self.backend_version.as_str()),
            ("adapter_version", self.adapter_version.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(CapabilityManifestIssueV1::EmptyField {
                    field: field.to_owned(),
                });
            }
        }
        if self.supported_tasks.is_empty() {
            issues.push(CapabilityManifestIssueV1::EmptyTaskSet);
        }
        if self.accepted_inputs.is_empty() {
            issues.push(CapabilityManifestIssueV1::EmptyInputSet);
        }
        if self.produced_outputs.is_empty() {
            issues.push(CapabilityManifestIssueV1::EmptyOutputSet);
        }

        let mut tasks = BTreeSet::new();
        for task in &self.supported_tasks {
            if !tasks.insert(*task) {
                issues.push(CapabilityManifestIssueV1::DuplicateTask { task: *task });
            }
        }
        let mut inputs = BTreeSet::new();
        for modality in &self.accepted_inputs {
            if !inputs.insert(*modality) {
                issues.push(CapabilityManifestIssueV1::DuplicateInput { modality: *modality });
            }
        }
        let mut outputs = BTreeSet::new();
        for modality in &self.produced_outputs {
            if !outputs.insert(*modality) {
                issues.push(CapabilityManifestIssueV1::DuplicateOutput { modality: *modality });
            }
        }
        let mut controls = BTreeSet::new();
        for capability in &self.controls {
            if !controls.insert(capability.control) {
                issues.push(CapabilityManifestIssueV1::DuplicateControl {
                    control: capability.control,
                });
            }
            if capability.guarantee == ControlGuaranteeV1::Unavailable {
                continue;
            }
            match &capability.evidence {
                None => issues.push(CapabilityManifestIssueV1::MissingCapabilityEvidence {
                    control: capability.control,
                }),
                Some(evidence) => {
                    if evidence.evidence_ref.trim().is_empty() {
                        issues.push(CapabilityManifestIssueV1::EmptyCapabilityEvidenceRef {
                            control: capability.control,
                        });
                    }
                    if !is_lowercase_sha256(&evidence.evidence_sha256) {
                        issues.push(CapabilityManifestIssueV1::InvalidCapabilityEvidenceDigest {
                            control: capability.control,
                        });
                    }
                }
            }
        }
        let mut observations = BTreeSet::new();
        for observation in &self.observations {
            if !observations.insert(*observation) {
                issues.push(CapabilityManifestIssueV1::DuplicateObservation {
                    observation: *observation,
                });
            }
        }
        if self.limitations.iter().any(|item| item.trim().is_empty()) {
            issues.push(CapabilityManifestIssueV1::EmptyLimitation);
        }
        issues
    }

    pub fn serialized_sha256(&self) -> Result<String, String> {
        serialized_sha256_v1(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequiredControlV1 {
    pub control: RealizationControlV1,
    pub strength: ControlRequirementStrengthV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicalRealizationRequestV1 {
    pub schema_version: u32,
    pub request_id: String,
    pub task: RealizationTaskV1,
    pub required_inputs: Vec<RealizationModalityV1>,
    pub required_output: RealizationModalityV1,
    pub required_controls: Vec<RequiredControlV1>,
    pub required_observations: Vec<RealizationObservationV1>,
    pub source_artifact_sha256: Option<String>,
    pub intent_sha256: String,
    pub consent_receipt_sha256: Option<String>,
    pub require_deterministic_replay: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RealizationRequestIssueV1 {
    UnsupportedSchemaVersion { found: u32 },
    EmptyRequestId,
    EmptyInputSet,
    DuplicateInput { modality: RealizationModalityV1 },
    DuplicateControl { control: RealizationControlV1 },
    DuplicateObservation { observation: RealizationObservationV1 },
    InvalidIntentDigest,
    InvalidSourceArtifactDigest,
    InvalidConsentReceiptDigest,
}

impl MusicalRealizationRequestV1 {
    pub fn validate(&self) -> Vec<RealizationRequestIssueV1> {
        let mut issues = Vec::new();
        if self.schema_version != REALIZATION_CONTRACT_SCHEMA_VERSION_V1 {
            issues.push(RealizationRequestIssueV1::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }
        if self.request_id.trim().is_empty() {
            issues.push(RealizationRequestIssueV1::EmptyRequestId);
        }
        if self.required_inputs.is_empty() {
            issues.push(RealizationRequestIssueV1::EmptyInputSet);
        }
        if !is_lowercase_sha256(&self.intent_sha256) {
            issues.push(RealizationRequestIssueV1::InvalidIntentDigest);
        }
        if self.source_artifact_sha256.as_deref().is_some_and(|v| !is_lowercase_sha256(v)) {
            issues.push(RealizationRequestIssueV1::InvalidSourceArtifactDigest);
        }
        if self.consent_receipt_sha256.as_deref().is_some_and(|v| !is_lowercase_sha256(v)) {
            issues.push(RealizationRequestIssueV1::InvalidConsentReceiptDigest);
        }

        let mut inputs = BTreeSet::new();
        for modality in &self.required_inputs {
            if !inputs.insert(*modality) {
                issues.push(RealizationRequestIssueV1::DuplicateInput { modality: *modality });
            }
        }
        let mut controls = BTreeSet::new();
        for required in &self.required_controls {
            if !controls.insert(required.control) {
                issues.push(RealizationRequestIssueV1::DuplicateControl {
                    control: required.control,
                });
            }
        }
        let mut observations = BTreeSet::new();
        for observation in &self.required_observations {
            if !observations.insert(*observation) {
                issues.push(RealizationRequestIssueV1::DuplicateObservation {
                    observation: *observation,
                });
            }
        }
        issues
    }

    pub fn serialized_sha256(&self) -> Result<String, String> {
        serialized_sha256_v1(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DispatchBlockerV1 {
    InvalidManifest { issues: Vec<CapabilityManifestIssueV1> },
    InvalidRequest { issues: Vec<RealizationRequestIssueV1> },
    TaskUnsupported { task: RealizationTaskV1 },
    InputUnsupported { modality: RealizationModalityV1 },
    OutputUnsupported { modality: RealizationModalityV1 },
    ControlUnavailable { control: RealizationControlV1 },
    ControlNotHard { control: RealizationControlV1, found: ControlGuaranteeV1 },
    ObservationUnavailable { observation: RealizationObservationV1 },
    DeterministicReplayUnavailable,
    DigestFailure { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DispatchQualificationV1 {
    pub backend_id: String,
    pub backend_version: String,
    pub capability_manifest_sha256: String,
    pub request_sha256: String,
    pub accepted_controls: Vec<RealizationControlV1>,
    pub accepted_observations: Vec<RealizationObservationV1>,
}

/// Qualifies compatibility only; it performs no network call and gives no
/// authority to execute, store, train on, publish or federate any artifact.
pub fn qualify_backend_v1(
    manifest: &BackendCapabilityManifestV1,
    request: &MusicalRealizationRequestV1,
) -> Result<DispatchQualificationV1, Vec<DispatchBlockerV1>> {
    let manifest_issues = manifest.validate();
    let request_issues = request.validate();
    let mut blockers = Vec::new();
    if !manifest_issues.is_empty() {
        blockers.push(DispatchBlockerV1::InvalidManifest { issues: manifest_issues });
    }
    if !request_issues.is_empty() {
        blockers.push(DispatchBlockerV1::InvalidRequest { issues: request_issues });
    }
    if !blockers.is_empty() {
        return Err(blockers);
    }

    if !manifest.supported_tasks.contains(&request.task) {
        blockers.push(DispatchBlockerV1::TaskUnsupported { task: request.task });
    }
    for modality in &request.required_inputs {
        if !manifest.accepted_inputs.contains(modality) {
            blockers.push(DispatchBlockerV1::InputUnsupported { modality: *modality });
        }
    }
    if !manifest.produced_outputs.contains(&request.required_output) {
        blockers.push(DispatchBlockerV1::OutputUnsupported {
            modality: request.required_output,
        });
    }
    for required in &request.required_controls {
        match manifest.controls.iter().find(|item| item.control == required.control) {
            None => blockers.push(DispatchBlockerV1::ControlUnavailable {
                control: required.control,
            }),
            Some(item) if item.guarantee == ControlGuaranteeV1::Unavailable => {
                blockers.push(DispatchBlockerV1::ControlUnavailable {
                    control: required.control,
                });
            }
            Some(item)
                if required.strength == ControlRequirementStrengthV1::HardRequired
                    && item.guarantee != ControlGuaranteeV1::Hard =>
            {
                blockers.push(DispatchBlockerV1::ControlNotHard {
                    control: required.control,
                    found: item.guarantee,
                });
            }
            Some(_) => {}
        }
    }
    for observation in &request.required_observations {
        if !manifest.observations.contains(observation) {
            blockers.push(DispatchBlockerV1::ObservationUnavailable {
                observation: *observation,
            });
        }
    }
    if request.require_deterministic_replay && !manifest.deterministic_replay {
        blockers.push(DispatchBlockerV1::DeterministicReplayUnavailable);
    }
    if !blockers.is_empty() {
        return Err(blockers);
    }

    let capability_manifest_sha256 = manifest
        .serialized_sha256()
        .map_err(|reason| vec![DispatchBlockerV1::DigestFailure { reason }])?;
    let request_sha256 = request
        .serialized_sha256()
        .map_err(|reason| vec![DispatchBlockerV1::DigestFailure { reason }])?;

    Ok(DispatchQualificationV1 {
        backend_id: manifest.backend_id.clone(),
        backend_version: manifest.backend_version.clone(),
        capability_manifest_sha256,
        request_sha256,
        accepted_controls: request.required_controls.iter().map(|item| item.control).collect(),
        accepted_observations: request.required_observations.clone(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealizationOutcomeV1 {
    Completed,
    Partial,
    Abstained,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicalRealizationReceiptV1 {
    pub schema_version: u32,
    pub request_id: String,
    pub backend_id: String,
    pub backend_version: String,
    pub adapter_version: String,
    pub capability_manifest_sha256: String,
    pub request_sha256: String,
    pub outcome: RealizationOutcomeV1,
    pub output_artifact_sha256: Vec<String>,
    pub accepted_controls: Vec<RealizationControlV1>,
    pub ignored_controls: Vec<RealizationControlV1>,
    pub abstention_or_failure_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RealizationReceiptIssueV1 {
    UnsupportedSchemaVersion { found: u32 },
    EmptyIdentity { field: String },
    InvalidManifestDigest,
    InvalidRequestDigest,
    CompletedWithoutOutput,
    InvalidOutputDigest,
    DuplicateOutputDigest,
    DuplicateAcceptedControl { control: RealizationControlV1 },
    DuplicateIgnoredControl { control: RealizationControlV1 },
    ControlBothAcceptedAndIgnored { control: RealizationControlV1 },
    MissingFailureReason,
}

impl MusicalRealizationReceiptV1 {
    pub fn validate(&self) -> Vec<RealizationReceiptIssueV1> {
        let mut issues = Vec::new();
        if self.schema_version != REALIZATION_CONTRACT_SCHEMA_VERSION_V1 {
            issues.push(RealizationReceiptIssueV1::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }
        for (field, value) in [
            ("request_id", self.request_id.as_str()),
            ("backend_id", self.backend_id.as_str()),
            ("backend_version", self.backend_version.as_str()),
            ("adapter_version", self.adapter_version.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(RealizationReceiptIssueV1::EmptyIdentity {
                    field: field.to_owned(),
                });
            }
        }
        if !is_lowercase_sha256(&self.capability_manifest_sha256) {
            issues.push(RealizationReceiptIssueV1::InvalidManifestDigest);
        }
        if !is_lowercase_sha256(&self.request_sha256) {
            issues.push(RealizationReceiptIssueV1::InvalidRequestDigest);
        }
        if self.outcome == RealizationOutcomeV1::Completed && self.output_artifact_sha256.is_empty()
        {
            issues.push(RealizationReceiptIssueV1::CompletedWithoutOutput);
        }
        let mut outputs = BTreeSet::new();
        for digest in &self.output_artifact_sha256 {
            if !is_lowercase_sha256(digest) {
                issues.push(RealizationReceiptIssueV1::InvalidOutputDigest);
            }
            if !outputs.insert(digest) {
                issues.push(RealizationReceiptIssueV1::DuplicateOutputDigest);
            }
        }
        let mut accepted = BTreeSet::new();
        for control in &self.accepted_controls {
            if !accepted.insert(*control) {
                issues.push(RealizationReceiptIssueV1::DuplicateAcceptedControl {
                    control: *control,
                });
            }
        }
        let mut ignored = BTreeSet::new();
        for control in &self.ignored_controls {
            if !ignored.insert(*control) {
                issues.push(RealizationReceiptIssueV1::DuplicateIgnoredControl {
                    control: *control,
                });
            }
        }
        for control in accepted.intersection(&ignored) {
            issues.push(RealizationReceiptIssueV1::ControlBothAcceptedAndIgnored {
                control: *control,
            });
        }
        if self.outcome != RealizationOutcomeV1::Completed
            && self.abstention_or_failure_reason.as_deref().is_none_or(|reason| reason.trim().is_empty())
        {
            issues.push(RealizationReceiptIssueV1::MissingFailureReason);
        }
        issues
    }
}

pub fn serialized_sha256_v1<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    let digest = Sha256::digest(&bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    Ok(encoded)
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence() -> CapabilityEvidenceV1 {
        CapabilityEvidenceV1 {
            evidence_ref: "test-fixture:exact-notes-v1".into(),
            evidence_sha256: "a".repeat(64),
        }
    }

    fn symbolic_manifest() -> BackendCapabilityManifestV1 {
        BackendCapabilityManifestV1 {
            schema_version: REALIZATION_CONTRACT_SCHEMA_VERSION_V1,
            backend_id: "symbolic-fixture".into(),
            backend_version: "1".into(),
            adapter_version: "1".into(),
            supported_tasks: vec![RealizationTaskV1::LocalizedEdit],
            accepted_inputs: vec![RealizationModalityV1::Score],
            produced_outputs: vec![RealizationModalityV1::Score, RealizationModalityV1::Midi],
            controls: vec![
                ControlCapabilityV1 {
                    control: RealizationControlV1::ExactNotes,
                    guarantee: ControlGuaranteeV1::Hard,
                    evidence: Some(evidence()),
                },
                ControlCapabilityV1 {
                    control: RealizationControlV1::Tempo,
                    guarantee: ControlGuaranteeV1::Hard,
                    evidence: Some(evidence()),
                },
            ],
            observations: vec![
                RealizationObservationV1::ExactScoreArtifact,
                RealizationObservationV1::EventLineage,
                RealizationObservationV1::EffectiveControlReport,
            ],
            deterministic_replay: true,
            seed_control: true,
            stateful_sessions: false,
            reset_supported: true,
            limitations: Vec::new(),
        }
    }

    fn exact_edit_request() -> MusicalRealizationRequestV1 {
        MusicalRealizationRequestV1 {
            schema_version: REALIZATION_CONTRACT_SCHEMA_VERSION_V1,
            request_id: "test-request-1".into(),
            task: RealizationTaskV1::LocalizedEdit,
            required_inputs: vec![RealizationModalityV1::Score],
            required_output: RealizationModalityV1::Score,
            required_controls: vec![
                RequiredControlV1 {
                    control: RealizationControlV1::ExactNotes,
                    strength: ControlRequirementStrengthV1::HardRequired,
                },
                RequiredControlV1 {
                    control: RealizationControlV1::Tempo,
                    strength: ControlRequirementStrengthV1::SupportedRequired,
                },
            ],
            required_observations: vec![
                RealizationObservationV1::ExactScoreArtifact,
                RealizationObservationV1::EventLineage,
            ],
            source_artifact_sha256: Some("b".repeat(64)),
            intent_sha256: "c".repeat(64),
            consent_receipt_sha256: None,
            require_deterministic_replay: true,
        }
    }

    fn audio_only_manifest() -> BackendCapabilityManifestV1 {
        BackendCapabilityManifestV1 {
            schema_version: REALIZATION_CONTRACT_SCHEMA_VERSION_V1,
            backend_id: "audio-stream-fixture".into(),
            backend_version: "preview-1".into(),
            adapter_version: "1".into(),
            supported_tasks: vec![
                RealizationTaskV1::FullGeneration,
                RealizationTaskV1::LiveStreaming,
            ],
            accepted_inputs: vec![
                RealizationModalityV1::Prompt,
                RealizationModalityV1::LiveAudio,
            ],
            produced_outputs: vec![
                RealizationModalityV1::Audio,
                RealizationModalityV1::LiveAudio,
            ],
            controls: vec![ControlCapabilityV1 {
                control: RealizationControlV1::PromptSteering,
                guarantee: ControlGuaranteeV1::BestEffort,
                evidence: Some(evidence()),
            }],
            observations: vec![RealizationObservationV1::AudioArtifactBytes],
            deterministic_replay: false,
            seed_control: false,
            stateful_sessions: true,
            reset_supported: true,
            limitations: vec!["No note-level event output".into()],
        }
    }

    #[test]
    fn compatible_symbolic_backend_qualifies_exact_note_edit() {
        let result = qualify_backend_v1(&symbolic_manifest(), &exact_edit_request())
            .expect("symbolic backend exposes hard exact-note control and event evidence");
        assert_eq!(result.backend_id, "symbolic-fixture");
        assert_eq!(
            result.accepted_controls,
            vec![RealizationControlV1::ExactNotes, RealizationControlV1::Tempo]
        );
        assert_eq!(result.capability_manifest_sha256.len(), 64);
        assert_eq!(result.request_sha256.len(), 64);
    }

    #[test]
    fn audio_only_backend_cannot_claim_exact_note_preservation() {
        let blockers = qualify_backend_v1(&audio_only_manifest(), &exact_edit_request()).unwrap_err();
        assert!(blockers.iter().any(|blocker| matches!(
            blocker,
            DispatchBlockerV1::TaskUnsupported { task: RealizationTaskV1::LocalizedEdit }
        )));
        assert!(blockers.iter().any(|blocker| matches!(
            blocker,
            DispatchBlockerV1::InputUnsupported { modality: RealizationModalityV1::Score }
        )));
        assert!(blockers.iter().any(|blocker| matches!(
            blocker,
            DispatchBlockerV1::OutputUnsupported { modality: RealizationModalityV1::Score }
        )));
        assert!(blockers.iter().any(|blocker| matches!(
            blocker,
            DispatchBlockerV1::ControlUnavailable { control: RealizationControlV1::ExactNotes }
        )));
        assert!(blockers.iter().any(|blocker| matches!(
            blocker,
            DispatchBlockerV1::ObservationUnavailable {
                observation: RealizationObservationV1::ExactScoreArtifact
            }
        )));
        assert!(blockers.contains(&DispatchBlockerV1::DeterministicReplayUnavailable));
    }

    #[test]
    fn best_effort_control_does_not_satisfy_hard_requirement() {
        let mut manifest = symbolic_manifest();
        manifest.controls[0].guarantee = ControlGuaranteeV1::BestEffort;
        let blockers = qualify_backend_v1(&manifest, &exact_edit_request()).unwrap_err();
        assert!(blockers.iter().any(|blocker| matches!(
            blocker,
            DispatchBlockerV1::ControlNotHard {
                control: RealizationControlV1::ExactNotes,
                found: ControlGuaranteeV1::BestEffort
            }
        )));
    }

    #[test]
    fn capability_claim_without_evidence_fails_closed() {
        let mut manifest = symbolic_manifest();
        manifest.controls[0].evidence = None;
        assert!(manifest.validate().contains(
            &CapabilityManifestIssueV1::MissingCapabilityEvidence {
                control: RealizationControlV1::ExactNotes
            }
        ));
        assert!(qualify_backend_v1(&manifest, &exact_edit_request()).is_err());
    }

    #[test]
    fn unavailable_capability_needs_no_evidence_but_never_qualifies() {
        let mut manifest = symbolic_manifest();
        manifest.controls[0].guarantee = ControlGuaranteeV1::Unavailable;
        manifest.controls[0].evidence = None;
        assert!(manifest.validate().is_empty());
        let blockers = qualify_backend_v1(&manifest, &exact_edit_request()).unwrap_err();
        assert!(blockers.contains(&DispatchBlockerV1::ControlUnavailable {
            control: RealizationControlV1::ExactNotes
        }));
    }

    #[test]
    fn completed_receipt_requires_a_bound_output_artifact() {
        let receipt = MusicalRealizationReceiptV1 {
            schema_version: REALIZATION_CONTRACT_SCHEMA_VERSION_V1,
            request_id: "request-1".into(),
            backend_id: "symbolic-fixture".into(),
            backend_version: "1".into(),
            adapter_version: "1".into(),
            capability_manifest_sha256: "a".repeat(64),
            request_sha256: "b".repeat(64),
            outcome: RealizationOutcomeV1::Completed,
            output_artifact_sha256: Vec::new(),
            accepted_controls: vec![RealizationControlV1::ExactNotes],
            ignored_controls: Vec::new(),
            abstention_or_failure_reason: None,
        };
        assert!(receipt
            .validate()
            .contains(&RealizationReceiptIssueV1::CompletedWithoutOutput));
    }

    #[test]
    fn receipt_cannot_mark_one_control_both_accepted_and_ignored() {
        let receipt = MusicalRealizationReceiptV1 {
            schema_version: REALIZATION_CONTRACT_SCHEMA_VERSION_V1,
            request_id: "request-1".into(),
            backend_id: "symbolic-fixture".into(),
            backend_version: "1".into(),
            adapter_version: "1".into(),
            capability_manifest_sha256: "a".repeat(64),
            request_sha256: "b".repeat(64),
            outcome: RealizationOutcomeV1::Partial,
            output_artifact_sha256: vec!["d".repeat(64)],
            accepted_controls: vec![RealizationControlV1::Tempo],
            ignored_controls: vec![RealizationControlV1::Tempo],
            abstention_or_failure_reason: Some("renderer ignored tempo".into()),
        };
        assert!(receipt
            .validate()
            .contains(&RealizationReceiptIssueV1::ControlBothAcceptedAndIgnored {
                control: RealizationControlV1::Tempo
            }));
    }

    #[test]
    fn duplicate_request_control_is_rejected() {
        let mut request = exact_edit_request();
        request.required_controls.push(request.required_controls[0].clone());
        assert!(request.validate().contains(
            &RealizationRequestIssueV1::DuplicateControl {
                control: RealizationControlV1::ExactNotes
            }
        ));
    }
}
