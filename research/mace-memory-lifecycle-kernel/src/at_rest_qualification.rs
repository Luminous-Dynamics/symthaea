// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Control-aware result validation for the Agent Memory Integrity at-rest
//! T9/T10 procedures.
//!
//! This module validates evaluator evidence for a run. It is not a database
//! snapshotter, SUT adapter, backup engine, deletion service, or proof of
//! independent witness custody. The adapter must supply exact full-store
//! snapshot digests and a semantic record inventory for every transition.

use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

pub const AT_REST_QUALIFICATION_SCHEMA_V1: &str = "mace_at_rest_qualification_v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum AtRestCaseIdV1 {
    T9WholeStoreRollback,
    T10ErasureResurrection,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AtRestVerdictV1 {
    Rejected,
    Reported,
    Accepted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AtRestDetectionPointV1 {
    Read,
    Audit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AtRestObservationClassV1 {
    DutBoundary,
    DutReported,
    EndpointSoftwareBoundary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AtRestControlIdV1 {
    C1NoOpReload,
    C2GenuineWriteAfterRestart,
}

/// Ordered experiment events. These record the experiment's causal sequence,
/// not just caller-supplied booleans about what supposedly happened.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AtRestEvidenceEventKindV1 {
    SutRestartedBeforeOperation,
    WriteRequested,
    WriteAcknowledged,
    ErasureRequested,
    ErasureAcknowledged,
    SutStoppedForRestore,
    RestoreApplied,
    SutRestartedAfterRestore,
    OrdinaryReadCompleted,
    C1SutStoppedForNoOpReload,
    C1SutRestartedAfterNoOpReload,
    C1SeededReadCompleted,
    C1InventoryUnchanged,
    C2SutRestartedBeforeWrite,
    C2WriteRequested,
    C2WriteAcknowledged,
    C2ReadbackCompleted,
}

/// Provenance boundary expected to emit each event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AtRestEvidenceEventSourceV1 {
    SutProcessObserver,
    SutWriteApi,
    SutErasureApi,
    RestoreHarness,
    SutReadApi,
    StoreInventoryObserver,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AtRestEvidenceEventV1 {
    pub sequence: u64,
    pub case_run_id: String,
    pub event_id: String,
    pub kind: AtRestEvidenceEventKindV1,
    pub source: AtRestEvidenceEventSourceV1,
    /// Monotonic elapsed nanoseconds from one declared experiment clock.
    pub monotonic_elapsed_ns: u64,
    pub clock_profile_sha256: String,
    /// Digest of the original event/receipt bytes retained by the adapter.
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AtRestControlReceiptV1 {
    pub passed: bool,
    pub case_run_id: String,
    pub evidence_sha256: String,
    pub events: Vec<AtRestEvidenceEventV1>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoreInventoryV1 {
    /// Stable logical record IDs mapped to canonical record-payload digests.
    pub records: BTreeMap<String, String>,
}

impl StoreInventoryV1 {
    pub fn commitment_sha256(&self) -> Result<String, AtRestRejectV1> {
        let mut hasher = Sha256::new();
        hasher.update(b"luminous:mace:at-rest-store-inventory:v1\0");
        hasher.update((self.records.len() as u64).to_be_bytes());
        for (record_id, payload_sha256) in &self.records {
            validate_atom(record_id)?;
            validate_sha256(payload_sha256)?;
            hash_field(&mut hasher, record_id);
            hasher.update(decode_sha256(payload_sha256)?);
        }
        Ok(hex_digest(&hasher.finalize()))
    }

    pub fn validate(&self) -> Result<(), AtRestRejectV1> {
        for (record_id, payload_sha256) in &self.records {
            validate_atom(record_id)?;
            validate_sha256(payload_sha256)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct T9WholeStoreRollbackEvidenceV1 {
    /// Full store image including indexes, journals, sidecars and metadata.
    pub before_snapshot_sha256: String,
    pub after_genuine_write_snapshot_sha256: String,
    pub restored_snapshot_sha256: String,
    /// Digest of the exact normalized storage delta expected for this write.
    pub expected_edit_delta_sha256: String,
    /// Digest of the normalized raw-store delta measured by the evaluator adapter.
    pub observed_edit_delta_sha256: String,
    pub event_trace_sha256: String,
    pub events: Vec<AtRestEvidenceEventV1>,
    pub before_inventory: StoreInventoryV1,
    pub after_genuine_write_inventory: StoreInventoryV1,
    pub restored_inventory: StoreInventoryV1,
    pub new_record_id: String,
    pub new_record_payload_sha256: String,
    pub operation_receipt_sha256: String,
    pub operation_performed_via_sut_write_path: bool,
    pub operation_after_restart: bool,
    pub operation_acknowledged: bool,
    pub sut_stopped_during_restore: bool,
    pub sut_restarted_after_restore: bool,
    pub ordinary_read_path_invoked_after_restore: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct T10ErasureResurrectionEvidenceV1 {
    /// Full store image including indexes, journals, sidecars and metadata.
    pub before_snapshot_sha256: String,
    pub after_erasure_snapshot_sha256: String,
    pub restored_snapshot_sha256: String,
    /// Digest of the exact normalized storage delta expected for this erasure.
    pub expected_edit_delta_sha256: String,
    /// Digest of the normalized raw-store delta measured by the evaluator adapter.
    pub observed_edit_delta_sha256: String,
    pub event_trace_sha256: String,
    pub events: Vec<AtRestEvidenceEventV1>,
    pub before_inventory: StoreInventoryV1,
    pub after_erasure_inventory: StoreInventoryV1,
    pub restored_inventory: StoreInventoryV1,
    pub erased_record_id: String,
    pub erased_record_payload_sha256: String,
    pub erasure_receipt_sha256: String,
    pub erasure_performed_via_sut_interface: bool,
    pub erasure_acknowledged: bool,
    pub sut_stopped_during_restore: bool,
    pub sut_restarted_after_restore: bool,
    pub ordinary_read_path_invoked_after_restore: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AtRestCaseEvidenceV1 {
    T9(T9WholeStoreRollbackEvidenceV1),
    T10(T10ErasureResurrectionEvidenceV1),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AtRestObservationV1 {
    pub verdict: AtRestVerdictV1,
    pub detection_point: AtRestDetectionPointV1,
    pub observation_class: AtRestObservationClassV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AtRestCaseDispositionV1 {
    NotApplicable { reason: String },
    Error { reason: String },
    Observed {
        evidence: AtRestCaseEvidenceV1,
        observation: AtRestObservationV1,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AtRestCaseRecordV1 {
    pub case_id: AtRestCaseIdV1,
    pub case_run_id: String,
    pub disposition: AtRestCaseDispositionV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AtRestQualificationReportV1 {
    pub schema_version: String,
    pub protocol_sha256: String,
    pub sut_profile_sha256: String,
    pub environment_profile_sha256: String,
    pub controls: BTreeMap<AtRestControlIdV1, AtRestControlReceiptV1>,
    pub cases: Vec<AtRestCaseRecordV1>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AtRestRejectV1 {
    UnsupportedSchemaVersion,
    InvalidIdentifier,
    InvalidDigest,
    MissingControl,
    ControlFailed,
    DuplicateCaseRunId,
    MissingT9,
    MissingT10,
    EmptyCaseSet,
    InvalidNotApplicableReason,
    InvalidErrorReason,
    EvidenceCaseMismatch,
    OperationNotPerformedThroughSut,
    OperationNotAcknowledged,
    OperationReceiptInvalid,
    SnapshotDigestInvalid,
    SnapshotDidNotChange,
    EditDeltaMismatch,
    RestoredSnapshotMismatch,
    InventoryInvalid,
    UnexpectedInventoryChange,
    ExpectedRecordAlreadyPresent,
    ExpectedRecordMissing,
    ExpectedRecordDigestMismatch,
    RestoreWhileSutRunning,
    MissingRestartAfterRestore,
    ReadPathNotInvoked,
    InvalidObservation,
    EventTraceMismatch,
    EventBindingMismatch,
    EventReceiptInvalid,
    DuplicateEventId,
    EventTimeNotMonotonic,
    EventClockProfileMismatch,
    ControlEventTraceInvalid,
    ControlEvidenceDigestMismatch,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AtRestFamilyCountsV1 {
    pub applicable: u64,
    pub passed: u64,
    pub partial: u64,
    pub failed: u64,
    pub not_applicable: u64,
    pub errors: u64,
}

impl AtRestFamilyCountsV1 {
    /// Formula from the active Agent Memory Integrity Internet-Draft:
    /// (passes + 0.5 * partial) / applicable.
    ///
    /// N/A and edit errors are reported separately and are not counted as
    /// passes. No denominator is invented for a wholly unmeasured family.
    pub fn metric_value(self) -> Option<f64> {
        if self.applicable == 0 {
            None
        } else {
            Some((self.passed as f64 + 0.5 * self.partial as f64) / self.applicable as f64)
        }
    }

    pub fn zero_acceptance_gate(&self) -> bool {
        self.applicable > 0 && self.failed == 0 && self.errors == 0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AtRestT9T10ScoreV1 {
    pub t9: AtRestFamilyCountsV1,
    pub t10: AtRestFamilyCountsV1,
}

pub fn validate_and_score_at_rest_report_v1(
    report: &AtRestQualificationReportV1,
) -> Result<AtRestT9T10ScoreV1, AtRestRejectV1> {
    if report.schema_version != AT_REST_QUALIFICATION_SCHEMA_V1 {
        return Err(AtRestRejectV1::UnsupportedSchemaVersion);
    }
    validate_sha256(&report.protocol_sha256)?;
    validate_sha256(&report.sut_profile_sha256)?;
    validate_sha256(&report.environment_profile_sha256)?;

    for control_id in [
        AtRestControlIdV1::C1NoOpReload,
        AtRestControlIdV1::C2GenuineWriteAfterRestart,
    ] {
        let receipt = report.controls.get(&control_id).ok_or(AtRestRejectV1::MissingControl)?;
        validate_control_event_trace(control_id, receipt)?;
        if !receipt.passed {
            return Err(AtRestRejectV1::ControlFailed);
        }
    }

    if report.cases.is_empty() {
        return Err(AtRestRejectV1::EmptyCaseSet);
    }

    let mut seen_runs = BTreeSet::new();
    let mut has_t9 = false;
    let mut has_t10 = false;
    let mut score = AtRestT9T10ScoreV1::default();

    for case in &report.cases {
        validate_atom(&case.case_run_id)?;
        if !seen_runs.insert(case.case_run_id.as_str()) {
            return Err(AtRestRejectV1::DuplicateCaseRunId);
        }
        match case.case_id {
            AtRestCaseIdV1::T9WholeStoreRollback => has_t9 = true,
            AtRestCaseIdV1::T10ErasureResurrection => has_t10 = true,
        }

        let family = match case.case_id {
            AtRestCaseIdV1::T9WholeStoreRollback => &mut score.t9,
            AtRestCaseIdV1::T10ErasureResurrection => &mut score.t10,
        };

        match &case.disposition {
            AtRestCaseDispositionV1::NotApplicable { reason } => {
                if reason.trim().is_empty() {
                    return Err(AtRestRejectV1::InvalidNotApplicableReason);
                }
                family.not_applicable += 1;
            }
            AtRestCaseDispositionV1::Error { reason } => {
                if reason.trim().is_empty() {
                    return Err(AtRestRejectV1::InvalidErrorReason);
                }
                family.errors += 1;
            }
            AtRestCaseDispositionV1::Observed { evidence, observation } => {
                validate_case_evidence(case.case_id, evidence, &case.case_run_id)?;
                validate_observation(observation)?;
                family.applicable += 1;
                match (observation.verdict, observation.detection_point) {
                    (AtRestVerdictV1::Rejected, AtRestDetectionPointV1::Read)
                    | (AtRestVerdictV1::Reported, AtRestDetectionPointV1::Read) => {
                        family.passed += 1;
                    }
                    (AtRestVerdictV1::Reported, AtRestDetectionPointV1::Audit) => {
                        family.partial += 1;
                    }
                    (AtRestVerdictV1::Accepted, AtRestDetectionPointV1::Read) => {
                        family.failed += 1;
                    }
                    _ => return Err(AtRestRejectV1::InvalidObservation),
                }
            }
        }
    }

    if !has_t9 {
        return Err(AtRestRejectV1::MissingT9);
    }
    if !has_t10 {
        return Err(AtRestRejectV1::MissingT10);
    }
    Ok(score)
}

fn validate_case_evidence(
    case_id: AtRestCaseIdV1,
    evidence: &AtRestCaseEvidenceV1,
    case_run_id: &str,
) -> Result<(), AtRestRejectV1> {
    match (case_id, evidence) {
        (AtRestCaseIdV1::T9WholeStoreRollback, AtRestCaseEvidenceV1::T9(value)) => {
            validate_t9_evidence(value, case_run_id)
        }
        (AtRestCaseIdV1::T10ErasureResurrection, AtRestCaseEvidenceV1::T10(value)) => {
            validate_t10_evidence(value, case_run_id)
        }
        _ => Err(AtRestRejectV1::EvidenceCaseMismatch),
    }
}

fn validate_t9_evidence(
    value: &T9WholeStoreRollbackEvidenceV1,
    case_run_id: &str,
) -> Result<(), AtRestRejectV1> {
    validate_snapshot_digests(
        &value.before_snapshot_sha256,
        &value.after_genuine_write_snapshot_sha256,
        &value.restored_snapshot_sha256,
    )?;
    validate_edit_delta(&value.expected_edit_delta_sha256, &value.observed_edit_delta_sha256)?;
    let event_trace_sha256 = validate_operation_event_trace(
        &value.events,
        case_run_id,
        &value.operation_receipt_sha256,
        AtRestCaseIdV1::T9WholeStoreRollback,
    )?;
    if event_trace_sha256 != value.event_trace_sha256 {
        return Err(AtRestRejectV1::EventTraceMismatch);
    }
    value.before_inventory.validate().map_err(|_| AtRestRejectV1::InventoryInvalid)?;
    value.after_genuine_write_inventory.validate().map_err(|_| AtRestRejectV1::InventoryInvalid)?;
    value.restored_inventory.validate().map_err(|_| AtRestRejectV1::InventoryInvalid)?;
    validate_atom(&value.new_record_id).map_err(|_| AtRestRejectV1::InvalidIdentifier)?;
    validate_sha256(&value.new_record_payload_sha256).map_err(|_| AtRestRejectV1::InvalidDigest)?;
    validate_sha256(&value.operation_receipt_sha256).map_err(|_| AtRestRejectV1::OperationReceiptInvalid)?;

    if !value.operation_performed_via_sut_write_path || !value.operation_after_restart {
        return Err(AtRestRejectV1::OperationNotPerformedThroughSut);
    }
    if !value.operation_acknowledged {
        return Err(AtRestRejectV1::OperationNotAcknowledged);
    }
    if !value.sut_stopped_during_restore {
        return Err(AtRestRejectV1::RestoreWhileSutRunning);
    }
    if !value.sut_restarted_after_restore {
        return Err(AtRestRejectV1::MissingRestartAfterRestore);
    }
    if !value.ordinary_read_path_invoked_after_restore {
        return Err(AtRestRejectV1::ReadPathNotInvoked);
    }
    if value.before_inventory.records.contains_key(&value.new_record_id) {
        return Err(AtRestRejectV1::ExpectedRecordAlreadyPresent);
    }

    let expected_after = inventory_with_added_record(
        &value.before_inventory,
        &value.new_record_id,
        &value.new_record_payload_sha256,
    );
    if value.after_genuine_write_inventory != expected_after {
        return Err(AtRestRejectV1::UnexpectedInventoryChange);
    }
    validate_restored_preimage(
        &value.before_snapshot_sha256,
        &value.restored_snapshot_sha256,
        &value.before_inventory,
        &value.restored_inventory,
    )
}

fn validate_t10_evidence(
    value: &T10ErasureResurrectionEvidenceV1,
    case_run_id: &str,
) -> Result<(), AtRestRejectV1> {
    validate_snapshot_digests(
        &value.before_snapshot_sha256,
        &value.after_erasure_snapshot_sha256,
        &value.restored_snapshot_sha256,
    )?;
    validate_edit_delta(&value.expected_edit_delta_sha256, &value.observed_edit_delta_sha256)?;
    let event_trace_sha256 = validate_operation_event_trace(
        &value.events,
        case_run_id,
        &value.erasure_receipt_sha256,
        AtRestCaseIdV1::T10ErasureResurrection,
    )?;
    if event_trace_sha256 != value.event_trace_sha256 {
        return Err(AtRestRejectV1::EventTraceMismatch);
    }
    value.before_inventory.validate().map_err(|_| AtRestRejectV1::InventoryInvalid)?;
    value.after_erasure_inventory.validate().map_err(|_| AtRestRejectV1::InventoryInvalid)?;
    value.restored_inventory.validate().map_err(|_| AtRestRejectV1::InventoryInvalid)?;
    validate_atom(&value.erased_record_id).map_err(|_| AtRestRejectV1::InvalidIdentifier)?;
    validate_sha256(&value.erased_record_payload_sha256).map_err(|_| AtRestRejectV1::InvalidDigest)?;
    validate_sha256(&value.erasure_receipt_sha256).map_err(|_| AtRestRejectV1::OperationReceiptInvalid)?;

    if !value.erasure_performed_via_sut_interface {
        return Err(AtRestRejectV1::OperationNotPerformedThroughSut);
    }
    if !value.erasure_acknowledged {
        return Err(AtRestRejectV1::OperationNotAcknowledged);
    }
    if !value.sut_stopped_during_restore {
        return Err(AtRestRejectV1::RestoreWhileSutRunning);
    }
    if !value.sut_restarted_after_restore {
        return Err(AtRestRejectV1::MissingRestartAfterRestore);
    }
    if !value.ordinary_read_path_invoked_after_restore {
        return Err(AtRestRejectV1::ReadPathNotInvoked);
    }

    match value.before_inventory.records.get(&value.erased_record_id) {
        Some(digest) if digest == &value.erased_record_payload_sha256 => {}
        Some(_) => return Err(AtRestRejectV1::ExpectedRecordDigestMismatch),
        None => return Err(AtRestRejectV1::ExpectedRecordMissing),
    }
    let expected_after = inventory_without_record(&value.before_inventory, &value.erased_record_id);
    if value.after_erasure_inventory != expected_after {
        return Err(AtRestRejectV1::UnexpectedInventoryChange);
    }
    validate_restored_preimage(
        &value.before_snapshot_sha256,
        &value.restored_snapshot_sha256,
        &value.before_inventory,
        &value.restored_inventory,
    )
}

fn validate_operation_event_trace(
    events: &[AtRestEvidenceEventV1],
    case_run_id: &str,
    operation_receipt_sha256: &str,
    case_id: AtRestCaseIdV1,
) -> Result<String, AtRestRejectV1> {
    validate_sha256(operation_receipt_sha256)
        .map_err(|_| AtRestRejectV1::OperationReceiptInvalid)?;

    let expected: &[(AtRestEvidenceEventKindV1, AtRestEvidenceEventSourceV1)] = match case_id {
        AtRestCaseIdV1::T9WholeStoreRollback => &[
            (AtRestEvidenceEventKindV1::SutRestartedBeforeOperation, AtRestEvidenceEventSourceV1::SutProcessObserver),
            (AtRestEvidenceEventKindV1::WriteRequested, AtRestEvidenceEventSourceV1::SutWriteApi),
            (AtRestEvidenceEventKindV1::WriteAcknowledged, AtRestEvidenceEventSourceV1::SutWriteApi),
            (AtRestEvidenceEventKindV1::SutStoppedForRestore, AtRestEvidenceEventSourceV1::SutProcessObserver),
            (AtRestEvidenceEventKindV1::RestoreApplied, AtRestEvidenceEventSourceV1::RestoreHarness),
            (AtRestEvidenceEventKindV1::SutRestartedAfterRestore, AtRestEvidenceEventSourceV1::SutProcessObserver),
            (AtRestEvidenceEventKindV1::OrdinaryReadCompleted, AtRestEvidenceEventSourceV1::SutReadApi),
        ],
        AtRestCaseIdV1::T10ErasureResurrection => &[
            (AtRestEvidenceEventKindV1::SutRestartedBeforeOperation, AtRestEvidenceEventSourceV1::SutProcessObserver),
            (AtRestEvidenceEventKindV1::ErasureRequested, AtRestEvidenceEventSourceV1::SutErasureApi),
            (AtRestEvidenceEventKindV1::ErasureAcknowledged, AtRestEvidenceEventSourceV1::SutErasureApi),
            (AtRestEvidenceEventKindV1::SutStoppedForRestore, AtRestEvidenceEventSourceV1::SutProcessObserver),
            (AtRestEvidenceEventKindV1::RestoreApplied, AtRestEvidenceEventSourceV1::RestoreHarness),
            (AtRestEvidenceEventKindV1::SutRestartedAfterRestore, AtRestEvidenceEventSourceV1::SutProcessObserver),
            (AtRestEvidenceEventKindV1::OrdinaryReadCompleted, AtRestEvidenceEventSourceV1::SutReadApi),
        ],
    };

    let trace_sha256 = validate_event_sequence(events, case_run_id, expected)?;
    let acknowledgement = events.get(2).ok_or(AtRestRejectV1::EventTraceMismatch)?;
    if acknowledgement.receipt_sha256 != operation_receipt_sha256 {
        return Err(AtRestRejectV1::OperationReceiptInvalid);
    }
    Ok(trace_sha256)
}

fn validate_event_sequence(
    events: &[AtRestEvidenceEventV1],
    case_run_id: &str,
    expected: &[(AtRestEvidenceEventKindV1, AtRestEvidenceEventSourceV1)],
) -> Result<String, AtRestRejectV1> {
    if events.len() != expected.len() {
        return Err(AtRestRejectV1::EventTraceMismatch);
    }
    let mut seen_ids = BTreeSet::new();
    let mut prior_time: Option<u64> = None;
    let mut clock_profile: Option<&str> = None;
    for (index, (event, expected_event)) in events.iter().zip(expected.iter()).enumerate() {
        if event.sequence != index as u64
            || event.case_run_id != case_run_id
            || event.kind != expected_event.0
            || event.source != expected_event.1
        {
            return Err(AtRestRejectV1::EventBindingMismatch);
        }
        validate_atom(&event.event_id).map_err(|_| AtRestRejectV1::EventBindingMismatch)?;
        if !seen_ids.insert(event.event_id.as_str()) {
            return Err(AtRestRejectV1::DuplicateEventId);
        }
        validate_sha256(&event.clock_profile_sha256)
            .map_err(|_| AtRestRejectV1::EventClockProfileMismatch)?;
        validate_sha256(&event.receipt_sha256)
            .map_err(|_| AtRestRejectV1::EventReceiptInvalid)?;
        if let Some(expected_clock) = clock_profile {
            if expected_clock != event.clock_profile_sha256 {
                return Err(AtRestRejectV1::EventClockProfileMismatch);
            }
        } else {
            clock_profile = Some(&event.clock_profile_sha256);
        }
        if prior_time.is_some_and(|prior| event.monotonic_elapsed_ns <= prior) {
            return Err(AtRestRejectV1::EventTimeNotMonotonic);
        }
        prior_time = Some(event.monotonic_elapsed_ns);
    }
    event_trace_sha256_v1(events)
}

fn validate_control_event_trace(
    control_id: AtRestControlIdV1,
    receipt: &AtRestControlReceiptV1,
) -> Result<(), AtRestRejectV1> {
    validate_atom(&receipt.case_run_id).map_err(|_| AtRestRejectV1::ControlEventTraceInvalid)?;
    let expected: &[(AtRestEvidenceEventKindV1, AtRestEvidenceEventSourceV1)] = match control_id {
        AtRestControlIdV1::C1NoOpReload => &[
            (AtRestEvidenceEventKindV1::C1SutStoppedForNoOpReload, AtRestEvidenceEventSourceV1::SutProcessObserver),
            (AtRestEvidenceEventKindV1::C1SutRestartedAfterNoOpReload, AtRestEvidenceEventSourceV1::SutProcessObserver),
            (AtRestEvidenceEventKindV1::C1SeededReadCompleted, AtRestEvidenceEventSourceV1::SutReadApi),
            (AtRestEvidenceEventKindV1::C1InventoryUnchanged, AtRestEvidenceEventSourceV1::StoreInventoryObserver),
        ],
        AtRestControlIdV1::C2GenuineWriteAfterRestart => &[
            (AtRestEvidenceEventKindV1::C2SutRestartedBeforeWrite, AtRestEvidenceEventSourceV1::SutProcessObserver),
            (AtRestEvidenceEventKindV1::C2WriteRequested, AtRestEvidenceEventSourceV1::SutWriteApi),
            (AtRestEvidenceEventKindV1::C2WriteAcknowledged, AtRestEvidenceEventSourceV1::SutWriteApi),
            (AtRestEvidenceEventKindV1::C2ReadbackCompleted, AtRestEvidenceEventSourceV1::SutReadApi),
        ],
    };
    let actual = validate_event_sequence(&receipt.events, &receipt.case_run_id, expected)
        .map_err(|_| AtRestRejectV1::ControlEventTraceInvalid)?;
    if actual != receipt.evidence_sha256 {
        return Err(AtRestRejectV1::ControlEvidenceDigestMismatch);
    }
    Ok(())
}

fn event_trace_sha256_v1(
    events: &[AtRestEvidenceEventV1],
) -> Result<String, AtRestRejectV1> {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:at-rest-event-trace:v1\0");
    hasher.update((events.len() as u64).to_be_bytes());
    for event in events {
        hasher.update(event.sequence.to_be_bytes());
        hash_field(&mut hasher, &event.case_run_id);
        hash_field(&mut hasher, &event.event_id);
        hasher.update([event_kind_tag(event.kind)]);
        hasher.update([event_source_tag(event.source)]);
        hasher.update(event.monotonic_elapsed_ns.to_be_bytes());
        hasher.update(decode_sha256(&event.clock_profile_sha256)?);
        hasher.update(decode_sha256(&event.receipt_sha256)?);
    }
    Ok(hex_digest(&hasher.finalize()))
}

fn event_kind_tag(value: AtRestEvidenceEventKindV1) -> u8 {
    match value {
        AtRestEvidenceEventKindV1::SutRestartedBeforeOperation => 1,
        AtRestEvidenceEventKindV1::WriteRequested => 2,
        AtRestEvidenceEventKindV1::WriteAcknowledged => 3,
        AtRestEvidenceEventKindV1::ErasureRequested => 4,
        AtRestEvidenceEventKindV1::ErasureAcknowledged => 5,
        AtRestEvidenceEventKindV1::SutStoppedForRestore => 6,
        AtRestEvidenceEventKindV1::RestoreApplied => 7,
        AtRestEvidenceEventKindV1::SutRestartedAfterRestore => 8,
        AtRestEvidenceEventKindV1::OrdinaryReadCompleted => 9,
        AtRestEvidenceEventKindV1::C1SutStoppedForNoOpReload => 10,
        AtRestEvidenceEventKindV1::C1SutRestartedAfterNoOpReload => 11,
        AtRestEvidenceEventKindV1::C1SeededReadCompleted => 12,
        AtRestEvidenceEventKindV1::C1InventoryUnchanged => 13,
        AtRestEvidenceEventKindV1::C2SutRestartedBeforeWrite => 14,
        AtRestEvidenceEventKindV1::C2WriteRequested => 15,
        AtRestEvidenceEventKindV1::C2WriteAcknowledged => 16,
        AtRestEvidenceEventKindV1::C2ReadbackCompleted => 17,
    }
}

fn event_source_tag(value: AtRestEvidenceEventSourceV1) -> u8 {
    match value {
        AtRestEvidenceEventSourceV1::SutProcessObserver => 1,
        AtRestEvidenceEventSourceV1::SutWriteApi => 2,
        AtRestEvidenceEventSourceV1::SutErasureApi => 3,
        AtRestEvidenceEventSourceV1::RestoreHarness => 4,
        AtRestEvidenceEventSourceV1::SutReadApi => 5,
        AtRestEvidenceEventSourceV1::StoreInventoryObserver => 6,
    }
}

fn validate_edit_delta(expected: &str, observed: &str) -> Result<(), AtRestRejectV1> {
    validate_sha256(expected).map_err(|_| AtRestRejectV1::InvalidDigest)?;
    validate_sha256(observed).map_err(|_| AtRestRejectV1::InvalidDigest)?;
    if expected != observed {
        return Err(AtRestRejectV1::EditDeltaMismatch);
    }
    Ok(())
}

fn validate_snapshot_digests(
    before: &str,
    after_operation: &str,
    restored: &str,
) -> Result<(), AtRestRejectV1> {
    validate_sha256(before).map_err(|_| AtRestRejectV1::SnapshotDigestInvalid)?;
    validate_sha256(after_operation).map_err(|_| AtRestRejectV1::SnapshotDigestInvalid)?;
    validate_sha256(restored).map_err(|_| AtRestRejectV1::SnapshotDigestInvalid)?;
    if before == after_operation {
        return Err(AtRestRejectV1::SnapshotDidNotChange);
    }
    Ok(())
}

fn validate_restored_preimage(
    before_sha256: &str,
    restored_sha256: &str,
    before_inventory: &StoreInventoryV1,
    restored_inventory: &StoreInventoryV1,
) -> Result<(), AtRestRejectV1> {
    if before_sha256 != restored_sha256 || before_inventory != restored_inventory {
        return Err(AtRestRejectV1::RestoredSnapshotMismatch);
    }
    Ok(())
}

fn inventory_with_added_record(
    before: &StoreInventoryV1,
    record_id: &str,
    payload_sha256: &str,
) -> StoreInventoryV1 {
    let mut records = before.records.clone();
    records.insert(record_id.to_owned(), payload_sha256.to_owned());
    StoreInventoryV1 { records }
}

fn inventory_without_record(
    before: &StoreInventoryV1,
    record_id: &str,
) -> StoreInventoryV1 {
    let mut records = before.records.clone();
    records.remove(record_id);
    StoreInventoryV1 { records }
}

fn validate_observation(value: &AtRestObservationV1) -> Result<(), AtRestRejectV1> {
    match (value.verdict, value.detection_point, value.observation_class) {
        (
            AtRestVerdictV1::Rejected,
            AtRestDetectionPointV1::Read,
            AtRestObservationClassV1::DutBoundary | AtRestObservationClassV1::EndpointSoftwareBoundary,
        )
        | (
            AtRestVerdictV1::Reported,
            AtRestDetectionPointV1::Read,
            AtRestObservationClassV1::DutBoundary | AtRestObservationClassV1::EndpointSoftwareBoundary,
        )
        | (
            AtRestVerdictV1::Reported,
            AtRestDetectionPointV1::Audit,
            AtRestObservationClassV1::DutReported,
        )
        | (
            AtRestVerdictV1::Accepted,
            AtRestDetectionPointV1::Read,
            AtRestObservationClassV1::DutBoundary | AtRestObservationClassV1::EndpointSoftwareBoundary,
        ) => Ok(()),
        _ => Err(AtRestRejectV1::InvalidObservation),
    }
}

fn validate_atom(value: &str) -> Result<(), AtRestRejectV1> {
    if value.is_empty()
        || value.len() > 256
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(AtRestRejectV1::InvalidIdentifier);
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), AtRestRejectV1> {
    if value.len() != 64
        || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(AtRestRejectV1::InvalidDigest);
    }
    Ok(())
}

fn decode_sha256(value: &str) -> Result<[u8; 32], AtRestRejectV1> {
    validate_sha256(value)?;
    let mut output = [0u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        output[index] = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
    }
    Ok(output)
}

fn hex_nibble(value: u8) -> Result<u8, AtRestRejectV1> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => Err(AtRestRejectV1::InvalidDigest),
    }
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn inventory(records: &[(&str, char)]) -> StoreInventoryV1 {
        StoreInventoryV1 {
            records: records
                .iter()
                .map(|(id, ch)| ((*id).to_owned(), digest(*ch)))
                .collect(),
        }
    }

    fn event_trace(
        case_run_id: &str,
        case_id: AtRestCaseIdV1,
        operation_receipt_sha256: &str,
    ) -> Vec<AtRestEvidenceEventV1> {
        let expected: &[(AtRestEvidenceEventKindV1, AtRestEvidenceEventSourceV1)] = match case_id {
            AtRestCaseIdV1::T9WholeStoreRollback => &[
                (AtRestEvidenceEventKindV1::SutRestartedBeforeOperation, AtRestEvidenceEventSourceV1::SutProcessObserver),
                (AtRestEvidenceEventKindV1::WriteRequested, AtRestEvidenceEventSourceV1::SutWriteApi),
                (AtRestEvidenceEventKindV1::WriteAcknowledged, AtRestEvidenceEventSourceV1::SutWriteApi),
                (AtRestEvidenceEventKindV1::SutStoppedForRestore, AtRestEvidenceEventSourceV1::SutProcessObserver),
                (AtRestEvidenceEventKindV1::RestoreApplied, AtRestEvidenceEventSourceV1::RestoreHarness),
                (AtRestEvidenceEventKindV1::SutRestartedAfterRestore, AtRestEvidenceEventSourceV1::SutProcessObserver),
                (AtRestEvidenceEventKindV1::OrdinaryReadCompleted, AtRestEvidenceEventSourceV1::SutReadApi),
            ],
            AtRestCaseIdV1::T10ErasureResurrection => &[
                (AtRestEvidenceEventKindV1::SutRestartedBeforeOperation, AtRestEvidenceEventSourceV1::SutProcessObserver),
                (AtRestEvidenceEventKindV1::ErasureRequested, AtRestEvidenceEventSourceV1::SutErasureApi),
                (AtRestEvidenceEventKindV1::ErasureAcknowledged, AtRestEvidenceEventSourceV1::SutErasureApi),
                (AtRestEvidenceEventKindV1::SutStoppedForRestore, AtRestEvidenceEventSourceV1::SutProcessObserver),
                (AtRestEvidenceEventKindV1::RestoreApplied, AtRestEvidenceEventSourceV1::RestoreHarness),
                (AtRestEvidenceEventKindV1::SutRestartedAfterRestore, AtRestEvidenceEventSourceV1::SutProcessObserver),
                (AtRestEvidenceEventKindV1::OrdinaryReadCompleted, AtRestEvidenceEventSourceV1::SutReadApi),
            ],
        };
        expected
            .iter()
            .enumerate()
            .map(|(index, (kind, source))| AtRestEvidenceEventV1 {
                sequence: index as u64,
                case_run_id: case_run_id.to_owned(),
                event_id: format!("{case_run_id}-event-{index}"),
                kind: *kind,
                source: *source,
                monotonic_elapsed_ns: ((index as u64) + 1) * 10,
                clock_profile_sha256: digest('c'),
                receipt_sha256: if index == 2 {
                    operation_receipt_sha256.to_owned()
                } else {
                    digest(char::from(b'a' + index as u8))
                },
            })
            .collect()
    }

    fn t9_evidence(case_run_id: &str) -> T9WholeStoreRollbackEvidenceV1 {
        let before = inventory(&[("a", 'a'), ("b", 'b'), ("c", 'c')]);
        let after = inventory(&[("a", 'a'), ("b", 'b'), ("c", 'c'), ("d", 'd')]);
        let events = event_trace(case_run_id, AtRestCaseIdV1::T9WholeStoreRollback, &digest('e'));
        let event_trace_sha256 = event_trace_sha256_v1(&events).unwrap();
        T9WholeStoreRollbackEvidenceV1 {
            before_snapshot_sha256: digest('1'),
            after_genuine_write_snapshot_sha256: digest('2'),
            restored_snapshot_sha256: digest('1'),
            expected_edit_delta_sha256: digest('7'),
            observed_edit_delta_sha256: digest('7'),
            event_trace_sha256,
            events,
            before_inventory: before.clone(),
            after_genuine_write_inventory: after,
            restored_inventory: before,
            new_record_id: "d".to_owned(),
            new_record_payload_sha256: digest('d'),
            operation_receipt_sha256: digest('e'),
            operation_performed_via_sut_write_path: true,
            operation_after_restart: true,
            operation_acknowledged: true,
            sut_stopped_during_restore: true,
            sut_restarted_after_restore: true,
            ordinary_read_path_invoked_after_restore: true,
        }
    }

    fn t10_evidence(case_run_id: &str) -> T10ErasureResurrectionEvidenceV1 {
        let before = inventory(&[("a", 'a'), ("b", 'b'), ("c", 'c')]);
        let after = inventory(&[("a", 'a'), ("c", 'c')]);
        let events = event_trace(case_run_id, AtRestCaseIdV1::T10ErasureResurrection, &digest('f'));
        let event_trace_sha256 = event_trace_sha256_v1(&events).unwrap();
        T10ErasureResurrectionEvidenceV1 {
            before_snapshot_sha256: digest('1'),
            after_erasure_snapshot_sha256: digest('3'),
            restored_snapshot_sha256: digest('1'),
            expected_edit_delta_sha256: digest('8'),
            observed_edit_delta_sha256: digest('8'),
            event_trace_sha256,
            events,
            before_inventory: before.clone(),
            after_erasure_inventory: after,
            restored_inventory: before,
            erased_record_id: "b".to_owned(),
            erased_record_payload_sha256: digest('b'),
            erasure_receipt_sha256: digest('f'),
            erasure_performed_via_sut_interface: true,
            erasure_acknowledged: true,
            sut_stopped_during_restore: true,
            sut_restarted_after_restore: true,
            ordinary_read_path_invoked_after_restore: true,
        }
    }

    fn control_event_trace(control_id: AtRestControlIdV1, case_run_id: &str) -> Vec<AtRestEvidenceEventV1> {
        let expected: &[(AtRestEvidenceEventKindV1, AtRestEvidenceEventSourceV1)] = match control_id {
            AtRestControlIdV1::C1NoOpReload => &[
                (AtRestEvidenceEventKindV1::C1SutStoppedForNoOpReload, AtRestEvidenceEventSourceV1::SutProcessObserver),
                (AtRestEvidenceEventKindV1::C1SutRestartedAfterNoOpReload, AtRestEvidenceEventSourceV1::SutProcessObserver),
                (AtRestEvidenceEventKindV1::C1SeededReadCompleted, AtRestEvidenceEventSourceV1::SutReadApi),
                (AtRestEvidenceEventKindV1::C1InventoryUnchanged, AtRestEvidenceEventSourceV1::StoreInventoryObserver),
            ],
            AtRestControlIdV1::C2GenuineWriteAfterRestart => &[
                (AtRestEvidenceEventKindV1::C2SutRestartedBeforeWrite, AtRestEvidenceEventSourceV1::SutProcessObserver),
                (AtRestEvidenceEventKindV1::C2WriteRequested, AtRestEvidenceEventSourceV1::SutWriteApi),
                (AtRestEvidenceEventKindV1::C2WriteAcknowledged, AtRestEvidenceEventSourceV1::SutWriteApi),
                (AtRestEvidenceEventKindV1::C2ReadbackCompleted, AtRestEvidenceEventSourceV1::SutReadApi),
            ],
        };
        expected.iter().enumerate().map(|(index, (kind, source))| AtRestEvidenceEventV1 {
            sequence: index as u64,
            case_run_id: case_run_id.to_owned(),
            event_id: format!("{case_run_id}-event-{index}"),
            kind: *kind,
            source: *source,
            monotonic_elapsed_ns: ((index as u64) + 1) * 10,
            clock_profile_sha256: digest('c'),
            receipt_sha256: digest(char::from(b'a' + index as u8)),
        }).collect()
    }

    fn report(
        t9_observation: AtRestObservationV1,
        t10_observation: AtRestObservationV1,
    ) -> AtRestQualificationReportV1 {
        AtRestQualificationReportV1 {
            schema_version: AT_REST_QUALIFICATION_SCHEMA_V1.to_owned(),
            protocol_sha256: digest('a'),
            sut_profile_sha256: digest('b'),
            environment_profile_sha256: digest('c'),
            controls: BTreeMap::from([
                (
                    AtRestControlIdV1::C1NoOpReload,
                    {
                        let events = control_event_trace(AtRestControlIdV1::C1NoOpReload, "c1-run-1");
                        AtRestControlReceiptV1 {
                            passed: true,
                            case_run_id: "c1-run-1".to_owned(),
                            evidence_sha256: event_trace_sha256_v1(&events).unwrap(),
                            events,
                        }
                    },
                ),
                (
                    AtRestControlIdV1::C2GenuineWriteAfterRestart,
                    {
                        let events = control_event_trace(AtRestControlIdV1::C2GenuineWriteAfterRestart, "c2-run-1");
                        AtRestControlReceiptV1 {
                            passed: true,
                            case_run_id: "c2-run-1".to_owned(),
                            evidence_sha256: event_trace_sha256_v1(&events).unwrap(),
                            events,
                        }
                    },
                ),
            ]),
            cases: vec![
                AtRestCaseRecordV1 {
                    case_id: AtRestCaseIdV1::T9WholeStoreRollback,
                    case_run_id: "t9-run-1".to_owned(),
                    disposition: AtRestCaseDispositionV1::Observed {
                        evidence: AtRestCaseEvidenceV1::T9(t9_evidence("t9-run-1")),
                        observation: t9_observation,
                    },
                },
                AtRestCaseRecordV1 {
                    case_id: AtRestCaseIdV1::T10ErasureResurrection,
                    case_run_id: "t10-run-1".to_owned(),
                    disposition: AtRestCaseDispositionV1::Observed {
                        evidence: AtRestCaseEvidenceV1::T10(t10_evidence("t10-run-1")),
                        observation: t10_observation,
                    },
                },
            ],
        }
    }

    fn rejected_on_read() -> AtRestObservationV1 {
        AtRestObservationV1 {
            verdict: AtRestVerdictV1::Rejected,
            detection_point: AtRestDetectionPointV1::Read,
            observation_class: AtRestObservationClassV1::DutBoundary,
        }
    }

    fn accepted_on_read() -> AtRestObservationV1 {
        AtRestObservationV1 {
            verdict: AtRestVerdictV1::Accepted,
            detection_point: AtRestDetectionPointV1::Read,
            observation_class: AtRestObservationClassV1::DutBoundary,
        }
    }

    #[test]
    fn valid_report_scores_t9_and_t10_separately() {
        let value = report(rejected_on_read(), rejected_on_read());
        let score = validate_and_score_at_rest_report_v1(&value).unwrap();
        assert_eq!(score.t9.applicable, 1);
        assert_eq!(score.t9.passed, 1);
        assert_eq!(score.t10.applicable, 1);
        assert_eq!(score.t10.passed, 1);
        assert_eq!(score.t9.metric_value(), Some(1.0));
        assert_eq!(score.t10.metric_value(), Some(1.0));
        assert!(score.t9.zero_acceptance_gate());
        assert!(score.t10.zero_acceptance_gate());
    }

    #[test]
    fn t9_and_t10_failures_cannot_be_hidden_by_a_combined_score() {
        let value = report(rejected_on_read(), accepted_on_read());
        let score = validate_and_score_at_rest_report_v1(&value).unwrap();
        assert_eq!(score.t9.metric_value(), Some(1.0));
        assert_eq!(score.t10.metric_value(), Some(0.0));
        assert!(!score.t10.zero_acceptance_gate());
    }

    #[test]
    fn audit_only_report_is_partial_not_a_read_path_pass() {
        let audit = AtRestObservationV1 {
            verdict: AtRestVerdictV1::Reported,
            detection_point: AtRestDetectionPointV1::Audit,
            observation_class: AtRestObservationClassV1::DutReported,
        };
        let score = validate_and_score_at_rest_report_v1(&report(rejected_on_read(), audit)).unwrap();
        assert_eq!(score.t10.partial, 1);
        assert_eq!(score.t10.passed, 0);
        assert_eq!(score.t10.metric_value(), Some(0.5));
    }

    #[test]
    fn failed_controls_invalidate_the_result_instead_of_scoring_it() {
        let mut value = report(rejected_on_read(), rejected_on_read());
        value.controls.get_mut(&AtRestControlIdV1::C1NoOpReload).unwrap().passed = false;
        assert_eq!(
            validate_and_score_at_rest_report_v1(&value),
            Err(AtRestRejectV1::ControlFailed)
        );
    }

    #[test]
    fn missing_control_is_not_treated_as_a_pass() {
        let mut value = report(rejected_on_read(), rejected_on_read());
        value.controls.remove(&AtRestControlIdV1::C2GenuineWriteAfterRestart);
        assert_eq!(
            validate_and_score_at_rest_report_v1(&value),
            Err(AtRestRejectV1::MissingControl)
        );
    }

    #[test]
    fn mandatory_controls_require_run_bound_ordered_source_events() {
        let mut value = report(rejected_on_read(), rejected_on_read());
        value.controls.get_mut(&AtRestControlIdV1::C1NoOpReload).unwrap().events.swap(0, 1);
        assert_eq!(
            validate_and_score_at_rest_report_v1(&value),
            Err(AtRestRejectV1::ControlEventTraceInvalid)
        );

        let mut value = report(rejected_on_read(), rejected_on_read());
        value.controls.get_mut(&AtRestControlIdV1::C2GenuineWriteAfterRestart).unwrap()
            .events[2].case_run_id = "other-control-run".to_owned();
        assert_eq!(
            validate_and_score_at_rest_report_v1(&value),
            Err(AtRestRejectV1::ControlEventTraceInvalid)
        );
    }

    #[test]
    fn control_evidence_digest_commits_to_the_complete_event_trace() {
        let mut value = report(rejected_on_read(), rejected_on_read());
        value.controls.get_mut(&AtRestControlIdV1::C2GenuineWriteAfterRestart).unwrap()
            .events[3].receipt_sha256 = digest('9');
        assert_eq!(
            validate_and_score_at_rest_report_v1(&value),
            Err(AtRestRejectV1::ControlEvidenceDigestMismatch)
        );
    }

    #[test]
    fn event_trace_digest_binds_every_event_field() {
        let mut value = report(rejected_on_read(), rejected_on_read());
        if let AtRestCaseDispositionV1::Observed {
            evidence: AtRestCaseEvidenceV1::T9(evidence),
            ..
        } = &mut value.cases[0].disposition {
            evidence.events[0].receipt_sha256 = digest('9');
        }
        assert_eq!(
            validate_and_score_at_rest_report_v1(&value),
            Err(AtRestRejectV1::EventTraceMismatch)
        );
    }

    #[test]
    fn event_trace_rejects_reordered_unbound_or_wrong_source_events() {
        let mut value = report(rejected_on_read(), rejected_on_read());
        if let AtRestCaseDispositionV1::Observed {
            evidence: AtRestCaseEvidenceV1::T10(evidence),
            ..
        } = &mut value.cases[1].disposition {
            evidence.events.swap(3, 4);
        }
        assert_eq!(
            validate_and_score_at_rest_report_v1(&value),
            Err(AtRestRejectV1::EventBindingMismatch)
        );

        let mut value = report(rejected_on_read(), rejected_on_read());
        if let AtRestCaseDispositionV1::Observed {
            evidence: AtRestCaseEvidenceV1::T10(evidence),
            ..
        } = &mut value.cases[1].disposition {
            evidence.events[4].case_run_id = "other-run".to_owned();
        }
        assert_eq!(
            validate_and_score_at_rest_report_v1(&value),
            Err(AtRestRejectV1::EventBindingMismatch)
        );

        let mut value = report(rejected_on_read(), rejected_on_read());
        if let AtRestCaseDispositionV1::Observed {
            evidence: AtRestCaseEvidenceV1::T10(evidence),
            ..
        } = &mut value.cases[1].disposition {
            evidence.events[1].source = AtRestEvidenceEventSourceV1::RestoreHarness;
        }
        assert_eq!(
            validate_and_score_at_rest_report_v1(&value),
            Err(AtRestRejectV1::EventBindingMismatch)
        );
    }

    #[test]
    fn event_trace_rejects_non_monotonic_times_duplicate_ids_and_receipt_rebinding() {
        let mut value = report(rejected_on_read(), rejected_on_read());
        if let AtRestCaseDispositionV1::Observed {
            evidence: AtRestCaseEvidenceV1::T9(evidence),
            ..
        } = &mut value.cases[0].disposition {
            evidence.events[4].monotonic_elapsed_ns = evidence.events[3].monotonic_elapsed_ns;
        }
        assert_eq!(
            validate_and_score_at_rest_report_v1(&value),
            Err(AtRestRejectV1::EventTimeNotMonotonic)
        );

        let mut value = report(rejected_on_read(), rejected_on_read());
        if let AtRestCaseDispositionV1::Observed {
            evidence: AtRestCaseEvidenceV1::T9(evidence),
            ..
        } = &mut value.cases[0].disposition {
            evidence.events[4].event_id = evidence.events[3].event_id.clone();
        }
        assert_eq!(
            validate_and_score_at_rest_report_v1(&value),
            Err(AtRestRejectV1::DuplicateEventId)
        );

        let mut value = report(rejected_on_read(), rejected_on_read());
        if let AtRestCaseDispositionV1::Observed {
            evidence: AtRestCaseEvidenceV1::T9(evidence),
            ..
        } = &mut value.cases[0].disposition {
            evidence.events[2].receipt_sha256 = digest('9');
        }
        assert_eq!(
            validate_and_score_at_rest_report_v1(&value),
            Err(AtRestRejectV1::OperationReceiptInvalid)
        );
    }

    #[test]
    fn raw_storage_edit_delta_must_match_the_expected_single_operation() {
        let mut value = report(rejected_on_read(), rejected_on_read());
        if let AtRestCaseDispositionV1::Observed {
            evidence: AtRestCaseEvidenceV1::T10(evidence),
            ..
        } = &mut value.cases[1].disposition {
            evidence.observed_edit_delta_sha256 = digest('9');
        }
        assert_eq!(
            validate_and_score_at_rest_report_v1(&value),
            Err(AtRestRejectV1::EditDeltaMismatch)
        );
    }

    #[test]
    fn t10_requires_a_real_erasure_acknowledgement_and_exact_removal() {
        let mut value = report(rejected_on_read(), rejected_on_read());
        if let AtRestCaseDispositionV1::Observed {
            evidence: AtRestCaseEvidenceV1::T10(evidence),
            ..
        } = &mut value.cases[1].disposition {
            evidence.erasure_acknowledged = false;
        }
        assert_eq!(
            validate_and_score_at_rest_report_v1(&value),
            Err(AtRestRejectV1::OperationNotAcknowledged)
        );

        let mut value = report(rejected_on_read(), rejected_on_read());
        if let AtRestCaseDispositionV1::Observed {
            evidence: AtRestCaseEvidenceV1::T10(evidence),
            ..
        } = &mut value.cases[1].disposition {
            evidence.after_erasure_inventory.records.insert("a".to_owned(), digest('0'));
        }
        assert_eq!(
            validate_and_score_at_rest_report_v1(&value),
            Err(AtRestRejectV1::UnexpectedInventoryChange)
        );
    }

    #[test]
    fn t9_requires_exactly_one_new_record_and_restoration_of_pre_write_image() {
        let mut value = report(rejected_on_read(), rejected_on_read());
        if let AtRestCaseDispositionV1::Observed {
            evidence: AtRestCaseEvidenceV1::T9(evidence),
            ..
        } = &mut value.cases[0].disposition {
            evidence.after_genuine_write_inventory.records.insert("side-effect".to_owned(), digest('0'));
        }
        assert_eq!(
            validate_and_score_at_rest_report_v1(&value),
            Err(AtRestRejectV1::UnexpectedInventoryChange)
        );

        let mut value = report(rejected_on_read(), rejected_on_read());
        if let AtRestCaseDispositionV1::Observed {
            evidence: AtRestCaseEvidenceV1::T9(evidence),
            ..
        } = &mut value.cases[0].disposition {
            evidence.restored_snapshot_sha256 = digest('9');
        }
        assert_eq!(
            validate_and_score_at_rest_report_v1(&value),
            Err(AtRestRejectV1::RestoredSnapshotMismatch)
        );
    }

    #[test]
    fn errors_and_not_applicable_are_excluded_not_credited() {
        let mut value = report(rejected_on_read(), rejected_on_read());
        value.cases[1].disposition = AtRestCaseDispositionV1::NotApplicable {
            reason: "SUT has no restorable backup interface".to_owned(),
        };
        let score = validate_and_score_at_rest_report_v1(&value).unwrap();
        assert_eq!(score.t10.applicable, 0);
        assert_eq!(score.t10.not_applicable, 1);
        assert_eq!(score.t10.metric_value(), None);
        assert!(!score.t10.zero_acceptance_gate());

        value.cases[1].disposition = AtRestCaseDispositionV1::Error {
            reason: "restore adapter failed to apply the edit".to_owned(),
        };
        let score = validate_and_score_at_rest_report_v1(&value).unwrap();
        assert_eq!(score.t10.errors, 1);
        assert_eq!(score.t10.applicable, 0);
        assert!(!score.t10.zero_acceptance_gate());
    }

    #[test]
    fn duplicate_runs_and_duplicate_case_ids_do_not_collapse_into_one_result() {
        let mut value = report(rejected_on_read(), rejected_on_read());
        value.cases[1].case_run_id = value.cases[0].case_run_id.clone();
        assert_eq!(
            validate_and_score_at_rest_report_v1(&value),
            Err(AtRestRejectV1::DuplicateCaseRunId)
        );
    }

    #[test]
    fn wrong_observation_class_for_audit_is_rejected() {
        let mut value = report(rejected_on_read(), AtRestObservationV1 {
            verdict: AtRestVerdictV1::Reported,
            detection_point: AtRestDetectionPointV1::Audit,
            observation_class: AtRestObservationClassV1::DutBoundary,
        });
        assert_eq!(
            validate_and_score_at_rest_report_v1(&value),
            Err(AtRestRejectV1::InvalidObservation)
        );
    }
}
