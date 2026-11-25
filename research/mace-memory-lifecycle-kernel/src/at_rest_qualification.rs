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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AtRestControlReceiptV1 {
    pub passed: bool,
    pub evidence_sha256: String,
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
        self.applicable > 0 && self.failed == 0
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
        validate_sha256(&receipt.evidence_sha256)?;
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
                validate_case_evidence(case.case_id, evidence)?;
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
) -> Result<(), AtRestRejectV1> {
    match (case_id, evidence) {
        (AtRestCaseIdV1::T9WholeStoreRollback, AtRestCaseEvidenceV1::T9(value)) => {
            validate_t9_evidence(value)
        }
        (AtRestCaseIdV1::T10ErasureResurrection, AtRestCaseEvidenceV1::T10(value)) => {
            validate_t10_evidence(value)
        }
        _ => Err(AtRestRejectV1::EvidenceCaseMismatch),
    }
}

fn validate_t9_evidence(value: &T9WholeStoreRollbackEvidenceV1) -> Result<(), AtRestRejectV1> {
    validate_snapshot_digests(
        &value.before_snapshot_sha256,
        &value.after_genuine_write_snapshot_sha256,
        &value.restored_snapshot_sha256,
    )?;
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

fn validate_t10_evidence(value: &T10ErasureResurrectionEvidenceV1) -> Result<(), AtRestRejectV1> {
    validate_snapshot_digests(
        &value.before_snapshot_sha256,
        &value.after_erasure_snapshot_sha256,
        &value.restored_snapshot_sha256,
    )?;
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

    fn t9_evidence() -> T9WholeStoreRollbackEvidenceV1 {
        let before = inventory(&[("a", 'a'), ("b", 'b'), ("c", 'c')]);
        let after = inventory(&[("a", 'a'), ("b", 'b'), ("c", 'c'), ("d", 'd')]);
        T9WholeStoreRollbackEvidenceV1 {
            before_snapshot_sha256: digest('1'),
            after_genuine_write_snapshot_sha256: digest('2'),
            restored_snapshot_sha256: digest('1'),
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

    fn t10_evidence() -> T10ErasureResurrectionEvidenceV1 {
        let before = inventory(&[("a", 'a'), ("b", 'b'), ("c", 'c')]);
        let after = inventory(&[("a", 'a'), ("c", 'c')]);
        T10ErasureResurrectionEvidenceV1 {
            before_snapshot_sha256: digest('1'),
            after_erasure_snapshot_sha256: digest('3'),
            restored_snapshot_sha256: digest('1'),
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
                    AtRestControlReceiptV1 { passed: true, evidence_sha256: digest('d') },
                ),
                (
                    AtRestControlIdV1::C2GenuineWriteAfterRestart,
                    AtRestControlReceiptV1 { passed: true, evidence_sha256: digest('e') },
                ),
            ]),
            cases: vec![
                AtRestCaseRecordV1 {
                    case_id: AtRestCaseIdV1::T9WholeStoreRollback,
                    case_run_id: "t9-run-1".to_owned(),
                    disposition: AtRestCaseDispositionV1::Observed {
                        evidence: AtRestCaseEvidenceV1::T9(t9_evidence()),
                        observation: t9_observation,
                    },
                },
                AtRestCaseRecordV1 {
                    case_id: AtRestCaseIdV1::T10ErasureResurrection,
                    case_run_id: "t10-run-1".to_owned(),
                    disposition: AtRestCaseDispositionV1::Observed {
                        evidence: AtRestCaseEvidenceV1::T10(t10_evidence()),
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
