// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1E: blinded collection, append-only raw response evidence, and
//! collection-close contracts for the first perceptual study.
//!
//! Collection records only participant-facing choices (`A`/`B`, `left`/`right`).
//! It deliberately does not compute correctness, intervention direction, or any
//! arm-labelled outcome while collection is open. Those require the P1D private
//! audit and belong after collection close/unblinding authority.

use crate::evidence_digest::{
    canonical_json_sha256,
    perceptual_participant_schedule::{
        PerceptualCohortSlotsV1, PerceptualParticipantScheduleBookV1,
        PublicParticipantScheduleV1, TaskBlockOrderV1,
    },
    perceptual_stimulus_pack::{
        FrozenC6fRenderSubjectBindingV1, FrozenPerceptualStimulusPackV1,
    },
    perceptual_study_protocol::{
        FrozenPerceptualStudyProtocolV1, PerceptualTaskV1,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PERCEPTUAL_COLLECTION_AUTHORITY_VERSION: &str =
    "mel003-perceptual-collection-authority-v1";
pub const PERCEPTUAL_SESSION_EVIDENCE_VERSION: &str =
    "mel003-perceptual-session-evidence-v1";
pub const PERCEPTUAL_RAW_COLLECTION_VERSION: &str =
    "mel003-perceptual-raw-collection-v1";
pub const PERCEPTUAL_COLLECTION_CLOSE_VERSION: &str =
    "mel003-perceptual-collection-close-v1";
pub const ZERO_SHA256: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HumanStudyReviewDispositionV1 {
    Approved,
    Exempt,
    NotRequiredByResponsibleAuthority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PerceptualCollectionRoleV1 {
    CollectionOperator,
    EvidenceCustodian,
    BlindedMonitor,
    GovernanceOfficer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerceptualCollectionRunnerIdentityV1 {
    pub source_revision: String,
    pub binary_sha256: String,
    pub environment_sha256: String,
    pub runner_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualCollectionAuthorityV1 {
    pub authority_version: String,
    pub protocol_sha256: String,
    pub stimulus_pack_sha256: String,
    pub participant_schedule_sha256: String,
    pub external_preregistration_sha256: String,
    pub consent_form_sha256: String,
    pub participant_information_sha256: String,
    pub privacy_notice_sha256: String,
    pub recruitment_material_sha256: String,
    pub compensation_policy_sha256: String,
    pub human_study_review_disposition: HumanStudyReviewDispositionV1,
    pub human_study_review_reference: String,
    pub human_study_review_evidence_sha256: String,
    pub planned_open_utc: String,
    pub planned_close_utc: String,
    pub outcome_monitoring_prohibited: bool,
    pub private_audit_access_prohibited: bool,
    pub randomization_key_access_prohibited: bool,
    pub raw_identity_collection_prohibited: bool,
    pub response_correctness_computation_prohibited: bool,
    pub arm_label_derivation_prohibited: bool,
    pub collection_roles: Vec<PerceptualCollectionRoleV1>,
    pub runner: PerceptualCollectionRunnerIdentityV1,
    pub authority_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualCollectionAuthorityIssueV1 {
    InvalidProtocol,
    InvalidStimulusPack,
    WrongVersion,
    SerializationFailed,
    DigestMismatch { field: String },
    InvalidDigest { field: String },
    EmptyField { field: String },
    MissingProtection { field: String },
    DuplicateRole { role: PerceptualCollectionRoleV1 },
    MissingRole { role: PerceptualCollectionRoleV1 },
    AuthorityDigestMismatch,
}

#[derive(Serialize)]
struct CollectionAuthorityCommitment<'a> {
    authority_version: &'a str,
    protocol_sha256: &'a str,
    stimulus_pack_sha256: &'a str,
    participant_schedule_sha256: &'a str,
    external_preregistration_sha256: &'a str,
    consent_form_sha256: &'a str,
    participant_information_sha256: &'a str,
    privacy_notice_sha256: &'a str,
    recruitment_material_sha256: &'a str,
    compensation_policy_sha256: &'a str,
    human_study_review_disposition: HumanStudyReviewDispositionV1,
    human_study_review_reference: &'a str,
    human_study_review_evidence_sha256: &'a str,
    planned_open_utc: &'a str,
    planned_close_utc: &'a str,
    outcome_monitoring_prohibited: bool,
    private_audit_access_prohibited: bool,
    randomization_key_access_prohibited: bool,
    raw_identity_collection_prohibited: bool,
    response_correctness_computation_prohibited: bool,
    arm_label_derivation_prohibited: bool,
    collection_roles: &'a [PerceptualCollectionRoleV1],
    runner: &'a PerceptualCollectionRunnerIdentityV1,
}

pub fn collection_authority_commitment(
    authority: &FrozenPerceptualCollectionAuthorityV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&CollectionAuthorityCommitment {
        authority_version: &authority.authority_version,
        protocol_sha256: &authority.protocol_sha256,
        stimulus_pack_sha256: &authority.stimulus_pack_sha256,
        participant_schedule_sha256: &authority.participant_schedule_sha256,
        external_preregistration_sha256: &authority.external_preregistration_sha256,
        consent_form_sha256: &authority.consent_form_sha256,
        participant_information_sha256: &authority.participant_information_sha256,
        privacy_notice_sha256: &authority.privacy_notice_sha256,
        recruitment_material_sha256: &authority.recruitment_material_sha256,
        compensation_policy_sha256: &authority.compensation_policy_sha256,
        human_study_review_disposition: authority.human_study_review_disposition,
        human_study_review_reference: &authority.human_study_review_reference,
        human_study_review_evidence_sha256: &authority.human_study_review_evidence_sha256,
        planned_open_utc: &authority.planned_open_utc,
        planned_close_utc: &authority.planned_close_utc,
        outcome_monitoring_prohibited: authority.outcome_monitoring_prohibited,
        private_audit_access_prohibited: authority.private_audit_access_prohibited,
        randomization_key_access_prohibited: authority.randomization_key_access_prohibited,
        raw_identity_collection_prohibited: authority.raw_identity_collection_prohibited,
        response_correctness_computation_prohibited: authority
            .response_correctness_computation_prohibited,
        arm_label_derivation_prohibited: authority.arm_label_derivation_prohibited,
        collection_roles: &authority.collection_roles,
        runner: &authority.runner,
    })
}

pub fn seal_collection_authority(
    authority: &mut FrozenPerceptualCollectionAuthorityV1,
) -> Result<(), serde_json::Error> {
    authority.collection_roles.sort();
    authority.authority_sha256 = collection_authority_commitment(authority)?;
    Ok(())
}

pub fn validate_collection_authority(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    authority: &FrozenPerceptualCollectionAuthorityV1,
) -> Vec<PerceptualCollectionAuthorityIssueV1> {
    let mut issues = Vec::new();
    if !protocol.validate().is_empty() {
        issues.push(PerceptualCollectionAuthorityIssueV1::InvalidProtocol);
    }
    if !stimulus_pack.validate(protocol, render_binding).is_empty() {
        issues.push(PerceptualCollectionAuthorityIssueV1::InvalidStimulusPack);
    }
    if authority.authority_version != PERCEPTUAL_COLLECTION_AUTHORITY_VERSION {
        issues.push(PerceptualCollectionAuthorityIssueV1::WrongVersion);
    }
    verify_digest(
        "protocol_sha256",
        canonical_json_sha256(protocol),
        &authority.protocol_sha256,
        &mut issues,
    );
    verify_digest(
        "stimulus_pack_sha256",
        canonical_json_sha256(stimulus_pack),
        &authority.stimulus_pack_sha256,
        &mut issues,
    );
    verify_digest(
        "participant_schedule_sha256",
        canonical_json_sha256(schedule),
        &authority.participant_schedule_sha256,
        &mut issues,
    );
    if authority.external_preregistration_sha256
        != protocol.external_preregistration.record_sha256
    {
        issues.push(PerceptualCollectionAuthorityIssueV1::DigestMismatch {
            field: "external_preregistration_sha256".into(),
        });
    }
    for (field, digest) in [
        ("external_preregistration_sha256", authority.external_preregistration_sha256.as_str()),
        ("consent_form_sha256", authority.consent_form_sha256.as_str()),
        ("participant_information_sha256", authority.participant_information_sha256.as_str()),
        ("privacy_notice_sha256", authority.privacy_notice_sha256.as_str()),
        ("recruitment_material_sha256", authority.recruitment_material_sha256.as_str()),
        ("compensation_policy_sha256", authority.compensation_policy_sha256.as_str()),
        ("human_study_review_evidence_sha256", authority.human_study_review_evidence_sha256.as_str()),
        ("runner.binary_sha256", authority.runner.binary_sha256.as_str()),
        ("runner.environment_sha256", authority.runner.environment_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(PerceptualCollectionAuthorityIssueV1::InvalidDigest {
                field: field.into(),
            });
        }
    }
    for (field, value) in [
        ("human_study_review_reference", authority.human_study_review_reference.as_str()),
        ("planned_open_utc", authority.planned_open_utc.as_str()),
        ("planned_close_utc", authority.planned_close_utc.as_str()),
        ("runner.source_revision", authority.runner.source_revision.as_str()),
        ("runner.runner_version", authority.runner.runner_version.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(PerceptualCollectionAuthorityIssueV1::EmptyField {
                field: field.into(),
            });
        }
    }
    for (field, protected) in [
        ("outcome_monitoring_prohibited", authority.outcome_monitoring_prohibited),
        ("private_audit_access_prohibited", authority.private_audit_access_prohibited),
        ("randomization_key_access_prohibited", authority.randomization_key_access_prohibited),
        ("raw_identity_collection_prohibited", authority.raw_identity_collection_prohibited),
        (
            "response_correctness_computation_prohibited",
            authority.response_correctness_computation_prohibited,
        ),
        ("arm_label_derivation_prohibited", authority.arm_label_derivation_prohibited),
    ] {
        if !protected {
            issues.push(PerceptualCollectionAuthorityIssueV1::MissingProtection {
                field: field.into(),
            });
        }
    }
    let mut roles = BTreeSet::new();
    for role in &authority.collection_roles {
        if !roles.insert(*role) {
            issues.push(PerceptualCollectionAuthorityIssueV1::DuplicateRole { role: *role });
        }
    }
    for role in [
        PerceptualCollectionRoleV1::CollectionOperator,
        PerceptualCollectionRoleV1::EvidenceCustodian,
        PerceptualCollectionRoleV1::BlindedMonitor,
        PerceptualCollectionRoleV1::GovernanceOfficer,
    ] {
        if !roles.contains(&role) {
            issues.push(PerceptualCollectionAuthorityIssueV1::MissingRole { role });
        }
    }
    match collection_authority_commitment(authority) {
        Ok(found) if found == authority.authority_sha256 => {}
        Ok(_) => issues.push(PerceptualCollectionAuthorityIssueV1::AuthorityDigestMismatch),
        Err(_) => issues.push(PerceptualCollectionAuthorityIssueV1::SerializationFailed),
    }
    issues
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RawPerceptualChoiceV1 {
    AbxA,
    AbxB,
    DirectionLeft,
    DirectionRight,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionAbortReasonV1 {
    ParticipantStopped,
    TechnicalFailure,
    GovernanceStop,
    OtherNonOutcomeOperational,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionStatusV1 {
    Complete,
    Aborted(SessionAbortReasonV1),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParticipantEligibilityAttestationV1 {
    pub informed_consent_confirmed: bool,
    pub minimum_age_eligibility_confirmed: bool,
    pub stereo_playback_check_passed: bool,
    pub comprehension_practice_passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerceptualTrialResponseRecordV1 {
    pub sequence: u32,
    pub previous_record_sha256: String,
    pub trial_id: String,
    pub item_id: String,
    pub seed: u64,
    pub task: PerceptualTaskV1,
    pub choice: RawPerceptualChoiceV1,
    pub trial_started_elapsed_ms: u64,
    pub response_submitted_elapsed_ms: u64,
    pub record_sha256: String,
}

#[derive(Serialize)]
struct TrialResponseCommitment<'a> {
    sequence: u32,
    previous_record_sha256: &'a str,
    trial_id: &'a str,
    item_id: &'a str,
    seed: u64,
    task: PerceptualTaskV1,
    choice: RawPerceptualChoiceV1,
    trial_started_elapsed_ms: u64,
    response_submitted_elapsed_ms: u64,
}

pub fn trial_response_commitment(
    record: &PerceptualTrialResponseRecordV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&TrialResponseCommitment {
        sequence: record.sequence,
        previous_record_sha256: &record.previous_record_sha256,
        trial_id: &record.trial_id,
        item_id: &record.item_id,
        seed: record.seed,
        task: record.task,
        choice: record.choice,
        trial_started_elapsed_ms: record.trial_started_elapsed_ms,
        response_submitted_elapsed_ms: record.response_submitted_elapsed_ms,
    })
}

pub fn seal_response_chain(
    records: &mut [PerceptualTrialResponseRecordV1],
) -> Result<(), serde_json::Error> {
    let mut previous = ZERO_SHA256.to_string();
    for (index, record) in records.iter_mut().enumerate() {
        record.sequence = index as u32;
        record.previous_record_sha256 = previous;
        record.record_sha256 = trial_response_commitment(record)?;
        previous = record.record_sha256.clone();
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerceptualSessionEvidenceV1 {
    pub session_version: String,
    pub participant_token: String,
    pub participant_schedule_sha256: String,
    pub eligibility: ParticipantEligibilityAttestationV1,
    pub status: SessionStatusV1,
    pub records: Vec<PerceptualTrialResponseRecordV1>,
    pub session_sha256: String,
}

#[derive(Serialize)]
struct SessionCommitment<'a> {
    session_version: &'a str,
    participant_token: &'a str,
    participant_schedule_sha256: &'a str,
    eligibility: &'a ParticipantEligibilityAttestationV1,
    status: SessionStatusV1,
    records: &'a [PerceptualTrialResponseRecordV1],
}

pub fn session_commitment(
    session: &PerceptualSessionEvidenceV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&SessionCommitment {
        session_version: &session.session_version,
        participant_token: &session.participant_token,
        participant_schedule_sha256: &session.participant_schedule_sha256,
        eligibility: &session.eligibility,
        status: session.status,
        records: &session.records,
    })
}

pub fn seal_session_evidence(
    session: &mut PerceptualSessionEvidenceV1,
) -> Result<(), serde_json::Error> {
    seal_response_chain(&mut session.records)?;
    session.session_sha256 = session_commitment(session)?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualSessionEvidenceIssueV1 {
    WrongVersion,
    UnknownParticipant,
    ParticipantScheduleDigestMismatch,
    EligibilityNotSatisfied { field: String },
    TooManyRecords { found: usize, expected: usize },
    CompleteSessionRecordCountMismatch { found: usize, expected: usize },
    AbortedSessionNotPrefix { found: usize, expected: usize },
    SequenceMismatch { index: usize },
    PreviousDigestMismatch { index: usize },
    InvalidRecordDigest { index: usize },
    RecordDigestMismatch { index: usize },
    TrialBindingMismatch { index: usize },
    ChoiceTaskMismatch { index: usize },
    NonMonotonicTrialTime { index: usize },
    SessionDigestMismatch,
    SerializationFailed,
}

pub fn validate_session_evidence(
    schedule_book: &PerceptualParticipantScheduleBookV1,
    session: &PerceptualSessionEvidenceV1,
) -> Vec<PerceptualSessionEvidenceIssueV1> {
    let mut issues = Vec::new();
    if session.session_version != PERCEPTUAL_SESSION_EVIDENCE_VERSION {
        issues.push(PerceptualSessionEvidenceIssueV1::WrongVersion);
    }
    let expected_schedule_digest = match canonical_json_sha256(schedule_book) {
        Ok(value) => value,
        Err(_) => {
            issues.push(PerceptualSessionEvidenceIssueV1::SerializationFailed);
            String::new()
        }
    };
    if session.participant_schedule_sha256 != expected_schedule_digest {
        issues.push(PerceptualSessionEvidenceIssueV1::ParticipantScheduleDigestMismatch);
    }
    for (field, value) in [
        ("informed_consent_confirmed", session.eligibility.informed_consent_confirmed),
        (
            "minimum_age_eligibility_confirmed",
            session.eligibility.minimum_age_eligibility_confirmed,
        ),
        ("stereo_playback_check_passed", session.eligibility.stereo_playback_check_passed),
        ("comprehension_practice_passed", session.eligibility.comprehension_practice_passed),
    ] {
        if !value {
            issues.push(PerceptualSessionEvidenceIssueV1::EligibilityNotSatisfied {
                field: field.into(),
            });
        }
    }

    let Some(public_schedule) = schedule_book
        .schedules
        .iter()
        .find(|candidate| candidate.participant_token == session.participant_token)
    else {
        issues.push(PerceptualSessionEvidenceIssueV1::UnknownParticipant);
        return issues;
    };
    let expected = flattened_trial_order(public_schedule);
    if session.records.len() > expected.len() {
        issues.push(PerceptualSessionEvidenceIssueV1::TooManyRecords {
            found: session.records.len(),
            expected: expected.len(),
        });
    }
    match session.status {
        SessionStatusV1::Complete if session.records.len() != expected.len() => {
            issues.push(
                PerceptualSessionEvidenceIssueV1::CompleteSessionRecordCountMismatch {
                    found: session.records.len(),
                    expected: expected.len(),
                },
            );
        }
        SessionStatusV1::Aborted(_) if session.records.len() >= expected.len() => {
            issues.push(PerceptualSessionEvidenceIssueV1::AbortedSessionNotPrefix {
                found: session.records.len(),
                expected: expected.len(),
            });
        }
        _ => {}
    }

    let mut previous = ZERO_SHA256;
    let mut prior_submitted = 0u64;
    for (index, record) in session.records.iter().enumerate() {
        if record.sequence != index as u32 {
            issues.push(PerceptualSessionEvidenceIssueV1::SequenceMismatch { index });
        }
        if record.previous_record_sha256 != previous {
            issues.push(PerceptualSessionEvidenceIssueV1::PreviousDigestMismatch { index });
        }
        if !is_sha256(&record.record_sha256) {
            issues.push(PerceptualSessionEvidenceIssueV1::InvalidRecordDigest { index });
        }
        match trial_response_commitment(record) {
            Ok(value) if value == record.record_sha256 => {}
            Ok(_) => issues.push(PerceptualSessionEvidenceIssueV1::RecordDigestMismatch { index }),
            Err(_) => issues.push(PerceptualSessionEvidenceIssueV1::SerializationFailed),
        }
        if record.trial_started_elapsed_ms < prior_submitted
            || record.response_submitted_elapsed_ms < record.trial_started_elapsed_ms
        {
            issues.push(PerceptualSessionEvidenceIssueV1::NonMonotonicTrialTime { index });
        }
        prior_submitted = record.response_submitted_elapsed_ms;

        match expected.get(index) {
            Some(expected_trial)
                if expected_trial.trial_id == record.trial_id
                    && expected_trial.item_id == record.item_id
                    && expected_trial.seed == record.seed
                    && expected_trial.task == record.task => {}
            _ => issues.push(PerceptualSessionEvidenceIssueV1::TrialBindingMismatch { index }),
        }
        let choice_valid = matches!(
            (record.task, record.choice),
            (PerceptualTaskV1::AbxDiscrimination, RawPerceptualChoiceV1::AbxA)
                | (PerceptualTaskV1::AbxDiscrimination, RawPerceptualChoiceV1::AbxB)
                | (
                    PerceptualTaskV1::DirectionalRearticulation2Afc,
                    RawPerceptualChoiceV1::DirectionLeft
                )
                | (
                    PerceptualTaskV1::DirectionalRearticulation2Afc,
                    RawPerceptualChoiceV1::DirectionRight
                )
        );
        if !choice_valid {
            issues.push(PerceptualSessionEvidenceIssueV1::ChoiceTaskMismatch { index });
        }
        previous = &record.record_sha256;
    }
    match session_commitment(session) {
        Ok(value) if value == session.session_sha256 => {}
        Ok(_) => issues.push(PerceptualSessionEvidenceIssueV1::SessionDigestMismatch),
        Err(_) => issues.push(PerceptualSessionEvidenceIssueV1::SerializationFailed),
    }
    issues
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpectedTrialV1 {
    trial_id: String,
    item_id: String,
    seed: u64,
    task: PerceptualTaskV1,
}

fn flattened_trial_order(schedule: &PublicParticipantScheduleV1) -> Vec<ExpectedTrialV1> {
    let abx = schedule.abx_trials.iter().map(|trial| ExpectedTrialV1 {
        trial_id: trial.trial_id.clone(),
        item_id: trial.item_id.clone(),
        seed: trial.seed,
        task: PerceptualTaskV1::AbxDiscrimination,
    });
    let directional = schedule.directional_trials.iter().map(|trial| ExpectedTrialV1 {
        trial_id: trial.trial_id.clone(),
        item_id: trial.item_id.clone(),
        seed: trial.seed,
        task: PerceptualTaskV1::DirectionalRearticulation2Afc,
    });
    match schedule.task_order {
        TaskBlockOrderV1::AbxThenDirectional => abx.chain(directional).collect(),
        TaskBlockOrderV1::DirectionalThenAbx => directional.chain(abx).collect(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawPerceptualCollectionV1 {
    pub collection_version: String,
    pub collection_authority_sha256: String,
    pub protocol_sha256: String,
    pub stimulus_pack_sha256: String,
    pub participant_schedule_sha256: String,
    pub cohort_id: String,
    pub sessions: Vec<PerceptualSessionEvidenceV1>,
    pub raw_dataset_sha256: String,
}

#[derive(Serialize)]
struct RawCollectionCommitment<'a> {
    collection_version: &'a str,
    collection_authority_sha256: &'a str,
    protocol_sha256: &'a str,
    stimulus_pack_sha256: &'a str,
    participant_schedule_sha256: &'a str,
    cohort_id: &'a str,
    sessions: &'a [PerceptualSessionEvidenceV1],
}

pub fn raw_collection_commitment(
    collection: &RawPerceptualCollectionV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&RawCollectionCommitment {
        collection_version: &collection.collection_version,
        collection_authority_sha256: &collection.collection_authority_sha256,
        protocol_sha256: &collection.protocol_sha256,
        stimulus_pack_sha256: &collection.stimulus_pack_sha256,
        participant_schedule_sha256: &collection.participant_schedule_sha256,
        cohort_id: &collection.cohort_id,
        sessions: &collection.sessions,
    })
}

pub fn seal_raw_collection(
    collection: &mut RawPerceptualCollectionV1,
) -> Result<(), serde_json::Error> {
    collection
        .sessions
        .sort_by(|left, right| left.participant_token.cmp(&right.participant_token));
    collection.raw_dataset_sha256 = raw_collection_commitment(collection)?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RawPerceptualCollectionIssueV1 {
    WrongVersion,
    InvalidAuthority,
    DigestMismatch { field: String },
    DuplicateParticipantSession { participant_token: String },
    UnknownParticipantSession { participant_token: String },
    InvalidSession { participant_token: String, issues: Vec<PerceptualSessionEvidenceIssueV1> },
    RawDatasetDigestMismatch,
    SerializationFailed,
}

pub fn validate_raw_collection(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    authority: &FrozenPerceptualCollectionAuthorityV1,
    collection: &RawPerceptualCollectionV1,
) -> Vec<RawPerceptualCollectionIssueV1> {
    let mut issues = Vec::new();
    if collection.collection_version != PERCEPTUAL_RAW_COLLECTION_VERSION {
        issues.push(RawPerceptualCollectionIssueV1::WrongVersion);
    }
    if !validate_collection_authority(protocol, stimulus_pack, render_binding, schedule, authority)
        .is_empty()
    {
        issues.push(RawPerceptualCollectionIssueV1::InvalidAuthority);
    }
    verify_raw_digest(
        "collection_authority_sha256",
        Ok(authority.authority_sha256.clone()),
        &collection.collection_authority_sha256,
        &mut issues,
    );
    verify_raw_digest(
        "protocol_sha256",
        canonical_json_sha256(protocol),
        &collection.protocol_sha256,
        &mut issues,
    );
    verify_raw_digest(
        "stimulus_pack_sha256",
        canonical_json_sha256(stimulus_pack),
        &collection.stimulus_pack_sha256,
        &mut issues,
    );
    verify_raw_digest(
        "participant_schedule_sha256",
        canonical_json_sha256(schedule),
        &collection.participant_schedule_sha256,
        &mut issues,
    );

    let scheduled: BTreeSet<_> = schedule
        .schedules
        .iter()
        .map(|entry| entry.participant_token.as_str())
        .collect();
    let mut seen = BTreeSet::new();
    for session in &collection.sessions {
        if !seen.insert(session.participant_token.as_str()) {
            issues.push(RawPerceptualCollectionIssueV1::DuplicateParticipantSession {
                participant_token: session.participant_token.clone(),
            });
        }
        if !scheduled.contains(session.participant_token.as_str()) {
            issues.push(RawPerceptualCollectionIssueV1::UnknownParticipantSession {
                participant_token: session.participant_token.clone(),
            });
        }
        let session_issues = validate_session_evidence(schedule, session);
        if !session_issues.is_empty() {
            issues.push(RawPerceptualCollectionIssueV1::InvalidSession {
                participant_token: session.participant_token.clone(),
                issues: session_issues,
            });
        }
    }
    match raw_collection_commitment(collection) {
        Ok(value) if value == collection.raw_dataset_sha256 => {}
        Ok(_) => issues.push(RawPerceptualCollectionIssueV1::RawDatasetDigestMismatch),
        Err(_) => issues.push(RawPerceptualCollectionIssueV1::SerializationFailed),
    }
    issues
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualCollectionCloseReasonV1 {
    PlannedCompletionsReached,
    MaximumEnrollmentReachedBeforeTarget,
    FrozenDeadlineReached,
    GovernanceOrSafetyStop,
    OperationalIntegrityStop,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerceptualCollectionCloseV1 {
    pub close_version: String,
    pub collection_authority_sha256: String,
    pub raw_dataset_sha256: String,
    pub participant_schedule_sha256: String,
    pub close_reason: PerceptualCollectionCloseReasonV1,
    pub completed_sessions: usize,
    pub aborted_sessions: usize,
    /// Participants who withdrew permission to retain their study data before
    /// seal. Their session/response rows must be absent from the raw dataset.
    pub withdrawn_and_deleted_sessions: usize,
    pub unused_enrollment_slots: usize,
    /// Screening failures occur before scored-session evidence and are retained
    /// only as an aggregate operational count.
    pub screening_failure_count: usize,
    pub closed_at_utc: String,
    pub outcome_monitoring_performed_before_close: bool,
    pub private_audit_accessed_before_close: bool,
    pub randomization_key_revealed_before_close: bool,
    pub close_sha256: String,
}

#[derive(Serialize)]
struct CollectionCloseCommitment<'a> {
    close_version: &'a str,
    collection_authority_sha256: &'a str,
    raw_dataset_sha256: &'a str,
    participant_schedule_sha256: &'a str,
    close_reason: PerceptualCollectionCloseReasonV1,
    completed_sessions: usize,
    aborted_sessions: usize,
    withdrawn_and_deleted_sessions: usize,
    unused_enrollment_slots: usize,
    screening_failure_count: usize,
    closed_at_utc: &'a str,
    outcome_monitoring_performed_before_close: bool,
    private_audit_accessed_before_close: bool,
    randomization_key_revealed_before_close: bool,
}

pub fn collection_close_commitment(
    close: &PerceptualCollectionCloseV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&CollectionCloseCommitment {
        close_version: &close.close_version,
        collection_authority_sha256: &close.collection_authority_sha256,
        raw_dataset_sha256: &close.raw_dataset_sha256,
        participant_schedule_sha256: &close.participant_schedule_sha256,
        close_reason: close.close_reason,
        completed_sessions: close.completed_sessions,
        aborted_sessions: close.aborted_sessions,
        withdrawn_and_deleted_sessions: close.withdrawn_and_deleted_sessions,
        unused_enrollment_slots: close.unused_enrollment_slots,
        screening_failure_count: close.screening_failure_count,
        closed_at_utc: &close.closed_at_utc,
        outcome_monitoring_performed_before_close: close
            .outcome_monitoring_performed_before_close,
        private_audit_accessed_before_close: close.private_audit_accessed_before_close,
        randomization_key_revealed_before_close: close.randomization_key_revealed_before_close,
    })
}

pub fn seal_collection_close(
    close: &mut PerceptualCollectionCloseV1,
) -> Result<(), serde_json::Error> {
    close.close_sha256 = collection_close_commitment(close)?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualCollectionCloseIssueV1 {
    WrongVersion,
    InvalidRawCollection,
    AuthorityDigestMismatch,
    RawDatasetDigestMismatch,
    ParticipantScheduleDigestMismatch,
    SessionCountMismatch,
    EnrollmentPartitionMismatch { found: usize, expected: usize },
    PlannedTargetCloseMismatch,
    EnrollmentCapCloseMismatch,
    EmptyCloseTime,
    PrematureOutcomeMonitoring,
    PrematurePrivateAuditAccess,
    PrematureKeyReveal,
    CloseDigestMismatch,
    SerializationFailed,
}

pub fn validate_collection_close(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    authority: &FrozenPerceptualCollectionAuthorityV1,
    collection: &RawPerceptualCollectionV1,
    close: &PerceptualCollectionCloseV1,
) -> Vec<PerceptualCollectionCloseIssueV1> {
    let mut issues = Vec::new();
    if close.close_version != PERCEPTUAL_COLLECTION_CLOSE_VERSION {
        issues.push(PerceptualCollectionCloseIssueV1::WrongVersion);
    }
    if !validate_raw_collection(
        protocol,
        stimulus_pack,
        render_binding,
        schedule,
        authority,
        collection,
    )
    .is_empty()
    {
        issues.push(PerceptualCollectionCloseIssueV1::InvalidRawCollection);
    }
    if close.collection_authority_sha256 != authority.authority_sha256 {
        issues.push(PerceptualCollectionCloseIssueV1::AuthorityDigestMismatch);
    }
    if close.raw_dataset_sha256 != collection.raw_dataset_sha256 {
        issues.push(PerceptualCollectionCloseIssueV1::RawDatasetDigestMismatch);
    }
    match canonical_json_sha256(schedule) {
        Ok(value) if value == close.participant_schedule_sha256 => {}
        Ok(_) => issues.push(PerceptualCollectionCloseIssueV1::ParticipantScheduleDigestMismatch),
        Err(_) => issues.push(PerceptualCollectionCloseIssueV1::SerializationFailed),
    }

    let completed = collection
        .sessions
        .iter()
        .filter(|session| session.status == SessionStatusV1::Complete)
        .count();
    let aborted = collection.sessions.len() - completed;
    if close.completed_sessions != completed || close.aborted_sessions != aborted {
        issues.push(PerceptualCollectionCloseIssueV1::SessionCountMismatch);
    }
    let enrolled_or_consumed = close.completed_sessions
        + close.aborted_sessions
        + close.withdrawn_and_deleted_sessions
        + close.unused_enrollment_slots;
    if enrolled_or_consumed != protocol.sample_size.maximum_enrolled_participants {
        issues.push(PerceptualCollectionCloseIssueV1::EnrollmentPartitionMismatch {
            found: enrolled_or_consumed,
            expected: protocol.sample_size.maximum_enrolled_participants,
        });
    }
    match close.close_reason {
        PerceptualCollectionCloseReasonV1::PlannedCompletionsReached
            if close.completed_sessions
                != protocol.sample_size.planned_completed_participants =>
        {
            issues.push(PerceptualCollectionCloseIssueV1::PlannedTargetCloseMismatch);
        }
        PerceptualCollectionCloseReasonV1::MaximumEnrollmentReachedBeforeTarget
            if close.unused_enrollment_slots != 0
                || close.completed_sessions
                    >= protocol.sample_size.planned_completed_participants =>
        {
            issues.push(PerceptualCollectionCloseIssueV1::EnrollmentCapCloseMismatch);
        }
        _ => {}
    }
    if close.closed_at_utc.trim().is_empty() {
        issues.push(PerceptualCollectionCloseIssueV1::EmptyCloseTime);
    }
    if close.outcome_monitoring_performed_before_close {
        issues.push(PerceptualCollectionCloseIssueV1::PrematureOutcomeMonitoring);
    }
    if close.private_audit_accessed_before_close {
        issues.push(PerceptualCollectionCloseIssueV1::PrematurePrivateAuditAccess);
    }
    if close.randomization_key_revealed_before_close {
        issues.push(PerceptualCollectionCloseIssueV1::PrematureKeyReveal);
    }
    match collection_close_commitment(close) {
        Ok(value) if value == close.close_sha256 => {}
        Ok(_) => issues.push(PerceptualCollectionCloseIssueV1::CloseDigestMismatch),
        Err(_) => issues.push(PerceptualCollectionCloseIssueV1::SerializationFailed),
    }
    issues
}

fn verify_digest(
    field: &str,
    result: Result<String, serde_json::Error>,
    expected: &str,
    issues: &mut Vec<PerceptualCollectionAuthorityIssueV1>,
) {
    match result {
        Ok(value) if value == expected => {}
        Ok(_) => issues.push(PerceptualCollectionAuthorityIssueV1::DigestMismatch {
            field: field.into(),
        }),
        Err(_) => issues.push(PerceptualCollectionAuthorityIssueV1::SerializationFailed),
    }
}

fn verify_raw_digest(
    field: &str,
    result: Result<String, serde_json::Error>,
    expected: &str,
    issues: &mut Vec<RawPerceptualCollectionIssueV1>,
) {
    match result {
        Ok(value) if value == expected => {}
        Ok(_) => issues.push(RawPerceptualCollectionIssueV1::DigestMismatch {
            field: field.into(),
        }),
        Err(_) => issues.push(RawPerceptualCollectionIssueV1::SerializationFailed),
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_chain_is_append_order_sensitive() {
        let mut records = vec![
            PerceptualTrialResponseRecordV1 {
                sequence: 99,
                previous_record_sha256: String::new(),
                trial_id: "trial-1".into(),
                item_id: "sonata-seed-3".into(),
                seed: 3,
                task: PerceptualTaskV1::AbxDiscrimination,
                choice: RawPerceptualChoiceV1::AbxA,
                trial_started_elapsed_ms: 1_000,
                response_submitted_elapsed_ms: 2_000,
                record_sha256: String::new(),
            },
            PerceptualTrialResponseRecordV1 {
                sequence: 99,
                previous_record_sha256: String::new(),
                trial_id: "trial-2".into(),
                item_id: "sonata-seed-11".into(),
                seed: 11,
                task: PerceptualTaskV1::AbxDiscrimination,
                choice: RawPerceptualChoiceV1::AbxB,
                trial_started_elapsed_ms: 2_100,
                response_submitted_elapsed_ms: 3_000,
                record_sha256: String::new(),
            },
        ];
        seal_response_chain(&mut records).unwrap();
        assert_eq!(records[0].sequence, 0);
        assert_eq!(records[0].previous_record_sha256, ZERO_SHA256);
        assert_eq!(records[1].sequence, 1);
        assert_eq!(records[1].previous_record_sha256, records[0].record_sha256);
        let original = records[1].record_sha256.clone();
        records[1].choice = RawPerceptualChoiceV1::AbxA;
        assert_ne!(trial_response_commitment(&records[1]).unwrap(), original);
    }

    #[test]
    fn task_specific_raw_choice_has_no_correctness_or_arm_label() {
        let choice = RawPerceptualChoiceV1::DirectionLeft;
        assert!(matches!(choice, RawPerceptualChoiceV1::DirectionLeft));
    }

    #[test]
    fn close_commitment_detects_premature_unblinding_flag_change() {
        let mut close = PerceptualCollectionCloseV1 {
            close_version: PERCEPTUAL_COLLECTION_CLOSE_VERSION.into(),
            collection_authority_sha256: "a".repeat(64),
            raw_dataset_sha256: "b".repeat(64),
            participant_schedule_sha256: "c".repeat(64),
            close_reason: PerceptualCollectionCloseReasonV1::FrozenDeadlineReached,
            completed_sessions: 0,
            aborted_sessions: 0,
            withdrawn_and_deleted_sessions: 0,
            unused_enrollment_slots: 10,
            screening_failure_count: 0,
            closed_at_utc: "2026-09-19T12:00:00Z".into(),
            outcome_monitoring_performed_before_close: false,
            private_audit_accessed_before_close: false,
            randomization_key_revealed_before_close: false,
            close_sha256: String::new(),
        };
        seal_collection_close(&mut close).unwrap();
        let sealed = close.close_sha256.clone();
        close.randomization_key_revealed_before_close = true;
        assert_ne!(collection_close_commitment(&close).unwrap(), sealed);
    }
}
