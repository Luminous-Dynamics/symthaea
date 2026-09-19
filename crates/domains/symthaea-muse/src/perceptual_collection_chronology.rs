// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1ER: restricted absolute-time chronology for perceptual collection.
//!
//! P1E already proves trial order inside each pseudonymous session. This repair
//! adds the temporal theorem that preregistration, authority, scored sessions,
//! collection close, and sealing occurred in the frozen order and window.
//!
//! Absolute session timestamps remain restricted custodian evidence. They are
//! intentionally not added to P1F/P1G analysis rows.

use crate::evidence_digest::{
    canonical_json_sha256,
    perceptual_collection_evidence::{
        FrozenPerceptualCollectionAuthorityV1, PerceptualCollectionCloseReasonV1,
        PerceptualCollectionCloseV1, RawPerceptualCollectionV1,
        validate_collection_authority, validate_collection_close, validate_raw_collection,
    },
    perceptual_participant_schedule::PerceptualParticipantScheduleBookV1,
    perceptual_stimulus_pack::{
        FrozenC6fRenderSubjectBindingV1, FrozenPerceptualStimulusPackV1,
    },
    perceptual_study_protocol::FrozenPerceptualStudyProtocolV1,
};
use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PERCEPTUAL_CHRONOLOGY_POLICY_VERSION: &str =
    "mel003-perceptual-chronology-policy-v1";
pub const PERCEPTUAL_CHRONOLOGY_RECEIPT_VERSION: &str =
    "mel003-perceptual-chronology-receipt-v1";
pub const PERCEPTUAL_STOP_AUTHORITY_VERSION: &str =
    "mel003-perceptual-stop-authority-v1";
pub const UTC_TIMESTAMP_PROFILE_V1: &str = "rfc3339-utc-seconds-z-v1";
pub const MAX_CLOCK_SKEW_MS_V1: u64 = 5_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChronologyClockSourceV1 {
    /// Absolute UTC from the collection runner is cross-checked against P1E's
    /// monotonic elapsed response times. This is not an external timestamp
    /// authority; stronger notarization belongs to a separate integrity layer.
    RunnerSystemUtcWithMonotonicElapsedCrosscheck,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualChronologyPolicyV1 {
    pub policy_version: String,
    pub protocol_sha256: String,
    pub collection_authority_sha256: String,
    pub preregistration_frozen_at_utc: String,
    pub authority_frozen_at_utc: String,
    pub planned_open_utc: String,
    pub planned_close_utc: String,
    pub timestamp_profile: String,
    pub clock_source: ChronologyClockSourceV1,
    pub maximum_clock_skew_ms: u64,
    pub policy_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RestrictedSessionChronologyV1 {
    pub participant_token: String,
    pub session_sha256: String,
    pub session_started_at_utc: String,
    pub session_ended_at_utc: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerceptualStopAuthorityReceiptV1 {
    pub receipt_version: String,
    pub stop_reason: PerceptualCollectionCloseReasonV1,
    pub decided_at_utc: String,
    pub effective_stop_at_utc: String,
    pub authority_reference: String,
    pub evidence_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualCollectionChronologyV1 {
    pub receipt_version: String,
    pub chronology_policy_sha256: String,
    pub raw_dataset_sha256: String,
    pub collection_close_sha256: String,
    pub sessions: Vec<RestrictedSessionChronologyV1>,
    pub effective_collection_close_at_utc: String,
    pub artifact_sealed_at_utc: String,
    pub stop_authority: Option<PerceptualStopAuthorityReceiptV1>,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualChronologyIssueV1 {
    InvalidCollectionAuthority,
    InvalidRawCollection,
    InvalidCollectionClose,
    WrongPolicyVersion,
    WrongReceiptVersion,
    WrongTimestampProfile,
    WrongClockSource,
    WrongClockSkew,
    ProtocolDigestMismatch,
    AuthorityDigestMismatch,
    RawDatasetDigestMismatch,
    CollectionCloseDigestMismatch,
    InvalidTimestamp { field: String },
    PreregistrationTimestampMismatch,
    PlannedOpenTimestampMismatch,
    PlannedCloseTimestampMismatch,
    PreregistrationNotBeforeAuthority,
    AuthorityAfterPlannedOpen,
    PlannedWindowInvalid,
    WrongSessionChronologyCount { found: usize, expected: usize },
    DuplicateSessionChronology { participant_token: String },
    MissingSessionChronology { participant_token: String },
    UnexpectedSessionChronology { participant_token: String },
    SessionDigestMismatch { participant_token: String },
    SessionBeforeOpen { participant_token: String },
    SessionEndBeforeStart { participant_token: String },
    SessionAfterEffectiveClose { participant_token: String },
    SessionAfterPlannedClose { participant_token: String },
    ElapsedTimeExceedsAbsoluteWindow { participant_token: String },
    EffectiveCloseMismatch,
    EffectiveCloseAfterPlannedClose,
    ArtifactSealedBeforeEffectiveClose,
    DeadlineCloseMismatch,
    MissingStopAuthority,
    UnexpectedStopAuthority,
    WrongStopAuthorityVersion,
    StopReasonMismatch,
    EmptyStopAuthorityReference,
    InvalidStopAuthorityDigest,
    StopDecisionAfterEffectiveStop,
    StopEffectiveTimeMismatch,
    StopAuthorityDigestMismatch,
    PolicySerializationFailed,
    PolicyDigestMismatch,
    ReceiptSerializationFailed,
    ReceiptDigestMismatch,
}

pub fn seal_chronology_policy(
    policy: &mut FrozenPerceptualChronologyPolicyV1,
) -> Result<(), serde_json::Error> {
    policy.policy_sha256 = chronology_policy_commitment(policy)?;
    Ok(())
}

pub fn chronology_policy_commitment(
    policy: &FrozenPerceptualChronologyPolicyV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = policy.clone();
    unsigned.policy_sha256.clear();
    canonical_json_sha256(&unsigned)
}

pub fn seal_stop_authority(
    stop: &mut PerceptualStopAuthorityReceiptV1,
) -> Result<(), serde_json::Error> {
    stop.receipt_sha256 = stop_authority_commitment(stop)?;
    Ok(())
}

pub fn stop_authority_commitment(
    stop: &PerceptualStopAuthorityReceiptV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = stop.clone();
    unsigned.receipt_sha256.clear();
    canonical_json_sha256(&unsigned)
}

pub fn seal_collection_chronology(
    receipt: &mut FrozenPerceptualCollectionChronologyV1,
) -> Result<(), serde_json::Error> {
    receipt
        .sessions
        .sort_by(|left, right| left.participant_token.cmp(&right.participant_token));
    receipt.receipt_sha256 = collection_chronology_commitment(receipt)?;
    Ok(())
}

pub fn collection_chronology_commitment(
    receipt: &FrozenPerceptualCollectionChronologyV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = receipt.clone();
    unsigned.receipt_sha256.clear();
    canonical_json_sha256(&unsigned)
}

#[allow(clippy::too_many_arguments)]
pub fn validate_collection_chronology(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    authority: &FrozenPerceptualCollectionAuthorityV1,
    collection: &RawPerceptualCollectionV1,
    close: &PerceptualCollectionCloseV1,
    policy: &FrozenPerceptualChronologyPolicyV1,
    receipt: &FrozenPerceptualCollectionChronologyV1,
) -> Vec<PerceptualChronologyIssueV1> {
    let mut issues = Vec::new();
    if !validate_collection_authority(protocol, stimulus_pack, render_binding, schedule, authority)
        .is_empty()
    {
        issues.push(PerceptualChronologyIssueV1::InvalidCollectionAuthority);
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
        issues.push(PerceptualChronologyIssueV1::InvalidRawCollection);
    }
    if !validate_collection_close(
        protocol,
        stimulus_pack,
        render_binding,
        schedule,
        authority,
        collection,
        close,
    )
    .is_empty()
    {
        issues.push(PerceptualChronologyIssueV1::InvalidCollectionClose);
    }

    validate_policy(protocol, authority, policy, &mut issues);
    if receipt.receipt_version != PERCEPTUAL_CHRONOLOGY_RECEIPT_VERSION {
        issues.push(PerceptualChronologyIssueV1::WrongReceiptVersion);
    }
    if receipt.chronology_policy_sha256 != policy.policy_sha256 {
        issues.push(PerceptualChronologyIssueV1::PolicyDigestMismatch);
    }
    if receipt.raw_dataset_sha256 != collection.raw_dataset_sha256 {
        issues.push(PerceptualChronologyIssueV1::RawDatasetDigestMismatch);
    }
    if receipt.collection_close_sha256 != close.close_sha256 {
        issues.push(PerceptualChronologyIssueV1::CollectionCloseDigestMismatch);
    }

    let Some(planned_open) = parse_field(
        "policy.planned_open_utc",
        &policy.planned_open_utc,
        &mut issues,
    ) else {
        return finish_receipt_validation(receipt, issues);
    };
    let Some(planned_close) = parse_field(
        "policy.planned_close_utc",
        &policy.planned_close_utc,
        &mut issues,
    ) else {
        return finish_receipt_validation(receipt, issues);
    };
    let Some(effective_close) = parse_field(
        "receipt.effective_collection_close_at_utc",
        &receipt.effective_collection_close_at_utc,
        &mut issues,
    ) else {
        return finish_receipt_validation(receipt, issues);
    };
    let Some(artifact_sealed) = parse_field(
        "receipt.artifact_sealed_at_utc",
        &receipt.artifact_sealed_at_utc,
        &mut issues,
    ) else {
        return finish_receipt_validation(receipt, issues);
    };

    if receipt.effective_collection_close_at_utc != close.closed_at_utc {
        issues.push(PerceptualChronologyIssueV1::EffectiveCloseMismatch);
    }
    if effective_close
        .signed_duration_since(planned_close)
        .num_milliseconds()
        > policy.maximum_clock_skew_ms as i64
    {
        issues.push(PerceptualChronologyIssueV1::EffectiveCloseAfterPlannedClose);
    }
    if artifact_sealed < effective_close {
        issues.push(PerceptualChronologyIssueV1::ArtifactSealedBeforeEffectiveClose);
    }

    validate_sessions(
        collection,
        receipt,
        &planned_open,
        &planned_close,
        &effective_close,
        policy.maximum_clock_skew_ms,
        &mut issues,
    );
    validate_close_reason(
        close,
        receipt,
        &planned_close,
        &effective_close,
        policy.maximum_clock_skew_ms,
        &mut issues,
    );

    finish_receipt_validation(receipt, issues)
}

fn validate_policy(
    protocol: &FrozenPerceptualStudyProtocolV1,
    authority: &FrozenPerceptualCollectionAuthorityV1,
    policy: &FrozenPerceptualChronologyPolicyV1,
    issues: &mut Vec<PerceptualChronologyIssueV1>,
) {
    if policy.policy_version != PERCEPTUAL_CHRONOLOGY_POLICY_VERSION {
        issues.push(PerceptualChronologyIssueV1::WrongPolicyVersion);
    }
    if policy.timestamp_profile != UTC_TIMESTAMP_PROFILE_V1 {
        issues.push(PerceptualChronologyIssueV1::WrongTimestampProfile);
    }
    if policy.clock_source
        != ChronologyClockSourceV1::RunnerSystemUtcWithMonotonicElapsedCrosscheck
    {
        issues.push(PerceptualChronologyIssueV1::WrongClockSource);
    }
    if policy.maximum_clock_skew_ms != MAX_CLOCK_SKEW_MS_V1 {
        issues.push(PerceptualChronologyIssueV1::WrongClockSkew);
    }
    match canonical_json_sha256(protocol) {
        Ok(value) if value == policy.protocol_sha256 => {}
        _ => issues.push(PerceptualChronologyIssueV1::ProtocolDigestMismatch),
    }
    if policy.collection_authority_sha256 != authority.authority_sha256 {
        issues.push(PerceptualChronologyIssueV1::AuthorityDigestMismatch);
    }
    if policy.preregistration_frozen_at_utc != protocol.external_preregistration.frozen_at_utc {
        issues.push(PerceptualChronologyIssueV1::PreregistrationTimestampMismatch);
    }
    if policy.planned_open_utc != authority.planned_open_utc {
        issues.push(PerceptualChronologyIssueV1::PlannedOpenTimestampMismatch);
    }
    if policy.planned_close_utc != authority.planned_close_utc {
        issues.push(PerceptualChronologyIssueV1::PlannedCloseTimestampMismatch);
    }

    let prereg = parse_field(
        "policy.preregistration_frozen_at_utc",
        &policy.preregistration_frozen_at_utc,
        issues,
    );
    let authority_frozen = parse_field(
        "policy.authority_frozen_at_utc",
        &policy.authority_frozen_at_utc,
        issues,
    );
    let open = parse_field("policy.planned_open_utc", &policy.planned_open_utc, issues);
    let close = parse_field("policy.planned_close_utc", &policy.planned_close_utc, issues);
    if let (Some(prereg), Some(authority_frozen)) = (prereg.as_ref(), authority_frozen.as_ref()) {
        if prereg >= authority_frozen {
            issues.push(PerceptualChronologyIssueV1::PreregistrationNotBeforeAuthority);
        }
    }
    if let (Some(authority_frozen), Some(open)) = (authority_frozen.as_ref(), open.as_ref()) {
        if authority_frozen > open {
            issues.push(PerceptualChronologyIssueV1::AuthorityAfterPlannedOpen);
        }
    }
    if let (Some(open), Some(close)) = (open.as_ref(), close.as_ref()) {
        if open >= close {
            issues.push(PerceptualChronologyIssueV1::PlannedWindowInvalid);
        }
    }
    match chronology_policy_commitment(policy) {
        Ok(value) if value == policy.policy_sha256 => {}
        Ok(_) => issues.push(PerceptualChronologyIssueV1::PolicyDigestMismatch),
        Err(_) => issues.push(PerceptualChronologyIssueV1::PolicySerializationFailed),
    }
}

fn validate_sessions(
    collection: &RawPerceptualCollectionV1,
    receipt: &FrozenPerceptualCollectionChronologyV1,
    planned_open: &DateTime<Utc>,
    planned_close: &DateTime<Utc>,
    effective_close: &DateTime<Utc>,
    maximum_clock_skew_ms: u64,
    issues: &mut Vec<PerceptualChronologyIssueV1>,
) {
    if receipt.sessions.len() != collection.sessions.len() {
        issues.push(PerceptualChronologyIssueV1::WrongSessionChronologyCount {
            found: receipt.sessions.len(),
            expected: collection.sessions.len(),
        });
    }
    let source_by_participant: BTreeMap<_, _> = collection
        .sessions
        .iter()
        .map(|session| (session.participant_token.as_str(), session))
        .collect();
    let mut seen = BTreeSet::new();
    for chronology in &receipt.sessions {
        if !seen.insert(chronology.participant_token.as_str()) {
            issues.push(PerceptualChronologyIssueV1::DuplicateSessionChronology {
                participant_token: chronology.participant_token.clone(),
            });
            continue;
        }
        let Some(session) = source_by_participant.get(chronology.participant_token.as_str()) else {
            issues.push(PerceptualChronologyIssueV1::UnexpectedSessionChronology {
                participant_token: chronology.participant_token.clone(),
            });
            continue;
        };
        if chronology.session_sha256 != session.session_sha256 {
            issues.push(PerceptualChronologyIssueV1::SessionDigestMismatch {
                participant_token: chronology.participant_token.clone(),
            });
        }
        let Some(start) = parse_field(
            "session.session_started_at_utc",
            &chronology.session_started_at_utc,
            issues,
        ) else {
            continue;
        };
        let Some(end) = parse_field(
            "session.session_ended_at_utc",
            &chronology.session_ended_at_utc,
            issues,
        ) else {
            continue;
        };
        if start < *planned_open {
            issues.push(PerceptualChronologyIssueV1::SessionBeforeOpen {
                participant_token: chronology.participant_token.clone(),
            });
        }
        if end < start {
            issues.push(PerceptualChronologyIssueV1::SessionEndBeforeStart {
                participant_token: chronology.participant_token.clone(),
            });
        }
        if end > *effective_close {
            issues.push(PerceptualChronologyIssueV1::SessionAfterEffectiveClose {
                participant_token: chronology.participant_token.clone(),
            });
        }
        if end > *planned_close {
            issues.push(PerceptualChronologyIssueV1::SessionAfterPlannedClose {
                participant_token: chronology.participant_token.clone(),
            });
        }
        let wall_ms = end.signed_duration_since(start).num_milliseconds();
        let last_elapsed_ms = session
            .records
            .last()
            .map(|record| record.response_submitted_elapsed_ms)
            .unwrap_or(0);
        if wall_ms < 0
            || (wall_ms as u128 + maximum_clock_skew_ms as u128) < last_elapsed_ms as u128
        {
            issues.push(PerceptualChronologyIssueV1::ElapsedTimeExceedsAbsoluteWindow {
                participant_token: chronology.participant_token.clone(),
            });
        }
    }
    for session in &collection.sessions {
        if !seen.contains(session.participant_token.as_str()) {
            issues.push(PerceptualChronologyIssueV1::MissingSessionChronology {
                participant_token: session.participant_token.clone(),
            });
        }
    }
}

fn validate_close_reason(
    close: &PerceptualCollectionCloseV1,
    receipt: &FrozenPerceptualCollectionChronologyV1,
    planned_close: &DateTime<Utc>,
    effective_close: &DateTime<Utc>,
    maximum_clock_skew_ms: u64,
    issues: &mut Vec<PerceptualChronologyIssueV1>,
) {
    match close.close_reason {
        PerceptualCollectionCloseReasonV1::FrozenDeadlineReached => {
            if absolute_delta_ms(effective_close, planned_close) > maximum_clock_skew_ms as i64 {
                issues.push(PerceptualChronologyIssueV1::DeadlineCloseMismatch);
            }
            if receipt.stop_authority.is_some() {
                issues.push(PerceptualChronologyIssueV1::UnexpectedStopAuthority);
            }
        }
        PerceptualCollectionCloseReasonV1::PlannedCompletionsReached
        | PerceptualCollectionCloseReasonV1::MaximumEnrollmentReachedBeforeTarget => {
            if receipt.stop_authority.is_some() {
                issues.push(PerceptualChronologyIssueV1::UnexpectedStopAuthority);
            }
        }
        PerceptualCollectionCloseReasonV1::GovernanceOrSafetyStop
        | PerceptualCollectionCloseReasonV1::OperationalIntegrityStop => {
            let Some(stop) = receipt.stop_authority.as_ref() else {
                issues.push(PerceptualChronologyIssueV1::MissingStopAuthority);
                return;
            };
            validate_stop_authority(close.close_reason, effective_close, stop, issues);
        }
    }
}

fn validate_stop_authority(
    expected_reason: PerceptualCollectionCloseReasonV1,
    effective_close: &DateTime<Utc>,
    stop: &PerceptualStopAuthorityReceiptV1,
    issues: &mut Vec<PerceptualChronologyIssueV1>,
) {
    if stop.receipt_version != PERCEPTUAL_STOP_AUTHORITY_VERSION {
        issues.push(PerceptualChronologyIssueV1::WrongStopAuthorityVersion);
    }
    if stop.stop_reason != expected_reason {
        issues.push(PerceptualChronologyIssueV1::StopReasonMismatch);
    }
    if stop.authority_reference.trim().is_empty() {
        issues.push(PerceptualChronologyIssueV1::EmptyStopAuthorityReference);
    }
    if !is_sha256(&stop.evidence_sha256) {
        issues.push(PerceptualChronologyIssueV1::InvalidStopAuthorityDigest);
    }
    let decided = parse_field("stop.decided_at_utc", &stop.decided_at_utc, issues);
    let stopped = parse_field("stop.effective_stop_at_utc", &stop.effective_stop_at_utc, issues);
    if let (Some(decided), Some(stopped)) = (decided.as_ref(), stopped.as_ref()) {
        if decided > stopped {
            issues.push(PerceptualChronologyIssueV1::StopDecisionAfterEffectiveStop);
        }
        if stopped != effective_close {
            issues.push(PerceptualChronologyIssueV1::StopEffectiveTimeMismatch);
        }
    }
    match stop_authority_commitment(stop) {
        Ok(value) if value == stop.receipt_sha256 => {}
        _ => issues.push(PerceptualChronologyIssueV1::StopAuthorityDigestMismatch),
    }
}

fn finish_receipt_validation(
    receipt: &FrozenPerceptualCollectionChronologyV1,
    mut issues: Vec<PerceptualChronologyIssueV1>,
) -> Vec<PerceptualChronologyIssueV1> {
    match collection_chronology_commitment(receipt) {
        Ok(value) if value == receipt.receipt_sha256 => {}
        Ok(_) => issues.push(PerceptualChronologyIssueV1::ReceiptDigestMismatch),
        Err(_) => issues.push(PerceptualChronologyIssueV1::ReceiptSerializationFailed),
    }
    issues
}

fn parse_field(
    field: &str,
    value: &str,
    issues: &mut Vec<PerceptualChronologyIssueV1>,
) -> Option<DateTime<Utc>> {
    match parse_canonical_utc_seconds(value) {
        Some(timestamp) => Some(timestamp),
        None => {
            issues.push(PerceptualChronologyIssueV1::InvalidTimestamp {
                field: field.into(),
            });
            None
        }
    }
}

pub fn parse_canonical_utc_seconds(value: &str) -> Option<DateTime<Utc>> {
    let parsed = DateTime::parse_from_rfc3339(value).ok()?;
    if parsed.offset().local_minus_utc() != 0 {
        return None;
    }
    let utc = parsed.with_timezone(&Utc);
    if utc.to_rfc3339_opts(SecondsFormat::Secs, true) != value {
        return None;
    }
    Some(utc)
}

fn absolute_delta_ms(left: &DateTime<Utc>, right: &DateTime<Utc>) -> i64 {
    left.signed_duration_since(*right)
        .num_milliseconds()
        .saturating_abs()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_timestamp_accepts_only_utc_z_seconds() {
        assert!(parse_canonical_utc_seconds("2026-09-19T12:34:56Z").is_some());
        assert!(parse_canonical_utc_seconds("2026-09-19T12:34:56+00:00").is_none());
        assert!(parse_canonical_utc_seconds("2026-09-19T14:34:56+02:00").is_none());
        assert!(parse_canonical_utc_seconds("2026-09-19T12:34:56.000Z").is_none());
        assert!(parse_canonical_utc_seconds("2026-09-19 12:34:56Z").is_none());
    }

    #[test]
    fn deadline_delta_respects_frozen_skew_budget() {
        let planned = parse_canonical_utc_seconds("2026-09-19T12:00:00Z").unwrap();
        let within = parse_canonical_utc_seconds("2026-09-19T12:00:05Z").unwrap();
        let outside = parse_canonical_utc_seconds("2026-09-19T12:00:06Z").unwrap();
        assert_eq!(absolute_delta_ms(&within, &planned), 5_000);
        assert_eq!(absolute_delta_ms(&outside, &planned), 6_000);
    }

    #[test]
    fn stop_authority_commitment_detects_time_change() {
        let mut stop = PerceptualStopAuthorityReceiptV1 {
            receipt_version: PERCEPTUAL_STOP_AUTHORITY_VERSION.into(),
            stop_reason: PerceptualCollectionCloseReasonV1::OperationalIntegrityStop,
            decided_at_utc: "2026-09-19T12:00:00Z".into(),
            effective_stop_at_utc: "2026-09-19T12:00:05Z".into(),
            authority_reference: "ops-incident-1".into(),
            evidence_sha256: "a".repeat(64),
            receipt_sha256: String::new(),
        };
        seal_stop_authority(&mut stop).unwrap();
        let sealed = stop.receipt_sha256.clone();
        stop.effective_stop_at_utc = "2026-09-19T12:00:06Z".into();
        assert_ne!(stop_authority_commitment(&stop).unwrap(), sealed);
    }

    #[test]
    fn chronology_receipt_sorting_is_participant_deterministic() {
        let mut receipt = FrozenPerceptualCollectionChronologyV1 {
            receipt_version: PERCEPTUAL_CHRONOLOGY_RECEIPT_VERSION.into(),
            chronology_policy_sha256: "a".repeat(64),
            raw_dataset_sha256: "b".repeat(64),
            collection_close_sha256: "c".repeat(64),
            sessions: vec![
                RestrictedSessionChronologyV1 {
                    participant_token: "p-2".into(),
                    session_sha256: "d".repeat(64),
                    session_started_at_utc: "2026-09-19T12:00:00Z".into(),
                    session_ended_at_utc: "2026-09-19T12:01:00Z".into(),
                },
                RestrictedSessionChronologyV1 {
                    participant_token: "p-1".into(),
                    session_sha256: "e".repeat(64),
                    session_started_at_utc: "2026-09-19T11:00:00Z".into(),
                    session_ended_at_utc: "2026-09-19T11:01:00Z".into(),
                },
            ],
            effective_collection_close_at_utc: "2026-09-19T13:00:00Z".into(),
            artifact_sealed_at_utc: "2026-09-19T13:00:01Z".into(),
            stop_authority: None,
            receipt_sha256: String::new(),
        };
        seal_collection_chronology(&mut receipt).unwrap();
        assert_eq!(receipt.sessions[0].participant_token, "p-1");
        assert_eq!(receipt.sessions[1].participant_token, "p-2");
    }
}
