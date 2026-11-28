// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bitemporal replay for environment policy and procedure revisions.
//!
//! A reversion to prior content is a new revision with new source-lineage and
//! authority receipts. It never resurrects the prior revision identity.
//!
//! The resolver answers: what revision was effective at valid_at_ms, according
//! to only events recorded by known_at_ms? Authority receipts are supplied by
//! an external verifier; this crate does not authenticate their issuer.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION_V1: &str = "mace_environment_policy_revision_replay_v1";
pub const GENESIS_EVENT_SHA256_V1: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RevisionEventKindV1 {
    Activate,
    Withdraw,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyRevisionEventV1 {
    pub schema_version: String,
    pub stream_id: String,
    pub environment_id: String,
    pub procedure_id: String,
    pub scope_id: String,
    pub sequence: u64,
    pub event_id: String,
    pub kind: RevisionEventKindV1,
    /// Present only for Activate. Every activation receives a fresh ID.
    pub revision_id: Option<String>,
    /// Present only for Withdraw. Identifies the revision being withdrawn.
    pub target_revision_id: Option<String>,
    /// Revision expected to be active immediately before this event.
    pub previous_revision_id: Option<String>,
    /// For a reversion, points to an earlier revision with the same content.
    /// The new revision still requires new lineage and authority receipts.
    pub reversion_of_revision_id: Option<String>,
    pub policy_content_sha256: String,
    pub source_lineage_receipt_sha256: String,
    pub authority_receipt_sha256: String,
    /// When the policy becomes effective in the environment.
    pub effective_at_ms: u64,
    /// When the system learned/recorded this event.
    pub recorded_at_ms: u64,
    /// Append-only hash chain; genesis uses 64 zeroes.
    pub previous_event_sha256: String,
    pub event_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyReplayQueryV1 {
    pub stream_id: String,
    pub environment_id: String,
    pub procedure_id: String,
    pub scope_id: String,
    pub valid_at_ms: u64,
    pub known_at_ms: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionStatusV1 {
    ActiveRevision,
    NoActiveRevision,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivePolicyRevisionV1 {
    pub revision_id: String,
    pub policy_content_sha256: String,
    pub source_lineage_receipt_sha256: String,
    pub authority_receipt_sha256: String,
    pub effective_at_ms: u64,
    pub recorded_at_ms: u64,
    pub revision_sequence: u64,
    pub reversion_of_revision_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyResolutionV1 {
    pub schema_version: String,
    pub record_type: String,
    pub query: PolicyReplayQueryV1,
    pub status: ResolutionStatusV1,
    pub active_revision: Option<ActivePolicyRevisionV1>,
    pub applied_event_count: usize,
    pub last_applied_sequence: Option<u64>,
    /// Digest of only the events visible and effective under this query.
    pub applied_event_prefix_sha256: String,
    pub resolution_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayRejectV1 {
    InvalidIdentity,
    InvalidDigest,
    InvalidEventShape,
    InvalidEventHash,
    InvalidEventChain,
    InvalidSequence,
    InvalidTimeOrder,
    StreamIdentityMismatch,
    QueryIdentityMismatch,
    ExpectedHeadMismatch,
    UnverifiedAuthorityReceipt,
    InvalidPreviousRevision,
    DuplicateRevisionId,
    InvalidReversion,
    ContentReuseRequiresReversion,
    InvalidWithdrawal,
    SerializationFailure,
}

fn digest_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn digest_serialized<T: Serialize>(value: &T) -> Result<String, ReplayRejectV1> {
    serde_json::to_vec(value)
        .map(|bytes| digest_bytes(&bytes))
        .map_err(|_| ReplayRejectV1::SerializationFailure)
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= 512
}

fn event_digest(event: &PolicyRevisionEventV1) -> Result<String, ReplayRejectV1> {
    let mut unsigned = event.clone();
    unsigned.event_sha256.clear();
    digest_serialized(&unsigned)
}

impl PolicyRevisionEventV1 {
    pub fn verify_integrity(&self) -> bool {
        self.schema_version == SCHEMA_VERSION_V1
            && event_digest(self).is_ok_and(|digest| digest == self.event_sha256)
    }
}

fn validate_event_shape(event: &PolicyRevisionEventV1) -> Result<(), ReplayRejectV1> {
    if event.schema_version != SCHEMA_VERSION_V1
        || !nonempty(&event.stream_id)
        || !nonempty(&event.environment_id)
        || !nonempty(&event.procedure_id)
        || !nonempty(&event.scope_id)
        || !nonempty(&event.event_id)
        || event.sequence == 0
    {
        return Err(ReplayRejectV1::InvalidIdentity);
    }
    for digest in [
        event.policy_content_sha256.as_str(),
        event.source_lineage_receipt_sha256.as_str(),
        event.authority_receipt_sha256.as_str(),
        event.previous_event_sha256.as_str(),
    ] {
        if !valid_digest(digest) {
            return Err(ReplayRejectV1::InvalidDigest);
        }
    }
    match event.kind {
        RevisionEventKindV1::Activate => {
            if event.revision_id.as_deref().is_none_or(|value| !nonempty(value))
                || event.target_revision_id.is_some()
            {
                return Err(ReplayRejectV1::InvalidEventShape);
            }
        }
        RevisionEventKindV1::Withdraw => {
            if event.revision_id.is_some()
                || event.target_revision_id.as_deref().is_none_or(|value| !nonempty(value))
                || event.previous_revision_id.is_none()
                || event.reversion_of_revision_id.is_some()
            {
                return Err(ReplayRejectV1::InvalidEventShape);
            }
        }
    }
    if let Some(value) = event.previous_revision_id.as_deref() {
        if !nonempty(value) {
            return Err(ReplayRejectV1::InvalidEventShape);
        }
    }
    if let Some(value) = event.reversion_of_revision_id.as_deref() {
        if !nonempty(value) {
            return Err(ReplayRejectV1::InvalidEventShape);
        }
    }
    Ok(())
}

/// Seal one event after the caller has supplied its predecessor event digest.
/// This validates shape and integrity only; replay validates transition semantics.
pub fn seal_policy_revision_event_v1(
    mut event: PolicyRevisionEventV1,
) -> Result<PolicyRevisionEventV1, ReplayRejectV1> {
    event.schema_version = SCHEMA_VERSION_V1.to_owned();
    validate_event_shape(&event)?;
    event.event_sha256.clear();
    event.event_sha256 = event_digest(&event)?;
    Ok(event)
}

fn resolution_digest(value: &PolicyResolutionV1) -> Result<String, ReplayRejectV1> {
    let mut unsigned = value.clone();
    unsigned.resolution_sha256.clear();
    digest_serialized(&unsigned)
}

impl PolicyResolutionV1 {
    pub fn verify_integrity(&self) -> bool {
        resolution_digest(self).is_ok_and(|digest| digest == self.resolution_sha256)
    }
}

#[derive(Clone, Debug)]
struct ActiveRevisionState {
    revision_id: String,
    policy_content_sha256: String,
    source_lineage_receipt_sha256: String,
    authority_receipt_sha256: String,
    effective_at_ms: u64,
    recorded_at_ms: u64,
    revision_sequence: u64,
    reversion_of_revision_id: Option<String>,
}

fn validate_complete_event_stream(
    events: &[PolicyRevisionEventV1],
    expected_stream_head_sha256: &str,
) -> Result<(), ReplayRejectV1> {
    if !valid_digest(expected_stream_head_sha256) {
        return Err(ReplayRejectV1::InvalidDigest);
    }
    if events.is_empty() {
        if expected_stream_head_sha256 != GENESIS_EVENT_SHA256_V1 {
            return Err(ReplayRejectV1::ExpectedHeadMismatch);
        }
        return Ok(());
    }

    let first = &events[0];
    let mut previous_event_sha256 = GENESIS_EVENT_SHA256_V1.to_owned();
    let mut previous_recorded_at = None;
    let mut previous_effective_at = None;
    let mut seen_event_ids = BTreeSet::new();

    for (index, event) in events.iter().enumerate() {
        validate_event_shape(event)?;
        if !event.verify_integrity() {
            return Err(ReplayRejectV1::InvalidEventHash);
        }
        if event.sequence != index as u64 + 1 {
            return Err(ReplayRejectV1::InvalidSequence);
        }
        if event.previous_event_sha256 != previous_event_sha256 {
            return Err(ReplayRejectV1::InvalidEventChain);
        }
        if event.stream_id != first.stream_id
            || event.environment_id != first.environment_id
            || event.procedure_id != first.procedure_id
            || event.scope_id != first.scope_id
        {
            return Err(ReplayRejectV1::StreamIdentityMismatch);
        }
        if previous_recorded_at.is_some_and(|value| event.recorded_at_ms < value)
            || previous_effective_at.is_some_and(|value| event.effective_at_ms < value)
        {
            return Err(ReplayRejectV1::InvalidTimeOrder);
        }
        if !seen_event_ids.insert(event.event_id.as_str()) {
            return Err(ReplayRejectV1::InvalidEventChain);
        }
        previous_event_sha256.clone_from(&event.event_sha256);
        previous_recorded_at = Some(event.recorded_at_ms);
        previous_effective_at = Some(event.effective_at_ms);
    }

    if previous_event_sha256 != expected_stream_head_sha256 {
        return Err(ReplayRejectV1::ExpectedHeadMismatch);
    }
    Ok(())
}

/// Replay one complete append-only policy stream under bitemporal query semantics.
///
/// Only events satisfying both recorded_at <= known_at and effective_at <= valid_at
/// can change the resolved state. Events are ordered by monotone stream sequence;
/// this V1 protocol does not permit retroactive effective-time insertion.
pub fn resolve_environment_policy_v1(
    events: &[PolicyRevisionEventV1],
    query: &PolicyReplayQueryV1,
    verified_authority_receipt_sha256: &BTreeSet<String>,
    expected_stream_head_sha256: &str,
) -> Result<PolicyResolutionV1, ReplayRejectV1> {
    if !nonempty(&query.stream_id)
        || !nonempty(&query.environment_id)
        || !nonempty(&query.procedure_id)
        || !nonempty(&query.scope_id)
    {
        return Err(ReplayRejectV1::InvalidIdentity);
    }
    for digest in verified_authority_receipt_sha256 {
        if !valid_digest(digest) {
            return Err(ReplayRejectV1::InvalidDigest);
        }
    }

    validate_complete_event_stream(events, expected_stream_head_sha256)?;
    if let Some(first) = events.first() {
        if first.stream_id != query.stream_id
            || first.environment_id != query.environment_id
            || first.procedure_id != query.procedure_id
            || first.scope_id != query.scope_id
        {
            return Err(ReplayRejectV1::QueryIdentityMismatch);
        }
    }

    let mut active: Option<ActiveRevisionState> = None;
    let mut activated_revisions: BTreeMap<String, ActiveRevisionState> = BTreeMap::new();
    let mut applied_events: Vec<&PolicyRevisionEventV1> = Vec::new();

    for event in events {
        // Stream timestamps are nondecreasing, so an event outside either
        // information/time boundary implies all subsequent events are outside it.
        if event.recorded_at_ms > query.known_at_ms || event.effective_at_ms > query.valid_at_ms {
            break;
        }
        if !verified_authority_receipt_sha256.contains(&event.authority_receipt_sha256) {
            return Err(ReplayRejectV1::UnverifiedAuthorityReceipt);
        }

        match event.kind {
            RevisionEventKindV1::Activate => {
                let revision_id = event.revision_id.as_deref().ok_or(ReplayRejectV1::InvalidEventShape)?;
                if event.previous_revision_id.as_deref() != active.as_ref().map(|state| state.revision_id.as_str()) {
                    return Err(ReplayRejectV1::InvalidPreviousRevision);
                }
                if activated_revisions.contains_key(revision_id) {
                    return Err(ReplayRejectV1::DuplicateRevisionId);
                }

                if let Some(origin_id) = event.reversion_of_revision_id.as_deref() {
                    let origin = activated_revisions.get(origin_id).ok_or(ReplayRejectV1::InvalidReversion)?;
                    if origin.policy_content_sha256 != event.policy_content_sha256
                        || origin_id == revision_id
                        || origin.source_lineage_receipt_sha256 == event.source_lineage_receipt_sha256
                        || origin.authority_receipt_sha256 == event.authority_receipt_sha256
                    {
                        return Err(ReplayRejectV1::InvalidReversion);
                    }
                } else if activated_revisions.values().any(|state| {
                    state.policy_content_sha256 == event.policy_content_sha256
                }) {
                    return Err(ReplayRejectV1::ContentReuseRequiresReversion);
                }

                let state = ActiveRevisionState {
                    revision_id: revision_id.to_owned(),
                    policy_content_sha256: event.policy_content_sha256.clone(),
                    source_lineage_receipt_sha256: event.source_lineage_receipt_sha256.clone(),
                    authority_receipt_sha256: event.authority_receipt_sha256.clone(),
                    effective_at_ms: event.effective_at_ms,
                    recorded_at_ms: event.recorded_at_ms,
                    revision_sequence: event.sequence,
                    reversion_of_revision_id: event.reversion_of_revision_id.clone(),
                };
                activated_revisions.insert(revision_id.to_owned(), state.clone());
                active = Some(state);
            }
            RevisionEventKindV1::Withdraw => {
                let target = event.target_revision_id.as_deref().ok_or(ReplayRejectV1::InvalidEventShape)?;
                let current = active.as_ref().ok_or(ReplayRejectV1::InvalidWithdrawal)?;
                if event.previous_revision_id.as_deref() != Some(current.revision_id.as_str())
                    || target != current.revision_id
                    || event.policy_content_sha256 != current.policy_content_sha256
                {
                    return Err(ReplayRejectV1::InvalidWithdrawal);
                }
                active = None;
            }
        }
        applied_events.push(event);
    }

    let applied_event_prefix_sha256 = digest_serialized(&applied_events)?;
    let active_revision = active.map(|state| ActivePolicyRevisionV1 {
        revision_id: state.revision_id,
        policy_content_sha256: state.policy_content_sha256,
        source_lineage_receipt_sha256: state.source_lineage_receipt_sha256,
        authority_receipt_sha256: state.authority_receipt_sha256,
        effective_at_ms: state.effective_at_ms,
        recorded_at_ms: state.recorded_at_ms,
        revision_sequence: state.revision_sequence,
        reversion_of_revision_id: state.reversion_of_revision_id,
    });
    let mut resolution = PolicyResolutionV1 {
        schema_version: SCHEMA_VERSION_V1.to_owned(),
        record_type: "policy_resolution".to_owned(),
        query: query.clone(),
        status: if active_revision.is_some() {
            ResolutionStatusV1::ActiveRevision
        } else {
            ResolutionStatusV1::NoActiveRevision
        },
        active_revision,
        applied_event_count: applied_events.len(),
        last_applied_sequence: applied_events.last().map(|event| event.sequence),
        applied_event_prefix_sha256,
        resolution_sha256: String::new(),
    };
    resolution.resolution_sha256 = resolution_digest(&resolution)?;
    Ok(resolution)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: &str) -> String {
        digest_bytes(value.as_bytes())
    }

    fn base_event(
        sequence: u64,
        previous_event_sha256: String,
        kind: RevisionEventKindV1,
        revision_id: Option<&str>,
        target_revision_id: Option<&str>,
        previous_revision_id: Option<&str>,
        reversion_of_revision_id: Option<&str>,
        content: &str,
        source: &str,
        authority: &str,
        effective_at_ms: u64,
        recorded_at_ms: u64,
    ) -> PolicyRevisionEventV1 {
        PolicyRevisionEventV1 {
            schema_version: SCHEMA_VERSION_V1.into(),
            stream_id: "env-a/procedure-checkout".into(),
            environment_id: "env-a".into(),
            procedure_id: "checkout".into(),
            scope_id: "tenant-7".into(),
            sequence,
            event_id: format!("event-{sequence}"),
            kind,
            revision_id: revision_id.map(str::to_owned),
            target_revision_id: target_revision_id.map(str::to_owned),
            previous_revision_id: previous_revision_id.map(str::to_owned),
            reversion_of_revision_id: reversion_of_revision_id.map(str::to_owned),
            policy_content_sha256: digest(content),
            source_lineage_receipt_sha256: digest(source),
            authority_receipt_sha256: digest(authority),
            effective_at_ms,
            recorded_at_ms,
            previous_event_sha256,
            event_sha256: String::new(),
        }
    }

    fn append(
        events: &mut Vec<PolicyRevisionEventV1>,
        kind: RevisionEventKindV1,
        revision_id: Option<&str>,
        target_revision_id: Option<&str>,
        previous_revision_id: Option<&str>,
        reversion_of_revision_id: Option<&str>,
        content: &str,
        source: &str,
        authority: &str,
        effective_at_ms: u64,
        recorded_at_ms: u64,
    ) {
        let previous = events.last().map_or_else(
            || GENESIS_EVENT_SHA256_V1.to_owned(),
            |event| event.event_sha256.clone(),
        );
        let event = base_event(
            events.len() as u64 + 1,
            previous,
            kind,
            revision_id,
            target_revision_id,
            previous_revision_id,
            reversion_of_revision_id,
            content,
            source,
            authority,
            effective_at_ms,
            recorded_at_ms,
        );
        events.push(seal_policy_revision_event_v1(event).unwrap());
    }

    fn authority_set(events: &[PolicyRevisionEventV1]) -> BTreeSet<String> {
        events.iter().map(|event| event.authority_receipt_sha256.clone()).collect()
    }

    fn query(valid_at_ms: u64, known_at_ms: u64) -> PolicyReplayQueryV1 {
        PolicyReplayQueryV1 {
            stream_id: "env-a/procedure-checkout".into(),
            environment_id: "env-a".into(),
            procedure_id: "checkout".into(),
            scope_id: "tenant-7".into(),
            valid_at_ms,
            known_at_ms,
        }
    }

    fn resolve(
        events: &[PolicyRevisionEventV1],
        valid_at_ms: u64,
        known_at_ms: u64,
    ) -> Result<PolicyResolutionV1, ReplayRejectV1> {
        let authorities = authority_set(events);
        let head = events.last().map_or_else(
            || GENESIS_EVENT_SHA256_V1.to_owned(),
            |event| event.event_sha256.clone(),
        );
        resolve_environment_policy_v1(events, &query(valid_at_ms, known_at_ms), &authorities, &head)
    }

    #[test]
    fn bitemporal_query_returns_only_events_known_and_effective_at_the_snapshot() {
        let mut events = Vec::new();
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-a1"), None, None, None,
            "policy-a", "source-a", "authority-a", 100, 100);
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-b1"), None, Some("rev-a1"), None,
            "policy-b", "source-b", "authority-b", 200, 300);

        let before_notice = resolve(&events, 250, 250).unwrap();
        assert_eq!(before_notice.active_revision.as_ref().map(|value| value.revision_id.as_str()), Some("rev-a1"));
        let after_notice = resolve(&events, 250, 300).unwrap();
        assert_eq!(after_notice.active_revision.as_ref().map(|value| value.revision_id.as_str()), Some("rev-b1"));
        assert_eq!(before_notice.applied_event_count, 1);
        assert_eq!(after_notice.applied_event_count, 2);
    }

    #[test]
    fn valid_time_before_change_keeps_old_revision_even_when_new_notice_is_known() {
        let mut events = Vec::new();
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-a1"), None, None, None,
            "policy-a", "source-a", "authority-a", 100, 100);
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-b1"), None, Some("rev-a1"), None,
            "policy-b", "source-b", "authority-b", 200, 300);

        let historical = resolve(&events, 150, 300).unwrap();
        assert_eq!(historical.active_revision.as_ref().map(|value| value.revision_id.as_str()), Some("rev-a1"));
    }

    #[test]
    fn a_to_b_to_a_reversion_creates_new_revision_and_new_provenance() {
        let mut events = Vec::new();
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-a1"), None, None, None,
            "policy-a", "source-a-v1", "authority-a-v1", 100, 100);
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-b1"), None, Some("rev-a1"), None,
            "policy-b", "source-b", "authority-b", 200, 200);
        append(&mut events, RevisionEventKindV1::Withdraw, None, Some("rev-b1"), Some("rev-b1"), None,
            "policy-b", "withdraw-b", "withdraw-authority", 300, 300);
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-a2"), None, None, Some("rev-a1"),
            "policy-a", "source-a-v2", "authority-a-v2", 301, 301);

        let result = resolve(&events, 400, 400).unwrap();
        let active = result.active_revision.unwrap();
        assert_eq!(active.revision_id, "rev-a2");
        assert_eq!(active.policy_content_sha256, digest("policy-a"));
        assert_eq!(active.reversion_of_revision_id.as_deref(), Some("rev-a1"));
        assert_ne!(active.source_lineage_receipt_sha256, digest("source-a-v1"));
        assert_ne!(active.authority_receipt_sha256, digest("authority-a-v1"));
    }

    #[test]
    fn withdrawn_policy_does_not_resurrect_without_explicit_new_activation() {
        let mut events = Vec::new();
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-a1"), None, None, None,
            "policy-a", "source-a", "authority-a", 100, 100);
        append(&mut events, RevisionEventKindV1::Withdraw, None, Some("rev-a1"), Some("rev-a1"), None,
            "policy-a", "withdraw-a", "withdraw-authority", 200, 200);
        let result = resolve(&events, 250, 250).unwrap();
        assert_eq!(result.status, ResolutionStatusV1::NoActiveRevision);
        assert_eq!(result.active_revision, None);
    }

    #[test]
    fn activation_requires_verified_authority_receipt() {
        let mut events = Vec::new();
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-a1"), None, None, None,
            "policy-a", "source-a", "authority-a", 100, 100);
        let authorities = BTreeSet::new();
        let head = events.last().unwrap().event_sha256.clone();
        assert_eq!(
            resolve_environment_policy_v1(&events, &query(100, 100), &authorities, &head),
            Err(ReplayRejectV1::UnverifiedAuthorityReceipt)
        );
    }

    #[test]
    fn scope_or_stream_identity_cannot_be_crossed() {
        let mut events = Vec::new();
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-a1"), None, None, None,
            "policy-a", "source-a", "authority-a", 100, 100);
        let mut wrong = query(100, 100);
        wrong.scope_id = "tenant-other".into();
        let authorities = authority_set(&events);
        let head = events.last().unwrap().event_sha256.clone();
        assert_eq!(
            resolve_environment_policy_v1(&events, &wrong, &authorities, &head),
            Err(ReplayRejectV1::QueryIdentityMismatch)
        );
    }

    #[test]
    fn same_content_requires_an_explicit_reversion_origin() {
        let mut events = Vec::new();
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-a1"), None, None, None,
            "policy-a", "source-a-v1", "authority-a-v1", 100, 100);
        append(&mut events, RevisionEventKindV1::Withdraw, None, Some("rev-a1"), Some("rev-a1"), None,
            "policy-a", "withdraw-a", "withdraw-authority", 200, 200);
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-a2"), None, None, None,
            "policy-a", "source-a-v2", "authority-a-v2", 300, 300);
        assert_eq!(resolve(&events, 350, 350), Err(ReplayRejectV1::ContentReuseRequiresReversion));
    }

    #[test]
    fn reversion_must_have_new_revision_id_and_fresh_source_and_authority_receipts() {
        let mut events = Vec::new();
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-a1"), None, None, None,
            "policy-a", "source-a-v1", "authority-a-v1", 100, 100);
        append(&mut events, RevisionEventKindV1::Withdraw, None, Some("rev-a1"), Some("rev-a1"), None,
            "policy-a", "withdraw-a", "withdraw-authority", 200, 200);
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-a2"), None, None, Some("rev-a1"),
            "policy-a", "source-a-v1", "authority-a-v2", 300, 300);
        assert_eq!(resolve(&events, 350, 350), Err(ReplayRejectV1::InvalidReversion));
    }

    #[test]
    fn duplicate_revision_id_is_rejected() {
        let mut events = Vec::new();
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-a1"), None, None, None,
            "policy-a", "source-a-v1", "authority-a-v1", 100, 100);
        append(&mut events, RevisionEventKindV1::Withdraw, None, Some("rev-a1"), Some("rev-a1"), None,
            "policy-a", "withdraw-a", "withdraw-authority", 200, 200);
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-a1"), None, None, Some("rev-a1"),
            "policy-a", "source-a-v2", "authority-a-v2", 300, 300);
        assert_eq!(resolve(&events, 350, 350), Err(ReplayRejectV1::DuplicateRevisionId));
    }

    #[test]
    fn invalid_withdrawal_target_cannot_silently_clear_another_revision() {
        let mut events = Vec::new();
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-a1"), None, None, None,
            "policy-a", "source-a", "authority-a", 100, 100);
        append(&mut events, RevisionEventKindV1::Withdraw, None, Some("other-rev"), Some("rev-a1"), None,
            "policy-a", "withdraw-a", "withdraw-authority", 200, 200);
        assert_eq!(resolve(&events, 250, 250), Err(ReplayRejectV1::InvalidWithdrawal));
    }

    #[test]
    fn event_chain_detects_tampering_missing_events_and_wrong_head() {
        let mut events = Vec::new();
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-a1"), None, None, None,
            "policy-a", "source-a", "authority-a", 100, 100);
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-b1"), None, Some("rev-a1"), None,
            "policy-b", "source-b", "authority-b", 200, 200);

        let mut changed = events.clone();
        changed[0].policy_content_sha256 = digest("tampered");
        let authorities = authority_set(&events);
        let head = events.last().unwrap().event_sha256.clone();
        assert_eq!(
            resolve_environment_policy_v1(&changed, &query(250, 250), &authorities, &head),
            Err(ReplayRejectV1::InvalidEventHash)
        );

        let omitted = vec![events[1].clone()];
        assert_eq!(
            resolve_environment_policy_v1(&omitted, &query(250, 250), &authorities, &head),
            Err(ReplayRejectV1::InvalidSequence)
        );
        assert_eq!(
            resolve_environment_policy_v1(&events, &query(250, 250), &authorities, &digest("wrong-head")),
            Err(ReplayRejectV1::ExpectedHeadMismatch)
        );
    }

    #[test]
    fn no_known_or_effective_event_returns_no_active_revision() {
        let mut events = Vec::new();
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-a1"), None, None, None,
            "policy-a", "source-a", "authority-a", 200, 300);
        let not_known = resolve(&events, 250, 250).unwrap();
        assert_eq!(not_known.status, ResolutionStatusV1::NoActiveRevision);
        let not_yet_effective = resolve(&events, 150, 300).unwrap();
        assert_eq!(not_yet_effective.status, ResolutionStatusV1::NoActiveRevision);
    }

    #[test]
    fn resolution_receipt_is_integrity_bound_to_both_times_and_active_revision() {
        let mut events = Vec::new();
        append(&mut events, RevisionEventKindV1::Activate, Some("rev-a1"), None, None, None,
            "policy-a", "source-a", "authority-a", 100, 100);
        let result = resolve(&events, 120, 140).unwrap();
        assert!(result.verify_integrity());
        let mut tampered = result.clone();
        tampered.query.known_at_ms = 90;
        assert!(!tampered.verify_integrity());
    }
}
