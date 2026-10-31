// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Closed-world, query-conditioned event enumeration over trusted memory evidence.
//!
//! A ranked top-k hit set is not evidence that every matching event was found.
//! This module separates three claims:
//! 1. the source inventory is complete for one exact EvidenceIndex snapshot;
//! 2. every inventoried source received exactly one extraction disposition;
//! 3. event selection was evaluated against the complete extracted inventory.
//!
//! The receipt proves source coverage and deterministic filtering, not semantic
//! extraction recall. An extractor can still omit an event from a row marked
//! Complete; that failure must be measured against gold annotations. The batch
//! therefore pins the extractor and schema digests, and benchmark evaluation
//! must report event-level precision/recall separately.
//!
//! Session identity is taken from the snapshot-bound source trajectory after
//! matching a trusted ProofCandidateAdmission. Speaker identity and role are
//! explicit. When a question requires user attribution, any potentially
//! relevant event with unknown speaker attribution makes the result
//! unanswerable rather than silently treating assistant advice as user evidence.

use crate::memory_evidence::{
    EvidenceInventorySnapshot, EvidenceRecord, EVIDENCE_SCHEMA_VERSION,
};
use crate::memory_proof_admission::ProofCandidateAdmission;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const EVENT_ENUMERATION_PROTOCOL_VERSION: &str = "BROCA-CLOSED-WORLD-EVENTS-V1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventSpeakerRole {
    User,
    Assistant,
    Tool,
    System,
    External,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventExtractionStatus {
    Complete,
    Incomplete,
    UnsupportedSource,
    ParseFailure,
}

/// One normalized event extracted from one immutable evidence record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractedMemoryEvent {
    pub event_id: String,
    pub event_type_id: String,
    pub entity_id: String,
    pub actor_id: Option<String>,
    pub speaker_id: Option<String>,
    pub speaker_role: EventSpeakerRole,
    pub occurred_at_unix_ms: Option<u64>,
}

impl ExtractedMemoryEvent {
    fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("event_id", self.event_id.as_str()),
            ("event_type_id", self.event_type_id.as_str()),
            ("entity_id", self.entity_id.as_str()),
        ] {
            require_non_empty(value, field)?;
        }
        if let Some(actor_id) = &self.actor_id {
            require_non_empty(actor_id, "actor_id")?;
        }
        if let Some(speaker_id) = &self.speaker_id {
            require_non_empty(speaker_id, "speaker_id")?;
        }
        match (self.speaker_role, self.speaker_id.as_deref()) {
            (EventSpeakerRole::Unknown, None) => {}
            (EventSpeakerRole::Unknown, Some(_)) => {
                return Err("unknown speaker role cannot carry a speaker ID".into());
            }
            (_, Some(_)) => {}
            (_, None) => {
                return Err("attributed speaker role requires a speaker ID".into());
            }
        }
        Ok(())
    }
}

/// Speaker identity/role resolved by a trusted source adapter from the
/// original immutable message envelope, not inferred from event text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustedSpeakerAttribution {
    pub evidence_id: String,
    pub source_revision_id: String,
    pub source_sha256: String,
    pub source_pointer: String,
    pub source_trajectory_id: String,
    pub speaker_id: Option<String>,
    pub speaker_role: EventSpeakerRole,
    pub attribution_receipt_sha256: String,
}

impl TrustedSpeakerAttribution {
    pub fn computed_receipt_sha256(&self) -> Result<String, String> {
        hash_serializable(
            &(
                &self.evidence_id,
                &self.source_revision_id,
                &self.source_sha256,
                &self.source_pointer,
                &self.source_trajectory_id,
                &self.speaker_id,
                self.speaker_role,
            ),
            "trusted speaker attribution",
        )
    }

    fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("evidence_id", self.evidence_id.as_str()),
            ("source_revision_id", self.source_revision_id.as_str()),
            ("source_pointer", self.source_pointer.as_str()),
            ("source_trajectory_id", self.source_trajectory_id.as_str()),
        ] {
            require_non_empty(value, field)?;
        }
        validate_sha256(&self.source_sha256, "source_sha256")?;
        validate_sha256(&self.attribution_receipt_sha256, "attribution_receipt_sha256")?;
        match (self.speaker_role, self.speaker_id.as_deref()) {
            (EventSpeakerRole::Unknown, None) => {}
            (EventSpeakerRole::Unknown, Some(_)) => {
                return Err("unknown trusted speaker role cannot carry a speaker ID".into());
            }
            (_, Some(speaker_id)) => require_non_empty(speaker_id, "speaker_id")?,
            (_, None) => {
                return Err("trusted speaker role requires a speaker ID".into());
            }
        }
        if self.computed_receipt_sha256()? != self.attribution_receipt_sha256 {
            return Err("trusted speaker attribution receipt digest mismatch".into());
        }
        Ok(())
    }
}

/// Exactly one row must exist for each item in the evidence inventory.
/// An empty events vector with Complete status means that the extractor
/// examined the source and found no event under its extraction contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventExtractionRow {
    pub source_evidence_id: String,
    pub source_sha256: String,
    pub payload_sha256: String,
    pub status: EventExtractionStatus,
    pub events: Vec<ExtractedMemoryEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventExtractionBatch {
    pub protocol_version: String,
    pub source_inventory_sha256: String,
    pub extractor_id: String,
    pub extractor_sha256: String,
    pub extraction_schema_sha256: String,
    pub rows: Vec<EventExtractionRow>,
}

impl EventExtractionBatch {
    pub fn validate_against(
        &self,
        inventory: &EvidenceInventorySnapshot,
        admissions: &[ProofCandidateAdmission],
        speaker_attributions: &[TrustedSpeakerAttribution],
        principal_id: &str,
    ) -> Result<EventExtractionCoverage, String> {
        inventory.verify_internal_consistency()?;
        require_non_empty(principal_id, "principal_id")?;
        if self.protocol_version != EVENT_ENUMERATION_PROTOCOL_VERSION {
            return Err("unsupported event enumeration protocol version".into());
        }
        if self.source_inventory_sha256 != inventory.receipt.inventory_sha256 {
            return Err("event extraction batch is bound to a different source inventory".into());
        }
        require_non_empty(&self.extractor_id, "extractor_id")?;
        validate_sha256(&self.extractor_sha256, "extractor_sha256")?;
        validate_sha256(&self.extraction_schema_sha256, "extraction_schema_sha256")?;

        let mut admission_by_id = BTreeMap::<&str, &ProofCandidateAdmission>::new();
        for admission in admissions {
            require_non_empty(&admission.evidence_id, "admission evidence_id")?;
            if admission_by_id.insert(&admission.evidence_id, admission).is_some() {
                return Err(format!("duplicate source admission {:?}", admission.evidence_id));
            }
            if admission.index_state_sha256 != inventory.receipt.index_state_sha256 {
                return Err(format!(
                    "admission {:?} is bound to a different EvidenceIndex snapshot",
                    admission.evidence_id
                ));
            }
            if admission.scope_id != inventory.receipt.scope_id {
                return Err(format!(
                    "admission {:?} is outside the inventory scope",
                    admission.evidence_id
                ));
            }
            if admission.authorized_principal_id != principal_id || !admission.authorization_accepted {
                return Err(format!(
                    "source admission {:?} is not authorized for the requested principal",
                    admission.evidence_id
                ));
            }
            require_non_empty(&admission.source_revision_id, "source_revision_id")?;
            if admission.token_cost == 0 {
                return Err(format!(
                    "source admission {:?} has no rendered evidence cost",
                    admission.evidence_id
                ));
            }
            if admission.recorded_at_unix_ms > inventory.receipt.known_at_unix_ms
                || admission.retracted_at_unix_ms.is_some_and(|time| {
                    time <= inventory.receipt.known_at_unix_ms
                })
            {
                return Err(format!(
                    "source admission {:?} is not current at the inventory known_at",
                    admission.evidence_id
                ));
            }
            validate_sha256(
                &admission.authorization_receipt_sha256,
                "authorization_receipt_sha256",
            )?;
            require_non_empty(&admission.source_family_id, "source_family_id")?;
        }

        if admission_by_id.len() != inventory.items.len() {
            return Err("source admissions must cover the complete evidence inventory exactly".into());
        }

        let inventory_by_id: BTreeMap<&str, &EvidenceRecord> = inventory.items
            .iter()
            .map(|item| (item.record.evidence_id.as_str(), &item.record))
            .collect();
        let mut speaker_by_id = BTreeMap::<&str, &TrustedSpeakerAttribution>::new();
        for speaker in speaker_attributions {
            speaker.validate()?;
            if speaker_by_id.insert(&speaker.evidence_id, speaker).is_some() {
                return Err(format!("duplicate trusted speaker attribution {:?}", speaker.evidence_id));
            }
            let record = inventory_by_id.get(speaker.evidence_id.as_str())
                .ok_or_else(|| format!("speaker attribution is outside the inventory: {:?}", speaker.evidence_id))?;
            let admission = admission_by_id.get(speaker.evidence_id.as_str())
                .ok_or_else(|| format!("speaker attribution has no trusted source admission: {:?}", speaker.evidence_id))?;
            if speaker.source_revision_id != admission.source_revision_id
                || speaker.source_sha256 != record.source.source_sha256
                || speaker.source_pointer != record.source.source_pointer
                || speaker.source_trajectory_id != record.source.trajectory_id
                || speaker.source_trajectory_id != admission.source_trajectory_id
            {
                return Err(format!("trusted speaker attribution source binding mismatch for {:?}", speaker.evidence_id));
            }
        }
        if speaker_by_id.len() != inventory.items.len() {
            return Err("trusted speaker attribution must cover every inventoried source exactly".into());
        }

        let mut rows_by_id = BTreeMap::<&str, &EventExtractionRow>::new();
        let mut event_ids = BTreeSet::new();
        let mut all_rows_complete = true;

        for row in &self.rows {
            require_non_empty(&row.source_evidence_id, "source_evidence_id")?;
            validate_sha256(&row.source_sha256, "source_sha256")?;
            validate_sha256(&row.payload_sha256, "payload_sha256")?;
            if rows_by_id.insert(&row.source_evidence_id, row).is_some() {
                return Err(format!(
                    "duplicate event-extraction row for {:?}",
                    row.source_evidence_id
                ));
            }
            let record = inventory_by_id
                .get(row.source_evidence_id.as_str())
                .ok_or_else(|| format!(
                    "event-extraction row references evidence outside inventory: {:?}",
                    row.source_evidence_id
                ))?;
            let admission = admission_by_id
                .get(row.source_evidence_id.as_str())
                .ok_or_else(|| format!(
                    "event-extraction row has no trusted admission: {:?}",
                    row.source_evidence_id
                ))?;
            if row.source_sha256 != record.source.source_sha256
                || row.payload_sha256 != record.content_sha256
            {
                return Err(format!(
                    "event-extraction row source digest mismatch for {:?}",
                    row.source_evidence_id
                ));
            }
            if admission.source_trajectory_id != record.source.trajectory_id
                || admission.source_sha256 != record.source.source_sha256
                || admission.source_pointer != record.source.source_pointer
                || admission.source_step_id != record.source.step_id
                || admission.source_step_index != record.source.step_index
                || admission.valid_from_unix_ms != record.valid_from_unix_ms
                || admission.valid_until_unix_ms != record.valid_until_unix_ms
            {
                return Err(format!(
                    "trusted admission source identity mismatch for {:?}",
                    row.source_evidence_id
                ));
            }
            all_rows_complete &= row.status == EventExtractionStatus::Complete;
            let trusted_speaker = speaker_by_id[row.source_evidence_id.as_str()];
            for event in &row.events {
                event.validate()?;
                if event.speaker_id != trusted_speaker.speaker_id
                    || event.speaker_role != trusted_speaker.speaker_role
                {
                    return Err(format!(
                        "extracted event speaker does not match trusted source attribution for {:?}",
                        row.source_evidence_id
                    ));
                }
                if !event_ids.insert(event.event_id.as_str()) {
                    return Err(format!("duplicate extracted event ID {:?}", event.event_id));
                }
            }
        }

        if rows_by_id.len() != inventory_by_id.len() {
            let missing: Vec<&str> = inventory_by_id
                .keys()
                .copied()
                .filter(|evidence_id| !rows_by_id.contains_key(evidence_id))
                .collect();
            return Err(format!(
                "event extraction omitted inventoried source records: {:?}",
                missing
            ));
        }
        if admission_by_id.keys().any(|evidence_id| !inventory_by_id.contains_key(evidence_id)) {
            return Err("unused source admission outside inventory".into());
        }

        let extraction_batch_sha256 = hash_serializable(self, "event extraction batch")?;
        Ok(EventExtractionCoverage {
            source_inventory_sha256: inventory.receipt.inventory_sha256.clone(),
            extraction_batch_sha256,
            source_record_count: inventory.items.len(),
            extraction_row_count: rows_by_id.len(),
            extracted_event_count: event_ids.len(),
            all_rows_complete,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventExtractionCoverage {
    pub source_inventory_sha256: String,
    pub extraction_batch_sha256: String,
    pub source_record_count: usize,
    pub extraction_row_count: usize,
    pub extracted_event_count: usize,
    pub all_rows_complete: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventEnumerationQuery {
    pub decision_id: String,
    pub principal_id: String,
    pub scope_id: String,
    pub valid_at_unix_ms: u64,
    pub known_at_unix_ms: u64,
    pub event_type_id: String,
    pub entity_id: String,
    pub actor_id: Option<String>,
    pub required_speaker_id: Option<String>,
    pub required_speaker_role: Option<EventSpeakerRole>,
    pub require_speaker_attribution: bool,
    pub occurred_from_unix_ms: Option<u64>,
    /// Exclusive upper boundary.
    pub occurred_until_unix_ms: Option<u64>,
    pub minimum_distinct_sessions: u32,
    pub minimum_independent_families: u32,
}

impl EventEnumerationQuery {
    pub fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("decision_id", self.decision_id.as_str()),
            ("principal_id", self.principal_id.as_str()),
            ("scope_id", self.scope_id.as_str()),
            ("event_type_id", self.event_type_id.as_str()),
            ("entity_id", self.entity_id.as_str()),
        ] {
            require_non_empty(value, field)?;
        }
        if let Some(actor_id) = &self.actor_id {
            require_non_empty(actor_id, "actor_id")?;
        }
        if let Some(speaker_id) = &self.required_speaker_id {
            require_non_empty(speaker_id, "required_speaker_id")?;
        }
        if self.occurred_from_unix_ms
            .zip(self.occurred_until_unix_ms)
            .is_some_and(|(start, end)| start >= end)
        {
            return Err("event occurrence interval must be non-empty and half-open".into());
        }
        if self.require_speaker_attribution
            && self.required_speaker_id.is_none()
            && self.required_speaker_role.is_none()
        {
            return Err("speaker attribution requires an explicit speaker ID or role".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnumeratedMemoryEvent {
    pub source_evidence_id: String,
    pub source_family_id: String,
    pub source_session_id: String,
    pub event: ExtractedMemoryEvent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventEnumerationReceipt {
    pub protocol_version: String,
    pub decision_id: String,
    pub source_inventory_sha256: String,
    pub extraction_batch_sha256: String,
    pub query_sha256: String,
    pub matching_event_ids_sha256: String,
    pub enumerated_source_count: usize,
    pub extraction_row_count: usize,
    pub matching_event_count: usize,
    pub unattributed_relevant_event_count: usize,
    pub distinct_session_count: usize,
    pub distinct_source_family_count: usize,
    pub inventory_complete: bool,
    pub extraction_complete: bool,
    pub speaker_attribution_complete: bool,
    pub answer_is_permitted: bool,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventEnumerationResult {
    pub matching_events: Vec<EnumeratedMemoryEvent>,
    pub receipt: EventEnumerationReceipt,
}

impl EventEnumerationResult {
    pub fn verify_internal_consistency(&self) -> Result<(), String> {
        let expected_ids: Vec<&str> = self.matching_events
            .iter()
            .map(|item| item.event.event_id.as_str())
            .collect();
        if expected_ids.len() != self.receipt.matching_event_count {
            return Err("enumeration match count does not match result items".into());
        }
        if sha256_hex(&serde_json::to_vec(&expected_ids)
            .map_err(|error| format!("cannot serialize event IDs: {error}"))?)
            != self.receipt.matching_event_ids_sha256
        {
            return Err("enumerated event ID digest mismatch".into());
        }
        let computed = event_receipt_digest(&self.receipt)?;
        if computed != self.receipt.receipt_sha256 {
            return Err("event enumeration receipt digest mismatch".into());
        }
        let mut event_ids = BTreeSet::new();
        let mut sessions = BTreeSet::new();
        let mut families = BTreeSet::new();
        for item in &self.matching_events {
            require_non_empty(&item.source_evidence_id, "source_evidence_id")?;
            require_non_empty(&item.source_family_id, "source_family_id")?;
            require_non_empty(&item.source_session_id, "source_session_id")?;
            item.event.validate()?;
            if !event_ids.insert(item.event.event_id.as_str()) {
                return Err(format!("duplicate result event ID {:?}", item.event.event_id));
            }
            sessions.insert(item.source_session_id.as_str());
            families.insert(item.source_family_id.as_str());
        }
        if sessions.len() != self.receipt.distinct_session_count
            || families.len() != self.receipt.distinct_source_family_count
        {
            return Err("enumeration diversity counts do not match returned events".into());
        }
        if self.receipt.answer_is_permitted
            && (!self.receipt.inventory_complete
                || !self.receipt.extraction_complete
                || !self.receipt.speaker_attribution_complete)
        {
            return Err("answer permission exceeds enumeration coverage".into());
        }
        Ok(())
    }

    /// Re-run the closed-world derivation against the original immutable
    /// inventory, admission envelopes, extraction batch and query. A digest
    /// alone is not a signature: this independent recomputation is the
    /// authoritative check before a count is used for an answer.
    pub fn verify_against(
        &self,
        inventory: &EvidenceInventorySnapshot,
        batch: &EventExtractionBatch,
        admissions: &[ProofCandidateAdmission],
        speaker_attributions: &[TrustedSpeakerAttribution],
        query: &EventEnumerationQuery,
    ) -> Result<(), String> {
        self.verify_internal_consistency()?;
        let recomputed = enumerate_events_closed_world(
            inventory, batch, admissions, speaker_attributions, query
        )?;
        if &recomputed != self {
            return Err("event enumeration result does not match independent recomputation".into());
        }
        Ok(())
    }
}

/// Exhaustively filter a fully covered extraction batch. This is not a
/// similarity search: every active source in the exact inventory must have a
/// row, and every extracted event is checked against the typed query.
pub fn enumerate_events_closed_world(
    inventory: &EvidenceInventorySnapshot,
    batch: &EventExtractionBatch,
    admissions: &[ProofCandidateAdmission],
    speaker_attributions: &[TrustedSpeakerAttribution],
    query: &EventEnumerationQuery,
) -> Result<EventEnumerationResult, String> {
    query.validate()?;
    if inventory.receipt.scope_id != query.scope_id
        || inventory.receipt.valid_at_unix_ms != query.valid_at_unix_ms
        || inventory.receipt.known_at_unix_ms != query.known_at_unix_ms
    {
        return Err("event enumeration query does not match evidence inventory coordinates".into());
    }
    let coverage = batch.validate_against(
        inventory,
        admissions,
        speaker_attributions,
        &query.principal_id,
    )?;
    let admission_by_id: BTreeMap<&str, &ProofCandidateAdmission> = admissions
        .iter()
        .map(|admission| (admission.evidence_id.as_str(), admission))
        .collect();

    let mut matching_events = Vec::new();
    let mut unattributed_relevant_event_count = 0usize;
    let mut candidate_event_count = 0usize;

    for row in &batch.rows {
        let admission = admission_by_id[row.source_evidence_id.as_str()];
        for event in &row.events {
            if !base_event_matches(event, query) {
                continue;
            }
            candidate_event_count += 1;
            if query.require_speaker_attribution
                && (event.speaker_role == EventSpeakerRole::Unknown || event.speaker_id.is_none())
            {
                unattributed_relevant_event_count += 1;
                continue;
            }
            if !speaker_matches(event, query) {
                continue;
            }
            matching_events.push(EnumeratedMemoryEvent {
                source_evidence_id: row.source_evidence_id.clone(),
                source_family_id: admission.source_family_id.clone(),
                source_session_id: admission.source_trajectory_id.clone(),
                event: event.clone(),
            });
        }
    }

    matching_events.sort_by(|left, right| {
        left.event.event_id.cmp(&right.event.event_id)
            .then_with(|| left.source_evidence_id.cmp(&right.source_evidence_id))
    });
    let matching_event_ids: Vec<&str> = matching_events
        .iter()
        .map(|item| item.event.event_id.as_str())
        .collect();
    let matching_event_ids_sha256 = hash_serializable(&matching_event_ids, "matching event IDs")?;
    let distinct_sessions: BTreeSet<&str> = matching_events
        .iter()
        .map(|item| item.source_session_id.as_str())
        .collect();
    let distinct_families: BTreeSet<&str> = matching_events
        .iter()
        .map(|item| item.source_family_id.as_str())
        .collect();
    let inventory_complete = inventory.receipt.complete
        && coverage.source_record_count == coverage.extraction_row_count;
    let extraction_complete = coverage.all_rows_complete;
    let speaker_attribution_complete = !query.require_speaker_attribution
        || candidate_event_count == 0
        || unattributed_relevant_event_count == 0;
    let diversity_satisfied = distinct_sessions.len() >= query.minimum_distinct_sessions as usize
        && distinct_families.len() >= query.minimum_independent_families as usize;
    let answer_is_permitted = inventory_complete
        && extraction_complete
        && speaker_attribution_complete
        && diversity_satisfied;

    let mut receipt = EventEnumerationReceipt {
        protocol_version: EVENT_ENUMERATION_PROTOCOL_VERSION.into(),
        decision_id: query.decision_id.clone(),
        source_inventory_sha256: coverage.source_inventory_sha256,
        extraction_batch_sha256: coverage.extraction_batch_sha256,
        query_sha256: hash_serializable(query, "event enumeration query")?,
        matching_event_ids_sha256,
        enumerated_source_count: coverage.source_record_count,
        extraction_row_count: coverage.extraction_row_count,
        matching_event_count: matching_events.len(),
        unattributed_relevant_event_count,
        distinct_session_count: distinct_sessions.len(),
        distinct_source_family_count: distinct_families.len(),
        inventory_complete,
        extraction_complete,
        speaker_attribution_complete,
        answer_is_permitted,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = event_receipt_digest(&receipt)?;
    let result = EventEnumerationResult {
        matching_events,
        receipt,
    };
    result.verify_internal_consistency()?;
    Ok(result)
}

fn base_event_matches(event: &ExtractedMemoryEvent, query: &EventEnumerationQuery) -> bool {
    event.event_type_id == query.event_type_id
        && event.entity_id == query.entity_id
        && query.actor_id.as_ref().is_none_or(|actor_id| event.actor_id.as_ref() == Some(actor_id))
        && query.occurred_from_unix_ms
            .is_none_or(|start| event.occurred_at_unix_ms.is_some_and(|time| time >= start))
        && query.occurred_until_unix_ms
            .is_none_or(|end| event.occurred_at_unix_ms.is_some_and(|time| time < end))
}

fn speaker_matches(event: &ExtractedMemoryEvent, query: &EventEnumerationQuery) -> bool {
    query.required_speaker_id.as_ref()
        .is_none_or(|speaker_id| event.speaker_id.as_ref() == Some(speaker_id))
        && query.required_speaker_role
            .is_none_or(|role| event.speaker_role == role)
}

fn event_receipt_digest(receipt: &EventEnumerationReceipt) -> Result<String, String> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    hash_serializable(&canonical, "event enumeration receipt")
}

fn hash_serializable<T: Serialize>(value: &T, label: &str) -> Result<String, String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("cannot serialize {label}: {error}"))?;
    Ok(sha256_hex(&bytes))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_sha256(value: &str, field: &str) -> Result<(), String> {
    if value.len() != 64
        || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{field} must be a 64-character lowercase SHA-256 digest"));
    }
    Ok(())
}

fn require_non_empty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(format!("{field} must be non-empty and trimmed"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_evidence::{
        EvidenceIndex, EvidencePayload, EvidenceRecord, EvidenceSource, EVIDENCE_SCHEMA_VERSION,
    };
    use std::path::Path;
    use std::collections::BTreeSet;

    fn digest(value: &str) -> String {
        sha256_hex(value.as_bytes())
    }

    fn inventory_and_admissions() -> (EvidenceInventorySnapshot, Vec<ProofCandidateAdmission>) {
        let source_a = serde_json::to_vec(&serde_json::json!({"evidence":"user booked a hotel"})).unwrap();
        let source_b = serde_json::to_vec(&serde_json::json!({"evidence":"assistant suggested a hotel"})).unwrap();
        let mut index = EvidenceIndex::new();
        let record_a = EvidenceRecord {
            schema_version: EVIDENCE_SCHEMA_VERSION.into(),
            evidence_id: "evidence-user".into(),
            hdc_id: 1,
            scope_id: "scope-1".into(),
            source: EvidenceSource {
                trajectory_id: "session-user".into(),
                step_id: Some("turn-1".into()),
                step_index: Some(1),
                source_sha256: digest(std::str::from_utf8(&source_a).unwrap()),
                source_pointer: "/evidence".into(),
            },
            payload: EvidencePayload::Text { value: "user booked a hotel".into() },
            content_sha256: digest("user booked a hotel"),
            lifecycle: crate::memory_evidence::EvidenceLifecycle::Active,
            occurred_at_unix_ms: Some(100),
            valid_from_unix_ms: Some(0),
            valid_until_unix_ms: None,
            superseded_by: None,
        };
        let record_b = EvidenceRecord {
            schema_version: EVIDENCE_SCHEMA_VERSION.into(),
            evidence_id: "evidence-assistant".into(),
            hdc_id: 2,
            scope_id: "scope-1".into(),
            source: EvidenceSource {
                trajectory_id: "session-assistant".into(),
                step_id: Some("turn-2".into()),
                step_index: Some(2),
                source_sha256: digest(std::str::from_utf8(&source_b).unwrap()),
                source_pointer: "/evidence".into(),
            },
            payload: EvidencePayload::Text { value: "assistant suggested a hotel".into() },
            content_sha256: digest("assistant suggested a hotel"),
            lifecycle: crate::memory_evidence::EvidenceLifecycle::Active,
            occurred_at_unix_ms: Some(110),
            valid_from_unix_ms: Some(0),
            valid_until_unix_ms: None,
            superseded_by: None,
        };
        index.insert(record_a.clone(), &source_a).unwrap();
        index.insert(record_b.clone(), &source_b).unwrap();
        let inventory = index
            .enumerate_active_at_snapshot("scope-1", 200, 300, Path::new("."), 10)
            .unwrap();
        let admissions = vec![
            admission_for(&record_a, &inventory.receipt, "family-user"),
            admission_for(&record_b, &inventory.receipt, "family-assistant"),
        ];
        (inventory, admissions)
    }

    fn admission_for(
        record: &EvidenceRecord,
        inventory: &EvidenceInventoryReceipt,
        family: &str,
    ) -> ProofCandidateAdmission {
        ProofCandidateAdmission {
            evidence_id: record.evidence_id.clone(),
            source_revision_id: format!("revision-{}", record.evidence_id),
            source_trajectory_id: record.source.trajectory_id.clone(),
            source_step_id: record.source.step_id.clone(),
            source_step_index: record.source.step_index,
            source_sha256: record.source.source_sha256.clone(),
            source_pointer: record.source.source_pointer.clone(),
            source_family_id: family.into(),
            scope_id: record.scope_id.clone(),
            index_state_sha256: inventory.index_state_sha256.clone(),
            recorded_at_unix_ms: 50,
            retracted_at_unix_ms: None,
            valid_from_unix_ms: record.valid_from_unix_ms,
            valid_until_unix_ms: record.valid_until_unix_ms,
            authorized_principal_id: "principal-1".into(),
            authorization_accepted: true,
            authorization_receipt_sha256: digest("authorization receipt"),
            conflicts_with: BTreeSet::new(),
            token_cost: 5,
        }
    }

    fn speaker_attributions(
        inventory: &EvidenceInventorySnapshot,
        admissions: &[ProofCandidateAdmission],
    ) -> Vec<TrustedSpeakerAttribution> {
        inventory.items.iter().map(|item| {
            let admission = admissions.iter()
                .find(|admission| admission.evidence_id == item.record.evidence_id)
                .unwrap();
            let (speaker_id, speaker_role) = if item.record.evidence_id == "evidence-user" {
                (Some("user-1".into()), EventSpeakerRole::User)
            } else {
                (Some("assistant".into()), EventSpeakerRole::Assistant)
            };
            let mut attribution = TrustedSpeakerAttribution {
                evidence_id: item.record.evidence_id.clone(),
                source_revision_id: admission.source_revision_id.clone(),
                source_sha256: item.record.source.source_sha256.clone(),
                source_pointer: item.record.source.source_pointer.clone(),
                source_trajectory_id: item.record.source.trajectory_id.clone(),
                speaker_id,
                speaker_role,
                attribution_receipt_sha256: String::new(),
            };
            attribution.attribution_receipt_sha256 =
                attribution.computed_receipt_sha256().unwrap();
            attribution
        }).collect()
    }

    fn batch(inventory: &EvidenceInventorySnapshot) -> EventExtractionBatch {
        let mut rows = Vec::new();
        for item in &inventory.items {
            let events = if item.record.evidence_id == "evidence-user" {
                vec![ExtractedMemoryEvent {
                    event_id: "event-user-booked".into(),
                    event_type_id: "hotel.booking".into(),
                    entity_id: "hotel-123".into(),
                    actor_id: Some("user-1".into()),
                    speaker_id: Some("user-1".into()),
                    speaker_role: EventSpeakerRole::User,
                    occurred_at_unix_ms: Some(100),
                }]
            } else {
                vec![ExtractedMemoryEvent {
                    event_id: "event-assistant-suggested".into(),
                    event_type_id: "hotel.booking".into(),
                    entity_id: "hotel-123".into(),
                    actor_id: Some("assistant".into()),
                    speaker_id: Some("assistant".into()),
                    speaker_role: EventSpeakerRole::Assistant,
                    occurred_at_unix_ms: Some(110),
                }]
            };
            rows.push(EventExtractionRow {
                source_evidence_id: item.record.evidence_id.clone(),
                source_sha256: item.record.source.source_sha256.clone(),
                payload_sha256: item.record.content_sha256.clone(),
                status: EventExtractionStatus::Complete,
                events,
            });
        }
        EventExtractionBatch {
            protocol_version: EVENT_ENUMERATION_PROTOCOL_VERSION.into(),
            source_inventory_sha256: inventory.receipt.inventory_sha256.clone(),
            extractor_id: "event-extractor-v1".into(),
            extractor_sha256: digest("extractor"),
            extraction_schema_sha256: digest("event schema"),
            rows,
        }
    }

    fn query() -> EventEnumerationQuery {
        EventEnumerationQuery {
            decision_id: "decision-1".into(),
            principal_id: "principal-1".into(),
            scope_id: "scope-1".into(),
            valid_at_unix_ms: 200,
            known_at_unix_ms: 300,
            event_type_id: "hotel.booking".into(),
            entity_id: "hotel-123".into(),
            actor_id: None,
            required_speaker_id: Some("user-1".into()),
            required_speaker_role: Some(EventSpeakerRole::User),
            require_speaker_attribution: true,
            occurred_from_unix_ms: Some(0),
            occurred_until_unix_ms: Some(200),
            minimum_distinct_sessions: 1,
            minimum_independent_families: 1,
        }
    }

    #[test]
    fn closed_world_count_uses_user_speaker_and_excludes_assistant_advice() {
        let (inventory, admissions) = inventory_and_admissions();
        let result = enumerate_events_closed_world(
            &inventory,
            &batch(&inventory),
            &admissions,
            &query(),
        ).unwrap();

        assert_eq!(result.receipt.enumerated_source_count, 2);
        assert_eq!(result.receipt.extraction_row_count, 2);
        assert_eq!(result.receipt.matching_event_count, 1);
        assert_eq!(result.matching_events[0].event.event_id, "event-user-booked");
        assert_eq!(result.receipt.distinct_session_count, 1);
        assert!(result.receipt.answer_is_permitted);
        result.verify_internal_consistency().unwrap();
    }

    #[test]
    fn missing_source_row_prevents_a_closed_world_count() {
        let (inventory, admissions) = inventory_and_admissions();
        let mut extraction = batch(&inventory);
        extraction.rows.pop();
        assert!(enumerate_events_closed_world(
            &inventory, &extraction, &admissions, &speaker_attributions(&inventory, &admissions), &query()
        ).unwrap_err().contains("omitted inventoried source records"));
    }

    #[test]
    fn incomplete_extraction_status_disables_answer_permission() {
        let (inventory, admissions) = inventory_and_admissions();
        let mut extraction = batch(&inventory);
        extraction.rows[1].status = EventExtractionStatus::ParseFailure;
        let result = enumerate_events_closed_world(
            &inventory, &extraction, &admissions, &speaker_attributions(&inventory, &admissions), &query()
        ).unwrap();
        assert!(!result.receipt.extraction_complete);
        assert!(!result.receipt.answer_is_permitted);
    }

    #[test]
    fn unknown_speaker_on_a_relevant_event_forces_abstention() {
        let (inventory, admissions) = inventory_and_admissions();
        let mut extraction = batch(&inventory);
        extraction.rows[1].events[0].speaker_id = None;
        extraction.rows[1].events[0].speaker_role = EventSpeakerRole::Unknown;
        let result = enumerate_events_closed_world(
            &inventory, &extraction, &admissions, &speaker_attributions(&inventory, &admissions), &query()
        ).unwrap();
        assert_eq!(result.receipt.unattributed_relevant_event_count, 1);
        assert!(!result.receipt.speaker_attribution_complete);
        assert!(!result.receipt.answer_is_permitted);
    }

    #[test]
    fn invalid_bitemporal_admission_cannot_certify_closed_world_event_coverage() {
        let (inventory, mut admissions) = inventory_and_admissions();
        admissions[0].recorded_at_unix_ms = inventory.receipt.known_at_unix_ms + 1;
        assert!(enumerate_events_closed_world(
            &inventory, &batch(&inventory), &admissions, &speaker_attributions(&inventory, &admissions), &query()
        ).unwrap_err().contains("not current at the inventory known_at"));

        let (inventory, mut admissions) = inventory_and_admissions();
        admissions[0].valid_until_unix_ms = Some(50);
        assert!(enumerate_events_closed_world(
            &inventory, &batch(&inventory), &admissions, &speaker_attributions(&inventory, &admissions), &query()
        ).unwrap_err().contains("source identity mismatch"));
    }

    #[test]
    fn changed_inventory_or_admission_snapshot_is_rejected() {
        let (inventory, mut admissions) = inventory_and_admissions();
        admissions[0].index_state_sha256 = digest("another snapshot");
        assert!(enumerate_events_closed_world(
            &inventory, &batch(&inventory), &admissions, &speaker_attributions(&inventory, &admissions), &query()
        ).unwrap_err().contains("different EvidenceIndex snapshot"));
    }

    #[test]
    fn authoritative_verification_recomputes_all_answer_gates() {
        let (inventory, admissions) = inventory_and_admissions();
        let extraction = batch(&inventory);
        let query = query();
        let result = enumerate_events_closed_world(
            &inventory, &extraction, &admissions, &speaker_attributions(&inventory, &admissions), &query
        ).unwrap();
        result.verify_against(&inventory, &extraction, &admissions, &speaker_attributions(&inventory, &admissions), &query).unwrap();

        let mut tampered = result.clone();
        tampered.receipt.unattributed_relevant_event_count = 0;
        tampered.receipt.speaker_attribution_complete = false;
        tampered.receipt.answer_is_permitted = false;
        tampered.receipt.receipt_sha256 = event_receipt_digest(&tampered.receipt).unwrap();
        assert!(tampered
            .verify_against(&inventory, &extraction, &admissions, &speaker_attributions(&inventory, &admissions), &query)
            .unwrap_err()
            .contains("independent recomputation"));
    }

    #[test]
    fn receipt_detects_mutated_matching_event() {
        let (inventory, admissions) = inventory_and_admissions();
        let mut result = enumerate_events_closed_world(
            &inventory, &batch(&inventory), &admissions, &speaker_attributions(&inventory, &admissions), &query()
        ).unwrap();
        result.matching_events[0].event.event_id = "forged-event".into();
        assert!(result.verify_internal_consistency().is_err());
    }
}
