// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bitemporal, rights-scoped evidence events for existing memory tiers.
//!
//! This module is an additive evidence spine, not another vector store. HDC,
//! semantic and episodic memory remain responsible for indexing and retrieval;
//! this ledger answers whether a memory claim is admissible for a particular
//! purpose, valid-time and knowledge-time view.
//!
//! Important boundary: this module validates declared structure and
//! relationships. It does not authenticate issuers, verify digital
//! signatures, adjudicate truth, enforce storage immutability, provide a
//! durable external witness, or replace the decision-scoped qualification
//! service in the MACE adapter.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA_VERSION: &str = "memory_evidence_event_v1";

/// Event types are append-only transitions; corrections and erasures do not
/// overwrite or silently delete earlier records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceEventKind {
    Assertion,
    Correction,
    Retraction,
    ConsentGranted,
    ConsentRevoked,
    ErasureTombstone,
}

/// Commitment to the source material from which a claim was produced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceCommitment {
    pub source_id: String,
    pub source_revision: String,
    /// Accepted encodings: sha256: followed by 64 lowercase hex, or blake3: likewise.
    pub source_digest: String,
}

/// Rights and purpose restrictions attached to a memory assertion or consent
/// record. An empty allowed_purposes set is invalid.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryRights {
    pub allowed_purposes: Vec<String>,
    pub consent_required: bool,
    /// Exact ConsentGranted event ID, not a free-form statement that consent
    /// exists. The referenced event is checked at query time.
    pub consent_grant_event_id: Option<String>,
    pub retention_until_millis: Option<u64>,
}

/// Append-only event record for an assertion, correction, revocation or
/// erasure. known_at_millis is when this memory system learned the event;
/// valid_from_millis / valid_until_millis describe when the claim applies
/// in the world. They must not be conflated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceEvent {
    pub schema_version: String,
    pub event_id: String,
    /// Monotonic append position within this ledger.
    pub sequence: u64,
    pub kind: EvidenceEventKind,
    /// Stable identity for the proposition being asserted/corrected/retracted.
    pub claim_key: Option<String>,
    pub source: Option<SourceCommitment>,
    /// Digest of the normalized claim payload, distinct from source bytes.
    pub payload_digest: Option<String>,
    pub issuer_id: String,
    /// Declared issuer authority scopes. These are not cryptographically
    /// authenticated by this module.
    pub authority_scopes: Vec<String>,
    pub rights: MemoryRights,
    /// Source occurrence time; absent for control events without source data.
    pub observed_at_millis: Option<u64>,
    /// Knowledge/ingestion time in this memory system.
    pub known_at_millis: u64,
    /// World-validity interval, inclusive at the beginning and exclusive at
    /// the end. None means no declared end, not proof of eternal validity.
    pub valid_from_millis: u64,
    pub valid_until_millis: Option<u64>,
    /// Optional source/derivation parents. Every parent must be an earlier
    /// assertion/correction and every parent remains required for use.
    pub parents: Vec<String>,
    /// Targets for correction, retraction, consent revocation or erasure.
    pub targets: Vec<String>,
    /// Consent ID used only by ConsentGranted / ConsentRevoked records and
    /// referenced by a claim when a consent grant is required.
    pub consent_id: Option<String>,
}

/// Bitemporal query and use-policy context. Exact scope matching is
/// intentional: wildcard/implicit inheritance is not inferred.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceQuery {
    pub valid_at_millis: u64,
    pub known_at_millis: u64,
    pub purpose: String,
    pub required_authority: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IneligibilityReason {
    UnknownEvent,
    NotAClaim,
    NotYetKnown,
    OutsideValidityInterval,
    PurposeNotPermitted,
    AuthorityNotPermitted,
    RetentionExpired,
    Superseded,
    Retracted,
    Erased,
    ConsentMissing,
    ConsentGrantUnavailable,
    ConsentRevoked,
    ConsentOutOfScope,
    ParentNotAdmissible,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceDecision {
    pub event_id: String,
    pub admissible: bool,
    pub reason: Option<IneligibilityReason>,
    pub blocking_event_id: Option<String>,
}

impl EvidenceDecision {
    fn admit(event_id: &str) -> Self {
        Self {
            event_id: event_id.to_string(),
            admissible: true,
            reason: None,
            blocking_event_id: None,
        }
    }

    fn reject(
        event_id: &str,
        reason: IneligibilityReason,
        blocking_event_id: Option<&str>,
    ) -> Self {
        Self {
            event_id: event_id.to_string(),
            admissible: false,
            reason: Some(reason),
            blocking_event_id: blocking_event_id.map(str::to_string),
        }
    }
}

/// In-memory ledger implementation. Durable persistence, external witnessing
/// and cryptographic issuer verification are deliberately separate services.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Default)]
pub struct EvidenceLedger {
    events: Vec<EvidenceEvent>,
}

/// A memory value paired with the evidence event that qualifies its use.
/// This wrapper lets existing semantic/episodic types adopt provenance
/// without breaking their public struct-literal constructors.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceBound<T> {
    value: T,
    evidence_event_id: String,
}

impl<T> EvidenceBound<T> {
    pub fn value(&self) -> &T {
        &self.value
    }

    pub fn evidence_event_id(&self) -> &str {
        &self.evidence_event_id
    }

    pub fn into_value(self) -> T {
        self.value
    }
}

impl<'de> Deserialize<'de> for EvidenceLedger {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct WireLedger {
            events: Vec<EvidenceEvent>,
        }

        let wire = WireLedger::deserialize(deserializer)?;
        validate_events(&wire.events).map_err(serde::de::Error::custom)?;
        Ok(Self { events: wire.events })
    }
}

impl EvidenceLedger {
    pub fn new(events: Vec<EvidenceEvent>) -> Result<Self, String> {
        validate_events(&events)?;
        Ok(Self { events })
    }

    pub fn empty() -> Self {
        Self::default()
    }

    pub fn events(&self) -> &[EvidenceEvent] {
        &self.events
    }

    /// Append a new event only if the complete resulting ledger remains
    /// structurally valid. Existing events are never modified or removed.
    pub fn append(&mut self, event: EvidenceEvent) -> Result<(), String> {
        if let Some(last) = self.events.last() {
            if event.sequence <= last.sequence {
                return Err("new evidence sequence must exceed the last appended sequence".to_string());
            }
        }
        let mut candidate = self.events.clone();
        candidate.push(event.clone());
        validate_events(&candidate)?;
        self.events.push(event);
        Ok(())
    }

    /// Evaluate one record at a valid-time / knowledge-time point with
    /// explicit purpose and authority requirements.
    pub fn evaluate(&self, event_id: &str, query: &EvidenceQuery) -> EvidenceDecision {
        if query.purpose.trim().is_empty() || query.required_authority.trim().is_empty() {
            return EvidenceDecision::reject(
                event_id,
                IneligibilityReason::PurposeNotPermitted,
                None,
            );
        }
        let Some(event) = self.events.iter().find(|event| event.event_id == event_id) else {
            return EvidenceDecision::reject(event_id, IneligibilityReason::UnknownEvent, None);
        };
        self.evaluate_inner(event, query, &mut BTreeSet::new())
    }

    /// Pair an existing memory value with an admissible evidence event.
    /// Downstream consumers should accept this wrapper, not an unqualified
    /// value, at decision/action boundaries.
    pub fn bind<T>(
        &self,
        value: T,
        event_id: &str,
        query: &EvidenceQuery,
    ) -> Result<EvidenceBound<T>, EvidenceDecision> {
        let decision = self.evaluate(event_id, query);
        if decision.admissible {
            Ok(EvidenceBound {
                value,
                evidence_event_id: event_id.to_string(),
            })
        } else {
            Err(decision)
        }
    }

    /// Return only admissible assertion/correction IDs, in append order.
    pub fn eligible_event_ids(&self, query: &EvidenceQuery) -> Vec<String> {
        self.events
            .iter()
            .filter(|event| matches!(event.kind, EvidenceEventKind::Assertion | EvidenceEventKind::Correction))
            .filter_map(|event| {
                let decision = self.evaluate(&event.event_id, query);
                decision.admissible.then(|| event.event_id.clone())
            })
            .collect()
    }

    fn evaluate_inner(
        &self,
        event: &EvidenceEvent,
        query: &EvidenceQuery,
        visiting: &mut BTreeSet<String>,
    ) -> EvidenceDecision {
        if !visiting.insert(event.event_id.clone()) {
            return EvidenceDecision::reject(
                &event.event_id,
                IneligibilityReason::ParentNotAdmissible,
                Some(&event.event_id),
            );
        }

        let reject = |reason, blocker: Option<&str>| {
            EvidenceDecision::reject(&event.event_id, reason, blocker)
        };

        if !matches!(event.kind, EvidenceEventKind::Assertion | EvidenceEventKind::Correction) {
            return reject(IneligibilityReason::NotAClaim, None);
        }
        if event.known_at_millis > query.known_at_millis {
            return reject(IneligibilityReason::NotYetKnown, None);
        }
        if query.valid_at_millis < event.valid_from_millis
            || event.valid_until_millis.is_some_and(|until| query.valid_at_millis >= until)
        {
            return reject(IneligibilityReason::OutsideValidityInterval, None);
        }
        if !event.rights.allowed_purposes.iter().any(|purpose| purpose == &query.purpose) {
            return reject(IneligibilityReason::PurposeNotPermitted, None);
        }
        if !event.authority_scopes.iter().any(|scope| scope == &query.required_authority) {
            return reject(IneligibilityReason::AuthorityNotPermitted, None);
        }
        if event.rights.retention_until_millis.is_some_and(|until| query.known_at_millis >= until) {
            return reject(IneligibilityReason::RetentionExpired, None);
        }

        if let Some(blocker) = self.invalidating_event(event, query) {
            let reason = match blocker.kind {
                EvidenceEventKind::Correction => IneligibilityReason::Superseded,
                EvidenceEventKind::Retraction => IneligibilityReason::Retracted,
                EvidenceEventKind::ErasureTombstone => IneligibilityReason::Erased,
                _ => IneligibilityReason::ParentNotAdmissible,
            };
            return reject(reason, Some(&blocker.event_id));
        }

        if event.rights.consent_required && event.rights.consent_grant_event_id.is_none() {
            return reject(IneligibilityReason::ConsentMissing, None);
        }
        if let Some(grant_id) = event.rights.consent_grant_event_id.as_deref() {
            let Some(grant) = self.events.iter().find(|candidate| candidate.event_id == grant_id) else {
                return reject(IneligibilityReason::ConsentGrantUnavailable, Some(grant_id));
            };
            if grant.kind != EvidenceEventKind::ConsentGranted || grant.consent_id != event.consent_id {
                return reject(IneligibilityReason::ConsentGrantUnavailable, Some(grant_id));
            }
            if grant.known_at_millis > query.known_at_millis
                || query.valid_at_millis < grant.valid_from_millis
                || grant.valid_until_millis.is_some_and(|until| query.valid_at_millis >= until)
                || !grant.rights.allowed_purposes.iter().any(|purpose| purpose == &query.purpose)
                || grant.rights.retention_until_millis.is_some_and(|until| query.known_at_millis >= until)
                || self.invalidating_event(grant, query).is_some()
            {
                return reject(IneligibilityReason::ConsentOutOfScope, Some(grant_id));
            }
            if let Some(revoke) = self.events.iter().find(|candidate| {
                candidate.kind == EvidenceEventKind::ConsentRevoked
                    && candidate.issuer_id == grant.issuer_id
                    && candidate.targets.iter().any(|target| target == grant_id)
                    && candidate.known_at_millis <= query.known_at_millis
                    && query.valid_at_millis >= candidate.valid_from_millis
                    && candidate.valid_until_millis.is_none_or(|until| query.valid_at_millis < until)
            }) {
                return reject(IneligibilityReason::ConsentRevoked, Some(&revoke.event_id));
            }
        }

        for parent_id in &event.parents {
            let Some(parent) = self.events.iter().find(|candidate| candidate.event_id == parent_id) else {
                return reject(IneligibilityReason::ParentNotAdmissible, Some(parent_id));
            };
            let decision = self.evaluate_inner(parent, query, visiting);
            if !decision.admissible {
                return reject(IneligibilityReason::ParentNotAdmissible, Some(parent_id));
            }
        }

        visiting.remove(&event.event_id);
        EvidenceDecision::admit(&event.event_id)
    }

    fn invalidating_event<'a>(
        &'a self,
        event: &EvidenceEvent,
        query: &EvidenceQuery,
    ) -> Option<&'a EvidenceEvent> {
        self.events.iter().find(|candidate| {
            let temporal_match = candidate.known_at_millis <= query.known_at_millis
                && query.valid_at_millis >= candidate.valid_from_millis
                && candidate.valid_until_millis.is_none_or(|until| query.valid_at_millis < until);
            let scope_match = match candidate.kind {
                EvidenceEventKind::Correction | EvidenceEventKind::Retraction => {
                    candidate.authority_scopes.iter().any(|scope| scope == &query.required_authority)
                        && candidate.rights.allowed_purposes.iter().any(|purpose| purpose == &query.purpose)
                }
                // Erasure tombstones are purpose-independent; the caller must
                // only append them after the external rights/authority verifier
                // has accepted the erasure receipt.
                EvidenceEventKind::ErasureTombstone => {
                    candidate.authority_scopes.iter().any(|scope| scope == "memory-erasure")
                }
                _ => false,
            };
            candidate.sequence > event.sequence
                && temporal_match
                && scope_match
                && candidate.targets.iter().any(|target| target == &event.event_id)
        })
    }
}

/// Validate an event log for identity uniqueness, append order, temporal
/// intervals, typed event semantics and backward-only references.
pub fn validate_events(events: &[EvidenceEvent]) -> Result<(), String> {
    let mut ids: BTreeMap<&str, &EvidenceEvent> = BTreeMap::new();
    let mut sequences = BTreeSet::new();
    let mut previous_sequence = None;
    let mut previous_known_at = None;

    for (index, event) in events.iter().enumerate() {
        if event.schema_version != SCHEMA_VERSION {
            return Err(format!("unsupported schema_version at event index {}", index));
        }
        require_nonempty(&event.event_id, "event_id")?;
        require_nonempty(&event.issuer_id, "issuer_id")?;
        if ids.insert(event.event_id.as_str(), event).is_some() {
            return Err(format!("duplicate event_id: {}", event.event_id));
        }
        if !sequences.insert(event.sequence) {
            return Err(format!("duplicate sequence: {}", event.sequence));
        }
        if previous_sequence.is_some_and(|previous| event.sequence <= previous) {
            return Err("events must be in strictly increasing sequence order".to_string());
        }
        previous_sequence = Some(event.sequence);
        if previous_known_at.is_some_and(|previous| event.known_at_millis < previous) {
            return Err("known_at_millis must be non-decreasing in append order".to_string());
        }
        previous_known_at = Some(event.known_at_millis);

        validate_scopes(&event.authority_scopes, "authority_scopes")?;
        validate_scopes(&event.rights.allowed_purposes, "allowed_purposes")?;
        if event.valid_until_millis.is_some_and(|until| until <= event.valid_from_millis) {
            return Err(format!("invalid valid-time interval: {}", event.event_id));
        }
        if event.rights.retention_until_millis.is_some_and(|until| until <= event.known_at_millis) {
            return Err(format!("retention deadline is not after knowledge time: {}", event.event_id));
        }
        ensure_unique_nonempty(&event.parents, "parents")?;
        ensure_unique_nonempty(&event.targets, "targets")?;

        match event.kind {
            EvidenceEventKind::Assertion | EvidenceEventKind::Correction => {
                require_claim_fields(event)?;
                let observed_at = event
                    .observed_at_millis
                    .ok_or_else(|| format!("claim event missing observed_at_millis: {}", event.event_id))?;
                if observed_at > event.known_at_millis {
                    return Err(format!("claim observed_at_millis is later than known_at_millis: {}", event.event_id));
                }
                if event.consent_id.is_some() && event.rights.consent_grant_event_id.is_none() {
                    return Err(format!("claim consent_id lacks grant reference: {}", event.event_id));
                }
                if event.rights.consent_required
                    && (event.consent_id.is_none() || event.rights.consent_grant_event_id.is_none())
                {
                    return Err(format!("consent-required claim lacks explicit grant: {}", event.event_id));
                }
                if event.kind == EvidenceEventKind::Assertion && !event.targets.is_empty() {
                    return Err(format!("assertion cannot have targets: {}", event.event_id));
                }
                if event.kind == EvidenceEventKind::Correction {
                    if event.targets.is_empty() {
                        return Err(format!("correction must supersede at least one event: {}", event.event_id));
                    }
                    for target in &event.targets {
                        let prior = prior_event(&ids, target, event.sequence)?;
                        if prior.claim_key != event.claim_key
                            || !matches!(prior.kind, EvidenceEventKind::Assertion | EvidenceEventKind::Correction)
                        {
                            return Err(format!("correction target must be an earlier claim with the same claim_key: {}", target));
                        }
                    }
                }
                for parent in &event.parents {
                    let prior = prior_event(&ids, parent, event.sequence)?;
                    if !matches!(prior.kind, EvidenceEventKind::Assertion | EvidenceEventKind::Correction) {
                        return Err(format!("derivation parent is not a claim event: {}", parent));
                    }
                }
                if let Some(grant_id) = event.rights.consent_grant_event_id.as_deref() {
                    let grant = prior_event(&ids, grant_id, event.sequence)?;
                    if grant.kind != EvidenceEventKind::ConsentGranted || grant.consent_id != event.consent_id {
                        return Err(format!("consent grant reference mismatch: {}", event.event_id));
                    }
                }
            }
            EvidenceEventKind::Retraction => {
                require_control_fields(event)?;
                require_claim_key(event)?;
                if event.consent_id.is_some() {
                    return Err(format!("retraction cannot carry consent_id: {}", event.event_id));
                }
                if event.targets.is_empty() {
                    return Err(format!("retraction must target at least one claim: {}", event.event_id));
                }
                for target in &event.targets {
                    let prior = prior_event(&ids, target, event.sequence)?;
                    if prior.claim_key != event.claim_key
                        || !matches!(prior.kind, EvidenceEventKind::Assertion | EvidenceEventKind::Correction)
                    {
                        return Err(format!("retraction target must be an earlier claim with the same claim_key: {}", target));
                    }
                }
            }
            EvidenceEventKind::ConsentGranted => {
                require_control_fields(event)?;
                if event.claim_key.is_some() {
                    return Err(format!("consent grant cannot have a claim_key: {}", event.event_id));
                }
                require_consent_id(event)?;
                if !event.targets.is_empty() {
                    return Err(format!("consent grant cannot have targets: {}", event.event_id));
                }
            }
            EvidenceEventKind::ConsentRevoked => {
                require_control_fields(event)?;
                if event.claim_key.is_some() {
                    return Err(format!("consent revocation cannot have a claim_key: {}", event.event_id));
                }
                require_consent_id(event)?;
                if event.targets.is_empty() {
                    return Err(format!("consent revocation must target a grant: {}", event.event_id));
                }
                for target in &event.targets {
                    let grant = prior_event(&ids, target, event.sequence)?;
                    if grant.kind != EvidenceEventKind::ConsentGranted || grant.consent_id != event.consent_id {
                        return Err(format!("revocation target must be a prior matching consent grant: {}", target));
                    }
                    if grant.issuer_id != event.issuer_id {
                        return Err(format!("consent revocation issuer does not match grant issuer: {}", target));
                    }
                }
            }
            EvidenceEventKind::ErasureTombstone => {
                require_control_fields(event)?;
                if !event.authority_scopes.iter().any(|scope| scope == "memory-erasure") {
                    return Err(format!("erasure tombstone lacks memory-erasure authority scope: {}", event.event_id));
                }
                if event.claim_key.is_some() || event.consent_id.is_some() {
                    return Err(format!("erasure tombstone cannot have claim_key or consent_id: {}", event.event_id));
                }
                if event.targets.is_empty() {
                    return Err(format!("erasure tombstone must target at least one event: {}", event.event_id));
                }
                for target in &event.targets {
                    prior_event(&ids, target, event.sequence)?;
                }
            }
        }
    }
    Ok(())
}

fn require_claim_fields(event: &EvidenceEvent) -> Result<(), String> {
    require_claim_key(event)?;
    let source = event
        .source
        .as_ref()
        .ok_or_else(|| format!("claim event missing source: {}", event.event_id))?;
    require_nonempty(&source.source_id, "source_id")?;
    require_nonempty(&source.source_revision, "source_revision")?;
    validate_digest(&source.source_digest)?;
    let payload = event
        .payload_digest
        .as_deref()
        .ok_or_else(|| format!("claim event missing payload_digest: {}", event.event_id))?;
    validate_digest(payload)?;
    Ok(())
}

fn require_control_fields(event: &EvidenceEvent) -> Result<(), String> {
    if event.source.is_some() || event.payload_digest.is_some() || event.observed_at_millis.is_some() {
        return Err(format!("control event has claim-only fields: {}", event.event_id));
    }
    if event.rights.consent_grant_event_id.is_some() || event.rights.consent_required {
        return Err(format!("control event cannot require an external consent grant: {}", event.event_id));
    }
    if !event.parents.is_empty() {
        return Err(format!("control event cannot have derivation parents: {}", event.event_id));
    }
    Ok(())
}

fn require_claim_key(event: &EvidenceEvent) -> Result<(), String> {
    let claim_key = event
        .claim_key
        .as_deref()
        .ok_or_else(|| format!("event missing claim_key: {}", event.event_id))?;
    require_nonempty(claim_key, "claim_key")
}

fn require_consent_id(event: &EvidenceEvent) -> Result<(), String> {
    let consent_id = event
        .consent_id
        .as_deref()
        .ok_or_else(|| format!("event missing consent_id: {}", event.event_id))?;
    require_nonempty(consent_id, "consent_id")
}

fn prior_event<'a>(
    ids: &'a BTreeMap<&'a str, &'a EvidenceEvent>,
    event_id: &str,
    current_sequence: u64,
) -> Result<&'a EvidenceEvent, String> {
    let prior = ids
        .get(event_id)
        .copied()
        .ok_or_else(|| format!("unknown or forward event reference: {}", event_id))?;
    if prior.sequence >= current_sequence {
        return Err(format!("event reference is not earlier: {}", event_id));
    }
    Ok(prior)
}

fn validate_scopes(scopes: &[String], field: &str) -> Result<(), String> {
    if scopes.is_empty() {
        return Err(format!("{} must not be empty", field));
    }
    let mut seen = BTreeSet::new();
    for scope in scopes {
        require_nonempty(scope, field)?;
        if !seen.insert(scope.as_str()) {
            return Err(format!("duplicate {} value: {}", field, scope));
        }
    }
    Ok(())
}

fn ensure_unique_nonempty(values: &[String], field: &str) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for value in values {
        require_nonempty(value, field)?;
        if !seen.insert(value.as_str()) {
            return Err(format!("duplicate {} reference: {}", field, value));
        }
    }
    Ok(())
}

fn require_nonempty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{} must not be empty", field))
    } else {
        Ok(())
    }
}

fn validate_digest(value: &str) -> Result<(), String> {
    let (algorithm, digest) = value
        .split_once(':')
        .ok_or_else(|| "digest must be algorithm:hex".to_string())?;
    if !matches!(algorithm, "sha256" | "blake3")
        || digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("invalid content digest: {}", value));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rights() -> MemoryRights {
        MemoryRights {
            allowed_purposes: vec!["task-planning".to_string()],
            consent_required: false,
            consent_grant_event_id: None,
            retention_until_millis: None,
        }
    }

    fn claim(
        event_id: &str,
        sequence: u64,
        known_at_millis: u64,
        kind: EvidenceEventKind,
        targets: Vec<&str>,
    ) -> EvidenceEvent {
        EvidenceEvent {
            schema_version: SCHEMA_VERSION.to_string(),
            event_id: event_id.to_string(),
            sequence,
            kind,
            claim_key: Some("project.deadline".to_string()),
            source: Some(SourceCommitment {
                source_id: "user-session-1".to_string(),
                source_revision: "turn-1".to_string(),
                source_digest: format!("blake3:{}", "a".repeat(64)),
            }),
            payload_digest: Some(format!("sha256:{}", "b".repeat(64))),
            issuer_id: "user-1".to_string(),
            authority_scopes: vec!["user-assertion".to_string()],
            rights: rights(),
            observed_at_millis: Some(known_at_millis),
            known_at_millis,
            valid_from_millis: 0,
            valid_until_millis: None,
            parents: Vec::new(),
            targets: targets.into_iter().map(str::to_string).collect(),
            consent_id: None,
        }
    }

    fn query(valid_at_millis: u64, known_at_millis: u64) -> EvidenceQuery {
        EvidenceQuery {
            valid_at_millis,
            known_at_millis,
            purpose: "task-planning".to_string(),
            required_authority: "user-assertion".to_string(),
        }
    }

    fn consent_grant() -> EvidenceEvent {
        EvidenceEvent {
            schema_version: SCHEMA_VERSION.to_string(),
            event_id: "consent-1".to_string(),
            sequence: 1,
            kind: EvidenceEventKind::ConsentGranted,
            claim_key: None,
            source: None,
            payload_digest: None,
            issuer_id: "user-1".to_string(),
            authority_scopes: vec!["user-consent".to_string()],
            rights: rights(),
            observed_at_millis: None,
            known_at_millis: 10,
            valid_from_millis: 0,
            valid_until_millis: None,
            parents: Vec::new(),
            targets: Vec::new(),
            consent_id: Some("consent-alpha".to_string()),
        }
    }

    #[test]
    fn bitemporal_query_separates_valid_time_from_knowledge_time() {
        let event = claim("claim-1", 1, 20, EvidenceEventKind::Assertion, Vec::new());
        let ledger = EvidenceLedger::new(vec![event]).unwrap();
        assert_eq!(
            ledger.evaluate("claim-1", &query(15, 19)).reason,
            Some(IneligibilityReason::NotYetKnown)
        );
        assert!(ledger.evaluate("claim-1", &query(15, 20)).admissible);
    }

    #[test]
    fn correction_supersedes_only_after_its_knowledge_time() {
        let original = claim("claim-1", 1, 10, EvidenceEventKind::Assertion, Vec::new());
        let correction = claim("claim-2", 2, 30, EvidenceEventKind::Correction, vec!["claim-1"]);
        let ledger = EvidenceLedger::new(vec![original, correction]).unwrap();
        assert!(ledger.evaluate("claim-1", &query(20, 20)).admissible);
        assert_eq!(
            ledger.evaluate("claim-1", &query(20, 30)).reason,
            Some(IneligibilityReason::Superseded)
        );
        assert!(ledger.evaluate("claim-2", &query(20, 30)).admissible);
    }

    #[test]
    fn future_validity_correction_does_not_erase_current_past_truth() {
        let original = claim("claim-1", 1, 10, EvidenceEventKind::Assertion, Vec::new());
        let mut correction = claim("claim-2", 2, 30, EvidenceEventKind::Correction, vec!["claim-1"]);
        correction.valid_from_millis = 100;
        let ledger = EvidenceLedger::new(vec![original, correction]).unwrap();
        assert!(ledger.evaluate("claim-1", &query(99, 30)).admissible);
        assert_eq!(
            ledger.evaluate("claim-1", &query(100, 30)).reason,
            Some(IneligibilityReason::Superseded)
        );
    }

    #[test]
    fn correction_target_must_share_claim_key_and_be_prior() {
        let original = claim("claim-1", 1, 10, EvidenceEventKind::Assertion, Vec::new());
        let mut correction = claim("claim-2", 2, 30, EvidenceEventKind::Correction, vec!["claim-1"]);
        correction.claim_key = Some("different.claim".to_string());
        assert!(EvidenceLedger::new(vec![original.clone(), correction]).is_err());

        let forward = claim("claim-2", 2, 30, EvidenceEventKind::Correction, vec!["claim-3"]);
        let later = claim("claim-3", 3, 31, EvidenceEventKind::Assertion, Vec::new());
        assert!(EvidenceLedger::new(vec![original, forward, later]).is_err());
    }

    #[test]
    fn derived_claim_is_blocked_when_a_required_parent_is_retracted() {
        let parent = claim("claim-1", 1, 10, EvidenceEventKind::Assertion, Vec::new());
        let mut derived = claim("claim-2", 2, 20, EvidenceEventKind::Assertion, Vec::new());
        derived.claim_key = Some("derived.deadline".to_string());
        derived.parents = vec!["claim-1".to_string()];
        let mut retraction = claim("retract-1", 3, 40, EvidenceEventKind::Retraction, vec!["claim-1"]);
        retraction.payload_digest = None;
        retraction.source = None;
        retraction.observed_at_millis = None;
        let ledger = EvidenceLedger::new(vec![parent, derived, retraction]).unwrap();
        assert!(ledger.evaluate("claim-2", &query(30, 30)).admissible);
        assert_eq!(
            ledger.evaluate("claim-2", &query(30, 40)).reason,
            Some(IneligibilityReason::ParentNotAdmissible)
        );
    }

    #[test]
    fn consent_revocation_is_checked_at_use_time() {
        let grant = consent_grant();
        let mut event = claim("claim-1", 2, 20, EvidenceEventKind::Assertion, Vec::new());
        event.consent_id = Some("consent-alpha".to_string());
        event.rights.consent_required = true;
        event.rights.consent_grant_event_id = Some("consent-1".to_string());

        let revoke = EvidenceEvent {
            schema_version: SCHEMA_VERSION.to_string(),
            event_id: "consent-revoke-1".to_string(),
            sequence: 3,
            kind: EvidenceEventKind::ConsentRevoked,
            claim_key: None,
            source: None,
            payload_digest: None,
            issuer_id: "user-1".to_string(),
            authority_scopes: vec!["user-consent".to_string()],
            rights: rights(),
            observed_at_millis: None,
            known_at_millis: 40,
            valid_from_millis: 40,
            valid_until_millis: None,
            parents: Vec::new(),
            targets: vec!["consent-1".to_string()],
            consent_id: Some("consent-alpha".to_string()),
        };
        let ledger = EvidenceLedger::new(vec![grant, event, revoke]).unwrap();
        assert!(ledger.evaluate("claim-1", &query(30, 30)).admissible);
        assert!(ledger.evaluate("claim-1", &query(39, 40)).admissible);
        assert_eq!(
            ledger.evaluate("claim-1", &query(40, 40)).reason,
            Some(IneligibilityReason::ConsentRevoked)
        );
    }

    #[test]
    fn unqualified_memory_value_cannot_be_bound() {
        let event = claim("claim-1", 1, 10, EvidenceEventKind::Assertion, Vec::new());
        let ledger = EvidenceLedger::new(vec![event]).unwrap();
        let query = query(20, 20);
        let bound = ledger.bind("deadline: Friday", "claim-1", &query).unwrap();
        assert_eq!(bound.value(), &"deadline: Friday");
        assert_eq!(bound.evidence_event_id(), "claim-1");

        let mut wrong_purpose = query;
        wrong_purpose.purpose = "unrelated-purpose".to_string();
        let rejected = ledger.bind("do not expose", "claim-1", &wrong_purpose).unwrap_err();
        assert_eq!(rejected.reason, Some(IneligibilityReason::PurposeNotPermitted));
    }

    #[test]
    fn consent_revocation_requires_the_granting_issuer() {
        let grant = consent_grant();
        let revoke = EvidenceEvent {
            schema_version: SCHEMA_VERSION.to_string(),
            event_id: "consent-revoke-1".to_string(),
            sequence: 2,
            kind: EvidenceEventKind::ConsentRevoked,
            claim_key: None,
            source: None,
            payload_digest: None,
            issuer_id: "attacker".to_string(),
            authority_scopes: vec!["user-consent".to_string()],
            rights: rights(),
            observed_at_millis: None,
            known_at_millis: 20,
            valid_from_millis: 20,
            valid_until_millis: None,
            parents: Vec::new(),
            targets: vec!["consent-1".to_string()],
            consent_id: Some("consent-alpha".to_string()),
        };
        assert!(EvidenceLedger::new(vec![grant, revoke]).is_err());
    }

    #[test]
    fn purpose_and_authority_are_exact_fail_closed_gates() {
        let event = claim("claim-1", 1, 10, EvidenceEventKind::Assertion, Vec::new());
        let ledger = EvidenceLedger::new(vec![event]).unwrap();
        let mut wrong_purpose = query(20, 20);
        wrong_purpose.purpose = "unrelated-purpose".to_string();
        assert_eq!(
            ledger.evaluate("claim-1", &wrong_purpose).reason,
            Some(IneligibilityReason::PurposeNotPermitted)
        );
        let mut wrong_authority = query(20, 20);
        wrong_authority.required_authority = "administrator-assertion".to_string();
        assert_eq!(
            ledger.evaluate("claim-1", &wrong_authority).reason,
            Some(IneligibilityReason::AuthorityNotPermitted)
        );
    }

    #[test]
    fn erasure_tombstone_survives_ledger_reconstruction() {
        let original = claim("claim-1", 1, 10, EvidenceEventKind::Assertion, Vec::new());
        let mut tombstone = claim("erase-1", 2, 30, EvidenceEventKind::ErasureTombstone, vec!["claim-1"]);
        tombstone.claim_key = None;
        tombstone.source = None;
        tombstone.payload_digest = None;
        tombstone.observed_at_millis = None;
        tombstone.authority_scopes = vec!["memory-erasure".to_string()];
        let first = EvidenceLedger::new(vec![original, tombstone]).unwrap();
        assert_eq!(
            first.evaluate("claim-1", &query(20, 30)).reason,
            Some(IneligibilityReason::Erased)
        );
        let restored = EvidenceLedger::new(first.events().to_vec()).unwrap();
        assert_eq!(
            restored.evaluate("claim-1", &query(20, 30)).reason,
            Some(IneligibilityReason::Erased)
        );
    }

    #[test]
    fn valid_ledger_round_trips_through_validating_deserializer() {
        let ledger = EvidenceLedger::new(vec![
            claim("claim-1", 1, 10, EvidenceEventKind::Assertion, Vec::new())
        ]).unwrap();
        let encoded = serde_json::to_string(&ledger).unwrap();
        let decoded: EvidenceLedger = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, ledger);
    }

    #[test]
    fn deserializer_rejects_cross_event_invalid_references() {
        let invalid = claim(
            "claim-2",
            2,
            20,
            EvidenceEventKind::Correction,
            vec!["missing-parent-claim"],
        );
        let encoded = serde_json::json!({ "events": [invalid] }).to_string();
        let decoded = serde_json::from_str::<EvidenceLedger>(&encoded);
        assert!(decoded.is_err());
    }

    // EvidenceLedger deserialization is intentionally custom: every decoded
    // snapshot passes the same cross-event validator as EvidenceLedger::new.
    // EvidenceBound is intentionally not deserializable; it must be minted by
    // EvidenceLedger::bind at the use boundary.

    #[test]
    fn append_is_monotonic_and_failed_append_is_atomic() {
        let mut ledger = EvidenceLedger::new(vec![
            claim("claim-1", 1, 10, EvidenceEventKind::Assertion, Vec::new())
        ]).unwrap();
        let before = ledger.events().to_vec();
        let result = ledger.append(claim("claim-2", 1, 20, EvidenceEventKind::Assertion, Vec::new()));
        assert!(result.is_err());
        assert_eq!(ledger.events(), before.as_slice());
        ledger.append(claim("claim-2", 2, 20, EvidenceEventKind::Assertion, Vec::new())).unwrap();
        assert_eq!(ledger.events().len(), 2);
    }

    #[test]
    fn knowledge_time_must_not_move_backward_in_append_order() {
        let first = claim("claim-1", 1, 20, EvidenceEventKind::Assertion, Vec::new());
        let second = claim("claim-2", 2, 19, EvidenceEventKind::Assertion, Vec::new());
        assert!(EvidenceLedger::new(vec![first, second]).is_err());
    }

    #[test]
    fn erasing_a_consent_grant_invalidates_dependents() {
        let grant = consent_grant();
        let mut claim_event = claim("claim-1", 2, 20, EvidenceEventKind::Assertion, Vec::new());
        claim_event.consent_id = Some("consent-alpha".to_string());
        claim_event.rights.consent_required = true;
        claim_event.rights.consent_grant_event_id = Some("consent-1".to_string());

        let mut tombstone = claim("erase-consent", 3, 30, EvidenceEventKind::ErasureTombstone, vec!["consent-1"]);
        tombstone.claim_key = None;
        tombstone.source = None;
        tombstone.payload_digest = None;
        tombstone.observed_at_millis = None;
        tombstone.consent_id = None;
        tombstone.authority_scopes = vec!["memory-erasure".to_string()];

        let ledger = EvidenceLedger::new(vec![grant, claim_event, tombstone]).unwrap();
        assert_eq!(
            ledger.evaluate("claim-1", &query(35, 30)).reason,
            Some(IneligibilityReason::ConsentOutOfScope)
        );
    }

    #[test]
    fn rejects_bad_digest_and_empty_scope() {
        let mut event = claim("claim-1", 1, 10, EvidenceEventKind::Assertion, Vec::new());
        event.payload_digest = Some("sha256:xyz".to_string());
        assert!(EvidenceLedger::new(vec![event.clone()]).is_err());
        event.payload_digest = Some(format!("sha256:{}", "c".repeat(64)));
        event.rights.allowed_purposes.clear();
        assert!(EvidenceLedger::new(vec![event]).is_err());
    }

    #[test]
    fn rejects_source_observation_after_knowledge_time() {
        let mut event = claim("claim-1", 1, 10, EvidenceEventKind::Assertion, Vec::new());
        event.observed_at_millis = Some(11);
        assert!(EvidenceLedger::new(vec![event]).is_err());
    }

    #[test]
    fn retraction_cannot_smuggle_a_consent_identifier() {
        let prior = claim("claim-1", 1, 10, EvidenceEventKind::Assertion, Vec::new());
        let mut retraction = claim("retract-1", 2, 20, EvidenceEventKind::Retraction, vec!["claim-1"]);
        retraction.source = None;
        retraction.payload_digest = None;
        retraction.observed_at_millis = None;
        retraction.consent_id = Some("unexpected".to_string());
        assert!(EvidenceLedger::new(vec![prior, retraction]).is_err());
    }

    #[test]
    fn rejects_invalid_validity_interval() {
        let mut event = claim("claim-1", 1, 10, EvidenceEventKind::Assertion, Vec::new());
        event.valid_until_millis = Some(0);
        assert!(EvidenceLedger::new(vec![event]).is_err());
    }

    #[test]
    fn unknown_and_control_events_are_not_eligible_claims() {
        let mut invalid_retraction = claim("retract-1", 1, 20, EvidenceEventKind::Retraction, vec!["missing"]);
        invalid_retraction.payload_digest = None;
        invalid_retraction.source = None;
        invalid_retraction.observed_at_millis = None;
        assert!(EvidenceLedger::new(vec![invalid_retraction]).is_err());

        let ledger = EvidenceLedger::empty();
        assert_eq!(
            ledger.evaluate("missing", &query(30, 30)).reason,
            Some(IneligibilityReason::UnknownEvent)
        );
    }
}
