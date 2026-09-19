// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Content-addressed temporal key lifecycle history.
//!
//! The ordinary trust snapshot answers whether a key may grant authority now.
//! This module answers a different question: what was the key's authority state
//! at a particular historical instant, given everything now known about its
//! lifecycle? In particular, a later compromise discovery may invalidate an
//! interval in the past without rewriting or deleting the original trust state.

use serde::Serialize;

use crate::{FramedDigest, Sha256Digest, SignatureAlgorithm};

pub const KEY_LIFECYCLE_HISTORY_SCHEMA: &str = "symthaea.key-lifecycle-history.v1";
const KEY_LIFECYCLE_EVENT_DOMAIN: &str = "symthaea.key-lifecycle-event.identity.v1";
const KEY_LIFECYCLE_HISTORY_DOMAIN: &str = "symthaea.key-lifecycle-history.identity.v1";
pub const MAX_KEY_LIFECYCLE_EVENTS: usize = 256;
pub const MAX_TEMPORAL_KEY_ID_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum KeyLifecycleEventKind {
    Activate,
    Retire,
    RevokeProspectively,
    /// A later investigation established that the key must be treated as
    /// compromised from this earlier instant onward.
    CompromisedSince { compromised_since_unix_s: u64 },
    ReplaceWith {
        successor_algorithm: SignatureAlgorithm,
        successor_key_id: String,
        successor_verification_key_sha256: Sha256Digest,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyLifecycleEventDraft {
    pub recorded_at_unix_s: u64,
    /// Effective time of Activate/Retire/Revoke/Replace. For CompromisedSince,
    /// the retroactive boundary is carried by the event kind and this field is
    /// the time the compromise finding enters the lifecycle lineage.
    pub effective_at_unix_s: u64,
    pub kind: KeyLifecycleEventKind,
    pub rationale_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KeyLifecycleEvent {
    sequence: u64,
    recorded_at_unix_s: u64,
    effective_at_unix_s: u64,
    kind: KeyLifecycleEventKind,
    rationale_sha256: Sha256Digest,
    previous_event_sha256: Option<Sha256Digest>,
    event_sha256: Sha256Digest,
}

impl KeyLifecycleEvent {
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn recorded_at_unix_s(&self) -> u64 {
        self.recorded_at_unix_s
    }

    pub fn effective_at_unix_s(&self) -> u64 {
        self.effective_at_unix_s
    }

    pub fn kind(&self) -> &KeyLifecycleEventKind {
        &self.kind
    }

    pub fn rationale_sha256(&self) -> &Sha256Digest {
        &self.rationale_sha256
    }

    pub fn previous_event_sha256(&self) -> Option<&Sha256Digest> {
        self.previous_event_sha256.as_ref()
    }

    pub fn event_sha256(&self) -> &Sha256Digest {
        &self.event_sha256
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalKeyStatus {
    NotYetActive,
    Active,
    Retired,
    Revoked,
    Compromised,
    Replaced,
}

impl TemporalKeyStatus {
    pub const fn permits_historical_authority(self) -> bool {
        matches!(self, Self::Active)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyLifecycleHistoryIssue {
    InvalidAlgorithm,
    InvalidKeyId,
    KeyIdTooLong,
    EmptyEvents,
    TooManyEvents { actual: usize, maximum: usize },
    FirstEventMustActivate,
    DuplicateActivation,
    RecordedTimeRegressed,
    InvalidEffectiveTime,
    DuplicateTerminalTransition,
    InvalidSuccessorAlgorithm,
    InvalidSuccessorKeyId,
    SuccessorKeyIdTooLong,
    SelfReplacement,
    CompromiseBoundaryAfterRecording,
    CompromiseBoundaryMovedForward,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KeyLifecycleHistory {
    schema_version: String,
    algorithm: SignatureAlgorithm,
    key_id: String,
    verification_key_sha256: Sha256Digest,
    events: Vec<KeyLifecycleEvent>,
    history_sha256: Sha256Digest,
}

impl KeyLifecycleHistory {
    pub fn new(
        algorithm: SignatureAlgorithm,
        key_id: impl Into<String>,
        verification_key_sha256: Sha256Digest,
        drafts: Vec<KeyLifecycleEventDraft>,
    ) -> Result<Self, Vec<KeyLifecycleHistoryIssue>> {
        let key_id = key_id.into();
        let mut issues = validate_identity(&algorithm, &key_id);
        if drafts.is_empty() {
            issues.push(KeyLifecycleHistoryIssue::EmptyEvents);
        }
        if drafts.len() > MAX_KEY_LIFECYCLE_EVENTS {
            issues.push(KeyLifecycleHistoryIssue::TooManyEvents {
                actual: drafts.len(),
                maximum: MAX_KEY_LIFECYCLE_EVENTS,
            });
        }
        if drafts
            .first()
            .is_some_and(|draft| !matches!(&draft.kind, KeyLifecycleEventKind::Activate))
        {
            issues.push(KeyLifecycleHistoryIssue::FirstEventMustActivate);
        }

        let mut previous_recorded_at = None;
        let mut activation_count = 0usize;
        let mut terminal_seen = false;
        let mut earliest_compromise: Option<u64> = None;
        for draft in &drafts {
            if previous_recorded_at.is_some_and(|previous| draft.recorded_at_unix_s < previous) {
                issues.push(KeyLifecycleHistoryIssue::RecordedTimeRegressed);
            }
            previous_recorded_at = Some(draft.recorded_at_unix_s);
            if draft.effective_at_unix_s > draft.recorded_at_unix_s
                && !matches!(&draft.kind, KeyLifecycleEventKind::Activate)
            {
                issues.push(KeyLifecycleHistoryIssue::InvalidEffectiveTime);
            }

            match &draft.kind {
                KeyLifecycleEventKind::Activate => {
                    activation_count += 1;
                    if activation_count > 1 {
                        issues.push(KeyLifecycleHistoryIssue::DuplicateActivation);
                    }
                }
                KeyLifecycleEventKind::Retire
                | KeyLifecycleEventKind::RevokeProspectively
                | KeyLifecycleEventKind::ReplaceWith { .. } => {
                    if terminal_seen {
                        issues.push(KeyLifecycleHistoryIssue::DuplicateTerminalTransition);
                    }
                    terminal_seen = true;
                }
                KeyLifecycleEventKind::CompromisedSince {
                    compromised_since_unix_s,
                } => {
                    if *compromised_since_unix_s > draft.recorded_at_unix_s {
                        issues.push(KeyLifecycleHistoryIssue::CompromiseBoundaryAfterRecording);
                    }
                    if earliest_compromise
                        .is_some_and(|previous| *compromised_since_unix_s > previous)
                    {
                        issues.push(KeyLifecycleHistoryIssue::CompromiseBoundaryMovedForward);
                    }
                    earliest_compromise = Some(*compromised_since_unix_s);
                }
            }

            if let KeyLifecycleEventKind::ReplaceWith {
                successor_algorithm,
                successor_key_id,
                ..
            } = &draft.kind
            {
                if !successor_algorithm.is_canonical() {
                    issues.push(KeyLifecycleHistoryIssue::InvalidSuccessorAlgorithm);
                }
                if successor_key_id.trim().is_empty() || successor_key_id != successor_key_id.trim() {
                    issues.push(KeyLifecycleHistoryIssue::InvalidSuccessorKeyId);
                }
                if successor_key_id.len() > MAX_TEMPORAL_KEY_ID_BYTES {
                    issues.push(KeyLifecycleHistoryIssue::SuccessorKeyIdTooLong);
                }
                if successor_algorithm == &algorithm && successor_key_id == &key_id {
                    issues.push(KeyLifecycleHistoryIssue::SelfReplacement);
                }
            }
        }

        if !issues.is_empty() {
            return Err(issues);
        }

        let mut events = Vec::with_capacity(drafts.len());
        let mut previous_event_sha256 = None;
        for (index, draft) in drafts.into_iter().enumerate() {
            let sequence = index as u64 + 1;
            let event_sha256 = event_digest(
                &algorithm,
                &key_id,
                &verification_key_sha256,
                sequence,
                draft.recorded_at_unix_s,
                draft.effective_at_unix_s,
                &draft.kind,
                &draft.rationale_sha256,
                previous_event_sha256.as_ref(),
            );
            events.push(KeyLifecycleEvent {
                sequence,
                recorded_at_unix_s: draft.recorded_at_unix_s,
                effective_at_unix_s: draft.effective_at_unix_s,
                kind: draft.kind,
                rationale_sha256: draft.rationale_sha256,
                previous_event_sha256: previous_event_sha256.clone(),
                event_sha256: event_sha256.clone(),
            });
            previous_event_sha256 = Some(event_sha256);
        }

        let history_sha256 = history_digest(
            &algorithm,
            &key_id,
            &verification_key_sha256,
            &events,
        );
        Ok(Self {
            schema_version: KEY_LIFECYCLE_HISTORY_SCHEMA.into(),
            algorithm,
            key_id,
            verification_key_sha256,
            events,
            history_sha256,
        })
    }

    pub fn algorithm(&self) -> &SignatureAlgorithm {
        &self.algorithm
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    pub fn verification_key_sha256(&self) -> &Sha256Digest {
        &self.verification_key_sha256
    }

    pub fn events(&self) -> &[KeyLifecycleEvent] {
        &self.events
    }

    pub fn history_sha256(&self) -> &Sha256Digest {
        &self.history_sha256
    }

    /// Evaluate historical authority using the entire known lifecycle. A later
    /// CompromisedSince event therefore reaches backward to its explicit
    /// compromise boundary, while ordinary retirement/revocation remains
    /// prospective from its effective time.
    pub fn status_at(&self, unix_s: u64) -> TemporalKeyStatus {
        let mut status = TemporalKeyStatus::NotYetActive;
        let mut earliest_compromise: Option<u64> = None;

        for event in &self.events {
            if let KeyLifecycleEventKind::CompromisedSince {
                compromised_since_unix_s,
            } = &event.kind
            {
                earliest_compromise = Some(
                    earliest_compromise.map_or(*compromised_since_unix_s, |previous| {
                        previous.min(*compromised_since_unix_s)
                    }),
                );
            }
        }
        if earliest_compromise.is_some_and(|boundary| unix_s >= boundary) {
            return TemporalKeyStatus::Compromised;
        }

        for event in &self.events {
            if event.effective_at_unix_s > unix_s {
                continue;
            }
            status = match &event.kind {
                KeyLifecycleEventKind::Activate => TemporalKeyStatus::Active,
                KeyLifecycleEventKind::Retire => TemporalKeyStatus::Retired,
                KeyLifecycleEventKind::RevokeProspectively => TemporalKeyStatus::Revoked,
                KeyLifecycleEventKind::ReplaceWith { .. } => TemporalKeyStatus::Replaced,
                KeyLifecycleEventKind::CompromisedSince { .. } => status,
            };
        }
        status
    }

    pub fn historically_authoritative_at(&self, unix_s: u64) -> bool {
        self.status_at(unix_s).permits_historical_authority()
    }
}

fn validate_identity(
    algorithm: &SignatureAlgorithm,
    key_id: &str,
) -> Vec<KeyLifecycleHistoryIssue> {
    let mut issues = Vec::new();
    if !algorithm.is_canonical() {
        issues.push(KeyLifecycleHistoryIssue::InvalidAlgorithm);
    }
    if key_id.trim().is_empty() || key_id != key_id.trim() {
        issues.push(KeyLifecycleHistoryIssue::InvalidKeyId);
    }
    if key_id.len() > MAX_TEMPORAL_KEY_ID_BYTES {
        issues.push(KeyLifecycleHistoryIssue::KeyIdTooLong);
    }
    issues
}

#[allow(clippy::too_many_arguments)]
fn event_digest(
    algorithm: &SignatureAlgorithm,
    key_id: &str,
    verification_key_sha256: &Sha256Digest,
    sequence: u64,
    recorded_at_unix_s: u64,
    effective_at_unix_s: u64,
    kind: &KeyLifecycleEventKind,
    rationale_sha256: &Sha256Digest,
    previous_event_sha256: Option<&Sha256Digest>,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(KEY_LIFECYCLE_EVENT_DOMAIN);
    digest.text(KEY_LIFECYCLE_HISTORY_SCHEMA);
    digest_algorithm(&mut digest, algorithm);
    digest.text(key_id);
    digest.text(verification_key_sha256.as_str());
    digest.text(&sequence.to_string());
    digest.text(&recorded_at_unix_s.to_string());
    digest.text(&effective_at_unix_s.to_string());
    digest_event_kind(&mut digest, kind);
    digest.text(rationale_sha256.as_str());
    digest.optional_sha(previous_event_sha256);
    digest.digest()
}

fn history_digest(
    algorithm: &SignatureAlgorithm,
    key_id: &str,
    verification_key_sha256: &Sha256Digest,
    events: &[KeyLifecycleEvent],
) -> Sha256Digest {
    let mut digest = FramedDigest::new(KEY_LIFECYCLE_HISTORY_DOMAIN);
    digest.text(KEY_LIFECYCLE_HISTORY_SCHEMA);
    digest_algorithm(&mut digest, algorithm);
    digest.text(key_id);
    digest.text(verification_key_sha256.as_str());
    for event in events {
        digest.text("event");
        digest.text(event.event_sha256.as_str());
    }
    digest.digest()
}

fn digest_algorithm(digest: &mut FramedDigest, algorithm: &SignatureAlgorithm) {
    match algorithm {
        SignatureAlgorithm::Ed25519 => digest.text("builtin:ed25519"),
        SignatureAlgorithm::MlDsa65 => digest.text("builtin:ml-dsa-65"),
        SignatureAlgorithm::MlDsa87 => digest.text("builtin:ml-dsa-87"),
        SignatureAlgorithm::Other(name) => {
            digest.text("other");
            digest.text(name);
        }
    }
}

fn digest_event_kind(digest: &mut FramedDigest, kind: &KeyLifecycleEventKind) {
    match kind {
        KeyLifecycleEventKind::Activate => digest.text("activate"),
        KeyLifecycleEventKind::Retire => digest.text("retire"),
        KeyLifecycleEventKind::RevokeProspectively => digest.text("revoke-prospectively"),
        KeyLifecycleEventKind::CompromisedSince {
            compromised_since_unix_s,
        } => {
            digest.text("compromised-since");
            digest.text(&compromised_since_unix_s.to_string());
        }
        KeyLifecycleEventKind::ReplaceWith {
            successor_algorithm,
            successor_key_id,
            successor_verification_key_sha256,
        } => {
            digest.text("replace-with");
            digest_algorithm(digest, successor_algorithm);
            digest.text(successor_key_id);
            digest.text(successor_verification_key_sha256.as_str());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    fn draft(
        recorded_at_unix_s: u64,
        effective_at_unix_s: u64,
        kind: KeyLifecycleEventKind,
    ) -> KeyLifecycleEventDraft {
        KeyLifecycleEventDraft {
            recorded_at_unix_s,
            effective_at_unix_s,
            kind,
            rationale_sha256: sha(&format!("reason-{recorded_at_unix_s}")),
        }
    }

    fn history(events: Vec<KeyLifecycleEventDraft>) -> KeyLifecycleHistory {
        KeyLifecycleHistory::new(
            SignatureAlgorithm::Ed25519,
            "reviewer",
            sha("verification-key"),
            events,
        )
        .unwrap()
    }

    #[test]
    fn retirement_does_not_rewrite_prior_authority() {
        let history = history(vec![
            draft(100, 100, KeyLifecycleEventKind::Activate),
            draft(600, 600, KeyLifecycleEventKind::Retire),
        ]);
        assert_eq!(history.status_at(500), TemporalKeyStatus::Active);
        assert_eq!(history.status_at(700), TemporalKeyStatus::Retired);
        assert!(history.historically_authoritative_at(500));
    }

    #[test]
    fn prospective_revocation_does_not_rewrite_prior_authority() {
        let history = history(vec![
            draft(100, 100, KeyLifecycleEventKind::Activate),
            draft(600, 600, KeyLifecycleEventKind::RevokeProspectively),
        ]);
        assert_eq!(history.status_at(500), TemporalKeyStatus::Active);
        assert_eq!(history.status_at(700), TemporalKeyStatus::Revoked);
    }

    #[test]
    fn later_compromise_discovery_reaches_back_to_explicit_boundary() {
        let history = history(vec![
            draft(100, 100, KeyLifecycleEventKind::Activate),
            draft(
                800,
                800,
                KeyLifecycleEventKind::CompromisedSince {
                    compromised_since_unix_s: 450,
                },
            ),
        ]);
        assert_eq!(history.status_at(400), TemporalKeyStatus::Active);
        assert_eq!(history.status_at(500), TemporalKeyStatus::Compromised);
        assert!(!history.historically_authoritative_at(500));
    }

    #[test]
    fn later_compromise_updates_can_only_move_boundary_earlier() {
        let invalid = KeyLifecycleHistory::new(
            SignatureAlgorithm::Ed25519,
            "reviewer",
            sha("verification-key"),
            vec![
                draft(100, 100, KeyLifecycleEventKind::Activate),
                draft(
                    700,
                    700,
                    KeyLifecycleEventKind::CompromisedSince {
                        compromised_since_unix_s: 300,
                    },
                ),
                draft(
                    800,
                    800,
                    KeyLifecycleEventKind::CompromisedSince {
                        compromised_since_unix_s: 400,
                    },
                ),
            ],
        );
        assert!(invalid.is_err());
    }

    #[test]
    fn event_chain_and_history_identity_are_deterministic() {
        let events = vec![
            draft(100, 100, KeyLifecycleEventKind::Activate),
            draft(600, 600, KeyLifecycleEventKind::Retire),
        ];
        let left = history(events.clone());
        let right = history(events);
        assert_eq!(left.history_sha256(), right.history_sha256());
        assert_eq!(
            left.events()[1].previous_event_sha256(),
            Some(left.events()[0].event_sha256())
        );
    }
}
