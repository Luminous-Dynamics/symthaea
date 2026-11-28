// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Causal verifier for privacy-minimized Symthaea interaction benchmark traces.
//!
//! JSON Schema validates record shape. This module validates cross-record
//! causality: ordering, clock-domain monotonicity, one-decision turn control,
//! immutable semantic digests, live authorization, action authority, and
//! observed output lifecycle. It intentionally does not infer audio turns,
//! authenticate source components, attest hardware, or prove that a compositor
//! observer is independent.

use serde_json::Value;
use std::collections::{HashMap, HashSet};

pub const INTERACTION_TRACE_SCHEMA_V1: &str = "symthaea.interaction-trace.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InteractionTraceError {
    InvalidJson,
    InvalidRoot(&'static str),
    MissingField { index: Option<usize>, field: &'static str },
    InvalidField { index: Option<usize>, field: &'static str },
    UnknownEventType { index: usize },
    DuplicateSequence { index: usize },
    SequenceRegression { index: usize },
    ClockRegression { index: usize },
    InputArrivalRegression { index: usize },
    LateTurnDecision { index: usize },
    DuplicateTurnCandidate { index: usize },
    ConflictingCueDigest { index: usize },
    MissingAuthorization { index: usize },
    AuthorizationMismatch { index: usize },
    PrivacyMismatch { index: usize },
    PolicyVersionRegression { index: usize },
    AuthorizationExpired { index: usize },
    AuthorizationRevoked { index: usize },
    SourceClockMismatch { index: usize },
    ActionAuthorizationMissing { index: usize },
    GroundingSupportMissing { index: usize },
    OutputStateViolation { index: usize },
    FrameSequenceRegression { index: usize },
    OutputNotStopped { output_id: String },
}

impl std::fmt::Display for InteractionTraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for InteractionTraceError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InteractionTraceSummary {
    pub event_count: usize,
    pub input_chunks: usize,
    pub semantic_cues: usize,
    pub turn_decisions: usize,
    pub authorizations: usize,
    pub submitted_outputs: usize,
    pub presented_frames: usize,
    pub stopped_outputs: usize,
    pub grounding_probes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct AuthorizationKey {
    semantic_event_id: String,
    canonical_cue_digest: String,
    grant_id: String,
    form: String,
    channel: String,
    destination: String,
    privacy_class: String,
}

#[derive(Debug, Clone)]
struct AuthorizationRecord {
    sequence: u64,
    key: AuthorizationKey,
    clock_domain_id: String,
    expires_at_monotonic_ns: u64,
    action_scope: HashSet<String>,
}

#[derive(Debug, Clone)]
struct OutputRecord {
    key: AuthorizationKey,
    authorization_sequence: u64,
    last_frame_index: Option<u64>,
    stopped: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SemanticEventRecord {
    digest: String,
    clock_domain_id: String,
    sequence: u64,
    privacy_class: String,
}

pub struct InteractionTraceVerifier;

impl InteractionTraceVerifier {
    /// Parse and verify a serialized interaction trace.
    pub fn verify_json(input: &str) -> Result<InteractionTraceSummary, InteractionTraceError> {
        let value: Value =
            serde_json::from_str(input).map_err(|_| InteractionTraceError::InvalidJson)?;
        Self::verify_value(&value)
    }

    /// Verify an already-parsed trace value. Cross-event rules are enforced in
    /// event order and use source-clock samples for permit expiry comparisons.
    pub fn verify_value(
        trace: &Value,
    ) -> Result<InteractionTraceSummary, InteractionTraceError> {
        if !trace.is_object() {
            return Err(InteractionTraceError::InvalidRoot("trace"));
        }

        let schema = root_string(trace, "schema")?;
        if schema != INTERACTION_TRACE_SCHEMA_V1 {
            return Err(InteractionTraceError::InvalidRoot("schema"));
        }
        validate_identifier(root_string(trace, "trace_id")?, None, "trace_id")?;
        validate_identifier(root_string(trace, "session_id")?, None, "session_id")?;
        validate_nonempty(root_string(trace, "protocol")?, None, "protocol")?;
        validate_hex(root_string(trace, "source_commit")?, 40, None, "source_commit")?;
        validate_hex(root_string(trace, "dataset_digest")?, 64, None, "dataset_digest")?;
        if !trace.get("run_manifest").is_some_and(Value::is_object) {
            return Err(InteractionTraceError::InvalidRoot("run_manifest"));
        }
        let events = trace
            .get("events")
            .and_then(Value::as_array)
            .filter(|events| !events.is_empty())
            .ok_or(InteractionTraceError::InvalidRoot("events"))?;

        let mut summary = InteractionTraceSummary {
            event_count: events.len(),
            ..Default::default()
        };
        let mut last_sequence = None;
        let mut last_policy_version = trace
            .get("run_manifest")
            .and_then(|manifest| manifest.get("policy_version"))
            .and_then(Value::as_u64)
            .ok_or(InteractionTraceError::InvalidRoot("run_manifest.policy_version"))?;
        let mut last_event_time_by_clock: HashMap<String, u64> = HashMap::new();
        let mut last_source_time_by_clock: HashMap<String, u64> = HashMap::new();
        let mut seen_sequences = HashSet::new();
        let mut semantic_events: HashMap<String, SemanticEventRecord> = HashMap::new();
        let mut turn_candidates = HashSet::new();
        let mut authorization_by_sequence: HashMap<u64, AuthorizationRecord> = HashMap::new();
        let mut authorization_by_key: HashMap<AuthorizationKey, Vec<u64>> = HashMap::new();
        let mut revoked_grants: HashMap<String, u64> = HashMap::new();
        let mut cancelled_events: HashMap<String, u64> = HashMap::new();
        let mut global_invalidation_sequence = None;
        let mut paused = false;
        let mut outputs: HashMap<String, OutputRecord> = HashMap::new();

        for (index, event) in events.iter().enumerate() {
            if !event.is_object() {
                return Err(InteractionTraceError::InvalidField {
                    index: Some(index),
                    field: "event",
                });
            }
            let sequence = event_u64(event, "sequence", index)?;
            if !seen_sequences.insert(sequence) {
                return Err(InteractionTraceError::DuplicateSequence { index });
            }
            if last_sequence.is_some_and(|last| sequence < last) {
                return Err(InteractionTraceError::SequenceRegression { index });
            }
            last_sequence = Some(sequence);

            let clock_domain_id = event_string(event, "clock_domain_id", index)?.to_owned();
            validate_identifier(&clock_domain_id, Some(index), "clock_domain_id")?;
            let at = event_u64(event, "at_monotonic_ns", index)?;
            if last_event_time_by_clock
                .get(&clock_domain_id)
                .is_some_and(|last| at < *last)
            {
                return Err(InteractionTraceError::ClockRegression { index });
            }
            last_event_time_by_clock.insert(clock_domain_id.clone(), at);

            let event_type = event_string(event, "event_type", index)?;
            match event_type {
                "input_chunk" => {
                    validate_hex(event_string(event, "payload_digest", index)?, 64, Some(index), "payload_digest")?;
                    let arrival = event_u64(event, "arrival_at_monotonic_ns", index)?;
                    if arrival < at {
                        return Err(InteractionTraceError::InputArrivalRegression { index });
                    }
                    let modality = event_string(event, "modality", index)?;
                    if !matches!(modality, "audio" | "video" | "text") {
                        return Err(InteractionTraceError::InvalidField { index: Some(index), field: "modality" });
                    }
                    summary.input_chunks += 1;
                }
                "semantic_cue" => {
                    let event_id = event_string(event, "event_id", index)?.to_owned();
                    validate_identifier(&event_id, Some(index), "event_id")?;
                    let digest = event_string(event, "canonical_cue_digest", index)?.to_owned();
                    validate_hex(&digest, 64, Some(index), "canonical_cue_digest")?;
                    let privacy_class = event_string(event, "privacy_class", index)?.to_owned();
                    if !matches!(privacy_class.as_str(), "public" | "local_sensitive" | "restricted") {
                        return Err(InteractionTraceError::InvalidField { index: Some(index), field: "privacy_class" });
                    }
                    if let Some(previous) = semantic_events.get(&event_id) {
                        if previous.digest != digest {
                            return Err(InteractionTraceError::ConflictingCueDigest { index });
                        }
                        if previous.privacy_class != privacy_class {
                            return Err(InteractionTraceError::PrivacyMismatch { index });
                        }
                        if previous.clock_domain_id != clock_domain_id {
                            return Err(InteractionTraceError::AuthorizationMismatch { index });
                        }
                    } else {
                        semantic_events.insert(
                            event_id,
                            SemanticEventRecord {
                                digest,
                                clock_domain_id: clock_domain_id.clone(),
                                sequence,
                                privacy_class,
                            },
                        );
                    }
                    summary.semantic_cues += 1;
                }
                "turn_decision" => {
                    let candidate_id = event_string(event, "candidate_id", index)?.to_owned();
                    validate_identifier(&candidate_id, Some(index), "candidate_id")?;
                    if !turn_candidates.insert(candidate_id) {
                        return Err(InteractionTraceError::DuplicateTurnCandidate { index });
                    }
                    let confidence = event
                        .get("confidence")
                        .and_then(Value::as_f64)
                        .ok_or(InteractionTraceError::InvalidField {
                            index: Some(index),
                            field: "confidence",
                        })?;
                    if !(0.0..=1.0).contains(&confidence) {
                        return Err(InteractionTraceError::InvalidField {
                            index: Some(index),
                            field: "confidence",
                        });
                    }
                    let emitted = event_u64(event, "emitted_at_monotonic_ns", index)?;
                    let deadline = event_u64(event, "deadline_at_monotonic_ns", index)?;
                    if emitted < at || emitted > deadline {
                        return Err(InteractionTraceError::LateTurnDecision { index });
                    }
                    let decision = event_string(event, "decision", index)?;
                    if !matches!(decision, "wait" | "respond" | "continue" | "yield" | "abstain") {
                        return Err(InteractionTraceError::InvalidField { index: Some(index), field: "decision" });
                    }
                    summary.turn_decisions += 1;
                }
                "authorization" => {
                    let decision = event_string(event, "decision", index)?;
                    let grant_id = event_string(event, "grant_id", index)?.to_owned();
                    validate_identifier(&grant_id, Some(index), "grant_id")?;
                    let semantic_event_id = event_string(event, "semantic_event_id", index)?.to_owned();
                    let canonical_cue_digest = event_string(event, "canonical_cue_digest", index)?.to_owned();
                    let form = event_string(event, "form", index)?.to_owned();
                    let channel = event_string(event, "channel", index)?.to_owned();
                    let destination = event_string(event, "destination", index)?.to_owned();
                    let privacy_class = event_string(event, "privacy_class", index)?.to_owned();
                    let policy_version = event_u64(event, "policy_version", index)?;
                    if policy_version != last_policy_version {
                        return Err(InteractionTraceError::PolicyVersionRegression { index });
                    }
                    let action_scope_values = event
                        .get("action_scope")
                        .and_then(Value::as_array)
                        .ok_or(InteractionTraceError::InvalidField {
                            index: Some(index),
                            field: "action_scope",
                        })?;
                    if action_scope_values.len() > 32 {
                        return Err(InteractionTraceError::InvalidField {
                            index: Some(index),
                            field: "action_scope",
                        });
                    }
                    let mut action_scope = HashSet::new();
                    for value in action_scope_values {
                        let action = value.as_str().ok_or(InteractionTraceError::InvalidField {
                            index: Some(index),
                            field: "action_scope",
                        })?;
                        validate_identifier(action, Some(index), "action_scope")?;
                        if !action_scope.insert(action.to_owned()) {
                            return Err(InteractionTraceError::InvalidField {
                                index: Some(index),
                                field: "action_scope",
                            });
                        }
                    }
                    let semantic = semantic_events.get(&semantic_event_id)
                        .ok_or(InteractionTraceError::MissingAuthorization { index })?;
                    if semantic.digest != canonical_cue_digest
                        || semantic.clock_domain_id != clock_domain_id
                        || semantic.sequence >= sequence
                    {
                        return Err(InteractionTraceError::AuthorizationMismatch { index });
                    }
                    if semantic.privacy_class != privacy_class {
                        return Err(InteractionTraceError::PrivacyMismatch { index });
                    }

                    if decision == "authorized" {
                        let permit_digest = event_string(event, "permit_digest", index)?;
                        validate_hex(permit_digest, 64, Some(index), "permit_digest")?;
                        let expires = event_u64(event, "expires_at_monotonic_ns", index)?;
                        if expires <= at {
                            return Err(InteractionTraceError::AuthorizationExpired { index });
                        }
                        if paused {
                            return Err(InteractionTraceError::AuthorizationRevoked { index });
                        }
                        let key = AuthorizationKey {
                            semantic_event_id,
                            canonical_cue_digest,
                            grant_id,
                            form,
                            channel,
                            destination,
                            privacy_class,
                        };
                        let record = AuthorizationRecord {
                            sequence,
                            key: key.clone(),
                            clock_domain_id: clock_domain_id.clone(),
                            expires_at_monotonic_ns: expires,
                            action_scope,
                        };
                        authorization_by_sequence.insert(sequence, record);
                        authorization_by_key.entry(key).or_default().push(sequence);
                        summary.authorizations += 1;
                    } else if matches!(decision, "revoked" | "expired") {
                        revoked_grants.insert(grant_id, sequence);
                    } else if decision == "denied" {
                        // A denied request does not revoke an independently
                        // issued, still-live permit for the same grant.
                    } else {
                        return Err(InteractionTraceError::InvalidField {
                            index: Some(index),
                            field: "decision",
                        });
                    }
                }
                "action" => {
                    let authorization_sequence = event_u64(event, "authorization_sequence", index)?;
                    let auth = authorization_by_sequence
                        .get(&authorization_sequence)
                        .filter(|auth| auth.sequence < sequence)
                        .ok_or(InteractionTraceError::ActionAuthorizationMissing { index })?;
                    let side_effect = event
                        .get("side_effect")
                        .and_then(Value::as_bool)
                        .ok_or(InteractionTraceError::InvalidField {
                            index: Some(index),
                            field: "side_effect",
                        })?;
                    let action_kind = event_string(event, "action_kind", index)?;
                    if side_effect
                        && (auth.key.channel != "actuator"
                            || !auth.action_scope.contains(action_kind))
                    {
                        return Err(InteractionTraceError::ActionAuthorizationMissing { index });
                    }
                    check_live_authorization(
                        auth,
                        sequence,
                        &clock_domain_id,
                        at,
                        paused,
                        global_invalidation_sequence,
                        &revoked_grants,
                        &cancelled_events,
                        index,
                    )?;
                }
                "presentation" => {
                    let output_id = event_string(event, "output_id", index)?.to_owned();
                    validate_identifier(&output_id, Some(index), "output_id")?;
                    let key = AuthorizationKey {
                        semantic_event_id: event_string(event, "semantic_event_id", index)?.to_owned(),
                        canonical_cue_digest: event_string(event, "canonical_cue_digest", index)?.to_owned(),
                        grant_id: event_string(event, "grant_id", index)?.to_owned(),
                        form: event_string(event, "form", index)?.to_owned(),
                        channel: event_string(event, "channel", index)?.to_owned(),
                        destination: event_string(event, "destination", index)?.to_owned(),
                        privacy_class: event_string(event, "privacy_class", index)?.to_owned(),
                    };
                    let semantic = semantic_events.get(&key.semantic_event_id)
                        .ok_or(InteractionTraceError::AuthorizationMismatch { index })?;
                    if semantic.digest != key.canonical_cue_digest
                        || semantic.sequence >= sequence
                    {
                        return Err(InteractionTraceError::AuthorizationMismatch { index });
                    }
                    if semantic.privacy_class != key.privacy_class {
                        return Err(InteractionTraceError::PrivacyMismatch { index });
                    }
                    let source_clock = event_string(event, "source_clock_domain_id", index)?.to_owned();
                    validate_identifier(&source_clock, Some(index), "source_clock_domain_id")?;
                    let source_now = event_u64(event, "source_now_monotonic_ns", index)?;
                    if source_clock == clock_domain_id && source_now < at {
                        return Err(InteractionTraceError::ClockRegression { index });
                    }
                    if last_source_time_by_clock
                        .get(&source_clock)
                        .is_some_and(|last| source_now < *last)
                    {
                        return Err(InteractionTraceError::ClockRegression { index });
                    }
                    last_source_time_by_clock.insert(source_clock.clone(), source_now);

                    let stage = event_string(event, "stage", index)?;
                    let observed_by = optional_event_string(event, "observed_by", index)?;
                    let frame_index = optional_event_u64(event, "frame_index", index)?;

                    match stage {
                        "adapter_submitted" => {
                            let auth_sequence = find_authorization(
                                &authorization_by_key,
                                &key,
                                sequence,
                            ).ok_or(InteractionTraceError::MissingAuthorization { index })?;
                            let auth = authorization_by_sequence
                                .get(&auth_sequence)
                                .ok_or(InteractionTraceError::MissingAuthorization { index })?;
                            check_live_authorization(
                                auth,
                                sequence,
                                &source_clock,
                                source_now,
                                paused,
                                global_invalidation_sequence,
                                &revoked_grants,
                                &cancelled_events,
                                index,
                            )?;
                            if frame_index.is_some() || observed_by.is_some() || outputs.contains_key(&output_id) {
                                return Err(InteractionTraceError::OutputStateViolation { index });
                            }
                            outputs.insert(
                                output_id,
                                OutputRecord {
                                    key,
                                    authorization_sequence: auth_sequence,
                                    last_frame_index: None,
                                    stopped: false,
                                },
                            );
                            summary.submitted_outputs += 1;
                        }
                        "frame_presented" => {
                            let output = outputs.get_mut(&output_id)
                                .ok_or(InteractionTraceError::OutputStateViolation { index })?;
                            if output.stopped || output.key != key {
                                return Err(InteractionTraceError::OutputStateViolation { index });
                            }
                            let frame = frame_index.ok_or(InteractionTraceError::OutputStateViolation { index })?;
                            if output.last_frame_index.is_some_and(|last| frame <= last) {
                                return Err(InteractionTraceError::FrameSequenceRegression { index });
                            }
                            if observed_by.is_none_or(str::is_empty) {
                                return Err(InteractionTraceError::OutputStateViolation { index });
                            }
                            let auth = authorization_by_sequence
                                .get(&output.authorization_sequence)
                                .ok_or(InteractionTraceError::MissingAuthorization { index })?;
                            check_live_authorization(
                                auth,
                                sequence,
                                &source_clock,
                                source_now,
                                paused,
                                global_invalidation_sequence,
                                &revoked_grants,
                                &cancelled_events,
                                index,
                            )?;
                            output.last_frame_index = Some(frame);
                            summary.presented_frames += 1;
                        }
                        "output_stopped" => {
                            let output = outputs.get_mut(&output_id)
                                .ok_or(InteractionTraceError::OutputStateViolation { index })?;
                            if output.stopped || output.key != key || observed_by.is_none_or(str::is_empty) {
                                return Err(InteractionTraceError::OutputStateViolation { index });
                            }
                            let frame = frame_index.ok_or(InteractionTraceError::OutputStateViolation { index })?;
                            if output.last_frame_index.is_some_and(|last| frame <= last) {
                                return Err(InteractionTraceError::FrameSequenceRegression { index });
                            }
                            let auth = authorization_by_sequence
                                .get(&output.authorization_sequence)
                                .ok_or(InteractionTraceError::MissingAuthorization { index })?;
                            if auth.clock_domain_id != source_clock {
                                return Err(InteractionTraceError::SourceClockMismatch { index });
                            }
                            output.last_frame_index = Some(frame);
                            output.stopped = true;
                            summary.stopped_outputs += 1;
                        }
                        _ => {
                            return Err(InteractionTraceError::InvalidField {
                                index: Some(index),
                                field: "stage",
                            });
                        }
                    }
                }
                "policy_change" => {
                    let change = event_string(event, "change", index)?;
                    let next_policy_version = event_u64(event, "policy_version", index)?;
                    if next_policy_version <= last_policy_version {
                        return Err(InteractionTraceError::PolicyVersionRegression { index });
                    }
                    last_policy_version = next_policy_version;
                    let grant_id = optional_event_string(event, "grant_id", index)?;
                    let affected_event_id = optional_event_string(event, "affected_event_id", index)?;
                    match change {
                        "pause" => {
                            paused = true;
                            global_invalidation_sequence = Some(sequence);
                        }
                        "resume" => {
                            paused = false;
                            global_invalidation_sequence = Some(sequence);
                        }
                        "revoke" | "replace_grant" => {
                            let grant_id = grant_id.ok_or(InteractionTraceError::InvalidField {
                                index: Some(index),
                                field: "grant_id",
                            })?;
                            revoked_grants.insert(grant_id.to_owned(), sequence);
                        }
                        "cancel_event" => {
                            let event_id = affected_event_id.ok_or(InteractionTraceError::InvalidField {
                                index: Some(index),
                                field: "affected_event_id",
                            })?;
                            cancelled_events.insert(event_id.to_owned(), sequence);
                        }
                        "reduce_motion" | "privacy_downgrade" => {
                            global_invalidation_sequence = Some(sequence);
                            if let Some(grant_id) = grant_id {
                                revoked_grants.insert(grant_id.to_owned(), sequence);
                            }
                        }
                        _ => {
                            return Err(InteractionTraceError::InvalidField {
                                index: Some(index),
                                field: "change",
                            });
                        }
                    }
                }
                "grounding_probe" => {
                    let support = event
                        .get("supporting_sequences")
                        .and_then(Value::as_array)
                        .ok_or(InteractionTraceError::InvalidField {
                            index: Some(index),
                            field: "supporting_sequences",
                        })?;
                    let abstained = event
                        .get("abstained")
                        .and_then(Value::as_bool)
                        .ok_or(InteractionTraceError::InvalidField {
                            index: Some(index),
                            field: "abstained",
                        })?;
                    if !abstained && support.is_empty() {
                        return Err(InteractionTraceError::GroundingSupportMissing { index });
                    }
                    for item in support {
                        let sequence = item.as_u64().ok_or(InteractionTraceError::InvalidField {
                            index: Some(index),
                            field: "supporting_sequences",
                        })?;
                        if sequence >= event_u64(event, "sequence", index)? || !seen_sequences.contains(&sequence) {
                            return Err(InteractionTraceError::GroundingSupportMissing { index });
                        }
                    }
                    summary.grounding_probes += 1;
                }
                "resource_sample" => {}
                _ => return Err(InteractionTraceError::UnknownEventType { index }),
            }
        }

        if let Some((output_id, _)) = outputs.iter().find(|(_, output)| !output.stopped) {
            return Err(InteractionTraceError::OutputNotStopped {
                output_id: output_id.clone(),
            });
        }

        Ok(summary)
    }
}

fn find_authorization(
    by_key: &HashMap<AuthorizationKey, Vec<u64>>,
    key: &AuthorizationKey,
    before_sequence: u64,
) -> Option<u64> {
    by_key
        .get(key)?
        .iter()
        .rev()
        .find(|sequence| **sequence < before_sequence)
        .copied()
}

fn check_live_authorization(
    auth: &AuthorizationRecord,
    current_sequence: u64,
    source_clock_domain_id: &str,
    source_now_monotonic_ns: u64,
    paused: bool,
    global_invalidation_sequence: Option<u64>,
    revoked_grants: &HashMap<String, u64>,
    cancelled_events: &HashMap<String, u64>,
    index: usize,
) -> Result<(), InteractionTraceError> {
    if auth.sequence >= current_sequence {
        return Err(InteractionTraceError::MissingAuthorization { index });
    }
    if auth.clock_domain_id != source_clock_domain_id {
        return Err(InteractionTraceError::SourceClockMismatch { index });
    }
    if source_now_monotonic_ns >= auth.expires_at_monotonic_ns {
        return Err(InteractionTraceError::AuthorizationExpired { index });
    }
    if paused
        || global_invalidation_sequence.is_some_and(|changed| {
            changed > auth.sequence && changed < current_sequence
        })
        || revoked_grants
            .get(&auth.key.grant_id)
            .is_some_and(|revoked| *revoked > auth.sequence && *revoked < current_sequence)
        || cancelled_events
            .get(&auth.key.semantic_event_id)
            .is_some_and(|cancelled| *cancelled > auth.sequence && *cancelled < current_sequence)
    {
        return Err(InteractionTraceError::AuthorizationRevoked { index });
    }
    Ok(())
}

fn root_string<'a>(
    value: &'a Value,
    field: &'static str,
) -> Result<&'a str, InteractionTraceError> {
    required_string(value, field, None)
}

fn event_string<'a>(
    value: &'a Value,
    field: &'static str,
    index: usize,
) -> Result<&'a str, InteractionTraceError> {
    required_string(value, field, Some(index))
}

fn required_string<'a>(
    value: &'a Value,
    field: &'static str,
    index: Option<usize>,
) -> Result<&'a str, InteractionTraceError> {
    value
        .get(field)
        .ok_or(InteractionTraceError::MissingField { index, field })?
        .as_str()
        .ok_or(InteractionTraceError::InvalidField { index, field })
}

fn event_u64(
    value: &Value,
    field: &'static str,
    index: usize,
) -> Result<u64, InteractionTraceError> {
    value
        .get(field)
        .ok_or(InteractionTraceError::MissingField {
            index: Some(index),
            field,
        })?
        .as_u64()
        .ok_or(InteractionTraceError::InvalidField {
            index: Some(index),
            field,
        })
}

fn optional_event_u64(
    value: &Value,
    field: &'static str,
    index: usize,
) -> Result<Option<u64>, InteractionTraceError> {
    match value.get(field) {
        Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or(InteractionTraceError::InvalidField {
                index: Some(index),
                field,
            }),
        None => Err(InteractionTraceError::MissingField {
            index: Some(index),
            field,
        }),
    }
}

fn optional_event_string<'a>(
    value: &'a Value,
    field: &'static str,
    index: usize,
) -> Result<Option<&'a str>, InteractionTraceError> {
    match value.get(field) {
        Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_str()
            .map(Some)
            .ok_or(InteractionTraceError::InvalidField {
                index: Some(index),
                field,
            }),
        None => Err(InteractionTraceError::MissingField {
            index: Some(index),
            field,
        }),
    }
}

fn validate_identifier(
    value: &str,
    index: Option<usize>,
    field: &'static str,
) -> Result<(), InteractionTraceError> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err(InteractionTraceError::InvalidField { index, field });
    }
    Ok(())
}

fn validate_nonempty(
    value: &str,
    index: Option<usize>,
    field: &'static str,
) -> Result<(), InteractionTraceError> {
    if value.is_empty() || value.len() > 256 {
        return Err(InteractionTraceError::InvalidField { index, field });
    }
    Ok(())
}

fn validate_hex(
    value: &str,
    expected_len: usize,
    index: Option<usize>,
    field: &'static str,
) -> Result<(), InteractionTraceError> {
    if value.len() != expected_len || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(InteractionTraceError::InvalidField { index, field });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    const FIXTURES: &str =
        include_str!("../../../../docs/research/SYMTHAEA_INTERACTION_TRACE_FIXTURES_V1_2026-11-27.json");

    fn valid_trace() -> Value {
        let fixtures: Value = serde_json::from_str(FIXTURES).unwrap();
        fixtures["valid_traces"][0].clone()
    }

    fn append_event(trace: &mut Value, event: Value) {
        trace["events"].as_array_mut().unwrap().push(event);
    }

    #[test]
    fn published_positive_revocation_fixture_is_causally_valid() {
        let trace = valid_trace();
        let summary = InteractionTraceVerifier::verify_value(&trace).unwrap();
        assert_eq!(summary.event_count, 8);
        assert_eq!(summary.input_chunks, 1);
        assert_eq!(summary.semantic_cues, 1);
        assert_eq!(summary.authorizations, 1);
        assert_eq!(summary.submitted_outputs, 1);
        assert_eq!(summary.presented_frames, 1);
        assert_eq!(summary.stopped_outputs, 1);
    }

    #[test]
    fn every_published_negative_fixture_is_rejected() {
        let fixtures: Value = serde_json::from_str(FIXTURES).unwrap();
        for case in fixtures["invalid_cases"].as_array().unwrap() {
            assert!(
                InteractionTraceVerifier::verify_value(&case["trace"]).is_err(),
                "fixture {} unexpectedly passed",
                case["name"]
            );
        }
    }

    #[test]
    fn self_reported_outcome_claims_do_not_change_causal_verification() {
        let mut trace = valid_trace();
        let original = InteractionTraceVerifier::verify_value(&trace).unwrap();
        trace["unverified_claims"]["unauthorized_actions"] = json!(999);
        trace["unverified_claims"]["post_expiry_frames"] = json!(999);
        trace["unverified_claims"]["semantic_preservation_score"] = json!(0.0);
        let altered = InteractionTraceVerifier::verify_value(&trace).unwrap();
        assert_eq!(altered, original);
    }

    #[test]
    fn timestamp_regression_is_checked_within_each_clock_domain() {
        let mut trace = valid_trace();
        trace["events"][4]["source_now_monotonic_ns"] = json!(120);
        assert_eq!(
            InteractionTraceVerifier::verify_value(&trace),
            Err(InteractionTraceError::ClockRegression { index: 4 })
        );
    }

    #[test]
    fn renderer_clock_is_not_compared_to_the_source_clock() {
        let mut trace = valid_trace();
        trace["events"][4]["clock_domain_id"] = json!("clock:renderer-1");
        trace["events"][4]["at_monotonic_ns"] = json!(1);
        let result = InteractionTraceVerifier::verify_value(&trace);
        assert!(result.is_ok());
    }

    #[test]
    fn presentation_at_effective_expiry_is_rejected() {
        let mut trace = valid_trace();
        trace["events"][4]["source_now_monotonic_ns"] = json!(1000);
        assert_eq!(
            InteractionTraceVerifier::verify_value(&trace),
            Err(InteractionTraceError::AuthorizationExpired { index: 4 })
        );
    }

    #[test]
    fn same_domain_source_clock_cannot_be_backdated() {
        let mut trace = valid_trace();
        trace["events"][4]["source_now_monotonic_ns"] = json!(129);
        assert_eq!(
            InteractionTraceVerifier::verify_value(&trace),
            Err(InteractionTraceError::ClockRegression { index: 4 })
        );
    }

    #[test]
    fn turn_decision_emitted_after_deadline_is_rejected() {
        let mut trace = valid_trace();
        append_event(
            &mut trace,
            json!({
                "sequence":8,"at_monotonic_ns":160,"clock_domain_id":"clock:synthetic-1",
                "event_type":"turn_decision","candidate_id":"candidate:late",
                "decision":"respond","confidence":0.9,
                "deadline_at_monotonic_ns":170,"emitted_at_monotonic_ns":171
            }),
        );
        assert_eq!(
            InteractionTraceVerifier::verify_value(&trace),
            Err(InteractionTraceError::LateTurnDecision { index: 8 })
        );
    }

    #[test]
    fn a_turn_candidate_can_receive_only_one_decision() {
        let mut trace = valid_trace();
        append_event(
            &mut trace,
            json!({
                "sequence":8,"at_monotonic_ns":160,"clock_domain_id":"clock:synthetic-1",
                "event_type":"turn_decision","candidate_id":"candidate:one",
                "decision":"wait","confidence":0.8,
                "deadline_at_monotonic_ns":180,"emitted_at_monotonic_ns":165
            }),
        );
        append_event(
            &mut trace,
            json!({
                "sequence":9,"at_monotonic_ns":170,"clock_domain_id":"clock:synthetic-1",
                "event_type":"turn_decision","candidate_id":"candidate:one",
                "decision":"respond","confidence":0.99,
                "deadline_at_monotonic_ns":180,"emitted_at_monotonic_ns":175
            }),
        );
        assert_eq!(
            InteractionTraceVerifier::verify_value(&trace),
            Err(InteractionTraceError::DuplicateTurnCandidate { index: 9 })
        );
    }

    #[test]
    fn semantic_event_id_cannot_be_reused_with_a_new_digest() {
        let mut trace = valid_trace();
        append_event(
            &mut trace,
            json!({
                "sequence":8,"at_monotonic_ns":160,"clock_domain_id":"clock:synthetic-1",
                "event_type":"semantic_cue","event_id":"event:1",
                "canonical_cue_digest":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                "semantic_key":"turn:respond","privacy_class":"public","epistemic_status":"measured"
            }),
        );
        assert_eq!(
            InteractionTraceVerifier::verify_value(&trace),
            Err(InteractionTraceError::ConflictingCueDigest { index: 8 })
        );
    }

    #[test]
    fn actuator_side_effect_requires_an_actuator_authorization() {
        let mut trace = valid_trace();
        append_event(
            &mut trace,
            json!({
                "sequence":8,"at_monotonic_ns":160,"clock_domain_id":"clock:synthetic-1",
                "event_type":"action","action_id":"action:1","action_kind":"move",
                "authorization_sequence":2,"state":"completed","side_effect":true,
                "outcome_digest":null
            }),
        );
        assert_eq!(
            InteractionTraceVerifier::verify_value(&trace),
            Err(InteractionTraceError::ActionAuthorizationMissing { index: 8 })
        );
    }

    #[test]
    fn non_abstained_grounding_answers_require_prior_support() {
        let mut trace = valid_trace();
        append_event(
            &mut trace,
            json!({
                "sequence":8,"at_monotonic_ns":160,"clock_domain_id":"clock:synthetic-1",
                "event_type":"grounding_probe","probe_id":"probe:1","probe_kind":"known_fact",
                "answer_digest":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                "supporting_sequences":[],"abstained":false,"scored_correctly":null
            }),
        );
        assert_eq!(
            InteractionTraceVerifier::verify_value(&trace),
            Err(InteractionTraceError::GroundingSupportMissing { index: 8 })
        );
    }

    #[test]
    fn traces_cannot_end_with_an_unstopped_output() {
        let mut trace = valid_trace();
        trace["events"].as_array_mut().unwrap().pop();
        assert!(matches!(
            InteractionTraceVerifier::verify_value(&trace),
            Err(InteractionTraceError::OutputNotStopped { .. })
        ));
    }

    #[test]
    fn presentation_privacy_must_match_the_semantic_cue_and_grant() {
        let mut trace = valid_trace();
        trace["events"][4]["privacy_class"] = json!("restricted");
        assert_eq!(
            InteractionTraceVerifier::verify_value(&trace),
            Err(InteractionTraceError::PrivacyMismatch { index: 4 })
        );
    }

    #[test]
    fn policy_version_cannot_roll_back() {
        let mut trace = valid_trace();
        trace["events"][5]["policy_version"] = json!(2);
        assert_eq!(
            InteractionTraceVerifier::verify_value(&trace),
            Err(InteractionTraceError::PolicyVersionRegression { index: 5 })
        );
    }

    #[test]
    fn authorization_must_use_current_policy_version() {
        let mut trace = valid_trace();
        trace["events"][2]["policy_version"] = json!(2);
        assert_eq!(
            InteractionTraceVerifier::verify_value(&trace),
            Err(InteractionTraceError::PolicyVersionRegression { index: 2 })
        );
    }

    #[test]
    fn authorization_privacy_must_match_the_semantic_cue() {
        let mut trace = valid_trace();
        trace["events"][2]["privacy_class"] = json!("restricted");
        assert_eq!(
            InteractionTraceVerifier::verify_value(&trace),
            Err(InteractionTraceError::PrivacyMismatch { index: 2 })
        );
    }

    fn append_scoped_actuator_authorization(trace: &mut Value, action_scope: &[&str]) {
        append_event(
            trace,
            json!({
                "sequence":8,"at_monotonic_ns":160,"clock_domain_id":"clock:synthetic-1",
                "event_type":"authorization","decision":"authorized","grant_id":"grant:2",
                "permit_digest":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                "canonical_cue_digest":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                "channel":"actuator","destination":"local_private","expires_at_monotonic_ns":900,
                "policy_version":4,"semantic_event_id":"event:1","form":"living_constellation",
                "privacy_class":"public","action_scope":action_scope
            }),
        );
    }

    #[test]
    fn explicitly_scoped_actuator_action_is_accepted() {
        let mut trace = valid_trace();
        append_scoped_actuator_authorization(&mut trace, &["robot.move"]);
        append_event(
            &mut trace,
            json!({
                "sequence":9,"at_monotonic_ns":165,"clock_domain_id":"clock:synthetic-1",
                "event_type":"action","action_id":"action:move-1","action_kind":"robot.move",
                "authorization_sequence":8,"state":"completed","side_effect":true,
                "outcome_digest":null
            }),
        );
        assert!(InteractionTraceVerifier::verify_value(&trace).is_ok());
    }

    #[test]
    fn actuator_action_outside_the_permit_scope_is_rejected() {
        let mut trace = valid_trace();
        append_scoped_actuator_authorization(&mut trace, &["robot.move"]);
        append_event(
            &mut trace,
            json!({
                "sequence":9,"at_monotonic_ns":165,"clock_domain_id":"clock:synthetic-1",
                "event_type":"action","action_id":"action:open-1","action_kind":"robot.open",
                "authorization_sequence":8,"state":"completed","side_effect":true,
                "outcome_digest":null
            }),
        );
        assert_eq!(
            InteractionTraceVerifier::verify_value(&trace),
            Err(InteractionTraceError::ActionAuthorizationMissing { index: 9 })
        );
    }
}

    #[test]
    fn duplicate_action_scope_entries_are_rejected() {
        let mut trace = valid_trace();
        trace["events"][2]["action_scope"] = json!(["robot.move", "robot.move"]);
        assert_eq!(
            InteractionTraceVerifier::verify_value(&trace),
            Err(InteractionTraceError::InvalidField {
                index: Some(2),
                field: "action_scope"
            })
        );
    }