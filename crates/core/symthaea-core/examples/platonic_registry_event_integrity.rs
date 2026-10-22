// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic event-trace controls for enrollment, replacement, revocation,
//! rollback, and replay against a small in-memory registry state machine.
//!
//! This is a protocol model and regression benchmark. "Authority" is a synthetic
//! fixture attribute, not a cryptographically authenticated principal. The model
//! makes no claim that production storage or a real registry enforces these rules.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;

type BenchResult<T> = Result<T, Box<dyn Error>>;

const PROTOCOL_ID: &str = "platonic-registry-temporal-integrity-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Authority {
    Untrusted,
    Enroller,
    Registrar,
    Revoker,
    Recovery,
    Admin,
}

impl Authority {
    fn as_str(self) -> &'static str {
        match self {
            Self::Untrusted => "untrusted",
            Self::Enroller => "enroller",
            Self::Registrar => "registrar",
            Self::Revoker => "revoker",
            Self::Recovery => "recovery",
            Self::Admin => "admin",
        }
    }

    fn can_enroll(self) -> bool {
        matches!(self, Self::Enroller | Self::Admin)
    }

    fn can_replace(self) -> bool {
        matches!(self, Self::Registrar | Self::Admin)
    }

    fn can_revoke(self) -> bool {
        matches!(self, Self::Revoker | Self::Admin)
    }

    fn can_recover(self) -> bool {
        matches!(self, Self::Recovery | Self::Admin)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Action {
    Enroll {
        identity: String,
        digest: String,
        generation: u64,
    },
    Replace {
        identity: String,
        expected_generation: u64,
        generation: u64,
        digest: String,
    },
    Revoke {
        identity: String,
        expected_generation: u64,
        generation: u64,
    },
    Rollback {
        identity: String,
        target_generation: u64,
        target_digest: String,
    },
}

impl Action {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Enroll { .. } => "enroll",
            Self::Replace { .. } => "replace",
            Self::Revoke { .. } => "revoke",
            Self::Rollback { .. } => "rollback",
        }
    }

    fn canonical(&self) -> String {
        let mut output = String::new();
        match self {
            Self::Enroll {
                identity,
                digest,
                generation,
            } => {
                push_field(&mut output, "enroll");
                push_field(&mut output, identity);
                push_field(&mut output, digest);
                push_field(&mut output, &generation.to_string());
            }
            Self::Replace {
                identity,
                expected_generation,
                generation,
                digest,
            } => {
                push_field(&mut output, "replace");
                push_field(&mut output, identity);
                push_field(&mut output, &expected_generation.to_string());
                push_field(&mut output, &generation.to_string());
                push_field(&mut output, digest);
            }
            Self::Revoke {
                identity,
                expected_generation,
                generation,
            } => {
                push_field(&mut output, "revoke");
                push_field(&mut output, identity);
                push_field(&mut output, &expected_generation.to_string());
                push_field(&mut output, &generation.to_string());
            }
            Self::Rollback {
                identity,
                target_generation,
                target_digest,
            } => {
                push_field(&mut output, "rollback");
                push_field(&mut output, identity);
                push_field(&mut output, &target_generation.to_string());
                push_field(&mut output, target_digest);
            }
        }
        output
    }
}

/// Canonical, length-prefixed field encoding. This removes delimiter ambiguity
/// from fixture and state digests; these digests still do not authenticate events.
fn push_field(output: &mut String, field: &str) {
    output.push_str(&field.len().to_string());
    output.push(':');
    output.push_str(field);
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RegistryEvent {
    sequence: u64,
    event_id: String,
    authority: Authority,
    action: Action,
}

impl RegistryEvent {
    fn canonical(&self) -> String {
        let mut output = String::new();
        push_field(&mut output, &self.sequence.to_string());
        push_field(&mut output, &self.event_id);
        push_field(&mut output, self.authority.as_str());
        push_field(&mut output, &self.action.canonical());
        output
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecordStatus {
    Active,
    Revoked,
}

impl RecordStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Revoked => "revoked",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IdentityRecord {
    generation: u64,
    vector_digest: String,
    status: RecordStatus,
    /// Digests previously introduced for this identity. A rollback may select
    /// only a known digest and must create a *new*, forward-moving generation.
    history: BTreeSet<String>,
    /// Highest generation at which revocation was committed.
    revocation_floor: u64,
}

impl IdentityRecord {
    fn new(generation: u64, vector_digest: String) -> Self {
        let mut history = BTreeSet::new();
        history.insert(vector_digest.clone());
        Self {
            generation,
            vector_digest,
            status: RecordStatus::Active,
            history,
            revocation_floor: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Decision {
    accepted: bool,
    code: &'static str,
}

impl Decision {
    fn accepted() -> Self {
        Self {
            accepted: true,
            code: "accepted",
        }
    }

    fn rejected(code: &'static str) -> Self {
        Self {
            accepted: false,
            code,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct RegistryState {
    /// Sequence of the latest well-formed event envelope. Even a rejected
    /// operation consumes its unique envelope sequence to prevent silent reuse.
    last_sequence: u64,
    seen_event_ids: BTreeSet<String>,
    records: BTreeMap<String, IdentityRecord>,
    accepted_operations: usize,
    rejected_operations: usize,
    rejection_counts: BTreeMap<&'static str, usize>,
}

impl RegistryState {
    fn apply(&mut self, event: &RegistryEvent) -> Decision {
        let expected_sequence = self.last_sequence.checked_add(1);
        if expected_sequence != Some(event.sequence) {
            return self.count(Decision::rejected("non_monotonic_sequence"));
        }

        self.last_sequence = event.sequence;
        if self.seen_event_ids.contains(&event.event_id) {
            return self.count(Decision::rejected("duplicate_event_id"));
        }
        self.seen_event_ids.insert(event.event_id.clone());

        let decision = match self.transition(event) {
            Ok(()) => Decision::accepted(),
            Err(code) => Decision::rejected(code),
        };
        self.count(decision)
    }

    fn count(&mut self, decision: Decision) -> Decision {
        if decision.accepted {
            self.accepted_operations += 1;
        } else {
            self.rejected_operations += 1;
            *self.rejection_counts.entry(decision.code).or_default() += 1;
        }
        decision
    }

    fn transition(&mut self, event: &RegistryEvent) -> Result<(), &'static str> {
        match &event.action {
            Action::Enroll {
                identity,
                digest,
                generation,
            } => self.enroll(identity, digest, *generation, event.authority),
            Action::Replace {
                identity,
                expected_generation,
                generation,
                digest,
            } => self.replace(
                identity,
                *expected_generation,
                *generation,
                digest,
                event.authority,
            ),
            Action::Revoke {
                identity,
                expected_generation,
                generation,
            } => self.revoke(
                identity,
                *expected_generation,
                *generation,
                event.authority,
            ),
            Action::Rollback {
                identity,
                target_generation,
                target_digest,
            } => self.rollback(
                identity,
                *target_generation,
                target_digest,
                event.authority,
            ),
        }
    }

    fn enroll(
        &mut self,
        identity: &str,
        digest: &str,
        generation: u64,
        authority: Authority,
    ) -> Result<(), &'static str> {
        if digest.is_empty() {
            return Err("empty_vector_digest");
        }

        match self.records.get(identity) {
            None => {
                if !authority.can_enroll() {
                    return Err("unauthorized");
                }
                if generation != 1 {
                    return Err("initial_generation_not_one");
                }
                self.records.insert(
                    identity.to_owned(),
                    IdentityRecord::new(generation, digest.to_owned()),
                );
                Ok(())
            }
            Some(record) if record.status == RecordStatus::Active => {
                Err("identity_already_active")
            }
            Some(record) => {
                if generation <= record.generation {
                    return Err("stale_generation");
                }
                if !authority.can_recover() {
                    return Err("recovery_required");
                }
                let mut new_record = IdentityRecord::new(generation, digest.to_owned());
                new_record.history.extend(record.history.iter().cloned());
                new_record.revocation_floor =
                    record.revocation_floor.max(record.generation);
                self.records.insert(identity.to_owned(), new_record);
                Ok(())
            }
        }
    }

    fn replace(
        &mut self,
        identity: &str,
        expected_generation: u64,
        generation: u64,
        digest: &str,
        authority: Authority,
    ) -> Result<(), &'static str> {
        if !authority.can_replace() {
            return Err("unauthorized");
        }
        if digest.is_empty() {
            return Err("empty_vector_digest");
        }
        let record = self
            .records
            .get_mut(identity)
            .ok_or("identity_not_found")?;
        if record.status == RecordStatus::Revoked {
            return Err("revoked_tombstone");
        }
        if expected_generation != record.generation {
            return Err("stale_generation");
        }
        let next_generation = record
            .generation
            .checked_add(1)
            .ok_or("generation_exhausted")?;
        if generation != next_generation {
            return Err("generation_not_next");
        }
        if digest == record.vector_digest {
            return Err("same_digest_noop");
        }

        record.history.insert(digest.to_owned());
        record.vector_digest = digest.to_owned();
        record.generation = generation;
        Ok(())
    }

    fn revoke(
        &mut self,
        identity: &str,
        expected_generation: u64,
        generation: u64,
        authority: Authority,
    ) -> Result<(), &'static str> {
        if !authority.can_revoke() {
            return Err("unauthorized");
        }
        let record = self
            .records
            .get_mut(identity)
            .ok_or("identity_not_found")?;
        if record.status == RecordStatus::Revoked {
            return Err("already_revoked");
        }
        if expected_generation != record.generation {
            return Err("stale_generation");
        }
        let next_generation = record
            .generation
            .checked_add(1)
            .ok_or("generation_exhausted")?;
        if generation != next_generation {
            return Err("generation_not_next");
        }

        record.generation = generation;
        record.revocation_floor = generation;
        record.status = RecordStatus::Revoked;
        Ok(())
    }

    fn rollback(
        &mut self,
        identity: &str,
        target_generation: u64,
        target_digest: &str,
        authority: Authority,
    ) -> Result<(), &'static str> {
        if !authority.can_recover() {
            return Err("unauthorized");
        }
        let record = self
            .records
            .get_mut(identity)
            .ok_or("identity_not_found")?;
        // A snapshot rollback may not resurrect a tombstoned identity. Revival
        // requires an explicit forward-generation Recovery enrollment event.
        if record.status == RecordStatus::Revoked {
            return Err("revoked_tombstone");
        }
        let next_generation = record
            .generation
            .checked_add(1)
            .ok_or("generation_exhausted")?;
        if target_generation != next_generation {
            return Err("generation_not_next");
        }
        if !record.history.contains(target_digest) {
            return Err("unknown_snapshot_digest");
        }
        if target_digest == record.vector_digest {
            return Err("same_digest_noop");
        }

        record.vector_digest = target_digest.to_owned();
        record.generation = target_generation;
        Ok(())
    }

    fn state_digest(&self) -> String {
        let mut canonical = String::new();
        push_field(&mut canonical, "registry-state-v1");
        push_field(&mut canonical, &self.last_sequence.to_string());
        push_field(&mut canonical, &self.records.len().to_string());
        for (identity, record) in &self.records {
            push_field(&mut canonical, identity);
            push_field(&mut canonical, &record.generation.to_string());
            push_field(&mut canonical, &record.vector_digest);
            push_field(&mut canonical, record.status.as_str());
            push_field(&mut canonical, &record.revocation_floor.to_string());
            push_field(&mut canonical, &record.history.len().to_string());
            for digest in &record.history {
                push_field(&mut canonical, digest);
            }
        }
        push_field(&mut canonical, &self.seen_event_ids.len().to_string());
        for event_id in &self.seen_event_ids {
            push_field(&mut canonical, event_id);
        }
        blake3::hash(canonical.as_bytes()).to_hex().to_string()
    }

    fn rejection_summary(&self) -> String {
        if self.rejection_counts.is_empty() {
            return "none".to_owned();
        }
        self.rejection_counts
            .iter()
            .map(|(code, count)| format!("{code}:{count}"))
            .collect::<Vec<_>>()
            .join(";")
    }

    fn record_summary(&self) -> String {
        if self.records.is_empty() {
            return "none".to_owned();
        }
        self.records
            .iter()
            .map(|(identity, record)| {
                format!(
                    "{identity}:{}:{}:{}",
                    record.generation,
                    record.status.as_str(),
                    record.vector_digest
                )
            })
            .collect::<Vec<_>>()
            .join(";")
    }
}

#[derive(Debug)]
struct TraceCase {
    id: &'static str,
    events: Vec<RegistryEvent>,
    expected_codes: Vec<&'static str>,
}

fn event(
    sequence: u64,
    event_id: &str,
    authority: Authority,
    action: Action,
) -> RegistryEvent {
    RegistryEvent {
        sequence,
        event_id: event_id.to_owned(),
        authority,
        action,
    }
}

fn enroll(
    sequence: u64,
    event_id: &str,
    authority: Authority,
    identity: &str,
    digest: &str,
    generation: u64,
) -> RegistryEvent {
    event(
        sequence,
        event_id,
        authority,
        Action::Enroll {
            identity: identity.to_owned(),
            digest: digest.to_owned(),
            generation,
        },
    )
}

fn replace(
    sequence: u64,
    event_id: &str,
    authority: Authority,
    identity: &str,
    expected_generation: u64,
    generation: u64,
    digest: &str,
) -> RegistryEvent {
    event(
        sequence,
        event_id,
        authority,
        Action::Replace {
            identity: identity.to_owned(),
            expected_generation,
            generation,
            digest: digest.to_owned(),
        },
    )
}

fn revoke(
    sequence: u64,
    event_id: &str,
    authority: Authority,
    identity: &str,
    expected_generation: u64,
    generation: u64,
) -> RegistryEvent {
    event(
        sequence,
        event_id,
        authority,
        Action::Revoke {
            identity: identity.to_owned(),
            expected_generation,
            generation,
        },
    )
}

fn rollback(
    sequence: u64,
    event_id: &str,
    authority: Authority,
    identity: &str,
    target_generation: u64,
    target_digest: &str,
) -> RegistryEvent {
    event(
        sequence,
        event_id,
        authority,
        Action::Rollback {
            identity: identity.to_owned(),
            target_generation,
            target_digest: target_digest.to_owned(),
        },
    )
}

fn trace_cases() -> Vec<TraceCase> {
    vec![
        TraceCase {
            id: "valid_lifecycle_recovery",
            events: vec![
                enroll(1, "alice-enroll-1", Authority::Enroller, "alice", "vec-A", 1),
                replace(2, "alice-replace-2", Authority::Registrar, "alice", 1, 2, "vec-B"),
                revoke(3, "alice-revoke-3", Authority::Revoker, "alice", 2, 3),
                enroll(4, "alice-stale-enroll-4", Authority::Enroller, "alice", "vec-C", 2),
                rollback(5, "alice-rollback-5", Authority::Recovery, "alice", 4, "vec-A"),
                enroll(6, "alice-recover-6", Authority::Recovery, "alice", "vec-C", 4),
            ],
            expected_codes: vec![
                "accepted",
                "accepted",
                "accepted",
                "stale_generation",
                "revoked_tombstone",
                "accepted",
            ],
        },
        TraceCase {
            id: "unauthorized_initial_enroll",
            events: vec![enroll(
                1,
                "eve-enroll-1",
                Authority::Untrusted,
                "eve",
                "vec-E",
                1,
            )],
            expected_codes: vec!["unauthorized"],
        },
        TraceCase {
            id: "event_id_replay",
            events: vec![
                enroll(1, "reused-event-id", Authority::Enroller, "bob", "vec-B", 1),
                revoke(2, "reused-event-id", Authority::Revoker, "bob", 1, 2),
                revoke(3, "bob-revoke-3", Authority::Revoker, "bob", 1, 2),
            ],
            expected_codes: vec!["accepted", "duplicate_event_id", "accepted"],
        },
        TraceCase {
            id: "stale_replacement",
            events: vec![
                enroll(1, "carol-enroll-1", Authority::Enroller, "carol", "vec-C1", 1),
                replace(2, "carol-stale-replace-2", Authority::Registrar, "carol", 0, 1, "vec-C2"),
            ],
            expected_codes: vec!["accepted", "stale_generation"],
        },
        TraceCase {
            id: "authorized_active_rollback",
            events: vec![
                enroll(1, "dana-enroll-1", Authority::Enroller, "dana", "vec-D1", 1),
                replace(2, "dana-replace-2", Authority::Registrar, "dana", 1, 2, "vec-D2"),
                rollback(3, "dana-rollback-3", Authority::Recovery, "dana", 3, "vec-D1"),
            ],
            expected_codes: vec!["accepted", "accepted", "accepted"],
        },
        TraceCase {
            id: "unknown_snapshot_rejected",
            events: vec![
                enroll(1, "erin-enroll-1", Authority::Enroller, "erin", "vec-E1", 1),
                rollback(2, "erin-rollback-2", Authority::Recovery, "erin", 2, "never-enrolled"),
            ],
            expected_codes: vec!["accepted", "unknown_snapshot_digest"],
        },
        TraceCase {
            id: "unauthorized_revocation",
            events: vec![
                enroll(1, "frank-enroll-1", Authority::Enroller, "frank", "vec-F", 1),
                revoke(2, "frank-revoke-2", Authority::Enroller, "frank", 1, 2),
            ],
            expected_codes: vec!["accepted", "unauthorized"],
        },
        TraceCase {
            id: "sequence_regression",
            events: vec![
                enroll(1, "gina-enroll-1", Authority::Enroller, "gina", "vec-G", 1),
                revoke(1, "gina-revoke-replayed-sequence", Authority::Revoker, "gina", 1, 2),
                revoke(2, "gina-revoke-2", Authority::Revoker, "gina", 1, 2),
            ],
            expected_codes: vec!["accepted", "non_monotonic_sequence", "accepted"],
        },
        TraceCase {
            id: "recovery_authority_required",
            events: vec![
                enroll(1, "hank-enroll-1", Authority::Enroller, "hank", "vec-H", 1),
                revoke(2, "hank-revoke-2", Authority::Revoker, "hank", 1, 2),
                enroll(3, "hank-reenroll-without-recovery", Authority::Enroller, "hank", "vec-H2", 3),
            ],
            expected_codes: vec!["accepted", "accepted", "recovery_required"],
        },
        TraceCase {
            id: "generation_must_advance_exactly_once",
            events: vec![
                enroll(1, "ivy-enroll-1", Authority::Enroller, "ivy", "vec-I1", 1),
                replace(2, "ivy-skip-generation", Authority::Registrar, "ivy", 1, 3, "vec-I2"),
            ],
            expected_codes: vec!["accepted", "generation_not_next"],
        },
        TraceCase {
            id: "unknown_identity_cannot_be_revoked",
            events: vec![revoke(1, "no-such-revoke", Authority::Revoker, "absent", 0, 1)],
            expected_codes: vec!["identity_not_found"],
        },
        TraceCase {
            id: "duplicate_active_enrollment",
            events: vec![
                enroll(1, "jules-enroll-1", Authority::Enroller, "jules", "vec-J", 1),
                enroll(2, "jules-enroll-2", Authority::Enroller, "jules", "vec-J2", 2),
            ],
            expected_codes: vec!["accepted", "identity_already_active"],
        },
        TraceCase {
            id: "unauthorized_replacement",
            events: vec![
                enroll(1, "kai-enroll-1", Authority::Enroller, "kai", "vec-K1", 1),
                replace(2, "kai-unauthorized-replace", Authority::Enroller, "kai", 1, 2, "vec-K2"),
            ],
            expected_codes: vec!["accepted", "unauthorized"],
        },
    ]
}

fn main() -> BenchResult<()> {
    let source_sha = std::env::var("SYMTHAEA_SOURCE_SHA")
        .unwrap_or_else(|_| "UNBOUND_LOCAL".to_owned());
    let cases = trace_cases();
    let mut fixture_canonical = String::new();
    let mut execution_canonical = String::new();
    let mut total_steps = 0usize;
    let mut matching_steps = 0usize;
    let mut accepted_steps = 0usize;
    let mut rejected_steps = 0usize;
    let mut failed_cases = 0usize;

    println!("# protocol={PROTOCOL_ID}");
    println!("# model=synthetic in-memory state machine; authority attributes are not cryptographic authentication");
    println!("case_id,step,event_id,sequence,action,authority,expected_code,observed_code,step_pass,record_count,last_sequence,accepted_operations,rejected_operations,state_blake3,records");

    for case in &cases {
        let mut state = RegistryState::default();
        let mut case_ok = case.events.len() == case.expected_codes.len();

        for (index, event) in case.events.iter().enumerate() {
            let expected = case
                .expected_codes
                .get(index)
                .copied()
                .unwrap_or("missing_expectation");
            fixture_canonical.push_str(&format!("{}|{}|{}|", case.id, index + 1, event.canonical()));
            fixture_canonical.push_str(expected);
            fixture_canonical.push('\n');

            let decision = state.apply(event);
            total_steps += 1;
            if decision.accepted {
                accepted_steps += 1;
            } else {
                rejected_steps += 1;
            }
            let step_pass = decision.code == expected;
            if step_pass {
                matching_steps += 1;
            } else {
                case_ok = false;
            }

            let state_digest = state.state_digest();
            execution_canonical.push_str(&format!(
                "{}|{}|{}|{}|{}|{}\n",
                case.id,
                index + 1,
                event.event_id,
                expected,
                decision.code,
                state_digest
            ));
            println!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                case.id,
                index + 1,
                event.event_id,
                event.sequence,
                event.action.as_str(),
                event.authority.as_str(),
                expected,
                decision.code,
                if step_pass { "pass" } else { "fail" },
                state.records.len(),
                state.last_sequence,
                state.accepted_operations,
                state.rejected_operations,
                state_digest,
                state.record_summary()
            );
        }

        if case_ok {
            println!(
                "# case_summary=case:{},status:pass,steps:{},rejections:{}",
                case.id,
                case.events.len(),
                state.rejection_summary()
            );
        } else {
            failed_cases += 1;
            println!(
                "# case_summary=case:{},status:fail,steps:{},expected_steps:{},rejections:{}",
                case.id,
                case.events.len(),
                case.expected_codes.len(),
                state.rejection_summary()
            );
        }
    }

    let fixture_digest = blake3::hash(fixture_canonical.as_bytes()).to_hex().to_string();
    let execution_digest = blake3::hash(execution_canonical.as_bytes()).to_hex().to_string();
    println!(
        "# temporal_integrity_receipt=protocol:{PROTOCOL_ID},source_sha:{source_sha},case_count:{},step_count:{total_steps},matching_steps:{matching_steps},failed_cases:{failed_cases},accepted_operations:{accepted_steps},rejected_operations:{rejected_steps},fixture_blake3:{fixture_digest},execution_blake3:{execution_digest}",
        cases.len()
    );

    if failed_cases > 0 || matching_steps != total_steps {
        return Err(format!(
            "temporal integrity trace contract failed: {failed_cases} cases, {} of {total_steps} matching outcomes",
            total_steps.saturating_sub(matching_steps)
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply_all(events: &[RegistryEvent]) -> RegistryState {
        let mut state = RegistryState::default();
        for event in events {
            state.apply(event);
        }
        state
    }

    #[test]
    fn frozen_trace_suite_matches_every_expected_outcome() {
        let cases = trace_cases();
        assert!(cases.len() >= 10);
        let mut executed_steps = 0usize;
        for case in cases {
            assert_eq!(
                case.events.len(),
                case.expected_codes.len(),
                "fixture length mismatch in {}",
                case.id
            );
            let mut state = RegistryState::default();
            for (event, expected) in case.events.iter().zip(case.expected_codes.iter()) {
                let decision = state.apply(event);
                assert_eq!(
                    decision.code, *expected,
                    "unexpected decision in case {} at event {}",
                    case.id, event.event_id
                );
                executed_steps += 1;
            }
        }
        assert!(executed_steps >= 25, "suite should cover multiple temporal steps");
    }

    #[test]
    fn revoked_identity_cannot_be_resurrected_by_rollback() {
        let events = trace_cases()
            .into_iter()
            .find(|case| case.id == "valid_lifecycle_recovery")
            .expect("lifecycle case");
        let mut state = RegistryState::default();
        for (event, expected) in events.events.iter().zip(events.expected_codes.iter()) {
            let decision = state.apply(event);
            assert_eq!(decision.code, *expected);
            if event.event_id == "alice-rollback-5" {
                let record = state.records.get("alice").expect("tombstone");
                assert_eq!(record.status, RecordStatus::Revoked);
                assert_eq!(record.generation, 3);
            }
        }
        let record = state.records.get("alice").expect("recovered identity");
        assert_eq!(record.status, RecordStatus::Active);
        assert_eq!(record.generation, 4);
        assert_eq!(record.vector_digest, "vec-C");
        assert_eq!(record.revocation_floor, 3);
    }

    #[test]
    fn replayed_event_id_cannot_repeat_a_different_operation() {
        let case = trace_cases()
            .into_iter()
            .find(|case| case.id == "event_id_replay")
            .expect("replay case");
        let mut state = RegistryState::default();
        assert_eq!(state.apply(&case.events[0]).code, "accepted");
        let before = state.records.get("bob").expect("enrolled").clone();
        assert_eq!(state.apply(&case.events[1]).code, "duplicate_event_id");
        assert_eq!(state.records.get("bob").expect("still enrolled"), &before);
        assert_eq!(state.records["bob"].status, RecordStatus::Active);
        assert_eq!(state.apply(&case.events[2]).code, "accepted");
        assert_eq!(state.records["bob"].status, RecordStatus::Revoked);
    }

    #[test]
    fn sequence_regression_does_not_mutate_identity_state_or_watermark() {
        let case = trace_cases()
            .into_iter()
            .find(|case| case.id == "sequence_regression")
            .expect("sequence case");
        let mut state = RegistryState::default();
        assert_eq!(state.apply(&case.events[0]).code, "accepted");
        let before_record = state.records.get("gina").expect("enrolled").clone();
        let before_sequence = state.last_sequence;
        assert_eq!(
            state.apply(&case.events[1]).code,
            "non_monotonic_sequence"
        );
        assert_eq!(state.last_sequence, before_sequence);
        assert_eq!(state.records.get("gina").expect("unchanged"), &before_record);
    }

    #[test]
    fn unauthorized_replacement_does_not_change_the_identity_record() {
        let case = trace_cases()
            .into_iter()
            .find(|case| case.id == "unauthorized_replacement")
            .expect("unauthorized replacement case");
        let mut state = RegistryState::default();
        assert_eq!(state.apply(&case.events[0]).code, "accepted");
        let before = state.records.get("kai").expect("enrolled").clone();
        assert_eq!(state.apply(&case.events[1]).code, "unauthorized");
        assert_eq!(state.records.get("kai").expect("still enrolled"), &before);
    }

    #[test]
    fn valid_active_rollback_is_forward_versioned_and_limited_to_known_digest() {
        let case = trace_cases()
            .into_iter()
            .find(|case| case.id == "authorized_active_rollback")
            .expect("rollback case");
        let state = apply_all(&case.events);
        let record = state.records.get("dana").expect("record");
        assert_eq!(record.status, RecordStatus::Active);
        assert_eq!(record.generation, 3);
        assert_eq!(record.vector_digest, "vec-D1");
        assert!(record.history.contains("vec-D2"));
    }

    #[test]
    fn length_prefixed_event_encoding_has_no_delimiter_boundary_collision() {
        let left = enroll(
            1,
            "event-a",
            Authority::Enroller,
            "identity|split",
            "digest",
            1,
        );
        let right = enroll(
            1,
            "event-a",
            Authority::Enroller,
            "identity",
            "split|digest",
            1,
        );
        assert_ne!(left.canonical(), right.canonical());
    }

    #[test]
    fn generation_overflow_fails_closed_in_replace_revoke_and_rollback() {
        let mut state = RegistryState::default();
        let mut history = BTreeSet::new();
        history.insert("digest-max".to_owned());
        state.records.insert(
            "maxed".to_owned(),
            IdentityRecord {
                generation: u64::MAX,
                vector_digest: "digest-max".to_owned(),
                status: RecordStatus::Active,
                history,
                revocation_floor: 0,
            },
        );
        assert_eq!(
            state.replace("maxed", u64::MAX, u64::MAX, "digest-next", Authority::Registrar),
            Err("generation_exhausted")
        );
        assert_eq!(
            state.revoke("maxed", u64::MAX, u64::MAX, Authority::Revoker),
            Err("generation_exhausted")
        );
        assert_eq!(
            state.rollback("maxed", u64::MAX, "digest-max", Authority::Recovery),
            Err("generation_exhausted")
        );
    }

    #[test]
    fn registry_state_digest_is_deterministic_and_binds_identity_state() {
        let case = trace_cases()
            .into_iter()
            .find(|case| case.id == "authorized_active_rollback")
            .expect("rollback case");
        let first = apply_all(&case.events).state_digest();
        let second = apply_all(&case.events).state_digest();
        assert_eq!(first, second);

        let different = apply_all(&[
            enroll(1, "dana-enroll-1", Authority::Enroller, "dana", "different-vector", 1),
        ])
        .state_digest();
        assert_ne!(first, different);
    }
}
