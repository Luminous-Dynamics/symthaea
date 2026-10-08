//! SYM-CIV-007: tamper-evident governance audit-chain smoke.
//!
//! Uses SHA-256 with domain separation and length-prefixed fields. This is a
//! hash-chain smoke, not a Merkle transparency service: it has no Merkle
//! inclusion/consistency proofs, signed tree heads, or deployed witness network.
//! An independently retained checkpoint is necessary to detect a wholesale
//! rewrite of the chain.
//!
//! Claim ceiling: local hash-chain, checkpoint, and witness-quorum controls only.

use sha2::{Digest as ShaDigest, Sha256};
use std::collections::BTreeSet;

type Hash = [u8; 32];

const EVENT_DOMAIN: &[u8] = b"mycelix-governance-audit-event-v1\0";

#[derive(Clone, Debug, Eq, PartialEq)]
struct EventBody {
    event_key: String,
    actor: String,
    action: String,
    target: String,
    scope: String,
    reason: String,
    policy_version: String,
    effective_epoch: u64,
}

impl EventBody {
    fn new(
        event_key: &str,
        actor: &str,
        action: &str,
        target: &str,
        scope: &str,
        reason: &str,
        policy_version: &str,
        effective_epoch: u64,
    ) -> Self {
        Self {
            event_key: event_key.to_owned(),
            actor: actor.to_owned(),
            action: action.to_owned(),
            target: target.to_owned(),
            scope: scope.to_owned(),
            reason: reason.to_owned(),
            policy_version: policy_version.to_owned(),
            effective_epoch,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AuditEvent {
    sequence: u64,
    previous_hash: Option<Hash>,
    body: EventBody,
    hash: Hash,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AuditCheckpoint {
    log_id: String,
    tree_size: u64,
    head_hash: Option<Hash>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct WitnessReceipt {
    witness_id: String,
    authority_lineage: String,
    checkpoint: AuditCheckpoint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AuditFailure {
    DuplicateEventKey,
    NonMonotonicSequence,
    BrokenPreviousLink,
    EventHashMismatch,
    UnknownLog,
    FutureCheckpoint,
    CheckpointMismatch,
    StaleCheckpoint,
    ForkDetected,
    ConsistencyProofRequired,
    InsufficientWitnesses,
    DuplicateWitness,
    SharedWitnessLineage,
    WitnessCheckpointMismatch,
    NoIntegrityFailureToRecord,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct IntegrityFinding {
    log_id: String,
    checkpoint: AuditCheckpoint,
    detector: String,
    detected_epoch: u64,
    failure: AuditFailure,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AuditLog {
    log_id: String,
    events: Vec<AuditEvent>,
}

fn update_field(hasher: &mut Sha256, field: &[u8]) {
    hasher.update((field.len() as u64).to_be_bytes());
    hasher.update(field);
}

fn hash_event(
    log_id: &str,
    sequence: u64,
    previous_hash: Option<Hash>,
    body: &EventBody,
) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update(EVENT_DOMAIN);
    update_field(&mut hasher, log_id.as_bytes());
    hasher.update(sequence.to_be_bytes());

    match previous_hash {
        Some(previous) => {
            hasher.update([1_u8]);
            hasher.update(previous);
        }
        None => hasher.update([0_u8]),
    }

    update_field(&mut hasher, body.event_key.as_bytes());
    update_field(&mut hasher, body.actor.as_bytes());
    update_field(&mut hasher, body.action.as_bytes());
    update_field(&mut hasher, body.target.as_bytes());
    update_field(&mut hasher, body.scope.as_bytes());
    update_field(&mut hasher, body.reason.as_bytes());
    update_field(&mut hasher, body.policy_version.as_bytes());
    hasher.update(body.effective_epoch.to_be_bytes());

    hasher.finalize().into()
}

impl AuditLog {
    fn new(log_id: &str) -> Self {
        Self {
            log_id: log_id.to_owned(),
            events: Vec::new(),
        }
    }

    fn append(&mut self, body: EventBody) -> Result<AuditEvent, AuditFailure> {
        if self.events.iter().any(|event| event.body.event_key == body.event_key) {
            return Err(AuditFailure::DuplicateEventKey);
        }

        let sequence = self.events.len() as u64 + 1;
        let previous_hash = self.events.last().map(|event| event.hash);
        let hash = hash_event(&self.log_id, sequence, previous_hash, &body);
        let event = AuditEvent {
            sequence,
            previous_hash,
            body,
            hash,
        };
        self.events.push(event.clone());
        Ok(event)
    }

    fn verify(&self) -> Result<(), AuditFailure> {
        let mut previous_hash = None;
        let mut event_keys = BTreeSet::new();

        for (index, event) in self.events.iter().enumerate() {
            if event.sequence != index as u64 + 1 {
                return Err(AuditFailure::NonMonotonicSequence);
            }
            if event.previous_hash != previous_hash {
                return Err(AuditFailure::BrokenPreviousLink);
            }
            if !event_keys.insert(event.body.event_key.as_str()) {
                return Err(AuditFailure::DuplicateEventKey);
            }

            let expected = hash_event(
                &self.log_id,
                event.sequence,
                event.previous_hash,
                &event.body,
            );
            if expected != event.hash {
                return Err(AuditFailure::EventHashMismatch);
            }
            previous_hash = Some(event.hash);
        }
        Ok(())
    }

    fn checkpoint(&self) -> AuditCheckpoint {
        AuditCheckpoint {
            log_id: self.log_id.clone(),
            tree_size: self.events.len() as u64,
            head_hash: self.events.last().map(|event| event.hash),
        }
    }

    fn verify_checkpoint_prefix(
        &self,
        checkpoint: &AuditCheckpoint,
    ) -> Result<(), AuditFailure> {
        self.verify()?;

        if checkpoint.log_id != self.log_id {
            return Err(AuditFailure::UnknownLog);
        }
        if checkpoint.tree_size > self.events.len() as u64 {
            return Err(AuditFailure::FutureCheckpoint);
        }

        let expected_head = if checkpoint.tree_size == 0 {
            None
        } else {
            Some(self.events[(checkpoint.tree_size - 1) as usize].hash)
        };
        if checkpoint.head_hash != expected_head {
            return Err(AuditFailure::CheckpointMismatch);
        }
        Ok(())
    }

    fn verify_current_checkpoint(
        &self,
        checkpoint: &AuditCheckpoint,
    ) -> Result<(), AuditFailure> {
        self.verify_checkpoint_prefix(checkpoint)?;
        if checkpoint.tree_size != self.events.len() as u64
            || checkpoint.head_hash != self.events.last().map(|event| event.hash)
        {
            return Err(AuditFailure::StaleCheckpoint);
        }
        Ok(())
    }

    // Test-only adversarial helper: models an operator rewriting history and
    // recomputing the entire chain. Old external checkpoints must still expose it.
    fn recompute_chain_for_test(&mut self) {
        let mut previous_hash = None;
        for event in &mut self.events {
            event.previous_hash = previous_hash;
            event.hash = hash_event(
                &self.log_id,
                event.sequence,
                event.previous_hash,
                &event.body,
            );
            previous_hash = Some(event.hash);
        }
    }
}

fn detect_checkpoint_fork(
    first: &AuditCheckpoint,
    second: &AuditCheckpoint,
) -> Result<(), AuditFailure> {
    if first.log_id != second.log_id {
        return Err(AuditFailure::UnknownLog);
    }
    if first.tree_size != second.tree_size {
        // Different sizes require a Merkle consistency proof or equivalent
        // protocol; this chain smoke does not claim to prove cross-size consistency.
        return Err(AuditFailure::ConsistencyProofRequired);
    }
    if first.head_hash != second.head_hash {
        return Err(AuditFailure::ForkDetected);
    }
    Ok(())
}

fn verify_witness_quorum(
    receipts: &[WitnessReceipt],
    checkpoint: &AuditCheckpoint,
    minimum_witnesses: usize,
) -> Result<(), AuditFailure> {
    if receipts.len() < minimum_witnesses {
        return Err(AuditFailure::InsufficientWitnesses);
    }

    let mut witness_ids = BTreeSet::new();
    let mut lineages = BTreeSet::new();
    for receipt in receipts {
        if &receipt.checkpoint != checkpoint {
            return Err(AuditFailure::WitnessCheckpointMismatch);
        }
        if !witness_ids.insert(receipt.witness_id.as_str()) {
            return Err(AuditFailure::DuplicateWitness);
        }
        if !lineages.insert(receipt.authority_lineage.as_str()) {
            return Err(AuditFailure::SharedWitnessLineage);
        }
    }
    Ok(())
}

fn record_integrity_finding(
    log: &AuditLog,
    detector: &str,
    detected_epoch: u64,
) -> Result<IntegrityFinding, AuditFailure> {
    let failure = log
        .verify()
        .err()
        .ok_or(AuditFailure::NoIntegrityFailureToRecord)?;

    Ok(IntegrityFinding {
        log_id: log.log_id.clone(),
        checkpoint: log.checkpoint(),
        detector: detector.to_owned(),
        detected_epoch,
        failure,
    })
}

fn sample_body(key: &str, action: &str, reason: &str, epoch: u64) -> EventBody {
    EventBody::new(
        key,
        "authority-civ-1",
        action,
        "deployment-v1",
        "critical-scope-v1",
        reason,
        "policy-v1",
        epoch,
    )
}

fn main() {
    let mut log = AuditLog::new("civ-governance-log-v1");
    log.append(sample_body("event-1", "authorize", "quorum-approved", 100))
        .expect("first event");
    log.append(sample_body("event-2", "suspend", "rights-concern", 110))
        .expect("second event");
    let checkpoint_two = log.checkpoint();
    log.append(sample_body("event-3", "review", "contest-reviewed", 120))
        .expect("third event");
    let checkpoint_three = log.checkpoint();

    assert_eq!(log.verify(), Ok(()));

    // Log identity is inside the domain-separated event hash: copying the
    // same event bytes to another log identity is rejected as cross-log replay.
    let mut cross_log_replay = AuditLog::new("civ-governance-log-v2");
    cross_log_replay.events = log.events.clone();
    assert_eq!(
        cross_log_replay.verify(),
        Err(AuditFailure::EventHashMismatch)
    );

    assert_eq!(log.verify_checkpoint_prefix(&checkpoint_two), Ok(()));
    assert_eq!(
        log.verify_current_checkpoint(&checkpoint_two),
        Err(AuditFailure::StaleCheckpoint)
    );
    assert_eq!(log.verify_current_checkpoint(&checkpoint_three), Ok(()));

    // Duplicate event keys cannot append a second event.
    assert_eq!(
        log.append(sample_body("event-3", "revoke", "duplicate-key", 121)),
        Err(AuditFailure::DuplicateEventKey)
    );
    assert_eq!(log.events.len(), 3);

    // Payload mutation with the old digest is detected.
    let mut tampered = log.clone();
    tampered.events[1].body.reason = "rewritten-after-the-fact".to_owned();
    assert_eq!(tampered.verify(), Err(AuditFailure::EventHashMismatch));
    let original_tamper_hash = tampered.events[1].hash;
    let finding = record_integrity_finding(&tampered, "independent-auditor-1", 130)
        .expect("invalid history should create a separate finding");
    assert_eq!(finding.failure, AuditFailure::EventHashMismatch);
    assert_eq!(tampered.events[1].hash, original_tamper_hash);

    // Omission and reordering are detected by sequence and previous-head checks.
    let mut omitted = log.clone();
    omitted.events.remove(1);
    assert_eq!(omitted.verify(), Err(AuditFailure::NonMonotonicSequence));

    let mut reordered = log.clone();
    reordered.events.swap(0, 1);
    assert_eq!(reordered.verify(), Err(AuditFailure::NonMonotonicSequence));

    // A broken previous-head link is rejected.
    let mut broken_link = log.clone();
    broken_link.events[2].previous_hash = Some([0xA5; 32]);
    assert_eq!(broken_link.verify(), Err(AuditFailure::BrokenPreviousLink));

    // Even a wholesale rewrite with a recomputed chain cannot match a previously
    // retained checkpoint. Without external checkpoints/witnesses, hash chaining
    // alone cannot prove that a fully rewritten history is original.
    let mut rewritten_history = log.clone();
    rewritten_history.events[0].body.reason = "rewritten-and-rechained".to_owned();
    rewritten_history.recompute_chain_for_test();
    assert_eq!(rewritten_history.verify(), Ok(()));
    assert_eq!(
        rewritten_history.verify_checkpoint_prefix(&checkpoint_three),
        Err(AuditFailure::CheckpointMismatch)
    );

    // Two different heads at the same log size are an explicit fork.
    let mut fork = AuditLog::new("civ-governance-log-v1");
    fork.append(sample_body("event-1", "authorize", "quorum-approved", 100))
        .expect("fork event 1");
    fork.append(sample_body("event-2", "suspend", "rights-concern", 110))
        .expect("fork event 2");
    fork.append(sample_body("event-3", "review", "different-review-outcome", 120))
        .expect("fork event 3");
    assert_eq!(
        detect_checkpoint_fork(&checkpoint_three, &fork.checkpoint()),
        Err(AuditFailure::ForkDetected)
    );

    // Independent witnesses must attest to the exact same checkpoint.
    let witnesses = vec![
        WitnessReceipt {
            witness_id: "witness-a".to_owned(),
            authority_lineage: "lineage-a".to_owned(),
            checkpoint: checkpoint_three.clone(),
        },
        WitnessReceipt {
            witness_id: "witness-b".to_owned(),
            authority_lineage: "lineage-b".to_owned(),
            checkpoint: checkpoint_three.clone(),
        },
        WitnessReceipt {
            witness_id: "witness-c".to_owned(),
            authority_lineage: "lineage-c".to_owned(),
            checkpoint: checkpoint_three.clone(),
        },
    ];
    assert_eq!(
        verify_witness_quorum(&witnesses, &checkpoint_three, 3),
        Ok(())
    );

    let duplicated_witness = vec![
        witnesses[0].clone(),
        witnesses[0].clone(),
        witnesses[2].clone(),
    ];
    assert_eq!(
        verify_witness_quorum(&duplicated_witness, &checkpoint_three, 3),
        Err(AuditFailure::DuplicateWitness)
    );

    let shared_lineage = vec![
        witnesses[0].clone(),
        WitnessReceipt {
            witness_id: "witness-other-id".to_owned(),
            authority_lineage: "lineage-a".to_owned(),
            checkpoint: checkpoint_three.clone(),
        },
        witnesses[2].clone(),
    ];
    assert_eq!(
        verify_witness_quorum(&shared_lineage, &checkpoint_three, 3),
        Err(AuditFailure::SharedWitnessLineage)
    );

    let stale_witness = WitnessReceipt {
        witness_id: "witness-stale".to_owned(),
        authority_lineage: "lineage-stale".to_owned(),
        checkpoint: checkpoint_two,
    };
    assert_eq!(
        verify_witness_quorum(
            &[witnesses[0].clone(), witnesses[1].clone(), stale_witness],
            &checkpoint_three,
            3,
        ),
        Err(AuditFailure::WitnessCheckpointMismatch)
    );

    println!("SYM-CIV-007 PASS: hash-chain, checkpoint, fork and witness-shape smoke controls hold.");
    println!("Claim ceiling: SHA-256 chain smoke only; not a Merkle transparency log, signature verifier, or production witness network.");
}
