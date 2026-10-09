//! SYM-CIV-013: durable witness recovery and compare-and-advance model.
//!
//! This is a deterministic protocol smoke, NOT a filesystem durability claim.
//! "Sync" boundaries are represented by explicit durable snapshots so crashes,
//! recovery, and stale-snapshot restoration can be explored without claiming
//! that this model substitutes for fsync, atomic replacement, or a real monotonic
//! external anchor. Fixture receipts are hashes, not digital signatures.

use sha2::{Digest, Sha256};

type Hash = [u8; 32];

const RECORD_DOMAIN: &[u8] = b"mycelix-civ013-witness-record-v1\0";
const COMMIT_DOMAIN: &[u8] = b"mycelix-civ013-commit-marker-v1\0";
const RECEIPT_DOMAIN: &[u8] = b"TEST-ONLY-NOT-A-WITNESS-SIGNATURE-civ013-v1\0";
const SUPPORTED_PROTOCOL_VERSION: u16 = 1;
const DEFAULT_POLICY_VERSION: &str = "policy-v1";

#[derive(Clone, Debug, Eq, PartialEq)]
struct Record {
    protocol_version: u16,
    generation: u64,
    log_id: String,
    policy_version: String,
    anchor_digest: Hash,
    receipt_sequence: u64,
    receipt_digest: Option<Hash>,
    previous_record_digest: Option<Hash>,
    digest: Hash,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CommitMarker {
    generation: u64,
    record_digest: Hash,
    digest: Hash,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct DiskSnapshot {
    records: Vec<Record>,
    markers: Vec<CommitMarker>,
    quarantined_orphans: Vec<Record>,
    fork_evidence: Vec<ForkEvidence>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct IndependentAnchor {
    log_id: String,
    generation: u64,
    record_digest: Hash,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ForkEvidence {
    tree_size: u64,
    first_root: Hash,
    conflicting_root: Hash,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct WitnessReceipt {
    witness_id: String,
    generation: u64,
    record_digest: Hash,
    fixture_authenticator: Hash,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FaultPoint {
    BeforeDurableWrite,
    AfterRecordSync,
    AfterCommitMarkerSync,
    AfterExternalAnchorAdvance,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Failure {
    EmptyLogId,
    InvalidReceiptTailShape,
    ReceiptRollback,
    StalePredecessor,
    GenerationOverflow,
    JournalCorrupt,
    ExternalAnchorMismatch,
    RollbackDetected,
    PendingExternalAnchor,
    ExternalAnchorGap,
    InjectedCrash(FaultPoint),
    InvalidForkEvidence,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct WitnessModel {
    witness_id: String,
    disk: DiskSnapshot,
    external: IndependentAnchor,
}

fn hash_bytes(bytes: &[u8]) -> Hash {
    Sha256::digest(bytes).into()
}

fn encode_field(out: &mut Vec<u8>, value: &[u8]) {
    out.extend_from_slice(&(value.len() as u64).to_be_bytes());
    out.extend_from_slice(value);
}

fn option_hash(out: &mut Vec<u8>, value: Option<Hash>) {
    match value {
        Some(value) => {
            out.push(1);
            out.extend_from_slice(&value);
        }
        None => out.push(0),
    }
}

fn record_digest(
    protocol_version: u16,
    generation: u64,
    log_id: &str,
    policy_version: &str,
    anchor_digest: Hash,
    receipt_sequence: u64,
    receipt_digest: Option<Hash>,
    previous_record_digest: Option<Hash>,
) -> Hash {
    let mut encoded = Vec::new();
    encoded.extend_from_slice(RECORD_DOMAIN);
    encoded.extend_from_slice(&protocol_version.to_be_bytes());
    encoded.extend_from_slice(&generation.to_be_bytes());
    encode_field(&mut encoded, log_id.as_bytes());
    encode_field(&mut encoded, policy_version.as_bytes());
    encoded.extend_from_slice(&anchor_digest);
    encoded.extend_from_slice(&receipt_sequence.to_be_bytes());
    option_hash(&mut encoded, receipt_digest);
    option_hash(&mut encoded, previous_record_digest);
    hash_bytes(&encoded)
}

fn make_record(
    generation: u64,
    log_id: &str,
    policy_version: &str,
    anchor_digest: Hash,
    receipt_sequence: u64,
    receipt_digest: Option<Hash>,
    previous_record_digest: Option<Hash>,
) -> Record {
    let digest = record_digest(
        SUPPORTED_PROTOCOL_VERSION,
        generation,
        log_id,
        policy_version,
        anchor_digest,
        receipt_sequence,
        receipt_digest,
        previous_record_digest,
    );
    Record {
        protocol_version: SUPPORTED_PROTOCOL_VERSION,
        generation,
        log_id: log_id.to_owned(),
        policy_version: policy_version.to_owned(),
        anchor_digest,
        receipt_sequence,
        receipt_digest,
        previous_record_digest,
        digest,
    }
}

fn marker_digest(generation: u64, record_digest: Hash) -> Hash {
    let mut encoded = Vec::new();
    encoded.extend_from_slice(COMMIT_DOMAIN);
    encoded.extend_from_slice(&generation.to_be_bytes());
    encoded.extend_from_slice(&record_digest);
    hash_bytes(&encoded)
}

fn make_marker(record: &Record) -> CommitMarker {
    CommitMarker {
        generation: record.generation,
        record_digest: record.digest,
        digest: marker_digest(record.generation, record.digest),
    }
}

fn validate_receipt_tail(sequence: u64, digest: Option<Hash>) -> Result<(), Failure> {
    if (sequence == 0) != digest.is_none() {
        return Err(Failure::InvalidReceiptTailShape);
    }
    Ok(())
}

impl WitnessModel {
    fn bootstrap(
        witness_id: &str,
        log_id: &str,
        trusted_anchor_digest: Hash,
        receipt_sequence: u64,
        receipt_digest: Option<Hash>,
    ) -> Result<Self, Failure> {
        if log_id.is_empty() {
            return Err(Failure::EmptyLogId);
        }
        validate_receipt_tail(receipt_sequence, receipt_digest)?;
        let initial = make_record(
            1,
            log_id,
            DEFAULT_POLICY_VERSION,
            trusted_anchor_digest,
            receipt_sequence,
            receipt_digest,
            None,
        );
        let marker = make_marker(&initial);
        let disk = DiskSnapshot {
            records: vec![initial.clone()],
            markers: vec![marker],
            ..DiskSnapshot::default()
        };
        let external = IndependentAnchor {
            log_id: log_id.to_owned(),
            generation: initial.generation,
            record_digest: initial.digest,
        };
        Ok(Self {
            witness_id: witness_id.to_owned(),
            disk,
            external,
        })
    }

    /// Validate the committed prefix. Any record without a durable commit
    /// marker is quarantined, not treated as accepted state.
    fn recover_disk(disk: &mut DiskSnapshot) -> Result<Record, Failure> {
        if disk.markers.is_empty() || disk.markers.len() > disk.records.len() {
            return Err(Failure::JournalCorrupt);
        }

        let mut previous: Option<Record> = None;
        for index in 0..disk.markers.len() {
            let record = &disk.records[index];
            let marker = &disk.markers[index];
            let expected_generation =
                u64::try_from(index).map_err(|_| Failure::JournalCorrupt)? + 1;
            let expected_previous = previous.as_ref().map(|value| value.digest);
            if record.protocol_version != SUPPORTED_PROTOCOL_VERSION
                || record.policy_version.is_empty()
                || record.generation != expected_generation
                || record.previous_record_digest != expected_previous
                || record.digest
                    != record_digest(
                        record.protocol_version,
                        record.generation,
                        &record.log_id,
                        &record.policy_version,
                        record.anchor_digest,
                        record.receipt_sequence,
                        record.receipt_digest,
                        record.previous_record_digest,
                    )
                || marker.generation != record.generation
                || marker.record_digest != record.digest
                || marker.digest != marker_digest(record.generation, record.digest)
                || previous
                    .as_ref()
                    .is_some_and(|prior| prior.log_id != record.log_id)
                || validate_receipt_tail(record.receipt_sequence, record.receipt_digest).is_err()
            {
                return Err(Failure::JournalCorrupt);
            }
            previous = Some(record.clone());
        }

        if disk.records.len() > disk.markers.len() {
            let orphans = disk.records.split_off(disk.markers.len());
            disk.quarantined_orphans.extend(orphans);
        }
        previous.ok_or(Failure::JournalCorrupt)
    }

    fn recover(&mut self) -> Result<Record, Failure> {
        let accepted = Self::recover_disk(&mut self.disk)?;
        if self.external.log_id != accepted.log_id {
            return Err(Failure::ExternalAnchorMismatch);
        }
        if self.external.generation > accepted.generation {
            return Err(Failure::RollbackDetected);
        }
        if self.external.generation == accepted.generation {
            if self.external.record_digest != accepted.digest {
                return Err(Failure::ExternalAnchorMismatch);
            }
            return Ok(accepted);
        }
        Err(Failure::PendingExternalAnchor)
    }

    /// A local commit may have survived while the separate anti-rollback anchor
    /// did not advance. Reconcile only one committed successor whose predecessor
    /// equals the currently retained external anchor.
    fn reconcile_external_anchor(&mut self) -> Result<(), Failure> {
        let accepted = Self::recover_disk(&mut self.disk)?;
        if self.external.log_id != accepted.log_id {
            return Err(Failure::ExternalAnchorMismatch);
        }
        if accepted.generation == self.external.generation {
            return if accepted.digest == self.external.record_digest {
                Ok(())
            } else {
                Err(Failure::ExternalAnchorMismatch)
            };
        }
        let expected_next = self
            .external
            .generation
            .checked_add(1)
            .ok_or(Failure::GenerationOverflow)?;
        if accepted.generation != expected_next
            || accepted.previous_record_digest != Some(self.external.record_digest)
        {
            return Err(Failure::ExternalAnchorGap);
        }
        self.external.generation = accepted.generation;
        self.external.record_digest = accepted.digest;
        Ok(())
    }

    fn fixture_receipt(&self, record: &Record) -> WitnessReceipt {
        let mut encoded = Vec::new();
        encoded.extend_from_slice(RECEIPT_DOMAIN);
        encode_field(&mut encoded, self.witness_id.as_bytes());
        encoded.extend_from_slice(&record.generation.to_be_bytes());
        encoded.extend_from_slice(&record.digest);
        WitnessReceipt {
            witness_id: self.witness_id.clone(),
            generation: record.generation,
            record_digest: record.digest,
            fixture_authenticator: hash_bytes(&encoded),
        }
    }

    fn advance(
        &mut self,
        expected_generation: u64,
        expected_record_digest: Hash,
        proposed_anchor_digest: Hash,
        receipt_sequence: u64,
        receipt_digest: Option<Hash>,
        fault: Option<FaultPoint>,
    ) -> Result<WitnessReceipt, Failure> {
        validate_receipt_tail(receipt_sequence, receipt_digest)?;
        let current = self.recover()?;

        // A retry after an uncertain response may return a receipt only when
        // both the committed record and the independent anchor confirm the
        // identical already-committed transition.
        if current.generation == expected_generation.saturating_add(1)
            && current.previous_record_digest == Some(expected_record_digest)
            && current.anchor_digest == proposed_anchor_digest
            && current.receipt_sequence == receipt_sequence
            && current.receipt_digest == receipt_digest
        {
            return Ok(self.fixture_receipt(&current));
        }

        if current.generation != expected_generation
            || current.digest != expected_record_digest
        {
            return Err(Failure::StalePredecessor);
        }
        if receipt_sequence < current.receipt_sequence {
            return Err(Failure::ReceiptRollback);
        }
        let generation = current
            .generation
            .checked_add(1)
            .ok_or(Failure::GenerationOverflow)?;
        let candidate = make_record(
            generation,
            &current.log_id,
            &current.policy_version,
            proposed_anchor_digest,
            receipt_sequence,
            receipt_digest,
            Some(current.digest),
        );

        if fault == Some(FaultPoint::BeforeDurableWrite) {
            return Err(Failure::InjectedCrash(FaultPoint::BeforeDurableWrite));
        }
        // Boundary 1: record bytes are durable, but no commit marker exists.
        self.disk.records.push(candidate.clone());
        if fault == Some(FaultPoint::AfterRecordSync) {
            return Err(Failure::InjectedCrash(FaultPoint::AfterRecordSync));
        }
        // Boundary 2: commit marker is durable; local commit may be recovered,
        // but it is not accepted until the independent anchor advances.
        self.disk.markers.push(make_marker(&candidate));
        if fault == Some(FaultPoint::AfterCommitMarkerSync) {
            return Err(Failure::InjectedCrash(FaultPoint::AfterCommitMarkerSync));
        }
        // Boundary 3: independent rollback anchor acknowledges the record.
        self.external.generation = candidate.generation;
        self.external.record_digest = candidate.digest;
        if fault == Some(FaultPoint::AfterExternalAnchorAdvance) {
            return Err(Failure::InjectedCrash(FaultPoint::AfterExternalAnchorAdvance));
        }
        Ok(self.fixture_receipt(&candidate))
    }

    fn record_fork(
        &mut self,
        tree_size: u64,
        first_root: Hash,
        conflicting_root: Hash,
    ) -> Result<(), Failure> {
        if first_root == conflicting_root {
            return Err(Failure::InvalidForkEvidence);
        }
        self.disk.fork_evidence.push(ForkEvidence {
            tree_size,
            first_root,
            conflicting_root,
        });
        Ok(())
    }
}

fn main() {
    let initial_anchor = hash_bytes(b"trusted-bootstrap-checkpoint");
    let initial_tail = Some(hash_bytes(b"receipt-tail-4"));
    let proposed_anchor = hash_bytes(b"candidate-checkpoint-5");
    let proposed_tail = Some(hash_bytes(b"receipt-tail-5"));

    // Every simulated crash boundary must leave an unambiguous recovery path.
    for fault in [
        FaultPoint::BeforeDurableWrite,
        FaultPoint::AfterRecordSync,
        FaultPoint::AfterCommitMarkerSync,
        FaultPoint::AfterExternalAnchorAdvance,
    ] {
        let mut witness = WitnessModel::bootstrap(
            "witness-a",
            "civ-log-v1",
            initial_anchor,
            4,
            initial_tail,
        )
        .expect("trusted bootstrap");
        let initial = witness.recover().expect("recover bootstrap");
        assert!(matches!(
            witness.advance(
                initial.generation,
                initial.digest,
                proposed_anchor,
                5,
                proposed_tail,
                Some(fault),
            ),
            Err(Failure::InjectedCrash(actual)) if actual == fault
        ));

        match fault {
            FaultPoint::BeforeDurableWrite | FaultPoint::AfterRecordSync => {
                let recovered = witness.recover().expect("recover previous committed state");
                assert_eq!(recovered, initial);
                if fault == FaultPoint::AfterRecordSync {
                    assert_eq!(witness.disk.quarantined_orphans.len(), 1);
                }
                let receipt = witness
                    .advance(
                        initial.generation,
                        initial.digest,
                        proposed_anchor,
                        5,
                        proposed_tail,
                        None,
                    )
                    .expect("retry only after recovering prior committed state");
                assert_eq!(receipt.generation, initial.generation + 1);
            }
            FaultPoint::AfterCommitMarkerSync => {
                assert_eq!(witness.recover(), Err(Failure::PendingExternalAnchor));
                witness
                    .reconcile_external_anchor()
                    .expect("advance independent anchor only for exact committed successor");
                let receipt = witness
                    .advance(
                        initial.generation,
                        initial.digest,
                        proposed_anchor,
                        5,
                        proposed_tail,
                        None,
                    )
                    .expect("retry may attest only after independent anchor confirms");
                assert_eq!(receipt.generation, initial.generation + 1);
            }
            FaultPoint::AfterExternalAnchorAdvance => {
                let recovered = witness.recover().expect("external anchor confirms commit");
                assert_eq!(recovered.generation, initial.generation + 1);
                let receipt = witness
                    .advance(
                        initial.generation,
                        initial.digest,
                        proposed_anchor,
                        5,
                        proposed_tail,
                        None,
                    )
                    .expect("idempotent retry after confirmed commit");
                assert_eq!(receipt.record_digest, recovered.digest);
            }
        }
    }

    // Compare-and-advance: two proposals from one predecessor cannot both win.
    let mut raced = WitnessModel::bootstrap(
        "witness-race",
        "civ-log-v1",
        initial_anchor,
        4,
        initial_tail,
    )
    .expect("trusted bootstrap");
    let predecessor = raced.recover().expect("predecessor");
    raced
        .advance(
            predecessor.generation,
            predecessor.digest,
            proposed_anchor,
            5,
            proposed_tail,
            None,
        )
        .expect("first contender linearizes");
    assert_eq!(
        raced.advance(
            predecessor.generation,
            predecessor.digest,
            hash_bytes(b"conflicting-candidate"),
            5,
            Some(hash_bytes(b"conflicting-tail")),
            None,
        ),
        Err(Failure::StalePredecessor)
    );

    // A valid older snapshot is rollback when the separate anchor has advanced.
    let mut rollback = WitnessModel::bootstrap(
        "witness-rollback",
        "civ-log-v1",
        initial_anchor,
        4,
        initial_tail,
    )
    .expect("trusted bootstrap");
    let old_snapshot = rollback.disk.clone();
    let old_external = rollback.external.clone();
    let predecessor = rollback.recover().expect("predecessor");
    rollback
        .advance(
            predecessor.generation,
            predecessor.digest,
            proposed_anchor,
            5,
            proposed_tail,
            None,
        )
        .expect("advance witness");
    let advanced_external = rollback.external.clone();
    rollback.disk = old_snapshot;
    rollback.external = advanced_external;
    assert_ne!(old_external.generation, rollback.external.generation);
    assert_eq!(rollback.recover(), Err(Failure::RollbackDetected));

    // Fork evidence is retained independently and cannot change accepted state.
    let mut fork_store = WitnessModel::bootstrap(
        "witness-fork",
        "civ-log-v1",
        initial_anchor,
        4,
        initial_tail,
    )
    .expect("trusted bootstrap");
    let before_fork = fork_store.recover().expect("accepted state before fork");
    fork_store
        .record_fork(5, hash_bytes(b"root-a"), hash_bytes(b"root-b"))
        .expect("preserve conflicting view");
    assert_eq!(fork_store.recover(), Ok(before_fork.clone()));
    assert_eq!(fork_store.disk.fork_evidence.len(), 1);
    assert_eq!(
        fork_store.record_fork(5, hash_bytes(b"same"), hash_bytes(b"same")),
        Err(Failure::InvalidForkEvidence)
    );

    // Durable corruption and receipt rollback fail closed.
    let mut corrupt = WitnessModel::bootstrap(
        "witness-corrupt",
        "civ-log-v1",
        initial_anchor,
        4,
        initial_tail,
    )
    .expect("trusted bootstrap");
    corrupt.disk.markers[0].digest[0] ^= 1;
    assert_eq!(corrupt.recover(), Err(Failure::JournalCorrupt));
    let mut unsupported_version = WitnessModel::bootstrap(
        "witness-version",
        "civ-log-v1",
        initial_anchor,
        4,
        initial_tail,
    )
    .expect("trusted bootstrap");
    unsupported_version.disk.records[0].protocol_version = 2;
    assert_eq!(unsupported_version.recover(), Err(Failure::JournalCorrupt));
    let mut changed_policy = WitnessModel::bootstrap(
        "witness-policy",
        "civ-log-v1",
        initial_anchor,
        4,
        initial_tail,
    )
    .expect("trusted bootstrap");
    changed_policy.disk.records[0].policy_version = "policy-v2".to_owned();
    assert_eq!(changed_policy.recover(), Err(Failure::JournalCorrupt));
    let mut receipt_rollback = WitnessModel::bootstrap(
        "witness-receipt-rollback",
        "civ-log-v1",
        initial_anchor,
        4,
        initial_tail,
    )
    .expect("trusted bootstrap");
    let accepted = receipt_rollback.recover().expect("accepted state");
    assert_eq!(
        receipt_rollback.advance(
            accepted.generation,
            accepted.digest,
            proposed_anchor,
            3,
            Some(hash_bytes(b"older-receipt")),
            None,
        ),
        Err(Failure::ReceiptRollback)
    );

    assert_eq!(
        WitnessModel::bootstrap("bad-witness", "", initial_anchor, 4, initial_tail),
        Err(Failure::EmptyLogId)
    );
    println!("SYM-CIV-013 PASS: crash, stale-writer, and external rollback model cases exercised.");
    println!("Claim ceiling: simulated persistence; no real storage or signatures.");
}
