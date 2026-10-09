//! SYM-CIV-011: fail-closed local checkpoint acceptance.
//!
//! This smoke joins RFC 9162-shaped consistency proof verification to a local
//! monotonic acceptance state machine. Checkpoints are model inputs: this
//! executable does NOT authenticate signatures or connect to CIV-009's key
//! registry. An external caller must establish authenticity before invoking a
//! real acceptance service. The trusted bootstrap checkpoint is explicit.

use sha2::{Digest as ShaDigest, Sha256};
use std::collections::BTreeMap;

type Hash = [u8; 32];

const CHECKPOINT_ID_DOMAIN: &[u8] = b"mycelix-civ-accepted-checkpoint-id-v1\0";
const ACCEPTANCE_EVENT_DOMAIN: &[u8] = b"mycelix-civ-checkpoint-acceptance-event-v1\0";

#[derive(Clone, Debug, Eq, PartialEq)]
struct Checkpoint {
    log_id: String,
    tree_size: usize,
    root_hash: Hash,
    timestamp_epoch: u64,
    policy_version: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AcceptanceEvent {
    sequence: u64,
    previous_event_digest: Option<Hash>,
    prior_checkpoint_id: Option<Hash>,
    candidate_checkpoint_id: Hash,
    outcome: AttemptOutcome,
    reason: String,
    event_digest: Hash,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AttemptOutcome {
    TrustedAnchorInstalled,
    AcceptedGrowth,
    AcceptedRefresh,
    IdempotentReplay,
    Rejected,
    ForkDetected,
}

impl AttemptOutcome {
    fn id(self) -> &'static [u8] {
        match self {
            Self::TrustedAnchorInstalled => b"trusted-anchor-installed",
            Self::AcceptedGrowth => b"accepted-growth",
            Self::AcceptedRefresh => b"accepted-refresh",
            Self::IdempotentReplay => b"idempotent-replay",
            Self::Rejected => b"rejected",
            Self::ForkDetected => b"fork-detected",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AcceptanceDecision {
    AcceptedGrowth,
    AcceptedRefresh,
    IdempotentReplay,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ForkEvidence {
    log_id: String,
    tree_size: usize,
    first_checkpoint: Checkpoint,
    conflicting_checkpoint: Checkpoint,
    acceptance_event_sequence: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct CheckpointStore {
    current: BTreeMap<String, Checkpoint>,
    events: Vec<AcceptanceEvent>,
    forks: Vec<ForkEvidence>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AcceptanceFailure {
    EmptyLogId,
    AnchorAlreadyExists,
    UnanchoredLog,
    FutureCheckpoint,
    StaleCheckpoint,
    RollbackDetected,
    ForkDetected,
    TimestampRegression,
    MetadataConflict,
    MissingConsistencyProof,
    InvalidConsistencyProof,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProofFailure {
    InvalidTreeSize,
    InvalidProof,
    WrongRoot,
}

fn encode_field(output: &mut Vec<u8>, field: &[u8]) {
    output.extend_from_slice(&(field.len() as u64).to_be_bytes());
    output.extend_from_slice(field);
}

fn checkpoint_id(checkpoint: &Checkpoint) -> Hash {
    let mut encoded = Vec::new();
    encoded.extend_from_slice(CHECKPOINT_ID_DOMAIN);
    encode_field(&mut encoded, checkpoint.log_id.as_bytes());
    encoded.extend_from_slice(&(checkpoint.tree_size as u64).to_be_bytes());
    encoded.extend_from_slice(&checkpoint.root_hash);
    encoded.extend_from_slice(&checkpoint.timestamp_epoch.to_be_bytes());
    encode_field(&mut encoded, checkpoint.policy_version.as_bytes());
    Sha256::digest(encoded).into()
}

impl CheckpointStore {
    fn append_event(
        &mut self,
        prior_checkpoint_id: Option<Hash>,
        candidate: &Checkpoint,
        outcome: AttemptOutcome,
        reason: &str,
    ) -> u64 {
        let sequence = u64::try_from(self.events.len())
            .expect("acceptance event count fits u64")
            .checked_add(1)
            .expect("acceptance sequence cannot wrap");
        let previous_event_digest = self.events.last().map(|event| event.event_digest);
        let candidate_checkpoint_id = checkpoint_id(candidate);
        let mut encoded = Vec::new();
        encoded.extend_from_slice(ACCEPTANCE_EVENT_DOMAIN);
        encoded.extend_from_slice(&sequence.to_be_bytes());
        match previous_event_digest {
            Some(digest) => {
                encoded.push(1);
                encoded.extend_from_slice(&digest);
            }
            None => encoded.push(0),
        }
        match prior_checkpoint_id {
            Some(id) => {
                encoded.push(1);
                encoded.extend_from_slice(&id);
            }
            None => encoded.push(0),
        }
        encoded.extend_from_slice(&candidate_checkpoint_id);
        encode_field(&mut encoded, outcome.id());
        encode_field(&mut encoded, reason.as_bytes());
        let event_digest = Sha256::digest(encoded).into();
        self.events.push(AcceptanceEvent {
            sequence,
            previous_event_digest,
            prior_checkpoint_id,
            candidate_checkpoint_id,
            outcome,
            reason: reason.to_owned(),
            event_digest,
        });
        sequence
    }

    /// Install an explicit out-of-band trust anchor. This is not inferred from
    /// the first checkpoint observed on the wire.
    fn seed_trusted(
        &mut self,
        checkpoint: Checkpoint,
    ) -> Result<(), AcceptanceFailure> {
        if checkpoint.log_id.is_empty() {
            self.append_event(
                None,
                &checkpoint,
                AttemptOutcome::Rejected,
                "empty-log-id",
            );
            return Err(AcceptanceFailure::EmptyLogId);
        }
        if self.current.contains_key(&checkpoint.log_id) {
            let prior = self.current.get(&checkpoint.log_id).map(checkpoint_id);
            self.append_event(
                prior,
                &checkpoint,
                AttemptOutcome::Rejected,
                "trust-anchor-already-installed",
            );
            return Err(AcceptanceFailure::AnchorAlreadyExists);
        }
        let log_id = checkpoint.log_id.clone();
        self.append_event(
            None,
            &checkpoint,
            AttemptOutcome::TrustedAnchorInstalled,
            "explicit-trusted-bootstrap",
        );
        self.current.insert(log_id, checkpoint);
        Ok(())
    }

    fn reject(
        &mut self,
        prior: Option<&Checkpoint>,
        candidate: &Checkpoint,
        failure: AcceptanceFailure,
        reason: &str,
    ) -> Result<AcceptanceDecision, AcceptanceFailure> {
        self.append_event(
            prior.map(checkpoint_id),
            candidate,
            AttemptOutcome::Rejected,
            reason,
        );
        Err(failure)
    }

    fn accept(
        &mut self,
        candidate: Checkpoint,
        consistency_proof: Option<&[Hash]>,
        now_epoch: u64,
        max_age_epochs: u64,
    ) -> Result<AcceptanceDecision, AcceptanceFailure> {
        if candidate.log_id.is_empty() {
            return self.reject(
                None,
                &candidate,
                AcceptanceFailure::EmptyLogId,
                "empty-log-id",
            );
        }

        let Some(previous) = self.current.get(&candidate.log_id).cloned() else {
            return self.reject(
                None,
                &candidate,
                AcceptanceFailure::UnanchoredLog,
                "no-explicit-trust-anchor",
            );
        };

        if candidate.tree_size < previous.tree_size {
            return self.reject(
                Some(&previous),
                &candidate,
                AcceptanceFailure::RollbackDetected,
                "tree-size-rollback",
            );
        }

        // Conflicting same-size roots are retained before timestamp policy is
        // considered; an older or future timestamp must not choose a winner.
        if candidate.tree_size == previous.tree_size
            && candidate.root_hash != previous.root_hash
        {
            let sequence = self.append_event(
                Some(checkpoint_id(&previous)),
                &candidate,
                AttemptOutcome::ForkDetected,
                "same-log-same-size-conflicting-root",
            );
            self.forks.push(ForkEvidence {
                log_id: candidate.log_id.clone(),
                tree_size: candidate.tree_size,
                first_checkpoint: previous,
                conflicting_checkpoint: candidate,
                acceptance_event_sequence: sequence,
            });
            return Err(AcceptanceFailure::ForkDetected);
        }

        if candidate.timestamp_epoch > now_epoch {
            return self.reject(
                Some(&previous),
                &candidate,
                AcceptanceFailure::FutureCheckpoint,
                "checkpoint-from-future",
            );
        }
        if now_epoch - candidate.timestamp_epoch > max_age_epochs {
            return self.reject(
                Some(&previous),
                &candidate,
                AcceptanceFailure::StaleCheckpoint,
                "checkpoint-exceeds-freshness-window",
            );
        }
        if candidate.timestamp_epoch < previous.timestamp_epoch {
            return self.reject(
                Some(&previous),
                &candidate,
                AcceptanceFailure::TimestampRegression,
                "timestamp-regression",
            );
        }

        if candidate.tree_size == previous.tree_size {
            if candidate.timestamp_epoch == previous.timestamp_epoch
                && candidate.policy_version != previous.policy_version
            {
                return self.reject(
                    Some(&previous),
                    &candidate,
                    AcceptanceFailure::MetadataConflict,
                    "same-time-metadata-conflict",
                );
            }

            if checkpoint_id(&candidate) == checkpoint_id(&previous) {
                self.append_event(
                    Some(checkpoint_id(&previous)),
                    &candidate,
                    AttemptOutcome::IdempotentReplay,
                    "exact-checkpoint-replay",
                );
                return Ok(AcceptanceDecision::IdempotentReplay);
            }

            if candidate.timestamp_epoch == previous.timestamp_epoch {
                return self.reject(
                    Some(&previous),
                    &candidate,
                    AcceptanceFailure::MetadataConflict,
                    "same-size-same-root-different-identity",
                );
            }

            self.append_event(
                Some(checkpoint_id(&previous)),
                &candidate,
                AttemptOutcome::AcceptedRefresh,
                "same-root-newer-checkpoint",
            );
            self.current.insert(candidate.log_id.clone(), candidate);
            return Ok(AcceptanceDecision::AcceptedRefresh);
        }

        let Some(proof) = consistency_proof else {
            return self.reject(
                Some(&previous),
                &candidate,
                AcceptanceFailure::MissingConsistencyProof,
                "tree-growth-requires-consistency-proof",
            );
        };
        if verify_consistency(
            previous.tree_size,
            candidate.tree_size,
            &previous.root_hash,
            &candidate.root_hash,
            proof,
        )
        .is_err()
        {
            return self.reject(
                Some(&previous),
                &candidate,
                AcceptanceFailure::InvalidConsistencyProof,
                "consistency-proof-verification-failed",
            );
        }

        self.append_event(
            Some(checkpoint_id(&previous)),
            &candidate,
            AttemptOutcome::AcceptedGrowth,
            "verified-consistency-proof",
        );
        self.current.insert(candidate.log_id.clone(), candidate);
        Ok(AcceptanceDecision::AcceptedGrowth)
    }
}

fn leaf_hash(entry: &[u8]) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update([0x00_u8]);
    hasher.update(entry);
    hasher.finalize().into()
}

fn node_hash(left: &Hash, right: &Hash) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update([0x01_u8]);
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

fn largest_power_of_two_less_than(n: usize) -> usize {
    debug_assert!(n > 1);
    let mut power = 1_usize;
    while power * 2 < n {
        power *= 2;
    }
    power
}

fn merkle_tree_hash(entries: &[Vec<u8>]) -> Hash {
    match entries.len() {
        0 => Sha256::digest([]).into(),
        1 => leaf_hash(&entries[0]),
        n => {
            let k = largest_power_of_two_less_than(n);
            node_hash(
                &merkle_tree_hash(&entries[..k]),
                &merkle_tree_hash(&entries[k..]),
            )
        }
    }
}

fn consistency_subproof(
    old_size: usize,
    entries: &[Vec<u8>],
    complete_subtree: bool,
) -> Vec<Hash> {
    let n = entries.len();
    if old_size == n {
        return if complete_subtree {
            Vec::new()
        } else {
            vec![merkle_tree_hash(entries)]
        };
    }

    let k = largest_power_of_two_less_than(n);
    if old_size <= k {
        let mut proof = consistency_subproof(old_size, &entries[..k], complete_subtree);
        proof.push(merkle_tree_hash(&entries[k..]));
        proof
    } else {
        let mut proof = consistency_subproof(old_size - k, &entries[k..], false);
        proof.push(merkle_tree_hash(&entries[..k]));
        proof
    }
}

fn generate_consistency_proof(
    old_size: usize,
    entries: &[Vec<u8>],
) -> Result<Vec<Hash>, ProofFailure> {
    if old_size > entries.len() {
        return Err(ProofFailure::InvalidTreeSize);
    }
    if old_size == 0 || old_size == entries.len() {
        return Ok(Vec::new());
    }
    Ok(consistency_subproof(old_size, entries, true))
}

fn verify_consistency(
    old_size: usize,
    new_size: usize,
    old_root: &Hash,
    new_root: &Hash,
    consistency_proof: &[Hash],
) -> Result<(), ProofFailure> {
    if old_size > new_size {
        return Err(ProofFailure::InvalidTreeSize);
    }

    let empty_root: Hash = Sha256::digest([]).into();
    if old_size == 0 {
        if old_root != &empty_root || !consistency_proof.is_empty() {
            return Err(ProofFailure::InvalidProof);
        }
        return Ok(());
    }

    if old_size == new_size {
        if !consistency_proof.is_empty() {
            return Err(ProofFailure::InvalidProof);
        }
        return if old_root == new_root {
            Ok(())
        } else {
            Err(ProofFailure::WrongRoot)
        };
    }

    if new_size == 0 || consistency_proof.is_empty() {
        return Err(ProofFailure::InvalidProof);
    }

    let mut proof_nodes = Vec::with_capacity(consistency_proof.len() + 1);
    if old_size.is_power_of_two() {
        proof_nodes.push(*old_root);
    }
    proof_nodes.extend_from_slice(consistency_proof);

    let mut fn_index = old_size - 1;
    let mut sn_index = new_size - 1;
    while (fn_index & 1) == 1 {
        fn_index >>= 1;
        sn_index >>= 1;
    }

    let Some(first_node) = proof_nodes.first() else {
        return Err(ProofFailure::InvalidProof);
    };
    let mut first_root = *first_node;
    let mut second_root = *first_node;

    for proof_node in proof_nodes.iter().skip(1) {
        if sn_index == 0 {
            return Err(ProofFailure::InvalidProof);
        }

        if (fn_index & 1) == 1 || fn_index == sn_index {
            first_root = node_hash(proof_node, &first_root);
            second_root = node_hash(proof_node, &second_root);

            if (fn_index & 1) == 0 {
                while fn_index != 0 && (fn_index & 1) == 0 {
                    fn_index >>= 1;
                    sn_index >>= 1;
                }
            }
        } else {
            second_root = node_hash(&second_root, proof_node);
        }

        fn_index >>= 1;
        sn_index >>= 1;
    }

    if sn_index != 0 || first_root != *old_root || second_root != *new_root {
        return Err(ProofFailure::WrongRoot);
    }
    Ok(())
}

fn canonical_event(sequence: u64, action: &str, effective_epoch: u64) -> Vec<u8> {
    let mut event = Vec::new();
    event.extend_from_slice(b"mycelix-civ-governance-entry-v1\0");
    event.extend_from_slice(&sequence.to_be_bytes());
    encode_field(&mut event, b"authority-civ-1");
    encode_field(&mut event, action.as_bytes());
    encode_field(&mut event, b"deployment-v1");
    encode_field(&mut event, b"critical-scope-v1");
    encode_field(&mut event, b"reviewed");
    encode_field(&mut event, b"policy-v1");
    event.extend_from_slice(&effective_epoch.to_be_bytes());
    event
}

fn sample_entries(count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| {
            canonical_event(
                i as u64 + 1,
                if i % 2 == 0 { "authorize" } else { "suspend" },
                100 + i as u64,
            )
        })
        .collect()
}

fn checkpoint(log_id: &str, entries: &[Vec<u8>], timestamp_epoch: u64, policy: &str) -> Checkpoint {
    Checkpoint {
        log_id: log_id.to_owned(),
        tree_size: entries.len(),
        root_hash: merkle_tree_hash(entries),
        timestamp_epoch,
        policy_version: policy.to_owned(),
    }
}

fn main() {
    // Exhaust all old/new size pairs across powers of two and irregular trees.
    let mut valid_transitions = 0_usize;
    let mut rejected_tampered_proofs = 0_usize;
    for new_size in 2..=18 {
        let entries = sample_entries(new_size);
        for old_size in 1..new_size {
            let old = checkpoint("civ-log-v1", &entries[..old_size], 100, "policy-v1");
            let new = checkpoint("civ-log-v1", &entries, 101, "policy-v1");
            let proof = generate_consistency_proof(old_size, &entries).expect("valid size pair");

            let mut store = CheckpointStore::default();
            store.seed_trusted(old.clone()).expect("install explicit anchor");
            assert_eq!(
                store.accept(new.clone(), Some(&proof), 102, 10),
                Ok(AcceptanceDecision::AcceptedGrowth),
                "valid transition {old_size}->{new_size}"
            );
            assert_eq!(store.current.get("civ-log-v1"), Some(&new));
            valid_transitions += 1;

            let mut tampered = proof.clone();
            assert!(!tampered.is_empty(), "growth proof should be non-empty");
            tampered[0][0] ^= 0x40;
            let mut rejecting_store = CheckpointStore::default();
            rejecting_store
                .seed_trusted(old.clone())
                .expect("install explicit anchor");
            assert_eq!(
                rejecting_store.accept(new, Some(&tampered), 102, 10),
                Err(AcceptanceFailure::InvalidConsistencyProof),
                "tampered transition {old_size}->{new_size}"
            );
            assert_eq!(rejecting_store.current.get("civ-log-v1"), Some(&old));
            rejected_tampered_proofs += 1;
        }
    }

    // Growth without a proof is rejected and cannot advance the accepted head.
    let entries = sample_entries(7);
    let old = checkpoint("civ-log-v1", &entries[..3], 100, "policy-v1");
    let newer = checkpoint("civ-log-v1", &entries, 101, "policy-v1");
    let mut store = CheckpointStore::default();
    store.seed_trusted(old.clone()).expect("install trust anchor");
    assert_eq!(
        store.accept(newer.clone(), None, 102, 10),
        Err(AcceptanceFailure::MissingConsistencyProof)
    );
    assert_eq!(store.current.get("civ-log-v1"), Some(&old));

    // Rollback remains rejected even when the smaller head is otherwise fresh.
    let smaller = checkpoint("civ-log-v1", &entries[..2], 101, "policy-v1");
    assert_eq!(
        store.accept(smaller, None, 102, 10),
        Err(AcceptanceFailure::RollbackDetected)
    );
    assert_eq!(store.current.get("civ-log-v1"), Some(&old));

    // Conflicting same-size roots are retained as evidence; no root is selected.
    let mut fork_entries = entries.clone();
    fork_entries[2].push(0xA5);
    let fork = checkpoint("civ-log-v1", &fork_entries[..3], 99, "policy-v1");
    assert_ne!(old.root_hash, fork.root_hash);
    assert_eq!(
        store.accept(fork.clone(), None, 102, 10),
        Err(AcceptanceFailure::ForkDetected)
    );
    assert_eq!(store.current.get("civ-log-v1"), Some(&old));
    assert_eq!(store.forks.len(), 1);
    assert_eq!(store.forks[0].first_checkpoint, old);
    assert_eq!(store.forks[0].conflicting_checkpoint, fork);
    assert_eq!(store.events.last().unwrap().outcome, AttemptOutcome::ForkDetected);

    // Exact replay is idempotent; a same-time metadata substitution is not.
    let replay_anchor = checkpoint("replay-log", &sample_entries(4), 100, "policy-v1");
    let mut replay_store = CheckpointStore::default();
    replay_store
        .seed_trusted(replay_anchor.clone())
        .expect("install trust anchor");
    assert_eq!(
        replay_store.accept(replay_anchor.clone(), None, 102, 10),
        Ok(AcceptanceDecision::IdempotentReplay)
    );
    let metadata_substitution = Checkpoint {
        policy_version: "policy-substituted".to_owned(),
        ..replay_anchor.clone()
    };
    assert_eq!(
        replay_store.accept(metadata_substitution, None, 102, 10),
        Err(AcceptanceFailure::MetadataConflict)
    );

    // Future/stale timestamps and regressions all fail without state mutation.
    let future = Checkpoint {
        timestamp_epoch: 103,
        ..replay_anchor.clone()
    };
    assert_eq!(
        replay_store.accept(future, None, 102, 10),
        Err(AcceptanceFailure::FutureCheckpoint)
    );
    let stale = Checkpoint {
        timestamp_epoch: 10,
        ..replay_anchor.clone()
    };
    assert_eq!(
        replay_store.accept(stale, None, 102, 10),
        Err(AcceptanceFailure::StaleCheckpoint)
    );
    let regressed_time = Checkpoint {
        timestamp_epoch: 99,
        ..replay_anchor.clone()
    };
    assert_eq!(
        replay_store.accept(regressed_time, None, 102, 10),
        Err(AcceptanceFailure::TimestampRegression)
    );
    assert_eq!(replay_store.current.get("replay-log"), Some(&replay_anchor));

    // A first observed head cannot silently become a trust anchor for a new log.
    let unanchored = checkpoint("unknown-log", &sample_entries(2), 101, "policy-v1");
    assert_eq!(
        replay_store.accept(unanchored, None, 102, 10),
        Err(AcceptanceFailure::UnanchoredLog)
    );

    // The audit receipt is monotonic and binds the previous/current identities
    // and every accepted/rejected result. Failed attempts never change current.
    for (index, event) in store.events.iter().enumerate() {
        assert_eq!(event.sequence, index as u64 + 1);
        let expected_previous_event_digest = index
            .checked_sub(1)
            .map(|previous_index| store.events[previous_index].event_digest);
        assert_eq!(event.previous_event_digest, expected_previous_event_digest);
        assert_ne!(event.candidate_checkpoint_id, [0_u8; 32]);
        assert_ne!(event.event_digest, [0_u8; 32]);
        assert!(!event.reason.is_empty());
    }
    assert_eq!(
        store.forks[0].acceptance_event_sequence,
        store.events.last().unwrap().sequence
    );

    println!(
        "SYM-CIV-011 PASS: {valid_transitions} valid growth transitions and {rejected_tampered_proofs} tampered consistency proofs exercised."
    );
    println!("Claim ceiling: RFC 9162-shaped consistency proof and local post-authentication acceptance smoke; no signature authentication, durable store, or distributed consensus.");
}
