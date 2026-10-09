//! SYM-CIV-012: independent checkpoint anchors and receipt-tail rollback smoke.
//!
//! Models a log, locally chained governance receipts, and multiple logically
//! separate witness stores. Witnesses retain a prior checkpoint and verify both
//! Merkle consistency and the exact receipt-chain suffix before attesting.
//!
//! IMPORTANT: fixture-only attestations and in-memory state are used here. This
//! executable implements no real signatures, durable storage, network gossip,
//! or production anti-rollback service.

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

type Hash = [u8; 32];

const RECEIPT_EVENT_DOMAIN: &[u8] = b"mycelix-civ012-receipt-event-v1\0";
const ANCHOR_ID_DOMAIN: &[u8] = b"mycelix-civ012-checkpoint-anchor-v1\0";
const FIXTURE_ATTESTATION_DOMAIN: &[u8] = b"TEST-ONLY-NOT-A-WITNESS-SIGNATURE-civ012-v1\0";

// Explicit verifier work bounds; transport/parser allocation limits remain an outer layer.
const MAX_RECEIPT_SUFFIX_EVENTS: usize = 4_096;
const MAX_RECEIPT_EVENT_BODY_BYTES: usize = 64 * 1024;
const MAX_RECEIPT_SUFFIX_BYTES: usize = 4 * 1024 * 1024;
// A u64-sized append-only Merkle tree needs at most 64 consistency nodes.
const MAX_CONSISTENCY_PROOF_NODES: usize = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
struct ReceiptEvent {
    sequence: u64,
    previous_digest: Option<Hash>,
    body: Vec<u8>,
    digest: Hash,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CheckpointAnchor {
    protocol_version: u16,
    log_id: String,
    tree_size: usize,
    root_hash: Hash,
    timestamp_epoch: u64,
    policy_version: String,
    receipt_sequence: u64,
    receipt_digest: Option<Hash>,
    previous_anchor_digest: Option<Hash>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ForkEvidence {
    log_id: String,
    tree_size: usize,
    first_anchor: CheckpointAnchor,
    conflicting_anchor: CheckpointAnchor,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct WitnessAttestation {
    witness_id: String,
    authority_lineage: String,
    anchor_id: Hash,
    fixture_signature: Hash,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct WitnessStore {
    witness_id: String,
    authority_lineage: String,
    current: BTreeMap<String, CheckpointAnchor>,
    forks: Vec<ForkEvidence>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Failure {
    EmptyLogId,
    UnsupportedProtocolVersion,
    InvalidReceiptTailShape,
    UnanchoredLog,
    FutureCheckpoint,
    StaleCheckpoint,
    TimestampRegression,
    TreeRollback,
    ReceiptRollback,
    ReceiptSuffixGap,
    ReceiptSuffixInvalid,
    ReceiptSuffixResourceLimit,
    ReceiptTailMismatchForCandidate,
    MissingAnchorLink,
    MissingConsistencyProof,
    InvalidConsistencyProof,
    ForkDetected,
    InsufficientWitnesses,
    DuplicateWitness,
    SharedAuthorityLineage,
    WitnessAnchorMismatch,
    InvalidFixtureAttestation,
    LocalReceiptChainInvalid,
    LocalReceiptTailTruncated,
    LocalReceiptTailMismatch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Decision {
    AnchorAdvanced,
    TreeAdvanced,
    IdempotentReplay,
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

fn option_hash(output: &mut Vec<u8>, value: Option<Hash>) {
    match value {
        Some(hash) => {
            output.push(1);
            output.extend_from_slice(&hash);
        }
        None => output.push(0),
    }
}

fn receipt_event_digest(
    sequence: u64,
    previous_digest: Option<Hash>,
    body: &[u8],
) -> Hash {
    let mut encoded = Vec::new();
    encoded.extend_from_slice(RECEIPT_EVENT_DOMAIN);
    encoded.extend_from_slice(&sequence.to_be_bytes());
    option_hash(&mut encoded, previous_digest);
    encode_field(&mut encoded, body);
    Sha256::digest(encoded).into()
}

fn append_receipt(events: &mut Vec<ReceiptEvent>, body: &[u8]) {
    let sequence = u64::try_from(events.len())
        .expect("receipt count fits u64")
        .checked_add(1)
        .expect("receipt sequence cannot wrap");
    let previous_digest = events.last().map(|event| event.digest);
    let digest = receipt_event_digest(sequence, previous_digest, body);
    events.push(ReceiptEvent {
        sequence,
        previous_digest,
        body: body.to_vec(),
        digest,
    });
}

fn verify_receipt_chain(events: &[ReceiptEvent]) -> Result<(), Failure> {
    let mut previous_digest = None;
    for (index, event) in events.iter().enumerate() {
        let expected_sequence = u64::try_from(index)
            .expect("receipt count fits u64")
            .checked_add(1)
            .expect("receipt sequence cannot wrap");
        if event.sequence != expected_sequence || event.previous_digest != previous_digest {
            return Err(Failure::LocalReceiptChainInvalid);
        }
        let expected = receipt_event_digest(
            event.sequence,
            event.previous_digest,
            &event.body,
        );
        if expected != event.digest {
            return Err(Failure::LocalReceiptChainInvalid);
        }
        previous_digest = Some(event.digest);
    }
    Ok(())
}

/// Replay the new receipts from the independently retained tail. This is a
/// data-level suffix proof, not a compact proof or a digital signature.
fn verify_receipt_suffix(
    previous_sequence: u64,
    previous_digest: Option<Hash>,
    candidate_sequence: u64,
    candidate_digest: Option<Hash>,
    suffix: &[ReceiptEvent],
) -> Result<(), Failure> {
    if suffix.len() > MAX_RECEIPT_SUFFIX_EVENTS {
        return Err(Failure::ReceiptSuffixResourceLimit);
    }
    let mut suffix_bytes = 0_usize;
    for event in suffix {
        if event.body.len() > MAX_RECEIPT_EVENT_BODY_BYTES {
            return Err(Failure::ReceiptSuffixResourceLimit);
        }
        suffix_bytes = suffix_bytes
            .checked_add(event.body.len())
            .ok_or(Failure::ReceiptSuffixResourceLimit)?;
        if suffix_bytes > MAX_RECEIPT_SUFFIX_BYTES {
            return Err(Failure::ReceiptSuffixResourceLimit);
        }
    }
    if candidate_sequence < previous_sequence {
        return Err(Failure::ReceiptRollback);
    }
    let required = candidate_sequence - previous_sequence;
    if u64::try_from(suffix.len()).ok() != Some(required) {
        return Err(Failure::ReceiptSuffixGap);
    }

    let mut expected_sequence = previous_sequence;
    let mut rolling_digest = previous_digest;
    for event in suffix {
        expected_sequence = expected_sequence
            .checked_add(1)
            .ok_or(Failure::ReceiptSuffixInvalid)?;
        if event.sequence != expected_sequence || event.previous_digest != rolling_digest {
            return Err(Failure::ReceiptSuffixInvalid);
        }
        if receipt_event_digest(
            event.sequence,
            event.previous_digest,
            &event.body,
        ) != event.digest
        {
            return Err(Failure::ReceiptSuffixInvalid);
        }
        rolling_digest = Some(event.digest);
    }

    if expected_sequence != candidate_sequence || rolling_digest != candidate_digest {
        return Err(Failure::ReceiptTailMismatchForCandidate);
    }
    Ok(())
}

fn validate_anchor_shape(anchor: &CheckpointAnchor) -> Result<(), Failure> {
    if anchor.protocol_version != 1 {
        return Err(Failure::UnsupportedProtocolVersion);
    }
    if anchor.log_id.is_empty() {
        return Err(Failure::EmptyLogId);
    }
    if (anchor.receipt_sequence == 0) != anchor.receipt_digest.is_none() {
        return Err(Failure::InvalidReceiptTailShape);
    }
    Ok(())
}

fn anchor_identity(anchor: &CheckpointAnchor) -> Hash {
    let mut encoded = Vec::new();
    encoded.extend_from_slice(ANCHOR_ID_DOMAIN);
    encoded.extend_from_slice(&anchor.protocol_version.to_be_bytes());
    encode_field(&mut encoded, anchor.log_id.as_bytes());
    encoded.extend_from_slice(&(anchor.tree_size as u64).to_be_bytes());
    encoded.extend_from_slice(&anchor.root_hash);
    encoded.extend_from_slice(&anchor.timestamp_epoch.to_be_bytes());
    encode_field(&mut encoded, anchor.policy_version.as_bytes());
    encoded.extend_from_slice(&anchor.receipt_sequence.to_be_bytes());
    option_hash(&mut encoded, anchor.receipt_digest);
    option_hash(&mut encoded, anchor.previous_anchor_digest);
    Sha256::digest(encoded).into()
}

impl WitnessStore {
    fn new(witness_id: &str, authority_lineage: &str) -> Self {
        Self {
            witness_id: witness_id.to_owned(),
            authority_lineage: authority_lineage.to_owned(),
            current: BTreeMap::new(),
            forks: Vec::new(),
        }
    }

    /// Bootstrap is explicit and out of band. The observe path never silently
    /// trusts the first checkpoint it receives.
    fn seed_trusted(&mut self, anchor: CheckpointAnchor) -> Result<(), Failure> {
        validate_anchor_shape(&anchor)?;
        if self.current.contains_key(&anchor.log_id) {
            return Err(Failure::MissingAnchorLink);
        }
        self.current.insert(anchor.log_id.clone(), anchor);
        Ok(())
    }

    fn observe(
        &mut self,
        candidate: CheckpointAnchor,
        consistency_proof: Option<&[Hash]>,
        receipt_suffix: &[ReceiptEvent],
        now_epoch: u64,
        max_age_epochs: u64,
    ) -> Result<(Decision, WitnessAttestation), Failure> {
        validate_anchor_shape(&candidate)?;
        let Some(previous) = self.current.get(&candidate.log_id).cloned() else {
            return Err(Failure::UnanchoredLog);
        };

        // Preserve same-size root conflicts before timestamp or freshness
        // policy so those fields cannot silently choose the preferred view.
        if candidate.tree_size == previous.tree_size
            && candidate.root_hash != previous.root_hash
        {
            self.forks.push(ForkEvidence {
                log_id: candidate.log_id.clone(),
                tree_size: candidate.tree_size,
                first_anchor: previous,
                conflicting_anchor: candidate,
            });
            return Err(Failure::ForkDetected);
        }

        if anchor_identity(&candidate) == anchor_identity(&previous) {
            let attestation = fixture_attestation(
                &self.witness_id,
                &self.authority_lineage,
                anchor_identity(&candidate),
            );
            return Ok((Decision::IdempotentReplay, attestation));
        }

        if candidate.timestamp_epoch > now_epoch {
            return Err(Failure::FutureCheckpoint);
        }
        if now_epoch - candidate.timestamp_epoch > max_age_epochs {
            return Err(Failure::StaleCheckpoint);
        }
        if candidate.timestamp_epoch <= previous.timestamp_epoch {
            return Err(Failure::TimestampRegression);
        }
        if candidate.tree_size < previous.tree_size {
            return Err(Failure::TreeRollback);
        }
        if candidate.receipt_sequence < previous.receipt_sequence {
            return Err(Failure::ReceiptRollback);
        }
        if candidate.previous_anchor_digest != Some(anchor_identity(&previous)) {
            return Err(Failure::MissingAnchorLink);
        }
        verify_receipt_suffix(
            previous.receipt_sequence,
            previous.receipt_digest,
            candidate.receipt_sequence,
            candidate.receipt_digest,
            receipt_suffix,
        )?;

        let decision = if candidate.tree_size > previous.tree_size {
            let Some(proof) = consistency_proof else {
                return Err(Failure::MissingConsistencyProof);
            };
            verify_consistency(
                previous.tree_size,
                candidate.tree_size,
                &previous.root_hash,
                &candidate.root_hash,
                proof,
            )
            .map_err(|_| Failure::InvalidConsistencyProof)?;
            Decision::TreeAdvanced
        } else {
            if consistency_proof.is_some_and(|proof| !proof.is_empty()) {
                return Err(Failure::InvalidConsistencyProof);
            }
            Decision::AnchorAdvanced
        };

        let id = anchor_identity(&candidate);
        self.current.insert(candidate.log_id.clone(), candidate);
        let attestation = fixture_attestation(
            &self.witness_id,
            &self.authority_lineage,
            id,
        );
        Ok((decision, attestation))
    }
}

fn fixture_attestation(
    witness_id: &str,
    authority_lineage: &str,
    anchor_id: Hash,
) -> WitnessAttestation {
    let mut encoded = Vec::new();
    encoded.extend_from_slice(FIXTURE_ATTESTATION_DOMAIN);
    encode_field(&mut encoded, witness_id.as_bytes());
    encode_field(&mut encoded, authority_lineage.as_bytes());
    encoded.extend_from_slice(&anchor_id);
    let fixture_signature = Sha256::digest(encoded).into();
    WitnessAttestation {
        witness_id: witness_id.to_owned(),
        authority_lineage: authority_lineage.to_owned(),
        anchor_id,
        fixture_signature,
    }
}

fn verify_fixture_attestation(attestation: &WitnessAttestation) -> Result<(), Failure> {
    let expected = fixture_attestation(
        &attestation.witness_id,
        &attestation.authority_lineage,
        attestation.anchor_id,
    );
    if expected.fixture_signature != attestation.fixture_signature {
        return Err(Failure::InvalidFixtureAttestation);
    }
    Ok(())
}

fn verify_quorum(
    attestations: &[WitnessAttestation],
    expected_anchor: &CheckpointAnchor,
    minimum_witnesses: usize,
) -> Result<(), Failure> {
    if minimum_witnesses == 0 || attestations.len() < minimum_witnesses {
        return Err(Failure::InsufficientWitnesses);
    }
    let expected_id = anchor_identity(expected_anchor);
    let mut witnesses = BTreeSet::new();
    let mut lineages = BTreeSet::new();
    for attestation in attestations {
        verify_fixture_attestation(attestation)?;
        if attestation.anchor_id != expected_id {
            return Err(Failure::WitnessAnchorMismatch);
        }
        if !witnesses.insert(attestation.witness_id.as_str()) {
            return Err(Failure::DuplicateWitness);
        }
        if !lineages.insert(attestation.authority_lineage.as_str()) {
            return Err(Failure::SharedAuthorityLineage);
        }
    }
    Ok(())
}

/// Compare a restored local receipt chain with an independently retained
/// anchor. A valid prefix below the anchor is rollback/truncation.
fn reconcile_local_chain(
    events: &[ReceiptEvent],
    anchor: &CheckpointAnchor,
) -> Result<(), Failure> {
    verify_receipt_chain(events)?;
    if anchor.receipt_sequence == 0 {
        return if anchor.receipt_digest.is_none() {
            Ok(())
        } else {
            Err(Failure::LocalReceiptTailMismatch)
        };
    }
    if u64::try_from(events.len()).unwrap_or(u64::MAX) < anchor.receipt_sequence {
        return Err(Failure::LocalReceiptTailTruncated);
    }
    let index = usize::try_from(anchor.receipt_sequence - 1)
        .map_err(|_| Failure::LocalReceiptTailMismatch)?;
    let event = events.get(index).ok_or(Failure::LocalReceiptTailTruncated)?;
    if Some(event.digest) != anchor.receipt_digest {
        return Err(Failure::LocalReceiptTailMismatch);
    }
    Ok(())
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
    proof: &[Hash],
) -> Result<(), ProofFailure> {
    if proof.len() > MAX_CONSISTENCY_PROOF_NODES {
        return Err(ProofFailure::InvalidProof);
    }
    if old_size > new_size {
        return Err(ProofFailure::InvalidTreeSize);
    }
    let empty_root: Hash = Sha256::digest([]).into();
    if old_size == 0 {
        return if old_root == &empty_root && proof.is_empty() {
            Ok(())
        } else {
            Err(ProofFailure::InvalidProof)
        };
    }
    if old_size == new_size {
        if !proof.is_empty() {
            return Err(ProofFailure::InvalidProof);
        }
        return if old_root == new_root {
            Ok(())
        } else {
            Err(ProofFailure::WrongRoot)
        };
    }
    if new_size == 0 || proof.is_empty() {
        return Err(ProofFailure::InvalidProof);
    }

    let mut proof_nodes = Vec::with_capacity(proof.len() + 1);
    if old_size.is_power_of_two() {
        proof_nodes.push(*old_root);
    }
    proof_nodes.extend_from_slice(proof);
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

fn sample_entries(count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|index| {
            let mut entry = b"mycelix-civ012-log-entry-v1\0".to_vec();
            entry.extend_from_slice(&(index as u64 + 1).to_be_bytes());
            entry
        })
        .collect()
}

fn make_anchor(
    log_id: &str,
    entries: &[Vec<u8>],
    timestamp_epoch: u64,
    policy_version: &str,
    receipt_sequence: u64,
    receipt_digest: Option<Hash>,
    previous_anchor_digest: Option<Hash>,
) -> CheckpointAnchor {
    CheckpointAnchor {
        protocol_version: 1,
        log_id: log_id.to_owned(),
        tree_size: entries.len(),
        root_hash: merkle_tree_hash(entries),
        timestamp_epoch,
        policy_version: policy_version.to_owned(),
        receipt_sequence,
        receipt_digest,
        previous_anchor_digest,
    }
}

fn receipt_suffix(
    events: &[ReceiptEvent],
    previous_sequence: u64,
    candidate_sequence: u64,
) -> Vec<ReceiptEvent> {
    let start = usize::try_from(previous_sequence).expect("sequence fits usize");
    let end = usize::try_from(candidate_sequence).expect("sequence fits usize");
    events[start..end].to_vec()
}

fn main() {
    // Build a local receipt stream. The trusted witness anchor commits to event 4.
    let mut events = Vec::new();
    for sequence in 1..=8_u64 {
        let mut body = b"civ012-governance-receipt-v1\0".to_vec();
        body.extend_from_slice(&sequence.to_be_bytes());
        if sequence % 2 == 0 {
            body.extend_from_slice(b"suspend");
        } else {
            body.extend_from_slice(b"authorize");
        }
        append_receipt(&mut events, &body);
    }
    assert_eq!(verify_receipt_chain(&events), Ok(()));
    let initial_tail = Some(events[3].digest);

    let entries = sample_entries(7);
    let previous_entries = &entries[..3];
    let previous = make_anchor(
        "civ-log-v1",
        previous_entries,
        100,
        "policy-v1",
        4,
        initial_tail,
        None,
    );
    let candidate_entries = &entries[..5];
    let candidate = make_anchor(
        "civ-log-v1",
        candidate_entries,
        101,
        "policy-v1",
        7,
        Some(events[6].digest),
        Some(anchor_identity(&previous)),
    );
    let proof = generate_consistency_proof(previous.tree_size, candidate_entries)
        .expect("consistent growth proof");
    let suffix = receipt_suffix(&events, previous.receipt_sequence, candidate.receipt_sequence);

    // Three logically separate stores start with the same explicit trust anchor.
    // Collect attestations emitted by each successful state transition; do not
    // synthesize a separate quorum detached from the witness operations.
    let mut attestations = Vec::new();
    for (id, lineage) in [
        ("witness-a", "lineage-a"),
        ("witness-b", "lineage-b"),
        ("witness-c", "lineage-c"),
    ] {
        let mut store = WitnessStore::new(id, lineage);
        store.seed_trusted(previous.clone()).expect("explicit trusted bootstrap");
        let (decision, attestation) = store
            .observe(
                candidate.clone(),
                Some(&proof),
                &suffix,
                102,
                10,
            )
            .expect("witness should validate exact checkpoint transition");
        assert_eq!(decision, Decision::TreeAdvanced);
        assert_eq!(store.current.get("civ-log-v1"), Some(&candidate));
        attestations.push(attestation);
    }
    assert_eq!(verify_quorum(&attestations, &candidate, 3), Ok(()));

    // A restarted client detects a truncated receipt tail using the independently
    // retained sequence/digest in the witness anchor.
    assert_eq!(reconcile_local_chain(&events, &candidate), Ok(()));
    assert_eq!(
        reconcile_local_chain(&events[..6], &candidate),
        Err(Failure::LocalReceiptTailTruncated)
    );
    let mut tampered_events = events.clone();
    tampered_events[5].body.push(0xFF);
    assert_eq!(
        reconcile_local_chain(&tampered_events, &candidate),
        Err(Failure::LocalReceiptChainInvalid)
    );
    let mut changed_anchored_digest = events.clone();
    changed_anchored_digest[6].digest[0] ^= 1;
    assert_eq!(
        reconcile_local_chain(&changed_anchored_digest, &candidate),
        Err(Failure::LocalReceiptChainInvalid)
    );

    // A suffix gap or tail mismatch cannot advance witness state.
    let mut gap_store = WitnessStore::new("witness-gap", "lineage-gap");
    gap_store.seed_trusted(previous.clone()).expect("seed anchor");
    assert_eq!(
        gap_store.observe(
            candidate.clone(),
            Some(&proof),
            &suffix[..2],
            102,
            10,
        ),
        Err(Failure::ReceiptSuffixGap)
    );
    assert_eq!(gap_store.current.get("civ-log-v1"), Some(&previous));

    let mut malformed_suffix = suffix.clone();
    malformed_suffix[1] = malformed_suffix[0].clone();
    let mut malformed_suffix_store =
        WitnessStore::new("witness-malformed-suffix", "lineage-malformed-suffix");
    malformed_suffix_store.seed_trusted(previous.clone()).expect("seed anchor");
    assert_eq!(
        malformed_suffix_store.observe(
            candidate.clone(),
            Some(&proof),
            &malformed_suffix,
            102,
            10,
        ),
        Err(Failure::ReceiptSuffixInvalid)
    );
    assert_eq!(
        malformed_suffix_store.current.get("civ-log-v1"),
        Some(&previous)
    );

    // Reject oversized candidate data before hashing/replaying it; failures do not advance state.
    let mut oversized_body_suffix = suffix.clone();
    oversized_body_suffix[0].body = vec![0; MAX_RECEIPT_EVENT_BODY_BYTES + 1];
    let mut bounded_store = WitnessStore::new("witness-bounded", "lineage-bounded");
    bounded_store.seed_trusted(previous.clone()).expect("seed anchor");
    assert_eq!(
        bounded_store.observe(candidate.clone(), Some(&proof), &oversized_body_suffix, 102, 10),
        Err(Failure::ReceiptSuffixResourceLimit)
    );
    assert_eq!(bounded_store.current.get("civ-log-v1"), Some(&previous));

    let excessive_count = vec![suffix[0].clone(); MAX_RECEIPT_SUFFIX_EVENTS + 1];
    assert_eq!(
        bounded_store.observe(candidate.clone(), Some(&proof), &excessive_count, 102, 10),
        Err(Failure::ReceiptSuffixResourceLimit)
    );
    assert_eq!(bounded_store.current.get("civ-log-v1"), Some(&previous));

    let mut excessive_bytes = vec![suffix[0].clone(); 65];
    for event in &mut excessive_bytes {
        event.body = vec![0; MAX_RECEIPT_EVENT_BODY_BYTES];
    }
    assert_eq!(
        bounded_store.observe(candidate.clone(), Some(&proof), &excessive_bytes, 102, 10),
        Err(Failure::ReceiptSuffixResourceLimit)
    );
    assert_eq!(bounded_store.current.get("civ-log-v1"), Some(&previous));

    // Even syntactically valid proof arrays are rejected above the u64-tree bound.
    let excessive_proof = vec![[0xAA; 32]; MAX_CONSISTENCY_PROOF_NODES + 1];
    assert_eq!(
        bounded_store.observe(candidate.clone(), Some(&excessive_proof), &suffix, 102, 10),
        Err(Failure::InvalidConsistencyProof)
    );
    assert_eq!(bounded_store.current.get("civ-log-v1"), Some(&previous));

    let wrong_tail = CheckpointAnchor {
        receipt_digest: Some(events[5].digest),
        ..candidate.clone()
    };
    let mut wrong_tail_store = WitnessStore::new("witness-tail", "lineage-tail");
    wrong_tail_store.seed_trusted(previous.clone()).expect("seed anchor");
    assert_eq!(
        wrong_tail_store.observe(
            wrong_tail,
            Some(&proof),
            &suffix,
            102,
            10,
        ),
        Err(Failure::ReceiptTailMismatchForCandidate)
    );
    assert_eq!(wrong_tail_store.current.get("civ-log-v1"), Some(&previous));

    // First observation cannot silently become an anchor.
    let mut no_anchor_store = WitnessStore::new("witness-no-anchor", "lineage-no-anchor");
    assert_eq!(
        no_anchor_store.observe(candidate.clone(), Some(&proof), &suffix, 102, 10),
        Err(Failure::UnanchoredLog)
    );

    // Rollback is rejected and leaves witness state unchanged.
    let rollback_entries = entries[..2].to_vec();
    let rollback = make_anchor(
        "civ-log-v1",
        &rollback_entries,
        102,
        "policy-v1",
        8,
        Some(events[7].digest),
        Some(anchor_identity(&candidate)),
    );
    let mut rollback_store = WitnessStore::new("witness-rollback", "lineage-rollback");
    rollback_store.seed_trusted(candidate.clone()).expect("seed latest");
    assert_eq!(
        rollback_store.observe(rollback, None, &[], 103, 10),
        Err(Failure::TreeRollback)
    );
    assert_eq!(rollback_store.current.get("civ-log-v1"), Some(&candidate));

    // A same-size fork is retained even if its timestamp regresses.
    let mut fork_entries = candidate_entries.to_vec();
    fork_entries[2].push(0xFE);
    let fork = make_anchor(
        "civ-log-v1",
        &fork_entries,
        99,
        "policy-v1",
        7,
        Some(events[6].digest),
        Some(anchor_identity(&previous)),
    );
    assert_ne!(fork.root_hash, candidate.root_hash);
    let mut fork_store = WitnessStore::new("witness-fork", "lineage-fork");
    fork_store.seed_trusted(candidate.clone()).expect("seed candidate");
    assert_eq!(
        fork_store.observe(fork.clone(), None, &[], 102, 10),
        Err(Failure::ForkDetected)
    );
    assert_eq!(fork_store.forks.len(), 1);
    assert_eq!(fork_store.forks[0].first_anchor, candidate);
    assert_eq!(fork_store.forks[0].conflicting_anchor, fork);
    assert_eq!(
        fork_store.current.get("civ-log-v1"),
        Some(&fork_store.forks[0].first_anchor)
    );

    // Time failures preserve the independently retained state.
    let mut time_store = WitnessStore::new("witness-time", "lineage-time");
    time_store.seed_trusted(previous.clone()).expect("seed previous");
    let future = CheckpointAnchor {
        timestamp_epoch: 104,
        ..candidate.clone()
    };
    assert_eq!(
        time_store.observe(future, Some(&proof), &suffix, 103, 10),
        Err(Failure::FutureCheckpoint)
    );
    let stale = CheckpointAnchor {
        timestamp_epoch: 1,
        ..candidate.clone()
    };
    assert_eq!(
        time_store.observe(stale, Some(&proof), &suffix, 103, 10),
        Err(Failure::StaleCheckpoint)
    );
    let regressed = CheckpointAnchor {
        timestamp_epoch: 99,
        ..previous.clone()
    };
    assert_eq!(
        time_store.observe(regressed, None, &[], 102, 10),
        Err(Failure::TimestampRegression)
    );

    // Receipt rollback, wrong prior-anchor link, and unsupported anchor metadata
    // are independent failures and must not advance witness state.
    let receipt_rollback = make_anchor(
        "civ-log-v1",
        previous_entries,
        101,
        "policy-v1",
        3,
        Some(events[2].digest),
        Some(anchor_identity(&previous)),
    );
    assert_eq!(
        time_store.observe(receipt_rollback, None, &[], 102, 10),
        Err(Failure::ReceiptRollback)
    );
    let bad_link = CheckpointAnchor {
        previous_anchor_digest: Some([0x55; 32]),
        ..candidate.clone()
    };
    assert_eq!(
        time_store.observe(bad_link, Some(&proof), &suffix, 102, 10),
        Err(Failure::MissingAnchorLink)
    );
    let bad_protocol = CheckpointAnchor {
        protocol_version: 2,
        ..candidate.clone()
    };
    assert_eq!(
        time_store.observe(bad_protocol, Some(&proof), &suffix, 102, 10),
        Err(Failure::UnsupportedProtocolVersion)
    );
    let empty_log = CheckpointAnchor {
        log_id: String::new(),
        ..candidate.clone()
    };
    assert_eq!(
        time_store.observe(empty_log, Some(&proof), &suffix, 102, 10),
        Err(Failure::EmptyLogId)
    );
    let malformed_tail = CheckpointAnchor {
        receipt_sequence: 0,
        receipt_digest: Some(events[0].digest),
        ..candidate.clone()
    };
    assert_eq!(
        time_store.observe(malformed_tail, Some(&proof), &suffix, 102, 10),
        Err(Failure::InvalidReceiptTailShape)
    );
    assert_eq!(time_store.current.get("civ-log-v1"), Some(&previous));

    // Quorum guard tests cover threshold, duplicate identities, shared lineages,
    // tampered fixture attestations, and exact anchor binding.
    assert_eq!(
        verify_quorum(&attestations[..2], &candidate, 3),
        Err(Failure::InsufficientWitnesses)
    );
    let duplicate = vec![
        attestations[0].clone(),
        attestations[0].clone(),
        attestations[2].clone(),
    ];
    assert_eq!(
        verify_quorum(&duplicate, &candidate, 3),
        Err(Failure::DuplicateWitness)
    );
    let same_lineage = vec![
        attestations[0].clone(),
        fixture_attestation("witness-b2", "lineage-a", anchor_identity(&candidate)),
        attestations[2].clone(),
    ];
    assert_eq!(
        verify_quorum(&same_lineage, &candidate, 3),
        Err(Failure::SharedAuthorityLineage)
    );
    let mut forged = attestations.clone();
    forged[0].fixture_signature[0] ^= 1;
    assert_eq!(
        verify_quorum(&forged, &candidate, 3),
        Err(Failure::InvalidFixtureAttestation)
    );
    let other_anchor = CheckpointAnchor {
        policy_version: "other-policy".to_owned(),
        ..candidate.clone()
    };
    assert_eq!(
        verify_quorum(&attestations, &other_anchor, 3),
        Err(Failure::WitnessAnchorMismatch)
    );

    // After restart, a valid local prefix below the witness's retained tail is
    // rejected; suffixes after the anchor are not asserted by that anchor.
    let mut restart_store = WitnessStore::new("witness-restart", "lineage-restart");
    restart_store.seed_trusted(previous.clone()).expect("seed initial");
    restart_store
        .observe(candidate.clone(), Some(&proof), &suffix, 102, 10)
        .expect("advance witness");
    assert_eq!(
        reconcile_local_chain(
            &events[..5],
            restart_store.current.get("civ-log-v1").expect("current anchor"),
        ),
        Err(Failure::LocalReceiptTailTruncated)
    );
    assert_eq!(reconcile_local_chain(&events, &candidate), Ok(()));

    println!("SYM-CIV-012 PASS: witness anchors detect tail truncation, rollback and forks.");
    println!("Claim ceiling: fixture only; no real signatures, persistence, or network service.");
}
