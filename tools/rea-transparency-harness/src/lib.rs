//! Offline, bounded transparency proof verification.
//!
//! Implements RFC 6962 SHA-256 Merkle primitives. This crate does not fetch logs, validate
//! checkpoint signatures, authenticate publishers, or assert that signed statements are true.

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const MAX_PROOF_NODES: usize = 64;
const LEAF_PREFIX: u8 = 0x00;
const NODE_PREFIX: u8 = 0x01;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MerkleHash(pub [u8; 32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProofError {
    EmptyTree,
    InvalidTreeSize,
    LeafIndexOutOfRange,
    ProofTooLong,
    InvalidInclusionProof,
    InvalidConsistencyProof,
    ConsistencyRequired,
    UnexpectedConsistencyProof,
    WrongLog,
    CheckpointRollback,
    CheckpointEquivocation,
    InvalidWitnessQuorum,
    WitnessEquivocation,
    SubjectMismatch,
}

impl fmt::Display for ProofError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::EmptyTree => "empty tree has no inclusion proofs",
            Self::InvalidTreeSize => "tree size is invalid",
            Self::LeafIndexOutOfRange => "leaf index is outside tree size",
            Self::ProofTooLong => "proof exceeds the bounded path length",
            Self::InvalidInclusionProof => "Merkle inclusion proof is invalid",
            Self::InvalidConsistencyProof => "Merkle consistency proof is invalid",
            Self::ConsistencyRequired => "larger checkpoint requires a consistency proof",
            Self::UnexpectedConsistencyProof => "first checkpoint must not carry a prior proof",
            Self::WrongLog => "checkpoint belongs to a different transparency log",
            Self::CheckpointRollback => "checkpoint tree size is older than the pinned checkpoint",
            Self::CheckpointEquivocation => "same log and tree size have conflicting roots",
            Self::InvalidWitnessQuorum => "witness quorum configuration or evidence is invalid",
            Self::WitnessEquivocation => "one witness operator reported conflicting checkpoints",
            Self::SubjectMismatch => "transparency subject kind or digest differs from expectation",
        })
    }
}
impl std::error::Error for ProofError {}

/// RFC 6962: SHA-256(0x00 || leaf_data).
pub fn hash_leaf(leaf_data: &[u8]) -> MerkleHash {
    let mut hasher = Sha256::new();
    hasher.update([LEAF_PREFIX]);
    hasher.update(leaf_data);
    MerkleHash(hasher.finalize().into())
}

/// RFC 6962: SHA-256(0x01 || left || right).
pub fn hash_node(left: MerkleHash, right: MerkleHash) -> MerkleHash {
    let mut hasher = Sha256::new();
    hasher.update([NODE_PREFIX]);
    hasher.update(left.0);
    hasher.update(right.0);
    MerkleHash(hasher.finalize().into())
}

pub fn empty_tree_hash() -> MerkleHash {
    MerkleHash(Sha256::digest([]).into())
}

/// Verify an inclusion proof against a root supplied by the caller. The caller must separately
/// verify the checkpoint signature, log identity and VDS-specific leaf preimage.
pub fn verify_inclusion(
    leaf_data: &[u8],
    leaf_index: u64,
    tree_size: u64,
    audit_path: &[MerkleHash],
    expected_root: MerkleHash,
) -> Result<(), ProofError> {
    if tree_size == 0 {
        return Err(ProofError::EmptyTree);
    }
    if leaf_index >= tree_size {
        return Err(ProofError::LeafIndexOutOfRange);
    }
    if audit_path.len() > MAX_PROOF_NODES {
        return Err(ProofError::ProofTooLong);
    }

    let mut fn_index = leaf_index;
    let mut sn_index = tree_size - 1;
    let mut calculated = hash_leaf(leaf_data);

    for sibling in audit_path {
        if sn_index == 0 {
            return Err(ProofError::InvalidInclusionProof);
        }
        if (fn_index & 1) == 1 || fn_index == sn_index {
            calculated = hash_node(*sibling, calculated);
            while fn_index != 0 && (fn_index & 1) == 0 {
                fn_index >>= 1;
                sn_index >>= 1;
            }
        } else {
            calculated = hash_node(calculated, *sibling);
        }
        fn_index >>= 1;
        sn_index >>= 1;
    }

    if sn_index != 0 || calculated != expected_root {
        return Err(ProofError::InvalidInclusionProof);
    }
    Ok(())
}

/// Verify that a newer tree is an append-only extension of an older RFC 6962 tree.
pub fn verify_consistency(
    old_tree_size: u64,
    new_tree_size: u64,
    old_root: MerkleHash,
    new_root: MerkleHash,
    proof: &[MerkleHash],
) -> Result<(), ProofError> {
    if proof.len() > MAX_PROOF_NODES {
        return Err(ProofError::ProofTooLong);
    }
    if old_tree_size > new_tree_size {
        return Err(ProofError::InvalidTreeSize);
    }
    if old_tree_size == 0 {
        return if old_root == empty_tree_hash() && proof.is_empty() {
            Ok(())
        } else {
            Err(ProofError::InvalidConsistencyProof)
        };
    }
    if old_tree_size == new_tree_size {
        return if proof.is_empty() && old_root == new_root {
            Ok(())
        } else {
            Err(ProofError::InvalidConsistencyProof)
        };
    }
    if proof.is_empty() {
        return Err(ProofError::InvalidConsistencyProof);
    }

    let mut fn_index = old_tree_size - 1;
    let mut sn_index = new_tree_size - 1;
    while (fn_index & 1) == 1 {
        fn_index >>= 1;
        sn_index >>= 1;
    }

    let (mut first_root, mut second_root, mut proof_index) = if fn_index == 0 {
        (old_root, old_root, 0usize)
    } else {
        (proof[0], proof[0], 1usize)
    };

    while proof_index < proof.len() {
        if sn_index == 0 {
            return Err(ProofError::InvalidConsistencyProof);
        }
        let sibling = proof[proof_index];
        if (fn_index & 1) == 1 || fn_index == sn_index {
            first_root = hash_node(sibling, first_root);
            second_root = hash_node(sibling, second_root);
            while fn_index != 0 && (fn_index & 1) == 0 {
                fn_index >>= 1;
                sn_index >>= 1;
            }
        } else {
            second_root = hash_node(second_root, sibling);
        }
        fn_index >>= 1;
        sn_index >>= 1;
        proof_index += 1;
    }

    if sn_index != 0 || first_root != old_root || second_root != new_root {
        return Err(ProofError::InvalidConsistencyProof);
    }
    Ok(())
}

// Largest power of two strictly smaller than n; required by RFC 6962's recursive tree shape.
fn split_point(n: usize) -> usize {
    debug_assert!(n > 1);
    let highest_bit = usize::BITS - 1 - (n - 1).leading_zeros();
    1usize << highest_bit
}

/// Fixture helper: recursively compute the RFC 6962 Merkle Tree Hash.
pub fn merkle_tree_hash(leaves: &[Vec<u8>]) -> MerkleHash {
    match leaves.len() {
        0 => empty_tree_hash(),
        1 => hash_leaf(&leaves[0]),
        n => {
            let k = split_point(n);
            hash_node(
                merkle_tree_hash(&leaves[..k]),
                merkle_tree_hash(&leaves[k..]),
            )
        }
    }
}

/// Fixture helper: create the audit path for a leaf index.
pub fn inclusion_path(leaves: &[Vec<u8>], index: usize) -> Option<Vec<MerkleHash>> {
    if leaves.is_empty() || index >= leaves.len() {
        return None;
    }
    if leaves.len() == 1 {
        return Some(Vec::new());
    }
    let k = split_point(leaves.len());
    if index < k {
        let mut path = inclusion_path(&leaves[..k], index)?;
        path.push(merkle_tree_hash(&leaves[k..]));
        Some(path)
    } else {
        let mut path = inclusion_path(&leaves[k..], index - k)?;
        path.push(merkle_tree_hash(&leaves[..k]));
        Some(path)
    }
}

/// Fixture helper: create the RFC 6962 consistency path for an old prefix of leaves.
pub fn consistency_path(old_size: usize, leaves: &[Vec<u8>]) -> Option<Vec<MerkleHash>> {
    if old_size > leaves.len() {
        return None;
    }
    if old_size == 0 || old_size == leaves.len() {
        return Some(Vec::new());
    }
    Some(subproof(old_size, leaves, true))
}

fn subproof(old_size: usize, leaves: &[Vec<u8>], complete: bool) -> Vec<MerkleHash> {
    if old_size == leaves.len() {
        return if complete {
            Vec::new()
        } else {
            vec![merkle_tree_hash(leaves)]
        };
    }
    let k = split_point(leaves.len());
    if old_size <= k {
        let mut proof = subproof(old_size, &leaves[..k], complete);
        proof.push(merkle_tree_hash(&leaves[k..]));
        proof
    } else {
        let mut proof = subproof(old_size - k, &leaves[k..], false);
        proof.push(merkle_tree_hash(&leaves[..k]));
        proof
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LogId(pub [u8; 32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MerkleScheme {
    Rfc6962Sha256,
}

/// Checkpoint fields only. Validate its signature and trust root before passing it to the monitor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub log_id: LogId,
    pub scheme: MerkleScheme,
    pub tree_size: u64,
    pub root_hash: MerkleHash,
    pub timestamp_unix_ms: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointDisposition {
    FirstPinned,
    Unchanged,
    ConsistentExtension,
}

/// In-memory anti-rollback/equivocation guard. Persist accepted checkpoints atomically outside
/// this crate; reconstruct the monitor from the persisted, signature-validated checkpoint.
#[derive(Clone, Debug)]
pub struct CheckpointMonitor {
    log_id: LogId,
    current: Option<Checkpoint>,
}
impl CheckpointMonitor {
    pub fn new(log_id: LogId) -> Self {
        Self {
            log_id,
            current: None,
        }
    }
    pub fn current(&self) -> Option<&Checkpoint> {
        self.current.as_ref()
    }

    /// Call only after external checkpoint signature and log-key validation.
    pub fn observe_after_signature_validation(
        &mut self,
        next: Checkpoint,
        consistency_proof: Option<&[MerkleHash]>,
    ) -> Result<CheckpointDisposition, ProofError> {
        if next.log_id != self.log_id {
            return Err(ProofError::WrongLog);
        }
        match self.current.as_ref() {
            None => {
                if consistency_proof.is_some() {
                    return Err(ProofError::UnexpectedConsistencyProof);
                }
                self.current = Some(next);
                Ok(CheckpointDisposition::FirstPinned)
            }
            Some(current) => {
                if next.tree_size < current.tree_size {
                    return Err(ProofError::CheckpointRollback);
                }
                if next.tree_size == current.tree_size {
                    if next.root_hash != current.root_hash {
                        return Err(ProofError::CheckpointEquivocation);
                    }
                    if next.scheme != current.scheme {
                        return Err(ProofError::WrongLog);
                    }
                    return Ok(CheckpointDisposition::Unchanged);
                }
                let proof = consistency_proof.ok_or(ProofError::ConsistencyRequired)?;
                verify_consistency(
                    current.tree_size,
                    next.tree_size,
                    current.root_hash,
                    next.root_hash,
                    proof,
                )?;
                self.current = Some(next);
                Ok(CheckpointDisposition::ConsistentExtension)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransparencySubjectKind {
    ArtifactSha256,
    CodeTransparencyJwtSha256,
    SignedStatementSha256,
}

/// The subject kind is part of identity even when digest bytes happen to be equal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransparencySubject {
    pub kind: TransparencySubjectKind,
    pub digest: MerkleHash,
    pub media_type: String,
}

pub fn require_subject_binding(
    expected: &TransparencySubject,
    observed: &TransparencySubject,
) -> Result<(), ProofError> {
    if expected == observed {
        Ok(())
    } else {
        Err(ProofError::SubjectMismatch)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WitnessObservation {
    pub operator_id: String,
    pub key_id: String,
    pub checkpoint: Checkpoint,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WitnessQuorum {
    pub required_operators: usize,
    pub matching_operators: Vec<String>,
}
impl WitnessQuorum {
    pub fn is_met(&self) -> bool {
        self.matching_operators.len() >= self.required_operators
    }
}

/// Count independent operators, not keys. Multiple keys from one operator count once; two
/// different roots reported by the same operator for the same log/size are equivocation.
pub fn qualify_witness_quorum(
    expected: &Checkpoint,
    observations: &[WitnessObservation],
    required_operators: usize,
) -> Result<WitnessQuorum, ProofError> {
    if required_operators == 0 {
        return Err(ProofError::InvalidWitnessQuorum);
    }
    let mut per_operator: BTreeMap<&str, BTreeSet<(LogId, u64, MerkleHash)>> = BTreeMap::new();
    for item in observations {
        if item.operator_id.trim().is_empty() || item.key_id.trim().is_empty() {
            return Err(ProofError::InvalidWitnessQuorum);
        }
        if item.checkpoint.log_id == expected.log_id
            && item.checkpoint.tree_size == expected.tree_size
        {
            per_operator
                .entry(item.operator_id.as_str())
                .or_default()
                .insert((
                    item.checkpoint.log_id,
                    item.checkpoint.tree_size,
                    item.checkpoint.root_hash,
                ));
        }
    }
    if per_operator.values().any(|roots| roots.len() > 1) {
        return Err(ProofError::WitnessEquivocation);
    }
    let matching_operators = per_operator
        .into_iter()
        .filter_map(|(operator, roots)| {
            roots
                .iter()
                .any(|(log, size, root)| {
                    *log == expected.log_id
                        && *size == expected.tree_size
                        && *root == expected.root_hash
                })
                .then(|| operator.to_owned())
        })
        .collect();
    Ok(WitnessQuorum {
        required_operators,
        matching_operators,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaves(n: usize) -> Vec<Vec<u8>> {
        (0..n).map(|i| format!("leaf-{i}").into_bytes()).collect()
    }

    fn checkpoint(log: LogId, tree: &[Vec<u8>], timestamp: u64) -> Checkpoint {
        Checkpoint {
            log_id: log,
            scheme: MerkleScheme::Rfc6962Sha256,
            tree_size: tree.len() as u64,
            root_hash: merkle_tree_hash(tree),
            timestamp_unix_ms: timestamp,
        }
    }

    #[test]
    fn leaf_hash_is_domain_separated_and_empty_tree_is_sha256_empty() {
        assert_eq!(
            hash_leaf(b"").0,
            [
                0x6e, 0x34, 0x0b, 0x9c, 0xff, 0xb3, 0x7a, 0x98,
                0x9c, 0xa5, 0x44, 0xe6, 0xbb, 0x78, 0x0a, 0x2c,
                0x78, 0x90, 0x1d, 0x3f, 0xb3, 0x37, 0x38, 0x76,
                0x85, 0x11, 0xa3, 0x06, 0x17, 0xaf, 0xa0, 0x1d,
            ]
        );
        assert_eq!(
            empty_tree_hash().0,
            [
                0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
                0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
                0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
                0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
            ]
        );
        assert_ne!(hash_leaf(b"abc"), MerkleHash(Sha256::digest(b"abc").into()));
        assert_ne!(
            hash_node(hash_leaf(b"a"), hash_leaf(b"b")),
            hash_node(hash_leaf(b"b"), hash_leaf(b"a"))
        );
    }

    #[test]
    fn every_inclusion_path_verifies_for_tree_sizes_one_through_32() {
        for size in 1..=32 {
            let tree = leaves(size);
            let root = merkle_tree_hash(&tree);
            for index in 0..size {
                let proof = inclusion_path(&tree, index).unwrap();
                verify_inclusion(&tree[index], index as u64, size as u64, &proof, root).unwrap();
            }
        }
    }

    #[test]
    fn inclusion_rejects_mutated_leaf_index_root_path_and_extra_sibling() {
        let tree = leaves(7);
        let root = merkle_tree_hash(&tree);
        let proof = inclusion_path(&tree, 4).unwrap();
        assert_eq!(verify_inclusion(&tree[4], 4, 7, &proof, root), Ok(()));
        assert_eq!(
            verify_inclusion(&tree[4], 3, 7, &proof, root),
            Err(ProofError::InvalidInclusionProof)
        );
        assert_eq!(
            verify_inclusion(&tree[5], 4, 7, &proof, root),
            Err(ProofError::InvalidInclusionProof)
        );
        assert_eq!(
            verify_inclusion(&tree[4], 4, 7, &proof, hash_leaf(b"wrong")),
            Err(ProofError::InvalidInclusionProof)
        );
        let mut changed = proof.clone();
        changed[0] = hash_leaf(b"changed sibling");
        assert_eq!(
            verify_inclusion(&tree[4], 4, 7, &changed, root),
            Err(ProofError::InvalidInclusionProof)
        );
        let mut extra = proof;
        extra.push(hash_leaf(b"extra"));
        assert_eq!(
            verify_inclusion(&tree[4], 4, 7, &extra, root),
            Err(ProofError::InvalidInclusionProof)
        );
    }

    #[test]
    fn inclusion_rejects_invalid_index_empty_tree_and_oversized_path() {
        assert_eq!(
            verify_inclusion(b"x", 0, 0, &[], empty_tree_hash()),
            Err(ProofError::EmptyTree)
        );
        assert_eq!(
            verify_inclusion(b"x", 1, 1, &[], hash_leaf(b"x")),
            Err(ProofError::LeafIndexOutOfRange)
        );
        let long = vec![hash_leaf(b"sibling"); MAX_PROOF_NODES + 1];
        assert_eq!(
            verify_inclusion(b"x", 0, 2, &long, hash_leaf(b"x")),
            Err(ProofError::ProofTooLong)
        );
    }

    #[test]
    fn every_prefix_consistency_proof_verifies_for_tree_sizes_one_through_32() {
        for new_size in 1..=32 {
            let tree = leaves(new_size);
            let new_root = merkle_tree_hash(&tree);
            for old_size in 1..=new_size {
                let old_root = merkle_tree_hash(&tree[..old_size]);
                let proof = consistency_path(old_size, &tree).unwrap();
                verify_consistency(old_size as u64, new_size as u64, old_root, new_root, &proof)
                    .unwrap_or_else(|e| panic!("consistency {old_size}->{new_size}: {e:?}"));
            }
        }
    }

    #[test]
    fn consistency_rejects_rollback_wrong_root_and_mutated_path() {
        let tree = leaves(9);
        let proof = consistency_path(4, &tree).unwrap();
        let old = merkle_tree_hash(&tree[..4]);
        let new = merkle_tree_hash(&tree);
        assert_eq!(verify_consistency(4, 9, old, new, &proof), Ok(()));
        assert_eq!(
            verify_consistency(9, 4, new, old, &[]),
            Err(ProofError::InvalidTreeSize)
        );
        assert_eq!(
            verify_consistency(4, 9, old, hash_leaf(b"wrong"), &proof),
            Err(ProofError::InvalidConsistencyProof)
        );
        let mut changed = proof;
        changed[0] = hash_leaf(b"changed");
        assert_eq!(
            verify_consistency(4, 9, old, new, &changed),
            Err(ProofError::InvalidConsistencyProof)
        );
        assert_eq!(
            verify_consistency(4, 9, old, new, &[]),
            Err(ProofError::InvalidConsistencyProof)
        );
    }

    #[test]
    fn empty_and_identical_tree_consistency_boundaries_are_explicit() {
        let tree = leaves(3);
        let root = merkle_tree_hash(&tree);
        assert_eq!(
            verify_consistency(0, 3, empty_tree_hash(), root, &[]),
            Ok(())
        );
        assert_eq!(verify_consistency(3, 3, root, root, &[]), Ok(()));
        assert_eq!(
            verify_consistency(3, 3, root, hash_leaf(b"other"), &[]),
            Err(ProofError::InvalidConsistencyProof)
        );
    }

    #[test]
    fn checkpoint_monitor_blocks_rollback_and_same_size_equivocation() {
        let log = LogId([7; 32]);
        let old_tree = leaves(3);
        let new_tree = leaves(6);
        let old = checkpoint(log, &old_tree, 100);
        let new = checkpoint(log, &new_tree, 200);
        let mut monitor = CheckpointMonitor::new(log);
        assert_eq!(
            monitor.observe_after_signature_validation(old.clone(), None),
            Ok(CheckpointDisposition::FirstPinned)
        );
        assert_eq!(
            monitor.observe_after_signature_validation(old.clone(), None),
            Ok(CheckpointDisposition::Unchanged)
        );
        let conflicting = Checkpoint {
            root_hash: hash_leaf(b"split view"),
            ..old.clone()
        };
        assert_eq!(
            monitor.observe_after_signature_validation(conflicting, None),
            Err(ProofError::CheckpointEquivocation)
        );
        assert_eq!(
            monitor.observe_after_signature_validation(new.clone(), None),
            Err(ProofError::ConsistencyRequired)
        );
        let proof = consistency_path(old_tree.len(), &new_tree).unwrap();
        assert_eq!(
            monitor.observe_after_signature_validation(new, Some(&proof)),
            Ok(CheckpointDisposition::ConsistentExtension)
        );
        assert_eq!(
            monitor.observe_after_signature_validation(old, None),
            Err(ProofError::CheckpointRollback)
        );
    }

    #[test]
    fn checkpoint_monitor_requires_same_log_and_valid_consistency() {
        let log = LogId([8; 32]);
        let tree = leaves(2);
        let old = checkpoint(log, &tree[..1], 1);
        let new = checkpoint(log, &tree, 2);
        let mut monitor = CheckpointMonitor::new(log);
        monitor
            .observe_after_signature_validation(old, None)
            .unwrap();
        let wrong_log = Checkpoint {
            log_id: LogId([9; 32]),
            ..new.clone()
        };
        assert_eq!(
            monitor.observe_after_signature_validation(wrong_log, Some(&[])),
            Err(ProofError::WrongLog)
        );
        assert_eq!(
            monitor.observe_after_signature_validation(new.clone(), Some(&[])),
            Err(ProofError::InvalidConsistencyProof)
        );
        let proof = consistency_path(1, &tree).unwrap();
        assert_eq!(
            monitor.observe_after_signature_validation(new, Some(&proof)),
            Ok(CheckpointDisposition::ConsistentExtension)
        );
    }

    #[test]
    fn artifact_and_code_transparency_jwt_are_distinct_subject_kinds() {
        let digest = hash_leaf(b"same bytes");
        let artifact = TransparencySubject {
            kind: TransparencySubjectKind::ArtifactSha256,
            digest,
            media_type: "application/vnd.android.package-archive".into(),
        };
        let token = TransparencySubject {
            kind: TransparencySubjectKind::CodeTransparencyJwtSha256,
            digest,
            media_type: "application/jwt".into(),
        };
        assert_eq!(
            require_subject_binding(&artifact, &token),
            Err(ProofError::SubjectMismatch)
        );
        assert_eq!(require_subject_binding(&artifact, &artifact), Ok(()));
    }

    #[test]
    fn witness_quorum_counts_distinct_operators_not_key_count() {
        let log = LogId([3; 32]);
        let tree = leaves(2);
        let expected = checkpoint(log, &tree, 55);
        let observations = vec![
            WitnessObservation {
                operator_id: "operator-a".into(),
                key_id: "key-a1".into(),
                checkpoint: expected.clone(),
            },
            WitnessObservation {
                operator_id: "operator-a".into(),
                key_id: "key-a2".into(),
                checkpoint: expected.clone(),
            },
            WitnessObservation {
                operator_id: "operator-b".into(),
                key_id: "key-b1".into(),
                checkpoint: expected.clone(),
            },
        ];
        let quorum = qualify_witness_quorum(&expected, &observations, 2).unwrap();
        assert_eq!(quorum.matching_operators.len(), 2);
        assert!(quorum.is_met());
    }

    #[test]
    fn witness_operator_equivocation_is_not_counted_as_multiple_witnesses() {
        let log = LogId([4; 32]);
        let tree = leaves(2);
        let expected = checkpoint(log, &tree, 55);
        let conflicting = Checkpoint {
            root_hash: hash_leaf(b"other root"),
            ..expected.clone()
        };
        let observations = vec![
            WitnessObservation {
                operator_id: "operator-a".into(),
                key_id: "key-a1".into(),
                checkpoint: expected.clone(),
            },
            WitnessObservation {
                operator_id: "operator-a".into(),
                key_id: "key-a2".into(),
                checkpoint: conflicting,
            },
        ];
        assert_eq!(
            qualify_witness_quorum(&expected, &observations, 1),
            Err(ProofError::WitnessEquivocation)
        );
    }
}
