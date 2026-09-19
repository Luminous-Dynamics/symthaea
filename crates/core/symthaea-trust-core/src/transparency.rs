// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Domain-neutral append-only Merkle transparency substrate.
//!
//! TRANSP-001A is deliberately structural only. It proves content inclusion and
//! append-only consistency between tree heads, but does not authenticate a log
//! operator, establish trusted time, establish witness independence, or grant
//! scientific/fabrication authority. Signed checkpoints and witness quorums are
//! later layers.
//!
//! The tree shape and compact inclusion/consistency proof algorithms follow the
//! Certificate Transparency split-at-largest-power-of-two construction. Hash
//! domains are Symthaea-specific and entries additionally form an explicit
//! predecessor hash chain.

use serde::Serialize;

use crate::{FramedDigest, Sha256Digest, TrustUsage};

pub const TRANSPARENCY_LOG_SCHEMA: &str = "symthaea.transparency-log.v1";
pub const TRANSPARENCY_ENTRY_SCHEMA: &str = "symthaea.transparency-entry.v1";
pub const TRANSPARENCY_TREE_HEAD_SCHEMA: &str = "symthaea.transparency-tree-head.v1";
const ENTRY_DOMAIN: &str = "symthaea.transparency-entry.identity.v1";
const LEAF_DOMAIN: &str = "symthaea.transparency-merkle-leaf.v1";
const NODE_DOMAIN: &str = "symthaea.transparency-merkle-node.v1";
const EMPTY_DOMAIN: &str = "symthaea.transparency-merkle-empty.v1";
const TREE_HEAD_DOMAIN: &str = "symthaea.transparency-tree-head.identity.v1";
pub const MAX_TRANSPARENCY_ENTRIES: usize = 1_000_000;
pub const MAX_PROOF_NODES: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransparencyEntryDraft {
    pub kind: TrustUsage,
    pub subject_sha256: Sha256Digest,
    pub payload_sha256: Option<Sha256Digest>,
    pub context_sha256: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransparencyEntry {
    schema_version: String,
    sequence: u64,
    kind: TrustUsage,
    subject_sha256: Sha256Digest,
    payload_sha256: Option<Sha256Digest>,
    context_sha256: Option<Sha256Digest>,
    previous_entry_sha256: Option<Sha256Digest>,
    entry_sha256: Sha256Digest,
}

impl TransparencyEntry {
    pub fn sequence(&self) -> u64 { self.sequence }
    pub fn kind(&self) -> &TrustUsage { &self.kind }
    pub fn subject_sha256(&self) -> &Sha256Digest { &self.subject_sha256 }
    pub fn payload_sha256(&self) -> Option<&Sha256Digest> { self.payload_sha256.as_ref() }
    pub fn context_sha256(&self) -> Option<&Sha256Digest> { self.context_sha256.as_ref() }
    pub fn previous_entry_sha256(&self) -> Option<&Sha256Digest> {
        self.previous_entry_sha256.as_ref()
    }
    pub fn entry_sha256(&self) -> &Sha256Digest { &self.entry_sha256 }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransparencyLogError {
    CapacityExceeded,
    InvalidTreeSize,
    IndexOutOfBounds,
    FirstTreeSizeZero,
    FirstTreeLargerThanSecond,
    ProofTooDeep,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransparencyTreeHead {
    schema_version: String,
    tree_size: u64,
    root_sha256: Sha256Digest,
    last_entry_sha256: Option<Sha256Digest>,
    tree_head_sha256: Sha256Digest,
}

impl TransparencyTreeHead {
    pub fn tree_size(&self) -> u64 { self.tree_size }
    pub fn root_sha256(&self) -> &Sha256Digest { &self.root_sha256 }
    pub fn last_entry_sha256(&self) -> Option<&Sha256Digest> { self.last_entry_sha256.as_ref() }
    pub fn tree_head_sha256(&self) -> &Sha256Digest { &self.tree_head_sha256 }
    pub const fn operator_authority_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransparencyInclusionProof {
    tree_size: u64,
    leaf_index: u64,
    entry_sha256: Sha256Digest,
    leaf_sha256: Sha256Digest,
    root_sha256: Sha256Digest,
    path: Vec<Sha256Digest>,
}

impl TransparencyInclusionProof {
    pub fn tree_size(&self) -> u64 { self.tree_size }
    pub fn leaf_index(&self) -> u64 { self.leaf_index }
    pub fn entry_sha256(&self) -> &Sha256Digest { &self.entry_sha256 }
    pub fn root_sha256(&self) -> &Sha256Digest { &self.root_sha256 }
    pub fn path(&self) -> &[Sha256Digest] { &self.path }
    pub const fn authority_established(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransparencyConsistencyProof {
    first_tree_size: u64,
    second_tree_size: u64,
    first_root_sha256: Sha256Digest,
    second_root_sha256: Sha256Digest,
    path: Vec<Sha256Digest>,
}

impl TransparencyConsistencyProof {
    pub fn first_tree_size(&self) -> u64 { self.first_tree_size }
    pub fn second_tree_size(&self) -> u64 { self.second_tree_size }
    pub fn first_root_sha256(&self) -> &Sha256Digest { &self.first_root_sha256 }
    pub fn second_root_sha256(&self) -> &Sha256Digest { &self.second_root_sha256 }
    pub fn path(&self) -> &[Sha256Digest] { &self.path }
    pub const fn global_log_consistency_established(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransparencyProofError {
    EmptyTree,
    InvalidIndex,
    InvalidTreeSizes,
    ProofTooDeep,
    LeafIdentityMismatch,
    RootMismatch,
    InvalidProof,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransparencyLog {
    schema_version: String,
    entries: Vec<TransparencyEntry>,
}

impl Default for TransparencyLog {
    fn default() -> Self { Self::new() }
}

impl TransparencyLog {
    pub fn new() -> Self {
        Self { schema_version: TRANSPARENCY_LOG_SCHEMA.into(), entries: Vec::new() }
    }

    pub fn entries(&self) -> &[TransparencyEntry] { &self.entries }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    pub fn append(
        &mut self,
        draft: TransparencyEntryDraft,
    ) -> Result<Sha256Digest, TransparencyLogError> {
        if self.entries.len() >= MAX_TRANSPARENCY_ENTRIES {
            return Err(TransparencyLogError::CapacityExceeded);
        }
        let sequence = self.entries.len() as u64 + 1;
        let previous_entry_sha256 = self.entries.last().map(|entry| entry.entry_sha256.clone());
        let entry_sha256 = entry_digest(
            sequence,
            &draft.kind,
            &draft.subject_sha256,
            draft.payload_sha256.as_ref(),
            draft.context_sha256.as_ref(),
            previous_entry_sha256.as_ref(),
        );
        self.entries.push(TransparencyEntry {
            schema_version: TRANSPARENCY_ENTRY_SCHEMA.into(),
            sequence,
            kind: draft.kind,
            subject_sha256: draft.subject_sha256,
            payload_sha256: draft.payload_sha256,
            context_sha256: draft.context_sha256,
            previous_entry_sha256,
            entry_sha256: entry_sha256.clone(),
        });
        Ok(entry_sha256)
    }

    pub fn head(&self) -> TransparencyTreeHead {
        self.head_at(self.entries.len()).expect("current log size is valid")
    }

    pub fn head_at(&self, tree_size: usize) -> Result<TransparencyTreeHead, TransparencyLogError> {
        if tree_size > self.entries.len() {
            return Err(TransparencyLogError::InvalidTreeSize);
        }
        let leaves = self.leaf_hashes(tree_size);
        let root_sha256 = merkle_tree_hash(&leaves);
        let last_entry_sha256 = tree_size
            .checked_sub(1)
            .and_then(|index| self.entries.get(index))
            .map(|entry| entry.entry_sha256.clone());
        let tree_head_sha256 = tree_head_digest(
            tree_size as u64,
            &root_sha256,
            last_entry_sha256.as_ref(),
        );
        Ok(TransparencyTreeHead {
            schema_version: TRANSPARENCY_TREE_HEAD_SCHEMA.into(),
            tree_size: tree_size as u64,
            root_sha256,
            last_entry_sha256,
            tree_head_sha256,
        })
    }

    pub fn inclusion_proof(
        &self,
        leaf_index: usize,
        tree_size: usize,
    ) -> Result<TransparencyInclusionProof, TransparencyLogError> {
        if tree_size == 0 || tree_size > self.entries.len() {
            return Err(TransparencyLogError::InvalidTreeSize);
        }
        if leaf_index >= tree_size {
            return Err(TransparencyLogError::IndexOutOfBounds);
        }
        let leaves = self.leaf_hashes(tree_size);
        let path = inclusion_path(leaf_index, &leaves);
        if path.len() > MAX_PROOF_NODES {
            return Err(TransparencyLogError::ProofTooDeep);
        }
        Ok(TransparencyInclusionProof {
            tree_size: tree_size as u64,
            leaf_index: leaf_index as u64,
            entry_sha256: self.entries[leaf_index].entry_sha256.clone(),
            leaf_sha256: leaves[leaf_index].clone(),
            root_sha256: merkle_tree_hash(&leaves),
            path,
        })
    }

    pub fn consistency_proof(
        &self,
        first_tree_size: usize,
        second_tree_size: usize,
    ) -> Result<TransparencyConsistencyProof, TransparencyLogError> {
        if first_tree_size == 0 {
            return Err(TransparencyLogError::FirstTreeSizeZero);
        }
        if first_tree_size > second_tree_size {
            return Err(TransparencyLogError::FirstTreeLargerThanSecond);
        }
        if second_tree_size > self.entries.len() {
            return Err(TransparencyLogError::InvalidTreeSize);
        }
        let leaves = self.leaf_hashes(second_tree_size);
        let first_root_sha256 = merkle_tree_hash(&leaves[..first_tree_size]);
        let second_root_sha256 = merkle_tree_hash(&leaves);
        let path = if first_tree_size == second_tree_size {
            Vec::new()
        } else {
            consistency_subproof(first_tree_size, &leaves, true)
        };
        if path.len() > MAX_PROOF_NODES {
            return Err(TransparencyLogError::ProofTooDeep);
        }
        Ok(TransparencyConsistencyProof {
            first_tree_size: first_tree_size as u64,
            second_tree_size: second_tree_size as u64,
            first_root_sha256,
            second_root_sha256,
            path,
        })
    }

    fn leaf_hashes(&self, tree_size: usize) -> Vec<Sha256Digest> {
        self.entries[..tree_size]
            .iter()
            .map(|entry| merkle_leaf_hash(&entry.entry_sha256))
            .collect()
    }
}

pub fn verify_transparency_inclusion(
    proof: &TransparencyInclusionProof,
) -> Result<(), TransparencyProofError> {
    if proof.tree_size == 0 {
        return Err(TransparencyProofError::EmptyTree);
    }
    if proof.leaf_index >= proof.tree_size {
        return Err(TransparencyProofError::InvalidIndex);
    }
    if proof.path.len() > MAX_PROOF_NODES {
        return Err(TransparencyProofError::ProofTooDeep);
    }
    let expected_leaf = merkle_leaf_hash(&proof.entry_sha256);
    if expected_leaf != proof.leaf_sha256 {
        return Err(TransparencyProofError::LeafIdentityMismatch);
    }

    let mut fn_index = proof.leaf_index;
    let mut sn = proof.tree_size - 1;
    let mut root = proof.leaf_sha256.clone();
    for node in &proof.path {
        if sn == 0 {
            return Err(TransparencyProofError::InvalidProof);
        }
        if fn_index & 1 == 1 || fn_index == sn {
            root = merkle_node_hash(node, &root);
            if fn_index & 1 == 0 {
                while fn_index != 0 && fn_index & 1 == 0 {
                    fn_index >>= 1;
                    sn >>= 1;
                }
            }
        } else {
            root = merkle_node_hash(&root, node);
        }
        fn_index >>= 1;
        sn >>= 1;
    }
    if sn != 0 || root != proof.root_sha256 {
        return Err(TransparencyProofError::RootMismatch);
    }
    Ok(())
}

pub fn verify_transparency_consistency(
    proof: &TransparencyConsistencyProof,
) -> Result<(), TransparencyProofError> {
    if proof.first_tree_size == 0 || proof.first_tree_size > proof.second_tree_size {
        return Err(TransparencyProofError::InvalidTreeSizes);
    }
    if proof.path.len() > MAX_PROOF_NODES {
        return Err(TransparencyProofError::ProofTooDeep);
    }
    if proof.first_tree_size == proof.second_tree_size {
        if proof.path.is_empty() && proof.first_root_sha256 == proof.second_root_sha256 {
            return Ok(());
        }
        return Err(TransparencyProofError::InvalidProof);
    }
    if proof.path.is_empty() {
        return Err(TransparencyProofError::InvalidProof);
    }

    let mut path = proof.path.clone();
    if proof.first_tree_size.is_power_of_two() {
        path.insert(0, proof.first_root_sha256.clone());
    }
    let mut fn_index = proof.first_tree_size - 1;
    let mut sn = proof.second_tree_size - 1;
    while fn_index & 1 == 1 {
        fn_index >>= 1;
        sn >>= 1;
    }
    let mut first_root = path[0].clone();
    let mut second_root = path[0].clone();
    for node in path.iter().skip(1) {
        if sn == 0 {
            return Err(TransparencyProofError::InvalidProof);
        }
        if fn_index & 1 == 1 || fn_index == sn {
            first_root = merkle_node_hash(node, &first_root);
            second_root = merkle_node_hash(node, &second_root);
            if fn_index & 1 == 0 {
                while fn_index != 0 && fn_index & 1 == 0 {
                    fn_index >>= 1;
                    sn >>= 1;
                }
            }
        } else {
            second_root = merkle_node_hash(&second_root, node);
        }
        fn_index >>= 1;
        sn >>= 1;
    }
    if sn != 0
        || first_root != proof.first_root_sha256
        || second_root != proof.second_root_sha256
    {
        return Err(TransparencyProofError::RootMismatch);
    }
    Ok(())
}

fn entry_digest(
    sequence: u64,
    kind: &TrustUsage,
    subject_sha256: &Sha256Digest,
    payload_sha256: Option<&Sha256Digest>,
    context_sha256: Option<&Sha256Digest>,
    previous_entry_sha256: Option<&Sha256Digest>,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(ENTRY_DOMAIN);
    digest.text(TRANSPARENCY_ENTRY_SCHEMA);
    digest.text(&sequence.to_string());
    digest.text(kind.as_str());
    digest.text(subject_sha256.as_str());
    digest.optional_sha(payload_sha256);
    digest.optional_sha(context_sha256);
    digest.optional_sha(previous_entry_sha256);
    digest.digest()
}

fn merkle_leaf_hash(entry_sha256: &Sha256Digest) -> Sha256Digest {
    let mut digest = FramedDigest::new(LEAF_DOMAIN);
    digest.text(entry_sha256.as_str());
    digest.digest()
}

fn merkle_node_hash(left: &Sha256Digest, right: &Sha256Digest) -> Sha256Digest {
    let mut digest = FramedDigest::new(NODE_DOMAIN);
    digest.text(left.as_str());
    digest.text(right.as_str());
    digest.digest()
}

fn empty_root() -> Sha256Digest {
    FramedDigest::new(EMPTY_DOMAIN).digest()
}

fn merkle_tree_hash(leaves: &[Sha256Digest]) -> Sha256Digest {
    match leaves.len() {
        0 => empty_root(),
        1 => leaves[0].clone(),
        n => {
            let split = largest_power_of_two_less_than(n);
            merkle_node_hash(
                &merkle_tree_hash(&leaves[..split]),
                &merkle_tree_hash(&leaves[split..]),
            )
        }
    }
}

fn inclusion_path(index: usize, leaves: &[Sha256Digest]) -> Vec<Sha256Digest> {
    if leaves.len() <= 1 {
        return Vec::new();
    }
    let split = largest_power_of_two_less_than(leaves.len());
    if index < split {
        let mut path = inclusion_path(index, &leaves[..split]);
        path.push(merkle_tree_hash(&leaves[split..]));
        path
    } else {
        let mut path = inclusion_path(index - split, &leaves[split..]);
        path.push(merkle_tree_hash(&leaves[..split]));
        path
    }
}

fn consistency_subproof(
    first_tree_size: usize,
    leaves: &[Sha256Digest],
    complete_subtree: bool,
) -> Vec<Sha256Digest> {
    if first_tree_size == leaves.len() {
        return if complete_subtree {
            Vec::new()
        } else {
            vec![merkle_tree_hash(leaves)]
        };
    }
    let split = largest_power_of_two_less_than(leaves.len());
    if first_tree_size <= split {
        let mut proof = consistency_subproof(first_tree_size, &leaves[..split], complete_subtree);
        proof.push(merkle_tree_hash(&leaves[split..]));
        proof
    } else {
        let mut proof = consistency_subproof(first_tree_size - split, &leaves[split..], false);
        proof.push(merkle_tree_hash(&leaves[..split]));
        proof
    }
}

fn largest_power_of_two_less_than(value: usize) -> usize {
    debug_assert!(value > 1);
    let shift = usize::BITS - 1 - (value - 1).leading_zeros();
    1usize << shift
}

fn tree_head_digest(
    tree_size: u64,
    root_sha256: &Sha256Digest,
    last_entry_sha256: Option<&Sha256Digest>,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(TREE_HEAD_DOMAIN);
    digest.text(TRANSPARENCY_TREE_HEAD_SCHEMA);
    digest.text(&tree_size.to_string());
    digest.text(root_sha256.as_str());
    digest.optional_sha(last_entry_sha256);
    digest.text("structural-only");
    digest.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(value: &str) -> TrustUsage { TrustUsage::parse(value).unwrap() }
    fn sha(value: &str) -> Sha256Digest { Sha256Digest::of_bytes(value.as_bytes()) }
    fn draft(index: usize) -> TransparencyEntryDraft {
        TransparencyEntryDraft {
            kind: usage("science.qualification"),
            subject_sha256: sha(&format!("subject-{index}")),
            payload_sha256: Some(sha(&format!("payload-{index}"))),
            context_sha256: None,
        }
    }
    fn log(size: usize) -> TransparencyLog {
        let mut log = TransparencyLog::new();
        for index in 0..size { log.append(draft(index)).unwrap(); }
        log
    }

    #[test]
    fn entries_form_an_explicit_hash_chain() {
        let log = log(3);
        assert_eq!(log.entries()[0].previous_entry_sha256(), None);
        assert_eq!(
            log.entries()[1].previous_entry_sha256(),
            Some(log.entries()[0].entry_sha256())
        );
        assert_eq!(
            log.entries()[2].previous_entry_sha256(),
            Some(log.entries()[1].entry_sha256())
        );
    }

    #[test]
    fn every_entry_has_valid_inclusion_proof() {
        for size in 1..32 {
            let log = log(size);
            for index in 0..size {
                verify_transparency_inclusion(&log.inclusion_proof(index, size).unwrap()).unwrap();
            }
        }
    }

    #[test]
    fn every_prefix_pair_has_valid_compact_consistency_proof() {
        for second in 1..32 {
            let log = log(second);
            for first in 1..=second {
                let proof = log.consistency_proof(first, second).unwrap();
                verify_transparency_consistency(&proof).unwrap();
                assert!(proof.path().len() <= MAX_PROOF_NODES);
            }
        }
    }

    #[test]
    fn consistency_proof_is_logarithmically_bounded_for_reasonable_tree() {
        let log = log(1024);
        for first in [1, 3, 17, 511, 512, 777, 1023] {
            let proof = log.consistency_proof(first, 1024).unwrap();
            assert!(proof.path().len() <= 11);
            verify_transparency_consistency(&proof).unwrap();
        }
    }

    #[test]
    fn mutated_inclusion_leaf_is_rejected() {
        let log = log(7);
        let mut proof = log.inclusion_proof(3, 7).unwrap();
        proof.entry_sha256 = sha("substituted-entry");
        assert!(matches!(
            verify_transparency_inclusion(&proof),
            Err(TransparencyProofError::LeafIdentityMismatch)
        ));
    }

    #[test]
    fn mutated_consistency_root_is_rejected() {
        let log = log(13);
        let mut proof = log.consistency_proof(5, 13).unwrap();
        proof.first_root_sha256 = sha("substituted-old-root");
        assert!(verify_transparency_consistency(&proof).is_err());
    }

    #[test]
    fn same_size_consistency_is_empty_and_exact() {
        let log = log(8);
        let proof = log.consistency_proof(8, 8).unwrap();
        assert!(proof.path().is_empty());
        assert_eq!(proof.first_root_sha256(), proof.second_root_sha256());
        verify_transparency_consistency(&proof).unwrap();
    }

    #[test]
    fn tree_head_is_structural_not_authority() {
        let head = log(4).head();
        assert!(!head.operator_authority_established());
        assert!(!head.trusted_time_established());
    }
}
