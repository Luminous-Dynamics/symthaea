// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit boundary between local history witnesses and VDS consistency proofs.
//!
//! The local chained EvidenceHistory is deliberately not treated as a Merkle
//! VDS. This module also contains a concrete RFC 9162 SHA-256 verifier over an
//! independent ordered leaf sequence. Callers must explicitly map evidence
//! records into VDS leaves.

use crate::semantic_evidence_history::HistoryCheckpoint;
use sha2::{Digest, Sha256};

pub const VERSION: u16 = 1;
pub const DOMAIN: &[u8] = b"symthaea-swarm/semantic-evidence-vds";
pub const RFC9162_VDS_NAME: &str = "RFC9162_SHA256";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsistencyStatus {
    Valid,
    Invalid,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConsistencyRequest {
    older: HistoryCheckpoint,
    newer: HistoryCheckpoint,
}

impl ConsistencyRequest {
    pub fn new(older: HistoryCheckpoint, newer: HistoryCheckpoint) -> Self { Self { older, newer } }
    pub fn older(&self) -> HistoryCheckpoint { self.older }
    pub fn newer(&self) -> HistoryCheckpoint { self.newer }
    pub fn is_strict_extension_request(&self) -> bool { self.older.length() < self.newer.length() }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsistencyProof {
    vds: &'static str,
    version: u16,
    bytes: Vec<u8>,
}

impl ConsistencyProof {
    pub fn new(vds: &'static str, version: u16, bytes: Vec<u8>) -> Self {
        Self { vds, version, bytes }
    }
    pub fn vds(&self) -> &'static str { self.vds }
    pub fn version(&self) -> u16 { self.version }
    pub fn bytes(&self) -> &[u8] { &self.bytes }
}

pub trait HistoryVds {
    fn vds_name(&self) -> &'static str;
    fn prove_consistency(&self, _request: &ConsistencyRequest) -> Result<ConsistencyProof, ConsistencyError> {
        Err(ConsistencyError::Unsupported)
    }
    fn verify_consistency(&self, request: &ConsistencyRequest, proof: &ConsistencyProof) -> ConsistencyStatus;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ConsistencyError {
    #[error("this VDS adapter does not support consistency proofs")]
    Unsupported,
    #[error("a consistency proof cannot be generated for the supplied request")]
    CannotGenerate,
}

/// The current chained local witness remains explicitly outside the VDS layer.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChainedHistoryVds;

impl HistoryVds for ChainedHistoryVds {
    fn vds_name(&self) -> &'static str { "symthaea-chained-history-v1" }
    fn verify_consistency(
        &self,
        _request: &ConsistencyRequest,
        _proof: &ConsistencyProof,
    ) -> ConsistencyStatus {
        ConsistencyStatus::Unsupported
    }
}

/// RFC 9162 consistency proof represented in semantic form.
///
/// RFC 9942 maps this to CBOR as [old_size, new_size, consistency_path].
/// Encoding is intentionally separate from this verifier so it can later be
/// bound to the RFC 9942 receipt layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rfc9162ConsistencyProof {
    pub first: u64,
    pub second: u64,
    pub consistency_path: Vec<[u8; 32]>,
}

impl Rfc9162ConsistencyProof {
    pub fn new(first: u64, second: u64, consistency_path: Vec<[u8; 32]>) -> Self {
        Self { first, second, consistency_path }
    }
}

/// Concrete RFC 9162 SHA-256 Merkle VDS operations.
///
/// This VDS consumes an explicit ordered leaf sequence. It does not consume
/// HistoryCheckpoint because the local chained witness has different semantics.
#[derive(Debug, Clone, Copy, Default)]
pub struct Rfc9162Sha256Vds;

impl Rfc9162Sha256Vds {
    pub fn vds_name(&self) -> &'static str { RFC9162_VDS_NAME }
    pub fn root(&self, leaves: &[Vec<u8>]) -> [u8; 32] { merkle_tree_hash(leaves) }

    /// Generate the RFC 9162 minimal consistency proof for the first `first`
    /// leaves of the supplied ordered sequence.
    pub fn prove(&self, leaves: &[Vec<u8>], first: usize) -> Option<Rfc9162ConsistencyProof> {
        if first == 0 || first >= leaves.len() { return None; }
        let path = consistency_subproof(first, leaves, true);
        Some(Rfc9162ConsistencyProof::new(first as u64, leaves.len() as u64, path))
    }

    pub fn verify(
        &self,
        first_root: [u8; 32],
        second_root: [u8; 32],
        proof: &Rfc9162ConsistencyProof,
    ) -> bool {
        verify_rfc9162_consistency(first_root, second_root, proof)
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn leaf_hash(data: &[u8]) -> [u8; 32] {
    let mut input = Vec::with_capacity(1 + data.len());
    input.push(0x00);
    input.extend_from_slice(data);
    sha256(&input)
}

fn node_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut input = [0u8; 65];
    input[0] = 0x01;
    input[1..33].copy_from_slice(left);
    input[33..].copy_from_slice(right);
    sha256(&input)
}

fn merkle_tree_hash(leaves: &[Vec<u8>]) -> [u8; 32] {
    match leaves.len() {
        0 => sha256(&[]),
        1 => leaf_hash(&leaves[0]),
        n => {
            let k = largest_power_of_two_less_than(n);
            let left = merkle_tree_hash(&leaves[..k].to_vec());
            let right = merkle_tree_hash(&leaves[k..].to_vec());
            node_hash(&left, &right)
        }
    }
}

fn largest_power_of_two_less_than(n: usize) -> usize {
    debug_assert!(n > 1);
    let highest = 1usize << (usize::BITS - 1 - n.leading_zeros());
    if highest == n { highest >> 1 } else { highest }
}

fn consistency_subproof(m: usize, leaves: &[Vec<u8>], complete: bool) -> Vec<[u8; 32]> {
    if m == leaves.len() {
        return if complete { Vec::new() } else { vec![merkle_tree_hash(leaves)] };
    }

    let k = largest_power_of_two_less_than(leaves.len());
    let mut proof = if m <= k {
        consistency_subproof(m, &leaves[..k], complete)
    } else {
        consistency_subproof(m - k, &leaves[k..], false)
    };

    if m <= k {
        proof.push(merkle_tree_hash(&leaves[k..].to_vec()));
    } else {
        proof.push(merkle_tree_hash(&leaves[..k].to_vec()));
    }
    proof
}

fn verify_rfc9162_consistency(
    first_root: [u8; 32],
    second_root: [u8; 32],
    proof: &Rfc9162ConsistencyProof,
) -> bool {
    if proof.first == 0 || proof.first >= proof.second || proof.consistency_path.is_empty() {
        return false;
    }

    if proof.first.is_power_of_two() {
        let mut path = Vec::with_capacity(proof.consistency_path.len() + 1);
        path.push(first_root);
        path.extend_from_slice(&proof.consistency_path);
        verify_consistency_path(first_root, second_root, proof.first, proof.second, &path)
    } else {
        verify_consistency_path(
            first_root,
            second_root,
            proof.first,
            proof.second,
            &proof.consistency_path,
        )
    }
}

fn verify_consistency_path(
    first_root: [u8; 32],
    second_root: [u8; 32],
    first: u64,
    second: u64,
    path: &[[u8; 32]],
) -> bool {
    let mut fn_ = first - 1;
    let mut sn = second - 1;

    if fn_ & 1 == 1 {
        while fn_ & 1 == 1 {
            fn_ >>= 1;
            sn >>= 1;
        }
    }

    let Some(first_node) = path.first().copied() else { return false };
    let mut fr = first_node;
    let mut sr = first_node;

    for c in &path[1..] {
        if sn == 0 { return false; }

        if (fn_ & 1) == 1 || fn_ == sn {
            fr = node_hash(c, &fr);
            sr = node_hash(c, &sr);

            if fn_ & 1 == 0 {
                while fn_ & 1 == 0 && fn_ != 0 {
                    fn_ >>= 1;
                    sn >>= 1;
                }
            }
        } else {
            sr = node_hash(&sr, c);
        }

        fn_ >>= 1;
        sn >>= 1;
    }

    fr == first_root && sr == second_root && sn == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc9162_root_is_deterministic_and_order_sensitive() {
        let vds = Rfc9162Sha256Vds;
        let a = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
        let b = vec![b"b".to_vec(), b"a".to_vec(), b"c".to_vec()];
        assert_eq!(vds.root(&a), vds.root(&a));
        assert_ne!(vds.root(&a), vds.root(&b));
    }

    #[test]
    fn empty_and_singleton_roots_are_distinct() {
        let vds = Rfc9162Sha256Vds;
        assert_ne!(vds.root(&[]), vds.root(&[b"a".to_vec()]));
    }

    #[test]
    fn known_two_leaf_root_matches_definition() {
        let vds = Rfc9162Sha256Vds;
        let leaves = vec![b"a".to_vec(), b"b".to_vec()];
        let expected = node_hash(&leaf_hash(b"a"), &leaf_hash(b"b"));
        assert_eq!(vds.root(&leaves), expected);
    }

    #[test]
    fn generated_consistency_proofs_round_trip() {
        let vds = Rfc9162Sha256Vds;
        for n in 2..=12 {
            let leaves: Vec<Vec<u8>> = (0..n).map(|i| format!("leaf-{i}").into_bytes()).collect();
            let new_root = vds.root(&leaves);
            for first in 1..n {
                let proof = vds.prove(&leaves, first).expect("valid proof request");
                let old_root = vds.root(&leaves[..first].to_vec());
                assert!(vds.verify(old_root, new_root, &proof), "n={n}, first={first}");
            }
        }
    }

    #[test]
    fn malformed_consistency_proof_is_rejected() {
        let vds = Rfc9162Sha256Vds;
        let old = vds.root(&[b"a".to_vec()]);
        let new = vds.root(&[b"a".to_vec(), b"b".to_vec()]);
        let proof = Rfc9162ConsistencyProof::new(0, 2, vec![[0; 32]]);
        assert!(!vds.verify(old, new, &proof));
    }

    #[test]
    fn chained_history_remains_explicitly_unsupported() {
        let adapter = ChainedHistoryVds;
        let proof = ConsistencyProof::new(adapter.vds_name(), VERSION, Vec::new());
        let history = crate::semantic_evidence_history::EvidenceHistory::new();
        let checkpoint = history.checkpoint();
        let request = ConsistencyRequest::new(checkpoint, checkpoint);
        assert_eq!(
            adapter.verify_consistency(&request, &proof),
            ConsistencyStatus::Unsupported
        );
    }

    #[test]
    fn unsupported_is_not_invalid() {
        assert_ne!(ConsistencyStatus::Unsupported, ConsistencyStatus::Invalid);
        assert_ne!(ConsistencyStatus::Unsupported, ConsistencyStatus::Valid);
    }
}
