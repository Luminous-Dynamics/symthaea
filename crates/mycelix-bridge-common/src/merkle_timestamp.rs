// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Merkle Timestamp Chain — court-defensible prior art proofs.
//!
//! Batches invention claim hashes into periodic Merkle roots, creating
//! cryptographic proof that any individual claim existed before the root's
//! timestamp.  Each root chains to its predecessor, forming an append-only
//! linked list of timestamp anchors.
//!
//! ## Court-defensibility
//!
//! A [`MerkleInclusionProof`] contains everything a third party needs to
//! independently verify that a specific [`MerkleLeaf`] was included in a
//! [`MerkleTimestampRoot`]:
//!
//! 1. The leaf data (invention ID, witness commitment, registration time).
//! 2. The sibling path from leaf to root.
//! 3. The root itself (hash, timestamp, sequence number, chain link).
//!
//! Verification requires only Blake3 and no access to any other leaves.

use serde::{Deserialize, Serialize};

// ============================================================================
// Types
// ============================================================================

/// A leaf in the Merkle tree — one invention claim's witness commitment.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MerkleLeaf {
    /// Unique identifier of the invention (references `InventionClaim.id`).
    pub invention_id: String,
    /// Blake3 hash of the claim content (from `InventionClaim.witness_commitment`).
    pub witness_commitment: [u8; 32],
    /// Timestamp when the claim was registered (microseconds).
    pub registered_at: u64,
}

/// A completed Merkle root anchoring a batch of invention claims.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MerkleTimestampRoot {
    /// Blake3 Merkle root hash.
    pub root_hash: [u8; 32],
    /// Number of leaves in this batch.
    pub leaf_count: usize,
    /// Timestamp when the root was computed (microseconds).
    pub created_at: u64,
    /// Monotonically increasing sequence number (0, 1, 2, ...).
    pub sequence_number: u64,
    /// Hash of the previous root, forming a chain.  `None` for the first root.
    pub previous_root: Option<[u8; 32]>,
}

/// Proof that a specific leaf is included in a Merkle root.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MerkleInclusionProof {
    /// The leaf whose inclusion is being proved.
    pub leaf: MerkleLeaf,
    /// Zero-based index of the leaf in the batch.
    pub leaf_index: usize,
    /// Sibling hashes along the path from leaf to root.
    pub siblings: Vec<MerkleSibling>,
    /// The root that anchors this proof.
    pub root: MerkleTimestampRoot,
}

/// A sibling node in the Merkle path.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MerkleSibling {
    /// Hash of the sibling node.
    pub hash: [u8; 32],
    /// Whether the sibling is to the left or right.
    pub position: SiblingPosition,
}

/// Position of a sibling in a Merkle pair.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SiblingPosition {
    Left,
    Right,
}

// ============================================================================
// Builder
// ============================================================================

/// Accumulates leaves and computes periodic Merkle timestamp roots.
///
/// Maintains a ring buffer of recent roots (default 256) so that proofs
/// can reference historical roots without unbounded memory growth.
pub struct MerkleTimestampBuilder {
    /// Pending leaves not yet committed to a root.
    leaves: Vec<MerkleLeaf>,
    /// Ring buffer of computed roots.
    roots: Vec<MerkleTimestampRoot>,
    /// Maximum number of roots to retain.
    max_roots: usize,
    /// Next sequence number to assign.
    next_sequence: u64,
}

impl MerkleTimestampBuilder {
    /// Create an empty builder with the default ring buffer size (256).
    pub fn new() -> Self {
        Self {
            leaves: Vec::new(),
            roots: Vec::new(),
            max_roots: 256,
            next_sequence: 0,
        }
    }

    /// Create a builder with a custom ring buffer size.
    pub fn with_max_roots(max: usize) -> Self {
        Self {
            leaves: Vec::new(),
            roots: Vec::new(),
            max_roots: max,
            next_sequence: 0,
        }
    }

    /// Add a claim leaf to the current pending batch.
    pub fn add_leaf(&mut self, leaf: MerkleLeaf) {
        self.leaves.push(leaf);
    }

    /// Number of pending (uncommitted) leaves.
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    /// Compute the Merkle root over all pending leaves, chain to the
    /// previous root, store in the ring buffer, and clear the pending batch.
    ///
    /// `now` is the timestamp to record on the root (microseconds).
    pub fn compute_root(&mut self, now: u64) -> MerkleTimestampRoot {
        let leaf_hashes: Vec<[u8; 32]> = self.leaves.iter().map(blake3_leaf).collect();
        let root_hash = merkle_root_from_hashes(&leaf_hashes);

        let previous_root = self.roots.last().map(|r| r.root_hash);
        let seq = self.next_sequence;
        self.next_sequence += 1;

        let root = MerkleTimestampRoot {
            root_hash,
            leaf_count: self.leaves.len(),
            created_at: now,
            sequence_number: seq,
            previous_root,
        };

        // Ring buffer: drop oldest if at capacity.
        if self.roots.len() >= self.max_roots {
            self.roots.remove(0);
        }
        self.roots.push(root.clone());

        self.leaves.clear();
        root
    }

    /// Generate an inclusion proof for a leaf identified by `invention_id`
    /// in the **current pending batch** (before `compute_root`).
    ///
    /// Returns `None` if no leaf with that ID exists in the pending batch.
    pub fn generate_proof(&self, invention_id: &str) -> Option<MerkleInclusionProof> {
        let leaf_index = self
            .leaves
            .iter()
            .position(|l| l.invention_id == invention_id)?;

        let leaf_hashes: Vec<[u8; 32]> = self.leaves.iter().map(blake3_leaf).collect();
        let siblings = merkle_siblings(&leaf_hashes, leaf_index);
        let root_hash = merkle_root_from_hashes(&leaf_hashes);

        let previous_root = self.roots.last().map(|r| r.root_hash);

        // Build a "prospective" root — the root that *would* be created if
        // compute_root were called right now.
        let root = MerkleTimestampRoot {
            root_hash,
            leaf_count: self.leaves.len(),
            created_at: 0, // Caller should set this when finalising.
            sequence_number: self.next_sequence,
            previous_root,
        };

        Some(MerkleInclusionProof {
            leaf: self.leaves[leaf_index].clone(),
            leaf_index,
            siblings,
            root,
        })
    }

    /// Verify an inclusion proof.  This is a **static** operation — it does
    /// not require access to the builder's state.
    pub fn verify_proof(proof: &MerkleInclusionProof) -> bool {
        let mut current = blake3_leaf(&proof.leaf);

        for sibling in &proof.siblings {
            current = match sibling.position {
                SiblingPosition::Left => blake3_hash_pair(&sibling.hash, &current),
                SiblingPosition::Right => blake3_hash_pair(&current, &sibling.hash),
            };
        }

        current == proof.root.root_hash
    }

    /// The most recently computed root, if any.
    pub fn latest_root(&self) -> Option<&MerkleTimestampRoot> {
        self.roots.last()
    }

    /// All stored roots (up to `max_roots`).
    pub fn root_history(&self) -> &[MerkleTimestampRoot] {
        &self.roots
    }

    /// Look up a root by its sequence number.
    pub fn find_root_by_sequence(&self, seq: u64) -> Option<&MerkleTimestampRoot> {
        self.roots.iter().find(|r| r.sequence_number == seq)
    }
}

impl Default for MerkleTimestampBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Blake3(left || right).
fn blake3_hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(left);
    hasher.update(right);
    *hasher.finalize().as_bytes()
}

/// Blake3(invention_id || witness_commitment || registered_at as LE bytes).
fn blake3_leaf(leaf: &MerkleLeaf) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(leaf.invention_id.as_bytes());
    hasher.update(&leaf.witness_commitment);
    hasher.update(&leaf.registered_at.to_le_bytes());
    *hasher.finalize().as_bytes()
}

/// Compute the Merkle root from a list of leaf hashes.
///
/// Empty list produces Blake3("").  Single hash returns itself.
/// Odd-length layers promote the last element.
fn merkle_root_from_hashes(hashes: &[[u8; 32]]) -> [u8; 32] {
    if hashes.is_empty() {
        return *blake3::hash(b"").as_bytes();
    }
    if hashes.len() == 1 {
        return hashes[0];
    }

    let mut layer = hashes.to_vec();
    while layer.len() > 1 {
        let mut next = Vec::with_capacity((layer.len() + 1) / 2);
        let mut i = 0;
        while i + 1 < layer.len() {
            next.push(blake3_hash_pair(&layer[i], &layer[i + 1]));
            i += 2;
        }
        // Odd element promoted.
        if i < layer.len() {
            next.push(layer[i]);
        }
        layer = next;
    }
    layer[0]
}

/// Compute the sibling path for the leaf at `index` in `hashes`.
fn merkle_siblings(hashes: &[[u8; 32]], index: usize) -> Vec<MerkleSibling> {
    if hashes.len() <= 1 {
        return Vec::new();
    }

    let mut siblings = Vec::new();
    let mut layer = hashes.to_vec();
    let mut idx = index;

    while layer.len() > 1 {
        let pair_start = idx & !1; // Round down to even.
        let sibling_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };

        if sibling_idx < layer.len() {
            let position = if idx % 2 == 0 {
                SiblingPosition::Right
            } else {
                SiblingPosition::Left
            };
            siblings.push(MerkleSibling {
                hash: layer[sibling_idx],
                position,
            });
        }
        // else: odd element promoted, no sibling at this level.

        // Build next layer.
        let mut next = Vec::with_capacity((layer.len() + 1) / 2);
        let mut i = 0;
        while i + 1 < layer.len() {
            next.push(blake3_hash_pair(&layer[i], &layer[i + 1]));
            i += 2;
        }
        if i < layer.len() {
            next.push(layer[i]);
        }

        idx = pair_start / 2;
        layer = next;
    }

    siblings
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_leaf(id: &str, ts: u64) -> MerkleLeaf {
        MerkleLeaf {
            invention_id: id.to_string(),
            witness_commitment: *blake3::hash(id.as_bytes()).as_bytes(),
            registered_at: ts,
        }
    }

    // ---- 1. Empty tree ----

    #[test]
    fn empty_tree_computes_deterministic_root() {
        let mut builder = MerkleTimestampBuilder::new();
        let root = builder.compute_root(1000);
        assert_eq!(root.leaf_count, 0);
        assert_eq!(root.root_hash, *blake3::hash(b"").as_bytes());
        assert_eq!(root.sequence_number, 0);
        assert!(root.previous_root.is_none());
    }

    // ---- 2. Single leaf — root equals leaf hash ----

    #[test]
    fn single_leaf_root_equals_leaf_hash() {
        let mut builder = MerkleTimestampBuilder::new();
        let leaf = make_leaf("inv-001", 100);
        let expected = blake3_leaf(&leaf);
        builder.add_leaf(leaf);

        let root = builder.compute_root(200);
        assert_eq!(root.leaf_count, 1);
        assert_eq!(root.root_hash, expected);
    }

    // ---- 3. Two leaves — root = blake3(leaf0 || leaf1) ----

    #[test]
    fn two_leaves_root_is_pair_hash() {
        let mut builder = MerkleTimestampBuilder::new();
        let l0 = make_leaf("inv-0", 10);
        let l1 = make_leaf("inv-1", 20);
        let h0 = blake3_leaf(&l0);
        let h1 = blake3_leaf(&l1);
        builder.add_leaf(l0);
        builder.add_leaf(l1);

        let root = builder.compute_root(30);
        assert_eq!(root.leaf_count, 2);
        assert_eq!(root.root_hash, blake3_hash_pair(&h0, &h1));
    }

    // ---- 4. Three leaves (odd) — proper handling ----

    #[test]
    fn three_leaves_odd_promotion() {
        let mut builder = MerkleTimestampBuilder::new();
        let l0 = make_leaf("a", 1);
        let l1 = make_leaf("b", 2);
        let l2 = make_leaf("c", 3);
        let h0 = blake3_leaf(&l0);
        let h1 = blake3_leaf(&l1);
        let h2 = blake3_leaf(&l2);
        builder.add_leaf(l0);
        builder.add_leaf(l1);
        builder.add_leaf(l2);

        // Layer 1: [hash(h0,h1), h2]  (h2 promoted)
        // Layer 2: hash(hash(h0,h1), h2)
        let expected = blake3_hash_pair(&blake3_hash_pair(&h0, &h1), &h2);
        let root = builder.compute_root(100);
        assert_eq!(root.root_hash, expected);
        assert_eq!(root.leaf_count, 3);
    }

    // ---- 5. Power-of-2 leaves (4 and 8) ----

    #[test]
    fn four_leaves_balanced() {
        let mut builder = MerkleTimestampBuilder::new();
        let leaves: Vec<_> = (0..4).map(|i| make_leaf(&format!("inv-{i}"), i as u64)).collect();
        let hashes: Vec<_> = leaves.iter().map(|l| blake3_leaf(l)).collect();
        for l in leaves {
            builder.add_leaf(l);
        }

        let p01 = blake3_hash_pair(&hashes[0], &hashes[1]);
        let p23 = blake3_hash_pair(&hashes[2], &hashes[3]);
        let expected = blake3_hash_pair(&p01, &p23);

        let root = builder.compute_root(100);
        assert_eq!(root.root_hash, expected);
        assert_eq!(root.leaf_count, 4);
    }

    #[test]
    fn eight_leaves_balanced() {
        let mut builder = MerkleTimestampBuilder::new();
        let leaves: Vec<_> = (0..8).map(|i| make_leaf(&format!("inv-{i}"), i as u64)).collect();
        let hashes: Vec<_> = leaves.iter().map(|l| blake3_leaf(l)).collect();
        for l in leaves {
            builder.add_leaf(l);
        }

        // Manually compute balanced tree.
        let l1: Vec<_> = hashes.chunks(2).map(|c| blake3_hash_pair(&c[0], &c[1])).collect();
        let l2: Vec<_> = l1.chunks(2).map(|c| blake3_hash_pair(&c[0], &c[1])).collect();
        let expected = blake3_hash_pair(&l2[0], &l2[1]);

        let root = builder.compute_root(100);
        assert_eq!(root.root_hash, expected);
    }

    // ---- 6. Inclusion proof generation + verification ----

    #[test]
    fn inclusion_proof_verifies_each_position() {
        let mut builder = MerkleTimestampBuilder::new();
        let ids: Vec<String> = (0..5).map(|i| format!("inv-{i}")).collect();
        for (i, id) in ids.iter().enumerate() {
            builder.add_leaf(make_leaf(id, i as u64 * 10));
        }

        for id in &ids {
            let proof = builder.generate_proof(id).expect("proof should exist");
            assert_eq!(&proof.leaf.invention_id, id);
            assert!(
                MerkleTimestampBuilder::verify_proof(&proof),
                "proof for {id} should verify"
            );
        }
    }

    #[test]
    fn proof_for_nonexistent_leaf_is_none() {
        let mut builder = MerkleTimestampBuilder::new();
        builder.add_leaf(make_leaf("inv-0", 1));
        assert!(builder.generate_proof("inv-999").is_none());
    }

    // ---- 7. Tampered proof fails verification ----

    #[test]
    fn tampered_sibling_fails_verification() {
        let mut builder = MerkleTimestampBuilder::new();
        builder.add_leaf(make_leaf("inv-0", 1));
        builder.add_leaf(make_leaf("inv-1", 2));

        let mut proof = builder.generate_proof("inv-0").unwrap();
        assert!(MerkleTimestampBuilder::verify_proof(&proof));

        // Flip a bit in the first sibling hash.
        if let Some(sibling) = proof.siblings.first_mut() {
            sibling.hash[0] ^= 0x01;
        }
        assert!(!MerkleTimestampBuilder::verify_proof(&proof));
    }

    #[test]
    fn tampered_leaf_fails_verification() {
        let mut builder = MerkleTimestampBuilder::new();
        builder.add_leaf(make_leaf("inv-0", 1));
        builder.add_leaf(make_leaf("inv-1", 2));

        let mut proof = builder.generate_proof("inv-0").unwrap();
        proof.leaf.registered_at = 999; // Tamper with the leaf.
        assert!(!MerkleTimestampBuilder::verify_proof(&proof));
    }

    // ---- 8. Root chaining ----

    #[test]
    fn root_chaining() {
        let mut builder = MerkleTimestampBuilder::new();
        builder.add_leaf(make_leaf("a", 1));
        let root1 = builder.compute_root(100);
        assert!(root1.previous_root.is_none());

        builder.add_leaf(make_leaf("b", 2));
        let root2 = builder.compute_root(200);
        assert_eq!(root2.previous_root, Some(root1.root_hash));
    }

    // ---- 9. Sequence numbers increment ----

    #[test]
    fn sequence_numbers_increment() {
        let mut builder = MerkleTimestampBuilder::new();
        for i in 0..5u64 {
            builder.add_leaf(make_leaf(&format!("inv-{i}"), i));
            let root = builder.compute_root(i * 100);
            assert_eq!(root.sequence_number, i);
        }
    }

    // ---- 10. Ring buffer — oldest dropped ----

    #[test]
    fn ring_buffer_drops_oldest() {
        let max = 4;
        let mut builder = MerkleTimestampBuilder::with_max_roots(max);

        for i in 0..6u64 {
            builder.add_leaf(make_leaf(&format!("inv-{i}"), i));
            builder.compute_root(i * 100);
        }

        let history = builder.root_history();
        assert_eq!(history.len(), max);
        // Oldest surviving root should be sequence 2 (0 and 1 dropped).
        assert_eq!(history[0].sequence_number, 2);
        assert_eq!(history[3].sequence_number, 5);
    }

    // ---- 11. Serde roundtrip ----

    #[test]
    fn serde_roundtrip_root() {
        let mut builder = MerkleTimestampBuilder::new();
        builder.add_leaf(make_leaf("inv-0", 1));
        let root = builder.compute_root(100);

        let json = serde_json::to_string(&root).unwrap();
        let deser: MerkleTimestampRoot = serde_json::from_str(&json).unwrap();
        assert_eq!(root, deser);
    }

    #[test]
    fn serde_roundtrip_proof() {
        let mut builder = MerkleTimestampBuilder::new();
        builder.add_leaf(make_leaf("inv-0", 1));
        builder.add_leaf(make_leaf("inv-1", 2));
        builder.add_leaf(make_leaf("inv-2", 3));

        let proof = builder.generate_proof("inv-1").unwrap();
        let json = serde_json::to_string(&proof).unwrap();
        let deser: MerkleInclusionProof = serde_json::from_str(&json).unwrap();
        assert_eq!(proof, deser);
        assert!(MerkleTimestampBuilder::verify_proof(&deser));
    }

    // ---- 12. Large batch — 100 leaves, all proofs verify ----

    #[test]
    fn large_batch_all_proofs_verify() {
        let mut builder = MerkleTimestampBuilder::new();
        let ids: Vec<String> = (0..100).map(|i| format!("inv-{i:04}")).collect();
        for (i, id) in ids.iter().enumerate() {
            builder.add_leaf(make_leaf(id, i as u64));
        }

        for id in &ids {
            let proof = builder.generate_proof(id).expect("proof should exist");
            assert!(
                MerkleTimestampBuilder::verify_proof(&proof),
                "proof for {id} should verify"
            );
        }
    }

    // ---- 13. find_root_by_sequence ----

    #[test]
    fn find_root_by_sequence_works() {
        let mut builder = MerkleTimestampBuilder::new();
        for i in 0..5u64 {
            builder.add_leaf(make_leaf(&format!("inv-{i}"), i));
            builder.compute_root(i * 100);
        }

        for seq in 0..5u64 {
            let root = builder.find_root_by_sequence(seq);
            assert!(root.is_some(), "sequence {seq} should exist");
            assert_eq!(root.unwrap().sequence_number, seq);
        }

        assert!(builder.find_root_by_sequence(99).is_none());
    }

    // ---- Additional: latest_root ----

    #[test]
    fn latest_root_tracks_most_recent() {
        let mut builder = MerkleTimestampBuilder::new();
        assert!(builder.latest_root().is_none());

        builder.add_leaf(make_leaf("a", 1));
        let r1 = builder.compute_root(100);
        assert_eq!(builder.latest_root().unwrap().root_hash, r1.root_hash);

        builder.add_leaf(make_leaf("b", 2));
        let r2 = builder.compute_root(200);
        assert_eq!(builder.latest_root().unwrap().root_hash, r2.root_hash);
    }

    // ---- Additional: leaves cleared after compute_root ----

    #[test]
    fn leaves_cleared_after_compute_root() {
        let mut builder = MerkleTimestampBuilder::new();
        builder.add_leaf(make_leaf("a", 1));
        assert_eq!(builder.leaf_count(), 1);

        builder.compute_root(100);
        assert_eq!(builder.leaf_count(), 0);
    }

    // ---- Additional: proof on single-leaf batch ----

    #[test]
    fn proof_single_leaf_batch() {
        let mut builder = MerkleTimestampBuilder::new();
        builder.add_leaf(make_leaf("solo", 42));

        let proof = builder.generate_proof("solo").unwrap();
        assert!(proof.siblings.is_empty());
        assert!(MerkleTimestampBuilder::verify_proof(&proof));
    }
}
