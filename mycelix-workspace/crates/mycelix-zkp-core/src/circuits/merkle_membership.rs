// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Merkle tree membership proof circuit.
//!
//! Proves: "I know a leaf that exists in the Merkle tree with root R"
//! without revealing which leaf or its index.
//!
//! Used for:
//! - Property ownership: prove "I own a property" without revealing which one
//! - Care circle membership: prove "I'm in a circle" without revealing which
//! - Credential wallet: prove "I hold a credential" without revealing it
//!
//! ## Circuit Design
//!
//! For a tree of depth D, the proof verifies D hash steps from leaf to root.
//! Each hash step: parent = hash(left, right) where one child is the witness
//! sibling and the other is computed from the path.
//!
//! In Binius: XOR in hash is free, AND is the only cost.
//! Simple hash: H(a,b) = (a AND b) XOR (a XOR b) XOR constant
//! Production: SHA-256 (~16,000 AND per call, well-studied in Binius)

use serde::{Deserialize, Serialize};

/// Merkle membership proof — portable across backends.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerkleMembershipProof {
    /// The Merkle root (public — everyone knows this)
    pub root: [u8; 32],
    /// Proof bytes (STARK proof that the path is valid)
    pub proof_bytes: Vec<u8>,
    /// Domain tag for replay prevention
    pub domain_tag: String,
    /// Tree depth
    pub depth: u8,
}

/// Verify a Merkle membership proof structurally.
///
/// Full STARK verification happens in the Binius/Winterfell circuit.
/// This function validates the proof format.
pub fn validate_membership_proof(proof: &MerkleMembershipProof) -> Result<(), String> {
    if proof.proof_bytes.is_empty() {
        return Err("Empty proof bytes".to_string());
    }
    if proof.root == [0u8; 32] {
        return Err("Zero root".to_string());
    }
    if proof.depth == 0 || proof.depth > 32 {
        return Err(format!("Invalid depth: {} (must be 1-32)", proof.depth));
    }
    if !proof.domain_tag.starts_with("ZTML:") {
        return Err("Invalid domain tag".to_string());
    }
    Ok(())
}

/// Compute a simple Merkle root from leaves (for testing).
///
/// Uses SHA-256 as the hash function.
pub fn compute_merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    use sha2::{Digest, Sha256};

    if leaves.is_empty() {
        return [0u8; 32];
    }
    if leaves.len() == 1 {
        return leaves[0];
    }

    // Pad to power of 2
    let mut level: Vec<[u8; 32]> = leaves.to_vec();
    while level.len().count_ones() != 1 {
        level.push([0u8; 32]);
    }

    // Build tree bottom-up
    while level.len() > 1 {
        let mut next_level = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks(2) {
            let mut hasher = Sha256::new();
            hasher.update(&pair[0]);
            hasher.update(&pair[1]);
            let hash = hasher.finalize();
            let mut node = [0u8; 32];
            node.copy_from_slice(&hash);
            next_level.push(node);
        }
        level = next_level;
    }

    level[0]
}

/// Generate a Merkle path for a specific leaf index.
pub fn generate_merkle_path(leaves: &[[u8; 32]], leaf_index: usize) -> Vec<([u8; 32], bool)> {
    use sha2::{Digest, Sha256};

    let mut padded: Vec<[u8; 32]> = leaves.to_vec();
    while padded.len().count_ones() != 1 {
        padded.push([0u8; 32]);
    }

    let mut path = Vec::new();
    let mut idx = leaf_index;
    let mut level = padded;

    while level.len() > 1 {
        let sibling_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
        let is_left = idx % 2 == 0;
        path.push((level[sibling_idx], is_left));

        let mut next_level = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks(2) {
            let mut hasher = Sha256::new();
            hasher.update(&pair[0]);
            hasher.update(&pair[1]);
            let hash = hasher.finalize();
            let mut node = [0u8; 32];
            node.copy_from_slice(&hash);
            next_level.push(node);
        }
        idx /= 2;
        level = next_level;
    }

    path
}

/// Verify a Merkle path leads to the expected root.
pub fn verify_merkle_path(
    leaf: &[u8; 32],
    path: &[([u8; 32], bool)],
    expected_root: &[u8; 32],
) -> bool {
    use sha2::{Digest, Sha256};

    let mut current = *leaf;

    for (sibling, is_left) in path {
        let mut hasher = Sha256::new();
        if *is_left {
            hasher.update(&current);
            hasher.update(sibling);
        } else {
            hasher.update(sibling);
            hasher.update(&current);
        }
        let hash = hasher.finalize();
        current.copy_from_slice(&hash);
    }

    current == *expected_root
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn hash_leaf(data: &[u8]) -> [u8; 32] {
        let h = Sha256::digest(data);
        let mut leaf = [0u8; 32];
        leaf.copy_from_slice(&h);
        leaf
    }

    #[test]
    fn test_merkle_root_single_leaf() {
        let leaf = hash_leaf(b"property:123");
        let root = compute_merkle_root(&[leaf]);
        assert_eq!(root, leaf);
    }

    #[test]
    fn test_merkle_root_two_leaves() {
        let a = hash_leaf(b"property:1");
        let b = hash_leaf(b"property:2");
        let root = compute_merkle_root(&[a, b]);
        assert_ne!(root, a);
        assert_ne!(root, b);
    }

    #[test]
    fn test_merkle_path_verification() {
        let leaves: Vec<[u8; 32]> = (0..8)
            .map(|i| hash_leaf(format!("property:{}", i).as_bytes()))
            .collect();

        let root = compute_merkle_root(&leaves);

        // Verify path for each leaf
        for i in 0..8 {
            let path = generate_merkle_path(&leaves, i);
            assert!(
                verify_merkle_path(&leaves[i], &path, &root),
                "Path verification failed for leaf {}",
                i
            );
        }
    }

    #[test]
    fn test_wrong_leaf_fails() {
        let leaves: Vec<[u8; 32]> = (0..4)
            .map(|i| hash_leaf(format!("property:{}", i).as_bytes()))
            .collect();

        let root = compute_merkle_root(&leaves);
        let path = generate_merkle_path(&leaves, 0);

        // Wrong leaf should fail
        let wrong_leaf = hash_leaf(b"property:999");
        assert!(!verify_merkle_path(&wrong_leaf, &path, &root));
    }

    #[test]
    fn test_proof_structure_validation() {
        let valid = MerkleMembershipProof {
            root: [0xAA; 32],
            proof_bytes: vec![1, 2, 3],
            domain_tag: "ZTML:Commons:PropertyOwnership:v1".to_string(),
            depth: 8,
        };
        assert!(validate_membership_proof(&valid).is_ok());

        let empty_proof = MerkleMembershipProof {
            proof_bytes: vec![],
            ..valid.clone()
        };
        assert!(validate_membership_proof(&empty_proof).is_err());

        let zero_root = MerkleMembershipProof {
            root: [0; 32],
            ..valid.clone()
        };
        assert!(validate_membership_proof(&zero_root).is_err());
    }

    #[test]
    fn test_large_tree() {
        // 256 properties — realistic registry size
        let leaves: Vec<[u8; 32]> = (0..256)
            .map(|i| hash_leaf(format!("property:{}", i).as_bytes()))
            .collect();

        let root = compute_merkle_root(&leaves);

        // Verify random leaf
        let path = generate_merkle_path(&leaves, 137);
        assert!(verify_merkle_path(&leaves[137], &path, &root));
        assert_eq!(path.len(), 8); // log2(256) = 8 levels
    }
}
