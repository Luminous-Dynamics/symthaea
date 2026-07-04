// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Nullifier-based membership proof circuit.
//!
//! Proves: "I am a member of group G" without revealing:
//! - Which specific member I am
//! - Who else is in the group
//! - How many members there are
//!
//! Additionally prevents double-use: the nullifier is deterministic
//! (hash of secret + group_id), so using the same proof twice is detectable.
//!
//! Used for:
//! - Care circle membership (prove "I'm in a support circle" anonymously)
//! - Anonymous voting (prove eligibility, prevent double-voting)
//! - Credential presentation (prove "I hold a credential" without revealing which)

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A nullifier-based membership proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NullifierProof {
    /// Deterministic nullifier: hash(member_secret, group_id)
    /// Reveals nothing about identity, but prevents double-use
    pub nullifier: [u8; 32],
    /// Membership Merkle root (commitment to the group member set)
    pub group_root: [u8; 32],
    /// Proof bytes (STARK/Binius proof)
    pub proof_bytes: Vec<u8>,
    /// Domain tag
    pub domain_tag: String,
}

/// Compute a nullifier from a secret and group identifier.
///
/// The nullifier is deterministic: same (secret, group_id) always produces
/// the same nullifier. This prevents double-use (voting twice, claiming twice)
/// while hiding the member's identity.
pub fn compute_nullifier(member_secret: &[u8], group_id: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"ZTML:nullifier:v1:");
    hasher.update(member_secret);
    hasher.update(b":");
    hasher.update(group_id);
    let result = hasher.finalize();
    let mut nullifier = [0u8; 32];
    nullifier.copy_from_slice(&result);
    nullifier
}

/// Compute a member's leaf hash for the group Merkle tree.
pub fn compute_member_leaf(member_secret: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"ZTML:member:v1:");
    hasher.update(member_secret);
    let result = hasher.finalize();
    let mut leaf = [0u8; 32];
    leaf.copy_from_slice(&result);
    leaf
}

/// Validate nullifier proof structure.
pub fn validate_nullifier_proof(proof: &NullifierProof) -> Result<(), String> {
    if proof.proof_bytes.is_empty() {
        return Err("Empty proof bytes".to_string());
    }
    if proof.nullifier == [0u8; 32] {
        return Err("Zero nullifier".to_string());
    }
    if proof.group_root == [0u8; 32] {
        return Err("Zero group root".to_string());
    }
    if !proof.domain_tag.starts_with("ZTML:") {
        return Err("Invalid domain tag".to_string());
    }
    Ok(())
}

/// Check if a nullifier has been used before (simple in-memory set).
/// Production: check against DHT-stored nullifier set.
pub struct NullifierSet {
    used: std::collections::HashSet<[u8; 32]>,
}

impl NullifierSet {
    pub fn new() -> Self {
        Self {
            used: std::collections::HashSet::new(),
        }
    }

    /// Check if nullifier was already used. Returns true if NEW (first use).
    pub fn check_and_mark(&mut self, nullifier: &[u8; 32]) -> bool {
        self.used.insert(*nullifier)
    }

    /// Check without marking.
    pub fn is_used(&self, nullifier: &[u8; 32]) -> bool {
        self.used.contains(nullifier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuits::merkle_membership::{
        compute_merkle_root, generate_merkle_path, verify_merkle_path,
    };

    #[test]
    fn test_nullifier_deterministic() {
        let n1 = compute_nullifier(b"alice_secret", b"care_circle_42");
        let n2 = compute_nullifier(b"alice_secret", b"care_circle_42");
        assert_eq!(n1, n2, "same inputs must produce same nullifier");
    }

    #[test]
    fn test_different_secrets_different_nullifiers() {
        let n1 = compute_nullifier(b"alice_secret", b"circle_1");
        let n2 = compute_nullifier(b"bob_secret", b"circle_1");
        assert_ne!(n1, n2, "different members must have different nullifiers");
    }

    #[test]
    fn test_different_groups_different_nullifiers() {
        let n1 = compute_nullifier(b"alice_secret", b"circle_1");
        let n2 = compute_nullifier(b"alice_secret", b"circle_2");
        assert_ne!(
            n1, n2,
            "same member in different groups must have different nullifiers"
        );
    }

    #[test]
    fn test_nullifier_with_merkle_membership() {
        // 4 members in a care circle
        let members: Vec<&[u8]> = vec![
            b"alice_secret",
            b"bob_secret",
            b"carol_secret",
            b"dave_secret",
        ];
        let leaves: Vec<[u8; 32]> = members.iter().map(|s| compute_member_leaf(*s)).collect();
        let root = compute_merkle_root(&leaves);

        // Alice proves membership
        let alice_leaf = compute_member_leaf(b"alice_secret");
        let alice_path = generate_merkle_path(&leaves, 0);
        assert!(verify_merkle_path(&alice_leaf, &alice_path, &root));

        // Alice generates nullifier for this circle
        let nullifier = compute_nullifier(b"alice_secret", b"care_circle_42");
        assert_ne!(nullifier, [0u8; 32]);

        // Combined proof: Merkle path + nullifier
        let proof = NullifierProof {
            nullifier,
            group_root: root,
            proof_bytes: vec![1, 2, 3], // Placeholder
            domain_tag: "ZTML:Hearth:CircleMembership:v1".to_string(),
        };
        assert!(validate_nullifier_proof(&proof).is_ok());
    }

    #[test]
    fn test_double_use_prevention() {
        let mut nullifier_set = NullifierSet::new();
        let n = compute_nullifier(b"alice", b"vote_proposal_1");

        // First use: accepted
        assert!(
            nullifier_set.check_and_mark(&n),
            "first use should be accepted"
        );

        // Second use: rejected
        assert!(
            !nullifier_set.check_and_mark(&n),
            "second use should be rejected"
        );
        assert!(nullifier_set.is_used(&n));
    }

    #[test]
    fn test_non_member_cannot_generate_valid_path() {
        let members: Vec<&[u8]> = vec![b"alice", b"bob", b"carol"];
        let leaves: Vec<[u8; 32]> = members.iter().map(|s| compute_member_leaf(*s)).collect();
        let root = compute_merkle_root(&leaves);

        // Eve (not a member) tries to forge a path
        let eve_leaf = compute_member_leaf(b"eve_secret");
        let alice_path = generate_merkle_path(&leaves, 0);

        // Eve's leaf with Alice's path should fail
        assert!(
            !verify_merkle_path(&eve_leaf, &alice_path, &root),
            "non-member must not verify"
        );
    }

    #[test]
    fn test_proof_structure_validation() {
        let valid = NullifierProof {
            nullifier: [0xAA; 32],
            group_root: [0xBB; 32],
            proof_bytes: vec![1],
            domain_tag: "ZTML:Test:v1".to_string(),
        };
        assert!(validate_nullifier_proof(&valid).is_ok());

        let zero_null = NullifierProof {
            nullifier: [0; 32],
            ..valid.clone()
        };
        assert!(validate_nullifier_proof(&zero_null).is_err());
    }
}
