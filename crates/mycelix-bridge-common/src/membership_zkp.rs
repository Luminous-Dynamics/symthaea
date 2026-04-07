// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Privacy-preserving membership verification for cross-cluster bridges.
//!
//! Replaces patterns where member lists or ownership details are sent
//! in plaintext across cluster boundaries.
//!
//! ## Usage
//!
//! Instead of: `verify_property_ownership() → { is_owner, owner_did }`
//! Now:        `verify_property_ownership() → { is_owner, merkle_proof }`
//!
//! Instead of: `get_circle_members() → Vec<CircleMembership>`
//! Now:        `prove_circle_membership() → NullifierMembershipProof`

use serde::{Deserialize, Serialize};

/// Merkle membership proof for cross-cluster ownership verification.
///
/// Proves "entity exists in registry" without revealing which entity.
/// Used for: property ownership, credential wallet, resource registry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MembershipProof {
    /// Merkle root of the registry (public commitment)
    pub registry_root: Vec<u8>,
    /// Merkle path (sibling hashes from leaf to root)
    pub merkle_path: Vec<Vec<u8>>,
    /// Domain tag for replay prevention
    pub domain_tag: String,
    /// Whether the membership was verified
    pub verified: bool,
}

/// Nullifier-based anonymous membership proof.
///
/// Proves "I am a member" without revealing identity or other members.
/// Prevents double-use via deterministic nullifier.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnonymousMembershipProof {
    /// Deterministic nullifier: hash(member_secret, group_id)
    pub nullifier: Vec<u8>,
    /// Merkle root of the member set
    pub group_root: Vec<u8>,
    /// STARK proof bytes (optional — off-chain verification)
    pub proof_bytes: Vec<u8>,
    /// Domain tag
    pub domain_tag: String,
    /// Whether the proof was verified
    pub verified: bool,
}

/// Privacy-preserving property ownership response.
///
/// Replaces: `PropertyOwnershipResult { is_owner, owner_did }`
/// With:     `PrivateOwnershipResult { is_owner, proof }`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrivateOwnershipResult {
    /// Whether the queried agent is the owner
    pub is_owner: bool,
    /// Merkle proof of ownership (no identity revealed)
    pub proof: Option<MembershipProof>,
    /// Error message if verification failed
    pub error: Option<String>,
}

/// Privacy-preserving circle membership verification.
///
/// Replaces: `Vec<CircleMembership>` with all member keys
/// With:     `CircleMembershipVerification` with only proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircleMembershipVerification {
    /// Whether the queried agent is a member
    pub is_member: bool,
    /// Anonymous proof (nullifier + Merkle)
    pub proof: Option<AnonymousMembershipProof>,
    /// Error message
    pub error: Option<String>,
}

/// Validate membership proof structure.
pub fn validate_membership_proof(proof: &MembershipProof) -> Result<(), String> {
    if proof.registry_root.is_empty() {
        return Err("Empty registry root".to_string());
    }
    if proof.registry_root.len() != 32 {
        return Err("Registry root must be 32 bytes".to_string());
    }
    if !proof.domain_tag.starts_with("ZTML:") {
        return Err("Invalid domain tag".to_string());
    }
    Ok(())
}

/// Validate anonymous membership proof structure.
pub fn validate_anonymous_proof(proof: &AnonymousMembershipProof) -> Result<(), String> {
    if proof.nullifier.is_empty() || proof.nullifier.len() != 32 {
        return Err("Nullifier must be 32 bytes".to_string());
    }
    if proof.group_root.is_empty() || proof.group_root.len() != 32 {
        return Err("Group root must be 32 bytes".to_string());
    }
    if !proof.domain_tag.starts_with("ZTML:") {
        return Err("Invalid domain tag".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ownership_result_hides_identity() {
        let result = PrivateOwnershipResult {
            is_owner: true,
            proof: Some(MembershipProof {
                registry_root: vec![0xAA; 32],
                merkle_path: vec![vec![0xBB; 32], vec![0xCC; 32]],
                domain_tag: "ZTML:Commons:PropertyOwnership:v1".to_string(),
                verified: true,
            }),
            error: None,
        };

        // The result reveals ownership status but NOT the owner's identity
        assert!(result.is_owner);
        // No owner_did field exists — privacy by design
        let json = serde_json::to_string(&result).unwrap();
        assert!(!json.contains("owner_did"));
        assert!(!json.contains("did:"));
    }

    #[test]
    fn test_circle_membership_anonymous() {
        let result = CircleMembershipVerification {
            is_member: true,
            proof: Some(AnonymousMembershipProof {
                nullifier: vec![0xDD; 32],
                group_root: vec![0xEE; 32],
                proof_bytes: vec![1, 2, 3],
                domain_tag: "ZTML:Hearth:CircleMembership:v1".to_string(),
                verified: true,
            }),
            error: None,
        };

        // No member identities revealed
        let json = serde_json::to_string(&result).unwrap();
        assert!(!json.contains("AgentPubKey"));
        assert!(!json.contains("member_pubkey"));
        assert!(!json.contains("member_did"));
    }

    #[test]
    fn test_validation() {
        let valid = MembershipProof {
            registry_root: vec![0xAA; 32],
            merkle_path: vec![],
            domain_tag: "ZTML:Test:v1".to_string(),
            verified: true,
        };
        assert!(validate_membership_proof(&valid).is_ok());

        let bad_root = MembershipProof {
            registry_root: vec![0xAA; 16], // Wrong size
            ..valid.clone()
        };
        assert!(validate_membership_proof(&bad_root).is_err());
    }
}
