// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Domain separation tags for DASTARK proofs.
//!
//! Each cluster + proof type gets a unique domain tag to prevent
//! cross-protocol replay attacks. Format: `ZTML:{Cluster}:{ProofType}:v{N}`

use serde::{Deserialize, Serialize};

/// A domain separation tag that uniquely identifies a proof context.
///
/// Included in the signed message to prevent cross-protocol replay:
/// a governance vote proof cannot be replayed as a health attestation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DomainTag {
    /// Raw tag bytes (e.g. b"ZTML:Governance:AnonVote:v1")
    tag: Vec<u8>,
}

impl DomainTag {
    /// Create a domain tag from cluster name and proof type.
    ///
    /// # Example
    /// ```
    /// use mycelix_zkp_core::DomainTag;
    /// let tag = DomainTag::new("Governance", "AnonVote", 1);
    /// assert_eq!(tag.as_str(), "ZTML:Governance:AnonVote:v1");
    /// ```
    pub fn new(cluster: &str, proof_type: &str, version: u32) -> Self {
        let s = format!("ZTML:{}:{}:v{}", cluster, proof_type, version);
        Self {
            tag: s.into_bytes(),
        }
    }

    /// Create from raw bytes (for deserialization or legacy tags).
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { tag: bytes }
    }

    /// Get the tag as bytes for hashing.
    pub fn as_bytes(&self) -> &[u8] {
        &self.tag
    }

    /// Get the tag as a UTF-8 string (best-effort).
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.tag).unwrap_or("<invalid utf8>")
    }

    /// Validate that a tag matches expected cluster and proof type.
    pub fn matches(&self, cluster: &str, proof_type: &str) -> bool {
        let expected_prefix = format!("ZTML:{}:{}:", cluster, proof_type);
        self.as_str().starts_with(&expected_prefix)
    }
}

// --- Well-known domain tags (existing + new) ---

/// Existing: Federated learning gradient validation.
pub const TAG_FL_GRADIENT: &[u8] = b"ZTML:Gen7:AuthGradProof:v1";

/// Governance: Anonymous voting.
pub fn tag_governance_vote() -> DomainTag {
    DomainTag::new("Governance", "AnonVote", 1)
}

/// Identity: Selective credential disclosure.
pub fn tag_identity_disclosure() -> DomainTag {
    DomainTag::new("Identity", "SelectiveDisclosure", 1)
}

/// Consciousness: Tier verification (used by 6+ clusters).
pub fn tag_consciousness_tier() -> DomainTag {
    DomainTag::new("Consciousness", "TierProof", 1)
}

/// Finance: Transaction privacy.
pub fn tag_finance_tx() -> DomainTag {
    DomainTag::new("Finance", "TxPrivacy", 1)
}

/// Health: Record attestation.
pub fn tag_health_attest() -> DomainTag {
    DomainTag::new("Health", "RecordAttest", 1)
}

/// Mail: Sender authentication.
pub fn tag_mail_sender() -> DomainTag {
    DomainTag::new("Mail", "SenderAuth", 1)
}

/// Personal: Vault access.
pub fn tag_personal_vault() -> DomainTag {
    DomainTag::new("Personal", "VaultAccess", 1)
}

/// Supply chain: Provenance.
pub fn tag_supply_provenance() -> DomainTag {
    DomainTag::new("Supply", "Provenance", 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_tag_format() {
        let tag = DomainTag::new("Governance", "AnonVote", 1);
        assert_eq!(tag.as_str(), "ZTML:Governance:AnonVote:v1");
        assert_eq!(tag.as_bytes(), b"ZTML:Governance:AnonVote:v1");
    }

    #[test]
    fn test_domain_tag_matches() {
        let tag = DomainTag::new("Health", "RecordAttest", 1);
        assert!(tag.matches("Health", "RecordAttest"));
        assert!(!tag.matches("Health", "WrongType"));
        assert!(!tag.matches("Finance", "RecordAttest"));
    }

    #[test]
    fn test_all_tags_unique() {
        let tags: Vec<DomainTag> = vec![
            tag_governance_vote(),
            tag_identity_disclosure(),
            tag_finance_tx(),
            tag_health_attest(),
            tag_mail_sender(),
            tag_personal_vault(),
            tag_supply_provenance(),
        ];
        for i in 0..tags.len() {
            for j in (i + 1)..tags.len() {
                assert_ne!(
                    tags[i],
                    tags[j],
                    "Domain tags must be unique: {} vs {}",
                    tags[i].as_str(),
                    tags[j].as_str()
                );
            }
        }
    }

    #[test]
    fn test_legacy_tag_compat() {
        let legacy = DomainTag::from_bytes(TAG_FL_GRADIENT.to_vec());
        assert_eq!(legacy.as_str(), "ZTML:Gen7:AuthGradProof:v1");
        assert!(legacy.matches("Gen7", "AuthGradProof"));
    }
}
