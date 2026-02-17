//! Core types for the Multi-Factor Identity System

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Categories of identity factors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FactorCategory {
    /// Cryptographic factors (keys, signatures)
    Cryptographic,
    /// Biometric factors (face, fingerprint, etc.)
    Biometric,
    /// Social proof factors (guardians, attestations)
    SocialProof,
    /// External verification (Gitcoin, KYC)
    ExternalVerification,
    /// Knowledge factors (passwords, security questions)
    Knowledge,
}

impl FactorCategory {
    /// All factor categories
    pub fn all() -> Vec<FactorCategory> {
        vec![
            FactorCategory::Cryptographic,
            FactorCategory::Biometric,
            FactorCategory::SocialProof,
            FactorCategory::ExternalVerification,
            FactorCategory::Knowledge,
        ]
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            FactorCategory::Cryptographic => "Cryptographic",
            FactorCategory::Biometric => "Biometric",
            FactorCategory::SocialProof => "Social Proof",
            FactorCategory::ExternalVerification => "External Verification",
            FactorCategory::Knowledge => "Knowledge",
        }
    }
}

/// Status of an identity factor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactorStatus {
    /// Factor is active and verified
    Active,
    /// Factor is configured but not yet verified
    Pending,
    /// Factor has been revoked
    Revoked,
    /// Factor is temporarily inactive
    Inactive,
    /// Factor verification expired
    Expired,
}

/// Assurance levels aligned with Epistemic Charter v2.0 E-Axis
///
/// E0: Anonymous (0.0-0.2) - Unverified, read-only access
/// E1: Testimonial (0.2-0.4) - Basic key pair, limited interactions
/// E2: Privately Verifiable (0.4-0.6) - Multi-factor, community participation
/// E3: Cryptographically Proven (0.6-0.8) - Strong verification, governance voting
/// E4: Constitutionally Critical (0.8-1.0) - Maximum verification, constitutional proposals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssuranceLevel {
    /// Anonymous - No verification
    E0,
    /// Testimonial - Basic verification (single factor)
    E1,
    /// Privately Verifiable - Multi-factor from 2+ categories
    E2,
    /// Cryptographically Proven - 5+ factors, strong verification
    E3,
    /// Constitutionally Critical - Maximum verification
    E4,
}

impl AssuranceLevel {
    /// Get numerical value (0.0-1.0 range midpoint)
    pub fn score(&self) -> f32 {
        match self {
            AssuranceLevel::E0 => 0.1,
            AssuranceLevel::E1 => 0.3,
            AssuranceLevel::E2 => 0.5,
            AssuranceLevel::E3 => 0.7,
            AssuranceLevel::E4 => 0.9,
        }
    }

    /// Get minimum score threshold for this level
    pub fn min_threshold(&self) -> f32 {
        match self {
            AssuranceLevel::E0 => 0.0,
            AssuranceLevel::E1 => 0.2,
            AssuranceLevel::E2 => 0.4,
            AssuranceLevel::E3 => 0.6,
            AssuranceLevel::E4 => 0.8,
        }
    }

    /// Create from numerical score
    pub fn from_score(score: f32) -> Self {
        if score >= 0.8 {
            AssuranceLevel::E4
        } else if score >= 0.6 {
            AssuranceLevel::E3
        } else if score >= 0.4 {
            AssuranceLevel::E2
        } else if score >= 0.2 {
            AssuranceLevel::E1
        } else {
            AssuranceLevel::E0
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            AssuranceLevel::E0 => "Anonymous",
            AssuranceLevel::E1 => "Testimonial",
            AssuranceLevel::E2 => "Privately Verifiable",
            AssuranceLevel::E3 => "Cryptographically Proven",
            AssuranceLevel::E4 => "Constitutionally Critical",
        }
    }

    /// Get next higher level (if any)
    pub fn next(&self) -> Option<AssuranceLevel> {
        match self {
            AssuranceLevel::E0 => Some(AssuranceLevel::E1),
            AssuranceLevel::E1 => Some(AssuranceLevel::E2),
            AssuranceLevel::E2 => Some(AssuranceLevel::E3),
            AssuranceLevel::E3 => Some(AssuranceLevel::E4),
            AssuranceLevel::E4 => None,
        }
    }
}

impl PartialOrd for AssuranceLevel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AssuranceLevel {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score().partial_cmp(&other.score()).unwrap_or(Ordering::Equal)
    }
}

/// Identity scope for hierarchical identity support
#[derive(Debug, Clone, PartialEq)]
pub enum IdentityScope {
    /// Standard individual identity
    Individual { did: String },

    /// Delegated authority from principal to delegate
    Delegated {
        principal: String,
        delegate: String,
        scope: Vec<Capability>,
        expires_at: Option<chrono::DateTime<chrono::Utc>>,
    },

    /// Collective identity (DAO/multisig)
    Collective {
        dao_did: String,
        signers: Vec<String>,
        threshold: u8,
        governance_contract: String,
    },
}

/// Capabilities that can be delegated
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    /// Submit gradients on behalf of principal
    FLParticipation { round_limit: Option<u64> },
    /// Vote on proposals
    GovernanceVoting { proposal_types: Vec<String> },
    /// Manage storage and data
    DataManagement { read: bool, write: bool, delete: bool },
    /// Issue attestations
    AttestationAuthority { credential_types: Vec<String> },
    /// Treasury operations
    TreasuryAccess { spend_limit: Option<u64> },
    /// Custom capability
    Custom { name: String, parameters: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assurance_level_ordering() {
        assert!(AssuranceLevel::E4 > AssuranceLevel::E3);
        assert!(AssuranceLevel::E3 > AssuranceLevel::E2);
        assert!(AssuranceLevel::E2 > AssuranceLevel::E1);
        assert!(AssuranceLevel::E1 > AssuranceLevel::E0);
    }

    #[test]
    fn test_assurance_from_score() {
        assert_eq!(AssuranceLevel::from_score(0.1), AssuranceLevel::E0);
        assert_eq!(AssuranceLevel::from_score(0.25), AssuranceLevel::E1);
        assert_eq!(AssuranceLevel::from_score(0.5), AssuranceLevel::E2);
        assert_eq!(AssuranceLevel::from_score(0.7), AssuranceLevel::E3);
        assert_eq!(AssuranceLevel::from_score(0.9), AssuranceLevel::E4);
        assert_eq!(AssuranceLevel::from_score(1.0), AssuranceLevel::E4);
    }

    #[test]
    fn test_factor_categories() {
        let all = FactorCategory::all();
        assert_eq!(all.len(), 5);
        assert!(all.contains(&FactorCategory::Cryptographic));
        assert!(all.contains(&FactorCategory::Biometric));
    }

    #[test]
    fn test_assurance_level_next() {
        assert_eq!(AssuranceLevel::E0.next(), Some(AssuranceLevel::E1));
        assert_eq!(AssuranceLevel::E3.next(), Some(AssuranceLevel::E4));
        assert_eq!(AssuranceLevel::E4.next(), None);
    }
}
