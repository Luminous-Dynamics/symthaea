// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};

/// Epistemic tier levels for messages
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EpistemicTier {
    /// Tier 0: Null - Unverifiable belief
    Tier0Null,
    /// Tier 1: Testimonial - Personal attestation
    Tier1Testimonial,
    /// Tier 2: Privately Verifiable - Audit guild
    Tier2PrivatelyVerifiable,
    /// Tier 3: Cryptographically Proven - ZKP
    Tier3CryptographicallyProven,
    /// Tier 4: Publicly Reproducible - Open data/code
    Tier4PubliclyReproducible,
}

impl EpistemicTier {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Tier0Null),
            1 => Some(Self::Tier1Testimonial),
            2 => Some(Self::Tier2PrivatelyVerifiable),
            3 => Some(Self::Tier3CryptographicallyProven),
            4 => Some(Self::Tier4PubliclyReproducible),
            _ => None,
        }
    }

    pub fn to_u8(self) -> u8 {
        match self {
            Self::Tier0Null => 0,
            Self::Tier1Testimonial => 1,
            Self::Tier2PrivatelyVerifiable => 2,
            Self::Tier3CryptographicallyProven => 3,
            Self::Tier4PubliclyReproducible => 4,
        }
    }
}

impl std::fmt::Display for EpistemicTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tier0Null => write!(f, "Tier0 (Null)"),
            Self::Tier1Testimonial => write!(f, "Tier1 (Testimonial)"),
            Self::Tier2PrivatelyVerifiable => write!(f, "Tier2 (Privately Verifiable)"),
            Self::Tier3CryptographicallyProven => write!(f, "Tier3 (Cryptographically Proven)"),
            Self::Tier4PubliclyReproducible => write!(f, "Tier4 (Publicly Reproducible)"),
        }
    }
}

/// Mail message structure (matches DNA entry type)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MailMessage {
    /// Sender's DID
    pub from_did: String,
    /// Recipient's DID
    pub to_did: String,
    /// Encrypted subject (bytes)
    pub subject_encrypted: Vec<u8>,
    /// IPFS CID for message body
    pub body_cid: String,
    /// Unix timestamp
    pub timestamp: i64,
    /// Optional thread ID for replies
    pub thread_id: Option<String>,
    /// Epistemic tier classification
    pub epistemic_tier: EpistemicTier,
}

/// Trust score structure (matches DNA entry type)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustScore {
    /// DID being scored
    pub did: String,
    /// Trust score (0.0 - 1.0)
    pub score: f64,
    /// Last updated timestamp
    pub last_updated: i64,
    /// Source of trust score (MATL, local, etc.)
    pub source: String,
}

/// Contact entry structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contact {
    /// Contact's DID
    pub did: String,
    /// Display name
    pub name: String,
    /// Optional email
    pub email: Option<String>,
    /// Optional notes
    pub notes: Option<String>,
    /// Trust score
    pub trust_score: Option<f64>,
}

/// Message display format for CLI
#[derive(Debug, Clone)]
pub struct MessageDisplay {
    pub id: String,
    pub from: String,
    pub subject: String,
    pub timestamp: String,
    pub tier: EpistemicTier,
    pub read: bool,
    pub trust_score: Option<f64>,
}

/// DID registration request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DidRegistration {
    pub did: String,
    pub agent_pub_key: String,
    pub metadata: Option<serde_json::Value>,
}

/// DID resolution response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DidResolution {
    pub did: String,
    pub agent_pub_key: String,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Trust score update request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustScoreUpdate {
    pub did: String,
    pub score: f64,
}

/// Statistics about the local mail database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MailStats {
    pub total_messages: usize,
    pub unread_messages: usize,
    pub total_contacts: usize,
    pub total_trust_scores: usize,
    pub last_sync: Option<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epistemic_tier_conversion() {
        assert_eq!(EpistemicTier::from_u8(0), Some(EpistemicTier::Tier0Null));
        assert_eq!(
            EpistemicTier::from_u8(2),
            Some(EpistemicTier::Tier2PrivatelyVerifiable)
        );
        assert_eq!(EpistemicTier::from_u8(5), None);
    }

    #[test]
    fn test_epistemic_tier_display() {
        let tier = EpistemicTier::Tier2PrivatelyVerifiable;
        assert_eq!(tier.to_string(), "Tier2 (Privately Verifiable)");
    }
}
