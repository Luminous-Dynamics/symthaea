// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Support Session Bridge: links Mycelix support tickets to Sovereign RDP sessions.
//!
//! ## Flow
//!
//! ```text
//! Support Ticket Created → Triage assigns category/priority
//!   → [Optional] Governance approval for sensitive sessions
//!   → Generate RDP invite token (BLAKE3 HMAC, time-bounded)
//!   → Support agent connects via RDP with token
//!   → Session authenticated (MFDI + consciousness)
//!   → Diagnostics auto-run, actions proposed
//!   → Session resolves → Knowledge graduated
//!   → Full audit trail on Holochain
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Support ticket categories (mirrors symthaea-support::types).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SupportCategory {
    Network,
    Hardware,
    Software,
    Holochain,
    Mycelix,
    Security,
    General,
}

/// Ticket priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TicketPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Autonomy level for remote actions during the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AutonomyLevel {
    /// Read-only: diagnostics viewable but no actions.
    Advisory,
    /// Limited: pre-approved actions only (restart, clear cache).
    SemiAutonomous,
    /// Full: any action with rollback tracking.
    FullAutonomous,
}

/// A support session linking a ticket to an RDP session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportSession {
    /// The support ticket this session is for.
    pub ticket_id: String,
    /// Category from triage.
    pub category: SupportCategory,
    /// Priority from triage.
    pub priority: TicketPriority,
    /// Autonomy level (controls what the support agent can do).
    pub autonomy: AutonomyLevel,
    /// RDP session ID (set when agent connects).
    pub rdp_session_id: Option<String>,
    /// Support agent's identity (MFDI DID).
    pub agent_did: Option<String>,
    /// Device being supported (MFDI DID or hostname).
    pub device_id: String,
    /// When the invite was created.
    pub created_at: u64,
    /// When the invite expires.
    pub expires_at: u64,
    /// Whether governance approval was required and granted.
    pub governance_approved: bool,
    /// Governance proposal ID (if applicable).
    pub governance_proposal_id: Option<String>,
    /// Resolution notes (filled on session close).
    pub resolution: Option<String>,
}

/// Invite token for RDP support session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InviteToken {
    /// The token string (BLAKE3 HMAC).
    pub token: String,
    /// Ticket ID encoded in the token.
    pub ticket_id: String,
    /// Expiry timestamp (unix seconds).
    pub expires_at: u64,
}

/// Default invite duration (1 hour).
const INVITE_DURATION_SECS: u64 = 3600;

/// HMAC key domain separator for invite tokens.
const INVITE_HMAC_DOMAIN: &[u8] = b"sovereign-rdp-support-invite-v1";

/// Policy: which category+priority combinations require governance approval.
pub struct GovernancePolicy {
    /// Categories that always require governance.
    pub governed_categories: Vec<SupportCategory>,
    /// Priority threshold above which governance is required.
    pub governed_priority_threshold: TicketPriority,
}

impl Default for GovernancePolicy {
    fn default() -> Self {
        Self {
            governed_categories: vec![SupportCategory::Security],
            governed_priority_threshold: TicketPriority::Critical,
        }
    }
}

impl GovernancePolicy {
    /// Check if this session requires governance approval.
    pub fn requires_governance(&self, category: SupportCategory, priority: TicketPriority) -> bool {
        self.governed_categories.contains(&category) || priority >= self.governed_priority_threshold
    }
}

/// Manages support sessions and their RDP linkage.
pub struct SupportSessionManager {
    sessions: HashMap<String, SupportSession>,
    policy: GovernancePolicy,
    /// Secret key for HMAC token generation (derived from node identity).
    hmac_secret: [u8; 32],
}

impl SupportSessionManager {
    /// Create a new manager with a secret key for token generation.
    pub fn new(hmac_secret: [u8; 32]) -> Self {
        Self {
            sessions: HashMap::new(),
            policy: GovernancePolicy::default(),
            hmac_secret,
        }
    }

    /// Create a support session and generate an invite token.
    pub fn create_session(
        &mut self,
        ticket_id: &str,
        category: SupportCategory,
        priority: TicketPriority,
        autonomy: AutonomyLevel,
        device_id: &str,
    ) -> (SupportSession, InviteToken) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let expires = now + INVITE_DURATION_SECS;

        let session = SupportSession {
            ticket_id: ticket_id.to_string(),
            category,
            priority,
            autonomy,
            rdp_session_id: None,
            agent_did: None,
            device_id: device_id.to_string(),
            created_at: now,
            expires_at: expires,
            governance_approved: !self.policy.requires_governance(category, priority),
            governance_proposal_id: None,
            resolution: None,
        };

        let token = self.generate_token(ticket_id, expires);

        self.sessions.insert(ticket_id.to_string(), session.clone());
        (session, token)
    }

    /// Validate an invite token and return the associated session.
    pub fn validate_token(&self, token_str: &str) -> Option<&SupportSession> {
        // Find session by trying all tickets (token encodes ticket_id in HMAC).
        for (ticket_id, session) in &self.sessions {
            let expected = self.generate_token(ticket_id, session.expires_at);
            if expected.token == token_str {
                // Check expiry.
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                if now <= session.expires_at {
                    return Some(session);
                }
            }
        }
        None
    }

    /// Link an RDP session to a support ticket.
    pub fn link_rdp_session(
        &mut self,
        ticket_id: &str,
        rdp_session_id: &str,
        agent_did: &str,
    ) -> bool {
        if let Some(session) = self.sessions.get_mut(ticket_id) {
            // Must be governance-approved (or not requiring governance).
            if !session.governance_approved {
                return false;
            }
            session.rdp_session_id = Some(rdp_session_id.to_string());
            session.agent_did = Some(agent_did.to_string());
            true
        } else {
            false
        }
    }

    /// Mark governance approval for a session.
    pub fn approve_governance(&mut self, ticket_id: &str, proposal_id: &str) {
        if let Some(session) = self.sessions.get_mut(ticket_id) {
            session.governance_approved = true;
            session.governance_proposal_id = Some(proposal_id.to_string());
        }
    }

    /// Close a session with resolution notes.
    pub fn close_session(&mut self, ticket_id: &str, resolution: &str) -> Option<SupportSession> {
        if let Some(session) = self.sessions.get_mut(ticket_id) {
            session.resolution = Some(resolution.to_string());
        }
        self.sessions.remove(ticket_id)
    }

    /// Get a session by ticket ID.
    pub fn session(&self, ticket_id: &str) -> Option<&SupportSession> {
        self.sessions.get(ticket_id)
    }

    /// Find session by RDP session ID.
    pub fn session_by_rdp(&self, rdp_session_id: &str) -> Option<&SupportSession> {
        self.sessions
            .values()
            .find(|s| s.rdp_session_id.as_deref() == Some(rdp_session_id))
    }

    /// Whether the session requires governance and is still waiting.
    pub fn awaiting_governance(&self, ticket_id: &str) -> bool {
        self.sessions
            .get(ticket_id)
            .map(|s| {
                !s.governance_approved && self.policy.requires_governance(s.category, s.priority)
            })
            .unwrap_or(false)
    }

    /// Number of active sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Generate a BLAKE3 HMAC invite token.
    fn generate_token(&self, ticket_id: &str, expires_at: u64) -> InviteToken {
        let mut input = Vec::new();
        input.extend_from_slice(INVITE_HMAC_DOMAIN);
        input.extend_from_slice(ticket_id.as_bytes());
        input.extend_from_slice(&expires_at.to_le_bytes());

        let mac = blake3::keyed_hash(&self.hmac_secret, &input);
        let token = hex::encode(&mac.as_bytes()[..16]); // 128-bit token

        InviteToken {
            token,
            ticket_id: ticket_id.to_string(),
            expires_at,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_manager() -> SupportSessionManager {
        SupportSessionManager::new([42u8; 32])
    }

    #[test]
    fn test_create_session_and_token() {
        let mut mgr = test_manager();
        let (session, token) = mgr.create_session(
            "TICKET-001",
            SupportCategory::Software,
            TicketPriority::Medium,
            AutonomyLevel::SemiAutonomous,
            "device-abc",
        );

        assert_eq!(session.ticket_id, "TICKET-001");
        assert!(session.governance_approved); // Software+Medium doesn't need governance
        assert!(!token.token.is_empty());
        assert_eq!(mgr.session_count(), 1);
    }

    #[test]
    fn test_security_critical_requires_governance() {
        let mut mgr = test_manager();
        let (session, _) = mgr.create_session(
            "TICKET-002",
            SupportCategory::Security,
            TicketPriority::Critical,
            AutonomyLevel::FullAutonomous,
            "device-xyz",
        );

        assert!(!session.governance_approved); // Security requires governance
        assert!(mgr.awaiting_governance("TICKET-002"));

        // Link should fail without governance.
        assert!(!mgr.link_rdp_session("TICKET-002", "rdp-1", "did:mycelix:agent1"));

        // Approve governance.
        mgr.approve_governance("TICKET-002", "proposal-123");
        assert!(!mgr.awaiting_governance("TICKET-002"));

        // Link should succeed now.
        assert!(mgr.link_rdp_session("TICKET-002", "rdp-1", "did:mycelix:agent1"));
    }

    #[test]
    fn test_token_validation() {
        let mut mgr = test_manager();
        let (_, token) = mgr.create_session(
            "TICKET-003",
            SupportCategory::General,
            TicketPriority::Low,
            AutonomyLevel::Advisory,
            "device-test",
        );

        // Valid token.
        let session = mgr.validate_token(&token.token);
        assert!(session.is_some());
        assert_eq!(session.unwrap().ticket_id, "TICKET-003");

        // Invalid token.
        assert!(mgr.validate_token("bogus-token").is_none());
    }

    #[test]
    fn test_close_session_with_resolution() {
        let mut mgr = test_manager();
        mgr.create_session(
            "TICKET-004",
            SupportCategory::Hardware,
            TicketPriority::High,
            AutonomyLevel::SemiAutonomous,
            "device-hw",
        );

        let closed = mgr.close_session("TICKET-004", "Replaced faulty RAM module");
        assert!(closed.is_some());
        assert_eq!(
            closed.unwrap().resolution.as_deref(),
            Some("Replaced faulty RAM module")
        );
        assert_eq!(mgr.session_count(), 0);
    }

    #[test]
    fn test_find_by_rdp_session() {
        let mut mgr = test_manager();
        mgr.create_session(
            "TICKET-005",
            SupportCategory::Network,
            TicketPriority::Medium,
            AutonomyLevel::Advisory,
            "device-net",
        );
        mgr.link_rdp_session("TICKET-005", "rdp-session-42", "did:mycelix:agent2");

        let found = mgr.session_by_rdp("rdp-session-42");
        assert!(found.is_some());
        assert_eq!(found.unwrap().ticket_id, "TICKET-005");

        assert!(mgr.session_by_rdp("nonexistent").is_none());
    }

    #[test]
    fn test_critical_priority_requires_governance() {
        let policy = GovernancePolicy::default();

        // Critical priority always requires governance.
        assert!(policy.requires_governance(SupportCategory::General, TicketPriority::Critical));

        // Security category always requires governance.
        assert!(policy.requires_governance(SupportCategory::Security, TicketPriority::Low));

        // General + Medium does not.
        assert!(!policy.requires_governance(SupportCategory::General, TicketPriority::Medium));
    }
}
