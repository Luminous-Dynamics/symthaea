// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Governance-gated session approval for Sovereign RDP support sessions.
//!
//! For sensitive sessions (Security category, Critical priority), submit an
//! on-chain governance proposal before allowing RDP Active state.
//! Integrates with Mycelix governance via GovernanceManager::inject_event().

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::rdp_support_bridge::{SupportCategory, TicketPriority};

/// Governance request state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalState {
    /// Proposal submitted, awaiting votes.
    Pending,
    /// Community approved the session.
    Approved,
    /// Community denied the session.
    Denied,
    /// Proposal expired without resolution.
    Expired,
}

/// A pending governance request for RDP access.
#[derive(Debug, Clone)]
pub struct GovernanceRequest {
    /// Unique request ID.
    pub request_id: String,
    /// Support ticket ID.
    pub ticket_id: String,
    /// Agent requesting access (MFDI DID).
    pub agent_did: String,
    /// Device to be accessed.
    pub device_id: String,
    /// Current approval state.
    pub state: ApprovalState,
    /// When the request was submitted.
    pub submitted_at: Instant,
    /// Request expires after this duration.
    pub timeout: Duration,
    /// On-chain proposal ID (if submitted to Holochain).
    pub proposal_id: Option<String>,
    /// Category + priority for the proposal description.
    pub category: SupportCategory,
    pub priority: TicketPriority,
    /// Agent's consciousness phi at request time.
    pub agent_phi: f32,
}

/// Minimum consciousness for submitting governance proposals.
const GOV_PROPOSAL_PHI: f32 = 0.3;

/// Default governance request timeout.
const GOV_TIMEOUT_SECS: u64 = 1800; // 30 minutes

/// Governance gate for RDP support sessions.
pub struct GovernanceGate {
    /// Pending governance requests by ticket_id.
    requests: HashMap<String, GovernanceRequest>,
    /// Counter for generating request IDs.
    next_id: u64,
}

impl GovernanceGate {
    pub fn new() -> Self {
        Self {
            requests: HashMap::new(),
            next_id: 0,
        }
    }

    /// Submit a governance request for RDP access.
    ///
    /// Returns `None` if the agent's consciousness is below the proposal threshold.
    pub fn request_approval(
        &mut self,
        ticket_id: &str,
        agent_did: &str,
        device_id: &str,
        category: SupportCategory,
        priority: TicketPriority,
        agent_phi: f32,
    ) -> Option<GovernanceRequest> {
        // Consciousness gate.
        if agent_phi < GOV_PROPOSAL_PHI {
            return None;
        }

        self.next_id += 1;
        let request = GovernanceRequest {
            request_id: format!("gov-rdp-{}", self.next_id),
            ticket_id: ticket_id.to_string(),
            agent_did: agent_did.to_string(),
            device_id: device_id.to_string(),
            state: ApprovalState::Pending,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(GOV_TIMEOUT_SECS),
            proposal_id: None,
            category,
            priority,
            agent_phi,
        };

        self.requests.insert(ticket_id.to_string(), request.clone());
        Some(request)
    }

    /// Check the approval state for a ticket (non-blocking poll).
    pub fn check_approval(&mut self, ticket_id: &str) -> ApprovalState {
        let Some(request) = self.requests.get_mut(ticket_id) else {
            return ApprovalState::Denied;
        };

        // Check timeout.
        if request.submitted_at.elapsed() > request.timeout {
            request.state = ApprovalState::Expired;
        }

        request.state
    }

    /// Set the on-chain proposal ID after submission to Holochain.
    pub fn set_proposal_id(&mut self, ticket_id: &str, proposal_id: &str) {
        if let Some(request) = self.requests.get_mut(ticket_id) {
            request.proposal_id = Some(proposal_id.to_string());
        }
    }

    /// Mark a request as approved (called when governance tally completes).
    pub fn on_approved(&mut self, ticket_id: &str) {
        if let Some(request) = self.requests.get_mut(ticket_id) {
            request.state = ApprovalState::Approved;
        }
    }

    /// Mark a request as denied.
    pub fn on_denied(&mut self, ticket_id: &str) {
        if let Some(request) = self.requests.get_mut(ticket_id) {
            request.state = ApprovalState::Denied;
        }
    }

    /// Build a proposal description for submission to governance.
    pub fn build_proposal_description(request: &GovernanceRequest) -> String {
        format!(
            "Remote desktop support access requested.\n\
             Ticket: {}\n\
             Category: {:?}\n\
             Priority: {:?}\n\
             Agent: {}\n\
             Device: {}\n\
             Agent Φ: {:.2}",
            request.ticket_id,
            request.category,
            request.priority,
            request.agent_did,
            request.device_id,
            request.agent_phi,
        )
    }

    /// Clean up expired requests.
    pub fn tick(&mut self) {
        self.requests.retain(|_, r| {
            r.state == ApprovalState::Pending && r.submitted_at.elapsed() <= r.timeout
                || r.state == ApprovalState::Approved
        });
    }
}

impl Default for GovernanceGate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_approval_consciousness_gate() {
        let mut gate = GovernanceGate::new();

        // Below threshold.
        let result = gate.request_approval(
            "T1",
            "agent1",
            "device1",
            SupportCategory::Security,
            TicketPriority::Critical,
            0.1, // Below 0.3
        );
        assert!(result.is_none());

        // Above threshold.
        let result = gate.request_approval(
            "T1",
            "agent1",
            "device1",
            SupportCategory::Security,
            TicketPriority::Critical,
            0.5,
        );
        assert!(result.is_some());
        assert_eq!(gate.check_approval("T1"), ApprovalState::Pending);
    }

    #[test]
    fn test_approval_lifecycle() {
        let mut gate = GovernanceGate::new();
        gate.request_approval(
            "T2",
            "agent2",
            "device2",
            SupportCategory::Security,
            TicketPriority::High,
            0.4,
        );

        assert_eq!(gate.check_approval("T2"), ApprovalState::Pending);

        gate.on_approved("T2");
        assert_eq!(gate.check_approval("T2"), ApprovalState::Approved);
    }

    #[test]
    fn test_denial() {
        let mut gate = GovernanceGate::new();
        gate.request_approval(
            "T3",
            "agent3",
            "device3",
            SupportCategory::Network,
            TicketPriority::Critical,
            0.5,
        );

        gate.on_denied("T3");
        assert_eq!(gate.check_approval("T3"), ApprovalState::Denied);
    }

    #[test]
    fn test_unknown_ticket_returns_denied() {
        let mut gate = GovernanceGate::new();
        assert_eq!(gate.check_approval("nonexistent"), ApprovalState::Denied);
    }
}
