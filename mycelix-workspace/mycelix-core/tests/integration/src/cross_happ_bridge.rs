// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-hApp Bridge Integration Tests
//!
//! Tests inter-hApp communication patterns using mocked zome calls.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐     call_remote()    ┌─────────────────┐
//! │   Governance    │ ◄─────────────────── │   Reputation    │
//! │     hApp        │                      │     hApp        │
//! └────────┬────────┘                      └────────┬────────┘
//!          │                                        │
//!          │    ┌─────────────────────────┐        │
//!          └───►│      Bridge Zome        │◄───────┘
//!               │  (credential + events)  │
//!               └───────────┬─────────────┘
//!                           │
//!               ┌───────────▼───────────┐
//!               │   FL Aggregator       │
//!               │   (Shapley → Rewards) │
//!               └───────────────────────┘
//! ```
//!
//! ## Cross-hApp Patterns Tested
//!
//! 1. **Reputation Sharing**: Agent reputation flows from source hApp to bridge
//! 2. **Credential Verification**: hApp A requests credential proof from hApp B
//! 3. **Event Broadcasting**: Events propagate to subscribed hApps
//! 4. **Trust Gates**: Actions gated by cross-hApp reputation threshold
//!
//! ## Note on Full Conductor Tests
//!
//! Full SweetTest conductor tests require:
//! - Compiled WASM DNA files
//! - Holochain conductor setup (hc-sandbox)
//! - Multi-agent network simulation
//!
//! These tests provide mock implementations for CI until DNA is compiled.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// =============================================================================
// Mock Types (matching bridge_integrity)
// =============================================================================

/// Mock hApp registration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HappRegistration {
    pub happ_id: String,
    pub dna_hash: Option<String>,
    pub happ_name: String,
    pub capabilities: Vec<String>,
    pub registered_at: u64,
    pub registrant: String,
    pub active: bool,
}

/// Mock reputation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationRecord {
    pub agent: String,
    pub happ_id: String,
    pub happ_name: String,
    pub score: f64,
    pub interactions: u64,
    pub negative_interactions: u64,
    pub updated_at: u64,
    pub evidence_hash: Option<String>,
}

/// Cross-hApp reputation aggregate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossHappReputation {
    pub agent: String,
    pub aggregate: f64,
    pub scores: HashMap<String, f64>,
    pub total_interactions: u64,
}

impl CrossHappReputation {
    pub fn from_records(agent: &str, records: Vec<ReputationRecord>) -> Self {
        if records.is_empty() {
            return Self {
                agent: agent.to_string(),
                aggregate: 0.5, // Default for unknown
                scores: HashMap::new(),
                total_interactions: 0,
            };
        }

        let mut total_weight = 0u64;
        let mut weighted_sum = 0.0;
        let mut scores = HashMap::new();
        let mut total_interactions = 0u64;

        for record in &records {
            let weight = record.interactions.max(1);
            weighted_sum += record.score * (weight as f64);
            total_weight += weight;
            total_interactions += record.interactions;
            scores.insert(record.happ_id.clone(), record.score);
        }

        let aggregate = if total_weight > 0 {
            weighted_sum / (total_weight as f64)
        } else {
            0.5
        };

        Self {
            agent: agent.to_string(),
            aggregate,
            scores,
            total_interactions,
        }
    }

    pub fn is_trustworthy(&self, threshold: f64) -> bool {
        self.aggregate >= threshold
    }
}

/// Credential verification request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialRequest {
    pub request_id: String,
    pub credential_hash: String,
    pub credential_type: String,
    pub issuer_happ: String,
    pub subject_agent: String,
    pub requester: String,
    pub requested_at: u64,
}

/// Credential verification response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialResponse {
    pub request_id: String,
    pub is_valid: bool,
    pub issuer: String,
    pub claims: Vec<String>,
    pub issued_at: u64,
    pub expires_at: u64,
}

/// Bridge event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeEvent {
    pub event_type: String,
    pub source_happ: String,
    pub payload: Vec<u8>,
    pub timestamp: u64,
    pub targets: Vec<String>,
}

// =============================================================================
// Mock Call Remote Framework
// =============================================================================

/// Simulated call_remote result
pub type CallRemoteResult<T> = Result<T, CallRemoteError>;

/// Errors from call_remote
#[derive(Debug, Clone)]
pub enum CallRemoteError {
    NetworkUnreachable(String),
    ZomeNotFound(String),
    Unauthorized(String),
    Timeout,
    DeserializationFailed(String),
}

/// Mock zome call context
pub struct MockZomeContext {
    pub caller_agent: String,
    pub caller_happ: String,
    pub current_time: u64,
}

/// Mock bridge zome for testing cross-hApp patterns
pub struct MockBridgeZome {
    /// Registered hApps
    happs: HashMap<String, HappRegistration>,
    /// Reputation records: agent -> [records]
    reputations: HashMap<String, Vec<ReputationRecord>>,
    /// Credential requests: request_id -> request
    credential_requests: HashMap<String, CredentialRequest>,
    /// Credential responses: request_id -> response
    credential_responses: HashMap<String, CredentialResponse>,
    /// Events by type
    events: HashMap<String, Vec<BridgeEvent>>,
    /// Event counter for IDs
    event_counter: u64,
}

impl MockBridgeZome {
    pub fn new() -> Self {
        Self {
            happs: HashMap::new(),
            reputations: HashMap::new(),
            credential_requests: HashMap::new(),
            credential_responses: HashMap::new(),
            events: HashMap::new(),
            event_counter: 0,
        }
    }

    /// Register a hApp with the bridge
    pub fn register_happ(&mut self, ctx: &MockZomeContext, input: HappRegistration) -> CallRemoteResult<String> {
        let happ_id = input.happ_id.clone();
        self.happs.insert(happ_id.clone(), input);

        // Broadcast registration event
        self.broadcast_event(ctx, BridgeEvent {
            event_type: "happ_registered".to_string(),
            source_happ: ctx.caller_happ.clone(),
            payload: format!(r#"{{"happ_id":"{}"}}"#, happ_id).into_bytes(),
            timestamp: ctx.current_time,
            targets: vec![],
        })?;

        Ok(happ_id)
    }

    /// Record reputation for an agent from a specific hApp
    pub fn record_reputation(&mut self, ctx: &MockZomeContext, record: ReputationRecord) -> CallRemoteResult<()> {
        // Verify caller owns the hApp
        if let Some(happ) = self.happs.get(&record.happ_id) {
            if happ.registrant != ctx.caller_agent {
                return Err(CallRemoteError::Unauthorized(
                    "Only hApp owner can record reputation".to_string()
                ));
            }
        }

        // Validate score
        if record.score < 0.0 || record.score > 1.0 {
            return Err(CallRemoteError::DeserializationFailed(
                "Score must be between 0.0 and 1.0".to_string()
            ));
        }

        // Store reputation
        self.reputations
            .entry(record.agent.clone())
            .or_insert_with(Vec::new)
            .push(record.clone());

        // Broadcast event
        self.broadcast_event(ctx, BridgeEvent {
            event_type: "reputation_update".to_string(),
            source_happ: ctx.caller_happ.clone(),
            payload: serde_json::to_vec(&record).unwrap_or_default(),
            timestamp: ctx.current_time,
            targets: vec![],
        })?;

        Ok(())
    }

    /// Query reputation for an agent
    pub fn query_reputation(&self, agent: &str, happ_filter: Option<&str>) -> Vec<ReputationRecord> {
        self.reputations
            .get(agent)
            .map(|records| {
                records.iter()
                    .filter(|r| {
                        happ_filter.map_or(true, |h| r.happ_id == h)
                    })
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get cross-hApp reputation aggregate
    pub fn aggregate_reputation(&self, agent: &str) -> CrossHappReputation {
        let records = self.query_reputation(agent, None);
        CrossHappReputation::from_records(agent, records)
    }

    /// Check if agent is trustworthy
    pub fn is_trustworthy(&self, agent: &str, threshold: f64) -> bool {
        self.aggregate_reputation(agent).is_trustworthy(threshold)
    }

    /// Request credential verification
    pub fn request_credential_verification(
        &mut self,
        ctx: &MockZomeContext,
        request: CredentialRequest,
    ) -> CallRemoteResult<String> {
        let request_id = request.request_id.clone();
        self.credential_requests.insert(request_id.clone(), request);
        Ok(request_id)
    }

    /// Provide credential verification response
    pub fn verify_credential(
        &mut self,
        _ctx: &MockZomeContext,
        response: CredentialResponse,
    ) -> CallRemoteResult<()> {
        if !self.credential_requests.contains_key(&response.request_id) {
            return Err(CallRemoteError::DeserializationFailed(
                "Request not found".to_string()
            ));
        }
        self.credential_responses.insert(response.request_id.clone(), response);
        Ok(())
    }

    /// Get credential verification response
    pub fn get_verification_response(&self, request_id: &str) -> Option<CredentialResponse> {
        self.credential_responses.get(request_id).cloned()
    }

    /// Broadcast an event
    pub fn broadcast_event(&mut self, _ctx: &MockZomeContext, event: BridgeEvent) -> CallRemoteResult<u64> {
        self.event_counter += 1;
        self.events
            .entry(event.event_type.clone())
            .or_insert_with(Vec::new)
            .push(event);
        Ok(self.event_counter)
    }

    /// Get events by type since timestamp
    pub fn get_events(&self, event_type: &str, since: u64) -> Vec<BridgeEvent> {
        self.events
            .get(event_type)
            .map(|events| {
                events.iter()
                    .filter(|e| e.timestamp >= since)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Default for MockBridgeZome {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Cross-hApp Communication Simulator
// =============================================================================

/// Simulates cross-hApp communication in a multi-hApp environment
pub struct CrossHappSimulator {
    /// The central bridge zome
    pub bridge: MockBridgeZome,
    /// Registered hApps
    pub happs: Vec<String>,
    /// Current simulated time
    pub current_time: u64,
}

impl CrossHappSimulator {
    pub fn new() -> Self {
        Self {
            bridge: MockBridgeZome::new(),
            happs: Vec::new(),
            current_time: 1704067200, // 2024-01-01 00:00:00 UTC
        }
    }

    /// Register a new hApp
    pub fn register_happ(&mut self, happ_id: &str, owner: &str) -> CallRemoteResult<()> {
        let ctx = MockZomeContext {
            caller_agent: owner.to_string(),
            caller_happ: happ_id.to_string(),
            current_time: self.current_time,
        };

        let registration = HappRegistration {
            happ_id: happ_id.to_string(),
            dna_hash: Some(format!("uhCEk_{}", happ_id)),
            happ_name: format!("Mycelix {}", happ_id),
            capabilities: vec!["reputation".to_string()],
            registered_at: self.current_time,
            registrant: owner.to_string(),
            active: true,
        };

        self.bridge.register_happ(&ctx, registration)?;
        self.happs.push(happ_id.to_string());
        Ok(())
    }

    /// Create context for a hApp call
    pub fn context(&self, happ_id: &str, agent: &str) -> MockZomeContext {
        MockZomeContext {
            caller_agent: agent.to_string(),
            caller_happ: happ_id.to_string(),
            current_time: self.current_time,
        }
    }

    /// Advance simulation time
    pub fn advance_time(&mut self, seconds: u64) {
        self.current_time += seconds;
    }
}

impl Default for CrossHappSimulator {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_happ_registration() {
        let mut sim = CrossHappSimulator::new();

        // Register governance hApp
        sim.register_happ("governance", "owner1").unwrap();

        // Register finance hApp
        sim.register_happ("finance", "owner2").unwrap();

        // Verify registrations
        assert_eq!(sim.happs.len(), 2);

        // Check registration events
        let events = sim.bridge.get_events("happ_registered", 0);
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_cross_happ_reputation_sharing() {
        let mut sim = CrossHappSimulator::new();

        // Register two hApps
        sim.register_happ("marketplace", "owner1").unwrap();
        sim.register_happ("governance", "owner2").unwrap();

        let agent = "uhCAk_agent123";

        // Marketplace records reputation
        let ctx1 = sim.context("marketplace", "owner1");
        sim.bridge.record_reputation(&ctx1, ReputationRecord {
            agent: agent.to_string(),
            happ_id: "marketplace".to_string(),
            happ_name: "Mycelix Marketplace".to_string(),
            score: 0.9,
            interactions: 100,
            negative_interactions: 5,
            updated_at: sim.current_time,
            evidence_hash: None,
        }).unwrap();

        // Governance records reputation
        let ctx2 = sim.context("governance", "owner2");
        sim.bridge.record_reputation(&ctx2, ReputationRecord {
            agent: agent.to_string(),
            happ_id: "governance".to_string(),
            happ_name: "Mycelix Governance".to_string(),
            score: 0.8,
            interactions: 50,
            negative_interactions: 2,
            updated_at: sim.current_time,
            evidence_hash: None,
        }).unwrap();

        // Query cross-hApp aggregate
        let aggregate = sim.bridge.aggregate_reputation(agent);

        // Weighted average: (0.9 * 100 + 0.8 * 50) / 150 = 130/150 = 0.867
        assert!((aggregate.aggregate - 0.867).abs() < 0.01);
        assert_eq!(aggregate.scores.len(), 2);
        assert!(aggregate.is_trustworthy(0.5));
    }

    #[test]
    fn test_credential_verification_flow() {
        let mut sim = CrossHappSimulator::new();

        // Register hApps
        sim.register_happ("identity", "identity_owner").unwrap();
        sim.register_happ("marketplace", "market_owner").unwrap();

        let subject = "uhCAk_subject";

        // Marketplace requests credential verification from identity hApp
        let ctx = sim.context("marketplace", "market_owner");
        let request_id = sim.bridge.request_credential_verification(&ctx, CredentialRequest {
            request_id: "req_001".to_string(),
            credential_hash: "cred_abc123".to_string(),
            credential_type: "email_verified".to_string(),
            issuer_happ: "identity".to_string(),
            subject_agent: subject.to_string(),
            requester: "market_owner".to_string(),
            requested_at: sim.current_time,
        }).unwrap();

        assert_eq!(request_id, "req_001");

        // Identity hApp responds
        let ctx2 = sim.context("identity", "identity_owner");
        sim.bridge.verify_credential(&ctx2, CredentialResponse {
            request_id: "req_001".to_string(),
            is_valid: true,
            issuer: "identity_happ".to_string(),
            claims: vec!["email@example.com".to_string()],
            issued_at: sim.current_time - 86400,
            expires_at: sim.current_time + 31536000, // 1 year
        }).unwrap();

        // Marketplace retrieves response
        let response = sim.bridge.get_verification_response("req_001").unwrap();
        assert!(response.is_valid);
        assert_eq!(response.claims[0], "email@example.com");
    }

    #[test]
    fn test_event_broadcasting() {
        let mut sim = CrossHappSimulator::new();

        // Register hApps
        sim.register_happ("governance", "gov_owner").unwrap();
        sim.register_happ("finance", "fin_owner").unwrap();

        // Governance broadcasts proposal event
        let ctx = sim.context("governance", "gov_owner");
        sim.bridge.broadcast_event(&ctx, BridgeEvent {
            event_type: "proposal_created".to_string(),
            source_happ: "governance".to_string(),
            payload: r#"{"proposal_id":"P001","title":"Increase FL rewards"}"#.into(),
            timestamp: sim.current_time,
            targets: vec!["finance".to_string()], // Target specific hApp
        }).unwrap();

        sim.advance_time(60);

        // Finance broadcasts payment event
        let ctx2 = sim.context("finance", "fin_owner");
        sim.bridge.broadcast_event(&ctx2, BridgeEvent {
            event_type: "payment_completed".to_string(),
            source_happ: "finance".to_string(),
            payload: r#"{"payment_id":"PAY001","amount":"1000"}"#.into(),
            timestamp: sim.current_time,
            targets: vec![], // Broadcast to all
        }).unwrap();

        // Query events
        let proposals = sim.bridge.get_events("proposal_created", 0);
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].source_happ, "governance");

        let payments = sim.bridge.get_events("payment_completed", 0);
        assert_eq!(payments.len(), 1);
    }

    #[test]
    fn test_trust_gated_action() {
        let mut sim = CrossHappSimulator::new();

        sim.register_happ("governance", "owner1").unwrap();

        let trusted_agent = "uhCAk_trusted";
        let untrusted_agent = "uhCAk_untrusted";

        // Build reputation for trusted agent
        let ctx = sim.context("governance", "owner1");
        sim.bridge.record_reputation(&ctx, ReputationRecord {
            agent: trusted_agent.to_string(),
            happ_id: "governance".to_string(),
            happ_name: "Governance".to_string(),
            score: 0.85,
            interactions: 100,
            negative_interactions: 5,
            updated_at: sim.current_time,
            evidence_hash: None,
        }).unwrap();

        // Untrusted agent has low score
        sim.bridge.record_reputation(&ctx, ReputationRecord {
            agent: untrusted_agent.to_string(),
            happ_id: "governance".to_string(),
            happ_name: "Governance".to_string(),
            score: 0.3,
            interactions: 10,
            negative_interactions: 5,
            updated_at: sim.current_time,
            evidence_hash: None,
        }).unwrap();

        // Trust gate check (threshold 0.7)
        assert!(sim.bridge.is_trustworthy(trusted_agent, 0.7));
        assert!(!sim.bridge.is_trustworthy(untrusted_agent, 0.7));
    }

    #[test]
    fn test_unauthorized_reputation_recording() {
        let mut sim = CrossHappSimulator::new();

        // Register hApp with owner1
        sim.register_happ("marketplace", "owner1").unwrap();

        // Try to record reputation as different agent (should fail)
        let ctx = sim.context("marketplace", "attacker");
        let result = sim.bridge.record_reputation(&ctx, ReputationRecord {
            agent: "victim".to_string(),
            happ_id: "marketplace".to_string(),
            happ_name: "Marketplace".to_string(),
            score: 0.0, // Trying to tank victim's reputation
            interactions: 1,
            negative_interactions: 0,
            updated_at: sim.current_time,
            evidence_hash: None,
        });

        assert!(matches!(result, Err(CallRemoteError::Unauthorized(_))));
    }

    #[test]
    fn test_fl_rewards_to_reputation_flow() {
        let mut sim = CrossHappSimulator::new();

        // Register FL and bridge hApps
        sim.register_happ("fl_aggregator", "fl_owner").unwrap();

        // Simulate FL round completion
        let contributors = vec![
            ("node1", 0.25), // 25% share
            ("node2", 0.35), // 35% share
            ("node3", 0.40), // 40% share
        ];

        let ctx = sim.context("fl_aggregator", "fl_owner");

        // Record reputation based on FL contribution
        for (node, share) in contributors {
            // Higher share = higher reputation score
            let score = 0.5 + (share * 0.5); // Scale 0.5-1.0

            sim.bridge.record_reputation(&ctx, ReputationRecord {
                agent: node.to_string(),
                happ_id: "fl_aggregator".to_string(),
                happ_name: "FL Aggregator".to_string(),
                score,
                interactions: 1,
                negative_interactions: 0,
                updated_at: sim.current_time,
                evidence_hash: Some("shapley_proof_abc".to_string()),
            }).unwrap();
        }

        // Verify reputation reflects contribution
        let node3_rep = sim.bridge.aggregate_reputation("node3");
        let node1_rep = sim.bridge.aggregate_reputation("node1");

        assert!(node3_rep.aggregate > node1_rep.aggregate,
            "Higher contributor should have higher reputation");
    }

    #[test]
    fn test_multi_round_reputation_evolution() {
        let mut sim = CrossHappSimulator::new();
        sim.register_happ("fl", "fl_owner").unwrap();

        let agent = "consistent_contributor";
        let ctx = sim.context("fl", "fl_owner");

        // Record 5 rounds of good contributions
        for round in 1..=5 {
            sim.advance_time(3600); // 1 hour per round
            sim.bridge.record_reputation(&ctx, ReputationRecord {
                agent: agent.to_string(),
                happ_id: "fl".to_string(),
                happ_name: "FL".to_string(),
                score: 0.85 + (round as f64 * 0.02), // Improving score
                interactions: round,
                negative_interactions: 0,
                updated_at: sim.current_time,
                evidence_hash: None,
            }).unwrap();
        }

        // Query all reputation records
        let records = sim.bridge.query_reputation(agent, None);
        assert_eq!(records.len(), 5);

        // Aggregate should reflect cumulative good performance
        let aggregate = sim.bridge.aggregate_reputation(agent);
        assert!(aggregate.aggregate > 0.85);
        assert_eq!(aggregate.total_interactions, 15); // 1+2+3+4+5
    }
}
