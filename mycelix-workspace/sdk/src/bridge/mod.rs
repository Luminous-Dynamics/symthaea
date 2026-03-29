// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bridge Protocol
//!
//! Inter-hApp communication for the Mycelix ecosystem.
//! Enables cross-hApp reputation queries, credential verification,
//! event broadcasting, and Byzantine↔Identity integration.

// Bridge protocol types — struct fields documented via doc comments

pub mod byzantine_identity;

pub use byzantine_identity::{
    is_reputation_trustworthy,
    simple_trust_score,
    // Aggregated reputation types
    AggregatedReputation,
    AggregationConfig,
    ByzantineIdentityCoordinator,
    CoordinatorConfig,
    IdentityTrustAssessment,
    MatlNodeResult,
    NetworkHealth,
    NetworkStatusSummary,
    ReputationDataPoint,
    ReputationSource,
    TrustLevel,
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Messages for inter-hApp communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BridgeMessage {
    /// Query reputation across hApps
    ReputationQuery {
        /// Agent identifier to query
        agent: String,
        /// hApp originating the query
        source_happ: String,
    },

    /// Response with aggregated reputation
    ReputationResponse {
        /// Agent identifier
        agent: String,
        /// Per-hApp reputation scores
        scores: Vec<HappReputationScore>,
        /// Weighted aggregate score
        aggregate: f64,
    },

    /// Verify credential from another hApp
    CredentialVerify {
        /// Hash of the credential to verify
        credential_hash: String,
        /// hApp that issued the credential
        issuer_happ: String,
    },

    /// Credential verification result
    CredentialResult {
        /// Whether the credential is valid
        valid: bool,
        /// Credential issuer identifier
        issuer: String,
        /// Claims contained in the credential
        claims: Vec<String>,
    },

    /// Broadcast event to interested hApps
    EventBroadcast {
        /// Type of event being broadcast
        event_type: String,
        /// Serialized event payload
        payload: Vec<u8>,
        /// hApp originating the event
        source_happ: String,
        /// Unix timestamp of the event
        timestamp: u64,
    },

    /// Register hApp with bridge
    RegisterHapp {
        /// Unique hApp identifier
        happ_id: String,
        /// Human-readable hApp name
        happ_name: String,
        /// Capabilities offered by this hApp
        capabilities: Vec<String>,
    },
}

/// Reputation score from a specific hApp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HappReputationScore {
    /// hApp identifier
    pub happ_id: String,

    /// Human-readable hApp name
    pub happ_name: String,

    /// Reputation score [0.0, 1.0]
    pub score: f64,

    /// Number of interactions in this hApp
    pub interactions: u64,

    /// Last update timestamp
    pub last_updated: u64,
}

/// Cross-hApp reputation aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossHappReputation {
    /// Agent identifier
    pub agent: String,

    /// Individual hApp scores
    pub scores: Vec<HappReputationScore>,

    /// Weighted aggregate score
    pub aggregate: f64,

    /// Query timestamp
    pub queried_at: u64,
}

impl CrossHappReputation {
    /// Create from individual scores
    pub fn from_scores(agent: impl Into<String>, scores: Vec<HappReputationScore>) -> Self {
        let aggregate = Self::compute_aggregate(&scores);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            agent: agent.into(),
            scores,
            aggregate,
            queried_at: now,
        }
    }

    /// Compute weighted aggregate (by interaction count)
    fn compute_aggregate(scores: &[HappReputationScore]) -> f64 {
        if scores.is_empty() {
            return 0.5; // Default for unknown agents
        }

        let total_interactions: u64 = scores.iter().map(|s| s.interactions).sum();
        if total_interactions == 0 {
            // Equal weighting if no interactions
            return scores.iter().map(|s| s.score).sum::<f64>() / scores.len() as f64;
        }

        scores
            .iter()
            .map(|s| s.score * (s.interactions as f64 / total_interactions as f64))
            .sum()
    }

    /// Get the highest scoring hApp
    pub fn best_happ(&self) -> Option<&HappReputationScore> {
        self.scores.iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get the lowest scoring hApp
    pub fn worst_happ(&self) -> Option<&HappReputationScore> {
        self.scores.iter().min_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Check if agent is trustworthy (aggregate above threshold)
    pub fn is_trustworthy(&self, threshold: f64) -> bool {
        self.aggregate >= threshold
    }

    /// Total interactions across all hApps
    pub fn total_interactions(&self) -> u64 {
        self.scores.iter().map(|s| s.interactions).sum()
    }
}

/// hApp registration for bridge protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HappRegistration {
    /// Unique hApp identifier
    pub happ_id: String,

    /// Human-readable name
    pub happ_name: String,

    /// Supported capabilities
    pub capabilities: Vec<String>,

    /// Registration timestamp
    pub registered_at: u64,
}

/// Bridge event for pub/sub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeEvent {
    /// Event type (e.g., "reputation_update", "credential_issued")
    pub event_type: String,

    /// Source hApp
    pub source_happ: String,

    /// Event payload
    pub payload: Vec<u8>,

    /// Timestamp
    pub timestamp: u64,
}

impl BridgeEvent {
    /// Create a new bridge event
    pub fn new(
        event_type: impl Into<String>,
        source_happ: impl Into<String>,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            event_type: event_type.into(),
            source_happ: source_happ.into(),
            payload,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }
}

/// In-memory bridge for testing/local use
#[derive(Debug, Clone, Default)]
pub struct LocalBridge {
    registrations: HashMap<String, HappRegistration>,
    events: Vec<BridgeEvent>,
    reputations: HashMap<(String, String), HappReputationScore>, // (agent, happ) -> score
}

impl LocalBridge {
    /// Create a new local bridge
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a hApp
    pub fn register(&mut self, registration: HappRegistration) {
        self.registrations
            .insert(registration.happ_id.clone(), registration);
    }

    /// Record reputation for an agent in a hApp
    pub fn record_reputation(&mut self, agent: &str, score: HappReputationScore) {
        self.reputations
            .insert((agent.to_string(), score.happ_id.clone()), score);
    }

    /// Query cross-hApp reputation
    pub fn query_reputation(&self, agent: &str) -> CrossHappReputation {
        let scores: Vec<HappReputationScore> = self
            .reputations
            .iter()
            .filter(|((a, _), _)| a == agent)
            .map(|(_, score)| score.clone())
            .collect();

        CrossHappReputation::from_scores(agent, scores)
    }

    /// Broadcast an event
    pub fn broadcast(&mut self, event: BridgeEvent) {
        self.events.push(event);
    }

    /// Get events of a specific type
    pub fn get_events(&self, event_type: &str, since: u64) -> Vec<&BridgeEvent> {
        self.events
            .iter()
            .filter(|e| e.event_type == event_type && e.timestamp >= since)
            .collect()
    }

    /// Get all registered hApps
    pub fn registered_happs(&self) -> Vec<&HappRegistration> {
        self.registrations.values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_happ_reputation() {
        let scores = vec![
            HappReputationScore {
                happ_id: "mail".to_string(),
                happ_name: "Mycelix Mail".to_string(),
                score: 0.9,
                interactions: 100,
                last_updated: 0,
            },
            HappReputationScore {
                happ_id: "marketplace".to_string(),
                happ_name: "Mycelix Marketplace".to_string(),
                score: 0.8,
                interactions: 50,
                last_updated: 0,
            },
        ];

        let rep = CrossHappReputation::from_scores("agent1", scores);

        // Weighted average: (0.9 * 100 + 0.8 * 50) / 150 = (90 + 40) / 150 = 0.867
        assert!((rep.aggregate - 0.867).abs() < 0.01);
        assert!(rep.is_trustworthy(0.5));
    }

    #[test]
    fn test_local_bridge() {
        let mut bridge = LocalBridge::new();

        // Register hApps
        bridge.register(HappRegistration {
            happ_id: "mail".to_string(),
            happ_name: "Mycelix Mail".to_string(),
            capabilities: vec!["reputation".to_string()],
            registered_at: 0,
        });

        // Record reputation
        bridge.record_reputation(
            "agent1",
            HappReputationScore {
                happ_id: "mail".to_string(),
                happ_name: "Mycelix Mail".to_string(),
                score: 0.9,
                interactions: 50,
                last_updated: 0,
            },
        );

        // Query
        let rep = bridge.query_reputation("agent1");
        assert_eq!(rep.scores.len(), 1);
        assert!(rep.aggregate > 0.8);
    }
}
