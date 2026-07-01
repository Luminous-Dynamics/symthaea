//! Bridge Protocol stub

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BridgeMessage {
    ReputationQuery {
        agent: String,
        source_happ: String,
    },
    ReputationResponse {
        agent: String,
        scores: Vec<HappReputationScore>,
        aggregate: f64,
    },
    EventBroadcast {
        event_type: String,
        payload: Vec<u8>,
        source_happ: String,
        timestamp: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HappReputationScore {
    pub happ_id: String,
    pub happ_name: String,
    pub score: f64,
    pub interactions: u64,
    pub last_updated: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossHappReputation {
    pub agent: String,
    pub scores: Vec<HappReputationScore>,
    pub aggregate: f64,
    pub queried_at: u64,
}

impl CrossHappReputation {
    pub fn from_scores(agent: impl Into<String>, scores: Vec<HappReputationScore>) -> Self {
        let aggregate = if scores.is_empty() {
            0.5
        } else {
            let total_interactions: u64 = scores.iter().map(|s| s.interactions).sum();
            if total_interactions == 0 {
                scores.iter().map(|s| s.score).sum::<f64>() / scores.len() as f64
            } else {
                scores
                    .iter()
                    .map(|s| s.score * (s.interactions as f64 / total_interactions as f64))
                    .sum()
            }
        };
        Self {
            agent: agent.into(),
            scores,
            aggregate,
            queried_at: 0,
        }
    }

    pub fn is_trustworthy(&self, threshold: f64) -> bool {
        self.aggregate >= threshold
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeEvent {
    pub event_type: String,
    pub source_happ: String,
    pub payload: Vec<u8>,
    pub timestamp: u64,
}

impl BridgeEvent {
    pub fn new(
        event_type: impl Into<String>,
        source_happ: impl Into<String>,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            event_type: event_type.into(),
            source_happ: source_happ.into(),
            payload,
            timestamp: 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LocalBridge {
    events: Vec<BridgeEvent>,
    reputations: HashMap<(String, String), HappReputationScore>,
}

impl LocalBridge {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn broadcast(&mut self, event: BridgeEvent) {
        self.events.push(event);
    }

    pub fn query_reputation(&self, agent: &str) -> CrossHappReputation {
        let scores: Vec<HappReputationScore> = self
            .reputations
            .iter()
            .filter(|((a, _), _)| a == agent)
            .map(|(_, score)| score.clone())
            .collect();
        CrossHappReputation::from_scores(agent, scores)
    }
}
