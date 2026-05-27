// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Dark Spot DHT Actor
//!
//! Bridges the Mycelix GIS Dark Spot DHT with the brain's actor model,
//! enabling consciousness-aware ignorance sharing across the neural network.
//!
//! ## Architecture
//!
//! ```text
//! Brain Actors                      Dark Spot Network
//! ┌─────────────────┐              ┌──────────────────┐
//! │ Prefrontal      │──Query──────▶│                  │
//! │ (attention)     │              │  DarkSpotDHTActor│
//! │                 │◀─BlindSpots──│  (broker)        │
//! └─────────────────┘              │                  │
//! ┌─────────────────┐              │  ┌────────────┐  │
//! │ Hippocampus     │──Publish────▶│  │DarkSpotDHT │  │
//! │ (memory gaps)   │              │  │(ZK proofs) │  │
//! │                 │◀─Matches─────│  └────────────┘  │
//! └─────────────────┘              └──────────────────┘
//! ```
//!
//! ## Message Types
//!
//! - `PublishIgnorance`: Brain module discovered a knowledge gap
//! - `QueryBlindSpots`: Request collective ignorance patterns
//! - `FindKnowledge`: Search for knowledge to fill gaps
//! - `AddKnowledge`: Contribute knowledge to the network

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::{RwLock, oneshot};
use tracing::{debug, info, warn};

use super::actor_model::{ActorPriority, ActorTrait as Actor, OrganMessage, Response};
use crate::mycelix::gis::dht::{BlindSpot, DarkSpotDHT, KnowledgeMatch, ZKIgnoranceSignature};

// ============================================================================
// Dark Spot Actor Messages
// ============================================================================

/// Messages specific to Dark Spot DHT operations
#[derive(Debug)]
pub enum DarkSpotMessage {
    /// Publish an ignorance signature (I don't know X)
    PublishIgnorance {
        topic: String,
        category: String,
        eig: f32,
        reply: oneshot::Sender<DarkSpotResponse>,
    },

    /// Query for collective blind spots
    QueryBlindSpots {
        reply: oneshot::Sender<DarkSpotResponse>,
    },

    /// Find knowledge matching an ignorance signature
    FindKnowledge {
        signature_id: String,
        reply: oneshot::Sender<DarkSpotResponse>,
    },

    /// Add knowledge to help others
    AddKnowledge {
        id: String,
        content: String,
        category: String,
        reputation: f32,
        reply: oneshot::Sender<DarkSpotResponse>,
    },

    /// Reveal a blind spot (collective decision)
    RevealBlindSpot {
        blind_spot_id: String,
        participants: usize,
        reply: oneshot::Sender<DarkSpotResponse>,
    },

    /// Get statistics about the DHT
    GetStats {
        reply: oneshot::Sender<DarkSpotResponse>,
    },
}

/// Errors that can occur during actor message handling
#[derive(Debug, Clone, thiserror::Error)]
pub enum ActorError {
    /// Received an unexpected message type
    #[error("Unexpected message type: {0}")]
    UnexpectedMessage(String),

    /// Message handling failed
    #[error("Message handling failed: {0}")]
    HandlingFailed(String),

    /// Response variant mismatch
    #[error("Unexpected response variant: expected {expected}, got {actual}")]
    UnexpectedResponse {
        expected: &'static str,
        actual: String,
    },
}

/// Responses from Dark Spot operations
#[derive(Debug, Clone)]
pub enum DarkSpotResponse {
    /// Ignorance published successfully
    Published { signature_id: String },

    /// Blind spots detected
    BlindSpots { spots: Vec<CollectiveBlindSpot> },

    /// Knowledge matches found
    Matches { matches: Vec<MatchResult> },

    /// Knowledge added
    KnowledgeAdded { id: String },

    /// Blind spot revealed
    BlindSpotRevealed {
        id: String,
        category: String,
        revealed_topic: Option<String>,
    },

    /// DHT statistics
    Stats {
        signature_count: usize,
        knowledge_count: usize,
        blind_spot_count: usize,
    },

    /// Error response
    Error { message: String },
}

/// Simplified blind spot for actor responses
#[derive(Debug, Clone)]
pub struct CollectiveBlindSpot {
    pub id: String,
    pub category: String,
    pub signature_count: usize,
    pub average_eig: f32,
}

impl From<BlindSpot> for CollectiveBlindSpot {
    fn from(bs: BlindSpot) -> Self {
        Self {
            id: bs.id,
            category: bs.category,
            signature_count: bs.signature_count,
            average_eig: bs.average_eig,
        }
    }
}

/// Simplified match result for actor responses
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub claim_id: String,
    pub similarity: f32,
    pub responder_reputation: f32,
}

impl From<KnowledgeMatch> for MatchResult {
    fn from(m: KnowledgeMatch) -> Self {
        Self {
            claim_id: m.claim_id,
            similarity: m.similarity,
            responder_reputation: m.responder_reputation,
        }
    }
}

// ============================================================================
// Dark Spot DHT Actor
// ============================================================================

/// Actor that wraps the Dark Spot DHT for integration with the brain's actor model
///
/// This actor enables:
/// - Brain modules to publish their ignorance
/// - Collective blind spot detection across the network
/// - Knowledge matching to fill gaps
/// - Privacy-preserving ignorance sharing via ZK proofs
pub struct DarkSpotDHTActor {
    /// The underlying DHT
    dht: DarkSpotDHT,

    /// Cache of recent signatures (for find_matches without re-creating)
    signature_cache: std::collections::HashMap<String, ZKIgnoranceSignature>,

    /// Actor name
    name: String,

    /// Statistics
    stats: DarkSpotStats,
}

/// DHT statistics
#[derive(Debug, Clone, Default)]
pub struct DarkSpotStats {
    pub ignorance_published: usize,
    pub knowledge_added: usize,
    pub matches_found: usize,
    pub blind_spots_detected: usize,
}

impl DarkSpotDHTActor {
    /// Create a new Dark Spot DHT Actor
    pub fn new() -> Self {
        Self {
            dht: DarkSpotDHT::new(),
            signature_cache: std::collections::HashMap::new(),
            name: "DarkSpotDHT".to_string(),
            stats: DarkSpotStats::default(),
        }
    }

    /// Create with a custom name
    pub fn with_name(name: &str) -> Self {
        Self {
            dht: DarkSpotDHT::new(),
            signature_cache: std::collections::HashMap::new(),
            name: name.to_string(),
            stats: DarkSpotStats::default(),
        }
    }

    /// Handle a Dark Spot specific message
    pub fn handle_dark_spot_message(&mut self, msg: DarkSpotMessage) {
        match msg {
            DarkSpotMessage::PublishIgnorance {
                topic,
                category,
                eig,
                reply,
            } => {
                let response = self.publish_ignorance(&topic, &category, eig);
                let _ = reply.send(response);
            }

            DarkSpotMessage::QueryBlindSpots { reply } => {
                let response = self.query_blind_spots();
                let _ = reply.send(response);
            }

            DarkSpotMessage::FindKnowledge {
                signature_id,
                reply,
            } => {
                let response = self.find_knowledge(&signature_id);
                let _ = reply.send(response);
            }

            DarkSpotMessage::AddKnowledge {
                id,
                content,
                category,
                reputation,
                reply,
            } => {
                let response = self.add_knowledge(id, content, category, reputation);
                let _ = reply.send(response);
            }

            DarkSpotMessage::RevealBlindSpot {
                blind_spot_id,
                participants,
                reply,
            } => {
                let response = self.reveal_blind_spot(&blind_spot_id, participants);
                let _ = reply.send(response);
            }

            DarkSpotMessage::GetStats { reply } => {
                let response = self.get_stats();
                let _ = reply.send(response);
            }
        }
    }

    /// Publish ignorance to the DHT
    fn publish_ignorance(&mut self, topic: &str, category: &str, eig: f32) -> DarkSpotResponse {
        match self.dht.create_signature(topic, category, eig) {
            Ok(signature) => {
                let sig_id = signature.id.clone();

                match self.dht.publish(&signature) {
                    Ok(()) => {
                        // Cache the signature for later matching
                        self.signature_cache.insert(sig_id.clone(), signature);
                        self.stats.ignorance_published += 1;

                        info!(
                            actor = %self.name,
                            signature_id = %sig_id,
                            category = %category,
                            "Published ignorance signature"
                        );

                        DarkSpotResponse::Published {
                            signature_id: sig_id,
                        }
                    }
                    Err(e) => {
                        warn!(actor = %self.name, error = %e, "Failed to publish signature");
                        DarkSpotResponse::Error {
                            message: e.to_string(),
                        }
                    }
                }
            }
            Err(e) => {
                warn!(actor = %self.name, error = %e, "Failed to create signature");
                DarkSpotResponse::Error {
                    message: e.to_string(),
                }
            }
        }
    }

    /// Query for collective blind spots
    fn query_blind_spots(&mut self) -> DarkSpotResponse {
        let blind_spots = self.dht.detect_blind_spots();
        self.stats.blind_spots_detected = blind_spots.len();

        let spots: Vec<CollectiveBlindSpot> = blind_spots
            .into_iter()
            .map(CollectiveBlindSpot::from)
            .collect();

        debug!(
            actor = %self.name,
            count = spots.len(),
            "Detected blind spots"
        );

        DarkSpotResponse::BlindSpots { spots }
    }

    /// Find knowledge matching a signature
    fn find_knowledge(&mut self, signature_id: &str) -> DarkSpotResponse {
        match self.signature_cache.get(signature_id) {
            Some(signature) => {
                let matches = self.dht.find_matches(signature);
                self.stats.matches_found += matches.len();

                let results: Vec<MatchResult> =
                    matches.into_iter().map(MatchResult::from).collect();

                debug!(
                    actor = %self.name,
                    signature_id = %signature_id,
                    matches = results.len(),
                    "Found knowledge matches"
                );

                DarkSpotResponse::Matches { matches: results }
            }
            None => DarkSpotResponse::Error {
                message: format!("Signature {} not found in cache", signature_id),
            },
        }
    }

    /// Add knowledge to help others
    fn add_knowledge(
        &mut self,
        id: String,
        content: String,
        category: String,
        reputation: f32,
    ) -> DarkSpotResponse {
        self.dht
            .add_knowledge(id.clone(), content, category, reputation);
        self.stats.knowledge_added += 1;

        info!(
            actor = %self.name,
            knowledge_id = %id,
            "Added knowledge to DHT"
        );

        DarkSpotResponse::KnowledgeAdded { id }
    }

    /// Reveal a blind spot
    fn reveal_blind_spot(&mut self, blind_spot_id: &str, participants: usize) -> DarkSpotResponse {
        match self.dht.reveal_blind_spot(blind_spot_id, participants) {
            Some(revealed) => {
                info!(
                    actor = %self.name,
                    blind_spot_id = %revealed.id,
                    "Revealed blind spot"
                );

                DarkSpotResponse::BlindSpotRevealed {
                    id: revealed.id,
                    category: revealed.category,
                    revealed_topic: revealed.revealed_topic,
                }
            }
            None => DarkSpotResponse::Error {
                message: format!("Blind spot {} not found", blind_spot_id),
            },
        }
    }

    /// Get statistics
    fn get_stats(&self) -> DarkSpotResponse {
        let blind_spots = self.dht.detect_blind_spots();

        DarkSpotResponse::Stats {
            signature_count: self.stats.ignorance_published,
            knowledge_count: self.stats.knowledge_added,
            blind_spot_count: blind_spots.len(),
        }
    }

    /// Parse query text to extract ignorance request
    fn parse_ignorance_query(&self, question: &str) -> Option<(String, String, f32)> {
        // Simple parsing: "I don't know about [topic] in [category]"
        // Or: "ignorance: [topic] category=[category] eig=[value]"
        let lower = question.to_lowercase();

        if lower.contains("don't know") || lower.contains("ignorance:") {
            // Extract topic - everything after "about" or "ignorance:"
            let topic = if lower.contains("about ") {
                lower
                    .split("about ")
                    .nth(1)
                    .map(|s| s.split(" in ").next().unwrap_or(s))
                    .map(|s| s.trim().to_string())
            } else if lower.contains("ignorance:") {
                lower
                    .split("ignorance:")
                    .nth(1)
                    .map(|s| s.split("category=").next().unwrap_or(s))
                    .map(|s| s.trim().to_string())
            } else {
                None
            };

            // Extract category
            let category = if lower.contains(" in ") {
                lower.split(" in ").last().map(|s| s.trim().to_string())
            } else if lower.contains("category=") {
                lower
                    .split("category=")
                    .nth(1)
                    .map(|s| s.split_whitespace().next().unwrap_or(s))
                    .map(|s| s.trim().to_string())
            } else {
                Some("general".to_string())
            };

            // Extract EIG (default 0.5)
            let eig = if lower.contains("eig=") {
                lower
                    .split("eig=")
                    .nth(1)
                    .and_then(|s| s.split_whitespace().next())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.5)
            } else {
                0.5
            };

            if let (Some(t), Some(c)) = (topic, category) {
                return Some((t, c, eig));
            }
        }

        None
    }
}

impl Default for DarkSpotDHTActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Actor for DarkSpotDHTActor {
    async fn handle_message(&mut self, msg: OrganMessage) -> Result<()> {
        match msg {
            OrganMessage::Query {
                question,
                reply,
                hdc_semantic: _,
            } => {
                let response = if question.to_lowercase().contains("blind spot") {
                    // Query blind spots
                    match self.query_blind_spots() {
                        DarkSpotResponse::BlindSpots { spots } => {
                            let spot_info: Vec<String> = spots
                                .iter()
                                .map(|s| {
                                    format!(
                                        "- {}: {} signatures, avg EIG {:.2}",
                                        s.category, s.signature_count, s.average_eig
                                    )
                                })
                                .collect();

                            if spot_info.is_empty() {
                                "No collective blind spots detected yet.".to_string()
                            } else {
                                format!("Collective Blind Spots:\n{}", spot_info.join("\n"))
                            }
                        }
                        DarkSpotResponse::Error { message } => format!("Error: {}", message),
                        _ => "Unexpected response".to_string(),
                    }
                } else if question.to_lowercase().contains("stats") {
                    // Get statistics
                    match self.get_stats() {
                        DarkSpotResponse::Stats {
                            signature_count,
                            knowledge_count,
                            blind_spot_count,
                        } => {
                            format!(
                                "Dark Spot DHT Stats:\n\
                                 - Ignorance signatures: {}\n\
                                 - Knowledge entries: {}\n\
                                 - Blind spots: {}",
                                signature_count, knowledge_count, blind_spot_count
                            )
                        }
                        _ => "Error getting stats".to_string(),
                    }
                } else if let Some((topic, category, eig)) = self.parse_ignorance_query(&question) {
                    // Publish ignorance
                    match self.publish_ignorance(&topic, &category, eig) {
                        DarkSpotResponse::Published { signature_id } => {
                            format!(
                                "Published ignorance about '{}' in category '{}' (EIG: {:.2})\nSignature ID: {}",
                                topic, category, eig, signature_id
                            )
                        }
                        DarkSpotResponse::Error { message } => format!("Error: {}", message),
                        _ => "Unexpected response".to_string(),
                    }
                } else {
                    "Dark Spot DHT ready. Commands:\n\
                         - 'blind spots' - View collective ignorance patterns\n\
                         - 'stats' - View DHT statistics\n\
                         - \"I don't know about [topic] in [category]\" - Publish ignorance"
                        .to_string()
                };

                let _ = reply.send(response);
            }

            OrganMessage::Input {
                data: _,
                reply,
                hdc_semantic: _,
            } => {
                // Input messages could carry structured ignorance data
                let _ = reply.send(Response::Ok);
            }

            OrganMessage::Shutdown => {
                info!(actor = %self.name, "Dark Spot DHT Actor shutting down");
            }
        }

        Ok(())
    }

    fn priority(&self) -> ActorPriority {
        // Background priority - not time-critical
        ActorPriority::Background
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// Thread-Safe Wrapper for Shared Access
// ============================================================================

/// Thread-safe wrapper for DarkSpotDHTActor when shared across async contexts
pub struct SharedDarkSpotDHT {
    inner: Arc<RwLock<DarkSpotDHTActor>>,
}

impl SharedDarkSpotDHT {
    /// Create a new shared DHT actor
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(DarkSpotDHTActor::new())),
        }
    }

    /// Publish ignorance (async)
    pub async fn publish_ignorance(
        &self,
        topic: &str,
        category: &str,
        eig: f32,
    ) -> DarkSpotResponse {
        let mut actor = self.inner.write().await;
        actor.publish_ignorance(topic, category, eig)
    }

    /// Query blind spots (async)
    pub async fn query_blind_spots(&self) -> DarkSpotResponse {
        let mut actor = self.inner.write().await;
        actor.query_blind_spots()
    }

    /// Add knowledge (async)
    pub async fn add_knowledge(
        &self,
        id: String,
        content: String,
        category: String,
        reputation: f32,
    ) -> DarkSpotResponse {
        let mut actor = self.inner.write().await;
        actor.add_knowledge(id, content, category, reputation)
    }

    /// Find knowledge (async)
    pub async fn find_knowledge(&self, signature_id: &str) -> DarkSpotResponse {
        let mut actor = self.inner.write().await;
        actor.find_knowledge(signature_id)
    }

    /// Get statistics (async)
    pub async fn get_stats(&self) -> DarkSpotResponse {
        let actor = self.inner.read().await;
        actor.get_stats()
    }
}

impl Default for SharedDarkSpotDHT {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SharedDarkSpotDHT {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to extract expected response variant or return an error
    fn expect_published(response: DarkSpotResponse) -> Result<String, ActorError> {
        match response {
            DarkSpotResponse::Published { signature_id } => Ok(signature_id),
            other => {
                tracing::warn!("Received unexpected response: {:?}", other);
                Err(ActorError::UnexpectedResponse {
                    expected: "Published",
                    actual: format!("{:?}", other),
                })
            }
        }
    }

    fn expect_knowledge_added(response: DarkSpotResponse) -> Result<String, ActorError> {
        match response {
            DarkSpotResponse::KnowledgeAdded { id } => Ok(id),
            other => {
                tracing::warn!("Received unexpected response: {:?}", other);
                Err(ActorError::UnexpectedResponse {
                    expected: "KnowledgeAdded",
                    actual: format!("{:?}", other),
                })
            }
        }
    }

    fn expect_blind_spots(
        response: DarkSpotResponse,
    ) -> Result<Vec<CollectiveBlindSpot>, ActorError> {
        match response {
            DarkSpotResponse::BlindSpots { spots } => Ok(spots),
            other => {
                tracing::warn!("Received unexpected response: {:?}", other);
                Err(ActorError::UnexpectedResponse {
                    expected: "BlindSpots",
                    actual: format!("{:?}", other),
                })
            }
        }
    }

    fn expect_stats(response: DarkSpotResponse) -> Result<(usize, usize, usize), ActorError> {
        match response {
            DarkSpotResponse::Stats {
                signature_count,
                knowledge_count,
                blind_spot_count,
            } => Ok((signature_count, knowledge_count, blind_spot_count)),
            other => {
                tracing::warn!("Received unexpected response: {:?}", other);
                Err(ActorError::UnexpectedResponse {
                    expected: "Stats",
                    actual: format!("{:?}", other),
                })
            }
        }
    }

    #[test]
    fn test_dark_spot_actor_creation() {
        let actor = DarkSpotDHTActor::new();
        assert_eq!(actor.name(), "DarkSpotDHT");
    }

    #[test]
    fn test_publish_ignorance() -> Result<(), ActorError> {
        let mut actor = DarkSpotDHTActor::new();
        let response = actor.publish_ignorance("quantum entanglement", "physics", 0.8);

        let signature_id = expect_published(response)?;
        assert!(signature_id.starts_with("zk_physics_"));
        Ok(())
    }

    #[test]
    fn test_add_knowledge() -> Result<(), ActorError> {
        let mut actor = DarkSpotDHTActor::new();
        let response = actor.add_knowledge(
            "k1".to_string(),
            "Quantum entanglement is a phenomenon...".to_string(),
            "physics".to_string(),
            0.9,
        );

        let id = expect_knowledge_added(response)?;
        assert_eq!(id, "k1");
        Ok(())
    }

    #[test]
    fn test_query_blind_spots_empty() -> Result<(), ActorError> {
        let mut actor = DarkSpotDHTActor::new();
        let response = actor.query_blind_spots();

        let spots = expect_blind_spots(response)?;
        assert!(spots.is_empty());
        Ok(())
    }

    #[test]
    fn test_get_stats() -> Result<(), ActorError> {
        let mut actor = DarkSpotDHTActor::new();

        // Add some data
        actor.publish_ignorance("topic1", "cat1", 0.5);
        actor.add_knowledge(
            "k1".to_string(),
            "content".to_string(),
            "cat1".to_string(),
            0.8,
        );

        let response = actor.get_stats();

        let (signature_count, knowledge_count, _) = expect_stats(response)?;
        assert_eq!(signature_count, 1);
        assert_eq!(knowledge_count, 1);
        Ok(())
    }

    #[test]
    fn test_parse_ignorance_query() {
        let actor = DarkSpotDHTActor::new();

        // Test "I don't know about X in Y" format
        let result = actor.parse_ignorance_query("I don't know about quantum mechanics in physics");
        assert!(result.is_some());
        let (topic, category, _) = result.unwrap();
        assert_eq!(topic, "quantum mechanics");
        assert_eq!(category, "physics");

        // Test "ignorance:" format
        let result = actor.parse_ignorance_query("ignorance: neural networks category=ai eig=0.7");
        assert!(result.is_some());
        let (topic, category, eig) = result.unwrap();
        assert!(topic.contains("neural"));
        assert_eq!(category, "ai");
        assert!((eig - 0.7).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_shared_dht() -> Result<(), ActorError> {
        let shared = SharedDarkSpotDHT::new();

        // Publish from one "thread"
        let response = shared.publish_ignorance("topic", "category", 0.6).await;
        assert!(matches!(response, DarkSpotResponse::Published { .. }));

        // Query from another
        let stats = shared.get_stats().await;
        let (signature_count, _, _) = expect_stats(stats)?;
        assert_eq!(signature_count, 1);
        Ok(())
    }
}
