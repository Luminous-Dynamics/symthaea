/*!
P2P Swarm Intelligence - Collective Learning

Solves the "Isolation Problem" via hypervector swarm sharing:
1. **Gossipsub Protocol**: Efficient broadcast of hypervectors
2. **Kademlia DHT**: Distributed memory storage
3. **Collective Resonance**: Swarm consensus via vector addition
4. **Privacy-Preserving**: Share knowledge, not data

Each Sophia instance can learn from the swarm without centralized server.
*/

use anyhow::Result;
use libp2p::{
    gossipsub::{self, IdentTopic, MessageAuthenticity, Behaviour as GossipsubBehaviour},
    kad::{store::MemoryStore, Behaviour as KademliaBehaviour},
    noise, tcp, yamux, Multiaddr, PeerId, SwarmBuilder,
    swarm::NetworkBehaviour,
    futures::StreamExt,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, mpsc};

/// Combined network behaviour for Sophia swarm
#[derive(NetworkBehaviour)]
struct SophiaBehaviour {
    /// Gossipsub for pattern broadcasting
    gossipsub: GossipsubBehaviour,
    /// Kademlia for distributed storage
    kademlia: KademliaBehaviour<MemoryStore>,
}

/// Channel for outgoing messages
type MessageSender = mpsc::UnboundedSender<(IdentTopic, Vec<u8>)>;

/// Swarm message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmMessage {
    /// Share a learned pattern (hypervector + metadata)
    LearnedPattern {
        pattern: Vec<i8>,
        intent: String,
        confidence: f32,
        peer_id: String,
    },

    /// Query the swarm for similar patterns
    Query {
        query: Vec<i8>,
        context: String,
        requester: String,
    },

    /// Response to a query
    Response {
        patterns: Vec<Vec<i8>>,
        intents: Vec<String>,
        confidences: Vec<f32>,
        responder: String,
    },

    /// Heartbeat (keep-alive)
    Heartbeat {
        peer_id: String,
        timestamp: u64,
    },
}

/// P2P Swarm Intelligence Manager
pub struct SwarmIntelligence {
    /// libp2p peer ID
    peer_id: PeerId,

    /// Local knowledge cache (learned from swarm)
    knowledge_cache: Arc<RwLock<HashMap<String, Vec<i8>>>>,

    /// Peer statistics
    peer_stats: Arc<RwLock<HashMap<PeerId, PeerStats>>>,

    /// Swarm configuration
    config: SwarmConfig,

    /// Gossipsub topic for pattern sharing
    pattern_topic: IdentTopic,

    /// Channel to send messages to the swarm event loop
    message_tx: Option<MessageSender>,

    /// Swarm statistics
    stats: Arc<RwLock<SwarmStats>>,
}

/// Swarm-wide statistics
#[derive(Debug, Clone, Default)]
pub struct SwarmStats {
    /// Messages sent
    pub messages_sent: u64,
    /// Messages received
    pub messages_received: u64,
    /// Patterns shared
    pub patterns_shared: u64,
    /// Queries sent
    pub queries_sent: u64,
    /// Connected peers
    pub connected_peers: usize,
}

/// Peer statistics
#[derive(Debug, Clone)]
pub struct PeerStats {
    pub patterns_received: usize,
    pub queries_answered: usize,
    pub last_seen: std::time::SystemTime,
    pub reputation: f32,  // 0.0 to 1.0
}

impl Default for PeerStats {
    fn default() -> Self {
        Self {
            patterns_received: 0,
            queries_answered: 0,
            last_seen: std::time::SystemTime::UNIX_EPOCH,
            reputation: 0.5,  // Neutral starting reputation
        }
    }
}

/// Swarm configuration
#[derive(Debug, Clone)]
pub struct SwarmConfig {
    /// Enable swarm learning
    pub enabled: bool,

    /// Gossipsub topic for patterns
    pub pattern_topic: String,

    /// Maximum peers to connect
    pub max_peers: usize,

    /// Minimum reputation to trust pattern
    pub min_reputation: f32,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            pattern_topic: "sophia-patterns".to_string(),
            max_peers: 50,
            min_reputation: 0.6,
        }
    }
}

impl SwarmIntelligence {
    /// Create new swarm intelligence manager
    pub async fn new(config: SwarmConfig) -> Result<Self> {
        let keypair = libp2p::identity::Keypair::generate_ed25519();
        let peer_id = keypair.public().to_peer_id();

        // Create the gossipsub topic
        let pattern_topic = IdentTopic::new(&config.pattern_topic);

        tracing::info!("🐝 Initializing Swarm Intelligence: {}", peer_id);
        tracing::info!("   Topic: {}", config.pattern_topic);

        Ok(Self {
            peer_id,
            knowledge_cache: Arc::new(RwLock::new(HashMap::new())),
            peer_stats: Arc::new(RwLock::new(HashMap::new())),
            pattern_topic,
            message_tx: None,  // Set when swarm is started
            stats: Arc::new(RwLock::new(SwarmStats::default())),
            config,
        })
    }

    /// Start the P2P swarm event loop
    ///
    /// This spawns a background task that handles:
    /// - Connecting to bootstrap peers
    /// - Processing incoming messages
    /// - Sending outgoing messages via the channel
    pub async fn start(&mut self, bootstrap_peers: Vec<Multiaddr>) -> Result<()> {
        if !self.config.enabled {
            tracing::info!("🐝 Swarm disabled, skipping start");
            return Ok(());
        }

        let keypair = libp2p::identity::Keypair::generate_ed25519();
        let local_peer_id = keypair.public().to_peer_id();

        // Configure gossipsub
        let gossipsub_config = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(Duration::from_secs(1))
            .validation_mode(gossipsub::ValidationMode::Strict)
            .build()
            .map_err(|e| anyhow::anyhow!("Gossipsub config error: {}", e))?;

        let gossipsub = GossipsubBehaviour::new(
            MessageAuthenticity::Signed(keypair.clone()),
            gossipsub_config,
        ).map_err(|e| anyhow::anyhow!("Gossipsub creation error: {}", e))?;

        // Configure Kademlia
        let kademlia = KademliaBehaviour::new(local_peer_id, MemoryStore::new(local_peer_id));

        // Combine into our behaviour
        let behaviour = SophiaBehaviour { gossipsub, kademlia };

        // Build the swarm
        let mut swarm = SwarmBuilder::with_existing_identity(keypair)
            .with_tokio()
            .with_tcp(
                tcp::Config::default(),
                noise::Config::new,
                yamux::Config::default,
            )?
            .with_behaviour(|_| behaviour)?
            .build();

        // Subscribe to pattern topic
        swarm.behaviour_mut().gossipsub.subscribe(&self.pattern_topic)?;

        // Listen on a random port
        swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

        // Connect to bootstrap peers
        for addr in bootstrap_peers {
            tracing::info!("🐝 Dialing bootstrap peer: {}", addr);
            swarm.dial(addr)?;
        }

        // Create message channel
        let (tx, mut rx) = mpsc::unbounded_channel::<(IdentTopic, Vec<u8>)>();
        self.message_tx = Some(tx);

        // Clone data for the event loop
        let knowledge_cache = self.knowledge_cache.clone();
        let _peer_stats = self.peer_stats.clone();
        let stats = self.stats.clone();
        let _pattern_topic = self.pattern_topic.clone();

        // Spawn the event loop
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    // Handle outgoing messages
                    Some((topic, data)) = rx.recv() => {
                        if let Err(e) = swarm.behaviour_mut().gossipsub.publish(topic, data) {
                            tracing::warn!("Failed to publish message: {}", e);
                        } else {
                            let mut s = stats.write().await;
                            s.messages_sent += 1;
                        }
                    }
                    // Handle swarm events
                    event = swarm.select_next_some() => {
                        use libp2p::swarm::SwarmEvent;
                        match event {
                            SwarmEvent::Behaviour(SophiaBehaviourEvent::Gossipsub(
                                gossipsub::Event::Message { message, .. }
                            )) => {
                                // Decode and process incoming message
                                if let Ok(msg) = bincode::deserialize::<SwarmMessage>(&message.data) {
                                    let mut s = stats.write().await;
                                    s.messages_received += 1;
                                    drop(s);

                                    match msg {
                                        SwarmMessage::LearnedPattern { pattern, intent, confidence, peer_id } => {
                                            tracing::info!("📥 Received pattern '{}' from {} (conf: {:.1}%)",
                                                intent, peer_id, confidence * 100.0);
                                            let mut cache = knowledge_cache.write().await;
                                            cache.insert(intent, pattern);
                                        }
                                        SwarmMessage::Query { query: _query, context, requester } => {
                                            tracing::info!("🔍 Received query from {}: {}", requester, context);
                                            // TODO: Search local cache and respond
                                        }
                                        SwarmMessage::Response { patterns, intents: _intents, confidences: _confidences, responder } => {
                                            tracing::info!("📬 Received {} patterns from {}", patterns.len(), responder);
                                        }
                                        SwarmMessage::Heartbeat { peer_id, timestamp } => {
                                            tracing::debug!("💓 Heartbeat from {} at {}", peer_id, timestamp);
                                        }
                                    }
                                }
                            }
                            SwarmEvent::NewListenAddr { address, .. } => {
                                tracing::info!("🐝 Listening on {}", address);
                            }
                            SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                                tracing::info!("🤝 Connected to peer: {}", peer_id);
                                let mut s = stats.write().await;
                                s.connected_peers += 1;
                            }
                            SwarmEvent::ConnectionClosed { peer_id, .. } => {
                                tracing::info!("👋 Disconnected from peer: {}", peer_id);
                                let mut s = stats.write().await;
                                s.connected_peers = s.connected_peers.saturating_sub(1);
                            }
                            _ => {}
                        }
                    }
                }
            }
        });

        tracing::info!("🐝 Swarm event loop started");
        Ok(())
    }

    /// Get swarm statistics
    pub async fn stats(&self) -> SwarmStats {
        self.stats.read().await.clone()
    }

    /// Broadcast learned pattern to swarm
    pub async fn share_pattern(
        &self,
        pattern: Vec<i8>,
        intent: String,
        confidence: f32,
    ) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let message = SwarmMessage::LearnedPattern {
            pattern,
            intent,
            confidence,
            peer_id: self.peer_id.to_string(),
        };

        // Serialize and broadcast via gossipsub
        let payload = bincode::serialize(&message)?;

        tracing::info!(
            "📡 Broadcasting learned pattern to swarm (confidence: {:.1}%)",
            confidence * 100.0
        );

        // Send via message channel to the swarm event loop
        if let Some(ref tx) = self.message_tx {
            tx.send((self.pattern_topic.clone(), payload))
                .map_err(|e| anyhow::anyhow!("Failed to send to swarm: {}", e))?;

            // Update stats
            let mut stats = self.stats.write().await;
            stats.patterns_shared += 1;
        } else {
            tracing::warn!("🐝 Swarm not started - message not sent");
        }

        Ok(())
    }

    /// Query swarm for similar patterns
    pub async fn query_swarm(&self, query: Vec<i8>, context: String) -> Result<Vec<SwarmResponse>> {
        if !self.config.enabled {
            return Ok(vec![]);
        }

        tracing::info!("🔍 Querying swarm for similar patterns");

        let message = SwarmMessage::Query {
            query,
            context,
            requester: self.peer_id.to_string(),
        };

        // Serialize and broadcast query
        let payload = bincode::serialize(&message)?;

        // Send via message channel
        if let Some(ref tx) = self.message_tx {
            tx.send((self.pattern_topic.clone(), payload))
                .map_err(|e| anyhow::anyhow!("Failed to send query to swarm: {}", e))?;

            // Update stats
            let mut stats = self.stats.write().await;
            stats.queries_sent += 1;
        } else {
            tracing::warn!("🐝 Swarm not started - query not sent");
        }

        // Note: In a full implementation, we'd wait for responses
        // For now, check local knowledge cache for matches
        let cache = self.knowledge_cache.read().await;
        let responses: Vec<SwarmResponse> = cache
            .iter()
            .take(5)  // Limit to 5 results
            .map(|(intent, pattern)| SwarmResponse {
                pattern: pattern.clone(),
                intent: intent.clone(),
                confidence: 0.8,  // Default confidence for cached patterns
                source_peer: self.peer_id,
            })
            .collect();

        Ok(responses)
    }

    /// Receive pattern from peer
    pub async fn receive_pattern(
        &self,
        peer_id: PeerId,
        pattern: Vec<i8>,
        intent: String,
        confidence: f32,
    ) -> Result<()> {
        // Check peer reputation
        let reputation = {
            let stats = self.peer_stats.read().await;
            stats
                .get(&peer_id)
                .map(|s| s.reputation)
                .unwrap_or(1.0)  // New peers start with full reputation
        };

        if reputation < self.config.min_reputation {
            tracing::warn!("⚠️  Ignoring pattern from low-reputation peer: {}", peer_id);
            return Ok(());
        }

        // Store in knowledge cache
        {
            let mut cache = self.knowledge_cache.write().await;
            cache.insert(intent.clone(), pattern);
        }

        // Update peer stats
        {
            let mut stats = self.peer_stats.write().await;
            let peer_stat = stats.entry(peer_id).or_insert(PeerStats::default());
            peer_stat.patterns_received += 1;
            peer_stat.last_seen = std::time::SystemTime::now();

            // Boost reputation for high-confidence patterns
            if confidence > 0.9 {
                peer_stat.reputation = (peer_stat.reputation + 0.05).min(1.0);
            }
        }

        tracing::info!(
            "✅ Received pattern from peer {} (confidence: {:.1}%)",
            peer_id,
            confidence * 100.0
        );

        Ok(())
    }

    /// Collective resonance: Aggregate patterns from multiple peers
    ///
    /// Combines hypervectors via bundling (superposition) to find consensus
    pub async fn collective_resonance(
        &self,
        patterns: Vec<Vec<i8>>,
    ) -> Result<Vec<i8>> {
        if patterns.is_empty() {
            return Ok(vec![0i8; 10_000]);
        }

        let dim = patterns[0].len();
        let mut bundled = vec![0i32; dim];

        // Bundle (superposition): Add all patterns
        for pattern in &patterns {
            for i in 0..dim {
                bundled[i] += pattern[i] as i32;
            }
        }

        // Sign determines consensus (-1 or +1)
        let consensus: Vec<i8> = bundled
            .iter()
            .map(|&x| if x >= 0 { 1 } else { -1 })
            .collect();

        tracing::info!(
            "🌊 Collective resonance computed from {} patterns",
            patterns.len()
        );

        Ok(consensus)
    }

    /// Update peer reputation (reward/penalize)
    pub async fn update_reputation(&self, peer_id: PeerId, delta: f32) {
        let mut stats = self.peer_stats.write().await;

        if let Some(peer_stat) = stats.get_mut(&peer_id) {
            peer_stat.reputation = (peer_stat.reputation + delta).clamp(0.0, 1.0);

            tracing::info!(
                "Updated reputation for {}: {:.2}",
                peer_id,
                peer_stat.reputation
            );
        }
    }

    /// Ban peer (set reputation to 0)
    pub async fn ban_peer(&self, peer_id: PeerId) {
        let mut stats = self.peer_stats.write().await;

        if let Some(peer_stat) = stats.get_mut(&peer_id) {
            peer_stat.reputation = 0.0;
            tracing::warn!("🚫 Banned peer: {}", peer_id);
        }
    }

    /// Get peer ID
    pub fn peer_id(&self) -> PeerId {
        self.peer_id
    }
}

/// Swarm response (from query)
#[derive(Debug, Clone)]
pub struct SwarmResponse {
    pub pattern: Vec<i8>,
    pub intent: String,
    pub confidence: f32,
    pub source_peer: PeerId,
}

// ============================================================================
// Swarm Advisor - Decision Logic for Mycelix Integration
// ============================================================================
// NOTE: This section is disabled until sophia_swarm module is integrated.
// TODO: Re-enable when Mycelix protocol integration is complete.
// See: https://github.com/Luminous-Dynamics/mycelix for protocol details.
// ============================================================================

/*
use crate::sophia_swarm::{
    CompositeTrustScore, DkgClient, EvaluatedClaim, MatlClient, MfdiClient, PatternQuery,
    SophiaPattern, SophiaSwarmClient,
};

/// Suggestion decision types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionDecisionKind {
    /// Safe and high-quality - auto-apply without user confirmation
    AutoApply,

    /// Reasonable but needs confirmation - ask user first
    AskUser,

    /// Too risky - reject and don't present to user
    Reject,
}

/// Evaluated suggestion with decision
#[derive(Debug, Clone)]
pub struct SuggestionDecision {
    pub decision: SuggestionDecisionKind,
    pub claim: EvaluatedClaim,
    pub reason: String,
}

/// SwarmAdvisor - Wraps SophiaSwarmClient and makes trust-based decisions
///
/// This wraps any Mycelix swarm client and translates MATL trust scores into
/// actionable decisions:
/// - High trust (≥0.8) → Auto-apply (safe to execute immediately)
/// - Medium trust (0.5-0.8) → Ask user (show suggestion, require confirmation)
/// - Low trust (<0.5) → Reject (don't even show)
pub struct SwarmAdvisor<C: SophiaSwarmClient> {
    /// Underlying Mycelix swarm client
    client: Arc<C>,

    /// Minimum trust score for auto-apply (default: 0.8)
    pub auto_apply_min_trust: f64,

    /// Minimum trust score to show user (default: 0.5)
    pub ask_user_min_trust: f64,
}

impl<C: SophiaSwarmClient> SwarmAdvisor<C> {
    /// Create new SwarmAdvisor with default thresholds
    pub fn new(client: Arc<C>) -> Self {
        Self {
            client,
            auto_apply_min_trust: 0.8,
            ask_user_min_trust: 0.5,
        }
    }

    /// Create with custom trust thresholds
    pub fn with_thresholds(client: Arc<C>, auto_apply: f64, ask_user: f64) -> Self {
        Self {
            client,
            auto_apply_min_trust: auto_apply,
            ask_user_min_trust: ask_user,
        }
    }

    /// Query swarm and make decisions about suggestions
    pub async fn get_suggestions(
        &self,
        query: PatternQuery,
        limit: usize,
    ) -> Result<Vec<SuggestionDecision>> {
        // Query DKG for matching patterns
        let claims = self.client.query_claims(query, limit).await?;

        // Evaluate each claim and make decision
        let mut suggestions = Vec::new();

        for claim in claims {
            let decision = self.evaluate_claim(&claim);
            suggestions.push(decision);
        }

        // Filter out rejected suggestions
        suggestions.retain(|s| s.decision != SuggestionDecisionKind::Reject);

        // Sort by trust score (highest first)
        suggestions.sort_by(|a, b| {
            b.claim
                .trust_score
                .composite
                .partial_cmp(&a.claim.trust_score.composite)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(suggestions)
    }

    /// Evaluate a claim and decide what to do with it
    fn evaluate_claim(&self, claim: &EvaluatedClaim) -> SuggestionDecision {
        let trust = claim.trust_score.composite;

        // Check for cartel risk in the trust score
        // (MATL includes cartel detection in composite score)
        let is_suspicious = claim.trust_score.tcdm_score < 0.6;

        // Decide based on trust score
        let (decision, reason) = if is_suspicious {
            (
                SuggestionDecisionKind::Reject,
                format!(
                    "Cartel risk detected (TCDM: {:.2})",
                    claim.trust_score.tcdm_score
                ),
            )
        } else if trust >= self.auto_apply_min_trust {
            (
                SuggestionDecisionKind::AutoApply,
                format!(
                    "High trust score: {:.2} (PoGQ: {:.2}, TCDM: {:.2})",
                    trust, claim.trust_score.pogq_score, claim.trust_score.tcdm_score
                ),
            )
        } else if trust >= self.ask_user_min_trust {
            (
                SuggestionDecisionKind::AskUser,
                format!("Medium trust score: {:.2} - requires confirmation", trust),
            )
        } else {
            (
                SuggestionDecisionKind::Reject,
                format!("Trust score too low: {:.2}", trust),
            )
        };

        SuggestionDecision {
            decision,
            claim: claim.clone(),
            reason,
        }
    }

    /// Publish pattern to swarm
    pub async fn publish_pattern(&self, pattern: SophiaPattern) -> Result<()> {
        self.client.publish_pattern_claim(pattern).await?;
        Ok(())
    }

    /// Get trust score for an agent
    pub async fn agent_trust(&self, did: &str) -> Result<CompositeTrustScore> {
        self.client.trust_for_agent(did).await
    }
}
*/

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_swarm_creation() {
        let swarm = SwarmIntelligence::new(SwarmConfig::default())
            .await
            .unwrap();

        assert!(swarm.peer_id.to_string().len() > 0);
    }

    #[tokio::test]
    async fn test_collective_resonance() {
        let swarm = SwarmIntelligence::new(SwarmConfig::default())
            .await
            .unwrap();

        let patterns = vec![
            vec![1i8; 100],
            vec![1i8; 100],
            vec![-1i8; 100],
        ];

        let consensus = swarm.collective_resonance(patterns).await.unwrap();

        // Majority should win (2 votes for +1, 1 for -1)
        assert_eq!(consensus[0], 1);
    }

    #[tokio::test]
    async fn test_reputation_system() {
        let swarm = SwarmIntelligence::new(SwarmConfig::default())
            .await
            .unwrap();

        let fake_peer = PeerId::random();

        // Receive pattern (initializes reputation)
        swarm
            .receive_pattern(fake_peer, vec![1; 10_000], "test".to_string(), 0.95)
            .await
            .unwrap();

        // Check reputation was updated
        let stats = swarm.peer_stats.read().await;
        let peer_stat = stats.get(&fake_peer).unwrap();
        assert!(peer_stat.reputation > 1.0);  // Should be boosted
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let swarm = SwarmIntelligence::new(SwarmConfig::default())
            .await
            .unwrap();

        let stats = swarm.stats().await;
        assert_eq!(stats.connected_peers, 0);
    }
}
