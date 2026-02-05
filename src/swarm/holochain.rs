//! Holochain Cortex Layer - Trust & Identity for the Nervous System
//!
//! This module implements the "Cortex" of Symthaea's nervous system:
//! the long-term memory, trust graph, and identity verification layer
//! backed by Holochain DHT.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                       HOLOCHAIN CORTEX LAYER                                │
//! │                                                                             │
//! │  ┌─────────────────────────────────────────────────────────────────────┐   │
//! │  │                    AGENT IDENTITY                                    │   │
//! │  │                                                                      │   │
//! │  │  • Ed25519 Public Key (AgentPubKey)                                 │   │
//! │  │  • Derived from private key seed                                     │   │
//! │  │  • Unique identifier across all Holochain apps                       │   │
//! │  └─────────────────────────────────────────────────────────────────────┘   │
//! │                                  │                                          │
//! │                                  ▼                                          │
//! │  ┌─────────────────────────────────────────────────────────────────────┐   │
//! │  │                    DHT TRUST GRAPH                                   │   │
//! │  │                                                                      │   │
//! │  │  AgentInfo {                                                         │   │
//! │  │    agent_key: "uhCAk...",                                            │   │
//! │  │    phi_history: [0.72, 0.85, 0.91],                                  │   │
//! │  │    reputation_score: 0.89,                                           │   │
//! │  │    vouched_by: [...agent_keys...],                                   │   │
//! │  │    revocations: [...],                                               │   │
//! │  │  }                                                                   │   │
//! │  └─────────────────────────────────────────────────────────────────────┘   │
//! │                                  │                                          │
//! │                                  ▼                                          │
//! │  ┌─────────────────────────────────────────────────────────────────────┐   │
//! │  │                    TRUST VERIFICATION                                │   │
//! │  │                                                                      │   │
//! │  │  1. Agent submits signed challenge                                   │   │
//! │  │  2. Verify Ed25519 signature                                         │   │
//! │  │  3. Query DHT for AgentInfo                                          │   │
//! │  │  4. Check reputation_score >= min_trust                              │   │
//! │  │  5. Return TrustLevel                                                │   │
//! │  └─────────────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Production Integration
//!
//! To connect to a real Holochain conductor:
//!
//! ```rust,ignore
//! // Add to Cargo.toml:
//! // holochain_client = "0.6"
//! // holochain_zome_types = "0.4"
//!
//! use holochain_client::{AppWebsocket, InstalledAppInfo};
//!
//! let mut ws = AppWebsocket::connect("ws://localhost:8888").await?;
//! let info = ws.app_info("symthaea-trust".into()).await?;
//! ```

use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// AGENT IDENTITY
// ============================================================================

/// Holochain Agent Public Key (placeholder type)
///
/// In production, this would be `holochain_zome_types::AgentPubKey`
/// which is a 39-byte Ed25519 public key in Base64 format.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentPubKey(String);

impl AgentPubKey {
    /// Create from a string
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }

    /// Get the key as a string
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Validate key format (simplified)
    pub fn is_valid(&self) -> bool {
        // Real format: uhCAk... (44 chars, Base64)
        self.0.len() >= 8 && (self.0.starts_with("uhCAk") || self.0.starts_with("test_"))
    }

    /// Create a test key
    pub fn test_key(id: u32) -> Self {
        Self(format!("test_agent_{:08x}", id))
    }
}

impl std::fmt::Display for AgentPubKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.0[..8.min(self.0.len())])
    }
}

/// Agent information stored in the DHT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    /// The agent's public key
    pub agent_key: AgentPubKey,

    /// Nickname (optional, set by agent)
    pub nickname: Option<String>,

    /// Historical Φ values (consciousness metric)
    pub phi_history: Vec<f64>,

    /// Computed reputation score (0.0 to 1.0)
    pub reputation_score: f64,

    /// Agents who have vouched for this agent
    pub vouched_by: Vec<AgentPubKey>,

    /// Count of successful interactions
    pub interaction_count: u64,

    /// Count of reported issues/violations
    pub violation_count: u64,

    /// When this agent first joined the network
    pub joined_at: u64,

    /// Last activity timestamp
    pub last_active: u64,

    /// Agent's declared capabilities
    pub capabilities: Vec<String>,

    /// Whether the agent is currently active
    pub is_active: bool,
}

impl AgentInfo {
    /// Create new agent info with default values
    pub fn new(agent_key: AgentPubKey) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            agent_key,
            nickname: None,
            phi_history: Vec::new(),
            reputation_score: 0.5, // Start neutral
            vouched_by: Vec::new(),
            interaction_count: 0,
            violation_count: 0,
            joined_at: now,
            last_active: now,
            capabilities: Vec::new(),
            is_active: true,
        }
    }

    /// Update reputation based on interactions
    pub fn update_reputation(&mut self) {
        // Simple reputation formula:
        // base_trust + vouch_bonus - violation_penalty
        let vouch_bonus = (self.vouched_by.len() as f64 * 0.05).min(0.3);
        let violation_penalty = (self.violation_count as f64 * 0.1).min(0.5);
        let interaction_bonus = (self.interaction_count as f64 / 1000.0).min(0.2);

        self.reputation_score = (0.5 + vouch_bonus + interaction_bonus - violation_penalty)
            .clamp(0.0, 1.0);
    }

    /// Record a new Φ reading
    pub fn record_phi(&mut self, phi: f64) {
        self.phi_history.push(phi);
        if self.phi_history.len() > 100 {
            self.phi_history.remove(0);
        }
        self.last_active = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
    }

    /// Get average Φ value
    pub fn average_phi(&self) -> f64 {
        if self.phi_history.is_empty() {
            return 0.0;
        }
        self.phi_history.iter().sum::<f64>() / self.phi_history.len() as f64
    }
}

// ============================================================================
// TRUST VERIFICATION
// ============================================================================

/// A signed challenge response from an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedChallenge {
    /// The original nonce
    pub nonce: Vec<u8>,

    /// Ed25519 signature over the nonce
    pub signature: Vec<u8>,

    /// The agent's public key
    pub agent_key: AgentPubKey,

    /// Timestamp when signed
    pub signed_at: u64,
}

/// Result of trust verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustVerificationResult {
    /// Whether verification succeeded
    pub verified: bool,

    /// The agent's public key
    pub agent_key: Option<AgentPubKey>,

    /// Trust score (0.0 to 1.0)
    pub trust_score: f64,

    /// Agent info from DHT
    pub agent_info: Option<AgentInfo>,

    /// Reason for failure (if any)
    pub error: Option<String>,
}

impl TrustVerificationResult {
    /// Create a successful result
    pub fn success(agent_key: AgentPubKey, trust_score: f64, agent_info: AgentInfo) -> Self {
        Self {
            verified: true,
            agent_key: Some(agent_key),
            trust_score,
            agent_info: Some(agent_info),
            error: None,
        }
    }

    /// Create a failed result
    pub fn failure(reason: impl Into<String>) -> Self {
        Self {
            verified: false,
            agent_key: None,
            trust_score: 0.0,
            agent_info: None,
            error: Some(reason.into()),
        }
    }
}

// ============================================================================
// HOLOCHAIN CLIENT
// ============================================================================

/// Configuration for the Holochain client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolochainConfig {
    /// WebSocket URL for the Holochain conductor
    pub conductor_url: String,

    /// App ID for the trust zome
    pub app_id: String,

    /// Zome name
    pub zome_name: String,

    /// Connection timeout
    pub timeout_ms: u64,

    /// Minimum reputation to trust
    pub min_reputation: f64,

    /// Enable mock mode (no real Holochain connection)
    pub mock_mode: bool,
}

impl Default for HolochainConfig {
    fn default() -> Self {
        Self {
            conductor_url: "ws://localhost:8888".to_string(),
            app_id: "symthaea-trust".to_string(),
            zome_name: "trust".to_string(),
            timeout_ms: 5000,
            min_reputation: 0.3,
            mock_mode: true, // Default to mock for development
        }
    }
}

/// The Holochain Cortex client
///
/// Provides trust verification and identity services via Holochain DHT.
#[derive(Debug)]
pub struct HolochainCortex {
    /// Configuration
    config: HolochainConfig,

    /// Local agent key (our identity)
    local_agent: Option<AgentPubKey>,

    /// Cached agent info (mock mode) - LRU cache with bounded capacity
    agent_cache: LruCache<AgentPubKey, AgentInfo>,

    /// Connection state
    connected: bool,

    /// Statistics
    stats: CortexStats,
}

/// Statistics for the Cortex
#[derive(Debug, Clone, Default)]
pub struct CortexStats {
    /// Total verifications attempted
    pub verifications_attempted: u64,

    /// Successful verifications
    pub verifications_succeeded: u64,

    /// Failed verifications
    pub verifications_failed: u64,

    /// DHT queries made
    pub dht_queries: u64,

    /// Cache hits
    pub cache_hits: u64,

    /// Cache misses
    pub cache_misses: u64,

    /// Cache evictions (entries removed due to capacity limit)
    pub cache_evictions: u64,
}

/// Cache statistics snapshot
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of evictions
    pub evictions: u64,
    /// Current cache size
    pub current_size: usize,
    /// Maximum cache capacity
    pub capacity: usize,
    /// Hit rate (hits / (hits + misses))
    pub hit_rate: f64,
}

/// Default capacity for the agent cache
const DEFAULT_CACHE_CAPACITY: usize = 1000;

impl HolochainCortex {
    /// Create a new Holochain Cortex client with default cache capacity (1000 agents)
    pub fn new(config: HolochainConfig) -> Self {
        Self::with_cache_capacity(config, DEFAULT_CACHE_CAPACITY)
    }

    /// Create a new Holochain Cortex client with custom cache capacity
    pub fn with_cache_capacity(config: HolochainConfig, cache_capacity: usize) -> Self {
        let capacity = NonZeroUsize::new(cache_capacity).unwrap_or(NonZeroUsize::new(1).unwrap());
        Self {
            config,
            local_agent: None,
            agent_cache: LruCache::new(capacity),
            connected: false,
            stats: CortexStats::default(),
        }
    }

    /// Connect to the Holochain conductor
    ///
    /// In production, this would:
    /// ```rust,ignore
    /// let ws = AppWebsocket::connect(&self.config.conductor_url).await?;
    /// self.client = Some(ws);
    /// ```
    pub async fn connect(&mut self) -> Result<(), CortexError> {
        if self.config.mock_mode {
            self.connected = true;
            tracing::info!("HolochainCortex connected (MOCK MODE)");
            return Ok(());
        }

        // TODO: Real Holochain connection
        // let ws = AppWebsocket::connect(&self.config.conductor_url).await?;
        // self.client = Some(ws);

        Err(CortexError::NotImplemented(
            "Real Holochain connection not yet implemented".into(),
        ))
    }

    /// Set the local agent key
    pub fn set_local_agent(&mut self, agent_key: AgentPubKey) {
        self.local_agent = Some(agent_key.clone());

        // Create local agent info if not exists (use contains to check without marking as used)
        if !self.agent_cache.contains(&agent_key) {
            self.put_agent_info(agent_key.clone(), AgentInfo::new(agent_key));
        }
    }

    /// Get the local agent key
    pub fn local_agent(&self) -> Option<&AgentPubKey> {
        self.local_agent.as_ref()
    }

    /// Verify a signed challenge
    pub fn verify_challenge(&mut self, challenge: &SignedChallenge) -> TrustVerificationResult {
        self.stats.verifications_attempted += 1;

        // Validate agent key format
        if !challenge.agent_key.is_valid() {
            self.stats.verifications_failed += 1;
            return TrustVerificationResult::failure("Invalid agent key format");
        }

        // In production: verify Ed25519 signature
        // use ed25519_dalek::{Verifier, PublicKey, Signature};
        // let pk = PublicKey::from_bytes(...)?;
        // pk.verify(&challenge.nonce, &signature)?;

        // For now, mock verification (accept if key looks valid)
        if challenge.signature.len() < 32 {
            self.stats.verifications_failed += 1;
            return TrustVerificationResult::failure("Invalid signature length");
        }

        // Look up or create agent info
        let agent_info = self.get_or_create_agent_info(&challenge.agent_key);
        let trust_score = agent_info.reputation_score;

        // Check minimum trust
        if trust_score < self.config.min_reputation {
            self.stats.verifications_failed += 1;
            return TrustVerificationResult::failure(format!(
                "Reputation {} below minimum {}",
                trust_score, self.config.min_reputation
            ));
        }

        self.stats.verifications_succeeded += 1;
        TrustVerificationResult::success(challenge.agent_key.clone(), trust_score, agent_info)
    }

    /// Get or create agent info (mock mode)
    fn get_or_create_agent_info(&mut self, agent_key: &AgentPubKey) -> AgentInfo {
        if let Some(info) = self.agent_cache.get(agent_key) {
            self.stats.cache_hits += 1;
            return info.clone();
        }

        self.stats.cache_misses += 1;
        self.stats.dht_queries += 1;
        let info = AgentInfo::new(agent_key.clone());
        self.put_agent_info(agent_key.clone(), info.clone());
        info
    }

    /// Insert agent info into cache, tracking evictions
    fn put_agent_info(&mut self, agent_key: AgentPubKey, info: AgentInfo) {
        // Check if we're at capacity and this is a new key
        let was_at_capacity = self.agent_cache.len() == self.agent_cache.cap().get();
        let is_new_key = !self.agent_cache.contains(&agent_key);

        if was_at_capacity && is_new_key {
            self.stats.cache_evictions += 1;
        }

        self.agent_cache.put(agent_key, info);
    }

    /// Query agent info from DHT
    pub async fn query_agent(&mut self, agent_key: &AgentPubKey) -> Result<AgentInfo, CortexError> {
        if self.config.mock_mode {
            return Ok(self.get_or_create_agent_info(agent_key));
        }

        // TODO: Real DHT query
        // let response = self.client.call_zome(
        //     self.config.zome_name.clone(),
        //     "get_agent_info",
        //     agent_key,
        // ).await?;

        Err(CortexError::NotImplemented(
            "Real DHT query not yet implemented".into(),
        ))
    }

    /// Record a successful interaction with an agent
    pub fn record_interaction(&mut self, agent_key: &AgentPubKey) {
        if let Some(info) = self.agent_cache.get_mut(agent_key) {
            info.interaction_count += 1;
            info.update_reputation();
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        let total_accesses = self.stats.cache_hits + self.stats.cache_misses;
        let hit_rate = if total_accesses > 0 {
            self.stats.cache_hits as f64 / total_accesses as f64
        } else {
            0.0
        };

        CacheStats {
            hits: self.stats.cache_hits,
            misses: self.stats.cache_misses,
            evictions: self.stats.cache_evictions,
            current_size: self.agent_cache.len(),
            capacity: self.agent_cache.cap().get(),
            hit_rate,
        }
    }

    /// Record a violation by an agent
    pub fn record_violation(&mut self, agent_key: &AgentPubKey, reason: &str) {
        if let Some(info) = self.agent_cache.get_mut(agent_key) {
            info.violation_count += 1;
            info.update_reputation();
            tracing::warn!(
                agent = %agent_key,
                reason = reason,
                "Recorded violation for agent"
            );
        }
    }

    /// Peek at agent info without marking as recently used
    pub fn peek_agent(&self, agent_key: &AgentPubKey) -> Option<&AgentInfo> {
        self.agent_cache.peek(agent_key)
    }

    /// Vouch for another agent
    pub fn vouch_for(&mut self, agent_key: &AgentPubKey) -> Result<(), CortexError> {
        let local = self.local_agent.as_ref()
            .ok_or(CortexError::NoLocalAgent)?
            .clone();

        if let Some(info) = self.agent_cache.get_mut(agent_key) {
            if !info.vouched_by.contains(&local) {
                info.vouched_by.push(local);
                info.update_reputation();
            }
            Ok(())
        } else {
            Err(CortexError::AgentNotFound(agent_key.clone()))
        }
    }

    /// Record Phi for an agent
    pub fn record_phi(&mut self, agent_key: &AgentPubKey, phi: f64) {
        if let Some(info) = self.agent_cache.get_mut(agent_key) {
            info.record_phi(phi);
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &CortexStats {
        &self.stats
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Get configuration
    pub fn config(&self) -> &HolochainConfig {
        &self.config
    }

    /// Shutdown the client
    pub async fn shutdown(&mut self) {
        self.connected = false;
        tracing::info!("HolochainCortex shutdown");
    }
}

impl Default for HolochainCortex {
    fn default() -> Self {
        Self::new(HolochainConfig::default())
    }
}

// ============================================================================
// ERRORS
// ============================================================================

/// Errors from the Holochain Cortex
#[derive(Debug, Clone)]
pub enum CortexError {
    /// Feature not yet implemented
    NotImplemented(String),

    /// Connection error
    ConnectionError(String),

    /// Agent not found in DHT
    AgentNotFound(AgentPubKey),

    /// No local agent set
    NoLocalAgent,

    /// Verification failed
    VerificationFailed(String),

    /// DHT query failed
    DhtError(String),
}

impl std::fmt::Display for CortexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
            Self::ConnectionError(msg) => write!(f, "Connection error: {}", msg),
            Self::AgentNotFound(key) => write!(f, "Agent not found: {}", key),
            Self::NoLocalAgent => write!(f, "No local agent set"),
            Self::VerificationFailed(msg) => write!(f, "Verification failed: {}", msg),
            Self::DhtError(msg) => write!(f, "DHT error: {}", msg),
        }
    }
}

impl std::error::Error for CortexError {}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_pub_key() {
        let key = AgentPubKey::new("uhCAkTestKey12345678901234567890123456");
        assert!(key.is_valid());

        let test_key = AgentPubKey::test_key(42);
        assert!(test_key.is_valid());

        let invalid = AgentPubKey::new("short");
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_agent_info() {
        let key = AgentPubKey::test_key(1);
        let mut info = AgentInfo::new(key);

        assert_eq!(info.reputation_score, 0.5);
        assert!(info.phi_history.is_empty());

        // Record some Φ values
        info.record_phi(0.7);
        info.record_phi(0.8);
        assert_eq!(info.phi_history.len(), 2);
        assert!((info.average_phi() - 0.75).abs() < 0.01);

        // Test reputation update
        info.vouched_by.push(AgentPubKey::test_key(2));
        info.update_reputation();
        assert!(info.reputation_score > 0.5);
    }

    #[test]
    fn test_cortex_creation() {
        let cortex = HolochainCortex::default();
        assert!(cortex.config.mock_mode);
        assert!(!cortex.is_connected());
    }

    #[tokio::test]
    async fn test_cortex_mock_connect() {
        let mut cortex = HolochainCortex::default();
        cortex.connect().await.unwrap();
        assert!(cortex.is_connected());
    }

    #[test]
    fn test_verify_challenge() {
        let mut cortex = HolochainCortex::default();
        cortex.connected = true;

        let challenge = SignedChallenge {
            nonce: vec![1, 2, 3, 4],
            signature: vec![0; 64], // 64-byte Ed25519 signature
            agent_key: AgentPubKey::test_key(1),
            signed_at: 0,
        };

        let result = cortex.verify_challenge(&challenge);
        assert!(result.verified);
        assert!(result.trust_score >= 0.0);
    }

    #[test]
    fn test_verify_invalid_key() {
        let mut cortex = HolochainCortex::default();
        cortex.connected = true;

        let challenge = SignedChallenge {
            nonce: vec![1, 2, 3, 4],
            signature: vec![0; 64],
            agent_key: AgentPubKey::new("bad"), // Invalid key
            signed_at: 0,
        };

        let result = cortex.verify_challenge(&challenge);
        assert!(!result.verified);
    }

    #[test]
    fn test_vouch_for() {
        let mut cortex = HolochainCortex::default();

        let local_key = AgentPubKey::test_key(1);
        let peer_key = AgentPubKey::test_key(2);

        cortex.set_local_agent(local_key.clone());
        cortex.agent_cache.put(peer_key.clone(), AgentInfo::new(peer_key.clone()));

        let initial_rep = cortex.agent_cache.get(&peer_key).unwrap().reputation_score;
        cortex.vouch_for(&peer_key).unwrap();

        let new_rep = cortex.agent_cache.get(&peer_key).unwrap().reputation_score;
        assert!(new_rep > initial_rep);
    }

    #[test]
    fn test_record_violation() {
        let mut cortex = HolochainCortex::default();

        let key = AgentPubKey::test_key(1);
        cortex.agent_cache.put(key.clone(), AgentInfo::new(key.clone()));

        let initial_rep = cortex.agent_cache.get(&key).unwrap().reputation_score;
        cortex.record_violation(&key, "test violation");

        let new_rep = cortex.agent_cache.get(&key).unwrap().reputation_score;
        assert!(new_rep < initial_rep);
    }

    #[test]
    fn test_lru_cache_eviction() {
        // Create cortex with small cache capacity for testing
        let cortex_config = HolochainConfig::default();
        let mut cortex = HolochainCortex::with_cache_capacity(cortex_config, 3);

        // Add 3 agents (fills cache)
        for i in 0..3 {
            let key = AgentPubKey::test_key(i);
            cortex.agent_cache.put(key, AgentInfo::new(AgentPubKey::test_key(i)));
        }
        assert_eq!(cortex.agent_cache.len(), 3);

        // Add a 4th agent - should evict the LRU entry
        let key4 = AgentPubKey::test_key(100);
        cortex.put_agent_info(key4.clone(), AgentInfo::new(key4));

        assert_eq!(cortex.agent_cache.len(), 3);
        assert_eq!(cortex.stats.cache_evictions, 1);

        // First agent (key 0) should have been evicted
        assert!(cortex.agent_cache.peek(&AgentPubKey::test_key(0)).is_none());
    }

    #[test]
    fn test_cache_stats() {
        let mut cortex = HolochainCortex::default();

        // Generate some cache activity
        let key1 = AgentPubKey::test_key(1);
        let key2 = AgentPubKey::test_key(2);

        // Miss (creates new entry)
        cortex.get_or_create_agent_info(&key1);

        // Hit
        cortex.get_or_create_agent_info(&key1);

        // Miss (creates new entry)
        cortex.get_or_create_agent_info(&key2);

        let stats = cortex.cache_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.current_size, 2);
        assert!((stats.hit_rate - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_trust_verification_result() {
        let success = TrustVerificationResult::success(
            AgentPubKey::test_key(1),
            0.8,
            AgentInfo::new(AgentPubKey::test_key(1)),
        );
        assert!(success.verified);

        let failure = TrustVerificationResult::failure("test error");
        assert!(!failure.verified);
    }
}
