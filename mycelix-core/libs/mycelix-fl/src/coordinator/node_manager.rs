// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Node registration, authentication, and rate limiting.
//!
//! The [`NodeManager`] tracks registered nodes, blacklisted nodes, and
//! per-node submission rate limits (both per-minute and per-round).

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::error::FlError;

use super::config::CoordinatorConfig;

/// Credentials presented by a node during registration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeCredential {
    /// Unique node identifier.
    pub node_id: String,

    /// Ed25519 public key (32 bytes). `None` if authentication is not required.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_key: Option<Vec<u8>>,

    /// DID identifier. `None` if not using decentralized identity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub did: Option<String>,

    /// Unix timestamp of registration (seconds since epoch).
    pub registered_at: u64,
}

/// Manages node lifecycle: registration, blacklisting, and rate limiting.
pub struct NodeManager {
    /// Registered nodes by node_id.
    registered: HashMap<String, NodeCredential>,

    /// Blacklisted node IDs with reason.
    blacklisted: HashMap<String, String>,

    /// Per-node submission timestamps for per-minute rate limiting.
    submission_times: HashMap<String, Vec<Instant>>,

    /// Per-node submission count for the current round.
    round_submissions: HashMap<String, usize>,

    /// Configuration reference.
    config: CoordinatorConfig,
}

impl NodeManager {
    /// Create a new NodeManager with the given configuration.
    pub fn new(config: CoordinatorConfig) -> Self {
        Self {
            registered: HashMap::new(),
            blacklisted: HashMap::new(),
            submission_times: HashMap::new(),
            round_submissions: HashMap::new(),
            config,
        }
    }

    /// Register a node with the given credentials.
    ///
    /// # Errors
    ///
    /// Returns [`FlError::MaxNodesReached`] if the node limit is reached.
    /// Returns [`FlError::NodeBlacklisted`] if the node is blacklisted.
    pub fn register_node(&mut self, credential: NodeCredential) -> Result<(), FlError> {
        if self.blacklisted.contains_key(&credential.node_id) {
            return Err(FlError::NodeBlacklisted(credential.node_id.clone()));
        }

        if self.registered.len() >= self.config.max_nodes
            && !self.registered.contains_key(&credential.node_id)
        {
            return Err(FlError::MaxNodesReached(self.config.max_nodes));
        }

        self.registered
            .insert(credential.node_id.clone(), credential);
        Ok(())
    }

    /// Check if a node is registered.
    pub fn is_registered(&self, node_id: &str) -> bool {
        self.registered.contains_key(node_id)
    }

    /// Check if a node is blacklisted.
    pub fn is_blacklisted(&self, node_id: &str) -> bool {
        self.blacklisted.contains_key(node_id)
    }

    /// Blacklist a node with a reason. Removes the node from the registered set.
    pub fn blacklist(&mut self, node_id: &str, reason: String) {
        self.registered.remove(node_id);
        self.blacklisted.insert(node_id.to_string(), reason);
    }

    /// Check and enforce per-minute rate limiting for a node.
    ///
    /// This prunes timestamps older than 60 seconds, then checks if the
    /// node has exceeded `rate_limit_per_minute`.
    ///
    /// # Errors
    ///
    /// Returns [`FlError::RateLimitExceeded`] if the rate limit is exceeded.
    pub fn check_rate_limit(&mut self, node_id: &str) -> Result<(), FlError> {
        let now = Instant::now();
        let one_minute_ago = now - std::time::Duration::from_secs(60);

        let times = self
            .submission_times
            .entry(node_id.to_string())
            .or_default();

        // Prune old timestamps
        times.retain(|&t| t > one_minute_ago);

        if times.len() >= self.config.rate_limit_per_minute {
            return Err(FlError::RateLimitExceeded(node_id.to_string()));
        }

        // Also check per-round limit
        let round_count = self.round_submissions.get(node_id).copied().unwrap_or(0);
        if round_count >= self.config.rate_limit_per_round {
            return Err(FlError::RateLimitExceeded(node_id.to_string()));
        }

        Ok(())
    }

    /// Record a successful submission from a node (updates rate limit counters).
    pub fn record_submission(&mut self, node_id: &str) {
        let now = Instant::now();
        self.submission_times
            .entry(node_id.to_string())
            .or_default()
            .push(now);

        *self
            .round_submissions
            .entry(node_id.to_string())
            .or_insert(0) += 1;
    }

    /// Reset per-round counters. Called at the start of each new round.
    pub fn new_round(&mut self) {
        self.round_submissions.clear();
    }

    /// Number of registered (non-blacklisted) nodes.
    pub fn node_count(&self) -> usize {
        self.registered.len()
    }

    /// List of active node IDs (registered and not blacklisted).
    pub fn active_nodes(&self) -> Vec<String> {
        self.registered
            .keys()
            .filter(|id| !self.blacklisted.contains_key(id.as_str()))
            .cloned()
            .collect()
    }

    /// Get a reference to the blacklisted nodes.
    pub fn blacklisted_nodes(&self) -> &HashMap<String, String> {
        &self.blacklisted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> CoordinatorConfig {
        CoordinatorConfig {
            max_nodes: 10,
            rate_limit_per_minute: 10,
            rate_limit_per_round: 1,
            ..CoordinatorConfig::default()
        }
    }

    fn make_credential(node_id: &str) -> NodeCredential {
        NodeCredential {
            node_id: node_id.to_string(),
            public_key: None,
            did: None,
            registered_at: 0,
        }
    }

    #[test]
    fn test_register_node_and_verify() {
        let mut mgr = NodeManager::new(default_config());
        let cred = make_credential("node-1");
        mgr.register_node(cred).unwrap();
        assert!(mgr.is_registered("node-1"));
        assert!(!mgr.is_registered("node-2"));
    }

    #[test]
    fn test_blacklist_node() {
        let mut mgr = NodeManager::new(default_config());
        mgr.register_node(make_credential("node-1")).unwrap();
        assert!(mgr.is_registered("node-1"));
        assert!(!mgr.is_blacklisted("node-1"));

        mgr.blacklist("node-1", "malicious".into());
        assert!(mgr.is_blacklisted("node-1"));
        // Blacklisting also removes from registered
        assert!(!mgr.is_registered("node-1"));
    }

    #[test]
    fn test_blacklisted_node_cannot_register() {
        let mut mgr = NodeManager::new(default_config());
        mgr.blacklist("bad-node", "banned".into());
        let result = mgr.register_node(make_credential("bad-node"));
        assert!(matches!(result, Err(FlError::NodeBlacklisted(_))));
    }

    #[test]
    fn test_max_nodes_limit() {
        let config = CoordinatorConfig {
            max_nodes: 2,
            ..default_config()
        };
        let mut mgr = NodeManager::new(config);
        mgr.register_node(make_credential("a")).unwrap();
        mgr.register_node(make_credential("b")).unwrap();
        let result = mgr.register_node(make_credential("c"));
        assert!(matches!(result, Err(FlError::MaxNodesReached(2))));
    }

    #[test]
    fn test_per_minute_rate_limit() {
        let config = CoordinatorConfig {
            rate_limit_per_minute: 3,
            rate_limit_per_round: 100, // high so per-round doesn't interfere
            ..default_config()
        };
        let mut mgr = NodeManager::new(config);
        mgr.register_node(make_credential("node-1")).unwrap();

        // Submit 3 times (within limit)
        for _ in 0..3 {
            mgr.check_rate_limit("node-1").unwrap();
            mgr.record_submission("node-1");
        }

        // 4th should be rejected
        let result = mgr.check_rate_limit("node-1");
        assert!(matches!(result, Err(FlError::RateLimitExceeded(_))));
    }

    #[test]
    fn test_per_round_rate_limit() {
        let config = CoordinatorConfig {
            rate_limit_per_minute: 100,
            rate_limit_per_round: 1,
            ..default_config()
        };
        let mut mgr = NodeManager::new(config);
        mgr.register_node(make_credential("node-1")).unwrap();

        mgr.check_rate_limit("node-1").unwrap();
        mgr.record_submission("node-1");

        // Second submission in same round should fail
        let result = mgr.check_rate_limit("node-1");
        assert!(matches!(result, Err(FlError::RateLimitExceeded(_))));
    }

    #[test]
    fn test_new_round_resets_per_round_counters() {
        let config = CoordinatorConfig {
            rate_limit_per_round: 1,
            rate_limit_per_minute: 100,
            ..default_config()
        };
        let mut mgr = NodeManager::new(config);
        mgr.register_node(make_credential("node-1")).unwrap();

        mgr.check_rate_limit("node-1").unwrap();
        mgr.record_submission("node-1");

        // Reset for new round
        mgr.new_round();

        // Should be allowed again
        mgr.check_rate_limit("node-1").unwrap();
    }

    #[test]
    fn test_unknown_node_not_registered() {
        let mgr = NodeManager::new(default_config());
        assert!(!mgr.is_registered("unknown"));
        assert!(!mgr.is_blacklisted("unknown"));
    }

    #[test]
    fn test_active_nodes_excludes_blacklisted() {
        let mut mgr = NodeManager::new(default_config());
        mgr.register_node(make_credential("a")).unwrap();
        mgr.register_node(make_credential("b")).unwrap();
        mgr.register_node(make_credential("c")).unwrap();
        mgr.blacklist("b", "bad".into());

        let active = mgr.active_nodes();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&"a".to_string()));
        assert!(active.contains(&"c".to_string()));
        assert!(!active.contains(&"b".to_string()));
    }

    #[test]
    fn test_node_count() {
        let mut mgr = NodeManager::new(default_config());
        assert_eq!(mgr.node_count(), 0);
        mgr.register_node(make_credential("a")).unwrap();
        assert_eq!(mgr.node_count(), 1);
        mgr.register_node(make_credential("b")).unwrap();
        assert_eq!(mgr.node_count(), 2);
    }
}
