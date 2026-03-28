// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Top-level FL coordinator orchestrating node management, rounds, and aggregation.
//!
//! [`FLCoordinator`] ties together [`NodeManager`], [`Round`], and the
//! aggregation defenses into a single interface for running federated
//! learning training.

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::time::{timeout, Duration};

use crate::defenses::Defense;
use crate::error::FlError;
use crate::pogq::config::PoGQv41Config;
use crate::pogq::v41_enhanced::PoGQv41Enhanced;
use crate::types::{AggregationResult, DefenseConfig, Gradient};

use super::config::CoordinatorConfig;
use super::node_manager::{NodeCredential, NodeManager};
use super::round::{Round, RoundStatus};

/// Summary of a completed (or failed) round, suitable for serialization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoundSummary {
    /// Round number.
    pub round_number: u64,
    /// Final status as a string.
    pub status: String,
    /// Number of nodes that submitted gradients.
    pub node_count: usize,
    /// Number of nodes excluded by defense/PoGQ.
    pub excluded_count: usize,
    /// Round duration in milliseconds.
    pub duration_ms: u64,
}

/// The top-level FL coordinator.
///
/// Manages the full lifecycle: node registration, round execution,
/// gradient validation, Byzantine detection (via PoGQ), and aggregation.
pub struct FLCoordinator {
    config: CoordinatorConfig,
    defense_config: DefenseConfig,
    node_manager: NodeManager,
    current_round: Option<Round>,
    round_counter: u64,
    round_history: Vec<RoundSummary>,
    defense: Box<dyn Defense + Send>,
    pogq: Option<PoGQv41Enhanced>,
}

impl FLCoordinator {
    /// Create a new coordinator.
    ///
    /// # Arguments
    ///
    /// * `config` — Coordinator configuration
    /// * `defense` — Aggregation defense algorithm (e.g., FedAvg, TrimmedMean, RFA)
    /// * `pogq_config` — Optional PoGQ configuration. When `Some`, PoGQ is used
    ///   to detect and exclude Byzantine nodes before aggregation.
    pub fn new(
        config: CoordinatorConfig,
        defense: Box<dyn Defense + Send>,
        pogq_config: Option<PoGQv41Config>,
    ) -> Self {
        let node_manager = NodeManager::new(config.clone());
        let pogq = pogq_config.map(PoGQv41Enhanced::new);

        Self {
            config,
            defense_config: DefenseConfig::default(),
            node_manager,
            current_round: None,
            round_counter: 0,
            round_history: Vec::new(),
            defense,
            pogq,
        }
    }

    /// Set a custom defense configuration for the aggregation algorithm.
    pub fn set_defense_config(&mut self, config: DefenseConfig) {
        self.defense_config = config;
    }

    /// Register a node with the coordinator.
    pub fn register_node(&mut self, credential: NodeCredential) -> Result<(), FlError> {
        self.node_manager.register_node(credential)
    }

    /// Start a new FL round.
    ///
    /// # Errors
    ///
    /// Returns [`FlError::InsufficientGradients`] if fewer than `min_nodes` are registered.
    /// Returns [`FlError::InvalidRoundState`] if a round is already in progress.
    pub fn start_round(&mut self) -> Result<u64, FlError> {
        // Check no round is currently in progress
        if let Some(ref round) = self.current_round {
            if round.status == RoundStatus::Collecting
                || round.status == RoundStatus::Aggregating
            {
                return Err(FlError::InvalidRoundState {
                    expected: "no active round".into(),
                    got: round.status.to_string(),
                });
            }
        }

        if self.node_manager.node_count() < self.config.min_nodes {
            return Err(FlError::InsufficientGradients {
                got: self.node_manager.node_count(),
                need: self.config.min_nodes,
            });
        }

        self.round_counter += 1;
        self.node_manager.new_round();

        let mut round = Round::new(self.round_counter);
        round.start();
        self.current_round = Some(round);

        Ok(self.round_counter)
    }

    /// Submit a gradient for the current round.
    ///
    /// Validates: node is registered, not blacklisted, within rate limits,
    /// gradient is non-empty with finite values, and no duplicate submission.
    pub fn submit_gradient(&mut self, gradient: Gradient) -> Result<(), FlError> {
        let node_id = gradient.node_id.clone();

        // 1. Check node is registered
        if !self.node_manager.is_registered(&node_id) {
            return Err(FlError::NodeNotRegistered(node_id));
        }

        // 2. Check node is not blacklisted
        if self.node_manager.is_blacklisted(&node_id) {
            return Err(FlError::NodeBlacklisted(node_id));
        }

        // 3. Check rate limit
        self.node_manager.check_rate_limit(&node_id)?;

        // 4. Validate gradient values
        if gradient.values.is_empty() {
            return Err(FlError::EmptyGradient);
        }
        if gradient.values.iter().any(|v| !v.is_finite()) {
            return Err(FlError::NonFiniteGradient(node_id));
        }

        // 5. Submit to current round
        let round = self
            .current_round
            .as_mut()
            .ok_or(FlError::NoActiveRound)?;

        round.submit_gradient(gradient)?;

        // 6. Record submission for rate limiting
        self.node_manager.record_submission(&node_id);

        Ok(())
    }

    /// Complete the current round: run PoGQ detection (if enabled), then aggregate.
    ///
    /// # Errors
    ///
    /// Returns [`FlError::NoActiveRound`] if no round is in progress.
    /// Returns [`FlError::InsufficientGradients`] if fewer than `min_nodes` submitted.
    pub fn complete_round(&mut self) -> Result<AggregationResult, FlError> {
        let round = self
            .current_round
            .as_mut()
            .ok_or(FlError::NoActiveRound)?;

        if !round.has_enough_nodes(self.config.min_nodes) {
            let got = round.gradients.len();
            round.fail(format!(
                "insufficient gradients: {} < {}",
                got, self.config.min_nodes
            ));

            let summary = RoundSummary {
                round_number: round.round_number,
                status: "Failed".into(),
                node_count: got,
                excluded_count: 0,
                duration_ms: round.duration_ms().unwrap_or(0),
            };
            self.round_history.push(summary);

            return Err(FlError::InsufficientGradients {
                got,
                need: self.config.min_nodes,
            });
        }

        // Take gradients from the round
        let gradients = round.gradients.clone();

        // Run PoGQ detection if enabled
        let result = if let Some(ref mut pogq) = self.pogq {
            pogq.aggregate(&gradients)?
        } else {
            // Use the defense directly
            self.defense.aggregate(&gradients, &self.defense_config)?
        };

        // Record summary
        let summary = RoundSummary {
            round_number: round.round_number,
            status: "Complete".into(),
            node_count: gradients.len(),
            excluded_count: result.excluded_nodes.len(),
            duration_ms: round.duration_ms().unwrap_or(0),
        };

        round.complete(result.clone());
        self.round_history.push(summary);

        Ok(result)
    }

    /// Run a complete round with a timeout, receiving gradients via an async channel.
    ///
    /// Starts a round, collects gradients until `min_nodes` are met or the
    /// timeout expires, then completes the round.
    pub async fn run_round_async(
        &mut self,
        mut rx: mpsc::Receiver<Gradient>,
    ) -> Result<AggregationResult, FlError> {
        self.start_round()?;
        let deadline = Duration::from_secs(self.config.round_timeout_secs);

        let min_nodes = self.config.min_nodes;
        match timeout(deadline, async {
            while let Some(gradient) = rx.recv().await {
                let _ = self.submit_gradient(gradient);
                if self
                    .current_round
                    .as_ref()
                    .map_or(false, |r| r.has_enough_nodes(min_nodes))
                {
                    // Continue collecting briefly or break — break for now
                    break;
                }
            }
        })
        .await
        {
            Ok(()) => {}
            Err(_) => {
                // Timeout — proceed with whatever we collected
            }
        }

        self.complete_round()
    }

    /// Get the history of completed rounds.
    pub fn round_history(&self) -> &[RoundSummary] {
        &self.round_history
    }

    /// Get the current round number, if a round is active.
    pub fn current_round_number(&self) -> Option<u64> {
        self.current_round.as_ref().map(|r| r.round_number)
    }

    /// Number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.node_manager.node_count()
    }

    /// Reference to the coordinator config.
    pub fn config(&self) -> &CoordinatorConfig {
        &self.config
    }

    /// Reference to the node manager.
    pub fn node_manager(&self) -> &NodeManager {
        &self.node_manager
    }

    /// Mutable reference to the node manager.
    pub fn node_manager_mut(&mut self) -> &mut NodeManager {
        &mut self.node_manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defenses::FedAvg;

    fn default_coordinator() -> FLCoordinator {
        FLCoordinator::new(
            CoordinatorConfig {
                min_nodes: 3,
                max_nodes: 100,
                rate_limit_per_minute: 100,
                rate_limit_per_round: 1,
                ..CoordinatorConfig::default()
            },
            Box::new(FedAvg),
            None,
        )
    }

    fn coordinator_with_pogq() -> FLCoordinator {
        FLCoordinator::new(
            CoordinatorConfig {
                min_nodes: 3,
                max_nodes: 100,
                rate_limit_per_minute: 100,
                rate_limit_per_round: 1,
                ..CoordinatorConfig::default()
            },
            Box::new(FedAvg),
            Some(PoGQv41Config {
                warm_up_rounds: 0,
                k_quarantine: 1,
                beta: 0.0,
                ..PoGQv41Config::default()
            }),
        )
    }

    fn register_nodes(coord: &mut FLCoordinator, ids: &[&str]) {
        for id in ids {
            coord
                .register_node(NodeCredential {
                    node_id: id.to_string(),
                    public_key: None,
                    did: None,
                    registered_at: 0,
                })
                .unwrap();
        }
    }

    fn honest_gradient(node_id: &str, round: u64, dim: usize) -> Gradient {
        let values: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        Gradient::new(node_id, values, round)
    }

    fn byzantine_gradient(node_id: &str, round: u64, dim: usize) -> Gradient {
        let values: Vec<f32> = (0..dim).map(|i| -((i as f32 + 1.0) * 0.1)).collect();
        Gradient::new(node_id, values, round)
    }

    // -----------------------------------------------------------------------
    // Full round lifecycle
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_round_lifecycle() {
        let mut coord = default_coordinator();
        let nodes = ["alice", "bob", "carol", "dave", "eve"];
        register_nodes(&mut coord, &nodes);

        assert_eq!(coord.node_count(), 5);

        let round_num = coord.start_round().unwrap();
        assert_eq!(round_num, 1);

        for &node in &nodes {
            coord
                .submit_gradient(honest_gradient(node, 1, 50))
                .unwrap();
        }

        let result = coord.complete_round().unwrap();
        assert_eq!(result.included_nodes.len(), 5);
        assert!(result.excluded_nodes.is_empty());
        assert_eq!(result.gradient.len(), 50);

        // All values should be positive (honest gradients)
        for &v in &result.gradient {
            assert!(v > 0.0);
        }

        // Round history should have one entry
        assert_eq!(coord.round_history().len(), 1);
        assert_eq!(coord.round_history()[0].status, "Complete");
        assert_eq!(coord.round_history()[0].node_count, 5);
    }

    // -----------------------------------------------------------------------
    // PoGQ detection: Byzantine excluded
    // -----------------------------------------------------------------------

    #[test]
    fn test_pogq_detects_byzantine() {
        let mut coord = coordinator_with_pogq();
        let honest = ["h1", "h2", "h3", "h4", "h5"];
        let byzantine = "byz";
        let mut all_nodes: Vec<&str> = honest.to_vec();
        all_nodes.push(byzantine);
        register_nodes(&mut coord, &all_nodes);

        let dim = 50;

        // Run several rounds to build PoGQ history (Byzantine needs k_quarantine rounds)
        for round in 1..=5 {
            coord.start_round().unwrap();

            for &h in &honest {
                coord
                    .submit_gradient(honest_gradient(h, round, dim))
                    .unwrap();
            }
            coord
                .submit_gradient(byzantine_gradient(byzantine, round, dim))
                .unwrap();

            let result = coord.complete_round().unwrap();

            // After enough rounds, Byzantine should be excluded
            if round >= 3 {
                assert!(
                    result.excluded_nodes.contains(&byzantine.to_string()),
                    "round {}: Byzantine should be excluded, but excluded={:?}",
                    round,
                    result.excluded_nodes
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Unregistered node rejected
    // -----------------------------------------------------------------------

    #[test]
    fn test_unregistered_node_rejected() {
        let mut coord = default_coordinator();
        register_nodes(&mut coord, &["a", "b", "c"]);
        coord.start_round().unwrap();

        let result = coord.submit_gradient(honest_gradient("unknown", 1, 10));
        assert!(matches!(result, Err(FlError::NodeNotRegistered(_))));
    }

    // -----------------------------------------------------------------------
    // Insufficient nodes fails round
    // -----------------------------------------------------------------------

    #[test]
    fn test_insufficient_nodes_for_start() {
        let mut coord = default_coordinator();
        register_nodes(&mut coord, &["a"]); // Only 1, need 3
        let result = coord.start_round();
        assert!(matches!(
            result,
            Err(FlError::InsufficientGradients { .. })
        ));
    }

    #[test]
    fn test_insufficient_gradients_for_complete() {
        let mut coord = default_coordinator();
        register_nodes(&mut coord, &["a", "b", "c"]);
        coord.start_round().unwrap();

        // Only submit 1 gradient (need 3)
        coord
            .submit_gradient(honest_gradient("a", 1, 10))
            .unwrap();

        let result = coord.complete_round();
        assert!(matches!(
            result,
            Err(FlError::InsufficientGradients { .. })
        ));

        // Should be recorded as failed in history
        assert_eq!(coord.round_history().len(), 1);
        assert_eq!(coord.round_history()[0].status, "Failed");
    }

    // -----------------------------------------------------------------------
    // Non-finite gradient rejected
    // -----------------------------------------------------------------------

    #[test]
    fn test_nonfinite_gradient_rejected() {
        let mut coord = default_coordinator();
        register_nodes(&mut coord, &["a", "b", "c"]);
        coord.start_round().unwrap();

        let bad = Gradient::new("a", vec![1.0, f32::NAN, 3.0], 1);
        let result = coord.submit_gradient(bad);
        assert!(matches!(result, Err(FlError::NonFiniteGradient(_))));

        let inf = Gradient::new("b", vec![1.0, f32::INFINITY, 3.0], 1);
        let result = coord.submit_gradient(inf);
        assert!(matches!(result, Err(FlError::NonFiniteGradient(_))));
    }

    // -----------------------------------------------------------------------
    // Empty gradient rejected
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_gradient_rejected() {
        let mut coord = default_coordinator();
        register_nodes(&mut coord, &["a", "b", "c"]);
        coord.start_round().unwrap();

        let empty = Gradient::new("a", vec![], 1);
        let result = coord.submit_gradient(empty);
        assert!(matches!(result, Err(FlError::EmptyGradient)));
    }

    // -----------------------------------------------------------------------
    // Round history tracks multiple rounds
    // -----------------------------------------------------------------------

    #[test]
    fn test_round_history_tracks_summaries() {
        let mut coord = default_coordinator();
        register_nodes(&mut coord, &["a", "b", "c"]);

        for round in 1..=3 {
            coord.start_round().unwrap();
            for node in &["a", "b", "c"] {
                coord
                    .submit_gradient(honest_gradient(node, round, 10))
                    .unwrap();
            }
            coord.complete_round().unwrap();
        }

        assert_eq!(coord.round_history().len(), 3);
        for (i, summary) in coord.round_history().iter().enumerate() {
            assert_eq!(summary.round_number, (i + 1) as u64);
            assert_eq!(summary.status, "Complete");
            assert_eq!(summary.node_count, 3);
        }
    }

    // -----------------------------------------------------------------------
    // Async round with channel
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_run_round_async() {
        let mut coord = FLCoordinator::new(
            CoordinatorConfig {
                min_nodes: 3,
                max_nodes: 100,
                round_timeout_secs: 5,
                rate_limit_per_minute: 100,
                rate_limit_per_round: 1,
                ..CoordinatorConfig::default()
            },
            Box::new(FedAvg),
            None,
        );
        register_nodes(&mut coord, &["a", "b", "c"]);

        let (tx, rx) = mpsc::channel(10);

        // Spawn a task to send gradients
        tokio::spawn(async move {
            for node in &["a", "b", "c"] {
                tx.send(honest_gradient(node, 1, 10)).await.unwrap();
            }
        });

        let result = coord.run_round_async(rx).await.unwrap();
        assert_eq!(result.included_nodes.len(), 3);
    }

    // -----------------------------------------------------------------------
    // Duplicate submission in same round rejected
    // -----------------------------------------------------------------------

    #[test]
    fn test_duplicate_submission_rejected() {
        let mut coord = default_coordinator();
        register_nodes(&mut coord, &["a", "b", "c"]);
        coord.start_round().unwrap();

        coord
            .submit_gradient(honest_gradient("a", 1, 10))
            .unwrap();

        // The per-round rate limit (1) will reject the second submission
        let result = coord.submit_gradient(honest_gradient("a", 1, 10));
        assert!(result.is_err());
    }
}
