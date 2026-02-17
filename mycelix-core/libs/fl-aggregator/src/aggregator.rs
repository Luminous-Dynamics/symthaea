//! Core aggregator implementation with memory management and async support.

use crate::byzantine::{ByzantineAggregator, Defense, DefenseConfig};
use crate::error::{AggregatorError, Result};
use crate::metrics::AggregatorMetrics;
use crate::{Gradient, NodeId, Round};

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Configuration for the aggregator.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregatorConfig {
    /// Expected number of nodes per round.
    pub expected_nodes: usize,

    /// Byzantine defense algorithm.
    pub defense: Defense,

    /// Maximum memory usage in bytes (0 = unlimited).
    pub max_memory_bytes: usize,

    /// Maximum gradient dimension allowed.
    pub max_gradient_dim: usize,

    /// Timeout for round completion.
    pub round_timeout: Duration,

    /// Whether to allow late submissions (after round complete).
    pub allow_late_submissions: bool,

    /// Minimum number of nodes to proceed with aggregation.
    pub min_nodes_to_aggregate: Option<usize>,
}

impl Default for AggregatorConfig {
    fn default() -> Self {
        Self {
            expected_nodes: 10,
            defense: Defense::Krum { f: 1 },
            max_memory_bytes: 2_000_000_000, // 2GB
            max_gradient_dim: 100_000_000,   // 100M parameters
            round_timeout: Duration::from_secs(300),
            allow_late_submissions: false,
            min_nodes_to_aggregate: None,
        }
    }
}

impl AggregatorConfig {
    /// Set expected number of nodes.
    pub fn with_expected_nodes(mut self, n: usize) -> Self {
        self.expected_nodes = n;
        self
    }

    /// Set defense algorithm.
    pub fn with_defense(mut self, defense: Defense) -> Self {
        self.defense = defense;
        self
    }

    /// Set memory limit.
    pub fn with_max_memory(mut self, bytes: usize) -> Self {
        self.max_memory_bytes = bytes;
        self
    }

    /// Set round timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.round_timeout = timeout;
        self
    }
}

/// Submission record for a gradient.
#[derive(Clone, Debug)]
pub struct GradientSubmission {
    /// Node that submitted.
    pub node_id: NodeId,

    /// The gradient data.
    pub gradient: Gradient,

    /// When it was submitted.
    pub timestamp: Instant,

    /// Optional signature for verification.
    pub signature: Option<Vec<u8>>,
}

/// Synchronous aggregator for single-threaded use.
pub struct Aggregator {
    config: AggregatorConfig,
    byzantine: ByzantineAggregator,
    round: Round,
    submissions: HashMap<NodeId, GradientSubmission>,
    gradient_dim: Option<usize>,
    round_start: Instant,
    metrics: AggregatorMetrics,
}

impl Aggregator {
    /// Create a new aggregator.
    pub fn new(config: AggregatorConfig) -> Self {
        let defense_config = DefenseConfig::with_defense(config.defense.clone());
        let byzantine = ByzantineAggregator::new(defense_config);

        Self {
            config,
            byzantine,
            round: 0,
            submissions: HashMap::new(),
            gradient_dim: None,
            round_start: Instant::now(),
            metrics: AggregatorMetrics::new(),
        }
    }

    /// Submit a gradient from a node.
    pub fn submit(&mut self, node_id: impl Into<NodeId>, gradient: Gradient) -> Result<()> {
        let node_id = node_id.into();

        // Check for duplicate submission
        if self.submissions.contains_key(&node_id) {
            return Err(AggregatorError::DuplicateSubmission {
                node: node_id,
                round: self.round,
            });
        }

        // Validate gradient dimension
        let dim = gradient.len();
        if dim > self.config.max_gradient_dim {
            return Err(AggregatorError::InvalidConfig(format!(
                "Gradient dimension {} exceeds max {}",
                dim, self.config.max_gradient_dim
            )));
        }

        // Validate gradient values are finite (no NaN or Infinity)
        for (i, &val) in gradient.iter().enumerate() {
            if !val.is_finite() {
                return Err(AggregatorError::InvalidConfig(format!(
                    "Gradient from {} contains non-finite value at index {}: {}",
                    node_id, i, val
                )));
            }
        }

        // Check dimension consistency
        match self.gradient_dim {
            Some(expected) if expected != dim => {
                return Err(AggregatorError::DimensionMismatch {
                    expected,
                    got: dim,
                });
            }
            None => self.gradient_dim = Some(dim),
            _ => {}
        }

        // Check memory limit
        let gradient_bytes = dim * std::mem::size_of::<f32>();
        let current_memory = self.memory_usage();
        if self.config.max_memory_bytes > 0
            && current_memory + gradient_bytes > self.config.max_memory_bytes
        {
            return Err(AggregatorError::MemoryLimitExceeded {
                used: current_memory + gradient_bytes,
                limit: self.config.max_memory_bytes,
            });
        }

        // Store submission
        let submission = GradientSubmission {
            node_id: node_id.clone(),
            gradient,
            timestamp: Instant::now(),
            signature: None,
        };
        self.submissions.insert(node_id.clone(), submission);

        self.metrics.record_submission(&node_id, dim);
        tracing::debug!(
            "Gradient submitted: node={}, round={}, dim={}, total={}/{}",
            node_id,
            self.round,
            dim,
            self.submissions.len(),
            self.config.expected_nodes
        );

        Ok(())
    }

    /// Check if the round is complete.
    pub fn is_round_complete(&self) -> bool {
        self.submissions.len() >= self.config.expected_nodes
    }

    /// Check if minimum nodes have submitted.
    pub fn can_aggregate(&self) -> bool {
        let min = self
            .config
            .min_nodes_to_aggregate
            .unwrap_or(self.config.expected_nodes);
        self.submissions.len() >= min
    }

    /// Get the number of submissions in current round.
    pub fn submission_count(&self) -> usize {
        self.submissions.len()
    }

    /// Finalize the round and return aggregated gradient.
    ///
    /// # Performance
    /// This method moves gradients out of submissions rather than cloning them,
    /// avoiding a full copy of potentially gigabytes of gradient data.
    pub fn finalize_round(&mut self) -> Result<Gradient> {
        if self.submissions.is_empty() {
            return Err(AggregatorError::NoGradients(self.round));
        }

        // Performance fix: Move gradients out instead of cloning to avoid
        // copying potentially gigabytes of data. Since we clear submissions
        // after aggregation anyway, this is safe and much faster.
        let submissions = std::mem::take(&mut self.submissions);
        let gradients: Vec<Gradient> = submissions
            .into_values()
            .map(|s| s.gradient)
            .collect();

        // Aggregate using Byzantine defense
        let start = Instant::now();
        let result = self.byzantine.aggregate(&gradients)?;
        let aggregation_time = start.elapsed();

        self.metrics
            .record_aggregation(self.round, gradients.len(), aggregation_time);

        tracing::info!(
            "Round {} finalized: {} gradients aggregated in {:?}",
            self.round,
            gradients.len(),
            aggregation_time
        );

        // Clear for next round (submissions already moved out via take)
        self.gradient_dim = None;
        self.round += 1;
        self.round_start = Instant::now();

        Ok(result)
    }

    /// Get current round number.
    pub fn current_round(&self) -> Round {
        self.round
    }

    /// Get current memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        match self.gradient_dim {
            Some(dim) => self.submissions.len() * dim * std::mem::size_of::<f32>(),
            None => 0,
        }
    }

    /// Get elapsed time since round start.
    pub fn round_elapsed(&self) -> Duration {
        self.round_start.elapsed()
    }

    /// Get metrics.
    pub fn metrics(&self) -> &AggregatorMetrics {
        &self.metrics
    }

    /// Reset the aggregator state.
    pub fn reset(&mut self) {
        self.submissions.clear();
        self.gradient_dim = None;
        self.round = 0;
        self.round_start = Instant::now();
    }
}

/// Async aggregator for concurrent node handling.
#[derive(Clone)]
pub struct AsyncAggregator {
    inner: Arc<RwLock<Aggregator>>,
    registered_nodes: Arc<RwLock<HashSet<NodeId>>>,
}

impl AsyncAggregator {
    /// Create a new async aggregator.
    pub fn new(config: AggregatorConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Aggregator::new(config))),
            registered_nodes: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Register a node.
    pub async fn register_node(&self, node_id: impl Into<NodeId>) -> Result<()> {
        let node_id = node_id.into();
        let mut nodes = self.registered_nodes.write().await;
        nodes.insert(node_id.clone());
        tracing::info!("Node registered: {}", node_id);
        Ok(())
    }

    /// Unregister a node.
    pub async fn unregister_node(&self, node_id: &str) -> Result<()> {
        let mut nodes = self.registered_nodes.write().await;
        nodes.remove(node_id);
        tracing::info!("Node unregistered: {}", node_id);
        Ok(())
    }

    /// Check if a node is registered.
    pub async fn is_registered(&self, node_id: &str) -> bool {
        let nodes = self.registered_nodes.read().await;
        nodes.contains(node_id)
    }

    /// Submit a gradient from a node.
    pub async fn submit(&self, node_id: impl Into<NodeId>, gradient: Gradient) -> Result<()> {
        let node_id = node_id.into();

        // Verify node is registered
        if !self.is_registered(&node_id).await {
            return Err(AggregatorError::InvalidNode(node_id));
        }

        let mut aggregator = self.inner.write().await;
        aggregator.submit(node_id, gradient)
    }

    /// Check if round is complete.
    pub async fn is_round_complete(&self) -> bool {
        let aggregator = self.inner.read().await;
        aggregator.is_round_complete()
    }

    /// Get aggregated gradient if round is complete.
    pub async fn get_aggregated_gradient(&self) -> Result<Option<Gradient>> {
        let mut aggregator = self.inner.write().await;
        if aggregator.is_round_complete() {
            Ok(Some(aggregator.finalize_round()?))
        } else {
            Ok(None)
        }
    }

    /// Force finalize even if not all nodes submitted.
    pub async fn force_finalize(&self) -> Result<Gradient> {
        let mut aggregator = self.inner.write().await;
        aggregator.finalize_round()
    }

    /// Get current status.
    pub async fn status(&self) -> AggregatorStatus {
        let aggregator = self.inner.read().await;
        let nodes = self.registered_nodes.read().await;

        AggregatorStatus {
            round: aggregator.current_round(),
            registered_nodes: nodes.len(),
            submitted_nodes: aggregator.submission_count(),
            expected_nodes: aggregator.config.expected_nodes,
            memory_bytes: aggregator.memory_usage(),
            round_elapsed: aggregator.round_elapsed(),
            is_complete: aggregator.is_round_complete(),
            algorithm: aggregator.config.defense.to_string(),
        }
    }

    /// Get metrics.
    pub async fn metrics(&self) -> AggregatorMetrics {
        let aggregator = self.inner.read().await;
        aggregator.metrics().clone()
    }
}

/// Status information for monitoring.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregatorStatus {
    pub round: Round,
    pub registered_nodes: usize,
    pub submitted_nodes: usize,
    pub expected_nodes: usize,
    pub memory_bytes: usize,
    #[serde(with = "humantime_serde")]
    pub round_elapsed: Duration,
    pub is_complete: bool,
    /// Name of the aggregation/defense algorithm (e.g., "FedAvg", "Krum(f=1)")
    pub algorithm: String,
}

mod humantime_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = humantime::format_duration(*duration).to_string();
        s.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        humantime::parse_duration(&s).map_err(serde::de::Error::custom)
    }
}

/// Wrapper for proof-verified aggregation
#[cfg(feature = "proofs")]
pub mod verified {
    use super::*;
    use crate::proofs::{ProofConfig, ProofError, ProofResult};
    use crate::proofs::integration::{
        VerifiedGradientSubmission, ProofVerifyingSubmitter,
    };
    use crate::proofs::integration::aggregator::VerifiedAggregationConfig;

    /// Aggregator that verifies gradient proofs before accepting submissions
    pub struct VerifiedAggregator {
        inner: Aggregator,
        verified_config: VerifiedAggregationConfig,
    }

    impl VerifiedAggregator {
        /// Create a new verified aggregator
        pub fn new(config: AggregatorConfig, verified_config: VerifiedAggregationConfig) -> Self {
            Self {
                inner: Aggregator::new(config),
                verified_config,
            }
        }

        /// Submit a verified gradient - proofs are verified before acceptance
        pub fn submit_verified(
            &mut self,
            mut submission: VerifiedGradientSubmission,
        ) -> ProofResult<()> {
            // Check if proofs are required
            if self.verified_config.require_proofs && !submission.has_proofs() {
                return Err(ProofError::InvalidPublicInputs(
                    "Proofs are required but not provided".to_string()
                ));
            }

            // Verify proofs if present
            if submission.has_proofs() {
                let result = submission.verify()?;
                if !result.valid {
                    return Err(ProofError::VerificationFailed(
                        format!("Proof verification failed: {:?}", result.details)
                    ));
                }
            }

            // Submit to inner aggregator
            self.inner.submit(
                submission.node_id.clone(),
                submission.gradient,
            ).map_err(|e| ProofError::InvalidPublicInputs(e.to_string()))?;

            Ok(())
        }

        /// Submit a gradient with auto-generated proofs
        pub fn submit_with_proof(
            &mut self,
            node_id: impl Into<NodeId>,
            gradient: Gradient,
        ) -> ProofResult<()> {
            let node_id = node_id.into();
            let mut submission = VerifiedGradientSubmission::new(node_id, gradient);

            submission.generate_proofs(
                self.verified_config.max_gradient_norm,
                self.verified_config.proof_config.clone(),
            )?;

            self.submit_verified(submission)
        }

        /// Submit a gradient without proofs (only works if proofs are optional)
        pub fn submit_unverified(
            &mut self,
            node_id: impl Into<NodeId>,
            gradient: Gradient,
        ) -> ProofResult<()> {
            if self.verified_config.require_proofs {
                return Err(ProofError::InvalidPublicInputs(
                    "Proofs are required".to_string()
                ));
            }

            self.inner.submit(node_id, gradient)
                .map_err(|e| ProofError::InvalidPublicInputs(e.to_string()))
        }

        /// Check if round is complete
        pub fn is_round_complete(&self) -> bool {
            self.inner.is_round_complete()
        }

        /// Finalize the round
        pub fn finalize_round(&mut self) -> Result<Gradient> {
            self.inner.finalize_round()
        }

        /// Get current round
        pub fn current_round(&self) -> Round {
            self.inner.current_round()
        }

        /// Get submission count
        pub fn submission_count(&self) -> usize {
            self.inner.submission_count()
        }

        /// Get verified aggregation config
        pub fn verified_config(&self) -> &VerifiedAggregationConfig {
            &self.verified_config
        }

        /// Get inner aggregator metrics
        pub fn metrics(&self) -> &AggregatorMetrics {
            self.inner.metrics()
        }
    }

    impl ProofVerifyingSubmitter for VerifiedAggregator {
        fn submit_verified(
            &mut self,
            submission: VerifiedGradientSubmission,
        ) -> ProofResult<()> {
            self.submit_verified(submission)
        }

        fn requires_proofs(&self) -> bool {
            self.verified_config.require_proofs
        }

        fn proof_config(&self) -> ProofConfig {
            self.verified_config.proof_config.clone()
        }
    }

    /// Async verified aggregator for concurrent use
    pub struct VerifiedAsyncAggregator {
        inner: Arc<RwLock<VerifiedAggregator>>,
        registered_nodes: Arc<RwLock<HashSet<NodeId>>>,
    }

    impl VerifiedAsyncAggregator {
        /// Create a new async verified aggregator
        pub fn new(config: AggregatorConfig, verified_config: VerifiedAggregationConfig) -> Self {
            Self {
                inner: Arc::new(RwLock::new(VerifiedAggregator::new(config, verified_config))),
                registered_nodes: Arc::new(RwLock::new(HashSet::new())),
            }
        }

        /// Register a node
        pub async fn register_node(&self, node_id: impl Into<NodeId>) -> Result<()> {
            let node_id = node_id.into();
            let mut nodes = self.registered_nodes.write().await;
            nodes.insert(node_id.clone());
            tracing::info!("Node registered: {}", node_id);
            Ok(())
        }

        /// Check if a node is registered
        pub async fn is_registered(&self, node_id: &str) -> bool {
            let nodes = self.registered_nodes.read().await;
            nodes.contains(node_id)
        }

        /// Submit a verified gradient
        pub async fn submit_verified(
            &self,
            submission: VerifiedGradientSubmission,
        ) -> ProofResult<()> {
            // Verify node is registered
            if !self.is_registered(&submission.node_id).await {
                return Err(ProofError::InvalidPublicInputs(
                    format!("Node not registered: {}", submission.node_id)
                ));
            }

            let mut aggregator = self.inner.write().await;
            aggregator.submit_verified(submission)
        }

        /// Submit a gradient with auto-generated proofs
        pub async fn submit_with_proof(
            &self,
            node_id: impl Into<NodeId>,
            gradient: Gradient,
        ) -> ProofResult<()> {
            let node_id = node_id.into();

            // Verify node is registered
            if !self.is_registered(&node_id).await {
                return Err(ProofError::InvalidPublicInputs(
                    format!("Node not registered: {}", node_id)
                ));
            }

            let mut aggregator = self.inner.write().await;
            aggregator.submit_with_proof(node_id, gradient)
        }

        /// Check if round is complete
        pub async fn is_round_complete(&self) -> bool {
            let aggregator = self.inner.read().await;
            aggregator.is_round_complete()
        }

        /// Finalize round
        pub async fn finalize_round(&self) -> Result<Gradient> {
            let mut aggregator = self.inner.write().await;
            aggregator.finalize_round()
        }

        /// Get status
        pub async fn status(&self) -> VerifiedAggregatorStatus {
            let aggregator = self.inner.read().await;
            let nodes = self.registered_nodes.read().await;

            VerifiedAggregatorStatus {
                round: aggregator.current_round(),
                registered_nodes: nodes.len(),
                submitted_nodes: aggregator.submission_count(),
                expected_nodes: aggregator.inner.config.expected_nodes,
                require_proofs: aggregator.verified_config.require_proofs,
                max_gradient_norm: aggregator.verified_config.max_gradient_norm,
                is_complete: aggregator.is_round_complete(),
            }
        }
    }

    /// Status for verified aggregator
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct VerifiedAggregatorStatus {
        pub round: Round,
        pub registered_nodes: usize,
        pub submitted_nodes: usize,
        pub expected_nodes: usize,
        pub require_proofs: bool,
        pub max_gradient_norm: f32,
        pub is_complete: bool,
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::byzantine::Defense;
        use crate::proofs::SecurityLevel;
        use ndarray::array;

        fn test_config() -> ProofConfig {
            ProofConfig {
                security_level: SecurityLevel::Standard96,
                parallel: false,
                max_proof_size: 0,
            }
        }

        #[test]
        fn test_verified_aggregator_with_proofs() {
            let config = AggregatorConfig::default()
                .with_expected_nodes(2)
                .with_defense(Defense::FedAvg);

            let verified_config = VerifiedAggregationConfig {
                require_proofs: true,
                max_gradient_norm: 10.0,
                proof_config: test_config(),
            };

            let mut aggregator = VerifiedAggregator::new(config, verified_config);

            // Submit with proof
            aggregator.submit_with_proof("node1", array![0.1, 0.2, 0.3]).unwrap();
            aggregator.submit_with_proof("node2", array![0.4, 0.5, 0.6]).unwrap();

            assert!(aggregator.is_round_complete());
            let result = aggregator.finalize_round().unwrap();
            assert_eq!(result.len(), 3);
        }

        #[test]
        fn test_verified_aggregator_requires_proofs() {
            let config = AggregatorConfig::default()
                .with_expected_nodes(2)
                .with_defense(Defense::FedAvg);

            let verified_config = VerifiedAggregationConfig {
                require_proofs: true,
                max_gradient_norm: 10.0,
                proof_config: test_config(),
            };

            let mut aggregator = VerifiedAggregator::new(config, verified_config);

            // Should fail without proofs
            let result = aggregator.submit_unverified("node1", array![0.1, 0.2]);
            assert!(result.is_err());
        }

        #[test]
        fn test_verified_aggregator_optional_proofs() {
            let config = AggregatorConfig::default()
                .with_expected_nodes(2)
                .with_defense(Defense::FedAvg);

            let verified_config = VerifiedAggregationConfig {
                require_proofs: false,
                max_gradient_norm: 10.0,
                proof_config: test_config(),
            };

            let mut aggregator = VerifiedAggregator::new(config, verified_config);

            // Should work without proofs
            aggregator.submit_unverified("node1", array![0.1, 0.2]).unwrap();
            aggregator.submit_with_proof("node2", array![0.3, 0.4]).unwrap();

            assert!(aggregator.is_round_complete());
        }

        #[tokio::test]
        async fn test_verified_async_aggregator() {
            let config = AggregatorConfig::default()
                .with_expected_nodes(2)
                .with_defense(Defense::FedAvg);

            let verified_config = VerifiedAggregationConfig {
                require_proofs: true,
                max_gradient_norm: 10.0,
                proof_config: test_config(),
            };

            let aggregator = VerifiedAsyncAggregator::new(config, verified_config);

            aggregator.register_node("node1").await.unwrap();
            aggregator.register_node("node2").await.unwrap();

            aggregator.submit_with_proof("node1", array![0.1, 0.2]).await.unwrap();
            assert!(!aggregator.is_round_complete().await);

            aggregator.submit_with_proof("node2", array![0.3, 0.4]).await.unwrap();
            assert!(aggregator.is_round_complete().await);

            let result = aggregator.finalize_round().await.unwrap();
            assert_eq!(result.len(), 2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};

    #[test]
    fn test_basic_aggregation() {
        let config = AggregatorConfig::default()
            .with_expected_nodes(3)
            .with_defense(Defense::FedAvg);

        let mut aggregator = Aggregator::new(config);

        aggregator.submit("node1", array![1.0, 2.0, 3.0]).unwrap();
        aggregator.submit("node2", array![4.0, 5.0, 6.0]).unwrap();
        aggregator.submit("node3", array![7.0, 8.0, 9.0]).unwrap();

        assert!(aggregator.is_round_complete());

        let result = aggregator.finalize_round().unwrap();
        assert_eq!(result, array![4.0, 5.0, 6.0]);
        assert_eq!(aggregator.current_round(), 1);
    }

    #[test]
    fn test_duplicate_submission_error() {
        let config = AggregatorConfig::default().with_expected_nodes(3);
        let mut aggregator = Aggregator::new(config);

        aggregator.submit("node1", array![1.0]).unwrap();
        let result = aggregator.submit("node1", array![2.0]);

        assert!(matches!(
            result,
            Err(AggregatorError::DuplicateSubmission { .. })
        ));
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let config = AggregatorConfig::default().with_expected_nodes(3);
        let mut aggregator = Aggregator::new(config);

        aggregator.submit("node1", array![1.0, 2.0]).unwrap();
        let result = aggregator.submit("node2", array![1.0, 2.0, 3.0]);

        assert!(matches!(
            result,
            Err(AggregatorError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_memory_limit() {
        let config = AggregatorConfig::default()
            .with_expected_nodes(2)
            .with_max_memory(100); // Very small limit

        let mut aggregator = Aggregator::new(config);

        // This should exceed memory limit
        let large_gradient = Array1::zeros(1000);
        let result = aggregator.submit("node1", large_gradient);

        assert!(matches!(
            result,
            Err(AggregatorError::MemoryLimitExceeded { .. })
        ));
    }

    #[tokio::test]
    async fn test_async_aggregator() {
        let config = AggregatorConfig::default()
            .with_expected_nodes(2)
            .with_defense(Defense::FedAvg);

        let aggregator = AsyncAggregator::new(config);

        // Register nodes
        aggregator.register_node("node1").await.unwrap();
        aggregator.register_node("node2").await.unwrap();

        // Submit gradients
        aggregator.submit("node1", array![1.0, 2.0]).await.unwrap();
        assert!(!aggregator.is_round_complete().await);

        aggregator.submit("node2", array![3.0, 4.0]).await.unwrap();
        assert!(aggregator.is_round_complete().await);

        // Get result
        let result = aggregator.get_aggregated_gradient().await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), array![2.0, 3.0]);
    }

    #[tokio::test]
    async fn test_unregistered_node_error() {
        let config = AggregatorConfig::default();
        let aggregator = AsyncAggregator::new(config);

        let result = aggregator.submit("unknown", array![1.0]).await;
        assert!(matches!(result, Err(AggregatorError::InvalidNode(_))));
    }
}
