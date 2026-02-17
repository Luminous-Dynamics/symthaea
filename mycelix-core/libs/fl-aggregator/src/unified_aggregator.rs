//! Unified Aggregator for multi-paradigm federated learning.
//!
//! This module provides a unified aggregation interface that handles both:
//! - Traditional dense gradient-based FL (via `ByzantineAggregator`)
//! - Hyperdimensional computing (HDC) FL (via `HdcByzantineAggregator`)
//!
//! # Architecture
//!
//! The `UnifiedAggregator` accepts `UnifiedPayload` submissions and automatically
//! routes them to the appropriate aggregation backend based on payload type:
//!
//! - Dense/Sparse/Quantized → `ByzantineAggregator`
//! - Hypervector/BinaryHypervector → `HdcByzantineAggregator`
//!
//! # Example
//!
//! ```rust,ignore
//! use fl_aggregator::unified_aggregator::{
//!     UnifiedAggregator, UnifiedAggregatorConfig,
//!     PayloadSubmission, UnifiedAggregationResult,
//! };
//! use fl_aggregator::payload::{UnifiedPayload, DenseGradient, Hypervector};
//!
//! let config = UnifiedAggregatorConfig::default();
//! let mut aggregator = UnifiedAggregator::new(config);
//!
//! // Submit dense gradients
//! aggregator.submit("node1", DenseGradient::from_vec(vec![1.0, 2.0]).into())?;
//! aggregator.submit("node2", DenseGradient::from_vec(vec![3.0, 4.0]).into())?;
//!
//! // Or submit hypervectors
//! // aggregator.submit("node1", Hypervector::random(16384).into())?;
//!
//! if aggregator.can_aggregate() {
//!     let result = aggregator.finalize_round()?;
//!     match result {
//!         UnifiedAggregationResult::Dense(g) => println!("Dense result: {:?}", g),
//!         UnifiedAggregationResult::Hyper(h) => println!("HDC result: {} components", h.components.len()),
//!         UnifiedAggregationResult::Binary(b) => println!("Binary result: {} bits", b.bit_dimension),
//!     }
//! }
//! ```

use crate::byzantine::{ByzantineAggregator, Defense, DefenseConfig};
use crate::error::{AggregatorError, Result};
use crate::hdc_byzantine::{HdcByzantineAggregator, HdcDefense, HdcDefenseConfig};
use crate::payload::{
    BinaryHypervector, DenseGradient,
    Hypervector, Payload, PayloadType, UnifiedPayload,
};
use crate::{Gradient, NodeId, Round};

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

// =============================================================================
// Unified Aggregator Types
// =============================================================================

/// Generic payload submission record.
#[derive(Clone, Debug)]
pub struct PayloadSubmission<P> {
    /// Node that submitted.
    pub node_id: NodeId,

    /// The payload data.
    pub payload: P,

    /// When it was submitted.
    pub timestamp: Instant,

    /// Round number.
    pub round: u64,
}

impl<P> PayloadSubmission<P> {
    /// Create a new submission.
    pub fn new(node_id: impl Into<NodeId>, payload: P, round: u64) -> Self {
        Self {
            node_id: node_id.into(),
            payload,
            timestamp: Instant::now(),
            round,
        }
    }
}

/// Configuration for the unified aggregator.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedAggregatorConfig {
    /// Expected number of nodes per round.
    pub expected_nodes: usize,

    /// Byzantine defense for dense gradients.
    pub dense_defense: Defense,

    /// HDC defense for hypervectors.
    pub hdc_defense: HdcDefense,

    /// Whether to require homogeneous payload types.
    /// If true, all submissions in a round must be the same type.
    pub require_homogeneous: bool,

    /// Maximum memory usage in bytes (0 = unlimited).
    pub max_memory_bytes: usize,

    /// Minimum nodes to proceed with aggregation (if less than expected_nodes).
    pub min_nodes_to_aggregate: Option<usize>,
}

impl Default for UnifiedAggregatorConfig {
    fn default() -> Self {
        Self {
            expected_nodes: 10,
            dense_defense: Defense::Krum { f: 1 },
            hdc_defense: HdcDefense::SimilarityFilter { threshold: 0.7 },
            require_homogeneous: true,
            max_memory_bytes: 2_000_000_000, // 2GB
            min_nodes_to_aggregate: None,
        }
    }
}

impl UnifiedAggregatorConfig {
    /// Set expected number of nodes.
    pub fn with_expected_nodes(mut self, n: usize) -> Self {
        self.expected_nodes = n;
        self
    }

    /// Set dense defense algorithm.
    pub fn with_dense_defense(mut self, defense: Defense) -> Self {
        self.dense_defense = defense;
        self
    }

    /// Set HDC defense algorithm.
    pub fn with_hdc_defense(mut self, defense: HdcDefense) -> Self {
        self.hdc_defense = defense;
        self
    }

    /// Set whether to require homogeneous payloads.
    pub fn with_require_homogeneous(mut self, require: bool) -> Self {
        self.require_homogeneous = require;
        self
    }

    /// Set memory limit.
    pub fn with_max_memory(mut self, bytes: usize) -> Self {
        self.max_memory_bytes = bytes;
        self
    }

    /// Set minimum nodes to aggregate.
    pub fn with_min_nodes(mut self, min: usize) -> Self {
        self.min_nodes_to_aggregate = Some(min);
        self
    }
}

/// Result of unified aggregation.
#[derive(Clone, Debug)]
pub enum UnifiedAggregationResult {
    /// Dense gradient result.
    Dense(DenseGradient),

    /// Hypervector (i8) result.
    Hyper(Hypervector),

    /// Binary hypervector result.
    Binary(BinaryHypervector),
}

impl UnifiedAggregationResult {
    /// Get the payload type of the result.
    pub fn payload_type(&self) -> PayloadType {
        match self {
            UnifiedAggregationResult::Dense(_) => PayloadType::DenseGradient,
            UnifiedAggregationResult::Hyper(_) => PayloadType::HyperEncoded,
            UnifiedAggregationResult::Binary(_) => PayloadType::BinaryHypervector,
        }
    }

    /// Get size in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            UnifiedAggregationResult::Dense(g) => g.size_bytes(),
            UnifiedAggregationResult::Hyper(h) => h.size_bytes(),
            UnifiedAggregationResult::Binary(b) => b.size_bytes(),
        }
    }
}

// =============================================================================
// Unified Aggregator Implementation
// =============================================================================

/// Unified aggregator that handles both dense and HDC payloads.
pub struct UnifiedAggregator {
    config: UnifiedAggregatorConfig,
    dense_aggregator: ByzantineAggregator,
    hdc_aggregator: HdcByzantineAggregator,
    round: Round,
    submissions: HashMap<NodeId, PayloadSubmission<UnifiedPayload>>,
    detected_type: Option<PayloadType>,
    round_start: Instant,
    memory_used: usize,
}

impl UnifiedAggregator {
    /// Create a new unified aggregator.
    pub fn new(config: UnifiedAggregatorConfig) -> Self {
        let dense_config = DefenseConfig::with_defense(config.dense_defense.clone());
        let hdc_config = HdcDefenseConfig::new(config.hdc_defense.clone());

        Self {
            config,
            dense_aggregator: ByzantineAggregator::new(dense_config),
            hdc_aggregator: HdcByzantineAggregator::new(hdc_config),
            round: 0,
            submissions: HashMap::new(),
            detected_type: None,
            round_start: Instant::now(),
            memory_used: 0,
        }
    }

    /// Submit a payload from a node.
    pub fn submit(&mut self, node_id: impl Into<NodeId>, payload: UnifiedPayload) -> Result<()> {
        let node_id = node_id.into();

        // Check for duplicate submission
        if self.submissions.contains_key(&node_id) {
            return Err(AggregatorError::DuplicateSubmission {
                node: node_id,
                round: self.round,
            });
        }

        // Validate payload
        if !payload.is_valid() {
            return Err(AggregatorError::InvalidConfig(format!(
                "Invalid payload from node {}",
                node_id
            )));
        }

        let payload_type = payload.payload_type();

        // Check for homogeneous payloads if required
        if self.config.require_homogeneous {
            match &self.detected_type {
                Some(expected) if *expected != payload_type => {
                    return Err(AggregatorError::InvalidConfig(format!(
                        "Homogeneous mode: expected {:?} but got {:?} from node {}",
                        expected, payload_type, node_id
                    )));
                }
                None => self.detected_type = Some(payload_type),
                _ => {}
            }
        }

        // Check memory limit
        let payload_bytes = payload.size_bytes();
        if self.config.max_memory_bytes > 0
            && self.memory_used + payload_bytes > self.config.max_memory_bytes
        {
            return Err(AggregatorError::MemoryLimitExceeded {
                used: self.memory_used + payload_bytes,
                limit: self.config.max_memory_bytes,
            });
        }

        // Store submission
        let submission = PayloadSubmission::new(node_id.clone(), payload, self.round);
        self.submissions.insert(node_id.clone(), submission);
        self.memory_used += payload_bytes;

        tracing::debug!(
            "Payload submitted: node={}, type={:?}, round={}, total={}/{}",
            node_id,
            payload_type,
            self.round,
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

    /// Get current round number.
    pub fn current_round(&self) -> Round {
        self.round
    }

    /// Get elapsed time since round start.
    pub fn round_elapsed(&self) -> std::time::Duration {
        self.round_start.elapsed()
    }

    /// Get current memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.memory_used
    }

    /// Get the detected payload type for this round.
    pub fn detected_type(&self) -> Option<PayloadType> {
        self.detected_type
    }

    /// Finalize the round and return aggregated result.
    pub fn finalize_round(&mut self) -> Result<UnifiedAggregationResult> {
        if self.submissions.is_empty() {
            return Err(AggregatorError::NoGradients(self.round));
        }

        // Determine aggregation strategy based on detected type
        let result = match self.detected_type {
            Some(PayloadType::DenseGradient)
            | Some(PayloadType::SparseGradient)
            | Some(PayloadType::QuantizedGradient) => {
                self.aggregate_dense()?
            }
            Some(PayloadType::HyperEncoded) => self.aggregate_hypervector()?,
            Some(PayloadType::BinaryHypervector) => self.aggregate_binary()?,
            None => {
                // Mixed or unknown - try to infer from first submission
                let first_type = self
                    .submissions
                    .values()
                    .next()
                    .map(|s| s.payload.payload_type());
                match first_type {
                    Some(PayloadType::HyperEncoded) => self.aggregate_hypervector()?,
                    Some(PayloadType::BinaryHypervector) => self.aggregate_binary()?,
                    _ => self.aggregate_dense()?,
                }
            }
        };

        let submission_count = self.submissions.len();
        let aggregation_time = self.round_start.elapsed();

        tracing::info!(
            "Round {} finalized: {} payloads aggregated in {:?}, result type: {:?}",
            self.round,
            submission_count,
            aggregation_time,
            result.payload_type()
        );

        // Clear for next round
        self.submissions.clear();
        self.detected_type = None;
        self.memory_used = 0;
        self.round += 1;
        self.round_start = Instant::now();

        Ok(result)
    }

    /// Aggregate dense gradients (includes sparse and quantized).
    fn aggregate_dense(&self) -> Result<UnifiedAggregationResult> {
        // Convert all to dense gradients
        let gradients: Vec<Gradient> = self
            .submissions
            .values()
            .filter_map(|s| s.payload.to_dense())
            .map(|d| d.values)
            .collect();

        if gradients.is_empty() {
            return Err(AggregatorError::NoGradients(self.round));
        }

        let result = self.dense_aggregator.aggregate(&gradients)?;
        Ok(UnifiedAggregationResult::Dense(DenseGradient::from_array(result)))
    }

    /// Aggregate hypervectors.
    fn aggregate_hypervector(&self) -> Result<UnifiedAggregationResult> {
        let hypervectors: Vec<Hypervector> = self
            .submissions
            .values()
            .filter_map(|s| match &s.payload {
                UnifiedPayload::Hyper(h) => Some(h.clone()),
                _ => None,
            })
            .collect();

        if hypervectors.is_empty() {
            return Err(AggregatorError::NoGradients(self.round));
        }

        // Convert to references for the aggregator API
        let refs: Vec<&Hypervector> = hypervectors.iter().collect();
        let result = self.hdc_aggregator.aggregate(&refs)?;
        Ok(UnifiedAggregationResult::Hyper(result))
    }

    /// Aggregate binary hypervectors.
    fn aggregate_binary(&self) -> Result<UnifiedAggregationResult> {
        let hypervectors: Vec<BinaryHypervector> = self
            .submissions
            .values()
            .filter_map(|s| match &s.payload {
                UnifiedPayload::Binary(b) => Some(b.clone()),
                _ => None,
            })
            .collect();

        if hypervectors.is_empty() {
            return Err(AggregatorError::NoGradients(self.round));
        }

        // Convert to references for the aggregator API
        let refs: Vec<&BinaryHypervector> = hypervectors.iter().collect();
        let result = self.hdc_aggregator.aggregate_binary(&refs)?;
        Ok(UnifiedAggregationResult::Binary(result))
    }

    /// Reset the aggregator state.
    pub fn reset(&mut self) {
        self.submissions.clear();
        self.detected_type = None;
        self.memory_used = 0;
        self.round = 0;
        self.round_start = Instant::now();
    }
}

// =============================================================================
// Async Unified Aggregator
// =============================================================================

/// Async unified aggregator for concurrent node handling.
pub struct AsyncUnifiedAggregator {
    inner: Arc<RwLock<UnifiedAggregator>>,
    registered_nodes: Arc<RwLock<HashSet<NodeId>>>,
}

impl AsyncUnifiedAggregator {
    /// Create a new async unified aggregator.
    pub fn new(config: UnifiedAggregatorConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(UnifiedAggregator::new(config))),
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

    /// Get the number of registered nodes.
    pub async fn registered_count(&self) -> usize {
        let nodes = self.registered_nodes.read().await;
        nodes.len()
    }

    /// Submit a payload from a node.
    pub async fn submit(&self, node_id: impl Into<NodeId>, payload: UnifiedPayload) -> Result<()> {
        let node_id = node_id.into();

        // Verify node is registered
        if !self.is_registered(&node_id).await {
            return Err(AggregatorError::InvalidNode(node_id));
        }

        let mut aggregator = self.inner.write().await;
        aggregator.submit(node_id, payload)
    }

    /// Check if round is complete.
    pub async fn is_round_complete(&self) -> bool {
        let aggregator = self.inner.read().await;
        aggregator.is_round_complete()
    }

    /// Check if aggregation can proceed.
    pub async fn can_aggregate(&self) -> bool {
        let aggregator = self.inner.read().await;
        aggregator.can_aggregate()
    }

    /// Get submission count.
    pub async fn submission_count(&self) -> usize {
        let aggregator = self.inner.read().await;
        aggregator.submission_count()
    }

    /// Get aggregated result if round is complete.
    pub async fn get_aggregated(&self) -> Result<Option<UnifiedAggregationResult>> {
        let mut aggregator = self.inner.write().await;
        if aggregator.is_round_complete() {
            Ok(Some(aggregator.finalize_round()?))
        } else {
            Ok(None)
        }
    }

    /// Force finalize even if not all nodes submitted.
    pub async fn force_finalize(&self) -> Result<UnifiedAggregationResult> {
        let mut aggregator = self.inner.write().await;
        aggregator.finalize_round()
    }

    /// Get current round.
    pub async fn current_round(&self) -> Round {
        let aggregator = self.inner.read().await;
        aggregator.current_round()
    }

    /// Get current status.
    pub async fn status(&self) -> UnifiedAggregatorStatus {
        let aggregator = self.inner.read().await;
        let nodes = self.registered_nodes.read().await;

        UnifiedAggregatorStatus {
            round: aggregator.current_round(),
            registered_nodes: nodes.len(),
            submitted_nodes: aggregator.submission_count(),
            expected_nodes: aggregator.config.expected_nodes,
            detected_type: aggregator.detected_type(),
            memory_bytes: aggregator.memory_usage(),
            round_elapsed: aggregator.round_elapsed(),
            is_complete: aggregator.is_round_complete(),
            can_aggregate: aggregator.can_aggregate(),
        }
    }

    /// Reset the aggregator.
    pub async fn reset(&self) {
        let mut aggregator = self.inner.write().await;
        aggregator.reset();
    }
}

impl Clone for AsyncUnifiedAggregator {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            registered_nodes: Arc::clone(&self.registered_nodes),
        }
    }
}

/// Status information for monitoring.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedAggregatorStatus {
    /// Current round number.
    pub round: Round,

    /// Number of registered nodes.
    pub registered_nodes: usize,

    /// Number of nodes that have submitted.
    pub submitted_nodes: usize,

    /// Expected number of nodes.
    pub expected_nodes: usize,

    /// Detected payload type for this round.
    pub detected_type: Option<PayloadType>,

    /// Current memory usage in bytes.
    pub memory_bytes: usize,

    /// Time elapsed since round start.
    #[serde(with = "humantime_serde")]
    pub round_elapsed: std::time::Duration,

    /// Whether the round is complete.
    pub is_complete: bool,

    /// Whether aggregation can proceed.
    pub can_aggregate: bool,
}

mod humantime_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = humantime::format_duration(*duration).to_string();
        s.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        humantime::parse_duration(&s).map_err(serde::de::Error::custom)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::SparseGradient;
    use ndarray::array;

    fn test_config() -> UnifiedAggregatorConfig {
        UnifiedAggregatorConfig::default()
            .with_expected_nodes(3)
            .with_dense_defense(Defense::FedAvg)
            .with_hdc_defense(HdcDefense::HdcBundle)
    }

    #[test]
    fn test_dense_aggregation() {
        let mut aggregator = UnifiedAggregator::new(test_config());

        aggregator
            .submit("node1", DenseGradient::from_vec(vec![1.0, 2.0, 3.0]).into())
            .unwrap();
        aggregator
            .submit("node2", DenseGradient::from_vec(vec![4.0, 5.0, 6.0]).into())
            .unwrap();
        aggregator
            .submit("node3", DenseGradient::from_vec(vec![7.0, 8.0, 9.0]).into())
            .unwrap();

        assert!(aggregator.is_round_complete());
        assert_eq!(aggregator.detected_type(), Some(PayloadType::DenseGradient));

        let result = aggregator.finalize_round().unwrap();
        match result {
            UnifiedAggregationResult::Dense(g) => {
                assert_eq!(g.values, array![4.0, 5.0, 6.0]);
            }
            _ => panic!("Expected dense result"),
        }

        assert_eq!(aggregator.current_round(), 1);
    }

    #[test]
    fn test_hypervector_aggregation() {
        let config = UnifiedAggregatorConfig::default()
            .with_expected_nodes(3)
            .with_hdc_defense(HdcDefense::HdcBundle);
        let mut aggregator = UnifiedAggregator::new(config);

        aggregator
            .submit("node1", Hypervector::new(vec![10, 20, 30]).into())
            .unwrap();
        aggregator
            .submit("node2", Hypervector::new(vec![20, 30, 40]).into())
            .unwrap();
        aggregator
            .submit("node3", Hypervector::new(vec![30, 40, 50]).into())
            .unwrap();

        assert!(aggregator.is_round_complete());
        assert_eq!(aggregator.detected_type(), Some(PayloadType::HyperEncoded));

        let result = aggregator.finalize_round().unwrap();
        match result {
            UnifiedAggregationResult::Hyper(h) => {
                assert_eq!(h.components.len(), 3);
                // Average: (10+20+30)/3=20, (20+30+40)/3=30, (30+40+50)/3=40
                assert_eq!(h.components, vec![20, 30, 40]);
            }
            _ => panic!("Expected hypervector result"),
        }
    }

    #[test]
    fn test_binary_aggregation() {
        let config = UnifiedAggregatorConfig::default()
            .with_expected_nodes(3)
            .with_hdc_defense(HdcDefense::HdcBundle);
        let mut aggregator = UnifiedAggregator::new(config);

        // Create binary hypervectors
        let bhv1 = BinaryHypervector::from_bits(&[true, true, false, false]);
        let bhv2 = BinaryHypervector::from_bits(&[true, false, true, false]);
        let bhv3 = BinaryHypervector::from_bits(&[true, true, true, false]);

        aggregator.submit("node1", bhv1.into()).unwrap();
        aggregator.submit("node2", bhv2.into()).unwrap();
        aggregator.submit("node3", bhv3.into()).unwrap();

        assert!(aggregator.is_round_complete());

        let result = aggregator.finalize_round().unwrap();
        match result {
            UnifiedAggregationResult::Binary(b) => {
                // Majority: 3/3=true, 2/3=true, 2/3=true, 0/3=false
                assert_eq!(b.get_bit(0), Some(true));
                assert_eq!(b.get_bit(1), Some(true));
                assert_eq!(b.get_bit(2), Some(true));
                assert_eq!(b.get_bit(3), Some(false));
            }
            _ => panic!("Expected binary result"),
        }
    }

    #[test]
    fn test_sparse_to_dense_aggregation() {
        let mut aggregator = UnifiedAggregator::new(test_config());

        // Submit sparse gradients
        let sparse1 = SparseGradient::new(vec![0, 2], vec![1.0, 3.0], 4);
        let sparse2 = SparseGradient::new(vec![0, 2], vec![2.0, 6.0], 4);
        let sparse3 = SparseGradient::new(vec![0, 2], vec![3.0, 9.0], 4);

        aggregator.submit("node1", sparse1.into()).unwrap();
        aggregator.submit("node2", sparse2.into()).unwrap();
        aggregator.submit("node3", sparse3.into()).unwrap();

        let result = aggregator.finalize_round().unwrap();
        match result {
            UnifiedAggregationResult::Dense(g) => {
                assert_eq!(g.values.len(), 4);
                assert!((g.values[0] - 2.0).abs() < 0.01);
                assert!((g.values[2] - 6.0).abs() < 0.01);
            }
            _ => panic!("Expected dense result"),
        }
    }

    #[test]
    fn test_duplicate_submission_error() {
        let mut aggregator = UnifiedAggregator::new(test_config());

        aggregator
            .submit("node1", DenseGradient::from_vec(vec![1.0]).into())
            .unwrap();
        let result = aggregator.submit("node1", DenseGradient::from_vec(vec![2.0]).into());

        assert!(matches!(
            result,
            Err(AggregatorError::DuplicateSubmission { .. })
        ));
    }

    #[test]
    fn test_homogeneous_mode_error() {
        let config = UnifiedAggregatorConfig::default()
            .with_expected_nodes(3)
            .with_require_homogeneous(true);
        let mut aggregator = UnifiedAggregator::new(config);

        aggregator
            .submit("node1", DenseGradient::from_vec(vec![1.0]).into())
            .unwrap();

        // Should fail - trying to submit hypervector when dense was detected
        let result = aggregator.submit("node2", Hypervector::new(vec![1]).into());

        assert!(matches!(result, Err(AggregatorError::InvalidConfig(_))));
    }

    #[test]
    fn test_mixed_mode() {
        let config = UnifiedAggregatorConfig::default()
            .with_expected_nodes(3)
            .with_require_homogeneous(false);
        let mut aggregator = UnifiedAggregator::new(config);

        // In non-homogeneous mode, we can submit different types
        // (though aggregation will only use compatible types)
        aggregator
            .submit("node1", DenseGradient::from_vec(vec![1.0]).into())
            .unwrap();
        aggregator
            .submit("node2", DenseGradient::from_vec(vec![2.0]).into())
            .unwrap();
        aggregator
            .submit("node3", DenseGradient::from_vec(vec![3.0]).into())
            .unwrap();

        assert!(aggregator.is_round_complete());
    }

    #[test]
    fn test_memory_limit() {
        let config = UnifiedAggregatorConfig::default()
            .with_expected_nodes(2)
            .with_max_memory(50); // Very small limit

        let mut aggregator = UnifiedAggregator::new(config);

        // This should exceed memory limit
        let large_gradient = DenseGradient::from_vec(vec![0.0; 100]);
        let result = aggregator.submit("node1", large_gradient.into());

        assert!(matches!(
            result,
            Err(AggregatorError::MemoryLimitExceeded { .. })
        ));
    }

    #[test]
    fn test_hdc_similarity_filter() {
        // Test that similarity filtering works via unified aggregator
        let config = UnifiedAggregatorConfig::default()
            .with_expected_nodes(4)
            .with_hdc_defense(HdcDefense::SimilarityFilter { threshold: 0.5 });
        let mut aggregator = UnifiedAggregator::new(config);

        // Similar hypervectors should pass filter
        aggregator
            .submit("node1", Hypervector::new(vec![100, 50, 25]).into())
            .unwrap();
        aggregator
            .submit("node2", Hypervector::new(vec![90, 45, 20]).into())
            .unwrap();
        aggregator
            .submit("node3", Hypervector::new(vec![95, 48, 23]).into())
            .unwrap();
        // Outlier
        aggregator
            .submit("node4", Hypervector::new(vec![-100, -50, -25]).into())
            .unwrap();

        let result = aggregator.finalize_round().unwrap();
        match result {
            UnifiedAggregationResult::Hyper(h) => {
                // Result should be similar to the majority, not the outlier
                assert!(h.components[0] > 0);
            }
            _ => panic!("Expected hypervector result"),
        }
    }

    #[test]
    fn test_hdc_krum_via_unified() {
        // Test that HdcKrum works via unified aggregator
        let config = UnifiedAggregatorConfig::default()
            .with_expected_nodes(5)
            .with_hdc_defense(HdcDefense::HdcKrum { f: 1 });
        let mut aggregator = UnifiedAggregator::new(config);

        // Similar hypervectors
        aggregator
            .submit("node1", Hypervector::new(vec![100, 100, 100]).into())
            .unwrap();
        aggregator
            .submit("node2", Hypervector::new(vec![95, 95, 95]).into())
            .unwrap();
        aggregator
            .submit("node3", Hypervector::new(vec![90, 90, 90]).into())
            .unwrap();
        aggregator
            .submit("node4", Hypervector::new(vec![88, 88, 88]).into())
            .unwrap();
        // Outlier
        aggregator
            .submit("node5", Hypervector::new(vec![-100, -100, -100]).into())
            .unwrap();

        let result = aggregator.finalize_round().unwrap();
        match result {
            UnifiedAggregationResult::Hyper(h) => {
                // Should not select the outlier
                assert!(h.components[0] > 0);
            }
            _ => panic!("Expected hypervector result"),
        }
    }

    #[tokio::test]
    async fn test_async_aggregator() {
        let config = UnifiedAggregatorConfig::default()
            .with_expected_nodes(2)
            .with_dense_defense(Defense::FedAvg);

        let aggregator = AsyncUnifiedAggregator::new(config);

        // Register nodes
        aggregator.register_node("node1").await.unwrap();
        aggregator.register_node("node2").await.unwrap();

        assert!(aggregator.is_registered("node1").await);
        assert!(aggregator.is_registered("node2").await);
        assert!(!aggregator.is_registered("node3").await);

        // Submit gradients
        aggregator
            .submit("node1", DenseGradient::from_vec(vec![1.0, 2.0]).into())
            .await
            .unwrap();
        assert!(!aggregator.is_round_complete().await);

        aggregator
            .submit("node2", DenseGradient::from_vec(vec![3.0, 4.0]).into())
            .await
            .unwrap();
        assert!(aggregator.is_round_complete().await);

        // Get result
        let result = aggregator.get_aggregated().await.unwrap();
        assert!(result.is_some());

        match result.unwrap() {
            UnifiedAggregationResult::Dense(g) => {
                assert_eq!(g.values, array![2.0, 3.0]);
            }
            _ => panic!("Expected dense result"),
        }

        // Round should have advanced
        assert_eq!(aggregator.current_round().await, 1);
    }

    #[tokio::test]
    async fn test_async_unregistered_node_error() {
        let config = UnifiedAggregatorConfig::default();
        let aggregator = AsyncUnifiedAggregator::new(config);

        let result = aggregator
            .submit("unknown", DenseGradient::from_vec(vec![1.0]).into())
            .await;

        assert!(matches!(result, Err(AggregatorError::InvalidNode(_))));
    }

    #[tokio::test]
    async fn test_async_status() {
        let config = UnifiedAggregatorConfig::default()
            .with_expected_nodes(3)
            .with_min_nodes(2);

        let aggregator = AsyncUnifiedAggregator::new(config);

        aggregator.register_node("node1").await.unwrap();
        aggregator.register_node("node2").await.unwrap();

        aggregator
            .submit("node1", DenseGradient::from_vec(vec![1.0]).into())
            .await
            .unwrap();

        let status = aggregator.status().await;
        assert_eq!(status.round, 0);
        assert_eq!(status.registered_nodes, 2);
        assert_eq!(status.submitted_nodes, 1);
        assert_eq!(status.expected_nodes, 3);
        assert!(!status.is_complete);
        assert!(!status.can_aggregate);

        aggregator
            .submit("node2", DenseGradient::from_vec(vec![2.0]).into())
            .await
            .unwrap();

        let status = aggregator.status().await;
        assert!(status.can_aggregate); // min_nodes = 2
        assert!(!status.is_complete); // expected_nodes = 3
    }

    #[test]
    fn test_payload_submission() {
        let submission = PayloadSubmission::new(
            "test_node",
            DenseGradient::from_vec(vec![1.0, 2.0]),
            5,
        );

        assert_eq!(submission.node_id, "test_node");
        assert_eq!(submission.round, 5);
        assert_eq!(submission.payload.values.len(), 2);
    }

    #[test]
    fn test_unified_aggregation_result() {
        let dense_result = UnifiedAggregationResult::Dense(
            DenseGradient::from_vec(vec![1.0, 2.0, 3.0])
        );
        assert_eq!(dense_result.payload_type(), PayloadType::DenseGradient);
        assert_eq!(dense_result.size_bytes(), 12);

        let hyper_result = UnifiedAggregationResult::Hyper(
            Hypervector::new(vec![1, 2, 3, 4])
        );
        assert_eq!(hyper_result.payload_type(), PayloadType::HyperEncoded);
        assert_eq!(hyper_result.size_bytes(), 4);

        let binary_result = UnifiedAggregationResult::Binary(
            BinaryHypervector::from_bits(&[true, false, true, false, true, false, true, false])
        );
        assert_eq!(binary_result.payload_type(), PayloadType::BinaryHypervector);
        assert_eq!(binary_result.size_bytes(), 1);
    }

    #[test]
    fn test_config_builders() {
        let config = UnifiedAggregatorConfig::default()
            .with_expected_nodes(20)
            .with_dense_defense(Defense::Median)
            .with_hdc_defense(HdcDefense::HdcKrum { f: 2 })
            .with_require_homogeneous(false)
            .with_max_memory(1_000_000)
            .with_min_nodes(10);

        assert_eq!(config.expected_nodes, 20);
        assert_eq!(config.dense_defense, Defense::Median);
        assert!(matches!(config.hdc_defense, HdcDefense::HdcKrum { f: 2 }));
        assert!(!config.require_homogeneous);
        assert_eq!(config.max_memory_bytes, 1_000_000);
        assert_eq!(config.min_nodes_to_aggregate, Some(10));
    }
}
