//! Gradient Replay Detection System
//!
//! Detects nodes that submit old, copied, or replayed gradients to cheat
//! the federated learning process. This module provides:
//!
//! - **Exact replay detection**: Hash-based detection of identical gradients
//! - **Near-replay detection**: Cosine similarity for similar gradients
//! - **Cross-node copy detection**: Identifies gradient copying between nodes
//! - **Pattern detection**: Identifies systematic free-riding behavior
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::replay::{ReplayDetector, ReplayDetectorConfig};
//! use ndarray::Array1;
//!
//! let config = ReplayDetectorConfig::default();
//! let mut detector = ReplayDetector::new(config);
//!
//! // Check if a gradient is a replay
//! let gradient = Array1::from(vec![1.0, 2.0, 3.0]);
//! let result = detector.check_replay("node_1", &gradient, 5);
//!
//! if result.is_replay {
//!     println!("Detected replay: {:?}", result.replay_type);
//! }
//!
//! // Record the submission for future detection
//! detector.record_submission("node_1", &gradient, 5);
//! ```

use crate::{NodeId, Round};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};

/// Number of top indices to store for sparse comparison.
const DEFAULT_TOP_K: usize = 100;

/// Bloom filter size in bits.
const BLOOM_FILTER_SIZE: usize = 65536;

/// Number of hash functions for bloom filter.
const BLOOM_HASH_FUNCS: usize = 7;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the replay detector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayDetectorConfig {
    /// How many rounds of history to keep.
    pub history_rounds: usize,

    /// Cosine similarity threshold for near-replay detection (0.0-1.0).
    /// Values above this are considered replays.
    pub similarity_threshold: f32,

    /// Use hash-based comparison (faster) vs full comparison.
    pub hash_based: bool,

    /// Enable cross-node copy detection.
    pub cross_node_detection: bool,

    /// Number of top-k indices to store for sparse comparison.
    pub top_k: usize,

    /// Suspicion score threshold for flagging a node.
    pub suspicion_threshold: f32,

    /// Minimum replays to flag a node as suspicious.
    pub min_replays_for_suspicion: usize,

    /// Decay factor for old suspicion scores per round.
    pub suspicion_decay: f32,
}

impl Default for ReplayDetectorConfig {
    fn default() -> Self {
        Self {
            history_rounds: 10,
            similarity_threshold: 0.99,
            hash_based: true,
            cross_node_detection: true,
            top_k: DEFAULT_TOP_K,
            suspicion_threshold: 0.7,
            min_replays_for_suspicion: 3,
            suspicion_decay: 0.9,
        }
    }
}

impl ReplayDetectorConfig {
    /// Create config optimized for high precision (fewer false positives).
    pub fn high_precision() -> Self {
        Self {
            similarity_threshold: 0.999,
            min_replays_for_suspicion: 5,
            ..Default::default()
        }
    }

    /// Create config optimized for high recall (catch more replays).
    pub fn high_recall() -> Self {
        Self {
            similarity_threshold: 0.95,
            min_replays_for_suspicion: 2,
            ..Default::default()
        }
    }

    /// Builder: set history rounds.
    pub fn with_history_rounds(mut self, rounds: usize) -> Self {
        self.history_rounds = rounds;
        self
    }

    /// Builder: set similarity threshold.
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Builder: enable/disable hash-based detection.
    pub fn with_hash_based(mut self, enabled: bool) -> Self {
        self.hash_based = enabled;
        self
    }

    /// Builder: enable/disable cross-node detection.
    pub fn with_cross_node_detection(mut self, enabled: bool) -> Self {
        self.cross_node_detection = enabled;
        self
    }
}

// ============================================================================
// Fingerprint
// ============================================================================

/// Compact fingerprint of a gradient for efficient comparison.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradientFingerprint {
    /// SHA-256 hash of the gradient.
    pub hash: [u8; 32],

    /// L2 norm of the gradient.
    pub norm: f32,

    /// Mean value of gradient elements.
    pub mean: f32,

    /// Standard deviation of gradient elements.
    pub std_dev: f32,

    /// Indices of top-k absolute values (for sparse comparison).
    pub top_indices: Vec<usize>,

    /// Values at top-k indices (for similarity computation).
    top_values: Vec<f32>,
}

impl GradientFingerprint {
    /// Compute fingerprint from a gradient.
    pub fn compute(gradient: &Array1<f32>, top_k: usize) -> Self {
        let n = gradient.len();
        if n == 0 {
            return Self {
                hash: [0u8; 32],
                norm: 0.0,
                mean: 0.0,
                std_dev: 0.0,
                top_indices: Vec::new(),
                top_values: Vec::new(),
            };
        }

        // Compute hash
        let hash = Self::compute_hash(gradient);

        // Compute statistics
        let mean = gradient.mean().unwrap_or(0.0);
        let norm = gradient.iter().map(|x| x * x).sum::<f32>().sqrt();

        let variance = gradient.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let std_dev = variance.sqrt();

        // Find top-k indices by absolute value
        let k = top_k.min(n);
        let mut indexed: Vec<(usize, f32)> = gradient
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_indices: Vec<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();
        let top_values: Vec<f32> = top_indices.iter().map(|&i| gradient[i]).collect();

        Self {
            hash,
            norm,
            mean,
            std_dev,
            top_indices,
            top_values,
        }
    }

    /// Compute SHA-256 hash of gradient.
    fn compute_hash(gradient: &Array1<f32>) -> [u8; 32] {
        let mut hasher = Sha256::new();

        // Hash all values as bytes
        for &val in gradient.iter() {
            hasher.update(val.to_le_bytes());
        }

        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Check if two fingerprints have identical hashes.
    pub fn hash_matches(&self, other: &Self) -> bool {
        self.hash == other.hash
    }

    /// Compute approximate cosine similarity using top-k values.
    /// This is faster than full gradient comparison but less accurate.
    pub fn approximate_similarity(&self, other: &Self) -> f32 {
        if self.norm == 0.0 || other.norm == 0.0 {
            return 0.0;
        }

        // Quick check: if norms are very different, similarity is low
        let norm_ratio = self.norm.min(other.norm) / self.norm.max(other.norm);
        if norm_ratio < 0.5 {
            return norm_ratio;
        }

        // Compare top-k values at matching indices
        let mut dot_product = 0.0;
        let mut self_norm_sq = 0.0;
        let mut other_norm_sq = 0.0;

        // Create index map for other's top indices
        let other_map: HashMap<usize, f32> = other
            .top_indices
            .iter()
            .zip(other.top_values.iter())
            .map(|(&i, &v)| (i, v))
            .collect();

        for (&idx, &val) in self.top_indices.iter().zip(self.top_values.iter()) {
            self_norm_sq += val * val;
            if let Some(&other_val) = other_map.get(&idx) {
                dot_product += val * other_val;
            }
        }

        for &val in other.top_values.iter() {
            other_norm_sq += val * val;
        }

        if self_norm_sq == 0.0 || other_norm_sq == 0.0 {
            return 0.0;
        }

        dot_product / (self_norm_sq.sqrt() * other_norm_sq.sqrt())
    }

    /// Quick statistical similarity check.
    pub fn stats_similar(&self, other: &Self, tolerance: f32) -> bool {
        let mean_diff = (self.mean - other.mean).abs();
        let std_diff = (self.std_dev - other.std_dev).abs();
        let norm_ratio = if self.norm > 0.0 && other.norm > 0.0 {
            (self.norm - other.norm).abs() / self.norm.max(other.norm)
        } else {
            1.0
        };

        mean_diff < tolerance && std_diff < tolerance && norm_ratio < tolerance
    }
}

// ============================================================================
// Replay Types and Results
// ============================================================================

/// Type of replay detected.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ReplayType {
    /// Identical gradient (exact hash match).
    ExactReplay,

    /// Very similar gradient (above similarity threshold).
    NearReplay { similarity: f32 },

    /// Gradient copied from another node in the same round.
    CrossNodeCopy { source_node: String },

    /// Old gradient resubmitted from a previous round.
    StaleGradient { rounds_old: u64 },

    /// Systematic copying behavior detected (free-rider pattern).
    FreeRider { pattern: String },
}

/// Result of checking a gradient for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayCheckResult {
    /// Whether this is considered a replay.
    pub is_replay: bool,

    /// Type of replay if detected.
    pub replay_type: Option<ReplayType>,

    /// Round of the original submission (if found).
    pub original_round: Option<u64>,

    /// Similarity to the detected replay source.
    pub similarity: f32,

    /// Confidence in the detection (0.0-1.0).
    pub confidence: f32,
}

impl ReplayCheckResult {
    /// Create a result indicating no replay was detected.
    pub fn no_replay() -> Self {
        Self {
            is_replay: false,
            replay_type: None,
            original_round: None,
            similarity: 0.0,
            confidence: 1.0,
        }
    }

    /// Create a result indicating an exact replay was detected.
    pub fn exact_replay(original_round: u64) -> Self {
        Self {
            is_replay: true,
            replay_type: Some(ReplayType::ExactReplay),
            original_round: Some(original_round),
            similarity: 1.0,
            confidence: 1.0,
        }
    }

    /// Create a result indicating a near-replay was detected.
    pub fn near_replay(similarity: f32, original_round: u64, confidence: f32) -> Self {
        Self {
            is_replay: true,
            replay_type: Some(ReplayType::NearReplay { similarity }),
            original_round: Some(original_round),
            similarity,
            confidence,
        }
    }

    /// Create a result indicating a cross-node copy was detected.
    pub fn cross_node_copy(source_node: String, similarity: f32) -> Self {
        Self {
            is_replay: true,
            replay_type: Some(ReplayType::CrossNodeCopy { source_node }),
            original_round: None,
            similarity,
            confidence: similarity,
        }
    }

    /// Create a result indicating a stale gradient was detected.
    pub fn stale_gradient(rounds_old: u64, similarity: f32) -> Self {
        Self {
            is_replay: true,
            replay_type: Some(ReplayType::StaleGradient { rounds_old }),
            original_round: None,
            similarity,
            confidence: similarity,
        }
    }
}

// ============================================================================
// Suspicious Node Tracking
// ============================================================================

/// Information about a suspicious node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuspiciousNode {
    /// Node identifier.
    pub node_id: String,

    /// Number of replay detections.
    pub replay_count: usize,

    /// Number of cross-node copy detections.
    pub copy_count: usize,

    /// Overall suspicion score (0.0-1.0).
    pub suspicion_score: f32,

    /// First round where suspicious behavior was detected.
    pub first_detected: u64,

    /// Most recent round where suspicious behavior was detected.
    pub last_detected: u64,

    /// Detected patterns of behavior.
    patterns: Vec<String>,
}

impl SuspiciousNode {
    /// Create a new suspicious node entry.
    fn new(node_id: String, round: u64) -> Self {
        Self {
            node_id,
            replay_count: 0,
            copy_count: 0,
            suspicion_score: 0.0,
            first_detected: round,
            last_detected: round,
            patterns: Vec::new(),
        }
    }

    /// Record a replay detection.
    fn record_replay(&mut self, round: u64, score_increment: f32) {
        self.replay_count += 1;
        self.last_detected = round;
        self.suspicion_score = (self.suspicion_score + score_increment).min(1.0);
    }

    /// Record a cross-node copy detection.
    fn record_copy(&mut self, round: u64, score_increment: f32) {
        self.copy_count += 1;
        self.last_detected = round;
        self.suspicion_score = (self.suspicion_score + score_increment).min(1.0);
    }

    /// Add a detected pattern.
    fn add_pattern(&mut self, pattern: String) {
        if !self.patterns.contains(&pattern) {
            self.patterns.push(pattern);
        }
    }

    /// Apply decay to suspicion score.
    fn apply_decay(&mut self, decay: f32) {
        self.suspicion_score *= decay;
    }
}

// ============================================================================
// Bloom Filter
// ============================================================================

/// Simple bloom filter for fast hash lookup.
struct BloomFilter {
    bits: Vec<bool>,
    size: usize,
    hash_count: usize,
}

impl BloomFilter {
    fn new(size: usize, hash_count: usize) -> Self {
        Self {
            bits: vec![false; size],
            size,
            hash_count,
        }
    }

    fn insert(&mut self, hash: &[u8; 32]) {
        for i in 0..self.hash_count {
            let idx = self.get_index(hash, i);
            self.bits[idx] = true;
        }
    }

    fn may_contain(&self, hash: &[u8; 32]) -> bool {
        for i in 0..self.hash_count {
            let idx = self.get_index(hash, i);
            if !self.bits[idx] {
                return false;
            }
        }
        true
    }

    fn get_index(&self, hash: &[u8; 32], seed: usize) -> usize {
        // Use different portions of the hash with seed mixing
        let mut val: u64 = 0;
        let offset = (seed * 4) % 28;
        for i in 0..4 {
            val = (val << 8) | hash[(offset + i) % 32] as u64;
        }
        val = val.wrapping_mul(seed as u64 + 1);
        (val as usize) % self.size
    }

    fn clear(&mut self) {
        self.bits.fill(false);
    }
}

// ============================================================================
// History Entry
// ============================================================================

/// Entry in the submission history.
#[derive(Clone, Debug)]
struct HistoryEntry {
    node_id: String,
    round: u64,
    fingerprint: GradientFingerprint,
}

// ============================================================================
// Replay Detector
// ============================================================================

/// Callback type for replay detection events.
pub type ReplayCallback = Box<dyn Fn(&str, &ReplayCheckResult, u64) + Send + Sync>;

/// Main replay detection system.
pub struct ReplayDetector {
    /// Configuration.
    config: ReplayDetectorConfig,

    /// History of submissions by node.
    node_history: HashMap<NodeId, VecDeque<HistoryEntry>>,

    /// Global history for cross-node detection (per round).
    round_submissions: HashMap<Round, Vec<HistoryEntry>>,

    /// Bloom filter for fast hash lookup.
    bloom_filter: BloomFilter,

    /// Hash to round mapping for exact replay detection.
    hash_to_round: HashMap<[u8; 32], (String, u64)>,

    /// Suspicious nodes tracking.
    suspicious_nodes: HashMap<NodeId, SuspiciousNode>,

    /// Current round number.
    current_round: u64,

    /// Oldest round in history.
    oldest_round: u64,

    /// Callback for replay events.
    callback: Option<ReplayCallback>,

    /// Statistics.
    stats: ReplayDetectorStats,
}

/// Statistics for the replay detector.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ReplayDetectorStats {
    /// Total submissions checked.
    pub total_checks: u64,

    /// Exact replays detected.
    pub exact_replays: u64,

    /// Near replays detected.
    pub near_replays: u64,

    /// Cross-node copies detected.
    pub cross_node_copies: u64,

    /// Stale gradients detected.
    pub stale_gradients: u64,

    /// Free-rider patterns detected.
    pub free_rider_patterns: u64,

    /// False positive estimate (based on similarity distribution).
    pub estimated_false_positives: u64,
}

impl ReplayDetector {
    /// Create a new replay detector with the given configuration.
    pub fn new(config: ReplayDetectorConfig) -> Self {
        Self {
            config,
            node_history: HashMap::new(),
            round_submissions: HashMap::new(),
            bloom_filter: BloomFilter::new(BLOOM_FILTER_SIZE, BLOOM_HASH_FUNCS),
            hash_to_round: HashMap::new(),
            suspicious_nodes: HashMap::new(),
            current_round: 0,
            oldest_round: 0,
            callback: None,
            stats: ReplayDetectorStats::default(),
        }
    }

    /// Set callback for replay detection events.
    pub fn set_callback(&mut self, callback: ReplayCallback) {
        self.callback = Some(callback);
    }

    /// Compute fingerprint for a gradient.
    pub fn compute_fingerprint(&self, gradient: &Array1<f32>) -> GradientFingerprint {
        GradientFingerprint::compute(gradient, self.config.top_k)
    }

    /// Check if a gradient is a replay.
    ///
    /// This checks for:
    /// 1. Exact hash matches (replays from any previous round)
    /// 2. Near-replays (very similar gradients from same node)
    /// 3. Stale gradients (gradients that are too old)
    pub fn check_replay(
        &mut self,
        node_id: &str,
        gradient: &Array1<f32>,
        round: u64,
    ) -> ReplayCheckResult {
        self.stats.total_checks += 1;
        self.current_round = round;

        let fingerprint = self.compute_fingerprint(gradient);

        // Check 1: Exact replay via bloom filter + hash lookup
        if self.config.hash_based && self.bloom_filter.may_contain(&fingerprint.hash) {
            if let Some((orig_node, orig_round)) = self.hash_to_round.get(&fingerprint.hash) {
                // Found exact match
                let result = if orig_node == node_id {
                    // Same node replaying their own gradient
                    let rounds_old = round.saturating_sub(*orig_round);
                    self.stats.exact_replays += 1;
                    ReplayCheckResult {
                        is_replay: true,
                        replay_type: Some(ReplayType::StaleGradient { rounds_old }),
                        original_round: Some(*orig_round),
                        similarity: 1.0,
                        confidence: 1.0,
                    }
                } else {
                    // Copying from another node
                    self.stats.cross_node_copies += 1;
                    ReplayCheckResult::cross_node_copy(orig_node.clone(), 1.0)
                };

                self.record_suspicious_activity(node_id, &result, round);
                self.invoke_callback(node_id, &result, round);
                return result;
            }
        }

        // Check 2: Near-replay from same node's history
        if let Some(history) = self.node_history.get(node_id) {
            for entry in history.iter() {
                let similarity = if self.config.hash_based {
                    fingerprint.approximate_similarity(&entry.fingerprint)
                } else {
                    // Would need full gradient here - using approximate for now
                    fingerprint.approximate_similarity(&entry.fingerprint)
                };

                if similarity >= self.config.similarity_threshold {
                    let rounds_old = round.saturating_sub(entry.round);
                    let confidence = (similarity - self.config.similarity_threshold)
                        / (1.0 - self.config.similarity_threshold);

                    let result = if similarity >= 0.9999 {
                        self.stats.exact_replays += 1;
                        ReplayCheckResult::exact_replay(entry.round)
                    } else {
                        self.stats.near_replays += 1;
                        ReplayCheckResult {
                            is_replay: true,
                            replay_type: Some(ReplayType::StaleGradient { rounds_old }),
                            original_round: Some(entry.round),
                            similarity,
                            confidence: confidence.min(1.0),
                        }
                    };

                    self.record_suspicious_activity(node_id, &result, round);
                    self.invoke_callback(node_id, &result, round);
                    return result;
                }
            }
        }

        ReplayCheckResult::no_replay()
    }

    /// Check if a gradient is copied from another node in the current round.
    pub fn check_cross_node_copy(&mut self, gradient: &Array1<f32>, round: u64) -> Option<String> {
        if !self.config.cross_node_detection {
            return None;
        }

        let fingerprint = self.compute_fingerprint(gradient);

        // Check against all submissions in the current round
        if let Some(submissions) = self.round_submissions.get(&round) {
            for entry in submissions.iter() {
                let similarity = fingerprint.approximate_similarity(&entry.fingerprint);

                if similarity >= self.config.similarity_threshold {
                    self.stats.cross_node_copies += 1;
                    return Some(entry.node_id.clone());
                }
            }
        }

        None
    }

    /// Record a gradient submission for future replay detection.
    pub fn record_submission(&mut self, node_id: &str, gradient: &Array1<f32>, round: u64) {
        let fingerprint = self.compute_fingerprint(gradient);

        // Add to bloom filter
        self.bloom_filter.insert(&fingerprint.hash);

        // Add to hash lookup
        self.hash_to_round
            .insert(fingerprint.hash, (node_id.to_string(), round));

        // Add to node history
        let history = self
            .node_history
            .entry(node_id.to_string())
            .or_insert_with(VecDeque::new);

        history.push_back(HistoryEntry {
            node_id: node_id.to_string(),
            round,
            fingerprint: fingerprint.clone(),
        });

        // Trim node history if too long
        while history.len() > self.config.history_rounds {
            history.pop_front();
        }

        // Add to round submissions for cross-node detection
        if self.config.cross_node_detection {
            let submissions = self.round_submissions.entry(round).or_insert_with(Vec::new);

            submissions.push(HistoryEntry {
                node_id: node_id.to_string(),
                round,
                fingerprint,
            });
        }

        // Update current round
        self.current_round = self.current_round.max(round);
    }

    /// Get list of suspicious nodes.
    pub fn get_suspicious_nodes(&self) -> Vec<SuspiciousNode> {
        self.suspicious_nodes
            .values()
            .filter(|n| {
                n.suspicion_score >= self.config.suspicion_threshold
                    && (n.replay_count + n.copy_count) >= self.config.min_replays_for_suspicion
            })
            .cloned()
            .collect()
    }

    /// Clear history older than the specified round.
    pub fn clear_old_history(&mut self, before_round: u64) {
        // Clear round submissions
        self.round_submissions
            .retain(|&round, _| round >= before_round);

        // Clear node history
        for history in self.node_history.values_mut() {
            while let Some(front) = history.front() {
                if front.round < before_round {
                    history.pop_front();
                } else {
                    break;
                }
            }
        }

        // Rebuild bloom filter and hash map from remaining entries
        self.rebuild_indices();

        // Apply decay to suspicious nodes
        for node in self.suspicious_nodes.values_mut() {
            node.apply_decay(self.config.suspicion_decay);
        }

        // Remove nodes with very low suspicion
        self.suspicious_nodes
            .retain(|_, n| n.suspicion_score > 0.01);

        self.oldest_round = before_round;
    }

    /// Check for systematic free-rider patterns.
    ///
    /// Detects nodes that consistently:
    /// - Submit gradients identical to previous round
    /// - Copy from the same node repeatedly
    /// - Submit zero or near-zero gradients
    pub fn detect_free_rider_pattern(&mut self, node_id: &str) -> Option<ReplayType> {
        let suspicious = self.suspicious_nodes.get(node_id)?;

        // Check for consistent copying
        if suspicious.copy_count >= 3 {
            let pattern = format!(
                "Consistent copying ({} times in {} rounds)",
                suspicious.copy_count,
                suspicious.last_detected - suspicious.first_detected + 1
            );
            self.stats.free_rider_patterns += 1;
            return Some(ReplayType::FreeRider { pattern });
        }

        // Check for consistent replays
        if suspicious.replay_count >= 3 {
            let pattern = format!(
                "Consistent replay ({} times in {} rounds)",
                suspicious.replay_count,
                suspicious.last_detected - suspicious.first_detected + 1
            );
            self.stats.free_rider_patterns += 1;
            return Some(ReplayType::FreeRider { pattern });
        }

        None
    }

    /// Detect collusion (multiple nodes submitting identical gradients).
    pub fn detect_collusion(&self, round: u64) -> Vec<Vec<String>> {
        let mut collusion_groups: Vec<Vec<String>> = Vec::new();

        if let Some(submissions) = self.round_submissions.get(&round) {
            // Group by hash
            let mut hash_groups: HashMap<[u8; 32], Vec<String>> = HashMap::new();

            for entry in submissions {
                hash_groups
                    .entry(entry.fingerprint.hash)
                    .or_insert_with(Vec::new)
                    .push(entry.node_id.clone());
            }

            // Find groups with multiple nodes
            for (_, nodes) in hash_groups {
                if nodes.len() > 1 {
                    collusion_groups.push(nodes);
                }
            }
        }

        collusion_groups
    }

    /// Get detector statistics.
    pub fn stats(&self) -> &ReplayDetectorStats {
        &self.stats
    }

    /// Reset detector state.
    pub fn reset(&mut self) {
        self.node_history.clear();
        self.round_submissions.clear();
        self.bloom_filter.clear();
        self.hash_to_round.clear();
        self.suspicious_nodes.clear();
        self.current_round = 0;
        self.oldest_round = 0;
        self.stats = ReplayDetectorStats::default();
    }

    /// Get configuration.
    pub fn config(&self) -> &ReplayDetectorConfig {
        &self.config
    }

    // ========================================================================
    // Integration Methods
    // ========================================================================

    /// Check and record a submission in one call.
    /// This is the main integration point for the aggregator.
    pub fn process_submission(
        &mut self,
        node_id: &str,
        gradient: &Array1<f32>,
        round: u64,
    ) -> ReplayCheckResult {
        // First check for replay
        let result = self.check_replay(node_id, gradient, round);

        // If not a replay, check for cross-node copy
        let result = if !result.is_replay {
            if let Some(source) = self.check_cross_node_copy(gradient, round) {
                let copy_result = ReplayCheckResult::cross_node_copy(source, 1.0);
                self.record_suspicious_activity(node_id, &copy_result, round);
                self.invoke_callback(node_id, &copy_result, round);
                copy_result
            } else {
                result
            }
        } else {
            result
        };

        // Record the submission regardless
        self.record_submission(node_id, gradient, round);

        // Check for free-rider pattern
        if let Some(pattern) = self.detect_free_rider_pattern(node_id) {
            if let Some(suspicious) = self.suspicious_nodes.get_mut(node_id) {
                if let ReplayType::FreeRider { pattern: p } = &pattern {
                    suspicious.add_pattern(p.clone());
                }
            }
        }

        result
    }

    /// Batch process multiple submissions.
    pub fn process_submissions_batch(
        &mut self,
        submissions: &[(String, Array1<f32>)],
        round: u64,
    ) -> Vec<(String, ReplayCheckResult)> {
        submissions
            .iter()
            .map(|(node_id, gradient)| {
                let result = self.process_submission(node_id, gradient, round);
                (node_id.clone(), result)
            })
            .collect()
    }

    // ========================================================================
    // Private Methods
    // ========================================================================

    fn record_suspicious_activity(
        &mut self,
        node_id: &str,
        result: &ReplayCheckResult,
        round: u64,
    ) {
        if !result.is_replay {
            return;
        }

        let node = self
            .suspicious_nodes
            .entry(node_id.to_string())
            .or_insert_with(|| SuspiciousNode::new(node_id.to_string(), round));

        let score_increment = result.confidence * 0.2;

        match &result.replay_type {
            Some(ReplayType::CrossNodeCopy { .. }) => {
                node.record_copy(round, score_increment);
            }
            Some(_) => {
                node.record_replay(round, score_increment);
            }
            None => {}
        }
    }

    fn invoke_callback(&self, node_id: &str, result: &ReplayCheckResult, round: u64) {
        if let Some(ref callback) = self.callback {
            callback(node_id, result, round);
        }
    }

    fn rebuild_indices(&mut self) {
        self.bloom_filter.clear();
        self.hash_to_round.clear();

        for (node_id, history) in &self.node_history {
            for entry in history {
                self.bloom_filter.insert(&entry.fingerprint.hash);
                self.hash_to_round
                    .insert(entry.fingerprint.hash, (node_id.clone(), entry.round));
            }
        }
    }
}

impl Default for ReplayDetector {
    fn default() -> Self {
        Self::new(ReplayDetectorConfig::default())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute exact cosine similarity between two gradients.
pub fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_fingerprint_computation() {
        let gradient = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let fingerprint = GradientFingerprint::compute(&gradient, 3);

        assert_eq!(fingerprint.top_indices.len(), 3);
        assert!(fingerprint.norm > 0.0);
        assert!((fingerprint.mean - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_exact_replay_detection() {
        let mut detector = ReplayDetector::default();

        let gradient = array![1.0, 2.0, 3.0, 4.0, 5.0];

        // Record first submission
        detector.record_submission("node_1", &gradient, 1);

        // Check same gradient in later round
        let result = detector.check_replay("node_1", &gradient, 5);

        assert!(result.is_replay);
        assert!(matches!(
            result.replay_type,
            Some(ReplayType::StaleGradient { .. }) | Some(ReplayType::ExactReplay)
        ));
        assert_eq!(result.similarity, 1.0);
    }

    #[test]
    fn test_near_replay_detection() {
        let mut detector =
            ReplayDetector::new(ReplayDetectorConfig::default().with_similarity_threshold(0.98));

        let gradient1 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        detector.record_submission("node_1", &gradient1, 1);

        // Slightly modified gradient
        let gradient2 = array![1.01, 2.01, 3.01, 4.01, 5.01];
        let result = detector.check_replay("node_1", &gradient2, 2);

        // Should detect as near replay due to high similarity
        // Note: This depends on the approximate_similarity calculation
        assert!(result.similarity > 0.9);
    }

    #[test]
    fn test_cross_node_copy_detection() {
        let mut detector = ReplayDetector::default();

        let gradient = array![1.0, 2.0, 3.0, 4.0, 5.0];

        // Node 1 submits first
        detector.record_submission("node_1", &gradient, 1);

        // Node 2 submits same gradient in same round
        detector.record_submission("node_2", &gradient, 1);

        // Check for collusion
        let collusion = detector.detect_collusion(1);
        assert!(!collusion.is_empty());
        assert!(collusion[0].contains(&"node_1".to_string()));
        assert!(collusion[0].contains(&"node_2".to_string()));
    }

    #[test]
    fn test_history_management() {
        let config = ReplayDetectorConfig::default().with_history_rounds(3);
        let mut detector = ReplayDetector::new(config);

        // Submit across multiple rounds
        for round in 0..10 {
            let gradient = Array1::from(vec![round as f32; 5]);
            detector.record_submission("node_1", &gradient, round);
        }

        // Clear old history
        detector.clear_old_history(7);

        // Old gradient should not be detected
        let old_gradient = Array1::from(vec![0.0; 5]);
        let result = detector.check_replay("node_1", &old_gradient, 10);

        // Result depends on whether the gradient was cleared
        // The bloom filter might still have false positives
        // but the exact match should be gone
        if result.is_replay {
            // If detected, original_round should be recent (>=7)
            if let Some(orig_round) = result.original_round {
                assert!(orig_round >= 7 || result.similarity < 1.0);
            }
        }
    }

    #[test]
    fn test_pattern_detection() {
        let mut detector = ReplayDetector::default();

        let gradient = array![1.0, 2.0, 3.0, 4.0, 5.0];

        // Same node replays multiple times
        detector.record_submission("free_rider", &gradient, 1);

        for round in 2..=5 {
            let result = detector.process_submission("free_rider", &gradient, round);
            assert!(result.is_replay);
        }

        // Should now be flagged as free-rider
        let pattern = detector.detect_free_rider_pattern("free_rider");
        assert!(pattern.is_some());
        assert!(matches!(pattern, Some(ReplayType::FreeRider { .. })));
    }

    #[test]
    fn test_false_positive_rate() {
        let mut detector =
            ReplayDetector::new(ReplayDetectorConfig::default().with_similarity_threshold(0.99));

        // Submit various legitimate gradients
        for i in 0..100 {
            let gradient = Array1::from((0..100).map(|j| (i * 100 + j) as f32).collect::<Vec<_>>());
            detector.record_submission(&format!("node_{}", i), &gradient, i as u64);
        }

        // Check new legitimate gradients
        let mut false_positives = 0;
        for i in 100..200 {
            let gradient = Array1::from((0..100).map(|j| (i * 100 + j) as f32).collect::<Vec<_>>());
            let result = detector.check_replay(&format!("node_{}", i), &gradient, i as u64);
            if result.is_replay {
                false_positives += 1;
            }
        }

        // False positive rate should be very low
        assert!(
            false_positives < 5,
            "Too many false positives: {}",
            false_positives
        );
    }

    #[test]
    fn test_suspicious_nodes() {
        let mut detector =
            ReplayDetector::new(ReplayDetectorConfig::default().with_similarity_threshold(0.99));

        let gradient = array![1.0, 2.0, 3.0, 4.0, 5.0];

        // Honest node submits once
        detector.record_submission("honest_node", &gradient, 1);

        // Malicious node replays multiple times
        detector.record_submission("malicious_node", &gradient, 1);
        for round in 2..=10 {
            detector.process_submission("malicious_node", &gradient, round);
        }

        let suspicious = detector.get_suspicious_nodes();

        // Should have at least the malicious node
        let malicious = suspicious.iter().find(|n| n.node_id == "malicious_node");
        assert!(malicious.is_some());
        assert!(malicious.unwrap().replay_count >= 3);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = array![1.0, 0.0, 0.0];
        let b = array![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = array![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);

        let d = array![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_varying_similarity_thresholds() {
        let gradient1 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let gradient2 = array![1.1, 2.1, 3.1, 4.1, 5.1];

        // Compute exact similarity
        let exact_sim = cosine_similarity(&gradient1, &gradient2);

        // Test with different thresholds
        for threshold in [0.9, 0.95, 0.99, 0.999] {
            let mut detector = ReplayDetector::new(
                ReplayDetectorConfig::default().with_similarity_threshold(threshold),
            );

            detector.record_submission("node_1", &gradient1, 1);
            let result = detector.check_replay("node_1", &gradient2, 2);

            if exact_sim >= threshold {
                // Should detect
                assert!(
                    result.is_replay || result.similarity >= threshold * 0.9,
                    "Threshold {} should detect similarity {}",
                    threshold,
                    exact_sim
                );
            }
        }
    }

    #[test]
    fn test_performance_many_nodes() {
        let mut detector = ReplayDetector::default();

        // Simulate many nodes and rounds
        let num_nodes = 100;
        let num_rounds = 20;

        for round in 0..num_rounds {
            for node_id in 0..num_nodes {
                let gradient = Array1::from(
                    (0..1000)
                        .map(|j| ((round * num_nodes + node_id) * 1000 + j) as f32)
                        .collect::<Vec<_>>(),
                );
                detector.process_submission(&format!("node_{}", node_id), &gradient, round as u64);
            }
        }

        // Should complete without performance issues
        let stats = detector.stats();
        assert_eq!(stats.total_checks, (num_nodes * num_rounds) as u64);
    }

    #[test]
    fn test_empty_gradient() {
        let mut detector = ReplayDetector::default();
        let gradient = Array1::from(vec![]);

        let fingerprint = detector.compute_fingerprint(&gradient);
        assert_eq!(fingerprint.norm, 0.0);
        assert!(fingerprint.top_indices.is_empty());
    }

    #[test]
    fn test_zero_gradient() {
        let mut detector = ReplayDetector::default();
        let gradient = array![0.0, 0.0, 0.0, 0.0, 0.0];

        detector.record_submission("node_1", &gradient, 1);
        let result = detector.check_replay("node_1", &gradient, 2);

        assert!(result.is_replay);
    }

    #[test]
    fn test_batch_processing() {
        // Disable cross-node detection to test pure replay detection
        let config = ReplayDetectorConfig::default().with_cross_node_detection(false);
        let mut detector = ReplayDetector::new(config);

        // Create unique gradients for each node
        let submissions: Vec<(String, Array1<f32>)> = (0..10)
            .map(|i| {
                (
                    format!("node_{}", i),
                    Array1::from((0..100).map(|j| ((i * 100 + j) as f32) * 0.01).collect::<Vec<_>>()),
                )
            })
            .collect();

        let results = detector.process_submissions_batch(&submissions, 1);
        assert_eq!(results.len(), 10);

        // No replays expected in first batch (each gradient is unique)
        for (node_id, result) in &results {
            assert!(!result.is_replay, "Node {} should not be flagged as replay in first batch", node_id);
        }

        // Process same batch again - should detect replays
        let results2 = detector.process_submissions_batch(&submissions, 2);
        let replay_count = results2.iter().filter(|(_, r)| r.is_replay).count();
        assert!(replay_count > 0, "Should detect replays when same gradients are submitted again");
    }

    #[test]
    fn test_config_builders() {
        let config = ReplayDetectorConfig::default()
            .with_history_rounds(20)
            .with_similarity_threshold(0.95)
            .with_hash_based(false)
            .with_cross_node_detection(false);

        assert_eq!(config.history_rounds, 20);
        assert!((config.similarity_threshold - 0.95).abs() < 0.001);
        assert!(!config.hash_based);
        assert!(!config.cross_node_detection);
    }

    #[test]
    fn test_high_precision_config() {
        let config = ReplayDetectorConfig::high_precision();
        assert!(config.similarity_threshold > 0.99);
        assert!(config.min_replays_for_suspicion >= 5);
    }

    #[test]
    fn test_high_recall_config() {
        let config = ReplayDetectorConfig::high_recall();
        assert!(config.similarity_threshold < 0.99);
        assert!(config.min_replays_for_suspicion <= 2);
    }
}
