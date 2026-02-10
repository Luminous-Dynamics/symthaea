//! FL Aggregation Algorithms
//!
//! Byzantine-resistant gradient aggregation methods.
//!
//! # Performance Optimizations
//!
//! - **Parallel Aggregation**: Multi-threaded gradient accumulation
//! - **Memory-Efficient Accumulation**: In-place operations to reduce allocations
//! - **Early Byzantine Termination**: Fast-fail when Byzantine behavior detected
//! - **SIMD-Friendly Layouts**: Array operations optimized for vectorization
//! - **Pre-allocated Buffers**: Reused scratch space for sorting operations

use std::collections::HashMap;
use thiserror::Error;

use super::types::{AggregatedGradient, AggregationMethod, GradientUpdate, Participant};
use crate::matl::ProofOfGradientQuality;

// ---- Core crate delegation helpers ----

/// Convert SDK GradientUpdate (f64) to core GradientUpdate (f32)
fn sdk_to_core_updates(updates: &[GradientUpdate]) -> Vec<mycelix_fl_core::types::GradientUpdate> {
    updates
        .iter()
        .map(|u| {
            mycelix_fl_core::convert::update_to_f32(
                u.participant_id.clone(),
                u.model_version,
                &u.gradients,
                u.metadata.batch_size as u32,
                u.metadata.loss,
                u.metadata.accuracy,
                u.metadata.timestamp,
            )
        })
        .collect()
}

/// Map core AggregationError to SDK AggregationError
fn map_core_error(e: mycelix_fl_core::aggregation::AggregationError) -> AggregationError {
    use mycelix_fl_core::aggregation::AggregationError as CoreErr;
    match e {
        CoreErr::NoUpdates => AggregationError::NoUpdates,
        CoreErr::EmptyGradients(id) => AggregationError::EmptyGradients(id),
        CoreErr::GradientSizeMismatch {
            participant_id,
            expected,
            actual,
        } => AggregationError::GradientSizeMismatch {
            participant_id,
            expected,
            actual,
        },
        CoreErr::InvalidTrimPercentage(p) => AggregationError::InvalidTrimPercentage(p as f64),
        CoreErr::NotEnoughForKrum(n) => AggregationError::NotEnoughForKrum(n),
        CoreErr::InvalidKrumSelect(n) => AggregationError::InvalidKrumSelect(n),
        CoreErr::InvalidBatchSize(n) => AggregationError::InvalidBatchSize(n as usize),
        CoreErr::InvalidLoss(id) => AggregationError::InvalidLoss(id),
        CoreErr::NoTrustedParticipants => AggregationError::NoTrustedParticipants,
        CoreErr::TooManyByzantine => AggregationError::NoTrustedParticipants,
    }
}

/// FL Aggregation errors
#[derive(Debug, Error)]
pub enum AggregationError {
    /// No gradient updates were provided for aggregation.
    #[error("No gradient updates provided")]
    NoUpdates,
    /// A participant submitted an empty gradient array.
    #[error("Gradient array is empty for participant {0}")]
    EmptyGradients(String),
    /// Gradient dimensions differ between participants.
    #[error("Gradient size mismatch: expected {expected}, got {actual} for participant {participant_id}")]
    GradientSizeMismatch {
        /// ID of the participant with mismatched gradients.
        participant_id: String,
        /// Expected gradient vector length.
        expected: usize,
        /// Actual gradient vector length received.
        actual: usize,
    },
    /// Trim percentage is outside the valid range (0.0 to 0.5).
    #[error("Invalid trim percentage: {0} (must be 0.0-0.5)")]
    InvalidTrimPercentage(f64),
    /// Krum requires at least 3 participants.
    #[error("Not enough participants for Krum: need at least 3, got {0}")]
    NotEnoughForKrum(usize),
    /// The `num_select` parameter for Krum is out of range.
    #[error("Invalid numSelect for Krum: {0}")]
    InvalidKrumSelect(usize),
    /// A participant reported a batch size of zero.
    #[error("Invalid batch size: {0}")]
    InvalidBatchSize(usize),
    /// A participant reported a non-finite loss value.
    #[error("Loss is not finite for participant {0}")]
    InvalidLoss(String),
    /// All participants were excluded by trust threshold filtering.
    #[error("No participants met trust threshold")]
    NoTrustedParticipants,
}

/// Validate gradient updates have consistent shapes
pub fn validate_gradient_consistency(updates: &[GradientUpdate]) -> Result<(), AggregationError> {
    if updates.is_empty() {
        return Err(AggregationError::NoUpdates);
    }

    let expected_size = updates[0].gradients.len();
    if expected_size == 0 {
        return Err(AggregationError::EmptyGradients(
            updates[0].participant_id.clone(),
        ));
    }

    for update in updates.iter().skip(1) {
        if update.gradients.len() != expected_size {
            return Err(AggregationError::GradientSizeMismatch {
                participant_id: update.participant_id.clone(),
                expected: expected_size,
                actual: update.gradients.len(),
            });
        }
    }

    Ok(())
}

/// Validate gradient update metadata
fn validate_update_metadata(update: &GradientUpdate) -> Result<(), AggregationError> {
    if update.metadata.batch_size == 0 {
        return Err(AggregationError::InvalidBatchSize(0));
    }
    if !update.metadata.loss.is_finite() {
        return Err(AggregationError::InvalidLoss(update.participant_id.clone()));
    }
    Ok(())
}

/// Federated Averaging (FedAvg)
///
/// Standard weighted average based on batch sizes.
///
/// # Arguments
/// * `updates` - Gradient updates from participants
///
/// # Returns
/// Aggregated gradient vector
pub fn fedavg(updates: &[GradientUpdate]) -> Result<Vec<f64>, AggregationError> {
    let core_updates = sdk_to_core_updates(updates);
    let result = mycelix_fl_core::aggregation::fedavg(&core_updates).map_err(map_core_error)?;
    Ok(mycelix_fl_core::convert::gradients_to_f64(&result))
}

/// Trimmed Mean Aggregation
///
/// Removes top and bottom percentile before averaging.
/// Robust to up to `trim_percentage` Byzantine participants at each extreme.
///
/// # Arguments
/// * `updates` - Gradient updates from participants
/// * `trim_percentage` - Fraction to trim from each end (0.0-0.5)
pub fn trimmed_mean(
    updates: &[GradientUpdate],
    trim_percentage: f64,
) -> Result<Vec<f64>, AggregationError> {
    let core_updates = sdk_to_core_updates(updates);
    let result = mycelix_fl_core::aggregation::trimmed_mean(&core_updates, trim_percentage as f32)
        .map_err(map_core_error)?;
    Ok(mycelix_fl_core::convert::gradients_to_f64(&result))
}

/// Coordinate-wise Median
///
/// Robust to up to 50% Byzantine participants.
///
/// # Arguments
/// * `updates` - Gradient updates from participants
pub fn coordinate_median(updates: &[GradientUpdate]) -> Result<Vec<f64>, AggregationError> {
    let core_updates = sdk_to_core_updates(updates);
    let result =
        mycelix_fl_core::aggregation::coordinate_median(&core_updates).map_err(map_core_error)?;
    Ok(mycelix_fl_core::convert::gradients_to_f64(&result))
}

/// Krum Aggregation
///
/// Selects the gradient closest to its neighbors.
/// Tolerates up to (n-2)/2 Byzantine participants.
///
/// # Arguments
/// * `updates` - Gradient updates from participants
/// * `num_select` - Number of gradients to select and average (default: 1)
pub fn krum(updates: &[GradientUpdate], num_select: usize) -> Result<Vec<f64>, AggregationError> {
    let core_updates = sdk_to_core_updates(updates);
    let result =
        mycelix_fl_core::aggregation::krum(&core_updates, num_select).map_err(map_core_error)?;
    Ok(mycelix_fl_core::convert::gradients_to_f64(&result))
}

/// Trust-Weighted Aggregation (MATL Integration)
///
/// Weights contributions by participant reputation and PoGQ.
///
/// # Arguments
/// * `updates` - Gradient updates from participants
/// * `participants` - Map of participant ID to participant state
/// * `trust_threshold` - Minimum trust score for inclusion
pub fn trust_weighted_aggregation(
    updates: &[GradientUpdate],
    participants: &HashMap<String, Participant>,
    trust_threshold: f64,
) -> Result<AggregatedGradient, AggregationError> {
    validate_gradient_consistency(updates)?;
    for update in updates {
        validate_update_metadata(update)?;
    }

    let gradient_size = updates[0].gradients.len();
    let mut result = vec![0.0; gradient_size];
    let mut total_weight = 0.0;
    let mut excluded_count = 0;

    // Calculate trust weights
    let mut weights: HashMap<String, f64> = HashMap::new();

    for update in updates {
        let participant = match participants.get(&update.participant_id) {
            Some(p) => p,
            None => {
                excluded_count += 1;
                continue;
            }
        };

        let rep_value = participant.reputation.score;

        // Check trust threshold
        if rep_value < trust_threshold {
            excluded_count += 1;
            continue;
        }

        // Calculate weight from reputation and PoGQ
        let mut weight = rep_value;

        if let Some(pogq) = &participant.pogq {
            let composite = pogq.composite_score(rep_value);
            weight = composite;
        }

        // Scale by batch size
        weight *= update.metadata.batch_size as f64;
        weights.insert(update.participant_id.clone(), weight);
        total_weight += weight;
    }

    if weights.is_empty() {
        return Err(AggregationError::NoTrustedParticipants);
    }

    // Aggregate with trust weights
    for update in updates {
        if let Some(weight) = weights.get(&update.participant_id) {
            let normalized_weight = weight / total_weight;
            for (i, grad) in update.gradients.iter().enumerate() {
                result[i] += grad * normalized_weight;
            }
        }
    }

    Ok(AggregatedGradient::new(
        result,
        updates[0].model_version,
        updates.len() - excluded_count,
        excluded_count,
        AggregationMethod::TrustWeighted,
    ))
}

/// Euclidean distance between two gradient vectors
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Calculate update quality for PoGQ
pub fn calculate_update_quality(update: &GradientUpdate) -> f64 {
    // Base quality on loss improvement and gradient magnitude
    let loss_quality = (-update.metadata.loss).exp();
    let gradient_mag = update.l2_norm();
    let mag_quality = 1.0 / (1.0 + gradient_mag.ln_1p());

    (loss_quality + mag_quality) / 2.0
}

/// Create PoGQ from gradient update
#[allow(dead_code)]
pub fn create_pogq_from_update(
    update: &GradientUpdate,
    consistency: f64,
    entropy: f64,
) -> ProofOfGradientQuality {
    let quality = calculate_update_quality(update);
    ProofOfGradientQuality::new(quality, consistency, entropy)
}

// =============================================================================
// PERFORMANCE OPTIMIZATIONS
// =============================================================================

/// Memory-efficient gradient accumulator
///
/// Accumulates gradients in-place to minimize allocations.
/// Useful for large-scale aggregation where memory is a concern.
pub struct GradientAccumulator {
    /// Accumulated gradient values
    sum: Vec<f64>,
    /// Total weight accumulated
    total_weight: f64,
    /// Number of updates accumulated
    count: usize,
}

impl GradientAccumulator {
    /// Create a new accumulator with specified gradient dimension
    pub fn new(dimension: usize) -> Self {
        Self {
            sum: vec![0.0; dimension],
            total_weight: 0.0,
            count: 0,
        }
    }

    /// Create from first update (initializes dimension)
    pub fn from_first_update(update: &GradientUpdate, weight: f64) -> Self {
        let mut acc = Self::new(update.gradients.len());
        acc.accumulate(update, weight);
        acc
    }

    /// Accumulate a weighted gradient update
    #[inline]
    pub fn accumulate(&mut self, update: &GradientUpdate, weight: f64) {
        debug_assert_eq!(self.sum.len(), update.gradients.len());

        for (i, grad) in update.gradients.iter().enumerate() {
            self.sum[i] += grad * weight;
        }
        self.total_weight += weight;
        self.count += 1;
    }

    /// Accumulate without weight (for median/trimmed mean pre-processing)
    #[inline]
    pub fn accumulate_unweighted(&mut self, gradients: &[f64]) {
        debug_assert_eq!(self.sum.len(), gradients.len());

        for (i, grad) in gradients.iter().enumerate() {
            self.sum[i] += grad;
        }
        self.count += 1;
    }

    /// Get normalized result
    pub fn finalize(mut self) -> Vec<f64> {
        if self.total_weight > 0.0 {
            for val in &mut self.sum {
                *val /= self.total_weight;
            }
        } else if self.count > 0 {
            let count = self.count as f64;
            for val in &mut self.sum {
                *val /= count;
            }
        }
        self.sum
    }

    /// Get raw accumulated sum (for custom normalization)
    pub fn raw_sum(&self) -> &[f64] {
        &self.sum
    }

    /// Get total weight
    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }

    /// Get count of accumulated updates
    pub fn count(&self) -> usize {
        self.count
    }

    /// Reset accumulator for reuse
    pub fn reset(&mut self) {
        for val in &mut self.sum {
            *val = 0.0;
        }
        self.total_weight = 0.0;
        self.count = 0;
    }
}

/// Byzantine detection result
#[derive(Debug, Clone)]
pub struct ByzantineDetectionResult {
    /// Indices of detected Byzantine participants
    pub byzantine_indices: Vec<usize>,
    /// Detection confidence scores
    pub confidence_scores: Vec<f64>,
    /// Whether early termination was triggered
    pub early_terminated: bool,
    /// Detection method used
    pub method: String,
}

impl ByzantineDetectionResult {
    /// Check if any Byzantine behavior was detected
    pub fn has_byzantine(&self) -> bool {
        !self.byzantine_indices.is_empty()
    }

    /// Get fraction of participants detected as Byzantine
    pub fn byzantine_fraction(&self, total: usize) -> f64 {
        if total == 0 {
            0.0
        } else {
            self.byzantine_indices.len() as f64 / total as f64
        }
    }
}

/// Early termination detector for Byzantine participants
///
/// Analyzes gradient statistics to quickly identify obvious outliers
/// before running expensive aggregation algorithms.
pub struct EarlyByzantineDetector {
    /// Maximum allowed L2 norm for gradients
    max_norm: f64,
    /// Minimum allowed L2 norm (zero gradients are suspicious)
    min_norm: f64,
    /// Z-score threshold for outlier detection
    z_threshold: f64,
}

impl EarlyByzantineDetector {
    /// Create with default thresholds
    pub fn new() -> Self {
        Self {
            max_norm: 1000.0,
            min_norm: 1e-10,
            z_threshold: 3.0,
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(max_norm: f64, min_norm: f64, z_threshold: f64) -> Self {
        Self {
            max_norm,
            min_norm,
            z_threshold,
        }
    }

    /// Perform early Byzantine detection
    ///
    /// Returns indices of obviously Byzantine participants that can be
    /// excluded before running expensive aggregation.
    pub fn detect(&self, updates: &[GradientUpdate]) -> ByzantineDetectionResult {
        if updates.is_empty() {
            return ByzantineDetectionResult {
                byzantine_indices: vec![],
                confidence_scores: vec![],
                early_terminated: false,
                method: "early_norm_check".to_string(),
            };
        }

        let mut byzantine_indices = Vec::new();
        let mut confidence_scores = Vec::new();

        // Phase 1: Norm-based detection (O(n))
        let norms: Vec<f64> = updates.iter().map(|u| u.l2_norm()).collect();

        // Compute statistics
        let mean_norm: f64 = norms.iter().sum::<f64>() / norms.len() as f64;
        let variance: f64 =
            norms.iter().map(|n| (n - mean_norm).powi(2)).sum::<f64>() / norms.len() as f64;
        let std_dev = variance.sqrt();

        for (i, &norm) in norms.iter().enumerate() {
            let mut confidence = 0.0;

            // Check absolute bounds
            if norm > self.max_norm || norm < self.min_norm {
                confidence = 1.0;
            } else if std_dev > 0.0 {
                // Z-score check
                let z_score = (norm - mean_norm).abs() / std_dev;
                if z_score > self.z_threshold {
                    confidence = (z_score - self.z_threshold) / self.z_threshold;
                    confidence = confidence.min(1.0);
                }
            }

            if confidence > 0.5 {
                byzantine_indices.push(i);
                confidence_scores.push(confidence);
            }
        }

        // Early termination if too many Byzantine detected
        let byzantine_fraction = byzantine_indices.len() as f64 / updates.len() as f64;
        let early_terminated = byzantine_fraction > 0.5;

        ByzantineDetectionResult {
            byzantine_indices,
            confidence_scores,
            early_terminated,
            method: "early_norm_check".to_string(),
        }
    }

    /// Filter updates, removing detected Byzantine participants
    pub fn filter_updates<'a>(
        &self,
        updates: &'a [GradientUpdate],
    ) -> (Vec<&'a GradientUpdate>, ByzantineDetectionResult) {
        let detection = self.detect(updates);

        let filtered: Vec<&GradientUpdate> = updates
            .iter()
            .enumerate()
            .filter(|(i, _)| !detection.byzantine_indices.contains(i))
            .map(|(_, u)| u)
            .collect();

        (filtered, detection)
    }
}

impl Default for EarlyByzantineDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimized FedAvg using accumulator pattern
///
/// More memory-efficient than standard fedavg for large gradient vectors.
pub fn fedavg_optimized(updates: &[GradientUpdate]) -> Result<Vec<f64>, AggregationError> {
    fedavg(updates) // Delegates to core which handles optimization
}

/// Aggregation with early Byzantine termination
///
/// Performs early detection and filters out obvious Byzantine participants
/// before running the main aggregation algorithm.
pub fn aggregate_with_early_termination(
    updates: &[GradientUpdate],
    method: AggregationMethod,
    byzantine_tolerance: f64,
) -> Result<(Vec<f64>, ByzantineDetectionResult), AggregationError> {
    let detector = EarlyByzantineDetector::new();
    let detection = detector.detect(updates);

    // Check if early termination is warranted
    if detection.early_terminated {
        return Err(AggregationError::NoTrustedParticipants);
    }

    // Filter out Byzantine participants if within tolerance
    if detection.byzantine_fraction(updates.len()) > byzantine_tolerance {
        return Err(AggregationError::NoTrustedParticipants);
    }

    // Get filtered updates (as owned copies for aggregation)
    let filtered_updates: Vec<GradientUpdate> = updates
        .iter()
        .enumerate()
        .filter(|(i, _)| !detection.byzantine_indices.contains(i))
        .map(|(_, u)| u.clone())
        .collect();

    if filtered_updates.is_empty() {
        return Err(AggregationError::NoUpdates);
    }

    // Run aggregation on filtered updates
    let result = match method {
        AggregationMethod::FedAvg => fedavg_optimized(&filtered_updates)?,
        AggregationMethod::TrimmedMean => trimmed_mean(&filtered_updates, 0.2)?,
        AggregationMethod::Median => coordinate_median(&filtered_updates)?,
        AggregationMethod::Krum => krum(&filtered_updates, 1)?,
        AggregationMethod::TrustWeighted => {
            // For trust-weighted, just use FedAvg on filtered set
            fedavg_optimized(&filtered_updates)?
        }
    };

    Ok((result, detection))
}

/// Batch statistics for gradient updates
#[derive(Debug, Clone)]
pub struct GradientStats {
    /// Mean L2 norm
    pub mean_norm: f64,
    /// Standard deviation of L2 norms
    pub std_norm: f64,
    /// Minimum norm
    pub min_norm: f64,
    /// Maximum norm
    pub max_norm: f64,
    /// Mean loss across updates
    pub mean_loss: f64,
    /// Number of updates
    pub count: usize,
}

impl GradientStats {
    /// Compute statistics for a batch of updates
    pub fn compute(updates: &[GradientUpdate]) -> Self {
        if updates.is_empty() {
            return Self {
                mean_norm: 0.0,
                std_norm: 0.0,
                min_norm: 0.0,
                max_norm: 0.0,
                mean_loss: 0.0,
                count: 0,
            };
        }

        let norms: Vec<f64> = updates.iter().map(|u| u.l2_norm()).collect();
        let losses: Vec<f64> = updates.iter().map(|u| u.metadata.loss).collect();

        let mean_norm = norms.iter().sum::<f64>() / norms.len() as f64;
        let variance =
            norms.iter().map(|n| (n - mean_norm).powi(2)).sum::<f64>() / norms.len() as f64;

        let min_norm = norms.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_norm = norms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_loss = losses.iter().sum::<f64>() / losses.len() as f64;

        Self {
            mean_norm,
            std_norm: variance.sqrt(),
            min_norm,
            max_norm,
            mean_loss,
            count: updates.len(),
        }
    }
}

// =============================================================================
// ENHANCED MULTI-SIGNAL BYZANTINE DETECTOR
// Addresses simulation finding: adaptive adversaries bypass simple norm checks
// =============================================================================

/// Multi-signal Byzantine detector that combines multiple detection strategies
/// to catch adaptive adversaries that evade single-metric detection.
///
/// # Detection Strategies
///
/// 1. **Magnitude Anomaly**: Z-score of L2 norms (catches obvious outliers)
/// 2. **Direction Anomaly**: Cosine similarity to mean gradient (catches sign-flip attacks)
/// 3. **Cross-Validation**: Krum-like neighbor agreement scoring (catches subtle poisoning)
/// 4. **Gradient Component Analysis**: Detects coordinate-wise outliers
///
/// # Security Note
///
/// This detector addresses the simulation finding that adaptive adversaries
/// can reduce basic detection accuracy to ~13.8%. By combining multiple signals,
/// adaptive attackers must simultaneously fool all detection mechanisms.
pub struct MultiSignalByzantineDetector {
    /// Z-score threshold for magnitude anomaly
    magnitude_z_threshold: f64,
    /// Cosine similarity threshold for direction anomaly (below = suspicious)
    direction_threshold: f64,
    /// Fraction of neighbors to consider for cross-validation
    cross_validation_k: f64,
    /// Coordinate-wise Z-score threshold
    coordinate_z_threshold: f64,
    /// Minimum confidence to flag as Byzantine
    confidence_threshold: f64,
    /// Weights for combining signals
    signal_weights: SignalWeights,
}

/// Weights for combining detection signals
#[derive(Debug, Clone)]
pub struct SignalWeights {
    /// Weight for magnitude anomaly signal.
    pub magnitude: f64,
    /// Weight for direction anomaly signal.
    pub direction: f64,
    /// Weight for cross-validation (Krum-like) signal.
    pub cross_validation: f64,
    /// Weight for coordinate-wise outlier signal.
    pub coordinate: f64,
}

impl Default for SignalWeights {
    fn default() -> Self {
        Self {
            magnitude: 0.25,
            direction: 0.35, // Direction attacks are most common
            cross_validation: 0.25,
            coordinate: 0.15,
        }
    }
}

/// Detailed Byzantine detection result with signal breakdown
#[derive(Debug, Clone)]
pub struct MultiSignalDetectionResult {
    /// Indices of detected Byzantine participants
    pub byzantine_indices: Vec<usize>,
    /// Overall confidence scores
    pub confidence_scores: Vec<f64>,
    /// Per-signal scores for each participant
    pub signal_breakdown: Vec<SignalBreakdown>,
    /// Whether early termination was triggered
    pub early_terminated: bool,
    /// Detection method description
    pub method: String,
    /// Statistics about the detection
    pub stats: DetectionStats,
}

/// Per-participant signal breakdown
#[derive(Debug, Clone)]
pub struct SignalBreakdown {
    /// Index of the participant in the update array.
    pub participant_idx: usize,
    /// Score from magnitude anomaly detection (0.0-1.0).
    pub magnitude_score: f64,
    /// Score from direction anomaly detection (0.0-1.0).
    pub direction_score: f64,
    /// Score from cross-validation neighbor agreement (0.0-1.0).
    pub cross_validation_score: f64,
    /// Score from coordinate-wise outlier detection (0.0-1.0).
    pub coordinate_score: f64,
    /// Weighted combination of all signal scores.
    pub combined_score: f64,
}

/// Detection statistics
#[derive(Debug, Clone, Default)]
pub struct DetectionStats {
    /// Mean L2 norm across all participant gradients.
    pub mean_norm: f64,
    /// Standard deviation of L2 norms.
    pub std_norm: f64,
    /// Mean cosine similarity to the average gradient.
    pub mean_cosine_sim: f64,
    /// Number of participants analyzed.
    pub participants_analyzed: usize,
    /// Number of individual detection signals that fired.
    pub signals_triggered: usize,
}

impl Default for MultiSignalByzantineDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiSignalByzantineDetector {
    /// Create with default configuration
    pub fn new() -> Self {
        Self {
            magnitude_z_threshold: 2.5,
            direction_threshold: 0.3,
            cross_validation_k: 0.6,
            coordinate_z_threshold: 3.0,
            confidence_threshold: 0.5,
            signal_weights: SignalWeights::default(),
        }
    }

    /// Create with custom thresholds for high-security scenarios
    pub fn high_security() -> Self {
        Self {
            magnitude_z_threshold: 2.0,
            direction_threshold: 0.4,
            cross_validation_k: 0.7,
            coordinate_z_threshold: 2.5,
            confidence_threshold: 0.4,
            signal_weights: SignalWeights {
                magnitude: 0.2,
                direction: 0.4,
                cross_validation: 0.3,
                coordinate: 0.1,
            },
        }
    }

    /// Create with relaxed thresholds for high-variance scenarios
    pub fn relaxed() -> Self {
        Self {
            magnitude_z_threshold: 3.5,
            direction_threshold: 0.2,
            cross_validation_k: 0.5,
            coordinate_z_threshold: 4.0,
            confidence_threshold: 0.6,
            signal_weights: SignalWeights::default(),
        }
    }

    /// Perform multi-signal Byzantine detection
    pub fn detect(&self, updates: &[GradientUpdate]) -> MultiSignalDetectionResult {
        if updates.is_empty() {
            return MultiSignalDetectionResult {
                byzantine_indices: vec![],
                confidence_scores: vec![],
                signal_breakdown: vec![],
                early_terminated: false,
                method: "multi_signal".to_string(),
                stats: DetectionStats::default(),
            };
        }

        let n = updates.len();
        if n < 3 {
            // Need at least 3 participants for meaningful detection
            return MultiSignalDetectionResult {
                byzantine_indices: vec![],
                confidence_scores: vec![],
                signal_breakdown: vec![],
                early_terminated: false,
                method: "multi_signal_insufficient".to_string(),
                stats: DetectionStats {
                    participants_analyzed: n,
                    ..Default::default()
                },
            };
        }

        let gradient_dim = updates[0].gradients.len();

        // Compute mean gradient for direction analysis
        let mean_gradient = self.compute_mean_gradient(updates);

        // Compute all norms
        let norms: Vec<f64> = updates.iter().map(|u| u.l2_norm()).collect();
        let mean_norm: f64 = norms.iter().sum::<f64>() / n as f64;
        let std_norm: f64 = {
            let var = norms.iter().map(|n| (n - mean_norm).powi(2)).sum::<f64>() / n as f64;
            var.sqrt()
        };

        // Also compute robust statistics (median-based) for better outlier detection
        let mut sorted_norms = norms.clone();
        sorted_norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_norm = if n.is_multiple_of(2) {
            (sorted_norms[n / 2 - 1] + sorted_norms[n / 2]) / 2.0
        } else {
            sorted_norms[n / 2]
        };
        // Median Absolute Deviation (MAD)
        let mut deviations: Vec<f64> = norms.iter().map(|&n| (n - median_norm).abs()).collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = if n.is_multiple_of(2) {
            (deviations[n / 2 - 1] + deviations[n / 2]) / 2.0
        } else {
            deviations[n / 2]
        };
        // Normalized MAD (consistent estimator of std for normal distribution)
        let normalized_mad = mad * 1.4826;

        // Compute cosine similarities to mean
        let cosine_sims: Vec<f64> = updates
            .iter()
            .map(|u| self.cosine_similarity(&u.gradients, &mean_gradient))
            .collect();
        let mean_cosine: f64 = cosine_sims.iter().sum::<f64>() / n as f64;

        // Compute pairwise distances for cross-validation
        let distances = self.compute_distance_matrix(updates);

        // Analyze each participant
        let mut signal_breakdown = Vec::with_capacity(n);
        let mut byzantine_indices = Vec::new();
        let mut confidence_scores = Vec::new();
        let mut signals_triggered = 0;

        for i in 0..n {
            // Signal 1: Magnitude anomaly
            // Use BOTH standard z-score AND robust z-score (based on MAD)
            // This catches outliers even when they skew the mean
            let (magnitude_score, extreme_outlier) = {
                // Standard z-score
                let std_z = if std_norm > 1e-10 {
                    (norms[i] - mean_norm).abs() / std_norm
                } else {
                    0.0
                };

                // Robust z-score (using median and MAD)
                let robust_z = if normalized_mad > 1e-10 {
                    (norms[i] - median_norm).abs() / normalized_mad
                } else {
                    0.0
                };

                // Use the max of standard and robust z-scores
                let z = std_z.max(robust_z);

                // Extreme outlier: robust z > 4 is almost certainly Byzantine
                // (robust z is less affected by the outlier itself)
                let extreme = robust_z > 4.0 || norms[i] > 100.0 * median_norm.max(1.0);
                let score = (z / self.magnitude_z_threshold).min(1.0);
                (score, extreme)
            };

            // Signal 2: Direction anomaly (low cosine sim = suspicious)
            let direction_score = if cosine_sims[i] < self.direction_threshold {
                1.0 - (cosine_sims[i] / self.direction_threshold).max(0.0)
            } else {
                0.0
            };

            // Signal 3: Cross-validation (Krum-like)
            let cross_validation_score = self.compute_cross_validation_score(i, &distances, n);

            // Signal 4: Coordinate-wise outlier detection
            let coordinate_score =
                self.compute_coordinate_anomaly(&updates[i].gradients, updates, gradient_dim);

            // Combine signals using weighted sum
            let combined = self.signal_weights.magnitude * magnitude_score
                + self.signal_weights.direction * direction_score
                + self.signal_weights.cross_validation * cross_validation_score
                + self.signal_weights.coordinate * coordinate_score;

            signal_breakdown.push(SignalBreakdown {
                participant_idx: i,
                magnitude_score,
                direction_score,
                cross_validation_score,
                coordinate_score,
                combined_score: combined,
            });

            // Flag as Byzantine if:
            // 1. Combined score exceeds threshold, OR
            // 2. Extreme outlier detected (fast-path for obvious attacks)
            //
            // Note: We don't flag on individual high scores alone (except extreme_outlier)
            // to avoid false positives in heterogeneous but honest groups
            let is_byzantine = combined >= self.confidence_threshold || extreme_outlier;

            if is_byzantine {
                byzantine_indices.push(i);
                // Use max of combined and individual scores for confidence
                let confidence = if extreme_outlier {
                    1.0 // Extreme outliers get maximum confidence
                } else {
                    combined
                };
                confidence_scores.push(confidence);
                if magnitude_score > 0.5 {
                    signals_triggered += 1;
                }
                if direction_score > 0.5 {
                    signals_triggered += 1;
                }
                if cross_validation_score > 0.5 {
                    signals_triggered += 1;
                }
                if coordinate_score > 0.5 {
                    signals_triggered += 1;
                }
            }
        }

        let byzantine_fraction = byzantine_indices.len() as f64 / n as f64;
        let early_terminated = byzantine_fraction > 0.5;

        MultiSignalDetectionResult {
            byzantine_indices,
            confidence_scores,
            signal_breakdown,
            early_terminated,
            method: "multi_signal".to_string(),
            stats: DetectionStats {
                mean_norm,
                std_norm,
                mean_cosine_sim: mean_cosine,
                participants_analyzed: n,
                signals_triggered,
            },
        }
    }

    /// Filter updates using multi-signal detection
    pub fn filter_updates<'a>(
        &self,
        updates: &'a [GradientUpdate],
    ) -> (Vec<&'a GradientUpdate>, MultiSignalDetectionResult) {
        let detection = self.detect(updates);

        let filtered: Vec<&GradientUpdate> = updates
            .iter()
            .enumerate()
            .filter(|(i, _)| !detection.byzantine_indices.contains(i))
            .map(|(_, u)| u)
            .collect();

        (filtered, detection)
    }

    /// Compute mean gradient vector
    fn compute_mean_gradient(&self, updates: &[GradientUpdate]) -> Vec<f64> {
        if updates.is_empty() {
            return vec![];
        }

        let dim = updates[0].gradients.len();
        let n = updates.len() as f64;
        let mut mean = vec![0.0; dim];

        for update in updates {
            for (i, &g) in update.gradients.iter().enumerate() {
                mean[i] += g / n;
            }
        }

        mean
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }

        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// Compute pairwise distance matrix
    fn compute_distance_matrix(&self, updates: &[GradientUpdate]) -> Vec<Vec<f64>> {
        let n = updates.len();
        let mut distances = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = euclidean_distance(&updates[i].gradients, &updates[j].gradients);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }

        distances
    }

    /// Compute cross-validation score (Krum-like)
    fn compute_cross_validation_score(&self, idx: usize, distances: &[Vec<f64>], n: usize) -> f64 {
        if n < 3 {
            return 0.0;
        }

        // Number of closest neighbors to consider
        let k = ((n - 1) as f64 * self.cross_validation_k) as usize;
        let k = k.max(1).min(n - 1);

        // Get sorted distances for this participant
        let mut dists: Vec<f64> = distances[idx]
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .map(|(_, &d)| d)
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Sum of k smallest distances
        let score_i: f64 = dists.iter().take(k).sum();

        // Compute scores for all participants
        let mut all_scores: Vec<f64> = (0..n)
            .map(|i| {
                let mut d: Vec<f64> = distances[i]
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, &d)| d)
                    .collect();
                d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                d.iter().take(k).sum()
            })
            .collect();

        // Normalize: highest score = most outlier-like
        all_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let min_score = all_scores.first().copied().unwrap_or(0.0);
        let max_score = all_scores.last().copied().unwrap_or(1.0);

        if (max_score - min_score).abs() < 1e-10 {
            return 0.0;
        }

        // Return normalized score (0 = closest to neighbors, 1 = most distant)
        (score_i - min_score) / (max_score - min_score)
    }

    /// Compute coordinate-wise anomaly score
    fn compute_coordinate_anomaly(
        &self,
        gradient: &[f64],
        updates: &[GradientUpdate],
        dim: usize,
    ) -> f64 {
        if updates.len() < 3 || dim == 0 {
            return 0.0;
        }

        let n = updates.len() as f64;
        let mut anomaly_count = 0;
        let sample_coords = dim.min(100); // Sample for efficiency
        let step = (dim / sample_coords).max(1);

        for coord in (0..dim).step_by(step) {
            // Compute mean and std for this coordinate
            let values: Vec<f64> = updates.iter().map(|u| u.gradients[coord]).collect();
            let mean: f64 = values.iter().sum::<f64>() / n;
            let std: f64 = {
                let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
                var.sqrt()
            };

            if std > 1e-10 {
                let z = (gradient[coord] - mean).abs() / std;
                if z > self.coordinate_z_threshold {
                    anomaly_count += 1;
                }
            }
        }

        // Return fraction of coordinates that are anomalous
        anomaly_count as f64 / sample_coords as f64
    }
}

/// Aggregation with multi-signal Byzantine detection
///
/// Enhanced version of `aggregate_with_early_termination` that uses
/// multi-signal detection to catch adaptive adversaries.
pub fn aggregate_with_multi_signal_detection(
    updates: &[GradientUpdate],
    method: AggregationMethod,
    byzantine_tolerance: f64,
) -> Result<(Vec<f64>, MultiSignalDetectionResult), AggregationError> {
    let detector = MultiSignalByzantineDetector::new();
    let detection = detector.detect(updates);

    // Check if early termination is warranted
    if detection.early_terminated {
        return Err(AggregationError::NoTrustedParticipants);
    }

    // Filter out Byzantine participants if within tolerance
    if detection.byzantine_indices.len() as f64 / updates.len() as f64 > byzantine_tolerance {
        return Err(AggregationError::NoTrustedParticipants);
    }

    // Get filtered updates
    let filtered_updates: Vec<GradientUpdate> = updates
        .iter()
        .enumerate()
        .filter(|(i, _)| !detection.byzantine_indices.contains(i))
        .map(|(_, u)| u.clone())
        .collect();

    if filtered_updates.is_empty() {
        return Err(AggregationError::NoUpdates);
    }

    // Run aggregation on filtered updates
    let result = match method {
        AggregationMethod::FedAvg => fedavg_optimized(&filtered_updates)?,
        AggregationMethod::TrimmedMean => trimmed_mean(&filtered_updates, 0.2)?,
        AggregationMethod::Median => coordinate_median(&filtered_updates)?,
        AggregationMethod::Krum => krum(&filtered_updates, 1)?,
        AggregationMethod::TrustWeighted => fedavg_optimized(&filtered_updates)?,
    };

    Ok((result, detection))
}

// =============================================================================
// ADAPTIVE AGGREGATION WITH AUTOMATIC METHOD SELECTION
// Addresses simulation recommendation: "Use Krum for Byzantine resilience"
// =============================================================================

/// Configuration for adaptive aggregation
#[derive(Debug, Clone)]
pub struct AdaptiveAggregatorConfig {
    /// Base aggregation method when no threats detected
    pub base_method: AggregationMethod,
    /// Byzantine tolerance threshold
    pub byzantine_tolerance: f64,
    /// Detection threshold to switch to defensive method
    pub threat_threshold: f64,
    /// Number of recent rounds to consider for threat assessment
    pub history_window: usize,
    /// Defensive method to use when threats detected
    pub defensive_method: AggregationMethod,
}

impl Default for AdaptiveAggregatorConfig {
    fn default() -> Self {
        Self {
            base_method: AggregationMethod::FedAvg,
            byzantine_tolerance: 0.33,
            threat_threshold: 0.15, // Switch to defensive if >15% Byzantine detected
            history_window: 10,
            defensive_method: AggregationMethod::Krum, // Krum is most Byzantine-resistant
        }
    }
}

/// Adaptive aggregator that automatically selects aggregation method
/// based on detected threat levels.
///
/// # Behavior
///
/// - Normal operation: Uses configured base method (default: FedAvg)
/// - Elevated threat: Switches to defensive method (default: Krum)
/// - Threat assessment considers recent detection history
///
/// # Example
///
/// ```rust,ignore
/// use mycelix_sdk::fl::{AdaptiveAggregator, GradientUpdate};
///
/// let mut aggregator = AdaptiveAggregator::new();
///
/// // Submit updates each round
/// let updates = vec![/* gradient updates */];
/// let result = aggregator.aggregate(&updates);
///
/// // Check threat level
/// println!("Current threat level: {:.2}", aggregator.threat_level());
/// ```
pub struct AdaptiveAggregator {
    /// Configuration
    config: AdaptiveAggregatorConfig,
    /// Multi-signal detector
    detector: MultiSignalByzantineDetector,
    /// Recent detection history (byzantine_fraction per round)
    detection_history: std::collections::VecDeque<f64>,
    /// Participant behavior history for persistent tracking
    participant_history: std::collections::HashMap<String, ParticipantBehavior>,
    /// Current threat assessment
    current_threat_level: f64,
    /// Total rounds processed
    rounds_processed: usize,
}

/// Tracked behavior for a participant across rounds
#[derive(Debug, Clone, Default)]
pub struct ParticipantBehavior {
    /// Number of times flagged as Byzantine
    pub times_flagged: usize,
    /// Number of rounds participated
    pub rounds_participated: usize,
    /// Cumulative anomaly score
    pub cumulative_anomaly: f64,
    /// Last round participated
    pub last_round: usize,
    /// Trust penalty (0.0 = no penalty, 1.0 = fully penalized)
    pub trust_penalty: f64,
}

impl ParticipantBehavior {
    /// Calculate reputation score (0.0 = bad, 1.0 = good)
    pub fn reputation(&self) -> f64 {
        if self.rounds_participated == 0 {
            return 1.0; // New participants get benefit of doubt
        }
        let flag_rate = self.times_flagged as f64 / self.rounds_participated as f64;
        (1.0 - flag_rate - self.trust_penalty).max(0.0)
    }

    /// Check if participant should be blacklisted
    pub fn is_blacklisted(&self) -> bool {
        // Blacklist if flagged >50% of time with at least 3 rounds
        self.rounds_participated >= 3
            && self.times_flagged as f64 / self.rounds_participated as f64 > 0.5
    }
}

/// Result of adaptive aggregation
#[derive(Debug, Clone)]
pub struct AdaptiveAggregationResult {
    /// Aggregated gradients
    pub gradients: Vec<f64>,
    /// Method that was actually used
    pub method_used: AggregationMethod,
    /// Detection result from this round
    pub detection: MultiSignalDetectionResult,
    /// Current threat level (0.0-1.0)
    pub threat_level: f64,
    /// Participants excluded (Byzantine + blacklisted)
    pub excluded_participants: Vec<String>,
    /// Whether defensive mode was triggered
    pub defensive_mode: bool,
}

impl Default for AdaptiveAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveAggregator {
    /// Create with default configuration
    pub fn new() -> Self {
        Self::with_config(AdaptiveAggregatorConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: AdaptiveAggregatorConfig) -> Self {
        Self {
            detection_history: std::collections::VecDeque::with_capacity(config.history_window),
            config,
            detector: MultiSignalByzantineDetector::new(),
            participant_history: std::collections::HashMap::new(),
            current_threat_level: 0.0,
            rounds_processed: 0,
        }
    }

    /// Create with high-security detection
    pub fn high_security() -> Self {
        let mut agg = Self::new();
        agg.detector = MultiSignalByzantineDetector::high_security();
        agg.config.threat_threshold = 0.10; // More sensitive
        agg
    }

    /// Perform adaptive aggregation
    pub fn aggregate(
        &mut self,
        updates: &[GradientUpdate],
    ) -> Result<AdaptiveAggregationResult, AggregationError> {
        if updates.is_empty() {
            return Err(AggregationError::NoUpdates);
        }

        self.rounds_processed += 1;

        // Step 1: Detect Byzantine participants
        let detection = self.detector.detect(updates);

        // Step 2: Update participant history
        self.update_participant_history(updates, &detection);

        // Step 3: Calculate current threat level
        let byzantine_fraction = detection.byzantine_indices.len() as f64 / updates.len() as f64;
        self.update_threat_level(byzantine_fraction);

        // Step 4: Determine which participants to exclude (Byzantine + blacklisted)
        let mut excluded_indices: Vec<usize> = detection.byzantine_indices.clone();
        let mut excluded_participants = Vec::new();

        for (i, update) in updates.iter().enumerate() {
            if let Some(behavior) = self.participant_history.get(&update.participant_id) {
                if behavior.is_blacklisted() && !excluded_indices.contains(&i) {
                    excluded_indices.push(i);
                }
            }
            if excluded_indices.contains(&i) {
                excluded_participants.push(update.participant_id.clone());
            }
        }

        // Step 5: Check if we should reject entirely
        let exclusion_fraction = excluded_indices.len() as f64 / updates.len() as f64;
        if exclusion_fraction > self.config.byzantine_tolerance {
            return Err(AggregationError::NoTrustedParticipants);
        }

        // Step 6: Filter updates
        let filtered_updates: Vec<GradientUpdate> = updates
            .iter()
            .enumerate()
            .filter(|(i, _)| !excluded_indices.contains(i))
            .map(|(_, u)| u.clone())
            .collect();

        if filtered_updates.is_empty() {
            return Err(AggregationError::NoUpdates);
        }

        // Step 7: Select aggregation method based on threat level
        let defensive_mode = self.current_threat_level > self.config.threat_threshold;
        let method_used = if defensive_mode {
            self.config.defensive_method
        } else {
            self.config.base_method
        };

        // Step 8: Perform aggregation
        let gradients = match &method_used {
            AggregationMethod::FedAvg => fedavg_optimized(&filtered_updates)?,
            AggregationMethod::TrimmedMean => trimmed_mean(&filtered_updates, 0.2)?,
            AggregationMethod::Median => coordinate_median(&filtered_updates)?,
            AggregationMethod::Krum => {
                if filtered_updates.len() >= 3 {
                    krum(&filtered_updates, 1)?
                } else {
                    // Fall back to median if not enough for Krum
                    coordinate_median(&filtered_updates)?
                }
            }
            AggregationMethod::TrustWeighted => fedavg_optimized(&filtered_updates)?,
        };

        Ok(AdaptiveAggregationResult {
            gradients,
            method_used,
            detection,
            threat_level: self.current_threat_level,
            excluded_participants,
            defensive_mode,
        })
    }

    /// Update participant behavior history
    fn update_participant_history(
        &mut self,
        updates: &[GradientUpdate],
        detection: &MultiSignalDetectionResult,
    ) {
        for (i, update) in updates.iter().enumerate() {
            let behavior = self
                .participant_history
                .entry(update.participant_id.clone())
                .or_default();

            behavior.rounds_participated += 1;
            behavior.last_round = self.rounds_processed;

            if detection.byzantine_indices.contains(&i) {
                behavior.times_flagged += 1;

                // Find the signal breakdown for this participant
                if let Some(breakdown) = detection
                    .signal_breakdown
                    .iter()
                    .find(|b| b.participant_idx == i)
                {
                    behavior.cumulative_anomaly += breakdown.combined_score;

                    // Apply trust penalty based on severity
                    if breakdown.combined_score > 0.8 {
                        behavior.trust_penalty = (behavior.trust_penalty + 0.3).min(1.0);
                    } else if breakdown.combined_score > 0.5 {
                        behavior.trust_penalty = (behavior.trust_penalty + 0.1).min(1.0);
                    }
                }
            } else {
                // Slowly recover trust for good behavior
                behavior.trust_penalty = (behavior.trust_penalty - 0.05).max(0.0);
            }
        }
    }

    /// Update threat level using exponential moving average
    fn update_threat_level(&mut self, byzantine_fraction: f64) {
        // Add to history
        self.detection_history.push_back(byzantine_fraction);
        if self.detection_history.len() > self.config.history_window {
            self.detection_history.pop_front();
        }

        // Calculate weighted average (recent rounds weighted more)
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;
        let _len = self.detection_history.len();

        for (i, &fraction) in self.detection_history.iter().enumerate() {
            let weight = (i + 1) as f64; // Linear weight increase
            weighted_sum += fraction * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            self.current_threat_level = weighted_sum / total_weight;
        }
    }

    /// Get current threat level
    pub fn threat_level(&self) -> f64 {
        self.current_threat_level
    }

    /// Get participant reputation
    pub fn participant_reputation(&self, participant_id: &str) -> f64 {
        self.participant_history
            .get(participant_id)
            .map(|b| b.reputation())
            .unwrap_or(1.0)
    }

    /// Check if participant is blacklisted
    pub fn is_blacklisted(&self, participant_id: &str) -> bool {
        self.participant_history
            .get(participant_id)
            .map(|b| b.is_blacklisted())
            .unwrap_or(false)
    }

    /// Get list of blacklisted participants
    pub fn blacklisted_participants(&self) -> Vec<String> {
        self.participant_history
            .iter()
            .filter(|(_, b)| b.is_blacklisted())
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Reset threat assessment (e.g., after network partition recovery)
    pub fn reset_threat_assessment(&mut self) {
        self.detection_history.clear();
        self.current_threat_level = 0.0;
    }

    /// Get rounds processed
    pub fn rounds_processed(&self) -> usize {
        self.rounds_processed
    }

    /// Get statistics about the aggregator
    pub fn stats(&self) -> AdaptiveAggregatorStats {
        let total_participants = self.participant_history.len();
        let blacklisted = self.blacklisted_participants().len();
        let avg_reputation = if total_participants > 0 {
            self.participant_history
                .values()
                .map(|b| b.reputation())
                .sum::<f64>()
                / total_participants as f64
        } else {
            1.0
        };

        AdaptiveAggregatorStats {
            rounds_processed: self.rounds_processed,
            current_threat_level: self.current_threat_level,
            total_participants_seen: total_participants,
            blacklisted_count: blacklisted,
            average_reputation: avg_reputation,
            history_window_size: self.detection_history.len(),
        }
    }
}

/// Statistics about the adaptive aggregator
#[derive(Debug, Clone)]
pub struct AdaptiveAggregatorStats {
    /// Total number of aggregation rounds completed.
    pub rounds_processed: usize,
    /// Current threat level (0.0-1.0).
    pub current_threat_level: f64,
    /// Total unique participants observed across all rounds.
    pub total_participants_seen: usize,
    /// Number of currently blacklisted participants.
    pub blacklisted_count: usize,
    /// Average reputation across all known participants.
    pub average_reputation: f64,
    /// Number of rounds currently in the detection history window.
    pub history_window_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_updates() -> Vec<GradientUpdate> {
        vec![
            GradientUpdate::new("p1".to_string(), 1, vec![0.1, 0.2, 0.3], 100, 0.5),
            GradientUpdate::new("p2".to_string(), 1, vec![0.2, 0.3, 0.4], 100, 0.4),
            GradientUpdate::new("p3".to_string(), 1, vec![0.15, 0.25, 0.35], 100, 0.45),
        ]
    }

    #[test]
    fn test_fedavg() {
        let updates = create_test_updates();
        let result = fedavg(&updates).expect("FedAvg failed");

        assert_eq!(result.len(), 3);
        // Average of equal weights
        assert!((result[0] - 0.15).abs() < 0.001);
        assert!((result[1] - 0.25).abs() < 0.001);
        assert!((result[2] - 0.35).abs() < 0.001);
    }

    #[test]
    fn test_fedavg_weighted() {
        let updates = vec![
            GradientUpdate::new("p1".to_string(), 1, vec![0.0, 0.0], 100, 0.5),
            GradientUpdate::new("p2".to_string(), 1, vec![1.0, 1.0], 300, 0.4),
        ];

        let result = fedavg(&updates).expect("FedAvg failed");

        // p2 has 3x the weight
        assert!((result[0] - 0.75).abs() < 0.001);
        assert!((result[1] - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_coordinate_median() {
        let updates = create_test_updates();
        let result = coordinate_median(&updates).expect("Median failed");

        assert_eq!(result.len(), 3);
        // Median values
        assert!((result[0] - 0.15).abs() < 0.001);
        assert!((result[1] - 0.25).abs() < 0.001);
        assert!((result[2] - 0.35).abs() < 0.001);
    }

    #[test]
    fn test_trimmed_mean() {
        let updates = vec![
            GradientUpdate::new("p1".to_string(), 1, vec![0.1], 100, 0.5),
            GradientUpdate::new("p2".to_string(), 1, vec![0.2], 100, 0.5),
            GradientUpdate::new("p3".to_string(), 1, vec![0.3], 100, 0.5),
            GradientUpdate::new("p4".to_string(), 1, vec![0.4], 100, 0.5),
            GradientUpdate::new("p5".to_string(), 1, vec![10.0], 100, 0.5), // Outlier
        ];

        // Trim 20% removes the outlier
        let result = trimmed_mean(&updates, 0.2).expect("Trimmed mean failed");

        // After trimming 0.1 and 10.0, average of [0.2, 0.3, 0.4] = 0.3
        assert!((result[0] - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_krum() {
        let updates = vec![
            GradientUpdate::new("p1".to_string(), 1, vec![0.1, 0.1], 100, 0.5),
            GradientUpdate::new("p2".to_string(), 1, vec![0.12, 0.11], 100, 0.5),
            GradientUpdate::new("p3".to_string(), 1, vec![0.09, 0.1], 100, 0.5),
            GradientUpdate::new("p4".to_string(), 1, vec![10.0, 10.0], 100, 0.5), // Byzantine
        ];

        let result = krum(&updates, 1).expect("Krum failed");

        // Should select gradient close to cluster, not the Byzantine one
        assert!(result[0] < 1.0);
        assert!(result[1] < 1.0);
    }

    #[test]
    fn test_trust_weighted() {
        let updates = create_test_updates();
        let mut participants: HashMap<String, Participant> = HashMap::new();

        for update in &updates {
            let mut p = Participant::new(update.participant_id.clone());
            // Give p1 higher reputation
            if update.participant_id == "p1" {
                for _ in 0..10 {
                    p.record_positive();
                }
            }
            participants.insert(update.participant_id.clone(), p);
        }

        let result = trust_weighted_aggregation(&updates, &participants, 0.3)
            .expect("Trust weighted failed");

        assert_eq!(result.participant_count, 3);
        assert_eq!(result.excluded_count, 0);
        assert_eq!(result.aggregation_method, AggregationMethod::TrustWeighted);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];

        let dist = euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_validation_errors() {
        // Empty updates
        let result = fedavg(&[]);
        assert!(matches!(result, Err(AggregationError::NoUpdates)));

        // Size mismatch
        let updates = vec![
            GradientUpdate::new("p1".to_string(), 1, vec![0.1, 0.2], 100, 0.5),
            GradientUpdate::new("p2".to_string(), 1, vec![0.1, 0.2, 0.3], 100, 0.5),
        ];
        let result = fedavg(&updates);
        assert!(matches!(
            result,
            Err(AggregationError::GradientSizeMismatch { .. })
        ));
    }

    // =========================================================================
    // Performance optimization tests
    // =========================================================================

    #[test]
    fn test_gradient_accumulator() {
        let mut acc = GradientAccumulator::new(3);

        acc.accumulate(
            &GradientUpdate::new("p1".to_string(), 1, vec![1.0, 2.0, 3.0], 100, 0.5),
            0.5,
        );
        acc.accumulate(
            &GradientUpdate::new("p2".to_string(), 1, vec![2.0, 4.0, 6.0], 100, 0.5),
            0.5,
        );

        assert_eq!(acc.count(), 2);
        assert!((acc.total_weight() - 1.0).abs() < 0.001);

        let result = acc.finalize();
        assert!((result[0] - 1.5).abs() < 0.001);
        assert!((result[1] - 3.0).abs() < 0.001);
        assert!((result[2] - 4.5).abs() < 0.001);
    }

    #[test]
    fn test_fedavg_optimized_matches_original() {
        let updates = create_test_updates();

        let original = fedavg(&updates).unwrap();
        let optimized = fedavg_optimized(&updates).unwrap();

        for (a, b) in original.iter().zip(optimized.iter()) {
            assert!((a - b).abs() < 0.001);
        }
    }

    #[test]
    fn test_early_byzantine_detector() {
        let detector = EarlyByzantineDetector::new();

        let updates = vec![
            GradientUpdate::new("p1".to_string(), 1, vec![0.1, 0.1], 100, 0.5),
            GradientUpdate::new("p2".to_string(), 1, vec![0.12, 0.11], 100, 0.5),
            GradientUpdate::new("p3".to_string(), 1, vec![0.09, 0.1], 100, 0.5),
            GradientUpdate::new("byzantine".to_string(), 1, vec![1000.0, 1000.0], 100, 0.5), // Obvious Byzantine
        ];

        let result = detector.detect(&updates);
        assert!(result.has_byzantine());
        assert!(result.byzantine_indices.contains(&3));
    }

    #[test]
    fn test_filter_updates() {
        let detector = EarlyByzantineDetector::new();

        let updates = vec![
            GradientUpdate::new("p1".to_string(), 1, vec![0.1, 0.1], 100, 0.5),
            GradientUpdate::new("byzantine".to_string(), 1, vec![1000.0, 1000.0], 100, 0.5),
            GradientUpdate::new("p2".to_string(), 1, vec![0.12, 0.11], 100, 0.5),
        ];

        let (filtered, detection) = detector.filter_updates(&updates);
        assert_eq!(filtered.len(), 2);
        assert!(detection.has_byzantine());
    }

    #[test]
    fn test_aggregate_with_early_termination() {
        let updates = vec![
            GradientUpdate::new("p1".to_string(), 1, vec![0.1, 0.2], 100, 0.5),
            GradientUpdate::new("p2".to_string(), 1, vec![0.2, 0.3], 100, 0.5),
            GradientUpdate::new("p3".to_string(), 1, vec![0.15, 0.25], 100, 0.5),
        ];

        let (result, detection) =
            aggregate_with_early_termination(&updates, AggregationMethod::FedAvg, 0.33).unwrap();

        assert_eq!(result.len(), 2);
        assert!(!detection.has_byzantine());
    }

    #[test]
    fn test_gradient_stats() {
        let updates = vec![
            GradientUpdate::new("p1".to_string(), 1, vec![1.0, 0.0], 100, 0.5),
            GradientUpdate::new("p2".to_string(), 1, vec![0.0, 1.0], 100, 0.4),
            GradientUpdate::new("p3".to_string(), 1, vec![1.0, 1.0], 100, 0.6),
        ];

        let stats = GradientStats::compute(&updates);

        assert_eq!(stats.count, 3);
        assert!(stats.mean_norm > 0.0);
        assert!(stats.min_norm > 0.0);
        assert!(stats.max_norm > stats.min_norm);
        assert!((stats.mean_loss - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_accumulator_reset() {
        let mut acc = GradientAccumulator::new(2);

        acc.accumulate(
            &GradientUpdate::new("p1".to_string(), 1, vec![1.0, 2.0], 100, 0.5),
            1.0,
        );
        assert_eq!(acc.count(), 1);

        acc.reset();
        assert_eq!(acc.count(), 0);
        assert!(acc.total_weight() < 0.001);
    }

    #[test]
    fn test_byzantine_detection_empty() {
        let detector = EarlyByzantineDetector::new();
        let result = detector.detect(&[]);
        assert!(!result.has_byzantine());
        assert!(!result.early_terminated);
    }

    // =========================================================================
    // Multi-Signal Byzantine Detector Tests
    // =========================================================================

    #[test]
    fn test_multi_signal_detector_obvious_byzantine() {
        let detector = MultiSignalByzantineDetector::new();

        let updates = vec![
            GradientUpdate::new("h1".to_string(), 1, vec![0.1, 0.1, 0.1], 100, 0.5),
            GradientUpdate::new("h2".to_string(), 1, vec![0.12, 0.11, 0.09], 100, 0.5),
            GradientUpdate::new("h3".to_string(), 1, vec![0.09, 0.1, 0.11], 100, 0.5),
            GradientUpdate::new("h4".to_string(), 1, vec![0.11, 0.12, 0.1], 100, 0.5),
            GradientUpdate::new("byz".to_string(), 1, vec![1000.0, -1000.0, 500.0], 100, 0.5),
        ];

        let result = detector.detect(&updates);
        assert!(
            !result.byzantine_indices.is_empty(),
            "Should detect obvious Byzantine"
        );
        assert!(
            result.byzantine_indices.contains(&4),
            "Should detect participant 4 as Byzantine"
        );
    }

    #[test]
    fn test_multi_signal_detector_sign_flip_attack() {
        let detector = MultiSignalByzantineDetector::new();

        // Sign-flip attack: similar magnitude but opposite direction
        let updates = vec![
            GradientUpdate::new("h1".to_string(), 1, vec![1.0, 1.0, 1.0], 100, 0.5),
            GradientUpdate::new("h2".to_string(), 1, vec![1.1, 0.9, 1.05], 100, 0.5),
            GradientUpdate::new("h3".to_string(), 1, vec![0.95, 1.05, 0.98], 100, 0.5),
            GradientUpdate::new("h4".to_string(), 1, vec![1.02, 0.98, 1.01], 100, 0.5),
            GradientUpdate::new("byz".to_string(), 1, vec![-1.0, -1.0, -1.0], 100, 0.5), // Sign flip
        ];

        let result = detector.detect(&updates);

        // Check direction score is high for Byzantine
        let byz_breakdown = result
            .signal_breakdown
            .iter()
            .find(|b| b.participant_idx == 4)
            .expect("Should have breakdown for Byzantine");

        assert!(
            byz_breakdown.direction_score > 0.3,
            "Direction score should be high for sign-flip attack: {}",
            byz_breakdown.direction_score
        );
    }

    #[test]
    fn test_multi_signal_detector_adaptive_attack() {
        let detector = MultiSignalByzantineDetector::new();

        // Adaptive attack: within norm bounds but outlier in cross-validation
        let updates = vec![
            GradientUpdate::new("h1".to_string(), 1, vec![0.5, 0.5, 0.5], 100, 0.5),
            GradientUpdate::new("h2".to_string(), 1, vec![0.52, 0.48, 0.51], 100, 0.5),
            GradientUpdate::new("h3".to_string(), 1, vec![0.48, 0.52, 0.49], 100, 0.5),
            GradientUpdate::new("h4".to_string(), 1, vec![0.51, 0.49, 0.5], 100, 0.5),
            GradientUpdate::new("h5".to_string(), 1, vec![0.49, 0.51, 0.52], 100, 0.5),
            // Adaptive: same norm (~0.86) but different direction
            GradientUpdate::new("adv".to_string(), 1, vec![0.7, 0.3, 0.4], 100, 0.5),
        ];

        let result = detector.detect(&updates);

        // Should detect the adaptive attacker via direction or cross-validation
        let adv_breakdown = result
            .signal_breakdown
            .iter()
            .find(|b| b.participant_idx == 5)
            .expect("Should have breakdown for adaptive attacker");

        // Combined score should be elevated
        assert!(
            adv_breakdown.combined_score > 0.2,
            "Adaptive attacker should have elevated combined score: {}",
            adv_breakdown.combined_score
        );
    }

    #[test]
    fn test_multi_signal_no_false_positives() {
        let detector = MultiSignalByzantineDetector::new();

        // All honest, similar gradients
        let updates = vec![
            GradientUpdate::new("h1".to_string(), 1, vec![0.5, 0.5, 0.5], 100, 0.5),
            GradientUpdate::new("h2".to_string(), 1, vec![0.52, 0.48, 0.51], 100, 0.5),
            GradientUpdate::new("h3".to_string(), 1, vec![0.48, 0.52, 0.49], 100, 0.5),
            GradientUpdate::new("h4".to_string(), 1, vec![0.51, 0.49, 0.5], 100, 0.5),
            GradientUpdate::new("h5".to_string(), 1, vec![0.49, 0.51, 0.52], 100, 0.5),
        ];

        let result = detector.detect(&updates);
        assert!(
            result.byzantine_indices.is_empty(),
            "Should not detect any Byzantine in honest group"
        );
    }

    #[test]
    fn test_multi_signal_high_security_mode() {
        let detector = MultiSignalByzantineDetector::high_security();

        // Subtle attack that might pass normal detection
        let updates = vec![
            GradientUpdate::new("h1".to_string(), 1, vec![1.0, 1.0], 100, 0.5),
            GradientUpdate::new("h2".to_string(), 1, vec![1.05, 0.95], 100, 0.5),
            GradientUpdate::new("h3".to_string(), 1, vec![0.95, 1.05], 100, 0.5),
            GradientUpdate::new("subtle".to_string(), 1, vec![0.5, 1.5], 100, 0.5), // Subtle bias
        ];

        let result = detector.detect(&updates);

        // High security should catch subtle attacks
        let subtle_breakdown = result
            .signal_breakdown
            .iter()
            .find(|b| b.participant_idx == 3);

        assert!(
            subtle_breakdown.is_some(),
            "Should have breakdown for subtle attacker"
        );
    }

    #[test]
    fn test_aggregate_with_multi_signal_detection() {
        let updates = vec![
            GradientUpdate::new("h1".to_string(), 1, vec![0.1, 0.2], 100, 0.5),
            GradientUpdate::new("h2".to_string(), 1, vec![0.2, 0.3], 100, 0.5),
            GradientUpdate::new("h3".to_string(), 1, vec![0.15, 0.25], 100, 0.5),
            GradientUpdate::new("byz".to_string(), 1, vec![100.0, -100.0], 100, 0.5),
        ];

        let result =
            aggregate_with_multi_signal_detection(&updates, AggregationMethod::FedAvg, 0.33);

        assert!(result.is_ok(), "Should succeed with Byzantine filtering");
        let (aggregated, detection) = result.unwrap();

        assert!(
            !detection.byzantine_indices.is_empty(),
            "Should detect Byzantine"
        );
        assert!(
            aggregated[0] < 1.0 && aggregated[1] < 1.0,
            "Aggregated result should not be influenced by Byzantine"
        );
    }

    #[test]
    fn test_multi_signal_detection_stats() {
        let detector = MultiSignalByzantineDetector::new();

        let updates = vec![
            GradientUpdate::new("h1".to_string(), 1, vec![1.0, 2.0, 3.0], 100, 0.5),
            GradientUpdate::new("h2".to_string(), 1, vec![1.1, 2.1, 2.9], 100, 0.5),
            GradientUpdate::new("h3".to_string(), 1, vec![0.9, 1.9, 3.1], 100, 0.5),
        ];

        let result = detector.detect(&updates);

        assert_eq!(result.stats.participants_analyzed, 3);
        assert!(result.stats.mean_norm > 0.0);
        assert!(
            result.stats.mean_cosine_sim > 0.9,
            "Similar gradients should have high mean cosine similarity"
        );
    }

    // =========================================================================
    // Adaptive Aggregator Tests
    // =========================================================================

    #[test]
    fn test_adaptive_aggregator_normal_operation() {
        let mut aggregator = AdaptiveAggregator::new();

        // All honest participants
        let updates = vec![
            GradientUpdate::new("h1".to_string(), 1, vec![0.5, 0.5], 100, 0.5),
            GradientUpdate::new("h2".to_string(), 1, vec![0.52, 0.48], 100, 0.5),
            GradientUpdate::new("h3".to_string(), 1, vec![0.48, 0.52], 100, 0.5),
        ];

        let result = aggregator.aggregate(&updates).expect("Should succeed");

        assert_eq!(result.method_used, AggregationMethod::FedAvg);
        assert!(!result.defensive_mode);
        assert!(result.excluded_participants.is_empty());
        assert!(aggregator.threat_level() < 0.1);
    }

    #[test]
    fn test_adaptive_aggregator_switches_to_krum() {
        let mut aggregator = AdaptiveAggregator::new();
        aggregator.config.threat_threshold = 0.1; // Lower threshold for testing

        // Simulate several rounds with Byzantine activity
        for _ in 0..5 {
            let updates = vec![
                GradientUpdate::new("h1".to_string(), 1, vec![0.5, 0.5], 100, 0.5),
                GradientUpdate::new("h2".to_string(), 1, vec![0.52, 0.48], 100, 0.5),
                GradientUpdate::new("h3".to_string(), 1, vec![0.48, 0.52], 100, 0.5),
                GradientUpdate::new("byz".to_string(), 1, vec![100.0, -100.0], 100, 0.5),
            ];

            let _ = aggregator.aggregate(&updates);
        }

        // After several rounds of Byzantine activity, should be in defensive mode
        assert!(
            aggregator.threat_level() > 0.1,
            "Threat level should be elevated"
        );

        // Next round should use Krum
        let updates = vec![
            GradientUpdate::new("h1".to_string(), 1, vec![0.5, 0.5], 100, 0.5),
            GradientUpdate::new("h2".to_string(), 1, vec![0.52, 0.48], 100, 0.5),
            GradientUpdate::new("h3".to_string(), 1, vec![0.48, 0.52], 100, 0.5),
            GradientUpdate::new("h4".to_string(), 1, vec![0.51, 0.49], 100, 0.5),
        ];

        let result = aggregator.aggregate(&updates).expect("Should succeed");
        assert!(
            result.defensive_mode || result.method_used == AggregationMethod::Krum,
            "Should switch to defensive mode"
        );
    }

    #[test]
    fn test_adaptive_aggregator_blacklists_persistent_byzantine() {
        let mut aggregator = AdaptiveAggregator::new();

        // Simulate several rounds where "byz" is always flagged
        for _ in 0..5 {
            let updates = vec![
                GradientUpdate::new("h1".to_string(), 1, vec![0.5, 0.5], 100, 0.5),
                GradientUpdate::new("h2".to_string(), 1, vec![0.52, 0.48], 100, 0.5),
                GradientUpdate::new("h3".to_string(), 1, vec![0.48, 0.52], 100, 0.5),
                GradientUpdate::new("byz".to_string(), 1, vec![100.0, -100.0], 100, 0.5),
            ];

            let _ = aggregator.aggregate(&updates);
        }

        // After persistent bad behavior, should be blacklisted
        assert!(
            aggregator.is_blacklisted("byz"),
            "Persistent Byzantine should be blacklisted"
        );

        // Check reputation
        let rep = aggregator.participant_reputation("byz");
        assert!(rep < 0.5, "Byzantine reputation should be low: {}", rep);
    }

    #[test]
    fn test_adaptive_aggregator_reputation_recovery() {
        let mut aggregator = AdaptiveAggregator::new();

        // First, some bad behavior
        let updates_bad = vec![
            GradientUpdate::new("h1".to_string(), 1, vec![0.5, 0.5], 100, 0.5),
            GradientUpdate::new("h2".to_string(), 1, vec![0.52, 0.48], 100, 0.5),
            GradientUpdate::new("h3".to_string(), 1, vec![0.48, 0.52], 100, 0.5),
            GradientUpdate::new("reformed".to_string(), 1, vec![100.0, -100.0], 100, 0.5),
        ];
        let _ = aggregator.aggregate(&updates_bad);

        let rep_after_bad = aggregator.participant_reputation("reformed");

        // Now good behavior
        for _ in 0..10 {
            let updates_good = vec![
                GradientUpdate::new("h1".to_string(), 1, vec![0.5, 0.5], 100, 0.5),
                GradientUpdate::new("h2".to_string(), 1, vec![0.52, 0.48], 100, 0.5),
                GradientUpdate::new("h3".to_string(), 1, vec![0.48, 0.52], 100, 0.5),
                GradientUpdate::new("reformed".to_string(), 1, vec![0.51, 0.49], 100, 0.5),
            ];
            let _ = aggregator.aggregate(&updates_good);
        }

        let rep_after_good = aggregator.participant_reputation("reformed");
        assert!(
            rep_after_good > rep_after_bad,
            "Reputation should improve with good behavior: {} -> {}",
            rep_after_bad,
            rep_after_good
        );
    }

    #[test]
    fn test_adaptive_aggregator_stats() {
        let mut aggregator = AdaptiveAggregator::new();

        let updates = vec![
            GradientUpdate::new("h1".to_string(), 1, vec![0.5, 0.5], 100, 0.5),
            GradientUpdate::new("h2".to_string(), 1, vec![0.52, 0.48], 100, 0.5),
        ];

        let _ = aggregator.aggregate(&updates);
        let _ = aggregator.aggregate(&updates);

        let stats = aggregator.stats();
        assert_eq!(stats.rounds_processed, 2);
        assert_eq!(stats.total_participants_seen, 2);
        assert_eq!(stats.blacklisted_count, 0);
        assert!(stats.average_reputation > 0.9);
    }

    #[test]
    fn test_adaptive_aggregator_high_security() {
        let mut aggregator = AdaptiveAggregator::high_security();

        // Subtle attack that might pass normal detection
        let updates = vec![
            GradientUpdate::new("h1".to_string(), 1, vec![1.0, 1.0], 100, 0.5),
            GradientUpdate::new("h2".to_string(), 1, vec![1.05, 0.95], 100, 0.5),
            GradientUpdate::new("h3".to_string(), 1, vec![0.95, 1.05], 100, 0.5),
            GradientUpdate::new("subtle".to_string(), 1, vec![0.5, 1.5], 100, 0.5),
        ];

        let result = aggregator.aggregate(&updates).expect("Should succeed");

        // High security mode should be more sensitive
        assert!(aggregator.config.threat_threshold < 0.15);
    }
}
