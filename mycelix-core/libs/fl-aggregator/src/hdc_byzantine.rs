//! HDC-native Byzantine-resistant aggregation algorithms.
//!
//! This module provides Byzantine fault tolerance algorithms specifically designed
//! for hyperdimensional computing (HDC) representations. Unlike traditional
//! gradient-based approaches that operate in Euclidean space, these algorithms
//! leverage the unique properties of hypervectors:
//!
//! - **High dimensionality** provides natural noise robustness
//! - **Similarity metrics** (cosine, Hamming) detect outliers efficiently
//! - **Bundling operations** aggregate without conversion to dense formats
//!
//! # Defense Algorithms
//!
//! - [`HdcDefense::HdcBundle`]: Simple bundling without Byzantine protection
//! - [`HdcDefense::SimilarityFilter`]: Reject hypervectors below similarity threshold
//! - [`HdcDefense::WeightedBundle`]: Weight contributions by similarity with softmax
//! - [`HdcDefense::HdcKrum`]: Krum algorithm adapted for hypervector similarities
//! - [`HdcDefense::HdcMultiKrum`]: Multi-Krum selecting top-k by similarity scores
//! - [`HdcDefense::HdcRobust`]: Combined filtering and weighted bundling
//!
//! # Example
//!
//! ```rust
//! use fl_aggregator::hdc_byzantine::{HdcByzantineAggregator, HdcDefenseConfig, HdcDefense};
//! use fl_aggregator::payload::Hypervector;
//!
//! // Create aggregator with similarity filtering
//! let config = HdcDefenseConfig::new(HdcDefense::SimilarityFilter { threshold: 0.5 });
//! let aggregator = HdcByzantineAggregator::new(config);
//!
//! // Aggregate hypervectors from multiple nodes
//! let hvs = vec![
//!     Hypervector::random(16384),
//!     Hypervector::random(16384),
//!     Hypervector::random(16384),
//! ];
//! let refs: Vec<_> = hvs.iter().collect();
//! let result = aggregator.aggregate(&refs).unwrap();
//! ```

use crate::error::{AggregatorError, Result};
use crate::payload::{BinaryHypervector, HdcAggregatable, Hypervector};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

// =============================================================================
// HdcDefense Enum
// =============================================================================

/// HDC-native Byzantine defense algorithm selection.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum HdcDefense {
    /// Simple bundling aggregation (no Byzantine protection).
    ///
    /// Uses standard HDC bundling (element-wise addition with normalization).
    /// Fast but vulnerable to Byzantine attacks.
    HdcBundle,

    /// Similarity-based filtering.
    ///
    /// Computes component-wise median hypervector, then rejects any hypervector
    /// with similarity below the threshold before bundling the remainder.
    SimilarityFilter {
        /// Minimum cosine/Hamming similarity to median (0.0 to 1.0).
        threshold: f32,
    },

    /// Weighted bundling with softmax temperature.
    ///
    /// Each hypervector's contribution is weighted by its similarity to the
    /// component-wise median. Softmax temperature controls weight distribution:
    /// - Lower temp (0.1): more concentrated on high-similarity vectors
    /// - Higher temp (1.0+): more uniform weighting
    WeightedBundle {
        /// Softmax temperature for weight computation.
        softmax_temp: f32,
    },

    /// Krum algorithm adapted for HDC.
    ///
    /// Selects the single hypervector with highest sum of similarities to its
    /// n-f-2 most similar neighbors. Uses similarity (not distance) metrics.
    HdcKrum {
        /// Maximum number of Byzantine nodes to tolerate.
        f: usize,
    },

    /// Multi-Krum adapted for HDC.
    ///
    /// Selects top-k hypervectors by Krum score, then bundles them.
    HdcMultiKrum {
        /// Maximum number of Byzantine nodes to tolerate.
        f: usize,
        /// Number of hypervectors to select and bundle.
        k: usize,
    },

    /// Combined robust aggregation.
    ///
    /// First filters by similarity threshold, then applies weighted bundling
    /// to the remaining hypervectors. Provides layered Byzantine protection.
    HdcRobust {
        /// Minimum similarity to median for inclusion.
        similarity_threshold: f32,
        /// Softmax temperature for weight computation.
        weight_temp: f32,
    },
}

impl Default for HdcDefense {
    fn default() -> Self {
        HdcDefense::SimilarityFilter { threshold: 0.5 }
    }
}

impl std::fmt::Display for HdcDefense {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HdcDefense::HdcBundle => write!(f, "HdcBundle"),
            HdcDefense::SimilarityFilter { threshold } => {
                write!(f, "SimilarityFilter(threshold={})", threshold)
            }
            HdcDefense::WeightedBundle { softmax_temp } => {
                write!(f, "WeightedBundle(temp={})", softmax_temp)
            }
            HdcDefense::HdcKrum { f: byzantine } => write!(f, "HdcKrum(f={})", byzantine),
            HdcDefense::HdcMultiKrum { f: byzantine, k } => {
                write!(f, "HdcMultiKrum(f={}, k={})", byzantine, k)
            }
            HdcDefense::HdcRobust {
                similarity_threshold,
                weight_temp,
            } => {
                write!(
                    f,
                    "HdcRobust(threshold={}, temp={})",
                    similarity_threshold, weight_temp
                )
            }
        }
    }
}

// =============================================================================
// HdcDefenseConfig
// =============================================================================

/// Configuration for HDC Byzantine defense algorithms.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HdcDefenseConfig {
    /// The defense algorithm to use.
    pub defense: HdcDefense,

    /// Minimum number of hypervectors required for aggregation.
    ///
    /// If None, uses algorithm-specific defaults.
    pub min_hypervectors: Option<usize>,
}

impl Default for HdcDefenseConfig {
    fn default() -> Self {
        Self {
            defense: HdcDefense::default(),
            min_hypervectors: None,
        }
    }
}

impl HdcDefenseConfig {
    /// Create a new config with the specified defense algorithm.
    pub fn new(defense: HdcDefense) -> Self {
        Self {
            defense,
            min_hypervectors: None,
        }
    }

    /// Create a high-security configuration.
    ///
    /// Uses HdcRobust with conservative thresholds for maximum Byzantine tolerance.
    pub fn high_security() -> Self {
        Self {
            defense: HdcDefense::HdcRobust {
                similarity_threshold: 0.6,
                weight_temp: 0.5,
            },
            min_hypervectors: Some(5),
        }
    }

    /// Create a fast configuration with minimal overhead.
    ///
    /// Uses simple bundling for maximum throughput when Byzantine tolerance
    /// is not required (e.g., trusted environments).
    pub fn fast() -> Self {
        Self {
            defense: HdcDefense::HdcBundle,
            min_hypervectors: Some(1),
        }
    }

    /// Set minimum hypervector count.
    pub fn with_min_hypervectors(mut self, min: usize) -> Self {
        self.min_hypervectors = Some(min);
        self
    }

    /// Get minimum hypervectors required for this defense.
    pub fn min_required(&self) -> usize {
        if let Some(min) = self.min_hypervectors {
            return min;
        }

        match &self.defense {
            HdcDefense::HdcBundle => 1,
            HdcDefense::SimilarityFilter { .. } => 3, // Need enough for meaningful median
            HdcDefense::WeightedBundle { .. } => 2,
            HdcDefense::HdcKrum { f } => 2 * f + 3,
            HdcDefense::HdcMultiKrum { f, .. } => 2 * f + 3,
            HdcDefense::HdcRobust { .. } => 3,
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Safe float comparison that handles NaN values.
fn safe_f32_cmp(a: &f32, b: &f32) -> Ordering {
    match a.partial_cmp(b) {
        Some(ord) => ord,
        None => {
            tracing::warn!(
                "NaN value encountered in HDC Byzantine comparison (a={}, b={})",
                a,
                b
            );
            Ordering::Equal
        }
    }
}

/// Compute component-wise median of hypervectors.
///
/// For each dimension, computes the median value across all input hypervectors.
/// This is robust to outliers and provides a stable reference point.
pub fn component_median(hvs: &[&Hypervector]) -> Option<Hypervector> {
    if hvs.is_empty() {
        return None;
    }

    let dim = hvs[0].components.len();
    if !hvs.iter().all(|hv| hv.components.len() == dim) {
        return None;
    }

    let n = hvs.len();
    let mut result = vec![0i8; dim];

    for d in 0..dim {
        let mut values: Vec<i8> = hvs.iter().map(|hv| hv.components[d]).collect();
        values.sort();

        result[d] = if n % 2 == 0 {
            // Average of two middle values (rounded)
            let mid = n / 2;
            ((values[mid - 1] as i16 + values[mid] as i16) / 2) as i8
        } else {
            values[n / 2]
        };
    }

    Some(Hypervector::new(result))
}

/// Compute component-wise median of binary hypervectors.
///
/// Uses majority voting for each bit position.
pub fn component_median_binary(hvs: &[&BinaryHypervector]) -> Option<BinaryHypervector> {
    if hvs.is_empty() {
        return None;
    }

    let bit_dim = hvs[0].bit_dimension;
    if !hvs.iter().all(|hv| hv.bit_dimension == bit_dim) {
        return None;
    }

    // Majority voting is equivalent to median for binary values
    BinaryHypervector::bundle(hvs)
}

/// Compute softmax of a slice of f32 values.
///
/// Applies temperature scaling before normalization.
pub fn softmax(values: &[f32], temperature: f32) -> Vec<f32> {
    if values.is_empty() {
        return vec![];
    }

    let temp = temperature.max(1e-10); // Prevent division by zero

    // Scale by temperature and find max for numerical stability
    let scaled: Vec<f32> = values.iter().map(|v| v / temp).collect();
    let max_val = scaled
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) for stability
    let exp_vals: Vec<f32> = scaled.iter().map(|v| (v - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();

    if sum < 1e-10 {
        // Uniform distribution if sum is too small
        let uniform = 1.0 / values.len() as f32;
        return vec![uniform; values.len()];
    }

    exp_vals.iter().map(|e| e / sum).collect()
}

/// Bundle hypervectors with specified weights.
///
/// Each hypervector's contribution is scaled by its weight before summation.
pub fn weighted_bundle(hvs: &[&Hypervector], weights: &[f32]) -> Option<Hypervector> {
    if hvs.is_empty() || hvs.len() != weights.len() {
        return None;
    }

    let dim = hvs[0].components.len();
    if !hvs.iter().all(|hv| hv.components.len() == dim) {
        return None;
    }

    // Weighted sum using f32 for precision
    let mut sums: Vec<f32> = vec![0.0; dim];

    for (hv, &weight) in hvs.iter().zip(weights.iter()) {
        for (i, &c) in hv.components.iter().enumerate() {
            sums[i] += c as f32 * weight;
        }
    }

    // Convert back to i8, clamping to valid range
    let components: Vec<i8> = sums
        .iter()
        .map(|&s| s.round().clamp(-128.0, 127.0) as i8)
        .collect();

    Some(Hypervector::new(components))
}

/// Bundle binary hypervectors with specified weights.
///
/// Uses weighted voting: each bit is set if weighted sum exceeds threshold.
pub fn weighted_bundle_binary(hvs: &[&BinaryHypervector], weights: &[f32]) -> Option<BinaryHypervector> {
    if hvs.is_empty() || hvs.len() != weights.len() {
        return None;
    }

    let bit_dim = hvs[0].bit_dimension;
    if !hvs.iter().all(|hv| hv.bit_dimension == bit_dim) {
        return None;
    }

    let total_weight: f32 = weights.iter().sum();
    let threshold = total_weight / 2.0;

    let byte_count = (bit_dim + 7) / 8;
    let mut data = vec![0u8; byte_count];

    for bit_idx in 0..bit_dim {
        let weighted_sum: f32 = hvs
            .iter()
            .zip(weights.iter())
            .filter(|(hv, _)| hv.get_bit(bit_idx).unwrap_or(false))
            .map(|(_, w)| w)
            .sum();

        if weighted_sum > threshold {
            data[bit_idx / 8] |= 1 << (bit_idx % 8);
        }
    }

    Some(BinaryHypervector::new(data, bit_dim))
}

// =============================================================================
// HdcByzantineAggregator
// =============================================================================

/// HDC-native Byzantine-resistant aggregator.
///
/// This aggregator operates directly on hypervector representations without
/// converting to dense gradient formats. It leverages HDC similarity metrics
/// for Byzantine detection and bundling operations for aggregation.
pub struct HdcByzantineAggregator {
    config: HdcDefenseConfig,
}

impl HdcByzantineAggregator {
    /// Create a new HDC Byzantine aggregator with the given configuration.
    pub fn new(config: HdcDefenseConfig) -> Self {
        Self { config }
    }

    /// Get the defense configuration.
    pub fn config(&self) -> &HdcDefenseConfig {
        &self.config
    }

    /// Aggregate hypervectors using the configured defense algorithm.
    pub fn aggregate(&self, hvs: &[&Hypervector]) -> Result<Hypervector> {
        let n = hvs.len();
        let min_required = self.config.min_required();

        if n < min_required {
            return Err(AggregatorError::InsufficientGradients {
                have: n,
                need: min_required,
                defense: self.config.defense.to_string(),
            });
        }

        // Validate dimension consistency
        if !hvs.is_empty() {
            let dim = hvs[0].components.len();
            for hv in hvs.iter() {
                if hv.components.len() != dim {
                    return Err(AggregatorError::DimensionMismatch {
                        expected: dim,
                        got: hv.components.len(),
                    });
                }
            }
        }

        match &self.config.defense {
            HdcDefense::HdcBundle => self.hdc_bundle(hvs),
            HdcDefense::SimilarityFilter { threshold } => {
                self.similarity_filter(hvs, *threshold)
            }
            HdcDefense::WeightedBundle { softmax_temp } => {
                self.weighted_bundle_aggregate(hvs, *softmax_temp)
            }
            HdcDefense::HdcKrum { f } => self.hdc_krum(hvs, *f),
            HdcDefense::HdcMultiKrum { f, k } => self.hdc_multi_krum(hvs, *f, *k),
            HdcDefense::HdcRobust {
                similarity_threshold,
                weight_temp,
            } => self.hdc_robust(hvs, *similarity_threshold, *weight_temp),
        }
    }

    /// Aggregate binary hypervectors using the configured defense algorithm.
    pub fn aggregate_binary(&self, hvs: &[&BinaryHypervector]) -> Result<BinaryHypervector> {
        let n = hvs.len();
        let min_required = self.config.min_required();

        if n < min_required {
            return Err(AggregatorError::InsufficientGradients {
                have: n,
                need: min_required,
                defense: self.config.defense.to_string(),
            });
        }

        // Validate dimension consistency
        if !hvs.is_empty() {
            let dim = hvs[0].bit_dimension;
            for hv in hvs.iter() {
                if hv.bit_dimension != dim {
                    return Err(AggregatorError::DimensionMismatch {
                        expected: dim,
                        got: hv.bit_dimension,
                    });
                }
            }
        }

        match &self.config.defense {
            HdcDefense::HdcBundle => self.hdc_bundle_binary(hvs),
            HdcDefense::SimilarityFilter { threshold } => {
                self.similarity_filter_binary(hvs, *threshold)
            }
            HdcDefense::WeightedBundle { softmax_temp } => {
                self.weighted_bundle_binary_aggregate(hvs, *softmax_temp)
            }
            HdcDefense::HdcKrum { f } => self.hdc_krum_binary(hvs, *f),
            HdcDefense::HdcMultiKrum { f, k } => self.hdc_multi_krum_binary(hvs, *f, *k),
            HdcDefense::HdcRobust {
                similarity_threshold,
                weight_temp,
            } => self.hdc_robust_binary(hvs, *similarity_threshold, *weight_temp),
        }
    }

    // =========================================================================
    // Hypervector (i8) Algorithms
    // =========================================================================

    /// Simple bundling without protection.
    fn hdc_bundle(&self, hvs: &[&Hypervector]) -> Result<Hypervector> {
        Hypervector::bundle(hvs).ok_or_else(|| {
            AggregatorError::Internal("Failed to bundle hypervectors".to_string())
        })
    }

    /// Similarity filtering: reject HVs below threshold similarity to median.
    fn similarity_filter(&self, hvs: &[&Hypervector], threshold: f32) -> Result<Hypervector> {
        // Compute component-wise median as reference
        let median = component_median(hvs).ok_or_else(|| {
            AggregatorError::Internal("Failed to compute median hypervector".to_string())
        })?;

        // Filter by similarity
        let filtered: Vec<&Hypervector> = hvs
            .iter()
            .filter(|hv| hv.cosine_similarity(&median) >= threshold)
            .copied()
            .collect();

        let rejected = hvs.len() - filtered.len();
        if rejected > 0 {
            tracing::debug!(
                "SimilarityFilter: rejected {}/{} hypervectors below threshold {}",
                rejected,
                hvs.len(),
                threshold
            );
        }

        if filtered.is_empty() {
            return Err(AggregatorError::Internal(
                "All hypervectors rejected by similarity filter".to_string(),
            ));
        }

        Hypervector::bundle(&filtered).ok_or_else(|| {
            AggregatorError::Internal("Failed to bundle filtered hypervectors".to_string())
        })
    }

    /// Weighted bundling: weight by similarity with softmax.
    fn weighted_bundle_aggregate(&self, hvs: &[&Hypervector], temp: f32) -> Result<Hypervector> {
        // Compute component-wise median as reference
        let median = component_median(hvs).ok_or_else(|| {
            AggregatorError::Internal("Failed to compute median hypervector".to_string())
        })?;

        // Compute similarities
        let similarities: Vec<f32> = hvs.iter().map(|hv| hv.cosine_similarity(&median)).collect();

        // Apply softmax to get weights
        let weights = softmax(&similarities, temp);

        tracing::debug!(
            "WeightedBundle: similarities={:?}, weights={:?}",
            similarities,
            weights
        );

        weighted_bundle(hvs, &weights).ok_or_else(|| {
            AggregatorError::Internal("Failed to compute weighted bundle".to_string())
        })
    }

    /// HDC Krum: select HV with highest sum of similarities to n-f-2 nearest.
    fn hdc_krum(&self, hvs: &[&Hypervector], f: usize) -> Result<Hypervector> {
        let n = hvs.len();

        if n <= 2 * f + 2 {
            return Err(AggregatorError::InsufficientGradients {
                have: n,
                need: 2 * f + 3,
                defense: format!("HdcKrum(f={})", f),
            });
        }

        let scores = self.compute_hdc_krum_scores(hvs, f);

        // Find hypervector with highest score (highest similarity sum = most central)
        let best_idx = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| safe_f32_cmp(a, b))
            .map(|(i, _)| i)
            .ok_or_else(|| {
                AggregatorError::Internal("No hypervectors to select from in HdcKrum".to_string())
            })?;

        tracing::debug!(
            "HdcKrum selected hypervector {} with score {:.4}",
            best_idx,
            scores[best_idx]
        );

        Ok(hvs[best_idx].clone())
    }

    /// HDC Multi-Krum: select top-k by Krum score, then bundle.
    fn hdc_multi_krum(&self, hvs: &[&Hypervector], f: usize, k: usize) -> Result<Hypervector> {
        let n = hvs.len();

        if n <= 2 * f + 2 {
            return Err(AggregatorError::InsufficientGradients {
                have: n,
                need: 2 * f + 3,
                defense: format!("HdcMultiKrum(f={}, k={})", f, k),
            });
        }

        if k > n - f {
            return Err(AggregatorError::InvalidConfig(format!(
                "HdcMultiKrum k={} > n-f={}",
                k,
                n - f
            )));
        }

        let scores = self.compute_hdc_krum_scores(hvs, f);

        // Sort indices by score (descending - highest similarity is best)
        let mut indexed_scores: Vec<_> = scores.iter().enumerate().collect();
        indexed_scores.sort_by(|(_, a), (_, b)| safe_f32_cmp(b, a)); // Descending

        // Select top k
        let selected: Vec<&Hypervector> = indexed_scores
            .iter()
            .take(k)
            .map(|(i, _)| hvs[*i])
            .collect();

        tracing::debug!(
            "HdcMultiKrum selected {} hypervectors with scores: {:?}",
            k,
            indexed_scores
                .iter()
                .take(k)
                .map(|(i, s)| format!("{}({:.2})", i, s))
                .collect::<Vec<_>>()
        );

        Hypervector::bundle(&selected).ok_or_else(|| {
            AggregatorError::Internal("Failed to bundle selected hypervectors".to_string())
        })
    }

    /// Compute HDC Krum scores (sum of similarities to k nearest neighbors).
    fn compute_hdc_krum_scores(&self, hvs: &[&Hypervector], f: usize) -> Vec<f32> {
        let n = hvs.len();
        let take_count = n.saturating_sub(f).saturating_sub(2);

        hvs.iter()
            .enumerate()
            .map(|(i, hvi)| {
                // Compute similarities to all other hypervectors
                let mut similarities: Vec<f32> = hvs
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, hvj)| hvi.cosine_similarity(hvj))
                    .collect();

                // Sort descending (highest similarity first) and sum top n-f-2
                similarities.sort_by(|a, b| safe_f32_cmp(b, a));
                similarities.iter().take(take_count).sum()
            })
            .collect()
    }

    /// HDC Robust: filter by similarity, then weighted bundle.
    fn hdc_robust(
        &self,
        hvs: &[&Hypervector],
        threshold: f32,
        temp: f32,
    ) -> Result<Hypervector> {
        // Compute component-wise median
        let median = component_median(hvs).ok_or_else(|| {
            AggregatorError::Internal("Failed to compute median hypervector".to_string())
        })?;

        // Filter by similarity threshold
        let filtered_with_sim: Vec<(&Hypervector, f32)> = hvs
            .iter()
            .map(|hv| (*hv, hv.cosine_similarity(&median)))
            .filter(|(_, sim)| *sim >= threshold)
            .collect();

        let rejected = hvs.len() - filtered_with_sim.len();
        if rejected > 0 {
            tracing::debug!(
                "HdcRobust: rejected {}/{} hypervectors below threshold {}",
                rejected,
                hvs.len(),
                threshold
            );
        }

        if filtered_with_sim.is_empty() {
            return Err(AggregatorError::Internal(
                "All hypervectors rejected by HdcRobust filter".to_string(),
            ));
        }

        // Apply weighted bundling to filtered set
        let filtered_hvs: Vec<&Hypervector> = filtered_with_sim.iter().map(|(hv, _)| *hv).collect();
        let similarities: Vec<f32> = filtered_with_sim.iter().map(|(_, sim)| *sim).collect();
        let weights = softmax(&similarities, temp);

        weighted_bundle(&filtered_hvs, &weights).ok_or_else(|| {
            AggregatorError::Internal("Failed to compute weighted bundle in HdcRobust".to_string())
        })
    }

    // =========================================================================
    // Binary Hypervector Algorithms
    // =========================================================================

    /// Simple bundling for binary hypervectors.
    fn hdc_bundle_binary(&self, hvs: &[&BinaryHypervector]) -> Result<BinaryHypervector> {
        BinaryHypervector::bundle(hvs).ok_or_else(|| {
            AggregatorError::Internal("Failed to bundle binary hypervectors".to_string())
        })
    }

    /// Similarity filtering for binary hypervectors.
    fn similarity_filter_binary(
        &self,
        hvs: &[&BinaryHypervector],
        threshold: f32,
    ) -> Result<BinaryHypervector> {
        let median = component_median_binary(hvs).ok_or_else(|| {
            AggregatorError::Internal("Failed to compute median binary hypervector".to_string())
        })?;

        let filtered: Vec<&BinaryHypervector> = hvs
            .iter()
            .filter(|hv| hv.hamming_similarity(&median) >= threshold)
            .copied()
            .collect();

        let rejected = hvs.len() - filtered.len();
        if rejected > 0 {
            tracing::debug!(
                "SimilarityFilter (binary): rejected {}/{} below threshold {}",
                rejected,
                hvs.len(),
                threshold
            );
        }

        if filtered.is_empty() {
            return Err(AggregatorError::Internal(
                "All binary hypervectors rejected by similarity filter".to_string(),
            ));
        }

        BinaryHypervector::bundle(&filtered).ok_or_else(|| {
            AggregatorError::Internal("Failed to bundle filtered binary hypervectors".to_string())
        })
    }

    /// Weighted bundling for binary hypervectors.
    fn weighted_bundle_binary_aggregate(
        &self,
        hvs: &[&BinaryHypervector],
        temp: f32,
    ) -> Result<BinaryHypervector> {
        let median = component_median_binary(hvs).ok_or_else(|| {
            AggregatorError::Internal("Failed to compute median binary hypervector".to_string())
        })?;

        let similarities: Vec<f32> = hvs
            .iter()
            .map(|hv| hv.hamming_similarity(&median))
            .collect();
        let weights = softmax(&similarities, temp);

        weighted_bundle_binary(hvs, &weights).ok_or_else(|| {
            AggregatorError::Internal("Failed to compute weighted binary bundle".to_string())
        })
    }

    /// HDC Krum for binary hypervectors.
    fn hdc_krum_binary(&self, hvs: &[&BinaryHypervector], f: usize) -> Result<BinaryHypervector> {
        let n = hvs.len();

        if n <= 2 * f + 2 {
            return Err(AggregatorError::InsufficientGradients {
                have: n,
                need: 2 * f + 3,
                defense: format!("HdcKrum(f={}) [binary]", f),
            });
        }

        let scores = self.compute_hdc_krum_scores_binary(hvs, f);

        let best_idx = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| safe_f32_cmp(a, b))
            .map(|(i, _)| i)
            .ok_or_else(|| {
                AggregatorError::Internal(
                    "No binary hypervectors to select from in HdcKrum".to_string(),
                )
            })?;

        tracing::debug!(
            "HdcKrum (binary) selected {} with score {:.4}",
            best_idx,
            scores[best_idx]
        );

        Ok(hvs[best_idx].clone())
    }

    /// HDC Multi-Krum for binary hypervectors.
    fn hdc_multi_krum_binary(
        &self,
        hvs: &[&BinaryHypervector],
        f: usize,
        k: usize,
    ) -> Result<BinaryHypervector> {
        let n = hvs.len();

        if n <= 2 * f + 2 {
            return Err(AggregatorError::InsufficientGradients {
                have: n,
                need: 2 * f + 3,
                defense: format!("HdcMultiKrum(f={}, k={}) [binary]", f, k),
            });
        }

        if k > n - f {
            return Err(AggregatorError::InvalidConfig(format!(
                "HdcMultiKrum k={} > n-f={} [binary]",
                k,
                n - f
            )));
        }

        let scores = self.compute_hdc_krum_scores_binary(hvs, f);

        let mut indexed_scores: Vec<_> = scores.iter().enumerate().collect();
        indexed_scores.sort_by(|(_, a), (_, b)| safe_f32_cmp(b, a));

        let selected: Vec<&BinaryHypervector> = indexed_scores
            .iter()
            .take(k)
            .map(|(i, _)| hvs[*i])
            .collect();

        BinaryHypervector::bundle(&selected).ok_or_else(|| {
            AggregatorError::Internal("Failed to bundle selected binary hypervectors".to_string())
        })
    }

    /// Compute HDC Krum scores for binary hypervectors.
    fn compute_hdc_krum_scores_binary(&self, hvs: &[&BinaryHypervector], f: usize) -> Vec<f32> {
        let n = hvs.len();
        let take_count = n.saturating_sub(f).saturating_sub(2);

        hvs.iter()
            .enumerate()
            .map(|(i, hvi)| {
                let mut similarities: Vec<f32> = hvs
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, hvj)| hvi.hamming_similarity(hvj))
                    .collect();

                similarities.sort_by(|a, b| safe_f32_cmp(b, a));
                similarities.iter().take(take_count).sum()
            })
            .collect()
    }

    /// HDC Robust for binary hypervectors.
    fn hdc_robust_binary(
        &self,
        hvs: &[&BinaryHypervector],
        threshold: f32,
        temp: f32,
    ) -> Result<BinaryHypervector> {
        let median = component_median_binary(hvs).ok_or_else(|| {
            AggregatorError::Internal("Failed to compute median binary hypervector".to_string())
        })?;

        let filtered_with_sim: Vec<(&BinaryHypervector, f32)> = hvs
            .iter()
            .map(|hv| (*hv, hv.hamming_similarity(&median)))
            .filter(|(_, sim)| *sim >= threshold)
            .collect();

        if filtered_with_sim.is_empty() {
            return Err(AggregatorError::Internal(
                "All binary hypervectors rejected by HdcRobust filter".to_string(),
            ));
        }

        let filtered_hvs: Vec<&BinaryHypervector> =
            filtered_with_sim.iter().map(|(hv, _)| *hv).collect();
        let similarities: Vec<f32> = filtered_with_sim.iter().map(|(_, sim)| *sim).collect();
        let weights = softmax(&similarities, temp);

        weighted_bundle_binary(&filtered_hvs, &weights).ok_or_else(|| {
            AggregatorError::Internal(
                "Failed to compute weighted binary bundle in HdcRobust".to_string(),
            )
        })
    }
}

// =============================================================================
// Generic trait implementation for HdcAggregatable types
// =============================================================================

/// Extension trait for aggregating any HdcAggregatable type with Byzantine defense.
pub trait HdcByzantineAggregate: HdcAggregatable {
    /// Aggregate with Byzantine defense using the provided aggregator.
    fn aggregate_byzantine(
        items: &[&Self],
        aggregator: &HdcByzantineAggregator,
    ) -> Result<Self>
    where
        Self: Sized;
}

impl HdcByzantineAggregate for Hypervector {
    fn aggregate_byzantine(
        items: &[&Self],
        aggregator: &HdcByzantineAggregator,
    ) -> Result<Self> {
        aggregator.aggregate(items)
    }
}

impl HdcByzantineAggregate for BinaryHypervector {
    fn aggregate_byzantine(
        items: &[&Self],
        aggregator: &HdcByzantineAggregator,
    ) -> Result<Self> {
        aggregator.aggregate_binary(items)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create test hypervectors with known values
    fn make_hv(seed: i8, dim: usize) -> Hypervector {
        let components: Vec<i8> = (0..dim)
            .map(|i| ((seed as i32 + i as i32) % 256 - 128).clamp(-128, 127) as i8)
            .collect();
        Hypervector::new(components)
    }

    fn make_similar_hvs(base: &Hypervector, count: usize, noise: i8) -> Vec<Hypervector> {
        (0..count)
            .map(|i| {
                let components: Vec<i8> = base
                    .components
                    .iter()
                    .map(|&c| {
                        let offset = ((i as i8).wrapping_mul(noise)) % 10;
                        (c as i16 + offset as i16).clamp(-128, 127) as i8
                    })
                    .collect();
                Hypervector::new(components)
            })
            .collect()
    }

    fn make_byzantine_hv(dim: usize) -> Hypervector {
        // Create a hypervector that's very different (inverted)
        let components: Vec<i8> = (0..dim).map(|_| 127).collect();
        Hypervector::new(components)
    }

    // =========================================================================
    // Helper function tests
    // =========================================================================

    #[test]
    fn test_softmax() {
        let values = vec![1.0, 2.0, 3.0];
        let weights = softmax(&values, 1.0);

        assert_eq!(weights.len(), 3);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(weights[2] > weights[1]);
        assert!(weights[1] > weights[0]);
    }

    #[test]
    fn test_softmax_temperature() {
        let values = vec![1.0, 2.0, 3.0];

        // Low temperature: more concentrated
        let low_temp = softmax(&values, 0.1);
        // High temperature: more uniform
        let high_temp = softmax(&values, 10.0);

        // Low temp should have larger difference between max and min
        let low_diff = low_temp[2] - low_temp[0];
        let high_diff = high_temp[2] - high_temp[0];
        assert!(low_diff > high_diff);
    }

    #[test]
    fn test_softmax_empty() {
        let weights = softmax(&[], 1.0);
        assert!(weights.is_empty());
    }

    #[test]
    fn test_component_median() {
        let hv1 = Hypervector::new(vec![10, 20, 30]);
        let hv2 = Hypervector::new(vec![12, 22, 32]);
        let hv3 = Hypervector::new(vec![14, 24, 34]);

        let median = component_median(&[&hv1, &hv2, &hv3]).unwrap();

        assert_eq!(median.components, vec![12, 22, 32]);
    }

    #[test]
    fn test_component_median_even() {
        let hv1 = Hypervector::new(vec![10, 20]);
        let hv2 = Hypervector::new(vec![14, 24]);

        let median = component_median(&[&hv1, &hv2]).unwrap();

        // Average of 10 and 14 = 12, average of 20 and 24 = 22
        assert_eq!(median.components, vec![12, 22]);
    }

    #[test]
    fn test_weighted_bundle() {
        let hv1 = Hypervector::new(vec![100, 0]);
        let hv2 = Hypervector::new(vec![0, 100]);

        // Equal weights
        let result = weighted_bundle(&[&hv1, &hv2], &[0.5, 0.5]).unwrap();
        assert_eq!(result.components, vec![50, 50]);

        // Weighted towards hv1
        let result2 = weighted_bundle(&[&hv1, &hv2], &[0.8, 0.2]).unwrap();
        assert_eq!(result2.components, vec![80, 20]);
    }

    // =========================================================================
    // HdcBundle tests
    // =========================================================================

    #[test]
    fn test_hdc_bundle() {
        let config = HdcDefenseConfig::new(HdcDefense::HdcBundle);
        let aggregator = HdcByzantineAggregator::new(config);

        let hvs = vec![
            Hypervector::new(vec![10, 20, 30]),
            Hypervector::new(vec![20, 30, 40]),
            Hypervector::new(vec![30, 40, 50]),
        ];
        let refs: Vec<_> = hvs.iter().collect();

        let result = aggregator.aggregate(&refs).unwrap();

        // Average: 20, 30, 40
        assert_eq!(result.components, vec![20, 30, 40]);
    }

    // =========================================================================
    // SimilarityFilter tests
    // =========================================================================

    #[test]
    fn test_similarity_filter_rejects_byzantine() {
        let config = HdcDefenseConfig::new(HdcDefense::SimilarityFilter { threshold: 0.5 });
        let aggregator = HdcByzantineAggregator::new(config);

        // Create similar hypervectors
        let base = make_hv(0, 100);
        let similar_hvs = make_similar_hvs(&base, 4, 2);
        let byzantine = make_byzantine_hv(100);

        let mut all_hvs = similar_hvs.clone();
        all_hvs.push(byzantine.clone());

        let refs: Vec<_> = all_hvs.iter().collect();
        let result = aggregator.aggregate(&refs).unwrap();

        // Result should be similar to base, not byzantine
        let sim_to_base = result.cosine_similarity(&base);
        let sim_to_byzantine = result.cosine_similarity(&byzantine);

        assert!(
            sim_to_base > sim_to_byzantine,
            "Result should be more similar to honest nodes than byzantine"
        );
    }

    #[test]
    fn test_similarity_filter_all_rejected() {
        let config = HdcDefenseConfig::new(HdcDefense::SimilarityFilter { threshold: 0.99 });
        let aggregator = HdcByzantineAggregator::new(config);

        // Create diverse hypervectors that won't all pass high threshold
        let hvs = vec![
            Hypervector::new(vec![100, 0, 0]),
            Hypervector::new(vec![0, 100, 0]),
            Hypervector::new(vec![0, 0, 100]),
        ];
        let refs: Vec<_> = hvs.iter().collect();

        let result = aggregator.aggregate(&refs);
        assert!(
            result.is_err(),
            "Should fail when all hypervectors are rejected"
        );
    }

    // =========================================================================
    // WeightedBundle tests
    // =========================================================================

    #[test]
    fn test_weighted_bundle_favors_similar() {
        let config = HdcDefenseConfig::new(HdcDefense::WeightedBundle { softmax_temp: 0.5 });
        let aggregator = HdcByzantineAggregator::new(config);

        // Three similar + one outlier
        let base = make_hv(50, 100);
        let similar_hvs = make_similar_hvs(&base, 3, 1);
        let outlier = make_byzantine_hv(100);

        let mut all_hvs = similar_hvs.clone();
        all_hvs.push(outlier.clone());

        let refs: Vec<_> = all_hvs.iter().collect();
        let result = aggregator.aggregate(&refs).unwrap();

        // Result should be closer to the similar group
        let sim_to_base = result.cosine_similarity(&base);
        assert!(
            sim_to_base > 0.5,
            "Result should be similar to the honest majority"
        );
    }

    // =========================================================================
    // HdcKrum tests
    // =========================================================================

    #[test]
    fn test_hdc_krum_selects_central() {
        let config = HdcDefenseConfig::new(HdcDefense::HdcKrum { f: 1 });
        let aggregator = HdcByzantineAggregator::new(config);

        // 5 nodes: 4 similar, 1 byzantine (requires n >= 2*f+3 = 5)
        let base = make_hv(30, 50);
        let mut hvs = make_similar_hvs(&base, 4, 2);
        hvs.push(make_byzantine_hv(50));

        let refs: Vec<_> = hvs.iter().collect();
        let result = aggregator.aggregate(&refs).unwrap();

        // Should select one of the similar hypervectors, not the byzantine
        let sim_to_base = result.cosine_similarity(&base);
        assert!(
            sim_to_base > 0.5,
            "Krum should select a hypervector similar to the honest majority"
        );
    }

    #[test]
    fn test_hdc_krum_insufficient_nodes() {
        let config = HdcDefenseConfig::new(HdcDefense::HdcKrum { f: 2 });
        let aggregator = HdcByzantineAggregator::new(config);

        // Need 2*2+3 = 7 nodes, only have 4
        let hvs: Vec<_> = (0..4).map(|i| make_hv(i as i8, 50)).collect();
        let refs: Vec<_> = hvs.iter().collect();

        let result = aggregator.aggregate(&refs);
        assert!(matches!(
            result,
            Err(AggregatorError::InsufficientGradients { .. })
        ));
    }

    // =========================================================================
    // HdcMultiKrum tests
    // =========================================================================

    #[test]
    fn test_hdc_multi_krum() {
        let config = HdcDefenseConfig::new(HdcDefense::HdcMultiKrum { f: 1, k: 3 });
        let aggregator = HdcByzantineAggregator::new(config);

        let base = make_hv(20, 50);
        let mut hvs = make_similar_hvs(&base, 4, 2);
        hvs.push(make_byzantine_hv(50));

        let refs: Vec<_> = hvs.iter().collect();
        let result = aggregator.aggregate(&refs).unwrap();

        // Result should be bundle of top 3, excluding byzantine
        let sim_to_base = result.cosine_similarity(&base);
        assert!(
            sim_to_base > 0.5,
            "MultiKrum bundle should be similar to honest majority"
        );
    }

    #[test]
    fn test_hdc_multi_krum_invalid_k() {
        let config = HdcDefenseConfig::new(HdcDefense::HdcMultiKrum { f: 1, k: 10 });
        let aggregator = HdcByzantineAggregator::new(config);

        let hvs: Vec<_> = (0..5).map(|i| make_hv(i as i8, 50)).collect();
        let refs: Vec<_> = hvs.iter().collect();

        let result = aggregator.aggregate(&refs);
        assert!(matches!(result, Err(AggregatorError::InvalidConfig(_))));
    }

    // =========================================================================
    // HdcRobust tests
    // =========================================================================

    #[test]
    fn test_hdc_robust() {
        let config = HdcDefenseConfig::new(HdcDefense::HdcRobust {
            similarity_threshold: 0.3,
            weight_temp: 0.5,
        });
        let aggregator = HdcByzantineAggregator::new(config);

        let base = make_hv(40, 100);
        let mut hvs = make_similar_hvs(&base, 4, 2);
        hvs.push(make_byzantine_hv(100));

        let refs: Vec<_> = hvs.iter().collect();
        let result = aggregator.aggregate(&refs).unwrap();

        let sim_to_base = result.cosine_similarity(&base);
        assert!(
            sim_to_base > 0.5,
            "HdcRobust should produce result similar to honest majority"
        );
    }

    // =========================================================================
    // Binary hypervector tests
    // =========================================================================

    #[test]
    fn test_binary_bundle() {
        let config = HdcDefenseConfig::new(HdcDefense::HdcBundle);
        let aggregator = HdcByzantineAggregator::new(config);

        let hvs = vec![
            BinaryHypervector::from_bits(&[true, true, false, false]),
            BinaryHypervector::from_bits(&[true, false, true, false]),
            BinaryHypervector::from_bits(&[true, true, true, false]),
        ];
        let refs: Vec<_> = hvs.iter().collect();

        let result = aggregator.aggregate_binary(&refs).unwrap();

        // Majority voting: true(3), true(2), true(2), false(0)
        assert_eq!(result.get_bit(0), Some(true));
        assert_eq!(result.get_bit(1), Some(true));
        assert_eq!(result.get_bit(2), Some(true));
        assert_eq!(result.get_bit(3), Some(false));
    }

    #[test]
    fn test_binary_similarity_filter() {
        let config = HdcDefenseConfig::new(HdcDefense::SimilarityFilter { threshold: 0.5 });
        let aggregator = HdcByzantineAggregator::new(config);

        // Create similar binary hypervectors
        let hv1 = BinaryHypervector::from_bits(&[true, true, true, true, false, false, false, false]);
        let hv2 = BinaryHypervector::from_bits(&[true, true, true, false, false, false, false, false]);
        let hv3 = BinaryHypervector::from_bits(&[true, true, false, true, false, false, false, false]);
        // Byzantine: all opposite
        let byzantine = BinaryHypervector::from_bits(&[false, false, false, false, true, true, true, true]);

        let hvs = vec![hv1, hv2, hv3, byzantine];
        let refs: Vec<_> = hvs.iter().collect();

        let result = aggregator.aggregate_binary(&refs).unwrap();

        // Should be more similar to honest group
        let sim_to_honest = result.hamming_similarity(&hvs[0]);
        let sim_to_byzantine = result.hamming_similarity(&hvs[3]);

        assert!(
            sim_to_honest > sim_to_byzantine,
            "Result should be closer to honest nodes"
        );
    }

    #[test]
    fn test_binary_krum() {
        let config = HdcDefenseConfig::new(HdcDefense::HdcKrum { f: 1 });
        let aggregator = HdcByzantineAggregator::new(config);

        // Create 5 binary hypervectors (need n >= 2*1+3 = 5)
        let base_bits: Vec<bool> = (0..64).map(|i| i % 3 != 0).collect();
        let hv1 = BinaryHypervector::from_bits(&base_bits);

        let mut hvs = vec![hv1.clone()];
        for i in 1..4 {
            let bits: Vec<bool> = base_bits
                .iter()
                .enumerate()
                .map(|(j, &b)| if j % 20 == i { !b } else { b })
                .collect();
            hvs.push(BinaryHypervector::from_bits(&bits));
        }
        // Byzantine: inverted
        let byzantine_bits: Vec<bool> = base_bits.iter().map(|&b| !b).collect();
        hvs.push(BinaryHypervector::from_bits(&byzantine_bits));

        let refs: Vec<_> = hvs.iter().collect();
        let result = aggregator.aggregate_binary(&refs).unwrap();

        let sim_to_base = result.hamming_similarity(&hvs[0]);
        assert!(sim_to_base > 0.5, "Krum should select honest hypervector");
    }

    // =========================================================================
    // Config tests
    // =========================================================================

    #[test]
    fn test_config_defaults() {
        let config = HdcDefenseConfig::default();
        assert!(matches!(
            config.defense,
            HdcDefense::SimilarityFilter { threshold: _ }
        ));
        assert_eq!(config.min_hypervectors, None);
    }

    #[test]
    fn test_config_high_security() {
        let config = HdcDefenseConfig::high_security();
        assert!(matches!(config.defense, HdcDefense::HdcRobust { .. }));
        assert_eq!(config.min_hypervectors, Some(5));
    }

    #[test]
    fn test_config_fast() {
        let config = HdcDefenseConfig::fast();
        assert!(matches!(config.defense, HdcDefense::HdcBundle));
        assert_eq!(config.min_hypervectors, Some(1));
    }

    #[test]
    fn test_min_required() {
        assert_eq!(
            HdcDefenseConfig::new(HdcDefense::HdcBundle).min_required(),
            1
        );
        assert_eq!(
            HdcDefenseConfig::new(HdcDefense::HdcKrum { f: 1 }).min_required(),
            5
        );
        assert_eq!(
            HdcDefenseConfig::new(HdcDefense::HdcKrum { f: 2 }).min_required(),
            7
        );
        assert_eq!(
            HdcDefenseConfig::new(HdcDefense::SimilarityFilter { threshold: 0.5 }).min_required(),
            3
        );
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = HdcDefenseConfig::new(HdcDefense::HdcBundle);
        let aggregator = HdcByzantineAggregator::new(config);

        let hvs = vec![
            Hypervector::new(vec![1, 2, 3]),
            Hypervector::new(vec![1, 2, 3, 4]), // Different dimension
        ];
        let refs: Vec<_> = hvs.iter().collect();

        let result = aggregator.aggregate(&refs);
        assert!(matches!(
            result,
            Err(AggregatorError::DimensionMismatch { .. })
        ));
    }

    // =========================================================================
    // Edge case tests
    // =========================================================================

    #[test]
    fn test_single_hypervector_bundle() {
        let config = HdcDefenseConfig::new(HdcDefense::HdcBundle);
        let aggregator = HdcByzantineAggregator::new(config);

        let hv = Hypervector::new(vec![10, 20, 30]);
        let refs = vec![&hv];

        let result = aggregator.aggregate(&refs).unwrap();
        assert_eq!(result.components, hv.components);
    }

    #[test]
    fn test_empty_input() {
        let config = HdcDefenseConfig::new(HdcDefense::HdcBundle);
        let aggregator = HdcByzantineAggregator::new(config);

        let refs: Vec<&Hypervector> = vec![];
        let result = aggregator.aggregate(&refs);

        assert!(matches!(
            result,
            Err(AggregatorError::InsufficientGradients { .. })
        ));
    }

    #[test]
    fn test_defense_display() {
        assert_eq!(format!("{}", HdcDefense::HdcBundle), "HdcBundle");
        assert_eq!(
            format!("{}", HdcDefense::SimilarityFilter { threshold: 0.5 }),
            "SimilarityFilter(threshold=0.5)"
        );
        assert_eq!(
            format!("{}", HdcDefense::HdcKrum { f: 2 }),
            "HdcKrum(f=2)"
        );
        assert_eq!(
            format!("{}", HdcDefense::HdcMultiKrum { f: 1, k: 3 }),
            "HdcMultiKrum(f=1, k=3)"
        );
    }
}
