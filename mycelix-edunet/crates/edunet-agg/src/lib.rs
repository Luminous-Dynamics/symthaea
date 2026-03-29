// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # EduNet Aggregation
//!
//! Robust aggregation algorithms for federated learning that resist poisoning attacks.
//!
//! This crate provides privacy-preserving and Byzantine-robust aggregation methods
//! for federated learning in decentralized educational systems.
//!
//! ## Features
//!
//! - **Trimmed mean aggregation**: Trims outliers before averaging (configurable breakdown point)
//! - **Median aggregation**: Maximum robustness (50% breakdown point)
//! - **Weighted aggregation**: Incorporates sample counts or reputation scores
//! - **Gradient clipping**: L2 norm clipping for differential privacy
//!
//! ## Example
//!
//! ```
//! use edunet_agg::{trimmed_mean, AggregationConfig};
//!
//! // Model updates from 4 participants
//! let updates = vec![
//!     vec![1.0, 2.0],
//!     vec![2.0, 3.0],
//!     vec![3.0, 4.0],
//!     vec![100.0, 200.0], // Malicious outlier
//! ];
//!
//! let config = AggregationConfig {
//!     trim_percent: 0.25, // Trim top and bottom 25%
//!     min_updates: 3,
//! };
//!
//! // Aggregation ignores the outlier
//! let result = trimmed_mean(&updates, &config).unwrap();
//! assert!((result[0] - 2.5).abs() < 0.1);
//! ```
//!
//! ## Security Considerations
//!
//! - **Poisoning attacks**: Use trimmed_mean or median for Byzantine robustness
//! - **Privacy**: Apply L2 norm clipping before aggregation
//! - **Sybil attacks**: Combine with weighted aggregation based on reputation
//!
//! ## Performance
//!
//! - Trimmed mean: O(n log n) per dimension (sorting)
//! - Median: O(n log n) per dimension (sorting)
//! - Weighted mean: O(n) per dimension
//! - Gradient clipping: O(d) where d is vector dimension

pub mod errors;
pub mod methods;

pub use errors::*;
pub use methods::*;

/// Configuration for aggregation methods.
///
/// Controls robustness-efficiency tradeoffs in trimmed mean aggregation.
///
/// # Examples
///
/// ```
/// use edunet_agg::AggregationConfig;
///
/// // Conservative: trim 20% from each end
/// let conservative = AggregationConfig {
///     trim_percent: 0.2,
///     min_updates: 5,
/// };
///
/// // Default: trim 10% from each end
/// let default = AggregationConfig::default();
/// assert_eq!(default.trim_percent, 0.1);
/// ```
#[derive(Debug, Clone)]
pub struct AggregationConfig {
    /// Trim percentage for trimmed mean (0.0 to 0.5).
    ///
    /// Higher values provide more robustness but less statistical efficiency.
    /// - `0.1`: 10% breakdown point (mild robustness)
    /// - `0.25`: 25% breakdown point (moderate robustness)
    /// - `0.4`: 40% breakdown point (strong robustness)
    pub trim_percent: f64,

    /// Minimum number of updates required for aggregation.
    ///
    /// Enforces a minimum sample size for statistical validity.
    /// Should be at least `ceil(1 / (2 * trim_percent))` for meaningful trimming.
    pub min_updates: usize,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            trim_percent: 0.1, // Trim top and bottom 10%
            min_updates: 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AggregationConfig::default();
        assert_eq!(config.trim_percent, 0.1);
        assert_eq!(config.min_updates, 3);
    }
}
