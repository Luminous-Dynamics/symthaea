// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rate limiting algorithm implementations

use serde::{Deserialize, Serialize};

/// Supported rate limiting algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Algorithm {
    /// Sliding window log - most accurate, higher memory
    SlidingWindowLog,
    /// Sliding window counter - good balance
    SlidingWindowCounter,
    /// Token bucket - allows bursts
    TokenBucket,
    /// Fixed window - simplest, boundary issues
    FixedWindow,
    /// Leaky bucket - constant output rate
    LeakyBucket,
}

impl Algorithm {
    /// Get the Lua script key for this algorithm
    pub fn script_key(&self) -> &'static str {
        match self {
            Algorithm::SlidingWindowLog => "sliding_window_log",
            Algorithm::SlidingWindowCounter => "sliding_window_counter",
            Algorithm::TokenBucket => "token_bucket",
            Algorithm::FixedWindow => "fixed_window",
            Algorithm::LeakyBucket => "leaky_bucket",
        }
    }

    /// Whether this algorithm supports partial consumption
    pub fn supports_partial(&self) -> bool {
        matches!(self, Algorithm::TokenBucket | Algorithm::LeakyBucket)
    }

    /// Memory usage relative to others (1-5 scale)
    pub fn memory_usage(&self) -> u8 {
        match self {
            Algorithm::SlidingWindowLog => 5,
            Algorithm::SlidingWindowCounter => 3,
            Algorithm::TokenBucket => 2,
            Algorithm::FixedWindow => 1,
            Algorithm::LeakyBucket => 2,
        }
    }

    /// Accuracy relative to others (1-5 scale)
    pub fn accuracy(&self) -> u8 {
        match self {
            Algorithm::SlidingWindowLog => 5,
            Algorithm::SlidingWindowCounter => 4,
            Algorithm::TokenBucket => 4,
            Algorithm::FixedWindow => 2,
            Algorithm::LeakyBucket => 5,
        }
    }
}

impl std::fmt::Display for Algorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.script_key())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_script_keys() {
        assert_eq!(Algorithm::SlidingWindowLog.script_key(), "sliding_window_log");
        assert_eq!(Algorithm::TokenBucket.script_key(), "token_bucket");
        assert_eq!(Algorithm::FixedWindow.script_key(), "fixed_window");
    }

    #[test]
    fn test_algorithm_properties() {
        assert!(Algorithm::TokenBucket.supports_partial());
        assert!(!Algorithm::FixedWindow.supports_partial());

        assert!(Algorithm::SlidingWindowLog.memory_usage() > Algorithm::FixedWindow.memory_usage());
        assert!(Algorithm::SlidingWindowLog.accuracy() > Algorithm::FixedWindow.accuracy());
    }
}
