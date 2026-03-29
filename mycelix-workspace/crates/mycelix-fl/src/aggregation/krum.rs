// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Krum aggregation algorithm — delegates to mycelix-fl-core

use crate::types::{AggregationError, GradientUpdate};

/// Krum: select the gradient closest to its neighbors
///
/// Tolerates up to (n-2)/2 Byzantine participants.
/// Computes pairwise L2 distances, scores each gradient by sum of
/// nearest (n-f-2) neighbors, selects the one with minimum score.
///
/// Delegates to `mycelix_fl_core::aggregation::krum` with error conversion.
/// Note: fl-core may error on invalid `num_select` where the previous
/// implementation silently clamped. Callers should validate num_select.
pub fn krum(updates: &[GradientUpdate], num_select: usize) -> Result<Vec<f32>, AggregationError> {
    // Preserve the clamping behavior from the original implementation
    let n = updates.len();
    let clamped = if n > 0 {
        num_select.min(n).max(1)
    } else {
        num_select
    };
    mycelix_fl_core::aggregation::krum(updates, clamped).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_krum_selects_honest() {
        let honest1 = GradientUpdate::new("h1".to_string(), 1, vec![1.0, 1.0], 100, 0.5);
        let honest2 = GradientUpdate::new("h2".to_string(), 1, vec![1.1, 0.9], 100, 0.5);
        let honest3 = GradientUpdate::new("h3".to_string(), 1, vec![0.9, 1.1], 100, 0.5);
        let bad = GradientUpdate::new("bad".to_string(), 1, vec![100.0, -100.0], 100, 0.5);

        let result = krum(&[honest1, honest2, honest3, bad], 1).unwrap();
        // Should select one of the honest gradients (close to [1.0, 1.0])
        assert!((result[0] - 1.0).abs() < 0.5);
        assert!((result[1] - 1.0).abs() < 0.5);
    }
}
