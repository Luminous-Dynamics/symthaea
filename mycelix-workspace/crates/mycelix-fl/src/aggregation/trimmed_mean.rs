//! Trimmed Mean aggregation — delegates to mycelix-fl-core

use crate::types::{AggregationError, GradientUpdate};

/// Trimmed Mean: removes top and bottom percentile per dimension
///
/// Robust to Byzantine participants up to trim_percentage.
/// Delegates to `mycelix_fl_core::aggregation::trimmed_mean` with error conversion.
pub fn trimmed_mean(
    updates: &[GradientUpdate],
    trim_percentage: f32,
) -> Result<Vec<f32>, AggregationError> {
    mycelix_fl_core::aggregation::trimmed_mean(updates, trim_percentage).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trimmed_mean_removes_outliers() {
        let u1 = GradientUpdate::new("p1".to_string(), 1, vec![1.0], 100, 0.5);
        let u2 = GradientUpdate::new("p2".to_string(), 1, vec![1.1], 100, 0.5);
        let u3 = GradientUpdate::new("p3".to_string(), 1, vec![0.9], 100, 0.5);
        let u4 = GradientUpdate::new("p4".to_string(), 1, vec![100.0], 100, 0.5); // outlier

        let result = trimmed_mean(&[u1, u2, u3, u4], 0.25).unwrap();
        // After trimming 1 from each end, middle values average ~1.0
        assert!((result[0] - 1.0).abs() < 0.2);
    }
}
