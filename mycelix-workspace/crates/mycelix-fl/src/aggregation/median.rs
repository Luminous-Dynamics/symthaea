//! Coordinate-wise Median aggregation — delegates to mycelix-fl-core

use crate::types::{AggregationError, GradientUpdate};

/// Coordinate-wise Median: takes median per coordinate
///
/// Robust to up to 50% Byzantine participants.
/// Delegates to `mycelix_fl_core::aggregation::coordinate_median` with error conversion.
pub fn coordinate_median(updates: &[GradientUpdate]) -> Result<Vec<f32>, AggregationError> {
    mycelix_fl_core::aggregation::coordinate_median(updates).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_robust_to_outlier() {
        let updates: Vec<GradientUpdate> = vec![
            GradientUpdate::new("p1".to_string(), 1, vec![1.0], 100, 0.5),
            GradientUpdate::new("p2".to_string(), 1, vec![1.1], 100, 0.5),
            GradientUpdate::new("p3".to_string(), 1, vec![0.9], 100, 0.5),
            GradientUpdate::new("bad".to_string(), 1, vec![1000.0], 100, 0.5),
            GradientUpdate::new("bad2".to_string(), 1, vec![-1000.0], 100, 0.5),
        ];

        let result = coordinate_median(&updates).unwrap();
        assert!((result[0] - 1.0).abs() < 0.001);
    }
}
