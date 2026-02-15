//! Federated Averaging (FedAvg) — delegates to mycelix-fl-core

use crate::types::{AggregationError, GradientUpdate};

/// Standard Federated Averaging: batch-size weighted mean
///
/// Delegates to `mycelix_fl_core::aggregation::fedavg` with error conversion.
pub fn fedavg(updates: &[GradientUpdate]) -> Result<Vec<f32>, AggregationError> {
    mycelix_fl_core::aggregation::fedavg(updates).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fedavg_equal_weights() {
        let u1 = GradientUpdate::new("p1".to_string(), 1, vec![1.0, 2.0], 100, 0.5);
        let u2 = GradientUpdate::new("p2".to_string(), 1, vec![3.0, 4.0], 100, 0.5);
        let result = fedavg(&[u1, u2]).unwrap();
        assert!((result[0] - 2.0).abs() < 0.001);
        assert!((result[1] - 3.0).abs() < 0.001);
    }
}
