//! Core FL types
//!
//! Canonical data structures for federated learning, using f32 precision.

use serde::{Deserialize, Serialize};

/// Gradient update metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GradientMetadata {
    /// Number of samples in this batch
    pub batch_size: u32,
    /// Training loss
    pub loss: f32,
    /// Optional accuracy metric
    pub accuracy: Option<f32>,
    /// Timestamp of the update (Unix seconds)
    pub timestamp: u64,
}

impl GradientMetadata {
    pub fn new(batch_size: u32, loss: f32) -> Self {
        Self {
            batch_size,
            loss,
            accuracy: None,
            timestamp: 0,
        }
    }

    pub fn with_accuracy(batch_size: u32, loss: f32, accuracy: f32) -> Self {
        Self {
            batch_size,
            loss,
            accuracy: Some(accuracy),
            timestamp: 0,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.batch_size > 0 && self.loss.is_finite()
    }
}

/// Gradient update from a participant
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GradientUpdate {
    /// Participant identifier
    pub participant_id: String,
    /// Model version this update is for
    pub model_version: u64,
    /// Gradient values (f32)
    pub gradients: Vec<f32>,
    /// Update metadata
    pub metadata: GradientMetadata,
}

impl GradientUpdate {
    pub fn new(
        participant_id: String,
        model_version: u64,
        gradients: Vec<f32>,
        batch_size: u32,
        loss: f32,
    ) -> Self {
        Self {
            participant_id,
            model_version,
            gradients,
            metadata: GradientMetadata::new(batch_size, loss),
        }
    }

    /// Get gradient dimension
    pub fn dimension(&self) -> usize {
        self.gradients.len()
    }

    /// Calculate L2 norm of gradients
    pub fn l2_norm(&self) -> f32 {
        self.gradients.iter().map(|g| g * g).sum::<f32>().sqrt()
    }

    /// Validate the update
    pub fn is_valid(&self) -> bool {
        !self.gradients.is_empty()
            && self.metadata.is_valid()
            && self.gradients.iter().all(|g| g.is_finite())
    }
}

/// Aggregated gradient result
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AggregatedGradient {
    /// Aggregated gradient values
    pub gradients: Vec<f32>,
    /// Model version
    pub model_version: u64,
    /// Number of participants included
    pub participant_count: usize,
    /// Number of participants excluded
    pub excluded_count: usize,
    /// Aggregation method used
    pub method: AggregationMethod,
}

impl AggregatedGradient {
    pub fn new(
        gradients: Vec<f32>,
        model_version: u64,
        participant_count: usize,
        excluded_count: usize,
        method: AggregationMethod,
    ) -> Self {
        Self {
            gradients,
            model_version,
            participant_count,
            excluded_count,
            method,
        }
    }

    pub fn dimension(&self) -> usize {
        self.gradients.len()
    }
}

/// Participant state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Participant {
    /// Participant identifier
    pub id: String,
    /// Reputation score (0.0-1.0)
    pub reputation: f32,
    /// Composite trust score
    pub trust_score: f32,
    /// Number of rounds participated
    pub rounds_participated: u32,
    /// Successful contributions
    pub successful_contributions: u32,
}

impl Participant {
    pub fn new(id: String) -> Self {
        Self {
            id,
            reputation: 0.5,
            trust_score: 0.5,
            rounds_participated: 0,
            successful_contributions: 0,
        }
    }

    pub fn with_reputation(id: String, reputation: f32) -> Self {
        Self {
            id,
            reputation,
            trust_score: reputation,
            rounds_participated: 0,
            successful_contributions: 0,
        }
    }
}

/// Aggregation methods
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Standard Federated Averaging (batch-size weighted)
    FedAvg,
    /// Trimmed Mean (removes outliers per dimension)
    TrimmedMean,
    /// Coordinate-wise Median (robust to 50% Byzantine)
    Median,
    /// Krum algorithm (selects closest to neighbors)
    Krum,
    /// Multi-Krum: selects and averages top-k gradients by Krum score
    MultiKrum,
    /// Geometric Median via Weiszfeld iterative algorithm
    GeometricMedian,
    /// Trust-weighted aggregation (reputation-based)
    #[default]
    TrustWeighted,
}

/// Maximum validated Byzantine tolerance
pub const MAX_BYZANTINE_TOLERANCE: f32 = 0.34;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_update() {
        let update = GradientUpdate::new("p1".to_string(), 1, vec![0.1, 0.2, 0.3], 100, 0.5);
        assert_eq!(update.dimension(), 3);
        assert!(update.is_valid());
        assert!(update.l2_norm() > 0.0);
    }

    #[test]
    fn test_gradient_metadata() {
        let meta = GradientMetadata::new(100, 0.5);
        assert!(meta.is_valid());

        let meta_acc = GradientMetadata::with_accuracy(100, 0.5, 0.95);
        assert_eq!(meta_acc.accuracy, Some(0.95));
    }

    #[test]
    fn test_participant() {
        let p = Participant::new("p1".to_string());
        assert_eq!(p.rounds_participated, 0);
        assert!((p.reputation - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_invalid_gradient() {
        let update = GradientUpdate::new("p1".to_string(), 1, vec![], 100, 0.5);
        assert!(!update.is_valid());

        let update_nan = GradientUpdate::new("p1".to_string(), 1, vec![f32::NAN], 100, 0.5);
        assert!(!update_nan.is_valid());
    }
}
