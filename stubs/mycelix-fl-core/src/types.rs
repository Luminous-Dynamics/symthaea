//! Core FL types (stub)

use serde::{Deserialize, Serialize};

/// Gradient update from a participant
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GradientUpdate {
    pub participant_id: String,
    pub model_version: u64,
    pub gradients: Vec<f32>,
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

    pub fn dimension(&self) -> usize {
        self.gradients.len()
    }

    pub fn l2_norm(&self) -> f32 {
        self.gradients.iter().map(|g| g * g).sum::<f32>().sqrt()
    }

    pub fn is_valid(&self) -> bool {
        !self.gradients.is_empty()
            && self.metadata.is_valid()
            && self.gradients.iter().all(|g| g.is_finite())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GradientMetadata {
    pub batch_size: u32,
    pub loss: f32,
    pub accuracy: Option<f32>,
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

    pub fn is_valid(&self) -> bool {
        self.batch_size > 0 && self.loss.is_finite()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AggregatedGradient {
    pub gradients: Vec<f32>,
    pub participant_count: usize,
}

pub const MAX_BYZANTINE_TOLERANCE: f32 = 0.34;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationMethod {
    FedAvg,
    TrustWeighted,
    Krum,
    TrimmedMean,
}
