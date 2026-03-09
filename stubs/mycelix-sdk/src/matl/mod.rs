//! MATL stub

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofOfGradientQuality {
    pub quality: f64,
    pub consistency: f64,
    pub entropy: f64,
    pub timestamp: u64,
}

impl ProofOfGradientQuality {
    pub fn new(quality: f64, consistency: f64, entropy: f64) -> Self {
        Self {
            quality: quality.clamp(0.0, 1.0),
            consistency: consistency.clamp(0.0, 1.0),
            entropy: entropy.max(0.0),
            timestamp: 0,
        }
    }

    pub fn composite_score(&self, reputation: f64) -> f64 {
        (0.4 * self.quality + 0.3 * self.consistency + 0.3 * reputation).clamp(0.0, 1.0)
    }
}
