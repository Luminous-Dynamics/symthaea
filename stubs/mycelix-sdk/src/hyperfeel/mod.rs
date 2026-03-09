//! HyperFeel stub

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingConfig {
    pub dimension: usize,
    pub num_hashes: usize,
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            dimension: 16384,
            num_hashes: 8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperGradient {
    pub data: Vec<u8>,
    pub original_dim: usize,
    pub model_version: u64,
    pub source_id: String,
    pub compression_ratio: f32,
}

#[derive(Debug, Clone)]
pub struct HyperFeelEncoder {
    config: EncodingConfig,
}

impl HyperFeelEncoder {
    pub fn new(config: EncodingConfig) -> Self {
        Self { config }
    }

    pub fn encode_gradient(
        &mut self,
        gradient: &[f32],
        model_version: u64,
        source_id: &str,
    ) -> HyperGradient {
        let compressed_size = self.config.dimension / 8;
        HyperGradient {
            data: vec![0u8; compressed_size],
            original_dim: gradient.len(),
            model_version,
            source_id: source_id.to_string(),
            compression_ratio: gradient.len() as f32 * 4.0 / compressed_size as f32,
        }
    }

    pub fn similarity(&self, a: &HyperGradient, b: &HyperGradient) -> f32 {
        if a.data.len() != b.data.len() || a.data.is_empty() {
            return 0.0;
        }
        let matching_bits: u32 = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(x, y)| (!(x ^ y)).count_ones())
            .sum();
        let total_bits = a.data.len() as f32 * 8.0;
        (2.0 * matching_bits as f32 / total_bits) - 1.0
    }
}
