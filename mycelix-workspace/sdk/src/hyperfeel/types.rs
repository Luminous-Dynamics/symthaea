//! HyperFeel Types and Data Structures

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// HV16 dimension: 16,384 bits = 2,048 bytes
pub const HV16_DIMENSION: usize = 16384;
/// HV16 byte size
pub const HV16_BYTES: usize = 2048;

/// Encoding configuration for HyperFeel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingConfig {
    /// Hypervector dimension (must be power of 2, >= 1024)
    pub dimension: usize,
    /// Enable temporal trajectory tracking
    pub use_temporal: bool,
    /// Quantization bits (1, 2, 4, 8, or 16)
    pub quantize_bits: u8,
    /// Random seed for reproducible projection
    pub projection_seed: u64,
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            dimension: HV16_DIMENSION,
            use_temporal: true,
            quantize_bits: 8,
            projection_seed: 42,
        }
    }
}

impl EncodingConfig {
    /// Create config with custom dimension
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        assert!(dimension >= 1024, "dimension must be >= 1024");
        assert!(dimension.is_power_of_two(), "dimension must be power of 2");
        self.dimension = dimension;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.dimension < 1024 {
            return Err(format!("dimension must be >= 1024, got {}", self.dimension));
        }
        if !self.dimension.is_power_of_two() {
            return Err(format!(
                "dimension must be power of 2, got {}",
                self.dimension
            ));
        }
        if ![1, 2, 4, 8, 16].contains(&self.quantize_bits) {
            return Err(format!(
                "quantize_bits must be 1,2,4,8,16, got {}",
                self.quantize_bits
            ));
        }
        Ok(())
    }
}

/// Compressed gradient representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperGradient {
    /// Unique FL node identifier
    pub node_id: String,
    /// Federation round number
    pub round_num: u32,
    /// 32-byte random nonce (replay prevention)
    pub nonce: [u8; 32],
    /// HV16 encoding (2,048 bytes)
    pub hypervector: Vec<u8>,
    /// Gradient magnitude (L2 norm)
    pub quality_score: f32,
    /// Unix timestamp
    pub timestamp: u64,
    /// Original gradient size in bytes
    pub original_size: usize,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Encoding time in milliseconds
    pub encode_time_ms: f32,
    /// SHA3-256 hash of original gradient
    pub gradient_hash: [u8; 32],
}

impl HyperGradient {
    /// Create a new HyperGradient
    pub fn new(
        node_id: String,
        round_num: u32,
        hypervector: Vec<u8>,
        quality_score: f32,
        original_size: usize,
        encode_time_ms: f32,
        gradient_hash: [u8; 32],
    ) -> Self {
        // Generate random nonce
        let mut nonce = [0u8; 32];
        #[cfg(feature = "getrandom-js")]
        getrandom::getrandom(&mut nonce).unwrap_or_else(|_| {
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            for (i, byte) in ts.to_le_bytes().iter().cycle().take(32).enumerate() {
                nonce[i] = *byte;
            }
        });
        #[cfg(not(feature = "getrandom-js"))]
        {
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            for (i, byte) in ts.to_le_bytes().iter().cycle().take(32).enumerate() {
                nonce[i] = *byte;
            }
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let compression_ratio = if !hypervector.is_empty() {
            original_size as f32 / hypervector.len() as f32
        } else {
            0.0
        };

        Self {
            node_id,
            round_num,
            nonce,
            hypervector,
            quality_score,
            timestamp,
            original_size,
            compression_ratio,
            encode_time_ms,
            gradient_hash,
        }
    }

    /// Check if this gradient is beneficial (positive quality)
    pub fn is_beneficial(&self) -> bool {
        self.quality_score > 0.0
    }

    /// Serialize to bytes for transmission
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(data)
    }

    /// Get hypervector size in KB
    pub fn size_kb(&self) -> f32 {
        self.hypervector.len() as f32 / 1024.0
    }
}

/// Temporal trajectory entry for tracking gradient history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEntry {
    /// Round number
    pub round: u32,
    /// Hypervector bytes
    pub hypervector: Vec<u8>,
    /// Unix timestamp
    pub timestamp: u64,
}

/// Similarity result between two hypergradients
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    /// Cosine similarity [-1, 1]
    pub cosine_similarity: f32,
    /// Hamming distance (bit differences)
    pub hamming_distance: usize,
    /// Normalized Hamming similarity [0, 1]
    pub hamming_similarity: f32,
}

impl SimilarityResult {
    /// Create new similarity result
    pub fn new(cosine_similarity: f32, hamming_distance: usize, total_bits: usize) -> Self {
        let hamming_similarity = 1.0 - (hamming_distance as f32 / total_bits as f32);
        Self {
            cosine_similarity,
            hamming_distance,
            hamming_similarity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = EncodingConfig::default();
        assert!(config.validate().is_ok());

        let bad_config = EncodingConfig {
            dimension: 512, // Too small
            ..Default::default()
        };
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_hypergradient_creation() {
        let hg = HyperGradient::new(
            "node-1".to_string(),
            1,
            vec![0u8; HV16_BYTES],
            0.5,
            4_000_000,
            10.0,
            [0u8; 32],
        );

        assert_eq!(hg.node_id, "node-1");
        assert_eq!(hg.round_num, 1);
        assert_eq!(hg.hypervector.len(), HV16_BYTES);
        assert!(hg.compression_ratio > 1900.0);
    }
}
