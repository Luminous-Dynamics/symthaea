use crate::hypervector::Hypervector;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use time::OffsetDateTime;

/// Errors that can occur during HyperGradient construction.
#[derive(Debug, Error)]
pub enum HyperGradientError {
    #[error("gradient length must be > 0")]
    EmptyGradient,
}

/// Minimal metadata about a model layer participating in a gradient.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerMetadata {
    pub layer_index: u32,
    pub size: usize,
}

/// Placeholder signature type for future cryptographic integration.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Signature {
    pub bytes: Vec<u8>,
}

/// Hypervector-encoded gradient with lightweight metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperGradient {
    /// 16,384-dimensional hypervector encoding the update.
    pub vector: Hypervector,

    /// Layer-level metadata useful for diagnostics/debugging.
    pub layer_info: Vec<LayerMetadata>,

    /// Unix timestamp (seconds) for temporal coordination.
    pub timestamp: i64,

    /// Optional signature for authenticity.
    pub signature: Signature,
}

impl HyperGradient {
    /// Encode a traditional dense gradient into a hypervector.
    ///
    /// This is a deliberately simple encoder:
    /// - Gradients are quantized to i8 in [-127, 127].
    /// - Components are folded into the fixed dimensionality by wrap-around.
    pub fn from_gradient(gradient: &[f32]) -> Result<Self, HyperGradientError> {
        if gradient.is_empty() {
            return Err(HyperGradientError::EmptyGradient);
        }

        let mut hv = Hypervector::zero();
        let dim = Hypervector::DIMENSION;

        for (idx, &value) in gradient.iter().enumerate() {
            let quantized = (value * 127.0).round().clamp(-127.0, 127.0) as i8;
            let pos = idx % dim;
            let accum = hv.data[pos] as i16 + quantized as i16;
            hv.data[pos] = accum.clamp(i8::MIN as i16, i8::MAX as i16) as i8;
        }

        // Simple single-layer metadata; callers can override if they want.
        let layer_info = vec![LayerMetadata {
            layer_index: 0,
            size: gradient.len(),
        }];

        Ok(Self {
            vector: hv,
            layer_info,
            timestamp: OffsetDateTime::now_utc().unix_timestamp(),
            signature: Signature::default(),
        })
    }

    /// Aggregate multiple hypergradients via element-wise bundling.
    pub fn aggregate(hypergradients: &[HyperGradient]) -> Option<Self> {
        if hypergradients.is_empty() {
            return None;
        }

        let mut acc = Hypervector::zero();
        for hg in hypergradients {
            acc = acc.add(&hg.vector);
        }
        let normalized = acc.normalize(hypergradients.len());

        Some(Self {
            vector: normalized,
            layer_info: Vec::new(),
            timestamp: OffsetDateTime::now_utc().unix_timestamp(),
            signature: Signature::default(),
        })
    }
}

