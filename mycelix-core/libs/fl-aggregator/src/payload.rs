//! Unified payload types for multi-paradigm federated learning.
//!
//! Supports both traditional gradient-based FL and hyperdimensional computing (HDC) FL.
//!
//! # Payload Types
//!
//! - [`DenseGradient`]: Traditional FL gradients (`Array1<f32>`)
//! - [`Hypervector`]: HyperFeel-encoded compressed gradients (16KB)
//! - [`BinaryHypervector`]: Native Symthaea binary HDC (2KB)
//! - [`SparseGradient`]: Sparse representation for bandwidth efficiency
//! - [`QuantizedGradient`]: Reduced precision gradients
//!
//! # Example
//!
//! ```rust
//! use fl_aggregator::payload::{Payload, DenseGradient, Hypervector};
//!
//! // Traditional dense gradient
//! let dense = DenseGradient::from_vec(vec![1.0, 2.0, 3.0]);
//!
//! // HyperFeel-encoded hypervector
//! let hyper = Hypervector::random(16384);
//!
//! // Both implement the Payload trait
//! println!("Dense size: {} bytes", dense.size_bytes());
//! println!("Hyper size: {} bytes", hyper.size_bytes());
//! ```

use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Core trait for all FL payload types.
///
/// This trait provides a common interface for different payload representations
/// used in federated learning, enabling the aggregator to handle both traditional
/// gradient-based FL and HDC-based FL uniformly.
pub trait Payload: Clone + Send + Sync + Debug {
    /// Get the dimension of the payload.
    fn dimension(&self) -> usize;

    /// Get the size in bytes.
    fn size_bytes(&self) -> usize;

    /// Check if the payload is valid (no NaN, proper dimensions, etc.).
    fn is_valid(&self) -> bool;

    /// Get the payload type identifier.
    fn payload_type(&self) -> PayloadType;
}

/// Enumeration of supported payload types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PayloadType {
    /// Traditional dense gradient (Array1<f32>).
    DenseGradient,
    /// HyperFeel-encoded gradient (16,384 × i8).
    HyperEncoded,
    /// Native Symthaea binary hypervector (16,384 bits).
    BinaryHypervector,
    /// Sparse gradient (indices + values).
    SparseGradient,
    /// Quantized gradient (reduced precision).
    QuantizedGradient,
}

impl std::fmt::Display for PayloadType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PayloadType::DenseGradient => write!(f, "DenseGradient"),
            PayloadType::HyperEncoded => write!(f, "HyperEncoded"),
            PayloadType::BinaryHypervector => write!(f, "BinaryHypervector"),
            PayloadType::SparseGradient => write!(f, "SparseGradient"),
            PayloadType::QuantizedGradient => write!(f, "QuantizedGradient"),
        }
    }
}

// =============================================================================
// Dense Gradient (Traditional FL)
// =============================================================================

/// Traditional dense gradient for gradient-based federated learning.
///
/// This is the standard representation used in most FL systems,
/// containing a vector of f32 values representing model parameter updates.
#[derive(Clone, Debug)]
pub struct DenseGradient {
    /// The gradient values.
    pub values: ndarray::Array1<f32>,
}

// Custom Serialize/Deserialize for DenseGradient since ndarray doesn't have it by default
impl serde::Serialize for DenseGradient {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("DenseGradient", 1)?;
        state.serialize_field("values", &self.values.to_vec())?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for DenseGradient {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct DenseGradientHelper {
            values: Vec<f32>,
        }

        let helper = DenseGradientHelper::deserialize(deserializer)?;
        Ok(DenseGradient {
            values: ndarray::Array1::from(helper.values),
        })
    }
}

impl DenseGradient {
    /// Create from a vector.
    pub fn from_vec(values: Vec<f32>) -> Self {
        Self {
            values: ndarray::Array1::from(values),
        }
    }

    /// Create from an ndarray.
    pub fn from_array(values: ndarray::Array1<f32>) -> Self {
        Self { values }
    }

    /// Create a zero gradient of given dimension.
    pub fn zeros(dim: usize) -> Self {
        Self {
            values: ndarray::Array1::zeros(dim),
        }
    }

    /// Get the underlying array.
    pub fn as_array(&self) -> &ndarray::Array1<f32> {
        &self.values
    }

    /// Get mutable reference to the underlying array.
    pub fn as_array_mut(&mut self) -> &mut ndarray::Array1<f32> {
        &mut self.values
    }

    /// Convert to Vec.
    pub fn to_vec(&self) -> Vec<f32> {
        self.values.to_vec()
    }

    /// Calculate L2 norm.
    pub fn l2_norm(&self) -> f32 {
        self.values.iter().map(|x| x.powi(2)).sum::<f32>().sqrt()
    }

    /// Validate all values are finite.
    pub fn validate_finite(&self) -> Result<(), String> {
        for (i, &val) in self.values.iter().enumerate() {
            if !val.is_finite() {
                return Err(format!("Non-finite value at index {}: {}", i, val));
            }
        }
        Ok(())
    }
}

impl Payload for DenseGradient {
    fn dimension(&self) -> usize {
        self.values.len()
    }

    fn size_bytes(&self) -> usize {
        self.values.len() * std::mem::size_of::<f32>()
    }

    fn is_valid(&self) -> bool {
        !self.values.is_empty() && self.values.iter().all(|v| v.is_finite())
    }

    fn payload_type(&self) -> PayloadType {
        PayloadType::DenseGradient
    }
}

impl From<ndarray::Array1<f32>> for DenseGradient {
    fn from(values: ndarray::Array1<f32>) -> Self {
        Self { values }
    }
}

impl From<Vec<f32>> for DenseGradient {
    fn from(values: Vec<f32>) -> Self {
        Self::from_vec(values)
    }
}

impl From<DenseGradient> for ndarray::Array1<f32> {
    fn from(gradient: DenseGradient) -> Self {
        gradient.values
    }
}

// =============================================================================
// Hypervector (HyperFeel HDC)
// =============================================================================

/// Standard hypervector dimension (HyperFeel/Symthaea compatible).
pub const HYPERVECTOR_DIM: usize = 16384;

/// HyperFeel-encoded hypervector gradient.
///
/// Uses 16,384 × i8 representation, achieving ~2000x compression
/// compared to dense gradients while preserving training quality.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hypervector {
    /// The hypervector components (i8 range: -128 to 127).
    pub components: Vec<i8>,

    /// Encoding metadata.
    pub metadata: HypervectorMetadata,
}

/// Metadata about the hypervector encoding.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HypervectorMetadata {
    /// Encoder version (e.g., "hyperfeel-v2").
    pub encoder_version: String,

    /// Original gradient dimension before encoding.
    pub original_dimension: usize,

    /// Compression ratio achieved.
    pub compression_ratio: f32,

    /// Random projection seed for reproducibility.
    pub projection_seed: Option<i64>,

    /// Whether causal encoding was used.
    pub use_causal: bool,

    /// Whether temporal encoding was used.
    pub use_temporal: bool,
}

impl Hypervector {
    /// Create a new hypervector from components.
    pub fn new(components: Vec<i8>) -> Self {
        Self {
            components,
            metadata: HypervectorMetadata::default(),
        }
    }

    /// Create a new hypervector with metadata.
    pub fn with_metadata(components: Vec<i8>, metadata: HypervectorMetadata) -> Self {
        Self { components, metadata }
    }

    /// Create a zero hypervector.
    pub fn zeros(dim: usize) -> Self {
        Self {
            components: vec![0i8; dim],
            metadata: HypervectorMetadata::default(),
        }
    }

    /// Create a random hypervector (for testing).
    pub fn random(dim: usize) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::time::{SystemTime, UNIX_EPOCH};

        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let mut hasher = DefaultHasher::new();
        let components: Vec<i8> = (0..dim)
            .map(|i| {
                (i as u64).hash(&mut hasher);
                seed.hash(&mut hasher);
                ((hasher.finish() % 256) as i16 - 128) as i8
            })
            .collect();

        Self {
            components,
            metadata: HypervectorMetadata {
                encoder_version: "random".to_string(),
                ..Default::default()
            },
        }
    }

    /// Calculate cosine similarity with another hypervector.
    pub fn cosine_similarity(&self, other: &Hypervector) -> f32 {
        if self.components.len() != other.components.len() {
            return 0.0;
        }

        let dot: i64 = self
            .components
            .iter()
            .zip(&other.components)
            .map(|(&a, &b)| a as i64 * b as i64)
            .sum();

        let norm_a: f64 = self
            .components
            .iter()
            .map(|&x| (x as i64).pow(2) as f64)
            .sum::<f64>()
            .sqrt();

        let norm_b: f64 = other
            .components
            .iter()
            .map(|&x| (x as i64).pow(2) as f64)
            .sum::<f64>()
            .sqrt();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }

        (dot as f64 / (norm_a * norm_b)) as f32
    }

    /// Bundle (add) hypervectors with majority voting.
    ///
    /// This is the standard HDC aggregation operation.
    pub fn bundle(vectors: &[&Hypervector]) -> Option<Hypervector> {
        if vectors.is_empty() {
            return None;
        }

        let dim = vectors[0].components.len();
        if !vectors.iter().all(|v| v.components.len() == dim) {
            return None;
        }

        // Sum components (using i32 to avoid overflow)
        let mut sums: Vec<i32> = vec![0; dim];
        for hv in vectors {
            for (i, &c) in hv.components.iter().enumerate() {
                sums[i] += c as i32;
            }
        }

        // Normalize back to i8 range
        let n = vectors.len() as i32;
        let components: Vec<i8> = sums
            .iter()
            .map(|&s| (s / n).clamp(-128, 127) as i8)
            .collect();

        Some(Hypervector {
            components,
            metadata: HypervectorMetadata {
                encoder_version: "bundled".to_string(),
                ..Default::default()
            },
        })
    }

    /// Bind (XOR-like) two hypervectors.
    pub fn bind(&self, other: &Hypervector) -> Option<Hypervector> {
        if self.components.len() != other.components.len() {
            return None;
        }

        let components: Vec<i8> = self
            .components
            .iter()
            .zip(&other.components)
            .map(|(&a, &b)| ((a as i32 * b as i32) / 128).clamp(-128, 127) as i8)
            .collect();

        Some(Hypervector::new(components))
    }

    /// Permute the hypervector by shifting components.
    pub fn permute(&self, shift: usize) -> Hypervector {
        let dim = self.components.len();
        let shift = shift % dim;
        let mut components = vec![0i8; dim];

        for i in 0..dim {
            components[(i + shift) % dim] = self.components[i];
        }

        Hypervector::new(components)
    }
}

impl Payload for Hypervector {
    fn dimension(&self) -> usize {
        self.components.len()
    }

    fn size_bytes(&self) -> usize {
        self.components.len() * std::mem::size_of::<i8>()
    }

    fn is_valid(&self) -> bool {
        !self.components.is_empty()
    }

    fn payload_type(&self) -> PayloadType {
        PayloadType::HyperEncoded
    }
}

// =============================================================================
// Binary Hypervector (Symthaea Native)
// =============================================================================

/// Native binary hypervector for Symthaea HDC.
///
/// Uses packed bits for maximum efficiency: 16,384 bits = 2KB.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BinaryHypervector {
    /// Packed bits (dimension / 8 bytes).
    pub data: Vec<u8>,

    /// Bit dimension.
    pub bit_dimension: usize,
}

impl BinaryHypervector {
    /// Create a new binary hypervector.
    pub fn new(data: Vec<u8>, bit_dimension: usize) -> Self {
        Self { data, bit_dimension }
    }

    /// Create from a bit slice.
    pub fn from_bits(bits: &[bool]) -> Self {
        let bit_dimension = bits.len();
        let byte_count = (bit_dimension + 7) / 8;
        let mut data = vec![0u8; byte_count];

        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                data[i / 8] |= 1 << (i % 8);
            }
        }

        Self { data, bit_dimension }
    }

    /// Create a zero binary hypervector.
    pub fn zeros(bit_dimension: usize) -> Self {
        let byte_count = (bit_dimension + 7) / 8;
        Self {
            data: vec![0u8; byte_count],
            bit_dimension,
        }
    }

    /// Create a random binary hypervector.
    pub fn random(bit_dimension: usize) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::time::{SystemTime, UNIX_EPOCH};

        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let byte_count = (bit_dimension + 7) / 8;
        let mut hasher = DefaultHasher::new();
        let data: Vec<u8> = (0..byte_count)
            .map(|i| {
                (i as u64).hash(&mut hasher);
                seed.hash(&mut hasher);
                hasher.finish() as u8
            })
            .collect();

        Self { data, bit_dimension }
    }

    /// Get a specific bit.
    pub fn get_bit(&self, index: usize) -> Option<bool> {
        if index >= self.bit_dimension {
            return None;
        }
        Some((self.data[index / 8] >> (index % 8)) & 1 == 1)
    }

    /// Set a specific bit.
    pub fn set_bit(&mut self, index: usize, value: bool) {
        if index >= self.bit_dimension {
            return;
        }
        if value {
            self.data[index / 8] |= 1 << (index % 8);
        } else {
            self.data[index / 8] &= !(1 << (index % 8));
        }
    }

    /// Calculate Hamming distance (number of differing bits).
    pub fn hamming_distance(&self, other: &BinaryHypervector) -> usize {
        self.data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| (a ^ b).count_ones() as usize)
            .sum()
    }

    /// Calculate Hamming similarity (normalized).
    pub fn hamming_similarity(&self, other: &BinaryHypervector) -> f32 {
        let distance = self.hamming_distance(other);
        1.0 - (distance as f32 / self.bit_dimension as f32)
    }

    /// XOR binding operation.
    pub fn xor(&self, other: &BinaryHypervector) -> Option<BinaryHypervector> {
        if self.bit_dimension != other.bit_dimension {
            return None;
        }

        let data: Vec<u8> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| a ^ b)
            .collect();

        Some(BinaryHypervector {
            data,
            bit_dimension: self.bit_dimension,
        })
    }

    /// Bundle (majority voting) multiple binary hypervectors.
    pub fn bundle(vectors: &[&BinaryHypervector]) -> Option<BinaryHypervector> {
        if vectors.is_empty() {
            return None;
        }

        let bit_dim = vectors[0].bit_dimension;
        if !vectors.iter().all(|v| v.bit_dimension == bit_dim) {
            return None;
        }

        let byte_count = (bit_dim + 7) / 8;
        let threshold = vectors.len() / 2;

        let mut data = vec![0u8; byte_count];

        for bit_idx in 0..bit_dim {
            let ones_count = vectors
                .iter()
                .filter(|v| v.get_bit(bit_idx).unwrap_or(false))
                .count();

            if ones_count > threshold {
                data[bit_idx / 8] |= 1 << (bit_idx % 8);
            }
        }

        Some(BinaryHypervector {
            data,
            bit_dimension: bit_dim,
        })
    }

    /// Convert to Hypervector (i8 representation).
    pub fn to_hypervector(&self) -> Hypervector {
        let components: Vec<i8> = (0..self.bit_dimension)
            .map(|i| if self.get_bit(i).unwrap_or(false) { 127 } else { -128 })
            .collect();

        Hypervector::new(components)
    }

    /// Create from Hypervector (threshold at 0).
    pub fn from_hypervector(hv: &Hypervector) -> Self {
        let bits: Vec<bool> = hv.components.iter().map(|&c| c >= 0).collect();
        Self::from_bits(&bits)
    }
}

impl Payload for BinaryHypervector {
    fn dimension(&self) -> usize {
        self.bit_dimension
    }

    fn size_bytes(&self) -> usize {
        self.data.len()
    }

    fn is_valid(&self) -> bool {
        !self.data.is_empty() && self.data.len() == (self.bit_dimension + 7) / 8
    }

    fn payload_type(&self) -> PayloadType {
        PayloadType::BinaryHypervector
    }
}

// =============================================================================
// Sparse Gradient
// =============================================================================

/// Sparse gradient representation for bandwidth efficiency.
///
/// Only stores non-zero (or significant) gradient values with their indices.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SparseGradient {
    /// Indices of non-zero values.
    pub indices: Vec<u32>,

    /// Values at those indices.
    pub values: Vec<f32>,

    /// Full dimension of the gradient.
    pub full_dimension: usize,
}

impl SparseGradient {
    /// Create a new sparse gradient.
    pub fn new(indices: Vec<u32>, values: Vec<f32>, full_dimension: usize) -> Self {
        Self {
            indices,
            values,
            full_dimension,
        }
    }

    /// Create from dense gradient with threshold.
    pub fn from_dense(dense: &DenseGradient, threshold: f32) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (i, &v) in dense.values.iter().enumerate() {
            if v.abs() > threshold {
                indices.push(i as u32);
                values.push(v);
            }
        }

        Self {
            indices,
            values,
            full_dimension: dense.values.len(),
        }
    }

    /// Convert to dense gradient.
    pub fn to_dense(&self) -> DenseGradient {
        let mut values = ndarray::Array1::zeros(self.full_dimension);
        for (&idx, &val) in self.indices.iter().zip(&self.values) {
            if (idx as usize) < self.full_dimension {
                values[idx as usize] = val;
            }
        }
        DenseGradient { values }
    }

    /// Get sparsity ratio (non-zeros / total).
    pub fn sparsity_ratio(&self) -> f32 {
        if self.full_dimension == 0 {
            return 0.0;
        }
        self.indices.len() as f32 / self.full_dimension as f32
    }

    /// Get compression ratio (dense size / sparse size).
    pub fn compression_ratio(&self) -> f32 {
        let dense_size = self.full_dimension * std::mem::size_of::<f32>();
        let sparse_size = self.size_bytes();
        if sparse_size == 0 {
            return 0.0;
        }
        dense_size as f32 / sparse_size as f32
    }
}

impl Payload for SparseGradient {
    fn dimension(&self) -> usize {
        self.full_dimension
    }

    fn size_bytes(&self) -> usize {
        self.indices.len() * std::mem::size_of::<u32>()
            + self.values.len() * std::mem::size_of::<f32>()
    }

    fn is_valid(&self) -> bool {
        self.indices.len() == self.values.len()
            && self.indices.iter().all(|&i| (i as usize) < self.full_dimension)
            && self.values.iter().all(|v| v.is_finite())
    }

    fn payload_type(&self) -> PayloadType {
        PayloadType::SparseGradient
    }
}

// =============================================================================
// Quantized Gradient
// =============================================================================

/// Quantized gradient with reduced precision.
///
/// Uses fewer bits per value (1, 2, 4, or 8) for bandwidth reduction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizedGradient {
    /// Quantized data.
    pub data: Vec<u8>,

    /// Bits per value.
    pub bits_per_value: u8,

    /// Dequantization scale.
    pub scale: f32,

    /// Dequantization zero point.
    pub zero_point: f32,

    /// Original dimension.
    pub dimension: usize,
}

impl QuantizedGradient {
    /// Create from dense gradient with specified bits.
    pub fn from_dense(dense: &DenseGradient, bits: u8) -> Self {
        assert!(bits == 1 || bits == 2 || bits == 4 || bits == 8);

        let values = &dense.values;
        let min_val = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let levels = (1u32 << bits) as f32;
        let scale = if range.abs() < 1e-10 { 1.0 } else { range / (levels - 1.0) };
        let zero_point = min_val;

        // Quantize values
        let quantized: Vec<u8> = values
            .iter()
            .map(|&v| {
                let normalized = (v - zero_point) / scale;
                normalized.round().clamp(0.0, levels - 1.0) as u8
            })
            .collect();

        // Pack bits
        let data = Self::pack_bits(&quantized, bits);

        Self {
            data,
            bits_per_value: bits,
            scale,
            zero_point,
            dimension: values.len(),
        }
    }

    /// Convert to dense gradient.
    pub fn to_dense(&self) -> DenseGradient {
        let quantized = Self::unpack_bits(&self.data, self.bits_per_value, self.dimension);
        let values: Vec<f32> = quantized
            .iter()
            .map(|&q| q as f32 * self.scale + self.zero_point)
            .collect();

        DenseGradient::from_vec(values)
    }

    /// Pack values into bytes.
    fn pack_bits(values: &[u8], bits: u8) -> Vec<u8> {
        match bits {
            8 => values.to_vec(),
            4 => {
                values
                    .chunks(2)
                    .map(|chunk| {
                        let low = chunk.get(0).copied().unwrap_or(0) & 0x0F;
                        let high = (chunk.get(1).copied().unwrap_or(0) & 0x0F) << 4;
                        low | high
                    })
                    .collect()
            }
            2 => {
                values
                    .chunks(4)
                    .map(|chunk| {
                        let mut byte = 0u8;
                        for (i, &v) in chunk.iter().enumerate() {
                            byte |= (v & 0x03) << (i * 2);
                        }
                        byte
                    })
                    .collect()
            }
            1 => {
                values
                    .chunks(8)
                    .map(|chunk| {
                        let mut byte = 0u8;
                        for (i, &v) in chunk.iter().enumerate() {
                            byte |= (v & 0x01) << i;
                        }
                        byte
                    })
                    .collect()
            }
            _ => values.to_vec(),
        }
    }

    /// Unpack bytes into values.
    fn unpack_bits(data: &[u8], bits: u8, count: usize) -> Vec<u8> {
        match bits {
            8 => data[..count.min(data.len())].to_vec(),
            4 => {
                let mut values = Vec::with_capacity(count);
                for &byte in data {
                    if values.len() < count {
                        values.push(byte & 0x0F);
                    }
                    if values.len() < count {
                        values.push((byte >> 4) & 0x0F);
                    }
                }
                values.truncate(count);
                values
            }
            2 => {
                let mut values = Vec::with_capacity(count);
                for &byte in data {
                    for i in 0..4 {
                        if values.len() >= count {
                            break;
                        }
                        values.push((byte >> (i * 2)) & 0x03);
                    }
                }
                values
            }
            1 => {
                let mut values = Vec::with_capacity(count);
                for &byte in data {
                    for i in 0..8 {
                        if values.len() >= count {
                            break;
                        }
                        values.push((byte >> i) & 0x01);
                    }
                }
                values
            }
            _ => data[..count.min(data.len())].to_vec(),
        }
    }
}

impl Payload for QuantizedGradient {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn size_bytes(&self) -> usize {
        self.data.len()
    }

    fn is_valid(&self) -> bool {
        !self.data.is_empty()
            && self.scale.is_finite()
            && self.zero_point.is_finite()
            && (self.bits_per_value == 1
                || self.bits_per_value == 2
                || self.bits_per_value == 4
                || self.bits_per_value == 8)
    }

    fn payload_type(&self) -> PayloadType {
        PayloadType::QuantizedGradient
    }
}

// =============================================================================
// Aggregation Traits
// =============================================================================

/// Trait for payload types that support dense-style aggregation.
///
/// This enables Byzantine-resistant aggregation algorithms like Krum, Median, etc.
pub trait DenseAggregatable: Payload {
    /// Convert to dense gradient for aggregation.
    fn to_dense_gradient(&self) -> DenseGradient;

    /// Create from dense gradient after aggregation.
    fn from_dense_gradient(gradient: DenseGradient) -> Self;
}

impl DenseAggregatable for DenseGradient {
    fn to_dense_gradient(&self) -> DenseGradient {
        self.clone()
    }

    fn from_dense_gradient(gradient: DenseGradient) -> Self {
        gradient
    }
}

impl DenseAggregatable for SparseGradient {
    fn to_dense_gradient(&self) -> DenseGradient {
        self.to_dense()
    }

    fn from_dense_gradient(gradient: DenseGradient) -> Self {
        // Convert back to sparse (threshold = 0 to preserve all non-zero)
        SparseGradient::from_dense(&gradient, 0.0)
    }
}

impl DenseAggregatable for QuantizedGradient {
    fn to_dense_gradient(&self) -> DenseGradient {
        self.to_dense()
    }

    fn from_dense_gradient(gradient: DenseGradient) -> Self {
        // Requantize with same bit depth (default to 8)
        QuantizedGradient::from_dense(&gradient, 8)
    }
}

/// Trait for payload types that support HDC-style bundling aggregation.
pub trait HdcAggregatable: Payload {
    /// Bundle (aggregate) multiple payloads using HDC operations.
    fn bundle_hdc(payloads: &[&Self]) -> Option<Self>
    where
        Self: Sized;

    /// Calculate similarity between two payloads.
    fn similarity(&self, other: &Self) -> f32;
}

impl HdcAggregatable for Hypervector {
    fn bundle_hdc(payloads: &[&Self]) -> Option<Self> {
        Hypervector::bundle(payloads)
    }

    fn similarity(&self, other: &Self) -> f32 {
        self.cosine_similarity(other)
    }
}

impl HdcAggregatable for BinaryHypervector {
    fn bundle_hdc(payloads: &[&Self]) -> Option<Self> {
        BinaryHypervector::bundle(payloads)
    }

    fn similarity(&self, other: &Self) -> f32 {
        self.hamming_similarity(other)
    }
}

// =============================================================================
// Unified Payload Enum
// =============================================================================

/// Unified payload type for dynamic dispatch.
///
/// Use this when the payload type is not known at compile time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum UnifiedPayload {
    Dense(DenseGradient),
    Hyper(Hypervector),
    Binary(BinaryHypervector),
    Sparse(SparseGradient),
    Quantized(QuantizedGradient),
}

impl UnifiedPayload {
    /// Get the payload type.
    pub fn payload_type(&self) -> PayloadType {
        match self {
            UnifiedPayload::Dense(_) => PayloadType::DenseGradient,
            UnifiedPayload::Hyper(_) => PayloadType::HyperEncoded,
            UnifiedPayload::Binary(_) => PayloadType::BinaryHypervector,
            UnifiedPayload::Sparse(_) => PayloadType::SparseGradient,
            UnifiedPayload::Quantized(_) => PayloadType::QuantizedGradient,
        }
    }

    /// Get dimension.
    pub fn dimension(&self) -> usize {
        match self {
            UnifiedPayload::Dense(p) => p.dimension(),
            UnifiedPayload::Hyper(p) => p.dimension(),
            UnifiedPayload::Binary(p) => p.dimension(),
            UnifiedPayload::Sparse(p) => p.dimension(),
            UnifiedPayload::Quantized(p) => p.dimension(),
        }
    }

    /// Get size in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            UnifiedPayload::Dense(p) => p.size_bytes(),
            UnifiedPayload::Hyper(p) => p.size_bytes(),
            UnifiedPayload::Binary(p) => p.size_bytes(),
            UnifiedPayload::Sparse(p) => p.size_bytes(),
            UnifiedPayload::Quantized(p) => p.size_bytes(),
        }
    }

    /// Check validity.
    pub fn is_valid(&self) -> bool {
        match self {
            UnifiedPayload::Dense(p) => p.is_valid(),
            UnifiedPayload::Hyper(p) => p.is_valid(),
            UnifiedPayload::Binary(p) => p.is_valid(),
            UnifiedPayload::Sparse(p) => p.is_valid(),
            UnifiedPayload::Quantized(p) => p.is_valid(),
        }
    }

    /// Try to convert to dense gradient.
    pub fn to_dense(&self) -> Option<DenseGradient> {
        match self {
            UnifiedPayload::Dense(p) => Some(p.clone()),
            UnifiedPayload::Sparse(p) => Some(p.to_dense()),
            UnifiedPayload::Quantized(p) => Some(p.to_dense()),
            _ => None,
        }
    }
}

impl From<DenseGradient> for UnifiedPayload {
    fn from(p: DenseGradient) -> Self {
        UnifiedPayload::Dense(p)
    }
}

impl From<Hypervector> for UnifiedPayload {
    fn from(p: Hypervector) -> Self {
        UnifiedPayload::Hyper(p)
    }
}

impl From<BinaryHypervector> for UnifiedPayload {
    fn from(p: BinaryHypervector) -> Self {
        UnifiedPayload::Binary(p)
    }
}

impl From<SparseGradient> for UnifiedPayload {
    fn from(p: SparseGradient) -> Self {
        UnifiedPayload::Sparse(p)
    }
}

impl From<QuantizedGradient> for UnifiedPayload {
    fn from(p: QuantizedGradient) -> Self {
        UnifiedPayload::Quantized(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_gradient() {
        let gradient = DenseGradient::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(gradient.dimension(), 3);
        assert_eq!(gradient.size_bytes(), 12);
        assert!(gradient.is_valid());
        assert_eq!(gradient.payload_type(), PayloadType::DenseGradient);
    }

    #[test]
    fn test_dense_gradient_validation() {
        let valid = DenseGradient::from_vec(vec![1.0, 2.0]);
        assert!(valid.is_valid());

        let invalid = DenseGradient::from_vec(vec![1.0, f32::NAN]);
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_hypervector() {
        let hv = Hypervector::random(HYPERVECTOR_DIM);
        assert_eq!(hv.dimension(), HYPERVECTOR_DIM);
        assert_eq!(hv.size_bytes(), HYPERVECTOR_DIM);
        assert!(hv.is_valid());
        assert_eq!(hv.payload_type(), PayloadType::HyperEncoded);
    }

    #[test]
    fn test_hypervector_bundle() {
        let hv1 = Hypervector::new(vec![10, 20, 30]);
        let hv2 = Hypervector::new(vec![20, 30, 40]);
        let hv3 = Hypervector::new(vec![30, 40, 50]);

        let bundled = Hypervector::bundle(&[&hv1, &hv2, &hv3]).unwrap();
        assert_eq!(bundled.dimension(), 3);
        // Average: (10+20+30)/3=20, (20+30+40)/3=30, (30+40+50)/3=40
        assert_eq!(bundled.components, vec![20, 30, 40]);
    }

    #[test]
    fn test_hypervector_similarity() {
        let hv1 = Hypervector::new(vec![100, 0, 0]);
        let hv2 = Hypervector::new(vec![100, 0, 0]);
        let sim = hv1.cosine_similarity(&hv2);
        assert!((sim - 1.0).abs() < 0.001);

        let hv3 = Hypervector::new(vec![0, 100, 0]);
        let sim2 = hv1.cosine_similarity(&hv3);
        assert!(sim2.abs() < 0.001);
    }

    #[test]
    fn test_binary_hypervector() {
        let bhv = BinaryHypervector::random(HYPERVECTOR_DIM);
        assert_eq!(bhv.dimension(), HYPERVECTOR_DIM);
        assert_eq!(bhv.size_bytes(), HYPERVECTOR_DIM / 8);
        assert!(bhv.is_valid());
        assert_eq!(bhv.payload_type(), PayloadType::BinaryHypervector);
    }

    #[test]
    fn test_binary_hypervector_operations() {
        let bits1 = vec![true, false, true, false, true, true, false, false];
        let bits2 = vec![true, true, true, false, false, true, false, false];

        let bhv1 = BinaryHypervector::from_bits(&bits1);
        let bhv2 = BinaryHypervector::from_bits(&bits2);

        // XOR
        let xored = bhv1.xor(&bhv2).unwrap();
        // 1^1=0, 0^1=1, 1^1=0, 0^0=0, 1^0=1, 1^1=0, 0^0=0, 0^0=0
        assert_eq!(xored.get_bit(0), Some(false));
        assert_eq!(xored.get_bit(1), Some(true));

        // Hamming distance
        let dist = bhv1.hamming_distance(&bhv2);
        assert_eq!(dist, 2); // Positions 1 and 4 differ
    }

    #[test]
    fn test_binary_hypervector_bundle() {
        let bhv1 = BinaryHypervector::from_bits(&[true, true, false, false]);
        let bhv2 = BinaryHypervector::from_bits(&[true, false, true, false]);
        let bhv3 = BinaryHypervector::from_bits(&[true, true, true, false]);

        let bundled = BinaryHypervector::bundle(&[&bhv1, &bhv2, &bhv3]).unwrap();
        // Majority: 3/3=true, 2/3=true, 2/3=true, 0/3=false
        assert_eq!(bundled.get_bit(0), Some(true));
        assert_eq!(bundled.get_bit(1), Some(true));
        assert_eq!(bundled.get_bit(2), Some(true));
        assert_eq!(bundled.get_bit(3), Some(false));
    }

    #[test]
    fn test_sparse_gradient() {
        let dense = DenseGradient::from_vec(vec![0.0, 1.0, 0.0, 2.0, 0.0]);
        let sparse = SparseGradient::from_dense(&dense, 0.5);

        assert_eq!(sparse.indices, vec![1, 3]);
        assert_eq!(sparse.values, vec![1.0, 2.0]);
        assert_eq!(sparse.full_dimension, 5);

        let recovered = sparse.to_dense();
        assert_eq!(recovered.to_vec(), vec![0.0, 1.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn test_quantized_gradient() {
        let dense = DenseGradient::from_vec(vec![0.0, 0.5, 1.0]);
        let quantized = QuantizedGradient::from_dense(&dense, 8);

        assert!(quantized.is_valid());
        assert_eq!(quantized.dimension, 3);

        let recovered = quantized.to_dense();
        // Should be approximately equal (within quantization error)
        for (a, b) in dense.values.iter().zip(recovered.values.iter()) {
            assert!((a - b).abs() < 0.1);
        }
    }

    #[test]
    fn test_unified_payload() {
        let dense = UnifiedPayload::Dense(DenseGradient::from_vec(vec![1.0, 2.0]));
        assert_eq!(dense.payload_type(), PayloadType::DenseGradient);
        assert_eq!(dense.dimension(), 2);
        assert!(dense.is_valid());
        assert!(dense.to_dense().is_some());

        let hyper = UnifiedPayload::Hyper(Hypervector::random(100));
        assert_eq!(hyper.payload_type(), PayloadType::HyperEncoded);
        assert!(hyper.to_dense().is_none());
    }

    #[test]
    fn test_dense_aggregatable() {
        let dense = DenseGradient::from_vec(vec![1.0, 2.0, 3.0]);
        let converted = dense.to_dense_gradient();
        let back = DenseGradient::from_dense_gradient(converted);
        assert_eq!(dense.to_vec(), back.to_vec());
    }

    #[test]
    fn test_hdc_aggregatable() {
        let hv1 = Hypervector::new(vec![10, 20, 30]);
        let hv2 = Hypervector::new(vec![20, 30, 40]);

        let bundled = Hypervector::bundle_hdc(&[&hv1, &hv2]).unwrap();
        assert_eq!(bundled.components.len(), 3);

        let sim = hv1.similarity(&hv2);
        assert!(sim > 0.0);
    }

    #[test]
    fn test_binary_to_hypervector_conversion() {
        let binary = BinaryHypervector::from_bits(&[true, false, true, false]);
        let hyper = binary.to_hypervector();

        assert_eq!(hyper.components.len(), 4);
        assert_eq!(hyper.components[0], 127);  // true -> 127
        assert_eq!(hyper.components[1], -128); // false -> -128

        let back = BinaryHypervector::from_hypervector(&hyper);
        assert_eq!(back.bit_dimension, 4);
        assert_eq!(back.get_bit(0), Some(true));
        assert_eq!(back.get_bit(1), Some(false));
    }
}
