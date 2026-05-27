// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Neural Bridge: Direct LLM Activation to HDC Conversion
//!
//! This module implements the "Hyperdimensional Probe" - a trained linear
//! projection that maps LLM internal activations directly to Symthaea's
//! 16,384-dimensional HDC space.
//!
//! ## Philosophy
//!
//! Instead of treating the LLM as a "Conversation Partner" (Text -> Text),
//! we treat it as a "Semantic Sensor" (Activations -> Vectors).
//!
//! This eliminates hallucination risk and provides direct access to the
//! LLM's internal semantic representations.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
//! │   Concept    │     │  LLM Residual    │     │   HDC Vector     │
//! │  "Democracy" │────>│  Stream Layer N  │────>│   [16384 bits]   │
//! └──────────────┘     │   [4096 floats]  │     └──────────────────┘
//!                      └────────┬─────────┘
//!                               │
//!                               v
//!                      ┌──────────────────┐
//!                      │  Trained Probe   │
//!                      │   W @ activation │
//!                      │   + sign()       │
//!                      └──────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::perception::NeuralBridge;
//!
//! // Load trained probe weights
//! let bridge = NeuralBridge::load("models/neural_bridge/probe_weights.npy")?;
//!
//! // Get activation from external source (e.g., Python/ONNX)
//! let activation: Vec<f32> = get_llm_activation("Democracy");
//!
//! // Project to HDC space
//! let hdc = bridge.project_to_packed(&activation)?;
//!
//! // Use in consciousness processing
//! let similarity = hdc.xor_similarity(&other_concept);
//! ```
//!
//! ## Training
//!
//! The probe is trained using Python scripts in `scripts/neural_bridge/`:
//!
//! 1. `activation_extractor.py` - Extract activations from LLM
//! 2. `train_probe.py` - Train the linear projection
//!
//! The trained weights are saved as NumPy `.npy` files for cross-language
//! compatibility.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result, bail};

use symthaea_core::hdc::{HDC_DIMENSION, PackedBipolar};

/// Neural Bridge for LLM activation -> HDC conversion.
///
/// Holds a trained linear projection matrix W where:
/// `hdc_vector = sign(W @ activation)`
///
/// The matrix W is trained offline to map LLM hidden states
/// to Symthaea's HDC space while preserving semantic structure.
pub struct NeuralBridge {
    /// Projection matrix: [HDC_DIMENSION, input_dim]
    /// Stored in row-major order
    weights: Vec<f32>,

    /// Input dimension (LLM hidden size, e.g., 2048 or 4096)
    input_dim: usize,

    /// Output dimension (always HDC_DIMENSION = 16384)
    output_dim: usize,
}

impl NeuralBridge {
    /// Load a trained probe from a NumPy `.npy` file.
    ///
    /// The file should contain a 2D float32 array of shape
    /// `[HDC_DIMENSION, input_dim]`.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the `.npy` weights file
    ///
    /// # Returns
    ///
    /// * `Result<Self>` - The loaded neural bridge or an error
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let bridge = NeuralBridge::load("models/neural_bridge/probe_weights.npy")?;
    /// println!("Input dim: {}, Output dim: {}", bridge.input_dim(), bridge.output_dim());
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        let file = File::open(path)
            .with_context(|| format!("Failed to open probe weights: {}", path.display()))?;
        let mut reader = BufReader::new(file);

        let (shape, data) = parse_npy(&mut reader).with_context(|| "Failed to parse NumPy file")?;

        if shape.len() != 2 {
            bail!("Expected 2D array, got {}D", shape.len());
        }

        let (output_dim, input_dim) = (shape[0], shape[1]);

        if output_dim != HDC_DIMENSION {
            bail!("Output dimension mismatch: expected {HDC_DIMENSION}, got {output_dim}");
        }

        Ok(Self {
            weights: data,
            input_dim,
            output_dim,
        })
    }

    /// Create a NeuralBridge with pre-loaded weights.
    ///
    /// Useful for testing or when weights are embedded in the binary.
    ///
    /// # Arguments
    ///
    /// * `weights` - Flattened weight matrix [output_dim * input_dim]
    /// * `input_dim` - Input dimension (LLM hidden size)
    /// * `output_dim` - Output dimension (should be HDC_DIMENSION)
    pub fn from_weights(weights: Vec<f32>, input_dim: usize, output_dim: usize) -> Result<Self> {
        if weights.len() != input_dim * output_dim {
            bail!(
                "Weight size mismatch: expected {}, got {}",
                input_dim * output_dim,
                weights.len()
            );
        }

        Ok(Self {
            weights,
            input_dim,
            output_dim,
        })
    }

    /// Project an LLM activation to continuous HDC space.
    ///
    /// Computes `W @ activation` where W is the trained projection matrix.
    ///
    /// # Arguments
    ///
    /// * `activation` - LLM hidden state vector of length `input_dim`
    ///
    /// # Returns
    ///
    /// * `Result<Vec<f32>>` - Continuous HDC vector of length 16384
    ///
    /// # Errors
    ///
    /// Returns an error if `activation.len() != input_dim`.
    pub fn project(&self, activation: &[f32]) -> Result<Vec<f32>> {
        if activation.len() != self.input_dim {
            bail!(
                "Activation dimension mismatch: expected {}, got {}",
                self.input_dim,
                activation.len()
            );
        }

        // Matrix-vector multiplication: W @ activation
        // W is [output_dim, input_dim] in row-major order
        let mut output = vec![0.0f32; self.output_dim];

        for (i, out_val) in output.iter_mut().enumerate() {
            let row_start = i * self.input_dim;
            let row = &self.weights[row_start..row_start + self.input_dim];

            // Dot product of row with activation
            *out_val = row.iter().zip(activation.iter()).map(|(w, a)| w * a).sum();
        }

        Ok(output)
    }

    /// Project and convert to bipolar representation {-1, +1}.
    ///
    /// Computes `sign(W @ activation)`.
    ///
    /// # Arguments
    ///
    /// * `activation` - LLM hidden state vector
    ///
    /// # Returns
    ///
    /// * `Result<Vec<i8>>` - Bipolar HDC vector
    pub fn project_to_bipolar(&self, activation: &[f32]) -> Result<Vec<i8>> {
        let continuous = self.project(activation)?;

        Ok(continuous
            .into_iter()
            .map(|v| if v > 0.0 { 1i8 } else { -1i8 })
            .collect())
    }

    /// Project to packed bipolar for efficient similarity computation.
    ///
    /// This is the preferred method for integration with Symthaea's
    /// consciousness processing, as PackedBipolar provides fast
    /// XOR-based similarity computation.
    ///
    /// # Arguments
    ///
    /// * `activation` - LLM hidden state vector
    ///
    /// # Returns
    ///
    /// * `Result<PackedBipolar>` - Packed bipolar HDC vector
    pub fn project_to_packed(&self, activation: &[f32]) -> Result<PackedBipolar> {
        let bipolar = self.project_to_bipolar(activation)?;
        Ok(PackedBipolar::from_bipolar(&bipolar))
    }

    /// Get input dimension (LLM hidden size).
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get output dimension (always HDC_DIMENSION = 16384).
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Get a reference to the raw weight matrix.
    ///
    /// Shape is [output_dim, input_dim] in row-major order.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
}

/// Parse a NumPy `.npy` file (v1.0/v2.0 format, float32 little-endian).
///
/// Returns the shape and data as a flat vector.
fn parse_npy<R: Read>(reader: &mut R) -> Result<(Vec<usize>, Vec<f32>)> {
    // Read magic number: "\x93NUMPY"
    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic)?;

    if &magic != b"\x93NUMPY" {
        bail!("Invalid NumPy magic number: {magic:?}");
    }

    // Read version (major, minor)
    let mut version = [0u8; 2];
    reader.read_exact(&mut version)?;

    // Read header length
    let header_len = if version[0] == 1 {
        // Version 1.0: 2-byte header length
        let mut len_bytes = [0u8; 2];
        reader.read_exact(&mut len_bytes)?;
        u16::from_le_bytes(len_bytes) as usize
    } else if version[0] == 2 {
        // Version 2.0: 4-byte header length
        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes)?;
        u32::from_le_bytes(len_bytes) as usize
    } else {
        bail!("Unsupported NumPy version: {}.{}", version[0], version[1]);
    };

    // Read header string
    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header)?;
    let header_str = String::from_utf8_lossy(&header);

    // Parse dtype (must be float32 little-endian)
    if !header_str.contains("'<f4'") && !header_str.contains("float32") {
        bail!("Expected float32 dtype, got header: {}", header_str.trim());
    }

    // Parse shape
    let shape = parse_shape(&header_str)?;

    // Check for Fortran order (we only support C order)
    if header_str.contains("'fortran_order': True") {
        bail!("Fortran order arrays not supported");
    }

    // Calculate number of elements (with overflow protection)
    let n_elements: usize = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| anyhow::anyhow!("NumPy shape overflow: {:?}", shape))?;

    if n_elements > 100_000_000 {
        anyhow::bail!("NumPy array too large: {} elements", n_elements);
    }

    let byte_count = n_elements
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("Byte count overflow for {} elements", n_elements))?;

    // Read data (float32 little-endian) — read raw bytes then convert safely
    let mut raw_bytes = vec![0u8; byte_count];
    reader.read_exact(&mut raw_bytes)?;

    let data: Vec<f32> = raw_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok((shape, data))
}

/// Parse shape tuple from NumPy header string.
///
/// Header looks like: "{'descr': '<f4', 'fortran_order': False, 'shape': (16384, 2048), }"
fn parse_shape(header: &str) -> Result<Vec<usize>> {
    // Find 'shape': (dim1, dim2, ...)
    let shape_start = header
        .find("'shape':")
        .ok_or_else(|| anyhow::anyhow!("Could not find 'shape' in header: {}", header.trim()))?;

    let paren_start = header[shape_start..]
        .find('(')
        .ok_or_else(|| anyhow::anyhow!("Could not find shape tuple opening parenthesis"))?
        + shape_start;

    let paren_end = header[paren_start..]
        .find(')')
        .ok_or_else(|| anyhow::anyhow!("Could not find shape tuple closing parenthesis"))?
        + paren_start;

    let shape_str = &header[paren_start + 1..paren_end];

    // Parse comma-separated dimensions
    let dims: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                trimmed.parse().ok()
            }
        })
        .collect();

    if dims.is_empty() {
        bail!("Could not parse any dimensions from shape: {shape_str}");
    }

    Ok(dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_shape_2d() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (16384, 2048), }";
        let shape = parse_shape(header).unwrap();
        assert_eq!(shape, vec![16384, 2048]);
    }

    #[test]
    fn test_parse_shape_1d() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1024,), }";
        let shape = parse_shape(header).unwrap();
        assert_eq!(shape, vec![1024]);
    }

    #[test]
    fn test_projection_dimensions() {
        // Create a small probe for testing
        let input_dim = 4;
        let output_dim = 8;

        let bridge = NeuralBridge {
            weights: vec![0.1; input_dim * output_dim],
            input_dim,
            output_dim,
        };

        let activation = vec![1.0, 2.0, 3.0, 4.0];
        let result = bridge.project(&activation).unwrap();

        assert_eq!(result.len(), output_dim);
    }

    #[test]
    fn test_bipolar_output_range() {
        let input_dim = 4;
        let output_dim = 8;

        // Weights that produce alternating positive/negative
        let mut weights = vec![0.0; input_dim * output_dim];
        for i in 0..output_dim {
            for j in 0..input_dim {
                weights[i * input_dim + j] = if i % 2 == 0 { 1.0 } else { -1.0 };
            }
        }

        let bridge = NeuralBridge {
            weights,
            input_dim,
            output_dim,
        };

        let activation = vec![1.0; input_dim];
        let bipolar = bridge.project_to_bipolar(&activation).unwrap();

        // All values should be -1 or +1
        assert!(bipolar.iter().all(|&v| v == 1 || v == -1));

        // Check expected pattern
        assert_eq!(bipolar[0], 1); // Even rows have positive weights -> positive output
        assert_eq!(bipolar[1], -1); // Odd rows have negative weights -> negative output
    }

    #[test]
    fn test_dimension_validation() {
        let bridge = NeuralBridge {
            weights: vec![0.0; 4 * 8],
            input_dim: 4,
            output_dim: 8,
        };

        // Wrong dimension should fail
        let wrong_activation = vec![1.0, 2.0, 3.0]; // 3 instead of 4
        assert!(bridge.project(&wrong_activation).is_err());
    }

    #[test]
    fn test_from_weights_validation() {
        // Correct size
        let result = NeuralBridge::from_weights(vec![0.0; 32], 4, 8);
        assert!(result.is_ok());

        // Wrong size
        let result = NeuralBridge::from_weights(vec![0.0; 30], 4, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_npy_shape_overflow_protection() {
        use std::io::Cursor;

        // Build a minimal .npy v1.0 with shape that overflows usize on multiply
        let header_content = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, 2), }}",
            usize::MAX
        );

        // Pad to align (npy spec: header_len + magic(6) + version(2) + len_field(2) = multiple of 64)
        let preamble_len = 6 + 2 + 2; // magic + version + header_len field
        let total_unpadded = preamble_len + header_content.len() + 1; // +1 for newline
        let padded_total = total_unpadded.div_ceil(64) * 64;
        let pad_len = padded_total - total_unpadded;

        let mut header_bytes = header_content.into_bytes();
        header_bytes.extend(std::iter::repeat_n(b' ', pad_len));
        header_bytes.push(b'\n');

        let mut buf = Vec::new();
        buf.extend_from_slice(b"\x93NUMPY");
        buf.extend_from_slice(&[1u8, 0u8]); // version 1.0
        buf.extend_from_slice(&(header_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(&header_bytes);

        let mut cursor = Cursor::new(buf);
        let result = parse_npy(&mut cursor);
        assert!(result.is_err(), "Should fail on overflowing shape");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("overflow") || err_msg.contains("too large"),
            "Error should mention overflow, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_npy_too_large_array() {
        use std::io::Cursor;

        // Shape that doesn't overflow but exceeds the 100M element sanity cap
        let header_content = "{'descr': '<f4', 'fortran_order': False, 'shape': (200000000, 1), }";

        let preamble_len = 6 + 2 + 2;
        let total_unpadded = preamble_len + header_content.len() + 1;
        let padded_total = total_unpadded.div_ceil(64) * 64;
        let pad_len = padded_total - total_unpadded;

        let mut header_bytes = header_content.as_bytes().to_vec();
        header_bytes.extend(std::iter::repeat_n(b' ', pad_len));
        header_bytes.push(b'\n');

        let mut buf = Vec::new();
        buf.extend_from_slice(b"\x93NUMPY");
        buf.extend_from_slice(&[1u8, 0u8]);
        buf.extend_from_slice(&(header_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(&header_bytes);

        let mut cursor = Cursor::new(buf);
        let result = parse_npy(&mut cursor);
        assert!(result.is_err(), "Should reject array > 100M elements");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("too large"),
            "Error should mention 'too large', got: {}",
            err_msg
        );
    }

    #[test]
    fn test_projection_correctness() {
        // Hand-compute expected result
        let input_dim = 2;
        let output_dim = 3;

        // W = [[1, 2],
        //      [3, 4],
        //      [5, 6]]
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let bridge = NeuralBridge {
            weights,
            input_dim,
            output_dim,
        };

        // activation = [1, 1]
        // output[0] = 1*1 + 2*1 = 3
        // output[1] = 3*1 + 4*1 = 7
        // output[2] = 5*1 + 6*1 = 11
        let activation = vec![1.0, 1.0];
        let result = bridge.project(&activation).unwrap();

        assert!((result[0] - 3.0).abs() < 1e-6);
        assert!((result[1] - 7.0).abs() < 1e-6);
        assert!((result[2] - 11.0).abs() < 1e-6);
    }
}
