// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gen-7 HYPERION-FL: Python Bindings via PyO3
//!
//! Provides zkSTARK proof generation and verification for gradient validity
//! in federated learning systems.
//!
//! ## Features
//!
//! - Generate proofs that gradients are valid (no NaN/Inf, bounded norm)
//! - Verify proofs for integration with DHT validators
//! - Dilithium post-quantum authentication for client identity
//!
//! ## Example
//!
//! ```python
//! import gen7_zkstark as zk
//!
//! # Generate a gradient proof
//! gradient = [0.1, -0.2, 0.3, ...]  # Your gradient values
//! proof = zk.prove_gradient_validity(
//!     node_id=bytes(32),
//!     round_number=1,
//!     model_hash=bytes(32),
//!     gradient=gradient,
//! )
//!
//! # Verify the proof
//! result = zk.verify_gradient_proof(proof)
//! print(f"Valid: {result['is_valid']}")
//! ```

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use pyo3::exceptions::PyRuntimeError;

// Phase 2.5: Dilithium authentication
mod dilithium;

// RISC Zero zkVM
use risc0_zkvm::{default_prover, ExecutorEnv, Receipt};
use sha2::{Sha256, Digest};
use serde::{Deserialize, Serialize};

// Import the methods
use methods::{METHOD_ELF, METHOD_ID};

// =============================================================================
// Constants
// =============================================================================

/// Q16.16 scale factor
const FIXED_SCALE: f32 = 65536.0;

/// Default maximum norm squared (1000.0^2 in Q16.16)
const DEFAULT_MAX_NORM_SQUARED: i64 = 1_000_000 * 65536;

// =============================================================================
// Internal Types
// =============================================================================

/// Internal proof inputs structure
#[derive(Debug, Clone)]
struct ProofInputs {
    node_id: [u8; 32],
    round_number: u64,
    gradient_hash: [u8; 32],
    model_hash: [u8; 32],
    max_norm_squared: i64,
    gradient_len: u32,
}

/// Internal proof outputs structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProofOutputs {
    node_id: [u8; 32],
    round_number: u64,
    gradient_hash: [u8; 32],
    model_hash: [u8; 32],
    gradient_len: u32,
    norm_squared: i64,
    mean: i32,
    variance: i32,
    is_valid: bool,
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Convert f32 to Q16.16 fixed-point
fn f32_to_fixed(f: f32) -> i32 {
    (f * FIXED_SCALE) as i32
}

/// Convert Q16.16 fixed-point to f32
fn fixed_to_f32(fixed: i32) -> f32 {
    fixed as f32 / FIXED_SCALE
}

/// Compute SHA256 hash of gradient
fn hash_gradient(gradient: &[i32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for &g in gradient {
        hasher.update(g.to_le_bytes());
    }
    hasher.finalize().into()
}

/// Generate proof internally
fn generate_proof(
    inputs: &ProofInputs,
    gradient: &[i32],
) -> Result<Receipt, Box<dyn std::error::Error>> {
    let mut env_builder = ExecutorEnv::builder();

    // Write public inputs
    env_builder.write(&inputs.node_id)?;
    env_builder.write(&inputs.round_number)?;
    env_builder.write(&inputs.gradient_hash)?;
    env_builder.write(&inputs.model_hash)?;
    env_builder.write(&inputs.max_norm_squared)?;
    env_builder.write(&inputs.gradient_len)?;

    // Write private witness (gradient values)
    for &g in gradient {
        env_builder.write(&g)?;
    }

    let env = env_builder.build()?;

    // Generate proof
    let prover = default_prover();
    let prove_info = prover.prove(env, METHOD_ELF)?;

    Ok(prove_info.receipt)
}

/// Extract outputs from receipt journal
fn extract_outputs(receipt: &Receipt) -> Result<ProofOutputs, Box<dyn std::error::Error>> {
    let bytes = receipt.journal.bytes.as_slice();
    let mut offset = 0;

    fn read_bytes<const N: usize>(data: &[u8], offset: &mut usize) -> [u8; N] {
        let mut arr = [0u8; N];
        arr.copy_from_slice(&data[*offset..*offset + N]);
        *offset += N;
        arr
    }

    fn read_u32(data: &[u8], offset: &mut usize) -> u32 {
        let bytes = read_bytes::<4>(data, offset);
        u32::from_le_bytes(bytes)
    }

    fn read_u64(data: &[u8], offset: &mut usize) -> u64 {
        let bytes = read_bytes::<8>(data, offset);
        u64::from_le_bytes(bytes)
    }

    fn read_i32(data: &[u8], offset: &mut usize) -> i32 {
        let bytes = read_bytes::<4>(data, offset);
        i32::from_le_bytes(bytes)
    }

    fn read_i64(data: &[u8], offset: &mut usize) -> i64 {
        let bytes = read_bytes::<8>(data, offset);
        i64::from_le_bytes(bytes)
    }

    let node_id = read_bytes::<32>(bytes, &mut offset);
    let round_number = read_u64(bytes, &mut offset);
    let gradient_hash = read_bytes::<32>(bytes, &mut offset);
    let model_hash = read_bytes::<32>(bytes, &mut offset);
    let gradient_len = read_u32(bytes, &mut offset);
    let norm_squared = read_i64(bytes, &mut offset);
    let mean = read_i32(bytes, &mut offset);
    let variance = read_i32(bytes, &mut offset);
    let is_valid = bytes[offset] == 1;

    Ok(ProofOutputs {
        node_id,
        round_number,
        gradient_hash,
        model_hash,
        gradient_len,
        norm_squared,
        mean,
        variance,
        is_valid,
    })
}

/// Verify receipt and extract outputs
fn verify_receipt(receipt: &Receipt) -> Result<ProofOutputs, Box<dyn std::error::Error>> {
    receipt.verify(METHOD_ID)?;
    extract_outputs(receipt)
}

// =============================================================================
// Python API
// =============================================================================

/// Generate a zkSTARK proof of gradient validity
///
/// This function proves that:
/// 1. The gradient contains no invalid values
/// 2. The gradient L2 norm is within bounds
/// 3. The gradient matches the claimed commitment
/// 4. The gradient is bound to a specific node and round
///
/// Args:
///     node_id (bytes): 32-byte node identifier (typically public key hash)
///     round_number (int): Training round number for replay protection
///     model_hash (bytes): 32-byte hash of the global model being updated
///     gradient (list[float]): Gradient values to prove
///     max_norm_squared (int, optional): Maximum allowed L2 norm squared (0 = default)
///
/// Returns:
///     bytes: Serialized zkSTARK proof
///
/// Raises:
///     RuntimeError: If proof generation fails
#[pyfunction]
#[pyo3(signature = (node_id, round_number, model_hash, gradient, max_norm_squared=0))]
fn prove_gradient_validity(
    py: Python,
    node_id: Vec<u8>,
    round_number: u64,
    model_hash: Vec<u8>,
    gradient: Vec<f32>,
    max_norm_squared: i64,
) -> PyResult<Py<PyBytes>> {
    // Validate inputs
    if node_id.len() != 32 {
        return Err(PyRuntimeError::new_err(
            format!("node_id must be 32 bytes, got {}", node_id.len())
        ));
    }
    if model_hash.len() != 32 {
        return Err(PyRuntimeError::new_err(
            format!("model_hash must be 32 bytes, got {}", model_hash.len())
        ));
    }
    if gradient.is_empty() {
        return Err(PyRuntimeError::new_err("gradient cannot be empty"));
    }

    // Convert gradient to Q16.16 fixed-point
    let gradient_fixed: Vec<i32> = gradient.iter().map(|&x| f32_to_fixed(x)).collect();
    let gradient_hash = hash_gradient(&gradient_fixed);

    // Convert node_id and model_hash to arrays
    let mut node_id_arr = [0u8; 32];
    node_id_arr.copy_from_slice(&node_id);

    let mut model_hash_arr = [0u8; 32];
    model_hash_arr.copy_from_slice(&model_hash);

    // Create proof inputs
    let inputs = ProofInputs {
        node_id: node_id_arr,
        round_number,
        gradient_hash,
        model_hash: model_hash_arr,
        max_norm_squared: if max_norm_squared > 0 { max_norm_squared } else { DEFAULT_MAX_NORM_SQUARED },
        gradient_len: gradient_fixed.len() as u32,
    };

    // Generate proof
    let receipt = generate_proof(&inputs, &gradient_fixed)
        .map_err(|e| PyRuntimeError::new_err(format!("Proof generation failed: {}", e)))?;

    // Serialize receipt
    let serialized = bincode::serialize(&receipt)
        .map_err(|e| PyRuntimeError::new_err(format!("Serialization failed: {}", e)))?;

    Ok(PyBytes::new(py, &serialized).into())
}

/// Verify a zkSTARK proof of gradient validity
///
/// Args:
///     proof_bytes (bytes): Serialized zkSTARK proof from prove_gradient_validity()
///
/// Returns:
///     dict: {
///         "node_id": bytes,
///         "round_number": int,
///         "gradient_hash": bytes,
///         "model_hash": bytes,
///         "gradient_len": int,
///         "norm_squared": int,
///         "norm": float,  # L2 norm as float
///         "mean": float,
///         "variance": float,
///         "is_valid": bool
///     }
///
/// Raises:
///     RuntimeError: If verification fails
#[pyfunction]
fn verify_gradient_proof(py: Python, proof_bytes: &[u8]) -> PyResult<PyObject> {
    // Deserialize receipt
    let receipt: Receipt = bincode::deserialize(proof_bytes)
        .map_err(|e| PyRuntimeError::new_err(format!("Deserialization failed: {}", e)))?;

    // Verify proof
    let outputs = verify_receipt(&receipt)
        .map_err(|e| PyRuntimeError::new_err(format!("Verification failed: {}", e)))?;

    // Create Python dict
    let result = PyDict::new(py);
    result.set_item("node_id", PyBytes::new(py, &outputs.node_id))?;
    result.set_item("round_number", outputs.round_number)?;
    result.set_item("gradient_hash", PyBytes::new(py, &outputs.gradient_hash))?;
    result.set_item("model_hash", PyBytes::new(py, &outputs.model_hash))?;
    result.set_item("gradient_len", outputs.gradient_len)?;
    result.set_item("norm_squared", outputs.norm_squared)?;
    result.set_item("norm", ((outputs.norm_squared as f64) / (FIXED_SCALE as f64)).sqrt())?;
    result.set_item("mean", fixed_to_f32(outputs.mean))?;
    result.set_item("variance", fixed_to_f32(outputs.variance))?;
    result.set_item("is_valid", outputs.is_valid)?;

    Ok(result.into())
}

/// Compute SHA256 hash of gradient (for commitment)
///
/// Args:
///     gradient (list[float]): Gradient values
///
/// Returns:
///     bytes: 32-byte SHA256 hash
#[pyfunction]
fn hash_gradient_py(py: Python, gradient: Vec<f32>) -> PyResult<Py<PyBytes>> {
    let gradient_fixed: Vec<i32> = gradient.iter().map(|&x| f32_to_fixed(x)).collect();
    let hash = hash_gradient(&gradient_fixed);
    Ok(PyBytes::new(py, &hash).into())
}

/// Compute gradient statistics
///
/// Args:
///     gradient (list[float]): Gradient values
///
/// Returns:
///     dict: {
///         "len": int,
///         "norm_squared": float,
///         "norm": float,
///         "mean": float,
///         "variance": float,
///         "min": float,
///         "max": float
///     }
#[pyfunction]
fn compute_gradient_stats(py: Python, gradient: Vec<f32>) -> PyResult<PyObject> {
    if gradient.is_empty() {
        let result = PyDict::new(py);
        result.set_item("len", 0)?;
        result.set_item("norm_squared", 0.0)?;
        result.set_item("norm", 0.0)?;
        result.set_item("mean", 0.0)?;
        result.set_item("variance", 0.0)?;
        result.set_item("min", 0.0)?;
        result.set_item("max", 0.0)?;
        return Ok(result.into());
    }

    let len = gradient.len();

    // Compute norm squared
    let norm_squared: f64 = gradient.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let norm = norm_squared.sqrt();

    // Compute mean
    let sum: f64 = gradient.iter().map(|&x| x as f64).sum();
    let mean = sum / len as f64;

    // Compute variance
    let variance: f64 = gradient.iter()
        .map(|&x| {
            let diff = (x as f64) - mean;
            diff * diff
        })
        .sum::<f64>() / len as f64;

    // Find min/max
    let min = gradient.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = gradient.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let result = PyDict::new(py);
    result.set_item("len", len)?;
    result.set_item("norm_squared", norm_squared)?;
    result.set_item("norm", norm)?;
    result.set_item("mean", mean)?;
    result.set_item("variance", variance)?;
    result.set_item("min", min)?;
    result.set_item("max", max)?;

    Ok(result.into())
}

/// Check if a gradient would pass validity checks
///
/// This is a local check without generating a proof. Useful for
/// pre-validation before expensive proof generation.
///
/// Args:
///     gradient (list[float]): Gradient values
///     max_norm (float, optional): Maximum allowed L2 norm (default: 1000.0)
///
/// Returns:
///     tuple[bool, str]: (is_valid, message)
#[pyfunction]
#[pyo3(signature = (gradient, max_norm=1000.0))]
fn check_gradient_validity(gradient: Vec<f32>, max_norm: f64) -> (bool, String) {
    if gradient.is_empty() {
        return (false, "Gradient is empty".to_string());
    }

    // Check for NaN/Inf
    for (i, &g) in gradient.iter().enumerate() {
        if g.is_nan() {
            return (false, format!("Gradient contains NaN at index {}", i));
        }
        if g.is_infinite() {
            return (false, format!("Gradient contains Infinity at index {}", i));
        }
    }

    // Check norm
    let norm_squared: f64 = gradient.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let norm = norm_squared.sqrt();

    if norm > max_norm {
        return (false, format!("Gradient norm {} exceeds maximum {}", norm, max_norm));
    }

    if norm < 1e-10 {
        return (false, "Gradient norm is too small (near-zero gradient)".to_string());
    }

    (true, "Gradient is valid".to_string())
}

/// Convert f32 to Q16.16 fixed-point
#[pyfunction]
fn to_fixed(value: f32) -> i32 {
    f32_to_fixed(value)
}

/// Convert Q16.16 fixed-point to f32
#[pyfunction]
fn from_fixed(value: i32) -> f32 {
    fixed_to_f32(value)
}

// =============================================================================
// Legacy API (backward compatibility)
// =============================================================================

/// [DEPRECATED] Use prove_gradient_validity instead
///
/// This function is kept for backward compatibility with existing code.
#[pyfunction]
#[pyo3(signature = (model_params, gradient, local_data, local_labels, num_samples, input_dim, num_classes, epochs, learning_rate))]
#[allow(unused_variables)]
fn prove_gradient_zkstark(
    py: Python,
    model_params: Vec<f32>,
    gradient: Vec<f32>,
    local_data: Vec<f32>,
    local_labels: Vec<u8>,
    num_samples: usize,
    input_dim: usize,
    num_classes: usize,
    epochs: u32,
    learning_rate: f32,
) -> PyResult<Py<PyBytes>> {
    // Create a deterministic node_id and model_hash from inputs
    let mut hasher = Sha256::new();
    hasher.update(b"node_id_legacy");
    let node_id: Vec<u8> = hasher.finalize().to_vec();

    let mut hasher = Sha256::new();
    for &p in &model_params {
        hasher.update(p.to_le_bytes());
    }
    let model_hash: Vec<u8> = hasher.finalize().to_vec();

    prove_gradient_validity(py, node_id, epochs as u64, model_hash, gradient, 0)
}

/// [DEPRECATED] Use verify_gradient_proof instead
#[pyfunction]
fn verify_gradient_zkstark(py: Python, proof_bytes: &[u8]) -> PyResult<PyObject> {
    verify_gradient_proof(py, proof_bytes)
}

/// [DEPRECATED] Use hash_gradient_py instead
#[pyfunction]
fn hash_model_params_py(py: Python, model_params: Vec<f32>) -> PyResult<Py<PyBytes>> {
    hash_gradient_py(py, model_params)
}

// =============================================================================
// Python Module
// =============================================================================

/// gen7_zkstark: zkSTARK proof generation for gradient validity
///
/// This module provides cryptographic proofs that gradients are valid
/// for use in federated learning systems.
///
/// ## Quick Start
///
/// ```python
/// import gen7_zkstark as zk
///
/// # Check gradient validity (fast, local)
/// is_valid, msg = zk.check_gradient_validity(gradient)
///
/// # Generate proof (slow, cryptographic)
/// proof = zk.prove_gradient_validity(
///     node_id=my_node_id,
///     round_number=current_round,
///     model_hash=model_hash,
///     gradient=gradient,
/// )
///
/// # Verify proof
/// result = zk.verify_gradient_proof(proof)
/// assert result["is_valid"]
/// ```
#[pymodule]
fn gen7_zkstark(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core proof functions
    m.add_function(wrap_pyfunction!(prove_gradient_validity, m)?)?;
    m.add_function(wrap_pyfunction!(verify_gradient_proof, m)?)?;

    // Utility functions
    m.add_function(wrap_pyfunction!(hash_gradient_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_gradient_stats, m)?)?;
    m.add_function(wrap_pyfunction!(check_gradient_validity, m)?)?;
    m.add_function(wrap_pyfunction!(to_fixed, m)?)?;
    m.add_function(wrap_pyfunction!(from_fixed, m)?)?;

    // Legacy API (deprecated)
    m.add_function(wrap_pyfunction!(prove_gradient_zkstark, m)?)?;
    m.add_function(wrap_pyfunction!(verify_gradient_zkstark, m)?)?;
    m.add_function(wrap_pyfunction!(hash_model_params_py, m)?)?;

    // Phase 2.5: Dilithium authentication
    dilithium::register_dilithium_module(m)?;

    Ok(())
}
