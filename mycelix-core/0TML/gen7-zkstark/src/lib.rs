// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Gen-7 HYPERION-FL: Python Bindings via PyO3
//
// Exports zkSTARK proof generation and verification to Python

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::exceptions::PyRuntimeError;

// Re-export host functionality
use risc0_zkvm::{default_prover, ExecutorEnv, Receipt};
use sha2::{Sha256, Digest};
use serde::{Deserialize, Serialize};

// Import the methods
use methods::{METHOD_ELF, METHOD_ID};

/// Public inputs for gradient provenance proof
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProofInputs {
    model_hash: [u8; 32],
    gradient_hash: [u8; 32],
    epochs: u32,
    lr_fixed: i32,
}

/// Private witness data
#[derive(Debug, Clone)]
struct PrivateWitness {
    local_data: Vec<i32>,
    local_labels: Vec<u8>,
    model_params: Vec<i32>,
    num_samples: usize,
    input_dim: usize,
    num_classes: usize,
}

/// Generate zkSTARK proof (internal function)
fn generate_proof(
    public_inputs: &ProofInputs,
    private_witness: &PrivateWitness,
) -> Result<Receipt, Box<dyn std::error::Error>> {
    let mut env_builder = ExecutorEnv::builder();

    // Write public inputs
    env_builder.write(&public_inputs.model_hash)?;
    env_builder.write(&public_inputs.gradient_hash)?;
    env_builder.write(&public_inputs.epochs)?;
    env_builder.write(&public_inputs.lr_fixed)?;

    // Write private witness
    env_builder.write(&private_witness.num_samples)?;
    env_builder.write(&private_witness.input_dim)?;
    for &x in &private_witness.local_data {
        env_builder.write(&x)?;
    }
    env_builder.write(&private_witness.num_classes)?;
    for &y in &private_witness.local_labels {
        env_builder.write(&y)?;
    }
    env_builder.write(&private_witness.model_params.len())?;
    for &param in &private_witness.model_params {
        env_builder.write(&param)?;
    }

    let env = env_builder.build()?;

    // Generate proof
    let prover = default_prover();
    let prove_info = prover.prove(env, METHOD_ELF)?;

    Ok(prove_info.receipt)
}

/// Verify zkSTARK proof (internal function)
fn verify_receipt(receipt: &Receipt) -> Result<(Vec<u8>, u32, usize), Box<dyn std::error::Error>> {
    receipt.verify(METHOD_ID)?;

    let gradient_hash: [u8; 32] = receipt.journal.decode()?;
    let epochs: u32 = receipt.journal.decode()?;
    let num_samples: usize = receipt.journal.decode()?;

    Ok((gradient_hash.to_vec(), epochs, num_samples))
}

/// Helper: Hash model parameters
fn hash_model_params(params: &[i32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for &param in params {
        hasher.update(param.to_le_bytes());
    }
    hasher.finalize().into()
}

/// Helper: Hash gradient
fn hash_gradient(gradient: &[i32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for &g in gradient {
        hasher.update(g.to_le_bytes());
    }
    hasher.finalize().into()
}

/// Helper: Convert f32 to Q16.16 fixed-point
fn f32_to_fixed(f: f32) -> i32 {
    (f * 65536.0) as i32
}

/// Helper: Convert Q16.16 to f32
fn fixed_to_f32(fixed: i32) -> f32 {
    fixed as f32 / 65536.0
}

/// Python wrapper: Generate zkSTARK proof for gradient provenance
///
/// Arguments:
///     model_params (list[float]): Initial model parameters
///     gradient (list[float]): Computed gradient to prove
///     local_data (list[float]): Private training data (flattened)
///     local_labels (list[int]): Private labels (one-hot or class indices)
///     num_samples (int): Number of training samples
///     input_dim (int): Dimension of each sample
///     num_classes (int): Number of output classes
///     epochs (int): Number of training epochs
///     learning_rate (float): Learning rate used
///
/// Returns:
///     bytes: Serialized zkSTARK proof (Receipt)
///
/// Raises:
///     RuntimeError: If proof generation fails
#[pyfunction]
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
    // Convert to Q16.16 fixed-point
    let model_params_fixed: Vec<i32> = model_params.iter().map(|&x| f32_to_fixed(x)).collect();
    let gradient_fixed: Vec<i32> = gradient.iter().map(|&x| f32_to_fixed(x)).collect();
    let local_data_fixed: Vec<i32> = local_data.iter().map(|&x| f32_to_fixed(x)).collect();

    // Compute hashes
    let model_hash = hash_model_params(&model_params_fixed);
    let gradient_hash = hash_gradient(&gradient_fixed);

    // Create proof inputs
    let public_inputs = ProofInputs {
        model_hash,
        gradient_hash,
        epochs,
        lr_fixed: f32_to_fixed(learning_rate),
    };

    let private_witness = PrivateWitness {
        local_data: local_data_fixed,
        local_labels,
        model_params: model_params_fixed,
        num_samples,
        input_dim,
        num_classes,
    };

    // Generate proof
    let receipt = generate_proof(&public_inputs, &private_witness)
        .map_err(|e| PyRuntimeError::new_err(format!("Proof generation failed: {}", e)))?;

    // Serialize receipt
    let serialized = bincode::serialize(&receipt)
        .map_err(|e| PyRuntimeError::new_err(format!("Serialization failed: {}", e)))?;

    Ok(PyBytes::new(py, &serialized).into())
}

/// Python wrapper: Verify zkSTARK proof
///
/// Arguments:
///     proof_bytes (bytes): Serialized zkSTARK proof from prove_gradient_zkstark()
///
/// Returns:
///     dict: {
///         "gradient_hash": bytes,
///         "epochs": int,
///         "num_samples": int,
///         "verified": bool
///     }
///
/// Raises:
///     RuntimeError: If verification fails
#[pyfunction]
fn verify_gradient_zkstark(py: Python, proof_bytes: &[u8]) -> PyResult<PyObject> {
    // Deserialize receipt
    let receipt: Receipt = bincode::deserialize(proof_bytes)
        .map_err(|e| PyRuntimeError::new_err(format!("Deserialization failed: {}", e)))?;

    // Verify proof
    let (gradient_hash, epochs, num_samples) = verify_receipt(&receipt)
        .map_err(|e| PyRuntimeError::new_err(format!("Verification failed: {}", e)))?;

    // Create Python dict
    let result = pyo3::types::PyDict::new(py);
    result.set_item("gradient_hash", PyBytes::new(py, &gradient_hash))?;
    result.set_item("epochs", epochs)?;
    result.set_item("num_samples", num_samples)?;
    result.set_item("verified", true)?;

    Ok(result.into())
}

/// Python wrapper: Hash model parameters (for testing/debugging)
#[pyfunction]
fn hash_model_params_py(py: Python, model_params: Vec<f32>) -> PyResult<Py<PyBytes>> {
    let model_params_fixed: Vec<i32> = model_params.iter().map(|&x| f32_to_fixed(x)).collect();
    let hash = hash_model_params(&model_params_fixed);
    Ok(PyBytes::new(py, &hash).into())
}

/// Python wrapper: Hash gradient (for testing/debugging)
#[pyfunction]
fn hash_gradient_py(py: Python, gradient: Vec<f32>) -> PyResult<Py<PyBytes>> {
    let gradient_fixed: Vec<i32> = gradient.iter().map(|&x| f32_to_fixed(x)).collect();
    let hash = hash_gradient(&gradient_fixed);
    Ok(PyBytes::new(py, &hash).into())
}

/// Python module: gen7_zkstark
///
/// Provides zkSTARK proof generation and verification for Gen-7 HYPERION-FL
#[pymodule]
fn gen7_zkstark(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prove_gradient_zkstark, m)?)?;
    m.add_function(wrap_pyfunction!(verify_gradient_zkstark, m)?)?;
    m.add_function(wrap_pyfunction!(hash_model_params_py, m)?)?;
    m.add_function(wrap_pyfunction!(hash_gradient_py, m)?)?;
    Ok(())
}
