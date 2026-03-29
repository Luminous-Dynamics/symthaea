// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// PyO3 wrapper for VSV-STARK

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use vsv_stark_air::{generate_proof, verify_proof, PublicInputs, PrivateWitness};

/// Generate a STARK proof for one round of PoGQ decision logic
///
/// Args:
///     public_json (dict): Public inputs as Python dict
///     witness_json (dict): Private witness as Python dict
///
/// Returns:
///     tuple[bytes, str]: (proof_bytes, proof_hex)
///
/// Example:
///     ```python
///     public = {
///         "h_calib": "a" * 64,
///         "h_model": "b" * 64,
///         "h_grad": "c" * 64,
///         "beta_fp": 55705,
///         "w": 3,
///         "k": 2,
///         "m": 3,
///         "egregious_cap_fp": 65535,
///         "threshold_fp": 58982,
///         "ema_prev_fp": 49152,
///         "consec_viol_prev": 1,
///         "consec_clear_prev": 0,
///         "quarantined_prev": 0,
///         "current_round": 5,
///         "quarantine_out": 0
///     }
///     witness = {
///         "x_t_fp": 52428,
///         "in_warmup": 0,
///         "violation_t": 1,
///         "release_t": 0
///     }
///     proof_bytes, proof_hex = generate_round_proof(public, witness)
///     ```
#[pyfunction]
fn generate_round_proof(
    py: Python,
    public_json: &PyDict,
    witness_json: &PyDict,
) -> PyResult<(PyObject, String)> {
    // Convert Python dict to JSON string
    let public_str = serde_json::to_string(&public_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Failed to serialize public inputs: {}", e)
        ))?;

    let witness_str = serde_json::to_string(&witness_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Failed to serialize witness: {}", e)
        ))?;

    // Parse to Rust structs
    let public = PublicInputs::from_json(&public_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    let witness = PrivateWitness::from_json(&witness_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    // Generate proof
    let proof_bytes = generate_proof(&public, &witness)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    // Convert to hex string
    let proof_hex = hex::encode(&proof_bytes);

    // Create PyBytes object
    let py_bytes = PyBytes::new(py, &proof_bytes).into();

    Ok((py_bytes, proof_hex))
}

/// Verify a STARK proof
///
/// Args:
///     proof_bytes (bytes): Serialized STARK proof
///     public_json_str (str): Public inputs as JSON string
///
/// Returns:
///     bool: True if proof is valid, False otherwise
///
/// Example:
///     ```python
///     import json
///
///     public = {...}  # Same as above
///     proof_bytes, _ = generate_round_proof(public, witness)
///
///     is_valid = verify_round_proof(proof_bytes, json.dumps(public))
///     assert is_valid
///     ```
#[pyfunction]
fn verify_round_proof(
    proof_bytes: &[u8],
    public_json_str: &str,
) -> PyResult<bool> {
    // Parse public inputs
    let public = PublicInputs::from_json(public_json_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    // Verify proof
    let is_valid = verify_proof(proof_bytes, &public)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    Ok(is_valid)
}

/// Convert a float value to fixed-point representation
///
/// Args:
///     value (float): Value in range [0, 1]
///
/// Returns:
///     int: Fixed-point representation (value * 65536)
#[pyfunction]
fn to_fixed(value: f64) -> PyResult<u64> {
    if value < 0.0 || value > 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Value must be in [0, 1], got {}", value)
        ));
    }
    Ok(vsv_stark_air::to_fixed(value))
}

/// Convert a fixed-point value to float
///
/// Args:
///     value_fp (int): Fixed-point representation
///
/// Returns:
///     float: Float value in range [0, 1]
#[pyfunction]
fn from_fixed(value_fp: u64) -> PyResult<f64> {
    Ok(vsv_stark_air::from_fixed(value_fp))
}

/// VSV-STARK: Verifiable PoGQ Decision Logic
///
/// This module provides STARK proofs for Byzantine detection in federated learning.
/// It proves the correctness of PoGQ-v4.1's per-round decision logic:
/// - EMA update
/// - Warm-up logic
/// - Hysteresis counters
/// - Conformal threshold comparison
/// - Quarantine output
#[pymodule]
fn vsv_stark(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_round_proof, m)?)?;
    m.add_function(wrap_pyfunction!(verify_round_proof, m)?)?;
    m.add_function(wrap_pyfunction!(to_fixed, m)?)?;
    m.add_function(wrap_pyfunction!(from_fixed, m)?)?;

    // Add version constant
    m.add("__version__", "0.1.0")?;

    // Add scale constant
    m.add("SCALE", vsv_stark_air::SCALE)?;

    Ok(())
}
