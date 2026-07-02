// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Python bindings for Winterfell PoGQ prover
//!
//! Enables Python FL code to generate and verify PoGQ proofs.

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use crate::air::PublicInputs;
#[cfg(feature = "python")]
use crate::PoGQProver;

/// Python-accessible proof result
#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone)]
pub struct PyProofResult {
    #[pyo3(get)]
    pub proof_bytes: Vec<u8>,

    #[pyo3(get)]
    pub total_ms: u128,

    #[pyo3(get)]
    pub proof_size_kb: usize,
}

/// Python-accessible public inputs
#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone)]
pub struct PyPublicInputs {
    #[pyo3(get, set)]
    pub beta: u64,

    #[pyo3(get, set)]
    pub w: u64,

    #[pyo3(get, set)]
    pub k: u64,

    #[pyo3(get, set)]
    pub m: u64,

    #[pyo3(get, set)]
    pub threshold: u64,

    #[pyo3(get, set)]
    pub ema_init: u64,

    #[pyo3(get, set)]
    pub viol_init: u64,

    #[pyo3(get, set)]
    pub clear_init: u64,

    #[pyo3(get, set)]
    pub quar_init: u64,

    #[pyo3(get, set)]
    pub round_init: u64,

    #[pyo3(get, set)]
    pub quar_out: u64,

    #[pyo3(get, set)]
    pub trace_length: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyPublicInputs {
    #[new]
    pub fn new(
        beta: u64,
        w: u64,
        k: u64,
        m: u64,
        threshold: u64,
        ema_init: u64,
        viol_init: u64,
        clear_init: u64,
        quar_init: u64,
        round_init: u64,
        quar_out: u64,
        trace_length: usize,
    ) -> Self {
        Self {
            beta,
            w,
            k,
            m,
            threshold,
            ema_init,
            viol_init,
            clear_init,
            quar_init,
            round_init,
            quar_out,
            trace_length,
        }
    }

}

// Helper function to convert PyPublicInputs to internal PublicInputs
#[cfg(feature = "python")]
impl PyPublicInputs {
    fn to_rust(&self) -> PublicInputs {
        PublicInputs {
            beta: self.beta,
            w: self.w,
            k: self.k,
            m: self.m,
            threshold: self.threshold,
            ema_init: self.ema_init,
            viol_init: self.viol_init,
            clear_init: self.clear_init,
            quar_init: self.quar_init,
            round_init: self.round_init,
            quar_out: self.quar_out,
            trace_length: self.trace_length,
        }
    }
}

/// Python-accessible PoGQ prover
#[cfg(feature = "python")]
#[pyclass]
pub struct PyPoGQProver {
    prover: PoGQProver,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyPoGQProver {
    #[new]
    pub fn new() -> Self {
        Self {
            prover: PoGQProver::new(),
        }
    }

    /// Generate a PoGQ proof
    ///
    /// # Arguments
    /// * `public_inputs` - Public parameters and expected output
    /// * `witness_scores` - Quality scores for each round (private witness)
    ///
    /// # Returns
    /// PyProofResult containing proof bytes, timing, and size
    pub fn prove_exec(
        &self,
        public_inputs: PyPublicInputs,
        witness_scores: Vec<u64>,
    ) -> PyResult<PyProofResult> {
        let public = public_inputs.to_rust();

        let result = self.prover.prove_exec(public, witness_scores)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Proof generation failed: {:?}", e)
            ))?;

        let proof_size_kb = result.proof_bytes.len() / 1024;

        Ok(PyProofResult {
            proof_bytes: result.proof_bytes,
            total_ms: result.total_ms as u128,  // Convert u64 to u128 for Python
            proof_size_kb,
        })
    }

    /// Verify a PoGQ proof
    ///
    /// # Arguments
    /// * `proof_bytes` - Serialized proof
    /// * `public_inputs` - Public parameters to verify against
    ///
    /// # Returns
    /// True if proof is valid, False otherwise
    pub fn verify_proof(
        &self,
        proof_bytes: Vec<u8>,
        public_inputs: PyPublicInputs,
    ) -> PyResult<bool> {
        let public = public_inputs.to_rust();

        let valid = self.prover.verify_proof(&proof_bytes, public)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Proof verification failed: {:?}", e)
            ))?;

        Ok(valid)
    }
}

/// Helper function to convert float (0.0-1.0) to Q16.16 fixed-point
#[cfg(feature = "python")]
#[pyfunction]
pub fn scale_float(val: f64) -> u64 {
    (val * 65536.0) as u64
}

/// Python module definition
#[cfg(feature = "python")]
#[pymodule]
fn winterfell_pogq(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPoGQProver>()?;
    m.add_class::<PyPublicInputs>()?;
    m.add_class::<PyProofResult>()?;
    m.add_function(wrap_pyfunction!(scale_float, m)?)?;
    Ok(())
}
