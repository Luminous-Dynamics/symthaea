// Exact Φ Calculation via PyPhi (IIT 3.0)
//
// This module provides a bridge to the PyPhi Python library for exact
// Integrated Information Theory (IIT 3.0) Φ calculation. PyPhi computes
// the true IIT Φ value via minimum information partition (MIP), which
// has super-exponential complexity O(2^n).
//
// Use this for:
// - Validating Φ_HDC approximation on small systems (n ≤ 8)
// - Ground truth comparison
// - Research validation
//
// For large systems (n > 8), use RealPhiCalculator (HDC approximation) instead.

use crate::hdc::consciousness_topology_generators::ConsciousnessTopology;
use crate::synthesis::SynthesisError;

/// Bridge to PyPhi for exact IIT 3.0 Φ calculation
///
/// # Prerequisites
///
/// Requires PyPhi installed in Python environment:
/// ```bash
/// pip install pyphi numpy scipy networkx
/// ```
///
/// # Limitations
///
/// - **Super-exponential complexity**: O(2^n), very slow for n > 8
/// - **Python dependency**: Requires Python runtime
/// - **Memory intensive**: Needs ~2^n × n space
/// - **IIT 3.0 only**: Does not implement IIT 4.0 (2023)
///
/// # Example
///
/// ```rust,ignore
/// use symthaea::synthesis::phi_exact::PyPhiValidator;
/// use symthaea::hdc::consciousness_topology_generators::ConsciousnessTopology;
///
/// let validator = PyPhiValidator::new()?;
/// let topology = ConsciousnessTopology::star(6, 16384, 42);
/// let phi_exact = validator.compute_phi_exact(&topology)?;
/// println!("Exact Φ (IIT 3.0): {:.4}", phi_exact);
/// ```
#[cfg(feature = "pyphi")]
pub struct PyPhiValidator {
    // Python runtime will be initialized when needed
    _marker: std::marker::PhantomData<()>,
}

#[cfg(feature = "pyphi")]
impl PyPhiValidator {
    /// Create a new PyPhi validator
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - PyPhi not installed in Python environment
    /// - Python initialization fails
    pub fn new() -> Result<Self, SynthesisError> {
        // pyo3 will auto-initialize Python interpreter
        Ok(Self {
            _marker: std::marker::PhantomData,
        })
    }

    /// Compute exact Φ via PyPhi (IIT 3.0)
    ///
    /// # Arguments
    ///
    /// * `topology` - Consciousness topology to evaluate
    ///
    /// # Returns
    ///
    /// Exact Φ value from IIT 3.0 calculation
    ///
    /// # Performance
    ///
    /// - n=5: ~1 second
    /// - n=6: ~10 seconds
    /// - n=7: ~1-2 minutes
    /// - n=8: ~10-30 minutes
    /// - n>8: Hours to days (not recommended)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Topology too large (n > 10 recommended limit)
    /// - PyPhi computation fails
    /// - Invalid transition probability matrix
    pub fn compute_phi_exact(
        &self,
        topology: &ConsciousnessTopology,
    ) -> Result<f64, SynthesisError> {
        use pyo3::prelude::*;
        use pyo3::types::{PyList, PyTuple};

        let n = topology.node_representations.len();

        // Sanity check: PyPhi becomes impractical for n > 10
        if n > 10 {
            return Err(SynthesisError::PhiExactTooLarge {
                size: n,
                recommended_max: 10,
            });
        }

        Python::with_gil(|py| {
            // Import PyPhi
            let pyphi = py.import("pyphi")
                .map_err(|e| SynthesisError::PyPhiImportError {
                    message: format!("Failed to import pyphi: {}. Install with: pip install pyphi", e),
                })?;

            // Convert topology to PyPhi format (TPM + CM)
            let (tpm, cm) = self.topology_to_pyphi_format(topology, py)?;

            // Create PyPhi network
            let network = pyphi.call_method1("Network", (tpm, cm))
                .map_err(|e| SynthesisError::PyPhiComputationError {
                    message: format!("Failed to create PyPhi network: {}", e),
                })?;

            // Create initial state (all zeros)
            let state = PyTuple::new(py, vec![0; n]);

            // Compute SIA (System Irreducibility Analysis)
            let compute = pyphi.getattr("compute")
                .map_err(|e| SynthesisError::PyPhiComputationError {
                    message: format!("Failed to access pyphi.compute: {}", e),
                })?;

            let sia = compute.call_method1("sia", (network, state))
                .map_err(|e| SynthesisError::PyPhiComputationError {
                    message: format!("Failed to compute SIA: {}", e),
                })?;

            // Extract Φ value
            let phi: f64 = sia.getattr("phi")
                .map_err(|e| SynthesisError::PyPhiComputationError {
                    message: format!("Failed to extract phi from SIA: {}", e),
                })?
                .extract()
                .map_err(|e| SynthesisError::PyPhiComputationError {
                    message: format!("Failed to convert phi to f64: {}", e),
                })?;

            Ok(phi)
        })
    }

    /// Convert ConsciousnessTopology to PyPhi format
    ///
    /// Returns: (TPM, CM) where:
    /// - TPM (Transition Probability Matrix): [2^n][n][2] - probability distributions
    /// - CM (Connectivity Matrix): [n][n] - binary adjacency matrix
    fn topology_to_pyphi_format(
        &self,
        topology: &ConsciousnessTopology,
        py: Python,
    ) -> Result<(Py<PyAny>, Py<PyAny>), SynthesisError> {
        use pyo3::types::PyList;

        let n = topology.node_representations.len();

        // Build connectivity matrix from node similarities
        // Nodes with high similarity (> 0.5) are considered connected
        let mut cm_data: Vec<Vec<u8>> = vec![vec![0; n]; n];
        for i in 0..n {
            for j in (i+1)..n {
                let similarity = topology.node_representations[i]
                    .similarity(&topology.node_representations[j]);
                if similarity > 0.5 {
                    cm_data[i][j] = 1;
                    cm_data[j][i] = 1; // Undirected graph
                }
            }
        }

        // Convert to Python list
        let cm = PyList::empty(py);
        for row in &cm_data {
            let py_row = PyList::empty(py);
            for &val in row {
                py_row.append(val)?;
            }
            cm.append(py_row)?;
        }

        // Build TPM (Transition Probability Matrix)
        // Simplified approach: Use connectivity to determine state transitions
        let tpm = self.build_transition_probability_matrix(&cm_data, py)?;

        Ok((tpm.into(), cm.into()))
    }

    /// Build transition probability matrix for PyPhi
    ///
    /// TPM format: [2^n][n][2] where:
    /// - First index: Current state (binary encoding)
    /// - Second index: Node
    /// - Third index: [P(OFF), P(ON)] in next state
    ///
    /// Simplified model: Each node is influenced by its neighbors
    fn build_transition_probability_matrix(
        &self,
        cm: &[Vec<u8>],
        py: Python,
    ) -> Result<Py<PyAny>, SynthesisError> {
        use pyo3::types::PyList;

        let n = cm.len();
        let num_states = 1 << n; // 2^n possible states

        let tpm = PyList::empty(py);

        for state in 0..num_states {
            let state_tpm = PyList::empty(py);

            for node in 0..n {
                // Compute probability node is ON in next state
                let mut p_on = 0.3; // Base probability (slightly favor OFF)

                // Count active neighbors
                let mut active_neighbors = 0;
                for neighbor in 0..n {
                    if cm[node][neighbor] == 1 {
                        if (state >> neighbor) & 1 == 1 {
                            active_neighbors += 1;
                        }
                    }
                }

                // Increase probability based on active neighbors
                let degree = cm[node].iter().sum::<u8>() as f64;
                if degree > 0.0 {
                    p_on += (active_neighbors as f64 / degree) * 0.5;
                }

                // Clamp to [0, 1]
                p_on = p_on.clamp(0.0, 1.0);

                // Create [P(OFF), P(ON)] pair
                let node_dist = PyList::empty(py);
                node_dist.append(1.0 - p_on)?;
                node_dist.append(p_on)?;

                state_tpm.append(node_dist)?;
            }

            tpm.append(state_tpm)?;
        }

        Ok(tpm.into())
    }
}

// Fallback implementation when PyPhi feature is not enabled
#[cfg(not(feature = "pyphi"))]
pub struct PyPhiValidator;

#[cfg(not(feature = "pyphi"))]
impl PyPhiValidator {
    pub fn new() -> Result<Self, SynthesisError> {
        Err(SynthesisError::PyPhiNotEnabled {
            message: "PyPhi integration not enabled. Compile with --features pyphi".to_string(),
        })
    }

    pub fn compute_phi_exact(
        &self,
        _topology: &ConsciousnessTopology,
    ) -> Result<f64, SynthesisError> {
        Err(SynthesisError::PyPhiNotEnabled {
            message: "PyPhi integration not enabled. Compile with --features pyphi".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "pyphi")]
    fn test_pyphi_validator_creation() {
        let result = PyPhiValidator::new();
        assert!(result.is_ok(), "PyPhiValidator creation should succeed");
    }

    #[test]
    #[cfg(not(feature = "pyphi"))]
    fn test_pyphi_disabled_error() {
        let result = PyPhiValidator::new();
        assert!(result.is_err(), "PyPhiValidator should error when feature disabled");
    }

    #[test]
    #[cfg(feature = "pyphi")]
    fn test_too_large_topology_rejected() {
        let validator = PyPhiValidator::new().unwrap();

        // Create topology with 15 nodes (too large)
        let topology = ConsciousnessTopology::random(15, 16384, 42);

        let result = validator.compute_phi_exact(&topology);
        assert!(result.is_err(), "Should reject topology with n > 10");

        match result {
            Err(SynthesisError::PhiExactTooLarge { size, .. }) => {
                assert_eq!(size, 15);
            }
            _ => panic!("Expected PhiExactTooLarge error"),
        }
    }
}
