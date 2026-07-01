//! Substrate metadata for experiments.

/// Backend category used by a benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Classical CPU implementation.
    ClassicalCpu,
    /// Classical GPU implementation.
    ClassicalGpu,
    /// Quantum-inspired classical simulation.
    QuantumInspired,
    /// Quantum circuit exported for external tooling.
    QuantumCircuitExport,
    /// Simulated quantum backend.
    QuantumSimulator,
    /// Physical quantum hardware backend.
    QuantumHardware,
}

/// Confidence label for a substrate claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceLevel {
    /// The claim is speculative and should not be treated as established.
    Speculative,
    /// The claim is supported by local experiments only.
    Experimental,
    /// The claim has external validation or replication.
    Replicated,
    /// The claim is production-calibrated for a specific use.
    ProductionCalibrated,
}

/// Describes the substrate assumptions for one run.
#[derive(Debug, Clone, PartialEq)]
pub struct SubstrateProfile {
    /// Backend category.
    pub backend: BackendKind,
    /// Human-readable backend name.
    pub backend_name: String,
    /// Confidence attached to the result.
    pub confidence: ConfidenceLevel,
    /// Optional qubit count estimate.
    pub qubits_estimate: Option<usize>,
    /// Optional coherence assumption in microseconds.
    pub coherence_us: Option<f32>,
    /// Optional estimated shot count for circuit-style experiments.
    pub shots: Option<usize>,
    /// Notes and caveats.
    pub caveats: Vec<String>,
}

impl SubstrateProfile {
    /// Returns a conservative default for local CPU baselines.
    pub fn classical_cpu() -> Self {
        Self {
            backend: BackendKind::ClassicalCpu,
            backend_name: "classical-cpu-reference".to_string(),
            confidence: ConfidenceLevel::Experimental,
            qubits_estimate: None,
            coherence_us: None,
            shots: None,
            caveats: vec!["Reference implementation; not a quantum backend.".to_string()],
        }
    }

    /// Returns a quantum-inspired simulated substrate profile.
    pub fn quantum_inspired() -> Self {
        Self {
            backend: BackendKind::QuantumInspired,
            backend_name: "phase-hdc-classical-simulation".to_string(),
            confidence: ConfidenceLevel::Speculative,
            qubits_estimate: None,
            coherence_us: None,
            shots: None,
            caveats: vec![
                "Classical simulation of phase-like binding; no quantum advantage claimed."
                    .to_string(),
            ],
        }
    }

    /// Returns a circuit-export profile for tiny external backend experiments.
    pub fn circuit_export(qubits_estimate: usize) -> Self {
        Self {
            backend: BackendKind::QuantumCircuitExport,
            backend_name: "openqasm-export-only".to_string(),
            confidence: ConfidenceLevel::Speculative,
            qubits_estimate: Some(qubits_estimate),
            coherence_us: None,
            shots: None,
            caveats: vec![
                "Circuit export only; not executed by this crate.".to_string(),
                "External backend results must include backend, transpilation, noise model, and shot metadata.".to_string(),
            ],
        }
    }
}
