// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Quantum Information
//!
//! This module encodes quantum information concepts from qubits to quantum algorithms.
//!
//! ## Core Concepts
//!
//! - **Qubit**: Quantum bit, superposition of |0⟩ and |1⟩
//! - **Entanglement**: Non-classical correlations (Bell states, GHZ states)
//! - **Quantum Gates**: Unitary operations (Pauli, Hadamard, CNOT, etc.)
//! - **Measurement**: Collapse of superposition
//!
//! ## Quantum Error Correction
//!
//! - **Bit-flip code**: Protects against X errors
//! - **Phase-flip code**: Protects against Z errors
//! - **Shor code**: First complete QEC code
//! - **Surface codes**: Topological protection
//!
//! ## Quantum Algorithms
//!
//! - **Deutsch-Jozsa**: Exponential speedup for oracle problems
//! - **Grover**: Quadratic speedup for search
//! - **Shor**: Exponential speedup for factoring
//! - **Quantum simulation**: Simulating quantum systems

use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Pauli matrices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PauliMatrix {
    I, // Identity
    X, // Bit flip
    Y, // Combined
    Z, // Phase flip
}

impl PauliMatrix {
    fn domain(&self) -> &'static str {
        match self {
            PauliMatrix::I => "qi::pauli_i",
            PauliMatrix::X => "qi::pauli_x",
            PauliMatrix::Y => "qi::pauli_y",
            PauliMatrix::Z => "qi::pauli_z",
        }
    }
}

/// Quantum gate type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateType {
    // Single qubit gates
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
    Phase, // S gate
    TGate, // T gate (π/8)
    // Two qubit gates
    CNOT,
    CZ,
    SWAP,
    // Three qubit gates
    Toffoli,
    Fredkin,
}

impl GateType {
    fn domain(&self) -> &'static str {
        match self {
            GateType::Hadamard => "qi::hadamard",
            GateType::PauliX => "qi::pauli_x",
            GateType::PauliY => "qi::pauli_y",
            GateType::PauliZ => "qi::pauli_z",
            GateType::Phase => "qi::phase",
            GateType::TGate => "qi::t_gate",
            GateType::CNOT => "qi::cnot",
            GateType::CZ => "qi::cz",
            GateType::SWAP => "qi::swap",
            GateType::Toffoli => "qi::toffoli",
            GateType::Fredkin => "qi::fredkin",
        }
    }

    /// Number of qubits this gate acts on
    pub fn num_qubits(&self) -> u8 {
        match self {
            GateType::Hadamard
            | GateType::PauliX
            | GateType::PauliY
            | GateType::PauliZ
            | GateType::Phase
            | GateType::TGate => 1,
            GateType::CNOT | GateType::CZ | GateType::SWAP => 2,
            GateType::Toffoli | GateType::Fredkin => 3,
        }
    }
}

/// Entanglement type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntanglementType {
    /// Bell states (2 qubits, maximally entangled)
    Bell,
    /// GHZ state (N qubits: |00...0⟩ + |11...1⟩)
    GHZ,
    /// W state (N qubits: |100...0⟩ + |010...0⟩ + ...)
    W,
    /// Cluster state (graph state for MBQC)
    Cluster,
}

impl EntanglementType {
    fn domain(&self) -> &'static str {
        match self {
            EntanglementType::Bell => "qi::bell",
            EntanglementType::GHZ => "qi::ghz",
            EntanglementType::W => "qi::w_state",
            EntanglementType::Cluster => "qi::cluster",
        }
    }
}

/// Error correction code type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QECCode {
    BitFlip3,   // 3-qubit repetition code
    PhaseFlip3, // 3-qubit phase flip code
    Shor9,      // Shor's 9-qubit code
    Steane7,    // Steane's 7-qubit code
    Surface,    // Surface code (topological)
    Color,      // Color code
}

impl QECCode {
    fn domain(&self) -> &'static str {
        match self {
            QECCode::BitFlip3 => "qec::bit_flip_3",
            QECCode::PhaseFlip3 => "qec::phase_flip_3",
            QECCode::Shor9 => "qec::shor_9",
            QECCode::Steane7 => "qec::steane_7",
            QECCode::Surface => "qec::surface",
            QECCode::Color => "qec::color",
        }
    }

    /// Number of physical qubits
    pub fn physical_qubits(&self) -> u32 {
        match self {
            QECCode::BitFlip3 | QECCode::PhaseFlip3 => 3,
            QECCode::Steane7 => 7,
            QECCode::Shor9 => 9,
            QECCode::Surface | QECCode::Color => 17, // Typical small instance
        }
    }

    /// Number of logical qubits encoded
    pub fn logical_qubits(&self) -> u32 {
        1 // All these encode 1 logical qubit
    }
}

/// A qubit state
#[derive(Debug, Clone)]
pub struct Qubit {
    /// Probability amplitude of |0⟩
    pub alpha: f64,
    /// Probability amplitude of |1⟩
    pub beta: f64,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// A quantum gate
#[derive(Debug, Clone)]
pub struct QuantumGate {
    pub gate_type: GateType,
    pub vector: ContinuousHV,
}

/// An entangled state
#[derive(Debug, Clone)]
pub struct EntangledState {
    pub entanglement_type: EntanglementType,
    pub num_qubits: u32,
    /// Concurrence (0 = separable, 1 = maximally entangled)
    pub concurrence: f64,
    pub vector: ContinuousHV,
}

/// Quantum Information encoder
#[derive(Debug, Clone)]
pub struct QIEncoder {
    // Basis states
    pub ket_zero: ContinuousHV,
    pub ket_one: ContinuousHV,
    pub superposition: ContinuousHV,

    // Pauli operators
    pub pauli_i: ContinuousHV,
    pub pauli_x: ContinuousHV,
    pub pauli_y: ContinuousHV,
    pub pauli_z: ContinuousHV,

    // Key gates
    pub hadamard: ContinuousHV,
    pub cnot: ContinuousHV,
    pub t_gate: ContinuousHV,

    // Entanglement
    pub entanglement: ContinuousHV,
    pub bell_state: ContinuousHV,
    pub ghz_state: ContinuousHV,

    // Measurement
    pub measurement: ContinuousHV,
    pub collapse: ContinuousHV,
    pub projection: ContinuousHV,

    // Error correction
    pub error: ContinuousHV,
    pub syndrome: ContinuousHV,
    pub correction: ContinuousHV,
    pub logical_qubit: ContinuousHV,

    // Decoherence
    pub decoherence: ContinuousHV,
    pub t1_decay: ContinuousHV,     // Amplitude damping
    pub t2_dephasing: ContinuousHV, // Phase damping

    // Information concepts
    pub entropy: ContinuousHV,
    pub mutual_information: ContinuousHV,
    pub fidelity: ContinuousHV,
}

impl QIEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            ket_zero: genesis.hv("qi::ket_zero", PHYSICS_DIM),
            ket_one: genesis.hv("qi::ket_one", PHYSICS_DIM),
            superposition: genesis.hv("qi::superposition", PHYSICS_DIM),

            pauli_i: genesis.hv(PauliMatrix::I.domain(), PHYSICS_DIM),
            pauli_x: genesis.hv(PauliMatrix::X.domain(), PHYSICS_DIM),
            pauli_y: genesis.hv(PauliMatrix::Y.domain(), PHYSICS_DIM),
            pauli_z: genesis.hv(PauliMatrix::Z.domain(), PHYSICS_DIM),

            hadamard: genesis.hv(GateType::Hadamard.domain(), PHYSICS_DIM),
            cnot: genesis.hv(GateType::CNOT.domain(), PHYSICS_DIM),
            t_gate: genesis.hv(GateType::TGate.domain(), PHYSICS_DIM),

            entanglement: genesis.hv("qi::entanglement", PHYSICS_DIM),
            bell_state: genesis.hv(EntanglementType::Bell.domain(), PHYSICS_DIM),
            ghz_state: genesis.hv(EntanglementType::GHZ.domain(), PHYSICS_DIM),

            measurement: genesis.hv("qi::measurement", PHYSICS_DIM),
            collapse: genesis.hv("qi::collapse", PHYSICS_DIM),
            projection: genesis.hv("qi::projection", PHYSICS_DIM),

            error: genesis.hv("qec::error", PHYSICS_DIM),
            syndrome: genesis.hv("qec::syndrome", PHYSICS_DIM),
            correction: genesis.hv("qec::correction", PHYSICS_DIM),
            logical_qubit: genesis.hv("qec::logical_qubit", PHYSICS_DIM),

            decoherence: genesis.hv("qi::decoherence", PHYSICS_DIM),
            t1_decay: genesis.hv("qi::t1_decay", PHYSICS_DIM),
            t2_dephasing: genesis.hv("qi::t2_dephasing", PHYSICS_DIM),

            entropy: genesis.hv("qi::entropy", PHYSICS_DIM),
            mutual_information: genesis.hv("qi::mutual_information", PHYSICS_DIM),
            fidelity: genesis.hv("qi::fidelity", PHYSICS_DIM),
        }
    }

    /// Create |0⟩ state
    pub fn ket_0(&self) -> Qubit {
        Qubit {
            alpha: 1.0,
            beta: 0.0,
            vector: self.ket_zero.clone(),
        }
    }

    /// Create |1⟩ state
    pub fn ket_1(&self) -> Qubit {
        Qubit {
            alpha: 0.0,
            beta: 1.0,
            vector: self.ket_one.clone(),
        }
    }

    /// Create |+⟩ = (|0⟩ + |1⟩)/√2
    pub fn ket_plus(&self) -> Qubit {
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        let vector =
            ContinuousHV::bundle(&[&self.ket_zero, &self.ket_one]).bind(&self.superposition);
        Qubit {
            alpha: inv_sqrt2,
            beta: inv_sqrt2,
            vector,
        }
    }

    /// Create |-⟩ = (|0⟩ - |1⟩)/√2
    pub fn ket_minus(&self) -> Qubit {
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        let vector = ContinuousHV::bundle(&[&self.ket_zero, &self.ket_one.scale(-1.0)])
            .bind(&self.superposition);
        Qubit {
            alpha: inv_sqrt2,
            beta: -inv_sqrt2,
            vector,
        }
    }

    /// Apply Hadamard gate
    pub fn apply_hadamard(&self, qubit: &Qubit) -> Qubit {
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        let new_alpha = inv_sqrt2 * (qubit.alpha + qubit.beta);
        let new_beta = inv_sqrt2 * (qubit.alpha - qubit.beta);
        let vector = qubit.vector.bind(&self.hadamard);

        Qubit {
            alpha: new_alpha,
            beta: new_beta,
            vector,
        }
    }

    /// Apply Pauli-X (bit flip)
    pub fn apply_x(&self, qubit: &Qubit) -> Qubit {
        let vector = qubit.vector.bind(&self.pauli_x);
        Qubit {
            alpha: qubit.beta,
            beta: qubit.alpha,
            vector,
        }
    }

    /// Apply Pauli-Z (phase flip)
    pub fn apply_z(&self, qubit: &Qubit) -> Qubit {
        let vector = qubit.vector.bind(&self.pauli_z);
        Qubit {
            alpha: qubit.alpha,
            beta: -qubit.beta,
            vector,
        }
    }

    /// Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    pub fn bell_phi_plus(&self) -> EntangledState {
        let vector = ContinuousHV::bundle(&[
            &self.ket_zero.bind(&self.ket_zero),
            &self.ket_one.bind(&self.ket_one),
        ])
        .bind(&self.bell_state)
        .bind(&self.entanglement);

        EntangledState {
            entanglement_type: EntanglementType::Bell,
            num_qubits: 2,
            concurrence: 1.0, // Maximally entangled
            vector,
        }
    }

    /// Create Bell state |Ψ+⟩ = (|01⟩ + |10⟩)/√2
    pub fn bell_psi_plus(&self) -> EntangledState {
        let vector = ContinuousHV::bundle(&[
            &self.ket_zero.bind(&self.ket_one),
            &self.ket_one.bind(&self.ket_zero),
        ])
        .bind(&self.bell_state)
        .bind(&self.entanglement);

        EntangledState {
            entanglement_type: EntanglementType::Bell,
            num_qubits: 2,
            concurrence: 1.0,
            vector,
        }
    }

    /// Create GHZ state |GHZ_n⟩ = (|0...0⟩ + |1...1⟩)/√2
    pub fn ghz_state(&self, n: u32) -> EntangledState {
        let all_zeros = self.ket_zero.scale(n as f32);
        let all_ones = self.ket_one.scale(n as f32);
        let vector = ContinuousHV::bundle(&[&all_zeros, &all_ones])
            .bind(&self.ghz_state)
            .bind(&self.entanglement);

        EntangledState {
            entanglement_type: EntanglementType::GHZ,
            num_qubits: n,
            concurrence: 1.0,
            vector,
        }
    }

    /// Create a quantum gate
    pub fn create_gate(&self, gate_type: GateType, genesis: &GenesisSeed) -> QuantumGate {
        let vector = genesis.hv(gate_type.domain(), PHYSICS_DIM);
        QuantumGate { gate_type, vector }
    }

    /// Encode a logical qubit using error correction
    pub fn encode_logical(
        &self,
        qubit: &Qubit,
        code: QECCode,
        genesis: &GenesisSeed,
    ) -> ContinuousHV {
        let code_vec = genesis.hv(code.domain(), PHYSICS_DIM);
        qubit
            .vector
            .bind(&code_vec)
            .bind(&self.logical_qubit)
            .scale(code.physical_qubits() as f32)
    }

    /// Syndrome measurement for error detection
    pub fn measure_syndrome(&self, encoded: &ContinuousHV) -> ContinuousHV {
        encoded.bind(&self.syndrome).bind(&self.measurement)
    }

    /// No-cloning theorem: cannot perfectly copy quantum state
    /// This returns a "degraded" copy with reduced fidelity
    pub fn imperfect_copy(&self, qubit: &Qubit, fidelity: f64) -> Qubit {
        // Use fidelity-based seed for reproducibility
        let seed = (fidelity * 1e9) as u64;
        let noise = ContinuousHV::random(PHYSICS_DIM, seed).scale((1.0 - fidelity) as f32);
        let vector = ContinuousHV::weighted_bundle(
            &[&qubit.vector, &noise],
            &[fidelity as f32, (1.0 - fidelity) as f32],
        );

        Qubit {
            alpha: qubit.alpha * fidelity.sqrt(),
            beta: qubit.beta * fidelity.sqrt(),
            vector,
        }
    }

    /// Von Neumann entropy: S = -Tr(ρ log ρ)
    /// For pure state, S = 0; for maximally mixed, S = log(d)
    pub fn von_neumann_entropy(&self, purity: f64) -> f64 {
        if purity >= 1.0 {
            0.0
        } else if purity <= 0.0 {
            0.0
        } else {
            -purity * purity.log2() - (1.0 - purity) * (1.0 - purity).log2()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (QIEncoder, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("quantum information test");
        let encoder = QIEncoder::from_genesis(&genesis);
        (encoder, genesis)
    }

    #[test]
    fn test_qi_encoder_creation() {
        let (encoder, _) = setup();
        assert!(encoder.ket_zero.norm() > 0.0);
        assert!(encoder.entanglement.norm() > 0.0);
    }

    #[test]
    fn test_basis_states_orthogonal() {
        let (encoder, _) = setup();

        let zero = encoder.ket_0();
        let one = encoder.ket_1();

        // |0⟩ and |1⟩ should be different
        let sim = zero.vector.similarity(&one.vector);
        assert!(sim < 0.5, "|0⟩ and |1⟩ should be distinct: sim={}", sim);
    }

    #[test]
    fn test_hadamard_creates_superposition() {
        let (encoder, _) = setup();

        let zero = encoder.ket_0();
        let plus = encoder.apply_hadamard(&zero);

        // H|0⟩ = |+⟩ should have equal amplitudes
        assert!((plus.alpha - plus.beta).abs() < 0.01);

        // Should have superposition character
        let sup_sim = plus.vector.similarity(&encoder.superposition);
        assert!(sup_sim.abs() > 0.0);
    }

    #[test]
    fn test_pauli_x_flips() {
        let (encoder, _) = setup();

        let zero = encoder.ket_0();
        let flipped = encoder.apply_x(&zero);

        // X|0⟩ = |1⟩
        assert!((flipped.alpha - 0.0).abs() < 0.01);
        assert!((flipped.beta - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_bell_state_entangled() {
        let (encoder, _) = setup();

        let bell = encoder.bell_phi_plus();

        assert_eq!(bell.entanglement_type, EntanglementType::Bell);
        assert_eq!(bell.num_qubits, 2);
        assert!((bell.concurrence - 1.0).abs() < 0.01); // Maximally entangled

        // Should have entanglement character
        let ent_sim = bell.vector.similarity(&encoder.entanglement);
        assert!(ent_sim.abs() > 0.0);
    }

    #[test]
    fn test_ghz_state() {
        let (encoder, _) = setup();

        let ghz3 = encoder.ghz_state(3);
        let ghz5 = encoder.ghz_state(5);

        assert_eq!(ghz3.num_qubits, 3);
        assert_eq!(ghz5.num_qubits, 5);

        // Both should have maximal entanglement (concurrence = 1)
        assert!((ghz3.concurrence - 1.0).abs() < 0.01);
        assert!((ghz5.concurrence - 1.0).abs() < 0.01);

        // Larger GHZ state encodes more qubits via scaling
        // The norms differ proportionally to qubit count
        let norm3 = ghz3.vector.norm();
        let norm5 = ghz5.vector.norm();
        assert!(
            norm5 > norm3,
            "5-qubit GHZ should have larger norm than 3-qubit"
        );
    }

    #[test]
    fn test_error_correction_encoding() {
        let (encoder, genesis) = setup();

        let qubit = encoder.ket_0();
        let encoded = encoder.encode_logical(&qubit, QECCode::Shor9, &genesis);

        assert!(encoded.norm() > 0.0);

        // Should have logical qubit character
        let log_sim = encoded.similarity(&encoder.logical_qubit);
        assert!(log_sim.abs() > 0.0);
    }

    #[test]
    fn test_von_neumann_entropy() {
        let (encoder, _) = setup();

        // Pure state: S = 0
        let s_pure = encoder.von_neumann_entropy(1.0);
        assert!((s_pure - 0.0).abs() < 0.01);

        // Maximally mixed: S = 1 for qubit
        let s_mixed = encoder.von_neumann_entropy(0.5);
        assert!((s_mixed - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_no_cloning() {
        let (encoder, _) = setup();

        let original = encoder.ket_plus();
        let copy = encoder.imperfect_copy(&original, 0.9);

        // Copy should be similar but not identical
        let sim = original.vector.similarity(&copy.vector);
        assert!(sim > 0.5, "Copy should be similar to original");
        assert!(sim < 1.0, "Copy should not be perfect (no-cloning)");
    }
}
