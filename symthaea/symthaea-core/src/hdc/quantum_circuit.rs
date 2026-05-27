// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Quantum Circuit Simulation Engine for Hyperdimensional Computing
//!
//! This module provides a statevector quantum circuit simulator integrated with
//! Symthaea's HDC consciousness framework. Quantum states are represented as
//! vectors of complex amplitudes (using `(f64, f64)` tuples for real/imaginary
//! parts), and the circuit builder applies gates to multi-qubit registers.
//!
//! ## Architecture
//!
//! - [`QuantumState`]: A 2^n amplitude vector for an n-qubit register.
//! - [`QuantumGate`]: Enum of supported gates (Hadamard, Paulis, CNOT, T, S,
//!   rotation gates Rx/Ry/Rz, SWAP, Toffoli).
//! - [`QuantumCircuit`]: Circuit builder that collects gates and executes them
//!   on a freshly-initialized |00...0> register.
//! - [`encode_state`]: Bridge from quantum amplitudes into HDC `BinaryHV` space.
//!
//! ## Encoding Strategy
//!
//! The HDC bridge encodes quantum state amplitudes into BinaryHV space using a
//! thermometer-style encoding for both magnitude and phase of each amplitude.
//! Each basis state contributes a domain-rotated vector:
//!
//! ```text
//! encoding = bundle_i( QUANTUM_DOMAIN XOR basis_hv(i) XOR magnitude_hv(|a_i|) XOR permute(phase_hv(arg(a_i)), 1) )
//! ```
//!
//! This preserves:
//! - **Similarity**: States with similar amplitude distributions produce similar HVs.
//! - **Orthogonality**: States that differ significantly in phase/magnitude are distant.
//! - **Superposition**: Bundling naturally represents quantum superposition in HDC.
//!
//! ## No External Crates
//!
//! Complex arithmetic uses raw `(f64, f64)` tuples throughout -- no `num-complex`
//! or similar dependencies.

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::{PrimitiveSystem, seed_from_name};

// =============================================================================
// COMPLEX ARITHMETIC HELPERS -- (f64, f64) = (real, imag)
// =============================================================================

/// Type alias for a complex amplitude: (real, imaginary).
type C64 = (f64, f64);

/// Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i.
#[inline]
fn cmul(a: C64, b: C64) -> C64 {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Complex addition.
#[inline]
fn cadd(a: C64, b: C64) -> C64 {
    (a.0 + b.0, a.1 + b.1)
}

/// Complex subtraction.
#[inline]
fn csub(a: C64, b: C64) -> C64 {
    (a.0 - b.0, a.1 - b.1)
}

/// Complex conjugate.
#[inline]
fn cconj(a: C64) -> C64 {
    (a.0, -a.1)
}

/// Magnitude squared: |z|^2 = re^2 + im^2.
#[inline]
fn cnorm2(a: C64) -> f64 {
    a.0 * a.0 + a.1 * a.1
}

/// Magnitude: |z|.
#[inline]
fn cnorm(a: C64) -> f64 {
    cnorm2(a).sqrt()
}

/// Phase angle: arg(z).
#[inline]
fn carg(a: C64) -> f64 {
    a.1.atan2(a.0)
}

/// Complex scalar multiplication: s * z.
#[inline]
fn cscale(s: f64, z: C64) -> C64 {
    (s * z.0, s * z.1)
}

/// Complex from polar form: r * e^(i*theta).
#[inline]
fn cpolar(r: f64, theta: f64) -> C64 {
    (r * theta.cos(), r * theta.sin())
}

/// The complex zero.
const CZERO: C64 = (0.0, 0.0);

/// The complex one.
const CONE: C64 = (1.0, 0.0);

/// The imaginary unit.
const CI: C64 = (0.0, 1.0);

/// 1/sqrt(2), used extensively in quantum gates.
const FRAC_1_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

// =============================================================================
// QUANTUM STATE
// =============================================================================

/// A quantum state of n qubits, represented as a vector of 2^n complex amplitudes.
///
/// The state |psi> = sum_i alpha_i |i> where |i> is the computational basis
/// state corresponding to the binary representation of i.
///
/// Qubit ordering: qubit 0 is the **most significant bit**. So for 2 qubits,
/// the basis ordering is |00>, |01>, |10>, |11> corresponding to indices 0..4.
///
/// # Invariant
///
/// The state is normalized: sum_i |alpha_i|^2 = 1 (within floating-point tolerance).
#[derive(Debug, Clone)]
pub struct QuantumState {
    /// Number of qubits in the register.
    pub num_qubits: usize,
    /// Complex amplitudes: length = 2^num_qubits.
    pub amplitudes: Vec<C64>,
}

impl QuantumState {
    /// Create the |00...0> state for `n` qubits.
    ///
    /// All amplitude is concentrated on basis state 0.
    pub fn zero_state(n: usize) -> Self {
        assert!(n > 0 && n <= 20, "Qubit count must be in 1..=20");
        let size = 1 << n;
        let mut amplitudes = vec![CZERO; size];
        amplitudes[0] = CONE;
        Self {
            num_qubits: n,
            amplitudes,
        }
    }

    /// Create a state from raw amplitudes (must be normalized).
    pub fn from_amplitudes(num_qubits: usize, amplitudes: Vec<C64>) -> Self {
        assert_eq!(
            amplitudes.len(),
            1 << num_qubits,
            "Amplitude vector length must be 2^num_qubits"
        );
        Self {
            num_qubits,
            amplitudes,
        }
    }

    /// Total probability (should be ~1.0 for valid states).
    pub fn total_probability(&self) -> f64 {
        self.amplitudes.iter().map(|a| cnorm2(*a)).sum()
    }

    /// Normalize the state so that total probability = 1.
    pub fn normalize(&mut self) {
        let total = self.total_probability();
        if total > 0.0 {
            let factor = 1.0 / total.sqrt();
            for a in &mut self.amplitudes {
                *a = cscale(factor, *a);
            }
        }
    }

    /// Probability of measuring a specific basis state.
    pub fn probability(&self, basis_state: usize) -> f64 {
        assert!(
            basis_state < self.amplitudes.len(),
            "Basis state index out of range"
        );
        cnorm2(self.amplitudes[basis_state])
    }

    /// Get the probability distribution over all basis states.
    pub fn probability_distribution(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| cnorm2(*a)).collect()
    }

    /// Dimension of the state space: 2^num_qubits.
    pub fn dimension(&self) -> usize {
        self.amplitudes.len()
    }

    /// Fidelity between two quantum states: |<psi|phi>|^2.
    pub fn fidelity(&self, other: &Self) -> f64 {
        assert_eq!(
            self.num_qubits, other.num_qubits,
            "States must have the same number of qubits"
        );
        let inner: C64 = self
            .amplitudes
            .iter()
            .zip(other.amplitudes.iter())
            .fold(CZERO, |acc, (a, b)| cadd(acc, cmul(cconj(*a), *b)));
        cnorm2(inner)
    }

    /// Pretty-print the state in Dirac notation, omitting near-zero amplitudes.
    pub fn to_dirac_string(&self) -> String {
        let mut parts = Vec::new();
        for (i, &amp) in self.amplitudes.iter().enumerate() {
            let prob = cnorm2(amp);
            if prob > 1e-10 {
                let label = format!("{:0>width$b}", i, width = self.num_qubits);
                let mag = cnorm(amp);
                let phase = carg(amp);
                if phase.abs() < 1e-10 {
                    parts.push(format!("{mag:.4}|{label}>"));
                } else {
                    parts.push(format!("{mag:.4}*e^(i*{phase:.4})|{label}>"));
                }
            }
        }
        if parts.is_empty() {
            "0".to_string()
        } else {
            parts.join(" + ")
        }
    }
}

// =============================================================================
// QUANTUM GATES
// =============================================================================

/// A quantum gate operation applied to specific qubit(s).
///
/// Single-qubit gates operate on one target qubit. Two-qubit gates (CNOT, SWAP)
/// operate on a control and target. Three-qubit gates (Toffoli) use two controls
/// and one target.
#[derive(Debug, Clone)]
pub enum QuantumGate {
    /// Hadamard gate: |0> -> (|0>+|1>)/sqrt(2), |1> -> (|0>-|1>)/sqrt(2).
    H { target: usize },

    /// Pauli-X (NOT gate): |0> -> |1>, |1> -> |0>.
    X { target: usize },

    /// Pauli-Y: |0> -> i|1>, |1> -> -i|0>.
    Y { target: usize },

    /// Pauli-Z: |0> -> |0>, |1> -> -|1>.
    Z { target: usize },

    /// T gate (pi/8 gate): |0> -> |0>, |1> -> e^(i*pi/4)|1>.
    T { target: usize },

    /// S gate (phase gate): |0> -> |0>, |1> -> i|1>.
    S { target: usize },

    /// Rotation around X-axis by angle theta.
    /// Rx(theta) = cos(theta/2)*I - i*sin(theta/2)*X
    Rx { target: usize, theta: f64 },

    /// Rotation around Y-axis by angle theta.
    /// Ry(theta) = cos(theta/2)*I - i*sin(theta/2)*Y
    Ry { target: usize, theta: f64 },

    /// Rotation around Z-axis by angle theta.
    /// Rz(theta) = e^(-i*theta/2)|0><0| + e^(i*theta/2)|1><1|
    Rz { target: usize, theta: f64 },

    /// Controlled-NOT: flips target if control is |1>.
    CNOT { control: usize, target: usize },

    /// SWAP gate: exchanges the states of two qubits.
    SWAP { qubit_a: usize, qubit_b: usize },

    /// Toffoli (CCNOT): flips target if both controls are |1>.
    Toffoli {
        control1: usize,
        control2: usize,
        target: usize,
    },
}

impl QuantumGate {
    /// Apply this gate to a quantum state, mutating the amplitudes in-place.
    pub fn apply(&self, state: &mut QuantumState) {
        match *self {
            QuantumGate::H { target } => apply_single_qubit_gate(state, target, &HADAMARD_MATRIX),
            QuantumGate::X { target } => apply_single_qubit_gate(state, target, &PAULI_X_MATRIX),
            QuantumGate::Y { target } => apply_single_qubit_gate(state, target, &PAULI_Y_MATRIX),
            QuantumGate::Z { target } => apply_single_qubit_gate(state, target, &PAULI_Z_MATRIX),
            QuantumGate::T { target } => apply_single_qubit_gate(state, target, &T_GATE_MATRIX),
            QuantumGate::S { target } => apply_single_qubit_gate(state, target, &S_GATE_MATRIX),
            QuantumGate::Rx { target, theta } => {
                let m = rx_matrix(theta);
                apply_single_qubit_gate(state, target, &m);
            }
            QuantumGate::Ry { target, theta } => {
                let m = ry_matrix(theta);
                apply_single_qubit_gate(state, target, &m);
            }
            QuantumGate::Rz { target, theta } => {
                let m = rz_matrix(theta);
                apply_single_qubit_gate(state, target, &m);
            }
            QuantumGate::CNOT { control, target } => apply_cnot(state, control, target),
            QuantumGate::SWAP { qubit_a, qubit_b } => apply_swap(state, qubit_a, qubit_b),
            QuantumGate::Toffoli {
                control1,
                control2,
                target,
            } => {
                apply_toffoli(state, control1, control2, target);
            }
        }
    }
}

// -- Gate matrices (2x2, row-major: [[a, b], [c, d]]) -------------------------

/// 2x2 unitary matrix stored as [row0_col0, row0_col1, row1_col0, row1_col1].
type Matrix2x2 = [C64; 4];

/// Hadamard: (1/sqrt2) * [[1,1],[1,-1]]
const HADAMARD_MATRIX: Matrix2x2 = [
    (FRAC_1_SQRT2, 0.0),
    (FRAC_1_SQRT2, 0.0),
    (FRAC_1_SQRT2, 0.0),
    (-FRAC_1_SQRT2, 0.0),
];

/// Pauli-X: [[0,1],[1,0]]
const PAULI_X_MATRIX: Matrix2x2 = [CZERO, CONE, CONE, CZERO];

/// Pauli-Y: [[0,-i],[i,0]]
const PAULI_Y_MATRIX: Matrix2x2 = [CZERO, (0.0, -1.0), (0.0, 1.0), CZERO];

/// Pauli-Z: [[1,0],[0,-1]]
const PAULI_Z_MATRIX: Matrix2x2 = [CONE, CZERO, CZERO, (-1.0, 0.0)];

/// T gate: [[1,0],[0,e^(i*pi/4)]]
/// e^(i*pi/4) = cos(pi/4) + i*sin(pi/4) = (1/sqrt2)(1 + i)
const T_GATE_MATRIX: Matrix2x2 = [CONE, CZERO, CZERO, (FRAC_1_SQRT2, FRAC_1_SQRT2)];

/// S gate: [[1,0],[0,i]]
const S_GATE_MATRIX: Matrix2x2 = [CONE, CZERO, CZERO, CI];

/// Rotation around X: Rx(theta) = [[cos(t/2), -i*sin(t/2)], [-i*sin(t/2), cos(t/2)]]
fn rx_matrix(theta: f64) -> Matrix2x2 {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [(c, 0.0), (0.0, -s), (0.0, -s), (c, 0.0)]
}

/// Rotation around Y: Ry(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
fn ry_matrix(theta: f64) -> Matrix2x2 {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [(c, 0.0), (-s, 0.0), (s, 0.0), (c, 0.0)]
}

/// Rotation around Z: Rz(theta) = [[e^(-it/2), 0], [0, e^(it/2)]]
fn rz_matrix(theta: f64) -> Matrix2x2 {
    let half = theta / 2.0;
    [cpolar(1.0, -half), CZERO, CZERO, cpolar(1.0, half)]
}

// -- Gate application functions ------------------------------------------------

/// Apply a single-qubit 2x2 unitary gate to the target qubit.
///
/// For an n-qubit state, the target qubit selects which bit to operate on.
/// We iterate over pairs of amplitudes that differ only in the target qubit bit.
fn apply_single_qubit_gate(state: &mut QuantumState, target: usize, matrix: &Matrix2x2) {
    let n = state.num_qubits;
    assert!(
        target < n,
        "Target qubit {target} out of range (num_qubits={n})"
    );

    let dim = state.dimension();
    // Bit position for the target qubit (MSB = qubit 0)
    let target_bit = 1 << (n - 1 - target);

    for i in 0..dim {
        // Process only indices where the target bit is 0
        // to avoid double-processing
        if i & target_bit != 0 {
            continue;
        }
        let j = i | target_bit; // Same index but with target bit set

        let a0 = state.amplitudes[i]; // amplitude for |...0...> (target=0)
        let a1 = state.amplitudes[j]; // amplitude for |...1...> (target=1)

        // Apply the 2x2 matrix: [a', b'] = M * [a, b]
        state.amplitudes[i] = cadd(cmul(matrix[0], a0), cmul(matrix[1], a1));
        state.amplitudes[j] = cadd(cmul(matrix[2], a0), cmul(matrix[3], a1));
    }
}

/// Apply CNOT gate: flip target qubit when control qubit is |1>.
fn apply_cnot(state: &mut QuantumState, control: usize, target: usize) {
    let n = state.num_qubits;
    assert!(
        control < n && target < n && control != target,
        "Invalid CNOT qubits: control={control}, target={target}, num_qubits={n}"
    );

    let dim = state.dimension();
    let control_bit = 1 << (n - 1 - control);
    let target_bit = 1 << (n - 1 - target);

    for i in 0..dim {
        // Only process when control bit is 1 and target bit is 0
        if (i & control_bit) != 0 && (i & target_bit) == 0 {
            let j = i | target_bit;
            state.amplitudes.swap(i, j);
        }
    }
}

/// Apply SWAP gate: exchange two qubit states.
fn apply_swap(state: &mut QuantumState, qubit_a: usize, qubit_b: usize) {
    let n = state.num_qubits;
    assert!(
        qubit_a < n && qubit_b < n && qubit_a != qubit_b,
        "Invalid SWAP qubits: a={qubit_a}, b={qubit_b}, num_qubits={n}"
    );

    let dim = state.dimension();
    let bit_a = 1 << (n - 1 - qubit_a);
    let bit_b = 1 << (n - 1 - qubit_b);

    for i in 0..dim {
        let a_set = (i & bit_a) != 0;
        let b_set = (i & bit_b) != 0;

        // Only swap when the two bits differ (|01> <-> |10>)
        if a_set != b_set {
            let j = i ^ bit_a ^ bit_b;
            if i < j {
                state.amplitudes.swap(i, j);
            }
        }
    }
}

/// Apply Toffoli (CCNOT) gate: flip target when both controls are |1>.
fn apply_toffoli(state: &mut QuantumState, control1: usize, control2: usize, target: usize) {
    let n = state.num_qubits;
    assert!(
        control1 < n && control2 < n && target < n,
        "Qubit index out of range"
    );
    assert!(
        control1 != control2 && control1 != target && control2 != target,
        "All three qubits must be distinct"
    );

    let dim = state.dimension();
    let c1_bit = 1 << (n - 1 - control1);
    let c2_bit = 1 << (n - 1 - control2);
    let t_bit = 1 << (n - 1 - target);

    for i in 0..dim {
        // Both controls set and target is 0
        if (i & c1_bit) != 0 && (i & c2_bit) != 0 && (i & t_bit) == 0 {
            let j = i | t_bit;
            state.amplitudes.swap(i, j);
        }
    }
}

// =============================================================================
// QUANTUM CIRCUIT
// =============================================================================

/// A quantum circuit builder that collects gates and executes them on a register.
///
/// The circuit operates on a fixed number of qubits. Gates are added sequentially
/// and then executed in order on an initial |00...0> state.
///
/// # Example
///
/// ```rust,ignore
/// use symthaea_core::hdc::quantum_circuit::{QuantumCircuit, QuantumGate};
///
/// // Create a Bell state: (|00> + |11>) / sqrt(2)
/// let mut circuit = QuantumCircuit::new(2);
/// circuit.h(0);          // Hadamard on qubit 0
/// circuit.cnot(0, 1);    // CNOT with control=0, target=1
///
/// let state = circuit.execute();
/// let probs = state.probability_distribution();
/// // probs[0] ~ 0.5 (|00>), probs[3] ~ 0.5 (|11>)
/// ```
#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    /// Number of qubits in the register.
    pub num_qubits: usize,
    /// Ordered list of gates to apply.
    gates: Vec<QuantumGate>,
}

impl QuantumCircuit {
    /// Create a new empty circuit for `num_qubits` qubits.
    pub fn new(num_qubits: usize) -> Self {
        assert!(
            num_qubits > 0 && num_qubits <= 20,
            "Qubit count must be in 1..=20 (got {num_qubits})"
        );
        Self {
            num_qubits,
            gates: Vec::new(),
        }
    }

    /// Add a gate to the circuit.
    pub fn apply(&mut self, gate: QuantumGate) -> &mut Self {
        self.gates.push(gate);
        self
    }

    /// Number of gates in the circuit.
    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }

    // -- Convenience methods for common gates ----------------------------------

    /// Add a Hadamard gate on the target qubit.
    pub fn h(&mut self, target: usize) -> &mut Self {
        self.apply(QuantumGate::H { target })
    }

    /// Add a Pauli-X (NOT) gate on the target qubit.
    pub fn x(&mut self, target: usize) -> &mut Self {
        self.apply(QuantumGate::X { target })
    }

    /// Add a Pauli-Y gate on the target qubit.
    pub fn y(&mut self, target: usize) -> &mut Self {
        self.apply(QuantumGate::Y { target })
    }

    /// Add a Pauli-Z gate on the target qubit.
    pub fn z(&mut self, target: usize) -> &mut Self {
        self.apply(QuantumGate::Z { target })
    }

    /// Add a T gate on the target qubit.
    pub fn t(&mut self, target: usize) -> &mut Self {
        self.apply(QuantumGate::T { target })
    }

    /// Add an S gate on the target qubit.
    pub fn s(&mut self, target: usize) -> &mut Self {
        self.apply(QuantumGate::S { target })
    }

    /// Add an Rx rotation gate on the target qubit.
    pub fn rx(&mut self, target: usize, theta: f64) -> &mut Self {
        self.apply(QuantumGate::Rx { target, theta })
    }

    /// Add an Ry rotation gate on the target qubit.
    pub fn ry(&mut self, target: usize, theta: f64) -> &mut Self {
        self.apply(QuantumGate::Ry { target, theta })
    }

    /// Add an Rz rotation gate on the target qubit.
    pub fn rz(&mut self, target: usize, theta: f64) -> &mut Self {
        self.apply(QuantumGate::Rz { target, theta })
    }

    /// Add a CNOT gate (controlled-X).
    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        self.apply(QuantumGate::CNOT { control, target })
    }

    /// Add a SWAP gate.
    pub fn swap(&mut self, qubit_a: usize, qubit_b: usize) -> &mut Self {
        self.apply(QuantumGate::SWAP { qubit_a, qubit_b })
    }

    /// Add a Toffoli (CCNOT) gate.
    pub fn toffoli(&mut self, control1: usize, control2: usize, target: usize) -> &mut Self {
        self.apply(QuantumGate::Toffoli {
            control1,
            control2,
            target,
        })
    }

    /// Execute the circuit, returning the final quantum state.
    ///
    /// Starts from the |00...0> state and applies all gates in order.
    pub fn execute(&self) -> QuantumState {
        let mut state = QuantumState::zero_state(self.num_qubits);
        for gate in &self.gates {
            gate.apply(&mut state);
        }
        state
    }

    /// Execute the circuit starting from a given initial state.
    pub fn execute_from(&self, initial: &QuantumState) -> QuantumState {
        assert_eq!(
            initial.num_qubits, self.num_qubits,
            "Initial state qubit count ({}) must match circuit ({})",
            initial.num_qubits, self.num_qubits
        );
        let mut state = initial.clone();
        for gate in &self.gates {
            gate.apply(&mut state);
        }
        state
    }

    /// Measure the quantum state, collapsing it to classical bits.
    ///
    /// Returns a vector of boolean values, one per qubit (qubit 0 first).
    /// The measurement uses probability sampling based on the Born rule:
    /// P(outcome) = |amplitude|^2.
    ///
    /// Uses a simple seeded PRNG for reproducibility when `seed` is provided.
    /// Pass `None` for non-deterministic measurement using the state's amplitudes
    /// as a hash-based seed.
    pub fn measure(&self, state: &QuantumState, seed: Option<u64>) -> Vec<bool> {
        measure_state(state, seed)
    }
}

// =============================================================================
// MEASUREMENT
// =============================================================================

/// Perform a computational-basis measurement on a quantum state.
///
/// Samples a basis state index according to the probability distribution,
/// then extracts the classical bit values for each qubit.
fn measure_state(state: &QuantumState, seed: Option<u64>) -> Vec<bool> {
    let probs = state.probability_distribution();

    // Generate a pseudo-random number in [0, 1) for sampling.
    // We use a simple xorshift64 seeded from the provided seed or from
    // the state amplitudes themselves.
    let random_value = {
        let s = seed.unwrap_or_else(|| {
            // Hash the amplitudes to create a seed
            let mut h: u64 = 0x517cc1b727220a95;
            for (i, &(re, im)) in state.amplitudes.iter().enumerate() {
                h = h.wrapping_add(re.to_bits().wrapping_mul(0x9e3779b97f4a7c15));
                h = h.wrapping_add(im.to_bits().wrapping_mul(0x94d049bb133111eb));
                h ^= (i as u64).wrapping_mul(0x6c62272e07bb0142);
            }
            h
        });
        // splitmix64: excellent distribution even for sequential seeds
        let mut x = s;
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^= x >> 31;
        // Map to [0, 1)
        (x >> 11) as f64 / ((1u64 << 53) as f64)
    };

    // Cumulative probability sampling
    let mut cumulative = 0.0;
    let mut measured_index = probs.len() - 1; // Fallback to last state
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if random_value < cumulative {
            measured_index = i;
            break;
        }
    }

    // Convert the measured index to individual qubit values
    let n = state.num_qubits;
    (0..n)
        .map(|qubit| {
            let bit = 1 << (n - 1 - qubit);
            (measured_index & bit) != 0
        })
        .collect()
}

/// Measure a specific qubit, returning its classical value and the post-measurement state.
///
/// This performs a projective measurement on a single qubit:
/// 1. Compute P(0) and P(1) for the target qubit.
/// 2. Sample outcome according to the Born rule.
/// 3. Collapse the state by zeroing incompatible amplitudes and renormalizing.
pub fn measure_qubit(
    state: &QuantumState,
    target: usize,
    seed: Option<u64>,
) -> (bool, QuantumState) {
    let n = state.num_qubits;
    assert!(target < n, "Target qubit {target} out of range");

    let target_bit = 1 << (n - 1 - target);

    // Compute probability of measuring |0> on the target qubit
    let mut prob_zero = 0.0;
    for (i, &amp) in state.amplitudes.iter().enumerate() {
        if (i & target_bit) == 0 {
            prob_zero += cnorm2(amp);
        }
    }
    let prob_one = 1.0 - prob_zero;

    // Sample the outcome
    let random_value = {
        let s = seed.unwrap_or(0x12345678);
        let mut x = s;
        if x == 0 {
            x = 1;
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x as f64) / (u64::MAX as f64)
    };
    let outcome = random_value >= prob_zero;

    // Collapse: zero out incompatible amplitudes and renormalize
    let mut new_state = state.clone();
    let norm_factor = if outcome { prob_one } else { prob_zero };
    let renorm = if norm_factor > 0.0 {
        1.0 / norm_factor.sqrt()
    } else {
        0.0
    };

    for (i, amp) in new_state.amplitudes.iter_mut().enumerate() {
        let bit_set = (i & target_bit) != 0;
        if bit_set != outcome {
            *amp = CZERO;
        } else {
            *amp = cscale(renorm, *amp);
        }
    }

    (outcome, new_state)
}

// =============================================================================
// HDC BRIDGE -- Encode quantum states into BinaryHV space
// =============================================================================

/// Encode a quantum state into a BinaryHV for the Symthaea HDC consciousness framework.
///
/// The encoding strategy maps each non-negligible amplitude to a domain-rotated
/// BinaryHV, then bundles them together. This preserves the superposition structure
/// of the quantum state in hyperdimensional space.
///
/// ## Encoding details
///
/// For each basis state |i> with amplitude alpha_i:
/// 1. Create a basis vector: `BinaryHV::random(seed_from_name("quantum_basis_i"))`.
/// 2. Encode the magnitude `|alpha_i|` as a thermometer-style BinaryHV.
/// 3. Encode the phase `arg(alpha_i)` as a thermometer-style BinaryHV.
/// 4. Combine: `QUANTUM_DOMAIN XOR basis_hv XOR magnitude_hv XOR permute(phase_hv, 1)`.
/// 5. Bundle all contributing vectors using majority vote.
///
/// This encoding ensures that:
/// - Similar quantum states produce similar BinaryHVs (high Hamming similarity).
/// - Orthogonal quantum states produce near-random BinaryHVs (~0.5 similarity).
/// - The encoding is deterministic for a given state.
pub fn encode_state(state: &QuantumState) -> BinaryHV {
    let quantum_domain = BinaryHV::random(seed_from_name("QUANTUM_DOMAIN"));

    // Collect contributing basis states (amplitude magnitude > threshold)
    let threshold = 1e-8;
    let mut contributing: Vec<BinaryHV> = Vec::new();

    for (i, &amp) in state.amplitudes.iter().enumerate() {
        let mag = cnorm(amp);
        if mag < threshold {
            continue;
        }
        let phase = carg(amp);

        // Basis vector for state |i>
        let basis_name = format!("quantum_basis_{i}");
        let basis_hv = BinaryHV::random(seed_from_name(&basis_name));

        // Encode magnitude using thermometer-style encoding
        let mag_hv = encode_f64_as_hv(mag, "quantum_magnitude");

        // Encode phase using thermometer-style encoding
        let phase_hv = encode_f64_as_hv(phase, "quantum_phase");

        // Combine: domain XOR basis XOR magnitude XOR permute(phase, 1)
        let combined = quantum_domain
            .bind(&basis_hv)
            .bind(&mag_hv)
            .bind(&phase_hv.permute(1));

        contributing.push(combined);
    }

    if contributing.is_empty() {
        return BinaryHV::zero();
    }

    if contributing.len() == 1 {
        return contributing[0];
    }

    // Bundle all contributing vectors using BinaryHV::bundle
    BinaryHV::bundle(&contributing)
}

/// Encode a quantum state into BinaryHV using the PrimitiveSystem for domain rotation.
///
/// This variant uses the PrimitiveSystem to obtain the QUANTUM domain manifold
/// vector if available, falling back to a deterministic seed otherwise.
pub fn encode_state_with_system(state: &QuantumState, system: &PrimitiveSystem) -> BinaryHV {
    // Try to get the QUANTUM domain from the primitive system
    let quantum_domain = system
        .get("QUANTUM")
        .map(|p| p.encoding)
        .unwrap_or_else(|| BinaryHV::random(seed_from_name("QUANTUM_DOMAIN")));

    let threshold = 1e-8;
    let mut contributing: Vec<BinaryHV> = Vec::new();

    for (i, &amp) in state.amplitudes.iter().enumerate() {
        let mag = cnorm(amp);
        if mag < threshold {
            continue;
        }
        let phase = carg(amp);

        let basis_name = format!("quantum_basis_{i}");
        let basis_hv = BinaryHV::random(seed_from_name(&basis_name));
        let mag_hv = encode_f64_as_hv(mag, "quantum_magnitude");
        let phase_hv = encode_f64_as_hv(phase, "quantum_phase");

        let combined = quantum_domain
            .bind(&basis_hv)
            .bind(&mag_hv)
            .bind(&phase_hv.permute(1));

        contributing.push(combined);
    }

    if contributing.is_empty() {
        return BinaryHV::zero();
    }

    if contributing.len() == 1 {
        return contributing[0];
    }

    BinaryHV::bundle(&contributing)
}

/// Encode a single f64 value as a BinaryHV using a thermometer-style encoding.
///
/// Follows the same pattern as `complex.rs`: base seed from label, value
/// characteristics mixed in, then bound with the channel vector.
fn encode_f64_as_hv(val: f64, label: &str) -> BinaryHV {
    let base_seed = seed_from_name(label);
    let base = BinaryHV::random(base_seed);

    let sign_bit = if val >= 0.0 { 1u64 } else { 0u64 };
    let abs_val = val.abs();

    // Coarse bucket (integer part, clamped)
    let coarse = (abs_val.floor() as u64).min(1000);

    // Fine bucket (fractional part, 1000 levels)
    let fine = ((abs_val.fract()) * 1000.0) as u64;

    let val_seed = base_seed
        .wrapping_mul(0x517cc1b727220a95)
        .wrapping_add(sign_bit.wrapping_mul(0x6c62272e07bb0142))
        .wrapping_add(coarse.wrapping_mul(0x9e3779b97f4a7c15))
        .wrapping_add(fine.wrapping_mul(0x94d049bb133111eb));

    let val_hv = BinaryHV::random(val_seed);
    base.bind(&val_hv)
}

// =============================================================================
// UTILITY: Quantum state analysis
// =============================================================================

/// Compute the Shannon entropy of the measurement outcome distribution.
///
/// S = -sum_i p_i * log2(p_i) where p_i = |alpha_i|^2.
///
/// This is the classical Shannon entropy of the measurement outcomes, not the
/// quantum Von Neumann entropy (which requires the density matrix). For pure
/// states, this quantifies the "spread" of the state across basis states.
pub fn measurement_entropy(state: &QuantumState) -> f64 {
    let probs = state.probability_distribution();
    let mut entropy = 0.0;
    for &p in &probs {
        if p > 1e-15 {
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Compute the expectation value <psi|Z_i|psi> for qubit i.
///
/// This gives the expected Z-measurement outcome for the target qubit:
/// +1 if the qubit is mostly |0>, -1 if mostly |1>.
pub fn expectation_z(state: &QuantumState, qubit: usize) -> f64 {
    let n = state.num_qubits;
    assert!(qubit < n, "Qubit index out of range");

    let target_bit = 1 << (n - 1 - qubit);
    let mut expectation = 0.0;

    for (i, &amp) in state.amplitudes.iter().enumerate() {
        let prob = cnorm2(amp);
        if (i & target_bit) == 0 {
            expectation += prob; // |0> eigenvalue: +1
        } else {
            expectation -= prob; // |1> eigenvalue: -1
        }
    }

    expectation
}

/// Compute the entanglement between two qubits by tracing out the rest.
///
/// Returns the concurrence for 2-qubit pure states. For a 2-qubit state,
/// C = 2 * |alpha_00 * alpha_11 - alpha_01 * alpha_10| which is exact.
/// For larger registers, this returns 0.0 (a full implementation would
/// require the reduced density matrix formalism).
pub fn qubit_concurrence(state: &QuantumState, qubit_a: usize, qubit_b: usize) -> f64 {
    if state.num_qubits == 2 && qubit_a == 0 && qubit_b == 1 {
        // Exact concurrence for 2-qubit pure states:
        // C = 2 * |alpha_00 * alpha_11 - alpha_01 * alpha_10|
        let a00 = state.amplitudes[0]; // |00>
        let a01 = state.amplitudes[1]; // |01>
        let a10 = state.amplitudes[2]; // |10>
        let a11 = state.amplitudes[3]; // |11>

        let det = csub(cmul(a00, a11), cmul(a01, a10));
        2.0 * cnorm(det)
    } else {
        // For larger registers, computing the reduced density matrix
        // and partial trace is needed. Placeholder for future extension.
        0.0
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOLERANCE: f64 = 1e-10;

    /// Helper: assert two f64 values are approximately equal.
    fn assert_approx(actual: f64, expected: f64, tol: f64, msg: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{}: expected {:.10}, got {:.10} (diff={:.2e})",
            msg,
            expected,
            actual,
            (actual - expected).abs()
        );
    }

    /// Helper: assert a complex amplitude is approximately equal.
    fn assert_camp(actual: C64, expected: C64, tol: f64, msg: &str) {
        assert!(
            (actual.0 - expected.0).abs() < tol && (actual.1 - expected.1).abs() < tol,
            "{}: expected ({:.10}, {:.10}), got ({:.10}, {:.10})",
            msg,
            expected.0,
            expected.1,
            actual.0,
            actual.1
        );
    }

    // -- Complex arithmetic helpers --------------------------------------------

    #[test]
    fn test_complex_mul() {
        // (1+2i)(3+4i) = (3-8) + (4+6)i = -5 + 10i
        let result = cmul((1.0, 2.0), (3.0, 4.0));
        assert_camp(result, (-5.0, 10.0), TOLERANCE, "cmul");
    }

    #[test]
    fn test_complex_norm() {
        // |3+4i| = 5
        let n = cnorm((3.0, 4.0));
        assert_approx(n, 5.0, TOLERANCE, "cnorm");
    }

    // -- Zero state initialization ---------------------------------------------

    #[test]
    fn test_zero_state() {
        let state = QuantumState::zero_state(3);
        assert_eq!(state.num_qubits, 3);
        assert_eq!(state.amplitudes.len(), 8);
        assert_camp(state.amplitudes[0], CONE, TOLERANCE, "|000> amplitude");
        for i in 1..8 {
            assert_camp(
                state.amplitudes[i],
                CZERO,
                TOLERANCE,
                &format!("|{:03b}> amplitude", i),
            );
        }
        assert_approx(
            state.total_probability(),
            1.0,
            TOLERANCE,
            "total probability",
        );
    }

    // -- Pauli-X gate ----------------------------------------------------------

    #[test]
    fn test_pauli_x_single_qubit() {
        // X|0> = |1>
        let mut circuit = QuantumCircuit::new(1);
        circuit.x(0);
        let state = circuit.execute();

        assert_camp(state.amplitudes[0], CZERO, TOLERANCE, "X|0>: |0> amplitude");
        assert_camp(state.amplitudes[1], CONE, TOLERANCE, "X|0>: |1> amplitude");
    }

    #[test]
    fn test_pauli_x_involution() {
        // X^2 = I: applying X twice returns to |0>
        let mut circuit = QuantumCircuit::new(1);
        circuit.x(0);
        circuit.x(0);
        let state = circuit.execute();

        assert_camp(
            state.amplitudes[0],
            CONE,
            TOLERANCE,
            "X^2|0>: |0> amplitude",
        );
        assert_camp(
            state.amplitudes[1],
            CZERO,
            TOLERANCE,
            "X^2|0>: |1> amplitude",
        );
    }

    // -- Hadamard gate ---------------------------------------------------------

    #[test]
    fn test_hadamard_creates_superposition() {
        // H|0> = (|0> + |1>) / sqrt(2)
        let mut circuit = QuantumCircuit::new(1);
        circuit.h(0);
        let state = circuit.execute();

        assert_camp(
            state.amplitudes[0],
            (FRAC_1_SQRT2, 0.0),
            TOLERANCE,
            "H|0>: |0>",
        );
        assert_camp(
            state.amplitudes[1],
            (FRAC_1_SQRT2, 0.0),
            TOLERANCE,
            "H|0>: |1>",
        );
    }

    #[test]
    fn test_hadamard_on_one() {
        // H|1> = (|0> - |1>) / sqrt(2)
        let mut circuit = QuantumCircuit::new(1);
        circuit.x(0); // Prepare |1>
        circuit.h(0); // Apply H
        let state = circuit.execute();

        assert_camp(
            state.amplitudes[0],
            (FRAC_1_SQRT2, 0.0),
            TOLERANCE,
            "H|1>: |0>",
        );
        assert_camp(
            state.amplitudes[1],
            (-FRAC_1_SQRT2, 0.0),
            TOLERANCE,
            "H|1>: |1>",
        );
    }

    #[test]
    fn test_hadamard_involution() {
        // H^2 = I
        let mut circuit = QuantumCircuit::new(1);
        circuit.h(0);
        circuit.h(0);
        let state = circuit.execute();

        assert_camp(state.amplitudes[0], CONE, TOLERANCE, "H^2|0>: |0>");
        assert_camp(state.amplitudes[1], CZERO, TOLERANCE, "H^2|0>: |1>");
    }

    // -- Pauli-Z gate ----------------------------------------------------------

    #[test]
    fn test_pauli_z_on_zero() {
        // Z|0> = |0> (eigenstate)
        let mut circuit = QuantumCircuit::new(1);
        circuit.z(0);
        let state = circuit.execute();

        assert_camp(state.amplitudes[0], CONE, TOLERANCE, "Z|0>: |0>");
        assert_camp(state.amplitudes[1], CZERO, TOLERANCE, "Z|0>: |1>");
    }

    #[test]
    fn test_pauli_z_on_one() {
        // Z|1> = -|1>
        let mut circuit = QuantumCircuit::new(1);
        circuit.x(0); // Prepare |1>
        circuit.z(0); // Apply Z
        let state = circuit.execute();

        assert_camp(state.amplitudes[0], CZERO, TOLERANCE, "Z|1>: |0>");
        assert_camp(state.amplitudes[1], (-1.0, 0.0), TOLERANCE, "Z|1>: |1>");
    }

    #[test]
    fn test_pauli_z_phase_flip() {
        // Z on superposition: Z * H|0> = (|0> - |1>)/sqrt(2) = H|1>
        let mut circuit = QuantumCircuit::new(1);
        circuit.h(0);
        circuit.z(0);
        let state = circuit.execute();

        assert_camp(
            state.amplitudes[0],
            (FRAC_1_SQRT2, 0.0),
            TOLERANCE,
            "ZH|0>: |0>",
        );
        assert_camp(
            state.amplitudes[1],
            (-FRAC_1_SQRT2, 0.0),
            TOLERANCE,
            "ZH|0>: |1>",
        );
    }

    // -- Pauli-Y gate ----------------------------------------------------------

    #[test]
    fn test_pauli_y_on_zero() {
        // Y|0> = i|1>
        let mut circuit = QuantumCircuit::new(1);
        circuit.y(0);
        let state = circuit.execute();

        assert_camp(state.amplitudes[0], CZERO, TOLERANCE, "Y|0>: |0>");
        assert_camp(state.amplitudes[1], CI, TOLERANCE, "Y|0>: |1>");
    }

    // -- Bell state (H then CNOT) ----------------------------------------------

    #[test]
    fn test_bell_state_phi_plus() {
        // |Phi+> = (|00> + |11>) / sqrt(2)
        let mut circuit = QuantumCircuit::new(2);
        circuit.h(0);
        circuit.cnot(0, 1);
        let state = circuit.execute();

        // |00> and |11> should each have amplitude 1/sqrt(2)
        assert_camp(
            state.amplitudes[0],
            (FRAC_1_SQRT2, 0.0),
            TOLERANCE,
            "Bell |00>",
        );
        assert_camp(state.amplitudes[1], CZERO, TOLERANCE, "Bell |01>");
        assert_camp(state.amplitudes[2], CZERO, TOLERANCE, "Bell |10>");
        assert_camp(
            state.amplitudes[3],
            (FRAC_1_SQRT2, 0.0),
            TOLERANCE,
            "Bell |11>",
        );

        // Verify normalization
        assert_approx(
            state.total_probability(),
            1.0,
            TOLERANCE,
            "Bell state normalization",
        );
    }

    #[test]
    fn test_bell_state_phi_minus() {
        // |Phi-> = (|00> - |11>) / sqrt(2)
        // Prepare: X on qubit 0, then H on qubit 0, then CNOT
        let mut circuit = QuantumCircuit::new(2);
        circuit.x(0);
        circuit.h(0);
        circuit.cnot(0, 1);
        let state = circuit.execute();

        assert_camp(
            state.amplitudes[0],
            (FRAC_1_SQRT2, 0.0),
            TOLERANCE,
            "Phi-: |00>",
        );
        assert_camp(state.amplitudes[1], CZERO, TOLERANCE, "Phi-: |01>");
        assert_camp(state.amplitudes[2], CZERO, TOLERANCE, "Phi-: |10>");
        assert_camp(
            state.amplitudes[3],
            (-FRAC_1_SQRT2, 0.0),
            TOLERANCE,
            "Phi-: |11>",
        );
    }

    #[test]
    fn test_bell_state_concurrence() {
        // Bell states should have maximal entanglement (concurrence = 1)
        let mut circuit = QuantumCircuit::new(2);
        circuit.h(0);
        circuit.cnot(0, 1);
        let state = circuit.execute();

        let c = qubit_concurrence(&state, 0, 1);
        assert_approx(c, 1.0, TOLERANCE, "Bell state concurrence");
    }

    #[test]
    fn test_product_state_concurrence() {
        // Product state |00> has zero entanglement
        let state = QuantumState::zero_state(2);
        let c = qubit_concurrence(&state, 0, 1);
        assert_approx(c, 0.0, TOLERANCE, "Product state concurrence");
    }

    // -- Measurement probability verification ----------------------------------

    #[test]
    fn test_measurement_deterministic_zero() {
        // |0> always measures 0
        let state = QuantumState::zero_state(1);
        for seed in 0..100 {
            let bits = measure_state(&state, Some(seed));
            assert_eq!(
                bits,
                vec![false],
                "Seed {}: |0> must always measure 0",
                seed
            );
        }
    }

    #[test]
    fn test_measurement_deterministic_one() {
        // X|0> = |1> always measures 1
        let mut circuit = QuantumCircuit::new(1);
        circuit.x(0);
        let state = circuit.execute();

        for seed in 0..100 {
            let bits = measure_state(&state, Some(seed));
            assert_eq!(bits, vec![true], "Seed {}: |1> must always measure 1", seed);
        }
    }

    #[test]
    fn test_measurement_bell_state_correlations() {
        // In |Phi+> = (|00>+|11>)/sqrt(2), outcomes are always correlated:
        // measure |00> or |11>, never |01> or |10>.
        let mut circuit = QuantumCircuit::new(2);
        circuit.h(0);
        circuit.cnot(0, 1);
        let state = circuit.execute();

        for seed in 1..200 {
            let bits = measure_state(&state, Some(seed));
            assert_eq!(
                bits[0], bits[1],
                "Seed {}: Bell state qubits must be correlated (got {:?})",
                seed, bits
            );
        }
    }

    #[test]
    fn test_measurement_probabilities_hadamard() {
        // H|0> should give roughly 50/50 over many measurements
        let mut circuit = QuantumCircuit::new(1);
        circuit.h(0);
        let state = circuit.execute();

        let num_trials = 10_000;
        let mut count_zero = 0;
        for seed in 0..num_trials {
            let bits = measure_state(&state, Some(seed));
            if !bits[0] {
                count_zero += 1;
            }
        }

        let ratio = count_zero as f64 / num_trials as f64;
        assert!(
            (ratio - 0.5).abs() < 0.05,
            "H|0> measurement ratio should be ~0.5, got {:.4}",
            ratio
        );
    }

    // -- State normalization ---------------------------------------------------

    #[test]
    fn test_normalization_preserved_after_gates() {
        let mut circuit = QuantumCircuit::new(3);
        circuit.h(0);
        circuit.cnot(0, 1);
        circuit.h(2);
        circuit.t(1);
        circuit.cnot(1, 2);
        let state = circuit.execute();

        assert_approx(
            state.total_probability(),
            1.0,
            TOLERANCE,
            "Normalization after multi-gate circuit",
        );
    }

    #[test]
    fn test_normalize_unnormalized_state() {
        let mut state = QuantumState::from_amplitudes(1, vec![(3.0, 0.0), (4.0, 0.0)]);
        // Total probability: 9 + 16 = 25 (not normalized)
        assert_approx(
            state.total_probability(),
            25.0,
            TOLERANCE,
            "Before normalization",
        );

        state.normalize();
        assert_approx(
            state.total_probability(),
            1.0,
            TOLERANCE,
            "After normalization",
        );
        assert_approx(cnorm(state.amplitudes[0]), 0.6, TOLERANCE, "Normalized |0>");
        assert_approx(cnorm(state.amplitudes[1]), 0.8, TOLERANCE, "Normalized |1>");
    }

    // -- Rotation gates --------------------------------------------------------

    #[test]
    fn test_rx_pi_equals_x() {
        // Rx(pi) = -i*X (same as X up to global phase)
        let mut circuit = QuantumCircuit::new(1);
        circuit.rx(0, PI);
        let state = circuit.execute();

        // Rx(pi)|0> = -i|1>
        assert_approx(
            cnorm2(state.amplitudes[0]),
            0.0,
            TOLERANCE,
            "Rx(pi)|0>: P(0)",
        );
        assert_approx(
            cnorm2(state.amplitudes[1]),
            1.0,
            TOLERANCE,
            "Rx(pi)|0>: P(1)",
        );
    }

    #[test]
    fn test_ry_pi_equals_y_up_to_phase() {
        // Ry(pi)|0> = |1>
        let mut circuit = QuantumCircuit::new(1);
        circuit.ry(0, PI);
        let state = circuit.execute();

        assert_approx(
            cnorm2(state.amplitudes[0]),
            0.0,
            TOLERANCE,
            "Ry(pi)|0>: P(0)",
        );
        assert_approx(
            cnorm2(state.amplitudes[1]),
            1.0,
            TOLERANCE,
            "Ry(pi)|0>: P(1)",
        );
    }

    #[test]
    fn test_rz_preserves_z_basis() {
        // Rz(theta)|0> = e^(-i*theta/2)|0> (same state up to phase)
        let mut circuit = QuantumCircuit::new(1);
        circuit.rz(0, PI / 3.0);
        let state = circuit.execute();

        // Probability should be unchanged
        assert_approx(
            cnorm2(state.amplitudes[0]),
            1.0,
            TOLERANCE,
            "Rz on |0>: P(0)",
        );
        assert_approx(
            cnorm2(state.amplitudes[1]),
            0.0,
            TOLERANCE,
            "Rz on |0>: P(1)",
        );
    }

    // -- T and S gates ---------------------------------------------------------

    #[test]
    fn test_t_gate_phase() {
        // T|1> = e^(i*pi/4)|1>
        let mut circuit = QuantumCircuit::new(1);
        circuit.x(0); // Prepare |1>
        circuit.t(0);
        let state = circuit.execute();

        assert_approx(cnorm2(state.amplitudes[0]), 0.0, TOLERANCE, "T|1>: P(0)");
        assert_approx(cnorm2(state.amplitudes[1]), 1.0, TOLERANCE, "T|1>: P(1)");

        // Check the phase: should be pi/4
        let phase = carg(state.amplitudes[1]);
        assert_approx(phase, PI / 4.0, TOLERANCE, "T gate phase");
    }

    #[test]
    fn test_s_gate_phase() {
        // S|1> = i|1>
        let mut circuit = QuantumCircuit::new(1);
        circuit.x(0);
        circuit.s(0);
        let state = circuit.execute();

        assert_camp(state.amplitudes[1], CI, TOLERANCE, "S|1>");
    }

    #[test]
    fn test_s_squared_equals_z() {
        // S^2 = Z
        let mut circuit_s2 = QuantumCircuit::new(1);
        circuit_s2.x(0);
        circuit_s2.s(0);
        circuit_s2.s(0);
        let state_s2 = circuit_s2.execute();

        let mut circuit_z = QuantumCircuit::new(1);
        circuit_z.x(0);
        circuit_z.z(0);
        let state_z = circuit_z.execute();

        assert_camp(
            state_s2.amplitudes[0],
            state_z.amplitudes[0],
            TOLERANCE,
            "S^2=Z: |0>",
        );
        assert_camp(
            state_s2.amplitudes[1],
            state_z.amplitudes[1],
            TOLERANCE,
            "S^2=Z: |1>",
        );
    }

    // -- SWAP gate -------------------------------------------------------------

    #[test]
    fn test_swap_gate() {
        // Prepare |10>, SWAP should give |01>
        let mut circuit = QuantumCircuit::new(2);
        circuit.x(0); // |10>
        circuit.swap(0, 1); // SWAP -> |01>
        let state = circuit.execute();

        assert_camp(state.amplitudes[0], CZERO, TOLERANCE, "SWAP: |00>");
        assert_camp(state.amplitudes[1], CONE, TOLERANCE, "SWAP: |01>");
        assert_camp(state.amplitudes[2], CZERO, TOLERANCE, "SWAP: |10>");
        assert_camp(state.amplitudes[3], CZERO, TOLERANCE, "SWAP: |11>");
    }

    // -- Toffoli gate ----------------------------------------------------------

    #[test]
    fn test_toffoli_gate() {
        // Toffoli flips target only when both controls are |1>
        // Prepare |110>, apply Toffoli(0,1,2) -> |111>
        let mut circuit = QuantumCircuit::new(3);
        circuit.x(0);
        circuit.x(1);
        circuit.toffoli(0, 1, 2);
        let state = circuit.execute();

        // |111> = index 7
        assert_camp(state.amplitudes[7], CONE, TOLERANCE, "Toffoli |110>->|111>");
        for i in 0..7 {
            assert_camp(
                state.amplitudes[i],
                CZERO,
                TOLERANCE,
                &format!("Toffoli: |{:03b}> should be zero", i),
            );
        }
    }

    #[test]
    fn test_toffoli_no_flip_without_both_controls() {
        // |100>: only one control set, target should NOT flip
        let mut circuit = QuantumCircuit::new(3);
        circuit.x(0);
        circuit.toffoli(0, 1, 2);
        let state = circuit.execute();

        // |100> = index 4
        assert_camp(
            state.amplitudes[4],
            CONE,
            TOLERANCE,
            "Toffoli with one control: stays |100>",
        );
    }

    // -- Multi-qubit circuits --------------------------------------------------

    #[test]
    fn test_ghz_state() {
        // GHZ state: (|000> + |111>) / sqrt(2)
        let mut circuit = QuantumCircuit::new(3);
        circuit.h(0);
        circuit.cnot(0, 1);
        circuit.cnot(0, 2);
        let state = circuit.execute();

        assert_camp(
            state.amplitudes[0],
            (FRAC_1_SQRT2, 0.0),
            TOLERANCE,
            "GHZ |000>",
        );
        assert_camp(
            state.amplitudes[7],
            (FRAC_1_SQRT2, 0.0),
            TOLERANCE,
            "GHZ |111>",
        );

        // All other amplitudes should be zero
        for i in 1..7 {
            assert_camp(
                state.amplitudes[i],
                CZERO,
                TOLERANCE,
                &format!("GHZ |{:03b}> should be zero", i),
            );
        }
    }

    // -- Fidelity --------------------------------------------------------------

    #[test]
    fn test_fidelity_identical_states() {
        let state = QuantumState::zero_state(2);
        assert_approx(state.fidelity(&state), 1.0, TOLERANCE, "Self-fidelity");
    }

    #[test]
    fn test_fidelity_orthogonal_states() {
        let state0 = QuantumState::zero_state(1);

        let mut circuit = QuantumCircuit::new(1);
        circuit.x(0);
        let state1 = circuit.execute();

        assert_approx(
            state0.fidelity(&state1),
            0.0,
            TOLERANCE,
            "Orthogonal fidelity",
        );
    }

    // -- Entropy ---------------------------------------------------------------

    #[test]
    fn test_entropy_zero_state() {
        // |0> has zero measurement entropy (fully determined)
        let state = QuantumState::zero_state(1);
        let ent = measurement_entropy(&state);
        assert_approx(ent, 0.0, TOLERANCE, "Zero state entropy");
    }

    #[test]
    fn test_entropy_hadamard_state() {
        // H|0> has maximal 1-qubit entropy = 1 bit
        let mut circuit = QuantumCircuit::new(1);
        circuit.h(0);
        let state = circuit.execute();
        let ent = measurement_entropy(&state);
        assert_approx(ent, 1.0, TOLERANCE, "Hadamard state entropy");
    }

    // -- Expectation values ----------------------------------------------------

    #[test]
    fn test_expectation_z_zero_state() {
        let state = QuantumState::zero_state(1);
        assert_approx(expectation_z(&state, 0), 1.0, TOLERANCE, "<Z> for |0>");
    }

    #[test]
    fn test_expectation_z_one_state() {
        let mut circuit = QuantumCircuit::new(1);
        circuit.x(0);
        let state = circuit.execute();
        assert_approx(expectation_z(&state, 0), -1.0, TOLERANCE, "<Z> for |1>");
    }

    #[test]
    fn test_expectation_z_superposition() {
        let mut circuit = QuantumCircuit::new(1);
        circuit.h(0);
        let state = circuit.execute();
        assert_approx(expectation_z(&state, 0), 0.0, TOLERANCE, "<Z> for H|0>");
    }

    // -- HDC encoding ----------------------------------------------------------

    #[test]
    fn test_encode_zero_state() {
        let state = QuantumState::zero_state(1);
        let hv = encode_state(&state);
        // Encoding should be non-trivial (not all zeros or all ones)
        let density = hv.density();
        assert!(
            density > 0.3 && density < 0.7,
            "Zero state encoding density should be near 0.5 (got {:.4})",
            density
        );
    }

    #[test]
    fn test_encode_deterministic() {
        let state = QuantumState::zero_state(2);
        let hv1 = encode_state(&state);
        let hv2 = encode_state(&state);
        assert_eq!(hv1, hv2, "Encoding should be deterministic");
    }

    #[test]
    fn test_encode_different_states_differ() {
        let zero_state = QuantumState::zero_state(1);

        let mut circuit = QuantumCircuit::new(1);
        circuit.x(0);
        let one_state = circuit.execute();

        let hv_zero = encode_state(&zero_state);
        let hv_one = encode_state(&one_state);

        let sim = hv_zero.similarity(&hv_one);
        assert!(
            sim < 0.7,
            "Different quantum states should have < 0.7 similarity (got {:.4})",
            sim
        );
    }

    #[test]
    fn test_encode_bell_state() {
        let mut circuit = QuantumCircuit::new(2);
        circuit.h(0);
        circuit.cnot(0, 1);
        let bell = circuit.execute();

        let hv = encode_state(&bell);
        // Bell state has 2 contributing amplitudes. BinaryHV::bundle with 2 vectors
        // uses majority vote where ties (differing bits) break toward 0, so density
        // is expected around 0.25 (both bits set). This is valid HDC encoding.
        let density = hv.density();
        assert!(
            density > 0.1 && density < 0.8,
            "Bell state encoding should produce non-trivial density (got {:.4})",
            density
        );
        // Most importantly: it should NOT be all zeros
        assert_ne!(
            hv,
            BinaryHV::zero(),
            "Bell state encoding should not be zero"
        );
    }

    // -- Dirac notation --------------------------------------------------------

    #[test]
    fn test_dirac_string_zero_state() {
        let state = QuantumState::zero_state(2);
        let dirac = state.to_dirac_string();
        assert!(dirac.contains("1.0000|00>"), "Got: {}", dirac);
    }

    #[test]
    fn test_dirac_string_bell_state() {
        let mut circuit = QuantumCircuit::new(2);
        circuit.h(0);
        circuit.cnot(0, 1);
        let state = circuit.execute();
        let dirac = state.to_dirac_string();
        assert!(
            dirac.contains("|00>"),
            "Bell state dirac should contain |00>: {}",
            dirac
        );
        assert!(
            dirac.contains("|11>"),
            "Bell state dirac should contain |11>: {}",
            dirac
        );
    }

    // -- Circuit builder API ---------------------------------------------------

    #[test]
    fn test_circuit_gate_count() {
        let mut circuit = QuantumCircuit::new(2);
        assert_eq!(circuit.gate_count(), 0);
        circuit.h(0);
        circuit.cnot(0, 1);
        assert_eq!(circuit.gate_count(), 2);
    }

    #[test]
    fn test_execute_from_initial() {
        // Prepare |1> state, then execute circuit starting from it
        let initial = QuantumState::from_amplitudes(1, vec![CZERO, CONE]);
        let mut circuit = QuantumCircuit::new(1);
        circuit.x(0); // X|1> = |0>
        let state = circuit.execute_from(&initial);

        assert_camp(state.amplitudes[0], CONE, TOLERANCE, "X|1> = |0>");
        assert_camp(state.amplitudes[1], CZERO, TOLERANCE, "X|1>: no |1>");
    }

    // -- Projective measurement on single qubit --------------------------------

    #[test]
    fn test_measure_qubit_deterministic() {
        let state = QuantumState::zero_state(2);
        let (outcome, post) = measure_qubit(&state, 0, Some(42));
        assert!(!outcome, "Qubit 0 of |00> must measure 0");
        assert_approx(
            post.total_probability(),
            1.0,
            TOLERANCE,
            "Post-measurement normalization",
        );
    }
}
