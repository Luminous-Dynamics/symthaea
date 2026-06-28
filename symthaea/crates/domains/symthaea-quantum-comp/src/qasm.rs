//! Tiny OpenQASM export helpers.
//!
//! These helpers intentionally produce educational toy circuits only. They are
//! not backend execution adapters and do not perform transpilation.

use crate::classical_hdc::BinaryHypervector;
use crate::correlation_hdc::CorrelationBindingSketch;
use crate::errors::{QuantumCompError, Result};

/// Exports a toy OpenQASM 2 circuit that prepares parity bits for a small sketch.
///
/// The circuit represents each parity bit as a classical basis-state preparation.
/// It is useful for testing external tooling integration, not for claiming quantum
/// advantage or physical entanglement.
pub fn export_parity_basis_qasm(
    sketch: &CorrelationBindingSketch,
    max_bits: usize,
) -> Result<String> {
    if max_bits == 0 {
        return Err(QuantumCompError::InvalidDimension);
    }
    let bits = sketch.dimension().min(max_bits);
    let mut out = String::new();
    out.push_str("OPENQASM 2.0;\n");
    out.push_str("include \"qelib1.inc\";\n");
    out.push_str(&format!("qreg q[{}];\n", bits));
    out.push_str(&format!("creg c[{}];\n", bits));
    for i in 0..bits {
        if sketch.parity().bit(i).unwrap_or(false) {
            out.push_str(&format!("x q[{}];\n", i));
        }
    }
    for i in 0..bits {
        out.push_str(&format!("measure q[{}] -> c[{}];\n", i, i));
    }
    Ok(out)
}

/// Exports a tiny Bell-pair parity demonstration for one binary item/key bit.
///
/// This is only a teaching circuit showing how parity can be expressed through
/// correlated measurements. It does not encode a full HDC vector.
pub fn export_single_bit_bell_parity_demo(item_bit: bool, key_bit: bool) -> String {
    let parity = item_bit ^ key_bit;
    let mut out = String::new();
    out.push_str("OPENQASM 2.0;\n");
    out.push_str("include \"qelib1.inc\";\n");
    out.push_str("qreg q[2];\n");
    out.push_str("creg c[2];\n");
    out.push_str("h q[0];\n");
    out.push_str("cx q[0], q[1];\n");
    if parity {
        out.push_str("x q[1];\n");
    }
    out.push_str("measure q[0] -> c[0];\n");
    out.push_str("measure q[1] -> c[1];\n");
    out
}

/// Convenience function that creates a sketch and exports a small parity circuit.
pub fn export_binding_parity_qasm(
    item: &BinaryHypervector,
    key: &BinaryHypervector,
    max_bits: usize,
) -> Result<String> {
    let sketch = CorrelationBindingSketch::bind(item, key)?;
    export_parity_basis_qasm(&sketch, max_bits)
}

/// Exports a toy OpenQASM 2 circuit for several Bell-pair parity demonstrations.
///
/// Each pair is prepared as a Bell pair, optionally flips the second qubit to
/// encode the requested parity, then measures both qubits. This is an integration
/// artifact for external tooling tests, not a hardware-efficiency claim.
pub fn export_bell_parity_register(parity: &BinaryHypervector, max_pairs: usize) -> Result<String> {
    if max_pairs == 0 {
        return Err(QuantumCompError::InvalidDimension);
    }
    let pairs = parity.dimension().min(max_pairs);
    let qubits = pairs * 2;
    let mut out = String::new();
    out.push_str("OPENQASM 2.0;\n");
    out.push_str("include \"qelib1.inc\";\n");
    out.push_str(&format!("qreg q[{}];\n", qubits));
    out.push_str(&format!("creg c[{}];\n", qubits));
    for i in 0..pairs {
        let a = i * 2;
        let b = a + 1;
        out.push_str(&format!("h q[{}];\n", a));
        out.push_str(&format!("cx q[{}], q[{}];\n", a, b));
        if parity.bit(i).unwrap_or(false) {
            out.push_str(&format!("x q[{}];\n", b));
        }
    }
    for i in 0..qubits {
        out.push_str(&format!("measure q[{}] -> c[{}];\n", i, i));
    }
    Ok(out)
}
