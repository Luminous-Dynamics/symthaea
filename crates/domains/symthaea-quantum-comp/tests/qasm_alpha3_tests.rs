#![cfg(feature = "qasm-export")]

use symthaea_quantum_comp::{BinaryHypervector, qasm::export_bell_parity_register};

#[test]
fn bell_parity_register_exports() {
    let parity = BinaryHypervector::random(8, 5).unwrap();
    let qasm = export_bell_parity_register(&parity, 4).unwrap();
    assert!(qasm.contains("OPENQASM 2.0"));
    assert!(qasm.contains("qreg q[8]"));
    assert!(qasm.contains("cx q[0], q[1]"));
}
