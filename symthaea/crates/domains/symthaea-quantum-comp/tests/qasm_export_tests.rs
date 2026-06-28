#![cfg(feature = "qasm-export")]

use symthaea_quantum_comp::BinaryHypervector;
use symthaea_quantum_comp::qasm::export_binding_parity_qasm;

#[test]
fn qasm_export_contains_header_and_measurements() {
    let item = BinaryHypervector::random(8, 1).unwrap();
    let key = BinaryHypervector::random(8, 2).unwrap();
    let qasm = export_binding_parity_qasm(&item, &key, 8).unwrap();
    assert!(qasm.contains("OPENQASM 2.0"));
    assert!(qasm.contains("measure q[0] -> c[0];"));
}
