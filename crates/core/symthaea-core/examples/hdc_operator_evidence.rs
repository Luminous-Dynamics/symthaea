//! Emit deterministic machine-readable SIMD operator evidence.
//!
//! Research-only executable. This runs the scalar/SIMD conformance matrix and
//! emits JSON; it is not a throughput benchmark.

fn main() {
    println!(
        "{}",
        symthaea_core::hdc::operator_evidence::extended_resolution_evidence_json()
    );
}
