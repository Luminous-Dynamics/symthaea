// Compile the production verification example against the test-only
// symbolic-source shim. The full production HDC/workspace integration is
// explicitly outside this standalone harness.
include!("../../../../crates/bridges/symthaea-physics-bridge/examples/verify_invariants_formal.rs");
