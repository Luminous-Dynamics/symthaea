#![forbid(unsafe_code)]

// This standalone harness compiles the exact production source files. It does
// not copy or reimplement the allocation or sign-test algorithms.
#[path = "../../../src/consciousness/recursive_improvement/paired_sign_test.rs"]
pub mod paired_sign_test;
#[path = "../../../src/consciousness/recursive_improvement/reuse_error_budget.rs"]
pub mod reuse_error_budget;
#[path = "../../../src/consciousness/recursive_improvement/reuse_confidence_certificate.rs"]
pub mod reuse_confidence_certificate;
