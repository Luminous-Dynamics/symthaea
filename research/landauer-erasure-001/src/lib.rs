#![forbid(unsafe_code)]
#![deny(warnings)]

// Compile and test the exact production arithmetic source by path.
#[path = "../../../crates/core/symthaea-core/src/physics/landauer_erasure.rs"]
pub mod landauer_erasure;
