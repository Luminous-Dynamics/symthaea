#![forbid(unsafe_code)]
#![deny(warnings)]

// Compile the exact production source file, not a copied test implementation.
#[path = "../../../crates/core/symthaea-core/src/physics/energy_meter.rs"]
pub mod energy_meter;
