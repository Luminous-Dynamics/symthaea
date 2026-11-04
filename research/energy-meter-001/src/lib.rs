#![forbid(unsafe_code)]
#![deny(warnings)]

// Compile the exact production source files, not copied test implementations.
pub mod physics {
    #[path = "../../../crates/core/symthaea-core/src/physics/energy_meter.rs"]
    pub mod energy_meter;

    #[path = "../../../crates/core/symthaea-core/src/physics/energy_receipt.rs"]
    pub mod energy_receipt;
}
