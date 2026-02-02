//! RISC-0 Methods Library
//!
//! Exports the compiled guest programs (ELF) and their image IDs.
//!
//! # Building
//!
//! The guest programs are compiled during the build process:
//! ```bash
//! cd methods
//! cargo build --release
//! ```
//!
//! This generates:
//! - `GRADIENT_QUALITY_ELF`: The compiled RISC-V binary
//! - `GRADIENT_QUALITY_ID`: The image ID for verification

// Include the generated constants from build.rs
include!(concat!(env!("OUT_DIR"), "/methods.rs"));
