//! RISC-0 Methods Build Script
//!
//! Compiles guest programs to RISC-V and generates Rust constants
//! for the ELF binary and image ID.

use std::env;
use std::path::PathBuf;

fn main() {
    // Only build guest when risc0 toolchain is available
    if env::var("RISC0_TOOLCHAIN").is_ok() || env::var("RISC0_BUILD_GUEST").is_ok() {
        build_with_risc0();
    } else {
        build_stub();
    }
}

/// Build with real RISC-0 toolchain
#[cfg(feature = "risc0-build")]
fn build_with_risc0() {
    use risc0_build::{embed_methods_with_options, GuestOptions};

    let guest_pkg = PathBuf::from("gradient-quality/guest");

    embed_methods_with_options(
        &[guest_pkg],
        GuestOptions::default(),
    );
}

#[cfg(not(feature = "risc0-build"))]
fn build_with_risc0() {
    build_stub();
}

/// Generate stub when RISC-0 toolchain not available
fn build_stub() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let methods_rs = out_dir.join("methods.rs");

    // Generate stub constants
    let stub = r#"
/// Stub ELF for gradient quality guest (real binary requires risc0-build)
pub const GRADIENT_QUALITY_ELF: &[u8] = &[];

/// Stub image ID for gradient quality guest
pub const GRADIENT_QUALITY_ID: [u32; 8] = [0u32; 8];
"#;

    std::fs::write(methods_rs, stub).expect("Failed to write methods.rs stub");

    println!("cargo:warning=RISC-0 toolchain not found, using stub ELF");
    println!("cargo:warning=Set RISC0_BUILD_GUEST=1 to build real guest programs");
}
