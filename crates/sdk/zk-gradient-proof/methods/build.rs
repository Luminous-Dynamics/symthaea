// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Build script for RISC-0 guest methods compilation
//!
//! This script runs during `cargo build` and:
//! 1. Compiles the guest Rust code to RISC-V
//! 2. Generates the GRADIENT_QUALITY_ELF binary
//! 3. Generates the GRADIENT_QUALITY_ID (image identifier)

fn main() {
    risc0_build::embed_methods();
}
