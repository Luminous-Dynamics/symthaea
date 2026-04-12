// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Run the 100-function code generation benchmark.
//!
//! Run: `cargo run --example run_code_benchmark --features code_generation`
//!
//! Generates 100 canonical Rust functions via the native tier and reports:
//! - Overall non-todo rate (functions with real bodies)
//! - Per-category breakdown
//! - Failures list
//!
//! If trained weights exist, they're auto-loaded. Run train_code_sequencer first
//! for best results.

use symthaea::hdc::code_encoder::CodeHDEncoder;
use symthaea::language::code_generator::CodeGenerator;
use symthaea::language::sequencer_benchmark::run_benchmark;

fn main() {
    println!("=== 100-Function Code Generation Benchmark ===\n");

    // Create generator (auto-loads trained weights if available)
    let encoder = CodeHDEncoder::new(512);
    let generator = CodeGenerator::new(encoder);

    // Run benchmark
    let result = run_benchmark(&generator);

    // Print report
    println!("{}", result.report());
}
