// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sample tool — generate output for a batch of prompts and print them
//! so goldens can be written to match the generator's natural shape.
//! Not a scorer; just a bulk `generate_nix` caller.
//!
//! Run:
//!   cargo run --features code_generation --example nix_scorer_sample

use symthaea::language::nix_codegen::generate_nix;

fn main() {
    let prompts = [
        "configure a python data-science environment with jupyter and pandas",
        "set up a node development environment with typescript",
        "configure nvidia gpu drivers",
        "set up sway window manager",
        "open firewall ports 80 and 443",
        "configure postgresql service",
        "set up ipfs kubo node",
        "enable kde plasma desktop environment",
    ];
    for prompt in prompts {
        println!("=== {prompt} ===");
        let r = generate_nix(prompt);
        println!("{}", r.code);
        println!();
    }
}
