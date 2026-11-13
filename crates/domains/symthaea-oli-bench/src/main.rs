// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Run the OLI great-ape spatial cognition descriptive audit.

use std::env;
use std::fs;
use symthaea_oli_bench::audit_csv;

fn main() {
    if let Err(error) = run() {
        eprintln!("OLI ape spatial audit failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let input = match env::args().nth(1) {
        Some(path) => fs::read_to_string(path)?,
        None => include_str!("../fixtures/albiach2010great_pseudonymous.csv").to_owned(),
    };
    let report = audit_csv(&input).map_err(|error| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, error)
    })?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
