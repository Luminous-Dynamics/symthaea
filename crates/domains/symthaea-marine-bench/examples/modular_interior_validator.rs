// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Prints deterministic configuration receipts for the synthetic marine interior fixtures.
//! The fixtures intentionally produce an indeterminate (not approved) disposition.

use std::error::Error;

use symthaea_marine_bench::modular_interior::{
    validate_configuration, verify_validation_receipt, Configuration,
};

fn report(name: &str, json: &str) -> Result<bool, Box<dyn Error>> {
    let configuration: Configuration = serde_json::from_str(json)?;
    let receipt = validate_configuration(&configuration)?;
    if !verify_validation_receipt(&configuration, &receipt) {
        return Err(format!("receipt failed integrity check for {name}").into());
    }
    println!("=== {name} ===");
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(receipt.body.overall_disposition != "rejected")
}

fn main() -> Result<(), Box<dyn Error>> {
    let liveaboard = include_str!("../fixtures/modular-interiors/liveaboard-custom-layout-v1.json");
    let expedition = include_str!("../fixtures/modular-interiors/expedition-research-layout-v1.json");

    let liveaboard_ok = report("liveaboard", liveaboard)?;
    let expedition_ok = report("expedition", expedition)?;
    if !(liveaboard_ok && expedition_ok) {
        return Err("one or more configurations were rejected by bookkeeping checks".into());
    }
    Ok(())
}
