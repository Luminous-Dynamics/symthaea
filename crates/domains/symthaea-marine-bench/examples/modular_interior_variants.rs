// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Generates a drop-in sleeping-module alternative and prints the validation/impact report.
//! The sample fixture is synthetic; the candidate should remain evidence-indeterminate.

use std::error::Error;

use symthaea_marine_bench::modular_interior::Configuration;
use symthaea_marine_bench::modular_interior_variants::{
    generate_drop_in_variants, verify_variant_generation_report, ModuleVariant,
};

fn main() -> Result<(), Box<dyn Error>> {
    let json = include_str!("../fixtures/modular-interiors/liveaboard-custom-layout-v1.json");
    let baseline: Configuration = serde_json::from_str(json)?;
    let mut replacement = baseline.module_catalog.iter()
        .find(|module| module.module_id == "SLEEP-POD")
        .ok_or("fixture has no sleeping pod definition")?
        .clone();

    replacement.revision += 1;
    replacement.name.push_str(" longer berth");
    replacement.envelope_mm.length_mm += 100;
    // Ports and declared ratings are unchanged: this exercises direct-drop-in preflight.
    // The generated geometry, mass/CG, utility and hard-safety checks still run afterward.

    let variants = vec![ModuleVariant {
        variant_id: "SLEEP-POD-LONG-V2".to_owned(),
        replacement,
    }];
    let report = generate_drop_in_variants(&baseline, &variants)?;
    if !verify_variant_generation_report(&baseline, &variants, &report) {
        return Err("variant generation report failed deterministic recomputation".into());
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
