// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Compares two synthetic modular-interior configurations and prints revalidation impacts.
//! The report is change-impact evidence only, not a vessel safety certificate.

use std::error::Error;

use symthaea_marine_bench::modular_interior::Configuration;
use symthaea_marine_bench::modular_interior_impact::{
    analyze_configuration_change, verify_impact_report,
};

fn main() -> Result<(), Box<dyn Error>> {
    let liveaboard_json =
        include_str!("../fixtures/modular-interiors/liveaboard-custom-layout-v1.json");
    let expedition_json =
        include_str!("../fixtures/modular-interiors/expedition-research-layout-v1.json");
    let liveaboard: Configuration = serde_json::from_str(liveaboard_json)?;
    let expedition: Configuration = serde_json::from_str(expedition_json)?;

    let report = analyze_configuration_change(&liveaboard, &expedition)?;
    if !verify_impact_report(&liveaboard, &expedition, &report) {
        return Err("configuration-impact report failed its integrity check".into());
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
