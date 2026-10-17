// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use symthaea_marine_bench::catamaran_design_space::{
    generate_design_space_with_revision, verify_design_space_receipt, DesignSpaceSpec,
};

fn main() -> Result<(), Box<dyn Error>> {
    let source_revision = std::env::var("SYMTHAEA_SOURCE_REVISION")
        .unwrap_or_else(|_| "unavailable".to_owned());
    let spec = DesignSpaceSpec {
        length_waterline_m: 20.0,
        // These are illustrative sweep inputs only, not recommended hull ratios.
        hull_spacing_over_lwl: vec![0.35, 0.40, 0.45],
        demihull_beam_over_lwl: vec![0.08, 0.10, 0.12],
        draft_over_lwl: vec![0.05, 0.06],
        wet_deck_clearance_over_draft: vec![0.60, 0.75, 0.90],
        froude_numbers: vec![0.20, 0.30, 0.40],
        max_candidates: 256,
    };
    let receipt = generate_design_space_with_revision(&spec, &source_revision)?;
    if !verify_design_space_receipt(&receipt)? {
        return Err("generated design-space receipt did not verify".into());
    }
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
