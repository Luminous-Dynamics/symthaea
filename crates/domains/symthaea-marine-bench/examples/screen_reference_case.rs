// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;

use symthaea_marine_bench::{MarineFreightCase, assess};

fn main() -> Result<(), Box<dyn Error>> {
    let input = include_str!("../fixtures/marine_freight_reference_case_v2.synthetic.json");
    let case: MarineFreightCase = serde_json::from_str(input)?;
    let report = assess(&case)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
