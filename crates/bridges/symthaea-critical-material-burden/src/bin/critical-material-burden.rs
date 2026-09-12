// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::{env, fs, process};
use symthaea_critical_material_burden::{
    calculate_critical_material_burden, CriticalElementList,
};
use symthaea_discovery::CandidateId;

fn main() {
    if let Err(error) = run() {
        eprintln!("critical-material-burden: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let list_path = args.next().ok_or("missing <critical-elements.json>")?;
    let source_path = args.next().ok_or("missing <source-document>")?;
    let candidate_id = args.next().ok_or("missing <candidate-id>")?;
    let formula = args.next().ok_or("missing <formula>")?;
    if args.next().is_some() {
        return Err("usage: critical-material-burden <critical-elements.json> <source-document> <candidate-id> <formula>".into());
    }

    let list_bytes = fs::read(list_path)?;
    let source_document = fs::read(source_path)?;
    let list = CriticalElementList::from_json_bytes(&list_bytes)?;
    let receipt = calculate_critical_material_burden(
        CandidateId::new(candidate_id)?,
        &formula,
        &list,
        &source_document,
    )?;

    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
