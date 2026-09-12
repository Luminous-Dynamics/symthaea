// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::{env, fs, process};
use symthaea_discovery::CandidateId;
use symthaea_material_process_burden::{
    bind_process_burden_from_bytes, ProcessBinding, ProcessBindingBasis,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("material-process-burden: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let data_path = args.next().ok_or("missing <process-data.json>")?;
    let source_path = args.next().ok_or("missing <source-document>")?;
    let candidate_id = args.next().ok_or("missing <candidate-id>")?;
    let material_id = args.next().ok_or("missing <material-id>")?;
    let process_id = args.next().ok_or("missing <process-id>")?;
    let mapping_note = args.next();
    if args.next().is_some() {
        return Err("usage: material-process-burden <process-data.json> <source-document> <candidate-id> <material-id> <process-id> [explicit-mapping-note]".into());
    }

    let binding = ProcessBinding {
        candidate_id: CandidateId::new(candidate_id)?,
        material_id,
        process_id,
        basis: match mapping_note {
            Some(note) => ProcessBindingBasis::ExplicitMapping { note },
            None => ProcessBindingBasis::ExactMaterialId,
        },
    };

    let data = fs::read(data_path)?;
    let source_document = fs::read(source_path)?;
    let receipt = bind_process_burden_from_bytes(&data, &source_document, binding)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
