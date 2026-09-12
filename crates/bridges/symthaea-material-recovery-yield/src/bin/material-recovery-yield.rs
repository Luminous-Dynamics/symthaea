// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::{env, fs, process};
use symthaea_discovery::CandidateId;
use symthaea_material_recovery_yield::{
    bind_recovery_yield_from_bytes, RecoveryBinding, RecoveryBindingBasis,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("material-recovery-yield: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let data_path = args.next().ok_or("missing <recovery-data.json>")?;
    let source_path = args.next().ok_or("missing <source-document>")?;
    let candidate_id = args.next().ok_or("missing <candidate-id>")?;
    let material_id = args.next().ok_or("missing <material-id>")?;
    let process_id = args.next().ok_or("missing <process-id>")?;
    let mapping_note = args.next();
    if args.next().is_some() {
        return Err("usage: material-recovery-yield <recovery-data.json> <source-document> <candidate-id> <material-id> <process-id> [explicit-mapping-note]".into());
    }

    let binding = RecoveryBinding {
        candidate_id: CandidateId::new(candidate_id)?,
        material_id,
        process_id,
        basis: match mapping_note {
            Some(note) => RecoveryBindingBasis::ExplicitMapping { note },
            None => RecoveryBindingBasis::ExactMaterialId,
        },
    };

    let data = fs::read(data_path)?;
    let source_document = fs::read(source_path)?;
    let receipt = bind_recovery_yield_from_bytes(&data, &source_document, binding)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
