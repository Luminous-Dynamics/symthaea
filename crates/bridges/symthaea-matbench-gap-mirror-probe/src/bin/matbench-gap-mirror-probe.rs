// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::Serialize;
use std::{env, fs, process};
use symthaea_matbench_gap_mirror_probe::probe_known_github_mirror;

#[derive(Serialize)]
struct Output {
    receipt_sha256: String,
    receipt: symthaea_matbench_gap_mirror_probe::ExploratoryMirrorOverlapReceipt,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("matbench-gap-mirror-probe: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os();
    let _program = args.next();
    let input = args
        .next()
        .ok_or("usage: matbench-gap-mirror-probe <matbench_expt_gap.csv>")?;
    if args.next().is_some() {
        return Err("usage: matbench-gap-mirror-probe <matbench_expt_gap.csv>".into());
    }

    let csv_bytes = fs::read(input)?;
    let receipt = probe_known_github_mirror(&csv_bytes)?;
    let output = Output {
        receipt_sha256: receipt.sha256()?,
        receipt,
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
