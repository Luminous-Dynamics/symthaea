// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::Deserialize;
use std::{env, fs, process};
use symthaea_matbench_gap_mirror_probe::{
    probe_known_github_mirror, ExploratoryMirrorOverlapReceipt,
};

#[derive(Deserialize)]
struct ProbeOutput {
    receipt_sha256: String,
    receipt: ExploratoryMirrorOverlapReceipt,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("matbench-gap-mirror-verify: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os();
    let _program = args.next();
    let csv_path = args.next().ok_or(
        "usage: matbench-gap-mirror-verify <matbench_expt_gap.csv> <probe-output.json>",
    )?;
    let receipt_path = args.next().ok_or(
        "usage: matbench-gap-mirror-verify <matbench_expt_gap.csv> <probe-output.json>",
    )?;
    if args.next().is_some() {
        return Err(
            "usage: matbench-gap-mirror-verify <matbench_expt_gap.csv> <probe-output.json>".into(),
        );
    }

    let csv_bytes = fs::read(csv_path)?;
    let provided: ProbeOutput = serde_json::from_slice(&fs::read(receipt_path)?)?;
    provided.receipt.validate()?;
    let provided_sha = provided.receipt.sha256()?;
    if provided_sha != provided.receipt_sha256 {
        return Err(format!(
            "provided receipt digest mismatch: declared {}, recomputed {}",
            provided.receipt_sha256, provided_sha
        )
        .into());
    }

    let recomputed = probe_known_github_mirror(&csv_bytes)?;
    if recomputed != provided.receipt {
        return Err("receipt does not replay exactly from the supplied known-mirror CSV bytes".into());
    }
    let recomputed_sha = recomputed.sha256()?;
    if recomputed_sha != provided.receipt_sha256 {
        return Err("recomputed receipt digest differs from provided digest".into());
    }

    println!("VERIFIED {}", recomputed_sha);
    Ok(())
}
