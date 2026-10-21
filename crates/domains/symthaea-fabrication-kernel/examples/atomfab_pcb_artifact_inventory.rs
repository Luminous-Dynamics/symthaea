// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Produce a checkout-bound inventory of actual files listed by an artifact manifest.

use serde::Serialize;
use std::{
    env, fs,
    process::{Command, ExitCode},
};

use symthaea_fabrication_kernel::atomfab_pcb_artifacts::evaluate_artifact_inventory;

#[derive(Serialize)]
struct HeadBoundReceipt {
    schema: &'static str,
    git_head: String,
    exact_head_bound: bool,
    inventory: symthaea_fabrication_kernel::atomfab_pcb_artifacts::ArtifactInventoryReceipt,
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let manifest_path = match args.next() {
        Some(value) => value,
        None => {
            eprintln!("usage: atomfab_pcb_artifact_inventory MANIFEST_PATH ARTIFACT_ROOT");
            return ExitCode::from(2);
        }
    };
    let artifact_root = match args.next() {
        Some(value) => value,
        None => {
            eprintln!("usage: atomfab_pcb_artifact_inventory MANIFEST_PATH ARTIFACT_ROOT");
            return ExitCode::from(2);
        }
    };
    if args.next().is_some() {
        eprintln!("too many arguments");
        return ExitCode::from(2);
    }

    let declared_head = match env::var("ATOMFAB_GIT_HEAD") {
        Ok(value) => value,
        Err(_) => {
            eprintln!("ATOMFAB_GIT_HEAD is required");
            return ExitCode::from(2);
        }
    };
    let actual_head = match Command::new("git").args(["rev-parse", "HEAD"]).output() {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        }
        Ok(_) => {
            eprintln!("git rev-parse HEAD failed");
            return ExitCode::from(2);
        }
        Err(error) => {
            eprintln!("cannot run git rev-parse HEAD: {error}");
            return ExitCode::from(2);
        }
    };
    if declared_head.len() != 40
        || !declared_head.bytes().all(|byte| byte.is_ascii_hexdigit())
        || declared_head != actual_head
    {
        eprintln!("declared Git HEAD does not match actual checkout HEAD");
        return ExitCode::from(2);
    }

    let manifest = match fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot read {manifest_path}: {error}");
            return ExitCode::from(2);
        }
    };
    let inventory = match evaluate_artifact_inventory(&manifest, &artifact_root) {
        Ok(receipt) => receipt,
        Err(error) => {
            eprintln!("artifact inventory rejected: {error}");
            return ExitCode::from(2);
        }
    };

    let is_complete = inventory.status == "complete";
    let receipt = HeadBoundReceipt {
        schema: "atomfab-pcb-artifact-inventory-head-receipt-v0.1",
        git_head: actual_head,
        exact_head_bound: true,
        inventory,
    };

    match serde_json::to_string_pretty(&receipt) {
        Ok(json) => {
            println!("{json}");
            if is_complete {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(error) => {
            eprintln!("cannot serialize artifact inventory receipt: {error}");
            ExitCode::from(2)
        }
    }
}
