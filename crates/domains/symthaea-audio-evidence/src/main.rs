// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use symthaea_audio_evidence::check_manifest_bundle;

fn usage() {
    eprintln!(
        "Usage: audio-evidence-check <manifest.json> [bundle-root]\n\
         Verifies Open Audio Evidence project invariants and referenced file hashes.\n\
         It is not a complete JSON Schema validator or standards-compliance certificate."
    );
}

fn main() -> ExitCode {
    let mut args = env::args_os().skip(1);
    let Some(manifest_arg) = args.next() else {
        usage();
        return ExitCode::from(2);
    };
    let manifest_path = PathBuf::from(manifest_arg);
    let bundle_root = args.next().map(PathBuf::from).unwrap_or_else(|| {
        let parent = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        if parent.as_os_str().is_empty() {
            PathBuf::from(".")
        } else {
            parent.to_path_buf()
        }
    });
    if args.next().is_some() {
        usage();
        return ExitCode::from(2);
    }
    let manifest_bytes = match fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("ERROR: cannot read {}: {error}", manifest_path.display());
            return ExitCode::FAILURE;
        }
    };
    let manifest: Value = match serde_json::from_slice(&manifest_bytes) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("ERROR: invalid JSON in {}: {error}", manifest_path.display());
            return ExitCode::FAILURE;
        }
    };
    match check_manifest_bundle(&manifest, &bundle_root) {
        Ok(count) => {
            let status = manifest
                .get("data_status")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            if status == "planned" {
                println!("PLAN OK: no measurement verdict; {count} artifacts checked");
            } else {
                println!(
                    "MANIFEST OK: status={status}; {count} artifacts checked; \
                     acoustic gate and compliance not assessed"
                );
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}
