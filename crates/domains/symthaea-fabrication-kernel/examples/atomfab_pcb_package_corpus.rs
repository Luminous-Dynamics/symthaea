// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Validate and report the synthetic AtomFab PCB package corpus.
//!
//! This checks package-to-design/DRC/output identity bindings, not physical
//! manufacturability or authorization to execute the resulting files.

use std::{
    env, fs,
    process::{Command, ExitCode},
};
use symthaea_fabrication_kernel::atomfab_pcb_package::{
    EMBEDDED_PACKAGE_CORPUS, evaluate_package_corpus_bytes,
};

fn main() -> ExitCode {
    let declared_head = match env::var("ATOMFAB_GIT_HEAD") {
        Ok(value) => value,
        Err(_) => {
            eprintln!("ATOMFAB_GIT_HEAD is required; set it to the exact repository commit SHA");
            return ExitCode::from(2);
        }
    };

    let output = match Command::new("git").args(["rev-parse", "HEAD"]).output() {
        Ok(output) if output.status.success() => output,
        Ok(output) => {
            eprintln!(
                "git rev-parse HEAD failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return ExitCode::from(2);
        }
        Err(error) => {
            eprintln!("failed to execute git rev-parse HEAD: {error}");
            return ExitCode::from(2);
        }
    };
    let actual_head = match String::from_utf8(output.stdout) {
        Ok(value) => value.trim().to_string(),
        Err(error) => {
            eprintln!("git rev-parse HEAD returned non-UTF-8 output: {error}");
            return ExitCode::from(2);
        }
    };
    if !declared_head.trim().eq_ignore_ascii_case(&actual_head) {
        eprintln!("ATOMFAB_GIT_HEAD does not match actual checkout HEAD");
        return ExitCode::from(2);
    }

    let bytes = match env::args_os().nth(1) {
        Some(path) => match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) => {
                eprintln!(
                    "failed to read package corpus {}: {error}",
                    path.to_string_lossy()
                );
                return ExitCode::from(2);
            }
        },
        None => EMBEDDED_PACKAGE_CORPUS.as_bytes().to_vec(),
    };
    let receipt = match evaluate_package_corpus_bytes(&bytes, &declared_head, &actual_head) {
        Ok(receipt) => receipt,
        Err(error) => {
            eprintln!("AtomFab package corpus rejected: {error}");
            return ExitCode::from(2);
        }
    };

    match serde_json::to_string_pretty(&receipt) {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("failed to serialize package receipt: {error}");
            return ExitCode::from(2);
        }
    }
    if receipt.passed() {
        ExitCode::SUCCESS
    } else {
        eprintln!("AtomFab package corpus has expectation mismatches; status is failed");
        ExitCode::from(1)
    }
}
