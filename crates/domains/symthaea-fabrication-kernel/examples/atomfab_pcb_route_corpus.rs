// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Run the independent synthetic AtomFab PCB route/evidence evaluator.
//!
//! Usage:
//!   ATOMFAB_GIT_HEAD="$(git rev-parse HEAD)" \
//!     cargo run -p symthaea-fabrication-kernel --example atomfab_pcb_route_corpus
//!
//! An optional first argument may provide a replacement corpus JSON path. The
//! caller must provide an exact 40- or 64-character Git commit SHA; an unbound
//! result is not emitted as a passing receipt.

use std::{
    env, fs,
    process::{Command, ExitCode},
};
use symthaea_fabrication_kernel::atomfab_pcb_route::{EMBEDDED_CORPUS, evaluate_corpus_bytes};

fn main() -> ExitCode {
    let git_head = match env::var("ATOMFAB_GIT_HEAD") {
        Ok(value) => value,
        Err(_) => {
            eprintln!("ATOMFAB_GIT_HEAD is required; set it to the exact repository commit SHA");
            return ExitCode::from(2);
        }
    };

    let git_output = match Command::new("git").args(["rev-parse", "HEAD"]).output() {
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
    let actual_git_head = match String::from_utf8(git_output.stdout) {
        Ok(value) => value.trim().to_string(),
        Err(error) => {
            eprintln!("git rev-parse HEAD returned non-UTF-8 output: {error}");
            return ExitCode::from(2);
        }
    };
    if !git_head.trim().eq_ignore_ascii_case(&actual_git_head) {
        eprintln!("ATOMFAB_GIT_HEAD does not match actual checkout HEAD");
        return ExitCode::from(2);
    }

    let corpus_bytes = match env::args_os().nth(1) {
        Some(path) => match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) => {
                eprintln!("failed to read corpus {}: {error}", path.to_string_lossy());
                return ExitCode::from(2);
            }
        },
        None => EMBEDDED_CORPUS.as_bytes().to_vec(),
    };

    let receipt = match evaluate_corpus_bytes(&corpus_bytes, &git_head, &actual_git_head) {
        Ok(receipt) => receipt,
        Err(error) => {
            eprintln!("AtomFab corpus rejected: {error}");
            return ExitCode::from(2);
        }
    };

    match serde_json::to_string_pretty(&receipt) {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("failed to serialize AtomFab receipt: {error}");
            return ExitCode::from(2);
        }
    }

    if receipt.passed() {
        ExitCode::SUCCESS
    } else {
        eprintln!("AtomFab corpus has expectation mismatches; receipt status is failed");
        ExitCode::from(1)
    }
}
