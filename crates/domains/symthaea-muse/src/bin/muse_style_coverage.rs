// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Emit the deterministic 152-style source-coverage matrix as JSON.

use std::env;
use std::fs;
use std::io::{self, Write};
use std::process::ExitCode;

use symthaea_muse::style_coverage::collect_style_coverage;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muse_style_coverage: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut output_path = None::<String>;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output" | "-o" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a path"))?;
                if output_path.replace(value).is_some() {
                    return Err("output path was supplied more than once".into());
                }
            }
            "--help" | "-h" => {
                println!(
                    "Usage: muse_style_coverage [--output PATH]\n\
                     Emit the deterministic canonical Muse style coverage matrix as JSON.\n\
                     Without --output, JSON is written to stdout."
                );
                return Ok(());
            }
            other => return Err(format!("unknown argument {other:?}; use --help")),
        }
    }

    let report = collect_style_coverage();
    if !report.verify_digest() {
        return Err("generated report digest did not verify".into());
    }
    let mut bytes = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("serialize report: {error}"))?;
    bytes.push(b'\n');

    match output_path {
        Some(path) => {
            fs::write(&path, &bytes)
                .map_err(|error| format!("write {}: {error}", path))?;
            eprintln!(
                "wrote {} styles ({} routes, {} without routes); sha256={}",
                report.summary.catalog_entry_count,
                report.summary.composer_routes_resolved,
                report.summary.catalog_entries_without_routes,
                report.report_sha256
            );
        }
        None => {
            let stdout = io::stdout();
            let mut lock = stdout.lock();
            lock.write_all(&bytes)
                .map_err(|error| format!("write stdout: {error}"))?;
            lock.flush()
                .map_err(|error| format!("flush stdout: {error}"))?;
        }
    }
    Ok(())
}
