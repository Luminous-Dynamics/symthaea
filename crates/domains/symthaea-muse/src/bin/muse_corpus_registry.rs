// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Emit the rights-aware candidate music corpus registry as JSON.

use std::env;
use std::fs;
use std::io::{self, Write};
use std::process::ExitCode;

use symthaea_muse::music_corpus_registry::build_music_corpus_registry;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muse_corpus_registry: {error}");
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
                    "Usage: muse_corpus_registry [--output PATH]\n\
                     Emit candidate corpora, rights constraints and ingest gates as JSON.\n\
                     No dataset is downloaded or approved by this command.\n\
                     Without --output, JSON is written to stdout."
                );
                return Ok(());
            }
            other => return Err(format!("unknown argument {other:?}; use --help")),
        }
    }

    let registry = build_music_corpus_registry();
    if !registry.verify_digest() {
        return Err("generated corpus registry digest did not verify".into());
    }
    let mut bytes = serde_json::to_vec_pretty(&registry)
        .map_err(|error| format!("serialize corpus registry: {error}"))?;
    bytes.push(b'\n');

    match output_path {
        Some(path) => {
            fs::write(&path, &bytes)
                .map_err(|error| format!("write {}: {error}", path))?;
            eprintln!(
                "wrote {} candidate corpora; {} approved for ingest; sha256={}",
                registry.candidate_count,
                registry.current_ingest_approved_count,
                registry.registry_sha256
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
