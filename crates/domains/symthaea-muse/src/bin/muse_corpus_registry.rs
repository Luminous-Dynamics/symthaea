// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Build or verify a rights-aware Muse corpus registry snapshot.

use std::env;
use std::fs;
use std::io::{self, Write};
use std::process::ExitCode;

use symthaea_muse::corpus_registry::{
    CorpusArtifactRecord, CorpusRegistrySnapshot, build_registry_snapshot,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muse_corpus_registry: {error}");
            ExitCode::FAILURE
        }
    }
}

fn read_json<T: for<'de> serde::Deserialize<'de>>(path: &str) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {path}: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("parse {path}: {error}"))
}

fn write_output(bytes: &[u8], output: Option<&str>) -> Result<(), String> {
    match output {
        Some(path) => {
            fs::write(path, bytes).map_err(|error| format!("write {path}: {error}"))?;
            eprintln!("wrote registry snapshot to {path}");
        }
        None => {
            let stdout = io::stdout();
            let mut lock = stdout.lock();
            lock.write_all(bytes)
                .map_err(|error| format!("write stdout: {error}"))?;
            lock.flush().map_err(|error| format!("flush stdout: {error}"))?;
        }
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or_else(usage)?;
    if command == "--help" || command == "-h" {
        println!("{}", usage());
        return Ok(());
    }
    let mut input = None::<String>;
    let mut output = None::<String>;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" | "-i" => {
                let value = args.next().ok_or_else(|| format!("{arg} requires a path"))?;
                if input.replace(value).is_some() {
                    return Err("--input was supplied more than once".into());
                }
            }
            "--output" | "-o" => {
                let value = args.next().ok_or_else(|| format!("{arg} requires a path"))?;
                if output.replace(value).is_some() {
                    return Err("--output was supplied more than once".into());
                }
            }
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            other => return Err(format!("unknown argument {other:?}\n{}", usage())),
        }
    }

    let input = input.ok_or_else(|| "--input is required".to_string())?;
    match command.as_str() {
        "build" => {
            let artifacts: Vec<CorpusArtifactRecord> = read_json(&input)?;
            let snapshot = build_registry_snapshot(artifacts).map_err(|issues| {
                let details = issues
                    .iter()
                    .map(|issue| format!("{}: {}", issue.code, issue.message))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("registry validation failed:\n{details}")
            })?;
            if !snapshot.verify_digest() {
                return Err("newly built registry digest failed self-verification".into());
            }
            let mut bytes = serde_json::to_vec_pretty(&snapshot)
                .map_err(|error| format!("serialize registry: {error}"))?;
            bytes.push(b'\n');
            write_output(&bytes, output.as_deref())?;
            eprintln!(
                "validated {} artifacts; registry_sha256={}",
                snapshot.artifact_count,
                snapshot.registry_sha256
            );
        }
        "verify" => {
            if output.is_some() {
                return Err("--output is not supported with verify".into());
            }
            let snapshot: CorpusRegistrySnapshot = read_json(&input)?;
            if !snapshot.verify_digest() {
                return Err(
                    "registry snapshot schema, artifact count, record validation or digest failed"
                        .into(),
                );
            }
            println!(
                "valid registry snapshot: {} artifacts; sha256={}",
                snapshot.artifact_count,
                snapshot.registry_sha256
            );
        }
        other => {
            return Err(format!(
                "unknown command {other:?}\n{}",
                usage()
            ));
        }
    }
    Ok(())
}

fn usage() -> String {
    "Usage:\n  muse_corpus_registry build --input ARTIFACTS.json [--output SNAPSHOT.json]\n  muse_corpus_registry verify --input SNAPSHOT.json\n\nBuild input is a JSON array of CorpusArtifactRecord. Eligibility for a specific\nuse and jurisdiction is evaluated through the Rust API; registry registration\nalone does not authorize ingestion or training.".into()
}
