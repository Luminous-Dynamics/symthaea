// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use visual_display_corpus_verify::validate_corpus_str;

const DEFAULT_MANIFEST: &str = "docs/engineering/visual-display-workload-corpus-v0.1.json";
const DEFAULT_SCHEMA: &str = "docs/engineering/visual-display-workload-corpus.schema.json";

fn main() {
    let mut manifest_path = PathBuf::from(DEFAULT_MANIFEST);
    let mut schema_path = PathBuf::from(DEFAULT_SCHEMA);

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--manifest" => match args.next() {
                Some(path) => manifest_path = PathBuf::from(path),
                None => fail("--manifest requires a path"),
            },
            "--schema" => match args.next() {
                Some(path) => schema_path = PathBuf::from(path),
                None => fail("--schema requires a path"),
            },
            "--help" | "-h" => {
                print_help();
                return;
            }
            other => fail(&format!("unknown argument: {other}")),
        }
    }

    let manifest_text = read_file(&manifest_path);
    let schema_text = read_file(&schema_path);

    match validate_corpus_str(&manifest_text, &schema_text) {
        Ok(report) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report.to_json())
                    .expect("validation report must serialize")
            );
        }
        Err(errors) => {
            eprintln!("CORPUS_INVALID");
            for error in errors {
                eprintln!("- {error}");
            }
            std::process::exit(1);
        }
    }
}

fn read_file(path: &Path) -> String {
    match fs::read_to_string(path) {
        Ok(content) => content,
        Err(error) => fail(&format!("cannot read {}: {error}", path.display())),
    }
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(2);
}

fn print_help() {
    println!(
        "Validate VIS-DISP-CORPUS-001 without promoting it to a physical benchmark result.\n\
         Usage: visual-display-corpus-verify [--manifest PATH] [--schema PATH]\n\
         Defaults:\n\
           --manifest {DEFAULT_MANIFEST}\n\
           --schema   {DEFAULT_SCHEMA}"
    );
}
