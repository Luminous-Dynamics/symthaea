// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::env;
use std::fs;
use std::process;

use symthaea_sat_evidence_graph_verifier::{compute_evidence_graph_root, verify_augmentation};

fn usage(program: &str) -> ! {
    eprintln!("Usage: {program} <augmentation.json>");
    eprintln!("       {program} --print-root <augmentation.json>");
    process::exit(64);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let program = args.first().map(String::as_str).unwrap_or("sat-evidence-graph-verify");
    let (print_root, path) = match args.as_slice() {
        [_, path] => (false, path.as_str()),
        [_, flag, path] if flag == "--print-root" => (true, path.as_str()),
        _ => usage(program),
    };

    let input = match fs::read_to_string(path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("cannot read {path}: {error}");
            process::exit(66);
        }
    };

    if print_root {
        match compute_evidence_graph_root(&input) {
            Ok(root) => {
                println!("{root}");
                process::exit(0);
            }
            Err(error) => {
                eprintln!("cannot compute graph root: {error}");
                process::exit(2);
            }
        }
    }

    match verify_augmentation(&input) {
        Ok(report) => {
            println!("{}", serde_json::to_string_pretty(&report).expect("report serialization cannot fail"));
            if report.verified {
                process::exit(0);
            }
            if !report.semantic_valid {
                process::exit(2);
            }
            process::exit(3);
        }
        Err(error) => {
            eprintln!("invalid input: {error}");
            process::exit(2);
        }
    }
}
