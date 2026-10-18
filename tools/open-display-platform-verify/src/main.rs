// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use open_display_platform_verify::{validate_measurement_run_str, validate_platform_str};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_MANIFEST: &str = "docs/engineering/open-display-reference-platform-v0.1.json";
const DEFAULT_SCHEMA: &str = "docs/engineering/open-display-reference-platform.schema.json";
const DEFAULT_RUN_RECORD: &str = "docs/engineering/display-measurement-run-template-v0.1.json";
const DEFAULT_RUN_SCHEMA: &str = "docs/engineering/display-measurement-run.schema.json";

fn main() {
    let mut manifest_path = PathBuf::from(DEFAULT_MANIFEST);
    let mut schema_path = PathBuf::from(DEFAULT_SCHEMA);
    let mut run_record_path = PathBuf::from(DEFAULT_RUN_RECORD);
    let mut run_schema_path = PathBuf::from(DEFAULT_RUN_SCHEMA);
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
            "--run-record" => match args.next() {
                Some(path) => run_record_path = PathBuf::from(path),
                None => fail("--run-record requires a path"),
            },
            "--run-schema" => match args.next() {
                Some(path) => run_schema_path = PathBuf::from(path),
                None => fail("--run-schema requires a path"),
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
    let run_record_text = read_file(&run_record_path);
    let run_schema_text = read_file(&run_schema_path);

    let platform_report = match validate_platform_str(&manifest_text, &schema_text) {
        Ok(report) => report,
        Err(errors) => fail_with_errors("PLATFORM_CONTRACT_INVALID", errors),
    };
    let run_report = match validate_measurement_run_str(&run_record_text, &run_schema_text) {
        Ok(report) => report,
        Err(errors) => fail_with_errors("MEASUREMENT_RUN_CONTRACT_INVALID", errors),
    };

    let report = serde_json::json!({
        "platform": platform_report.to_json(),
        "measurement_run": run_report.to_json(),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("combined validation report must serialize")
    );
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

fn fail_with_errors(prefix: &str, errors: Vec<String>) -> ! {
    eprintln!("{prefix}");
    for error in errors {
        eprintln!("- {error}");
    }
    std::process::exit(1);
}

fn print_help() {
    println!(
        "Validate the ODRH-001 design and run-record contracts without claiming physical qualification.\n\
         Usage: open-display-platform-verify [--manifest PATH] [--schema PATH] [--run-record PATH] [--run-schema PATH]\n\
         Defaults:\n\
           --manifest   {DEFAULT_MANIFEST}\n\
           --schema     {DEFAULT_SCHEMA}\n\
           --run-record {DEFAULT_RUN_RECORD}\n\
           --run-schema {DEFAULT_RUN_SCHEMA}"
    );
}
