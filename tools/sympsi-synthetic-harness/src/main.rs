use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::PathBuf;

use sympsi_synthetic_harness::run_profile;

fn main() {
    if let Err(error) = run() {
        eprintln!("sympsi-synthetic-harness: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut profile: Option<String> = None;
    let mut git_head: Option<String> = None;
    let mut manifest = PathBuf::from("tools/sympsi-synthetic-harness/fixtures/manifest-v0.1.json");
    let mut output: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            "--profile" => {
                profile = Some(next_value(&mut args, "--profile")?);
            }
            "--git-head" => {
                git_head = Some(next_value(&mut args, "--git-head")?);
            }
            "--manifest" => {
                manifest = PathBuf::from(next_value(&mut args, "--manifest")?);
            }
            "--out" => {
                output = Some(PathBuf::from(next_value(&mut args, "--out")?));
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    let profile = profile
        .ok_or_else(|| "missing required --profile four-choice|five-choice argument".to_string())?;
    let git_head = git_head
        .ok_or_else(|| "missing required --git-head exact-commit-sha argument".to_string())?;
    let manifest_bytes = std::fs::read(&manifest)
        .map_err(|error| format!("cannot read manifest {}: {error}", manifest.display()))?;

    let receipt = run_profile(&profile, &git_head, &manifest_bytes)?;
    let mut bytes = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("receipt JSON encoding failed: {error}"))?;
    bytes.push(b'\n');

    if let Some(path) = output {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| format!("cannot create receipt {}: {error}", path.display()))?;
        file.write_all(&bytes)
            .map_err(|error| format!("cannot write receipt {}: {error}", path.display()))?;
        file.sync_all()
            .map_err(|error| format!("cannot sync receipt {}: {error}", path.display()))?;
        println!("receipt_path={}", path.display());
        println!("receipt_sha256={}", receipt.receipt_sha256);
        println!(
            "all_controls_passed={}",
            receipt.receipt.all_controls_passed
        );
        if !receipt.receipt.all_controls_passed {
            return Err("one or more synthetic qualification controls failed".into());
        }
    } else {
        io::stdout()
            .write_all(&bytes)
            .map_err(|error| format!("cannot write receipt to stdout: {error}"))?;
        if !receipt.receipt.all_controls_passed {
            return Err("one or more synthetic qualification controls failed".into());
        }
    }
    Ok(())
}

fn next_value(args: &mut impl Iterator<Item = String>, argument: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value after {argument}"))
}

fn print_help() {
    println!(
        "Offline synthetic-only SYMPsi qualification harness\n\
         Required: --profile four-choice|five-choice --git-head <40-char-sha>\n\
         Optional: --manifest <path> --out <new-path>\n\
         This tool does not ingest human data or evaluate real psi/AIA results."
    );
}
