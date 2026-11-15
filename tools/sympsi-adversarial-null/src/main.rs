use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::PathBuf;

use sympsi_adversarial_null::run_benchmark;

fn main() {
    if let Err(error) = run() {
        eprintln!("sympsi-adversarial-null: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut git_head: Option<String> = None;
    let mut manifest = PathBuf::from("tools/sympsi-adversarial-null/fixtures/manifest-v0.1.json");
    let mut output: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--help" | "-h" => {
                print_help();
                return Ok(());
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

    let git_head = git_head
        .ok_or_else(|| "missing required --git-head exact-commit-sha argument".to_string())?;
    let manifest_bytes = std::fs::read(&manifest)
        .map_err(|error| format!("cannot read manifest {}: {error}", manifest.display()))?;
    let envelope = run_benchmark(&git_head, &manifest_bytes)?;
    let mut bytes = serde_json::to_vec_pretty(&envelope)
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
        println!("receipt_sha256={}", envelope.receipt_sha256);
        println!(
            "all_controls_passed={}",
            envelope.receipt.all_controls_passed
        );
        if !envelope.receipt.all_controls_passed {
            return Err("one or more adversarial-null benchmark controls failed".into());
        }
    } else {
        io::stdout()
            .write_all(&bytes)
            .map_err(|error| format!("cannot write receipt to stdout: {error}"))?;
        if !envelope.receipt.all_controls_passed {
            return Err("one or more adversarial-null benchmark controls failed".into());
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
        "Offline synthetic adversarial-null benchmark\n\
         Required: --git-head <40-char-sha>\n\
         Optional: --manifest <path> --out <new-path>\n\
         No human data or real AIA/psi observations are accepted."
    );
}
