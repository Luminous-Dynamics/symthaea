use clap::{Parser, Subcommand};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ceb_receipt_verifier::{verify_pair, PairInputs};

#[derive(Debug, Parser)]
#[command(name = "ceb-receipt-verifier", version, about = "Offline verifier for signed CEB primary/regrade receipt pairs")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Verify one primary DSSE receipt and its independently signed pristine regrade.
    VerifyPair {
        #[arg(long)]
        task_contract: PathBuf,
        #[arg(long)]
        task_id: String,
        /// Expected source-tree SHA-256, lowercase hex without a prefix.
        #[arg(long)]
        source_tree_sha256: String,
        #[arg(long)]
        patch: PathBuf,
        #[arg(long)]
        primary: PathBuf,
        #[arg(long)]
        primary_artifacts_dir: PathBuf,
        #[arg(long)]
        regrade: PathBuf,
        #[arg(long)]
        regrade_artifacts_dir: PathBuf,
        #[arg(long)]
        trust_policy: PathBuf,
        /// Out-of-band trust-policy SHA-256, lowercase hex without a prefix.
        #[arg(long)]
        expected_trust_policy_sha256: String,
    },
}

fn read_limited(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let metadata = std::fs::metadata(path)
        .map_err(|error| format!("cannot stat {}: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!("{} is not a regular file", path.display()));
    }
    if metadata.len() > limit {
        return Err(format!("{} exceeds the {limit}-byte input limit", path.display()));
    }
    let file = File::open(path)
        .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    if bytes.len() as u64 > limit {
        return Err(format!("{} exceeds the {limit}-byte input limit", path.display()));
    }
    Ok(bytes)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let Command::VerifyPair {
        task_contract,
        task_id,
        source_tree_sha256,
        patch,
        primary,
        primary_artifacts_dir,
        regrade,
        regrade_artifacts_dir,
        trust_policy,
        expected_trust_policy_sha256,
    } = cli.command;

    let loaded = (|| -> Result<PairInputs, String> {
        Ok(PairInputs {
            task_contract: read_limited(&task_contract, 8 * 1024 * 1024)?,
            task_id,
            source_tree_sha256,
            patch: read_limited(&patch, 32 * 1024 * 1024)?,
            primary_envelope: read_limited(&primary, 8 * 1024 * 1024)?,
            primary_artifacts_dir,
            regrade_envelope: read_limited(&regrade, 8 * 1024 * 1024)?,
            regrade_artifacts_dir,
            trust_policy: read_limited(&trust_policy, 8 * 1024 * 1024)?,
            expected_trust_policy_sha256,
        })
    })();

    let report = match loaded {
        Ok(inputs) => verify_pair(inputs),
        Err(reason) => {
            let message = serde_json::json!({
                "report_version": 1,
                "verifier_version": env!("CARGO_PKG_VERSION"),
                "status": "protocol_violation",
                "accepted": false,
                "reasons": [reason],
                "claim_ceiling": "No acceptance claim; one or more verifier inputs could not be read safely."
            });
            println!("{}", serde_json::to_string_pretty(&message).unwrap_or_else(|_| "{\"status\":\"protocol_violation\"}".to_owned()));
            return ExitCode::from(2);
        }
    };

    match serde_json::to_string_pretty(&report) {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("cannot serialize verification report: {error}");
            return ExitCode::from(2);
        }
    }

    if report.accepted {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    }
}
