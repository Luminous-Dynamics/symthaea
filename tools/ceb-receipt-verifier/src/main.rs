use clap::{Parser, Subcommand};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ceb_receipt_verifier::{
    AgentViewInputs, PairInputs, TREE_DIGEST_ALGORITHM, hash_source_tree, verify_agent_view,
    verify_pair,
};

#[derive(Debug, Parser)]
#[command(
    name = "ceb-receipt-verifier",
    version,
    about = "Offline verifier for CEB source inventories and signed receipt pairs"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Compute a CEB-TREE-V1 SHA-256 for a clean source snapshot.
    HashTree {
        #[arg(long)]
        directory: PathBuf,
    },
    /// Verify a public-development source-only agent-view inventory against out-of-band pins.
    VerifyAgentView {
        #[arg(long)]
        directory: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        task_id: String,
        /// Out-of-band SHA-256 pin for the exact manifest bytes.
        #[arg(long)]
        expected_manifest_sha256: String,
        /// Out-of-band CEB-TREE-V1 SHA-256 pin for the packaged directory.
        #[arg(long)]
        expected_tree_sha256: String,
    },
    /// Verify one primary DSSE receipt and its independently signed pristine regrade.
    VerifyPair {
        #[arg(long)]
        task_contract: PathBuf,
        #[arg(long)]
        task_id: String,
        /// Out-of-band SHA-256 pin for the exact task-contract bytes.
        #[arg(long)]
        expected_task_contract_sha256: String,
        /// Clean source snapshot directory to hash with CEB-TREE-V1.
        #[arg(long)]
        source_tree_dir: PathBuf,
        /// Candidate result snapshot directory to hash with CEB-TREE-V1.
        #[arg(long)]
        result_tree_dir: PathBuf,
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
        return Err(format!(
            "{} exceeds the {limit}-byte input limit",
            path.display()
        ));
    }
    let file =
        File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    if bytes.len() as u64 > limit {
        return Err(format!(
            "{} exceeds the {limit}-byte input limit",
            path.display()
        ));
    }
    Ok(bytes)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let (
        task_contract,
        task_id,
        expected_task_contract_sha256,
        source_tree_dir,
        result_tree_dir,
        patch,
        primary,
        primary_artifacts_dir,
        regrade,
        regrade_artifacts_dir,
        trust_policy,
        expected_trust_policy_sha256,
    ) = match cli.command {
        Command::HashTree { directory } => {
            return match hash_source_tree(&directory) {
                Ok(sha256) => {
                    println!("algorithm={TREE_DIGEST_ALGORITHM}");
                    println!("sha256={sha256}");
                    ExitCode::SUCCESS
                }
                Err(reason) => {
                    eprintln!("tree hashing unresolved: {reason}");
                    ExitCode::from(2)
                }
            };
        }
        Command::VerifyAgentView {
            directory,
            manifest,
            task_id,
            expected_manifest_sha256,
            expected_tree_sha256,
        } => {
            let manifest_bytes = match read_limited(&manifest, 8 * 1024 * 1024) {
                Ok(bytes) => bytes,
                Err(reason) => {
                    let report = serde_json::json!({
                        "report_version": 1,
                        "verifier_version": env!("CARGO_PKG_VERSION"),
                        "status": "protocol_violation",
                        "accepted": false,
                        "task_id": null,
                        "manifest_sha256": null,
                        "tree_sha256": null,
                        "entry_count": 0,
                        "reasons": [reason],
                        "claim_ceiling": "No agent-view inventory acceptance claim; manifest could not be read safely."
                    });
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&report)
                            .unwrap_or_else(|_| "{\"status\":\"protocol_violation\"}".to_owned())
                    );
                    return ExitCode::from(2);
                }
            };
            let report = verify_agent_view(AgentViewInputs {
                snapshot_dir: directory,
                manifest: manifest_bytes,
                expected_manifest_sha256,
                expected_tree_sha256,
                expected_task_id: task_id,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .unwrap_or_else(|_| "{\"status\":\"protocol_violation\"}".to_owned())
            );
            return if report.accepted {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(2)
            };
        }
        Command::VerifyPair {
            task_contract,
            task_id,
            expected_task_contract_sha256,
            source_tree_dir,
            result_tree_dir,
            patch,
            primary,
            primary_artifacts_dir,
            regrade,
            regrade_artifacts_dir,
            trust_policy,
            expected_trust_policy_sha256,
        } => (
            task_contract,
            task_id,
            expected_task_contract_sha256,
            source_tree_dir,
            result_tree_dir,
            patch,
            primary,
            primary_artifacts_dir,
            regrade,
            regrade_artifacts_dir,
            trust_policy,
            expected_trust_policy_sha256,
        ),
    };

    let loaded = (|| -> Result<PairInputs, String> {
        Ok(PairInputs {
            task_contract: read_limited(&task_contract, 8 * 1024 * 1024)?,
            expected_task_contract_sha256,
            task_id,
            source_tree_dir,
            result_tree_dir,
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
            println!(
                "{}",
                serde_json::to_string_pretty(&message)
                    .unwrap_or_else(|_| "{\"status\":\"protocol_violation\"}".to_owned())
            );
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
