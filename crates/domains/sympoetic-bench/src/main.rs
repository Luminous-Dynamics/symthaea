use std::{fs, path::PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use serde::Serialize;
use sympoetic_bench::{
    Scorecard, Trace,
    trajectory::{TeamTrajectory, validate_trajectory},
    validate::{sha256_hex, validate_corpus, validate_scorecard, validate_trace},
};

#[derive(Debug, Parser)]
#[command(
    name = "sympoetic-bench",
    about = "Validate SympoeticBench scenarios, traces, and evidence scorecards"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Parse and validate every scenario in a JSONL corpus.
    ValidateCorpus {
        #[arg(long)]
        input: PathBuf,
    },
    /// Validate a session trace, including causal order and unknown tool outcomes.
    ValidateTrace {
        #[arg(long)]
        input: PathBuf,
    },
    /// Validate score ranges, provenance digests, and non-compensatory hard gates.
    ValidateScorecard {
        #[arg(long)]
        input: PathBuf,
    },
    /// Validate continuity, roster changes, consent, and digests across team cycles.
    ValidateTrajectory {
        #[arg(long)]
        input: PathBuf,
    },
    /// Print the SHA-256 digest of an artifact without changing it.
    Digest {
        #[arg(long)]
        input: PathBuf,
    },
}

#[derive(Serialize)]
struct ScorecardReceipt<'a> {
    status: &'static str,
    input_sha256: String,
    run_id: &'a str,
    scenario_id: &'a str,
    hard_gates_pass: bool,
    metric_evidence_count: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::ValidateCorpus { input } => {
            let bytes = fs::read(&input)
                .with_context(|| format!("could not read {}", input.display()))?;
            let (_, receipt) = validate_corpus(&bytes)?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }
        Command::ValidateTrace { input } => {
            let bytes = fs::read(&input)
                .with_context(|| format!("could not read {}", input.display()))?;
            let trace: Trace = serde_json::from_slice(&bytes)
                .with_context(|| {
                    format!("{} is not a valid trace JSON document", input.display())
                })?;
            let receipt = validate_trace(&trace, &bytes)?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
            if receipt.unresolved_unknown_outcome_count != 0 {
                bail!(
                    "trace is incomplete: an unknown tool outcome remains unreconciled"
                );
            }
        }
        Command::ValidateScorecard { input } => {
            let bytes = fs::read(&input)
                .with_context(|| format!("could not read {}", input.display()))?;
            let scorecard: Scorecard = serde_json::from_slice(&bytes)
                .with_context(|| {
                    format!("{} is not a valid scorecard JSON document", input.display())
                })?;
            validate_scorecard(&scorecard)?;
            let receipt = ScorecardReceipt {
                status: "pass",
                input_sha256: sha256_hex(&bytes),
                run_id: &scorecard.run_id,
                scenario_id: &scorecard.scenario_id,
                hard_gates_pass: scorecard.hard_gates.pass,
                metric_evidence_count: scorecard.metric_evidence.len(),
            };
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }
        Command::ValidateTrajectory { input } => {
            let bytes = fs::read(&input)
                .with_context(|| format!("could not read {}", input.display()))?;
            let trajectory: TeamTrajectory = serde_json::from_slice(&bytes)
                .with_context(|| {
                    format!("{} is not a valid team trajectory JSON document", input.display())
                })?;
            let receipt = validate_trajectory(&trajectory, &bytes)?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }
        Command::Digest { input } => {
            let bytes = fs::read(&input)
                .with_context(|| format!("could not read {}", input.display()))?;
            println!("{}", sha256_hex(&bytes));
        }
    }
    Ok(())
}
