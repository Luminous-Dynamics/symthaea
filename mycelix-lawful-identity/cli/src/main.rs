// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! `lawful-id` — the command-line interface for `mycelix-lawful-identity`.
//!
//! First run surfaces the threat model in plain language. Subsequent
//! runs respect the user's acknowledgement and let them manage legal
//! DIDs, classify issuers, and generate cross-DID proofs.
//!
//! Conductor integration is deferred — today's CLI handles the
//! onboarding disclosure and local bookkeeping (which DIDs the user
//! intends to claim). The zome-round-trip commands print the exact
//! conductor call the user should run until the `holochain_client`
//! dep lands.

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

mod disclosure;
mod state;

#[derive(Parser, Debug)]
#[command(
    name = "lawful-id",
    about = "State-facing identity CLI for Mycelix — dual-DID lawful identity manager",
    version,
    long_about = None,
)]
struct Cli {
    /// Override the config directory (default: per-OS ProjectDirs).
    #[arg(long, global = true)]
    config_dir: Option<PathBuf>,

    /// Bypass the first-run threat-model disclosure. Use only after you
    /// have read `mycelix-lawful-identity/docs/THREAT_MODEL.md` at
    /// least once. Disclosure still runs; this only skips the
    /// acknowledgement pause.
    #[arg(long, global = true)]
    no_pause: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Initialize the CLI state directory and run the threat-model
    /// disclosure. Safe to run more than once.
    Init,

    /// Print the threat-model disclosure text on demand.
    Disclose,

    /// Print the user's current state (legal DIDs, issuer tiers, etc.).
    Status,

    /// Record intent to create a new legal DID. Emits the conductor
    /// call the user must run once the happ is installed.
    NewLegalDid {
        /// Optional human-readable label. Local-only; never disclosed.
        #[arg(long)]
        label: Option<String>,
    },

    /// List the legal DIDs the CLI knows about locally.
    ListDids,

    /// Classify an external issuer.
    ClassifyIssuer {
        /// The issuer DID (e.g. `did:web:state.gov`).
        issuer_did: String,
        /// Tier: `sovereign`, `regulated`, or `peer`.
        #[arg(long, default_value = "peer")]
        tier: String,
        /// Freeform rationale stored in the classification entry.
        #[arg(long)]
        rationale: Option<String>,
    },

    /// Print the zome call sheet — the ready-to-paste conductor calls
    /// for everything the CLI currently stages locally.
    CallSheet,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("lawful-id: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let dirs = resolve_dirs(cli.config_dir.as_deref())?;
    fs::create_dir_all(&dirs.state_dir)?;

    let mut state = state::CliState::load_or_default(&dirs.state_dir)?;

    // Always run the disclosure on first real command — but only once
    // unless `disclose` is explicitly requested.
    if matches!(cli.command, Command::Disclose) {
        disclosure::print_disclosure(false);
        return Ok(());
    }

    if !state.disclosure_acknowledged_at.is_some() && !matches!(cli.command, Command::Init) {
        disclosure::print_disclosure(true);
        eprintln!();
        eprintln!("Run `lawful-id init` once you have read the above to acknowledge.");
        return Err("first-run disclosure required — run `lawful-id init`".into());
    }

    match cli.command {
        Command::Init => {
            disclosure::print_disclosure(false);
            if !cli.no_pause {
                use std::io::{self, Write};
                print!(
                    "\nI have read the threat model and understand network + device risks \
                     are NOT mitigated [type \"ack\" to acknowledge]: ",
                );
                io::stdout().flush().ok();
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                if input.trim().eq_ignore_ascii_case("ack") {
                    state.disclosure_acknowledged_at = Some(chrono_like_iso()?);
                    state.save(&dirs.state_dir)?;
                    println!("Acknowledged. Welcome.");
                } else {
                    eprintln!("Not acknowledged. Re-run `lawful-id init` when ready.");
                    return Ok(());
                }
            } else {
                state.disclosure_acknowledged_at = Some(chrono_like_iso()?);
                state.save(&dirs.state_dir)?;
                println!("Disclosure bypass (--no-pause). Recorded acknowledgement.");
            }
        }

        Command::Disclose => {
            // handled above
            unreachable!("Disclose handled before match")
        }

        Command::Status => {
            println!(
                "Disclosure acknowledged: {}",
                match &state.disclosure_acknowledged_at {
                    Some(when) => when.as_str(),
                    None => "(no — run `lawful-id init`)",
                }
            );
            println!(
                "Legal DIDs staged locally: {}",
                state.staged_legal_dids.len()
            );
            for (idx, did) in state.staged_legal_dids.iter().enumerate() {
                println!(
                    "  {}. label={} staged_at={}",
                    idx + 1,
                    did.label.as_deref().unwrap_or("(none)"),
                    did.staged_at
                );
            }
            println!(
                "Classified issuers staged: {}",
                state.staged_issuer_classifications.len()
            );
            for c in &state.staged_issuer_classifications {
                println!("  - {} → {}", c.issuer_did, c.tier);
            }
        }

        Command::NewLegalDid { label } => {
            let entry = state::StagedDid {
                label,
                staged_at: chrono_like_iso()?,
            };
            state.staged_legal_dids.push(entry);
            state.save(&dirs.state_dir)?;
            println!("Legal DID intent staged. Run `lawful-id call-sheet` for the conductor call.");
        }

        Command::ListDids => {
            if state.staged_legal_dids.is_empty() {
                println!("No staged DIDs. Use `lawful-id new-legal-did`.");
            } else {
                for (idx, did) in state.staged_legal_dids.iter().enumerate() {
                    println!(
                        "{}. label={} staged_at={}",
                        idx + 1,
                        did.label.as_deref().unwrap_or("(none)"),
                        did.staged_at
                    );
                }
            }
        }

        Command::ClassifyIssuer {
            issuer_did,
            tier,
            rationale,
        } => {
            let tier_normalized = match tier.to_lowercase().as_str() {
                "sovereign" | "regulated" | "peer" => tier.to_lowercase(),
                other => {
                    return Err(
                        format!("unknown tier: {other}. Use sovereign/regulated/peer.").into(),
                    )
                }
            };
            state
                .staged_issuer_classifications
                .push(state::StagedIssuerClassification {
                    issuer_did,
                    tier: tier_normalized,
                    rationale,
                    staged_at: chrono_like_iso()?,
                });
            state.save(&dirs.state_dir)?;
            println!(
                "Issuer classification staged. Run `lawful-id call-sheet` for the conductor call."
            );
        }

        Command::CallSheet => {
            print_call_sheet(&state);
        }
    }

    Ok(())
}

struct Dirs {
    state_dir: PathBuf,
}

fn resolve_dirs(
    override_path: Option<&std::path::Path>,
) -> Result<Dirs, Box<dyn std::error::Error>> {
    if let Some(p) = override_path {
        return Ok(Dirs {
            state_dir: p.to_path_buf(),
        });
    }
    let pd = ProjectDirs::from("net", "Mycelix", "lawful-id")
        .ok_or("could not resolve project dirs on this OS")?;
    Ok(Dirs {
        state_dir: pd.data_local_dir().to_path_buf(),
    })
}

fn chrono_like_iso() -> Result<String, Box<dyn std::error::Error>> {
    // Lightweight ISO-8601 without pulling in chrono. Good enough for
    // local-only audit records the user reads.
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now().duration_since(UNIX_EPOCH)?;
    let secs = d.as_secs();
    Ok(format!("{secs}.{:09}Z", d.subsec_nanos()))
}

fn print_call_sheet(state: &state::CliState) {
    println!("# ─────────────────────────────────────────────");
    println!("# Conductor call sheet — run these via `hc` or");
    println!("# via the holochain_client SDK. The CLI records");
    println!("# intent locally; a future release will call the");
    println!("# conductor directly.");
    println!("# ─────────────────────────────────────────────");
    for entry in &state.staged_legal_dids {
        println!();
        println!("# Create legal DID (label = {:?})", entry.label);
        println!("hc sandbox call --running=<admin-port> zome-call \\");
        println!("  --app-id mycelix-lawful-identity \\");
        println!("  --zome legal_did \\");
        println!("  --fn create_legal_did \\");
        println!(
            "  --payload '{}'",
            serde_json::json!({ "label": entry.label })
        );
    }
    for entry in &state.staged_issuer_classifications {
        println!();
        println!("# Classify issuer {}", entry.issuer_did);
        println!("hc sandbox call --running=<admin-port> zome-call \\");
        println!("  --app-id mycelix-lawful-identity \\");
        println!("  --zome issuer_trust_tier \\");
        println!("  --fn classify_issuer \\");
        println!(
            "  --payload '{}'",
            serde_json::json!({
                "issuer_did": entry.issuer_did,
                "tier": match entry.tier.as_str() {
                    "sovereign" => "Sovereign",
                    "regulated" => "RegulatedIntermediary",
                    _ => "Peer",
                },
                "rationale": entry.rationale,
            })
        );
    }
    if state.staged_legal_dids.is_empty() && state.staged_issuer_classifications.is_empty() {
        println!();
        println!("# Nothing staged. Use:");
        println!("#   lawful-id new-legal-did --label \"My passport holder\"");
        println!("#   lawful-id classify-issuer did:web:state.gov --tier sovereign");
    }
}

// Re-exports for tests in submodules.
#[derive(Serialize, Deserialize)]
#[allow(dead_code)]
struct _KeepSerdeLinked;
