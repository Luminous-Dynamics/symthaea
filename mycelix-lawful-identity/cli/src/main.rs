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
#[cfg(feature = "conductor")]
mod live;
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

    /// Record intent to create a new legal DID (or `--live` to call
    /// the conductor directly).
    NewLegalDid {
        /// Optional human-readable label. Local-only; never disclosed.
        #[arg(long)]
        label: Option<String>,
        /// Make a live zome call instead of just staging locally.
        /// Requires the `conductor` build feature and a running
        /// conductor on ws://localhost:33800 (admin) + ws://localhost:8888 (app).
        #[arg(long)]
        live: bool,
    },

    /// List the legal DIDs the CLI knows about locally (or `--live`
    /// to query the conductor directly).
    ListDids {
        /// Query the running conductor instead of local state.
        #[arg(long)]
        live: bool,
    },

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
        /// Make a live zome call instead of just staging locally.
        #[arg(long)]
        live: bool,
    },

    /// Print the zome call sheet — the ready-to-paste conductor calls
    /// for everything the CLI currently stages locally.
    CallSheet,

    /// Ping the conductor's `legal_did.ping` zome as a liveness check.
    /// Requires the `conductor` build feature.
    Ping,

    /// Request a fresh nonce from `cross_did_zkp.request_nonce`. Always
    /// live — nonces cannot be staged. Requires the `conductor` build
    /// feature.
    RequestNonce {
        /// DID of the verifier requesting the nonce.
        verifier_did: String,
    },

    /// Look up the latest tier classification for an issuer DID on the
    /// running conductor. Requires the `conductor` build feature.
    LookupTier {
        /// Issuer DID to query (e.g. `did:web:state.gov`).
        issuer_did: String,
    },

    /// Attach an imported credential (passport, mDL, SSN-derived
    /// attestation) to a legal DID you own. The actual credential
    /// body is not transmitted — only a hash commitment + issuer
    /// pointer. Requires the `conductor` build feature.
    ImportCredential {
        /// Legal DID you own that will hold this credential.
        #[arg(long)]
        legal_did: String,
        /// BLAKE3/SHA-256 hash commitment to the underlying VC.
        #[arg(long)]
        credential_hash: String,
        /// Issuer DID (e.g. `did:web:home.affairs.gov.za`).
        #[arg(long)]
        issuer_did: String,
        /// Credential type (`PassportCredential`, `MobileDriversLicense`, etc.).
        #[arg(long)]
        credential_type: String,
        /// ISO 8601 issuance date.
        #[arg(long)]
        issued_at: String,
        /// Optional ISO 8601 expiry date.
        #[arg(long)]
        expires_at: Option<String>,
        /// Optional revocation-check URL.
        #[arg(long)]
        revocation_check_url: Option<String>,
    },

    /// List credentials attached to one of your legal DIDs.
    /// Requires the `conductor` build feature.
    ListCredentials {
        /// Legal DID to query.
        legal_did: String,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("lawful-id: {err}");
            let mut source = err.source();
            while let Some(cause) = source {
                eprintln!("  caused by: {cause}");
                source = cause.source();
            }
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

        Command::NewLegalDid { label, live } => {
            if live {
                do_live_new_legal_did(label.clone())?;
            } else {
                let entry = state::StagedDid {
                    label,
                    staged_at: chrono_like_iso()?,
                };
                state.staged_legal_dids.push(entry);
                state.save(&dirs.state_dir)?;
                println!(
                    "Legal DID intent staged. Run `lawful-id call-sheet` for the conductor \
                     call, or pass `--live` next time to call directly."
                );
            }
        }

        Command::ListDids { live } => {
            if live {
                do_live_list_dids()?;
            } else if state.staged_legal_dids.is_empty() {
                println!("No staged DIDs. Use `lawful-id new-legal-did` or `--live` to query.");
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
            live,
        } => {
            let tier_normalized = match tier.to_lowercase().as_str() {
                "sovereign" | "regulated" | "peer" => tier.to_lowercase(),
                other => {
                    return Err(
                        format!("unknown tier: {other}. Use sovereign/regulated/peer.").into(),
                    )
                }
            };
            if live {
                do_live_classify_issuer(&issuer_did, &tier_normalized, rationale.clone())?;
            } else {
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
                    "Issuer classification staged. Run `lawful-id call-sheet` for the conductor \
                     call, or pass `--live` next time to call directly."
                );
            }
        }

        Command::CallSheet => {
            print_call_sheet(&state);
        }

        Command::Ping => do_live_ping()?,

        Command::RequestNonce { verifier_did } => do_live_request_nonce(&verifier_did)?,

        Command::LookupTier { issuer_did } => do_live_lookup_tier(&issuer_did)?,

        Command::ImportCredential {
            legal_did,
            credential_hash,
            issuer_did,
            credential_type,
            issued_at,
            expires_at,
            revocation_check_url,
        } => do_live_import_credential(
            legal_did,
            credential_hash,
            issuer_did,
            credential_type,
            issued_at,
            expires_at,
            revocation_check_url,
        )?,

        Command::ListCredentials { legal_did } => do_live_list_credentials(&legal_did)?,
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

// ============================================================================
// Live conductor dispatchers
// ============================================================================

#[cfg(not(feature = "conductor"))]
fn live_unavailable() -> Result<(), Box<dyn std::error::Error>> {
    Err(
        "--live / ping commands require the `conductor` build feature: \
         cargo build --features conductor"
            .into(),
    )
}

#[cfg(not(feature = "conductor"))]
fn do_live_new_legal_did(_label: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    live_unavailable()
}

#[cfg(not(feature = "conductor"))]
fn do_live_list_dids() -> Result<(), Box<dyn std::error::Error>> {
    live_unavailable()
}

#[cfg(not(feature = "conductor"))]
fn do_live_classify_issuer(
    _issuer: &str,
    _tier: &str,
    _rationale: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    live_unavailable()
}

#[cfg(not(feature = "conductor"))]
fn do_live_ping() -> Result<(), Box<dyn std::error::Error>> {
    live_unavailable()
}

#[cfg(not(feature = "conductor"))]
fn do_live_request_nonce(_verifier: &str) -> Result<(), Box<dyn std::error::Error>> {
    live_unavailable()
}

#[cfg(not(feature = "conductor"))]
fn do_live_lookup_tier(_issuer: &str) -> Result<(), Box<dyn std::error::Error>> {
    live_unavailable()
}

#[cfg(not(feature = "conductor"))]
#[allow(clippy::too_many_arguments)]
fn do_live_import_credential(
    _legal_did: String,
    _credential_hash: String,
    _issuer_did: String,
    _credential_type: String,
    _issued_at: String,
    _expires_at: Option<String>,
    _revocation_check_url: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    live_unavailable()
}

#[cfg(not(feature = "conductor"))]
fn do_live_list_credentials(_legal_did: &str) -> Result<(), Box<dyn std::error::Error>> {
    live_unavailable()
}

#[cfg(feature = "conductor")]
fn do_live_ping() -> Result<(), Box<dyn std::error::Error>> {
    live_runtime().block_on(async {
        let conn = live::LiveConductor::connect(
            live::default_admin_addr(),
            live::default_app_addr(),
            live::DEFAULT_APP_ID,
        )
        .await?;
        let pong = conn.ping_legal_did().await?;
        println!("{pong}");
        Ok::<_, Box<dyn std::error::Error>>(())
    })
}

#[cfg(feature = "conductor")]
fn do_live_new_legal_did(label: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    live_runtime().block_on(async {
        let conn = live::LiveConductor::connect(
            live::default_admin_addr(),
            live::default_app_addr(),
            live::DEFAULT_APP_ID,
        )
        .await?;
        let out = conn.create_legal_did(label).await?;
        println!("Created legal DID: {}", out.did);
        Ok::<_, Box<dyn std::error::Error>>(())
    })
}

#[cfg(feature = "conductor")]
fn do_live_list_dids() -> Result<(), Box<dyn std::error::Error>> {
    live_runtime().block_on(async {
        let conn = live::LiveConductor::connect(
            live::default_admin_addr(),
            live::default_app_addr(),
            live::DEFAULT_APP_ID,
        )
        .await?;
        let dids = conn.list_my_legal_dids().await?;
        if dids.is_empty() {
            println!("No legal DIDs yet. Use `lawful-id new-legal-did --live`.");
        } else {
            for (idx, d) in dids.iter().enumerate() {
                println!(
                    "{}. {} label={} created_at={}",
                    idx + 1,
                    d.did,
                    d.label.as_deref().unwrap_or("(none)"),
                    d.created_at
                );
            }
        }
        Ok::<_, Box<dyn std::error::Error>>(())
    })
}

#[cfg(feature = "conductor")]
fn do_live_classify_issuer(
    issuer_did: &str,
    tier: &str,
    rationale: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let wire_tier = live::IssuerTierWire::parse(tier)
        .ok_or_else(|| format!("unknown tier: {tier}. Use sovereign/regulated/peer."))?;
    live_runtime().block_on(async {
        let conn = live::LiveConductor::connect(
            live::default_admin_addr(),
            live::default_app_addr(),
            live::DEFAULT_APP_ID,
        )
        .await?;
        let result = conn
            .classify_issuer(issuer_did, wire_tier, rationale)
            .await?;
        println!(
            "Classified {} as {}. Action: {}",
            issuer_did,
            wire_tier.as_str(),
            result.action_hash
        );
        Ok::<_, Box<dyn std::error::Error>>(())
    })
}

#[cfg(feature = "conductor")]
fn do_live_request_nonce(verifier_did: &str) -> Result<(), Box<dyn std::error::Error>> {
    live_runtime().block_on(async {
        let conn = live::LiveConductor::connect(
            live::default_admin_addr(),
            live::default_app_addr(),
            live::DEFAULT_APP_ID,
        )
        .await?;
        let result = conn.request_nonce(verifier_did).await?;
        println!(
            "Nonce issued for {verifier_did}:\n  nonce_b64 = {}\n  action    = {}",
            result.nonce_b64, result.action_hash
        );
        Ok::<_, Box<dyn std::error::Error>>(())
    })
}

#[cfg(feature = "conductor")]
fn do_live_lookup_tier(issuer_did: &str) -> Result<(), Box<dyn std::error::Error>> {
    live_runtime().block_on(async {
        let conn = live::LiveConductor::connect(
            live::default_admin_addr(),
            live::default_app_addr(),
            live::DEFAULT_APP_ID,
        )
        .await?;
        match conn.lookup_tier(issuer_did).await? {
            Some(view) => println!(
                "{} → {} (classified_at={}){}",
                view.issuer_did,
                view.tier,
                view.classified_at,
                match view.rationale {
                    Some(r) => format!("\n  rationale: {r}"),
                    None => String::new(),
                }
            ),
            None => println!("No classification found for {issuer_did} (defaults to Peer)."),
        }
        Ok::<_, Box<dyn std::error::Error>>(())
    })
}

#[cfg(feature = "conductor")]
#[allow(clippy::too_many_arguments)]
fn do_live_import_credential(
    legal_did: String,
    credential_hash: String,
    issuer_did: String,
    credential_type: String,
    issued_at: String,
    expires_at: Option<String>,
    revocation_check_url: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    live_runtime().block_on(async {
        let conn = live::LiveConductor::connect(
            live::default_admin_addr(),
            live::default_app_addr(),
            live::DEFAULT_APP_ID,
        )
        .await?;
        let out = conn
            .import_credential(live::ImportCredentialInput {
                legal_did,
                credential_hash: credential_hash.clone(),
                issuer_did,
                credential_type,
                issued_at,
                expires_at,
                revocation_check_url,
            })
            .await?;
        println!(
            "Imported credential (hash {}). Action: {}",
            out.credential_hash, out.record_action_hash
        );
        Ok::<_, Box<dyn std::error::Error>>(())
    })
}

#[cfg(feature = "conductor")]
fn do_live_list_credentials(legal_did: &str) -> Result<(), Box<dyn std::error::Error>> {
    live_runtime().block_on(async {
        let conn = live::LiveConductor::connect(
            live::default_admin_addr(),
            live::default_app_addr(),
            live::DEFAULT_APP_ID,
        )
        .await?;
        let creds = conn.get_credentials_for_did(legal_did).await?;
        if creds.is_empty() {
            println!("No credentials on {legal_did}.");
        } else {
            println!("{legal_did} — {} credential(s):", creds.len());
            for (i, c) in creds.iter().enumerate() {
                println!(
                    "  {}. {} from {} (issued {}){}{}",
                    i + 1,
                    c.credential_type,
                    c.issuer_did,
                    c.issued_at,
                    match &c.expires_at {
                        Some(e) => format!(", expires {e}"),
                        None => String::new(),
                    },
                    match &c.revocation_check_url {
                        Some(u) => format!("\n     revocation: {u}"),
                        None => String::new(),
                    },
                );
                println!("     hash: {}", c.credential_hash);
            }
        }
        Ok::<_, Box<dyn std::error::Error>>(())
    })
}

#[cfg(feature = "conductor")]
fn live_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime")
}
