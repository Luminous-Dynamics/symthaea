// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Standalone bridge daemon: reads governance dispatch commands from stdin
//! (one JSON object per line) and forwards them as zome calls to a live
//! Holochain conductor via AppWebsocket.
//!
//! Usage:
//!   echo '{"SubmitProposal":{"id":"MIP-001","title":"Test","description":"A test proposal","proposal_type":"Standard","author":"did:key:z6Mk..."}}' | mycelix-conductor-bridge
//!
//! Or pipe from Symthaea's governance dispatch channel.

use clap::Parser;
use holochain_client::{AgentSigner, AppWebsocket, ClientAgentSigner, ZomeCallTarget};
use holochain_types::prelude::ExternIO;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};

/// Bridge daemon connecting Symthaea governance dispatch to a Holochain conductor.
#[derive(Parser)]
#[command(name = "mycelix-conductor-bridge")]
struct Cli {
    /// Conductor app WebSocket URL
    #[arg(long, default_value = "ws://localhost:8888")]
    conductor_url: String,

    /// Installed app ID on the conductor
    #[arg(long, default_value = "mycelix-governance")]
    app_id: String,

    /// Role name in the hApp manifest
    #[arg(long, default_value = "governance")]
    role: String,
}

// ---------------------------------------------------------------------------
// Command types (mirror the zome extern signatures)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum Command {
    /// Maps to proposals coordinator: create_proposal(Proposal)
    SubmitProposal(ProposalInput),
    /// Maps to voting coordinator: cast_vote(Vote)
    CastVote(VoteInput),
    /// Maps to proposals coordinator: get_active_proposals(())
    QueryActiveProposals,
    /// Maps to finance/finance_bridge: query_tend_balance({member_did})
    QueryTendBalance { member_did: String },
}

/// Envelope carrying a correlation id alongside the command. Lets the client
/// (e.g. `symtropy-mycelix-bridge`) match responses to requests even if
/// execution order shuffles them. Uses `request_id` (not `id`) to avoid
/// colliding with the existing `id` field inside `ProposalInput` when the
/// command is serde-flattened.
#[derive(Debug, Deserialize)]
struct Request {
    #[serde(default)]
    request_id: Option<u64>,
    #[serde(flatten)]
    command: Command,
}

/// Matches the Proposal entry in proposals_integrity (field subset).
/// Fields the zome requires; optional fields default sensibly on-chain.
#[derive(Debug, Serialize, Deserialize)]
struct ProposalInput {
    id: String,
    title: String,
    description: String,
    #[serde(default = "default_proposal_type")]
    proposal_type: String,
    author: String,
    #[serde(default = "default_status")]
    status: String,
    #[serde(default)]
    actions: String,
    #[serde(default)]
    discussion_url: Option<String>,
    // Timestamps: conductor uses Timestamp (i64 micros). Default to 0 = "now" on-chain.
    #[serde(default)]
    voting_starts: i64,
    #[serde(default)]
    voting_ends: i64,
    #[serde(default)]
    created: i64,
    #[serde(default)]
    updated: i64,
    #[serde(default)]
    version: u32,
}

fn default_proposal_type() -> String {
    "Standard".into()
}
fn default_status() -> String {
    "Draft".into()
}

#[derive(Debug, Serialize, Deserialize)]
struct VoteInput {
    proposal_id: String,
    voter_did: String,
    approve: bool,
    #[serde(default)]
    rationale: String,
}

#[derive(Serialize)]
struct BridgeResponse {
    /// Echoed from the inbound [`Request::request_id`] when the envelope
    /// parsed successfully. `None` for framing errors before a request
    /// could be identified (malformed JSON, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<u64>,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Connect to conductor
    let token: Vec<u8> = std::env::var("MYCELIX_APP_TOKEN")
        .unwrap_or_default()
        .into_bytes();
    let signer: Arc<dyn AgentSigner + Send + Sync> = Arc::new(ClientAgentSigner::default());

    eprintln!(
        "Connecting to conductor at {} (app: {}, role: {})",
        cli.conductor_url, cli.app_id, cli.role
    );

    let ws = match AppWebsocket::connect(&cli.conductor_url, token, signer, None).await {
        Ok(ws) => {
            eprintln!("Connected.");
            ws
        }
        Err(e) => {
            let resp = BridgeResponse {
                request_id: None,
                ok: false,
                data: None,
                error: Some(format!("Failed to connect to conductor: {e:?}")),
            };
            println!("{}", serde_json::to_string(&resp).unwrap());
            std::process::exit(1);
        }
    };

    // Read commands from stdin, one JSON per line
    let stdin = tokio::io::stdin();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();

    while let Ok(Some(line)) = lines.next_line().await {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = BridgeResponse {
                    request_id: None,
                    ok: false,
                    data: None,
                    error: Some(format!("Invalid JSON: {e}")),
                };
                println!("{}", serde_json::to_string(&resp).unwrap());
                continue;
            }
        };

        let request_id = req.request_id;
        let mut result = dispatch(&ws, &cli.role, req.command).await;
        result.request_id = request_id;
        println!("{}", serde_json::to_string(&result).unwrap());
    }

    eprintln!("stdin closed, exiting.");
}

async fn dispatch(ws: &AppWebsocket, role: &str, cmd: Command) -> BridgeResponse {
    match cmd {
        Command::SubmitProposal(input) => {
            call_zome(ws, role, "proposals", "create_proposal", &input).await
        }
        Command::CastVote(input) => {
            call_zome(ws, role, "voting", "cast_vote", &input).await
        }
        Command::QueryActiveProposals => {
            call_zome(ws, role, "proposals", "get_active_proposals", &()).await
        }
        Command::QueryTendBalance { member_did } => {
            // TEND balances live on the finance role, not governance — ignore
            // the CLI default role for this command and call finance directly.
            call_zome(
                ws,
                "finance",
                "finance_bridge",
                "query_tend_balance",
                &serde_json::json!({ "member_did": member_did }),
            )
            .await
        }
    }
}

async fn call_zome<P: Serialize + std::fmt::Debug>(
    ws: &AppWebsocket,
    role: &str,
    zome: &str,
    fn_name: &str,
    payload: &P,
) -> BridgeResponse {
    let input = match ExternIO::encode(payload) {
        Ok(i) => i,
        Err(e) => {
            return BridgeResponse {
                request_id: None,
                ok: false,
                data: None,
                error: Some(format!("Encode failed: {e}")),
            };
        }
    };

    match ws
        .call_zome(
            ZomeCallTarget::RoleName(role.to_string().into()),
            zome.into(),
            fn_name.into(),
            input,
        )
        .await
    {
        Ok(result) => match result.decode::<serde_json::Value>() {
            Ok(data) => BridgeResponse {
                request_id: None,
                ok: true,
                data: Some(data),
                error: None,
            },
            Err(e) => BridgeResponse {
                request_id: None,
                ok: false,
                data: None,
                error: Some(format!("Decode response failed: {e}")),
            },
        },
        Err(e) => BridgeResponse {
            request_id: None,
            ok: false,
            data: None,
            error: Some(format!("Zome call {zome}::{fn_name} failed: {e:?}")),
        },
    }
}
