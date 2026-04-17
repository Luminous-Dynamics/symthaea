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
use holochain_client::{
    AdminWebsocket, AgentSigner, AppInfo, AppWebsocket, AuthorizeSigningCredentialsPayload,
    CellInfo, ClientAgentSigner, IssueAppAuthenticationTokenPayload, ZomeCallTarget,
};
use holochain_types::prelude::{CellId, ExternIO};
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

    /// Conductor admin WebSocket URL. Used to auto-issue an app-auth token
    /// when `MYCELIX_APP_TOKEN` is unset. Ignored when the env var is
    /// present (the env-supplied token takes precedence).
    #[arg(long, default_value = "ws://localhost:33800")]
    admin_url: String,

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
    // Mycelix governance enforces a state machine: proposals MUST be
    // created in Draft status, then transitioned via
    // `update_proposal_status` to Active (and beyond). Attempting to
    // create with Active fails validation. Callers that want a different
    // initial status (not currently legal) can supply `status` explicitly.
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

    // Always connect the admin websocket. We use it for two things:
    // 1. Auto-issue an app-auth token when MYCELIX_APP_TOKEN is unset.
    // 2. Authorize signing credentials after the app connects (so zome
    //    calls can pass Holochain's `Provenance not found` check).
    let admin_addr = cli
        .admin_url
        .strip_prefix("ws://")
        .unwrap_or(&cli.admin_url)
        .to_string();
    let admin = match AdminWebsocket::connect(admin_addr, None).await {
        Ok(a) => a,
        Err(e) => {
            let resp = BridgeResponse {
                request_id: None,
                ok: false,
                data: None,
                error: Some(format!("Admin connect failed: {e:?}")),
            };
            println!("{}", serde_json::to_string(&resp).unwrap());
            std::process::exit(1);
        }
    };

    // Token: env-supplied or auto-issued.
    let token: Vec<u8> = match std::env::var("MYCELIX_APP_TOKEN") {
        Ok(t) if !t.is_empty() => {
            eprintln!("Using MYCELIX_APP_TOKEN from environment.");
            t.into_bytes()
        }
        _ => {
            eprintln!(
                "MYCELIX_APP_TOKEN unset — auto-issuing via admin websocket at {}",
                cli.admin_url
            );
            let payload = IssueAppAuthenticationTokenPayload {
                installed_app_id: cli.app_id.clone().into(),
                expiry_seconds: 3600,
                single_use: false,
            };
            match admin.issue_app_auth_token(payload).await {
                Ok(issued) => {
                    eprintln!("Auto-issued token ({} bytes)", issued.token.len());
                    issued.token
                }
                Err(e) => {
                    let resp = BridgeResponse {
                        request_id: None,
                        ok: false,
                        data: None,
                        error: Some(format!("Token issue failed: {e:?}")),
                    };
                    println!("{}", serde_json::to_string(&resp).unwrap());
                    std::process::exit(1);
                }
            }
        }
    };

    // Keep a concrete handle on ClientAgentSigner so we can call
    // add_credentials after authorize_signing_credentials returns. The
    // trait-object handle is what AppWebsocket needs.
    let signer_concrete = Arc::new(ClientAgentSigner::default());
    let signer: Arc<dyn AgentSigner + Send + Sync> = signer_concrete.clone();

    eprintln!(
        "Connecting to conductor at {} (app: {}, role: {})",
        cli.conductor_url, cli.app_id, cli.role
    );

    // AppWebsocket takes a host:port address (ToSocketAddrs). Strip the
    // ws:// prefix so users can keep the conventional URL form in the CLI.
    let app_addr = cli
        .conductor_url
        .strip_prefix("ws://")
        .unwrap_or(&cli.conductor_url)
        .to_string();

    let ws = match AppWebsocket::connect(app_addr, token, signer, None).await {
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

    // Fetch app info so we can discover every provisioned cell and
    // authorize signing credentials for each. Without this, zome calls
    // fail with `SignZomeCallError("Provenance not found")` — the signer
    // has no keys registered for the target cell.
    let app_info: AppInfo = match ws.app_info().await {
        Ok(Some(info)) => info,
        Ok(None) => {
            eprintln!("app_info returned None — is {} installed?", cli.app_id);
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Failed to fetch app_info: {e:?}");
            std::process::exit(1);
        }
    };

    // Authorize signing credentials only for cells we actually dispatch to.
    // The default --role cell handles SubmitProposal/CastVote/QueryActive-
    // Proposals; `finance` is hardcoded in QueryTendBalance's dispatch.
    // Authorizing every cell in a large multi-role app (10+ cells) times out
    // the admin websocket.
    let target_roles: &[&str] = &[cli.role.as_str(), "finance"];
    let mut cells_authorized = 0usize;
    for (role_name, cells) in &app_info.cell_info {
        if !target_roles.contains(&role_name.as_str()) {
            continue;
        }
        for cell_info in cells {
            let cell_id: CellId = match cell_info {
                CellInfo::Provisioned(c) => c.cell_id.clone(),
                CellInfo::Cloned(c) => c.cell_id.clone(),
                _ => continue,
            };
            match admin
                .authorize_signing_credentials(AuthorizeSigningCredentialsPayload {
                    cell_id: cell_id.clone(),
                    functions: None,
                })
                .await
            {
                Ok(credentials) => {
                    signer_concrete.add_credentials(cell_id, credentials);
                    cells_authorized += 1;
                    eprintln!("Authorized signing for role={role_name}");
                }
                Err(e) => {
                    eprintln!(
                        "WARN: authorize_signing_credentials failed for role={role_name}: {e:?}"
                    );
                }
            }
        }
    }
    eprintln!("Authorized {cells_authorized} cell(s). Ready.");

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
        Command::SubmitProposal(mut input) => {
            // The proposals zome's integrity validation rejects proposals
            // with `voting_ends <= voting_starts`, and our wire protocol's
            // `#[serde(default)]` fields land as 0. Fill in sensible defaults
            // when the client didn't supply timestamps: voting opens now,
            // closes in 7 days. Explicit client values (non-zero) pass
            // through untouched.
            use std::time::{SystemTime, UNIX_EPOCH};
            let now_micros = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as i64;
            const SEVEN_DAYS_MICROS: i64 = 7 * 24 * 3600 * 1_000_000;
            if input.voting_starts == 0 {
                input.voting_starts = now_micros;
            }
            if input.voting_ends == 0 {
                input.voting_ends = input.voting_starts + SEVEN_DAYS_MICROS;
            }
            if input.created == 0 {
                input.created = now_micros;
            }
            if input.updated == 0 {
                input.updated = now_micros;
            }
            // `actions` must be valid JSON per the zome's integrity check;
            // empty string isn't. Substitute an empty array when the client
            // left it unset.
            if input.actions.trim().is_empty() {
                input.actions = "[]".to_string();
            }
            // Initial version must be 1 per the zome's integrity check.
            if input.version == 0 {
                input.version = 1;
            }
            call_zome(ws, role, "proposals", "create_proposal", &input).await
        }
        Command::CastVote(input) => call_zome(ws, role, "voting", "cast_vote", &input).await,
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
        Ok(result) => {
            // Zomes return a variety of shapes: JSON-compatible structs
            // (create_proposal → Record), opaque hashes (ActionHash →
            // Vec<u8>), tuples, etc. The common denominator is that
            // ExternIO wraps msgpack bytes. We try a decode cascade:
            //   1. serde_json::Value — works for simple structs/arrays
            //   2. Vec<u8> — works for hashes
            //   3. transcode raw msgpack → JSON — fallback for complex
            //      types like Record{signed_action, entry}
            // If all three fail, surface the raw bytes as hex so the
            // caller can still inspect them.
            match result.decode::<serde_json::Value>() {
                Ok(data) => BridgeResponse {
                    request_id: None,
                    ok: true,
                    data: Some(data),
                    error: None,
                },
                Err(_) => match result.decode::<Vec<u8>>() {
                    Ok(bytes) => BridgeResponse {
                        request_id: None,
                        ok: true,
                        data: Some(serde_json::Value::String(hex::encode(&bytes))),
                        error: None,
                    },
                    Err(_) => {
                        // Transcode msgpack → JSON via rmp_serde +
                        // serde_json. Handles Records and other maps
                        // containing holo-hashes (which become strings).
                        let raw_bytes = result.as_bytes();
                        match rmp_serde::from_slice::<rmpv::Value>(raw_bytes) {
                            Ok(v) => {
                                let json = msgpack_to_json(v);
                                BridgeResponse {
                                    request_id: None,
                                    ok: true,
                                    data: Some(json),
                                    error: None,
                                }
                            }
                            Err(e) => BridgeResponse {
                                request_id: None,
                                ok: true,
                                data: Some(serde_json::Value::String(hex::encode(raw_bytes))),
                                error: Some(format!(
                                    "Returned raw hex; msgpack decode failed: {e}"
                                )),
                            },
                        }
                    }
                },
            }
        }
        Err(e) => BridgeResponse {
            request_id: None,
            ok: false,
            data: None,
            error: Some(format!("Zome call {zome}::{fn_name} failed: {e:?}")),
        },
    }
}

/// Convert `rmpv::Value` (raw msgpack) to `serde_json::Value`. Holo-hash
/// binary payloads become hex strings so they round-trip through JSON.
fn msgpack_to_json(v: rmpv::Value) -> serde_json::Value {
    use rmpv::Value as MP;
    use serde_json::Value as J;
    match v {
        MP::Nil => J::Null,
        MP::Boolean(b) => J::Bool(b),
        MP::Integer(i) => {
            if let Some(u) = i.as_u64() {
                J::Number(u.into())
            } else if let Some(s) = i.as_i64() {
                J::Number(s.into())
            } else if let Some(f) = i.as_f64() {
                serde_json::Number::from_f64(f)
                    .map(J::Number)
                    .unwrap_or(J::Null)
            } else {
                J::Null
            }
        }
        MP::F32(f) => serde_json::Number::from_f64(f as f64)
            .map(J::Number)
            .unwrap_or(J::Null),
        MP::F64(f) => serde_json::Number::from_f64(f)
            .map(J::Number)
            .unwrap_or(J::Null),
        MP::String(s) => J::String(s.into_str().unwrap_or_default()),
        MP::Binary(b) => J::String(hex::encode(&b)),
        MP::Array(a) => J::Array(a.into_iter().map(msgpack_to_json).collect()),
        MP::Map(m) => J::Object(
            m.into_iter()
                .map(|(k, v)| {
                    let key = match k {
                        MP::String(s) => s.into_str().unwrap_or_default(),
                        other => format!("{other:?}"),
                    };
                    (key, msgpack_to_json(v))
                })
                .collect(),
        ),
        MP::Ext(_, b) => J::String(hex::encode(&b)),
    }
}
