// Copyright (c) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Subprocess IPC layer.
//!
//! We spawn `mycelix-conductor-bridge` as a child process and exchange JSON
//! messages over its stdin / stdout. See module-level docs on [`crate`] for
//! the "why subprocess, not in-process" rationale.
//!
//! ## Protocol (Milestone 2)
//!
//! Every request carries a `request_id` (u64, minted monotonically by the
//! writer loop). Responses echo the same `request_id` so the reader loop can
//! correlate them even if the conductor reorders execution.
//!
//! **Request** (one JSON per line, written to subprocess stdin):
//! ```json
//! {"request_id": 0, "type": "QueryActiveProposals"}
//! {"request_id": 1, "type": "SubmitProposal", "id": "MIP-042", "title": "...",
//!  "description": "...", "author": "did:key:z6Mk..."}
//! {"request_id": 2, "type": "CastVote", "proposal_id": "MIP-042",
//!  "voter_did": "did:key:z6Mk...", "approve": true, "rationale": ""}
//! {"request_id": 3, "type": "QueryTendBalance", "member_did": "did:key:z6Mk..."}
//! ```
//!
//! **Response** (one JSON per line, read from subprocess stdout):
//! ```json
//! {"request_id": 0, "ok": true, "data": [...]}
//! {"request_id": 1, "ok": false, "error": "..."}
//! ```

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use bevy::prelude::*;
use bevy_tokio_tasks::TokioTasksRuntime;
use flume::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, Command};
use tokio::sync::Mutex;

use crate::config::MycelixConfig;
use crate::events::{MycelixRequest, MycelixResponse};
use crate::resource::{MycelixRequestOutbox, MycelixResponseInbox};

// ---------------------------------------------------------------------------
// Wire protocol types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct WireRequest {
    request_id: u64,
    #[serde(flatten)]
    command: WireCommand,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum WireCommand {
    QueryActiveProposals,
    SubmitProposal(WireProposalInput),
    CastVote(WireVoteInput),
    QueryTendBalance { member_did: String },
    GetProposal { proposal_id: String },
}

#[derive(Debug, Serialize)]
struct WireProposalInput {
    id: String,
    title: String,
    description: String,
    author: String,
}

#[derive(Debug, Serialize)]
struct WireVoteInput {
    proposal_id: String,
    voter_did: String,
    approve: bool,
    rationale: String,
}

#[derive(Debug, Deserialize)]
struct WireResponse {
    #[serde(default)]
    request_id: Option<u64>,
    ok: bool,
    #[serde(default)]
    data: Option<serde_json::Value>,
    #[serde(default)]
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// Correlation metadata
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum PendingKind {
    GetActiveProposals,
    ProposalSubmitted,
    VoteCast { proposal_id: String },
    TendBalance,
    Proposal { proposal_id: String },
}

#[derive(Debug)]
struct Pending {
    requester: Entity,
    kind: PendingKind,
}

type PendingMap = Arc<Mutex<HashMap<u64, Pending>>>;

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Startup system: spawn the subprocess + background IPC task.
pub(crate) fn spawn_dispatcher_task(
    runtime: Res<TokioTasksRuntime>,
    config: Res<MycelixConfig>,
    mut outbox: ResMut<MycelixRequestOutbox>,
) {
    let Some(req_rx) = outbox.rx.take() else {
        warn!("symtropy-mycelix-bridge: dispatcher task already started; skipping");
        return;
    };
    let Some(resp_tx) = outbox.response_tx.take() else {
        warn!("symtropy-mycelix-bridge: response sender missing; skipping");
        return;
    };

    let config = config.clone();

    runtime.spawn_background_task(move |_ctx| async move {
        if let Err(err) = run_dispatcher_loop(config, req_rx, resp_tx).await {
            error!(
                ?err,
                "symtropy-mycelix-bridge: dispatcher loop exited with error"
            );
        }
    });
}

/// Update-schedule system: drain the inbox into Bevy [`MycelixResponse`]
/// messages.
pub(crate) fn pump_responses(
    inbox: Res<MycelixResponseInbox>,
    mut writer: MessageWriter<MycelixResponse>,
) {
    for response in inbox.rx.try_iter() {
        writer.write(response);
    }
}

// ---------------------------------------------------------------------------
// Dispatcher loop (runs inside the tokio runtime)
// ---------------------------------------------------------------------------

async fn run_dispatcher_loop(
    config: MycelixConfig,
    req_rx: Receiver<MycelixRequest>,
    resp_tx: Sender<MycelixResponse>,
) -> Result<(), DispatcherError> {
    info!(
        binary = %config.bridge_binary.display(),
        conductor_url = %config.conductor_url,
        app_id = %config.app_id,
        role = %config.role,
        "symtropy-mycelix-bridge: spawning subprocess"
    );

    let mut child = match Command::new(&config.bridge_binary)
        .arg("--conductor-url")
        .arg(&config.conductor_url)
        .arg("--app-id")
        .arg(&config.app_id)
        .arg("--role")
        .arg(&config.role)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let reason = format!("spawn {:?}: {e}", config.bridge_binary);
            error!(%reason, "symtropy-mycelix-bridge: spawn failed");
            drain_with_error(req_rx, resp_tx, reason).await;
            return Err(DispatcherError::Spawn {
                path: config.bridge_binary.to_string_lossy().to_string(),
                source: e,
            });
        }
    };

    let stdin = child.stdin.take().ok_or(DispatcherError::MissingStdin)?;
    let stdout = child.stdout.take().ok_or(DispatcherError::MissingStdout)?;

    let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));
    let next_id = Arc::new(AtomicU64::new(0));

    let writer_task = {
        let pending = pending.clone();
        let next_id = next_id.clone();
        tokio::spawn(async move { writer_loop(stdin, req_rx, pending, next_id).await })
    };
    let reader_task = {
        let pending = pending.clone();
        tokio::spawn(async move { reader_loop(stdout, resp_tx, pending).await })
    };

    tokio::select! {
        res = writer_task => res.map_err(DispatcherError::Join)??,
        res = reader_task => res.map_err(DispatcherError::Join)??,
    };

    Ok(())
}

async fn drain_with_error(
    req_rx: Receiver<MycelixRequest>,
    resp_tx: Sender<MycelixResponse>,
    reason: String,
) {
    while let Ok(req) = req_rx.recv_async().await {
        let requester = req.requester();
        if resp_tx
            .send_async(MycelixResponse::Error {
                requester,
                reason: reason.clone(),
            })
            .await
            .is_err()
        {
            return;
        }
    }
}

async fn writer_loop(
    mut stdin: ChildStdin,
    req_rx: Receiver<MycelixRequest>,
    pending: PendingMap,
    next_id: Arc<AtomicU64>,
) -> Result<(), DispatcherError> {
    while let Ok(req) = req_rx.recv_async().await {
        let request_id = next_id.fetch_add(1, Ordering::SeqCst);
        let requester = req.requester();

        let (kind, command) = match req {
            MycelixRequest::GetActiveProposals { .. } => (
                PendingKind::GetActiveProposals,
                WireCommand::QueryActiveProposals,
            ),
            MycelixRequest::SubmitProposal {
                proposal_id,
                title,
                description,
                author_did,
                ..
            } => (
                PendingKind::ProposalSubmitted,
                WireCommand::SubmitProposal(WireProposalInput {
                    id: proposal_id,
                    title,
                    description,
                    author: author_did,
                }),
            ),
            MycelixRequest::CastVote {
                proposal_id,
                voter_did,
                approve,
                rationale,
                ..
            } => (
                PendingKind::VoteCast {
                    proposal_id: proposal_id.clone(),
                },
                WireCommand::CastVote(WireVoteInput {
                    proposal_id,
                    voter_did,
                    approve,
                    rationale,
                }),
            ),
            MycelixRequest::QueryTendBalance { member_did, .. } => (
                PendingKind::TendBalance,
                WireCommand::QueryTendBalance { member_did },
            ),
            MycelixRequest::GetProposal { proposal_id, .. } => (
                PendingKind::Proposal {
                    proposal_id: proposal_id.clone(),
                },
                WireCommand::GetProposal { proposal_id },
            ),
        };

        {
            let mut p = pending.lock().await;
            p.insert(request_id, Pending { requester, kind });
        }

        let wire = WireRequest {
            request_id,
            command,
        };
        let mut line = serde_json::to_string(&wire).map_err(DispatcherError::Serialise)?;
        line.push('\n');

        stdin
            .write_all(line.as_bytes())
            .await
            .map_err(DispatcherError::Stdin)?;
        stdin.flush().await.map_err(DispatcherError::Stdin)?;

        trace!(%request_id, %requester, "dispatched request to subprocess");
    }

    drop(stdin);
    Ok(())
}

async fn reader_loop<R>(
    stdout: R,
    resp_tx: Sender<MycelixResponse>,
    pending: PendingMap,
) -> Result<(), DispatcherError>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut lines = BufReader::new(stdout).lines();

    while let Some(line) = lines.next_line().await.map_err(DispatcherError::Stdout)? {
        let wire: WireResponse = match serde_json::from_str(&line) {
            Ok(w) => w,
            Err(e) => {
                warn!(line = %line, error = %e, "invalid JSON from subprocess");
                continue;
            }
        };

        let response = match wire.request_id {
            Some(id) => translate(id, wire, &pending).await,
            None => MycelixResponse::Error {
                requester: Entity::PLACEHOLDER,
                reason: wire
                    .error
                    .unwrap_or_else(|| "bridge error (no request_id)".to_string()),
            },
        };

        if resp_tx.send_async(response).await.is_err() {
            info!("symtropy-mycelix-bridge: inbox closed; reader exiting");
            return Ok(());
        }
    }

    Ok(())
}

async fn translate(id: u64, wire: WireResponse, pending: &PendingMap) -> MycelixResponse {
    let pending_entry = { pending.lock().await.remove(&id) };
    let Some(Pending { requester, kind }) = pending_entry else {
        return MycelixResponse::Error {
            requester: Entity::PLACEHOLDER,
            reason: format!("unknown request_id {id}"),
        };
    };

    if !wire.ok {
        return MycelixResponse::Error {
            requester,
            reason: wire
                .error
                .unwrap_or_else(|| "bridge reported failure with no reason".to_string()),
        };
    }

    match kind {
        PendingKind::GetActiveProposals => MycelixResponse::ActiveProposals {
            requester,
            proposals: match wire.data {
                Some(serde_json::Value::Array(arr)) => arr,
                Some(other) => vec![other],
                None => vec![],
            },
        },
        PendingKind::ProposalSubmitted => MycelixResponse::ProposalSubmitted {
            requester,
            action_hash: match wire.data {
                Some(serde_json::Value::String(s)) => s,
                Some(other) => other.to_string(),
                None => String::new(),
            },
        },
        PendingKind::VoteCast { proposal_id } => MycelixResponse::VoteCast {
            requester,
            proposal_id,
        },
        PendingKind::TendBalance => MycelixResponse::TendBalance {
            requester,
            balance: wire.data.unwrap_or(serde_json::Value::Null),
        },
        PendingKind::Proposal { proposal_id } => MycelixResponse::Proposal {
            requester,
            proposal_id,
            // `get_proposal` returns `Option<Record>` — the conductor
            // sends Null for None, an object for Some(Record). Preserve
            // that distinction: None → `record: None`, Some(object) →
            // `record: Some(object)`.
            record: match wire.data {
                Some(serde_json::Value::Null) | None => None,
                Some(other) => Some(other),
            },
        },
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum DispatcherError {
    #[error("failed to spawn bridge subprocess at {path:?}: {source}")]
    Spawn {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("subprocess stdin handle missing")]
    MissingStdin,
    #[error("subprocess stdout handle missing")]
    MissingStdout,
    #[error("failed to write to subprocess stdin: {0}")]
    Stdin(#[source] std::io::Error),
    #[error("failed to read from subprocess stdout: {0}")]
    Stdout(#[source] std::io::Error),
    #[error("failed to serialise request: {0}")]
    Serialise(#[source] serde_json::Error),
    #[error("tokio task panicked: {0}")]
    Join(#[source] tokio::task::JoinError),
}
