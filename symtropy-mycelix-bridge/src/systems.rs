// Copyright (c) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Subprocess IPC layer.
//!
//! We spawn `mycelix-conductor-bridge` as a child process and exchange JSON
//! messages over its stdin / stdout. See module-level docs on [`crate`] for
//! the "why subprocess, not in-process" rationale.
//!
//! ## Protocol (Milestone 1)
//!
//! **Request** (one JSON per line, written to subprocess stdin):
//! ```json
//! {"type": "QueryActiveProposals"}
//! ```
//!
//! **Response** (one JSON per line, read from subprocess stdout):
//! ```json
//! {"ok": true, "data": [...proposals...]}
//! {"ok": false, "error": "<reason>"}
//! ```
//!
//! Correlation is FIFO in Milestone 1: Nth response matches Nth request.
//! Milestone 2 adds opaque request-id round-tripping.

use std::collections::VecDeque;
use std::process::Stdio;

use bevy::prelude::*;
use bevy_tokio_tasks::TokioTasksRuntime;
use flume::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, Command};

use crate::config::MycelixConfig;
use crate::events::{MycelixRequest, MycelixResponse};
use crate::resource::{MycelixRequestOutbox, MycelixResponseInbox};

// ---------------------------------------------------------------------------
// Wire protocol types
// ---------------------------------------------------------------------------

/// What we serialise to subprocess stdin. Kept permissive — Milestone 2 adds
/// more variants; using `#[serde(tag = "type")]` matches the bridge's own
/// Command enum serde attribute.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum WireCommand {
    QueryActiveProposals,
}

/// What we deserialise from subprocess stdout. Matches
/// `mycelix-conductor-bridge::BridgeResponse`.
#[derive(Debug, Deserialize)]
struct WireResponse {
    ok: bool,
    #[serde(default)]
    data: Option<serde_json::Value>,
    #[serde(default)]
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Startup system: spawn the subprocess + background IPC task.
///
/// Registered by [`crate::BevyMycelixPlugin`]; not intended for direct use.
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
            error!(?err, "symtropy-mycelix-bridge: dispatcher loop exited with error");
        }
    });
}

/// Update-schedule system: drain the inbox into Bevy [`MycelixResponse`]
/// messages. Registered by [`crate::BevyMycelixPlugin`].
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
            // Drain any queued requests with an Error response so user systems
            // don't silently hang waiting for a subprocess that'll never start.
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

    // Task 1: pump requests → stdin.
    // Task 2: pump stdout → responses.
    // They share a `pending` queue of requester Entities so responses can be
    // correlated FIFO back to their originating request.
    let pending: std::sync::Arc<tokio::sync::Mutex<VecDeque<Entity>>> =
        std::sync::Arc::new(tokio::sync::Mutex::new(VecDeque::new()));

    let pending_tx = pending.clone();
    let writer_task = tokio::spawn(async move {
        writer_loop(stdin, req_rx, pending_tx).await
    });

    let pending_rx = pending.clone();
    let reader_task = tokio::spawn(async move {
        reader_loop(stdout, resp_tx, pending_rx).await
    });

    // If either task exits, the bridge session is done. Propagate the first
    // error we see; kill_on_drop on `child` cleans up the subprocess.
    tokio::select! {
        res = writer_task => res.map_err(DispatcherError::Join)??,
        res = reader_task => res.map_err(DispatcherError::Join)??,
    };

    Ok(())
}

/// When we can't even spawn the subprocess, translate every inbound
/// request into an [`MycelixResponse::Error`] so the Bevy side sees failure
/// rather than silence.
async fn drain_with_error(
    req_rx: Receiver<MycelixRequest>,
    resp_tx: Sender<MycelixResponse>,
    reason: String,
) {
    while let Ok(req) = req_rx.recv_async().await {
        let requester = match &req {
            MycelixRequest::GetActiveProposals { requester } => *requester,
        };
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

/// Forwards `MycelixRequest`s from the Bevy side into the subprocess's stdin
/// as JSON lines. Records each request's requester `Entity` into the shared
/// `pending` queue so the reader half can correlate responses.
async fn writer_loop(
    mut stdin: ChildStdin,
    req_rx: Receiver<MycelixRequest>,
    pending: std::sync::Arc<tokio::sync::Mutex<VecDeque<Entity>>>,
) -> Result<(), DispatcherError> {
    while let Ok(req) = req_rx.recv_async().await {
        let (requester, wire) = match req {
            MycelixRequest::GetActiveProposals { requester } => {
                (requester, WireCommand::QueryActiveProposals)
            }
        };

        let mut line = serde_json::to_string(&wire).map_err(DispatcherError::Serialise)?;
        line.push('\n');

        {
            let mut q = pending.lock().await;
            q.push_back(requester);
        }

        stdin
            .write_all(line.as_bytes())
            .await
            .map_err(DispatcherError::Stdin)?;
        stdin.flush().await.map_err(DispatcherError::Stdin)?;

        trace!(%requester, "dispatched request to subprocess");
    }

    // Request channel closed (plugin dropped). Close subprocess stdin so the
    // bridge shuts down cleanly.
    drop(stdin);
    Ok(())
}

/// Reads JSON lines from subprocess stdout, matches each to the earliest
/// pending requester `Entity`, and pushes a `MycelixResponse` onto the inbox
/// channel.
async fn reader_loop<R>(
    stdout: R,
    resp_tx: Sender<MycelixResponse>,
    pending: std::sync::Arc<tokio::sync::Mutex<VecDeque<Entity>>>,
) -> Result<(), DispatcherError>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut lines = BufReader::new(stdout).lines();

    while let Some(line) = lines
        .next_line()
        .await
        .map_err(DispatcherError::Stdout)?
    {
        let requester = {
            let mut q = pending.lock().await;
            q.pop_front().unwrap_or(Entity::PLACEHOLDER)
        };

        let response = match serde_json::from_str::<WireResponse>(&line) {
            Ok(wire) if wire.ok => MycelixResponse::ActiveProposals {
                requester,
                proposals: match wire.data {
                    Some(serde_json::Value::Array(arr)) => arr,
                    Some(other) => vec![other],
                    None => vec![],
                },
            },
            Ok(wire) => MycelixResponse::Error {
                requester,
                reason: wire
                    .error
                    .unwrap_or_else(|| "bridge reported failure with no reason".to_string()),
            },
            Err(e) => MycelixResponse::Error {
                requester,
                reason: format!("invalid JSON from bridge: {e}; line={line:?}"),
            },
        };

        if resp_tx.send_async(response).await.is_err() {
            info!("symtropy-mycelix-bridge: inbox closed; reader exiting");
            return Ok(());
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

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
