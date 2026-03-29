// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Mind Daemon
//!
//! Minimal TCP daemon that runs the ContinuousMind and exposes
//! a small JSON API for status and shutdown. This is the first
//! step toward wiring shell/GUI clients to the real mind instead
//! of simulated consciousness metrics.
//!
//! API (line-delimited JSON):
//! - `{"type":"status"}`   → current mind status snapshot
//! - `{"type":"shutdown"}` → request graceful shutdown

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tracing::{error, info, warn};

use symthaea::{ContinuousMind, MindConfig, MindState};

/// Mind daemon arguments
#[derive(Parser, Debug)]
#[command(name = "symthaea-mind")]
#[command(about = "Symthaea continuous mind daemon")]
#[command(version)]
struct Args {
    /// TCP address to listen on (e.g. 127.0.0.1:9500)
    #[arg(short, long, default_value = "127.0.0.1:9500")]
    listen: String,
}

/// Minimal external status view distilled from MindState
#[derive(Debug, Clone, Serialize)]
struct MindStatus {
    /// Current integrated information (Φ)
    pub phi: f64,
    /// Meta-awareness / self-monitoring level
    pub meta_awareness: f64,
    /// Cognitive load (0.0 = idle, 1.0 = max)
    pub cognitive_load: f64,
    /// Total cognitive cycles completed
    pub total_cycles: u64,
    /// Time since awakening in ms
    pub time_awake_ms: u64,
    /// Whether the mind currently considers itself conscious
    pub is_conscious: bool,
}

impl From<MindState> for MindStatus {
    fn from(s: MindState) -> Self {
        Self {
            phi: s.consciousness_level,
            meta_awareness: s.meta_awareness,
            cognitive_load: s.cognitive_load,
            total_cycles: s.total_cycles,
            time_awake_ms: s.time_awake_ms,
            is_conscious: s.is_conscious,
        }
    }
}

/// Requests understood by the mind daemon
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum MindRequest {
    #[serde(rename = "status")]
    Status,
    #[serde(rename = "shutdown")]
    Shutdown,
}

/// Responses produced by the mind daemon
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum MindResponse {
    #[serde(rename = "status")]
    Status {
        status: MindStatus,
        timestamp_ms: u64,
    },
    #[serde(rename = "shutdown_ack")]
    ShutdownAck,
    #[serde(rename = "error")]
    Error { message: String },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("symthaea_mind=info")
        .init();

    let args = Args::parse();
    let addr: SocketAddr = args
        .listen
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid listen address {}: {}", args.listen, e))?;

    info!("Starting Symthaea Mind daemon on {}", addr);

    // Initialize continuous mind and awaken it
    let config = MindConfig::default();
    let mut mind = ContinuousMind::new(config);
    mind.awaken();

    let shared_mind = Arc::new(Mutex::new(mind));

    // Start TCP listener
    let listener = TcpListener::bind(addr).await?;
    loop {
        let (stream, peer) = listener.accept().await?;
        info!("Client connected: {}", peer);

        let mind_handle = Arc::clone(&shared_mind);
        tokio::spawn(async move {
            if let Err(e) = handle_client(stream, mind_handle).await {
                error!("Client error: {}", e);
            }
        });
    }
}

async fn handle_client(stream: TcpStream, mind: Arc<Mutex<ContinuousMind>>) -> Result<()> {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            // EOF / disconnect
            break;
        }

        let req: MindRequest = match serde_json::from_str(line.trim()) {
            Ok(r) => r,
            Err(e) => {
                warn!("Invalid request: {} ({})", line.trim(), e);
                let resp = MindResponse::Error {
                    message: format!("Invalid request: {}", e),
                };
                send_response(&mut write_half, &resp).await?;
                continue;
            }
        };

        match req {
            MindRequest::Status => {
                let snapshot = match mind.lock() {
                    Ok(m) => m.snapshot(),
                    Err(poisoned) => {
                        error!("Mind mutex poisoned during status — recovering from poison");
                        poisoned.into_inner().snapshot()
                    }
                };
                let status = MindStatus::from(snapshot);
                let timestamp_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                let resp = MindResponse::Status {
                    status,
                    timestamp_ms,
                };
                send_response(&mut write_half, &resp).await?;
            }
            MindRequest::Shutdown => {
                {
                    match mind.lock() {
                        Ok(mut m) => m.request_shutdown(),
                        Err(poisoned) => {
                            error!("Mind mutex poisoned during shutdown — recovering from poison");
                            poisoned.into_inner().request_shutdown();
                        }
                    }
                }
                let resp = MindResponse::ShutdownAck;
                send_response(&mut write_half, &resp).await?;
                break;
            }
        }
    }

    Ok(())
}

async fn send_response<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    resp: &MindResponse,
) -> Result<()> {
    let json = serde_json::to_string(resp)?;
    writer.write_all(json.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    Ok(())
}
