// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea SSH WebSocket Relay
//!
//! Bridges browser WebSocket connections to SSH targets for nixos-anywhere
//! orchestration. The browser portal connects via WebSocket, sends SSH
//! commands, and receives streaming output with nixos-anywhere stage detection.
//!
//! # Usage
//! ```bash
//! cargo run --bin ssh-relay --features server -- --port 8091
//! ```

use async_ssh2_tokio::client::{AuthMethod, Client, ServerCheckMethod};
use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;

/// nixos-anywhere orchestration stages.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "PascalCase")]
enum NixosAnywhereStage {
    Connecting,
    UploadingKexec,
    Kexec,
    WaitingForReboot,
    Partitioning,
    Installing,
    Configuring,
    FinalReboot,
    Verifying,
    Complete,
}

impl NixosAnywhereStage {
    fn percentage(&self) -> u8 {
        match self {
            Self::Connecting => 5,
            Self::UploadingKexec => 15,
            Self::Kexec => 25,
            Self::WaitingForReboot => 35,
            Self::Partitioning => 50,
            Self::Installing => 70,
            Self::Configuring => 85,
            Self::FinalReboot => 92,
            Self::Verifying => 97,
            Self::Complete => 100,
        }
    }

    fn inoculation_phase(&self) -> &'static str {
        match self {
            Self::Connecting => "TrustVerification",
            Self::UploadingKexec | Self::Kexec | Self::WaitingForReboot => "FlakeEvaluation",
            Self::Partitioning => "DiskPreparation",
            Self::Installing | Self::Configuring => "StorePopulation",
            Self::FinalReboot => "MokEnrollment",
            Self::Verifying | Self::Complete => "FirstBreath",
        }
    }
}

/// Parse nixos-anywhere output to determine current stage.
fn parse_stage(output: &str) -> Option<NixosAnywhereStage> {
    let lower = output.to_lowercase();
    if lower.contains("uploading kexec") || lower.contains("copying kexec") {
        Some(NixosAnywhereStage::UploadingKexec)
    } else if lower.contains("executing kexec") || lower.contains("kexec -e") {
        Some(NixosAnywhereStage::Kexec)
    } else if lower.contains("waiting for") && lower.contains("reboot") {
        Some(NixosAnywhereStage::WaitingForReboot)
    } else if lower.contains("partitioning") || lower.contains("disko") {
        Some(NixosAnywhereStage::Partitioning)
    } else if lower.contains("installing") || lower.contains("nixos-install") {
        Some(NixosAnywhereStage::Installing)
    } else if lower.contains("configuring") || lower.contains("nixos-rebuild") {
        Some(NixosAnywhereStage::Configuring)
    } else if lower.contains("final reboot") {
        Some(NixosAnywhereStage::FinalReboot)
    } else if lower.contains("verification") || lower.contains("complete") {
        Some(NixosAnywhereStage::Complete)
    } else {
        None
    }
}

/// Rate limiter: 1 active session per IP.
struct SessionTracker {
    active: HashMap<String, Instant>,
    timeout_secs: u64,
}

impl SessionTracker {
    fn new(timeout_secs: u64) -> Self {
        Self {
            active: HashMap::new(),
            timeout_secs,
        }
    }

    fn try_acquire(&mut self, ip: &str) -> bool {
        let now = Instant::now();
        // Expire old sessions
        self.active
            .retain(|_, start| now.duration_since(*start).as_secs() < self.timeout_secs);
        if self.active.contains_key(ip) {
            return false;
        }
        self.active.insert(ip.to_string(), now);
        true
    }

    fn release(&mut self, ip: &str) {
        self.active.remove(ip);
    }
}

/// Client → Relay message.
#[derive(serde::Deserialize)]
struct ClientMessage {
    action: String,
    #[serde(default)]
    host: String,
    #[serde(default = "default_port")]
    port: u16,
    #[serde(default)]
    username: String,
    #[serde(default)]
    password: String,
    #[serde(default)]
    command: String,
}

fn default_port() -> u16 {
    22
}

/// Relay → Client message.
#[derive(serde::Serialize)]
struct RelayMessage {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stage: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    percentage: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase: Option<String>,
}

impl RelayMessage {
    fn connected() -> Self {
        Self {
            msg_type: "connected".into(),
            data: None,
            stream: None,
            code: None,
            message: Some("SSH connection established".into()),
            stage: None,
            percentage: None,
            phase: None,
        }
    }

    fn output(data: &str, stream: &str) -> Self {
        Self {
            msg_type: "output".into(),
            data: Some(data.into()),
            stream: Some(stream.into()),
            code: None,
            message: None,
            stage: None,
            percentage: None,
            phase: None,
        }
    }

    fn progress(stage: &NixosAnywhereStage) -> Self {
        Self {
            msg_type: "progress".into(),
            data: None,
            stream: None,
            code: None,
            message: Some(format!("{:?}", stage)),
            stage: Some(format!("{:?}", stage)),
            percentage: Some(stage.percentage()),
            phase: Some(stage.inoculation_phase().into()),
        }
    }

    fn exit(code: i32) -> Self {
        Self {
            msg_type: "exit".into(),
            data: None,
            stream: None,
            code: Some(code),
            message: None,
            stage: None,
            percentage: None,
            phase: None,
        }
    }

    fn error(msg: &str) -> Self {
        Self {
            msg_type: "error".into(),
            data: None,
            stream: None,
            code: None,
            message: Some(msg.into()),
            stage: None,
            percentage: None,
            phase: None,
        }
    }

    fn disks(disks_json: &str) -> Self {
        Self {
            msg_type: "disks".into(),
            data: Some(disks_json.into()),
            stream: None,
            code: None,
            message: None,
            stage: None,
            percentage: None,
            phase: None,
        }
    }

    fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

/// Parsed disk info from lsblk.
#[derive(Debug, serde::Serialize)]
struct DiskInfo {
    name: String,
    size: String,
    model: String,
    transport: String, // nvme, sata, usb, virtio
    disk_type: String, // disk, part, rom
    removable: bool,
}

/// Parse lsblk --json output into structured disk info.
fn parse_lsblk(json_str: &str) -> Vec<DiskInfo> {
    let parsed: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let devices = match parsed.get("blockdevices").and_then(|b| b.as_array()) {
        Some(arr) => arr,
        None => return Vec::new(),
    };

    devices
        .iter()
        .filter_map(|dev| {
            let dtype = dev.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if dtype != "disk" {
                return None;
            }
            let name = dev.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let size = dev.get("size").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let model = dev
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown")
                .trim()
                .to_string();
            let tran = dev
                .get("tran")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let rm = dev.get("rm").and_then(|v| v.as_bool()).unwrap_or(false);

            Some(DiskInfo {
                name: format!("/dev/{}", name),
                size,
                model,
                transport: if tran.is_empty() {
                    "unknown".into()
                } else {
                    tran
                },
                disk_type: dtype.into(),
                removable: rm,
            })
        })
        .collect()
}

type SharedTracker = Arc<Mutex<SessionTracker>>;

async fn handle_connection(
    stream: tokio::net::TcpStream,
    peer_addr: String,
    tracker: SharedTracker,
) {
    // Upgrade to WebSocket
    let ws_stream = match accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            eprintln!("[{}] WebSocket upgrade failed: {}", peer_addr, e);
            return;
        }
    };

    let (mut ws_tx, mut ws_rx) = ws_stream.split();
    let mut ssh_client: Option<Client> = None;

    eprintln!("[{}] WebSocket connected", peer_addr);

    while let Some(msg) = ws_rx.next().await {
        let msg = match msg {
            Ok(Message::Text(t)) => t,
            Ok(Message::Close(_)) => break,
            Ok(_) => continue,
            Err(e) => {
                eprintln!("[{}] WebSocket error: {}", peer_addr, e);
                break;
            }
        };

        let client_msg: ClientMessage = match serde_json::from_str(&msg) {
            Ok(m) => m,
            Err(e) => {
                let _ = ws_tx
                    .send(Message::Text(
                        RelayMessage::error(&format!("Invalid JSON: {}", e)).to_json(),
                    ))
                    .await;
                continue;
            }
        };

        match client_msg.action.as_str() {
            "connect" => {
                // Rate limit: 1 session per IP
                {
                    let mut t = tracker.lock().await;
                    if !t.try_acquire(&peer_addr) {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(
                                    "Rate limited: only 1 active session per IP allowed",
                                )
                                .to_json(),
                            ))
                            .await;
                        continue;
                    }
                }

                eprintln!(
                    "[{}] Connecting to {}@{}:{}",
                    peer_addr, client_msg.username, client_msg.host, client_msg.port
                );

                let auth = AuthMethod::with_password(&client_msg.password);

                match Client::connect(
                    (client_msg.host.as_str(), client_msg.port),
                    &client_msg.username,
                    auth,
                    ServerCheckMethod::NoCheck,
                )
                .await
                {
                    Ok(client) => {
                        eprintln!("[{}] SSH connected", peer_addr);
                        ssh_client = Some(client);
                        let _ = ws_tx
                            .send(Message::Text(RelayMessage::connected().to_json()))
                            .await;
                    }
                    Err(e) => {
                        eprintln!("[{}] SSH connection failed: {}", peer_addr, e);
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("SSH connection failed: {}", e))
                                    .to_json(),
                            ))
                            .await;
                        tracker.lock().await.release(&peer_addr);
                    }
                }
            }

            "exec" => {
                let client = match ssh_client.as_ref() {
                    Some(c) => c,
                    None => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error("Not connected. Send 'connect' first.")
                                    .to_json(),
                            ))
                            .await;
                        continue;
                    }
                };

                eprintln!("[{}] Executing: {}", peer_addr, &client_msg.command);

                // Send initial connecting stage
                let _ = ws_tx
                    .send(Message::Text(
                        RelayMessage::progress(&NixosAnywhereStage::Connecting).to_json(),
                    ))
                    .await;

                match client.execute(&client_msg.command).await {
                    Ok(result) => {
                        // Process stdout line by line for stage detection
                        let combined = format!("{}{}", result.stdout, result.stderr);
                        for line in combined.lines() {
                            if !line.trim().is_empty() {
                                // Send output
                                let _ = ws_tx
                                    .send(Message::Text(
                                        RelayMessage::output(line, "stdout").to_json(),
                                    ))
                                    .await;

                                // Check for stage transitions
                                if let Some(stage) = parse_stage(line) {
                                    let _ = ws_tx
                                        .send(Message::Text(
                                            RelayMessage::progress(&stage).to_json(),
                                        ))
                                        .await;
                                }
                            }
                        }

                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::exit(result.exit_status as i32).to_json(),
                            ))
                            .await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("Command execution failed: {}", e))
                                    .to_json(),
                            ))
                            .await;
                    }
                }
            }

            "discover_disks" => {
                let client = match ssh_client.as_ref() {
                    Some(c) => c,
                    None => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error("Not connected. Send 'connect' first.")
                                    .to_json(),
                            ))
                            .await;
                        continue;
                    }
                };

                eprintln!("[{}] Discovering disks...", peer_addr);
                match client
                    .execute("lsblk --json -o NAME,SIZE,MODEL,TYPE,TRAN,RM -b")
                    .await
                {
                    Ok(result) if result.exit_status == 0 => {
                        let disks = parse_lsblk(&result.stdout);
                        let disks_json =
                            serde_json::to_string(&disks).unwrap_or_else(|_| "[]".into());
                        eprintln!(
                            "[{}] Found {} disks",
                            peer_addr,
                            disks.len()
                        );
                        let _ = ws_tx
                            .send(Message::Text(RelayMessage::disks(&disks_json).to_json()))
                            .await;
                    }
                    Ok(result) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!(
                                    "lsblk failed (exit {}): {}",
                                    result.exit_status,
                                    result.stderr.chars().take(200).collect::<String>()
                                ))
                                .to_json(),
                            ))
                            .await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("Disk discovery failed: {}", e))
                                    .to_json(),
                            ))
                            .await;
                    }
                }
            }

            "disconnect" => {
                eprintln!("[{}] Client disconnecting", peer_addr);
                ssh_client = None;
                tracker.lock().await.release(&peer_addr);
                break;
            }

            _ => {
                let _ = ws_tx
                    .send(Message::Text(
                        RelayMessage::error(&format!("Unknown action: {}", client_msg.action))
                            .to_json(),
                    ))
                    .await;
            }
        }
    }

    // Cleanup
    if ssh_client.is_some() {
        tracker.lock().await.release(&peer_addr);
    }
    eprintln!("[{}] WebSocket disconnected", peer_addr);
}

#[tokio::main]
async fn main() {
    let port: u16 = std::env::args()
        .skip_while(|a| a != "--port")
        .nth(1)
        .and_then(|p| p.parse().ok())
        .unwrap_or(8091);

    let tracker: SharedTracker = Arc::new(Mutex::new(SessionTracker::new(1800))); // 30 min timeout

    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).await.unwrap();
    eprintln!("Symthaea SSH Relay listening on ws://{}", addr);
    eprintln!("  Protocol: connect → exec → disconnect");
    eprintln!("  Session timeout: 30 minutes");
    eprintln!("  Rate limit: 1 active session per IP");

    while let Ok((stream, addr)) = listener.accept().await {
        let peer = addr.ip().to_string();
        let tracker = tracker.clone();
        tokio::spawn(handle_connection(stream, peer, tracker));
    }
}
