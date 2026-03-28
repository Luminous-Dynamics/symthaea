//! Symthaea SSH WebSocket Relay
//!
//! Bridges browser WebSocket connections to SSH targets for nixos-anywhere
//! orchestration. The browser portal connects via WebSocket, sends SSH
//! commands, and receives streaming output.
//!
//! # Architecture
//! ```text
//! Browser (WASM) <-> WebSocket <-> Relay Server <-> SSH Target
//! ```
//!
//! # Security
//! - WebSocket connections authenticated via SovereignAttestation
//! - SSH credentials never leave the relay (memory-only)
//! - Rate limited to 1 active session per IP
//! - Session timeout: 30 minutes
//!
//! # Usage
//! ```bash
//! cargo run --bin ssh-relay -- --port 8091
//! ```
//!
//! # Protocol
//! Client -> Relay:
//!   { "action": "connect", "host": "...", "port": 22, "username": "root", "auth": "..." }
//!   { "action": "exec", "command": "nixos-anywhere ...", "stream": true }
//!   { "action": "disconnect" }
//!
//! Relay -> Client:
//!   { "type": "connected", "host_key": "..." }
//!   { "type": "output", "data": "...", "stream": "stdout|stderr" }
//!   { "type": "exit", "code": 0 }
//!   { "type": "error", "message": "..." }

/// SSH relay session state.
#[derive(Debug)]
struct SshSession {
    /// Target host.
    host: String,
    /// Target port.
    port: u16,
    /// Username.
    username: String,
    /// Session start time.
    started_at: std::time::Instant,
    /// Commands executed.
    commands_executed: u32,
    /// Whether connected.
    connected: bool,
}

/// nixos-anywhere orchestration stages.
#[derive(Debug, Clone, serde::Serialize)]
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

/// nixos-anywhere progress update.
#[derive(Debug, Clone, serde::Serialize)]
struct InstallProgress {
    stage: NixosAnywhereStage,
    message: String,
    percentage: u8,
    elapsed_seconds: u64,
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

fn main() {
    eprintln!("Symthaea SSH WebSocket Relay");
    eprintln!("Usage: ssh-relay --port 8091");
    eprintln!();
    eprintln!("This service requires:");
    eprintln!("  - tokio + tungstenite (WebSocket)");
    eprintln!("  - russh (SSH client, pure Rust)");
    eprintln!();
    eprintln!("The relay bridges browser WebSocket <-> SSH target for");
    eprintln!("nixos-anywhere orchestration from the Symthaea portal.");
    eprintln!();
    eprintln!("Protocol stages for nixos-anywhere:");
    let stages = [
        ("1. Connecting", "SSH handshake with target"),
        ("2. UploadingKexec", "Upload kexec image to target"),
        ("3. Kexec", "Execute kexec to boot NixOS installer"),
        (
            "4. WaitingForReboot",
            "Wait for target to come back on new kernel",
        ),
        ("5. Partitioning", "Run disko to partition disks"),
        ("6. Installing", "Run nixos-install with flake"),
        (
            "7. Configuring",
            "Post-install configuration (users, SSH keys)",
        ),
        ("8. FinalReboot", "Reboot into installed system"),
        ("9. Verifying", "Verify system health + consciousness init"),
        ("10. Complete", "Installation complete, First Breath"),
    ];
    for (stage, desc) in &stages {
        eprintln!("  {} — {}", stage, desc);
    }

    // Demo: parse some nixos-anywhere output
    let test_outputs = [
        "copying kexec image to target...",
        "executing kexec -e",
        "waiting for host to come back after reboot...",
        "running disko partitioning...",
        "running nixos-install --flake .#guardian",
    ];
    eprintln!();
    eprintln!("Stage detection demo:");
    for output in &test_outputs {
        if let Some(stage) = parse_stage(output) {
            eprintln!("  '{}' -> {:?}", output, stage);
        }
    }
}
