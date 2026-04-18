// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mesh Daemon Orchestrator
//!
//! Manages a `spore-mesh-daemon` child process — Symthaea's edge consciousness kernel.
//! The daemon runs at 15–50 Hz, reading text from stdin and producing JSON consciousness
//! deltas on stdout. This orchestrator spawns, pipes, and reaps the daemon lifecycle.
//!
//! ## Architecture
//!
//! ```text
//!   CognitiveLoopService
//!        │
//!        ▼ SwarmEvent::ConsciousnessUpdate
//!   MeshDaemonOrchestrator
//!        │          ▲
//!   stdin│          │stdout (JSON lines)
//!        ▼          │
//!   spore-mesh-daemon (child process, 15-50 Hz)
//! ```
//!
//! ## Design Decisions
//!
//! - **Subprocess isolation**: Avoids serde version conflicts, enables deployment on
//!   edge hardware (RPi Zero, ESP32) with the same binary.
//! - **kill_on_drop**: Prevents zombie processes on panic/crash.
//! - **Bounded channel (64)**: At 50 Hz, 64 slots = 1.3s buffer. Backpressure warning
//!   at 80% capacity (51 items) signals cognitive loop lag.

use serde::Deserialize;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::mpsc;

/// Channel capacity for daemon output buffering.
const CHANNEL_CAPACITY: usize = 64;
/// Warn when channel reaches this fraction of capacity.
const BACKPRESSURE_WARN_THRESHOLD: usize = 51; // ~80% of 64

/// Configuration for the mesh daemon subprocess.
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// Path to the `spore-mesh-daemon` binary (searched in $PATH if relative).
    pub binary_path: String,
    /// HDC dimension for the edge kernel (8192 for RPi, 16384 for desktop).
    pub dimension: u32,
    /// Target cycle rate in Hz (15 for edge, 50 for desktop).
    pub hz: u32,
    /// Whether to emit JSON telemetry on stderr.
    pub json_telemetry: bool,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            binary_path: "spore-mesh-daemon".to_string(),
            dimension: 8192,
            hz: 15,
            json_telemetry: false,
        }
    }
}

/// JSON output from the daemon's stdout (one line per cycle).
#[derive(Debug, Clone, Deserialize)]
pub struct ConsciousnessOutput {
    #[allow(dead_code)]
    pub kind: String,
    #[allow(dead_code)]
    pub cycle: u64,
    pub consciousness_level: f32,
    pub phi: f32,
    pub valence: f32,
    pub arousal: f32,
    #[allow(dead_code)]
    pub harmony_alignment: f32,
}

/// Manages the lifecycle of a `spore-mesh-daemon` child process.
///
/// Handles spawning, stdin/stdout piping, graceful shutdown, and zombie prevention.
pub struct MeshDaemonOrchestrator {
    child: Option<Child>,
    stdin_tx: Option<tokio::io::BufWriter<ChildStdin>>,
    stdout_rx: Option<mpsc::Receiver<ConsciousnessOutput>>,
    #[allow(dead_code)]
    config: DaemonConfig,
}

impl MeshDaemonOrchestrator {
    /// Spawn the mesh daemon as a child process.
    ///
    /// Returns immediately after the process starts. Use `drain_outputs()` to
    /// read consciousness deltas produced by the daemon each cycle.
    pub async fn spawn(config: DaemonConfig) -> anyhow::Result<Self> {
        let mut child = Command::new(&config.binary_path)
            .env("SPORE_DIM", config.dimension.to_string())
            .env("SPORE_HZ", config.hz.to_string())
            .env(
                "SPORE_JSON_TELEMETRY",
                if config.json_telemetry { "1" } else { "0" },
            )
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true) // Zombie prevention: kill child if orchestrator is dropped
            .spawn()?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to capture daemon stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to capture daemon stdout"))?;

        // Spawn async stdout reader → bounded mpsc channel
        let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
        tokio::spawn(Self::read_stdout_loop(stdout, tx));

        Ok(Self {
            child: Some(child),
            stdin_tx: Some(tokio::io::BufWriter::new(stdin)),
            stdout_rx: Some(rx),
            config,
        })
    }

    /// Non-blocking drain of all available daemon consciousness outputs.
    ///
    /// Call this each tick from the cognitive loop or WebSocket handler.
    /// Returns an empty vec if no outputs are available.
    pub fn drain_outputs(&mut self) -> Vec<ConsciousnessOutput> {
        let mut outputs = Vec::new();
        if let Some(ref mut rx) = self.stdout_rx {
            while let Ok(output) = rx.try_recv() {
                outputs.push(output);
            }
            // Backpressure warning: if we drained many, the loop might be lagging
            if outputs.len() >= BACKPRESSURE_WARN_THRESHOLD {
                tracing::warn!(
                    buffered = outputs.len(),
                    capacity = CHANNEL_CAPACITY,
                    "Mesh daemon output buffer near capacity — cognitive loop may be lagging"
                );
            }
        }
        outputs
    }

    /// Send text input to the daemon (queued for its next cycle).
    ///
    /// The daemon reads one line per cycle from stdin. If stdin is full,
    /// this will await briefly until the daemon consumes.
    pub async fn send_input(&mut self, text: &str) -> anyhow::Result<()> {
        if let Some(ref mut stdin) = self.stdin_tx {
            stdin.write_all(text.as_bytes()).await?;
            stdin.write_all(b"\n").await?;
            stdin.flush().await?;
        }
        Ok(())
    }

    /// Check if the daemon process is still alive.
    pub fn is_alive(&mut self) -> bool {
        if let Some(ref mut child) = self.child {
            // try_wait returns Ok(Some(status)) if exited, Ok(None) if still running
            matches!(child.try_wait(), Ok(None))
        } else {
            false
        }
    }

    /// Gracefully shut down the daemon.
    ///
    /// 1. Drops stdin (daemon sees EOF → exits gracefully)
    /// 2. Waits up to 5s for clean exit
    /// 3. Force-kills if still alive (zombie prevention)
    pub async fn shutdown(&mut self) {
        // Drop stdin → daemon sees EOF → should exit
        self.stdin_tx.take();
        self.stdout_rx.take();

        if let Some(ref mut child) = self.child {
            // Wait up to 5s for graceful exit
            let wait_result =
                tokio::time::timeout(Duration::from_secs(5), child.wait()).await;

            match wait_result {
                Ok(Ok(status)) => {
                    tracing::info!("Mesh daemon exited: {status}");
                }
                _ => {
                    // Timed out or error — force kill
                    let _ = child.kill().await;
                    tracing::warn!("Mesh daemon force-killed after 5s timeout");
                }
            }
        }
        self.child.take();
    }

    /// Internal: reads JSON lines from daemon stdout, sends to channel.
    async fn read_stdout_loop(stdout: ChildStdout, tx: mpsc::Sender<ConsciousnessOutput>) {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();

        while let Ok(Some(line)) = lines.next_line().await {
            // Skip empty lines and non-JSON output (warmup messages, etc.)
            if line.is_empty() || !line.starts_with('{') {
                continue;
            }

            match serde_json::from_str::<ConsciousnessOutput>(&line) {
                Ok(output) => {
                    if tx.send(output).await.is_err() {
                        // Receiver dropped — orchestrator shut down
                        break;
                    }
                }
                Err(e) => {
                    tracing::debug!(error = %e, line = %&line[..line.len().min(80)], "Daemon output parse error");
                }
            }
        }

        tracing::info!("Mesh daemon stdout reader exited");
    }
}

impl Drop for MeshDaemonOrchestrator {
    fn drop(&mut self) {
        // kill_on_drop(true) on the Child handles zombie prevention,
        // but log for observability
        if self.child.is_some() {
            tracing::debug!("MeshDaemonOrchestrator dropped — child process will be killed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daemon_config_default() {
        let config = DaemonConfig::default();
        assert_eq!(config.dimension, 8192);
        assert_eq!(config.hz, 15);
        assert!(!config.json_telemetry);
        assert_eq!(config.binary_path, "spore-mesh-daemon");
    }

    #[test]
    fn test_consciousness_output_deserialize() {
        let json = r#"{"kind":"delta","cycle":42,"consciousness_level":0.67,"phi":0.45,"valence":-0.1,"arousal":0.8,"harmony_alignment":0.55}"#;
        let output: ConsciousnessOutput = serde_json::from_str(json).unwrap();
        assert_eq!(output.cycle, 42);
        assert!((output.phi - 0.45).abs() < 0.001);
        assert!((output.valence - (-0.1)).abs() < 0.001);
        assert!((output.arousal - 0.8).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_spawn_missing_binary_returns_error() {
        let config = DaemonConfig {
            binary_path: "/nonexistent/binary/path".to_string(),
            ..DaemonConfig::default()
        };
        let result = MeshDaemonOrchestrator::spawn(config).await;
        assert!(result.is_err());
    }
}
