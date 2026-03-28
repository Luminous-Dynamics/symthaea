// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sovereign RDP Client binary.
//!
//! Connects to a Sovereign RDP server, receives encrypted frames,
//! reconstructs the display, and forwards input.
//!
//! ## Usage
//!
//! ```bash
//! # Terminal mode (no GUI dependency)
//! cargo run --bin symthaea-rdp-client --features rdp-client
//!
//! # GUI mode (egui window)
//! cargo run --bin symthaea-rdp-client --features "rdp-client,gui"
//! ```

use std::time::{Duration, Instant};

use anyhow::Result;
use tracing::info;

use symthaea::swarm::rdp_client::{ClientInbound, RdpClientHandle};
use symthaea::swarm::rdp_protocol::{
    ControlMessage, FullFrame, QuantizedPatch, RdpFrame, RdpSessionConfig,
};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "symthaea_rdp_client=info,symthaea=warn".into()),
        )
        .init();

    let server_addr = std::env::var("RDP_SERVER").unwrap_or_else(|_| "localhost".into());

    info!("Sovereign RDP Client");
    info!("  Server: {}", server_addr);

    let config = RdpSessionConfig::default();
    let (mut client, inbound_tx, _outbound_rx) = RdpClientHandle::new(&server_addr, config);

    // Simulate receiving a Welcome + test frame for demo purposes.
    // In production, the async actor handles QUIC connection.
    inbound_tx.send(ClientInbound::Welcome {
        patch_cols: 30,
        patch_rows: 17,
        target_fps: 5,
        consciousness_level: 0.5,
    })?;

    // Generate a synthetic full frame for demo.
    let n_tiles = 30 * 17;
    let patches: Vec<QuantizedPatch> = (0..n_tiles)
        .map(|i| {
            let lum = ((i * 3) % 256) as i8;
            QuantizedPatch {
                values: vec![lum; 64 * 64],
            }
        })
        .collect();
    inbound_tx.send(ClientInbound::Frame(RdpFrame::Full(FullFrame {
        frame_id: 1,
        timestamp_ms: 0,
        patch_cols: 30,
        patch_rows: 17,
        patches,
        consciousness_level: 0.5,
        harmony: "present".into(),
    })))?;

    // Poll loop (terminal mode — just logs status).
    let mut cycle: u64 = 0;
    loop {
        let start = Instant::now();

        let updated = client.poll();
        if updated && cycle % 100 == 0 {
            info!(
                "Frame {} applied | {}x{} | {:.1} FPS | Server Φ={:.2}",
                client.frame_buffer.last_frame_id,
                client.frame_buffer.width,
                client.frame_buffer.height,
                client.current_fps,
                client.server_consciousness,
            );
        }

        cycle += 1;
        if cycle >= 100 {
            info!(
                "Demo complete. {} frames applied, buffer {}x{}, {} bytes.",
                client.frame_buffer.frames_applied,
                client.frame_buffer.width,
                client.frame_buffer.height,
                client.frame_buffer.pixels.len(),
            );
            break;
        }

        let elapsed = start.elapsed();
        if elapsed < Duration::from_millis(200) {
            std::thread::sleep(Duration::from_millis(200) - elapsed);
        }
    }

    Ok(())
}
