// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sovereign RDP Server binary.
//!
//! Captures the screen, encodes via HDC+pixel hybrid codec, encrypts with
//! PQC session keys, and streams to connected clients.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --bin symthaea-rdp-server --features rdp-server
//! RDP_MIN_CONSCIOUSNESS=0.1 RDP_FPS=10 cargo run --bin symthaea-rdp-server --features rdp-server
//! ```

use std::time::{Duration, Instant};

use anyhow::Result;
use tracing::info;

use symthaea::swarm::rdp_capture::TestCapture;
use symthaea::swarm::rdp_input::{InputInjector, LoggingInjector};
use symthaea::swarm::rdp_protocol::RdpSessionConfig;
use symthaea::swarm::rdp_server::RdpServerHandle;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "symthaea_rdp_server=info,symthaea=warn".into()),
        )
        .init();

    let fps: u8 = std::env::var("RDP_FPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let min_consciousness: f32 = std::env::var("RDP_MIN_CONSCIOUSNESS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.1);
    let width: u32 = std::env::var("RDP_WIDTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1920);
    let height: u32 = std::env::var("RDP_HEIGHT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1080);

    let config = RdpSessionConfig {
        target_fps: fps,
        min_consciousness,
        ..RdpSessionConfig::default()
    };

    info!("Starting Sovereign RDP Server");
    info!("  Resolution: {}x{}", width, height);
    info!("  Target FPS: {}", fps);
    info!("  Min consciousness: {}", min_consciousness);

    // Screen capture backend.
    // TODO: Replace TestCapture with X11ShmCapture when rdp-server feature includes x11rb.
    let capture = Box::new(TestCapture::new(width, height));

    // Input injector.
    let mut injector = LoggingInjector::new(width, height);

    // Create server.
    let (mut server, _inbound_tx, _outbound_rx) =
        RdpServerHandle::new(capture, config, "symthaea-rdp-server");

    let cycle_interval = Duration::from_millis(1000 / fps.max(1) as u64);

    info!("Server running at {} Hz. Waiting for connections...", fps);

    // Main loop.
    let mut cycle: u64 = 0;
    loop {
        let start = Instant::now();

        // Tick the server (captures screen, encodes, queues frames).
        let inputs = server.tick(0.5); // TODO: get from CLS consciousness_level

        // Inject received inputs.
        for (_session_id, events) in &inputs {
            injector.process_events(events);
        }

        cycle += 1;
        if cycle % 500 == 0 {
            info!(
                "Cycle {}: frame_id={}, sessions={}, injected_events={}",
                cycle,
                server.frame_id(),
                server.session_count(),
                injector.events.len(),
            );
        }

        // Sleep to target FPS.
        let elapsed = start.elapsed();
        if elapsed < cycle_interval {
            std::thread::sleep(cycle_interval - elapsed);
        }
    }
}
