// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Holon Daemon
//!
//! Runs a CognitiveLoopService with Holon HTTP endpoints for Soma mobile bridge.
//! Soma devices (Android/iOS) connect via HTTP to exchange consciousness data.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --bin symthaea-holon --features api_module
//! HOLON_PORT=7778 HOLON_HZ=20 cargo run --bin symthaea-holon --features api_module
//! ```
//!
//! ## Environment Variables
//!
//! - `HOLON_PORT` — HTTP port (default: 7778)
//! - `HOLON_HZ` — Consciousness loop frequency in Hz (default: 20)
//! - `HOLON_LISTEN` — Listen address (default: 0.0.0.0)
//!
//! ## Endpoints
//!
//! - GET  /holon/status        — Desktop consciousness level + feature flags
//! - POST /holon/outbound      — Phone sends consciousness data to desktop
//! - GET  /holon/inbound       — Phone polls for desktop responses
//! - POST /holon/consciousness — Bidirectional consciousness exchange
//! - POST /holon/broca         — Language generation (stub)
//! - POST /holon/converse      — Conversation (stub)
//! - POST /holon/tts           — Text-to-speech (stub)

use std::sync::Arc;

use anyhow::Result;
use tracing::{info, warn};

use symthaea::api::holon::{holon_router, HolonHttpState};
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "symthaea_holon=info,symthaea=warn".into()),
        )
        .init();

    let port: u16 = std::env::var("HOLON_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7778);
    let cycle_hz: u32 = std::env::var("HOLON_HZ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let listen = std::env::var("HOLON_LISTEN").unwrap_or_else(|_| "0.0.0.0".into());

    let addr = format!("{}:{}", listen, port);
    let cycle_interval = std::time::Duration::from_micros(1_000_000 / cycle_hz as u64);

    info!("Initializing CognitiveLoopService...");
    let cls = CognitiveLoopService::new(CognitiveLoopConfig::default())?;

    // Extract the Holon inbound sender for the HTTP layer.
    let holon_tx = cls.holon_inbound_sender();
    let holon_http_state = Arc::new(HolonHttpState::new(holon_tx));

    // Build Holon HTTP router.
    let router = holon_router(holon_http_state.clone());

    // Spawn HTTP server.
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("Holon HTTP listening on http://{}", addr);
    info!("  GET  /holon/status");
    info!("  POST /holon/outbound");
    info!("  GET  /holon/inbound");
    info!("  POST /holon/consciousness");

    let http_state = holon_http_state.clone();
    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, router).await {
            tracing::error!("Holon HTTP server error: {}", e);
        }
    });

    // Run cognitive loop on a blocking thread (CLS is !Send due to internal state).
    info!(
        "Starting consciousness loop at {} Hz (interval {:?})",
        cycle_hz, cycle_interval
    );

    let consciousness_loop = tokio::task::spawn_blocking(move || {
        let mut cls = cls;
        let mut cycle: u64 = 0;
        loop {
            let start = std::time::Instant::now();

            // Run one cognitive cycle.
            let _result = cls.cycle("holon-daemon");

            // Update HTTP state with latest consciousness metrics.
            http_state.set_consciousness(cls.consciousness_level());
            http_state.peer_count.store(
                cls.holon_soma_peer_count() as u32,
                std::sync::atomic::Ordering::Relaxed,
            );

            // Update telemetry JSON for dashboard (every 10 cycles to avoid lock churn).
            if cycle % 10 == 0 {
                if let Ok(mut t) = http_state.telemetry_json.lock() {
                    *t = cls.holon_collective_qol_json();
                }
            }

            // Process pending Holon task requests (broca, converse).
            let tasks = cls.holon_drain_task_requests();
            for (device_id, task_type, payload) in tasks {
                let response_text = match task_type.as_str() {
                    "broca" => {
                        // Try real Broca generation if ssm_language feature is enabled.
                        #[cfg(feature = "ssm_language")]
                        {
                            cls.holon_broca_generate(&payload).unwrap_or_else(|| {
                                format!(
                                    "[Broca gated @ CL={:.2}] {}",
                                    cls.consciousness_level(),
                                    payload,
                                )
                            })
                        }
                        #[cfg(not(feature = "ssm_language"))]
                        {
                            format!("[Broca unavailable — enable ssm_language] {}", payload,)
                        }
                    }
                    "converse" => {
                        // Converse uses same Broca path with the user's text as context.
                        #[cfg(feature = "ssm_language")]
                        {
                            cls.holon_broca_generate(&payload).unwrap_or_else(|| {
                                format!(
                                    "[Converse gated @ CL={:.2}] {}",
                                    cls.consciousness_level(),
                                    payload,
                                )
                            })
                        }
                        #[cfg(not(feature = "ssm_language"))]
                        {
                            format!(
                                "[Converse @ CL={:.2}] {}",
                                cls.consciousness_level(),
                                payload,
                            )
                        }
                    }
                    _ => {
                        format!("[Task '{}'] {}", task_type, payload)
                    }
                };

                // Queue response for the requesting device.
                use symthaea::consciousness::holon_receiver::HolonResponse;
                cls.holon_send_to_soma(
                    &device_id,
                    HolonResponse::LanguageOutput {
                        text: response_text,
                    },
                );

                // Also push to HTTP outbound for polling via /holon/inbound.
                http_state.push_outbound(vec![
                    symthaea::consciousness::holon_receiver::HolonResponse::LanguageOutput {
                        text: format!("[{}] Response queued for device {}", task_type, device_id),
                    },
                ]);
            }

            cycle += 1;
            if cycle % 1000 == 0 {
                info!(
                    "Cycle {}: consciousness={:.4}, holon_peers={}, holon_processed={}",
                    cycle,
                    cls.consciousness_level(),
                    cls.holon_soma_peer_count(),
                    cls.holon_total_processed(),
                );
            }

            // Sleep to maintain target Hz.
            let elapsed = start.elapsed();
            if elapsed < cycle_interval {
                std::thread::sleep(cycle_interval - elapsed);
            }
        }
    });

    // Wait for consciousness loop (runs forever unless ctrl-c).
    tokio::select! {
        result = consciousness_loop => {
            if let Err(e) = result {
                warn!("Consciousness loop ended: {}", e);
            }
        }
        _ = tokio::signal::ctrl_c() => {
            info!("Shutting down Holon daemon...");
        }
    }

    Ok(())
}
