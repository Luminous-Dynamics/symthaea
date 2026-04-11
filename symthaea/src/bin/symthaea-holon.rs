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
//! - `HOLON_LISTEN` — Listen address (default: 127.0.0.1)
//! - `HOLON_TOKEN` — Optional shared token for all endpoints (recommended if binding to LAN)
//! - `HOLON_INSECURE_ALLOW_UNAUTH` — If set truthy, allow unauthenticated LAN bind (NOT recommended)
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

fn env_truthy(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

fn listen_is_loopback(listen: &str) -> bool {
    if listen == "localhost" {
        return true;
    }
    if let Ok(ip) = listen.parse::<std::net::IpAddr>() {
        return ip.is_loopback();
    }
    false
}

fn generate_auth_token() -> String {
    use std::io::Read;

    let mut bytes = [0u8; 32];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        if f.read_exact(&mut bytes).is_ok() {
            return bytes.iter().map(|b| format!("{:02x}", b)).collect();
        }
    }

    // Fallback: only used if /dev/urandom is unavailable.
    let seed = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    blake3::hash(seed.as_bytes()).to_hex().to_string()
}

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
    let listen = std::env::var("HOLON_LISTEN").unwrap_or_else(|_| "127.0.0.1".into());
    let insecure_allow_unauth = env_truthy("HOLON_INSECURE_ALLOW_UNAUTH");
    let configured_token = std::env::var("HOLON_TOKEN").ok().and_then(|t| {
        if t.trim().is_empty() {
            None
        } else {
            Some(t)
        }
    });

    let addr = format!("{}:{}", listen, port);
    let cycle_interval = std::time::Duration::from_micros(1_000_000 / cycle_hz as u64);

    // Secure-by-default: if binding to a non-loopback interface, require an auth token.
    let require_token = !listen_is_loopback(&listen) && !insecure_allow_unauth;
    let auth_token = if let Some(t) = configured_token {
        Some(t)
    } else if require_token {
        Some(generate_auth_token())
    } else {
        None
    };

    if !listen_is_loopback(&listen) && insecure_allow_unauth {
        warn!(
            "Holon is binding to a non-loopback address WITHOUT auth (HOLON_INSECURE_ALLOW_UNAUTH=1). This is insecure."
        );
    }

    info!("Initializing CognitiveLoopService...");
    // Enable federated learning coordinator — spawned at CLS construction if swarm feature
    // is available. Timeout-guarded (config.timeout_ms) so dead peers don't hang.
    let holon_config = {
        let mut c = CognitiveLoopConfig::default();
        #[cfg(feature = "swarm")]
        {
            c.federation_enabled = true;
        }
        c
    };
    let mut cls = CognitiveLoopService::new(holon_config)?;

    // Extract the Holon inbound sender for the HTTP layer.
    let holon_tx = cls.holon_inbound_sender();
    let holon_http_state = Arc::new(match auth_token {
        Some(ref t) => HolonHttpState::new_with_auth_token(holon_tx, t.clone()),
        None => HolonHttpState::new(holon_tx),
    });

    // Build Holon HTTP router.
    let router = holon_router(holon_http_state.clone());

    // Spawn HTTP server.
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("Holon HTTP listening on http://{}", addr);
    if let Some(ref t) = auth_token {
        info!("Holon auth token required (send Authorization: Bearer ... or ?token=...)");
        info!("  HOLON_TOKEN={}", t);
    }
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

    // Wire Iroh P2P — creates NetworkService, initializes Ed25519 attestation,
    // spawns NetworkServiceBridge so SwarmEvents flow into the CLS each cycle.
    // Feature-gated: only compiled when both `identity` and `swarm` are enabled.
    #[cfg(all(feature = "identity", feature = "swarm"))]
    {
        match cls.enable_network_attestation().await {
            Ok(()) => {
                info!("Iroh P2P swarm active — Ed25519 attestation enabled");
                // Spawn the inbound connection accept loop.
                // Peers who connect to us are registered and their SwarmEvents
                // flow into the CLS via the NetworkServiceBridge spawned above.
                if let Some(service) = cls.network_service().cloned() {
                    tokio::spawn(service.accept_connections());
                    info!("Iroh: inbound connection listener active");
                }
            }
            Err(e) => {
                warn!(error = %e, "P2P networking disabled — running local-only");
            }
        }
    }

    // Advertise Holon via mDNS so Soma can auto-discover on LAN.
    // Uses avahi-publish if available (NixOS has avahi by default).
    let mdns_port = port;
    tokio::spawn(async move {
        let hostname = std::env::var("HOSTNAME")
            .or_else(|_| std::env::var("HOST"))
            .unwrap_or_else(|_| "symthaea".to_string());
        let service_name = format!("Symthaea Holon ({})", hostname);
        info!("Advertising mDNS: _symthaea._tcp port {}", mdns_port);

        let result = tokio::process::Command::new("avahi-publish-service")
            .arg(&service_name)
            .arg("_symthaea._tcp")
            .arg(mdns_port.to_string())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();

        match result {
            Ok(mut child) => {
                info!("mDNS: advertising as '{}'", service_name);
                let _ = child.wait().await;
            }
            Err(_) => {
                warn!("mDNS: avahi-publish-service not found — Soma must connect manually");
            }
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
