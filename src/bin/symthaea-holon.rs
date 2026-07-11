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
//! - `HOLON_QUIC_PORT` — QUIC port for Phase I.C transport (default: `HOLON_PORT + 1`)
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
use tracing::{debug, info, warn};

use symthaea::api::holon::{HolonHttpState, holon_router};
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
#[cfg(feature = "holon-quic")]
use symthaea::swarm::quic_transport::{default_quic_port, spawn_holon_quic_server};

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
    if let Ok(mut f) = std::fs::File::open("/dev/urandom")
        && f.read_exact(&mut bytes).is_ok()
    {
        return bytes.iter().map(|b| format!("{:02x}", b)).collect();
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
    let configured_token = std::env::var("HOLON_TOKEN")
        .ok()
        .and_then(|t| if t.trim().is_empty() { None } else { Some(t) });

    let addr = format!("{}:{}", listen, port);
    let cycle_interval = std::time::Duration::from_micros(1_000_000 / cycle_hz as u64);
    #[cfg(feature = "holon-quic")]
    let quic_port: u16 = std::env::var("HOLON_QUIC_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| default_quic_port(port));

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

    #[cfg(feature = "holon-quic")]
    let _quic_server = {
        let quic_addr = std::net::SocketAddr::new(
            listen
                .parse()
                .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)),
            quic_port,
        );
        match spawn_holon_quic_server(holon_http_state.clone(), quic_addr) {
            Ok(server) => {
                info!(
                    "Holon QUIC transport active on quic://{}",
                    server.local_addr
                );
                Some(server)
            }
            Err(error) => {
                warn!(%error, "Holon QUIC transport disabled");
                None
            }
        }
    };

    // Wire Iroh P2P — creates NetworkService, initializes Ed25519 attestation,
    // spawns NetworkServiceBridge so SwarmEvents flow into the CLS each cycle.
    // Feature-gated: only compiled when both `identity` and `swarm` are enabled.
    #[cfg(all(feature = "identity", feature = "swarm"))]
    {
        match cls.enable_network_attestation().await {
            Ok(()) => {
                info!("Iroh P2P swarm active — Ed25519 attestation enabled");
                holon_http_state
                    .iroh_active
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                // Spawn the inbound connection accept loop.
                // Peers who connect to us are registered and their SwarmEvents
                // flow into the CLS via the NetworkServiceBridge spawned above.
                if let Some(service) = cls.network_service().cloned() {
                    tokio::spawn(service.clone().accept_connections());
                    info!("Iroh: inbound connection listener active");
                    // Log node ID and bootstrap ticket so peers can connect.
                    let node_id = service.node_id();
                    if !node_id.is_empty() {
                        info!("Iroh node ID: {}", node_id);
                    }
                    match service.create_ticket() {
                        Ok(ticket) => info!("Iroh bootstrap ticket (share with peers): {}", ticket),
                        Err(e) => debug!(error = %e, "Iroh ticket not available yet"),
                    }
                }

                // Bootstrap to known peers via SYMTHAEA_BOOTSTRAP_PEERS env var.
                // Format: comma-separated JSON EndpointAddr tickets (as logged above).
                // Example: SYMTHAEA_BOOTSTRAP_PEERS='{"ticket1",...},{"ticket2",...}'
                if let Ok(raw) = std::env::var("SYMTHAEA_BOOTSTRAP_PEERS")
                    && let Some(service) = cls.network_service().cloned()
                {
                    let tickets: Vec<&str> = raw
                        .split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .collect();
                    info!(
                        "Bootstrapping to {} peer(s) from SYMTHAEA_BOOTSTRAP_PEERS",
                        tickets.len()
                    );
                    let mut bootstrap_map: Vec<(String, String)> = Vec::new();
                    for ticket in tickets {
                        match service.connect_to_peer(ticket).await {
                            Ok(info) => {
                                info!(peer = %info.node_id, "Bootstrap connection established");
                                bootstrap_map.push((info.node_id.clone(), ticket.to_string()));
                            }
                            Err(e) => {
                                warn!(error = %e, ticket = %&ticket[..ticket.len().min(32)], "Bootstrap connect failed")
                            }
                        }
                    }
                    // Spawn reconnection loop for bootstrap peers that disconnect
                    if !bootstrap_map.is_empty() {
                        service.spawn_reconnection_loop(bootstrap_map);
                        info!("Bootstrap peer reconnection loop active");
                    }
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

    // Wire Mycelix governance dispatch — connects the consciousness loop's crisis detector
    // to the Mycelix emergency-incidents zome via a bounded sync channel.
    // Feature-gated: requires both `mycelix` (for MycelixBridge) and `safety-agents`
    // (for drain_pending_crisis_events). MockTransport consumer thread logs each dispatch.
    #[cfg(all(feature = "mycelix", feature = "safety-agents"))]
    let mut mycelix_bridge = {
        use symthaea::consciousness::mycelix_bridge::{GovernanceDispatchCommand, MycelixBridge};

        let (dispatch_tx, dispatch_rx) = MycelixBridge::create_governance_channel();
        let mut bridge = MycelixBridge::new("holon-daemon");
        bridge.set_governance_dispatch_tx(dispatch_tx);

        // MockTransport consumer: log each dispatched command.
        // Replace with HolochainTransport in a separate binary for real conductor calls.
        // (serde 1.0.203/1.0.228 conflict prevents holochain_client in this workspace)
        std::thread::spawn(move || {
            for cmd in dispatch_rx {
                let label = match &cmd {
                    GovernanceDispatchCommand::SubmitProposal { correlation_id, .. } => {
                        format!("SubmitProposal(cid={})", correlation_id)
                    }
                    GovernanceDispatchCommand::CastVote { correlation_id, .. } => {
                        format!("CastVote(cid={})", correlation_id)
                    }
                    GovernanceDispatchCommand::QueryActiveProposals => {
                        "QueryActiveProposals".to_string()
                    }
                    GovernanceDispatchCommand::EvaluateAsset { correlation_id, .. } => {
                        format!("EvaluateAsset(cid={})", correlation_id)
                    }
                    GovernanceDispatchCommand::DeclareCrisis {
                        correlation_id,
                        severity,
                        crisis_type,
                        confidence,
                        ..
                    } => {
                        format!(
                            "DeclareCrisis(cid={}, severity={}, type={}, confidence={:.2})",
                            correlation_id, severity, crisis_type, confidence
                        )
                    }
                    GovernanceDispatchCommand::SubmitRoboticsTelemetry {
                        correlation_id,
                        platform,
                        safety_level,
                        consciousness_level,
                        ..
                    } => {
                        format!(
                            "SubmitRoboticsTelemetry(cid={}, platform={}, safety={}, phi={:.2})",
                            correlation_id, platform, safety_level, consciousness_level
                        )
                    }
                };
                tracing::info!("Governance dispatch (MockTransport): {}", label);
            }
        });

        info!("Mycelix governance dispatch active (MockTransport)");
        bridge
    };

    // Wire mesh bridge: DualLayerMesh ↔ CognitivLoop via MeshBridgeHandle/Actor.
    // Feature-gated: only compiled when the `mesh` feature is enabled.
    // The actor runs as an async tokio task; the handle is passed to the blocking loop.
    #[cfg(feature = "mesh")]
    let (mut mesh_bridge_handle, mesh_outbound_rx) = {
        use symthaea::swarm::mesh::{DualLayerMesh, MeshBridgeHandle, MeshReceiver};

        // Derive a stable node_id from our Iroh public key, or fall back to pid-seeded bytes.
        let node_id_bytes: [u8; 32] = cls
            .network_service()
            .and_then(|svc| {
                let hex = svc.node_id();
                if hex.len() >= 64 {
                    let mut bytes = [0u8; 32];
                    for (i, b) in bytes.iter_mut().enumerate() {
                        *b = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap_or(0);
                    }
                    Some(bytes)
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                let mut b = [0u8; 32];
                let pid = std::process::id().to_le_bytes();
                b[..4].copy_from_slice(&pid);
                b
            });

        let (handle, actor) = MeshBridgeHandle::new(64, 128);
        let mesh = DualLayerMesh::new(node_id_bytes);
        let receiver = MeshReceiver::new();
        tokio::spawn(actor.run(mesh, receiver));
        info!("Mesh bridge active — MeshBridgeActor spawned");

        let outbound_rx = cls.take_mesh_outbound_rx();
        (handle, outbound_rx)
    };

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
            if cycle.is_multiple_of(10)
                && let Ok(mut t) = http_state.telemetry_json.lock()
            {
                *t = cls.holon_collective_qol_json();
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

            // Drain civic crisis events and dispatch to Mycelix governance cluster.
            // GovernanceManager (interval 37) queues CivicCrisisEvents in the CLS;
            // we forward them to the MycelixBridge which sends via governance_dispatch_tx.
            #[cfg(all(feature = "mycelix", feature = "safety-agents"))]
            {
                let crisis_events = cls.drain_pending_crisis_events();
                for event in crisis_events {
                    mycelix_bridge.dispatch_crisis(&event);
                    symthaea::api::metrics::global().increment("crisis_dispatches_total");
                }
            }

            // Drain mesh bridge — forward CLS outbound packets and receive inbound wisdom.
            #[cfg(feature = "mesh")]
            {
                use symthaea::cognitive_loop::SwarmEvent;

                // Inbound: wisdom from mesh peers → SwarmEvent for the CLS.
                let inbound = mesh_bridge_handle.drain_inbox();
                if !inbound.is_empty() {
                    let swarm_tx = cls.swarm_event_sender();
                    for pkt in inbound {
                        let peer_id = pkt
                            .source_id
                            .iter()
                            .map(|b| format!("{:02x}", b))
                            .collect::<String>();
                        let _ = swarm_tx.send(SwarmEvent::ConsciousnessUpdate {
                            peer_id,
                            phi: pkt.phi as f64,
                            valence: 0.0,
                            arousal: 0.0,
                        });
                    }
                }

                // Outbound: CLS-generated mesh packets → bridge actor → radio.
                if let Some(ref rx) = mesh_outbound_rx {
                    let mut packets = Vec::new();
                    while let Ok(pkt) = rx.try_recv() {
                        packets.push(pkt);
                    }
                    if !packets.is_empty() {
                        symthaea::api::metrics::global()
                            .add("mesh_packets_sent_total", packets.len() as u64);
                        mesh_bridge_handle.flush_outbox(packets);
                    }
                }
            }

            cycle += 1;
            if cycle.is_multiple_of(1000) {
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
            let Err(e) = result;
            warn!("Consciousness loop ended: {}", e);
        }
        _ = tokio::signal::ctrl_c() => {
            info!("Shutting down Holon daemon...");
        }
    }

    Ok(())
}
