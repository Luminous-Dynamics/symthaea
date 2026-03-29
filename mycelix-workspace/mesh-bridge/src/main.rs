// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Mesh Bridge
//!
//! Standalone binary that runs alongside a Holochain conductor.
//! Polls for new TEND exchanges, food logs, and emergency messages,
//! serializes them compactly, and relays over LoRa or WiFi-direct mesh.
//!
//! Architecture:
//! ```text
//! Conductor ←→ Poller → Serializer → Transport (LoRa / B.A.T.M.A.N.)
//!                                         ↕
//! Conductor ←→ Relay  ← Serializer ← Transport (LoRa / B.A.T.M.A.N.)
//! ```

use mycelix_mesh_bridge::{poller, relay, transport, BridgeMetrics};

use anyhow::Result;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    tracing::info!("Mycelix Mesh Bridge starting...");

    let conductor_url =
        std::env::var("CONDUCTOR_URL").unwrap_or_else(|_| "ws://localhost:8888".into());
    let poll_interval_secs: u64 = std::env::var("POLL_INTERVAL_SECS")
        .unwrap_or_else(|_| "30".into())
        .parse()
        .unwrap_or(30);

    // Validate configuration
    if !conductor_url.starts_with("ws://") && !conductor_url.starts_with("wss://") {
        anyhow::bail!(
            "CONDUCTOR_URL must start with ws:// or wss://, got: {conductor_url}\n\
             Example: CONDUCTOR_URL=ws://localhost:8888"
        );
    }
    if poll_interval_secs == 0 {
        anyhow::bail!("POLL_INTERVAL_SECS must be > 0");
    }

    tracing::info!("Config: conductor={conductor_url}, poll_interval={poll_interval_secs}s");
    if std::env::var("MESH_APP_TOKEN").unwrap_or_default().is_empty() {
        tracing::warn!(
            "MESH_APP_TOKEN not set — conductor auth may fail. \
             Set it to the app token from hApp installation."
        );
    }

    // Shared metrics + cancellation token
    let metrics = BridgeMetrics::new();
    let cancel = CancellationToken::new();

    // Select transport
    let transport = transport::create_transport()?;
    let transport_name = transport.name().to_string();
    tracing::info!("Transport: {}", transport_name);

    // Start poller (conductor → mesh)
    let poller_transport = transport.clone_box();
    let poller_url = conductor_url.clone();
    let poller_cancel = cancel.clone();
    let poller_metrics = metrics.clone();
    let poller_handle = tokio::spawn(async move {
        if let Err(e) = poller::run(
            &poller_url,
            poll_interval_secs,
            poller_transport,
            poller_cancel,
            poller_metrics,
        )
        .await
        {
            tracing::error!("Poller error: {e}");
        }
    });

    // Start relay (mesh → conductor)
    let relay_transport = transport;
    let relay_url = conductor_url;
    let relay_cancel = cancel.clone();
    let relay_metrics = metrics.clone();
    let relay_handle = tokio::spawn(async move {
        if let Err(e) =
            relay::run(&relay_url, relay_transport, relay_cancel, relay_metrics).await
        {
            tracing::error!("Relay error: {e}");
        }
    });

    // Health check HTTP endpoint (optional, for monitoring)
    let health_port: u16 = std::env::var("HEALTH_PORT")
        .unwrap_or_else(|_| "9100".into())
        .parse()
        .unwrap_or(9100);
    let health_metrics = metrics;
    let health_transport_name = transport_name;
    let health_handle = tokio::spawn(async move {
        if let Err(e) = serve_health(health_port, health_metrics, &health_transport_name).await {
            tracing::warn!("Health endpoint failed to start: {e}");
        }
    });

    // Wait for shutdown signal
    tokio::select! {
        _ = poller_handle => tracing::warn!("Poller exited"),
        _ = relay_handle => tracing::warn!("Relay exited"),
        _ = health_handle => tracing::warn!("Health server exited"),
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("Shutting down gracefully...");
            cancel.cancel();
            // Give in-flight operations up to 5 seconds to drain
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            tracing::info!("Shutdown complete.");
        },
    }

    Ok(())
}

/// Minimal HTTP health endpoint — responds to GET /health with JSON status + metrics,
/// and GET /metrics with Prometheus text exposition format.
/// Also sends systemd watchdog pings every 30 seconds if running under systemd.
async fn serve_health(port: u16, metrics: BridgeMetrics, transport_name: &str) -> Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let listener = TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    tracing::info!("Health endpoint listening on :{port}/health and :{port}/metrics");

    // Notify systemd we're ready (Type=notify)
    sd_notify("READY=1");

    // Spawn watchdog ticker if running under systemd
    tokio::spawn(async {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
            sd_notify("WATCHDOG=1");
        }
    });

    loop {
        let (mut socket, _) = listener.accept().await?;

        // Read the first chunk of the HTTP request to determine the path.
        let mut buf = [0u8; 512];
        let n = socket.read(&mut buf).await.unwrap_or(0);
        let request_head = String::from_utf8_lossy(&buf[..n]);
        let request_path = request_head
            .lines()
            .next()
            .unwrap_or("")
            .split_whitespace()
            .nth(1)
            .unwrap_or("/health");

        if request_path.contains("/metrics") {
            // Prometheus text exposition format
            let body_str = metrics.to_prometheus();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body_str.len(),
                body_str
            );
            let _ = socket.write_all(response.as_bytes()).await;
        } else {
            // JSON health response (default for /health and any other path)
            let uptime_secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let body = serde_json::json!({
                "status": "running",
                "service": "mycelix-mesh-bridge",
                "version": env!("CARGO_PKG_VERSION"),
                "transport": transport_name,
                "uptime_check": uptime_secs,
                "metrics": metrics.to_json(),
            });
            let body_str = body.to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body_str.len(),
                body_str
            );
            let _ = socket.write_all(response.as_bytes()).await;
        }
    }
}

/// Send a notification to systemd via $NOTIFY_SOCKET (if available).
/// No-op if not running under systemd.
fn sd_notify(msg: &str) {
    if let Ok(socket_path) = std::env::var("NOTIFY_SOCKET") {
        use std::os::unix::net::UnixDatagram;
        if let Ok(sock) = UnixDatagram::unbound() {
            let _ = sock.send_to(msg.as_bytes(), &socket_path);
        }
    }
}
