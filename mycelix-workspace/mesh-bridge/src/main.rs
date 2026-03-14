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

mod poller;
mod relay;
mod serializer;
mod transport;

use anyhow::Result;
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

    // Select transport
    let transport = transport::create_transport()?;
    tracing::info!("Transport: {}", transport.name());

    // Start poller (conductor → mesh)
    let poller_transport = transport.clone_box();
    let poller_url = conductor_url.clone();
    let poller_handle = tokio::spawn(async move {
        if let Err(e) = poller::run(&poller_url, poll_interval_secs, poller_transport).await {
            tracing::error!("Poller error: {e}");
        }
    });

    // Start relay (mesh → conductor)
    let relay_transport = transport;
    let relay_url = conductor_url;
    let relay_handle = tokio::spawn(async move {
        if let Err(e) = relay::run(&relay_url, relay_transport).await {
            tracing::error!("Relay error: {e}");
        }
    });

    tokio::select! {
        _ = poller_handle => tracing::warn!("Poller exited"),
        _ = relay_handle => tracing::warn!("Relay exited"),
        _ = tokio::signal::ctrl_c() => tracing::info!("Shutting down..."),
    }

    Ok(())
}
