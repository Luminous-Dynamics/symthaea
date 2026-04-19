// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Binary entry for the Pulse SMTP gateway.
//!
//! Usage:
//!     pulse-smtp-gateway --config /etc/pulse-gateway/config.toml
//!
//! NixOS module writes the config file via `environment.etc` and runs this
//! under systemd with `LoadCredential=` for secrets. Phase 5A nixosTest
//! invokes the same binary with a test config.

use pulse_smtp_gateway::{GatewayConfig, GatewayResult};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Debug)]
struct Args {
    config: PathBuf,
}

fn parse_args() -> anyhow::Result<Args> {
    let mut args = std::env::args().skip(1);
    let mut config = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" | "-c" => {
                config = args
                    .next()
                    .map(PathBuf::from)
                    .ok_or_else(|| anyhow::anyhow!("--config expects a path"))?
                    .into();
            }
            "--help" | "-h" => {
                println!("pulse-smtp-gateway --config <path>");
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown arg: {}", other),
        }
    }
    Ok(Args {
        config: config.ok_or_else(|| anyhow::anyhow!("--config is required"))?,
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // JSON-structured logs so NixOS journalctl can parse them.
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let args = parse_args()?;
    let cfg: GatewayConfig = GatewayConfig::from_path(&args.config)?;

    tracing::info!(
        hostname = %cfg.domain.hostname,
        domain = %cfg.domain.name,
        port = cfg.listener.port,
        "pulse-smtp-gateway starting"
    );

    // Phase 5A: we wire up the stub zome bridge and the rate-limiter,
    // leave the SMTP listener as a TODO because mailin-embedded's threaded
    // server integrates differently from tokio — it's wrapped in
    // spawn_blocking in the binary init sequence (Phase 5B).
    //
    // The purpose of this binary right now is to prove the crate links
    // and the config round-trips. The nixosTest in
    // `tests/pulse-gateway-e2e.nix` exercises the full pipeline via the
    // library surface directly, skipping the TCP layer.
    //
    // Phase 5B TODOs (tracked in PULSE_READINESS_PLAN.md §5.3):
    //   - Start `mailin-embedded::Server` on cfg.listener
    //   - Spawn outbound poller subscribed to zome signals
    //   - Wire up rspamd integration
    //   - Hook up the real InboundPipeline impl

    let _ = boot(&cfg).await?;

    // Idle loop — in Phase 5B this will be `tokio::signal::ctrl_c()` awaiting
    // shutdown while the SMTP server runs in spawn_blocking.
    tracing::info!("gateway initialised; awaiting shutdown signal");
    tokio::signal::ctrl_c().await?;
    tracing::info!("shutdown signal received; goodbye");
    Ok(())
}

/// Instantiate every gateway subsystem from `cfg`. Returns a handle the
/// eventual Phase 5B supervisor will use to orchestrate them.
///
/// For now this is a dry-run that proves everything constructs cleanly.
async fn boot(cfg: &GatewayConfig) -> GatewayResult<GatewayHandle> {
    use pulse_smtp_gateway::rate_limit::PerIpLimiter;
    use pulse_smtp_gateway::verp::VerpCodec;
    use pulse_smtp_gateway::zome::StubZomeBridge;

    let _limiter = PerIpLimiter::new(&cfg.rate_limit)?;

    let hmac_key = std::fs::read(&cfg.verp.hmac_secret_path).map_err(|e| {
        pulse_smtp_gateway::GatewayError::Config(format!(
            "VERP HMAC secret at {}: {}",
            cfg.verp.hmac_secret_path.display(),
            e
        ))
    })?;
    let _verp = VerpCodec::new(hmac_key, cfg.verp.prefix.clone(), cfg.domain.name.clone());

    let _zome = StubZomeBridge::new();
    tracing::info!("subsystems constructed: limiter, verp codec, zome stub");
    Ok(GatewayHandle {})
}

struct GatewayHandle {}
