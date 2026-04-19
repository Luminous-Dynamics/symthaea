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

    serve(cfg).await?;
    Ok(())
}

/// Boot all subsystems and run the SMTP listener until `ctrl-c`.
///
/// Phase 5A: uses `StubZomeBridge` — no real conductor needed. A lettre
/// client connecting to the listener will flow through the full pipeline
/// (rate-limit → parse → alias resolve → zome persist) with the stub
/// capturing each receive.
///
/// Phase 5B: swap `StubZomeBridge` for the real `holochain_client` impl
/// behind the `holochain-bridge` feature.
async fn serve(cfg: GatewayConfig) -> GatewayResult<()> {
    use pulse_smtp_gateway::pipeline::RealInboundPipeline;
    use pulse_smtp_gateway::rate_limit::PerIpLimiter;
    use pulse_smtp_gateway::receiver::SmtpReceiver;
    use pulse_smtp_gateway::verp::VerpCodec;
    use pulse_smtp_gateway::zome::StubZomeBridge;

    let limiter = PerIpLimiter::new(&cfg.rate_limit)?;

    // VERP codec is built now even though Phase 5A doesn't yet use it for
    // outbound — fails loudly if the HMAC secret file is missing, so
    // operators find the misconfig at boot instead of on first bounce.
    let hmac_key = std::fs::read(&cfg.verp.hmac_secret_path).map_err(|e| {
        pulse_smtp_gateway::GatewayError::Config(format!(
            "VERP HMAC secret at {}: {}",
            cfg.verp.hmac_secret_path.display(),
            e
        ))
    })?;
    let _verp = VerpCodec::new(hmac_key, cfg.verp.prefix.clone(), cfg.domain.name.clone());

    let zome = StubZomeBridge::new();

    // Pre-populate stub aliases from config — Phase 5A nixosTest path.
    // Phase 5B (real holochain_client) ignores this map entirely.
    for (alias, did) in &cfg.test_aliases {
        zome.set_alias(alias.clone(), did.clone()).await;
        tracing::debug!(%alias, %did, "registered test alias");
    }

    let pipeline = RealInboundPipeline::new(
        limiter,
        zome,
        cfg.domain.name.clone(),
        tokio::runtime::Handle::current(),
    );
    let receiver = SmtpReceiver::new(pipeline);

    let hostname = cfg.domain.hostname.clone();
    let bind_addr = format!("{}:{}", cfg.listener.bind, cfg.listener.port);
    let bind_addr_for_log = bind_addr.clone();

    // mailin-embedded's Server::serve() blocks the calling thread forever.
    // Run it inside spawn_blocking so the tokio runtime remains free to
    // handle the ctrl-c shutdown signal + future async work.
    let server_handle = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let mut server = mailin_embedded::Server::new(receiver);
        server.with_name(hostname);
        server
            .with_addr(bind_addr.as_str())
            .map_err(|e| anyhow::anyhow!("with_addr({}): {}", bind_addr, e))?;
        server
            .serve()
            .map_err(|e| anyhow::anyhow!("server.serve(): {}", e))?;
        Ok(())
    });

    tracing::info!(bind = %bind_addr_for_log, "SMTP listener bound");

    // Race: shutdown signal or server exit (probably panic).
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("shutdown signal received");
        }
        res = server_handle => {
            match res {
                Ok(Ok(())) => tracing::warn!("SMTP server returned cleanly"),
                Ok(Err(e)) => tracing::error!(error = %e, "SMTP server exited with error"),
                Err(e) => tracing::error!(error = %e, "SMTP server task panicked"),
            }
        }
    }

    Ok(())
}
