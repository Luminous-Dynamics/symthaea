// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mycelix Mail IMAP/SMTP Bridge Agent

mod config;
mod conductor;
mod oauth;
mod auth_server;

use anyhow::Result;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new("info,mycelix_mail_bridge=debug")
        }))
        .init();

    tracing::info!("Mycelix Mail Bridge Agent starting...");

    let config = config::load_config()?;
    tracing::info!("Loaded {} external account(s)", config.accounts.len());

    let conductor = conductor::ConductorClient::connect(&config.conductor_url).await?;
    tracing::info!("Connected to conductor at {}", config.conductor_url);

    let mut handles = Vec::new();
    for account in config.accounts.clone() {
        if !account.enabled {
            tracing::info!("Skipping disabled account: {}", account.email);
            continue;
        }

        let conductor = conductor.clone();
        let email = account.email.clone();
        let handle = tokio::spawn(async move {
            loop {
                let acct = account.clone();
                let cond = conductor.clone();
                match tokio::task::spawn_blocking(move || poll_imap_sync(&acct, &cond)).await {
                    Ok(Ok(count)) => {
                        if count > 0 {
                            tracing::info!("[{}] Fetched {} new message(s)", account.email, count);
                        }
                    }
                    Ok(Err(e)) => tracing::warn!("[{}] IMAP poll failed: {}", account.email, e),
                    Err(e) => tracing::error!("[{}] Task panicked: {}", account.email, e),
                }
                tokio::time::sleep(std::time::Duration::from_secs(
                    account.sync_interval_secs as u64
                )).await;
            }
        });
        handles.push(handle);
        tracing::info!("Started IMAP poller for {}", email);
    }

    tracing::info!("Bridge agent running. Press Ctrl+C to stop.");
    tokio::signal::ctrl_c().await?;
    tracing::info!("Shutting down.");
    Ok(())
}

/// Synchronous IMAP polling — runs in spawn_blocking.
fn poll_imap_sync(
    account: &config::AccountConfig,
    conductor: &conductor::ConductorClient,
) -> Result<usize> {
    let tls = native_tls::TlsConnector::new()?;
    let client = imap::connect(
        (&*account.imap_host, account.imap_port),
        &account.imap_host,
        &tls,
    )?;

    let mut session = client.login(&account.username, &account.password)
        .map_err(|e| anyhow::anyhow!("IMAP login failed: {}", e.0))?;

    session.select("INBOX")?;

    // Search for unseen messages
    let unseen = session.uid_search("UNSEEN")?;
    if unseen.is_empty() {
        session.logout().ok();
        return Ok(0);
    }

    let mut new_count = 0;
    let uids: Vec<String> = unseen.iter().map(|u| u.to_string()).collect();
    let uid_range = uids.join(",");

    let messages = session.uid_fetch(&uid_range, "(ENVELOPE BODY[TEXT])")?;

    for msg in messages.iter() {
        if let Some(envelope) = msg.envelope() {
            let subject = envelope.subject
                .map(|s| String::from_utf8_lossy(s).to_string())
                .unwrap_or_default();

            let from = envelope.from.as_ref()
                .and_then(|addrs| addrs.first())
                .map(|a| {
                    let mbox = a.mailbox.map(|m| String::from_utf8_lossy(m).to_string()).unwrap_or_default();
                    let host = a.host.map(|h| String::from_utf8_lossy(h).to_string()).unwrap_or_default();
                    format!("{mbox}@{host}")
                })
                .unwrap_or_else(|| "unknown".into());

            let body = msg.body()
                .map(|b| String::from_utf8_lossy(b).to_string())
                .unwrap_or_default();

            // Relay to conductor (blocking — we're already in spawn_blocking)
            let rt = tokio::runtime::Handle::current();
            if let Err(e) = rt.block_on(conductor.relay_inbound(&from, &subject, &body, &account.email)) {
                tracing::warn!("[{}] Relay failed: {}", account.email, e);
            }

            new_count += 1;
        }
    }

    session.logout().ok();
    Ok(new_count)
}
