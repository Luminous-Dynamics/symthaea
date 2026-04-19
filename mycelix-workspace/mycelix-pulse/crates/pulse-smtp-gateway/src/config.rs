// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! TOML configuration schema for the SMTP gateway.
//!
//! Loaded once at startup. All secrets (DKIM private keys, Dilithium signing
//! key, Holochain credentials) are referenced by path — the NixOS module
//! places them in `/run/secrets/pulse-gateway/` via `systemd.services...
//! LoadCredential=`, never inlined in the config file.

use serde::{Deserialize, Serialize};
use std::net::IpAddr;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GatewayConfig {
    pub domain: DomainConfig,
    pub listener: ListenerConfig,
    pub dns: DnsConfig,
    pub dkim: DkimConfig,
    pub rspamd: RspamdConfig,
    pub zome: ZomeConfig,
    pub rate_limit: RateLimitConfig,
    pub verp: VerpConfig,
    /// Pre-populated alias→DID map. Honored ONLY when the StubZomeBridge
    /// is in use (Phase 5A — no real conductor). Phase 5B reads aliases
    /// from the identity cluster's alias_registry zome and ignores this
    /// section even if present. Used by the nixosTest harness to seed
    /// `alice@mycelix.test → did:mycelix:alice` without round-tripping
    /// through a real cluster.
    #[serde(default)]
    pub test_aliases: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DomainConfig {
    /// Our authoritative domain — the one we receive mail for.
    /// Example: `mycelix.net`.
    pub name: String,
    /// Our hostname — what HELO announces and what PTR resolves to.
    /// Example: `mail.mycelix.net`. Must match forward+reverse DNS.
    pub hostname: String,
    /// Postmaster + abuse aliases — RFC 2142 mandatory. Routed to a
    /// specific Mycelix user via their DID.
    pub postmaster_did: String,
    pub abuse_did: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ListenerConfig {
    /// Port 25 in production, 2525 in VM tests (Phase 5A).
    pub port: u16,
    /// Bind address. Typically `0.0.0.0` for production (external MTAs
    /// can reach us), `127.0.0.1` for local testing.
    pub bind: IpAddr,
    /// Maximum message size in bytes (RFC 1870 SIZE extension).
    pub max_size: usize,
    /// TLS certificate path (Let's Encrypt via security.acme, or a test
    /// self-signed cert in Phase 5A).
    pub tls_cert: Option<PathBuf>,
    /// TLS private key path.
    pub tls_key: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DnsConfig {
    /// Upstream resolver — default to Cloudflare TLS for low-latency DNS
    /// lookups during SPF/DKIM/DMARC verification. Can point at a local
    /// unbound instance in Phase 5A nixosTest.
    pub upstream: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DkimConfig {
    /// Path to RSA private key (PKCS#1 PEM). Selector `rsa._domainkey`.
    /// Legacy compatibility — many older receivers still ignore Ed25519.
    pub rsa_key_path: PathBuf,
    /// Path to Ed25519 seed file (32 bytes raw). Selector
    /// `ed25519._domainkey`. Modern DKIM per RFC 8463.
    pub ed25519_key_path: PathBuf,
    /// Headers to sign — RFC 6376 §5.4 recommends at least From, Date,
    /// Subject, To, Message-ID.
    #[serde(default = "default_dkim_headers")]
    pub signed_headers: Vec<String>,
}

fn default_dkim_headers() -> Vec<String> {
    [
        "From",
        "To",
        "Subject",
        "Date",
        "Message-ID",
        "MIME-Version",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RspamdConfig {
    /// Local rspamd endpoint. Production: `http://127.0.0.1:11333`.
    pub endpoint: String,
    /// Accept actions below this threshold; reject at/above.
    /// Rspamd actions: `no action`, `greylist`, `add header`, `rewrite subject`,
    /// `soft reject`, `reject`.
    #[serde(default)]
    pub reject_actions: Vec<String>,
}

impl Default for RspamdConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:11333".into(),
            reject_actions: vec!["reject".into()],
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ZomeConfig {
    /// WebSocket URL of the Holochain conductor's app port.
    /// Production: `ws://localhost:8888`. Phase 5A: pointed at the
    /// conductor VM inside the nixosTest network.
    pub conductor_url: String,
    /// Installed app ID for Mycelix Mail on the conductor.
    #[serde(default = "default_app_id")]
    pub app_id: String,
    /// Gateway's own DID (registered in mycelix-identity under role
    /// MailGateway). Used to sign the `receive_external` entries so
    /// recipients can distinguish gateway-origin from peer-origin mail.
    pub gateway_did: String,
    /// Path to Dilithium3 signing key for the gateway's own DID.
    pub gateway_signing_key_path: PathBuf,
}

fn default_app_id() -> String {
    "mycelix_mail".into()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RateLimitConfig {
    /// Max messages accepted per source IP per minute.
    #[serde(default = "default_per_minute")]
    pub per_ip_per_minute: u32,
    /// Max messages accepted per source IP per hour. Pre-rspamd cheap cutoff.
    #[serde(default = "default_per_hour")]
    pub per_ip_per_hour: u32,
}

fn default_per_minute() -> u32 {
    30
}
fn default_per_hour() -> u32 {
    300
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VerpConfig {
    /// HMAC secret for VERP bounce-ID — prevents forgery of DSN routing.
    /// 32 bytes random, loaded from a secret file. Rotate via
    /// `pulse-gateway-rotate-verp-secret` helper (TODO Phase 5B).
    pub hmac_secret_path: PathBuf,
    /// VERP local-part prefix. Default: `bounce`. Outbound
    /// `MAIL FROM` becomes `<bounce+HMAC+b32(msg_id)@mycelix.net>`.
    #[serde(default = "default_verp_prefix")]
    pub prefix: String,
}

fn default_verp_prefix() -> String {
    "bounce".into()
}

impl GatewayConfig {
    /// Load + validate from a TOML file.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> crate::GatewayResult<Self> {
        let raw = std::fs::read_to_string(path.as_ref())?;
        let cfg: Self = toml::from_str(&raw)
            .map_err(|e| crate::GatewayError::Config(format!("TOML parse: {}", e)))?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> crate::GatewayResult<()> {
        if self.domain.name.is_empty() {
            return Err(crate::GatewayError::Config("domain.name empty".into()));
        }
        if !self.domain.hostname.contains('.') {
            return Err(crate::GatewayError::Config(
                "domain.hostname must be a FQDN".into(),
            ));
        }
        if self.listener.max_size < 1024 {
            return Err(crate::GatewayError::Config(
                "listener.max_size too small (< 1KB)".into(),
            ));
        }
        Ok(())
    }
}
