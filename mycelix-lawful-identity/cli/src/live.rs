// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Live conductor integration — the lawful-id CLI makes real zome
//! calls against a running Holochain conductor when built with the
//! `conductor` feature and invoked with the `--live` flag.
//!
//! Without the feature, the CLI stays lightweight (no tokio, no
//! holochain_client dep tree) and commands emit the call sheet only.
//! This module is only compiled when the feature is enabled.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;

use anyhow::{Context, Result};
use holochain_client::{
    AdminWebsocket, AgentSigner, AppInfo, AppWebsocket, AuthorizeSigningCredentialsPayload,
    CellInfo, ClientAgentSigner, ExternIO, IssueAppAuthenticationTokenPayload, ZomeCallTarget,
};
use holochain_types::prelude::CellId;

/// The shared ecosystem conductor defaults, per monorepo CLAUDE.md.
/// holochain_client 0.7 takes `impl ToSocketAddrs`, so we use
/// `(Ipv4Addr, u16)` tuples rather than `ws://host:port` URLs.
pub const DEFAULT_ADMIN_PORT: u16 = 33800;
pub const DEFAULT_APP_PORT: u16 = 8888;
pub const DEFAULT_APP_ID: &str = "mycelix-lawful-identity";
pub const DEFAULT_ROLE: &str = "lawful-identity";

pub fn default_admin_addr() -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, DEFAULT_ADMIN_PORT))
}

pub fn default_app_addr() -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, DEFAULT_APP_PORT))
}

/// Handle to a live conductor connection. Wraps the app websocket
/// plus the cell id we're calling into.
pub struct LiveConductor {
    app_ws: AppWebsocket,
    cell_id: CellId,
}

impl LiveConductor {
    /// Connect to the shared ecosystem conductor, authorize signing
    /// credentials, and resolve the cell id. All subsequent calls go
    /// through the returned handle.
    pub async fn connect(admin: SocketAddr, app: SocketAddr, app_id: &str) -> Result<Self> {
        let admin_ws = AdminWebsocket::connect(admin, None)
            .await
            .context("connect admin websocket")?;

        let token_response = admin_ws
            .issue_app_auth_token(IssueAppAuthenticationTokenPayload {
                installed_app_id: app_id.to_string().into(),
                expiry_seconds: 3600,
                single_use: false,
            })
            .await
            .context("issue app auth token")?;

        let signer = ClientAgentSigner::default();

        // Temporarily connect to read app_info so we can resolve the cell_id
        // before authorizing signing credentials.
        let signer_dyn: Arc<dyn AgentSigner + Send + Sync> = Arc::new(signer.clone());
        let bootstrap_ws =
            AppWebsocket::connect(app, token_response.token.clone(), signer_dyn.clone(), None)
                .await
                .context("connect app websocket (bootstrap)")?;
        let app_info: AppInfo = bootstrap_ws
            .app_info()
            .await
            .context("fetch app info")?
            .context("app info not available")?;
        let cell_id = find_cell_id(&app_info, DEFAULT_ROLE)?;
        drop(bootstrap_ws);

        let creds = admin_ws
            .authorize_signing_credentials(AuthorizeSigningCredentialsPayload {
                cell_id: cell_id.clone(),
                functions: None,
            })
            .await
            .context("authorize signing credentials")?;

        signer.add_credentials(cell_id.clone(), creds);
        let signer_dyn: Arc<dyn AgentSigner + Send + Sync> = Arc::new(signer);

        let app_ws = AppWebsocket::connect(app, token_response.token, signer_dyn, None)
            .await
            .context("connect app websocket (authorized)")?;

        Ok(Self { app_ws, cell_id })
    }

    /// Make a zome call and decode the response into `T`.
    async fn call<Out>(
        &self,
        zome: &str,
        func: &str,
        payload_json: serde_json::Value,
    ) -> Result<Out>
    where
        Out: serde::de::DeserializeOwned + std::fmt::Debug,
    {
        let encoded = encode_json_payload(payload_json)?;
        let response = self
            .app_ws
            .call_zome(
                ZomeCallTarget::CellId(self.cell_id.clone()),
                zome.into(),
                func.into(),
                encoded,
            )
            .await
            .map_err(|e| anyhow::anyhow!("zome call {zome}/{func} failed: {e:?}"))?;
        let out: Out = decode_response(&response)
            .with_context(|| format!("decoding response from {zome}/{func}"))?;
        Ok(out)
    }

    /// Call `legal_did.ping`. Cheap liveness check.
    pub async fn ping_legal_did(&self) -> Result<String> {
        self.call("legal_did", "ping", serde_json::Value::Null)
            .await
    }

    /// Call `legal_did.create_legal_did`.
    pub async fn create_legal_did(&self, label: Option<String>) -> Result<CreateLegalDidOutput> {
        self.call(
            "legal_did",
            "create_legal_did",
            serde_json::json!({ "label": label }),
        )
        .await
    }

    /// Call `legal_did.list_my_legal_dids`.
    pub async fn list_my_legal_dids(&self) -> Result<Vec<LegalDidRecord>> {
        self.call("legal_did", "list_my_legal_dids", serde_json::Value::Null)
            .await
    }

    /// Call `issuer_trust_tier.classify_issuer`.
    pub async fn classify_issuer(
        &self,
        issuer_did: &str,
        tier: IssuerTierWire,
        rationale: Option<String>,
    ) -> Result<ClassifyIssuerOutput> {
        self.call(
            "issuer_trust_tier",
            "classify_issuer",
            serde_json::json!({
                "issuer_did": issuer_did,
                "tier": tier.as_str(),
                "rationale": rationale,
            }),
        )
        .await
    }

    /// Call `issuer_trust_tier.lookup_tier`.
    pub async fn lookup_tier(&self, issuer_did: &str) -> Result<Option<serde_json::Value>> {
        self.call(
            "issuer_trust_tier",
            "lookup_tier",
            serde_json::json!({ "issuer_did": issuer_did }),
        )
        .await
    }

    /// Call `cross_did_zkp.request_nonce`.
    pub async fn request_nonce(&self, verifier_did: &str) -> Result<serde_json::Value> {
        self.call(
            "cross_did_zkp",
            "request_nonce",
            serde_json::json!({ "verifier_did": verifier_did }),
        )
        .await
    }
}

#[derive(Clone, Copy, Debug)]
pub enum IssuerTierWire {
    Sovereign,
    RegulatedIntermediary,
    Peer,
}

impl IssuerTierWire {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sovereign => "Sovereign",
            Self::RegulatedIntermediary => "RegulatedIntermediary",
            Self::Peer => "Peer",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_lowercase().as_str() {
            "sovereign" => Self::Sovereign,
            "regulated" | "regulated_intermediary" | "regulatedintermediary" => {
                Self::RegulatedIntermediary
            }
            "peer" => Self::Peer,
            _ => return None,
        })
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct CreateLegalDidOutput {
    pub did: String,
    /// `ActionHash` from holochain_types. The conductor serializes it
    /// as MessagePack bytes (not a JSON value); using the real type
    /// side-steps manual byte handling.
    pub action_hash: holochain_types::prelude::ActionHash,
}

#[derive(Debug, serde::Deserialize)]
pub struct ClassifyIssuerOutput {
    pub action_hash: holochain_types::prelude::ActionHash,
}

#[derive(Debug, serde::Deserialize)]
pub struct LegalDidRecord {
    pub did: String,
    pub created_at: String,
    pub label: Option<String>,
}

fn find_cell_id(app_info: &AppInfo, role: &str) -> Result<CellId> {
    for (role_name, cells) in &app_info.cell_info {
        if role_name != role {
            continue;
        }
        for cell_info in cells {
            if let CellInfo::Provisioned(cell) = cell_info {
                return Ok(cell.cell_id.clone());
            }
        }
    }
    anyhow::bail!("no provisioned cell found for role {role}")
}

/// Encode a JSON payload for a Holochain zome call. The conductor
/// expects MessagePack, not JSON, so we transcode.
fn encode_json_payload(value: serde_json::Value) -> Result<ExternIO> {
    // `ExternIO` is just a newtype over `Vec<u8>` produced by
    // holochain_serialized_bytes::encode on any serializable payload.
    let bytes = ExternIO::encode(value).map_err(|e| anyhow::anyhow!("encode payload: {e}"))?;
    Ok(bytes)
}

/// Decode a zome-call response into `T`.
fn decode_response<T: serde::de::DeserializeOwned + std::fmt::Debug>(io: &ExternIO) -> Result<T> {
    io.decode::<T>()
        .map_err(|e| anyhow::anyhow!("decode response: {e}"))
}
