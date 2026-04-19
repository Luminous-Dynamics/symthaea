// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bridge between the gateway and the Holochain conductor.
//!
//! The trait [`ZomeBridge`] is feature-gate-agnostic. Phase 5A test builds
//! use [`StubZomeBridge`] which logs + accepts everything, enabling the
//! full nixosTest pipeline without a real conductor. Phase 5B production
//! enables the `holochain-bridge` Cargo feature to link against
//! `holochain_client` and talk to a real conductor via WebSocket.

use async_trait::async_trait;

/// The zome surface the gateway uses. Four externs map to concrete
/// functions on the mail-bridge + messages coordinator zomes:
///
///   - `receive_external` → `mail_messages.receive_external_email`
///   - `subscribe_outbox` → zome signal stream (polled externally)
///   - `update_outbox_status` → `mail_messages.update_outbox_status`
///   - `receive_bounce` → `mail_messages.receive_bounce`
///
/// (These externs are Phase 5B additions to the messages zome — not yet
/// implemented; the zome functions are named in the plan for when we
/// land them.)
#[async_trait]
pub trait ZomeBridge: Send + Sync {
    /// Persist an externally-received email, addressed to the Mycelix
    /// user whose DID matches `recipient_did`. The gateway has already
    /// re-encrypted the body to the recipient's current-epoch Kyber
    /// pubkey before this call.
    async fn receive_external(
        &self,
        recipient_did: &str,
        from_external: &str,
        subject: &str,
        encrypted_body: &[u8],
        message_id: &str,
    ) -> crate::GatewayResult<()>;

    /// Update status of an OutboxEntry that the gateway attempted to
    /// deliver externally. `status` is one of "Sending", "Delivered",
    /// "Failed".
    async fn update_outbox_status(
        &self,
        message_id: &str,
        status: &str,
        diagnostic: Option<&str>,
    ) -> crate::GatewayResult<()>;

    /// A bounce has been received for our outbound message. Look up
    /// the original OutboxEntry by msg_id and set its status to Failed
    /// with the diagnostic.
    async fn receive_bounce(
        &self,
        message_id: &str,
        dsn_status: &str,
        diagnostic: &str,
    ) -> crate::GatewayResult<()>;

    /// Resolve a human-readable alias (e.g. `alice@mycelix.net`) to a
    /// DID by calling the identity cluster's `alias_registry` zome via
    /// the mail-bridge.
    async fn resolve_alias(&self, alias: &str) -> crate::GatewayResult<Option<String>>;
}

// ----------------------------------------------------------------------------
// Phase 5A stub — logs and returns success. Works without a conductor.
// ----------------------------------------------------------------------------

use tokio::sync::Mutex;

/// Test/development stub. Captures all inbound operations in memory so
/// `nixosTest` integration tests can assert on them.
#[derive(Default)]
pub struct StubZomeBridge {
    pub received: Mutex<Vec<StubRx>>,
    pub outbox_updates: Mutex<Vec<StubOutboxUpdate>>,
    pub bounces: Mutex<Vec<StubBounce>>,
    pub alias_map: Mutex<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Clone)]
pub struct StubRx {
    pub recipient_did: String,
    pub from_external: String,
    pub subject: String,
    pub encrypted_body: Vec<u8>,
    pub message_id: String,
}

#[derive(Debug, Clone)]
pub struct StubOutboxUpdate {
    pub message_id: String,
    pub status: String,
    pub diagnostic: Option<String>,
}

#[derive(Debug, Clone)]
pub struct StubBounce {
    pub message_id: String,
    pub dsn_status: String,
    pub diagnostic: String,
}

impl StubZomeBridge {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-populate alias → DID mapping for tests.
    pub async fn set_alias(&self, alias: impl Into<String>, did: impl Into<String>) {
        self.alias_map.lock().await.insert(alias.into(), did.into());
    }
}

#[async_trait]
impl ZomeBridge for StubZomeBridge {
    async fn receive_external(
        &self,
        recipient_did: &str,
        from_external: &str,
        subject: &str,
        encrypted_body: &[u8],
        message_id: &str,
    ) -> crate::GatewayResult<()> {
        tracing::info!(%recipient_did, %from_external, %message_id, "stub receive_external");
        self.received.lock().await.push(StubRx {
            recipient_did: recipient_did.to_string(),
            from_external: from_external.to_string(),
            subject: subject.to_string(),
            encrypted_body: encrypted_body.to_vec(),
            message_id: message_id.to_string(),
        });
        Ok(())
    }

    async fn update_outbox_status(
        &self,
        message_id: &str,
        status: &str,
        diagnostic: Option<&str>,
    ) -> crate::GatewayResult<()> {
        self.outbox_updates.lock().await.push(StubOutboxUpdate {
            message_id: message_id.to_string(),
            status: status.to_string(),
            diagnostic: diagnostic.map(str::to_string),
        });
        Ok(())
    }

    async fn receive_bounce(
        &self,
        message_id: &str,
        dsn_status: &str,
        diagnostic: &str,
    ) -> crate::GatewayResult<()> {
        self.bounces.lock().await.push(StubBounce {
            message_id: message_id.to_string(),
            dsn_status: dsn_status.to_string(),
            diagnostic: diagnostic.to_string(),
        });
        Ok(())
    }

    async fn resolve_alias(&self, alias: &str) -> crate::GatewayResult<Option<String>> {
        Ok(self.alias_map.lock().await.get(alias).cloned())
    }
}

// ----------------------------------------------------------------------------
// Phase 5B real holochain_client impl — behind feature flag.
// ----------------------------------------------------------------------------

#[cfg(feature = "holochain-bridge")]
pub mod real {
    // Wires to `holochain_client` version that targets Holochain 0.6
    // conductor. The actual implementation is Phase 5B work — we leave
    // the module present but unimplemented so that compile-checking the
    // crate under `--features holochain-bridge` fails loudly the moment
    // someone tries to use it without the wiring.
    //
    // Until Phase 5B, deploy with the default feature set (no
    // holochain-bridge) and use a separate gateway-to-conductor shim
    // or the StubZomeBridge.
}
