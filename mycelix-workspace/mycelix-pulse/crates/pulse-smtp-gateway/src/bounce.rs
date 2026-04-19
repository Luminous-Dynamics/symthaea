// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! RFC 3464 DSN handler — bridges bounces back to Pulse via the zome.
//!
//! Flow:
//!   external MTA → our :25 (RCPT TO matches VERP pattern)
//!   → receiver passes DSN to [`BounceHandler::handle`]
//!   → we VERP-decode the RCPT TO back to the original msg_id
//!   → we parse the DSN body for status + diagnostic
//!   → we call the zome's `receive_bounce` extern to update OutboxEntry.

use crate::parser::IncomingDsn;
use crate::verp::VerpCodec;
use crate::zome::ZomeBridge;

pub struct BounceHandler<B: ZomeBridge> {
    verp: VerpCodec,
    zome: B,
}

impl<B: ZomeBridge> BounceHandler<B> {
    pub fn new(verp: VerpCodec, zome: B) -> Self {
        Self { verp, zome }
    }

    /// Handle one DSN. `verp_recipient` is the RCPT TO from the envelope —
    /// this is what the external MTA bounced to.
    pub async fn handle(&self, verp_recipient: &str, dsn: IncomingDsn) -> crate::GatewayResult<()> {
        // Recover the original msg_id from VERP. Reject forged/unknown.
        let msg_id_bytes = self.verp.decode(verp_recipient)?;
        let msg_id = String::from_utf8(msg_id_bytes)
            .map_err(|e| crate::GatewayError::Verp(format!("msg_id not UTF-8: {}", e)))?;

        let status = dsn.status.clone().unwrap_or_else(|| "5.0.0".into());
        let diagnostic = dsn.diagnostic.clone().unwrap_or_default();

        // Log for operator visibility.
        tracing::info!(
            msg_id = %msg_id,
            status = %status,
            diagnostic = %diagnostic,
            "bounce received, forwarding to zome"
        );

        self.zome
            .receive_bounce(&msg_id, &status, &diagnostic)
            .await
    }
}
