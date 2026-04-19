// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Real [`InboundPipeline`] implementation wiring receiver → auth → parser
//! → alias → zome.
//!
//! This is what `main.rs` attaches to the `mailin-embedded::Server` in Phase
//! 5A. Runs inside `mailin-embedded`'s blocking thread pool, so the body
//! here is synchronous + uses `tokio::runtime::Handle::current().block_on`
//! to call async code.

use crate::parser::{parse_incoming, ParsedMessage};
use crate::rate_limit::PerIpLimiter;
use crate::receiver::InboundPipeline;
use crate::zome::ZomeBridge;
use mailin_embedded::response::{INTERNAL_ERROR, OK};
use mailin_embedded::Response;
use std::net::IpAddr;
use std::sync::Arc;

/// Production / VM-test pipeline. Takes ownership of the rate limiter and
/// the zome bridge; clones cheaply because both are Arc-wrapped internally
/// or Clone-bearing.
pub struct RealInboundPipeline<B: ZomeBridge + 'static> {
    inner: Arc<PipelineInner<B>>,
}

struct PipelineInner<B: ZomeBridge + 'static> {
    limiter: PerIpLimiter,
    zome: B,
    our_domain: String,
    runtime: tokio::runtime::Handle,
}

impl<B: ZomeBridge + 'static> Clone for RealInboundPipeline<B> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<B: ZomeBridge + 'static> RealInboundPipeline<B> {
    pub fn new(
        limiter: PerIpLimiter,
        zome: B,
        our_domain: String,
        runtime: tokio::runtime::Handle,
    ) -> Self {
        Self {
            inner: Arc::new(PipelineInner {
                limiter,
                zome,
                our_domain,
                runtime,
            }),
        }
    }

    /// Core inbound work — runs on the mailin-embedded worker thread but
    /// bridges to the tokio runtime for async zome calls.
    fn handle_sync(
        &self,
        client_ip: IpAddr,
        _helo: &str,
        mail_from: &str,
        rcpt_to: &[String],
        raw: &[u8],
    ) -> Response {
        // 1. Rate limit check. Pre-rspamd cheap cutoff.
        if let Err(e) = self.inner.limiter.check(client_ip) {
            tracing::warn!(%client_ip, error = %e, "rate limited");
            return mailin_embedded::Response::custom(451, "4.7.0 rate limit".into());
        }

        // 2. Parse RFC 5322 / RFC 3464.
        let parsed = match parse_incoming(raw) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(%client_ip, error = %e, "RFC 5322 parse failed");
                return mailin_embedded::Response::custom(550, "5.6.0 malformed message".into());
            }
        };

        match parsed {
            ParsedMessage::Dsn(dsn) => self.handle_dsn_sync(rcpt_to, dsn),
            ParsedMessage::Mail(mail) => self.handle_mail_sync(client_ip, mail_from, rcpt_to, mail),
        }
    }

    fn handle_mail_sync(
        &self,
        client_ip: IpAddr,
        mail_from: &str,
        rcpt_to: &[String],
        mail: crate::parser::IncomingMail,
    ) -> Response {
        let our_domain = &self.inner.our_domain;

        // For each recipient addressed to our domain, resolve alias →
        // DID, persist via zome.
        let mut accepted = 0usize;
        let mut unknown = Vec::new();

        for rcpt in rcpt_to {
            let (local, domain) = match rcpt.rsplit_once('@') {
                Some(p) => p,
                None => {
                    unknown.push(rcpt.clone());
                    continue;
                }
            };
            if !domain.eq_ignore_ascii_case(our_domain) {
                // Not our domain; silently drop (we don't relay for others).
                unknown.push(rcpt.clone());
                continue;
            }

            let alias = format!("{}@{}", local, domain);
            let resolve_fut = self.inner.zome.resolve_alias(&alias);
            let did_opt = match self.inner.runtime.block_on(resolve_fut) {
                Ok(d) => d,
                Err(e) => {
                    tracing::error!(%alias, error = %e, "alias resolve failed");
                    return mailin_embedded::Response::custom(
                        451,
                        "4.0.0 alias service unavailable, try again".into(),
                    );
                }
            };
            let did = match did_opt {
                Some(d) => d,
                None => {
                    tracing::info!(%alias, "no such alias");
                    unknown.push(rcpt.clone());
                    continue;
                }
            };

            // Persist via zome. Phase 5A: StubZomeBridge captures it.
            // Phase 5B: real zome write + re-encrypt via recipient's
            // current-epoch Kyber pubkey (not implemented yet).
            let msg_id = mail
                .message_id
                .clone()
                .unwrap_or_else(|| format!("ingest-{}-{}", client_ip, accepted));
            let body = mail.body_plain.as_deref().unwrap_or("").as_bytes();

            let persist_fut =
                self.inner
                    .zome
                    .receive_external(&did, mail_from, &mail.subject, body, &msg_id);
            if let Err(e) = self.inner.runtime.block_on(persist_fut) {
                tracing::error!(%alias, error = %e, "zome persist failed");
                return mailin_embedded::Response::custom(
                    451,
                    "4.0.0 DHT write failed, try again".into(),
                );
            }
            accepted += 1;
        }

        if accepted == 0 {
            // None of the recipients were resolvable / ours.
            return mailin_embedded::Response::custom(550, "5.1.1 no such user".into());
        }
        if !unknown.is_empty() {
            tracing::info!(?unknown, "partial accept — some rcpts unknown/foreign");
        }
        OK
    }

    fn handle_dsn_sync(&self, _rcpt_to: &[String], _dsn: crate::parser::IncomingDsn) -> Response {
        // DSN path — Phase 5A just logs and accepts. Phase 5B calls the
        // bounce handler with the VERP decoder.
        tracing::info!("DSN received (Phase 5A: logged, no-op)");
        OK
    }
}

impl<B: ZomeBridge + 'static> InboundPipeline for RealInboundPipeline<B> {
    fn handle_message(
        &self,
        client_ip: IpAddr,
        helo: &str,
        mail_from: &str,
        rcpt_to: &[String],
        raw: &[u8],
    ) -> Response {
        // Defensive catch: if anything panics inside handle_sync, the
        // SMTP connection needs a clean reply, not a dropped connection
        // that leaks.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.handle_sync(client_ip, helo, mail_from, rcpt_to, raw)
        })) {
            Ok(r) => r,
            Err(_) => {
                tracing::error!("inbound pipeline panicked");
                INTERNAL_ERROR
            }
        }
    }
}
