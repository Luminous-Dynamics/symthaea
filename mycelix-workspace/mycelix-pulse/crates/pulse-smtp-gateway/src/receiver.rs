// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! SMTP receiver — `mailin-embedded::Handler` implementation.
//!
//! Handles the SMTP state machine: HELO → MAIL FROM → RCPT TO → DATA →
//! body bytes. On `data_end`, runs the full inbound pipeline
//! (auth → rspamd → parse → alias → re-encrypt → zome write) and
//! returns an appropriate SMTP reply.
//!
//! Runs inside `mailin-embedded`'s blocking I/O model, which we wrap
//! with `tokio::task::spawn_blocking` from main.

use mailin_embedded::{response::*, Handler, Response};
use std::net::IpAddr;

/// Trait for the receiver's downstream pipeline. Abstracts out so we can
/// substitute a mock in unit tests that doesn't need a real auth/rspamd/zome
/// stack.
pub trait InboundPipeline: Clone + Send + Sync + 'static {
    /// Called once the full message body has been received. Implementations
    /// do all the work — auth, alias, re-encrypt, zome write — and return
    /// the SMTP reply code + text.
    fn handle_message(
        &self,
        client_ip: IpAddr,
        helo: &str,
        mail_from: &str,
        rcpt_to: &[String],
        raw: &[u8],
    ) -> Response;
}

/// `mailin-embedded` Handler that delegates all interesting logic to an
/// `InboundPipeline`. Holds per-connection state (peer IP, current
/// envelope, accumulating DATA buffer).
#[derive(Clone)]
pub struct SmtpReceiver<P: InboundPipeline> {
    pipeline: P,
    state: ConnectionState,
}

#[derive(Clone, Default)]
struct ConnectionState {
    peer_ip: Option<IpAddr>,
    helo: String,
    mail_from: String,
    rcpt_to: Vec<String>,
    data: Vec<u8>,
}

impl<P: InboundPipeline> SmtpReceiver<P> {
    pub fn new(pipeline: P) -> Self {
        Self {
            pipeline,
            state: ConnectionState::default(),
        }
    }

    fn reset_envelope(&mut self) {
        self.state.mail_from.clear();
        self.state.rcpt_to.clear();
        self.state.data.clear();
    }
}

impl<P: InboundPipeline> Handler for SmtpReceiver<P> {
    fn helo(&mut self, ip: std::net::IpAddr, domain: &str) -> Response {
        self.state.peer_ip = Some(ip);
        self.state.helo = domain.to_string();
        OK
    }

    fn mail(&mut self, _ip: std::net::IpAddr, _domain: &str, from: &str) -> Response {
        self.reset_envelope();
        self.state.mail_from = from.to_string();
        OK
    }

    fn rcpt(&mut self, to: &str) -> Response {
        self.state.rcpt_to.push(to.to_string());
        OK
    }

    fn data_start(
        &mut self,
        _domain: &str,
        _from: &str,
        _is8bit: bool,
        _to: &[String],
    ) -> Response {
        self.state.data.clear();
        OK
    }

    fn data(&mut self, buf: &[u8]) -> Result<(), std::io::Error> {
        self.state.data.extend_from_slice(buf);
        Ok(())
    }

    fn data_end(&mut self) -> Response {
        let peer_ip = match self.state.peer_ip {
            Some(ip) => ip,
            None => return INTERNAL_ERROR,
        };
        let helo = self.state.helo.clone();
        let mail_from = self.state.mail_from.clone();
        let rcpt_to = self.state.rcpt_to.clone();
        let data = std::mem::take(&mut self.state.data);

        let resp = self
            .pipeline
            .handle_message(peer_ip, &helo, &mail_from, &rcpt_to, &data);

        self.reset_envelope();
        resp
    }
}

/// Convenience constructor returning a reply with a custom code/message.
pub fn reply(code: u16, message: &str) -> Response {
    Response::custom(code, message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct DummyPipeline;
    impl InboundPipeline for DummyPipeline {
        fn handle_message(
            &self,
            _client_ip: IpAddr,
            _helo: &str,
            _mail_from: &str,
            _rcpt_to: &[String],
            _raw: &[u8],
        ) -> Response {
            OK
        }
    }

    #[test]
    fn receiver_accepts_minimal_envelope() {
        let mut r = SmtpReceiver::new(DummyPipeline);
        let ip: IpAddr = "198.51.100.1".parse().unwrap();
        let _ = r.helo(ip, "example.org");
        let _ = r.mail(ip, "example.org", "alice@example.org");
        let _ = r.rcpt("alice@mycelix.net");
        let _ = r.data_start(
            "example.org",
            "alice@example.org",
            false,
            &["alice@mycelix.net".into()],
        );
        let _ = r.data(b"Subject: hi\r\n\r\nhello\r\n");
        let resp = r.data_end();
        // 2xx code — DummyPipeline returns OK which is 250.
        assert!(format!("{:?}", resp).contains("250"));
    }
}
