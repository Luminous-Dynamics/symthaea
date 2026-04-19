// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Gateway error types.
//!
//! Distinct enum for the inbound pipeline — each variant maps to a specific
//! SMTP reply code so the receiver's `data_end` can produce the right
//! response without guessing.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum GatewayError {
    #[error("config error: {0}")]
    Config(String),

    #[error("DNS resolver error: {0}")]
    Dns(String),

    #[error("mail-auth verification failed: {0}")]
    Authentication(String),

    #[error("RFC 5322 parse error: {0}")]
    Parse(String),

    #[error("alias resolution failed for {0}")]
    AliasNotFound(String),

    #[error("rspamd classified as reject: {0}")]
    RspamdReject(String),

    #[error("rate limit exceeded for {0}")]
    RateLimit(String),

    #[error("idempotency cache hit (duplicate Message-ID from same sender)")]
    Duplicate,

    #[error("zome call failed: {0}")]
    Zome(String),

    #[error("DKIM signing failed: {0}")]
    DkimSign(String),

    #[error("outbound SMTP failed: {0}")]
    OutboundSmtp(String),

    #[error("VERP encoding error: {0}")]
    Verp(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl GatewayError {
    /// Map to an SMTP reply code for the receiver. 5xx is permanent, 4xx is
    /// temporary (defer + retry). Used by the SMTP handler to answer
    /// "should the sender retry this message?"
    pub fn smtp_reply_code(&self) -> u16 {
        match self {
            Self::Authentication(_) => 550, // policy: reject
            Self::RspamdReject(_) => 550,   // spam
            Self::RateLimit(_) => 451,      // temporary: try later
            Self::AliasNotFound(_) => 550,  // no such user
            Self::Duplicate => 250,         // pretend-success: already delivered
            Self::Zome(_) => 451,           // temporary: DHT write may recover
            Self::Parse(_) => 550,          // malformed, permanent
            Self::DkimSign(_) | Self::OutboundSmtp(_) | Self::Verp(_) => 451,
            Self::Config(_) | Self::Dns(_) | Self::Io(_) | Self::Other(_) => 451,
        }
    }

    /// Short status text sent to the SMTP peer. Intentionally terse — DSNs
    /// get parsed by downstream software, not humans.
    pub fn smtp_reply_text(&self) -> &'static str {
        match self {
            Self::Authentication(_) => "5.7.1 authentication failed",
            Self::RspamdReject(_) => "5.7.1 spam",
            Self::RateLimit(_) => "4.7.0 rate limit",
            Self::AliasNotFound(_) => "5.1.1 no such user",
            Self::Duplicate => "2.0.0 accepted (duplicate)",
            Self::Zome(_) => "4.0.0 DHT write failed, try again",
            Self::Parse(_) => "5.6.0 malformed message",
            _ => "4.0.0 try again",
        }
    }
}

pub type GatewayResult<T> = Result<T, GatewayError>;
