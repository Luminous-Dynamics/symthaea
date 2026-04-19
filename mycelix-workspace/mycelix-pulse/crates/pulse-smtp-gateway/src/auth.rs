// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Inbound authentication pipeline: SPF, DKIM, DMARC.
//!
//! Wraps `mail-auth`'s `MessageAuthenticator`. Produces a single verdict:
//! accept / quarantine / reject. The receiver's `data_end` hook calls
//! [`InboundAuth::verify_inbound`] and maps the decision to an SMTP reply
//! code via [`crate::error::GatewayError`].

use mail_auth::{
    dmarc::verify::DmarcParameters, spf::verify::SpfParameters, AuthenticatedMessage, DmarcResult,
    MessageAuthenticator,
};
use std::net::IpAddr;

/// Structured verdict of the inbound auth pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthVerdict {
    /// All checks aligned; accept.
    Pass,
    /// DMARC policy `quarantine`; DHT stores in a QuarantineEntry bucket.
    Quarantine { reason: String },
    /// DMARC policy `reject` AND aligned fail; 5xx back to sender.
    Reject { reason: String },
    /// No DMARC policy or policy is `none`; accept with notation.
    NoPolicy,
}

pub struct InboundAuth {
    resolver: MessageAuthenticator,
    /// Our own host domain (for SPF checks' HELO parameter).
    my_hostname: String,
}

impl InboundAuth {
    /// Construct with a DNS-over-TLS resolver pointed at Cloudflare. In
    /// Phase 5A tests, pass an `upstream` pointed at a mock resolver
    /// through [`Self::new_with_resolver`].
    pub fn new_cloudflare_tls(my_hostname: String) -> crate::GatewayResult<Self> {
        let resolver = MessageAuthenticator::new_cloudflare_tls()
            .map_err(|e| crate::GatewayError::Dns(format!("resolver init: {}", e)))?;
        Ok(Self {
            resolver,
            my_hostname,
        })
    }

    /// Construct with an injected authenticator — useful for tests that
    /// supply a mock resolver.
    pub fn new_with_resolver(resolver: MessageAuthenticator, my_hostname: String) -> Self {
        Self {
            resolver,
            my_hostname,
        }
    }

    /// Verify SPF + DKIM + DMARC on a fully-received RFC 5322 message.
    pub async fn verify_inbound(
        &self,
        raw: &[u8],
        client_ip: IpAddr,
        helo_domain: &str,
        mail_from: &str,
    ) -> AuthVerdict {
        let auth_msg = match AuthenticatedMessage::parse(raw) {
            Some(m) => m,
            None => {
                return AuthVerdict::Reject {
                    reason: "RFC 5322 parse failed in auth layer".into(),
                }
            }
        };

        // SPF: verify the MAIL-FROM identity against the sending IP.
        let spf = self
            .resolver
            .verify_spf(SpfParameters::verify_mail_from(
                client_ip,
                helo_domain,
                &self.my_hostname,
                mail_from,
            ))
            .await;

        let dkim = self.resolver.verify_dkim(&auth_msg).await;

        // RFC 5321.MailFrom domain — strip everything before '@'.
        let mail_from_domain = mail_from.rsplit('@').next().unwrap_or(mail_from);

        let dmarc = self
            .resolver
            .verify_dmarc(
                DmarcParameters::new(&auth_msg, &dkim, mail_from_domain, &spf)
                    .with_domain_suffix_fn(|d| psl::domain_str(d).unwrap_or(d)),
            )
            .await;

        // Decision:
        //   DMARC PASS (either aligned SPF or DKIM) → accept
        //   DMARC FAIL + policy reject → reject
        //   DMARC FAIL + policy quarantine → quarantine
        //   Else → accept but flag
        let spf_aligned = dmarc.spf_result() == &DmarcResult::Pass;
        let dkim_aligned = dmarc.dkim_result() == &DmarcResult::Pass;

        if spf_aligned || dkim_aligned {
            return AuthVerdict::Pass;
        }

        match dmarc.policy() {
            mail_auth::dmarc::Policy::Reject => AuthVerdict::Reject {
                reason: format!(
                    "DMARC reject: spf={:?} dkim={:?}",
                    dmarc.spf_result(),
                    dmarc.dkim_result()
                ),
            },
            mail_auth::dmarc::Policy::Quarantine => AuthVerdict::Quarantine {
                reason: "DMARC quarantine".into(),
            },
            _ => AuthVerdict::NoPolicy,
        }
    }
}
