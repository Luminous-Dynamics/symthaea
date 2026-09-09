// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transport/crypto-neutral authenticated machine-session evidence.
//!
//! Cryptographic verification belongs to Xenia (or another session provider). Maritime core
//! consumes only the post-verification facts needed to bound authority and fail closed when
//! identity, freshness, epoch, time trust or revocation state is unacceptable.

use serde::{Deserialize, Serialize};

/// Immutable evidence produced after an external secure-session verifier succeeds.
///
/// Live revocation is deliberately **not** stored here: a session may be valid when
/// authenticated and revoked one millisecond later. Current revocation state belongs in
/// [`MachineSessionContext`] so point-of-use evaluation cannot accidentally trust a stale
/// issuance-time boolean.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthenticatedMachineSession {
    pub session_id: String,
    /// Opaque stable binding to the authenticated peer identity (for example a Xenia signing
    /// identity fingerprint). Maritime core does not interpret or recompute it.
    pub peer_identity_binding: String,
    pub authenticated_at_ms: u64,
    pub expires_at_ms: u64,
    /// Provider-defined generation of the authority state that admitted this session.
    /// Point-of-use evaluation requires an exact match with the current authority generation.
    pub authority_epoch: u64,
    /// Opaque binding to transcript/signature/verification evidence owned by the provider.
    pub evidence_binding: String,
}

impl AuthenticatedMachineSession {
    pub fn validate_shape(&self) -> Result<(), &'static str> {
        if self.session_id.trim().is_empty() {
            return Err("session_id must not be empty");
        }
        if self.peer_identity_binding.trim().is_empty() {
            return Err("peer_identity_binding must not be empty");
        }
        if self.evidence_binding.trim().is_empty() {
            return Err("evidence_binding must not be empty");
        }
        if self.expires_at_ms <= self.authenticated_at_ms {
            return Err("session must have a positive validity interval");
        }
        Ok(())
    }
}

/// Current trust facts supplied at the point where session-derived authority is used.
///
/// This context is intentionally separate from immutable handshake evidence and deliberately
/// does **not** implement serde serialization. Providers must freshly construct it from their
/// authoritative time, enrollment and revocation state at the point of use; replaying a stale
/// serialized `revoked = false` context must not be a supported integration path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MachineSessionContext {
    pub now_ms: u64,
    pub authority_epoch: u64,
    pub trusted_time_available: bool,
    /// Current revocation result from the authority owner. This must not be an issuance-time
    /// snapshot copied from [`AuthenticatedMachineSession`].
    pub revoked: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MachineSessionTrust {
    Trusted,
    Malformed,
    UntrustedTime,
    NotYetValid,
    Expired,
    EpochMismatch,
    Revoked,
}

/// Evaluate only maritime-visible trust facts. This does not verify any signature, KEM,
/// fingerprint, transcript or certificate; providers must do that before constructing the
/// immutable evidence record and must refresh the live context at point of use.
pub fn evaluate_machine_session(
    session: &AuthenticatedMachineSession,
    context: MachineSessionContext,
) -> MachineSessionTrust {
    if session.validate_shape().is_err() {
        return MachineSessionTrust::Malformed;
    }
    if !context.trusted_time_available {
        return MachineSessionTrust::UntrustedTime;
    }
    if context.revoked {
        return MachineSessionTrust::Revoked;
    }
    if session.authority_epoch != context.authority_epoch {
        return MachineSessionTrust::EpochMismatch;
    }
    if context.now_ms < session.authenticated_at_ms {
        return MachineSessionTrust::NotYetValid;
    }
    if context.now_ms >= session.expires_at_ms {
        return MachineSessionTrust::Expired;
    }
    MachineSessionTrust::Trusted
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> AuthenticatedMachineSession {
        AuthenticatedMachineSession {
            session_id: "session-1".into(),
            peer_identity_binding: "xenia-fingerprint:abc".into(),
            authenticated_at_ms: 100,
            expires_at_ms: 200,
            authority_epoch: 9,
            evidence_binding: "xenia-transcript:xyz".into(),
        }
    }

    fn context(now_ms: u64) -> MachineSessionContext {
        MachineSessionContext {
            now_ms,
            authority_epoch: 9,
            trusted_time_available: true,
            revoked: false,
        }
    }

    #[test]
    fn valid_session_is_trusted_only_in_matching_time_and_epoch_context() {
        assert_eq!(
            evaluate_machine_session(&session(), context(150)),
            MachineSessionTrust::Trusted
        );
    }

    #[test]
    fn trusted_time_loss_fails_closed() {
        let mut current = context(150);
        current.trusted_time_available = false;
        assert_eq!(
            evaluate_machine_session(&session(), current),
            MachineSessionTrust::UntrustedTime
        );
    }

    #[test]
    fn live_revocation_dominates_immutable_session_evidence() {
        let issued = session();
        assert_eq!(
            evaluate_machine_session(&issued, context(150)),
            MachineSessionTrust::Trusted
        );

        let mut current = context(150);
        current.revoked = true;
        assert_eq!(
            evaluate_machine_session(&issued, current),
            MachineSessionTrust::Revoked
        );
    }

    #[test]
    fn stale_epoch_and_expiry_are_rejected() {
        let mut stale = context(150);
        stale.authority_epoch = 10;
        assert_eq!(
            evaluate_machine_session(&session(), stale),
            MachineSessionTrust::EpochMismatch
        );
        assert_eq!(
            evaluate_machine_session(&session(), context(200)),
            MachineSessionTrust::Expired
        );
    }

    #[test]
    fn future_dated_session_is_rejected() {
        assert_eq!(
            evaluate_machine_session(&session(), context(99)),
            MachineSessionTrust::NotYetValid
        );
    }
}
