// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transport/crypto-neutral authenticated machine-session evidence.
//!
//! Cryptographic verification belongs to Xenia (or another session provider). Maritime core
//! consumes only the post-verification facts needed to bound authority and fail closed when
//! identity, freshness, epoch, time trust or revocation state is unacceptable.

use serde::{Deserialize, Serialize};

/// Opaque evidence produced after an external secure-session verifier succeeds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthenticatedMachineSession {
    pub session_id: String,
    /// Opaque stable binding to the authenticated peer identity (for example a Xenia host
    /// identity fingerprint). Maritime core does not interpret or recompute it.
    pub peer_identity_binding: String,
    pub authenticated_at_ms: u64,
    pub expires_at_ms: u64,
    pub authority_epoch: u64,
    /// Opaque binding to transcript/signature/verification evidence owned by the provider.
    pub evidence_binding: String,
    pub revoked: bool,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineSessionContext {
    pub now_ms: u64,
    pub authority_epoch: u64,
    pub trusted_time_available: bool,
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
/// evidence record.
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
    if session.revoked {
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
            revoked: false,
        }
    }

    #[test]
    fn valid_session_is_trusted_only_in_matching_time_and_epoch_context() {
        assert_eq!(
            evaluate_machine_session(
                &session(),
                MachineSessionContext {
                    now_ms: 150,
                    authority_epoch: 9,
                    trusted_time_available: true,
                },
            ),
            MachineSessionTrust::Trusted
        );
    }

    #[test]
    fn trusted_time_loss_fails_closed() {
        assert_eq!(
            evaluate_machine_session(
                &session(),
                MachineSessionContext {
                    now_ms: 150,
                    authority_epoch: 9,
                    trusted_time_available: false,
                },
            ),
            MachineSessionTrust::UntrustedTime
        );
    }

    #[test]
    fn revocation_dominates_other_live_session_facts() {
        let mut revoked = session();
        revoked.revoked = true;
        assert_eq!(
            evaluate_machine_session(
                &revoked,
                MachineSessionContext {
                    now_ms: 150,
                    authority_epoch: 9,
                    trusted_time_available: true,
                },
            ),
            MachineSessionTrust::Revoked
        );
    }

    #[test]
    fn stale_epoch_and_expiry_are_rejected() {
        assert_eq!(
            evaluate_machine_session(
                &session(),
                MachineSessionContext {
                    now_ms: 150,
                    authority_epoch: 10,
                    trusted_time_available: true,
                },
            ),
            MachineSessionTrust::EpochMismatch
        );
        assert_eq!(
            evaluate_machine_session(
                &session(),
                MachineSessionContext {
                    now_ms: 200,
                    authority_epoch: 9,
                    trusted_time_available: true,
                },
            ),
            MachineSessionTrust::Expired
        );
    }
}
