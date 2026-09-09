// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transport/crypto-neutral authenticated machine-session evidence.
//!
//! Cryptographic verification belongs to Xenia (or another session provider). Maritime core
//! consumes only the post-verification facts needed to bound authority and fail closed when
//! identity, freshness, epoch, time trust or revocation state is unacceptable.

use serde::{Deserialize, Serialize};

/// Immutable claims handed to maritime core **after** an external secure-session provider
/// verifies its own evidence.
///
/// This type deliberately does not implement `Deserialize`: raw JSON is not authentication.
/// A provider adapter must first parse and verify its native evidence, then explicitly cross
/// the trust boundary through [`AuthenticatedMachineSession::from_verified_provider`].
///
/// Live revocation is also deliberately absent: a session may be valid when authenticated and
/// revoked one millisecond later. Current revocation state belongs in [`MachineSessionContext`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AuthenticatedMachineSession {
    schema: String,
    session_id: String,
    peer_identity_binding: String,
    authenticated_at_ms: u64,
    expires_at_ms: u64,
    authority_epoch: u64,
    evidence_binding: String,
}

impl AuthenticatedMachineSession {
    /// Cross the provider-verification boundary into maritime core.
    ///
    /// Calling this function is an explicit assertion by the adapter that the provider-native
    /// cryptographic/session verification has already succeeded. Maritime core validates the
    /// portable claim shape but intentionally does not duplicate provider cryptography.
    pub fn from_verified_provider(
        schema: impl Into<String>,
        session_id: impl Into<String>,
        peer_identity_binding: impl Into<String>,
        authenticated_at_ms: u64,
        expires_at_ms: u64,
        authority_epoch: u64,
        evidence_binding: impl Into<String>,
    ) -> Result<Self, &'static str> {
        let session = Self {
            schema: schema.into(),
            session_id: session_id.into(),
            peer_identity_binding: peer_identity_binding.into(),
            authenticated_at_ms,
            expires_at_ms,
            authority_epoch,
            evidence_binding: evidence_binding.into(),
        };
        session.validate_shape()?;
        Ok(session)
    }

    pub fn schema(&self) -> &str {
        &self.schema
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn peer_identity_binding(&self) -> &str {
        &self.peer_identity_binding
    }

    pub const fn authenticated_at_ms(&self) -> u64 {
        self.authenticated_at_ms
    }

    pub const fn expires_at_ms(&self) -> u64 {
        self.expires_at_ms
    }

    pub const fn authority_epoch(&self) -> u64 {
        self.authority_epoch
    }

    pub fn evidence_binding(&self) -> &str {
        &self.evidence_binding
    }

    pub fn validate_shape(&self) -> Result<(), &'static str> {
        if self.schema.trim().is_empty() {
            return Err("schema must not be empty");
        }
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

    fn validity_ms(&self) -> Option<u64> {
        self.expires_at_ms.checked_sub(self.authenticated_at_ms)
    }
}

/// Local acceptance policy for provider session evidence.
///
/// This policy is deliberately not serialized with provider evidence. The consumer decides
/// which exact provider schemas it understands and how long any one authenticated session may
/// remain usable, preventing silent acceptance of a future schema with changed semantics or an
/// otherwise-valid record with an excessive validity horizon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MachineSessionPolicy<'a> {
    /// Exact provider-owned schemas accepted by this consumer build/deployment.
    pub accepted_schemas: &'a [&'a str],
    /// Maximum allowed `expires_at_ms - authenticated_at_ms`. Zero denies every session.
    pub max_validity_ms: u64,
}

impl MachineSessionPolicy<'_> {
    fn accepts_schema(&self, schema: &str) -> bool {
        self.accepted_schemas
            .iter()
            .any(|accepted| *accepted == schema)
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
    UnsupportedSchema,
    ValidityTooLong,
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
    policy: MachineSessionPolicy<'_>,
) -> MachineSessionTrust {
    if session.validate_shape().is_err() {
        return MachineSessionTrust::Malformed;
    }
    if !policy.accepts_schema(session.schema()) {
        return MachineSessionTrust::UnsupportedSchema;
    }
    if session
        .validity_ms()
        .is_none_or(|validity| validity > policy.max_validity_ms)
    {
        return MachineSessionTrust::ValidityTooLong;
    }
    if !context.trusted_time_available {
        return MachineSessionTrust::UntrustedTime;
    }
    if context.revoked {
        return MachineSessionTrust::Revoked;
    }
    if session.authority_epoch() != context.authority_epoch {
        return MachineSessionTrust::EpochMismatch;
    }
    if context.now_ms < session.authenticated_at_ms() {
        return MachineSessionTrust::NotYetValid;
    }
    if context.now_ms >= session.expires_at_ms() {
        return MachineSessionTrust::Expired;
    }
    MachineSessionTrust::Trusted
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SCHEMA: &str = "test-session-evidence-v1";

    fn session() -> AuthenticatedMachineSession {
        AuthenticatedMachineSession::from_verified_provider(
            TEST_SCHEMA,
            "session-1",
            "xenia-fingerprint:abc",
            100,
            200,
            9,
            "xenia-transcript:xyz",
        )
        .unwrap()
    }

    fn context(now_ms: u64) -> MachineSessionContext {
        MachineSessionContext {
            now_ms,
            authority_epoch: 9,
            trusted_time_available: true,
            revoked: false,
        }
    }

    fn policy() -> MachineSessionPolicy<'static> {
        MachineSessionPolicy {
            accepted_schemas: &[TEST_SCHEMA],
            max_validity_ms: 100,
        }
    }

    #[test]
    fn provider_handoff_rejects_malformed_claim_shape() {
        assert!(
            AuthenticatedMachineSession::from_verified_provider(
                "",
                "session-1",
                "binding",
                100,
                200,
                9,
                "evidence"
            )
            .is_err()
        );
        assert!(
            AuthenticatedMachineSession::from_verified_provider(
                TEST_SCHEMA,
                "session-1",
                "binding",
                200,
                200,
                9,
                "evidence"
            )
            .is_err()
        );
    }

    #[test]
    fn valid_session_is_trusted_only_in_matching_policy_time_and_epoch_context() {
        assert_eq!(
            evaluate_machine_session(&session(), context(150), policy()),
            MachineSessionTrust::Trusted
        );
    }

    #[test]
    fn provider_schema_must_be_explicitly_accepted() {
        let future = AuthenticatedMachineSession::from_verified_provider(
            "test-session-evidence-v2",
            "session-1",
            "xenia-fingerprint:abc",
            100,
            200,
            9,
            "xenia-transcript:xyz",
        )
        .unwrap();
        assert_eq!(
            evaluate_machine_session(&future, context(150), policy()),
            MachineSessionTrust::UnsupportedSchema
        );
    }

    #[test]
    fn excessive_validity_horizon_fails_closed() {
        let too_long = AuthenticatedMachineSession::from_verified_provider(
            TEST_SCHEMA,
            "session-1",
            "xenia-fingerprint:abc",
            100,
            201,
            9,
            "xenia-transcript:xyz",
        )
        .unwrap();
        assert_eq!(
            evaluate_machine_session(&too_long, context(150), policy()),
            MachineSessionTrust::ValidityTooLong
        );
    }

    #[test]
    fn trusted_time_loss_fails_closed() {
        let mut current = context(150);
        current.trusted_time_available = false;
        assert_eq!(
            evaluate_machine_session(&session(), current, policy()),
            MachineSessionTrust::UntrustedTime
        );
    }

    #[test]
    fn live_revocation_dominates_immutable_session_evidence() {
        let issued = session();
        assert_eq!(
            evaluate_machine_session(&issued, context(150), policy()),
            MachineSessionTrust::Trusted
        );

        let mut current = context(150);
        current.revoked = true;
        assert_eq!(
            evaluate_machine_session(&issued, current, policy()),
            MachineSessionTrust::Revoked
        );
    }

    #[test]
    fn stale_epoch_and_expiry_are_rejected() {
        let mut stale = context(150);
        stale.authority_epoch = 10;
        assert_eq!(
            evaluate_machine_session(&session(), stale, policy()),
            MachineSessionTrust::EpochMismatch
        );
        assert_eq!(
            evaluate_machine_session(&session(), context(200), policy()),
            MachineSessionTrust::Expired
        );
    }

    #[test]
    fn future_dated_session_is_rejected() {
        assert_eq!(
            evaluate_machine_session(&session(), context(99), policy()),
            MachineSessionTrust::NotYetValid
        );
    }
}
