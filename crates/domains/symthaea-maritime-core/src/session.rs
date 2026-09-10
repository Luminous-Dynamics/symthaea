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
    /// Calling this function is an explicit assertion by the adapter that provider-native
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

    /// Provider schema that defined this verified evidence record.
    pub fn schema(&self) -> &str {
        &self.schema
    }

    /// External non-secret provider session identifier.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Provider-owned binding to the authenticated machine identity.
    pub fn peer_identity_binding(&self) -> &str {
        &self.peer_identity_binding
    }

    /// Trusted-time instant at which provider authority admitted the session.
    pub const fn authenticated_at_ms(&self) -> u64 {
        self.authenticated_at_ms
    }

    /// Exclusive hard validity horizon carried by the verified provider evidence.
    pub const fn expires_at_ms(&self) -> u64 {
        self.expires_at_ms
    }

    /// Authority generation that admitted the provider session.
    pub const fn authority_epoch(&self) -> u64 {
        self.authority_epoch
    }

    /// Provider-owned binding to the verified session/transcript evidence.
    pub fn evidence_binding(&self) -> &str {
        &self.evidence_binding
    }

    /// Validate only the provider-neutral immutable claim shape.
    pub fn validate_shape(&self) -> Result<(), &'static str> {
        for value in [
            self.schema.as_str(),
            self.session_id.as_str(),
            self.peer_identity_binding.as_str(),
            self.evidence_binding.as_str(),
        ] {
            if value.trim().is_empty() {
                return Err("session evidence text fields must not be empty");
            }
            if value.trim() != value || value.chars().any(char::is_control) {
                return Err("session evidence text fields must be canonical printable text");
            }
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
/// does **not** implement serde serialization. Its fields are private: a provider adapter must
/// explicitly cross [`MachineSessionContext::from_authority_provider`] after refreshing current
/// time, enrollment, revocation and authority generation. The context also carries the provider's
/// current peer-identity binding so state for one principal cannot accidentally authorize another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineSessionContext {
    peer_identity_binding: String,
    now_ms: u64,
    authority_epoch: u64,
    trusted_time_available: bool,
    revoked: bool,
}

impl MachineSessionContext {
    /// Cross the live authority-provider boundary into maritime core.
    ///
    /// Calling this constructor is an explicit assertion by the adapter that these values were
    /// freshly obtained from the provider's current authority source. Maritime core cannot and
    /// does not duplicate that provider's enrollment/revocation mechanism.
    pub fn from_authority_provider(
        peer_identity_binding: impl Into<String>,
        now_ms: u64,
        authority_epoch: u64,
        trusted_time_available: bool,
        revoked: bool,
    ) -> Result<Self, &'static str> {
        let peer_identity_binding = peer_identity_binding.into();
        if peer_identity_binding.trim().is_empty()
            || peer_identity_binding.trim() != peer_identity_binding
            || peer_identity_binding.chars().any(char::is_control)
        {
            return Err("authority context identity binding must be canonical printable text");
        }
        Ok(Self {
            peer_identity_binding,
            now_ms,
            authority_epoch,
            trusted_time_available,
            revoked,
        })
    }

    /// Provider-owned binding naming the principal whose live authority was checked.
    pub fn peer_identity_binding(&self) -> &str {
        &self.peer_identity_binding
    }

    /// Current provider-supplied trusted-time value.
    pub const fn now_ms(&self) -> u64 {
        self.now_ms
    }

    /// Current authority generation for this peer identity.
    pub const fn authority_epoch(&self) -> u64 {
        self.authority_epoch
    }

    /// Whether current time comes from a provider-approved trusted-time source.
    pub const fn trusted_time_available(&self) -> bool {
        self.trusted_time_available
    }

    /// Current revocation result for this peer identity.
    pub const fn revoked(&self) -> bool {
        self.revoked
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MachineSessionTrust {
    Trusted,
    Malformed,
    UnsupportedSchema,
    ValidityTooLong,
    ContextIdentityMismatch,
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
    context: &MachineSessionContext,
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
    if context.peer_identity_binding() != session.peer_identity_binding() {
        return MachineSessionTrust::ContextIdentityMismatch;
    }
    if !context.trusted_time_available() {
        return MachineSessionTrust::UntrustedTime;
    }
    if context.revoked() {
        return MachineSessionTrust::Revoked;
    }
    if session.authority_epoch() != context.authority_epoch() {
        return MachineSessionTrust::EpochMismatch;
    }
    if context.now_ms() < session.authenticated_at_ms() {
        return MachineSessionTrust::NotYetValid;
    }
    if context.now_ms() >= session.expires_at_ms() {
        return MachineSessionTrust::Expired;
    }
    MachineSessionTrust::Trusted
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SCHEMA: &str = "test-session-evidence-v1";
    const TEST_IDENTITY: &str = "xenia-fingerprint:abc";

    fn session() -> AuthenticatedMachineSession {
        AuthenticatedMachineSession::from_verified_provider(
            TEST_SCHEMA,
            "session-1",
            TEST_IDENTITY,
            100,
            200,
            9,
            "xenia-transcript:xyz",
        )
        .unwrap()
    }

    fn context_with(
        identity: &str,
        now_ms: u64,
        authority_epoch: u64,
        trusted_time_available: bool,
        revoked: bool,
    ) -> MachineSessionContext {
        MachineSessionContext::from_authority_provider(
            identity,
            now_ms,
            authority_epoch,
            trusted_time_available,
            revoked,
        )
        .unwrap()
    }

    fn context(now_ms: u64) -> MachineSessionContext {
        context_with(TEST_IDENTITY, now_ms, 9, true, false)
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
        assert!(
            AuthenticatedMachineSession::from_verified_provider(
                TEST_SCHEMA,
                " session-1",
                "binding",
                100,
                200,
                9,
                "evidence"
            )
            .is_err()
        );
    }

    #[test]
    fn authority_context_handoff_rejects_malformed_identity_binding() {
        assert!(
            MachineSessionContext::from_authority_provider("", 150, 9, true, false).is_err()
        );
        assert!(
            MachineSessionContext::from_authority_provider(" identity", 150, 9, true, false)
                .is_err()
        );
    }

    #[test]
    fn valid_session_is_trusted_only_in_matching_policy_time_and_epoch_context() {
        let current = context(150);
        assert_eq!(
            evaluate_machine_session(&session(), &current, policy()),
            MachineSessionTrust::Trusted
        );
    }

    #[test]
    fn provider_schema_must_be_explicitly_accepted() {
        let future = AuthenticatedMachineSession::from_verified_provider(
            "test-session-evidence-v2",
            "session-1",
            TEST_IDENTITY,
            100,
            200,
            9,
            "xenia-transcript:xyz",
        )
        .unwrap();
        let current = context(150);
        assert_eq!(
            evaluate_machine_session(&future, &current, policy()),
            MachineSessionTrust::UnsupportedSchema
        );
    }

    #[test]
    fn excessive_validity_horizon_fails_closed() {
        let too_long = AuthenticatedMachineSession::from_verified_provider(
            TEST_SCHEMA,
            "session-1",
            TEST_IDENTITY,
            100,
            201,
            9,
            "xenia-transcript:xyz",
        )
        .unwrap();
        let current = context(150);
        assert_eq!(
            evaluate_machine_session(&too_long, &current, policy()),
            MachineSessionTrust::ValidityTooLong
        );
    }

    #[test]
    fn live_context_must_name_the_same_peer_identity() {
        let wrong = context_with("xenia-fingerprint:different", 150, 9, true, false);
        assert_eq!(
            evaluate_machine_session(&session(), &wrong, policy()),
            MachineSessionTrust::ContextIdentityMismatch
        );
    }

    #[test]
    fn trusted_time_loss_fails_closed() {
        let current = context_with(TEST_IDENTITY, 150, 9, false, false);
        assert_eq!(
            evaluate_machine_session(&session(), &current, policy()),
            MachineSessionTrust::UntrustedTime
        );
    }

    #[test]
    fn live_revocation_dominates_immutable_session_evidence() {
        let issued = session();
        let current = context(150);
        assert_eq!(
            evaluate_machine_session(&issued, &current, policy()),
            MachineSessionTrust::Trusted
        );

        let revoked = context_with(TEST_IDENTITY, 150, 9, true, true);
        assert_eq!(
            evaluate_machine_session(&issued, &revoked, policy()),
            MachineSessionTrust::Revoked
        );
    }

    #[test]
    fn stale_epoch_and_expiry_are_rejected() {
        let stale = context_with(TEST_IDENTITY, 150, 10, true, false);
        assert_eq!(
            evaluate_machine_session(&session(), &stale, policy()),
            MachineSessionTrust::EpochMismatch
        );
        let expired = context(200);
        assert_eq!(
            evaluate_machine_session(&session(), &expired, policy()),
            MachineSessionTrust::Expired
        );
    }

    #[test]
    fn future_dated_session_is_rejected() {
        let current = context(99);
        assert_eq!(
            evaluate_machine_session(&session(), &current, policy()),
            MachineSessionTrust::NotYetValid
        );
    }
}
