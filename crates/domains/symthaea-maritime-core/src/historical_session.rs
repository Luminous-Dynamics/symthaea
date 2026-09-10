// SPDX-License-Identifier: AGPL-3.0-or-later
//! Provider-neutral handoff for historically qualified machine sessions.
//!
//! Live authority and historical authority answer different questions. `MachineSessionContext`
//! answers whether a session-derived action may be trusted **now**. This module accepts a positive
//! historical result that an external provider has already cryptographically verified and checks
//! that it names the exact immutable session and observation interval known to maritime core.
//!
//! The resulting type is deliberately non-serializable and does not implement or emulate live
//! authority context. Historical eligibility can support delayed evidence reconciliation, but can
//! never be fed into [`crate::evaluate_machine_session`] as current authorization.

use crate::{AuthenticatedMachineSession, MachineSessionPolicy};

fn canonical_text(value: &str) -> bool {
    !value.trim().is_empty()
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

/// Why a provider-qualified historical result could not cross into maritime core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoricalSessionHandoffError {
    /// Provider schema was not accepted by the local session policy.
    UnsupportedProviderSchema,
    /// Immutable session validity exceeds the deployment's local maximum.
    SessionValidityTooLong,
    /// Provider handoff named a different session-evidence schema.
    ProviderMismatch,
    /// Provider handoff named a different authenticated machine principal.
    IdentityMismatch,
    /// Provider handoff named a different non-secret session identifier.
    SessionIdMismatch,
    /// Provider handoff named different immutable provider/session evidence.
    SessionEvidenceMismatch,
    /// Provider handoff qualified a different authority generation.
    AuthorityEpochMismatch,
    /// Provider handoff did not bind the exact immutable session interval.
    SessionIntervalMismatch,
    /// Provider handoff claimed an empty or reversed session interval.
    InvalidSessionInterval,
    /// Observation predates the exact admitted session.
    ObservationBeforeSession,
    /// Observation is at or after the exact session's exclusive expiry.
    ObservationAfterSession,
    /// Provider history did not claim completeness through the observation.
    HistoryNotCovered,
    /// Provider-history binding was empty, padded, or contained control characters.
    InvalidHistoryBinding,
}

/// Non-serializable proof that an external authority provider historically qualified one exact
/// immutable machine session through one exact observation instant.
///
/// Construction is an explicit provider-verification handoff. The provider adapter must first
/// verify its native signed/history evidence. Maritime core then checks that the positive result
/// exactly matches the existing provider session, including the provider-owned transcript/session
/// evidence binding, local schema/lifetime policy, authority epoch, admitted-session interval, and
/// observation time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoricallyQualifiedMachineSessionV1 {
    provider_schema: String,
    session_id: String,
    peer_identity_binding: String,
    session_evidence_binding: String,
    authority_epoch: u64,
    session_authenticated_at_ms: u64,
    session_expires_at_ms: u64,
    observation_at_ms: u64,
    provider_history_binding: String,
    history_head_sequence: u64,
    history_observed_through_ms: u64,
}

impl HistoricallyQualifiedMachineSessionV1 {
    /// Cross a provider's already-verified historical authority result into maritime core.
    ///
    /// The redundant session fields are intentional: a historical result for another session,
    /// principal, provider, transcript/session proof, epoch, or validity interval must not be
    /// reusable merely because an observation timestamp happens to fit the local session.
    #[allow(clippy::too_many_arguments)]
    pub fn from_verified_provider(
        session: &AuthenticatedMachineSession,
        session_policy: MachineSessionPolicy<'_>,
        provider_schema: impl Into<String>,
        session_id: impl Into<String>,
        peer_identity_binding: impl Into<String>,
        session_evidence_binding: impl Into<String>,
        authority_epoch: u64,
        session_authenticated_at_ms: u64,
        session_expires_at_ms: u64,
        observation_at_ms: u64,
        provider_history_binding: impl Into<String>,
        history_head_sequence: u64,
        history_observed_through_ms: u64,
    ) -> Result<Self, HistoricalSessionHandoffError> {
        if !session_policy
            .accepted_schemas
            .iter()
            .any(|accepted| *accepted == session.schema())
        {
            return Err(HistoricalSessionHandoffError::UnsupportedProviderSchema);
        }
        let session_validity = session
            .expires_at_ms()
            .checked_sub(session.authenticated_at_ms())
            .ok_or(HistoricalSessionHandoffError::InvalidSessionInterval)?;
        if session_validity > session_policy.max_validity_ms {
            return Err(HistoricalSessionHandoffError::SessionValidityTooLong);
        }

        let provider_schema = provider_schema.into();
        let session_id = session_id.into();
        let peer_identity_binding = peer_identity_binding.into();
        let session_evidence_binding = session_evidence_binding.into();
        let provider_history_binding = provider_history_binding.into();

        if provider_schema != session.schema() {
            return Err(HistoricalSessionHandoffError::ProviderMismatch);
        }
        if peer_identity_binding != session.peer_identity_binding() {
            return Err(HistoricalSessionHandoffError::IdentityMismatch);
        }
        if session_id != session.session_id() {
            return Err(HistoricalSessionHandoffError::SessionIdMismatch);
        }
        if session_evidence_binding != session.evidence_binding() {
            return Err(HistoricalSessionHandoffError::SessionEvidenceMismatch);
        }
        if authority_epoch != session.authority_epoch() {
            return Err(HistoricalSessionHandoffError::AuthorityEpochMismatch);
        }
        if session_expires_at_ms <= session_authenticated_at_ms {
            return Err(HistoricalSessionHandoffError::InvalidSessionInterval);
        }
        if session_authenticated_at_ms != session.authenticated_at_ms()
            || session_expires_at_ms != session.expires_at_ms()
        {
            return Err(HistoricalSessionHandoffError::SessionIntervalMismatch);
        }
        if observation_at_ms < session_authenticated_at_ms {
            return Err(HistoricalSessionHandoffError::ObservationBeforeSession);
        }
        if observation_at_ms >= session_expires_at_ms {
            return Err(HistoricalSessionHandoffError::ObservationAfterSession);
        }
        if history_observed_through_ms < observation_at_ms {
            return Err(HistoricalSessionHandoffError::HistoryNotCovered);
        }
        if !canonical_text(&provider_history_binding) {
            return Err(HistoricalSessionHandoffError::InvalidHistoryBinding);
        }

        Ok(Self {
            provider_schema,
            session_id,
            peer_identity_binding,
            session_evidence_binding,
            authority_epoch,
            session_authenticated_at_ms,
            session_expires_at_ms,
            observation_at_ms,
            provider_history_binding,
            history_head_sequence,
            history_observed_through_ms,
        })
    }

    /// Session-evidence provider schema whose history was qualified.
    pub fn provider_schema(&self) -> &str { &self.provider_schema }
    /// Non-secret provider session identifier qualified by history.
    pub fn session_id(&self) -> &str { &self.session_id }
    /// Provider-owned binding to the exact authenticated machine principal.
    pub fn peer_identity_binding(&self) -> &str { &self.peer_identity_binding }
    /// Provider-owned binding to the exact authenticated session/transcript evidence.
    pub fn session_evidence_binding(&self) -> &str { &self.session_evidence_binding }
    /// Exact authority generation historically qualified by the provider.
    pub const fn authority_epoch(&self) -> u64 { self.authority_epoch }
    /// Exact session-admission instant historically qualified by the provider.
    pub const fn session_authenticated_at_ms(&self) -> u64 { self.session_authenticated_at_ms }
    /// Exact exclusive session-expiry instant historically qualified by the provider.
    pub const fn session_expires_at_ms(&self) -> u64 { self.session_expires_at_ms }
    /// Observation instant historically qualified by the provider.
    pub const fn observation_at_ms(&self) -> u64 { self.observation_at_ms }
    /// Opaque binding to provider-owned verified history evidence.
    pub fn provider_history_binding(&self) -> &str { &self.provider_history_binding }
    /// Monotonic provider history-head sequence that supported this qualification.
    pub const fn history_head_sequence(&self) -> u64 { self.history_head_sequence }
    /// Provider history's completeness horizon for this qualification.
    pub const fn history_observed_through_ms(&self) -> u64 { self.history_observed_through_ms }

    /// Whether this historical result names the exact immutable session and observation instant.
    pub fn matches_session_observation(
        &self,
        session: &AuthenticatedMachineSession,
        observation_at_ms: u64,
    ) -> bool {
        self.provider_schema == session.schema()
            && self.session_id == session.session_id()
            && self.peer_identity_binding == session.peer_identity_binding()
            && self.session_evidence_binding == session.evidence_binding()
            && self.authority_epoch == session.authority_epoch()
            && self.session_authenticated_at_ms == session.authenticated_at_ms()
            && self.session_expires_at_ms == session.expires_at_ms()
            && self.observation_at_ms == observation_at_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &str = "xenia-verified-machine-session-evidence-v1";
    const PRINCIPAL: &str = "xenia-signing-identity-v1:blake3-256:abc";
    const SESSION_EVIDENCE: &str = "xenia-handshake-transcript-v1:blake3-256:def";

    fn session() -> AuthenticatedMachineSession {
        AuthenticatedMachineSession::from_verified_provider(
            SCHEMA, "session-1", PRINCIPAL, 1_000, 2_000, 9, SESSION_EVIDENCE,
        ).unwrap()
    }

    fn policy() -> MachineSessionPolicy<'static> {
        MachineSessionPolicy { accepted_schemas: &[SCHEMA], max_validity_ms: 1_000 }
    }

    #[allow(clippy::too_many_arguments)]
    fn handoff(
        provider_schema: &str,
        session_id: &str,
        principal: &str,
        session_evidence: &str,
        epoch: u64,
        authenticated_at_ms: u64,
        expires_at_ms: u64,
        observation_at_ms: u64,
        history_binding: &str,
        history_head_sequence: u64,
        observed_through_ms: u64,
    ) -> Result<HistoricallyQualifiedMachineSessionV1, HistoricalSessionHandoffError> {
        HistoricallyQualifiedMachineSessionV1::from_verified_provider(
            &session(), policy(), provider_schema, session_id, principal, session_evidence, epoch,
            authenticated_at_ms, expires_at_ms, observation_at_ms, history_binding,
            history_head_sequence, observed_through_ms,
        )
    }

    fn qualify() -> Result<HistoricallyQualifiedMachineSessionV1, HistoricalSessionHandoffError> {
        handoff(
            SCHEMA, "session-1", PRINCIPAL, SESSION_EVIDENCE, 9, 1_000, 2_000, 1_500,
            "xenia-machine-authority-history-head-v1:abc", 7, 1_600,
        )
    }

    #[test]
    fn exact_historical_session_and_observation_are_accepted() {
        let qualified = qualify().unwrap();
        assert!(qualified.matches_session_observation(&session(), 1_500));
        assert_eq!(qualified.session_evidence_binding(), SESSION_EVIDENCE);
        assert_eq!(qualified.authority_epoch(), 9);
        assert_eq!(qualified.session_expires_at_ms(), 2_000);
        assert_eq!(qualified.history_head_sequence(), 7);
        assert_eq!(qualified.history_observed_through_ms(), 1_600);
    }

    #[test]
    fn historical_handoff_cannot_bypass_local_schema_or_lifetime_policy() {
        let unsupported = MachineSessionPolicy {
            accepted_schemas: &["other-provider-v1"],
            max_validity_ms: 1_000,
        };
        assert_eq!(
            HistoricallyQualifiedMachineSessionV1::from_verified_provider(
                &session(), unsupported, SCHEMA, "session-1", PRINCIPAL, SESSION_EVIDENCE, 9,
                1_000, 2_000, 1_500, "history:abc", 0, 1_600,
            ),
            Err(HistoricalSessionHandoffError::UnsupportedProviderSchema)
        );
        let too_short = MachineSessionPolicy { accepted_schemas: &[SCHEMA], max_validity_ms: 999 };
        assert_eq!(
            HistoricallyQualifiedMachineSessionV1::from_verified_provider(
                &session(), too_short, SCHEMA, "session-1", PRINCIPAL, SESSION_EVIDENCE, 9,
                1_000, 2_000, 1_500, "history:abc", 0, 1_600,
            ),
            Err(HistoricalSessionHandoffError::SessionValidityTooLong)
        );
    }

    #[test]
    fn mismatched_identity_session_evidence_epoch_interval_or_observation_fail_closed() {
        assert_eq!(
            handoff(SCHEMA, "session-1", "other-principal", SESSION_EVIDENCE, 9, 1_000, 2_000, 1_500, "history:abc", 0, 1_600),
            Err(HistoricalSessionHandoffError::IdentityMismatch)
        );
        assert_eq!(
            handoff(SCHEMA, "session-1", PRINCIPAL, "other-evidence", 9, 1_000, 2_000, 1_500, "history:abc", 0, 1_600),
            Err(HistoricalSessionHandoffError::SessionEvidenceMismatch)
        );
        assert_eq!(
            handoff(SCHEMA, "session-1", PRINCIPAL, SESSION_EVIDENCE, 10, 1_000, 2_000, 1_500, "history:abc", 0, 1_600),
            Err(HistoricalSessionHandoffError::AuthorityEpochMismatch)
        );
        assert_eq!(
            handoff(SCHEMA, "session-1", PRINCIPAL, SESSION_EVIDENCE, 9, 1_000, 2_001, 1_500, "history:abc", 0, 1_600),
            Err(HistoricalSessionHandoffError::SessionIntervalMismatch)
        );
        assert_eq!(
            handoff(SCHEMA, "session-1", PRINCIPAL, SESSION_EVIDENCE, 9, 1_000, 2_000, 2_000, "history:abc", 0, 2_000),
            Err(HistoricalSessionHandoffError::ObservationAfterSession)
        );
    }

    #[test]
    fn incomplete_history_or_bad_binding_fail_closed() {
        assert_eq!(
            handoff(SCHEMA, "session-1", PRINCIPAL, SESSION_EVIDENCE, 9, 1_000, 2_000, 1_500, "history:abc", 0, 1_499),
            Err(HistoricalSessionHandoffError::HistoryNotCovered)
        );
        assert_eq!(
            handoff(SCHEMA, "session-1", PRINCIPAL, SESSION_EVIDENCE, 9, 1_000, 2_000, 1_500, " history:abc", 0, 1_600),
            Err(HistoricalSessionHandoffError::InvalidHistoryBinding)
        );
    }
}
