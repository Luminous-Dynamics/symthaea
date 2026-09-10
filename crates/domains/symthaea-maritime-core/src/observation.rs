// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic association between mission-neutral observations and trusted machine sessions.
//!
//! This module does not introduce a new authentication scheme. Xenia (or another configured
//! provider) remains responsible for cryptographic session verification and live authority.
//! Maritime core only mints a deterministic observation binding after the provider-neutral
//! session contract evaluates `Trusted`. The resulting BLAKE3 value is an association/integrity
//! identifier suitable for an opaque downstream evidence field; it is not a signature or a
//! standalone proof of authentication.

use crate::{
    AuthenticatedMachineSession, MachineSessionContext, MachineSessionPolicy, MachineSessionTrust,
    evaluate_machine_session,
};
use serde::{Deserialize, Serialize};

/// Stable prefix for deterministic session-bound observation identifiers.
pub const SESSION_BOUND_OBSERVATION_PREFIX_V1: &str =
    "symthaea-maritime-session-bound-observation-v1:blake3-256:";

const MAX_PLATFORM_ID_BYTES: usize = 256;
const MAX_PAYLOAD_BYTES: usize = 60 * 1024;
const MAX_POSITION_REFERENCE_COUNT: usize = 64;
const MAX_POSITION_REFERENCE_BYTES: usize = 512;

/// Mission-neutral observation classes aligned with the current maritime evidence vocabulary.
/// Weapon, target, engagement and lethal-force semantics are intentionally absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaritimeObservationKind {
    StateObservation,
    HealthObservation,
    AssuranceTransition,
    AuthorityTransition,
    PositionEvidenceReference,
    CommunicationsState,
    LogisticsEvent,
    MaintenanceEvent,
    RecoveryEvent,
}

impl MaritimeObservationKind {
    /// Stable v1 wire label used by the binding algorithm.
    pub const fn as_wire_label(self) -> &'static str {
        match self {
            Self::StateObservation => "state_observation",
            Self::HealthObservation => "health_observation",
            Self::AssuranceTransition => "assurance_transition",
            Self::AuthorityTransition => "authority_transition",
            Self::PositionEvidenceReference => "position_evidence_reference",
            Self::CommunicationsState => "communications_state",
            Self::LogisticsEvent => "logistics_event",
            Self::MaintenanceEvent => "maintenance_event",
            Self::RecoveryEvent => "recovery_event",
        }
    }
}

/// Why a session-bound observation could not be minted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservationBindingError {
    /// The provider-neutral session gate did not evaluate to `Trusted`.
    SessionNotTrusted(MachineSessionTrust),
    /// `platform_id` was empty, padded, oversized or contained control characters.
    InvalidPlatformId,
    /// The payload exceeded the current Mycelix maritime-v1 payload bound.
    PayloadTooLarge,
    /// The payload was padded or was not valid JSON.
    InvalidPayloadJson,
    /// More position references were supplied than the v1 evidence contract permits.
    TooManyPositionReferences,
    /// A position evidence reference was empty, padded, oversized or contained controls.
    InvalidPositionReference,
    /// Position references were not strictly sorted and duplicate-free.
    NonCanonicalPositionReferences,
    /// The claimed observation predates the authenticated session.
    ObservationBeforeSession,
    /// The claimed observation is at or after the session's exclusive expiry.
    ObservationAfterSession,
    /// The observation claims a time later than the fresh authority check used to mint it.
    ObservationAfterAuthorityCheck,
}

/// In-process proof that maritime core associated one exact observation with a session that was
/// trusted at the time this value was minted.
///
/// Fields are private and this type deliberately has no serde surface. Persisting or transporting
/// the returned binding does not preserve live revocation state and must not be treated as a
/// signature. Downstream systems that need authenticated provenance must verify provider-owned
/// evidence through that provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionBoundObservationV1 {
    binding: String,
    platform_id: String,
    observed_at_us: u64,
    kind: MaritimeObservationKind,
}

impl SessionBoundObservationV1 {
    /// Opaque deterministic association identifier suitable for downstream evidence references.
    pub fn evidence_binding(&self) -> &str {
        &self.binding
    }

    /// Platform whose observation bytes were bound.
    pub fn platform_id(&self) -> &str {
        &self.platform_id
    }

    /// Claimed observation instant included in the binding.
    pub const fn observed_at_us(&self) -> u64 {
        self.observed_at_us
    }

    /// Mission-neutral observation class included in the binding.
    pub const fn kind(&self) -> MaritimeObservationKind {
        self.kind
    }
}

/// Bind exact observation bytes to an authenticated session only after current authority succeeds.
///
/// The algorithm commits to the provider schema, session identifier, peer identity, provider-owned
/// evidence binding, session admission interval/epoch, platform, observation time/class, exact JSON
/// payload bytes, and ordered position-evidence references. Mycelix may carry the returned value as
/// an opaque `evidence_binding` while independently owning sequence/predecessor continuity.
pub fn bind_observation_to_trusted_session(
    session: &AuthenticatedMachineSession,
    context: &MachineSessionContext,
    policy: MachineSessionPolicy<'_>,
    platform_id: impl Into<String>,
    observed_at_us: u64,
    kind: MaritimeObservationKind,
    payload_json: &str,
    position_evidence_refs: &[String],
) -> Result<SessionBoundObservationV1, ObservationBindingError> {
    let trust = evaluate_machine_session(session, context, policy);
    if trust != MachineSessionTrust::Trusted {
        return Err(ObservationBindingError::SessionNotTrusted(trust));
    }

    let platform_id = platform_id.into();
    if !canonical_text(&platform_id, MAX_PLATFORM_ID_BYTES) {
        return Err(ObservationBindingError::InvalidPlatformId);
    }
    if payload_json.len() > MAX_PAYLOAD_BYTES {
        return Err(ObservationBindingError::PayloadTooLarge);
    }
    if payload_json.trim() != payload_json
        || serde_json::from_str::<serde_json::Value>(payload_json).is_err()
    {
        return Err(ObservationBindingError::InvalidPayloadJson);
    }
    if position_evidence_refs.len() > MAX_POSITION_REFERENCE_COUNT {
        return Err(ObservationBindingError::TooManyPositionReferences);
    }
    for reference in position_evidence_refs {
        if !canonical_text(reference, MAX_POSITION_REFERENCE_BYTES) {
            return Err(ObservationBindingError::InvalidPositionReference);
        }
    }
    if position_evidence_refs
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(ObservationBindingError::NonCanonicalPositionReferences);
    }

    // Session/context time is millisecond precision while maritime evidence uses microseconds.
    // Compare on the provider's millisecond clock to avoid pretending sub-millisecond authority.
    let observed_at_ms = observed_at_us / 1_000;
    if observed_at_ms < session.authenticated_at_ms() {
        return Err(ObservationBindingError::ObservationBeforeSession);
    }
    if observed_at_ms >= session.expires_at_ms() {
        return Err(ObservationBindingError::ObservationAfterSession);
    }
    if observed_at_ms > context.now_ms() {
        return Err(ObservationBindingError::ObservationAfterAuthorityCheck);
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-maritime-session-bound-observation-v1\0");
    hash_bytes(&mut hasher, session.schema().as_bytes());
    hash_bytes(&mut hasher, session.session_id().as_bytes());
    hash_bytes(&mut hasher, session.peer_identity_binding().as_bytes());
    hash_bytes(&mut hasher, session.evidence_binding().as_bytes());
    hash_bytes(&mut hasher, &session.authenticated_at_ms().to_le_bytes());
    hash_bytes(&mut hasher, &session.expires_at_ms().to_le_bytes());
    hash_bytes(&mut hasher, &session.authority_epoch().to_le_bytes());
    hash_bytes(&mut hasher, platform_id.as_bytes());
    hash_bytes(&mut hasher, &observed_at_us.to_le_bytes());
    hash_bytes(&mut hasher, kind.as_wire_label().as_bytes());
    hash_bytes(&mut hasher, payload_json.as_bytes());
    for reference in position_evidence_refs {
        hash_bytes(&mut hasher, reference.as_bytes());
    }

    Ok(SessionBoundObservationV1 {
        binding: format!(
            "{SESSION_BOUND_OBSERVATION_PREFIX_V1}{}",
            hasher.finalize().to_hex()
        ),
        platform_id,
        observed_at_us,
        kind,
    })
}

fn canonical_text(value: &str, max_bytes: usize) -> bool {
    !value.is_empty()
        && value.len() <= max_bytes
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn hash_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &str = "test-session-evidence-v1";
    const IDENTITY: &str = "test-peer:abc";

    fn session() -> AuthenticatedMachineSession {
        AuthenticatedMachineSession::from_verified_provider(
            SCHEMA,
            "session-1",
            IDENTITY,
            1_000,
            2_000,
            9,
            "test-transcript:def",
        )
        .unwrap()
    }

    fn context(
        now_ms: u64,
        provider: &str,
        identity: &str,
        epoch: u64,
        revoked: bool,
    ) -> MachineSessionContext {
        MachineSessionContext::from_authority_provider(
            provider, identity, now_ms, epoch, true, revoked,
        )
        .unwrap()
    }

    fn policy() -> MachineSessionPolicy<'static> {
        MachineSessionPolicy {
            accepted_schemas: &[SCHEMA],
            max_validity_ms: 1_000,
        }
    }

    fn bind(payload: &str) -> Result<SessionBoundObservationV1, ObservationBindingError> {
        bind_observation_to_trusted_session(
            &session(),
            &context(1_500, SCHEMA, IDENTITY, 9, false),
            policy(),
            "auv-01",
            1_400_000,
            MaritimeObservationKind::HealthObservation,
            payload,
            &["mycelix-position:measurement:001".into()],
        )
    }

    #[test]
    fn trusted_session_mints_stable_binding() {
        let first = bind(r#"{"severity":"healthy"}"#).unwrap();
        let second = bind(r#"{"severity":"healthy"}"#).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.platform_id(), "auv-01");
        assert_eq!(first.observed_at_us(), 1_400_000);
        assert_eq!(first.kind(), MaritimeObservationKind::HealthObservation);
        assert!(first
            .evidence_binding()
            .starts_with(SESSION_BOUND_OBSERVATION_PREFIX_V1));
        assert_eq!(
            first.evidence_binding().len(),
            SESSION_BOUND_OBSERVATION_PREFIX_V1.len() + 64
        );
    }

    #[test]
    fn payload_and_reference_changes_change_the_binding() {
        let baseline = bind(r#"{"severity":"healthy"}"#).unwrap();
        let changed_payload = bind(r#"{"severity":"degraded"}"#).unwrap();
        assert_ne!(baseline.evidence_binding(), changed_payload.evidence_binding());

        let changed_reference = bind_observation_to_trusted_session(
            &session(),
            &context(1_500, SCHEMA, IDENTITY, 9, false),
            policy(),
            "auv-01",
            1_400_000,
            MaritimeObservationKind::HealthObservation,
            r#"{"severity":"healthy"}"#,
            &["mycelix-position:measurement:002".into()],
        )
        .unwrap();
        assert_ne!(baseline.evidence_binding(), changed_reference.evidence_binding());
    }

    #[test]
    fn untrusted_session_states_cannot_mint_observation_binding() {
        let session = session();
        let wrong_provider = context(1_500, "other-provider-v1", IDENTITY, 9, false);
        assert_eq!(
            bind_observation_to_trusted_session(
                &session,
                &wrong_provider,
                policy(),
                "auv-01",
                1_400_000,
                MaritimeObservationKind::StateObservation,
                "{}",
                &[],
            ),
            Err(ObservationBindingError::SessionNotTrusted(
                MachineSessionTrust::ContextProviderMismatch
            ))
        );

        let revoked = context(1_500, SCHEMA, IDENTITY, 9, true);
        assert_eq!(
            bind_observation_to_trusted_session(
                &session,
                &revoked,
                policy(),
                "auv-01",
                1_400_000,
                MaritimeObservationKind::StateObservation,
                "{}",
                &[],
            ),
            Err(ObservationBindingError::SessionNotTrusted(
                MachineSessionTrust::Revoked
            ))
        );
    }

    #[test]
    fn observation_time_must_be_inside_session_and_not_after_authority_check() {
        let session = session();
        let current = context(1_500, SCHEMA, IDENTITY, 9, false);

        assert_eq!(
            bind_observation_to_trusted_session(
                &session,
                &current,
                policy(),
                "auv-01",
                999_999,
                MaritimeObservationKind::StateObservation,
                "{}",
                &[],
            ),
            Err(ObservationBindingError::ObservationBeforeSession)
        );
        assert_eq!(
            bind_observation_to_trusted_session(
                &session,
                &current,
                policy(),
                "auv-01",
                2_000_000,
                MaritimeObservationKind::StateObservation,
                "{}",
                &[],
            ),
            Err(ObservationBindingError::ObservationAfterSession)
        );
        assert_eq!(
            bind_observation_to_trusted_session(
                &session,
                &current,
                policy(),
                "auv-01",
                1_501_000,
                MaritimeObservationKind::StateObservation,
                "{}",
                &[],
            ),
            Err(ObservationBindingError::ObservationAfterAuthorityCheck)
        );
    }

    #[test]
    fn downstream_wire_shape_constraints_fail_before_binding() {
        assert_eq!(
            bind(" {\"severity\":\"healthy\"}"),
            Err(ObservationBindingError::InvalidPayloadJson)
        );

        let session = session();
        let current = context(1_500, SCHEMA, IDENTITY, 9, false);
        let unsorted = vec!["position:z".into(), "position:a".into()];
        assert_eq!(
            bind_observation_to_trusted_session(
                &session,
                &current,
                policy(),
                "auv-01",
                1_400_000,
                MaritimeObservationKind::PositionEvidenceReference,
                "{}",
                &unsorted,
            ),
            Err(ObservationBindingError::NonCanonicalPositionReferences)
        );
    }
}
