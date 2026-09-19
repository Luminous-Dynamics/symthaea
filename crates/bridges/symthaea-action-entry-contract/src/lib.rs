// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Precommitted executor/connector identity and dispatch-incarnation contracts.
//!
//! This crate deliberately stops before effect entry.
//!
//! ```text
//! EntryBindingCommitmentV1
//!     != authenticated principal linkage
//!     != VerifiedExecutorBinding
//!     != current executor identity
//!     != DispatchPermitV2
//!     != effect entry
//!
//! DispatchIncarnationId
//!     != dispatch
//!     != remote receipt
//!     != effect application
//!     != finality
//! ```
//!
//! For effects using this profile, the exact entry binding is committed before
//! reservation through `EffectAuthorityBindingV2.parameters_digest`. The future
//! point-of-entry firewall must recompute that equality and consume a real
//! verifier-owned `VerifiedExecutorBinding`; it may not select a different
//! semantic principal, identity profile, or adapter after permit minting.

#![deny(unsafe_code)]

use sha2::{Digest as ShaDigest, Sha256};
use symthaea_action_admission::{
    DispatchPermitId, DispatchPermitV2, EffectAuthorityBindingV2,
};
use symthaea_action_runtime::{AttemptId, EffectBindingDigest, EffectIntentId, ReservationId};
use symthaea_authority::{Digest32 as AuthorityDigest32, PrincipalId};
use symthaea_authority_state::VerifiedAuthorityStateV2;
use symthaea_executor_identity::{
    ExecutorBindingMismatch, ExecutorIdentityChallenge, ExecutorIdentityRequirement,
    ExecutorProfileId, ExecutorRuntimeIncarnationId, VerifiedExecutorBinding,
};
use symthaea_interaction_core::{
    ConnectorIdentity, Digest32 as InteractionDigest32, PrincipalRef,
};
use thiserror::Error;

pub const ACTION_ENTRY_CONTRACT_SCHEMA_VERSION: u16 = 1;
const ENTRY_BINDING_DOMAIN: &[u8] = b"symthaea.action-entry.binding.v1\0";
const DISPATCH_INCARNATION_DOMAIN: &[u8] = b"symthaea.action-entry.dispatch-incarnation.v1\0";
const MAX_AUTHORITY_PRINCIPAL_BYTES: usize = 1024;

/// Ordinary policy/identity commitment selected before V2D reservation.
///
/// This binds two identity coordinates but does not prove they name the same
/// real-world subject. That same-subject theorem belongs to trusted identity
/// provisioning/composition, not this pure transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryBindingCommitmentV1 {
    domain_parameters_digest: AuthorityDigest32,
    authority_executor: PrincipalId,
    semantic_executor: PrincipalRef,
    identity_requirement: ExecutorIdentityRequirement,
    executor_profile: ExecutorProfileId,
    connector: ConnectorIdentity,
    adapter_implementation_digest: InteractionDigest32,
}

impl EntryBindingCommitmentV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        domain_parameters_digest: AuthorityDigest32,
        authority_executor: PrincipalId,
        semantic_executor: PrincipalRef,
        identity_requirement: ExecutorIdentityRequirement,
        executor_profile: ExecutorProfileId,
        connector: ConnectorIdentity,
        adapter_implementation_digest: InteractionDigest32,
    ) -> Result<Self, ActionEntryContractError> {
        if authority_executor.0.is_empty()
            || authority_executor.0.len() > MAX_AUTHORITY_PRINCIPAL_BYTES
        {
            return Err(ActionEntryContractError::InvalidAuthorityExecutor);
        }
        reject_zero_authority("domain parameters digest", domain_parameters_digest)?;
        reject_zero_interaction(
            "adapter implementation digest",
            adapter_implementation_digest,
        )?;
        Ok(Self {
            domain_parameters_digest,
            authority_executor,
            semantic_executor,
            identity_requirement,
            executor_profile,
            connector,
            adapter_implementation_digest,
        })
    }

    pub fn domain_parameters_digest(&self) -> AuthorityDigest32 {
        self.domain_parameters_digest
    }

    pub fn authority_executor(&self) -> &PrincipalId {
        &self.authority_executor
    }

    pub fn semantic_executor(&self) -> &PrincipalRef {
        &self.semantic_executor
    }

    pub fn identity_requirement(&self) -> &ExecutorIdentityRequirement {
        &self.identity_requirement
    }

    pub fn executor_profile(&self) -> ExecutorProfileId {
        self.executor_profile
    }

    pub fn connector(&self) -> &ConnectorIdentity {
        &self.connector
    }

    pub fn adapter_implementation_digest(&self) -> InteractionDigest32 {
        self.adapter_implementation_digest
    }

    /// SHA-256 commitment over the explicit cross-identity/adapter profile.
    #[must_use]
    pub fn digest(&self) -> AuthorityDigest32 {
        let mut transcript = Transcript::new(ENTRY_BINDING_DOMAIN);
        transcript.u16(ACTION_ENTRY_CONTRACT_SCHEMA_VERSION);
        transcript.text(&self.authority_executor.0);
        transcript.interaction_digest(self.semantic_executor.digest());
        transcript.interaction_digest(self.identity_requirement.digest());
        transcript.interaction_digest(self.executor_profile.digest());
        transcript.interaction_digest(self.connector.digest());
        transcript.interaction_digest(self.adapter_implementation_digest);
        transcript.authority_digest(self.domain_parameters_digest);
        transcript.finish_authority()
    }

    /// Verify that a V2D effect binding committed this exact entry profile.
    pub fn validate_effect_binding(
        &self,
        binding: &EffectAuthorityBindingV2,
    ) -> Result<(), ActionEntryContractError> {
        if &binding.executor != self.authority_executor() {
            return Err(ActionEntryContractError::AuthorityExecutorMismatch);
        }
        if binding.parameters_digest != self.digest() {
            return Err(ActionEntryContractError::EntryCommitmentMismatch);
        }
        Ok(())
    }

    /// Verify only the identity-bearing portion of a V2D permit.
    ///
    /// This does not re-establish authority/currentness and grants no entry.
    pub fn validate_permit(
        &self,
        permit: &DispatchPermitV2,
    ) -> Result<(), ActionEntryContractError> {
        self.validate_effect_binding(permit.effect_binding())
    }

    /// Verify that an ordinary executor challenge names the exact identity
    /// expectations already committed before permit minting.
    ///
    /// Challenge agreement is structural only; it is not proof that the nonce is
    /// fresh/trusted or that a live verified executor binding exists.
    pub fn validate_challenge(
        &self,
        challenge: &ExecutorIdentityChallenge,
    ) -> Result<(), ActionEntryContractError> {
        if challenge.principal_digest() != self.semantic_executor.digest() {
            return Err(ActionEntryContractError::ChallengePrincipalMismatch);
        }
        if challenge.requirement_digest() != self.identity_requirement.digest() {
            return Err(ActionEntryContractError::ChallengeRequirementMismatch);
        }
        if challenge.executor_profile() != self.executor_profile {
            return Err(ActionEntryContractError::ChallengeExecutorProfileMismatch);
        }
        Ok(())
    }
}

/// Stable provenance identity for one exact local dispatch carrier/incarnation.
///
/// The identity itself is ordinary data and grants no capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DispatchIncarnationId(AuthorityDigest32);

impl DispatchIncarnationId {
    pub const fn digest(self) -> AuthorityDigest32 {
        self.0
    }
}

/// Derive one dispatch-incarnation identity from the real live objects available
/// to the future effect-entry firewall.
///
/// This function deliberately performs only structural identity composition. In
/// particular, `VerifiedExecutorBinding::structural_match` does not independently
/// establish point-of-entry currentness. A future V2E-B firewall must require the
/// trusted executor-currentness theorem before invoking an adapter.
pub fn derive_dispatch_incarnation_id(
    permit: &DispatchPermitV2,
    entry: &EntryBindingCommitmentV1,
    challenge: &ExecutorIdentityChallenge,
    executor_binding: &VerifiedExecutorBinding,
    authority_state: &VerifiedAuthorityStateV2,
    runtime_session_nonce: [u8; 32],
) -> Result<DispatchIncarnationId, ActionEntryContractError> {
    entry.validate_permit(permit)?;
    entry.validate_challenge(challenge)?;
    executor_binding.structural_match(challenge, entry.identity_requirement())?;

    derive_dispatch_incarnation_from_parts(
        permit.permit_id(),
        permit.grant_digest(),
        permit.effect_intent_id(),
        permit.attempt_id(),
        permit.reservation_id(),
        permit.effect_binding_digest(),
        entry.digest(),
        executor_binding.digest(),
        entry.executor_profile(),
        challenge.runtime_incarnation(),
        entry.connector().digest(),
        entry.adapter_implementation_digest(),
        authority_state.snapshot_digest(),
        authority_state.state_sequence(),
        authority_state.source_frontier(),
        authority_state.state_policy_digest(),
        authority_state.time_policy_digest(),
        runtime_session_nonce,
    )
}

#[allow(clippy::too_many_arguments)]
fn derive_dispatch_incarnation_from_parts(
    permit_id: DispatchPermitId,
    grant_digest: AuthorityDigest32,
    effect_intent_id: EffectIntentId,
    attempt_id: AttemptId,
    reservation_id: ReservationId,
    effect_binding_digest: EffectBindingDigest,
    entry_binding_digest: AuthorityDigest32,
    executor_binding_digest: InteractionDigest32,
    executor_profile: ExecutorProfileId,
    runtime_incarnation: ExecutorRuntimeIncarnationId,
    connector_digest: InteractionDigest32,
    adapter_implementation_digest: InteractionDigest32,
    authority_snapshot_digest: AuthorityDigest32,
    authority_state_sequence: u64,
    authority_source_frontier: (u64, AuthorityDigest32),
    state_policy_digest: [u8; 32],
    time_policy_digest: [u8; 32],
    runtime_session_nonce: [u8; 32],
) -> Result<DispatchIncarnationId, ActionEntryContractError> {
    if runtime_session_nonce == [0; 32] {
        return Err(ActionEntryContractError::ZeroRuntimeSessionNonce);
    }
    reject_zero_interaction("executor binding digest", executor_binding_digest)?;

    let mut transcript = Transcript::new(DISPATCH_INCARNATION_DOMAIN);
    transcript.u16(ACTION_ENTRY_CONTRACT_SCHEMA_VERSION);
    transcript.authority_digest(permit_id.0);
    transcript.authority_digest(grant_digest);
    transcript.authority_digest(effect_intent_id.0);
    transcript.authority_digest(attempt_id.0);
    transcript.authority_digest(reservation_id.0);
    transcript.authority_digest(effect_binding_digest.0);
    transcript.authority_digest(entry_binding_digest);
    transcript.interaction_digest(executor_binding_digest);
    transcript.interaction_digest(executor_profile.digest());
    transcript.interaction_digest(runtime_incarnation.digest());
    transcript.interaction_digest(connector_digest);
    transcript.interaction_digest(adapter_implementation_digest);
    transcript.authority_digest(authority_snapshot_digest);
    transcript.u64(authority_state_sequence);
    transcript.u64(authority_source_frontier.0);
    transcript.authority_digest(authority_source_frontier.1);
    transcript.bytes32(state_policy_digest);
    transcript.bytes32(time_policy_digest);
    transcript.bytes32(runtime_session_nonce);
    Ok(DispatchIncarnationId(transcript.finish_authority()))
}

fn reject_zero_authority(
    field: &'static str,
    value: AuthorityDigest32,
) -> Result<(), ActionEntryContractError> {
    if value.0 == [0; 32] {
        Err(ActionEntryContractError::ZeroAuthorityDigest(field))
    } else {
        Ok(())
    }
}

fn reject_zero_interaction(
    field: &'static str,
    value: InteractionDigest32,
) -> Result<(), ActionEntryContractError> {
    if value.as_bytes() == &[0; 32] {
        Err(ActionEntryContractError::ZeroInteractionDigest(field))
    } else {
        Ok(())
    }
}

struct Transcript {
    hasher: Sha256,
}

impl Transcript {
    fn new(domain: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(domain);
        Self { hasher }
    }

    fn u16(&mut self, value: u16) {
        self.hasher.update(value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.hasher.update(value.to_be_bytes());
    }

    fn text(&mut self, value: &str) {
        let bytes = value.as_bytes();
        self.hasher.update((bytes.len() as u32).to_be_bytes());
        self.hasher.update(bytes);
    }

    fn bytes32(&mut self, value: [u8; 32]) {
        self.hasher.update(value);
    }

    fn authority_digest(&mut self, value: AuthorityDigest32) {
        self.hasher.update(value.0);
    }

    fn interaction_digest(&mut self, value: InteractionDigest32) {
        self.hasher.update(value.as_bytes());
    }

    fn finish_authority(self) -> AuthorityDigest32 {
        let digest = self.hasher.finalize();
        let mut bytes = [0_u8; 32];
        bytes.copy_from_slice(&digest);
        AuthorityDigest32(bytes)
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ActionEntryContractError {
    #[error("authority executor identifier is empty or oversized")]
    InvalidAuthorityExecutor,
    #[error("{0} must not be an all-zero authority digest")]
    ZeroAuthorityDigest(&'static str),
    #[error("{0} must not be an all-zero interaction digest")]
    ZeroInteractionDigest(&'static str),
    #[error("V2D effect binding names a different authority executor")]
    AuthorityExecutorMismatch,
    #[error("V2D effect parameter commitment does not equal the exact entry binding")]
    EntryCommitmentMismatch,
    #[error("executor challenge names a different semantic principal")]
    ChallengePrincipalMismatch,
    #[error("executor challenge names a different identity requirement")]
    ChallengeRequirementMismatch,
    #[error("executor challenge names a different executor profile")]
    ChallengeExecutorProfileMismatch,
    #[error("verified executor binding does not structurally match: {0}")]
    ExecutorBinding(#[from] ExecutorBindingMismatch),
    #[error("dispatch runtime/session nonce must not be all-zero")]
    ZeroRuntimeSessionNonce,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_executor_identity::{
        ExecutorIdentityDimension, ExecutorIdentityDimensionSet, ExecutorIdentityProfile,
    };
    use symthaea_interaction_core::{
        ConnectorId, IdentityComponent, IdentityOrdering, NamespaceId,
    };

    fn adigest(byte: u8) -> AuthorityDigest32 {
        AuthorityDigest32([byte; 32])
    }

    fn idigest(byte: u8) -> InteractionDigest32 {
        InteractionDigest32::new([byte; 32])
    }

    fn principal(name: &str) -> PrincipalRef {
        PrincipalRef::new(
            NamespaceId::new("intx/workload").unwrap(),
            "executor",
            IdentityOrdering::NamedSet,
            vec![IdentityComponent::new("name", name).unwrap()],
        )
        .unwrap()
    }

    fn requirement() -> ExecutorIdentityRequirement {
        ExecutorIdentityRequirement::new(
            ExecutorIdentityProfile::ConsequentialDigital,
            ExecutorIdentityDimensionSet::new(&[
                ExecutorIdentityDimension::SessionPeer,
                ExecutorIdentityDimension::Workload,
                ExecutorIdentityDimension::Software,
                ExecutorIdentityDimension::ExecutorProfile,
            ]),
        )
        .unwrap()
    }

    fn connector() -> ConnectorIdentity {
        ConnectorIdentity::new(
            ConnectorId::new(NamespaceId::new("intx/connector").unwrap(), "hal").unwrap(),
            "native-v1",
            Some(idigest(0x33)),
        )
        .unwrap()
    }

    fn entry() -> EntryBindingCommitmentV1 {
        EntryBindingCommitmentV1::new(
            adigest(0x11),
            PrincipalId("hal-1".into()),
            principal("hal-1"),
            requirement(),
            ExecutorProfileId::new(idigest(0x22)).unwrap(),
            connector(),
            idigest(0x44),
        )
        .unwrap()
    }

    fn challenge(
        principal_ref: &PrincipalRef,
        requirement: &ExecutorIdentityRequirement,
        executor_profile: ExecutorProfileId,
    ) -> ExecutorIdentityChallenge {
        ExecutorIdentityChallenge::new(
            [0xa3; 32],
            principal_ref,
            executor_profile,
            ExecutorRuntimeIncarnationId::new(idigest(0x58)).unwrap(),
            requirement,
        )
        .unwrap()
    }

    fn binding(parameters_digest: AuthorityDigest32, executor: &str) -> EffectAuthorityBindingV2 {
        EffectAuthorityBindingV2 {
            subject: PrincipalId("robot-1".into()),
            executor: PrincipalId(executor.into()),
            purpose: symthaea_authority::PurposeId("actuation".into()),
            task: None,
            resource: symthaea_authority::ResourceRef("robot-1".into()),
            operation: symthaea_authority::Operation("move".into()),
            plan_digest: None,
            world_digest: None,
            parameters_digest,
            risk_charge: symthaea_authority::RiskBudget {
                mutation_units: 1,
                ..Default::default()
            },
        }
    }

    #[test]
    fn entry_binding_golden_vector_is_stable() {
        assert_eq!(
            entry().digest(),
            AuthorityDigest32([
                0x8c, 0x6f, 0xdc, 0x05, 0xc5, 0xde, 0x75, 0x89,
                0xf5, 0x15, 0x47, 0xcd, 0x54, 0x60, 0x69, 0x05,
                0x44, 0x49, 0xc1, 0x66, 0x4a, 0x16, 0xa5, 0xa1,
                0x31, 0xd9, 0x04, 0xe5, 0xa4, 0x2a, 0x61, 0x23,
            ])
        );
    }

    #[test]
    fn every_entry_identity_component_changes_commitment() {
        let baseline = entry().digest();

        let changed_authority = EntryBindingCommitmentV1::new(
            adigest(0x11),
            PrincipalId("hal-2".into()),
            principal("hal-1"),
            requirement(),
            ExecutorProfileId::new(idigest(0x22)).unwrap(),
            connector(),
            idigest(0x44),
        )
        .unwrap();
        assert_ne!(baseline, changed_authority.digest());

        let changed_semantic = EntryBindingCommitmentV1::new(
            adigest(0x11),
            PrincipalId("hal-1".into()),
            principal("hal-2"),
            requirement(),
            ExecutorProfileId::new(idigest(0x22)).unwrap(),
            connector(),
            idigest(0x44),
        )
        .unwrap();
        assert_ne!(baseline, changed_semantic.digest());

        let changed_domain = EntryBindingCommitmentV1::new(
            adigest(0x12),
            PrincipalId("hal-1".into()),
            principal("hal-1"),
            requirement(),
            ExecutorProfileId::new(idigest(0x22)).unwrap(),
            connector(),
            idigest(0x44),
        )
        .unwrap();
        assert_ne!(baseline, changed_domain.digest());

        let changed_profile = EntryBindingCommitmentV1::new(
            adigest(0x11),
            PrincipalId("hal-1".into()),
            principal("hal-1"),
            requirement(),
            ExecutorProfileId::new(idigest(0x23)).unwrap(),
            connector(),
            idigest(0x44),
        )
        .unwrap();
        assert_ne!(baseline, changed_profile.digest());

        let changed_adapter = EntryBindingCommitmentV1::new(
            adigest(0x11),
            PrincipalId("hal-1".into()),
            principal("hal-1"),
            requirement(),
            ExecutorProfileId::new(idigest(0x22)).unwrap(),
            connector(),
            idigest(0x45),
        )
        .unwrap();
        assert_ne!(baseline, changed_adapter.digest());
    }

    #[test]
    fn effect_binding_must_precommit_exact_entry_profile() {
        let entry = entry();
        entry
            .validate_effect_binding(&binding(entry.digest(), "hal-1"))
            .unwrap();

        assert_eq!(
            entry
                .validate_effect_binding(&binding(adigest(0xee), "hal-1"))
                .unwrap_err(),
            ActionEntryContractError::EntryCommitmentMismatch
        );
        assert_eq!(
            entry
                .validate_effect_binding(&binding(entry.digest(), "hal-2"))
                .unwrap_err(),
            ActionEntryContractError::AuthorityExecutorMismatch
        );
    }

    #[test]
    fn challenge_must_match_precommitted_semantic_identity() {
        let entry = entry();
        let good = challenge(
            entry.semantic_executor(),
            entry.identity_requirement(),
            entry.executor_profile(),
        );
        entry.validate_challenge(&good).unwrap();

        let wrong_principal = challenge(
            &principal("hal-2"),
            entry.identity_requirement(),
            entry.executor_profile(),
        );
        assert_eq!(
            entry.validate_challenge(&wrong_principal).unwrap_err(),
            ActionEntryContractError::ChallengePrincipalMismatch
        );

        let dev_requirement = ExecutorIdentityRequirement::new(
            ExecutorIdentityProfile::DevelopmentSimulation,
            ExecutorIdentityDimensionSet::new(&[ExecutorIdentityDimension::ExecutorProfile]),
        )
        .unwrap();
        let wrong_requirement = challenge(
            entry.semantic_executor(),
            &dev_requirement,
            entry.executor_profile(),
        );
        assert_eq!(
            entry.validate_challenge(&wrong_requirement).unwrap_err(),
            ActionEntryContractError::ChallengeRequirementMismatch
        );

        let wrong_profile = challenge(
            entry.semantic_executor(),
            entry.identity_requirement(),
            ExecutorProfileId::new(idigest(0x23)).unwrap(),
        );
        assert_eq!(
            entry.validate_challenge(&wrong_profile).unwrap_err(),
            ActionEntryContractError::ChallengeExecutorProfileMismatch
        );
    }

    #[test]
    fn dispatch_incarnation_golden_vector_is_stable() {
        let id = derive_dispatch_incarnation_from_parts(
            DispatchPermitId(adigest(0x51)),
            adigest(0x52),
            EffectIntentId(adigest(0x53)),
            AttemptId(adigest(0x54)),
            ReservationId(adigest(0x55)),
            EffectBindingDigest(adigest(0x56)),
            entry().digest(),
            idigest(0x57),
            ExecutorProfileId::new(idigest(0x22)).unwrap(),
            ExecutorRuntimeIncarnationId::new(idigest(0x58)).unwrap(),
            connector().digest(),
            idigest(0x44),
            adigest(0x59),
            12,
            (11, adigest(0x5a)),
            [0x5b; 32],
            [0x5c; 32],
            [0x5d; 32],
        )
        .unwrap();
        assert_eq!(
            id.digest(),
            AuthorityDigest32([
                0x0a, 0xf9, 0x68, 0xc5, 0xc6, 0x1f, 0xc5, 0xa2,
                0xac, 0x29, 0x12, 0x70, 0x8b, 0xae, 0x9b, 0x78,
                0x5b, 0xdd, 0x51, 0x15, 0xdc, 0xba, 0x43, 0x41,
                0x03, 0x6f, 0xf8, 0x17, 0x04, 0xf9, 0x0f, 0x46,
            ])
        );
    }

    #[test]
    fn dispatch_incarnation_changes_with_live_carrier_inputs() {
        let baseline = derive_dispatch_incarnation_from_parts(
            DispatchPermitId(adigest(0x51)), adigest(0x52), EffectIntentId(adigest(0x53)),
            AttemptId(adigest(0x54)), ReservationId(adigest(0x55)),
            EffectBindingDigest(adigest(0x56)), entry().digest(), idigest(0x57),
            ExecutorProfileId::new(idigest(0x22)).unwrap(),
            ExecutorRuntimeIncarnationId::new(idigest(0x58)).unwrap(),
            connector().digest(), idigest(0x44), adigest(0x59), 12,
            (11, adigest(0x5a)), [0x5b; 32], [0x5c; 32], [0x5d; 32],
        ).unwrap();

        let changed_binding = derive_dispatch_incarnation_from_parts(
            DispatchPermitId(adigest(0x51)), adigest(0x52), EffectIntentId(adigest(0x53)),
            AttemptId(adigest(0x54)), ReservationId(adigest(0x55)),
            EffectBindingDigest(adigest(0x56)), entry().digest(), idigest(0x60),
            ExecutorProfileId::new(idigest(0x22)).unwrap(),
            ExecutorRuntimeIncarnationId::new(idigest(0x58)).unwrap(),
            connector().digest(), idigest(0x44), adigest(0x59), 12,
            (11, adigest(0x5a)), [0x5b; 32], [0x5c; 32], [0x5d; 32],
        ).unwrap();
        assert_ne!(baseline, changed_binding);

        let changed_runtime = derive_dispatch_incarnation_from_parts(
            DispatchPermitId(adigest(0x51)), adigest(0x52), EffectIntentId(adigest(0x53)),
            AttemptId(adigest(0x54)), ReservationId(adigest(0x55)),
            EffectBindingDigest(adigest(0x56)), entry().digest(), idigest(0x57),
            ExecutorProfileId::new(idigest(0x22)).unwrap(),
            ExecutorRuntimeIncarnationId::new(idigest(0x61)).unwrap(),
            connector().digest(), idigest(0x44), adigest(0x59), 12,
            (11, adigest(0x5a)), [0x5b; 32], [0x5c; 32], [0x5d; 32],
        ).unwrap();
        assert_ne!(baseline, changed_runtime);

        let changed_frontier = derive_dispatch_incarnation_from_parts(
            DispatchPermitId(adigest(0x51)), adigest(0x52), EffectIntentId(adigest(0x53)),
            AttemptId(adigest(0x54)), ReservationId(adigest(0x55)),
            EffectBindingDigest(adigest(0x56)), entry().digest(), idigest(0x57),
            ExecutorProfileId::new(idigest(0x22)).unwrap(),
            ExecutorRuntimeIncarnationId::new(idigest(0x58)).unwrap(),
            connector().digest(), idigest(0x44), adigest(0x59), 12,
            (13, adigest(0x5a)), [0x5b; 32], [0x5c; 32], [0x5d; 32],
        ).unwrap();
        assert_ne!(baseline, changed_frontier);
    }

    #[test]
    fn zero_runtime_nonce_and_placeholder_digests_fail_closed() {
        assert_eq!(
            EntryBindingCommitmentV1::new(
                AuthorityDigest32([0; 32]),
                PrincipalId("hal-1".into()),
                principal("hal-1"),
                requirement(),
                ExecutorProfileId::new(idigest(0x22)).unwrap(),
                connector(),
                idigest(0x44),
            )
            .unwrap_err(),
            ActionEntryContractError::ZeroAuthorityDigest("domain parameters digest")
        );

        let error = derive_dispatch_incarnation_from_parts(
            DispatchPermitId(adigest(0x51)), adigest(0x52), EffectIntentId(adigest(0x53)),
            AttemptId(adigest(0x54)), ReservationId(adigest(0x55)),
            EffectBindingDigest(adigest(0x56)), entry().digest(), idigest(0x57),
            ExecutorProfileId::new(idigest(0x22)).unwrap(),
            ExecutorRuntimeIncarnationId::new(idigest(0x58)).unwrap(),
            connector().digest(), idigest(0x44), adigest(0x59), 12,
            (11, adigest(0x5a)), [0x5b; 32], [0x5c; 32], [0; 32],
        ).unwrap_err();
        assert_eq!(error, ActionEntryContractError::ZeroRuntimeSessionNonce);
    }
}
