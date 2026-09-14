// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Composition theorem between signed governance authorization and the
//! qualifier anti-rollback state machine.
//!
//! The lower-level anti-rollback module intentionally retains unsigned
//! transition mechanics for isolated state-machine and filesystem fixtures. This
//! module defines the stronger path intended for future real admission: an
//! authority transition must consume a signed governance authorization that
//! names the exact predecessor/currentness/successor represented by the guard.

#![allow(dead_code)]

use super::v2_qualifier_anti_rollback::{
    V2QualifierAntiRollbackCheckpoint, V2QualifierAntiRollbackError,
    V2QualifierAntiRollbackGuard,
};
use super::v2_qualifier_authority::V2QualifierAuthorityRecord;
use super::v2_qualifier_currentness::V2QualifierAuthorityLineage;
use super::v2_qualifier_signer_policy::V2QualifierSignerPolicy;
use super::v2_signed_governance_authorization::V2SignedGovernanceAuthorization;

pub(super) const V2_AUTHORIZED_AUTHORITY_TRANSITION_REVISION: &str =
    "EUREKA.002.V2.AUTHORIZED_AUTHORITY_TRANSITION.v1";

/// One signed-governance-authorized anti-rollback authority transition.
///
/// The transition owns the governance authorization so the same move-only token
/// cannot be reused to authorize a second successor through this API. This still
/// is not durable: #3148-style persistence must commit `successor_checkpoint`
/// before the successor may be treated as restart-safe current state.
#[derive(Debug)]
pub(super) struct V2AuthorizedAuthorityTransition {
    predecessor_checkpoint: V2QualifierAntiRollbackCheckpoint,
    successor_checkpoint: V2QualifierAntiRollbackCheckpoint,
    authorization: V2SignedGovernanceAuthorization,
    commitment: [u8; 32],
}

impl V2AuthorizedAuthorityTransition {
    pub(super) fn predecessor_checkpoint(&self) -> &V2QualifierAntiRollbackCheckpoint {
        &self.predecessor_checkpoint
    }

    pub(super) fn successor_checkpoint(&self) -> &V2QualifierAntiRollbackCheckpoint {
        &self.successor_checkpoint
    }

    pub(super) fn authorization(&self) -> &V2SignedGovernanceAuthorization {
        &self.authorization
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2AuthorizedAuthorityTransitionError {
    AntiRollback(V2QualifierAntiRollbackError),
    GovernancePredecessorMismatch,
    GovernanceCurrentnessMismatch,
    GovernanceSignerPolicyMismatch,
    GovernanceSuccessorMismatch,
    ResultingCheckpointMismatch,
}

/// Consume one signed governance authorization and advance the exact current
/// anti-rollback authority state it names.
pub(super) fn advance_authority_with_signed_governance(
    guard: V2QualifierAntiRollbackGuard,
    lineage: V2QualifierAuthorityLineage,
    signer_policy: &V2QualifierSignerPolicy,
    successor_authority: V2QualifierAuthorityRecord,
    authorization: V2SignedGovernanceAuthorization,
) -> Result<
    (
        V2AuthorizedAuthorityTransition,
        V2QualifierAntiRollbackGuard,
        V2QualifierAuthorityLineage,
    ),
    V2AuthorizedAuthorityTransitionError,
> {
    guard
        .verify_current(&lineage, signer_policy)
        .map_err(V2AuthorizedAuthorityTransitionError::AntiRollback)?;

    let predecessor_checkpoint = guard.checkpoint().clone();
    if authorization.predecessor_authority_commitment()
        != predecessor_checkpoint.authority_commitment()
    {
        return Err(V2AuthorizedAuthorityTransitionError::GovernancePredecessorMismatch);
    }
    if authorization.predecessor_currentness_commitment()
        != predecessor_checkpoint.currentness_commitment()
    {
        return Err(V2AuthorizedAuthorityTransitionError::GovernanceCurrentnessMismatch);
    }
    if authorization.signer_policy_sequence() != signer_policy.sequence()
        || authorization.signer_policy_commitment() != signer_policy.commitment()
        || predecessor_checkpoint.signer_policy_sequence() != signer_policy.sequence()
        || predecessor_checkpoint.signer_policy_commitment() != signer_policy.commitment()
    {
        return Err(V2AuthorizedAuthorityTransitionError::GovernanceSignerPolicyMismatch);
    }
    if authorization.authority_sequence() != successor_authority.sequence()
        || authorization.authority_commitment() != successor_authority.commitment()
    {
        return Err(V2AuthorizedAuthorityTransitionError::GovernanceSuccessorMismatch);
    }

    let authorization_commitment = authorization.commitment();
    let (next_guard, next_lineage) = guard
        .advance_authority(lineage, successor_authority, signer_policy)
        .map_err(V2AuthorizedAuthorityTransitionError::AntiRollback)?;
    let successor_checkpoint = next_guard.checkpoint().clone();

    if successor_checkpoint.predecessor_checkpoint_commitment()
        != Some(predecessor_checkpoint.commitment())
        || successor_checkpoint.authority_sequence() != authorization.authority_sequence()
        || successor_checkpoint.authority_commitment() != authorization.authority_commitment()
        || successor_checkpoint.currentness_commitment() != next_lineage.commitment()
        || successor_checkpoint.signer_policy_sequence() != signer_policy.sequence()
        || successor_checkpoint.signer_policy_commitment() != signer_policy.commitment()
    {
        return Err(V2AuthorizedAuthorityTransitionError::ResultingCheckpointMismatch);
    }

    let mut transition = V2AuthorizedAuthorityTransition {
        predecessor_checkpoint,
        successor_checkpoint,
        authorization,
        commitment: [0_u8; 32],
    };
    transition.commitment = transition_commitment(&transition, authorization_commitment);

    Ok((transition, next_guard, next_lineage))
}

fn transition_commitment(
    transition: &V2AuthorizedAuthorityTransition,
    authorization_commitment: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_AUTHORIZED_AUTHORITY_TRANSITION_REVISION.as_bytes(),
    );
    bytes.extend_from_slice(&transition.predecessor_checkpoint.commitment());
    bytes.extend_from_slice(&transition.successor_checkpoint.commitment());
    bytes.extend_from_slice(&authorization_commitment);
    bytes.extend_from_slice(&transition.authorization.predecessor_authority_commitment());
    bytes.extend_from_slice(&transition.authorization.predecessor_currentness_commitment());
    bytes.extend_from_slice(&transition.authorization.authority_sequence().to_le_bytes());
    bytes.extend_from_slice(&transition.authorization.authority_commitment());
    *blake3::hash(&bytes).as_bytes()
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::super::v2_openssh_admission_verifier::V2OpenSshAdmissionSignature;
    use super::super::v2_qualifier_authority::{
        V2QualifierAuthorityProfile, V2QualifierAuthorityRecord,
    };
    use super::super::v2_qualifier_signer_policy::{
        V2GovernanceSigner, V2QualifierSignerPolicy,
    };
    use super::super::v2_signed_governance_authorization::authorize_signed_authority_rotation;
    use super::*;

    const MANIFEST: &[u8] = include_bytes!("fixtures/eureka-v2-signed-transition-rotation.env");
    const ALPHA_PUB: &str = include_str!("fixtures/eureka-v2-signed-transition-alpha.pub");
    const BETA_PUB: &str = include_str!("fixtures/eureka-v2-signed-transition-beta.pub");
    const ALPHA_SIG: &[u8] = include_bytes!("fixtures/eureka-v2-signed-transition-alpha.sig");
    const BETA_SIG: &[u8] = include_bytes!("fixtures/eureka-v2-signed-transition-beta.sig");
    const ALPHA_PRINCIPAL: &str = "fixture-transition-alpha@example.invalid";
    const BETA_PRINCIPAL: &str = "fixture-transition-beta@example.invalid";
    const ALPHA_KEY_SHA256: &str =
        "98c9c52c03393af5fabb34d1a41a60a87073525c52f5fb7ca0a51c7de931da2d";
    const BETA_KEY_SHA256: &str =
        "70535ce88cf2b7575de01e7e98559e95fa66d18dd7317f9411500f3930dcfbd7";

    fn policy() -> V2QualifierSignerPolicy {
        let alpha = V2GovernanceSigner::from_hex(
            ALPHA_PRINCIPAL,
            "ssh-ed25519",
            ALPHA_KEY_SHA256,
        )
        .unwrap();
        let beta = V2GovernanceSigner::from_hex(
            BETA_PRINCIPAL,
            "ssh-ed25519",
            BETA_KEY_SHA256,
        )
        .unwrap();
        V2QualifierSignerPolicy::genesis(1, 2, vec![beta, alpha]).unwrap()
    }

    fn signatures() -> [V2OpenSshAdmissionSignature<'static>; 2] {
        [
            V2OpenSshAdmissionSignature::new(ALPHA_PRINCIPAL, ALPHA_PUB, ALPHA_SIG),
            V2OpenSshAdmissionSignature::new(BETA_PRINCIPAL, BETA_PUB, BETA_SIG),
        ]
    }

    fn predecessor() -> V2QualifierAuthorityRecord {
        V2QualifierAuthorityRecord::genesis(
            1,
            V2QualifierAuthorityProfile::current_from_hex(&"c".repeat(64), &"d".repeat(64))
                .unwrap(),
        )
        .unwrap()
    }

    fn successor(predecessor: &V2QualifierAuthorityRecord) -> V2QualifierAuthorityRecord {
        V2QualifierAuthorityRecord::rotate(
            predecessor,
            predecessor.commitment(),
            2,
            V2QualifierAuthorityProfile::current_from_hex(&"e".repeat(64), &"d".repeat(64))
                .unwrap(),
        )
        .unwrap()
    }

    fn authorization(
        lineage: &V2QualifierAuthorityLineage,
        predecessor: &V2QualifierAuthorityRecord,
        successor: &V2QualifierAuthorityRecord,
        policy: &V2QualifierSignerPolicy,
    ) -> V2SignedGovernanceAuthorization {
        authorize_signed_authority_rotation(
            MANIFEST,
            policy,
            &signatures(),
            lineage,
            predecessor,
            successor,
        )
        .unwrap()
    }

    #[test]
    fn signed_governance_is_consumed_into_exact_anti_rollback_transition() {
        let predecessor = predecessor();
        let successor = successor(&predecessor);
        let lineage = V2QualifierAuthorityLineage::activate_genesis(predecessor.clone()).unwrap();
        let policy = policy();
        let guard = V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy).unwrap();
        let authorization = authorization(&lineage, &predecessor, &successor, &policy);
        let authorization_commitment = authorization.commitment();

        let (transition, next_guard, next_lineage) = advance_authority_with_signed_governance(
            guard,
            lineage,
            &policy,
            successor,
            authorization,
        )
        .unwrap();

        assert_eq!(
            transition.predecessor_checkpoint().authority_commitment(),
            predecessor.commitment()
        );
        assert_eq!(transition.successor_checkpoint(), next_guard.checkpoint());
        assert_eq!(
            transition.successor_checkpoint().authority_commitment(),
            next_lineage.current_authority_commitment()
        );
        assert_eq!(
            transition.authorization().commitment(),
            authorization_commitment
        );
        assert_ne!(transition.commitment(), [0_u8; 32]);
    }

    #[test]
    fn authorization_for_one_successor_cannot_advance_a_sibling() {
        let predecessor = predecessor();
        let approved = successor(&predecessor);
        let sibling = V2QualifierAuthorityRecord::rotate(
            &predecessor,
            predecessor.commitment(),
            2,
            V2QualifierAuthorityProfile::current_from_hex(&"e".repeat(64), &"f".repeat(64))
                .unwrap(),
        )
        .unwrap();
        let lineage = V2QualifierAuthorityLineage::activate_genesis(predecessor.clone()).unwrap();
        let policy = policy();
        let guard = V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy).unwrap();
        let authorization = authorization(&lineage, &predecessor, &approved, &policy);

        assert_eq!(
            advance_authority_with_signed_governance(
                guard,
                lineage,
                &policy,
                sibling,
                authorization,
            )
            .unwrap_err(),
            V2AuthorizedAuthorityTransitionError::GovernanceSuccessorMismatch
        );
    }

    #[test]
    fn authorization_cannot_cross_signer_policy_state() {
        let predecessor = predecessor();
        let successor = successor(&predecessor);
        let lineage = V2QualifierAuthorityLineage::activate_genesis(predecessor.clone()).unwrap();
        let policy = policy();
        let guard = V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy).unwrap();
        let authorization = authorization(&lineage, &predecessor, &successor, &policy);
        let replacement_signer = V2GovernanceSigner::from_hex(
            "replacement@example.invalid",
            "ssh-ed25519",
            &"9".repeat(64),
        )
        .unwrap();
        let changed_policy = V2QualifierSignerPolicy::rotate(
            &policy,
            policy.commitment(),
            2,
            1,
            vec![replacement_signer],
        )
        .unwrap();

        assert_eq!(
            advance_authority_with_signed_governance(
                guard,
                lineage,
                &changed_policy,
                successor,
                authorization,
            )
            .unwrap_err(),
            V2AuthorizedAuthorityTransitionError::AntiRollback(
                V2QualifierAntiRollbackError::CurrentStateMismatch
            )
        );
    }

    #[test]
    fn authorized_transition_source_has_no_root_persistence_or_execution_surface() {
        let production = include_str!("v2_authorized_anti_rollback_transition.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "V2AdmittedQualifierRoot",
            "persist_successor",
            "V2CanaryAuthorization",
            "predict_ticket",
            "reveal(",
            "score_consequence",
            "execution_authority_granted=true",
            "PRIVATE KEY",
        ] {
            assert!(
                !production.contains(forbidden),
                "forbidden authorized-transition surface: {forbidden}"
            );
        }
        assert!(production.contains("authorization.authority_commitment()"));
        assert!(production.contains("guard.advance_authority"));
    }
}
