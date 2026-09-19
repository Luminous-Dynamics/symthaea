// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-PRESENTCURRENT-511: conservative presentation-currentness binding.
//!
//! A presentation may be semantically complete and render-artifact exact yet
//! still become stale before confirmation or execution. This module binds an
//! exact `PresentationId` to the policy, intent, authority, information,
//! temporal, transaction, and materiality state roots that were current when the
//! presentation was produced.
//!
//! Version 1 is deliberately conservative: any change to any bound root makes
//! the presentation stale. Later dependency-closure work may prove that some
//! changes are irrelevant and allow narrower currentness snapshots, but this
//! tranche does not infer irrelevance.

use core::fmt;

use crate::assurance::{ActionRequestId, PresentationId, PresentationManifest, ValidationError};
use crate::assurance_materiality::{MaterialityProfileId, PolicyContextId};

const SNAPSHOT_DOMAIN: &[u8] = b"symthaea.presentation.v1/currentness-snapshot\0";
const CERTIFICATE_DOMAIN: &[u8] = b"symthaea.presentation.v1/currentness-certificate\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(pub [u8; 32]);

        impl $name {
            pub const ZERO: Self = Self([0; 32]);

            pub const fn as_bytes(&self) -> &[u8; 32] {
                &self.0
            }

            pub fn is_zero(&self) -> bool {
                self.0 == [0; 32]
            }
        }
    };
}

digest_id!(IntentStateRoot);
digest_id!(AuthorityStateRoot);
digest_id!(InformationStateRoot);
digest_id!(TemporalStateRoot);
digest_id!(TransactionSnapshotRoot);
digest_id!(PresentationSnapshotId);
digest_id!(PresentationFreshnessCertificateId);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PresentationSnapshot {
    pub policy_context_id: PolicyContextId,
    pub policy_generation: u64,
    pub action_request_id: ActionRequestId,
    pub materiality_profile_id: MaterialityProfileId,
    pub intent_state_root: IntentStateRoot,
    pub authority_state_root: AuthorityStateRoot,
    pub information_state_root: InformationStateRoot,
    pub temporal_state_root: TemporalStateRoot,
    pub transaction_snapshot_root: TransactionSnapshotRoot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PresentationFreshnessCertificate {
    pub presentation_id: PresentationId,
    pub presentation_generation: u64,
    pub snapshot_id: PresentationSnapshotId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresentationCurrentnessError {
    ZeroPolicyContext,
    ZeroPolicyGeneration,
    ZeroActionRequest,
    ZeroMaterialityProfile,
    ZeroIntentStateRoot,
    ZeroAuthorityStateRoot,
    ZeroInformationStateRoot,
    ZeroTemporalStateRoot,
    ZeroTransactionSnapshotRoot,
    ZeroPresentationId,
    ZeroPresentationGeneration,
    ZeroSnapshotId,
    ActionMismatch,
    PresentationMismatch,
    PresentationGenerationMismatch,
    PresentedSnapshotMismatch,
    PolicyContextChanged,
    PolicyGenerationChanged,
    MaterialityProfileChanged,
    IntentStateChanged,
    AuthorityStateChanged,
    InformationStateChanged,
    TemporalStateChanged,
    TransactionStateChanged,
    Presentation(ValidationError),
}

impl fmt::Display for PresentationCurrentnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for PresentationCurrentnessError {}

impl From<ValidationError> for PresentationCurrentnessError {
    fn from(value: ValidationError) -> Self {
        Self::Presentation(value)
    }
}

impl PresentationSnapshot {
    pub fn validate(&self) -> Result<(), PresentationCurrentnessError> {
        if self.policy_context_id.is_zero() {
            return Err(PresentationCurrentnessError::ZeroPolicyContext);
        }
        if self.policy_generation == 0 {
            return Err(PresentationCurrentnessError::ZeroPolicyGeneration);
        }
        if self.action_request_id.is_zero() {
            return Err(PresentationCurrentnessError::ZeroActionRequest);
        }
        if self.materiality_profile_id.is_zero() {
            return Err(PresentationCurrentnessError::ZeroMaterialityProfile);
        }
        if self.intent_state_root.is_zero() {
            return Err(PresentationCurrentnessError::ZeroIntentStateRoot);
        }
        if self.authority_state_root.is_zero() {
            return Err(PresentationCurrentnessError::ZeroAuthorityStateRoot);
        }
        if self.information_state_root.is_zero() {
            return Err(PresentationCurrentnessError::ZeroInformationStateRoot);
        }
        if self.temporal_state_root.is_zero() {
            return Err(PresentationCurrentnessError::ZeroTemporalStateRoot);
        }
        if self.transaction_snapshot_root.is_zero() {
            return Err(PresentationCurrentnessError::ZeroTransactionSnapshotRoot);
        }
        Ok(())
    }

    pub fn snapshot_id(&self) -> Result<PresentationSnapshotId, PresentationCurrentnessError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(SNAPSHOT_DOMAIN);
        hasher.update(self.policy_context_id.as_bytes());
        hasher.update(&self.policy_generation.to_le_bytes());
        hasher.update(self.action_request_id.as_bytes());
        hasher.update(self.materiality_profile_id.as_bytes());
        hasher.update(self.intent_state_root.as_bytes());
        hasher.update(self.authority_state_root.as_bytes());
        hasher.update(self.information_state_root.as_bytes());
        hasher.update(self.temporal_state_root.as_bytes());
        hasher.update(self.transaction_snapshot_root.as_bytes());
        Ok(PresentationSnapshotId(*hasher.finalize().as_bytes()))
    }

    fn compare_current(
        &self,
        current: &Self,
    ) -> Result<(), PresentationCurrentnessError> {
        current.validate()?;
        if self.action_request_id != current.action_request_id {
            return Err(PresentationCurrentnessError::ActionMismatch);
        }
        if self.policy_context_id != current.policy_context_id {
            return Err(PresentationCurrentnessError::PolicyContextChanged);
        }
        if self.policy_generation != current.policy_generation {
            return Err(PresentationCurrentnessError::PolicyGenerationChanged);
        }
        if self.materiality_profile_id != current.materiality_profile_id {
            return Err(PresentationCurrentnessError::MaterialityProfileChanged);
        }
        if self.intent_state_root != current.intent_state_root {
            return Err(PresentationCurrentnessError::IntentStateChanged);
        }
        if self.authority_state_root != current.authority_state_root {
            return Err(PresentationCurrentnessError::AuthorityStateChanged);
        }
        if self.information_state_root != current.information_state_root {
            return Err(PresentationCurrentnessError::InformationStateChanged);
        }
        if self.temporal_state_root != current.temporal_state_root {
            return Err(PresentationCurrentnessError::TemporalStateChanged);
        }
        if self.transaction_snapshot_root != current.transaction_snapshot_root {
            return Err(PresentationCurrentnessError::TransactionStateChanged);
        }
        Ok(())
    }
}

impl PresentationFreshnessCertificate {
    pub fn from_manifest(
        manifest: &PresentationManifest,
        presented_snapshot: &PresentationSnapshot,
    ) -> Result<Self, PresentationCurrentnessError> {
        manifest.validate()?;
        presented_snapshot.validate()?;
        if manifest.action_request_id != presented_snapshot.action_request_id {
            return Err(PresentationCurrentnessError::ActionMismatch);
        }
        Ok(Self {
            presentation_id: manifest.presentation_id()?,
            presentation_generation: manifest.presentation_generation,
            snapshot_id: presented_snapshot.snapshot_id()?,
        })
    }

    pub fn certificate_id(
        &self,
    ) -> Result<PresentationFreshnessCertificateId, PresentationCurrentnessError> {
        self.validate_nonzero()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(CERTIFICATE_DOMAIN);
        hasher.update(self.presentation_id.as_bytes());
        hasher.update(&self.presentation_generation.to_le_bytes());
        hasher.update(self.snapshot_id.as_bytes());
        Ok(PresentationFreshnessCertificateId(
            *hasher.finalize().as_bytes(),
        ))
    }

    /// Requires both that the certificate still describes the presentation that
    /// was shown and that every conservatively bound state root is unchanged.
    pub fn verify_current(
        &self,
        manifest: &PresentationManifest,
        presented_snapshot: &PresentationSnapshot,
        current_snapshot: &PresentationSnapshot,
    ) -> Result<(), PresentationCurrentnessError> {
        self.validate_nonzero()?;
        manifest.validate()?;
        presented_snapshot.validate()?;
        current_snapshot.validate()?;

        if manifest.action_request_id != presented_snapshot.action_request_id {
            return Err(PresentationCurrentnessError::ActionMismatch);
        }
        if self.presentation_id != manifest.presentation_id()? {
            return Err(PresentationCurrentnessError::PresentationMismatch);
        }
        if self.presentation_generation != manifest.presentation_generation {
            return Err(PresentationCurrentnessError::PresentationGenerationMismatch);
        }
        if self.snapshot_id != presented_snapshot.snapshot_id()? {
            return Err(PresentationCurrentnessError::PresentedSnapshotMismatch);
        }

        presented_snapshot.compare_current(current_snapshot)
    }

    fn validate_nonzero(&self) -> Result<(), PresentationCurrentnessError> {
        if self.presentation_id.is_zero() {
            return Err(PresentationCurrentnessError::ZeroPresentationId);
        }
        if self.presentation_generation == 0 {
            return Err(PresentationCurrentnessError::ZeroPresentationGeneration);
        }
        if self.snapshot_id.is_zero() {
            return Err(PresentationCurrentnessError::ZeroSnapshotId);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance::{
        IntentId, LocaleProfileId, MaterialEffect, MaterialField, MaterialFieldKind,
        PresentedField, PresentationContextId, PrincipalId, RenderedCommitment,
        RendererProfileId, SemanticCommitment,
    };

    fn bytes(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    fn manifest() -> PresentationManifest {
        PresentationManifest {
            action_request_id: ActionRequestId(bytes(1)),
            intent_id: IntentId(bytes(2)),
            principal_id: PrincipalId(bytes(3)),
            presentation_context_id: PresentationContextId(bytes(4)),
            renderer_profile_id: RendererProfileId(bytes(5)),
            locale_profile_id: LocaleProfileId(bytes(6)),
            presentation_generation: 7,
            material_effect: MaterialEffect {
                fields: vec![MaterialField {
                    kind: MaterialFieldKind::ActionClass,
                    semantic: SemanticCommitment(bytes(10)),
                }],
            },
            presented_fields: vec![PresentedField {
                kind: MaterialFieldKind::ActionClass,
                semantic: SemanticCommitment(bytes(10)),
                rendered: RenderedCommitment(bytes(20)),
            }],
            warnings: vec![],
        }
    }

    fn snapshot() -> PresentationSnapshot {
        PresentationSnapshot {
            policy_context_id: PolicyContextId(bytes(30)),
            policy_generation: 4,
            action_request_id: ActionRequestId(bytes(1)),
            materiality_profile_id: MaterialityProfileId(bytes(31)),
            intent_state_root: IntentStateRoot(bytes(32)),
            authority_state_root: AuthorityStateRoot(bytes(33)),
            information_state_root: InformationStateRoot(bytes(34)),
            temporal_state_root: TemporalStateRoot(bytes(35)),
            transaction_snapshot_root: TransactionSnapshotRoot(bytes(36)),
        }
    }

    #[test]
    fn exact_snapshot_is_current() {
        let m = manifest();
        let presented = snapshot();
        let cert = PresentationFreshnessCertificate::from_manifest(&m, &presented).unwrap();
        assert!(cert.verify_current(&m, &presented, &presented).is_ok());
    }

    #[test]
    fn intent_state_drift_invalidates_presentation() {
        let m = manifest();
        let presented = snapshot();
        let cert = PresentationFreshnessCertificate::from_manifest(&m, &presented).unwrap();
        let mut current = presented;
        current.intent_state_root = IntentStateRoot(bytes(80));
        assert_eq!(
            cert.verify_current(&m, &presented, &current),
            Err(PresentationCurrentnessError::IntentStateChanged)
        );
    }

    #[test]
    fn authority_state_drift_invalidates_presentation() {
        let m = manifest();
        let presented = snapshot();
        let cert = PresentationFreshnessCertificate::from_manifest(&m, &presented).unwrap();
        let mut current = presented;
        current.authority_state_root = AuthorityStateRoot(bytes(81));
        assert_eq!(
            cert.verify_current(&m, &presented, &current),
            Err(PresentationCurrentnessError::AuthorityStateChanged)
        );
    }

    #[test]
    fn information_state_drift_invalidates_presentation() {
        let m = manifest();
        let presented = snapshot();
        let cert = PresentationFreshnessCertificate::from_manifest(&m, &presented).unwrap();
        let mut current = presented;
        current.information_state_root = InformationStateRoot(bytes(82));
        assert_eq!(
            cert.verify_current(&m, &presented, &current),
            Err(PresentationCurrentnessError::InformationStateChanged)
        );
    }

    #[test]
    fn temporal_state_drift_invalidates_presentation() {
        let m = manifest();
        let presented = snapshot();
        let cert = PresentationFreshnessCertificate::from_manifest(&m, &presented).unwrap();
        let mut current = presented;
        current.temporal_state_root = TemporalStateRoot(bytes(83));
        assert_eq!(
            cert.verify_current(&m, &presented, &current),
            Err(PresentationCurrentnessError::TemporalStateChanged)
        );
    }

    #[test]
    fn transaction_state_drift_invalidates_presentation() {
        let m = manifest();
        let presented = snapshot();
        let cert = PresentationFreshnessCertificate::from_manifest(&m, &presented).unwrap();
        let mut current = presented;
        current.transaction_snapshot_root = TransactionSnapshotRoot(bytes(84));
        assert_eq!(
            cert.verify_current(&m, &presented, &current),
            Err(PresentationCurrentnessError::TransactionStateChanged)
        );
    }

    #[test]
    fn policy_context_drift_invalidates_presentation() {
        let m = manifest();
        let presented = snapshot();
        let cert = PresentationFreshnessCertificate::from_manifest(&m, &presented).unwrap();
        let mut current = presented;
        current.policy_context_id = PolicyContextId(bytes(85));
        assert_eq!(
            cert.verify_current(&m, &presented, &current),
            Err(PresentationCurrentnessError::PolicyContextChanged)
        );
    }

    #[test]
    fn policy_generation_drift_invalidates_presentation() {
        let m = manifest();
        let presented = snapshot();
        let cert = PresentationFreshnessCertificate::from_manifest(&m, &presented).unwrap();
        let mut current = presented;
        current.policy_generation += 1;
        assert_eq!(
            cert.verify_current(&m, &presented, &current),
            Err(PresentationCurrentnessError::PolicyGenerationChanged)
        );
    }

    #[test]
    fn materiality_profile_drift_invalidates_presentation() {
        let m = manifest();
        let presented = snapshot();
        let cert = PresentationFreshnessCertificate::from_manifest(&m, &presented).unwrap();
        let mut current = presented;
        current.materiality_profile_id = MaterialityProfileId(bytes(86));
        assert_eq!(
            cert.verify_current(&m, &presented, &current),
            Err(PresentationCurrentnessError::MaterialityProfileChanged)
        );
    }

    #[test]
    fn action_drift_invalidates_presentation() {
        let m = manifest();
        let presented = snapshot();
        let cert = PresentationFreshnessCertificate::from_manifest(&m, &presented).unwrap();
        let mut current = presented;
        current.action_request_id = ActionRequestId(bytes(87));
        assert_eq!(
            cert.verify_current(&m, &presented, &current),
            Err(PresentationCurrentnessError::ActionMismatch)
        );
    }

    #[test]
    fn presentation_generation_drift_invalidates_certificate() {
        let m = manifest();
        let presented = snapshot();
        let cert = PresentationFreshnessCertificate::from_manifest(&m, &presented).unwrap();
        let mut changed = m.clone();
        changed.presentation_generation += 1;
        assert_eq!(
            cert.verify_current(&changed, &presented, &presented),
            Err(PresentationCurrentnessError::PresentationMismatch)
        );
    }
}
