// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-ANDROIDPOLICYSTORE-531: Rust-owned in-memory policy-session ownership.
//!
//! QUAL-ANDROIDPOLICYSESSION-530 defines exact install/replace/revoke/currentness
//! transitions, but a standalone state value can still be paired with caller-
//! supplied policy evidence. This tranche owns both the complete 529 bundle and
//! the 530 session state behind one store boundary. Event-time callers may name an
//! expected session state, but they do not supply the policy bundle being judged.
//!
//! The store computes every transition and transition-receipt identity before
//! mutating its fields, so a rejected transition leaves the prior in-memory state
//! intact. This is single-process Rust ownership, not durable or distributed
//! atomic persistence; a later engine/native-store tranche must supply that layer.

use core::fmt;

use crate::assurance_android_policy_bundle::{
    AndroidTouchPolicyBundle, AndroidTouchPolicyBundleError, AndroidTouchPolicyBundleId,
};
use crate::assurance_android_policy_session::{
    AndroidTouchPolicyCurrentCertificate, AndroidTouchPolicySessionAuthorizationId,
    AndroidTouchPolicySessionError, AndroidTouchPolicySessionExpectation,
    AndroidTouchPolicySessionId, AndroidTouchPolicySessionNonce,
    AndroidTouchPolicySessionState, AndroidTouchPolicySessionStateId,
    AndroidTouchPolicySessionStatus, AndroidTouchPolicySessionTransitionReceipt,
    AndroidTouchPolicySessionTransitionReceiptId,
};

#[derive(Clone, Debug)]
pub struct AndroidTouchPolicySessionStore {
    state: Option<AndroidTouchPolicySessionState>,
    bundle: Option<AndroidTouchPolicyBundle>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTouchPolicyStoreSnapshot {
    pub session_id: AndroidTouchPolicySessionId,
    pub generation: u64,
    pub state_id: AndroidTouchPolicySessionStateId,
    pub bundle_id: AndroidTouchPolicyBundleId,
    pub status: AndroidTouchPolicySessionStatus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTouchPolicyStoreTransition {
    pub receipt: AndroidTouchPolicySessionTransitionReceipt,
    pub receipt_id: AndroidTouchPolicySessionTransitionReceiptId,
    pub snapshot: AndroidTouchPolicyStoreSnapshot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AndroidTouchPolicyStoreError {
    Empty,
    AlreadyInstalled,
    StoreInvariantViolation,
    Session(AndroidTouchPolicySessionError),
    Bundle(AndroidTouchPolicyBundleError),
}

impl fmt::Display for AndroidTouchPolicyStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AndroidTouchPolicyStoreError {}

impl From<AndroidTouchPolicySessionError> for AndroidTouchPolicyStoreError {
    fn from(value: AndroidTouchPolicySessionError) -> Self {
        Self::Session(value)
    }
}

impl From<AndroidTouchPolicyBundleError> for AndroidTouchPolicyStoreError {
    fn from(value: AndroidTouchPolicyBundleError) -> Self {
        Self::Bundle(value)
    }
}

impl Default for AndroidTouchPolicySessionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl AndroidTouchPolicySessionStore {
    pub const fn new() -> Self {
        Self {
            state: None,
            bundle: None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.state.is_none() && self.bundle.is_none()
    }

    pub fn validate(&self) -> Result<(), AndroidTouchPolicyStoreError> {
        match (&self.state, &self.bundle) {
            (None, None) => Ok(()),
            (Some(state), Some(bundle)) => {
                state.validate()?;
                let identity = bundle.validate()?;
                if identity.bundle_id != state.bundle_id {
                    return Err(AndroidTouchPolicyStoreError::StoreInvariantViolation);
                }
                Ok(())
            }
            _ => Err(AndroidTouchPolicyStoreError::StoreInvariantViolation),
        }
    }

    pub fn snapshot(&self) -> Result<AndroidTouchPolicyStoreSnapshot, AndroidTouchPolicyStoreError> {
        self.validate()?;
        let state = self.state.as_ref().ok_or(AndroidTouchPolicyStoreError::Empty)?;
        Ok(AndroidTouchPolicyStoreSnapshot {
            session_id: state.session_id,
            generation: state.generation,
            state_id: state.state_id()?,
            bundle_id: state.bundle_id,
            status: state.status,
        })
    }

    pub fn expectation(
        &self,
    ) -> Result<AndroidTouchPolicySessionExpectation, AndroidTouchPolicyStoreError> {
        let snapshot = self.snapshot()?;
        Ok(AndroidTouchPolicySessionExpectation {
            session_id: snapshot.session_id,
            generation: snapshot.generation,
            state_id: snapshot.state_id,
            bundle_id: snapshot.bundle_id,
        })
    }

    pub fn install(
        &mut self,
        bundle: &AndroidTouchPolicyBundle,
        authorization_id: AndroidTouchPolicySessionAuthorizationId,
        nonce: AndroidTouchPolicySessionNonce,
    ) -> Result<AndroidTouchPolicyStoreTransition, AndroidTouchPolicyStoreError> {
        self.validate()?;
        if !self.is_empty() {
            return Err(AndroidTouchPolicyStoreError::AlreadyInstalled);
        }

        let (next_state, receipt) =
            AndroidTouchPolicySessionState::install(bundle, authorization_id, nonce)?;
        let receipt_id = receipt.receipt_id()?;
        let state_id = next_state.state_id()?;
        let snapshot = AndroidTouchPolicyStoreSnapshot {
            session_id: next_state.session_id,
            generation: next_state.generation,
            state_id,
            bundle_id: next_state.bundle_id,
            status: next_state.status,
        };

        // Commit only after every semantic identity above has succeeded.
        self.state = Some(next_state);
        self.bundle = Some(*bundle);
        Ok(AndroidTouchPolicyStoreTransition {
            receipt,
            receipt_id,
            snapshot,
        })
    }

    pub fn replace(
        &mut self,
        expected_state_id: AndroidTouchPolicySessionStateId,
        bundle: &AndroidTouchPolicyBundle,
        authorization_id: AndroidTouchPolicySessionAuthorizationId,
    ) -> Result<AndroidTouchPolicyStoreTransition, AndroidTouchPolicyStoreError> {
        self.validate()?;
        let current = *self.state.as_ref().ok_or(AndroidTouchPolicyStoreError::Empty)?;

        let (next_state, receipt) =
            current.replace(expected_state_id, bundle, authorization_id)?;
        let receipt_id = receipt.receipt_id()?;
        let state_id = next_state.state_id()?;
        let snapshot = AndroidTouchPolicyStoreSnapshot {
            session_id: next_state.session_id,
            generation: next_state.generation,
            state_id,
            bundle_id: next_state.bundle_id,
            status: next_state.status,
        };

        self.state = Some(next_state);
        self.bundle = Some(*bundle);
        Ok(AndroidTouchPolicyStoreTransition {
            receipt,
            receipt_id,
            snapshot,
        })
    }

    pub fn revoke(
        &mut self,
        expected_state_id: AndroidTouchPolicySessionStateId,
        authorization_id: AndroidTouchPolicySessionAuthorizationId,
    ) -> Result<AndroidTouchPolicyStoreTransition, AndroidTouchPolicyStoreError> {
        self.validate()?;
        let current = *self.state.as_ref().ok_or(AndroidTouchPolicyStoreError::Empty)?;

        let (next_state, receipt) = current.revoke(expected_state_id, authorization_id)?;
        let receipt_id = receipt.receipt_id()?;
        let state_id = next_state.state_id()?;
        let snapshot = AndroidTouchPolicyStoreSnapshot {
            session_id: next_state.session_id,
            generation: next_state.generation,
            state_id,
            bundle_id: next_state.bundle_id,
            status: next_state.status,
        };

        self.state = Some(next_state);
        // The bundle remains present as immutable historical/current-session policy
        // evidence even though currentness is now revoked.
        Ok(AndroidTouchPolicyStoreTransition {
            receipt,
            receipt_id,
            snapshot,
        })
    }

    pub fn verify_current(
        &self,
        expected: &AndroidTouchPolicySessionExpectation,
    ) -> Result<AndroidTouchPolicyCurrentCertificate, AndroidTouchPolicyStoreError> {
        self.validate()?;
        let state = self.state.as_ref().ok_or(AndroidTouchPolicyStoreError::Empty)?;
        let bundle = self.bundle.as_ref().ok_or(AndroidTouchPolicyStoreError::Empty)?;
        Ok(state.verify_current(expected, bundle)?)
    }

    pub fn active_bundle_id(
        &self,
    ) -> Result<AndroidTouchPolicyBundleId, AndroidTouchPolicyStoreError> {
        let snapshot = self.snapshot()?;
        if snapshot.status == AndroidTouchPolicySessionStatus::Revoked {
            return Err(AndroidTouchPolicyStoreError::Session(
                AndroidTouchPolicySessionError::Revoked,
            ));
        }
        Ok(snapshot.bundle_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance_android_attention::{
        AndroidAttentionPolicyContextId, AndroidAttentionPolicyStateRoot,
        AndroidAttentionRequirement,
    };
    use crate::assurance_android_attention_policy::AndroidAttentionPolicyAdmission;
    use crate::assurance_android_motion_event::AndroidMotionEventRequirement;
    use crate::assurance_android_motion_policy::AndroidMotionPolicyAdmission;
    use crate::assurance_ingress_policy::{
        IngressAdmissionPolicy, IngressPolicyContextId, IngressPolicyStateRoot,
    };
    use crate::assurance_platform_ingress::{PlatformIngressKind, PlatformIngressProfile};
    use crate::assurance_soma_interaction::{ScreenCaptureProfileId, TouchInputProfileId};
    use symthaea_core::assurance_interaction_continuity::InputAttesterId;
    use symthaea_core::assurance_trusted_surface::TrustedSurfaceId;

    fn d(value: u8) -> [u8; 32] { [value; 32] }
    fn auth(value: u8) -> AndroidTouchPolicySessionAuthorizationId {
        AndroidTouchPolicySessionAuthorizationId(d(value))
    }
    fn nonce(value: u8) -> AndroidTouchPolicySessionNonce {
        AndroidTouchPolicySessionNonce(d(value))
    }

    fn bundle(profile_seed: u8) -> AndroidTouchPolicyBundle {
        let attention_requirement = AndroidAttentionRequirement {
            policy_context_id: AndroidAttentionPolicyContextId(d(1)),
            policy_generation: 4,
            policy_state_root: AndroidAttentionPolicyStateRoot(d(2)),
            trusted_surface_id: TrustedSurfaceId(d(3)),
            minimum_sdk_int: 31,
            require_window_focus: true,
            require_flag_secure: true,
            require_hide_application_overlays: true,
            require_filter_touches_when_obscured: true,
            require_view_attached: true,
            require_view_shown: true,
            require_top_resumed: true,
            forbid_multi_window: true,
        };
        let attention_admission = AndroidAttentionPolicyAdmission {
            policy_context_id: attention_requirement.policy_context_id,
            policy_generation: attention_requirement.policy_generation,
            policy_state_root: attention_requirement.policy_state_root,
            authorized_requirement_id: attention_requirement.requirement_id().unwrap(),
        };
        let motion_requirement = AndroidMotionEventRequirement {
            attention_requirement_id: attention_requirement.requirement_id().unwrap(),
            reject_fully_obscured: true,
            reject_partially_obscured: true,
            require_single_pointer: true,
        };
        let motion_admission = AndroidMotionPolicyAdmission {
            policy_context_id: attention_requirement.policy_context_id,
            policy_generation: attention_requirement.policy_generation,
            policy_state_root: attention_requirement.policy_state_root,
            attention_policy_admission_id: attention_admission.admission_id().unwrap(),
            authorized_motion_requirement_id: motion_requirement.requirement_id().unwrap(),
        };
        let profile = PlatformIngressProfile {
            platform: PlatformIngressKind::AndroidJni,
            abi_version: 1,
            capture_profile_id: ScreenCaptureProfileId(d(profile_seed)),
            touch_input_profile_id: TouchInputProfileId(d(profile_seed.wrapping_add(1))),
            max_frame_bytes: 1024,
        };
        let policy = IngressAdmissionPolicy {
            policy_context_id: IngressPolicyContextId(d(22)),
            policy_generation: 7,
            policy_state_root: IngressPolicyStateRoot(d(23)),
            platform: PlatformIngressKind::AndroidJni,
            ingress_profile_id: profile.profile_id().unwrap(),
            trusted_surface_id: attention_requirement.trusted_surface_id,
            input_attester_id: InputAttesterId(d(8)),
            allow_frame: false,
            allow_touch: true,
        };
        AndroidTouchPolicyBundle {
            ingress_policy: policy,
            ingress_profile: profile,
            attention_policy_admission: attention_admission,
            attention_requirement,
            motion_policy_admission: motion_admission,
            motion_requirement,
        }
    }

    #[test]
    fn empty_store_has_no_current_policy() {
        let store = AndroidTouchPolicySessionStore::new();
        assert!(store.is_empty());
        assert_eq!(store.snapshot(), Err(AndroidTouchPolicyStoreError::Empty));
    }

    #[test]
    fn install_owns_bundle_and_currentness() {
        let mut store = AndroidTouchPolicySessionStore::new();
        let b = bundle(20);
        let transition = store.install(&b, auth(40), nonce(41)).unwrap();
        assert_eq!(transition.snapshot.generation, 1);
        let expected = store.expectation().unwrap();
        let cert = store.verify_current(&expected).unwrap();
        assert_eq!(cert.bundle_id, b.bundle_id().unwrap());
        assert_eq!(store.active_bundle_id().unwrap(), b.bundle_id().unwrap());
    }

    #[test]
    fn second_install_is_rejected() {
        let mut store = AndroidTouchPolicySessionStore::new();
        let b = bundle(20);
        store.install(&b, auth(40), nonce(41)).unwrap();
        assert_eq!(
            store.install(&b, auth(42), nonce(43)),
            Err(AndroidTouchPolicyStoreError::AlreadyInstalled)
        );
    }

    #[test]
    fn failed_stale_replace_does_not_mutate_store() {
        let mut store = AndroidTouchPolicySessionStore::new();
        let b1 = bundle(20);
        let b2 = bundle(30);
        store.install(&b1, auth(40), nonce(41)).unwrap();
        let before = store.snapshot().unwrap();
        assert_eq!(
            store.replace(AndroidTouchPolicySessionStateId(d(99)), &b2, auth(42)),
            Err(AndroidTouchPolicyStoreError::Session(
                AndroidTouchPolicySessionError::StateMismatch,
            ))
        );
        assert_eq!(store.snapshot().unwrap(), before);
        assert_eq!(store.active_bundle_id().unwrap(), b1.bundle_id().unwrap());
    }

    #[test]
    fn exact_replace_commits_new_bundle_after_validation() {
        let mut store = AndroidTouchPolicySessionStore::new();
        let b1 = bundle(20);
        let b2 = bundle(30);
        store.install(&b1, auth(40), nonce(41)).unwrap();
        let before = store.snapshot().unwrap();
        let transition = store.replace(before.state_id, &b2, auth(42)).unwrap();
        assert_eq!(transition.snapshot.generation, 2);
        assert_eq!(store.active_bundle_id().unwrap(), b2.bundle_id().unwrap());
        assert_ne!(transition.snapshot.state_id, before.state_id);
    }

    #[test]
    fn revoke_preserves_bundle_evidence_but_blocks_currentness() {
        let mut store = AndroidTouchPolicySessionStore::new();
        let b = bundle(20);
        store.install(&b, auth(40), nonce(41)).unwrap();
        let before = store.snapshot().unwrap();
        let transition = store.revoke(before.state_id, auth(42)).unwrap();
        assert_eq!(transition.snapshot.status, AndroidTouchPolicySessionStatus::Revoked);
        let expected = store.expectation().unwrap();
        assert_eq!(
            store.verify_current(&expected),
            Err(AndroidTouchPolicyStoreError::Session(
                AndroidTouchPolicySessionError::Revoked,
            ))
        );
        assert_eq!(
            store.active_bundle_id(),
            Err(AndroidTouchPolicyStoreError::Session(
                AndroidTouchPolicySessionError::Revoked,
            ))
        );
    }

    #[test]
    fn internal_bundle_state_divergence_fails_closed() {
        let mut store = AndroidTouchPolicySessionStore::new();
        let b1 = bundle(20);
        let b2 = bundle(30);
        store.install(&b1, auth(40), nonce(41)).unwrap();
        store.bundle = Some(b2);
        assert_eq!(
            store.validate(),
            Err(AndroidTouchPolicyStoreError::StoreInvariantViolation)
        );
    }
}
