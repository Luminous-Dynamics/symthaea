// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-ANDROIDPOLICYSESSION-530: monotonic Rust-owned currentness for one exact
//! Android touch-policy bundle.
//!
//! QUAL-ANDROIDPOLICYBUNDLE-529 closes the semantic policy required to judge an
//! assurance-bearing Android touch. A valid bundle is still only policy evidence:
//! event-time callers must not be able to install, replace, revoke, resurrect, or
//! otherwise select the current policy under which their own observations are
//! judged.
//!
//! This tranche defines a small forward-only session state machine. Installation,
//! replacement, and revocation consume opaque nonzero authorization identities
//! issued by an upstream authority plane; this module binds those identities into
//! immutable state transitions but does not authenticate or mint them. Actual
//! cross-process compare-and-swap persistence remains the responsibility of the
//! protected transaction/store layer.

use core::fmt;

use symthaea_core::assurance_trusted_surface::TrustedSurfaceId;

use crate::assurance_android_policy_bundle::{
    AndroidTouchPolicyBundle, AndroidTouchPolicyBundleError, AndroidTouchPolicyBundleId,
};

const SESSION_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-touch-policy-session\0";
const STATE_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-touch-policy-session-state\0";
const TRANSITION_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-touch-policy-session-transition\0";
const CURRENT_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-touch-policy-current\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(pub [u8; 32]);

        impl $name {
            pub const ZERO: Self = Self([0; 32]);
            pub const fn as_bytes(&self) -> &[u8; 32] { &self.0 }
            pub fn is_zero(&self) -> bool { self.0 == [0; 32] }
        }
    };
}

digest_id!(AndroidTouchPolicySessionAuthorizationId);
digest_id!(AndroidTouchPolicySessionNonce);
digest_id!(AndroidTouchPolicySessionId);
digest_id!(AndroidTouchPolicySessionStateId);
digest_id!(AndroidTouchPolicySessionTransitionReceiptId);
digest_id!(AndroidTouchPolicyCurrentCertificateId);

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AndroidTouchPolicySessionStatus {
    Active = 1,
    Revoked = 2,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AndroidTouchPolicySessionTransitionKind {
    Install = 1,
    Replace = 2,
    Revoke = 3,
}

/// Canonical current state for one Android policy session. The complete policy
/// remains separately validated 529 evidence; this state stores only its identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTouchPolicySessionState {
    pub session_id: AndroidTouchPolicySessionId,
    pub generation: u64,
    pub bundle_id: AndroidTouchPolicyBundleId,
    pub status: AndroidTouchPolicySessionStatus,
    pub predecessor_state_id: AndroidTouchPolicySessionStateId,
    pub last_transition: AndroidTouchPolicySessionTransitionKind,
    pub transition_authorization_id: AndroidTouchPolicySessionAuthorizationId,
}

/// Immutable transition evidence. A later protected store may use prior/next
/// state identities as compare-and-swap operands; this object is not itself a
/// distributed atomic write.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTouchPolicySessionTransitionReceipt {
    pub kind: AndroidTouchPolicySessionTransitionKind,
    pub authorization_id: AndroidTouchPolicySessionAuthorizationId,
    pub session_id: AndroidTouchPolicySessionId,
    pub prior_state_id: AndroidTouchPolicySessionStateId,
    pub next_state_id: AndroidTouchPolicySessionStateId,
    pub prior_generation: u64,
    pub next_generation: u64,
    pub prior_bundle_id: AndroidTouchPolicyBundleId,
    pub next_bundle_id: AndroidTouchPolicyBundleId,
}

/// Exact event-time expectation. Naming an expectation does not make it current.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTouchPolicySessionExpectation {
    pub session_id: AndroidTouchPolicySessionId,
    pub generation: u64,
    pub state_id: AndroidTouchPolicySessionStateId,
    pub bundle_id: AndroidTouchPolicyBundleId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTouchPolicyCurrentCertificate {
    pub session_id: AndroidTouchPolicySessionId,
    pub generation: u64,
    pub state_id: AndroidTouchPolicySessionStateId,
    pub bundle_id: AndroidTouchPolicyBundleId,
    pub trusted_surface_id: TrustedSurfaceId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AndroidTouchPolicySessionError {
    ZeroAuthorization,
    ZeroSessionNonce,
    ZeroSessionId,
    ZeroGeneration,
    ZeroBundle,
    ZeroState,
    InvalidInitialState,
    InvalidSuccessorState,
    Revoked,
    SessionMismatch,
    GenerationMismatch,
    StateMismatch,
    BundleMismatch,
    SameBundleReplacement,
    AuthorizationReplay,
    GenerationOverflow,
    InvalidTransitionReceipt,
    Bundle(AndroidTouchPolicyBundleError),
}

impl fmt::Display for AndroidTouchPolicySessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{self:?}") }
}

impl std::error::Error for AndroidTouchPolicySessionError {}

impl From<AndroidTouchPolicyBundleError> for AndroidTouchPolicySessionError {
    fn from(value: AndroidTouchPolicyBundleError) -> Self { Self::Bundle(value) }
}

impl AndroidTouchPolicySessionState {
    pub fn validate(&self) -> Result<(), AndroidTouchPolicySessionError> {
        if self.session_id.is_zero() {
            return Err(AndroidTouchPolicySessionError::ZeroSessionId);
        }
        if self.generation == 0 {
            return Err(AndroidTouchPolicySessionError::ZeroGeneration);
        }
        if self.bundle_id.is_zero() {
            return Err(AndroidTouchPolicySessionError::ZeroBundle);
        }
        if self.transition_authorization_id.is_zero() {
            return Err(AndroidTouchPolicySessionError::ZeroAuthorization);
        }

        if self.generation == 1 {
            if self.last_transition != AndroidTouchPolicySessionTransitionKind::Install
                || !self.predecessor_state_id.is_zero()
                || self.status != AndroidTouchPolicySessionStatus::Active
            {
                return Err(AndroidTouchPolicySessionError::InvalidInitialState);
            }
        } else if self.last_transition == AndroidTouchPolicySessionTransitionKind::Install
            || self.predecessor_state_id.is_zero()
        {
            return Err(AndroidTouchPolicySessionError::InvalidSuccessorState);
        }

        match (self.status, self.last_transition) {
            (
                AndroidTouchPolicySessionStatus::Revoked,
                AndroidTouchPolicySessionTransitionKind::Revoke,
            ) => {}
            (AndroidTouchPolicySessionStatus::Revoked, _) => {
                return Err(AndroidTouchPolicySessionError::InvalidSuccessorState)
            }
            (
                AndroidTouchPolicySessionStatus::Active,
                AndroidTouchPolicySessionTransitionKind::Revoke,
            ) => return Err(AndroidTouchPolicySessionError::InvalidSuccessorState),
            (AndroidTouchPolicySessionStatus::Active, _) => {}
        }
        Ok(())
    }

    pub fn state_id(
        &self,
    ) -> Result<AndroidTouchPolicySessionStateId, AndroidTouchPolicySessionError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(STATE_DOMAIN);
        hasher.update(self.session_id.as_bytes());
        hasher.update(&self.generation.to_le_bytes());
        hasher.update(self.bundle_id.as_bytes());
        hasher.update(&[self.status as u8]);
        hasher.update(self.predecessor_state_id.as_bytes());
        hasher.update(&[self.last_transition as u8]);
        hasher.update(self.transition_authorization_id.as_bytes());
        Ok(AndroidTouchPolicySessionStateId(*hasher.finalize().as_bytes()))
    }

    pub fn install(
        bundle: &AndroidTouchPolicyBundle,
        authorization_id: AndroidTouchPolicySessionAuthorizationId,
        nonce: AndroidTouchPolicySessionNonce,
    ) -> Result<(Self, AndroidTouchPolicySessionTransitionReceipt), AndroidTouchPolicySessionError> {
        if authorization_id.is_zero() {
            return Err(AndroidTouchPolicySessionError::ZeroAuthorization);
        }
        if nonce.is_zero() {
            return Err(AndroidTouchPolicySessionError::ZeroSessionNonce);
        }
        let bundle_id = bundle.bundle_id()?;

        let mut hasher = blake3::Hasher::new();
        hasher.update(SESSION_DOMAIN);
        hasher.update(authorization_id.as_bytes());
        hasher.update(nonce.as_bytes());
        hasher.update(bundle_id.as_bytes());
        let session_id = AndroidTouchPolicySessionId(*hasher.finalize().as_bytes());

        let state = Self {
            session_id,
            generation: 1,
            bundle_id,
            status: AndroidTouchPolicySessionStatus::Active,
            predecessor_state_id: AndroidTouchPolicySessionStateId::ZERO,
            last_transition: AndroidTouchPolicySessionTransitionKind::Install,
            transition_authorization_id: authorization_id,
        };
        let next_state_id = state.state_id()?;
        let receipt = AndroidTouchPolicySessionTransitionReceipt {
            kind: AndroidTouchPolicySessionTransitionKind::Install,
            authorization_id,
            session_id,
            prior_state_id: AndroidTouchPolicySessionStateId::ZERO,
            next_state_id,
            prior_generation: 0,
            next_generation: 1,
            prior_bundle_id: AndroidTouchPolicyBundleId::ZERO,
            next_bundle_id: bundle_id,
        };
        receipt.validate()?;
        Ok((state, receipt))
    }

    pub fn replace(
        &self,
        expected_state_id: AndroidTouchPolicySessionStateId,
        bundle: &AndroidTouchPolicyBundle,
        authorization_id: AndroidTouchPolicySessionAuthorizationId,
    ) -> Result<(Self, AndroidTouchPolicySessionTransitionReceipt), AndroidTouchPolicySessionError> {
        self.validate()?;
        if self.status == AndroidTouchPolicySessionStatus::Revoked {
            return Err(AndroidTouchPolicySessionError::Revoked);
        }
        if authorization_id.is_zero() {
            return Err(AndroidTouchPolicySessionError::ZeroAuthorization);
        }
        if authorization_id == self.transition_authorization_id {
            return Err(AndroidTouchPolicySessionError::AuthorizationReplay);
        }
        let prior_state_id = self.state_id()?;
        if expected_state_id != prior_state_id {
            return Err(AndroidTouchPolicySessionError::StateMismatch);
        }
        let next_bundle_id = bundle.bundle_id()?;
        if next_bundle_id == self.bundle_id {
            return Err(AndroidTouchPolicySessionError::SameBundleReplacement);
        }
        let next_generation = self
            .generation
            .checked_add(1)
            .ok_or(AndroidTouchPolicySessionError::GenerationOverflow)?;

        let next = Self {
            session_id: self.session_id,
            generation: next_generation,
            bundle_id: next_bundle_id,
            status: AndroidTouchPolicySessionStatus::Active,
            predecessor_state_id: prior_state_id,
            last_transition: AndroidTouchPolicySessionTransitionKind::Replace,
            transition_authorization_id: authorization_id,
        };
        let next_state_id = next.state_id()?;
        let receipt = AndroidTouchPolicySessionTransitionReceipt {
            kind: AndroidTouchPolicySessionTransitionKind::Replace,
            authorization_id,
            session_id: self.session_id,
            prior_state_id,
            next_state_id,
            prior_generation: self.generation,
            next_generation,
            prior_bundle_id: self.bundle_id,
            next_bundle_id,
        };
        receipt.validate()?;
        Ok((next, receipt))
    }

    pub fn revoke(
        &self,
        expected_state_id: AndroidTouchPolicySessionStateId,
        authorization_id: AndroidTouchPolicySessionAuthorizationId,
    ) -> Result<(Self, AndroidTouchPolicySessionTransitionReceipt), AndroidTouchPolicySessionError> {
        self.validate()?;
        if self.status == AndroidTouchPolicySessionStatus::Revoked {
            return Err(AndroidTouchPolicySessionError::Revoked);
        }
        if authorization_id.is_zero() {
            return Err(AndroidTouchPolicySessionError::ZeroAuthorization);
        }
        if authorization_id == self.transition_authorization_id {
            return Err(AndroidTouchPolicySessionError::AuthorizationReplay);
        }
        let prior_state_id = self.state_id()?;
        if expected_state_id != prior_state_id {
            return Err(AndroidTouchPolicySessionError::StateMismatch);
        }
        let next_generation = self
            .generation
            .checked_add(1)
            .ok_or(AndroidTouchPolicySessionError::GenerationOverflow)?;
        let next = Self {
            session_id: self.session_id,
            generation: next_generation,
            bundle_id: self.bundle_id,
            status: AndroidTouchPolicySessionStatus::Revoked,
            predecessor_state_id: prior_state_id,
            last_transition: AndroidTouchPolicySessionTransitionKind::Revoke,
            transition_authorization_id: authorization_id,
        };
        let next_state_id = next.state_id()?;
        let receipt = AndroidTouchPolicySessionTransitionReceipt {
            kind: AndroidTouchPolicySessionTransitionKind::Revoke,
            authorization_id,
            session_id: self.session_id,
            prior_state_id,
            next_state_id,
            prior_generation: self.generation,
            next_generation,
            prior_bundle_id: self.bundle_id,
            next_bundle_id: self.bundle_id,
        };
        receipt.validate()?;
        Ok((next, receipt))
    }

    pub fn verify_current(
        &self,
        expected: &AndroidTouchPolicySessionExpectation,
        bundle: &AndroidTouchPolicyBundle,
    ) -> Result<AndroidTouchPolicyCurrentCertificate, AndroidTouchPolicySessionError> {
        self.validate()?;
        if self.status == AndroidTouchPolicySessionStatus::Revoked {
            return Err(AndroidTouchPolicySessionError::Revoked);
        }
        if expected.session_id != self.session_id {
            return Err(AndroidTouchPolicySessionError::SessionMismatch);
        }
        if expected.generation != self.generation {
            return Err(AndroidTouchPolicySessionError::GenerationMismatch);
        }
        let state_id = self.state_id()?;
        if expected.state_id != state_id {
            return Err(AndroidTouchPolicySessionError::StateMismatch);
        }
        if expected.bundle_id != self.bundle_id {
            return Err(AndroidTouchPolicySessionError::BundleMismatch);
        }
        let bundle_identity = bundle.validate()?;
        if bundle_identity.bundle_id != self.bundle_id {
            return Err(AndroidTouchPolicySessionError::BundleMismatch);
        }
        Ok(AndroidTouchPolicyCurrentCertificate {
            session_id: self.session_id,
            generation: self.generation,
            state_id,
            bundle_id: self.bundle_id,
            trusted_surface_id: bundle_identity.trusted_surface_id,
        })
    }
}

impl AndroidTouchPolicySessionTransitionReceipt {
    pub fn validate(&self) -> Result<(), AndroidTouchPolicySessionError> {
        if self.authorization_id.is_zero() {
            return Err(AndroidTouchPolicySessionError::ZeroAuthorization);
        }
        if self.session_id.is_zero() {
            return Err(AndroidTouchPolicySessionError::ZeroSessionId);
        }
        if self.next_state_id.is_zero() || self.next_bundle_id.is_zero() {
            return Err(AndroidTouchPolicySessionError::InvalidTransitionReceipt);
        }

        match self.kind {
            AndroidTouchPolicySessionTransitionKind::Install => {
                if !self.prior_state_id.is_zero()
                    || self.prior_generation != 0
                    || !self.prior_bundle_id.is_zero()
                    || self.next_generation != 1
                {
                    return Err(AndroidTouchPolicySessionError::InvalidTransitionReceipt);
                }
            }
            AndroidTouchPolicySessionTransitionKind::Replace => {
                if self.prior_state_id.is_zero()
                    || self.prior_generation == 0
                    || self.prior_bundle_id.is_zero()
                    || self.prior_bundle_id == self.next_bundle_id
                {
                    return Err(AndroidTouchPolicySessionError::InvalidTransitionReceipt);
                }
                let expected = self
                    .prior_generation
                    .checked_add(1)
                    .ok_or(AndroidTouchPolicySessionError::GenerationOverflow)?;
                if self.next_generation != expected {
                    return Err(AndroidTouchPolicySessionError::InvalidTransitionReceipt);
                }
            }
            AndroidTouchPolicySessionTransitionKind::Revoke => {
                if self.prior_state_id.is_zero()
                    || self.prior_generation == 0
                    || self.prior_bundle_id.is_zero()
                    || self.prior_bundle_id != self.next_bundle_id
                {
                    return Err(AndroidTouchPolicySessionError::InvalidTransitionReceipt);
                }
                let expected = self
                    .prior_generation
                    .checked_add(1)
                    .ok_or(AndroidTouchPolicySessionError::GenerationOverflow)?;
                if self.next_generation != expected {
                    return Err(AndroidTouchPolicySessionError::InvalidTransitionReceipt);
                }
            }
        }
        Ok(())
    }

    pub fn receipt_id(
        &self,
    ) -> Result<AndroidTouchPolicySessionTransitionReceiptId, AndroidTouchPolicySessionError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(TRANSITION_DOMAIN);
        hasher.update(&[self.kind as u8]);
        hasher.update(self.authorization_id.as_bytes());
        hasher.update(self.session_id.as_bytes());
        hasher.update(self.prior_state_id.as_bytes());
        hasher.update(self.next_state_id.as_bytes());
        hasher.update(&self.prior_generation.to_le_bytes());
        hasher.update(&self.next_generation.to_le_bytes());
        hasher.update(self.prior_bundle_id.as_bytes());
        hasher.update(self.next_bundle_id.as_bytes());
        Ok(AndroidTouchPolicySessionTransitionReceiptId(*hasher.finalize().as_bytes()))
    }
}

impl AndroidTouchPolicyCurrentCertificate {
    pub fn certificate_id(
        &self,
    ) -> Result<AndroidTouchPolicyCurrentCertificateId, AndroidTouchPolicySessionError> {
        if self.session_id.is_zero() {
            return Err(AndroidTouchPolicySessionError::ZeroSessionId);
        }
        if self.generation == 0 {
            return Err(AndroidTouchPolicySessionError::ZeroGeneration);
        }
        if self.state_id.is_zero() {
            return Err(AndroidTouchPolicySessionError::ZeroState);
        }
        if self.bundle_id.is_zero() {
            return Err(AndroidTouchPolicySessionError::ZeroBundle);
        }
        if self.trusted_surface_id.is_zero() {
            return Err(AndroidTouchPolicySessionError::BundleMismatch);
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(CURRENT_DOMAIN);
        hasher.update(self.session_id.as_bytes());
        hasher.update(&self.generation.to_le_bytes());
        hasher.update(self.state_id.as_bytes());
        hasher.update(self.bundle_id.as_bytes());
        hasher.update(self.trusted_surface_id.as_bytes());
        Ok(AndroidTouchPolicyCurrentCertificateId(*hasher.finalize().as_bytes()))
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

    fn d(value: u8) -> [u8; 32] { [value; 32] }

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
        let ingress_profile = PlatformIngressProfile {
            platform: PlatformIngressKind::AndroidJni,
            abi_version: 1,
            capture_profile_id: ScreenCaptureProfileId(d(profile_seed)),
            touch_input_profile_id: TouchInputProfileId(d(profile_seed.wrapping_add(1))),
            max_frame_bytes: 1024,
        };
        let ingress_policy = IngressAdmissionPolicy {
            policy_context_id: IngressPolicyContextId(d(22)),
            policy_generation: 7,
            policy_state_root: IngressPolicyStateRoot(d(23)),
            platform: PlatformIngressKind::AndroidJni,
            ingress_profile_id: ingress_profile.profile_id().unwrap(),
            trusted_surface_id: attention_requirement.trusted_surface_id,
            input_attester_id: InputAttesterId(d(8)),
            allow_frame: false,
            allow_touch: true,
        };
        AndroidTouchPolicyBundle {
            ingress_policy,
            ingress_profile,
            attention_policy_admission: attention_admission,
            attention_requirement,
            motion_policy_admission: motion_admission,
            motion_requirement,
        }
    }

    fn auth(value: u8) -> AndroidTouchPolicySessionAuthorizationId {
        AndroidTouchPolicySessionAuthorizationId(d(value))
    }
    fn nonce(value: u8) -> AndroidTouchPolicySessionNonce {
        AndroidTouchPolicySessionNonce(d(value))
    }

    #[test]
    fn exact_bundle_installs_active_generation_one() {
        let b = bundle(20);
        let (state, receipt) =
            AndroidTouchPolicySessionState::install(&b, auth(40), nonce(41)).unwrap();
        assert_eq!(state.generation, 1);
        assert_eq!(state.status, AndroidTouchPolicySessionStatus::Active);
        assert_eq!(receipt.kind, AndroidTouchPolicySessionTransitionKind::Install);
        assert!(!receipt.receipt_id().unwrap().is_zero());
        let expected = AndroidTouchPolicySessionExpectation {
            session_id: state.session_id,
            generation: state.generation,
            state_id: state.state_id().unwrap(),
            bundle_id: state.bundle_id,
        };
        assert!(!state
            .verify_current(&expected, &b)
            .unwrap()
            .certificate_id()
            .unwrap()
            .is_zero());
    }

    #[test]
    fn stale_expected_state_rejects_replacement() {
        let b1 = bundle(20);
        let b2 = bundle(30);
        let (state, _) =
            AndroidTouchPolicySessionState::install(&b1, auth(40), nonce(41)).unwrap();
        assert_eq!(
            state.replace(AndroidTouchPolicySessionStateId(d(99)), &b2, auth(42)),
            Err(AndroidTouchPolicySessionError::StateMismatch)
        );
    }

    #[test]
    fn replacement_advances_generation_and_stales_old_expectation() {
        let b1 = bundle(20);
        let b2 = bundle(30);
        let (state, _) =
            AndroidTouchPolicySessionState::install(&b1, auth(40), nonce(41)).unwrap();
        let old = AndroidTouchPolicySessionExpectation {
            session_id: state.session_id,
            generation: state.generation,
            state_id: state.state_id().unwrap(),
            bundle_id: state.bundle_id,
        };
        let (next, receipt) = state.replace(old.state_id, &b2, auth(42)).unwrap();
        assert_eq!(next.generation, 2);
        assert_eq!(receipt.prior_state_id, old.state_id);
        assert_ne!(next.bundle_id, state.bundle_id);
        assert_eq!(
            next.verify_current(&old, &b2),
            Err(AndroidTouchPolicySessionError::GenerationMismatch)
        );
    }

    #[test]
    fn same_bundle_replacement_is_rejected() {
        let b = bundle(20);
        let (state, _) =
            AndroidTouchPolicySessionState::install(&b, auth(40), nonce(41)).unwrap();
        assert_eq!(
            state.replace(state.state_id().unwrap(), &b, auth(42)),
            Err(AndroidTouchPolicySessionError::SameBundleReplacement)
        );
    }

    #[test]
    fn authorization_replay_is_rejected() {
        let b1 = bundle(20);
        let b2 = bundle(30);
        let (state, _) =
            AndroidTouchPolicySessionState::install(&b1, auth(40), nonce(41)).unwrap();
        assert_eq!(
            state.replace(state.state_id().unwrap(), &b2, auth(40)),
            Err(AndroidTouchPolicySessionError::AuthorizationReplay)
        );
    }

    #[test]
    fn revocation_is_forward_only() {
        let b1 = bundle(20);
        let b2 = bundle(30);
        let (state, _) =
            AndroidTouchPolicySessionState::install(&b1, auth(40), nonce(41)).unwrap();
        let (revoked, receipt) = state.revoke(state.state_id().unwrap(), auth(42)).unwrap();
        assert_eq!(revoked.status, AndroidTouchPolicySessionStatus::Revoked);
        assert_eq!(receipt.kind, AndroidTouchPolicySessionTransitionKind::Revoke);
        let expected = AndroidTouchPolicySessionExpectation {
            session_id: revoked.session_id,
            generation: revoked.generation,
            state_id: revoked.state_id().unwrap(),
            bundle_id: revoked.bundle_id,
        };
        assert_eq!(
            revoked.verify_current(&expected, &b1),
            Err(AndroidTouchPolicySessionError::Revoked)
        );
        assert_eq!(
            revoked.replace(revoked.state_id().unwrap(), &b2, auth(43)),
            Err(AndroidTouchPolicySessionError::Revoked)
        );
    }

    #[test]
    fn generation_overflow_rejects_without_panicking() {
        let b1 = bundle(20);
        let b2 = bundle(30);
        let state = AndroidTouchPolicySessionState {
            session_id: AndroidTouchPolicySessionId(d(50)),
            generation: u64::MAX,
            bundle_id: b1.bundle_id().unwrap(),
            status: AndroidTouchPolicySessionStatus::Active,
            predecessor_state_id: AndroidTouchPolicySessionStateId(d(51)),
            last_transition: AndroidTouchPolicySessionTransitionKind::Replace,
            transition_authorization_id: auth(52),
        };
        assert_eq!(
            state.replace(state.state_id().unwrap(), &b2, auth(53)),
            Err(AndroidTouchPolicySessionError::GenerationOverflow)
        );
    }

    #[test]
    fn substituted_bundle_cannot_satisfy_currentness() {
        let b1 = bundle(20);
        let b2 = bundle(30);
        let (state, _) =
            AndroidTouchPolicySessionState::install(&b1, auth(40), nonce(41)).unwrap();
        let expected = AndroidTouchPolicySessionExpectation {
            session_id: state.session_id,
            generation: state.generation,
            state_id: state.state_id().unwrap(),
            bundle_id: state.bundle_id,
        };
        assert_eq!(
            state.verify_current(&expected, &b2),
            Err(AndroidTouchPolicySessionError::BundleMismatch)
        );
    }

    #[test]
    fn zero_install_prerequisites_are_rejected() {
        let b = bundle(20);
        assert_eq!(
            AndroidTouchPolicySessionState::install(
                &b,
                AndroidTouchPolicySessionAuthorizationId::ZERO,
                nonce(41),
            ),
            Err(AndroidTouchPolicySessionError::ZeroAuthorization)
        );
        assert_eq!(
            AndroidTouchPolicySessionState::install(
                &b,
                auth(40),
                AndroidTouchPolicySessionNonce::ZERO,
            ),
            Err(AndroidTouchPolicySessionError::ZeroSessionNonce)
        );
    }

    #[test]
    fn malformed_initial_state_and_successor_receipt_are_rejected() {
        let b = bundle(20);
        let state = AndroidTouchPolicySessionState {
            session_id: AndroidTouchPolicySessionId(d(50)),
            generation: 1,
            bundle_id: b.bundle_id().unwrap(),
            status: AndroidTouchPolicySessionStatus::Active,
            predecessor_state_id: AndroidTouchPolicySessionStateId(d(51)),
            last_transition: AndroidTouchPolicySessionTransitionKind::Install,
            transition_authorization_id: auth(52),
        };
        assert_eq!(
            state.validate(),
            Err(AndroidTouchPolicySessionError::InvalidInitialState)
        );

        let impossible = AndroidTouchPolicySessionTransitionReceipt {
            kind: AndroidTouchPolicySessionTransitionKind::Replace,
            authorization_id: auth(60),
            session_id: AndroidTouchPolicySessionId(d(61)),
            prior_state_id: AndroidTouchPolicySessionStateId(d(62)),
            next_state_id: AndroidTouchPolicySessionStateId(d(63)),
            prior_generation: 0,
            next_generation: 1,
            prior_bundle_id: AndroidTouchPolicyBundleId(d(64)),
            next_bundle_id: AndroidTouchPolicyBundleId(d(65)),
        };
        assert_eq!(
            impossible.validate(),
            Err(AndroidTouchPolicySessionError::InvalidTransitionReceipt)
        );
    }
}
