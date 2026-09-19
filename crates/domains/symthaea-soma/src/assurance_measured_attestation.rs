// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-MEASUREDATTEST-519: cryptographically authenticate measured Soma
//! framebuffer/touch observations.
//!
//! QUAL-SOMAINTERACTION-517 defines exact observation identities and
//! QUAL-SOMAINTERACTIONVERIFY-518 recomputes them from raw adapter inputs. This
//! module adds cryptographic attribution for those measured statements while
//! reusing QUAL-ATTESTERAUTH-515's crypto-agile verification-key and verifier
//! primitives. No signature algorithm is hard-coded here; QUAL-ED25519ADAPTER-516
//! is one concrete implementation of the verifier contract.
//!
//! A valid signature proves only that the currently bound measurement-attester
//! key signed the exact observation identity. It does not prove capture-path
//! completeness, overlay absence, hardware display fidelity, or human origin of
//! a touch event. Those remain independent platform-attestation propositions.

use core::fmt;

use symthaea_core::assurance_attester_auth::{
    AttestationVerifier, KeyStateRoot, SignatureCommitment, SignatureSuiteId,
    VerificationKey, VerificationKeyId,
};

use crate::assurance_soma_interaction::{
    ScreenFrameObservation, ScreenFrameObservationId, TouchEventObservation,
    TouchEventObservationId,
};
use crate::assurance_soma_interaction_verify::{
    MeasuredInteractionVerificationCertificate,
    MeasuredInteractionVerificationCertificateId,
};

const BINDING_DOMAIN: &[u8] = b"symthaea.soma.presentation.v1/measurement-key-binding\0";
const MESSAGE_DOMAIN: &[u8] = b"symthaea.soma.presentation.v1/measurement-attestation-message\0";
const SIGNATURE_DOMAIN: &[u8] = b"symthaea.soma.presentation.v1/measurement-signature\0";
const CERTIFICATE_DOMAIN: &[u8] = b"symthaea.soma.presentation.v1/measurement-auth-certificate\0";
const BUNDLE_DOMAIN: &[u8] = b"symthaea.soma.presentation.v1/authenticated-measured-interaction\0";

const MAX_SIGNATURE_BYTES: usize = 64 * 1024;

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

digest_id!(MeasurementAttesterId);
digest_id!(MeasurementKeyBindingId);
digest_id!(MeasurementMessageId);
digest_id!(AuthenticatedMeasurementCertificateId);
digest_id!(AuthenticatedMeasuredInteractionId);

/// Qualified function of one measurement-attester key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum MeasurementRole {
    FramebufferCapture = 1,
    TouchInputCapture = 2,
}

/// Exact measured object whose identity is authenticated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeasurementSubject {
    Frame(ScreenFrameObservationId),
    Touch(TouchEventObservationId),
}

/// Policy/state binding from one measured-adapter identity to one exact key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeasurementKeyBinding {
    pub attester_id: MeasurementAttesterId,
    pub role: MeasurementRole,
    pub signature_suite_id: SignatureSuiteId,
    pub verification_key_id: VerificationKeyId,
    pub key_epoch: u64,
    pub key_state_root: KeyStateRoot,
}

/// Signature statement over one exact measurement observation identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SignedMeasurement {
    pub binding_id: MeasurementKeyBindingId,
    pub attester_id: MeasurementAttesterId,
    pub role: MeasurementRole,
    pub signature_suite_id: SignatureSuiteId,
    pub verification_key_id: VerificationKeyId,
    pub key_epoch: u64,
    pub key_state_root: KeyStateRoot,
    pub subject: MeasurementSubject,
    pub message_id: MeasurementMessageId,
    pub signature: Vec<u8>,
}

/// Cryptographically authenticated statement about one exact 517 observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuthenticatedMeasurementCertificate {
    pub binding_id: MeasurementKeyBindingId,
    pub attester_id: MeasurementAttesterId,
    pub role: MeasurementRole,
    pub subject: MeasurementSubject,
    pub message_id: MeasurementMessageId,
    pub verification_key_id: VerificationKeyId,
    pub key_epoch: u64,
    pub key_state_root: KeyStateRoot,
    pub signature_commitment: SignatureCommitment,
}

/// Links the 518 raw-input verification theorem to independently authenticated
/// frame and touch measurement statements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuthenticatedMeasuredInteraction {
    pub verification_certificate_id: MeasuredInteractionVerificationCertificateId,
    pub frame_auth_certificate_id: AuthenticatedMeasurementCertificateId,
    pub touch_auth_certificate_id: AuthenticatedMeasurementCertificateId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeasuredAttestationError {
    ZeroAttester,
    ZeroSignatureSuite,
    ZeroVerificationKey,
    ZeroKeyEpoch,
    ZeroKeyStateRoot,
    ZeroBinding,
    ZeroSubject,
    EmptySignature,
    SignatureTooLarge,
    RoleSubjectMismatch,
    BindingMismatch,
    AttesterMismatch,
    RoleMismatch,
    SignatureSuiteMismatch,
    VerificationKeyMismatch,
    KeyEpochMismatch,
    KeyStateChanged,
    SubjectMismatch,
    MessageMismatch,
    SignatureInvalid,
    VerificationFrameMismatch,
    VerificationTouchMismatch,
    ZeroAuthenticatedBundleComponent,
}

impl fmt::Display for MeasuredAttestationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for MeasuredAttestationError {}

impl MeasurementSubject {
    fn validate_for_role(self, role: MeasurementRole) -> Result<(), MeasuredAttestationError> {
        let valid = matches!(
            (self, role),
            (Self::Frame(_), MeasurementRole::FramebufferCapture)
                | (Self::Touch(_), MeasurementRole::TouchInputCapture)
        );
        if !valid {
            return Err(MeasuredAttestationError::RoleSubjectMismatch);
        }
        let is_zero = match self {
            Self::Frame(id) => id.is_zero(),
            Self::Touch(id) => id.is_zero(),
        };
        if is_zero {
            return Err(MeasuredAttestationError::ZeroSubject);
        }
        Ok(())
    }

    fn update_hasher(self, hasher: &mut blake3::Hasher) {
        match self {
            Self::Frame(id) => {
                hasher.update(&[1]);
                hasher.update(id.as_bytes());
            }
            Self::Touch(id) => {
                hasher.update(&[2]);
                hasher.update(id.as_bytes());
            }
        }
    }
}

impl MeasurementKeyBinding {
    pub fn validate(&self) -> Result<(), MeasuredAttestationError> {
        if self.attester_id.is_zero() {
            return Err(MeasuredAttestationError::ZeroAttester);
        }
        if self.signature_suite_id.is_zero() {
            return Err(MeasuredAttestationError::ZeroSignatureSuite);
        }
        if self.verification_key_id.is_zero() {
            return Err(MeasuredAttestationError::ZeroVerificationKey);
        }
        if self.key_epoch == 0 {
            return Err(MeasuredAttestationError::ZeroKeyEpoch);
        }
        if self.key_state_root.is_zero() {
            return Err(MeasuredAttestationError::ZeroKeyStateRoot);
        }
        Ok(())
    }

    pub fn binding_id(&self) -> Result<MeasurementKeyBindingId, MeasuredAttestationError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(BINDING_DOMAIN);
        hasher.update(self.attester_id.as_bytes());
        hasher.update(&[self.role as u8]);
        hasher.update(self.signature_suite_id.as_bytes());
        hasher.update(self.verification_key_id.as_bytes());
        hasher.update(&self.key_epoch.to_le_bytes());
        hasher.update(self.key_state_root.as_bytes());
        Ok(MeasurementKeyBindingId(*hasher.finalize().as_bytes()))
    }
}

pub fn frame_message_id(
    binding: &MeasurementKeyBinding,
    frame: &ScreenFrameObservation,
) -> Result<MeasurementMessageId, MeasuredAttestationError> {
    measurement_message_id(
        binding,
        MeasurementSubject::Frame(
            frame
                .observation_id()
                .map_err(|_| MeasuredAttestationError::SubjectMismatch)?,
        ),
    )
}

pub fn touch_message_id(
    binding: &MeasurementKeyBinding,
    touch: &TouchEventObservation,
) -> Result<MeasurementMessageId, MeasuredAttestationError> {
    measurement_message_id(
        binding,
        MeasurementSubject::Touch(
            touch
                .observation_id()
                .map_err(|_| MeasuredAttestationError::SubjectMismatch)?,
        ),
    )
}

fn measurement_message_id(
    binding: &MeasurementKeyBinding,
    subject: MeasurementSubject,
) -> Result<MeasurementMessageId, MeasuredAttestationError> {
    binding.validate()?;
    subject.validate_for_role(binding.role)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(MESSAGE_DOMAIN);
    hasher.update(binding.binding_id()?.as_bytes());
    subject.update_hasher(&mut hasher);
    Ok(MeasurementMessageId(*hasher.finalize().as_bytes()))
}

impl SignedMeasurement {
    pub fn from_binding(
        binding: &MeasurementKeyBinding,
        subject: MeasurementSubject,
        message_id: MeasurementMessageId,
        signature: Vec<u8>,
    ) -> Result<Self, MeasuredAttestationError> {
        binding.validate()?;
        subject.validate_for_role(binding.role)?;
        if message_id.is_zero() {
            return Err(MeasuredAttestationError::MessageMismatch);
        }
        if signature.is_empty() {
            return Err(MeasuredAttestationError::EmptySignature);
        }
        if signature.len() > MAX_SIGNATURE_BYTES {
            return Err(MeasuredAttestationError::SignatureTooLarge);
        }
        Ok(Self {
            binding_id: binding.binding_id()?,
            attester_id: binding.attester_id,
            role: binding.role,
            signature_suite_id: binding.signature_suite_id,
            verification_key_id: binding.verification_key_id,
            key_epoch: binding.key_epoch,
            key_state_root: binding.key_state_root,
            subject,
            message_id,
            signature,
        })
    }

    fn validate(&self) -> Result<(), MeasuredAttestationError> {
        if self.binding_id.is_zero() {
            return Err(MeasuredAttestationError::ZeroBinding);
        }
        if self.attester_id.is_zero() {
            return Err(MeasuredAttestationError::ZeroAttester);
        }
        self.subject.validate_for_role(self.role)?;
        if self.signature_suite_id.is_zero() {
            return Err(MeasuredAttestationError::ZeroSignatureSuite);
        }
        if self.verification_key_id.is_zero() {
            return Err(MeasuredAttestationError::ZeroVerificationKey);
        }
        if self.key_epoch == 0 {
            return Err(MeasuredAttestationError::ZeroKeyEpoch);
        }
        if self.key_state_root.is_zero() {
            return Err(MeasuredAttestationError::ZeroKeyStateRoot);
        }
        if self.message_id.is_zero() {
            return Err(MeasuredAttestationError::MessageMismatch);
        }
        if self.signature.is_empty() {
            return Err(MeasuredAttestationError::EmptySignature);
        }
        if self.signature.len() > MAX_SIGNATURE_BYTES {
            return Err(MeasuredAttestationError::SignatureTooLarge);
        }
        Ok(())
    }

    fn signature_commitment(&self) -> Result<SignatureCommitment, MeasuredAttestationError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(SIGNATURE_DOMAIN);
        hasher.update(&(self.signature.len() as u64).to_le_bytes());
        hasher.update(&self.signature);
        Ok(SignatureCommitment(*hasher.finalize().as_bytes()))
    }
}

impl AuthenticatedMeasurementCertificate {
    pub fn authenticate_frame<V: AttestationVerifier>(
        binding: &MeasurementKeyBinding,
        current_key_state_root: KeyStateRoot,
        verification_key: &VerificationKey,
        frame: &ScreenFrameObservation,
        signed: &SignedMeasurement,
        verifier: &V,
    ) -> Result<Self, MeasuredAttestationError> {
        let subject = MeasurementSubject::Frame(
            frame
                .observation_id()
                .map_err(|_| MeasuredAttestationError::SubjectMismatch)?,
        );
        let message_id = measurement_message_id(binding, subject)?;
        Self::authenticate(
            binding,
            current_key_state_root,
            verification_key,
            subject,
            message_id,
            signed,
            verifier,
        )
    }

    pub fn authenticate_touch<V: AttestationVerifier>(
        binding: &MeasurementKeyBinding,
        current_key_state_root: KeyStateRoot,
        verification_key: &VerificationKey,
        touch: &TouchEventObservation,
        signed: &SignedMeasurement,
        verifier: &V,
    ) -> Result<Self, MeasuredAttestationError> {
        let subject = MeasurementSubject::Touch(
            touch
                .observation_id()
                .map_err(|_| MeasuredAttestationError::SubjectMismatch)?,
        );
        let message_id = measurement_message_id(binding, subject)?;
        Self::authenticate(
            binding,
            current_key_state_root,
            verification_key,
            subject,
            message_id,
            signed,
            verifier,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn authenticate<V: AttestationVerifier>(
        binding: &MeasurementKeyBinding,
        current_key_state_root: KeyStateRoot,
        verification_key: &VerificationKey,
        expected_subject: MeasurementSubject,
        expected_message_id: MeasurementMessageId,
        signed: &SignedMeasurement,
        verifier: &V,
    ) -> Result<Self, MeasuredAttestationError> {
        binding.validate()?;
        signed.validate()?;
        verification_key
            .validate()
            .map_err(|_| MeasuredAttestationError::VerificationKeyMismatch)?;
        if current_key_state_root.is_zero() {
            return Err(MeasuredAttestationError::ZeroKeyStateRoot);
        }
        if current_key_state_root != binding.key_state_root {
            return Err(MeasuredAttestationError::KeyStateChanged);
        }

        let binding_id = binding.binding_id()?;
        let verification_key_id = verification_key
            .key_id()
            .map_err(|_| MeasuredAttestationError::VerificationKeyMismatch)?;
        if verification_key.suite_id != binding.signature_suite_id {
            return Err(MeasuredAttestationError::SignatureSuiteMismatch);
        }
        if verification_key_id != binding.verification_key_id {
            return Err(MeasuredAttestationError::VerificationKeyMismatch);
        }
        if signed.binding_id != binding_id {
            return Err(MeasuredAttestationError::BindingMismatch);
        }
        if signed.attester_id != binding.attester_id {
            return Err(MeasuredAttestationError::AttesterMismatch);
        }
        if signed.role != binding.role {
            return Err(MeasuredAttestationError::RoleMismatch);
        }
        if signed.signature_suite_id != binding.signature_suite_id {
            return Err(MeasuredAttestationError::SignatureSuiteMismatch);
        }
        if signed.verification_key_id != binding.verification_key_id {
            return Err(MeasuredAttestationError::VerificationKeyMismatch);
        }
        if signed.key_epoch != binding.key_epoch {
            return Err(MeasuredAttestationError::KeyEpochMismatch);
        }
        if signed.key_state_root != binding.key_state_root {
            return Err(MeasuredAttestationError::KeyStateChanged);
        }
        if signed.subject != expected_subject {
            return Err(MeasuredAttestationError::SubjectMismatch);
        }
        if signed.message_id != expected_message_id {
            return Err(MeasuredAttestationError::MessageMismatch);
        }
        if !verifier.verify(
            binding.signature_suite_id,
            &verification_key.public_key,
            expected_message_id.as_bytes(),
            &signed.signature,
        ) {
            return Err(MeasuredAttestationError::SignatureInvalid);
        }

        Ok(Self {
            binding_id,
            attester_id: binding.attester_id,
            role: binding.role,
            subject: expected_subject,
            message_id: expected_message_id,
            verification_key_id,
            key_epoch: binding.key_epoch,
            key_state_root: binding.key_state_root,
            signature_commitment: signed.signature_commitment()?,
        })
    }

    pub fn certificate_id(
        &self,
    ) -> Result<AuthenticatedMeasurementCertificateId, MeasuredAttestationError> {
        if self.binding_id.is_zero()
            || self.attester_id.is_zero()
            || self.message_id.is_zero()
            || self.verification_key_id.is_zero()
            || self.key_epoch == 0
            || self.key_state_root.is_zero()
            || self.signature_commitment.is_zero()
        {
            return Err(MeasuredAttestationError::ZeroBinding);
        }
        self.subject.validate_for_role(self.role)?;

        let mut hasher = blake3::Hasher::new();
        hasher.update(CERTIFICATE_DOMAIN);
        hasher.update(self.binding_id.as_bytes());
        hasher.update(self.attester_id.as_bytes());
        hasher.update(&[self.role as u8]);
        self.subject.update_hasher(&mut hasher);
        hasher.update(self.message_id.as_bytes());
        hasher.update(self.verification_key_id.as_bytes());
        hasher.update(&self.key_epoch.to_le_bytes());
        hasher.update(self.key_state_root.as_bytes());
        hasher.update(self.signature_commitment.as_bytes());
        Ok(AuthenticatedMeasurementCertificateId(
            *hasher.finalize().as_bytes(),
        ))
    }
}

impl AuthenticatedMeasuredInteraction {
    pub fn link(
        verification: &MeasuredInteractionVerificationCertificate,
        frame_auth: &AuthenticatedMeasurementCertificate,
        touch_auth: &AuthenticatedMeasurementCertificate,
    ) -> Result<Self, MeasuredAttestationError> {
        if frame_auth.subject != MeasurementSubject::Frame(verification.frame_observation_id) {
            return Err(MeasuredAttestationError::VerificationFrameMismatch);
        }
        if frame_auth.role != MeasurementRole::FramebufferCapture {
            return Err(MeasuredAttestationError::RoleMismatch);
        }
        if touch_auth.subject != MeasurementSubject::Touch(verification.touch_observation_id) {
            return Err(MeasuredAttestationError::VerificationTouchMismatch);
        }
        if touch_auth.role != MeasurementRole::TouchInputCapture {
            return Err(MeasuredAttestationError::RoleMismatch);
        }

        Ok(Self {
            verification_certificate_id: verification
                .certificate_id()
                .map_err(|_| MeasuredAttestationError::VerificationFrameMismatch)?,
            frame_auth_certificate_id: frame_auth.certificate_id()?,
            touch_auth_certificate_id: touch_auth.certificate_id()?,
        })
    }

    pub fn interaction_id(
        &self,
    ) -> Result<AuthenticatedMeasuredInteractionId, MeasuredAttestationError> {
        if self.verification_certificate_id.is_zero()
            || self.frame_auth_certificate_id.is_zero()
            || self.touch_auth_certificate_id.is_zero()
        {
            return Err(MeasuredAttestationError::ZeroAuthenticatedBundleComponent);
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(BUNDLE_DOMAIN);
        hasher.update(self.verification_certificate_id.as_bytes());
        hasher.update(self.frame_auth_certificate_id.as_bytes());
        hasher.update(self.touch_auth_certificate_id.as_bytes());
        Ok(AuthenticatedMeasuredInteractionId(
            *hasher.finalize().as_bytes(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance_soma_interaction::{
        ScreenCaptureProfileId, TouchInputProfileId,
    };
    use crate::touch_body::{TouchAction, TouchEvent};
    use symthaea_core::assurance_attester_auth::VerificationKey;
    use symthaea_core::assurance_interaction_continuity::InputAttesterId;
    use symthaea_core::assurance_trusted_surface::TrustedSurfaceId;

    fn bytes(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    struct TestVerifier;

    impl AttestationVerifier for TestVerifier {
        fn verify(
            &self,
            _suite_id: SignatureSuiteId,
            public_key: &[u8],
            message: &[u8; 32],
            signature: &[u8],
        ) -> bool {
            signature == test_signature(public_key, message)
        }
    }

    fn test_signature(public_key: &[u8], message: &[u8; 32]) -> Vec<u8> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(public_key.len() as u64).to_le_bytes());
        hasher.update(public_key);
        hasher.update(message);
        hasher.finalize().as_bytes().to_vec()
    }

    fn key() -> VerificationKey {
        VerificationKey {
            suite_id: SignatureSuiteId(bytes(10)),
            public_key: b"measurement-test-key".to_vec(),
        }
    }

    fn binding(role: MeasurementRole, key: &VerificationKey, root: KeyStateRoot) -> MeasurementKeyBinding {
        MeasurementKeyBinding {
            attester_id: MeasurementAttesterId(bytes(match role {
                MeasurementRole::FramebufferCapture => 20,
                MeasurementRole::TouchInputCapture => 21,
            })),
            role,
            signature_suite_id: key.suite_id,
            verification_key_id: key.key_id().unwrap(),
            key_epoch: 2,
            key_state_root: root,
        }
    }

    fn frame() -> ScreenFrameObservation {
        ScreenFrameObservation::from_rgb(
            ScreenCaptureProfileId(bytes(30)),
            TrustedSurfaceId(bytes(31)),
            3,
            4,
            0,
            2,
            1,
            &[1, 2, 3, 4, 5, 6],
        )
        .unwrap()
    }

    fn touch() -> TouchEventObservation {
        TouchEventObservation::from_touch_event(
            TouchInputProfileId(bytes(32)),
            InputAttesterId(bytes(33)),
            4,
            5,
            &TouchEvent {
                x: 0.25,
                y: 0.75,
                action: TouchAction::Up,
                pressure: 0.5,
                timestamp_ms: 999,
            },
        )
        .unwrap()
    }

    fn signed_frame(
        binding: &MeasurementKeyBinding,
        key: &VerificationKey,
        frame: &ScreenFrameObservation,
    ) -> SignedMeasurement {
        let subject = MeasurementSubject::Frame(frame.observation_id().unwrap());
        let message = frame_message_id(binding, frame).unwrap();
        SignedMeasurement::from_binding(
            binding,
            subject,
            message,
            test_signature(&key.public_key, message.as_bytes()),
        )
        .unwrap()
    }

    fn signed_touch(
        binding: &MeasurementKeyBinding,
        key: &VerificationKey,
        touch: &TouchEventObservation,
    ) -> SignedMeasurement {
        let subject = MeasurementSubject::Touch(touch.observation_id().unwrap());
        let message = touch_message_id(binding, touch).unwrap();
        SignedMeasurement::from_binding(
            binding,
            subject,
            message,
            test_signature(&key.public_key, message.as_bytes()),
        )
        .unwrap()
    }

    #[test]
    fn exact_frame_measurement_authenticates() {
        let key = key();
        let root = KeyStateRoot(bytes(40));
        let binding = binding(MeasurementRole::FramebufferCapture, &key, root);
        let frame = frame();
        let signed = signed_frame(&binding, &key, &frame);
        let cert = AuthenticatedMeasurementCertificate::authenticate_frame(
            &binding,
            root,
            &key,
            &frame,
            &signed,
            &TestVerifier,
        )
        .unwrap();
        assert_ne!(cert.certificate_id().unwrap(), AuthenticatedMeasurementCertificateId::ZERO);
    }

    #[test]
    fn exact_touch_measurement_authenticates() {
        let key = key();
        let root = KeyStateRoot(bytes(41));
        let binding = binding(MeasurementRole::TouchInputCapture, &key, root);
        let touch = touch();
        let signed = signed_touch(&binding, &key, &touch);
        assert!(AuthenticatedMeasurementCertificate::authenticate_touch(
            &binding,
            root,
            &key,
            &touch,
            &signed,
            &TestVerifier,
        )
        .is_ok());
    }

    #[test]
    fn stale_measurement_key_state_is_rejected() {
        let key = key();
        let root = KeyStateRoot(bytes(42));
        let binding = binding(MeasurementRole::FramebufferCapture, &key, root);
        let frame = frame();
        let signed = signed_frame(&binding, &key, &frame);
        assert_eq!(
            AuthenticatedMeasurementCertificate::authenticate_frame(
                &binding,
                KeyStateRoot(bytes(43)),
                &key,
                &frame,
                &signed,
                &TestVerifier,
            ),
            Err(MeasuredAttestationError::KeyStateChanged)
        );
    }

    #[test]
    fn role_substitution_is_rejected() {
        let key = key();
        let root = KeyStateRoot(bytes(44));
        let binding = binding(MeasurementRole::TouchInputCapture, &key, root);
        assert_eq!(
            frame_message_id(&binding, &frame()),
            Err(MeasuredAttestationError::RoleSubjectMismatch)
        );
    }

    #[test]
    fn modified_signature_is_rejected() {
        let key = key();
        let root = KeyStateRoot(bytes(45));
        let binding = binding(MeasurementRole::FramebufferCapture, &key, root);
        let frame = frame();
        let mut signed = signed_frame(&binding, &key, &frame);
        signed.signature[0] ^= 1;
        assert_eq!(
            AuthenticatedMeasurementCertificate::authenticate_frame(
                &binding,
                root,
                &key,
                &frame,
                &signed,
                &TestVerifier,
            ),
            Err(MeasuredAttestationError::SignatureInvalid)
        );
    }
}
