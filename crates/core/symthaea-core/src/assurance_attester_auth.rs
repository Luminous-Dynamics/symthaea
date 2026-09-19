// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-ATTESTERAUTH-515: crypto-agile authentication for presentation-surface
//! and confirmation-input attesters.
//!
//! QUAL-TRUSTEDSURFACE-512 and QUAL-INTERACTIONCONTINUITY-514 deliberately treat
//! adapter observations as statements rather than self-authenticating facts. This
//! module closes that boundary without coupling `symthaea-core` to one signature
//! algorithm: an exact attester/role/key/epoch/current-key-state binding defines
//! which verification key is authorized, a canonical message commits the exact
//! observation being authenticated, and an injected `AttestationVerifier` checks
//! the concrete signature suite.
//!
//! The theorem is conditional on the supplied key binding and verifier being
//! independently qualified. A valid signature proves possession of the bound key
//! for the exact message; it does not prove physical display/input-device
//! integrity, human perception, or that the key-to-attester binding is legally or
//! institutionally correct beyond the active policy state supplied here.

use core::fmt;

use crate::assurance_interaction_continuity::{
    ConfirmationInputObservation, InputAttesterId, InteractionContinuityError,
};
use crate::assurance_trusted_surface::{
    SurfaceAttesterId, SurfaceDeliveryObservation, TrustedSurfaceError,
};

const KEY_DOMAIN: &[u8] = b"symthaea.presentation.v1/attestation-verification-key\0";
const BINDING_DOMAIN: &[u8] = b"symthaea.presentation.v1/attester-key-binding\0";
const SURFACE_MESSAGE_DOMAIN: &[u8] = b"symthaea.presentation.v1/surface-attestation-message\0";
const INPUT_MESSAGE_DOMAIN: &[u8] = b"symthaea.presentation.v1/input-attestation-message\0";
const SIGNATURE_DOMAIN: &[u8] = b"symthaea.presentation.v1/signature-commitment\0";
const CERTIFICATE_DOMAIN: &[u8] = b"symthaea.presentation.v1/authenticated-attestation-certificate\0";

const MAX_KEY_BYTES: usize = 16 * 1024;
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

digest_id!(SignatureSuiteId);
digest_id!(VerificationKeyId);
digest_id!(KeyStateRoot);
digest_id!(AttesterKeyBindingId);
digest_id!(AttestationMessageId);
digest_id!(SignatureCommitment);
digest_id!(AuthenticatedAttestationCertificateId);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum AttesterRole {
    SurfaceDelivery = 1,
    ConfirmationInput = 2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttesterSubject {
    Surface(SurfaceAttesterId),
    Input(InputAttesterId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerificationKey {
    pub suite_id: SignatureSuiteId,
    pub public_key: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttesterKeyBinding {
    pub subject: AttesterSubject,
    pub role: AttesterRole,
    pub signature_suite_id: SignatureSuiteId,
    pub verification_key_id: VerificationKeyId,
    pub key_epoch: u64,
    pub key_state_root: KeyStateRoot,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SignedAttestation {
    pub binding_id: AttesterKeyBindingId,
    pub subject: AttesterSubject,
    pub role: AttesterRole,
    pub signature_suite_id: SignatureSuiteId,
    pub verification_key_id: VerificationKeyId,
    pub key_epoch: u64,
    pub key_state_root: KeyStateRoot,
    pub message_id: AttestationMessageId,
    pub signature: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuthenticatedAttestationCertificate {
    pub binding_id: AttesterKeyBindingId,
    pub subject: AttesterSubject,
    pub role: AttesterRole,
    pub message_id: AttestationMessageId,
    pub verification_key_id: VerificationKeyId,
    pub key_epoch: u64,
    pub key_state_root: KeyStateRoot,
    pub signature_commitment: SignatureCommitment,
}

/// Concrete cryptographic verification lives outside this dependency-light core.
/// Implementations must define the exact semantics named by `suite_id`.
pub trait AttestationVerifier {
    fn verify(
        &self,
        suite_id: SignatureSuiteId,
        public_key: &[u8],
        message: &[u8; 32],
        signature: &[u8],
    ) -> bool;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttesterAuthError {
    ZeroSignatureSuite,
    EmptyVerificationKey,
    VerificationKeyTooLarge,
    ZeroVerificationKey,
    ZeroKeyEpoch,
    ZeroKeyStateRoot,
    RoleSubjectMismatch,
    ZeroBinding,
    EmptySignature,
    SignatureTooLarge,
    BindingMismatch,
    SubjectMismatch,
    RoleMismatch,
    SignatureSuiteMismatch,
    VerificationKeyMismatch,
    KeyEpochMismatch,
    KeyStateChanged,
    MessageMismatch,
    ObservationAttesterMismatch,
    SignatureInvalid,
    TrustedSurface(TrustedSurfaceError),
    InteractionContinuity(InteractionContinuityError),
}

impl fmt::Display for AttesterAuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AttesterAuthError {}

impl From<TrustedSurfaceError> for AttesterAuthError {
    fn from(value: TrustedSurfaceError) -> Self {
        Self::TrustedSurface(value)
    }
}

impl From<InteractionContinuityError> for AttesterAuthError {
    fn from(value: InteractionContinuityError) -> Self {
        Self::InteractionContinuity(value)
    }
}

impl AttesterSubject {
    fn validate_for_role(self, role: AttesterRole) -> Result<(), AttesterAuthError> {
        let valid = matches!(
            (self, role),
            (Self::Surface(_), AttesterRole::SurfaceDelivery)
                | (Self::Input(_), AttesterRole::ConfirmationInput)
        );
        if !valid {
            return Err(AttesterAuthError::RoleSubjectMismatch);
        }
        match self {
            Self::Surface(id) if id.is_zero() => Err(AttesterAuthError::SubjectMismatch),
            Self::Input(id) if id.is_zero() => Err(AttesterAuthError::SubjectMismatch),
            _ => Ok(()),
        }
    }

    fn update_hasher(self, hasher: &mut blake3::Hasher) {
        match self {
            Self::Surface(id) => {
                hasher.update(&[1]);
                hasher.update(id.as_bytes());
            }
            Self::Input(id) => {
                hasher.update(&[2]);
                hasher.update(id.as_bytes());
            }
        }
    }
}

impl VerificationKey {
    pub fn validate(&self) -> Result<(), AttesterAuthError> {
        if self.suite_id.is_zero() {
            return Err(AttesterAuthError::ZeroSignatureSuite);
        }
        if self.public_key.is_empty() {
            return Err(AttesterAuthError::EmptyVerificationKey);
        }
        if self.public_key.len() > MAX_KEY_BYTES {
            return Err(AttesterAuthError::VerificationKeyTooLarge);
        }
        Ok(())
    }

    pub fn key_id(&self) -> Result<VerificationKeyId, AttesterAuthError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(KEY_DOMAIN);
        hasher.update(self.suite_id.as_bytes());
        put_bytes(&mut hasher, &self.public_key);
        Ok(VerificationKeyId(*hasher.finalize().as_bytes()))
    }
}

impl AttesterKeyBinding {
    pub fn validate(&self) -> Result<(), AttesterAuthError> {
        self.subject.validate_for_role(self.role)?;
        if self.signature_suite_id.is_zero() {
            return Err(AttesterAuthError::ZeroSignatureSuite);
        }
        if self.verification_key_id.is_zero() {
            return Err(AttesterAuthError::ZeroVerificationKey);
        }
        if self.key_epoch == 0 {
            return Err(AttesterAuthError::ZeroKeyEpoch);
        }
        if self.key_state_root.is_zero() {
            return Err(AttesterAuthError::ZeroKeyStateRoot);
        }
        Ok(())
    }

    pub fn binding_id(&self) -> Result<AttesterKeyBindingId, AttesterAuthError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(BINDING_DOMAIN);
        self.subject.update_hasher(&mut hasher);
        hasher.update(&[self.role as u8]);
        hasher.update(self.signature_suite_id.as_bytes());
        hasher.update(self.verification_key_id.as_bytes());
        hasher.update(&self.key_epoch.to_le_bytes());
        hasher.update(self.key_state_root.as_bytes());
        Ok(AttesterKeyBindingId(*hasher.finalize().as_bytes()))
    }
}

impl SignedAttestation {
    pub fn from_binding(
        binding: &AttesterKeyBinding,
        message_id: AttestationMessageId,
        signature: Vec<u8>,
    ) -> Result<Self, AttesterAuthError> {
        binding.validate()?;
        if message_id.is_zero() {
            return Err(AttesterAuthError::MessageMismatch);
        }
        if signature.is_empty() {
            return Err(AttesterAuthError::EmptySignature);
        }
        if signature.len() > MAX_SIGNATURE_BYTES {
            return Err(AttesterAuthError::SignatureTooLarge);
        }
        Ok(Self {
            binding_id: binding.binding_id()?,
            subject: binding.subject,
            role: binding.role,
            signature_suite_id: binding.signature_suite_id,
            verification_key_id: binding.verification_key_id,
            key_epoch: binding.key_epoch,
            key_state_root: binding.key_state_root,
            message_id,
            signature,
        })
    }

    fn validate(&self) -> Result<(), AttesterAuthError> {
        if self.binding_id.is_zero() {
            return Err(AttesterAuthError::ZeroBinding);
        }
        self.subject.validate_for_role(self.role)?;
        if self.signature_suite_id.is_zero() {
            return Err(AttesterAuthError::ZeroSignatureSuite);
        }
        if self.verification_key_id.is_zero() {
            return Err(AttesterAuthError::ZeroVerificationKey);
        }
        if self.key_epoch == 0 {
            return Err(AttesterAuthError::ZeroKeyEpoch);
        }
        if self.key_state_root.is_zero() {
            return Err(AttesterAuthError::ZeroKeyStateRoot);
        }
        if self.message_id.is_zero() {
            return Err(AttesterAuthError::MessageMismatch);
        }
        if self.signature.is_empty() {
            return Err(AttesterAuthError::EmptySignature);
        }
        if self.signature.len() > MAX_SIGNATURE_BYTES {
            return Err(AttesterAuthError::SignatureTooLarge);
        }
        Ok(())
    }

    fn signature_commitment(&self) -> Result<SignatureCommitment, AttesterAuthError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(SIGNATURE_DOMAIN);
        put_bytes(&mut hasher, &self.signature);
        Ok(SignatureCommitment(*hasher.finalize().as_bytes()))
    }
}

pub fn surface_attestation_message_id(
    binding: &AttesterKeyBinding,
    observation: &SurfaceDeliveryObservation,
) -> Result<AttestationMessageId, AttesterAuthError> {
    binding.validate()?;
    observation.validate()?;
    let expected_subject = AttesterSubject::Surface(observation.attester_id);
    if binding.role != AttesterRole::SurfaceDelivery {
        return Err(AttesterAuthError::RoleMismatch);
    }
    if binding.subject != expected_subject {
        return Err(AttesterAuthError::ObservationAttesterMismatch);
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(SURFACE_MESSAGE_DOMAIN);
    hasher.update(binding.binding_id()?.as_bytes());
    hasher.update(observation.observation_id()?.as_bytes());
    Ok(AttestationMessageId(*hasher.finalize().as_bytes()))
}

pub fn input_attestation_message_id(
    binding: &AttesterKeyBinding,
    observation: &ConfirmationInputObservation,
) -> Result<AttestationMessageId, AttesterAuthError> {
    binding.validate()?;
    observation.validate()?;
    let expected_subject = AttesterSubject::Input(observation.input_attester_id);
    if binding.role != AttesterRole::ConfirmationInput {
        return Err(AttesterAuthError::RoleMismatch);
    }
    if binding.subject != expected_subject {
        return Err(AttesterAuthError::ObservationAttesterMismatch);
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(INPUT_MESSAGE_DOMAIN);
    hasher.update(binding.binding_id()?.as_bytes());
    hasher.update(observation.observation_id()?.as_bytes());
    Ok(AttestationMessageId(*hasher.finalize().as_bytes()))
}

impl AuthenticatedAttestationCertificate {
    pub fn authenticate_surface<V: AttestationVerifier>(
        binding: &AttesterKeyBinding,
        current_key_state_root: KeyStateRoot,
        verification_key: &VerificationKey,
        observation: &SurfaceDeliveryObservation,
        signed: &SignedAttestation,
        verifier: &V,
    ) -> Result<Self, AttesterAuthError> {
        let message_id = surface_attestation_message_id(binding, observation)?;
        Self::authenticate(
            binding,
            current_key_state_root,
            verification_key,
            message_id,
            signed,
            verifier,
        )
    }

    pub fn authenticate_input<V: AttestationVerifier>(
        binding: &AttesterKeyBinding,
        current_key_state_root: KeyStateRoot,
        verification_key: &VerificationKey,
        observation: &ConfirmationInputObservation,
        signed: &SignedAttestation,
        verifier: &V,
    ) -> Result<Self, AttesterAuthError> {
        let message_id = input_attestation_message_id(binding, observation)?;
        Self::authenticate(
            binding,
            current_key_state_root,
            verification_key,
            message_id,
            signed,
            verifier,
        )
    }

    fn authenticate<V: AttestationVerifier>(
        binding: &AttesterKeyBinding,
        current_key_state_root: KeyStateRoot,
        verification_key: &VerificationKey,
        expected_message_id: AttestationMessageId,
        signed: &SignedAttestation,
        verifier: &V,
    ) -> Result<Self, AttesterAuthError> {
        binding.validate()?;
        verification_key.validate()?;
        signed.validate()?;
        if current_key_state_root.is_zero() {
            return Err(AttesterAuthError::ZeroKeyStateRoot);
        }
        if current_key_state_root != binding.key_state_root {
            return Err(AttesterAuthError::KeyStateChanged);
        }

        let binding_id = binding.binding_id()?;
        let verification_key_id = verification_key.key_id()?;
        if verification_key.suite_id != binding.signature_suite_id {
            return Err(AttesterAuthError::SignatureSuiteMismatch);
        }
        if verification_key_id != binding.verification_key_id {
            return Err(AttesterAuthError::VerificationKeyMismatch);
        }
        if signed.binding_id != binding_id {
            return Err(AttesterAuthError::BindingMismatch);
        }
        if signed.subject != binding.subject {
            return Err(AttesterAuthError::SubjectMismatch);
        }
        if signed.role != binding.role {
            return Err(AttesterAuthError::RoleMismatch);
        }
        if signed.signature_suite_id != binding.signature_suite_id {
            return Err(AttesterAuthError::SignatureSuiteMismatch);
        }
        if signed.verification_key_id != binding.verification_key_id {
            return Err(AttesterAuthError::VerificationKeyMismatch);
        }
        if signed.key_epoch != binding.key_epoch {
            return Err(AttesterAuthError::KeyEpochMismatch);
        }
        if signed.key_state_root != binding.key_state_root {
            return Err(AttesterAuthError::KeyStateChanged);
        }
        if signed.message_id != expected_message_id {
            return Err(AttesterAuthError::MessageMismatch);
        }
        if !verifier.verify(
            binding.signature_suite_id,
            &verification_key.public_key,
            expected_message_id.as_bytes(),
            &signed.signature,
        ) {
            return Err(AttesterAuthError::SignatureInvalid);
        }

        Ok(Self {
            binding_id,
            subject: binding.subject,
            role: binding.role,
            message_id: expected_message_id,
            verification_key_id,
            key_epoch: binding.key_epoch,
            key_state_root: binding.key_state_root,
            signature_commitment: signed.signature_commitment()?,
        })
    }

    pub fn certificate_id(
        &self,
    ) -> Result<AuthenticatedAttestationCertificateId, AttesterAuthError> {
        self.validate_nonzero()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(CERTIFICATE_DOMAIN);
        hasher.update(self.binding_id.as_bytes());
        self.subject.update_hasher(&mut hasher);
        hasher.update(&[self.role as u8]);
        hasher.update(self.message_id.as_bytes());
        hasher.update(self.verification_key_id.as_bytes());
        hasher.update(&self.key_epoch.to_le_bytes());
        hasher.update(self.key_state_root.as_bytes());
        hasher.update(self.signature_commitment.as_bytes());
        Ok(AuthenticatedAttestationCertificateId(
            *hasher.finalize().as_bytes(),
        ))
    }

    fn validate_nonzero(&self) -> Result<(), AttesterAuthError> {
        if self.binding_id.is_zero() {
            return Err(AttesterAuthError::ZeroBinding);
        }
        self.subject.validate_for_role(self.role)?;
        if self.message_id.is_zero() {
            return Err(AttesterAuthError::MessageMismatch);
        }
        if self.verification_key_id.is_zero() {
            return Err(AttesterAuthError::ZeroVerificationKey);
        }
        if self.key_epoch == 0 {
            return Err(AttesterAuthError::ZeroKeyEpoch);
        }
        if self.key_state_root.is_zero() {
            return Err(AttesterAuthError::ZeroKeyStateRoot);
        }
        if self.signature_commitment.is_zero() {
            return Err(AttesterAuthError::SignatureInvalid);
        }
        Ok(())
    }
}

fn put_len(hasher: &mut blake3::Hasher, len: usize) {
    hasher.update(&(len as u64).to_le_bytes());
}

fn put_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    put_len(hasher, bytes.len());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance::{
        ActionRequestId, AuthenticationContextId, ConfirmationId, InteractionContextId,
        PresentationContextId, PresentationId, PrincipalId,
    };
    use crate::assurance_interaction_continuity::{
        ConfirmationInputObservationId, InputEventNonce, InteractionContinuityProfileId,
    };
    use crate::assurance_render_artifact::{RenderArtifactCertificateId, RenderArtifactSetRoot};
    use crate::assurance_trusted_surface::{
        InteractionNonce, SurfaceDeliveryCertificateId, TrustedSurfaceId,
        TrustedSurfaceProfileId,
    };

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
        put_bytes(&mut hasher, public_key);
        hasher.update(message);
        hasher.finalize().as_bytes().to_vec()
    }

    fn suite() -> SignatureSuiteId {
        SignatureSuiteId(bytes(70))
    }

    fn key() -> VerificationKey {
        VerificationKey {
            suite_id: suite(),
            public_key: b"qualified-test-key".to_vec(),
        }
    }

    fn surface_observation() -> SurfaceDeliveryObservation {
        SurfaceDeliveryObservation {
            surface_profile_id: TrustedSurfaceProfileId(bytes(40)),
            surface_id: TrustedSurfaceId(bytes(41)),
            attester_id: SurfaceAttesterId(bytes(42)),
            surface_generation: 3,
            presentation_id: PresentationId(bytes(43)),
            presentation_context_id: PresentationContextId(bytes(44)),
            render_artifact_certificate_id: RenderArtifactCertificateId(bytes(45)),
            artifact_set_root: RenderArtifactSetRoot(bytes(46)),
            freshness_certificate_id: crate::assurance_presentation_currentness::PresentationFreshnessCertificateId(bytes(47)),
            delivery_sequence: 5,
            interaction_nonce: InteractionNonce(bytes(48)),
        }
    }

    fn input_observation() -> ConfirmationInputObservation {
        ConfirmationInputObservation {
            continuity_profile_id: InteractionContinuityProfileId(bytes(50)),
            principal_id: PrincipalId(bytes(51)),
            action_request_id: ActionRequestId(bytes(52)),
            presentation_id: PresentationId(bytes(53)),
            confirmation_id: ConfirmationId(bytes(54)),
            presentation_context_id: PresentationContextId(bytes(55)),
            interaction_context_id: InteractionContextId(bytes(56)),
            authentication_context_id: AuthenticationContextId(bytes(57)),
            surface_delivery_certificate_id: SurfaceDeliveryCertificateId(bytes(58)),
            predecessor_delivery_observation_id: crate::assurance_trusted_surface::SurfaceDeliveryObservationId(bytes(59)),
            input_attester_id: InputAttesterId(bytes(60)),
            interaction_generation: 4,
            input_sequence: 6,
            input_nonce: InputEventNonce(bytes(61)),
        }
    }

    fn surface_binding(key: &VerificationKey, root: KeyStateRoot) -> AttesterKeyBinding {
        AttesterKeyBinding {
            subject: AttesterSubject::Surface(SurfaceAttesterId(bytes(42))),
            role: AttesterRole::SurfaceDelivery,
            signature_suite_id: key.suite_id,
            verification_key_id: key.key_id().unwrap(),
            key_epoch: 2,
            key_state_root: root,
        }
    }

    fn input_binding(key: &VerificationKey, root: KeyStateRoot) -> AttesterKeyBinding {
        AttesterKeyBinding {
            subject: AttesterSubject::Input(InputAttesterId(bytes(60))),
            role: AttesterRole::ConfirmationInput,
            signature_suite_id: key.suite_id,
            verification_key_id: key.key_id().unwrap(),
            key_epoch: 2,
            key_state_root: root,
        }
    }

    #[test]
    fn exact_surface_signature_authenticates() {
        let key = key();
        let root = KeyStateRoot(bytes(80));
        let binding = surface_binding(&key, root);
        let observation = surface_observation();
        let message = surface_attestation_message_id(&binding, &observation).unwrap();
        let signed = SignedAttestation::from_binding(
            &binding,
            message,
            test_signature(&key.public_key, message.as_bytes()),
        )
        .unwrap();
        let cert = AuthenticatedAttestationCertificate::authenticate_surface(
            &binding,
            root,
            &key,
            &observation,
            &signed,
            &TestVerifier,
        )
        .unwrap();
        assert_ne!(cert.certificate_id().unwrap(), AuthenticatedAttestationCertificateId::ZERO);
    }

    #[test]
    fn exact_input_signature_authenticates() {
        let key = key();
        let root = KeyStateRoot(bytes(81));
        let binding = input_binding(&key, root);
        let observation = input_observation();
        let message = input_attestation_message_id(&binding, &observation).unwrap();
        let signed = SignedAttestation::from_binding(
            &binding,
            message,
            test_signature(&key.public_key, message.as_bytes()),
        )
        .unwrap();
        assert!(AuthenticatedAttestationCertificate::authenticate_input(
            &binding,
            root,
            &key,
            &observation,
            &signed,
            &TestVerifier,
        )
        .is_ok());
    }

    #[test]
    fn stale_key_state_is_rejected() {
        let key = key();
        let root = KeyStateRoot(bytes(82));
        let binding = surface_binding(&key, root);
        let observation = surface_observation();
        let message = surface_attestation_message_id(&binding, &observation).unwrap();
        let signed = SignedAttestation::from_binding(
            &binding,
            message,
            test_signature(&key.public_key, message.as_bytes()),
        )
        .unwrap();
        assert_eq!(
            AuthenticatedAttestationCertificate::authenticate_surface(
                &binding,
                KeyStateRoot(bytes(83)),
                &key,
                &observation,
                &signed,
                &TestVerifier,
            ),
            Err(AttesterAuthError::KeyStateChanged)
        );
    }

    #[test]
    fn wrong_verification_key_is_rejected() {
        let key = key();
        let root = KeyStateRoot(bytes(84));
        let binding = surface_binding(&key, root);
        let observation = surface_observation();
        let message = surface_attestation_message_id(&binding, &observation).unwrap();
        let signed = SignedAttestation::from_binding(
            &binding,
            message,
            test_signature(&key.public_key, message.as_bytes()),
        )
        .unwrap();
        let wrong_key = VerificationKey {
            suite_id: suite(),
            public_key: b"wrong-key".to_vec(),
        };
        assert_eq!(
            AuthenticatedAttestationCertificate::authenticate_surface(
                &binding,
                root,
                &wrong_key,
                &observation,
                &signed,
                &TestVerifier,
            ),
            Err(AttesterAuthError::VerificationKeyMismatch)
        );
    }

    #[test]
    fn modified_signature_is_rejected() {
        let key = key();
        let root = KeyStateRoot(bytes(85));
        let binding = surface_binding(&key, root);
        let observation = surface_observation();
        let message = surface_attestation_message_id(&binding, &observation).unwrap();
        let mut signature = test_signature(&key.public_key, message.as_bytes());
        signature[0] ^= 1;
        let signed = SignedAttestation::from_binding(&binding, message, signature).unwrap();
        assert_eq!(
            AuthenticatedAttestationCertificate::authenticate_surface(
                &binding,
                root,
                &key,
                &observation,
                &signed,
                &TestVerifier,
            ),
            Err(AttesterAuthError::SignatureInvalid)
        );
    }

    #[test]
    fn role_substitution_is_rejected() {
        let key = key();
        let root = KeyStateRoot(bytes(86));
        let mut binding = surface_binding(&key, root);
        binding.role = AttesterRole::ConfirmationInput;
        assert_eq!(
            binding.validate(),
            Err(AttesterAuthError::RoleSubjectMismatch)
        );
    }

    #[test]
    fn observation_attester_mismatch_is_rejected() {
        let key = key();
        let root = KeyStateRoot(bytes(87));
        let binding = surface_binding(&key, root);
        let mut observation = surface_observation();
        observation.attester_id = SurfaceAttesterId(bytes(99));
        assert_eq!(
            surface_attestation_message_id(&binding, &observation),
            Err(AttesterAuthError::ObservationAttesterMismatch)
        );
    }
}
