// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed cryptographic provider boundary for flight evidence.
//!
//! `FlightRecorder` intentionally uses FNV-1a only for deterministic replay
//! equality. This module prevents that checksum from being presented as an
//! authenticity proof: production signing and verification require an explicit
//! external cryptographic provider and bind the signature to canonical recorder
//! bytes plus the segment seal.

use serde::{Deserialize, Serialize};

use crate::flight_recorder::{FlightRecorder, FlightRecorderError, FlightSegmentSeal};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceSignatureError {
    ProviderUnavailable,
    ProviderFailure,
    EmptyAlgorithm,
    EmptyKeyId,
    EmptyDigest,
    EmptySignature,
    RecorderInvalid,
    SealMismatch,
    DigestMismatch,
    SignatureInvalid,
    HexEncodingInvalid,
}

pub trait EvidenceCryptoProvider {
    /// Cryptographic digest algorithm identifier, for example `sha256`.
    fn digest_algorithm(&self) -> &str;
    /// Signature scheme identifier, for example `ed25519` or `ml-dsa-65`.
    fn signature_scheme(&self) -> &str;
    /// Stable public-key or certificate identifier.
    fn key_id(&self) -> &str;
    fn digest(&self, message: &[u8]) -> Result<Vec<u8>, EvidenceSignatureError>;
    fn sign_digest(&self, digest: &[u8]) -> Result<Vec<u8>, EvidenceSignatureError>;
    fn verify_digest(
        &self,
        digest: &[u8],
        signature: &[u8],
    ) -> Result<bool, EvidenceSignatureError>;
}

/// Default provider: cryptographic operations are unavailable rather than
/// silently falling back to a non-cryptographic checksum.
#[derive(Debug, Default, Clone, Copy)]
pub struct UnavailableEvidenceCrypto;

impl EvidenceCryptoProvider for UnavailableEvidenceCrypto {
    fn digest_algorithm(&self) -> &str {
        "unavailable"
    }

    fn signature_scheme(&self) -> &str {
        "unavailable"
    }

    fn key_id(&self) -> &str {
        ""
    }

    fn digest(&self, _message: &[u8]) -> Result<Vec<u8>, EvidenceSignatureError> {
        Err(EvidenceSignatureError::ProviderUnavailable)
    }

    fn sign_digest(&self, _digest: &[u8]) -> Result<Vec<u8>, EvidenceSignatureError> {
        Err(EvidenceSignatureError::ProviderUnavailable)
    }

    fn verify_digest(
        &self,
        _digest: &[u8],
        _signature: &[u8],
    ) -> Result<bool, EvidenceSignatureError> {
        Err(EvidenceSignatureError::ProviderUnavailable)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignedFlightSegmentSeal {
    pub seal: FlightSegmentSeal,
    pub digest_algorithm: String,
    pub evidence_digest_hex: String,
    pub signature_scheme: String,
    pub key_id: String,
    pub signature_hex: String,
}

impl SignedFlightSegmentSeal {
    pub fn sign<P: EvidenceCryptoProvider>(
        recorder: &FlightRecorder,
        provider: &P,
    ) -> Result<Self, EvidenceSignatureError> {
        validate_provider(provider)?;
        recorder.verify_record_chain().map_err(map_recorder_error)?;
        let canonical = recorder.canonical_json().map_err(map_recorder_error)?;
        let seal = recorder.seal_segment().map_err(map_recorder_error)?;
        let message = signature_message(&canonical, &seal)?;
        let digest = provider.digest(&message)?;
        if digest.is_empty() {
            return Err(EvidenceSignatureError::EmptyDigest);
        }
        let signature = provider.sign_digest(&digest)?;
        if signature.is_empty() {
            return Err(EvidenceSignatureError::EmptySignature);
        }
        Ok(Self {
            seal,
            digest_algorithm: provider.digest_algorithm().to_string(),
            evidence_digest_hex: encode_hex(&digest),
            signature_scheme: provider.signature_scheme().to_string(),
            key_id: provider.key_id().to_string(),
            signature_hex: encode_hex(&signature),
        })
    }

    pub fn verify<P: EvidenceCryptoProvider>(
        &self,
        recorder: &FlightRecorder,
        provider: &P,
    ) -> Result<(), EvidenceSignatureError> {
        validate_provider(provider)?;
        if self.digest_algorithm != provider.digest_algorithm()
            || self.signature_scheme != provider.signature_scheme()
            || self.key_id != provider.key_id()
        {
            return Err(EvidenceSignatureError::ProviderFailure);
        }
        recorder.verify_record_chain().map_err(map_recorder_error)?;
        let current_seal = recorder.seal_segment().map_err(map_recorder_error)?;
        if current_seal != self.seal {
            return Err(EvidenceSignatureError::SealMismatch);
        }
        let canonical = recorder.canonical_json().map_err(map_recorder_error)?;
        let message = signature_message(&canonical, &current_seal)?;
        let computed_digest = provider.digest(&message)?;
        let recorded_digest = decode_hex(&self.evidence_digest_hex)?;
        if computed_digest != recorded_digest {
            return Err(EvidenceSignatureError::DigestMismatch);
        }
        let signature = decode_hex(&self.signature_hex)?;
        if !provider.verify_digest(&computed_digest, &signature)? {
            return Err(EvidenceSignatureError::SignatureInvalid);
        }
        Ok(())
    }
}

fn validate_provider<P: EvidenceCryptoProvider>(
    provider: &P,
) -> Result<(), EvidenceSignatureError> {
    if provider.digest_algorithm().trim().is_empty() {
        return Err(EvidenceSignatureError::EmptyAlgorithm);
    }
    if provider.signature_scheme().trim().is_empty() {
        return Err(EvidenceSignatureError::EmptyAlgorithm);
    }
    if provider.key_id().trim().is_empty() {
        return Err(EvidenceSignatureError::EmptyKeyId);
    }
    Ok(())
}

fn signature_message(
    canonical_recorder: &[u8],
    seal: &FlightSegmentSeal,
) -> Result<Vec<u8>, EvidenceSignatureError> {
    let seal_bytes =
        serde_json::to_vec(seal).map_err(|_| EvidenceSignatureError::ProviderFailure)?;
    let mut message = Vec::with_capacity(canonical_recorder.len() + seal_bytes.len() + 64);
    message.extend_from_slice(b"symthaea-helicopter-flight-evidence-v1\0");
    message.extend_from_slice(&(canonical_recorder.len() as u64).to_le_bytes());
    message.extend_from_slice(canonical_recorder);
    message.extend_from_slice(&(seal_bytes.len() as u64).to_le_bytes());
    message.extend_from_slice(&seal_bytes);
    Ok(message)
}

fn map_recorder_error(_error: FlightRecorderError) -> EvidenceSignatureError {
    EvidenceSignatureError::RecorderInvalid
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn decode_hex(value: &str) -> Result<Vec<u8>, EvidenceSignatureError> {
    if value.len() % 2 != 0 {
        return Err(EvidenceSignatureError::HexEncodingInvalid);
    }
    let bytes = value.as_bytes();
    let mut output = Vec::with_capacity(bytes.len() / 2);
    for index in (0..bytes.len()).step_by(2) {
        let high = decode_nibble(bytes[index])?;
        let low = decode_nibble(bytes[index + 1])?;
        output.push((high << 4) | low);
    }
    Ok(output)
}

fn decode_nibble(value: u8) -> Result<u8, EvidenceSignatureError> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err(EvidenceSignatureError::HexEncodingInvalid),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flight_recorder::{FlightEvent, FlightEventKind, FlightLogManifest};

    struct TestOnlyProvider;

    impl EvidenceCryptoProvider for TestOnlyProvider {
        fn digest_algorithm(&self) -> &str {
            "test-only-fnv64"
        }

        fn signature_scheme(&self) -> &str {
            "test-only-mac"
        }

        fn key_id(&self) -> &str {
            "test-key"
        }

        fn digest(&self, message: &[u8]) -> Result<Vec<u8>, EvidenceSignatureError> {
            let mut hash = 0xcbf29ce484222325u64;
            for byte in message {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            Ok(hash.to_le_bytes().to_vec())
        }

        fn sign_digest(&self, digest: &[u8]) -> Result<Vec<u8>, EvidenceSignatureError> {
            Ok(digest.iter().map(|byte| byte ^ 0x5a).collect())
        }

        fn verify_digest(
            &self,
            digest: &[u8],
            signature: &[u8],
        ) -> Result<bool, EvidenceSignatureError> {
            Ok(signature == digest.iter().map(|byte| byte ^ 0x5a).collect::<Vec<_>>())
        }
    }

    fn recorder() -> FlightRecorder {
        let mut recorder = FlightRecorder::new(
            FlightLogManifest {
                schema_version: "1".to_string(),
                scenario_id: "signature-test".to_string(),
                controller_id: "controller".to_string(),
                seed: 7,
                physics_hz: 300.0,
            },
            4,
        )
        .unwrap();
        recorder
            .record_event(FlightEvent {
                sequence: 1,
                monotonic_time_s: 0.0,
                kind: FlightEventKind::OperatorAnnotation {
                    text: "start".to_string(),
                },
            })
            .unwrap();
        recorder
    }

    #[test]
    fn external_provider_signs_and_verifies_canonical_segment() {
        let recorder = recorder();
        let signed = SignedFlightSegmentSeal::sign(&recorder, &TestOnlyProvider).unwrap();
        assert!(signed.verify(&recorder, &TestOnlyProvider).is_ok());
    }

    #[test]
    fn unavailable_provider_never_falls_back_to_fnv() {
        assert_eq!(
            SignedFlightSegmentSeal::sign(&recorder(), &UnavailableEvidenceCrypto),
            Err(EvidenceSignatureError::EmptyKeyId)
        );
    }

    #[test]
    fn changed_seal_is_rejected() {
        let recorder = recorder();
        let mut signed = SignedFlightSegmentSeal::sign(&recorder, &TestOnlyProvider).unwrap();
        signed.seal.record_count += 1;
        assert_eq!(
            signed.verify(&recorder, &TestOnlyProvider),
            Err(EvidenceSignatureError::SealMismatch)
        );
    }
}
