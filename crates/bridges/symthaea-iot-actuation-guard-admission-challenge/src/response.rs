// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::{Deserialize, Serialize};
use symthaea_authority::Digest32;
use symthaea_iot_posture::{DeviceAttestationResultV1, MAX_ATTESTATION_SIGNATURE_BYTES};

use crate::{
    ADMISSION_DEVICE_REALITY_RESPONSE_SCHEMA_VERSION, ATTESTATION_OBJECT_DOMAIN,
    AdmissionChallengeError, AdmissionRealityChallengeV1, MAX_ADMISSION_DEVICE_ATTESTATION_BYTES,
    MAX_ADMISSION_DEVICE_RESPONSE_BYTES, RESPONSE_DOMAIN, digest_bytes,
};

/// Portable response from a device-appraisal boundary. Only the exact signed attestation
/// result crosses the wire; no guard trust/policy/runtime/clock state can accompany it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmissionDeviceRealityResponseV1 {
    pub schema_version: u16,
    pub raw_attestation_result: Vec<u8>,
}

impl AdmissionDeviceRealityResponseV1 {
    pub fn validate_structure(&self) -> Result<(), AdmissionChallengeError> {
        if self.schema_version != ADMISSION_DEVICE_REALITY_RESPONSE_SCHEMA_VERSION {
            return Err(AdmissionChallengeError::UnsupportedResponseSchema);
        }
        if self.raw_attestation_result.is_empty()
            || self.raw_attestation_result.len() > MAX_ADMISSION_DEVICE_ATTESTATION_BYTES
        {
            return Err(AdmissionChallengeError::AttestationSizeOutOfBounds);
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, AdmissionChallengeError> {
        self.validate_structure()?;
        let bytes = bincode::serialize(self).map_err(AdmissionChallengeError::Encoding)?;
        if bytes.len() > MAX_ADMISSION_DEVICE_RESPONSE_BYTES {
            return Err(AdmissionChallengeError::ResponseSizeOutOfBounds);
        }
        Ok(bytes)
    }

    pub fn digest(&self) -> Result<Digest32, AdmissionChallengeError> {
        let bytes = self.canonical_bytes()?;
        Ok(digest_bytes(RESPONSE_DOMAIN, &bytes))
    }
}

/// Canonically decoded, challenge-correlated evidence. It is not trusted device reality
/// until a fixed relying-party verifier authenticates the retained signature/key lineage.
#[derive(Debug)]
pub struct DecodedAdmissionDeviceRealityEvidence {
    result: DeviceAttestationResultV1,
    attestation_object_digest: Digest32,
    response_digest: Digest32,
}

impl DecodedAdmissionDeviceRealityEvidence {
    pub fn result(&self) -> &DeviceAttestationResultV1 {
        &self.result
    }

    pub const fn attestation_object_digest(&self) -> Digest32 {
        self.attestation_object_digest
    }

    pub const fn response_digest(&self) -> Digest32 {
        self.response_digest
    }

    pub fn into_result(self) -> DeviceAttestationResultV1 {
        self.result
    }
}

pub fn decode_admission_device_reality_response(
    frame: &[u8],
    challenge: &AdmissionRealityChallengeV1,
) -> Result<DecodedAdmissionDeviceRealityEvidence, AdmissionChallengeError> {
    challenge.validate()?;
    if frame.is_empty() || frame.len() > MAX_ADMISSION_DEVICE_RESPONSE_BYTES {
        return Err(AdmissionChallengeError::ResponseSizeOutOfBounds);
    }

    let response: AdmissionDeviceRealityResponseV1 =
        bincode::deserialize(frame).map_err(AdmissionChallengeError::Decoding)?;
    response.validate_structure()?;
    if response.canonical_bytes()? != frame {
        return Err(AdmissionChallengeError::NonCanonicalResponseEncoding);
    }

    let result: DeviceAttestationResultV1 = bincode::deserialize(&response.raw_attestation_result)
        .map_err(|_| AdmissionChallengeError::InvalidAttestationEncoding)?;
    result
        .body
        .validate_structure()
        .map_err(|_| AdmissionChallengeError::InvalidAttestationStructure)?;
    if result.signature.is_empty() || result.signature.len() > MAX_ATTESTATION_SIGNATURE_BYTES {
        return Err(AdmissionChallengeError::InvalidAttestationSignatureSize);
    }

    let canonical_attestation =
        bincode::serialize(&result).map_err(|_| AdmissionChallengeError::InvalidAttestationEncoding)?;
    if canonical_attestation != response.raw_attestation_result {
        return Err(AdmissionChallengeError::NonCanonicalAttestationEncoding);
    }
    if result.body.device != *challenge.device() {
        return Err(AdmissionChallengeError::AttestationDeviceMismatch);
    }
    if result.body.challenge_digest != challenge.digest()? {
        return Err(AdmissionChallengeError::AttestationChallengeMismatch);
    }

    Ok(DecodedAdmissionDeviceRealityEvidence {
        result,
        attestation_object_digest: digest_bytes(ATTESTATION_OBJECT_DOMAIN, &canonical_attestation),
        response_digest: response.digest()?,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use symthaea_authority::Digest32;
    use symthaea_iot_posture::{
        DEVICE_ATTESTATION_RESULT_SCHEMA_VERSION, DeviceAttestationResultBodyV1,
    };

    use super::*;

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn result(challenge: &AdmissionRealityChallengeV1) -> DeviceAttestationResultV1 {
        DeviceAttestationResultV1 {
            body: DeviceAttestationResultBodyV1 {
                schema_version: DEVICE_ATTESTATION_RESULT_SCHEMA_VERSION,
                verifier_id: "verifier:fleet-a".into(),
                key_id: "device-key-1".into(),
                algorithm: "ed25519-rfc8032".into(),
                device: challenge.device().clone(),
                challenge_digest: challenge.digest().unwrap(),
                appraised_at_unix_s: 11,
                expires_at_unix_s: 12,
                evidence_digest: d(6),
                reference_values_digest: d(7),
                appraisal_policy_digest: d(8),
                running_firmware: d(9),
                last_accepted_sequence: Some(3),
                observations: BTreeMap::from([("pressure_x100".into(), 20_000)]),
            },
            signature: vec![0x44; 64],
        }
    }

    fn frame(result: &DeviceAttestationResultV1) -> Vec<u8> {
        AdmissionDeviceRealityResponseV1 {
            schema_version: ADMISSION_DEVICE_REALITY_RESPONSE_SCHEMA_VERSION,
            raw_attestation_result: bincode::serialize(result).unwrap(),
        }
        .canonical_bytes()
        .unwrap()
    }

    #[test]
    fn exact_response_binds_challenge_and_whole_signed_object() {
        let challenge = AdmissionRealityChallengeV1::fixture();
        let result = result(&challenge);
        let decoded = decode_admission_device_reality_response(&frame(&result), &challenge).unwrap();
        assert_eq!(decoded.result(), &result);
        assert_ne!(decoded.attestation_object_digest(), Digest32([0; 32]));
    }

    #[test]
    fn another_reservation_challenge_rejects_same_attestation() {
        let challenge = AdmissionRealityChallengeV1::fixture();
        let result = result(&challenge);
        let mut other = challenge.clone();
        other.test_set_reservation_digest(Digest32([0x99; 32]));
        assert!(matches!(
            decode_admission_device_reality_response(&frame(&result), &other),
            Err(AdmissionChallengeError::AttestationChallengeMismatch)
        ));
    }

    #[test]
    fn signature_bytes_change_full_attestation_object_commitment() {
        let challenge = AdmissionRealityChallengeV1::fixture();
        let result_a = result(&challenge);
        let mut result_b = result_a.clone();
        result_b.signature[0] ^= 1;
        let a = decode_admission_device_reality_response(&frame(&result_a), &challenge).unwrap();
        let b = decode_admission_device_reality_response(&frame(&result_b), &challenge).unwrap();
        assert_ne!(a.attestation_object_digest(), b.attestation_object_digest());
    }

    #[test]
    fn trailing_response_data_is_not_canonical() {
        let challenge = AdmissionRealityChallengeV1::fixture();
        let result = result(&challenge);
        let mut bytes = frame(&result);
        bytes.extend_from_slice(b"caller-owned-policy");
        assert!(decode_admission_device_reality_response(&bytes, &challenge).is_err());
    }
}
