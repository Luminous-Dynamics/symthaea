// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strict, cryptography-free metadata checks for OpenSSH SSHSIG admission
//! signatures.
//!
//! OpenSSH remains the cryptographic verifier. This module only freezes the
//! exact v1 envelope semantics that EUREKA is willing to ask OpenSSH to verify:
//! SSHSIG version 1, exact application namespace, empty reserved field,
//! SHA-512 prehash, and a signature/public-key algorithm matching the exact
//! policy-bound public key material.

#![allow(dead_code)]

pub(super) const V2_ADMISSION_SSHSIG_HASH_ALGORITHM: &str = "sha512";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2SshSigMetadataError {
    InvalidArmor,
    InvalidBase64,
    InvalidWireFormat,
    WrongMagic,
    WrongVersion,
    PublicKeyMismatch,
    KeyTypeMismatch,
    WrongNamespace,
    ReservedFieldNotEmpty,
    WrongHashAlgorithm,
    SignatureAlgorithmMismatch,
    EmptySignature,
}

pub(super) fn validate_admission_sshsig_v1(
    armored_signature: &[u8],
    canonical_public_key_material: &str,
    expected_key_type: &str,
    expected_namespace: &str,
) -> Result<(), V2SshSigMetadataError> {
    let signature_blob = decode_armored_signature(armored_signature)?;
    let expected_public_key = decode_public_key_material(canonical_public_key_material)?;

    let mut outer = SshCursor::new(&signature_blob);
    if outer.take_exact(6)? != b"SSHSIG" {
        return Err(V2SshSigMetadataError::WrongMagic);
    }
    if outer.read_u32()? != 1 {
        return Err(V2SshSigMetadataError::WrongVersion);
    }

    let embedded_public_key = outer.read_string()?;
    if embedded_public_key != expected_public_key.as_slice() {
        return Err(V2SshSigMetadataError::PublicKeyMismatch);
    }
    let mut public_key = SshCursor::new(embedded_public_key);
    if public_key.read_string()? != expected_key_type.as_bytes() {
        return Err(V2SshSigMetadataError::KeyTypeMismatch);
    }

    if outer.read_string()? != expected_namespace.as_bytes() {
        return Err(V2SshSigMetadataError::WrongNamespace);
    }
    if !outer.read_string()?.is_empty() {
        return Err(V2SshSigMetadataError::ReservedFieldNotEmpty);
    }
    if outer.read_string()? != V2_ADMISSION_SSHSIG_HASH_ALGORITHM.as_bytes() {
        return Err(V2SshSigMetadataError::WrongHashAlgorithm);
    }

    let signature = outer.read_string()?;
    outer.finish()?;

    let mut signature = SshCursor::new(signature);
    if signature.read_string()? != expected_key_type.as_bytes() {
        return Err(V2SshSigMetadataError::SignatureAlgorithmMismatch);
    }
    if signature.read_string()?.is_empty() {
        return Err(V2SshSigMetadataError::EmptySignature);
    }
    signature.finish()?;
    Ok(())
}

fn decode_armored_signature(input: &[u8]) -> Result<Vec<u8>, V2SshSigMetadataError> {
    let text = std::str::from_utf8(input).map_err(|_| V2SshSigMetadataError::InvalidArmor)?;
    if text.contains('\r') {
        return Err(V2SshSigMetadataError::InvalidArmor);
    }

    let mut lines: Vec<&str> = text.split('\n').collect();
    if lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }
    if lines.len() < 3
        || lines.first() != Some(&"-----BEGIN SSH SIGNATURE-----")
        || lines.last() != Some(&"-----END SSH SIGNATURE-----")
    {
        return Err(V2SshSigMetadataError::InvalidArmor);
    }

    let mut encoded = String::new();
    for line in &lines[1..lines.len() - 1] {
        if line.is_empty()
            || !line.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/' | b'=')
            })
        {
            return Err(V2SshSigMetadataError::InvalidArmor);
        }
        encoded.push_str(line);
    }
    decode_base64(encoded.as_bytes())
}

fn decode_public_key_material(input: &str) -> Result<Vec<u8>, V2SshSigMetadataError> {
    let mut fields = input.split_ascii_whitespace();
    let key_type = fields.next().ok_or(V2SshSigMetadataError::InvalidWireFormat)?;
    let encoded = fields.next().ok_or(V2SshSigMetadataError::InvalidWireFormat)?;
    if fields.next().is_some() || key_type.is_empty() || encoded.is_empty() {
        return Err(V2SshSigMetadataError::InvalidWireFormat);
    }
    decode_base64(encoded.as_bytes())
}

fn decode_base64(input: &[u8]) -> Result<Vec<u8>, V2SshSigMetadataError> {
    if input.is_empty() || input.len() % 4 != 0 {
        return Err(V2SshSigMetadataError::InvalidBase64);
    }

    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    let chunk_count = input.len() / 4;
    for (chunk_index, chunk) in input.chunks_exact(4).enumerate() {
        let is_last = chunk_index + 1 == chunk_count;
        let a = base64_value(chunk[0])?;
        let b = base64_value(chunk[1])?;
        let c_pad = chunk[2] == b'=';
        let d_pad = chunk[3] == b'=';

        if c_pad {
            if !d_pad || !is_last || (b & 0x0f) != 0 {
                return Err(V2SshSigMetadataError::InvalidBase64);
            }
            out.push((a << 2) | (b >> 4));
            continue;
        }

        let c = base64_value(chunk[2])?;
        out.push((a << 2) | (b >> 4));
        out.push((b << 4) | (c >> 2));

        if d_pad {
            if !is_last || (c & 0x03) != 0 {
                return Err(V2SshSigMetadataError::InvalidBase64);
            }
            continue;
        }

        let d = base64_value(chunk[3])?;
        out.push((c << 6) | d);
    }
    Ok(out)
}

fn base64_value(byte: u8) -> Result<u8, V2SshSigMetadataError> {
    match byte {
        b'A'..=b'Z' => Ok(byte - b'A'),
        b'a'..=b'z' => Ok(byte - b'a' + 26),
        b'0'..=b'9' => Ok(byte - b'0' + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        _ => Err(V2SshSigMetadataError::InvalidBase64),
    }
}

struct SshCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> SshCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_u32(&mut self) -> Result<u32, V2SshSigMetadataError> {
        let bytes = self.take_exact(4)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_string(&mut self) -> Result<&'a [u8], V2SshSigMetadataError> {
        let length = self.read_u32()? as usize;
        self.take_exact(length)
    }

    fn take_exact(&mut self, length: usize) -> Result<&'a [u8], V2SshSigMetadataError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(V2SshSigMetadataError::InvalidWireFormat)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(V2SshSigMetadataError::InvalidWireFormat)?;
        self.offset = end;
        Ok(value)
    }

    fn finish(&self) -> Result<(), V2SshSigMetadataError> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(V2SshSigMetadataError::InvalidWireFormat)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::v2_qualifier_signer_policy::{
        V2_QUALIFIER_ADMISSION_SSH_NAMESPACE, V2_QUALIFIER_SIGNER_POLICY_SSH_NAMESPACE,
    };

    const ALPHA_PUB: &str =
        include_str!("fixtures/eureka-v2-signed-admission-alpha.pub");
    const ALPHA_SIG: &[u8] =
        include_bytes!("fixtures/eureka-v2-signed-admission-alpha.sig");
    const ALPHA_WRONG_NAMESPACE_SIG: &[u8] = include_bytes!(
        "fixtures/eureka-v2-signed-admission-alpha-wrong-namespace.sig"
    );

    fn canonical_key() -> String {
        ALPHA_PUB
            .split_ascii_whitespace()
            .take(2)
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn exact_fixture_metadata_is_admitted() {
        validate_admission_sshsig_v1(
            ALPHA_SIG,
            &canonical_key(),
            "ssh-ed25519",
            V2_QUALIFIER_ADMISSION_SSH_NAMESPACE,
        )
        .unwrap();
    }

    #[test]
    fn cross_namespace_fixture_fails_before_crypto_verification() {
        assert_eq!(
            validate_admission_sshsig_v1(
                ALPHA_WRONG_NAMESPACE_SIG,
                &canonical_key(),
                "ssh-ed25519",
                V2_QUALIFIER_ADMISSION_SSH_NAMESPACE,
            )
            .unwrap_err(),
            V2SshSigMetadataError::WrongNamespace
        );
        validate_admission_sshsig_v1(
            ALPHA_WRONG_NAMESPACE_SIG,
            &canonical_key(),
            "ssh-ed25519",
            V2_QUALIFIER_SIGNER_POLICY_SSH_NAMESPACE,
        )
        .unwrap();
    }

    #[test]
    fn wrong_key_type_or_public_key_is_rejected() {
        assert_eq!(
            validate_admission_sshsig_v1(
                ALPHA_SIG,
                &canonical_key(),
                "sk-ssh-ed25519@openssh.com",
                V2_QUALIFIER_ADMISSION_SSH_NAMESPACE,
            )
            .unwrap_err(),
            V2SshSigMetadataError::KeyTypeMismatch
        );

        let beta = include_str!("fixtures/eureka-v2-signed-admission-beta.pub")
            .split_ascii_whitespace()
            .take(2)
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(
            validate_admission_sshsig_v1(
                ALPHA_SIG,
                &beta,
                "ssh-ed25519",
                V2_QUALIFIER_ADMISSION_SSH_NAMESPACE,
            )
            .unwrap_err(),
            V2SshSigMetadataError::PublicKeyMismatch
        );
    }

    #[test]
    fn armor_and_canonical_base64_fail_closed() {
        let mut broken_header = ALPHA_SIG.to_vec();
        broken_header[0] = b'!';
        assert_eq!(
            validate_admission_sshsig_v1(
                &broken_header,
                &canonical_key(),
                "ssh-ed25519",
                V2_QUALIFIER_ADMISSION_SSH_NAMESPACE,
            )
            .unwrap_err(),
            V2SshSigMetadataError::InvalidArmor
        );

        let mut bad_pad_bits = ALPHA_SIG.to_vec();
        let pad = bad_pad_bits
            .windows(4)
            .rposition(|window| window == b"Dw==")
            .expect("fixture ends with canonical two-byte padding quantum");
        bad_pad_bits[pad + 1] = b'x';
        assert_eq!(
            validate_admission_sshsig_v1(
                &bad_pad_bits,
                &canonical_key(),
                "ssh-ed25519",
                V2_QUALIFIER_ADMISSION_SSH_NAMESPACE,
            )
            .unwrap_err(),
            V2SshSigMetadataError::InvalidBase64
        );
    }

    #[test]
    fn metadata_source_contains_no_signature_crypto_or_execution_authority() {
        let production = include_str!("v2_openssh_sshsig_metadata.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "ssh-keygen",
            "ed25519_dalek",
            "PRIVATE KEY",
            "V2AdmittedQualifierRoot",
            "V2CanaryAuthorization",
            "predict_ticket",
            "score_consequence",
        ] {
            assert!(!production.contains(forbidden), "forbidden metadata surface: {forbidden}");
        }
        assert!(production.contains("sha512"));
        assert!(production.contains("SSHSIG"));
    }
}
