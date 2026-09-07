// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical, representation-independent commitments for Forge v1 protocol types.
//!
//! Serde is intentionally not part of this protocol. These encodings use fixed
//! domain separators, explicit enum tags, big-endian integer widths, and bounded
//! length-prefixed UTF-8 fields so another repository can reproduce the exact
//! bytes without depending on Symthaea's Rust layout or serialization choices.
//!
//! The base typed Forge objects remain intentionally lightweight. This commitment
//! profile adds stricter v1 byte bounds at the cross-repository commitment seam;
//! overlong identifiers fail closed rather than being hashed or used to size an
//! allocation from untrusted input.

use super::forge::{
    ForgeArtifactIdentityV1, ForgeCapabilityScopeV1, ForgeClaimScopeV1,
    ForgeDigestAlgorithmV1, ForgeExecutionEvidenceV1, ForgeExecutionModeV1,
    ForgeExecutionProfileIdentityV1, ForgeProtocolError, ForgeVerificationRequestV1,
};
use std::error::Error;
use std::fmt;

/// Maximum encoded byte length of a Forge v1 WASM export/entry-point name.
pub const MAX_FORGE_ENTRY_POINT_BYTES_V1: usize = 256;
/// Maximum encoded byte length of a Forge v1 execution-profile identifier.
pub const MAX_FORGE_EXECUTION_PROFILE_BYTES_V1: usize = 128;

/// Domain separator for [`ForgeVerificationRequestV1`] canonical bytes.
pub const FORGE_REQUEST_COMMITMENT_DOMAIN_V1: &[u8] =
    b"symthaea.forge.verification-request.v1\0";
/// Domain separator for [`ForgeCapabilityScopeV1`] canonical bytes.
pub const FORGE_SCOPE_COMMITMENT_DOMAIN_V1: &[u8] = b"symthaea.forge.capability-scope.v1\0";
/// Domain separator for [`ForgeExecutionEvidenceV1`] canonical bytes.
pub const FORGE_EVIDENCE_COMMITMENT_DOMAIN_V1: &[u8] =
    b"symthaea.forge.execution-evidence.v1\0";

/// Failures while constructing canonical Forge v1 protocol bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgeCanonicalizationError {
    /// The underlying typed Forge object is structurally invalid.
    Protocol(ForgeProtocolError),
    /// A variable-length field exceeds the frozen v1 commitment profile.
    FieldTooLong {
        /// Stable field name used in diagnostics.
        field: &'static str,
        /// Maximum permitted UTF-8 byte length.
        max_bytes: usize,
        /// Actual UTF-8 byte length.
        actual_bytes: usize,
    },
}

impl fmt::Display for ForgeCanonicalizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Protocol(error) => write!(f, "invalid Forge protocol object: {error}"),
            Self::FieldTooLong {
                field,
                max_bytes,
                actual_bytes,
            } => write!(
                f,
                "Forge field {field} is {actual_bytes} bytes; v1 maximum is {max_bytes} bytes"
            ),
        }
    }
}

impl Error for ForgeCanonicalizationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Protocol(error) => Some(error),
            Self::FieldTooLong { .. } => None,
        }
    }
}

impl From<ForgeProtocolError> for ForgeCanonicalizationError {
    fn from(value: ForgeProtocolError) -> Self {
        Self::Protocol(value)
    }
}

fn digest_algorithm_tag(algorithm: ForgeDigestAlgorithmV1) -> u8 {
    match algorithm {
        ForgeDigestAlgorithmV1::Blake3_256 => 1,
    }
}

fn claim_scope_tag(scope: ForgeClaimScopeV1) -> u8 {
    match scope {
        ForgeClaimScopeV1::EntryPointProtocolSuccess => 1,
    }
}

fn execution_mode_tag(mode: ForgeExecutionModeV1) -> u8 {
    match mode {
        ForgeExecutionModeV1::Simulated => 1,
        ForgeExecutionModeV1::Real => 2,
    }
}

fn append_artifact_identity(out: &mut Vec<u8>, artifact: &ForgeArtifactIdentityV1) {
    out.push(digest_algorithm_tag(artifact.digest_algorithm));
    out.extend_from_slice(&artifact.digest);
    out.extend_from_slice(&artifact.byte_len.to_be_bytes());
}

fn append_bounded_utf8(
    out: &mut Vec<u8>,
    field: &'static str,
    value: &str,
    configured_max_bytes: usize,
) -> Result<(), ForgeCanonicalizationError> {
    let bytes = value.as_bytes();
    let max_bytes = configured_max_bytes.min(u16::MAX as usize);
    if bytes.len() > max_bytes {
        return Err(ForgeCanonicalizationError::FieldTooLong {
            field,
            max_bytes,
            actual_bytes: bytes.len(),
        });
    }

    let len = u16::try_from(bytes.len()).map_err(|_| ForgeCanonicalizationError::FieldTooLong {
        field,
        max_bytes,
        actual_bytes: bytes.len(),
    })?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

fn append_execution_profile_identity(
    out: &mut Vec<u8>,
    profile: &ForgeExecutionProfileIdentityV1,
) -> Result<(), ForgeCanonicalizationError> {
    append_bounded_utf8(
        out,
        "execution_profile_id",
        &profile.profile_id,
        MAX_FORGE_EXECUTION_PROFILE_BYTES_V1,
    )?;
    out.extend_from_slice(&profile.profile_commitment);
    Ok(())
}

impl ForgeVerificationRequestV1 {
    /// Return the frozen, representation-independent v1 request encoding.
    ///
    /// This is the byte sequence cross-repository mappings, signatures, receipts,
    /// and audit records should commit to. It is intentionally unrelated to serde.
    pub fn canonical_bytes_v1(&self) -> Result<Vec<u8>, ForgeCanonicalizationError> {
        self.validate()?;

        let mut out = Vec::with_capacity(
            FORGE_REQUEST_COMMITMENT_DOMAIN_V1.len()
                + 2
                + 1
                + 32
                + 8
                + 2
                + MAX_FORGE_ENTRY_POINT_BYTES_V1
                + 2
                + MAX_FORGE_EXECUTION_PROFILE_BYTES_V1
                + 32
                + 1,
        );
        out.extend_from_slice(FORGE_REQUEST_COMMITMENT_DOMAIN_V1);
        out.extend_from_slice(&self.protocol_version.to_be_bytes());
        append_artifact_identity(&mut out, &self.artifact);
        append_bounded_utf8(
            &mut out,
            "entry_point",
            &self.entry_point,
            MAX_FORGE_ENTRY_POINT_BYTES_V1,
        )?;
        append_execution_profile_identity(&mut out, &self.execution_profile)?;
        out.push(claim_scope_tag(self.claim_scope));
        Ok(out)
    }

    /// BLAKE3-256 commitment to [`Self::canonical_bytes_v1`].
    pub fn canonical_commitment_v1(&self) -> Result<[u8; 32], ForgeCanonicalizationError> {
        let bytes = self.canonical_bytes_v1()?;
        Ok(*blake3::hash(&bytes).as_bytes())
    }
}

impl ForgeCapabilityScopeV1 {
    /// Return the frozen, representation-independent v1 capability-scope encoding.
    ///
    /// The resulting commitment describes scope only; it is not an authority,
    /// signature, grant, validity certificate, or revocation statement.
    pub fn canonical_bytes_v1(&self) -> Result<Vec<u8>, ForgeCanonicalizationError> {
        self.validate()?;

        let mut out = Vec::with_capacity(
            FORGE_SCOPE_COMMITMENT_DOMAIN_V1.len()
                + 2
                + 1
                + 32
                + 8
                + 2
                + MAX_FORGE_ENTRY_POINT_BYTES_V1
                + 2
                + MAX_FORGE_EXECUTION_PROFILE_BYTES_V1
                + 32
                + 1
                + 1,
        );
        out.extend_from_slice(FORGE_SCOPE_COMMITMENT_DOMAIN_V1);
        out.extend_from_slice(&self.protocol_version.to_be_bytes());
        append_artifact_identity(&mut out, &self.artifact);
        append_bounded_utf8(
            &mut out,
            "entry_point",
            &self.entry_point,
            MAX_FORGE_ENTRY_POINT_BYTES_V1,
        )?;
        append_execution_profile_identity(&mut out, &self.execution_profile)?;
        out.push(claim_scope_tag(self.claim_scope));
        out.push(u8::from(self.allow_real_execution));
        Ok(out)
    }

    /// BLAKE3-256 commitment to [`Self::canonical_bytes_v1`].
    pub fn canonical_commitment_v1(&self) -> Result<[u8; 32], ForgeCanonicalizationError> {
        let bytes = self.canonical_bytes_v1()?;
        Ok(*blake3::hash(&bytes).as_bytes())
    }
}

impl ForgeExecutionEvidenceV1 {
    /// Return the frozen, representation-independent v1 execution-evidence encoding.
    ///
    /// Evidence commits to the exact request commitment, explicit execution mode,
    /// and the complete signed `i32` raw return. Non-canonical verifier results
    /// such as `257` remain commit-able evidence and are not narrowed or discarded.
    pub fn canonical_bytes_v1(&self) -> Result<Vec<u8>, ForgeCanonicalizationError> {
        self.validate()?;
        let request_commitment = self.request.canonical_commitment_v1()?;

        let mut out = Vec::with_capacity(FORGE_EVIDENCE_COMMITMENT_DOMAIN_V1.len() + 32 + 1 + 4);
        out.extend_from_slice(FORGE_EVIDENCE_COMMITMENT_DOMAIN_V1);
        out.extend_from_slice(&request_commitment);
        out.push(execution_mode_tag(self.mode));
        out.extend_from_slice(&self.raw_return.to_be_bytes());
        Ok(out)
    }

    /// BLAKE3-256 commitment to [`Self::canonical_bytes_v1`].
    pub fn canonical_commitment_v1(&self) -> Result<[u8; 32], ForgeCanonicalizationError> {
        let bytes = self.canonical_bytes_v1()?;
        Ok(*blake3::hash(&bytes).as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_plane::forge::FORGE_PROTOCOL_VERSION;

    fn artifact() -> ForgeArtifactIdentityV1 {
        ForgeArtifactIdentityV1::new_blake3_256([0xAB; 32], 128).unwrap()
    }

    fn profile() -> ForgeExecutionProfileIdentityV1 {
        ForgeExecutionProfileIdentityV1::new("wasmtime-fuel-v1", [0xCD; 32]).unwrap()
    }

    fn request() -> ForgeVerificationRequestV1 {
        ForgeVerificationRequestV1::new(artifact(), "verify", profile()).unwrap()
    }

    fn frozen_request_bytes() -> Vec<u8> {
        let mut expected = FORGE_REQUEST_COMMITMENT_DOMAIN_V1.to_vec();
        expected.extend_from_slice(&FORGE_PROTOCOL_VERSION.to_be_bytes());
        expected.push(1); // Blake3_256
        expected.extend_from_slice(&[0xAB; 32]);
        expected.extend_from_slice(&128u64.to_be_bytes());
        expected.extend_from_slice(&6u16.to_be_bytes());
        expected.extend_from_slice(b"verify");
        expected.extend_from_slice(&16u16.to_be_bytes());
        expected.extend_from_slice(b"wasmtime-fuel-v1");
        expected.extend_from_slice(&[0xCD; 32]);
        expected.push(1); // EntryPointProtocolSuccess
        expected
    }

    fn frozen_scope_bytes(allow_real_execution: bool) -> Vec<u8> {
        let mut expected = FORGE_SCOPE_COMMITMENT_DOMAIN_V1.to_vec();
        expected.extend_from_slice(&FORGE_PROTOCOL_VERSION.to_be_bytes());
        expected.push(1); // Blake3_256
        expected.extend_from_slice(&[0xAB; 32]);
        expected.extend_from_slice(&128u64.to_be_bytes());
        expected.extend_from_slice(&6u16.to_be_bytes());
        expected.extend_from_slice(b"verify");
        expected.extend_from_slice(&16u16.to_be_bytes());
        expected.extend_from_slice(b"wasmtime-fuel-v1");
        expected.extend_from_slice(&[0xCD; 32]);
        expected.push(1); // EntryPointProtocolSuccess
        expected.push(u8::from(allow_real_execution));
        expected
    }

    #[test]
    fn request_canonical_vector_is_frozen_and_serde_independent() {
        let req = request();
        let expected = frozen_request_bytes();
        assert_eq!(req.canonical_bytes_v1().unwrap(), expected);
        assert_eq!(
            req.canonical_commitment_v1().unwrap(),
            *blake3::hash(&frozen_request_bytes()).as_bytes()
        );
    }

    #[test]
    fn scope_canonical_vector_is_frozen_and_binds_real_execution_permission() {
        let req = request();
        let simulation_only = ForgeCapabilityScopeV1::exact_for_request(&req, false).unwrap();
        let real_allowed = ForgeCapabilityScopeV1::exact_for_request(&req, true).unwrap();

        assert_eq!(
            simulation_only.canonical_bytes_v1().unwrap(),
            frozen_scope_bytes(false)
        );
        assert_eq!(
            real_allowed.canonical_bytes_v1().unwrap(),
            frozen_scope_bytes(true)
        );
        assert_ne!(
            simulation_only.canonical_commitment_v1().unwrap(),
            real_allowed.canonical_commitment_v1().unwrap()
        );
    }

    #[test]
    fn request_commitment_changes_on_every_v1_identity_dimension() {
        let base = request();
        let base_commitment = base.canonical_commitment_v1().unwrap();

        let mut artifact_digest = base.clone();
        artifact_digest.artifact.digest[0] ^= 0xFF;
        assert_ne!(
            artifact_digest.canonical_commitment_v1().unwrap(),
            base_commitment
        );

        let mut artifact_len = base.clone();
        artifact_len.artifact.byte_len += 1;
        assert_ne!(artifact_len.canonical_commitment_v1().unwrap(), base_commitment);

        let mut entry_point = base.clone();
        entry_point.entry_point.push_str("_other");
        assert_ne!(entry_point.canonical_commitment_v1().unwrap(), base_commitment);

        let mut profile_id = base.clone();
        profile_id.execution_profile.profile_id.push_str("-other");
        assert_ne!(profile_id.canonical_commitment_v1().unwrap(), base_commitment);

        let mut profile_commitment = base.clone();
        profile_commitment.execution_profile.profile_commitment[0] ^= 0xFF;
        assert_ne!(
            profile_commitment.canonical_commitment_v1().unwrap(),
            base_commitment
        );
    }

    #[test]
    fn canonicalization_rejects_unbounded_identifiers() {
        let too_long_entry = ForgeVerificationRequestV1::new(
            artifact(),
            "x".repeat(MAX_FORGE_ENTRY_POINT_BYTES_V1 + 1),
            profile(),
        )
        .unwrap();
        assert!(matches!(
            too_long_entry.canonical_bytes_v1(),
            Err(ForgeCanonicalizationError::FieldTooLong {
                field: "entry_point",
                ..
            })
        ));

        let long_profile = ForgeExecutionProfileIdentityV1::new(
            "p".repeat(MAX_FORGE_EXECUTION_PROFILE_BYTES_V1 + 1),
            [0xCD; 32],
        )
        .unwrap();
        let too_long_profile =
            ForgeVerificationRequestV1::new(artifact(), "verify", long_profile).unwrap();
        assert!(matches!(
            too_long_profile.canonical_bytes_v1(),
            Err(ForgeCanonicalizationError::FieldTooLong {
                field: "execution_profile_id",
                ..
            })
        ));
    }

    #[test]
    fn canonicalization_rechecks_protocol_validity() {
        let mut req = request();
        req.protocol_version += 1;
        assert!(matches!(
            req.canonical_bytes_v1(),
            Err(ForgeCanonicalizationError::Protocol(
                ForgeProtocolError::UnsupportedProtocolVersion(_)
            ))
        ));
    }

    #[test]
    fn canonicalization_rechecks_profile_identity() {
        let mut req = request();
        req.execution_profile.profile_commitment = [0u8; 32];
        assert!(matches!(
            req.canonical_bytes_v1(),
            Err(ForgeCanonicalizationError::Protocol(
                ForgeProtocolError::ZeroExecutionProfileCommitment
            ))
        ));
    }

    #[test]
    fn evidence_commitment_binds_request_mode_and_exact_signed_raw_return() {
        let req = request();
        let simulated =
            ForgeExecutionEvidenceV1::new(req.clone(), ForgeExecutionModeV1::Simulated, 1)
                .unwrap();
        let real = ForgeExecutionEvidenceV1::new(req.clone(), ForgeExecutionModeV1::Real, 1).unwrap();
        let invalid =
            ForgeExecutionEvidenceV1::new(req.clone(), ForgeExecutionModeV1::Real, 257).unwrap();
        let negative = ForgeExecutionEvidenceV1::new(req, ForgeExecutionModeV1::Real, -255).unwrap();

        assert_ne!(
            simulated.canonical_commitment_v1().unwrap(),
            real.canonical_commitment_v1().unwrap()
        );
        assert_ne!(
            real.canonical_commitment_v1().unwrap(),
            invalid.canonical_commitment_v1().unwrap()
        );
        assert_ne!(
            invalid.canonical_commitment_v1().unwrap(),
            negative.canonical_commitment_v1().unwrap()
        );

        // Invalid protocol returns remain evidence and are committed exactly;
        // interpretation still fails closed in the typed Forge protocol.
        assert!(invalid.protocol_status().is_err());
        assert!(negative.protocol_status().is_err());
    }

    #[test]
    fn evidence_commitment_changes_when_request_identity_changes() {
        let base = ForgeExecutionEvidenceV1::new(request(), ForgeExecutionModeV1::Real, 1).unwrap();
        let mut changed_request = request();
        changed_request.execution_profile.profile_commitment[31] ^= 0x01;
        let changed =
            ForgeExecutionEvidenceV1::new(changed_request, ForgeExecutionModeV1::Real, 1).unwrap();

        assert_ne!(
            base.canonical_commitment_v1().unwrap(),
            changed.canonical_commitment_v1().unwrap()
        );
    }
}
