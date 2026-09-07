// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed, non-authoritative Forge verification protocol.
//!
//! These types preserve the identity of a WASM verification request and the
//! exact result observed from an execution. They deliberately do **not** grant
//! Forge authority, prove artifact provenance, establish sandbox containment,
//! or authorize promotion/hot-loading.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// Current version of the typed Forge request/evidence schema.
pub const FORGE_PROTOCOL_VERSION: u16 = 1;

/// Digest algorithm used to identify the exact candidate artifact bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForgeDigestAlgorithmV1 {
    /// BLAKE3 with a 256-bit digest.
    Blake3_256,
}

/// Content identity for one candidate artifact.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForgeArtifactIdentityV1 {
    /// Digest algorithm used for `digest`.
    pub digest_algorithm: ForgeDigestAlgorithmV1,
    /// Digest of the exact artifact bytes.
    pub digest: [u8; 32],
    /// Exact artifact length in bytes.
    pub byte_len: u64,
}

impl ForgeArtifactIdentityV1 {
    /// Construct a BLAKE3-256 artifact identity.
    pub fn new_blake3_256(digest: [u8; 32], byte_len: u64) -> Result<Self, ForgeProtocolError> {
        if byte_len == 0 {
            return Err(ForgeProtocolError::EmptyArtifact);
        }
        Ok(Self {
            digest_algorithm: ForgeDigestAlgorithmV1::Blake3_256,
            digest,
            byte_len,
        })
    }

    /// Validate structural invariants for an artifact identity.
    pub fn validate(&self) -> Result<(), ForgeProtocolError> {
        if self.byte_len == 0 {
            return Err(ForgeProtocolError::EmptyArtifact);
        }
        Ok(())
    }
}

/// Exact identity of the execution profile requested for Forge verification.
///
/// `profile_id` is the human/operational identifier. `profile_commitment` binds
/// the exact separately versioned runtime/resource/containment semantics behind
/// that identifier. Reusing a profile name with changed semantics must therefore
/// produce a different identity.
///
/// This object names requested execution semantics; it does not prove that the
/// profile is available, current, trusted, or authorized for the caller.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForgeExecutionProfileIdentityV1 {
    /// Human/operational execution-profile identifier.
    pub profile_id: String,
    /// Exact commitment to the selected profile definition.
    pub profile_commitment: [u8; 32],
}

impl ForgeExecutionProfileIdentityV1 {
    /// Construct a profile identity from an explicit identifier and commitment.
    pub fn new(
        profile_id: impl Into<String>,
        profile_commitment: [u8; 32],
    ) -> Result<Self, ForgeProtocolError> {
        let profile = Self {
            profile_id: profile_id.into(),
            profile_commitment,
        };
        profile.validate()?;
        Ok(profile)
    }

    /// Validate structural profile-identity invariants.
    pub fn validate(&self) -> Result<(), ForgeProtocolError> {
        if self.profile_id.trim().is_empty() {
            return Err(ForgeProtocolError::EmptyExecutionProfile);
        }
        if self.profile_commitment == [0u8; 32] {
            return Err(ForgeProtocolError::ZeroExecutionProfileCommitment);
        }
        Ok(())
    }
}

/// The only claim this first protocol version knows how to support.
///
/// A successful result means only that the requested entry point returned the
/// canonical protocol success value under the recorded execution mode/profile.
/// It does not imply semantic correctness of the artifact or authorize promotion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForgeClaimScopeV1 {
    /// The requested entry point returned the canonical success marker.
    EntryPointProtocolSuccess,
}

/// Non-authoritative request to verify one exact candidate artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForgeVerificationRequestV1 {
    /// Schema version. Must equal [`FORGE_PROTOCOL_VERSION`].
    pub protocol_version: u16,
    /// Exact artifact content identity.
    pub artifact: ForgeArtifactIdentityV1,
    /// Exact exported entry point to invoke.
    pub entry_point: String,
    /// Exact requested execution-profile identity.
    ///
    /// This binds both the profile name and its semantics; it does not prove the
    /// caller is authorized to use that profile.
    pub execution_profile: ForgeExecutionProfileIdentityV1,
    /// Exact proposition the resulting evidence may support.
    pub claim_scope: ForgeClaimScopeV1,
}

impl ForgeVerificationRequestV1 {
    /// Construct a structurally valid verification request.
    pub fn new(
        artifact: ForgeArtifactIdentityV1,
        entry_point: impl Into<String>,
        execution_profile: ForgeExecutionProfileIdentityV1,
    ) -> Result<Self, ForgeProtocolError> {
        let request = Self {
            protocol_version: FORGE_PROTOCOL_VERSION,
            artifact,
            entry_point: entry_point.into(),
            execution_profile,
            claim_scope: ForgeClaimScopeV1::EntryPointProtocolSuccess,
        };
        request.validate()?;
        Ok(request)
    }

    /// Validate structural request invariants.
    ///
    /// Call this after deserializing untrusted wire data before using the request.
    pub fn validate(&self) -> Result<(), ForgeProtocolError> {
        if self.protocol_version != FORGE_PROTOCOL_VERSION {
            return Err(ForgeProtocolError::UnsupportedProtocolVersion(
                self.protocol_version,
            ));
        }
        self.artifact.validate()?;
        if self.entry_point.trim().is_empty() {
            return Err(ForgeProtocolError::EmptyEntryPoint);
        }
        self.execution_profile.validate()?;
        Ok(())
    }
}

/// Exact resource/profile scope that a separate Forge authority may carry.
///
/// This type is deliberately **not** an authority token. It has no issuer,
/// signature, generation, expiry, or revocation semantics. A current authenticated
/// policy/grant must carry this scope before it can authorize an effect.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForgeCapabilityScopeV1 {
    /// Schema version. Must equal [`FORGE_PROTOCOL_VERSION`].
    pub protocol_version: u16,
    /// Exact artifact identity permitted by the scope.
    pub artifact: ForgeArtifactIdentityV1,
    /// Exact entry point permitted by the scope.
    pub entry_point: String,
    /// Exact execution-profile identity permitted by the scope.
    pub execution_profile: ForgeExecutionProfileIdentityV1,
    /// Exact claim scope permitted by the scope.
    pub claim_scope: ForgeClaimScopeV1,
    /// Whether a real execution may be requested. Simulation remains non-effectful.
    pub allow_real_execution: bool,
}

impl ForgeCapabilityScopeV1 {
    /// Build the least-privilege scope for one exact request.
    pub fn exact_for_request(
        request: &ForgeVerificationRequestV1,
        allow_real_execution: bool,
    ) -> Result<Self, ForgeProtocolError> {
        request.validate()?;
        Ok(Self {
            protocol_version: FORGE_PROTOCOL_VERSION,
            artifact: request.artifact.clone(),
            entry_point: request.entry_point.clone(),
            execution_profile: request.execution_profile.clone(),
            claim_scope: request.claim_scope,
            allow_real_execution,
        })
    }

    /// Validate structural scope invariants.
    pub fn validate(&self) -> Result<(), ForgeProtocolError> {
        if self.protocol_version != FORGE_PROTOCOL_VERSION {
            return Err(ForgeProtocolError::UnsupportedProtocolVersion(
                self.protocol_version,
            ));
        }
        self.artifact.validate()?;
        if self.entry_point.trim().is_empty() {
            return Err(ForgeProtocolError::EmptyEntryPoint);
        }
        self.execution_profile.validate()?;
        Ok(())
    }

    /// Whether this resource scope exactly matches a request and execution mode.
    ///
    /// `true` means only that the scope matches. It does not prove that a current
    /// authenticated authority actually granted or still authorizes this scope.
    pub fn matches_request(
        &self,
        request: &ForgeVerificationRequestV1,
        mode: ForgeExecutionModeV1,
    ) -> Result<bool, ForgeProtocolError> {
        self.validate()?;
        request.validate()?;

        if mode == ForgeExecutionModeV1::Real && !self.allow_real_execution {
            return Ok(false);
        }

        Ok(self.artifact == request.artifact
            && self.entry_point == request.entry_point
            && self.execution_profile == request.execution_profile
            && self.claim_scope == request.claim_scope)
    }
}

/// Whether the recorded operation was simulated or actually executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForgeExecutionModeV1 {
    /// No real module execution occurred.
    Simulated,
    /// A real execution was observed under the exact requested execution profile.
    Real,
}

/// Validated interpretation of the v1 verifier return protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForgeProtocolStatusV1 {
    /// Canonical return value `0`.
    Failure,
    /// Canonical return value `1`.
    Success,
}

impl ForgeProtocolStatusV1 {
    /// Interpret an exact `i32` verifier return without lossy narrowing.
    pub fn from_raw_return(raw_return: i32) -> Result<Self, ForgeProtocolError> {
        match raw_return {
            0 => Ok(Self::Failure),
            1 => Ok(Self::Success),
            other => Err(ForgeProtocolError::InvalidProtocolReturn(other)),
        }
    }
}

/// Evidence produced by one execution attempt for one exact request.
///
/// The request is embedded directly rather than referenced by an ad-hoc hash so
/// evidence cannot silently drift to a different artifact, entry point, profile,
/// profile definition, or claim scope before canonical request commitment is
/// standardized.
///
/// `raw_return` is always preserved exactly, including non-canonical values. An
/// invalid verifier result is still valuable evidence; protocol interpretation is
/// intentionally a separate fallible step via [`Self::protocol_status`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForgeExecutionEvidenceV1 {
    /// The exact request this execution answers.
    pub request: ForgeVerificationRequestV1,
    /// Whether execution was simulated or real.
    pub mode: ForgeExecutionModeV1,
    /// Exact raw `i32` returned by the verifier entry point.
    pub raw_return: i32,
}

impl ForgeExecutionEvidenceV1 {
    /// Record one exact execution observation.
    ///
    /// The request must be structurally valid, but `raw_return` is not filtered:
    /// anomalous/non-canonical return values must remain observable evidence.
    pub fn new(
        request: ForgeVerificationRequestV1,
        mode: ForgeExecutionModeV1,
        raw_return: i32,
    ) -> Result<Self, ForgeProtocolError> {
        request.validate()?;
        Ok(Self {
            request,
            mode,
            raw_return,
        })
    }

    /// Validate the embedded request after deserializing untrusted evidence.
    pub fn validate(&self) -> Result<(), ForgeProtocolError> {
        self.request.validate()
    }

    /// Interpret the exact raw verifier return according to the v1 protocol.
    pub fn protocol_status(&self) -> Result<ForgeProtocolStatusV1, ForgeProtocolError> {
        ForgeProtocolStatusV1::from_raw_return(self.raw_return)
    }

    /// Whether this evidence observed canonical success during real execution.
    ///
    /// This remains only an execution observation. Authority, containment,
    /// provenance, and promotion policy must be proven separately.
    pub fn is_real_protocol_success(&self) -> bool {
        self.mode == ForgeExecutionModeV1::Real
            && matches!(self.protocol_status(), Ok(ForgeProtocolStatusV1::Success))
    }
}

/// Structural/protocol validation failures for Forge v1 requests and results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgeProtocolError {
    /// Artifact length was zero.
    EmptyArtifact,
    /// Requested entry point was empty or whitespace-only.
    EmptyEntryPoint,
    /// Requested execution-profile identifier was empty or whitespace-only.
    EmptyExecutionProfile,
    /// Requested execution-profile commitment was the all-zero placeholder.
    ZeroExecutionProfileCommitment,
    /// Wire/request schema version is unsupported.
    UnsupportedProtocolVersion(u16),
    /// Verifier returned a non-canonical value other than `0` or `1`.
    InvalidProtocolReturn(i32),
}

impl fmt::Display for ForgeProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyArtifact => write!(f, "Forge artifact must contain at least one byte"),
            Self::EmptyEntryPoint => write!(f, "Forge entry point must be explicit and non-empty"),
            Self::EmptyExecutionProfile => {
                write!(f, "Forge execution profile must be explicit and non-empty")
            }
            Self::ZeroExecutionProfileCommitment => {
                write!(f, "Forge execution profile commitment must be explicit and non-zero")
            }
            Self::UnsupportedProtocolVersion(version) => {
                write!(f, "unsupported Forge protocol version {version}")
            }
            Self::InvalidProtocolReturn(value) => {
                write!(f, "invalid Forge verifier return {value}; expected exactly 0 or 1")
            }
        }
    }
}

impl Error for ForgeProtocolError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact() -> ForgeArtifactIdentityV1 {
        ForgeArtifactIdentityV1::new_blake3_256([0xAB; 32], 128).unwrap()
    }

    fn profile() -> ForgeExecutionProfileIdentityV1 {
        ForgeExecutionProfileIdentityV1::new("wasmtime-fuel-v1", [0xCD; 32]).unwrap()
    }

    fn request() -> ForgeVerificationRequestV1 {
        ForgeVerificationRequestV1::new(artifact(), "verify", profile()).unwrap()
    }

    #[test]
    fn artifact_identity_rejects_zero_length() {
        assert!(matches!(
            ForgeArtifactIdentityV1::new_blake3_256([0xAB; 32], 0),
            Err(ForgeProtocolError::EmptyArtifact)
        ));
    }

    #[test]
    fn execution_profile_requires_explicit_id_and_commitment() {
        assert!(matches!(
            ForgeExecutionProfileIdentityV1::new("   ", [0xCD; 32]),
            Err(ForgeProtocolError::EmptyExecutionProfile)
        ));
        assert!(matches!(
            ForgeExecutionProfileIdentityV1::new("wasmtime-fuel-v1", [0u8; 32]),
            Err(ForgeProtocolError::ZeroExecutionProfileCommitment)
        ));
    }

    #[test]
    fn request_requires_explicit_entry_point() {
        assert!(matches!(
            ForgeVerificationRequestV1::new(artifact(), "   ", profile()),
            Err(ForgeProtocolError::EmptyEntryPoint)
        ));
    }

    #[test]
    fn request_rejects_invalid_profile_after_deserialization() {
        let mut req = request();
        req.execution_profile.profile_commitment = [0u8; 32];
        assert!(matches!(
            req.validate(),
            Err(ForgeProtocolError::ZeroExecutionProfileCommitment)
        ));
    }

    #[test]
    fn request_rejects_wrong_protocol_version_after_deserialization() {
        let mut req = request();
        req.protocol_version = FORGE_PROTOCOL_VERSION + 1;
        assert!(matches!(
            req.validate(),
            Err(ForgeProtocolError::UnsupportedProtocolVersion(_))
        ));
    }

    #[test]
    fn exact_capability_scope_matches_only_the_bound_request() {
        let req = request();
        let scope = ForgeCapabilityScopeV1::exact_for_request(&req, true).unwrap();

        assert!(
            scope
                .matches_request(&req, ForgeExecutionModeV1::Simulated)
                .unwrap()
        );
        assert!(
            scope
                .matches_request(&req, ForgeExecutionModeV1::Real)
                .unwrap()
        );

        let mut different_artifact = req.clone();
        different_artifact.artifact.digest[0] ^= 0xFF;
        assert!(
            !scope
                .matches_request(&different_artifact, ForgeExecutionModeV1::Real)
                .unwrap()
        );

        let mut different_entry = req.clone();
        different_entry.entry_point = "verify_other".into();
        assert!(
            !scope
                .matches_request(&different_entry, ForgeExecutionModeV1::Real)
                .unwrap()
        );

        let mut different_profile_id = req.clone();
        different_profile_id.execution_profile.profile_id = "different-profile".into();
        assert!(
            !scope
                .matches_request(&different_profile_id, ForgeExecutionModeV1::Real)
                .unwrap()
        );

        let mut different_profile_commitment = req.clone();
        different_profile_commitment.execution_profile.profile_commitment[0] ^= 0xFF;
        assert!(
            !scope
                .matches_request(&different_profile_commitment, ForgeExecutionModeV1::Real)
                .unwrap()
        );
    }

    #[test]
    fn simulation_only_scope_never_matches_real_execution() {
        let req = request();
        let scope = ForgeCapabilityScopeV1::exact_for_request(&req, false).unwrap();

        assert!(
            scope
                .matches_request(&req, ForgeExecutionModeV1::Simulated)
                .unwrap()
        );
        assert!(
            !scope
                .matches_request(&req, ForgeExecutionModeV1::Real)
                .unwrap()
        );
    }

    #[test]
    fn result_protocol_accepts_only_exact_zero_or_one() {
        assert_eq!(
            ForgeProtocolStatusV1::from_raw_return(0).unwrap(),
            ForgeProtocolStatusV1::Failure
        );
        assert_eq!(
            ForgeProtocolStatusV1::from_raw_return(1).unwrap(),
            ForgeProtocolStatusV1::Success
        );

        for raw in [257, -255, -1, i32::MAX] {
            assert!(matches!(
                ForgeProtocolStatusV1::from_raw_return(raw),
                Err(ForgeProtocolError::InvalidProtocolReturn(value)) if value == raw
            ));
        }
    }

    #[test]
    fn evidence_preserves_exact_request_identity() {
        let req = request();
        let evidence = ForgeExecutionEvidenceV1::new(req.clone(), ForgeExecutionModeV1::Real, 1)
            .unwrap();

        assert_eq!(evidence.request, req);
        assert_eq!(evidence.raw_return, 1);
        assert_eq!(
            evidence.protocol_status().unwrap(),
            ForgeProtocolStatusV1::Success
        );
    }

    #[test]
    fn invalid_protocol_return_is_preserved_as_evidence() {
        for raw in [257, -255, -1, i32::MAX] {
            let evidence = ForgeExecutionEvidenceV1::new(
                request(),
                ForgeExecutionModeV1::Real,
                raw,
            )
            .unwrap();
            assert_eq!(evidence.raw_return, raw);
            assert!(matches!(
                evidence.protocol_status(),
                Err(ForgeProtocolError::InvalidProtocolReturn(value)) if value == raw
            ));
            assert!(!evidence.is_real_protocol_success());
        }
    }

    #[test]
    fn simulated_success_never_counts_as_real_protocol_success() {
        let simulated = ForgeExecutionEvidenceV1::new(
            request(),
            ForgeExecutionModeV1::Simulated,
            1,
        )
        .unwrap();
        assert!(!simulated.is_real_protocol_success());

        let real = ForgeExecutionEvidenceV1::new(request(), ForgeExecutionModeV1::Real, 1).unwrap();
        assert!(real.is_real_protocol_success());
    }

    #[test]
    fn canonical_failure_never_counts_as_success() {
        for mode in [ForgeExecutionModeV1::Simulated, ForgeExecutionModeV1::Real] {
            let evidence = ForgeExecutionEvidenceV1::new(request(), mode, 0).unwrap();
            assert_eq!(
                evidence.protocol_status().unwrap(),
                ForgeProtocolStatusV1::Failure
            );
            assert!(!evidence.is_real_protocol_success());
        }
    }
}
