// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Approver-evidence boundaries for local and external approval profiles.
//!
//! This module deliberately preserves claim strength:
//!
//! - serialized local peer credentials are audit evidence only;
//! - a verified local-peer value is non-Serde/non-Clone and may only be minted
//!   inside this crate by a kernel-backed transport adapter;
//! - Xenia identity/permit verification remains owned by Xenia; Nixward stores
//!   only an externally verified evidence reference until a qualified bridge exists.
//!
//! None of these values are themselves execution authorization.

use super::temporal::UnixMillisV1;
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const LOCAL_PEER_EVIDENCE_DOMAIN: &[u8] = b"nixward-local-unix-peer-credential-evidence-v1";
const MAX_REF_BYTES: usize = 1024;

/// Profile identifying what kind of approver evidence a reference names.
///
/// This enum does not equal assurance. Policy must decide which profile is
/// sufficient for a particular operation class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApproverEvidenceProfileV1 {
    /// Kernel-observed local Unix peer credentials from a qualified local IPC adapter.
    LocalUnixPeerCredentialV1,
    /// Provider-verified Xenia principal/authentication evidence owned by Xenia.
    XeniaAuthenticatedPrincipalV1,
    /// Exact state-bound Xenia permit evidence owned by Xenia.
    XeniaStateBoundPermitV1,
}

/// Portable reference to approver evidence produced by the profile owner.
///
/// A reference is provenance metadata. Deserializing it does not recreate a
/// verified local peer, a Xenia principal, or execution authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApproverEvidenceRefV1 {
    pub profile: ApproverEvidenceProfileV1,
    pub evidence_digest: String,
}

impl ApproverEvidenceRefV1 {
    pub fn validate_shape(&self) -> Result<(), ApproverEvidenceErrorV1> {
        validate_digest(&self.evidence_digest, "approver evidence digest")
    }
}

/// Serializable audit evidence describing one kernel peer-credential observation.
///
/// Construction/deserialization alone does **not** establish that these values
/// actually came from the kernel. The non-serializable positive type below owns
/// that distinction at the Nixward boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalUnixPeerCredentialEvidenceV1 {
    pub effective_uid: u32,
    pub effective_gid: u32,
    pub process_id: Option<u32>,
    pub transport_instance_ref: String,
    pub observed_at_unix_ms: u64,
}

impl LocalUnixPeerCredentialEvidenceV1 {
    pub fn new_unverified(
        effective_uid: u32,
        effective_gid: u32,
        process_id: Option<u32>,
        transport_instance_ref: impl Into<String>,
        observed_at: UnixMillisV1,
    ) -> Result<Self, ApproverEvidenceErrorV1> {
        let evidence = Self {
            effective_uid,
            effective_gid,
            process_id,
            transport_instance_ref: transport_instance_ref.into(),
            observed_at_unix_ms: observed_at.as_u64(),
        };
        evidence.validate_shape()?;
        Ok(evidence)
    }

    pub fn validate_shape(&self) -> Result<(), ApproverEvidenceErrorV1> {
        validate_ref(&self.transport_instance_ref, "transport instance ref")?;
        if self.process_id == Some(0) {
            return Err(ApproverEvidenceErrorV1::InvalidProcessId);
        }
        Ok(())
    }

    /// Deterministic audit identity independent of serde/JSON formatting.
    pub fn digest(&self) -> Result<String, ApproverEvidenceErrorV1> {
        self.validate_shape()?;
        let mut h = Hasher::new();
        h.update(LOCAL_PEER_EVIDENCE_DOMAIN);
        put_u32(&mut h, self.effective_uid);
        put_u32(&mut h, self.effective_gid);
        match self.process_id {
            Some(pid) => {
                put_u8(&mut h, 1);
                put_u32(&mut h, pid);
            }
            None => put_u8(&mut h, 0),
        }
        put_str(&mut h, &self.transport_instance_ref);
        put_u64(&mut h, self.observed_at_unix_ms);
        Ok(h.finalize().to_hex().to_string())
    }

    pub fn evidence_ref(&self) -> Result<ApproverEvidenceRefV1, ApproverEvidenceErrorV1> {
        Ok(ApproverEvidenceRefV1 {
            profile: ApproverEvidenceProfileV1::LocalUnixPeerCredentialV1,
            evidence_digest: self.digest()?,
        })
    }
}

/// Live positive result meaning a Nixward-owned transport adapter obtained the
/// enclosed credentials from its kernel peer-credential API.
///
/// This type intentionally does not implement `Serialize`, `Deserialize`,
/// `Clone`, or `Copy`. There is no public constructor from serialized evidence.
pub struct VerifiedLocalUnixPeerCredentialV1 {
    evidence: LocalUnixPeerCredentialEvidenceV1,
    evidence_ref: ApproverEvidenceRefV1,
}

impl VerifiedLocalUnixPeerCredentialV1 {
    /// Crate-private construction seam for the future Unix-domain-socket adapter.
    ///
    /// The adapter is responsible for obtaining these values from the kernel on
    /// the accepted connection. Request JSON, environment variables, `$USER`,
    /// file metadata, or caller-supplied UID fields are not valid inputs to that
    /// adapter theorem.
    pub(crate) fn from_kernel_peer_observation(
        effective_uid: u32,
        effective_gid: u32,
        process_id: Option<u32>,
        transport_instance_ref: impl Into<String>,
        observed_at: UnixMillisV1,
    ) -> Result<Self, ApproverEvidenceErrorV1> {
        let evidence = LocalUnixPeerCredentialEvidenceV1::new_unverified(
            effective_uid,
            effective_gid,
            process_id,
            transport_instance_ref,
            observed_at,
        )?;
        let evidence_ref = evidence.evidence_ref()?;
        Ok(Self {
            evidence,
            evidence_ref,
        })
    }

    pub fn audit_evidence(&self) -> &LocalUnixPeerCredentialEvidenceV1 {
        &self.evidence
    }

    pub fn evidence_ref(&self) -> &ApproverEvidenceRefV1 {
        &self.evidence_ref
    }
}

impl std::fmt::Debug for VerifiedLocalUnixPeerCredentialV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifiedLocalUnixPeerCredentialV1")
            .field("evidence_ref", &self.evidence_ref)
            .finish_non_exhaustive()
    }
}

/// Build a portable reference to externally verified Xenia evidence.
///
/// Nixward does not verify the referenced Xenia proof here. The qualified Xenia
/// bridge must supply the exact digest only after completing the owning Xenia
/// authentication/permit theorem. This function merely gives that digest a typed
/// profile at the Nixward audit boundary.
pub fn xenia_evidence_ref_v1(
    profile: ApproverEvidenceProfileV1,
    evidence_digest: impl Into<String>,
) -> Result<ApproverEvidenceRefV1, ApproverEvidenceErrorV1> {
    if !matches!(
        profile,
        ApproverEvidenceProfileV1::XeniaAuthenticatedPrincipalV1
            | ApproverEvidenceProfileV1::XeniaStateBoundPermitV1
    ) {
        return Err(ApproverEvidenceErrorV1::WrongExternalProfile);
    }
    let reference = ApproverEvidenceRefV1 {
        profile,
        evidence_digest: evidence_digest.into(),
    };
    reference.validate_shape()?;
    Ok(reference)
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ApproverEvidenceErrorV1 {
    #[error("empty or oversized field: {0}")]
    InvalidRef(&'static str),
    #[error("invalid canonical digest field: {0}")]
    InvalidDigest(&'static str),
    #[error("peer process id must be nonzero when present")]
    InvalidProcessId,
    #[error("external evidence helper requires a Xenia evidence profile")]
    WrongExternalProfile,
}

fn validate_ref(value: &str, field: &'static str) -> Result<(), ApproverEvidenceErrorV1> {
    if value.trim().is_empty() || value.len() > MAX_REF_BYTES {
        Err(ApproverEvidenceErrorV1::InvalidRef(field))
    } else {
        Ok(())
    }
}

fn validate_digest(value: &str, field: &'static str) -> Result<(), ApproverEvidenceErrorV1> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(ApproverEvidenceErrorV1::InvalidDigest(field));
    }
    Ok(())
}

fn put_u8(h: &mut Hasher, value: u8) {
    h.update(&[value]);
}

fn put_u32(h: &mut Hasher, value: u32) {
    h.update(&value.to_be_bytes());
}

fn put_u64(h: &mut Hasher, value: u64) {
    h.update(&value.to_be_bytes());
}

fn put_str(h: &mut Hasher, value: &str) {
    put_u64(h, value.len() as u64);
    h.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observed() -> LocalUnixPeerCredentialEvidenceV1 {
        LocalUnixPeerCredentialEvidenceV1::new_unverified(
            1000,
            100,
            Some(4242),
            "unix-socket-instance:test-1",
            UnixMillisV1::new(1_000),
        )
        .unwrap()
    }

    #[test]
    fn local_peer_evidence_digest_is_deterministic_and_semantic() {
        let a = observed();
        let b = observed();
        assert_eq!(a.digest().unwrap(), b.digest().unwrap());

        let json = serde_json::to_vec(&a).unwrap();
        assert_ne!(
            a.digest().unwrap(),
            blake3::hash(&json).to_hex().to_string(),
            "local peer evidence identity must not become serde/JSON hashing"
        );
    }

    #[test]
    fn changed_uid_pid_transport_or_time_changes_identity() {
        let baseline = observed().digest().unwrap();

        let uid = LocalUnixPeerCredentialEvidenceV1::new_unverified(
            1001,
            100,
            Some(4242),
            "unix-socket-instance:test-1",
            UnixMillisV1::new(1_000),
        )
        .unwrap();
        assert_ne!(baseline, uid.digest().unwrap());

        let pid = LocalUnixPeerCredentialEvidenceV1::new_unverified(
            1000,
            100,
            Some(4243),
            "unix-socket-instance:test-1",
            UnixMillisV1::new(1_000),
        )
        .unwrap();
        assert_ne!(baseline, pid.digest().unwrap());

        let transport = LocalUnixPeerCredentialEvidenceV1::new_unverified(
            1000,
            100,
            Some(4242),
            "unix-socket-instance:test-2",
            UnixMillisV1::new(1_000),
        )
        .unwrap();
        assert_ne!(baseline, transport.digest().unwrap());

        let time = LocalUnixPeerCredentialEvidenceV1::new_unverified(
            1000,
            100,
            Some(4242),
            "unix-socket-instance:test-1",
            UnixMillisV1::new(1_001),
        )
        .unwrap();
        assert_ne!(baseline, time.digest().unwrap());
    }

    #[test]
    fn serialized_local_evidence_is_not_verified_local_peer() {
        let evidence = observed();
        let json = serde_json::to_string(&evidence).unwrap();
        let restored: LocalUnixPeerCredentialEvidenceV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, evidence);

        // There is intentionally no public conversion from `restored` into
        // `VerifiedLocalUnixPeerCredentialV1`.
        assert_eq!(
            restored.evidence_ref().unwrap().profile,
            ApproverEvidenceProfileV1::LocalUnixPeerCredentialV1
        );
    }

    #[test]
    fn kernel_positive_type_exposes_only_bound_audit_evidence() {
        let verified = VerifiedLocalUnixPeerCredentialV1::from_kernel_peer_observation(
            1000,
            100,
            Some(4242),
            "unix-socket-instance:test-1",
            UnixMillisV1::new(1_000),
        )
        .unwrap();

        assert_eq!(verified.audit_evidence(), &observed());
        assert_eq!(
            verified.evidence_ref().profile,
            ApproverEvidenceProfileV1::LocalUnixPeerCredentialV1
        );
        assert_eq!(
            verified.evidence_ref().evidence_digest,
            observed().digest().unwrap()
        );
    }

    #[test]
    fn xenia_reference_is_typed_but_does_not_verify_xenia() {
        let digest = "ab".repeat(32);
        let principal = xenia_evidence_ref_v1(
            ApproverEvidenceProfileV1::XeniaAuthenticatedPrincipalV1,
            digest.clone(),
        )
        .unwrap();
        assert_eq!(principal.evidence_digest, digest);

        let permit = xenia_evidence_ref_v1(
            ApproverEvidenceProfileV1::XeniaStateBoundPermitV1,
            "cd".repeat(32),
        )
        .unwrap();
        assert_eq!(
            permit.profile,
            ApproverEvidenceProfileV1::XeniaStateBoundPermitV1
        );

        assert_eq!(
            xenia_evidence_ref_v1(
                ApproverEvidenceProfileV1::LocalUnixPeerCredentialV1,
                "ef".repeat(32),
            )
            .unwrap_err(),
            ApproverEvidenceErrorV1::WrongExternalProfile
        );
    }

    #[test]
    fn invalid_process_and_digest_shapes_fail_closed() {
        assert_eq!(
            LocalUnixPeerCredentialEvidenceV1::new_unverified(
                1000,
                100,
                Some(0),
                "unix-socket-instance:test-1",
                UnixMillisV1::new(1_000),
            )
            .unwrap_err(),
            ApproverEvidenceErrorV1::InvalidProcessId
        );

        assert_eq!(
            xenia_evidence_ref_v1(
                ApproverEvidenceProfileV1::XeniaAuthenticatedPrincipalV1,
                "not-a-digest",
            )
            .unwrap_err(),
            ApproverEvidenceErrorV1::InvalidDigest("approver evidence digest")
        );
    }
}
