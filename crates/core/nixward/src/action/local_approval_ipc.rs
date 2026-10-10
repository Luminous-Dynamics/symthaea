// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Kernel-backed local approval IPC identity observation.
//!
//! The lower #5299 source establishes kernel observation of Unix peer
//! credentials. In the private LOCAL-007 transport profile this adapter is
//! deliberately narrowed further:
//!
//! ```text
//! accepted UnixStream
//! + successful SO_PEERCRED observation through pinned nix 0.27
//! + peer effective UID == daemon effective UID
//! -> VerifiedLocalUnixPeerCredentialV1
//! ```
//!
//! Filesystem mode 0600 is defense in depth, not the identity theorem: privileged
//! processes may bypass DAC. The server-side SO_PEERCRED UID comparison is
//! therefore mandatory for this same-UID profile.
//!
//! The resulting credential still proves only a local process credential at this
//! connection boundary. It does not identify the human behind the process,
//! establish organizational authority, or grant Nix execution authority. A future
//! group/polkit/Xenia profile should use a separately qualified admission adapter
//! rather than silently widening this same-UID function.

use super::approver_evidence::{ApproverEvidenceErrorV1, VerifiedLocalUnixPeerCredentialV1};
use super::temporal::UnixMillisV1;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LocalApprovalIpcErrorV1 {
    #[error("kernel peer credential observation failed: {0}")]
    PeerCredentialObservation(String),
    #[error("kernel peer process id is invalid: {0}")]
    InvalidPeerProcessId(i64),
    #[error(
        "same-UID local approval profile rejected peer uid {observed_uid}; daemon effective uid is {expected_uid}"
    )]
    PeerEffectiveUidMismatch {
        expected_uid: u32,
        observed_uid: u32,
    },
    #[error(transparent)]
    ApproverEvidence(#[from] ApproverEvidenceErrorV1),
}

/// Observe and admit the kernel-reported credentials of an already accepted Linux
/// Unix-domain stream under the private same-effective-UID local approval profile.
///
/// `transport_instance_ref` is listener/session provenance and is bound into the
/// evidence identity, but it is not itself peer identity. UID/GID/PID come only
/// from `SO_PEERCRED`. The observed UID must additionally equal the daemon's
/// kernel-reported effective UID before the non-serializable positive type is
/// minted.
#[cfg(target_os = "linux")]
pub fn observe_linux_unix_peer_v1(
    stream: &std::os::unix::net::UnixStream,
    transport_instance_ref: impl Into<String>,
    observed_at: UnixMillisV1,
) -> Result<VerifiedLocalUnixPeerCredentialV1, LocalApprovalIpcErrorV1> {
    use nix::sys::socket::{getsockopt, sockopt::PeerCredentials};

    let credentials = getsockopt(stream, PeerCredentials)
        .map_err(|err| LocalApprovalIpcErrorV1::PeerCredentialObservation(err.to_string()))?;

    let expected_uid = nix::unistd::geteuid().as_raw();
    validate_same_effective_uid_v1(credentials.uid(), expected_uid)?;

    let raw_pid = credentials.pid();
    let process_id = u32::try_from(raw_pid)
        .map_err(|_| LocalApprovalIpcErrorV1::InvalidPeerProcessId(i64::from(raw_pid)))?;
    if process_id == 0 {
        return Err(LocalApprovalIpcErrorV1::InvalidPeerProcessId(0));
    }

    VerifiedLocalUnixPeerCredentialV1::from_kernel_peer_observation(
        credentials.uid(),
        credentials.gid(),
        Some(process_id),
        transport_instance_ref,
        observed_at,
    )
    .map_err(LocalApprovalIpcErrorV1::from)
}

fn validate_same_effective_uid_v1(
    observed_uid: u32,
    expected_uid: u32,
) -> Result<(), LocalApprovalIpcErrorV1> {
    if observed_uid == expected_uid {
        Ok(())
    } else {
        Err(LocalApprovalIpcErrorV1::PeerEffectiveUidMismatch {
            expected_uid,
            observed_uid,
        })
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use nix::sys::socket::UnixCredentials;
    use std::os::unix::net::UnixStream;

    #[test]
    fn connected_unix_stream_observes_kernel_peer_credentials() {
        let (left, _right) = UnixStream::pair().unwrap();
        let verified = observe_linux_unix_peer_v1(
            &left,
            "unix-socket-instance:test",
            UnixMillisV1::new(1_000),
        )
        .unwrap();

        let expected = UnixCredentials::new();
        let evidence = verified.audit_evidence();
        assert_eq!(evidence.effective_uid, expected.uid());
        assert_eq!(evidence.effective_uid, nix::unistd::geteuid().as_raw());
        assert_eq!(evidence.effective_gid, expected.gid());
        assert_eq!(evidence.process_id, Some(expected.pid() as u32));
        assert_eq!(evidence.process_id, Some(std::process::id()));
        assert_eq!(evidence.transport_instance_ref, "unix-socket-instance:test");
        assert_eq!(evidence.observed_at_unix_ms, 1_000);
    }

    #[test]
    fn same_uid_profile_rejects_privilege_bypass_identity_mismatch() {
        assert!(validate_same_effective_uid_v1(1000, 1000).is_ok());
        assert_eq!(
            validate_same_effective_uid_v1(0, 1000).unwrap_err(),
            LocalApprovalIpcErrorV1::PeerEffectiveUidMismatch {
                expected_uid: 1000,
                observed_uid: 0,
            }
        );
        assert_eq!(
            validate_same_effective_uid_v1(1001, 1000).unwrap_err(),
            LocalApprovalIpcErrorV1::PeerEffectiveUidMismatch {
                expected_uid: 1000,
                observed_uid: 1001,
            }
        );
    }

    #[test]
    fn peer_evidence_reference_is_bound_to_kernel_observation() {
        let (left, _right) = UnixStream::pair().unwrap();
        let verified = observe_linux_unix_peer_v1(
            &left,
            "unix-socket-instance:a",
            UnixMillisV1::new(1_000),
        )
        .unwrap();

        assert_eq!(
            verified.evidence_ref().evidence_digest,
            verified.audit_evidence().digest().unwrap()
        );
    }

    #[test]
    fn transport_reference_remains_provenance_not_peer_identity() {
        let (left, _right) = UnixStream::pair().unwrap();
        let a = observe_linux_unix_peer_v1(
            &left,
            "unix-socket-instance:a",
            UnixMillisV1::new(1_000),
        )
        .unwrap();
        let b = observe_linux_unix_peer_v1(
            &left,
            "unix-socket-instance:b",
            UnixMillisV1::new(1_000),
        )
        .unwrap();

        assert_eq!(a.audit_evidence().effective_uid, b.audit_evidence().effective_uid);
        assert_eq!(a.audit_evidence().effective_gid, b.audit_evidence().effective_gid);
        assert_eq!(a.audit_evidence().process_id, b.audit_evidence().process_id);
        assert_ne!(a.evidence_ref().evidence_digest, b.evidence_ref().evidence_digest);
    }
}
