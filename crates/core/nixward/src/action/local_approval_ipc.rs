// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Kernel-backed local approval IPC identity observation.
//!
//! This module is intentionally small. It does not define an approval protocol,
//! socket-path lifecycle, message framing, or authorization policy. Its only
//! positive theorem on Linux is:
//!
//! ```text
//! accepted UnixStream
//! + successful SO_PEERCRED observation through pinned nix 0.27
//! -> VerifiedLocalUnixPeerCredentialV1
//! ```
//!
//! The resulting credential proves a local process credential at this connection
//! boundary. It does not identify the human behind that process and does not grant
//! Nix execution authority.

use super::approver_evidence::{ApproverEvidenceErrorV1, VerifiedLocalUnixPeerCredentialV1};
use super::temporal::UnixMillisV1;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LocalApprovalIpcErrorV1 {
    #[error("kernel peer credential observation failed: {0}")]
    PeerCredentialObservation(String),
    #[error("kernel peer process id is invalid: {0}")]
    InvalidPeerProcessId(i64),
    #[error(transparent)]
    ApproverEvidence(#[from] ApproverEvidenceErrorV1),
}

/// Observe the kernel-reported credentials of an already accepted Linux Unix
/// domain stream and mint the corresponding non-serializable positive type.
///
/// `transport_instance_ref` is provenance supplied by the owning listener/session
/// layer. It is bound into the evidence identity but is not itself proof of the
/// peer credential. The UID/GID/PID fields come only from `SO_PEERCRED` through
/// `nix::sys::socket::getsockopt`.
#[cfg(target_os = "linux")]
pub fn observe_linux_unix_peer_v1(
    stream: &std::os::unix::net::UnixStream,
    transport_instance_ref: impl Into<String>,
    observed_at: UnixMillisV1,
) -> Result<VerifiedLocalUnixPeerCredentialV1, LocalApprovalIpcErrorV1> {
    use nix::sys::socket::{getsockopt, sockopt::PeerCredentials};

    let credentials = getsockopt(stream, PeerCredentials)
        .map_err(|err| LocalApprovalIpcErrorV1::PeerCredentialObservation(err.to_string()))?;

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
        assert_eq!(evidence.effective_gid, expected.gid());
        assert_eq!(evidence.process_id, Some(expected.pid() as u32));
        assert_eq!(evidence.process_id, Some(std::process::id()));
        assert_eq!(evidence.transport_instance_ref, "unix-socket-instance:test");
        assert_eq!(evidence.observed_at_unix_ms, 1_000);
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
