// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Minimal privileged ingress boundary for consequential cyber-physical requests.
//!
//! This crate is deliberately **non-actuating**. It owns no device/HAL handle and cannot
//! create a `FinalActuatorPermit` or `CompleteJitHalLease`. Its responsibility is narrower:
//!
//! 1. accept exactly one bounded request on one Unix-domain connection;
//! 2. authenticate the connecting Linux UID from kernel peer credentials;
//! 3. decode the evidence-only guard protocol canonically;
//! 4. verify the exact Xenia receipt/payload with the fixed hybrid verifier;
//! 5. require the physical-interlock report to name the same envelope, device, and
//!    current transport-trust generation; and
//! 6. return only an audit-bearing `EvidenceVerifiedNoActuation` result.
//!
//! Caller-selected trust heads, clocks, verifiers, policies, runtime observations,
//! semantic checkpoints, and HAL configuration are absent from the IPC request. Later
//! guard stages must still perform device semantic acceptance, controller-key trust,
//! final-gate composition, complete JIT revocation fencing, and the one physical I/O
//! attempt. Successful ingress therefore proves evidence consistency, **not authority**.

#![deny(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("symthaea-iot-actuation-guard is Linux-only and requires kernel Unix peer credentials");

use std::collections::BTreeSet;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use symthaea_authority::Digest32;
use symthaea_iot_actuation_guard_protocol::{
    DecodedGuardEvidence, GuardProtocolError, MAX_GUARD_REQUEST_FRAME_BYTES,
    decode_canonical_guard_request,
};
use symthaea_iot_transport_receipt::{
    TransportReceiptError, TransportTrustHead, TransportTrustRegistry, VerifiedTransportEnvelope,
};
use symthaea_iot_xenia_hybrid_verifier::verify_xenia_physical_effect_receipt;
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::time::timeout;

/// Current minimal response schema.
pub const ACTUATION_GUARD_RESPONSE_SCHEMA_VERSION: u16 = 1;
/// Socket mode: only the service identity may open the endpoint through filesystem ACLs.
pub const ACTUATION_GUARD_SOCKET_MODE: u32 = 0o600;
/// Upper bound on one authorized client's time holding the single-flight ingress slot.
pub const MAX_GUARD_REQUEST_TIMEOUT_MS: u64 = 2_000;
/// Response frames are intentionally tiny and contain no trust/security internals.
const MAX_GUARD_RESPONSE_FRAME_BYTES: usize = 1024;

/// Kernel-authenticated identity of one connected Unix peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuardPeerIdentity {
    uid: u32,
    gid: u32,
    pid: Option<i32>,
}

impl GuardPeerIdentity {
    /// Linux UID authenticated by the kernel socket credential API.
    pub const fn uid(&self) -> u32 {
        self.uid
    }

    /// Linux GID authenticated by the kernel socket credential API.
    pub const fn gid(&self) -> u32 {
        self.gid
    }

    /// Linux PID when supplied by the kernel.
    pub const fn pid(&self) -> Option<i32> {
        self.pid
    }
}

/// Exact local UID policy for callers allowed to submit evidence to the guard.
///
/// There is deliberately no implicit `uid == 0` or socket-owner bypass. Root must be
/// explicitly provisioned if a deployment wants root to be an accepted caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardPeerPolicy {
    allowed_uids: BTreeSet<u32>,
}

impl GuardPeerPolicy {
    /// Construct a non-empty exact UID allowlist.
    pub fn new(allowed_uids: BTreeSet<u32>) -> Result<Self, GuardServerError> {
        if allowed_uids.is_empty() {
            return Err(GuardServerError::EmptyAllowedUidSet);
        }
        Ok(Self { allowed_uids })
    }

    /// Read-only configured UID surface.
    pub fn allowed_uids(&self) -> &BTreeSet<u32> {
        &self.allowed_uids
    }

    /// Authenticate one already-connected Unix peer against kernel credentials.
    pub fn authorize(&self, stream: &UnixStream) -> Result<GuardPeerIdentity, GuardServerError> {
        let credentials = stream
            .peer_cred()
            .map_err(GuardServerError::PeerCredentialRead)?;
        let identity = GuardPeerIdentity {
            uid: credentials.uid(),
            gid: credentials.gid(),
            pid: credentials.pid(),
        };
        if !self.allowed_uids.contains(&identity.uid) {
            return Err(GuardServerError::PeerUidDenied(identity.uid));
        }
        Ok(identity)
    }
}

/// Guard-owned Xenia relying-party state.
///
/// The independently retained `anchored_transport_head` is provisioning/state-recovery
/// input owned by the privileged process. It never crosses the unprivileged IPC wire.
#[derive(Debug)]
pub struct GuardIngressState {
    transport_registry: TransportTrustRegistry,
    anchored_transport_head: TransportTrustHead,
}

impl GuardIngressState {
    /// Bind a current transport registry to its independently retained anti-rollback head.
    pub fn new(
        transport_registry: TransportTrustRegistry,
        anchored_transport_head: TransportTrustHead,
    ) -> Result<Self, GuardIngressError> {
        if transport_registry.head() != anchored_transport_head {
            return Err(GuardIngressError::AnchoredTransportHeadMismatch);
        }
        Ok(Self {
            transport_registry,
            anchored_transport_head,
        })
    }

    /// Current independently anchored transport trust generation.
    pub const fn anchored_transport_head(&self) -> TransportTrustHead {
        self.anchored_transport_head
    }

    /// Verify one complete canonical guard frame using the guard's local clock and fixed
    /// Xenia hybrid verifier. No physical authority is minted by this function.
    pub fn verify_frame(&self, frame: &[u8]) -> Result<VerifiedGuardIngress, GuardIngressError> {
        self.verify_frame_at(frame, system_unix_ms()?)
    }

    fn verify_frame_at(
        &self,
        frame: &[u8],
        now_unix_ms: u64,
    ) -> Result<VerifiedGuardIngress, GuardIngressError> {
        // Re-check the separately retained anchor at each request so future guard-owned
        // state reload/rotation cannot accidentally run under an unanchored registry.
        if self.transport_registry.head() != self.anchored_transport_head {
            return Err(GuardIngressError::AnchoredTransportHeadMismatch);
        }

        let decoded = decode_canonical_guard_request(frame)?;
        let transport = verify_xenia_physical_effect_receipt(
            &self.transport_registry,
            decoded.raw_transport_receipt(),
            decoded.raw_physical_effect_payload(),
            now_unix_ms,
        )?;

        let report = decoded.interlock_report();
        let report_digest = report
            .digest()
            .map_err(|_| GuardIngressError::InvalidInterlockReport)?;

        if report.envelope_digest != transport.envelope_digest() {
            return Err(GuardIngressError::InterlockEnvelopeMismatch);
        }
        if report.transport_trust_head != transport.trust_head() {
            return Err(GuardIngressError::InterlockTransportTrustMismatch);
        }
        if report.device != transport.envelope().command.device {
            return Err(GuardIngressError::InterlockDeviceMismatch);
        }
        if report.checked_at_unix_ms < transport.opened_at_unix_ms() {
            return Err(GuardIngressError::InterlockPredatesAuthenticatedTransport);
        }
        if now_unix_ms < report.checked_at_unix_ms || now_unix_ms >= report.expires_at_unix_ms {
            return Err(GuardIngressError::InterlockReportNotFresh);
        }

        Ok(VerifiedGuardIngress {
            decoded,
            transport,
            interlock_report_digest: report_digest,
            verified_at_unix_ms: now_unix_ms,
        })
    }
}

/// Opaque non-authorizing result of the privileged ingress verification stage.
///
/// This is deliberately neither `Clone` nor serializable. Later guard-local stages may
/// consume it to continue semantic/interlock/JIT verification, but it is not a permit.
#[derive(Debug)]
pub struct VerifiedGuardIngress {
    decoded: DecodedGuardEvidence,
    transport: VerifiedTransportEnvelope,
    interlock_report_digest: Digest32,
    verified_at_unix_ms: u64,
}

impl VerifiedGuardIngress {
    /// Audit-only commitment to the exact outer IPC evidence frame.
    pub const fn request_digest(&self) -> Digest32 {
        self.decoded.request_digest()
    }

    /// Exact authenticated semantic envelope commitment.
    pub const fn envelope_digest(&self) -> Digest32 {
        self.transport.envelope_digest()
    }

    /// Exact signed Xenia receipt-body commitment.
    pub const fn transport_receipt_digest(&self) -> Digest32 {
        self.transport.receipt_digest()
    }

    /// Guard-owned transport trust generation used for receipt verification.
    pub const fn transport_trust_head(&self) -> TransportTrustHead {
        self.transport.trust_head()
    }

    /// Domain-separated digest of the exact interlock report carried over IPC.
    pub const fn interlock_report_digest(&self) -> Digest32 {
        self.interlock_report_digest
    }

    /// Guard-local time at which ingress consistency was verified.
    pub const fn verified_at_unix_ms(&self) -> u64 {
        self.verified_at_unix_ms
    }

    /// Consume the ingress proof for a later guard-local stage.
    ///
    /// Returning the underlying objects does not create actuator authority; both remain
    /// evidence/type-state inputs to the downstream semantic, controller-trust and JIT
    /// gates.
    pub fn into_parts(self) -> (DecodedGuardEvidence, VerifiedTransportEnvelope, Digest32) {
        (
            self.decoded,
            self.transport,
            self.interlock_report_digest,
        )
    }
}

/// Minimal service configuration. Runtime-directory ownership and device allowlisting
/// belong to the NixOS/systemd unit rather than this wire/process crate.
#[derive(Debug, Clone)]
pub struct ActuationGuardServerConfig {
    /// Absolute Unix socket path inside a service-manager-owned runtime directory.
    pub socket_path: PathBuf,
    /// Exact kernel UID allowlist.
    pub peer_policy: GuardPeerPolicy,
    /// Whole-request deadline for one authorized peer.
    pub request_timeout: Duration,
}

impl ActuationGuardServerConfig {
    /// Validate deployment-independent server bounds.
    pub fn validate(&self) -> Result<(), GuardServerError> {
        if !self.socket_path.is_absolute() || self.socket_path.file_name().is_none() {
            return Err(GuardServerError::SocketPathMustBeAbsolute);
        }
        if self.request_timeout.is_zero()
            || self.request_timeout > Duration::from_millis(MAX_GUARD_REQUEST_TIMEOUT_MS)
        {
            return Err(GuardServerError::RequestTimeoutOutOfBounds);
        }
        if self.peer_policy.allowed_uids.is_empty() {
            return Err(GuardServerError::EmptyAllowedUidSet);
        }
        Ok(())
    }
}

/// Minimal Linux Unix-socket ingress server.
///
/// `serve_one` requires `&mut self` and handles one connection synchronously from accept
/// through response. The crate exposes no spawn-based/concurrent client path.
pub struct ActuationGuardServer {
    listener: UnixListener,
    config: ActuationGuardServerConfig,
    ingress: GuardIngressState,
}

impl ActuationGuardServer {
    /// Bind a fresh socket. The parent directory must already exist and the socket path
    /// must not: service-manager runtime-directory lifecycle is intentionally not hidden
    /// inside this privileged process.
    pub async fn bind(
        config: ActuationGuardServerConfig,
        ingress: GuardIngressState,
    ) -> Result<Self, GuardServerError> {
        config.validate()?;
        let parent = config
            .socket_path
            .parent()
            .ok_or(GuardServerError::SocketPathMustBeAbsolute)?;
        let metadata = tokio::fs::metadata(parent)
            .await
            .map_err(GuardServerError::RuntimeDirectoryMetadata)?;
        if !metadata.is_dir() {
            return Err(GuardServerError::RuntimeDirectoryNotDirectory);
        }
        if path_exists(&config.socket_path).await? {
            return Err(GuardServerError::SocketPathAlreadyExists);
        }

        let listener = UnixListener::bind(&config.socket_path).map_err(GuardServerError::Bind)?;
        let permissions = std::fs::Permissions::from_mode(ACTUATION_GUARD_SOCKET_MODE);
        if let Err(error) = tokio::fs::set_permissions(&config.socket_path, permissions).await {
            // Fail closed: a socket with unknown permissions must not be left usable.
            drop(listener);
            let _ = tokio::fs::remove_file(&config.socket_path).await;
            return Err(GuardServerError::SocketPermissions(error));
        }

        Ok(Self {
            listener,
            config,
            ingress,
        })
    }

    /// Accept and completely process exactly one connection before another may start.
    ///
    /// Unauthorized peers are closed without an application response. Authorized peers
    /// receive only `EvidenceVerifiedNoActuation` or the generic `Rejected` status; the
    /// detailed rejection remains local in the returned outcome for privileged audit.
    pub async fn serve_one(&mut self) -> Result<GuardServeOutcome, GuardServerError> {
        let (mut stream, _) = self.listener.accept().await.map_err(GuardServerError::Accept)?;
        let peer = self.config.peer_policy.authorize(&stream)?;

        let frame = match timeout(self.config.request_timeout, read_request_frame(&mut stream)).await
        {
            Err(_) => {
                let rejection = GuardRequestRejection::RequestTimedOut;
                let _ = write_response(&mut stream, GuardResponseStatusV1::Rejected).await;
                return Ok(GuardServeOutcome::Rejected { peer, rejection });
            }
            Ok(Err(error)) => {
                let rejection = GuardRequestRejection::Framing(error);
                let _ = write_response(&mut stream, GuardResponseStatusV1::Rejected).await;
                return Ok(GuardServeOutcome::Rejected { peer, rejection });
            }
            Ok(Ok(frame)) => frame,
        };

        match self.ingress.verify_frame(&frame) {
            Ok(ingress) => {
                write_response(
                    &mut stream,
                    GuardResponseStatusV1::EvidenceVerifiedNoActuation,
                )
                .await?;
                Ok(GuardServeOutcome::EvidenceVerifiedNoActuation { peer, ingress })
            }
            Err(error) => {
                let _ = write_response(&mut stream, GuardResponseStatusV1::Rejected).await;
                Ok(GuardServeOutcome::Rejected {
                    peer,
                    rejection: GuardRequestRejection::Ingress(error),
                })
            }
        }
    }

    /// Bound socket path for deployment/audit inspection.
    pub fn socket_path(&self) -> &Path {
        &self.config.socket_path
    }
}

/// Privileged local outcome from one completely serialized connection.
#[derive(Debug)]
pub enum GuardServeOutcome {
    /// Evidence crossed the ingress boundary and passed fixed transport/cross-binding
    /// verification. No actuator authority or physical I/O occurred.
    EvidenceVerifiedNoActuation {
        /// Kernel-authenticated caller.
        peer: GuardPeerIdentity,
        /// Non-authorizing verified ingress state for a later local guard stage.
        ingress: VerifiedGuardIngress,
    },
    /// Authorized peer was rejected. Detailed reason is local-only and is never written
    /// to the unprivileged socket.
    Rejected {
        /// Kernel-authenticated caller.
        peer: GuardPeerIdentity,
        /// Detailed privileged audit reason.
        rejection: GuardRequestRejection,
    },
}

/// Minimal unprivileged response status. The wire deliberately exposes no rejection
/// reason, trust generation, key identity, device state or policy detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuardResponseStatusV1 {
    /// Evidence is internally consistent so far; no actuation occurred or was authorized.
    EvidenceVerifiedNoActuation,
    /// Generic fail-closed result.
    Rejected,
}

/// Tiny response wire frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardResponseV1 {
    /// Fail-closed response schema version.
    pub schema_version: u16,
    /// Intentionally coarse status.
    pub status: GuardResponseStatusV1,
}

async fn read_request_frame(stream: &mut UnixStream) -> Result<Vec<u8>, GuardFrameError> {
    let mut length_bytes = [0u8; 4];
    stream
        .read_exact(&mut length_bytes)
        .await
        .map_err(GuardFrameError::Read)?;
    let length = u32::from_le_bytes(length_bytes) as usize;
    if length == 0 || length > MAX_GUARD_REQUEST_FRAME_BYTES {
        return Err(GuardFrameError::LengthOutOfBounds(length));
    }
    let mut frame = vec![0u8; length];
    stream
        .read_exact(&mut frame)
        .await
        .map_err(GuardFrameError::Read)?;
    Ok(frame)
}

async fn write_response(
    stream: &mut UnixStream,
    status: GuardResponseStatusV1,
) -> Result<(), GuardServerError> {
    let response = GuardResponseV1 {
        schema_version: ACTUATION_GUARD_RESPONSE_SCHEMA_VERSION,
        status,
    };
    let bytes = bincode::serialize(&response).map_err(GuardServerError::ResponseEncoding)?;
    if bytes.is_empty() || bytes.len() > MAX_GUARD_RESPONSE_FRAME_BYTES {
        return Err(GuardServerError::ResponseEncodingOutOfBounds);
    }
    let length = u32::try_from(bytes.len()).map_err(|_| GuardServerError::ResponseEncodingOutOfBounds)?;
    stream
        .write_all(&length.to_le_bytes())
        .await
        .map_err(GuardServerError::ResponseWrite)?;
    stream
        .write_all(&bytes)
        .await
        .map_err(GuardServerError::ResponseWrite)?;
    stream.flush().await.map_err(GuardServerError::ResponseWrite)?;
    stream.shutdown().await.map_err(GuardServerError::ResponseWrite)?;
    Ok(())
}

async fn path_exists(path: &Path) -> Result<bool, GuardServerError> {
    match tokio::fs::symlink_metadata(path).await {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(GuardServerError::SocketPathMetadata(error)),
    }
}

fn system_unix_ms() -> Result<u64, GuardIngressError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| GuardIngressError::SystemClockBeforeUnixEpoch)?;
    u64::try_from(elapsed.as_millis()).map_err(|_| GuardIngressError::SystemClockOverflow)
}

/// Detailed local-only rejection from an authorized client's request.
#[derive(Debug, Error)]
pub enum GuardRequestRejection {
    /// Client occupied the single-flight slot longer than configured.
    #[error("authorized guard request timed out")]
    RequestTimedOut,
    /// Length-prefix or request read failed.
    #[error("guard request framing failed: {0}")]
    Framing(#[source] GuardFrameError),
    /// Evidence failed privileged ingress verification.
    #[error("guard ingress verification failed: {0}")]
    Ingress(#[source] GuardIngressError),
}

/// Fail-closed ingress verification errors. These are privileged audit details and must
/// not be serialized back to the unprivileged caller.
#[derive(Debug, Error)]
pub enum GuardIngressError {
    /// Current transport registry differs from the independently retained head.
    #[error("guard transport registry does not match independently anchored head")]
    AnchoredTransportHeadMismatch,
    /// Evidence-only outer request failed canonical/bounded parsing.
    #[error("guard protocol rejected evidence: {0}")]
    Protocol(#[from] GuardProtocolError),
    /// Xenia exact-payload receipt failed fixed hybrid/current-trust verification.
    #[error("Xenia transport verification failed: {0}")]
    Transport(#[from] TransportReceiptError),
    /// Interlock report failed its domain-separated digest contract.
    #[error("physical interlock report is invalid")]
    InvalidInterlockReport,
    /// Interlock report names another physical envelope.
    #[error("physical interlock report envelope differs from authenticated Xenia envelope")]
    InterlockEnvelopeMismatch,
    /// Interlock report names another transport-trust generation.
    #[error("physical interlock report transport trust differs from authenticated receipt")]
    InterlockTransportTrustMismatch,
    /// Interlock report names another physical device.
    #[error("physical interlock report device differs from authenticated command")]
    InterlockDeviceMismatch,
    /// Hardware observation predates Xenia's authenticated receipt acceptance.
    #[error("physical interlock report predates authenticated Xenia transport")]
    InterlockPredatesAuthenticatedTransport,
    /// Hardware report is not current at the guard's local time.
    #[error("physical interlock report is not fresh at guard verification time")]
    InterlockReportNotFresh,
    /// System wall clock is before Unix epoch.
    #[error("guard system clock is before Unix epoch")]
    SystemClockBeforeUnixEpoch,
    /// System wall-clock milliseconds do not fit the protocol time domain.
    #[error("guard system clock overflow")]
    SystemClockOverflow,
}

/// Length-prefixed request framing errors.
#[derive(Debug, Error)]
pub enum GuardFrameError {
    /// Socket read failed.
    #[error("request read failed: {0}")]
    Read(#[source] std::io::Error),
    /// Frame length is zero or exceeds the guard protocol limit.
    #[error("request length {0} is outside accepted bounds")]
    LengthOutOfBounds(usize),
}

/// Process/socket setup and response transport errors.
#[derive(Debug, Error)]
pub enum GuardServerError {
    /// Empty UID policy would make caller authorization ambiguous.
    #[error("actuation guard allowed UID set must not be empty")]
    EmptyAllowedUidSet,
    /// Socket must live at an absolute service-manager-owned path.
    #[error("actuation guard socket path must be absolute")]
    SocketPathMustBeAbsolute,
    /// Whole authorized request timeout is zero or too large.
    #[error("actuation guard request timeout is outside accepted bounds")]
    RequestTimeoutOutOfBounds,
    /// Runtime directory metadata could not be read.
    #[error("actuation guard runtime directory metadata failed: {0}")]
    RuntimeDirectoryMetadata(#[source] std::io::Error),
    /// Runtime parent exists but is not a directory.
    #[error("actuation guard runtime parent is not a directory")]
    RuntimeDirectoryNotDirectory,
    /// Socket-path metadata check failed.
    #[error("actuation guard socket-path metadata failed: {0}")]
    SocketPathMetadata(#[source] std::io::Error),
    /// Pre-existing socket/file is never unlinked implicitly by the guard.
    #[error("actuation guard socket path already exists")]
    SocketPathAlreadyExists,
    /// Unix socket bind failed.
    #[error("actuation guard socket bind failed: {0}")]
    Bind(#[source] std::io::Error),
    /// Socket could not be forced to mode 0600.
    #[error("actuation guard socket permission hardening failed: {0}")]
    SocketPermissions(#[source] std::io::Error),
    /// Accept failed.
    #[error("actuation guard socket accept failed: {0}")]
    Accept(#[source] std::io::Error),
    /// Kernel peer credential lookup failed.
    #[error("actuation guard kernel peer credential lookup failed: {0}")]
    PeerCredentialRead(#[source] std::io::Error),
    /// Peer UID is not explicitly provisioned.
    #[error("actuation guard peer UID {0} is not allowed")]
    PeerUidDenied(u32),
    /// Tiny response could not be serialized.
    #[error("actuation guard response encoding failed: {0}")]
    ResponseEncoding(#[source] bincode::Error),
    /// Response unexpectedly exceeded its tiny fixed ceiling.
    #[error("actuation guard response encoding exceeded fixed bound")]
    ResponseEncodingOutOfBounds,
    /// Generic response write/close failed.
    #[error("actuation guard response write failed: {0}")]
    ResponseWrite(#[source] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_policy_is_exact_and_never_implicitly_allows_root() {
        assert!(GuardPeerPolicy::new(BTreeSet::new()).is_err());
        let policy = GuardPeerPolicy::new(BTreeSet::from([1000])).unwrap();
        assert_eq!(policy.allowed_uids(), &BTreeSet::from([1000]));
        assert!(!policy.allowed_uids().contains(&0));
    }

    #[tokio::test]
    async fn kernel_peer_credentials_gate_exact_uid() {
        let (left, right) = UnixStream::pair().unwrap();
        let current_uid = left.peer_cred().unwrap().uid();
        let allow = GuardPeerPolicy::new(BTreeSet::from([current_uid])).unwrap();
        let identity = allow.authorize(&right).unwrap();
        assert_eq!(identity.uid(), current_uid);

        let denied_uid = if current_uid == u32::MAX {
            current_uid - 1
        } else {
            current_uid + 1
        };
        let deny = GuardPeerPolicy::new(BTreeSet::from([denied_uid])).unwrap();
        assert!(matches!(
            deny.authorize(&right),
            Err(GuardServerError::PeerUidDenied(uid)) if uid == current_uid
        ));
    }

    #[test]
    fn server_config_requires_absolute_path_and_short_bounded_timeout() {
        let policy = GuardPeerPolicy::new(BTreeSet::from([1000])).unwrap();
        let relative = ActuationGuardServerConfig {
            socket_path: PathBuf::from("guard.sock"),
            peer_policy: policy.clone(),
            request_timeout: Duration::from_millis(500),
        };
        assert!(matches!(
            relative.validate(),
            Err(GuardServerError::SocketPathMustBeAbsolute)
        ));

        let too_long = ActuationGuardServerConfig {
            socket_path: PathBuf::from("/run/symthaea-actuation/guard.sock"),
            peer_policy: policy,
            request_timeout: Duration::from_millis(MAX_GUARD_REQUEST_TIMEOUT_MS + 1),
        };
        assert!(matches!(
            too_long.validate(),
            Err(GuardServerError::RequestTimeoutOutOfBounds)
        ));
    }

    #[test]
    fn response_vocabulary_cannot_claim_authority_or_actuation() {
        let response = GuardResponseV1 {
            schema_version: ACTUATION_GUARD_RESPONSE_SCHEMA_VERSION,
            status: GuardResponseStatusV1::EvidenceVerifiedNoActuation,
        };
        let bytes = bincode::serialize(&response).unwrap();
        let decoded: GuardResponseV1 = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded, response);
    }
}
