// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Protected same-UID Unix-socket transport for local Nixward approval.
//!
//! This module composes the local approval primitives without promoting transport
//! success into execution authority:
//!
//! ```text
//! private Unix socket
//! + kernel SO_PEERCRED
//! + strict identity-free LocalApprovalSubmissionV1
//! + daemon-incarnation-bound LocalApprovalRequestStoreV1
//! + atomic request consumption
//! -> ConsumedLocalApprovalDecisionV1
//! != Nix execution authority
//! ```
//!
//! V1 is intentionally a same-effective-UID profile. The runtime directory is
//! 0700 and the socket is 0600. A future system-service/group/Xenia profile must
//! define an explicit principal/policy theorem rather than weakening this default.

use super::daemon_incarnation::LiveDaemonIncarnationV1;
use super::local_approval::LocalApprovalDecisionKindV1;
use super::local_approval_ipc::{LocalApprovalIpcErrorV1, observe_linux_unix_peer_v1};
use super::local_approval_store::{
    ConsumedLocalApprovalDecisionV1, LocalApprovalRequestStoreErrorV1,
    LocalApprovalRequestStoreV1,
};
use super::local_approval_submission::LocalApprovalSubmissionV1;
use super::temporal::UnixMillisV1;
use blake3::Hasher;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::Shutdown;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;

pub const LOCAL_APPROVAL_PROTOCOL_V1: &str = "nixward-local-approval-ipc-v1";
pub const LOCAL_APPROVAL_SOCKET_FILENAME_V1: &str = "approval-v1.sock";
pub const LOCAL_APPROVAL_MAX_FRAME_BYTES_V1: usize = 16 * 1024;
const RUNTIME_DIR_MODE_V1: u32 = 0o700;
const SOCKET_MODE_V1: u32 = 0o600;
const IO_TIMEOUT_V1: Duration = Duration::from_secs(2);
const SOCKET_INSTANCE_DOMAIN_V1: &[u8] = b"nixward-local-approval-socket-instance-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalApprovalWireRequestV1 {
    pub protocol: String,
    pub submission: LocalApprovalSubmissionV1,
}

impl LocalApprovalWireRequestV1 {
    pub fn new(submission: LocalApprovalSubmissionV1) -> Self {
        Self {
            protocol: LOCAL_APPROVAL_PROTOCOL_V1.to_string(),
            submission,
        }
    }

    fn validate_protocol(&self) -> Result<(), LocalApprovalSocketErrorV1> {
        if self.protocol == LOCAL_APPROVAL_PROTOCOL_V1 {
            Ok(())
        } else {
            Err(LocalApprovalSocketErrorV1::ProtocolMismatch {
                observed: self.protocol.clone(),
            })
        }
    }
}

/// `DecisionConsumed` means only that the exact pending approval decision was
/// atomically consumed. It does not mean a Nix effect was authorized or executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalApprovalAckStatusV1 {
    DecisionConsumed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalApprovalAckV1 {
    pub protocol: String,
    pub request_id: String,
    pub decision: LocalApprovalDecisionKindV1,
    pub status: LocalApprovalAckStatusV1,
    pub consumed_at_unix_ms: u64,
}

impl LocalApprovalAckV1 {
    fn validate_for_submission(
        &self,
        submission: &LocalApprovalSubmissionV1,
    ) -> Result<(), LocalApprovalSocketErrorV1> {
        if self.protocol != LOCAL_APPROVAL_PROTOCOL_V1 {
            return Err(LocalApprovalSocketErrorV1::ProtocolMismatch {
                observed: self.protocol.clone(),
            });
        }
        if self.request_id != submission.request_id {
            return Err(LocalApprovalSocketErrorV1::AckRequestMismatch);
        }
        if self.decision != submission.decision {
            return Err(LocalApprovalSocketErrorV1::AckDecisionMismatch);
        }
        Ok(())
    }
}

/// Live listener for one daemon/socket incarnation. This type is deliberately
/// non-Serde/non-Clone. Drop removes only the exact device/inode it originally
/// bound, so cleanup cannot blindly unlink a replacement pathname.
pub struct LocalApprovalSocketServerV1 {
    listener: UnixListener,
    socket_path: PathBuf,
    transport_instance_ref: String,
    owner_uid: u32,
    socket_dev: u64,
    socket_ino: u64,
}

impl LocalApprovalSocketServerV1 {
    pub fn bind_default(
        daemon_incarnation: &LiveDaemonIncarnationV1,
    ) -> Result<Self, LocalApprovalSocketErrorV1> {
        let runtime_dir = default_local_approval_runtime_dir_v1()?;
        Self::bind_in(&runtime_dir, daemon_incarnation)
    }

    pub fn bind_in(
        runtime_dir: &Path,
        daemon_incarnation: &LiveDaemonIncarnationV1,
    ) -> Result<Self, LocalApprovalSocketErrorV1> {
        Self::bind_in_with_nonce_v1(runtime_dir, daemon_incarnation, os_random_32_v1()?)
    }

    fn bind_in_with_nonce_v1(
        runtime_dir: &Path,
        daemon_incarnation: &LiveDaemonIncarnationV1,
        socket_instance_nonce: [u8; 32],
    ) -> Result<Self, LocalApprovalSocketErrorV1> {
        let owner_uid = current_euid_v1();
        prepare_private_runtime_dir_v1(runtime_dir, owner_uid)?;

        let socket_path = runtime_dir.join(LOCAL_APPROVAL_SOCKET_FILENAME_V1);
        prepare_socket_path_for_bind_v1(&socket_path, owner_uid)?;

        let listener = UnixListener::bind(&socket_path)
            .map_err(|err| io_error("bind local approval socket", &socket_path, err))?;
        fs::set_permissions(&socket_path, fs::Permissions::from_mode(SOCKET_MODE_V1))
            .map_err(|err| io_error("chmod local approval socket", &socket_path, err))?;

        let metadata = fs::symlink_metadata(&socket_path)
            .map_err(|err| io_error("stat bound local approval socket", &socket_path, err))?;
        validate_bound_socket_metadata_v1(&socket_path, &metadata, owner_uid)?;

        let socket_dev = metadata.dev();
        let socket_ino = metadata.ino();
        let transport_instance_ref = transport_instance_ref_v1(
            daemon_incarnation,
            &socket_path,
            socket_dev,
            socket_ino,
            socket_instance_nonce,
        );

        Ok(Self {
            listener,
            socket_path,
            transport_instance_ref,
            owner_uid,
            socket_dev,
            socket_ino,
        })
    }

    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    pub fn transport_instance_ref(&self) -> &str {
        &self.transport_instance_ref
    }

    /// Accept one decision. Peer-credential observation time and decision
    /// evaluation time are deliberately distinct: `accept()` may block and frame
    /// receipt may take time, so sampling one `now` before accept would make a
    /// later legitimate client timestamp appear to come from the future.
    pub fn accept_and_consume(
        &self,
        store: &LocalApprovalRequestStoreV1,
    ) -> Result<ConsumedLocalApprovalDecisionV1, LocalApprovalSocketErrorV1> {
        let (mut stream, _) = self
            .listener
            .accept()
            .map_err(|err| io_error("accept local approval client", &self.socket_path, err))?;
        configure_session_timeouts_v1(&stream, &self.socket_path)?;

        let peer_observed_at = system_unix_millis_v1()?;
        let verified_peer = observe_linux_unix_peer_v1(
            &stream,
            self.transport_instance_ref.clone(),
            peer_observed_at,
        )?;

        let request: LocalApprovalWireRequestV1 = read_json_frame_v1(&mut stream)?;
        request.validate_protocol()?;
        let evaluated_at = system_unix_millis_v1()?;
        self.consume_and_ack_v1(&mut stream, store, request, &verified_peer, evaluated_at)
    }

    fn accept_and_consume_at_v1(
        &self,
        store: &LocalApprovalRequestStoreV1,
        peer_observed_at: UnixMillisV1,
        evaluated_at: UnixMillisV1,
    ) -> Result<ConsumedLocalApprovalDecisionV1, LocalApprovalSocketErrorV1> {
        let (mut stream, _) = self
            .listener
            .accept()
            .map_err(|err| io_error("accept local approval client", &self.socket_path, err))?;
        configure_session_timeouts_v1(&stream, &self.socket_path)?;
        let verified_peer = observe_linux_unix_peer_v1(
            &stream,
            self.transport_instance_ref.clone(),
            peer_observed_at,
        )?;
        let request: LocalApprovalWireRequestV1 = read_json_frame_v1(&mut stream)?;
        request.validate_protocol()?;
        self.consume_and_ack_v1(&mut stream, store, request, &verified_peer, evaluated_at)
    }

    fn consume_and_ack_v1(
        &self,
        stream: &mut UnixStream,
        store: &LocalApprovalRequestStoreV1,
        request: LocalApprovalWireRequestV1,
        verified_peer: &super::approver_evidence::VerifiedLocalUnixPeerCredentialV1,
        evaluated_at: UnixMillisV1,
    ) -> Result<ConsumedLocalApprovalDecisionV1, LocalApprovalSocketErrorV1> {
        let consumed =
            store.consume_verified_submission(&request.submission, verified_peer, evaluated_at)?;
        let ack = LocalApprovalAckV1 {
            protocol: LOCAL_APPROVAL_PROTOCOL_V1.to_string(),
            request_id: consumed.request_id().to_string(),
            decision: consumed.decision_kind(),
            status: LocalApprovalAckStatusV1::DecisionConsumed,
            consumed_at_unix_ms: consumed.consumed_at().as_u64(),
        };

        // ACK occurs only after atomic consume. Failure drops the consumed live
        // token from this call, so effect authority cannot proceed. The daemon must
        // explicitly issue a fresh request if it wants the operator to try again.
        write_json_frame_v1(stream, &ack).map_err(|err| {
            LocalApprovalSocketErrorV1::AckWriteAfterConsume {
                request_id: consumed.request_id().to_string(),
                cause: Box::new(err),
            }
        })?;
        let _ = stream.shutdown(Shutdown::Both);
        Ok(consumed)
    }
}

impl std::fmt::Debug for LocalApprovalSocketServerV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalApprovalSocketServerV1")
            .field("socket_path", &self.socket_path)
            .field("transport_instance_ref", &self.transport_instance_ref)
            .field("owner_uid", &self.owner_uid)
            .field("socket_dev", &self.socket_dev)
            .field("socket_ino", &self.socket_ino)
            .finish_non_exhaustive()
    }
}

impl Drop for LocalApprovalSocketServerV1 {
    fn drop(&mut self) {
        let Ok(metadata) = fs::symlink_metadata(&self.socket_path) else {
            return;
        };
        if metadata.file_type().is_socket()
            && !metadata.file_type().is_symlink()
            && metadata.uid() == self.owner_uid
            && metadata.dev() == self.socket_dev
            && metadata.ino() == self.socket_ino
        {
            let _ = fs::remove_file(&self.socket_path);
        }
    }
}

/// Same-UID V1 client helper for a future TUI integration.
pub fn submit_local_approval_v1(
    socket_path: &Path,
    submission: &LocalApprovalSubmissionV1,
) -> Result<LocalApprovalAckV1, LocalApprovalSocketErrorV1> {
    validate_same_uid_client_endpoint_v1(socket_path)?;
    let mut stream = UnixStream::connect(socket_path)
        .map_err(|err| io_error("connect local approval socket", socket_path, err))?;
    configure_session_timeouts_v1(&stream, socket_path)?;
    write_json_frame_v1(
        &mut stream,
        &LocalApprovalWireRequestV1::new(submission.clone()),
    )?;
    let ack: LocalApprovalAckV1 = read_json_frame_v1(&mut stream)?;
    ack.validate_for_submission(submission)?;
    let _ = stream.shutdown(Shutdown::Both);
    Ok(ack)
}

/// Resolve a runtime-only path. Never fall back to persistent state or `/tmp`.
pub fn default_local_approval_runtime_dir_v1() -> Result<PathBuf, LocalApprovalSocketErrorV1> {
    if let Some(explicit) = env::var_os("NIXWARD_RUNTIME_DIR") {
        let path = PathBuf::from(explicit);
        require_absolute_v1(&path)?;
        return Ok(path);
    }
    if let Some(xdg_runtime) = env::var_os("XDG_RUNTIME_DIR") {
        let base = PathBuf::from(xdg_runtime);
        require_absolute_v1(&base)?;
        return Ok(base.join("nixward"));
    }

    let uid = current_euid_v1();
    if uid == 0 {
        Ok(PathBuf::from("/run/nixward"))
    } else {
        Ok(PathBuf::from(format!("/run/user/{uid}/nixward")))
    }
}

fn current_euid_v1() -> u32 {
    nix::unistd::geteuid().as_raw()
}

fn os_random_32_v1() -> Result<[u8; 32], LocalApprovalSocketErrorV1> {
    let mut bytes = [0_u8; 32];
    getrandom::getrandom(&mut bytes)
        .map_err(|err| LocalApprovalSocketErrorV1::OsRandom(err.to_string()))?;
    Ok(bytes)
}

fn prepare_private_runtime_dir_v1(
    runtime_dir: &Path,
    owner_uid: u32,
) -> Result<(), LocalApprovalSocketErrorV1> {
    require_absolute_v1(runtime_dir)?;
    reject_existing_symlink_components_v1(runtime_dir)?;

    match fs::symlink_metadata(runtime_dir) {
        Ok(metadata) => validate_or_tighten_runtime_dir_v1(runtime_dir, &metadata, owner_uid),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            let parent = runtime_dir
                .parent()
                .ok_or_else(|| LocalApprovalSocketErrorV1::RuntimeParentMissing {
                    path: runtime_dir.to_path_buf(),
                })?;
            let parent_metadata = fs::symlink_metadata(parent)
                .map_err(|err| io_error("stat local approval runtime parent", parent, err))?;
            if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
                return Err(LocalApprovalSocketErrorV1::RuntimeParentUnsafe {
                    path: parent.to_path_buf(),
                });
            }

            match fs::create_dir(runtime_dir) {
                Ok(()) => {}
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(err) => {
                    return Err(io_error(
                        "create local approval runtime directory",
                        runtime_dir,
                        err,
                    ));
                }
            }
            let metadata = fs::symlink_metadata(runtime_dir).map_err(|err| {
                io_error(
                    "stat created local approval runtime directory",
                    runtime_dir,
                    err,
                )
            })?;
            validate_or_tighten_runtime_dir_v1(runtime_dir, &metadata, owner_uid)
        }
        Err(err) => Err(io_error(
            "stat local approval runtime directory",
            runtime_dir,
            err,
        )),
    }
}

fn validate_or_tighten_runtime_dir_v1(
    runtime_dir: &Path,
    metadata: &fs::Metadata,
    owner_uid: u32,
) -> Result<(), LocalApprovalSocketErrorV1> {
    if metadata.file_type().is_symlink() {
        return Err(LocalApprovalSocketErrorV1::RuntimePathContainsSymlink {
            path: runtime_dir.to_path_buf(),
        });
    }
    if !metadata.is_dir() {
        return Err(LocalApprovalSocketErrorV1::RuntimePathNotDirectory {
            path: runtime_dir.to_path_buf(),
        });
    }
    if metadata.uid() != owner_uid {
        return Err(LocalApprovalSocketErrorV1::RuntimeOwnerMismatch {
            path: runtime_dir.to_path_buf(),
            expected_uid: owner_uid,
            observed_uid: metadata.uid(),
        });
    }

    if metadata.mode() & 0o7777 != RUNTIME_DIR_MODE_V1 {
        fs::set_permissions(runtime_dir, fs::Permissions::from_mode(RUNTIME_DIR_MODE_V1))
            .map_err(|err| io_error("tighten local approval runtime directory", runtime_dir, err))?;
    }

    let verified = fs::symlink_metadata(runtime_dir).map_err(|err| {
        io_error(
            "re-stat local approval runtime directory",
            runtime_dir,
            err,
        )
    })?;
    if verified.file_type().is_symlink()
        || !verified.is_dir()
        || verified.uid() != owner_uid
        || verified.mode() & 0o7777 != RUNTIME_DIR_MODE_V1
    {
        return Err(LocalApprovalSocketErrorV1::RuntimeDirectoryVerificationFailed {
            path: runtime_dir.to_path_buf(),
        });
    }
    Ok(())
}

fn prepare_socket_path_for_bind_v1(
    socket_path: &Path,
    owner_uid: u32,
) -> Result<(), LocalApprovalSocketErrorV1> {
    match fs::symlink_metadata(socket_path) {
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(io_error(
            "stat existing local approval socket",
            socket_path,
            err,
        )),
        Ok(metadata) => {
            if metadata.file_type().is_symlink() {
                return Err(LocalApprovalSocketErrorV1::ExistingEndpointSymlink {
                    path: socket_path.to_path_buf(),
                });
            }
            if !metadata.file_type().is_socket() {
                return Err(LocalApprovalSocketErrorV1::ExistingEndpointWrongType {
                    path: socket_path.to_path_buf(),
                });
            }
            if metadata.uid() != owner_uid {
                return Err(LocalApprovalSocketErrorV1::ExistingEndpointOwnerMismatch {
                    path: socket_path.to_path_buf(),
                    expected_uid: owner_uid,
                    observed_uid: metadata.uid(),
                });
            }

            match UnixStream::connect(socket_path) {
                Ok(stream) => {
                    drop(stream);
                    Err(LocalApprovalSocketErrorV1::EndpointAlreadyActive {
                        path: socket_path.to_path_buf(),
                    })
                }
                Err(err)
                    if matches!(
                        err.kind(),
                        std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::NotFound
                    ) =>
                {
                    match fs::remove_file(socket_path) {
                        Ok(()) => Ok(()),
                        Err(remove_err)
                            if remove_err.kind() == std::io::ErrorKind::NotFound =>
                        {
                            Ok(())
                        }
                        Err(remove_err) => Err(io_error(
                            "remove stale local approval socket",
                            socket_path,
                            remove_err,
                        )),
                    }
                }
                Err(err) => Err(io_error(
                    "probe existing local approval socket",
                    socket_path,
                    err,
                )),
            }
        }
    }
}

fn validate_bound_socket_metadata_v1(
    socket_path: &Path,
    metadata: &fs::Metadata,
    owner_uid: u32,
) -> Result<(), LocalApprovalSocketErrorV1> {
    if metadata.file_type().is_symlink() || !metadata.file_type().is_socket() {
        return Err(LocalApprovalSocketErrorV1::BoundEndpointVerificationFailed {
            path: socket_path.to_path_buf(),
        });
    }
    if metadata.uid() != owner_uid {
        return Err(LocalApprovalSocketErrorV1::ExistingEndpointOwnerMismatch {
            path: socket_path.to_path_buf(),
            expected_uid: owner_uid,
            observed_uid: metadata.uid(),
        });
    }
    if metadata.mode() & 0o7777 != SOCKET_MODE_V1 {
        return Err(LocalApprovalSocketErrorV1::BoundEndpointVerificationFailed {
            path: socket_path.to_path_buf(),
        });
    }
    Ok(())
}

fn validate_same_uid_client_endpoint_v1(
    socket_path: &Path,
) -> Result<(), LocalApprovalSocketErrorV1> {
    require_absolute_v1(socket_path)?;
    reject_existing_symlink_components_v1(socket_path)?;
    let metadata = fs::symlink_metadata(socket_path)
        .map_err(|err| io_error("stat local approval client endpoint", socket_path, err))?;
    validate_bound_socket_metadata_v1(socket_path, &metadata, current_euid_v1())
}

fn require_absolute_v1(path: &Path) -> Result<(), LocalApprovalSocketErrorV1> {
    if path.is_absolute() {
        Ok(())
    } else {
        Err(LocalApprovalSocketErrorV1::RuntimePathNotAbsolute {
            path: path.to_path_buf(),
        })
    }
}

fn reject_existing_symlink_components_v1(
    path: &Path,
) -> Result<(), LocalApprovalSocketErrorV1> {
    let mut prefix = PathBuf::new();
    for component in path.components() {
        match component {
            Component::RootDir => {
                prefix.push(Path::new("/"));
                continue;
            }
            Component::CurDir => continue,
            Component::ParentDir | Component::Prefix(_) => {
                return Err(LocalApprovalSocketErrorV1::RuntimePathTraversal {
                    path: path.to_path_buf(),
                });
            }
            Component::Normal(part) => prefix.push(part),
        }

        match fs::symlink_metadata(&prefix) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(LocalApprovalSocketErrorV1::RuntimePathContainsSymlink {
                    path: prefix,
                });
            }
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => break,
            Err(err) => {
                return Err(io_error(
                    "inspect local approval path component",
                    &prefix,
                    err,
                ));
            }
        }
    }
    Ok(())
}

fn transport_instance_ref_v1(
    daemon_incarnation: &LiveDaemonIncarnationV1,
    socket_path: &Path,
    socket_dev: u64,
    socket_ino: u64,
    socket_instance_nonce: [u8; 32],
) -> String {
    let daemon_ref = daemon_incarnation.reference();
    let mut hasher = Hasher::new();
    hasher.update(SOCKET_INSTANCE_DOMAIN_V1);
    put_bytes_v1(&mut hasher, daemon_ref.as_bytes());
    put_bytes_v1(&mut hasher, socket_path.as_os_str().as_bytes());
    hasher.update(&socket_dev.to_be_bytes());
    hasher.update(&socket_ino.to_be_bytes());
    hasher.update(&socket_instance_nonce);
    format!(
        "nixward-local-approval-socket-instance-v1:{}",
        hasher.finalize().to_hex()
    )
}

fn put_bytes_v1(hasher: &mut Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

fn configure_session_timeouts_v1(
    stream: &UnixStream,
    path: &Path,
) -> Result<(), LocalApprovalSocketErrorV1> {
    stream
        .set_read_timeout(Some(IO_TIMEOUT_V1))
        .map_err(|err| io_error("set local approval read timeout", path, err))?;
    stream
        .set_write_timeout(Some(IO_TIMEOUT_V1))
        .map_err(|err| io_error("set local approval write timeout", path, err))?;
    Ok(())
}

fn write_json_frame_v1<T: Serialize>(
    stream: &mut UnixStream,
    value: &T,
) -> Result<(), LocalApprovalSocketErrorV1> {
    let payload = serde_json::to_vec(value)
        .map_err(|err| LocalApprovalSocketErrorV1::JsonEncode(err.to_string()))?;
    if payload.is_empty() {
        return Err(LocalApprovalSocketErrorV1::EmptyFrame);
    }
    if payload.len() > LOCAL_APPROVAL_MAX_FRAME_BYTES_V1 {
        return Err(LocalApprovalSocketErrorV1::FrameTooLarge {
            observed: payload.len(),
            maximum: LOCAL_APPROVAL_MAX_FRAME_BYTES_V1,
        });
    }
    let length = u32::try_from(payload.len())
        .map_err(|_| LocalApprovalSocketErrorV1::FrameLengthOverflow)?;
    stream
        .write_all(&length.to_be_bytes())
        .map_err(|err| LocalApprovalSocketErrorV1::SessionIo(err.to_string()))?;
    stream
        .write_all(&payload)
        .map_err(|err| LocalApprovalSocketErrorV1::SessionIo(err.to_string()))?;
    stream
        .flush()
        .map_err(|err| LocalApprovalSocketErrorV1::SessionIo(err.to_string()))?;
    Ok(())
}

fn read_json_frame_v1<T: DeserializeOwned>(
    stream: &mut UnixStream,
) -> Result<T, LocalApprovalSocketErrorV1> {
    let mut length_bytes = [0_u8; 4];
    stream
        .read_exact(&mut length_bytes)
        .map_err(|err| LocalApprovalSocketErrorV1::SessionIo(err.to_string()))?;
    let length = u32::from_be_bytes(length_bytes) as usize;
    if length == 0 {
        return Err(LocalApprovalSocketErrorV1::EmptyFrame);
    }
    if length > LOCAL_APPROVAL_MAX_FRAME_BYTES_V1 {
        return Err(LocalApprovalSocketErrorV1::FrameTooLarge {
            observed: length,
            maximum: LOCAL_APPROVAL_MAX_FRAME_BYTES_V1,
        });
    }
    let mut payload = vec![0_u8; length];
    stream
        .read_exact(&mut payload)
        .map_err(|err| LocalApprovalSocketErrorV1::SessionIo(err.to_string()))?;
    serde_json::from_slice(&payload)
        .map_err(|err| LocalApprovalSocketErrorV1::JsonDecode(err.to_string()))
}

fn system_unix_millis_v1() -> Result<UnixMillisV1, LocalApprovalSocketErrorV1> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| LocalApprovalSocketErrorV1::SystemClock(err.to_string()))?;
    let millis = u64::try_from(duration.as_millis())
        .map_err(|_| LocalApprovalSocketErrorV1::SystemClockOverflow)?;
    Ok(UnixMillisV1::new(millis))
}

fn io_error(
    operation: &'static str,
    path: &Path,
    error: std::io::Error,
) -> LocalApprovalSocketErrorV1 {
    LocalApprovalSocketErrorV1::Io {
        operation,
        path: path.to_path_buf(),
        error: error.to_string(),
    }
}

#[derive(Debug, Error)]
pub enum LocalApprovalSocketErrorV1 {
    #[error("local approval runtime path must be absolute: {path:?}")]
    RuntimePathNotAbsolute { path: PathBuf },
    #[error("local approval runtime path contains traversal: {path:?}")]
    RuntimePathTraversal { path: PathBuf },
    #[error("local approval runtime path contains a symlink component: {path:?}")]
    RuntimePathContainsSymlink { path: PathBuf },
    #[error("local approval runtime path is not a directory: {path:?}")]
    RuntimePathNotDirectory { path: PathBuf },
    #[error("local approval runtime parent is missing: {path:?}")]
    RuntimeParentMissing { path: PathBuf },
    #[error("local approval runtime parent is unsafe: {path:?}")]
    RuntimeParentUnsafe { path: PathBuf },
    #[error(
        "local approval runtime owner mismatch for {path:?}: expected uid {expected_uid}, observed {observed_uid}"
    )]
    RuntimeOwnerMismatch {
        path: PathBuf,
        expected_uid: u32,
        observed_uid: u32,
    },
    #[error("local approval runtime directory failed post-change verification: {path:?}")]
    RuntimeDirectoryVerificationFailed { path: PathBuf },
    #[error("pre-existing local approval endpoint is a symlink: {path:?}")]
    ExistingEndpointSymlink { path: PathBuf },
    #[error("pre-existing local approval endpoint is not a Unix socket: {path:?}")]
    ExistingEndpointWrongType { path: PathBuf },
    #[error(
        "pre-existing local approval endpoint owner mismatch for {path:?}: expected uid {expected_uid}, observed {observed_uid}"
    )]
    ExistingEndpointOwnerMismatch {
        path: PathBuf,
        expected_uid: u32,
        observed_uid: u32,
    },
    #[error("another local approval listener is already active at {path:?}")]
    EndpointAlreadyActive { path: PathBuf },
    #[error("bound local approval endpoint failed type/owner/mode verification: {path:?}")]
    BoundEndpointVerificationFailed { path: PathBuf },
    #[error("local approval protocol mismatch: observed {observed:?}")]
    ProtocolMismatch { observed: String },
    #[error("local approval acknowledgement names another request")]
    AckRequestMismatch,
    #[error("local approval acknowledgement reports another decision")]
    AckDecisionMismatch,
    #[error("local approval frame must not be empty")]
    EmptyFrame,
    #[error("local approval frame is {observed} bytes; maximum is {maximum}")]
    FrameTooLarge { observed: usize, maximum: usize },
    #[error("local approval frame length cannot be represented on the wire")]
    FrameLengthOverflow,
    #[error("local approval JSON encoding failed: {0}")]
    JsonEncode(String),
    #[error("local approval JSON decoding failed: {0}")]
    JsonDecode(String),
    #[error("local approval socket session I/O failed: {0}")]
    SessionIo(String),
    #[error("system wall clock unavailable: {0}")]
    SystemClock(String),
    #[error("system wall-clock milliseconds overflow u64")]
    SystemClockOverflow,
    #[error("operating-system randomness unavailable for socket incarnation: {0}")]
    OsRandom(String),
    #[error("{operation} failed for {path:?}: {error}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        error: String,
    },
    #[error(transparent)]
    PeerCredential(#[from] LocalApprovalIpcErrorV1),
    #[error(transparent)]
    RequestStore(#[from] LocalApprovalRequestStoreErrorV1),
    #[error(
        "approval decision for request {request_id} was consumed, but acknowledgement write failed: {cause}"
    )]
    AckWriteAfterConsume {
        request_id: String,
        #[source]
        cause: Box<LocalApprovalSocketErrorV1>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::authorization::NixActionIntentV1;
    use crate::action::executor::NixOSCommand;
    use crate::action::{LocalApprovalDecisionKindV1, PendingNixApprovalRequestV1};
    use std::os::unix::fs::symlink;
    use std::sync::Arc;
    use std::thread;

    fn ms(value: u64) -> UnixMillisV1 {
        UnixMillisV1::new(value)
    }

    fn request_for(daemon: &LiveDaemonIncarnationV1) -> PendingNixApprovalRequestV1 {
        let intent = NixActionIntentV1::from_command(
            "machine:workstation",
            Some("generation:42".to_string()),
            &NixOSCommand::RebuildSwitch {
                flake: Some(".#workstation".to_string()),
                extra_args: vec![],
            },
        )
        .unwrap();
        daemon
            .create_approval_request(
                &intent,
                "nixos-rebuild switch --flake .#workstation",
                "local-human-v1",
                ms(1_000),
                ms(2_000),
            )
            .unwrap()
    }

    fn private_runtime_path() -> (tempfile::TempDir, PathBuf) {
        let parent = tempfile::tempdir().unwrap();
        let runtime = parent.path().join("runtime");
        (parent, runtime)
    }

    #[test]
    fn runtime_and_socket_are_private_same_uid_objects() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let (_parent, runtime) = private_runtime_path();
        let server = LocalApprovalSocketServerV1::bind_in(&runtime, &daemon).unwrap();
        let runtime_meta = fs::symlink_metadata(&runtime).unwrap();
        let socket_meta = fs::symlink_metadata(server.socket_path()).unwrap();
        assert_eq!(runtime_meta.uid(), current_euid_v1());
        assert_eq!(runtime_meta.mode() & 0o7777, 0o700);
        assert!(socket_meta.file_type().is_socket());
        assert_eq!(socket_meta.uid(), current_euid_v1());
        assert_eq!(socket_meta.mode() & 0o7777, 0o600);
    }

    #[test]
    fn runtime_symlink_is_rejected_without_touching_target() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let parent = tempfile::tempdir().unwrap();
        let target = parent.path().join("target");
        let runtime = parent.path().join("runtime");
        fs::create_dir(&target).unwrap();
        symlink(&target, &runtime).unwrap();
        let err = LocalApprovalSocketServerV1::bind_in(&runtime, &daemon).unwrap_err();
        assert!(matches!(
            err,
            LocalApprovalSocketErrorV1::RuntimePathContainsSymlink { .. }
        ));
        assert!(target.is_dir());
    }

    #[test]
    fn preexisting_regular_endpoint_is_rejected_and_preserved() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let (_parent, runtime) = private_runtime_path();
        prepare_private_runtime_dir_v1(&runtime, current_euid_v1()).unwrap();
        let endpoint = runtime.join(LOCAL_APPROVAL_SOCKET_FILENAME_V1);
        fs::write(&endpoint, b"do-not-delete").unwrap();
        let err = LocalApprovalSocketServerV1::bind_in(&runtime, &daemon).unwrap_err();
        assert!(matches!(
            err,
            LocalApprovalSocketErrorV1::ExistingEndpointWrongType { .. }
        ));
        assert_eq!(fs::read(&endpoint).unwrap(), b"do-not-delete");
    }

    #[test]
    fn active_existing_listener_is_not_unlinked() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let (_parent, runtime) = private_runtime_path();
        prepare_private_runtime_dir_v1(&runtime, current_euid_v1()).unwrap();
        let endpoint = runtime.join(LOCAL_APPROVAL_SOCKET_FILENAME_V1);
        let existing = UnixListener::bind(&endpoint).unwrap();
        let err = LocalApprovalSocketServerV1::bind_in(&runtime, &daemon).unwrap_err();
        assert!(matches!(
            err,
            LocalApprovalSocketErrorV1::EndpointAlreadyActive { .. }
        ));
        assert!(fs::symlink_metadata(&endpoint)
            .unwrap()
            .file_type()
            .is_socket());
        drop(existing);
    }

    #[test]
    fn stale_owned_socket_is_replaced() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let (_parent, runtime) = private_runtime_path();
        prepare_private_runtime_dir_v1(&runtime, current_euid_v1()).unwrap();
        let endpoint = runtime.join(LOCAL_APPROVAL_SOCKET_FILENAME_V1);
        let stale = UnixListener::bind(&endpoint).unwrap();
        drop(stale);
        let server = LocalApprovalSocketServerV1::bind_in(&runtime, &daemon).unwrap();
        assert!(UnixStream::connect(server.socket_path()).is_ok());
    }

    #[test]
    fn drop_removes_only_exact_bound_inode() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let (_parent, runtime) = private_runtime_path();
        let server = LocalApprovalSocketServerV1::bind_in(&runtime, &daemon).unwrap();
        let endpoint = server.socket_path().to_path_buf();
        fs::remove_file(&endpoint).unwrap();
        let replacement = UnixListener::bind(&endpoint).unwrap();
        let replacement_ino = fs::symlink_metadata(&endpoint).unwrap().ino();
        drop(server);
        assert_eq!(fs::symlink_metadata(&endpoint).unwrap().ino(), replacement_ino);
        drop(replacement);
    }

    #[test]
    fn socket_instance_nonce_changes_transport_identity_deterministically() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let path = Path::new("/run/user/1000/nixward/approval-v1.sock");
        let first = transport_instance_ref_v1(&daemon, path, 1, 2, [3; 32]);
        let same = transport_instance_ref_v1(&daemon, path, 1, 2, [3; 32]);
        let second = transport_instance_ref_v1(&daemon, path, 1, 2, [4; 32]);
        assert_eq!(first, same);
        assert_ne!(first, second);
    }

    #[test]
    fn strict_wire_envelope_rejects_identity_fields() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let request = request_for(&daemon);
        let submission = LocalApprovalSubmissionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_200),
        )
        .unwrap();
        let mut value = serde_json::to_value(LocalApprovalWireRequestV1::new(submission)).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("uid".to_string(), serde_json::json!(0));
        assert!(serde_json::from_value::<LocalApprovalWireRequestV1>(value).is_err());
    }

    #[test]
    fn end_to_end_socket_consumes_exact_request_and_returns_non_authority_ack() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let store = Arc::new(LocalApprovalRequestStoreV1::new(&daemon));
        let request = request_for(&daemon);
        let request_id = request.request_id().unwrap();
        store.install_pending(request.clone()).unwrap();
        let submission = LocalApprovalSubmissionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_200),
        )
        .unwrap();
        let (_parent, runtime) = private_runtime_path();
        let server = LocalApprovalSocketServerV1::bind_in(&runtime, &daemon).unwrap();
        let socket_path = server.socket_path().to_path_buf();
        let client_submission = submission.clone();
        let client = thread::spawn(move || {
            submit_local_approval_v1(&socket_path, &client_submission).unwrap()
        });

        let consumed = server
            .accept_and_consume_at_v1(&store, ms(1_100), ms(1_300))
            .unwrap();
        let ack = client.join().unwrap();
        assert_eq!(consumed.request_id(), request_id);
        assert_eq!(ack.request_id, request_id);
        assert_eq!(ack.status, LocalApprovalAckStatusV1::DecisionConsumed);
        assert_eq!(store.pending_count().unwrap(), 0);
        assert!(consumed
            .decision_evidence()
            .approver_ref
            .starts_with("nixward-approver-evidence-v1:local-unix-peer-credential-v1:"));
    }

    #[test]
    fn malformed_wire_does_not_consume_legitimate_request() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let store = Arc::new(LocalApprovalRequestStoreV1::new(&daemon));
        let request = request_for(&daemon);
        let request_id = request.request_id().unwrap();
        store.install_pending(request.clone()).unwrap();
        let submission = LocalApprovalSubmissionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_200),
        )
        .unwrap();
        let (_parent, runtime) = private_runtime_path();
        let server = LocalApprovalSocketServerV1::bind_in(&runtime, &daemon).unwrap();
        let socket_path = server.socket_path().to_path_buf();
        let hostile = thread::spawn(move || {
            let mut stream = UnixStream::connect(&socket_path).unwrap();
            let mut value =
                serde_json::to_value(LocalApprovalWireRequestV1::new(submission)).unwrap();
            value
                .as_object_mut()
                .unwrap()
                .insert("uid".to_string(), serde_json::json!(0));
            write_json_frame_v1(&mut stream, &value).unwrap();
            let _ = stream.shutdown(Shutdown::Both);
        });
        let err = server
            .accept_and_consume_at_v1(&store, ms(1_100), ms(1_300))
            .unwrap_err();
        hostile.join().unwrap();
        assert!(matches!(err, LocalApprovalSocketErrorV1::JsonDecode(_)));
        assert!(store.is_pending(&request_id).unwrap());
    }

    #[test]
    fn oversized_frame_does_not_touch_pending_store() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let store = Arc::new(LocalApprovalRequestStoreV1::new(&daemon));
        let request = request_for(&daemon);
        let request_id = request.request_id().unwrap();
        store.install_pending(request).unwrap();
        let (_parent, runtime) = private_runtime_path();
        let server = LocalApprovalSocketServerV1::bind_in(&runtime, &daemon).unwrap();
        let socket_path = server.socket_path().to_path_buf();
        let client = thread::spawn(move || {
            let mut stream = UnixStream::connect(&socket_path).unwrap();
            let too_large = (LOCAL_APPROVAL_MAX_FRAME_BYTES_V1 as u32 + 1).to_be_bytes();
            stream.write_all(&too_large).unwrap();
            let _ = stream.shutdown(Shutdown::Both);
        });
        let err = server
            .accept_and_consume_at_v1(&store, ms(1_100), ms(1_300))
            .unwrap_err();
        client.join().unwrap();
        assert!(matches!(
            err,
            LocalApprovalSocketErrorV1::FrameTooLarge { .. }
        ));
        assert!(store.is_pending(&request_id).unwrap());
    }

    #[test]
    fn replay_after_success_cannot_consume_again() {
        let daemon = LiveDaemonIncarnationV1::generate().unwrap();
        let store = Arc::new(LocalApprovalRequestStoreV1::new(&daemon));
        let request = request_for(&daemon);
        store.install_pending(request.clone()).unwrap();
        let submission = LocalApprovalSubmissionV1::for_request(
            &request,
            LocalApprovalDecisionKindV1::Approved,
            ms(1_200),
        )
        .unwrap();
        let (_parent, runtime) = private_runtime_path();
        let server = LocalApprovalSocketServerV1::bind_in(&runtime, &daemon).unwrap();

        let first_path = server.socket_path().to_path_buf();
        let first_submission = submission.clone();
        let first_client = thread::spawn(move || {
            submit_local_approval_v1(&first_path, &first_submission).unwrap()
        });
        server
            .accept_and_consume_at_v1(&store, ms(1_100), ms(1_300))
            .unwrap();
        first_client.join().unwrap();

        let replay_path = server.socket_path().to_path_buf();
        let replay_client = thread::spawn(move || {
            let mut stream = UnixStream::connect(&replay_path).unwrap();
            write_json_frame_v1(
                &mut stream,
                &LocalApprovalWireRequestV1::new(submission),
            )
            .unwrap();
            let _ = stream.shutdown(Shutdown::Both);
        });
        let err = server
            .accept_and_consume_at_v1(&store, ms(1_301), ms(1_302))
            .unwrap_err();
        replay_client.join().unwrap();
        assert!(matches!(
            err,
            LocalApprovalSocketErrorV1::RequestStore(
                LocalApprovalRequestStoreErrorV1::RequestNotPending
            )
        ));
        assert_eq!(store.pending_count().unwrap(), 0);
    }
}
