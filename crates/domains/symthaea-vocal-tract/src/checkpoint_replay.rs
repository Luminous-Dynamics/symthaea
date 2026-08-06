// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Restart-durable replay protection for supervised checkpoint protocols.
//!
//! The in-memory protocol guards remain useful for tests and single-process
//! deployments. This module provides the stronger contract: an authenticated,
//! bounded replay window whose updates are serialized across cooperating
//! processes and synchronized before an accepted request is reported.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

use crate::lock_exclusive;

pub const CHECKPOINT_REPLAY_STATE_SCHEMA: &str = "symthaea.checkpoint-replay-state.v1";
pub const KEY_AGENT_REPLAY_CONTEXT: CheckpointReplayContext =
    CheckpointReplayContext(*b"key-agent-replay");
pub const MONOTONIC_REPLAY_CONTEXT: CheckpointReplayContext =
    CheckpointReplayContext(*b"monotonic-replay");
pub const MAX_DURABLE_REPLAY_ENTRIES: usize = 65_536;
const MAX_REPLAY_STATE_BYTES: u64 = 8 * 1024 * 1024;
const REPLAY_STATE_NAME: &str = "checkpoint-replay.state";
const REPLAY_LOCK_NAME: &str = ".checkpoint-replay.lock";
const REPLAY_AUTH_DOMAIN: &[u8] = b"symthaea-checkpoint-replay-state-v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointReplayContext(pub [u8; 16]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CheckpointReplayProtectionLevel {
    ProcessLocal,
    RestartDurable,
}

#[derive(Debug)]
pub enum CheckpointReplayError {
    InvalidKey,
    InvalidCapacity,
    ZeroRequestId,
    Replay,
    InvalidState,
    TooLarge,
    Encoding,
    Unavailable(&'static str),
    Io(std::io::Error),
}

impl CheckpointReplayError {
    pub fn is_replay_or_invalid_id(&self) -> bool {
        matches!(self, Self::Replay | Self::ZeroRequestId)
    }
}

impl std::fmt::Display for CheckpointReplayError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidKey => write!(formatter, "checkpoint replay-state key is invalid"),
            Self::InvalidCapacity => write!(formatter, "checkpoint replay capacity is invalid"),
            Self::ZeroRequestId => write!(formatter, "checkpoint request identifier is zero"),
            Self::Replay => write!(formatter, "checkpoint request identifier was replayed"),
            Self::InvalidState => write!(formatter, "checkpoint replay state is invalid"),
            Self::TooLarge => write!(formatter, "checkpoint replay state exceeds its bound"),
            Self::Encoding => write!(formatter, "checkpoint replay state encoding failed"),
            Self::Unavailable(reason) => write!(
                formatter,
                "checkpoint replay state is unavailable: {reason}"
            ),
            Self::Io(error) => write!(formatter, "checkpoint replay-state I/O failed: {error}"),
        }
    }
}

impl std::error::Error for CheckpointReplayError {}

impl From<std::io::Error> for CheckpointReplayError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub trait CheckpointRequestReplayProtector: Send + Sync {
    fn protection_level(&self) -> CheckpointReplayProtectionLevel;

    fn verify_and_record(&self, request_id: [u8; 16]) -> Result<(), CheckpointReplayError>;
}

pub struct CheckpointReplayStateKey([u8; 32]);

impl CheckpointReplayStateKey {
    pub fn new(bytes: [u8; 32]) -> Result<Self, CheckpointReplayError> {
        if bytes.iter().all(|byte| *byte == 0) {
            return Err(CheckpointReplayError::InvalidKey);
        }
        Ok(Self(bytes))
    }

    pub fn generate() -> Result<Self, CheckpointReplayError> {
        let mut bytes = [0u8; 32];
        getrandom::fill(&mut bytes).map_err(|_| CheckpointReplayError::Unavailable("entropy"))?;
        let result = Self::new(bytes);
        bytes.zeroize();
        result
    }
}

impl Drop for CheckpointReplayStateKey {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DurableReplayState {
    schema: String,
    context: CheckpointReplayContext,
    capacity: u32,
    order: Vec<[u8; 16]>,
    sorted_membership: Vec<[u8; 16]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DurableReplayStateWire {
    body: Vec<u8>,
    authentication_tag: [u8; 32],
}

/// Authenticated replay window anchored to an opened directory capability.
///
/// Replayed identifiers are found by binary search. Accepted unique identifiers
/// may require bounded vector insertion and an atomic state rewrite; callers
/// should combine this guard with the protocol request and connection budgets.
pub struct DurableCheckpointReplayGuard {
    root: PathBuf,
    context: CheckpointReplayContext,
    capacity: usize,
    key: CheckpointReplayStateKey,
    local_lock: Mutex<()>,
    pinned_root: Mutex<Option<Arc<File>>>,
}

impl DurableCheckpointReplayGuard {
    pub fn new(
        root: impl Into<PathBuf>,
        context: CheckpointReplayContext,
        capacity: usize,
        key: CheckpointReplayStateKey,
    ) -> Result<Self, CheckpointReplayError> {
        if capacity == 0 || capacity > MAX_DURABLE_REPLAY_ENTRIES {
            return Err(CheckpointReplayError::InvalidCapacity);
        }
        Ok(Self {
            root: root.into(),
            context,
            capacity,
            key,
            local_lock: Mutex::new(()),
            pinned_root: Mutex::new(None),
        })
    }

    pub fn context(&self) -> CheckpointReplayContext {
        self.context
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn retained_request_count(&self) -> Result<usize, CheckpointReplayError> {
        let _local = self
            .local_lock
            .lock()
            .map_err(|_| CheckpointReplayError::Unavailable("replay lock poisoned"))?;
        let lock_file = self.open_lock_file()?;
        let _kernel = lock_exclusive(&lock_file)
            .map_err(|_| CheckpointReplayError::Unavailable("replay kernel lock"))?;
        Ok(self.read_state_locked()?.order.len())
    }

    fn verify_and_record_locked(&self, request_id: [u8; 16]) -> Result<(), CheckpointReplayError> {
        if request_id == [0u8; 16] {
            return Err(CheckpointReplayError::ZeroRequestId);
        }
        let _local = self
            .local_lock
            .lock()
            .map_err(|_| CheckpointReplayError::Unavailable("replay lock poisoned"))?;
        let lock_file = self.open_lock_file()?;
        let _kernel = lock_exclusive(&lock_file)
            .map_err(|_| CheckpointReplayError::Unavailable("replay kernel lock"))?;
        let mut state = self.read_state_locked()?;
        match state.sorted_membership.binary_search(&request_id) {
            Ok(_) => return Err(CheckpointReplayError::Replay),
            Err(_insertion) => {
                if state.order.len() == self.capacity {
                    let expired = state.order.remove(0);
                    let index = state
                        .sorted_membership
                        .binary_search(&expired)
                        .map_err(|_| CheckpointReplayError::InvalidState)?;
                    state.sorted_membership.remove(index);
                }
                let insertion = state
                    .sorted_membership
                    .binary_search(&request_id)
                    .unwrap_err();
                state.sorted_membership.insert(insertion, request_id);
                state.order.push(request_id);
            }
        }
        self.write_state_locked(&state)
    }

    fn read_state_locked(&self) -> Result<DurableReplayState, CheckpointReplayError> {
        let path = self.operation_root_path()?.join(REPLAY_STATE_NAME);
        let file = match open_regular_file(&path, false, false) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(self.empty_state());
            }
            Err(error) => return Err(error.into()),
        };
        let metadata = file.metadata()?;
        if metadata.len() == 0 || metadata.len() > MAX_REPLAY_STATE_BYTES {
            return Err(CheckpointReplayError::TooLarge);
        }
        let mut encoded = Vec::with_capacity(metadata.len() as usize);
        file.take(MAX_REPLAY_STATE_BYTES.saturating_add(1))
            .read_to_end(&mut encoded)?;
        if encoded.is_empty() || encoded.len() as u64 > MAX_REPLAY_STATE_BYTES {
            return Err(CheckpointReplayError::TooLarge);
        }
        let wire: DurableReplayStateWire =
            postcard::from_bytes(&encoded).map_err(|_| CheckpointReplayError::Encoding)?;
        if !constant_time_equal(
            &wire.authentication_tag,
            &authenticate_state(&wire.body, &self.key),
        ) {
            return Err(CheckpointReplayError::InvalidState);
        }
        let state: DurableReplayState =
            postcard::from_bytes(&wire.body).map_err(|_| CheckpointReplayError::Encoding)?;
        self.validate_state(&state)?;
        Ok(state)
    }

    fn write_state_locked(&self, state: &DurableReplayState) -> Result<(), CheckpointReplayError> {
        self.validate_state(state)?;
        let body = postcard::to_stdvec(state).map_err(|_| CheckpointReplayError::Encoding)?;
        let wire = DurableReplayStateWire {
            authentication_tag: authenticate_state(&body, &self.key),
            body,
        };
        let encoded = postcard::to_stdvec(&wire).map_err(|_| CheckpointReplayError::Encoding)?;
        if encoded.is_empty() || encoded.len() as u64 > MAX_REPLAY_STATE_BYTES {
            return Err(CheckpointReplayError::TooLarge);
        }
        let root = self.ensure_root()?;
        let operation_root = self.operation_root_path()?;
        let mut nonce = [0u8; 16];
        getrandom::fill(&mut nonce).map_err(|_| CheckpointReplayError::Unavailable("entropy"))?;
        let suffix = nonce
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let temp = operation_root.join(format!(
            ".checkpoint-replay-{}-{suffix}.tmp",
            std::process::id(),
        ));
        let target = operation_root.join(REPLAY_STATE_NAME);
        let result = (|| {
            let mut file = open_regular_file(&temp, true, true)?;
            file.write_all(&encoded)?;
            file.sync_all()?;
            fs::rename(&temp, &target)?;
            root.sync_all()?;
            Ok::<(), CheckpointReplayError>(())
        })();
        let _ = fs::remove_file(&temp);
        result
    }

    fn empty_state(&self) -> DurableReplayState {
        DurableReplayState {
            schema: CHECKPOINT_REPLAY_STATE_SCHEMA.to_owned(),
            context: self.context,
            capacity: self.capacity as u32,
            order: Vec::new(),
            sorted_membership: Vec::new(),
        }
    }

    fn validate_state(&self, state: &DurableReplayState) -> Result<(), CheckpointReplayError> {
        if state.schema != CHECKPOINT_REPLAY_STATE_SCHEMA
            || state.context != self.context
            || state.capacity as usize != self.capacity
            || state.order.len() > self.capacity
            || state.sorted_membership.len() != state.order.len()
            || state.order.contains(&[0u8; 16])
            || !state
                .sorted_membership
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        {
            return Err(CheckpointReplayError::InvalidState);
        }
        for request_id in &state.order {
            if state.sorted_membership.binary_search(request_id).is_err() {
                return Err(CheckpointReplayError::InvalidState);
            }
        }
        Ok(())
    }

    fn open_lock_file(&self) -> Result<File, CheckpointReplayError> {
        let path = self.operation_root_path()?.join(REPLAY_LOCK_NAME);
        open_regular_file(&path, true, false).map_err(Into::into)
    }

    fn ensure_root(&self) -> Result<Arc<File>, CheckpointReplayError> {
        let mut pinned = self
            .pinned_root
            .lock()
            .map_err(|_| CheckpointReplayError::Unavailable("replay root lock poisoned"))?;
        if let Some(root) = pinned.as_ref() {
            return Ok(Arc::clone(root));
        }
        fs::create_dir_all(&self.root)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
            fs::set_permissions(&self.root, fs::Permissions::from_mode(0o700))?;
            let mut options = OpenOptions::new();
            options
                .read(true)
                .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW);
            let root = Arc::new(options.open(&self.root)?);
            if !root.metadata()?.is_dir() {
                return Err(CheckpointReplayError::InvalidState);
            }
            *pinned = Some(Arc::clone(&root));
            Ok(root)
        }
        #[cfg(not(unix))]
        {
            let root = Arc::new(File::open(&self.root)?);
            *pinned = Some(Arc::clone(&root));
            Ok(root)
        }
    }

    fn operation_root_path(&self) -> Result<PathBuf, CheckpointReplayError> {
        let root = self.ensure_root()?;
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            let path = PathBuf::from(format!("/proc/self/fd/{}", root.as_raw_fd()));
            if !path.is_dir() {
                return Err(CheckpointReplayError::Unavailable(
                    "descriptor-relative replay root unavailable",
                ));
            }
            Ok(path)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = root;
            Ok(self.root.clone())
        }
    }
}

impl CheckpointRequestReplayProtector for DurableCheckpointReplayGuard {
    fn protection_level(&self) -> CheckpointReplayProtectionLevel {
        CheckpointReplayProtectionLevel::RestartDurable
    }

    fn verify_and_record(&self, request_id: [u8; 16]) -> Result<(), CheckpointReplayError> {
        self.verify_and_record_locked(request_id)
    }
}

fn authenticate_state(body: &[u8], key: &CheckpointReplayStateKey) -> [u8; 32] {
    let mut input = Vec::with_capacity(REPLAY_AUTH_DOMAIN.len() + body.len());
    input.extend_from_slice(REPLAY_AUTH_DOMAIN);
    input.extend_from_slice(body);
    *blake3::keyed_hash(&key.0, &input).as_bytes()
}

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0u8, |difference, (left, right)| difference | (left ^ right))
        == 0
}

fn open_regular_file(path: &Path, create: bool, create_new: bool) -> std::io::Result<File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut options = OpenOptions::new();
        options
            .read(true)
            .write(true)
            .create(create)
            .create_new(create_new)
            .mode(0o600)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
        let file = options.open(path)?;
        if !file.metadata()?.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "checkpoint replay object is not a regular file",
            ));
        }
        Ok(file)
    }
    #[cfg(not(unix))]
    {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(create)
            .create_new(create_new)
            .open(path)?;
        if !file.metadata()?.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "checkpoint replay object is not a regular file",
            ));
        }
        Ok(file)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_root(label: &str) -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("symthaea-{label}-{}-{suffix}", std::process::id(),))
    }

    #[test]
    fn durable_window_rejects_replay_after_restart() {
        let root = temporary_root("durable-replay");
        let request = [0x31; 16];
        DurableCheckpointReplayGuard::new(
            &root,
            KEY_AGENT_REPLAY_CONTEXT,
            4,
            CheckpointReplayStateKey::new([0x41; 32]).unwrap(),
        )
        .unwrap()
        .verify_and_record(request)
        .unwrap();
        let restarted = DurableCheckpointReplayGuard::new(
            &root,
            KEY_AGENT_REPLAY_CONTEXT,
            4,
            CheckpointReplayStateKey::new([0x41; 32]).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            restarted.verify_and_record(request),
            Err(CheckpointReplayError::Replay)
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn bounded_window_evicts_oldest_identifier() {
        let root = temporary_root("durable-replay-eviction");
        let guard = DurableCheckpointReplayGuard::new(
            &root,
            MONOTONIC_REPLAY_CONTEXT,
            2,
            CheckpointReplayStateKey::new([0x42; 32]).unwrap(),
        )
        .unwrap();
        guard.verify_and_record([1; 16]).unwrap();
        guard.verify_and_record([2; 16]).unwrap();
        guard.verify_and_record([3; 16]).unwrap();
        guard.verify_and_record([1; 16]).unwrap();
        assert_eq!(guard.retained_request_count().unwrap(), 2);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn wrong_key_and_wrong_context_fail_closed() {
        let root = temporary_root("durable-replay-binding");
        DurableCheckpointReplayGuard::new(
            &root,
            KEY_AGENT_REPLAY_CONTEXT,
            4,
            CheckpointReplayStateKey::new([0x43; 32]).unwrap(),
        )
        .unwrap()
        .verify_and_record([9; 16])
        .unwrap();
        let wrong_key = DurableCheckpointReplayGuard::new(
            &root,
            KEY_AGENT_REPLAY_CONTEXT,
            4,
            CheckpointReplayStateKey::new([0x44; 32]).unwrap(),
        )
        .unwrap();
        assert!(wrong_key.verify_and_record([8; 16]).is_err());
        let wrong_context = DurableCheckpointReplayGuard::new(
            &root,
            MONOTONIC_REPLAY_CONTEXT,
            4,
            CheckpointReplayStateKey::new([0x43; 32]).unwrap(),
        )
        .unwrap();
        assert!(wrong_context.verify_and_record([8; 16]).is_err());
        let _ = fs::remove_dir_all(root);
    }
}
