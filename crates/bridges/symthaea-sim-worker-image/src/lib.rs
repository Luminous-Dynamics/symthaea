// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Immutable executable identity for supervised simulation workers.
//!
//! The path-based supervisor in `symthaea-sim-worker` hashes its worker before
//! and after execution, but an ordinary filesystem path still has a narrow
//! substitution race between hashing and `exec`. This crate closes that identity
//! gap on Linux by copying the selected worker bytes into a sealable `memfd`,
//! making that descriptor executable, applying immutable content seals, and then
//! invoking the existing supervisor through `/proc/self/fd/<n>` while the exact
//! descriptor remains open across `exec`.
//!
//! This establishes executable-byte identity, not a complete hostile-code
//! sandbox. Seccomp, network/mount/user namespaces and cgroups remain separate
//! containment claims.

#![deny(unsafe_op_in_unsafe_fn)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::ffi::CString;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use symthaea_sim_bridge::SimulationRequest;
use symthaea_sim_worker::{
    SupervisedInvocation, SupervisedWorker, SupervisorError, SupervisorLimits, WorkerFrameLimits,
};
use thiserror::Error;

/// Versioned identity for the executable-image binding implemented here.
pub const SEALED_WORKER_IMAGE_PROFILE_V1: &str =
    "symthaea.simulation.worker-image.memfd-sealed-v1";

const IMAGE_BINDING_DOMAIN_V1: &[u8] = b"symthaea.simulation.worker-image.memfd-sealed-v1\0";
const MAX_WORKER_IMAGE_BYTES: u64 = 512 * 1024 * 1024;

#[cfg(target_os = "linux")]
const REQUIRED_SEALS: libc::c_int =
    libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;

/// Persistable audit evidence for one sealed-image worker invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedWorkerImageEvidence {
    pub profile: String,
    pub source_sha256: String,
    pub image_sha256: String,
    pub worker_sha256: String,
    pub image_size_bytes: u64,
    pub seal_mask: u32,
    pub supervisor_profile: String,
    pub worker_protocol: String,
    pub binding_sha256: String,
}

/// Exact worker bytes copied into an immutable executable `memfd`.
#[derive(Debug)]
pub struct SealedWorkerImage {
    #[cfg(target_os = "linux")]
    image: File,
    exec_path: PathBuf,
    source_sha256: [u8; 32],
    image_sha256: [u8; 32],
    image_size_bytes: u64,
    seal_mask: u32,
}

impl SealedWorkerImage {
    /// Copy an exact worker executable into a sealed anonymous executable image.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, SealedWorkerImageError> {
        #[cfg(not(target_os = "linux"))]
        {
            let _ = path;
            return Err(SealedWorkerImageError::UnsupportedPlatform);
        }

        #[cfg(target_os = "linux")]
        Self::from_path_linux(path.as_ref())
    }

    #[cfg(target_os = "linux")]
    fn from_path_linux(path: &Path) -> Result<Self, SealedWorkerImageError> {
        let metadata = fs::metadata(path)?;
        if !metadata.is_file() {
            return Err(SealedWorkerImageError::SourceNotRegularFile);
        }
        if metadata.len() == 0 || metadata.len() > MAX_WORKER_IMAGE_BYTES {
            return Err(SealedWorkerImageError::SourceSize {
                actual: metadata.len(),
                maximum: MAX_WORKER_IMAGE_BYTES,
            });
        }

        let bytes = fs::read(path)?;
        let source_sha256: [u8; 32] = Sha256::digest(&bytes).into();
        let name = CString::new("symthaea-sim-worker-image").expect("static memfd name");
        // Intentionally omit MFD_CLOEXEC: /proc/self/fd/<n> must remain valid in
        // the forked child until the kernel has resolved the executable image.
        let raw_fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_ALLOW_SEALING) };
        if raw_fd < 0 {
            return Err(SealedWorkerImageError::MemfdCreate(
                io::Error::last_os_error(),
            ));
        }
        let owned = unsafe { OwnedFd::from_raw_fd(raw_fd) };
        let mut image = File::from(owned);

        image.write_all(&bytes)?;
        image.flush()?;
        image.seek(SeekFrom::Start(0))?;
        let copied_sha256 = sha256_reader(&mut image)?;
        if copied_sha256 != source_sha256 {
            return Err(SealedWorkerImageError::CopyDigestMismatch);
        }

        if unsafe { libc::fchmod(image.as_raw_fd(), 0o500) } != 0 {
            return Err(SealedWorkerImageError::ImagePermission(
                io::Error::last_os_error(),
            ));
        }
        if unsafe { libc::fcntl(image.as_raw_fd(), libc::F_ADD_SEALS, REQUIRED_SEALS) } < 0 {
            return Err(SealedWorkerImageError::SealOperation(
                io::Error::last_os_error(),
            ));
        }
        let seals = unsafe { libc::fcntl(image.as_raw_fd(), libc::F_GET_SEALS) };
        if seals < 0 {
            return Err(SealedWorkerImageError::SealOperation(
                io::Error::last_os_error(),
            ));
        }
        if seals & REQUIRED_SEALS != REQUIRED_SEALS {
            return Err(SealedWorkerImageError::MissingRequiredSeals {
                actual: seals as u32,
                required: REQUIRED_SEALS as u32,
            });
        }

        image.seek(SeekFrom::Start(0))?;
        let image_sha256 = sha256_reader(&mut image)?;
        image.seek(SeekFrom::Start(0))?;
        if image_sha256 != source_sha256 {
            return Err(SealedWorkerImageError::SealedDigestMismatch);
        }

        let exec_path = PathBuf::from(format!("/proc/self/fd/{}", image.as_raw_fd()));
        if sha256_path(&exec_path)? != image_sha256 {
            return Err(SealedWorkerImageError::ProcFdDigestMismatch);
        }

        Ok(Self {
            image,
            exec_path,
            source_sha256,
            image_sha256,
            image_size_bytes: metadata.len(),
            seal_mask: seals as u32,
        })
    }

    pub const fn source_sha256(&self) -> [u8; 32] {
        self.source_sha256
    }

    pub const fn image_sha256(&self) -> [u8; 32] {
        self.image_sha256
    }

    pub const fn image_size_bytes(&self) -> u64 {
        self.image_size_bytes
    }

    pub const fn seal_mask(&self) -> u32 {
        self.seal_mask
    }

    pub fn exec_path(&self) -> &Path {
        &self.exec_path
    }

    /// Execute with the parent supervisor's default limits and frame ceilings.
    pub fn execute(
        &self,
        manifest_bytes: &[u8],
        component_bytes: &[u8],
        request: &SimulationRequest,
    ) -> Result<SealedWorkerInvocation, SealedWorkerImageError> {
        self.execute_with_limits(
            manifest_bytes,
            component_bytes,
            request,
            SupervisorLimits::default(),
            WorkerFrameLimits::default(),
        )
    }

    /// Execute while preserving an explicit supervisor and protocol envelope.
    pub fn execute_with_limits(
        &self,
        manifest_bytes: &[u8],
        component_bytes: &[u8],
        request: &SimulationRequest,
        limits: SupervisorLimits,
        frame_limits: WorkerFrameLimits,
    ) -> Result<SealedWorkerInvocation, SealedWorkerImageError> {
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (
                manifest_bytes,
                component_bytes,
                request,
                limits,
                frame_limits,
            );
            return Err(SealedWorkerImageError::UnsupportedPlatform);
        }

        #[cfg(target_os = "linux")]
        {
            self.verify_current_image()?;
            let fd = self.image.as_raw_fd();
            if fd < 0 || u64::try_from(fd).unwrap_or(u64::MAX) >= limits.open_files {
                return Err(SealedWorkerImageError::DescriptorOutsideNoFileLimit {
                    fd,
                    open_files: limits.open_files,
                });
            }

            let worker = SupervisedWorker::new(&self.exec_path)
                .with_limits(limits)?
                .with_frame_limits(frame_limits);
            let invocation = worker.execute(manifest_bytes, component_bytes, request)?;

            if invocation.worker_sha256() != self.image_sha256 {
                return Err(SealedWorkerImageError::WorkerDigestMismatch);
            }
            self.verify_current_image()?;

            SealedWorkerInvocation::new(
                invocation,
                self.source_sha256,
                self.image_sha256,
                self.image_size_bytes,
                self.seal_mask,
            )
        }
    }

    #[cfg(target_os = "linux")]
    fn verify_current_image(&self) -> Result<(), SealedWorkerImageError> {
        let seals = unsafe { libc::fcntl(self.image.as_raw_fd(), libc::F_GET_SEALS) };
        if seals < 0 {
            return Err(SealedWorkerImageError::SealOperation(
                io::Error::last_os_error(),
            ));
        }
        if seals & REQUIRED_SEALS != REQUIRED_SEALS || seals as u32 != self.seal_mask {
            return Err(SealedWorkerImageError::SealDrift {
                expected: self.seal_mask,
                actual: seals as u32,
            });
        }
        if sha256_path(&self.exec_path)? != self.image_sha256 {
            return Err(SealedWorkerImageError::ImageDigestDrift);
        }
        Ok(())
    }
}

/// Supervised technical invocation plus independently sealed executable identity.
#[derive(Debug)]
pub struct SealedWorkerInvocation {
    invocation: SupervisedInvocation,
    source_sha256: [u8; 32],
    image_sha256: [u8; 32],
    image_size_bytes: u64,
    seal_mask: u32,
    binding_sha256: [u8; 32],
}

impl SealedWorkerInvocation {
    fn new(
        invocation: SupervisedInvocation,
        source_sha256: [u8; 32],
        image_sha256: [u8; 32],
        image_size_bytes: u64,
        seal_mask: u32,
    ) -> Result<Self, SealedWorkerImageError> {
        if source_sha256 != image_sha256 || invocation.worker_sha256() != image_sha256 {
            return Err(SealedWorkerImageError::WorkerDigestMismatch);
        }
        let binding_sha256 = image_binding_sha256_v1(
            source_sha256,
            image_sha256,
            invocation.worker_sha256(),
            image_size_bytes,
            seal_mask,
            invocation.supervisor_profile(),
            &invocation.worker().profile,
        );
        Ok(Self {
            invocation,
            source_sha256,
            image_sha256,
            image_size_bytes,
            seal_mask,
            binding_sha256,
        })
    }

    pub fn invocation(&self) -> &SupervisedInvocation {
        &self.invocation
    }

    pub const fn binding_sha256(&self) -> [u8; 32] {
        self.binding_sha256
    }

    pub fn evidence(&self) -> SealedWorkerImageEvidence {
        SealedWorkerImageEvidence {
            profile: SEALED_WORKER_IMAGE_PROFILE_V1.into(),
            source_sha256: hex_digest(self.source_sha256),
            image_sha256: hex_digest(self.image_sha256),
            worker_sha256: hex_digest(self.invocation.worker_sha256()),
            image_size_bytes: self.image_size_bytes,
            seal_mask: self.seal_mask,
            supervisor_profile: self.invocation.supervisor_profile().into(),
            worker_protocol: self.invocation.worker().profile.clone(),
            binding_sha256: hex_digest(self.binding_sha256),
        }
    }

    /// Re-verify the immutable identity relationship recorded by this wrapper.
    pub fn verify(&self) -> Result<(), SealedWorkerImageError> {
        if self.source_sha256 != self.image_sha256
            || self.invocation.worker_sha256() != self.image_sha256
        {
            return Err(SealedWorkerImageError::WorkerDigestMismatch);
        }
        let expected = image_binding_sha256_v1(
            self.source_sha256,
            self.image_sha256,
            self.invocation.worker_sha256(),
            self.image_size_bytes,
            self.seal_mask,
            self.invocation.supervisor_profile(),
            &self.invocation.worker().profile,
        );
        if expected != self.binding_sha256 {
            return Err(SealedWorkerImageError::BindingDigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum SealedWorkerImageError {
    #[error("sealed worker images are currently supported only on Linux")]
    UnsupportedPlatform,
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("worker source path is not a regular file")]
    SourceNotRegularFile,
    #[error("worker source size {actual} is outside the supported range 1..={maximum}")]
    SourceSize { actual: u64, maximum: u64 },
    #[error("memfd_create failed: {0}")]
    MemfdCreate(#[source] io::Error),
    #[error("failed to make sealed worker image executable: {0}")]
    ImagePermission(#[source] io::Error),
    #[error("failed to apply/read worker-image seals: {0}")]
    SealOperation(#[source] io::Error),
    #[error("worker copy digest changed while constructing the image")]
    CopyDigestMismatch,
    #[error("sealed worker image digest does not match source bytes")]
    SealedDigestMismatch,
    #[error("/proc/self/fd executable view does not match the sealed image")]
    ProcFdDigestMismatch,
    #[error("required worker-image seals are missing: actual={actual:#x}, required={required:#x}")]
    MissingRequiredSeals { actual: u32, required: u32 },
    #[error("worker-image seals drifted: expected={expected:#x}, actual={actual:#x}")]
    SealDrift { expected: u32, actual: u32 },
    #[error("sealed worker image digest drifted")]
    ImageDigestDrift,
    #[error("sealed worker fd {fd} is outside RLIMIT_NOFILE={open_files}")]
    DescriptorOutsideNoFileLimit { fd: i32, open_files: u64 },
    #[error(transparent)]
    Supervisor(#[from] SupervisorError),
    #[error("executed worker digest does not equal the sealed image digest")]
    WorkerDigestMismatch,
    #[error("sealed worker image binding digest does not match")]
    BindingDigestMismatch,
}

fn sha256_reader(reader: &mut impl Read) -> Result<[u8; 32], io::Error> {
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().into())
}

fn sha256_path(path: &Path) -> Result<[u8; 32], io::Error> {
    let mut file = File::open(path)?;
    sha256_reader(&mut file)
}

fn image_binding_sha256_v1(
    source_sha256: [u8; 32],
    image_sha256: [u8; 32],
    worker_sha256: [u8; 32],
    image_size_bytes: u64,
    seal_mask: u32,
    supervisor_profile: &str,
    worker_protocol: &str,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(IMAGE_BINDING_DOMAIN_V1);
    hasher.update(source_sha256);
    hasher.update(image_sha256);
    hasher.update(worker_sha256);
    hasher.update(image_size_bytes.to_le_bytes());
    hasher.update(seal_mask.to_le_bytes());
    put_string(&mut hasher, supervisor_profile);
    put_string(&mut hasher, worker_protocol);
    hasher.finalize().into()
}

fn put_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: [u8; 32]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push(TABLE[(byte >> 4) as usize] as char);
        out.push(TABLE[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::process;

    #[test]
    fn content_seals_reject_write_and_truncate() {
        let exe = std::env::current_exe().unwrap();
        let image = SealedWorkerImage::from_path(exe).unwrap();
        assert_eq!(
            image.seal_mask() & REQUIRED_SEALS as u32,
            REQUIRED_SEALS as u32
        );

        match OpenOptions::new().read(true).write(true).open(image.exec_path()) {
            Ok(mut writable) => {
                assert!(writable.write_all(b"mutation").is_err());
                assert!(writable.set_len(0).is_err());
            }
            Err(_) => {
                // Opening the sealed executable writable may itself fail closed.
            }
        }
        image.verify_current_image().unwrap();
    }

    #[test]
    fn source_path_can_change_without_changing_sealed_image() {
        let source = std::env::current_exe().unwrap();
        let temp = std::env::temp_dir().join(format!(
            "symthaea-sealed-worker-source-{}-{}",
            process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        fs::copy(&source, &temp).unwrap();
        let image = SealedWorkerImage::from_path(&temp).unwrap();
        let digest = image.image_sha256();

        fs::write(&temp, b"source path replaced after sealing").unwrap();
        assert_eq!(image.image_sha256(), digest);
        image.verify_current_image().unwrap();
        let _ = fs::remove_file(temp);
    }
}
