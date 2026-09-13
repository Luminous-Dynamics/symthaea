// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Composition of immutable worker executable identity with invocation-specific
//! pre-input cgroup-v2 placement.
//!
//! This crate deliberately does not redefine either underlying theorem:
//! `symthaea-sim-worker-image` still owns the meaning of the sealed executable
//! image, while `symthaea-sim-worker` / `symthaea-sim-worker-cgroup` own the
//! meaning of cgroup placement. This layer only proves that one exact technical
//! invocation satisfied both at once.

#![deny(unsafe_op_in_unsafe_fn)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read};
use std::os::fd::AsRawFd;
use symthaea_sim_bridge::SimulationRequest;
use symthaea_sim_worker::{
    CgroupPlacementEvidence, SupervisedInvocation, SupervisedWorker, SupervisorError,
    SupervisorLimits, WorkerFrameLimits,
};
use symthaea_sim_worker_cgroup::CgroupV2Lease;
use symthaea_sim_worker_image::{
    SealedWorkerImage, SealedWorkerImageEvidence, SEALED_WORKER_IMAGE_PROFILE_V1,
};
use thiserror::Error;

pub const SEALED_CGROUP_WORKER_PROFILE_V1: &str =
    "symthaea.simulation.worker-image-cgroup.pre-input-v1";
const COMPOSITE_DOMAIN_V1: &[u8] =
    b"symthaea.simulation.worker-image-cgroup.pre-input-v1\0";
const IMAGE_BINDING_DOMAIN_V1: &[u8] = b"symthaea.simulation.worker-image.memfd-sealed-v1\0";

#[cfg(target_os = "linux")]
const REQUIRED_SEALS: libc::c_int =
    libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedCgroupWorkerEvidence {
    pub profile: String,
    pub image: SealedWorkerImageEvidence,
    pub placement: CgroupPlacementEvidence,
    pub composite_sha256: String,
}

impl SealedCgroupWorkerEvidence {
    /// Verify serialized structure and all nested commitments. This is audit
    /// verification only; it cannot recreate either the sealed image, cgroup
    /// lease, or process authority.
    pub fn verify(&self) -> Result<(), SealedCgroupWorkerError> {
        if self.profile != SEALED_CGROUP_WORKER_PROFILE_V1 {
            return Err(SealedCgroupWorkerError::ProfileMismatch);
        }
        if self.image.profile != SEALED_WORKER_IMAGE_PROFILE_V1 {
            return Err(SealedCgroupWorkerError::ImageProfileMismatch);
        }
        let source = parse_hex_32(&self.image.source_sha256)?;
        let image = parse_hex_32(&self.image.image_sha256)?;
        let worker = parse_hex_32(&self.image.worker_sha256)?;
        if source != image || image != worker {
            return Err(SealedCgroupWorkerError::ImageDigestMismatch);
        }
        let image_binding = image_binding_sha256_v1(
            source,
            image,
            worker,
            self.image.image_size_bytes,
            self.image.seal_mask,
            &self.image.supervisor_profile,
            &self.image.worker_protocol,
        );
        if self.image.binding_sha256 != hex_digest(image_binding) {
            return Err(SealedCgroupWorkerError::ImageBindingDigestMismatch);
        }
        self.placement.verify()?;
        let expected = composite_sha256_v1(
            &self.image.binding_sha256,
            &self.placement.placement_sha256,
        )?;
        if self.composite_sha256 != hex_digest(expected) {
            return Err(SealedCgroupWorkerError::CompositeDigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct SealedCgroupWorkerInvocation {
    invocation: SupervisedInvocation,
    image_evidence: SealedWorkerImageEvidence,
    composite_sha256: [u8; 32],
}

impl SealedCgroupWorkerInvocation {
    pub fn invocation(&self) -> &SupervisedInvocation {
        &self.invocation
    }

    pub fn placement(&self) -> &CgroupPlacementEvidence {
        self.invocation
            .cgroup_placement()
            .expect("sealed cgroup invocation always has placement evidence")
    }

    pub const fn composite_sha256(&self) -> [u8; 32] {
        self.composite_sha256
    }

    pub fn image_evidence(&self) -> &SealedWorkerImageEvidence {
        &self.image_evidence
    }

    pub fn evidence(&self) -> SealedCgroupWorkerEvidence {
        SealedCgroupWorkerEvidence {
            profile: SEALED_CGROUP_WORKER_PROFILE_V1.into(),
            image: self.image_evidence.clone(),
            placement: self.placement().clone(),
            composite_sha256: hex_digest(self.composite_sha256),
        }
    }

    pub fn verify(&self) -> Result<(), SealedCgroupWorkerError> {
        let source = parse_hex_32(&self.image_evidence.source_sha256)?;
        let image = parse_hex_32(&self.image_evidence.image_sha256)?;
        let worker = parse_hex_32(&self.image_evidence.worker_sha256)?;
        if source != image || image != worker || worker != self.invocation.worker_sha256() {
            return Err(SealedCgroupWorkerError::WorkerDigestMismatch);
        }
        self.placement().verify()?;
        let image_binding = image_binding_sha256_v1(
            source,
            image,
            worker,
            self.image_evidence.image_size_bytes,
            self.image_evidence.seal_mask,
            self.invocation.supervisor_profile(),
            &self.invocation.worker().profile,
        );
        if self.image_evidence.binding_sha256 != hex_digest(image_binding) {
            return Err(SealedCgroupWorkerError::ImageBindingDigestMismatch);
        }
        let expected = composite_sha256_v1(
            &self.image_evidence.binding_sha256,
            &self.placement().placement_sha256,
        )?;
        if expected != self.composite_sha256 {
            return Err(SealedCgroupWorkerError::CompositeDigestMismatch);
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn execute_sealed_image_in_cgroup(
    image: &SealedWorkerImage,
    cgroup: &CgroupV2Lease,
    manifest_bytes: &[u8],
    component_bytes: &[u8],
    request: &SimulationRequest,
    limits: SupervisorLimits,
    frame_limits: WorkerFrameLimits,
) -> Result<SealedCgroupWorkerInvocation, SealedCgroupWorkerError> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (
            image,
            cgroup,
            manifest_bytes,
            component_bytes,
            request,
            limits,
            frame_limits,
        );
        return Err(SealedCgroupWorkerError::UnsupportedPlatform);
    }

    #[cfg(target_os = "linux")]
    {
        verify_live_image(image, limits)?;
        let worker = SupervisedWorker::new(image.exec_path())
            .with_limits(limits)?
            .with_frame_limits(frame_limits);
        let invocation = worker.execute_in_cgroup(
            manifest_bytes,
            component_bytes,
            request,
            cgroup,
        )?;
        if invocation.worker_sha256() != image.image_sha256() {
            return Err(SealedCgroupWorkerError::WorkerDigestMismatch);
        }
        let placement = invocation
            .cgroup_placement()
            .ok_or(SealedCgroupWorkerError::MissingPlacement)?;
        placement.verify()?;
        verify_live_image(image, limits)?;

        let image_binding = image_binding_sha256_v1(
            image.source_sha256(),
            image.image_sha256(),
            invocation.worker_sha256(),
            image.image_size_bytes(),
            image.seal_mask(),
            invocation.supervisor_profile(),
            &invocation.worker().profile,
        );
        let image_evidence = SealedWorkerImageEvidence {
            profile: SEALED_WORKER_IMAGE_PROFILE_V1.into(),
            source_sha256: hex_digest(image.source_sha256()),
            image_sha256: hex_digest(image.image_sha256()),
            worker_sha256: hex_digest(invocation.worker_sha256()),
            image_size_bytes: image.image_size_bytes(),
            seal_mask: image.seal_mask(),
            supervisor_profile: invocation.supervisor_profile().into(),
            worker_protocol: invocation.worker().profile.clone(),
            binding_sha256: hex_digest(image_binding),
        };
        let composite_sha256 = composite_sha256_v1(
            &image_evidence.binding_sha256,
            &placement.placement_sha256,
        )?;
        let result = SealedCgroupWorkerInvocation {
            invocation,
            image_evidence,
            composite_sha256,
        };
        result.verify()?;
        Ok(result)
    }
}

#[cfg(target_os = "linux")]
fn verify_live_image(
    image: &SealedWorkerImage,
    limits: SupervisorLimits,
) -> Result<(), SealedCgroupWorkerError> {
    limits.validate()?;
    if image.source_sha256() != image.image_sha256() {
        return Err(SealedCgroupWorkerError::ImageDigestMismatch);
    }
    let fd = image
        .exec_path()
        .file_name()
        .and_then(|value| value.to_str())
        .and_then(|value| value.parse::<i32>().ok())
        .ok_or(SealedCgroupWorkerError::InvalidProcFdPath)?;
    if fd < 0 || u64::try_from(fd).unwrap_or(u64::MAX) >= limits.open_files {
        return Err(SealedCgroupWorkerError::DescriptorOutsideNoFileLimit {
            fd,
            open_files: limits.open_files,
        });
    }

    let file = File::open(image.exec_path())?;
    let digest = sha256_reader(file)?;
    if digest != image.image_sha256() {
        return Err(SealedCgroupWorkerError::ImageDigestMismatch);
    }
    let file = File::open(image.exec_path())?;
    let seals = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
    if seals < 0 {
        return Err(SealedCgroupWorkerError::SealOperation(
            io::Error::last_os_error(),
        ));
    }
    if seals as u32 != image.seal_mask() || seals & REQUIRED_SEALS != REQUIRED_SEALS {
        return Err(SealedCgroupWorkerError::SealMismatch {
            expected: image.seal_mask(),
            actual: seals as u32,
        });
    }
    Ok(())
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

fn composite_sha256_v1(
    image_binding_sha256: &str,
    placement_sha256: &str,
) -> Result<[u8; 32], SealedCgroupWorkerError> {
    let image_binding = parse_hex_32(image_binding_sha256)?;
    let placement = parse_hex_32(placement_sha256)?;
    let mut hasher = Sha256::new();
    hasher.update(COMPOSITE_DOMAIN_V1);
    hasher.update(image_binding);
    hasher.update(placement);
    Ok(hasher.finalize().into())
}

fn sha256_reader(mut reader: impl Read) -> Result<[u8; 32], io::Error> {
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

fn put_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn parse_hex_32(value: &str) -> Result<[u8; 32], SealedCgroupWorkerError> {
    if value.len() != 64 {
        return Err(SealedCgroupWorkerError::InvalidHexDigest);
    }
    let bytes = value.as_bytes();
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(bytes.chunks_exact(2)) {
        *slot = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
    }
    Ok(out)
}

fn hex_nibble(value: u8) -> Result<u8, SealedCgroupWorkerError> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => Err(SealedCgroupWorkerError::InvalidHexDigest),
    }
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

#[derive(Debug, Error)]
pub enum SealedCgroupWorkerError {
    #[error("sealed cgroup worker composition is currently supported only on Linux")]
    UnsupportedPlatform,
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Supervisor(#[from] SupervisorError),
    #[error("sealed worker image profile mismatch")]
    ImageProfileMismatch,
    #[error("sealed+cgroup composite profile mismatch")]
    ProfileMismatch,
    #[error("sealed worker image digest relationship does not match")]
    ImageDigestMismatch,
    #[error("worker digest does not match the exact sealed image")]
    WorkerDigestMismatch,
    #[error("sealed image binding digest does not match image-v1 semantics")]
    ImageBindingDigestMismatch,
    #[error("cgroup placement evidence is missing")]
    MissingPlacement,
    #[error("sealed+cgroup composite digest does not match")]
    CompositeDigestMismatch,
    #[error("sealed worker executable path is not a canonical /proc/self/fd descriptor path")]
    InvalidProcFdPath,
    #[error("sealed worker fd {fd} is outside RLIMIT_NOFILE={open_files}")]
    DescriptorOutsideNoFileLimit { fd: i32, open_files: u64 },
    #[error("failed to read worker-image seals: {0}")]
    SealOperation(#[source] io::Error),
    #[error("sealed worker-image seals differ: expected={expected:#x}, actual={actual:#x}")]
    SealMismatch { expected: u32, actual: u32 },
    #[error("invalid canonical lowercase SHA-256 hex digest")]
    InvalidHexDigest,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trip_is_canonical() {
        let value = [0xabu8; 32];
        let encoded = hex_digest(value);
        assert_eq!(parse_hex_32(&encoded).unwrap(), value);
        assert!(parse_hex_32(&encoded.to_uppercase()).is_err());
    }

    #[test]
    fn composite_digest_is_domain_separated_and_ordered() {
        let a = hex_digest([0x11; 32]);
        let b = hex_digest([0x22; 32]);
        assert_ne!(
            composite_sha256_v1(&a, &b).unwrap(),
            composite_sha256_v1(&b, &a).unwrap()
        );
    }
}
