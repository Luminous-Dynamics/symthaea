// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Read-only, observer-sealed content commitments for systemd unit sources.
//!
//! The input is an exact FragmentPath/DropInPaths identity already obtained
//! from the systemd observer. The module returns digests and file identity
//! metadata, never file contents.

use super::post_state::{NixPostStateErrorV1, NixSystemdUnitDefinitionIdentityV1};
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use thiserror::Error;

const DEFINITION_CONTENT_DOMAIN_V1: &[u8] = b"nixward-systemd-definition-content-v1";
const DEFINITION_FILE_DOMAIN_V1: &[u8] = b"nixward-systemd-definition-file-v1";
const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_FILES: usize = 65;
const MAX_PATH_BYTES: usize = 4096;

#[cfg(target_os = "linux")]
const RESOLVE_IN_ROOT: u64 = 0x10;
#[cfg(target_os = "linux")]
const RESOLVE_NO_MAGICLINKS: u64 = 0x02;

#[cfg(target_os = "linux")]
#[repr(C)]
struct OpenHowV1 {
    flags: u64,
    mode: u64,
    resolve: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixDefinitionFileContentDigestV1 {
    pub path: String,
    pub content_digest: String,
    pub size_bytes: u64,
    pub device_id: u64,
    pub inode: u64,
    pub mtime_seconds: i64,
    pub mtime_nanoseconds: i64,
    pub ctime_seconds: i64,
    pub ctime_nanoseconds: i64,
}

impl NixDefinitionFileContentDigestV1 {
    pub fn validate_shape(&self) -> Result<(), NixSystemdDefinitionContentErrorV1> {
        validate_path(&self.path)?;
        validate_digest(&self.content_digest)?;
        if self.size_bytes > MAX_FILE_BYTES || self.inode == 0 {
            return Err(NixSystemdDefinitionContentErrorV1::InvalidFileIdentity);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixSystemdDefinitionContentCommitmentV1 {
    pub unit: String,
    pub definition_identity: NixSystemdUnitDefinitionIdentityV1,
    pub manager_owner: String,
    pub files: Vec<NixDefinitionFileContentDigestV1>,
    pub overall_digest: String,
}

impl NixSystemdDefinitionContentCommitmentV1 {
    pub fn validate_shape(&self) -> Result<(), NixSystemdDefinitionContentErrorV1> {
        super::service_domain::NixServiceOperationV1::new(
            self.unit.clone(),
            super::service_domain::NixServiceOperationKindV1::Start,
        )
        .map_err(|error| {
            NixSystemdDefinitionContentErrorV1::InvalidServiceUnit(error.to_string())
        })?;
        self.definition_identity
            .validate_shape()
            .map_err(NixSystemdDefinitionContentErrorV1::InvalidDefinitionIdentity)?;
        validate_manager_owner(&self.manager_owner)?;
        if self.files.is_empty() || self.files.len() > MAX_FILES {
            return Err(NixSystemdDefinitionContentErrorV1::InvalidFileCount);
        }
        for file in &self.files {
            file.validate_shape()?;
        }
        for pair in self.files.windows(2) {
            if pair[0].path >= pair[1].path {
                return Err(NixSystemdDefinitionContentErrorV1::UnsortedOrDuplicatePaths);
            }
        }
        validate_digest(&self.overall_digest)?;
        if self.overall_digest != compute_overall_digest(&self.files)? {
            return Err(NixSystemdDefinitionContentErrorV1::OverallDigestMismatch);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, NixSystemdDefinitionContentErrorV1> {
        self.validate_shape()?;
        Ok(self.overall_digest.clone())
    }
}

pub struct NixVerifiedSystemdDefinitionContentCommitmentV1 {
    commitment: NixSystemdDefinitionContentCommitmentV1,
}

impl std::fmt::Debug for NixVerifiedSystemdDefinitionContentCommitmentV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NixVerifiedSystemdDefinitionContentCommitmentV1")
            .field("digest", &self.commitment.overall_digest)
            .finish_non_exhaustive()
    }
}

impl NixVerifiedSystemdDefinitionContentCommitmentV1 {
    pub(crate) fn from_observer(
        unit: &str,
        identity: &NixSystemdUnitDefinitionIdentityV1,
        manager_owner: &str,
    ) -> Result<Self, NixSystemdDefinitionContentErrorV1> {
        identity
            .validate_shape()
            .map_err(NixSystemdDefinitionContentErrorV1::InvalidDefinitionIdentity)?;

        let mut paths = Vec::with_capacity(1 + identity.drop_in_paths.len());
        paths.push(identity.fragment_path.clone());
        paths.extend(identity.drop_in_paths.iter().cloned());
        paths.sort();
        paths.dedup();
        if paths.is_empty() || paths.len() > MAX_FILES {
            return Err(NixSystemdDefinitionContentErrorV1::InvalidFileCount);
        }

        let mut files = Vec::with_capacity(paths.len());
        for path in paths {
            files.push(hash_exact_file(&path)?);
        }

        let commitment = NixSystemdDefinitionContentCommitmentV1 {
            unit: unit.to_string(),
            definition_identity: identity.clone(),
            manager_owner: manager_owner.to_string(),
            overall_digest: compute_overall_digest(
                unit,
                identity,
                manager_owner,
                &files,
            )?,
            files,
        };
        commitment.validate_shape()?;
        Ok(Self { commitment })
    }

    pub(crate) fn as_ref(&self) -> &NixSystemdDefinitionContentCommitmentV1 {
        &self.commitment
    }

    pub fn digest(&self) -> &str {
        &self.commitment.overall_digest
    }
}

fn compute_overall_digest(
    unit: &str,
    identity: &NixSystemdUnitDefinitionIdentityV1,
    manager_owner: &str,
    files: &[NixDefinitionFileContentDigestV1],
) -> Result<String, NixSystemdDefinitionContentErrorV1> {
    if files.is_empty() || files.len() > MAX_FILES {
        return Err(NixSystemdDefinitionContentErrorV1::InvalidFileCount);
    }
    super::service_domain::NixServiceOperationV1::new(
        unit.to_string(),
        super::service_domain::NixServiceOperationKindV1::Start,
    )
    .map_err(|error| {
        NixSystemdDefinitionContentErrorV1::InvalidServiceUnit(error.to_string())
    })?;
    identity
        .validate_shape()
        .map_err(NixSystemdDefinitionContentErrorV1::InvalidDefinitionIdentity)?;
    validate_manager_owner(manager_owner)?;

    let mut hasher = Hasher::new();
    hasher.update(DEFINITION_CONTENT_DOMAIN_V1);
    put_str(&mut hasher, unit);
    put_str(&mut hasher, &identity.fragment_path);
    put_u64(&mut hasher, identity.drop_in_paths.len() as u64);
    for path in &identity.drop_in_paths {
        put_str(&mut hasher, path);
    }
    put_str(&mut hasher, manager_owner);
    put_u64(&mut hasher, files.len() as u64);    if files.is_empty() || files.len() > MAX_FILES {
        return Err(NixSystemdDefinitionContentErrorV1::InvalidFileCount);
    }
    let mut hasher = Hasher::new();
    hasher.update(DEFINITION_CONTENT_DOMAIN_V1);
    put_u64(&mut hasher, files.len() as u64);
    for file in files {
        file.validate_shape()?;
        hasher.update(DEFINITION_FILE_DOMAIN_V1);
        put_str(&mut hasher, &file.path);
        put_str(&mut hasher, &file.content_digest);
        put_u64(&mut hasher, file.size_bytes);
        put_u64(&mut hasher, file.device_id);
        put_u64(&mut hasher, file.inode);
        put_i64(&mut hasher, file.mtime_seconds);
        put_i64(&mut hasher, file.mtime_nanoseconds);
        put_i64(&mut hasher, file.ctime_seconds);
        put_i64(&mut hasher, file.ctime_nanoseconds);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

}

fn hash_exact_file(path: &str) -> Result<NixDefinitionFileContentDigestV1, NixSystemdDefinitionContentErrorV1> {
    validate_path(path)?;
    let mut file = open_read_only_exact(path)?;
    let before = metadata_identity(&file)?;
    let first_digest = hash_open_file(&mut file)?;
    file.seek(SeekFrom::Start(0)).map_err(NixSystemdDefinitionContentErrorV1::Seek)?;
    let second_digest = hash_open_file(&mut file)?;
    if first_digest != second_digest {
        return Err(NixSystemdDefinitionContentErrorV1::ContentChangedDuringObservation);
    }
    let after = metadata_identity(&file)?;
    if before != after {
        return Err(NixSystemdDefinitionContentErrorV1::MetadataChangedDuringObservation);
    }
    Ok(NixDefinitionFileContentDigestV1 {
        path: path.to_string(),
        content_digest: first_digest,
        size_bytes: after.size,
        device_id: after.device,
        inode: after.inode,
        mtime_seconds: after.mtime_seconds,
        mtime_nanoseconds: after.mtime_nanoseconds,
        ctime_seconds: after.ctime_seconds,
        ctime_nanoseconds: after.ctime_nanoseconds,
    })
}

fn hash_open_file(file: &mut File) -> Result<String, NixSystemdDefinitionContentErrorV1> {
    let mut hasher = Hasher::new();
    let mut total = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(NixSystemdDefinitionContentErrorV1::Read)?;
        if read == 0 { break; }
        total = total.checked_add(read as u64).ok_or(NixSystemdDefinitionContentErrorV1::FileTooLarge)?;
        if total > MAX_FILE_BYTES {
            return Err(NixSystemdDefinitionContentErrorV1::FileTooLarge);
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetadataIdentityV1 {
    device: u64, inode: u64, size: u64,
    mtime_seconds: i64, mtime_nanoseconds: i64,
    ctime_seconds: i64, ctime_nanoseconds: i64,
}

fn metadata_identity(file: &File) -> Result<MetadataIdentityV1, NixSystemdDefinitionContentErrorV1> {
    let metadata = file.metadata().map_err(NixSystemdDefinitionContentErrorV1::Metadata)?;
    if !metadata.file_type().is_file() {
        return Err(NixSystemdDefinitionContentErrorV1::NotRegularFile);
    }
    if metadata.size() > MAX_FILE_BYTES {
        return Err(NixSystemdDefinitionContentErrorV1::FileTooLarge);
    }
    Ok(MetadataIdentityV1 {
        device: metadata.dev(), inode: metadata.ino(), size: metadata.size(),
        mtime_seconds: metadata.mtime(), mtime_nanoseconds: metadata.mtime_nsec(),
        ctime_seconds: metadata.ctime(), ctime_nanoseconds: metadata.ctime_nsec(),
    })
}

#[cfg(target_os = "linux")]
fn open_read_only_exact(path: &str) -> Result<File, NixSystemdDefinitionContentErrorV1> {
    let root = File::open("/").map_err(NixSystemdDefinitionContentErrorV1::Open)?;
    let how = OpenHowV1 {
        flags: (libc::O_RDONLY | libc::O_CLOEXEC) as u64,
        mode: 0,
        resolve: RESOLVE_IN_ROOT | RESOLVE_NO_MAGICLINKS,
    };
    let mut bytes = Vec::with_capacity(path.len() + 1);
    bytes.extend_from_slice(path.as_bytes());
    bytes.push(0);
    let fd = unsafe {
        libc::syscall(
            libc::SYS_openat2, root.as_raw_fd(), bytes.as_ptr(), &how,
            std::mem::size_of::<OpenHowV1>(),
        )
    };
    if fd < 0 {
        return Err(NixSystemdDefinitionContentErrorV1::Open(std::io::Error::last_os_error()));
    }
    Ok(unsafe { File::from_raw_fd(fd as RawFd) })
}

#[cfg(not(target_os = "linux"))]
fn open_read_only_exact(path: &str) -> Result<File, NixSystemdDefinitionContentErrorV1> {
    std::fs::OpenOptions::new().read(true).open(path).map_err(NixSystemdDefinitionContentErrorV1::Open)
}

fn validate_manager_owner(value: &str) -> Result<(), NixSystemdDefinitionContentErrorV1> {
    if value.is_empty() || value.len() > 255 || !value.starts_with(':') {
        return Err(NixSystemdDefinitionContentErrorV1::InvalidManagerOwner);
    }
    let mut elements = value[1..].split('.');
    let first = elements.next().unwrap_or_default();
    if first.is_empty() || elements.next().is_none() {
        return Err(NixSystemdDefinitionContentErrorV1::InvalidManagerOwner);
    }
    for element in std::iter::once(first).chain(elements) {
        if element.is_empty()
            || !element
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
        {
            return Err(NixSystemdDefinitionContentErrorV1::InvalidManagerOwner);
        }
    }
    Ok(())
}

fn validate_path(path: &str) -> Result<(), NixSystemdDefinitionContentErrorV1> {
    if path.is_empty() || path.len() > MAX_PATH_BYTES || !path.starts_with('/') || path.as_bytes().contains(&0) {
        return Err(NixSystemdDefinitionContentErrorV1::InvalidPath);
    }
    let mut components = path.split('/');
    if components.next() != Some("") {
        return Err(NixSystemdDefinitionContentErrorV1::InvalidPath);
    }
    for component in components {
        if component.is_empty() || component == "." || component == ".." {
            return Err(NixSystemdDefinitionContentErrorV1::InvalidPath);
        }
    }
    Ok(())
}

fn validate_digest(value: &str) -> Result<(), NixSystemdDefinitionContentErrorV1> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(NixSystemdDefinitionContentErrorV1::InvalidDigest);
    }
    Ok(())
}

fn put_u64(hasher: &mut Hasher, value: u64) { hasher.update(&value.to_be_bytes()); }
fn put_i64(hasher: &mut Hasher, value: i64) { hasher.update(&value.to_be_bytes()); }
fn put_str(hasher: &mut Hasher, value: &str) { hasher.update(&(value.len() as u64).to_be_bytes()); hasher.update(value.as_bytes()); }

#[derive(Debug, Error)]
pub enum NixSystemdDefinitionContentErrorV1 {
    #[error("invalid definition identity: {0}")]
    InvalidDefinitionIdentity(NixPostStateErrorV1),
    #[error("invalid path")] InvalidPath,
    #[error("invalid digest")] InvalidDigest,
    #[error("definition contains too many files or no files")] InvalidFileCount,
    #[error("definition file is too large")] FileTooLarge,
    #[error("definition paths are not sorted or contain duplicates")] UnsortedOrDuplicatePaths,
    #[error("overall definition-content digest does not match file records")] OverallDigestMismatch,
    #[error("path open failed: {0}")] Open(#[source] std::io::Error),
    #[error("file read failed: {0}")] Read(#[source] std::io::Error),
    #[error("file seek failed: {0}")] Seek(#[source] std::io::Error),
    #[error("file metadata read failed: {0}")] Metadata(#[source] std::io::Error),
    #[error("definition source is not a regular file")] NotRegularFile,
    #[error("file content changed during repeated observation")] ContentChangedDuringObservation,
    #[error("file metadata changed during repeated observation")] MetadataChangedDuringObservation,
    #[error("invalid file identity")] InvalidFileIdentity,
    #[error("invalid service unit: {0}")] InvalidServiceUnit(String),
    #[error("invalid systemd manager unique owner")] InvalidManagerOwner,

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_file_content_commitment_is_deterministic() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("nginx.service");
        std::fs::write(&path, b"[Service]\nExecStart=/bin/true\n").unwrap();
        let identity = NixSystemdUnitDefinitionIdentityV1::new(path.to_str().unwrap(), vec![]).unwrap();
        let first = NixVerifiedSystemdDefinitionContentCommitmentV1::from_observer(
            "nginx.service",
            &identity,
            ":1.42",
        ).unwrap();
        let second = NixVerifiedSystemdDefinitionContentCommitmentV1::from_observer(&identity).unwrap();
        assert_eq!(first.digest(), second.digest());
        assert_eq!(first.as_ref().files.len(), 1);
    }

    #[test]
    fn content_change_changes_commitment() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("nginx.service");
        std::fs::write(&path, b"a").unwrap();
        let path = path.to_str().unwrap();
        let identity = NixSystemdUnitDefinitionIdentityV1::new(path, vec![]).unwrap();
        let first = NixVerifiedSystemdDefinitionContentCommitmentV1::from_observer(&identity).unwrap();
        std::fs::write(path, b"b").unwrap();
        let second = NixVerifiedSystemdDefinitionContentCommitmentV1::from_observer(&identity).unwrap();
        assert_ne!(first.digest(), second.digest());
    }

    #[test]
    fn forged_overall_digest_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("nginx.service");
        std::fs::write(&path, b"safe").unwrap();
        let identity = NixSystemdUnitDefinitionIdentityV1::new(path.to_str().unwrap(), vec![]).unwrap();
        let verified = NixVerifiedSystemdDefinitionContentCommitmentV1::from_observer(&identity).unwrap();
        let mut forged = verified.as_ref().clone();
        forged.overall_digest = "f".repeat(64);
        assert_eq!(forged.validate_shape().unwrap_err(), NixSystemdDefinitionContentErrorV1::OverallDigestMismatch);
    }

    #[test]
    fn invalid_path_is_rejected_before_open() {
        let error = hash_exact_file("/tmp/../etc/shadow").unwrap_err();
        assert_eq!(error.to_string(), "invalid path");
    }
}
