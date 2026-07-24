// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded inspection and attestation verification for stored 3MF/OPC packages.

use crate::attestation::{
    AttestationPolicy, AttestationTrustContext, AttestationVerificationReport,
    AttestedFabricationManifest, ManifestSignatureVerifier, verify_attested_manifest,
    verify_attested_manifest_with_trust,
};
use crate::audit::{AuditJournal, digest_audit_journal};
use crate::crypto_digest::Sha256Digest;
use crate::export::crc32;
use crate::provenance::{FabricationManifest, digest_fabrication_manifest};
use crate::trust::{TrustSnapshot, digest_trust_snapshot};
use std::collections::BTreeMap;

const LOCAL_FILE_HEADER: u32 = 0x0403_4b50;
const CENTRAL_DIRECTORY_HEADER: u32 = 0x0201_4b50;
const END_OF_CENTRAL_DIRECTORY: u32 = 0x0605_4b50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackageInspectionLimits {
    pub max_package_bytes: usize,
    pub max_entries: usize,
    pub max_entry_bytes: usize,
    pub max_path_bytes: usize,
}

impl Default for PackageInspectionLimits {
    fn default() -> Self {
        Self {
            max_package_bytes: 256 * 1024 * 1024,
            max_entries: 64,
            max_entry_bytes: 128 * 1024 * 1024,
            max_path_bytes: 256,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackageError {
    InvalidLimits,
    PackageTooLarge {
        actual: usize,
        maximum: usize,
    },
    Truncated,
    InvalidSignature {
        offset: usize,
        signature: u32,
    },
    MissingCentralDirectory,
    CentralDirectoryMismatch(String),
    MissingEndOfCentralDirectory,
    UnsupportedFlags(u16),
    UnsupportedCompression(u16),
    EntryCountExceeded {
        maximum: usize,
    },
    EntryTooLarge {
        path: String,
        actual: usize,
        maximum: usize,
    },
    PathTooLong(usize),
    InvalidUtf8Path,
    UnsafePath(String),
    DuplicateEntry(String),
    SizeMismatch,
    CrcMismatch(String),
    MissingRequiredEntry(&'static str),
    InvalidManifest(String),
    InvalidAttestation(String),
    InvalidDigest(String),
    MissingAttestation,
    ManifestCopyMismatch,
    ManifestDigestMismatch,
    MissingTrustSnapshot,
    MissingAuditJournal,
    InvalidTrustSnapshot(String),
    TrustSnapshotDigestMismatch,
    InvalidAuditJournal(String),
    AuditJournalDigestMismatch,
    AuditHeadMismatch,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Inspected3mfPackage {
    pub entries: BTreeMap<String, Vec<u8>>,
    pub manifest: Option<FabricationManifest>,
    pub attestation: Option<AttestedFabricationManifest>,
    pub manifest_digest: Option<Sha256Digest>,
    pub trust_snapshot: Option<TrustSnapshot>,
    pub trust_snapshot_digest: Option<Sha256Digest>,
    pub audit_journal: Option<AuditJournal>,
    pub audit_journal_digest: Option<Sha256Digest>,
    pub audit_head: Option<Sha256Digest>,
}

pub fn inspect_3mf_package(
    bytes: &[u8],
    limits: PackageInspectionLimits,
) -> Result<Inspected3mfPackage, PackageError> {
    validate_limits(limits)?;
    if bytes.len() > limits.max_package_bytes {
        return Err(PackageError::PackageTooLarge {
            actual: bytes.len(),
            maximum: limits.max_package_bytes,
        });
    }

    let mut entries = BTreeMap::new();
    let mut local_metadata = BTreeMap::new();
    let mut offset = 0usize;
    while offset + 4 <= bytes.len() {
        let signature = read_u32(bytes, offset)?;
        if signature == CENTRAL_DIRECTORY_HEADER || signature == END_OF_CENTRAL_DIRECTORY {
            break;
        }
        if signature != LOCAL_FILE_HEADER {
            return Err(PackageError::InvalidSignature { offset, signature });
        }
        if entries.len() >= limits.max_entries {
            return Err(PackageError::EntryCountExceeded {
                maximum: limits.max_entries,
            });
        }
        let flags = read_u16(bytes, offset + 6)?;
        if flags != 0 {
            return Err(PackageError::UnsupportedFlags(flags));
        }
        let compression = read_u16(bytes, offset + 8)?;
        if compression != 0 {
            return Err(PackageError::UnsupportedCompression(compression));
        }
        let expected_crc = read_u32(bytes, offset + 14)?;
        let compressed_size = read_u32(bytes, offset + 18)? as usize;
        let uncompressed_size = read_u32(bytes, offset + 22)? as usize;
        if compressed_size != uncompressed_size {
            return Err(PackageError::SizeMismatch);
        }
        let path_len = read_u16(bytes, offset + 26)? as usize;
        let extra_len = read_u16(bytes, offset + 28)? as usize;
        if path_len == 0 || path_len > limits.max_path_bytes {
            return Err(PackageError::PathTooLong(path_len));
        }
        let path_start = offset.checked_add(30).ok_or(PackageError::Truncated)?;
        let path_end = path_start
            .checked_add(path_len)
            .ok_or(PackageError::Truncated)?;
        let data_start = path_end
            .checked_add(extra_len)
            .ok_or(PackageError::Truncated)?;
        let data_end = data_start
            .checked_add(compressed_size)
            .ok_or(PackageError::Truncated)?;
        if data_end > bytes.len() {
            return Err(PackageError::Truncated);
        }
        let path = std::str::from_utf8(&bytes[path_start..path_end])
            .map_err(|_| PackageError::InvalidUtf8Path)?
            .to_string();
        validate_path(&path)?;
        if uncompressed_size > limits.max_entry_bytes {
            return Err(PackageError::EntryTooLarge {
                path,
                actual: uncompressed_size,
                maximum: limits.max_entry_bytes,
            });
        }
        let data = &bytes[data_start..data_end];
        if crc32(data) != expected_crc {
            return Err(PackageError::CrcMismatch(path));
        }
        if entries.insert(path.clone(), data.to_vec()).is_some() {
            return Err(PackageError::DuplicateEntry(path));
        }
        local_metadata.insert(path, (expected_crc, uncompressed_size, offset as u32));
        offset = data_end;
    }

    validate_central_directory(bytes, offset, &local_metadata)?;

    for required in ["[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"] {
        if !entries.contains_key(required) {
            return Err(PackageError::MissingRequiredEntry(required));
        }
    }

    let manifest = entries
        .get("Metadata/fabrication-manifest.json")
        .map(|bytes| {
            serde_json::from_slice(bytes)
                .map_err(|error| PackageError::InvalidManifest(error.to_string()))
        })
        .transpose()?;
    let attestation: Option<crate::attestation::AttestedFabricationManifest> = entries
        .get("Metadata/fabrication-attestation.json")
        .map(|bytes| {
            serde_json::from_slice(bytes)
                .map_err(|error| PackageError::InvalidAttestation(error.to_string()))
        })
        .transpose()?;
    let manifest_digest = entries
        .get("Metadata/fabrication-manifest.sha256")
        .map(|bytes| {
            let text = std::str::from_utf8(bytes)
                .map_err(|error| PackageError::InvalidDigest(error.to_string()))?;
            Sha256Digest::from_hex(text.trim())
                .map_err(|error| PackageError::InvalidDigest(format!("{error:?}")))
        })
        .transpose()?;
    let trust_snapshot = entries
        .get("Metadata/fabrication-trust-snapshot.json")
        .map(|bytes| {
            serde_json::from_slice(bytes)
                .map_err(|error| PackageError::InvalidTrustSnapshot(error.to_string()))
        })
        .transpose()?;
    let trust_snapshot_digest =
        parse_optional_digest(&entries, "Metadata/fabrication-trust-snapshot.sha256")?;
    let audit_journal: Option<crate::audit::AuditJournal> = entries
        .get("Metadata/fabrication-audit-journal.json")
        .map(|bytes| {
            serde_json::from_slice(bytes)
                .map_err(|error| PackageError::InvalidAuditJournal(error.to_string()))
        })
        .transpose()?;
    let audit_journal_digest =
        parse_optional_digest(&entries, "Metadata/fabrication-audit-journal.sha256")?;
    let audit_head = parse_optional_digest(&entries, "Metadata/fabrication-audit-head.sha256")?;

    if let Some(attested) = &attestation {
        if let Some(copied) = &manifest {
            if copied != &attested.manifest {
                return Err(PackageError::ManifestCopyMismatch);
            }
        }
        let computed = digest_fabrication_manifest(&attested.manifest)
            .map_err(|error| PackageError::InvalidManifest(error.to_string()))?;
        if computed != attested.manifest_digest
            || manifest_digest.is_some_and(|digest| digest != computed)
        {
            return Err(PackageError::ManifestDigestMismatch);
        }
    } else if let (Some(manifest), Some(digest)) = (&manifest, manifest_digest) {
        let computed = digest_fabrication_manifest(manifest)
            .map_err(|error| PackageError::InvalidManifest(error.to_string()))?;
        if computed != digest {
            return Err(PackageError::ManifestDigestMismatch);
        }
    }

    if let Some(snapshot) = &trust_snapshot {
        let computed = digest_trust_snapshot(snapshot)
            .map_err(|error| PackageError::InvalidTrustSnapshot(format!("{error:?}")))?;
        if trust_snapshot_digest != Some(computed) {
            return Err(PackageError::TrustSnapshotDigestMismatch);
        }
    } else if trust_snapshot_digest.is_some() {
        return Err(PackageError::MissingTrustSnapshot);
    }

    if let Some(journal) = &audit_journal {
        let report = journal.verify();
        if !report.intact() {
            return Err(PackageError::InvalidAuditJournal(format!(
                "{:?}",
                report.violations
            )));
        }
        let computed = digest_audit_journal(journal)
            .map_err(|error| PackageError::InvalidAuditJournal(format!("{error:?}")))?;
        if audit_journal_digest != Some(computed) {
            return Err(PackageError::AuditJournalDigestMismatch);
        }
        if audit_head != journal.head() {
            return Err(PackageError::AuditHeadMismatch);
        }
    } else if audit_journal_digest.is_some() || audit_head.is_some() {
        return Err(PackageError::InvalidAuditJournal(
            "audit digest exists without journal".into(),
        ));
    }

    Ok(Inspected3mfPackage {
        entries,
        manifest,
        attestation,
        manifest_digest,
        trust_snapshot,
        trust_snapshot_digest,
        audit_journal,
        audit_journal_digest,
        audit_head,
    })
}

pub fn verify_attested_3mf_package(
    bytes: &[u8],
    limits: PackageInspectionLimits,
    policy: &AttestationPolicy,
    verifier: &dyn ManifestSignatureVerifier,
) -> Result<AttestationVerificationReport, PackageError> {
    let package = inspect_3mf_package(bytes, limits)?;
    let attested = package
        .attestation
        .ok_or(PackageError::MissingAttestation)?;
    Ok(verify_attested_manifest(&attested, policy, verifier))
}

pub fn verify_governed_3mf_package(
    bytes: &[u8],
    limits: PackageInspectionLimits,
    policy: &AttestationPolicy,
    verifier: &dyn ManifestSignatureVerifier,
    evaluation_time_unix_s: u64,
) -> Result<AttestationVerificationReport, PackageError> {
    let package = inspect_3mf_package(bytes, limits)?;
    let attested = package
        .attestation
        .ok_or(PackageError::MissingAttestation)?;
    let snapshot = package
        .trust_snapshot
        .ok_or(PackageError::MissingTrustSnapshot)?;
    let audit = package
        .audit_journal
        .ok_or(PackageError::MissingAuditJournal)?;
    if !audit.verify().intact() {
        return Err(PackageError::InvalidAuditJournal(
            "governed verification requires an intact audit journal".into(),
        ));
    }
    Ok(verify_attested_manifest_with_trust(
        &attested,
        policy,
        verifier,
        AttestationTrustContext {
            evaluation_time_unix_s,
            snapshot: &snapshot,
        },
    ))
}

fn parse_optional_digest(
    entries: &BTreeMap<String, Vec<u8>>,
    path: &str,
) -> Result<Option<Sha256Digest>, PackageError> {
    entries
        .get(path)
        .map(|bytes| {
            let text = std::str::from_utf8(bytes)
                .map_err(|error| PackageError::InvalidDigest(error.to_string()))?;
            Sha256Digest::from_hex(text.trim())
                .map_err(|error| PackageError::InvalidDigest(format!("{error:?}")))
        })
        .transpose()
}

fn validate_central_directory(
    bytes: &[u8],
    central_start: usize,
    local_metadata: &BTreeMap<String, (u32, usize, u32)>,
) -> Result<(), PackageError> {
    if read_u32(bytes, central_start).ok() != Some(CENTRAL_DIRECTORY_HEADER) {
        return Err(PackageError::MissingCentralDirectory);
    }
    let mut offset = central_start;
    let mut central_entries = BTreeMap::new();
    while read_u32(bytes, offset).ok() == Some(CENTRAL_DIRECTORY_HEADER) {
        let flags = read_u16(bytes, offset + 8)?;
        let compression = read_u16(bytes, offset + 10)?;
        if flags != 0 {
            return Err(PackageError::UnsupportedFlags(flags));
        }
        if compression != 0 {
            return Err(PackageError::UnsupportedCompression(compression));
        }
        let crc = read_u32(bytes, offset + 16)?;
        let compressed_size = read_u32(bytes, offset + 20)? as usize;
        let uncompressed_size = read_u32(bytes, offset + 24)? as usize;
        if compressed_size != uncompressed_size {
            return Err(PackageError::SizeMismatch);
        }
        let path_len = read_u16(bytes, offset + 28)? as usize;
        let extra_len = read_u16(bytes, offset + 30)? as usize;
        let comment_len = read_u16(bytes, offset + 32)? as usize;
        let local_offset = read_u32(bytes, offset + 42)?;
        let path_start = offset.checked_add(46).ok_or(PackageError::Truncated)?;
        let path_end = path_start
            .checked_add(path_len)
            .ok_or(PackageError::Truncated)?;
        let next = path_end
            .checked_add(extra_len)
            .and_then(|value| value.checked_add(comment_len))
            .ok_or(PackageError::Truncated)?;
        if next > bytes.len() {
            return Err(PackageError::Truncated);
        }
        let path = std::str::from_utf8(&bytes[path_start..path_end])
            .map_err(|_| PackageError::InvalidUtf8Path)?
            .to_string();
        validate_path(&path)?;
        if central_entries
            .insert(path.clone(), (crc, uncompressed_size, local_offset))
            .is_some()
        {
            return Err(PackageError::DuplicateEntry(path));
        }
        offset = next;
    }

    if &central_entries != local_metadata {
        return Err(PackageError::CentralDirectoryMismatch(
            "central entries do not exactly match local headers".into(),
        ));
    }
    if read_u32(bytes, offset).ok() != Some(END_OF_CENTRAL_DIRECTORY) {
        return Err(PackageError::MissingEndOfCentralDirectory);
    }
    let disk_number = read_u16(bytes, offset + 4)?;
    let central_disk = read_u16(bytes, offset + 6)?;
    let entries_on_disk = read_u16(bytes, offset + 8)? as usize;
    let total_entries = read_u16(bytes, offset + 10)? as usize;
    let central_size = read_u32(bytes, offset + 12)? as usize;
    let declared_central_start = read_u32(bytes, offset + 16)? as usize;
    let comment_len = read_u16(bytes, offset + 20)? as usize;
    let expected_end = offset
        .checked_add(22)
        .and_then(|value| value.checked_add(comment_len))
        .ok_or(PackageError::Truncated)?;
    if expected_end != bytes.len()
        || disk_number != 0
        || central_disk != 0
        || entries_on_disk != local_metadata.len()
        || total_entries != local_metadata.len()
        || declared_central_start != central_start
        || central_size != offset - central_start
    {
        return Err(PackageError::CentralDirectoryMismatch(
            "end-of-central-directory metadata is inconsistent".into(),
        ));
    }
    Ok(())
}

fn validate_limits(limits: PackageInspectionLimits) -> Result<(), PackageError> {
    if limits.max_package_bytes == 0
        || limits.max_entries == 0
        || limits.max_entry_bytes == 0
        || limits.max_path_bytes == 0
    {
        return Err(PackageError::InvalidLimits);
    }
    Ok(())
}

fn validate_path(path: &str) -> Result<(), PackageError> {
    if path.starts_with('/') || path.contains('\\') {
        return Err(PackageError::UnsafePath(path.to_string()));
    }
    if path
        .split('/')
        .any(|component| component.is_empty() || component == "." || component == "..")
    {
        return Err(PackageError::UnsafePath(path.to_string()));
    }
    Ok(())
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, PackageError> {
    let end = offset.checked_add(2).ok_or(PackageError::Truncated)?;
    let value = bytes.get(offset..end).ok_or(PackageError::Truncated)?;
    Ok(u16::from_le_bytes(value.try_into().expect("two bytes")))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, PackageError> {
    let end = offset.checked_add(4).ok_or(PackageError::Truncated)?;
    let value = bytes.get(offset..end).ok_or(PackageError::Truncated)?;
    Ok(u32::from_le_bytes(value.try_into().expect("four bytes")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attestation::{
        ManifestSignatureVerifier, ManifestSigner, SignatureAlgorithm, attest_fabrication_manifest,
    };
    use crate::crypto_digest::sha256;
    use crate::export::{export_3mf_package, export_3mf_package_with_attestation};
    use crate::mesh::TriangleMesh;
    use crate::provenance::{FabricationManifest, StableFingerprint};

    struct TestProvider;

    impl ManifestSigner for TestProvider {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Other("test".into())
        }
        fn key_id(&self) -> &str {
            "test-key"
        }
        fn sign(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            Ok(sha256(message).0.to_vec())
        }
    }

    impl ManifestSignatureVerifier for TestProvider {
        fn verify(
            &self,
            algorithm: &SignatureAlgorithm,
            key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            Ok(algorithm == &SignatureAlgorithm::Other("test".into())
                && key_id == "test-key"
                && signature == sha256(message).0.as_slice())
        }
    }

    fn mesh() -> TriangleMesh {
        TriangleMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![[0, 1, 2]],
        }
    }

    fn manifest() -> FabricationManifest {
        let fingerprint = StableFingerprint([1, 2, 3, 4]);
        FabricationManifest {
            schema_version: "symthaea.fabrication.manifest.v1".into(),
            geometry: fingerprint,
            process_policy: fingerprint,
            process_evidence: fingerprint,
            minimum_feature_policy: fingerprint,
            minimum_feature_evidence: fingerprint,
            slice_config: fingerprint,
            slice_layers: fingerprint,
            toolpath_config: fingerprint,
            machine_profile: fingerprint,
            gcode_program: fingerprint,
            pipeline: fingerprint,
            layer_count: 1,
            command_count: 1,
            total_extrusion_mm: 1.0,
        }
    }

    #[test]
    fn core_package_passes_bounded_inspection() {
        let package = export_3mf_package(&mesh());
        let inspected = inspect_3mf_package(&package, PackageInspectionLimits::default()).unwrap();
        assert_eq!(inspected.entries.len(), 3);
    }

    #[test]
    fn attested_package_verifies_end_to_end() {
        let provider = TestProvider;
        let attested = attest_fabrication_manifest(manifest(), &[&provider]).unwrap();
        let package = export_3mf_package_with_attestation(&mesh(), &attested).unwrap();
        let report = verify_attested_3mf_package(
            &package,
            PackageInspectionLimits::default(),
            &AttestationPolicy::default(),
            &provider,
        )
        .unwrap();
        assert!(report.trusted(), "{:#?}", report.violations);
    }

    #[test]
    fn payload_tampering_is_rejected_by_crc() {
        let mut package = export_3mf_package(&mesh());
        let needle = b"<model";
        let offset = package
            .windows(needle.len())
            .position(|window| window == needle)
            .unwrap();
        package[offset] ^= 1;
        assert!(matches!(
            inspect_3mf_package(&package, PackageInspectionLimits::default()),
            Err(PackageError::CrcMismatch(_))
        ));
    }

    #[test]
    fn local_headers_without_central_directory_are_rejected() {
        let package = export_3mf_package(&mesh());
        let central = package
            .windows(4)
            .position(|window| window == CENTRAL_DIRECTORY_HEADER.to_le_bytes().as_slice())
            .unwrap();
        assert!(matches!(
            inspect_3mf_package(&package[..central], PackageInspectionLimits::default()),
            Err(PackageError::MissingCentralDirectory) | Err(PackageError::Truncated)
        ));
    }

    #[test]
    fn package_budget_is_fail_closed() {
        let package = export_3mf_package(&mesh());
        let limits = PackageInspectionLimits {
            max_package_bytes: package.len() - 1,
            ..PackageInspectionLimits::default()
        };
        assert!(matches!(
            inspect_3mf_package(&package, limits),
            Err(PackageError::PackageTooLarge { .. })
        ));
    }

    #[test]
    fn governed_package_binds_trust_and_audit_evidence() {
        use crate::audit::{AuditAction, AuditJournal};
        use crate::export::export_3mf_package_with_governance;
        use crate::trust::{KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot};
        use std::collections::BTreeSet;

        let provider = TestProvider;
        let attested = attest_fabrication_manifest(manifest(), &[&provider]).unwrap();
        let trust = TrustSnapshot::new(
            1,
            100,
            1_000,
            vec![KeyTrustRecord {
                algorithm: SignatureAlgorithm::Other("test".into()),
                key_id: "test-key".into(),
                not_before_unix_s: 100,
                not_after_unix_s: Some(900),
                status: KeyLifecycleStatus::Active,
                usages: BTreeSet::from([KeyUsage::FabricationManifest]),
            }],
        )
        .unwrap();
        let mut audit = AuditJournal::default();
        audit
            .append(
                500,
                "release-test",
                AuditAction::AttestationVerified,
                attested.manifest_digest,
                Some(digest_trust_snapshot(&trust).unwrap()),
            )
            .unwrap();
        let package =
            export_3mf_package_with_governance(&mesh(), &attested, &trust, &audit).unwrap();
        let inspected = inspect_3mf_package(&package, PackageInspectionLimits::default()).unwrap();
        assert_eq!(inspected.trust_snapshot.as_ref(), Some(&trust));
        assert_eq!(inspected.audit_head, audit.head());
        assert!(
            verify_governed_3mf_package(
                &package,
                PackageInspectionLimits::default(),
                &AttestationPolicy::default(),
                &provider,
                500,
            )
            .unwrap()
            .trusted()
        );
    }
}
