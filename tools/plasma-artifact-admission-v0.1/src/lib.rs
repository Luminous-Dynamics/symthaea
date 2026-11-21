//! Fail-closed source, rights, and artifact-fixity admission checks.
//!
//! This validates caller-supplied declarations. It does not fetch metadata,
//! download files, hash bytes, interpret license text, or prove scientific validity.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RightsStatus { Verified, Unverified, Missing }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission { Granted, Denied, Unknown }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResearchAction {
    LocalInspection,
    ExecuteResearch,
    CreateDerivative,
    Redistribute,
    CommercialUse,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RightsReview {
    pub status: RightsStatus,
    pub license_identifier: Option<String>,
    pub evidence_uri: Option<String>,
    pub local_inspection: Permission,
    pub execute_research: Permission,
    pub create_derivative: Permission,
    pub redistribute: Permission,
    pub commercial_use: Permission,
}

impl RightsReview {
    /// Unknown or merely asserted permissions never grant an action.
    pub fn permits(&self, action: ResearchAction) -> bool {
        if self.status != RightsStatus::Verified { return false; }
        let decision = match action {
            ResearchAction::LocalInspection => self.local_inspection,
            ResearchAction::ExecuteResearch => self.execute_research,
            ResearchAction::CreateDerivative => self.create_derivative,
            ResearchAction::Redistribute => self.redistribute,
            ResearchAction::CommercialUse => self.commercial_use,
        };
        decision == Permission::Granted
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ChecksumAlgorithm { Md5, Sha256 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChecksumOrigin { RepositoryMetadata, LocalComputation }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredChecksum {
    pub algorithm: ChecksumAlgorithm,
    pub value: String,
    pub origin: ChecksumOrigin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquisitionState { MetadataOnly, Acquired, Quarantined }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactFile {
    pub name: String,
    /// None means the repository page did not expose an exact byte count.
    pub size_bytes: Option<u64>,
    pub checksums: Vec<DeclaredChecksum>,
    pub acquisition: AcquisitionState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactManifest {
    pub doi: String,
    pub record_version: String,
    pub metadata_uri: String,
    pub retrieved_on: String,
    pub rights: RightsReview,
    pub files: Vec<ArtifactFile>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity { Blocker, Review }

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditIssue {
    InvalidDoi,
    InvalidMetadataUri,
    InvalidRetrievalDate,
    EmptyRecordVersion,
    NoFileEntries,
    EmptyFileName,
    DuplicateFileName(String),
    ZeroByteFile(String),
    MalformedChecksum(String),
    DuplicateRepositoryChecksum {
        algorithm: ChecksumAlgorithm,
        value: String,
        first_file: String,
        second_file: String,
    },
    AcquiredFileMissingLocalSha256(String),
    LocalChecksumForUnacquiredFile(String),
    QuarantinedFile(String),
    RightsMissing,
    RightsUnverified,
    VerifiedRightsMissingLicense,
    VerifiedRightsMissingEvidence,
    VerifiedRightsHasUnknownPermission,
}

impl AuditIssue {
    pub fn severity(&self) -> IssueSeverity {
        match self {
            Self::DuplicateRepositoryChecksum { .. } => IssueSeverity::Review,
            _ => IssueSeverity::Blocker,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditDisposition {
    Blocked,
    ReviewRequired,
    MetadataOnly,
    /// A declaration check only; not proof that bytes were hashed correctly.
    FixityEvidenceDeclared,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditReport {
    pub disposition: AuditDisposition,
    pub issues: Vec<AuditIssue>,
}

fn non_empty(value: &str) -> bool { !value.trim().is_empty() }

fn is_https_uri(value: &str) -> bool {
    value.starts_with("https://") && value.len() > "https://".len()
}

fn is_zenodo_doi(value: &str) -> bool {
    value.strip_prefix("10.5281/zenodo.")
        .is_some_and(|suffix| !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit()))
}

fn is_valid_calendar_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 10
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || !bytes.iter().enumerate()
            .all(|(index, byte)| index == 4 || index == 7 || byte.is_ascii_digit())
    {
        return false;
    }

    let Ok(year) = value[0..4].parse::<u32>() else { return false; };
    let Ok(month) = value[5..7].parse::<u32>() else { return false; };
    let Ok(day) = value[8..10].parse::<u32>() else { return false; };
    if year == 0 { return false; }

    let leap_year = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => return false,
    };
    (1..=days_in_month).contains(&day)
}

fn valid_checksum(algorithm: ChecksumAlgorithm, value: &str) -> bool {
    let expected_len = match algorithm { ChecksumAlgorithm::Md5 => 32, ChecksumAlgorithm::Sha256 => 64 };
    value.len() == expected_len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn push_unique(issues: &mut Vec<AuditIssue>, issue: AuditIssue) {
    if !issues.contains(&issue) { issues.push(issue); }
}

fn rights_has_unknown_permission(rights: &RightsReview) -> bool {
    [
        rights.local_inspection, rights.execute_research, rights.create_derivative,
        rights.redistribute, rights.commercial_use,
    ].contains(&Permission::Unknown)
}

/// Audit declarations without fetching or trusting remote bytes.
///
/// A repeated repository-declared checksum across differently named files is
/// a review signal, not proof their contents are equal. Repository MD5 is never
/// accepted as local SHA-256 fixity evidence.
pub fn audit_manifest(manifest: &ArtifactManifest) -> AuditReport {
    let mut issues = Vec::new();

    if !is_zenodo_doi(&manifest.doi) { push_unique(&mut issues, AuditIssue::InvalidDoi); }
    if !is_https_uri(&manifest.metadata_uri) { push_unique(&mut issues, AuditIssue::InvalidMetadataUri); }
    if !is_valid_calendar_date(&manifest.retrieved_on) { push_unique(&mut issues, AuditIssue::InvalidRetrievalDate); }
    if !non_empty(&manifest.record_version) { push_unique(&mut issues, AuditIssue::EmptyRecordVersion); }
    if manifest.files.is_empty() { push_unique(&mut issues, AuditIssue::NoFileEntries); }

    match manifest.rights.status {
        RightsStatus::Missing => push_unique(&mut issues, AuditIssue::RightsMissing),
        RightsStatus::Unverified => push_unique(&mut issues, AuditIssue::RightsUnverified),
        RightsStatus::Verified => {
            if !manifest.rights.license_identifier.as_deref().is_some_and(non_empty) {
                push_unique(&mut issues, AuditIssue::VerifiedRightsMissingLicense);
            }
            if !manifest.rights.evidence_uri.as_deref().is_some_and(is_https_uri) {
                push_unique(&mut issues, AuditIssue::VerifiedRightsMissingEvidence);
            }
            if rights_has_unknown_permission(&manifest.rights) {
                push_unique(&mut issues, AuditIssue::VerifiedRightsHasUnknownPermission);
            }
        }
    }

    let mut filenames = BTreeSet::new();
    let mut repository_checksums: BTreeMap<(ChecksumAlgorithm, String), String> = BTreeMap::new();

    for file in &manifest.files {
        if !non_empty(&file.name) { push_unique(&mut issues, AuditIssue::EmptyFileName); }
        if !filenames.insert(file.name.as_str()) {
            push_unique(&mut issues, AuditIssue::DuplicateFileName(file.name.clone()));
        }
        if file.size_bytes == Some(0) {
            push_unique(&mut issues, AuditIssue::ZeroByteFile(file.name.clone()));
        }

        let has_local_sha256 = file.checksums.iter().any(|checksum| {
            checksum.algorithm == ChecksumAlgorithm::Sha256
                && checksum.origin == ChecksumOrigin::LocalComputation
                && valid_checksum(checksum.algorithm, &checksum.value)
        });

        for checksum in &file.checksums {
            if !valid_checksum(checksum.algorithm, &checksum.value) {
                push_unique(&mut issues, AuditIssue::MalformedChecksum(file.name.clone()));
                continue;
            }
            if checksum.origin == ChecksumOrigin::RepositoryMetadata {
                let key = (checksum.algorithm, checksum.value.to_ascii_lowercase());
                if let Some(first_file) = repository_checksums.get(&key) {
                    if first_file != &file.name {
                        push_unique(&mut issues, AuditIssue::DuplicateRepositoryChecksum {
                            algorithm: checksum.algorithm,
                            value: checksum.value.to_ascii_lowercase(),
                            first_file: first_file.clone(),
                            second_file: file.name.clone(),
                        });
                    }
                } else {
                    repository_checksums.insert(key, file.name.clone());
                }
            }
            if checksum.origin == ChecksumOrigin::LocalComputation
                && file.acquisition != AcquisitionState::Acquired
            {
                push_unique(&mut issues, AuditIssue::LocalChecksumForUnacquiredFile(file.name.clone()));
            }
        }

        match file.acquisition {
            AcquisitionState::MetadataOnly => {}
            AcquisitionState::Acquired if !has_local_sha256 => {
                push_unique(&mut issues, AuditIssue::AcquiredFileMissingLocalSha256(file.name.clone()));
            }
            AcquisitionState::Acquired => {}
            AcquisitionState::Quarantined => {
                push_unique(&mut issues, AuditIssue::QuarantinedFile(file.name.clone()));
            }
        }
    }

    let has_blocker = issues.iter().any(|issue| issue.severity() == IssueSeverity::Blocker);
    let has_review = issues.iter().any(|issue| issue.severity() == IssueSeverity::Review);
    let disposition = if has_blocker {
        AuditDisposition::Blocked
    } else if has_review {
        AuditDisposition::ReviewRequired
    } else if manifest.files.iter().any(|file| file.acquisition != AcquisitionState::Acquired) {
        AuditDisposition::MetadataOnly
    } else {
        AuditDisposition::FixityEvidenceDeclared
    };
    AuditReport { disposition, issues }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verified_rights() -> RightsReview {
        RightsReview {
            status: RightsStatus::Verified,
            license_identifier: Some("CC-BY-4.0".into()),
            evidence_uri: Some("https://zenodo.org/records/example".into()),
            local_inspection: Permission::Granted,
            execute_research: Permission::Granted,
            create_derivative: Permission::Granted,
            redistribute: Permission::Granted,
            commercial_use: Permission::Granted,
        }
    }

    fn manifest() -> ArtifactManifest {
        ArtifactManifest {
            doi: "10.5281/zenodo.1234567".into(),
            record_version: "1".into(),
            metadata_uri: "https://zenodo.org/records/1234567".into(),
            retrieved_on: "2026-11-19".into(),
            rights: verified_rights(),
            files: vec![ArtifactFile {
                name: "data.bin".into(),
                size_bytes: Some(12),
                checksums: vec![DeclaredChecksum {
                    algorithm: ChecksumAlgorithm::Sha256,
                    value: "a".repeat(64),
                    origin: ChecksumOrigin::LocalComputation,
                }],
                acquisition: AcquisitionState::Acquired,
            }],
        }
    }

    #[test]
    fn verified_rights_and_local_sha256_only_declare_fixity_evidence() {
        let report = audit_manifest(&manifest());
        assert_eq!(report.disposition, AuditDisposition::FixityEvidenceDeclared);
        assert!(report.issues.is_empty());
    }

    #[test]
    fn unverified_rights_block_all_research_actions() {
        let mut value = manifest();
        value.rights.status = RightsStatus::Unverified;
        for action in [ResearchAction::LocalInspection, ResearchAction::ExecuteResearch,
            ResearchAction::CreateDerivative, ResearchAction::Redistribute, ResearchAction::CommercialUse] {
            assert!(!value.rights.permits(action));
        }
        assert_eq!(audit_manifest(&value).disposition, AuditDisposition::Blocked);
    }

    #[test]
    fn unknown_permission_never_grants_action() {
        let mut value = manifest();
        value.rights.redistribute = Permission::Unknown;
        assert!(!value.rights.permits(ResearchAction::Redistribute));
        assert!(audit_manifest(&value).issues.contains(&AuditIssue::VerifiedRightsHasUnknownPermission));
    }

    #[test]
    fn duplicate_repository_md5_across_distinct_names_requires_review() {
        let mut value = manifest();
        let md5 = "0123456789abcdef0123456789abcdef".to_owned();
        value.files[0].checksums = vec![DeclaredChecksum {
            algorithm: ChecksumAlgorithm::Md5, value: md5.clone(),
            origin: ChecksumOrigin::RepositoryMetadata,
        }];
        value.files[0].acquisition = AcquisitionState::MetadataOnly;
        value.files.push(ArtifactFile {
            name: "other.bin".into(), size_bytes: Some(99),
            checksums: vec![DeclaredChecksum {
                algorithm: ChecksumAlgorithm::Md5, value: md5,
                origin: ChecksumOrigin::RepositoryMetadata,
            }],
            acquisition: AcquisitionState::MetadataOnly,
        });
        let report = audit_manifest(&value);
        assert_eq!(report.disposition, AuditDisposition::ReviewRequired);
        assert!(report.issues.iter().any(|issue| matches!(issue,
            AuditIssue::DuplicateRepositoryChecksum { .. })));
    }

    #[test]
    fn repository_md5_does_not_satisfy_local_fixity_requirement() {
        let mut value = manifest();
        value.files[0].checksums = vec![DeclaredChecksum {
            algorithm: ChecksumAlgorithm::Md5,
            value: "0123456789abcdef0123456789abcdef".into(),
            origin: ChecksumOrigin::RepositoryMetadata,
        }];
        assert!(audit_manifest(&value).issues.contains(
            &AuditIssue::AcquiredFileMissingLocalSha256("data.bin".into())));
    }

    #[test]
    fn acquired_file_requires_a_local_sha256_declaration() {
        let mut value = manifest();
        value.files[0].checksums.clear();
        assert!(audit_manifest(&value).issues.contains(
            &AuditIssue::AcquiredFileMissingLocalSha256("data.bin".into())));
    }

    #[test]
    fn metadata_only_file_cannot_claim_a_locally_computed_checksum() {
        let mut value = manifest();
        value.files[0].acquisition = AcquisitionState::MetadataOnly;
        assert!(audit_manifest(&value).issues.contains(
            &AuditIssue::LocalChecksumForUnacquiredFile("data.bin".into())));
    }

    #[test]
    fn malformed_digest_is_rejected() {
        let mut value = manifest();
        value.files[0].checksums[0].value = "not-a-sha256".into();
        assert!(audit_manifest(&value).issues.contains(&AuditIssue::MalformedChecksum("data.bin".into())));
    }

    #[test]
    fn quarantine_blocks_admission() {
        let mut value = manifest();
        value.files[0].acquisition = AcquisitionState::Quarantined;
        assert!(audit_manifest(&value).issues.contains(&AuditIssue::QuarantinedFile("data.bin".into())));
    }

    #[test]
    fn verified_rights_require_license_identity_and_evidence() {
        let mut value = manifest();
        value.rights.license_identifier = None;
        value.rights.evidence_uri = None;
        let report = audit_manifest(&value);
        assert!(report.issues.contains(&AuditIssue::VerifiedRightsMissingLicense));
        assert!(report.issues.contains(&AuditIssue::VerifiedRightsMissingEvidence));
    }

    #[test]
    fn impossible_calendar_date_is_rejected() {
        let mut value = manifest();
        value.retrieved_on = "2026-02-29".into();
        assert!(audit_manifest(&value).issues.contains(&AuditIssue::InvalidRetrievalDate));
    }

    #[test]
    fn leap_day_is_accepted_for_a_leap_year() {
        let mut value = manifest();
        value.retrieved_on = "2024-02-29".into();
        assert!(!audit_manifest(&value).issues.contains(&AuditIssue::InvalidRetrievalDate));
    }

    #[test]
    fn non_numeric_zenodo_doi_suffix_is_rejected() {
        let mut value = manifest();
        value.doi = "10.5281/zenodo/not-a-record".into();
        assert!(audit_manifest(&value).issues.contains(&AuditIssue::InvalidDoi));
    }

    #[test]
    fn empty_file_manifest_is_not_admissible() {
        let mut value = manifest();
        value.files.clear();
        assert!(audit_manifest(&value).issues.contains(&AuditIssue::NoFileEntries));
    }
}
