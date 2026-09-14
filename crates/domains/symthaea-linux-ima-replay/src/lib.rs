// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical Linux IMA SHA-256 measurement-list replay.
//!
//! This is deliberately a narrow first provider theorem for ASSURE-RUNTIME-002:
//! canonical little-endian `binary_runtime_measurements_sha256`, SHA-256 template
//! digests/PCR extends, and reviewed `ima-ng` / `ima-sig` templates only.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const IMA_REPLAY_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.linux-ima-replay-policy.v1";
pub const IMA_REPLAY_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.linux-ima-replay-report.v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.linux-ima-replay-policy.digest.v1\0";
const LIST_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.linux-ima-canonical-list.digest.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.linux-ima-replay-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.linux-ima-replay-qualification.digest.v1\0";

const SHA256_LEN: usize = 32;
const HEADER_BYTES_BEFORE_NAME: usize = 4 + SHA256_LEN + 4;
const MAX_TEXT_BYTES: usize = 512;
const MAX_EVIDENCE_REFS: usize = 128;
const MAX_REQUIRED_MEASUREMENTS: usize = 4096;
const ABSOLUTE_MAX_MEASUREMENT_LIST_BYTES: u64 = 256 * 1024 * 1024;
const ABSOLUTE_MAX_RECORDS: u32 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SupportedImaTemplate {
    ImaNg,
    ImaSig,
}

impl SupportedImaTemplate {
    fn as_str(self) -> &'static str {
        match self {
            Self::ImaNg => "ima-ng",
            Self::ImaSig => "ima-sig",
        }
    }

    const fn expected_fields(self) -> usize {
        match self {
            Self::ImaNg => 2,
            Self::ImaSig => 3,
        }
    }
}

/// One required measured artifact. A name digest is optional and is never
/// accepted without the content digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequiredImaMeasurement {
    pub measurement_id: String,
    pub event_sha256: [u8; SHA256_LEN],
    pub event_name_sha256: Option<[u8; SHA256_LEN]>,
}

impl RequiredImaMeasurement {
    fn validate(&self) -> bool {
        canonical_text(&self.measurement_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImaReplayPolicy {
    pub schema_version: String,
    pub policy_id: String,
    /// PCR whose ordered extends are being independently replayed.
    pub expected_pcr: u32,
    /// Starting value for this replay lineage. A full hard-boot list normally
    /// uses all-zero bytes; non-zero seeds require their own external lineage proof.
    pub expected_initial_pcr: [u8; SHA256_LEN],
    pub allowed_templates: Vec<SupportedImaTemplate>,
    pub required_measurements: Vec<RequiredImaMeasurement>,
    pub max_measurement_list_bytes: u64,
    pub max_records: u32,
    pub max_template_name_bytes: u32,
    pub max_template_data_bytes: u32,
    pub max_field_bytes: u32,
    pub evidence_refs: Vec<String>,
}

impl ImaReplayPolicy {
    pub fn validate(&self) -> bool {
        if self.schema_version != IMA_REPLAY_POLICY_SCHEMA_V1
            || !canonical_text(&self.policy_id)
            || self.expected_pcr > 31
            || self.allowed_templates.is_empty()
            || self.required_measurements.len() > MAX_REQUIRED_MEASUREMENTS
            || self.max_measurement_list_bytes == 0
            || self.max_measurement_list_bytes > ABSOLUTE_MAX_MEASUREMENT_LIST_BYTES
            || self.max_records == 0
            || self.max_records > ABSOLUTE_MAX_RECORDS
            || self.max_template_name_bytes == 0
            || self.max_template_name_bytes > 1024
            || self.max_template_data_bytes == 0
            || self.max_template_data_bytes > 16 * 1024 * 1024
            || self.max_field_bytes == 0
            || self.max_field_bytes > self.max_template_data_bytes
            || !valid_refs(&self.evidence_refs)
        {
            return false;
        }

        let templates = self
            .allowed_templates
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if templates.len() != self.allowed_templates.len() {
            return false;
        }

        let mut ids = BTreeSet::new();
        for requirement in &self.required_measurements {
            if !requirement.validate() || !ids.insert(requirement.measurement_id.as_str()) {
                return false;
            }
        }
        true
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.policy_id);
        push_u32(&mut hasher, self.expected_pcr);
        hasher.update(&self.expected_initial_pcr);

        let mut templates = self.allowed_templates.clone();
        templates.sort();
        for template in templates {
            push_field(&mut hasher, template.as_str());
        }

        let mut requirements = self.required_measurements.clone();
        requirements.sort_by(|left, right| left.measurement_id.cmp(&right.measurement_id));
        for requirement in requirements {
            push_field(&mut hasher, &requirement.measurement_id);
            hasher.update(&requirement.event_sha256);
            match requirement.event_name_sha256 {
                Some(value) => {
                    hasher.update(&[1]);
                    hasher.update(&value);
                }
                None => hasher.update(&[0]),
            }
        }

        push_u64(&mut hasher, self.max_measurement_list_bytes);
        for value in [
            self.max_records,
            self.max_template_name_bytes,
            self.max_template_data_bytes,
            self.max_field_bytes,
        ] {
            push_u32(&mut hasher, value);
        }
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImaReplayDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImaReplayIssue {
    InvalidPolicy,
    EmptyMeasurementList,
    MeasurementListTooLarge { observed: u64, maximum: u64 },
    RecordLimitExceeded,
    TruncatedRecord { record_index: u32, stage: String },
    TemplateNameTooLarge { record_index: u32, observed: u32 },
    InvalidTemplateName { record_index: u32 },
    UnsupportedTemplate { record_index: u32, template_name: String },
    TemplateDataTooLarge { record_index: u32, observed: u32 },
    FieldTooLarge { record_index: u32, field_index: u32, observed: u32 },
    FieldCountMismatch { record_index: u32, expected: u32, observed: u32 },
    TemplateDataTrailingBytes { record_index: u32, trailing: u32 },
    TemplateDigestMismatch { record_index: u32 },
    InvalidEventDigestField { record_index: u32 },
    UnsupportedEventDigestAlgorithm { record_index: u32, algorithm: String },
    InvalidEventNameField { record_index: u32 },
    MissingRequiredMeasurement { measurement_id: String },
    PcrMismatch,
}

impl ImaReplayIssue {
    fn is_invalid(&self) -> bool {
        matches!(
            self,
            Self::InvalidPolicy
                | Self::EmptyMeasurementList
                | Self::MeasurementListTooLarge { .. }
                | Self::RecordLimitExceeded
                | Self::TruncatedRecord { .. }
                | Self::TemplateNameTooLarge { .. }
                | Self::InvalidTemplateName { .. }
                | Self::UnsupportedTemplate { .. }
                | Self::TemplateDataTooLarge { .. }
                | Self::FieldTooLarge { .. }
                | Self::FieldCountMismatch { .. }
                | Self::TemplateDataTrailingBytes { .. }
                | Self::TemplateDigestMismatch { .. }
                | Self::InvalidEventDigestField { .. }
                | Self::InvalidEventNameField { .. }
        )
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::EmptyMeasurementList => "empty-measurement-list".into(),
            Self::MeasurementListTooLarge { observed, maximum } => {
                format!("measurement-list-too-large:{observed}:{maximum}")
            }
            Self::RecordLimitExceeded => "record-limit-exceeded".into(),
            Self::TruncatedRecord { record_index, stage } => {
                format!("truncated:{record_index}:{stage}")
            }
            Self::TemplateNameTooLarge { record_index, observed } => {
                format!("template-name-too-large:{record_index}:{observed}")
            }
            Self::InvalidTemplateName { record_index } => {
                format!("invalid-template-name:{record_index}")
            }
            Self::UnsupportedTemplate { record_index, template_name } => {
                format!("unsupported-template:{record_index}:{template_name}")
            }
            Self::TemplateDataTooLarge { record_index, observed } => {
                format!("template-data-too-large:{record_index}:{observed}")
            }
            Self::FieldTooLarge { record_index, field_index, observed } => {
                format!("field-too-large:{record_index}:{field_index}:{observed}")
            }
            Self::FieldCountMismatch { record_index, expected, observed } => {
                format!("field-count-mismatch:{record_index}:{expected}:{observed}")
            }
            Self::TemplateDataTrailingBytes { record_index, trailing } => {
                format!("template-data-trailing:{record_index}:{trailing}")
            }
            Self::TemplateDigestMismatch { record_index } => {
                format!("template-digest-mismatch:{record_index}")
            }
            Self::InvalidEventDigestField { record_index } => {
                format!("invalid-event-digest:{record_index}")
            }
            Self::UnsupportedEventDigestAlgorithm { record_index, algorithm } => {
                format!("unsupported-event-digest-algorithm:{record_index}:{algorithm}")
            }
            Self::InvalidEventNameField { record_index } => {
                format!("invalid-event-name:{record_index}")
            }
            Self::MissingRequiredMeasurement { measurement_id } => {
                format!("missing-required:{measurement_id}")
            }
            Self::PcrMismatch => "pcr-mismatch".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImaReplayReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub measurement_list_digest: String,
    pub expected_pcr: u32,
    pub initial_pcr: [u8; SHA256_LEN],
    pub replayed_pcr: [u8; SHA256_LEN],
    pub expected_final_pcr: [u8; SHA256_LEN],
    pub record_count: u32,
    pub selected_pcr_record_count: u32,
    pub template_counts: BTreeMap<String, u32>,
    pub matched_required_measurements: Vec<String>,
    pub disposition: ImaReplayDisposition,
    pub issues: Vec<ImaReplayIssue>,
}

impl ImaReplayReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.measurement_list_digest.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        push_u32(&mut hasher, self.expected_pcr);
        hasher.update(&self.initial_pcr);
        hasher.update(&self.replayed_pcr);
        hasher.update(&self.expected_final_pcr);
        push_u32(&mut hasher, self.record_count);
        push_u32(&mut hasher, self.selected_pcr_record_count);
        for (name, count) in &self.template_counts {
            push_field(&mut hasher, name);
            push_u32(&mut hasher, *count);
        }
        for measurement_id in &self.matched_required_measurements {
            push_field(&mut hasher, measurement_id);
        }
        push_field(
            &mut hasher,
            match self.disposition {
                ImaReplayDisposition::Invalid => "invalid",
                ImaReplayDisposition::Blocked => "blocked",
                ImaReplayDisposition::Qualified => "qualified",
            },
        );
        for issue in &self.issues {
            push_field(&mut hasher, &issue.code());
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

/// Capability-bearing replay result. It is intentionally not serializable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedImaReplay {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    measurement_list_digest: String,
    expected_pcr: u32,
    final_pcr: [u8; SHA256_LEN],
    record_count: u32,
    selected_pcr_record_count: u32,
    matched_required_measurements: Vec<String>,
}

impl VerifiedImaReplay {
    pub fn qualification_digest(&self) -> &str {
        &self.qualification_digest
    }

    pub fn report_digest(&self) -> &str {
        &self.report_digest
    }

    pub fn policy_digest(&self) -> &str {
        &self.policy_digest
    }

    pub fn measurement_list_digest(&self) -> &str {
        &self.measurement_list_digest
    }

    pub const fn expected_pcr(&self) -> u32 {
        self.expected_pcr
    }

    pub const fn final_pcr(&self) -> [u8; SHA256_LEN] {
        self.final_pcr
    }

    pub const fn record_count(&self) -> u32 {
        self.record_count
    }

    pub const fn selected_pcr_record_count(&self) -> u32 {
        self.selected_pcr_record_count
    }

    pub fn matched_required_measurements(&self) -> &[String] {
        &self.matched_required_measurements
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImaReplayQualification {
    pub report: ImaReplayReport,
    verified: VerifiedImaReplay,
}

impl ImaReplayQualification {
    pub fn verified(&self) -> &VerifiedImaReplay {
        &self.verified
    }

    pub fn into_verified(self) -> VerifiedImaReplay {
        self.verified
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

pub fn assess_canonical_ima_sha256(
    canonical_binary_measurements: &[u8],
    expected_final_pcr: [u8; SHA256_LEN],
    policy: &ImaReplayPolicy,
) -> ImaReplayReport {
    let policy_digest = policy.canonical_digest();
    if policy_digest.is_none() {
        return terminal_report(
            policy,
            policy_digest,
            bounded_list_digest(canonical_binary_measurements),
            expected_final_pcr,
            ImaReplayIssue::InvalidPolicy,
        );
    }

    if canonical_binary_measurements.len() as u64 > policy.max_measurement_list_bytes {
        return terminal_report(
            policy,
            policy_digest,
            "unavailable:measurement-list-too-large".into(),
            expected_final_pcr,
            ImaReplayIssue::MeasurementListTooLarge {
                observed: canonical_binary_measurements.len() as u64,
                maximum: policy.max_measurement_list_bytes,
            },
        );
    }

    let measurement_list_digest = digest_list(canonical_binary_measurements);
    if canonical_binary_measurements.is_empty() {
        return terminal_report(
            policy,
            policy_digest,
            measurement_list_digest,
            expected_final_pcr,
            ImaReplayIssue::EmptyMeasurementList,
        );
    }

    let allowed = policy
        .allowed_templates
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut cursor = 0usize;
    let mut record_count = 0u32;
    let mut selected_count = 0u32;
    let mut replayed_pcr = policy.expected_initial_pcr;
    let mut template_counts = BTreeMap::<String, u32>::new();
    let mut observed_measurements = Vec::<ObservedMeasurement>::new();
    let mut issues = Vec::<ImaReplayIssue>::new();

    while cursor < canonical_binary_measurements.len() {
        if record_count >= policy.max_records {
            issues.push(ImaReplayIssue::RecordLimitExceeded);
            break;
        }

        let record_index = record_count;
        match parse_record(
            canonical_binary_measurements,
            &mut cursor,
            record_index,
            policy,
            &allowed,
        ) {
            Ok(record) => {
                *template_counts
                    .entry(record.template.as_str().to_string())
                    .or_default() += 1;
                if record.pcr == policy.expected_pcr {
                    replayed_pcr = extend_sha256(replayed_pcr, record.template_digest);
                    selected_count = selected_count.saturating_add(1);
                }
                observed_measurements.push(record.measurement);
                record_count = record_count.saturating_add(1);
            }
            Err(issue) => {
                issues.push(issue);
                break;
            }
        }
    }

    let mut matched = BTreeSet::<String>::new();
    if !issues.iter().any(ImaReplayIssue::is_invalid) {
        for requirement in &policy.required_measurements {
            if observed_measurements
                .iter()
                .any(|measurement| measurement.matches(requirement))
            {
                matched.insert(requirement.measurement_id.clone());
            } else {
                issues.push(ImaReplayIssue::MissingRequiredMeasurement {
                    measurement_id: requirement.measurement_id.clone(),
                });
            }
        }

        if replayed_pcr != expected_final_pcr {
            issues.push(ImaReplayIssue::PcrMismatch);
        }
    }

    let disposition = if issues.iter().any(ImaReplayIssue::is_invalid) {
        ImaReplayDisposition::Invalid
    } else if issues.is_empty() {
        ImaReplayDisposition::Qualified
    } else {
        ImaReplayDisposition::Blocked
    };

    ImaReplayReport {
        schema_version: IMA_REPLAY_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest,
        measurement_list_digest,
        expected_pcr: policy.expected_pcr,
        initial_pcr: policy.expected_initial_pcr,
        replayed_pcr,
        expected_final_pcr,
        record_count,
        selected_pcr_record_count: selected_count,
        template_counts,
        matched_required_measurements: matched.into_iter().collect(),
        disposition,
        issues,
    }
}

pub fn verify_canonical_ima_sha256(
    canonical_binary_measurements: &[u8],
    expected_final_pcr: [u8; SHA256_LEN],
    policy: &ImaReplayPolicy,
) -> Result<ImaReplayQualification, ImaReplayReport> {
    let report = assess_canonical_ima_sha256(
        canonical_binary_measurements,
        expected_final_pcr,
        policy,
    );
    if report.disposition != ImaReplayDisposition::Qualified {
        return Err(report);
    }

    let policy_digest = report
        .policy_digest
        .clone()
        .expect("qualified report always has a policy digest");
    let report_digest = report.canonical_digest();
    let qualification_digest = qualification_digest(&report, &report_digest, &policy_digest);
    let verified = VerifiedImaReplay {
        qualification_digest,
        report_digest,
        policy_digest,
        measurement_list_digest: report.measurement_list_digest.clone(),
        expected_pcr: report.expected_pcr,
        final_pcr: report.replayed_pcr,
        record_count: report.record_count,
        selected_pcr_record_count: report.selected_pcr_record_count,
        matched_required_measurements: report.matched_required_measurements.clone(),
    };
    Ok(ImaReplayQualification { report, verified })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ObservedMeasurement {
    event_sha256: [u8; SHA256_LEN],
    event_name_sha256: [u8; SHA256_LEN],
}

impl ObservedMeasurement {
    fn matches(&self, required: &RequiredImaMeasurement) -> bool {
        self.event_sha256 == required.event_sha256
            && required
                .event_name_sha256
                .is_none_or(|expected| expected == self.event_name_sha256)
    }
}

struct ParsedRecord {
    pcr: u32,
    template: SupportedImaTemplate,
    template_digest: [u8; SHA256_LEN],
    measurement: ObservedMeasurement,
}

fn parse_record(
    bytes: &[u8],
    cursor: &mut usize,
    record_index: u32,
    policy: &ImaReplayPolicy,
    allowed: &BTreeSet<SupportedImaTemplate>,
) -> Result<ParsedRecord, ImaReplayIssue> {
    if bytes.len().saturating_sub(*cursor) < HEADER_BYTES_BEFORE_NAME {
        return Err(truncated(record_index, "record-header"));
    }

    let pcr = read_u32(bytes, cursor, record_index, "pcr")?;
    let template_digest = read_array32(bytes, cursor, record_index, "template-digest")?;
    let name_len = read_u32(bytes, cursor, record_index, "template-name-length")?;
    if name_len > policy.max_template_name_bytes {
        return Err(ImaReplayIssue::TemplateNameTooLarge {
            record_index,
            observed: name_len,
        });
    }

    let name_bytes = read_bytes(
        bytes,
        cursor,
        name_len as usize,
        record_index,
        "template-name",
    )?;
    let name = std::str::from_utf8(name_bytes)
        .map_err(|_| ImaReplayIssue::InvalidTemplateName { record_index })?;
    let template = match name {
        "ima-ng" => SupportedImaTemplate::ImaNg,
        "ima-sig" => SupportedImaTemplate::ImaSig,
        other => {
            return Err(ImaReplayIssue::UnsupportedTemplate {
                record_index,
                template_name: other.to_string(),
            });
        }
    };
    if !allowed.contains(&template) {
        return Err(ImaReplayIssue::UnsupportedTemplate {
            record_index,
            template_name: name.to_string(),
        });
    }

    let data_len = read_u32(bytes, cursor, record_index, "template-data-length")?;
    if data_len > policy.max_template_data_bytes {
        return Err(ImaReplayIssue::TemplateDataTooLarge {
            record_index,
            observed: data_len,
        });
    }
    let data = read_bytes(
        bytes,
        cursor,
        data_len as usize,
        record_index,
        "template-data",
    )?;

    // For non-legacy IMA templates, canonical binary template data is exactly
    // the concatenation of u32_le(field_len) || field_bytes. The kernel hashes
    // those exact bytes for the per-bank template digest.
    if sha256_array(data) != template_digest {
        return Err(ImaReplayIssue::TemplateDigestMismatch { record_index });
    }

    let fields = parse_fields(
        data,
        record_index,
        template.expected_fields(),
        policy.max_field_bytes,
    )?;
    let event_sha256 = parse_event_digest(fields[0], record_index)?;
    let event_name = parse_event_name(fields[1], record_index)?;
    let event_name_sha256 = sha256_array(event_name);

    Ok(ParsedRecord {
        pcr,
        template,
        template_digest,
        measurement: ObservedMeasurement {
            event_sha256,
            event_name_sha256,
        },
    })
}

fn parse_fields<'a>(
    data: &'a [u8],
    record_index: u32,
    expected_fields: usize,
    max_field_bytes: u32,
) -> Result<Vec<&'a [u8]>, ImaReplayIssue> {
    let mut cursor = 0usize;
    let mut fields = Vec::with_capacity(expected_fields);

    while cursor < data.len() && fields.len() < expected_fields {
        if data.len().saturating_sub(cursor) < 4 {
            return Err(truncated(record_index, "field-length"));
        }
        let len = u32::from_le_bytes(
            data[cursor..cursor + 4]
                .try_into()
                .expect("four bytes are present"),
        );
        cursor += 4;
        if len > max_field_bytes {
            return Err(ImaReplayIssue::FieldTooLarge {
                record_index,
                field_index: fields.len() as u32,
                observed: len,
            });
        }
        if data.len().saturating_sub(cursor) < len as usize {
            return Err(truncated(record_index, "field-data"));
        }
        fields.push(&data[cursor..cursor + len as usize]);
        cursor += len as usize;
    }

    if fields.len() != expected_fields {
        return Err(ImaReplayIssue::FieldCountMismatch {
            record_index,
            expected: expected_fields as u32,
            observed: fields.len() as u32,
        });
    }
    if cursor != data.len() {
        return Err(ImaReplayIssue::TemplateDataTrailingBytes {
            record_index,
            trailing: (data.len() - cursor) as u32,
        });
    }
    Ok(fields)
}

/// `d-ng` is encoded as `<hash-algo> ':' '\0' <raw digest>`.
fn parse_event_digest(
    field: &[u8],
    record_index: u32,
) -> Result<[u8; SHA256_LEN], ImaReplayIssue> {
    let Some(colon) = field.iter().position(|byte| *byte == b':') else {
        return Err(ImaReplayIssue::InvalidEventDigestField { record_index });
    };
    if colon == 0 || field.get(colon + 1) != Some(&0) {
        return Err(ImaReplayIssue::InvalidEventDigestField { record_index });
    }

    let algorithm = std::str::from_utf8(&field[..colon])
        .map_err(|_| ImaReplayIssue::InvalidEventDigestField { record_index })?;
    if algorithm != "sha256" {
        return Err(ImaReplayIssue::UnsupportedEventDigestAlgorithm {
            record_index,
            algorithm: algorithm.to_string(),
        });
    }

    let digest = &field[colon + 2..];
    if digest.len() != SHA256_LEN {
        return Err(ImaReplayIssue::InvalidEventDigestField { record_index });
    }
    Ok(digest
        .try_into()
        .expect("digest length was checked to be SHA-256"))
}

fn parse_event_name<'a>(
    field: &'a [u8],
    record_index: u32,
) -> Result<&'a [u8], ImaReplayIssue> {
    if field.is_empty()
        || field.last() != Some(&0)
        || field[..field.len() - 1].contains(&0)
    {
        return Err(ImaReplayIssue::InvalidEventNameField { record_index });
    }
    Ok(&field[..field.len() - 1])
}

fn terminal_report(
    policy: &ImaReplayPolicy,
    policy_digest: Option<String>,
    measurement_list_digest: String,
    expected_final_pcr: [u8; SHA256_LEN],
    issue: ImaReplayIssue,
) -> ImaReplayReport {
    ImaReplayReport {
        schema_version: IMA_REPLAY_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest,
        measurement_list_digest,
        expected_pcr: policy.expected_pcr,
        initial_pcr: policy.expected_initial_pcr,
        replayed_pcr: policy.expected_initial_pcr,
        expected_final_pcr,
        record_count: 0,
        selected_pcr_record_count: 0,
        template_counts: BTreeMap::new(),
        matched_required_measurements: Vec::new(),
        disposition: ImaReplayDisposition::Invalid,
        issues: vec![issue],
    }
}

fn read_u32(
    bytes: &[u8],
    cursor: &mut usize,
    record_index: u32,
    stage: &str,
) -> Result<u32, ImaReplayIssue> {
    let raw = read_bytes(bytes, cursor, 4, record_index, stage)?;
    Ok(u32::from_le_bytes(
        raw.try_into().expect("four bytes are present"),
    ))
}

fn read_array32(
    bytes: &[u8],
    cursor: &mut usize,
    record_index: u32,
    stage: &str,
) -> Result<[u8; SHA256_LEN], ImaReplayIssue> {
    let raw = read_bytes(bytes, cursor, SHA256_LEN, record_index, stage)?;
    Ok(raw.try_into().expect("32 bytes are present"))
}

fn read_bytes<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    len: usize,
    record_index: u32,
    stage: &str,
) -> Result<&'a [u8], ImaReplayIssue> {
    if bytes.len().saturating_sub(*cursor) < len {
        return Err(truncated(record_index, stage));
    }
    let start = *cursor;
    *cursor += len;
    Ok(&bytes[start..start + len])
}

fn truncated(record_index: u32, stage: &str) -> ImaReplayIssue {
    ImaReplayIssue::TruncatedRecord {
        record_index,
        stage: stage.to_string(),
    }
}

fn sha256_array(bytes: &[u8]) -> [u8; SHA256_LEN] {
    Sha256::digest(bytes).into()
}

fn extend_sha256(
    previous: [u8; SHA256_LEN],
    event: [u8; SHA256_LEN],
) -> [u8; SHA256_LEN] {
    let mut hasher = Sha256::new();
    hasher.update(previous);
    hasher.update(event);
    hasher.finalize().into()
}

fn bounded_list_digest(bytes: &[u8]) -> String {
    if bytes.len() as u64 > ABSOLUTE_MAX_MEASUREMENT_LIST_BYTES {
        "unavailable:absolute-measurement-list-limit".into()
    } else {
        digest_list(bytes)
    }
}

fn digest_list(bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(LIST_DIGEST_DOMAIN);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn qualification_digest(
    report: &ImaReplayReport,
    report_digest: &str,
    policy_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    push_field(&mut hasher, report_digest);
    push_field(&mut hasher, policy_digest);
    push_field(&mut hasher, &report.measurement_list_digest);
    push_u32(&mut hasher, report.expected_pcr);
    hasher.update(&report.replayed_pcr);
    push_u32(&mut hasher, report.record_count);
    push_u32(&mut hasher, report.selected_pcr_record_count);
    for measurement_id in &report.matched_required_measurements {
        push_field(&mut hasher, measurement_id);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT_BYTES
        && !value.chars().any(char::is_control)
}

fn valid_refs(values: &[String]) -> bool {
    !values.is_empty()
        && values.len() <= MAX_EVIDENCE_REFS
        && values.iter().all(|value| canonical_text(value))
        && values.iter().collect::<BTreeSet<_>>().len() == values.len()
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn push_u32(hasher: &mut blake3::Hasher, value: u32) {
    hasher.update(&value.to_le_bytes());
}

fn push_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.to_vec();
    refs.sort();
    for reference in refs {
        push_field(hasher, &reference);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN_HEX: &str = "0a000000e7236da53a1ad2d019fe5aab57da61002442b2cca64459b834d2ad94151f9d6306000000696d612d6e6763000000280000007368613235363a00ac6a53ce2193d74798af66f1749cab2c79157f6733aac20bc3666b1d32bd644a330000002f6e69782f73746f72652f61616161616161612d73796d74686165612d76657269666965722f62696e2f7665726966696572000a000000f6995fce7da011d68d46cd14185ffbb610ca6194cc2a37f1e2e7fb6462d8fc6a07000000696d612d73696777000000280000007368613235363a00aa7b55320f1f0507e52c681566e47ef4c752984b7a6f86d9f27aabfd331d367e320000002f6e69782f73746f72652f62626262626262622d76657269666965722d646570732f6c69622f6c696270726f6f662e736f0011000000030201746573742d7369676e6174757265";
    const GOLDEN_FINAL_PCR: [u8; 32] =
        hex32("093e0b931087c787da1b0561741c2cac75e749547ead4c6d7e71c4c13d2d746e");
    const VERIFIER_DIGEST: [u8; 32] =
        hex32("ac6a53ce2193d74798af66f1749cab2c79157f6733aac20bc3666b1d32bd644a");
    const VERIFIER_NAME_DIGEST: [u8; 32] =
        hex32("969022aed0ef2fcdb971244dde8ce36c6364eeb198a911671d6ac7114216da40");
    const DEP_DIGEST: [u8; 32] =
        hex32("aa7b55320f1f0507e52c681566e47ef4c752984b7a6f86d9f27aabfd331d367e");

    const fn hex32(value: &str) -> [u8; 32] {
        let bytes = value.as_bytes();
        let mut out = [0u8; 32];
        let mut index = 0;
        while index < 32 {
            out[index] = (nibble(bytes[index * 2]) << 4) | nibble(bytes[index * 2 + 1]);
            index += 1;
        }
        out
    }

    const fn nibble(value: u8) -> u8 {
        match value {
            b'0'..=b'9' => value - b'0',
            b'a'..=b'f' => value - b'a' + 10,
            _ => panic!("invalid hex"),
        }
    }

    fn decode_hex(value: &str) -> Vec<u8> {
        let bytes = value.as_bytes();
        assert_eq!(bytes.len() % 2, 0);
        (0..bytes.len() / 2)
            .map(|index| {
                (nibble(bytes[index * 2]) << 4) | nibble(bytes[index * 2 + 1])
            })
            .collect()
    }

    fn policy() -> ImaReplayPolicy {
        ImaReplayPolicy {
            schema_version: IMA_REPLAY_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:linux-ima-replay:golden-v1".into(),
            expected_pcr: 10,
            expected_initial_pcr: [0u8; 32],
            allowed_templates: vec![SupportedImaTemplate::ImaNg, SupportedImaTemplate::ImaSig],
            required_measurements: vec![
                RequiredImaMeasurement {
                    measurement_id: "verifier-executable".into(),
                    event_sha256: VERIFIER_DIGEST,
                    event_name_sha256: Some(VERIFIER_NAME_DIGEST),
                },
                RequiredImaMeasurement {
                    measurement_id: "proof-dependency".into(),
                    event_sha256: DEP_DIGEST,
                    event_name_sha256: None,
                },
            ],
            max_measurement_list_bytes: 1024 * 1024,
            max_records: 64,
            max_template_name_bytes: 64,
            max_template_data_bytes: 4096,
            max_field_bytes: 2048,
            evidence_refs: vec!["linux-kernel:ima-canonical-format".into()],
        }
    }

    #[test]
    fn checked_in_golden_vector_replays_to_frozen_sha256_pcr() {
        let bytes = decode_hex(GOLDEN_HEX);
        let qualified =
            verify_canonical_ima_sha256(&bytes, GOLDEN_FINAL_PCR, &policy()).unwrap();
        assert_eq!(qualified.report.record_count, 2);
        assert_eq!(qualified.report.selected_pcr_record_count, 2);
        assert_eq!(qualified.report.replayed_pcr, GOLDEN_FINAL_PCR);
        assert_eq!(qualified.report.disposition, ImaReplayDisposition::Qualified);
        assert_eq!(
            qualified.verified().matched_required_measurements(),
            &["proof-dependency".to_string(), "verifier-executable".to_string()]
        );
        assert!(!qualified.grants_physical_authority());
        assert!(!qualified.verified().grants_physical_authority());
    }

    #[test]
    fn truncation_is_structurally_invalid() {
        let mut bytes = decode_hex(GOLDEN_HEX);
        bytes.pop();
        let report = assess_canonical_ima_sha256(&bytes, GOLDEN_FINAL_PCR, &policy());
        assert_eq!(report.disposition, ImaReplayDisposition::Invalid);
        assert!(report
            .issues
            .iter()
            .any(|issue| matches!(issue, ImaReplayIssue::TruncatedRecord { .. })));
    }

    #[test]
    fn reordered_records_do_not_match_frozen_pcr() {
        let bytes = decode_hex(GOLDEN_HEX);
        let first_len = first_record_len(&bytes);
        let mut reordered = bytes[first_len..].to_vec();
        reordered.extend_from_slice(&bytes[..first_len]);
        let report = assess_canonical_ima_sha256(&reordered, GOLDEN_FINAL_PCR, &policy());
        assert_eq!(report.disposition, ImaReplayDisposition::Blocked);
        assert!(report.issues.contains(&ImaReplayIssue::PcrMismatch));
    }

    #[test]
    fn duplicate_insertion_changes_pcr_and_blocks() {
        let bytes = decode_hex(GOLDEN_HEX);
        let first_len = first_record_len(&bytes);
        let mut duplicated = bytes[..first_len].to_vec();
        duplicated.extend_from_slice(&bytes[..first_len]);
        duplicated.extend_from_slice(&bytes[first_len..]);
        let report = assess_canonical_ima_sha256(&duplicated, GOLDEN_FINAL_PCR, &policy());
        assert_eq!(report.disposition, ImaReplayDisposition::Blocked);
        assert!(report.issues.contains(&ImaReplayIssue::PcrMismatch));
    }

    #[test]
    fn template_data_tamper_is_invalid_before_pcr_comparison() {
        let mut bytes = decode_hex(GOLDEN_HEX);
        let marker = b"symthaea-verifier";
        let offset = bytes
            .windows(marker.len())
            .position(|window| window == marker)
            .unwrap();
        bytes[offset] ^= 1;
        let report = assess_canonical_ima_sha256(&bytes, GOLDEN_FINAL_PCR, &policy());
        assert_eq!(report.disposition, ImaReplayDisposition::Invalid);
        assert!(report
            .issues
            .iter()
            .any(|issue| matches!(issue, ImaReplayIssue::TemplateDigestMismatch { .. })));
    }

    #[test]
    fn unsupported_template_fails_closed() {
        let mut bytes = decode_hex(GOLDEN_HEX);
        let offset = bytes
            .windows(6)
            .position(|window| window == b"ima-ng")
            .unwrap();
        bytes[offset..offset + 6].copy_from_slice(b"ima-xx");
        let report = assess_canonical_ima_sha256(&bytes, GOLDEN_FINAL_PCR, &policy());
        assert_eq!(report.disposition, ImaReplayDisposition::Invalid);
        assert!(report
            .issues
            .iter()
            .any(|issue| matches!(issue, ImaReplayIssue::UnsupportedTemplate { .. })));
    }

    #[test]
    fn missing_required_artifact_blocks_even_when_pcr_matches() {
        let bytes = decode_hex(GOLDEN_HEX);
        let mut policy = policy();
        policy.required_measurements.push(RequiredImaMeasurement {
            measurement_id: "missing-config".into(),
            event_sha256: [7u8; 32],
            event_name_sha256: None,
        });
        let report = assess_canonical_ima_sha256(&bytes, GOLDEN_FINAL_PCR, &policy);
        assert_eq!(report.disposition, ImaReplayDisposition::Blocked);
        assert!(report.issues.iter().any(|issue| {
            matches!(
                issue,
                ImaReplayIssue::MissingRequiredMeasurement { measurement_id }
                    if measurement_id == "missing-config"
            )
        }));
    }

    #[test]
    fn event_name_binding_is_content_plus_name_not_path_alone() {
        let bytes = decode_hex(GOLDEN_HEX);
        let mut policy = policy();
        policy.required_measurements[0].event_name_sha256 = Some([9u8; 32]);
        let report = assess_canonical_ima_sha256(&bytes, GOLDEN_FINAL_PCR, &policy);
        assert_eq!(report.disposition, ImaReplayDisposition::Blocked);
        assert!(report.issues.iter().any(|issue| {
            matches!(
                issue,
                ImaReplayIssue::MissingRequiredMeasurement { measurement_id }
                    if measurement_id == "verifier-executable"
            )
        }));
    }

    #[test]
    fn measurement_list_resource_cap_fails_before_unbounded_processing() {
        let bytes = decode_hex(GOLDEN_HEX);
        let mut policy = policy();
        policy.max_measurement_list_bytes = (bytes.len() - 1) as u64;
        let report = assess_canonical_ima_sha256(&bytes, GOLDEN_FINAL_PCR, &policy);
        assert_eq!(report.disposition, ImaReplayDisposition::Invalid);
        assert!(matches!(
            report.issues.as_slice(),
            [ImaReplayIssue::MeasurementListTooLarge { .. }]
        ));
    }

    fn first_record_len(bytes: &[u8]) -> usize {
        let name_len = u32::from_le_bytes(bytes[36..40].try_into().unwrap()) as usize;
        let data_len_offset = 40 + name_len;
        let data_len = u32::from_le_bytes(
            bytes[data_len_offset..data_len_offset + 4]
                .try_into()
                .unwrap(),
        ) as usize;
        data_len_offset + 4 + data_len
    }
}
