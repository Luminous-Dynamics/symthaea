// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! TPM2 fixed-property and runtime-subject qualification for assurance evidence.

#![deny(unsafe_code)]

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use symthaea_assurance_tpm2_tools_adapter::{
    ToolExecution, Tpm2AdapterError, Tpm2NvCounterObservation, Tpm2ToolsAdapterPolicy,
    Tpm2ToolsExecutor,
};

const QUALIFICATION_DIGEST_SCHEMA: &[u8] = b"symthaea-tpm2-platform-qualification-v1\0";
const RUNTIME_SUBJECT_DIGEST_SCHEMA: &[u8] = b"symthaea-tpm2-runtime-subject-v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tpm2RuntimeSubjectEvidence {
    pub adapter_artifact_ref: String,
    pub adapter_artifact_digest: String,
    pub runtime_closure_ref: String,
    pub runtime_closure_digest: String,
    pub verified_by_ref: String,
    pub verification_ref: String,
    pub verified_at_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl Tpm2RuntimeSubjectEvidence {
    pub fn validate(&self) -> bool {
        !self.adapter_artifact_ref.trim().is_empty()
            && valid_digest(&self.adapter_artifact_digest)
            && !self.runtime_closure_ref.trim().is_empty()
            && valid_digest(&self.runtime_closure_digest)
            && !self.verified_by_ref.trim().is_empty()
            && !self.verification_ref.trim().is_empty()
            && !self.evidence_refs.is_empty()
            && self
                .evidence_refs
                .iter()
                .all(|value| !value.trim().is_empty())
    }

    pub fn subject_digest(&self) -> String {
        let mut refs = self.evidence_refs.clone();
        refs.sort();
        let mut hasher = blake3::Hasher::new();
        hasher.update(RUNTIME_SUBJECT_DIGEST_SCHEMA);
        push_field(&mut hasher, &self.adapter_artifact_ref);
        push_field(&mut hasher, &self.adapter_artifact_digest);
        push_field(&mut hasher, &self.runtime_closure_ref);
        push_field(&mut hasher, &self.runtime_closure_digest);
        push_field(&mut hasher, &self.verified_by_ref);
        push_field(&mut hasher, &self.verification_ref);
        push_field(&mut hasher, &self.verified_at_ms.to_string());
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tpm2FixedProperties {
    pub family_indicator: u32,
    pub specification_revision: u32,
    pub manufacturer: u32,
    pub firmware_version_1: u32,
    pub firmware_version_2: u32,
    pub raw_output_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tpm2PlatformQualificationPolicy {
    pub schema_version: String,
    pub qualification_id: String,
    pub getcap_path: String,
    pub expected_getcap_blake3: String,
    pub expected_family_indicator: u32,
    pub expected_specification_revision: u32,
    pub expected_manufacturer: u32,
    pub expected_firmware_version_1: u32,
    pub expected_firmware_version_2: u32,
    pub evidence_refs: Vec<String>,
}

impl Tpm2PlatformQualificationPolicy {
    pub fn validate(&self) -> bool {
        !self.schema_version.trim().is_empty()
            && !self.qualification_id.trim().is_empty()
            && Path::new(&self.getcap_path).is_absolute()
            && valid_digest(&self.expected_getcap_blake3)
            && self.expected_family_indicator != 0
            && self.expected_specification_revision != 0
            && self.expected_manufacturer != 0
            && !self.evidence_refs.is_empty()
            && self
                .evidence_refs
                .iter()
                .all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tpm2PlatformQualificationRecord {
    pub schema_version: String,
    pub qualification_id: String,
    pub adapter_id: String,
    pub trust_store_ref: String,
    pub tcti: String,
    pub nv_index: u32,
    pub nv_name: String,
    pub before_observation_digest: String,
    pub after_observation_digest: String,
    pub before_counter_value: u64,
    pub after_counter_value: u64,
    pub fixed_properties: Tpm2FixedProperties,
    pub getcap_blake3: String,
    pub runtime_subject_digest: String,
    pub qualified_at_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl Tpm2PlatformQualificationRecord {
    pub fn qualification_digest(&self) -> String {
        let mut refs = self.evidence_refs.clone();
        refs.sort();
        let mut hasher = blake3::Hasher::new();
        hasher.update(QUALIFICATION_DIGEST_SCHEMA);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.qualification_id);
        push_field(&mut hasher, &self.adapter_id);
        push_field(&mut hasher, &self.trust_store_ref);
        push_field(&mut hasher, &self.tcti);
        push_field(&mut hasher, &format!("{:#x}", self.nv_index));
        push_field(&mut hasher, &self.nv_name);
        push_field(&mut hasher, &self.before_observation_digest);
        push_field(&mut hasher, &self.after_observation_digest);
        push_field(&mut hasher, &self.before_counter_value.to_string());
        push_field(&mut hasher, &self.after_counter_value.to_string());
        push_field(
            &mut hasher,
            &self.fixed_properties.family_indicator.to_string(),
        );
        push_field(
            &mut hasher,
            &self.fixed_properties.specification_revision.to_string(),
        );
        push_field(&mut hasher, &self.fixed_properties.manufacturer.to_string());
        push_field(
            &mut hasher,
            &self.fixed_properties.firmware_version_1.to_string(),
        );
        push_field(
            &mut hasher,
            &self.fixed_properties.firmware_version_2.to_string(),
        );
        push_field(&mut hasher, &self.fixed_properties.raw_output_digest);
        push_field(&mut hasher, &self.getcap_blake3);
        push_field(&mut hasher, &self.runtime_subject_digest);
        push_field(&mut hasher, &self.qualified_at_ms.to_string());
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tpm2PlatformQualificationError {
    InvalidPolicy,
    InvalidRuntimeSubject,
    RuntimeSubjectVerifiedAfterQualification,
    InvalidCounterObservation,
    ObservationOrderInvalid,
    QualificationNotBracketedByObservations,
    CounterRegressed,
    NvPublicIdentityChanged,
    Adapter(Tpm2AdapterError),
    GetcapDigestMismatch,
    GetcapFailed,
    GetcapEmittedStderr,
    InvalidGetcapOutput,
    DuplicateProperty(String),
    MissingProperty(&'static str),
    FamilyIndicatorMismatch,
    SpecificationRevisionMismatch,
    ManufacturerMismatch,
    FirmwareVersionMismatch,
}

pub fn qualify_tpm2_platform(
    policy: &Tpm2PlatformQualificationPolicy,
    adapter_policy: &Tpm2ToolsAdapterPolicy,
    runtime_subject: &Tpm2RuntimeSubjectEvidence,
    before: &Tpm2NvCounterObservation,
    after: &Tpm2NvCounterObservation,
    qualified_at_ms: u64,
    executor: &impl Tpm2ToolsExecutor,
) -> Result<Tpm2PlatformQualificationRecord, Tpm2PlatformQualificationError> {
    if !policy.validate() || !adapter_policy.validate() {
        return Err(Tpm2PlatformQualificationError::InvalidPolicy);
    }
    if !runtime_subject.validate() {
        return Err(Tpm2PlatformQualificationError::InvalidRuntimeSubject);
    }
    if runtime_subject.verified_at_ms > qualified_at_ms {
        return Err(Tpm2PlatformQualificationError::RuntimeSubjectVerifiedAfterQualification);
    }
    if !before.validate(adapter_policy) || !after.validate(adapter_policy) {
        return Err(Tpm2PlatformQualificationError::InvalidCounterObservation);
    }
    if after.observed_at_ms < before.observed_at_ms {
        return Err(Tpm2PlatformQualificationError::ObservationOrderInvalid);
    }
    if qualified_at_ms < before.observed_at_ms || qualified_at_ms > after.observed_at_ms {
        return Err(Tpm2PlatformQualificationError::QualificationNotBracketedByObservations);
    }
    if after.counter_value < before.counter_value {
        return Err(Tpm2PlatformQualificationError::CounterRegressed);
    }
    if before.public_evidence.nv_name != after.public_evidence.nv_name
        || before.nv_index != after.nv_index
        || before.trust_store_ref != after.trust_store_ref
        || before.counter_epoch != after.counter_epoch
    {
        return Err(Tpm2PlatformQualificationError::NvPublicIdentityChanged);
    }

    let getcap_digest = executor
        .executable_blake3(&policy.getcap_path)
        .map_err(Tpm2PlatformQualificationError::Adapter)?;
    if getcap_digest != policy.expected_getcap_blake3 {
        return Err(Tpm2PlatformQualificationError::GetcapDigestMismatch);
    }
    let run = executor
        .execute(
            &policy.getcap_path,
            &[
                "-T".into(),
                adapter_policy.tcti.clone(),
                "properties-fixed".into(),
            ],
        )
        .map_err(Tpm2PlatformQualificationError::Adapter)?;
    ensure_clean_success(&run)?;
    let fixed_properties = parse_fixed_properties(&run.stdout)?;
    check_expected_properties(policy, &fixed_properties)?;

    let mut evidence_refs = policy.evidence_refs.clone();
    evidence_refs.extend(runtime_subject.evidence_refs.iter().cloned());
    evidence_refs.sort();
    evidence_refs.dedup();

    Ok(Tpm2PlatformQualificationRecord {
        schema_version: "1".into(),
        qualification_id: policy.qualification_id.clone(),
        adapter_id: adapter_policy.adapter_id.clone(),
        trust_store_ref: adapter_policy.trust_store_ref.clone(),
        tcti: adapter_policy.tcti.clone(),
        nv_index: adapter_policy.nv_index,
        nv_name: adapter_policy.expected_nv_name.clone(),
        before_observation_digest: before.observation_digest(),
        after_observation_digest: after.observation_digest(),
        before_counter_value: before.counter_value,
        after_counter_value: after.counter_value,
        fixed_properties,
        getcap_blake3: getcap_digest,
        runtime_subject_digest: runtime_subject.subject_digest(),
        qualified_at_ms,
        evidence_refs,
    })
}

pub fn parse_fixed_properties(
    stdout: &[u8],
) -> Result<Tpm2FixedProperties, Tpm2PlatformQualificationError> {
    let text = std::str::from_utf8(stdout)
        .map_err(|_| Tpm2PlatformQualificationError::InvalidGetcapOutput)?;
    let mut current_property: Option<String> = None;
    let mut raw_values = BTreeMap::<String, u32>::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("TPM2_PT_") && trimmed.ends_with(':') {
            current_property = Some(trimmed.trim_end_matches(':').to_string());
            continue;
        }
        let Some(raw) = trimmed.strip_prefix("raw:") else {
            continue;
        };
        let Some(property) = current_property.as_ref() else {
            continue;
        };
        let value = parse_raw_u32(raw.trim())?;
        if raw_values.insert(property.clone(), value).is_some() {
            return Err(Tpm2PlatformQualificationError::DuplicateProperty(
                property.clone(),
            ));
        }
    }

    Ok(Tpm2FixedProperties {
        family_indicator: required(&raw_values, "TPM2_PT_FAMILY_INDICATOR")?,
        specification_revision: required(&raw_values, "TPM2_PT_REVISION")?,
        manufacturer: required(&raw_values, "TPM2_PT_MANUFACTURER")?,
        firmware_version_1: required(&raw_values, "TPM2_PT_FIRMWARE_VERSION_1")?,
        firmware_version_2: required(&raw_values, "TPM2_PT_FIRMWARE_VERSION_2")?,
        raw_output_digest: blake3_digest(stdout),
    })
}

fn check_expected_properties(
    policy: &Tpm2PlatformQualificationPolicy,
    properties: &Tpm2FixedProperties,
) -> Result<(), Tpm2PlatformQualificationError> {
    if properties.family_indicator != policy.expected_family_indicator {
        return Err(Tpm2PlatformQualificationError::FamilyIndicatorMismatch);
    }
    if properties.specification_revision != policy.expected_specification_revision {
        return Err(Tpm2PlatformQualificationError::SpecificationRevisionMismatch);
    }
    if properties.manufacturer != policy.expected_manufacturer {
        return Err(Tpm2PlatformQualificationError::ManufacturerMismatch);
    }
    if properties.firmware_version_1 != policy.expected_firmware_version_1
        || properties.firmware_version_2 != policy.expected_firmware_version_2
    {
        return Err(Tpm2PlatformQualificationError::FirmwareVersionMismatch);
    }
    Ok(())
}

fn ensure_clean_success(run: &ToolExecution) -> Result<(), Tpm2PlatformQualificationError> {
    if run.exit_code != Some(0) {
        return Err(Tpm2PlatformQualificationError::GetcapFailed);
    }
    if !run.stderr.is_empty() {
        return Err(Tpm2PlatformQualificationError::GetcapEmittedStderr);
    }
    Ok(())
}

fn parse_raw_u32(value: &str) -> Result<u32, Tpm2PlatformQualificationError> {
    if let Some(hex) = value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
        u32::from_str_radix(hex, 16)
            .map_err(|_| Tpm2PlatformQualificationError::InvalidGetcapOutput)
    } else {
        value
            .parse::<u32>()
            .map_err(|_| Tpm2PlatformQualificationError::InvalidGetcapOutput)
    }
}

fn required(
    values: &BTreeMap<String, u32>,
    key: &'static str,
) -> Result<u32, Tpm2PlatformQualificationError> {
    values
        .get(key)
        .copied()
        .ok_or(Tpm2PlatformQualificationError::MissingProperty(key))
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn blake3_digest(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

fn valid_digest(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|digest| {
        digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_assurance_tpm2_tools_adapter::{
        Tpm2NvPublicEvidence, Tpm2ReadHierarchy,
    };

    const GETCAP_TOOL: &[u8] = b"fake-tpm2-getcap-binary";
    const NAME: &str = "000bdeadbeef";

    #[derive(Debug, Clone)]
    struct FakeExecutor {
        getcap_stdout: Vec<u8>,
        getcap_digest: String,
    }

    impl Tpm2ToolsExecutor for FakeExecutor {
        fn executable_blake3(&self, executable: &str) -> Result<String, Tpm2AdapterError> {
            if executable == "/nix/store/getcap/bin/tpm2_getcap" {
                Ok(self.getcap_digest.clone())
            } else {
                Err(Tpm2AdapterError::Io(executable.into()))
            }
        }

        fn execute(
            &self,
            executable: &str,
            args: &[String],
        ) -> Result<ToolExecution, Tpm2AdapterError> {
            assert_eq!(executable, "/nix/store/getcap/bin/tpm2_getcap");
            assert_eq!(
                args,
                &["-T", "device:/dev/tpmrm0", "properties-fixed"]
            );
            Ok(ToolExecution {
                exit_code: Some(0),
                stdout: self.getcap_stdout.clone(),
                stderr: vec![],
            })
        }
    }

    fn fixed_output(firmware_1: u32) -> Vec<u8> {
        format!(
            "TPM2_PT_FAMILY_INDICATOR:\n  raw: 0x322E3000\n  value: \"2.0\"\nTPM2_PT_REVISION:\n  raw: 0xB9\n  value: 1.85\nTPM2_PT_MANUFACTURER:\n  raw: 0x49465800\n  value: \"IFX\"\nTPM2_PT_FIRMWARE_VERSION_1:\n  raw: 0x{firmware_1:X}\nTPM2_PT_FIRMWARE_VERSION_2:\n  raw: 0x1020304\n"
        )
        .into_bytes()
    }

    fn adapter_policy() -> Tpm2ToolsAdapterPolicy {
        Tpm2ToolsAdapterPolicy {
            schema_version: "1".into(),
            adapter_id: "adapter:1".into(),
            logical_store_id: "policy-root".into(),
            trust_store_ref: "trust-store:tpm2:1".into(),
            counter_epoch: "epoch:1".into(),
            nv_index: 0x0150_0016,
            expected_nv_name: NAME.into(),
            tcti: "device:/dev/tpmrm0".into(),
            read_hierarchy: Tpm2ReadHierarchy::Owner,
            nvreadpublic_path: "/nix/store/public/bin/tpm2_nvreadpublic".into(),
            nvread_path: "/nix/store/read/bin/tpm2_nvread".into(),
            expected_nvreadpublic_blake3: format!("blake3:{}", "1".repeat(64)),
            expected_nvread_blake3: format!("blake3:{}", "2".repeat(64)),
            minimum_counter_value: 1,
            evidence_refs: vec!["review:adapter".into()],
        }
    }

    fn observation(counter: u64, observed_at_ms: u64) -> Tpm2NvCounterObservation {
        let policy = adapter_policy();
        Tpm2NvCounterObservation {
            adapter_id: policy.adapter_id.clone(),
            logical_store_id: policy.logical_store_id.clone(),
            trust_store_ref: policy.trust_store_ref.clone(),
            counter_epoch: policy.counter_epoch.clone(),
            nv_index: policy.nv_index,
            tcti: policy.tcti.clone(),
            counter_value: counter,
            observed_at_ms,
            nvreadpublic_blake3: policy.expected_nvreadpublic_blake3,
            nvread_blake3: policy.expected_nvread_blake3,
            public_evidence: Tpm2NvPublicEvidence {
                nv_index: policy.nv_index,
                nv_name: NAME.into(),
                attributes_friendly: "ownerread|nt=counter".into(),
                data_size: 8,
                raw_public_output_blake3: format!("blake3:{}", "3".repeat(64)),
            },
            raw_counter_blake3: format!("blake3:{}", "4".repeat(64)),
            evidence_refs: vec!["observation:fixture".into()],
        }
    }

    fn runtime_subject() -> Tpm2RuntimeSubjectEvidence {
        Tpm2RuntimeSubjectEvidence {
            adapter_artifact_ref: "git:adapter-head".into(),
            adapter_artifact_digest: format!("blake3:{}", "5".repeat(64)),
            runtime_closure_ref: "nix:closure:tpm2-tools".into(),
            runtime_closure_digest: format!("blake3:{}", "6".repeat(64)),
            verified_by_ref: "verifier:closure".into(),
            verification_ref: "verification:closure:1".into(),
            verified_at_ms: 1_500,
            evidence_refs: vec!["audit:closure".into()],
        }
    }

    fn policy() -> Tpm2PlatformQualificationPolicy {
        Tpm2PlatformQualificationPolicy {
            schema_version: "1".into(),
            qualification_id: "tpm2-platform:node-1".into(),
            getcap_path: "/nix/store/getcap/bin/tpm2_getcap".into(),
            expected_getcap_blake3: blake3_digest(GETCAP_TOOL),
            expected_family_indicator: 0x322E_3000,
            expected_specification_revision: 0xB9,
            expected_manufacturer: 0x4946_5800,
            expected_firmware_version_1: 0x0047_000C,
            expected_firmware_version_2: 0x0102_0304,
            evidence_refs: vec!["review:platform-policy".into()],
        }
    }

    fn executor(firmware_1: u32) -> FakeExecutor {
        FakeExecutor {
            getcap_stdout: fixed_output(firmware_1),
            getcap_digest: blake3_digest(GETCAP_TOOL),
        }
    }

    #[test]
    fn exact_platform_and_runtime_subject_qualify() {
        let record = qualify_tpm2_platform(
            &policy(),
            &adapter_policy(),
            &runtime_subject(),
            &observation(42, 1_000),
            &observation(42, 2_000),
            1_800,
            &executor(0x0047_000C),
        )
        .unwrap();
        assert_eq!(record.fixed_properties.manufacturer, 0x4946_5800);
        assert_eq!(record.before_counter_value, 42);
        assert_eq!(record.after_counter_value, 42);
        assert!(record.qualification_digest().starts_with("blake3:"));
        assert!(!record.grants_physical_authority());
    }

    #[test]
    fn firmware_substitution_is_rejected() {
        assert_eq!(
            qualify_tpm2_platform(
                &policy(),
                &adapter_policy(),
                &runtime_subject(),
                &observation(42, 1_000),
                &observation(42, 2_000),
                1_800,
                &executor(0x0047_000D),
            ),
            Err(Tpm2PlatformQualificationError::FirmwareVersionMismatch)
        );
    }

    #[test]
    fn counter_regression_across_platform_query_is_rejected() {
        assert_eq!(
            qualify_tpm2_platform(
                &policy(),
                &adapter_policy(),
                &runtime_subject(),
                &observation(42, 1_000),
                &observation(41, 2_000),
                1_800,
                &executor(0x0047_000C),
            ),
            Err(Tpm2PlatformQualificationError::CounterRegressed)
        );
    }

    #[test]
    fn getcap_tool_drift_is_rejected() {
        let fake = FakeExecutor {
            getcap_stdout: fixed_output(0x0047_000C),
            getcap_digest: format!("blake3:{}", "f".repeat(64)),
        };
        assert_eq!(
            qualify_tpm2_platform(
                &policy(),
                &adapter_policy(),
                &runtime_subject(),
                &observation(42, 1_000),
                &observation(42, 2_000),
                1_800,
                &fake,
            ),
            Err(Tpm2PlatformQualificationError::GetcapDigestMismatch)
        );
    }

    #[test]
    fn invalid_runtime_closure_evidence_is_rejected() {
        let mut subject = runtime_subject();
        subject.runtime_closure_digest = "not-a-digest".into();
        assert_eq!(
            qualify_tpm2_platform(
                &policy(),
                &adapter_policy(),
                &subject,
                &observation(42, 1_000),
                &observation(42, 2_000),
                1_800,
                &executor(0x0047_000C),
            ),
            Err(Tpm2PlatformQualificationError::InvalidRuntimeSubject)
        );
    }

    #[test]
    fn parser_rejects_duplicate_required_property() {
        let mut output = fixed_output(0x0047_000C);
        output.extend_from_slice(b"TPM2_PT_MANUFACTURER:\n  raw: 0x49465800\n");
        assert_eq!(
            parse_fixed_properties(&output),
            Err(Tpm2PlatformQualificationError::DuplicateProperty(
                "TPM2_PT_MANUFACTURER".into()
            ))
        );
    }
}
