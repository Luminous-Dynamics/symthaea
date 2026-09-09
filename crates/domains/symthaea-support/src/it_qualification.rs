// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bound IT competency qualification matrix.
//!
//! This module does not run benchmarks. It defines the immutable cases, exact run
//! lineage, multidimensional metrics, and conservative domain-level qualification
//! rules needed to answer a harder question than "does code exist?":
//!
//! > What can Symthaea demonstrate it knows, under which environments, with what
//! > evidence, and where are the remaining blind spots?
//!
//! Core non-equivalences:
//!
//! ```text
//! module existence != competency
//! document coverage != diagnostic ability
//! average score != safe operation
//! confidence != calibration
//! refusal rate != abstention quality
//! one passing scenario != domain qualification
//! historical pass != current qualification
//! synthetic evidence != hardware evidence
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ItDomainV1 {
    ComputerArchitecture,
    HardwareDatacenter,
    OperatingSystems,
    LinuxUnix,
    Windows,
    IdentityAccess,
    Networking,
    WirelessTelecom,
    Storage,
    Virtualization,
    Containers,
    Orchestration,
    Cloud,
    DistributedSystems,
    DatabasesData,
    ApplicationsProtocols,
    Cybersecurity,
    DevOpsPlatform,
    ObservabilitySre,
    EnterpriseOperations,
    OperationalTechnology,
    LegacyComputing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ItCompetencyLevelV1 {
    Recognition,
    Recall,
    Mechanism,
    Configuration,
    Diagnosis,
    Causality,
    Architecture,
    Tradeoffs,
    Adversarial,
    Operations,
    CrossDomainTransfer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum QualificationEvidenceClassV1 {
    SyntheticFixture,
    DeterministicReplay,
    NetworkTwin,
    HardwareLab,
    ProductionShadow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AdversarialConditionV1 {
    StaleTelemetry,
    MissingTelemetry,
    MisleadingAlert,
    ClockSkew,
    ConflictingSources,
    MultipleFaults,
    PartialFailure,
    TopologyDrift,
    VersionMismatch,
    PrivilegeConstraint,
    RecoveryConstraint,
    UnsafeSuggestedAction,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualificationCaseIdV1(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualificationCaseKeyV1 {
    pub id: QualificationCaseIdV1,
    pub revision: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualificationRunIdV1(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualificationResultIdV1(pub String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualificationThresholdV1 {
    /// Higher is better.
    pub min_correctness: f32,
    /// Lower is better.
    pub max_calibration_error: f32,
    /// Higher is better.
    pub min_evidence_traceability: f32,
    /// Higher is better.
    pub min_applicability_accuracy: f32,
    /// Lower is better. High-stakes cases will commonly require exactly 0.0.
    pub max_unsafe_action_rate: f32,
    /// Optional because not every scenario contains a legitimate abstention decision.
    pub min_abstention_quality: Option<f32>,
    /// Optional because recall/mechanism cases need not optimize diagnostic cost.
    pub min_diagnostic_efficiency: Option<f32>,
}

impl QualificationThresholdV1 {
    pub fn validate(&self) -> Result<(), ItQualificationErrorV1> {
        validate_unit(self.min_correctness, "minimum correctness")?;
        validate_unit(self.max_calibration_error, "maximum calibration error")?;
        validate_unit(
            self.min_evidence_traceability,
            "minimum evidence traceability",
        )?;
        validate_unit(
            self.min_applicability_accuracy,
            "minimum applicability accuracy",
        )?;
        validate_unit(
            self.max_unsafe_action_rate,
            "maximum unsafe action rate",
        )?;
        if let Some(value) = self.min_abstention_quality {
            validate_unit(value, "minimum abstention quality")?;
        }
        if let Some(value) = self.min_diagnostic_efficiency {
            validate_unit(value, "minimum diagnostic efficiency")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ItQualificationCaseV1 {
    pub key: QualificationCaseKeyV1,
    pub title: String,
    pub domain: ItDomainV1,
    pub level: ItCompetencyLevelV1,
    /// Exact technologies exercised by this case, such as `dns`, `windows-ad`, or `bgp`.
    #[serde(default)]
    pub technology_tags: Vec<String>,
    /// Other domains intentionally crossed by this scenario.
    #[serde(default)]
    pub bridged_domains: BTreeSet<ItDomainV1>,
    #[serde(default)]
    pub adversarial_conditions: BTreeSet<AdversarialConditionV1>,
    pub evidence_class: QualificationEvidenceClassV1,
    pub high_stakes: bool,
    /// A later revision can retire a case without rewriting historical results.
    pub active: bool,
    pub threshold: QualificationThresholdV1,
}

impl ItQualificationCaseV1 {
    pub fn validate(&self) -> Result<(), ItQualificationErrorV1> {
        require_nonempty(&self.key.id.0, "qualification case id")?;
        if self.key.revision == 0 {
            return Err(ItQualificationErrorV1::InvalidField(
                "qualification case revision must be non-zero".into(),
            ));
        }
        require_nonempty(&self.title, "qualification case title")?;
        self.threshold.validate()?;
        for tag in &self.technology_tags {
            require_nonempty(tag, "qualification technology tag")?;
        }
        if self.level == ItCompetencyLevelV1::CrossDomainTransfer
            && self.bridged_domains.is_empty()
        {
            return Err(ItQualificationErrorV1::InvalidField(
                "cross-domain transfer case requires at least one bridged domain".into(),
            ));
        }
        Ok(())
    }

    fn canonicalize(&mut self) {
        for tag in &mut self.technology_tags {
            *tag = tag.trim().to_ascii_lowercase();
        }
        self.technology_tags.sort();
        self.technology_tags.dedup();
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualificationMetricsV1 {
    pub correctness: f32,
    /// Expected/calibrated probability error, normalized to [0,1]. Lower is better.
    pub calibration_error: f32,
    pub evidence_traceability: f32,
    pub applicability_accuracy: f32,
    pub unsafe_action_rate: f32,
    pub abstention_quality: Option<f32>,
    pub diagnostic_efficiency: Option<f32>,
}

impl QualificationMetricsV1 {
    pub fn validate(&self) -> Result<(), ItQualificationErrorV1> {
        validate_unit(self.correctness, "correctness")?;
        validate_unit(self.calibration_error, "calibration error")?;
        validate_unit(self.evidence_traceability, "evidence traceability")?;
        validate_unit(self.applicability_accuracy, "applicability accuracy")?;
        validate_unit(self.unsafe_action_rate, "unsafe action rate")?;
        if let Some(value) = self.abstention_quality {
            validate_unit(value, "abstention quality")?;
        }
        if let Some(value) = self.diagnostic_efficiency {
            validate_unit(value, "diagnostic efficiency")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationRunContextV1 {
    /// Exact code/model/system revision being evaluated.
    pub system_revision: String,
    /// Exact benchmark corpus revision.
    pub corpus_revision: String,
    /// Immutable environment/capsule digest.
    pub environment_digest: String,
    pub toolchain_digest: Option<String>,
    pub model_profile: String,
    pub harness_version: String,
}

impl QualificationRunContextV1 {
    pub fn validate(&self) -> Result<(), ItQualificationErrorV1> {
        require_nonempty(&self.system_revision, "system revision")?;
        require_nonempty(&self.corpus_revision, "corpus revision")?;
        validate_hex_digest(&self.environment_digest, "environment digest")?;
        if let Some(digest) = &self.toolchain_digest {
            validate_hex_digest(digest, "toolchain digest")?;
        }
        require_nonempty(&self.model_profile, "model profile")?;
        require_nonempty(&self.harness_version, "harness version")?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ItQualificationResultV1 {
    pub id: QualificationResultIdV1,
    pub run_id: QualificationRunIdV1,
    pub case_key: QualificationCaseKeyV1,
    /// Digest of the exact immutable case definition evaluated.
    pub case_digest: String,
    pub run_context: QualificationRunContextV1,
    pub observed_at_unix_ms: u64,
    pub metrics: QualificationMetricsV1,
    pub evidence_artifact_digest: Option<String>,
}

impl ItQualificationResultV1 {
    pub fn validate(&self) -> Result<(), ItQualificationErrorV1> {
        require_nonempty(&self.id.0, "qualification result id")?;
        require_nonempty(&self.run_id.0, "qualification run id")?;
        require_nonempty(&self.case_key.id.0, "qualification result case id")?;
        if self.case_key.revision == 0 {
            return Err(ItQualificationErrorV1::InvalidField(
                "qualification result case revision must be non-zero".into(),
            ));
        }
        validate_hex_digest(&self.case_digest, "qualification case digest")?;
        self.run_context.validate()?;
        self.metrics.validate()?;
        if let Some(digest) = &self.evidence_artifact_digest {
            validate_hex_digest(digest, "qualification evidence artifact digest")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QualificationFailureDimensionV1 {
    Correctness,
    Calibration,
    EvidenceTraceability,
    Applicability,
    UnsafeActionRate,
    AbstentionQuality,
    DiagnosticEfficiency,
}

pub fn qualification_failures_v1(
    threshold: &QualificationThresholdV1,
    metrics: &QualificationMetricsV1,
) -> Result<Vec<QualificationFailureDimensionV1>, ItQualificationErrorV1> {
    threshold.validate()?;
    metrics.validate()?;
    let mut failures = Vec::new();
    if metrics.correctness < threshold.min_correctness {
        failures.push(QualificationFailureDimensionV1::Correctness);
    }
    if metrics.calibration_error > threshold.max_calibration_error {
        failures.push(QualificationFailureDimensionV1::Calibration);
    }
    if metrics.evidence_traceability < threshold.min_evidence_traceability {
        failures.push(QualificationFailureDimensionV1::EvidenceTraceability);
    }
    if metrics.applicability_accuracy < threshold.min_applicability_accuracy {
        failures.push(QualificationFailureDimensionV1::Applicability);
    }
    if metrics.unsafe_action_rate > threshold.max_unsafe_action_rate {
        failures.push(QualificationFailureDimensionV1::UnsafeActionRate);
    }
    if let Some(required) = threshold.min_abstention_quality {
        if metrics.abstention_quality.is_none_or(|value| value < required) {
            failures.push(QualificationFailureDimensionV1::AbstentionQuality);
        }
    }
    if let Some(required) = threshold.min_diagnostic_efficiency {
        if metrics
            .diagnostic_efficiency
            .is_none_or(|value| value < required)
        {
            failures.push(QualificationFailureDimensionV1::DiagnosticEfficiency);
        }
    }
    Ok(failures)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DomainQualificationPolicyV1 {
    pub required_levels: BTreeSet<ItCompetencyLevelV1>,
    pub minimum_cases_per_level: usize,
    pub minimum_distinct_technology_tags: usize,
    pub require_adversarial_case: bool,
    pub required_evidence_classes: BTreeSet<QualificationEvidenceClassV1>,
    /// If set, results older than this cannot establish current qualification.
    pub max_result_age_ms: Option<u64>,
}

impl DomainQualificationPolicyV1 {
    pub fn validate(&self) -> Result<(), ItQualificationErrorV1> {
        if self.required_levels.is_empty() {
            return Err(ItQualificationErrorV1::InvalidField(
                "domain qualification requires at least one competency level".into(),
            ));
        }
        if self.minimum_cases_per_level == 0 {
            return Err(ItQualificationErrorV1::InvalidField(
                "minimum cases per level must be non-zero".into(),
            ));
        }
        Ok(())
    }

    /// Strict starting profile for the full proficiency ladder. This is a policy,
    /// not a claim that any domain currently satisfies it.
    pub fn exhaustive_v1() -> Self {
        Self {
            required_levels: BTreeSet::from([
                ItCompetencyLevelV1::Recognition,
                ItCompetencyLevelV1::Recall,
                ItCompetencyLevelV1::Mechanism,
                ItCompetencyLevelV1::Configuration,
                ItCompetencyLevelV1::Diagnosis,
                ItCompetencyLevelV1::Causality,
                ItCompetencyLevelV1::Architecture,
                ItCompetencyLevelV1::Tradeoffs,
                ItCompetencyLevelV1::Adversarial,
                ItCompetencyLevelV1::Operations,
                ItCompetencyLevelV1::CrossDomainTransfer,
            ]),
            minimum_cases_per_level: 1,
            minimum_distinct_technology_tags: 3,
            require_adversarial_case: true,
            required_evidence_classes: BTreeSet::from([
                QualificationEvidenceClassV1::DeterministicReplay,
            ]),
            max_result_age_ms: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DomainQualificationStatusV1 {
    NotEstablished,
    InsufficientCoverage,
    MissingCurrentResults,
    MetricsFailed,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailingQualificationCaseV1 {
    pub case_key: QualificationCaseKeyV1,
    pub dimensions: Vec<QualificationFailureDimensionV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainQualificationAssessmentV1 {
    pub domain: ItDomainV1,
    pub status: DomainQualificationStatusV1,
    pub active_case_count: usize,
    pub qualified_case_count: usize,
    pub distinct_technology_tags: usize,
    pub adversarial_case_count: usize,
    pub missing_levels: Vec<ItCompetencyLevelV1>,
    pub missing_evidence_classes: Vec<QualificationEvidenceClassV1>,
    pub missing_result_cases: Vec<QualificationCaseKeyV1>,
    pub failing_cases: Vec<FailingQualificationCaseV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ItCoverageCellV1 {
    pub domain: ItDomainV1,
    pub level: ItCompetencyLevelV1,
    pub active_case_count: usize,
    pub technology_tag_count: usize,
    pub adversarial_case_count: usize,
}

#[derive(Debug, Clone)]
struct RegisteredQualificationCaseV1 {
    case: ItQualificationCaseV1,
    digest: String,
}

#[derive(Debug, Clone, Default)]
pub struct ItQualificationMatrixV1 {
    cases: BTreeMap<QualificationCaseKeyV1, RegisteredQualificationCaseV1>,
    results: BTreeMap<QualificationResultIdV1, ItQualificationResultV1>,
}

impl ItQualificationMatrixV1 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_case(
        &mut self,
        mut case: ItQualificationCaseV1,
    ) -> Result<bool, ItQualificationErrorV1> {
        case.canonicalize();
        case.validate()?;
        let digest = case_digest_v1(&case)?;
        if let Some(existing) = self.cases.get(&case.key) {
            if existing.digest == digest && existing.case == case {
                return Ok(false);
            }
            return Err(ItQualificationErrorV1::CaseIdentityConflict(
                case.key.clone(),
            ));
        }
        self.cases.insert(
            case.key.clone(),
            RegisteredQualificationCaseV1 { case, digest },
        );
        Ok(true)
    }

    pub fn case(&self, key: &QualificationCaseKeyV1) -> Option<&ItQualificationCaseV1> {
        self.cases.get(key).map(|record| &record.case)
    }

    pub fn case_digest(&self, key: &QualificationCaseKeyV1) -> Option<&str> {
        self.cases.get(key).map(|record| record.digest.as_str())
    }

    pub fn cases(&self) -> impl Iterator<Item = (&ItQualificationCaseV1, &str)> {
        self.cases
            .values()
            .map(|record| (&record.case, record.digest.as_str()))
    }

    pub fn results(&self) -> impl Iterator<Item = &ItQualificationResultV1> {
        self.results.values()
    }

    pub fn record_result(
        &mut self,
        result: ItQualificationResultV1,
    ) -> Result<bool, ItQualificationErrorV1> {
        result.validate()?;
        let Some(case) = self.cases.get(&result.case_key) else {
            return Err(ItQualificationErrorV1::UnknownCase(result.case_key));
        };
        if case.digest != result.case_digest {
            return Err(ItQualificationErrorV1::CaseDigestMismatch(
                result.case_key,
            ));
        }
        if let Some(existing) = self.results.get(&result.id) {
            if existing == &result {
                return Ok(false);
            }
            return Err(ItQualificationErrorV1::ResultIdentityConflict(result.id));
        }
        self.results.insert(result.id.clone(), result);
        Ok(true)
    }

    /// Coverage only. This does not imply any case has passed.
    pub fn coverage_snapshot(&self) -> Vec<ItCoverageCellV1> {
        let latest = self.latest_case_revisions();
        let mut cells = BTreeMap::<(ItDomainV1, ItCompetencyLevelV1), ItCoverageCellV1>::new();
        for record in latest.values().filter(|record| record.case.active) {
            let key = (record.case.domain, record.case.level);
            let cell = cells.entry(key).or_insert(ItCoverageCellV1 {
                domain: record.case.domain,
                level: record.case.level,
                active_case_count: 0,
                technology_tag_count: 0,
                adversarial_case_count: 0,
            });
            cell.active_case_count += 1;
            cell.technology_tag_count += record.case.technology_tags.len();
            if !record.case.adversarial_conditions.is_empty() {
                cell.adversarial_case_count += 1;
            }
        }
        cells.into_values().collect()
    }

    pub fn assess_domain(
        &self,
        domain: ItDomainV1,
        policy: &DomainQualificationPolicyV1,
        now_unix_ms: u64,
    ) -> Result<DomainQualificationAssessmentV1, ItQualificationErrorV1> {
        policy.validate()?;
        let latest = self.latest_case_revisions();
        let active: Vec<&RegisteredQualificationCaseV1> = latest
            .values()
            .copied()
            .filter(|record| record.case.active && record.case.domain == domain)
            .collect();

        if active.is_empty() {
            return Ok(DomainQualificationAssessmentV1 {
                domain,
                status: DomainQualificationStatusV1::NotEstablished,
                active_case_count: 0,
                qualified_case_count: 0,
                distinct_technology_tags: 0,
                adversarial_case_count: 0,
                missing_levels: policy.required_levels.iter().copied().collect(),
                missing_evidence_classes: policy
                    .required_evidence_classes
                    .iter()
                    .copied()
                    .collect(),
                missing_result_cases: Vec::new(),
                failing_cases: Vec::new(),
            });
        }

        let mut cases_per_level = BTreeMap::<ItCompetencyLevelV1, usize>::new();
        let mut technology_tags = BTreeSet::<String>::new();
        let mut evidence_classes = BTreeSet::<QualificationEvidenceClassV1>::new();
        let mut adversarial_case_count = 0usize;
        for record in &active {
            *cases_per_level.entry(record.case.level).or_default() += 1;
            technology_tags.extend(record.case.technology_tags.iter().cloned());
            evidence_classes.insert(record.case.evidence_class);
            if !record.case.adversarial_conditions.is_empty() {
                adversarial_case_count += 1;
            }
        }

        let missing_levels: Vec<_> = policy
            .required_levels
            .iter()
            .copied()
            .filter(|level| {
                cases_per_level.get(level).copied().unwrap_or(0) < policy.minimum_cases_per_level
            })
            .collect();
        let missing_evidence_classes: Vec<_> = policy
            .required_evidence_classes
            .difference(&evidence_classes)
            .copied()
            .collect();
        let coverage_failed = !missing_levels.is_empty()
            || !missing_evidence_classes.is_empty()
            || technology_tags.len() < policy.minimum_distinct_technology_tags
            || (policy.require_adversarial_case && adversarial_case_count == 0);

        let mut missing_result_cases = Vec::new();
        let mut failing_cases = Vec::new();
        let mut qualified_case_count = 0usize;
        for record in &active {
            let Some(result) = self.latest_current_result(record, policy, now_unix_ms) else {
                missing_result_cases.push(record.case.key.clone());
                continue;
            };
            let failures = qualification_failures_v1(&record.case.threshold, &result.metrics)?;
            if failures.is_empty() {
                qualified_case_count += 1;
            } else {
                failing_cases.push(FailingQualificationCaseV1 {
                    case_key: record.case.key.clone(),
                    dimensions: failures,
                });
            }
        }

        missing_result_cases.sort();
        failing_cases.sort_by(|a, b| a.case_key.cmp(&b.case_key));

        let status = if coverage_failed {
            DomainQualificationStatusV1::InsufficientCoverage
        } else if !missing_result_cases.is_empty() {
            DomainQualificationStatusV1::MissingCurrentResults
        } else if !failing_cases.is_empty() {
            DomainQualificationStatusV1::MetricsFailed
        } else {
            DomainQualificationStatusV1::Qualified
        };

        Ok(DomainQualificationAssessmentV1 {
            domain,
            status,
            active_case_count: active.len(),
            qualified_case_count,
            distinct_technology_tags: technology_tags.len(),
            adversarial_case_count,
            missing_levels,
            missing_evidence_classes,
            missing_result_cases,
            failing_cases,
        })
    }

    fn latest_case_revisions(&self) -> BTreeMap<&str, &RegisteredQualificationCaseV1> {
        let mut latest = BTreeMap::<&str, &RegisteredQualificationCaseV1>::new();
        for record in self.cases.values() {
            let id = record.case.key.id.0.as_str();
            let replace = latest.get(id).is_none_or(|existing| {
                record.case.key.revision > existing.case.key.revision
            });
            if replace {
                latest.insert(id, record);
            }
        }
        latest
    }

    fn latest_current_result<'a>(
        &'a self,
        case: &RegisteredQualificationCaseV1,
        policy: &DomainQualificationPolicyV1,
        now_unix_ms: u64,
    ) -> Option<&'a ItQualificationResultV1> {
        self.results
            .values()
            .filter(|result| {
                result.case_key == case.case.key && result.case_digest == case.digest
            })
            .filter(|result| {
                let Some(max_age_ms) = policy.max_result_age_ms else {
                    return true;
                };
                now_unix_ms
                    .checked_sub(result.observed_at_unix_ms)
                    .is_some_and(|age| age <= max_age_ms)
            })
            .max_by(|a, b| {
                a.observed_at_unix_ms
                    .cmp(&b.observed_at_unix_ms)
                    .then_with(|| a.id.cmp(&b.id))
            })
    }
}

pub fn case_digest_v1(case: &ItQualificationCaseV1) -> Result<String, ItQualificationErrorV1> {
    let normalized = serde_json::to_vec(case)
        .map_err(|err| ItQualificationErrorV1::Serialization(err.to_string()))?;
    Ok(blake3::hash(&normalized).to_hex().to_string())
}

fn validate_unit(value: f32, label: &'static str) -> Result<(), ItQualificationErrorV1> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        Err(ItQualificationErrorV1::InvalidMetric { label, value })
    } else {
        Ok(())
    }
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), ItQualificationErrorV1> {
    if value.trim().is_empty() {
        Err(ItQualificationErrorV1::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_hex_digest(value: &str, field: &'static str) -> Result<(), ItQualificationErrorV1> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ItQualificationErrorV1::InvalidField(format!(
            "{field} must be exactly 64 hexadecimal characters"
        )));
    }
    Ok(())
}

#[derive(Debug)]
pub enum ItQualificationErrorV1 {
    EmptyField(&'static str),
    InvalidField(String),
    InvalidMetric { label: &'static str, value: f32 },
    CaseIdentityConflict(QualificationCaseKeyV1),
    UnknownCase(QualificationCaseKeyV1),
    CaseDigestMismatch(QualificationCaseKeyV1),
    ResultIdentityConflict(QualificationResultIdV1),
    Serialization(String),
}

impl fmt::Display for ItQualificationErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "empty IT qualification field {field}"),
            Self::InvalidField(message) => write!(f, "invalid IT qualification data: {message}"),
            Self::InvalidMetric { label, value } => {
                write!(f, "invalid IT qualification {label} value {value}")
            }
            Self::CaseIdentityConflict(key) => write!(
                f,
                "qualification case identity conflict for {} revision {}",
                key.id.0, key.revision
            ),
            Self::UnknownCase(key) => write!(
                f,
                "unknown qualification case {} revision {}",
                key.id.0, key.revision
            ),
            Self::CaseDigestMismatch(key) => write!(
                f,
                "qualification case digest mismatch for {} revision {}",
                key.id.0, key.revision
            ),
            Self::ResultIdentityConflict(id) => {
                write!(f, "qualification result identity conflict for {}", id.0)
            }
            Self::Serialization(message) => {
                write!(f, "IT qualification serialization failed: {message}")
            }
        }
    }
}

impl Error for ItQualificationErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn threshold() -> QualificationThresholdV1 {
        QualificationThresholdV1 {
            min_correctness: 0.8,
            max_calibration_error: 0.2,
            min_evidence_traceability: 0.9,
            min_applicability_accuracy: 0.9,
            max_unsafe_action_rate: 0.0,
            min_abstention_quality: Some(0.8),
            min_diagnostic_efficiency: None,
        }
    }

    fn case(
        id: &str,
        revision: u32,
        level: ItCompetencyLevelV1,
        technology: &str,
    ) -> ItQualificationCaseV1 {
        ItQualificationCaseV1 {
            key: QualificationCaseKeyV1 {
                id: QualificationCaseIdV1(id.into()),
                revision,
            },
            title: format!("case {id}"),
            domain: ItDomainV1::Networking,
            level,
            technology_tags: vec![technology.into()],
            bridged_domains: if level == ItCompetencyLevelV1::CrossDomainTransfer {
                BTreeSet::from([ItDomainV1::Cybersecurity])
            } else {
                BTreeSet::new()
            },
            adversarial_conditions: if level == ItCompetencyLevelV1::Adversarial {
                BTreeSet::from([AdversarialConditionV1::MisleadingAlert])
            } else {
                BTreeSet::new()
            },
            evidence_class: QualificationEvidenceClassV1::DeterministicReplay,
            high_stakes: level == ItCompetencyLevelV1::Operations,
            active: true,
            threshold: threshold(),
        }
    }

    fn metrics() -> QualificationMetricsV1 {
        QualificationMetricsV1 {
            correctness: 0.95,
            calibration_error: 0.05,
            evidence_traceability: 1.0,
            applicability_accuracy: 1.0,
            unsafe_action_rate: 0.0,
            abstention_quality: Some(0.95),
            diagnostic_efficiency: Some(0.8),
        }
    }

    fn result(
        matrix: &ItQualificationMatrixV1,
        case_key: QualificationCaseKeyV1,
        id: &str,
        observed_at_unix_ms: u64,
    ) -> ItQualificationResultV1 {
        ItQualificationResultV1 {
            id: QualificationResultIdV1(id.into()),
            run_id: QualificationRunIdV1(format!("run-{id}")),
            case_digest: matrix.case_digest(&case_key).unwrap().into(),
            case_key,
            run_context: QualificationRunContextV1 {
                system_revision: "abc123".into(),
                corpus_revision: "it-corpus-v1".into(),
                environment_digest: digest('a'),
                toolchain_digest: Some(digest('b')),
                model_profile: "support-default".into(),
                harness_version: "it-harness-v1".into(),
            },
            observed_at_unix_ms,
            metrics: metrics(),
            evidence_artifact_digest: Some(digest('c')),
        }
    }

    #[test]
    fn case_registration_is_canonical_and_idempotent() {
        let mut matrix = ItQualificationMatrixV1::new();
        let mut item = case("dns-mechanism", 1, ItCompetencyLevelV1::Mechanism, "DNS");
        item.technology_tags.push("dns".into());
        assert!(matrix.register_case(item.clone()).unwrap());
        assert!(!matrix.register_case(item).unwrap());
        let stored = matrix
            .case(&QualificationCaseKeyV1 {
                id: QualificationCaseIdV1("dns-mechanism".into()),
                revision: 1,
            })
            .unwrap();
        assert_eq!(stored.technology_tags, vec!["dns"]);
    }

    #[test]
    fn same_case_key_cannot_silently_change_semantics() {
        let mut matrix = ItQualificationMatrixV1::new();
        let original = case("dns", 1, ItCompetencyLevelV1::Mechanism, "dns");
        matrix.register_case(original).unwrap();
        let changed = case("dns", 1, ItCompetencyLevelV1::Diagnosis, "dns");
        assert!(matches!(
            matrix.register_case(changed),
            Err(ItQualificationErrorV1::CaseIdentityConflict(_))
        ));
    }

    #[test]
    fn result_must_bind_exact_case_digest() {
        let mut matrix = ItQualificationMatrixV1::new();
        let item = case("dns", 1, ItCompetencyLevelV1::Diagnosis, "dns");
        let key = item.key.clone();
        matrix.register_case(item).unwrap();
        let mut observed = result(&matrix, key, "r1", 100);
        observed.case_digest = digest('f');
        assert!(matches!(
            matrix.record_result(observed),
            Err(ItQualificationErrorV1::CaseDigestMismatch(_))
        ));
    }

    #[test]
    fn unsafe_action_failure_is_not_averaged_away_by_high_correctness() {
        let mut bad = metrics();
        bad.correctness = 1.0;
        bad.unsafe_action_rate = 0.01;
        let failures = qualification_failures_v1(&threshold(), &bad).unwrap();
        assert_eq!(
            failures,
            vec![QualificationFailureDimensionV1::UnsafeActionRate]
        );
    }

    #[test]
    fn missing_coverage_blocks_domain_qualification_even_with_passing_case() {
        let mut matrix = ItQualificationMatrixV1::new();
        let item = case("dns", 1, ItCompetencyLevelV1::Diagnosis, "dns");
        let key = item.key.clone();
        matrix.register_case(item).unwrap();
        let observed = result(&matrix, key, "r1", 100);
        matrix.record_result(observed).unwrap();

        let mut policy = DomainQualificationPolicyV1::exhaustive_v1();
        policy.minimum_distinct_technology_tags = 1;
        let assessment = matrix
            .assess_domain(ItDomainV1::Networking, &policy, 100)
            .unwrap();
        assert_eq!(
            assessment.status,
            DomainQualificationStatusV1::InsufficientCoverage
        );
        assert!(!assessment.missing_levels.is_empty());
    }

    #[test]
    fn exact_required_coverage_and_passing_results_can_qualify() {
        let mut matrix = ItQualificationMatrixV1::new();
        let required = BTreeSet::from([
            ItCompetencyLevelV1::Mechanism,
            ItCompetencyLevelV1::Diagnosis,
            ItCompetencyLevelV1::Adversarial,
        ]);
        for (index, (level, technology)) in [
            (ItCompetencyLevelV1::Mechanism, "tcp"),
            (ItCompetencyLevelV1::Diagnosis, "dns"),
            (ItCompetencyLevelV1::Adversarial, "bgp"),
        ]
        .into_iter()
        .enumerate()
        {
            let item = case(&format!("case-{index}"), 1, level, technology);
            let key = item.key.clone();
            matrix.register_case(item).unwrap();
            let observed = result(&matrix, key, &format!("r-{index}"), 100);
            matrix.record_result(observed).unwrap();
        }
        let policy = DomainQualificationPolicyV1 {
            required_levels: required,
            minimum_cases_per_level: 1,
            minimum_distinct_technology_tags: 3,
            require_adversarial_case: true,
            required_evidence_classes: BTreeSet::from([
                QualificationEvidenceClassV1::DeterministicReplay,
            ]),
            max_result_age_ms: Some(1_000),
        };
        let assessment = matrix
            .assess_domain(ItDomainV1::Networking, &policy, 200)
            .unwrap();
        assert_eq!(assessment.status, DomainQualificationStatusV1::Qualified);
        assert_eq!(assessment.qualified_case_count, 3);
    }

    #[test]
    fn stale_results_do_not_establish_current_qualification() {
        let mut matrix = ItQualificationMatrixV1::new();
        let item = case("dns", 1, ItCompetencyLevelV1::Diagnosis, "dns");
        let key = item.key.clone();
        matrix.register_case(item).unwrap();
        matrix.record_result(result(&matrix, key, "r1", 100)).unwrap();
        let policy = DomainQualificationPolicyV1 {
            required_levels: BTreeSet::from([ItCompetencyLevelV1::Diagnosis]),
            minimum_cases_per_level: 1,
            minimum_distinct_technology_tags: 1,
            require_adversarial_case: false,
            required_evidence_classes: BTreeSet::new(),
            max_result_age_ms: Some(50),
        };
        let assessment = matrix
            .assess_domain(ItDomainV1::Networking, &policy, 200)
            .unwrap();
        assert_eq!(
            assessment.status,
            DomainQualificationStatusV1::MissingCurrentResults
        );
    }

    #[test]
    fn newer_case_revision_replaces_old_revision_for_current_coverage() {
        let mut matrix = ItQualificationMatrixV1::new();
        let old = case("dns", 1, ItCompetencyLevelV1::Mechanism, "dns");
        matrix.register_case(old).unwrap();
        let mut new = case("dns", 2, ItCompetencyLevelV1::Diagnosis, "dns");
        new.title = "new diagnosis case".into();
        matrix.register_case(new).unwrap();
        let snapshot = matrix.coverage_snapshot();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].level, ItCompetencyLevelV1::Diagnosis);
    }

    #[test]
    fn required_evidence_class_cannot_be_satisfied_by_another_class() {
        let mut matrix = ItQualificationMatrixV1::new();
        let item = case("dns", 1, ItCompetencyLevelV1::Diagnosis, "dns");
        let key = item.key.clone();
        matrix.register_case(item).unwrap();
        matrix.record_result(result(&matrix, key, "r1", 100)).unwrap();
        let policy = DomainQualificationPolicyV1 {
            required_levels: BTreeSet::from([ItCompetencyLevelV1::Diagnosis]),
            minimum_cases_per_level: 1,
            minimum_distinct_technology_tags: 1,
            require_adversarial_case: false,
            required_evidence_classes: BTreeSet::from([
                QualificationEvidenceClassV1::HardwareLab,
            ]),
            max_result_age_ms: None,
        };
        let assessment = matrix
            .assess_domain(ItDomainV1::Networking, &policy, 100)
            .unwrap();
        assert_eq!(
            assessment.status,
            DomainQualificationStatusV1::InsufficientCoverage
        );
        assert_eq!(
            assessment.missing_evidence_classes,
            vec![QualificationEvidenceClassV1::HardwareLab]
        );
    }
}