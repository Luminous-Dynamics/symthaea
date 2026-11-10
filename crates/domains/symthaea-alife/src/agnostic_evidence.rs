// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # OLI evidence adapter for Symthaea artificial-life research
//!
//! This is Symthaea's typed, evidence-linked interpretation layer. It is not a second
//! canonical fixture schema or a life detector. The interoperable M0 fixture contract and
//! JSON Schema are owned by Symtropy's `tools/oli-m0-validator`; this module deliberately
//! uses the same 16 property names and six property-assessment values.
//!
//! Raw observation outcomes are separate from property assessments. In particular,
//! `NoDetection` describes what a measurement did not detect under its declared coverage.
//! It is not a property state, and it is never equivalent to absence.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

/// Property names and states are aligned with OLI M0 fixture v0.1.
pub const SYMTHAEA_OLI_CONTEXT_VERSION: &str = "oli-m0-fixture-v0.1";

/// Substrate-agnostic properties using the canonical OLI M0 property keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Property {
    Organization,
    Boundary,
    EnergyMatterThroughput,
    Persistence,
    HomeostasisRegulation,
    SelfMaintenance,
    InformationRetention,
    InformationTransfer,
    Reproduction,
    Heredity,
    Adaptation,
    Learning,
    Agency,
    Intelligence,
    CollectiveIntelligence,
    SentienceConsciousness,
}

/// The six property-assessment values in the canonical OLI M0 contract.
///
/// These are scoped evidence dispositions, not a universal life or intelligence score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PropertyAssessment {
    SupportedWithinScope,
    ContradictedWithinScope,
    InsufficientEvidence,
    NotTested,
    NotObservableWithCurrentInstrument,
    ModelInadequate,
}

/// Provenance class of an evidence record.
///
/// These classes must not be collapsed: a simulation result is not a laboratory
/// observation, and a literature synthesis is not independent replication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceBasis {
    DirectMeasurement,
    ControlledIntervention,
    Simulation,
    LiteratureSynthesis,
    DerivedInference,
}

/// What happened in the observation or analysis record.
///
/// A non-detection is an observation outcome, not an assertion that a property is absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceOutcome {
    SignalDetected,
    NoDetection,
    InterventionResponse,
    SimulationResult,
    DerivedResult,
    InvalidMeasurement,
    NotAcquired,
}

/// Auditable evidence metadata.
///
/// `source_ref` may be a DOI, URL, repository path, archive identifier or other stable
/// reference. `artifact_sha256` binds an acquired artifact when one is available.
/// `independence_group` identifies known shared-source dependencies; a different record
/// ID alone must never be treated as proof of independent evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRecord {
    pub id: String,
    pub basis: EvidenceBasis,
    pub outcome: EvidenceOutcome,
    pub method: String,
    pub source_ref: Option<String>,
    pub artifact_sha256: Option<String>,
    pub independence_group: Option<String>,
}

/// Observation opportunity and detector coverage for a non-detection or
/// not-observable assessment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationCoverage {
    pub modality: String,
    pub scope: String,
    pub observation_duration_seconds: Option<f64>,
    /// Fraction of the relevant spatial domain sampled, in [0, 1].
    pub spatial_coverage_fraction: Option<f64>,
    /// Instrument- or method-specific detection limit in the stated units.
    pub detection_limit: Option<f64>,
}

impl ObservationCoverage {
    fn validate(&self, path: &str, issues: &mut Vec<String>) {
        if self.modality.trim().is_empty() {
            issues.push(format!("{path}.modality must not be empty"));
        }
        if self.scope.trim().is_empty() {
            issues.push(format!("{path}.scope must not be empty"));
        }
        if let Some(duration) = self.observation_duration_seconds
            && (!duration.is_finite() || duration <= 0.0)
        {
            issues.push(format!("{path}.observation_duration_seconds must be finite and > 0"));
        }
        if let Some(fraction) = self.spatial_coverage_fraction
            && (!fraction.is_finite() || !(0.0..=1.0).contains(&fraction))
        {
            issues.push(format!("{path}.spatial_coverage_fraction must be finite and in [0, 1]"));
        }
        if let Some(limit) = self.detection_limit
            && (!limit.is_finite() || limit < 0.0)
        {
            issues.push(format!("{path}.detection_limit must be finite and >= 0"));
        }
        if self.observation_duration_seconds.is_none()
            && self.spatial_coverage_fraction.is_none()
            && self.detection_limit.is_none()
        {
            issues.push(format!(
                "{path} requires quantitative duration, spatial coverage, or detection limit"
            ));
        }
    }
}

/// Evidence for one property, retaining the canonical M0 assessment vocabulary.
///
/// A `NoDetection` evidence record requires coverage metadata. It cannot be used as the
/// sole basis for `SupportedWithinScope`. This prevents absence-by-default and obvious
/// confidence laundering, while leaving scientific judgment about adequate detection
/// power to the benchmark protocol and independent reviewers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropertyFinding {
    pub property: Property,
    pub assessment: PropertyAssessment,
    /// IDs of evidence records in the containing SystemProfile.
    pub evidence_ids: Vec<String>,
    pub coverage: Option<ObservationCoverage>,
    pub notes: String,
}

/// Broad mechanism family for an alternative explanation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisClass {
    AbioticPhysical,
    EarthLikeBiological,
    AlternativeBiochemical,
    Artificial,
    CollectiveEmergence,
    Unknown,
    Other,
}

/// A rival mechanism that must make discriminating predictions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlternativeHypothesis {
    pub id: String,
    pub class: HypothesisClass,
    pub mechanism: String,
    pub discriminating_predictions: Vec<String>,
    pub evidence_ids: Vec<String>,
}

/// Symthaea's typed interpretation of a candidate system.
///
/// Substrate and organizational scale remain open strings so previously unimagined
/// substrates do not require an enum migration. This profile intentionally has no
/// `alive`, `intelligent`, or `conscious` scalar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemProfile {
    pub schema_version: String,
    pub system_id: String,
    pub substrate_family: String,
    pub organizational_scale: String,
    pub findings: Vec<PropertyFinding>,
    pub evidence: Vec<EvidenceRecord>,
    pub alternative_hypotheses: Vec<AlternativeHypothesis>,
}

impl SystemProfile {
    /// Validate structure and evidence references, not scientific truth.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut issues = Vec::new();

        if self.schema_version.trim().is_empty() {
            issues.push("schema_version must not be empty".to_owned());
        }
        if self.system_id.trim().is_empty() {
            issues.push("system_id must not be empty".to_owned());
        }
        if self.substrate_family.trim().is_empty() {
            issues.push("substrate_family must not be empty".to_owned());
        }
        if self.organizational_scale.trim().is_empty() {
            issues.push("organizational_scale must not be empty".to_owned());
        }

        let mut evidence_ids = HashSet::new();
        for (index, record) in self.evidence.iter().enumerate() {
            let path = format!("evidence[{index}]");
            if record.id.trim().is_empty() {
                issues.push(format!("{path}.id must not be empty"));
            } else if !evidence_ids.insert(record.id.as_str()) {
                issues.push(format!("duplicate evidence id: {}", record.id));
            }
            if record.method.trim().is_empty() {
                issues.push(format!("{path}.method must not be empty"));
            }
            if record
                .source_ref
                .as_ref()
                .is_some_and(|source| source.trim().is_empty())
            {
                issues.push(format!("{path}.source_ref must be omitted rather than empty"));
            }
            if let Some(digest) = &record.artifact_sha256
                && (digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()))
            {
                issues.push(format!(
                    "{path}.artifact_sha256 must contain 64 hexadecimal characters"
                ));
            }
        }

        let mut properties = HashSet::new();
        for (index, finding) in self.findings.iter().enumerate() {
            let path = format!("findings[{index}]");
            if !properties.insert(finding.property) {
                issues.push(format!("duplicate property finding: {:?}", finding.property));
            }

            let mut no_detection = false;
            for evidence_id in &finding.evidence_ids {
                match self.evidence.iter().find(|record| record.id == *evidence_id) {
                    Some(record) if record.outcome == EvidenceOutcome::NoDetection => {
                        no_detection = true;
                    }
                    Some(_) => {}
                    None => issues.push(format!(
                        "{path} references unknown evidence id: {evidence_id}"
                    )),
                }
            }

            match finding.assessment {
                PropertyAssessment::SupportedWithinScope
                | PropertyAssessment::ContradictedWithinScope => {
                    if finding.evidence_ids.is_empty() {
                        issues.push(format!(
                            "{path} assessment {:?} requires at least one evidence record",
                            finding.assessment
                        ));
                    }
                }
                PropertyAssessment::NotObservableWithCurrentInstrument => {
                    if finding.evidence_ids.is_empty() {
                        issues.push(format!(
                            "{path} NotObservable requires evidence about the instrument or measurement"
                        ));
                    }
                    if finding.coverage.is_none() {
                        issues.push(format!(
                            "{path} NotObservable requires observation coverage"
                        ));
                    }
                }
                PropertyAssessment::InsufficientEvidence
                | PropertyAssessment::NotTested
                | PropertyAssessment::ModelInadequate => {}
            }

            if no_detection {
                match &finding.coverage {
                    Some(coverage) => coverage.validate(&format!("{path}.coverage"), &mut issues),
                    None => issues.push(format!(
                        "{path} references NoDetection evidence and therefore requires coverage"
                    )),
                }
                if finding.assessment == PropertyAssessment::SupportedWithinScope {
                    let has_non_negative_basis = finding.evidence_ids.iter().any(|id| {
                        self.evidence.iter().any(|record| {
                            record.id == *id
                                && record.outcome != EvidenceOutcome::NoDetection
                                && record.outcome != EvidenceOutcome::InvalidMeasurement
                                && record.outcome != EvidenceOutcome::NotAcquired
                        })
                    });
                    if !has_non_negative_basis {
                        issues.push(format!(
                            "{path} cannot be SupportedWithinScope solely from NoDetection evidence"
                        ));
                    }
                }
            }
        }

        let mut hypothesis_ids = HashSet::new();
        for (index, hypothesis) in self.alternative_hypotheses.iter().enumerate() {
            let path = format!("alternative_hypotheses[{index}]");
            if hypothesis.id.trim().is_empty() {
                issues.push(format!("{path}.id must not be empty"));
            } else if !hypothesis_ids.insert(hypothesis.id.as_str()) {
                issues.push(format!("duplicate hypothesis id: {}", hypothesis.id));
            }
            if hypothesis.mechanism.trim().is_empty() {
                issues.push(format!("{path}.mechanism must not be empty"));
            }
            if hypothesis.discriminating_predictions.is_empty()
                || hypothesis
                    .discriminating_predictions
                    .iter()
                    .any(|prediction| prediction.trim().is_empty())
            {
                issues.push(format!(
                    "{path} requires non-empty discriminating predictions"
                ));
            }
            for evidence_id in &hypothesis.evidence_ids {
                if !evidence_ids.contains(evidence_id.as_str()) {
                    issues.push(format!(
                        "{path} references unknown evidence id: {evidence_id}"
                    ));
                }
            }
        }

        if issues.is_empty() {
            Ok(())
        } else {
            Err(issues)
        }
    }
}
