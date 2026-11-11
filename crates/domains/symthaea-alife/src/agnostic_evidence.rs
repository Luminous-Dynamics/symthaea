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
pub const SYMTHAEA_OLI_CONTEXT_VERSION: &str = "symthaea-oli-context-v0.4.0";

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
    /// Qualified instrument or method capability evidence showing that a property
    /// cannot be resolved within the declared operating envelope.
    InstrumentLimitation,
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

/// Explicit reasons that a present property or biosignature could be missed.
///
/// These are observation-risk annotations, not assertions that the risk occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FalseNegativeRisk {
    LowAbundance,
    LowActivityOrDormancy,
    SpatialHeterogeneity,
    TemporalAliasing,
    PreservationOrDegradation,
    InstrumentSensitivity,
    MatrixInterference,
    SignatureMismatch,
    Unknown,
}

fn default_false_negative_risks() -> Vec<FalseNegativeRisk> {
    vec![FalseNegativeRisk::Unknown]
}

/// Observation opportunity and detector coverage for a non-detection or
/// not-observable assessment.
///
/// The record captures both what fraction was sampled and known reasons a real signal
/// could have been missed. A non-detection is never a global statement of absence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationCoverage {
    pub modality: String,
    pub scope: String,
    pub observation_duration_seconds: Option<f64>,
    /// Fraction of the relevant spatial domain sampled, in [0, 1].
    pub spatial_coverage_fraction: Option<f64>,
    /// Fraction of the relevant temporal window covered, in [0, 1].
    #[serde(default)]
    pub temporal_coverage_fraction: Option<f64>,
    /// Interval between samples/observations, in seconds.
    #[serde(default)]
    pub temporal_sampling_interval_seconds: Option<f64>,
    /// Characteristic timescale of the target process, in seconds.
    #[serde(default)]
    pub target_process_timescale_seconds: Option<f64>,
    /// Instrument- or method-specific detection limit.
    pub detection_limit: Option<f64>,
    /// Units for the detection limit, required when a limit is present.
    #[serde(default)]
    pub detection_limit_units: Option<String>,
    /// Reference abundance/concentration at which detection probability was calibrated.
    #[serde(default)]
    pub reference_abundance: Option<f64>,
    /// Empirically calibrated detection probability at the reference abundance, in [0, 1].
    #[serde(default)]
    pub detection_probability_at_reference: Option<f64>,
    /// Evidence record that documents the calibration curve or reference standard.
    #[serde(default)]
    pub calibration_evidence_id: Option<String>,
    /// Known or unresolved failure modes relevant to a false-negative interpretation.
    #[serde(default = "default_false_negative_risks")]
    pub false_negative_risks: Vec<FalseNegativeRisk>,
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
        for (name, value) in [
            ("temporal_sampling_interval_seconds", self.temporal_sampling_interval_seconds),
            ("target_process_timescale_seconds", self.target_process_timescale_seconds),
        ] {
            if let Some(value) = value && (!value.is_finite() || value <= 0.0) {
                issues.push(format!("{path}.{name} must be finite and > 0"));
            }
        }
        for (name, value) in [
            ("spatial_coverage_fraction", self.spatial_coverage_fraction),
            ("temporal_coverage_fraction", self.temporal_coverage_fraction),
        ] {
            if let Some(value) = value && (!value.is_finite() || !(0.0..=1.0).contains(&value)) {
                issues.push(format!("{path}.{name} must be finite and in [0, 1]"));
            }
        }
        if let Some(limit) = self.detection_limit
            && (!limit.is_finite() || limit < 0.0)
        {
            issues.push(format!("{path}.detection_limit must be finite and >= 0"));
        }
        if self.detection_limit.is_some()
            && self.detection_limit_units.as_ref().is_none_or(|units| units.trim().is_empty())
        {
            issues.push(format!("{path}.detection_limit requires non-empty units"));
        }
        if self.detection_limit.is_none() && self.detection_limit_units.is_some() {
            issues.push(format!("{path}.detection_limit_units requires a detection_limit"));
        }
        if let Some(reference) = self.reference_abundance
            && (!reference.is_finite() || reference < 0.0)
        {
            issues.push(format!("{path}.reference_abundance must be finite and >= 0"));
        }
        if let Some(probability) = self.detection_probability_at_reference
            && (!probability.is_finite() || !(0.0..=1.0).contains(&probability))
        {
            issues.push(format!("{path}.detection_probability_at_reference must be finite and in [0, 1]"));
        }
        if self.detection_probability_at_reference.is_some()
            && (self.reference_abundance.is_none()
                || self.calibration_evidence_id.as_ref().is_none_or(|id| id.trim().is_empty()))
        {
            issues.push(format!(
                "{path}.detection_probability_at_reference requires reference_abundance and calibration_evidence_id"
            ));
        }
        if self.detection_probability_at_reference.is_none()
            && (self.reference_abundance.is_some() || self.calibration_evidence_id.is_some())
        {
            issues.push(format!(
                "{path}.reference_abundance and calibration_evidence_id require detection_probability_at_reference"
            ));
        }
        if let (Some(interval), Some(timescale)) = (
            self.temporal_sampling_interval_seconds,
            self.target_process_timescale_seconds,
        ) {
            if interval > timescale
                && !self.false_negative_risks.contains(&FalseNegativeRisk::TemporalAliasing)
            {
                issues.push(format!(
                    "{path} sampling interval exceeds target process timescale and must record TemporalAliasing"
                ));
            }
        }
        let unique_risks: HashSet<_> = self.false_negative_risks.iter().collect();
        if unique_risks.len() != self.false_negative_risks.len() {
            issues.push(format!("{path}.false_negative_risks must not contain duplicates"));
        }
        if self.observation_duration_seconds.is_none()
            && self.spatial_coverage_fraction.is_none()
            && self.temporal_coverage_fraction.is_none()
            && self.detection_limit.is_none()
        {
            issues.push(format!(
                "{path} requires quantitative duration, spatial coverage, temporal coverage, or detection limit"
            ));
        }
    }
}

/// Adjudication state for a declared evidence conflict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictResolution {
    /// Conflicting observations are retained but do not justify a definitive disposition.
    Unresolved,
    /// The conflict has been adjudicated in favour of the supporting evidence.
    ResolvedForSupport,
    /// The conflict has been adjudicated in favour of the contradicting evidence.
    ResolvedForContradiction,
}

/// Explicitly records evidence on both sides of one scoped proposition.
///
/// `independently_sourced` is a claim about provenance groups, not a truth score. When
/// true, every record on both sides must declare a non-empty `independence_group`, and
/// the supporting and contradicting group sets must be disjoint. Distinct evidence IDs
/// alone never establish independent evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceConflict {
    pub proposition: String,
    pub supporting_evidence_ids: Vec<String>,
    pub contradicting_evidence_ids: Vec<String>,
    pub independently_sourced: bool,
    pub resolution: ConflictResolution,
    pub rationale: String,
}

impl EvidenceConflict {
    fn validate(
        &self,
        path: &str,
        evidence: &[EvidenceRecord],
        property_evidence_ids: &[String],
        assessment: PropertyAssessment,
        issues: &mut Vec<String>,
    ) {
        if self.proposition.trim().is_empty() {
            issues.push(format!("{path}.proposition must not be empty"));
        }
        if self.rationale.trim().is_empty() {
            issues.push(format!("{path}.rationale must not be empty"));
        }
        if self.supporting_evidence_ids.is_empty() {
            issues.push(format!("{path} requires supporting evidence ids"));
        }
        if self.contradicting_evidence_ids.is_empty() {
            issues.push(format!("{path} requires contradicting evidence ids"));
        }

        let mut seen = HashSet::<String>::new();
        let mut supporting_groups = HashSet::<String>::new();
        let mut contradicting_groups = HashSet::<String>::new();
        let mut supporting_artifacts = HashSet::<String>::new();
        let mut contradicting_artifacts = HashSet::<String>::new();
        let mut groups_complete = true;

        for id in &self.supporting_evidence_ids {
            if !seen.insert(id.clone()) {
                issues.push(format!(
                    "{path} evidence id {id} is duplicated or appears on both sides of the conflict"
                ));
            }
            if !property_evidence_ids.iter().any(|property_id| property_id == id) {
                issues.push(format!(
                    "{path} conflict evidence id {id} is not linked by the containing property finding"
                ));
            }
            match evidence.iter().find(|record| record.id == *id) {
                Some(record) => {
                    if let Some(digest) = &record.artifact_sha256 {
                        supporting_artifacts.insert(digest.to_ascii_lowercase());
                    }
                    match &record.independence_group {
                        Some(group) if !group.trim().is_empty() => {
                            supporting_groups.insert(group.clone());
                        }
                        _ => groups_complete = false,
                    }
                }
                None => issues.push(format!(
                    "{path} references unknown supporting evidence id: {id}"
                )),
            }
        }

        for id in &self.contradicting_evidence_ids {
            if !seen.insert(id.clone()) {
                issues.push(format!(
                    "{path} evidence id {id} is duplicated or appears on both sides of the conflict"
                ));
            }
            if !property_evidence_ids.iter().any(|property_id| property_id == id) {
                issues.push(format!(
                    "{path} conflict evidence id {id} is not linked by the containing property finding"
                ));
            }
            match evidence.iter().find(|record| record.id == *id) {
                Some(record) => {
                    if let Some(digest) = &record.artifact_sha256 {
                        contradicting_artifacts.insert(digest.to_ascii_lowercase());
                    }
                    match &record.independence_group {
                        Some(group) if !group.trim().is_empty() => {
                            contradicting_groups.insert(group.clone());
                        }
                        _ => groups_complete = false,
                    }
                }
                None => issues.push(format!(
                    "{path} references unknown contradicting evidence id: {id}"
                )),
            }
        }

        if self.independently_sourced {
            if !groups_complete {
                issues.push(format!(
                    "{path} cannot claim independent sourcing when an evidence record lacks a non-empty independence_group"
                ));
            }
            if !supporting_groups.is_disjoint(&contradicting_groups) {
                issues.push(format!(
                    "{path} cannot claim independent sourcing because supporting and contradicting evidence share a common-cause group"
                ));
            }
            if !supporting_artifacts.is_disjoint(&contradicting_artifacts) {
                issues.push(format!(
                    "{path} cannot claim independent sourcing because supporting and contradicting evidence share an identical artifact digest"
                ));
            }
        }

        match self.resolution {
            ConflictResolution::Unresolved => {
                if matches!(
                    assessment,
                    PropertyAssessment::SupportedWithinScope
                        | PropertyAssessment::ContradictedWithinScope
                ) {
                    issues.push(format!(
                        "{path} is unresolved and cannot accompany a definitive property disposition"
                    ));
                }
            }
            ConflictResolution::ResolvedForSupport => {
                if !self.independently_sourced {
                    issues.push(format!(
                        "{path} cannot resolve in favour of support without independently sourced evidence"
                    ));
                }
                if assessment != PropertyAssessment::SupportedWithinScope {
                    issues.push(format!(
                        "{path} resolves for support but the property assessment is not SupportedWithinScope"
                    ));
                }
            }
            ConflictResolution::ResolvedForContradiction => {
                if !self.independently_sourced {
                    issues.push(format!(
                        "{path} cannot resolve in favour of contradiction without independently sourced evidence"
                    ));
                }
                if assessment != PropertyAssessment::ContradictedWithinScope {
                    issues.push(format!(
                        "{path} resolves for contradiction but the property assessment is not ContradictedWithinScope"
                    ));
                }
            }
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
    /// Explicit supporting/contradicting evidence relationships and their adjudication.
    #[serde(default)]
    pub evidence_conflicts: Vec<EvidenceConflict>,
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
            if record
                .independence_group
                .as_ref()
                .is_some_and(|group| group.trim().is_empty())
            {
                issues.push(format!(
                    "{path}.independence_group must be omitted rather than empty"
                ));
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

            for (conflict_index, conflict) in finding.evidence_conflicts.iter().enumerate() {
                conflict.validate(
                    &format!("{path}.evidence_conflicts[{conflict_index}]"),
                    &self.evidence,
                    &finding.evidence_ids,
                    finding.assessment,
                    &mut issues,
                );
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
                    let has_instrument_limitation = finding.evidence_ids.iter().any(|id| {
                        self.evidence.iter().any(|record| {
                            record.id == *id
                                && record.outcome == EvidenceOutcome::InstrumentLimitation
                        })
                    });
                    if !has_instrument_limitation {
                        issues.push(format!(
                            "{path} NotObservable requires linked InstrumentLimitation evidence; InvalidMeasurement and NoDetection are not instrument-capability proofs"
                        ));
                    }
                    if finding.coverage.is_none() {
                        issues.push(format!(
                            "{path} NotObservable requires observation coverage"
                        ));
                    }
                    if finding.coverage.as_ref().is_some_and(|coverage| coverage.false_negative_risks.is_empty()) {
                        issues.push(format!(
                            "{path} NotObservable requires at least one explicit false-negative risk mode"
                        ));
                    }
                }
                PropertyAssessment::ModelInadequate => {
                    let has_adequacy_diagnostic = finding.evidence_ids.iter().any(|id| {
                        self.evidence.iter().any(|record| {
                            record.id == *id
                                && record.basis == EvidenceBasis::DerivedInference
                                && record.outcome == EvidenceOutcome::DerivedResult
                        })
                    });
                    if !has_adequacy_diagnostic {
                        issues.push(format!(
                            "{path} ModelInadequate requires a derived model-adequacy diagnostic"
                        ));
                    }
                }
                PropertyAssessment::InsufficientEvidence | PropertyAssessment::NotTested => {}
            }

            if let Some(coverage) = &finding.coverage {
                coverage.validate(&format!("{path}.coverage"), &mut issues);
                if let Some(calibration_id) = &coverage.calibration_evidence_id {
                    match self.evidence.iter().find(|record| record.id == *calibration_id) {
                        Some(record)
                            if record.outcome != EvidenceOutcome::InvalidMeasurement
                                && record.outcome != EvidenceOutcome::NotAcquired => {}
                        _ => issues.push(format!(
                            "{path}.coverage.calibration_evidence_id references missing or unusable evidence: {calibration_id}"
                        )),
                    }
                }
            }

            if no_detection {
                match &finding.coverage {
                    Some(coverage) if coverage.false_negative_risks.is_empty() => {
                        issues.push(format!(
                            "{path} NoDetection coverage requires at least one explicit false-negative risk mode"
                        ));
                    }
                    Some(_) => {}
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
