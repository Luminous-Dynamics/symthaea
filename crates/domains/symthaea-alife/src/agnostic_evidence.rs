// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence contract for the Open Life & Intelligence (OLI) research programme.
//!
//! This module is deliberately an epistemic data model, not a life detector. It keeps
//! observations, non-detections, contradictions, simulations, and hypotheses distinct.
//! In particular, `NotDetected` is never equivalent to `Absent`: a non-detection is
//! meaningful only together with its observation method and coverage context.
//!
//! The schema does not calculate a universal "life score". Properties of life-like
//! organization, agency, cognition, and collectives remain separately assessable.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

/// Current version of the experimental OLI interchange contract.
pub const OLI_SCHEMA_VERSION: &str = "0.1.0";

/// Independently testable system properties.
///
/// These axes are intentionally not a checklist that every living system must satisfy,
/// and no aggregate score is defined. A finding on one axis does not imply a finding on
/// another (for example, self-organization does not imply agency or life).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Property {
    // Organization and maintenance
    BoundaryMaintenance,
    EnergyCoupling,
    SelfMaintenance,
    MaterialTurnover,
    InformationPersistence,
    SelfRepair,
    // Adaptation and evolution
    EnvironmentalSensing,
    AdaptiveResponse,
    SelfProduction,
    Heredity,
    CumulativeEvolution,
    // Agency and cognition
    GoalDirectedAgency,
    CounterfactualSensitivity,
    Learning,
    Prediction,
    Generalization,
    ModelBasedPlanning,
    // Collective organization
    CollectiveCoordination,
}

/// State of evidence for one property.
///
/// `NotDetected` means only that the stated observation process did not detect the
/// property. It does not establish that the property is absent. `NotAssessed` means
/// that no adequate assessment has been made. These are intentionally different states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceState {
    Observed,
    NotDetected,
    Contradicted,
    Conflicted,
    NotAssessed,
    NotApplicable,
}

/// Provenance class for an evidence record.
///
/// These categories must not be silently collapsed: a simulation result is not a
/// laboratory observation, and a literature synthesis is not an independent replication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceBasis {
    DirectMeasurement,
    ControlledIntervention,
    Simulation,
    LiteratureSynthesis,
    DerivedInference,
}

/// An evidence record with enough provenance to audit its origin.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRecord {
    /// Stable identifier within this profile.
    pub id: String,
    pub basis: EvidenceBasis,
    /// Instrument, protocol, simulator, or analytical method.
    pub method: String,
    /// DOI, URL, repository path, archive identifier, or other source reference.
    pub source_ref: Option<String>,
    /// SHA-256 of the preserved source artifact, when available.
    pub artifact_sha256: Option<String>,
    /// Shared identifier for observations that are not statistically independent.
    pub independence_group: Option<String>,
}

/// Observation opportunity and detector coverage for a non-detection.
///
/// At least one quantitative coverage field is required when this is used to support a
/// `NotDetected` finding. The free-text scope records what the measurement actually
/// covered; it must not be used as a substitute for reporting known detection limits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationCoverage {
    pub modality: String,
    pub scope: String,
    pub observation_duration_seconds: Option<f64>,
    /// Fraction of the relevant spatial domain sampled, in [0, 1].
    pub spatial_coverage_fraction: Option<f64>,
    /// Instrument- or method-specific detection limit in the stated measurement units.
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
            issues.push(format!(
                "{path}.observation_duration_seconds must be finite and greater than zero"
            ));
        }

        if let Some(fraction) = self.spatial_coverage_fraction
            && (!fraction.is_finite() || !(0.0..=1.0).contains(&fraction))
        {
            issues.push(format!(
                "{path}.spatial_coverage_fraction must be finite and in [0, 1]"
            ));
        }

        if let Some(limit) = self.detection_limit
            && (!limit.is_finite() || limit < 0.0)
        {
            issues.push(format!("{path}.detection_limit must be finite and non-negative"));
        }

        if self.observation_duration_seconds.is_none()
            && self.spatial_coverage_fraction.is_none()
            && self.detection_limit.is_none()
        {
            issues.push(format!(
                "{path} must include a quantitative duration, spatial coverage, or detection limit"
            ));
        }
    }
}

/// Evidence attached to one independently assessed property.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropertyFinding {
    pub property: Property,
    pub state: EvidenceState,
    /// IDs of evidence records in the containing SystemProfile.
    pub evidence_ids: Vec<String>,
    /// Required for `NotDetected`; describes the opportunity to observe the property.
    pub coverage: Option<ObservationCoverage>,
    /// Scope, caveats, or rationale. This is not a substitute for evidence references.
    pub notes: String,
}

/// Broad mechanism family for an alternative explanation.
///
/// The labels are deliberately open to extension and do not assert that any one family
/// exhausts the space of possible systems.
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

/// A rival mechanism that can be challenged by an observation or intervention.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlternativeHypothesis {
    pub id: String,
    pub class: HypothesisClass,
    pub mechanism: String,
    /// Observable outcomes that would discriminate this hypothesis from its rivals.
    pub discriminating_predictions: Vec<String>,
    /// Evidence records relevant to this explanation.
    pub evidence_ids: Vec<String>,
}

/// A substrate-agnostic description of an investigated system.
///
/// `substrate_family` and `organizational_scale` are strings rather than closed enums:
/// unfamiliar substrates and scales must not require a schema change before they can be
/// recorded. This object contains no binary life verdict and no consciousness assertion.
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
    /// Validate referential integrity and the minimum epistemic requirements.
    ///
    /// Validation is intentionally structural: it does not decide whether a scientific
    /// interpretation is correct, whether evidence is independent, or whether a system
    /// qualifies as life or intelligence.
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
                issues.push(format!("{path}.artifact_sha256 must be 64 hexadecimal characters"));
            }
        }

        let mut properties = HashSet::new();
        for (index, finding) in self.findings.iter().enumerate() {
            let path = format!("findings[{index}]");
            if !properties.insert(finding.property) {
                issues.push(format!("duplicate property finding: {:?}", finding.property));
            }

            for evidence_id in &finding.evidence_ids {
                if !evidence_ids.contains(evidence_id.as_str()) {
                    issues.push(format!(
                        "{path} references unknown evidence id: {evidence_id}"
                    ));
                }
            }

            match finding.state {
                EvidenceState::Observed | EvidenceState::Contradicted => {
                    if finding.evidence_ids.is_empty() {
                        issues.push(format!(
                            "{path} in state {:?} requires at least one evidence record",
                            finding.state
                        ));
                    }
                }
                EvidenceState::NotDetected => {
                    if finding.evidence_ids.is_empty() {
                        issues.push(format!(
                            "{path} NotDetected requires a record of the measurement that did not detect the property"
                        ));
                    }
                    match &finding.coverage {
                        Some(coverage) => coverage.validate(
                            &format!("{path}.coverage"),
                            &mut issues,
                        ),
                        None => issues.push(format!(
                            "{path} NotDetected requires observation coverage"
                        )),
                    }
                }
                EvidenceState::Conflicted => {
                    if finding.evidence_ids.len() < 2 {
                        issues.push(format!(
                            "{path} Conflicted requires at least two evidence records"
                        ));
                    }
                }
                EvidenceState::NotAssessed | EvidenceState::NotApplicable => {}
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
