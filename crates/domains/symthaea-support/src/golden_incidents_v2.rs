// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Golden IT incident corpus V2: multi-fault and currentness-aware public fixtures.
//!
//! V2 is additive. V1 remains immutable so historical qualification runs keep the
//! exact schema/case lineage they were executed against.
//!
//! Public fixtures still contain solver-visible evidence only. Ground truth,
//! action outcomes, accepted/prohibited remediation, and verification oracles
//! belong to a separate private benchmark artifact.

use crate::golden_incidents::{DiagnosticActionRiskV1, GoldenIncidentEvidenceKindV1};
use crate::it_qualification::{
    AdversarialConditionV1, ItCompetencyLevelV1, ItDomainV1, ItQualificationCaseV1,
    ItQualificationErrorV1, QualificationCaseIdV1, QualificationCaseKeyV1,
    QualificationEvidenceClassV1, QualificationThresholdV1,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const GOLDEN_INCIDENT_SCHEMA_V2: &str = "symthaea-it-golden-incidents-v2";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoldenEvidenceCurrentnessV2 {
    Fresh,
    Stale,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticAuthorityRequirementV2 {
    None,
    ReadOnly,
    /// Active probing may be proposed only through a separately bounded diagnostic
    /// authority path. This value is not itself execution authority.
    BoundedActiveDiagnostic,
    ChangeProposalOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenIncidentEvidenceV2 {
    pub id: String,
    pub kind: GoldenIncidentEvidenceKindV1,
    pub summary: String,
    pub offset_ms: i64,
    pub currentness: GoldenEvidenceCurrentnessV2,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenDiagnosticActionV2 {
    pub id: String,
    pub description: String,
    pub information_goal: String,
    pub risk: DiagnosticActionRiskV1,
    pub authority: DiagnosticAuthorityRequirementV2,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenIncidentCaseV2 {
    pub id: String,
    pub revision: u32,
    pub title: String,
    pub domain: ItDomainV1,
    pub level: ItCompetencyLevelV1,
    #[serde(default)]
    pub bridged_domains: BTreeSet<ItDomainV1>,
    #[serde(default)]
    pub technology_tags: Vec<String>,
    #[serde(default)]
    pub adversarial_conditions: BTreeSet<AdversarialConditionV1>,
    pub evidence_class: QualificationEvidenceClassV1,
    pub high_stakes: bool,
    pub symptom: String,
    pub threshold: QualificationThresholdV1,
    pub initial_evidence: Vec<GoldenIncidentEvidenceV2>,
    pub diagnostic_actions: Vec<GoldenDiagnosticActionV2>,
}

impl GoldenIncidentCaseV2 {
    pub fn validate(&self) -> Result<(), GoldenIncidentErrorV2> {
        require_nonempty(&self.id, "golden incident id")?;
        if self.revision == 0 {
            return Err(GoldenIncidentErrorV2::InvalidField(
                "golden incident revision must be non-zero".into(),
            ));
        }
        require_nonempty(&self.title, "golden incident title")?;
        require_nonempty(&self.symptom, "golden incident symptom")?;
        self.threshold.validate()?;

        if self.initial_evidence.is_empty() {
            return Err(GoldenIncidentErrorV2::InvalidField(
                "golden incident requires initial evidence".into(),
            ));
        }
        if self.diagnostic_actions.is_empty() {
            return Err(GoldenIncidentErrorV2::InvalidField(
                "golden incident requires diagnostic actions".into(),
            ));
        }
        if self.high_stakes && self.threshold.max_unsafe_action_rate != 0.0 {
            return Err(GoldenIncidentErrorV2::InvalidField(
                "high-stakes golden incidents require zero tolerated unsafe-action rate".into(),
            ));
        }
        if self.level == ItCompetencyLevelV1::CrossDomainTransfer && self.bridged_domains.is_empty() {
            return Err(GoldenIncidentErrorV2::InvalidField(
                "cross-domain golden incident requires a bridged domain".into(),
            ));
        }
        if self
            .adversarial_conditions
            .contains(&AdversarialConditionV1::StaleTelemetry)
            && !self
                .initial_evidence
                .iter()
                .any(|e| e.currentness == GoldenEvidenceCurrentnessV2::Stale)
        {
            return Err(GoldenIncidentErrorV2::InvalidField(
                "StaleTelemetry case must expose at least one stale evidence item".into(),
            ));
        }
        if self
            .adversarial_conditions
            .contains(&AdversarialConditionV1::MultipleFaults)
            && !matches!(
                self.level,
                ItCompetencyLevelV1::Causality
                    | ItCompetencyLevelV1::Adversarial
                    | ItCompetencyLevelV1::Operations
                    | ItCompetencyLevelV1::CrossDomainTransfer
            )
        {
            return Err(GoldenIncidentErrorV2::InvalidField(
                "MultipleFaults cases must exercise causality-or-higher competency".into(),
            ));
        }

        let mut evidence_ids = BTreeSet::new();
        for evidence in &self.initial_evidence {
            require_nonempty(&evidence.id, "golden evidence id")?;
            require_nonempty(&evidence.summary, "golden evidence summary")?;
            if !evidence_ids.insert(evidence.id.as_str()) {
                return Err(GoldenIncidentErrorV2::DuplicateEvidenceId(evidence.id.clone()));
            }
        }

        let mut action_ids = BTreeSet::new();
        for action in &self.diagnostic_actions {
            require_nonempty(&action.id, "golden diagnostic action id")?;
            require_nonempty(&action.description, "golden diagnostic action description")?;
            require_nonempty(&action.information_goal, "golden diagnostic information goal")?;
            if !action_ids.insert(action.id.as_str()) {
                return Err(GoldenIncidentErrorV2::DuplicateActionId(action.id.clone()));
            }
            if matches!(action.risk, DiagnosticActionRiskV1::Disruptive | DiagnosticActionRiskV1::Destructive)
                && action.authority != DiagnosticAuthorityRequirementV2::ChangeProposalOnly
            {
                return Err(GoldenIncidentErrorV2::InvalidField(format!(
                    "risky action {} must remain proposal-only",
                    action.id
                )));
            }
            if action.authority == DiagnosticAuthorityRequirementV2::BoundedActiveDiagnostic
                && matches!(action.risk, DiagnosticActionRiskV1::Passive | DiagnosticActionRiskV1::Destructive)
            {
                return Err(GoldenIncidentErrorV2::InvalidField(format!(
                    "bounded active diagnostic {} has incompatible risk class",
                    action.id
                )));
            }
        }

        for tag in &self.technology_tags {
            require_nonempty(tag, "golden incident technology tag")?;
        }
        Ok(())
    }

    pub fn qualification_case(&self) -> Result<ItQualificationCaseV1, GoldenIncidentErrorV2> {
        self.validate()?;
        Ok(ItQualificationCaseV1 {
            key: QualificationCaseKeyV1 {
                id: QualificationCaseIdV1(self.id.clone()),
                revision: self.revision,
            },
            title: self.title.clone(),
            domain: self.domain,
            level: self.level,
            technology_tags: self.technology_tags.clone(),
            bridged_domains: self.bridged_domains.clone(),
            adversarial_conditions: self.adversarial_conditions.clone(),
            evidence_class: self.evidence_class,
            high_stakes: self.high_stakes,
            active: true,
            threshold: self.threshold.clone(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenIncidentCorpusV2 {
    pub schema_version: String,
    pub cases: Vec<GoldenIncidentCaseV2>,
}

impl GoldenIncidentCorpusV2 {
    pub fn validate(&self) -> Result<(), GoldenIncidentErrorV2> {
        if self.schema_version != GOLDEN_INCIDENT_SCHEMA_V2 {
            return Err(GoldenIncidentErrorV2::UnsupportedSchema(self.schema_version.clone()));
        }
        if self.cases.is_empty() {
            return Err(GoldenIncidentErrorV2::InvalidField(
                "golden incident corpus must contain at least one case".into(),
            ));
        }
        let mut keys = BTreeSet::new();
        for case in &self.cases {
            case.validate()?;
            let key = (case.id.as_str(), case.revision);
            if !keys.insert(key) {
                return Err(GoldenIncidentErrorV2::DuplicateCaseKey {
                    id: case.id.clone(),
                    revision: case.revision,
                });
            }
        }
        Ok(())
    }
}

pub fn seed_golden_incidents_v2() -> Result<GoldenIncidentCorpusV2, GoldenIncidentErrorV2> {
    let corpus: GoldenIncidentCorpusV2 = serde_json::from_str(include_str!(
        "../data/it_golden_incidents_public_v2.json"
    ))
    .map_err(|err| GoldenIncidentErrorV2::Parse(err.to_string()))?;
    corpus.validate()?;
    Ok(corpus)
}

#[derive(Debug)]
pub enum GoldenIncidentErrorV2 {
    Parse(String),
    UnsupportedSchema(String),
    EmptyField(&'static str),
    InvalidField(String),
    DuplicateCaseKey { id: String, revision: u32 },
    DuplicateEvidenceId(String),
    DuplicateActionId(String),
    Qualification(ItQualificationErrorV1),
}

impl fmt::Display for GoldenIncidentErrorV2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(message) => write!(f, "golden incident V2 parse failed: {message}"),
            Self::UnsupportedSchema(schema) => write!(f, "unsupported golden incident V2 schema {schema}"),
            Self::EmptyField(field) => write!(f, "empty golden incident V2 field {field}"),
            Self::InvalidField(message) => write!(f, "invalid golden incident V2: {message}"),
            Self::DuplicateCaseKey { id, revision } => write!(f, "duplicate golden incident V2 {id} revision {revision}"),
            Self::DuplicateEvidenceId(id) => write!(f, "duplicate golden evidence V2 id {id}"),
            Self::DuplicateActionId(id) => write!(f, "duplicate golden action V2 id {id}"),
            Self::Qualification(err) => write!(f, "invalid golden qualification V2 case: {err}"),
        }
    }
}

impl Error for GoldenIncidentErrorV2 {}

impl From<ItQualificationErrorV1> for GoldenIncidentErrorV2 {
    fn from(value: ItQualificationErrorV1) -> Self {
        Self::Qualification(value)
    }
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), GoldenIncidentErrorV2> {
    if value.trim().is_empty() {
        Err(GoldenIncidentErrorV2::EmptyField(field))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::it_qualification::ItQualificationMatrixV1;

    #[test]
    fn v2_public_corpus_is_valid_and_registers() {
        let corpus = seed_golden_incidents_v2().unwrap();
        assert!(corpus.cases.len() >= 4);
        let mut matrix = ItQualificationMatrixV1::new();
        for incident in &corpus.cases {
            matrix.register_case(incident.qualification_case().unwrap()).unwrap();
        }
        assert_eq!(matrix.cases().count(), corpus.cases.len());
    }

    #[test]
    fn v2_public_fixture_contains_no_oracle_fields() {
        let value: serde_json::Value = serde_json::from_str(include_str!(
            "../data/it_golden_incidents_public_v2.json"
        ))
        .unwrap();
        let serialized_keys = collect_object_keys(&value);
        for forbidden in [
            "root_cause",
            "causal_chain",
            "action_outcomes",
            "required_findings",
            "acceptable_remediations",
            "prohibited_remediations",
            "verification",
            "oracle",
            "oracles",
        ] {
            assert!(!serialized_keys.contains(forbidden), "public V2 fixture leaked oracle field {forbidden}");
        }
    }

    #[test]
    fn v2_exercises_multifault_stale_and_bounded_active_reasoning() {
        let corpus = seed_golden_incidents_v2().unwrap();
        assert!(corpus.cases.iter().all(|case| case.adversarial_conditions.contains(&AdversarialConditionV1::MultipleFaults)));
        assert!(corpus.cases.iter().any(|case| case.adversarial_conditions.contains(&AdversarialConditionV1::StaleTelemetry)));
        assert!(corpus.cases.iter().any(|case| case.initial_evidence.iter().any(|e| e.currentness == GoldenEvidenceCurrentnessV2::Stale)));
        assert!(corpus.cases.iter().any(|case| case.diagnostic_actions.iter().any(|a| a.authority == DiagnosticAuthorityRequirementV2::BoundedActiveDiagnostic)));
    }

    #[test]
    fn v2_every_case_keeps_safe_evidence_path_and_risky_distractor() {
        let corpus = seed_golden_incidents_v2().unwrap();
        for case in corpus.cases {
            assert!(case.diagnostic_actions.iter().any(|action| matches!(action.risk, DiagnosticActionRiskV1::Passive | DiagnosticActionRiskV1::ReadOnly)));
            assert!(case.diagnostic_actions.iter().any(|action| matches!(action.risk, DiagnosticActionRiskV1::Disruptive | DiagnosticActionRiskV1::Destructive)));
        }
    }

    fn collect_object_keys(value: &serde_json::Value) -> BTreeSet<&str> {
        fn walk<'a>(value: &'a serde_json::Value, out: &mut BTreeSet<&'a str>) {
            match value {
                serde_json::Value::Object(map) => {
                    for (key, value) in map {
                        out.insert(key.as_str());
                        walk(value, out);
                    }
                }
                serde_json::Value::Array(values) => {
                    for value in values {
                        walk(value, out);
                    }
                }
                _ => {}
            }
        }
        let mut out = BTreeSet::new();
        walk(value, &mut out);
        out
    }
}
