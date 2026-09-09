// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public-side schema and loader for reproducible IT golden incidents.
//!
//! Solver-visible scenario material is deliberately separated from oracle/ground-
//! truth material. The production support crate embeds only the public seed corpus;
//! oracle fixtures are included only in this module's tests. A future benchmark
//! harness should preserve the same process/file boundary.

use crate::it_qualification::{
    AdversarialConditionV1, ItCompetencyLevelV1, ItDomainV1, ItQualificationCaseV1,
    ItQualificationErrorV1, QualificationCaseIdV1, QualificationCaseKeyV1,
    QualificationEvidenceClassV1, QualificationThresholdV1,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const GOLDEN_INCIDENT_SCHEMA_V1: &str = "symthaea-it-golden-incidents-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoldenIncidentEvidenceKindV1 {
    UserReport,
    Configuration,
    ChangeEvent,
    Metric,
    Log,
    PacketSummary,
    Topology,
    SyntheticProbe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticActionRiskV1 {
    Passive,
    ReadOnly,
    LowRisk,
    Disruptive,
    Destructive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticAuthorityRequirementV1 {
    None,
    ReadOnly,
    ChangeProposalOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenIncidentEvidenceV1 {
    pub id: String,
    pub kind: GoldenIncidentEvidenceKindV1,
    pub summary: String,
    /// Relative event/observation time within the deterministic fixture.
    pub offset_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenDiagnosticActionV1 {
    pub id: String,
    pub description: String,
    pub information_goal: String,
    pub risk: DiagnosticActionRiskV1,
    pub authority: DiagnosticAuthorityRequirementV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenIncidentCaseV1 {
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
    pub initial_evidence: Vec<GoldenIncidentEvidenceV1>,
    pub diagnostic_actions: Vec<GoldenDiagnosticActionV1>,
}

impl GoldenIncidentCaseV1 {
    pub fn validate(&self) -> Result<(), GoldenIncidentErrorV1> {
        require_nonempty(&self.id, "golden incident id")?;
        if self.revision == 0 {
            return Err(GoldenIncidentErrorV1::InvalidField(
                "golden incident revision must be non-zero".into(),
            ));
        }
        require_nonempty(&self.title, "golden incident title")?;
        require_nonempty(&self.symptom, "golden incident symptom")?;
        self.threshold.validate()?;
        if self.initial_evidence.is_empty() {
            return Err(GoldenIncidentErrorV1::InvalidField(
                "golden incident requires initial evidence".into(),
            ));
        }
        if self.diagnostic_actions.is_empty() {
            return Err(GoldenIncidentErrorV1::InvalidField(
                "golden incident requires diagnostic actions".into(),
            ));
        }

        let mut evidence_ids = BTreeSet::new();
        for evidence in &self.initial_evidence {
            require_nonempty(&evidence.id, "golden evidence id")?;
            require_nonempty(&evidence.summary, "golden evidence summary")?;
            if !evidence_ids.insert(evidence.id.as_str()) {
                return Err(GoldenIncidentErrorV1::DuplicateEvidenceId(
                    evidence.id.clone(),
                ));
            }
        }

        let mut action_ids = BTreeSet::new();
        for action in &self.diagnostic_actions {
            require_nonempty(&action.id, "golden diagnostic action id")?;
            require_nonempty(&action.description, "golden diagnostic action description")?;
            require_nonempty(&action.information_goal, "golden diagnostic information goal")?;
            if !action_ids.insert(action.id.as_str()) {
                return Err(GoldenIncidentErrorV1::DuplicateActionId(
                    action.id.clone(),
                ));
            }
        }

        for tag in &self.technology_tags {
            require_nonempty(tag, "golden incident technology tag")?;
        }
        if self.level == ItCompetencyLevelV1::CrossDomainTransfer
            && self.bridged_domains.is_empty()
        {
            return Err(GoldenIncidentErrorV1::InvalidField(
                "cross-domain golden incident requires a bridged domain".into(),
            ));
        }
        Ok(())
    }

    pub fn qualification_case(&self) -> Result<ItQualificationCaseV1, GoldenIncidentErrorV1> {
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
pub struct GoldenIncidentCorpusV1 {
    pub schema_version: String,
    pub cases: Vec<GoldenIncidentCaseV1>,
}

impl GoldenIncidentCorpusV1 {
    pub fn validate(&self) -> Result<(), GoldenIncidentErrorV1> {
        if self.schema_version != GOLDEN_INCIDENT_SCHEMA_V1 {
            return Err(GoldenIncidentErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        if self.cases.is_empty() {
            return Err(GoldenIncidentErrorV1::InvalidField(
                "golden incident corpus must contain at least one case".into(),
            ));
        }
        let mut keys = BTreeSet::new();
        for case in &self.cases {
            case.validate()?;
            let key = (case.id.as_str(), case.revision);
            if !keys.insert(key) {
                return Err(GoldenIncidentErrorV1::DuplicateCaseKey {
                    id: case.id.clone(),
                    revision: case.revision,
                });
            }
        }
        Ok(())
    }
}

/// Built-in solver-visible seed corpus. Oracle/action-outcome data is not embedded
/// in this non-test function.
pub fn seed_golden_incidents_v1() -> Result<GoldenIncidentCorpusV1, GoldenIncidentErrorV1> {
    let corpus: GoldenIncidentCorpusV1 = serde_json::from_str(include_str!(
        "../data/it_golden_incidents_public_v1.json"
    ))
    .map_err(|err| GoldenIncidentErrorV1::Parse(err.to_string()))?;
    corpus.validate()?;
    Ok(corpus)
}

#[derive(Debug)]
pub enum GoldenIncidentErrorV1 {
    Parse(String),
    UnsupportedSchema(String),
    EmptyField(&'static str),
    InvalidField(String),
    DuplicateCaseKey { id: String, revision: u32 },
    DuplicateEvidenceId(String),
    DuplicateActionId(String),
    Qualification(ItQualificationErrorV1),
}

impl fmt::Display for GoldenIncidentErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(message) => write!(f, "golden incident parse failed: {message}"),
            Self::UnsupportedSchema(schema) => {
                write!(f, "unsupported golden incident schema {schema}")
            }
            Self::EmptyField(field) => write!(f, "empty golden incident field {field}"),
            Self::InvalidField(message) => write!(f, "invalid golden incident: {message}"),
            Self::DuplicateCaseKey { id, revision } => {
                write!(f, "duplicate golden incident {id} revision {revision}")
            }
            Self::DuplicateEvidenceId(id) => write!(f, "duplicate golden evidence id {id}"),
            Self::DuplicateActionId(id) => write!(f, "duplicate golden action id {id}"),
            Self::Qualification(err) => write!(f, "invalid golden qualification case: {err}"),
        }
    }
}

impl Error for GoldenIncidentErrorV1 {}

impl From<ItQualificationErrorV1> for GoldenIncidentErrorV1 {
    fn from(value: ItQualificationErrorV1) -> Self {
        Self::Qualification(value)
    }
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), GoldenIncidentErrorV1> {
    if value.trim().is_empty() {
        Err(GoldenIncidentErrorV1::EmptyField(field))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::it_qualification::ItQualificationMatrixV1;
    use std::collections::{BTreeMap, BTreeSet};

    #[derive(Debug, Deserialize)]
    struct OracleCorpusV1 {
        schema_version: String,
        oracles: Vec<OracleCaseV1>,
    }

    #[derive(Debug, Deserialize)]
    struct OracleCaseV1 {
        id: String,
        revision: u32,
        root_cause: String,
        causal_chain: Vec<String>,
        action_outcomes: BTreeMap<String, Vec<String>>,
        required_findings: Vec<String>,
        acceptable_remediations: Vec<String>,
        prohibited_remediations: Vec<String>,
        verification: Vec<String>,
    }

    fn seed_oracles() -> OracleCorpusV1 {
        serde_json::from_str(include_str!(
            "../data/oracle/it_golden_incidents_oracle_v1.json"
        ))
        .unwrap()
    }

    #[test]
    fn public_seed_corpus_is_valid_and_registers_into_qualification_matrix() {
        let corpus = seed_golden_incidents_v1().unwrap();
        assert!(corpus.cases.len() >= 4);
        let mut matrix = ItQualificationMatrixV1::new();
        for incident in &corpus.cases {
            matrix.register_case(incident.qualification_case().unwrap()).unwrap();
        }
        assert_eq!(matrix.cases().count(), corpus.cases.len());
    }

    #[test]
    fn oracle_keys_match_public_keys_and_only_reference_offered_actions() {
        let public = seed_golden_incidents_v1().unwrap();
        let oracle = seed_oracles();
        assert_eq!(oracle.schema_version, GOLDEN_INCIDENT_SCHEMA_V1);

        let public_by_key: BTreeMap<_, _> = public
            .cases
            .iter()
            .map(|case| ((case.id.as_str(), case.revision), case))
            .collect();
        let oracle_keys: BTreeSet<_> = oracle
            .oracles
            .iter()
            .map(|case| (case.id.as_str(), case.revision))
            .collect();
        let public_keys: BTreeSet<_> = public_by_key.keys().copied().collect();
        assert_eq!(oracle_keys, public_keys);

        for oracle_case in &oracle.oracles {
            let public_case = public_by_key[&(oracle_case.id.as_str(), oracle_case.revision)];
            let action_ids: BTreeSet<_> = public_case
                .diagnostic_actions
                .iter()
                .map(|action| action.id.as_str())
                .collect();
            assert!(oracle_case
                .action_outcomes
                .keys()
                .all(|action| action_ids.contains(action.as_str())));
        }
    }

    #[test]
    fn oracle_ground_truth_is_nonempty_and_not_verbatim_in_public_fixture() {
        let public = seed_golden_incidents_v1().unwrap();
        let oracle = seed_oracles();
        for oracle_case in &oracle.oracles {
            assert!(!oracle_case.root_cause.trim().is_empty());
            assert!(!oracle_case.causal_chain.is_empty());
            assert!(!oracle_case.required_findings.is_empty());
            assert!(!oracle_case.acceptable_remediations.is_empty());
            assert!(!oracle_case.prohibited_remediations.is_empty());
            assert!(!oracle_case.verification.is_empty());

            let public_case = public
                .cases
                .iter()
                .find(|case| case.id == oracle_case.id && case.revision == oracle_case.revision)
                .unwrap();
            let serialized = serde_json::to_string(public_case).unwrap().to_ascii_lowercase();
            assert!(!serialized.contains(&oracle_case.root_cause.to_ascii_lowercase()));
        }
    }

    #[test]
    fn every_seed_case_offers_at_least_one_read_only_test_and_one_risky_distractor() {
        let corpus = seed_golden_incidents_v1().unwrap();
        for case in corpus.cases {
            assert!(case.diagnostic_actions.iter().any(|action| {
                matches!(
                    action.risk,
                    DiagnosticActionRiskV1::Passive | DiagnosticActionRiskV1::ReadOnly
                )
            }));
            assert!(case.diagnostic_actions.iter().any(|action| {
                matches!(
                    action.risk,
                    DiagnosticActionRiskV1::Disruptive | DiagnosticActionRiskV1::Destructive
                )
            }));
        }
    }
}