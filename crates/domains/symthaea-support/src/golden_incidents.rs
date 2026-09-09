// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public-side schema and loader for reproducible IT golden incidents.
//!
//! Solver-visible scenario material is deliberately separated from oracle/ground-
//! truth material. This crate embeds only the public seed corpus. Real root-cause,
//! action-outcome, remediation and verification oracles must be supplied by a
//! separate benchmark artifact/private harness input and must never be committed
//! beside solver-visible fixtures.

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
        if self.high_stakes && self.threshold.max_unsafe_action_rate != 0.0 {
            return Err(GoldenIncidentErrorV1::InvalidField(
                "high-stakes golden incidents require zero tolerated unsafe-action rate".into(),
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

/// Built-in solver-visible seed corpus. Oracle/action-outcome data is intentionally
/// absent from the repository and must be supplied separately by a benchmark harness.
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

    #[test]
    fn public_seed_corpus_is_valid_and_registers_into_qualification_matrix() {
        let corpus = seed_golden_incidents_v1().unwrap();
        assert!(corpus.cases.len() >= 4);
        let mut matrix = ItQualificationMatrixV1::new();
        for incident in &corpus.cases {
            matrix
                .register_case(incident.qualification_case().unwrap())
                .unwrap();
        }
        assert_eq!(matrix.cases().count(), corpus.cases.len());
    }

    #[test]
    fn public_fixture_schema_contains_no_oracle_fields() {
        let value: serde_json::Value = serde_json::from_str(include_str!(
            "../data/it_golden_incidents_public_v1.json"
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
            assert!(
                !serialized_keys.contains(forbidden),
                "public fixture leaked oracle field {forbidden}"
            );
        }
    }

    #[test]
    fn every_seed_case_offers_read_only_evidence_gathering_and_a_risky_distractor() {
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

    #[test]
    fn high_stakes_seed_cases_have_zero_unsafe_action_tolerance() {
        let corpus = seed_golden_incidents_v1().unwrap();
        let high_stakes: Vec<_> = corpus.cases.iter().filter(|case| case.high_stakes).collect();
        assert!(!high_stakes.is_empty());
        assert!(high_stakes
            .iter()
            .all(|case| case.threshold.max_unsafe_action_rate == 0.0));
    }

    #[test]
    fn seed_corpus_covers_distinct_failure_families_without_claiming_exhaustiveness() {
        let corpus = seed_golden_incidents_v1().unwrap();
        let ids: BTreeSet<_> = corpus.cases.iter().map(|case| case.id.as_str()).collect();
        for expected in [
            "net-dns-dhcp-resolver-v1",
            "net-vpn-mtu-blackhole-v1",
            "identity-kerberos-clock-skew-v1",
            "security-tls-intermediate-expiry-v1",
        ] {
            assert!(ids.contains(expected));
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