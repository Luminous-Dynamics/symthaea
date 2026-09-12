// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact-substance hazard evidence with caller-declared scoring policy.
//!
//! This crate never infers compound/material hazard from constituent elements.
//! A candidate must be explicitly bound to one exact source substance record.
//! Numerical scoring has no built-in weights: the caller supplies an immutable
//! policy JSON whose exact bytes and semantic content are both bound into the
//! receipt.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_discovery::{
    CandidateId, EvidenceKind, EvidenceRef, FidelityLevel, ModelProvenance, Prediction,
    UncertaintyEstimate,
};
use symthaea_energy_material_screening::unit;
use thiserror::Error;

pub const HAZARD_SCORE_METRIC: &str = "human_environmental_hazard_score";
pub const CAPABILITY_CLASSIFICATION: &str =
    "EXACT-SUBSTANCE HAZARD EVIDENCE ONLY -- policy-defined hazard score is not toxicological dose-response, exposure assessment, safety certification, or deployment authority.";
const DATASET_DIGEST_DOMAIN: &[u8] = b"symthaea.substance-hazard.dataset.v0\0";
const POLICY_DIGEST_DOMAIN: &[u8] = b"symthaea.substance-hazard.policy.v0\0";
const RECEIPT_DIGEST_DOMAIN: &[u8] = b"symthaea.substance-hazard.receipt.v0\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HazardDomain {
    HumanHealth,
    Environmental,
    Physical,
    Other,
}

impl HazardDomain {
    fn code(self) -> u8 {
        match self {
            Self::HumanHealth => 0,
            Self::Environmental => 1,
            Self::Physical => 2,
            Self::Other => 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HazardStatement {
    pub code: String,
    pub statement: String,
}

impl HazardStatement {
    fn validate(&self) -> Result<(), HazardError> {
        if self.code.trim().is_empty() || self.statement.trim().is_empty() {
            return Err(HazardError::InvalidDataset(
                "hazard code and statement must be non-empty".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubstanceHazardRecord {
    pub substance_id: String,
    pub name: String,
    #[serde(default)]
    pub hazards: Vec<HazardStatement>,
}

impl SubstanceHazardRecord {
    fn validate(&self) -> Result<(), HazardError> {
        if self.substance_id.trim().is_empty() || self.name.trim().is_empty() {
            return Err(HazardError::InvalidDataset(
                "substance_id and name must be non-empty".into(),
            ));
        }
        let mut codes = BTreeSet::new();
        for hazard in &self.hazards {
            hazard.validate()?;
            if !codes.insert(hazard.code.as_str()) {
                return Err(HazardError::InvalidDataset(format!(
                    "substance {:?} repeats hazard code {:?}",
                    self.substance_id, hazard.code
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HazardDataset {
    pub source_title: String,
    pub source_version: String,
    pub source_uri: String,
    pub source_document_sha256: String,
    pub normalized_capture_sha256: String,
    pub normalized_at_utc: String,
    pub extraction_note: String,
    pub records: Vec<SubstanceHazardRecord>,
}

#[derive(Debug, Deserialize)]
struct RawHazardDataset {
    source_title: String,
    source_version: String,
    source_uri: String,
    source_document_sha256: String,
    normalized_at_utc: String,
    extraction_note: String,
    records: Vec<SubstanceHazardRecord>,
}

impl HazardDataset {
    pub fn from_json_bytes(raw_json: &[u8]) -> Result<Self, HazardError> {
        let raw: RawHazardDataset = serde_json::from_slice(raw_json)?;
        let mut records = raw.records;
        records.sort_by(|left, right| left.substance_id.cmp(&right.substance_id));
        for record in &mut records {
            record.hazards.sort_by(|left, right| left.code.cmp(&right.code));
        }
        let dataset = Self {
            source_title: raw.source_title,
            source_version: raw.source_version,
            source_uri: raw.source_uri,
            source_document_sha256: raw.source_document_sha256,
            normalized_capture_sha256: sha256_bytes(raw_json),
            normalized_at_utc: raw.normalized_at_utc,
            extraction_note: raw.extraction_note,
            records,
        };
        dataset.validate()?;
        Ok(dataset)
    }

    pub fn validate(&self) -> Result<(), HazardError> {
        for (name, value) in [
            ("source_title", self.source_title.as_str()),
            ("source_version", self.source_version.as_str()),
            ("source_uri", self.source_uri.as_str()),
            ("normalized_at_utc", self.normalized_at_utc.as_str()),
            ("extraction_note", self.extraction_note.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(HazardError::InvalidDataset(format!("{name} cannot be blank")));
            }
        }
        for (name, digest) in [
            ("source_document_sha256", self.source_document_sha256.as_str()),
            ("normalized_capture_sha256", self.normalized_capture_sha256.as_str()),
        ] {
            if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(HazardError::InvalidDataset(format!(
                    "{name} must be 64 hexadecimal characters"
                )));
            }
        }
        if self.records.is_empty() {
            return Err(HazardError::InvalidDataset(
                "hazard dataset must contain at least one substance record".into(),
            ));
        }
        let mut ids = BTreeSet::new();
        for record in &self.records {
            record.validate()?;
            if !ids.insert(record.substance_id.as_str()) {
                return Err(HazardError::InvalidDataset(format!(
                    "duplicate substance id {:?}",
                    record.substance_id
                )));
            }
        }
        Ok(())
    }

    pub fn verify_source_document(&self, source_document: &[u8]) -> Result<(), HazardError> {
        self.validate()?;
        let actual = sha256_bytes(source_document);
        if !actual.eq_ignore_ascii_case(&self.source_document_sha256) {
            return Err(HazardError::SourceDocumentDigestMismatch {
                expected: self.source_document_sha256.clone(),
                actual,
            });
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, HazardError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(DATASET_DIGEST_DOMAIN);
        hasher.update(bytes);
        Ok(hex_lower(&hasher.finalize()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HazardAggregation {
    MaximumRuleWeight,
    SumRuleWeights,
}

impl HazardAggregation {
    fn code(self) -> u8 {
        match self {
            Self::MaximumRuleWeight => 0,
            Self::SumRuleWeights => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HazardRule {
    pub code: String,
    pub domain: HazardDomain,
    pub weight: f64,
    pub rationale: String,
}

impl HazardRule {
    fn validate(&self) -> Result<(), HazardError> {
        if self.code.trim().is_empty() || self.rationale.trim().is_empty() {
            return Err(HazardError::InvalidPolicy(
                "hazard rule code and rationale must be non-empty".into(),
            ));
        }
        if !self.weight.is_finite() || self.weight < 0.0 {
            return Err(HazardError::InvalidPolicy(format!(
                "hazard rule {:?} requires a finite non-negative weight",
                self.code
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HazardScoringPolicy {
    pub policy_id: String,
    pub rationale: String,
    pub aggregation: HazardAggregation,
    pub included_domains: Vec<HazardDomain>,
    pub rules: Vec<HazardRule>,
}

impl HazardScoringPolicy {
    pub fn from_json_bytes(raw_json: &[u8]) -> Result<(Self, String), HazardError> {
        let policy: Self = serde_json::from_slice(raw_json)?;
        policy.validate()?;
        Ok((policy, sha256_bytes(raw_json)))
    }

    pub fn validate(&self) -> Result<(), HazardError> {
        if self.policy_id.trim().is_empty() || self.rationale.trim().is_empty() {
            return Err(HazardError::InvalidPolicy(
                "policy_id and rationale must be non-empty".into(),
            ));
        }
        if self.included_domains.is_empty() {
            return Err(HazardError::InvalidPolicy(
                "at least one hazard domain must be included".into(),
            ));
        }
        let domains: BTreeSet<HazardDomain> = self.included_domains.iter().copied().collect();
        if domains.len() != self.included_domains.len() {
            return Err(HazardError::InvalidPolicy(
                "included_domains contains duplicates".into(),
            ));
        }
        if !domains.contains(&HazardDomain::HumanHealth)
            && !domains.contains(&HazardDomain::Environmental)
        {
            return Err(HazardError::InvalidPolicy(
                "Tier-1 hazard policy must include human_health and/or environmental hazards"
                    .into(),
            ));
        }
        if self.rules.is_empty() {
            return Err(HazardError::InvalidPolicy(
                "hazard policy requires at least one explicit rule".into(),
            ));
        }
        let mut codes = BTreeSet::new();
        for rule in &self.rules {
            rule.validate()?;
            if !codes.insert(rule.code.as_str()) {
                return Err(HazardError::InvalidPolicy(format!(
                    "duplicate hazard scoring rule {:?}",
                    rule.code
                )));
            }
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, HazardError> {
        self.validate()?;
        let mut domains = self.included_domains.clone();
        domains.sort();
        let mut rules = self.rules.clone();
        rules.sort_by(|left, right| left.code.cmp(&right.code));

        let mut hasher = Sha256::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
        update_text(&mut hasher, &self.policy_id);
        update_text(&mut hasher, &self.rationale);
        hasher.update([self.aggregation.code()]);
        for domain in domains {
            hasher.update([domain.code()]);
        }
        hasher.update([0xff]);
        for rule in rules {
            update_text(&mut hasher, &rule.code);
            hasher.update([rule.domain.code()]);
            hasher.update(rule.weight.to_bits().to_le_bytes());
            update_text(&mut hasher, &rule.rationale);
        }
        Ok(hex_lower(&hasher.finalize()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HazardBindingBasis {
    ExactSubstanceId,
    ExplicitMapping { note: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HazardBinding {
    pub candidate_id: CandidateId,
    pub substance_id: String,
    pub basis: HazardBindingBasis,
}

impl HazardBinding {
    pub fn validate(&self) -> Result<(), HazardError> {
        CandidateId::new(self.candidate_id.0.clone())?;
        if self.substance_id.trim().is_empty() {
            return Err(HazardError::InvalidBinding(
                "substance_id cannot be blank".into(),
            ));
        }
        match &self.basis {
            HazardBindingBasis::ExactSubstanceId => {
                if self.candidate_id.0 != self.substance_id {
                    return Err(HazardError::InvalidBinding(
                        "ExactSubstanceId requires candidate_id == substance_id".into(),
                    ));
                }
            }
            HazardBindingBasis::ExplicitMapping { note } => {
                if note.trim().is_empty() {
                    return Err(HazardError::InvalidBinding(
                        "explicit candidate/substance mapping requires a non-empty note".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredHazardStatement {
    pub code: String,
    pub statement: String,
    pub domain: HazardDomain,
    pub weight: f64,
    pub included_in_score: bool,
    pub rule_rationale: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HazardEvidenceReceipt {
    pub schema: String,
    pub capability_classification: String,
    pub binding: HazardBinding,
    pub substance_name: String,
    pub source_document_sha256: String,
    pub normalized_capture_sha256: String,
    pub hazard_dataset_sha256: String,
    pub policy_json_sha256: String,
    pub policy_sha256: String,
    pub policy_id: String,
    pub aggregation: HazardAggregation,
    pub included_domains: Vec<HazardDomain>,
    pub scored_statements: Vec<ScoredHazardStatement>,
    pub hazard_score: f64,
    pub prediction: Prediction,
}

impl HazardEvidenceReceipt {
    pub fn sha256(&self) -> Result<String, HazardError> {
        let bytes = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(RECEIPT_DIGEST_DOMAIN);
        hasher.update(bytes);
        Ok(hex_lower(&hasher.finalize()))
    }
}

pub fn calculate_hazard_evidence_from_bytes(
    normalized_dataset_json: &[u8],
    source_document: &[u8],
    policy_json: &[u8],
    binding: HazardBinding,
) -> Result<HazardEvidenceReceipt, HazardError> {
    let dataset = HazardDataset::from_json_bytes(normalized_dataset_json)?;
    dataset.verify_source_document(source_document)?;
    let (policy, policy_json_sha256) = HazardScoringPolicy::from_json_bytes(policy_json)?;
    binding.validate()?;

    let record = dataset
        .records
        .iter()
        .find(|record| record.substance_id == binding.substance_id)
        .ok_or_else(|| HazardError::SubstanceNotFound(binding.substance_id.clone()))?;

    let rules: BTreeMap<&str, &HazardRule> =
        policy.rules.iter().map(|rule| (rule.code.as_str(), rule)).collect();
    let included_domains: BTreeSet<HazardDomain> =
        policy.included_domains.iter().copied().collect();

    let mut scored_statements = Vec::with_capacity(record.hazards.len());
    let mut included_weights = Vec::new();
    for hazard in &record.hazards {
        let rule = rules
            .get(hazard.code.as_str())
            .copied()
            .ok_or_else(|| HazardError::UnmappedHazardCode(hazard.code.clone()))?;
        let included = included_domains.contains(&rule.domain);
        if included {
            included_weights.push(rule.weight);
        }
        scored_statements.push(ScoredHazardStatement {
            code: hazard.code.clone(),
            statement: hazard.statement.clone(),
            domain: rule.domain,
            weight: rule.weight,
            included_in_score: included,
            rule_rationale: rule.rationale.clone(),
        });
    }

    let hazard_score = match policy.aggregation {
        HazardAggregation::MaximumRuleWeight => included_weights
            .iter()
            .copied()
            .fold(0.0_f64, f64::max),
        HazardAggregation::SumRuleWeights => included_weights.iter().sum(),
    };
    if !hazard_score.is_finite() || hazard_score < 0.0 {
        return Err(HazardError::InvalidPolicy(
            "hazard aggregation produced an invalid score".into(),
        ));
    }

    let dataset_sha256 = dataset.sha256()?;
    let policy_sha256 = policy.sha256()?;
    let mut model = ModelProvenance::named("caller-declared exact-substance hazard policy")?;
    model.version = Some(format!(
        "v0;policy={};aggregation={:?}",
        policy.policy_id, policy.aggregation
    ));
    model.input_digest = Some(format!(
        "dataset-sha256:{dataset_sha256};policy-sha256:{policy_sha256}"
    ));

    let binding_assumption = match &binding.basis {
        HazardBindingBasis::ExactSubstanceId => {
            "Candidate identity exactly matches the source substance identifier.".to_owned()
        }
        HazardBindingBasis::ExplicitMapping { note } => format!(
            "Candidate-to-substance identity is caller-declared, not independently proven: {note}"
        ),
    };

    let prediction = Prediction {
        metric: HAZARD_SCORE_METRIC.into(),
        value: hazard_score,
        unit: unit::SCORE.into(),
        uncertainty: UncertaintyEstimate::new(1.0, 0.0)?,
        fidelity: FidelityLevel::Analytical,
        model,
        assumptions: vec![
            "The hazard score is a caller-declared policy transform over exact source hazard codes; it is not a universal toxicological scale.".into(),
            "No hazard is inferred from constituent elements. Evidence applies only to the explicitly bound source substance record.".into(),
            "Every source hazard code must have an explicit scoring rule, including codes excluded from the selected human/environmental domains. Unknown codes fail closed.".into(),
            "A score of zero can result from an empty source hazard list or explicit zero/excluded policy weights; it is not proof that the substance is hazard-free.".into(),
            "The score contains no dose, route, exposure, concentration, persistence, bioavailability, life-cycle, or use-context model unless encoded separately by the source and policy.".into(),
            "No calibrated predictive uncertainty is available for this policy transform; epistemic uncertainty is marked fully unknown.".into(),
            binding_assumption,
        ],
        evidence: vec![EvidenceRef {
            id: format!("substance-hazard:{}", record.substance_id),
            kind: EvidenceKind::Dataset,
            uri: Some(dataset.source_uri.clone()),
            digest: Some(format!("sha256:{}", dataset.source_document_sha256)),
            note: Some(format!(
                "normalized capture sha256={}; source version={}; extraction note={}",
                dataset.normalized_capture_sha256, dataset.source_version, dataset.extraction_note
            )),
        }],
    };
    prediction.validate()?;

    Ok(HazardEvidenceReceipt {
        schema: "symthaea.substance-hazard.receipt.v0".into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        binding,
        substance_name: record.name.clone(),
        source_document_sha256: dataset.source_document_sha256.clone(),
        normalized_capture_sha256: dataset.normalized_capture_sha256.clone(),
        hazard_dataset_sha256: dataset_sha256,
        policy_json_sha256,
        policy_sha256,
        policy_id: policy.policy_id,
        aggregation: policy.aggregation,
        included_domains: policy.included_domains,
        scored_statements,
        hazard_score,
        prediction,
    })
}

#[derive(Debug, Error)]
pub enum HazardError {
    #[error("invalid hazard dataset: {0}")]
    InvalidDataset(String),
    #[error("invalid hazard scoring policy: {0}")]
    InvalidPolicy(String),
    #[error("hazard source document SHA-256 mismatch: expected {expected}, got {actual}")]
    SourceDocumentDigestMismatch { expected: String, actual: String },
    #[error("invalid candidate/substance binding: {0}")]
    InvalidBinding(String),
    #[error("substance {0:?} is absent from the normalized hazard dataset")]
    SubstanceNotFound(String),
    #[error("source hazard code {0:?} has no explicit scoring-policy rule")]
    UnmappedHazardCode(String),
    #[error(transparent)]
    Discovery(#[from] symthaea_discovery::DiscoveryError),
    #[error("hazard JSON failed to parse/serialize: {0}")]
    Json(#[from] serde_json::Error),
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

fn update_text(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &[u8] = b"fixture authoritative hazard source";

    fn dataset_json() -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "source_title": "Fixture Substance Hazard Register",
            "source_version": "2026-fixture",
            "source_uri": "https://example.invalid/hazards",
            "source_document_sha256": sha256_bytes(SOURCE),
            "normalized_at_utc": "2026-09-12T00:00:00Z",
            "extraction_note": "test fixture only",
            "records": [
                {
                    "substance_id": "substance-1",
                    "name": "Fixture compound",
                    "hazards": [
                        {"code": "H-A", "statement": "fixture human-health hazard"},
                        {"code": "H-B", "statement": "fixture environmental hazard"},
                        {"code": "H-C", "statement": "fixture physical hazard"}
                    ]
                },
                {
                    "substance_id": "substance-clean-record",
                    "name": "Fixture record with no listed hazards",
                    "hazards": []
                }
            ]
        }))
        .unwrap()
    }

    fn policy_json(aggregation: HazardAggregation) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "policy_id": "fixture-human-environment-v0",
            "rationale": "fixture policy demonstrates explicit weights only",
            "aggregation": aggregation,
            "included_domains": ["human_health", "environmental"],
            "rules": [
                {"code": "H-A", "domain": "human_health", "weight": 3.0, "rationale": "fixture human weight"},
                {"code": "H-B", "domain": "environmental", "weight": 2.0, "rationale": "fixture environmental weight"},
                {"code": "H-C", "domain": "physical", "weight": 9.0, "rationale": "explicitly mapped but outside selected domains"}
            ]
        }))
        .unwrap()
    }

    fn explicit_binding() -> HazardBinding {
        HazardBinding {
            candidate_id: CandidateId::new("candidate-a").unwrap(),
            substance_id: "substance-1".into(),
            basis: HazardBindingBasis::ExplicitMapping {
                note: "fixture asserts exact substance identity for test only".into(),
            },
        }
    }

    #[test]
    fn caller_policy_controls_score_and_physical_hazard_is_not_silently_counted() {
        let receipt = calculate_hazard_evidence_from_bytes(
            &dataset_json(),
            SOURCE,
            &policy_json(HazardAggregation::MaximumRuleWeight),
            explicit_binding(),
        )
        .unwrap();
        assert_eq!(receipt.hazard_score, 3.0);
        let physical = receipt
            .scored_statements
            .iter()
            .find(|statement| statement.code == "H-C")
            .unwrap();
        assert!(!physical.included_in_score);
        assert_eq!(receipt.prediction.metric, HAZARD_SCORE_METRIC);
        assert_eq!(receipt.prediction.fidelity, FidelityLevel::Analytical);
        assert_eq!(receipt.prediction.evidence[0].kind, EvidenceKind::Dataset);
    }

    #[test]
    fn aggregation_policy_changes_method_and_score() {
        let max = calculate_hazard_evidence_from_bytes(
            &dataset_json(),
            SOURCE,
            &policy_json(HazardAggregation::MaximumRuleWeight),
            explicit_binding(),
        )
        .unwrap();
        let sum = calculate_hazard_evidence_from_bytes(
            &dataset_json(),
            SOURCE,
            &policy_json(HazardAggregation::SumRuleWeights),
            explicit_binding(),
        )
        .unwrap();
        assert_eq!(max.hazard_score, 3.0);
        assert_eq!(sum.hazard_score, 5.0);
        assert_ne!(max.policy_sha256, sum.policy_sha256);
    }

    #[test]
    fn unmapped_source_hazard_fails_closed_even_if_domain_might_be_excluded() {
        let mut value: serde_json::Value = serde_json::from_slice(&policy_json(
            HazardAggregation::MaximumRuleWeight,
        ))
        .unwrap();
        value["rules"].as_array_mut().unwrap().retain(|rule| rule["code"] != "H-C");
        let policy = serde_json::to_vec(&value).unwrap();
        assert!(matches!(
            calculate_hazard_evidence_from_bytes(
                &dataset_json(),
                SOURCE,
                &policy,
                explicit_binding(),
            ),
            Err(HazardError::UnmappedHazardCode(code)) if code == "H-C"
        ));
    }

    #[test]
    fn exact_substance_identity_cannot_be_assumed_from_candidate_name() {
        let binding = HazardBinding {
            candidate_id: CandidateId::new("candidate-a").unwrap(),
            substance_id: "substance-1".into(),
            basis: HazardBindingBasis::ExactSubstanceId,
        };
        assert!(calculate_hazard_evidence_from_bytes(
            &dataset_json(),
            SOURCE,
            &policy_json(HazardAggregation::MaximumRuleWeight),
            binding,
        )
        .is_err());
    }

    #[test]
    fn empty_source_hazard_list_can_score_zero_but_is_not_hazard_free_proof() {
        let binding = HazardBinding {
            candidate_id: CandidateId::new("substance-clean-record").unwrap(),
            substance_id: "substance-clean-record".into(),
            basis: HazardBindingBasis::ExactSubstanceId,
        };
        let receipt = calculate_hazard_evidence_from_bytes(
            &dataset_json(),
            SOURCE,
            &policy_json(HazardAggregation::MaximumRuleWeight),
            binding,
        )
        .unwrap();
        assert_eq!(receipt.hazard_score, 0.0);
        assert!(receipt.prediction.assumptions.iter().any(|assumption| {
            assumption.contains("not proof that the substance is hazard-free")
        }));
    }

    #[test]
    fn wrong_source_bytes_and_duplicate_policy_codes_fail_closed() {
        assert!(matches!(
            calculate_hazard_evidence_from_bytes(
                &dataset_json(),
                b"wrong bytes",
                &policy_json(HazardAggregation::MaximumRuleWeight),
                explicit_binding(),
            ),
            Err(HazardError::SourceDocumentDigestMismatch { .. })
        ));

        let mut value: serde_json::Value = serde_json::from_slice(&policy_json(
            HazardAggregation::MaximumRuleWeight,
        ))
        .unwrap();
        let duplicate = value["rules"][0].clone();
        value["rules"].as_array_mut().unwrap().push(duplicate);
        let policy = serde_json::to_vec(&value).unwrap();
        assert!(HazardScoringPolicy::from_json_bytes(&policy).is_err());
    }
}
