// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Cryptographically committed preregistration contracts for RCA experiments.
//!
//! The contract is frozen before result-bearing execution. It composes with the
//! existing `symthaea-evidence-plane::seed_plan::SeedPlan`, but its identity is a
//! BLAKE3 commitment over canonical serialized contract bytes. The evidence
//! plane's `config_hash()` is intentionally not used here because it is explicitly
//! non-cryptographic and unstable across Rust versions.

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashSet;
use symthaea_evidence_plane::seed_plan::SeedPlan;

pub const EXPERIMENT_CONTRACT_SCHEMA_VERSION: u16 = 1;
pub const MAX_CONFIDENCE_BPS: u16 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirectionV1 {
    HigherIsBetter,
    LowerIsBetter,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricSpecV1 {
    pub name: String,
    pub unit: String,
    pub direction: MetricDirectionV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedPointThresholdV1 {
    pub numerator: i64,
    pub scale: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentContractDraftV1 {
    pub schema_version: u16,
    pub experiment_id: String,
    pub registered_at_unix_ms: u64,
    pub registered_by: String,

    /// Digest of the exact hypothesis/specification being tested.
    pub hypothesis_digest: String,
    /// Qualified baseline code/config identity.
    pub baseline_digest: String,
    /// Candidate code/config identity.
    pub candidate_digest: String,
    pub development_corpus_digest: String,
    pub held_out_corpus_digest: String,
    pub evaluator_digest: String,

    pub primary_metric: MetricSpecV1,
    #[serde(default)]
    pub secondary_metrics: Vec<MetricSpecV1>,
    pub minimum_meaningful_effect: FixedPointThresholdV1,
    /// Statistical confidence requirement in basis points, 1..=10_000.
    pub confidence_bps: u16,
    pub compute_ceiling_microunits: Option<u64>,
    pub wall_time_ceiling_ms: Option<u64>,

    /// Falsification criteria are mandatory and frozen before execution.
    pub falsification_criteria: Vec<String>,
    /// All legitimate outcome interpretations must be declared before results.
    pub allowed_outcomes: Vec<String>,
}

impl ExperimentContractDraftV1 {
    pub fn validate(&self) -> Result<(), ExperimentContractError> {
        if self.schema_version != EXPERIMENT_CONTRACT_SCHEMA_VERSION {
            return Err(ExperimentContractError::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }
        require_nonempty(&self.experiment_id, "experiment_id")?;
        require_nonempty(&self.registered_by, "registered_by")?;
        for digest in [
            &self.hypothesis_digest,
            &self.baseline_digest,
            &self.candidate_digest,
            &self.development_corpus_digest,
            &self.held_out_corpus_digest,
            &self.evaluator_digest,
        ] {
            validate_digest(digest)?;
        }
        if self.baseline_digest == self.candidate_digest {
            return Err(ExperimentContractError::BaselineEqualsCandidate);
        }
        if self.development_corpus_digest == self.held_out_corpus_digest {
            return Err(ExperimentContractError::DevelopmentEqualsHeldOut);
        }

        validate_metric(&self.primary_metric)?;
        let mut metric_names = HashSet::new();
        metric_names.insert(self.primary_metric.name.as_str());
        for metric in &self.secondary_metrics {
            validate_metric(metric)?;
            if !metric_names.insert(metric.name.as_str()) {
                return Err(ExperimentContractError::DuplicateMetric {
                    name: metric.name.clone(),
                });
            }
        }

        if self.minimum_meaningful_effect.scale == 0 {
            return Err(ExperimentContractError::ZeroThresholdScale);
        }
        if self.confidence_bps == 0 || self.confidence_bps > MAX_CONFIDENCE_BPS {
            return Err(ExperimentContractError::ConfidenceOutOfRange {
                found: self.confidence_bps,
            });
        }
        reject_empty_list(&self.falsification_criteria, "falsification_criteria")?;
        reject_empty_list(&self.allowed_outcomes, "allowed_outcomes")?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExperimentContractBodyV1 {
    draft: ExperimentContractDraftV1,
    seed_plan: SeedPlanCommitmentV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedPlanCommitmentV1 {
    fingerprint: String,
    blake3_digest: String,
}

impl SeedPlanCommitmentV1 {
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn blake3_digest(&self) -> &str {
        &self.blake3_digest
    }

    pub fn from_seed_plan(seed_plan: &SeedPlan) -> Result<Self, ExperimentContractError> {
        validate_seed_plan(seed_plan)?;
        let bytes = serde_json::to_vec(seed_plan)
            .map_err(|error| ExperimentContractError::Serialization(error.to_string()))?;
        Ok(Self {
            fingerprint: seed_plan.fingerprint(),
            blake3_digest: blake3_digest(&bytes),
        })
    }

    fn validate_shape(&self) -> Result<(), ExperimentContractError> {
        require_nonempty(&self.fingerprint, "seed_plan.fingerprint")?;
        validate_blake3_digest(&self.blake3_digest)
    }
}

/// Immutable preregistration artifact. No mutation methods are exposed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegisteredExperimentContractV1 {
    body: ExperimentContractBodyV1,
    contract_digest: String,
}

impl RegisteredExperimentContractV1 {
    pub fn register(
        draft: ExperimentContractDraftV1,
        seed_plan: &SeedPlan,
    ) -> Result<Self, ExperimentContractError> {
        draft.validate()?;
        let seed_plan = SeedPlanCommitmentV1::from_seed_plan(seed_plan)?;
        let body = ExperimentContractBodyV1 { draft, seed_plan };
        let contract_digest = digest_body(&body)?;
        Ok(Self {
            body,
            contract_digest,
        })
    }

    pub fn draft(&self) -> &ExperimentContractDraftV1 {
        &self.body.draft
    }

    pub fn seed_plan_commitment(&self) -> &SeedPlanCommitmentV1 {
        &self.body.seed_plan
    }

    pub fn contract_digest(&self) -> &str {
        &self.contract_digest
    }

    /// Recompute the commitment from an actual seed plan at the run boundary.
    pub fn matches_seed_plan(&self, seed_plan: &SeedPlan) -> Result<bool, ExperimentContractError> {
        Ok(SeedPlanCommitmentV1::from_seed_plan(seed_plan)? == self.body.seed_plan)
    }

    pub fn verify_integrity(&self) -> Result<(), ExperimentContractError> {
        self.body.draft.validate()?;
        self.body.seed_plan.validate_shape()?;
        let expected = digest_body(&self.body)?;
        if expected != self.contract_digest {
            return Err(ExperimentContractError::ContractDigestMismatch {
                expected,
                found: self.contract_digest.clone(),
            });
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRegisteredExperimentContractV1 {
    body: ExperimentContractBodyV1,
    contract_digest: String,
}

impl<'de> Deserialize<'de> for RegisteredExperimentContractV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawRegisteredExperimentContractV1::deserialize(deserializer)?;
        let value = Self {
            body: raw.body,
            contract_digest: raw.contract_digest,
        };
        value.verify_integrity().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperimentContractError {
    UnsupportedSchemaVersion { found: u16 },
    EmptyField { field: &'static str },
    EmptyList { field: &'static str },
    MalformedDigest,
    MalformedBlake3Digest,
    BaselineEqualsCandidate,
    DevelopmentEqualsHeldOut,
    DuplicateMetric { name: String },
    ZeroThresholdScale,
    ConfidenceOutOfRange { found: u16 },
    InvalidSeedPlan,
    InvalidSeedPlanShape,
    Serialization(String),
    ContractDigestMismatch { expected: String, found: String },
}

impl std::fmt::Display for ExperimentContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported experiment contract schema version {found}; expected {EXPERIMENT_CONTRACT_SCHEMA_VERSION}"
            ),
            Self::EmptyField { field } => write!(f, "experiment contract field {field} must be non-empty"),
            Self::EmptyList { field } => write!(f, "experiment contract list {field} must be non-empty"),
            Self::MalformedDigest => f.write_str("identity digest must be sha256:<64 hex> or blake3:<64 hex>"),
            Self::MalformedBlake3Digest => f.write_str("contract digest must be blake3:<64 hex>"),
            Self::BaselineEqualsCandidate => f.write_str("baseline and candidate identities must differ"),
            Self::DevelopmentEqualsHeldOut => f.write_str("development and held-out corpus identities must differ"),
            Self::DuplicateMetric { name } => write!(f, "duplicate experiment metric {name}"),
            Self::ZeroThresholdScale => f.write_str("minimum meaningful effect scale must be non-zero"),
            Self::ConfidenceOutOfRange { found } => write!(f, "confidence requirement {found} bps is outside 1..={MAX_CONFIDENCE_BPS}"),
            Self::InvalidSeedPlan => f.write_str("seed plan violates registered seed discipline"),
            Self::InvalidSeedPlanShape => f.write_str("seed plan serialization did not contain canonical confirmatory/development arrays"),
            Self::Serialization(error) => write!(f, "experiment contract serialization failed: {error}"),
            Self::ContractDigestMismatch { expected, found } => write!(f, "experiment contract digest mismatch: expected {expected}, found {found}"),
        }
    }
}

impl std::error::Error for ExperimentContractError {}

fn validate_metric(metric: &MetricSpecV1) -> Result<(), ExperimentContractError> {
    require_nonempty(&metric.name, "metric.name")?;
    require_nonempty(&metric.unit, "metric.unit")
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), ExperimentContractError> {
    if value.trim().is_empty() {
        Err(ExperimentContractError::EmptyField { field })
    } else {
        Ok(())
    }
}

fn reject_empty_list(values: &[String], field: &'static str) -> Result<(), ExperimentContractError> {
    if values.is_empty() || values.iter().any(|value| value.trim().is_empty()) {
        Err(ExperimentContractError::EmptyList { field })
    } else {
        Ok(())
    }
}

fn validate_digest(digest: &str) -> Result<(), ExperimentContractError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(ExperimentContractError::MalformedDigest);
    };
    if !matches!(algorithm, "sha256" | "blake3")
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(ExperimentContractError::MalformedDigest);
    }
    Ok(())
}

fn validate_blake3_digest(digest: &str) -> Result<(), ExperimentContractError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(ExperimentContractError::MalformedBlake3Digest);
    };
    if algorithm != "blake3"
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(ExperimentContractError::MalformedBlake3Digest);
    }
    Ok(())
}

fn digest_body(body: &ExperimentContractBodyV1) -> Result<String, ExperimentContractError> {
    let bytes = serde_json::to_vec(body)
        .map_err(|error| ExperimentContractError::Serialization(error.to_string()))?;
    Ok(blake3_digest(&bytes))
}

fn blake3_digest(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

/// Revalidate even a deserialized `SeedPlan` by reconstructing it through the
/// evidence plane's only legitimate registration API.
fn validate_seed_plan(seed_plan: &SeedPlan) -> Result<(), ExperimentContractError> {
    let value = serde_json::to_value(seed_plan)
        .map_err(|error| ExperimentContractError::Serialization(error.to_string()))?;
    let object = value
        .as_object()
        .ok_or(ExperimentContractError::InvalidSeedPlanShape)?;
    let confirmatory = parse_seed_array(object.get("confirmatory"))?;
    let development = parse_seed_array(object.get("development"))?;
    SeedPlan::register(confirmatory, development)
        .map(|_| ())
        .map_err(|_| ExperimentContractError::InvalidSeedPlan)
}

fn parse_seed_array(value: Option<&serde_json::Value>) -> Result<Vec<u64>, ExperimentContractError> {
    value
        .and_then(serde_json::Value::as_array)
        .ok_or(ExperimentContractError::InvalidSeedPlanShape)?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .ok_or(ExperimentContractError::InvalidSeedPlanShape)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D: &str = "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const E: &str = "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const F: &str = "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn seeds() -> SeedPlan {
        SeedPlan::register((100..108).collect(), vec![1, 2, 3]).unwrap()
    }

    fn draft() -> ExperimentContractDraftV1 {
        ExperimentContractDraftV1 {
            schema_version: EXPERIMENT_CONTRACT_SCHEMA_VERSION,
            experiment_id: "rca-meta-controller-shadow-v1".into(),
            registered_at_unix_ms: 1_788_704_000_000,
            registered_by: "qualification-harness".into(),
            hypothesis_digest: A.into(),
            baseline_digest: B.into(),
            candidate_digest: C.into(),
            development_corpus_digest: D.into(),
            held_out_corpus_digest: E.into(),
            evaluator_digest: F.into(),
            primary_metric: MetricSpecV1 {
                name: "selective-risk".into(),
                unit: "fraction".into(),
                direction: MetricDirectionV1::LowerIsBetter,
            },
            secondary_metrics: vec![MetricSpecV1 {
                name: "compute-per-correct".into(),
                unit: "microunits".into(),
                direction: MetricDirectionV1::LowerIsBetter,
            }],
            minimum_meaningful_effect: FixedPointThresholdV1 {
                numerator: 100_000,
                scale: 1_000_000,
            },
            confidence_bps: 9_500,
            compute_ceiling_microunits: Some(3_000_000),
            wall_time_ceiling_ms: Some(60_000),
            falsification_criteria: vec![
                "candidate fails to beat compute-matched baseline by registered effect".into(),
            ],
            allowed_outcomes: vec![
                "candidate wins: advance to next shadow gate".into(),
                "no meaningful difference: prefer simpler baseline".into(),
                "candidate regresses: reject candidate".into(),
            ],
        }
    }

    #[test]
    fn registration_binds_seed_plan_cryptographically() {
        let plan = seeds();
        let contract = RegisteredExperimentContractV1::register(draft(), &plan).unwrap();
        assert!(contract.contract_digest().starts_with("blake3:"));
        assert!(contract.seed_plan_commitment().blake3_digest().starts_with("blake3:"));
        assert!(contract.matches_seed_plan(&plan).unwrap());

        let other = SeedPlan::register((200..208).collect(), vec![1, 2, 3]).unwrap();
        assert!(!contract.matches_seed_plan(&other).unwrap());
    }

    #[test]
    fn post_registration_metric_tampering_is_detected() {
        let contract = RegisteredExperimentContractV1::register(draft(), &seeds()).unwrap();
        let mut value = serde_json::to_value(&contract).unwrap();
        value["body"]["draft"]["primary_metric"]["name"] =
            serde_json::Value::String("convenient-post-hoc-metric".into());
        assert!(serde_json::from_value::<RegisteredExperimentContractV1>(value).is_err());
    }

    #[test]
    fn post_registration_threshold_tampering_is_detected() {
        let contract = RegisteredExperimentContractV1::register(draft(), &seeds()).unwrap();
        let mut value = serde_json::to_value(&contract).unwrap();
        value["body"]["draft"]["minimum_meaningful_effect"]["numerator"] =
            serde_json::json!(1);
        assert!(serde_json::from_value::<RegisteredExperimentContractV1>(value).is_err());
    }

    #[test]
    fn baseline_candidate_and_corpora_must_be_distinct() {
        let mut raw = draft();
        raw.candidate_digest = raw.baseline_digest.clone();
        assert_eq!(raw.validate(), Err(ExperimentContractError::BaselineEqualsCandidate));

        let mut raw = draft();
        raw.held_out_corpus_digest = raw.development_corpus_digest.clone();
        assert_eq!(raw.validate(), Err(ExperimentContractError::DevelopmentEqualsHeldOut));
    }

    #[test]
    fn falsification_and_allowed_outcomes_are_mandatory() {
        let mut raw = draft();
        raw.falsification_criteria.clear();
        assert_eq!(
            raw.validate(),
            Err(ExperimentContractError::EmptyList {
                field: "falsification_criteria"
            })
        );

        let mut raw = draft();
        raw.allowed_outcomes.clear();
        assert_eq!(
            raw.validate(),
            Err(ExperimentContractError::EmptyList {
                field: "allowed_outcomes"
            })
        );
    }

    #[test]
    fn duplicate_metrics_are_rejected() {
        let mut raw = draft();
        raw.secondary_metrics.push(raw.primary_metric.clone());
        assert!(matches!(
            raw.validate(),
            Err(ExperimentContractError::DuplicateMetric { .. })
        ));
    }

    #[test]
    fn registered_contract_revalidates_after_persistence() {
        let contract = RegisteredExperimentContractV1::register(draft(), &seeds()).unwrap();
        let json = serde_json::to_string(&contract).unwrap();
        let decoded: RegisteredExperimentContractV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, contract);
        decoded.verify_integrity().unwrap();
    }
}