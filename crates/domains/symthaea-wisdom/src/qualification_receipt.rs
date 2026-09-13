// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fail-closed qualification receipts for the frozen WCARE evaluation contract.
//!
//! A qualification claim must bind the exact subject, contract, corpus manifest,
//! environment, and per-scenario evidence. Missing, excluded, infrastructure-
//! failed, or unscored required evidence is indeterminate rather than passing.

use std::collections::{BTreeMap, BTreeSet};

use crate::evaluation_contract::{
    EvidenceTier, GateClass, ScenarioSpec, WCARE_V1_SCENARIOS,
};

pub const WCARE_V1_CONTRACT_ID: &str = "symthaea.wcare.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationTarget {
    MechanismQualified,
    AdversariallyQualified,
    LongitudinallyQualified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioOutcome {
    Pass,
    Fail,
    InfrastructureError,
    Excluded,
    NotScored,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScenarioResult {
    pub scenario_id: String,
    pub outcome: ScenarioOutcome,
    /// Immutable evidence reference: receipt digest, artifact path+digest, or
    /// infrastructure-failure receipt. Empty evidence references are rejected.
    pub evidence_ref: String,
}

impl ScenarioResult {
    pub fn new(
        scenario_id: impl Into<String>,
        outcome: ScenarioOutcome,
        evidence_ref: impl Into<String>,
    ) -> Result<Self, QualificationError> {
        let scenario_id = scenario_id.into();
        let evidence_ref = evidence_ref.into();
        if scenario_id.trim().is_empty() {
            return Err(QualificationError::EmptyScenarioId);
        }
        if evidence_ref.trim().is_empty() {
            return Err(QualificationError::EmptyEvidenceRef(scenario_id));
        }
        if scenario_spec(&scenario_id).is_none() {
            return Err(QualificationError::UnknownScenario(scenario_id));
        }
        Ok(Self {
            scenario_id,
            outcome,
            evidence_ref,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualificationReason {
    MissingRequiredScenario(String),
    HardFailViolation(String),
    ComparativeFailure(String),
    InfrastructureIndeterminate(String),
    ExcludedRequiredScenario(String),
    UnscoredRequiredScenario(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationStatus {
    Qualified,
    Blocked,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualificationAssessment {
    pub status: QualificationStatus,
    pub reasons: BTreeSet<QualificationReason>,
    pub evaluated_scenarios: usize,
    pub required_scenarios: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualificationReceipt {
    pub subject_ref: String,
    pub contract_id: String,
    pub corpus_manifest_sha256: String,
    pub environment_ref: String,
    results: BTreeMap<String, ScenarioResult>,
}

impl QualificationReceipt {
    pub fn try_new(
        subject_ref: impl Into<String>,
        contract_id: impl Into<String>,
        corpus_manifest_sha256: impl Into<String>,
        environment_ref: impl Into<String>,
        results: impl IntoIterator<Item = ScenarioResult>,
    ) -> Result<Self, QualificationError> {
        let subject_ref = nonempty(subject_ref.into(), QualificationError::EmptySubjectRef)?;
        let contract_id = nonempty(contract_id.into(), QualificationError::EmptyContractId)?;
        let corpus_manifest_sha256 = corpus_manifest_sha256.into();
        validate_sha256(&corpus_manifest_sha256)?;
        let environment_ref = nonempty(
            environment_ref.into(),
            QualificationError::EmptyEnvironmentRef,
        )?;

        if contract_id != WCARE_V1_CONTRACT_ID {
            return Err(QualificationError::UnsupportedContract(contract_id));
        }

        let mut indexed = BTreeMap::new();
        for result in results {
            if scenario_spec(&result.scenario_id).is_none() {
                return Err(QualificationError::UnknownScenario(result.scenario_id));
            }
            let id = result.scenario_id.clone();
            if indexed.insert(id.clone(), result).is_some() {
                return Err(QualificationError::DuplicateScenario(id));
            }
        }

        Ok(Self {
            subject_ref,
            contract_id,
            corpus_manifest_sha256,
            environment_ref,
            results: indexed,
        })
    }

    pub fn results(&self) -> &BTreeMap<String, ScenarioResult> {
        &self.results
    }

    pub fn assess(&self, target: QualificationTarget) -> QualificationAssessment {
        let required: Vec<_> = WCARE_V1_SCENARIOS
            .iter()
            .filter(|scenario| tier_required(scenario.tier, target))
            .collect();
        let mut reasons = BTreeSet::new();

        for scenario in &required {
            let Some(result) = self.results.get(scenario.id) else {
                reasons.insert(QualificationReason::MissingRequiredScenario(
                    scenario.id.to_string(),
                ));
                continue;
            };

            match result.outcome {
                ScenarioOutcome::Pass => {}
                ScenarioOutcome::Fail => match scenario.gate {
                    GateClass::HardFail => {
                        reasons.insert(QualificationReason::HardFailViolation(
                            scenario.id.to_string(),
                        ));
                    }
                    GateClass::Comparative => {
                        reasons.insert(QualificationReason::ComparativeFailure(
                            scenario.id.to_string(),
                        ));
                    }
                    GateClass::Diagnostic => {}
                },
                ScenarioOutcome::InfrastructureError => {
                    reasons.insert(QualificationReason::InfrastructureIndeterminate(
                        scenario.id.to_string(),
                    ));
                }
                ScenarioOutcome::Excluded => {
                    reasons.insert(QualificationReason::ExcludedRequiredScenario(
                        scenario.id.to_string(),
                    ));
                }
                ScenarioOutcome::NotScored => {
                    if scenario.gate != GateClass::Diagnostic {
                        reasons.insert(QualificationReason::UnscoredRequiredScenario(
                            scenario.id.to_string(),
                        ));
                    }
                }
            }
        }

        let blocked = reasons.iter().any(|reason| {
            matches!(
                reason,
                QualificationReason::HardFailViolation(_)
                    | QualificationReason::ComparativeFailure(_)
            )
        });
        let indeterminate = reasons.iter().any(|reason| {
            matches!(
                reason,
                QualificationReason::MissingRequiredScenario(_)
                    | QualificationReason::InfrastructureIndeterminate(_)
                    | QualificationReason::ExcludedRequiredScenario(_)
                    | QualificationReason::UnscoredRequiredScenario(_)
            )
        });

        let status = if blocked {
            QualificationStatus::Blocked
        } else if indeterminate {
            QualificationStatus::Indeterminate
        } else {
            QualificationStatus::Qualified
        };

        QualificationAssessment {
            status,
            reasons,
            evaluated_scenarios: required
                .iter()
                .filter(|scenario| self.results.contains_key(scenario.id))
                .count(),
            required_scenarios: required.len(),
        }
    }
}

fn scenario_spec(id: &str) -> Option<&'static ScenarioSpec> {
    WCARE_V1_SCENARIOS.iter().find(|scenario| scenario.id == id)
}

fn tier_required(tier: EvidenceTier, target: QualificationTarget) -> bool {
    match target {
        QualificationTarget::MechanismQualified => tier == EvidenceTier::Mechanism,
        QualificationTarget::AdversariallyQualified => {
            matches!(tier, EvidenceTier::Mechanism | EvidenceTier::Adversarial)
        }
        QualificationTarget::LongitudinallyQualified => matches!(
            tier,
            EvidenceTier::Mechanism | EvidenceTier::Adversarial | EvidenceTier::Longitudinal
        ),
    }
}

fn nonempty(value: String, error: QualificationError) -> Result<String, QualificationError> {
    if value.trim().is_empty() {
        Err(error)
    } else {
        Ok(value)
    }
}

fn validate_sha256(value: &str) -> Result<(), QualificationError> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        Ok(())
    } else {
        Err(QualificationError::InvalidCorpusDigest(value.to_string()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationError {
    EmptyScenarioId,
    EmptyEvidenceRef(String),
    UnknownScenario(String),
    DuplicateScenario(String),
    EmptySubjectRef,
    EmptyContractId,
    UnsupportedContract(String),
    InvalidCorpusDigest(String),
    EmptyEnvironmentRef,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest() -> String {
        "a".repeat(64)
    }

    fn result(spec: &ScenarioSpec, outcome: ScenarioOutcome) -> ScenarioResult {
        ScenarioResult::new(spec.id, outcome, format!("receipt:{}", spec.id)).unwrap()
    }

    fn all_results() -> Vec<ScenarioResult> {
        WCARE_V1_SCENARIOS
            .iter()
            .map(|spec| result(spec, ScenarioOutcome::Pass))
            .collect()
    }

    fn receipt(results: Vec<ScenarioResult>) -> QualificationReceipt {
        QualificationReceipt::try_new(
            "commit:deadbeef",
            WCARE_V1_CONTRACT_ID,
            digest(),
            "nix:env-root",
            results,
        )
        .unwrap()
    }

    #[test]
    fn all_required_results_pass_qualifies_longitudinal_target() {
        let assessment = receipt(all_results()).assess(QualificationTarget::LongitudinallyQualified);
        assert_eq!(assessment.status, QualificationStatus::Qualified);
        assert!(assessment.reasons.is_empty());
    }

    #[test]
    fn missing_required_case_is_indeterminate_not_pass() {
        let mut results = all_results();
        results.retain(|result| result.scenario_id != "WCARE-V1-A04");
        let assessment = receipt(results).assess(QualificationTarget::AdversariallyQualified);
        assert_eq!(assessment.status, QualificationStatus::Indeterminate);
        assert!(assessment.reasons.contains(
            &QualificationReason::MissingRequiredScenario("WCARE-V1-A04".into())
        ));
    }

    #[test]
    fn hard_fail_violation_blocks_even_when_everything_else_passes() {
        let mut results = all_results();
        let target = results
            .iter_mut()
            .find(|result| result.scenario_id == "WCARE-V1-A04")
            .unwrap();
        target.outcome = ScenarioOutcome::Fail;
        let assessment = receipt(results).assess(QualificationTarget::AdversariallyQualified);
        assert_eq!(assessment.status, QualificationStatus::Blocked);
        assert!(assessment.reasons.contains(
            &QualificationReason::HardFailViolation("WCARE-V1-A04".into())
        ));
    }

    #[test]
    fn infrastructure_failure_is_indeterminate_not_failure_or_pass() {
        let mut results = all_results();
        let target = results
            .iter_mut()
            .find(|result| result.scenario_id == "WCARE-V1-A08")
            .unwrap();
        target.outcome = ScenarioOutcome::InfrastructureError;
        let assessment = receipt(results).assess(QualificationTarget::AdversariallyQualified);
        assert_eq!(assessment.status, QualificationStatus::Indeterminate);
    }

    #[test]
    fn excluded_required_case_is_indeterminate() {
        let mut results = all_results();
        let target = results
            .iter_mut()
            .find(|result| result.scenario_id == "WCARE-V1-L02")
            .unwrap();
        target.outcome = ScenarioOutcome::Excluded;
        let assessment = receipt(results).assess(QualificationTarget::LongitudinallyQualified);
        assert_eq!(assessment.status, QualificationStatus::Indeterminate);
    }

    #[test]
    fn comparative_case_must_be_scored_before_promotion() {
        let mut results = all_results();
        let target = results
            .iter_mut()
            .find(|result| result.scenario_id == "WCARE-V1-A13")
            .unwrap();
        target.outcome = ScenarioOutcome::NotScored;
        let assessment = receipt(results).assess(QualificationTarget::AdversariallyQualified);
        assert_eq!(assessment.status, QualificationStatus::Indeterminate);
    }

    #[test]
    fn duplicate_scenario_is_rejected_at_receipt_construction() {
        let spec = WCARE_V1_SCENARIOS[0];
        let duplicate = vec![result(&spec, ScenarioOutcome::Pass), result(&spec, ScenarioOutcome::Pass)];
        assert!(matches!(
            QualificationReceipt::try_new(
                "commit:x",
                WCARE_V1_CONTRACT_ID,
                digest(),
                "env:x",
                duplicate,
            ),
            Err(QualificationError::DuplicateScenario(_))
        ));
    }

    #[test]
    fn invalid_digest_is_rejected() {
        assert!(matches!(
            QualificationReceipt::try_new(
                "commit:x",
                WCARE_V1_CONTRACT_ID,
                "not-a-digest",
                "env:x",
                all_results(),
            ),
            Err(QualificationError::InvalidCorpusDigest(_))
        ));
    }
}
