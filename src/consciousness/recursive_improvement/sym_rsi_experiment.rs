// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fail-closed experiment kernel for SYM-RSI-001.
//!
//! This module encodes the preregistered experiment structure in machine-checkable
//! types. It does not execute recursive self-improvement. Its job is narrower:
//! freeze arm identity, seed partitions, domain identity, result receipts, and the
//! two primary contrasts before measured execution begins.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const SYM_RSI_001_MANIFEST_SCHEMA: &str = "symthaea.sym-rsi-001.manifest.v1";
pub const SYM_RSI_001_RECEIPT_SCHEMA: &str = "symthaea.sym-rsi-001.run-receipt.v1";
pub const SYM_RSI_001_CONTRAST_SCHEMA: &str = "symthaea.sym-rsi-001.primary-contrast.v1";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum ExperimentArm {
    AFixedExploration,
    BGroundedDreamFeedback,
    CExactReplayPolicyImprovement,
    DReplayPlusGroundedDreaming,
}

impl ExperimentArm {
    pub const ALL: [Self; 4] = [
        Self::AFixedExploration,
        Self::BGroundedDreamFeedback,
        Self::CExactReplayPolicyImprovement,
        Self::DReplayPlusGroundedDreaming,
    ];
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum EvaluationSplit {
    TrainingReplay,
    HeldOutReplay,
    FreshExecution,
    OutOfDistribution,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainSeedPlan {
    pub training_replay: Vec<u64>,
    pub held_out_replay: Vec<u64>,
    pub fresh_execution: Vec<u64>,
    pub out_of_distribution: Vec<u64>,
}

impl DomainSeedPlan {
    pub fn validate(&self, domain_id: &str) -> Result<(), ExperimentHarnessError> {
        let named = [
            (EvaluationSplit::TrainingReplay, &self.training_replay),
            (EvaluationSplit::HeldOutReplay, &self.held_out_replay),
            (EvaluationSplit::FreshExecution, &self.fresh_execution),
            (EvaluationSplit::OutOfDistribution, &self.out_of_distribution),
        ];

        let mut seen = BTreeSet::new();
        for (split, seeds) in named {
            if seeds.is_empty() {
                return Err(ExperimentHarnessError::EmptySeedSplit {
                    domain_id: domain_id.to_owned(),
                    split,
                });
            }
            for &seed in seeds {
                if !seen.insert(seed) {
                    return Err(ExperimentHarnessError::OverlappingSeed {
                        domain_id: domain_id.to_owned(),
                        seed,
                    });
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExperimentDomainSpec {
    pub domain_id: String,
    pub adapter_version: String,
    pub max_evaluator_calls: u64,
    pub seeds: DomainSeedPlan,
}

impl ExperimentDomainSpec {
    pub fn validate(&self) -> Result<(), ExperimentHarnessError> {
        if self.domain_id.trim().is_empty() || self.adapter_version.trim().is_empty() {
            return Err(ExperimentHarnessError::MissingRequiredIdentity(
                self.domain_id.clone(),
            ));
        }
        if self.max_evaluator_calls == 0 {
            return Err(ExperimentHarnessError::ZeroEvaluationBudget(
                self.domain_id.clone(),
            ));
        }
        self.seeds.validate(&self.domain_id)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymRsiExperimentManifest {
    pub schema: String,
    pub experiment_id: String,
    pub preregistration_digest: String,
    pub subject_digest: String,
    pub environment_digest: String,
    pub arms: Vec<ExperimentArm>,
    pub domains: Vec<ExperimentDomainSpec>,
    pub beta_cost: f64,
    pub beta_parallelism: f64,
    pub held_out_quality_tolerance: f64,
}

impl SymRsiExperimentManifest {
    pub fn validate(&self) -> Result<(), ExperimentHarnessError> {
        if self.schema != SYM_RSI_001_MANIFEST_SCHEMA {
            return Err(ExperimentHarnessError::WrongSchema(self.schema.clone()));
        }
        for identity in [
            &self.experiment_id,
            &self.preregistration_digest,
            &self.subject_digest,
            &self.environment_digest,
        ] {
            if identity.trim().is_empty() {
                return Err(ExperimentHarnessError::MissingRequiredIdentity(
                    "manifest".into(),
                ));
            }
        }

        let observed_arms: BTreeSet<_> = self.arms.iter().copied().collect();
        let expected_arms: BTreeSet<_> = ExperimentArm::ALL.into_iter().collect();
        if observed_arms != expected_arms || self.arms.len() != ExperimentArm::ALL.len() {
            return Err(ExperimentHarnessError::ArmSetMismatch);
        }

        if self.domains.len() < 3 {
            return Err(ExperimentHarnessError::InsufficientDomainCount(
                self.domains.len(),
            ));
        }
        let mut ids = BTreeSet::new();
        for domain in &self.domains {
            domain.validate()?;
            if !ids.insert(domain.domain_id.clone()) {
                return Err(ExperimentHarnessError::DuplicateDomain(
                    domain.domain_id.clone(),
                ));
            }
        }

        for (name, value) in [
            ("beta_cost", self.beta_cost),
            ("beta_parallelism", self.beta_parallelism),
            ("held_out_quality_tolerance", self.held_out_quality_tolerance),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ExperimentHarnessError::InvalidManifestScalar(
                    name.into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArmRunMetrics {
    pub best_solution_quality: f64,
    pub evaluator_calls: u64,
    pub normalized_compute_cost: f64,
    pub brier_score: Option<f64>,
    pub regression_rate: f64,
    pub policy_churn_rate: f64,
    pub replay_pool_coverage: f64,
    pub unsupported_action_rate: f64,
    pub safety_constraint_violations: u64,
    pub authority_boundary_violations: u64,
}

impl ArmRunMetrics {
    pub fn validate(&self) -> Result<(), ExperimentHarnessError> {
        if !self.best_solution_quality.is_finite()
            || !self.normalized_compute_cost.is_finite()
            || self.normalized_compute_cost < 0.0
            || self.evaluator_calls == 0
        {
            return Err(ExperimentHarnessError::InvalidPrimaryMetric);
        }
        if let Some(brier) = self.brier_score
            && (!brier.is_finite() || !(0.0..=1.0).contains(&brier))
        {
            return Err(ExperimentHarnessError::InvalidRate("brier_score".into()));
        }
        for (name, rate) in [
            ("regression_rate", self.regression_rate),
            ("policy_churn_rate", self.policy_churn_rate),
            ("replay_pool_coverage", self.replay_pool_coverage),
            ("unsupported_action_rate", self.unsupported_action_rate),
        ] {
            if !rate.is_finite() || !(0.0..=1.0).contains(&rate) {
                return Err(ExperimentHarnessError::InvalidRate(name.into()));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymRsiRunReceipt {
    pub schema: String,
    pub experiment_id: String,
    pub preregistration_digest: String,
    pub subject_digest: String,
    pub environment_digest: String,
    pub domain_id: String,
    pub adapter_version: String,
    pub arm: ExperimentArm,
    pub split: EvaluationSplit,
    pub seed: u64,
    pub policy_id: String,
    pub metrics: ArmRunMetrics,
    pub evidence_digest: String,
    /// Must remain false. Generated/model-derived content may suggest actions or
    /// hypotheses but is not allowed to silently acquire empirical authority.
    pub generated_evidence_promoted: bool,
}

impl SymRsiRunReceipt {
    pub fn validate(&self) -> Result<(), ExperimentHarnessError> {
        if self.schema != SYM_RSI_001_RECEIPT_SCHEMA {
            return Err(ExperimentHarnessError::WrongSchema(self.schema.clone()));
        }
        for identity in [
            &self.experiment_id,
            &self.preregistration_digest,
            &self.subject_digest,
            &self.environment_digest,
            &self.domain_id,
            &self.adapter_version,
            &self.policy_id,
            &self.evidence_digest,
        ] {
            if identity.trim().is_empty() {
                return Err(ExperimentHarnessError::MissingRequiredIdentity(
                    "run-receipt".into(),
                ));
            }
        }
        if self.generated_evidence_promoted {
            return Err(ExperimentHarnessError::GeneratedEvidencePromotion);
        }
        self.metrics.validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrimaryContrastKind {
    ReplayVsFixed,
    ReplayDreamVsReplay,
}

impl PrimaryContrastKind {
    pub fn arms(self) -> (ExperimentArm, ExperimentArm) {
        match self {
            Self::ReplayVsFixed => (
                ExperimentArm::AFixedExploration,
                ExperimentArm::CExactReplayPolicyImprovement,
            ),
            Self::ReplayDreamVsReplay => (
                ExperimentArm::CExactReplayPolicyImprovement,
                ExperimentArm::DReplayPlusGroundedDreaming,
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrimaryContrastReceipt {
    pub schema: String,
    pub kind: PrimaryContrastKind,
    pub experiment_id: String,
    pub subject_digest: String,
    pub environment_digest: String,
    pub domain_id: String,
    pub split: EvaluationSplit,
    pub seed: u64,
    /// Treatment minus baseline. Positive means higher measured quality.
    pub quality_delta: f64,
    /// Treatment minus baseline. Negative means fewer evaluator calls.
    pub evaluator_call_delta: i128,
    /// Treatment minus baseline. Negative means lower normalized cost.
    pub compute_cost_delta: f64,
    pub safety_violation_delta: i128,
    pub authority_violation_delta: i128,
}

pub fn build_primary_contrast(
    kind: PrimaryContrastKind,
    baseline: &SymRsiRunReceipt,
    treatment: &SymRsiRunReceipt,
) -> Result<PrimaryContrastReceipt, ExperimentHarnessError> {
    baseline.validate()?;
    treatment.validate()?;

    let (expected_baseline, expected_treatment) = kind.arms();
    if baseline.arm != expected_baseline || treatment.arm != expected_treatment {
        return Err(ExperimentHarnessError::WrongContrastArms);
    }

    if baseline.experiment_id != treatment.experiment_id
        || baseline.preregistration_digest != treatment.preregistration_digest
        || baseline.subject_digest != treatment.subject_digest
        || baseline.environment_digest != treatment.environment_digest
        || baseline.domain_id != treatment.domain_id
        || baseline.adapter_version != treatment.adapter_version
        || baseline.split != treatment.split
        || baseline.seed != treatment.seed
    {
        return Err(ExperimentHarnessError::MismatchedContrastSubjects);
    }

    Ok(PrimaryContrastReceipt {
        schema: SYM_RSI_001_CONTRAST_SCHEMA.into(),
        kind,
        experiment_id: baseline.experiment_id.clone(),
        subject_digest: baseline.subject_digest.clone(),
        environment_digest: baseline.environment_digest.clone(),
        domain_id: baseline.domain_id.clone(),
        split: baseline.split,
        seed: baseline.seed,
        quality_delta: treatment.metrics.best_solution_quality
            - baseline.metrics.best_solution_quality,
        evaluator_call_delta: treatment.metrics.evaluator_calls as i128
            - baseline.metrics.evaluator_calls as i128,
        compute_cost_delta: treatment.metrics.normalized_compute_cost
            - baseline.metrics.normalized_compute_cost,
        safety_violation_delta: treatment.metrics.safety_constraint_violations as i128
            - baseline.metrics.safety_constraint_violations as i128,
        authority_violation_delta: treatment.metrics.authority_boundary_violations as i128
            - baseline.metrics.authority_boundary_violations as i128,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperimentHarnessError {
    WrongSchema(String),
    MissingRequiredIdentity(String),
    ArmSetMismatch,
    InsufficientDomainCount(usize),
    DuplicateDomain(String),
    ZeroEvaluationBudget(String),
    EmptySeedSplit {
        domain_id: String,
        split: EvaluationSplit,
    },
    OverlappingSeed {
        domain_id: String,
        seed: u64,
    },
    InvalidManifestScalar(String),
    InvalidPrimaryMetric,
    InvalidRate(String),
    GeneratedEvidencePromotion,
    WrongContrastArms,
    MismatchedContrastSubjects,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed_plan(base: u64) -> DomainSeedPlan {
        DomainSeedPlan {
            training_replay: vec![base, base + 1],
            held_out_replay: vec![base + 10],
            fresh_execution: vec![base + 20],
            out_of_distribution: vec![base + 30],
        }
    }

    fn domain(id: &str, base: u64) -> ExperimentDomainSpec {
        ExperimentDomainSpec {
            domain_id: id.into(),
            adapter_version: "v1".into(),
            max_evaluator_calls: 64,
            seeds: seed_plan(base),
        }
    }

    fn manifest() -> SymRsiExperimentManifest {
        SymRsiExperimentManifest {
            schema: SYM_RSI_001_MANIFEST_SCHEMA.into(),
            experiment_id: "SYM-RSI-001".into(),
            preregistration_digest: "pre-digest".into(),
            subject_digest: "subject-digest".into(),
            environment_digest: "env-digest".into(),
            arms: ExperimentArm::ALL.to_vec(),
            domains: vec![
                domain("branching-search", 1),
                domain("delayed-planning", 101),
                domain("symbolic-sequence", 201),
            ],
            beta_cost: 0.05,
            beta_parallelism: 0.01,
            held_out_quality_tolerance: 0.02,
        }
    }

    fn receipt(arm: ExperimentArm, seed: u64) -> SymRsiRunReceipt {
        SymRsiRunReceipt {
            schema: SYM_RSI_001_RECEIPT_SCHEMA.into(),
            experiment_id: "SYM-RSI-001".into(),
            preregistration_digest: "pre-digest".into(),
            subject_digest: "subject-digest".into(),
            environment_digest: "env-digest".into(),
            domain_id: "branching-search".into(),
            adapter_version: "v1".into(),
            arm,
            split: EvaluationSplit::HeldOutReplay,
            seed,
            policy_id: format!("policy-{arm:?}"),
            metrics: ArmRunMetrics {
                best_solution_quality: 0.7,
                evaluator_calls: 32,
                normalized_compute_cost: 10.0,
                brier_score: Some(0.2),
                regression_rate: 0.0,
                policy_churn_rate: 0.1,
                replay_pool_coverage: 0.8,
                unsupported_action_rate: 0.05,
                safety_constraint_violations: 0,
                authority_boundary_violations: 0,
            },
            evidence_digest: "evidence-digest".into(),
            generated_evidence_promoted: false,
        }
    }

    #[test]
    fn manifest_requires_all_four_arms_and_three_domains() {
        let mut m = manifest();
        assert_eq!(m.validate(), Ok(()));

        m.arms.pop();
        assert_eq!(m.validate(), Err(ExperimentHarnessError::ArmSetMismatch));
    }

    #[test]
    fn seed_splits_are_pairwise_disjoint() {
        let mut m = manifest();
        m.domains[0].seeds.held_out_replay = vec![1];
        assert_eq!(
            m.validate(),
            Err(ExperimentHarnessError::OverlappingSeed {
                domain_id: "branching-search".into(),
                seed: 1,
            })
        );
    }

    #[test]
    fn generated_evidence_cannot_self_promote() {
        let mut r = receipt(ExperimentArm::BGroundedDreamFeedback, 11);
        r.generated_evidence_promoted = true;
        assert_eq!(
            r.validate(),
            Err(ExperimentHarnessError::GeneratedEvidencePromotion)
        );
    }

    #[test]
    fn primary_contrast_requires_exact_subject_pairing() {
        let baseline = receipt(ExperimentArm::AFixedExploration, 11);
        let mut treatment = receipt(ExperimentArm::CExactReplayPolicyImprovement, 12);
        assert_eq!(
            build_primary_contrast(PrimaryContrastKind::ReplayVsFixed, &baseline, &treatment),
            Err(ExperimentHarnessError::MismatchedContrastSubjects)
        );

        treatment.seed = 11;
        treatment.metrics.best_solution_quality = 0.8;
        treatment.metrics.evaluator_calls = 24;
        let contrast = build_primary_contrast(
            PrimaryContrastKind::ReplayVsFixed,
            &baseline,
            &treatment,
        )
        .unwrap();
        assert!((contrast.quality_delta - 0.1).abs() < 1e-12);
        assert_eq!(contrast.evaluator_call_delta, -8);
    }
}
