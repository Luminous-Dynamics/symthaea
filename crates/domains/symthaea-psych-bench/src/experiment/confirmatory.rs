// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fail-closed helpers for confirmatory architecture experiments.
//!
//! These wrappers make two scientific assumptions explicit in the type/API
//! boundary: generated environments are the independent unit for confirmatory
//! paired inference, and practical effect claims must be interpreted relative
//! to a predeclared smallest effect size of interest (SESOI).

use crate::experiment::{ClaimLedgerEntry, PairedEstimate, TaskProgram, paired_delta_bca};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const CLAIM_EVIDENCE_BINDING_SCHEMA_V1: &str = "symthaea.claim-evidence-binding/v1";

fn looks_like_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

/// One already-aggregated candidate/control observation for one independently
/// generated environment.
///
/// Multiple representation, learner, or stream runs belonging to the same
/// environment must be aggregated before constructing this value. This prevents
/// nested RNG runs from being silently flattened into pseudoreplicates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentPair {
    pub environment_digest: String,
    pub candidate: f64,
    pub control: f64,
}

/// Compute candidate-minus-control uncertainty with environments as the explicit
/// independent unit.
pub fn paired_environment_delta_bca(
    pairs: &[EnvironmentPair],
    n_resamples: usize,
    seed: u64,
) -> Result<PairedEstimate, String> {
    if pairs.len() < 3 {
        return Err(
            "confirmatory paired inference requires at least three independent environments".into(),
        );
    }

    let mut seen = BTreeSet::new();
    let mut candidate = Vec::with_capacity(pairs.len());
    let mut control = Vec::with_capacity(pairs.len());

    for pair in pairs {
        if !looks_like_digest(&pair.environment_digest) {
            return Err("environment digest must be a 32-byte hex digest".into());
        }
        if !seen.insert(pair.environment_digest.to_ascii_lowercase()) {
            return Err("duplicate environment digest would create pseudoreplication".into());
        }
        if !pair.candidate.is_finite() || !pair.control.is_finite() {
            return Err("environment-paired observations must be finite".into());
        }
        candidate.push(pair.candidate);
        control.push(pair.control);
    }

    paired_delta_bca(&candidate, &control, n_resamples, seed)
}

/// Predeclared practical interpretation of a paired confidence interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PracticalEffect {
    /// The complete confidence interval exceeds the positive SESOI.
    MeaningfulGain,
    /// The complete confidence interval is below the negative SESOI.
    MeaningfulRegression,
    /// The complete confidence interval lies inside the equivalence region.
    Equivalent,
    /// The interval overlaps both practically important and negligible regions.
    Inconclusive,
}

/// Classify a candidate-minus-control estimate against a frozen SESOI.
///
/// This is deliberately interval-based. A favorable point estimate alone cannot
/// produce `MeaningfulGain`, and an underpowered result cannot produce
/// `Equivalent` unless the entire interval fits inside the equivalence region.
pub fn classify_practical_effect(
    estimate: &PairedEstimate,
    sesoi: f64,
) -> Result<PracticalEffect, String> {
    if !sesoi.is_finite() || sesoi <= 0.0 {
        return Err(
            "SESOI must be finite and strictly positive for practical-effect classification".into(),
        );
    }
    if !estimate.mean_delta.is_finite()
        || !estimate.ci95_low.is_finite()
        || !estimate.ci95_high.is_finite()
        || estimate.ci95_low > estimate.ci95_high
    {
        return Err("paired estimate must contain a finite ordered confidence interval".into());
    }

    if estimate.ci95_low > sesoi {
        Ok(PracticalEffect::MeaningfulGain)
    } else if estimate.ci95_high < -sesoi {
        Ok(PracticalEffect::MeaningfulRegression)
    } else if estimate.ci95_low >= -sesoi && estimate.ci95_high <= sesoi {
        Ok(PracticalEffect::Equivalent)
    } else {
        Ok(PracticalEffect::Inconclusive)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportOverlapPolicy {
    /// Training and evaluation support descriptors must be disjoint.
    Disjoint,
    /// Exact support overlap is intentional and therefore explicitly declared.
    AllowDeclaredOverlap,
}

/// Validate support descriptors beyond the base `TaskProgram` structural checks.
///
/// Empty or duplicated descriptors are always rejected. Exact train/evaluation
/// overlap is rejected unless the experiment explicitly declares that overlap is
/// part of its design (for example an IID retention evaluation).
pub fn validate_task_support(
    program: &TaskProgram,
    policy: SupportOverlapPolicy,
) -> Result<(), String> {
    program.validate()?;

    fn normalized_set(values: &[String], label: &str) -> Result<BTreeSet<String>, String> {
        let mut set = BTreeSet::new();
        for value in values {
            let normalized = value.trim().to_string();
            if normalized.is_empty() {
                return Err(format!("{label} support contains an empty descriptor"));
            }
            if !set.insert(normalized) {
                return Err(format!("{label} support contains a duplicate descriptor"));
            }
        }
        Ok(set)
    }

    let train = normalized_set(&program.train_support, "training")?;
    let eval = normalized_set(&program.eval_support, "evaluation")?;

    if policy == SupportOverlapPolicy::Disjoint && train.iter().any(|item| eval.contains(item)) {
        return Err("training/evaluation support overlaps under a disjoint-support policy".into());
    }

    Ok(())
}

/// Cryptographic provenance binding for one claim-ledger entry.
///
/// A claim is not considered provenance-complete merely because it contains a
/// `ProvenanceStatus::Valid` flag. This binding ties the claim identifier to the
/// exact experiment manifest, task programs, result artifact, and analysis code
/// revision used to produce the claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimEvidenceBinding {
    pub schema: String,
    pub claim_id: String,
    pub manifest_digest: String,
    pub task_program_digests: Vec<String>,
    pub result_digest: String,
    pub analysis_code_revision: String,
}

impl ClaimEvidenceBinding {
    pub fn validate_for_claim(&self, claim: &ClaimLedgerEntry) -> Result<(), String> {
        claim.validate()?;
        if self.schema != CLAIM_EVIDENCE_BINDING_SCHEMA_V1 {
            return Err(format!(
                "unsupported claim-evidence binding schema: {}",
                self.schema
            ));
        }
        if self.claim_id != claim.claim_id {
            return Err("claim-evidence binding claim id does not match the ledger entry".into());
        }
        if !looks_like_digest(&self.manifest_digest) || !looks_like_digest(&self.result_digest) {
            return Err("manifest/result digests must be 32-byte hex digests".into());
        }
        if self.task_program_digests.is_empty() {
            return Err("at least one task-program digest must be bound to a claim".into());
        }
        let mut tasks = BTreeSet::new();
        for digest in &self.task_program_digests {
            if !looks_like_digest(digest) {
                return Err("task-program digests must be 32-byte hex digests".into());
            }
            if !tasks.insert(digest.to_ascii_lowercase()) {
                return Err("duplicate task-program digest in claim evidence binding".into());
            }
        }
        if self.analysis_code_revision.trim().is_empty() {
            return Err("analysis code revision must be recorded".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::{
        CLAIM_LEDGER_SCHEMA_V1, ContextVisibility, EvidenceOutcome, GeneralizationLevel,
        ProvenanceStatus, ReplicationState, ResourceStatus, RuleExpr, SupportKind,
        TASK_PROGRAM_SCHEMA_V1, TimingRegime,
    };

    fn digest(n: u64) -> String {
        format!("{n:064x}")
    }

    fn task(train: &[&str], eval: &[&str]) -> TaskProgram {
        TaskProgram {
            schema: TASK_PROGRAM_SCHEMA_V1.into(),
            program_id: "support-test".into(),
            family: "relational".into(),
            rule: RuleExpr::Ne {
                left: "x".into(),
                right: "y".into(),
            },
            context_visibility: ContextVisibility::TaskFree,
            timing_regime: TimingRegime::Uniform,
            positive_examples: 8,
            negative_examples: 8,
            train_support: train.iter().map(|v| (*v).to_string()).collect(),
            eval_support: eval.iter().map(|v| (*v).to_string()).collect(),
            oracle_digest: digest(99),
        }
    }

    fn claim() -> ClaimLedgerEntry {
        ClaimLedgerEntry {
            schema: CLAIM_LEDGER_SCHEMA_V1.into(),
            claim_id: "retention-advantage".into(),
            statement: "candidate reduces forgetting".into(),
            outcome: EvidenceOutcome::Supported,
            support_kind: SupportKind::Observed,
            generalization: GeneralizationLevel::IidConfirm,
            replication: ReplicationState::Unreplicated,
            resources: ResourceStatus::NotAssessed,
            provenance: ProvenanceStatus::Valid,
            qualifiers: Vec::new(),
        }
    }

    fn binding() -> ClaimEvidenceBinding {
        ClaimEvidenceBinding {
            schema: CLAIM_EVIDENCE_BINDING_SCHEMA_V1.into(),
            claim_id: "retention-advantage".into(),
            manifest_digest: digest(10),
            task_program_digests: vec![digest(11), digest(12)],
            result_digest: digest(13),
            analysis_code_revision: "deadbeef".into(),
        }
    }

    #[test]
    fn environment_pairing_rejects_pseudoreplicated_environment_ids() {
        let pairs = vec![
            EnvironmentPair {
                environment_digest: digest(1),
                candidate: 0.8,
                control: 0.7,
            },
            EnvironmentPair {
                environment_digest: digest(1),
                candidate: 0.82,
                control: 0.71,
            },
            EnvironmentPair {
                environment_digest: digest(2),
                candidate: 0.79,
                control: 0.70,
            },
        ];
        assert!(paired_environment_delta_bca(&pairs, 500, 42).is_err());
    }

    #[test]
    fn environment_pairing_uses_one_pair_per_environment() {
        let pairs = vec![
            EnvironmentPair {
                environment_digest: digest(1),
                candidate: 0.80,
                control: 0.70,
            },
            EnvironmentPair {
                environment_digest: digest(2),
                candidate: 0.82,
                control: 0.72,
            },
            EnvironmentPair {
                environment_digest: digest(3),
                candidate: 0.78,
                control: 0.68,
            },
            EnvironmentPair {
                environment_digest: digest(4),
                candidate: 0.81,
                control: 0.71,
            },
        ];
        let estimate = paired_environment_delta_bca(&pairs, 500, 42).unwrap();
        assert_eq!(estimate.n_pairs, 4);
        assert!((estimate.mean_delta - 0.10).abs() < 1e-12);
    }

    #[test]
    fn sesoi_classification_requires_interval_level_evidence() {
        let estimate = |low, high| PairedEstimate {
            n_pairs: 20,
            mean_delta: (low + high) / 2.0,
            ci95_low: low,
            ci95_high: high,
        };

        assert_eq!(
            classify_practical_effect(&estimate(0.06, 0.12), 0.05).unwrap(),
            PracticalEffect::MeaningfulGain
        );
        assert_eq!(
            classify_practical_effect(&estimate(-0.12, -0.06), 0.05).unwrap(),
            PracticalEffect::MeaningfulRegression
        );
        assert_eq!(
            classify_practical_effect(&estimate(-0.03, 0.04), 0.05).unwrap(),
            PracticalEffect::Equivalent
        );
        assert_eq!(
            classify_practical_effect(&estimate(0.01, 0.08), 0.05).unwrap(),
            PracticalEffect::Inconclusive
        );
    }

    #[test]
    fn support_validation_fails_closed_on_accidental_overlap() {
        let program = task(&["depth<=2", "known_roles"], &["depth=3", "known_roles"]);
        assert!(validate_task_support(&program, SupportOverlapPolicy::Disjoint).is_err());
        validate_task_support(&program, SupportOverlapPolicy::AllowDeclaredOverlap).unwrap();
    }

    #[test]
    fn support_validation_rejects_duplicates_after_trimming() {
        let program = task(&["depth<=2", " depth<=2 "], &["depth=3"]);
        assert!(
            validate_task_support(&program, SupportOverlapPolicy::AllowDeclaredOverlap).is_err()
        );
    }

    #[test]
    fn claim_evidence_binding_requires_exact_claim_and_artifact_identity() {
        let claim = claim();
        let binding = binding();
        binding.validate_for_claim(&claim).unwrap();

        let mut wrong_claim = binding.clone();
        wrong_claim.claim_id = "different-claim".into();
        assert!(wrong_claim.validate_for_claim(&claim).is_err());

        let mut duplicate_task = binding;
        duplicate_task.task_program_digests = vec![digest(11), digest(11)];
        assert!(duplicate_task.validate_for_claim(&claim).is_err());
    }
}
