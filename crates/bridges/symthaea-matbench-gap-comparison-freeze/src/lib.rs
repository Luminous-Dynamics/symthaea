// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pre-truth comparison freeze for the Matbench Benchmark Zero policy ladder.
//!
//! This crate constructs all V0 rankings together from one frozen retained
//! universe and commits the comparison design before any benchmark truth is
//! supplied to measurement code.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use symthaea_energy_benchmark_zero::{
    BandgapTarget, BenchmarkError, ScreeningMethodProvenance, ScreeningRun,
};
use symthaea_energy_benchmark_zero_exact::{candidate_universe_sha256, ExactUniverseError};
use symthaea_matbench_gap_baseline_screening::{
    screen_baseline, BaselineScreeningPolicy, BaselineScreeningReceipt, ScreeningError,
};
use symthaea_matbench_gap_exposure_plan::{ExposedTrainingExclusionPlan, PlanError};
use symthaea_matbench_gap_learned_screening::{
    screen_learned, LearnedScreeningError, LearnedScreeningReceipt,
};
use thiserror::Error;

pub const FREEZE_SCHEMA: &str = "symthaea.matbench-gap.comparison-freeze.v0";
pub const CAPABILITY_CLASSIFICATION: &str =
    "PRE-TRUTH COMPARISON FREEZE ONLY -- not benchmark truth, not a superiority claim, and not candidate promotion authority.";

const LEGACY_PREDICTION_SURFACE_DOMAIN: &[u8] =
    b"symthaea.matbench-gap.legacy-prediction-surface.v0\0";
const COMPARISON_SUBJECT_DOMAIN: &[u8] =
    b"symthaea.matbench-gap.comparison-subject.v0\0";
const FREEZE_RECEIPT_DOMAIN: &[u8] =
    b"symthaea.matbench-gap.comparison-freeze-receipt.v0\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointDirection {
    Minimize,
    Maximize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonEndpoint {
    TargetRegretEv,
    TopKHits,
    MeanAbsPredictionErrorEv,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EndpointSpec {
    pub endpoint: ComparisonEndpoint,
    pub direction: EndpointDirection,
}

/// Fixed non-scalar endpoint vector for V0.
///
/// No weighted sum, hidden score, or post-truth endpoint selection is defined.
pub fn endpoint_contract() -> Vec<EndpointSpec> {
    vec![
        EndpointSpec {
            endpoint: ComparisonEndpoint::TargetRegretEv,
            direction: EndpointDirection::Minimize,
        },
        EndpointSpec {
            endpoint: ComparisonEndpoint::TopKHits,
            direction: EndpointDirection::Maximize,
        },
        EndpointSpec {
            endpoint: ComparisonEndpoint::MeanAbsPredictionErrorEv,
            direction: EndpointDirection::Minimize,
        },
    ]
}

/// One immutable pre-truth comparison subject containing all V0 policy outputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparisonFreezeReceipt {
    pub schema: String,
    pub capability_classification: String,
    /// Exact full exposure-plan identity retained for provenance. Deliberately
    /// excluded from the truth-free comparison-subject digest.
    pub source_plan_sha256: String,
    pub source_composition_order_sha256: String,
    pub source_partition_sha256: String,
    pub symthaea_training_snapshot_sha256: String,
    pub retained_universe_sha256: String,
    /// Order-independent identity of the exact candidate-id set common to all
    /// policies and later exact-universe truth measurement.
    pub candidate_universe_sha256: String,
    pub candidate_count: usize,
    pub target: BandgapTarget,
    /// Preregistered shortlist/evaluation budget used later by Benchmark Zero.
    pub top_k: usize,
    /// Seed for the deterministic random-order null control.
    pub random_seed: u64,
    pub endpoints: Vec<EndpointSpec>,
    /// Bit-exact candidate -> prediction surface shared by the two legacy
    /// controls. Their only intended difference is ordering.
    pub legacy_prediction_surface_sha256: String,
    pub random_control: BaselineScreeningReceipt,
    pub legacy_target_distance: BaselineScreeningReceipt,
    pub learned: LearnedScreeningReceipt,
    /// Truth-free identity of the complete preregistered comparison subject.
    pub comparison_subject_sha256: String,
}

impl ComparisonFreezeReceipt {
    pub fn validate(&self) -> Result<(), ComparisonFreezeError> {
        if self.schema != FREEZE_SCHEMA
            || self.capability_classification != CAPABILITY_CLASSIFICATION
        {
            return Err(ComparisonFreezeError::InvalidFreeze(
                "schema or capability classification was altered".into(),
            ));
        }
        for (name, value) in [
            ("source plan digest", self.source_plan_sha256.as_str()),
            (
                "source composition-order digest",
                self.source_composition_order_sha256.as_str(),
            ),
            ("source partition digest", self.source_partition_sha256.as_str()),
            (
                "training snapshot digest",
                self.symthaea_training_snapshot_sha256.as_str(),
            ),
            (
                "retained universe digest",
                self.retained_universe_sha256.as_str(),
            ),
            (
                "candidate universe digest",
                self.candidate_universe_sha256.as_str(),
            ),
            (
                "legacy prediction-surface digest",
                self.legacy_prediction_surface_sha256.as_str(),
            ),
            (
                "comparison-subject digest",
                self.comparison_subject_sha256.as_str(),
            ),
        ] {
            validate_256_bit_hex_digest(value, name)?;
        }

        self.target.validate()?;
        if self.candidate_count == 0 || self.top_k == 0 || self.top_k > self.candidate_count {
            return Err(ComparisonFreezeError::InvalidFreeze(
                "top_k must be in 1..=candidate_count and candidate_count must be non-zero".into(),
            ));
        }
        if self.endpoints != endpoint_contract() {
            return Err(ComparisonFreezeError::InvalidFreeze(
                "endpoint contract differs from fixed V0 non-scalar endpoint vector".into(),
            ));
        }

        self.random_control.validate()?;
        self.legacy_target_distance.validate()?;
        self.learned.validate()?;

        for ranked_count in [
            self.random_control.ranked_candidate_count,
            self.legacy_target_distance.ranked_candidate_count,
            self.learned.ranked_candidate_count,
        ] {
            if ranked_count != self.candidate_count {
                return Err(ComparisonFreezeError::InvalidFreeze(
                    "every policy must rank the complete frozen candidate universe".into(),
                ));
            }
        }

        match &self.random_control.policy {
            BaselineScreeningPolicy::DeterministicRandomOrder { seed }
                if *seed == self.random_seed => {}
            _ => {
                return Err(ComparisonFreezeError::InvalidFreeze(
                    "random-control receipt does not bind the preregistered random seed".into(),
                ));
            }
        }
        if !matches!(
            &self.legacy_target_distance.policy,
            BaselineScreeningPolicy::TargetDistance
        ) {
            return Err(ComparisonFreezeError::InvalidFreeze(
                "legacy heuristic receipt is not the TargetDistance policy".into(),
            ));
        }

        for target in [
            self.random_control.target,
            self.legacy_target_distance.target,
            self.learned.target,
        ] {
            if target != self.target {
                return Err(ComparisonFreezeError::InvalidFreeze(
                    "all policy receipts must share the exact preregistered target".into(),
                ));
            }
        }

        require_shared_plan_identity(self)?;
        require_exact_candidate_set_parity(self)?;
        require_identical_legacy_prediction_surface(
            &self.random_control.run,
            &self.legacy_target_distance.run,
        )?;

        let observed_surface = legacy_prediction_surface_sha256(&self.random_control.run)?;
        if observed_surface != self.legacy_prediction_surface_sha256 {
            return Err(ComparisonFreezeError::InvalidFreeze(
                "legacy prediction-surface digest does not match recorded predictions".into(),
            ));
        }

        let observed_candidate_digest = candidate_universe_sha256(
            self.learned
                .run
                .ranked
                .iter()
                .map(|record| record.candidate_id.as_str()),
        )?;
        if observed_candidate_digest != self.candidate_universe_sha256 {
            return Err(ComparisonFreezeError::InvalidFreeze(
                "candidate-universe digest does not match policy candidate ids".into(),
            ));
        }

        let expected_subject = comparison_subject_sha256(self)?;
        if expected_subject != self.comparison_subject_sha256 {
            return Err(ComparisonFreezeError::InvalidFreeze(
                "comparison-subject digest does not match frozen truth-free inputs".into(),
            ));
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, ComparisonFreezeError> {
        self.validate()?;
        domain_separated_sha256(FREEZE_RECEIPT_DOMAIN, self)
    }
}

/// Generate all V0 policies from the same frozen truth-free plan and commit the
/// comparison design before any benchmark truth is accepted.
pub fn freeze_comparison(
    plan: &ExposedTrainingExclusionPlan,
    target: BandgapTarget,
    top_k: usize,
    random_seed: u64,
) -> Result<ComparisonFreezeReceipt, ComparisonFreezeError> {
    plan.validate_against_current_training_snapshot()?;
    target.validate()?;

    let random_control = screen_baseline(
        plan,
        target,
        BaselineScreeningPolicy::DeterministicRandomOrder { seed: random_seed },
    )?;
    let legacy_target_distance =
        screen_baseline(plan, target, BaselineScreeningPolicy::TargetDistance)?;
    let learned = screen_learned(plan, target)?;

    let candidate_count = plan.retained_composition_count;
    if candidate_count == 0 || top_k == 0 || top_k > candidate_count {
        return Err(ComparisonFreezeError::InvalidFreeze(
            "top_k must be in 1..=retained candidate count".into(),
        ));
    }

    let plan_candidate_ids: Vec<&str> = plan.retained_candidate_ids().collect();
    let candidate_universe_sha256 = candidate_universe_sha256(plan_candidate_ids.iter().copied())?;
    let legacy_prediction_surface_sha256 =
        legacy_prediction_surface_sha256(&random_control.run)?;

    let mut receipt = ComparisonFreezeReceipt {
        schema: FREEZE_SCHEMA.into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        source_plan_sha256: plan.sha256()?,
        source_composition_order_sha256: plan.source_composition_order_sha256.clone(),
        source_partition_sha256: plan.partition_sha256.clone(),
        symthaea_training_snapshot_sha256: plan.symthaea_training_snapshot_sha256.clone(),
        retained_universe_sha256: plan.retained_universe_sha256.clone(),
        candidate_universe_sha256,
        candidate_count,
        target,
        top_k,
        random_seed,
        endpoints: endpoint_contract(),
        legacy_prediction_surface_sha256,
        random_control,
        legacy_target_distance,
        learned,
        comparison_subject_sha256: String::new(),
    };
    receipt.comparison_subject_sha256 = comparison_subject_sha256(&receipt)?;
    receipt.validate()?;
    Ok(receipt)
}

/// Replay all three policies from the exact supplied plan and require the same
/// frozen comparison subject and receipt.
pub fn verify_comparison_freeze(
    plan: &ExposedTrainingExclusionPlan,
    expected: &ComparisonFreezeReceipt,
) -> Result<(), ComparisonFreezeError> {
    expected.validate()?;
    plan.validate_against_current_training_snapshot()?;
    if expected.source_plan_sha256 != plan.sha256()?
        || expected.source_composition_order_sha256 != plan.source_composition_order_sha256
        || expected.source_partition_sha256 != plan.partition_sha256
        || expected.symthaea_training_snapshot_sha256 != plan.symthaea_training_snapshot_sha256
        || expected.retained_universe_sha256 != plan.retained_universe_sha256
        || expected.candidate_count != plan.retained_composition_count
    {
        return Err(ComparisonFreezeError::PlanIdentityMismatch);
    }

    let observed = freeze_comparison(
        plan,
        expected.target,
        expected.top_k,
        expected.random_seed,
    )?;
    if &observed != expected {
        return Err(ComparisonFreezeError::ReplayMismatch);
    }
    Ok(())
}

fn require_shared_plan_identity(
    receipt: &ComparisonFreezeReceipt,
) -> Result<(), ComparisonFreezeError> {
    let expected = (
        receipt.source_plan_sha256.as_str(),
        receipt.source_composition_order_sha256.as_str(),
        receipt.source_partition_sha256.as_str(),
        receipt.symthaea_training_snapshot_sha256.as_str(),
        receipt.retained_universe_sha256.as_str(),
    );
    for observed in [
        (
            receipt.random_control.source_plan_sha256.as_str(),
            receipt.random_control.source_composition_order_sha256.as_str(),
            receipt.random_control.source_partition_sha256.as_str(),
            receipt.random_control.symthaea_training_snapshot_sha256.as_str(),
            receipt.random_control.retained_universe_sha256.as_str(),
        ),
        (
            receipt.legacy_target_distance.source_plan_sha256.as_str(),
            receipt
                .legacy_target_distance
                .source_composition_order_sha256
                .as_str(),
            receipt.legacy_target_distance.source_partition_sha256.as_str(),
            receipt
                .legacy_target_distance
                .symthaea_training_snapshot_sha256
                .as_str(),
            receipt.legacy_target_distance.retained_universe_sha256.as_str(),
        ),
        (
            receipt.learned.source_plan_sha256.as_str(),
            receipt.learned.source_composition_order_sha256.as_str(),
            receipt.learned.source_partition_sha256.as_str(),
            receipt.learned.symthaea_training_snapshot_sha256.as_str(),
            receipt.learned.retained_universe_sha256.as_str(),
        ),
    ] {
        if observed != expected {
            return Err(ComparisonFreezeError::InvalidFreeze(
                "policy receipts do not share the frozen source-plan identities".into(),
            ));
        }
    }
    Ok(())
}

fn require_exact_candidate_set_parity(
    receipt: &ComparisonFreezeReceipt,
) -> Result<(), ComparisonFreezeError> {
    let random = candidate_set(&receipt.random_control.run)?;
    let legacy = candidate_set(&receipt.legacy_target_distance.run)?;
    let learned = candidate_set(&receipt.learned.run)?;
    if random != legacy || random != learned || random.len() != receipt.candidate_count {
        return Err(ComparisonFreezeError::CandidateSetMismatch);
    }
    Ok(())
}

fn candidate_set(run: &ScreeningRun) -> Result<BTreeSet<&str>, ComparisonFreezeError> {
    run.validate()?;
    Ok(run
        .ranked
        .iter()
        .map(|record| record.candidate_id.as_str())
        .collect())
}

fn prediction_surface(run: &ScreeningRun) -> Result<BTreeMap<String, u64>, ComparisonFreezeError> {
    run.validate()?;
    Ok(run
        .ranked
        .iter()
        .map(|record| (record.candidate_id.clone(), record.predicted_gap_ev.to_bits()))
        .collect())
}

fn require_identical_legacy_prediction_surface(
    random: &ScreeningRun,
    legacy: &ScreeningRun,
) -> Result<(), ComparisonFreezeError> {
    if prediction_surface(random)? != prediction_surface(legacy)? {
        return Err(ComparisonFreezeError::LegacyPredictionSurfaceMismatch);
    }
    Ok(())
}

fn legacy_prediction_surface_sha256(
    run: &ScreeningRun,
) -> Result<String, ComparisonFreezeError> {
    domain_separated_sha256(LEGACY_PREDICTION_SURFACE_DOMAIN, &prediction_surface(run)?)
}

#[derive(Serialize)]
struct PolicyCommitment<'a> {
    role: &'static str,
    method: &'a ScreeningMethodProvenance,
    screening_subject_sha256: &'a str,
    ranking_digest: &'a str,
}

#[derive(Serialize)]
struct ComparisonSubject<'a> {
    source_composition_order_sha256: &'a str,
    source_partition_sha256: &'a str,
    symthaea_training_snapshot_sha256: &'a str,
    retained_universe_sha256: &'a str,
    candidate_universe_sha256: &'a str,
    candidate_count: usize,
    target: BandgapTarget,
    top_k: usize,
    random_seed: u64,
    endpoints: &'a [EndpointSpec],
    legacy_prediction_surface_sha256: &'a str,
    policies: [PolicyCommitment<'a>; 3],
}

fn comparison_subject_sha256(
    receipt: &ComparisonFreezeReceipt,
) -> Result<String, ComparisonFreezeError> {
    let subject = ComparisonSubject {
        source_composition_order_sha256: &receipt.source_composition_order_sha256,
        source_partition_sha256: &receipt.source_partition_sha256,
        symthaea_training_snapshot_sha256: &receipt.symthaea_training_snapshot_sha256,
        retained_universe_sha256: &receipt.retained_universe_sha256,
        candidate_universe_sha256: &receipt.candidate_universe_sha256,
        candidate_count: receipt.candidate_count,
        target: receipt.target,
        top_k: receipt.top_k,
        random_seed: receipt.random_seed,
        endpoints: &receipt.endpoints,
        legacy_prediction_surface_sha256: &receipt.legacy_prediction_surface_sha256,
        policies: [
            PolicyCommitment {
                role: "deterministic_random_null",
                method: &receipt.random_control.run.method,
                screening_subject_sha256: &receipt.random_control.screening_subject_sha256,
                ranking_digest: &receipt.random_control.ranking_digest,
            },
            PolicyCommitment {
                role: "legacy_target_distance",
                method: &receipt.legacy_target_distance.run.method,
                screening_subject_sha256: &receipt
                    .legacy_target_distance
                    .screening_subject_sha256,
                ranking_digest: &receipt.legacy_target_distance.ranking_digest,
            },
            PolicyCommitment {
                role: "composition_only_learned",
                method: &receipt.learned.run.method,
                screening_subject_sha256: &receipt.learned.screening_subject_sha256,
                ranking_digest: &receipt.learned.ranking_digest,
            },
        ],
    };
    domain_separated_sha256(COMPARISON_SUBJECT_DOMAIN, &subject)
}

fn validate_256_bit_hex_digest(
    value: &str,
    name: &str,
) -> Result<(), ComparisonFreezeError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ComparisonFreezeError::InvalidFreeze(format!(
            "{name} must be exactly 64 hexadecimal characters"
        )));
    }
    Ok(())
}

fn domain_separated_sha256<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<String, ComparisonFreezeError> {
    let encoded = serde_json::to_vec(value)?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(encoded);
    let digest = hasher.finalize();
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(output)
}

#[derive(Debug, Error)]
pub enum ComparisonFreezeError {
    #[error("exposure plan rejected: {0}")]
    Plan(#[from] PlanError),
    #[error("Benchmark Zero contract rejected: {0}")]
    Benchmark(#[from] BenchmarkError),
    #[error("baseline screening rejected: {0}")]
    Baseline(#[from] ScreeningError),
    #[error("learned screening rejected: {0}")]
    Learned(#[from] LearnedScreeningError),
    #[error("exact candidate-universe contract rejected: {0}")]
    ExactUniverse(#[from] ExactUniverseError),
    #[error("policy candidate sets differ")]
    CandidateSetMismatch,
    #[error("random and legacy controls do not share the exact prediction surface")]
    LegacyPredictionSurfaceMismatch,
    #[error("comparison freeze does not bind the supplied exposure plan")]
    PlanIdentityMismatch,
    #[error("comparison freeze replay differs from supplied receipt")]
    ReplayMismatch,
    #[error("invalid comparison freeze: {0}")]
    InvalidFreeze(String),
    #[error("JSON encoding failed: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use symthaea_energy_benchmark_zero::{ScreeningMethodProvenance, ScreeningRecord};

    fn run(order: &[(&str, f64)]) -> ScreeningRun {
        ScreeningRun {
            method: ScreeningMethodProvenance {
                method_id: "fixture".into(),
                version: "v0".into(),
                training_slices: BTreeSet::new(),
            },
            target: BandgapTarget::new(1.0, 2.0).unwrap(),
            ranked: order
                .iter()
                .map(|(candidate, prediction)| ScreeningRecord {
                    candidate_id: (*candidate).into(),
                    predicted_gap_ev: *prediction,
                    uncertainty_ev: None,
                })
                .collect(),
        }
    }

    #[test]
    fn endpoint_contract_is_non_scalar_and_directional() {
        assert_eq!(
            endpoint_contract(),
            vec![
                EndpointSpec {
                    endpoint: ComparisonEndpoint::TargetRegretEv,
                    direction: EndpointDirection::Minimize,
                },
                EndpointSpec {
                    endpoint: ComparisonEndpoint::TopKHits,
                    direction: EndpointDirection::Maximize,
                },
                EndpointSpec {
                    endpoint: ComparisonEndpoint::MeanAbsPredictionErrorEv,
                    direction: EndpointDirection::Minimize,
                },
            ]
        );
    }

    #[test]
    fn legacy_prediction_surface_is_order_independent() {
        let first = run(&[("A", 1.1), ("B", 1.5), ("C", 2.0)]);
        let second = run(&[("C", 2.0), ("A", 1.1), ("B", 1.5)]);
        require_identical_legacy_prediction_surface(&first, &second).unwrap();
        assert_eq!(
            legacy_prediction_surface_sha256(&first).unwrap(),
            legacy_prediction_surface_sha256(&second).unwrap()
        );
    }

    #[test]
    fn prediction_change_is_not_explained_as_ordering_only() {
        let first = run(&[("A", 1.1), ("B", 1.5)]);
        let second = run(&[("B", 1.5), ("A", 1.2)]);
        assert!(matches!(
            require_identical_legacy_prediction_surface(&first, &second),
            Err(ComparisonFreezeError::LegacyPredictionSurfaceMismatch)
        ));
    }

    #[test]
    fn candidate_set_parity_ignores_order_but_not_membership() {
        let first_run = run(&[("A", 1.0), ("B", 2.0)]);
        let reordered_run = run(&[("B", 2.0), ("A", 1.0)]);
        let different_run = run(&[("A", 1.0), ("C", 2.0)]);
        let first = candidate_set(&first_run).unwrap();
        let reordered = candidate_set(&reordered_run).unwrap();
        let different = candidate_set(&different_run).unwrap();
        assert_eq!(first, reordered);
        assert_ne!(first, different);
    }
}
