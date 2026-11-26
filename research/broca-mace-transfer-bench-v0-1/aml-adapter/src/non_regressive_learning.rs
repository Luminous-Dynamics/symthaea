// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-gated, non-regressive learning admission.
//!
//! An adaptive update is compared on identical stable and adaptation task
//! populations before and after the update. Evaluation protocol, benchmark
//! release, grader and comparison group are pinned, and every run must be
//! independently verified. A stable-stratum regression beyond the consumer's
//! declared budget rejects the update even when adaptation performance rises.
//!
//! This module evaluates score receipts; it does not train, roll back, deploy,
//! or authorize a model update.

use sha2::{Digest, Sha256};

const MAX_TASKS_V1: usize = 1_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LearningStratumV1 {
    Stable,
    Adaptation,
}

impl LearningStratumV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Stable => 1,
            Self::Adaptation => 2,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LearningTaskScoreV1 {
    /// Stable opaque identity; must be sorted and unique within a run.
    pub task_id: String,
    /// Official per-task score in basis points: 0..=10,000.
    /// Timeout/abstention/invalid outcomes must use the official grader's score.
    pub score_basis_points: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LearningRunSnapshotV1 {
    pub run_id: String,
    pub benchmark_id: String,
    pub benchmark_release_sha256: String,
    pub split_id: String,
    pub stratum: LearningStratumV1,
    /// Pins prompts, tools, decoding, resource policy and scorer protocol.
    pub evaluation_protocol_sha256: String,
    pub grader_sha256: String,
    /// Digest of the exact sorted task IDs in task_scores.
    pub task_set_sha256: String,
    /// Pins the matched model family/architecture/runtime comparison group.
    pub comparison_group_sha256: String,
    /// Digest of the candidate's complete learned state/checkpoint.
    pub candidate_state_sha256: String,
    /// Raw, immutable run receipt or official evaluator output commitment.
    pub run_receipt_sha256: String,
    pub task_scores: Vec<LearningTaskScoreV1>,
}

impl LearningRunSnapshotV1 {
    pub fn computed_task_set_sha256(&self) -> String {
        task_set_digest(&self.task_scores)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LearningRunVerificationReceiptV1 {
    pub verifier_id: String,
    pub verification_receipt_sha256: String,
}

pub trait LearningRunVerifierV1 {
    /// Return Some only after independently verifying the exact benchmark,
    /// task population, grader, candidate state, run receipt and score output.
    /// Missing/unverifiable receipts must return None.
    fn verify_run(
        &self,
        run: &LearningRunSnapshotV1,
    ) -> Option<LearningRunVerificationReceiptV1>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LearningRetentionPolicyV1 {
    /// Maximum permitted stable-stratum regression, in basis points.
    /// 100 basis points = 1.00 percentage point.
    pub maximum_stable_regression_basis_points: u16,
    /// Minimum required adaptation-stratum gain, in basis points.
    /// Zero means the adaptation stratum must not regress.
    pub minimum_adaptation_gain_basis_points: u16,
}

impl Default for LearningRetentionPolicyV1 {
    fn default() -> Self {
        Self {
            maximum_stable_regression_basis_points: 100,
            minimum_adaptation_gain_basis_points: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LearningGateDispositionV1 {
    AcceptedWithinRetentionBudget,
    RejectedStableRegression,
    RejectedInsufficientAdaptationGain,
    InconclusiveUnverifiedRun,
}

impl LearningGateDispositionV1 {
    fn tag(self) -> u8 {
        match self {
            Self::AcceptedWithinRetentionBudget => 1,
            Self::RejectedStableRegression => 2,
            Self::RejectedInsufficientAdaptationGain => 3,
            Self::InconclusiveUnverifiedRun => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LearningGateRejectV1 {
    InvalidIdentifier,
    InvalidDigest,
    EmptyTaskSet,
    TooManyTasks,
    UnsortedOrDuplicateTaskIds,
    InvalidTaskScore,
    TaskSetDigestMismatch,
    DuplicateRunId,
    StratumMismatch,
    BenchmarkMismatch,
    EvaluationProtocolMismatch,
    ComparisonGroupMismatch,
    CandidateStateMismatch,
    CandidateStateDidNotChange,
    TaskPopulationMismatch,
    InvalidRetentionPolicy,
    InvalidVerifierReceipt,
    ArithmeticOverflow,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LearningGateAssessmentV1 {
    pub disposition: LearningGateDispositionV1,
    /// Sum(after scores) - sum(before scores), in basis-point units.
    /// Divide by the corresponding task count to obtain mean per-task change.
    pub stable_delta_sum_basis_points: Option<i128>,
    pub stable_task_count: u64,
    pub adaptation_delta_sum_basis_points: Option<i128>,
    pub adaptation_task_count: u64,
    pub maximum_stable_regression_basis_points: u16,
    pub minimum_adaptation_gain_basis_points: u16,
    /// Verifier receipts in stable-before, stable-after, adaptation-before,
    /// adaptation-after order. Empty for an inconclusive/unverified gate.
    pub verifier_ids: Vec<String>,
    pub verification_receipt_sha256: Vec<String>,
    pub assessment_sha256: String,
}

impl LearningGateAssessmentV1 {
    pub fn verify_integrity(&self) -> bool {
        self.assessment_sha256 == assessment_digest(
            self.disposition,
            self.stable_delta_sum_basis_points,
            self.stable_task_count,
            self.adaptation_delta_sum_basis_points,
            self.adaptation_task_count,
            self.maximum_stable_regression_basis_points,
            self.minimum_adaptation_gain_basis_points,
            &self.verifier_ids,
            &self.verification_receipt_sha256,
        )
    }

    pub fn admits_learning_update(&self) -> bool {
        self.disposition == LearningGateDispositionV1::AcceptedWithinRetentionBudget
            && self.verify_integrity()
    }
}

/// Compare four independently verified runs:
/// stable-before, stable-after, adaptation-before, adaptation-after.
///
/// Both before/after pairs must have the exact same task IDs. All four runs
/// must share the benchmark release, split, grader, evaluation protocol and
/// comparison group. Before runs must use the same candidate state; after
/// runs must use the same new candidate state, and those states must differ.
///
/// The result is an admission assessment only. The separate deployment or
/// model-update authority decides whether to commit, roll back or quarantine.
pub fn evaluate_non_regressive_learning_v1(
    stable_before: &LearningRunSnapshotV1,
    stable_after: &LearningRunSnapshotV1,
    adaptation_before: &LearningRunSnapshotV1,
    adaptation_after: &LearningRunSnapshotV1,
    policy: LearningRetentionPolicyV1,
    verifier: Option<&dyn LearningRunVerifierV1>,
) -> Result<LearningGateAssessmentV1, LearningGateRejectV1> {
    validate_policy(policy)?;
    validate_run(stable_before)?;
    validate_run(stable_after)?;
    validate_run(adaptation_before)?;
    validate_run(adaptation_after)?;

    if stable_before.stratum != LearningStratumV1::Stable
        || stable_after.stratum != LearningStratumV1::Stable
        || adaptation_before.stratum != LearningStratumV1::Adaptation
        || adaptation_after.stratum != LearningStratumV1::Adaptation
    {
        return Err(LearningGateRejectV1::StratumMismatch);
    }

    validate_same_evaluation(stable_before, stable_after)?;
    validate_same_evaluation(stable_before, adaptation_before)?;
    validate_same_evaluation(stable_before, adaptation_after)?;
    validate_same_evaluation(stable_after, adaptation_after)?;

    if stable_before.candidate_state_sha256 != adaptation_before.candidate_state_sha256
        || stable_after.candidate_state_sha256 != adaptation_after.candidate_state_sha256
    {
        return Err(LearningGateRejectV1::CandidateStateMismatch);
    }
    if stable_before.candidate_state_sha256 == stable_after.candidate_state_sha256 {
        return Err(LearningGateRejectV1::CandidateStateDidNotChange);
    }

    ensure_distinct_run_ids(&[
        stable_before,
        stable_after,
        adaptation_before,
        adaptation_after,
    ])?;
    validate_task_pair(stable_before, stable_after)?;
    validate_task_pair(adaptation_before, adaptation_after)?;

    let Some(verifier) = verifier else {
        return Ok(inconclusive_assessment(policy));
    };
    let runs = [stable_before, stable_after, adaptation_before, adaptation_after];
    let mut verifier_ids = Vec::with_capacity(4);
    let mut verification_digests = Vec::with_capacity(4);
    for run in runs {
        let Some(receipt) = verifier.verify_run(run) else {
            return Ok(inconclusive_assessment(policy));
        };
        if !valid_atom(&receipt.verifier_id, 512) {
            return Err(LearningGateRejectV1::InvalidVerifierReceipt);
        }
        validate_digest(&receipt.verification_receipt_sha256)
            .map_err(|_| LearningGateRejectV1::InvalidVerifierReceipt)?;
        verifier_ids.push(receipt.verifier_id);
        verification_digests.push(receipt.verification_receipt_sha256);
    }

    let stable_before_sum = score_sum(stable_before)?;
    let stable_after_sum = score_sum(stable_after)?;
    let adaptation_before_sum = score_sum(adaptation_before)?;
    let adaptation_after_sum = score_sum(adaptation_after)?;
    let stable_delta = i128::from(stable_after_sum) - i128::from(stable_before_sum);
    let adaptation_delta = i128::from(adaptation_after_sum) - i128::from(adaptation_before_sum);
    let stable_count = stable_before.task_scores.len() as u64;
    let adaptation_count = adaptation_before.task_scores.len() as u64;

    let maximum_regression = i128::from(policy.maximum_stable_regression_basis_points)
        .checked_mul(i128::from(stable_count))
        .ok_or(LearningGateRejectV1::ArithmeticOverflow)?;
    let minimum_gain = i128::from(policy.minimum_adaptation_gain_basis_points)
        .checked_mul(i128::from(adaptation_count))
        .ok_or(LearningGateRejectV1::ArithmeticOverflow)?;

    let disposition = if stable_delta < -maximum_regression {
        LearningGateDispositionV1::RejectedStableRegression
    } else if adaptation_delta < minimum_gain {
        LearningGateDispositionV1::RejectedInsufficientAdaptationGain
    } else {
        LearningGateDispositionV1::AcceptedWithinRetentionBudget
    };

    Ok(build_assessment(
        disposition,
        Some(stable_delta),
        stable_count,
        Some(adaptation_delta),
        adaptation_count,
        policy,
        verifier_ids,
        verification_digests,
    ))
}

fn validate_run(run: &LearningRunSnapshotV1) -> Result<(), LearningGateRejectV1> {
    if !valid_atom(&run.run_id, 512)
        || !valid_atom(&run.benchmark_id, 512)
        || !valid_atom(&run.split_id, 512)
    {
        return Err(LearningGateRejectV1::InvalidIdentifier);
    }
    validate_digest(&run.benchmark_release_sha256)?;
    validate_digest(&run.evaluation_protocol_sha256)?;
    validate_digest(&run.grader_sha256)?;
    validate_digest(&run.task_set_sha256)?;
    validate_digest(&run.comparison_group_sha256)?;
    validate_digest(&run.candidate_state_sha256)?;
    validate_digest(&run.run_receipt_sha256)?;
    if run.task_scores.is_empty() {
        return Err(LearningGateRejectV1::EmptyTaskSet);
    }
    if run.task_scores.len() > MAX_TASKS_V1 {
        return Err(LearningGateRejectV1::TooManyTasks);
    }
    for score in &run.task_scores {
        if !valid_atom(&score.task_id, 512) {
            return Err(LearningGateRejectV1::InvalidIdentifier);
        }
        if score.score_basis_points > 10_000 {
            return Err(LearningGateRejectV1::InvalidTaskScore);
        }
    }
    if !run.task_scores.windows(2).all(|pair| pair[0].task_id < pair[1].task_id) {
        return Err(LearningGateRejectV1::UnsortedOrDuplicateTaskIds);
    }
    if run.computed_task_set_sha256() != run.task_set_sha256 {
        return Err(LearningGateRejectV1::TaskSetDigestMismatch);
    }
    Ok(())
}

fn validate_same_evaluation(
    left: &LearningRunSnapshotV1,
    right: &LearningRunSnapshotV1,
) -> Result<(), LearningGateRejectV1> {
    if left.benchmark_id != right.benchmark_id
        || left.benchmark_release_sha256 != right.benchmark_release_sha256
        || left.split_id != right.split_id
    {
        return Err(LearningGateRejectV1::BenchmarkMismatch);
    }
    if left.evaluation_protocol_sha256 != right.evaluation_protocol_sha256
        || left.grader_sha256 != right.grader_sha256
    {
        return Err(LearningGateRejectV1::EvaluationProtocolMismatch);
    }
    if left.comparison_group_sha256 != right.comparison_group_sha256 {
        return Err(LearningGateRejectV1::ComparisonGroupMismatch);
    }
    Ok(())
}

fn validate_task_pair(
    before: &LearningRunSnapshotV1,
    after: &LearningRunSnapshotV1,
) -> Result<(), LearningGateRejectV1> {
    if before.stratum != after.stratum
        || before.task_set_sha256 != after.task_set_sha256
        || before.task_scores.len() != after.task_scores.len()
        || before.task_scores.iter().zip(&after.task_scores)
            .any(|(left, right)| left.task_id != right.task_id)
    {
        return Err(LearningGateRejectV1::TaskPopulationMismatch);
    }
    Ok(())
}

fn ensure_distinct_run_ids(
    runs: &[&LearningRunSnapshotV1],
) -> Result<(), LearningGateRejectV1> {
    for left in 0..runs.len() {
        for right in (left + 1)..runs.len() {
            if runs[left].run_id == runs[right].run_id {
                return Err(LearningGateRejectV1::DuplicateRunId);
            }
        }
    }
    Ok(())
}

fn score_sum(run: &LearningRunSnapshotV1) -> Result<u64, LearningGateRejectV1> {
    run.task_scores.iter().try_fold(0_u64, |sum, item| {
        sum.checked_add(u64::from(item.score_basis_points))
            .ok_or(LearningGateRejectV1::ArithmeticOverflow)
    })
}

fn validate_policy(policy: LearningRetentionPolicyV1) -> Result<(), LearningGateRejectV1> {
    if policy.maximum_stable_regression_basis_points > 10_000
        || policy.minimum_adaptation_gain_basis_points > 10_000
    {
        return Err(LearningGateRejectV1::InvalidRetentionPolicy);
    }
    Ok(())
}

fn inconclusive_assessment(policy: LearningRetentionPolicyV1) -> LearningGateAssessmentV1 {
    build_assessment(
        LearningGateDispositionV1::InconclusiveUnverifiedRun,
        None,
        0,
        None,
        0,
        policy,
        Vec::new(),
        Vec::new(),
    )
}

fn build_assessment(
    disposition: LearningGateDispositionV1,
    stable_delta: Option<i128>,
    stable_count: u64,
    adaptation_delta: Option<i128>,
    adaptation_count: u64,
    policy: LearningRetentionPolicyV1,
    verifier_ids: Vec<String>,
    verification_digests: Vec<String>,
) -> LearningGateAssessmentV1 {
    let assessment_sha256 = assessment_digest(
        disposition,
        stable_delta,
        stable_count,
        adaptation_delta,
        adaptation_count,
        policy,
        &verifier_ids,
        &verification_digests,
    );
    LearningGateAssessmentV1 {
        disposition,
        stable_delta_sum_basis_points: stable_delta,
        stable_task_count: stable_count,
        adaptation_delta_sum_basis_points: adaptation_delta,
        adaptation_task_count: adaptation_count,
        maximum_stable_regression_basis_points: policy.maximum_stable_regression_basis_points,
        minimum_adaptation_gain_basis_points: policy.minimum_adaptation_gain_basis_points,
        verifier_ids,
        verification_receipt_sha256: verification_digests,
        assessment_sha256,
    }
}

fn task_set_digest(scores: &[LearningTaskScoreV1]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:learning-task-set:v1\0");
    hasher.update((scores.len() as u64).to_be_bytes());
    for score in scores {
        hash_field(&mut hasher, &score.task_id);
    }
    hex_digest(hasher.finalize().as_slice())
}

fn assessment_digest(
    disposition: LearningGateDispositionV1,
    stable_delta: Option<i128>,
    stable_count: u64,
    adaptation_delta: Option<i128>,
    adaptation_count: u64,
    policy: LearningRetentionPolicyV1,
    verifier_ids: &[String],
    verification_digests: &[String],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:non-regressive-learning-assessment:v1\0");
    hasher.update([disposition.tag()]);
    hash_optional_i128(&mut hasher, stable_delta);
    hasher.update(stable_count.to_be_bytes());
    hash_optional_i128(&mut hasher, adaptation_delta);
    hasher.update(adaptation_count.to_be_bytes());
    hasher.update(policy.maximum_stable_regression_basis_points.to_be_bytes());
    hasher.update(policy.minimum_adaptation_gain_basis_points.to_be_bytes());
    hash_strings(&mut hasher, verifier_ids);
    hash_strings(&mut hasher, verification_digests);
    hex_digest(hasher.finalize().as_slice())
}

fn hash_optional_i128(hasher: &mut Sha256, value: Option<i128>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            hasher.update(value.to_be_bytes());
        }
        None => hasher.update([0]),
    }
}

fn hash_strings(hasher: &mut Sha256, values: &[String]) {
    hasher.update((values.len() as u64).to_be_bytes());
    for value in values {
        hash_field(hasher, value);
    }
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn valid_atom(value: &str, maximum: usize) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= maximum
        && !value.chars().any(char::is_control)
}

fn validate_digest(value: &str) -> Result<(), LearningGateRejectV1> {
    if value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(LearningGateRejectV1::InvalidDigest)
    }
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes.as_ref().iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(prefix: &str, values: &[u16]) -> Vec<LearningTaskScoreV1> {
        values.iter().enumerate().map(|(index, value)| LearningTaskScoreV1 {
            task_id: format!("{prefix}-{index:03}"),
            score_basis_points: *value,
        }).collect()
    }

    fn run(
        id: &str,
        stratum: LearningStratumV1,
        candidate: &str,
        task_prefix: &str,
        values: &[u16],
    ) -> LearningRunSnapshotV1 {
        let task_scores = scores(task_prefix, values);
        let mut result = LearningRunSnapshotV1 {
            run_id: id.to_owned(),
            benchmark_id: "servelearn-retention-fixture".into(),
            benchmark_release_sha256: "1".repeat(64),
            split_id: "frozen-paired-split".into(),
            stratum,
            evaluation_protocol_sha256: "2".repeat(64),
            grader_sha256: "3".repeat(64),
            task_set_sha256: String::new(),
            comparison_group_sha256: "4".repeat(64),
            candidate_state_sha256: candidate.to_owned(),
            run_receipt_sha256: "5".repeat(64),
            task_scores,
        };
        result.task_set_sha256 = result.computed_task_set_sha256();
        result
    }

    fn four_runs(stable_after_values: &[u16], adaptation_after_values: &[u16]) -> (
        LearningRunSnapshotV1,
        LearningRunSnapshotV1,
        LearningRunSnapshotV1,
        LearningRunSnapshotV1,
    ) {
        (
            run("stable-before", LearningStratumV1::Stable, &"a".repeat(64), "stable", &[9_000, 9_000]),
            run("stable-after", LearningStratumV1::Stable, &"b".repeat(64), "stable", stable_after_values),
            run("adapt-before", LearningStratumV1::Adaptation, &"a".repeat(64), "adapt", &[5_000, 5_000]),
            run("adapt-after", LearningStratumV1::Adaptation, &"b".repeat(64), "adapt", adaptation_after_values),
        )
    }

    struct FixtureRunVerifier;

    impl LearningRunVerifierV1 for FixtureRunVerifier {
        fn verify_run(&self, run: &LearningRunSnapshotV1) -> Option<LearningRunVerificationReceiptV1> {
            Some(LearningRunVerificationReceiptV1 {
                verifier_id: "fixture-independent-run-verifier".into(),
                verification_receipt_sha256: if run.run_id.is_empty() { "0".repeat(64) } else { "6".repeat(64) },
            })
        }
    }

    #[test]
    fn accepts_learning_with_stable_regression_exactly_at_one_percentage_point() {
        let (sb, sa, ab, aa) = four_runs(&[8_900, 8_900], &[6_000, 6_000]);
        let result = evaluate_non_regressive_learning_v1(
            &sb, &sa, &ab, &aa, LearningRetentionPolicyV1::default(), Some(&FixtureRunVerifier),
        ).unwrap();
        assert_eq!(result.disposition, LearningGateDispositionV1::AcceptedWithinRetentionBudget);
        assert_eq!(result.stable_delta_sum_basis_points, Some(-200));
        assert_eq!(result.adaptation_delta_sum_basis_points, Some(2_000));
        assert!(result.admits_learning_update());
    }

    #[test]
    fn rejects_stable_regression_beyond_the_declared_budget() {
        let (sb, sa, ab, aa) = four_runs(&[8_700, 8_700], &[7_000, 7_000]);
        let result = evaluate_non_regressive_learning_v1(
            &sb, &sa, &ab, &aa, LearningRetentionPolicyV1::default(), Some(&FixtureRunVerifier),
        ).unwrap();
        assert_eq!(result.disposition, LearningGateDispositionV1::RejectedStableRegression);
        assert!(!result.admits_learning_update());
    }

    #[test]
    fn rejects_an_update_that_does_not_improve_the_adaptation_stratum() {
        let (sb, sa, ab, aa) = four_runs(&[9_000, 9_000], &[4_900, 4_900]);
        let result = evaluate_non_regressive_learning_v1(
            &sb, &sa, &ab, &aa, LearningRetentionPolicyV1::default(), Some(&FixtureRunVerifier),
        ).unwrap();
        assert_eq!(result.disposition, LearningGateDispositionV1::RejectedInsufficientAdaptationGain);
    }

    #[test]
    fn missing_or_unverified_run_receipt_is_inconclusive() {
        let (sb, sa, ab, aa) = four_runs(&[9_000, 9_000], &[6_000, 6_000]);
        let result = evaluate_non_regressive_learning_v1(
            &sb, &sa, &ab, &aa, LearningRetentionPolicyV1::default(), None,
        ).unwrap();
        assert_eq!(result.disposition, LearningGateDispositionV1::InconclusiveUnverifiedRun);
        assert!(!result.admits_learning_update());
        assert!(result.verify_integrity());
    }

    #[test]
    fn different_task_population_cannot_be_claimed_as_paired_learning() {
        let (sb, mut sa, ab, aa) = four_runs(&[9_000, 9_000], &[6_000, 6_000]);
        sa.task_scores[1].task_id = "different-task".into();
        sa.task_scores.sort_by(|left, right| left.task_id.cmp(&right.task_id));
        sa.task_set_sha256 = sa.computed_task_set_sha256();
        assert_eq!(
            evaluate_non_regressive_learning_v1(
                &sb, &sa, &ab, &aa, LearningRetentionPolicyV1::default(), Some(&FixtureRunVerifier),
            ),
            Err(LearningGateRejectV1::TaskPopulationMismatch)
        );
    }

    #[test]
    fn protocol_or_grader_change_invalidates_a_matched_comparison() {
        let (sb, mut sa, ab, aa) = four_runs(&[9_000, 9_000], &[6_000, 6_000]);
        sa.grader_sha256 = "9".repeat(64);
        assert_eq!(
            evaluate_non_regressive_learning_v1(
                &sb, &sa, &ab, &aa, LearningRetentionPolicyV1::default(), Some(&FixtureRunVerifier),
            ),
            Err(LearningGateRejectV1::EvaluationProtocolMismatch)
        );
    }

    #[test]
    fn duplicate_run_ids_and_invalid_task_scores_are_rejected() {
        let (sb, mut sa, ab, aa) = four_runs(&[9_000, 9_000], &[6_000, 6_000]);
        sa.run_id = sb.run_id.clone();
        assert_eq!(
            evaluate_non_regressive_learning_v1(
                &sb, &sa, &ab, &aa, LearningRetentionPolicyV1::default(), Some(&FixtureRunVerifier),
            ),
            Err(LearningGateRejectV1::DuplicateRunId)
        );

        let (sb, mut sa, ab, aa) = four_runs(&[9_000, 9_000], &[6_000, 6_000]);
        sa.task_scores[0].score_basis_points = 10_001;
        assert_eq!(
            evaluate_non_regressive_learning_v1(
                &sb, &sa, &ab, &aa, LearningRetentionPolicyV1::default(), Some(&FixtureRunVerifier),
            ),
            Err(LearningGateRejectV1::InvalidTaskScore)
        );
    }
}
