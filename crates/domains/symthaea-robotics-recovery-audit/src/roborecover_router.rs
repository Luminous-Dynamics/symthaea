//! Conservative train-only policy portfolio selection for RoboRecover.
//!
//! This is an offline/reference selector, not a robot controller or actuator
//! authority. It consumes competence summaries built from the training split,
//! requires exact source/checkpoint/evaluator/cross-fit identities, ranks only
//! policies with sufficient known support and acceptable conservative risk,
//! and abstains when the predicted context or evidence is insufficient.

use std::collections::{BTreeMap, BTreeSet};

use super::roborecover_score::{
    BenchmarkPlatform, BenchmarkSplit, StageTypeGroup,
};

#[derive(Clone, Debug, PartialEq)]
pub struct RouterConfig {
    pub minimum_planned_trials: u32,
    pub minimum_known_coverage: f64,
    /// Minimum Wilson lower bound for physical-safe completion. Without this,
    /// a router could select the least-bad policy even when every policy is poor.
    pub minimum_physical_safe_success_lower_bound: f64,
    pub maximum_adverse_risk_upper_bound: f64,
    pub maximum_p95_latency_ms: f64,
    pub minimum_context_confidence: f64,
    pub wilson_z_score: f64,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            minimum_planned_trials: 50,
            minimum_known_coverage: 0.90,
            minimum_physical_safe_success_lower_bound: 0.60,
            maximum_adverse_risk_upper_bound: 0.10,
            maximum_p95_latency_ms: 1_000.0,
            minimum_context_confidence: 0.80,
            wilson_z_score: 1.96,
        }
    }
}

/// Runtime context must be inferred from deployable observations.
/// It must not be populated from RoboRecover evaluator-only stage/type labels.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictedPolicyContext {
    pub group: Option<StageTypeGroup>,
    pub confidence: f64,
}

/// All portfolio members must use the same frozen training and cross-fit identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouterEvidenceIdentity {
    pub platform: BenchmarkPlatform,
    pub source_repository: String,
    pub source_revision: String,
    pub train_split_manifest_blob_sha: String,
    pub evaluator_digest: String,
    pub competence_table_digest: String,
    pub crossfit_partition_digest: String,
    /// Exact robot/arm embodiment; do not pool competence across embodiments.
    pub robot_embodiment_digest: String,
    /// Observation and action contract used to produce and execute policies.
    pub observation_action_schema_digest: String,
    /// Frozen feature/context encoder used to infer PredictedPolicyContext.
    pub context_encoder_digest: String,
    /// Runtime and hardware identity for latency measurements.
    pub runtime_environment_digest: String,
}

/// Policy competence counts are derived only from the training split.
///
/// Unknown, contradictory, invalid and missing trials do not become successes.
/// For risk admission, all such unresolved trials are conservatively treated as
/// potentially adverse. A physical-safe success is the only success numerator.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyCompetenceRecord {
    pub identity: RouterEvidenceIdentity,
    pub split: BenchmarkSplit,
    pub group: StageTypeGroup,
    pub policy_id: String,
    pub policy_digest: String,
    pub planned_trials: u32,
    pub physical_safe_successes: u32,
    pub successful_but_unsafe: u32,
    pub task_failures: u32,
    pub unknown_or_contradictory: u32,
    pub invalid_rows: u32,
    pub p95_inference_latency_ms: f64,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum RouterRejection {
    HoldoutSplit,
    IdentityMismatch,
    DuplicateCheckpointEvidence,
    InvalidPolicyIdentity,
    InvalidCounts,
    InsufficientKnownCoverage,
    AdverseRiskBudgetExceeded,
    PhysicalSafeSuccessLowerBoundTooLow,
    LatencyBudgetExceeded,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AbstainReason {
    NoPredictedContext,
    ContextConfidenceTooLow,
    NoCompatibleTrainingEvidence,
    NoPolicyMeetsEvidenceAndRiskBudgets,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectedPolicy {
    pub policy_id: String,
    pub policy_digest: String,
    pub physical_safe_successes: u32,
    pub planned_trials: u32,
    pub known_coverage: f64,
    pub physical_safe_success_lower_bound: f64,
    pub adverse_risk_upper_bound: f64,
    pub p95_inference_latency_ms: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RouterDecision {
    /// Provenance and predicted context are returned with every decision receipt.
    pub evidence_identity: RouterEvidenceIdentity,
    pub predicted_context: PredictedPolicyContext,
    pub selected: Option<SelectedPolicy>,
    pub abstain_reason: Option<AbstainReason>,
    pub compatible_records_considered: usize,
    pub eligible_policy_count: usize,
    pub rejected_records: BTreeMap<RouterRejection, usize>,
}

/// Wilson score interval for a binomial proportion. Returns None for invalid
/// counts, invalid z scores, or zero trials.
pub fn wilson_bounds(successes: u32, trials: u32, z_score: f64) -> Option<(f64, f64)> {
    if trials == 0
        || successes > trials
        || !z_score.is_finite()
        || z_score <= 0.0
    {
        return None;
    }

    let n = f64::from(trials);
    let successes = f64::from(successes);
    let z_squared = z_score * z_score;
    let observed = successes / n;
    let denominator = 1.0 + z_squared / n;
    let center = (observed + z_squared / (2.0 * n)) / denominator;
    let variance = observed * (1.0 - observed) / n
        + z_squared / (4.0 * n * n);
    let margin = z_score * variance.sqrt() / denominator;

    Some(((center - margin).max(0.0), (center + margin).min(1.0)))
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn config_is_valid(config: &RouterConfig) -> bool {
    config.minimum_planned_trials > 0
        && config.minimum_known_coverage.is_finite()
        && (0.0..=1.0).contains(&config.minimum_known_coverage)
        && config.minimum_physical_safe_success_lower_bound.is_finite()
        && (0.0..=1.0).contains(&config.minimum_physical_safe_success_lower_bound)
        && config.maximum_adverse_risk_upper_bound.is_finite()
        && (0.0..=1.0).contains(&config.maximum_adverse_risk_upper_bound)
        && config.maximum_p95_latency_ms.is_finite()
        && config.maximum_p95_latency_ms > 0.0
        && config.minimum_context_confidence.is_finite()
        && (0.0..=1.0).contains(&config.minimum_context_confidence)
        && config.wilson_z_score.is_finite()
        && config.wilson_z_score > 0.0
}

fn reject(
    rejected: &mut BTreeMap<RouterRejection, usize>,
    reason: RouterRejection,
) {
    *rejected.entry(reason).or_insert(0) += 1;
}

/// Select one policy using training-only competence evidence.
///
/// The ordering is lexicographic, not an opaque weighted score:
/// 1. highest Wilson lower bound on physical-safe completion;
/// 2. lowest Wilson upper bound on adverse/unknown risk;
/// 3. lowest p95 inference latency;
/// 4. stable lexical policy ID tie-break.
///
/// Unsafe successes are never successes. Unknown, contradictory, invalid and
/// missing outcomes are included in the adverse-risk upper-bound numerator.
pub fn select_policy(
    context: &PredictedPolicyContext,
    identity: &RouterEvidenceIdentity,
    records: &[PolicyCompetenceRecord],
    config: &RouterConfig,
) -> Result<RouterDecision, &'static str> {
    if !config_is_valid(config) {
        return Err("invalid_router_config");
    }
    if identity.source_repository.trim().is_empty()
        || !valid_hex(&identity.source_revision, 40)
        || !valid_hex(&identity.train_split_manifest_blob_sha, 40)
        || !valid_hex(&identity.evaluator_digest, 64)
        || !valid_hex(&identity.competence_table_digest, 64)
        || !valid_hex(&identity.crossfit_partition_digest, 64)
        || !valid_hex(&identity.robot_embodiment_digest, 64)
        || !valid_hex(&identity.observation_action_schema_digest, 64)
        || !valid_hex(&identity.context_encoder_digest, 64)
        || !valid_hex(&identity.runtime_environment_digest, 64)
    {
        return Err("invalid_router_evidence_identity");
    }

    let Some(group) = context.group else {
        return Ok(RouterDecision {
            evidence_identity: identity.clone(),
            predicted_context: context.clone(),
            selected: None,
            abstain_reason: Some(AbstainReason::NoPredictedContext),
            compatible_records_considered: 0,
            eligible_policy_count: 0,
            rejected_records: BTreeMap::new(),
        });
    };

    if !context.confidence.is_finite()
        || context.confidence < config.minimum_context_confidence
        || context.confidence > 1.0
    {
        return Ok(RouterDecision {
            evidence_identity: identity.clone(),
            predicted_context: context.clone(),
            selected: None,
            abstain_reason: Some(AbstainReason::ContextConfidenceTooLow),
            compatible_records_considered: 0,
            eligible_policy_count: 0,
            rejected_records: BTreeMap::new(),
        });
    }

    let mut rejected = BTreeMap::new();
    let matching_records = records
        .iter()
        .filter(|record| record.group == group)
        .collect::<Vec<_>>();
    let mut checkpoint_counts = BTreeMap::<String, usize>::new();
    for record in &matching_records {
        *checkpoint_counts
            .entry(record.policy_digest.clone())
            .or_insert(0) += 1;
    }
    let duplicate_checkpoints = checkpoint_counts
        .iter()
        .filter_map(|(digest, count)| (*count > 1).then(|| digest.clone()))
        .collect::<BTreeSet<_>>();

    let mut compatible_records_considered = 0;
    let mut eligible = Vec::<SelectedPolicy>::new();

    for record in matching_records {
        compatible_records_considered += 1;

        if duplicate_checkpoints.contains(&record.policy_digest) {
            reject(&mut rejected, RouterRejection::DuplicateCheckpointEvidence);
            continue;
        }

        if record.split != BenchmarkSplit::Train {
            reject(&mut rejected, RouterRejection::HoldoutSplit);
            continue;
        }

        if &record.identity != identity {
            reject(&mut rejected, RouterRejection::IdentityMismatch);
            continue;
        }

        if record.policy_id.trim().is_empty() || !valid_hex(&record.policy_digest, 64) {
            reject(&mut rejected, RouterRejection::InvalidPolicyIdentity);
            continue;
        }

        let Some(accounted) = record.physical_safe_successes
            .checked_add(record.successful_but_unsafe)
            .and_then(|count| count.checked_add(record.task_failures))
            .and_then(|count| count.checked_add(record.unknown_or_contradictory))
            .and_then(|count| count.checked_add(record.invalid_rows))
        else {
            reject(&mut rejected, RouterRejection::InvalidCounts);
            continue;
        };
        if record.planned_trials == 0
            || accounted > record.planned_trials
            || record.physical_safe_successes > accounted
        {
            reject(&mut rejected, RouterRejection::InvalidCounts);
            continue;
        }

        let missing = record.planned_trials - accounted;
        let known = record.physical_safe_successes
            + record.successful_but_unsafe
            + record.task_failures;
        let known_coverage = f64::from(known) / f64::from(record.planned_trials);
        if record.planned_trials < config.minimum_planned_trials
            || known_coverage < config.minimum_known_coverage
        {
            reject(&mut rejected, RouterRejection::InsufficientKnownCoverage);
            continue;
        }

        if !record.p95_inference_latency_ms.is_finite()
            || record.p95_inference_latency_ms <= 0.0
            || record.p95_inference_latency_ms > config.maximum_p95_latency_ms
        {
            reject(&mut rejected, RouterRejection::LatencyBudgetExceeded);
            continue;
        }

        let Some((success_lower, _)) = wilson_bounds(
            record.physical_safe_successes,
            record.planned_trials,
            config.wilson_z_score,
        ) else {
            reject(&mut rejected, RouterRejection::InvalidCounts);
            continue;
        };

        let possible_adverse = record.successful_but_unsafe
            + record.unknown_or_contradictory
            + record.invalid_rows
            + missing;
        let Some((_, adverse_upper)) = wilson_bounds(
            possible_adverse,
            record.planned_trials,
            config.wilson_z_score,
        ) else {
            reject(&mut rejected, RouterRejection::InvalidCounts);
            continue;
        };

        if success_lower < config.minimum_physical_safe_success_lower_bound {
            reject(&mut rejected, RouterRejection::PhysicalSafeSuccessLowerBoundTooLow);
            continue;
        }
        if adverse_upper > config.maximum_adverse_risk_upper_bound {
            reject(&mut rejected, RouterRejection::AdverseRiskBudgetExceeded);
            continue;
        }

        eligible.push(SelectedPolicy {
            policy_id: record.policy_id.clone(),
            policy_digest: record.policy_digest.clone(),
            physical_safe_successes: record.physical_safe_successes,
            planned_trials: record.planned_trials,
            known_coverage,
            physical_safe_success_lower_bound: success_lower,
            adverse_risk_upper_bound: adverse_upper,
            p95_inference_latency_ms: record.p95_inference_latency_ms,
        });
    }

    eligible.sort_by(|left, right| {
        right
            .physical_safe_success_lower_bound
            .total_cmp(&left.physical_safe_success_lower_bound)
            .then_with(|| {
                left.adverse_risk_upper_bound
                    .total_cmp(&right.adverse_risk_upper_bound)
            })
            .then_with(|| {
                left.p95_inference_latency_ms
                    .total_cmp(&right.p95_inference_latency_ms)
            })
            .then_with(|| left.policy_id.cmp(&right.policy_id))
            .then_with(|| left.policy_digest.cmp(&right.policy_digest))
    });

    let selected = eligible.first().cloned();
    let abstain_reason = if selected.is_some() {
        None
    } else if compatible_records_considered == 0 {
        Some(AbstainReason::NoCompatibleTrainingEvidence)
    } else {
        Some(AbstainReason::NoPolicyMeetsEvidenceAndRiskBudgets)
    };

    Ok(RouterDecision {
        evidence_identity: identity.clone(),
        predicted_context: context.clone(),
        selected,
        abstain_reason,
        compatible_records_considered,
        eligible_policy_count: eligible.len(),
        rejected_records: rejected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roborecover_score::{
        BenchmarkPlatform, DeviationType, RecoveryStage, StageTypeGroup,
    };

    fn group() -> StageTypeGroup {
        StageTypeGroup {
            stage: RecoveryStage::Approach,
            deviation: DeviationType::WrongPose,
        }
    }

    fn identity() -> RouterEvidenceIdentity {
        RouterEvidenceIdentity {
            platform: BenchmarkPlatform::RoboTwin,
            source_repository: "RUCKBReasoning/RoboRecover".to_owned(),
            source_revision: "a".repeat(40),
            train_split_manifest_blob_sha: "b".repeat(40),
            evaluator_digest: "c".repeat(64),
            competence_table_digest: "d".repeat(64),
            crossfit_partition_digest: "e".repeat(64),
            robot_embodiment_digest: "1".repeat(64),
            observation_action_schema_digest: "2".repeat(64),
            context_encoder_digest: "3".repeat(64),
            runtime_environment_digest: "4".repeat(64),
        }
    }

    fn record(policy_id: &str, successes: u32, unsafe_successes: u32) -> PolicyCompetenceRecord {
        PolicyCompetenceRecord {
            identity: identity(),
            split: BenchmarkSplit::Train,
            group: group(),
            policy_id: policy_id.to_owned(),
            policy_digest: "f".repeat(64),
            planned_trials: 100,
            physical_safe_successes: successes,
            successful_but_unsafe: unsafe_successes,
            task_failures: 100 - successes - unsafe_successes,
            unknown_or_contradictory: 0,
            invalid_rows: 0,
            p95_inference_latency_ms: 250.0,
        }
    }

    fn context() -> PredictedPolicyContext {
        PredictedPolicyContext {
            group: Some(group()),
            confidence: 0.95,
        }
    }

    #[test]
    fn wilson_bounds_are_ordered_and_include_observed_rate() {
        let (lower, upper) = wilson_bounds(80, 100, 1.96).unwrap();
        assert!(lower < 0.80);
        assert!(upper > 0.80);
        assert!(lower < upper);
    }

    #[test]
    fn selects_highest_safe_success_lower_bound_not_raw_success_rate() {
        let records = vec![
            record("policy-b", 72, 0),
            record("policy-a", 82, 0),
        ];
        let decision = select_policy(&context(), &identity(), &records, &RouterConfig::default())
            .unwrap();

        assert_eq!(decision.selected.unwrap().policy_id, "policy-a");
        assert_eq!(decision.eligible_policy_count, 2);
        assert_eq!(decision.abstain_reason, None);
    }

    #[test]
    fn held_out_competence_records_are_never_routed() {
        let mut holdout = record("test-policy", 99, 0);
        holdout.split = BenchmarkSplit::Test;
        let decision = select_policy(
            &context(),
            &identity(),
            &[holdout],
            &RouterConfig::default(),
        )
        .unwrap();

        assert!(decision.selected.is_none());
        assert_eq!(
            decision.rejected_records.get(&RouterRejection::HoldoutSplit),
            Some(&1)
        );
    }

    #[test]
    fn uncertain_context_abstains_before_policy_selection() {
        let mut predicted = context();
        predicted.confidence = 0.40;
        let decision = select_policy(
            &predicted,
            &identity(),
            &[record("policy-a", 99, 0)],
            &RouterConfig::default(),
        )
        .unwrap();

        assert_eq!(decision.selected, None);
        assert_eq!(
            decision.abstain_reason,
            Some(AbstainReason::ContextConfidenceTooLow)
        );
    }

    #[test]
    fn missing_predicted_group_abstains() {
        let mut predicted = context();
        predicted.group = None;
        let decision = select_policy(
            &predicted,
            &identity(),
            &[record("policy-a", 99, 0)],
            &RouterConfig::default(),
        )
        .unwrap();

        assert_eq!(decision.abstain_reason, Some(AbstainReason::NoPredictedContext));
    }

    #[test]
    fn unsafe_successes_count_against_the_adverse_risk_budget() {
        let records = vec![record("unsafe-policy", 80, 20)];
        let decision = select_policy(
            &context(),
            &identity(),
            &records,
            &RouterConfig::default(),
        )
        .unwrap();

        assert!(decision.selected.is_none());
        assert_eq!(
            decision.rejected_records.get(&RouterRejection::AdverseRiskBudgetExceeded),
            Some(&1)
        );
    }

    #[test]
    fn unknown_missing_and_invalid_trials_are_treated_as_possible_adverse() {
        let mut candidate = record("uncertain-policy", 90, 0);
        candidate.unknown_or_contradictory = 5;
        candidate.task_failures = 0;
        // Five planned trials are unaccounted/missing.
        let decision = select_policy(
            &context(),
            &identity(),
            &[candidate],
            &RouterConfig::default(),
        )
        .unwrap();

        assert!(decision.selected.is_none());
        assert_eq!(
            decision.rejected_records.get(&RouterRejection::AdverseRiskBudgetExceeded),
            Some(&1)
        );
    }

    #[test]
    fn provenance_or_crossfit_identity_mismatch_is_rejected() {
        let mut candidate = record("wrong-fold", 99, 0);
        candidate.identity.crossfit_partition_digest = "0".repeat(64);
        let decision = select_policy(
            &context(),
            &identity(),
            &[candidate],
            &RouterConfig::default(),
        )
        .unwrap();

        assert!(decision.selected.is_none());
        assert_eq!(
            decision.rejected_records.get(&RouterRejection::IdentityMismatch),
            Some(&1)
        );
    }

    #[test]
    fn latency_budget_is_a_hard_constraint() {
        let mut candidate = record("slow-policy", 99, 0);
        candidate.p95_inference_latency_ms = 5_000.0;
        let decision = select_policy(
            &context(),
            &identity(),
            &[candidate],
            &RouterConfig::default(),
        )
        .unwrap();

        assert!(decision.selected.is_none());
        assert_eq!(
            decision.rejected_records.get(&RouterRejection::LatencyBudgetExceeded),
            Some(&1)
        );
    }

    #[test]
    fn insufficient_training_support_abstains() {
        let mut candidate = record("small-sample", 9, 0);
        candidate.planned_trials = 10;
        candidate.task_failures = 1;
        let decision = select_policy(
            &context(),
            &identity(),
            &[candidate],
            &RouterConfig::default(),
        )
        .unwrap();

        assert!(decision.selected.is_none());
        assert_eq!(
            decision.rejected_records.get(&RouterRejection::InsufficientKnownCoverage),
            Some(&1)
        );
    }

    #[test]
    fn tie_break_is_deterministic_by_policy_id() {
        let records = vec![
            record("policy-z", 90, 0),
            record("policy-a", 90, 0),
        ];
        let decision = select_policy(
            &context(),
            &identity(),
            &records,
            &RouterConfig::default(),
        )
        .unwrap();

        assert_eq!(decision.selected.unwrap().policy_id, "policy-a");
    }

    #[test]
    fn malformed_counts_and_policy_digests_are_rejected() {
        let mut bad_counts = record("over-counted", 99, 0);
        bad_counts.task_failures = 99;
        let mut bad_digest = record("bad-digest", 99, 0);
        bad_digest.policy_digest = "latest".to_owned();
        let decision = select_policy(
            &context(),
            &identity(),
            &[bad_counts, bad_digest],
            &RouterConfig::default(),
        )
        .unwrap();

        assert_eq!(decision.selected, None);
        assert_eq!(
            decision.rejected_records.get(&RouterRejection::InvalidCounts),
            Some(&1)
        );
        assert_eq!(
            decision.rejected_records.get(&RouterRejection::InvalidPolicyIdentity),
            Some(&1)
        );
    }

    #[test]
    fn unknown_stage_type_records_are_not_considered_for_other_predicted_groups() {
        let mut other = record("other-group", 99, 0);
        other.group = StageTypeGroup {
            stage: RecoveryStage::Place,
            deviation: DeviationType::SceneChange,
        };
        let decision = select_policy(
            &context(),
            &identity(),
            &[other],
            &RouterConfig::default(),
        )
        .unwrap();

        assert_eq!(decision.compatible_records_considered, 0);
        assert_eq!(
            decision.abstain_reason,
            Some(AbstainReason::NoCompatibleTrainingEvidence)
        );
    }

    #[test]
    fn duplicate_checkpoint_evidence_is_not_counted_twice() {
        let first = record("policy-a", 99, 0);
        let mut second = record("policy-a-copy", 95, 0);
        second.policy_digest = first.policy_digest.clone();
        let decision = select_policy(
            &context(),
            &identity(),
            &[first, second],
            &RouterConfig::default(),
        ).unwrap();

        assert_eq!(decision.selected, None);
        assert_eq!(
            decision.rejected_records.get(&RouterRejection::DuplicateCheckpointEvidence),
            Some(&2)
        );
    }


    #[test]
    fn cross_platform_competence_cannot_be_mixed() {
        let mut candidate = record("libero-policy", 99, 0);
        candidate.identity.platform = BenchmarkPlatform::Libero;
        let decision = select_policy(
            &context(),
            &identity(),
            &[candidate],
            &RouterConfig::default(),
        )
        .unwrap();

        assert!(decision.selected.is_none());
        assert_eq!(
            decision.rejected_records.get(&RouterRejection::IdentityMismatch),
            Some(&1)
        );
    }


    #[test]
    fn low_safe_recovery_lower_bound_abstains_even_when_risk_is_low() {
        let candidate = record("weak-policy", 0, 0);
        let decision = select_policy(
            &context(),
            &identity(),
            &[candidate],
            &RouterConfig::default(),
        )
        .unwrap();

        assert!(decision.selected.is_none());
        assert_eq!(
            decision.rejected_records.get(
                &RouterRejection::PhysicalSafeSuccessLowerBoundTooLow
            ),
            Some(&1)
        );
    }

    #[test]
    fn embodiment_schema_context_encoder_and_runtime_are_part_of_identity() {
        let mutations: [fn(&mut RouterEvidenceIdentity); 4] = [
            |identity| identity.robot_embodiment_digest = "9".repeat(64),
            |identity| identity.observation_action_schema_digest = "8".repeat(64),
            |identity| identity.context_encoder_digest = "7".repeat(64),
            |identity| identity.runtime_environment_digest = "6".repeat(64),
        ];

        for mutate in mutations {
            let expected = identity();
            let mut supplied = identity();
            mutate(&mut supplied);

            let decision = select_policy(
                &context(),
                &supplied,
                &[record("policy", 99, 0)],
                &RouterConfig::default(),
            )
            .unwrap();

            assert!(decision.selected.is_none());
            assert_eq!(
                decision.rejected_records.get(&RouterRejection::IdentityMismatch),
                Some(&1)
            );
            assert_eq!(decision.evidence_identity, supplied);
            assert_ne!(decision.evidence_identity, expected);
        }
    }

    #[test]
    fn overflowing_outcome_counts_are_rejected_instead_of_saturating() {
        let mut candidate = record("overflow", 1, 0);
        candidate.planned_trials = u32::MAX;
        candidate.physical_safe_successes = u32::MAX;
        candidate.successful_but_unsafe = u32::MAX;
        candidate.task_failures = u32::MAX;
        candidate.unknown_or_contradictory = u32::MAX;
        candidate.invalid_rows = u32::MAX;

        let decision = select_policy(
            &context(),
            &identity(),
            &[candidate],
            &RouterConfig::default(),
        )
        .unwrap();
        assert!(decision.selected.is_none());
        assert_eq!(
            decision.rejected_records.get(&RouterRejection::InvalidCounts),
            Some(&1)
        );
    }

    #[test]
    fn router_decision_carries_source_and_predicted_context_receipt() {
        let identity = identity();
        let context = context();
        let decision = select_policy(
            &context,
            &identity,
            &[record("policy-a", 99, 0)],
            &RouterConfig::default(),
        )
        .unwrap();

        assert_eq!(decision.evidence_identity, identity);
        assert_eq!(decision.predicted_context, context);
    }

    #[test]
    fn minimum_success_bound_configuration_is_validated() {
        let mut config = RouterConfig::default();
        config.minimum_physical_safe_success_lower_bound = f64::NAN;
        assert_eq!(
            select_policy(&context(), &identity(), &[], &config),
            Err("invalid_router_config")
        );
    }

}
