#![forbid(unsafe_code)]

//! Proof-progress-per-computation benchmark.
//!
//! The planner sees candidate operation metadata (target claims, estimated
//! gain, cost, and mandatory status). The evaluator-only support mask is held
//! separately and is revealed only after an operation is selected. This is a
//! deterministic synthetic causal benchmark, not a production scheduler or
//! a claim of state-of-the-art performance.

use std::cmp::Ordering;

pub const TEMPORAL: u8 = 0b0001;
pub const INDEPENDENCE: u8 = 0b0010;
pub const AUTHORITY: u8 = 0b0100;
pub const PROOF_PATH: u8 = 0b1000;
pub const REQUIRED: u8 = TEMPORAL | INDEPENDENCE | AUTHORITY | PROOF_PATH;
pub const REQUIRED_MANDATORY: u8 = AUTHORITY | PROOF_PATH;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    FixedDepth,
    GenericVoc,
    BudgetMatchedRandom,
    ProofObligationGuided,
    UnverifiedConfidence,
    OracleMinimumCost,
}

impl Policy {
    pub const ALL: [Self; 6] = [
        Self::FixedDepth,
        Self::GenericVoc,
        Self::BudgetMatchedRandom,
        Self::ProofObligationGuided,
        Self::UnverifiedConfidence,
        Self::OracleMinimumCost,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FixedDepth => "fixed_depth",
            Self::GenericVoc => "generic_voc",
            Self::BudgetMatchedRandom => "budget_matched_random",
            Self::ProofObligationGuided => "proof_obligation_guided",
            Self::UnverifiedConfidence => "unverified_confidence",
            Self::OracleMinimumCost => "oracle_minimum_cost",
        }
    }

    pub const fn promotion_candidate(self) -> bool {
        !matches!(self, Self::UnverifiedConfidence | Self::OracleMinimumCost)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationSpec {
    pub id: &'static str,
    /// Claims the operation may be able to resolve; planner-visible.
    pub advertised_mask: u8,
    /// Estimated usefulness on a generic, pre-outcome scale.
    pub predicted_gain: u32,
    /// Abstract work units, not FLOPs or wall-clock time.
    pub cost: u32,
    /// Mandatory safety/proof check; its cost must be reserved.
    pub mandatory_mask: u8,
}

#[derive(Debug, Clone)]
struct EvaluatorOperation {
    public: OperationSpec,
    /// Evaluator-only truth: claims independently verified after execution.
    verified_mask: u8,
}

#[derive(Debug, Clone)]
pub struct TaskCase {
    pub id: &'static str,
    pub required_mask: u8,
    operations: Vec<EvaluatorOperation>,
}

impl TaskCase {
    pub fn operation_catalogue(&self) -> Vec<OperationSpec> {
        self.operations.iter().map(|o| o.public.clone()).collect()
    }

    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    fn mandatory_mask(&self) -> u8 {
        self.operations
            .iter()
            .fold(0, |mask, op| mask | op.public.mandatory_mask)
            & self.required_mask
    }

    fn mandatory_cost(&self, attempted: &[bool]) -> u32 {
        self.operations
            .iter()
            .enumerate()
            .filter(|(i, op)| op.public.mandatory_mask != 0 && !attempted[*i])
            .map(|(_, op)| op.public.cost)
            .sum()
    }
}

fn op(
    id: &'static str,
    advertised_mask: u8,
    verified_mask: u8,
    cost: u32,
    predicted_gain: u32,
    mandatory_mask: u8,
) -> EvaluatorOperation {
    EvaluatorOperation {
        public: OperationSpec {
            id,
            advertised_mask,
            predicted_gain,
            cost,
            mandatory_mask,
        },
        verified_mask,
    }
}

pub fn build_cases() -> Vec<TaskCase> {
    use crate::{AUTHORITY as A, INDEPENDENCE as I, PROOF_PATH as P, TEMPORAL as T};

    vec![
        TaskCase {
            id: "redundant_broad_retrieval",
            required_mask: REQUIRED,
            operations: vec![
                op("broad_retrieval", REQUIRED, T, 5, 100, 0),
                op("deep_reasoning", 0, 0, 4, 95, 0),
                op("temporal_lookup", T, T, 2, 30, 0),
                op("independent_source", I, I, 3, 25, 0),
                op("authority_check", A, A, 1, 20, A),
                op("proof_verifier", P, P, 1, 20, P),
                op("correlated_duplicate", T | I, 0, 2, 28, 0),
                op("evidence_refresh", T, T, 4, 12, 0),
                op("alternative_source", I, I, 5, 10, 0),
                op("extra_context", 0, 0, 1, 10, 0),
            ],
        },
        TaskCase {
            id: "valid_multi_obligation_packet",
            required_mask: REQUIRED,
            operations: vec![
                op("combined_temporal_independent", T | I, T | I, 3, 55, 0),
                op("temporal_lookup", T, T, 3, 25, 0),
                op("independent_source", I, I, 3, 25, 0),
                op("authority_check", A, A, 1, 20, A),
                op("proof_verifier", P, P, 1, 20, P),
                op("deep_reasoning", 0, 0, 4, 80, 0),
                op("invalid_broad_packet", REQUIRED, T, 1, 60, 0),
                op("extra_context", 0, 0, 1, 15, 0),
            ],
        },
        TaskCase {
            id: "attractive_but_stale_evidence",
            required_mask: REQUIRED,
            operations: vec![
                op("freshness_claim", T, 0, 1, 90, 0),
                op("historical_temporal_check", T, T, 4, 20, 0),
                op("independent_source", I, I, 2, 20, 0),
                op("authority_check", A, A, 1, 15, A),
                op("proof_verifier", P, P, 1, 15, P),
                op("broad_retrieval", REQUIRED, T, 3, 65, 0),
                op("deep_reasoning", 0, 0, 3, 75, 0),
                op("context_expansion", 0, 0, 2, 25, 0),
            ],
        },
        TaskCase {
            id: "correlated_source_trap",
            required_mask: REQUIRED,
            operations: vec![
                op("two_copied_sources", I, 0, 1, 85, 0),
                op("independence_audit", I, I, 4, 20, 0),
                op("temporal_lookup", T, T, 2, 20, 0),
                op("authority_check", A, A, 1, 15, A),
                op("proof_verifier", P, P, 1, 15, P),
                op("broad_retrieval", REQUIRED, T | I, 4, 70, 0),
                op("source_comparison", I, I, 3, 30, 0),
                op("extra_reasoning", 0, 0, 2, 40, 0),
            ],
        },
        TaskCase {
            id: "minimum_sufficient_combined_packet",
            required_mask: REQUIRED,
            operations: vec![
                op("complete_verified_packet", REQUIRED, REQUIRED, 6, 70, REQUIRED_MANDATORY),
                op("temporal_lookup", T, T, 3, 20, 0),
                op("independent_source", I, I, 3, 20, 0),
                op("authority_check", A, A, 1, 15, 0),
                op("proof_verifier", P, P, 1, 15, 0),
                op("generic_reflection", 0, 0, 4, 90, 0),
                op("redundant_packet", T | I, 0, 1, 30, 0),
                op("extra_context", 0, 0, 1, 10, 0),
            ],
        },
        TaskCase {
            id: "retrieval_noise_and_supersession",
            required_mask: REQUIRED,
            operations: vec![
                op("superseded_document", REQUIRED, 0, 1, 100, 0),
                op("current_temporal_source", T, T, 3, 20, 0),
                op("independent_source", I, I, 2, 20, 0),
                op("authority_check", A, A, 1, 15, A),
                op("proof_verifier", P, P, 1, 15, P),
                op("wide_context_search", T | I, T, 4, 55, 0),
                op("independent_alternative", I, I, 4, 10, 0),
                op("extra_reasoning", 0, 0, 2, 35, 0),
            ],
        },
        TaskCase {
            id: "operation_family_shift",
            required_mask: REQUIRED,
            operations: vec![
                op("old_temporal_tool", T, 0, 1, 95, 0),
                op("new_temporal_tool", T, T, 5, 35, 0),
                op("family_specific_independence", I, I, 4, 20, 0),
                op("authority_check", A, A, 1, 15, A),
                op("proof_verifier", P, P, 1, 15, P),
                op("broad_reasoner", REQUIRED, T | I, 6, 80, 0),
                op("extra_reasoning", 0, 0, 2, 30, 0),
                op("clarification", T, T, 7, 12, 0),
            ],
        },
        TaskCase {
            id: "high_redundancy_long_catalogue",
            required_mask: REQUIRED,
            operations: vec![
                op("context_1", 0, 0, 1, 40, 0),
                op("context_2", 0, 0, 2, 45, 0),
                op("context_3", 0, 0, 2, 50, 0),
                op("weak_temporal", T, 0, 1, 80, 0),
                op("temporal_lookup", T, T, 3, 25, 0),
                op("source_retrieval", I, I, 3, 25, 0),
                op("authority_check", A, A, 1, 20, A),
                op("proof_verifier", P, P, 1, 20, P),
                op("duplicate_independent", I, 0, 1, 60, 0),
                op("alternative_source", I, I, 5, 15, 0),
                op("long_reflection", 0, 0, 5, 100, 0),
                op("all_claim_retrieval", REQUIRED, T | I, 4, 65, 0),
            ],
        },
    ]
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrialResult {
    pub task_id: &'static str,
    pub policy: Policy,
    pub budget: u32,
    pub spent: u32,
    pub operation_count: usize,
    pub verified_obligations: u32,
    pub required_obligations: u32,
    pub reported_complete: bool,
    pub completion_verified: bool,
    pub false_completion: bool,
    pub false_support_count: u32,
    pub mandatory_required_mask: u8,
    pub mandatory_verified_mask: u8,
    pub missed_mandatory_count: u32,
    pub redundant_operation_count: u32,
    pub wasted_operation_cost: u32,
    pub correct_abstention: bool,
    pub cost_to_first_sufficient_proof: Option<u32>,
    pub oracle_min_cost: Option<u32>,
    pub cost_regret: Option<i64>,
}

fn popcount(mask: u8) -> u32 {
    mask.count_ones()
}

fn ratio_cmp(a_gain: u32, a_cost: u32, b_gain: u32, b_cost: u32) -> Ordering {
    let left = u64::from(a_gain) * u64::from(b_cost.max(1));
    let right = u64::from(b_gain) * u64::from(a_cost.max(1));
    left.cmp(&right)
}

fn oracle_subset(case: &TaskCase) -> Option<(u64, u32)> {
    if case.operations.len() > 20 {
        return None;
    }

    let n = case.operations.len();
    let mandatory_indices: Vec<usize> = case.operations.iter().enumerate()
        .filter_map(|(i, op)| (op.public.mandatory_mask != 0).then_some(i))
        .collect();
    let mandatory_mask = case.mandatory_mask();
    let mut best: Option<(u64, u32, u32)> = None;

    for subset in 0u64..(1u64 << n) {
        if mandatory_indices.iter().any(|i| subset & (1u64 << i) == 0) {
            continue;
        }
        let mut cost = 0u32;
        let mut covered = 0u8;
        let mut count = 0u32;
        for (i, operation) in case.operations.iter().enumerate() {
            if subset & (1u64 << i) != 0 {
                cost = cost.saturating_add(operation.public.cost);
                covered |= operation.verified_mask;
                count += 1;
            }
        }
        if covered & case.required_mask != case.required_mask
            || covered & mandatory_mask != mandatory_mask
        {
            continue;
        }
        let replace = match best {
            None => true,
            Some((_, best_cost, best_count)) => {
                cost < best_cost || (cost == best_cost && count < best_count)
            }
        };
        if replace {
            best = Some((subset, cost, count));
        }
    }

    best.map(|(subset, cost, _)| (subset, cost))
}

fn choose_next(
    case: &TaskCase,
    policy: Policy,
    attempted: &[bool],
    unresolved: u8,
    spent: u32,
    budget: u32,
    seed: u64,
) -> Option<usize> {
    let remaining = budget.saturating_sub(spent);
    let affordable: Vec<usize> = case.operations.iter().enumerate()
        .filter_map(|(i, op)| (!attempted[i] && op.public.cost <= remaining).then_some(i))
        .collect();
    if affordable.is_empty() {
        return None;
    }

    match policy {
        Policy::FixedDepth => affordable.into_iter().next(),
        Policy::GenericVoc | Policy::UnverifiedConfidence => {
            affordable.into_iter().max_by(|a, b| {
                let oa = &case.operations[*a].public;
                let ob = &case.operations[*b].public;
                ratio_cmp(oa.predicted_gain, oa.cost, ob.predicted_gain, ob.cost)
                    .then_with(|| ob.cost.cmp(&oa.cost))
                    .then_with(|| ob.id.cmp(oa.id))
            })
        }
        Policy::BudgetMatchedRandom => {
            let mut state = seed
                .wrapping_add(u64::from(spent).wrapping_mul(6364136223846793005))
                .wrapping_add(attempted.iter().filter(|x| **x).count() as u64);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            Some(affordable[(state as usize) % affordable.len()])
        }
        Policy::ProofObligationGuided => {
            let mandatory_pending: Vec<usize> = affordable.iter().copied()
                .filter(|i| case.operations[*i].public.mandatory_mask != 0)
                .collect();
            if !mandatory_pending.is_empty() {
                let total_mandatory = case.mandatory_cost(attempted);
                if total_mandatory > remaining {
                    return None;
                }
                return mandatory_pending.into_iter().min_by(|a, b| {
                    case.operations[*a].public.cost.cmp(&case.operations[*b].public.cost)
                        .then_with(|| case.operations[*a].public.id.cmp(case.operations[*b].public.id))
                });
            }

            affordable.into_iter()
                .filter(|i| case.operations[*i].public.advertised_mask & unresolved != 0)
                .max_by(|a, b| {
                    let oa = &case.operations[*a].public;
                    let ob = &case.operations[*b].public;
                    let ga = popcount(oa.advertised_mask & unresolved);
                    let gb = popcount(ob.advertised_mask & unresolved);
                    ratio_cmp(ga, oa.cost, gb, ob.cost)
                        .then_with(|| oa.predicted_gain.cmp(&ob.predicted_gain))
                        .then_with(|| ob.cost.cmp(&oa.cost))
                        .then_with(|| ob.id.cmp(oa.id))
                })
        }
        Policy::OracleMinimumCost => {
            let (subset, _) = oracle_subset(case)?;
            affordable.into_iter()
                .filter(|i| subset & (1u64 << i) != 0)
                .min_by(|a, b| {
                    case.operations[*a].public.cost.cmp(&case.operations[*b].public.cost)
                        .then_with(|| case.operations[*a].public.id.cmp(case.operations[*b].public.id))
                })
        }
    }
}

pub fn run_trial(case: &TaskCase, policy: Policy, budget: u32, seed: u64) -> TrialResult {
    let mut attempted = vec![false; case.operations.len()];
    let mut spent = 0u32;
    let mut verified_mask = 0u8;
    let mut claimed_mask = 0u8;
    let mut mandatory_verified_mask = 0u8;
    let mut operations = 0usize;
    let mut redundant_operation_count = 0u32;
    let mut wasted_operation_cost = 0u32;
    let mut cost_to_first_sufficient_proof = None;

    loop {
        let unresolved = case.required_mask & !verified_mask;
        let complete = unresolved == 0
            && mandatory_verified_mask & case.mandatory_mask() == case.mandatory_mask();
        if complete {
            cost_to_first_sufficient_proof = Some(spent);
            break;
        }

        let Some(index) = choose_next(case, policy, &attempted, unresolved, spent, budget, seed) else {
            break;
        };
        let operation = &case.operations[index];
        attempted[index] = true;
        operations += 1;
        spent = spent.saturating_add(operation.public.cost);

        let actual = operation.verified_mask & case.required_mask;
        let newly_verified = actual & !verified_mask;
        verified_mask |= actual;
        if operation.public.mandatory_mask != 0 {
            mandatory_verified_mask |= actual & operation.public.mandatory_mask;
        }

        if policy == Policy::UnverifiedConfidence {
            claimed_mask |= operation.public.advertised_mask & case.required_mask;
        } else {
            claimed_mask = verified_mask;
        }

        if newly_verified == 0 && operation.public.mandatory_mask == 0 {
            redundant_operation_count += 1;
            wasted_operation_cost = wasted_operation_cost.saturating_add(operation.public.cost);
        }

        if verified_mask & case.required_mask == case.required_mask
            && mandatory_verified_mask & case.mandatory_mask() == case.mandatory_mask()
            && cost_to_first_sufficient_proof.is_none()
        {
            cost_to_first_sufficient_proof = Some(spent);
        }
    }

    let completion_verified = verified_mask & case.required_mask == case.required_mask
        && mandatory_verified_mask & case.mandatory_mask() == case.mandatory_mask();
    let all_mandatory_attempted = case.operations.iter().enumerate()
        .all(|(i, op)| op.public.mandatory_mask == 0 || attempted[i]);
    let reported_complete = match policy {
        Policy::UnverifiedConfidence => {
            claimed_mask & case.required_mask == case.required_mask && all_mandatory_attempted
        }
        _ => completion_verified,
    };
    let false_support_count = popcount(claimed_mask & !verified_mask & case.required_mask);
    let false_completion = reported_complete && !completion_verified;
    let correct_abstention = !reported_complete && !completion_verified;
    let missed_mandatory_count = popcount(case.mandatory_mask() & !mandatory_verified_mask);
    let (oracle_min_cost, cost_regret) = match oracle_subset(case) {
        Some((_, minimum)) => {
            let regret = if completion_verified {
                Some(i64::from(spent) - i64::from(minimum))
            } else {
                None
            };
            (Some(minimum), regret)
        }
        None => (None, None),
    };

    TrialResult {
        task_id: case.id,
        policy,
        budget,
        spent,
        operation_count: operations,
        verified_obligations: popcount(verified_mask & case.required_mask),
        required_obligations: popcount(case.required_mask),
        reported_complete,
        completion_verified,
        false_completion,
        false_support_count,
        mandatory_required_mask: case.mandatory_mask(),
        mandatory_verified_mask,
        missed_mandatory_count,
        redundant_operation_count,
        wasted_operation_cost,
        correct_abstention,
        cost_to_first_sufficient_proof,
        oracle_min_cost,
        cost_regret,
    }
}

pub fn run_matrix(seed: u64) -> Vec<TrialResult> {
    let budgets = [3, 5, 7, 10, 14];
    let mut rows = Vec::new();
    for case in build_cases() {
        for budget in budgets {
            for policy in Policy::ALL {
                rows.push(run_trial(&case, policy, budget, seed));
            }
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proof_guided_completes_minimum_sufficient_multi_obligation_packet() {
        let case = build_cases().into_iter()
            .find(|c| c.id == "valid_multi_obligation_packet").unwrap();
        let result = run_trial(&case, Policy::ProofObligationGuided, 7, 1729);
        assert!(result.completion_verified);
        assert_eq!(result.spent, 5);
        assert_eq!(result.missed_mandatory_count, 0);
        assert_eq!(result.false_support_count, 0);
    }

    #[test]
    fn proof_guided_does_not_accept_stale_evidence_claim_as_temporal_support() {
        let case = build_cases().into_iter()
            .find(|c| c.id == "attractive_but_stale_evidence").unwrap();
        let result = run_trial(&case, Policy::ProofObligationGuided, 10, 1729);
        assert!(result.completion_verified);
        assert_eq!(result.false_support_count, 0);
        assert_eq!(result.missed_mandatory_count, 0);
    }

    #[test]
    fn unverified_confidence_control_exposes_false_support() {
        let case = build_cases().into_iter()
            .find(|c| c.id == "attractive_but_stale_evidence").unwrap();
        let result = run_trial(&case, Policy::UnverifiedConfidence, 10, 1729);
        assert!(result.false_support_count > 0 || result.false_completion);
    }

    #[test]
    fn mandatory_work_is_not_displaced_by_nonmandatory_gain() {
        let case = build_cases().into_iter()
            .find(|c| c.id == "minimum_sufficient_combined_packet").unwrap();
        let result = run_trial(&case, Policy::ProofObligationGuided, 5, 1729);
        assert_eq!(result.spent, 0);
        assert!(!result.completion_verified);
        assert_eq!(result.missed_mandatory_count, 2);
    }

    #[test]
    fn oracle_cost_is_no_greater_than_any_verified_controller_cost() {
        for case in build_cases() {
            let oracle = run_trial(&case, Policy::OracleMinimumCost, 100, 1729);
            assert!(oracle.completion_verified, "{}", case.id);
            let proof = run_trial(&case, Policy::ProofObligationGuided, 100, 1729);
            assert!(proof.completion_verified, "{}", case.id);
            assert!(oracle.spent <= proof.spent, "{}", case.id);
        }
    }

    #[test]
    fn budgets_are_never_exceeded() {
        for row in run_matrix(42) {
            assert!(row.spent <= row.budget, "{}", row.task_id);
        }
    }

    #[test]
    fn proof_aware_policy_never_false_completes() {
        for case in build_cases() {
            for budget in [3, 5, 7, 10, 14] {
                let result = run_trial(&case, Policy::ProofObligationGuided, budget, 42);
                assert!(!result.false_completion, "{}:{}", case.id, budget);
                assert_eq!(result.false_support_count, 0);
            }
        }
    }

    #[test]
    fn oracle_is_not_a_promotion_candidate() {
        assert!(!Policy::OracleMinimumCost.promotion_candidate());
        assert!(!Policy::UnverifiedConfidence.promotion_candidate());
        assert!(Policy::ProofObligationGuided.promotion_candidate());
    }

    #[test]
    fn random_policy_is_seed_reproducible() {
        let case = build_cases().remove(0);
        let a = run_trial(&case, Policy::BudgetMatchedRandom, 10, 1729);
        let b = run_trial(&case, Policy::BudgetMatchedRandom, 10, 1729);
        assert_eq!(a, b);
    }

    #[test]
    fn every_case_has_a_nonzero_mandatory_verification_contract() {
        for case in build_cases() {
            assert_eq!(case.mandatory_mask(), REQUIRED_MANDATORY, "{}", case.id);
        }
    }

    #[test]
    fn exhaustive_budget_curve_contains_low_and_high_budget_points() {
        let case = build_cases().remove(1);
        let low = run_trial(&case, Policy::ProofObligationGuided, 3, 10);
        let high = run_trial(&case, Policy::ProofObligationGuided, 14, 10);
        assert!(high.spent >= low.spent);
        assert!(high.verified_obligations >= low.verified_obligations);
    }

    #[test]
    fn redundant_operation_cost_is_reported_separately() {
        let case = build_cases().remove(0);
        let result = run_trial(&case, Policy::FixedDepth, 14, 10);
        assert!(result.redundant_operation_count > 0);
        assert!(result.wasted_operation_cost > 0);
    }
}
