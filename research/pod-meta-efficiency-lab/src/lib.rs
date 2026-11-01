#![forbid(unsafe_code)]

//! Proof-progress-per-computation benchmark.
//!
//! The planner sees candidate operation metadata (target claims, estimated
//! gain, cost, and mandatory status). The evaluator-only support mask is held
//! separately and is revealed only after an operation is selected. This is a
//! deterministic synthetic causal benchmark, not a production scheduler or
//! a claim of state-of-the-art performance.

use pod_meta_lab::{
    CandidateOperation as PodCandidateOperation, Obligation as PodObligation, ObligationState,
    OperationKind, SchedulingClass, Selection as PodSelection, SelectionContext,
    choose_next as pod_meta_choose_next,
};
use std::cmp::Ordering;
use std::collections::BTreeSet;

pub const TEMPORAL: u8 = 0b0001;
pub const INDEPENDENCE: u8 = 0b0010;
pub const AUTHORITY: u8 = 0b0100;
pub const PROOF_PATH: u8 = 0b1000;
pub const REQUIRED: u8 = TEMPORAL | INDEPENDENCE | AUTHORITY | PROOF_PATH;
pub const REQUIRED_MANDATORY: u8 = AUTHORITY | PROOF_PATH;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    FixedOrder,
    GenericVoc,
    BudgetMatchedRandom,
    PodMetaRouter,
    MaskCoverageGreedy,
    UnverifiedConfidence,
    OracleMinimumCost,
}

impl Policy {
    pub const ALL: [Self; 7] = [
        Self::FixedOrder,
        Self::GenericVoc,
        Self::BudgetMatchedRandom,
        Self::PodMetaRouter,
        Self::MaskCoverageGreedy,
        Self::UnverifiedConfidence,
        Self::OracleMinimumCost,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FixedOrder => "fixed_order",
            Self::GenericVoc => "generic_voc",
            Self::BudgetMatchedRandom => "budget_matched_random",
            Self::PodMetaRouter => "pod_meta_router",
            Self::MaskCoverageGreedy => "mask_coverage_greedy",
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
                op(
                    "complete_verified_packet",
                    REQUIRED,
                    REQUIRED,
                    6,
                    70,
                    REQUIRED_MANDATORY,
                ),
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
    pub seed: u64,
    pub policy: Policy,
    pub budget: u32,
    pub spent: u32,
    pub operation_count: usize,
    pub scheduler_candidate_rejections: u32,
    pub verified_obligations: u32,
    pub required_obligations: u32,
    pub reported_complete: bool,
    pub completion_verified: bool,
    pub false_completion: bool,
    pub false_support_count: u32,
    pub mandatory_required_mask: u8,
    pub mandatory_verified_mask: u8,
    pub missed_mandatory_count: u32,
    pub rejected_candidate_count: u32,
    pub rejected_candidate_cost: u32,
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

fn operation_kind(mask: u8) -> OperationKind {
    if mask == 0 {
        OperationKind::RepairDerivation
    } else if mask & TEMPORAL != 0 && mask & INDEPENDENCE != 0 {
        OperationKind::CompareAlternatives
    } else if mask & INDEPENDENCE != 0 {
        OperationKind::SeekIndependentEvidence
    } else if mask & TEMPORAL != 0 {
        OperationKind::RefreshEvidence
    } else {
        OperationKind::Verify
    }
}

/// Return the subset of advertised obligations that the typed operation kind
/// can address under this benchmark's declared obligation-state profile.
fn operation_admissible_mask(mask: u8) -> u8 {
    match operation_kind(mask) {
        OperationKind::CompareAlternatives => mask & (INDEPENDENCE | AUTHORITY | PROOF_PATH),
        OperationKind::SeekIndependentEvidence => mask & INDEPENDENCE,
        OperationKind::RefreshEvidence => mask & TEMPORAL,
        OperationKind::Verify => mask & (AUTHORITY | PROOF_PATH),
        _ => 0,
    }
}

fn oracle_subset(case: &TaskCase) -> Option<(u64, u32)> {
    if case.operations.len() > 20 {
        return None;
    }

    let n = case.operations.len();
    let mandatory_indices: Vec<usize> = case
        .operations
        .iter()
        .enumerate()
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
                covered |= operation.verified_mask
                    & operation_admissible_mask(operation.public.advertised_mask);
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

fn choose_with_pod_meta_router(
    case: &TaskCase,
    attempted: &[bool],
    attempted_without_progress: &BTreeSet<(String, String)>,
    verified_mask: u8,
    progress_key: &str,
    spent: u32,
    budget: u32,
) -> (Option<usize>, u8, u32) {
    let remaining = budget.saturating_sub(spent);
    let mandatory_remaining_cost = case.mandatory_cost(attempted);
    if mandatory_remaining_cost > remaining {
        // Fail closed before invoking the generic selector: do not spend
        // partial mandatory capacity if the complete mandatory set cannot fit.
        return (None, 0, 0);
    }

    let obligation_rows = [
        (TEMPORAL, "temporal", ObligationState::StaleOrRetracted, 6),
        (INDEPENDENCE, "independence", ObligationState::Contested, 6),
        (
            AUTHORITY,
            "authority",
            ObligationState::CandidateEvidenceAvailable,
            8,
        ),
        (
            PROOF_PATH,
            "proof_path",
            ObligationState::CandidateEvidenceAvailable,
            8,
        ),
    ];

    let obligations: Vec<PodObligation> = obligation_rows
        .iter()
        .filter(|(bit, _, _, _)| case.required_mask & *bit != 0)
        .map(|(bit, id, initial_state, priority)| {
            let state = if verified_mask & *bit != 0 {
                ObligationState::SupportedUnderProfile
            } else {
                *initial_state
            };
            PodObligation::new(*id, state, *priority).with_state_identity(progress_key.to_owned())
        })
        .collect();

    let all_ids = vec![
        "temporal".to_owned(),
        "independence".to_owned(),
        "authority".to_owned(),
        "proof_path".to_owned(),
    ];
    let operations: Vec<PodCandidateOperation> = case
        .operations
        .iter()
        .map(|operation| {
            let mask = operation.public.advertised_mask & case.required_mask;
            let target_mask = operation_admissible_mask(mask);
            let targets: Vec<String> = obligation_rows
                .iter()
                .filter(|(bit, _, _, _)| target_mask & *bit != 0)
                .map(|(_, id, _, _)| (*id).to_owned())
                .collect();
            let targets = if targets.is_empty() {
                all_ids.clone()
            } else {
                targets
            };

            // One typed operation family is used per candidate. For mixed
            // profiles the real POD-Meta scheduler chooses only compatible
            // obligations; the evaluator below credits only those selected.
            let kind = operation_kind(mask);
            let class = if operation.public.mandatory_mask != 0 {
                SchedulingClass::Mandatory
            } else {
                SchedulingClass::Competitive
            };
            let candidate = PodCandidateOperation::new(
                operation.public.id,
                kind,
                targets,
                class,
                u64::from(operation.public.cost),
                progress_key.to_owned(),
            );
            if target_mask == 0 {
                candidate.inapplicable()
            } else {
                candidate
            }
        })
        .collect();

    let decision = pod_meta_choose_next(&SelectionContext {
        obligations: &obligations,
        operations: &operations,
        attempted_without_progress,
        remaining_work_units: u64::from(remaining),
        mandatory_reserve_units: u64::from(mandatory_remaining_cost),
    });
    let rejection_count = decision.rejected.len() as u32;

    match decision.selection {
        PodSelection::Run {
            operation_id,
            obligation_ids,
            ..
        } => {
            let selected_mask = obligation_ids.iter().fold(0u8, |mask, id| {
                mask | match id.as_str() {
                    "temporal" => TEMPORAL,
                    "independence" => INDEPENDENCE,
                    "authority" => AUTHORITY,
                    "proof_path" => PROOF_PATH,
                    _ => 0,
                }
            });
            let selected_index = case
                .operations
                .iter()
                .position(|operation| operation.public.id == operation_id.as_str());
            (selected_index, selected_mask, rejection_count)
        }
        PodSelection::Stop(_) => (None, 0, rejection_count),
    }
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
    let affordable: Vec<usize> = case
        .operations
        .iter()
        .enumerate()
        .filter_map(|(i, op)| (!attempted[i] && op.public.cost <= remaining).then_some(i))
        .collect();
    if affordable.is_empty() {
        return None;
    }

    match policy {
        Policy::FixedOrder => affordable.into_iter().next(),
        Policy::GenericVoc | Policy::UnverifiedConfidence => {
            let candidates: Vec<usize> = if policy == Policy::UnverifiedConfidence {
                affordable
            } else {
                affordable
                    .into_iter()
                    .filter(|i| {
                        operation_admissible_mask(case.operations[*i].public.advertised_mask)
                            & unresolved
                            != 0
                    })
                    .collect()
            };
            candidates.into_iter().max_by(|a, b| {
                let oa = &case.operations[*a].public;
                let ob = &case.operations[*b].public;
                ratio_cmp(oa.predicted_gain, oa.cost, ob.predicted_gain, ob.cost)
                    .then_with(|| ob.cost.cmp(&oa.cost))
                    .then_with(|| ob.id.cmp(oa.id))
            })
        }
        Policy::BudgetMatchedRandom => {
            let eligible: Vec<usize> = affordable
                .into_iter()
                .filter(|i| {
                    operation_admissible_mask(case.operations[*i].public.advertised_mask)
                        & unresolved
                        != 0
                })
                .collect();
            if eligible.is_empty() {
                return None;
            }
            let mut state = seed
                .wrapping_add(u64::from(spent).wrapping_mul(6364136223846793005))
                .wrapping_add(attempted.iter().filter(|x| **x).count() as u64);
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            Some(eligible[((state >> 32) as usize) % eligible.len()])
        }
        Policy::PodMetaRouter => None,
        Policy::MaskCoverageGreedy => {
            let mandatory_remaining: Vec<usize> = case
                .operations
                .iter()
                .enumerate()
                .filter_map(|(i, operation)| {
                    (operation.public.mandatory_mask != 0 && !attempted[i]).then_some(i)
                })
                .collect();
            if !mandatory_remaining.is_empty() {
                // Reserve the complete remaining mandatory verification budget
                // before spending on competitive or opportunistic operations.
                let total_mandatory = case.mandatory_cost(attempted);
                if total_mandatory > remaining {
                    return None;
                }
                return mandatory_remaining.into_iter().min_by(|a, b| {
                    case.operations[*a]
                        .public
                        .cost
                        .cmp(&case.operations[*b].public.cost)
                        .then_with(|| {
                            case.operations[*a]
                                .public
                                .id
                                .cmp(case.operations[*b].public.id)
                        })
                });
            }

            affordable
                .into_iter()
                .filter(|i| {
                    operation_admissible_mask(case.operations[*i].public.advertised_mask)
                        & unresolved
                        != 0
                })
                .max_by(|a, b| {
                    let oa = &case.operations[*a].public;
                    let ob = &case.operations[*b].public;
                    let ga = popcount(operation_admissible_mask(oa.advertised_mask) & unresolved);
                    let gb = popcount(operation_admissible_mask(ob.advertised_mask) & unresolved);
                    ratio_cmp(ga, oa.cost, gb, ob.cost)
                        .then_with(|| oa.predicted_gain.cmp(&ob.predicted_gain))
                        .then_with(|| ob.cost.cmp(&oa.cost))
                        .then_with(|| ob.id.cmp(oa.id))
                })
        }
        Policy::OracleMinimumCost => {
            let (subset, _) = oracle_subset(case)?;
            affordable
                .into_iter()
                .filter(|i| subset & (1u64 << i) != 0)
                .min_by(|a, b| {
                    case.operations[*a]
                        .public
                        .cost
                        .cmp(&case.operations[*b].public.cost)
                        .then_with(|| {
                            case.operations[*a]
                                .public
                                .id
                                .cmp(case.operations[*b].public.id)
                        })
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
    let mut scheduler_candidate_rejections = 0u32;
    let mut attempted_without_progress = BTreeSet::<(String, String)>::new();
    let mut progress_version = 0u32;
    let mut rejected_candidate_count = 0u32;
    let mut rejected_candidate_cost = 0u32;
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

        let progress_key = format!("proof-state-{progress_version}");
        let (next, selected_obligation_mask) = if policy == Policy::PodMetaRouter {
            let (selected, selected_mask, rejected) = choose_with_pod_meta_router(
                case,
                &attempted,
                &attempted_without_progress,
                verified_mask,
                &progress_key,
                spent,
                budget,
            );
            scheduler_candidate_rejections =
                scheduler_candidate_rejections.saturating_add(rejected);
            (selected, selected_mask)
        } else {
            (
                choose_next(case, policy, &attempted, unresolved, spent, budget, seed),
                case.required_mask,
            )
        };
        let Some(index) = next else {
            break;
        };
        let unresolved_before = case.required_mask & !verified_mask;
        let operation = &case.operations[index];
        let selected_obligation_mask = if policy == Policy::PodMetaRouter {
            selected_obligation_mask
        } else {
            operation_admissible_mask(operation.public.advertised_mask)
        };
        attempted[index] = true;
        operations += 1;
        spent = spent.saturating_add(operation.public.cost);

        let actual = operation.verified_mask & case.required_mask & selected_obligation_mask;
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

        if newly_verified == 0 {
            attempted_without_progress.insert((operation.public.id.to_owned(), progress_key));
        } else {
            progress_version = progress_version.saturating_add(1);
        }

        if newly_verified == 0 && operation.public.mandatory_mask == 0 {
            if operation_admissible_mask(operation.public.advertised_mask) & unresolved_before != 0
            {
                // The attempted operation targeted an open obligation, but
                // independent verification rejected its proposed support.
                rejected_candidate_count += 1;
                rejected_candidate_cost =
                    rejected_candidate_cost.saturating_add(operation.public.cost);
            } else {
                // The operation could not address any currently open proof
                // obligation, so its cost is redundant/irrelevant in this task.
                redundant_operation_count += 1;
                wasted_operation_cost = wasted_operation_cost.saturating_add(operation.public.cost);
            }
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
    let all_mandatory_attempted = case
        .operations
        .iter()
        .enumerate()
        .all(|(i, op)| op.public.mandatory_mask == 0 || attempted[i]);
    let reported_complete = match policy {
        Policy::UnverifiedConfidence => {
            claimed_mask & case.required_mask == case.required_mask && all_mandatory_attempted
        }
        _ => completion_verified,
    };
    let false_support_count = popcount(claimed_mask & !verified_mask & case.required_mask);
    let false_completion = reported_complete && !completion_verified;
    let correct_abstention = !reported_complete && !completion_verified && false_support_count == 0;
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
        seed,
        policy,
        budget,
        spent,
        operation_count: operations,
        scheduler_candidate_rejections,
        verified_obligations: popcount(verified_mask & case.required_mask),
        required_obligations: popcount(case.required_mask),
        reported_complete,
        completion_verified,
        false_completion,
        false_support_count,
        mandatory_required_mask: case.mandatory_mask(),
        mandatory_verified_mask,
        missed_mandatory_count,
        rejected_candidate_count,
        rejected_candidate_cost,
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
        let case = build_cases()
            .into_iter()
            .find(|c| c.id == "valid_multi_obligation_packet")
            .unwrap();
        let result = run_trial(&case, Policy::MaskCoverageGreedy, 14, 1729);
        assert!(result.completion_verified);
        assert_eq!(result.spent, 9);
        assert_eq!(result.missed_mandatory_count, 0);
        assert_eq!(result.false_support_count, 0);
    }

    #[test]
    fn proof_guided_does_not_accept_stale_evidence_claim_as_temporal_support() {
        let case = build_cases()
            .into_iter()
            .find(|c| c.id == "attractive_but_stale_evidence")
            .unwrap();
        let result = run_trial(&case, Policy::MaskCoverageGreedy, 10, 1729);
        assert!(result.completion_verified);
        assert_eq!(result.false_support_count, 0);
        assert_eq!(result.missed_mandatory_count, 0);
        assert!(result.rejected_candidate_count > 0);
        assert!(result.rejected_candidate_cost > 0);
    }

    #[test]
    fn unverified_confidence_control_exposes_false_support() {
        let case = build_cases()
            .into_iter()
            .find(|c| c.id == "attractive_but_stale_evidence")
            .unwrap();
        let result = run_trial(&case, Policy::UnverifiedConfidence, 10, 1729);
        assert!(result.false_support_count > 0 || result.false_completion);
        assert!(!result.correct_abstention);
    }

    #[test]
    fn mandatory_work_is_not_displaced_by_nonmandatory_gain() {
        let case = build_cases()
            .into_iter()
            .find(|c| c.id == "minimum_sufficient_combined_packet")
            .unwrap();
        let result = run_trial(&case, Policy::MaskCoverageGreedy, 5, 1729);
        assert_eq!(result.spent, 0);
        assert!(!result.completion_verified);
        assert_eq!(result.missed_mandatory_count, 2);
    }

    #[test]
    fn actual_pod_meta_router_completes_verified_workflow_under_matched_budget() {
        let case = build_cases()
            .into_iter()
            .find(|c| c.id == "valid_multi_obligation_packet")
            .unwrap();
        let result = run_trial(&case, Policy::PodMetaRouter, 14, 1729);
        assert!(result.completion_verified);
        assert_eq!(result.missed_mandatory_count, 0);
        assert_eq!(result.false_support_count, 0);
        assert!(result.scheduler_candidate_rejections > 0);
    }

    #[test]
    fn actual_pod_meta_router_preserves_mandatory_budget_reserve() {
        let case = build_cases()
            .into_iter()
            .find(|c| c.id == "minimum_sufficient_combined_packet")
            .unwrap();
        let result = run_trial(&case, Policy::PodMetaRouter, 5, 1729);
        assert_eq!(result.spent, 0);
        assert!(!result.completion_verified);
        assert_eq!(result.missed_mandatory_count, 2);
    }

    #[test]
    fn actual_pod_meta_router_never_credits_unselected_obligations() {
        let case = build_cases()
            .into_iter()
            .find(|c| c.id == "valid_multi_obligation_packet")
            .unwrap();
        let result = run_trial(&case, Policy::PodMetaRouter, 14, 1729);
        assert!(result.completion_verified);
        assert_eq!(result.verified_obligations, result.required_obligations);
        assert_eq!(result.false_support_count, 0);
    }

    #[test]
    fn actual_pod_meta_router_never_false_completes_across_budget_curve() {
        for case in build_cases() {
            for budget in [3, 5, 7, 10, 14, 20] {
                let result = run_trial(&case, Policy::PodMetaRouter, budget, 1729);
                assert!(!result.false_completion, "{}:{}", case.id, budget);
                assert_eq!(result.false_support_count, 0, "{}:{}", case.id, budget);
            }
        }
    }

    #[test]
    fn all_policy_arms_share_the_typed_operation_admissibility_contract() {
        assert_eq!(
            operation_admissible_mask(TEMPORAL | INDEPENDENCE),
            INDEPENDENCE
        );
        assert_eq!(operation_admissible_mask(TEMPORAL), TEMPORAL);
        assert_eq!(operation_admissible_mask(INDEPENDENCE), INDEPENDENCE);
        assert_eq!(
            operation_admissible_mask(AUTHORITY | PROOF_PATH),
            AUTHORITY | PROOF_PATH
        );
        assert_eq!(operation_admissible_mask(0), 0);
    }

    #[test]
    fn oracle_cost_is_no_greater_than_any_verified_controller_cost() {
        for case in build_cases() {
            let oracle = run_trial(&case, Policy::OracleMinimumCost, 100, 1729);
            assert!(oracle.completion_verified, "{}", case.id);
            let proof = run_trial(&case, Policy::MaskCoverageGreedy, 100, 1729);
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
                let result = run_trial(&case, Policy::MaskCoverageGreedy, budget, 42);
                assert!(!result.false_completion, "{}:{}", case.id, budget);
                assert_eq!(result.false_support_count, 0);
            }
        }
    }

    #[test]
    fn oracle_is_not_a_promotion_candidate() {
        assert!(!Policy::OracleMinimumCost.promotion_candidate());
        assert!(!Policy::UnverifiedConfidence.promotion_candidate());
        assert!(Policy::MaskCoverageGreedy.promotion_candidate());
        assert!(Policy::PodMetaRouter.promotion_candidate());
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
            let covered = case.operations.iter().fold(0u8, |mask, operation| {
                mask | (operation_admissible_mask(operation.public.advertised_mask)
                    & operation.public.mandatory_mask)
            });
            assert_eq!(covered, REQUIRED_MANDATORY, "{}", case.id);
        }
    }

    #[test]
    fn exhaustive_budget_curve_contains_low_and_high_budget_points() {
        let case = build_cases().remove(1);
        let low = run_trial(&case, Policy::MaskCoverageGreedy, 3, 10);
        let high = run_trial(&case, Policy::MaskCoverageGreedy, 14, 10);
        assert!(high.spent >= low.spent);
        assert!(high.verified_obligations >= low.verified_obligations);
    }

    #[test]
    fn rejected_evidence_and_irrelevant_computation_have_separate_costs() {
        let cases = build_cases();
        let stale = cases
            .iter()
            .find(|c| c.id == "attractive_but_stale_evidence")
            .unwrap();
        let rejected = run_trial(stale, Policy::MaskCoverageGreedy, 10, 10);
        assert!(rejected.rejected_candidate_count > 0);
        assert!(rejected.rejected_candidate_cost > 0);

        let redundant = run_trial(&cases[0], Policy::FixedOrder, 14, 10);
        assert!(redundant.redundant_operation_count > 0);
        assert!(redundant.wasted_operation_cost > 0);
    }
}
