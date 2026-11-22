// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Receipt-gated pairwise portfolio interactions.
//! delta(i,j) = U({i,j}) - U({i}) - U({j}) + U(empty)
//! Positive values model complementarity; negative values model redundancy.

use super::*;
use std::collections::{BTreeMap, BTreeSet};

const MAX_INTERACTION_PAIRS: usize = 4_096;
const MAX_INTERACTION_OPTIONS: usize = 90;
const MIN_TRAINING_PAIRS: u32 = 200;
const MIN_CALIBRATION_PAIRS: u32 = 64;
const MIN_HELDOUT_PAIRS: u32 = 64;
const MIN_HELDOUT_MULTI_EVENT_SAMPLES: u32 = 32;
const MIN_INDEPENDENT_SEEDS: u16 = 5;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PortfolioInteractionReject {
    Portfolio(PortfolioReject),
    InvalidProposalDigest,
    InvalidOptionSetDigest,
    ContextMismatch,
    InvalidInteractionDigest,
    PairNotCanonical,
    DuplicatePair,
    PairReferencesUnknownOption,
    SameEventPair,
    IncompletePairMatrix,
    TooManyPairs,
    CalibrationUnavailable,
    CalibrationMismatch,
    CalibrationDatasetOverlap,
    CalibrationSplitNotIndependent,
    CalibrationEvidenceInsufficient,
    CalibrationQualityInsufficient,
    CalibrationNotValidAtSnapshot,
    InvalidCalibrationReceipt,
    ValueOverflow,
    SearchLimitExceeded,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairwiseInteractionEstimateV1 {
    pub left_option_sha256: String,
    pub right_option_sha256: String,
    /// Second-order utility difference in normalized-utility micro-units.
    pub interaction_micros: i64,
    pub absolute_error_bound_micros: u64,
}

impl PairwiseInteractionEstimateV1 {
    pub fn conservative_value_micros(&self) -> Result<i64, PortfolioInteractionReject> {
        let error = i64::try_from(self.absolute_error_bound_micros)
            .map_err(|_| PortfolioInteractionReject::ValueOverflow)?;
        self.interaction_micros
            .checked_sub(error)
            .ok_or(PortfolioInteractionReject::ValueOverflow)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PortfolioInteractionProposalV1 {
    pub scope: Scope,
    pub as_known: AsKnownQuery,
    pub budget: MemoryPortfolioBudgetV1,
    pub target_model_sha256: String,
    pub baseline_policy_sha256: String,
    pub task_protocol_sha256: String,
    pub interaction_model_sha256: String,
    pub option_set_sha256: String,
    pub training_dataset_sha256: String,
    pub calibration_dataset_sha256: String,
    pub heldout_evaluation_sha256: String,
    pub task_family_split_sha256: String,
    pub source_root_split_sha256: String,
    pub pairs: Vec<PairwiseInteractionEstimateV1>,
    pub proposal_sha256: String,
}

impl PortfolioInteractionProposalV1 {
    pub fn canonical_digest(&self) -> String {
        let mut fields = vec![
            scope_code(self.scope).to_owned(),
            self.as_known.valid_at_ms.to_string(),
            self.as_known.known_at_ms.to_string(),
            self.budget.context_tokens.to_string(),
            self.budget.reobservation_calls.to_string(),
            self.budget.latency_work_ms.to_string(),
            self.budget.estimated_cost_micros.to_string(),
            self.budget.expected_failure_loss_micros.to_string(),
            self.target_model_sha256.clone(),
            self.baseline_policy_sha256.clone(),
            self.task_protocol_sha256.clone(),
            self.interaction_model_sha256.clone(),
            self.option_set_sha256.clone(),
            self.training_dataset_sha256.clone(),
            self.calibration_dataset_sha256.clone(),
            self.heldout_evaluation_sha256.clone(),
            self.task_family_split_sha256.clone(),
            self.source_root_split_sha256.clone(),
            self.pairs.len().to_string(),
        ];
        for pair in &self.pairs {
            fields.extend([
                pair.left_option_sha256.clone(),
                pair.right_option_sha256.clone(),
                pair.interaction_micros.to_string(),
                pair.absolute_error_bound_micros.to_string(),
            ]);
        }
        digest_fields(b"mace:portfolio-interaction-proposal:v1\0", &fields)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PortfolioInteractionCalibrationReceiptV1 {
    pub proposal_sha256: String,
    pub target_model_sha256: String,
    pub baseline_policy_sha256: String,
    pub task_protocol_sha256: String,
    pub interaction_model_sha256: String,
    pub training_dataset_sha256: String,
    pub calibration_dataset_sha256: String,
    pub heldout_evaluation_sha256: String,
    pub task_family_split_sha256: String,
    pub source_root_split_sha256: String,
    pub training_pair_count: u32,
    pub calibration_pair_count: u32,
    pub heldout_pair_count: u32,
    pub independent_seed_count: u16,
    pub task_families_disjoint: bool,
    pub source_roots_disjoint: bool,
    pub calibration_mae_micros: u64,
    pub base_rate_mae_micros: u64,
    pub heldout_mae_micros: u64,
    pub heldout_base_rate_mae_micros: u64,
    /// Paired-cluster-bootstrap 95% lower bound for held-out MAE improvement.
    pub heldout_mae_improvement_lower_95_micros: i64,
    pub heldout_multi_event_sample_count: u32,
    pub heldout_higher_order_residual_micros: u64,
    pub maximum_allowed_higher_order_residual_micros: u64,
    pub verified_at_ms: i64,
    pub valid_until_ms: i64,
    pub verifier_id: String,
    pub receipt_sha256: String,
}

impl PortfolioInteractionCalibrationReceiptV1 {
    pub fn canonical_digest(&self) -> String {
        digest_fields(
            b"mycelix:mace:portfolio-interaction-calibration:v1\0",
            &[
                self.proposal_sha256.clone(),
                self.target_model_sha256.clone(),
                self.baseline_policy_sha256.clone(),
                self.task_protocol_sha256.clone(),
                self.interaction_model_sha256.clone(),
                self.training_dataset_sha256.clone(),
                self.calibration_dataset_sha256.clone(),
                self.heldout_evaluation_sha256.clone(),
                self.task_family_split_sha256.clone(),
                self.source_root_split_sha256.clone(),
                self.training_pair_count.to_string(),
                self.calibration_pair_count.to_string(),
                self.heldout_pair_count.to_string(),
                self.independent_seed_count.to_string(),
                self.task_families_disjoint.to_string(),
                self.source_roots_disjoint.to_string(),
                self.calibration_mae_micros.to_string(),
                self.base_rate_mae_micros.to_string(),
                self.heldout_mae_micros.to_string(),
                self.heldout_base_rate_mae_micros.to_string(),
                self.heldout_mae_improvement_lower_95_micros.to_string(),
                self.heldout_multi_event_sample_count.to_string(),
                self.heldout_higher_order_residual_micros.to_string(),
                self.maximum_allowed_higher_order_residual_micros.to_string(),
                self.verified_at_ms.to_string(),
                self.valid_until_ms.to_string(),
                self.verifier_id.clone(),
            ],
        )
    }
}

/// Production implementations authenticate the issuer and validate split
/// lineage/metrics against independent evaluation evidence. Fixture verifiers
/// are for tests only.
pub trait PortfolioInteractionVerifierV1 {
    fn verify(
        &self,
        proposal: &PortfolioInteractionProposalV1,
    ) -> Option<PortfolioInteractionCalibrationReceiptV1>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PortfolioInteractionContextV1 {
    pub scope: Scope,
    pub as_known: AsKnownQuery,
    pub budget: MemoryPortfolioBudgetV1,
    pub target_model_sha256: String,
    pub baseline_policy_sha256: String,
    pub task_protocol_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedPortfolioInteractionSetV1 {
    proposal: PortfolioInteractionProposalV1,
    calibration: PortfolioInteractionCalibrationReceiptV1,
}

impl VerifiedPortfolioInteractionSetV1 {
    pub fn proposal_sha256(&self) -> &str {
        &self.proposal.proposal_sha256
    }

    pub fn calibration_receipt_sha256(&self) -> &str {
        &self.calibration.receipt_sha256
    }

    pub fn pair_count(&self) -> usize {
        self.proposal.pairs.len()
    }
}

fn pair_key(left: &str, right: &str) -> (String, String) {
    if left < right {
        (left.to_owned(), right.to_owned())
    } else {
        (right.to_owned(), left.to_owned())
    }
}

fn option_set_digest(options: &[MemoryPortfolioOptionV1]) -> String {
    let mut digests: Vec<_> = options
        .iter()
        .map(|option| option.option_sha256.clone())
        .collect();
    digests.sort();
    digest_fields(b"mace:portfolio-interaction-option-set:v1\0", &digests)
}

fn validate_interaction_proposal(
    proposal: &PortfolioInteractionProposalV1,
    options: &[MemoryPortfolioOptionV1],
    context: &PortfolioInteractionContextV1,
) -> Result<(), PortfolioInteractionReject> {
    prepare_options(options, context.budget).map_err(PortfolioInteractionReject::Portfolio)?;
    if options.len() > MAX_INTERACTION_OPTIONS {
        return Err(PortfolioInteractionReject::TooManyPairs);
    }
    if proposal.proposal_sha256 != proposal.canonical_digest() {
        return Err(PortfolioInteractionReject::InvalidProposalDigest);
    }
    if proposal.scope != context.scope
        || proposal.as_known != context.as_known
        || proposal.budget != context.budget
        || proposal.target_model_sha256 != context.target_model_sha256
        || proposal.baseline_policy_sha256 != context.baseline_policy_sha256
        || proposal.task_protocol_sha256 != context.task_protocol_sha256
    {
        return Err(PortfolioInteractionReject::ContextMismatch);
    }

    let digests = [
        &proposal.target_model_sha256,
        &proposal.baseline_policy_sha256,
        &proposal.task_protocol_sha256,
        &proposal.interaction_model_sha256,
        &proposal.option_set_sha256,
        &proposal.training_dataset_sha256,
        &proposal.calibration_dataset_sha256,
        &proposal.heldout_evaluation_sha256,
        &proposal.task_family_split_sha256,
        &proposal.source_root_split_sha256,
    ];
    if digests.iter().any(|digest| !is_lower_sha256(digest)) {
        return Err(PortfolioInteractionReject::InvalidInteractionDigest);
    }
    if proposal.option_set_sha256 != option_set_digest(options) {
        return Err(PortfolioInteractionReject::InvalidOptionSetDigest);
    }
    if proposal.training_dataset_sha256 == proposal.calibration_dataset_sha256
        || proposal.training_dataset_sha256 == proposal.heldout_evaluation_sha256
        || proposal.calibration_dataset_sha256 == proposal.heldout_evaluation_sha256
    {
        return Err(PortfolioInteractionReject::CalibrationDatasetOverlap);
    }
    if proposal.pairs.len() > MAX_INTERACTION_PAIRS {
        return Err(PortfolioInteractionReject::TooManyPairs);
    }

    let option_by_digest: BTreeMap<_, _> = options
        .iter()
        .map(|option| (option.option_sha256.as_str(), option))
        .collect();
    let mut expected_pairs = BTreeSet::new();
    for left in 0..options.len() {
        for right in (left + 1)..options.len() {
            if options[left].event_id == options[right].event_id {
                continue;
            }
            expected_pairs.insert(pair_key(
                &options[left].option_sha256,
                &options[right].option_sha256,
            ));
        }
    }

    let mut observed_pairs = BTreeSet::new();
    let mut previous_key: Option<(String, String)> = None;
    for pair in &proposal.pairs {
        if !is_lower_sha256(&pair.left_option_sha256)
            || !is_lower_sha256(&pair.right_option_sha256)
            || pair.left_option_sha256 >= pair.right_option_sha256
        {
            return Err(PortfolioInteractionReject::PairNotCanonical);
        }
        let key = (
            pair.left_option_sha256.clone(),
            pair.right_option_sha256.clone(),
        );
        if previous_key.as_ref().is_some_and(|previous| previous >= &key) {
            return Err(PortfolioInteractionReject::PairNotCanonical);
        }
        previous_key = Some(key.clone());

        if !observed_pairs.insert(key.clone()) {
            return Err(PortfolioInteractionReject::DuplicatePair);
        }
        let Some(left) = option_by_digest.get(pair.left_option_sha256.as_str()) else {
            return Err(PortfolioInteractionReject::PairReferencesUnknownOption);
        };
        let Some(right) = option_by_digest.get(pair.right_option_sha256.as_str()) else {
            return Err(PortfolioInteractionReject::PairReferencesUnknownOption);
        };
        if left.event_id == right.event_id {
            return Err(PortfolioInteractionReject::SameEventPair);
        }
        pair.conservative_value_micros()?;
    }
    if observed_pairs != expected_pairs {
        return Err(PortfolioInteractionReject::IncompletePairMatrix);
    }
    Ok(())
}

/// Verify model identity, calibration evidence, independent splits, pair-matrix
/// completeness and as-known validity before authorizing portfolio selection.
pub fn verify_portfolio_interactions(
    proposal: PortfolioInteractionProposalV1,
    options: &[MemoryPortfolioOptionV1],
    context: &PortfolioInteractionContextV1,
    verifier: &impl PortfolioInteractionVerifierV1,
) -> Result<VerifiedPortfolioInteractionSetV1, PortfolioInteractionReject> {
    validate_interaction_proposal(&proposal, options, context)?;
    let calibration = verifier
        .verify(&proposal)
        .ok_or(PortfolioInteractionReject::CalibrationUnavailable)?;

    if calibration.proposal_sha256 != proposal.proposal_sha256
        || calibration.target_model_sha256 != proposal.target_model_sha256
        || calibration.baseline_policy_sha256 != proposal.baseline_policy_sha256
        || calibration.task_protocol_sha256 != proposal.task_protocol_sha256
        || calibration.interaction_model_sha256 != proposal.interaction_model_sha256
        || calibration.training_dataset_sha256 != proposal.training_dataset_sha256
        || calibration.calibration_dataset_sha256 != proposal.calibration_dataset_sha256
        || calibration.heldout_evaluation_sha256 != proposal.heldout_evaluation_sha256
        || calibration.task_family_split_sha256 != proposal.task_family_split_sha256
        || calibration.source_root_split_sha256 != proposal.source_root_split_sha256
    {
        return Err(PortfolioInteractionReject::CalibrationMismatch);
    }
    if calibration.training_dataset_sha256 == calibration.calibration_dataset_sha256
        || calibration.training_dataset_sha256 == calibration.heldout_evaluation_sha256
        || calibration.calibration_dataset_sha256 == calibration.heldout_evaluation_sha256
    {
        return Err(PortfolioInteractionReject::CalibrationDatasetOverlap);
    }
    if !calibration.task_families_disjoint || !calibration.source_roots_disjoint {
        return Err(PortfolioInteractionReject::CalibrationSplitNotIndependent);
    }
    if calibration.training_pair_count < MIN_TRAINING_PAIRS
        || calibration.calibration_pair_count < MIN_CALIBRATION_PAIRS
        || calibration.heldout_pair_count < MIN_HELDOUT_PAIRS
        || calibration.independent_seed_count < MIN_INDEPENDENT_SEEDS
    {
        return Err(PortfolioInteractionReject::CalibrationEvidenceInsufficient);
    }
    if calibration.calibration_mae_micros >= calibration.base_rate_mae_micros
        || calibration.heldout_mae_micros >= calibration.heldout_base_rate_mae_micros
        || calibration.heldout_mae_improvement_lower_95_micros <= 0
        || calibration.heldout_higher_order_residual_micros
            > calibration.maximum_allowed_higher_order_residual_micros
    {
        return Err(PortfolioInteractionReject::CalibrationQualityInsufficient);
    }
    let distinct_events: BTreeSet<_> =
        options.iter().map(MemoryPortfolioOptionV1::event_id).collect();
    if distinct_events.len() >= 3
        && calibration.heldout_multi_event_sample_count
            < MIN_HELDOUT_MULTI_EVENT_SAMPLES
    {
        return Err(PortfolioInteractionReject::CalibrationEvidenceInsufficient);
    }
    if calibration.verified_at_ms > context.as_known.known_at_ms
        || calibration.valid_until_ms <= context.as_known.known_at_ms
    {
        return Err(PortfolioInteractionReject::CalibrationNotValidAtSnapshot);
    }
    if calibration.verifier_id.trim().is_empty()
        || !is_lower_sha256(&calibration.receipt_sha256)
        || calibration.receipt_sha256 != calibration.canonical_digest()
    {
        return Err(PortfolioInteractionReject::InvalidCalibrationReceipt);
    }

    Ok(VerifiedPortfolioInteractionSetV1 {
        proposal,
        calibration,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectedPortfolioInteractionV1 {
    pub left_event_id: u32,
    pub right_event_id: u32,
    pub left_option_sha256: String,
    pub right_option_sha256: String,
    pub predicted_interaction_micros: i64,
    pub absolute_error_bound_micros: u64,
    pub conservative_interaction_micros: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InteractionAwarePortfolioPlanV1 {
    /// The underlying selection value is additive; the total below includes
    /// conservative pairwise interactions.
    pub selection: MemoryPortfolioPlanV1,
    pub proposal_sha256: String,
    pub calibration_receipt_sha256: String,
    pub additive_value_micros: i64,
    pub conservative_interaction_value_micros: i64,
    pub total_conservative_value_micros: i64,
    pub selected_interactions: Vec<SelectedPortfolioInteractionV1>,
    pub plan_sha256: String,
}

impl InteractionAwarePortfolioPlanV1 {
    pub fn canonical_digest(&self) -> String {
        let mut fields = vec![
            self.selection.plan_sha256.clone(),
            self.proposal_sha256.clone(),
            self.calibration_receipt_sha256.clone(),
            self.additive_value_micros.to_string(),
            self.conservative_interaction_value_micros.to_string(),
            self.total_conservative_value_micros.to_string(),
            self.selected_interactions.len().to_string(),
        ];
        for pair in &self.selected_interactions {
            fields.extend([
                pair.left_event_id.to_string(),
                pair.right_event_id.to_string(),
                pair.left_option_sha256.clone(),
                pair.right_option_sha256.clone(),
                pair.predicted_interaction_micros.to_string(),
                pair.absolute_error_bound_micros.to_string(),
                pair.conservative_interaction_micros.to_string(),
            ]);
        }
        digest_fields(b"mace:interaction-aware-portfolio-plan:v1\0", &fields)
    }

    pub fn is_canonical(&self) -> bool {
        if !self.selection.is_canonical()
            || self.selection.strategy != PortfolioStrategyV1::PairwiseInteractionAware
            || !is_lower_sha256(&self.proposal_sha256)
            || !is_lower_sha256(&self.calibration_receipt_sha256)
            || self.selection.total_net_value_micros != self.additive_value_micros
            || !is_lower_sha256(&self.plan_sha256)
            || self.plan_sha256 != self.canonical_digest()
        {
            return false;
        }

        let selected: BTreeMap<_, _> = self
            .selection
            .selected
            .iter()
            .map(|item| (item.option_sha256.as_str(), item.event_id))
            .collect();
        let selected_items: Vec<_> = self.selection.selected.iter().collect();
        let mut expected_pairs = BTreeSet::new();
        for left in 0..selected_items.len() {
            for right in (left + 1)..selected_items.len() {
                if selected_items[left].event_id == selected_items[right].event_id {
                    return false;
                }
                expected_pairs.insert(pair_key(
                    &selected_items[left].option_sha256,
                    &selected_items[right].option_sha256,
                ));
            }
        }

        let mut actual_pairs = BTreeSet::new();
        let mut previous: Option<(String, String)> = None;
        let mut interaction_sum = 0_i64;
        for pair in &self.selected_interactions {
            if pair.left_option_sha256 >= pair.right_option_sha256
                || !is_lower_sha256(&pair.left_option_sha256)
                || !is_lower_sha256(&pair.right_option_sha256)
            {
                return false;
            }
            let key = (
                pair.left_option_sha256.clone(),
                pair.right_option_sha256.clone(),
            );
            if previous.as_ref().is_some_and(|value| value >= &key)
                || !actual_pairs.insert(key.clone())
            {
                return false;
            }
            previous = Some(key);

            if selected.get(pair.left_option_sha256.as_str()) != Some(&pair.left_event_id)
                || selected.get(pair.right_option_sha256.as_str()) != Some(&pair.right_event_id)
                || pair.left_event_id == pair.right_event_id
            {
                return false;
            }
            let Ok(error) = i64::try_from(pair.absolute_error_bound_micros) else {
                return false;
            };
            let Some(conservative) = pair.predicted_interaction_micros.checked_sub(error) else {
                return false;
            };
            if conservative != pair.conservative_interaction_micros {
                return false;
            }
            let Some(next) = interaction_sum.checked_add(conservative) else {
                return false;
            };
            interaction_sum = next;
        }

        if actual_pairs != expected_pairs
            || interaction_sum != self.conservative_interaction_value_micros
        {
            return false;
        }
        self.additive_value_micros
            .checked_add(self.conservative_interaction_value_micros)
            == Some(self.total_conservative_value_micros)
    }

    /// A self-consistent rehash is not enough: the selected terms must exactly
    /// match the separately verified, calibrated interaction proposal.
    pub fn is_admissible_with(&self, verified: &VerifiedPortfolioInteractionSetV1) -> bool {
        if !self.is_canonical()
            || self.proposal_sha256 != verified.proposal.proposal_sha256
            || self.calibration_receipt_sha256 != verified.calibration.receipt_sha256
        {
            return false;
        }
        let estimates: BTreeMap<_, _> = verified
            .proposal
            .pairs
            .iter()
            .map(|pair| {
                (
                    (
                        pair.left_option_sha256.as_str(),
                        pair.right_option_sha256.as_str(),
                    ),
                    pair,
                )
            })
            .collect();
        self.selected_interactions.iter().all(|selected| {
            estimates
                .get(&(
                    selected.left_option_sha256.as_str(),
                    selected.right_option_sha256.as_str(),
                ))
                .is_some_and(|estimate| {
                    estimate.interaction_micros == selected.predicted_interaction_micros
                        && estimate.absolute_error_bound_micros
                            == selected.absolute_error_bound_micros
                        && estimate
                            .conservative_value_micros()
                            .ok()
                            == Some(selected.conservative_interaction_micros)
                })
        })
    }
}

#[derive(Clone, Debug, Default)]
struct InteractionSearchState {
    metrics: MemoryPortfolioMetricsV1,
    additive_value_micros: i64,
    interaction_value_micros: i64,
    selected_indices: Vec<usize>,
    selected_interactions: Vec<SelectedPortfolioInteractionV1>,
}

impl InteractionSearchState {
    fn objective_value(&self) -> Result<i64, PortfolioInteractionReject> {
        self.additive_value_micros
            .checked_add(self.interaction_value_micros)
            .ok_or(PortfolioInteractionReject::ValueOverflow)
    }
}

#[derive(Clone)]
struct InteractionSearchBounds {
    singleton_suffix: Vec<i64>,
    option_group_max: Vec<Vec<i64>>,
    group_pair_max: Vec<Vec<i64>>,
}

fn make_interaction_bounds(
    prepared: &PreparedPortfolio,
    options: &[MemoryPortfolioOptionV1],
    pairs: &BTreeMap<(String, String), PairwiseInteractionEstimateV1>,
) -> Result<InteractionSearchBounds, PortfolioInteractionReject> {
    let group_count = prepared.groups.len();
    let mut singleton_suffix = vec![0_i64; group_count + 1];
    for group in (0..group_count).rev() {
        let maximum = prepared.groups[group]
            .iter()
            .map(|index| options[*index].net_value_micros)
            .max()
            .unwrap_or(0)
            .max(0);
        singleton_suffix[group] = singleton_suffix[group + 1]
            .checked_add(maximum)
            .ok_or(PortfolioInteractionReject::ValueOverflow)?;
    }

    let mut option_group_max = vec![vec![0_i64; group_count]; options.len()];
    let mut group_pair_max = vec![vec![0_i64; group_count]; group_count];
    for left_group in 0..group_count {
        for right_group in (left_group + 1)..group_count {
            let mut maximum = 0_i64;
            for left_index in &prepared.groups[left_group] {
                for right_index in &prepared.groups[right_group] {
                    let key = pair_key(
                        &options[*left_index].option_sha256,
                        &options[*right_index].option_sha256,
                    );
                    let estimate = pairs
                        .get(&key)
                        .ok_or(PortfolioInteractionReject::IncompletePairMatrix)?;
                    let value = estimate.conservative_value_micros()?;
                    maximum = maximum.max(value).max(0);
                    option_group_max[*left_index][right_group] =
                        option_group_max[*left_index][right_group].max(value).max(0);
                    option_group_max[*right_index][left_group] =
                        option_group_max[*right_index][left_group].max(value).max(0);
                }
            }
            group_pair_max[left_group][right_group] = maximum;
        }
    }

    Ok(InteractionSearchBounds {
        singleton_suffix,
        option_group_max,
        group_pair_max,
    })
}

fn interaction_upper_bound(
    state: &InteractionSearchState,
    next_group: usize,
    prepared: &PreparedPortfolio,
    bounds: &InteractionSearchBounds,
) -> Result<i64, PortfolioInteractionReject> {
    let mut upper = state.objective_value()?;
    upper = upper
        .checked_add(bounds.singleton_suffix[next_group])
        .ok_or(PortfolioInteractionReject::ValueOverflow)?;

    for selected in &state.selected_indices {
        for future_group in next_group..prepared.groups.len() {
            upper = upper
                .checked_add(bounds.option_group_max[*selected][future_group])
                .ok_or(PortfolioInteractionReject::ValueOverflow)?;
        }
    }
    for left_group in next_group..prepared.groups.len() {
        for right_group in (left_group + 1)..prepared.groups.len() {
            upper = upper
                .checked_add(bounds.group_pair_max[left_group][right_group])
                .ok_or(PortfolioInteractionReject::ValueOverflow)?;
        }
    }
    Ok(upper)
}

fn is_better_interaction_state(
    candidate: &InteractionSearchState,
    incumbent: &InteractionSearchState,
    options: &[MemoryPortfolioOptionV1],
) -> Result<bool, PortfolioInteractionReject> {
    let candidate_value = candidate.objective_value()?;
    let incumbent_value = incumbent.objective_value()?;
    if candidate_value != incumbent_value {
        return Ok(candidate_value > incumbent_value);
    }
    let candidate_resources = (
        candidate.metrics.context_tokens,
        candidate.metrics.reobservation_calls,
        candidate.metrics.latency_work_ms,
        candidate.metrics.estimated_cost_micros,
        candidate.metrics.expected_failure_loss_micros,
    );
    let incumbent_resources = (
        incumbent.metrics.context_tokens,
        incumbent.metrics.reobservation_calls,
        incumbent.metrics.latency_work_ms,
        incumbent.metrics.estimated_cost_micros,
        incumbent.metrics.expected_failure_loss_micros,
    );
    Ok(candidate_resources < incumbent_resources
        || (candidate_resources == incumbent_resources
            && selected_key(&candidate.selected_indices, options)
                < selected_key(&incumbent.selected_indices, options)))
}

fn search_interaction_exact(
    group_index: usize,
    prepared: &PreparedPortfolio,
    options: &[MemoryPortfolioOptionV1],
    pairs: &BTreeMap<(String, String), PairwiseInteractionEstimateV1>,
    bounds: &InteractionSearchBounds,
    budget: MemoryPortfolioBudgetV1,
    state: &mut InteractionSearchState,
    best: &mut InteractionSearchState,
    nodes: &mut u64,
    max_nodes: u64,
) -> Result<(), PortfolioInteractionReject> {
    *nodes = nodes.saturating_add(1);
    if *nodes > max_nodes {
        return Err(PortfolioInteractionReject::SearchLimitExceeded);
    }
    if interaction_upper_bound(state, group_index, prepared, bounds)?
        < best.objective_value()?
    {
        return Ok(());
    }
    if group_index == prepared.groups.len() {
        if is_better_interaction_state(state, best, options)? {
            *best = state.clone();
        }
        return Ok(());
    }

    // Discard this event.
    search_interaction_exact(
        group_index + 1,
        prepared,
        options,
        pairs,
        bounds,
        budget,
        state,
        best,
        nodes,
        max_nodes,
    )?;

    // Select at most one representation for the event.
    for index in &prepared.groups[group_index] {
        let option = &options[*index];
        let Some(next_metrics) = state.metrics.add_option(option, budget) else {
            continue;
        };

        let mut next_interaction = state.interaction_value_micros;
        let mut added_pairs = Vec::new();
        for prior_index in &state.selected_indices {
            let prior = &options[*prior_index];
            let key = pair_key(&prior.option_sha256, &option.option_sha256);
            let estimate = pairs
                .get(&key)
                .ok_or(PortfolioInteractionReject::IncompletePairMatrix)?;
            let conservative = estimate.conservative_value_micros()?;
            next_interaction = next_interaction
                .checked_add(conservative)
                .ok_or(PortfolioInteractionReject::ValueOverflow)?;

            let (left, right) = if prior.option_sha256 < option.option_sha256 {
                (prior, option)
            } else {
                (option, prior)
            };
            added_pairs.push(SelectedPortfolioInteractionV1 {
                left_event_id: left.event_id,
                right_event_id: right.event_id,
                left_option_sha256: left.option_sha256.clone(),
                right_option_sha256: right.option_sha256.clone(),
                predicted_interaction_micros: estimate.interaction_micros,
                absolute_error_bound_micros: estimate.absolute_error_bound_micros,
                conservative_interaction_micros: conservative,
            });
        }

        let next_additive = state
            .additive_value_micros
            .checked_add(option.net_value_micros)
            .ok_or(PortfolioInteractionReject::ValueOverflow)?;
        let prior_metrics = state.metrics;
        let prior_additive = state.additive_value_micros;
        let prior_interaction = state.interaction_value_micros;
        let prior_selected_len = state.selected_indices.len();
        let prior_pair_len = state.selected_interactions.len();

        state.metrics = next_metrics;
        state.additive_value_micros = next_additive;
        state.interaction_value_micros = next_interaction;
        state.selected_indices.push(*index);
        state.selected_interactions.extend(added_pairs);

        search_interaction_exact(
            group_index + 1,
            prepared,
            options,
            pairs,
            bounds,
            budget,
            state,
            best,
            nodes,
            max_nodes,
        )?;

        state.selected_indices.truncate(prior_selected_len);
        state.selected_interactions.truncate(prior_pair_len);
        state.metrics = prior_metrics;
        state.additive_value_micros = prior_additive;
        state.interaction_value_micros = prior_interaction;
    }
    Ok(())
}

/// Optimize a complete calibrated pairwise interaction model. Unknown pairs
/// are rejected during verification, not silently assumed to be zero.
pub fn optimize_memory_portfolio_pairwise(
    options: &[MemoryPortfolioOptionV1],
    budget: MemoryPortfolioBudgetV1,
    verified: &VerifiedPortfolioInteractionSetV1,
) -> Result<InteractionAwarePortfolioPlanV1, PortfolioInteractionReject> {
    optimize_memory_portfolio_pairwise_with_limit(
        options,
        budget,
        verified,
        DEFAULT_MAX_SEARCH_NODES,
    )
}

pub fn optimize_memory_portfolio_pairwise_with_limit(
    options: &[MemoryPortfolioOptionV1],
    budget: MemoryPortfolioBudgetV1,
    verified: &VerifiedPortfolioInteractionSetV1,
    max_search_nodes: u64,
) -> Result<InteractionAwarePortfolioPlanV1, PortfolioInteractionReject> {
    if max_search_nodes == 0 {
        return Err(PortfolioInteractionReject::SearchLimitExceeded);
    }
    let prepared = prepare_options(options, budget)
        .map_err(PortfolioInteractionReject::Portfolio)?;
    if option_set_digest(options) != verified.proposal.option_set_sha256
        || prepared.scope != verified.proposal.scope
        || prepared.as_known != verified.proposal.as_known
        || budget != verified.proposal.budget
    {
        return Err(PortfolioInteractionReject::ContextMismatch);
    }

    let pair_map: BTreeMap<_, _> = verified
        .proposal
        .pairs
        .iter()
        .map(|pair| {
            (
                (
                    pair.left_option_sha256.clone(),
                    pair.right_option_sha256.clone(),
                ),
                pair.clone(),
            )
        })
        .collect();
    let bounds = make_interaction_bounds(&prepared, options, &pair_map)?;
    let mut state = InteractionSearchState::default();
    let mut best = InteractionSearchState::default();
    let mut nodes = 0_u64;
    search_interaction_exact(
        0,
        &prepared,
        options,
        &pair_map,
        &bounds,
        budget,
        &mut state,
        &mut best,
        &mut nodes,
        max_search_nodes,
    )?;

    let mut selection = make_plan(
        PortfolioStrategyV1::PairwiseInteractionAware,
        prepared,
        options,
        &best.selected_indices,
        best.metrics,
        best.additive_value_micros,
        budget,
        nodes,
    );
    selection.search_nodes = nodes;
    selection.plan_sha256 = selection.canonical_digest();

    best.selected_interactions.sort_by(|left, right| {
        left.left_option_sha256
            .cmp(&right.left_option_sha256)
            .then_with(|| left.right_option_sha256.cmp(&right.right_option_sha256))
    });
    let total = best.objective_value()?;
    let mut plan = InteractionAwarePortfolioPlanV1 {
        selection,
        proposal_sha256: verified.proposal.proposal_sha256.clone(),
        calibration_receipt_sha256: verified.calibration.receipt_sha256.clone(),
        additive_value_micros: best.additive_value_micros,
        conservative_interaction_value_micros: best.interaction_value_micros,
        total_conservative_value_micros: total,
        selected_interactions: best.selected_interactions,
        plan_sha256: String::new(),
    };
    plan.plan_sha256 = plan.canonical_digest();
    if !plan.is_admissible_with(verified) {
        return Err(PortfolioInteractionReject::InvalidProposalDigest);
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use broca_mace_policy::{Objective, Regime};

    fn scope() -> Scope {
        Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        }
    }

    fn digest(value: &str) -> String {
        sha256_hex(value.as_bytes())
    }

    fn option(event_id: u32, value: i64) -> MemoryPortfolioOptionV1 {
        let mut result = MemoryPortfolioOptionV1 {
            event_id,
            evidence_sha256: digest(&format!("evidence-{event_id}")),
            profile_sha256: digest(&format!("profile-{event_id}")),
            plan_sha256: digest(&format!("plan-{event_id}")),
            option_sha256: String::new(),
            scope: scope(),
            as_known: AsKnownQuery {
                valid_at_ms: 100,
                known_at_ms: 300,
            },
            authority_scope_sha256: digest("authority"),
            disposition: ReobservationDispositionV1::PersistKnowledge,
            context_tokens: 4,
            reobservation_calls: 0,
            latency_work_ms: 0,
            estimated_cost_micros: 1,
            expected_failure_loss_micros: 0,
            net_value_micros: value,
        };
        result.option_sha256 = result.canonical_digest();
        result
    }

    fn budget() -> MemoryPortfolioBudgetV1 {
        MemoryPortfolioBudgetV1 {
            context_tokens: 32,
            reobservation_calls: 4,
            latency_work_ms: 10_000,
            estimated_cost_micros: 100_000,
            expected_failure_loss_micros: 100_000,
        }
    }

    fn context() -> PortfolioInteractionContextV1 {
        PortfolioInteractionContextV1 {
            scope: scope(),
            as_known: AsKnownQuery {
                valid_at_ms: 100,
                known_at_ms: 300,
            },
            budget: budget(),
            target_model_sha256: digest("target-model"),
            baseline_policy_sha256: digest("no-memory-policy"),
            task_protocol_sha256: digest("task-protocol"),
        }
    }

    fn proposal(
        options: &[MemoryPortfolioOptionV1],
        interaction: i64,
        error: u64,
    ) -> PortfolioInteractionProposalV1 {
        let mut pairs = Vec::new();
        for left in 0..options.len() {
            for right in (left + 1)..options.len() {
                if options[left].event_id == options[right].event_id {
                    continue;
                }
                let (left_option_sha256, right_option_sha256) =
                    pair_key(&options[left].option_sha256, &options[right].option_sha256);
                pairs.push(PairwiseInteractionEstimateV1 {
                    left_option_sha256,
                    right_option_sha256,
                    interaction_micros: interaction,
                    absolute_error_bound_micros: error,
                });
            }
        }
        pairs.sort_by(|left, right| {
            left.left_option_sha256
                .cmp(&right.left_option_sha256)
                .then_with(|| left.right_option_sha256.cmp(&right.right_option_sha256))
        });
        let ctx = context();
        let mut result = PortfolioInteractionProposalV1 {
            scope: ctx.scope,
            as_known: ctx.as_known,
            budget: ctx.budget,
            target_model_sha256: ctx.target_model_sha256.clone(),
            baseline_policy_sha256: ctx.baseline_policy_sha256.clone(),
            task_protocol_sha256: ctx.task_protocol_sha256.clone(),
            interaction_model_sha256: digest("pairwise-model"),
            option_set_sha256: option_set_digest(options),
            training_dataset_sha256: digest("training"),
            calibration_dataset_sha256: digest("calibration"),
            heldout_evaluation_sha256: digest("heldout"),
            task_family_split_sha256: digest("task-families"),
            source_root_split_sha256: digest("source-roots"),
            pairs,
            proposal_sha256: String::new(),
        };
        result.proposal_sha256 = result.canonical_digest();
        result
    }

    fn receipt(
        proposal: &PortfolioInteractionProposalV1,
    ) -> PortfolioInteractionCalibrationReceiptV1 {
        let mut result = PortfolioInteractionCalibrationReceiptV1 {
            proposal_sha256: proposal.proposal_sha256.clone(),
            target_model_sha256: proposal.target_model_sha256.clone(),
            baseline_policy_sha256: proposal.baseline_policy_sha256.clone(),
            task_protocol_sha256: proposal.task_protocol_sha256.clone(),
            interaction_model_sha256: proposal.interaction_model_sha256.clone(),
            training_dataset_sha256: proposal.training_dataset_sha256.clone(),
            calibration_dataset_sha256: proposal.calibration_dataset_sha256.clone(),
            heldout_evaluation_sha256: proposal.heldout_evaluation_sha256.clone(),
            task_family_split_sha256: proposal.task_family_split_sha256.clone(),
            source_root_split_sha256: proposal.source_root_split_sha256.clone(),
            training_pair_count: 256,
            calibration_pair_count: 128,
            heldout_pair_count: 128,
            independent_seed_count: 5,
            task_families_disjoint: true,
            source_roots_disjoint: true,
            calibration_mae_micros: 80,
            base_rate_mae_micros: 120,
            heldout_mae_micros: 85,
            heldout_base_rate_mae_micros: 120,
            heldout_mae_improvement_lower_95_micros: 12,
            heldout_multi_event_sample_count: 128,
            heldout_higher_order_residual_micros: 2_500,
            maximum_allowed_higher_order_residual_micros: 5_000,
            verified_at_ms: 250,
            valid_until_ms: 500,
            verifier_id: "fixture-only".to_owned(),
            receipt_sha256: String::new(),
        };
        result.receipt_sha256 = result.canonical_digest();
        result
    }

    struct FixtureVerifier {
        receipt: PortfolioInteractionCalibrationReceiptV1,
    }

    impl PortfolioInteractionVerifierV1 for FixtureVerifier {
        fn verify(
            &self,
            _proposal: &PortfolioInteractionProposalV1,
        ) -> Option<PortfolioInteractionCalibrationReceiptV1> {
            Some(self.receipt.clone())
        }
    }

    fn verify(
        options: &[MemoryPortfolioOptionV1],
        interaction: i64,
        error: u64,
    ) -> VerifiedPortfolioInteractionSetV1 {
        let proposal = proposal(options, interaction, error);
        let verifier = FixtureVerifier {
            receipt: receipt(&proposal),
        };
        verify_portfolio_interactions(proposal, options, &context(), &verifier).unwrap()
    }

    #[test]
    fn redundant_memories_are_not_double_valued() {
        let options = vec![option(1, 100), option(2, 100)];
        let verified = verify(&options, -150, 0);
        let additive = optimize_memory_portfolio_exact(&options, budget()).unwrap();
        let pairwise = optimize_memory_portfolio_pairwise(&options, budget(), &verified).unwrap();

        assert_eq!(additive.total_net_value_micros, 200);
        assert_eq!(pairwise.total_conservative_value_micros, 100);
        assert_eq!(pairwise.selection.selected.len(), 1);
        assert!(pairwise.is_admissible_with(&verified));
    }

    #[test]
    fn complementary_memories_capture_joint_value() {
        let options = vec![option(1, 20), option(2, 20)];
        let verified = verify(&options, 100, 10);
        let pairwise = optimize_memory_portfolio_pairwise(&options, budget(), &verified).unwrap();

        assert_eq!(pairwise.additive_value_micros, 40);
        assert_eq!(pairwise.conservative_interaction_value_micros, 90);
        assert_eq!(pairwise.total_conservative_value_micros, 130);
        assert_eq!(pairwise.selection.selected.len(), 2);
    }

    #[test]
    fn positive_point_interaction_is_reduced_by_its_error_bound() {
        let options = vec![option(1, 5), option(2, 5)];
        let verified = verify(&options, 20, 25);
        let pairwise = optimize_memory_portfolio_pairwise(&options, budget(), &verified).unwrap();

        assert_eq!(pairwise.total_conservative_value_micros, 5);
        assert_eq!(pairwise.selection.selected.len(), 1);
    }

    #[test]
    fn missing_pair_estimate_is_not_assumed_to_be_zero() {
        let options = vec![option(1, 10), option(2, 10), option(3, 10)];
        let mut proposal = proposal(&options, 0, 0);
        proposal.pairs.pop();
        proposal.proposal_sha256 = proposal.canonical_digest();
        let verifier = FixtureVerifier {
            receipt: receipt(&proposal),
        };
        assert_eq!(
            verify_portfolio_interactions(proposal, &options, &context(), &verifier),
            Err(PortfolioInteractionReject::IncompletePairMatrix)
        );
    }

    #[test]
    fn calibration_split_overlap_or_weak_model_is_rejected() {
        let options = vec![option(1, 10), option(2, 10)];
        let proposal = proposal(&options, 0, 0);
        let mut bad = receipt(&proposal);
        bad.calibration_dataset_sha256 = bad.training_dataset_sha256.clone();
        bad.receipt_sha256 = bad.canonical_digest();
        let verifier = FixtureVerifier { receipt: bad };
        assert_eq!(
            verify_portfolio_interactions(proposal.clone(), &options, &context(), &verifier),
            Err(PortfolioInteractionReject::CalibrationDatasetOverlap)
        );

        let mut weak = receipt(&proposal);
        weak.calibration_mae_micros = weak.base_rate_mae_micros;
        weak.receipt_sha256 = weak.canonical_digest();
        let verifier = FixtureVerifier {
            receipt: weak.clone(),
        };
        assert_eq!(
            verify_portfolio_interactions(
                proposal.clone(),
                &options,
                &context(),
                &verifier,
            ),
            Err(PortfolioInteractionReject::CalibrationQualityInsufficient)
        );

        let mut weak_heldout = receipt(&proposal);
        weak_heldout.heldout_mae_improvement_lower_95_micros = 0;
        weak_heldout.receipt_sha256 = weak_heldout.canonical_digest();
        let verifier = FixtureVerifier {
            receipt: weak_heldout,
        };
        assert_eq!(
            verify_portfolio_interactions(
                proposal,
                &options,
                &context(),
                &verifier,
            ),
            Err(PortfolioInteractionReject::CalibrationQualityInsufficient)
        );
    }

    #[test]
    fn higher_order_residual_above_preregistered_margin_rejects_pairwise_model() {
        let options = vec![option(1, 10), option(2, 10), option(3, 10)];
        let proposal = proposal(&options, 10, 1);
        let mut bad = receipt(&proposal);
        bad.heldout_higher_order_residual_micros = 5_001;
        bad.receipt_sha256 = bad.canonical_digest();
        let verifier = FixtureVerifier { receipt: bad };
        assert_eq!(
            verify_portfolio_interactions(proposal, &options, &context(), &verifier),
            Err(PortfolioInteractionReject::CalibrationQualityInsufficient)
        );
    }

    #[test]
    fn three_event_portfolio_requires_multi_event_holdout_samples() {
        let options = vec![option(1, 10), option(2, 10), option(3, 10)];
        let proposal = proposal(&options, 10, 1);
        let mut insufficient = receipt(&proposal);
        insufficient.heldout_multi_event_sample_count = 31;
        insufficient.receipt_sha256 = insufficient.canonical_digest();
        let verifier = FixtureVerifier {
            receipt: insufficient,
        };
        assert_eq!(
            verify_portfolio_interactions(proposal, &options, &context(), &verifier),
            Err(PortfolioInteractionReject::CalibrationEvidenceInsufficient)
        );
    }

    #[test]
    fn pairwise_search_limit_fails_closed() {
        let options = vec![option(1, 30), option(2, 20), option(3, 10)];
        let verified = verify(&options, 20, 0);
        assert_eq!(
            optimize_memory_portfolio_pairwise_with_limit(
                &options,
                budget(),
                &verified,
                1,
            ),
            Err(PortfolioInteractionReject::SearchLimitExceeded)
        );
    }

    #[test]
    fn rehashed_plan_cannot_substitute_a_different_verified_interaction() {
        let options = vec![option(1, 40), option(2, 30)];
        let verified = verify(&options, 80, 5);
        let mut plan =
            optimize_memory_portfolio_pairwise(&options, budget(), &verified).unwrap();
        assert!(plan.is_admissible_with(&verified));

        plan.selected_interactions[0].predicted_interaction_micros += 1;
        plan.selected_interactions[0].conservative_interaction_micros += 1;
        plan.conservative_interaction_value_micros += 1;
        plan.total_conservative_value_micros += 1;
        plan.plan_sha256 = plan.canonical_digest();

        assert!(plan.is_canonical());
        assert!(!plan.is_admissible_with(&verified));
    }

    #[test]
    fn option_order_permutation_does_not_change_interaction_plan() {
        let options = vec![option(1, 30), option(2, 20), option(3, 10)];
        let verified = verify(&options, -5, 2);
        let first = optimize_memory_portfolio_pairwise(&options, budget(), &verified).unwrap();
        let reversed: Vec<_> = options.iter().cloned().rev().collect();
        let verified_reversed = verify(&reversed, -5, 2);
        let second =
            optimize_memory_portfolio_pairwise(&reversed, budget(), &verified_reversed).unwrap();
        assert_eq!(first, second);
    }
}
