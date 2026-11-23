// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Joint allocation of verified memory representations under shared budgets.
//!
//! This module chooses among already qualified per-item representation options.
//! It does not verify sources, execute tools, write memory or grant authority.

use std::collections::{BTreeMap, BTreeSet};

use broca_mace_ingress::AsKnownQuery;
use broca_mace_policy::Scope;
use sha2::{Digest, Sha256};

use crate::reobservation::{
    ReobservationAvailabilityV1, ReobservationDispositionV1, ReobservationPlanV1,
    VerifiedReobservationProfileV1,
};
use crate::salience::VerifiedFutureUsePredictionV1;

mod interaction;
pub use interaction::{
    optimize_memory_portfolio_pairwise,
    optimize_memory_portfolio_pairwise_with_limit,
    verify_portfolio_interactions,
    InteractionAwarePortfolioPlanV1,
    PairwiseInteractionEstimateV1,
    PortfolioInteractionCalibrationReceiptV1,
    PortfolioInteractionContextV1,
    PortfolioInteractionProposalV1,
    PortfolioInteractionReject,
    PortfolioInteractionVerifierV1,
    SelectedPortfolioInteractionV1,
    VerifiedPortfolioInteractionSetV1,
};

const MAX_EVENT_GROUPS: usize = 24;
const MAX_PORTFOLIO_FRONTIER: usize = 128;
const DEFAULT_MAX_SEARCH_NODES: u64 = 2_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryPortfolioBudgetV1 {
    pub context_tokens: u32,
    pub reobservation_calls: u32,
    /// Sum of re-observation service latency, not parallel wall-clock latency.
    pub latency_work_ms: u64,
    pub estimated_cost_micros: u64,
    pub expected_failure_loss_micros: u64,
}

impl MemoryPortfolioBudgetV1 {
    pub fn is_valid(self) -> bool {
        self.context_tokens <= 65_536
            && self.reobservation_calls <= 1_024
            && self.latency_work_ms <= 86_400_000
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PortfolioStrategyV1 {
    ExactBounded,
    GreedyHighestNetValue,
    ParetoFrontier,
    PairwiseInteractionAware,
}

impl PortfolioStrategyV1 {
    fn code(self) -> &'static str {
        match self {
            Self::ExactBounded => "exact_bounded",
            Self::GreedyHighestNetValue => "greedy_highest_net_value",
            Self::ParetoFrontier => "pareto_frontier",
            Self::PairwiseInteractionAware => "pairwise_interaction_aware",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PortfolioReject {
    InvalidBudget,
    EmptyOptions,
    InvalidOptionDigest,
    InvalidPlanDigest,
    PlanProfileMismatch,
    PredictionMismatch,
    SnapshotMismatch,
    AuthorityScopeMismatch,
    InvalidPlanValue,
    InvalidRepresentationOption,
    PlanNotAdmissible,
    ReobservationProfileRejected(crate::reobservation::ReobservationReject),
    DuplicateOption,
    ConflictingEventIdentity(u32),
    DuplicateEvidenceAcrossEvents(String),
    InconsistentPortfolioScope,
    InconsistentKnowledgeSnapshot,
    InconsistentAuthorityScope,
    TooManyEventGroups,
    SearchLimitExceeded,
    FrontierLimitExceeded,
    ValueOverflow,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryPortfolioOptionV1 {
    event_id: u32,
    evidence_sha256: String,
    profile_sha256: String,
    plan_sha256: String,
    option_sha256: String,
    scope: Scope,
    as_known: AsKnownQuery,
    authority_scope_sha256: String,
    disposition: ReobservationDispositionV1,
    context_tokens: u32,
    reobservation_calls: u32,
    latency_work_ms: u64,
    estimated_cost_micros: u64,
    expected_failure_loss_micros: u64,
    net_value_micros: i64,
}

impl MemoryPortfolioOptionV1 {
    /// Construct only from the exact plan, calibrated prediction and opaque
    /// verified source profile that produced the representation decision.
    pub fn from_verified_plan(
        plan: &ReobservationPlanV1,
        prediction: &VerifiedFutureUsePredictionV1,
        profile: &VerifiedReobservationProfileV1,
        policy: &crate::reobservation::ReobservationPolicyV1,
    ) -> Result<Self, PortfolioReject> {
        let admissible = crate::reobservation::enumerate_memory_representation_options(
            prediction,
            profile,
            policy.clone(),
        )
        .map_err(PortfolioReject::ReobservationProfileRejected)?;
        if !admissible
            .iter()
            .any(|candidate| candidate.plan_sha256 == plan.plan_sha256)
        {
            return Err(PortfolioReject::PlanNotAdmissible);
        }

        let proposal = profile.proposal();
        let raw_prediction = prediction.prediction();
        let calibration = prediction.calibration_receipt();

        if !is_lower_sha256(&plan.plan_sha256)
            || plan.plan_sha256 != plan.canonical_digest()
        {
            return Err(PortfolioReject::InvalidPlanDigest);
        }
        if !is_lower_sha256(&plan.evidence_sha256)
            || !is_lower_sha256(&plan.prediction_sha256)
            || !is_lower_sha256(&plan.profile_sha256)
        {
            return Err(PortfolioReject::InvalidOptionDigest);
        }
        if plan.profile_sha256 != profile.profile_sha256()
            || plan.event_id != proposal.event_id
            || plan.evidence_sha256 != proposal.evidence_sha256
        {
            return Err(PortfolioReject::PlanProfileMismatch);
        }
        if plan.prediction_sha256 != raw_prediction.prediction_sha256
            || raw_prediction.event_id != proposal.event_id
            || raw_prediction.evidence_sha256 != proposal.evidence_sha256
            || raw_prediction.target_scope != proposal.scope
            || raw_prediction.expected_net_value_micros() != plan.expected_value_micros
            || calibration.target_scope != proposal.scope
        {
            return Err(PortfolioReject::PredictionMismatch);
        }
        if plan.as_known != proposal.as_known || calibration.as_known != proposal.as_known {
            return Err(PortfolioReject::SnapshotMismatch);
        }
        if plan.authority_scope != proposal.authority_scope
            || !plan.authority_scope.is_valid()
        {
            return Err(PortfolioReject::AuthorityScopeMismatch);
        }
        if plan.expected_failure_loss_micros < 0
            || plan.net_value_micros <= 0
            || plan.context_tokens_reserved == 0
            || plan.decision_deadline_ms < plan.as_known.known_at_ms
        {
            return Err(PortfolioReject::InvalidPlanValue);
        }

        let cost_i64 = i64::try_from(plan.estimated_cost_micros)
            .map_err(|_| PortfolioReject::InvalidPlanValue)?;
        let expected_net = plan
            .expected_value_micros
            .checked_sub(cost_i64)
            .and_then(|value| value.checked_sub(plan.expected_failure_loss_micros))
            .ok_or(PortfolioReject::InvalidPlanValue)?;
        if expected_net != plan.net_value_micros {
            return Err(PortfolioReject::InvalidPlanValue);
        }

        let (calls, latency_work_ms) = match plan.disposition {
            ReobservationDispositionV1::PersistKnowledge => {
                if plan.locator_sha256.is_some() || plan.snapshot_valid_until_ms.is_some() {
                    return Err(PortfolioReject::InvalidRepresentationOption);
                }
                (0, 0)
            }
            ReobservationDispositionV1::CacheSnapshot => {
                if plan.locator_sha256.is_some()
                    || plan.snapshot_valid_until_ms != proposal.snapshot_valid_until_ms
                    || plan.snapshot_valid_until_ms
                        .is_none_or(|expiry| expiry <= plan.decision_deadline_ms)
                {
                    return Err(PortfolioReject::InvalidRepresentationOption);
                }
                (0, 0)
            }
            ReobservationDispositionV1::StoreLocator => {
                let Some(locator) = plan.locator_sha256.as_deref() else {
                    return Err(PortfolioReject::InvalidRepresentationOption);
                };
                let ReobservationAvailabilityV1::Available {
                    available_at_ms,
                    locator_sha256,
                } = &proposal.availability
                else {
                    return Err(PortfolioReject::InvalidRepresentationOption);
                };
                if !is_lower_sha256(locator)
                    || locator != locator_sha256
                    || *available_at_ms > plan.decision_deadline_ms
                    || plan.snapshot_valid_until_ms.is_some()
                {
                    return Err(PortfolioReject::InvalidRepresentationOption);
                }
                (1, u64::from(proposal.reobservation_latency_ms))
            }
            ReobservationDispositionV1::Discard => {
                return Err(PortfolioReject::InvalidRepresentationOption);
            }
        };

        let mut option = Self {
            event_id: plan.event_id,
            evidence_sha256: plan.evidence_sha256.clone(),
            profile_sha256: plan.profile_sha256.clone(),
            plan_sha256: plan.plan_sha256.clone(),
            option_sha256: String::new(),
            scope: proposal.scope,
            as_known: plan.as_known,
            authority_scope_sha256: plan.authority_scope.canonical_digest(),
            disposition: plan.disposition,
            context_tokens: plan.context_tokens_reserved,
            reobservation_calls: calls,
            latency_work_ms,
            estimated_cost_micros: plan.estimated_cost_micros,
            expected_failure_loss_micros: plan.expected_failure_loss_micros as u64,
            net_value_micros: plan.net_value_micros,
        };
        option.option_sha256 = option.canonical_digest();
        Ok(option)
    }

    pub fn event_id(&self) -> u32 {
        self.event_id
    }

    pub fn disposition(&self) -> ReobservationDispositionV1 {
        self.disposition
    }

    pub fn net_value_micros(&self) -> i64 {
        self.net_value_micros
    }

    pub fn option_sha256(&self) -> &str {
        &self.option_sha256
    }

    pub fn authority_scope_sha256(&self) -> &str {
        &self.authority_scope_sha256
    }

    fn canonical_digest(&self) -> String {
        digest_fields(
            b"mace:memory-portfolio-option:v1\0",
            &[
                self.event_id.to_string(),
                self.evidence_sha256.clone(),
                self.profile_sha256.clone(),
                self.plan_sha256.clone(),
                scope_code(self.scope).to_owned(),
                self.as_known.valid_at_ms.to_string(),
                self.as_known.known_at_ms.to_string(),
                self.authority_scope_sha256.clone(),
                disposition_code(self.disposition).to_owned(),
                self.context_tokens.to_string(),
                self.reobservation_calls.to_string(),
                self.latency_work_ms.to_string(),
                self.estimated_cost_micros.to_string(),
                self.expected_failure_loss_micros.to_string(),
                self.net_value_micros.to_string(),
            ],
        )
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MemoryPortfolioMetricsV1 {
    pub context_tokens: u64,
    pub reobservation_calls: u64,
    pub latency_work_ms: u64,
    pub estimated_cost_micros: u64,
    pub expected_failure_loss_micros: u64,
}

impl MemoryPortfolioMetricsV1 {
    fn add_option(
        self,
        option: &MemoryPortfolioOptionV1,
        budget: MemoryPortfolioBudgetV1,
    ) -> Option<Self> {
        let next = Self {
            context_tokens: self.context_tokens.checked_add(u64::from(option.context_tokens))?,
            reobservation_calls: self
                .reobservation_calls
                .checked_add(u64::from(option.reobservation_calls))?,
            latency_work_ms: self.latency_work_ms.checked_add(option.latency_work_ms)?,
            estimated_cost_micros: self
                .estimated_cost_micros
                .checked_add(option.estimated_cost_micros)?,
            expected_failure_loss_micros: self
                .expected_failure_loss_micros
                .checked_add(option.expected_failure_loss_micros)?,
        };
        (next.context_tokens <= u64::from(budget.context_tokens)
            && next.reobservation_calls <= u64::from(budget.reobservation_calls)
            && next.latency_work_ms <= budget.latency_work_ms
            && next.estimated_cost_micros <= budget.estimated_cost_micros
            && next.expected_failure_loss_micros <= budget.expected_failure_loss_micros)
            .then_some(next)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryPortfolioSelectedItemV1 {
    pub event_id: u32,
    pub disposition: ReobservationDispositionV1,
    pub option_sha256: String,
    pub representation_plan_sha256: String,
    pub net_value_micros: i64,
    pub context_tokens: u32,
    pub reobservation_calls: u32,
    pub latency_work_ms: u64,
    pub estimated_cost_micros: u64,
    pub expected_failure_loss_micros: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryPortfolioPlanV1 {
    pub strategy: PortfolioStrategyV1,
    pub scope: Scope,
    pub as_known: AsKnownQuery,
    pub authority_scope_sha256: String,
    pub budget: MemoryPortfolioBudgetV1,
    pub selected: Vec<MemoryPortfolioSelectedItemV1>,
    pub metrics: MemoryPortfolioMetricsV1,
    pub total_net_value_micros: i64,
    pub search_nodes: u64,
    pub plan_sha256: String,
}

impl MemoryPortfolioPlanV1 {
    pub fn canonical_digest(&self) -> String {
        let mut fields = vec![
            self.strategy.code().to_owned(),
            scope_code(self.scope).to_owned(),
            self.as_known.valid_at_ms.to_string(),
            self.as_known.known_at_ms.to_string(),
            self.authority_scope_sha256.clone(),
            self.budget.context_tokens.to_string(),
            self.budget.reobservation_calls.to_string(),
            self.budget.latency_work_ms.to_string(),
            self.budget.estimated_cost_micros.to_string(),
            self.budget.expected_failure_loss_micros.to_string(),
            self.metrics.context_tokens.to_string(),
            self.metrics.reobservation_calls.to_string(),
            self.metrics.latency_work_ms.to_string(),
            self.metrics.estimated_cost_micros.to_string(),
            self.metrics.expected_failure_loss_micros.to_string(),
            self.total_net_value_micros.to_string(),
            self.search_nodes.to_string(),
            self.selected.len().to_string(),
        ];
        for item in &self.selected {
            fields.extend([
                item.event_id.to_string(),
                disposition_code(item.disposition).to_owned(),
                item.option_sha256.clone(),
                item.representation_plan_sha256.clone(),
                item.net_value_micros.to_string(),
                item.context_tokens.to_string(),
                item.reobservation_calls.to_string(),
                item.latency_work_ms.to_string(),
                item.estimated_cost_micros.to_string(),
                item.expected_failure_loss_micros.to_string(),
            ]);
        }
        digest_fields(b"mace:memory-portfolio-plan:v1\0", &fields)
    }

    pub fn is_canonical(&self) -> bool {
        if !is_lower_sha256(&self.plan_sha256)
            || self.plan_sha256 != self.canonical_digest()
            || !self.budget.is_valid()
        {
            return false;
        }
        if matches!(
            self.strategy,
            PortfolioStrategyV1::ExactBounded
                | PortfolioStrategyV1::ParetoFrontier
                | PortfolioStrategyV1::PairwiseInteractionAware
        ) && self.search_nodes == 0
        {
            return false;
        }
        if self.strategy == PortfolioStrategyV1::GreedyHighestNetValue
            && self.search_nodes != 0
        {
            return false;
        }

        let mut event_ids = BTreeSet::new();
        let mut option_digests = BTreeSet::new();
        let mut totals = MemoryPortfolioMetricsV1::default();
        let mut value = 0_i64;
        for item in &self.selected {
            if item.event_id == 0
                || item.net_value_micros <= 0
                || item.context_tokens == 0
                || !is_lower_sha256(&item.option_sha256)
                || !is_lower_sha256(&item.representation_plan_sha256)
                || !event_ids.insert(item.event_id)
                || !option_digests.insert(item.option_sha256.clone())
            {
                return false;
            }
            let Some(next_context) = totals
                .context_tokens
                .checked_add(u64::from(item.context_tokens))
            else {
                return false;
            };
            let Some(next_calls) = totals
                .reobservation_calls
                .checked_add(u64::from(item.reobservation_calls))
            else {
                return false;
            };
            let Some(next_latency) = totals.latency_work_ms.checked_add(item.latency_work_ms)
            else {
                return false;
            };
            let Some(next_cost) = totals
                .estimated_cost_micros
                .checked_add(item.estimated_cost_micros)
            else {
                return false;
            };
            let Some(next_failure) = totals
                .expected_failure_loss_micros
                .checked_add(item.expected_failure_loss_micros)
            else {
                return false;
            };
            let Some(next_value) = value.checked_add(item.net_value_micros) else {
                return false;
            };
            totals = MemoryPortfolioMetricsV1 {
                context_tokens: next_context,
                reobservation_calls: next_calls,
                latency_work_ms: next_latency,
                estimated_cost_micros: next_cost,
                expected_failure_loss_micros: next_failure,
            };
            value = next_value;
        }
        totals == self.metrics
            && value == self.total_net_value_micros
            && totals.context_tokens <= u64::from(self.budget.context_tokens)
            && totals.reobservation_calls <= u64::from(self.budget.reobservation_calls)
            && totals.latency_work_ms <= self.budget.latency_work_ms
            && totals.estimated_cost_micros <= self.budget.estimated_cost_micros
            && totals.expected_failure_loss_micros <= self.budget.expected_failure_loss_micros
    }
}

#[derive(Clone)]
struct PreparedPortfolio {
    groups: Vec<Vec<usize>>,
    scope: Scope,
    as_known: AsKnownQuery,
    authority_scope_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryPortfolioFrontierV1 {
    pub budget: MemoryPortfolioBudgetV1,
    pub plans: Vec<MemoryPortfolioPlanV1>,
    pub search_nodes: u64,
    pub frontier_sha256: String,
}

impl MemoryPortfolioFrontierV1 {
    pub fn canonical_digest(&self) -> String {
        let mut fields = vec![
            self.budget.context_tokens.to_string(),
            self.budget.reobservation_calls.to_string(),
            self.budget.latency_work_ms.to_string(),
            self.budget.estimated_cost_micros.to_string(),
            self.budget.expected_failure_loss_micros.to_string(),
            self.search_nodes.to_string(),
            self.plans.len().to_string(),
        ];
        fields.extend(self.plans.iter().map(|plan| plan.plan_sha256.clone()));
        digest_fields(b"mace:memory-portfolio-frontier:v1\0", &fields)
    }

    pub fn is_canonical(&self) -> bool {
        if !self.budget.is_valid()
            || self.plans.is_empty()
            || self.plans.len() > MAX_PORTFOLIO_FRONTIER
            || self.search_nodes == 0
            || !is_lower_sha256(&self.frontier_sha256)
            || self.frontier_sha256 != self.canonical_digest()
        {
            return false;
        }

        let Some(first) = self.plans.first() else {
            return false;
        };
        if self.plans.iter().any(|plan| {
            !plan.is_canonical()
                || plan.strategy != PortfolioStrategyV1::ParetoFrontier
                || plan.budget != self.budget
                || plan.scope != first.scope
                || plan.as_known != first.as_known
                || plan.authority_scope_sha256 != first.authority_scope_sha256
                || plan.search_nodes != self.search_nodes
        }) {
            return false;
        }
        if self
            .plans
            .windows(2)
            .any(|pair| frontier_plan_cmp(&pair[0], &pair[1]).is_gt())
        {
            return false;
        }

        for left in 0..self.plans.len() {
            for right in (left + 1)..self.plans.len() {
                if self.plans[left].plan_sha256 == self.plans[right].plan_sha256
                    || plan_dominates(&self.plans[left], &self.plans[right])
                    || plan_dominates(&self.plans[right], &self.plans[left])
                {
                    return false;
                }
            }
        }
        true
    }
}

fn prepare_options(
    options: &[MemoryPortfolioOptionV1],
    budget: MemoryPortfolioBudgetV1,
) -> Result<PreparedPortfolio, PortfolioReject> {
    if !budget.is_valid() {
        return Err(PortfolioReject::InvalidBudget);
    }
    if options.is_empty() {
        return Err(PortfolioReject::EmptyOptions);
    }

    let first = &options[0];
    let mut seen_options = BTreeSet::new();
    let mut groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    let mut event_identity: BTreeMap<u32, (String, String)> = BTreeMap::new();
    let mut evidence_event: BTreeMap<String, u32> = BTreeMap::new();

    for (index, option) in options.iter().enumerate() {
        if option.event_id == 0
            || !is_lower_sha256(&option.evidence_sha256)
            || !is_lower_sha256(&option.profile_sha256)
            || !is_lower_sha256(&option.plan_sha256)
            || !is_lower_sha256(&option.option_sha256)
            || option.option_sha256 != option.canonical_digest()
            || option.net_value_micros <= 0
        {
            return Err(PortfolioReject::InvalidOptionDigest);
        }
        if option.scope != first.scope {
            return Err(PortfolioReject::InconsistentPortfolioScope);
        }
        if option.as_known != first.as_known {
            return Err(PortfolioReject::InconsistentKnowledgeSnapshot);
        }
        if option.authority_scope_sha256 != first.authority_scope_sha256 {
            return Err(PortfolioReject::InconsistentAuthorityScope);
        }
        if !seen_options.insert(option.option_sha256.clone()) {
            return Err(PortfolioReject::DuplicateOption);
        }
        match evidence_event.entry(option.evidence_sha256.clone()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(option.event_id);
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                if *entry.get() != option.event_id {
                    return Err(PortfolioReject::DuplicateEvidenceAcrossEvents(
                        option.evidence_sha256.clone(),
                    ));
                }
            }
        }

        match event_identity.entry(option.event_id) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert((
                    option.evidence_sha256.clone(),
                    option.profile_sha256.clone(),
                ));
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                if entry.get()
                    != &(option.evidence_sha256.clone(), option.profile_sha256.clone())
                {
                    return Err(PortfolioReject::ConflictingEventIdentity(option.event_id));
                }
            }
        }
        groups.entry(option.event_id).or_default().push(index);
    }

    if groups.len() > MAX_EVENT_GROUPS {
        return Err(PortfolioReject::TooManyEventGroups);
    }
    let mut groups: Vec<Vec<usize>> = groups.into_values().collect();
    for group in &mut groups {
        group.sort_by(|left, right| {
            options[*right]
                .net_value_micros
                .cmp(&options[*left].net_value_micros)
                .then_with(|| options[*left].option_sha256.cmp(&options[*right].option_sha256))
        });
    }

    Ok(PreparedPortfolio {
        groups,
        scope: first.scope,
        as_known: first.as_known,
        authority_scope_sha256: first.authority_scope_sha256.clone(),
    })
}

#[derive(Clone, Debug, Default)]
struct SearchState {
    metrics: MemoryPortfolioMetricsV1,
    value_micros: i64,
    selected_indices: Vec<usize>,
}

fn selected_key(
    selected: &[usize],
    options: &[MemoryPortfolioOptionV1],
) -> Vec<String> {
    selected
        .iter()
        .map(|index| options[*index].option_sha256.clone())
        .collect()
}

fn is_better_state(
    candidate: &SearchState,
    incumbent: &SearchState,
    options: &[MemoryPortfolioOptionV1],
) -> bool {
    if candidate.value_micros != incumbent.value_micros {
        return candidate.value_micros > incumbent.value_micros;
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
    candidate_resources < incumbent_resources
        || (candidate_resources == incumbent_resources
            && selected_key(&candidate.selected_indices, options)
                < selected_key(&incumbent.selected_indices, options))
}

fn search_exact(
    group_index: usize,
    prepared: &PreparedPortfolio,
    options: &[MemoryPortfolioOptionV1],
    suffix_upper_bound: &[i64],
    budget: MemoryPortfolioBudgetV1,
    state: &mut SearchState,
    best: &mut SearchState,
    nodes: &mut u64,
    max_nodes: u64,
) -> Result<(), PortfolioReject> {
    *nodes = (*nodes).saturating_add(1);
    if *nodes > max_nodes {
        return Err(PortfolioReject::SearchLimitExceeded);
    }
    if state
        .value_micros
        .saturating_add(suffix_upper_bound[group_index])
        < best.value_micros
    {
        return Ok(());
    }
    if group_index == prepared.groups.len() {
        if is_better_state(state, best, options) {
            *best = state.clone();
        }
        return Ok(());
    }

    // Discard this evidence item.
    search_exact(
        group_index + 1,
        prepared,
        options,
        suffix_upper_bound,
        budget,
        state,
        best,
        nodes,
        max_nodes,
    )?;

    // Choose at most one representation of this event.
    for index in &prepared.groups[group_index] {
        let option = &options[*index];
        let Some(metrics) = state.metrics.add_option(option, budget) else {
            continue;
        };
        let next_value = state
            .value_micros
            .checked_add(option.net_value_micros)
            .ok_or(PortfolioReject::ValueOverflow)?;
        let prior_metrics = state.metrics;
        let prior_value = state.value_micros;
        state.metrics = metrics;
        state.value_micros = next_value;
        state.selected_indices.push(*index);

        search_exact(
            group_index + 1,
            prepared,
            options,
            suffix_upper_bound,
            budget,
            state,
            best,
            nodes,
            max_nodes,
        )?;

        state.selected_indices.pop();
        state.metrics = prior_metrics;
        state.value_micros = prior_value;
    }
    Ok(())
}

fn make_plan(
    strategy: PortfolioStrategyV1,
    prepared: PreparedPortfolio,
    options: &[MemoryPortfolioOptionV1],
    selected_indices: &[usize],
    metrics: MemoryPortfolioMetricsV1,
    value_micros: i64,
    budget: MemoryPortfolioBudgetV1,
    search_nodes: u64,
) -> MemoryPortfolioPlanV1 {
    let mut selected: Vec<_> = selected_indices
        .iter()
        .map(|index| {
            let option = &options[*index];
            MemoryPortfolioSelectedItemV1 {
                event_id: option.event_id,
                disposition: option.disposition,
                option_sha256: option.option_sha256.clone(),
                representation_plan_sha256: option.plan_sha256.clone(),
                net_value_micros: option.net_value_micros,
                context_tokens: option.context_tokens,
                reobservation_calls: option.reobservation_calls,
                latency_work_ms: option.latency_work_ms,
                estimated_cost_micros: option.estimated_cost_micros,
                expected_failure_loss_micros: option.expected_failure_loss_micros,
            }
        })
        .collect();
    selected.sort_by_key(|item| (item.event_id, item.option_sha256.clone()));

    let mut plan = MemoryPortfolioPlanV1 {
        strategy,
        scope: prepared.scope,
        as_known: prepared.as_known,
        authority_scope_sha256: prepared.authority_scope_sha256,
        budget,
        selected,
        metrics,
        total_net_value_micros: value_micros,
        search_nodes,
        plan_sha256: String::new(),
    };
    plan.plan_sha256 = plan.canonical_digest();
    plan
}

pub fn optimize_memory_portfolio_exact(
    options: &[MemoryPortfolioOptionV1],
    budget: MemoryPortfolioBudgetV1,
) -> Result<MemoryPortfolioPlanV1, PortfolioReject> {
    optimize_memory_portfolio_exact_with_limit(
        options,
        budget,
        DEFAULT_MAX_SEARCH_NODES,
    )
}

pub fn optimize_memory_portfolio_exact_with_limit(
    options: &[MemoryPortfolioOptionV1],
    budget: MemoryPortfolioBudgetV1,
    max_search_nodes: u64,
) -> Result<MemoryPortfolioPlanV1, PortfolioReject> {
    if max_search_nodes == 0 {
        return Err(PortfolioReject::SearchLimitExceeded);
    }
    let prepared = prepare_options(options, budget)?;
    let mut suffix_upper_bound = vec![0_i64; prepared.groups.len() + 1];
    for group_index in (0..prepared.groups.len()).rev() {
        let best_value = prepared.groups[group_index]
            .iter()
            .map(|index| options[*index].net_value_micros)
            .max()
            .unwrap_or(0)
            .max(0);
        suffix_upper_bound[group_index] = suffix_upper_bound[group_index + 1]
            .checked_add(best_value)
            .ok_or(PortfolioReject::ValueOverflow)?;
    }

    let mut state = SearchState::default();
    let mut best = SearchState::default();
    let mut nodes = 0_u64;
    search_exact(
        0,
        &prepared,
        options,
        &suffix_upper_bound,
        budget,
        &mut state,
        &mut best,
        &mut nodes,
        max_search_nodes,
    )?;

    let mut result = make_plan(
        PortfolioStrategyV1::ExactBounded,
        prepared,
        options,
        &best.selected_indices,
        best.metrics,
        best.value_micros,
        budget,
        nodes,
    );
    result.search_nodes = nodes;
    result.plan_sha256 = result.canonical_digest();
    Ok(result)
}

/// Diagnostic baseline: choose the highest individual net-value option first.
/// Unlike the exact bounded search, it is not guaranteed to find the portfolio
/// with the greatest total value.
pub fn optimize_memory_portfolio_greedy(
    options: &[MemoryPortfolioOptionV1],
    budget: MemoryPortfolioBudgetV1,
) -> Result<MemoryPortfolioPlanV1, PortfolioReject> {
    let prepared = prepare_options(options, budget)?;
    let mut indices: Vec<usize> = (0..options.len()).collect();
    indices.sort_by(|left, right| {
        options[*right]
            .net_value_micros
            .cmp(&options[*left].net_value_micros)
            .then_with(|| options[*left].event_id.cmp(&options[*right].event_id))
            .then_with(|| options[*left].option_sha256.cmp(&options[*right].option_sha256))
    });

    let mut selected = Vec::new();
    let mut selected_events = BTreeSet::new();
    let mut metrics = MemoryPortfolioMetricsV1::default();
    let mut value_micros = 0_i64;

    for index in indices {
        let option = &options[index];
        if selected_events.contains(&option.event_id) {
            continue;
        }
        let Some(next_metrics) = metrics.add_option(option, budget) else {
            continue;
        };
        value_micros = value_micros
            .checked_add(option.net_value_micros)
            .ok_or(PortfolioReject::ValueOverflow)?;
        metrics = next_metrics;
        selected_events.insert(option.event_id);
        selected.push(index);
    }

    Ok(make_plan(
        PortfolioStrategyV1::GreedyHighestNetValue,
        prepared,
        options,
        &selected,
        metrics,
        value_micros,
        budget,
        0,
    ))
}

fn resource_vector(
    metrics: MemoryPortfolioMetricsV1,
) -> (u64, u64, u64, u64, u64) {
    (
        metrics.context_tokens,
        metrics.reobservation_calls,
        metrics.latency_work_ms,
        metrics.estimated_cost_micros,
        metrics.expected_failure_loss_micros,
    )
}

fn plan_dominates(
    left: &MemoryPortfolioPlanV1,
    right: &MemoryPortfolioPlanV1,
) -> bool {
    let left_resources = resource_vector(left.metrics);
    let right_resources = resource_vector(right.metrics);
    let weakly_better = left.total_net_value_micros >= right.total_net_value_micros
        && left_resources.0 <= right_resources.0
        && left_resources.1 <= right_resources.1
        && left_resources.2 <= right_resources.2
        && left_resources.3 <= right_resources.3
        && left_resources.4 <= right_resources.4;
    let strictly_better = left.total_net_value_micros > right.total_net_value_micros
        || left_resources != right_resources;
    weakly_better && strictly_better
}

fn plan_selected_key(plan: &MemoryPortfolioPlanV1) -> Vec<String> {
    plan.selected
        .iter()
        .map(|item| item.option_sha256.clone())
        .collect()
}

fn insert_frontier_plan(
    frontier: &mut Vec<MemoryPortfolioPlanV1>,
    candidate: MemoryPortfolioPlanV1,
    max_frontier: usize,
) -> Result<(), PortfolioReject> {
    for existing in frontier.iter() {
        if plan_dominates(existing, &candidate) {
            return Ok(());
        }
        let same_objective = existing.total_net_value_micros
            == candidate.total_net_value_micros
            && existing.metrics == candidate.metrics;
        if same_objective && plan_selected_key(existing) <= plan_selected_key(&candidate) {
            return Ok(());
        }
    }

    frontier.retain(|existing| {
        let same_objective = existing.total_net_value_micros
            == candidate.total_net_value_micros
            && existing.metrics == candidate.metrics;
        !plan_dominates(&candidate, existing)
            && !(same_objective
                && plan_selected_key(&candidate) < plan_selected_key(existing))
    });
    frontier.push(candidate);
    if frontier.len() > max_frontier {
        return Err(PortfolioReject::FrontierLimitExceeded);
    }
    Ok(())
}

fn search_frontier(
    group_index: usize,
    prepared: &PreparedPortfolio,
    options: &[MemoryPortfolioOptionV1],
    budget: MemoryPortfolioBudgetV1,
    state: &mut SearchState,
    frontier: &mut Vec<MemoryPortfolioPlanV1>,
    nodes: &mut u64,
    max_nodes: u64,
    max_frontier: usize,
) -> Result<(), PortfolioReject> {
    *nodes = (*nodes).saturating_add(1);
    if *nodes > max_nodes {
        return Err(PortfolioReject::SearchLimitExceeded);
    }
    if group_index == prepared.groups.len() {
        let plan = make_plan(
            PortfolioStrategyV1::ParetoFrontier,
            prepared.clone(),
            options,
            &state.selected_indices,
            state.metrics,
            state.value_micros,
            budget,
            *nodes,
        );
        return insert_frontier_plan(frontier, plan, max_frontier);
    }

    // Discard this evidence item.
    search_frontier(
        group_index + 1,
        prepared,
        options,
        budget,
        state,
        frontier,
        nodes,
        max_nodes,
        max_frontier,
    )?;

    // Choose at most one representation of this event.
    for index in &prepared.groups[group_index] {
        let option = &options[*index];
        let Some(metrics) = state.metrics.add_option(option, budget) else {
            continue;
        };
        let next_value = state
            .value_micros
            .checked_add(option.net_value_micros)
            .ok_or(PortfolioReject::ValueOverflow)?;
        let prior_metrics = state.metrics;
        let prior_value = state.value_micros;
        state.metrics = metrics;
        state.value_micros = next_value;
        state.selected_indices.push(*index);

        search_frontier(
            group_index + 1,
            prepared,
            options,
            budget,
            state,
            frontier,
            nodes,
            max_nodes,
            max_frontier,
        )?;

        state.selected_indices.pop();
        state.metrics = prior_metrics;
        state.value_micros = prior_value;
    }
    Ok(())
}

fn frontier_plan_cmp(
    left: &MemoryPortfolioPlanV1,
    right: &MemoryPortfolioPlanV1,
) -> std::cmp::Ordering {
    right
        .total_net_value_micros
        .cmp(&left.total_net_value_micros)
        .then_with(|| resource_vector(left.metrics).cmp(&resource_vector(right.metrics)))
        .then_with(|| plan_selected_key(left).cmp(&plan_selected_key(right)))
}

fn sort_frontier(plans: &mut [MemoryPortfolioPlanV1]) {
    plans.sort_by(frontier_plan_cmp);
}

pub fn optimize_memory_portfolio_frontier(
    options: &[MemoryPortfolioOptionV1],
    budget: MemoryPortfolioBudgetV1,
) -> Result<MemoryPortfolioFrontierV1, PortfolioReject> {
    optimize_memory_portfolio_frontier_with_limits(
        options,
        budget,
        DEFAULT_MAX_SEARCH_NODES,
        MAX_PORTFOLIO_FRONTIER,
    )
}

/// Enumerate every nondominated feasible portfolio within explicit search and
/// frontier bounds. Any bound exhaustion rejects the whole result; no partial
/// frontier is returned.
pub fn optimize_memory_portfolio_frontier_with_limits(
    options: &[MemoryPortfolioOptionV1],
    budget: MemoryPortfolioBudgetV1,
    max_search_nodes: u64,
    max_frontier: usize,
) -> Result<MemoryPortfolioFrontierV1, PortfolioReject> {
    if max_search_nodes == 0 {
        return Err(PortfolioReject::SearchLimitExceeded);
    }
    if max_frontier == 0 {
        return Err(PortfolioReject::FrontierLimitExceeded);
    }
    let prepared = prepare_options(options, budget)?;
    let mut state = SearchState::default();
    let mut frontier = Vec::new();
    let mut nodes = 0_u64;
    search_frontier(
        0,
        &prepared,
        options,
        budget,
        &mut state,
        &mut frontier,
        &mut nodes,
        max_search_nodes,
        max_frontier,
    )?;
    for plan in &mut frontier {
        plan.search_nodes = nodes;
        plan.plan_sha256 = plan.canonical_digest();
    }
    sort_frontier(&mut frontier);

    let mut result = MemoryPortfolioFrontierV1 {
        budget,
        plans: frontier,
        search_nodes: nodes,
        frontier_sha256: String::new(),
    };
    result.frontier_sha256 = result.canonical_digest();
    Ok(result)
}

fn disposition_code(disposition: ReobservationDispositionV1) -> &'static str {
    match disposition {
        ReobservationDispositionV1::PersistKnowledge => "persist_knowledge",
        ReobservationDispositionV1::StoreLocator => "store_locator",
        ReobservationDispositionV1::CacheSnapshot => "cache_snapshot",
        ReobservationDispositionV1::Discard => "discard",
    }
}

fn scope_code(scope: Scope) -> &'static str {
    use broca_mace_policy::{Objective, Regime};
    match (scope.regime, scope.objective) {
        (Regime::Baseline, Objective::Quality) => "baseline:quality",
        (Regime::Baseline, Objective::Cost) => "baseline:cost",
        (Regime::Baseline, Objective::Speed) => "baseline:speed",
        (Regime::Shifted, Objective::Quality) => "shifted:quality",
        (Regime::Shifted, Objective::Cost) => "shifted:cost",
        (Regime::Shifted, Objective::Speed) => "shifted:speed",
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest_fields(domain: &[u8], fields: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for field in fields {
        hasher.update((field.len() as u64).to_be_bytes());
        hasher.update(field.as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
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

    fn option(
        event_id: u32,
        label: &str,
        disposition: ReobservationDispositionV1,
        net_value_micros: i64,
        context_tokens: u32,
        reobservation_calls: u32,
        latency_work_ms: u64,
        estimated_cost_micros: u64,
        expected_failure_loss_micros: u64,
    ) -> MemoryPortfolioOptionV1 {
        let mut result = MemoryPortfolioOptionV1 {
            event_id,
            evidence_sha256: digest(&format!("evidence-{event_id}")),
            profile_sha256: digest(&format!("profile-{event_id}")),
            plan_sha256: digest(&format!("plan-{event_id}-{label}")),
            option_sha256: String::new(),
            scope: scope(),
            as_known: AsKnownQuery {
                valid_at_ms: 100,
                known_at_ms: 300,
            },
            authority_scope_sha256: digest("authority"),
            disposition,
            context_tokens,
            reobservation_calls,
            latency_work_ms,
            estimated_cost_micros,
            expected_failure_loss_micros,
            net_value_micros,
        };
        result.option_sha256 = result.canonical_digest();
        result
    }

    fn budget(
        tokens: u32,
        calls: u32,
        latency: u64,
        cost: u64,
        risk: u64,
    ) -> MemoryPortfolioBudgetV1 {
        MemoryPortfolioBudgetV1 {
            context_tokens: tokens,
            reobservation_calls: calls,
            latency_work_ms: latency,
            estimated_cost_micros: cost,
            expected_failure_loss_micros: risk,
        }
    }

    #[test]
    fn exact_joint_allocation_beats_highest_value_greedy_under_token_budget() {
        let options = vec![
            option(
                1,
                "large",
                ReobservationDispositionV1::PersistKnowledge,
                100,
                10,
                0,
                0,
                10,
                0,
            ),
            option(
                2,
                "small-a",
                ReobservationDispositionV1::StoreLocator,
                60,
                6,
                1,
                10,
                10,
                1,
            ),
            option(
                3,
                "small-b",
                ReobservationDispositionV1::StoreLocator,
                60,
                6,
                1,
                10,
                10,
                1,
            ),
        ];
        let budget = budget(12, 2, 20, 100, 10);
        let exact = optimize_memory_portfolio_exact(&options, budget).unwrap();
        let greedy = optimize_memory_portfolio_greedy(&options, budget).unwrap();

        assert!(exact.is_canonical());
        assert!(greedy.is_canonical());
        assert_eq!(exact.total_net_value_micros, 120);
        assert_eq!(greedy.total_net_value_micros, 100);
        assert_eq!(
            exact.selected.iter().map(|item| item.event_id).collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn rehashed_plan_with_forged_resource_totals_is_not_canonical() {
        let options = vec![option(
            1,
            "one",
            ReobservationDispositionV1::PersistKnowledge,
            50,
            4,
            0,
            0,
            5,
            1,
        )];
        let mut plan = optimize_memory_portfolio_exact(
            &options,
            budget(8, 1, 10, 20, 5),
        )
        .unwrap();
        assert!(plan.is_canonical());

        plan.metrics.estimated_cost_micros = 0;
        plan.plan_sha256 = plan.canonical_digest();
        assert!(!plan.is_canonical());
    }

    #[test]
    fn only_one_representation_of_each_event_can_be_selected() {
        let options = vec![
            option(
                7,
                "persist",
                ReobservationDispositionV1::PersistKnowledge,
                100,
                8,
                0,
                0,
                10,
                0,
            ),
            option(
                7,
                "locator",
                ReobservationDispositionV1::StoreLocator,
                80,
                2,
                1,
                10,
                5,
                4,
            ),
            option(
                8,
                "other",
                ReobservationDispositionV1::CacheSnapshot,
                60,
                4,
                0,
                0,
                10,
                0,
            ),
        ];
        let plan = optimize_memory_portfolio_exact(&options, budget(12, 1, 10, 100, 10))
            .unwrap();
        assert_eq!(plan.selected.iter().filter(|item| item.event_id == 7).count(), 1);
        assert_eq!(plan.total_net_value_micros, 160);
    }

    #[test]
    fn locator_is_replaced_by_persisted_option_when_shared_call_budget_is_zero() {
        let options = vec![
            option(
                1,
                "locator",
                ReobservationDispositionV1::StoreLocator,
                110,
                1,
                1,
                20,
                10,
                5,
            ),
            option(
                1,
                "persist",
                ReobservationDispositionV1::PersistKnowledge,
                90,
                8,
                0,
                0,
                15,
                0,
            ),
        ];
        let plan = optimize_memory_portfolio_exact(&options, budget(16, 0, 0, 100, 10))
            .unwrap();
        assert_eq!(plan.selected.len(), 1);
        assert_eq!(
            plan.selected[0].disposition,
            ReobservationDispositionV1::PersistKnowledge
        );
    }

    #[test]
    fn all_shared_resource_budgets_are_enforced() {
        let options = vec![
            option(
                1,
                "risk",
                ReobservationDispositionV1::StoreLocator,
                100,
                2,
                1,
                20,
                10,
                50,
            ),
            option(
                2,
                "safe",
                ReobservationDispositionV1::CacheSnapshot,
                60,
                2,
                0,
                0,
                10,
                0,
            ),
        ];
        let plan = optimize_memory_portfolio_exact(&options, budget(10, 0, 100, 100, 0))
            .unwrap();
        assert_eq!(plan.selected.len(), 1);
        assert_eq!(plan.selected[0].event_id, 2);
        assert_eq!(plan.metrics.reobservation_calls, 0);
        assert_eq!(plan.metrics.expected_failure_loss_micros, 0);
    }

    #[test]
    fn conflicting_scope_snapshot_or_authority_is_rejected() {
        let first = option(
            1,
            "a",
            ReobservationDispositionV1::PersistKnowledge,
            100,
            2,
            0,
            0,
            1,
            0,
        );
        let mut wrong_scope = option(
            2,
            "b",
            ReobservationDispositionV1::PersistKnowledge,
            100,
            2,
            0,
            0,
            1,
            0,
        );
        wrong_scope.scope.objective = Objective::Cost;
        wrong_scope.option_sha256 = wrong_scope.canonical_digest();
        assert_eq!(
            optimize_memory_portfolio_exact(
                &[first.clone(), wrong_scope],
                budget(10, 2, 10, 10, 10)
            ),
            Err(PortfolioReject::InconsistentPortfolioScope)
        );

        let mut wrong_snapshot = first.clone();
        wrong_snapshot.as_known.known_at_ms += 1;
        wrong_snapshot.option_sha256 = wrong_snapshot.canonical_digest();
        assert_eq!(
            optimize_memory_portfolio_exact(
                &[first.clone(), wrong_snapshot],
                budget(10, 2, 10, 10, 10)
            ),
            Err(PortfolioReject::InconsistentKnowledgeSnapshot)
        );

        let mut wrong_authority = first.clone();
        wrong_authority.authority_scope_sha256 = digest("other-authority");
        wrong_authority.option_sha256 = wrong_authority.canonical_digest();
        assert_eq!(
            optimize_memory_portfolio_exact(
                &[first, wrong_authority],
                budget(10, 2, 10, 10, 10)
            ),
            Err(PortfolioReject::InconsistentAuthorityScope)
        );
    }

    #[test]
    fn exact_evidence_payload_cannot_be_double_counted_under_two_event_ids() {
        let first = option(
            11,
            "first",
            ReobservationDispositionV1::PersistKnowledge,
            100,
            4,
            0,
            0,
            1,
            0,
        );
        let mut second = option(
            12,
            "second",
            ReobservationDispositionV1::PersistKnowledge,
            100,
            4,
            0,
            0,
            1,
            0,
        );
        second.evidence_sha256 = first.evidence_sha256.clone();
        second.option_sha256 = second.canonical_digest();

        assert_eq!(
            optimize_memory_portfolio_exact(
                &[first, second],
                budget(16, 2, 10, 10, 10),
            ),
            Err(PortfolioReject::DuplicateEvidenceAcrossEvents(
                digest("evidence-11")
            ))
        );
    }

    #[test]
    fn portfolio_search_is_deterministic_under_candidate_order_permutation() {
        let options = vec![
            option(
                1,
                "one",
                ReobservationDispositionV1::PersistKnowledge,
                30,
                3,
                0,
                0,
                5,
                0,
            ),
            option(
                2,
                "two",
                ReobservationDispositionV1::CacheSnapshot,
                40,
                4,
                0,
                0,
                5,
                0,
            ),
            option(
                3,
                "three",
                ReobservationDispositionV1::StoreLocator,
                50,
                2,
                1,
                5,
                5,
                1,
            ),
        ];
        let first = optimize_memory_portfolio_exact(&options, budget(7, 1, 10, 20, 2))
            .unwrap();
        let reversed: Vec<_> = options.iter().cloned().rev().collect();
        let second = optimize_memory_portfolio_exact(&reversed, budget(7, 1, 10, 20, 2))
            .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn pareto_frontier_preserves_value_resource_tradeoffs_and_drops_dominated_plans() {
        let options = vec![
            option(
                1,
                "high-value-persist",
                ReobservationDispositionV1::PersistKnowledge,
                100,
                10,
                0,
                0,
                1,
                0,
            ),
            option(
                1,
                "compact-locator",
                ReobservationDispositionV1::StoreLocator,
                75,
                4,
                1,
                10,
                1,
                1,
            ),
            option(
                2,
                "complementary-persist",
                ReobservationDispositionV1::PersistKnowledge,
                70,
                8,
                0,
                0,
                1,
                0,
            ),
            option(
                3,
                "dominated-persist",
                ReobservationDispositionV1::PersistKnowledge,
                20,
                10,
                1,
                10,
                1,
                1,
            ),
        ];
        let frontier = optimize_memory_portfolio_frontier(
            &options,
            budget(12, 1, 20, 100, 10),
        )
        .unwrap();

        assert!(frontier.is_canonical());
        let values: BTreeSet<_> = frontier
            .plans
            .iter()
            .map(|plan| plan.total_net_value_micros)
            .collect();
        assert!(values.contains(&145));
        assert!(values.contains(&100));
        assert!(values.contains(&75));
        assert!(values.contains(&70));
        assert!(!values.contains(&20));
    }

    #[test]
    fn rehashed_frontier_with_dominated_plan_is_not_semantically_canonical() {
        let options = vec![
            option(
                1,
                "high",
                ReobservationDispositionV1::PersistKnowledge,
                100,
                10,
                0,
                0,
                1,
                0,
            ),
            option(
                1,
                "compact",
                ReobservationDispositionV1::StoreLocator,
                75,
                4,
                1,
                10,
                1,
                1,
            ),
        ];
        let mut frontier = optimize_memory_portfolio_frontier(
            &options,
            budget(12, 1, 20, 100, 10),
        )
        .unwrap();
        assert!(frontier.is_canonical());

        let mut dominated = frontier.plans[0].clone();
        dominated.total_net_value_micros =
            dominated.total_net_value_micros.saturating_sub(1);
        dominated.plan_sha256 = dominated.canonical_digest();
        frontier.plans.push(dominated);
        frontier.frontier_sha256 = frontier.canonical_digest();
        assert!(!frontier.is_canonical());
    }

    #[test]
    fn pareto_frontier_is_deterministic_under_input_permutation() {
        let options = vec![
            option(
                1,
                "one",
                ReobservationDispositionV1::PersistKnowledge,
                30,
                3,
                0,
                0,
                5,
                0,
            ),
            option(
                2,
                "two",
                ReobservationDispositionV1::CacheSnapshot,
                40,
                4,
                0,
                0,
                5,
                0,
            ),
            option(
                3,
                "three",
                ReobservationDispositionV1::StoreLocator,
                50,
                2,
                1,
                5,
                5,
                1,
            ),
        ];
        let first = optimize_memory_portfolio_frontier(
            &options,
            budget(7, 1, 10, 20, 2),
        )
        .unwrap();
        let reversed: Vec<_> = options.iter().cloned().rev().collect();
        let second = optimize_memory_portfolio_frontier(
            &reversed,
            budget(7, 1, 10, 20, 2),
        )
        .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn pareto_frontier_limit_exhaustion_returns_no_partial_result() {
        let options = vec![
            option(
                1,
                "high",
                ReobservationDispositionV1::PersistKnowledge,
                100,
                10,
                0,
                0,
                1,
                0,
            ),
            option(
                1,
                "compact",
                ReobservationDispositionV1::StoreLocator,
                75,
                4,
                1,
                10,
                1,
                1,
            ),
        ];
        assert_eq!(
            optimize_memory_portfolio_frontier_with_limits(
                &options,
                budget(12, 1, 20, 100, 10),
                100,
                1,
            ),
            Err(PortfolioReject::FrontierLimitExceeded)
        );
    }

    #[test]
    fn exact_search_fails_closed_when_node_limit_is_exhausted() {
        let options: Vec<_> = (1..=8)
            .map(|event| {
                option(
                    event,
                    "option",
                    ReobservationDispositionV1::PersistKnowledge,
                    10,
                    1,
                    0,
                    0,
                    1,
                    0,
                )
            })
            .collect();
        assert_eq!(
            optimize_memory_portfolio_exact_with_limit(
                &options,
                budget(8, 8, 100, 100, 100),
                1,
            ),
            Err(PortfolioReject::SearchLimitExceeded)
        );
    }
}
