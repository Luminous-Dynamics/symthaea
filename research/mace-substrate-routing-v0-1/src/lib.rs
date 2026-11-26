// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Research-only, deterministic routing among already-qualified memory
//! substrates. This module selects representations; it does not admit evidence,
//! verify authority/lifecycle, or authorize downstream actions.

use std::collections::BTreeSet;

pub const MACE_SUBSTRATE_ROUTER_MAX_SUBSTRATES_V1: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum QueryRegimeV1 {
    ExactIdentifier,
    CurrentState,
    HistoricalAsKnown,
    MultiHopCausal,
    SequentialAction,
    BroadRecall,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataVolatilityV1 {
    StableWithinHorizon,
    MayChange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum SubstrateKindV1 {
    RawTimeline,
    BitemporalLedger,
    DependencyGraph,
    SparseLexical,
    DenseSemantic,
    TypedHdc,
    HierarchicalSummary,
    ProceduralSkill,
    LiveObservation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteActionV1 {
    ExactCandidateLookup,
    ReadCurrentProjection,
    ReadHistoricalProjection,
    ExpandVerifiedParentClosure,
    SemanticCandidateLookup,
    AssociativeCandidateLookup,
    ReadQualifiedSummary,
    ReadRawSource,
    ReobserveMutableSource,
    ReadQualifiedProcedure,
    CheckProcedurePreconditions,
    ResolveCanonicalEvidence,
    RevalidateLifecycleAtUse,
    EnforceContextAndResourceCaps,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResourceVectorV1 {
    pub context_bytes: u64,
    pub context_tokens: u64,
    pub model_calls: u64,
    pub tool_calls: u64,
    pub verification_ops: u64,
    pub cpu_micros: u64,
    pub latency_micros: u64,
    pub storage_bytes: u64,
}

impl ResourceVectorV1 {
    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        Some(Self {
            context_bytes: self.context_bytes.checked_add(rhs.context_bytes)?,
            context_tokens: self.context_tokens.checked_add(rhs.context_tokens)?,
            model_calls: self.model_calls.checked_add(rhs.model_calls)?,
            tool_calls: self.tool_calls.checked_add(rhs.tool_calls)?,
            verification_ops: self.verification_ops.checked_add(rhs.verification_ops)?,
            cpu_micros: self.cpu_micros.checked_add(rhs.cpu_micros)?,
            latency_micros: self.latency_micros.checked_add(rhs.latency_micros)?,
            storage_bytes: self.storage_bytes.checked_add(rhs.storage_bytes)?,
        })
    }

    pub fn componentwise_le(self, budget: Self) -> bool {
        self.context_bytes <= budget.context_bytes
            && self.context_tokens <= budget.context_tokens
            && self.model_calls <= budget.model_calls
            && self.tool_calls <= budget.tool_calls
            && self.verification_ops <= budget.verification_ops
            && self.cpu_micros <= budget.cpu_micros
            && self.latency_micros <= budget.latency_micros
            && self.storage_bytes <= budget.storage_bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubstrateProfileV1 {
    pub kind: SubstrateKindV1,
    /// Digest assigned by the independently pinned substrate catalogue.
    pub profile_sha256: [u8; 32],
    pub enabled: bool,
    /// Estimated per-invocation cost, not a measured resource receipt.
    pub estimated_cost: ResourceVectorV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouteRequestV1 {
    pub decision_need_sha256: [u8; 32],
    pub substrate_catalog_sha256: [u8; 32],
    pub regime: QueryRegimeV1,
    pub volatility: DataVolatilityV1,
    pub structured_content_query: bool,
    pub requires_qualified_skill: bool,
    pub resource_budget: ResourceVectorV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouteStepV1 {
    /// None denotes a global gate that applies to the completed route.
    pub substrate: Option<SubstrateKindV1>,
    pub action: RouteActionV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubstrateBindingV1 {
    pub kind: SubstrateKindV1,
    pub profile_sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteFailureModeV1 {
    AbstainOrClarifyThroughUpstreamPolicy,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoutePlanV1 {
    pub decision_need_sha256: [u8; 32],
    pub substrate_catalog_sha256: [u8; 32],
    pub regime: QueryRegimeV1,
    pub selected_substrates: Vec<SubstrateBindingV1>,
    pub steps: Vec<RouteStepV1>,
    pub estimated_cost: ResourceVectorV1,
    pub declared_budget: ResourceVectorV1,
    pub estimate_only: bool,
    pub requires_canonical_resolution: bool,
    pub requires_lifecycle_revalidation_at_use: bool,
    pub any_step_failure_is_non_qualifying: bool,
    pub failure_mode: RouteFailureModeV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteRejectV1 {
    DuplicateSubstrateProfile,
    RequiredSubstrateUnavailable,
    TooManySubstrates,
    ResourceCostOverflow,
    ResourceBudgetExceeded,
}

fn profile_for(
    kind: SubstrateKindV1,
    profiles: &[SubstrateProfileV1],
) -> Result<SubstrateProfileV1, RouteRejectV1> {
    let mut matches = profiles.iter().filter(|profile| profile.kind == kind);
    let first = matches.next().copied()
        .ok_or(RouteRejectV1::RequiredSubstrateUnavailable)?;
    if matches.next().is_some() {
        return Err(RouteRejectV1::DuplicateSubstrateProfile);
    }
    if !first.enabled {
        return Err(RouteRejectV1::RequiredSubstrateUnavailable);
    }
    Ok(first)
}

fn add_step(
    kind: SubstrateKindV1,
    action: RouteActionV1,
    profiles: &[SubstrateProfileV1],
    selected: &mut Vec<SubstrateBindingV1>,
    steps: &mut Vec<RouteStepV1>,
) -> Result<(), RouteRejectV1> {
    let profile = profile_for(kind, profiles)?;
    if !selected.iter().any(|binding| binding.kind == kind) {
        if selected.len() >= MACE_SUBSTRATE_ROUTER_MAX_SUBSTRATES_V1 {
            return Err(RouteRejectV1::TooManySubstrates);
        }
        selected.push(SubstrateBindingV1 {
            kind,
            profile_sha256: profile.profile_sha256,
        });
    }
    steps.push(RouteStepV1 {
        substrate: Some(kind),
        action,
    });
    Ok(())
}

fn add_global_step(steps: &mut Vec<RouteStepV1>, action: RouteActionV1) {
    steps.push(RouteStepV1 {
        substrate: None,
        action,
    });
}

fn first_available(
    candidates: &[SubstrateKindV1],
    profiles: &[SubstrateProfileV1],
) -> Result<SubstrateKindV1, RouteRejectV1> {
    for kind in candidates {
        if profile_for(*kind, profiles).is_ok() {
            return Ok(*kind);
        }
    }
    Err(RouteRejectV1::RequiredSubstrateUnavailable)
}

/// Creates a deterministic, fail-closed query plan over pre-qualified
/// substrates. The profile catalogue and DecisionNeed digest must be supplied
/// by upstream owners and pinned outside candidate memory.
pub fn route_mace_query_v1(
    request: &RouteRequestV1,
    profiles: &[SubstrateProfileV1],
) -> Result<RoutePlanV1, RouteRejectV1> {
    // Reject duplicate catalogue entries even when the selected route would
    // not happen to consult the duplicate kind.
    let mut seen = BTreeSet::new();
    for profile in profiles {
        if !seen.insert(profile.kind) {
            return Err(RouteRejectV1::DuplicateSubstrateProfile);
        }
    }

    let mut selected = Vec::new();
    let mut steps = Vec::new();

    match request.regime {
        QueryRegimeV1::ExactIdentifier => {
            let search = first_available(
                &[SubstrateKindV1::SparseLexical, SubstrateKindV1::RawTimeline],
                profiles,
            )?;
            let action = if search == SubstrateKindV1::SparseLexical {
                RouteActionV1::ExactCandidateLookup
            } else {
                RouteActionV1::ReadRawSource
            };
            add_step(search, action, profiles, &mut selected, &mut steps)?;
            if search == SubstrateKindV1::SparseLexical
                && profiles.iter().any(|profile| {
                    profile.kind == SubstrateKindV1::RawTimeline && profile.enabled
                })
            {
                add_step(
                    SubstrateKindV1::RawTimeline,
                    RouteActionV1::ReadRawSource,
                    profiles,
                    &mut selected,
                    &mut steps,
                )?;
            }
        }
        QueryRegimeV1::CurrentState => {
            add_step(
                SubstrateKindV1::BitemporalLedger,
                RouteActionV1::ReadCurrentProjection,
                profiles,
                &mut selected,
                &mut steps,
            )?;
            if request.volatility == DataVolatilityV1::MayChange {
                add_step(
                    SubstrateKindV1::LiveObservation,
                    RouteActionV1::ReobserveMutableSource,
                    profiles,
                    &mut selected,
                    &mut steps,
                )?;
            }
        }
        QueryRegimeV1::HistoricalAsKnown => {
            add_step(
                SubstrateKindV1::BitemporalLedger,
                RouteActionV1::ReadHistoricalProjection,
                profiles,
                &mut selected,
                &mut steps,
            )?;
            add_step(
                SubstrateKindV1::RawTimeline,
                RouteActionV1::ReadRawSource,
                profiles,
                &mut selected,
                &mut steps,
            )?;
        }
        QueryRegimeV1::MultiHopCausal => {
            add_step(
                SubstrateKindV1::DependencyGraph,
                RouteActionV1::ExpandVerifiedParentClosure,
                profiles,
                &mut selected,
                &mut steps,
            )?;
            add_step(
                SubstrateKindV1::RawTimeline,
                RouteActionV1::ReadRawSource,
                profiles,
                &mut selected,
                &mut steps,
            )?;
        }
        QueryRegimeV1::SequentialAction => {
            add_step(
                SubstrateKindV1::BitemporalLedger,
                RouteActionV1::ReadCurrentProjection,
                profiles,
                &mut selected,
                &mut steps,
            )?;
            if request.volatility == DataVolatilityV1::MayChange {
                add_step(
                    SubstrateKindV1::LiveObservation,
                    RouteActionV1::ReobserveMutableSource,
                    profiles,
                    &mut selected,
                    &mut steps,
                )?;
            }
            if request.requires_qualified_skill {
                add_step(
                    SubstrateKindV1::ProceduralSkill,
                    RouteActionV1::ReadQualifiedProcedure,
                    profiles,
                    &mut selected,
                    &mut steps,
                )?;
                add_step(
                    SubstrateKindV1::ProceduralSkill,
                    RouteActionV1::CheckProcedurePreconditions,
                    profiles,
                    &mut selected,
                    &mut steps,
                )?;
            }
        }
        QueryRegimeV1::BroadRecall => {
            let candidates = if request.structured_content_query {
                [
                    SubstrateKindV1::TypedHdc,
                    SubstrateKindV1::DenseSemantic,
                    SubstrateKindV1::SparseLexical,
                    SubstrateKindV1::RawTimeline,
                ]
            } else {
                [
                    SubstrateKindV1::DenseSemantic,
                    SubstrateKindV1::TypedHdc,
                    SubstrateKindV1::SparseLexical,
                    SubstrateKindV1::RawTimeline,
                ]
            };
            let chosen = first_available(&candidates, profiles)?;
            let action = match chosen {
                SubstrateKindV1::TypedHdc => RouteActionV1::AssociativeCandidateLookup,
                SubstrateKindV1::DenseSemantic => RouteActionV1::SemanticCandidateLookup,
                SubstrateKindV1::SparseLexical => RouteActionV1::ExactCandidateLookup,
                _ => RouteActionV1::ReadRawSource,
            };
            add_step(chosen, action, profiles, &mut selected, &mut steps)?;
            if chosen == SubstrateKindV1::HierarchicalSummary {
                add_step(
                    chosen,
                    RouteActionV1::ReadQualifiedSummary,
                    profiles,
                    &mut selected,
                    &mut steps,
                )?;
                add_step(
                    SubstrateKindV1::DependencyGraph,
                    RouteActionV1::ExpandVerifiedParentClosure,
                    profiles,
                    &mut selected,
                    &mut steps,
                )?;
            }
        }
    }

    add_global_step(&mut steps, RouteActionV1::ResolveCanonicalEvidence);
    add_global_step(&mut steps, RouteActionV1::RevalidateLifecycleAtUse);
    add_global_step(&mut steps, RouteActionV1::EnforceContextAndResourceCaps);

    let mut estimated_cost = ResourceVectorV1::default();
    for binding in &selected {
        let profile = profile_for(binding.kind, profiles)?;
        if profile.profile_sha256 != binding.profile_sha256 {
            return Err(RouteRejectV1::RequiredSubstrateUnavailable);
        }
        estimated_cost = estimated_cost
            .checked_add(profile.estimated_cost)
            .ok_or(RouteRejectV1::ResourceCostOverflow)?;
    }
    if !estimated_cost.componentwise_le(request.resource_budget) {
        return Err(RouteRejectV1::ResourceBudgetExceeded);
    }

    Ok(RoutePlanV1 {
        decision_need_sha256: request.decision_need_sha256,
        substrate_catalog_sha256: request.substrate_catalog_sha256,
        regime: request.regime,
        selected_substrates: selected,
        steps,
        estimated_cost,
        declared_budget: request.resource_budget,
        estimate_only: true,
        requires_canonical_resolution: true,
        requires_lifecycle_revalidation_at_use: true,
        any_step_failure_is_non_qualifying: true,
        failure_mode: RouteFailureModeV1::AbstainOrClarifyThroughUpstreamPolicy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROFILE_DIGEST: [u8; 32] = [7; 32];
    const NEED_DIGEST: [u8; 32] = [1; 32];
    const CATALOG_DIGEST: [u8; 32] = [2; 32];

    fn cost(context_tokens: u64, model_calls: u64, tool_calls: u64) -> ResourceVectorV1 {
        ResourceVectorV1 {
            context_bytes: context_tokens.saturating_mul(4),
            context_tokens,
            model_calls,
            tool_calls,
            verification_ops: 1,
            cpu_micros: 100,
            latency_micros: 1_000,
            storage_bytes: 0,
        }
    }

    fn profile(kind: SubstrateKindV1, estimated_cost: ResourceVectorV1) -> SubstrateProfileV1 {
        SubstrateProfileV1 {
            kind,
            profile_sha256: PROFILE_DIGEST,
            enabled: true,
            estimated_cost,
        }
    }

    fn request(regime: QueryRegimeV1) -> RouteRequestV1 {
        RouteRequestV1 {
            decision_need_sha256: NEED_DIGEST,
            substrate_catalog_sha256: CATALOG_DIGEST,
            regime,
            volatility: DataVolatilityV1::StableWithinHorizon,
            structured_content_query: false,
            requires_qualified_skill: false,
            resource_budget: ResourceVectorV1 {
                context_bytes: 1_000_000,
                context_tokens: 250_000,
                model_calls: 100,
                tool_calls: 100,
                verification_ops: 100,
                cpu_micros: 10_000_000,
                latency_micros: 10_000_000,
                storage_bytes: 10_000_000,
            },
        }
    }

    #[test]
    fn current_state_uses_bitemporal_projection() {
        let req = request(QueryRegimeV1::CurrentState);
        let profiles = [profile(
            SubstrateKindV1::BitemporalLedger,
            cost(100, 0, 0),
        )];
        let plan = route_mace_query_v1(&req, &profiles).unwrap();
        assert_eq!(
            plan.selected_substrates[0].kind,
            SubstrateKindV1::BitemporalLedger
        );
        assert!(plan.steps.iter().any(|step| {
            step.action == RouteActionV1::ReadCurrentProjection
        }));
    }

    #[test]
    fn mutable_current_state_requires_live_reobservation() {
        let mut req = request(QueryRegimeV1::CurrentState);
        req.volatility = DataVolatilityV1::MayChange;
        let profiles = [
            profile(SubstrateKindV1::BitemporalLedger, cost(100, 0, 0)),
            profile(SubstrateKindV1::LiveObservation, cost(100, 1, 1)),
        ];
        let plan = route_mace_query_v1(&req, &profiles).unwrap();
        assert!(plan.steps.iter().any(|step| {
            step.action == RouteActionV1::ReobserveMutableSource
        }));
    }

    #[test]
    fn mutable_current_state_fails_closed_without_live_source() {
        let mut req = request(QueryRegimeV1::CurrentState);
        req.volatility = DataVolatilityV1::MayChange;
        let profiles = [profile(
            SubstrateKindV1::BitemporalLedger,
            cost(100, 0, 0),
        )];
        assert_eq!(
            route_mace_query_v1(&req, &profiles),
            Err(RouteRejectV1::RequiredSubstrateUnavailable)
        );
    }

    #[test]
    fn historical_query_keeps_bitemporal_and_raw_source() {
        let req = request(QueryRegimeV1::HistoricalAsKnown);
        let profiles = [
            profile(SubstrateKindV1::BitemporalLedger, cost(100, 0, 0)),
            profile(SubstrateKindV1::RawTimeline, cost(100, 0, 0)),
        ];
        let plan = route_mace_query_v1(&req, &profiles).unwrap();
        assert_eq!(plan.selected_substrates.len(), 2);
        assert!(plan.steps.iter().any(|step| {
            step.action == RouteActionV1::ReadHistoricalProjection
        }));
    }

    #[test]
    fn causal_query_expands_verified_dependency_closure() {
        let req = request(QueryRegimeV1::MultiHopCausal);
        let profiles = [
            profile(SubstrateKindV1::DependencyGraph, cost(100, 0, 1)),
            profile(SubstrateKindV1::RawTimeline, cost(100, 0, 0)),
        ];
        let plan = route_mace_query_v1(&req, &profiles).unwrap();
        assert!(plan.steps.iter().any(|step| {
            step.action == RouteActionV1::ExpandVerifiedParentClosure
        }));
    }

    #[test]
    fn sequential_action_checks_qualified_skill_preconditions() {
        let mut req = request(QueryRegimeV1::SequentialAction);
        req.requires_qualified_skill = true;
        let profiles = [
            profile(SubstrateKindV1::BitemporalLedger, cost(100, 0, 0)),
            profile(SubstrateKindV1::ProceduralSkill, cost(100, 0, 0)),
        ];
        let plan = route_mace_query_v1(&req, &profiles).unwrap();
        assert!(plan.steps.iter().any(|step| {
            step.action == RouteActionV1::CheckProcedurePreconditions
        }));
    }

    #[test]
    fn structured_recall_prefers_typed_hdc() {
        let mut req = request(QueryRegimeV1::BroadRecall);
        req.structured_content_query = true;
        let profiles = [
            profile(SubstrateKindV1::DenseSemantic, cost(100, 1, 0)),
            profile(SubstrateKindV1::TypedHdc, cost(100, 0, 0)),
        ];
        let plan = route_mace_query_v1(&req, &profiles).unwrap();
        assert_eq!(
            plan.selected_substrates[0].kind,
            SubstrateKindV1::TypedHdc
        );
    }

    #[test]
    fn unstructured_recall_prefers_dense_semantic() {
        let req = request(QueryRegimeV1::BroadRecall);
        let profiles = [
            profile(SubstrateKindV1::TypedHdc, cost(100, 0, 0)),
            profile(SubstrateKindV1::DenseSemantic, cost(100, 1, 0)),
        ];
        let plan = route_mace_query_v1(&req, &profiles).unwrap();
        assert_eq!(
            plan.selected_substrates[0].kind,
            SubstrateKindV1::DenseSemantic
        );
    }

    #[test]
    fn exact_identifier_falls_back_to_raw_scan_when_sparse_is_unavailable() {
        let req = request(QueryRegimeV1::ExactIdentifier);
        let profiles = [profile(SubstrateKindV1::RawTimeline, cost(100, 0, 0))];
        let plan = route_mace_query_v1(&req, &profiles).unwrap();
        assert_eq!(
            plan.selected_substrates[0].kind,
            SubstrateKindV1::RawTimeline
        );
    }

    #[test]
    fn every_plan_requires_canonical_resolution_and_current_use_revalidation() {
        let req = request(QueryRegimeV1::BroadRecall);
        let profiles = [profile(SubstrateKindV1::DenseSemantic, cost(100, 1, 0))];
        let plan = route_mace_query_v1(&req, &profiles).unwrap();
        assert!(plan.requires_canonical_resolution);
        assert!(plan.requires_lifecycle_revalidation_at_use);
        assert!(plan.any_step_failure_is_non_qualifying);
        assert!(plan.steps.iter().any(|step| {
            step.action == RouteActionV1::ResolveCanonicalEvidence
        }));
        assert!(plan.steps.iter().any(|step| {
            step.action == RouteActionV1::RevalidateLifecycleAtUse
        }));
    }

    #[test]
    fn componentwise_budget_rejects_tradeoff_substitution() {
        let mut req = request(QueryRegimeV1::BroadRecall);
        req.resource_budget.context_tokens = 1;
        req.resource_budget.model_calls = 10;
        let profiles = [profile(
            SubstrateKindV1::DenseSemantic,
            cost(2, 0, 0),
        )];
        assert_eq!(
            route_mace_query_v1(&req, &profiles),
            Err(RouteRejectV1::ResourceBudgetExceeded)
        );
    }

    #[test]
    fn duplicate_profiles_are_rejected_even_when_unselected() {
        let req = request(QueryRegimeV1::BroadRecall);
        let profiles = [
            profile(SubstrateKindV1::DenseSemantic, cost(10, 0, 0)),
            profile(SubstrateKindV1::DenseSemantic, cost(20, 0, 0)),
        ];
        assert_eq!(
            route_mace_query_v1(&req, &profiles),
            Err(RouteRejectV1::DuplicateSubstrateProfile)
        );
    }

    #[test]
    fn estimated_cost_overflow_fails_closed() {
        let req = request(QueryRegimeV1::HistoricalAsKnown);
        let profiles = [
            profile(
                SubstrateKindV1::BitemporalLedger,
                ResourceVectorV1 { context_bytes: u64::MAX, ..ResourceVectorV1::default() },
            ),
            profile(
                SubstrateKindV1::RawTimeline,
                ResourceVectorV1 { context_bytes: 1, ..ResourceVectorV1::default() },
            ),
        ];
        assert_eq!(
            route_mace_query_v1(&req, &profiles),
            Err(RouteRejectV1::ResourceCostOverflow)
        );
    }

    #[test]
    fn plan_is_deterministic_under_catalog_order_permutation() {
        let req = request(QueryRegimeV1::HistoricalAsKnown);
        let a = [
            profile(SubstrateKindV1::BitemporalLedger, cost(100, 0, 0)),
            profile(SubstrateKindV1::RawTimeline, cost(100, 0, 0)),
        ];
        let b = [a[1], a[0]];
        assert_eq!(
            route_mace_query_v1(&req, &a).unwrap(),
            route_mace_query_v1(&req, &b).unwrap()
        );
    }
}
