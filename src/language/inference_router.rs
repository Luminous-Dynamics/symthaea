// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! IF-8 deterministic inference capability router.
//!
//! Routing is deliberately split into two stages:
//! 1. hard eligibility gates (policy/capability/privacy/resource state);
//! 2. deterministic integer scoring among already-eligible candidates.
//!
//! A routing plan is not execution authority. Resource snapshots may become stale;
//! the selected provider guard must still successfully `reserve()` immediately
//! before permit/execution preparation.

#[cfg(not(test))]
use super::inference_contract::{
    AdmittedInferenceRoute, ExecutionLocation, InferenceAdmissionError, InferenceCandidate,
    InferencePolicy, InferenceRequest, ModelIdentity,
};
#[cfg(not(test))]
use super::inference_resource_guard::InferenceResourceGuard;

#[cfg(test)]
use crate::inference_contract::{
    AdmittedInferenceRoute, ExecutionLocation, InferenceAdmissionError, InferenceCandidate,
    InferencePolicy, InferenceRequest, ModelIdentity,
};
#[cfg(test)]
use crate::inference_resource_guard::InferenceResourceGuard;

use std::cmp::Ordering;
use std::fmt;

const SCORE_MAX: u16 = 1_000;
const UNKNOWN_OBSERVATION_SCORE: u16 = 500;

/// Non-authoritative observed quality/reliability/latency inputs.
///
/// These values may influence preference ordering but never bypass hard admission
/// or resource gates. Quality/reliability use fixed-point thousandths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InferenceRouteMetrics {
    quality_milli: Option<u16>,
    reliability_milli: Option<u16>,
    latency_millis: Option<u64>,
}

impl InferenceRouteMetrics {
    pub fn new(
        quality_milli: Option<u16>,
        reliability_milli: Option<u16>,
        latency_millis: Option<u64>,
    ) -> Result<Self, InferenceRouterError> {
        for value in [quality_milli, reliability_milli].into_iter().flatten() {
            if value > SCORE_MAX {
                return Err(InferenceRouterError::MetricOutOfRange);
            }
        }
        Ok(Self {
            quality_milli,
            reliability_milli,
            latency_millis,
        })
    }

    pub const fn quality_milli(&self) -> Option<u16> {
        self.quality_milli
    }

    pub const fn reliability_milli(&self) -> Option<u16> {
        self.reliability_milli
    }

    pub const fn latency_millis(&self) -> Option<u64> {
        self.latency_millis
    }
}

/// Integer-only score weights. Hard gates are never represented as weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InferenceRouterWeights {
    pub quality: u16,
    pub reliability: u16,
    pub locality: u16,
    pub resource_headroom: u16,
    pub cost_efficiency: u16,
    pub latency: u16,
}

impl InferenceRouterWeights {
    /// Strong local/sovereign preference while still valuing measured quality.
    pub const fn sovereign_balanced() -> Self {
        Self {
            quality: 5,
            reliability: 4,
            locality: 5,
            resource_headroom: 3,
            cost_efficiency: 3,
            latency: 2,
        }
    }
}

impl Default for InferenceRouterWeights {
    fn default() -> Self {
        Self::sovereign_balanced()
    }
}

/// One candidate offered to the planner.
pub struct InferenceRoutingCandidate<'a> {
    pub candidate: &'a InferenceCandidate,
    /// Required for remote-provider/community-peer candidates. Optional for local
    /// execution; when present it is still honored as a hard resource boundary.
    pub resource_guard: Option<&'a InferenceResourceGuard>,
    pub metrics: InferenceRouteMetrics,
}

impl fmt::Debug for InferenceRoutingCandidate<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InferenceRoutingCandidate")
            .field("provider_id", &self.candidate.provider_id)
            .field("model", &self.candidate.model)
            .field("location", &self.candidate.location)
            .field("resource_guard_present", &self.resource_guard.is_some())
            .field("metrics", &self.metrics)
            .finish()
    }
}

/// Inspectable score components for one admitted candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InferenceRouteScore {
    pub quality_milli: u16,
    pub reliability_milli: u16,
    pub locality_milli: u16,
    pub resource_headroom_milli: u16,
    pub cost_efficiency_milli: u16,
    pub latency_milli: u16,
    pub total: u64,
}

/// One eligible route in deterministic preference order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankedInferenceRoute {
    candidate: InferenceCandidate,
    route: AdmittedInferenceRoute,
    score: InferenceRouteScore,
    stable_key: String,
}

impl RankedInferenceRoute {
    pub fn candidate(&self) -> &InferenceCandidate {
        &self.candidate
    }

    pub fn route(&self) -> &AdmittedInferenceRoute {
        &self.route
    }

    pub const fn score(&self) -> InferenceRouteScore {
        self.score
    }

    pub fn stable_key(&self) -> &str {
        &self.stable_key
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceRouteRejectionReason {
    Admission(InferenceAdmissionError),
    ResourceStateRequired,
    CredentialBlocked,
    CoolingDown { until_millis: u64 },
    RequestBudgetExhausted,
    TokenBudgetExhausted { required: u64, remaining: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedInferenceRoute {
    pub provider_id: String,
    pub model_key: String,
    pub reason: InferenceRouteRejectionReason,
}

/// Pure selection result. `ranked[0]` is preferred, but callers should preserve
/// the full order as a fallback plan and attempt a real resource reservation before
/// minting any execution permit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceRoutingPlan {
    ranked: Vec<RankedInferenceRoute>,
    rejected: Vec<RejectedInferenceRoute>,
    reserved_token_exposure: u64,
}

impl InferenceRoutingPlan {
    pub fn ranked(&self) -> &[RankedInferenceRoute] {
        &self.ranked
    }

    pub fn rejected(&self) -> &[RejectedInferenceRoute] {
        &self.rejected
    }

    pub const fn reserved_token_exposure(&self) -> u64 {
        self.reserved_token_exposure
    }

    pub fn preferred(&self) -> Option<&RankedInferenceRoute> {
        self.ranked.first()
    }
}

/// Build a deterministic ordered fallback plan.
pub fn plan_inference_routes(
    policy: &InferencePolicy,
    request: &InferenceRequest,
    candidates: &[InferenceRoutingCandidate<'_>],
    weights: InferenceRouterWeights,
    now_millis: u64,
) -> Result<InferenceRoutingPlan, InferenceRouterError> {
    let reserved_token_exposure = request
        .requirements
        .estimated_input_tokens
        .checked_add(request.requirements.max_output_tokens)
        .ok_or(InferenceRouterError::TokenExposureOverflow)?;

    let mut ranked = Vec::new();
    let mut rejected = Vec::new();

    for offered in candidates {
        let model_key = model_stable_key(&offered.candidate.model);
        let route = match policy.admit(request, offered.candidate) {
            Ok(route) => route,
            Err(error) => {
                rejected.push(RejectedInferenceRoute {
                    provider_id: offered.candidate.provider_id.clone(),
                    model_key,
                    reason: InferenceRouteRejectionReason::Admission(error),
                });
                continue;
            }
        };

        let resource = match resource_eligibility(
            offered.candidate,
            offered.resource_guard,
            reserved_token_exposure,
            now_millis,
        ) {
            Ok(resource) => resource,
            Err(reason) => {
                rejected.push(RejectedInferenceRoute {
                    provider_id: offered.candidate.provider_id.clone(),
                    model_key,
                    reason,
                });
                continue;
            }
        };

        let score = score_candidate(
            policy,
            offered.candidate,
            offered.metrics,
            resource,
            reserved_token_exposure,
            weights,
        );
        let stable_key = format!("{}\0{}", offered.candidate.provider_id, model_key);
        ranked.push(RankedInferenceRoute {
            candidate: offered.candidate.clone(),
            route,
            score,
            stable_key,
        });
    }

    ranked.sort_by(compare_ranked_routes);
    rejected.sort_by(|a, b| {
        a.provider_id
            .cmp(&b.provider_id)
            .then_with(|| a.model_key.cmp(&b.model_key))
    });

    Ok(InferenceRoutingPlan {
        ranked,
        rejected,
        reserved_token_exposure,
    })
}

#[derive(Debug, Clone, Copy)]
struct ResourceView {
    remaining_requests: Option<u64>,
    remaining_tokens: Option<u64>,
}

fn resource_eligibility(
    candidate: &InferenceCandidate,
    guard: Option<&InferenceResourceGuard>,
    reserved_tokens: u64,
    now_millis: u64,
) -> Result<ResourceView, InferenceRouteRejectionReason> {
    let needs_guard = matches!(
        candidate.location,
        ExecutionLocation::RemoteProvider | ExecutionLocation::CommunityPeer
    );
    let Some(guard) = guard else {
        if needs_guard {
            return Err(InferenceRouteRejectionReason::ResourceStateRequired);
        }
        return Ok(ResourceView {
            remaining_requests: None,
            remaining_tokens: None,
        });
    };

    if guard.is_credential_blocked() {
        return Err(InferenceRouteRejectionReason::CredentialBlocked);
    }
    if let Some(until_millis) = guard.cooldown_until_millis()
        && now_millis < until_millis
    {
        return Err(InferenceRouteRejectionReason::CoolingDown { until_millis });
    }

    let remaining_requests = guard.effective_remaining_requests();
    if remaining_requests == Some(0) {
        return Err(InferenceRouteRejectionReason::RequestBudgetExhausted);
    }

    let remaining_tokens = guard.effective_remaining_tokens();
    if let Some(remaining) = remaining_tokens
        && remaining < reserved_tokens
    {
        return Err(InferenceRouteRejectionReason::TokenBudgetExhausted {
            required: reserved_tokens,
            remaining,
        });
    }

    Ok(ResourceView {
        remaining_requests,
        remaining_tokens,
    })
}

fn score_candidate(
    policy: &InferencePolicy,
    candidate: &InferenceCandidate,
    metrics: InferenceRouteMetrics,
    resource: ResourceView,
    reserved_tokens: u64,
    weights: InferenceRouterWeights,
) -> InferenceRouteScore {
    let quality = metrics.quality_milli.unwrap_or(UNKNOWN_OBSERVATION_SCORE);
    let reliability = metrics
        .reliability_milli
        .unwrap_or(UNKNOWN_OBSERVATION_SCORE);
    let locality = locality_score(candidate.location);
    let resource_headroom = resource_headroom_score(resource, reserved_tokens);
    let cost_efficiency = cost_efficiency_score(policy, candidate);
    let latency = latency_score(metrics.latency_millis);

    let total = weighted(quality, weights.quality)
        + weighted(reliability, weights.reliability)
        + weighted(locality, weights.locality)
        + weighted(resource_headroom, weights.resource_headroom)
        + weighted(cost_efficiency, weights.cost_efficiency)
        + weighted(latency, weights.latency);

    InferenceRouteScore {
        quality_milli: quality,
        reliability_milli: reliability,
        locality_milli: locality,
        resource_headroom_milli: resource_headroom,
        cost_efficiency_milli: cost_efficiency,
        latency_milli: latency,
        total,
    }
}

fn weighted(score: u16, weight: u16) -> u64 {
    u64::from(score) * u64::from(weight)
}

fn locality_score(location: ExecutionLocation) -> u16 {
    match location {
        ExecutionLocation::LocalProcess => 1_000,
        ExecutionLocation::LocalDevice => 950,
        ExecutionLocation::LocalNetwork => 800,
        ExecutionLocation::CommunityPeer => 550,
        ExecutionLocation::RemoteProvider => 300,
    }
}

fn resource_headroom_score(resource: ResourceView, reserved_tokens: u64) -> u16 {
    let request_score = match resource.remaining_requests {
        None => 1_000,
        Some(0) => 0,
        Some(1) => 100,
        Some(2..=4) => 300,
        Some(5..=19) => 600,
        Some(_) => 1_000,
    };

    let token_score = match (resource.remaining_tokens, reserved_tokens) {
        (None, _) | (_, 0) => 1_000,
        (Some(remaining), required) => {
            let scaled = (u128::from(remaining) * 250) / u128::from(required);
            u16::try_from(scaled.min(1_000)).unwrap_or(1_000)
        }
    };

    request_score.min(token_score)
}

fn cost_efficiency_score(policy: &InferencePolicy, candidate: &InferenceCandidate) -> u16 {
    let charge = candidate.max_charge_microusd.unwrap_or(u64::MAX);
    if charge == 0 {
        return 1_000;
    }
    if policy.max_charge_microusd == 0 {
        return 0;
    }
    let fraction = (u128::from(charge) * 1_000) / u128::from(policy.max_charge_microusd);
    let penalty = u16::try_from(fraction.min(1_000)).unwrap_or(1_000);
    1_000 - penalty
}

fn latency_score(latency_millis: Option<u64>) -> u16 {
    let Some(latency) = latency_millis else {
        return UNKNOWN_OBSERVATION_SCORE;
    };
    let denominator = 1_000u128 + u128::from(latency);
    let score = 1_000_000u128 / denominator;
    u16::try_from(score.min(1_000)).unwrap_or(1_000)
}

fn compare_ranked_routes(a: &RankedInferenceRoute, b: &RankedInferenceRoute) -> Ordering {
    // Descending preference for score components, then cheaper/localer route,
    // finally a stable provider/model key. No input-order tie break exists.
    b.score
        .total
        .cmp(&a.score.total)
        .then_with(|| b.score.quality_milli.cmp(&a.score.quality_milli))
        .then_with(|| b.score.reliability_milli.cmp(&a.score.reliability_milli))
        .then_with(|| {
            a.route
                .admitted_max_charge_microusd()
                .cmp(&b.route.admitted_max_charge_microusd())
        })
        .then_with(|| locality_sort_rank(a.route.location()).cmp(&locality_sort_rank(b.route.location())))
        .then_with(|| a.stable_key.cmp(&b.stable_key))
}

fn locality_sort_rank(location: ExecutionLocation) -> u8 {
    match location {
        ExecutionLocation::LocalProcess => 0,
        ExecutionLocation::LocalDevice => 1,
        ExecutionLocation::LocalNetwork => 2,
        ExecutionLocation::CommunityPeer => 3,
        ExecutionLocation::RemoteProvider => 4,
    }
}

fn model_stable_key(model: &ModelIdentity) -> String {
    match model {
        ModelIdentity::ContentVerified { digest } => format!("1:{digest}"),
        ModelIdentity::ProviderAttested {
            provider,
            declared_model,
        } => format!("2:{provider}:{declared_model}"),
        ModelIdentity::OpaqueEndpoint { endpoint_id } => format!("3:{endpoint_id}"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceRouterError {
    MetricOutOfRange,
    TokenExposureOverflow,
}

impl fmt::Display for InferenceRouterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MetricOutOfRange => write!(f, "routing metric must be within 0..=1000"),
            Self::TokenExposureOverflow => write!(f, "planned token exposure overflowed u64"),
        }
    }
}

impl std::error::Error for InferenceRouterError {}
