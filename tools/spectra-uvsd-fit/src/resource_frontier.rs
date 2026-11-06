// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Componentwise resource feasibility and Pareto acquisition frontiers.
//! Does not scalarize incomparable resource dimensions or authenticate costs.

use std::collections::BTreeSet;
use std::fmt;

pub const RESOURCE_FRONTIER_METHOD_ID: &str =
    "spectra-componentwise-resource-feasible-pareto-frontier-v1";

/// Integer resource quantities: replications, milliseconds (CPU/GPU/wall),
/// human seconds, and micro-units of the contract's frozen currency.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResourceUnits {
    pub simulation_replications: u64,
    pub cpu_millis: u64,
    pub gpu_millis: u64,
    pub wall_millis: u64,
    pub human_seconds: u64,
    pub currency_microunits: u64,
}

impl ResourceUnits {
    pub fn componentwise_leq(self, other: Self) -> bool {
        self.simulation_replications <= other.simulation_replications
            && self.cpu_millis <= other.cpu_millis
            && self.gpu_millis <= other.gpu_millis
            && self.wall_millis <= other.wall_millis
            && self.human_seconds <= other.human_seconds
            && self.currency_microunits <= other.currency_microunits
    }

    pub fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            simulation_replications: self.simulation_replications.checked_add(other.simulation_replications)?,
            cpu_millis: self.cpu_millis.checked_add(other.cpu_millis)?,
            gpu_millis: self.gpu_millis.checked_add(other.gpu_millis)?,
            wall_millis: self.wall_millis.checked_add(other.wall_millis)?,
            human_seconds: self.human_seconds.checked_add(other.human_seconds)?,
            currency_microunits: self.currency_microunits.checked_add(other.currency_microunits)?,
        })
    }

    pub fn checked_sub(self, other: Self) -> Option<Self> {
        Some(Self {
            simulation_replications: self.simulation_replications.checked_sub(other.simulation_replications)?,
            cpu_millis: self.cpu_millis.checked_sub(other.cpu_millis)?,
            gpu_millis: self.gpu_millis.checked_sub(other.gpu_millis)?,
            wall_millis: self.wall_millis.checked_sub(other.wall_millis)?,
            human_seconds: self.human_seconds.checked_sub(other.human_seconds)?,
            currency_microunits: self.currency_microunits.checked_sub(other.currency_microunits)?,
        })
    }

    fn strict_cost_improvement(self, other: Self) -> bool {
        self.simulation_replications < other.simulation_replications
            || self.cpu_millis < other.cpu_millis
            || self.gpu_millis < other.gpu_millis
            || self.wall_millis < other.wall_millis
            || self.human_seconds < other.human_seconds
            || self.currency_microunits < other.currency_microunits
    }

    fn cmp_lexicographic(self, other: Self) -> std::cmp::Ordering {
        (
            self.simulation_replications,
            self.cpu_millis,
            self.gpu_millis,
            self.wall_millis,
            self.human_seconds,
            self.currency_microunits,
        )
            .cmp(&(
                other.simulation_replications,
                other.cpu_millis,
                other.gpu_millis,
                other.wall_millis,
                other.human_seconds,
                other.currency_microunits,
            ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceCostUpperBound {
    pub units: ResourceUnits,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceFrontierContract {
    pub policy_version: String,
    pub currency_code: String,
    pub remaining_budget: ResourceUnits,
}

/// Benefit is a projected certificate-gap contraction, not an inferential bound.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceProjectionCandidate {
    pub candidate_id: String,
    pub projected_certificate_gap_contraction: f64,
    pub upper_cost: ResourceCostUpperBound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceCandidateDisposition {
    OnParetoFrontier,
    Dominated,
    ExceedsRemainingBudget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceCandidateDecision {
    pub candidate_id: String,
    pub disposition: ResourceCandidateDisposition,
    pub dominated_by: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResourceFrontierReceipt {
    pub method_id: &'static str,
    pub policy_version: String,
    pub currency_code: String,
    pub remaining_budget: ResourceUnits,
    pub frontier_candidates: Vec<ResourceProjectionCandidate>,
    pub candidate_decisions: Vec<ResourceCandidateDecision>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceFrontierError {
    EmptyPolicyVersion,
    InvalidCurrencyCode,
    EmptyCandidateSet,
    EmptyCandidateId,
    DuplicateCandidateId,
    InvalidProjectedBenefit,
    ResourceLedgerOverflow,
    CostExceedsRemainingBudget,
    ReplayMismatch,
}

impl fmt::Display for ResourceFrontierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyPolicyVersion => "resource-frontier policy version must not be empty",
            Self::InvalidCurrencyCode => "currency code must contain exactly three uppercase ASCII letters",
            Self::EmptyCandidateSet => "resource frontier requires at least one candidate",
            Self::EmptyCandidateId => "candidate ID must be nonempty",
            Self::DuplicateCandidateId => "candidate IDs must be unique",
            Self::InvalidProjectedBenefit => "projected certificate-gap contraction must be finite and nonnegative",
            Self::ResourceLedgerOverflow => "resource charge accumulation overflowed u64",
            Self::CostExceedsRemainingBudget => "actual resource charge exceeds remaining budget",
            Self::ReplayMismatch => "resource frontier receipt does not reproduce from its contract and candidates",
        };
        f.write_str(message)
    }
}
impl std::error::Error for ResourceFrontierError {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResourceChargeLedger {
    pub charged: ResourceUnits,
}

impl ResourceChargeLedger {
    pub fn charge(&mut self, actual: ResourceUnits) -> Result<(), ResourceFrontierError> {
        self.charged = self.charged
            .checked_add(actual)
            .ok_or(ResourceFrontierError::ResourceLedgerOverflow)?;
        Ok(())
    }

    pub fn remaining(&self, total_budget: ResourceUnits) -> Result<ResourceUnits, ResourceFrontierError> {
        total_budget.checked_sub(self.charged)
            .ok_or(ResourceFrontierError::CostExceedsRemainingBudget)
    }
}

/// Componentwise feasible Pareto frontier. Exact benefit/cost duplicates are
/// canonicalized by lexicographically smallest stable candidate ID.
pub fn build_resource_feasible_pareto_frontier(
    contract: &ResourceFrontierContract,
    candidates: &[ResourceProjectionCandidate],
) -> Result<ResourceFrontierReceipt, ResourceFrontierError> {
    validate_contract(contract)?;
    if candidates.is_empty() {
        return Err(ResourceFrontierError::EmptyCandidateSet);
    }

    let mut sorted = candidates.to_vec();
    sorted.sort_by(|a,b| a.candidate_id.cmp(&b.candidate_id));
    let mut ids = BTreeSet::new();
    for candidate in &sorted {
        if candidate.candidate_id.trim().is_empty() {
            return Err(ResourceFrontierError::EmptyCandidateId);
        }
        if !ids.insert(candidate.candidate_id.as_str()) {
            return Err(ResourceFrontierError::DuplicateCandidateId);
        }
        if !candidate.projected_certificate_gap_contraction.is_finite()
            || candidate.projected_certificate_gap_contraction < 0.0
        {
            return Err(ResourceFrontierError::InvalidProjectedBenefit);
        }
    }

    let feasible = sorted.iter()
        .filter(|candidate| candidate.upper_cost.units.componentwise_leq(contract.remaining_budget))
        .cloned()
        .collect::<Vec<_>>();
    let mut frontier_candidates = Vec::new();
    let mut candidate_decisions = Vec::with_capacity(sorted.len());

    for candidate in &sorted {
        if !candidate.upper_cost.units.componentwise_leq(contract.remaining_budget) {
            candidate_decisions.push(ResourceCandidateDecision {
                candidate_id: candidate.candidate_id.clone(),
                disposition: ResourceCandidateDisposition::ExceedsRemainingBudget,
                dominated_by: None,
            });
            continue;
        }
        let dominator = feasible.iter().find(|other| {
            other.candidate_id != candidate.candidate_id && dominates(other, candidate)
        });
        if let Some(other) = dominator {
            candidate_decisions.push(ResourceCandidateDecision {
                candidate_id: candidate.candidate_id.clone(),
                disposition: ResourceCandidateDisposition::Dominated,
                dominated_by: Some(other.candidate_id.clone()),
            });
        } else {
            candidate_decisions.push(ResourceCandidateDecision {
                candidate_id: candidate.candidate_id.clone(),
                disposition: ResourceCandidateDisposition::OnParetoFrontier,
                dominated_by: None,
            });
            frontier_candidates.push(candidate.clone());
        }
    }

    frontier_candidates.sort_by(|a,b| {
        b.projected_certificate_gap_contraction.total_cmp(&a.projected_certificate_gap_contraction)
            .then_with(|| a.upper_cost.units.cmp_lexicographic(b.upper_cost.units))
            .then_with(|| a.candidate_id.cmp(&b.candidate_id))
    });

    Ok(ResourceFrontierReceipt {
        method_id: RESOURCE_FRONTIER_METHOD_ID,
        policy_version: contract.policy_version.clone(),
        currency_code: contract.currency_code.clone(),
        remaining_budget: contract.remaining_budget,
        frontier_candidates,
        candidate_decisions,
    })
}

pub fn verify_resource_frontier_replay(
    contract: &ResourceFrontierContract,
    candidates: &[ResourceProjectionCandidate],
    receipt: &ResourceFrontierReceipt,
) -> Result<(), ResourceFrontierError> {
    let expected = build_resource_feasible_pareto_frontier(contract, candidates)?;
    if &expected == receipt { Ok(()) } else { Err(ResourceFrontierError::ReplayMismatch) }
}

fn validate_contract(contract: &ResourceFrontierContract) -> Result<(), ResourceFrontierError> {
    if contract.policy_version.trim().is_empty() {
        return Err(ResourceFrontierError::EmptyPolicyVersion);
    }
    let code = contract.currency_code.as_bytes();
    if code.len() != 3 || code.iter().any(|c| !c.is_ascii_uppercase()) {
        return Err(ResourceFrontierError::InvalidCurrencyCode);
    }
    Ok(())
}

fn dominates(a: &ResourceProjectionCandidate, b: &ResourceProjectionCandidate) -> bool {
    let benefit_ge = a.projected_certificate_gap_contraction >= b.projected_certificate_gap_contraction;
    let cost_le = a.upper_cost.units.componentwise_leq(b.upper_cost.units);
    if !benefit_ge || !cost_le { return false; }
    let strict_benefit = a.projected_certificate_gap_contraction > b.projected_certificate_gap_contraction;
    let strict_cost = a.upper_cost.units.strict_cost_improvement(b.upper_cost.units);
    strict_benefit || strict_cost || a.candidate_id < b.candidate_id
}

#[cfg(test)]
mod tests {
    use super::*;

    fn units(rep:u64,cpu:u64,gpu:u64,wall:u64,human:u64,money:u64)->ResourceUnits {
        ResourceUnits {
            simulation_replications:rep,cpu_millis:cpu,gpu_millis:gpu,
            wall_millis:wall,human_seconds:human,currency_microunits:money
        }
    }
    fn contract(budget:ResourceUnits)->ResourceFrontierContract {
        ResourceFrontierContract {
            policy_version:"resource-frontier-test-v1".into(),
            currency_code:"USD".into(),remaining_budget:budget
        }
    }
    fn candidate(id:&str,benefit:f64,cost:ResourceUnits)->ResourceProjectionCandidate {
        ResourceProjectionCandidate {
            candidate_id:id.into(),projected_certificate_gap_contraction:benefit,
            upper_cost:ResourceCostUpperBound{units:cost}
        }
    }

    #[test]
    fn preserves_incomparable_resource_tradeoffs_and_filters_dominated() {
        let budget=units(100,100,100,100,100,100);
        let candidates=vec![
            candidate("cpu-cheap",0.3,units(5,10,0,10,0,1)),
            candidate("gpu-fast",0.7,units(5,0,10,10,0,1)),
            candidate("balanced",0.6,units(5,5,5,10,0,1)),
            candidate("dominated",0.2,units(6,11,1,11,1,2)),
            candidate("human-over",1.0,units(1,1,1,1,101,1)),
        ];
        let result=build_resource_feasible_pareto_frontier(&contract(budget),&candidates).unwrap();
        let frontier=result.frontier_candidates.iter().map(|c|c.candidate_id.as_str()).collect::<Vec<_>>();
        assert!(frontier.contains(&"cpu-cheap"));
        assert!(frontier.contains(&"gpu-fast"));
        assert!(frontier.contains(&"balanced"));
        assert!(!frontier.contains(&"dominated"));
        assert!(!frontier.contains(&"human-over"));
        assert!(result.candidate_decisions.iter().any(|d|d.candidate_id=="dominated" && d.disposition==ResourceCandidateDisposition::Dominated));
        assert!(result.candidate_decisions.iter().any(|d|d.candidate_id=="human-over" && d.disposition==ResourceCandidateDisposition::ExceedsRemainingBudget));
    }

    #[test]
    fn exact_duplicates_are_canonicalized_by_candidate_id() {
        let b=contract(units(10,10,10,10,10,10));
        let candidates=vec![
            candidate("z-copy",0.5,units(1,1,1,1,1,1)),
            candidate("a-copy",0.5,units(1,1,1,1,1,1))
        ];
        let r=build_resource_feasible_pareto_frontier(&b,&candidates).unwrap();
        assert_eq!(r.frontier_candidates.len(),1);
        assert_eq!(r.frontier_candidates[0].candidate_id,"a-copy");
        let duplicate=r.candidate_decisions.iter().find(|d|d.candidate_id=="z-copy").unwrap();
        assert_eq!(duplicate.disposition,ResourceCandidateDisposition::Dominated);
        assert_eq!(duplicate.dominated_by.as_deref(),Some("a-copy"));
    }

    #[test]
    fn all_six_budget_dimensions_are_hard_constraints() {
        let budget=units(10,20,30,40,50,60);
        let candidates=vec![
            candidate("fit",0.0,budget),
            candidate("rep-over",1.0,units(11,0,0,0,0,0)),
            candidate("cpu-over",1.0,units(0,21,0,0,0,0)),
            candidate("gpu-over",1.0,units(0,0,31,0,0,0)),
            candidate("wall-over",1.0,units(0,0,0,41,0,0)),
            candidate("human-over",1.0,units(0,0,0,0,51,0)),
            candidate("money-over",1.0,units(0,0,0,0,0,61))
        ];
        let r=build_resource_feasible_pareto_frontier(&contract(budget),&candidates).unwrap();
        assert_eq!(r.candidate_decisions.iter().filter(|d|d.disposition==ResourceCandidateDisposition::ExceedsRemainingBudget).count(),6);
    }

    #[test]
    fn input_order_is_irrelevant_and_replay_rejects_mutation() {
        let c=contract(units(20,20,20,20,20,20));
        let candidates=vec![
            candidate("b",0.4,units(3,5,0,5,0,2)),
            candidate("a",0.6,units(3,0,5,5,0,2)),
            candidate("c",0.2,units(4,6,1,6,1,3))
        ];
        let a=build_resource_feasible_pareto_frontier(&c,&candidates).unwrap();
        let mut reverse=candidates.clone();reverse.reverse();
        let b=build_resource_feasible_pareto_frontier(&c,&reverse).unwrap();
        assert_eq!(a,b);
        assert_eq!(verify_resource_frontier_replay(&c,&candidates,&a),Ok(()));
        let mut tampered=a.clone();tampered.frontier_candidates.pop();
        assert_eq!(verify_resource_frontier_replay(&c,&candidates,&tampered),Err(ResourceFrontierError::ReplayMismatch));
    }

    #[test]
    fn invalid_contract_and_candidates_fail_closed() {
        let budget=units(10,10,10,10,10,10);
        let one=[candidate("x",1.0,ResourceUnits::default())];
        let mut c=contract(budget);c.policy_version.clear();
        assert_eq!(build_resource_feasible_pareto_frontier(&c,&one),Err(ResourceFrontierError::EmptyPolicyVersion));
        let mut c=contract(budget);c.currency_code="usd".into();
        assert_eq!(build_resource_feasible_pareto_frontier(&c,&one),Err(ResourceFrontierError::InvalidCurrencyCode));
        assert_eq!(build_resource_feasible_pareto_frontier(&contract(budget),&[]),Err(ResourceFrontierError::EmptyCandidateSet));
        assert_eq!(build_resource_feasible_pareto_frontier(&contract(budget),&[candidate(" ",1.0,ResourceUnits::default())]),Err(ResourceFrontierError::EmptyCandidateId));
        assert_eq!(build_resource_feasible_pareto_frontier(&contract(budget),&[candidate("x",f64::NAN,ResourceUnits::default())]),Err(ResourceFrontierError::InvalidProjectedBenefit));
        assert_eq!(build_resource_feasible_pareto_frontier(&contract(budget),&[candidate("x",-0.1,ResourceUnits::default())]),Err(ResourceFrontierError::InvalidProjectedBenefit));
        assert_eq!(build_resource_feasible_pareto_frontier(&contract(budget),&one.iter().cloned().chain(one.iter().cloned()).collect::<Vec<_>>()),Err(ResourceFrontierError::DuplicateCandidateId));
    }

    #[test]
    fn resource_ledger_detects_overflow_and_budget_overrun() {
        let mut ledger=ResourceChargeLedger::default();
        ledger.charge(units(2,3,4,5,6,7)).unwrap();
        assert_eq!(ledger.remaining(units(4,6,8,10,12,14)),Ok(units(2,3,4,5,6,7)));
        assert_eq!(ledger.remaining(units(1,6,8,10,12,14)),Err(ResourceFrontierError::CostExceedsRemainingBudget));
        ledger.charged.simulation_replications=u64::MAX;
        assert_eq!(ledger.charge(units(1,0,0,0,0,0)),Err(ResourceFrontierError::ResourceLedgerOverflow));
    }
}
