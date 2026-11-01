//! Candidate-side joint temporal resource allocation.
//!
//! This is a deterministic greedy baseline, not an optimal solver. It chooses
//! among candidate-authored action bundles using conservative predicted
//! temporal-error-debt reduction, explicit per-resource dual prices and hard
//! vector budget checks. No evaluator-only outcomes, future stream, hidden
//! event times, or oracle state are represented in this crate.
//!
//! Action bundles may jointly describe sensing + memory + compute + verification.
//! Mutually exclusive alternatives share a nonzero conflict_group. The
//! evaluator must still measure realized interactions and compare against its
//! independent exact small-instance oracle.

use std::collections::BTreeSet;
use std::fmt;

pub const MAX_CANDIDATE_ACTIONS: usize = 128;
pub const PRICE_SCALE: u128 = 1_000_000;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResourceVector {
    pub persistent_bytes: u64,
    pub retained_byte_us: u128,
    pub io_bytes: u64,
    pub compute_units: u64,
    pub observations: u64,
    pub energy_nj: u64,
    pub verification_units: u64,
}

impl ResourceVector {
    pub fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            persistent_bytes: self.persistent_bytes.checked_add(other.persistent_bytes)?,
            retained_byte_us: self.retained_byte_us.checked_add(other.retained_byte_us)?,
            io_bytes: self.io_bytes.checked_add(other.io_bytes)?,
            compute_units: self.compute_units.checked_add(other.compute_units)?,
            observations: self.observations.checked_add(other.observations)?,
            energy_nj: self.energy_nj.checked_add(other.energy_nj)?,
            verification_units: self.verification_units.checked_add(other.verification_units)?,
        })
    }

    pub fn fits_within(self, budget: Self) -> bool {
        self.persistent_bytes <= budget.persistent_bytes
            && self.retained_byte_us <= budget.retained_byte_us
            && self.io_bytes <= budget.io_bytes
            && self.compute_units <= budget.compute_units
            && self.observations <= budget.observations
            && self.energy_nj <= budget.energy_nj
            && self.verification_units <= budget.verification_units
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResourcePrices {
    /// Integer dual prices per declared resource unit.
    pub persistent_bytes: u64,
    pub retained_byte_us: u64,
    pub io_bytes: u64,
    pub compute_units: u64,
    pub observations: u64,
    pub energy_nj: u64,
    pub verification_units: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CandidateAction {
    pub action_id: u64,
    /// Zero means this action is independent. Nonzero IDs are mutually
    /// exclusive choice groups; at most one action in a group is selected.
    pub conflict_group: u64,
    /// Candidate-estimated reduction in future temporal error debt.
    pub predicted_ted_reduction_us: u64,
    /// Conservative uncertainty radius subtracted before selection.
    pub uncertainty_us: u64,
    /// Required vector of costs. Missing/unknown costs are not representable.
    pub resources: ResourceVector,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointActionPlan {
    /// Action IDs in deterministic greedy selection order.
    pub selected_action_ids: Vec<u64>,
    pub resources_used: ResourceVector,
    /// Sum of conservative per-action utility estimates; not evaluator truth.
    pub conservative_predicted_ted_reduction_us: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ControllerError {
    TooManyActions { requested: usize, maximum: usize },
    ZeroActionId,
    DuplicateActionId { action_id: u64 },
    ResourceArithmeticOverflow,
    ScoreArithmeticOverflow,
    InvalidPriceUpdateRate,
    ZeroMaximumPrice,
}

impl fmt::Display for ControllerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyActions { requested, maximum } => {
                write!(f, "candidate action count {requested} exceeds maximum {maximum}")
            }
            Self::ZeroActionId => write!(f, "candidate action ID must be nonzero"),
            Self::DuplicateActionId { action_id } => {
                write!(f, "duplicate candidate action ID {action_id}")
            }
            Self::ResourceArithmeticOverflow => write!(f, "resource accounting overflowed"),
            Self::ScoreArithmeticOverflow => write!(f, "candidate utility/cost score overflowed"),
            Self::InvalidPriceUpdateRate => write!(f, "dual-price update rate must be in 1..=1000 permille"),
            Self::ZeroMaximumPrice => write!(f, "maximum dual price must be nonzero"),
        }
    }
}

impl std::error::Error for ControllerError {}

/// Select a feasible joint action plan.
///
/// The policy is intentionally simple and auditable:
/// 1. subtract each action's uncertainty radius from its predicted benefit;
/// 2. price each resource dimension independently;
/// 3. sort by positive net priced value with deterministic tie-breaks;
/// 4. greedily add actions only when every hard resource cap remains satisfied;
/// 5. choose at most one action from each nonzero conflict group.
///
/// This is a baseline, not a globally optimal multidimensional knapsack solver.
/// The prices are explicit policy inputs, not hidden unit conversions. The
/// evaluator must report realized outcomes and compare against its own oracle.
pub fn select_joint_plan(
    actions: &[CandidateAction],
    budget: ResourceVector,
    prices: ResourcePrices,
) -> Result<JointActionPlan, ControllerError> {
    if actions.len() > MAX_CANDIDATE_ACTIONS {
        return Err(ControllerError::TooManyActions {
            requested: actions.len(),
            maximum: MAX_CANDIDATE_ACTIONS,
        });
    }

    let mut ids = BTreeSet::new();
    let mut scored = Vec::with_capacity(actions.len());
    for action in actions {
        if action.action_id == 0 {
            return Err(ControllerError::ZeroActionId);
        }
        if !ids.insert(action.action_id) {
            return Err(ControllerError::DuplicateActionId {
                action_id: action.action_id,
            });
        }

        let conservative_benefit = action
            .predicted_ted_reduction_us
            .saturating_sub(action.uncertainty_us);
        if conservative_benefit == 0 {
            continue;
        }
        let net_score = priced_net_score(conservative_benefit, action.resources, prices)?;
        if net_score > 0 {
            scored.push((net_score, conservative_benefit, *action));
        }
    }

    scored.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| right.1.cmp(&left.1))
            .then_with(|| left.2.action_id.cmp(&right.2.action_id))
    });

    let mut selected_action_ids = Vec::new();
    let mut resources_used = ResourceVector::default();
    let mut conservative_predicted_ted_reduction_us = 0_u64;
    let mut selected_conflict_groups = BTreeSet::new();

    for (_, conservative_benefit, action) in scored {
        if action.conflict_group != 0
            && selected_conflict_groups.contains(&action.conflict_group)
        {
            continue;
        }

        let next_resources = resources_used
            .checked_add(action.resources)
            .ok_or(ControllerError::ResourceArithmeticOverflow)?;
        if !next_resources.fits_within(budget) {
            continue;
        }
        let next_benefit = conservative_predicted_ted_reduction_us
            .checked_add(conservative_benefit)
            .ok_or(ControllerError::ScoreArithmeticOverflow)?;

        selected_action_ids.push(action.action_id);
        resources_used = next_resources;
        conservative_predicted_ted_reduction_us = next_benefit;
        if action.conflict_group != 0 {
            selected_conflict_groups.insert(action.conflict_group);
        }
    }

    Ok(JointActionPlan {
        selected_action_ids,
        resources_used,
        conservative_predicted_ted_reduction_us,
    })
}

fn priced_net_score(
    benefit_us: u64,
    resources: ResourceVector,
    prices: ResourcePrices,
) -> Result<u128, ControllerError> {
    let benefit = u128::from(benefit_us)
        .checked_mul(PRICE_SCALE)
        .ok_or(ControllerError::ScoreArithmeticOverflow)?;
    let mut cost = 0_u128;
    let components = [
        (u128::from(resources.persistent_bytes), prices.persistent_bytes),
        (resources.retained_byte_us, prices.retained_byte_us),
        (u128::from(resources.io_bytes), prices.io_bytes),
        (u128::from(resources.compute_units), prices.compute_units),
        (u128::from(resources.observations), prices.observations),
        (u128::from(resources.energy_nj), prices.energy_nj),
        (u128::from(resources.verification_units), prices.verification_units),
    ];
    for (quantity, price) in components {
        let priced = quantity
            .checked_mul(u128::from(price))
            .ok_or(ControllerError::ScoreArithmeticOverflow)?;
        cost = cost
            .checked_add(priced)
            .ok_or(ControllerError::ScoreArithmeticOverflow)?;
    }
    Ok(benefit.saturating_sub(cost))
}

/// Update explicit resource dual prices from measured use after an episode.
///
/// Prices rise when a resource exceeds budget, relax only when utilization is
/// below 90%, and otherwise remain unchanged. This conservative multiplicative
/// rule is a benchmark baseline, not a theoretically optimal step-size scheme.
/// It never changes hard budget enforcement in select_joint_plan.
pub fn update_resource_prices(
    current: ResourcePrices,
    measured_use: ResourceVector,
    budget: ResourceVector,
    step_permille: u16,
    maximum_price: u64,
) -> Result<ResourcePrices, ControllerError> {
    if !(1..=1000).contains(&step_permille) {
        return Err(ControllerError::InvalidPriceUpdateRate);
    }
    if maximum_price == 0 {
        return Err(ControllerError::ZeroMaximumPrice);
    }

    Ok(ResourcePrices {
        persistent_bytes: update_one_price(current.persistent_bytes, measured_use.persistent_bytes.into(), budget.persistent_bytes.into(), step_permille, maximum_price),
        retained_byte_us: update_one_price(current.retained_byte_us, measured_use.retained_byte_us, budget.retained_byte_us, step_permille, maximum_price),
        io_bytes: update_one_price(current.io_bytes, measured_use.io_bytes.into(), budget.io_bytes.into(), step_permille, maximum_price),
        compute_units: update_one_price(current.compute_units, measured_use.compute_units.into(), budget.compute_units.into(), step_permille, maximum_price),
        observations: update_one_price(current.observations, measured_use.observations.into(), budget.observations.into(), step_permille, maximum_price),
        energy_nj: update_one_price(current.energy_nj, measured_use.energy_nj.into(), budget.energy_nj.into(), step_permille, maximum_price),
        verification_units: update_one_price(current.verification_units, measured_use.verification_units.into(), budget.verification_units.into(), step_permille, maximum_price),
    })
}

fn update_one_price(
    current: u64,
    measured_use: u128,
    budget: u128,
    step_permille: u16,
    maximum_price: u64,
) -> u64 {
    let step = u64::from(step_permille);
    if measured_use > budget {
        let increase = (current.saturating_mul(step) / 1000).max(1);
        current.saturating_add(increase).min(maximum_price)
    } else if budget > 0 && measured_use < budget - budget / 10 {
        let decrease = (current.saturating_mul(step) / 1000).max(1);
        current.saturating_sub(decrease)
    } else {
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resources(compute: u64, observations: u64, energy: u64) -> ResourceVector {
        ResourceVector {
            persistent_bytes: 10,
            retained_byte_us: 100,
            io_bytes: 20,
            compute_units: compute,
            observations,
            energy_nj: energy,
            verification_units: 1,
        }
    }

    fn action(
        action_id: u64,
        conflict_group: u64,
        benefit: u64,
        uncertainty: u64,
        resources: ResourceVector,
    ) -> CandidateAction {
        CandidateAction {
            action_id,
            conflict_group,
            predicted_ted_reduction_us: benefit,
            uncertainty_us: uncertainty,
            resources,
        }
    }

    fn budget() -> ResourceVector {
        ResourceVector {
            persistent_bytes: 20,
            retained_byte_us: 200,
            io_bytes: 40,
            compute_units: 10,
            observations: 10,
            energy_nj: 10,
            verification_units: 2,
        }
    }

    #[test]
    fn every_resource_dimension_is_a_hard_ceiling() {
        let actions = vec![
            action(1, 0, 100, 0, resources(3, 3, 3)),
            action(2, 0, 90, 0, resources(3, 3, 3)),
        ];
        let plan = select_joint_plan(&actions, budget(), ResourcePrices::default()).unwrap();
        assert!(plan.resources_used.fits_within(budget()));
        assert_eq!(plan.selected_action_ids.len(), 2);
    }

    #[test]
    fn one_binding_resource_rejects_an_otherwise_valuable_action() {
        let actions = vec![action(1, 0, 10_000, 0, resources(11, 1, 1))];
        let plan = select_joint_plan(&actions, budget(), ResourcePrices::default()).unwrap();
        assert!(plan.selected_action_ids.is_empty());
    }

    #[test]
    fn input_order_does_not_change_the_selected_plan() {
        let a = action(1, 0, 100, 0, resources(4, 2, 2));
        let b = action(2, 0, 90, 0, resources(4, 2, 2));
        let forward = select_joint_plan(&[a, b], budget(), ResourcePrices::default()).unwrap();
        let reverse = select_joint_plan(&[b, a], budget(), ResourcePrices::default()).unwrap();
        assert_eq!(forward, reverse);
    }

    #[test]
    fn equal_scores_use_action_id_as_deterministic_tie_break() {
        let a = action(8, 0, 100, 0, ResourceVector::default());
        let b = action(3, 0, 100, 0, ResourceVector::default());
        let plan = select_joint_plan(&[a, b], budget(), ResourcePrices::default()).unwrap();
        assert_eq!(plan.selected_action_ids, vec![3, 8]);
    }

    #[test]
    fn uncertainty_radius_is_subtracted_before_selection() {
        let actions = vec![action(1, 0, 100, 100, ResourceVector::default())];
        let plan = select_joint_plan(&actions, budget(), ResourcePrices::default()).unwrap();
        assert!(plan.selected_action_ids.is_empty());
        assert_eq!(plan.conservative_predicted_ted_reduction_us, 0);
    }

    #[test]
    fn mutually_exclusive_actions_share_one_conflict_group() {
        let actions = vec![
            action(1, 7, 100, 0, ResourceVector::default()),
            action(2, 7, 150, 0, ResourceVector::default()),
            action(3, 0, 10, 0, ResourceVector::default()),
        ];
        let plan = select_joint_plan(&actions, budget(), ResourcePrices::default()).unwrap();
        assert_eq!(plan.selected_action_ids, vec![2, 3]);
    }

    #[test]
    fn duplicate_and_zero_action_ids_fail_closed() {
        let duplicate = vec![
            action(1, 0, 10, 0, ResourceVector::default()),
            action(1, 0, 20, 0, ResourceVector::default()),
        ];
        assert_eq!(
            select_joint_plan(&duplicate, budget(), ResourcePrices::default()),
            Err(ControllerError::DuplicateActionId { action_id: 1 })
        );
        let zero = vec![action(0, 0, 10, 0, ResourceVector::default())];
        assert_eq!(
            select_joint_plan(&zero, budget(), ResourcePrices::default()),
            Err(ControllerError::ZeroActionId)
        );
    }

    #[test]
    fn resource_accounting_overflow_fails_closed() {
        let left = ResourceVector { persistent_bytes: u64::MAX, ..ResourceVector::default() };
        let right = ResourceVector { persistent_bytes: 1, ..ResourceVector::default() };
        assert_eq!(left.checked_add(right), None);
    }

    #[test]
    fn measured_overuse_raises_only_the_binding_dual_price() {
        let measured = ResourceVector { observations: 11, ..ResourceVector::default() };
        let limits = ResourceVector { observations: 10, ..ResourceVector::default() };
        let updated = update_resource_prices(
            ResourcePrices::default(), measured, limits, 100, 1_000
        ).unwrap();
        assert_eq!(updated.observations, 1);
        assert_eq!(updated.compute_units, 0);
        assert_eq!(updated.energy_nj, 0);
    }

    #[test]
    fn sustained_headroom_relaxes_prices() {
        let current = ResourcePrices { observations: 10, ..ResourcePrices::default() };
        let measured = ResourceVector { observations: 5, ..ResourceVector::default() };
        let limits = ResourceVector { observations: 10, ..ResourceVector::default() };
        let updated = update_resource_prices(current, measured, limits, 100, 1_000).unwrap();
        assert_eq!(updated.observations, 9);
    }

    #[test]
    fn zero_budget_overuse_is_visible_and_price_bounded() {
        let measured = ResourceVector { energy_nj: 1, ..ResourceVector::default() };
        let updated = update_resource_prices(
            ResourcePrices { energy_nj: 999, ..ResourcePrices::default() },
            measured,
            ResourceVector::default(),
            1_000,
            1_000,
        ).unwrap();
        assert_eq!(updated.energy_nj, 1_000);
    }

    #[test]
    fn empty_action_set_is_a_valid_noop() {
        let plan = select_joint_plan(&[], budget(), ResourcePrices::default()).unwrap();
        assert!(plan.selected_action_ids.is_empty());
        assert_eq!(plan.resources_used, ResourceVector::default());
    }
}
