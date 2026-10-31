//! Unit-safe, vector-valued acquisition budgets for multi-objective NAS.
//!
//! This module avoids silently converting compute time, device occupancy, instrument time,
//! energy, money, storage and operator time into one weighted scalar. Each action has a complete
//! resource vector, including explicit zero entries for resources it does not consume. A budget
//! is checked component-wise after subtracting spent resources and protected reserves.
//!
//! The scheduler returns the non-dominated feasible action frontier using a conservative lower
//! bound on regret reduction. It does not select one action when multiple resource/value trade-offs
//! remain incomparable; the caller must supply an explicit budget profile or policy preference.
//!
//! This module does not enforce empirical qualification tokens or run-wide pruning alpha. Those
//! remain separate mandatory gates before operational actions are executed.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const REGISTRY_SCHEMA: &str = "symthaea.nas.resource-registry";
const REGISTRY_VERSION: &str = "0.1.0";
const REGISTRY_DOMAIN: &[u8] = b"symthaea.nas.resource-registry.v0.1";
const SHA256_HEX_LEN: usize = 64;
const LOWER_BOUND_Z: f64 = 1.96;
const VALUE_TOLERANCE: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceClass {
    AcceleratorTime,
    CpuTime,
    TargetDeviceTime,
    InstrumentTime,
    Energy,
    Money,
    Storage,
    OperatorTime,
    ConcurrencySlots,
    Custom,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceDefinition {
    pub resource_id: String,
    pub unit: String,
    pub class: ResourceClass,
    /// Digest of the model/instrument used to account for this resource quantity.
    pub cost_model_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceRegistry {
    pub schema: String,
    pub version: String,
    pub resources: Vec<ResourceDefinition>,
    pub registry_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceQuantity {
    pub resource_id: String,
    pub unit: String,
    /// Integer accounting prevents accidental floating-point unit conversion or rounding.
    /// A zero value must be explicit; omission is not equivalent to zero.
    pub units: u64,
    pub cost_model_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceCostVector {
    pub action_id: String,
    pub resource_registry_sha256: String,
    /// Exactly one sorted entry for every resource in the frozen registry.
    pub resources: Vec<ResourceQuantity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceBudgetLimit {
    pub resource_id: String,
    pub unit: String,
    pub total_units: u64,
    pub already_spent_units: u64,
    pub protected_reserve_units: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceHeadroom {
    pub resource_id: String,
    pub unit: String,
    pub available_units: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceBudget {
    pub resource_registry_sha256: String,
    /// Exactly one sorted limit for every resource in the frozen registry.
    pub limits: Vec<ResourceBudgetLimit>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResourceBudgetAction {
    pub action_id: String,
    pub expected_regret_reduction: f64,
    pub numerical_standard_error: f64,
    pub resource_cost: ResourceCostVector,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionBudgetAssessment {
    pub action_id: String,
    pub conservative_regret_reduction: f64,
    /// Whether every resource-cost component fits the remaining hard budget.
    pub resource_feasible: bool,
    /// Whether the conservative lower bound on regret reduction is strictly positive.
    pub benefit_eligible: bool,
    pub blocked_resources: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResourceActionFrontier {
    pub resource_registry_sha256: String,
    pub assessments: Vec<ActionBudgetAssessment>,
    /// Non-dominated actions. No scalar exchange rate between different resource types is used.
    pub frontier_action_ids: Vec<String>,
    pub blocked_action_ids: Vec<String>,
    pub resource_headroom: Vec<ResourceHeadroom>,
    pub lower_bound_z: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceBudgetError {
    pub field: String,
    pub message: String,
}

impl ResourceBudgetError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ResourceBudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for ResourceBudgetError {}

pub fn compute_resource_registry_sha256(
    registry: &ResourceRegistry,
) -> Result<String, ResourceBudgetError> {
    let mut canonical = registry.clone();
    canonical.resources.sort_by(|left, right| left.resource_id.cmp(&right.resource_id));
    canonical.registry_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| ResourceBudgetError::new("registry", error.to_string()))?;
    Ok(digest_bytes(REGISTRY_DOMAIN, &bytes))
}

pub fn validate_resource_registry(
    registry: &ResourceRegistry,
) -> Result<(), ResourceBudgetError> {
    if registry.schema != REGISTRY_SCHEMA || registry.version != REGISTRY_VERSION {
        return Err(ResourceBudgetError::new(
            "schema_version",
            "resource registry schema or version is unsupported",
        ));
    }
    if registry.resources.is_empty() {
        return Err(ResourceBudgetError::new(
            "resources",
            "at least one typed resource is required",
        ));
    }
    if !is_sha256(&registry.registry_sha256) {
        return Err(ResourceBudgetError::new(
            "registry_sha256",
            "must be a lowercase hexadecimal SHA-256 digest",
        ));
    }
    let mut ids = HashSet::with_capacity(registry.resources.len());
    for (index, resource) in registry.resources.iter().enumerate() {
        if resource.resource_id.trim().is_empty()
            || resource.unit.trim().is_empty()
            || !ids.insert(resource.resource_id.as_str())
        {
            return Err(ResourceBudgetError::new(
                format!("resources[{index}]"),
                "resource IDs and units must be non-empty and resource IDs unique",
            ));
        }
        if !is_sha256(&resource.cost_model_sha256) {
            return Err(ResourceBudgetError::new(
                format!("resources[{index}].cost_model_sha256"),
                "must be a lowercase hexadecimal SHA-256 digest",
            ));
        }
    }
    if registry
        .resources
        .windows(2)
        .any(|pair| pair[0].resource_id >= pair[1].resource_id)
    {
        return Err(ResourceBudgetError::new(
            "resources",
            "resources must be strictly sorted by resource_id",
        ));
    }
    if compute_resource_registry_sha256(registry)? != registry.registry_sha256 {
        return Err(ResourceBudgetError::new(
            "registry_sha256",
            "does not match the canonical resource registry",
        ));
    }
    Ok(())
}

/// Validate a budget vector and compute remaining headroom after spent amounts and reserves.
pub fn remaining_resource_headroom(
    registry: &ResourceRegistry,
    budget: &ResourceBudget,
) -> Result<Vec<ResourceBudgetLimit>, ResourceBudgetError> {
    validate_resource_registry(registry)?;
    if budget.resource_registry_sha256 != registry.registry_sha256 {
        return Err(ResourceBudgetError::new(
            "resource_registry_sha256",
            "budget is bound to a different resource registry",
        ));
    }
    if budget.limits.len() != registry.resources.len() {
        return Err(ResourceBudgetError::new(
            "limits",
            "budget must include exactly one explicit entry for every registered resource",
        ));
    }
    let mut headroom = Vec::with_capacity(registry.resources.len());
    for (index, definition) in registry.resources.iter().enumerate() {
        let limit = &budget.limits[index];
        if limit.resource_id != definition.resource_id || limit.unit != definition.unit {
            return Err(ResourceBudgetError::new(
                format!("limits[{index}]"),
                "budget resource IDs and units must exactly match registry order",
            ));
        }
        let committed = limit
            .already_spent_units
            .checked_add(limit.protected_reserve_units)
            .ok_or_else(|| {
                ResourceBudgetError::new(
                    format!("limits[{index}]"),
                    "spent plus protected reserve overflows u64",
                )
            })?;
        let remaining = limit.total_units.checked_sub(committed).ok_or_else(|| {
            ResourceBudgetError::new(
                format!("limits[{index}]"),
                "spent resources plus protected reserves exceed the total budget",
            )
        })?;
        headroom.push(ResourceHeadroom {
            resource_id: limit.resource_id.clone(),
            unit: limit.unit.clone(),
            available_units: remaining,
        });
    }
    Ok(headroom)
}

fn validate_cost_vector(
    registry: &ResourceRegistry,
    vector: &ResourceCostVector,
) -> Result<(), ResourceBudgetError> {
    if vector.action_id.trim().is_empty() {
        return Err(ResourceBudgetError::new(
            "action_id",
            "must be non-empty",
        ));
    }
    if vector.resource_registry_sha256 != registry.registry_sha256 {
        return Err(ResourceBudgetError::new(
            "resource_registry_sha256",
            "action cost was created against a different resource registry",
        ));
    }
    if vector.resources.len() != registry.resources.len() {
        return Err(ResourceBudgetError::new(
            "resources",
            "action cost must explicitly state every resource (including zero quantities)",
        ));
    }
    for (index, (resource, definition)) in vector
        .resources
        .iter()
        .zip(&registry.resources)
        .enumerate()
    {
        if resource.resource_id != definition.resource_id
            || resource.unit != definition.unit
            || resource.cost_model_sha256 != definition.cost_model_sha256
        {
            return Err(ResourceBudgetError::new(
                format!("resources[{index}]"),
                "resource identity, unit or cost model does not match the frozen registry",
            ));
        }
    }
    if !vector.resources.iter().any(|resource| resource.units > 0) {
        return Err(ResourceBudgetError::new(
            "resources",
            "an action must consume at least one resource",
        ));
    }
    Ok(())
}

/// Evaluate actions under hard per-resource budgets and return their Pareto frontier.
///
/// An action is feasible only if every resource quantity fits the corresponding headroom.
/// Its benefit is conservatively estimated as expected regret reduction - z * numerical SE.
/// The function never converts a resource vector into one weighted scalar.
pub fn plan_resource_action_frontier(
    registry: &ResourceRegistry,
    budget: &ResourceBudget,
    actions: &[ResourceBudgetAction],
) -> Result<ResourceActionFrontier, ResourceBudgetError> {
    let headroom = remaining_resource_headroom(registry, budget)?;
    if actions.is_empty() {
        return Err(ResourceBudgetError::new(
            "actions",
            "at least one candidate action is required",
        ));
    }

    let mut seen_ids = HashSet::with_capacity(actions.len());
    let mut assessments = Vec::with_capacity(actions.len());
    let mut feasible_indices = Vec::new();

    for (index, action) in actions.iter().enumerate() {
        if action.action_id.trim().is_empty() || !seen_ids.insert(action.action_id.as_str()) {
            return Err(ResourceBudgetError::new(
                format!("actions[{index}].action_id"),
                "action IDs must be non-empty and unique",
            ));
        }
        if action.resource_cost.action_id != action.action_id {
            return Err(ResourceBudgetError::new(
                format!("actions[{index}].resource_cost.action_id"),
                "resource cost vector belongs to a different action",
            ));
        }
        if !action.expected_regret_reduction.is_finite()
            || !action.numerical_standard_error.is_finite()
            || action.numerical_standard_error < 0.0
        {
            return Err(ResourceBudgetError::new(
                format!("actions[{index}]"),
                "regret reduction and numerical error must be finite; error must be non-negative",
            ));
        }
        validate_cost_vector(registry, &action.resource_cost)?;
        let conservative_regret_reduction =
            action.expected_regret_reduction - LOWER_BOUND_Z * action.numerical_standard_error;
        let blocked_resources: Vec<String> = action
            .resource_cost
            .resources
            .iter()
            .zip(&headroom)
            .filter_map(|(cost, available)| {
                (cost.units > available.available_units).then(|| cost.resource_id.clone())
            })
            .collect();
        let resource_feasible = blocked_resources.is_empty();
        let benefit_eligible = conservative_regret_reduction > 0.0;
        if resource_feasible && benefit_eligible {
            feasible_indices.push(index);
        }
        assessments.push(ActionBudgetAssessment {
            action_id: action.action_id.clone(),
            conservative_regret_reduction,
            resource_feasible,
            benefit_eligible,
            blocked_resources,
        });
    }

    let mut frontier_indices = Vec::new();
    for candidate_index in &feasible_indices {
        let candidate = &actions[*candidate_index];
        let candidate_benefit = assessments[*candidate_index].conservative_regret_reduction;
        let dominated = feasible_indices.iter().any(|other_index| {
            if other_index == candidate_index {
                return false;
            }
            let other = &actions[*other_index];
            let other_benefit = assessments[*other_index].conservative_regret_reduction;
            let no_worse = other_benefit + VALUE_TOLERANCE >= candidate_benefit
                && other
                    .resource_cost
                    .resources
                    .iter()
                    .zip(&candidate.resource_cost.resources)
                    .all(|(other_cost, candidate_cost)| other_cost.units <= candidate_cost.units);
            let strictly_better = other_benefit > candidate_benefit + VALUE_TOLERANCE
                || other
                    .resource_cost
                    .resources
                    .iter()
                    .zip(&candidate.resource_cost.resources)
                    .any(|(other_cost, candidate_cost)| other_cost.units < candidate_cost.units);
            no_worse && strictly_better
        });
        if !dominated {
            frontier_indices.push(*candidate_index);
        }
    }

    frontier_indices.sort_by(|left, right| {
        actions[*left].action_id.cmp(&actions[*right].action_id)
    });
    let frontier_action_ids = frontier_indices
        .iter()
        .map(|index| actions[*index].action_id.clone())
        .collect();
    let blocked_action_ids = assessments
        .iter()
        .filter(|assessment| !assessment.resource_feasible || !assessment.benefit_eligible)
        .map(|assessment| assessment.action_id.clone())
        .collect();

    Ok(ResourceActionFrontier {
        resource_registry_sha256: registry.registry_sha256.clone(),
        assessments,
        frontier_action_ids,
        blocked_action_ids,
        resource_headroom: headroom,
        lower_bound_z: LOWER_BOUND_Z,
    })
}


fn is_sha256(value: &str) -> bool {
    value.len() == SHA256_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest_bytes(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn registry() -> ResourceRegistry {
        let mut value = ResourceRegistry {
            schema: REGISTRY_SCHEMA.into(),
            version: REGISTRY_VERSION.into(),
            resources: vec![
                ResourceDefinition {
                    resource_id: "device-time".into(),
                    unit: "milliseconds".into(),
                    class: ResourceClass::TargetDeviceTime,
                    cost_model_sha256: sha('a'),
                },
                ResourceDefinition {
                    resource_id: "energy".into(),
                    unit: "microjoules".into(),
                    class: ResourceClass::Energy,
                    cost_model_sha256: sha('b'),
                },
                ResourceDefinition {
                    resource_id: "gpu-time".into(),
                    unit: "milliseconds".into(),
                    class: ResourceClass::AcceleratorTime,
                    cost_model_sha256: sha('c'),
                },
            ],
            registry_sha256: String::new(),
        };
        value.registry_sha256 = compute_resource_registry_sha256(&value).unwrap();
        value
    }

    fn budget() -> ResourceBudget {
        ResourceBudget {
            resource_registry_sha256: registry().registry_sha256,
            limits: vec![
                ResourceBudgetLimit {
                    resource_id: "device-time".into(),
                    unit: "milliseconds".into(),
                    total_units: 10_000,
                    already_spent_units: 2_000,
                    protected_reserve_units: 1_000,
                },
                ResourceBudgetLimit {
                    resource_id: "energy".into(),
                    unit: "microjoules".into(),
                    total_units: 50_000,
                    already_spent_units: 10_000,
                    protected_reserve_units: 5_000,
                },
                ResourceBudgetLimit {
                    resource_id: "gpu-time".into(),
                    unit: "milliseconds".into(),
                    total_units: 20_000,
                    already_spent_units: 4_000,
                    protected_reserve_units: 2_000,
                },
            ],
        }
    }

    fn cost(action_id: &str, device: u64, energy: u64, gpu: u64) -> ResourceCostVector {
        ResourceCostVector {
            action_id: action_id.into(),
            resource_registry_sha256: registry().registry_sha256,
            resources: vec![
                ResourceQuantity {
                    resource_id: "device-time".into(),
                    unit: "milliseconds".into(),
                    units: device,
                    cost_model_sha256: sha('a'),
                },
                ResourceQuantity {
                    resource_id: "energy".into(),
                    unit: "microjoules".into(),
                    units: energy,
                    cost_model_sha256: sha('b'),
                },
                ResourceQuantity {
                    resource_id: "gpu-time".into(),
                    unit: "milliseconds".into(),
                    units: gpu,
                    cost_model_sha256: sha('c'),
                },
            ],
        }
    }

    fn action(id: &str, benefit: f64, se: f64, cost: ResourceCostVector) -> ResourceBudgetAction {
        ResourceBudgetAction {
            action_id: id.into(),
            expected_regret_reduction: benefit,
            numerical_standard_error: se,
            resource_cost: cost,
        }
    }

    #[test]
    fn resource_registry_digest_is_order_invariant() {
        let first = registry();
        let mut second = first.clone();
        second.resources.reverse();
        assert_eq!(
            first.registry_sha256,
            compute_resource_registry_sha256(&second).unwrap()
        );
    }

    #[test]
    fn rejects_noncanonical_resource_registry_order() {
        let mut invalid = registry();
        invalid.resources.reverse();
        invalid.registry_sha256 = compute_resource_registry_sha256(&invalid).unwrap();
        assert!(validate_resource_registry(&invalid).is_err());
    }

    #[test]
    fn rejects_stale_registry_digest_and_changed_cost_model() {
        let mut changed = registry();
        changed.resources[0].cost_model_sha256 = sha('f');
        assert!(validate_resource_registry(&changed).is_err());
        changed.registry_sha256 = compute_resource_registry_sha256(&changed).unwrap();
        assert_ne!(registry().registry_sha256, changed.registry_sha256);
    }

    #[test]
    fn budget_subtracts_spent_and_protected_reserves_per_resource() {
        let headroom = remaining_resource_headroom(&registry(), &budget()).unwrap();
        assert_eq!(headroom[0].total_units, 7_000);
        assert_eq!(headroom[1].total_units, 35_000);
        assert_eq!(headroom[2].total_units, 14_000);
    }

    #[test]
    fn rejects_unknown_or_reordered_budget_dimensions() {
        let mut bad = budget();
        bad.limits.swap(0, 1);
        assert!(remaining_resource_headroom(&registry(), &bad).is_err());
    }

    #[test]
    fn rejects_spent_plus_reserve_overflow() {
        let mut bad = budget();
        bad.limits[0].already_spent_units = u64::MAX;
        bad.limits[0].protected_reserve_units = 1;
        assert!(remaining_resource_headroom(&registry(), &bad).is_err());
    }

    #[test]
    fn rejects_spend_and_reserve_that_exceed_total_budget() {
        let mut bad = budget();
        bad.limits[0].already_spent_units = 9_500;
        bad.limits[0].protected_reserve_units = 1_000;
        assert!(remaining_resource_headroom(&registry(), &bad).is_err());
    }

    #[test]
    fn action_cost_requires_explicit_zero_for_unused_resources() {
        let missing = ResourceCostVector {
            action_id: "a".into(),
            resource_registry_sha256: registry().registry_sha256,
            resources: vec![ResourceQuantity {
                resource_id: "device-time".into(),
                unit: "milliseconds".into(),
                units: 1,
                cost_model_sha256: sha('a'),
            }],
        };
        assert!(validate_cost_vector(&registry(), &missing).is_err());
    }

    #[test]
    fn action_cost_rejects_mismatched_unit_or_cost_model() {
        let mut invalid = cost("a", 1, 0, 0);
        invalid.resources[1].unit = "joules".into();
        assert!(validate_cost_vector(&registry(), &invalid).is_err());
        invalid = cost("a", 1, 0, 0);
        invalid.resources[1].cost_model_sha256 = sha('d');
        assert!(validate_cost_vector(&registry(), &invalid).is_err());
    }

    #[test]
    fn action_cost_rejects_all_zero_vector() {
        assert!(validate_cost_vector(&registry(), &cost("a", 0, 0, 0)).is_err());
    }

    #[test]
    fn budget_feasibility_is_component_wise_not_scalarized() {
        let actions = vec![
            action("low-device-high-energy", 0.5, 0.01, cost("low-device-high-energy", 1_000, 30_000, 0)),
            action("balanced", 0.4, 0.01, cost("balanced", 4_000, 10_000, 4_000)),
            action("too-much-energy", 0.9, 0.01, cost("too-much-energy", 1_000, 40_001, 0)),
        ];
        let result = plan_resource_action_frontier(&registry(), &budget(), &actions).unwrap();
        assert_eq!(result.blocked_action_ids, vec!["too-much-energy"]);
        assert!(result.frontier_action_ids.contains(&"low-device-high-energy".into()));
        assert!(result.frontier_action_ids.contains(&"balanced".into()));
    }

    #[test]
    fn higher_benefit_lower_cost_action_dominates_another() {
        let actions = vec![
            action("dominated", 0.4, 0.0, cost("dominated", 2_000, 20_000, 2_000)),
            action("dominant", 0.6, 0.0, cost("dominant", 1_000, 10_000, 1_000)),
        ];
        let result = plan_resource_action_frontier(&registry(), &budget(), &actions).unwrap();
        assert_eq!(result.frontier_action_ids, vec!["dominant"]);
    }

    #[test]
    fn distinguishes_resource_feasibility_from_nonpositive_benefit() {
        let actions = vec![action(
            "low-value-within-budget",
            0.0,
            0.0,
            cost("low-value-within-budget", 1, 1, 1),
        )];
        let result = plan_resource_action_frontier(&registry(), &budget(), &actions).unwrap();
        assert!(result.assessments[0].resource_feasible);
        assert!(!result.assessments[0].benefit_eligible);
        assert!(result.frontier_action_ids.is_empty());
        assert_eq!(result.blocked_action_ids, vec!["low-value-within-budget"]);
    }

    #[test]
    fn conservative_lower_bound_blocks_numerically_ambiguous_action() {
        let actions = vec![action("uncertain", 0.02, 0.02, cost("uncertain", 1, 1, 1))];
        let result = plan_resource_action_frontier(&registry(), &budget(), &actions).unwrap();
        assert!(result.frontier_action_ids.is_empty());
        assert_eq!(result.blocked_action_ids, vec!["uncertain"]);
    }

    #[test]
    fn action_cannot_be_bound_to_different_registry_or_cost_id() {
        let mut wrong_registry = cost("wrong", 1, 1, 1);
        wrong_registry.resource_registry_sha256 = sha('f');
        assert!(validate_cost_vector(&registry(), &wrong_registry).is_err());

        let mut wrong_action_id = action("right", 0.2, 0.0, cost("wrong", 1, 1, 1));
        assert!(plan_resource_action_frontier(&registry(), &budget(), &[wrong_action_id.clone()]).is_err());
        wrong_action_id.resource_cost.action_id = "right".into();
        assert!(plan_resource_action_frontier(&registry(), &budget(), &[wrong_action_id]).is_ok());
    }

    #[test]
    fn non_finite_benefit_and_duplicate_action_ids_fail_closed() {
        let actions = vec![action("bad", f64::NAN, 0.0, cost("bad", 1, 1, 1))];
        assert!(plan_resource_action_frontier(&registry(), &budget(), &actions).is_err());
        let duplicated = vec![
            action("same", 0.2, 0.0, cost("same", 1, 1, 1)),
            action("same", 0.3, 0.0, cost("same", 2, 2, 2)),
        ];
        assert!(plan_resource_action_frontier(&registry(), &budget(), &duplicated).is_err());
    }
}
