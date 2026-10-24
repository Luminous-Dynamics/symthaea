// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Finite shared-resource screening for resilient systems.
//!
//! Unlike the companion continuity screen, this module treats resource stocks as
//! finite shared state: services compete for energy, water, material, fuel, or
//! other caller-defined resources, replenishment is bounded by storage capacity,
//! and resource-supply disruptions can propagate into service deficits.
//!
//! This is a deterministic screening abstraction, not a physical simulator or a
//! policy recommender. Service priorities are an explicit allocation policy;
//! results must be inspected per service because lower-priority services can be
//! starved. Resource and service units remain domain-defined and must not be
//! summed across unlike quantities.

use std::collections::{HashMap, HashSet};

const EPSILON: f64 = 1.0e-12;

#[derive(Clone, Debug, PartialEq)]
pub struct ResourceRequirement {
    /// Identifier of a declared resource.
    pub resource_id: String,
    /// Units of resource consumed per one delivered service unit.
    pub units_per_service_unit: f64,
}

impl ResourceRequirement {
    pub fn new(resource_id: impl Into<String>, units_per_service_unit: f64) -> Self {
        Self {
            resource_id: resource_id.into(),
            units_per_service_unit,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SharedResource {
    pub id: String,
    /// Inventory available before replenishment at step zero.
    pub initial_stock: f64,
    /// Hard inventory ceiling, in the same units as initial_stock.
    pub storage_capacity: f64,
    /// Nominal replenishment per discrete time step.
    pub replenishment_per_step: f64,
}

impl SharedResource {
    pub fn new(
        id: impl Into<String>,
        initial_stock: f64,
        storage_capacity: f64,
        replenishment_per_step: f64,
    ) -> Self {
        Self {
            id: id.into(),
            initial_stock,
            storage_capacity,
            replenishment_per_step,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResourceService {
    pub id: String,
    /// Required delivered service per time step, in a caller-defined unit.
    pub demand: f64,
    /// Nominal maximum delivered service per time step.
    pub capacity: f64,
    /// Lower numbers are served first when independent services compete for stock.
    /// This is a declared policy input, not an ethically or socially optimal order.
    pub priority: u32,
    /// Required services that bottleneck this service by their fulfillment fraction.
    pub depends_on: Vec<String>,
    /// Finite shared resources consumed per delivered service unit.
    pub resource_requirements: Vec<ResourceRequirement>,
}

impl ResourceService {
    pub fn new(
        id: impl Into<String>,
        demand: f64,
        capacity: f64,
        priority: u32,
        depends_on: impl IntoIterator<Item = impl Into<String>>,
        resource_requirements: impl IntoIterator<Item = ResourceRequirement>,
    ) -> Self {
        Self {
            id: id.into(),
            demand,
            capacity,
            priority,
            depends_on: depends_on.into_iter().map(Into::into).collect(),
            resource_requirements: resource_requirements.into_iter().collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResourceSupplyDisruption {
    pub resource_id: String,
    pub start_step: usize,
    /// Number of steps from the initial supply multiplier to full nominal supply.
    pub recovery_steps: usize,
    /// Supply multiplier at start_step, in [0, 1].
    pub supply_factor_at_start: f64,
}

impl ResourceSupplyDisruption {
    pub fn new(
        resource_id: impl Into<String>,
        start_step: usize,
        recovery_steps: usize,
        supply_factor_at_start: f64,
    ) -> Self {
        Self {
            resource_id: resource_id.into(),
            start_step,
            recovery_steps,
            supply_factor_at_start,
        }
    }

    fn factor_at(&self, step: usize) -> f64 {
        if step < self.start_step {
            return 1.0;
        }
        let elapsed = step - self.start_step;
        if elapsed >= self.recovery_steps {
            return 1.0;
        }
        self.supply_factor_at_start
            + (1.0 - self.supply_factor_at_start) * elapsed as f64
                / self.recovery_steps as f64
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResourceScreeningConfig {
    pub horizon_steps: usize,
    pub service_floor: f64,
}

impl ResourceScreeningConfig {
    pub fn new(horizon_steps: usize, service_floor: f64) -> Self {
        Self {
            horizon_steps,
            service_floor,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResourceServiceTrace {
    pub service_id: String,
    pub delivered_service: Vec<f64>,
    pub fulfillment_fraction: Vec<f64>,
    pub unmet_demand: Vec<f64>,
    pub minimum_fulfillment_fraction: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SharedResourceTrace {
    pub resource_id: String,
    /// Inventory remaining after all services have been allocated at each step.
    pub inventory_after_step: Vec<f64>,
    /// Actual replenishment that fit under storage capacity at each step.
    pub replenishment_added: Vec<f64>,
    /// Resource consumed by services at each step.
    pub consumed: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResourceScreeningReport {
    pub horizon_steps: usize,
    pub service_floor: f64,
    pub first_floor_violation_step: Option<usize>,
    pub last_floor_violation_step: Option<usize>,
    /// First step from which all services stay above their floor for the remainder
    /// of the horizon. None means stable recovery is not observed in the horizon.
    pub stable_recovery_step: Option<usize>,
    pub recovery_delay_steps: Option<usize>,
    pub services: Vec<ResourceServiceTrace>,
    pub resources: Vec<SharedResourceTrace>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResourceScreeningError {
    EmptyServiceSet,
    ZeroHorizon,
    InvalidServiceFloor,
    EmptyIdentifier,
    DuplicateService(String),
    DuplicateResource(String),
    InvalidDemand(String),
    InvalidCapacity(String),
    InvalidInitialStock(String),
    InvalidStorageCapacity(String),
    InvalidReplenishment(String),
    UnknownDependency { service_id: String, dependency_id: String },
    DuplicateDependency { service_id: String, dependency_id: String },
    DependencyCycle(String),
    UnknownResourceRequirement { service_id: String, resource_id: String },
    DuplicateResourceRequirement { service_id: String, resource_id: String },
    InvalidResourceRate { service_id: String, resource_id: String },
    UnknownDisruptionResource(String),
    DisruptionStartsOutsideHorizon(String),
    ZeroRecoverySteps(String),
    InvalidDisruptionFactor(String),
}

impl std::fmt::Display for ResourceScreeningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyServiceSet => write!(f, "at least one service is required"),
            Self::ZeroHorizon => write!(f, "horizon_steps must be greater than zero"),
            Self::InvalidServiceFloor => write!(f, "service_floor must be finite and in [0, 1]"),
            Self::EmptyIdentifier => write!(f, "identifiers must not be empty"),
            Self::DuplicateService(id) => write!(f, "duplicate service id: {id}"),
            Self::DuplicateResource(id) => write!(f, "duplicate resource id: {id}"),
            Self::InvalidDemand(id) => write!(f, "demand must be finite and positive for {id}"),
            Self::InvalidCapacity(id) => {
                write!(f, "capacity must be finite and non-negative for {id}")
            }
            Self::InvalidInitialStock(id) => {
                write!(f, "initial stock must be finite and non-negative for {id}")
            }
            Self::InvalidStorageCapacity(id) => {
                write!(f, "storage capacity must be finite and at least initial stock for {id}")
            }
            Self::InvalidReplenishment(id) => {
                write!(f, "replenishment must be finite and non-negative for {id}")
            }
            Self::UnknownDependency { service_id, dependency_id } => {
                write!(f, "service {service_id} depends on unknown service {dependency_id}")
            }
            Self::DuplicateDependency { service_id, dependency_id } => {
                write!(f, "service {service_id} repeats dependency {dependency_id}")
            }
            Self::DependencyCycle(id) => write!(f, "dependency cycle includes service {id}"),
            Self::UnknownResourceRequirement { service_id, resource_id } => {
                write!(f, "service {service_id} requires unknown resource {resource_id}")
            }
            Self::DuplicateResourceRequirement { service_id, resource_id } => {
                write!(f, "service {service_id} repeats resource {resource_id}")
            }
            Self::InvalidResourceRate { service_id, resource_id } => write!(
                f,
                "resource use rate must be finite and positive for {service_id}/{resource_id}"
            ),
            Self::UnknownDisruptionResource(id) => {
                write!(f, "disruption references unknown resource {id}")
            }
            Self::DisruptionStartsOutsideHorizon(id) => {
                write!(f, "disruption for {id} starts outside the simulation horizon")
            }
            Self::ZeroRecoverySteps(id) => {
                write!(f, "disruption for {id} must have recovery_steps > 0")
            }
            Self::InvalidDisruptionFactor(id) => {
                write!(f, "supply factor for {id} must be finite and in [0, 1]")
            }
        }
    }
}

impl std::error::Error for ResourceScreeningError {}

fn compare_service_indices(
    left: usize,
    right: usize,
    services: &[ResourceService],
) -> std::cmp::Ordering {
    services[left]
        .priority
        .cmp(&services[right].priority)
        .then_with(|| services[left].id.cmp(&services[right].id))
}

fn validate_and_order_services(
    services: &[ResourceService],
    resources: &[SharedResource],
) -> Result<(HashMap<String, usize>, HashMap<String, usize>, Vec<usize>), ResourceScreeningError> {
    if services.is_empty() {
        return Err(ResourceScreeningError::EmptyServiceSet);
    }

    let mut service_index = HashMap::with_capacity(services.len());
    for (i, service) in services.iter().enumerate() {
        if service.id.trim().is_empty() {
            return Err(ResourceScreeningError::EmptyIdentifier);
        }
        if service_index.insert(service.id.clone(), i).is_some() {
            return Err(ResourceScreeningError::DuplicateService(service.id.clone()));
        }
        if !service.demand.is_finite() || service.demand <= 0.0 {
            return Err(ResourceScreeningError::InvalidDemand(service.id.clone()));
        }
        if !service.capacity.is_finite() || service.capacity < 0.0 {
            return Err(ResourceScreeningError::InvalidCapacity(service.id.clone()));
        }
    }

    let mut resource_index = HashMap::with_capacity(resources.len());
    for (i, resource) in resources.iter().enumerate() {
        if resource.id.trim().is_empty() {
            return Err(ResourceScreeningError::EmptyIdentifier);
        }
        if resource_index.insert(resource.id.clone(), i).is_some() {
            return Err(ResourceScreeningError::DuplicateResource(resource.id.clone()));
        }
        if !resource.initial_stock.is_finite() || resource.initial_stock < 0.0 {
            return Err(ResourceScreeningError::InvalidInitialStock(resource.id.clone()));
        }
        if !resource.storage_capacity.is_finite()
            || resource.storage_capacity < resource.initial_stock
        {
            return Err(ResourceScreeningError::InvalidStorageCapacity(resource.id.clone()));
        }
        if !resource.replenishment_per_step.is_finite() || resource.replenishment_per_step < 0.0 {
            return Err(ResourceScreeningError::InvalidReplenishment(resource.id.clone()));
        }
    }

    for service in services {
        let mut seen_dependencies = HashSet::new();
        for dependency_id in &service.depends_on {
            if !service_index.contains_key(dependency_id) {
                return Err(ResourceScreeningError::UnknownDependency {
                    service_id: service.id.clone(),
                    dependency_id: dependency_id.clone(),
                });
            }
            if !seen_dependencies.insert(dependency_id) {
                return Err(ResourceScreeningError::DuplicateDependency {
                    service_id: service.id.clone(),
                    dependency_id: dependency_id.clone(),
                });
            }
        }

        let mut seen_resources = HashSet::new();
        for requirement in &service.resource_requirements {
            if !resource_index.contains_key(&requirement.resource_id) {
                return Err(ResourceScreeningError::UnknownResourceRequirement {
                    service_id: service.id.clone(),
                    resource_id: requirement.resource_id.clone(),
                });
            }
            if !seen_resources.insert(&requirement.resource_id) {
                return Err(ResourceScreeningError::DuplicateResourceRequirement {
                    service_id: service.id.clone(),
                    resource_id: requirement.resource_id.clone(),
                });
            }
            if !requirement.units_per_service_unit.is_finite()
                || requirement.units_per_service_unit <= 0.0
            {
                return Err(ResourceScreeningError::InvalidResourceRate {
                    service_id: service.id.clone(),
                    resource_id: requirement.resource_id.clone(),
                });
            }
        }
    }

    fn visit(
        i: usize,
        services: &[ResourceService],
        service_index: &HashMap<String, usize>,
        state: &mut [u8],
        order: &mut Vec<usize>,
    ) -> Result<(), ResourceScreeningError> {
        match state[i] {
            2 => return Ok(()),
            1 => return Err(ResourceScreeningError::DependencyCycle(services[i].id.clone())),
            _ => {}
        }
        state[i] = 1;
        let mut dependencies: Vec<usize> = services[i]
            .depends_on
            .iter()
            .map(|id| service_index[id])
            .collect();
        dependencies.sort_by(|a, b| compare_service_indices(*a, *b, services));
        for dependency in dependencies {
            visit(dependency, services, service_index, state, order)?;
        }
        state[i] = 2;
        order.push(i);
        Ok(())
    }

    let mut roots: Vec<usize> = (0..services.len()).collect();
    roots.sort_by(|a, b| compare_service_indices(*a, *b, services));
    let mut state = vec![0; services.len()];
    let mut order = Vec::with_capacity(services.len());
    for i in roots {
        visit(i, services, &service_index, &mut state, &mut order)?;
    }
    Ok((service_index, resource_index, order))
}

/// Run a finite-stock service-continuity scenario.
///
/// At the start of each step, each resource is replenished up to its storage cap.
/// Supply disruptions scale replenishment, not already-stored inventory. Services
/// are processed in dependency-safe priority order. Each service receives at most
/// its demand, nominal capacity, and the amount permitted by every required
/// resource's current stock. The weakest dependency fulfillment fraction scales
/// nominal service capacity. Resource stocks are decremented by actual delivered
/// service times the declared per-unit resource use.
///
/// This policy is deterministic, but priority can starve lower-ranked services.
/// Inspect per-service traces and evaluate alternative priority policies separately.
pub fn screen_resource_constrained_services(
    services: &[ResourceService],
    resources: &[SharedResource],
    disruptions: &[ResourceSupplyDisruption],
    config: &ResourceScreeningConfig,
) -> Result<ResourceScreeningReport, ResourceScreeningError> {
    if config.horizon_steps == 0 {
        return Err(ResourceScreeningError::ZeroHorizon);
    }
    if !config.service_floor.is_finite() || !(0.0..=1.0).contains(&config.service_floor) {
        return Err(ResourceScreeningError::InvalidServiceFloor);
    }

    let (service_index, resource_index, order) =
        validate_and_order_services(services, resources)?;
    for disruption in disruptions {
        if !resource_index.contains_key(&disruption.resource_id) {
            return Err(ResourceScreeningError::UnknownDisruptionResource(
                disruption.resource_id.clone(),
            ));
        }
        if disruption.start_step >= config.horizon_steps {
            return Err(ResourceScreeningError::DisruptionStartsOutsideHorizon(
                disruption.resource_id.clone(),
            ));
        }
        if disruption.recovery_steps == 0 {
            return Err(ResourceScreeningError::ZeroRecoverySteps(
                disruption.resource_id.clone(),
            ));
        }
        if !disruption.supply_factor_at_start.is_finite()
            || !(0.0..=1.0).contains(&disruption.supply_factor_at_start)
        {
            return Err(ResourceScreeningError::InvalidDisruptionFactor(
                disruption.resource_id.clone(),
            ));
        }
    }

    let mut inventory: Vec<f64> = resources.iter().map(|r| r.initial_stock).collect();
    let mut service_traces: Vec<ResourceServiceTrace> = services
        .iter()
        .map(|service| ResourceServiceTrace {
            service_id: service.id.clone(),
            delivered_service: Vec::with_capacity(config.horizon_steps),
            fulfillment_fraction: Vec::with_capacity(config.horizon_steps),
            unmet_demand: Vec::with_capacity(config.horizon_steps),
            minimum_fulfillment_fraction: 1.0,
        })
        .collect();
    let mut resource_traces: Vec<SharedResourceTrace> = resources
        .iter()
        .map(|resource| SharedResourceTrace {
            resource_id: resource.id.clone(),
            inventory_after_step: Vec::with_capacity(config.horizon_steps),
            replenishment_added: Vec::with_capacity(config.horizon_steps),
            consumed: Vec::with_capacity(config.horizon_steps),
        })
        .collect();

    let mut first_floor_violation_step = None;
    let mut last_floor_violation_step = None;
    let mut floor_ok_by_step = Vec::with_capacity(config.horizon_steps);

    for step in 0..config.horizon_steps {
        let previous_inventory = inventory.clone();
        let mut actual_replenishment = vec![0.0; resources.len()];
        for (ri, resource) in resources.iter().enumerate() {
            let mut supply_factor = 1.0_f64;
            for disruption in disruptions {
                if disruption.resource_id == resource.id {
                    supply_factor *= disruption.factor_at(step);
                }
            }
            let before = inventory[ri];
            inventory[ri] = (before + resource.replenishment_per_step * supply_factor)
                .min(resource.storage_capacity);
            actual_replenishment[ri] = inventory[ri] - before;
        }

        let mut fulfillment = vec![0.0_f64; services.len()];
        let mut delivered = vec![0.0_f64; services.len()];
        let mut consumed = vec![0.0_f64; resources.len()];

        for &si in &order {
            let service = &services[si];
            let dependency_fraction = service
                .depends_on
                .iter()
                .map(|id| fulfillment[service_index[id]])
                .fold(1.0_f64, f64::min);
            let wanted = service.demand.min(service.capacity * dependency_fraction);

            let mut resource_limited_output = f64::INFINITY;
            for requirement in &service.resource_requirements {
                let ri = resource_index[&requirement.resource_id];
                resource_limited_output = resource_limited_output
                    .min(inventory[ri] / requirement.units_per_service_unit);
            }
            let output = wanted.min(resource_limited_output);
            delivered[si] = output.max(0.0);
            fulfillment[si] = (delivered[si] / service.demand).clamp(0.0, 1.0);

            for requirement in &service.resource_requirements {
                let ri = resource_index[&requirement.resource_id];
                let amount = delivered[si] * requirement.units_per_service_unit;
                inventory[ri] = (inventory[ri] - amount).max(0.0);
                consumed[ri] += amount;
            }
        }

        let mut all_at_floor = true;
        for si in 0..services.len() {
            let fraction = fulfillment[si];
            all_at_floor &= fraction + EPSILON >= config.service_floor;
            service_traces[si].delivered_service.push(delivered[si]);
            service_traces[si].fulfillment_fraction.push(fraction);
            service_traces[si]
                .unmet_demand
                .push((services[si].demand - delivered[si]).max(0.0));
            service_traces[si].minimum_fulfillment_fraction = service_traces[si]
                .minimum_fulfillment_fraction
                .min(fraction);
        }
        floor_ok_by_step.push(all_at_floor);
        if !all_at_floor {
            first_floor_violation_step.get_or_insert(step);
            last_floor_violation_step = Some(step);
        }

        for ri in 0..resources.len() {
            resource_traces[ri].inventory_after_step.push(inventory[ri]);
            resource_traces[ri].replenishment_added.push(actual_replenishment[ri]);
            resource_traces[ri].consumed.push(consumed[ri]);
            debug_assert!(inventory[ri] <= resources[ri].storage_capacity + EPSILON);
            debug_assert!(inventory[ri] >= 0.0);
            debug_assert!(inventory[ri] <= previous_inventory[ri]
                + actual_replenishment[ri] + EPSILON);
        }
    }

    let stable_recovery_step = if let Some(last_violation) = last_floor_violation_step {
        (last_violation + 1..config.horizon_steps)
            .find(|candidate| floor_ok_by_step[*candidate..].iter().all(|ok| *ok))
    } else {
        Some(0)
    };
    let recovery_delay_steps = match (first_floor_violation_step, stable_recovery_step) {
        (Some(first), Some(recovered)) => Some(recovered.saturating_sub(first)),
        (None, Some(_)) => Some(0),
        _ => None,
    };

    Ok(ResourceScreeningReport {
        horizon_steps: config.horizon_steps,
        service_floor: config.service_floor,
        first_floor_violation_step,
        last_floor_violation_step,
        stable_recovery_step,
        recovery_delay_steps,
        services: service_traces,
        resources: resource_traces,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(steps: usize, floor: f64) -> ResourceScreeningConfig {
        ResourceScreeningConfig::new(steps, floor)
    }

    fn one_resource(stock: f64, cap: f64, supply: f64) -> Vec<SharedResource> {
        vec![SharedResource::new("water", stock, cap, supply)]
    }

    fn one_service(demand: f64, capacity: f64, use_rate: f64) -> Vec<ResourceService> {
        vec![ResourceService::new(
            "treatment",
            demand,
            capacity,
            0,
            Vec::<String>::new(),
            [ResourceRequirement::new("water", use_rate)],
        )]
    }

    #[test]
    fn enough_resource_preserves_nominal_service() {
        let report = screen_resource_constrained_services(
            &one_service(1.0, 1.0, 1.0),
            &one_resource(3.0, 3.0, 0.0),
            &[],
            &config(3, 1.0),
        )
        .unwrap();
        assert!(report.services[0].fulfillment_fraction.iter().all(|v| *v == 1.0));
        assert_eq!(report.services[0].unmet_demand, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn finite_stock_exhaustion_degrades_service() {
        let report = screen_resource_constrained_services(
            &one_service(1.0, 1.0, 1.0),
            &one_resource(1.0, 1.0, 0.0),
            &[],
            &config(3, 0.9),
        )
        .unwrap();
        assert_eq!(report.services[0].fulfillment_fraction, vec![1.0, 0.0, 0.0]);
        assert_eq!(report.first_floor_violation_step, Some(1));
        assert_eq!(report.resources[0].inventory_after_step, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn finite_replenishment_supports_only_declared_rate() {
        let report = screen_resource_constrained_services(
            &one_service(1.0, 1.0, 1.0),
            &one_resource(0.0, 5.0, 0.5),
            &[],
            &config(4, 0.5),
        )
        .unwrap();
        assert_eq!(report.services[0].fulfillment_fraction, vec![0.5, 0.5, 0.5, 0.5]);
        assert_eq!(report.resources[0].inventory_after_step, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn storage_capacity_limits_replenishment() {
        let report = screen_resource_constrained_services(
            &one_service(1.0, 1.0, 1.0),
            &one_resource(0.0, 1.0, 3.0),
            &[],
            &config(2, 1.0),
        )
        .unwrap();
        assert_eq!(report.resources[0].replenishment_added, vec![1.0, 1.0]);
        assert_eq!(report.services[0].fulfillment_fraction, vec![1.0, 1.0]);
    }

    #[test]
    fn supply_disruption_propagates_after_initial_inventory_is_exhausted() {
        let disruption = ResourceSupplyDisruption::new("water", 1, 2, 0.0);
        let report = screen_resource_constrained_services(
            &one_service(1.0, 1.0, 1.0),
            &one_resource(0.0, 3.0, 1.0),
            &[disruption],
            &config(5, 0.9),
        )
        .unwrap();
        assert_eq!(report.services[0].fulfillment_fraction, vec![1.0, 0.0, 0.5, 1.0, 1.0]);
        assert_eq!(report.first_floor_violation_step, Some(1));
        assert_eq!(report.stable_recovery_step, Some(3));
    }

    #[test]
    fn shared_resource_is_allocated_by_declared_priority() {
        let services = vec![
            ResourceService::new("clinic", 1.0, 1.0, 0, Vec::<String>::new(),
                [ResourceRequirement::new("power", 1.0)]),
            ResourceService::new("workshop", 1.0, 1.0, 1, Vec::<String>::new(),
                [ResourceRequirement::new("power", 1.0)]),
        ];
        let resources = [SharedResource::new("power", 1.0, 1.0, 0.0)];
        let report = screen_resource_constrained_services(
            &services, &resources, &[], &config(1, 0.0),
        ).unwrap();
        let clinic = report.services.iter().find(|s| s.service_id == "clinic").unwrap();
        let workshop = report.services.iter().find(|s| s.service_id == "workshop").unwrap();
        assert_eq!(clinic.fulfillment_fraction, vec![1.0]);
        assert_eq!(workshop.fulfillment_fraction, vec![0.0]);
    }

    #[test]
    fn equal_priority_allocation_is_stable_by_identifier_not_input_order() {
        let a = ResourceService::new("alpha", 1.0, 1.0, 3, Vec::<String>::new(),
            [ResourceRequirement::new("power", 1.0)]);
        let z = ResourceService::new("zeta", 1.0, 1.0, 3, Vec::<String>::new(),
            [ResourceRequirement::new("power", 1.0)]);
        let resources = [SharedResource::new("power", 1.0, 1.0, 0.0)];
        let first = screen_resource_constrained_services(&[z.clone(), a.clone()], &resources, &[], &config(1, 0.0)).unwrap();
        let second = screen_resource_constrained_services(&[a, z], &resources, &[], &config(1, 0.0)).unwrap();
        let alpha_first = first.services.iter().find(|s| s.service_id == "alpha").unwrap();
        let alpha_second = second.services.iter().find(|s| s.service_id == "alpha").unwrap();
        let zeta_first = first.services.iter().find(|s| s.service_id == "zeta").unwrap();
        let zeta_second = second.services.iter().find(|s| s.service_id == "zeta").unwrap();
        assert_eq!(alpha_first.fulfillment_fraction, alpha_second.fulfillment_fraction);
        assert_eq!(zeta_first.fulfillment_fraction, zeta_second.fulfillment_fraction);
        assert_eq!(alpha_first.fulfillment_fraction, vec![1.0]);
    }

    #[test]
    fn upstream_service_and_resource_bottlenecks_compose() {
        let services = vec![
            ResourceService::new("power", 1.0, 1.0, 0, Vec::<String>::new(),
                [ResourceRequirement::new("fuel", 1.0)]),
            ResourceService::new("water", 1.0, 1.0, 1, ["power"],
                [ResourceRequirement::new("chemical", 1.0)]),
        ];
        let resources = [
            SharedResource::new("fuel", 0.0, 0.0, 0.0),
            SharedResource::new("chemical", 2.0, 2.0, 0.0),
        ];
        let report = screen_resource_constrained_services(
            &services, &resources, &[], &config(2, 0.9),
        ).unwrap();
        let water = report.services.iter().find(|s| s.service_id == "water").unwrap();
        assert_eq!(water.fulfillment_fraction, vec![0.0, 0.0]);
    }

    #[test]
    fn invalid_and_unknown_resource_requirements_fail_closed() {
        let unknown = vec![ResourceService::new("pump", 1.0, 1.0, 0, Vec::<String>::new(),
            [ResourceRequirement::new("unknown", 1.0)])];
        assert!(matches!(
            screen_resource_constrained_services(&unknown, &[], &[], &config(1, 0.0)),
            Err(ResourceScreeningError::UnknownResourceRequirement { .. })
        ));

        let invalid_rate = vec![ResourceService::new("pump", 1.0, 1.0, 0, Vec::<String>::new(),
            [ResourceRequirement::new("water", f64::NAN)])];
        assert!(matches!(
            screen_resource_constrained_services(&invalid_rate, &one_resource(1.0, 1.0, 0.0), &[], &config(1, 0.0)),
            Err(ResourceScreeningError::InvalidResourceRate { .. })
        ));
    }

    #[test]
    fn resource_stock_and_supply_inputs_fail_closed() {
        assert!(matches!(
            screen_resource_constrained_services(&one_service(1.0, 1.0, 1.0), &one_resource(-1.0, 1.0, 0.0), &[], &config(1, 0.0)),
            Err(ResourceScreeningError::InvalidInitialStock(_))
        ));
        assert!(matches!(
            screen_resource_constrained_services(&one_service(1.0, 1.0, 1.0), &one_resource(2.0, 1.0, 0.0), &[], &config(1, 0.0)),
            Err(ResourceScreeningError::InvalidStorageCapacity(_))
        ));
    }

    #[test]
    fn dependency_cycle_fails_closed() {
        let services = vec![
            ResourceService::new("a", 1.0, 1.0, 0, ["b"], Vec::<ResourceRequirement>::new()),
            ResourceService::new("b", 1.0, 1.0, 0, ["a"], Vec::<ResourceRequirement>::new()),
        ];
        assert!(matches!(
            screen_resource_constrained_services(&services, &[], &[], &config(1, 0.0)),
            Err(ResourceScreeningError::DependencyCycle(_))
        ));
    }

    #[test]
    fn deterministic_for_identical_resource_scenarios() {
        let services = one_service(1.0, 1.0, 1.0);
        let resources = one_resource(2.0, 3.0, 0.5);
        let disruptions = [ResourceSupplyDisruption::new("water", 1, 3, 0.0)];
        let a = screen_resource_constrained_services(&services, &resources, &disruptions, &config(5, 0.8)).unwrap();
        let b = screen_resource_constrained_services(&services, &resources, &disruptions, &config(5, 0.8)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn resource_balance_conserves_stock_replenishment_and_consumption() {
        let services = [
            ResourceService::new("pump", 1.0, 1.0, 0, Vec::<String>::new(),
                [ResourceRequirement::new("water", 2.0)]),
        ];
        let resources = [SharedResource::new("water", 2.0, 4.0, 1.0)];
        let report =
            screen_resource_constrained_services(&services, &resources, &[], &config(4, 0.0))
                .unwrap();
        let trace = &report.resources[0];
        let mut previous = resources[0].initial_stock;
        for step in 0..report.horizon_steps {
            let expected = previous + trace.replenishment_added[step] - trace.consumed[step];
            assert!((trace.inventory_after_step[step] - expected).abs() < 1.0e-9);
            assert!(trace.inventory_after_step[step] >= 0.0);
            assert!(trace.inventory_after_step[step] <= resources[0].storage_capacity);
            previous = trace.inventory_after_step[step];
        }
    }

    #[test]
    fn service_is_limited_by_the_scarcest_of_multiple_resources() {
        let services = [ResourceService::new(
            "manufacture",
            1.0,
            1.0,
            0,
            Vec::<String>::new(),
            [
                ResourceRequirement::new("water", 1.0),
                ResourceRequirement::new("power", 0.5),
            ],
        )];
        let resources = [
            SharedResource::new("water", 10.0, 10.0, 0.0),
            SharedResource::new("power", 0.25, 0.25, 0.0),
        ];
        let report =
            screen_resource_constrained_services(&services, &resources, &[], &config(1, 0.0))
                .unwrap();
        assert_eq!(report.services[0].delivered_service, vec![0.5]);
        let power = report.resources.iter().find(|r| r.resource_id == "power").unwrap();
        assert_eq!(power.consumed, vec![0.25]);
    }

    #[test]
    fn unknown_and_invalid_supply_disruptions_fail_closed() {
        let unknown = ResourceSupplyDisruption::new("unknown", 0, 1, 0.0);
        assert!(matches!(
            screen_resource_constrained_services(
                &one_service(1.0, 1.0, 1.0),
                &one_resource(1.0, 1.0, 0.0),
                &[unknown],
                &config(2, 0.0),
            ),
            Err(ResourceScreeningError::UnknownDisruptionResource(_))
        ));

        let bad_factor = ResourceSupplyDisruption::new("water", 0, 1, f64::NAN);
        assert!(matches!(
            screen_resource_constrained_services(
                &one_service(1.0, 1.0, 1.0),
                &one_resource(1.0, 1.0, 0.0),
                &[bad_factor],
                &config(2, 0.0),
            ),
            Err(ResourceScreeningError::InvalidDisruptionFactor(_))
        ));
    }

    #[test]
    fn duplicate_service_and_resource_ids_fail_closed() {
        let duplicate_services = [
            ResourceService::new("x", 1.0, 1.0, 0, Vec::<String>::new(), Vec::<ResourceRequirement>::new()),
            ResourceService::new("x", 1.0, 1.0, 1, Vec::<String>::new(), Vec::<ResourceRequirement>::new()),
        ];
        assert!(matches!(
            screen_resource_constrained_services(&duplicate_services, &[], &[], &config(1, 0.0)),
            Err(ResourceScreeningError::DuplicateService(_))
        ));

        let duplicate_resources = [
            SharedResource::new("water", 0.0, 1.0, 0.0),
            SharedResource::new("water", 0.0, 1.0, 0.0),
        ];
        assert!(matches!(
            screen_resource_constrained_services(
                &one_service(1.0, 1.0, 1.0),
                &duplicate_resources,
                &[],
                &config(1, 0.0),
            ),
            Err(ResourceScreeningError::DuplicateResource(_))
        ));
    }

    #[test]
    fn invalid_horizon_floor_and_replenishment_fail_closed() {
        assert!(matches!(
            screen_resource_constrained_services(
                &one_service(1.0, 1.0, 1.0),
                &one_resource(1.0, 1.0, 0.0),
                &[],
                &config(0, 0.8),
            ),
            Err(ResourceScreeningError::ZeroHorizon)
        ));
        assert!(matches!(
            screen_resource_constrained_services(
                &one_service(1.0, 1.0, 1.0),
                &one_resource(1.0, 1.0, 0.0),
                &[],
                &config(1, f64::NAN),
            ),
            Err(ResourceScreeningError::InvalidServiceFloor)
        ));
        assert!(matches!(
            screen_resource_constrained_services(
                &one_service(1.0, 1.0, 1.0),
                &one_resource(1.0, 1.0, -0.1),
                &[],
                &config(1, 0.0),
            ),
            Err(ResourceScreeningError::InvalidReplenishment(_))
        ));
    }

    #[test]
    fn unknown_and_repeated_service_dependencies_fail_closed() {
        let unknown = [ResourceService::new(
            "water",
            1.0,
            1.0,
            0,
            ["not-a-service"],
            Vec::<ResourceRequirement>::new(),
        )];
        assert!(matches!(
            screen_resource_constrained_services(&unknown, &[], &[], &config(1, 0.0)),
            Err(ResourceScreeningError::UnknownDependency { .. })
        ));

        let repeated = [
            ResourceService::new(
                "water",
                1.0,
                1.0,
                0,
                ["power", "power"],
                Vec::<ResourceRequirement>::new(),
            ),
            ResourceService::new(
                "power",
                1.0,
                1.0,
                1,
                Vec::<String>::new(),
                Vec::<ResourceRequirement>::new(),
            ),
        ];
        assert!(matches!(
            screen_resource_constrained_services(&repeated, &[], &[], &config(1, 0.0)),
            Err(ResourceScreeningError::DuplicateDependency { .. })
        ));
    }
}
