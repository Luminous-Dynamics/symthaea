// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic service-continuity screening for resilient-system feasibility studies.
//!
//! This is an intentionally small, auditable abstraction, not a physical process
//! simulator. Each service has a demand and nominal supply capacity. Dependencies
//! transmit the weakest upstream fulfillment fraction, and scheduled disruptions
//! reduce capacity before linear recovery. Outputs are service-unit-steps, not
//! automatically kWh, litres, kilograms, lives, or money. Use a service-specific
//! unit and time step before interpreting the score.
//!
//! The module exists to make scenario assumptions explicit and to test the
//! direction of effects (redundancy helps; common-mode loss hurts; slower repair
//! cannot improve recovery). It does not estimate real-world failure probabilities.

use std::collections::{HashMap, HashSet};

const EPSILON: f64 = 1.0e-12;

#[derive(Clone, Debug, PartialEq)]
pub struct EssentialService {
    /// Stable identifier, unique within a model.
    pub id: String,
    /// Required service demand per time step, in a caller-defined unit.
    pub demand: f64,
    /// Nominal available capacity per time step, in the same unit as demand.
    pub capacity: f64,
    /// Services that this service depends on. The weakest dependency is the bottleneck.
    pub depends_on: Vec<String>,
}

impl EssentialService {
    pub fn new(
        id: impl Into<String>,
        demand: f64,
        capacity: f64,
        depends_on: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            id: id.into(),
            demand,
            capacity,
            depends_on: depends_on.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CapacityDisruption {
    /// Service whose capacity is affected.
    pub service_id: String,
    /// First discrete time step affected.
    pub start_step: usize,
    /// Number of steps from the initial capacity factor to full nominal capacity.
    pub recovery_steps: usize,
    /// Capacity multiplier at start_step; 0.0 means total loss, 1.0 means no loss.
    pub capacity_factor_at_start: f64,
}

impl CapacityDisruption {
    pub fn new(
        service_id: impl Into<String>,
        start_step: usize,
        recovery_steps: usize,
        capacity_factor_at_start: f64,
    ) -> Self {
        Self {
            service_id: service_id.into(),
            start_step,
            recovery_steps,
            capacity_factor_at_start,
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
        self.capacity_factor_at_start
            + (1.0 - self.capacity_factor_at_start) * elapsed as f64
                / self.recovery_steps as f64
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScreeningConfig {
    /// Number of discrete time steps to simulate; step 0 is included.
    pub horizon_steps: usize,
    /// Minimum acceptable service fulfillment fraction, in [0, 1].
    pub service_floor: f64,
}

impl ScreeningConfig {
    pub fn new(horizon_steps: usize, service_floor: f64) -> Self {
        Self {
            horizon_steps,
            service_floor,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServiceSeries {
    pub service_id: String,
    /// Effective capacity after dependency and disruption effects.
    pub effective_capacity: Vec<f64>,
    /// Fulfillment fraction, capped at 1.0.
    pub fulfillment_fraction: Vec<f64>,
    /// Unmet demand per step, never negative.
    pub unmet_demand: Vec<f64>,
    pub minimum_fulfillment_fraction: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScreeningReport {
    pub horizon_steps: usize,
    pub service_floor: f64,
    /// Sum of unmet demand across services and time. Units are service-unit-steps.
    pub total_unmet_service_unit_steps: f64,
    pub first_floor_violation_step: Option<usize>,
    pub last_floor_violation_step: Option<usize>,
    /// Earliest step after which every service remains above the floor for the rest
    /// of the modeled horizon. None means the floor was not stably recovered.
    pub stable_recovery_step: Option<usize>,
    /// stable_recovery_step - first_floor_violation_step, if both exist.
    pub recovery_delay_steps: Option<usize>,
    pub all_services_at_floor_at_horizon: bool,
    pub services: Vec<ServiceSeries>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScreeningError {
    EmptyServiceSet,
    ZeroHorizon,
    InvalidServiceFloor,
    EmptyServiceId,
    DuplicateServiceId(String),
    InvalidDemand(String),
    InvalidCapacity(String),
    UnknownDependency { service_id: String, dependency_id: String },
    DuplicateDependency { service_id: String, dependency_id: String },
    DependencyCycle(String),
    UnknownDisruptionService(String),
    DisruptionStartsOutsideHorizon(String),
    ZeroRecoverySteps(String),
    InvalidDisruptionFactor(String),
}

impl std::fmt::Display for ScreeningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyServiceSet => write!(f, "at least one essential service is required"),
            Self::ZeroHorizon => write!(f, "horizon_steps must be greater than zero"),
            Self::InvalidServiceFloor => write!(f, "service_floor must be finite and in [0, 1]"),
            Self::EmptyServiceId => write!(f, "service id must not be empty"),
            Self::DuplicateServiceId(id) => write!(f, "duplicate service id: {id}"),
            Self::InvalidDemand(id) => write!(f, "demand must be finite and positive for {id}"),
            Self::InvalidCapacity(id) => {
                write!(f, "capacity must be finite and non-negative for {id}")
            }
            Self::UnknownDependency { service_id, dependency_id } => {
                write!(f, "service {service_id} depends on unknown service {dependency_id}")
            }
            Self::DuplicateDependency { service_id, dependency_id } => {
                write!(f, "service {service_id} repeats dependency {dependency_id}")
            }
            Self::DependencyCycle(id) => write!(f, "dependency cycle includes service {id}"),
            Self::UnknownDisruptionService(id) => {
                write!(f, "disruption references unknown service {id}")
            }
            Self::DisruptionStartsOutsideHorizon(id) => {
                write!(f, "disruption for {id} starts outside the simulation horizon")
            }
            Self::ZeroRecoverySteps(id) => {
                write!(f, "disruption for {id} must have recovery_steps > 0")
            }
            Self::InvalidDisruptionFactor(id) => write!(
                f,
                "disruption capacity factor for {id} must be finite and in [0, 1]"
            ),
        }
    }
}

impl std::error::Error for ScreeningError {}

/// Validate inputs and compute a stable topological order. The order ensures that
/// every dependency is evaluated before its dependent service, independent of the
/// order in which services were supplied.
fn validate_and_order(
    services: &[EssentialService],
) -> Result<(HashMap<String, usize>, Vec<usize>), ScreeningError> {
    if services.is_empty() {
        return Err(ScreeningError::EmptyServiceSet);
    }

    let mut index_by_id = HashMap::with_capacity(services.len());
    for (index, service) in services.iter().enumerate() {
        if service.id.trim().is_empty() {
            return Err(ScreeningError::EmptyServiceId);
        }
        if index_by_id.insert(service.id.clone(), index).is_some() {
            return Err(ScreeningError::DuplicateServiceId(service.id.clone()));
        }
        if !service.demand.is_finite() || service.demand <= 0.0 {
            return Err(ScreeningError::InvalidDemand(service.id.clone()));
        }
        if !service.capacity.is_finite() || service.capacity < 0.0 {
            return Err(ScreeningError::InvalidCapacity(service.id.clone()));
        }
    }

    for service in services {
        let mut seen = HashSet::new();
        for dependency_id in &service.depends_on {
            if !index_by_id.contains_key(dependency_id) {
                return Err(ScreeningError::UnknownDependency {
                    service_id: service.id.clone(),
                    dependency_id: dependency_id.clone(),
                });
            }
            if !seen.insert(dependency_id) {
                return Err(ScreeningError::DuplicateDependency {
                    service_id: service.id.clone(),
                    dependency_id: dependency_id.clone(),
                });
            }
        }
    }

    fn visit(
        index: usize,
        services: &[EssentialService],
        index_by_id: &HashMap<String, usize>,
        state: &mut [u8],
        order: &mut Vec<usize>,
    ) -> Result<(), ScreeningError> {
        match state[index] {
            2 => return Ok(()),
            1 => return Err(ScreeningError::DependencyCycle(services[index].id.clone())),
            _ => {}
        }
        state[index] = 1;
        for dependency_id in &services[index].depends_on {
            let dependency_index = index_by_id[dependency_id];
            visit(dependency_index, services, index_by_id, state, order)?;
        }
        state[index] = 2;
        order.push(index);
        Ok(())
    }

    let mut state = vec![0; services.len()];
    let mut order = Vec::with_capacity(services.len());
    for index in 0..services.len() {
        visit(index, services, &index_by_id, &mut state, &mut order)?;
    }
    Ok((index_by_id, order))
}

/// Run one deterministic service-continuity scenario.
///
/// Model contract:
/// - dependencies are a directed acyclic graph;
/// - a dependent service's capacity is multiplied by the weakest dependency's
///   fulfillment fraction (a conservative, dimensionless bottleneck abstraction);
/// - simultaneous disruptions multiply their capacity factors;
/// - each disruption recovers linearly to nominal capacity over recovery_steps;
/// - service fulfillment is capped at 100%, while surplus nominal capacity remains
///   available as margin;
/// - total unmet service is summed over all services and modeled steps.
pub fn screen_service_continuity(
    services: &[EssentialService],
    disruptions: &[CapacityDisruption],
    config: &ScreeningConfig,
) -> Result<ScreeningReport, ScreeningError> {
    if config.horizon_steps == 0 {
        return Err(ScreeningError::ZeroHorizon);
    }
    if !config.service_floor.is_finite()
        || !(0.0..=1.0).contains(&config.service_floor)
    {
        return Err(ScreeningError::InvalidServiceFloor);
    }

    let (index_by_id, order) = validate_and_order(services)?;
    for disruption in disruptions {
        if !index_by_id.contains_key(&disruption.service_id) {
            return Err(ScreeningError::UnknownDisruptionService(
                disruption.service_id.clone(),
            ));
        }
        if disruption.start_step >= config.horizon_steps {
            return Err(ScreeningError::DisruptionStartsOutsideHorizon(
                disruption.service_id.clone(),
            ));
        }
        if disruption.recovery_steps == 0 {
            return Err(ScreeningError::ZeroRecoverySteps(
                disruption.service_id.clone(),
            ));
        }
        if !disruption.capacity_factor_at_start.is_finite()
            || !(0.0..=1.0).contains(&disruption.capacity_factor_at_start)
        {
            return Err(ScreeningError::InvalidDisruptionFactor(
                disruption.service_id.clone(),
            ));
        }
    }

    let mut series: Vec<ServiceSeries> = services
        .iter()
        .map(|service| ServiceSeries {
            service_id: service.id.clone(),
            effective_capacity: Vec::with_capacity(config.horizon_steps),
            fulfillment_fraction: Vec::with_capacity(config.horizon_steps),
            unmet_demand: Vec::with_capacity(config.horizon_steps),
            minimum_fulfillment_fraction: 1.0,
        })
        .collect();

    let mut total_unmet_service_unit_steps = 0.0;
    let mut first_floor_violation_step = None;
    let mut last_floor_violation_step = None;
    let mut every_service_at_floor = Vec::with_capacity(config.horizon_steps);

    for step in 0..config.horizon_steps {
        let mut incident_factor = vec![1.0_f64; services.len()];
        for disruption in disruptions {
            let index = index_by_id[&disruption.service_id];
            incident_factor[index] *= disruption.factor_at(step);
        }

        // The topological order makes each dependency's current-step fulfillment known.
        let mut effective_capacity = vec![0.0_f64; services.len()];
        let mut fulfillment_fraction = vec![0.0_f64; services.len()];
        let mut unmet_demand = vec![0.0_f64; services.len()];

        for &index in &order {
            let service = &services[index];
            let dependency_fraction = service
                .depends_on
                .iter()
                .map(|id| fulfillment_fraction[index_by_id[id]])
                .fold(1.0_f64, f64::min);

            let capacity = service.capacity * incident_factor[index] * dependency_fraction;
            let fulfillment = (capacity / service.demand).clamp(0.0, 1.0);
            let unmet = (service.demand - capacity).max(0.0);

            effective_capacity[index] = capacity;
            fulfillment_fraction[index] = fulfillment;
            unmet_demand[index] = unmet;
        }

        let all_at_floor = fulfillment_fraction
            .iter()
            .all(|fraction| *fraction + EPSILON >= config.service_floor);
        every_service_at_floor.push(all_at_floor);

        if !all_at_floor {
            first_floor_violation_step.get_or_insert(step);
            last_floor_violation_step = Some(step);
        }

        for index in 0..services.len() {
            series[index].effective_capacity.push(effective_capacity[index]);
            series[index].fulfillment_fraction.push(fulfillment_fraction[index]);
            series[index].unmet_demand.push(unmet_demand[index]);
            series[index].minimum_fulfillment_fraction = series[index]
                .minimum_fulfillment_fraction
                .min(fulfillment_fraction[index]);
            total_unmet_service_unit_steps += unmet_demand[index];
        }
    }

    let stable_recovery_step = if let Some(last_violation) = last_floor_violation_step {
        (last_violation + 1..config.horizon_steps).find(|candidate| {
            every_service_at_floor[*candidate..].iter().all(|at_floor| *at_floor)
        })
    } else {
        // Already above floor for the entire horizon.
        Some(0)
    };

    let recovery_delay_steps = match (first_floor_violation_step, stable_recovery_step) {
        (Some(first), Some(recovered)) => Some(recovered.saturating_sub(first)),
        (None, Some(_)) => Some(0),
        _ => None,
    };

    Ok(ScreeningReport {
        horizon_steps: config.horizon_steps,
        service_floor: config.service_floor,
        total_unmet_service_unit_steps,
        first_floor_violation_step,
        last_floor_violation_step,
        stable_recovery_step,
        recovery_delay_steps,
        all_services_at_floor_at_horizon: every_service_at_floor.last().copied().unwrap_or(false),
        services: series,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(horizon_steps: usize, floor: f64) -> ScreeningConfig {
        ScreeningConfig::new(horizon_steps, floor)
    }

    fn single_service(capacity: f64) -> Vec<EssentialService> {
        vec![EssentialService::new("power", 1.0, capacity, Vec::<String>::new())]
    }

    #[test]
    fn no_disruption_has_no_deficit() {
        let report = screen_service_continuity(&single_service(1.0), &[], &config(8, 1.0))
            .expect("valid scenario");
        assert_eq!(report.total_unmet_service_unit_steps, 0.0);
        assert_eq!(report.first_floor_violation_step, None);
        assert_eq!(report.stable_recovery_step, Some(0));
        assert!(report.all_services_at_floor_at_horizon);
    }

    #[test]
    fn redundant_capacity_survives_single_source_loss() {
        // Capacity 2 against demand 1 represents two equivalent unit-capacity sources.
        let disruption = CapacityDisruption::new("power", 2, 3, 0.5);
        let report = screen_service_continuity(
            &single_service(2.0),
            &[disruption],
            &config(8, 1.0),
        )
        .expect("valid N-1 scenario");
        assert_eq!(report.total_unmet_service_unit_steps, 0.0);
        assert_eq!(report.first_floor_violation_step, None);
    }

    #[test]
    fn common_mode_loss_causes_a_service_deficit() {
        let disruption = CapacityDisruption::new("power", 2, 3, 0.0);
        let report = screen_service_continuity(
            &single_service(1.0),
            &[disruption],
            &config(8, 1.0),
        )
        .expect("valid common-mode scenario");
        assert!(report.total_unmet_service_unit_steps > 0.0);
        assert_eq!(report.first_floor_violation_step, Some(2));
        assert!(report.all_services_at_floor_at_horizon);
    }

    #[test]
    fn slow_recovery_carries_more_unmet_service_than_fast_recovery() {
        let services = single_service(1.0);
        let fast = CapacityDisruption::new("power", 1, 2, 0.0);
        let slow = CapacityDisruption::new("power", 1, 6, 0.0);
        let fast_report =
            screen_service_continuity(&services, &[fast], &config(10, 0.8)).unwrap();
        let slow_report =
            screen_service_continuity(&services, &[slow], &config(10, 0.8)).unwrap();
        assert!(
            slow_report.total_unmet_service_unit_steps > fast_report.total_unmet_service_unit_steps
        );
        assert!(
            slow_report.recovery_delay_steps.unwrap() > fast_report.recovery_delay_steps.unwrap()
        );
    }

    #[test]
    fn upstream_power_loss_propagates_to_dependent_water_service() {
        let services = vec![
            EssentialService::new("power", 1.0, 1.0, Vec::<String>::new()),
            EssentialService::new("water", 1.0, 1.0, ["power"]),
        ];
        let loss = CapacityDisruption::new("power", 1, 3, 0.0);
        let report = screen_service_continuity(&services, &[loss], &config(6, 0.9)).unwrap();
        let water = report.services.iter().find(|s| s.service_id == "water").unwrap();
        assert_eq!(water.fulfillment_fraction[1], 0.0);
        assert!(report.total_unmet_service_unit_steps >= 2.0);
    }

    #[test]
    fn service_order_does_not_change_dependency_result() {
        let services_a = vec![
            EssentialService::new("power", 1.0, 0.5, Vec::<String>::new()),
            EssentialService::new("water", 1.0, 1.0, ["power"]),
        ];
        let services_b = vec![
            EssentialService::new("water", 1.0, 1.0, ["power"]),
            EssentialService::new("power", 1.0, 0.5, Vec::<String>::new()),
        ];
        let a = screen_service_continuity(&services_a, &[], &config(2, 0.0)).unwrap();
        let b = screen_service_continuity(&services_b, &[], &config(2, 0.0)).unwrap();
        let water_a = a.services.iter().find(|s| s.service_id == "water").unwrap();
        let water_b = b.services.iter().find(|s| s.service_id == "water").unwrap();
        assert_eq!(water_a.fulfillment_fraction, water_b.fulfillment_fraction);
        assert_eq!(water_a.fulfillment_fraction[0], 0.5);
    }

    #[test]
    fn dependency_cycle_fails_closed() {
        let services = vec![
            EssentialService::new("power", 1.0, 1.0, ["water"]),
            EssentialService::new("water", 1.0, 1.0, ["power"]),
        ];
        let error = screen_service_continuity(&services, &[], &config(3, 0.8)).unwrap_err();
        assert!(matches!(error, ScreeningError::DependencyCycle(_)));
    }

    #[test]
    fn unknown_dependency_fails_closed() {
        let services = vec![EssentialService::new("water", 1.0, 1.0, ["electricity"])];
        let error = screen_service_continuity(&services, &[], &config(3, 0.8)).unwrap_err();
        assert!(matches!(error, ScreeningError::UnknownDependency { .. }));
    }

    #[test]
    fn invalid_numeric_inputs_fail_closed() {
        let bad_demand = vec![EssentialService::new("water", 0.0, 1.0, Vec::<String>::new())];
        assert!(matches!(
            screen_service_continuity(&bad_demand, &[], &config(3, 0.8)),
            Err(ScreeningError::InvalidDemand(_))
        ));
        assert!(matches!(
            screen_service_continuity(&single_service(1.0), &[], &config(3, f64::NAN)),
            Err(ScreeningError::InvalidServiceFloor)
        ));
    }

    #[test]
    fn disruption_input_must_be_valid() {
        let bad = CapacityDisruption::new("power", 1, 0, 0.0);
        assert!(matches!(
            screen_service_continuity(&single_service(1.0), &[bad], &config(3, 0.8)),
            Err(ScreeningError::ZeroRecoverySteps(_))
        ));
    }

    #[test]
    fn same_inputs_produce_identical_reports() {
        let services = vec![
            EssentialService::new("power", 1.0, 1.0, Vec::<String>::new()),
            EssentialService::new("water", 1.0, 1.0, ["power"]),
        ];
        let disruptions = [CapacityDisruption::new("power", 1, 4, 0.0)];
        let first = screen_service_continuity(&services, &disruptions, &config(7, 0.8)).unwrap();
        let second = screen_service_continuity(&services, &disruptions, &config(7, 0.8)).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn severe_capacity_loss_cannot_improve_service() {
        let mild = CapacityDisruption::new("power", 1, 4, 0.5);
        let severe = CapacityDisruption::new("power", 1, 4, 0.1);
        let services = single_service(1.0);
        let mild_report = screen_service_continuity(&services, &[mild], &config(8, 0.8)).unwrap();
        let severe_report =
            screen_service_continuity(&services, &[severe], &config(8, 0.8)).unwrap();
        assert!(
            severe_report.total_unmet_service_unit_steps
                >= mild_report.total_unmet_service_unit_steps
        );
    }

    #[test]
    fn extra_capacity_cannot_increase_unmet_service() {
        let disruption = CapacityDisruption::new("power", 1, 3, 0.2);
        let less = screen_service_continuity(
            &single_service(1.0),
            &[disruption.clone()],
            &config(8, 0.8),
        )
        .unwrap();
        let more = screen_service_continuity(
            &single_service(2.0),
            &[disruption],
            &config(8, 0.8),
        )
        .unwrap();
        assert!(
            more.total_unmet_service_unit_steps <= less.total_unmet_service_unit_steps
        );
        assert!(
            more.services[0]
                .fulfillment_fraction
                .iter()
                .zip(&less.services[0].fulfillment_fraction)
                .all(|(a, b)| a + EPSILON >= *b)
        );
    }

    #[test]
    fn horizon_ending_before_floor_recovery_keeps_recovery_unknown() {
        let disruption = CapacityDisruption::new("power", 1, 20, 0.0);
        let report = screen_service_continuity(
            &single_service(1.0),
            &[disruption],
            &config(5, 0.9),
        )
        .unwrap();
        assert!(!report.all_services_at_floor_at_horizon);
        assert_eq!(report.stable_recovery_step, None);
        assert_eq!(report.recovery_delay_steps, None);
    }

    #[test]
    fn recovery_is_after_the_final_violation_not_a_temporary_rebound() {
        let events = [
            CapacityDisruption::new("power", 1, 1, 0.0),
            CapacityDisruption::new("power", 4, 3, 0.0),
        ];
        let report =
            screen_service_continuity(&single_service(1.0), &events, &config(9, 0.8)).unwrap();
        assert_eq!(report.first_floor_violation_step, Some(1));
        assert_eq!(report.last_floor_violation_step, Some(6));
        assert_eq!(report.stable_recovery_step, Some(7));
    }

    #[test]
    fn empty_model_and_zero_horizon_fail_closed() {
        assert!(matches!(
            screen_service_continuity(&[], &[], &config(3, 0.8)),
            Err(ScreeningError::EmptyServiceSet)
        ));
        assert!(matches!(
            screen_service_continuity(&single_service(1.0), &[], &config(0, 0.8)),
            Err(ScreeningError::ZeroHorizon)
        ));
    }

    #[test]
    fn duplicate_service_and_dependency_ids_fail_closed() {
        let duplicate_ids = vec![
            EssentialService::new("power", 1.0, 1.0, Vec::<String>::new()),
            EssentialService::new("power", 1.0, 1.0, Vec::<String>::new()),
        ];
        assert!(matches!(
            screen_service_continuity(&duplicate_ids, &[], &config(3, 0.8)),
            Err(ScreeningError::DuplicateServiceId(_))
        ));

        let repeated_dependency =
            vec![EssentialService::new("water", 1.0, 1.0, ["power", "power"]),
                 EssentialService::new("power", 1.0, 1.0, Vec::<String>::new())];
        assert!(matches!(
            screen_service_continuity(&repeated_dependency, &[], &config(3, 0.8)),
            Err(ScreeningError::DuplicateDependency { .. })
        ));
    }

    #[test]
    fn negative_capacity_and_unknown_disruption_target_fail_closed() {
        let negative_capacity = vec![EssentialService::new(
            "power",
            1.0,
            -0.1,
            Vec::<String>::new(),
        )];
        assert!(matches!(
            screen_service_continuity(&negative_capacity, &[], &config(3, 0.8)),
            Err(ScreeningError::InvalidCapacity(_))
        ));
        let unknown_target = CapacityDisruption::new("unknown", 1, 2, 0.0);
        assert!(matches!(
            screen_service_continuity(&single_service(1.0), &[unknown_target], &config(3, 0.8)),
            Err(ScreeningError::UnknownDisruptionService(_))
        ));
    }

    #[test]
    fn disruption_must_start_inside_horizon() {
        let outside = CapacityDisruption::new("power", 3, 2, 0.0);
        assert!(matches!(
            screen_service_continuity(&single_service(1.0), &[outside], &config(3, 0.8)),
            Err(ScreeningError::DisruptionStartsOutsideHorizon(_))
        ));
    }

    #[test]
    fn disruption_capacity_factor_must_be_in_unit_interval() {
        for factor in [-0.01, 1.01, f64::NAN, f64::INFINITY] {
            let invalid = CapacityDisruption::new("power", 1, 2, factor);
            assert!(matches!(
                screen_service_continuity(&single_service(1.0), &[invalid], &config(3, 0.8)),
                Err(ScreeningError::InvalidDisruptionFactor(_))
            ));
        }
    }
}
