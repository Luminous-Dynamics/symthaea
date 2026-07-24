// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fault-containment architecture and common-cause dependency analysis.
//!
//! Declaring two lanes does not establish redundancy when they share power,
//! sensors, buses, or actuators. This module propagates component failures over
//! required dependency edges, evaluates service availability through explicit
//! alternatives, and enumerates single-component faults that defeat critical
//! services. It is an architectural analysis model, not a physical FMEA.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FlightComponent {
    PowerLaneA,
    PowerLaneB,
    SensorLaneA,
    SensorLaneB,
    FlightComputerA,
    FlightComputerB,
    ActuatorLaneA,
    ActuatorLaneB,
    MainRotorSystem,
    TailRotorSystem,
    EngineControl,
    EvidenceRecorder,
    CommonPowerFeed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ContainmentZone {
    LaneA,
    LaneB,
    SharedPropulsion,
    SharedEvidence,
    CommonCause,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FlightService {
    Navigation,
    AttitudeControl,
    VerticalControl,
    YawControl,
    EngineManagement,
    EvidenceRecording,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentBinding {
    pub component: FlightComponent,
    pub zone: ContainmentZone,
}

/// Failure of `upstream` makes `downstream` unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequiredDependency {
    pub upstream: FlightComponent,
    pub downstream: FlightComponent,
}

/// A service is available when any alternative contains only available components.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceRequirement {
    pub service: FlightService,
    pub critical: bool,
    pub alternatives: Vec<Vec<FlightComponent>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultContainmentError {
    EmptyArchitecture,
    DuplicateComponent,
    UnknownDependencyComponent,
    SelfDependency,
    DuplicateService,
    EmptyServiceAlternative,
    UnknownServiceComponent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaultContainmentAssessment {
    pub initial_faults: Vec<FlightComponent>,
    pub unavailable_components: Vec<FlightComponent>,
    pub affected_zones: Vec<ContainmentZone>,
    pub lost_services: Vec<FlightService>,
    pub lost_critical_services: Vec<FlightService>,
    /// True when a fault originating in one lane causes propagated failures in
    /// another lane through declared dependencies.
    pub cross_zone_propagation: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SinglePointFailureReport {
    pub component: FlightComponent,
    pub lost_critical_services: Vec<FlightService>,
}

#[derive(Debug, Clone)]
pub struct FaultContainmentArchitecture {
    components: Vec<ComponentBinding>,
    dependencies: Vec<RequiredDependency>,
    services: Vec<ServiceRequirement>,
}

impl FaultContainmentArchitecture {
    pub fn new(
        components: Vec<ComponentBinding>,
        dependencies: Vec<RequiredDependency>,
        services: Vec<ServiceRequirement>,
    ) -> Result<Self, FaultContainmentError> {
        if components.is_empty() || services.is_empty() {
            return Err(FaultContainmentError::EmptyArchitecture);
        }
        let mut known = BTreeSet::new();
        for binding in &components {
            if !known.insert(binding.component) {
                return Err(FaultContainmentError::DuplicateComponent);
            }
        }
        for dependency in &dependencies {
            if dependency.upstream == dependency.downstream {
                return Err(FaultContainmentError::SelfDependency);
            }
            if !known.contains(&dependency.upstream) || !known.contains(&dependency.downstream) {
                return Err(FaultContainmentError::UnknownDependencyComponent);
            }
        }
        let mut service_ids = BTreeSet::new();
        for requirement in &services {
            if !service_ids.insert(requirement.service) {
                return Err(FaultContainmentError::DuplicateService);
            }
            if requirement.alternatives.is_empty()
                || requirement.alternatives.iter().any(Vec::is_empty)
            {
                return Err(FaultContainmentError::EmptyServiceAlternative);
            }
            if requirement
                .alternatives
                .iter()
                .flatten()
                .any(|component| !known.contains(component))
            {
                return Err(FaultContainmentError::UnknownServiceComponent);
            }
        }
        Ok(Self {
            components,
            dependencies,
            services,
        })
    }

    /// Nominal dual-lane architecture with shared physical propulsion systems.
    pub fn default_dual_lane() -> Self {
        use ContainmentZone::*;
        use FlightComponent::*;
        use FlightService::*;
        let components = vec![
            ComponentBinding {
                component: PowerLaneA,
                zone: LaneA,
            },
            ComponentBinding {
                component: SensorLaneA,
                zone: LaneA,
            },
            ComponentBinding {
                component: FlightComputerA,
                zone: LaneA,
            },
            ComponentBinding {
                component: ActuatorLaneA,
                zone: LaneA,
            },
            ComponentBinding {
                component: PowerLaneB,
                zone: LaneB,
            },
            ComponentBinding {
                component: SensorLaneB,
                zone: LaneB,
            },
            ComponentBinding {
                component: FlightComputerB,
                zone: LaneB,
            },
            ComponentBinding {
                component: ActuatorLaneB,
                zone: LaneB,
            },
            ComponentBinding {
                component: MainRotorSystem,
                zone: SharedPropulsion,
            },
            ComponentBinding {
                component: TailRotorSystem,
                zone: SharedPropulsion,
            },
            ComponentBinding {
                component: EngineControl,
                zone: SharedPropulsion,
            },
            ComponentBinding {
                component: EvidenceRecorder,
                zone: SharedEvidence,
            },
        ];
        let dependencies = vec![
            RequiredDependency {
                upstream: PowerLaneA,
                downstream: SensorLaneA,
            },
            RequiredDependency {
                upstream: PowerLaneA,
                downstream: FlightComputerA,
            },
            RequiredDependency {
                upstream: PowerLaneA,
                downstream: ActuatorLaneA,
            },
            RequiredDependency {
                upstream: PowerLaneB,
                downstream: SensorLaneB,
            },
            RequiredDependency {
                upstream: PowerLaneB,
                downstream: FlightComputerB,
            },
            RequiredDependency {
                upstream: PowerLaneB,
                downstream: ActuatorLaneB,
            },
            RequiredDependency {
                upstream: FlightComputerA,
                downstream: ActuatorLaneA,
            },
            RequiredDependency {
                upstream: FlightComputerB,
                downstream: ActuatorLaneB,
            },
        ];
        let lane_a = vec![SensorLaneA, FlightComputerA, ActuatorLaneA];
        let lane_b = vec![SensorLaneB, FlightComputerB, ActuatorLaneB];
        let services = vec![
            ServiceRequirement {
                service: Navigation,
                critical: true,
                alternatives: vec![
                    vec![SensorLaneA, FlightComputerA],
                    vec![SensorLaneB, FlightComputerB],
                ],
            },
            ServiceRequirement {
                service: AttitudeControl,
                critical: true,
                alternatives: vec![
                    [lane_a.clone(), vec![MainRotorSystem]].concat(),
                    [lane_b.clone(), vec![MainRotorSystem]].concat(),
                ],
            },
            ServiceRequirement {
                service: VerticalControl,
                critical: true,
                alternatives: vec![
                    vec![FlightComputerA, ActuatorLaneA, MainRotorSystem],
                    vec![FlightComputerB, ActuatorLaneB, MainRotorSystem],
                ],
            },
            ServiceRequirement {
                service: YawControl,
                critical: true,
                alternatives: vec![
                    vec![FlightComputerA, ActuatorLaneA, TailRotorSystem],
                    vec![FlightComputerB, ActuatorLaneB, TailRotorSystem],
                ],
            },
            ServiceRequirement {
                service: EngineManagement,
                critical: true,
                alternatives: vec![
                    vec![FlightComputerA, EngineControl],
                    vec![FlightComputerB, EngineControl],
                ],
            },
            ServiceRequirement {
                service: EvidenceRecording,
                critical: false,
                alternatives: vec![
                    vec![FlightComputerA, EvidenceRecorder],
                    vec![FlightComputerB, EvidenceRecorder],
                ],
            },
        ];
        Self::new(components, dependencies, services)
            .expect("default fault-containment architecture must remain valid")
    }

    pub fn assess(
        &self,
        initial_faults: &[FlightComponent],
    ) -> Result<FaultContainmentAssessment, FaultContainmentError> {
        let known: BTreeSet<_> = self
            .components
            .iter()
            .map(|binding| binding.component)
            .collect();
        if initial_faults.iter().any(|fault| !known.contains(fault)) {
            return Err(FaultContainmentError::UnknownDependencyComponent);
        }
        let initial: BTreeSet<_> = initial_faults.iter().copied().collect();
        let mut unavailable = initial.clone();
        let mut queue: VecDeque<_> = initial.iter().copied().collect();
        while let Some(failed) = queue.pop_front() {
            for dependency in self
                .dependencies
                .iter()
                .filter(|dependency| dependency.upstream == failed)
            {
                if unavailable.insert(dependency.downstream) {
                    queue.push_back(dependency.downstream);
                }
            }
        }

        let mut lost_services = Vec::new();
        let mut lost_critical_services = Vec::new();
        for requirement in &self.services {
            let available = requirement.alternatives.iter().any(|alternative| {
                alternative
                    .iter()
                    .all(|component| !unavailable.contains(component))
            });
            if !available {
                lost_services.push(requirement.service);
                if requirement.critical {
                    lost_critical_services.push(requirement.service);
                }
            }
        }

        let affected_zones: BTreeSet<_> = self
            .components
            .iter()
            .filter(|binding| unavailable.contains(&binding.component))
            .map(|binding| binding.zone)
            .collect();
        let initial_zones: BTreeSet<_> = self
            .components
            .iter()
            .filter(|binding| initial.contains(&binding.component))
            .map(|binding| binding.zone)
            .collect();
        let cross_zone_propagation = unavailable.len() > initial.len()
            && affected_zones
                .iter()
                .any(|zone| !initial_zones.contains(zone));

        Ok(FaultContainmentAssessment {
            initial_faults: initial.into_iter().collect(),
            unavailable_components: unavailable.into_iter().collect(),
            affected_zones: affected_zones.into_iter().collect(),
            lost_services,
            lost_critical_services,
            cross_zone_propagation,
        })
    }

    /// Enumerate components whose single failure loses at least one critical service.
    pub fn single_point_failures(
        &self,
    ) -> Result<Vec<SinglePointFailureReport>, FaultContainmentError> {
        let mut reports = Vec::new();
        for binding in &self.components {
            let assessment = self.assess(&[binding.component])?;
            if !assessment.lost_critical_services.is_empty() {
                reports.push(SinglePointFailureReport {
                    component: binding.component,
                    lost_critical_services: assessment.lost_critical_services,
                });
            }
        }
        Ok(reports)
    }
}

impl Default for FaultContainmentArchitecture {
    fn default() -> Self {
        Self::default_dual_lane()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_lane_power_failure_is_contained() {
        let architecture = FaultContainmentArchitecture::default();
        let assessment = architecture.assess(&[FlightComponent::PowerLaneA]).unwrap();
        assert!(assessment.lost_critical_services.is_empty());
        assert!(!assessment.cross_zone_propagation);
        assert!(
            assessment
                .unavailable_components
                .contains(&FlightComponent::FlightComputerA)
        );
        assert!(
            !assessment
                .unavailable_components
                .contains(&FlightComponent::FlightComputerB)
        );
    }

    #[test]
    fn both_lane_power_failure_loses_control_services() {
        let assessment = FaultContainmentArchitecture::default()
            .assess(&[FlightComponent::PowerLaneA, FlightComponent::PowerLaneB])
            .unwrap();
        assert!(
            assessment
                .lost_critical_services
                .contains(&FlightService::Navigation)
        );
        assert!(
            assessment
                .lost_critical_services
                .contains(&FlightService::AttitudeControl)
        );
    }

    #[test]
    fn shared_rotor_system_is_reported_as_single_point_failure() {
        let reports = FaultContainmentArchitecture::default()
            .single_point_failures()
            .unwrap();
        let rotor = reports
            .iter()
            .find(|report| report.component == FlightComponent::MainRotorSystem)
            .unwrap();
        assert!(
            rotor
                .lost_critical_services
                .contains(&FlightService::VerticalControl)
        );
        assert!(
            rotor
                .lost_critical_services
                .contains(&FlightService::AttitudeControl)
        );
    }

    #[test]
    fn common_power_feed_exposes_hidden_cross_lane_dependency() {
        use ContainmentZone::*;
        use FlightComponent::*;
        let mut architecture = FaultContainmentArchitecture::default();
        architecture.components.push(ComponentBinding {
            component: CommonPowerFeed,
            zone: CommonCause,
        });
        architecture.dependencies.extend([
            RequiredDependency {
                upstream: CommonPowerFeed,
                downstream: PowerLaneA,
            },
            RequiredDependency {
                upstream: CommonPowerFeed,
                downstream: PowerLaneB,
            },
        ]);
        let assessment = architecture.assess(&[CommonPowerFeed]).unwrap();
        assert!(assessment.cross_zone_propagation);
        assert!(
            assessment
                .lost_critical_services
                .contains(&FlightService::Navigation)
        );
    }
}
