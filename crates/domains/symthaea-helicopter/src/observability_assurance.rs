// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Measurement-dependency and observability assurance.
//!
//! A numerically healthy estimator may still be unobservable when all active
//! measurements share one failure domain or when a state depends on a single
//! surviving source. This module evaluates source count and independence-domain
//! diversity for each safety-relevant estimated quantity.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EstimatedQuantity {
    HorizontalPosition,
    Altitude,
    HorizontalVelocity,
    VerticalVelocity,
    Attitude,
    AngularRate,
    Heading,
    MainRotorSpeed,
    TailRotorSpeed,
    FuelQuantity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ObservationSource {
    GnssA,
    GnssB,
    ImuA,
    ImuB,
    BarometerA,
    BarometerB,
    MagnetometerA,
    MagnetometerB,
    MainRotorTachA,
    MainRotorTachB,
    TailRotorTachA,
    TailRotorTachB,
    FuelGaugeA,
    FuelGaugeB,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ObservationDomain {
    SatelliteConstellation,
    InertialLaneA,
    InertialLaneB,
    PressureLaneA,
    PressureLaneB,
    MagneticLaneA,
    MagneticLaneB,
    RotorElectricalLaneA,
    RotorElectricalLaneB,
    FuelElectricalLaneA,
    FuelElectricalLaneB,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensorCapability {
    pub source: ObservationSource,
    pub domain: ObservationDomain,
    pub quantities: Vec<EstimatedQuantity>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantityObservabilityRequirement {
    pub quantity: EstimatedQuantity,
    pub minimum_sources: usize,
    pub minimum_independent_domains: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensorAvailability {
    pub source: ObservationSource,
    pub healthy: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantityObservabilityStatus {
    Observable,
    Degraded,
    Unobservable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantityObservabilityAssessment {
    pub quantity: EstimatedQuantity,
    pub status: QuantityObservabilityStatus,
    pub active_sources: Vec<ObservationSource>,
    pub independent_domains: Vec<ObservationDomain>,
    pub required_sources: usize,
    pub required_domains: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservabilityAssessment {
    pub quantities: Vec<QuantityObservabilityAssessment>,
    pub all_quantities_observable: bool,
    pub unobservable_quantities: Vec<EstimatedQuantity>,
    pub degraded_quantities: Vec<EstimatedQuantity>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservabilityError {
    EmptyModel,
    DuplicateSource,
    EmptyCapability,
    DuplicateRequirement,
    InvalidRequirement,
    UnknownAvailabilitySource,
    DuplicateAvailabilitySource,
}

#[derive(Debug, Clone)]
pub struct ObservabilityAssuranceModel {
    capabilities: Vec<SensorCapability>,
    requirements: Vec<QuantityObservabilityRequirement>,
}

impl ObservabilityAssuranceModel {
    pub fn new(
        capabilities: Vec<SensorCapability>,
        requirements: Vec<QuantityObservabilityRequirement>,
    ) -> Result<Self, ObservabilityError> {
        if capabilities.is_empty() || requirements.is_empty() {
            return Err(ObservabilityError::EmptyModel);
        }
        let mut sources = BTreeSet::new();
        for capability in &capabilities {
            if !sources.insert(capability.source) {
                return Err(ObservabilityError::DuplicateSource);
            }
            if capability.quantities.is_empty() {
                return Err(ObservabilityError::EmptyCapability);
            }
            let unique_quantities: BTreeSet<_> = capability.quantities.iter().copied().collect();
            if unique_quantities.len() != capability.quantities.len() {
                return Err(ObservabilityError::EmptyCapability);
            }
        }
        let mut quantities = BTreeSet::new();
        for requirement in &requirements {
            if !quantities.insert(requirement.quantity) {
                return Err(ObservabilityError::DuplicateRequirement);
            }
            if requirement.minimum_sources == 0
                || requirement.minimum_independent_domains == 0
                || requirement.minimum_independent_domains > requirement.minimum_sources
            {
                return Err(ObservabilityError::InvalidRequirement);
            }
            let providers: Vec<_> = capabilities
                .iter()
                .filter(|capability| capability.quantities.contains(&requirement.quantity))
                .collect();
            let domains: BTreeSet<_> = providers.iter().map(|provider| provider.domain).collect();
            if providers.len() < requirement.minimum_sources
                || domains.len() < requirement.minimum_independent_domains
            {
                return Err(ObservabilityError::InvalidRequirement);
            }
        }
        Ok(Self {
            capabilities,
            requirements,
        })
    }

    pub fn assess(
        &self,
        availability: &[SensorAvailability],
    ) -> Result<ObservabilityAssessment, ObservabilityError> {
        let known_sources: BTreeSet<_> = self
            .capabilities
            .iter()
            .map(|capability| capability.source)
            .collect();
        let mut seen = BTreeSet::new();
        let mut health = BTreeMap::new();
        for entry in availability {
            if !known_sources.contains(&entry.source) {
                return Err(ObservabilityError::UnknownAvailabilitySource);
            }
            if !seen.insert(entry.source) {
                return Err(ObservabilityError::DuplicateAvailabilitySource);
            }
            health.insert(entry.source, entry.healthy);
        }

        let mut quantities = Vec::with_capacity(self.requirements.len());
        for requirement in &self.requirements {
            let mut active_sources = Vec::new();
            let mut domains = BTreeSet::new();
            for capability in self.capabilities.iter().filter(|capability| {
                capability.quantities.contains(&requirement.quantity)
                    && health.get(&capability.source).copied().unwrap_or(false)
            }) {
                active_sources.push(capability.source);
                domains.insert(capability.domain);
            }
            active_sources.sort();
            let independent_domains: Vec<_> = domains.into_iter().collect();
            let enough_sources = active_sources.len() >= requirement.minimum_sources;
            let enough_domains =
                independent_domains.len() >= requirement.minimum_independent_domains;
            let status = if enough_sources && enough_domains {
                QuantityObservabilityStatus::Observable
            } else if active_sources.is_empty() {
                QuantityObservabilityStatus::Unobservable
            } else {
                QuantityObservabilityStatus::Degraded
            };
            quantities.push(QuantityObservabilityAssessment {
                quantity: requirement.quantity,
                status,
                active_sources,
                independent_domains,
                required_sources: requirement.minimum_sources,
                required_domains: requirement.minimum_independent_domains,
            });
        }

        let unobservable_quantities = quantities
            .iter()
            .filter(|assessment| assessment.status == QuantityObservabilityStatus::Unobservable)
            .map(|assessment| assessment.quantity)
            .collect();
        let degraded_quantities = quantities
            .iter()
            .filter(|assessment| assessment.status == QuantityObservabilityStatus::Degraded)
            .map(|assessment| assessment.quantity)
            .collect();
        let all_quantities_observable = quantities
            .iter()
            .all(|assessment| assessment.status == QuantityObservabilityStatus::Observable);
        Ok(ObservabilityAssessment {
            quantities,
            all_quantities_observable,
            unobservable_quantities,
            degraded_quantities,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> ObservabilityAssuranceModel {
        ObservabilityAssuranceModel::new(
            vec![
                SensorCapability {
                    source: ObservationSource::ImuA,
                    domain: ObservationDomain::InertialLaneA,
                    quantities: vec![EstimatedQuantity::Attitude, EstimatedQuantity::AngularRate],
                },
                SensorCapability {
                    source: ObservationSource::ImuB,
                    domain: ObservationDomain::InertialLaneB,
                    quantities: vec![EstimatedQuantity::Attitude, EstimatedQuantity::AngularRate],
                },
                SensorCapability {
                    source: ObservationSource::GnssA,
                    domain: ObservationDomain::SatelliteConstellation,
                    quantities: vec![EstimatedQuantity::HorizontalPosition],
                },
            ],
            vec![
                QuantityObservabilityRequirement {
                    quantity: EstimatedQuantity::Attitude,
                    minimum_sources: 2,
                    minimum_independent_domains: 2,
                },
                QuantityObservabilityRequirement {
                    quantity: EstimatedQuantity::HorizontalPosition,
                    minimum_sources: 1,
                    minimum_independent_domains: 1,
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn independent_imus_satisfy_attitude_requirement() {
        let assessment = model()
            .assess(&[
                SensorAvailability {
                    source: ObservationSource::ImuA,
                    healthy: true,
                },
                SensorAvailability {
                    source: ObservationSource::ImuB,
                    healthy: true,
                },
                SensorAvailability {
                    source: ObservationSource::GnssA,
                    healthy: true,
                },
            ])
            .unwrap();
        assert!(assessment.all_quantities_observable);
    }

    #[test]
    fn one_imu_is_degraded_even_when_finite() {
        let assessment = model()
            .assess(&[
                SensorAvailability {
                    source: ObservationSource::ImuA,
                    healthy: true,
                },
                SensorAvailability {
                    source: ObservationSource::ImuB,
                    healthy: false,
                },
                SensorAvailability {
                    source: ObservationSource::GnssA,
                    healthy: true,
                },
            ])
            .unwrap();
        assert_eq!(
            assessment.degraded_quantities,
            vec![EstimatedQuantity::Attitude]
        );
    }

    #[test]
    fn missing_position_source_is_unobservable() {
        let assessment = model()
            .assess(&[
                SensorAvailability {
                    source: ObservationSource::ImuA,
                    healthy: true,
                },
                SensorAvailability {
                    source: ObservationSource::ImuB,
                    healthy: true,
                },
            ])
            .unwrap();
        assert_eq!(
            assessment.unobservable_quantities,
            vec![EstimatedQuantity::HorizontalPosition]
        );
    }
}
