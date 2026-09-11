// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Neutral transport-network vocabulary for infrastructure trade studies.
//!
//! This module deliberately does not implement routing, physics, economics,
//! scheduling, or actuator control. It only provides evidence-bearing graph
//! types so landers, elevators, mass drivers, tugs, surface haulage, and future
//! modes can be compared without encoding a preferred answer.

use serde::{Deserialize, Serialize};

use crate::{AssetId, EvidenceStatus, InterfaceRef};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransportEdgeId(pub String);

impl TransportEdgeId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn is_well_formed(&self) -> bool {
        !self.0.trim().is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TransportMode {
    EarthLaunch,
    LunarLander,
    SurfaceHaul,
    GuidedSurfaceFreight,
    LunarElevator,
    MassDriver,
    SolarElectricTug,
    NuclearElectricTug,
    ChemicalTug,
    MomentumExchangeTether,
    DomainSpecific(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CargoKind {
    General,
    Fragile,
    BulkCommodity,
    Cryogenic,
    Pressurized,
    Hazardous,
    Biological,
    HumanRated,
    DomainSpecific(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CargoProfile {
    pub cargo_id: String,
    pub kind: CargoKind,
    pub mass_kg: f64,
    pub dimensions_m: [f64; 3],
    /// Maximum allowed positive acceleration magnitude, m/s^2.
    pub max_acceleration_m_s2: f64,
    /// Maximum allowed vibration RMS proxy in the declared protocol units.
    /// This leaf type carries the scalar but not a vibration-spectrum model.
    pub max_vibration_rms: f64,
    /// Maximum acceptable end-to-end transit time, seconds, when time-critical.
    pub max_transit_time_s: Option<f64>,
    pub temperature_min_k: Option<f64>,
    pub temperature_max_k: Option<f64>,
    pub contamination_sensitive: bool,
    pub custody_required: bool,
    pub evidence_refs: Vec<String>,
}

impl CargoProfile {
    pub fn is_well_formed(&self) -> bool {
        !self.cargo_id.trim().is_empty()
            && self.mass_kg.is_finite()
            && self.mass_kg > 0.0
            && self
                .dimensions_m
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            && self.max_acceleration_m_s2.is_finite()
            && self.max_acceleration_m_s2 > 0.0
            && self.max_vibration_rms.is_finite()
            && self.max_vibration_rms >= 0.0
            && self
                .max_transit_time_s
                .is_none_or(|value| value.is_finite() && value > 0.0)
            && valid_temperature_range(self.temperature_min_k, self.temperature_max_k)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundedMetric {
    pub lower: f64,
    pub nominal: f64,
    pub upper: f64,
    pub unit: String,
    pub evidence_status: EvidenceStatus,
    pub evidence_refs: Vec<String>,
}

impl BoundedMetric {
    pub fn is_well_formed(&self) -> bool {
        self.lower.is_finite()
            && self.nominal.is_finite()
            && self.upper.is_finite()
            && self.lower <= self.nominal
            && self.nominal <= self.upper
            && !self.unit.trim().is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransportLocationRef {
    /// Named reference frame or location namespace. This type does not define
    /// the transform itself; orbital/surface domain adapters own that contract.
    pub frame: String,
    /// Stable location/site/orbit identifier meaningful in `frame`.
    pub location: String,
    pub evidence_refs: Vec<String>,
}

impl TransportLocationRef {
    pub fn is_well_formed(&self) -> bool {
        !self.frame.trim().is_empty() && !self.location.trim().is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransportNode {
    pub asset: AssetId,
    pub label: String,
    pub location: TransportLocationRef,
    pub interfaces: Vec<InterfaceRef>,
    /// Optional evidence-bearing storage capacity, kg of generic cargo mass.
    /// Domain-specific storage constraints belong in higher-level adapters.
    pub storage_capacity_kg: Option<BoundedMetric>,
    pub evidence_refs: Vec<String>,
}

impl TransportNode {
    pub fn is_well_formed(&self) -> bool {
        self.asset.is_well_formed()
            && !self.label.trim().is_empty()
            && self.location.is_well_formed()
            && self.interfaces.iter().all(InterfaceRef::is_well_formed)
            && self
                .storage_capacity_kg
                .as_ref()
                .is_none_or(BoundedMetric::is_well_formed)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransportEdge {
    pub edge_id: TransportEdgeId,
    pub mode: TransportMode,
    pub origin: AssetId,
    pub destination: AssetId,
    pub supported_cargo: Vec<CargoKind>,
    pub payload_per_trip_kg: BoundedMetric,
    pub annual_capacity_kg: BoundedMetric,
    pub transit_time_s: BoundedMetric,
    pub electrical_energy_kwh_per_kg: BoundedMetric,
    pub propellant_kg_per_kg: BoundedMetric,
    /// Dimensionless probability in [0,1].
    pub delivery_success_probability: BoundedMetric,
    /// Dimensionless long-run availability in [0,1].
    pub availability_fraction: BoundedMetric,
    pub required_interfaces: Vec<InterfaceRef>,
    pub infrastructure_dependencies: Vec<AssetId>,
    pub evidence_refs: Vec<String>,
}

impl TransportEdge {
    pub fn is_well_formed(&self) -> bool {
        self.edge_id.is_well_formed()
            && self.origin.is_well_formed()
            && self.destination.is_well_formed()
            && self.origin != self.destination
            && !self.supported_cargo.is_empty()
            && metric_is_non_negative(&self.payload_per_trip_kg)
            && metric_is_non_negative(&self.annual_capacity_kg)
            && metric_is_non_negative(&self.transit_time_s)
            && metric_is_non_negative(&self.electrical_energy_kwh_per_kg)
            && metric_is_non_negative(&self.propellant_kg_per_kg)
            && metric_is_probability(&self.delivery_success_probability)
            && metric_is_probability(&self.availability_fraction)
            && self
                .required_interfaces
                .iter()
                .all(InterfaceRef::is_well_formed)
            && self
                .infrastructure_dependencies
                .iter()
                .all(AssetId::is_well_formed)
    }

    pub fn supports(&self, cargo: &CargoProfile) -> bool {
        self.is_well_formed()
            && cargo.is_well_formed()
            && self.supported_cargo.iter().any(|kind| kind == &cargo.kind)
            && cargo.mass_kg <= self.payload_per_trip_kg.upper
            && self
                .transit_time_s
                .upper
                .is_finite()
            && cargo
                .max_transit_time_s
                .is_none_or(|limit| self.transit_time_s.upper <= limit)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TransportGraph {
    pub nodes: Vec<TransportNode>,
    pub edges: Vec<TransportEdge>,
}

impl TransportGraph {
    pub fn is_well_formed(&self) -> bool {
        if self.nodes.iter().any(|node| !node.is_well_formed())
            || self.edges.iter().any(|edge| !edge.is_well_formed())
        {
            return false;
        }

        for (index, node) in self.nodes.iter().enumerate() {
            if self.nodes[index + 1..]
                .iter()
                .any(|other| other.asset == node.asset)
            {
                return false;
            }
        }
        for (index, edge) in self.edges.iter().enumerate() {
            if self.edges[index + 1..]
                .iter()
                .any(|other| other.edge_id == edge.edge_id)
            {
                return false;
            }
        }

        self.edges.iter().all(|edge| {
            self.nodes.iter().any(|node| node.asset == edge.origin)
                && self.nodes.iter().any(|node| node.asset == edge.destination)
                && edge.infrastructure_dependencies.iter().all(|dependency| {
                    self.nodes.iter().any(|node| &node.asset == dependency)
                })
        })
    }
}

fn valid_temperature_range(min_k: Option<f64>, max_k: Option<f64>) -> bool {
    let valid = |value: f64| value.is_finite() && value > 0.0;
    match (min_k, max_k) {
        (None, None) => true,
        (Some(min), None) => valid(min),
        (None, Some(max)) => valid(max),
        (Some(min), Some(max)) => valid(min) && valid(max) && min <= max,
    }
}

fn metric_is_non_negative(metric: &BoundedMetric) -> bool {
    metric.is_well_formed() && metric.lower >= 0.0
}

fn metric_is_probability(metric: &BoundedMetric) -> bool {
    metric.is_well_formed() && metric.lower >= 0.0 && metric.upper <= 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metric(lower: f64, nominal: f64, upper: f64, unit: &str) -> BoundedMetric {
        BoundedMetric {
            lower,
            nominal,
            upper,
            unit: unit.into(),
            evidence_status: EvidenceStatus::Modeled,
            evidence_refs: vec!["model-v0".into()],
        }
    }

    fn node(id: &str) -> TransportNode {
        TransportNode {
            asset: AssetId::new(id),
            label: id.into(),
            location: TransportLocationRef {
                frame: "reference-network".into(),
                location: id.into(),
                evidence_refs: vec![],
            },
            interfaces: vec![],
            storage_capacity_kg: None,
            evidence_refs: vec![],
        }
    }

    fn edge(mode: TransportMode) -> TransportEdge {
        TransportEdge {
            edge_id: TransportEdgeId::new("edge-1"),
            mode,
            origin: AssetId::new("surface"),
            destination: AssetId::new("eml1"),
            supported_cargo: vec![CargoKind::General, CargoKind::Fragile],
            payload_per_trip_kg: metric(10.0, 100.0, 1_000.0, "kg"),
            annual_capacity_kg: metric(1_000.0, 10_000.0, 100_000.0, "kg/year"),
            transit_time_s: metric(1_000.0, 10_000.0, 100_000.0, "s"),
            electrical_energy_kwh_per_kg: metric(0.0, 1.0, 10.0, "kWh/kg"),
            propellant_kg_per_kg: metric(0.0, 0.1, 1.0, "kg/kg"),
            delivery_success_probability: metric(0.90, 0.98, 0.999, "1"),
            availability_fraction: metric(0.50, 0.90, 0.99, "1"),
            required_interfaces: vec![],
            infrastructure_dependencies: vec![],
            evidence_refs: vec![],
        }
    }

    fn cargo() -> CargoProfile {
        CargoProfile {
            cargo_id: "cargo-1".into(),
            kind: CargoKind::General,
            mass_kg: 50.0,
            dimensions_m: [1.0, 1.0, 1.0],
            max_acceleration_m_s2: 10.0,
            max_vibration_rms: 1.0,
            max_transit_time_s: None,
            temperature_min_k: None,
            temperature_max_k: None,
            contamination_sensitive: false,
            custody_required: true,
            evidence_refs: vec![],
        }
    }

    #[test]
    fn bounded_metric_requires_ordered_bounds() {
        assert!(metric(1.0, 2.0, 3.0, "kg").is_well_formed());
        assert!(!metric(3.0, 2.0, 1.0, "kg").is_well_formed());
    }

    #[test]
    fn graph_rejects_dangling_transport_edge() {
        let graph = TransportGraph {
            nodes: vec![node("surface")],
            edges: vec![edge(TransportMode::LunarElevator)],
        };
        assert!(!graph.is_well_formed());
    }

    #[test]
    fn graph_treats_elevator_and_lander_as_peer_modes() {
        let mut elevator = edge(TransportMode::LunarElevator);
        elevator.edge_id = TransportEdgeId::new("elevator");
        let mut lander = edge(TransportMode::LunarLander);
        lander.edge_id = TransportEdgeId::new("lander");
        let graph = TransportGraph {
            nodes: vec![node("surface"), node("eml1")],
            edges: vec![elevator, lander],
        };
        assert!(graph.is_well_formed());
    }

    #[test]
    fn edge_supports_cargo_only_inside_declared_mass_and_time_envelope() {
        let edge = edge(TransportMode::LunarElevator);
        let mut cargo = cargo();
        assert!(edge.supports(&cargo));
        cargo.mass_kg = 2_000.0;
        assert!(!edge.supports(&cargo));
    }

    #[test]
    fn probability_metrics_must_stay_inside_unit_interval() {
        let mut edge = edge(TransportMode::LunarLander);
        edge.delivery_success_probability = metric(0.9, 1.0, 1.1, "1");
        assert!(!edge.is_well_formed());
    }

    #[test]
    fn human_cargo_is_explicit_not_implied_by_general_cargo() {
        let edge = edge(TransportMode::LunarElevator);
        let mut cargo = cargo();
        cargo.kind = CargoKind::HumanRated;
        assert!(!edge.supports(&cargo));
    }
}
