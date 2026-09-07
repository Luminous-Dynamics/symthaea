// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Quality-enforced routing over the typed resource graph.
//!
//! Every port must declare its quality contract when entering this graph. For
//! quality-sensitive resource kinds, an explicit `NotApplicable` declaration is
//! rejected. This prevents a caller from validating quantity/conservation while
//! accidentally omitting a required grade/provenance check.

use crate::{
    QualityError, QualityViolation, ResourceQualityProfile, ResourceQualityRequirement,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_resource_model::{
    PortDirection, ResourceEdge, ResourceError, ResourceGraph, ResourceKey, ResourceKind,
    ResourcePort,
};
use thiserror::Error;

/// Resource kinds that must carry quality contracts in a qualified graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualityEnforcementPolicy {
    required_kinds: BTreeSet<ResourceKind>,
}

impl QualityEnforcementPolicy {
    pub fn new(required_kinds: impl IntoIterator<Item = ResourceKind>) -> Self {
        Self {
            required_kinds: required_kinds.into_iter().collect(),
        }
    }

    /// Physical defaults where quantity alone is routinely insufficient for
    /// routability: heat grade, cooling grade, and water quality.
    pub fn physical_defaults() -> Self {
        Self::new([
            ResourceKind::ThermalEnergy,
            ResourceKind::CoolingCapacity,
            ResourceKind::Water,
        ])
    }

    /// Explicitly permit quantity-only routing for every resource kind.
    pub fn permissive() -> Self {
        Self::new([])
    }

    pub fn requires(&self, kind: ResourceKind) -> bool {
        self.required_kinds.contains(&kind)
    }
}

impl Default for QualityEnforcementPolicy {
    fn default() -> Self {
        Self::physical_defaults()
    }
}

/// Quality declaration attached to one resource port.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PortQualityContract {
    /// Grade is deliberately irrelevant for this port. This is an explicit opt-out,
    /// not an omitted field, and may be forbidden by the graph policy.
    NotApplicable,
    /// Outbound resource grade supplied by this port.
    Provide(ResourceQualityProfile),
    /// Inbound grade required by this port.
    Require(ResourceQualityRequirement),
    /// A bidirectional interface that both provides and requires a defined grade.
    Bidirectional {
        provided: ResourceQualityProfile,
        required: ResourceQualityRequirement,
    },
}

impl PortQualityContract {
    fn provided(&self) -> Option<&ResourceQualityProfile> {
        match self {
            Self::Provide(profile) | Self::Bidirectional { provided: profile, .. } => Some(profile),
            Self::NotApplicable | Self::Require(_) => None,
        }
    }

    fn required(&self) -> Option<&ResourceQualityRequirement> {
        match self {
            Self::Require(requirement) | Self::Bidirectional { required: requirement, .. } => {
                Some(requirement)
            }
            Self::NotApplicable | Self::Provide(_) => None,
        }
    }

    fn validate(
        &self,
        node: &str,
        port: &ResourcePort,
        policy: &QualityEnforcementPolicy,
    ) -> Result<(), QualifiedGraphError> {
        if policy.requires(port.capacity.key.kind) && matches!(self, Self::NotApplicable) {
            return Err(QualifiedGraphError::QualityRequired {
                node: node.to_owned(),
                port: port.id.clone(),
                kind: port.capacity.key.kind,
            });
        }

        let direction_ok = match port.direction {
            PortDirection::Input => matches!(self, Self::Require(_) | Self::NotApplicable),
            PortDirection::Output => matches!(self, Self::Provide(_) | Self::NotApplicable),
            PortDirection::Bidirectional => {
                matches!(self, Self::Bidirectional { .. } | Self::NotApplicable)
            }
        };
        if !direction_ok {
            return Err(QualifiedGraphError::ContractDirectionMismatch {
                node: node.to_owned(),
                port: port.id.clone(),
                direction: port.direction,
            });
        }

        if let Some(profile) = self.provided() {
            validate_key(node, port, profile.key)?;
        }
        if let Some(requirement) = self.required() {
            validate_key(node, port, requirement.key)?;
        }
        Ok(())
    }
}

fn validate_key(
    node: &str,
    port: &ResourcePort,
    quality_key: ResourceKey,
) -> Result<(), QualifiedGraphError> {
    if quality_key != port.capacity.key {
        return Err(QualifiedGraphError::ContractKeyMismatch {
            node: node.to_owned(),
            port: port.id.clone(),
            port_key: port.capacity.key,
            quality_key,
        });
    }
    Ok(())
}

/// Resource port plus the grade contract that must travel with it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualifiedPort {
    pub port: ResourcePort,
    pub quality: PortQualityContract,
}

impl QualifiedPort {
    pub fn new(port: ResourcePort, quality: PortQualityContract) -> Self {
        Self { port, quality }
    }
}

/// Resource graph that enforces quantity, aggregate port capacity, and quality
/// compatibility as one routing operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualifiedResourceGraph {
    graph: ResourceGraph,
    contracts: BTreeMap<(String, String), PortQualityContract>,
    policy: QualityEnforcementPolicy,
}

impl QualifiedResourceGraph {
    pub fn new(policy: QualityEnforcementPolicy) -> Self {
        Self {
            graph: ResourceGraph::default(),
            contracts: BTreeMap::new(),
            policy,
        }
    }

    pub fn add_node(
        &mut self,
        node_id: impl Into<String>,
        ports: impl IntoIterator<Item = QualifiedPort>,
    ) -> Result<(), QualifiedGraphError> {
        let node_id = node_id.into();
        let mut plain_ports = Vec::new();
        let mut pending_contracts = Vec::new();

        for qualified in ports {
            qualified
                .quality
                .validate(&node_id, &qualified.port, &self.policy)?;
            pending_contracts.push((qualified.port.id.clone(), qualified.quality));
            plain_ports.push(qualified.port);
        }

        self.graph.add_node(node_id.clone(), plain_ports)?;
        for (port_id, contract) in pending_contracts {
            self.contracts
                .insert((node_id.clone(), port_id), contract);
        }
        Ok(())
    }

    /// Connect an edge only after the destination's grade requirement, if any,
    /// has been checked against the source's declared grade.
    pub fn connect(&mut self, edge: ResourceEdge) -> Result<(), QualifiedGraphError> {
        let source_contract = self
            .contracts
            .get(&(edge.from_node.clone(), edge.from_port.clone()))
            .ok_or_else(|| QualifiedGraphError::MissingContract {
                node: edge.from_node.clone(),
                port: edge.from_port.clone(),
            })?;
        let destination_contract = self
            .contracts
            .get(&(edge.to_node.clone(), edge.to_port.clone()))
            .ok_or_else(|| QualifiedGraphError::MissingContract {
                node: edge.to_node.clone(),
                port: edge.to_port.clone(),
            })?;

        if let Some(requirement) = destination_contract.required() {
            let profile = source_contract.provided().ok_or_else(|| {
                QualifiedGraphError::MissingProducerProfile {
                    edge_id: edge.id.clone(),
                    node: edge.from_node.clone(),
                    port: edge.from_port.clone(),
                }
            })?;
            let compatibility = requirement.evaluate(profile)?;
            if !compatibility.compatible {
                return Err(QualifiedGraphError::QualityMismatch {
                    edge_id: edge.id.clone(),
                    violations: compatibility.violations,
                });
            }
        }

        self.graph.connect(edge)?;
        Ok(())
    }

    pub fn graph(&self) -> &ResourceGraph {
        &self.graph
    }

    pub fn policy(&self) -> &QualityEnforcementPolicy {
        &self.policy
    }
}

impl Default for QualifiedResourceGraph {
    fn default() -> Self {
        Self::new(QualityEnforcementPolicy::default())
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum QualifiedGraphError {
    #[error(transparent)]
    Resource(#[from] ResourceError),
    #[error(transparent)]
    Quality(#[from] QualityError),
    #[error("quality contract is required for {kind:?} port {node}/{port}")]
    QualityRequired {
        node: String,
        port: String,
        kind: ResourceKind,
    },
    #[error("quality contract direction does not match {direction:?} port {node}/{port}")]
    ContractDirectionMismatch {
        node: String,
        port: String,
        direction: PortDirection,
    },
    #[error(
        "quality key {quality_key:?} does not match resource key {port_key:?} on {node}/{port}"
    )]
    ContractKeyMismatch {
        node: String,
        port: String,
        port_key: ResourceKey,
        quality_key: ResourceKey,
    },
    #[error("missing explicit quality contract for port {node}/{port}")]
    MissingContract { node: String, port: String },
    #[error("edge {edge_id} requires source quality but {node}/{port} has no producer profile")]
    MissingProducerProfile {
        edge_id: String,
        node: String,
        port: String,
    },
    #[error("edge {edge_id} fails resource-grade compatibility: {violations:?}")]
    QualityMismatch {
        edge_id: String,
        violations: Vec<QualityViolation>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NumericQualityConstraint, QualityMetric};
    use symthaea_resource_model::{ResourceAmount, ResourceUnit};

    fn amount(kind: ResourceKind, value: f64) -> ResourceAmount {
        ResourceAmount::new(kind, ResourceUnit::Watt, value).unwrap()
    }

    fn port(
        id: &str,
        direction: PortDirection,
        kind: ResourceKind,
        capacity: f64,
        quality: PortQualityContract,
    ) -> QualifiedPort {
        QualifiedPort::new(
            ResourcePort {
                id: id.into(),
                direction,
                capacity: amount(kind, capacity),
            },
            quality,
        )
    }

    fn heat_profile(temp_c: f64) -> ResourceQualityProfile {
        let key = ResourceKey::new(ResourceKind::ThermalEnergy, ResourceUnit::Watt).unwrap();
        let mut profile = ResourceQualityProfile::new(key);
        profile
            .set_numeric(QualityMetric::TemperatureCelsius, temp_c)
            .unwrap();
        profile
    }

    fn heat_requirement(min_temp_c: f64) -> ResourceQualityRequirement {
        let key = ResourceKey::new(ResourceKind::ThermalEnergy, ResourceUnit::Watt).unwrap();
        let mut requirement = ResourceQualityRequirement::new(key);
        requirement
            .require_numeric(NumericQualityConstraint {
                metric: QualityMetric::TemperatureCelsius,
                minimum: Some(min_temp_c),
                maximum: None,
            })
            .unwrap();
        requirement
    }

    fn edge(id: &str, watts: f64) -> ResourceEdge {
        ResourceEdge {
            id: id.into(),
            from_node: "compute".into(),
            from_port: "heat-out".into(),
            to_node: "sink".into(),
            to_port: "heat-in".into(),
            amount: amount(ResourceKind::ThermalEnergy, watts),
            loss_fraction: 0.0,
        }
    }

    #[test]
    fn physical_policy_rejects_implicit_unqualified_heat() {
        let mut graph = QualifiedResourceGraph::default();
        let result = graph.add_node(
            "compute",
            [port(
                "heat-out",
                PortDirection::Output,
                ResourceKind::ThermalEnergy,
                100.0,
                PortQualityContract::NotApplicable,
            )],
        );
        assert!(matches!(result, Err(QualifiedGraphError::QualityRequired { .. })));
    }

    #[test]
    fn compatible_heat_is_routed_and_counted_by_underlying_graph() {
        let mut graph = QualifiedResourceGraph::default();
        graph
            .add_node(
                "compute",
                [port(
                    "heat-out",
                    PortDirection::Output,
                    ResourceKind::ThermalEnergy,
                    100.0,
                    PortQualityContract::Provide(heat_profile(48.0)),
                )],
            )
            .unwrap();
        graph
            .add_node(
                "sink",
                [port(
                    "heat-in",
                    PortDirection::Input,
                    ResourceKind::ThermalEnergy,
                    100.0,
                    PortQualityContract::Require(heat_requirement(35.0)),
                )],
            )
            .unwrap();
        graph.connect(edge("heat", 80.0)).unwrap();
        assert_eq!(graph.graph().edges().len(), 1);
    }

    #[test]
    fn conserved_but_wrong_grade_heat_is_rejected_before_routing() {
        let mut graph = QualifiedResourceGraph::default();
        graph
            .add_node(
                "compute",
                [port(
                    "heat-out",
                    PortDirection::Output,
                    ResourceKind::ThermalEnergy,
                    100.0,
                    PortQualityContract::Provide(heat_profile(42.0)),
                )],
            )
            .unwrap();
        graph
            .add_node(
                "sink",
                [port(
                    "heat-in",
                    PortDirection::Input,
                    ResourceKind::ThermalEnergy,
                    100.0,
                    PortQualityContract::Require(heat_requirement(60.0)),
                )],
            )
            .unwrap();
        let result = graph.connect(edge("wrong-grade", 80.0));
        assert!(matches!(
            result,
            Err(QualifiedGraphError::QualityMismatch { edge_id, .. }) if edge_id == "wrong-grade"
        ));
        assert!(graph.graph().edges().is_empty());
    }

    #[test]
    fn qualified_graph_preserves_aggregate_capacity_enforcement() {
        let mut graph = QualifiedResourceGraph::default();
        graph
            .add_node(
                "compute",
                [port(
                    "heat-out",
                    PortDirection::Output,
                    ResourceKind::ThermalEnergy,
                    100.0,
                    PortQualityContract::Provide(heat_profile(48.0)),
                )],
            )
            .unwrap();
        graph
            .add_node(
                "sink",
                [port(
                    "heat-in",
                    PortDirection::Input,
                    ResourceKind::ThermalEnergy,
                    100.0,
                    PortQualityContract::Require(heat_requirement(35.0)),
                )],
            )
            .unwrap();
        graph.connect(edge("first", 60.0)).unwrap();
        let result = graph.connect(edge("second", 50.0));
        assert!(matches!(
            result,
            Err(QualifiedGraphError::Resource(
                ResourceError::AggregatePortCapacityExceeded { .. }
            ))
        ));
    }

    #[test]
    fn quantity_only_electricity_must_still_be_declared_explicitly() {
        let mut graph = QualifiedResourceGraph::default();
        graph
            .add_node(
                "grid",
                [port(
                    "out",
                    PortDirection::Output,
                    ResourceKind::Electricity,
                    100.0,
                    PortQualityContract::NotApplicable,
                )],
            )
            .unwrap();
        graph
            .add_node(
                "load",
                [port(
                    "in",
                    PortDirection::Input,
                    ResourceKind::Electricity,
                    100.0,
                    PortQualityContract::NotApplicable,
                )],
            )
            .unwrap();
        graph
            .connect(ResourceEdge {
                id: "power".into(),
                from_node: "grid".into(),
                from_port: "out".into(),
                to_node: "load".into(),
                to_port: "in".into(),
                amount: amount(ResourceKind::Electricity, 75.0),
                loss_fraction: 0.01,
            })
            .unwrap();
    }

    #[test]
    fn consumer_requirement_without_producer_profile_fails_closed() {
        let compute_key = ResourceKey::new(ResourceKind::Compute, ResourceUnit::GpuSecond).unwrap();
        let mut requirement = ResourceQualityRequirement::new(compute_key);
        requirement.require_tag("accelerator.class", "trusted-a").unwrap();

        let mut graph = QualifiedResourceGraph::new(QualityEnforcementPolicy::permissive());
        graph
            .add_node(
                "compute",
                [QualifiedPort::new(
                    ResourcePort {
                        id: "out".into(),
                        direction: PortDirection::Output,
                        capacity: ResourceAmount::new(
                            ResourceKind::Compute,
                            ResourceUnit::GpuSecond,
                            10.0,
                        )
                        .unwrap(),
                    },
                    PortQualityContract::NotApplicable,
                )],
            )
            .unwrap();
        graph
            .add_node(
                "sink",
                [QualifiedPort::new(
                    ResourcePort {
                        id: "in".into(),
                        direction: PortDirection::Input,
                        capacity: ResourceAmount::new(
                            ResourceKind::Compute,
                            ResourceUnit::GpuSecond,
                            10.0,
                        )
                        .unwrap(),
                    },
                    PortQualityContract::Require(requirement),
                )],
            )
            .unwrap();
        let result = graph.connect(ResourceEdge {
            id: "compute".into(),
            from_node: "compute".into(),
            from_port: "out".into(),
            to_node: "sink".into(),
            to_port: "in".into(),
            amount: ResourceAmount::new(ResourceKind::Compute, ResourceUnit::GpuSecond, 5.0)
                .unwrap(),
            loss_fraction: 0.0,
        });
        assert!(matches!(
            result,
            Err(QualifiedGraphError::MissingProducerProfile { .. })
        ));
    }
}
