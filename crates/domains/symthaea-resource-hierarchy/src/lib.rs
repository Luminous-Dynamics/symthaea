// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Recursive, self-similar resource-node composition.
//!
//! Every node exposes the same external resource vocabulary regardless of scale.
//! The hierarchy preserves child identity and requires explicit aggregation rules;
//! parents do not implicitly absorb child authority or silently invent capacity.
//!
//! Containment scale and semantic role are intentionally orthogonal. A battery,
//! greenhouse, compute module, or water subsystem can occupy the same structural
//! depth without pretending to be the same kind of asset.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_resource_model::{ResourceAmount, ResourceEnvelope, ResourceKey};
use thiserror::Error;

/// Coarse containment depth used to enforce monotone hierarchy.
///
/// The generic vocabulary (`Component`, `Assembly`, `Zone`, `Facility`) is suitable
/// for mixed civic infrastructure. The compute-oriented names (`Device`, `Rack`,
/// `Hall`, `Site`) are retained as same-rank domain vocabulary for existing callers.
/// Containment must always be checked with [`NodeScale::rank`] / `can_contain`; the
/// derived enum ordering is not a containment relation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NodeScale {
    Device,
    Rack,
    Hall,
    Site,
    Community,
    Region,
    Federation,
    Component,
    Assembly,
    Zone,
    Facility,
}

impl NodeScale {
    pub fn rank(self) -> u8 {
        match self {
            Self::Device | Self::Component => 0,
            Self::Rack | Self::Assembly => 1,
            Self::Hall | Self::Zone => 2,
            Self::Site | Self::Facility => 3,
            Self::Community => 4,
            Self::Region => 5,
            Self::Federation => 6,
        }
    }

    pub fn can_contain(self, child: Self) -> bool {
        self.rank() > child.rank()
    }
}

/// Semantic function of a node, independent of containment depth.
///
/// Roles are descriptive metadata, not authority. They do not affect parent/child
/// permission, resource conservation, or operating-envelope validity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub enum NodeRole {
    #[default]
    Generic,
    Compute,
    EnergyGeneration,
    EnergyStorage,
    ThermalSource,
    ThermalStorage,
    ThermalConsumer,
    Cooling,
    WaterSystem,
    Network,
    Building,
    Greenhouse,
    Workshop,
    Ecological,
    CommunityService,
    /// Namespaced/open extension point for domain roles not yet standardized.
    Custom(String),
}

/// How a parent may summarize a particular resource dimension across children.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationRule {
    /// Sum child capacities. Suitable for independently additive capacities.
    Sum,
    /// Use the minimum child capacity. Useful when every child must satisfy a bound.
    Minimum,
    /// Use the maximum child capacity. Useful for representative peak capability.
    Maximum,
    /// Do not expose this dimension at the parent boundary.
    Omit,
}

/// Externally visible resource node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceNode {
    pub id: String,
    pub label: String,
    pub scale: NodeScale,
    #[serde(default)]
    pub role: NodeRole,
    pub local_envelope: ResourceEnvelope,
}

impl ResourceNode {
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        scale: NodeScale,
        local_envelope: ResourceEnvelope,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            scale,
            role: NodeRole::Generic,
            local_envelope,
        }
    }

    pub fn with_role(mut self, role: NodeRole) -> Self {
        self.role = role;
        self
    }
}

/// Compact parent-facing summary. Child identities are retained explicitly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeSummary {
    pub node_id: String,
    pub scale: NodeScale,
    pub role: NodeRole,
    pub direct_children: Vec<String>,
    pub envelope: ResourceEnvelope,
}

/// Hierarchy that composes resource nodes while preserving bounded locality.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceHierarchy {
    nodes: BTreeMap<String, ResourceNode>,
    parent_of: BTreeMap<String, String>,
    children_of: BTreeMap<String, BTreeSet<String>>,
    aggregation_rules: BTreeMap<String, BTreeMap<ResourceKey, AggregationRule>>,
}

impl ResourceHierarchy {
    pub fn insert_root(&mut self, node: ResourceNode) -> Result<(), HierarchyError> {
        if self.nodes.contains_key(&node.id) {
            return Err(HierarchyError::DuplicateNode(node.id));
        }
        self.children_of.entry(node.id.clone()).or_default();
        self.nodes.insert(node.id.clone(), node);
        Ok(())
    }

    /// Insert a child beneath a strictly larger-scale parent.
    ///
    /// A child may have only one structural parent. Peer/federation relationships
    /// belong in higher protocol layers rather than being smuggled into containment.
    pub fn insert_child(
        &mut self,
        parent_id: &str,
        child: ResourceNode,
    ) -> Result<(), HierarchyError> {
        let parent = self
            .nodes
            .get(parent_id)
            .ok_or_else(|| HierarchyError::UnknownNode(parent_id.to_owned()))?;
        if self.nodes.contains_key(&child.id) {
            return Err(HierarchyError::DuplicateNode(child.id));
        }
        if !parent.scale.can_contain(child.scale) {
            return Err(HierarchyError::InvalidScaleContainment {
                parent: parent.scale,
                child: child.scale,
            });
        }

        let child_id = child.id.clone();
        self.nodes.insert(child_id.clone(), child);
        self.parent_of
            .insert(child_id.clone(), parent_id.to_owned());
        self.children_of
            .entry(parent_id.to_owned())
            .or_default()
            .insert(child_id.clone());
        self.children_of.entry(child_id).or_default();
        Ok(())
    }

    pub fn set_aggregation_rule(
        &mut self,
        node_id: &str,
        key: ResourceKey,
        rule: AggregationRule,
    ) -> Result<(), HierarchyError> {
        if !self.nodes.contains_key(node_id) {
            return Err(HierarchyError::UnknownNode(node_id.to_owned()));
        }
        self.aggregation_rules
            .entry(node_id.to_owned())
            .or_default()
            .insert(key, rule);
        Ok(())
    }

    pub fn node(&self, id: &str) -> Option<&ResourceNode> {
        self.nodes.get(id)
    }

    pub fn parent(&self, id: &str) -> Option<&str> {
        self.parent_of.get(id).map(String::as_str)
    }

    pub fn children(&self, id: &str) -> impl Iterator<Item = &str> {
        self.children_of
            .get(id)
            .into_iter()
            .flat_map(|children| children.iter().map(String::as_str))
    }

    /// Summarize only direct children plus the node's own local envelope.
    ///
    /// Recursive aggregation is explicit: callers can summarize lower layers first
    /// and materialize their exported capacity into the child's local envelope if
    /// desired. This prevents hidden double counting across nested boundaries.
    pub fn summarize(&self, node_id: &str) -> Result<NodeSummary, HierarchyError> {
        let node = self
            .nodes
            .get(node_id)
            .ok_or_else(|| HierarchyError::UnknownNode(node_id.to_owned()))?;
        let children: Vec<&ResourceNode> = self
            .children(node_id)
            .filter_map(|child_id| self.nodes.get(child_id))
            .collect();

        let mut result = node.local_envelope.clone();
        let rules = self.aggregation_rules.get(node_id);

        if let Some(rules) = rules {
            for (&key, &rule) in rules {
                if rule == AggregationRule::Omit {
                    continue;
                }
                let values: Vec<f64> = children
                    .iter()
                    .map(|child| child.local_envelope.capacity(key))
                    .collect();
                if values.is_empty() {
                    continue;
                }
                let aggregate = match rule {
                    AggregationRule::Sum => values.iter().sum(),
                    AggregationRule::Minimum => values.iter().copied().fold(f64::INFINITY, f64::min),
                    AggregationRule::Maximum => values.iter().copied().fold(0.0, f64::max),
                    AggregationRule::Omit => unreachable!(),
                };
                if aggregate > 0.0 {
                    let amount = ResourceAmount {
                        key,
                        value: aggregate,
                    };
                    result.insert(amount);
                }
            }
        }

        Ok(NodeSummary {
            node_id: node.id.clone(),
            scale: node.scale,
            role: node.role.clone(),
            direct_children: children.iter().map(|child| child.id.clone()).collect(),
            envelope: result,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum HierarchyError {
    #[error("duplicate resource node {0}")]
    DuplicateNode(String),
    #[error("unknown resource node {0}")]
    UnknownNode(String),
    #[error("parent scale {parent:?} cannot contain child scale {child:?}")]
    InvalidScaleContainment {
        parent: NodeScale,
        child: NodeScale,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_resource_model::{ResourceKind, ResourceUnit};

    fn watts(value: f64) -> ResourceAmount {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, value).unwrap()
    }

    fn gpu_seconds(value: f64) -> ResourceAmount {
        ResourceAmount::new(ResourceKind::Compute, ResourceUnit::GpuSecond, value).unwrap()
    }

    fn envelope(amounts: impl IntoIterator<Item = ResourceAmount>) -> ResourceEnvelope {
        let mut envelope = ResourceEnvelope::default();
        for amount in amounts {
            envelope.insert(amount);
        }
        envelope
    }

    #[test]
    fn compute_and_generic_scale_names_share_containment_ranks() {
        assert_eq!(NodeScale::Device.rank(), NodeScale::Component.rank());
        assert_eq!(NodeScale::Rack.rank(), NodeScale::Assembly.rank());
        assert_eq!(NodeScale::Hall.rank(), NodeScale::Zone.rank());
        assert_eq!(NodeScale::Site.rank(), NodeScale::Facility.rank());
        assert!(NodeScale::Facility.can_contain(NodeScale::Assembly));
        assert!(NodeScale::Site.can_contain(NodeScale::Component));
        assert!(!NodeScale::Facility.can_contain(NodeScale::Site));
    }

    #[test]
    fn scale_containment_is_strictly_monotone() {
        assert!(NodeScale::Site.can_contain(NodeScale::Rack));
        assert!(!NodeScale::Rack.can_contain(NodeScale::Site));
        assert!(!NodeScale::Site.can_contain(NodeScale::Site));
    }

    #[test]
    fn semantic_role_is_independent_of_scale() {
        let battery = ResourceNode::new(
            "battery",
            "Battery",
            NodeScale::Component,
            ResourceEnvelope::default(),
        )
        .with_role(NodeRole::EnergyStorage);
        let greenhouse = ResourceNode::new(
            "greenhouse",
            "Greenhouse",
            NodeScale::Facility,
            ResourceEnvelope::default(),
        )
        .with_role(NodeRole::Greenhouse);
        assert_eq!(battery.scale, NodeScale::Component);
        assert_eq!(battery.role, NodeRole::EnergyStorage);
        assert_eq!(greenhouse.scale, NodeScale::Facility);
        assert_eq!(greenhouse.role, NodeRole::Greenhouse);
    }

    #[test]
    fn child_identity_parent_boundary_and_role_are_preserved() {
        let mut hierarchy = ResourceHierarchy::default();
        hierarchy
            .insert_root(
                ResourceNode::new(
                    "site-a",
                    "Site A",
                    NodeScale::Site,
                    ResourceEnvelope::default(),
                )
                .with_role(NodeRole::Compute),
            )
            .unwrap();
        hierarchy
            .insert_child(
                "site-a",
                ResourceNode::new(
                    "rack-1",
                    "Rack 1",
                    NodeScale::Rack,
                    envelope([watts(40.0)]),
                )
                .with_role(NodeRole::Compute),
            )
            .unwrap();

        assert_eq!(hierarchy.parent("rack-1"), Some("site-a"));
        assert_eq!(hierarchy.children("site-a").collect::<Vec<_>>(), vec!["rack-1"]);
        assert_eq!(hierarchy.node("rack-1").unwrap().label, "Rack 1");
        let summary = hierarchy.summarize("site-a").unwrap();
        assert_eq!(summary.role, NodeRole::Compute);
    }

    #[test]
    fn invalid_reverse_containment_is_rejected() {
        let mut hierarchy = ResourceHierarchy::default();
        hierarchy
            .insert_root(ResourceNode::new(
                "rack",
                "Rack",
                NodeScale::Rack,
                ResourceEnvelope::default(),
            ))
            .unwrap();
        let result = hierarchy.insert_child(
            "rack",
            ResourceNode::new(
                "site",
                "Site",
                NodeScale::Site,
                ResourceEnvelope::default(),
            ),
        );
        assert!(matches!(
            result,
            Err(HierarchyError::InvalidScaleContainment { .. })
        ));
    }

    #[test]
    fn parent_does_not_aggregate_without_explicit_rule() {
        let mut hierarchy = ResourceHierarchy::default();
        hierarchy
            .insert_root(ResourceNode::new(
                "site",
                "Site",
                NodeScale::Site,
                ResourceEnvelope::default(),
            ))
            .unwrap();
        hierarchy
            .insert_child(
                "site",
                ResourceNode::new(
                    "rack-a",
                    "A",
                    NodeScale::Rack,
                    envelope([gpu_seconds(10.0)]),
                ),
            )
            .unwrap();

        let summary = hierarchy.summarize("site").unwrap();
        assert_eq!(summary.envelope.capacity(gpu_seconds(0.0).key), 0.0);
    }

    #[test]
    fn explicit_sum_rule_aggregates_additive_child_capacity() {
        let mut hierarchy = ResourceHierarchy::default();
        hierarchy
            .insert_root(ResourceNode::new(
                "site",
                "Site",
                NodeScale::Site,
                envelope([watts(5.0)]),
            ))
            .unwrap();
        hierarchy
            .insert_child(
                "site",
                ResourceNode::new(
                    "rack-a",
                    "A",
                    NodeScale::Rack,
                    envelope([watts(40.0), gpu_seconds(10.0)]),
                ),
            )
            .unwrap();
        hierarchy
            .insert_child(
                "site",
                ResourceNode::new(
                    "rack-b",
                    "B",
                    NodeScale::Rack,
                    envelope([watts(60.0), gpu_seconds(20.0)]),
                ),
            )
            .unwrap();

        hierarchy
            .set_aggregation_rule("site", watts(0.0).key, AggregationRule::Sum)
            .unwrap();
        hierarchy
            .set_aggregation_rule("site", gpu_seconds(0.0).key, AggregationRule::Sum)
            .unwrap();

        let summary = hierarchy.summarize("site").unwrap();
        assert_eq!(summary.envelope.capacity(watts(0.0).key), 105.0);
        assert_eq!(summary.envelope.capacity(gpu_seconds(0.0).key), 30.0);
        assert_eq!(summary.direct_children, vec!["rack-a", "rack-b"]);
    }

    #[test]
    fn minimum_rule_can_represent_all_children_bottleneck() {
        let mut hierarchy = ResourceHierarchy::default();
        hierarchy
            .insert_root(ResourceNode::new(
                "site",
                "Site",
                NodeScale::Site,
                ResourceEnvelope::default(),
            ))
            .unwrap();
        hierarchy
            .insert_child(
                "site",
                ResourceNode::new(
                    "rack-a",
                    "A",
                    NodeScale::Rack,
                    envelope([watts(40.0)]),
                ),
            )
            .unwrap();
        hierarchy
            .insert_child(
                "site",
                ResourceNode::new(
                    "rack-b",
                    "B",
                    NodeScale::Rack,
                    envelope([watts(60.0)]),
                ),
            )
            .unwrap();
        hierarchy
            .set_aggregation_rule("site", watts(0.0).key, AggregationRule::Minimum)
            .unwrap();

        assert_eq!(
            hierarchy
                .summarize("site")
                .unwrap()
                .envelope
                .capacity(watts(0.0).key),
            40.0
        );
    }
}
