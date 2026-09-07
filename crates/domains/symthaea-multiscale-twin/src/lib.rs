// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conservative hierarchical aggregation for engineered digital twins.
//!
//! This crate binds existing `TwinState` values to resource-hierarchy nodes and
//! produces parent-facing summaries without forwarding raw child telemetry.
//! Uncertainty aggregation is intentionally conservative: a parent summary may
//! never report less epistemic or aleatoric uncertainty than its most uncertain
//! direct child.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use symthaea_digital_twin::TwinState;
use symthaea_resource_hierarchy::{NodeScale, ResourceHierarchy};
use thiserror::Error;

/// Reduction rule for bounded scalar metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricRule {
    Minimum,
    Maximum,
    Mean,
}

/// Explicit policy for parent-facing twin aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TwinAggregationPolicy {
    pub health: MetricRule,
    pub free_energy: MetricRule,
}

impl Default for TwinAggregationPolicy {
    fn default() -> Self {
        Self {
            health: MetricRule::Minimum,
            free_energy: MetricRule::Maximum,
        }
    }
}

/// Compact child record retained in a composite summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChildTwinSummary {
    pub node_id: String,
    pub twin_id: String,
    pub health: f64,
    pub free_energy: f64,
    pub epistemic_uncertainty: f64,
    pub aleatoric_uncertainty: f64,
}

/// Parent-facing twin state that deliberately excludes raw child telemetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompositeTwinSummary {
    pub node_id: String,
    pub scale: NodeScale,
    pub health: f64,
    pub free_energy: f64,
    pub epistemic_uncertainty: f64,
    pub aleatoric_uncertainty: f64,
    pub children: Vec<ChildTwinSummary>,
}

impl CompositeTwinSummary {
    pub fn child_count(&self) -> usize {
        self.children.len()
    }
}

/// Binds live/recorded twins to nodes without changing containment authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiscaleTwinRegistry {
    hierarchy: ResourceHierarchy,
    twins_by_node: BTreeMap<String, TwinState>,
}

impl MultiscaleTwinRegistry {
    pub fn new(hierarchy: ResourceHierarchy) -> Self {
        Self {
            hierarchy,
            twins_by_node: BTreeMap::new(),
        }
    }

    pub fn hierarchy(&self) -> &ResourceHierarchy {
        &self.hierarchy
    }

    pub fn bind_twin(&mut self, node_id: &str, twin: TwinState) -> Result<(), TwinHierarchyError> {
        if self.hierarchy.node(node_id).is_none() {
            return Err(TwinHierarchyError::UnknownNode(node_id.to_owned()));
        }
        validate_twin(&twin)?;
        self.twins_by_node.insert(node_id.to_owned(), twin);
        Ok(())
    }

    pub fn twin(&self, node_id: &str) -> Option<&TwinState> {
        self.twins_by_node.get(node_id)
    }

    /// Aggregate direct child twins into a parent-facing summary.
    ///
    /// Raw telemetry and prediction-residual histories are not copied upward.
    /// Child identity and scalar uncertainty are preserved explicitly.
    pub fn summarize_node(
        &self,
        node_id: &str,
        policy: TwinAggregationPolicy,
    ) -> Result<CompositeTwinSummary, TwinHierarchyError> {
        let node = self
            .hierarchy
            .node(node_id)
            .ok_or_else(|| TwinHierarchyError::UnknownNode(node_id.to_owned()))?;
        let child_ids: Vec<&str> = self.hierarchy.children(node_id).collect();
        if child_ids.is_empty() {
            return Err(TwinHierarchyError::NoChildren(node_id.to_owned()));
        }

        let mut children = Vec::with_capacity(child_ids.len());
        for child_id in child_ids {
            let twin = self
                .twins_by_node
                .get(child_id)
                .ok_or_else(|| TwinHierarchyError::MissingTwin(child_id.to_owned()))?;
            validate_twin(twin)?;
            children.push(ChildTwinSummary {
                node_id: child_id.to_owned(),
                twin_id: twin.id.clone(),
                health: twin.health,
                free_energy: twin.free_energy,
                epistemic_uncertainty: twin.epistemic_uncertainty,
                aleatoric_uncertainty: twin.aleatoric_uncertainty,
            });
        }

        let health_values: Vec<f64> = children.iter().map(|child| child.health).collect();
        let free_energy_values: Vec<f64> = children.iter().map(|child| child.free_energy).collect();
        let health = reduce(&health_values, policy.health);
        let free_energy = reduce(&free_energy_values, policy.free_energy);
        let epistemic_uncertainty = children
            .iter()
            .map(|child| child.epistemic_uncertainty)
            .fold(0.0, f64::max);
        let aleatoric_uncertainty = children
            .iter()
            .map(|child| child.aleatoric_uncertainty)
            .fold(0.0, f64::max);

        Ok(CompositeTwinSummary {
            node_id: node_id.to_owned(),
            scale: node.scale,
            health,
            free_energy,
            epistemic_uncertainty,
            aleatoric_uncertainty,
            children,
        })
    }
}

fn reduce(values: &[f64], rule: MetricRule) -> f64 {
    debug_assert!(!values.is_empty());
    match rule {
        MetricRule::Minimum => values.iter().copied().fold(f64::INFINITY, f64::min),
        MetricRule::Maximum => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        MetricRule::Mean => values.iter().sum::<f64>() / values.len() as f64,
    }
}

fn validate_unit_interval(label: &'static str, value: f64) -> Result<(), TwinHierarchyError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(TwinHierarchyError::InvalidMetric { label, value });
    }
    Ok(())
}

fn validate_twin(twin: &TwinState) -> Result<(), TwinHierarchyError> {
    validate_unit_interval("health", twin.health)?;
    validate_unit_interval("epistemic_uncertainty", twin.epistemic_uncertainty)?;
    validate_unit_interval("aleatoric_uncertainty", twin.aleatoric_uncertainty)?;
    if !twin.free_energy.is_finite() || twin.free_energy < 0.0 {
        return Err(TwinHierarchyError::InvalidMetric {
            label: "free_energy",
            value: twin.free_energy,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum TwinHierarchyError {
    #[error("unknown hierarchy node {0}")]
    UnknownNode(String),
    #[error("node {0} has no direct children to summarize")]
    NoChildren(String),
    #[error("no digital twin is bound to child node {0}")]
    MissingTwin(String),
    #[error("invalid twin metric {label}: {value}")]
    InvalidMetric { label: &'static str, value: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_digital_twin::{AssetClass, TwinState};
    use symthaea_resource_hierarchy::{NodeScale, ResourceNode};
    use symthaea_resource_model::ResourceEnvelope;

    fn hierarchy() -> ResourceHierarchy {
        let mut hierarchy = ResourceHierarchy::default();
        hierarchy
            .insert_root(ResourceNode::new(
                "site",
                "Community Compute Site",
                NodeScale::Site,
                ResourceEnvelope::default(),
            ))
            .unwrap();
        hierarchy
            .insert_child(
                "site",
                ResourceNode::new(
                    "rack-a",
                    "Rack A",
                    NodeScale::Rack,
                    ResourceEnvelope::default(),
                ),
            )
            .unwrap();
        hierarchy
            .insert_child(
                "site",
                ResourceNode::new(
                    "rack-b",
                    "Rack B",
                    NodeScale::Rack,
                    ResourceEnvelope::default(),
                ),
            )
            .unwrap();
        hierarchy
    }

    fn twin(id: &str, health: f64, free_energy: f64, epistemic: f64, aleatoric: f64) -> TwinState {
        let mut twin = TwinState::new(id, AssetClass::SystemOfSystems, id);
        twin.health = health;
        twin.free_energy = free_energy;
        twin.epistemic_uncertainty = epistemic;
        twin.aleatoric_uncertainty = aleatoric;
        twin
    }

    #[test]
    fn binding_requires_a_known_hierarchy_node() {
        let mut registry = MultiscaleTwinRegistry::new(hierarchy());
        let result = registry.bind_twin("unknown", twin("t", 1.0, 0.0, 0.1, 0.1));
        assert!(matches!(result, Err(TwinHierarchyError::UnknownNode(_))));
    }

    #[test]
    fn default_policy_is_conservative_for_health_and_surprise() {
        let mut registry = MultiscaleTwinRegistry::new(hierarchy());
        registry
            .bind_twin("rack-a", twin("twin-a", 0.9, 0.2, 0.1, 0.05))
            .unwrap();
        registry
            .bind_twin("rack-b", twin("twin-b", 0.6, 0.8, 0.3, 0.2))
            .unwrap();

        let summary = registry
            .summarize_node("site", TwinAggregationPolicy::default())
            .unwrap();
        assert_eq!(summary.health, 0.6);
        assert_eq!(summary.free_energy, 0.8);
    }

    #[test]
    fn aggregation_never_launders_child_uncertainty() {
        let mut registry = MultiscaleTwinRegistry::new(hierarchy());
        registry
            .bind_twin("rack-a", twin("twin-a", 0.9, 0.2, 0.72, 0.11))
            .unwrap();
        registry
            .bind_twin("rack-b", twin("twin-b", 0.8, 0.3, 0.21, 0.44))
            .unwrap();

        let summary = registry
            .summarize_node(
                "site",
                TwinAggregationPolicy {
                    health: MetricRule::Mean,
                    free_energy: MetricRule::Mean,
                },
            )
            .unwrap();
        assert_eq!(summary.epistemic_uncertainty, 0.72);
        assert_eq!(summary.aleatoric_uncertainty, 0.44);
    }

    #[test]
    fn parent_summary_retains_child_identity_without_raw_telemetry() {
        let mut registry = MultiscaleTwinRegistry::new(hierarchy());
        registry
            .bind_twin("rack-a", twin("sensor-twin-a", 0.9, 0.2, 0.1, 0.1))
            .unwrap();
        registry
            .bind_twin("rack-b", twin("sensor-twin-b", 0.8, 0.3, 0.2, 0.2))
            .unwrap();

        let summary = registry
            .summarize_node("site", TwinAggregationPolicy::default())
            .unwrap();
        assert_eq!(summary.child_count(), 2);
        assert_eq!(summary.children[0].node_id, "rack-a");
        assert_eq!(summary.children[0].twin_id, "sensor-twin-a");
        assert_eq!(summary.children[1].node_id, "rack-b");
    }

    #[test]
    fn missing_child_twin_blocks_summary() {
        let mut registry = MultiscaleTwinRegistry::new(hierarchy());
        registry
            .bind_twin("rack-a", twin("twin-a", 0.9, 0.2, 0.1, 0.1))
            .unwrap();
        let result = registry.summarize_node("site", TwinAggregationPolicy::default());
        assert!(matches!(result, Err(TwinHierarchyError::MissingTwin(id)) if id == "rack-b"));
    }

    #[test]
    fn invalid_uncertainty_is_rejected_at_binding_boundary() {
        let mut registry = MultiscaleTwinRegistry::new(hierarchy());
        let result = registry.bind_twin("rack-a", twin("bad", 0.9, 0.2, 1.2, 0.1));
        assert!(matches!(
            result,
            Err(TwinHierarchyError::InvalidMetric {
                label: "epistemic_uncertainty",
                ..
            })
        ));
    }
}
