// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic relay-topology and surface-reachability assessment.
//!
//! Link quality in the plant is a local scalar; team operations require an
//! explicit graph. The mesh retains bounded ordered link updates and computes
//! a widest path, maximizing the weakest link to surface or a peer. Stale
//! links are excluded rather than treated as weak but current evidence.

use crate::team::AgentId;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const DEFAULT_MESH_LINK_CAPACITY: usize = 128;
pub const DEFAULT_MESH_STALE_STEPS: u64 = 600;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MeshNodeId {
    Surface,
    Agent(AgentId),
    Relay(u16),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeshLink {
    pub first: MeshNodeId,
    pub second: MeshNodeId,
    pub epoch: u32,
    pub sequence: u64,
    pub observed_step: u64,
    pub quality: f64,
    pub capacity_ratio: f64,
}

impl MeshLink {
    pub fn is_valid(self) -> bool {
        self.first != self.second
            && self.quality.is_finite()
            && (0.0..=1.0).contains(&self.quality)
            && self.capacity_ratio.is_finite()
            && (0.0..=1.0).contains(&self.capacity_ratio)
    }

    pub fn canonical_key(self) -> (MeshNodeId, MeshNodeId) {
        if self.first <= self.second {
            (self.first, self.second)
        } else {
            (self.second, self.first)
        }
    }

    fn version_is_newer_than(self, other: Self) -> bool {
        self.epoch > other.epoch || (self.epoch == other.epoch && self.sequence > other.sequence)
    }

    pub fn effective_quality(self) -> f64 {
        self.quality.min(self.capacity_ratio)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshLinkRejection {
    Invalid,
    Replay,
    Equivocation,
    Capacity,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeshAssessment {
    pub reachable: bool,
    pub bottleneck_quality: f64,
    pub hops: usize,
    pub fresh_links_considered: usize,
    pub stale_links_ignored: usize,
}

impl MeshAssessment {
    pub const fn unreachable() -> Self {
        Self {
            reachable: false,
            bottleneck_quality: 0.0,
            hops: 0,
            fresh_links_considered: 0,
            stale_links_ignored: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelayMesh {
    capacity: usize,
    stale_after_steps: u64,
    links: BTreeMap<(MeshNodeId, MeshNodeId), MeshLink>,
}

impl RelayMesh {
    pub fn new(capacity: usize, stale_after_steps: u64) -> Self {
        Self {
            capacity: capacity.max(1),
            stale_after_steps: stale_after_steps.max(1),
            links: BTreeMap::new(),
        }
    }

    pub fn merge(&mut self, link: MeshLink) -> Result<(), MeshLinkRejection> {
        if !link.is_valid() {
            return Err(MeshLinkRejection::Invalid);
        }
        let key = link.canonical_key();
        if let Some(existing) = self.links.get(&key).copied() {
            if link == existing {
                return Err(MeshLinkRejection::Replay);
            }
            if link.epoch == existing.epoch && link.sequence == existing.sequence {
                return Err(MeshLinkRejection::Equivocation);
            }
            if !link.version_is_newer_than(existing) {
                return Err(MeshLinkRejection::Replay);
            }
        } else if self.links.len() >= self.capacity {
            return Err(MeshLinkRejection::Capacity);
        }
        self.links.insert(key, link);
        Ok(())
    }

    pub fn assess(
        &self,
        source: MeshNodeId,
        destination: MeshNodeId,
        current_step: u64,
    ) -> MeshAssessment {
        if source == destination {
            return MeshAssessment {
                reachable: true,
                bottleneck_quality: 1.0,
                hops: 0,
                fresh_links_considered: 0,
                stale_links_ignored: 0,
            };
        }
        let stale_links_ignored = self
            .links
            .values()
            .filter(|link| current_step.saturating_sub(link.observed_step) > self.stale_after_steps)
            .count();
        let fresh: Vec<_> = self
            .links
            .values()
            .copied()
            .filter(|link| {
                current_step.saturating_sub(link.observed_step) <= self.stale_after_steps
            })
            .collect();
        let mut nodes = BTreeSet::new();
        nodes.insert(source);
        nodes.insert(destination);
        for link in &fresh {
            nodes.insert(link.first);
            nodes.insert(link.second);
        }
        let mut quality = BTreeMap::new();
        let mut hops = BTreeMap::new();
        let mut visited = BTreeSet::new();
        for node in nodes.iter().copied() {
            quality.insert(node, 0.0f64);
            hops.insert(node, usize::MAX);
        }
        quality.insert(source, 1.0);
        hops.insert(source, 0);

        loop {
            let next = nodes
                .iter()
                .copied()
                .filter(|node| !visited.contains(node))
                .max_by(|left, right| {
                    quality
                        .get(left)
                        .copied()
                        .unwrap_or(0.0)
                        .total_cmp(&quality.get(right).copied().unwrap_or(0.0))
                        .then_with(|| right.cmp(left))
                });
            let Some(node) = next else {
                break;
            };
            let node_quality = quality.get(&node).copied().unwrap_or(0.0);
            if node_quality <= 0.0 {
                break;
            }
            visited.insert(node);
            if node == destination {
                break;
            }
            let node_hops = hops.get(&node).copied().unwrap_or(usize::MAX);
            for link in &fresh {
                let neighbor = if link.first == node {
                    Some(link.second)
                } else if link.second == node {
                    Some(link.first)
                } else {
                    None
                };
                let Some(neighbor) = neighbor else {
                    continue;
                };
                let candidate = node_quality.min(link.effective_quality());
                let candidate_hops = node_hops.saturating_add(1);
                let existing = quality.get(&neighbor).copied().unwrap_or(0.0);
                let existing_hops = hops.get(&neighbor).copied().unwrap_or(usize::MAX);
                if candidate > existing || (candidate == existing && candidate_hops < existing_hops)
                {
                    quality.insert(neighbor, candidate);
                    hops.insert(neighbor, candidate_hops);
                }
            }
        }
        let bottleneck_quality = quality.get(&destination).copied().unwrap_or(0.0);
        MeshAssessment {
            reachable: bottleneck_quality > 0.0,
            bottleneck_quality,
            hops: if bottleneck_quality > 0.0 {
                hops.get(&destination).copied().unwrap_or(0)
            } else {
                0
            },
            fresh_links_considered: fresh.len(),
            stale_links_ignored,
        }
    }

    pub fn assess_surface(&self, local_agent: AgentId, current_step: u64) -> MeshAssessment {
        self.assess(
            MeshNodeId::Agent(local_agent),
            MeshNodeId::Surface,
            current_step,
        )
    }

    pub fn link_count(&self) -> usize {
        self.links.len()
    }

    pub fn clear(&mut self) {
        self.links.clear();
    }
}

impl Default for RelayMesh {
    fn default() -> Self {
        Self::new(DEFAULT_MESH_LINK_CAPACITY, DEFAULT_MESH_STALE_STEPS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link(first: MeshNodeId, second: MeshNodeId, quality: f64) -> MeshLink {
        MeshLink {
            first,
            second,
            epoch: 1,
            sequence: 1,
            observed_step: 10,
            quality,
            capacity_ratio: 1.0,
        }
    }

    #[test]
    fn widest_path_prefers_stronger_multihop_route() {
        let agent = MeshNodeId::Agent(AgentId::new(2));
        let mut mesh = RelayMesh::default();
        assert_eq!(mesh.merge(link(agent, MeshNodeId::Surface, 0.2)), Ok(()));
        assert_eq!(mesh.merge(link(agent, MeshNodeId::Relay(1), 0.8)), Ok(()));
        assert_eq!(
            mesh.merge(link(MeshNodeId::Relay(1), MeshNodeId::Surface, 0.7)),
            Ok(())
        );
        let assessment = mesh.assess(agent, MeshNodeId::Surface, 20);
        assert!(assessment.reachable);
        assert_eq!(assessment.hops, 2);
        assert!((assessment.bottleneck_quality - 0.7).abs() < 1e-9);
    }

    #[test]
    fn stale_links_do_not_create_false_reachability() {
        let agent = MeshNodeId::Agent(AgentId::new(2));
        let mut mesh = RelayMesh::new(4, 5);
        assert_eq!(mesh.merge(link(agent, MeshNodeId::Surface, 0.9)), Ok(()));
        let assessment = mesh.assess(agent, MeshNodeId::Surface, 16);
        assert!(!assessment.reachable);
        assert_eq!(assessment.stale_links_ignored, 1);
    }

    #[test]
    fn equal_version_conflicts_are_rejected() {
        let agent = MeshNodeId::Agent(AgentId::new(2));
        let mut mesh = RelayMesh::default();
        let first = link(agent, MeshNodeId::Surface, 0.9);
        let mut conflict = first;
        conflict.quality = 0.1;
        assert_eq!(mesh.merge(first), Ok(()));
        assert_eq!(mesh.merge(conflict), Err(MeshLinkRejection::Equivocation));
    }
}
