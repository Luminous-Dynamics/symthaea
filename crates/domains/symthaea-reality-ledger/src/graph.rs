// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Persistent graph of worlds that have existed, independent of current context.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::types::{WorldDescriptor, WorldId, WorldLineageId};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WorldKey {
    pub world_id: WorldId,
    pub lineage_id: WorldLineageId,
}

impl From<&WorldDescriptor> for WorldKey {
    fn from(world: &WorldDescriptor) -> Self {
        Self {
            world_id: world.world_id.clone(),
            lineage_id: world.lineage_id.clone(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldGraph {
    worlds: BTreeMap<WorldKey, WorldDescriptor>,
}

impl WorldGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.worlds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.worlds.is_empty()
    }

    pub fn get(&self, key: &WorldKey) -> Option<&WorldDescriptor> {
        self.worlds.get(key)
    }

    pub fn worlds(&self) -> impl Iterator<Item = &WorldDescriptor> {
        self.worlds.values()
    }

    pub fn insert(&mut self, world: WorldDescriptor) -> Result<(), WorldGraphError> {
        world
            .validate()
            .map_err(|error| WorldGraphError::InvalidWorld(error.to_string()))?;
        let key = WorldKey::from(&world);
        if self.worlds.contains_key(&key) {
            return Err(WorldGraphError::DuplicateWorld);
        }

        if let Some(parent) = &world.parent {
            let parent_key = WorldKey {
                world_id: parent.world_id.clone(),
                lineage_id: parent.lineage_id.clone(),
            };
            let parent_world = self
                .worlds
                .get(&parent_key)
                .ok_or(WorldGraphError::MissingParent)?;
            if world.generation_depth != parent_world.generation_depth + 1 {
                return Err(WorldGraphError::GenerationDepthMismatch);
            }
        } else if !self.worlds.is_empty() {
            // Multiple independent roots are allowed, but every root must be an
            // actual root. WorldDescriptor::validate already enforces depth 0.
        }

        self.worlds.insert(key, world);
        self.verify_acyclic()?;
        Ok(())
    }

    pub fn children_of<'a>(&'a self, key: &'a WorldKey) -> impl Iterator<Item = &'a WorldDescriptor> {
        self.worlds.values().filter(move |world| {
            world.parent.as_ref().is_some_and(|parent| {
                parent.world_id == key.world_id && parent.lineage_id == key.lineage_id
            })
        })
    }

    pub fn verify(&self) -> Result<(), WorldGraphError> {
        for world in self.worlds.values() {
            world
                .validate()
                .map_err(|error| WorldGraphError::InvalidWorld(error.to_string()))?;
            if let Some(parent) = &world.parent {
                let parent_key = WorldKey {
                    world_id: parent.world_id.clone(),
                    lineage_id: parent.lineage_id.clone(),
                };
                let parent_world = self
                    .worlds
                    .get(&parent_key)
                    .ok_or(WorldGraphError::MissingParent)?;
                if world.generation_depth != parent_world.generation_depth + 1 {
                    return Err(WorldGraphError::GenerationDepthMismatch);
                }
            }
        }
        self.verify_acyclic()
    }

    fn verify_acyclic(&self) -> Result<(), WorldGraphError> {
        for start in self.worlds.keys() {
            let mut cursor = Some(start.clone());
            let mut seen = BTreeMap::<WorldKey, ()>::new();
            while let Some(key) = cursor {
                if seen.insert(key.clone(), ()).is_some() {
                    return Err(WorldGraphError::CycleDetected);
                }
                let Some(world) = self.worlds.get(&key) else {
                    break;
                };
                cursor = world.parent.as_ref().map(|parent| WorldKey {
                    world_id: parent.world_id.clone(),
                    lineage_id: parent.lineage_id.clone(),
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorldGraphError {
    #[error("invalid world descriptor: {0}")]
    InvalidWorld(String),
    #[error("world graph already contains this world/lineage identity")]
    DuplicateWorld,
    #[error("derived world parent is not present in the graph")]
    MissingParent,
    #[error("derived world generation depth does not equal parent depth + 1")]
    GenerationDepthMismatch,
    #[error("world graph contains a parent cycle")]
    CycleDetected,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{RealityLayer, WorldOrigin, WorldParentRef, WorldRelation};

    fn root() -> WorldDescriptor {
        WorldDescriptor {
            world_id: WorldId("studio".into()),
            lineage_id: WorldLineageId("studio-lineage".into()),
            layer: RealityLayer::DigitalCommitted,
            origin: WorldOrigin::DigitalHost { host_kind: "bevy".into() },
            parent: None,
            generation_depth: 0,
            creator_id: "host".into(),
        }
    }

    #[test]
    fn sibling_counterfactuals_survive_after_context_exit() {
        let root = root();
        let mut graph = WorldGraph::new();
        graph.insert(root.clone()).unwrap();
        for id in ["ghost-a", "ghost-b"] {
            graph
                .insert(WorldDescriptor {
                    world_id: WorldId(id.into()),
                    lineage_id: WorldLineageId(format!("{id}-lineage")),
                    layer: RealityLayer::Counterfactual,
                    origin: WorldOrigin::CounterfactualBranch,
                    parent: Some(WorldParentRef {
                        world_id: root.world_id.clone(),
                        lineage_id: root.lineage_id.clone(),
                        relation: WorldRelation::CounterfactualOf,
                    }),
                    generation_depth: 1,
                    creator_id: "ghost-engine".into(),
                })
                .unwrap();
        }
        assert_eq!(graph.len(), 3);
        assert_eq!(graph.children_of(&WorldKey::from(&root)).count(), 2);
        graph.verify().unwrap();
    }
}
