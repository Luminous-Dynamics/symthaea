// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit nested-world context so simulations cannot silently overwrite their parent reality.

use serde::{Deserialize, Serialize};

use crate::types::{RealityTypeError, WorldDescriptor};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealityContextStack {
    max_depth: u32,
    stack: Vec<WorldDescriptor>,
}

impl RealityContextStack {
    pub fn new(root: WorldDescriptor, max_depth: u32) -> Result<Self, RealityContextError> {
        root.validate().map_err(RealityContextError::Type)?;
        if root.parent.is_some() || root.generation_depth != 0 {
            return Err(RealityContextError::RootMustBeRoot);
        }
        if max_depth == 0 {
            return Err(RealityContextError::ZeroMaxDepth);
        }
        Ok(Self {
            max_depth,
            stack: vec![root],
        })
    }

    pub fn current(&self) -> &WorldDescriptor {
        self.stack
            .last()
            .expect("RealityContextStack always contains its root world")
    }

    pub fn root(&self) -> &WorldDescriptor {
        &self.stack[0]
    }

    pub fn depth(&self) -> u32 {
        (self.stack.len() - 1) as u32
    }

    pub fn lineage(&self) -> &[WorldDescriptor] {
        &self.stack
    }

    pub fn enter_child(&mut self, child: WorldDescriptor) -> Result<(), RealityContextError> {
        child.validate().map_err(RealityContextError::Type)?;
        let parent = child
            .parent
            .as_ref()
            .ok_or(RealityContextError::ChildMissingParent)?;
        let current = self.current();
        if parent.world_id != current.world_id || parent.lineage_id != current.lineage_id {
            return Err(RealityContextError::ParentDoesNotMatchCurrentWorld);
        }
        let expected_depth = current
            .generation_depth
            .checked_add(1)
            .ok_or(RealityContextError::DepthOverflow)?;
        if child.generation_depth != expected_depth {
            return Err(RealityContextError::GenerationDepthMismatch {
                expected: expected_depth,
                actual: child.generation_depth,
            });
        }
        if child.generation_depth > self.max_depth {
            return Err(RealityContextError::MaximumDepthExceeded {
                maximum: self.max_depth,
                requested: child.generation_depth,
            });
        }
        if self.stack.iter().any(|world| {
            world.world_id == child.world_id && world.lineage_id == child.lineage_id
        }) {
            return Err(RealityContextError::WorldCycle);
        }
        self.stack.push(child);
        Ok(())
    }

    pub fn leave_current(&mut self) -> Result<WorldDescriptor, RealityContextError> {
        if self.stack.len() == 1 {
            return Err(RealityContextError::CannotLeaveRoot);
        }
        Ok(self.stack.pop().expect("length checked above"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RealityContextError {
    #[error("reality type error: {0}")]
    Type(#[from] RealityTypeError),
    #[error("maximum nested reality depth must be nonzero")]
    ZeroMaxDepth,
    #[error("context stack root descriptor must be a root world")]
    RootMustBeRoot,
    #[error("child world is missing its parent reference")]
    ChildMissingParent,
    #[error("child parent does not match the currently inhabited world")]
    ParentDoesNotMatchCurrentWorld,
    #[error("generation depth mismatch: expected {expected}, got {actual}")]
    GenerationDepthMismatch { expected: u32, actual: u32 },
    #[error("nested reality depth overflow")]
    DepthOverflow,
    #[error("maximum nested reality depth {maximum} exceeded by {requested}")]
    MaximumDepthExceeded { maximum: u32, requested: u32 },
    #[error("world lineage would cycle back into an already active world")]
    WorldCycle,
    #[error("cannot leave the root world")]
    CannotLeaveRoot,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        RealityLayer, WorldId, WorldLineageId, WorldOrigin, WorldParentRef, WorldRelation,
    };

    fn root() -> WorldDescriptor {
        WorldDescriptor {
            world_id: WorldId("studio".into()),
            lineage_id: WorldLineageId("lineage".into()),
            layer: RealityLayer::DigitalCommitted,
            origin: WorldOrigin::DigitalHost { host_kind: "bevy".into() },
            parent: None,
            generation_depth: 0,
            creator_id: "host".into(),
        }
    }

    fn ghost(id: &str, parent: &WorldDescriptor, depth: u32) -> WorldDescriptor {
        WorldDescriptor {
            world_id: WorldId(id.into()),
            lineage_id: WorldLineageId(format!("lineage-{id}")),
            layer: RealityLayer::Counterfactual,
            origin: WorldOrigin::CounterfactualBranch,
            parent: Some(WorldParentRef {
                world_id: parent.world_id.clone(),
                lineage_id: parent.lineage_id.clone(),
                relation: WorldRelation::CounterfactualOf,
            }),
            generation_depth: depth,
            creator_id: "counterfactual-engine".into(),
        }
    }

    #[test]
    fn nested_counterfactuals_preserve_explicit_parentage() {
        let root = root();
        let mut stack = RealityContextStack::new(root.clone(), 3).unwrap();
        let a = ghost("a", &root, 1);
        stack.enter_child(a.clone()).unwrap();
        stack.enter_child(ghost("b", &a, 2)).unwrap();
        assert_eq!(stack.depth(), 2);
        assert_eq!(stack.current().world_id.0, "b");
        stack.leave_current().unwrap();
        assert_eq!(stack.current().world_id.0, "a");
    }

    #[test]
    fn cannot_enter_child_of_some_other_world() {
        let root = root();
        let mut stack = RealityContextStack::new(root.clone(), 3).unwrap();
        let mut child = ghost("wrong", &root, 1);
        child.parent.as_mut().unwrap().world_id = WorldId("other".into());
        assert_eq!(
            stack.enter_child(child),
            Err(RealityContextError::ParentDoesNotMatchCurrentWorld)
        );
    }
}
