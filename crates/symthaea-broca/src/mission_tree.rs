// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mission Tree — Hierarchical decomposition of complex strategic goals.
//!
//! Allows Broca to manage 'Mission Trees' where a global intent is recursively
//! broken down into verifiable sub-missions.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MissionStatus {
    Pending,
    Dreaming,
    Foraging,
    Verifying,
    Resolved,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionNode {
    pub id: String,
    pub intent: ContinuousHV,
    pub status: MissionStatus,
    pub children: Vec<MissionNode>,
    pub coherence: f32,
}

impl MissionNode {
    pub fn new(id: &str, intent: ContinuousHV) -> Self {
        Self {
            id: id.to_string(),
            intent,
            status: MissionStatus::Pending,
            children: Vec::new(),
            coherence: 0.0,
        }
    }

    /// Recursively check if the entire subtree is resolved.
    pub fn is_resolved(&self) -> bool {
        if self.status != MissionStatus::Resolved {
            return false;
        }
        self.children.iter().all(|c| c.is_resolved())
    }
}

pub struct MissionTree {
    pub root: MissionNode,
}

impl MissionTree {
    pub fn new(id: &str, global_intent: ContinuousHV) -> Self {
        Self {
            root: MissionNode::new(id, global_intent),
        }
    }

    /// Find the next 'Pending' leaf node to work on.
    pub fn next_task(&mut self) -> Option<&mut MissionNode> {
        Self::find_pending(&mut self.root)
    }

    fn find_pending(node: &mut MissionNode) -> Option<&mut MissionNode> {
        if node.status == MissionStatus::Pending && node.children.is_empty() {
            return Some(node);
        }
        for child in &mut node.children {
            if let Some(pending) = Self::find_pending(child) {
                return Some(pending);
            }
        }
        None
    }
}
