// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Prefrontal Cortex: Executive Functions and Working Memory
//!
//! Simulates prefrontal cortex functionality including:
//! - Working memory maintenance
//! - Executive control and task switching
//! - Decision making and planning
//! - Inhibition and impulse control

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use symthaea_core::hdc::ContinuousHV;

/// Configuration for prefrontal simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefrontalConfig {
    /// Working memory capacity (number of items)
    pub working_memory_capacity: usize,
    /// Embedding dimension
    pub dimension: usize,
    /// Attention decay rate
    pub attention_decay: f32,
    /// Inhibition threshold
    pub inhibition_threshold: f32,
    /// Planning horizon (steps ahead)
    pub planning_horizon: usize,
}

impl Default for PrefrontalConfig {
    fn default() -> Self {
        Self {
            working_memory_capacity: 7,
            dimension: 512,
            attention_decay: 0.1,
            inhibition_threshold: 0.5,
            planning_horizon: 5,
        }
    }
}

/// An item in working memory
#[derive(Debug, Clone)]
pub struct WorkingMemoryItem {
    /// Unique identifier
    pub id: String,
    /// Content embedding
    pub embedding: ContinuousHV,
    /// Current activation level
    pub activation: f32,
    /// Priority weight
    pub priority: f32,
    /// Time step when added
    pub added_at: u64,
    /// Number of times rehearsed
    pub rehearsal_count: u32,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl WorkingMemoryItem {
    /// Create a new working memory item
    pub fn new(id: impl Into<String>, embedding: ContinuousHV) -> Self {
        Self {
            id: id.into(),
            embedding,
            activation: 1.0,
            priority: 1.0,
            added_at: 0,
            rehearsal_count: 0,
            metadata: HashMap::new(),
        }
    }

    /// Decay activation
    pub fn decay(&mut self, rate: f32) {
        self.activation = (self.activation - rate).max(0.0);
    }

    /// Rehearse to boost activation
    pub fn rehearse(&mut self) {
        self.activation = (self.activation + 0.3).min(1.0);
        self.rehearsal_count += 1;
    }
}

/// A planned action in the goal stack
#[derive(Debug, Clone)]
pub struct PlannedAction {
    /// Action identifier
    pub id: String,
    /// Goal this action serves
    pub goal_id: String,
    /// Action embedding
    pub embedding: ContinuousHV,
    /// Expected outcome embedding
    pub expected_outcome: ContinuousHV,
    /// Priority (higher = more urgent)
    pub priority: f32,
    /// Estimated effort
    pub effort: f32,
    /// Dependencies (other action IDs)
    pub dependencies: Vec<String>,
    /// Status
    pub status: ActionStatus,
}

/// Status of a planned action
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionStatus {
    /// Waiting for dependencies
    Pending,
    /// Ready to execute
    Ready,
    /// Currently executing
    InProgress,
    /// Completed successfully
    Completed,
    /// Failed or abandoned
    Failed,
    /// Inhibited (suppressed)
    Inhibited,
}

/// Result of executive decision
#[derive(Debug, Clone)]
pub struct ExecutiveDecision {
    /// Selected action ID
    pub selected_action: Option<String>,
    /// Decision confidence
    pub confidence: f32,
    /// Inhibited actions
    pub inhibited: Vec<String>,
    /// Working memory state used
    pub memory_items_considered: usize,
    /// Reasoning trace
    pub reasoning: Vec<String>,
}

/// The prefrontal cortex system
#[derive(Debug)]
pub struct PrefrontalCortex {
    /// Configuration
    config: PrefrontalConfig,
    /// Working memory contents
    working_memory: VecDeque<WorkingMemoryItem>,
    /// Goal/action stack for planning
    action_stack: Vec<PlannedAction>,
    /// Current focus (item ID)
    current_focus: Option<String>,
    /// Time step counter
    time_step: u64,
    /// Statistics
    stats: PrefrontalStats,
    /// Items evicted with high enough activation to graduate to episodic memory.
    /// Drained by callers via `drain_graduates()`.
    graduation_queue: Vec<WorkingMemoryItem>,
}

/// Statistics for prefrontal operations
#[derive(Debug, Clone, Default)]
pub struct PrefrontalStats {
    /// Total items processed through working memory
    pub items_processed: u64,
    /// Total decisions made
    pub decisions_made: u64,
    /// Total actions inhibited
    pub inhibitions: u64,
    /// Successful task completions
    pub tasks_completed: u64,
    /// Average working memory utilization
    pub avg_memory_utilization: f32,
    /// Total items graduated to episodic memory
    pub graduations: u64,
    /// Total items evicted without graduation (below threshold)
    pub evictions_dropped: u64,
}

/// Minimum activation level for an evicted item to graduate to episodic memory.
/// Items below this threshold are silently dropped; those above are queued for
/// graduation to episodic storage via the MemoryCoordinator.
const GRADUATION_ACTIVATION_THRESHOLD: f32 = 0.3;

impl PrefrontalCortex {
    /// Create a new prefrontal cortex
    pub fn new(config: PrefrontalConfig) -> Self {
        Self {
            config,
            working_memory: VecDeque::new(),
            action_stack: Vec::new(),
            current_focus: None,
            time_step: 0,
            stats: PrefrontalStats::default(),
            graduation_queue: Vec::new(),
        }
    }

    /// Advance time and decay activations.
    ///
    /// Items that decay below the minimum threshold are evicted. If their
    /// activation is above `GRADUATION_ACTIVATION_THRESHOLD`, they are queued
    /// for graduation to episodic memory rather than being silently dropped.
    pub fn tick(&mut self) {
        self.time_step += 1;

        // Decay working memory activations
        for item in self.working_memory.iter_mut() {
            item.decay(self.config.attention_decay);
        }

        // Partition: items below minimum activation are evicted
        let mut kept = VecDeque::with_capacity(self.working_memory.len());
        for item in self.working_memory.drain(..) {
            if item.activation > 0.01 {
                kept.push_back(item);
            } else if item.activation >= GRADUATION_ACTIVATION_THRESHOLD || item.rehearsal_count > 0
            {
                // High enough residual activation or rehearsed → graduate
                self.stats.graduations += 1;
                self.graduation_queue.push(item);
            } else {
                self.stats.evictions_dropped += 1;
            }
        }
        self.working_memory = kept;

        // Update average memory utilization
        let utilization =
            self.working_memory.len() as f32 / self.config.working_memory_capacity as f32;
        let n = (self.time_step as f32).max(1.0);
        self.stats.avg_memory_utilization =
            (self.stats.avg_memory_utilization * (n - 1.0) + utilization) / n;
    }

    /// Add item to working memory.
    ///
    /// When at capacity, the lowest-activation item is evicted. Evicted items
    /// with activation above `GRADUATION_ACTIVATION_THRESHOLD` are queued for
    /// graduation to episodic memory.
    pub fn add_to_memory(&mut self, mut item: WorkingMemoryItem) -> bool {
        // Check if already in memory
        if self.working_memory.iter().any(|i| i.id == item.id) {
            // Rehearse existing item
            if let Some(existing) = self.working_memory.iter_mut().find(|i| i.id == item.id) {
                existing.rehearse();
            }
            return true;
        }

        // Check capacity
        if self.working_memory.len() >= self.config.working_memory_capacity {
            // Remove lowest activation item
            if let Some(min_idx) = self
                .working_memory
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.activation.total_cmp(&b.activation))
                .map(|(i, _)| i)
            {
                if let Some(evicted) = self.working_memory.remove(min_idx) {
                    if evicted.activation >= GRADUATION_ACTIVATION_THRESHOLD {
                        self.stats.graduations += 1;
                        self.graduation_queue.push(evicted);
                    } else {
                        self.stats.evictions_dropped += 1;
                    }
                }
            }
        }

        item.added_at = self.time_step;
        self.working_memory.push_back(item);
        self.stats.items_processed += 1;
        true
    }

    /// Get item from working memory by ID
    pub fn get_memory_item(&self, id: &str) -> Option<&WorkingMemoryItem> {
        self.working_memory.iter().find(|i| i.id == id)
    }

    /// Set current focus
    pub fn set_focus(&mut self, item_id: &str) -> bool {
        if let Some(item) = self.working_memory.iter_mut().find(|i| i.id == item_id) {
            item.rehearse();
            self.current_focus = Some(item_id.to_string());
            true
        } else {
            false
        }
    }

    /// Add a planned action
    pub fn add_action(&mut self, action: PlannedAction) {
        self.action_stack.push(action);
    }

    /// Make an executive decision about which action to take
    pub fn decide(&mut self) -> ExecutiveDecision {
        self.stats.decisions_made += 1;
        let mut reasoning = Vec::new();
        let mut inhibited = Vec::new();

        // Update action statuses based on dependencies
        let completed_ids: Vec<_> = self
            .action_stack
            .iter()
            .filter(|a| a.status == ActionStatus::Completed)
            .map(|a| a.id.clone())
            .collect();

        for action in self.action_stack.iter_mut() {
            if action.status == ActionStatus::Pending {
                let deps_met = action
                    .dependencies
                    .iter()
                    .all(|d| completed_ids.contains(d));
                if deps_met {
                    action.status = ActionStatus::Ready;
                }
            }
        }

        // Get ready actions
        let ready_actions: Vec<_> = self
            .action_stack
            .iter()
            .filter(|a| a.status == ActionStatus::Ready)
            .collect();

        reasoning.push(format!(
            "{} actions ready for execution",
            ready_actions.len()
        ));

        // Apply inhibition to low-priority actions when capacity is limited
        let memory_utilization =
            self.working_memory.len() as f32 / self.config.working_memory_capacity as f32;

        if memory_utilization > self.config.inhibition_threshold {
            reasoning.push("Memory utilization high, applying inhibition".to_string());

            for action in self.action_stack.iter_mut() {
                if action.status == ActionStatus::Ready && action.priority < 0.3 {
                    action.status = ActionStatus::Inhibited;
                    inhibited.push(action.id.clone());
                    self.stats.inhibitions += 1;
                }
            }
        }

        // Select highest priority ready action
        let selected = self
            .action_stack
            .iter_mut()
            .filter(|a| a.status == ActionStatus::Ready)
            .max_by(|a, b| a.priority.total_cmp(&b.priority))
            .map(|a| {
                a.status = ActionStatus::InProgress;
                a.id.clone()
            });

        let confidence = if let Some(ref sel_id) = selected {
            self.action_stack
                .iter()
                .find(|a| &a.id == sel_id)
                .map(|a| a.priority)
                .unwrap_or(0.0)
        } else {
            0.0
        };

        ExecutiveDecision {
            selected_action: selected,
            confidence,
            inhibited,
            memory_items_considered: self.working_memory.len(),
            reasoning,
        }
    }

    /// Mark an action as completed
    pub fn complete_action(&mut self, action_id: &str) -> bool {
        if let Some(action) = self.action_stack.iter_mut().find(|a| a.id == action_id) {
            action.status = ActionStatus::Completed;
            self.stats.tasks_completed += 1;
            true
        } else {
            false
        }
    }

    /// Get current working memory contents
    pub fn memory_contents(&self) -> &VecDeque<WorkingMemoryItem> {
        &self.working_memory
    }

    /// Get current focus
    pub fn current_focus(&self) -> Option<&str> {
        self.current_focus.as_deref()
    }

    /// Get statistics
    pub fn stats(&self) -> &PrefrontalStats {
        &self.stats
    }

    /// Drain graduated items from the internal queue.
    ///
    /// Call this after `tick()` or `add_to_memory()` to retrieve evicted items
    /// that are candidates for episodic memory storage.
    pub fn drain_graduates(&mut self) -> Vec<WorkingMemoryItem> {
        self.graduation_queue.drain(..).collect()
    }

    /// Plan a sequence of actions to achieve a goal
    pub fn plan(&self, goal: &ContinuousHV, available_actions: &[PlannedAction]) -> Vec<String> {
        // Simple greedy planning: select actions that move toward goal
        let mut plan = Vec::new();
        let mut current_state = goal.clone();
        let mut remaining_actions: Vec<_> = available_actions.iter().collect();

        for _ in 0..self.config.planning_horizon {
            if remaining_actions.is_empty() {
                break;
            }

            // Find action whose expected outcome is most similar to goal
            let best_idx = remaining_actions
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    let sim_a = a.expected_outcome.similarity(&current_state);
                    let sim_b = b.expected_outcome.similarity(&current_state);
                    sim_a.total_cmp(&sim_b)
                })
                .map(|(i, _)| i);

            if let Some(idx) = best_idx {
                let action = remaining_actions.remove(idx);
                plan.push(action.id.clone());
                current_state = action.expected_outcome.clone();
            }
        }

        plan
    }
}

impl Default for PrefrontalCortex {
    fn default() -> Self {
        Self::new(PrefrontalConfig::default())
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn test_prefrontal_creation() {
        let pfc = PrefrontalCortex::default();
        assert_eq!(pfc.working_memory.len(), 0);
        assert_eq!(pfc.stats.decisions_made, 0);
    }

    #[test]
    fn test_working_memory() {
        let mut pfc = PrefrontalCortex::default();

        let item = WorkingMemoryItem::new("test1", ContinuousHV::random(512, 0xCAFE_0001));
        assert!(pfc.add_to_memory(item));
        assert_eq!(pfc.working_memory.len(), 1);
    }

    #[test]
    fn test_memory_capacity() {
        let mut config = PrefrontalConfig::default();
        config.working_memory_capacity = 3;
        let mut pfc = PrefrontalCortex::new(config);

        for i in 0..5 {
            let item = WorkingMemoryItem::new(
                format!("item{}", i),
                ContinuousHV::random(512, 0xCAFE_0002 + i as u64),
            );
            pfc.add_to_memory(item);
        }

        assert_eq!(pfc.working_memory.len(), 3);
    }

    #[test]
    fn test_focus_setting() {
        let mut pfc = PrefrontalCortex::default();

        let item = WorkingMemoryItem::new("focus_test", ContinuousHV::random(512, 0xCAFE_0010));
        pfc.add_to_memory(item);

        assert!(pfc.set_focus("focus_test"));
        assert_eq!(pfc.current_focus(), Some("focus_test"));
    }

    #[test]
    fn test_eviction_graduates_high_activation() {
        let mut config = PrefrontalConfig::default();
        config.working_memory_capacity = 2;
        let mut pfc = PrefrontalCortex::new(config);

        // Add two items
        let mut item1 = WorkingMemoryItem::new("high", ContinuousHV::random(512, 1));
        item1.activation = 0.8; // Above graduation threshold
        pfc.add_to_memory(item1);

        let item2 = WorkingMemoryItem::new("stay", ContinuousHV::random(512, 2));
        pfc.add_to_memory(item2);

        // Adding a third should evict the lowest-activation one
        // item2 has activation 1.0 (fresh), item1 has 0.8 — item1 is evicted
        let item3 = WorkingMemoryItem::new("new", ContinuousHV::random(512, 3));
        pfc.add_to_memory(item3);

        // The evicted high-activation item should be in the graduation queue
        let graduates = pfc.drain_graduates();
        assert_eq!(graduates.len(), 1);
        assert_eq!(graduates[0].id, "high");
        assert!(graduates[0].activation >= GRADUATION_ACTIVATION_THRESHOLD);
        assert_eq!(pfc.stats.graduations, 1);
    }

    #[test]
    fn test_eviction_drops_low_activation() {
        let mut config = PrefrontalConfig::default();
        config.working_memory_capacity = 2;
        let mut pfc = PrefrontalCortex::new(config);

        let mut item1 = WorkingMemoryItem::new("low", ContinuousHV::random(512, 1));
        item1.activation = 0.1; // Below graduation threshold
        pfc.add_to_memory(item1);

        let item2 = WorkingMemoryItem::new("stay", ContinuousHV::random(512, 2));
        pfc.add_to_memory(item2);

        // Evict the low-activation item
        let item3 = WorkingMemoryItem::new("new", ContinuousHV::random(512, 3));
        pfc.add_to_memory(item3);

        let graduates = pfc.drain_graduates();
        assert!(
            graduates.is_empty(),
            "Low-activation items should not graduate"
        );
        assert_eq!(pfc.stats.evictions_dropped, 1);
    }

    #[test]
    fn test_executive_decision() {
        let mut pfc = PrefrontalCortex::default();

        let action = PlannedAction {
            id: "action1".to_string(),
            goal_id: "goal1".to_string(),
            embedding: ContinuousHV::random(512, 0xCAFE_0020),
            expected_outcome: ContinuousHV::random(512, 0xCAFE_0021),
            priority: 0.8,
            effort: 0.5,
            dependencies: Vec::new(),
            status: ActionStatus::Ready,
        };

        pfc.add_action(action);
        let decision = pfc.decide();

        assert!(decision.selected_action.is_some());
        assert!(decision.confidence > 0.5);
    }
}
