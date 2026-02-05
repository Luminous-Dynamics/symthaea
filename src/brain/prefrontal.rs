//! # Prefrontal Cortex: Executive Functions and Working Memory
//!
//! Simulates prefrontal cortex functionality including:
//! - Working memory maintenance
//! - Executive control and task switching
//! - Decision making and planning
//! - Inhibition and impulse control

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use symthaea_core::hdc::RealHV;

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
    pub embedding: RealHV,
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
    pub fn new(id: impl Into<String>, embedding: RealHV) -> Self {
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
    pub embedding: RealHV,
    /// Expected outcome embedding
    pub expected_outcome: RealHV,
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
}

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
        }
    }

    /// Advance time and decay activations
    pub fn tick(&mut self) {
        self.time_step += 1;

        // Decay working memory activations
        for item in self.working_memory.iter_mut() {
            item.decay(self.config.attention_decay);
        }

        // Remove items with zero activation
        self.working_memory.retain(|item| item.activation > 0.01);

        // Update average memory utilization
        let utilization = self.working_memory.len() as f32 / self.config.working_memory_capacity as f32;
        let n = (self.time_step as f32).max(1.0);
        self.stats.avg_memory_utilization =
            (self.stats.avg_memory_utilization * (n - 1.0) + utilization) / n;
    }

    /// Add item to working memory
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
            if let Some(min_idx) = self.working_memory.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.activation.total_cmp(&b.activation))
                .map(|(i, _)| i)
            {
                self.working_memory.remove(min_idx);
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
        let completed_ids: Vec<_> = self.action_stack.iter()
            .filter(|a| a.status == ActionStatus::Completed)
            .map(|a| a.id.clone())
            .collect();

        for action in self.action_stack.iter_mut() {
            if action.status == ActionStatus::Pending {
                let deps_met = action.dependencies.iter()
                    .all(|d| completed_ids.contains(d));
                if deps_met {
                    action.status = ActionStatus::Ready;
                }
            }
        }

        // Get ready actions
        let ready_actions: Vec<_> = self.action_stack.iter()
            .filter(|a| a.status == ActionStatus::Ready)
            .collect();

        reasoning.push(format!("{} actions ready for execution", ready_actions.len()));

        // Apply inhibition to low-priority actions when capacity is limited
        let memory_utilization = self.working_memory.len() as f32 / self.config.working_memory_capacity as f32;

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
        let selected = self.action_stack.iter_mut()
            .filter(|a| a.status == ActionStatus::Ready)
            .max_by(|a, b| a.priority.total_cmp(&b.priority))
            .map(|a| {
                a.status = ActionStatus::InProgress;
                a.id.clone()
            });

        let confidence = if selected.is_some() {
            let action = self.action_stack.iter().find(|a| Some(&a.id) == selected.as_ref()).unwrap();
            action.priority
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

    /// Plan a sequence of actions to achieve a goal
    pub fn plan(&self, goal: &RealHV, available_actions: &[PlannedAction]) -> Vec<String> {
        // Simple greedy planning: select actions that move toward goal
        let mut plan = Vec::new();
        let mut current_state = goal.clone();
        let mut remaining_actions: Vec<_> = available_actions.iter().collect();

        for _ in 0..self.config.planning_horizon {
            if remaining_actions.is_empty() {
                break;
            }

            // Find action whose expected outcome is most similar to goal
            let best_idx = remaining_actions.iter()
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

        let item = WorkingMemoryItem::new("test1", RealHV::random(512, 0xCAFE_0001));
        assert!(pfc.add_to_memory(item));
        assert_eq!(pfc.working_memory.len(), 1);
    }

    #[test]
    fn test_memory_capacity() {
        let mut config = PrefrontalConfig::default();
        config.working_memory_capacity = 3;
        let mut pfc = PrefrontalCortex::new(config);

        for i in 0..5 {
            let item = WorkingMemoryItem::new(format!("item{}", i), RealHV::random(512, 0xCAFE_0002 + i as u64));
            pfc.add_to_memory(item);
        }

        assert_eq!(pfc.working_memory.len(), 3);
    }

    #[test]
    fn test_focus_setting() {
        let mut pfc = PrefrontalCortex::default();

        let item = WorkingMemoryItem::new("focus_test", RealHV::random(512, 0xCAFE_0010));
        pfc.add_to_memory(item);

        assert!(pfc.set_focus("focus_test"));
        assert_eq!(pfc.current_focus(), Some("focus_test"));
    }

    #[test]
    fn test_executive_decision() {
        let mut pfc = PrefrontalCortex::default();

        let action = PlannedAction {
            id: "action1".to_string(),
            goal_id: "goal1".to_string(),
            embedding: RealHV::random(512, 0xCAFE_0020),
            expected_outcome: RealHV::random(512, 0xCAFE_0021),
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
