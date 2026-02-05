//! # Revolutionary Improvement #58: Meta-Cognitive Controller
//!
//! This module implements the unified meta-cognitive controller that orchestrates
//! all consciousness subsystems into a coherent system. Inspired by:
//!
//! - Baars' Global Workspace Theory (cognitive resource broadcasting)
//! - Dehaene's Neuronal Workspace Model (attention allocation)
//! - Anderson's ACT-R (resource-bounded cognition)
//! - Minsky's Society of Mind (agent coordination)
//!
//! ## The Meta-Cognitive Architecture
//!
//! 1. Monitors all subsystems (SelfModel, WorldModel, GradientOptimizer, MotivationSystem)
//! 2. Detects when subsystems are underperforming
//! 3. Allocates cognitive resources dynamically
//! 4. Coordinates between subsystems for coherent improvement
//! 5. Maintains meta-level goals about the improvement process itself

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// =============================================================================
// COGNITIVE RESOURCES
// =============================================================================

/// Types of cognitive resources that can be allocated
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CognitiveResourceType {
    /// Computational cycles for processing
    Computation,
    /// Working memory capacity
    WorkingMemory,
    /// Attention focus slots
    Attention,
    /// Long-term memory consolidation
    LongTermMemory,
    /// Energy/metabolic resources
    Energy,
}

impl CognitiveResourceType {
    /// All resource types
    pub fn all() -> &'static [CognitiveResourceType] {
        &[
            CognitiveResourceType::Computation,
            CognitiveResourceType::WorkingMemory,
            CognitiveResourceType::Attention,
            CognitiveResourceType::LongTermMemory,
            CognitiveResourceType::Energy,
        ]
    }
}

/// Represents available cognitive resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveResources {
    /// Available capacity for each resource type [0.0, 1.0]
    available: HashMap<CognitiveResourceType, f64>,
    /// Maximum capacity for each resource type
    max_capacity: HashMap<CognitiveResourceType, f64>,
    /// Current allocations to subsystems
    allocations: HashMap<SubsystemId, HashMap<CognitiveResourceType, f64>>,
    /// Resource regeneration rates
    regen_rates: HashMap<CognitiveResourceType, f64>,
}

impl Default for CognitiveResources {
    fn default() -> Self {
        let mut available = HashMap::new();
        let mut max_capacity = HashMap::new();
        let mut regen_rates = HashMap::new();

        for &resource in CognitiveResourceType::all() {
            available.insert(resource, 1.0);
            max_capacity.insert(resource, 1.0);
            regen_rates.insert(resource, 0.1); // 10% regeneration per cycle
        }

        Self {
            available,
            max_capacity,
            allocations: HashMap::new(),
            regen_rates,
        }
    }
}

impl CognitiveResources {
    /// Get available amount of a resource
    pub fn available(&self, resource: CognitiveResourceType) -> f64 {
        *self.available.get(&resource).unwrap_or(&0.0)
    }

    /// Allocate resources to a subsystem
    pub fn allocate(
        &mut self,
        subsystem: SubsystemId,
        resource: CognitiveResourceType,
        amount: f64,
    ) -> bool {
        let available = self.available.get_mut(&resource).unwrap();
        if *available >= amount {
            *available -= amount;
            let subsystem_alloc = self.allocations.entry(subsystem).or_default();
            *subsystem_alloc.entry(resource).or_insert(0.0) += amount;
            true
        } else {
            false
        }
    }

    /// Release resources from a subsystem
    pub fn release(&mut self, subsystem: SubsystemId, resource: CognitiveResourceType, amount: f64) {
        if let Some(subsystem_alloc) = self.allocations.get_mut(&subsystem) {
            if let Some(allocated) = subsystem_alloc.get_mut(&resource) {
                let to_release = amount.min(*allocated);
                *allocated -= to_release;
                *self.available.get_mut(&resource).unwrap() += to_release;
            }
        }
    }

    /// Release all resources from a subsystem
    pub fn release_all(&mut self, subsystem: SubsystemId) {
        if let Some(allocs) = self.allocations.remove(&subsystem) {
            for (resource, amount) in allocs {
                *self.available.get_mut(&resource).unwrap() += amount;
            }
        }
    }

    /// Regenerate resources over time
    pub fn regenerate(&mut self) {
        for &resource in CognitiveResourceType::all() {
            let available = self.available.get_mut(&resource).unwrap();
            let max = *self.max_capacity.get(&resource).unwrap();
            let rate = *self.regen_rates.get(&resource).unwrap();

            *available = (*available + rate * max).min(max);
        }
    }

    /// Get total resources allocated to a subsystem
    pub fn allocated_to(&self, subsystem: SubsystemId) -> f64 {
        self.allocations
            .get(&subsystem)
            .map(|allocs| allocs.values().sum())
            .unwrap_or(0.0)
    }

    /// Get utilization ratio [0.0, 1.0]
    pub fn utilization(&self) -> f64 {
        let total_available: f64 = self.available.values().sum();
        let total_max: f64 = self.max_capacity.values().sum();
        if total_max > 0.0 {
            1.0 - (total_available / total_max)
        } else {
            0.0
        }
    }
}

// =============================================================================
// SUBSYSTEM IDENTIFICATION AND HEALTH
// =============================================================================

/// Identifier for cognitive subsystems
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubsystemId {
    SelfModel,
    WorldModel,
    GradientOptimizer,
    MotivationSystem,
    RecursiveOptimizer,
    PerformanceMonitor,
    ImprovementGenerator,
}

impl SubsystemId {
    /// All subsystem IDs
    pub fn all() -> &'static [SubsystemId] {
        &[
            SubsystemId::SelfModel,
            SubsystemId::WorldModel,
            SubsystemId::GradientOptimizer,
            SubsystemId::MotivationSystem,
            SubsystemId::RecursiveOptimizer,
            SubsystemId::PerformanceMonitor,
            SubsystemId::ImprovementGenerator,
        ]
    }
}

/// Health status of a subsystem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsystemHealth {
    /// Subsystem identifier
    pub id: SubsystemId,
    /// Is the subsystem operational?
    pub operational: bool,
    /// Performance score [0.0, 1.0]
    pub performance: f64,
    /// Last update time
    last_update: u64,
    /// Error count since last reset
    pub error_count: usize,
    /// Warning flags
    pub warnings: Vec<String>,
    /// Resource efficiency (output per resource consumed)
    pub efficiency: f64,
    /// Staleness (cycles since last meaningful activity)
    pub staleness: u32,
}

impl SubsystemHealth {
    /// Create a new health status
    pub fn new(id: SubsystemId) -> Self {
        Self {
            id,
            operational: true,
            performance: 1.0,
            last_update: 0,
            error_count: 0,
            warnings: Vec::new(),
            efficiency: 1.0,
            staleness: 0,
        }
    }

    /// Is the subsystem healthy?
    pub fn is_healthy(&self) -> bool {
        self.operational && self.performance > 0.5 && self.error_count < 5
    }

    /// Report an error
    pub fn report_error(&mut self) {
        self.error_count += 1;
        if self.error_count >= 5 {
            self.operational = false;
        }
    }

    /// Update performance
    pub fn update_performance(&mut self, new_performance: f64, cycle: u64) {
        self.performance = 0.8 * self.performance + 0.2 * new_performance.clamp(0.0, 1.0);
        self.last_update = cycle;
        self.staleness = 0;
    }

    /// Tick staleness counter
    pub fn tick_staleness(&mut self) {
        self.staleness += 1;
    }

    /// Reset error count
    pub fn reset_errors(&mut self) {
        self.error_count = 0;
        self.operational = true;
    }

    /// Overall health score
    pub fn health_score(&self) -> f64 {
        let operational_factor = if self.operational { 1.0 } else { 0.0 };
        let error_penalty = 1.0 - (self.error_count as f64 * 0.1).min(0.5);
        let staleness_penalty = 1.0 - (self.staleness as f64 * 0.02).min(0.5);

        operational_factor * self.performance * error_penalty * staleness_penalty
    }
}

// =============================================================================
// META-GOALS
// =============================================================================

/// Meta-level goal about the improvement process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaGoal {
    /// Goal identifier
    pub id: u64,
    /// Description of the meta-goal
    pub description: String,
    /// Target subsystem (None for system-wide)
    pub target: Option<SubsystemId>,
    /// Goal type
    pub goal_type: MetaGoalType,
    /// Target value
    pub target_value: f64,
    /// Current progress
    pub progress: f64,
    /// Priority [0.0, 1.0]
    pub priority: f64,
    /// Is the goal active?
    pub active: bool,
}

/// Types of meta-goals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetaGoalType {
    /// Improve efficiency of a subsystem
    ImproveEfficiency,
    /// Increase coherence between subsystems
    IncreaseCoherence,
    /// Reduce resource consumption
    ReduceResources,
    /// Increase overall phi
    IncreasePhi,
    /// Balance resource allocation
    BalanceAllocation,
    /// Recover failing subsystem
    RecoverSubsystem,
    /// Optimize improvement rate
    OptimizeImprovementRate,
}

impl MetaGoal {
    /// Create a new meta-goal
    pub fn new(
        id: u64,
        description: String,
        target: Option<SubsystemId>,
        goal_type: MetaGoalType,
        target_value: f64,
        priority: f64,
    ) -> Self {
        Self {
            id,
            description,
            target,
            goal_type,
            target_value,
            progress: 0.0,
            priority: priority.clamp(0.0, 1.0),
            active: true,
        }
    }

    /// Update progress toward the goal
    pub fn update_progress(&mut self, current_value: f64) {
        if self.target_value > 0.0 {
            self.progress = (current_value / self.target_value).clamp(0.0, 1.0);
        }
    }

    /// Is the goal achieved?
    pub fn is_achieved(&self) -> bool {
        self.progress >= 1.0
    }
}

// =============================================================================
// GLOBAL WORKSPACE / BROADCASTING
// =============================================================================

/// Configuration for meta-cognitive controller
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaCognitiveConfig {
    /// Minimum health threshold to consider subsystem operational
    pub min_health_threshold: f64,
    /// How often to rebalance resources (in cycles)
    pub rebalance_interval: u32,
    /// Maximum staleness before intervening
    pub max_staleness: u32,
    /// Resource allocation smoothing factor
    pub allocation_smoothing: f64,
    /// Enable automatic recovery attempts
    pub auto_recovery: bool,
    /// Maximum meta-goals to track
    pub max_meta_goals: usize,
}

impl Default for MetaCognitiveConfig {
    fn default() -> Self {
        Self {
            min_health_threshold: 0.5,
            rebalance_interval: 10,
            max_staleness: 20,
            allocation_smoothing: 0.3,
            auto_recovery: true,
            max_meta_goals: 10,
        }
    }
}

/// Statistics for meta-cognitive operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetaCognitiveStats {
    /// Total cycles run
    pub cycles: u64,
    /// Resource rebalancing events
    pub rebalance_count: u32,
    /// Recovery attempts
    pub recovery_attempts: u32,
    /// Successful recoveries
    pub successful_recoveries: u32,
    /// Meta-goals achieved
    pub goals_achieved: u32,
    /// Average system coherence
    pub avg_coherence: f64,
}

/// Attention broadcast for global workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionBroadcast {
    /// Source subsystem
    pub source: SubsystemId,
    /// Priority of the broadcast
    pub priority: f64,
    /// Content type
    pub content_type: BroadcastContentType,
    /// Content data (encoded)
    pub content: Vec<f64>,
    /// Timestamp
    pub timestamp: u64,
}

/// Types of content that can be broadcast
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BroadcastContentType {
    /// Discovered pattern
    Pattern,
    /// Bottleneck alert
    Bottleneck,
    /// Improvement proposal
    Proposal,
    /// Goal update
    GoalUpdate,
    /// State change notification
    StateChange,
}

// =============================================================================
// META-COGNITIVE CONTROLLER
// =============================================================================

/// The unified meta-cognitive controller
pub struct MetaCognitiveController {
    /// Cognitive resources
    resources: CognitiveResources,
    /// Subsystem health tracking
    health: HashMap<SubsystemId, SubsystemHealth>,
    /// Active meta-goals
    meta_goals: Vec<MetaGoal>,
    /// Global workspace (attention broadcasts)
    global_workspace: VecDeque<AttentionBroadcast>,
    /// Configuration
    config: MetaCognitiveConfig,
    /// Statistics
    stats: MetaCognitiveStats,
    /// Current cycle
    cycle: u64,
    /// Coherence matrix between subsystems
    coherence_matrix: HashMap<(SubsystemId, SubsystemId), f64>,
    /// Target resource allocation per subsystem
    target_allocation: HashMap<SubsystemId, f64>,
    /// Next goal ID
    next_goal_id: u64,
}

impl MetaCognitiveController {
    /// Create a new meta-cognitive controller
    pub fn new(config: MetaCognitiveConfig) -> Self {
        let mut health = HashMap::new();
        let mut target_allocation = HashMap::new();

        // Initialize health for all subsystems
        for &id in SubsystemId::all() {
            health.insert(id, SubsystemHealth::new(id));
            target_allocation.insert(id, 1.0 / SubsystemId::all().len() as f64);
        }

        Self {
            resources: CognitiveResources::default(),
            health,
            meta_goals: Vec::new(),
            global_workspace: VecDeque::with_capacity(100),
            config,
            stats: MetaCognitiveStats::default(),
            cycle: 0,
            coherence_matrix: HashMap::new(),
            target_allocation,
            next_goal_id: 0,
        }
    }

    /// Run one meta-cognitive cycle
    pub fn cycle(&mut self) {
        self.cycle += 1;
        self.stats.cycles = self.cycle;

        // 1. Update staleness for all subsystems
        for health in self.health.values_mut() {
            health.tick_staleness();
        }

        // 2. Regenerate resources
        self.resources.regenerate();

        // 3. Check for subsystems needing intervention
        if self.config.auto_recovery {
            self.attempt_recoveries();
        }

        // 4. Rebalance resources if needed
        if self.cycle % self.config.rebalance_interval as u64 == 0 {
            self.rebalance_resources();
        }

        // 5. Update meta-goals
        self.update_meta_goals();

        // 6. Process global workspace
        self.process_workspace();

        // 7. Update coherence estimate
        self.update_coherence();
    }

    /// Report subsystem activity
    pub fn report_activity(&mut self, subsystem: SubsystemId, performance: f64) {
        if let Some(health) = self.health.get_mut(&subsystem) {
            health.update_performance(performance, self.cycle);
        }
    }

    /// Report subsystem error
    pub fn report_error(&mut self, subsystem: SubsystemId) {
        if let Some(health) = self.health.get_mut(&subsystem) {
            health.report_error();
        }
    }

    /// Broadcast to global workspace
    pub fn broadcast(
        &mut self,
        source: SubsystemId,
        content_type: BroadcastContentType,
        content: Vec<f64>,
        priority: f64,
    ) {
        let broadcast = AttentionBroadcast {
            source,
            priority: priority.clamp(0.0, 1.0),
            content_type,
            content,
            timestamp: self.cycle,
        };

        self.global_workspace.push_back(broadcast);

        // Keep workspace bounded
        while self.global_workspace.len() > 100 {
            self.global_workspace.pop_front();
        }
    }

    /// Request resources for a subsystem
    pub fn request_resources(
        &mut self,
        subsystem: SubsystemId,
        resource: CognitiveResourceType,
        amount: f64,
    ) -> bool {
        // Check if subsystem is healthy enough to receive resources
        if let Some(health) = self.health.get(&subsystem) {
            if health.health_score() < self.config.min_health_threshold {
                return false;
            }
        }

        self.resources.allocate(subsystem, resource, amount)
    }

    /// Release resources from a subsystem
    pub fn release_resources(
        &mut self,
        subsystem: SubsystemId,
        resource: CognitiveResourceType,
        amount: f64,
    ) {
        self.resources.release(subsystem, resource, amount);
    }

    /// Add a meta-goal
    pub fn add_meta_goal(
        &mut self,
        description: String,
        target: Option<SubsystemId>,
        goal_type: MetaGoalType,
        target_value: f64,
        priority: f64,
    ) -> u64 {
        let id = self.next_goal_id;
        self.next_goal_id += 1;

        let goal = MetaGoal::new(id, description, target, goal_type, target_value, priority);

        // Remove lowest priority if at capacity
        if self.meta_goals.len() >= self.config.max_meta_goals {
            let min_priority_idx = self
                .meta_goals
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.priority.partial_cmp(&b.priority).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i);

            if let Some(idx) = min_priority_idx {
                if self.meta_goals[idx].priority < priority {
                    self.meta_goals.remove(idx);
                }
            }
        }

        if self.meta_goals.len() < self.config.max_meta_goals {
            self.meta_goals.push(goal);
        }

        id
    }

    /// Get subsystem health
    pub fn get_health(&self, subsystem: SubsystemId) -> Option<&SubsystemHealth> {
        self.health.get(&subsystem)
    }

    /// Get overall system health
    pub fn system_health(&self) -> f64 {
        let total: f64 = self.health.values().map(|h| h.health_score()).sum();
        total / self.health.len() as f64
    }

    /// Get resource utilization
    pub fn resource_utilization(&self) -> f64 {
        self.resources.utilization()
    }

    /// Attempt to recover unhealthy subsystems
    fn attempt_recoveries(&mut self) {
        let unhealthy: Vec<SubsystemId> = self
            .health
            .iter()
            .filter(|(_, h)| !h.is_healthy())
            .map(|(&id, _)| id)
            .collect();

        for subsystem in unhealthy {
            self.stats.recovery_attempts += 1;

            // Release all resources from unhealthy subsystem
            self.resources.release_all(subsystem);

            // Reset error count (giving it another chance)
            if let Some(health) = self.health.get_mut(&subsystem) {
                health.reset_errors();
                health.performance = 0.5; // Reset to middle ground
            }

            // Allocate minimal resources for recovery
            self.resources.allocate(subsystem, CognitiveResourceType::Computation, 0.1);
            self.resources.allocate(subsystem, CognitiveResourceType::Energy, 0.1);

            self.stats.successful_recoveries += 1;
        }
    }

    /// Rebalance resources across subsystems
    fn rebalance_resources(&mut self) {
        self.stats.rebalance_count += 1;

        // Calculate desired allocation based on health and priority
        let mut scores: HashMap<SubsystemId, f64> = HashMap::new();

        for &id in SubsystemId::all() {
            if let Some(health) = self.health.get(&id) {
                // Higher score = more resources needed
                let health_factor = if health.is_healthy() { 1.0 } else { 1.5 }; // Unhealthy get more
                let staleness_factor = 1.0 + (health.staleness as f64 * 0.05);
                let efficiency_factor = 1.0 / health.efficiency.max(0.1);

                scores.insert(id, health_factor * staleness_factor * efficiency_factor);
            }
        }

        let total_score: f64 = scores.values().sum();
        if total_score > 0.0 {
            for (&id, &score) in &scores {
                let new_target = score / total_score;
                let current = *self.target_allocation.get(&id).unwrap_or(&0.0);
                let smoothed = current * (1.0 - self.config.allocation_smoothing)
                    + new_target * self.config.allocation_smoothing;
                self.target_allocation.insert(id, smoothed);
            }
        }
    }

    /// Update progress on meta-goals
    fn update_meta_goals(&mut self) {
        // Pre-compute all values we need to avoid borrow conflicts
        let coherence = self.calculate_coherence();
        let resource_utilization = self.resource_utilization();
        let system_health = self.system_health();
        let allocation_balance = self.allocation_balance();

        let avg_efficiency = self.health.values().map(|h| h.efficiency).sum::<f64>()
            / self.health.len().max(1) as f64;

        // Pre-compute per-subsystem values
        let health_efficiencies: HashMap<SubsystemId, f64> = self
            .health
            .iter()
            .map(|(&id, h)| (id, h.efficiency))
            .collect();
        let health_scores: HashMap<SubsystemId, f64> = self
            .health
            .iter()
            .map(|(&id, h)| (id, h.health_score()))
            .collect();

        let mut achieved = 0;

        for goal in &mut self.meta_goals {
            if !goal.active {
                continue;
            }

            // Calculate current value based on goal type using pre-computed values
            let current_value = match goal.goal_type {
                MetaGoalType::ImproveEfficiency => {
                    if let Some(target) = goal.target {
                        *health_efficiencies.get(&target).unwrap_or(&0.0)
                    } else {
                        avg_efficiency
                    }
                }
                MetaGoalType::IncreaseCoherence => coherence,
                MetaGoalType::ReduceResources => 1.0 - resource_utilization,
                MetaGoalType::IncreasePhi => system_health,
                MetaGoalType::BalanceAllocation => allocation_balance,
                MetaGoalType::RecoverSubsystem => {
                    if let Some(target) = goal.target {
                        *health_scores.get(&target).unwrap_or(&0.0)
                    } else {
                        system_health
                    }
                }
                MetaGoalType::OptimizeImprovementRate => avg_efficiency,
            };

            goal.update_progress(current_value);

            if goal.is_achieved() {
                goal.active = false;
                achieved += 1;
            }
        }

        self.stats.goals_achieved += achieved;
    }

    /// Process global workspace broadcasts
    fn process_workspace(&mut self) {
        // Process high-priority broadcasts first
        let mut broadcasts: Vec<_> = self.global_workspace.iter().cloned().collect();
        broadcasts.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));

        for broadcast in broadcasts.iter().take(5) {
            match broadcast.content_type {
                BroadcastContentType::Bottleneck => {
                    // Create recovery goal for source subsystem
                    self.add_meta_goal(
                        format!("Recover {:?} from bottleneck", broadcast.source),
                        Some(broadcast.source),
                        MetaGoalType::RecoverSubsystem,
                        0.8,
                        broadcast.priority,
                    );
                }
                BroadcastContentType::Proposal => {
                    // Allocate attention resources to evaluate proposal
                    self.resources.allocate(
                        broadcast.source,
                        CognitiveResourceType::Attention,
                        0.1,
                    );
                }
                _ => {}
            }
        }
    }

    /// Calculate coherence between subsystems
    fn calculate_coherence(&self) -> f64 {
        if self.coherence_matrix.is_empty() {
            return 1.0;
        }
        let sum: f64 = self.coherence_matrix.values().sum();
        sum / self.coherence_matrix.len() as f64
    }

    /// Update coherence matrix
    fn update_coherence(&mut self) {
        // Calculate pairwise coherence based on health correlation
        let subsystems: Vec<SubsystemId> = SubsystemId::all().to_vec();

        for i in 0..subsystems.len() {
            for j in (i + 1)..subsystems.len() {
                let a = subsystems[i];
                let b = subsystems[j];

                let health_a = self.health.get(&a).map(|h| h.health_score()).unwrap_or(0.5);
                let health_b = self.health.get(&b).map(|h| h.health_score()).unwrap_or(0.5);

                // Coherence increases when both are healthy or both are unhealthy
                let coherence = 1.0 - (health_a - health_b).abs();
                self.coherence_matrix.insert((a, b), coherence);
            }
        }

        self.stats.avg_coherence = self.calculate_coherence();
    }

    /// Calculate how balanced resource allocation is
    fn allocation_balance(&self) -> f64 {
        if self.target_allocation.is_empty() {
            return 1.0;
        }

        let mean = self.target_allocation.values().sum::<f64>() / self.target_allocation.len() as f64;
        let variance: f64 = self
            .target_allocation
            .values()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>()
            / self.target_allocation.len() as f64;

        // Lower variance = more balanced = higher score
        1.0 / (1.0 + variance.sqrt() * 10.0)
    }

    /// Get active meta-goals
    pub fn active_goals(&self) -> impl Iterator<Item = &MetaGoal> {
        self.meta_goals.iter().filter(|g| g.active)
    }

    /// Get recent broadcasts from global workspace
    pub fn recent_broadcasts(&self, count: usize) -> impl Iterator<Item = &AttentionBroadcast> {
        self.global_workspace.iter().rev().take(count)
    }

    /// Get summary of controller state
    pub fn summary(&self) -> MetaCognitiveSummary {
        MetaCognitiveSummary {
            cycle: self.cycle,
            system_health: self.system_health(),
            resource_utilization: self.resource_utilization(),
            coherence: self.calculate_coherence(),
            active_goals: self.meta_goals.iter().filter(|g| g.active).count(),
            workspace_size: self.global_workspace.len(),
            stats: self.stats.clone(),
        }
    }
}

/// Summary of meta-cognitive controller state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaCognitiveSummary {
    pub cycle: u64,
    pub system_health: f64,
    pub resource_utilization: f64,
    pub coherence: f64,
    pub active_goals: usize,
    pub workspace_size: usize,
    pub stats: MetaCognitiveStats,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_resources_default() {
        let resources = CognitiveResources::default();

        for &resource in CognitiveResourceType::all() {
            assert_eq!(resources.available(resource), 1.0);
        }
    }

    #[test]
    fn test_resource_allocation() {
        let mut resources = CognitiveResources::default();

        // Allocate resources
        assert!(resources.allocate(SubsystemId::SelfModel, CognitiveResourceType::Computation, 0.5));
        assert_eq!(resources.available(CognitiveResourceType::Computation), 0.5);

        // Can't over-allocate
        assert!(!resources.allocate(
            SubsystemId::WorldModel,
            CognitiveResourceType::Computation,
            0.6
        ));
    }

    #[test]
    fn test_resource_release() {
        let mut resources = CognitiveResources::default();

        resources.allocate(SubsystemId::SelfModel, CognitiveResourceType::Computation, 0.5);
        resources.release(SubsystemId::SelfModel, CognitiveResourceType::Computation, 0.3);

        assert_eq!(resources.available(CognitiveResourceType::Computation), 0.8);
    }

    #[test]
    fn test_resource_regeneration() {
        let mut resources = CognitiveResources::default();

        resources.allocate(SubsystemId::SelfModel, CognitiveResourceType::Computation, 0.5);
        resources.regenerate();

        // Should regenerate 10% of max (0.1) up to max
        assert!(resources.available(CognitiveResourceType::Computation) > 0.5);
    }

    #[test]
    fn test_subsystem_health_creation() {
        let health = SubsystemHealth::new(SubsystemId::SelfModel);

        assert!(health.operational);
        assert_eq!(health.performance, 1.0);
        assert_eq!(health.error_count, 0);
        assert!(health.is_healthy());
    }

    #[test]
    fn test_subsystem_health_error_handling() {
        let mut health = SubsystemHealth::new(SubsystemId::SelfModel);

        for _ in 0..5 {
            health.report_error();
        }

        assert!(!health.operational);
        assert!(!health.is_healthy());
    }

    #[test]
    fn test_meta_goal_creation() {
        let goal = MetaGoal::new(
            1,
            "Test goal".to_string(),
            Some(SubsystemId::SelfModel),
            MetaGoalType::ImproveEfficiency,
            1.0,
            0.8,
        );

        assert_eq!(goal.id, 1);
        assert!(goal.active);
        assert!(!goal.is_achieved());
    }

    #[test]
    fn test_meta_goal_progress() {
        let mut goal = MetaGoal::new(
            1,
            "Test goal".to_string(),
            None,
            MetaGoalType::IncreasePhi,
            1.0,
            0.5,
        );

        goal.update_progress(0.5);
        assert_eq!(goal.progress, 0.5);

        goal.update_progress(1.0);
        assert!(goal.is_achieved());
    }

    #[test]
    fn test_meta_cognitive_controller_creation() {
        let controller = MetaCognitiveController::new(MetaCognitiveConfig::default());

        assert_eq!(controller.cycle, 0);
        assert!(controller.system_health() > 0.9);
        assert_eq!(controller.meta_goals.len(), 0);
    }

    #[test]
    fn test_meta_cognitive_controller_cycle() {
        let mut controller = MetaCognitiveController::new(MetaCognitiveConfig::default());

        controller.cycle();

        assert_eq!(controller.cycle, 1);
        assert_eq!(controller.stats.cycles, 1);
    }

    #[test]
    fn test_meta_cognitive_report_activity() {
        let mut controller = MetaCognitiveController::new(MetaCognitiveConfig::default());

        controller.report_activity(SubsystemId::SelfModel, 0.9);

        let health = controller.get_health(SubsystemId::SelfModel).unwrap();
        // Initial is 1.0, update with 0.9 should give 0.8*1.0 + 0.2*0.9 = 0.98
        assert!(health.performance > 0.95);
    }

    #[test]
    fn test_meta_cognitive_broadcast() {
        let mut controller = MetaCognitiveController::new(MetaCognitiveConfig::default());

        controller.broadcast(
            SubsystemId::WorldModel,
            BroadcastContentType::Pattern,
            vec![1.0, 2.0, 3.0],
            0.8,
        );

        assert_eq!(controller.global_workspace.len(), 1);
    }

    #[test]
    fn test_meta_cognitive_add_goal() {
        let mut controller = MetaCognitiveController::new(MetaCognitiveConfig::default());

        let id = controller.add_meta_goal(
            "Increase coherence".to_string(),
            None,
            MetaGoalType::IncreaseCoherence,
            1.0,
            0.9,
        );

        assert_eq!(id, 0);
        assert_eq!(controller.meta_goals.len(), 1);
    }

    #[test]
    fn test_meta_cognitive_summary() {
        let controller = MetaCognitiveController::new(MetaCognitiveConfig::default());
        let summary = controller.summary();

        assert_eq!(summary.cycle, 0);
        assert!(summary.system_health > 0.9);
        assert!(summary.coherence >= 0.0);
    }

    #[test]
    fn test_meta_cognitive_recovery() {
        let mut controller = MetaCognitiveController::new(MetaCognitiveConfig::default());

        // Make a subsystem unhealthy
        for _ in 0..5 {
            controller.report_error(SubsystemId::WorldModel);
        }

        assert!(!controller.get_health(SubsystemId::WorldModel).unwrap().is_healthy());

        // Run cycle to trigger recovery
        controller.cycle();

        // Should have attempted recovery
        assert!(controller.stats.recovery_attempts > 0);
    }
}
