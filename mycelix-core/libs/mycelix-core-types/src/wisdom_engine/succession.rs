//! Pattern Succession Automation
//!
//! Component 20: Automatically manages pattern replacement and evolution.
//!
//! When a pattern is superseded:
//! - Automatically identifies and migrates dependents to the new pattern
//! - Transfers trust relationships appropriately
//! - Generates migration explanations and guidance
//! - Tracks succession lineage for historical understanding
//! - Manages gradual deprecation with grace periods

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use super::{PatternId, SymthaeaId, DomainId};

// ==============================================================================
// COMPONENT 20: PATTERN SUCCESSION AUTOMATION
// ==============================================================================
//
// This component manages the lifecycle of pattern replacement - when one pattern
// evolves into or is replaced by another. It handles:
//
// 1. Succession Declaration - Recording that pattern B supersedes pattern A
// 2. Dependent Migration - Helping patterns that depended on A move to B
// 3. Trust Transfer - Moving trust relationships appropriately
// 4. Grace Periods - Allowing gradual transition
// 5. Lineage Tracking - Understanding how patterns evolved
//
// Philosophy: Evolution, not abandonment. Patterns don't just die - they
// transform. We honor their history while enabling progress.

/// Reasons why one pattern might supersede another
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SuccessionReason {
    /// The new pattern is simply better (higher success rate, etc.)
    BetterPerformance,

    /// The old pattern had issues that were fixed
    BugFix,

    /// The old pattern had security vulnerabilities
    SecurityFix,

    /// The domain/context changed, making the old pattern less applicable
    ContextEvolution,

    /// The old pattern was merged with others
    Consolidation,

    /// The old pattern was split into more specialized patterns
    Specialization,

    /// Major refactoring while preserving functionality
    Refactoring,

    /// External requirements changed (regulations, standards)
    ExternalRequirement,

    /// Community consensus to replace
    CommunityDecision,

    /// Automated detection of superior alternative
    AutoDetected,

    /// Custom reason
    Custom(String),
}

impl SuccessionReason {
    /// Get a human-readable name
    pub fn name(&self) -> &str {
        match self {
            SuccessionReason::BetterPerformance => "Better Performance",
            SuccessionReason::BugFix => "Bug Fix",
            SuccessionReason::SecurityFix => "Security Fix",
            SuccessionReason::ContextEvolution => "Context Evolution",
            SuccessionReason::Consolidation => "Consolidation",
            SuccessionReason::Specialization => "Specialization",
            SuccessionReason::Refactoring => "Refactoring",
            SuccessionReason::ExternalRequirement => "External Requirement",
            SuccessionReason::CommunityDecision => "Community Decision",
            SuccessionReason::AutoDetected => "Auto-Detected",
            SuccessionReason::Custom(_) => "Custom",
        }
    }

    /// Check if this reason indicates urgency
    pub fn is_urgent(&self) -> bool {
        matches!(
            self,
            SuccessionReason::SecurityFix | SuccessionReason::ExternalRequirement
        )
    }
}

/// Status of a succession
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SuccessionStatus {
    /// Succession declared but not yet active
    Pending,

    /// Succession is active, migration in progress
    Active,

    /// Grace period - old pattern still usable but deprecated
    GracePeriod,

    /// Migration complete, old pattern fully retired
    Complete,

    /// Succession was cancelled
    Cancelled,
}

impl SuccessionStatus {
    /// Check if the succession is still in progress
    pub fn is_in_progress(&self) -> bool {
        matches!(
            self,
            SuccessionStatus::Pending | SuccessionStatus::Active | SuccessionStatus::GracePeriod
        )
    }

    /// Check if the succession is final
    pub fn is_final(&self) -> bool {
        matches!(
            self,
            SuccessionStatus::Complete | SuccessionStatus::Cancelled
        )
    }
}

/// A record of one pattern superseding another
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternSuccession {
    /// Unique identifier for this succession
    pub succession_id: u64,

    /// The pattern being replaced (predecessor)
    pub predecessor_id: PatternId,

    /// The pattern replacing it (successor)
    pub successor_id: PatternId,

    /// Why this succession is happening
    pub reason: SuccessionReason,

    /// Detailed explanation
    pub explanation: String,

    /// Current status
    pub status: SuccessionStatus,

    /// When the succession was declared
    pub declared_at: u64,

    /// When the succession became active
    pub activated_at: Option<u64>,

    /// When the grace period ends
    pub grace_period_ends: Option<u64>,

    /// When the succession was completed
    pub completed_at: Option<u64>,

    /// Agent who declared this succession
    pub declared_by: SymthaeaId,

    /// Domains affected by this succession
    pub affected_domains: Vec<DomainId>,

    /// Trust transfer ratio (how much trust to transfer, 0.0-1.0)
    pub trust_transfer_ratio: f32,

    /// Whether to auto-migrate dependents
    pub auto_migrate_dependents: bool,

    /// Notes and comments
    pub notes: Vec<String>,
}

impl PatternSuccession {
    /// Create a new succession record
    pub fn new(
        succession_id: u64,
        predecessor_id: PatternId,
        successor_id: PatternId,
        reason: SuccessionReason,
        declared_by: SymthaeaId,
        timestamp: u64,
    ) -> Self {
        Self {
            succession_id,
            predecessor_id,
            successor_id,
            reason,
            explanation: String::new(),
            status: SuccessionStatus::Pending,
            declared_at: timestamp,
            activated_at: None,
            grace_period_ends: None,
            completed_at: None,
            declared_by,
            affected_domains: Vec::new(),
            trust_transfer_ratio: 0.8, // Default: transfer 80% of trust
            auto_migrate_dependents: true,
            notes: Vec::new(),
        }
    }

    /// Set explanation
    pub fn with_explanation(mut self, explanation: &str) -> Self {
        self.explanation = explanation.to_string();
        self
    }

    /// Set affected domains
    pub fn with_domains(mut self, domains: Vec<DomainId>) -> Self {
        self.affected_domains = domains;
        self
    }

    /// Set trust transfer ratio
    pub fn with_trust_transfer(mut self, ratio: f32) -> Self {
        self.trust_transfer_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Set grace period duration
    pub fn with_grace_period(mut self, duration: u64, timestamp: u64) -> Self {
        self.grace_period_ends = Some(timestamp + duration);
        self
    }

    /// Disable auto-migration
    pub fn without_auto_migration(mut self) -> Self {
        self.auto_migrate_dependents = false;
        self
    }

    /// Activate the succession
    pub fn activate(&mut self, timestamp: u64) {
        self.status = SuccessionStatus::Active;
        self.activated_at = Some(timestamp);
    }

    /// Enter grace period
    pub fn enter_grace_period(&mut self, duration: u64, timestamp: u64) {
        self.status = SuccessionStatus::GracePeriod;
        self.grace_period_ends = Some(timestamp + duration);
    }

    /// Complete the succession
    pub fn complete(&mut self, timestamp: u64) {
        self.status = SuccessionStatus::Complete;
        self.completed_at = Some(timestamp);
    }

    /// Cancel the succession
    pub fn cancel(&mut self, reason: &str) {
        self.status = SuccessionStatus::Cancelled;
        self.notes.push(format!("Cancelled: {}", reason));
    }

    /// Add a note
    pub fn add_note(&mut self, note: &str) {
        self.notes.push(note.to_string());
    }

    /// Check if grace period has expired
    pub fn grace_period_expired(&self, current_time: u64) -> bool {
        if let Some(ends) = self.grace_period_ends {
            current_time >= ends
        } else {
            false
        }
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "Succession #{}: {} -> {} ({:?}) - {}",
            self.succession_id,
            self.predecessor_id,
            self.successor_id,
            self.status,
            self.reason.name()
        )
    }
}

/// A migration plan for moving dependents from predecessor to successor
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MigrationPlan {
    /// The succession this plan is for
    pub succession_id: u64,

    /// Patterns that need to be migrated (dependents of predecessor)
    pub patterns_to_migrate: Vec<PatternId>,

    /// Patterns that have been migrated
    pub migrated_patterns: Vec<PatternId>,

    /// Patterns that failed migration
    pub failed_patterns: Vec<(PatternId, String)>,

    /// Patterns explicitly excluded from migration
    pub excluded_patterns: Vec<PatternId>,

    /// Estimated effort for each pattern
    pub effort_estimates: HashMap<PatternId, MigrationEffort>,

    /// Total migration progress (0.0-1.0)
    pub progress: f32,

    /// When the plan was created
    pub created_at: u64,

    /// When migration started
    pub started_at: Option<u64>,

    /// When migration completed
    pub completed_at: Option<u64>,
}

impl MigrationPlan {
    /// Create a new migration plan
    pub fn new(succession_id: u64, patterns_to_migrate: Vec<PatternId>, timestamp: u64) -> Self {
        Self {
            succession_id,
            patterns_to_migrate,
            migrated_patterns: Vec::new(),
            failed_patterns: Vec::new(),
            excluded_patterns: Vec::new(),
            effort_estimates: HashMap::new(),
            progress: 0.0,
            created_at: timestamp,
            started_at: None,
            completed_at: None,
        }
    }

    /// Start the migration
    pub fn start(&mut self, timestamp: u64) {
        self.started_at = Some(timestamp);
    }

    /// Mark a pattern as migrated
    pub fn mark_migrated(&mut self, pattern_id: PatternId) {
        if let Some(pos) = self.patterns_to_migrate.iter().position(|&p| p == pattern_id) {
            self.patterns_to_migrate.remove(pos);
            self.migrated_patterns.push(pattern_id);
            self.update_progress();
        }
    }

    /// Mark a pattern as failed
    pub fn mark_failed(&mut self, pattern_id: PatternId, reason: &str) {
        if let Some(pos) = self.patterns_to_migrate.iter().position(|&p| p == pattern_id) {
            self.patterns_to_migrate.remove(pos);
            self.failed_patterns.push((pattern_id, reason.to_string()));
            self.update_progress();
        }
    }

    /// Exclude a pattern from migration
    pub fn exclude(&mut self, pattern_id: PatternId) {
        if let Some(pos) = self.patterns_to_migrate.iter().position(|&p| p == pattern_id) {
            self.patterns_to_migrate.remove(pos);
            self.excluded_patterns.push(pattern_id);
            self.update_progress();
        }
    }

    /// Update progress calculation
    fn update_progress(&mut self) {
        let total = self.migrated_patterns.len()
            + self.failed_patterns.len()
            + self.excluded_patterns.len()
            + self.patterns_to_migrate.len();

        if total == 0 {
            self.progress = 1.0;
        } else {
            self.progress = (self.migrated_patterns.len() + self.excluded_patterns.len()) as f32
                / total as f32;
        }
    }

    /// Check if migration is complete
    pub fn is_complete(&self) -> bool {
        self.patterns_to_migrate.is_empty()
    }

    /// Complete the migration
    pub fn complete(&mut self, timestamp: u64) {
        self.completed_at = Some(timestamp);
        self.progress = 1.0;
    }

    /// Get remaining patterns count
    pub fn remaining(&self) -> usize {
        self.patterns_to_migrate.len()
    }

    /// Get success rate
    pub fn success_rate(&self) -> f32 {
        let total = self.migrated_patterns.len() + self.failed_patterns.len();
        if total == 0 {
            1.0
        } else {
            self.migrated_patterns.len() as f32 / total as f32
        }
    }

    /// Get a summary
    pub fn summary(&self) -> String {
        format!(
            "Migration Plan: {}/{} complete ({:.0}%), {} failed, {} excluded",
            self.migrated_patterns.len(),
            self.migrated_patterns.len() + self.patterns_to_migrate.len() + self.failed_patterns.len(),
            self.progress * 100.0,
            self.failed_patterns.len(),
            self.excluded_patterns.len()
        )
    }
}

/// Estimated effort for migrating a pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MigrationEffort {
    /// Trivial - automatic migration possible
    Trivial,

    /// Easy - minor adjustments needed
    Easy,

    /// Medium - significant changes required
    Medium,

    /// Hard - major rework needed
    Hard,

    /// Manual - cannot be automated, requires human intervention
    Manual,
}

impl MigrationEffort {
    /// Get effort as a numeric multiplier
    pub fn multiplier(&self) -> f32 {
        match self {
            MigrationEffort::Trivial => 0.1,
            MigrationEffort::Easy => 0.5,
            MigrationEffort::Medium => 1.0,
            MigrationEffort::Hard => 2.0,
            MigrationEffort::Manual => 5.0,
        }
    }

    /// Check if this can be automated
    pub fn is_automatable(&self) -> bool {
        !matches!(self, MigrationEffort::Manual)
    }
}

/// A migration instruction for a specific pattern
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MigrationInstruction {
    /// Pattern being migrated
    pub pattern_id: PatternId,

    /// From pattern (predecessor)
    pub from_pattern: PatternId,

    /// To pattern (successor)
    pub to_pattern: PatternId,

    /// Steps to perform
    pub steps: Vec<String>,

    /// What changes in the dependency relationship
    pub dependency_changes: Vec<DependencyChange>,

    /// Warnings or caveats
    pub warnings: Vec<String>,

    /// Estimated effort
    pub effort: MigrationEffort,

    /// Whether this is automated or requires manual action
    pub requires_manual_action: bool,
}

impl MigrationInstruction {
    /// Create a new migration instruction
    pub fn new(
        pattern_id: PatternId,
        from_pattern: PatternId,
        to_pattern: PatternId,
        effort: MigrationEffort,
    ) -> Self {
        Self {
            pattern_id,
            from_pattern,
            to_pattern,
            steps: Vec::new(),
            dependency_changes: Vec::new(),
            warnings: Vec::new(),
            effort,
            requires_manual_action: matches!(effort, MigrationEffort::Manual),
        }
    }

    /// Add a step
    pub fn with_step(mut self, step: &str) -> Self {
        self.steps.push(step.to_string());
        self
    }

    /// Add a warning
    pub fn with_warning(mut self, warning: &str) -> Self {
        self.warnings.push(warning.to_string());
        self
    }

    /// Add a dependency change
    pub fn with_dependency_change(mut self, change: DependencyChange) -> Self {
        self.dependency_changes.push(change);
        self
    }

    /// Format as human-readable instructions
    pub fn format(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "Migration: Pattern {} from {} to {}\n",
            self.pattern_id, self.from_pattern, self.to_pattern
        ));
        output.push_str(&format!("Effort: {:?}\n", self.effort));

        if !self.steps.is_empty() {
            output.push_str("\nSteps:\n");
            for (i, step) in self.steps.iter().enumerate() {
                output.push_str(&format!("  {}. {}\n", i + 1, step));
            }
        }

        if !self.warnings.is_empty() {
            output.push_str("\nWarnings:\n");
            for warning in &self.warnings {
                output.push_str(&format!("  ⚠ {}\n", warning));
            }
        }

        if self.requires_manual_action {
            output.push_str("\n⚠ This migration requires manual intervention.\n");
        }

        output
    }
}

/// A change to a dependency relationship
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DependencyChange {
    /// Type of change
    pub change_type: DependencyChangeType,

    /// The dependency being changed
    pub pattern_id: PatternId,

    /// Old value (if applicable)
    pub old_value: Option<PatternId>,

    /// New value (if applicable)
    pub new_value: Option<PatternId>,

    /// Description of the change
    pub description: String,
}

/// Types of dependency changes
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DependencyChangeType {
    /// Replacing one dependency with another
    Replace,

    /// Adding a new dependency
    Add,

    /// Removing a dependency
    Remove,

    /// Changing dependency strength
    StrengthChange,

    /// Changing dependency type
    TypeChange,
}

/// Configuration for the succession manager
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SuccessionConfig {
    /// Default grace period duration
    pub default_grace_period: u64,

    /// Default trust transfer ratio
    pub default_trust_transfer: f32,

    /// Auto-activate successions
    pub auto_activate: bool,

    /// Auto-complete after grace period
    pub auto_complete: bool,

    /// Auto-migrate dependents
    pub auto_migrate: bool,

    /// Minimum successor success rate to allow succession
    pub min_successor_success_rate: f32,

    /// Require manual approval for successions
    pub require_approval: bool,

    /// Maximum simultaneous active successions per pattern
    pub max_active_per_pattern: usize,
}

impl Default for SuccessionConfig {
    fn default() -> Self {
        Self {
            default_grace_period: 7 * 24 * 3600, // 7 days
            default_trust_transfer: 0.8,
            auto_activate: true,
            auto_complete: false, // Require explicit completion
            auto_migrate: true,
            min_successor_success_rate: 0.5,
            require_approval: false,
            max_active_per_pattern: 1,
        }
    }
}

impl SuccessionConfig {
    /// Create a conservative configuration (more manual control)
    pub fn conservative() -> Self {
        Self {
            auto_activate: false,
            auto_complete: false,
            auto_migrate: false,
            require_approval: true,
            default_grace_period: 30 * 24 * 3600, // 30 days
            ..Default::default()
        }
    }

    /// Create an aggressive configuration (faster transitions)
    pub fn aggressive() -> Self {
        Self {
            auto_activate: true,
            auto_complete: true,
            auto_migrate: true,
            require_approval: false,
            default_grace_period: 24 * 3600, // 1 day
            ..Default::default()
        }
    }
}

/// Statistics for the succession manager
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SuccessionStats {
    /// Total successions declared
    pub total_declared: u64,

    /// Currently active successions
    pub active_count: usize,

    /// Completed successions
    pub completed_count: u64,

    /// Cancelled successions
    pub cancelled_count: u64,

    /// Average migration success rate
    pub avg_migration_success_rate: f32,

    /// Most common succession reasons
    pub common_reasons: Vec<(String, u32)>,

    /// Average grace period duration
    pub avg_grace_period: f32,

    /// Patterns with most successions
    pub most_succeeded_patterns: Vec<(PatternId, u32)>,
}

/// The succession manager
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SuccessionManager {
    /// Configuration
    config: SuccessionConfig,

    /// All successions
    successions: Vec<PatternSuccession>,

    /// Migration plans
    migration_plans: HashMap<u64, MigrationPlan>,

    /// Next succession ID
    next_succession_id: u64,

    /// Index: predecessor -> succession IDs
    by_predecessor: HashMap<PatternId, Vec<u64>>,

    /// Index: successor -> succession IDs
    by_successor: HashMap<PatternId, Vec<u64>>,

    /// Statistics
    stats: SuccessionStats,
}

impl Default for SuccessionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SuccessionManager {
    /// Create a new succession manager
    pub fn new() -> Self {
        Self {
            config: SuccessionConfig::default(),
            successions: Vec::new(),
            migration_plans: HashMap::new(),
            next_succession_id: 1,
            by_predecessor: HashMap::new(),
            by_successor: HashMap::new(),
            stats: SuccessionStats::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: SuccessionConfig) -> Self {
        Self {
            config,
            ..Self::new()
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &SuccessionConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: SuccessionConfig) {
        self.config = config;
    }

    /// Declare a new succession
    pub fn declare_succession(
        &mut self,
        predecessor_id: PatternId,
        successor_id: PatternId,
        reason: SuccessionReason,
        declared_by: SymthaeaId,
        timestamp: u64,
    ) -> PatternSuccession {
        let succession = PatternSuccession::new(
            self.next_succession_id,
            predecessor_id,
            successor_id,
            reason,
            declared_by,
            timestamp,
        );

        self.next_succession_id += 1;
        succession
    }

    /// Register a succession
    pub fn register(&mut self, mut succession: PatternSuccession, timestamp: u64) -> u64 {
        let id = succession.succession_id;

        // Auto-activate if configured
        if self.config.auto_activate {
            succession.activate(timestamp);
        }

        // Update indexes
        self.by_predecessor
            .entry(succession.predecessor_id)
            .or_default()
            .push(id);
        self.by_successor
            .entry(succession.successor_id)
            .or_default()
            .push(id);

        // Update stats
        self.stats.total_declared += 1;

        self.successions.push(succession);
        id
    }

    /// Get a succession by ID
    pub fn get_succession(&self, succession_id: u64) -> Option<&PatternSuccession> {
        self.successions.iter().find(|s| s.succession_id == succession_id)
    }

    /// Get a mutable succession by ID
    pub fn get_succession_mut(&mut self, succession_id: u64) -> Option<&mut PatternSuccession> {
        self.successions.iter_mut().find(|s| s.succession_id == succession_id)
    }

    /// Get all successions
    pub fn all_successions(&self) -> &[PatternSuccession] {
        &self.successions
    }

    /// Get active successions
    pub fn active_successions(&self) -> Vec<&PatternSuccession> {
        self.successions.iter().filter(|s| s.status.is_in_progress()).collect()
    }

    /// Get successions for a predecessor pattern
    pub fn successions_from(&self, predecessor_id: PatternId) -> Vec<&PatternSuccession> {
        self.by_predecessor
            .get(&predecessor_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|&id| self.get_succession(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get successions to a successor pattern
    pub fn successions_to(&self, successor_id: PatternId) -> Vec<&PatternSuccession> {
        self.by_successor
            .get(&successor_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|&id| self.get_succession(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get the current successor for a pattern (latest active succession)
    pub fn current_successor(&self, pattern_id: PatternId) -> Option<PatternId> {
        self.successions_from(pattern_id)
            .into_iter()
            .filter(|s| matches!(s.status, SuccessionStatus::Active | SuccessionStatus::GracePeriod | SuccessionStatus::Complete))
            .max_by_key(|s| s.declared_at)
            .map(|s| s.successor_id)
    }

    /// Get the full lineage of a pattern (all predecessors)
    pub fn get_lineage(&self, pattern_id: PatternId) -> Vec<PatternId> {
        let mut lineage = Vec::new();
        let mut current = pattern_id;

        // Find predecessors
        while let Some(succession) = self.successions_to(current)
            .into_iter()
            .filter(|s| s.status == SuccessionStatus::Complete)
            .max_by_key(|s| s.declared_at)
        {
            lineage.push(succession.predecessor_id);
            current = succession.predecessor_id;
        }

        lineage.reverse();
        lineage
    }

    /// Get all descendants of a pattern (all successors recursively)
    pub fn get_descendants(&self, pattern_id: PatternId) -> Vec<PatternId> {
        let mut descendants = Vec::new();
        let mut queue = vec![pattern_id];

        while let Some(current) = queue.pop() {
            for succession in self.successions_from(current) {
                if !descendants.contains(&succession.successor_id) {
                    descendants.push(succession.successor_id);
                    queue.push(succession.successor_id);
                }
            }
        }

        descendants
    }

    /// Create a migration plan for a succession
    pub fn create_migration_plan(
        &mut self,
        succession_id: u64,
        dependents: Vec<PatternId>,
        timestamp: u64,
    ) -> Option<&MigrationPlan> {
        if self.get_succession(succession_id).is_none() {
            return None;
        }

        let plan = MigrationPlan::new(succession_id, dependents, timestamp);
        self.migration_plans.insert(succession_id, plan);
        self.migration_plans.get(&succession_id)
    }

    /// Get a migration plan
    pub fn get_migration_plan(&self, succession_id: u64) -> Option<&MigrationPlan> {
        self.migration_plans.get(&succession_id)
    }

    /// Get a mutable migration plan
    pub fn get_migration_plan_mut(&mut self, succession_id: u64) -> Option<&mut MigrationPlan> {
        self.migration_plans.get_mut(&succession_id)
    }

    /// Generate migration instruction for a pattern
    pub fn generate_migration_instruction(
        &self,
        succession_id: u64,
        pattern_id: PatternId,
    ) -> Option<MigrationInstruction> {
        let succession = self.get_succession(succession_id)?;

        let effort = MigrationEffort::Easy; // Default, would be computed based on pattern analysis

        let mut instruction = MigrationInstruction::new(
            pattern_id,
            succession.predecessor_id,
            succession.successor_id,
            effort,
        );

        instruction.steps.push(format!(
            "Update dependency from pattern {} to pattern {}",
            succession.predecessor_id, succession.successor_id
        ));

        if !succession.explanation.is_empty() {
            instruction.steps.push(format!("Note: {}", succession.explanation));
        }

        instruction.dependency_changes.push(DependencyChange {
            change_type: DependencyChangeType::Replace,
            pattern_id,
            old_value: Some(succession.predecessor_id),
            new_value: Some(succession.successor_id),
            description: format!(
                "Replace dependency on {} with {}",
                succession.predecessor_id, succession.successor_id
            ),
        });

        Some(instruction)
    }

    /// Activate a succession
    pub fn activate(&mut self, succession_id: u64, timestamp: u64) -> bool {
        if let Some(succession) = self.get_succession_mut(succession_id) {
            succession.activate(timestamp);
            true
        } else {
            false
        }
    }

    /// Start grace period for a succession
    pub fn start_grace_period(&mut self, succession_id: u64, timestamp: u64) -> bool {
        let duration = self.config.default_grace_period;
        if let Some(succession) = self.get_succession_mut(succession_id) {
            succession.enter_grace_period(duration, timestamp);
            true
        } else {
            false
        }
    }

    /// Complete a succession
    pub fn complete(&mut self, succession_id: u64, timestamp: u64) -> bool {
        if let Some(succession) = self.get_succession_mut(succession_id) {
            succession.complete(timestamp);
            self.stats.completed_count += 1;
            true
        } else {
            false
        }
    }

    /// Cancel a succession
    pub fn cancel(&mut self, succession_id: u64, reason: &str) -> bool {
        if let Some(succession) = self.get_succession_mut(succession_id) {
            succession.cancel(reason);
            self.stats.cancelled_count += 1;
            true
        } else {
            false
        }
    }

    /// Check and auto-complete successions with expired grace periods
    pub fn process_grace_periods(&mut self, current_time: u64) -> Vec<u64> {
        if !self.config.auto_complete {
            return Vec::new();
        }

        let mut completed = Vec::new();

        for succession in &mut self.successions {
            if succession.status == SuccessionStatus::GracePeriod
                && succession.grace_period_expired(current_time)
            {
                succession.complete(current_time);
                completed.push(succession.succession_id);
                self.stats.completed_count += 1;
            }
        }

        completed
    }

    /// Calculate trust to transfer for a succession
    pub fn calculate_trust_transfer(
        &self,
        succession_id: u64,
        predecessor_trust: f32,
    ) -> Option<f32> {
        let succession = self.get_succession(succession_id)?;
        Some(predecessor_trust * succession.trust_transfer_ratio)
    }

    /// Get statistics
    pub fn stats(&self) -> SuccessionStats {
        let mut stats = self.stats.clone();
        stats.active_count = self.active_successions().len();

        // Calculate average migration success rate
        let migration_rates: Vec<f32> = self.migration_plans.values()
            .filter(|p| p.completed_at.is_some())
            .map(|p| p.success_rate())
            .collect();

        if !migration_rates.is_empty() {
            stats.avg_migration_success_rate =
                migration_rates.iter().sum::<f32>() / migration_rates.len() as f32;
        }

        stats
    }

    /// Generate a summary report
    pub fn summary_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Succession Summary ===\n\n");
        report.push_str(&format!("Total declared: {}\n", self.stats.total_declared));
        report.push_str(&format!("Active: {}\n", self.active_successions().len()));
        report.push_str(&format!("Completed: {}\n", self.stats.completed_count));
        report.push_str(&format!("Cancelled: {}\n", self.stats.cancelled_count));

        let active = self.active_successions();
        if !active.is_empty() {
            report.push_str("\nActive Successions:\n");
            for succession in active {
                report.push_str(&format!("  {}\n", succession.summary()));
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_succession_reason_properties() {
        assert!(SuccessionReason::SecurityFix.is_urgent());
        assert!(SuccessionReason::ExternalRequirement.is_urgent());
        assert!(!SuccessionReason::BetterPerformance.is_urgent());
    }

    #[test]
    fn test_succession_status_properties() {
        assert!(SuccessionStatus::Active.is_in_progress());
        assert!(SuccessionStatus::GracePeriod.is_in_progress());
        assert!(SuccessionStatus::Complete.is_final());
        assert!(SuccessionStatus::Cancelled.is_final());
    }

    #[test]
    fn test_pattern_succession_creation() {
        let succession = PatternSuccession::new(
            1,
            10, // predecessor
            20, // successor
            SuccessionReason::BetterPerformance,
            99, // declared_by
            1000,
        )
        .with_explanation("New pattern is faster")
        .with_trust_transfer(0.9);

        assert_eq!(succession.predecessor_id, 10);
        assert_eq!(succession.successor_id, 20);
        assert_eq!(succession.trust_transfer_ratio, 0.9);
        assert_eq!(succession.status, SuccessionStatus::Pending);
    }

    #[test]
    fn test_succession_lifecycle() {
        let mut succession = PatternSuccession::new(
            1,
            10,
            20,
            SuccessionReason::BugFix,
            99,
            1000,
        );

        assert_eq!(succession.status, SuccessionStatus::Pending);

        succession.activate(1100);
        assert_eq!(succession.status, SuccessionStatus::Active);
        assert_eq!(succession.activated_at, Some(1100));

        succession.enter_grace_period(3600, 1200);
        assert_eq!(succession.status, SuccessionStatus::GracePeriod);
        assert_eq!(succession.grace_period_ends, Some(1200 + 3600));

        assert!(!succession.grace_period_expired(1300));
        assert!(succession.grace_period_expired(5000));

        succession.complete(5000);
        assert_eq!(succession.status, SuccessionStatus::Complete);
    }

    #[test]
    fn test_migration_plan() {
        let mut plan = MigrationPlan::new(1, vec![1, 2, 3, 4, 5], 1000);

        assert_eq!(plan.remaining(), 5);
        assert!(!plan.is_complete());
        assert_eq!(plan.progress, 0.0);

        plan.mark_migrated(1);
        plan.mark_migrated(2);
        assert_eq!(plan.remaining(), 3);
        assert!((plan.progress - 0.4).abs() < 0.01);

        plan.mark_failed(3, "Conflict detected");
        assert_eq!(plan.failed_patterns.len(), 1);

        plan.exclude(4);
        plan.mark_migrated(5);
        assert!(plan.is_complete());
    }

    #[test]
    fn test_migration_instruction() {
        let instruction = MigrationInstruction::new(1, 10, 20, MigrationEffort::Easy)
            .with_step("Update import statement")
            .with_step("Adjust parameters")
            .with_warning("Check for breaking changes");

        assert_eq!(instruction.steps.len(), 2);
        assert_eq!(instruction.warnings.len(), 1);
        assert!(!instruction.requires_manual_action);

        let manual = MigrationInstruction::new(1, 10, 20, MigrationEffort::Manual);
        assert!(manual.requires_manual_action);
    }

    #[test]
    fn test_succession_manager_basic() {
        let mut manager = SuccessionManager::new();

        let succession = manager.declare_succession(
            10, 20,
            SuccessionReason::BetterPerformance,
            99,
            1000,
        );

        let id = manager.register(succession, 1000);
        assert_eq!(id, 1);

        let retrieved = manager.get_succession(1);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().predecessor_id, 10);
    }

    #[test]
    fn test_succession_manager_lineage() {
        let mut manager = SuccessionManager::new();

        // Create chain: 1 -> 2 -> 3
        let s1 = manager.declare_succession(1, 2, SuccessionReason::BetterPerformance, 99, 1000);
        manager.register(s1, 1000);
        manager.complete(1, 2000);

        let s2 = manager.declare_succession(2, 3, SuccessionReason::BetterPerformance, 99, 2000);
        manager.register(s2, 2000);
        manager.complete(2, 3000);

        let lineage = manager.get_lineage(3);
        assert_eq!(lineage, vec![1, 2]);

        let descendants = manager.get_descendants(1);
        assert!(descendants.contains(&2));
        assert!(descendants.contains(&3));
    }

    #[test]
    fn test_succession_manager_current_successor() {
        let mut manager = SuccessionManager::new();

        let s1 = manager.declare_succession(1, 2, SuccessionReason::BugFix, 99, 1000);
        manager.register(s1, 1000);

        let successor = manager.current_successor(1);
        assert_eq!(successor, Some(2));

        // Pattern 3 has no successor
        let no_successor = manager.current_successor(3);
        assert_eq!(no_successor, None);
    }

    #[test]
    fn test_succession_manager_grace_period() {
        let mut manager = SuccessionManager::with_config(SuccessionConfig {
            auto_complete: true,
            default_grace_period: 1000,
            ..Default::default()
        });

        let succession = manager.declare_succession(1, 2, SuccessionReason::BugFix, 99, 1000);
        let id = manager.register(succession, 1000);

        manager.start_grace_period(id, 2000);

        // Before grace period ends
        let completed = manager.process_grace_periods(2500);
        assert!(completed.is_empty());

        // After grace period ends
        let completed = manager.process_grace_periods(3500);
        assert_eq!(completed, vec![id]);

        let succession = manager.get_succession(id).unwrap();
        assert_eq!(succession.status, SuccessionStatus::Complete);
    }

    #[test]
    fn test_trust_transfer_calculation() {
        let mut manager = SuccessionManager::new();

        let succession = manager.declare_succession(1, 2, SuccessionReason::BugFix, 99, 1000)
            .with_trust_transfer(0.75);
        let id = manager.register(succession, 1000);

        let transferred = manager.calculate_trust_transfer(id, 0.8);
        assert_eq!(transferred, Some(0.6)); // 0.8 * 0.75 = 0.6
    }

    #[test]
    fn test_succession_config_presets() {
        let conservative = SuccessionConfig::conservative();
        assert!(conservative.require_approval);
        assert!(!conservative.auto_activate);

        let aggressive = SuccessionConfig::aggressive();
        assert!(!aggressive.require_approval);
        assert!(aggressive.auto_complete);
    }
}
