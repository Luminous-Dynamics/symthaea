// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pattern Dependencies & Prerequisites
//!
//! Component 16 tracks relationships between patterns:
//! - Prerequisites: Pattern A must be applied before Pattern B
//! - Dependencies: Pattern A requires Pattern B to function correctly
//! - Conflicts: Pattern A and Pattern B cannot be used together
//! - Enhancements: Pattern A works better when combined with Pattern B

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::collections::{HashMap, HashSet};

use super::PatternId;

// ==============================================================================
// COMPONENT 16: PATTERN DEPENDENCIES & PREREQUISITES
// ==============================================================================
//
// This component tracks relationships between patterns:
// - Prerequisites: Pattern A must be applied before Pattern B
// - Dependencies: Pattern A requires Pattern B to function correctly
// - Conflicts: Pattern A and Pattern B cannot be used together
// - Enhancements: Pattern A works better when combined with Pattern B
//
// This enables:
// - Dependency resolution when applying patterns
// - Conflict detection before pattern application
// - Optimal pattern ordering suggestions
// - Impact analysis when deprecating patterns

/// Type of relationship between two patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PatternRelationType {
    /// Pattern A must be applied before Pattern B (temporal ordering)
    Prerequisite,
    /// Pattern A requires Pattern B to function correctly (functional dependency)
    Requires,
    /// Pattern A and Pattern B cannot be used together
    ConflictsWith,
    /// Pattern A works better when combined with Pattern B
    EnhancedBy,
    /// Pattern A is a specialization of Pattern B
    Specializes,
    /// Pattern A generalizes Pattern B
    Generalizes,
    /// Pattern A replaces/supersedes Pattern B
    Supersedes,
    /// Patterns are complementary (work well together but neither requires the other)
    ComplementaryWith,
}

impl PatternRelationType {
    /// Check if this relation is blocking (prevents usage)
    pub fn is_blocking(&self) -> bool {
        matches!(self, PatternRelationType::ConflictsWith)
    }

    /// Check if this relation requires resolution before use
    pub fn requires_resolution(&self) -> bool {
        matches!(
            self,
            PatternRelationType::Prerequisite | PatternRelationType::Requires
        )
    }

    /// Check if this is a positive relationship
    pub fn is_positive(&self) -> bool {
        matches!(
            self,
            PatternRelationType::EnhancedBy
                | PatternRelationType::ComplementaryWith
                | PatternRelationType::Specializes
                | PatternRelationType::Generalizes
        )
    }

    /// Get the inverse relation type (if applicable)
    pub fn inverse(&self) -> Option<PatternRelationType> {
        match self {
            PatternRelationType::Prerequisite => None, // Not symmetric
            PatternRelationType::Requires => None,     // Not symmetric
            PatternRelationType::ConflictsWith => Some(PatternRelationType::ConflictsWith),
            PatternRelationType::EnhancedBy => None, // Not symmetric
            PatternRelationType::Specializes => Some(PatternRelationType::Generalizes),
            PatternRelationType::Generalizes => Some(PatternRelationType::Specializes),
            PatternRelationType::Supersedes => None, // Not symmetric
            PatternRelationType::ComplementaryWith => Some(PatternRelationType::ComplementaryWith),
        }
    }
}

/// Strength/confidence of a dependency relationship
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DependencyStrength {
    /// Absolute requirement - cannot function without
    Required,
    /// Strong recommendation - significantly better with
    Strong,
    /// Moderate suggestion - somewhat better with
    Moderate,
    /// Weak hint - slightly better with
    Weak,
    /// Discovered automatically (confidence varies)
    Discovered { confidence: f32 },
}

impl DependencyStrength {
    /// Get numeric weight for this strength
    pub fn weight(&self) -> f32 {
        match self {
            DependencyStrength::Required => 1.0,
            DependencyStrength::Strong => 0.8,
            DependencyStrength::Moderate => 0.5,
            DependencyStrength::Weak => 0.2,
            DependencyStrength::Discovered { confidence } => *confidence,
        }
    }

    /// Check if this is a hard requirement
    pub fn is_required(&self) -> bool {
        matches!(self, DependencyStrength::Required)
    }
}

impl Default for DependencyStrength {
    fn default() -> Self {
        DependencyStrength::Moderate
    }
}

/// A single dependency relationship between two patterns
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternDependency {
    /// The pattern that has the dependency (source)
    pub from_pattern: PatternId,
    /// The pattern being depended on (target)
    pub to_pattern: PatternId,
    /// Type of relationship
    pub relation_type: PatternRelationType,
    /// Strength of the dependency
    pub strength: DependencyStrength,
    /// When this dependency was established
    pub created_at: u64,
    /// Who/what established this dependency
    pub created_by: String,
    /// Optional description of why this dependency exists
    pub reason: Option<String>,
    /// Whether this was discovered automatically
    pub is_discovered: bool,
    /// Number of times this dependency has been validated in practice
    pub validation_count: u32,
    /// Success rate when dependency is satisfied
    pub satisfied_success_rate: f32,
    /// Success rate when dependency is not satisfied
    pub unsatisfied_success_rate: f32,
}

impl PatternDependency {
    /// Create a new dependency
    pub fn new(
        from_pattern: PatternId,
        to_pattern: PatternId,
        relation_type: PatternRelationType,
        strength: DependencyStrength,
        timestamp: u64,
        creator: impl Into<String>,
    ) -> Self {
        Self {
            from_pattern,
            to_pattern,
            relation_type,
            strength,
            created_at: timestamp,
            created_by: creator.into(),
            reason: None,
            is_discovered: false,
            validation_count: 0,
            satisfied_success_rate: 0.0,
            unsatisfied_success_rate: 0.0,
        }
    }

    /// Create a discovered dependency
    pub fn discovered(
        from_pattern: PatternId,
        to_pattern: PatternId,
        relation_type: PatternRelationType,
        confidence: f32,
        timestamp: u64,
    ) -> Self {
        Self {
            from_pattern,
            to_pattern,
            relation_type,
            strength: DependencyStrength::Discovered { confidence },
            created_at: timestamp,
            created_by: "auto_discovery".to_string(),
            reason: None,
            is_discovered: true,
            validation_count: 0,
            satisfied_success_rate: 0.0,
            unsatisfied_success_rate: 0.0,
        }
    }

    /// Add a reason for this dependency
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Record a validation of this dependency
    pub fn record_validation(&mut self, was_satisfied: bool, was_successful: bool) {
        self.validation_count += 1;

        // Update success rates using exponential moving average
        let alpha = 0.1;
        let outcome = if was_successful { 1.0 } else { 0.0 };

        if was_satisfied {
            self.satisfied_success_rate =
                self.satisfied_success_rate * (1.0 - alpha) + outcome * alpha;
        } else {
            self.unsatisfied_success_rate =
                self.unsatisfied_success_rate * (1.0 - alpha) + outcome * alpha;
        }
    }

    /// Calculate the importance of this dependency based on success rate difference
    pub fn importance(&self) -> f32 {
        if self.validation_count < 5 {
            return self.strength.weight();
        }

        // Higher importance if satisfied success rate >> unsatisfied success rate
        let diff = self.satisfied_success_rate - self.unsatisfied_success_rate;
        (diff.max(0.0) + self.strength.weight()) / 2.0
    }
}

/// Configuration for dependency management
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DependencyConfig {
    /// Enable automatic dependency discovery
    pub auto_discover: bool,
    /// Minimum co-occurrence count before suggesting dependency
    pub discovery_threshold: u32,
    /// Minimum confidence for auto-discovered dependencies
    pub min_discovery_confidence: f32,
    /// Whether to enforce required dependencies
    pub enforce_required: bool,
    /// Whether to warn about conflicts
    pub warn_on_conflicts: bool,
    /// Maximum depth for transitive dependency resolution
    pub max_resolution_depth: usize,
}

impl Default for DependencyConfig {
    fn default() -> Self {
        Self {
            auto_discover: true,
            discovery_threshold: 10,
            min_discovery_confidence: 0.7,
            enforce_required: true,
            warn_on_conflicts: true,
            max_resolution_depth: 10,
        }
    }
}

/// Result of dependency resolution
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DependencyResolution {
    /// The pattern being resolved
    pub pattern_id: PatternId,
    /// Ordered list of patterns that must be applied first
    pub prerequisites: Vec<PatternId>,
    /// Required patterns that must be present
    pub required: Vec<PatternId>,
    /// Patterns that conflict with this one
    pub conflicts: Vec<PatternId>,
    /// Patterns that would enhance this one
    pub enhancements: Vec<PatternId>,
    /// Whether resolution was successful
    pub is_resolved: bool,
    /// Any issues found during resolution
    pub issues: Vec<DependencyIssue>,
}

impl DependencyResolution {
    /// Create a new resolution
    pub fn new(pattern_id: PatternId) -> Self {
        Self {
            pattern_id,
            prerequisites: Vec::new(),
            required: Vec::new(),
            conflicts: Vec::new(),
            enhancements: Vec::new(),
            is_resolved: true,
            issues: Vec::new(),
        }
    }

    /// Check if there are any blocking issues
    pub fn has_blocking_issues(&self) -> bool {
        self.issues.iter().any(|i| i.is_blocking)
    }

    /// Get all patterns needed before applying this one
    pub fn all_prerequisites(&self) -> Vec<PatternId> {
        let mut all = self.prerequisites.clone();
        all.extend(self.required.iter().copied());
        all.sort();
        all.dedup();
        all
    }
}

/// An issue found during dependency resolution
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DependencyIssue {
    /// Type of issue
    pub issue_type: DependencyIssueType,
    /// Patterns involved
    pub patterns_involved: Vec<PatternId>,
    /// Description of the issue
    pub description: String,
    /// Whether this issue blocks pattern application
    pub is_blocking: bool,
}

/// Types of dependency issues
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DependencyIssueType {
    /// Circular dependency detected
    CircularDependency,
    /// Missing required dependency
    MissingRequired,
    /// Conflict with another pattern
    Conflict,
    /// Prerequisite not met
    UnmetPrerequisite,
    /// Deprecated dependency
    DeprecatedDependency,
    /// Resolution depth exceeded
    DepthExceeded,
}

/// Registry for tracking pattern dependencies
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternDependencyRegistry {
    /// All registered dependencies
    dependencies: Vec<PatternDependency>,
    /// Index: from_pattern -> list of dependency indices
    from_index: HashMap<PatternId, Vec<usize>>,
    /// Index: to_pattern -> list of dependency indices (reverse lookup)
    to_index: HashMap<PatternId, Vec<usize>>,
    /// Configuration
    pub config: DependencyConfig,
    /// Co-occurrence tracking for auto-discovery
    cooccurrences: HashMap<(PatternId, PatternId), CooccurrenceData>,
}

/// Data for tracking co-occurrences
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct CooccurrenceData {
    /// Times both patterns were used together
    together_count: u32,
    /// Times first pattern was used without second (reserved for future use)
    _first_without_second: u32,
    /// Times second pattern was used without first (reserved for future use)
    _second_without_first: u32,
    /// Success rate when used together
    together_success_rate: f32,
    /// Success rate when first used alone
    first_alone_success_rate: f32,
}

impl PatternDependencyRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            dependencies: Vec::new(),
            from_index: HashMap::new(),
            to_index: HashMap::new(),
            config: DependencyConfig::default(),
            cooccurrences: HashMap::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: DependencyConfig) -> Self {
        Self {
            dependencies: Vec::new(),
            from_index: HashMap::new(),
            to_index: HashMap::new(),
            config,
            cooccurrences: HashMap::new(),
        }
    }

    /// Add a dependency
    pub fn add_dependency(&mut self, dependency: PatternDependency) {
        // Check if already exists
        if self.has_dependency(
            dependency.from_pattern,
            dependency.to_pattern,
            dependency.relation_type,
        ) {
            return;
        }

        let idx = self.dependencies.len();
        self.from_index
            .entry(dependency.from_pattern)
            .or_default()
            .push(idx);
        self.to_index
            .entry(dependency.to_pattern)
            .or_default()
            .push(idx);
        self.dependencies.push(dependency);
    }

    /// Check if a dependency exists
    pub fn has_dependency(
        &self,
        from: PatternId,
        to: PatternId,
        relation_type: PatternRelationType,
    ) -> bool {
        self.from_index
            .get(&from)
            .map(|indices| {
                indices.iter().any(|&i| {
                    let d = &self.dependencies[i];
                    d.to_pattern == to && d.relation_type == relation_type
                })
            })
            .unwrap_or(false)
    }

    /// Get all dependencies from a pattern
    pub fn dependencies_from(&self, pattern_id: PatternId) -> Vec<&PatternDependency> {
        self.from_index
            .get(&pattern_id)
            .map(|indices| indices.iter().map(|&i| &self.dependencies[i]).collect())
            .unwrap_or_default()
    }

    /// Get all dependencies to a pattern (reverse lookup)
    pub fn dependencies_to(&self, pattern_id: PatternId) -> Vec<&PatternDependency> {
        self.to_index
            .get(&pattern_id)
            .map(|indices| indices.iter().map(|&i| &self.dependencies[i]).collect())
            .unwrap_or_default()
    }

    /// Get dependencies of a specific type from a pattern
    pub fn dependencies_of_type(
        &self,
        pattern_id: PatternId,
        relation_type: PatternRelationType,
    ) -> Vec<&PatternDependency> {
        self.dependencies_from(pattern_id)
            .into_iter()
            .filter(|d| d.relation_type == relation_type)
            .collect()
    }

    /// Get all patterns that this pattern requires
    pub fn required_patterns(&self, pattern_id: PatternId) -> Vec<PatternId> {
        self.dependencies_from(pattern_id)
            .iter()
            .filter(|d| d.relation_type == PatternRelationType::Requires)
            .map(|d| d.to_pattern)
            .collect()
    }

    /// Get all patterns that this pattern conflicts with
    pub fn conflicting_patterns(&self, pattern_id: PatternId) -> Vec<PatternId> {
        self.dependencies_from(pattern_id)
            .iter()
            .filter(|d| d.relation_type == PatternRelationType::ConflictsWith)
            .map(|d| d.to_pattern)
            .collect()
    }

    /// Get all prerequisites for a pattern
    pub fn prerequisites(&self, pattern_id: PatternId) -> Vec<PatternId> {
        self.dependencies_from(pattern_id)
            .iter()
            .filter(|d| d.relation_type == PatternRelationType::Prerequisite)
            .map(|d| d.to_pattern)
            .collect()
    }

    /// Resolve all dependencies for a pattern (including transitive)
    pub fn resolve(
        &self,
        pattern_id: PatternId,
        active_patterns: &[PatternId],
    ) -> DependencyResolution {
        let mut resolution = DependencyResolution::new(pattern_id);
        let mut visited = HashSet::new();
        let mut stack = vec![pattern_id];
        let mut depth = 0;

        while let Some(current) = stack.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);
            depth += 1;

            if depth > self.config.max_resolution_depth {
                resolution.issues.push(DependencyIssue {
                    issue_type: DependencyIssueType::DepthExceeded,
                    patterns_involved: vec![pattern_id],
                    description: format!(
                        "Dependency resolution exceeded max depth of {}",
                        self.config.max_resolution_depth
                    ),
                    is_blocking: false,
                });
                resolution.is_resolved = false;
                break;
            }

            for dep in self.dependencies_from(current) {
                match dep.relation_type {
                    PatternRelationType::Prerequisite => {
                        if !resolution.prerequisites.contains(&dep.to_pattern) {
                            resolution.prerequisites.push(dep.to_pattern);
                            stack.push(dep.to_pattern);
                        }
                    }
                    PatternRelationType::Requires => {
                        if !resolution.required.contains(&dep.to_pattern) {
                            resolution.required.push(dep.to_pattern);
                            stack.push(dep.to_pattern);
                        }
                        // Check if required pattern is active
                        if self.config.enforce_required
                            && !active_patterns.contains(&dep.to_pattern)
                        {
                            resolution.issues.push(DependencyIssue {
                                issue_type: DependencyIssueType::MissingRequired,
                                patterns_involved: vec![current, dep.to_pattern],
                                description: format!(
                                    "Pattern {} requires pattern {} which is not active",
                                    current, dep.to_pattern
                                ),
                                is_blocking: dep.strength.is_required(),
                            });
                        }
                    }
                    PatternRelationType::ConflictsWith => {
                        if !resolution.conflicts.contains(&dep.to_pattern) {
                            resolution.conflicts.push(dep.to_pattern);
                        }
                        // Check if conflicting pattern is active
                        if self.config.warn_on_conflicts
                            && active_patterns.contains(&dep.to_pattern)
                        {
                            resolution.issues.push(DependencyIssue {
                                issue_type: DependencyIssueType::Conflict,
                                patterns_involved: vec![current, dep.to_pattern],
                                description: format!(
                                    "Pattern {} conflicts with active pattern {}",
                                    current, dep.to_pattern
                                ),
                                is_blocking: true,
                            });
                        }
                    }
                    PatternRelationType::EnhancedBy | PatternRelationType::ComplementaryWith => {
                        if !resolution.enhancements.contains(&dep.to_pattern) {
                            resolution.enhancements.push(dep.to_pattern);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Check for circular dependencies
        if self.has_circular_dependency(pattern_id) {
            resolution.issues.push(DependencyIssue {
                issue_type: DependencyIssueType::CircularDependency,
                patterns_involved: vec![pattern_id],
                description: "Circular dependency detected".to_string(),
                is_blocking: true,
            });
            resolution.is_resolved = false;
        }

        // Sort prerequisites by depth (deepest first)
        resolution.prerequisites.reverse();

        resolution
    }

    /// Check if a pattern has a circular dependency
    pub fn has_circular_dependency(&self, pattern_id: PatternId) -> bool {
        let mut visited = HashSet::new();
        let mut stack = HashSet::new();
        self.detect_cycle(pattern_id, &mut visited, &mut stack)
    }

    fn detect_cycle(
        &self,
        pattern_id: PatternId,
        visited: &mut HashSet<PatternId>,
        stack: &mut HashSet<PatternId>,
    ) -> bool {
        if stack.contains(&pattern_id) {
            return true;
        }
        if visited.contains(&pattern_id) {
            return false;
        }

        visited.insert(pattern_id);
        stack.insert(pattern_id);

        for dep in self.dependencies_from(pattern_id) {
            if dep.relation_type.requires_resolution()
                && self.detect_cycle(dep.to_pattern, visited, stack)
            {
                return true;
            }
        }

        stack.remove(&pattern_id);
        false
    }

    /// Record pattern usage for auto-discovery
    pub fn record_usage(
        &mut self,
        patterns_used: &[PatternId],
        was_successful: bool,
        timestamp: u64,
    ) {
        if !self.config.auto_discover || patterns_used.len() < 2 {
            return;
        }

        // Update co-occurrence data for all pairs
        for i in 0..patterns_used.len() {
            for j in (i + 1)..patterns_used.len() {
                let key = if patterns_used[i] < patterns_used[j] {
                    (patterns_used[i], patterns_used[j])
                } else {
                    (patterns_used[j], patterns_used[i])
                };

                let data = self.cooccurrences.entry(key).or_default();
                data.together_count += 1;

                let alpha = 0.1;
                let outcome = if was_successful { 1.0 } else { 0.0 };
                data.together_success_rate =
                    data.together_success_rate * (1.0 - alpha) + outcome * alpha;
            }
        }

        // Check for potential dependencies to discover
        self.try_discover_dependencies(timestamp);
    }

    /// Try to discover new dependencies from co-occurrence data
    fn try_discover_dependencies(&mut self, timestamp: u64) {
        let threshold = self.config.discovery_threshold;
        let min_confidence = self.config.min_discovery_confidence;

        let discoveries: Vec<_> = self
            .cooccurrences
            .iter()
            .filter_map(|(&(p1, p2), data)| {
                if data.together_count < threshold {
                    return None;
                }

                // Check if patterns are significantly better together
                let improvement = data.together_success_rate - data.first_alone_success_rate;
                if improvement > min_confidence {
                    // p1 is enhanced by p2
                    Some(PatternDependency::discovered(
                        p1,
                        p2,
                        PatternRelationType::EnhancedBy,
                        improvement,
                        timestamp,
                    ))
                } else {
                    None
                }
            })
            .collect();

        for dep in discoveries {
            if !self.has_dependency(dep.from_pattern, dep.to_pattern, dep.relation_type) {
                self.add_dependency(dep);
            }
        }
    }

    /// Get patterns that depend on a given pattern
    pub fn dependents(&self, pattern_id: PatternId) -> Vec<PatternId> {
        self.dependencies_to(pattern_id)
            .iter()
            .filter(|d| d.relation_type.requires_resolution())
            .map(|d| d.from_pattern)
            .collect()
    }

    /// Get impact analysis if a pattern were to be removed/deprecated
    pub fn deprecation_impact(&self, pattern_id: PatternId) -> Vec<PatternId> {
        let mut affected = Vec::new();
        let mut stack = vec![pattern_id];
        let mut visited = HashSet::new();

        while let Some(current) = stack.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            for dep in self.dependencies_to(current) {
                if dep.relation_type.requires_resolution() {
                    affected.push(dep.from_pattern);
                    stack.push(dep.from_pattern);
                }
            }
        }

        affected.sort();
        affected.dedup();
        affected
    }

    /// Remove all dependencies for a pattern
    pub fn remove_pattern(&mut self, pattern_id: PatternId) {
        // Mark indices to remove
        let indices_to_remove: Vec<usize> = self
            .dependencies
            .iter()
            .enumerate()
            .filter(|(_, d)| d.from_pattern == pattern_id || d.to_pattern == pattern_id)
            .map(|(i, _)| i)
            .collect();

        // Remove in reverse order to maintain valid indices
        for idx in indices_to_remove.into_iter().rev() {
            let dep = self.dependencies.remove(idx);

            // Update indices (this is O(n) but keeps things simple)
            self.rebuild_indices();

            // Remove from cooccurrence tracking
            self.cooccurrences
                .retain(|&(p1, p2), _| p1 != dep.from_pattern && p2 != dep.to_pattern);
        }
    }

    fn rebuild_indices(&mut self) {
        self.from_index.clear();
        self.to_index.clear();

        for (idx, dep) in self.dependencies.iter().enumerate() {
            self.from_index
                .entry(dep.from_pattern)
                .or_default()
                .push(idx);
            self.to_index.entry(dep.to_pattern).or_default().push(idx);
        }
    }

    /// Get statistics about dependencies
    pub fn stats(&self) -> DependencyStats {
        let mut relation_counts = HashMap::new();
        for dep in &self.dependencies {
            *relation_counts.entry(dep.relation_type).or_insert(0) += 1;
        }

        let discovered_count = self.dependencies.iter().filter(|d| d.is_discovered).count();

        DependencyStats {
            total_dependencies: self.dependencies.len(),
            total_patterns: self.from_index.len(),
            discovered_dependencies: discovered_count,
            manual_dependencies: self.dependencies.len() - discovered_count,
            relation_counts,
            avg_dependencies_per_pattern: if self.from_index.is_empty() {
                0.0
            } else {
                self.dependencies.len() as f32 / self.from_index.len() as f32
            },
        }
    }

    /// Get all dependencies
    pub fn all_dependencies(&self) -> &[PatternDependency] {
        &self.dependencies
    }
}

impl Default for PatternDependencyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about dependencies
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DependencyStats {
    /// Total number of dependencies
    pub total_dependencies: usize,
    /// Total patterns with dependencies
    pub total_patterns: usize,
    /// Dependencies discovered automatically
    pub discovered_dependencies: usize,
    /// Dependencies added manually
    pub manual_dependencies: usize,
    /// Count by relation type
    pub relation_counts: HashMap<PatternRelationType, usize>,
    /// Average dependencies per pattern
    pub avg_dependencies_per_pattern: f32,
}
