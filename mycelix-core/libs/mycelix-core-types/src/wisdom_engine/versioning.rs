//! Pattern Versioning & Evolution
//!
//! Component 15 provides version control for patterns as they evolve.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use super::{PatternId, SymthaeaId};

// ==============================================================================
// COMPONENT 15: PATTERN VERSIONING & EVOLUTION
// ==============================================================================
//
// This component enables patterns to evolve over time while maintaining their
// history. Key features:
//
// 1. SEMANTIC VERSIONING - Patterns have major.minor.patch versions
// 2. EVOLUTION TRACKING - Record why and how patterns change
// 3. BRANCHING - Create experimental variants of patterns
// 4. LINEAGE - Track parent-child relationships between versions
// 5. ROLLBACK - Ability to revert to previous versions
//
// This allows the system to:
// - Improve patterns based on feedback while keeping history
// - Experiment with variations without affecting production
// - Understand how patterns have evolved over time
// - Maintain audit trails for pattern changes

/// Semantic version for patterns (major.minor.patch)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternVersion {
    /// Major version - breaking changes or fundamental shifts
    pub major: u16,
    /// Minor version - new capabilities, backward compatible
    pub minor: u16,
    /// Patch version - bug fixes, refinements
    pub patch: u16,
}

impl PatternVersion {
    /// Create a new version
    pub fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Initial version (1.0.0)
    pub fn initial() -> Self {
        Self {
            major: 1,
            minor: 0,
            patch: 0,
        }
    }

    /// Increment major version (resets minor and patch)
    pub fn bump_major(&self) -> Self {
        Self {
            major: self.major + 1,
            minor: 0,
            patch: 0,
        }
    }

    /// Increment minor version (resets patch)
    pub fn bump_minor(&self) -> Self {
        Self {
            major: self.major,
            minor: self.minor + 1,
            patch: 0,
        }
    }

    /// Increment patch version
    pub fn bump_patch(&self) -> Self {
        Self {
            major: self.major,
            minor: self.minor,
            patch: self.patch + 1,
        }
    }

    /// Check if this version is newer than another
    pub fn is_newer_than(&self, other: &PatternVersion) -> bool {
        if self.major != other.major {
            return self.major > other.major;
        }
        if self.minor != other.minor {
            return self.minor > other.minor;
        }
        self.patch > other.patch
    }

    /// Check compatibility (same major version)
    pub fn is_compatible_with(&self, other: &PatternVersion) -> bool {
        self.major == other.major
    }

    /// Format as string "major.minor.patch"
    pub fn to_string(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl core::fmt::Display for PatternVersion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl PartialOrd for PatternVersion {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PatternVersion {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        match self.major.cmp(&other.major) {
            core::cmp::Ordering::Equal => match self.minor.cmp(&other.minor) {
                core::cmp::Ordering::Equal => self.patch.cmp(&other.patch),
                other => other,
            },
            other => other,
        }
    }
}

/// Reason why a pattern was evolved/updated
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PatternEvolutionReason {
    /// Initial creation
    Initial,
    /// Performance improvement based on feedback
    PerformanceImprovement {
        old_success_rate: f32,
        new_success_rate: f32,
    },
    /// Bug fix in the pattern
    BugFix { issue_description: String },
    /// Refinement based on usage patterns
    Refinement { refinement_details: String },
    /// Generalization to handle more cases
    Generalization { new_domains: Vec<String> },
    /// Specialization for specific use case
    Specialization { target_domain: String },
    /// Merge of multiple pattern variants
    Merge { merged_from: Vec<PatternId> },
    /// Split from a more general pattern
    Split { split_from: PatternId },
    /// Community contribution/suggestion
    CommunityContribution {
        contributor_id: SymthaeaId,
        description: String,
    },
    /// Automatic evolution by the system
    AutoEvolution { trigger: String },
    /// Rollback to a previous version
    Rollback {
        rolled_back_from: PatternVersion,
        reason: String,
    },
    /// Branch creation for experimentation
    BranchCreation { branch_name: String },
    /// Branch merge back to main
    BranchMerge { branch_name: String },
}

/// Information about a specific version of a pattern
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternVersionInfo {
    /// The version number
    pub version: PatternVersion,
    /// Pattern ID this version belongs to
    pub pattern_id: PatternId,
    /// When this version was created
    pub created_at: u64,
    /// Who created this version
    pub created_by: String,
    /// Why this version was created
    pub evolution_reason: PatternEvolutionReason,
    /// Parent version (None for initial version)
    pub parent_version: Option<PatternVersion>,
    /// Success rate at time of versioning
    pub success_rate_snapshot: f32,
    /// Usage count at time of versioning
    pub usage_count_snapshot: u64,
    /// Solution text at this version
    pub solution_snapshot: String,
    /// Whether this version is the current active version
    pub is_current: bool,
    /// Whether this version is on a branch
    pub branch_name: Option<String>,
    /// Tags for this version
    pub tags: Vec<String>,
    /// Changelog/notes for this version
    pub changelog: Option<String>,
}

impl PatternVersionInfo {
    /// Create a new version info for initial version
    pub fn initial(
        pattern_id: PatternId,
        solution: String,
        timestamp: u64,
        creator: impl Into<String>,
    ) -> Self {
        Self {
            version: PatternVersion::initial(),
            pattern_id,
            created_at: timestamp,
            created_by: creator.into(),
            evolution_reason: PatternEvolutionReason::Initial,
            parent_version: None,
            success_rate_snapshot: 0.0,
            usage_count_snapshot: 0,
            solution_snapshot: solution,
            is_current: true,
            branch_name: None,
            tags: Vec::new(),
            changelog: None,
        }
    }

    /// Create a new version from parent
    pub fn evolve_from(
        parent: &PatternVersionInfo,
        new_version: PatternVersion,
        reason: PatternEvolutionReason,
        solution: String,
        success_rate: f32,
        usage_count: u64,
        timestamp: u64,
        creator: impl Into<String>,
    ) -> Self {
        Self {
            version: new_version,
            pattern_id: parent.pattern_id,
            created_at: timestamp,
            created_by: creator.into(),
            evolution_reason: reason,
            parent_version: Some(parent.version),
            success_rate_snapshot: success_rate,
            usage_count_snapshot: usage_count,
            solution_snapshot: solution,
            is_current: true,
            branch_name: parent.branch_name.clone(),
            tags: Vec::new(),
            changelog: None,
        }
    }

    /// Add a tag to this version
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        self.tags.push(tag.into());
    }

    /// Set changelog for this version
    pub fn set_changelog(&mut self, changelog: impl Into<String>) {
        self.changelog = Some(changelog.into());
    }
}

/// Configuration for pattern versioning behavior
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VersioningConfig {
    /// Enable automatic versioning on significant changes
    pub auto_version: bool,
    /// Minimum success rate change to trigger auto-versioning (0.0-1.0)
    pub auto_version_threshold: f32,
    /// Maximum versions to keep per pattern (0 = unlimited)
    pub max_versions_per_pattern: usize,
    /// Enable branching for experimentation
    pub allow_branching: bool,
    /// Maximum branches per pattern
    pub max_branches_per_pattern: usize,
    /// Auto-prune old versions older than this (in days, 0 = never)
    pub prune_versions_older_than_days: u32,
    /// Keep at least this many versions even when pruning
    pub min_versions_to_keep: usize,
}

impl Default for VersioningConfig {
    fn default() -> Self {
        Self {
            auto_version: true,
            auto_version_threshold: 0.1, // 10% change triggers new version
            max_versions_per_pattern: 100,
            allow_branching: true,
            max_branches_per_pattern: 5,
            prune_versions_older_than_days: 365, // Keep for 1 year
            min_versions_to_keep: 5,
        }
    }
}

/// A branch for experimental pattern variations
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternBranch {
    /// Branch name
    pub name: String,
    /// Pattern this branch belongs to
    pub pattern_id: PatternId,
    /// Version where the branch started
    pub branch_point: PatternVersion,
    /// Current head version on this branch
    pub head_version: PatternVersion,
    /// When the branch was created
    pub created_at: u64,
    /// Who created the branch
    pub created_by: String,
    /// Branch description
    pub description: Option<String>,
    /// Whether this branch is still active
    pub is_active: bool,
    /// Whether this branch has been merged
    pub is_merged: bool,
    /// When it was merged (if merged)
    pub merged_at: Option<u64>,
}

impl PatternBranch {
    /// Create a new branch
    pub fn new(
        name: impl Into<String>,
        pattern_id: PatternId,
        branch_point: PatternVersion,
        timestamp: u64,
        creator: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            pattern_id,
            branch_point,
            head_version: branch_point,
            created_at: timestamp,
            created_by: creator.into(),
            description: None,
            is_active: true,
            is_merged: false,
            merged_at: None,
        }
    }

    /// Mark branch as merged
    pub fn mark_merged(&mut self, timestamp: u64) {
        self.is_merged = true;
        self.is_active = false;
        self.merged_at = Some(timestamp);
    }

    /// Close branch without merging
    pub fn close(&mut self) {
        self.is_active = false;
    }
}

/// Registry for tracking pattern versions
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternVersionRegistry {
    /// Version history per pattern (pattern_id -> versions)
    versions: HashMap<PatternId, Vec<PatternVersionInfo>>,
    /// Branches per pattern
    branches: HashMap<PatternId, Vec<PatternBranch>>,
    /// Configuration
    pub config: VersioningConfig,
}

impl Default for PatternVersionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternVersionRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            versions: HashMap::new(),
            branches: HashMap::new(),
            config: VersioningConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: VersioningConfig) -> Self {
        Self {
            versions: HashMap::new(),
            branches: HashMap::new(),
            config,
        }
    }

    /// Get read-only access to all version histories
    pub fn versions(&self) -> &HashMap<PatternId, Vec<PatternVersionInfo>> {
        &self.versions
    }

    /// Get read-only access to all branches
    pub fn branches(&self) -> &HashMap<PatternId, Vec<PatternBranch>> {
        &self.branches
    }

    /// Register initial version of a pattern
    pub fn register_initial(
        &mut self,
        pattern_id: PatternId,
        solution: String,
        timestamp: u64,
        creator: impl Into<String>,
    ) {
        let version_info = PatternVersionInfo::initial(pattern_id, solution, timestamp, creator);
        self.versions.insert(pattern_id, vec![version_info]);
    }

    /// Get current version of a pattern
    pub fn current_version(&self, pattern_id: PatternId) -> Option<&PatternVersionInfo> {
        self.versions
            .get(&pattern_id)?
            .iter()
            .find(|v| v.is_current && v.branch_name.is_none())
    }

    /// Get all versions of a pattern (main branch only)
    pub fn all_versions(&self, pattern_id: PatternId) -> Vec<&PatternVersionInfo> {
        self.versions
            .get(&pattern_id)
            .map(|versions| {
                versions
                    .iter()
                    .filter(|v| v.branch_name.is_none())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get specific version
    pub fn get_version(
        &self,
        pattern_id: PatternId,
        version: PatternVersion,
    ) -> Option<&PatternVersionInfo> {
        self.versions
            .get(&pattern_id)?
            .iter()
            .find(|v| v.version == version)
    }

    /// Evolve a pattern to a new version
    pub fn evolve(
        &mut self,
        pattern_id: PatternId,
        reason: PatternEvolutionReason,
        solution: String,
        success_rate: f32,
        usage_count: u64,
        timestamp: u64,
        creator: impl Into<String>,
    ) -> Option<PatternVersion> {
        let versions = self.versions.get_mut(&pattern_id)?;

        // Find current version
        let current_idx = versions
            .iter()
            .position(|v| v.is_current && v.branch_name.is_none())?;

        // Determine new version based on reason (extract data before mutation)
        let current_version = versions[current_idx].version;
        let new_version = match &reason {
            PatternEvolutionReason::Initial => PatternVersion::initial(),
            PatternEvolutionReason::BugFix { .. } => current_version.bump_patch(),
            PatternEvolutionReason::Refinement { .. } => current_version.bump_patch(),
            PatternEvolutionReason::PerformanceImprovement { .. } => current_version.bump_minor(),
            PatternEvolutionReason::Generalization { .. } => current_version.bump_minor(),
            PatternEvolutionReason::Specialization { .. } => current_version.bump_minor(),
            PatternEvolutionReason::Merge { .. } => current_version.bump_major(),
            PatternEvolutionReason::Split { .. } => current_version.bump_major(),
            PatternEvolutionReason::CommunityContribution { .. } => current_version.bump_minor(),
            PatternEvolutionReason::AutoEvolution { .. } => current_version.bump_patch(),
            PatternEvolutionReason::Rollback { .. } => current_version.bump_patch(),
            PatternEvolutionReason::BranchCreation { .. } => current_version.bump_patch(),
            PatternEvolutionReason::BranchMerge { .. } => current_version.bump_major(),
        };

        // Clone current info for evolve_from before mutation
        let current_info = versions[current_idx].clone();

        // Mark current as no longer current
        versions[current_idx].is_current = false;

        // Create new version
        let new_info = PatternVersionInfo::evolve_from(
            &current_info,
            new_version,
            reason,
            solution,
            success_rate,
            usage_count,
            timestamp,
            creator,
        );

        versions.push(new_info);

        // Prune if needed
        self.prune_versions(pattern_id, timestamp);

        Some(new_version)
    }

    /// Create a branch for experimentation
    pub fn create_branch(
        &mut self,
        pattern_id: PatternId,
        branch_name: impl Into<String>,
        timestamp: u64,
        creator: impl Into<String>,
    ) -> Result<(), &'static str> {
        if !self.config.allow_branching {
            return Err("Branching is disabled");
        }

        let branch_name = branch_name.into();

        // Check if pattern exists and clone data we need
        let current_info = self
            .current_version(pattern_id)
            .ok_or("Pattern not found")?
            .clone();

        // Check branch limit
        let existing_branches = self.branches.entry(pattern_id).or_default();
        let active_count = existing_branches.iter().filter(|b| b.is_active).count();
        if active_count >= self.config.max_branches_per_pattern {
            return Err("Maximum branches reached");
        }

        // Check if branch name already exists
        if existing_branches
            .iter()
            .any(|b| b.name == branch_name && b.is_active)
        {
            return Err("Branch name already exists");
        }

        let branch = PatternBranch::new(
            branch_name.clone(),
            pattern_id,
            current_info.version,
            timestamp,
            creator,
        );

        existing_branches.push(branch);

        // Create initial branch version
        if let Some(versions) = self.versions.get_mut(&pattern_id) {
            let mut branch_version = current_info;
            branch_version.branch_name = Some(branch_name.clone());
            branch_version.evolution_reason =
                PatternEvolutionReason::BranchCreation { branch_name };
            branch_version.created_at = timestamp;
            versions.push(branch_version);
        }

        Ok(())
    }

    /// Get active branches for a pattern
    pub fn active_branches(&self, pattern_id: PatternId) -> Vec<&PatternBranch> {
        self.branches
            .get(&pattern_id)
            .map(|branches| branches.iter().filter(|b| b.is_active).collect())
            .unwrap_or_default()
    }

    /// Get all branches for a pattern
    pub fn all_branches(&self, pattern_id: PatternId) -> Vec<&PatternBranch> {
        self.branches
            .get(&pattern_id)
            .map(|branches| branches.iter().collect())
            .unwrap_or_default()
    }

    /// Merge a branch back to main
    pub fn merge_branch(
        &mut self,
        pattern_id: PatternId,
        branch_name: &str,
        timestamp: u64,
        creator: impl Into<String>,
    ) -> Result<PatternVersion, &'static str> {
        // Find the branch
        let branches = self
            .branches
            .get_mut(&pattern_id)
            .ok_or("Pattern not found")?;
        let branch_idx = branches
            .iter()
            .position(|b| b.name == branch_name && b.is_active)
            .ok_or("Branch not found or not active")?;

        // Get branch head version
        let branch_head = self
            .versions
            .get(&pattern_id)
            .and_then(|v| {
                v.iter()
                    .filter(|vi| vi.branch_name.as_ref() == Some(&branch_name.to_string()))
                    .max_by_key(|vi| vi.version)
            })
            .ok_or("No versions found on branch")?
            .clone();

        // Mark branch as merged
        branches[branch_idx].mark_merged(timestamp);

        // Evolve main with merge
        let new_version = self
            .evolve(
                pattern_id,
                PatternEvolutionReason::BranchMerge {
                    branch_name: branch_name.to_string(),
                },
                branch_head.solution_snapshot,
                branch_head.success_rate_snapshot,
                branch_head.usage_count_snapshot,
                timestamp,
                creator,
            )
            .ok_or("Failed to create merge version")?;

        Ok(new_version)
    }

    /// Rollback to a previous version
    pub fn rollback(
        &mut self,
        pattern_id: PatternId,
        target_version: PatternVersion,
        reason: impl Into<String>,
        timestamp: u64,
        creator: impl Into<String>,
    ) -> Result<PatternVersion, &'static str> {
        let target = self
            .get_version(pattern_id, target_version)
            .ok_or("Target version not found")?
            .clone();

        let current = self
            .current_version(pattern_id)
            .ok_or("No current version")?;
        let current_version = current.version;

        let new_version = self
            .evolve(
                pattern_id,
                PatternEvolutionReason::Rollback {
                    rolled_back_from: current_version,
                    reason: reason.into(),
                },
                target.solution_snapshot,
                target.success_rate_snapshot,
                target.usage_count_snapshot,
                timestamp,
                creator,
            )
            .ok_or("Failed to create rollback version")?;

        Ok(new_version)
    }

    /// Check if a pattern should be auto-versioned based on changes
    pub fn should_auto_version(&self, pattern_id: PatternId, new_success_rate: f32) -> bool {
        if !self.config.auto_version {
            return false;
        }

        if let Some(current) = self.current_version(pattern_id) {
            let change = (new_success_rate - current.success_rate_snapshot).abs();
            change >= self.config.auto_version_threshold
        } else {
            false
        }
    }

    /// Prune old versions according to config
    pub fn prune_versions(&mut self, pattern_id: PatternId, current_time: u64) {
        if self.config.max_versions_per_pattern == 0
            && self.config.prune_versions_older_than_days == 0
        {
            return; // No pruning configured
        }

        let versions = match self.versions.get_mut(&pattern_id) {
            Some(v) => v,
            None => return,
        };

        // Always keep current version
        let current_idx = versions.iter().position(|v| v.is_current);

        // Calculate age threshold
        let age_threshold = if self.config.prune_versions_older_than_days > 0 {
            let secs = self.config.prune_versions_older_than_days as u64 * 24 * 60 * 60;
            current_time.saturating_sub(secs)
        } else {
            0
        };

        // Sort by version (newest first) for pruning
        let mut indices_to_keep: Vec<usize> = Vec::new();

        // Always keep current
        if let Some(idx) = current_idx {
            indices_to_keep.push(idx);
        }

        // Keep recent versions
        for (i, _v) in versions.iter().enumerate() {
            if indices_to_keep.len() >= self.config.min_versions_to_keep {
                break;
            }
            if !indices_to_keep.contains(&i) {
                indices_to_keep.push(i);
            }
        }

        // Keep versions newer than age threshold
        for (i, v) in versions.iter().enumerate() {
            if v.created_at >= age_threshold && !indices_to_keep.contains(&i) {
                indices_to_keep.push(i);
            }
        }

        // Apply max limit
        if self.config.max_versions_per_pattern > 0
            && indices_to_keep.len() > self.config.max_versions_per_pattern
        {
            indices_to_keep.sort();
            indices_to_keep.truncate(self.config.max_versions_per_pattern);
        }

        // Remove versions not in keep list
        indices_to_keep.sort();
        let mut new_versions = Vec::new();
        for (i, v) in versions.drain(..).enumerate() {
            if indices_to_keep.contains(&i) {
                new_versions.push(v);
            }
        }
        *versions = new_versions;
    }

    /// Get version history with lineage
    pub fn version_lineage(
        &self,
        pattern_id: PatternId,
    ) -> Vec<(PatternVersion, Option<PatternVersion>)> {
        self.versions
            .get(&pattern_id)
            .map(|versions| {
                versions
                    .iter()
                    .filter(|v| v.branch_name.is_none())
                    .map(|v| (v.version, v.parent_version))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Count total versions tracked
    pub fn total_versions(&self) -> usize {
        self.versions.values().map(|v| v.len()).sum()
    }

    /// Count patterns being versioned
    pub fn patterns_count(&self) -> usize {
        self.versions.len()
    }

    /// Get statistics
    pub fn stats(&self) -> VersioningStats {
        let total_patterns = self.versions.len();
        let total_versions = self.total_versions();
        let total_branches: usize = self.branches.values().map(|b| b.len()).sum();
        let active_branches: usize = self
            .branches
            .values()
            .map(|branches| branches.iter().filter(|b| b.is_active).count())
            .sum();

        let avg_versions_per_pattern = if total_patterns > 0 {
            total_versions as f32 / total_patterns as f32
        } else {
            0.0
        };

        VersioningStats {
            total_patterns,
            total_versions,
            total_branches,
            active_branches,
            avg_versions_per_pattern,
        }
    }
}

/// Statistics about pattern versioning
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VersioningStats {
    /// Total patterns being versioned
    pub total_patterns: usize,
    /// Total versions across all patterns
    pub total_versions: usize,
    /// Total branches ever created
    pub total_branches: usize,
    /// Currently active branches
    pub active_branches: usize,
    /// Average versions per pattern
    pub avg_versions_per_pattern: f32,
}
