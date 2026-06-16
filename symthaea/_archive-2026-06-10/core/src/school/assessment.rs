// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Assessment System - Progress Tracking and Mastery Measurement
//!
//! The AssessmentTracker monitors learning progress across objectives
//! and curricula, tracking mastery levels and completion status.
//!
//! ## Mastery Model
//!
//! Uses a multi-factor mastery model based on:
//! - Number of successful learning sessions
//! - Error reduction over time (reality check history)
//! - Time since last practice (forgetting curve)
//! - Φ gain consistency
//!
//! ## The Forgetting Curve
//!
//! ```text
//! Retention
//!    │
//!  1 │━━━━━━━━━━━
//!    │          ╲
//!    │           ╲
//!    │            ╲____________________
//!  0 └───────────────────────────────── Time
//!         Decay without practice
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════════════════════════════
// MASTERY LEVELS
// ═══════════════════════════════════════════════════════════════════════════════

/// Mastery level for a learning objective
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum MasteryLevel {
    /// Not yet attempted (0.0)
    Untouched,

    /// Just started learning (0.0 - 0.2)
    Novice,

    /// Basic understanding (0.2 - 0.4)
    Beginner,

    /// Working knowledge (0.4 - 0.6)
    Intermediate,

    /// Solid competence (0.6 - 0.8)
    Proficient,

    /// Expert-level mastery (0.8 - 1.0)
    Expert,
}

impl MasteryLevel {
    /// Convert from a mastery score (0.0 - 1.0)
    pub fn from_score(score: f32) -> Self {
        match score {
            x if x <= 0.0 => MasteryLevel::Untouched,
            x if x < 0.2 => MasteryLevel::Novice,
            x if x < 0.4 => MasteryLevel::Beginner,
            x if x < 0.6 => MasteryLevel::Intermediate,
            x if x < 0.8 => MasteryLevel::Proficient,
            _ => MasteryLevel::Expert,
        }
    }

    /// Convert to a representative score
    pub fn as_f32(&self) -> f32 {
        match self {
            MasteryLevel::Untouched => 0.0,
            MasteryLevel::Novice => 0.1,
            MasteryLevel::Beginner => 0.3,
            MasteryLevel::Intermediate => 0.5,
            MasteryLevel::Proficient => 0.7,
            MasteryLevel::Expert => 0.9,
        }
    }

    /// Get display name
    pub fn as_str(&self) -> &str {
        match self {
            MasteryLevel::Untouched => "Untouched",
            MasteryLevel::Novice => "Novice",
            MasteryLevel::Beginner => "Beginner",
            MasteryLevel::Intermediate => "Intermediate",
            MasteryLevel::Proficient => "Proficient",
            MasteryLevel::Expert => "Expert",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// OBJECTIVE PROGRESS
// ═══════════════════════════════════════════════════════════════════════════════

/// Progress record for a single learning objective
#[derive(Debug, Clone)]
pub struct ObjectiveProgress {
    /// Objective ID
    pub objective_id: String,

    /// Current mastery score (0.0 - 1.0)
    pub mastery_score: f32,

    /// Number of learning sessions
    pub session_count: usize,

    /// Number of successful sessions
    pub success_count: usize,

    /// Total Φ gained from this objective
    pub total_phi_gained: f32,

    /// Average error from reality checks
    pub average_error: f32,

    /// Time of first learning attempt
    pub first_attempt: Option<Instant>,

    /// Time of most recent learning attempt
    pub last_attempt: Option<Instant>,

    /// Time to reach current mastery (cumulative)
    pub total_learning_time: Duration,

    /// Is this objective complete?
    pub completed: bool,
}

impl ObjectiveProgress {
    /// Create new progress record
    pub fn new(objective_id: &str) -> Self {
        Self {
            objective_id: objective_id.to_string(),
            mastery_score: 0.0,
            session_count: 0,
            success_count: 0,
            total_phi_gained: 0.0,
            average_error: 0.0,
            first_attempt: None,
            last_attempt: None,
            total_learning_time: Duration::ZERO,
            completed: false,
        }
    }

    /// Get current mastery level
    pub fn mastery_level(&self) -> MasteryLevel {
        MasteryLevel::from_score(self.mastery_score)
    }

    /// Get success rate (0.0 - 1.0)
    pub fn success_rate(&self) -> f32 {
        if self.session_count == 0 {
            0.0
        } else {
            self.success_count as f32 / self.session_count as f32
        }
    }

    /// Calculate time since last practice
    pub fn time_since_practice(&self) -> Option<Duration> {
        self.last_attempt.map(|t| t.elapsed())
    }

    /// Apply forgetting curve decay
    pub fn apply_decay(&mut self, decay_rate: f32) {
        if let Some(elapsed) = self.time_since_practice() {
            // Exponential decay: mastery *= e^(-decay_rate * days)
            let days = elapsed.as_secs_f32() / 86400.0;
            let decay_factor = (-decay_rate * days).exp();
            self.mastery_score *= decay_factor;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CURRICULUM PROGRESS
// ═══════════════════════════════════════════════════════════════════════════════

/// Progress record for an entire curriculum
#[derive(Debug, Clone)]
pub struct CurriculumProgress {
    /// Curriculum ID
    pub curriculum_id: String,

    /// IDs of completed objectives
    pub completed_objectives: Vec<String>,

    /// IDs of objectives in progress
    pub in_progress_objectives: Vec<String>,

    /// Overall completion percentage
    pub completion_percent: f32,

    /// Average mastery across all objectives
    pub average_mastery: f32,

    /// Total Φ gained from this curriculum
    pub total_phi_gained: f32,

    /// Time when curriculum was started
    pub started_at: Option<Instant>,

    /// Time when curriculum was completed
    pub completed_at: Option<Instant>,
}

impl CurriculumProgress {
    /// Create new curriculum progress
    pub fn new(curriculum_id: &str) -> Self {
        Self {
            curriculum_id: curriculum_id.to_string(),
            completed_objectives: Vec::new(),
            in_progress_objectives: Vec::new(),
            completion_percent: 0.0,
            average_mastery: 0.0,
            total_phi_gained: 0.0,
            started_at: None,
            completed_at: None,
        }
    }

    /// Check if curriculum is complete
    pub fn is_complete(&self) -> bool {
        self.completed_at.is_some()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ASSESSMENT TRACKER
// ═══════════════════════════════════════════════════════════════════════════════

/// Main assessment tracker for the School system
pub struct AssessmentTracker {
    /// Progress by objective ID
    objective_progress: HashMap<String, ObjectiveProgress>,

    /// Progress by curriculum ID
    curriculum_progress: HashMap<String, CurriculumProgress>,

    /// Forgetting curve decay rate (per day)
    decay_rate: f32,

    /// Mastery threshold for completion
    completion_threshold: f32,

    /// Total learning sessions
    total_sessions: usize,

    /// Total Φ gained across all learning
    total_phi_gained: f32,

    /// When tracking started
    started_at: Instant,
}

impl AssessmentTracker {
    /// Create a new assessment tracker
    pub fn new(decay_rate: f32, completion_threshold: f32) -> Self {
        Self {
            objective_progress: HashMap::new(),
            curriculum_progress: HashMap::new(),
            decay_rate,
            completion_threshold,
            total_sessions: 0,
            total_phi_gained: 0.0,
            started_at: Instant::now(),
        }
    }

    /// Record a learning session for an objective
    pub fn record_session(
        &mut self,
        objective_id: &str,
        phi_gained: f32,
        error: f32,
        session_duration: Duration,
        successful: bool,
    ) {
        // Get or create progress record
        let progress = self
            .objective_progress
            .entry(objective_id.to_string())
            .or_insert_with(|| ObjectiveProgress::new(objective_id));

        // Update session counts
        progress.session_count += 1;
        if successful {
            progress.success_count += 1;
        }

        // Set first attempt time
        if progress.first_attempt.is_none() {
            progress.first_attempt = Some(Instant::now());
        }
        progress.last_attempt = Some(Instant::now());

        // Update cumulative stats
        progress.total_phi_gained += phi_gained;
        progress.total_learning_time += session_duration;

        // Update average error (running average)
        let n = progress.session_count as f32;
        progress.average_error = progress.average_error * (n - 1.0) / n + error / n;

        // Calculate new mastery score
        // Formula: weighted combination of factors
        let success_factor = progress.success_rate();
        let error_factor = 1.0 - (progress.average_error * 10.0).min(1.0);
        let phi_factor = (progress.total_phi_gained * 10.0).min(1.0);
        let session_factor = (progress.session_count as f32 / 10.0).min(1.0);

        progress.mastery_score =
            (success_factor * 0.3 + error_factor * 0.3 + phi_factor * 0.2 + session_factor * 0.2)
                .clamp(0.0, 1.0);

        // Check for completion
        if progress.mastery_score >= self.completion_threshold {
            progress.completed = true;
        }

        // Update global stats
        self.total_sessions += 1;
        self.total_phi_gained += phi_gained;
    }

    /// Record objective completion in a curriculum
    pub fn record_objective_completion(
        &mut self,
        curriculum_id: &str,
        objective_id: &str,
        total_objectives: usize,
    ) {
        // Get or create curriculum progress
        let curriculum = self
            .curriculum_progress
            .entry(curriculum_id.to_string())
            .or_insert_with(|| CurriculumProgress::new(curriculum_id));

        // Set start time if first objective
        if curriculum.started_at.is_none() {
            curriculum.started_at = Some(Instant::now());
        }

        // Remove from in-progress, add to completed
        curriculum
            .in_progress_objectives
            .retain(|id| id != objective_id);
        if !curriculum
            .completed_objectives
            .contains(&objective_id.to_string())
        {
            curriculum
                .completed_objectives
                .push(objective_id.to_string());
        }

        // Update completion percentage
        curriculum.completion_percent =
            curriculum.completed_objectives.len() as f32 / total_objectives as f32 * 100.0;

        // Mark curriculum complete if all objectives done
        if curriculum.completed_objectives.len() >= total_objectives {
            curriculum.completed_at = Some(Instant::now());
        }
    }

    /// Mark an objective as in-progress
    pub fn mark_in_progress(&mut self, curriculum_id: &str, objective_id: &str) {
        let curriculum = self
            .curriculum_progress
            .entry(curriculum_id.to_string())
            .or_insert_with(|| CurriculumProgress::new(curriculum_id));

        if !curriculum
            .in_progress_objectives
            .contains(&objective_id.to_string())
        {
            curriculum
                .in_progress_objectives
                .push(objective_id.to_string());
        }

        if curriculum.started_at.is_none() {
            curriculum.started_at = Some(Instant::now());
        }
    }

    /// Get progress for an objective
    pub fn get_objective_progress(&self, objective_id: &str) -> Option<&ObjectiveProgress> {
        self.objective_progress.get(objective_id)
    }

    /// Get progress for a curriculum
    pub fn get_curriculum_progress(&self, curriculum_id: &str) -> Option<&CurriculumProgress> {
        self.curriculum_progress.get(curriculum_id)
    }

    /// Get all objective progress
    pub fn all_objective_progress(&self) -> impl Iterator<Item = &ObjectiveProgress> {
        self.objective_progress.values()
    }

    /// Get all curriculum progress
    pub fn all_curriculum_progress(&self) -> impl Iterator<Item = &CurriculumProgress> {
        self.curriculum_progress.values()
    }

    /// Apply forgetting curve decay to all objectives
    pub fn apply_decay(&mut self) {
        for progress in self.objective_progress.values_mut() {
            progress.apply_decay(self.decay_rate);

            // Re-check completion after decay
            if progress.mastery_score < self.completion_threshold {
                progress.completed = false;
            }
        }
    }

    /// Check if an objective is mastered
    pub fn is_mastered(&self, objective_id: &str) -> bool {
        self.objective_progress
            .get(objective_id)
            .map(|p| p.completed)
            .unwrap_or(false)
    }

    /// Check if a curriculum is complete
    pub fn is_curriculum_complete(&self, curriculum_id: &str) -> bool {
        self.curriculum_progress
            .get(curriculum_id)
            .map(|p| p.is_complete())
            .unwrap_or(false)
    }

    /// Get objectives that need review (mastery decaying)
    pub fn get_review_needed(&self, threshold: f32) -> Vec<&ObjectiveProgress> {
        self.objective_progress
            .values()
            .filter(|p| p.completed && p.mastery_score < self.completion_threshold - threshold)
            .collect()
    }

    /// Get overall statistics
    pub fn stats(&self) -> AssessmentStats {
        let total_objectives = self.objective_progress.len();
        let completed_objectives = self
            .objective_progress
            .values()
            .filter(|p| p.completed)
            .count();

        let avg_mastery = if total_objectives > 0 {
            self.objective_progress
                .values()
                .map(|p| p.mastery_score)
                .sum::<f32>()
                / total_objectives as f32
        } else {
            0.0
        };

        let total_curricula = self.curriculum_progress.len();
        let completed_curricula = self
            .curriculum_progress
            .values()
            .filter(|p| p.is_complete())
            .count();

        AssessmentStats {
            total_objectives,
            completed_objectives,
            average_mastery: avg_mastery,
            total_curricula,
            completed_curricula,
            total_sessions: self.total_sessions,
            total_phi_gained: self.total_phi_gained,
            time_learning: self.started_at.elapsed(),
        }
    }

    /// Get mastery level distribution
    pub fn mastery_distribution(&self) -> MasteryDistribution {
        let mut dist = MasteryDistribution::default();

        for progress in self.objective_progress.values() {
            match progress.mastery_level() {
                MasteryLevel::Untouched => dist.untouched += 1,
                MasteryLevel::Novice => dist.novice += 1,
                MasteryLevel::Beginner => dist.beginner += 1,
                MasteryLevel::Intermediate => dist.intermediate += 1,
                MasteryLevel::Proficient => dist.proficient += 1,
                MasteryLevel::Expert => dist.expert += 1,
            }
        }

        dist
    }
}

impl AssessmentTracker {
    /// Create a new default tracker (used by School)
    pub fn default_for_school() -> Self {
        Self::new(0.1, 0.7)
    }

    /// Record a learning event (simplified interface for School)
    pub fn record_learning(
        &mut self,
        objective_id: &str,
        phi_gained: f32,
        was_hallucination: bool,
    ) {
        self.record_session(
            objective_id,
            phi_gained,
            if was_hallucination { 0.1 } else { 0.01 }, // Error based on hallucination
            std::time::Duration::from_secs(60),         // Default 1 minute session
            !was_hallucination,                         // Success if not hallucination
        );
    }

    /// Get count of mastered objectives
    pub fn mastered_count(&self) -> usize {
        self.objective_progress
            .values()
            .filter(|p| p.completed)
            .count()
    }

    /// Get mastery level for an objective
    pub fn mastery_level(&self, objective_id: &str) -> MasteryLevel {
        self.objective_progress
            .get(objective_id)
            .map(|p| p.mastery_level())
            .unwrap_or(MasteryLevel::Untouched)
    }
}

impl Default for AssessmentTracker {
    fn default() -> Self {
        Self::new(0.1, 0.7) // Decay 10% per day, complete at 70% mastery
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STATISTICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Overall assessment statistics
#[derive(Debug, Clone)]
pub struct AssessmentStats {
    /// Total objectives tracked
    pub total_objectives: usize,

    /// Objectives marked complete
    pub completed_objectives: usize,

    /// Average mastery score
    pub average_mastery: f32,

    /// Total curricula tracked
    pub total_curricula: usize,

    /// Curricula completed
    pub completed_curricula: usize,

    /// Total learning sessions
    pub total_sessions: usize,

    /// Total Φ gained
    pub total_phi_gained: f32,

    /// Time since tracking started
    pub time_learning: Duration,
}

/// Distribution of mastery levels
#[derive(Debug, Clone, Default)]
pub struct MasteryDistribution {
    pub untouched: usize,
    pub novice: usize,
    pub beginner: usize,
    pub intermediate: usize,
    pub proficient: usize,
    pub expert: usize,
}

impl MasteryDistribution {
    /// Get total count
    pub fn total(&self) -> usize {
        self.untouched
            + self.novice
            + self.beginner
            + self.intermediate
            + self.proficient
            + self.expert
    }

    /// Get percentage at each level
    pub fn percentages(&self) -> MasteryPercentages {
        let total = self.total() as f32;
        if total == 0.0 {
            return MasteryPercentages::default();
        }

        MasteryPercentages {
            untouched: self.untouched as f32 / total * 100.0,
            novice: self.novice as f32 / total * 100.0,
            beginner: self.beginner as f32 / total * 100.0,
            intermediate: self.intermediate as f32 / total * 100.0,
            proficient: self.proficient as f32 / total * 100.0,
            expert: self.expert as f32 / total * 100.0,
        }
    }
}

/// Percentages of mastery levels
#[derive(Debug, Clone, Default)]
pub struct MasteryPercentages {
    pub untouched: f32,
    pub novice: f32,
    pub beginner: f32,
    pub intermediate: f32,
    pub proficient: f32,
    pub expert: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mastery_levels() {
        assert_eq!(MasteryLevel::from_score(0.0), MasteryLevel::Untouched);
        assert_eq!(MasteryLevel::from_score(0.15), MasteryLevel::Novice);
        assert_eq!(MasteryLevel::from_score(0.35), MasteryLevel::Beginner);
        assert_eq!(MasteryLevel::from_score(0.55), MasteryLevel::Intermediate);
        assert_eq!(MasteryLevel::from_score(0.75), MasteryLevel::Proficient);
        assert_eq!(MasteryLevel::from_score(0.95), MasteryLevel::Expert);
    }

    #[test]
    fn test_assessment_tracker_creation() {
        let tracker = AssessmentTracker::new(0.1, 0.7);
        let stats = tracker.stats();

        assert_eq!(stats.total_objectives, 0);
        assert_eq!(stats.total_sessions, 0);
        assert_eq!(stats.total_phi_gained, 0.0);
    }

    #[test]
    fn test_record_session() {
        let mut tracker = AssessmentTracker::new(0.1, 0.7);

        tracker.record_session(
            "test-obj",
            0.01,                    // phi gained
            0.001,                   // error
            Duration::from_secs(60), // duration
            true,                    // successful
        );

        let progress = tracker.get_objective_progress("test-obj").unwrap();
        assert_eq!(progress.session_count, 1);
        assert_eq!(progress.success_count, 1);
        assert!(progress.mastery_score > 0.0);
    }

    #[test]
    fn test_mastery_progression() {
        let mut tracker = AssessmentTracker::new(0.1, 0.7);

        // Many successful sessions should lead to high mastery
        for _ in 0..15 {
            tracker.record_session("test-obj", 0.01, 0.001, Duration::from_secs(60), true);
        }

        let progress = tracker.get_objective_progress("test-obj").unwrap();
        assert!(progress.mastery_score >= 0.7);
        assert!(progress.completed);
    }

    #[test]
    fn test_curriculum_progress() {
        let mut tracker = AssessmentTracker::new(0.1, 0.7);

        tracker.mark_in_progress("nixos", "nix-basics");
        tracker.record_objective_completion("nixos", "nix-basics", 5);
        tracker.record_objective_completion("nixos", "nix-lang", 5);

        let progress = tracker.get_curriculum_progress("nixos").unwrap();
        assert_eq!(progress.completed_objectives.len(), 2);
        assert!((progress.completion_percent - 40.0).abs() < 0.1);
        assert!(!progress.is_complete());
    }

    #[test]
    fn test_curriculum_completion() {
        let mut tracker = AssessmentTracker::new(0.1, 0.7);

        for i in 0..3 {
            tracker.record_objective_completion("small", &format!("obj-{}", i), 3);
        }

        let progress = tracker.get_curriculum_progress("small").unwrap();
        assert!(progress.is_complete());
        assert!((progress.completion_percent - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_mastery_distribution() {
        let mut tracker = AssessmentTracker::new(0.1, 0.7);

        // Create objectives at different mastery levels
        tracker.record_session("beginner", 0.001, 0.01, Duration::from_secs(30), true);

        for _ in 0..10 {
            tracker.record_session("expert", 0.02, 0.0001, Duration::from_secs(60), true);
        }

        let dist = tracker.mastery_distribution();
        assert_eq!(dist.total(), 2);
    }

    #[test]
    fn test_review_needed() {
        let mut tracker = AssessmentTracker::new(0.1, 0.7);

        // Build up mastery
        for _ in 0..15 {
            tracker.record_session("decaying", 0.01, 0.001, Duration::from_secs(60), true);
        }

        // Force decay by directly modifying (simulating time passage)
        if let Some(progress) = tracker.objective_progress.get_mut("decaying") {
            progress.mastery_score = 0.55; // Decayed below threshold
        }

        let needs_review = tracker.get_review_needed(0.0);
        assert_eq!(needs_review.len(), 1);
    }

    #[test]
    fn test_success_rate() {
        let mut tracker = AssessmentTracker::new(0.1, 0.7);

        tracker.record_session("mixed", 0.01, 0.01, Duration::from_secs(60), true);
        tracker.record_session("mixed", 0.005, 0.02, Duration::from_secs(60), false);
        tracker.record_session("mixed", 0.01, 0.01, Duration::from_secs(60), true);

        let progress = tracker.get_objective_progress("mixed").unwrap();
        assert!((progress.success_rate() - 0.666).abs() < 0.01);
    }
}
