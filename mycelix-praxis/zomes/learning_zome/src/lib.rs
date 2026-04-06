// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Learning Zome
//!
//! Manages courses, learning materials, and learner progress.

use praxis_core::CourseId;
use serde::{Deserialize, Serialize};

/// Course entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Course {
    /// Unique identifier
    pub course_id: CourseId,

    /// Course title
    pub title: String,

    /// Description
    pub description: String,

    /// Course creator agent
    pub creator: String,

    /// Tags for discovery
    pub tags: Vec<String>,

    /// Model ID for personalization (if using FL)
    pub model_id: Option<String>,

    /// Creation timestamp
    pub created_at: i64,

    /// Last updated timestamp
    pub updated_at: i64,

    /// Metadata
    pub metadata: Option<serde_json::Value>,
}

/// Learner progress entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnerProgress {
    /// Course being studied
    pub course_id: CourseId,

    /// Learner agent
    pub learner: String,

    /// Progress percentage (0-100)
    pub progress_percent: f32,

    /// Completed modules/lessons
    pub completed_items: Vec<String>,

    /// Current learning model version
    pub model_version: Option<String>,

    /// Last activity timestamp
    pub last_active: i64,

    /// Metadata (scores, time spent, etc.)
    pub metadata: Option<serde_json::Value>,
}

/// Learning activity entry (for privacy-preserving analytics)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningActivity {
    /// Course context
    pub course_id: CourseId,

    /// Activity type (view, quiz, exercise, etc.)
    pub activity_type: String,

    /// Item identifier (lesson ID, quiz ID, etc.)
    pub item_id: String,

    /// Outcome (score, completion, etc.)
    pub outcome: Option<f32>,

    /// Time spent (seconds)
    pub duration_secs: u32,

    /// Timestamp
    pub timestamp: i64,
}

// TODO: Implement HDK entry definitions and validation
// TODO: Implement zome functions for:
//   - create_course
//   - update_course
//   - get_course
//   - list_courses
//   - enroll
//   - update_progress
//   - get_progress
//   - record_activity

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_course_creation() {
        let course = Course {
            course_id: CourseId("course-1".to_string()),
            title: "Introduction to Rust".to_string(),
            description: "Learn Rust programming".to_string(),
            creator: "agent123".to_string(),
            tags: vec!["programming".to_string(), "rust".to_string()],
            model_id: None,
            created_at: 1234567890,
            updated_at: 1234567890,
            metadata: None,
        };

        assert_eq!(course.title, "Introduction to Rust");
    }
}
