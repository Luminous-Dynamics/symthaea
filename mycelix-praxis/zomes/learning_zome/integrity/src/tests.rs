// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Unit tests for Learning Zome validation logic
//
// Tests all validation functions comprehensively to ensure data integrity

use super::*;
use praxis_core::CourseId;

// =================================================================================
// Helper Functions for Test Data Generation
// =================================================================================

/// Create a valid Course entry for testing
fn create_valid_course() -> Course {
    Course {
        course_id: CourseId("test-course-001".to_string()),
        title: "Introduction to Rust Programming".to_string(),
        description: "Learn Rust from the ground up with hands-on examples".to_string(),
        creator: "creator-agent-123".to_string(),
        tags: vec!["rust".to_string(), "programming".to_string()],
        model_id: Some("model-v1".to_string()),
        created_at: 1234567890,
        updated_at: 1234567890,
        metadata: None,
    }
}

/// Create a valid LearnerProgress entry for testing
fn create_valid_progress() -> LearnerProgress {
    LearnerProgress {
        course_id: CourseId("test-course-001".to_string()),
        learner: "learner-agent-456".to_string(),
        progress_percent: 50.0,
        completed_items: vec!["module-1".to_string(), "module-2".to_string()],
        model_version: Some("v1".to_string()),
        last_active: 1234567890,
        metadata: None,
    }
}

/// Create a valid LearningActivity entry for testing
fn create_valid_activity() -> LearningActivity {
    LearningActivity {
        course_id: CourseId("test-course-001".to_string()),
        activity_type: "quiz".to_string(),
        item_id: "quiz-01".to_string(),
        outcome: Some(85.0),
        duration_secs: 300, // 5 minutes
        timestamp: 1234567890,
    }
}

// =================================================================================
// Course Validation Tests
// =================================================================================

#[cfg(test)]
mod course_validation_tests {
    use super::*;

    #[test]
    fn test_valid_course() {
        let course = create_valid_course();
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_course_empty_title() {
        let mut course = create_valid_course();
        course.title = "".to_string();
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_course_whitespace_only_title() {
        let mut course = create_valid_course();
        course.title = "   \t\n  ".to_string();
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_course_title_too_long() {
        let mut course = create_valid_course();
        course.title = "a".repeat(201); // Max is 200
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_course_title_exactly_max_length() {
        let mut course = create_valid_course();
        course.title = "a".repeat(200); // Exactly max
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_course_empty_description() {
        let mut course = create_valid_course();
        course.description = "".to_string();
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_course_description_too_long() {
        let mut course = create_valid_course();
        course.description = "a".repeat(5001); // Max is 5000
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_course_description_exactly_max_length() {
        let mut course = create_valid_course();
        course.description = "a".repeat(5000); // Exactly max
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_course_updated_before_created() {
        let mut course = create_valid_course();
        course.created_at = 2000;
        course.updated_at = 1000; // Before created
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_course_updated_same_as_created() {
        let mut course = create_valid_course();
        course.created_at = 1000;
        course.updated_at = 1000; // Same time is valid
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_course_too_many_tags() {
        let mut course = create_valid_course();
        course.tags = (0..11).map(|i| format!("tag-{}", i)).collect(); // Max is 10
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_course_exactly_max_tags() {
        let mut course = create_valid_course();
        course.tags = (0..10).map(|i| format!("tag-{}", i)).collect(); // Exactly max
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_course_empty_tag() {
        let mut course = create_valid_course();
        course.tags = vec!["valid".to_string(), "".to_string()];
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_course_whitespace_only_tag() {
        let mut course = create_valid_course();
        course.tags = vec!["valid".to_string(), "  \t  ".to_string()];
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_course_tag_too_long() {
        let mut course = create_valid_course();
        course.tags = vec!["a".repeat(51)]; // Max is 50
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_course_tag_exactly_max_length() {
        let mut course = create_valid_course();
        course.tags = vec!["a".repeat(50)]; // Exactly max
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_course_no_tags() {
        let mut course = create_valid_course();
        course.tags = vec![];
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_course_minimal_valid() {
        let course = Course {
            course_id: CourseId("min".to_string()),
            title: "A".to_string(), // 1 char is valid
            description: "B".to_string(), // 1 char is valid
            creator: "c".to_string(),
            tags: vec![],
            model_id: None,
            created_at: 0,
            updated_at: 0,
            metadata: None,
        };
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }
}

// =================================================================================
// LearnerProgress Validation Tests
// =================================================================================

#[cfg(test)]
mod learner_progress_validation_tests {
    use super::*;

    #[test]
    fn test_valid_progress() {
        let progress = create_valid_progress();
        let result = validate_learner_progress(&progress);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_progress_negative_percent() {
        let mut progress = create_valid_progress();
        progress.progress_percent = -1.0;
        let result = validate_learner_progress(&progress);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_progress_zero_percent() {
        let mut progress = create_valid_progress();
        progress.progress_percent = 0.0; // Valid
        let result = validate_learner_progress(&progress);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_progress_hundred_percent() {
        let mut progress = create_valid_progress();
        progress.progress_percent = 100.0; // Valid
        let result = validate_learner_progress(&progress);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_progress_over_hundred_percent() {
        let mut progress = create_valid_progress();
        progress.progress_percent = 100.1;
        let result = validate_learner_progress(&progress);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_progress_fractional_percent() {
        let mut progress = create_valid_progress();
        progress.progress_percent = 42.7; // Valid
        let result = validate_learner_progress(&progress);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_progress_too_many_completed_items() {
        let mut progress = create_valid_progress();
        progress.completed_items = (0..1001).map(|i| format!("item-{}", i)).collect(); // Max is 1000
        let result = validate_learner_progress(&progress);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_progress_exactly_max_completed_items() {
        let mut progress = create_valid_progress();
        progress.completed_items = (0..1000).map(|i| format!("item-{}", i)).collect(); // Exactly max
        let result = validate_learner_progress(&progress);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_progress_no_completed_items() {
        let mut progress = create_valid_progress();
        progress.completed_items = vec![];
        let result = validate_learner_progress(&progress);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_progress_minimal_valid() {
        let progress = LearnerProgress {
            course_id: CourseId("c".to_string()),
            learner: "l".to_string(),
            progress_percent: 0.0,
            completed_items: vec![],
            model_version: None,
            last_active: 0,
            metadata: None,
        };
        let result = validate_learner_progress(&progress);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }
}

// =================================================================================
// LearningActivity Validation Tests
// =================================================================================

#[cfg(test)]
mod learning_activity_validation_tests {
    use super::*;

    #[test]
    fn test_valid_activity() {
        let activity = create_valid_activity();
        let result = validate_learning_activity(&activity);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_activity_empty_activity_type() {
        let mut activity = create_valid_activity();
        activity.activity_type = "".to_string();
        let result = validate_learning_activity(&activity);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_activity_whitespace_only_activity_type() {
        let mut activity = create_valid_activity();
        activity.activity_type = "  \t\n  ".to_string();
        let result = validate_learning_activity(&activity);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_activity_empty_item_id() {
        let mut activity = create_valid_activity();
        activity.item_id = "".to_string();
        let result = validate_learning_activity(&activity);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_activity_whitespace_only_item_id() {
        let mut activity = create_valid_activity();
        activity.item_id = "  \t  ".to_string();
        let result = validate_learning_activity(&activity);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_activity_duration_too_long() {
        let mut activity = create_valid_activity();
        activity.duration_secs = 86401; // Max is 86400 (24 hours)
        let result = validate_learning_activity(&activity);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_activity_duration_exactly_max() {
        let mut activity = create_valid_activity();
        activity.duration_secs = 86400; // Exactly 24 hours
        let result = validate_learning_activity(&activity);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_activity_duration_zero() {
        let mut activity = create_valid_activity();
        activity.duration_secs = 0; // Valid
        let result = validate_learning_activity(&activity);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_activity_outcome_negative() {
        let mut activity = create_valid_activity();
        activity.outcome = Some(-1.0);
        let result = validate_learning_activity(&activity);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_activity_outcome_zero() {
        let mut activity = create_valid_activity();
        activity.outcome = Some(0.0); // Valid
        let result = validate_learning_activity(&activity);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_activity_outcome_hundred() {
        let mut activity = create_valid_activity();
        activity.outcome = Some(100.0); // Valid
        let result = validate_learning_activity(&activity);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_activity_outcome_over_hundred() {
        let mut activity = create_valid_activity();
        activity.outcome = Some(100.1);
        let result = validate_learning_activity(&activity);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_activity_outcome_fractional() {
        let mut activity = create_valid_activity();
        activity.outcome = Some(73.5); // Valid
        let result = validate_learning_activity(&activity);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_activity_outcome_none() {
        let mut activity = create_valid_activity();
        activity.outcome = None; // Valid (outcome is optional)
        let result = validate_learning_activity(&activity);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_activity_minimal_valid() {
        let activity = LearningActivity {
            course_id: CourseId("c".to_string()),
            activity_type: "t".to_string(), // 1 char is valid
            item_id: "i".to_string(), // 1 char is valid
            outcome: None,
            duration_secs: 0,
            timestamp: 0,
        };
        let result = validate_learning_activity(&activity);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }
}

// =================================================================================
// Edge Case and Integration Tests
// =================================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_course_with_special_characters_in_title() {
        let mut course = create_valid_course();
        course.title = "C++ & Rust: A ❤️ Story (2024) [Tutorial]".to_string();
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_course_with_unicode_in_description() {
        let mut course = create_valid_course();
        course.description = "学习编程 - Learn programming 学习 🚀".to_string();
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_progress_with_duplicate_completed_items() {
        let mut progress = create_valid_progress();
        progress.completed_items = vec![
            "module-1".to_string(),
            "module-1".to_string(), // Duplicate (validation doesn't prevent this)
        ];
        let result = validate_learner_progress(&progress);
        // Validation allows duplicates (business logic decision)
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_activity_with_very_long_type_string() {
        let mut activity = create_valid_activity();
        activity.activity_type = "a".repeat(1000); // Very long but no max limit
        let result = validate_learning_activity(&activity);
        // No length limit on activity_type (only non-empty check)
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }
}

// =================================================================================
// Security Validation Tests (is_finite guards)
// =================================================================================

#[cfg(test)]
mod security_validation_tests {
    use super::*;

    // ---- LearnerProgress NaN/Inf guards ----

    #[test]
    fn test_progress_nan_rejected() {
        let mut progress = create_valid_progress();
        progress.progress_percent = f32::NAN;
        let result = validate_learner_progress(&progress);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_progress_inf_rejected() {
        let mut progress = create_valid_progress();
        progress.progress_percent = f32::INFINITY;
        let result = validate_learner_progress(&progress);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_progress_neg_inf_rejected() {
        let mut progress = create_valid_progress();
        progress.progress_percent = f32::NEG_INFINITY;
        let result = validate_learner_progress(&progress);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_progress_valid_passes() {
        let progress = create_valid_progress();
        let result = validate_learner_progress(&progress);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    // ---- LearningActivity NaN/Inf guards ----

    #[test]
    fn test_activity_nan_outcome_rejected() {
        let mut activity = create_valid_activity();
        activity.outcome = Some(f32::NAN);
        let result = validate_learning_activity(&activity);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_activity_inf_outcome_rejected() {
        let mut activity = create_valid_activity();
        activity.outcome = Some(f32::INFINITY);
        let result = validate_learning_activity(&activity);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_activity_neg_inf_outcome_rejected() {
        let mut activity = create_valid_activity();
        activity.outcome = Some(f32::NEG_INFINITY);
        let result = validate_learning_activity(&activity);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    // ---- Course validation ----

    #[test]
    fn test_course_valid_passes() {
        let course = create_valid_course();
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn test_course_title_too_long_rejected() {
        let mut course = create_valid_course();
        course.title = "x".repeat(201);
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_course_empty_title_rejected() {
        let mut course = create_valid_course();
        course.title = "".to_string();
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_course_empty_description_rejected() {
        let mut course = create_valid_course();
        course.description = "".to_string();
        let result = validate_course(&course);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }
}
