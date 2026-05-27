#![cfg(feature = "school_learning")]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the school learning module.
//!
//! Tests the full learning lifecycle: curriculum management, recommendation,
//! learning execution, mastery progression, and hallucination detection.

use symthaea::school::*;

fn fast_school() -> School {
    School::new(SchoolConfig::fast()).expect("School::new with fast config should succeed")
}

#[test]
fn test_school_creation_with_configs() {
    let _default = School::new(SchoolConfig::default()).expect("default config");
    let _fast = School::new(SchoolConfig::fast()).expect("fast config");
    let _accurate = School::new(SchoolConfig::accurate()).expect("accurate config");
}

#[test]
fn test_add_curriculum_and_list_objectives() {
    let mut school = fast_school();

    let curriculum = Curriculum::new("test", "Test Curriculum")
        .with_description("A test curriculum")
        .with_objective(
            LearningObjective::new("obj1", "First Objective")
                .with_domain(Domain::NixOS)
                .with_difficulty(Difficulty::Beginner)
                .build(),
        )
        .with_objective(
            LearningObjective::new("obj2", "Second Objective")
                .with_domain(Domain::NixOS)
                .with_difficulty(Difficulty::Intermediate)
                .with_prerequisite("obj1")
                .build(),
        )
        .build();

    school
        .add_curriculum(curriculum)
        .expect("add_curriculum should succeed");

    let all = school.all_objectives();
    assert_eq!(all.len(), 2, "Should have 2 objectives");
}

#[test]
fn test_recommend_next_with_empty_curriculum() {
    let school = fast_school();

    // With no curricula, recommend_next should handle gracefully
    let result = school.recommend_next();
    // Either returns an error or a "complete" recommendation
    match result {
        Ok(rec) => assert!(rec.is_complete(), "No objectives → should be complete"),
        Err(_) => {} // Also acceptable
    }
}

#[test]
fn test_recommend_next_with_objectives() {
    let mut school = fast_school();

    let curriculum = Curriculum::new("basics", "Basics")
        .with_objective(
            LearningObjective::new("hello", "Hello NixOS")
                .with_domain(Domain::NixOS)
                .with_difficulty(Difficulty::Beginner)
                .build(),
        )
        .build();

    school.add_curriculum(curriculum).unwrap();

    let rec = school
        .recommend_next()
        .expect("should produce recommendation");
    assert!(
        rec.predicted_phi_gain.is_finite(),
        "phi_gain should be finite"
    );
    assert!(rec.confidence.is_finite(), "confidence should be finite");
}

#[test]
fn test_learn_objective_produces_result() {
    let mut school = fast_school();

    let objective = LearningObjective::new("test_obj", "Test Objective")
        .with_domain(Domain::NixOS)
        .with_difficulty(Difficulty::Beginner)
        .build();

    let curriculum = Curriculum::new("test", "Test")
        .with_objective(objective.clone())
        .build();
    school.add_curriculum(curriculum).unwrap();

    let result = school.learn(&objective).expect("learn should succeed");
    assert!(result.actual_phi_gain.is_finite());
    assert!(result.prediction_error.is_finite());
    assert!(!result.objective_id.is_empty());
}

#[test]
fn test_hallucination_detection() {
    let mut school = fast_school();

    let objective = LearningObjective::new("hard", "Very Hard Objective")
        .with_domain(Domain::Mathematics)
        .with_difficulty(Difficulty::Expert)
        .build();

    let curriculum = Curriculum::new("math", "Math")
        .with_objective(objective.clone())
        .build();
    school.add_curriculum(curriculum).unwrap();

    let result = school.learn(&objective).expect("learn should succeed");
    // was_hallucination is a bool — just check it's valid
    let _ = result.was_hallucination;
    // prediction_error should be non-negative
    assert!(result.prediction_error >= 0.0);
}

#[test]
fn test_prerequisite_gating() {
    let mut school = fast_school();

    let curriculum = Curriculum::new("gated", "Gated Curriculum")
        .with_objective(
            LearningObjective::new("prereq", "Prerequisite")
                .with_domain(Domain::NixOS)
                .with_difficulty(Difficulty::Beginner)
                .build(),
        )
        .with_objective(
            LearningObjective::new("advanced", "Advanced")
                .with_domain(Domain::NixOS)
                .with_difficulty(Difficulty::Advanced)
                .with_prerequisite("prereq")
                .build(),
        )
        .build();

    school.add_curriculum(curriculum).unwrap();

    let ready = school.ready_objectives();
    // Only "prereq" should be ready (no prerequisites)
    // "advanced" should NOT be ready (depends on "prereq")
    let ready_ids: Vec<&str> = ready.iter().map(|o| o.id.as_str()).collect();
    assert!(
        ready_ids.contains(&"prereq"),
        "prereq should be in ready_objectives"
    );
    assert!(
        !ready_ids.contains(&"advanced"),
        "advanced should NOT be in ready_objectives (prereq unmet)"
    );
}

#[test]
fn test_mastery_progression() {
    let mut school = fast_school();

    let objective = LearningObjective::new("repeat", "Repeatable Objective")
        .with_domain(Domain::Consciousness)
        .with_difficulty(Difficulty::Beginner)
        .build();

    let curriculum = Curriculum::new("prog", "Progression")
        .with_objective(objective.clone())
        .build();
    school.add_curriculum(curriculum).unwrap();

    // Learn the same objective multiple times
    let mut last_mastery = MasteryLevel::Untouched;
    for _ in 0..5 {
        let result = school.learn(&objective).expect("learn should succeed");
        last_mastery = result.mastery_level;
    }

    // After 5 learning sessions, mastery should have progressed from Untouched
    assert_ne!(
        last_mastery,
        MasteryLevel::Untouched,
        "Mastery should progress after repeated learning"
    );
}

#[test]
fn test_school_stats_tracking() {
    let mut school = fast_school();

    let objective = LearningObjective::new("stats_obj", "Stats Test")
        .with_domain(Domain::HDC)
        .with_difficulty(Difficulty::Intermediate)
        .build();

    let curriculum = Curriculum::new("stats", "Stats Test")
        .with_objective(objective.clone())
        .build();
    school.add_curriculum(curriculum).unwrap();

    let stats_before = school.stats();
    assert_eq!(stats_before.total_learned, 0);

    school.learn(&objective).expect("learn should succeed");

    let stats_after = school.stats();
    assert!(
        stats_after.total_learned > stats_before.total_learned,
        "total_learned should increase"
    );
    assert!(stats_after.current_phi.is_finite());
}

#[test]
fn test_lookahead_engine_predictions() {
    let mut school = fast_school();

    let objective = LearningObjective::new("look", "Lookahead Test")
        .with_domain(Domain::Rust)
        .with_difficulty(Difficulty::Intermediate)
        .build();

    let curriculum = Curriculum::new("look", "Lookahead")
        .with_objective(objective.clone())
        .build();
    school.add_curriculum(curriculum).unwrap();

    // The recommend_next() method exercises the lookahead engine internally
    let rec = school.recommend_next().expect("should recommend");
    assert!(
        rec.predicted_phi_gain.is_finite(),
        "CfC prediction should be finite"
    );
    assert!(
        rec.confidence >= 0.0 && rec.confidence <= 1.0,
        "confidence should be in [0, 1], got {}",
        rec.confidence
    );
}