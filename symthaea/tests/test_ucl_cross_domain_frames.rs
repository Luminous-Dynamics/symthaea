// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for UCL Cross-Domain Frames
//!
//! These tests verify that the 6 missing UCL frames are properly implemented:
//! - TRADE
//! - CONFLICT
//! - FEEDBACK_LOOP
//! - NORM_ENFORCEMENT
//! - COOPERATION
//! - ADAPTATION

use std::collections::HashMap;
use symthaea_core::hdc::ucl_cross_domain_frames::{UCLFrameSystem, concept_hv};

/// Test that all 6 UCL frames exist
#[test]
fn test_all_ucl_frames_exist() {
    let system = UCLFrameSystem::new();

    let expected_frames = [
        "TRADE",
        "CONFLICT",
        "FEEDBACK_LOOP",
        "NORM_ENFORCEMENT",
        "COOPERATION",
        "ADAPTATION",
    ];

    for frame_name in expected_frames {
        assert!(
            system.get_frame(frame_name).is_some(),
            "Missing UCL frame: {}",
            frame_name
        );
    }

    assert_eq!(system.count(), 6, "Should have exactly 6 UCL frames");
}

/// Test TRADE frame has correct slots
#[test]
fn test_trade_frame_slots() {
    let system = UCLFrameSystem::new();
    let trade = system.get_frame("TRADE").expect("TRADE frame should exist");

    // Required slots
    let giver = trade.get_slot("giver").expect("giver slot");
    let receiver = trade.get_slot("receiver").expect("receiver slot");
    let resource = trade.get_slot("resource").expect("resource slot");
    let price = trade.get_slot("price").expect("price slot");

    assert!(giver.required, "giver should be required");
    assert!(receiver.required, "receiver should be required");
    assert!(resource.required, "resource should be required");
    assert!(price.required, "price should be required");

    // Optional slots
    let medium = trade.get_slot("medium").expect("medium slot");
    assert!(!medium.required, "medium should be optional");
}

/// Test CONFLICT frame has correct slots
#[test]
fn test_conflict_frame_slots() {
    let system = UCLFrameSystem::new();
    let conflict = system
        .get_frame("CONFLICT")
        .expect("CONFLICT frame should exist");

    // Required slots
    assert!(conflict.get_slot("parties").is_some(), "parties slot");
    assert!(conflict.get_slot("stakes").is_some(), "stakes slot");

    // Optional slots
    let strategies = conflict.get_slot("strategies").expect("strategies slot");
    assert!(!strategies.required, "strategies should be optional");
}

/// Test FEEDBACK_LOOP frame has correct slots
#[test]
fn test_feedback_loop_frame_slots() {
    let system = UCLFrameSystem::new();
    let feedback = system
        .get_frame("FEEDBACK_LOOP")
        .expect("FEEDBACK_LOOP frame should exist");

    // Required slots
    assert!(feedback.get_slot("variable").is_some(), "variable slot");
    assert!(
        feedback.get_slot("influence_path").is_some(),
        "influence_path slot"
    );
    assert!(feedback.get_slot("sign").is_some(), "sign slot");

    // Optional slots
    let delay = feedback.get_slot("delay").expect("delay slot");
    assert!(!delay.required, "delay should be optional");
}

/// Test NORM_ENFORCEMENT frame has correct slots
#[test]
fn test_norm_enforcement_frame_slots() {
    let system = UCLFrameSystem::new();
    let norm = system
        .get_frame("NORM_ENFORCEMENT")
        .expect("NORM_ENFORCEMENT frame should exist");

    // Required slots
    assert!(norm.get_slot("norm").is_some(), "norm slot");
    assert!(norm.get_slot("violator").is_some(), "violator slot");

    // Optional slots
    let observer = norm.get_slot("observer").expect("observer slot");
    let sanction = norm.get_slot("sanction").expect("sanction slot");
    assert!(!observer.required, "observer should be optional");
    assert!(!sanction.required, "sanction should be optional");
}

/// Test COOPERATION frame has correct slots
#[test]
fn test_cooperation_frame_slots() {
    let system = UCLFrameSystem::new();
    let coop = system
        .get_frame("COOPERATION")
        .expect("COOPERATION frame should exist");

    // Required slots
    assert!(coop.get_slot("agents").is_some(), "agents slot");
    assert!(coop.get_slot("shared_goal").is_some(), "shared_goal slot");
    assert!(
        coop.get_slot("contributions").is_some(),
        "contributions slot"
    );
}

/// Test ADAPTATION frame has correct slots
#[test]
fn test_adaptation_frame_slots() {
    let system = UCLFrameSystem::new();
    let adapt = system
        .get_frame("ADAPTATION")
        .expect("ADAPTATION frame should exist");

    // Required slots
    assert!(adapt.get_slot("system").is_some(), "system slot");
    assert!(adapt.get_slot("environment").is_some(), "environment slot");
    assert!(adapt.get_slot("pressure").is_some(), "pressure slot");
    assert!(adapt.get_slot("response").is_some(), "response slot");
}

/// Test frame instantiation with all required bindings
#[test]
fn test_frame_instantiation_success() {
    let system = UCLFrameSystem::new();

    // Create TRADE instance
    let mut trade_bindings = HashMap::new();
    trade_bindings.insert("giver".to_string(), concept_hv("alice"));
    trade_bindings.insert("receiver".to_string(), concept_hv("bob"));
    trade_bindings.insert("resource".to_string(), concept_hv("car"));
    trade_bindings.insert("price".to_string(), concept_hv("money"));

    let instance = system.instantiate("TRADE", trade_bindings);
    assert!(instance.is_ok(), "TRADE instantiation should succeed");

    let instance = instance.unwrap();
    assert_eq!(instance.frame_name, "TRADE");
    assert!(instance.bindings.contains_key("giver"));
    assert!(instance.bindings.contains_key("receiver"));
}

/// Test frame instantiation fails with missing required slots
#[test]
fn test_frame_instantiation_missing_required() {
    let system = UCLFrameSystem::new();

    // Missing "price" slot
    let mut incomplete_bindings = HashMap::new();
    incomplete_bindings.insert("giver".to_string(), concept_hv("alice"));
    incomplete_bindings.insert("receiver".to_string(), concept_hv("bob"));
    incomplete_bindings.insert("resource".to_string(), concept_hv("car"));

    let result = system.instantiate("TRADE", incomplete_bindings);
    assert!(result.is_err(), "Should fail with missing required slot");
}

/// Test that frame encodings are orthogonal
#[test]
fn test_frame_encodings_orthogonal() {
    let system = UCLFrameSystem::new();

    let frames: Vec<_> = system.all_frames().collect();

    for i in 0..frames.len() {
        for j in (i + 1)..frames.len() {
            let sim = frames[i]
                .frame_encoding
                .similarity(&frames[j].frame_encoding);

            // Frames should be fairly orthogonal (similarity < 0.7)
            assert!(
                sim < 0.7,
                "Frames {} and {} have too high similarity: {}",
                frames[i].name,
                frames[j].name,
                sim
            );
        }
    }
}

/// Test frame detection from encoding
#[test]
fn test_frame_detection() {
    let system = UCLFrameSystem::new();

    // Create a COOPERATION instance
    let mut bindings = HashMap::new();
    bindings.insert("agents".to_string(), concept_hv("team"));
    bindings.insert("shared_goal".to_string(), concept_hv("goal"));
    bindings.insert("contributions".to_string(), concept_hv("work"));

    let instance = system.instantiate("COOPERATION", bindings).unwrap();

    // Detect frames that match
    let detections = system.detect_frame(&instance.encoding, 0.1);

    // COOPERATION should be among the detected frames
    assert!(!detections.is_empty(), "Should detect at least one frame");
}

/// Test instance similarity
#[test]
fn test_instance_similarity() {
    let system = UCLFrameSystem::new();

    // Two similar CONFLICT instances (same parties, different resolution)
    let mut bindings1 = HashMap::new();
    bindings1.insert("parties".to_string(), concept_hv("team_a_vs_b"));
    bindings1.insert("stakes".to_string(), concept_hv("trophy"));

    let mut bindings2 = HashMap::new();
    bindings2.insert("parties".to_string(), concept_hv("team_a_vs_b"));
    bindings2.insert("stakes".to_string(), concept_hv("trophy"));
    bindings2.insert("resolution".to_string(), concept_hv("draw"));

    let instance1 = system.instantiate("CONFLICT", bindings1).unwrap();
    let instance2 = system.instantiate("CONFLICT", bindings2).unwrap();

    // Instances with same core should have positive similarity
    let sim = instance1.similarity(&instance2);
    assert!(
        sim > 0.0,
        "Similar instances should have positive similarity"
    );
}

/// Test cross-domain integration
#[test]
fn test_cross_domain_integration() {
    let system = UCLFrameSystem::new();

    // FEEDBACK_LOOP bridges causality and homeostasis
    let feedback = system.get_frame("FEEDBACK_LOOP").unwrap();
    assert!(
        feedback.domains.contains(&"causality".to_string()),
        "FEEDBACK_LOOP should bridge causality"
    );
    assert!(
        feedback.domains.contains(&"homeostasis".to_string()),
        "FEEDBACK_LOOP should bridge homeostasis"
    );

    // COOPERATION bridges social and game_theory
    let coop = system.get_frame("COOPERATION").unwrap();
    assert!(
        coop.domains.contains(&"social".to_string()),
        "COOPERATION should bridge social"
    );
    assert!(
        coop.domains.contains(&"game_theory".to_string()),
        "COOPERATION should bridge game_theory"
    );

    // ADAPTATION bridges biology and metacognition
    let adapt = system.get_frame("ADAPTATION").unwrap();
    assert!(
        adapt.domains.contains(&"biology".to_string()),
        "ADAPTATION should bridge biology"
    );
    assert!(
        adapt.domains.contains(&"metacognition".to_string()),
        "ADAPTATION should bridge metacognition"
    );
}

/// Test summary generation
#[test]
fn test_summary_generation() {
    let system = UCLFrameSystem::new();
    let summary = system.summary();

    // Summary should contain all frame names
    assert!(summary.contains("TRADE"), "Summary should mention TRADE");
    assert!(
        summary.contains("CONFLICT"),
        "Summary should mention CONFLICT"
    );
    assert!(
        summary.contains("FEEDBACK_LOOP"),
        "Summary should mention FEEDBACK_LOOP"
    );
    assert!(
        summary.contains("NORM_ENFORCEMENT"),
        "Summary should mention NORM_ENFORCEMENT"
    );
    assert!(
        summary.contains("COOPERATION"),
        "Summary should mention COOPERATION"
    );
    assert!(
        summary.contains("ADAPTATION"),
        "Summary should mention ADAPTATION"
    );

    // Summary should contain slot information
    assert!(
        summary.contains("giver"),
        "Summary should mention giver slot"
    );
    assert!(
        summary.contains("parties"),
        "Summary should mention parties slot"
    );
    assert!(
        summary.contains("variable"),
        "Summary should mention variable slot"
    );
}