// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Praxis hApp sweettest integration tests.
//!
//! Tests learning workflows including course management, progress tracking,
//! federated learning, and credentials using the Holochain sweettest framework.
//!
//! Prerequisites:
//!   cd mycelix-praxis && cargo build --release --target wasm32-unknown-unknown
//!   hc dna pack dna/ -o dna/praxis.dna
//!
//! Run: cargo test --release -p mycelix-sweettest -- --ignored praxis
//!
//! Updated for Holochain 0.6 sweettest API.

mod harness;

use harness::*;
use holochain::prelude::*;
use serial_test::serial;

// =============================================================================
// Course Management Tests
// =============================================================================

/// Test: Create a course and retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_create_and_get_course() {
    let agents = setup_test_agents(
        &DnaPaths::praxis(),
        "mycelix-praxis",
        1,
    )
    .await;

    let instructor = &agents[0];

    // Create a course
    let course_input = serde_json::json!({
        "course_id": "course_rust_101",
        "title": "Rust Programming Fundamentals",
        "description": "Learn Rust from the ground up",
        "creator": format!("did:mycelix:{}", instructor.agent_pubkey),
        "tags": ["rust", "programming", "systems"],
        "model_id": null,
        "created_at": Timestamp::now().as_micros(),
        "updated_at": Timestamp::now().as_micros(),
        "metadata": null
    });

    let course_record: Record = instructor
        .call_zome_fn("learning", "create_course", course_input)
        .await;

    let action_hash = course_record.action_hashed().hash.clone();
    assert!(!action_hash.as_ref().is_empty(), "Course should be created");

    // Retrieve the course
    let retrieved: Option<Record> = instructor
        .call_zome_fn("learning", "get_course", action_hash)
        .await;

    assert!(retrieved.is_some(), "Course should be retrievable");
}

/// Test: List courses returns created courses.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_list_courses() {
    let agents = setup_test_agents(
        &DnaPaths::praxis(),
        "mycelix-praxis",
        1,
    )
    .await;

    let instructor = &agents[0];

    // Create multiple courses
    for i in 1..=3 {
        let course_input = serde_json::json!({
            "course_id": format!("course_list_test_{}", i),
            "title": format!("Test Course {}", i),
            "description": format!("Description for course {}", i),
            "creator": format!("did:mycelix:{}", instructor.agent_pubkey),
            "tags": ["test"],
            "model_id": null,
            "created_at": Timestamp::now().as_micros(),
            "updated_at": Timestamp::now().as_micros(),
            "metadata": null
        });

        let _: Record = instructor
            .call_zome_fn("learning", "create_course", course_input)
            .await;
    }

    // List all courses
    let courses: Vec<Record> = instructor
        .call_zome_fn("learning", "list_courses", ())
        .await;

    assert!(courses.len() >= 3, "Should have at least 3 courses");
}

// =============================================================================
// Learner Progress Tests
// =============================================================================

/// Test: Enroll in a course and update progress.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_enrollment_and_progress() {
    let agents = setup_test_agents(
        &DnaPaths::praxis(),
        "mycelix-praxis",
        2,
    )
    .await;

    let instructor = &agents[0];
    let learner = &agents[1];

    // Instructor creates a course
    let course_input = serde_json::json!({
        "course_id": "course_progress_test",
        "title": "Progress Tracking Course",
        "description": "A course to test progress tracking",
        "creator": format!("did:mycelix:{}", instructor.agent_pubkey),
        "tags": ["test"],
        "model_id": null,
        "created_at": Timestamp::now().as_micros(),
        "updated_at": Timestamp::now().as_micros(),
        "metadata": null
    });

    let course_record: Record = instructor
        .call_zome_fn("learning", "create_course", course_input)
        .await;

    let course_hash = course_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    // Learner enrolls
    let enroll_input = serde_json::json!({
        "course_action_hash": course_hash,
        "learner_did": format!("did:mycelix:{}", learner.agent_pubkey)
    });

    let _: Record = learner
        .call_zome_fn("learning", "enroll", enroll_input)
        .await;

    // Update progress
    let progress_input = serde_json::json!({
        "course_id": "course_progress_test",
        "learner": format!("did:mycelix:{}", learner.agent_pubkey),
        "progress_percent": 25.0,
        "completed_items": ["lesson_1"],
        "model_version": null,
        "last_active": Timestamp::now().as_micros(),
        "metadata": null
    });

    let progress_record: Record = learner
        .call_zome_fn("learning", "update_progress", progress_input)
        .await;

    assert!(
        !progress_record.action_hashed().hash.as_ref().is_empty(),
        "Progress should be recorded"
    );
}

/// Test: Record learning activity.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_record_activity() {
    let agents = setup_test_agents(
        &DnaPaths::praxis(),
        "mycelix-praxis",
        1,
    )
    .await;

    let learner = &agents[0];

    // Record a quiz activity
    let activity_input = serde_json::json!({
        "course_id": "course_activity_test",
        "activity_type": "quiz",
        "item_id": "quiz_chapter_1",
        "outcome": 85.0,
        "duration_secs": 300,
        "timestamp": Timestamp::now().as_micros()
    });

    let activity_record: Record = learner
        .call_zome_fn("learning", "record_activity", activity_input)
        .await;

    assert!(
        !activity_record.action_hashed().hash.as_ref().is_empty(),
        "Activity should be recorded"
    );
}

// =============================================================================
// Federated Learning Tests
// =============================================================================

/// Test: Create FL model and submit gradient update.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_fl_gradient_submission() {
    let agents = setup_test_agents(
        &DnaPaths::praxis(),
        "mycelix-praxis",
        2,
    )
    .await;

    let coordinator = &agents[0];
    let participant = &agents[1];

    // Coordinator creates a model
    let model_input = serde_json::json!({
        "model_id": "model_fl_test_v1",
        "model_type": "course_recommendation",
        "parameters": {},
        "privacy_budget": 1.0,
        "min_participants": 2,
        "created_by": format!("did:mycelix:{}", coordinator.agent_pubkey),
        "created_at": Timestamp::now().as_micros()
    });

    let model_record: Record = coordinator
        .call_zome_fn("fl", "create_model", model_input)
        .await;

    assert!(
        !model_record.action_hashed().hash.as_ref().is_empty(),
        "FL model should be created"
    );

    wait_for_dht_sync().await;

    // Participant submits gradient (clipped for differential privacy)
    let gradient_input = serde_json::json!({
        "model_id": "model_fl_test_v1",
        "gradient_hash": "QmGradientHash123",
        "noise_added": true,
        "clip_norm": 1.0,
        "sample_count": 100,
        "submitted_by": format!("did:mycelix:{}", participant.agent_pubkey),
        "submitted_at": Timestamp::now().as_micros()
    });

    let gradient_record: Record = participant
        .call_zome_fn("fl", "upload_gradient", gradient_input)
        .await;

    assert!(
        !gradient_record.action_hashed().hash.as_ref().is_empty(),
        "Gradient should be submitted"
    );
}

// =============================================================================
// Credential Tests
// =============================================================================

/// Test: Issue and verify completion credential.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_issue_completion_credential() {
    let agents = setup_test_agents(
        &DnaPaths::praxis(),
        "mycelix-praxis",
        2,
    )
    .await;

    let issuer = &agents[0];
    let learner = &agents[1];

    // Issue a completion credential
    let credential_input = serde_json::json!({
        "subject_did": format!("did:mycelix:{}", learner.agent_pubkey),
        "issuer_did": format!("did:mycelix:{}", issuer.agent_pubkey),
        "credential_type": ["VerifiableCredential", "CourseCompletionCredential"],
        "claims": {
            "courseId": "rust_101",
            "completionDate": "2024-01-15",
            "grade": "A",
            "skills": ["rust", "systems-programming"]
        },
        "issuance_date": Timestamp::now().as_micros(),
        "expiration_date": null
    });

    let credential_record: Record = issuer
        .call_zome_fn("credential", "issue_credential", credential_input)
        .await;

    assert!(
        !credential_record.action_hashed().hash.as_ref().is_empty(),
        "Credential should be issued"
    );

    wait_for_dht_sync().await;

    // Verify the credential
    let verify_input = serde_json::json!({
        "credential_hash": credential_record.action_hashed().hash
    });

    let verification: serde_json::Value = learner
        .call_zome_fn("credential", "verify_credential", verify_input)
        .await;

    // Verification should return valid status
    assert!(
        verification.get("valid").is_some() || verification.get("status").is_some(),
        "Verification result should include validity"
    );
}

// =============================================================================
// DAO Governance Tests
// =============================================================================

/// Test: Create curriculum proposal and vote.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_curriculum_proposal_voting() {
    let agents = setup_test_agents(
        &DnaPaths::praxis(),
        "mycelix-praxis",
        3,
    )
    .await;

    let proposer = &agents[0];
    let voter1 = &agents[1];
    let voter2 = &agents[2];

    let now = Timestamp::now();
    let voting_ends = Timestamp::from_micros(now.as_micros() + 7 * 24 * 60 * 60 * 1_000_000);

    // Create a curriculum proposal
    let proposal_input = serde_json::json!({
        "proposal_id": "prop_curriculum_001",
        "title": "Add Advanced Rust Course to Core Curriculum",
        "description": "Proposal to include advanced systems programming in the required curriculum",
        "proposal_type": "Curriculum",
        "proposer_did": format!("did:mycelix:{}", proposer.agent_pubkey),
        "voting_starts": now.as_micros(),
        "voting_ends": voting_ends.as_micros(),
        "quorum_threshold": 0.5,
        "pass_threshold": 0.66
    });

    let proposal_record: Record = proposer
        .call_zome_fn("dao", "create_proposal", proposal_input)
        .await;

    assert!(
        !proposal_record.action_hashed().hash.as_ref().is_empty(),
        "Proposal should be created"
    );

    wait_for_dht_sync().await;
    wait_for_dht_sync().await; // Extra time for 3 conductors

    // Voters cast votes
    for (voter, choice) in [(voter1, "For"), (voter2, "For")] {
        let vote_input = serde_json::json!({
            "proposal_id": "prop_curriculum_001",
            "voter_did": format!("did:mycelix:{}", voter.agent_pubkey),
            "choice": choice,
            "reason": "Supports skill development"
        });

        let _: Record = voter
            .call_zome_fn("dao", "cast_vote", vote_input)
            .await;
    }

    wait_for_dht_sync().await;

    // Query votes
    let votes: Vec<Record> = proposer
        .call_zome_fn("dao", "get_proposal_votes", "prop_curriculum_001".to_string())
        .await;

    assert!(votes.len() >= 2, "Should have at least 2 votes recorded");
}
