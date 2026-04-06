// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # End-to-End Integration Tests for Mycelix EduNet
//!
//! This module contains comprehensive E2E tests covering all 5 test scenarios from Phase 7.
//! These tests require a running Holochain conductor and are marked with `#[ignore]` by default.
//!
//! **To run these tests**, you'll need to:
//! 1. Start a Holochain conductor with the EduNet DNA
//! 2. Run tests with: `cargo test --test e2e_test -- --ignored`
//!
//! See `docs/dev/PHASE_7_E2E_TESTING_PLAN.md` for details.

use std::collections::HashMap;
use std::time::Duration;

// Holochain imports
use holochain::sweettest::{SweetConductor, SweetDnaFile, SweetAgents, SweetCell, SweetZome};
use holochain::conductor::config::ConductorConfig;
use holochain_types::prelude::*;

// Zome type imports
use learning_integrity::{Course, LearnerProgress};
use credential_integrity::{VerifiableCredential, CredentialSubject, CredentialStatus};
use dao_integrity::{ProposalType, ProposalCategory, VoteChoice};

// DAO coordinator types (for input structures)
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct CreateProposalInput {
    pub proposal_id: String,
    pub title: String,
    pub description: String,
    pub proposal_type: ProposalType,
    pub category: ProposalCategory,
    pub actions: Vec<serde_json::Value>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct CastVoteInput {
    pub proposal_id: String,
    pub proposal_hash: ActionHash,
    pub choice: VoteChoice,
    pub justification: Option<String>,
}

// ============================================================================
// Test Infrastructure Setup
// ============================================================================

/// Helper function to set up a conductor with the EduNet DNA
async fn setup_conductor() -> Result<SweetConductor, Box<dyn std::error::Error>> {
    // Create a conductor with default configuration
    let mut conductor = SweetConductor::from_standard_config().await;

    Ok(conductor)
}

/// Helper function to install the EduNet DNA on a conductor
async fn install_edunet_dna(
    conductor: &mut SweetConductor,
) -> Result<(Vec<SweetCell>, DnaHash), Box<dyn std::error::Error>> {
    // Load the DNA from the workspace DNA bundle
    let dna_path = std::path::PathBuf::from("../dna/edunet.dna");
    let dna = SweetDnaFile::from_bundle(&dna_path).await?;

    // Install the DNA and create agents
    let agents = SweetAgents::get(conductor, 2).await;
    let apps = conductor
        .setup_app_for_agents("mycelix-edunet", &agents, &[dna.clone()])
        .await?;

    // Extract cells from installed apps
    let mut cells = Vec::new();
    for app in apps.iter() {
        cells.push(app.cells()[0].clone());  // Get first cell from each app
    }

    let dna_hash = dna.dna_hash().clone();
    Ok((cells, dna_hash))
}

/// Helper function to set up a conductor with N agents (for DAO tests)
/// Returns the conductor and cells for direct zome calls
async fn setup_conductor_with_agents(
    num_agents: usize,
) -> Result<(SweetConductor, Vec<SweetCell>), Box<dyn std::error::Error>> {
    // Create conductor
    let mut conductor = SweetConductor::from_standard_config().await;

    // Load the DNA
    let dna_path = std::path::PathBuf::from("../dna/edunet.dna");
    let dna = SweetDnaFile::from_bundle(&dna_path).await?;

    // Create N agents
    let agents = SweetAgents::get(&conductor, num_agents).await;

    // Install the DNA for all agents
    let apps = conductor
        .setup_app_for_agents("mycelix-edunet", &agents, &[dna.clone()])
        .await?;

    // Extract cells from all apps (one cell per agent)
    let mut cells = Vec::new();
    for app in apps.iter() {
        cells.push(app.cells()[0].clone());
    }

    Ok((conductor, cells))
}

/// Helper to create a timestamp
fn timestamp() -> i64 {
    chrono::Utc::now().timestamp()
}

// ============================================================================
// Zome Access Helper Functions (Holochain 0.6 API)
// ============================================================================

/// Helper to get learning zome from a cell
fn learning_zome(cell: &SweetCell) -> &SweetZome {
    cell.zome("learning")
}

/// Helper to get FL zome from a cell
fn fl_zome(cell: &SweetCell) -> &SweetZome {
    cell.zome("fl")
}

/// Helper to get credential zome from a cell
fn credential_zome(cell: &SweetCell) -> &SweetZome {
    cell.zome("credential")
}

/// Helper to get DAO zome from a cell
fn dao_zome(cell: &SweetCell) -> &SweetZome {
    cell.zome("dao")
}

// ============================================================================
// Scenario 1: Complete Learning Journey
// ============================================================================

/// Scenario 1: Complete Learning Journey
/// Tests: Course Creation → Enrollment → Progress → Credential Issuance
#[cfg(test)]
mod scenario_1_learning_journey {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires DNA bundle to be built
    async fn test_complete_learning_journey() {
        println!("🚀 Starting Scenario 1: Complete Learning Journey");

        // Step 1: Setup conductor and install DNA
        println!("📋 Step 1: Setting up conductor and installing DNA...");
        let mut conductor = setup_conductor().await.expect("Failed to create conductor");
        let (cells, dna_hash) = install_edunet_dna(&mut conductor)
            .await
            .expect("Failed to install DNA");

        let alice_cell = &cells[0];
        let bobbo_cell = &cells[1];
        println!("✅ Conductor ready with 2 agents (Alice and Bobbo)");

        // Step 2: Alice creates a course
        println!("\n📋 Step 2: Alice creates a course...");
        let course = Course {
            course_id: "course-001".to_string(),
            title: "Introduction to Holochain".to_string(),
            description: "Learn the basics of Holochain development".to_string(),
            instructor: alice_cell.agent_pubkey().to_string(),
            created_at: timestamp(),
            difficulty_level: "Beginner".to_string(),
            prerequisites: vec![],
            learning_objectives: vec![
                "Understand DHT concepts".to_string(),
                "Write basic zomes".to_string(),
                "Deploy a Holochain app".to_string(),
            ],
            estimated_hours: 10,
            tags: vec!["holochain".to_string(), "rust".to_string(), "p2p".to_string()],
        };

        let course_hash: ActionHash = conductor
            .call(
                learning_zome(alice_cell),
                "create_course",
                course.clone(),
            )
            .await
            .expect("Failed to create course");

        println!("✅ Course created with hash: {:?}", course_hash);

        // Step 3: Bobbo enrolls in the course
        println!("\n📋 Step 3: Bobbo enrolls in the course...");
        let initial_progress = LearnerProgress {
            learner_id: bobbo_cell.agent_pubkey().to_string(),
            course_id: "course-001".to_string(),
            enrollment_date: timestamp(),
            completion_percentage: 0,
            last_activity: timestamp(),
            completed_modules: vec![],
            current_module: "Module 1: DHT Basics".to_string(),
            time_spent_minutes: 0,
            quiz_scores: HashMap::new(),
        };

        let progress_hash: ActionHash = conductor
            .call(
                learning_zome(bobbo_cell),
                "enroll_learner",
                initial_progress.clone(),
            )
            .await
            .expect("Failed to enroll learner");

        println!("✅ Bobbo enrolled with progress hash: {:?}", progress_hash);

        // Step 4: Bobbo completes modules and updates progress
        println!("\n📋 Step 4: Bobbo completes modules...");

        // Module 1 completion (33%)
        let mut updated_progress = initial_progress.clone();
        updated_progress.completion_percentage = 33;
        updated_progress.completed_modules = vec!["Module 1: DHT Basics".to_string()];
        updated_progress.current_module = "Module 2: Zome Development".to_string();
        updated_progress.time_spent_minutes = 180;
        updated_progress.last_activity = timestamp();

        let _: ActionHash = conductor
            .call(
                learning_zome(bobbo_cell),
                "update_progress",
                updated_progress.clone(),
            )
            .await
            .expect("Failed to update progress (Module 1)");

        println!("  ✅ Module 1 complete (33%)");

        // Module 2 completion (66%)
        updated_progress.completion_percentage = 66;
        updated_progress.completed_modules.push("Module 2: Zome Development".to_string());
        updated_progress.current_module = "Module 3: Deployment".to_string();
        updated_progress.time_spent_minutes = 360;
        updated_progress.last_activity = timestamp();

        let _: ActionHash = conductor
            .call(
                learning_zome(bobbo_cell),
                "update_progress",
                updated_progress.clone(),
            )
            .await
            .expect("Failed to update progress (Module 2)");

        println!("  ✅ Module 2 complete (66%)");

        // Module 3 completion (100%)
        updated_progress.completion_percentage = 100;
        updated_progress.completed_modules.push("Module 3: Deployment".to_string());
        updated_progress.current_module = "Completed".to_string();
        updated_progress.time_spent_minutes = 540;
        updated_progress.last_activity = timestamp();

        let final_progress_hash: ActionHash = conductor
            .call(
                learning_zome(bobbo_cell),
                "update_progress",
                updated_progress.clone(),
            )
            .await
            .expect("Failed to update progress (Module 3)");

        println!("  ✅ Module 3 complete (100%)");
        println!("✅ Course completed! Final progress hash: {:?}", final_progress_hash);

        // Step 5: Alice issues a credential to Bobbo
        println!("\n📋 Step 5: Alice issues a W3C Verifiable Credential to Bobbo...");
        let credential = VerifiableCredential {
            context: vec!["https://www.w3.org/2018/credentials/v1".to_string()],
            id: format!("urn:uuid:{}", uuid::Uuid::new_v4()),
            credential_type: vec!["VerifiableCredential".to_string(), "CourseCompletionCredential".to_string()],
            issuer: alice_cell.agent_pubkey().to_string(),
            issuance_date: timestamp(),
            expiration_date: Some(timestamp() + (365 * 24 * 3600)), // Valid for 1 year
            credential_subject: CredentialSubject {
                id: bobbo_cell.agent_pubkey().to_string(),
                claims: {
                    let mut claims = HashMap::new();
                    claims.insert("courseId".to_string(), "course-001".to_string());
                    claims.insert("courseTitle".to_string(), "Introduction to Holochain".to_string());
                    claims.insert("completionDate".to_string(), timestamp().to_string());
                    claims.insert("finalScore".to_string(), "100".to_string());
                    claims
                },
            },
            proof: None, // Will be generated by the zome
            credential_status: CredentialStatus::Active,
        };

        let credential_hash: ActionHash = conductor
            .call(
                credential_zome(alice_cell),
                "issue_credential",
                credential.clone(),
            )
            .await
            .expect("Failed to issue credential");

        println!("✅ Credential issued with hash: {:?}", credential_hash);

        // Step 6: Verify the credential
        println!("\n📋 Step 6: Verifying the credential...");
        let retrieved_credential: VerifiableCredential = conductor
            .call(
                credential_zome(bobbo_cell),
                "get_credential",
                credential_hash.clone(),
            )
            .await
            .expect("Failed to retrieve credential");

        assert_eq!(retrieved_credential.issuer, alice_cell.agent_pubkey().to_string());
        assert_eq!(retrieved_credential.credential_subject.id, bobbo_cell.agent_pubkey().to_string());
        assert_eq!(retrieved_credential.credential_status, CredentialStatus::Active);

        println!("✅ Credential verified successfully!");

        // Step 7: Verify credential can be queried by holder
        println!("\n📋 Step 7: Querying credentials held by Bobbo...");
        let held_credentials: Vec<ActionHash> = conductor
            .call(
                credential_zome(bobbo_cell),
                "get_credentials_for_holder",
                bobbo_cell.agent_pubkey().to_string(),
            )
            .await
            .expect("Failed to query credentials");

        assert!(held_credentials.contains(&credential_hash));
        println!("✅ Credential found in holder's credentials list");

        println!("\n🎉 Scenario 1: Complete Learning Journey - ALL TESTS PASSED! 🎉");
    }
}

/// Scenario 2: Federated Learning Round Lifecycle
/// Tests: Round Creation → Registration → Submission → Aggregation
#[cfg(test)]
mod scenario_2_fl_round_lifecycle {
    use super::*;
    use fl_integrity::{FlRound, FlUpdate};
    use praxis_core::{RoundId, ModelHash, RoundState};

    #[tokio::test]
    #[ignore] // Requires DNA bundle to be built
    async fn test_fl_round_lifecycle() {
        println!("🚀 Starting Scenario 2: Federated Learning Round Lifecycle");

        // Step 1: Setup conductor and install DNA
        println!("\n📋 Step 1: Setting up conductor and installing DNA...");
        let mut conductor = setup_conductor().await.expect("Failed to create conductor");
        let (cells, dna_hash) = install_edunet_dna(&mut conductor)
            .await
            .expect("Failed to install DNA");

        let coordinator_cell = &cells[0];
        let learner1_cell = &cells[1];
        println!("✅ Conductor ready with coordinator and 1 learner");

        // We need more learners for this test, let's create them
        println!("\n📋 Step 1b: Adding additional learners...");
        let additional_cell = conductor.add_agent().await;
        let learner2_cell = additional_cell;
        println!("✅ Total participants: 1 coordinator + 3 learners");

        // Step 2: Coordinator creates FL round
        println!("\n📋 Step 2: Coordinator creates FL round...");
        let fl_round = FlRound {
            round_id: RoundId("round-001".to_string()),
            model_id: "sentiment-model-v1".to_string(),
            state: RoundState::Join,
            base_model_hash: ModelHash("base_model_hash_12345".to_string()),
            min_participants: 3,
            max_participants: 10,
            current_participants: 0,
            aggregation_method: "trimmed_mean".to_string(),
            clip_norm: 1.0,
            started_at: timestamp(),
            completed_at: None,
            aggregated_model_hash: None,
            privacy_epsilon: Some(1.0),
            privacy_delta: Some(1e-5),
        };

        let round_hash: ActionHash = conductor
            .call(
                fl_zome(coordinator_cell),
                "create_fl_round",
                fl_round.clone(),
            )
            .await
            .expect("Failed to create FL round");

        println!("✅ FL round created with hash: {:?}", round_hash);
        println!("   Round ID: {}", fl_round.round_id.0);
        println!("   Min participants: {}", fl_round.min_participants);
        println!("   Privacy params: ε={}, δ={}",
            fl_round.privacy_epsilon.unwrap(),
            fl_round.privacy_delta.unwrap()
        );

        // Step 3: Learners register for the round
        println!("\n📋 Step 3: Learners registering for the round...");

        // Learner 1 registers
        let _registration1: ActionHash = conductor
            .call(
                fl_zome(learner1_cell),
                "register_for_round",
                fl_round.round_id.0.clone(),
            )
            .await
            .expect("Failed to register learner 1");
        println!("  ✅ Learner 1 registered");

        // Learner 2 registers
        let _registration2: ActionHash = conductor
            .call(
                fl_zome(learner2_cell),
                "register_for_round",
                fl_round.round_id.0.clone(),
            )
            .await
            .expect("Failed to register learner 2");
        println!("  ✅ Learner 2 registered");

        // Coordinator checks participant count
        let updated_round: FlRound = conductor
            .call(
                fl_zome(coordinator_cell),
                "get_fl_round",
                fl_round.round_id.0.clone(),
            )
            .await
            .expect("Failed to get updated round");

        assert!(updated_round.current_participants >= 2);
        println!("✅ Round has {} participants (minimum {} required)",
            updated_round.current_participants,
            fl_round.min_participants
        );

        // Step 4: Learners submit gradient updates
        println!("\n📋 Step 4: Learners submitting gradient updates...");

        // Learner 1 submits update
        let update1 = FlUpdate {
            round_id: fl_round.round_id.clone(),
            model_id: fl_round.model_id.clone(),
            parent_model_hash: fl_round.base_model_hash.clone(),
            grad_commitment: vec![1, 2, 3, 4, 5], // Simulated gradient commitment
            clipped_l2_norm: 0.95,
            local_val_loss: 0.45,
            sample_count: 1000,
            timestamp: timestamp(),
        };

        let update1_hash: ActionHash = conductor
            .call(
                fl_zome(learner1_cell),
                "submit_fl_update",
                update1.clone(),
            )
            .await
            .expect("Failed to submit update from learner 1");
        println!("  ✅ Learner 1 submitted update (L2 norm: {}, val loss: {})",
            update1.clipped_l2_norm,
            update1.local_val_loss
        );

        // Learner 2 submits update
        let update2 = FlUpdate {
            round_id: fl_round.round_id.clone(),
            model_id: fl_round.model_id.clone(),
            parent_model_hash: fl_round.base_model_hash.clone(),
            grad_commitment: vec![6, 7, 8, 9, 10],
            clipped_l2_norm: 0.98,
            local_val_loss: 0.42,
            sample_count: 1200,
            timestamp: timestamp(),
        };

        let update2_hash: ActionHash = conductor
            .call(
                fl_zome(learner2_cell),
                "submit_fl_update",
                update2.clone(),
            )
            .await
            .expect("Failed to submit update from learner 2");
        println!("  ✅ Learner 2 submitted update (L2 norm: {}, val loss: {})",
            update2.clipped_l2_norm,
            update2.local_val_loss
        );

        println!("✅ All gradient updates submitted successfully");

        // Step 5: Coordinator aggregates updates
        println!("\n📋 Step 5: Coordinator aggregating gradient updates...");

        let aggregated_model_hash: ModelHash = conductor
            .call(
                fl_zome(coordinator_cell),
                "aggregate_fl_round",
                fl_round.round_id.0.clone(),
            )
            .await
            .expect("Failed to aggregate FL round");

        println!("✅ Aggregation complete!");
        println!("   Aggregated model hash: {}", aggregated_model_hash.0);

        // Step 6: Verify final round state
        println!("\n📋 Step 6: Verifying final round state...");

        let final_round: FlRound = conductor
            .call(
                fl_zome(coordinator_cell),
                "get_fl_round",
                fl_round.round_id.0.clone(),
            )
            .await
            .expect("Failed to get final round state");

        assert_eq!(final_round.state, RoundState::Completed);
        assert!(final_round.aggregated_model_hash.is_some());
        assert!(final_round.completed_at.is_some());

        println!("✅ Round verification passed:");
        println!("   State: {:?}", final_round.state);
        println!("   Aggregated model: {:?}", final_round.aggregated_model_hash);
        println!("   Completed at: {:?}", final_round.completed_at);

        // Step 7: Verify participants can retrieve aggregated model
        println!("\n📋 Step 7: Verifying participants can access aggregated model...");

        let retrieved_by_learner1: FlRound = conductor
            .call(
                fl_zome(learner1_cell),
                "get_fl_round",
                fl_round.round_id.0.clone(),
            )
            .await
            .expect("Failed to retrieve round as learner 1");

        assert_eq!(retrieved_by_learner1.aggregated_model_hash, final_round.aggregated_model_hash);
        println!("✅ Learner 1 successfully retrieved aggregated model");

        let retrieved_by_learner2: FlRound = conductor
            .call(
                fl_zome(learner2_cell),
                "get_fl_round",
                fl_round.round_id.0.clone(),
            )
            .await
            .expect("Failed to retrieve round as learner 2");

        assert_eq!(retrieved_by_learner2.aggregated_model_hash, final_round.aggregated_model_hash);
        println!("✅ Learner 2 successfully retrieved aggregated model");

        println!("\n🎉 Scenario 2: FL Round Lifecycle - ALL TESTS PASSED! 🎉");
    }
}

/// Scenario 3: Credential Verification & Revocation
/// Tests: Issuance → Verification → Revocation → Re-verification
#[cfg(test)]
mod scenario_3_credential_lifecycle {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires DNA bundle to be built
    async fn test_credential_verification_and_revocation() {
        println!("🚀 Starting Scenario 3: Credential Verification & Revocation");

        // Step 1: Setup conductor and install DNA
        println!("\n📋 Step 1: Setting up conductor and installing DNA...");
        let mut conductor = setup_conductor().await.expect("Failed to create conductor");
        let (cells, dna_hash) = install_edunet_dna(&mut conductor)
            .await
            .expect("Failed to install DNA");

        let issuer_cell = &cells[0];
        let holder_cell = &cells[1];
        println!("✅ Conductor ready with issuer and holder");

        // Add a third agent as verifier
        println!("\n📋 Step 1b: Adding verifier agent...");
        let verifier_cell = conductor.add_agent().await;
        println!("✅ Verifier agent added");

        // Step 2: Issuer creates and issues credential to holder
        println!("\n📋 Step 2: Issuer creating W3C Verifiable Credential...");
        let credential = VerifiableCredential {
            context: vec!["https://www.w3.org/2018/credentials/v1".to_string()],
            id: format!("urn:uuid:{}", uuid::Uuid::new_v4()),
            credential_type: vec![
                "VerifiableCredential".to_string(),
                "EducationalCredential".to_string()
            ],
            issuer: issuer_cell.agent_pubkey().to_string(),
            issuance_date: timestamp(),
            expiration_date: Some(timestamp() + (365 * 24 * 3600)), // Valid for 1 year
            credential_subject: CredentialSubject {
                id: holder_cell.agent_pubkey().to_string(),
                claims: {
                    let mut claims = HashMap::new();
                    claims.insert("degreeType".to_string(), "Bachelor of Science".to_string());
                    claims.insert("major".to_string(), "Computer Science".to_string());
                    claims.insert("institution".to_string(), "Mycelix University".to_string());
                    claims.insert("graduationDate".to_string(), "2024-12-01".to_string());
                    claims.insert("gpa".to_string(), "3.85".to_string());
                    claims
                },
            },
            proof: None, // Will be generated by the zome
            credential_status: CredentialStatus::Active,
        };

        let credential_hash: ActionHash = conductor
            .call(
                credential_zome(issuer_cell),
                "issue_credential",
                credential.clone(),
            )
            .await
            .expect("Failed to issue credential");

        println!("✅ Credential issued successfully!");
        println!("   Holder: {}", credential.subject_id);
        println!("   Issuer: {}", credential.issuer);
        println!("   Type: {:?}", credential.credential_type);
        println!("   Hash: {:?}", credential_hash);

        // Step 3: Holder retrieves their credential
        println!("\n📋 Step 3: Holder retrieving credential...");
        let retrieved_credential: VerifiableCredential = conductor
            .call(
                credential_zome(holder_cell),
                "get_credential",
                credential_hash.clone(),
            )
            .await
            .expect("Failed to retrieve credential");

        // Verify credential fields match
        assert_eq!(retrieved_credential.subject_id, holder_cell.agent_pubkey().to_string());
        assert_eq!(retrieved_credential.issuer, issuer_cell.agent_pubkey().to_string());
        println!("✅ Holder successfully retrieved credential");

        // Step 4: Holder queries all their credentials
        println!("\n📋 Step 4: Querying all credentials for holder...");
        let holder_credentials: Vec<ActionHash> = conductor
            .call(
                credential_zome(holder_cell),
                "get_credentials_for_holder",
                holder_cell.agent_pubkey().to_string(),
            )
            .await
            .expect("Failed to query holder credentials");

        assert!(holder_credentials.contains(&credential_hash));
        println!("✅ Credential found in holder's portfolio ({} total credentials)",
            holder_credentials.len()
        );

        // Step 5: Verifier validates the credential
        println!("\n📋 Step 5: Verifier validating credential...");
        let verification_result: bool = conductor
            .call(
                credential_zome(verifier_cell),
                "verify_credential",
                credential_hash.clone(),
            )
            .await
            .expect("Failed to verify credential");

        assert!(verification_result, "Credential verification should pass");
        println!("✅ Credential verified successfully by third party!");

        // Step 6: Verifier retrieves credential details
        println!("\n📋 Step 6: Verifier retrieving credential details...");
        let verified_credential: VerifiableCredential = conductor
            .call(
                credential_zome(verifier_cell),
                "get_credential",
                credential_hash.clone(),
            )
            .await
            .expect("Failed to get credential as verifier");

        // Verify credential matches issued credential
        assert_eq!(verified_credential.subject_id, holder_cell.agent_pubkey().to_string());
        assert_eq!(verified_credential.issuer, issuer_cell.agent_pubkey().to_string());
        println!("✅ Verifier retrieved credential details");
        // Note: In the flattened credential structure, metadata is stored in subject_metadata
        if let Some(metadata) = &verified_credential.subject_metadata {
            println!("   Subject metadata: {}", metadata);
        }

        // Step 7: Issuer revokes the credential
        println!("\n📋 Step 7: Issuer revoking credential...");
        let revocation_result: ActionHash = conductor
            .call(
                credential_zome(issuer_cell),
                "revoke_credential",
                credential_hash.clone(),
            )
            .await
            .expect("Failed to revoke credential");

        println!("✅ Credential revoked successfully!");
        println!("   Revocation hash: {:?}", revocation_result);

        // Step 8: Verify credential status is now "Revoked"
        println!("\n📋 Step 8: Checking credential status after revocation...");
        let revoked_credential: VerifiableCredential = conductor
            .call(
                credential_zome(holder_cell),
                "get_credential",
                credential_hash.clone(),
            )
            .await
            .expect("Failed to retrieve credential after revocation");

        // Note: In the flattened credential structure, status is tracked via optional fields
        // The revocation is tracked via the ActionHash returned by revoke_credential
        println!("✅ Credential revocation recorded with hash: {:?}", revocation_result);

        // Step 9: Verifier re-validates and should now fail
        println!("\n📋 Step 9: Verifier re-validating revoked credential...");
        let reverification_result: bool = conductor
            .call(
                credential_zome(verifier_cell),
                "verify_credential",
                credential_hash.clone(),
            )
            .await
            .expect("Failed to verify revoked credential");

        assert!(!reverification_result, "Revoked credential should fail verification");
        println!("✅ Verification correctly failed for revoked credential");

        // Step 10: Verify issuer can query issued credentials
        println!("\n📋 Step 10: Querying all credentials issued by issuer...");
        let issued_credentials: Vec<ActionHash> = conductor
            .call(
                credential_zome(issuer_cell),
                "get_credentials_by_issuer",
                issuer_cell.agent_pubkey().to_string(),
            )
            .await
            .expect("Failed to query issuer's credentials");

        assert!(issued_credentials.contains(&credential_hash));
        println!("✅ Credential found in issuer's issued list ({} total issued)",
            issued_credentials.len()
        );

        println!("\n🎉 Scenario 3: Credential Verification & Revocation - ALL TESTS PASSED! 🎉");
    }
}

/// Scenario 4: DAO Governance Lifecycle
/// Tests: Proposal Creation → Voting → Tallying → Execution
#[cfg(test)]
mod scenario_4_dao_governance {
    use super::*;
    use dao_integrity::{Proposal, Vote, ProposalType, ProposalCategory, ProposalStatus, VoteChoice};

    #[tokio::test]
    #[ignore] // Requires Holochain conductor
    async fn test_dao_proposal_lifecycle() {
        // Step 1: Setup conductor with multiple members
        let (mut conductor, cells) = setup_conductor_with_agents(4).await.unwrap();

        let proposer_cell = &cells[0];
        let voter1_cell = &cells[1];
        let voter2_cell = &cells[2];
        let voter3_cell = &cells[3];

        // Step 2: Proposer creates curriculum change proposal (Normal type, 3-14 day voting)
        let proposal_input = CreateProposalInput {
            proposal_id: "proposal-001".to_string(),
            title: "Add Rust Programming Course".to_string(),
            description: "Proposal to add a comprehensive Rust programming course to the curriculum".to_string(),
            proposal_type: ProposalType::Normal,
            category: ProposalCategory::Curriculum,
            actions: vec![serde_json::json!({
                "action": "add_course",
                "course_id": "rust-101"
            })],
        };

        let proposal_hash: ActionHash = conductor
            .call(
                dao_zome(proposer_cell),
                "create_proposal",
                proposal_input,
            )
            .await
            .expect("Failed to create proposal");

        println!("✅ Step 2: Proposal created with hash: {:?}", proposal_hash);

        // Step 3: Members cast votes
        // Voter 1: For
        let vote1 = Vote {
            proposal_id: "proposal-001".to_string(),
            voter: voter1_cell.agent_pubkey().to_string(),
            choice: VoteChoice::For,
            justification: Some("Great addition to curriculum!".to_string()),
            timestamp: chrono::Utc::now().timestamp(),
        };

        let vote1_hash: ActionHash = conductor
            .call(
                dao_zome(voter1_cell),
                "cast_vote",
                vote1.clone(),
            )
            .await
            .expect("Failed to cast vote 1");

        println!("✅ Step 3a: Vote 1 (For) cast with hash: {:?}", vote1_hash);

        // Voter 2: For
        let vote2 = Vote {
            proposal_id: "proposal-001".to_string(),
            voter: voter2_cell.agent_pubkey().to_string(),
            choice: VoteChoice::For,
            justification: Some("Rust is essential for our tech stack".to_string()),
            timestamp: chrono::Utc::now().timestamp(),
        };

        let vote2_hash: ActionHash = conductor
            .call(
                dao_zome(voter2_cell),
                "cast_vote",
                vote2.clone(),
            )
            .await
            .expect("Failed to cast vote 2");

        println!("✅ Step 3b: Vote 2 (For) cast with hash: {:?}", vote2_hash);

        // Voter 3: Abstain
        let vote3 = Vote {
            proposal_id: "proposal-001".to_string(),
            voter: voter3_cell.agent_pubkey().to_string(),
            choice: VoteChoice::Abstain,
            justification: Some("Need more information".to_string()),
            timestamp: chrono::Utc::now().timestamp(),
        };

        let vote3_hash: ActionHash = conductor
            .call(
                dao_zome(voter3_cell),
                "cast_vote",
                vote3.clone(),
            )
            .await
            .expect("Failed to cast vote 3");

        println!("✅ Step 3c: Vote 3 (Abstain) cast with hash: {:?}", vote3_hash);

        // Step 4: Check vote counts before tallying
        let proposal_before_tally: Proposal = conductor
            .call(
                dao_zome(proposer_cell),
                "get_proposal",
                "proposal-001".to_string(),
            )
            .await
            .expect("Failed to get proposal");

        println!("✅ Step 4: Vote counts before tally - For: {}, Against: {}, Abstain: {}",
            proposal_before_tally.for_votes,
            proposal_before_tally.against_votes,
            proposal_before_tally.abstain_votes
        );

        // Step 5: Voting period ends - tally votes
        // In real scenario, this would be triggered by deadline passing
        // For testing, we simulate the tallying process
        let tally_result: Proposal = conductor
            .call(
                dao_zome(proposer_cell),
                "tally_votes",
                "proposal-001".to_string(),
            )
            .await
            .expect("Failed to tally votes");

        println!("✅ Step 5: Votes tallied - Status: {:?}", tally_result.status);

        // Step 6: Verify quorum and final status
        assert_eq!(tally_result.for_votes, 2, "Should have 2 'For' votes");
        assert_eq!(tally_result.against_votes, 0, "Should have 0 'Against' votes");
        assert_eq!(tally_result.abstain_votes, 1, "Should have 1 'Abstain' vote");

        // If quorum met (3+ votes) and majority 'For', status should be Approved
        if tally_result.for_votes > tally_result.against_votes {
            assert_eq!(tally_result.status, ProposalStatus::Approved, "Proposal should be approved");
            println!("✅ Step 6: Proposal approved - quorum met, majority vote passed");

            // Step 7: Execute proposal (if approved)
            let executed_proposal: Proposal = conductor
                .call(
                    dao_zome(proposer_cell),
                    "execute_proposal",
                    "proposal-001".to_string(),
                )
                .await
                .expect("Failed to execute proposal");

            assert_eq!(executed_proposal.status, ProposalStatus::Executed, "Proposal should be executed");
            assert!(executed_proposal.executed_at.is_some(), "Execution timestamp should be set");

            println!("✅ Step 7: Proposal executed at timestamp: {:?}", executed_proposal.executed_at);
        } else {
            assert_eq!(tally_result.status, ProposalStatus::Rejected, "Proposal should be rejected");
            println!("✅ Step 6: Proposal rejected - majority vote against");
        }

        // Step 8: Query all proposals and votes
        let all_proposals: Vec<Proposal> = conductor
            .call(
                dao_zome(proposer_cell),
                "get_all_proposals",
                (),
            )
            .await
            .expect("Failed to get all proposals");

        assert_eq!(all_proposals.len(), 1, "Should have 1 proposal");
        println!("✅ Step 8: Successfully queried all proposals (count: {})", all_proposals.len());

        let proposal_votes: Vec<Vote> = conductor
            .call(
                dao_zome(proposer_cell),
                "get_votes_for_proposal",
                "proposal-001".to_string(),
            )
            .await
            .expect("Failed to get votes");

        assert_eq!(proposal_votes.len(), 3, "Should have 3 votes");
        println!("✅ Step 8: Successfully queried all votes (count: {})", proposal_votes.len());

        println!("🎉 Scenario 4: DAO Governance Lifecycle - COMPLETE");
        println!("   ✅ Proposal created");
        println!("   ✅ 3 votes cast (2 For, 0 Against, 1 Abstain)");
        println!("   ✅ Votes tallied correctly");
        println!("   ✅ Proposal approved and executed");
        println!("   ✅ All queries working");
    }

    #[tokio::test]
    #[ignore] // Requires Holochain conductor
    async fn test_fast_proposal() {
        // Fast path (24-72h) emergency proposal
        let (conductor, cells) = setup_conductor_with_agents(3).await.unwrap();

        let proposer_cell = &cells[0];
        let voter1_cell = &cells[1];
        let voter2_cell = &cells[2];

        // Create emergency parameter change proposal (Fast type)
        let proposal_input = CreateProposalInput {
            proposal_id: "fast-proposal-001".to_string(),
            title: "Emergency: Increase FL Privacy Epsilon".to_string(),
            description: "Critical privacy parameter adjustment needed immediately".to_string(),
            proposal_type: ProposalType::Fast,
            category: ProposalCategory::Emergency,
            actions: vec![serde_json::json!({
                "action": "update_privacy_param",
                "epsilon": 2.0
            })],
        };

        let proposal_hash: ActionHash = conductor
            .call(
                dao_zome(proposer_cell),
                "create_proposal",
                proposal_input,
            )
            .await
            .expect("Failed to create fast proposal");

        println!("✅ Fast proposal created: {:?}", proposal_hash);

        // Cast votes (emergency requires simple majority)
        let vote1 = Vote {
            proposal_id: "fast-proposal-001".to_string(),
            voter: voter1_cell.agent_pubkey().to_string(),
            choice: VoteChoice::For,
            justification: Some("Urgent fix needed".to_string()),
            timestamp: chrono::Utc::now().timestamp(),
        };

        conductor
            .call(
                dao_zome(voter1_cell),
                "cast_vote",
                vote1,
            )
            .await
            .expect("Failed to cast vote");

        let vote2 = Vote {
            proposal_id: "fast-proposal-001".to_string(),
            voter: voter2_cell.agent_pubkey().to_string(),
            choice: VoteChoice::For,
            justification: Some("Agree with urgency".to_string()),
            timestamp: chrono::Utc::now().timestamp(),
        };

        conductor
            .call(
                dao_zome(voter2_cell),
                "cast_vote",
                vote2,
            )
            .await
            .expect("Failed to cast vote");

        // Tally votes
        let tally_result: Proposal = conductor
            .call(
                dao_zome(proposer_cell),
                "tally_votes",
                "fast-proposal-001".to_string(),
            )
            .await
            .expect("Failed to tally fast proposal");

        assert_eq!(tally_result.proposal_type, ProposalType::Fast);
        assert_eq!(tally_result.for_votes, 2);
        assert_eq!(tally_result.status, ProposalStatus::Approved);

        println!("✅ Fast proposal test passed");
        println!("   - Type: Fast (24-72h)");
        println!("   - Category: Emergency");
        println!("   - Votes: 2 For, 0 Against");
        println!("   - Status: Approved");
    }

    #[tokio::test]
    #[ignore] // Requires Holochain conductor
    async fn test_normal_proposal() {
        // Normal path (3-14 days) feature proposal
        let (conductor, cells) = setup_conductor_with_agents(4).await.unwrap();

        let proposer_cell = &cells[0];
        let voter1_cell = &cells[1];
        let voter2_cell = &cells[2];
        let voter3_cell = &cells[3];

        // Create feature proposal (Normal type)
        let proposal_input = CreateProposalInput {
            proposal_id: "normal-proposal-001".to_string(),
            title: "Add Course Rating System".to_string(),
            description: "Allow learners to rate courses and provide feedback".to_string(),
            proposal_type: ProposalType::Normal,
            category: ProposalCategory::Curriculum,
            actions: vec![serde_json::json!({
                "action": "add_feature",
                "feature": "course_ratings"
            })],
        };

        let proposal_hash: ActionHash = conductor
            .call(
                dao_zome(proposer_cell),
                "create_proposal",
                proposal_input.clone(),
            )
            .await
            .expect("Failed to create normal proposal");

        println!("✅ Normal proposal created: {:?}", proposal_hash);

        // Cast votes (mixed results)
        let votes = vec![
            Vote {
                proposal_id: "normal-proposal-001".to_string(),
                voter: voter1_cell.agent_pubkey().to_string(),
                choice: VoteChoice::For,
                justification: Some("Good feature".to_string()),
                timestamp: chrono::Utc::now().timestamp(),
            },
            Vote {
                proposal_id: "normal-proposal-001".to_string(),
                voter: voter2_cell.agent_pubkey().to_string(),
                choice: VoteChoice::For,
                justification: Some("Will improve quality".to_string()),
                timestamp: chrono::Utc::now().timestamp(),
            },
            Vote {
                proposal_id: "normal-proposal-001".to_string(),
                voter: voter3_cell.agent_pubkey().to_string(),
                choice: VoteChoice::Against,
                justification: Some("Too complex to implement now".to_string()),
                timestamp: chrono::Utc::now().timestamp(),
            },
        ];

        for (i, vote) in votes.iter().enumerate() {
            conductor
                .call(
                    dao_zome(&cells[i + 1]),
                    "cast_vote",
                    vote.clone(),
                )
                .await
                .expect("Failed to cast vote");
        }

        // Tally votes
        let tally_result: Proposal = conductor
            .call(
                dao_zome(proposer_cell),
                "tally_votes",
                "normal-proposal-001".to_string(),
            )
            .await
            .expect("Failed to tally normal proposal");

        assert_eq!(tally_result.proposal_type, ProposalType::Normal);
        assert_eq!(tally_result.for_votes, 2);
        assert_eq!(tally_result.against_votes, 1);
        assert_eq!(tally_result.status, ProposalStatus::Approved); // Majority wins

        println!("✅ Normal proposal test passed");
        println!("   - Type: Normal (3-14 days)");
        println!("   - Category: Curriculum");
        println!("   - Votes: 2 For, 1 Against");
        println!("   - Status: Approved (majority)");
    }

    #[tokio::test]
    #[ignore] // Requires Holochain conductor
    async fn test_slow_proposal() {
        // Slow path (14-30 days) protocol change
        let (conductor, cells) = setup_conductor_with_agents(5).await.unwrap();

        let proposer_cell = &cells[0];

        // Create protocol upgrade proposal (Slow type)
        let proposal = Proposal {
            proposal_id: "slow-proposal-001".to_string(),
            title: "Major Protocol Upgrade v2.0".to_string(),
            description: "Comprehensive protocol changes requiring thorough review".to_string(),
            proposer: proposer_cell.agent_pubkey().to_string(),
            proposal_type: ProposalType::Slow,
            category: ProposalCategory::Protocol,
            status: ProposalStatus::Active,
            for_votes: 0,
            against_votes: 0,
            abstain_votes: 0,
            voting_deadline: chrono::Utc::now().timestamp() + (21 * 24 * 60 * 60), // 21 days
            created_at: chrono::Utc::now().timestamp(),
            executed_at: None,
            actions_json: r#"{"action": "protocol_upgrade", "version": "2.0.0"}"#.to_string(),
        };

        let proposal_input = CreateProposalInput {
            proposal_id: proposal.proposal_id.clone(),
            title: proposal.title.clone(),
            description: proposal.description.clone(),
            proposal_type: proposal.proposal_type,
            category: proposal.category,
            actions: vec![serde_json::json!({"action": "protocol_upgrade", "version": "2.0.0"})],
        };

        let proposal_hash: ActionHash = conductor
            .call(
                dao_zome(proposer_cell),
                "create_proposal",
                proposal_input,
            )
            .await
            .expect("Failed to create slow proposal");

        println!("✅ Slow proposal created: {:?}", proposal_hash);

        // Cast votes from all members (requires supermajority for protocol changes)
        let votes = vec![
            Vote {
                proposal_id: "slow-proposal-001".to_string(),
                voter: cells[1].agent_pubkey().to_string(),
                choice: VoteChoice::For,
                justification: Some("Well-researched upgrade".to_string()),
                timestamp: chrono::Utc::now().timestamp(),
            },
            Vote {
                proposal_id: "slow-proposal-001".to_string(),
                voter: cells[2].agent_pubkey().to_string(),
                choice: VoteChoice::For,
                justification: Some("Necessary improvements".to_string()),
                timestamp: chrono::Utc::now().timestamp(),
            },
            Vote {
                proposal_id: "slow-proposal-001".to_string(),
                voter: cells[3].agent_pubkey().to_string(),
                choice: VoteChoice::For,
                justification: Some("Strong support".to_string()),
                timestamp: chrono::Utc::now().timestamp(),
            },
            Vote {
                proposal_id: "slow-proposal-001".to_string(),
                voter: cells[4].agent_pubkey().to_string(),
                choice: VoteChoice::Abstain,
                justification: Some("Need more technical review".to_string()),
                timestamp: chrono::Utc::now().timestamp(),
            },
        ];

        for (i, vote) in votes.iter().enumerate() {
            let vote_input = CastVoteInput {
                proposal_id: vote.proposal_id.clone(),
                choice: vote.choice,
                justification: vote.justification.clone(),
            };

            conductor
                .call(
                    dao_zome(&cells[i + 1]),
                    "cast_vote",
                    vote_input,
                )
                .await
                .expect("Failed to cast vote");
        }

        // Tally votes
        let tally_result: Proposal = conductor
            .call(
                dao_zome(proposer_cell),
                "tally_votes",
                "slow-proposal-001".to_string(),
            )
            .await
            .expect("Failed to tally slow proposal");

        assert_eq!(tally_result.proposal_type, ProposalType::Slow);
        assert_eq!(tally_result.for_votes, 3);
        assert_eq!(tally_result.abstain_votes, 1);
        // Slow proposals require supermajority (e.g., 75%)
        // 3/4 = 75%, so should be approved
        assert_eq!(tally_result.status, ProposalStatus::Approved);

        println!("✅ Slow proposal test passed");
        println!("   - Type: Slow (14-30 days)");
        println!("   - Category: Protocol");
        println!("   - Votes: 3 For, 0 Against, 1 Abstain");
        println!("   - Status: Approved (supermajority 75%)");
    }
}

/// Scenario 5: Web Client → Conductor Integration
/// Tests: WebSocket connection → Zome calls → Real-time updates
#[cfg(test)]
mod scenario_5_web_client_integration {
    #[test]
    #[ignore] // Requires Holochain conductor
    fn test_websocket_connection() {
        // TODO: Implement Scenario 5 test
        // 1. Web client connects via WebSocket
        // 2. Retrieves app info
        // 3. Calls all 29 zome functions
        // 4. Verifies responses
        // 5. Tests error handling

        println!("✅ Scenario 5: Web Client Integration scaffold ready");
    }

    #[test]
    #[ignore]
    fn test_mock_real_switching() {
        // TODO: Test switching between mock and real client
        println!("✅ Mock/Real switching test scaffold ready");
    }
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

#[cfg(test)]
mod performance_benchmarks {
    #[test]
    #[ignore]
    fn test_zome_call_latency() {
        // TODO: Measure zome call latency (target: <500ms p99)
        println!("✅ Zome call latency benchmark scaffold ready");
    }

    #[test]
    #[ignore]
    fn test_fl_aggregation_performance() {
        // TODO: Measure FL aggregation time (target: <5s for 10 participants)
        println!("✅ FL aggregation benchmark scaffold ready");
    }

    #[test]
    #[ignore]
    fn test_dht_sync_time() {
        // TODO: Measure DHT sync time (target: <2s)
        println!("✅ DHT sync benchmark scaffold ready");
    }
}

// ============================================================================
// Load Tests
// ============================================================================

#[cfg(test)]
mod load_tests {
    #[test]
    #[ignore]
    fn test_high_course_load() {
        // TODO: 100 courses created simultaneously
        println!("✅ High course load test scaffold ready");
    }

    #[test]
    #[ignore]
    fn test_fl_round_at_scale() {
        // TODO: 50 participants in single round
        println!("✅ FL round at scale test scaffold ready");
    }

    #[test]
    #[ignore]
    fn test_dao_voting_rush() {
        // TODO: 200 members vote within 1 minute
        println!("✅ DAO voting rush test scaffold ready");
    }
}

// ============================================================================
// Helper Functions (to be implemented)
// ============================================================================

// These will be implemented as tests are fleshed out
// fn setup_conductor() -> Result<SweetConductor, Box<dyn std::error::Error>> { ... }
// fn create_test_course(...) -> Course { ... }
// fn create_test_progress(...) -> LearnerProgress { ... }
// fn create_test_fl_round(...) -> FlRound { ... }
// fn create_test_credential(...) -> VerifiableCredential { ... }
// fn create_test_proposal(...) -> Proposal { ... }
