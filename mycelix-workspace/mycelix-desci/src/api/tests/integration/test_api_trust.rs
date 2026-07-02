// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trust API endpoint tests
//!
//! Tests all trust-related API endpoints.

use super::helpers::*;
use serde_json::Value;

#[tokio::test]
async fn test_get_trust_score_new_participant() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let participant = "new.participant@example.com";
    let response = server.get(&format!("/trust/{}", participant)).await;

    assert_eq!(response.status(), 200);

    let trust_response: Value = response.json().await.expect("Failed to parse trust response");
    assert_eq!(trust_response["participant"], participant);
    assert!(trust_response["score"].is_number(), "Should have a numeric score");
    assert!(trust_response["last_updated"].is_string(), "Should have timestamp");
}

#[tokio::test]
async fn test_update_trust_score_positive() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let participant = "researcher@university.edu";

    // Get initial score
    let response = server.get(&format!("/trust/{}", participant)).await;
    let initial: Value = response.json().await.expect("Failed to parse response");
    let initial_score = initial["score"].as_f64().unwrap();

    // Apply positive update
    let update_request = serde_json::json!({
        "positive": true,
        "weight": 1.0
    });

    let response = server.put(&format!("/trust/{}", participant), &update_request).await;
    assert_eq!(response.status(), 200);

    let updated: Value = response.json().await.expect("Failed to parse response");
    let new_score = updated["score"].as_f64().unwrap();

    assert!(
        new_score >= initial_score,
        "Positive update should not decrease score"
    );
}

#[tokio::test]
async fn test_update_trust_score_negative() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let participant = "test.participant@example.com";

    // Get initial score
    let response = server.get(&format!("/trust/{}", participant)).await;
    let initial: Value = response.json().await.expect("Failed to parse response");
    let initial_score = initial["score"].as_f64().unwrap();

    // Apply negative update
    let update_request = serde_json::json!({
        "positive": false,
        "weight": 1.0
    });

    let response = server.put(&format!("/trust/{}", participant), &update_request).await;
    assert_eq!(response.status(), 200);

    let updated: Value = response.json().await.expect("Failed to parse response");
    let new_score = updated["score"].as_f64().unwrap();

    assert!(
        new_score <= initial_score,
        "Negative update should not increase score"
    );
}

#[tokio::test]
async fn test_update_trust_score_with_different_weights() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let participant1 = "participant1@test.com";
    let participant2 = "participant2@test.com";

    // Get initial scores
    let response = server.get(&format!("/trust/{}", participant1)).await;
    let initial1: Value = response.json().await.expect("Failed to parse response");
    let initial_score1 = initial1["score"].as_f64().unwrap();

    let response = server.get(&format!("/trust/{}", participant2)).await;
    let initial2: Value = response.json().await.expect("Failed to parse response");
    let initial_score2 = initial2["score"].as_f64().unwrap();

    // Apply updates with different weights
    let update1 = serde_json::json!({
        "positive": true,
        "weight": 0.5
    });

    let update2 = serde_json::json!({
        "positive": true,
        "weight": 2.0
    });

    server.put(&format!("/trust/{}", participant1), &update1).await;
    server.put(&format!("/trust/{}", participant2), &update2).await;

    // Verify scores changed appropriately
    let response = server.get(&format!("/trust/{}", participant1)).await;
    let updated1: Value = response.json().await.expect("Failed to parse response");
    let new_score1 = updated1["score"].as_f64().unwrap();

    let response = server.get(&format!("/trust/{}", participant2)).await;
    let updated2: Value = response.json().await.expect("Failed to parse response");
    let new_score2 = updated2["score"].as_f64().unwrap();

    assert!(new_score1 >= initial_score1, "Score should increase with positive update");
    assert!(new_score2 >= initial_score2, "Score should increase with positive update");
}

#[tokio::test]
async fn test_get_trust_stats() {
    let server = TestServer::start().await.expect("Failed to start test server");

    // Create some trust updates
    for i in 0..5 {
        let participant = format!("participant{}@test.com", i);
        let update = serde_json::json!({
            "positive": i % 2 == 0,
            "weight": 1.0
        });
        server.put(&format!("/trust/{}", participant), &update).await;
    }

    // Get trust stats
    let response = server.get("/trust/stats").await;
    assert_eq!(response.status(), 200);

    let stats: Value = response.json().await.expect("Failed to parse stats response");

    assert!(stats["total_participants"].is_number(), "Should have total_participants");
    assert!(stats["average_score"].is_number(), "Should have average_score");
    assert!(stats["median_score"].is_number(), "Should have median_score");

    let total_participants = stats["total_participants"].as_u64().unwrap();
    assert!(total_participants >= 5, "Should have at least 5 participants");
}

#[tokio::test]
async fn test_trust_score_bounds() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let participant = "bounds.test@example.com";

    // Apply many positive updates
    for _ in 0..20 {
        let update = serde_json::json!({
            "positive": true,
            "weight": 5.0
        });
        server.put(&format!("/trust/{}", participant), &update).await;
    }

    // Check score is bounded
    let response = server.get(&format!("/trust/{}", participant)).await;
    let result: Value = response.json().await.expect("Failed to parse response");
    let score = result["score"].as_f64().unwrap();

    assert!(score >= 0.0, "Score should not be negative");
    assert!(score <= 100.0, "Score should be bounded (typically ≤ 100)");

    // Apply many negative updates
    for _ in 0..20 {
        let update = serde_json::json!({
            "positive": false,
            "weight": 5.0
        });
        server.put(&format!("/trust/{}", participant), &update).await;
    }

    // Check score is still bounded
    let response = server.get(&format!("/trust/{}", participant)).await;
    let result: Value = response.json().await.expect("Failed to parse response");
    let score = result["score"].as_f64().unwrap();

    assert!(score >= 0.0, "Score should not be negative even after many negative updates");
}

#[tokio::test]
async fn test_trust_multiple_participants() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let participants = vec![
        "alice@research.org",
        "bob@university.edu",
        "carol@institute.com",
    ];

    // Update scores for all participants
    for participant in &participants {
        let update = serde_json::json!({
            "positive": true,
            "weight": 1.0
        });
        server.put(&format!("/trust/{}", participant), &update).await;
    }

    // Verify all participants have scores
    for participant in &participants {
        let response = server.get(&format!("/trust/{}", participant)).await;
        assert_eq!(response.status(), 200);

        let result: Value = response.json().await.expect("Failed to parse response");
        assert_eq!(result["participant"], *participant);
        assert!(result["score"].is_number());
    }
}

#[tokio::test]
async fn test_trust_score_persistence() {
    let server = TestServer::start().await.expect("Failed to start test server");

    let participant = "persistent@example.com";

    // Update score
    let update = serde_json::json!({
        "positive": true,
        "weight": 2.5
    });
    server.put(&format!("/trust/{}", participant), &update).await;

    // Get score
    let response = server.get(&format!("/trust/{}", participant)).await;
    let first: Value = response.json().await.expect("Failed to parse response");
    let first_score = first["score"].as_f64().unwrap();

    // Get score again - should be the same
    let response = server.get(&format!("/trust/{}", participant)).await;
    let second: Value = response.json().await.expect("Failed to parse response");
    let second_score = second["score"].as_f64().unwrap();

    assert_eq!(first_score, second_score, "Score should persist between requests");
}
