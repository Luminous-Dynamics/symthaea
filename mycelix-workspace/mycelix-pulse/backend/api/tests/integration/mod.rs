// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for Mycelix Mail API
//!
//! Tests the full API request/response cycle

pub mod auth_tests;
pub mod email_tests;
pub mod contact_tests;
pub mod trust_tests;
pub mod collaboration_tests;
pub mod security_tests;
pub mod migration_tests;
pub mod ml_tests;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use serde_json::json;
use sqlx::PgPool;
use uuid::Uuid;
use std::sync::Arc;
use once_cell::sync::Lazy;
use tokio::sync::Mutex;

// ============================================================================
// Test Context
// ============================================================================

/// Shared test context for integration tests
pub struct TestContext {
    pub pool: PgPool,
    pub test_user_id: Uuid,
    pub test_user_email: String,
    pub auth_token: String,
}

impl TestContext {
    /// Create a new test context with isolated database
    pub async fn new() -> Self {
        let database_url = std::env::var("TEST_DATABASE_URL")
            .unwrap_or_else(|_| "postgres://test:test@localhost:5432/mycelix_test".to_string());

        let pool = PgPool::connect(&database_url)
            .await
            .expect("Failed to connect to test database");

        // Run migrations
        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .expect("Failed to run migrations");

        // Create test user
        let test_user_id = Uuid::new_v4();
        let test_user_email = format!("test-{}@example.com", Uuid::new_v4());
        let auth_token = format!("test_token_{}", Uuid::new_v4());

        sqlx::query(
            "INSERT INTO users (id, email, password_hash, created_at) VALUES ($1, $2, $3, NOW())",
        )
        .bind(test_user_id)
        .bind(&test_user_email)
        .bind("$argon2id$v=19$m=65536,t=3,p=4$test_hash")
        .execute(&pool)
        .await
        .expect("Failed to create test user");

        // Create session
        sqlx::query(
            "INSERT INTO user_sessions (id, user_id, token_hash, expires_at, created_at)
             VALUES ($1, $2, $3, NOW() + INTERVAL '1 hour', NOW())",
        )
        .bind(Uuid::new_v4())
        .bind(test_user_id)
        .bind(&auth_token)
        .execute(&pool)
        .await
        .expect("Failed to create test session");

        Self {
            pool,
            test_user_id,
            test_user_email,
            auth_token,
        }
    }

    /// Create a secondary test user
    pub async fn create_user(&self, email: &str) -> Uuid {
        let user_id = Uuid::new_v4();
        sqlx::query(
            "INSERT INTO users (id, email, password_hash, created_at) VALUES ($1, $2, $3, NOW())",
        )
        .bind(user_id)
        .bind(email)
        .bind("$argon2id$v=19$m=65536,t=3,p=4$test_hash")
        .execute(&self.pool)
        .await
        .expect("Failed to create user");
        user_id
    }

    /// Create a test email
    pub async fn create_email(&self, subject: &str, body: &str) -> Uuid {
        let email_id = Uuid::new_v4();
        sqlx::query(
            "INSERT INTO emails (id, user_id, subject, body_text, from_address, to_addresses, received_at)
             VALUES ($1, $2, $3, $4, 'sender@example.com', ARRAY['recipient@example.com'], NOW())",
        )
        .bind(email_id)
        .bind(self.test_user_id)
        .bind(subject)
        .bind(body)
        .execute(&self.pool)
        .await
        .expect("Failed to create email");
        email_id
    }

    /// Create a test contact
    pub async fn create_contact(&self, email: &str, trust_score: f64) -> Uuid {
        let contact_id = Uuid::new_v4();
        sqlx::query(
            "INSERT INTO contacts (id, user_id, email, trust_score, created_at) VALUES ($1, $2, $3, $4, NOW())",
        )
        .bind(contact_id)
        .bind(self.test_user_id)
        .bind(email)
        .bind(trust_score)
        .execute(&self.pool)
        .await
        .expect("Failed to create contact");
        contact_id
    }

    /// Clean up test data
    pub async fn cleanup(&self) {
        sqlx::query("DELETE FROM users WHERE id = $1")
            .bind(self.test_user_id)
            .execute(&self.pool)
            .await
            .ok();
    }
}

// ============================================================================
// Test Utilities
// ============================================================================

/// Test utilities
pub mod utils {
    use super::*;

    /// Create a test request with JSON body
    pub fn json_request(method: &str, uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    /// Create a test request with auth header
    pub fn authed_request(method: &str, uri: &str, token: &str) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header("Authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap()
    }

    /// Create a test request with auth and JSON body
    pub fn authed_json_request(
        method: &str,
        uri: &str,
        token: &str,
        body: serde_json::Value,
    ) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    /// Assert approximate floating point equality
    pub fn assert_approx_eq(a: f64, b: f64, epsilon: f64) {
        let diff = (a - b).abs();
        assert!(
            diff < epsilon,
            "Values not approximately equal: {} vs {} (diff: {}, epsilon: {})",
            a, b, diff, epsilon
        );
    }

    /// Generate random email address
    pub fn random_email() -> String {
        format!("test-{}@example.com", Uuid::new_v4())
    }
}

// ============================================================================
// Test Macros
// ============================================================================

/// Setup and teardown for integration tests
#[macro_export]
macro_rules! integration_test {
    ($name:ident, $body:expr) => {
        #[tokio::test]
        async fn $name() {
            let ctx = TestContext::new().await;
            let result = std::panic::AssertUnwindSafe(async { $body(&ctx).await })
                .catch_unwind()
                .await;
            ctx.cleanup().await;
            if let Err(e) = result {
                std::panic::resume_unwind(e);
            }
        }
    };
}
