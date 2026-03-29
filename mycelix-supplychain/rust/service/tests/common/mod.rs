// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Common test utilities

use axum::{
    middleware,
    routing::{get, post},
    Router,
};
use serde_json::{json, Value};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

/// Create a test app with in-memory SQLite database
pub async fn create_test_app() -> Router {
    // Generate test keypair
    let keypair = crypto::KeyPair::generate();

    // Create in-memory database
    let db = mycelix_erp_service::db::Database::new("sqlite::memory:")
        .await
        .expect("Failed to create test database");

    // Create app state
    let state = Arc::new(mycelix_erp_service::AppState {
        keypair,
        db: Some(db),
        claims: tokio::sync::RwLock::new(std::collections::HashMap::new()),
        pool: None, // No PostgreSQL pool for integration tests
    });

    // Configure CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build router (same as production)
    Router::new()
        .route("/health", get(mycelix_erp_service::health::health))
        .route("/metrics", get(mycelix_erp_service::api::metrics_endpoint))
        .route("/v1/events", post(mycelix_erp_service::api::ingest_event))
        .route(
            "/v1/events/batch",
            post(mycelix_erp_service::batch::ingest_batch),
        )
        .route("/v1/claims/:id", get(mycelix_erp_service::api::get_claim))
        .route(
            "/v1/batches/:batch_id/claims",
            get(mycelix_erp_service::lineage_api::get_batch_claims),
        )
        .route(
            "/v1/lineage/:batch_id",
            get(mycelix_erp_service::lineage_api::get_lineage),
        )
        .route(
            "/v1/claims",
            get(mycelix_erp_service::lineage_api::search_claims),
        )
        .route("/v1/verify", post(mycelix_erp_service::api::verify_vc))
        .layer(cors)
        .layer(middleware::from_fn(
            mycelix_erp_service::security::security_headers_middleware,
        ))
        .with_state(state)
}

/// Create a test Verifiable Credential
pub fn create_test_vc(
    batch_id: &str,
    product_id: &str,
    event_type: &str,
    prev_claim_ids: Option<Vec<String>>,
) -> Value {
    let mut credential_subject = json!({
        "eventType": event_type,
        "productId": product_id,
        "batchId": batch_id,
        "quantity": 1000.0,
        "unit": "kg",
        "facility": {
            "id": "FAC-TEST-001",
            "name": "Test Facility"
        },
        "timestamp": "2025-11-16T10:00:00Z"
    });

    // Add prevBatchIds for TRANSFORMED events
    if event_type == "TRANSFORMED" {
        if let Some(ref prev_ids) = prev_claim_ids {
            credential_subject["prevClaimIds"] = json!(prev_ids);
        }
    }

    // Add shipment for SHIPPED events
    if event_type == "SHIPPED" {
        credential_subject["shipment"] = json!({
            "shipmentId": "SHIP-001",
            "carrier": "Test Carrier",
            "trackingNumber": "TRACK-123",
            "origin": "Origin Location",
            "destination": "Destination Location"
        });
    }

    json!({
        "@context": [
            "https://www.w3.org/2018/credentials/v1",
            "https://mycelix.org/contexts/supply-chain/v1"
        ],
        "type": ["VerifiableCredential", "SupplyChainEvent"],
        "issuer": "did:mycelix:org:test-issuer",
        "issuanceDate": "2025-11-16T10:00:00Z",
        "credentialSubject": credential_subject
    })
}
