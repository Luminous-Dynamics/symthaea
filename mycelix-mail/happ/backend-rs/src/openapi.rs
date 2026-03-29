// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! OpenAPI documentation for Mycelix-Mail API
//!
//! Provides Swagger UI at `/api-docs` and OpenAPI JSON at `/api-docs/openapi.json`

use utoipa::{
    openapi::security::{HttpAuthScheme, HttpBuilder, SecurityScheme},
    Modify, OpenApi,
};

/// Security scheme modifier for JWT bearer authentication
struct SecurityAddon;

impl Modify for SecurityAddon {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        if let Some(components) = openapi.components.as_mut() {
            components.add_security_scheme(
                "bearer_auth",
                SecurityScheme::Http(
                    HttpBuilder::new()
                        .scheme(HttpAuthScheme::Bearer)
                        .bearer_format("JWT")
                        .description(Some("JWT token obtained from /api/auth/login"))
                        .build(),
                ),
            );
        }
    }
}

/// Mycelix-Mail API Documentation
#[derive(OpenApi)]
#[openapi(
    info(
        title = "Mycelix-Mail API",
        version = "0.1.0",
        description = r#"
# Mycelix-Mail Backend API

Decentralized email system built on Holochain with MATL-based trust scoring.

## Features

- **End-to-end encryption**: Messages encrypted with X25519 + ChaCha20-Poly1305
- **DHT-based DID resolution**: No centralized identity provider
- **MATL trust scoring**: Adaptive trust layer for spam filtering
- **IPFS storage**: Message bodies stored on content-addressed network
- **Real-time updates**: WebSocket events for new mail and trust updates

## Authentication

All authenticated endpoints require a JWT token in the `Authorization` header:

```
Authorization: Bearer <token>
```

Obtain a token via `/api/auth/login` or `/api/auth/register`.

## Rate Limiting

API requests are rate-limited to prevent abuse:
- Default: 60 requests per minute per IP
- Burst: Up to 10 concurrent requests

## Epistemic Tiers

Messages are classified by their epistemic strength:
- **Tier 0 (Null)**: No epistemic claim
- **Tier 1 (Testimonial)**: Based on personal testimony
- **Tier 2 (Privately Verifiable)**: Can be verified by recipient
- **Tier 3 (Cryptographically Proven)**: Verified via cryptographic proof
- **Tier 4 (Publicly Reproducible)**: Independently verifiable by anyone

## WebSocket Events

Connect to `/ws/events` for real-time updates:
- `NewMail`: New email received
- `TrustUpdated`: Trust score changed
- `Connected`: Connection established
"#,
        contact(
            name = "Luminous Dynamics",
            url = "https://github.com/Luminous-Dynamics/Mycelix-Mail"
        ),
        license(
            name = "MIT",
            url = "https://opensource.org/licenses/MIT"
        )
    ),
    servers(
        (url = "/api", description = "API base path")
    ),
    tags(
        (name = "auth", description = "Authentication endpoints"),
        (name = "emails", description = "Email operations"),
        (name = "trust", description = "Trust score and MATL operations"),
        (name = "claims", description = "Verifiable claims and credentials (0TML integration)"),
        (name = "ai", description = "Local AI insights (Symthaea integration)"),
        (name = "trust-graph", description = "Trust graph visualization and attestations"),
        (name = "health", description = "Health check endpoints")
    ),
    paths(
        // Auth
        crate::routes::auth::register,
        crate::routes::auth::login,
        crate::routes::auth::get_current_user,
        crate::routes::auth::refresh_token,
        // Emails
        crate::routes::emails::send_email,
        crate::routes::emails::get_inbox,
        crate::routes::emails::get_outbox,
        crate::routes::emails::get_email,
        crate::routes::emails::delete_email,
        crate::routes::emails::get_thread,
        crate::routes::emails::report_spam,
        crate::routes::emails::mark_not_spam,
        // Trust
        crate::routes::trust::get_trust_score,
        crate::routes::trust::get_trust_scores_batch,
        crate::routes::trust::check_byzantine,
        crate::routes::trust::get_cross_happ_reputation,
        crate::routes::trust::get_cache_stats,
        crate::routes::trust::invalidate_cache,
        // Claims (0TML integration)
        crate::routes::claims::verify_claims,
        crate::routes::claims::list_credentials,
        crate::routes::claims::get_credential,
        crate::routes::claims::attach_claim,
        crate::routes::claims::get_assurance_level,
        // AI (Symthaea integration)
        crate::routes::ai::analyze_email,
        crate::routes::ai::summarize_thread,
        crate::routes::ai::detect_intent,
        crate::routes::ai::suggest_replies,
        crate::routes::ai::explain_trust,
        crate::routes::ai::ai_status,
        // Trust Graph
        crate::routes::trust_graph::get_trust_graph,
        crate::routes::trust_graph::find_trust_path,
        crate::routes::trust_graph::create_attestation,
        crate::routes::trust_graph::revoke_attestation,
        crate::routes::trust_graph::get_trusted_by,
        crate::routes::trust_graph::get_trusters,
    ),
    components(
        schemas(
            // Core types
            crate::types::Email,
            crate::types::SendEmailInput,
            crate::types::EpistemicTier,
            crate::types::TrustScoreInfo,
            crate::types::Contact,
            crate::types::AddContactInput,
            // Auth types
            crate::types::LoginRequest,
            crate::types::LoginResponse,
            crate::types::RegisterRequest,
            crate::types::UserClaims,
            // DID types
            crate::types::RegisterDidInput,
            crate::types::DidResolution,
            crate::types::SpamReportInput,
            // Query/Response types
            crate::types::InboxFilter,
            crate::types::ApiError,
            crate::types::HealthResponse,
            crate::types::WsMessage,
            // Auth response types
            crate::routes::auth::CurrentUserResponse,
            // Email response types
            crate::routes::emails::SpamReportBody,
            crate::routes::emails::DeleteResponse,
            crate::routes::emails::SpamReportResponse,
            crate::routes::emails::PaginatedEmailResponse,
            // Trust response types
            crate::routes::trust::ByzantineCheckResult,
            crate::routes::trust::CrossHappReputationResponse,
            crate::routes::trust::HappScoreInfo,
            crate::routes::trust::CacheStatsResponse,
            // Claims types (0TML integration)
            crate::services::claims::AssuranceLevel,
            crate::services::claims::VerifiableClaim,
            crate::services::claims::VerifiableCredential,
            crate::services::claims::CredentialType,
            crate::services::claims::ProofType,
            crate::services::claims::ClaimVerification,
            crate::services::claims::EmailClaims,
            crate::routes::claims::VerifyClaimsRequest,
            crate::routes::claims::VerifyClaimsResponse,
            crate::routes::claims::CredentialsListResponse,
            crate::routes::claims::AttachClaimRequest,
            crate::routes::claims::AttachClaimResponse,
            crate::routes::claims::AssuranceLevelResponse,
            // AI types (Symthaea integration)
            crate::services::ai::AIInsights,
            crate::services::ai::EmailIntent,
            crate::services::ai::ThreadSummary,
            crate::services::ai::ReplySuggestion,
            crate::services::ai::TrustExplanation,
            crate::services::ai::ConsciousnessState,
            crate::routes::ai::AnalyzeEmailRequest,
            crate::routes::ai::SummarizeThreadRequest,
            crate::routes::ai::DetectIntentRequest,
            crate::routes::ai::DetectIntentResponse,
            crate::routes::ai::SuggestRepliesRequest,
            crate::routes::ai::SuggestRepliesResponse,
            crate::routes::ai::ExplainTrustRequest,
            // Trust Graph types
            crate::services::trust_graph::TrustGraph,
            crate::services::trust_graph::TrustNode,
            crate::services::trust_graph::TrustEdge,
            crate::services::trust_graph::TrustPath,
            crate::services::trust_graph::TrustHop,
            crate::services::trust_graph::TrustAttestation,
            crate::services::trust_graph::RelationType,
            crate::routes::trust_graph::GraphQuery,
            crate::routes::trust_graph::CreateAttestationRequest,
            crate::routes::trust_graph::CreateAttestationResponse,
            crate::routes::trust_graph::TrustEdgesResponse,
        )
    ),
    modifiers(&SecurityAddon)
)]
pub struct ApiDoc;
