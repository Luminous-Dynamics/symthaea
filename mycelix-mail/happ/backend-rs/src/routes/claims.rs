// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Claims API routes
//!
//! Endpoints for verifiable claims and credentials.

use axum::{
    extract::{Extension, Path},
    routing::{get, post},
    Json, Router,
};

use crate::error::{AppError, AppResult};
use crate::middleware::Claims as JwtClaims;
use crate::routes::AppState;
use crate::services::claims::{
    AssuranceLevel, ClaimVerification, ClaimsService,
    EmailClaims, ProofType, VerifiableClaim, VerifiableCredential,
};

/// Create the claims router
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/verify", post(verify_claims))
        .route("/credentials", get(list_credentials))
        .route("/credentials/:id", get(get_credential))
        .route("/attach", post(attach_claim))
        .route("/assurance/:did", get(get_assurance_level))
}

/// Request to verify claims
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct VerifyClaimsRequest {
    pub claims: Vec<VerifiableClaim>,
    pub assurance_level: Option<AssuranceLevel>,
}

/// Response with verification results
#[derive(Debug, serde::Serialize, utoipa::ToSchema)]
pub struct VerifyClaimsResponse {
    pub verifications: Vec<ClaimVerification>,
    pub overall_epistemic_tier: u8,
    pub all_verified: bool,
}

/// Verify claims attached to an email
#[utoipa::path(
    post,
    path = "/claims/verify",
    tag = "claims",
    request_body = VerifyClaimsRequest,
    responses(
        (status = 200, description = "Claims verified", body = VerifyClaimsResponse),
        (status = 400, description = "Invalid claims", body = crate::types::ApiError),
    ),
    security(("bearer_auth" = []))
)]
pub async fn verify_claims(
    Extension(_claims): Extension<JwtClaims>,
    Extension(_state): Extension<AppState>,
    Json(request): Json<VerifyClaimsRequest>,
) -> AppResult<Json<VerifyClaimsResponse>> {
    let claims_service = ClaimsService::new();

    let email_claims = EmailClaims {
        claims: request.claims,
        assurance_level: request.assurance_level.unwrap_or(AssuranceLevel::E0Anonymous),
        verifications: vec![],
    };

    let verifications = claims_service.verify_email_claims(&email_claims).await;

    let overall_tier = verifications
        .iter()
        .map(|v| v.epistemic_tier)
        .max()
        .unwrap_or(0);

    let all_verified = verifications.iter().all(|v| v.verified);

    Ok(Json(VerifyClaimsResponse {
        verifications,
        overall_epistemic_tier: overall_tier,
        all_verified,
    }))
}

/// Response with user's credentials
#[derive(Debug, serde::Serialize, utoipa::ToSchema)]
pub struct CredentialsListResponse {
    pub credentials: Vec<VerifiableCredential>,
    pub assurance_level: AssuranceLevel,
}

/// List user's verifiable credentials
#[utoipa::path(
    get,
    path = "/claims/credentials",
    tag = "claims",
    responses(
        (status = 200, description = "List of credentials", body = CredentialsListResponse),
    ),
    security(("bearer_auth" = []))
)]
pub async fn list_credentials(
    Extension(claims): Extension<JwtClaims>,
    Extension(_state): Extension<AppState>,
) -> AppResult<Json<CredentialsListResponse>> {
    // In production: fetch from 0TML identity store
    // For now, return placeholder
    Ok(Json(CredentialsListResponse {
        credentials: vec![],
        assurance_level: AssuranceLevel::E1VerifiedEmail,
    }))
}

/// Get a specific credential by ID
#[utoipa::path(
    get,
    path = "/claims/credentials/{id}",
    tag = "claims",
    params(
        ("id" = String, Path, description = "Credential ID")
    ),
    responses(
        (status = 200, description = "Credential details", body = VerifiableCredential),
        (status = 404, description = "Credential not found", body = crate::types::ApiError),
    ),
    security(("bearer_auth" = []))
)]
pub async fn get_credential(
    Extension(_claims): Extension<JwtClaims>,
    Path(id): Path<String>,
) -> AppResult<Json<VerifiableCredential>> {
    // In production: fetch from 0TML
    Err(AppError::NotFound(format!("Credential {} not found", id)))
}

/// Request to attach a claim to an email
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct AttachClaimRequest {
    pub claim: VerifiableClaim,
    pub proof_type: ProofType,
}

/// Response with attached claim and proof
#[derive(Debug, serde::Serialize, utoipa::ToSchema)]
pub struct AttachClaimResponse {
    pub claim: VerifiableClaim,
    pub proof: String,
    pub epistemic_tier: u8,
}

/// Attach a verifiable claim to an outgoing email
#[utoipa::path(
    post,
    path = "/claims/attach",
    tag = "claims",
    request_body = AttachClaimRequest,
    responses(
        (status = 200, description = "Claim attached with proof", body = AttachClaimResponse),
        (status = 400, description = "Failed to generate proof", body = crate::types::ApiError),
    ),
    security(("bearer_auth" = []))
)]
pub async fn attach_claim(
    Extension(_claims): Extension<JwtClaims>,
    Json(request): Json<AttachClaimRequest>,
) -> AppResult<Json<AttachClaimResponse>> {
    let claims_service = ClaimsService::new();

    let proof = claims_service
        .generate_proof(&request.claim, request.proof_type)
        .await
        .map_err(|e| AppError::ValidationError(e.to_string()))?;

    let epistemic_tier = request.claim.epistemic_tier();

    Ok(Json(AttachClaimResponse {
        claim: request.claim,
        proof,
        epistemic_tier,
    }))
}

/// Response with assurance level info
#[derive(Debug, serde::Serialize, utoipa::ToSchema)]
pub struct AssuranceLevelResponse {
    pub did: String,
    pub level: AssuranceLevel,
    pub attack_cost_usd: u64,
    pub trust_multiplier: f64,
}

/// Get assurance level for a DID
#[utoipa::path(
    get,
    path = "/claims/assurance/{did}",
    tag = "claims",
    params(
        ("did" = String, Path, description = "DID to check")
    ),
    responses(
        (status = 200, description = "Assurance level", body = AssuranceLevelResponse),
    ),
    security(("bearer_auth" = []))
)]
pub async fn get_assurance_level(
    Extension(_claims): Extension<JwtClaims>,
    Path(did): Path<String>,
) -> AppResult<Json<AssuranceLevelResponse>> {
    // In production: fetch from 0TML identity system
    // For now, return default
    let level = AssuranceLevel::E1VerifiedEmail;

    Ok(Json(AssuranceLevelResponse {
        did,
        level,
        attack_cost_usd: level.attack_cost_usd(),
        trust_multiplier: level.trust_multiplier(),
    }))
}
