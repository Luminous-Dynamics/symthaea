// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Local AI API routes
//!
//! Endpoints for privacy-preserving AI features powered by Symthaea.

use axum::{
    extract::Extension,
    routing::{get, post},
    Json, Router,
};

use crate::error::AppResult;
use crate::middleware::Claims as JwtClaims;
use crate::routes::AppState;
use crate::services::ai::{
    AIInsights, ConsciousnessState, EmailIntent, LocalAIService,
    ReplySuggestion, ThreadSummary, TrustExplanation,
};
use crate::types::Email;

/// Create the AI router
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/analyze", post(analyze_email))
        .route("/summarize", post(summarize_thread))
        .route("/intent", post(detect_intent))
        .route("/suggest-reply", post(suggest_replies))
        .route("/explain-trust", post(explain_trust))
        .route("/status", get(ai_status))
}

/// Request to analyze an email
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct AnalyzeEmailRequest {
    pub email: Email,
}

/// Analyze an email with local AI
#[utoipa::path(
    post,
    path = "/ai/analyze",
    tag = "ai",
    request_body = AnalyzeEmailRequest,
    responses(
        (status = 200, description = "AI analysis", body = AIInsights),
        (status = 500, description = "Analysis failed", body = crate::types::ApiError),
    ),
    security(("bearer_auth" = []))
)]
pub async fn analyze_email(
    Extension(_claims): Extension<JwtClaims>,
    Json(request): Json<AnalyzeEmailRequest>,
) -> AppResult<Json<AIInsights>> {
    let ai_service = LocalAIService::new();
    let insights = ai_service.analyze_email(&request.email).await?;
    Ok(Json(insights))
}

/// Request to summarize a thread
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct SummarizeThreadRequest {
    pub emails: Vec<Email>,
}

/// Summarize an email thread
#[utoipa::path(
    post,
    path = "/ai/summarize",
    tag = "ai",
    request_body = SummarizeThreadRequest,
    responses(
        (status = 200, description = "Thread summary", body = ThreadSummary),
        (status = 400, description = "Empty thread", body = crate::types::ApiError),
    ),
    security(("bearer_auth" = []))
)]
pub async fn summarize_thread(
    Extension(_claims): Extension<JwtClaims>,
    Json(request): Json<SummarizeThreadRequest>,
) -> AppResult<Json<ThreadSummary>> {
    let ai_service = LocalAIService::new();
    let summary = ai_service.summarize_thread(&request.emails).await?;
    Ok(Json(summary))
}

/// Request to detect intent
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct DetectIntentRequest {
    pub email: Email,
}

/// Response with detected intent
#[derive(Debug, serde::Serialize, utoipa::ToSchema)]
pub struct DetectIntentResponse {
    pub intent: EmailIntent,
    pub suggested_action: String,
    pub priority_weight: f32,
}

/// Detect the intent of an email
#[utoipa::path(
    post,
    path = "/ai/intent",
    tag = "ai",
    request_body = DetectIntentRequest,
    responses(
        (status = 200, description = "Detected intent", body = DetectIntentResponse),
    ),
    security(("bearer_auth" = []))
)]
pub async fn detect_intent(
    Extension(_claims): Extension<JwtClaims>,
    Json(request): Json<DetectIntentRequest>,
) -> AppResult<Json<DetectIntentResponse>> {
    let ai_service = LocalAIService::new();
    let intent = ai_service.detect_intent(&request.email).await?;

    Ok(Json(DetectIntentResponse {
        suggested_action: intent.suggested_action().to_string(),
        priority_weight: intent.priority_weight(),
        intent,
    }))
}

/// Request for reply suggestions
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct SuggestRepliesRequest {
    pub email: Email,
}

/// Response with reply suggestions
#[derive(Debug, serde::Serialize, utoipa::ToSchema)]
pub struct SuggestRepliesResponse {
    pub suggestions: Vec<ReplySuggestion>,
}

/// Generate reply suggestions for an email
#[utoipa::path(
    post,
    path = "/ai/suggest-reply",
    tag = "ai",
    request_body = SuggestRepliesRequest,
    responses(
        (status = 200, description = "Reply suggestions", body = SuggestRepliesResponse),
    ),
    security(("bearer_auth" = []))
)]
pub async fn suggest_replies(
    Extension(_claims): Extension<JwtClaims>,
    Json(request): Json<SuggestRepliesRequest>,
) -> AppResult<Json<SuggestRepliesResponse>> {
    let ai_service = LocalAIService::new();
    let suggestions = ai_service.suggest_replies(&request.email).await?;
    Ok(Json(SuggestRepliesResponse { suggestions }))
}

/// Request to explain trust
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct ExplainTrustRequest {
    pub sender_did: String,
    pub trust_score: f64,
    pub trust_path: Vec<(String, String, f64)>,
}

/// Explain trust reasoning using causal analysis
#[utoipa::path(
    post,
    path = "/ai/explain-trust",
    tag = "ai",
    request_body = ExplainTrustRequest,
    responses(
        (status = 200, description = "Trust explanation", body = TrustExplanation),
    ),
    security(("bearer_auth" = []))
)]
pub async fn explain_trust(
    Extension(_claims): Extension<JwtClaims>,
    Json(request): Json<ExplainTrustRequest>,
) -> AppResult<Json<TrustExplanation>> {
    let ai_service = LocalAIService::new();
    let explanation = ai_service
        .explain_trust(&request.sender_did, request.trust_score, &request.trust_path)
        .await?;
    Ok(Json(explanation))
}

/// Get AI system status
#[utoipa::path(
    get,
    path = "/ai/status",
    tag = "ai",
    responses(
        (status = 200, description = "AI status", body = ConsciousnessState),
    ),
    security(("bearer_auth" = []))
)]
pub async fn ai_status(
    Extension(_claims): Extension<JwtClaims>,
) -> AppResult<Json<ConsciousnessState>> {
    let ai_service = LocalAIService::new();
    let state = ai_service.consciousness_state().await;
    Ok(Json(state))
}
