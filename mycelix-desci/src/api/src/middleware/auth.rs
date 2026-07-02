// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! JWT bearer-token authentication middleware
//!
//! Gates mutating endpoints (POST/PUT/DELETE) behind a valid JWT. Read-only
//! GET endpoints stay public. Mirrors the JWT pattern already used in
//! `mycelix-pulse/happ/backend-rs/src/middleware/auth.rs` (HS256, `sub`/`exp`
//! claims), simplified to a request-gate rather than a full identity extractor
//! since DeSci does not yet have a user/session model.

use axum::{
    Json,
    extract::Request,
    http::{StatusCode, header::AUTHORIZATION},
    middleware::Next,
    response::{IntoResponse, Response},
};
use jsonwebtoken::{DecodingKey, Validation, decode};
use serde::{Deserialize, Serialize};

use crate::error::ApiError;

/// Minimal JWT claims required to authorize a mutating request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (caller identity, e.g. DID or API client ID).
    pub sub: String,
    /// Expiration timestamp (seconds since epoch).
    pub exp: i64,
}

/// Reads the JWT signing secret from the `JWT_SECRET` environment variable.
///
/// No hardcoded fallback is provided: an unset secret is a misconfiguration
/// that must fail loudly rather than silently accepting any token signed
/// with a well-known default.
fn jwt_secret() -> Result<String, AuthError> {
    std::env::var("JWT_SECRET").map_err(|_| AuthError {
        message: "Server misconfiguration: JWT_SECRET is not set".to_string(),
    })
}

/// Authentication error, rendered as a 401 with the standard `ApiError` body.
pub struct AuthError {
    message: String,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let body = Json(ApiError::new("UNAUTHORIZED", self.message));
        (StatusCode::UNAUTHORIZED, body).into_response()
    }
}

/// `axum::middleware::from_fn` gate: requires `Authorization: Bearer <jwt>`
/// with a valid, unexpired signature. Apply only to mutating routes via
/// `.route_layer(...)` — read-only GET routes must not be wrapped with this.
pub async fn require_auth(req: Request, next: Next) -> Result<Response, AuthError> {
    let auth_header = req
        .headers()
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| AuthError {
            message: "Missing Authorization header".to_string(),
        })?;

    let token = auth_header
        .strip_prefix("Bearer ")
        .ok_or_else(|| AuthError {
            message: "Authorization header must use the Bearer scheme".to_string(),
        })?;

    let secret = jwt_secret()?;

    decode::<Claims>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &Validation::default(),
    )
    .map_err(|e| AuthError {
        message: format!("Invalid or expired token: {}", e),
    })?;

    Ok(next.run(req).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::Body,
        http::{Request as HttpRequest, StatusCode as HttpStatusCode},
        middleware::from_fn,
        routing::post,
    };
    use jsonwebtoken::{EncodingKey, Header, encode};
    use tower::ServiceExt;

    const TEST_SECRET: &str = "test-secret-for-unit-tests-only";

    fn token_with_exp(secret: &str, exp: i64) -> String {
        let claims = Claims {
            sub: "test-subject".to_string(),
            exp,
        };
        encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .expect("token encode")
    }

    async fn protected_handler() -> &'static str {
        "ok"
    }

    fn app() -> Router {
        Router::new()
            .route("/protected", post(protected_handler))
            .route_layer(from_fn(require_auth))
    }

    #[tokio::test]
    async fn rejects_missing_authorization_header() {
        // SAFETY: tests run single-threaded per-process env mutation is fine here.
        unsafe { std::env::set_var("JWT_SECRET", TEST_SECRET) };
        let response = app()
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/protected")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), HttpStatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn rejects_invalid_token() {
        unsafe { std::env::set_var("JWT_SECRET", TEST_SECRET) };
        let response = app()
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/protected")
                    .header(AUTHORIZATION, "Bearer not-a-real-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), HttpStatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn rejects_expired_token() {
        unsafe { std::env::set_var("JWT_SECRET", TEST_SECRET) };
        let expired = token_with_exp(TEST_SECRET, chrono::Utc::now().timestamp() - 3600);
        let response = app()
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/protected")
                    .header(AUTHORIZATION, format!("Bearer {}", expired))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), HttpStatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn accepts_valid_token() {
        unsafe { std::env::set_var("JWT_SECRET", TEST_SECRET) };
        let valid = token_with_exp(TEST_SECRET, chrono::Utc::now().timestamp() + 3600);
        let response = app()
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/protected")
                    .header(AUTHORIZATION, format!("Bearer {}", valid))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), HttpStatusCode::OK);
    }
}
