// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Authentication middleware for Axum

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use std::sync::Arc;

use super::{Claims, JwtService, UserRole};

/// Auth state for middleware
#[derive(Clone)]
pub struct AuthState {
    pub jwt: Arc<JwtService>,
}

impl AuthState {
    pub fn new(jwt: JwtService) -> Self {
        Self { jwt: Arc::new(jwt) }
    }
}

/// Authenticated user extracted from token
#[derive(Debug, Clone)]
pub struct AuthUser {
    pub claims: Claims,
}

impl AuthUser {
    pub fn user_id(&self) -> uuid::Uuid {
        self.claims.sub
    }

    pub fn email(&self) -> &str {
        &self.claims.email
    }

    pub fn role(&self) -> UserRole {
        self.claims.role
    }

    pub fn tenant_id(&self) -> Option<uuid::Uuid> {
        self.claims.tenant_id
    }

    pub fn has_role(&self, required: UserRole) -> bool {
        self.claims.role.has_permission(&required)
    }
}

/// Extract bearer token from Authorization header
fn extract_bearer_token(req: &Request) -> Option<&str> {
    req.headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
}

/// Authentication middleware - requires valid JWT
pub async fn auth_middleware(
    State(auth_state): State<AuthState>,
    mut req: Request,
    next: Next,
) -> Response {
    let token = match extract_bearer_token(&req) {
        Some(t) => t,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({
                    "error": "Missing authorization header",
                    "code": "AUTH_MISSING"
                })),
            )
                .into_response();
        }
    };

    match auth_state.jwt.validate_access_token(token) {
        Ok(claims) => {
            // Insert authenticated user into request extensions
            req.extensions_mut().insert(AuthUser { claims });
            next.run(req).await
        }
        Err(e) => {
            let (status, code, message) = match e {
                super::JwtError::Expired => (
                    StatusCode::UNAUTHORIZED,
                    "TOKEN_EXPIRED",
                    "Token has expired",
                ),
                super::JwtError::WrongType => (
                    StatusCode::UNAUTHORIZED,
                    "WRONG_TOKEN_TYPE",
                    "Invalid token type",
                ),
                _ => (
                    StatusCode::UNAUTHORIZED,
                    "INVALID_TOKEN",
                    "Invalid or malformed token",
                ),
            };

            (status, Json(json!({ "error": message, "code": code }))).into_response()
        }
    }
}

/// Optional authentication middleware - allows unauthenticated requests
pub async fn optional_auth_middleware(
    State(auth_state): State<AuthState>,
    mut req: Request,
    next: Next,
) -> Response {
    if let Some(token) = extract_bearer_token(&req) {
        if let Ok(claims) = auth_state.jwt.validate_access_token(token) {
            req.extensions_mut().insert(AuthUser { claims });
        }
    }
    next.run(req).await
}

/// Role-based access control middleware factory
pub fn require_role(required: UserRole) -> impl Fn(AuthUser) -> Result<(), (StatusCode, Json<serde_json::Value>)> + Clone {
    move |user: AuthUser| {
        if user.has_role(required) {
            Ok(())
        } else {
            Err((
                StatusCode::FORBIDDEN,
                Json(json!({
                    "error": "Insufficient permissions",
                    "code": "FORBIDDEN",
                    "required_role": format!("{:?}", required)
                })),
            ))
        }
    }
}

/// Extract auth user from request extensions
pub fn get_auth_user(req: &Request<Body>) -> Option<&AuthUser> {
    req.extensions().get::<AuthUser>()
}

/// Axum extractor for authenticated user
#[derive(Debug, Clone)]
pub struct ExtractAuthUser(pub AuthUser);

#[axum::async_trait]
impl<S> axum::extract::FromRequestParts<S> for ExtractAuthUser
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<serde_json::Value>);

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        _state: &S,
    ) -> Result<Self, Self::Rejection> {
        parts
            .extensions
            .get::<AuthUser>()
            .cloned()
            .map(ExtractAuthUser)
            .ok_or_else(|| {
                (
                    StatusCode::UNAUTHORIZED,
                    Json(json!({
                        "error": "Authentication required",
                        "code": "AUTH_REQUIRED"
                    })),
                )
            })
    }
}

/// Axum extractor for JWT claims directly
/// Requires auth middleware to run first to populate the AuthUser extension
#[axum::async_trait]
impl<S> axum::extract::FromRequestParts<S> for Claims
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<serde_json::Value>);

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        _state: &S,
    ) -> Result<Self, Self::Rejection> {
        parts
            .extensions
            .get::<AuthUser>()
            .map(|auth_user| auth_user.claims.clone())
            .ok_or_else(|| {
                (
                    StatusCode::UNAUTHORIZED,
                    Json(json!({
                        "error": "Authentication required",
                        "code": "AUTH_REQUIRED"
                    })),
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_permissions() {
        let admin_claims = Claims {
            sub: uuid::Uuid::new_v4(),
            email: "admin@test.com".to_string(),
            role: UserRole::Admin,
            tenant_id: None,
            token_type: super::super::TokenType::Access,
            iat: 0,
            exp: i64::MAX,
            iss: "test".to_string(),
        };

        let admin = AuthUser { claims: admin_claims };
        assert!(admin.has_role(UserRole::Admin));
        assert!(admin.has_role(UserRole::Manager));
        assert!(admin.has_role(UserRole::Viewer));

        let viewer_claims = Claims {
            sub: uuid::Uuid::new_v4(),
            email: "viewer@test.com".to_string(),
            role: UserRole::Viewer,
            tenant_id: None,
            token_type: super::super::TokenType::Access,
            iat: 0,
            exp: i64::MAX,
            iss: "test".to_string(),
        };

        let viewer = AuthUser { claims: viewer_claims };
        assert!(viewer.has_role(UserRole::Viewer));
        assert!(!viewer.has_role(UserRole::Admin));
        assert!(!viewer.has_role(UserRole::Manager));
    }
}
