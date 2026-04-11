// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! JWT authentication middleware

use axum::{
    extract::FromRequestParts,
    http::{header::AUTHORIZATION, request::Parts, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};

use crate::types::ApiError;

/// JWT claims stored in token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (DID)
    pub sub: String,
    /// Agent public key (base64)
    pub agent_pub_key: String,
    /// Expiration timestamp
    pub exp: i64,
    /// Issued at timestamp
    pub iat: i64,
}

impl Claims {
    /// Create new claims for a user
    pub fn new(did: String, agent_pub_key: String, expiration_hours: u64) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            sub: did,
            agent_pub_key,
            exp: now + (expiration_hours as i64 * 3600),
            iat: now,
        }
    }
}

/// Authenticated user extracted from JWT
#[derive(Debug, Clone)]
pub struct AuthenticatedUser {
    pub did: String,
    pub agent_pub_key: String,
}

/// JWT authentication error
pub struct AuthError {
    message: String,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let body = Json(ApiError::new("AUTH_ERROR", self.message));
        (StatusCode::UNAUTHORIZED, body).into_response()
    }
}

/// Extract authenticated user from request
#[axum::async_trait]
impl<S> FromRequestParts<S> for AuthenticatedUser
where
    S: Send + Sync,
{
    type Rejection = AuthError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // Get Authorization header
        let auth_header = parts
            .headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| AuthError {
                message: "Missing Authorization header".to_string(),
            })?;

        // Extract Bearer token
        let token = auth_header
            .strip_prefix("Bearer ")
            .ok_or_else(|| AuthError {
                message: "Invalid Authorization header format".to_string(),
            })?;

        // Get JWT secret from extensions (set by middleware)
        let jwt_secret = parts
            .extensions
            .get::<JwtSecret>()
            .ok_or_else(|| AuthError {
                message: "JWT configuration error".to_string(),
            })?;

        // Decode and validate token
        let token_data = decode::<Claims>(
            token,
            &DecodingKey::from_secret(jwt_secret.0.as_bytes()),
            &Validation::default(),
        )
        .map_err(|e| AuthError {
            message: format!("Invalid token: {}", e),
        })?;

        Ok(AuthenticatedUser {
            did: token_data.claims.sub,
            agent_pub_key: token_data.claims.agent_pub_key,
        })
    }
}

/// JWT secret key wrapper for extension injection
#[derive(Clone)]
pub struct JwtSecret(pub String);

/// Create a JWT token
pub fn create_token(claims: &Claims, secret: &str) -> Result<String, jsonwebtoken::errors::Error> {
    encode(
        &Header::default(),
        claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
}

/// Verify a JWT token
pub fn verify_token(token: &str, secret: &str) -> Result<Claims, jsonwebtoken::errors::Error> {
    let token_data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &Validation::default(),
    )?;
    Ok(token_data.claims)
}
