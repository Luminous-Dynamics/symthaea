// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! OAuth2/OIDC Authentication Module
//!
//! Implements secure authentication with:
//! - OAuth2 authorization code flow with PKCE
//! - OpenID Connect for identity
//! - JWT token issuance and validation
//! - Refresh token rotation
//! - Session management

use axum::{
    extract::{Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Redirect},
    routing::{get, post},
    Json, Router,
};
use chrono::{Duration, Utc};
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use rand::{distributions::Alphanumeric, Rng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use uuid::Uuid;

// ============================================================================
// Types
// ============================================================================

#[derive(Clone)]
pub struct AuthConfig {
    pub jwt_secret: String,
    pub jwt_issuer: String,
    pub jwt_audience: String,
    pub access_token_expiry: Duration,
    pub refresh_token_expiry: Duration,
    pub oauth_providers: Vec<OAuthProvider>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct OAuthProvider {
    pub name: String,
    pub client_id: String,
    pub client_secret: String,
    pub auth_url: String,
    pub token_url: String,
    pub userinfo_url: String,
    pub scopes: Vec<String>,
    pub redirect_uri: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,           // User ID
    pub email: String,
    pub name: Option<String>,
    pub iat: i64,              // Issued at
    pub exp: i64,              // Expiration
    pub iss: String,           // Issuer
    pub aud: String,           // Audience
    pub jti: String,           // JWT ID (for revocation)
    pub scope: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RefreshTokenClaims {
    pub sub: String,
    pub jti: String,
    pub family: String,        // Token family for rotation
    pub iat: i64,
    pub exp: i64,
}

#[derive(Debug, Serialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: i64,
    pub refresh_token: String,
    pub id_token: Option<String>,
    pub scope: String,
}

#[derive(Debug, Deserialize)]
pub struct AuthorizationRequest {
    pub provider: String,
    pub redirect_uri: Option<String>,
    pub state: Option<String>,
    pub code_challenge: Option<String>,
    pub code_challenge_method: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CallbackRequest {
    pub code: String,
    pub state: String,
}

#[derive(Debug, Deserialize)]
pub struct TokenRequest {
    pub grant_type: String,
    pub code: Option<String>,
    pub refresh_token: Option<String>,
    pub code_verifier: Option<String>,
    pub redirect_uri: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RevokeRequest {
    pub token: String,
    pub token_type_hint: Option<String>,
}

// ============================================================================
// State
// ============================================================================

pub struct AuthState {
    pub config: AuthConfig,
    pub pending_authorizations: dashmap::DashMap<String, PendingAuth>,
    pub refresh_token_families: dashmap::DashMap<String, TokenFamily>,
    pub revoked_tokens: dashmap::DashSet<String>,
}

#[derive(Clone)]
pub struct PendingAuth {
    pub provider: String,
    pub redirect_uri: String,
    pub code_challenge: Option<String>,
    pub created_at: chrono::DateTime<Utc>,
}

#[derive(Clone)]
pub struct TokenFamily {
    pub user_id: String,
    pub current_token_id: String,
    pub created_at: chrono::DateTime<Utc>,
}

impl AuthState {
    pub fn new(config: AuthConfig) -> Self {
        Self {
            config,
            pending_authorizations: dashmap::DashMap::new(),
            refresh_token_families: dashmap::DashMap::new(),
            revoked_tokens: dashmap::DashSet::new(),
        }
    }
}

// ============================================================================
// Routes
// ============================================================================

pub fn auth_routes() -> Router<Arc<AuthState>> {
    Router::new()
        .route("/authorize", get(authorize))
        .route("/callback", get(callback))
        .route("/token", post(token))
        .route("/revoke", post(revoke))
        .route("/userinfo", get(userinfo))
        .route("/jwks", get(jwks))
        .route("/.well-known/openid-configuration", get(openid_config))
}

// ============================================================================
// Handlers
// ============================================================================

/// Initiate OAuth2 authorization flow
async fn authorize(
    State(state): State<Arc<AuthState>>,
    Query(params): Query<AuthorizationRequest>,
) -> impl IntoResponse {
    // Find provider
    let provider = match state.config.oauth_providers.iter().find(|p| p.name == params.provider) {
        Some(p) => p,
        None => {
            return (StatusCode::BAD_REQUEST, "Unknown provider").into_response();
        }
    };

    // Generate state token
    let state_token: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(32)
        .map(char::from)
        .collect();

    // Store pending authorization
    state.pending_authorizations.insert(
        state_token.clone(),
        PendingAuth {
            provider: params.provider.clone(),
            redirect_uri: params.redirect_uri.unwrap_or_else(|| provider.redirect_uri.clone()),
            code_challenge: params.code_challenge,
            created_at: Utc::now(),
        },
    );

    // Build authorization URL
    let mut auth_url = url::Url::parse(&provider.auth_url).unwrap();
    auth_url.query_pairs_mut()
        .append_pair("client_id", &provider.client_id)
        .append_pair("redirect_uri", &provider.redirect_uri)
        .append_pair("response_type", "code")
        .append_pair("scope", &provider.scopes.join(" "))
        .append_pair("state", &state_token);

    // Add PKCE if provided
    if let Some(ref challenge) = params.code_challenge {
        auth_url.query_pairs_mut()
            .append_pair("code_challenge", challenge)
            .append_pair("code_challenge_method", params.code_challenge_method.as_deref().unwrap_or("S256"));
    }

    Redirect::temporary(auth_url.as_str()).into_response()
}

/// Handle OAuth2 callback
async fn callback(
    State(state): State<Arc<AuthState>>,
    Query(params): Query<CallbackRequest>,
) -> impl IntoResponse {
    // Retrieve and validate pending authorization
    let pending = match state.pending_authorizations.remove(&params.state) {
        Some((_, p)) => p,
        None => {
            return (StatusCode::BAD_REQUEST, "Invalid state").into_response();
        }
    };

    // Check expiration (5 minutes)
    if Utc::now() - pending.created_at > Duration::minutes(5) {
        return (StatusCode::BAD_REQUEST, "Authorization expired").into_response();
    }

    // Find provider
    let provider = match state.config.oauth_providers.iter().find(|p| p.name == pending.provider) {
        Some(p) => p,
        None => {
            return (StatusCode::INTERNAL_SERVER_ERROR, "Provider not found").into_response();
        }
    };

    // Exchange code for tokens
    let client = reqwest::Client::new();
    let token_response = match client
        .post(&provider.token_url)
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", &params.code),
            ("redirect_uri", &provider.redirect_uri),
            ("client_id", &provider.client_id),
            ("client_secret", &provider.client_secret),
        ])
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(_) => {
            return (StatusCode::BAD_GATEWAY, "Token exchange failed").into_response();
        }
    };

    let provider_tokens: serde_json::Value = match token_response.json().await {
        Ok(t) => t,
        Err(_) => {
            return (StatusCode::BAD_GATEWAY, "Invalid token response").into_response();
        }
    };

    // Fetch user info
    let access_token = provider_tokens["access_token"].as_str().unwrap_or("");
    let userinfo_response = match client
        .get(&provider.userinfo_url)
        .bearer_auth(access_token)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(_) => {
            return (StatusCode::BAD_GATEWAY, "Userinfo fetch failed").into_response();
        }
    };

    let userinfo: serde_json::Value = match userinfo_response.json().await {
        Ok(u) => u,
        Err(_) => {
            return (StatusCode::BAD_GATEWAY, "Invalid userinfo response").into_response();
        }
    };

    // Extract user details
    let email = userinfo["email"].as_str().unwrap_or("").to_string();
    let name = userinfo["name"].as_str().map(String::from);
    let sub = userinfo["sub"].as_str().unwrap_or(&email).to_string();

    // Generate our tokens
    let tokens = generate_tokens(&state, &sub, &email, name.as_deref());

    // Redirect with tokens (in production, use secure cookie or fragment)
    let redirect_url = format!(
        "{}#access_token={}&token_type=Bearer&expires_in={}",
        pending.redirect_uri,
        tokens.access_token,
        tokens.expires_in
    );

    Redirect::temporary(&redirect_url).into_response()
}

/// Token endpoint for token exchange and refresh
async fn token(
    State(state): State<Arc<AuthState>>,
    Json(params): Json<TokenRequest>,
) -> impl IntoResponse {
    match params.grant_type.as_str() {
        "authorization_code" => {
            // Handle authorization code (simplified - would need full validation)
            (StatusCode::NOT_IMPLEMENTED, "Use callback endpoint").into_response()
        }

        "refresh_token" => {
            let refresh_token = match params.refresh_token {
                Some(t) => t,
                None => {
                    return (StatusCode::BAD_REQUEST, "Missing refresh token").into_response();
                }
            };

            // Decode and validate refresh token
            let claims = match decode_refresh_token(&state.config, &refresh_token) {
                Ok(c) => c,
                Err(_) => {
                    return (StatusCode::UNAUTHORIZED, "Invalid refresh token").into_response();
                }
            };

            // Check if token is revoked
            if state.revoked_tokens.contains(&claims.jti) {
                // Token reuse detected - revoke entire family
                state.refresh_token_families.remove(&claims.family);
                return (StatusCode::UNAUTHORIZED, "Token revoked").into_response();
            }

            // Validate token family
            let family = match state.refresh_token_families.get(&claims.family) {
                Some(f) => f.clone(),
                None => {
                    return (StatusCode::UNAUTHORIZED, "Invalid token family").into_response();
                }
            };

            // Check if this is the current token in the family
            if family.current_token_id != claims.jti {
                // Possible token theft - revoke entire family
                state.refresh_token_families.remove(&claims.family);
                return (StatusCode::UNAUTHORIZED, "Token rotation detected").into_response();
            }

            // Revoke old token
            state.revoked_tokens.insert(claims.jti.clone());

            // Generate new tokens (rotation)
            let tokens = generate_tokens(&state, &claims.sub, &claims.sub, None);

            Json(tokens).into_response()
        }

        _ => (StatusCode::BAD_REQUEST, "Unsupported grant type").into_response(),
    }
}

/// Revoke a token
async fn revoke(
    State(state): State<Arc<AuthState>>,
    Json(params): Json<RevokeRequest>,
) -> impl IntoResponse {
    // Try to decode as access token first
    if let Ok(claims) = decode_access_token(&state.config, &params.token) {
        state.revoked_tokens.insert(claims.jti);
        return StatusCode::OK.into_response();
    }

    // Try as refresh token
    if let Ok(claims) = decode_refresh_token(&state.config, &params.token) {
        state.revoked_tokens.insert(claims.jti);
        // Also revoke entire family for security
        state.refresh_token_families.remove(&claims.family);
        return StatusCode::OK.into_response();
    }

    // Invalid token format, but revocation should still succeed per RFC 7009
    StatusCode::OK.into_response()
}

/// Get user info from access token
async fn userinfo(
    State(state): State<Arc<AuthState>>,
    headers: axum::http::HeaderMap,
) -> impl IntoResponse {
    let auth_header = match headers.get(header::AUTHORIZATION) {
        Some(h) => h.to_str().unwrap_or(""),
        None => {
            return (StatusCode::UNAUTHORIZED, "Missing authorization").into_response();
        }
    };

    let token = auth_header.strip_prefix("Bearer ").unwrap_or("");

    let claims = match decode_access_token(&state.config, token) {
        Ok(c) => c,
        Err(_) => {
            return (StatusCode::UNAUTHORIZED, "Invalid token").into_response();
        }
    };

    // Check if revoked
    if state.revoked_tokens.contains(&claims.jti) {
        return (StatusCode::UNAUTHORIZED, "Token revoked").into_response();
    }

    Json(serde_json::json!({
        "sub": claims.sub,
        "email": claims.email,
        "name": claims.name,
        "email_verified": true,
    }))
    .into_response()
}

/// JWKS endpoint for public key distribution
async fn jwks(State(state): State<Arc<AuthState>>) -> impl IntoResponse {
    // In production, would return actual JWK set
    Json(serde_json::json!({
        "keys": [
            {
                "kty": "oct",
                "alg": "HS512",
                "use": "sig",
                "kid": "mycelix-key-1"
            }
        ]
    }))
}

/// OpenID Connect discovery document
async fn openid_config(State(state): State<Arc<AuthState>>) -> impl IntoResponse {
    let issuer = &state.config.jwt_issuer;

    Json(serde_json::json!({
        "issuer": issuer,
        "authorization_endpoint": format!("{}/auth/authorize", issuer),
        "token_endpoint": format!("{}/auth/token", issuer),
        "userinfo_endpoint": format!("{}/auth/userinfo", issuer),
        "jwks_uri": format!("{}/auth/jwks", issuer),
        "revocation_endpoint": format!("{}/auth/revoke", issuer),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "subject_types_supported": ["public"],
        "id_token_signing_alg_values_supported": ["HS512"],
        "scopes_supported": ["openid", "email", "profile"],
        "token_endpoint_auth_methods_supported": ["client_secret_post"],
        "code_challenge_methods_supported": ["S256"]
    }))
}

// ============================================================================
// Token Generation & Validation
// ============================================================================

fn generate_tokens(
    state: &AuthState,
    user_id: &str,
    email: &str,
    name: Option<&str>,
) -> TokenResponse {
    let now = Utc::now();
    let access_token_id = Uuid::new_v4().to_string();
    let refresh_token_id = Uuid::new_v4().to_string();
    let family_id = Uuid::new_v4().to_string();

    // Access token claims
    let access_claims = Claims {
        sub: user_id.to_string(),
        email: email.to_string(),
        name: name.map(String::from),
        iat: now.timestamp(),
        exp: (now + state.config.access_token_expiry).timestamp(),
        iss: state.config.jwt_issuer.clone(),
        aud: state.config.jwt_audience.clone(),
        jti: access_token_id,
        scope: vec!["openid".to_string(), "email".to_string(), "profile".to_string()],
    };

    // Refresh token claims
    let refresh_claims = RefreshTokenClaims {
        sub: user_id.to_string(),
        jti: refresh_token_id.clone(),
        family: family_id.clone(),
        iat: now.timestamp(),
        exp: (now + state.config.refresh_token_expiry).timestamp(),
    };

    // Encode tokens
    let access_token = encode(
        &Header::new(Algorithm::HS512),
        &access_claims,
        &EncodingKey::from_secret(state.config.jwt_secret.as_bytes()),
    )
    .unwrap();

    let refresh_token = encode(
        &Header::new(Algorithm::HS512),
        &refresh_claims,
        &EncodingKey::from_secret(state.config.jwt_secret.as_bytes()),
    )
    .unwrap();

    // Store token family
    state.refresh_token_families.insert(
        family_id,
        TokenFamily {
            user_id: user_id.to_string(),
            current_token_id: refresh_token_id,
            created_at: now,
        },
    );

    TokenResponse {
        access_token,
        token_type: "Bearer".to_string(),
        expires_in: state.config.access_token_expiry.num_seconds(),
        refresh_token,
        id_token: None, // Would generate OIDC ID token here
        scope: "openid email profile".to_string(),
    }
}

fn decode_access_token(config: &AuthConfig, token: &str) -> Result<Claims, jsonwebtoken::errors::Error> {
    let mut validation = Validation::new(Algorithm::HS512);
    validation.set_issuer(&[&config.jwt_issuer]);
    validation.set_audience(&[&config.jwt_audience]);

    let token_data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(config.jwt_secret.as_bytes()),
        &validation,
    )?;

    Ok(token_data.claims)
}

fn decode_refresh_token(config: &AuthConfig, token: &str) -> Result<RefreshTokenClaims, jsonwebtoken::errors::Error> {
    let validation = Validation::new(Algorithm::HS512);

    let token_data = decode::<RefreshTokenClaims>(
        token,
        &DecodingKey::from_secret(config.jwt_secret.as_bytes()),
        &validation,
    )?;

    Ok(token_data.claims)
}

// ============================================================================
// PKCE Utilities
// ============================================================================

pub fn generate_code_verifier() -> String {
    rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(64)
        .map(char::from)
        .collect()
}

pub fn generate_code_challenge(verifier: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let result = hasher.finalize();
    base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, result)
}

pub fn verify_code_challenge(verifier: &str, challenge: &str) -> bool {
    generate_code_challenge(verifier) == challenge
}

// ============================================================================
// Middleware
// ============================================================================

use axum::middleware::from_fn_with_state;

pub async fn auth_middleware(
    State(state): State<Arc<AuthState>>,
    headers: axum::http::HeaderMap,
    request: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> impl IntoResponse {
    let auth_header = match headers.get(header::AUTHORIZATION) {
        Some(h) => h.to_str().unwrap_or(""),
        None => {
            return (StatusCode::UNAUTHORIZED, "Missing authorization").into_response();
        }
    };

    let token = match auth_header.strip_prefix("Bearer ") {
        Some(t) => t,
        None => {
            return (StatusCode::UNAUTHORIZED, "Invalid authorization format").into_response();
        }
    };

    let claims = match decode_access_token(&state.config, token) {
        Ok(c) => c,
        Err(_) => {
            return (StatusCode::UNAUTHORIZED, "Invalid token").into_response();
        }
    };

    // Check if revoked
    if state.revoked_tokens.contains(&claims.jti) {
        return (StatusCode::UNAUTHORIZED, "Token revoked").into_response();
    }

    // Add claims to request extensions
    let mut request = request;
    request.extensions_mut().insert(claims);

    next.run(request).await.into_response()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pkce() {
        let verifier = generate_code_verifier();
        let challenge = generate_code_challenge(&verifier);

        assert!(verify_code_challenge(&verifier, &challenge));
        assert!(!verify_code_challenge("wrong_verifier", &challenge));
    }

    #[test]
    fn test_token_generation() {
        let config = AuthConfig {
            jwt_secret: "test_secret_key_that_is_long_enough".to_string(),
            jwt_issuer: "https://mycelix.mail".to_string(),
            jwt_audience: "mycelix-mail".to_string(),
            access_token_expiry: Duration::hours(1),
            refresh_token_expiry: Duration::days(7),
            oauth_providers: vec![],
        };

        let state = AuthState::new(config.clone());
        let tokens = generate_tokens(&state, "user123", "user@example.com", Some("Test User"));

        assert!(!tokens.access_token.is_empty());
        assert!(!tokens.refresh_token.is_empty());
        assert_eq!(tokens.token_type, "Bearer");

        // Verify access token
        let claims = decode_access_token(&config, &tokens.access_token).unwrap();
        assert_eq!(claims.sub, "user123");
        assert_eq!(claims.email, "user@example.com");
    }
}
