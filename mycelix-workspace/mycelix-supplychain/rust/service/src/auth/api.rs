// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Authentication API handlers
//!
//! Uses runtime SQL queries for flexibility without compile-time database dependencies.

use axum::{
    extract::State,
    http::StatusCode,
    routing::{delete, get, post},
    Json, Router,
};
use serde_json::json;
use sqlx::{PgPool, Row};
use std::sync::Arc;
use uuid::Uuid;

use super::{
    hash_password, verify_password, validate_password_strength,
    AuthState, ExtractAuthUser, JwtService,
    ChangePasswordRequest, CreateApiKeyRequest, LoginRequest, LoginResponse,
    PasswordResetRequest, RefreshRequest, RegisterRequest, RegisterResponse,
    Tenant, TenantPlan, TenantSettings, UpdateProfileRequest, User, UserResponse, UserRole,
};

/// Auth module state
#[derive(Clone)]
pub struct AuthApiState {
    pub jwt: Arc<JwtService>,
    pub db: PgPool,
}

impl AuthApiState {
    pub fn new(jwt: JwtService, db: PgPool) -> Self {
        Self {
            jwt: Arc::new(jwt),
            db,
        }
    }
}

/// Error response helper
fn error_response(status: StatusCode, code: &str, message: &str) -> (StatusCode, Json<serde_json::Value>) {
    (status, Json(json!({ "error": message, "code": code })))
}

/// Create auth router
pub fn router(state: AuthApiState) -> Router {
    let auth_state = AuthState::new((*state.jwt).clone());

    Router::new()
        // Public routes
        .route("/v1/auth/login", post(login))
        .route("/v1/auth/register", post(register))
        .route("/v1/auth/refresh", post(refresh))
        .route("/v1/auth/password-reset", post(request_password_reset))
        .route("/v1/auth/password-reset/confirm", post(confirm_password_reset))
        // Protected routes
        .route("/v1/auth/me", get(get_profile).patch(update_profile))
        .route("/v1/auth/logout", post(logout))
        .route("/v1/auth/change-password", post(change_password))
        .route("/v1/auth/api-keys", get(list_api_keys).post(create_api_key))
        .route("/v1/auth/api-keys/:key_id", delete(revoke_api_key))
        .with_state((state, auth_state))
}

/// Helper to parse UserRole from string
fn parse_role(s: &str) -> UserRole {
    match s {
        "ADMIN" => UserRole::Admin,
        "MANAGER" => UserRole::Manager,
        "ACCOUNTANT" => UserRole::Accountant,
        "WAREHOUSE" => UserRole::Warehouse,
        "SALES" => UserRole::Sales,
        _ => UserRole::Viewer,
    }
}

/// Helper to convert UserRole to string
fn role_to_string(role: UserRole) -> &'static str {
    match role {
        UserRole::Admin => "ADMIN",
        UserRole::Manager => "MANAGER",
        UserRole::Accountant => "ACCOUNTANT",
        UserRole::Warehouse => "WAREHOUSE",
        UserRole::Sales => "SALES",
        UserRole::Viewer => "VIEWER",
    }
}

/// Login handler
async fn login(
    State((state, _)): State<(AuthApiState, AuthState)>,
    Json(req): Json<LoginRequest>,
) -> Result<Json<LoginResponse>, (StatusCode, Json<serde_json::Value>)> {
    // Find user by email
    let row = sqlx::query(
        "SELECT id, email, name, password_hash, role::text, tenant_id, is_active, mfa_enabled, created_at, updated_at, last_login FROM users WHERE email = $1"
    )
    .bind(&req.email)
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Database error during login: {}", e);
        error_response(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", "Database error")
    })?;

    let user = row.map(|r| User {
        id: r.get("id"),
        email: r.get("email"),
        name: r.get("name"),
        password_hash: r.get("password_hash"),
        role: parse_role(r.get("role")),
        tenant_id: r.get("tenant_id"),
        is_active: r.get("is_active"),
        mfa_enabled: r.get("mfa_enabled"),
        created_at: r.get("created_at"),
        updated_at: r.get("updated_at"),
        last_login: r.get("last_login"),
    }).ok_or_else(|| {
        error_response(StatusCode::UNAUTHORIZED, "INVALID_CREDENTIALS", "Invalid email or password")
    })?;

    if !user.is_active {
        return Err(error_response(StatusCode::FORBIDDEN, "ACCOUNT_DISABLED", "Account is disabled"));
    }

    // Verify password
    let password_valid = verify_password(&req.password, &user.password_hash)
        .map_err(|_| error_response(StatusCode::INTERNAL_SERVER_ERROR, "VERIFY_ERROR", "Password verification failed"))?;

    if !password_valid {
        return Err(error_response(StatusCode::UNAUTHORIZED, "INVALID_CREDENTIALS", "Invalid email or password"));
    }

    // Update last login
    let _ = sqlx::query("UPDATE users SET last_login = NOW() WHERE id = $1")
        .bind(user.id)
        .execute(&state.db)
        .await;

    // Load tenant if user has one
    let tenant: Option<Tenant> = if let Some(tid) = user.tenant_id {
        sqlx::query("SELECT id, name, slug, plan::text, is_active, settings, created_at, updated_at FROM tenants WHERE id = $1")
            .bind(tid)
            .fetch_optional(&state.db)
            .await
            .ok()
            .flatten()
            .map(|r| Tenant {
                id: r.get("id"),
                name: r.get("name"),
                slug: r.get("slug"),
                plan: match r.get::<String, _>("plan").as_str() {
                    "PROFESSIONAL" => TenantPlan::Professional,
                    "ENTERPRISE" => TenantPlan::Enterprise,
                    _ => TenantPlan::Starter,
                },
                is_active: r.get("is_active"),
                settings: serde_json::from_value(r.get("settings")).unwrap_or_default(),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
            })
    } else {
        None
    };

    // Create tokens
    let (access_token, refresh_token, expires_in) = state
        .jwt
        .create_token_pair(user.id, &user.email, user.role, user.tenant_id)
        .map_err(|e| {
            tracing::error!("Token creation error: {}", e);
            error_response(StatusCode::INTERNAL_SERVER_ERROR, "TOKEN_ERROR", "Failed to create tokens")
        })?;

    Ok(Json(LoginResponse {
        access_token,
        refresh_token,
        token_type: "Bearer",
        expires_in,
        user: user.into(),
        tenant,
    }))
}

/// Register handler
async fn register(
    State((state, _)): State<(AuthApiState, AuthState)>,
    Json(req): Json<RegisterRequest>,
) -> Result<(StatusCode, Json<RegisterResponse>), (StatusCode, Json<serde_json::Value>)> {
    // Validate password
    validate_password_strength(&req.password).map_err(|errors| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "Password does not meet requirements",
                "code": "WEAK_PASSWORD",
                "details": errors
            })),
        )
    })?;

    // Check if email already exists
    let existing = sqlx::query("SELECT id FROM users WHERE email = $1")
        .bind(&req.email)
        .fetch_optional(&state.db)
        .await
        .map_err(|_| error_response(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", "Database error"))?;

    if existing.is_some() {
        return Err(error_response(StatusCode::CONFLICT, "EMAIL_EXISTS", "Email already registered"));
    }

    // Hash password
    let password_hash = hash_password(&req.password)
        .map_err(|_| error_response(StatusCode::INTERNAL_SERVER_ERROR, "HASH_ERROR", "Failed to hash password"))?;

    let user_id = Uuid::new_v4();
    let mut tenant: Option<Tenant> = None;
    let mut tenant_id: Option<Uuid> = None;

    // Create tenant if company name provided
    if let Some(company_name) = &req.company_name {
        let tid = Uuid::new_v4();
        let slug = company_name
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == ' ')
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join("-");

        let settings = TenantSettings::default();

        sqlx::query(
            "INSERT INTO tenants (id, name, slug, plan, is_active, settings, created_at, updated_at) VALUES ($1, $2, $3, 'STARTER', true, $4, NOW(), NOW())"
        )
        .bind(tid)
        .bind(company_name)
        .bind(&slug)
        .bind(serde_json::to_value(&settings).unwrap())
        .execute(&state.db)
        .await
        .map_err(|e| {
            tracing::error!("Failed to create tenant: {}", e);
            error_response(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", "Failed to create organization")
        })?;

        tenant_id = Some(tid);
        tenant = Some(Tenant {
            id: tid,
            name: company_name.clone(),
            slug,
            plan: TenantPlan::Starter,
            is_active: true,
            settings,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        });
    }

    // Create user
    let role = if tenant_id.is_some() {
        UserRole::Admin // First user of org is admin
    } else {
        UserRole::Viewer
    };

    sqlx::query(
        "INSERT INTO users (id, email, name, password_hash, role, tenant_id, is_active, mfa_enabled, created_at, updated_at) VALUES ($1, $2, $3, $4, $5, $6, true, false, NOW(), NOW())"
    )
    .bind(user_id)
    .bind(&req.email)
    .bind(&req.name)
    .bind(&password_hash)
    .bind(role_to_string(role))
    .bind(tenant_id)
    .execute(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Failed to create user: {}", e);
        error_response(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", "Failed to create user")
    })?;

    Ok((
        StatusCode::CREATED,
        Json(RegisterResponse {
            user: UserResponse {
                id: user_id,
                email: req.email,
                name: req.name,
                role,
                tenant_id,
                is_active: true,
                mfa_enabled: false,
                created_at: chrono::Utc::now(),
                last_login: None,
            },
            tenant,
            message: "Registration successful".to_string(),
        }),
    ))
}

/// Refresh token handler
async fn refresh(
    State((state, _)): State<(AuthApiState, AuthState)>,
    Json(req): Json<RefreshRequest>,
) -> Result<Json<LoginResponse>, (StatusCode, Json<serde_json::Value>)> {
    // Validate refresh token
    let claims = state.jwt.validate_refresh_token(&req.refresh_token).map_err(|e| {
        match e {
            super::JwtError::Expired => error_response(StatusCode::UNAUTHORIZED, "TOKEN_EXPIRED", "Refresh token expired"),
            _ => error_response(StatusCode::UNAUTHORIZED, "INVALID_TOKEN", "Invalid refresh token"),
        }
    })?;

    // Get fresh user data
    let row = sqlx::query(
        "SELECT id, email, name, password_hash, role::text, tenant_id, is_active, mfa_enabled, created_at, updated_at, last_login FROM users WHERE id = $1"
    )
    .bind(claims.sub)
    .fetch_optional(&state.db)
    .await
    .map_err(|_| error_response(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", "Database error"))?
    .ok_or_else(|| error_response(StatusCode::UNAUTHORIZED, "USER_NOT_FOUND", "User not found"))?;

    let user = User {
        id: row.get("id"),
        email: row.get("email"),
        name: row.get("name"),
        password_hash: row.get("password_hash"),
        role: parse_role(row.get("role")),
        tenant_id: row.get("tenant_id"),
        is_active: row.get("is_active"),
        mfa_enabled: row.get("mfa_enabled"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
        last_login: row.get("last_login"),
    };

    if !user.is_active {
        return Err(error_response(StatusCode::FORBIDDEN, "ACCOUNT_DISABLED", "Account is disabled"));
    }

    // Create new tokens
    let (access_token, refresh_token, expires_in) = state
        .jwt
        .create_token_pair(user.id, &user.email, user.role, user.tenant_id)
        .map_err(|_| error_response(StatusCode::INTERNAL_SERVER_ERROR, "TOKEN_ERROR", "Failed to create tokens"))?;

    Ok(Json(LoginResponse {
        access_token,
        refresh_token,
        token_type: "Bearer",
        expires_in,
        user: user.into(),
        tenant: None,
    }))
}

/// Get current user profile
async fn get_profile(
    ExtractAuthUser(auth): ExtractAuthUser,
    State((state, _)): State<(AuthApiState, AuthState)>,
) -> Result<Json<UserResponse>, (StatusCode, Json<serde_json::Value>)> {
    let row = sqlx::query(
        "SELECT id, email, name, password_hash, role::text, tenant_id, is_active, mfa_enabled, created_at, updated_at, last_login FROM users WHERE id = $1"
    )
    .bind(auth.user_id())
    .fetch_one(&state.db)
    .await
    .map_err(|_| error_response(StatusCode::NOT_FOUND, "USER_NOT_FOUND", "User not found"))?;

    let user = User {
        id: row.get("id"),
        email: row.get("email"),
        name: row.get("name"),
        password_hash: row.get("password_hash"),
        role: parse_role(row.get("role")),
        tenant_id: row.get("tenant_id"),
        is_active: row.get("is_active"),
        mfa_enabled: row.get("mfa_enabled"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
        last_login: row.get("last_login"),
    };

    Ok(Json(user.into()))
}

/// Update profile
async fn update_profile(
    ExtractAuthUser(auth): ExtractAuthUser,
    State((state, _)): State<(AuthApiState, AuthState)>,
    Json(req): Json<UpdateProfileRequest>,
) -> Result<Json<UserResponse>, (StatusCode, Json<serde_json::Value>)> {
    if let Some(name) = &req.name {
        sqlx::query("UPDATE users SET name = $1, updated_at = NOW() WHERE id = $2")
            .bind(name)
            .bind(auth.user_id())
            .execute(&state.db)
            .await
            .map_err(|_| error_response(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", "Failed to update profile"))?;
    }

    // Return updated profile
    get_profile(ExtractAuthUser(auth), State((state, AuthState::new(JwtService::from_env())))).await
}

/// Logout (placeholder - client should discard tokens)
async fn logout(
    ExtractAuthUser(_auth): ExtractAuthUser,
) -> (StatusCode, Json<serde_json::Value>) {
    (StatusCode::OK, Json(json!({ "message": "Logged out successfully" })))
}

/// Change password
async fn change_password(
    ExtractAuthUser(auth): ExtractAuthUser,
    State((state, _)): State<(AuthApiState, AuthState)>,
    Json(req): Json<ChangePasswordRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, Json<serde_json::Value>)> {
    // Validate new password
    validate_password_strength(&req.new_password).map_err(|errors| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "Password does not meet requirements",
                "code": "WEAK_PASSWORD",
                "details": errors
            })),
        )
    })?;

    // Get current user
    let row = sqlx::query("SELECT password_hash FROM users WHERE id = $1")
        .bind(auth.user_id())
        .fetch_one(&state.db)
        .await
        .map_err(|_| error_response(StatusCode::NOT_FOUND, "USER_NOT_FOUND", "User not found"))?;

    let password_hash: String = row.get("password_hash");

    // Verify current password
    let valid = verify_password(&req.current_password, &password_hash)
        .map_err(|_| error_response(StatusCode::INTERNAL_SERVER_ERROR, "VERIFY_ERROR", "Verification failed"))?;

    if !valid {
        return Err(error_response(StatusCode::UNAUTHORIZED, "INVALID_PASSWORD", "Current password is incorrect"));
    }

    // Hash new password
    let new_hash = hash_password(&req.new_password)
        .map_err(|_| error_response(StatusCode::INTERNAL_SERVER_ERROR, "HASH_ERROR", "Failed to hash password"))?;

    // Update password
    sqlx::query("UPDATE users SET password_hash = $1, updated_at = NOW() WHERE id = $2")
        .bind(&new_hash)
        .bind(auth.user_id())
        .execute(&state.db)
        .await
        .map_err(|_| error_response(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", "Failed to update password"))?;

    Ok((StatusCode::OK, Json(json!({ "message": "Password changed successfully" }))))
}

/// Request password reset
async fn request_password_reset(
    State((_state, _)): State<(AuthApiState, AuthState)>,
    Json(_req): Json<PasswordResetRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    (StatusCode::OK, Json(json!({ "message": "If the email exists, a reset link will be sent" })))
}

/// Confirm password reset
async fn confirm_password_reset(
    State((_state, _)): State<(AuthApiState, AuthState)>,
    Json(_req): Json<super::PasswordResetConfirm>,
) -> (StatusCode, Json<serde_json::Value>) {
    (StatusCode::NOT_IMPLEMENTED, Json(json!({ "error": "Not implemented" })))
}

/// List API keys
async fn list_api_keys(
    ExtractAuthUser(auth): ExtractAuthUser,
    State((state, _)): State<(AuthApiState, AuthState)>,
) -> Result<Json<Vec<super::ApiKeyResponse>>, (StatusCode, Json<serde_json::Value>)> {
    let rows = sqlx::query(
        "SELECT id, name, key_prefix, scopes, expires_at, created_at, last_used_at FROM api_keys WHERE user_id = $1 AND is_active = true"
    )
    .bind(auth.user_id())
    .fetch_all(&state.db)
    .await
    .map_err(|_| error_response(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", "Failed to fetch API keys"))?;

    let responses: Vec<super::ApiKeyResponse> = rows
        .into_iter()
        .map(|r| super::ApiKeyResponse {
            id: r.get("id"),
            name: r.get("name"),
            key_prefix: r.get("key_prefix"),
            key: None,
            scopes: serde_json::from_value(r.get("scopes")).unwrap_or_default(),
            expires_at: r.get("expires_at"),
            created_at: r.get("created_at"),
            last_used_at: r.get("last_used_at"),
        })
        .collect();

    Ok(Json(responses))
}

/// Create API key
async fn create_api_key(
    ExtractAuthUser(auth): ExtractAuthUser,
    State((state, _)): State<(AuthApiState, AuthState)>,
    Json(req): Json<CreateApiKeyRequest>,
) -> Result<(StatusCode, Json<super::ApiKeyResponse>), (StatusCode, Json<serde_json::Value>)> {
    let raw_key = format!("mk_{}", uuid::Uuid::new_v4().to_string().replace('-', ""));
    let key_prefix = raw_key.chars().take(10).collect::<String>();
    let key_hash = hash_password(&raw_key)
        .map_err(|_| error_response(StatusCode::INTERNAL_SERVER_ERROR, "HASH_ERROR", "Failed to hash key"))?;

    let key_id = Uuid::new_v4();
    let expires_at = req.expires_in_days.map(|days| {
        chrono::Utc::now() + chrono::Duration::days(days)
    });

    sqlx::query(
        "INSERT INTO api_keys (id, user_id, name, key_hash, key_prefix, scopes, expires_at, is_active, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, true, NOW())"
    )
    .bind(key_id)
    .bind(auth.user_id())
    .bind(&req.name)
    .bind(&key_hash)
    .bind(&key_prefix)
    .bind(serde_json::to_value(&req.scopes).unwrap())
    .bind(expires_at)
    .execute(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Failed to create API key: {}", e);
        error_response(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", "Failed to create API key")
    })?;

    Ok((
        StatusCode::CREATED,
        Json(super::ApiKeyResponse {
            id: key_id,
            name: req.name,
            key_prefix,
            key: Some(raw_key),
            scopes: req.scopes,
            expires_at,
            created_at: chrono::Utc::now(),
            last_used_at: None,
        }),
    ))
}

/// Revoke API key
async fn revoke_api_key(
    ExtractAuthUser(auth): ExtractAuthUser,
    State((state, _)): State<(AuthApiState, AuthState)>,
    axum::extract::Path(key_id): axum::extract::Path<Uuid>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    let result = sqlx::query("UPDATE api_keys SET is_active = false WHERE id = $1 AND user_id = $2")
        .bind(key_id)
        .bind(auth.user_id())
        .execute(&state.db)
        .await
        .map_err(|_| error_response(StatusCode::INTERNAL_SERVER_ERROR, "DB_ERROR", "Failed to revoke key"))?;

    if result.rows_affected() == 0 {
        return Err(error_response(StatusCode::NOT_FOUND, "KEY_NOT_FOUND", "API key not found"));
    }

    Ok(StatusCode::NO_CONTENT)
}
