// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! JWT token handling

use chrono::{Duration, Utc};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, TokenData, Validation};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use super::UserRole;

/// JWT configuration
#[derive(Debug, Clone)]
pub struct JwtConfig {
    /// Secret key for signing tokens
    pub secret: String,
    /// Access token expiry in seconds
    pub access_token_expiry: i64,
    /// Refresh token expiry in seconds
    pub refresh_token_expiry: i64,
    /// Token issuer
    pub issuer: String,
}

impl JwtConfig {
    /// Create config from environment variables
    pub fn from_env() -> Self {
        Self {
            secret: std::env::var("JWT_SECRET")
                .unwrap_or_else(|_| "mycelix-erp-dev-secret-change-in-production".to_string()),
            access_token_expiry: std::env::var("JWT_ACCESS_EXPIRY")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3600), // 1 hour
            refresh_token_expiry: std::env::var("JWT_REFRESH_EXPIRY")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(604800), // 7 days
            issuer: std::env::var("JWT_ISSUER")
                .unwrap_or_else(|_| "mycelix-erp".to_string()),
        }
    }
}

/// Token type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenType {
    Access,
    Refresh,
}

/// JWT claims
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user ID)
    pub sub: Uuid,
    /// User email
    pub email: String,
    /// User role
    pub role: UserRole,
    /// Tenant ID (optional)
    pub tenant_id: Option<Uuid>,
    /// Token type
    pub token_type: TokenType,
    /// Issued at
    pub iat: i64,
    /// Expiry
    pub exp: i64,
    /// Issuer
    pub iss: String,
}

#[derive(Debug, Error)]
pub enum JwtError {
    #[error("Token creation failed: {0}")]
    Creation(#[from] jsonwebtoken::errors::Error),
    #[error("Token expired")]
    Expired,
    #[error("Invalid token")]
    Invalid,
    #[error("Wrong token type")]
    WrongType,
}

/// JWT service for token operations
pub struct JwtService {
    config: JwtConfig,
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
}

impl JwtService {
    pub fn new(config: JwtConfig) -> Self {
        let encoding_key = EncodingKey::from_secret(config.secret.as_bytes());
        let decoding_key = DecodingKey::from_secret(config.secret.as_bytes());
        Self {
            config,
            encoding_key,
            decoding_key,
        }
    }

    pub fn from_env() -> Self {
        Self::new(JwtConfig::from_env())
    }

    /// Create access token
    pub fn create_access_token(
        &self,
        user_id: Uuid,
        email: &str,
        role: UserRole,
        tenant_id: Option<Uuid>,
    ) -> Result<String, JwtError> {
        let now = Utc::now();
        let exp = now + Duration::seconds(self.config.access_token_expiry);

        let claims = Claims {
            sub: user_id,
            email: email.to_string(),
            role,
            tenant_id,
            token_type: TokenType::Access,
            iat: now.timestamp(),
            exp: exp.timestamp(),
            iss: self.config.issuer.clone(),
        };

        encode(&Header::default(), &claims, &self.encoding_key).map_err(JwtError::Creation)
    }

    /// Create refresh token
    pub fn create_refresh_token(
        &self,
        user_id: Uuid,
        email: &str,
        role: UserRole,
        tenant_id: Option<Uuid>,
    ) -> Result<String, JwtError> {
        let now = Utc::now();
        let exp = now + Duration::seconds(self.config.refresh_token_expiry);

        let claims = Claims {
            sub: user_id,
            email: email.to_string(),
            role,
            tenant_id,
            token_type: TokenType::Refresh,
            iat: now.timestamp(),
            exp: exp.timestamp(),
            iss: self.config.issuer.clone(),
        };

        encode(&Header::default(), &claims, &self.encoding_key).map_err(JwtError::Creation)
    }

    /// Create both access and refresh tokens
    pub fn create_token_pair(
        &self,
        user_id: Uuid,
        email: &str,
        role: UserRole,
        tenant_id: Option<Uuid>,
    ) -> Result<(String, String, i64), JwtError> {
        let access = self.create_access_token(user_id, email, role, tenant_id)?;
        let refresh = self.create_refresh_token(user_id, email, role, tenant_id)?;
        Ok((access, refresh, self.config.access_token_expiry))
    }

    /// Validate and decode token
    pub fn validate_token(&self, token: &str) -> Result<TokenData<Claims>, JwtError> {
        let mut validation = Validation::default();
        validation.set_issuer(&[&self.config.issuer]);

        decode::<Claims>(token, &self.decoding_key, &validation).map_err(|e| {
            match e.kind() {
                jsonwebtoken::errors::ErrorKind::ExpiredSignature => JwtError::Expired,
                _ => JwtError::Invalid,
            }
        })
    }

    /// Validate access token specifically
    pub fn validate_access_token(&self, token: &str) -> Result<Claims, JwtError> {
        let token_data = self.validate_token(token)?;
        if token_data.claims.token_type != TokenType::Access {
            return Err(JwtError::WrongType);
        }
        Ok(token_data.claims)
    }

    /// Validate refresh token specifically
    pub fn validate_refresh_token(&self, token: &str) -> Result<Claims, JwtError> {
        let token_data = self.validate_token(token)?;
        if token_data.claims.token_type != TokenType::Refresh {
            return Err(JwtError::WrongType);
        }
        Ok(token_data.claims)
    }

    /// Get access token expiry in seconds
    pub fn access_token_expiry(&self) -> i64 {
        self.config.access_token_expiry
    }
}

impl Clone for JwtService {
    fn clone(&self) -> Self {
        Self::new(self.config.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_creation_and_validation() {
        let service = JwtService::new(JwtConfig {
            secret: "test-secret".to_string(),
            access_token_expiry: 3600,
            refresh_token_expiry: 86400,
            issuer: "test".to_string(),
        });

        let user_id = Uuid::new_v4();
        let (access, refresh, _) = service
            .create_token_pair(user_id, "test@example.com", UserRole::Admin, None)
            .unwrap();

        let access_claims = service.validate_access_token(&access).unwrap();
        assert_eq!(access_claims.sub, user_id);
        assert_eq!(access_claims.token_type, TokenType::Access);

        let refresh_claims = service.validate_refresh_token(&refresh).unwrap();
        assert_eq!(refresh_claims.sub, user_id);
        assert_eq!(refresh_claims.token_type, TokenType::Refresh);
    }

    #[test]
    fn test_wrong_token_type_rejected() {
        let service = JwtService::new(JwtConfig {
            secret: "test-secret".to_string(),
            access_token_expiry: 3600,
            refresh_token_expiry: 86400,
            issuer: "test".to_string(),
        });

        let user_id = Uuid::new_v4();
        let refresh = service
            .create_refresh_token(user_id, "test@example.com", UserRole::Admin, None)
            .unwrap();

        // Refresh token should not validate as access token
        assert!(matches!(
            service.validate_access_token(&refresh),
            Err(JwtError::WrongType)
        ));
    }
}
