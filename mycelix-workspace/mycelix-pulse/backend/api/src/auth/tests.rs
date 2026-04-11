// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Authentication module tests
//!
//! Comprehensive tests for OAuth2, JWT, and session management

use super::oauth2::*;
use axum::http::StatusCode;
use chrono::{Duration, Utc};

#[cfg(test)]
mod oauth2_tests {
    use super::*;

    fn test_config() -> OAuth2Config {
        OAuth2Config {
            client_id: "test-client".to_string(),
            client_secret: "test-secret".to_string(),
            redirect_uri: "http://localhost:3000/callback".to_string(),
            auth_url: "http://localhost:8080/auth".to_string(),
            token_url: "http://localhost:8080/token".to_string(),
            scopes: vec!["openid".to_string(), "email".to_string()],
        }
    }

    #[test]
    fn test_pkce_verifier_generation() {
        let verifier = PKCEVerifier::new();

        // Verifier should be 43-128 characters (base64url encoded)
        assert!(verifier.verifier.len() >= 43);
        assert!(verifier.verifier.len() <= 128);

        // Challenge should be base64url encoded SHA256 (43 chars)
        assert_eq!(verifier.challenge.len(), 43);

        // Challenge method should be S256
        assert_eq!(verifier.challenge_method, "S256");
    }

    #[test]
    fn test_pkce_verifier_uniqueness() {
        let v1 = PKCEVerifier::new();
        let v2 = PKCEVerifier::new();

        assert_ne!(v1.verifier, v2.verifier);
        assert_ne!(v1.challenge, v2.challenge);
    }

    #[tokio::test]
    async fn test_token_manager_creation() {
        let manager = TokenManager::new("test-secret-key-32-bytes-long!!".to_string());
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_token_manager_short_secret() {
        let manager = TokenManager::new("short".to_string());
        assert!(manager.is_err());
    }

    #[tokio::test]
    async fn test_jwt_token_generation_and_validation() {
        let manager = TokenManager::new("test-secret-key-32-bytes-long!!".to_string()).unwrap();

        let claims = Claims {
            sub: "user-123".to_string(),
            email: "test@example.com".to_string(),
            name: "Test User".to_string(),
            iat: Utc::now().timestamp() as usize,
            exp: (Utc::now() + Duration::hours(1)).timestamp() as usize,
            iss: "mycelix-mail".to_string(),
            aud: "mycelix-client".to_string(),
            token_type: TokenType::Access,
            permissions: vec!["read".to_string(), "write".to_string()],
        };

        let token = manager.generate_token(&claims).unwrap();
        assert!(!token.is_empty());

        let validated = manager.validate_token(&token).unwrap();
        assert_eq!(validated.sub, "user-123");
        assert_eq!(validated.email, "test@example.com");
    }

    #[tokio::test]
    async fn test_expired_token_rejection() {
        let manager = TokenManager::new("test-secret-key-32-bytes-long!!".to_string()).unwrap();

        let claims = Claims {
            sub: "user-123".to_string(),
            email: "test@example.com".to_string(),
            name: "Test User".to_string(),
            iat: (Utc::now() - Duration::hours(2)).timestamp() as usize,
            exp: (Utc::now() - Duration::hours(1)).timestamp() as usize, // Expired
            iss: "mycelix-mail".to_string(),
            aud: "mycelix-client".to_string(),
            token_type: TokenType::Access,
            permissions: vec![],
        };

        let token = manager.generate_token(&claims).unwrap();
        let result = manager.validate_token(&token);

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_refresh_token_generation() {
        let manager = TokenManager::new("test-secret-key-32-bytes-long!!".to_string()).unwrap();

        let (token, family_id) = manager.generate_refresh_token("user-123").await.unwrap();

        assert!(!token.is_empty());
        assert!(!family_id.is_empty());

        // Token should be stored
        let stored = manager.refresh_tokens.get(&token);
        assert!(stored.is_some());
    }

    #[tokio::test]
    async fn test_refresh_token_rotation() {
        let manager = TokenManager::new("test-secret-key-32-bytes-long!!".to_string()).unwrap();

        let (token1, family_id) = manager.generate_refresh_token("user-123").await.unwrap();

        // Rotate the token
        let (token2, new_family_id) = manager.rotate_refresh_token(&token1).await.unwrap();

        // Family ID should remain the same
        assert_eq!(family_id, new_family_id);

        // Old token should be invalidated
        assert!(manager.refresh_tokens.get(&token1).is_none());

        // New token should be valid
        assert!(manager.refresh_tokens.get(&token2).is_some());
    }

    #[tokio::test]
    async fn test_refresh_token_reuse_detection() {
        let manager = TokenManager::new("test-secret-key-32-bytes-long!!".to_string()).unwrap();

        let (token1, _) = manager.generate_refresh_token("user-123").await.unwrap();

        // First rotation succeeds
        let (token2, _) = manager.rotate_refresh_token(&token1).await.unwrap();

        // Attempting to reuse old token should fail and revoke family
        let result = manager.rotate_refresh_token(&token1).await;
        assert!(result.is_err());

        // New token should also be revoked (family compromise)
        assert!(manager.refresh_tokens.get(&token2).is_none());
    }

    #[tokio::test]
    async fn test_token_revocation() {
        let manager = TokenManager::new("test-secret-key-32-bytes-long!!".to_string()).unwrap();

        let (token, _) = manager.generate_refresh_token("user-123").await.unwrap();

        manager.revoke_token(&token).await;

        assert!(manager.refresh_tokens.get(&token).is_none());
    }
}

#[cfg(test)]
mod session_tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Clone)]
    struct MockSessionStore {
        sessions: std::sync::Arc<tokio::sync::RwLock<HashMap<String, String>>>,
    }

    impl MockSessionStore {
        fn new() -> Self {
            Self {
                sessions: std::sync::Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            }
        }

        async fn set(&self, key: &str, value: &str) {
            self.sessions.write().await.insert(key.to_string(), value.to_string());
        }

        async fn get(&self, key: &str) -> Option<String> {
            self.sessions.read().await.get(key).cloned()
        }

        async fn delete(&self, key: &str) {
            self.sessions.write().await.remove(key);
        }
    }

    #[tokio::test]
    async fn test_session_creation() {
        let store = MockSessionStore::new();

        store.set("session:abc123", r#"{"user_id": "user-1"}"#).await;

        let session = store.get("session:abc123").await;
        assert!(session.is_some());
        assert!(session.unwrap().contains("user-1"));
    }

    #[tokio::test]
    async fn test_session_deletion() {
        let store = MockSessionStore::new();

        store.set("session:abc123", "data").await;
        store.delete("session:abc123").await;

        assert!(store.get("session:abc123").await.is_none());
    }
}

#[cfg(test)]
mod permission_tests {
    use super::*;

    #[test]
    fn test_permission_checking() {
        let claims = Claims {
            sub: "user-123".to_string(),
            email: "test@example.com".to_string(),
            name: "Test User".to_string(),
            iat: 0,
            exp: 0,
            iss: "mycelix-mail".to_string(),
            aud: "mycelix-client".to_string(),
            token_type: TokenType::Access,
            permissions: vec![
                "email:read".to_string(),
                "email:write".to_string(),
                "contacts:read".to_string(),
            ],
        };

        assert!(claims.permissions.contains(&"email:read".to_string()));
        assert!(claims.permissions.contains(&"email:write".to_string()));
        assert!(!claims.permissions.contains(&"admin:all".to_string()));
    }

    #[test]
    fn test_token_type_distinction() {
        let access = TokenType::Access;
        let refresh = TokenType::Refresh;

        assert_ne!(
            format!("{:?}", access),
            format!("{:?}", refresh)
        );
    }
}
