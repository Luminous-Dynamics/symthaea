// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Redis caching layer for Mycelix Mail
//!
//! Provides caching with TTL, invalidation, and cache-aside pattern

use redis::{aio::ConnectionManager, AsyncCommands, RedisError};
use serde::{de::DeserializeOwned, Serialize};
use std::time::Duration;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CacheError {
    #[error("Redis error: {0}")]
    Redis(#[from] RedisError),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Cache miss")]
    Miss,
}

pub type CacheResult<T> = Result<T, CacheError>;

/// Cache configuration
#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub url: String,
    pub default_ttl: Duration,
    pub key_prefix: String,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            url: std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string()),
            default_ttl: Duration::from_secs(300), // 5 minutes
            key_prefix: "mycelix:".to_string(),
        }
    }
}

/// Redis cache client
#[derive(Clone)]
pub struct Cache {
    conn: ConnectionManager,
    config: CacheConfig,
}

impl Cache {
    /// Create a new cache connection
    pub async fn new(config: CacheConfig) -> CacheResult<Self> {
        let client = redis::Client::open(config.url.as_str())?;
        let conn = ConnectionManager::new(client).await?;

        Ok(Self { conn, config })
    }

    /// Build a prefixed key
    fn key(&self, key: &str) -> String {
        format!("{}{}", self.config.key_prefix, key)
    }

    /// Get a value from cache
    pub async fn get<T: DeserializeOwned>(&self, key: &str) -> CacheResult<T> {
        let mut conn = self.conn.clone();
        let full_key = self.key(key);

        let value: Option<String> = conn.get(&full_key).await?;

        match value {
            Some(json) => {
                let parsed: T = serde_json::from_str(&json)?;
                Ok(parsed)
            }
            None => Err(CacheError::Miss),
        }
    }

    /// Set a value in cache with default TTL
    pub async fn set<T: Serialize>(&self, key: &str, value: &T) -> CacheResult<()> {
        self.set_with_ttl(key, value, self.config.default_ttl).await
    }

    /// Set a value in cache with custom TTL
    pub async fn set_with_ttl<T: Serialize>(
        &self,
        key: &str,
        value: &T,
        ttl: Duration,
    ) -> CacheResult<()> {
        let mut conn = self.conn.clone();
        let full_key = self.key(key);
        let json = serde_json::to_string(value)?;

        conn.set_ex(&full_key, json, ttl.as_secs()).await?;

        Ok(())
    }

    /// Delete a value from cache
    pub async fn delete(&self, key: &str) -> CacheResult<()> {
        let mut conn = self.conn.clone();
        let full_key = self.key(key);

        conn.del(&full_key).await?;

        Ok(())
    }

    /// Delete all keys matching a pattern
    pub async fn delete_pattern(&self, pattern: &str) -> CacheResult<u64> {
        let mut conn = self.conn.clone();
        let full_pattern = self.key(pattern);

        let keys: Vec<String> = redis::cmd("KEYS")
            .arg(&full_pattern)
            .query_async(&mut conn)
            .await?;

        if keys.is_empty() {
            return Ok(0);
        }

        let count: u64 = conn.del(&keys).await?;

        Ok(count)
    }

    /// Check if a key exists
    pub async fn exists(&self, key: &str) -> CacheResult<bool> {
        let mut conn = self.conn.clone();
        let full_key = self.key(key);

        let exists: bool = conn.exists(&full_key).await?;

        Ok(exists)
    }

    /// Get or set (cache-aside pattern)
    pub async fn get_or_set<T, F, Fut>(
        &self,
        key: &str,
        ttl: Duration,
        f: F,
    ) -> CacheResult<T>
    where
        T: Serialize + DeserializeOwned,
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = CacheResult<T>>,
    {
        // Try to get from cache first
        if let Ok(cached) = self.get::<T>(key).await {
            return Ok(cached);
        }

        // Cache miss - compute value
        let value = f().await?;

        // Store in cache
        self.set_with_ttl(key, &value, ttl).await?;

        Ok(value)
    }

    /// Increment a counter
    pub async fn incr(&self, key: &str) -> CacheResult<i64> {
        let mut conn = self.conn.clone();
        let full_key = self.key(key);

        let value: i64 = conn.incr(&full_key, 1).await?;

        Ok(value)
    }

    /// Set expiration on a key
    pub async fn expire(&self, key: &str, ttl: Duration) -> CacheResult<()> {
        let mut conn = self.conn.clone();
        let full_key = self.key(key);

        conn.expire(&full_key, ttl.as_secs() as i64).await?;

        Ok(())
    }

    /// Get TTL of a key
    pub async fn ttl(&self, key: &str) -> CacheResult<i64> {
        let mut conn = self.conn.clone();
        let full_key = self.key(key);

        let ttl: i64 = conn.ttl(&full_key).await?;

        Ok(ttl)
    }

    /// Health check
    pub async fn health_check(&self) -> CacheResult<()> {
        let mut conn = self.conn.clone();
        let _: String = redis::cmd("PING").query_async(&mut conn).await?;
        Ok(())
    }
}

// ============================================================================
// Cache Keys
// ============================================================================

/// Standardized cache key builders
pub struct CacheKeys;

impl CacheKeys {
    pub fn user(user_id: &str) -> String {
        format!("user:{}", user_id)
    }

    pub fn user_emails(user_id: &str, page: u32) -> String {
        format!("user:{}:emails:{}", user_id, page)
    }

    pub fn email(email_id: &str) -> String {
        format!("email:{}", email_id)
    }

    pub fn contact(contact_id: &str) -> String {
        format!("contact:{}", contact_id)
    }

    pub fn contacts(user_id: &str) -> String {
        format!("user:{}:contacts", user_id)
    }

    pub fn trust_score(agent_id: &str) -> String {
        format!("trust:{}", agent_id)
    }

    pub fn session(session_id: &str) -> String {
        format!("session:{}", session_id)
    }

    pub fn rate_limit(ip: &str, endpoint: &str) -> String {
        format!("ratelimit:{}:{}", ip, endpoint)
    }

    pub fn oauth_state(state: &str) -> String {
        format!("oauth:state:{}", state)
    }

    pub fn pkce_verifier(state: &str) -> String {
        format!("oauth:pkce:{}", state)
    }
}

// ============================================================================
// Cached Repository Wrapper
// ============================================================================

use super::repositories::*;
use super::models::*;
use uuid::Uuid;

/// Cached user repository
pub struct CachedUserRepository {
    repo: UserRepository,
    cache: Cache,
}

impl CachedUserRepository {
    pub fn new(repo: UserRepository, cache: Cache) -> Self {
        Self { repo, cache }
    }

    pub async fn find_by_id(&self, id: Uuid) -> super::DbResult<Option<User>> {
        let key = CacheKeys::user(&id.to_string());

        // Try cache first
        if let Ok(user) = self.cache.get::<User>(&key).await {
            return Ok(Some(user));
        }

        // Cache miss - query DB
        let user = self.repo.find_by_id(id).await?;

        // Cache the result
        if let Some(ref u) = user {
            let _ = self.cache.set(&key, u).await;
        }

        Ok(user)
    }

    pub async fn invalidate(&self, id: Uuid) -> CacheResult<()> {
        let key = CacheKeys::user(&id.to_string());
        self.cache.delete(&key).await
    }
}

/// Cached contact repository
pub struct CachedContactRepository {
    repo: ContactRepository,
    cache: Cache,
}

impl CachedContactRepository {
    pub fn new(repo: ContactRepository, cache: Cache) -> Self {
        Self { repo, cache }
    }

    pub async fn find_by_id(&self, id: Uuid) -> super::DbResult<Option<Contact>> {
        let key = CacheKeys::contact(&id.to_string());

        if let Ok(contact) = self.cache.get::<Contact>(&key).await {
            return Ok(Some(contact));
        }

        let contact = self.repo.find_by_id(id).await?;

        if let Some(ref c) = contact {
            let _ = self.cache.set(&key, c).await;
        }

        Ok(contact)
    }

    pub async fn invalidate(&self, id: Uuid) -> CacheResult<()> {
        let key = CacheKeys::contact(&id.to_string());
        self.cache.delete(&key).await
    }

    pub async fn invalidate_user_contacts(&self, user_id: Uuid) -> CacheResult<()> {
        let pattern = format!("user:{}:contacts*", user_id);
        self.cache.delete_pattern(&pattern).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_builders() {
        assert_eq!(CacheKeys::user("123"), "user:123");
        assert_eq!(CacheKeys::user_emails("123", 1), "user:123:emails:1");
        assert_eq!(CacheKeys::email("abc"), "email:abc");
        assert_eq!(CacheKeys::trust_score("agent1"), "trust:agent1");
        assert_eq!(CacheKeys::rate_limit("1.2.3.4", "/api/send"), "ratelimit:1.2.3.4:/api/send");
    }

    #[test]
    fn test_default_config() {
        let config = CacheConfig::default();
        assert_eq!(config.default_ttl, Duration::from_secs(300));
        assert_eq!(config.key_prefix, "mycelix:");
    }
}
