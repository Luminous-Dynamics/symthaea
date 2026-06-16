// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! K1: Socket Authentication
//!
//! Token-based authentication for IPC clients connecting
//! to the Symthaea service.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

/// Authentication token for IPC clients
#[derive(Debug, Clone)]
pub struct AuthToken {
    /// Unique token identifier
    pub id: String,
    /// Token value (hashed for storage)
    pub token_hash: String,
    /// Client identifier
    pub client_id: String,
    /// Permissions granted
    pub permissions: Permissions,
    /// Creation time
    pub created_at: SystemTime,
    /// Expiration time
    pub expires_at: Option<SystemTime>,
    /// Last used time
    pub last_used: Option<SystemTime>,
}

impl AuthToken {
    /// Create a new auth token
    pub fn new(client_id: impl Into<String>, permissions: Permissions) -> Self {
        let id = generate_token_id();
        let token = generate_token();
        let token_hash = hash_token(&token);

        Self {
            id,
            token_hash,
            client_id: client_id.into(),
            permissions,
            created_at: SystemTime::now(),
            expires_at: None,
            last_used: None,
        }
    }

    /// Create a token with expiration
    pub fn with_expiry(mut self, duration: Duration) -> Self {
        self.expires_at = Some(SystemTime::now() + duration);
        self
    }

    /// Check if token is expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires) = self.expires_at {
            SystemTime::now() > expires
        } else {
            false
        }
    }

    /// Verify a token value against this token
    pub fn verify(&self, token: &str) -> bool {
        if self.is_expired() {
            return false;
        }
        hash_token(token) == self.token_hash
    }

    /// Update last used time
    pub fn touch(&mut self) {
        self.last_used = Some(SystemTime::now());
    }

    /// Check if token has specific permission
    pub fn has_permission(&self, permission: Permission) -> bool {
        self.permissions.has(permission)
    }
}

/// Permission levels for clients
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    /// Read status and metrics
    Read,
    /// Request completions and validation
    Query,
    /// Execute commands (phi-gated)
    Execute,
    /// Modify configuration
    Configure,
    /// Administrative actions
    Admin,
}

/// Set of permissions
#[derive(Debug, Clone, Default)]
pub struct Permissions {
    flags: u8,
}

impl Permissions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Read-only permissions
    pub fn read_only() -> Self {
        Self::new().with(Permission::Read)
    }

    /// Standard shell permissions
    pub fn shell() -> Self {
        Self::new()
            .with(Permission::Read)
            .with(Permission::Query)
            .with(Permission::Execute)
    }

    /// GUI permissions
    pub fn gui() -> Self {
        Self::new()
            .with(Permission::Read)
            .with(Permission::Query)
            .with(Permission::Configure)
    }

    /// Full permissions
    pub fn full() -> Self {
        Self::new()
            .with(Permission::Read)
            .with(Permission::Query)
            .with(Permission::Execute)
            .with(Permission::Configure)
            .with(Permission::Admin)
    }

    /// Add a permission
    pub fn with(mut self, permission: Permission) -> Self {
        self.flags |= 1 << (permission as u8);
        self
    }

    /// Check if has permission
    pub fn has(&self, permission: Permission) -> bool {
        self.flags & (1 << (permission as u8)) != 0
    }

    /// List all permissions
    pub fn list(&self) -> Vec<Permission> {
        let mut perms = Vec::new();
        for p in [
            Permission::Read,
            Permission::Query,
            Permission::Execute,
            Permission::Configure,
            Permission::Admin,
        ] {
            if self.has(p) {
                perms.push(p);
            }
        }
        perms
    }
}

/// Authentication provider
pub struct AuthProvider {
    /// Active tokens by ID
    tokens: HashMap<String, AuthToken>,
    /// Token lookup by hash
    token_lookup: HashMap<String, String>,
    /// Token storage path
    storage_path: Option<PathBuf>,
    /// Enable authentication
    enabled: bool,
    /// Allow unauthenticated local connections
    allow_local: bool,
}

impl AuthProvider {
    /// Create a new auth provider
    pub fn new() -> Self {
        Self {
            tokens: HashMap::new(),
            token_lookup: HashMap::new(),
            storage_path: None,
            enabled: true,
            allow_local: true,
        }
    }

    /// Set storage path for token persistence
    pub fn with_storage(mut self, path: impl Into<PathBuf>) -> Self {
        self.storage_path = Some(path.into());
        self
    }

    /// Disable authentication (not recommended for production)
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Disallow local unauthenticated connections
    pub fn require_auth_always(mut self) -> Self {
        self.allow_local = false;
        self
    }

    /// Check if authentication is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Create and register a new token
    pub fn create_token(
        &mut self,
        client_id: impl Into<String>,
        permissions: Permissions,
    ) -> (String, AuthToken) {
        let token_value = generate_token();
        let token = AuthToken {
            id: generate_token_id(),
            token_hash: hash_token(&token_value),
            client_id: client_id.into(),
            permissions,
            created_at: SystemTime::now(),
            expires_at: None,
            last_used: None,
        };

        self.token_lookup
            .insert(token.token_hash.clone(), token.id.clone());
        self.tokens.insert(token.id.clone(), token.clone());

        (token_value, token)
    }

    /// Create a token with expiration
    pub fn create_temp_token(
        &mut self,
        client_id: impl Into<String>,
        permissions: Permissions,
        duration: Duration,
    ) -> (String, AuthToken) {
        let (value, mut token) = self.create_token(client_id, permissions);
        token.expires_at = Some(SystemTime::now() + duration);
        self.tokens.insert(token.id.clone(), token.clone());
        (value, token)
    }

    /// Authenticate a token value
    pub fn authenticate(&mut self, token_value: &str) -> Result<&AuthToken, AuthError> {
        if !self.enabled {
            return Err(AuthError::Disabled);
        }

        let hash = hash_token(token_value);

        if let Some(id) = self.token_lookup.get(&hash).cloned() {
            if let Some(token) = self.tokens.get_mut(&id) {
                if token.is_expired() {
                    return Err(AuthError::TokenExpired);
                }
                token.touch();
                return Ok(token);
            }
        }

        Err(AuthError::InvalidToken)
    }

    /// Check if local connections are allowed without auth
    pub fn allows_local(&self) -> bool {
        self.allow_local
    }

    /// Authenticate from peer credentials (Unix socket)
    pub fn authenticate_local(&self, uid: u32, gid: u32) -> Result<LocalAuth, AuthError> {
        if !self.enabled || self.allow_local {
            // Local connections allowed
            let is_root = uid == 0;
            let permissions = if is_root {
                Permissions::full()
            } else {
                Permissions::shell()
            };

            return Ok(LocalAuth {
                uid,
                gid,
                permissions,
            });
        }

        Err(AuthError::LocalAuthDisabled)
    }

    /// Revoke a token
    pub fn revoke(&mut self, token_id: &str) -> bool {
        if let Some(token) = self.tokens.remove(token_id) {
            self.token_lookup.remove(&token.token_hash);
            true
        } else {
            false
        }
    }

    /// Revoke all tokens for a client
    pub fn revoke_client(&mut self, client_id: &str) {
        let to_remove: Vec<String> = self
            .tokens
            .iter()
            .filter(|(_, t)| t.client_id == client_id)
            .map(|(id, _)| id.clone())
            .collect();

        for id in to_remove {
            self.revoke(&id);
        }
    }

    /// Clean up expired tokens
    pub fn cleanup_expired(&mut self) {
        let expired: Vec<String> = self
            .tokens
            .iter()
            .filter(|(_, t)| t.is_expired())
            .map(|(id, _)| id.clone())
            .collect();

        for id in expired {
            self.revoke(&id);
        }
    }

    /// Get token by ID
    pub fn get_token(&self, token_id: &str) -> Option<&AuthToken> {
        self.tokens.get(token_id)
    }

    /// List all tokens for a client
    pub fn list_client_tokens(&self, client_id: &str) -> Vec<&AuthToken> {
        self.tokens
            .values()
            .filter(|t| t.client_id == client_id)
            .collect()
    }

    /// Persist tokens to storage
    pub fn save(&self) -> io::Result<()> {
        if let Some(path) = &self.storage_path {
            let data = self.serialize_tokens()?;
            fs::write(path, data)?;
        }
        Ok(())
    }

    /// Load tokens from storage
    pub fn load(&mut self) -> io::Result<()> {
        if let Some(path) = &self.storage_path {
            if path.exists() {
                let data = fs::read_to_string(path)?;
                self.deserialize_tokens(&data)?;
            }
        }
        Ok(())
    }

    fn serialize_tokens(&self) -> io::Result<String> {
        // Simple format: one token per line
        // id:client_id:hash:perms:created:expires
        let mut lines = Vec::new();
        for token in self.tokens.values() {
            let expires = token
                .expires_at
                .map(|t| {
                    format!(
                        "{}",
                        t.duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs()
                    )
                })
                .unwrap_or_else(|| "none".to_string());

            let created = token
                .created_at
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            let line = format!(
                "{}:{}:{}:{}:{}:{}",
                token.id,
                token.client_id,
                token.token_hash,
                token.permissions.flags,
                created,
                expires
            );
            lines.push(line);
        }
        Ok(lines.join("\n"))
    }

    fn deserialize_tokens(&mut self, data: &str) -> io::Result<()> {
        for line in data.lines() {
            let parts: Vec<&str> = line.split(':').collect();
            if parts.len() >= 6 {
                let id = parts[0].to_string();
                let client_id = parts[1].to_string();
                let token_hash = parts[2].to_string();
                let flags: u8 = parts[3].parse().unwrap_or(0);
                let created_secs: u64 = parts[4].parse().unwrap_or(0);
                let expires = if parts[5] == "none" {
                    None
                } else {
                    parts[5]
                        .parse::<u64>()
                        .ok()
                        .map(|s| SystemTime::UNIX_EPOCH + Duration::from_secs(s))
                };

                let token = AuthToken {
                    id: id.clone(),
                    token_hash: token_hash.clone(),
                    client_id,
                    permissions: Permissions { flags },
                    created_at: SystemTime::UNIX_EPOCH + Duration::from_secs(created_secs),
                    expires_at: expires,
                    last_used: None,
                };

                self.token_lookup.insert(token_hash, id.clone());
                self.tokens.insert(id, token);
            }
        }
        Ok(())
    }
}

impl Default for AuthProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Local authentication result
#[derive(Debug)]
pub struct LocalAuth {
    pub uid: u32,
    pub gid: u32,
    pub permissions: Permissions,
}

/// Authentication errors
#[derive(Debug, Clone)]
pub enum AuthError {
    /// Invalid token provided
    InvalidToken,
    /// Token has expired
    TokenExpired,
    /// Authentication is disabled
    Disabled,
    /// Local auth not allowed
    LocalAuthDisabled,
    /// Insufficient permissions
    InsufficientPermissions(Permission),
    /// IO error
    IoError(String),
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidToken => write!(f, "Invalid authentication token"),
            Self::TokenExpired => write!(f, "Token has expired"),
            Self::Disabled => write!(f, "Authentication is disabled"),
            Self::LocalAuthDisabled => write!(f, "Local authentication not allowed"),
            Self::InsufficientPermissions(p) => write!(f, "Missing permission: {p:?}"),
            Self::IoError(e) => write!(f, "IO error: {e}"),
        }
    }
}

impl std::error::Error for AuthError {}

/// Generate a cryptographically random authentication token.
///
/// Uses OS entropy via `OsRng` for unpredictable token values.
fn generate_token() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("sym_{hex}")
}

/// Generate a cryptographically random token ID.
fn generate_token_id() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("tok_{hex}")
}

/// Hash a token using BLAKE3 for secure, non-reversible storage.
fn hash_token(token: &str) -> String {
    blake3::hash(token.as_bytes()).to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permissions() {
        let perms = Permissions::shell();
        assert!(perms.has(Permission::Read));
        assert!(perms.has(Permission::Query));
        assert!(perms.has(Permission::Execute));
        assert!(!perms.has(Permission::Admin));
    }

    #[test]
    fn test_token_creation() {
        let mut provider = AuthProvider::new();
        let (value, token) = provider.create_token("test-client", Permissions::shell());

        assert!(value.starts_with("sym_"));
        assert_eq!(token.client_id, "test-client");
        assert!(token.has_permission(Permission::Execute));
    }

    #[test]
    fn test_token_authentication() {
        let mut provider = AuthProvider::new();
        let (value, _) = provider.create_token("test-client", Permissions::shell());

        let result = provider.authenticate(&value);
        assert!(result.is_ok());

        let bad_result = provider.authenticate("bad_token");
        assert!(bad_result.is_err());
    }

    #[test]
    fn test_token_expiration() {
        let mut provider = AuthProvider::new();
        let (value, _) = provider.create_temp_token(
            "test-client",
            Permissions::read_only(),
            Duration::from_millis(1),
        );

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(10));

        let result = provider.authenticate(&value);
        assert!(matches!(result, Err(AuthError::TokenExpired)));
    }

    #[test]
    fn test_token_revocation() {
        let mut provider = AuthProvider::new();
        let (value, token) = provider.create_token("test-client", Permissions::shell());

        assert!(provider.authenticate(&value).is_ok());
        assert!(provider.revoke(&token.id));
        assert!(provider.authenticate(&value).is_err());
    }

    #[test]
    fn test_tokens_are_unique() {
        let t1 = generate_token();
        let t2 = generate_token();
        assert_ne!(t1, t2, "Two generated tokens should be unique (OS entropy)");
    }

    #[test]
    fn test_token_hash_uses_blake3() {
        let token = "sym_test_token";
        let hash = hash_token(token);
        // BLAKE3 hex output is 64 characters
        assert_eq!(hash.len(), 64, "BLAKE3 hash should be 64 hex chars");
        // Same input should produce same hash
        assert_eq!(hash, hash_token(token));
    }

    #[test]
    fn test_local_auth() {
        let provider = AuthProvider::new();

        // Root gets full permissions
        let root_auth = provider.authenticate_local(0, 0).unwrap();
        assert!(root_auth.permissions.has(Permission::Admin));

        // Normal user gets shell permissions
        let user_auth = provider.authenticate_local(1000, 1000).unwrap();
        assert!(user_auth.permissions.has(Permission::Execute));
        assert!(!user_auth.permissions.has(Permission::Admin));
    }

    // ── AuthError construction and Display/Debug tests ──────────────

    #[test]
    fn test_auth_error_invalid_token_display() {
        let err = AuthError::InvalidToken;
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
        assert_eq!(msg, "Invalid authentication token");
    }

    #[test]
    fn test_auth_error_token_expired_display() {
        let err = AuthError::TokenExpired;
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
        assert_eq!(msg, "Token has expired");
    }

    #[test]
    fn test_auth_error_disabled_display() {
        let err = AuthError::Disabled;
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
        assert_eq!(msg, "Authentication is disabled");
    }

    #[test]
    fn test_auth_error_local_auth_disabled_display() {
        let err = AuthError::LocalAuthDisabled;
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
        assert_eq!(msg, "Local authentication not allowed");
    }

    #[test]
    fn test_auth_error_insufficient_permissions_display_all_variants() {
        let cases = [
            (Permission::Read, "Missing permission: Read"),
            (Permission::Query, "Missing permission: Query"),
            (Permission::Execute, "Missing permission: Execute"),
            (Permission::Configure, "Missing permission: Configure"),
            (Permission::Admin, "Missing permission: Admin"),
        ];
        for (perm, expected) in &cases {
            let err = AuthError::InsufficientPermissions(*perm);
            let msg = format!("{}", err);
            assert!(!msg.is_empty());
            assert_eq!(&msg, expected);
        }
    }

    #[test]
    fn test_auth_error_io_error_display() {
        let err = AuthError::IoError("connection refused".to_string());
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
        assert_eq!(msg, "IO error: connection refused");
    }

    #[test]
    fn test_auth_error_io_error_empty_inner_message() {
        let err = AuthError::IoError(String::new());
        let msg = format!("{}", err);
        // Even with empty inner string, the prefix is still present
        assert_eq!(msg, "IO error: ");
    }

    #[test]
    fn test_auth_error_debug_format_all_variants() {
        let variants: Vec<AuthError> = vec![
            AuthError::InvalidToken,
            AuthError::TokenExpired,
            AuthError::Disabled,
            AuthError::LocalAuthDisabled,
            AuthError::InsufficientPermissions(Permission::Admin),
            AuthError::IoError("test".to_string()),
        ];
        for err in &variants {
            let dbg = format!("{:?}", err);
            assert!(
                !dbg.is_empty(),
                "Debug output must be non-empty for {:?}",
                err
            );
        }
    }

    #[test]
    fn test_auth_error_clone() {
        let original = AuthError::InsufficientPermissions(Permission::Execute);
        let cloned = original.clone();
        assert_eq!(format!("{}", original), format!("{}", cloned));

        let original_io = AuthError::IoError("disk full".to_string());
        let cloned_io = original_io.clone();
        assert_eq!(format!("{}", original_io), format!("{}", cloned_io));
    }

    #[test]
    fn test_auth_error_is_std_error() {
        // Verify that AuthError implements std::error::Error by using it as a trait object
        let err: Box<dyn std::error::Error> = Box::new(AuthError::InvalidToken);
        assert!(!err.to_string().is_empty());

        let err: Box<dyn std::error::Error> = Box::new(AuthError::IoError("fail".into()));
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn test_auth_error_source_is_none() {
        // AuthError does not wrap an inner error, so source() should be None
        use std::error::Error;
        let variants: Vec<AuthError> = vec![
            AuthError::InvalidToken,
            AuthError::TokenExpired,
            AuthError::Disabled,
            AuthError::LocalAuthDisabled,
            AuthError::InsufficientPermissions(Permission::Read),
            AuthError::IoError("x".into()),
        ];
        for err in &variants {
            assert!(
                err.source().is_none(),
                "source() should be None for {:?}",
                err
            );
        }
    }

    // ── AuthError integration with AuthProvider error paths ─────────

    #[test]
    fn test_authenticate_disabled_returns_disabled_error() {
        let mut provider = AuthProvider::new().disabled();
        let result = provider.authenticate("any_token");
        assert!(matches!(result, Err(AuthError::Disabled)));
    }

    #[test]
    fn test_authenticate_bad_token_returns_invalid_token() {
        let mut provider = AuthProvider::new();
        let result = provider.authenticate("nonexistent_token_value");
        assert!(matches!(result, Err(AuthError::InvalidToken)));
    }

    #[test]
    fn test_local_auth_disabled_returns_error() {
        let provider = AuthProvider::new().require_auth_always();
        // With auth enabled AND allow_local=false, local auth should fail
        let result = provider.authenticate_local(1000, 1000);
        assert!(matches!(result, Err(AuthError::LocalAuthDisabled)));
    }
}
