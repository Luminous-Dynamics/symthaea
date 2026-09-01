// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! K1: Socket Authentication
//!
//! Token-based authentication for IPC clients connecting to the Symthaea
//! service. Unix peer credentials are identity evidence only: being local or
//! running as uid 0 does not itself create application authority.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

/// Authentication token for IPC clients.
#[derive(Debug, Clone)]
pub struct AuthToken {
    /// Unique token identifier.
    pub id: String,
    /// Token value, hashed for storage.
    pub token_hash: String,
    /// Client identifier.
    pub client_id: String,
    /// Permissions granted.
    pub permissions: Permissions,
    /// Creation time.
    pub created_at: SystemTime,
    /// Expiration time.
    pub expires_at: Option<SystemTime>,
    /// Last used time.
    pub last_used: Option<SystemTime>,
}

impl AuthToken {
    /// Create a new auth token record.
    ///
    /// Prefer [`AuthProvider::create_token`] when the cleartext token value is
    /// needed by a client; this constructor intentionally stores only its hash.
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

    /// Create a token record with expiration.
    pub fn with_expiry(mut self, duration: Duration) -> Self {
        self.expires_at = Some(SystemTime::now() + duration);
        self
    }

    /// Check whether the token is expired.
    pub fn is_expired(&self) -> bool {
        self.expires_at
            .is_some_and(|expires| SystemTime::now() > expires)
    }

    /// Verify a token value against this record.
    pub fn verify(&self, token: &str) -> bool {
        !self.is_expired() && hash_token(token) == self.token_hash
    }

    /// Update the last-used timestamp.
    pub fn touch(&mut self) {
        self.last_used = Some(SystemTime::now());
    }

    /// Check whether the token has a specific permission.
    pub fn has_permission(&self, permission: Permission) -> bool {
        self.permissions.has(permission)
    }
}

/// Legacy coarse permission levels for IPC clients.
///
/// These remain for compatibility while mutation paths migrate to explicit
/// bounded capabilities. In particular, `Execute` must not be inferred from
/// locality, uid, Phi, or model confidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    /// Read status and metrics.
    Read,
    /// Request completions and validation.
    Query,
    /// Execute commands through the legacy path.
    Execute,
    /// Modify configuration.
    Configure,
    /// Administrative actions.
    Admin,
}

/// Set of legacy permissions.
#[derive(Debug, Clone, Default)]
pub struct Permissions {
    flags: u8,
}

impl Permissions {
    /// Empty permission set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Read-only permissions.
    pub fn read_only() -> Self {
        Self::new().with(Permission::Read)
    }

    /// Legacy shell permissions.
    pub fn shell() -> Self {
        Self::new()
            .with(Permission::Read)
            .with(Permission::Query)
            .with(Permission::Execute)
    }

    /// Legacy GUI permissions.
    pub fn gui() -> Self {
        Self::new()
            .with(Permission::Read)
            .with(Permission::Query)
            .with(Permission::Configure)
    }

    /// Full legacy permissions.
    pub fn full() -> Self {
        Self::new()
            .with(Permission::Read)
            .with(Permission::Query)
            .with(Permission::Execute)
            .with(Permission::Configure)
            .with(Permission::Admin)
    }

    /// Add a permission.
    pub fn with(mut self, permission: Permission) -> Self {
        self.flags |= 1 << (permission as u8);
        self
    }

    /// Check whether this set includes a permission.
    pub fn has(&self, permission: Permission) -> bool {
        self.flags & (1 << (permission as u8)) != 0
    }

    /// List all included permissions.
    pub fn list(&self) -> Vec<Permission> {
        [
            Permission::Read,
            Permission::Query,
            Permission::Execute,
            Permission::Configure,
            Permission::Admin,
        ]
        .into_iter()
        .filter(|permission| self.has(*permission))
        .collect()
    }
}

/// Identity evidence obtained from Unix peer credentials.
///
/// This type deliberately contains no permissions. Authorization must be
/// supplied independently through a token/capability decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalPeerIdentity {
    /// Kernel-reported effective user id of the peer.
    pub uid: u32,
    /// Kernel-reported effective group id of the peer.
    pub gid: u32,
}

/// Authentication provider.
pub struct AuthProvider {
    tokens: HashMap<String, AuthToken>,
    token_lookup: HashMap<String, String>,
    storage_path: Option<PathBuf>,
    enabled: bool,
    /// Explicit compatibility switch for unauthenticated local *read-only*
    /// access. Disabled by default.
    allow_local_read_only: bool,
}

impl AuthProvider {
    /// Create a default-deny authentication provider.
    pub fn new() -> Self {
        Self {
            tokens: HashMap::new(),
            token_lookup: HashMap::new(),
            storage_path: None,
            enabled: true,
            allow_local_read_only: false,
        }
    }

    /// Set storage path for token persistence.
    pub fn with_storage(mut self, path: impl Into<PathBuf>) -> Self {
        self.storage_path = Some(path.into());
        self
    }

    /// Disable this authentication provider.
    ///
    /// Disabled does not mean implicitly authorized: authentication attempts,
    /// including local compatibility auth, return [`AuthError::Disabled`].
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Require authenticated authorization for every local connection.
    pub fn require_auth_always(mut self) -> Self {
        self.allow_local_read_only = false;
        self
    }

    /// Explicitly permit unauthenticated local read-only compatibility access.
    ///
    /// This never grants Query, Execute, Configure, or Admin, including to
    /// uid 0. Prefer token/capability authorization for new callers.
    pub fn allow_local_read_only(mut self) -> Self {
        self.allow_local_read_only = true;
        self
    }

    /// Check whether authentication is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Create and register a new token.
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

    /// Create and register a token with expiration.
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

    /// Authenticate a token value.
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

    /// Check whether explicit unauthenticated local read-only compatibility is enabled.
    pub fn allows_local(&self) -> bool {
        self.allow_local_read_only
    }

    /// Record Unix peer identity without conferring any application authority.
    pub fn identify_local(&self, uid: u32, gid: u32) -> LocalPeerIdentity {
        LocalPeerIdentity { uid, gid }
    }

    /// Authenticate through the explicit local read-only compatibility path.
    ///
    /// Unix peer credentials are not an authorization source. Default
    /// providers reject this path, and even explicit opt-in only grants Read.
    pub fn authenticate_local(&self, uid: u32, gid: u32) -> Result<LocalAuth, AuthError> {
        if !self.enabled {
            return Err(AuthError::Disabled);
        }
        if !self.allow_local_read_only {
            return Err(AuthError::LocalAuthDisabled);
        }

        Ok(LocalAuth {
            identity: self.identify_local(uid, gid),
            permissions: Permissions::read_only(),
        })
    }

    /// Revoke a token.
    pub fn revoke(&mut self, token_id: &str) -> bool {
        if let Some(token) = self.tokens.remove(token_id) {
            self.token_lookup.remove(&token.token_hash);
            true
        } else {
            false
        }
    }

    /// Revoke all tokens for a client.
    pub fn revoke_client(&mut self, client_id: &str) {
        let to_remove: Vec<String> = self
            .tokens
            .iter()
            .filter(|(_, token)| token.client_id == client_id)
            .map(|(id, _)| id.clone())
            .collect();
        for id in to_remove {
            self.revoke(&id);
        }
    }

    /// Clean up expired tokens.
    pub fn cleanup_expired(&mut self) {
        let expired: Vec<String> = self
            .tokens
            .iter()
            .filter(|(_, token)| token.is_expired())
            .map(|(id, _)| id.clone())
            .collect();
        for id in expired {
            self.revoke(&id);
        }
    }

    /// Get a token by id.
    pub fn get_token(&self, token_id: &str) -> Option<&AuthToken> {
        self.tokens.get(token_id)
    }

    /// List all tokens for one client.
    pub fn list_client_tokens(&self, client_id: &str) -> Vec<&AuthToken> {
        self.tokens
            .values()
            .filter(|token| token.client_id == client_id)
            .collect()
    }

    /// Persist token records using the existing line-oriented format.
    pub fn save(&self) -> io::Result<()> {
        if let Some(path) = &self.storage_path {
            fs::write(path, self.serialize_tokens()?)?;
        }
        Ok(())
    }

    /// Load token records using the existing line-oriented format.
    pub fn load(&mut self) -> io::Result<()> {
        if let Some(path) = &self.storage_path {
            if path.exists() {
                self.deserialize_tokens(&fs::read_to_string(path)?)?;
            }
        }
        Ok(())
    }

    fn serialize_tokens(&self) -> io::Result<String> {
        let mut lines = Vec::new();
        for token in self.tokens.values() {
            let expires = token
                .expires_at
                .map(|time| {
                    time.duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs()
                        .to_string()
                })
                .unwrap_or_else(|| "none".to_string());
            let created = token
                .created_at
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            lines.push(format!(
                "{}:{}:{}:{}:{}:{}",
                token.id,
                token.client_id,
                token.token_hash,
                token.permissions.flags,
                created,
                expires
            ));
        }
        Ok(lines.join("\n"))
    }

    fn deserialize_tokens(&mut self, data: &str) -> io::Result<()> {
        for line in data.lines() {
            let parts: Vec<&str> = line.split(':').collect();
            if parts.len() < 6 {
                continue;
            }
            let id = parts[0].to_string();
            let client_id = parts[1].to_string();
            let token_hash = parts[2].to_string();
            let flags: u8 = parts[3].parse().unwrap_or(0);
            let created_secs: u64 = parts[4].parse().unwrap_or(0);
            let expires_at = if parts[5] == "none" {
                None
            } else {
                parts[5]
                    .parse::<u64>()
                    .ok()
                    .map(|seconds| SystemTime::UNIX_EPOCH + Duration::from_secs(seconds))
            };
            let token = AuthToken {
                id: id.clone(),
                token_hash: token_hash.clone(),
                client_id,
                permissions: Permissions { flags },
                created_at: SystemTime::UNIX_EPOCH + Duration::from_secs(created_secs),
                expires_at,
                last_used: None,
            };
            self.token_lookup.insert(token_hash, id.clone());
            self.tokens.insert(id, token);
        }
        Ok(())
    }
}

impl Default for AuthProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of the explicit local read-only compatibility authentication path.
#[derive(Debug)]
pub struct LocalAuth {
    /// Kernel-provided identity evidence.
    pub identity: LocalPeerIdentity,
    /// Compatibility permissions. This path only grants Read.
    pub permissions: Permissions,
}

impl LocalAuth {
    /// Peer uid convenience accessor retained for migration readability.
    pub fn uid(&self) -> u32 {
        self.identity.uid
    }

    /// Peer gid convenience accessor retained for migration readability.
    pub fn gid(&self) -> u32 {
        self.identity.gid
    }
}

/// Authentication errors.
#[derive(Debug, Clone)]
pub enum AuthError {
    InvalidToken,
    TokenExpired,
    Disabled,
    LocalAuthDisabled,
    InsufficientPermissions(Permission),
    IoError(String),
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidToken => write!(f, "Invalid authentication token"),
            Self::TokenExpired => write!(f, "Token has expired"),
            Self::Disabled => write!(f, "Authentication is disabled"),
            Self::LocalAuthDisabled => write!(f, "Local authentication not allowed"),
            Self::InsufficientPermissions(permission) => {
                write!(f, "Missing permission: {permission:?}")
            }
            Self::IoError(error) => write!(f, "IO error: {error}"),
        }
    }
}

impl std::error::Error for AuthError {}

fn generate_token() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    format!("sym_{hex}")
}

fn generate_token_id() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    format!("tok_{hex}")
}

fn hash_token(token: &str) -> String {
    blake3::hash(token.as_bytes()).to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_permissions_remain_available_for_explicit_tokens() {
        let permissions = Permissions::shell();
        assert!(permissions.has(Permission::Read));
        assert!(permissions.has(Permission::Query));
        assert!(permissions.has(Permission::Execute));
        assert!(!permissions.has(Permission::Admin));
    }

    #[test]
    fn token_creation_and_authentication_round_trip() {
        let mut provider = AuthProvider::new();
        let (value, token) = provider.create_token("test-client", Permissions::shell());
        assert!(value.starts_with("sym_"));
        assert_eq!(token.client_id, "test-client");
        assert!(provider.authenticate(&value).is_ok());
        assert!(provider.authenticate("bad_token").is_err());
    }

    #[test]
    fn token_expiration_is_enforced() {
        let mut provider = AuthProvider::new();
        let (value, _) = provider.create_temp_token(
            "test-client",
            Permissions::read_only(),
            Duration::from_millis(1),
        );
        std::thread::sleep(Duration::from_millis(10));
        assert!(matches!(provider.authenticate(&value), Err(AuthError::TokenExpired)));
    }

    #[test]
    fn token_revocation_is_enforced() {
        let mut provider = AuthProvider::new();
        let (value, token) = provider.create_token("test-client", Permissions::shell());
        assert!(provider.authenticate(&value).is_ok());
        assert!(provider.revoke(&token.id));
        assert!(provider.authenticate(&value).is_err());
    }

    #[test]
    fn default_local_auth_denies_root_and_user() {
        let provider = AuthProvider::new();
        assert!(matches!(
            provider.authenticate_local(0, 0),
            Err(AuthError::LocalAuthDisabled)
        ));
        assert!(matches!(
            provider.authenticate_local(1000, 1000),
            Err(AuthError::LocalAuthDisabled)
        ));
    }

    #[test]
    fn peer_credentials_identify_but_do_not_authorize() {
        let provider = AuthProvider::new();
        let identity = provider.identify_local(1000, 1001);
        assert_eq!(identity, LocalPeerIdentity { uid: 1000, gid: 1001 });
    }

    #[test]
    fn explicit_local_compatibility_is_read_only_even_for_root() {
        let provider = AuthProvider::new().allow_local_read_only();
        let root = provider.authenticate_local(0, 0).unwrap();
        let user = provider.authenticate_local(1000, 1000).unwrap();
        for auth in [&root, &user] {
            assert!(auth.permissions.has(Permission::Read));
            assert!(!auth.permissions.has(Permission::Query));
            assert!(!auth.permissions.has(Permission::Execute));
            assert!(!auth.permissions.has(Permission::Configure));
            assert!(!auth.permissions.has(Permission::Admin));
        }
    }

    #[test]
    fn disabled_provider_does_not_turn_locality_into_authority() {
        let provider = AuthProvider::new().disabled().allow_local_read_only();
        assert!(matches!(provider.authenticate_local(0, 0), Err(AuthError::Disabled)));
    }

    #[test]
    fn token_hash_uses_blake3() {
        let token = "sym_test_token";
        let hash = hash_token(token);
        assert_eq!(hash.len(), 64);
        assert_eq!(hash, hash_token(token));
    }

    #[test]
    fn generated_tokens_are_unique() {
        assert_ne!(generate_token(), generate_token());
    }

    #[test]
    fn auth_error_display_is_stable() {
        assert_eq!(AuthError::InvalidToken.to_string(), "Invalid authentication token");
        assert_eq!(AuthError::TokenExpired.to_string(), "Token has expired");
        assert_eq!(AuthError::Disabled.to_string(), "Authentication is disabled");
        assert_eq!(
            AuthError::LocalAuthDisabled.to_string(),
            "Local authentication not allowed"
        );
        assert_eq!(
            AuthError::InsufficientPermissions(Permission::Execute).to_string(),
            "Missing permission: Execute"
        );
        assert_eq!(
            AuthError::IoError("connection refused".into()).to_string(),
            "IO error: connection refused"
        );
    }
}
