// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-Account Management for Mycelix Mail
//!
//! Allows users to manage multiple email accounts, identities,
//! and unified inbox views.

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Account types supported
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AccountType {
    /// Native Mycelix account with web-of-trust
    Mycelix,
    /// IMAP account (Gmail, Outlook, etc.)
    Imap,
    /// Exchange/Office 365
    Exchange,
    /// Google Workspace (OAuth)
    Google,
    /// Microsoft 365 (OAuth)
    Microsoft,
}

/// Connection status for an account
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConnectionStatus {
    Connected,
    Connecting,
    Disconnected,
    AuthError,
    NetworkError,
    Syncing,
}

/// IMAP configuration for external accounts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImapConfig {
    pub host: String,
    pub port: u16,
    pub use_tls: bool,
    pub use_starttls: bool,
    pub username: String,
    // Password stored encrypted separately
}

/// SMTP configuration for sending
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtpConfig {
    pub host: String,
    pub port: u16,
    pub use_tls: bool,
    pub use_starttls: bool,
    pub username: String,
    pub auth_method: SmtpAuthMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SmtpAuthMethod {
    Plain,
    Login,
    OAuth2,
    CramMd5,
}

/// OAuth2 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuth2Config {
    pub provider: OAuthProvider,
    pub client_id: String,
    pub scopes: Vec<String>,
    pub refresh_token_encrypted: Option<String>,
    pub access_token_encrypted: Option<String>,
    pub token_expiry: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OAuthProvider {
    Google,
    Microsoft,
    Custom { auth_url: String, token_url: String },
}

/// An email account/identity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailAccount {
    pub id: Uuid,
    pub user_id: Uuid,
    pub email_address: String,
    pub display_name: String,
    pub account_type: AccountType,

    // Connection settings
    pub imap_config: Option<ImapConfig>,
    pub smtp_config: Option<SmtpConfig>,
    pub oauth_config: Option<OAuth2Config>,

    // Status
    pub status: ConnectionStatus,
    pub last_sync: Option<DateTime<Utc>>,
    pub last_error: Option<String>,

    // Preferences
    pub is_primary: bool,
    pub color: Option<String>, // For UI differentiation
    pub signature: Option<String>,
    pub reply_to: Option<String>,

    // Sync settings
    pub sync_enabled: bool,
    pub sync_interval_minutes: u32,
    pub sync_folders: Vec<String>,
    pub days_to_sync: u32, // 0 = all

    // Metadata
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Folder mapping between accounts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderMapping {
    pub account_id: Uuid,
    pub local_folder: String,
    pub remote_folder: String,
    pub sync_direction: SyncDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncDirection {
    Bidirectional,
    DownloadOnly,
    UploadOnly,
}

/// Unified inbox configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedInboxConfig {
    pub user_id: Uuid,
    pub enabled: bool,
    pub included_accounts: Vec<Uuid>,
    pub group_by: UnifiedGrouping,
    pub sort_order: SortOrder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnifiedGrouping {
    None,
    ByAccount,
    ByDate,
    ByPriority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortOrder {
    DateDesc,
    DateAsc,
    SenderAsc,
    SenderDesc,
    SubjectAsc,
    SubjectDesc,
}

/// Service for managing multiple accounts
pub struct MultiAccountService {
    accounts: HashMap<Uuid, EmailAccount>,
    folder_mappings: HashMap<Uuid, Vec<FolderMapping>>,
}

impl MultiAccountService {
    pub fn new() -> Self {
        Self {
            accounts: HashMap::new(),
            folder_mappings: HashMap::new(),
        }
    }

    /// Add a new email account
    pub fn add_account(&mut self, account: EmailAccount) -> Result<Uuid, AccountError> {
        // Validate account configuration
        self.validate_account(&account)?;

        let id = account.id;
        self.accounts.insert(id, account);

        Ok(id)
    }

    /// Update an existing account
    pub fn update_account(&mut self, account: EmailAccount) -> Result<(), AccountError> {
        if !self.accounts.contains_key(&account.id) {
            return Err(AccountError::NotFound(account.id));
        }

        self.validate_account(&account)?;
        self.accounts.insert(account.id, account);

        Ok(())
    }

    /// Remove an account
    pub fn remove_account(&mut self, account_id: Uuid) -> Result<(), AccountError> {
        if self.accounts.remove(&account_id).is_none() {
            return Err(AccountError::NotFound(account_id));
        }

        self.folder_mappings.remove(&account_id);
        Ok(())
    }

    /// Get all accounts for a user
    pub fn get_user_accounts(&self, user_id: Uuid) -> Vec<&EmailAccount> {
        self.accounts
            .values()
            .filter(|a| a.user_id == user_id)
            .collect()
    }

    /// Get a specific account
    pub fn get_account(&self, account_id: Uuid) -> Option<&EmailAccount> {
        self.accounts.get(&account_id)
    }

    /// Get primary account for a user
    pub fn get_primary_account(&self, user_id: Uuid) -> Option<&EmailAccount> {
        self.accounts
            .values()
            .find(|a| a.user_id == user_id && a.is_primary)
    }

    /// Set primary account
    pub fn set_primary(&mut self, user_id: Uuid, account_id: Uuid) -> Result<(), AccountError> {
        // Unset current primary
        for account in self.accounts.values_mut() {
            if account.user_id == user_id {
                account.is_primary = account.id == account_id;
            }
        }

        Ok(())
    }

    /// Update account status
    pub fn update_status(&mut self, account_id: Uuid, status: ConnectionStatus, error: Option<String>) {
        if let Some(account) = self.accounts.get_mut(&account_id) {
            account.status = status;
            account.last_error = error;

            if matches!(account.status, ConnectionStatus::Connected) {
                account.last_sync = Some(Utc::now());
            }
        }
    }

    /// Get accounts that need syncing
    pub fn get_accounts_to_sync(&self) -> Vec<&EmailAccount> {
        let now = Utc::now();

        self.accounts
            .values()
            .filter(|a| {
                if !a.sync_enabled {
                    return false;
                }

                if let Some(last_sync) = a.last_sync {
                    let elapsed = now.signed_duration_since(last_sync);
                    elapsed.num_minutes() >= a.sync_interval_minutes as i64
                } else {
                    true // Never synced
                }
            })
            .collect()
    }

    /// Validate account configuration
    fn validate_account(&self, account: &EmailAccount) -> Result<(), AccountError> {
        if account.email_address.is_empty() {
            return Err(AccountError::InvalidConfig("Email address is required".to_string()));
        }

        if !account.email_address.contains('@') {
            return Err(AccountError::InvalidConfig("Invalid email address format".to_string()));
        }

        match account.account_type {
            AccountType::Imap => {
                if account.imap_config.is_none() {
                    return Err(AccountError::InvalidConfig("IMAP config required for IMAP accounts".to_string()));
                }
                if account.smtp_config.is_none() {
                    return Err(AccountError::InvalidConfig("SMTP config required for sending".to_string()));
                }
            }
            AccountType::Google | AccountType::Microsoft => {
                if account.oauth_config.is_none() {
                    return Err(AccountError::InvalidConfig("OAuth config required".to_string()));
                }
            }
            _ => {}
        }

        Ok(())
    }
}

impl Default for MultiAccountService {
    fn default() -> Self {
        Self::new()
    }
}

/// Account errors
#[derive(Debug, Clone)]
pub enum AccountError {
    NotFound(Uuid),
    InvalidConfig(String),
    AuthFailed(String),
    ConnectionFailed(String),
    SyncFailed(String),
}

impl std::fmt::Display for AccountError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "Account not found: {}", id),
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            Self::AuthFailed(msg) => write!(f, "Authentication failed: {}", msg),
            Self::ConnectionFailed(msg) => write!(f, "Connection failed: {}", msg),
            Self::SyncFailed(msg) => write!(f, "Sync failed: {}", msg),
        }
    }
}

impl std::error::Error for AccountError {}

/// Account builder for easier creation
pub struct AccountBuilder {
    account: EmailAccount,
}

impl AccountBuilder {
    pub fn new(user_id: Uuid, email: impl Into<String>) -> Self {
        Self {
            account: EmailAccount {
                id: Uuid::new_v4(),
                user_id,
                email_address: email.into(),
                display_name: String::new(),
                account_type: AccountType::Mycelix,
                imap_config: None,
                smtp_config: None,
                oauth_config: None,
                status: ConnectionStatus::Disconnected,
                last_sync: None,
                last_error: None,
                is_primary: false,
                color: None,
                signature: None,
                reply_to: None,
                sync_enabled: true,
                sync_interval_minutes: 5,
                sync_folders: vec!["INBOX".to_string()],
                days_to_sync: 30,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        }
    }

    pub fn display_name(mut self, name: impl Into<String>) -> Self {
        self.account.display_name = name.into();
        self
    }

    pub fn account_type(mut self, account_type: AccountType) -> Self {
        self.account.account_type = account_type;
        self
    }

    pub fn imap(mut self, host: impl Into<String>, port: u16, username: impl Into<String>) -> Self {
        self.account.imap_config = Some(ImapConfig {
            host: host.into(),
            port,
            use_tls: true,
            use_starttls: false,
            username: username.into(),
        });
        self.account.account_type = AccountType::Imap;
        self
    }

    pub fn smtp(mut self, host: impl Into<String>, port: u16, username: impl Into<String>) -> Self {
        self.account.smtp_config = Some(SmtpConfig {
            host: host.into(),
            port,
            use_tls: true,
            use_starttls: false,
            username: username.into(),
            auth_method: SmtpAuthMethod::Plain,
        });
        self
    }

    pub fn google_oauth(mut self, client_id: impl Into<String>) -> Self {
        self.account.oauth_config = Some(OAuth2Config {
            provider: OAuthProvider::Google,
            client_id: client_id.into(),
            scopes: vec![
                "https://mail.google.com/".to_string(),
                "https://www.googleapis.com/auth/gmail.send".to_string(),
            ],
            refresh_token_encrypted: None,
            access_token_encrypted: None,
            token_expiry: None,
        });
        self.account.account_type = AccountType::Google;
        self
    }

    pub fn microsoft_oauth(mut self, client_id: impl Into<String>) -> Self {
        self.account.oauth_config = Some(OAuth2Config {
            provider: OAuthProvider::Microsoft,
            client_id: client_id.into(),
            scopes: vec![
                "https://outlook.office.com/Mail.ReadWrite".to_string(),
                "https://outlook.office.com/Mail.Send".to_string(),
            ],
            refresh_token_encrypted: None,
            access_token_encrypted: None,
            token_expiry: None,
        });
        self.account.account_type = AccountType::Microsoft;
        self
    }

    pub fn as_primary(mut self) -> Self {
        self.account.is_primary = true;
        self
    }

    pub fn color(mut self, color: impl Into<String>) -> Self {
        self.account.color = Some(color.into());
        self
    }

    pub fn signature(mut self, signature: impl Into<String>) -> Self {
        self.account.signature = Some(signature.into());
        self
    }

    pub fn sync_interval(mut self, minutes: u32) -> Self {
        self.account.sync_interval_minutes = minutes;
        self
    }

    pub fn sync_folders(mut self, folders: Vec<String>) -> Self {
        self.account.sync_folders = folders;
        self
    }

    pub fn build(self) -> EmailAccount {
        self.account
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_account_builder() {
        let user_id = Uuid::new_v4();
        let account = AccountBuilder::new(user_id, "user@gmail.com")
            .display_name("Personal Gmail")
            .google_oauth("client_id_123")
            .as_primary()
            .color("#4285F4")
            .build();

        assert_eq!(account.email_address, "user@gmail.com");
        assert_eq!(account.account_type, AccountType::Google);
        assert!(account.is_primary);
        assert!(account.oauth_config.is_some());
    }

    #[test]
    fn test_multi_account_service() {
        let mut service = MultiAccountService::new();
        let user_id = Uuid::new_v4();

        let account1 = AccountBuilder::new(user_id, "work@company.com")
            .display_name("Work")
            .as_primary()
            .build();

        let account2 = AccountBuilder::new(user_id, "personal@gmail.com")
            .display_name("Personal")
            .google_oauth("client_id")
            .build();

        service.add_account(account1).unwrap();
        service.add_account(account2).unwrap();

        assert_eq!(service.get_user_accounts(user_id).len(), 2);
        assert!(service.get_primary_account(user_id).is_some());
    }
}
