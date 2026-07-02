// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Plaid Integration for Bank Account Connectivity
//!
//! Plaid provides secure bank account linking and transaction syncing.
//! This module handles:
//! - Link token creation for Plaid Link UI
//! - Access token exchange after user authentication
//! - Transaction synchronization
//! - Account balance retrieval
//! - Webhook processing for real-time updates
//!
//! # Configuration
//! Required environment variables:
//! - `PLAID_CLIENT_ID`: Your Plaid client ID
//! - `PLAID_SECRET`: Your Plaid secret (use sandbox for testing)
//! - `PLAID_ENV`: "sandbox", "development", or "production"
//! - `PLAID_WEBHOOK_URL`: Your webhook endpoint URL

use super::{client::IntegrationClient, IntegrationError, IntegrationStatus};
use serde::{Deserialize, Serialize};
use std::env;

/// Plaid environment (sandbox, development, production)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlaidEnvironment {
    Sandbox,
    Development,
    Production,
}

impl PlaidEnvironment {
    pub fn base_url(&self) -> &'static str {
        match self {
            PlaidEnvironment::Sandbox => "https://sandbox.plaid.com",
            PlaidEnvironment::Development => "https://development.plaid.com",
            PlaidEnvironment::Production => "https://production.plaid.com",
        }
    }
}

/// Plaid configuration loaded from environment
#[derive(Debug, Clone)]
pub struct PlaidConfig {
    pub client_id: String,
    pub secret: String,
    pub environment: PlaidEnvironment,
    pub webhook_url: Option<String>,
}

impl PlaidConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self, IntegrationError> {
        let client_id = env::var("PLAID_CLIENT_ID")
            .map_err(|_| IntegrationError::ConfigMissing("PLAID_CLIENT_ID".to_string()))?;

        let secret = env::var("PLAID_SECRET")
            .map_err(|_| IntegrationError::ConfigMissing("PLAID_SECRET".to_string()))?;

        let env_str = env::var("PLAID_ENV").unwrap_or_else(|_| "sandbox".to_string());
        let environment = match env_str.to_lowercase().as_str() {
            "sandbox" => PlaidEnvironment::Sandbox,
            "development" => PlaidEnvironment::Development,
            "production" => PlaidEnvironment::Production,
            _ => PlaidEnvironment::Sandbox,
        };

        let webhook_url = env::var("PLAID_WEBHOOK_URL").ok();

        Ok(Self {
            client_id,
            secret,
            environment,
            webhook_url,
        })
    }

    /// Check if Plaid is configured
    pub fn is_configured() -> bool {
        env::var("PLAID_CLIENT_ID").is_ok() && env::var("PLAID_SECRET").is_ok()
    }
}

/// Plaid API client
pub struct PlaidClient {
    config: PlaidConfig,
    http: IntegrationClient,
}

impl PlaidClient {
    /// Create a new Plaid client
    pub fn new() -> Result<Self, IntegrationError> {
        let config = PlaidConfig::from_env()?;
        let http = IntegrationClient::new()?;
        Ok(Self { config, http })
    }

    fn base_url(&self) -> &str {
        self.config.environment.base_url()
    }

    /// Create a link token for Plaid Link UI initialization
    pub async fn create_link_token(
        &self,
        user_id: &str,
        products: Vec<&str>,
    ) -> Result<LinkTokenResponse, IntegrationError> {
        let url = format!("{}/link/token/create", self.base_url());

        let request = LinkTokenRequest {
            client_id: &self.config.client_id,
            secret: &self.config.secret,
            user: PlaidUser {
                client_user_id: user_id,
            },
            client_name: "Mycelix ERP",
            products,
            country_codes: vec!["US"],
            language: "en",
            webhook: self.config.webhook_url.as_deref(),
        };

        tracing::info!(user_id = user_id, "Creating Plaid link token");

        self.http.post(&url, &request, None).await
    }

    /// Exchange a public token for an access token
    pub async fn exchange_public_token(
        &self,
        public_token: &str,
    ) -> Result<AccessTokenResponse, IntegrationError> {
        let url = format!("{}/item/public_token/exchange", self.base_url());

        let request = ExchangeTokenRequest {
            client_id: &self.config.client_id,
            secret: &self.config.secret,
            public_token,
        };

        tracing::info!("Exchanging Plaid public token");

        self.http.post(&url, &request, None).await
    }

    /// Get accounts linked to an access token
    pub async fn get_accounts(
        &self,
        access_token: &str,
    ) -> Result<AccountsResponse, IntegrationError> {
        let url = format!("{}/accounts/get", self.base_url());

        let request = AuthenticatedRequest {
            client_id: &self.config.client_id,
            secret: &self.config.secret,
            access_token,
        };

        tracing::info!("Getting Plaid accounts");

        self.http.post(&url, &request, None).await
    }

    /// Get account balances
    pub async fn get_balances(
        &self,
        access_token: &str,
    ) -> Result<BalancesResponse, IntegrationError> {
        let url = format!("{}/accounts/balance/get", self.base_url());

        let request = AuthenticatedRequest {
            client_id: &self.config.client_id,
            secret: &self.config.secret,
            access_token,
        };

        tracing::info!("Getting Plaid account balances");

        self.http.post(&url, &request, None).await
    }

    /// Sync transactions using the transactions/sync endpoint
    pub async fn sync_transactions(
        &self,
        access_token: &str,
        cursor: Option<&str>,
    ) -> Result<TransactionSyncResponse, IntegrationError> {
        let url = format!("{}/transactions/sync", self.base_url());

        let request = TransactionSyncRequest {
            client_id: &self.config.client_id,
            secret: &self.config.secret,
            access_token,
            cursor,
            count: Some(500),
        };

        tracing::info!(cursor = cursor, "Syncing Plaid transactions");

        self.http.post(&url, &request, None).await
    }

    /// Get item (connection) information
    pub async fn get_item(&self, access_token: &str) -> Result<ItemResponse, IntegrationError> {
        let url = format!("{}/item/get", self.base_url());

        let request = AuthenticatedRequest {
            client_id: &self.config.client_id,
            secret: &self.config.secret,
            access_token,
        };

        self.http.post(&url, &request, None).await
    }

    /// Remove an item (unlink bank account)
    pub async fn remove_item(&self, access_token: &str) -> Result<RemoveItemResponse, IntegrationError> {
        let url = format!("{}/item/remove", self.base_url());

        let request = AuthenticatedRequest {
            client_id: &self.config.client_id,
            secret: &self.config.secret,
            access_token,
        };

        tracing::info!("Removing Plaid item");

        self.http.post(&url, &request, None).await
    }
}

// ============================================================================
// Request Types
// ============================================================================

#[derive(Serialize)]
struct LinkTokenRequest<'a> {
    client_id: &'a str,
    secret: &'a str,
    user: PlaidUser<'a>,
    client_name: &'a str,
    products: Vec<&'a str>,
    country_codes: Vec<&'a str>,
    language: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    webhook: Option<&'a str>,
}

#[derive(Serialize)]
struct PlaidUser<'a> {
    client_user_id: &'a str,
}

#[derive(Serialize)]
struct ExchangeTokenRequest<'a> {
    client_id: &'a str,
    secret: &'a str,
    public_token: &'a str,
}

#[derive(Serialize)]
struct AuthenticatedRequest<'a> {
    client_id: &'a str,
    secret: &'a str,
    access_token: &'a str,
}

#[derive(Serialize)]
struct TransactionSyncRequest<'a> {
    client_id: &'a str,
    secret: &'a str,
    access_token: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    cursor: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    count: Option<u32>,
}

// ============================================================================
// Response Types
// ============================================================================

/// Link token response
#[derive(Debug, Deserialize)]
pub struct LinkTokenResponse {
    pub link_token: String,
    pub expiration: String,
    pub request_id: String,
}

/// Access token response
#[derive(Debug, Deserialize)]
pub struct AccessTokenResponse {
    pub access_token: String,
    pub item_id: String,
    pub request_id: String,
}

/// Accounts response
#[derive(Debug, Deserialize)]
pub struct AccountsResponse {
    pub accounts: Vec<PlaidAccount>,
    pub item: PlaidItem,
    pub request_id: String,
}

/// Balances response
#[derive(Debug, Deserialize)]
pub struct BalancesResponse {
    pub accounts: Vec<PlaidAccount>,
    pub item: PlaidItem,
    pub request_id: String,
}

/// Bank account information from Plaid
#[derive(Debug, Deserialize, Serialize)]
pub struct PlaidAccount {
    pub account_id: String,
    pub name: String,
    pub official_name: Option<String>,
    #[serde(rename = "type")]
    pub account_type: String,
    pub subtype: Option<String>,
    pub mask: Option<String>,
    pub balances: AccountBalances,
}

/// Account balance information
#[derive(Debug, Deserialize, Serialize)]
pub struct AccountBalances {
    pub available: Option<f64>,
    pub current: Option<f64>,
    pub limit: Option<f64>,
    pub iso_currency_code: Option<String>,
}

/// Plaid Item (bank connection)
#[derive(Debug, Deserialize)]
pub struct PlaidItem {
    pub item_id: String,
    pub institution_id: Option<String>,
    pub webhook: Option<String>,
    pub error: Option<PlaidError>,
    pub available_products: Vec<String>,
    pub billed_products: Vec<String>,
    pub consent_expiration_time: Option<String>,
}

/// Plaid error
#[derive(Debug, Deserialize)]
pub struct PlaidError {
    pub error_type: String,
    pub error_code: String,
    pub error_message: String,
    pub display_message: Option<String>,
}

/// Item response
#[derive(Debug, Deserialize)]
pub struct ItemResponse {
    pub item: PlaidItem,
    pub request_id: String,
}

/// Remove item response
#[derive(Debug, Deserialize)]
pub struct RemoveItemResponse {
    pub request_id: String,
}

/// Transaction sync response
#[derive(Debug, Deserialize)]
pub struct TransactionSyncResponse {
    pub added: Vec<PlaidTransaction>,
    pub modified: Vec<PlaidTransaction>,
    pub removed: Vec<RemovedTransaction>,
    pub next_cursor: String,
    pub has_more: bool,
    pub request_id: String,
}

/// Removed transaction reference
#[derive(Debug, Deserialize)]
pub struct RemovedTransaction {
    pub transaction_id: String,
}

/// Transaction from Plaid
#[derive(Debug, Deserialize, Serialize)]
pub struct PlaidTransaction {
    pub transaction_id: String,
    pub account_id: String,
    pub amount: f64,
    pub iso_currency_code: Option<String>,
    pub category: Option<Vec<String>>,
    pub category_id: Option<String>,
    pub merchant_name: Option<String>,
    pub name: String,
    pub date: String,
    pub pending: bool,
    pub payment_channel: String,
    pub transaction_type: Option<String>,
    pub location: Option<TransactionLocation>,
    pub payment_meta: Option<PaymentMeta>,
}

/// Transaction location
#[derive(Debug, Deserialize, Serialize)]
pub struct TransactionLocation {
    pub address: Option<String>,
    pub city: Option<String>,
    pub region: Option<String>,
    pub postal_code: Option<String>,
    pub country: Option<String>,
    pub lat: Option<f64>,
    pub lon: Option<f64>,
    pub store_number: Option<String>,
}

/// Payment metadata
#[derive(Debug, Deserialize, Serialize)]
pub struct PaymentMeta {
    pub reference_number: Option<String>,
    pub ppd_id: Option<String>,
    pub payee: Option<String>,
    pub by_order_of: Option<String>,
    pub payer: Option<String>,
    pub payment_method: Option<String>,
    pub payment_processor: Option<String>,
    pub reason: Option<String>,
}

// ============================================================================
// Webhook Types
// ============================================================================

/// Plaid webhook payload
#[derive(Debug, Deserialize)]
pub struct PlaidWebhook {
    pub webhook_type: String,
    pub webhook_code: String,
    pub item_id: String,
    #[serde(default)]
    pub error: Option<PlaidError>,
    #[serde(default)]
    pub new_transactions: Option<i32>,
    #[serde(default)]
    pub removed_transactions: Option<Vec<String>>,
}

impl PlaidWebhook {
    /// Parse webhook from JSON bytes
    pub fn from_bytes(payload: &[u8]) -> Result<Self, IntegrationError> {
        serde_json::from_slice(payload)
            .map_err(|e| IntegrationError::SerializationError(e.to_string()))
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check integration status
pub async fn check_status() -> IntegrationStatus {
    let configured = PlaidConfig::is_configured();
    IntegrationStatus {
        name: "Plaid".to_string(),
        configured,
        connected: false,
        last_sync: None,
        error: if configured {
            None
        } else {
            Some("Not configured - set PLAID_CLIENT_ID and PLAID_SECRET".to_string())
        },
    }
}

/// Map Plaid account type to internal GL account type
pub fn map_account_type(plaid_type: &str, subtype: Option<&str>) -> &'static str {
    match (plaid_type, subtype) {
        ("depository", Some("checking")) => "ASSET",
        ("depository", Some("savings")) => "ASSET",
        ("credit", _) => "LIABILITY",
        ("loan", _) => "LIABILITY",
        ("investment", _) => "ASSET",
        _ => "ASSET",
    }
}

/// Suggest GL account number for a Plaid transaction based on category
pub fn suggest_gl_account(categories: Option<&[String]>) -> Option<&'static str> {
    let cats = categories?;
    let primary = cats.first()?;

    match primary.as_str() {
        "Food and Drink" => Some("6200"),
        "Travel" => Some("6300"),
        "Shops" => Some("6100"),
        "Transfer" => None,
        "Payment" => None,
        "Recreation" => Some("6400"),
        "Service" => Some("6500"),
        _ => Some("6900"),
    }
}
