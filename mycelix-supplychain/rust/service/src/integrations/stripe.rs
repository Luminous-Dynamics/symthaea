// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Stripe Integration for Payment Processing
//!
//! Stripe provides payment processing, invoicing, and subscription management.
//! This module handles:
//! - Payment intent creation for one-time payments
//! - Subscription management for recurring billing
//! - Invoice generation and tracking
//! - Webhook processing for payment events
//! - Customer management
//!
//! # Configuration
//! Required environment variables:
//! - `STRIPE_SECRET_KEY`: Your Stripe secret key (sk_test_... or sk_live_...)
//! - `STRIPE_WEBHOOK_SECRET`: Webhook endpoint signing secret (whsec_...)

use super::{client::IntegrationClient, IntegrationError, IntegrationStatus};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;

/// Stripe configuration loaded from environment
#[derive(Debug, Clone)]
pub struct StripeConfig {
    pub secret_key: String,
    pub webhook_secret: Option<String>,
}

impl StripeConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self, IntegrationError> {
        let secret_key = env::var("STRIPE_SECRET_KEY")
            .map_err(|_| IntegrationError::ConfigMissing("STRIPE_SECRET_KEY".to_string()))?;

        let webhook_secret = env::var("STRIPE_WEBHOOK_SECRET").ok();

        Ok(Self {
            secret_key,
            webhook_secret,
        })
    }

    /// Check if Stripe is configured
    pub fn is_configured() -> bool {
        env::var("STRIPE_SECRET_KEY").is_ok()
    }

    /// Check if this is a test/sandbox environment
    pub fn is_test_mode(&self) -> bool {
        self.secret_key.starts_with("sk_test_")
    }
}

const STRIPE_API_BASE: &str = "https://api.stripe.com/v1";

/// Stripe API client
pub struct StripeClient {
    config: StripeConfig,
    http: IntegrationClient,
}

impl StripeClient {
    /// Create a new Stripe client
    pub fn new() -> Result<Self, IntegrationError> {
        let config = StripeConfig::from_env()?;
        let http = IntegrationClient::new()?;
        Ok(Self { config, http })
    }

    fn auth_header(&self) -> (&str, String) {
        ("Authorization", format!("Bearer {}", self.config.secret_key))
    }

    /// Create a customer
    pub async fn create_customer(
        &self,
        email: Option<&str>,
        name: Option<&str>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<Customer, IntegrationError> {
        let url = format!("{}/customers", STRIPE_API_BASE);

        let mut form: HashMap<String, String> = HashMap::new();
        if let Some(e) = email {
            form.insert("email".to_string(), e.to_string());
        }
        if let Some(n) = name {
            form.insert("name".to_string(), n.to_string());
        }
        if let Some(meta) = metadata {
            for (k, v) in meta {
                form.insert(format!("metadata[{}]", k), v);
            }
        }

        tracing::info!(email = email, name = name, "Creating Stripe customer");

        self.post_form(&url, &form).await
    }

    /// Get a customer by ID
    pub async fn get_customer(&self, customer_id: &str) -> Result<Customer, IntegrationError> {
        let url = format!("{}/customers/{}", STRIPE_API_BASE, customer_id);
        let (key, value) = self.auth_header();
        self.http.get(&url, Some((key, &value))).await
    }

    /// Create a payment intent
    pub async fn create_payment_intent(
        &self,
        amount: i64,
        currency: &str,
        customer_id: Option<&str>,
        description: Option<&str>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<PaymentIntent, IntegrationError> {
        let url = format!("{}/payment_intents", STRIPE_API_BASE);

        let mut form: HashMap<String, String> = HashMap::new();
        form.insert("amount".to_string(), amount.to_string());
        form.insert("currency".to_string(), currency.to_lowercase());

        if let Some(cust) = customer_id {
            form.insert("customer".to_string(), cust.to_string());
        }
        if let Some(desc) = description {
            form.insert("description".to_string(), desc.to_string());
        }
        if let Some(meta) = metadata {
            for (k, v) in meta {
                form.insert(format!("metadata[{}]", k), v);
            }
        }

        tracing::info!(
            amount = amount,
            currency = currency,
            "Creating Stripe payment intent"
        );

        self.post_form(&url, &form).await
    }

    /// Confirm a payment intent
    pub async fn confirm_payment_intent(
        &self,
        payment_intent_id: &str,
        payment_method: &str,
    ) -> Result<PaymentIntent, IntegrationError> {
        let url = format!(
            "{}/payment_intents/{}/confirm",
            STRIPE_API_BASE, payment_intent_id
        );

        let mut form: HashMap<String, String> = HashMap::new();
        form.insert("payment_method".to_string(), payment_method.to_string());

        tracing::info!(payment_intent_id = payment_intent_id, "Confirming payment intent");

        self.post_form(&url, &form).await
    }

    /// Create an invoice
    pub async fn create_invoice(
        &self,
        customer_id: &str,
        auto_advance: bool,
        collection_method: &str,
        days_until_due: Option<u32>,
    ) -> Result<Invoice, IntegrationError> {
        let url = format!("{}/invoices", STRIPE_API_BASE);

        let mut form: HashMap<String, String> = HashMap::new();
        form.insert("customer".to_string(), customer_id.to_string());
        form.insert("auto_advance".to_string(), auto_advance.to_string());
        form.insert("collection_method".to_string(), collection_method.to_string());

        if let Some(days) = days_until_due {
            form.insert("days_until_due".to_string(), days.to_string());
        }

        tracing::info!(customer_id = customer_id, "Creating Stripe invoice");

        self.post_form(&url, &form).await
    }

    /// Add an invoice item
    pub async fn create_invoice_item(
        &self,
        customer_id: &str,
        invoice_id: Option<&str>,
        amount: i64,
        currency: &str,
        description: &str,
    ) -> Result<InvoiceItem, IntegrationError> {
        let url = format!("{}/invoiceitems", STRIPE_API_BASE);

        let mut form: HashMap<String, String> = HashMap::new();
        form.insert("customer".to_string(), customer_id.to_string());
        form.insert("amount".to_string(), amount.to_string());
        form.insert("currency".to_string(), currency.to_lowercase());
        form.insert("description".to_string(), description.to_string());

        if let Some(inv) = invoice_id {
            form.insert("invoice".to_string(), inv.to_string());
        }

        tracing::info!(
            customer_id = customer_id,
            amount = amount,
            "Creating Stripe invoice item"
        );

        self.post_form(&url, &form).await
    }

    /// Finalize an invoice
    pub async fn finalize_invoice(&self, invoice_id: &str) -> Result<Invoice, IntegrationError> {
        let url = format!("{}/invoices/{}/finalize", STRIPE_API_BASE, invoice_id);
        let form: HashMap<String, String> = HashMap::new();

        tracing::info!(invoice_id = invoice_id, "Finalizing Stripe invoice");

        self.post_form(&url, &form).await
    }

    /// Send an invoice
    pub async fn send_invoice(&self, invoice_id: &str) -> Result<Invoice, IntegrationError> {
        let url = format!("{}/invoices/{}/send", STRIPE_API_BASE, invoice_id);
        let form: HashMap<String, String> = HashMap::new();

        tracing::info!(invoice_id = invoice_id, "Sending Stripe invoice");

        self.post_form(&url, &form).await
    }

    /// Create a subscription
    pub async fn create_subscription(
        &self,
        customer_id: &str,
        price_id: &str,
        quantity: Option<u32>,
    ) -> Result<Subscription, IntegrationError> {
        let url = format!("{}/subscriptions", STRIPE_API_BASE);

        let mut form: HashMap<String, String> = HashMap::new();
        form.insert("customer".to_string(), customer_id.to_string());
        form.insert("items[0][price]".to_string(), price_id.to_string());

        if let Some(qty) = quantity {
            form.insert("items[0][quantity]".to_string(), qty.to_string());
        }

        tracing::info!(
            customer_id = customer_id,
            price_id = price_id,
            "Creating Stripe subscription"
        );

        self.post_form(&url, &form).await
    }

    /// Cancel a subscription
    pub async fn cancel_subscription(
        &self,
        subscription_id: &str,
        at_period_end: bool,
    ) -> Result<Subscription, IntegrationError> {
        if at_period_end {
            let url = format!("{}/subscriptions/{}", STRIPE_API_BASE, subscription_id);
            let mut form: HashMap<String, String> = HashMap::new();
            form.insert("cancel_at_period_end".to_string(), "true".to_string());

            tracing::info!(
                subscription_id = subscription_id,
                "Canceling Stripe subscription at period end"
            );

            self.post_form(&url, &form).await
        } else {
            let url = format!("{}/subscriptions/{}", STRIPE_API_BASE, subscription_id);
            let (key, value) = self.auth_header();

            tracing::info!(
                subscription_id = subscription_id,
                "Canceling Stripe subscription immediately"
            );

            self.http.delete(&url, Some((key, &value))).await
        }
    }

    /// Create a refund
    pub async fn create_refund(
        &self,
        payment_intent_id: &str,
        amount: Option<i64>,
        reason: Option<&str>,
    ) -> Result<Refund, IntegrationError> {
        let url = format!("{}/refunds", STRIPE_API_BASE);

        let mut form: HashMap<String, String> = HashMap::new();
        form.insert("payment_intent".to_string(), payment_intent_id.to_string());

        if let Some(amt) = amount {
            form.insert("amount".to_string(), amt.to_string());
        }
        if let Some(r) = reason {
            form.insert("reason".to_string(), r.to_string());
        }

        tracing::info!(
            payment_intent_id = payment_intent_id,
            amount = amount,
            "Creating Stripe refund"
        );

        self.post_form(&url, &form).await
    }

    /// List charges for a customer
    pub async fn list_charges(
        &self,
        customer_id: Option<&str>,
        limit: Option<u32>,
    ) -> Result<ChargeList, IntegrationError> {
        let mut url = format!("{}/charges?", STRIPE_API_BASE);

        if let Some(cust) = customer_id {
            url.push_str(&format!("customer={}&", cust));
        }
        if let Some(l) = limit {
            url.push_str(&format!("limit={}&", l));
        }

        let (key, value) = self.auth_header();
        self.http.get(&url, Some((key, &value))).await
    }

    /// Helper to POST form-encoded data
    async fn post_form<T: serde::de::DeserializeOwned>(
        &self,
        url: &str,
        form: &HashMap<String, String>,
    ) -> Result<T, IntegrationError> {
        let client = reqwest::Client::new();
        let response = client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.config.secret_key))
            .form(form)
            .send()
            .await
            .map_err(|e| IntegrationError::NetworkError(e.to_string()))?;

        let status = response.status();
        if status.is_success() {
            response
                .json::<T>()
                .await
                .map_err(|e| IntegrationError::SerializationError(e.to_string()))
        } else {
            let error_body = response.text().await.unwrap_or_default();
            Err(IntegrationError::ApiError(format!(
                "Stripe API error ({}): {}",
                status, error_body
            )))
        }
    }
}

// ============================================================================
// Response Types
// ============================================================================

/// Stripe Customer
#[derive(Debug, Deserialize, Serialize)]
pub struct Customer {
    pub id: String,
    pub object: String,
    pub email: Option<String>,
    pub name: Option<String>,
    pub phone: Option<String>,
    pub created: i64,
    pub livemode: bool,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Payment Intent
#[derive(Debug, Deserialize, Serialize)]
pub struct PaymentIntent {
    pub id: String,
    pub object: String,
    pub amount: i64,
    pub currency: String,
    pub status: String,
    pub client_secret: Option<String>,
    pub customer: Option<String>,
    pub description: Option<String>,
    pub created: i64,
    pub livemode: bool,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Stripe Invoice
#[derive(Debug, Deserialize, Serialize)]
pub struct Invoice {
    pub id: String,
    pub object: String,
    pub customer: String,
    pub status: Option<String>,
    pub amount_due: i64,
    pub amount_paid: i64,
    pub amount_remaining: i64,
    pub currency: String,
    pub due_date: Option<i64>,
    pub hosted_invoice_url: Option<String>,
    pub invoice_pdf: Option<String>,
    pub number: Option<String>,
    pub created: i64,
    pub livemode: bool,
}

/// Invoice Item
#[derive(Debug, Deserialize, Serialize)]
pub struct InvoiceItem {
    pub id: String,
    pub object: String,
    pub amount: i64,
    pub currency: String,
    pub customer: String,
    pub description: Option<String>,
    pub invoice: Option<String>,
    pub livemode: bool,
}

/// Subscription
#[derive(Debug, Deserialize, Serialize)]
pub struct Subscription {
    pub id: String,
    pub object: String,
    pub customer: String,
    pub status: String,
    pub current_period_start: i64,
    pub current_period_end: i64,
    pub cancel_at_period_end: bool,
    pub created: i64,
    pub livemode: bool,
}

/// Refund
#[derive(Debug, Deserialize, Serialize)]
pub struct Refund {
    pub id: String,
    pub object: String,
    pub amount: i64,
    pub currency: String,
    pub payment_intent: Option<String>,
    pub status: String,
    pub reason: Option<String>,
    pub created: i64,
}

/// Charge
#[derive(Debug, Deserialize, Serialize)]
pub struct Charge {
    pub id: String,
    pub object: String,
    pub amount: i64,
    pub currency: String,
    pub customer: Option<String>,
    pub description: Option<String>,
    pub status: String,
    pub paid: bool,
    pub refunded: bool,
    pub created: i64,
}

/// Charge list response
#[derive(Debug, Deserialize)]
pub struct ChargeList {
    pub object: String,
    pub data: Vec<Charge>,
    pub has_more: bool,
    pub url: String,
}

// ============================================================================
// Webhook Types
// ============================================================================

/// Stripe webhook event
#[derive(Debug, Deserialize)]
pub struct StripeWebhookEvent {
    pub id: String,
    #[serde(rename = "type")]
    pub event_type: String,
    pub data: WebhookEventData,
    pub created: i64,
    pub livemode: bool,
}

#[derive(Debug, Deserialize)]
pub struct WebhookEventData {
    pub object: serde_json::Value,
}

impl StripeWebhookEvent {
    /// Parse webhook from JSON bytes
    pub fn from_bytes(payload: &[u8]) -> Result<Self, IntegrationError> {
        serde_json::from_slice(payload)
            .map_err(|e| IntegrationError::SerializationError(e.to_string()))
    }

    /// Verify webhook signature (HMAC-SHA256)
    pub fn verify_signature(
        payload: &[u8],
        signature: &str,
        secret: &str,
    ) -> Result<bool, IntegrationError> {
        use sha2::Sha256;
        use std::time::{SystemTime, UNIX_EPOCH};

        // Parse the signature header
        let parts: HashMap<&str, &str> = signature
            .split(',')
            .filter_map(|part| {
                let mut split = part.split('=');
                Some((split.next()?, split.next()?))
            })
            .collect();

        let timestamp = parts
            .get("t")
            .ok_or_else(|| IntegrationError::WebhookVerificationFailed)?
            .parse::<i64>()
            .map_err(|_| IntegrationError::WebhookVerificationFailed)?;

        let expected_sig = parts
            .get("v1")
            .ok_or(IntegrationError::WebhookVerificationFailed)?;

        // Check timestamp is within 5 minutes
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        if (now - timestamp).abs() > 300 {
            return Ok(false);
        }

        // Compute expected signature
        let signed_payload = format!(
            "{}.{}",
            timestamp,
            String::from_utf8_lossy(payload)
        );

        // Use HMAC-SHA256
        use hmac::{Hmac, Mac};
        type HmacSha256 = Hmac<Sha256>;

        let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
            .map_err(|_| IntegrationError::WebhookVerificationFailed)?;
        mac.update(signed_payload.as_bytes());

        let result = mac.finalize();
        let computed = hex::encode(result.into_bytes());

        Ok(&computed == *expected_sig)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check integration status
pub async fn check_status() -> IntegrationStatus {
    let configured = StripeConfig::is_configured();
    let is_test = StripeConfig::from_env()
        .map(|c| c.is_test_mode())
        .unwrap_or(true);

    IntegrationStatus {
        name: format!("Stripe ({})", if is_test { "test" } else { "live" }),
        configured,
        connected: false,
        last_sync: None,
        error: if configured {
            None
        } else {
            Some("Not configured - set STRIPE_SECRET_KEY".to_string())
        },
    }
}

/// Convert a Decimal amount to Stripe's integer cents format
pub fn decimal_to_cents(amount: Decimal) -> i64 {
    (amount * Decimal::from(100))
        .round()
        .to_string()
        .parse()
        .unwrap_or(0)
}

/// Convert Stripe's integer cents to Decimal
pub fn cents_to_decimal(cents: i64) -> Decimal {
    Decimal::new(cents, 2)
}

/// Map Stripe invoice status to internal status
pub fn map_invoice_status(stripe_status: &str) -> &'static str {
    match stripe_status {
        "draft" => "DRAFT",
        "open" => "SENT",
        "paid" => "PAID",
        "uncollectible" => "OVERDUE",
        "void" => "CANCELLED",
        _ => "DRAFT",
    }
}

/// Map Stripe payment intent status to internal status
pub fn map_payment_status(stripe_status: &str) -> &'static str {
    match stripe_status {
        "succeeded" => "COMPLETED",
        "canceled" => "CANCELLED",
        "processing" => "PENDING",
        "requires_payment_method" => "PENDING",
        "requires_confirmation" => "PENDING",
        "requires_action" => "PENDING",
        _ => "PENDING",
    }
}
