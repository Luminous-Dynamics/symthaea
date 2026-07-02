// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Contact Management
//!
//! Manages contacts (people) and companies (accounts).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

// ============================================================================
// Types
// ============================================================================

/// A company/account in the CRM
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Account {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub name: String,
    pub account_type: String, // CUSTOMER, PROSPECT, PARTNER, VENDOR
    pub industry: Option<String>,
    pub website: Option<String>,
    pub phone: Option<String>,
    pub email: Option<String>,
    pub billing_address: Option<String>,
    pub billing_city: Option<String>,
    pub billing_state: Option<String>,
    pub billing_postal_code: Option<String>,
    pub billing_country: Option<String>,
    pub shipping_address: Option<String>,
    pub annual_revenue: Option<rust_decimal::Decimal>,
    pub employee_count: Option<i32>,
    pub owner_id: Option<Uuid>,
    pub parent_account_id: Option<Uuid>,
    pub tags: Vec<String>,
    pub custom_fields: Option<serde_json::Value>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// A person/contact in the CRM
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Contact {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub account_id: Option<Uuid>,
    pub first_name: String,
    pub last_name: String,
    pub email: Option<String>,
    pub phone: Option<String>,
    pub mobile: Option<String>,
    pub job_title: Option<String>,
    pub department: Option<String>,
    pub mailing_address: Option<String>,
    pub mailing_city: Option<String>,
    pub mailing_state: Option<String>,
    pub mailing_postal_code: Option<String>,
    pub mailing_country: Option<String>,
    pub linkedin_url: Option<String>,
    pub twitter_handle: Option<String>,
    pub lead_source: Option<String>,
    pub owner_id: Option<Uuid>,
    pub tags: Vec<String>,
    pub custom_fields: Option<serde_json::Value>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Create account request
#[derive(Debug, Deserialize)]
pub struct CreateAccountRequest {
    pub name: String,
    pub account_type: String,
    pub industry: Option<String>,
    pub website: Option<String>,
    pub phone: Option<String>,
    pub email: Option<String>,
    pub billing_address: Option<String>,
    pub billing_city: Option<String>,
    pub billing_state: Option<String>,
    pub billing_postal_code: Option<String>,
    pub billing_country: Option<String>,
    pub owner_id: Option<Uuid>,
    pub tags: Option<Vec<String>>,
}

/// Create contact request
#[derive(Debug, Deserialize)]
pub struct CreateContactRequest {
    pub account_id: Option<Uuid>,
    pub first_name: String,
    pub last_name: String,
    pub email: Option<String>,
    pub phone: Option<String>,
    pub mobile: Option<String>,
    pub job_title: Option<String>,
    pub department: Option<String>,
    pub lead_source: Option<String>,
    pub owner_id: Option<Uuid>,
    pub tags: Option<Vec<String>>,
}

// ============================================================================
// Service
// ============================================================================

/// Contact Management Service
#[derive(Clone)]
pub struct ContactService {
    pool: PgPool,
}

impl ContactService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    // ========================================================================
    // Accounts
    // ========================================================================

    /// Create a new account
    pub async fn create_account(
        &self,
        tenant_id: Uuid,
        req: CreateAccountRequest,
    ) -> Result<Account, CrmError> {
        let id = Uuid::new_v4();
        let tags = req.tags.unwrap_or_default();

        sqlx::query(
            "INSERT INTO crm_accounts (
                id, tenant_id, name, account_type, industry, website, phone, email,
                billing_address, billing_city, billing_state, billing_postal_code,
                billing_country, owner_id, tags
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)",
        )
        .bind(id)
        .bind(tenant_id)
        .bind(&req.name)
        .bind(&req.account_type)
        .bind(&req.industry)
        .bind(&req.website)
        .bind(&req.phone)
        .bind(&req.email)
        .bind(&req.billing_address)
        .bind(&req.billing_city)
        .bind(&req.billing_state)
        .bind(&req.billing_postal_code)
        .bind(&req.billing_country)
        .bind(req.owner_id)
        .bind(&tags)
        .execute(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        self.get_account(id).await
    }

    /// Get an account by ID
    pub async fn get_account(&self, id: Uuid) -> Result<Account, CrmError> {
        sqlx::query_as::<_, Account>("SELECT * FROM crm_accounts WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(CrmError::Database)?
            .ok_or_else(|| CrmError::NotFound("Account not found".into()))
    }

    /// List accounts for a tenant
    pub async fn list_accounts(
        &self,
        tenant_id: Uuid,
        account_type: Option<&str>,
        search: Option<&str>,
        limit: i32,
        offset: i32,
    ) -> Result<Vec<Account>, CrmError> {
        let accounts = if let Some(atype) = account_type {
            if let Some(q) = search {
                let pattern = format!("%{}%", q);
                sqlx::query_as::<_, Account>(
                    "SELECT * FROM crm_accounts
                     WHERE tenant_id = $1 AND account_type = $2
                       AND (name ILIKE $3 OR email ILIKE $3)
                     ORDER BY name LIMIT $4 OFFSET $5",
                )
                .bind(tenant_id)
                .bind(atype)
                .bind(&pattern)
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            } else {
                sqlx::query_as::<_, Account>(
                    "SELECT * FROM crm_accounts
                     WHERE tenant_id = $1 AND account_type = $2
                     ORDER BY name LIMIT $3 OFFSET $4",
                )
                .bind(tenant_id)
                .bind(atype)
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            }
        } else if let Some(q) = search {
            let pattern = format!("%{}%", q);
            sqlx::query_as::<_, Account>(
                "SELECT * FROM crm_accounts
                 WHERE tenant_id = $1 AND (name ILIKE $2 OR email ILIKE $2)
                 ORDER BY name LIMIT $3 OFFSET $4",
            )
            .bind(tenant_id)
            .bind(&pattern)
            .bind(limit)
            .bind(offset)
            .fetch_all(&self.pool)
            .await
        } else {
            sqlx::query_as::<_, Account>(
                "SELECT * FROM crm_accounts
                 WHERE tenant_id = $1
                 ORDER BY name LIMIT $2 OFFSET $3",
            )
            .bind(tenant_id)
            .bind(limit)
            .bind(offset)
            .fetch_all(&self.pool)
            .await
        }
        .map_err(CrmError::Database)?;

        Ok(accounts)
    }

    /// Update an account
    pub async fn update_account(
        &self,
        id: Uuid,
        name: Option<&str>,
        industry: Option<&str>,
        website: Option<&str>,
        phone: Option<&str>,
        email: Option<&str>,
    ) -> Result<Account, CrmError> {
        sqlx::query(
            "UPDATE crm_accounts SET
                name = COALESCE($2, name),
                industry = COALESCE($3, industry),
                website = COALESCE($4, website),
                phone = COALESCE($5, phone),
                email = COALESCE($6, email),
                updated_at = NOW()
             WHERE id = $1",
        )
        .bind(id)
        .bind(name)
        .bind(industry)
        .bind(website)
        .bind(phone)
        .bind(email)
        .execute(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        self.get_account(id).await
    }

    // ========================================================================
    // Contacts
    // ========================================================================

    /// Create a new contact
    pub async fn create_contact(
        &self,
        tenant_id: Uuid,
        req: CreateContactRequest,
    ) -> Result<Contact, CrmError> {
        let id = Uuid::new_v4();
        let tags = req.tags.unwrap_or_default();

        sqlx::query(
            "INSERT INTO crm_contacts (
                id, tenant_id, account_id, first_name, last_name, email, phone,
                mobile, job_title, department, lead_source, owner_id, tags
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)",
        )
        .bind(id)
        .bind(tenant_id)
        .bind(req.account_id)
        .bind(&req.first_name)
        .bind(&req.last_name)
        .bind(&req.email)
        .bind(&req.phone)
        .bind(&req.mobile)
        .bind(&req.job_title)
        .bind(&req.department)
        .bind(&req.lead_source)
        .bind(req.owner_id)
        .bind(&tags)
        .execute(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        self.get_contact(id).await
    }

    /// Get a contact by ID
    pub async fn get_contact(&self, id: Uuid) -> Result<Contact, CrmError> {
        sqlx::query_as::<_, Contact>("SELECT * FROM crm_contacts WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(CrmError::Database)?
            .ok_or_else(|| CrmError::NotFound("Contact not found".into()))
    }

    /// List contacts for a tenant
    pub async fn list_contacts(
        &self,
        tenant_id: Uuid,
        account_id: Option<Uuid>,
        search: Option<&str>,
        limit: i32,
        offset: i32,
    ) -> Result<Vec<Contact>, CrmError> {
        let contacts = if let Some(aid) = account_id {
            sqlx::query_as::<_, Contact>(
                "SELECT * FROM crm_contacts
                 WHERE tenant_id = $1 AND account_id = $2
                 ORDER BY last_name, first_name LIMIT $3 OFFSET $4",
            )
            .bind(tenant_id)
            .bind(aid)
            .bind(limit)
            .bind(offset)
            .fetch_all(&self.pool)
            .await
        } else if let Some(q) = search {
            let pattern = format!("%{}%", q);
            sqlx::query_as::<_, Contact>(
                "SELECT * FROM crm_contacts
                 WHERE tenant_id = $1
                   AND (first_name ILIKE $2 OR last_name ILIKE $2 OR email ILIKE $2)
                 ORDER BY last_name, first_name LIMIT $3 OFFSET $4",
            )
            .bind(tenant_id)
            .bind(&pattern)
            .bind(limit)
            .bind(offset)
            .fetch_all(&self.pool)
            .await
        } else {
            sqlx::query_as::<_, Contact>(
                "SELECT * FROM crm_contacts
                 WHERE tenant_id = $1
                 ORDER BY last_name, first_name LIMIT $2 OFFSET $3",
            )
            .bind(tenant_id)
            .bind(limit)
            .bind(offset)
            .fetch_all(&self.pool)
            .await
        }
        .map_err(CrmError::Database)?;

        Ok(contacts)
    }

    /// Get contacts for an account
    pub async fn get_account_contacts(&self, account_id: Uuid) -> Result<Vec<Contact>, CrmError> {
        let contacts = sqlx::query_as::<_, Contact>(
            "SELECT * FROM crm_contacts WHERE account_id = $1 ORDER BY last_name, first_name",
        )
        .bind(account_id)
        .fetch_all(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        Ok(contacts)
    }

    /// Search contacts globally
    pub async fn search_contacts(
        &self,
        tenant_id: Uuid,
        query: &str,
        limit: i32,
    ) -> Result<Vec<Contact>, CrmError> {
        let pattern = format!("%{}%", query);
        let contacts = sqlx::query_as::<_, Contact>(
            "SELECT * FROM crm_contacts
             WHERE tenant_id = $1
               AND (first_name ILIKE $2 OR last_name ILIKE $2 OR email ILIKE $2 OR phone ILIKE $2)
             ORDER BY last_name, first_name
             LIMIT $3",
        )
        .bind(tenant_id)
        .bind(&pattern)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        Ok(contacts)
    }
}

// ============================================================================
// Errors
// ============================================================================

/// CRM errors
#[derive(Debug, thiserror::Error)]
pub enum CrmError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Validation error: {0}")]
    Validation(String),
}
