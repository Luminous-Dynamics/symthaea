// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lead Management
//!
//! Track and score potential customers through the sales funnel.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

use super::CrmError;

// ============================================================================
// Types
// ============================================================================

/// A sales lead
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Lead {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub first_name: String,
    pub last_name: String,
    pub email: Option<String>,
    pub phone: Option<String>,
    pub company: Option<String>,
    pub job_title: Option<String>,
    pub lead_source: Option<String>,
    pub lead_status: String, // NEW, CONTACTED, QUALIFIED, UNQUALIFIED, CONVERTED
    pub lead_score: i32,
    pub rating: Option<String>, // HOT, WARM, COLD
    pub industry: Option<String>,
    pub annual_revenue: Option<Decimal>,
    pub employee_count: Option<i32>,
    pub website: Option<String>,
    pub address: Option<String>,
    pub city: Option<String>,
    pub state: Option<String>,
    pub postal_code: Option<String>,
    pub country: Option<String>,
    pub description: Option<String>,
    pub owner_id: Option<Uuid>,
    pub converted_account_id: Option<Uuid>,
    pub converted_contact_id: Option<Uuid>,
    pub converted_opportunity_id: Option<Uuid>,
    pub converted_at: Option<DateTime<Utc>>,
    pub tags: Vec<String>,
    pub custom_fields: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Lead status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeadStatus {
    New,
    Contacted,
    Qualified,
    Unqualified,
    Converted,
}

impl LeadStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            LeadStatus::New => "NEW",
            LeadStatus::Contacted => "CONTACTED",
            LeadStatus::Qualified => "QUALIFIED",
            LeadStatus::Unqualified => "UNQUALIFIED",
            LeadStatus::Converted => "CONVERTED",
        }
    }
}

/// Create lead request
#[derive(Debug, Deserialize)]
pub struct CreateLeadRequest {
    pub first_name: String,
    pub last_name: String,
    pub email: Option<String>,
    pub phone: Option<String>,
    pub company: Option<String>,
    pub job_title: Option<String>,
    pub lead_source: Option<String>,
    pub industry: Option<String>,
    pub website: Option<String>,
    pub description: Option<String>,
    pub owner_id: Option<Uuid>,
    pub tags: Option<Vec<String>>,
}

/// Lead conversion result
#[derive(Debug, Serialize)]
pub struct LeadConversionResult {
    pub lead_id: Uuid,
    pub account_id: Uuid,
    pub contact_id: Uuid,
    pub opportunity_id: Option<Uuid>,
}

// ============================================================================
// Service
// ============================================================================

/// Lead Management Service
#[derive(Clone)]
pub struct LeadService {
    pool: PgPool,
}

impl LeadService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a new lead
    pub async fn create_lead(
        &self,
        tenant_id: Uuid,
        req: CreateLeadRequest,
    ) -> Result<Lead, CrmError> {
        let id = Uuid::new_v4();
        let tags = req.tags.unwrap_or_default();

        sqlx::query(
            "INSERT INTO crm_leads (
                id, tenant_id, first_name, last_name, email, phone, company,
                job_title, lead_source, industry, website, description, owner_id, tags
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)",
        )
        .bind(id)
        .bind(tenant_id)
        .bind(&req.first_name)
        .bind(&req.last_name)
        .bind(&req.email)
        .bind(&req.phone)
        .bind(&req.company)
        .bind(&req.job_title)
        .bind(&req.lead_source)
        .bind(&req.industry)
        .bind(&req.website)
        .bind(&req.description)
        .bind(req.owner_id)
        .bind(&tags)
        .execute(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        self.get_lead(id).await
    }

    /// Get a lead by ID
    pub async fn get_lead(&self, id: Uuid) -> Result<Lead, CrmError> {
        sqlx::query_as::<_, Lead>("SELECT * FROM crm_leads WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(CrmError::Database)?
            .ok_or_else(|| CrmError::NotFound("Lead not found".into()))
    }

    /// List leads for a tenant
    pub async fn list_leads(
        &self,
        tenant_id: Uuid,
        status: Option<&str>,
        owner_id: Option<Uuid>,
        limit: i32,
        offset: i32,
    ) -> Result<Vec<Lead>, CrmError> {
        let leads = match (status, owner_id) {
            (Some(s), Some(o)) => {
                sqlx::query_as::<_, Lead>(
                    "SELECT * FROM crm_leads
                     WHERE tenant_id = $1 AND lead_status = $2 AND owner_id = $3
                     ORDER BY created_at DESC LIMIT $4 OFFSET $5",
                )
                .bind(tenant_id)
                .bind(s)
                .bind(o)
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            }
            (Some(s), None) => {
                sqlx::query_as::<_, Lead>(
                    "SELECT * FROM crm_leads
                     WHERE tenant_id = $1 AND lead_status = $2
                     ORDER BY created_at DESC LIMIT $3 OFFSET $4",
                )
                .bind(tenant_id)
                .bind(s)
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            }
            (None, Some(o)) => {
                sqlx::query_as::<_, Lead>(
                    "SELECT * FROM crm_leads
                     WHERE tenant_id = $1 AND owner_id = $2
                     ORDER BY created_at DESC LIMIT $3 OFFSET $4",
                )
                .bind(tenant_id)
                .bind(o)
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            }
            (None, None) => {
                sqlx::query_as::<_, Lead>(
                    "SELECT * FROM crm_leads
                     WHERE tenant_id = $1
                     ORDER BY created_at DESC LIMIT $2 OFFSET $3",
                )
                .bind(tenant_id)
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            }
        }
        .map_err(CrmError::Database)?;

        Ok(leads)
    }

    /// Update lead status
    pub async fn update_status(
        &self,
        id: Uuid,
        status: LeadStatus,
    ) -> Result<Lead, CrmError> {
        sqlx::query(
            "UPDATE crm_leads SET lead_status = $2, updated_at = NOW() WHERE id = $1",
        )
        .bind(id)
        .bind(status.as_str())
        .execute(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        self.get_lead(id).await
    }

    /// Update lead score
    pub async fn update_score(&self, id: Uuid, score: i32) -> Result<Lead, CrmError> {
        let rating = if score >= 80 {
            "HOT"
        } else if score >= 50 {
            "WARM"
        } else {
            "COLD"
        };

        sqlx::query(
            "UPDATE crm_leads SET lead_score = $2, rating = $3, updated_at = NOW() WHERE id = $1",
        )
        .bind(id)
        .bind(score)
        .bind(rating)
        .execute(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        self.get_lead(id).await
    }

    /// Convert lead to account/contact/opportunity
    pub async fn convert_lead(
        &self,
        id: Uuid,
        create_opportunity: bool,
        opportunity_name: Option<&str>,
        opportunity_amount: Option<Decimal>,
    ) -> Result<LeadConversionResult, CrmError> {
        let lead = self.get_lead(id).await?;

        // Create account
        let account_id = Uuid::new_v4();
        sqlx::query(
            "INSERT INTO crm_accounts (id, tenant_id, name, account_type, industry, website, phone)
             VALUES ($1, $2, $3, 'CUSTOMER', $4, $5, $6)",
        )
        .bind(account_id)
        .bind(lead.tenant_id)
        .bind(lead.company.as_deref().unwrap_or(&format!("{} {}", lead.first_name, lead.last_name)))
        .bind(&lead.industry)
        .bind(&lead.website)
        .bind(&lead.phone)
        .execute(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        // Create contact
        let contact_id = Uuid::new_v4();
        sqlx::query(
            "INSERT INTO crm_contacts (id, tenant_id, account_id, first_name, last_name, email, phone, job_title, lead_source)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
        )
        .bind(contact_id)
        .bind(lead.tenant_id)
        .bind(account_id)
        .bind(&lead.first_name)
        .bind(&lead.last_name)
        .bind(&lead.email)
        .bind(&lead.phone)
        .bind(&lead.job_title)
        .bind(&lead.lead_source)
        .execute(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        // Create opportunity if requested
        let opportunity_id = if create_opportunity {
            let opp_id = Uuid::new_v4();
            let opp_name = opportunity_name.unwrap_or("New Opportunity");
            sqlx::query(
                "INSERT INTO crm_opportunities (id, tenant_id, account_id, name, amount, stage, owner_id)
                 VALUES ($1, $2, $3, $4, $5, 'PROSPECTING', $6)",
            )
            .bind(opp_id)
            .bind(lead.tenant_id)
            .bind(account_id)
            .bind(opp_name)
            .bind(opportunity_amount)
            .bind(lead.owner_id)
            .execute(&self.pool)
            .await
            .map_err(CrmError::Database)?;
            Some(opp_id)
        } else {
            None
        };

        // Update lead as converted
        sqlx::query(
            "UPDATE crm_leads SET
                lead_status = 'CONVERTED',
                converted_account_id = $2,
                converted_contact_id = $3,
                converted_opportunity_id = $4,
                converted_at = NOW(),
                updated_at = NOW()
             WHERE id = $1",
        )
        .bind(id)
        .bind(account_id)
        .bind(contact_id)
        .bind(opportunity_id)
        .execute(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        Ok(LeadConversionResult {
            lead_id: id,
            account_id,
            contact_id,
            opportunity_id,
        })
    }

    /// Get lead statistics
    pub async fn get_stats(&self, tenant_id: Uuid) -> Result<LeadStats, CrmError> {
        let (total, new, contacted, qualified, converted): (i64, i64, i64, i64, i64) =
            sqlx::query_as(
                "SELECT
                    COUNT(*),
                    COUNT(*) FILTER (WHERE lead_status = 'NEW'),
                    COUNT(*) FILTER (WHERE lead_status = 'CONTACTED'),
                    COUNT(*) FILTER (WHERE lead_status = 'QUALIFIED'),
                    COUNT(*) FILTER (WHERE lead_status = 'CONVERTED')
                 FROM crm_leads WHERE tenant_id = $1",
            )
            .bind(tenant_id)
            .fetch_one(&self.pool)
            .await
            .map_err(CrmError::Database)?;

        Ok(LeadStats {
            total,
            new,
            contacted,
            qualified,
            converted,
            conversion_rate: if total > 0 {
                (converted as f64 / total as f64) * 100.0
            } else {
                0.0
            },
        })
    }
}

/// Lead statistics
#[derive(Debug, Serialize)]
pub struct LeadStats {
    pub total: i64,
    pub new: i64,
    pub contacted: i64,
    pub qualified: i64,
    pub converted: i64,
    pub conversion_rate: f64,
}
