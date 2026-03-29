// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Opportunity (Sales Pipeline) Management
//!
//! Track deals through the sales pipeline from prospect to close.

use chrono::{DateTime, NaiveDate, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

use super::CrmError;

// ============================================================================
// Types
// ============================================================================

/// A sales opportunity
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Opportunity {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub account_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub stage: String,
    pub amount: Option<Decimal>,
    pub probability: i32,
    pub expected_revenue: Option<Decimal>,
    pub close_date: Option<NaiveDate>,
    pub lead_source: Option<String>,
    pub next_step: Option<String>,
    pub forecast_category: String, // PIPELINE, BEST_CASE, COMMIT, CLOSED
    pub is_closed: bool,
    pub is_won: bool,
    pub owner_id: Option<Uuid>,
    pub primary_contact_id: Option<Uuid>,
    pub campaign_id: Option<Uuid>,
    pub competitors: Vec<String>,
    pub tags: Vec<String>,
    pub custom_fields: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Pipeline stage configuration
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct PipelineStage {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub name: String,
    pub probability: i32,
    pub sort_order: i32,
    pub forecast_category: String,
    pub is_closed: bool,
    pub is_won: bool,
    pub is_active: bool,
}

/// Create opportunity request
#[derive(Debug, Deserialize)]
pub struct CreateOpportunityRequest {
    pub account_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub stage: Option<String>,
    pub amount: Option<Decimal>,
    pub close_date: Option<NaiveDate>,
    pub lead_source: Option<String>,
    pub owner_id: Option<Uuid>,
    pub primary_contact_id: Option<Uuid>,
    pub tags: Option<Vec<String>>,
}

/// Pipeline summary
#[derive(Debug, Serialize)]
pub struct PipelineSummary {
    pub stages: Vec<StageSummary>,
    pub total_pipeline_value: Decimal,
    pub total_weighted_value: Decimal,
    pub total_opportunities: i64,
    pub avg_deal_size: Decimal,
    pub win_rate: f64,
}

/// Stage summary
#[derive(Debug, Serialize)]
pub struct StageSummary {
    pub stage: String,
    pub count: i64,
    pub value: Decimal,
    pub weighted_value: Decimal,
}

// ============================================================================
// Service
// ============================================================================

/// Opportunity Management Service
#[derive(Clone)]
pub struct OpportunityService {
    pool: PgPool,
}

impl OpportunityService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a new opportunity
    pub async fn create_opportunity(
        &self,
        tenant_id: Uuid,
        req: CreateOpportunityRequest,
    ) -> Result<Opportunity, CrmError> {
        let id = Uuid::new_v4();
        let tags = req.tags.unwrap_or_default();
        let stage = req.stage.unwrap_or_else(|| "PROSPECTING".to_string());
        let probability = self.get_stage_probability(&stage);

        sqlx::query(
            "INSERT INTO crm_opportunities (
                id, tenant_id, account_id, name, description, stage, amount,
                probability, close_date, lead_source, owner_id, primary_contact_id, tags
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)",
        )
        .bind(id)
        .bind(tenant_id)
        .bind(req.account_id)
        .bind(&req.name)
        .bind(&req.description)
        .bind(&stage)
        .bind(req.amount)
        .bind(probability)
        .bind(req.close_date)
        .bind(&req.lead_source)
        .bind(req.owner_id)
        .bind(req.primary_contact_id)
        .bind(&tags)
        .execute(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        self.get_opportunity(id).await
    }

    /// Get stage probability
    fn get_stage_probability(&self, stage: &str) -> i32 {
        match stage {
            "PROSPECTING" => 10,
            "QUALIFICATION" => 20,
            "NEEDS_ANALYSIS" => 40,
            "VALUE_PROPOSITION" => 50,
            "PROPOSAL" => 60,
            "NEGOTIATION" => 80,
            "CLOSED_WON" => 100,
            "CLOSED_LOST" => 0,
            _ => 10,
        }
    }

    /// Get an opportunity by ID
    pub async fn get_opportunity(&self, id: Uuid) -> Result<Opportunity, CrmError> {
        sqlx::query_as::<_, Opportunity>("SELECT * FROM crm_opportunities WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(CrmError::Database)?
            .ok_or_else(|| CrmError::NotFound("Opportunity not found".into()))
    }

    /// List opportunities for a tenant
    pub async fn list_opportunities(
        &self,
        tenant_id: Uuid,
        stage: Option<&str>,
        owner_id: Option<Uuid>,
        is_closed: Option<bool>,
        limit: i32,
        offset: i32,
    ) -> Result<Vec<Opportunity>, CrmError> {
        let opportunities = match (stage, owner_id, is_closed) {
            (Some(s), _, _) => {
                sqlx::query_as::<_, Opportunity>(
                    "SELECT * FROM crm_opportunities
                     WHERE tenant_id = $1 AND stage = $2
                     ORDER BY close_date, amount DESC NULLS LAST
                     LIMIT $3 OFFSET $4",
                )
                .bind(tenant_id)
                .bind(s)
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            }
            (None, Some(o), _) => {
                sqlx::query_as::<_, Opportunity>(
                    "SELECT * FROM crm_opportunities
                     WHERE tenant_id = $1 AND owner_id = $2
                     ORDER BY close_date, amount DESC NULLS LAST
                     LIMIT $3 OFFSET $4",
                )
                .bind(tenant_id)
                .bind(o)
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            }
            (None, None, Some(closed)) => {
                sqlx::query_as::<_, Opportunity>(
                    "SELECT * FROM crm_opportunities
                     WHERE tenant_id = $1 AND is_closed = $2
                     ORDER BY close_date, amount DESC NULLS LAST
                     LIMIT $3 OFFSET $4",
                )
                .bind(tenant_id)
                .bind(closed)
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            }
            (None, None, None) => {
                sqlx::query_as::<_, Opportunity>(
                    "SELECT * FROM crm_opportunities
                     WHERE tenant_id = $1
                     ORDER BY close_date, amount DESC NULLS LAST
                     LIMIT $2 OFFSET $3",
                )
                .bind(tenant_id)
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            }
        }
        .map_err(CrmError::Database)?;

        Ok(opportunities)
    }

    /// Update opportunity stage
    pub async fn update_stage(
        &self,
        id: Uuid,
        stage: &str,
    ) -> Result<Opportunity, CrmError> {
        let probability = self.get_stage_probability(stage);
        let is_closed = stage.starts_with("CLOSED");
        let is_won = stage == "CLOSED_WON";

        let forecast_category = if is_won {
            "CLOSED"
        } else if is_closed {
            "OMITTED"
        } else if probability >= 80 {
            "COMMIT"
        } else if probability >= 50 {
            "BEST_CASE"
        } else {
            "PIPELINE"
        };

        sqlx::query(
            "UPDATE crm_opportunities SET
                stage = $2, probability = $3, is_closed = $4, is_won = $5,
                forecast_category = $6, updated_at = NOW()
             WHERE id = $1",
        )
        .bind(id)
        .bind(stage)
        .bind(probability)
        .bind(is_closed)
        .bind(is_won)
        .bind(forecast_category)
        .execute(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        self.get_opportunity(id).await
    }

    /// Update opportunity amount
    pub async fn update_amount(
        &self,
        id: Uuid,
        amount: Decimal,
    ) -> Result<Opportunity, CrmError> {
        sqlx::query(
            "UPDATE crm_opportunities SET amount = $2, updated_at = NOW() WHERE id = $1",
        )
        .bind(id)
        .bind(amount)
        .execute(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        self.get_opportunity(id).await
    }

    /// Get pipeline summary
    pub async fn get_pipeline_summary(&self, tenant_id: Uuid) -> Result<PipelineSummary, CrmError> {
        let stages: Vec<(String, i64, Decimal, i32)> = sqlx::query_as(
            "SELECT stage, COUNT(*), COALESCE(SUM(amount), 0), AVG(probability)::int
             FROM crm_opportunities
             WHERE tenant_id = $1 AND is_closed = false
             GROUP BY stage
             ORDER BY AVG(probability)",
        )
        .bind(tenant_id)
        .fetch_all(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        let stage_summaries: Vec<StageSummary> = stages
            .iter()
            .map(|(stage, count, value, prob)| {
                let weighted = *value * Decimal::from(*prob) / Decimal::from(100);
                StageSummary {
                    stage: stage.clone(),
                    count: *count,
                    value: *value,
                    weighted_value: weighted,
                }
            })
            .collect();

        let total_value: Decimal = stage_summaries.iter().map(|s| s.value).sum();
        let total_weighted: Decimal = stage_summaries.iter().map(|s| s.weighted_value).sum();
        let total_count: i64 = stage_summaries.iter().map(|s| s.count).sum();

        let (won_count, total_closed): (i64, i64) = sqlx::query_as(
            "SELECT
                COUNT(*) FILTER (WHERE is_won = true),
                COUNT(*) FILTER (WHERE is_closed = true)
             FROM crm_opportunities WHERE tenant_id = $1",
        )
        .bind(tenant_id)
        .fetch_one(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        Ok(PipelineSummary {
            stages: stage_summaries,
            total_pipeline_value: total_value,
            total_weighted_value: total_weighted,
            total_opportunities: total_count,
            avg_deal_size: if total_count > 0 {
                total_value / Decimal::from(total_count)
            } else {
                Decimal::ZERO
            },
            win_rate: if total_closed > 0 {
                (won_count as f64 / total_closed as f64) * 100.0
            } else {
                0.0
            },
        })
    }

    /// Get opportunities closing this month
    pub async fn get_closing_this_month(&self, tenant_id: Uuid) -> Result<Vec<Opportunity>, CrmError> {
        let opportunities = sqlx::query_as::<_, Opportunity>(
            "SELECT * FROM crm_opportunities
             WHERE tenant_id = $1
               AND is_closed = false
               AND close_date >= date_trunc('month', CURRENT_DATE)
               AND close_date < date_trunc('month', CURRENT_DATE) + INTERVAL '1 month'
             ORDER BY close_date, amount DESC",
        )
        .bind(tenant_id)
        .fetch_all(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        Ok(opportunities)
    }
}
