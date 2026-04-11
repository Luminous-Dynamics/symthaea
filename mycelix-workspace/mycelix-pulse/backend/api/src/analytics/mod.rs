// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Email Analytics
//!
//! Metrics, insights, and productivity dashboards

use chrono::{DateTime, Utc, Duration, Datelike, Timelike};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;

// ============================================================================
// Analytics Service
// ============================================================================

pub struct AnalyticsService {
    pool: PgPool,
}

impl AnalyticsService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get email volume metrics over time
    pub async fn get_email_volume(
        &self,
        user_id: Uuid,
        period: AnalyticsPeriod,
    ) -> Result<EmailVolumeStats, AnalyticsError> {
        let (start_date, interval) = self.get_period_params(&period);

        let received: Vec<(DateTime<Utc>, i64)> = sqlx::query_as(
            r#"
            SELECT date_trunc($1, received_at) as period, COUNT(*) as count
            FROM emails
            WHERE user_id = $2 AND received_at >= $3
            GROUP BY period
            ORDER BY period
            "#
        )
        .bind(&interval)
        .bind(user_id)
        .bind(start_date)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| AnalyticsError::Database(e.to_string()))?;

        let sent: Vec<(DateTime<Utc>, i64)> = sqlx::query_as(
            r#"
            SELECT date_trunc($1, sent_at) as period, COUNT(*) as count
            FROM sent_emails
            WHERE user_id = $2 AND sent_at >= $3
            GROUP BY period
            ORDER BY period
            "#
        )
        .bind(&interval)
        .bind(user_id)
        .bind(start_date)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| AnalyticsError::Database(e.to_string()))?;

        Ok(EmailVolumeStats {
            period,
            received: received.into_iter().map(|(d, c)| DataPoint { date: d, value: c }).collect(),
            sent: sent.into_iter().map(|(d, c)| DataPoint { date: d, value: c }).collect(),
            total_received: received.iter().map(|(_, c)| c).sum(),
            total_sent: sent.iter().map(|(_, c)| c).sum(),
        })
    }

    /// Get response time metrics
    pub async fn get_response_times(
        &self,
        user_id: Uuid,
        period: AnalyticsPeriod,
    ) -> Result<ResponseTimeStats, AnalyticsError> {
        let (start_date, _) = self.get_period_params(&period);

        // Average response time for replies
        let avg_response: Option<(f64,)> = sqlx::query_as(
            r#"
            SELECT AVG(EXTRACT(EPOCH FROM (reply.sent_at - original.received_at))) as avg_seconds
            FROM sent_emails reply
            JOIN emails original ON reply.in_reply_to = original.message_id
            WHERE reply.user_id = $1 AND reply.sent_at >= $2
            "#
        )
        .bind(user_id)
        .bind(start_date)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| AnalyticsError::Database(e.to_string()))?;

        // Response time distribution
        let distribution: Vec<(String, i64)> = sqlx::query_as(
            r#"
            SELECT
                CASE
                    WHEN EXTRACT(EPOCH FROM (reply.sent_at - original.received_at)) < 3600 THEN 'under_1h'
                    WHEN EXTRACT(EPOCH FROM (reply.sent_at - original.received_at)) < 14400 THEN '1h_to_4h'
                    WHEN EXTRACT(EPOCH FROM (reply.sent_at - original.received_at)) < 86400 THEN '4h_to_24h'
                    ELSE 'over_24h'
                END as bucket,
                COUNT(*) as count
            FROM sent_emails reply
            JOIN emails original ON reply.in_reply_to = original.message_id
            WHERE reply.user_id = $1 AND reply.sent_at >= $2
            GROUP BY bucket
            "#
        )
        .bind(user_id)
        .bind(start_date)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| AnalyticsError::Database(e.to_string()))?;

        Ok(ResponseTimeStats {
            period,
            average_seconds: avg_response.map(|(v,)| v).unwrap_or(0.0),
            distribution: distribution.into_iter().collect(),
        })
    }

    /// Get top senders/recipients
    pub async fn get_top_correspondents(
        &self,
        user_id: Uuid,
        limit: i64,
    ) -> Result<TopCorrespondents, AnalyticsError> {
        let top_senders: Vec<CorrespondentStats> = sqlx::query_as(
            r#"
            SELECT from_address as email, from_name as name, COUNT(*) as email_count
            FROM emails
            WHERE user_id = $1
            GROUP BY from_address, from_name
            ORDER BY email_count DESC
            LIMIT $2
            "#
        )
        .bind(user_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| AnalyticsError::Database(e.to_string()))?;

        let top_recipients: Vec<CorrespondentStats> = sqlx::query_as(
            r#"
            SELECT UNNEST(to_addresses) as email, NULL as name, COUNT(*) as email_count
            FROM sent_emails
            WHERE user_id = $1
            GROUP BY email
            ORDER BY email_count DESC
            LIMIT $2
            "#
        )
        .bind(user_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| AnalyticsError::Database(e.to_string()))?;

        Ok(TopCorrespondents {
            top_senders,
            top_recipients,
        })
    }

    /// Get email activity heatmap (hour of day / day of week)
    pub async fn get_activity_heatmap(
        &self,
        user_id: Uuid,
    ) -> Result<ActivityHeatmap, AnalyticsError> {
        let received_heatmap: Vec<(i32, i32, i64)> = sqlx::query_as(
            r#"
            SELECT
                EXTRACT(DOW FROM received_at)::int as day_of_week,
                EXTRACT(HOUR FROM received_at)::int as hour,
                COUNT(*) as count
            FROM emails
            WHERE user_id = $1
            GROUP BY day_of_week, hour
            "#
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| AnalyticsError::Database(e.to_string()))?;

        let sent_heatmap: Vec<(i32, i32, i64)> = sqlx::query_as(
            r#"
            SELECT
                EXTRACT(DOW FROM sent_at)::int as day_of_week,
                EXTRACT(HOUR FROM sent_at)::int as hour,
                COUNT(*) as count
            FROM sent_emails
            WHERE user_id = $1
            GROUP BY day_of_week, hour
            "#
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| AnalyticsError::Database(e.to_string()))?;

        // Convert to grid format
        let mut received_grid = vec![vec![0i64; 24]; 7];
        let mut sent_grid = vec![vec![0i64; 24]; 7];

        for (dow, hour, count) in received_heatmap {
            received_grid[dow as usize][hour as usize] = count;
        }
        for (dow, hour, count) in sent_heatmap {
            sent_grid[dow as usize][hour as usize] = count;
        }

        Ok(ActivityHeatmap {
            received: received_grid,
            sent: sent_grid,
            peak_receive_hour: self.find_peak_hour(&received_heatmap),
            peak_send_hour: self.find_peak_hour(&sent_heatmap),
        })
    }

    fn find_peak_hour(&self, data: &[(i32, i32, i64)]) -> i32 {
        let mut hour_totals = vec![0i64; 24];
        for (_, hour, count) in data {
            hour_totals[*hour as usize] += count;
        }
        hour_totals
            .iter()
            .enumerate()
            .max_by_key(|(_, v)| *v)
            .map(|(i, _)| i as i32)
            .unwrap_or(9)
    }

    /// Get folder distribution stats
    pub async fn get_folder_stats(
        &self,
        user_id: Uuid,
    ) -> Result<Vec<FolderStats>, AnalyticsError> {
        let stats: Vec<FolderStats> = sqlx::query_as(
            r#"
            SELECT
                folder,
                COUNT(*) as total_count,
                COUNT(*) FILTER (WHERE is_read = false) as unread_count,
                SUM(size_bytes) as total_size
            FROM emails
            WHERE user_id = $1
            GROUP BY folder
            ORDER BY total_count DESC
            "#
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| AnalyticsError::Database(e.to_string()))?;

        Ok(stats)
    }

    /// Get productivity insights
    pub async fn get_productivity_insights(
        &self,
        user_id: Uuid,
    ) -> Result<ProductivityInsights, AnalyticsError> {
        let now = Utc::now();
        let week_ago = now - Duration::days(7);
        let two_weeks_ago = now - Duration::days(14);

        // This week vs last week comparison
        let this_week: (i64, i64) = sqlx::query_as(
            r#"
            SELECT
                COUNT(*) FILTER (WHERE received_at >= $2) as received,
                (SELECT COUNT(*) FROM sent_emails WHERE user_id = $1 AND sent_at >= $2) as sent
            FROM emails
            WHERE user_id = $1 AND received_at >= $2
            "#
        )
        .bind(user_id)
        .bind(week_ago)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| AnalyticsError::Database(e.to_string()))?;

        let last_week: (i64, i64) = sqlx::query_as(
            r#"
            SELECT
                COUNT(*) FILTER (WHERE received_at >= $2 AND received_at < $3) as received,
                (SELECT COUNT(*) FROM sent_emails WHERE user_id = $1 AND sent_at >= $2 AND sent_at < $3) as sent
            FROM emails
            WHERE user_id = $1 AND received_at >= $2 AND received_at < $3
            "#
        )
        .bind(user_id)
        .bind(two_weeks_ago)
        .bind(week_ago)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| AnalyticsError::Database(e.to_string()))?;

        // Unread count
        let unread: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM emails WHERE user_id = $1 AND is_read = false"
        )
        .bind(user_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| AnalyticsError::Database(e.to_string()))?;

        // Email zero days this month
        let zero_inbox_days: (i64,) = sqlx::query_as(
            r#"
            SELECT COUNT(DISTINCT DATE(checked_at))
            FROM inbox_zero_log
            WHERE user_id = $1 AND checked_at >= date_trunc('month', NOW())
            "#
        )
        .bind(user_id)
        .fetch_one(&self.pool)
        .await
        .unwrap_or((0,));

        Ok(ProductivityInsights {
            emails_received_this_week: this_week.0,
            emails_sent_this_week: this_week.1,
            received_vs_last_week: if last_week.0 > 0 {
                ((this_week.0 - last_week.0) as f64 / last_week.0 as f64) * 100.0
            } else {
                0.0
            },
            sent_vs_last_week: if last_week.1 > 0 {
                ((this_week.1 - last_week.1) as f64 / last_week.1 as f64) * 100.0
            } else {
                0.0
            },
            current_unread: unread.0,
            inbox_zero_days_this_month: zero_inbox_days.0,
        })
    }

    fn get_period_params(&self, period: &AnalyticsPeriod) -> (DateTime<Utc>, String) {
        let now = Utc::now();
        match period {
            AnalyticsPeriod::Day => (now - Duration::days(1), "hour".to_string()),
            AnalyticsPeriod::Week => (now - Duration::weeks(1), "day".to_string()),
            AnalyticsPeriod::Month => (now - Duration::days(30), "day".to_string()),
            AnalyticsPeriod::Quarter => (now - Duration::days(90), "week".to_string()),
            AnalyticsPeriod::Year => (now - Duration::days(365), "month".to_string()),
        }
    }
}

// ============================================================================
// Team Analytics Service
// ============================================================================

pub struct TeamAnalyticsService {
    pool: PgPool,
}

impl TeamAnalyticsService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get team workload distribution
    pub async fn get_team_workload(
        &self,
        org_id: Uuid,
    ) -> Result<Vec<TeamMemberWorkload>, AnalyticsError> {
        let workload: Vec<TeamMemberWorkload> = sqlx::query_as(
            r#"
            SELECT
                u.id as user_id,
                u.name as user_name,
                COUNT(a.email_id) FILTER (WHERE a.status = 'open') as open_tickets,
                COUNT(a.email_id) FILTER (WHERE a.status = 'pending') as pending_tickets,
                COUNT(a.email_id) FILTER (WHERE a.completed_at >= NOW() - INTERVAL '7 days') as completed_this_week,
                AVG(EXTRACT(EPOCH FROM (a.completed_at - a.assigned_at))) FILTER (WHERE a.completed_at IS NOT NULL) as avg_resolution_seconds
            FROM users u
            LEFT JOIN email_assignments a ON u.id = a.assignee_id
            WHERE u.organization_id = $1
            GROUP BY u.id, u.name
            ORDER BY open_tickets DESC
            "#
        )
        .bind(org_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| AnalyticsError::Database(e.to_string()))?;

        Ok(workload)
    }

    /// Get SLA performance metrics
    pub async fn get_sla_performance(
        &self,
        org_id: Uuid,
        period: AnalyticsPeriod,
    ) -> Result<SlaPerformance, AnalyticsError> {
        let (start_date, _) = AnalyticsService::new(self.pool.clone()).get_period_params(&period);

        let metrics: (i64, i64, i64) = sqlx::query_as(
            r#"
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE first_response_at <= first_response_due) as met_first_response,
                COUNT(*) FILTER (WHERE resolved_at <= resolution_due) as met_resolution
            FROM sla_tickets
            WHERE organization_id = $1 AND created_at >= $2
            "#
        )
        .bind(org_id)
        .bind(start_date)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| AnalyticsError::Database(e.to_string()))?;

        Ok(SlaPerformance {
            period,
            total_tickets: metrics.0,
            first_response_met: metrics.1,
            resolution_met: metrics.2,
            first_response_rate: if metrics.0 > 0 { metrics.1 as f64 / metrics.0 as f64 * 100.0 } else { 0.0 },
            resolution_rate: if metrics.0 > 0 { metrics.2 as f64 / metrics.0 as f64 * 100.0 } else { 0.0 },
        })
    }
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AnalyticsPeriod {
    Day,
    Week,
    Month,
    Quarter,
    Year,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub date: DateTime<Utc>,
    pub value: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailVolumeStats {
    pub period: AnalyticsPeriod,
    pub received: Vec<DataPoint>,
    pub sent: Vec<DataPoint>,
    pub total_received: i64,
    pub total_sent: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTimeStats {
    pub period: AnalyticsPeriod,
    pub average_seconds: f64,
    pub distribution: HashMap<String, i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct CorrespondentStats {
    pub email: String,
    pub name: Option<String>,
    pub email_count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopCorrespondents {
    pub top_senders: Vec<CorrespondentStats>,
    pub top_recipients: Vec<CorrespondentStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityHeatmap {
    pub received: Vec<Vec<i64>>,  // 7 days x 24 hours
    pub sent: Vec<Vec<i64>>,
    pub peak_receive_hour: i32,
    pub peak_send_hour: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct FolderStats {
    pub folder: String,
    pub total_count: i64,
    pub unread_count: i64,
    pub total_size: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductivityInsights {
    pub emails_received_this_week: i64,
    pub emails_sent_this_week: i64,
    pub received_vs_last_week: f64,  // percentage change
    pub sent_vs_last_week: f64,
    pub current_unread: i64,
    pub inbox_zero_days_this_month: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct TeamMemberWorkload {
    pub user_id: Uuid,
    pub user_name: String,
    pub open_tickets: i64,
    pub pending_tickets: i64,
    pub completed_this_week: i64,
    pub avg_resolution_seconds: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaPerformance {
    pub period: AnalyticsPeriod,
    pub total_tickets: i64,
    pub first_response_met: i64,
    pub resolution_met: i64,
    pub first_response_rate: f64,
    pub resolution_rate: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum AnalyticsError {
    #[error("Database error: {0}")]
    Database(String),
    #[error("Invalid period: {0}")]
    InvalidPeriod(String),
}
