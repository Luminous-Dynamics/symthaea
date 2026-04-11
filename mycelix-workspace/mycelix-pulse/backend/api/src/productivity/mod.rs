// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Productivity Intelligence Module
//!
//! Provides response time analytics, relationship scores, productivity dashboards,
//! email load forecasting, writing analytics, and goals/streaks.

use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc, Weekday};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum ProductivityError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    #[error("Invalid date range: {0}")]
    InvalidDateRange(String),
}

// ============================================================================
// Response Time Analytics
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTimeStats {
    pub avg_response_time_hours: f32,
    pub median_response_time_hours: f32,
    pub fastest_response_minutes: i32,
    pub slowest_response_hours: i32,
    pub response_rate: f32, // percentage of emails responded to
    pub by_sender: Vec<SenderResponseStats>,
    pub by_day_of_week: HashMap<String, f32>,
    pub by_hour: HashMap<i32, f32>,
    pub trend: ResponseTrend,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SenderResponseStats {
    pub email: String,
    pub name: Option<String>,
    pub avg_response_time_hours: f32,
    pub total_emails: i32,
    pub total_responses: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseTrend {
    Improving,
    Stable,
    Declining,
}

pub struct ResponseTimeService {
    pool: PgPool,
}

impl ResponseTimeService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn get_stats(
        &self,
        user_id: Uuid,
        days: i32,
    ) -> Result<ResponseTimeStats, ProductivityError> {
        let since = Utc::now() - Duration::days(days as i64);

        // Get overall stats
        let overall = sqlx::query!(
            r#"
            SELECT
                AVG(EXTRACT(EPOCH FROM (response_at - received_at)) / 3600) as avg_hours,
                PERCENTILE_CONT(0.5) WITHIN GROUP (
                    ORDER BY EXTRACT(EPOCH FROM (response_at - received_at)) / 3600
                ) as median_hours,
                MIN(EXTRACT(EPOCH FROM (response_at - received_at)) / 60) as min_minutes,
                MAX(EXTRACT(EPOCH FROM (response_at - received_at)) / 3600) as max_hours,
                COUNT(*) FILTER (WHERE response_at IS NOT NULL) as responded,
                COUNT(*) as total
            FROM emails
            WHERE recipient_id = $1 AND received_at > $2
            "#,
            user_id,
            since
        )
        .fetch_one(&self.pool)
        .await?;

        // Get stats by sender
        let by_sender = sqlx::query!(
            r#"
            SELECT
                sender_email as email,
                sender_name as name,
                AVG(EXTRACT(EPOCH FROM (response_at - received_at)) / 3600) as avg_hours,
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE response_at IS NOT NULL) as responded
            FROM emails
            WHERE recipient_id = $1 AND received_at > $2
            GROUP BY sender_email, sender_name
            ORDER BY COUNT(*) DESC
            LIMIT 20
            "#,
            user_id,
            since
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| SenderResponseStats {
            email: row.email,
            name: row.name,
            avg_response_time_hours: row.avg_hours.unwrap_or(0.0) as f32,
            total_emails: row.total.unwrap_or(0) as i32,
            total_responses: row.responded.unwrap_or(0) as i32,
        })
        .collect();

        // Get stats by day of week
        let by_dow = sqlx::query!(
            r#"
            SELECT
                EXTRACT(DOW FROM response_at) as dow,
                AVG(EXTRACT(EPOCH FROM (response_at - received_at)) / 3600) as avg_hours
            FROM emails
            WHERE recipient_id = $1 AND received_at > $2 AND response_at IS NOT NULL
            GROUP BY EXTRACT(DOW FROM response_at)
            "#,
            user_id,
            since
        )
        .fetch_all(&self.pool)
        .await?;

        let mut by_day_of_week = HashMap::new();
        for row in by_dow {
            let day = match row.dow.unwrap_or(0.0) as i32 {
                0 => "Sunday",
                1 => "Monday",
                2 => "Tuesday",
                3 => "Wednesday",
                4 => "Thursday",
                5 => "Friday",
                6 => "Saturday",
                _ => "Unknown",
            };
            by_day_of_week.insert(day.to_string(), row.avg_hours.unwrap_or(0.0) as f32);
        }

        // Get stats by hour
        let by_hour_data = sqlx::query!(
            r#"
            SELECT
                EXTRACT(HOUR FROM response_at) as hour,
                AVG(EXTRACT(EPOCH FROM (response_at - received_at)) / 3600) as avg_hours
            FROM emails
            WHERE recipient_id = $1 AND received_at > $2 AND response_at IS NOT NULL
            GROUP BY EXTRACT(HOUR FROM response_at)
            "#,
            user_id,
            since
        )
        .fetch_all(&self.pool)
        .await?;

        let mut by_hour = HashMap::new();
        for row in by_hour_data {
            by_hour.insert(
                row.hour.unwrap_or(0.0) as i32,
                row.avg_hours.unwrap_or(0.0) as f32,
            );
        }

        // Calculate trend (compare last 7 days to previous 7 days)
        let trend = self.calculate_response_trend(user_id).await?;

        let total = overall.total.unwrap_or(0) as f32;
        let responded = overall.responded.unwrap_or(0) as f32;

        Ok(ResponseTimeStats {
            avg_response_time_hours: overall.avg_hours.unwrap_or(0.0) as f32,
            median_response_time_hours: overall.median_hours.unwrap_or(0.0) as f32,
            fastest_response_minutes: overall.min_minutes.unwrap_or(0.0) as i32,
            slowest_response_hours: overall.max_hours.unwrap_or(0.0) as i32,
            response_rate: if total > 0.0 { responded / total * 100.0 } else { 0.0 },
            by_sender,
            by_day_of_week,
            by_hour,
            trend,
        })
    }

    async fn calculate_response_trend(&self, user_id: Uuid) -> Result<ResponseTrend, ProductivityError> {
        let now = Utc::now();
        let last_week = now - Duration::days(7);
        let prev_week = now - Duration::days(14);

        let recent = sqlx::query_scalar!(
            r#"
            SELECT AVG(EXTRACT(EPOCH FROM (response_at - received_at)) / 3600) as "avg!"
            FROM emails
            WHERE recipient_id = $1 AND received_at > $2 AND response_at IS NOT NULL
            "#,
            user_id,
            last_week
        )
        .fetch_one(&self.pool)
        .await?;

        let previous = sqlx::query_scalar!(
            r#"
            SELECT AVG(EXTRACT(EPOCH FROM (response_at - received_at)) / 3600) as "avg!"
            FROM emails
            WHERE recipient_id = $1 AND received_at > $2 AND received_at <= $3
            AND response_at IS NOT NULL
            "#,
            user_id,
            prev_week,
            last_week
        )
        .fetch_one(&self.pool)
        .await?;

        let change = (recent - previous) / previous * 100.0;

        Ok(if change < -10.0 {
            ResponseTrend::Improving
        } else if change > 10.0 {
            ResponseTrend::Declining
        } else {
            ResponseTrend::Stable
        })
    }
}

// ============================================================================
// Relationship Scores
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipScore {
    pub contact_email: String,
    pub contact_name: Option<String>,
    pub overall_score: f32, // 0-100
    pub components: RelationshipComponents,
    pub last_interaction: DateTime<Utc>,
    pub relationship_type: RelationshipType,
    pub trend: RelationshipTrend,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipComponents {
    pub frequency_score: f32,    // How often you communicate
    pub recency_score: f32,      // How recent was last contact
    pub reciprocity_score: f32,  // Balance of sent/received
    pub response_score: f32,     // How quickly you respond to each other
    pub engagement_score: f32,   // Length of conversations
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    VeryActive,    // Daily/weekly contact
    Active,        // Regular contact
    Moderate,      // Occasional contact
    Dormant,       // No recent contact
    New,           // Recently started communicating
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipTrend {
    Strengthening,
    Stable,
    Weakening,
}

pub struct RelationshipService {
    pool: PgPool,
}

impl RelationshipService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn get_relationship_scores(
        &self,
        user_id: Uuid,
        limit: i32,
    ) -> Result<Vec<RelationshipScore>, ProductivityError> {
        let contacts = sqlx::query!(
            r#"
            SELECT
                contact_email,
                contact_name,
                sent_count,
                received_count,
                last_sent,
                last_received,
                avg_response_hours,
                first_contact
            FROM (
                SELECT
                    COALESCE(s.recipient_email, r.sender_email) as contact_email,
                    COALESCE(s.recipient_name, r.sender_name) as contact_name,
                    COALESCE(s.sent_count, 0) as sent_count,
                    COALESCE(r.received_count, 0) as received_count,
                    s.last_sent,
                    r.last_received,
                    r.avg_response_hours,
                    LEAST(s.first_sent, r.first_received) as first_contact
                FROM (
                    SELECT
                        recipient_email,
                        MAX(recipient_name) as recipient_name,
                        COUNT(*) as sent_count,
                        MAX(sent_at) as last_sent,
                        MIN(sent_at) as first_sent
                    FROM emails
                    WHERE sender_id = $1 AND sent_at IS NOT NULL
                    GROUP BY recipient_email
                ) s
                FULL OUTER JOIN (
                    SELECT
                        sender_email,
                        MAX(sender_name) as sender_name,
                        COUNT(*) as received_count,
                        MAX(received_at) as last_received,
                        MIN(received_at) as first_received,
                        AVG(EXTRACT(EPOCH FROM (response_at - received_at)) / 3600) as avg_response_hours
                    FROM emails
                    WHERE recipient_id = $1
                    GROUP BY sender_email
                ) r ON s.recipient_email = r.sender_email
            ) combined
            ORDER BY (COALESCE(sent_count, 0) + COALESCE(received_count, 0)) DESC
            LIMIT $2
            "#,
            user_id,
            limit as i64
        )
        .fetch_all(&self.pool)
        .await?;

        let mut scores = Vec::new();

        for contact in contacts {
            let sent = contact.sent_count.unwrap_or(0) as f32;
            let received = contact.received_count.unwrap_or(0) as f32;
            let total = sent + received;

            let last_interaction = contact
                .last_sent
                .max(contact.last_received)
                .unwrap_or_else(Utc::now);

            let days_since = (Utc::now() - last_interaction).num_days() as f32;
            let days_known = contact
                .first_contact
                .map(|f| (Utc::now() - f).num_days() as f32)
                .unwrap_or(1.0);

            // Calculate component scores
            let frequency_score = (total / days_known.max(1.0) * 10.0).min(100.0);
            let recency_score = (100.0 - days_since * 2.0).max(0.0);
            let reciprocity_score = if total > 0.0 {
                let ratio = (sent.min(received) / total.max(1.0)) * 2.0;
                ratio * 100.0
            } else {
                0.0
            };
            let response_score = contact
                .avg_response_hours
                .map(|h| (100.0 - h.min(48.0) as f32 * 2.0).max(0.0))
                .unwrap_or(50.0);
            let engagement_score = ((total / 10.0) * 20.0).min(100.0);

            let components = RelationshipComponents {
                frequency_score,
                recency_score,
                reciprocity_score,
                response_score,
                engagement_score,
            };

            let overall_score = (frequency_score * 0.25
                + recency_score * 0.25
                + reciprocity_score * 0.2
                + response_score * 0.15
                + engagement_score * 0.15)
                .min(100.0);

            let relationship_type = if days_since < 7.0 && total > 10.0 {
                RelationshipType::VeryActive
            } else if days_since < 30.0 && total > 5.0 {
                RelationshipType::Active
            } else if days_since < 90.0 {
                RelationshipType::Moderate
            } else if days_known < 30.0 {
                RelationshipType::New
            } else {
                RelationshipType::Dormant
            };

            // Simple trend (would be more sophisticated in production)
            let trend = if recency_score > 70.0 && frequency_score > 50.0 {
                RelationshipTrend::Strengthening
            } else if days_since > 60.0 {
                RelationshipTrend::Weakening
            } else {
                RelationshipTrend::Stable
            };

            scores.push(RelationshipScore {
                contact_email: contact.contact_email.unwrap_or_default(),
                contact_name: contact.contact_name,
                overall_score,
                components,
                last_interaction,
                relationship_type,
                trend,
            });
        }

        Ok(scores)
    }

    pub async fn get_relationship_with(
        &self,
        user_id: Uuid,
        contact_email: &str,
    ) -> Result<RelationshipScore, ProductivityError> {
        let scores = self.get_relationship_scores(user_id, 1000).await?;

        scores
            .into_iter()
            .find(|s| s.contact_email.eq_ignore_ascii_case(contact_email))
            .ok_or(ProductivityError::InsufficientData(
                "No interaction history with this contact".to_string(),
            ))
    }

    pub async fn get_dormant_relationships(
        &self,
        user_id: Uuid,
        days_inactive: i32,
    ) -> Result<Vec<RelationshipScore>, ProductivityError> {
        let scores = self.get_relationship_scores(user_id, 500).await?;

        Ok(scores
            .into_iter()
            .filter(|s| matches!(s.relationship_type, RelationshipType::Dormant))
            .collect())
    }
}

// ============================================================================
// Productivity Dashboard
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductivityDashboard {
    pub period: DashboardPeriod,
    pub email_volume: EmailVolume,
    pub productivity_score: f32,
    pub peak_hours: Vec<PeakHour>,
    pub category_breakdown: Vec<CategoryStat>,
    pub comparison: PeriodComparison,
    pub inbox_health: InboxHealth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardPeriod {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailVolume {
    pub sent: i32,
    pub received: i32,
    pub archived: i32,
    pub deleted: i32,
    pub daily_average_sent: f32,
    pub daily_average_received: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakHour {
    pub hour: i32,
    pub sent_count: i32,
    pub received_count: i32,
    pub is_productive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryStat {
    pub category: String,
    pub count: i32,
    pub percentage: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodComparison {
    pub sent_change_percent: f32,
    pub received_change_percent: f32,
    pub response_time_change_percent: f32,
    pub direction: ComparisonDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonDirection {
    Better,
    Same,
    Worse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboxHealth {
    pub unread_count: i32,
    pub oldest_unread_days: i32,
    pub inbox_zero_days: i32, // Days at inbox zero in period
    pub average_processing_time_hours: f32,
}

pub struct DashboardService {
    pool: PgPool,
}

impl DashboardService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn get_dashboard(
        &self,
        user_id: Uuid,
        days: i32,
    ) -> Result<ProductivityDashboard, ProductivityError> {
        let end = Utc::now();
        let start = end - Duration::days(days as i64);
        let prev_start = start - Duration::days(days as i64);

        // Email volume
        let volume = sqlx::query!(
            r#"
            SELECT
                COUNT(*) FILTER (WHERE sender_id = $1 AND sent_at IS NOT NULL) as sent,
                COUNT(*) FILTER (WHERE recipient_id = $1) as received,
                COUNT(*) FILTER (WHERE recipient_id = $1 AND is_archived) as archived,
                COUNT(*) FILTER (WHERE recipient_id = $1 AND is_deleted) as deleted
            FROM emails
            WHERE (sender_id = $1 OR recipient_id = $1)
            AND created_at BETWEEN $2 AND $3
            "#,
            user_id,
            start,
            end
        )
        .fetch_one(&self.pool)
        .await?;

        let email_volume = EmailVolume {
            sent: volume.sent.unwrap_or(0) as i32,
            received: volume.received.unwrap_or(0) as i32,
            archived: volume.archived.unwrap_or(0) as i32,
            deleted: volume.deleted.unwrap_or(0) as i32,
            daily_average_sent: volume.sent.unwrap_or(0) as f32 / days as f32,
            daily_average_received: volume.received.unwrap_or(0) as f32 / days as f32,
        };

        // Peak hours
        let hours = sqlx::query!(
            r#"
            SELECT
                EXTRACT(HOUR FROM created_at) as hour,
                COUNT(*) FILTER (WHERE sender_id = $1) as sent,
                COUNT(*) FILTER (WHERE recipient_id = $1) as received
            FROM emails
            WHERE (sender_id = $1 OR recipient_id = $1)
            AND created_at BETWEEN $2 AND $3
            GROUP BY EXTRACT(HOUR FROM created_at)
            ORDER BY hour
            "#,
            user_id,
            start,
            end
        )
        .fetch_all(&self.pool)
        .await?;

        let peak_hours: Vec<PeakHour> = hours
            .into_iter()
            .map(|h| {
                let hour = h.hour.unwrap_or(0.0) as i32;
                let sent = h.sent.unwrap_or(0) as i32;
                let received = h.received.unwrap_or(0) as i32;
                PeakHour {
                    hour,
                    sent_count: sent,
                    received_count: received,
                    is_productive: sent > received && hour >= 9 && hour <= 17,
                }
            })
            .collect();

        // Category breakdown
        let categories = sqlx::query!(
            r#"
            SELECT category, COUNT(*) as count
            FROM emails
            WHERE recipient_id = $1 AND created_at BETWEEN $2 AND $3
            AND category IS NOT NULL
            GROUP BY category
            "#,
            user_id,
            start,
            end
        )
        .fetch_all(&self.pool)
        .await?;

        let total_categorized: i64 = categories.iter().map(|c| c.count.unwrap_or(0)).sum();
        let category_breakdown: Vec<CategoryStat> = categories
            .into_iter()
            .map(|c| CategoryStat {
                category: c.category.unwrap_or_default(),
                count: c.count.unwrap_or(0) as i32,
                percentage: if total_categorized > 0 {
                    c.count.unwrap_or(0) as f32 / total_categorized as f32 * 100.0
                } else {
                    0.0
                },
            })
            .collect();

        // Comparison with previous period
        let prev_volume = sqlx::query!(
            r#"
            SELECT
                COUNT(*) FILTER (WHERE sender_id = $1 AND sent_at IS NOT NULL) as sent,
                COUNT(*) FILTER (WHERE recipient_id = $1) as received
            FROM emails
            WHERE (sender_id = $1 OR recipient_id = $1)
            AND created_at BETWEEN $2 AND $3
            "#,
            user_id,
            prev_start,
            start
        )
        .fetch_one(&self.pool)
        .await?;

        let prev_sent = prev_volume.sent.unwrap_or(1) as f32;
        let prev_received = prev_volume.received.unwrap_or(1) as f32;

        let comparison = PeriodComparison {
            sent_change_percent: ((email_volume.sent as f32 - prev_sent) / prev_sent) * 100.0,
            received_change_percent: ((email_volume.received as f32 - prev_received) / prev_received) * 100.0,
            response_time_change_percent: 0.0, // Would calculate properly
            direction: ComparisonDirection::Same,
        };

        // Inbox health
        let health = sqlx::query!(
            r#"
            SELECT
                COUNT(*) FILTER (WHERE NOT is_read) as unread,
                EXTRACT(DAY FROM NOW() - MIN(received_at) FILTER (WHERE NOT is_read)) as oldest_unread
            FROM emails
            WHERE recipient_id = $1 AND folder = 'inbox'
            "#,
            user_id
        )
        .fetch_one(&self.pool)
        .await?;

        let inbox_health = InboxHealth {
            unread_count: health.unread.unwrap_or(0) as i32,
            oldest_unread_days: health.oldest_unread.unwrap_or(0.0) as i32,
            inbox_zero_days: 0, // Would calculate from history
            average_processing_time_hours: 4.0, // Would calculate properly
        };

        // Calculate productivity score
        let productivity_score = self.calculate_productivity_score(
            &email_volume,
            &inbox_health,
            &comparison,
        );

        Ok(ProductivityDashboard {
            period: DashboardPeriod {
                start,
                end,
                label: format!("Last {} days", days),
            },
            email_volume,
            productivity_score,
            peak_hours,
            category_breakdown,
            comparison,
            inbox_health,
        })
    }

    fn calculate_productivity_score(
        &self,
        volume: &EmailVolume,
        health: &InboxHealth,
        comparison: &PeriodComparison,
    ) -> f32 {
        let mut score = 50.0;

        // Positive: Low unread count
        if health.unread_count < 10 {
            score += 20.0;
        } else if health.unread_count < 50 {
            score += 10.0;
        } else {
            score -= 10.0;
        }

        // Positive: Good sent/received ratio
        if volume.sent > 0 && volume.received > 0 {
            let ratio = volume.sent as f32 / volume.received as f32;
            if ratio > 0.3 && ratio < 3.0 {
                score += 15.0;
            }
        }

        // Positive: Processing quickly
        if health.oldest_unread_days < 3 {
            score += 15.0;
        } else if health.oldest_unread_days > 7 {
            score -= 15.0;
        }

        score.clamp(0.0, 100.0)
    }
}

// ============================================================================
// Email Load Forecasting
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailForecast {
    pub predictions: Vec<DailyPrediction>,
    pub weekly_pattern: WeeklyPattern,
    pub confidence: f32,
    pub factors: Vec<ForecastFactor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyPrediction {
    pub date: NaiveDate,
    pub predicted_received: i32,
    pub predicted_sent: i32,
    pub confidence_low: i32,
    pub confidence_high: i32,
    pub is_busy_day: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeeklyPattern {
    pub busiest_day: String,
    pub quietest_day: String,
    pub by_day: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastFactor {
    pub factor: String,
    pub impact: FactorImpact,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactorImpact {
    Increase,
    Decrease,
    Neutral,
}

pub struct ForecastService {
    pool: PgPool,
}

impl ForecastService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn get_forecast(
        &self,
        user_id: Uuid,
        days_ahead: i32,
    ) -> Result<EmailForecast, ProductivityError> {
        // Get historical daily averages by day of week
        let historical = sqlx::query!(
            r#"
            SELECT
                EXTRACT(DOW FROM received_at) as dow,
                COUNT(*) / (COUNT(DISTINCT DATE(received_at))) as avg_daily
            FROM emails
            WHERE recipient_id = $1 AND received_at > NOW() - INTERVAL '90 days'
            GROUP BY EXTRACT(DOW FROM received_at)
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?;

        let mut by_dow: HashMap<i32, f32> = HashMap::new();
        for h in &historical {
            by_dow.insert(
                h.dow.unwrap_or(0.0) as i32,
                h.avg_daily.unwrap_or(0) as f32,
            );
        }

        // Generate predictions
        let mut predictions = Vec::new();
        let today = Utc::now().date_naive();

        for i in 1..=days_ahead {
            let date = today + chrono::Duration::days(i as i64);
            let dow = date.weekday().num_days_from_sunday() as i32;
            let base_prediction = *by_dow.get(&dow).unwrap_or(&10.0);

            // Add some variance
            let variance = base_prediction * 0.2;

            predictions.push(DailyPrediction {
                date,
                predicted_received: base_prediction as i32,
                predicted_sent: (base_prediction * 0.5) as i32,
                confidence_low: (base_prediction - variance) as i32,
                confidence_high: (base_prediction + variance) as i32,
                is_busy_day: base_prediction > 20.0,
            });
        }

        // Calculate weekly pattern
        let day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
        let mut by_day = HashMap::new();
        let mut max_day = (0, 0.0f32);
        let mut min_day = (0, f32::MAX);

        for (dow, avg) in &by_dow {
            let name = day_names[*dow as usize].to_string();
            by_day.insert(name, *avg);

            if *avg > max_day.1 {
                max_day = (*dow, *avg);
            }
            if *avg < min_day.1 {
                min_day = (*dow, *avg);
            }
        }

        let weekly_pattern = WeeklyPattern {
            busiest_day: day_names[max_day.0 as usize].to_string(),
            quietest_day: day_names[min_day.0 as usize].to_string(),
            by_day,
        };

        // Identify factors
        let factors = vec![
            ForecastFactor {
                factor: "Weekly Pattern".to_string(),
                impact: FactorImpact::Neutral,
                description: format!("{} tends to be your busiest day", weekly_pattern.busiest_day),
            },
        ];

        Ok(EmailForecast {
            predictions,
            weekly_pattern,
            confidence: 0.75,
            factors,
        })
    }
}

// ============================================================================
// Writing Analytics
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WritingAnalytics {
    pub avg_email_length: i32,
    pub avg_subject_length: i32,
    pub readability_score: f32, // Flesch reading ease
    pub tone_analysis: ToneAnalysis,
    pub common_phrases: Vec<CommonPhrase>,
    pub writing_time: WritingTimeStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToneAnalysis {
    pub formality_score: f32, // 0 = casual, 100 = formal
    pub sentiment_score: f32, // -100 to 100
    pub directness_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonPhrase {
    pub phrase: String,
    pub count: i32,
    pub category: PhraseCategory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhraseCategory {
    Greeting,
    Closing,
    Filler,
    Action,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WritingTimeStats {
    pub avg_compose_time_minutes: f32,
    pub fastest_email_seconds: i32,
    pub slowest_email_minutes: i32,
}

pub struct WritingAnalyticsService {
    pool: PgPool,
}

impl WritingAnalyticsService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn get_analytics(
        &self,
        user_id: Uuid,
        days: i32,
    ) -> Result<WritingAnalytics, ProductivityError> {
        let since = Utc::now() - Duration::days(days as i64);

        let stats = sqlx::query!(
            r#"
            SELECT
                AVG(LENGTH(body_text)) as avg_length,
                AVG(LENGTH(subject)) as avg_subject,
                AVG(EXTRACT(EPOCH FROM (sent_at - created_at)) / 60) as avg_compose_time,
                MIN(EXTRACT(EPOCH FROM (sent_at - created_at))) as min_compose,
                MAX(EXTRACT(EPOCH FROM (sent_at - created_at)) / 60) as max_compose
            FROM emails
            WHERE sender_id = $1 AND sent_at IS NOT NULL AND sent_at > $2
            "#,
            user_id,
            since
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(WritingAnalytics {
            avg_email_length: stats.avg_length.unwrap_or(0.0) as i32,
            avg_subject_length: stats.avg_subject.unwrap_or(0.0) as i32,
            readability_score: 60.0, // Would calculate Flesch score
            tone_analysis: ToneAnalysis {
                formality_score: 65.0,
                sentiment_score: 20.0,
                directness_score: 70.0,
            },
            common_phrases: vec![], // Would analyze actual content
            writing_time: WritingTimeStats {
                avg_compose_time_minutes: stats.avg_compose_time.unwrap_or(0.0) as f32,
                fastest_email_seconds: stats.min_compose.unwrap_or(0.0) as i32,
                slowest_email_minutes: stats.max_compose.unwrap_or(0.0) as i32,
            },
        })
    }
}

// ============================================================================
// Goals & Streaks
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserGoals {
    pub goals: Vec<ProductivityGoal>,
    pub streaks: Vec<Streak>,
    pub achievements: Vec<Achievement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductivityGoal {
    pub id: Uuid,
    pub goal_type: GoalType,
    pub target: i32,
    pub current: i32,
    pub period: GoalPeriod,
    pub progress_percent: f32,
    pub status: GoalStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GoalType {
    InboxZero,
    ResponseTime,
    ProcessedEmails,
    SentEmails,
    UnsubscribeCount,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GoalPeriod {
    Daily,
    Weekly,
    Monthly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GoalStatus {
    OnTrack,
    AtRisk,
    Achieved,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Streak {
    pub streak_type: StreakType,
    pub current_count: i32,
    pub best_count: i32,
    pub last_achieved: DateTime<Utc>,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreakType {
    InboxZero,
    QuickResponse, // Responded within 1 hour
    DailyProcessing, // Processed all emails
    NoUnread, // Ended day with no unread
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Achievement {
    pub id: String,
    pub name: String,
    pub description: String,
    pub icon: String,
    pub earned_at: DateTime<Utc>,
}

pub struct GoalsService {
    pool: PgPool,
}

impl GoalsService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn get_goals(&self, user_id: Uuid) -> Result<UserGoals, ProductivityError> {
        let goals = sqlx::query!(
            r#"
            SELECT id, goal_type, target, current_value, period, achieved_at
            FROM user_goals
            WHERE user_id = $1 AND (period = 'daily' AND created_at > NOW() - INTERVAL '1 day'
                OR period = 'weekly' AND created_at > NOW() - INTERVAL '7 days'
                OR period = 'monthly' AND created_at > NOW() - INTERVAL '30 days')
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| {
            let progress = row.current_value as f32 / row.target as f32 * 100.0;
            ProductivityGoal {
                id: row.id,
                goal_type: serde_json::from_str(&row.goal_type).unwrap_or(GoalType::InboxZero),
                target: row.target,
                current: row.current_value,
                period: serde_json::from_str(&row.period).unwrap_or(GoalPeriod::Daily),
                progress_percent: progress.min(100.0),
                status: if row.achieved_at.is_some() {
                    GoalStatus::Achieved
                } else if progress >= 75.0 {
                    GoalStatus::OnTrack
                } else if progress >= 50.0 {
                    GoalStatus::AtRisk
                } else {
                    GoalStatus::Failed
                },
            }
        })
        .collect();

        let streaks = sqlx::query!(
            r#"
            SELECT streak_type, current_count, best_count, last_achieved, is_active
            FROM user_streaks
            WHERE user_id = $1
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| Streak {
            streak_type: serde_json::from_str(&row.streak_type).unwrap_or(StreakType::InboxZero),
            current_count: row.current_count,
            best_count: row.best_count,
            last_achieved: row.last_achieved,
            is_active: row.is_active,
        })
        .collect();

        let achievements = sqlx::query!(
            r#"
            SELECT a.id, a.name, a.description, a.icon, ua.earned_at
            FROM user_achievements ua
            JOIN achievements a ON ua.achievement_id = a.id
            WHERE ua.user_id = $1
            ORDER BY ua.earned_at DESC
            LIMIT 20
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| Achievement {
            id: row.id,
            name: row.name,
            description: row.description,
            icon: row.icon,
            earned_at: row.earned_at,
        })
        .collect();

        Ok(UserGoals {
            goals,
            streaks,
            achievements,
        })
    }

    pub async fn set_goal(
        &self,
        user_id: Uuid,
        goal_type: GoalType,
        target: i32,
        period: GoalPeriod,
    ) -> Result<ProductivityGoal, ProductivityError> {
        let goal_id = Uuid::new_v4();

        sqlx::query!(
            r#"
            INSERT INTO user_goals (id, user_id, goal_type, target, current_value, period, created_at)
            VALUES ($1, $2, $3, $4, 0, $5, NOW())
            "#,
            goal_id,
            user_id,
            serde_json::to_string(&goal_type).unwrap(),
            target,
            serde_json::to_string(&period).unwrap()
        )
        .execute(&self.pool)
        .await?;

        Ok(ProductivityGoal {
            id: goal_id,
            goal_type,
            target,
            current: 0,
            period,
            progress_percent: 0.0,
            status: GoalStatus::OnTrack,
        })
    }

    pub async fn update_streak(
        &self,
        user_id: Uuid,
        streak_type: StreakType,
        achieved: bool,
    ) -> Result<Streak, ProductivityError> {
        if achieved {
            sqlx::query!(
                r#"
                INSERT INTO user_streaks (user_id, streak_type, current_count, best_count,
                                         last_achieved, is_active)
                VALUES ($1, $2, 1, 1, NOW(), true)
                ON CONFLICT (user_id, streak_type) DO UPDATE SET
                    current_count = CASE
                        WHEN user_streaks.last_achieved > NOW() - INTERVAL '1 day'
                        THEN user_streaks.current_count + 1
                        ELSE 1
                    END,
                    best_count = GREATEST(user_streaks.best_count, user_streaks.current_count + 1),
                    last_achieved = NOW(),
                    is_active = true
                "#,
                user_id,
                serde_json::to_string(&streak_type).unwrap()
            )
            .execute(&self.pool)
            .await?;
        } else {
            sqlx::query!(
                r#"
                UPDATE user_streaks
                SET is_active = false
                WHERE user_id = $1 AND streak_type = $2
                "#,
                user_id,
                serde_json::to_string(&streak_type).unwrap()
            )
            .execute(&self.pool)
            .await?;
        }

        let streak = sqlx::query!(
            r#"
            SELECT streak_type, current_count, best_count, last_achieved, is_active
            FROM user_streaks
            WHERE user_id = $1 AND streak_type = $2
            "#,
            user_id,
            serde_json::to_string(&streak_type).unwrap()
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(Streak {
            streak_type,
            current_count: streak.current_count,
            best_count: streak.best_count,
            last_achieved: streak.last_achieved,
            is_active: streak.is_active,
        })
    }
}
