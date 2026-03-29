// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trust Recommendations
//!
//! Suggests trust actions based on network analysis and user behavior

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use uuid::Uuid;

/// Types of trust recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TrustRecommendation {
    /// Suggest verifying a frequently contacted person
    VerifyContact {
        contact_id: Uuid,
        email: String,
        name: Option<String>,
        interaction_count: i32,
        reason: String,
    },

    /// Suggest attesting to someone vouched by trusted contacts
    AttestFromNetwork {
        contact_id: Uuid,
        email: String,
        vouchers: Vec<VoucherInfo>,
        suggested_context: String,
    },

    /// Warning about trust decay
    TrustDecaying {
        contact_id: Uuid,
        email: String,
        current_score: f64,
        days_since_interaction: i32,
        action_suggestion: String,
    },

    /// Suggest revoking attestation due to inactivity or issues
    ConsiderRevocation {
        attestation_id: Uuid,
        subject_email: String,
        reason: String,
        last_interaction: DateTime<Utc>,
    },

    /// Mutual trust opportunity
    MutualTrust {
        contact_id: Uuid,
        email: String,
        their_trust_in_you: f64,
        your_trust_in_them: f64,
    },

    /// Suggested connection based on shared trusted contacts
    SuggestedConnection {
        contact_id: Uuid,
        email: String,
        mutual_contacts: Vec<String>,
        estimated_trust: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoucherInfo {
    pub email: String,
    pub name: Option<String>,
    pub trust_score: f64,
    pub attestation_context: String,
}

/// Recommendation priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrioritizedRecommendation {
    pub id: Uuid,
    pub recommendation: TrustRecommendation,
    pub priority: RecommendationPriority,
    pub created_at: DateTime<Utc>,
    pub dismissed: bool,
}

/// Trust recommendation service
pub struct TrustRecommendationService {
    pool: PgPool,
}

impl TrustRecommendationService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Generate all recommendations for a user
    pub async fn generate_recommendations(
        &self,
        user_id: Uuid,
    ) -> Result<Vec<PrioritizedRecommendation>, sqlx::Error> {
        let mut recommendations = Vec::new();

        // Get frequent contacts without verification
        let unverified = self.find_unverified_frequent_contacts(user_id).await?;
        recommendations.extend(unverified);

        // Find network-based attestation opportunities
        let network_based = self.find_network_attestation_opportunities(user_id).await?;
        recommendations.extend(network_based);

        // Check for decaying trust
        let decaying = self.find_decaying_trust(user_id).await?;
        recommendations.extend(decaying);

        // Find mutual trust opportunities
        let mutual = self.find_mutual_trust_opportunities(user_id).await?;
        recommendations.extend(mutual);

        // Sort by priority
        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Limit to top recommendations
        recommendations.truncate(20);

        Ok(recommendations)
    }

    /// Find frequently contacted people without trust verification
    async fn find_unverified_frequent_contacts(
        &self,
        user_id: Uuid,
    ) -> Result<Vec<PrioritizedRecommendation>, sqlx::Error> {
        let contacts: Vec<FrequentContact> = sqlx::query_as(
            r#"
            SELECT
                c.id as contact_id,
                c.email,
                c.name,
                COUNT(e.id) as interaction_count,
                COALESCE(ts.score, 0) as trust_score
            FROM contacts c
            LEFT JOIN emails e ON (
                e.user_id = $1 AND
                (e.from_address = c.email OR c.email = ANY(e.to_addresses))
            )
            LEFT JOIN trust_scores ts ON (
                ts.entity_id = c.id AND ts.context = 'global'
            )
            WHERE c.user_id = $1
            GROUP BY c.id, c.email, c.name, ts.score
            HAVING COUNT(e.id) >= 10 AND COALESCE(ts.score, 0) < 0.3
            ORDER BY COUNT(e.id) DESC
            LIMIT 10
            "#,
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(contacts
            .into_iter()
            .map(|c| PrioritizedRecommendation {
                id: Uuid::new_v4(),
                recommendation: TrustRecommendation::VerifyContact {
                    contact_id: c.contact_id,
                    email: c.email,
                    name: c.name,
                    interaction_count: c.interaction_count,
                    reason: format!(
                        "You've exchanged {} emails but haven't verified their identity",
                        c.interaction_count
                    ),
                },
                priority: if c.interaction_count > 50 {
                    RecommendationPriority::High
                } else {
                    RecommendationPriority::Medium
                },
                created_at: Utc::now(),
                dismissed: false,
            })
            .collect())
    }

    /// Find attestation opportunities from trusted network
    async fn find_network_attestation_opportunities(
        &self,
        user_id: Uuid,
    ) -> Result<Vec<PrioritizedRecommendation>, sqlx::Error> {
        // Find contacts that trusted people have attested to
        let opportunities: Vec<NetworkOpportunity> = sqlx::query_as(
            r#"
            WITH my_trusted AS (
                SELECT entity_id, score
                FROM trust_scores
                WHERE user_id = $1 AND score >= 0.7
            ),
            vouched_contacts AS (
                SELECT
                    a.subject_id,
                    c.email,
                    c.name,
                    ARRAY_AGG(DISTINCT tc.email) as voucher_emails,
                    ARRAY_AGG(DISTINCT a.context) as contexts,
                    AVG(mt.score) as avg_voucher_trust
                FROM attestations a
                JOIN my_trusted mt ON mt.entity_id = a.issuer_id
                JOIN contacts c ON c.id = a.subject_id
                LEFT JOIN contacts tc ON tc.id = a.issuer_id
                WHERE a.revoked_at IS NULL
                  AND a.subject_id NOT IN (
                      SELECT entity_id FROM trust_scores WHERE user_id = $1
                  )
                GROUP BY a.subject_id, c.email, c.name
                HAVING COUNT(DISTINCT a.issuer_id) >= 2
            )
            SELECT * FROM vouched_contacts
            ORDER BY avg_voucher_trust DESC
            LIMIT 5
            "#,
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(opportunities
            .into_iter()
            .map(|o| PrioritizedRecommendation {
                id: Uuid::new_v4(),
                recommendation: TrustRecommendation::AttestFromNetwork {
                    contact_id: o.subject_id,
                    email: o.email,
                    vouchers: o.voucher_emails
                        .iter()
                        .zip(o.contexts.iter())
                        .map(|(email, context)| VoucherInfo {
                            email: email.clone(),
                            name: None,
                            trust_score: o.avg_voucher_trust,
                            attestation_context: context.clone(),
                        })
                        .collect(),
                    suggested_context: o.contexts.first().cloned().unwrap_or_default(),
                },
                priority: RecommendationPriority::Medium,
                created_at: Utc::now(),
                dismissed: false,
            })
            .collect())
    }

    /// Find contacts with decaying trust
    async fn find_decaying_trust(
        &self,
        user_id: Uuid,
    ) -> Result<Vec<PrioritizedRecommendation>, sqlx::Error> {
        let decaying: Vec<DecayingTrust> = sqlx::query_as(
            r#"
            SELECT
                ts.entity_id as contact_id,
                c.email,
                ts.score as current_score,
                EXTRACT(DAY FROM NOW() - ts.last_interaction)::int as days_since
            FROM trust_scores ts
            JOIN contacts c ON c.id = ts.entity_id
            WHERE ts.user_id = $1
              AND ts.score >= 0.5
              AND ts.last_interaction < NOW() - INTERVAL '60 days'
            ORDER BY ts.score DESC
            LIMIT 5
            "#,
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(decaying
            .into_iter()
            .map(|d| PrioritizedRecommendation {
                id: Uuid::new_v4(),
                recommendation: TrustRecommendation::TrustDecaying {
                    contact_id: d.contact_id,
                    email: d.email,
                    current_score: d.current_score,
                    days_since_interaction: d.days_since,
                    action_suggestion: if d.days_since > 90 {
                        "Consider reaching out to maintain this connection".to_string()
                    } else {
                        "A quick email could reinforce this trust relationship".to_string()
                    },
                },
                priority: if d.current_score > 0.8 && d.days_since > 90 {
                    RecommendationPriority::High
                } else {
                    RecommendationPriority::Low
                },
                created_at: Utc::now(),
                dismissed: false,
            })
            .collect())
    }

    /// Find mutual trust opportunities
    async fn find_mutual_trust_opportunities(
        &self,
        user_id: Uuid,
    ) -> Result<Vec<PrioritizedRecommendation>, sqlx::Error> {
        let mutual: Vec<MutualTrust> = sqlx::query_as(
            r#"
            SELECT
                ts_theirs.user_id as contact_id,
                c.email,
                ts_theirs.score as their_trust_in_you,
                COALESCE(ts_mine.score, 0) as your_trust_in_them
            FROM trust_scores ts_theirs
            JOIN contacts c ON c.user_id = ts_theirs.user_id
            LEFT JOIN trust_scores ts_mine ON (
                ts_mine.user_id = $1 AND
                ts_mine.entity_id = ts_theirs.user_id
            )
            WHERE ts_theirs.entity_id = $1
              AND ts_theirs.score >= 0.7
              AND COALESCE(ts_mine.score, 0) < 0.5
            ORDER BY ts_theirs.score DESC
            LIMIT 5
            "#,
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(mutual
            .into_iter()
            .map(|m| PrioritizedRecommendation {
                id: Uuid::new_v4(),
                recommendation: TrustRecommendation::MutualTrust {
                    contact_id: m.contact_id,
                    email: m.email,
                    their_trust_in_you: m.their_trust_in_you,
                    your_trust_in_them: m.your_trust_in_them,
                },
                priority: RecommendationPriority::Medium,
                created_at: Utc::now(),
                dismissed: false,
            })
            .collect())
    }

    /// Dismiss a recommendation
    pub async fn dismiss_recommendation(
        &self,
        user_id: Uuid,
        recommendation_id: Uuid,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            INSERT INTO dismissed_recommendations (user_id, recommendation_id, dismissed_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (user_id, recommendation_id) DO NOTHING
            "#,
        )
        .bind(user_id)
        .bind(recommendation_id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get active (non-dismissed) recommendations
    pub async fn get_active_recommendations(
        &self,
        user_id: Uuid,
        limit: i32,
    ) -> Result<Vec<PrioritizedRecommendation>, sqlx::Error> {
        let all = self.generate_recommendations(user_id).await?;

        // Filter out dismissed
        let dismissed: Vec<Uuid> = sqlx::query_scalar(
            "SELECT recommendation_id FROM dismissed_recommendations WHERE user_id = $1",
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(all
            .into_iter()
            .filter(|r| !dismissed.contains(&r.id))
            .take(limit as usize)
            .collect())
    }
}

// Query result types
#[derive(Debug, sqlx::FromRow)]
struct FrequentContact {
    contact_id: Uuid,
    email: String,
    name: Option<String>,
    interaction_count: i32,
    trust_score: f64,
}

#[derive(Debug, sqlx::FromRow)]
struct NetworkOpportunity {
    subject_id: Uuid,
    email: String,
    name: Option<String>,
    voucher_emails: Vec<String>,
    contexts: Vec<String>,
    avg_voucher_trust: f64,
}

#[derive(Debug, sqlx::FromRow)]
struct DecayingTrust {
    contact_id: Uuid,
    email: String,
    current_score: f64,
    days_since: i32,
}

#[derive(Debug, sqlx::FromRow)]
struct MutualTrust {
    contact_id: Uuid,
    email: String,
    their_trust_in_you: f64,
    your_trust_in_them: f64,
}

/// Migration for recommendations
pub const RECOMMENDATIONS_MIGRATION: &str = r#"
CREATE TABLE IF NOT EXISTS dismissed_recommendations (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    recommendation_id UUID NOT NULL,
    dismissed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, recommendation_id)
);

CREATE INDEX IF NOT EXISTS idx_dismissed_recommendations_user
    ON dismissed_recommendations(user_id);
"#;
