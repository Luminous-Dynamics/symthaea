// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trust Decay Algorithms
//!
//! Trust scores decay over time without reinforcement, modeling the natural
//! degradation of trust in relationships that aren't maintained.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing::{info, warn};
use uuid::Uuid;

/// Decay function types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DecayFunction {
    /// Linear decay: score -= rate * days
    Linear,
    /// Exponential decay: score *= e^(-rate * days)
    Exponential,
    /// Sigmoid decay: slow start, fast middle, slow end
    Sigmoid,
    /// Step decay: drops at specific intervals
    Step,
    /// No decay (for permanent attestations)
    None,
}

/// Decay configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayConfig {
    /// Decay function to use
    pub function: DecayFunction,
    /// Rate of decay (interpretation depends on function)
    pub rate: f64,
    /// Minimum score (decay stops here)
    pub floor: f64,
    /// Days of inactivity before decay starts
    pub grace_period_days: i32,
    /// Context-specific decay rates
    pub context_rates: std::collections::HashMap<String, f64>,
}

impl Default for DecayConfig {
    fn default() -> Self {
        let mut context_rates = std::collections::HashMap::new();
        // Different contexts decay at different rates
        context_rates.insert("email_verified".to_string(), 0.001);  // Very slow
        context_rates.insert("identity_verified".to_string(), 0.0005);  // Slowest
        context_rates.insert("met_in_person".to_string(), 0.002);  // Slow
        context_rates.insert("professional".to_string(), 0.005);  // Medium
        context_rates.insert("transaction".to_string(), 0.01);  // Faster
        context_rates.insert("recommendation".to_string(), 0.02);  // Fast

        Self {
            function: DecayFunction::Exponential,
            rate: 0.005,  // ~0.5% per day base rate
            floor: 0.1,   // Never decay below 10%
            grace_period_days: 30,  // No decay for first 30 days
            context_rates,
        }
    }
}

/// Trust decay service
pub struct TrustDecayService {
    pool: PgPool,
    config: DecayConfig,
}

impl TrustDecayService {
    pub fn new(pool: PgPool, config: DecayConfig) -> Self {
        Self { pool, config }
    }

    /// Calculate decayed trust score
    pub fn calculate_decay(
        &self,
        original_score: f64,
        last_interaction: DateTime<Utc>,
        context: Option<&str>,
    ) -> f64 {
        let now = Utc::now();
        let days_elapsed = (now - last_interaction).num_days() as f64;

        // Apply grace period
        let effective_days = (days_elapsed - self.config.grace_period_days as f64).max(0.0);

        if effective_days <= 0.0 {
            return original_score;
        }

        // Get context-specific rate or default
        let rate = context
            .and_then(|c| self.config.context_rates.get(c))
            .copied()
            .unwrap_or(self.config.rate);

        let decayed = match self.config.function {
            DecayFunction::Linear => {
                original_score - (rate * effective_days)
            }
            DecayFunction::Exponential => {
                original_score * (-rate * effective_days).exp()
            }
            DecayFunction::Sigmoid => {
                // S-curve decay: slow-fast-slow
                let midpoint = 180.0;  // 6 months
                let steepness = 0.02;
                let sigmoid = 1.0 / (1.0 + (steepness * (effective_days - midpoint)).exp());
                original_score * sigmoid
            }
            DecayFunction::Step => {
                // Drop by 10% every 90 days
                let steps = (effective_days / 90.0).floor() as i32;
                original_score * (0.9_f64).powi(steps)
            }
            DecayFunction::None => original_score,
        };

        // Apply floor
        decayed.max(self.config.floor)
    }

    /// Apply decay to all trust scores in the database
    pub async fn apply_decay_batch(&self) -> Result<DecayBatchResult, sqlx::Error> {
        let mut updated = 0;
        let mut skipped = 0;

        // Get all trust scores that need decay calculation
        let scores: Vec<TrustScoreRecord> = sqlx::query_as(
            r#"
            SELECT id, entity_id, context, score, last_interaction, last_decay_at
            FROM trust_scores
            WHERE last_interaction < NOW() - INTERVAL '1 day' * $1
              AND (last_decay_at IS NULL OR last_decay_at < NOW() - INTERVAL '1 day')
            LIMIT 1000
            "#,
        )
        .bind(self.config.grace_period_days)
        .fetch_all(&self.pool)
        .await?;

        for record in scores {
            let decayed_score = self.calculate_decay(
                record.score,
                record.last_interaction,
                record.context.as_deref(),
            );

            // Only update if score actually changed
            if (decayed_score - record.score).abs() > 0.001 {
                sqlx::query(
                    r#"
                    UPDATE trust_scores
                    SET score = $2,
                        last_decay_at = NOW(),
                        decay_applied = COALESCE(decay_applied, 0) + 1
                    WHERE id = $1
                    "#,
                )
                .bind(record.id)
                .bind(decayed_score)
                .execute(&self.pool)
                .await?;

                updated += 1;

                // Log significant decays
                if record.score - decayed_score > 0.1 {
                    info!(
                        entity_id = %record.entity_id,
                        old_score = record.score,
                        new_score = decayed_score,
                        "Significant trust decay applied"
                    );
                }
            } else {
                skipped += 1;
            }
        }

        Ok(DecayBatchResult {
            processed: updated + skipped,
            updated,
            skipped,
        })
    }

    /// Reinforce trust (reset decay timer)
    pub async fn reinforce_trust(
        &self,
        entity_id: Uuid,
        context: &str,
        boost: Option<f64>,
    ) -> Result<f64, sqlx::Error> {
        let boost_amount = boost.unwrap_or(0.0).min(0.2);  // Max 20% boost

        let result: (f64,) = sqlx::query_as(
            r#"
            UPDATE trust_scores
            SET last_interaction = NOW(),
                score = LEAST(1.0, score + $3),
                reinforcement_count = COALESCE(reinforcement_count, 0) + 1
            WHERE entity_id = $1 AND context = $2
            RETURNING score
            "#,
        )
        .bind(entity_id)
        .bind(context)
        .bind(boost_amount)
        .fetch_one(&self.pool)
        .await?;

        Ok(result.0)
    }

    /// Get decay forecast for an entity
    pub fn forecast_decay(
        &self,
        current_score: f64,
        last_interaction: DateTime<Utc>,
        context: Option<&str>,
        days_ahead: i32,
    ) -> Vec<DecayForecast> {
        let mut forecasts = Vec::new();
        let now = Utc::now();

        for day in 0..=days_ahead {
            let future_date = now + Duration::days(day as i64);
            // Simulate as if last_interaction stays the same
            let adjusted_last = last_interaction;
            let days_from_now = (future_date - adjusted_last).num_days() as f64;

            let effective_days = (days_from_now - self.config.grace_period_days as f64).max(0.0);

            let rate = context
                .and_then(|c| self.config.context_rates.get(c))
                .copied()
                .unwrap_or(self.config.rate);

            let projected_score = match self.config.function {
                DecayFunction::Exponential => {
                    current_score * (-rate * effective_days).exp()
                }
                DecayFunction::Linear => {
                    current_score - (rate * effective_days)
                }
                _ => current_score,
            }.max(self.config.floor);

            forecasts.push(DecayForecast {
                date: future_date,
                projected_score,
                days_from_now: day,
            });
        }

        forecasts
    }
}

#[derive(Debug, Clone, sqlx::FromRow)]
struct TrustScoreRecord {
    id: Uuid,
    entity_id: Uuid,
    context: Option<String>,
    score: f64,
    last_interaction: DateTime<Utc>,
    last_decay_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayBatchResult {
    pub processed: i32,
    pub updated: i32,
    pub skipped: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayForecast {
    pub date: DateTime<Utc>,
    pub projected_score: f64,
    pub days_from_now: i32,
}

/// Trust reinforcement triggers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReinforcementTrigger {
    /// Received email from entity
    EmailReceived,
    /// Sent email to entity
    EmailSent,
    /// Email marked as important
    MarkedImportant,
    /// Added to contacts
    AddedToContacts,
    /// Received attestation
    AttestationReceived,
    /// Manual trust boost
    ManualBoost,
}

impl ReinforcementTrigger {
    /// Get the boost amount for this trigger
    pub fn boost_amount(&self) -> f64 {
        match self {
            Self::EmailReceived => 0.01,
            Self::EmailSent => 0.02,
            Self::MarkedImportant => 0.05,
            Self::AddedToContacts => 0.1,
            Self::AttestationReceived => 0.15,
            Self::ManualBoost => 0.1,
        }
    }
}

/// Background worker for periodic decay application
pub async fn decay_worker(
    pool: PgPool,
    config: DecayConfig,
    mut shutdown: tokio::sync::mpsc::Receiver<()>,
) {
    let service = TrustDecayService::new(pool, config);

    info!("Trust decay worker started");

    loop {
        tokio::select! {
            _ = shutdown.recv() => {
                info!("Trust decay worker shutting down");
                break;
            }
            _ = tokio::time::sleep(std::time::Duration::from_secs(3600)) => {
                // Run decay every hour
                match service.apply_decay_batch().await {
                    Ok(result) => {
                        info!(
                            processed = result.processed,
                            updated = result.updated,
                            "Trust decay batch completed"
                        );
                    }
                    Err(e) => {
                        warn!(error = %e, "Trust decay batch failed");
                    }
                }
            }
        }
    }
}

/// Migration for decay tracking
pub const DECAY_MIGRATION: &str = r#"
ALTER TABLE trust_scores
ADD COLUMN IF NOT EXISTS last_interaction TIMESTAMPTZ DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS last_decay_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS decay_applied INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS reinforcement_count INTEGER DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_trust_scores_decay
    ON trust_scores(last_interaction, last_decay_at)
    WHERE last_decay_at IS NULL OR last_decay_at < NOW() - INTERVAL '1 day';
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_decay() {
        let config = DecayConfig::default();
        let service = TrustDecayService::new(
            // Mock pool not needed for calculation tests
            unsafe { std::mem::zeroed() },
            config,
        );

        let original = 1.0;
        let last_interaction = Utc::now() - Duration::days(60);

        let decayed = service.calculate_decay(original, last_interaction, None);

        // Should be less than original but above floor
        assert!(decayed < original);
        assert!(decayed >= 0.1);
    }

    #[test]
    fn test_grace_period() {
        let config = DecayConfig {
            grace_period_days: 30,
            ..Default::default()
        };
        let service = TrustDecayService::new(
            unsafe { std::mem::zeroed() },
            config,
        );

        let original = 1.0;
        let last_interaction = Utc::now() - Duration::days(15);  // Within grace period

        let decayed = service.calculate_decay(original, last_interaction, None);

        // Should be unchanged during grace period
        assert!((decayed - original).abs() < 0.001);
    }

    #[test]
    fn test_floor() {
        let config = DecayConfig {
            floor: 0.2,
            rate: 1.0,  // Very high decay rate
            grace_period_days: 0,
            ..Default::default()
        };
        let service = TrustDecayService::new(
            unsafe { std::mem::zeroed() },
            config,
        );

        let original = 1.0;
        let last_interaction = Utc::now() - Duration::days(365);

        let decayed = service.calculate_decay(original, last_interaction, None);

        // Should not go below floor
        assert!((decayed - 0.2).abs() < 0.001);
    }
}
