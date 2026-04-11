// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Security Features
//!
//! Hardware key authentication, security auditing, and threat detection

pub mod webauthn;
pub mod phishing;
pub mod audit;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

// ============================================================================
// Security Audit Dashboard
// ============================================================================

/// Security event types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "security_event_type", rename_all = "snake_case")]
pub enum SecurityEventType {
    LoginSuccess,
    LoginFailed,
    LoginSuspicious,
    PasswordChanged,
    MfaEnabled,
    MfaDisabled,
    HardwareKeyAdded,
    HardwareKeyRemoved,
    SessionRevoked,
    PermissionChanged,
    DataExported,
    AccountLocked,
    AccountUnlocked,
    ApiKeyCreated,
    ApiKeyRevoked,
    PhishingDetected,
    MalwareBlocked,
}

/// Security event
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct SecurityEvent {
    pub id: Uuid,
    pub user_id: Uuid,
    pub event_type: SecurityEventType,
    pub severity: SecuritySeverity,
    pub description: String,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub location: Option<String>,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "security_severity", rename_all = "lowercase")]
pub enum SecuritySeverity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

/// Security metrics for dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetrics {
    pub total_events_24h: i64,
    pub failed_logins_24h: i64,
    pub suspicious_activities: i64,
    pub active_sessions: i64,
    pub mfa_enabled_users: i64,
    pub hardware_key_users: i64,
    pub blocked_threats: i64,
    pub risk_score: f64,
}

/// User security status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSecurityStatus {
    pub user_id: Uuid,
    pub mfa_enabled: bool,
    pub hardware_keys_count: i32,
    pub last_password_change: Option<DateTime<Utc>>,
    pub last_login: Option<DateTime<Utc>>,
    pub active_sessions: i32,
    pub recent_suspicious_events: i32,
    pub security_score: i32,
    pub recommendations: Vec<SecurityRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRecommendation {
    pub id: String,
    pub priority: SecuritySeverity,
    pub title: String,
    pub description: String,
    pub action_url: Option<String>,
}

/// Security audit service
pub struct SecurityAuditService {
    pool: PgPool,
}

impl SecurityAuditService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Log security event
    pub async fn log_event(
        &self,
        user_id: Uuid,
        event_type: SecurityEventType,
        description: &str,
        ip_address: Option<&str>,
        user_agent: Option<&str>,
        metadata: serde_json::Value,
    ) -> Result<SecurityEvent, sqlx::Error> {
        let severity = Self::determine_severity(event_type);

        let event = sqlx::query_as::<_, SecurityEvent>(
            r#"
            INSERT INTO security_events (
                id, user_id, event_type, severity, description,
                ip_address, user_agent, metadata, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
            RETURNING *
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(user_id)
        .bind(event_type)
        .bind(severity)
        .bind(description)
        .bind(ip_address)
        .bind(user_agent)
        .bind(&metadata)
        .fetch_one(&self.pool)
        .await?;

        // Check for alert conditions
        if severity == SecuritySeverity::High || severity == SecuritySeverity::Critical {
            self.trigger_security_alert(&event).await?;
        }

        Ok(event)
    }

    fn determine_severity(event_type: SecurityEventType) -> SecuritySeverity {
        match event_type {
            SecurityEventType::LoginSuccess
            | SecurityEventType::PasswordChanged
            | SecurityEventType::MfaEnabled
            | SecurityEventType::HardwareKeyAdded => SecuritySeverity::Info,

            SecurityEventType::LoginFailed
            | SecurityEventType::SessionRevoked
            | SecurityEventType::ApiKeyCreated => SecuritySeverity::Low,

            SecurityEventType::MfaDisabled
            | SecurityEventType::HardwareKeyRemoved
            | SecurityEventType::ApiKeyRevoked
            | SecurityEventType::DataExported => SecuritySeverity::Medium,

            SecurityEventType::LoginSuspicious
            | SecurityEventType::PermissionChanged
            | SecurityEventType::AccountLocked => SecuritySeverity::High,

            SecurityEventType::PhishingDetected
            | SecurityEventType::MalwareBlocked
            | SecurityEventType::AccountUnlocked => SecuritySeverity::Critical,
        }
    }

    async fn trigger_security_alert(&self, event: &SecurityEvent) -> Result<(), sqlx::Error> {
        // Create notification for user
        sqlx::query(
            r#"
            INSERT INTO notifications (id, user_id, type, title, body, data, created_at)
            VALUES ($1, $2, 'security_alert', $3, $4, $5, NOW())
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(event.user_id)
        .bind(format!("Security Alert: {:?}", event.event_type))
        .bind(&event.description)
        .bind(serde_json::json!({
            "event_id": event.id,
            "event_type": event.event_type,
            "severity": event.severity,
        }))
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get security metrics
    pub async fn get_metrics(&self, tenant_id: Option<Uuid>) -> Result<SecurityMetrics, sqlx::Error> {
        let metrics: SecurityMetricsRow = sqlx::query_as(
            r#"
            WITH recent_events AS (
                SELECT * FROM security_events
                WHERE created_at > NOW() - INTERVAL '24 hours'
            )
            SELECT
                (SELECT COUNT(*) FROM recent_events) as total_events_24h,
                (SELECT COUNT(*) FROM recent_events WHERE event_type = 'login_failed') as failed_logins_24h,
                (SELECT COUNT(*) FROM recent_events WHERE severity IN ('high', 'critical')) as suspicious_activities,
                (SELECT COUNT(*) FROM user_sessions WHERE expires_at > NOW()) as active_sessions,
                (SELECT COUNT(*) FROM users WHERE mfa_enabled = true) as mfa_enabled_users,
                (SELECT COUNT(DISTINCT user_id) FROM hardware_keys) as hardware_key_users,
                (SELECT COUNT(*) FROM recent_events WHERE event_type IN ('phishing_detected', 'malware_blocked')) as blocked_threats
            "#,
        )
        .fetch_one(&self.pool)
        .await?;

        // Calculate risk score (0-100)
        let risk_score = self.calculate_risk_score(&metrics);

        Ok(SecurityMetrics {
            total_events_24h: metrics.total_events_24h,
            failed_logins_24h: metrics.failed_logins_24h,
            suspicious_activities: metrics.suspicious_activities,
            active_sessions: metrics.active_sessions,
            mfa_enabled_users: metrics.mfa_enabled_users,
            hardware_key_users: metrics.hardware_key_users,
            blocked_threats: metrics.blocked_threats,
            risk_score,
        })
    }

    fn calculate_risk_score(&self, metrics: &SecurityMetricsRow) -> f64 {
        let mut score = 0.0;

        // Failed logins contribute to risk
        if metrics.failed_logins_24h > 100 {
            score += 20.0;
        } else if metrics.failed_logins_24h > 50 {
            score += 10.0;
        }

        // Suspicious activities
        score += (metrics.suspicious_activities as f64 * 5.0).min(30.0);

        // Blocked threats
        score += (metrics.blocked_threats as f64 * 3.0).min(20.0);

        // Low MFA adoption increases risk
        let mfa_ratio = if metrics.active_sessions > 0 {
            metrics.mfa_enabled_users as f64 / metrics.active_sessions as f64
        } else {
            1.0
        };
        if mfa_ratio < 0.5 {
            score += 15.0;
        }

        score.min(100.0)
    }

    /// Get user security status
    pub async fn get_user_status(&self, user_id: Uuid) -> Result<UserSecurityStatus, sqlx::Error> {
        let status: UserStatusRow = sqlx::query_as(
            r#"
            SELECT
                u.id as user_id,
                u.mfa_enabled,
                (SELECT COUNT(*) FROM hardware_keys WHERE user_id = u.id) as hardware_keys_count,
                u.password_changed_at as last_password_change,
                (SELECT MAX(created_at) FROM security_events WHERE user_id = u.id AND event_type = 'login_success') as last_login,
                (SELECT COUNT(*) FROM user_sessions WHERE user_id = u.id AND expires_at > NOW()) as active_sessions,
                (SELECT COUNT(*) FROM security_events WHERE user_id = u.id AND severity IN ('high', 'critical') AND created_at > NOW() - INTERVAL '7 days') as recent_suspicious_events
            FROM users u
            WHERE u.id = $1
            "#,
        )
        .bind(user_id)
        .fetch_one(&self.pool)
        .await?;

        let recommendations = self.generate_recommendations(&status);
        let security_score = self.calculate_user_security_score(&status);

        Ok(UserSecurityStatus {
            user_id: status.user_id,
            mfa_enabled: status.mfa_enabled,
            hardware_keys_count: status.hardware_keys_count,
            last_password_change: status.last_password_change,
            last_login: status.last_login,
            active_sessions: status.active_sessions,
            recent_suspicious_events: status.recent_suspicious_events,
            security_score,
            recommendations,
        })
    }

    fn generate_recommendations(&self, status: &UserStatusRow) -> Vec<SecurityRecommendation> {
        let mut recommendations = Vec::new();

        if !status.mfa_enabled {
            recommendations.push(SecurityRecommendation {
                id: "enable_mfa".to_string(),
                priority: SecuritySeverity::High,
                title: "Enable Two-Factor Authentication".to_string(),
                description: "Add an extra layer of security to your account".to_string(),
                action_url: Some("/settings/security/mfa".to_string()),
            });
        }

        if status.hardware_keys_count == 0 {
            recommendations.push(SecurityRecommendation {
                id: "add_hardware_key".to_string(),
                priority: SecuritySeverity::Medium,
                title: "Add a Hardware Security Key".to_string(),
                description: "Hardware keys provide the strongest protection against phishing".to_string(),
                action_url: Some("/settings/security/hardware-keys".to_string()),
            });
        }

        if let Some(last_change) = status.last_password_change {
            let days_since = (Utc::now() - last_change).num_days();
            if days_since > 180 {
                recommendations.push(SecurityRecommendation {
                    id: "change_password".to_string(),
                    priority: SecuritySeverity::Low,
                    title: "Update Your Password".to_string(),
                    description: format!("It's been {} days since your last password change", days_since),
                    action_url: Some("/settings/security/password".to_string()),
                });
            }
        }

        if status.recent_suspicious_events > 0 {
            recommendations.push(SecurityRecommendation {
                id: "review_activity".to_string(),
                priority: SecuritySeverity::High,
                title: "Review Recent Security Events".to_string(),
                description: "There have been suspicious activities on your account".to_string(),
                action_url: Some("/settings/security/activity".to_string()),
            });
        }

        recommendations
    }

    fn calculate_user_security_score(&self, status: &UserStatusRow) -> i32 {
        let mut score = 50; // Base score

        if status.mfa_enabled {
            score += 25;
        }

        if status.hardware_keys_count > 0 {
            score += 15;
        }

        if let Some(last_change) = status.last_password_change {
            let days_since = (Utc::now() - last_change).num_days();
            if days_since < 90 {
                score += 5;
            }
        }

        score -= status.recent_suspicious_events * 10;

        score.clamp(0, 100)
    }

    /// Get recent security events
    pub async fn get_recent_events(
        &self,
        user_id: Uuid,
        limit: i32,
    ) -> Result<Vec<SecurityEvent>, sqlx::Error> {
        sqlx::query_as::<_, SecurityEvent>(
            r#"
            SELECT * FROM security_events
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            "#,
        )
        .bind(user_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
    }
}

#[derive(sqlx::FromRow)]
struct SecurityMetricsRow {
    total_events_24h: i64,
    failed_logins_24h: i64,
    suspicious_activities: i64,
    active_sessions: i64,
    mfa_enabled_users: i64,
    hardware_key_users: i64,
    blocked_threats: i64,
}

#[derive(sqlx::FromRow)]
struct UserStatusRow {
    user_id: Uuid,
    mfa_enabled: bool,
    hardware_keys_count: i32,
    last_password_change: Option<DateTime<Utc>>,
    last_login: Option<DateTime<Utc>>,
    active_sessions: i32,
    recent_suspicious_events: i32,
}

/// Migration for security features
pub const SECURITY_MIGRATION: &str = r#"
DO $$ BEGIN
    CREATE TYPE security_event_type AS ENUM (
        'login_success', 'login_failed', 'login_suspicious',
        'password_changed', 'mfa_enabled', 'mfa_disabled',
        'hardware_key_added', 'hardware_key_removed',
        'session_revoked', 'permission_changed', 'data_exported',
        'account_locked', 'account_unlocked',
        'api_key_created', 'api_key_revoked',
        'phishing_detected', 'malware_blocked'
    );
EXCEPTION WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE security_severity AS ENUM ('info', 'low', 'medium', 'high', 'critical');
EXCEPTION WHEN duplicate_object THEN null;
END $$;

CREATE TABLE IF NOT EXISTS security_events (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    event_type security_event_type NOT NULL,
    severity security_severity NOT NULL,
    description TEXT NOT NULL,
    ip_address INET,
    user_agent TEXT,
    location VARCHAR(255),
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_security_events_user ON security_events(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_security_events_type ON security_events(event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_security_events_severity ON security_events(severity, created_at DESC);

-- Hardware Keys (WebAuthn)
CREATE TABLE IF NOT EXISTS hardware_keys (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    credential_id BYTEA NOT NULL UNIQUE,
    public_key BYTEA NOT NULL,
    counter INTEGER NOT NULL DEFAULT 0,
    transports TEXT[],
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_hardware_keys_user ON hardware_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_hardware_keys_credential ON hardware_keys(credential_id);

-- Add mfa_enabled and password_changed_at to users if not exists
ALTER TABLE users ADD COLUMN IF NOT EXISTS mfa_enabled BOOLEAN DEFAULT false;
ALTER TABLE users ADD COLUMN IF NOT EXISTS password_changed_at TIMESTAMPTZ;
"#;
