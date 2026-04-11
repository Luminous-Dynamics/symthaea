// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Custom Metrics for Mycelix Mail
//!
//! Defines application-specific metrics using OpenTelemetry.

use once_cell::sync::Lazy;
use opentelemetry::{
    global,
    metrics::{Counter, Histogram, UpDownCounter, Unit},
    KeyValue,
};
use std::time::Instant;

// ============================================================================
// Metric Definitions
// ============================================================================

/// HTTP request metrics
pub struct HttpMetrics {
    pub requests_total: Counter<u64>,
    pub request_duration: Histogram<f64>,
    pub requests_in_flight: UpDownCounter<i64>,
    pub response_size: Histogram<u64>,
}

impl HttpMetrics {
    fn new() -> Self {
        let meter = global::meter("mycelix.http");

        Self {
            requests_total: meter
                .u64_counter("http_requests_total")
                .with_description("Total number of HTTP requests")
                .init(),
            request_duration: meter
                .f64_histogram("http_request_duration_seconds")
                .with_description("HTTP request duration in seconds")
                .with_unit(Unit::new("s"))
                .init(),
            requests_in_flight: meter
                .i64_up_down_counter("http_requests_in_flight")
                .with_description("Number of HTTP requests currently being processed")
                .init(),
            response_size: meter
                .u64_histogram("http_response_size_bytes")
                .with_description("HTTP response size in bytes")
                .with_unit(Unit::new("By"))
                .init(),
        }
    }
}

pub static HTTP_METRICS: Lazy<HttpMetrics> = Lazy::new(HttpMetrics::new);

/// Email processing metrics
pub struct EmailMetrics {
    pub emails_received: Counter<u64>,
    pub emails_sent: Counter<u64>,
    pub emails_processed: Counter<u64>,
    pub email_processing_duration: Histogram<f64>,
    pub email_size: Histogram<u64>,
    pub attachments_processed: Counter<u64>,
    pub spam_detected: Counter<u64>,
    pub encryption_operations: Counter<u64>,
    pub delivery_failures: Counter<u64>,
    pub queue_length: UpDownCounter<i64>,
}

impl EmailMetrics {
    fn new() -> Self {
        let meter = global::meter("mycelix.email");

        Self {
            emails_received: meter
                .u64_counter("emails_received_total")
                .with_description("Total emails received")
                .init(),
            emails_sent: meter
                .u64_counter("emails_sent_total")
                .with_description("Total emails sent")
                .init(),
            emails_processed: meter
                .u64_counter("emails_processed_total")
                .with_description("Total emails processed")
                .init(),
            email_processing_duration: meter
                .f64_histogram("email_processing_duration_seconds")
                .with_description("Email processing duration")
                .with_unit(Unit::new("s"))
                .init(),
            email_size: meter
                .u64_histogram("email_size_bytes")
                .with_description("Email size in bytes")
                .with_unit(Unit::new("By"))
                .init(),
            attachments_processed: meter
                .u64_counter("attachments_processed_total")
                .with_description("Total attachments processed")
                .init(),
            spam_detected: meter
                .u64_counter("spam_detected_total")
                .with_description("Total spam emails detected")
                .init(),
            encryption_operations: meter
                .u64_counter("encryption_operations_total")
                .with_description("Total encryption/decryption operations")
                .init(),
            delivery_failures: meter
                .u64_counter("email_delivery_failures_total")
                .with_description("Total email delivery failures")
                .init(),
            queue_length: meter
                .i64_up_down_counter("email_queue_length")
                .with_description("Current email queue length")
                .init(),
        }
    }
}

pub static EMAIL_METRICS: Lazy<EmailMetrics> = Lazy::new(EmailMetrics::new);

/// Trust network metrics
pub struct TrustMetrics {
    pub attestations_created: Counter<u64>,
    pub attestations_revoked: Counter<u64>,
    pub trust_calculations: Counter<u64>,
    pub trust_calculation_duration: Histogram<f64>,
    pub trust_cache_hits: Counter<u64>,
    pub trust_cache_misses: Counter<u64>,
    pub trust_path_length: Histogram<f64>,
    pub trust_score_distribution: Histogram<f64>,
}

impl TrustMetrics {
    fn new() -> Self {
        let meter = global::meter("mycelix.trust");

        Self {
            attestations_created: meter
                .u64_counter("trust_attestations_created_total")
                .with_description("Total trust attestations created")
                .init(),
            attestations_revoked: meter
                .u64_counter("trust_attestations_revoked_total")
                .with_description("Total trust attestations revoked")
                .init(),
            trust_calculations: meter
                .u64_counter("trust_calculations_total")
                .with_description("Total trust score calculations")
                .init(),
            trust_calculation_duration: meter
                .f64_histogram("trust_calculation_duration_seconds")
                .with_description("Trust calculation duration")
                .with_unit(Unit::new("s"))
                .init(),
            trust_cache_hits: meter
                .u64_counter("trust_cache_hits_total")
                .with_description("Trust cache hits")
                .init(),
            trust_cache_misses: meter
                .u64_counter("trust_cache_misses_total")
                .with_description("Trust cache misses")
                .init(),
            trust_path_length: meter
                .f64_histogram("trust_path_length")
                .with_description("Trust path length distribution")
                .init(),
            trust_score_distribution: meter
                .f64_histogram("trust_score_distribution")
                .with_description("Distribution of calculated trust scores")
                .init(),
        }
    }
}

pub static TRUST_METRICS: Lazy<TrustMetrics> = Lazy::new(TrustMetrics::new);

/// Database metrics
pub struct DatabaseMetrics {
    pub queries_total: Counter<u64>,
    pub query_duration: Histogram<f64>,
    pub connections_active: UpDownCounter<i64>,
    pub connections_idle: UpDownCounter<i64>,
    pub query_errors: Counter<u64>,
    pub transactions_total: Counter<u64>,
    pub transaction_duration: Histogram<f64>,
}

impl DatabaseMetrics {
    fn new() -> Self {
        let meter = global::meter("mycelix.database");

        Self {
            queries_total: meter
                .u64_counter("db_queries_total")
                .with_description("Total database queries")
                .init(),
            query_duration: meter
                .f64_histogram("db_query_duration_seconds")
                .with_description("Database query duration")
                .with_unit(Unit::new("s"))
                .init(),
            connections_active: meter
                .i64_up_down_counter("db_connections_active")
                .with_description("Active database connections")
                .init(),
            connections_idle: meter
                .i64_up_down_counter("db_connections_idle")
                .with_description("Idle database connections")
                .init(),
            query_errors: meter
                .u64_counter("db_query_errors_total")
                .with_description("Database query errors")
                .init(),
            transactions_total: meter
                .u64_counter("db_transactions_total")
                .with_description("Total database transactions")
                .init(),
            transaction_duration: meter
                .f64_histogram("db_transaction_duration_seconds")
                .with_description("Database transaction duration")
                .with_unit(Unit::new("s"))
                .init(),
        }
    }
}

pub static DB_METRICS: Lazy<DatabaseMetrics> = Lazy::new(DatabaseMetrics::new);

/// Authentication metrics
pub struct AuthMetrics {
    pub login_attempts: Counter<u64>,
    pub login_successes: Counter<u64>,
    pub login_failures: Counter<u64>,
    pub token_generations: Counter<u64>,
    pub token_validations: Counter<u64>,
    pub token_refreshes: Counter<u64>,
    pub mfa_challenges: Counter<u64>,
    pub active_sessions: UpDownCounter<i64>,
}

impl AuthMetrics {
    fn new() -> Self {
        let meter = global::meter("mycelix.auth");

        Self {
            login_attempts: meter
                .u64_counter("auth_login_attempts_total")
                .with_description("Total login attempts")
                .init(),
            login_successes: meter
                .u64_counter("auth_login_successes_total")
                .with_description("Successful logins")
                .init(),
            login_failures: meter
                .u64_counter("auth_login_failures_total")
                .with_description("Failed logins")
                .init(),
            token_generations: meter
                .u64_counter("auth_token_generations_total")
                .with_description("Token generations")
                .init(),
            token_validations: meter
                .u64_counter("auth_token_validations_total")
                .with_description("Token validations")
                .init(),
            token_refreshes: meter
                .u64_counter("auth_token_refreshes_total")
                .with_description("Token refreshes")
                .init(),
            mfa_challenges: meter
                .u64_counter("auth_mfa_challenges_total")
                .with_description("MFA challenges issued")
                .init(),
            active_sessions: meter
                .i64_up_down_counter("auth_active_sessions")
                .with_description("Currently active sessions")
                .init(),
        }
    }
}

pub static AUTH_METRICS: Lazy<AuthMetrics> = Lazy::new(AuthMetrics::new);

// ============================================================================
// Helper Functions
// ============================================================================

/// Timer for measuring durations
pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn start() -> Self {
        Self { start: Instant::now() }
    }

    pub fn elapsed_seconds(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    pub fn record_http_request(&self, method: &str, path: &str, status: u16) {
        let duration = self.elapsed_seconds();
        let attributes = [
            KeyValue::new("method", method.to_string()),
            KeyValue::new("path", path.to_string()),
            KeyValue::new("status", status.to_string()),
        ];

        HTTP_METRICS.request_duration.record(duration, &attributes);
        HTTP_METRICS.requests_total.add(1, &attributes);
    }

    pub fn record_email_processing(&self, operation: &str, success: bool) {
        let duration = self.elapsed_seconds();
        let attributes = [
            KeyValue::new("operation", operation.to_string()),
            KeyValue::new("success", success.to_string()),
        ];

        EMAIL_METRICS.email_processing_duration.record(duration, &attributes);
        EMAIL_METRICS.emails_processed.add(1, &attributes);
    }

    pub fn record_trust_calculation(&self, cache_hit: bool) {
        let duration = self.elapsed_seconds();
        let attributes = [
            KeyValue::new("cache_hit", cache_hit.to_string()),
        ];

        TRUST_METRICS.trust_calculation_duration.record(duration, &attributes);
        TRUST_METRICS.trust_calculations.add(1, &attributes);

        if cache_hit {
            TRUST_METRICS.trust_cache_hits.add(1, &[]);
        } else {
            TRUST_METRICS.trust_cache_misses.add(1, &[]);
        }
    }

    pub fn record_db_query(&self, query_type: &str, success: bool) {
        let duration = self.elapsed_seconds();
        let attributes = [
            KeyValue::new("query_type", query_type.to_string()),
            KeyValue::new("success", success.to_string()),
        ];

        DB_METRICS.query_duration.record(duration, &attributes);
        DB_METRICS.queries_total.add(1, &attributes);

        if !success {
            DB_METRICS.query_errors.add(1, &attributes);
        }
    }
}

/// Record an email event
pub fn record_email_received(folder: &str, has_attachments: bool, encrypted: bool) {
    let attributes = [
        KeyValue::new("folder", folder.to_string()),
        KeyValue::new("has_attachments", has_attachments.to_string()),
        KeyValue::new("encrypted", encrypted.to_string()),
    ];
    EMAIL_METRICS.emails_received.add(1, &attributes);
}

pub fn record_email_sent(encrypted: bool) {
    let attributes = [
        KeyValue::new("encrypted", encrypted.to_string()),
    ];
    EMAIL_METRICS.emails_sent.add(1, &attributes);
}

pub fn record_spam_detected(confidence: f64) {
    let attributes = [
        KeyValue::new("confidence_bucket", format!("{:.1}", (confidence * 10.0).floor() / 10.0)),
    ];
    EMAIL_METRICS.spam_detected.add(1, &attributes);
}

/// Record trust events
pub fn record_attestation_created(level: i32) {
    let attributes = [
        KeyValue::new("level", level.to_string()),
    ];
    TRUST_METRICS.attestations_created.add(1, &attributes);
}

pub fn record_attestation_revoked(reason: &str) {
    let attributes = [
        KeyValue::new("reason", reason.to_string()),
    ];
    TRUST_METRICS.attestations_revoked.add(1, &attributes);
}

pub fn record_trust_score(email: &str, score: f64, path_length: usize) {
    TRUST_METRICS.trust_score_distribution.record(score, &[]);
    TRUST_METRICS.trust_path_length.record(path_length as f64, &[]);
}

/// Record auth events
pub fn record_login_attempt(success: bool, method: &str) {
    let attributes = [
        KeyValue::new("method", method.to_string()),
    ];

    AUTH_METRICS.login_attempts.add(1, &attributes);
    if success {
        AUTH_METRICS.login_successes.add(1, &attributes);
    } else {
        AUTH_METRICS.login_failures.add(1, &attributes);
    }
}

pub fn record_mfa_challenge(method: &str, success: bool) {
    let attributes = [
        KeyValue::new("method", method.to_string()),
        KeyValue::new("success", success.to_string()),
    ];
    AUTH_METRICS.mfa_challenges.add(1, &attributes);
}
