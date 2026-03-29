// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Security Anomaly Detection for Mycelix Mail
//!
//! Detects suspicious patterns in email behavior, login attempts,
//! and trust network changes to identify potential threats.

use std::collections::HashMap;
use std::net::IpAddr;
use std::time::{Duration, Instant};
use chrono::{DateTime, Utc, Timelike};
use serde::{Deserialize, Serialize};

/// Types of anomalies that can be detected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AnomalyType {
    /// Unusual login location or device
    SuspiciousLogin,
    /// Rapid succession of failed auth attempts
    BruteForceAttempt,
    /// Sudden spike in outbound emails
    EmailVolumeSpike,
    /// Email sent at unusual times
    UnusualSendTime,
    /// Sudden trust score manipulation
    TrustManipulation,
    /// Possible phishing attempt
    PhishingIndicator,
    /// Account takeover indicators
    AccountCompromise,
    /// Unusual attachment patterns
    SuspiciousAttachment,
    /// Mass unsubscribe or deletion
    BulkOperation,
    /// New device or location
    NewAccessPattern,
}

/// Severity levels for detected anomalies
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Ord, PartialOrd, Eq)]
pub enum Severity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// A detected anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    pub id: String,
    pub anomaly_type: AnomalyType,
    pub severity: Severity,
    pub description: String,
    pub detected_at: DateTime<Utc>,
    pub user_id: String,
    pub evidence: HashMap<String, String>,
    pub recommended_action: String,
    pub auto_mitigated: bool,
}

/// User behavioral profile for baseline comparison
#[derive(Debug, Clone, Default)]
pub struct UserBehaviorProfile {
    pub user_id: String,
    pub typical_send_hours: Vec<u32>,
    pub typical_send_volume_per_day: f64,
    pub known_devices: Vec<String>,
    pub known_ip_ranges: Vec<String>,
    pub known_locations: Vec<String>,
    pub typical_recipients: HashMap<String, u32>,
    pub last_updated: Option<DateTime<Utc>>,
}

impl UserBehaviorProfile {
    pub fn new(user_id: String) -> Self {
        Self {
            user_id,
            typical_send_hours: (9..18).collect(), // Default 9 AM - 6 PM
            typical_send_volume_per_day: 20.0,
            known_devices: Vec::new(),
            known_ip_ranges: Vec::new(),
            known_locations: Vec::new(),
            typical_recipients: HashMap::new(),
            last_updated: None,
        }
    }

    pub fn is_typical_hour(&self, hour: u32) -> bool {
        self.typical_send_hours.contains(&hour)
    }

    pub fn is_known_device(&self, device_id: &str) -> bool {
        self.known_devices.iter().any(|d| d == device_id)
    }
}

/// Event types for analysis
#[derive(Debug, Clone)]
pub enum SecurityEvent {
    LoginAttempt {
        user_id: String,
        success: bool,
        ip_address: IpAddr,
        device_fingerprint: String,
        location: Option<String>,
        timestamp: DateTime<Utc>,
    },
    EmailSent {
        user_id: String,
        recipient_count: usize,
        has_attachments: bool,
        attachment_types: Vec<String>,
        timestamp: DateTime<Utc>,
    },
    TrustChange {
        user_id: String,
        target_user: String,
        old_score: f64,
        new_score: f64,
        timestamp: DateTime<Utc>,
    },
    BulkOperation {
        user_id: String,
        operation_type: String,
        affected_count: usize,
        timestamp: DateTime<Utc>,
    },
}

/// Rate limiter for tracking event frequencies
struct RateLimiter {
    events: HashMap<String, Vec<Instant>>,
    window: Duration,
}

impl RateLimiter {
    fn new(window: Duration) -> Self {
        Self {
            events: HashMap::new(),
            window,
        }
    }

    fn record(&mut self, key: &str) {
        let now = Instant::now();
        let events = self.events.entry(key.to_string()).or_default();
        events.retain(|t| now.duration_since(*t) < self.window);
        events.push(now);
    }

    fn count(&mut self, key: &str) -> usize {
        let now = Instant::now();
        if let Some(events) = self.events.get_mut(key) {
            events.retain(|t| now.duration_since(*t) < self.window);
            events.len()
        } else {
            0
        }
    }
}

/// Main anomaly detection engine
pub struct AnomalyDetector {
    profiles: HashMap<String, UserBehaviorProfile>,
    failed_login_limiter: RateLimiter,
    email_send_limiter: RateLimiter,
    detection_rules: Vec<Box<dyn DetectionRule + Send + Sync>>,
}

impl AnomalyDetector {
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            failed_login_limiter: RateLimiter::new(Duration::from_secs(300)), // 5 min window
            email_send_limiter: RateLimiter::new(Duration::from_secs(3600)),  // 1 hour window
            detection_rules: Self::default_rules(),
        }
    }

    fn default_rules() -> Vec<Box<dyn DetectionRule + Send + Sync>> {
        vec![
            Box::new(BruteForceRule::new(5, Duration::from_secs(300))),
            Box::new(VolumeAnomalyRule::new(3.0)), // 3x normal volume
            Box::new(TimeAnomalyRule::new()),
            Box::new(PhishingRule::new()),
            Box::new(TrustManipulationRule::new(0.5)), // 50% change threshold
        ]
    }

    /// Analyze a security event for anomalies
    pub fn analyze(&mut self, event: &SecurityEvent) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();

        match event {
            SecurityEvent::LoginAttempt { user_id, success, ip_address, device_fingerprint, location, timestamp } => {
                // Track failed logins for brute force detection
                if !success {
                    let key = format!("login_fail:{}", user_id);
                    self.failed_login_limiter.record(&key);

                    let fail_count = self.failed_login_limiter.count(&key);
                    if fail_count >= 5 {
                        anomalies.push(Anomaly {
                            id: uuid::Uuid::new_v4().to_string(),
                            anomaly_type: AnomalyType::BruteForceAttempt,
                            severity: Severity::High,
                            description: format!(
                                "Detected {} failed login attempts in 5 minutes",
                                fail_count
                            ),
                            detected_at: *timestamp,
                            user_id: user_id.clone(),
                            evidence: HashMap::from([
                                ("ip_address".to_string(), ip_address.to_string()),
                                ("fail_count".to_string(), fail_count.to_string()),
                            ]),
                            recommended_action: "Temporarily lock account and require CAPTCHA".to_string(),
                            auto_mitigated: false,
                        });
                    }
                }

                // Check for new device/location
                let profile = self.get_or_create_profile(user_id);
                if *success && !profile.is_known_device(device_fingerprint) {
                    anomalies.push(Anomaly {
                        id: uuid::Uuid::new_v4().to_string(),
                        anomaly_type: AnomalyType::NewAccessPattern,
                        severity: Severity::Medium,
                        description: "Login from new device detected".to_string(),
                        detected_at: *timestamp,
                        user_id: user_id.clone(),
                        evidence: HashMap::from([
                            ("device".to_string(), device_fingerprint.clone()),
                            ("location".to_string(), location.clone().unwrap_or_default()),
                        ]),
                        recommended_action: "Send notification email to user".to_string(),
                        auto_mitigated: false,
                    });
                }
            }

            SecurityEvent::EmailSent { user_id, recipient_count, has_attachments, attachment_types, timestamp } => {
                let key = format!("email_send:{}", user_id);
                self.email_send_limiter.record(&key);

                let send_count = self.email_send_limiter.count(&key);
                let profile = self.get_or_create_profile(user_id);

                // Check for volume spike
                if send_count as f64 > profile.typical_send_volume_per_day * 3.0 {
                    anomalies.push(Anomaly {
                        id: uuid::Uuid::new_v4().to_string(),
                        anomaly_type: AnomalyType::EmailVolumeSpike,
                        severity: Severity::High,
                        description: format!(
                            "Email volume {} exceeds normal by 3x (typical: {:.0})",
                            send_count, profile.typical_send_volume_per_day
                        ),
                        detected_at: *timestamp,
                        user_id: user_id.clone(),
                        evidence: HashMap::from([
                            ("current_count".to_string(), send_count.to_string()),
                            ("typical_daily".to_string(), profile.typical_send_volume_per_day.to_string()),
                        ]),
                        recommended_action: "Rate limit outbound emails and verify user".to_string(),
                        auto_mitigated: false,
                    });
                }

                // Check for unusual send time
                let hour = timestamp.hour();
                if !profile.is_typical_hour(hour) {
                    anomalies.push(Anomaly {
                        id: uuid::Uuid::new_v4().to_string(),
                        anomaly_type: AnomalyType::UnusualSendTime,
                        severity: Severity::Low,
                        description: format!("Email sent at unusual hour: {}:00", hour),
                        detected_at: *timestamp,
                        user_id: user_id.clone(),
                        evidence: HashMap::from([
                            ("hour".to_string(), hour.to_string()),
                        ]),
                        recommended_action: "Log for pattern analysis".to_string(),
                        auto_mitigated: false,
                    });
                }

                // Check for suspicious attachments
                let suspicious_types = ["exe", "bat", "cmd", "scr", "js", "vbs", "ps1"];
                if *has_attachments {
                    for ext in attachment_types {
                        if suspicious_types.contains(&ext.to_lowercase().as_str()) {
                            anomalies.push(Anomaly {
                                id: uuid::Uuid::new_v4().to_string(),
                                anomaly_type: AnomalyType::SuspiciousAttachment,
                                severity: Severity::High,
                                description: format!("Suspicious attachment type: .{}", ext),
                                detected_at: *timestamp,
                                user_id: user_id.clone(),
                                evidence: HashMap::from([
                                    ("attachment_type".to_string(), ext.clone()),
                                ]),
                                recommended_action: "Quarantine email and scan attachment".to_string(),
                                auto_mitigated: false,
                            });
                        }
                    }
                }
            }

            SecurityEvent::TrustChange { user_id, target_user, old_score, new_score, timestamp } => {
                let change_magnitude = (new_score - old_score).abs();

                // Detect sudden large trust changes
                if change_magnitude > 0.5 {
                    anomalies.push(Anomaly {
                        id: uuid::Uuid::new_v4().to_string(),
                        anomaly_type: AnomalyType::TrustManipulation,
                        severity: if change_magnitude > 0.8 { Severity::High } else { Severity::Medium },
                        description: format!(
                            "Large trust score change: {:.2} -> {:.2} (delta: {:.2})",
                            old_score, new_score, change_magnitude
                        ),
                        detected_at: *timestamp,
                        user_id: user_id.clone(),
                        evidence: HashMap::from([
                            ("target_user".to_string(), target_user.clone()),
                            ("old_score".to_string(), old_score.to_string()),
                            ("new_score".to_string(), new_score.to_string()),
                        ]),
                        recommended_action: "Review trust change history and verify".to_string(),
                        auto_mitigated: false,
                    });
                }
            }

            SecurityEvent::BulkOperation { user_id, operation_type, affected_count, timestamp } => {
                if *affected_count > 100 {
                    anomalies.push(Anomaly {
                        id: uuid::Uuid::new_v4().to_string(),
                        anomaly_type: AnomalyType::BulkOperation,
                        severity: if *affected_count > 1000 { Severity::High } else { Severity::Medium },
                        description: format!(
                            "Bulk {} operation affecting {} items",
                            operation_type, affected_count
                        ),
                        detected_at: *timestamp,
                        user_id: user_id.clone(),
                        evidence: HashMap::from([
                            ("operation".to_string(), operation_type.clone()),
                            ("count".to_string(), affected_count.to_string()),
                        ]),
                        recommended_action: "Verify operation was intentional".to_string(),
                        auto_mitigated: false,
                    });
                }
            }
        }

        anomalies
    }

    /// Check email content for phishing indicators
    pub fn check_phishing(&self, email: &PhishingCheckInput) -> Option<Anomaly> {
        let mut risk_score = 0u32;
        let mut indicators = Vec::new();

        // Check for urgency language
        let urgency_patterns = [
            "urgent", "immediate action", "account suspended",
            "verify your account", "confirm your identity",
            "limited time", "act now", "expires soon",
        ];

        let content_lower = email.content.to_lowercase();
        for pattern in urgency_patterns {
            if content_lower.contains(pattern) {
                risk_score += 10;
                indicators.push(format!("Urgency pattern: '{}'", pattern));
            }
        }

        // Check for mismatched URLs
        if email.has_mismatched_urls {
            risk_score += 30;
            indicators.push("Mismatched display URL and actual URL".to_string());
        }

        // Check for suspicious sender
        if email.sender_domain_age_days < 30 {
            risk_score += 20;
            indicators.push(format!("New sender domain ({} days old)", email.sender_domain_age_days));
        }

        // Check for credential request
        let credential_patterns = ["password", "login", "credit card", "ssn", "social security"];
        for pattern in credential_patterns {
            if content_lower.contains(pattern) {
                risk_score += 15;
                indicators.push(format!("Credential-related term: '{}'", pattern));
            }
        }

        // Check sender trust score
        if email.sender_trust_score < 0.3 {
            risk_score += 25;
            indicators.push(format!("Low sender trust score: {:.2}", email.sender_trust_score));
        }

        if risk_score >= 40 {
            Some(Anomaly {
                id: uuid::Uuid::new_v4().to_string(),
                anomaly_type: AnomalyType::PhishingIndicator,
                severity: if risk_score >= 70 { Severity::Critical }
                         else if risk_score >= 50 { Severity::High }
                         else { Severity::Medium },
                description: format!("Potential phishing email (risk score: {})", risk_score),
                detected_at: Utc::now(),
                user_id: email.recipient_id.clone(),
                evidence: HashMap::from([
                    ("risk_score".to_string(), risk_score.to_string()),
                    ("indicators".to_string(), indicators.join("; ")),
                    ("sender".to_string(), email.sender_address.clone()),
                ]),
                recommended_action: if risk_score >= 70 {
                    "Block email and notify user".to_string()
                } else {
                    "Add phishing warning banner".to_string()
                },
                auto_mitigated: false,
            })
        } else {
            None
        }
    }

    fn get_or_create_profile(&mut self, user_id: &str) -> &UserBehaviorProfile {
        if !self.profiles.contains_key(user_id) {
            self.profiles.insert(
                user_id.to_string(),
                UserBehaviorProfile::new(user_id.to_string()),
            );
        }
        self.profiles.get(user_id).unwrap()
    }

    /// Update user profile based on observed behavior
    pub fn update_profile(&mut self, user_id: &str, event: &SecurityEvent) {
        let profile = self.profiles.entry(user_id.to_string())
            .or_insert_with(|| UserBehaviorProfile::new(user_id.to_string()));

        match event {
            SecurityEvent::LoginAttempt { device_fingerprint, success, .. } => {
                if *success && !profile.known_devices.contains(device_fingerprint) {
                    profile.known_devices.push(device_fingerprint.clone());
                }
            }
            SecurityEvent::EmailSent { timestamp, .. } => {
                let hour = timestamp.hour();
                if !profile.typical_send_hours.contains(&hour) {
                    profile.typical_send_hours.push(hour);
                }
            }
            _ => {}
        }

        profile.last_updated = Some(Utc::now());
    }
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Input for phishing check
pub struct PhishingCheckInput {
    pub sender_address: String,
    pub recipient_id: String,
    pub subject: String,
    pub content: String,
    pub has_mismatched_urls: bool,
    pub sender_domain_age_days: u32,
    pub sender_trust_score: f64,
}

/// Trait for custom detection rules
trait DetectionRule {
    fn check(&self, event: &SecurityEvent, profile: &UserBehaviorProfile) -> Option<Anomaly>;
    fn name(&self) -> &str;
}

/// Brute force detection rule
struct BruteForceRule {
    threshold: usize,
    window: Duration,
}

impl BruteForceRule {
    fn new(threshold: usize, window: Duration) -> Self {
        Self { threshold, window }
    }
}

impl DetectionRule for BruteForceRule {
    fn check(&self, _event: &SecurityEvent, _profile: &UserBehaviorProfile) -> Option<Anomaly> {
        // Implemented directly in AnomalyDetector::analyze for efficiency
        None
    }

    fn name(&self) -> &str {
        "brute_force"
    }
}

/// Volume anomaly detection
struct VolumeAnomalyRule {
    multiplier_threshold: f64,
}

impl VolumeAnomalyRule {
    fn new(multiplier_threshold: f64) -> Self {
        Self { multiplier_threshold }
    }
}

impl DetectionRule for VolumeAnomalyRule {
    fn check(&self, _event: &SecurityEvent, _profile: &UserBehaviorProfile) -> Option<Anomaly> {
        None // Implemented in main analyze method
    }

    fn name(&self) -> &str {
        "volume_anomaly"
    }
}

/// Time-based anomaly detection
struct TimeAnomalyRule;

impl TimeAnomalyRule {
    fn new() -> Self {
        Self
    }
}

impl DetectionRule for TimeAnomalyRule {
    fn check(&self, _event: &SecurityEvent, _profile: &UserBehaviorProfile) -> Option<Anomaly> {
        None
    }

    fn name(&self) -> &str {
        "time_anomaly"
    }
}

/// Phishing detection rule
struct PhishingRule;

impl PhishingRule {
    fn new() -> Self {
        Self
    }
}

impl DetectionRule for PhishingRule {
    fn check(&self, _event: &SecurityEvent, _profile: &UserBehaviorProfile) -> Option<Anomaly> {
        None
    }

    fn name(&self) -> &str {
        "phishing"
    }
}

/// Trust manipulation detection
struct TrustManipulationRule {
    threshold: f64,
}

impl TrustManipulationRule {
    fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl DetectionRule for TrustManipulationRule {
    fn check(&self, _event: &SecurityEvent, _profile: &UserBehaviorProfile) -> Option<Anomaly> {
        None
    }

    fn name(&self) -> &str {
        "trust_manipulation"
    }
}

/// Service for coordinating anomaly detection across the system
pub struct SecurityMonitorService {
    detector: AnomalyDetector,
    anomaly_history: Vec<Anomaly>,
    max_history_size: usize,
}

impl SecurityMonitorService {
    pub fn new() -> Self {
        Self {
            detector: AnomalyDetector::new(),
            anomaly_history: Vec::new(),
            max_history_size: 10000,
        }
    }

    /// Process a security event and return any detected anomalies
    pub fn process_event(&mut self, event: SecurityEvent) -> Vec<Anomaly> {
        // Update user profile
        if let Some(user_id) = Self::get_user_id(&event) {
            self.detector.update_profile(&user_id, &event);
        }

        // Detect anomalies
        let anomalies = self.detector.analyze(&event);

        // Store in history
        for anomaly in &anomalies {
            self.anomaly_history.push(anomaly.clone());
            if self.anomaly_history.len() > self.max_history_size {
                self.anomaly_history.remove(0);
            }
        }

        anomalies
    }

    /// Check incoming email for phishing
    pub fn check_inbound_email(&self, input: PhishingCheckInput) -> Option<Anomaly> {
        self.detector.check_phishing(&input)
    }

    /// Get recent anomalies for a user
    pub fn get_user_anomalies(&self, user_id: &str, limit: usize) -> Vec<&Anomaly> {
        self.anomaly_history
            .iter()
            .filter(|a| a.user_id == user_id)
            .rev()
            .take(limit)
            .collect()
    }

    /// Get all critical anomalies
    pub fn get_critical_anomalies(&self) -> Vec<&Anomaly> {
        self.anomaly_history
            .iter()
            .filter(|a| a.severity == Severity::Critical)
            .collect()
    }

    fn get_user_id(event: &SecurityEvent) -> Option<String> {
        match event {
            SecurityEvent::LoginAttempt { user_id, .. } => Some(user_id.clone()),
            SecurityEvent::EmailSent { user_id, .. } => Some(user_id.clone()),
            SecurityEvent::TrustChange { user_id, .. } => Some(user_id.clone()),
            SecurityEvent::BulkOperation { user_id, .. } => Some(user_id.clone()),
        }
    }
}

impl Default for SecurityMonitorService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn test_brute_force_detection() {
        let mut detector = AnomalyDetector::new();
        let timestamp = Utc::now();

        // Simulate 5 failed logins
        for i in 0..5 {
            let event = SecurityEvent::LoginAttempt {
                user_id: "user123".to_string(),
                success: false,
                ip_address: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
                device_fingerprint: format!("device_{}", i),
                location: Some("Unknown".to_string()),
                timestamp,
            };
            let anomalies = detector.analyze(&event);

            if i == 4 {
                assert!(!anomalies.is_empty());
                assert_eq!(anomalies[0].anomaly_type, AnomalyType::BruteForceAttempt);
            }
        }
    }

    #[test]
    fn test_phishing_detection() {
        let detector = AnomalyDetector::new();

        let input = PhishingCheckInput {
            sender_address: "support@suspicious-bank.com".to_string(),
            recipient_id: "user123".to_string(),
            subject: "URGENT: Your account has been suspended!".to_string(),
            content: "Click here to verify your account immediately. Your password needs to be confirmed.".to_string(),
            has_mismatched_urls: true,
            sender_domain_age_days: 5,
            sender_trust_score: 0.1,
        };

        let result = detector.check_phishing(&input);
        assert!(result.is_some());

        let anomaly = result.unwrap();
        assert_eq!(anomaly.anomaly_type, AnomalyType::PhishingIndicator);
        assert!(anomaly.severity >= Severity::High);
    }

    #[test]
    fn test_user_profile() {
        let mut profile = UserBehaviorProfile::new("user123".to_string());

        assert!(profile.is_typical_hour(10)); // 10 AM
        assert!(!profile.is_typical_hour(3)); // 3 AM

        profile.known_devices.push("device123".to_string());
        assert!(profile.is_known_device("device123"));
        assert!(!profile.is_known_device("unknown"));
    }
}
