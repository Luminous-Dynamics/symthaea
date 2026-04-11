// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Spam & Phishing Protection
//!
//! ML-based spam classification, phishing detection, and sender reputation

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;
use regex::Regex;

// ============================================================================
// Spam Classification Service
// ============================================================================

pub struct SpamClassifier {
    pool: PgPool,
    phishing_detector: PhishingDetector,
    reputation_service: ReputationService,
}

impl SpamClassifier {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool: pool.clone(),
            phishing_detector: PhishingDetector::new(),
            reputation_service: ReputationService::new(pool),
        }
    }

    /// Classify an incoming email
    pub async fn classify(
        &self,
        user_id: Uuid,
        email: &EmailForClassification,
    ) -> Result<ClassificationResult, SpamError> {
        let mut score = 0.0;
        let mut flags = Vec::new();

        // Check sender reputation
        let reputation = self.reputation_service
            .get_sender_reputation(&email.from_address)
            .await?;

        if reputation.spam_rate > 0.5 {
            score += 30.0;
            flags.push(SpamFlag::LowReputation);
        } else if reputation.spam_rate > 0.2 {
            score += 15.0;
        }

        // Check for phishing
        let phishing_result = self.phishing_detector.analyze(email);
        score += phishing_result.risk_score;
        flags.extend(phishing_result.flags);

        // Content analysis
        let content_score = self.analyze_content(email);
        score += content_score.score;
        flags.extend(content_score.flags);

        // Header analysis
        let header_score = self.analyze_headers(email);
        score += header_score.score;
        flags.extend(header_score.flags);

        // Check user's personal whitelist/blacklist
        if self.is_whitelisted(user_id, &email.from_address).await? {
            score = 0.0;
            flags.clear();
        } else if self.is_blacklisted(user_id, &email.from_address).await? {
            score = 100.0;
            flags.push(SpamFlag::UserBlacklisted);
        }

        // Determine classification
        let classification = if score >= 80.0 {
            SpamClassification::Spam
        } else if score >= 50.0 {
            SpamClassification::Suspicious
        } else if phishing_result.is_phishing {
            SpamClassification::Phishing
        } else {
            SpamClassification::Ham
        };

        // Store classification for learning
        self.store_classification(email, &classification, score).await?;

        Ok(ClassificationResult {
            classification,
            spam_score: score,
            flags,
            phishing_urls: phishing_result.suspicious_urls,
        })
    }

    fn analyze_content(&self, email: &EmailForClassification) -> ContentAnalysis {
        let mut score = 0.0;
        let mut flags = Vec::new();

        let body = email.body_text.to_lowercase();
        let subject = email.subject.to_lowercase();

        // Spam keywords
        let spam_keywords = [
            ("viagra", 20.0),
            ("casino", 15.0),
            ("lottery", 20.0),
            ("winner", 10.0),
            ("congratulations", 5.0),
            ("click here", 10.0),
            ("act now", 10.0),
            ("limited time", 8.0),
            ("free money", 25.0),
            ("nigerian prince", 30.0),
            ("inheritance", 15.0),
            ("wire transfer", 20.0),
            ("cryptocurrency", 10.0),
            ("bitcoin", 5.0),
            ("investment opportunity", 15.0),
        ];

        for (keyword, weight) in &spam_keywords {
            if body.contains(keyword) || subject.contains(keyword) {
                score += weight;
                flags.push(SpamFlag::SpamKeyword(keyword.to_string()));
            }
        }

        // Check for excessive caps
        let caps_ratio = email.subject.chars().filter(|c| c.is_uppercase()).count() as f64
            / email.subject.len().max(1) as f64;
        if caps_ratio > 0.5 && email.subject.len() > 10 {
            score += 10.0;
            flags.push(SpamFlag::ExcessiveCaps);
        }

        // Check for excessive punctuation
        let punct_count = email.subject.chars().filter(|c| *c == '!' || *c == '?').count();
        if punct_count > 3 {
            score += 5.0;
            flags.push(SpamFlag::ExcessivePunctuation);
        }

        // Check for suspicious patterns
        if body.contains("click") && body.contains("verify") {
            score += 15.0;
            flags.push(SpamFlag::VerificationRequest);
        }

        ContentAnalysis { score, flags }
    }

    fn analyze_headers(&self, email: &EmailForClassification) -> HeaderAnalysis {
        let mut score = 0.0;
        let mut flags = Vec::new();

        // Check for missing headers
        if email.message_id.is_empty() {
            score += 10.0;
            flags.push(SpamFlag::MissingMessageId);
        }

        // Check for spoofed from address
        if let Some(ref reply_to) = email.reply_to {
            if reply_to != &email.from_address {
                let from_domain = email.from_address.split('@').last().unwrap_or("");
                let reply_domain = reply_to.split('@').last().unwrap_or("");
                if from_domain != reply_domain {
                    score += 15.0;
                    flags.push(SpamFlag::MismatchedReplyTo);
                }
            }
        }

        // Check for suspicious sending patterns
        if email.received_headers.len() > 10 {
            score += 5.0;
            flags.push(SpamFlag::ExcessiveHops);
        }

        HeaderAnalysis { score, flags }
    }

    async fn is_whitelisted(&self, user_id: Uuid, email: &str) -> Result<bool, SpamError> {
        let result: Option<(bool,)> = sqlx::query_as(
            "SELECT true FROM sender_whitelist WHERE user_id = $1 AND email = $2",
        )
        .bind(user_id)
        .bind(email)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| SpamError::Database(e.to_string()))?;

        Ok(result.is_some())
    }

    async fn is_blacklisted(&self, user_id: Uuid, email: &str) -> Result<bool, SpamError> {
        let result: Option<(bool,)> = sqlx::query_as(
            "SELECT true FROM sender_blacklist WHERE user_id = $1 AND (email = $2 OR $2 LIKE '%@' || domain)",
        )
        .bind(user_id)
        .bind(email)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| SpamError::Database(e.to_string()))?;

        Ok(result.is_some())
    }

    async fn store_classification(
        &self,
        email: &EmailForClassification,
        classification: &SpamClassification,
        score: f64,
    ) -> Result<(), SpamError> {
        sqlx::query(
            r#"
            INSERT INTO spam_classifications (id, email_id, classification, score, created_at)
            VALUES ($1, $2, $3, $4, NOW())
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(email.id)
        .bind(classification.to_string())
        .bind(score)
        .execute(&self.pool)
        .await
        .map_err(|e| SpamError::Database(e.to_string()))?;

        Ok(())
    }

    /// User reports email as spam - used for learning
    pub async fn report_spam(
        &self,
        user_id: Uuid,
        email_id: Uuid,
    ) -> Result<(), SpamError> {
        // Get email details
        let email: (String, String) = sqlx::query_as(
            "SELECT from_address, body_text FROM emails WHERE id = $1 AND user_id = $2",
        )
        .bind(email_id)
        .bind(user_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| SpamError::Database(e.to_string()))?;

        // Update sender reputation
        self.reputation_service.report_spam(&email.0).await?;

        // Store user feedback for model training
        sqlx::query(
            r#"
            INSERT INTO spam_feedback (id, user_id, email_id, is_spam, created_at)
            VALUES ($1, $2, $3, true, NOW())
            ON CONFLICT (user_id, email_id) DO UPDATE SET is_spam = true
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(user_id)
        .bind(email_id)
        .execute(&self.pool)
        .await
        .map_err(|e| SpamError::Database(e.to_string()))?;

        Ok(())
    }

    /// User reports email as not spam - used for learning
    pub async fn report_not_spam(
        &self,
        user_id: Uuid,
        email_id: Uuid,
    ) -> Result<(), SpamError> {
        let email: (String,) = sqlx::query_as(
            "SELECT from_address FROM emails WHERE id = $1 AND user_id = $2",
        )
        .bind(email_id)
        .bind(user_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| SpamError::Database(e.to_string()))?;

        self.reputation_service.report_not_spam(&email.0).await?;

        sqlx::query(
            r#"
            INSERT INTO spam_feedback (id, user_id, email_id, is_spam, created_at)
            VALUES ($1, $2, $3, false, NOW())
            ON CONFLICT (user_id, email_id) DO UPDATE SET is_spam = false
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(user_id)
        .bind(email_id)
        .execute(&self.pool)
        .await
        .map_err(|e| SpamError::Database(e.to_string()))?;

        Ok(())
    }
}

// ============================================================================
// Phishing Detection
// ============================================================================

pub struct PhishingDetector {
    suspicious_tlds: Vec<&'static str>,
    known_brands: Vec<&'static str>,
}

impl PhishingDetector {
    fn new() -> Self {
        Self {
            suspicious_tlds: vec![".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".work"],
            known_brands: vec![
                "paypal", "amazon", "apple", "microsoft", "google", "facebook",
                "netflix", "bank", "chase", "wellsfargo", "citibank", "usps",
                "fedex", "ups", "dhl", "irs", "social security",
            ],
        }
    }

    fn analyze(&self, email: &EmailForClassification) -> PhishingAnalysis {
        let mut risk_score = 0.0;
        let mut flags = Vec::new();
        let mut suspicious_urls = Vec::new();

        let body = &email.body_text;
        let body_lower = body.to_lowercase();

        // Extract URLs
        let url_regex = Regex::new(r"https?://[^\s<>\"]+").unwrap();
        let urls: Vec<&str> = url_regex.find_iter(body).map(|m| m.as_str()).collect();

        for url in &urls {
            let url_lower = url.to_lowercase();

            // Check for suspicious TLDs
            for tld in &self.suspicious_tlds {
                if url_lower.contains(tld) {
                    risk_score += 15.0;
                    flags.push(SpamFlag::SuspiciousTLD);
                    suspicious_urls.push(url.to_string());
                }
            }

            // Check for IP addresses in URLs
            if Regex::new(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}").unwrap().is_match(url) {
                risk_score += 20.0;
                flags.push(SpamFlag::IpAddressUrl);
                suspicious_urls.push(url.to_string());
            }

            // Check for URL obfuscation
            if url.contains("%") || url.contains("@") {
                risk_score += 10.0;
                flags.push(SpamFlag::ObfuscatedUrl);
                suspicious_urls.push(url.to_string());
            }

            // Check for lookalike domains
            for brand in &self.known_brands {
                if url_lower.contains(brand) && !self.is_legitimate_domain(url, brand) {
                    risk_score += 25.0;
                    flags.push(SpamFlag::LookalikeDomain(brand.to_string()));
                    suspicious_urls.push(url.to_string());
                }
            }
        }

        // Check for urgency language combined with links
        let urgency_phrases = [
            "account suspended",
            "verify your account",
            "confirm your identity",
            "unusual activity",
            "security alert",
            "update your information",
            "expires today",
            "action required",
            "immediate action",
        ];

        for phrase in &urgency_phrases {
            if body_lower.contains(phrase) && !urls.is_empty() {
                risk_score += 15.0;
                flags.push(SpamFlag::UrgencyWithLinks);
                break;
            }
        }

        // Check for credential requests
        if (body_lower.contains("password") || body_lower.contains("login") || body_lower.contains("credential"))
            && (body_lower.contains("click") || body_lower.contains("verify"))
        {
            risk_score += 20.0;
            flags.push(SpamFlag::CredentialRequest);
        }

        let is_phishing = risk_score >= 40.0;

        PhishingAnalysis {
            is_phishing,
            risk_score,
            flags,
            suspicious_urls,
        }
    }

    fn is_legitimate_domain(&self, url: &str, brand: &str) -> bool {
        // Map of brands to their legitimate domains
        let legitimate_domains: HashMap<&str, Vec<&str>> = [
            ("paypal", vec!["paypal.com"]),
            ("amazon", vec!["amazon.com", "amazon.co.uk", "amazon.de", "aws.amazon.com"]),
            ("apple", vec!["apple.com", "icloud.com"]),
            ("microsoft", vec!["microsoft.com", "live.com", "outlook.com", "office.com"]),
            ("google", vec!["google.com", "gmail.com", "youtube.com"]),
            ("facebook", vec!["facebook.com", "fb.com", "meta.com"]),
            ("netflix", vec!["netflix.com"]),
        ]
        .into_iter()
        .collect();

        if let Some(domains) = legitimate_domains.get(brand) {
            return domains.iter().any(|d| url.contains(d));
        }

        false
    }
}

// ============================================================================
// Reputation Service
// ============================================================================

pub struct ReputationService {
    pool: PgPool,
}

impl ReputationService {
    fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    async fn get_sender_reputation(&self, email: &str) -> Result<SenderReputation, SpamError> {
        let domain = email.split('@').last().unwrap_or(email);

        // Check database for existing reputation
        let existing: Option<SenderReputation> = sqlx::query_as(
            r#"
            SELECT email, domain, spam_reports, total_emails, spam_rate,
                   last_seen, first_seen, trust_score
            FROM sender_reputation
            WHERE email = $1 OR domain = $2
            ORDER BY CASE WHEN email = $1 THEN 0 ELSE 1 END
            LIMIT 1
            "#,
        )
        .bind(email)
        .bind(domain)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| SpamError::Database(e.to_string()))?;

        Ok(existing.unwrap_or(SenderReputation {
            email: email.to_string(),
            domain: domain.to_string(),
            spam_reports: 0,
            total_emails: 0,
            spam_rate: 0.0,
            last_seen: Utc::now(),
            first_seen: Utc::now(),
            trust_score: 50.0,
        }))
    }

    async fn report_spam(&self, email: &str) -> Result<(), SpamError> {
        let domain = email.split('@').last().unwrap_or(email);

        sqlx::query(
            r#"
            INSERT INTO sender_reputation (email, domain, spam_reports, total_emails, last_seen, first_seen)
            VALUES ($1, $2, 1, 1, NOW(), NOW())
            ON CONFLICT (email) DO UPDATE SET
                spam_reports = sender_reputation.spam_reports + 1,
                total_emails = sender_reputation.total_emails + 1,
                spam_rate = (sender_reputation.spam_reports + 1)::float / (sender_reputation.total_emails + 1)::float,
                last_seen = NOW()
            "#,
        )
        .bind(email)
        .bind(domain)
        .execute(&self.pool)
        .await
        .map_err(|e| SpamError::Database(e.to_string()))?;

        Ok(())
    }

    async fn report_not_spam(&self, email: &str) -> Result<(), SpamError> {
        let domain = email.split('@').last().unwrap_or(email);

        sqlx::query(
            r#"
            INSERT INTO sender_reputation (email, domain, spam_reports, total_emails, last_seen, first_seen)
            VALUES ($1, $2, 0, 1, NOW(), NOW())
            ON CONFLICT (email) DO UPDATE SET
                total_emails = sender_reputation.total_emails + 1,
                spam_rate = sender_reputation.spam_reports::float / (sender_reputation.total_emails + 1)::float,
                last_seen = NOW()
            "#,
        )
        .bind(email)
        .bind(domain)
        .execute(&self.pool)
        .await
        .map_err(|e| SpamError::Database(e.to_string()))?;

        Ok(())
    }
}

// ============================================================================
// Quarantine Manager
// ============================================================================

pub struct QuarantineManager {
    pool: PgPool,
}

impl QuarantineManager {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get quarantined emails for a user
    pub async fn get_quarantined(
        &self,
        user_id: Uuid,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<QuarantinedEmail>, SpamError> {
        let emails: Vec<QuarantinedEmail> = sqlx::query_as(
            r#"
            SELECT e.id, e.from_address, e.subject, e.received_at,
                   sc.classification, sc.score
            FROM emails e
            JOIN spam_classifications sc ON sc.email_id = e.id
            WHERE e.user_id = $1 AND e.folder = 'Spam'
            ORDER BY e.received_at DESC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(user_id)
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| SpamError::Database(e.to_string()))?;

        Ok(emails)
    }

    /// Release email from quarantine
    pub async fn release(&self, user_id: Uuid, email_id: Uuid) -> Result<(), SpamError> {
        sqlx::query(
            "UPDATE emails SET folder = 'Inbox' WHERE id = $1 AND user_id = $2",
        )
        .bind(email_id)
        .bind(user_id)
        .execute(&self.pool)
        .await
        .map_err(|e| SpamError::Database(e.to_string()))?;

        Ok(())
    }

    /// Permanently delete quarantined email
    pub async fn delete(&self, user_id: Uuid, email_id: Uuid) -> Result<(), SpamError> {
        sqlx::query(
            "DELETE FROM emails WHERE id = $1 AND user_id = $2 AND folder = 'Spam'",
        )
        .bind(email_id)
        .bind(user_id)
        .execute(&self.pool)
        .await
        .map_err(|e| SpamError::Database(e.to_string()))?;

        Ok(())
    }

    /// Auto-delete old quarantined emails
    pub async fn cleanup_old(&self, days: i32) -> Result<u64, SpamError> {
        let result = sqlx::query(
            r#"
            DELETE FROM emails
            WHERE folder = 'Spam'
            AND received_at < NOW() - INTERVAL '1 day' * $1
            "#,
        )
        .bind(days)
        .execute(&self.pool)
        .await
        .map_err(|e| SpamError::Database(e.to_string()))?;

        Ok(result.rows_affected())
    }
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailForClassification {
    pub id: Uuid,
    pub from_address: String,
    pub reply_to: Option<String>,
    pub to_addresses: Vec<String>,
    pub subject: String,
    pub body_text: String,
    pub body_html: Option<String>,
    pub message_id: String,
    pub received_headers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub classification: SpamClassification,
    pub spam_score: f64,
    pub flags: Vec<SpamFlag>,
    pub phishing_urls: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SpamClassification {
    Ham,
    Spam,
    Suspicious,
    Phishing,
}

impl ToString for SpamClassification {
    fn to_string(&self) -> String {
        match self {
            SpamClassification::Ham => "ham".to_string(),
            SpamClassification::Spam => "spam".to_string(),
            SpamClassification::Suspicious => "suspicious".to_string(),
            SpamClassification::Phishing => "phishing".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpamFlag {
    LowReputation,
    SpamKeyword(String),
    ExcessiveCaps,
    ExcessivePunctuation,
    VerificationRequest,
    MissingMessageId,
    MismatchedReplyTo,
    ExcessiveHops,
    UserBlacklisted,
    SuspiciousTLD,
    IpAddressUrl,
    ObfuscatedUrl,
    LookalikeDomain(String),
    UrgencyWithLinks,
    CredentialRequest,
}

struct ContentAnalysis {
    score: f64,
    flags: Vec<SpamFlag>,
}

struct HeaderAnalysis {
    score: f64,
    flags: Vec<SpamFlag>,
}

struct PhishingAnalysis {
    is_phishing: bool,
    risk_score: f64,
    flags: Vec<SpamFlag>,
    suspicious_urls: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct SenderReputation {
    pub email: String,
    pub domain: String,
    pub spam_reports: i32,
    pub total_emails: i32,
    pub spam_rate: f64,
    pub last_seen: DateTime<Utc>,
    pub first_seen: DateTime<Utc>,
    pub trust_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct QuarantinedEmail {
    pub id: Uuid,
    pub from_address: String,
    pub subject: String,
    pub received_at: DateTime<Utc>,
    pub classification: String,
    pub score: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum SpamError {
    #[error("Database error: {0}")]
    Database(String),
}
