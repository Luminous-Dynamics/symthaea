// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phishing Detection
//!
//! Detect and protect against phishing attacks in emails

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use url::Url;
use uuid::Uuid;

/// Phishing detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhishingAnalysis {
    pub email_id: Uuid,
    pub is_phishing: bool,
    pub confidence: f64,
    pub risk_level: RiskLevel,
    pub indicators: Vec<PhishingIndicator>,
    pub safe_links: Vec<SafeLink>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RiskLevel {
    Safe,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhishingIndicator {
    pub indicator_type: IndicatorType,
    pub severity: RiskLevel,
    pub description: String,
    pub evidence: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum IndicatorType {
    SuspiciousLink,
    DomainSpoofing,
    UrgentLanguage,
    SensitiveInfoRequest,
    MismatchedUrls,
    SuspiciousAttachment,
    UnknownSender,
    ReplyToMismatch,
    HomoglyphAttack,
    SuspiciousHeader,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeLink {
    pub original_url: String,
    pub display_text: String,
    pub safe_url: String,
    pub domain: String,
    pub is_safe: bool,
    pub warnings: Vec<String>,
}

/// Phishing detector
pub struct PhishingDetector {
    known_phishing_domains: HashSet<String>,
    trusted_domains: HashSet<String>,
    suspicious_tlds: HashSet<String>,
    urgency_patterns: Vec<Regex>,
    sensitive_info_patterns: Vec<Regex>,
}

impl PhishingDetector {
    pub fn new() -> Self {
        Self {
            known_phishing_domains: Self::load_phishing_domains(),
            trusted_domains: Self::load_trusted_domains(),
            suspicious_tlds: Self::load_suspicious_tlds(),
            urgency_patterns: Self::compile_urgency_patterns(),
            sensitive_info_patterns: Self::compile_sensitive_patterns(),
        }
    }

    fn load_phishing_domains() -> HashSet<String> {
        // In production, this would be loaded from a database or API
        HashSet::from([
            "login-secure-bank.xyz".to_string(),
            "verify-account-now.tk".to_string(),
            "paypa1.com".to_string(),
            "amaz0n-security.com".to_string(),
        ])
    }

    fn load_trusted_domains() -> HashSet<String> {
        HashSet::from([
            "google.com".to_string(),
            "microsoft.com".to_string(),
            "apple.com".to_string(),
            "amazon.com".to_string(),
            "paypal.com".to_string(),
            "github.com".to_string(),
            "linkedin.com".to_string(),
        ])
    }

    fn load_suspicious_tlds() -> HashSet<String> {
        HashSet::from([
            "tk".to_string(),
            "ml".to_string(),
            "ga".to_string(),
            "cf".to_string(),
            "gq".to_string(),
            "xyz".to_string(),
            "top".to_string(),
            "click".to_string(),
            "link".to_string(),
        ])
    }

    fn compile_urgency_patterns() -> Vec<Regex> {
        vec![
            Regex::new(r"(?i)urgent|immediately|right now|asap").unwrap(),
            Regex::new(r"(?i)your account (will be|has been) (suspended|closed|locked)").unwrap(),
            Regex::new(r"(?i)within \d+ hours|24 hours|48 hours").unwrap(),
            Regex::new(r"(?i)act now|act immediately|take action").unwrap(),
            Regex::new(r"(?i)limited time|expires soon|last chance").unwrap(),
            Regex::new(r"(?i)verify (your|the) (account|identity|information)").unwrap(),
        ]
    }

    fn compile_sensitive_patterns() -> Vec<Regex> {
        vec![
            Regex::new(r"(?i)(enter|confirm|verify|update) (your )?(password|pin|ssn|social security)").unwrap(),
            Regex::new(r"(?i)credit card (number|details|information)").unwrap(),
            Regex::new(r"(?i)bank (account|details|information)").unwrap(),
            Regex::new(r"(?i)login credentials").unwrap(),
            Regex::new(r"(?i)(mother's maiden|security question)").unwrap(),
        ]
    }

    /// Analyze email for phishing
    pub fn analyze(&self, email: &EmailForAnalysis) -> PhishingAnalysis {
        let mut indicators = Vec::new();

        // Check sender
        indicators.extend(self.check_sender(&email.from, &email.reply_to));

        // Check links
        let (link_indicators, safe_links) = self.check_links(&email.body_html.as_deref().unwrap_or(&email.body_text));
        indicators.extend(link_indicators);

        // Check content
        indicators.extend(self.check_content(&email.subject, &email.body_text));

        // Check headers
        indicators.extend(self.check_headers(&email.headers));

        // Check attachments
        indicators.extend(self.check_attachments(&email.attachments));

        // Calculate overall risk
        let (is_phishing, confidence, risk_level) = self.calculate_risk(&indicators);

        // Generate recommendations
        let recommendations = self.generate_recommendations(&indicators, is_phishing);

        PhishingAnalysis {
            email_id: email.id,
            is_phishing,
            confidence,
            risk_level,
            indicators,
            safe_links,
            recommendations,
        }
    }

    fn check_sender(&self, from: &str, reply_to: &Option<String>) -> Vec<PhishingIndicator> {
        let mut indicators = Vec::new();

        // Extract domain from sender
        let from_domain = from.split('@').nth(1).unwrap_or("");

        // Check for domain spoofing (lookalike domains)
        for trusted in &self.trusted_domains {
            if self.is_lookalike(from_domain, trusted) {
                indicators.push(PhishingIndicator {
                    indicator_type: IndicatorType::DomainSpoofing,
                    severity: RiskLevel::Critical,
                    description: format!("Domain '{}' looks similar to trusted domain '{}'", from_domain, trusted),
                    evidence: Some(from.to_string()),
                });
            }
        }

        // Check reply-to mismatch
        if let Some(reply_to_addr) = reply_to {
            let reply_domain = reply_to_addr.split('@').nth(1).unwrap_or("");
            if reply_domain != from_domain {
                indicators.push(PhishingIndicator {
                    indicator_type: IndicatorType::ReplyToMismatch,
                    severity: RiskLevel::High,
                    description: "Reply-To address domain differs from sender domain".to_string(),
                    evidence: Some(format!("From: {}, Reply-To: {}", from, reply_to_addr)),
                });
            }
        }

        // Check known phishing domains
        if self.known_phishing_domains.contains(from_domain) {
            indicators.push(PhishingIndicator {
                indicator_type: IndicatorType::SuspiciousLink,
                severity: RiskLevel::Critical,
                description: "Sender domain is on known phishing list".to_string(),
                evidence: Some(from_domain.to_string()),
            });
        }

        indicators
    }

    fn check_links(&self, html: &str) -> (Vec<PhishingIndicator>, Vec<SafeLink>) {
        let mut indicators = Vec::new();
        let mut safe_links = Vec::new();

        // Extract links from HTML
        let link_regex = Regex::new(r#"<a[^>]*href=["']([^"']+)["'][^>]*>([^<]*)</a>"#).unwrap();

        for cap in link_regex.captures_iter(html) {
            let url = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            let display_text = cap.get(2).map(|m| m.as_str()).unwrap_or("");

            if let Ok(parsed_url) = Url::parse(url) {
                let domain = parsed_url.host_str().unwrap_or("");
                let mut warnings = Vec::new();
                let mut is_safe = true;

                // Check for mismatched display text (shows different URL)
                if display_text.starts_with("http") && !display_text.contains(domain) {
                    indicators.push(PhishingIndicator {
                        indicator_type: IndicatorType::MismatchedUrls,
                        severity: RiskLevel::High,
                        description: "Link displays different URL than actual destination".to_string(),
                        evidence: Some(format!("Shows: {}, Goes to: {}", display_text, url)),
                    });
                    warnings.push("Display text doesn't match actual URL".to_string());
                    is_safe = false;
                }

                // Check for known phishing domain
                if self.known_phishing_domains.contains(domain) {
                    indicators.push(PhishingIndicator {
                        indicator_type: IndicatorType::SuspiciousLink,
                        severity: RiskLevel::Critical,
                        description: "Link goes to known phishing domain".to_string(),
                        evidence: Some(url.to_string()),
                    });
                    warnings.push("Known phishing domain".to_string());
                    is_safe = false;
                }

                // Check for suspicious TLD
                if let Some(tld) = domain.split('.').last() {
                    if self.suspicious_tlds.contains(tld) {
                        indicators.push(PhishingIndicator {
                            indicator_type: IndicatorType::SuspiciousLink,
                            severity: RiskLevel::Medium,
                            description: "Link uses suspicious top-level domain".to_string(),
                            evidence: Some(format!("TLD: .{}", tld)),
                        });
                        warnings.push(format!("Suspicious TLD: .{}", tld));
                    }
                }

                // Check for homograph attacks (IDN spoofing)
                if domain.chars().any(|c| !c.is_ascii()) {
                    indicators.push(PhishingIndicator {
                        indicator_type: IndicatorType::HomoglyphAttack,
                        severity: RiskLevel::High,
                        description: "Link contains non-ASCII characters (potential homograph attack)".to_string(),
                        evidence: Some(domain.to_string()),
                    });
                    warnings.push("Contains non-ASCII characters".to_string());
                    is_safe = false;
                }

                // Check for IP address URLs
                if parsed_url.host().map(|h| h.is_ipv4() || h.is_ipv6()).unwrap_or(false) {
                    indicators.push(PhishingIndicator {
                        indicator_type: IndicatorType::SuspiciousLink,
                        severity: RiskLevel::High,
                        description: "Link uses IP address instead of domain name".to_string(),
                        evidence: Some(url.to_string()),
                    });
                    warnings.push("Uses IP address instead of domain".to_string());
                    is_safe = false;
                }

                safe_links.push(SafeLink {
                    original_url: url.to_string(),
                    display_text: display_text.to_string(),
                    safe_url: if is_safe { url.to_string() } else { String::new() },
                    domain: domain.to_string(),
                    is_safe,
                    warnings,
                });
            }
        }

        (indicators, safe_links)
    }

    fn check_content(&self, subject: &str, body: &str) -> Vec<PhishingIndicator> {
        let mut indicators = Vec::new();
        let text = format!("{} {}", subject, body);

        // Check for urgency patterns
        for pattern in &self.urgency_patterns {
            if pattern.is_match(&text) {
                indicators.push(PhishingIndicator {
                    indicator_type: IndicatorType::UrgentLanguage,
                    severity: RiskLevel::Medium,
                    description: "Email contains urgent/threatening language".to_string(),
                    evidence: pattern.find(&text).map(|m| m.as_str().to_string()),
                });
                break; // Only add one urgency indicator
            }
        }

        // Check for sensitive information requests
        for pattern in &self.sensitive_info_patterns {
            if pattern.is_match(&text) {
                indicators.push(PhishingIndicator {
                    indicator_type: IndicatorType::SensitiveInfoRequest,
                    severity: RiskLevel::High,
                    description: "Email requests sensitive personal information".to_string(),
                    evidence: pattern.find(&text).map(|m| m.as_str().to_string()),
                });
            }
        }

        indicators
    }

    fn check_headers(&self, headers: &std::collections::HashMap<String, String>) -> Vec<PhishingIndicator> {
        let mut indicators = Vec::new();

        // Check for SPF/DKIM/DMARC failures
        if let Some(auth_results) = headers.get("authentication-results") {
            if auth_results.contains("fail") {
                indicators.push(PhishingIndicator {
                    indicator_type: IndicatorType::SuspiciousHeader,
                    severity: RiskLevel::High,
                    description: "Email failed authentication checks (SPF/DKIM/DMARC)".to_string(),
                    evidence: Some(auth_results.clone()),
                });
            }
        }

        // Check for suspicious X-headers
        if let Some(mailer) = headers.get("x-mailer") {
            if mailer.to_lowercase().contains("php") || mailer.contains("script") {
                indicators.push(PhishingIndicator {
                    indicator_type: IndicatorType::SuspiciousHeader,
                    severity: RiskLevel::Medium,
                    description: "Email appears to be sent by automated script".to_string(),
                    evidence: Some(mailer.clone()),
                });
            }
        }

        indicators
    }

    fn check_attachments(&self, attachments: &[AttachmentInfo]) -> Vec<PhishingIndicator> {
        let mut indicators = Vec::new();

        let dangerous_extensions = [
            ".exe", ".scr", ".bat", ".cmd", ".vbs", ".js", ".jar",
            ".msi", ".dll", ".hta", ".ps1", ".wsf",
        ];

        let suspicious_extensions = [
            ".zip", ".rar", ".7z", ".iso", ".img",
        ];

        for attachment in attachments {
            let filename_lower = attachment.filename.to_lowercase();

            // Check for dangerous file types
            if dangerous_extensions.iter().any(|ext| filename_lower.ends_with(ext)) {
                indicators.push(PhishingIndicator {
                    indicator_type: IndicatorType::SuspiciousAttachment,
                    severity: RiskLevel::Critical,
                    description: "Attachment has potentially dangerous file type".to_string(),
                    evidence: Some(attachment.filename.clone()),
                });
            }

            // Check for double extensions (e.g., document.pdf.exe)
            let parts: Vec<&str> = attachment.filename.split('.').collect();
            if parts.len() > 2 {
                let second_last = parts[parts.len() - 2].to_lowercase();
                if ["pdf", "doc", "docx", "xls", "xlsx", "jpg", "png"].contains(&second_last.as_str()) {
                    indicators.push(PhishingIndicator {
                        indicator_type: IndicatorType::SuspiciousAttachment,
                        severity: RiskLevel::High,
                        description: "Attachment uses double extension (common malware technique)".to_string(),
                        evidence: Some(attachment.filename.clone()),
                    });
                }
            }

            // Check for password-protected archives (sometimes used to bypass scanning)
            if suspicious_extensions.iter().any(|ext| filename_lower.ends_with(ext)) {
                indicators.push(PhishingIndicator {
                    indicator_type: IndicatorType::SuspiciousAttachment,
                    severity: RiskLevel::Low,
                    description: "Email contains compressed archive attachment".to_string(),
                    evidence: Some(attachment.filename.clone()),
                });
            }
        }

        indicators
    }

    fn is_lookalike(&self, domain: &str, trusted: &str) -> bool {
        if domain == trusted {
            return false;
        }

        // Check for common substitutions
        let substitutions = [
            ('o', '0'), ('l', '1'), ('i', '1'), ('e', '3'),
            ('a', '4'), ('s', '5'), ('g', '9'),
        ];

        let mut normalized = domain.to_lowercase();
        for (from, to) in substitutions {
            normalized = normalized.replace(to, &from.to_string());
        }

        // Check Levenshtein distance
        let distance = levenshtein(&normalized, trusted);
        distance <= 2 && distance > 0
    }

    fn calculate_risk(&self, indicators: &[PhishingIndicator]) -> (bool, f64, RiskLevel) {
        if indicators.is_empty() {
            return (false, 0.0, RiskLevel::Safe);
        }

        let mut score = 0.0;

        for indicator in indicators {
            score += match indicator.severity {
                RiskLevel::Safe => 0.0,
                RiskLevel::Low => 0.1,
                RiskLevel::Medium => 0.25,
                RiskLevel::High => 0.4,
                RiskLevel::Critical => 0.6,
            };
        }

        let confidence = (score / 2.0).min(1.0);

        let risk_level = if score >= 1.0 {
            RiskLevel::Critical
        } else if score >= 0.6 {
            RiskLevel::High
        } else if score >= 0.3 {
            RiskLevel::Medium
        } else if score > 0.0 {
            RiskLevel::Low
        } else {
            RiskLevel::Safe
        };

        let is_phishing = score >= 0.5;

        (is_phishing, confidence, risk_level)
    }

    fn generate_recommendations(&self, indicators: &[PhishingIndicator], is_phishing: bool) -> Vec<String> {
        let mut recommendations = Vec::new();

        if is_phishing {
            recommendations.push("Do not click any links in this email".to_string());
            recommendations.push("Do not download or open any attachments".to_string());
            recommendations.push("Report this email as phishing".to_string());
        }

        for indicator in indicators {
            match indicator.indicator_type {
                IndicatorType::SensitiveInfoRequest => {
                    recommendations.push("Legitimate companies never ask for sensitive information via email".to_string());
                }
                IndicatorType::UrgentLanguage => {
                    recommendations.push("Be suspicious of emails creating urgency or fear".to_string());
                }
                IndicatorType::MismatchedUrls => {
                    recommendations.push("Hover over links to see the real destination before clicking".to_string());
                }
                IndicatorType::DomainSpoofing => {
                    recommendations.push("Verify the sender by contacting them through official channels".to_string());
                }
                _ => {}
            }
        }

        recommendations.dedup();
        recommendations
    }
}

impl Default for PhishingDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct EmailForAnalysis {
    pub id: Uuid,
    pub from: String,
    pub reply_to: Option<String>,
    pub subject: String,
    pub body_text: String,
    pub body_html: Option<String>,
    pub headers: std::collections::HashMap<String, String>,
    pub attachments: Vec<AttachmentInfo>,
}

#[derive(Debug, Clone)]
pub struct AttachmentInfo {
    pub filename: String,
    pub content_type: String,
    pub size: usize,
}

/// Simple Levenshtein distance
fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 { return n; }
    if n == 0 { return m; }

    let mut dp = vec![vec![0; n + 1]; m + 1];

    for i in 0..=m { dp[i][0] = i; }
    for j in 0..=n { dp[0][j] = j; }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i-1] == b_chars[j-1] { 0 } else { 1 };
            dp[i][j] = (dp[i-1][j] + 1)
                .min(dp[i][j-1] + 1)
                .min(dp[i-1][j-1] + cost);
        }
    }

    dp[m][n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phishing_detection() {
        let detector = PhishingDetector::new();

        let email = EmailForAnalysis {
            id: Uuid::new_v4(),
            from: "security@paypa1.com".to_string(),
            reply_to: Some("hacker@evil.tk".to_string()),
            subject: "URGENT: Your account will be suspended!".to_string(),
            body_text: "Click here immediately to verify your account and password.".to_string(),
            body_html: Some(r#"<a href="http://evil.tk/steal">Click here to verify</a>"#.to_string()),
            headers: std::collections::HashMap::new(),
            attachments: Vec::new(),
        };

        let result = detector.analyze(&email);

        assert!(result.is_phishing);
        assert!(result.confidence > 0.5);
        assert!(!result.indicators.is_empty());
    }

    #[test]
    fn test_lookalike_detection() {
        let detector = PhishingDetector::new();

        assert!(detector.is_lookalike("paypa1.com", "paypal.com"));
        assert!(detector.is_lookalike("g00gle.com", "google.com"));
        assert!(!detector.is_lookalike("paypal.com", "paypal.com"));
        assert!(!detector.is_lookalike("example.com", "paypal.com"));
    }
}
