// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Local Spam Detection
//!
//! ML-based spam detection using local inference (no cloud dependency)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;
use tracing::{debug, info};

/// Spam detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpamResult {
    /// Probability of being spam (0.0 - 1.0)
    pub spam_probability: f32,
    /// Classification
    pub classification: SpamClassification,
    /// Confidence score
    pub confidence: f32,
    /// Features that contributed to the classification
    pub features: Vec<SpamFeature>,
    /// Suggested action
    pub action: SpamAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SpamClassification {
    Ham,       // Not spam
    Spam,      // Definite spam
    Suspicious, // Possibly spam
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpamFeature {
    pub name: String,
    pub weight: f32,
    pub description: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SpamAction {
    Deliver,      // Normal delivery
    Quarantine,   // Hold for review
    Reject,       // Block/delete
    Flag,         // Deliver but mark
}

/// Spam detector using Naive Bayes with additional heuristics
pub struct SpamDetector {
    /// Word probabilities for ham
    ham_word_probs: RwLock<HashMap<String, f32>>,
    /// Word probabilities for spam
    spam_word_probs: RwLock<HashMap<String, f32>>,
    /// Prior probability of spam
    spam_prior: f32,
    /// Total ham messages trained on
    ham_count: RwLock<u64>,
    /// Total spam messages trained on
    spam_count: RwLock<u64>,
    /// Spam threshold
    spam_threshold: f32,
    /// Suspicious threshold
    suspicious_threshold: f32,
}

impl SpamDetector {
    pub fn new() -> Self {
        let mut detector = Self {
            ham_word_probs: RwLock::new(HashMap::new()),
            spam_word_probs: RwLock::new(HashMap::new()),
            spam_prior: 0.3, // Assume 30% of email is spam
            ham_count: RwLock::new(0),
            spam_count: RwLock::new(0),
            spam_threshold: 0.8,
            suspicious_threshold: 0.5,
        };

        // Load pre-trained weights
        detector.load_pretrained_weights();

        detector
    }

    /// Load pre-trained spam/ham word weights
    fn load_pretrained_weights(&mut self) {
        // Common spam indicators
        let spam_words: HashMap<&str, f32> = [
            ("viagra", 0.99),
            ("lottery", 0.95),
            ("winner", 0.85),
            ("congratulations", 0.7),
            ("urgent", 0.6),
            ("click", 0.5),
            ("free", 0.55),
            ("limited", 0.5),
            ("offer", 0.5),
            ("unsubscribe", 0.3),
            ("act now", 0.9),
            ("million", 0.7),
            ("prince", 0.95),
            ("inheritance", 0.9),
            ("bitcoin", 0.6),
            ("crypto", 0.5),
            ("guaranteed", 0.75),
            ("risk free", 0.85),
            ("no obligation", 0.8),
            ("cash", 0.55),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();

        // Common ham indicators
        let ham_words: HashMap<&str, f32> = [
            ("meeting", 0.8),
            ("schedule", 0.85),
            ("project", 0.9),
            ("update", 0.75),
            ("review", 0.8),
            ("document", 0.85),
            ("attached", 0.7),
            ("regarding", 0.8),
            ("follow up", 0.85),
            ("thank you", 0.75),
            ("best regards", 0.9),
            ("sincerely", 0.85),
            ("team", 0.8),
            ("deadline", 0.85),
            ("invoice", 0.7),
            ("report", 0.85),
            ("discuss", 0.8),
            ("feedback", 0.85),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();

        *self.spam_word_probs.write().unwrap() = spam_words;
        *self.ham_word_probs.write().unwrap() = ham_words;
        *self.spam_count.write().unwrap() = 10000;
        *self.ham_count.write().unwrap() = 30000;
    }

    /// Analyze email for spam
    pub fn analyze(&self, email: &EmailContent) -> SpamResult {
        let mut features = Vec::new();
        let mut spam_score = 0.0f32;
        let mut ham_score = 0.0f32;

        // Tokenize content
        let tokens = self.tokenize(&email.subject, &email.body);

        // Calculate Naive Bayes scores
        let spam_probs = self.spam_word_probs.read().unwrap();
        let ham_probs = self.ham_word_probs.read().unwrap();

        for token in &tokens {
            let spam_prob = spam_probs.get(token).copied().unwrap_or(0.01);
            let ham_prob = ham_probs.get(token).copied().unwrap_or(0.01);

            spam_score += spam_prob.ln();
            ham_score += ham_prob.ln();

            if spam_prob > 0.6 {
                features.push(SpamFeature {
                    name: token.clone(),
                    weight: spam_prob,
                    description: format!("Spam indicator word: '{}'", token),
                });
            }
        }

        // Add prior
        spam_score += self.spam_prior.ln();
        ham_score += (1.0 - self.spam_prior).ln();

        // Apply heuristic rules
        let heuristic_features = self.apply_heuristics(email);
        for feature in &heuristic_features {
            spam_score += feature.weight;
            features.push(feature.clone());
        }

        // Convert log probabilities to probability
        let max_score = spam_score.max(ham_score);
        let spam_exp = (spam_score - max_score).exp();
        let ham_exp = (ham_score - max_score).exp();
        let spam_probability = spam_exp / (spam_exp + ham_exp);

        // Determine classification
        let (classification, action) = if spam_probability >= self.spam_threshold {
            (SpamClassification::Spam, SpamAction::Quarantine)
        } else if spam_probability >= self.suspicious_threshold {
            (SpamClassification::Suspicious, SpamAction::Flag)
        } else {
            (SpamClassification::Ham, SpamAction::Deliver)
        };

        let confidence = if spam_probability >= 0.5 {
            spam_probability
        } else {
            1.0 - spam_probability
        };

        SpamResult {
            spam_probability,
            classification,
            confidence,
            features,
            action,
        }
    }

    /// Apply heuristic rules
    fn apply_heuristics(&self, email: &EmailContent) -> Vec<SpamFeature> {
        let mut features = Vec::new();

        // Check for suspicious sender patterns
        if email.from.contains("noreply") && email.from.contains("@") {
            let domain = email.from.split('@').nth(1).unwrap_or("");
            if !is_known_legitimate_domain(domain) {
                features.push(SpamFeature {
                    name: "suspicious_noreply".to_string(),
                    weight: 0.3,
                    description: "No-reply address from unknown domain".to_string(),
                });
            }
        }

        // Check for excessive capitalization
        let caps_ratio = email.subject.chars().filter(|c| c.is_uppercase()).count() as f32
            / email.subject.len().max(1) as f32;
        if caps_ratio > 0.5 && email.subject.len() > 5 {
            features.push(SpamFeature {
                name: "excessive_caps".to_string(),
                weight: 0.4,
                description: "Subject has excessive capitalization".to_string(),
            });
        }

        // Check for suspicious URLs
        let url_count = email.body.matches("http").count();
        if url_count > 5 {
            features.push(SpamFeature {
                name: "many_urls".to_string(),
                weight: 0.5,
                description: format!("Body contains {} URLs", url_count),
            });
        }

        // Check for urgency patterns
        let urgency_patterns = ["act now", "limited time", "expires", "immediate", "asap"];
        for pattern in urgency_patterns {
            if email.body.to_lowercase().contains(pattern)
                || email.subject.to_lowercase().contains(pattern)
            {
                features.push(SpamFeature {
                    name: "urgency_language".to_string(),
                    weight: 0.35,
                    description: format!("Contains urgency language: '{}'", pattern),
                });
                break;
            }
        }

        // Check for suspicious attachments
        for attachment in &email.attachment_names {
            if attachment.ends_with(".exe")
                || attachment.ends_with(".scr")
                || attachment.ends_with(".bat")
            {
                features.push(SpamFeature {
                    name: "dangerous_attachment".to_string(),
                    weight: 0.9,
                    description: format!("Dangerous attachment type: {}", attachment),
                });
            }
        }

        // Check sender/reply-to mismatch
        if let Some(ref reply_to) = email.reply_to {
            if !email.from.contains(&extract_domain(reply_to)) {
                features.push(SpamFeature {
                    name: "sender_mismatch".to_string(),
                    weight: 0.6,
                    description: "Reply-To domain differs from sender".to_string(),
                });
            }
        }

        features
    }

    /// Tokenize email content
    fn tokenize(&self, subject: &str, body: &str) -> Vec<String> {
        let combined = format!("{} {}", subject, body).to_lowercase();

        combined
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() >= 3 && s.len() <= 20)
            .map(|s| s.to_string())
            .collect()
    }

    /// Train on a new email
    pub fn train(&self, email: &EmailContent, is_spam: bool) {
        let tokens = self.tokenize(&email.subject, &email.body);

        if is_spam {
            let mut probs = self.spam_word_probs.write().unwrap();
            let mut count = self.spam_count.write().unwrap();
            *count += 1;

            for token in tokens {
                let entry = probs.entry(token).or_insert(0.01);
                // Incremental update
                *entry = (*entry * (*count as f32 - 1.0) + 1.0) / *count as f32;
            }
        } else {
            let mut probs = self.ham_word_probs.write().unwrap();
            let mut count = self.ham_count.write().unwrap();
            *count += 1;

            for token in tokens {
                let entry = probs.entry(token).or_insert(0.01);
                *entry = (*entry * (*count as f32 - 1.0) + 1.0) / *count as f32;
            }
        }

        debug!(is_spam = is_spam, "Trained on email");
    }
}

impl Default for SpamDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Email content for analysis
#[derive(Debug, Clone)]
pub struct EmailContent {
    pub from: String,
    pub reply_to: Option<String>,
    pub subject: String,
    pub body: String,
    pub attachment_names: Vec<String>,
    pub received_headers: Vec<String>,
}

fn is_known_legitimate_domain(domain: &str) -> bool {
    let known_domains = [
        "google.com",
        "gmail.com",
        "microsoft.com",
        "outlook.com",
        "apple.com",
        "amazon.com",
        "github.com",
        "linkedin.com",
        "twitter.com",
        "facebook.com",
    ];

    known_domains.iter().any(|d| domain.ends_with(d))
}

fn extract_domain(email: &str) -> String {
    email
        .split('@')
        .nth(1)
        .unwrap_or("")
        .split('>')
        .next()
        .unwrap_or("")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spam_detection() {
        let detector = SpamDetector::new();

        let spam_email = EmailContent {
            from: "winner@lottery.spam".to_string(),
            reply_to: None,
            subject: "CONGRATULATIONS! You won $1,000,000!".to_string(),
            body: "Click here to claim your lottery winnings. Act now! Limited time offer."
                .to_string(),
            attachment_names: vec![],
            received_headers: vec![],
        };

        let result = detector.analyze(&spam_email);
        assert!(result.spam_probability > 0.7);
        assert_eq!(result.classification, SpamClassification::Spam);
    }

    #[test]
    fn test_ham_detection() {
        let detector = SpamDetector::new();

        let ham_email = EmailContent {
            from: "colleague@company.com".to_string(),
            reply_to: None,
            subject: "Meeting update for project review".to_string(),
            body: "Hi team, just a quick update on the project schedule. Please review the attached document and provide feedback by Friday. Best regards."
                .to_string(),
            attachment_names: vec!["report.pdf".to_string()],
            received_headers: vec![],
        };

        let result = detector.analyze(&ham_email);
        assert!(result.spam_probability < 0.5);
        assert_eq!(result.classification, SpamClassification::Ham);
    }

    #[test]
    fn test_dangerous_attachment() {
        let detector = SpamDetector::new();

        let email = EmailContent {
            from: "sender@example.com".to_string(),
            reply_to: None,
            subject: "Important update".to_string(),
            body: "Please see attached".to_string(),
            attachment_names: vec!["invoice.exe".to_string()],
            received_headers: vec![],
        };

        let result = detector.analyze(&email);
        assert!(result.features.iter().any(|f| f.name == "dangerous_attachment"));
    }
}
