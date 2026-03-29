// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Smart Email Categorization & Priority Scoring
//!
//! AI-powered email classification using local ML models.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EmailCategory {
    Primary,      // Important personal and work emails
    Social,       // Social network notifications
    Promotions,   // Marketing, deals, offers
    Updates,      // Bills, receipts, confirmations
    Forums,       // Mailing lists, group discussions
    Newsletters,  // Subscribed newsletters
    Transactional,// Order confirmations, shipping
    Support,      // Customer service communications
    Calendar,     // Meeting invites, calendar updates
    Finance,      // Banking, investments
    Travel,       // Flight, hotel, travel bookings
    Security,     // Security alerts, password resets
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryPrediction {
    pub category: EmailCategory,
    pub confidence: f32,
    pub reasoning: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityScore {
    pub score: f32,        // 0.0 to 1.0 (1.0 = highest priority)
    pub level: PriorityLevel,
    pub factors: Vec<PriorityFactor>,
    pub suggested_action: SuggestedAction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PriorityLevel {
    Urgent,    // 0.9-1.0: Needs immediate attention
    High,      // 0.7-0.9: Important, respond soon
    Normal,    // 0.4-0.7: Regular priority
    Low,       // 0.2-0.4: Can wait
    Minimal,   // 0.0-0.2: FYI only
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityFactor {
    pub factor: String,
    pub weight: f32,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestedAction {
    ReplyUrgently,
    ReplyToday,
    ReviewLater,
    Archive,
    Delegate(String),
    FollowUp(i64),  // Unix timestamp
    NoActionNeeded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailFeatures {
    // Sender features
    pub sender_domain: String,
    pub sender_in_contacts: bool,
    pub sender_trust_score: f32,
    pub sender_reply_history: u32,
    pub sender_is_vip: bool,

    // Content features
    pub subject: String,
    pub body_preview: String,
    pub word_count: usize,
    pub has_attachments: bool,
    pub attachment_types: Vec<String>,
    pub has_links: bool,
    pub link_count: usize,

    // Thread features
    pub is_reply: bool,
    pub thread_length: usize,
    pub user_participated: bool,
    pub mentioned_by_name: bool,

    // Temporal features
    pub sent_time_hour: u8,
    pub sent_day_of_week: u8,
    pub response_expected_hours: Option<f32>,

    // Historical features
    pub similar_emails_opened_rate: f32,
    pub similar_emails_replied_rate: f32,
    pub similar_emails_archived_rate: f32,
}

// ============================================================================
// Categorization Engine
// ============================================================================

pub struct CategorizationEngine {
    keyword_patterns: HashMap<EmailCategory, Vec<String>>,
    domain_mappings: HashMap<String, EmailCategory>,
    model_weights: CategoryModelWeights,
}

#[derive(Debug, Clone)]
struct CategoryModelWeights {
    subject_weight: f32,
    body_weight: f32,
    sender_weight: f32,
    header_weight: f32,
}

impl Default for CategoryModelWeights {
    fn default() -> Self {
        Self {
            subject_weight: 0.35,
            body_weight: 0.30,
            sender_weight: 0.25,
            header_weight: 0.10,
        }
    }
}

impl CategorizationEngine {
    pub fn new() -> Self {
        let mut keyword_patterns = HashMap::new();
        let mut domain_mappings = HashMap::new();

        // Social patterns
        keyword_patterns.insert(
            EmailCategory::Social,
            vec![
                "followed you", "liked your", "commented on", "tagged you",
                "friend request", "connection request", "mentioned you",
                "new follower", "reacted to", "shared your",
            ].into_iter().map(String::from).collect(),
        );

        // Promotions patterns
        keyword_patterns.insert(
            EmailCategory::Promotions,
            vec![
                "% off", "sale", "discount", "deal", "offer", "promo",
                "limited time", "exclusive", "free shipping", "subscribe",
                "unsubscribe", "marketing", "newsletter",
            ].into_iter().map(String::from).collect(),
        );

        // Updates patterns
        keyword_patterns.insert(
            EmailCategory::Updates,
            vec![
                "receipt", "invoice", "bill", "statement", "confirmation",
                "order #", "tracking", "shipped", "delivered", "payment",
            ].into_iter().map(String::from).collect(),
        );

        // Security patterns
        keyword_patterns.insert(
            EmailCategory::Security,
            vec![
                "password reset", "verify your", "security alert", "suspicious",
                "two-factor", "2fa", "login attempt", "account security",
                "verification code", "confirm your identity",
            ].into_iter().map(String::from).collect(),
        );

        // Calendar patterns
        keyword_patterns.insert(
            EmailCategory::Calendar,
            vec![
                "meeting invite", "calendar invite", "event reminder",
                "rsvp", "invitation to", "scheduled for", "join meeting",
                "meeting request", "availability",
            ].into_iter().map(String::from).collect(),
        );

        // Domain mappings
        domain_mappings.insert("facebook.com".into(), EmailCategory::Social);
        domain_mappings.insert("twitter.com".into(), EmailCategory::Social);
        domain_mappings.insert("linkedin.com".into(), EmailCategory::Social);
        domain_mappings.insert("instagram.com".into(), EmailCategory::Social);

        domain_mappings.insert("amazon.com".into(), EmailCategory::Updates);
        domain_mappings.insert("ups.com".into(), EmailCategory::Updates);
        domain_mappings.insert("fedex.com".into(), EmailCategory::Updates);

        domain_mappings.insert("chase.com".into(), EmailCategory::Finance);
        domain_mappings.insert("bankofamerica.com".into(), EmailCategory::Finance);
        domain_mappings.insert("paypal.com".into(), EmailCategory::Finance);

        Self {
            keyword_patterns,
            domain_mappings,
            model_weights: CategoryModelWeights::default(),
        }
    }

    pub fn categorize(&self, features: &EmailFeatures) -> CategoryPrediction {
        let mut scores: HashMap<EmailCategory, f32> = HashMap::new();
        let mut reasoning: Vec<String> = Vec::new();

        // Check domain mapping first
        if let Some(category) = self.domain_mappings.get(&features.sender_domain) {
            scores.insert(category.clone(), 0.8);
            reasoning.push(format!("Sender domain {} maps to {:?}", features.sender_domain, category));
        }

        // Keyword analysis
        let text = format!("{} {}", features.subject, features.body_preview).to_lowercase();
        for (category, keywords) in &self.keyword_patterns {
            let mut match_count = 0;
            for keyword in keywords {
                if text.contains(keyword) {
                    match_count += 1;
                }
            }
            if match_count > 0 {
                let score = (match_count as f32 * 0.2).min(0.9);
                *scores.entry(category.clone()).or_insert(0.0) += score;
                reasoning.push(format!("Found {} keywords for {:?}", match_count, category));
            }
        }

        // Sender-based signals
        if features.sender_in_contacts && features.sender_trust_score > 0.7 {
            *scores.entry(EmailCategory::Primary).or_insert(0.0) += 0.3;
            reasoning.push("Sender is in contacts with high trust".into());
        }

        if features.sender_is_vip {
            *scores.entry(EmailCategory::Primary).or_insert(0.0) += 0.4;
            reasoning.push("Sender is marked as VIP".into());
        }

        // Thread participation
        if features.user_participated {
            *scores.entry(EmailCategory::Primary).or_insert(0.0) += 0.2;
            reasoning.push("User has participated in this thread".into());
        }

        if features.mentioned_by_name {
            *scores.entry(EmailCategory::Primary).or_insert(0.0) += 0.3;
            reasoning.push("User mentioned by name".into());
        }

        // Default to Primary if no clear category
        if scores.is_empty() {
            scores.insert(EmailCategory::Primary, 0.5);
            reasoning.push("No clear category signals, defaulting to Primary".into());
        }

        // Find best category
        let (best_category, best_score) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(c, s)| (c.clone(), *s))
            .unwrap_or((EmailCategory::Primary, 0.5));

        CategoryPrediction {
            category: best_category,
            confidence: best_score.min(1.0),
            reasoning,
        }
    }
}

// ============================================================================
// Priority Scoring Engine
// ============================================================================

pub struct PriorityEngine {
    vip_senders: Vec<String>,
    urgent_keywords: Vec<String>,
    time_sensitive_patterns: Vec<String>,
}

impl PriorityEngine {
    pub fn new() -> Self {
        Self {
            vip_senders: vec![],
            urgent_keywords: vec![
                "urgent", "asap", "immediately", "critical", "important",
                "deadline", "time-sensitive", "action required", "response needed",
                "by end of day", "by eod", "please respond", "waiting on you",
            ].into_iter().map(String::from).collect(),
            time_sensitive_patterns: vec![
                "expires", "ending soon", "last chance", "final reminder",
                "overdue", "past due", "action needed by",
            ].into_iter().map(String::from).collect(),
        }
    }

    pub fn add_vip_sender(&mut self, email: &str) {
        self.vip_senders.push(email.to_lowercase());
    }

    pub fn calculate_priority(&self, features: &EmailFeatures) -> PriorityScore {
        let mut score = 0.5f32;
        let mut factors = Vec::new();

        // VIP sender boost
        if features.sender_is_vip || self.vip_senders.iter().any(|v| features.sender_domain.contains(v)) {
            score += 0.25;
            factors.push(PriorityFactor {
                factor: "vip_sender".into(),
                weight: 0.25,
                description: "Email is from a VIP contact".into(),
            });
        }

        // Trust score influence
        if features.sender_trust_score > 0.8 {
            score += 0.15;
            factors.push(PriorityFactor {
                factor: "high_trust".into(),
                weight: 0.15,
                description: format!("Sender has high trust score ({:.2})", features.sender_trust_score),
            });
        } else if features.sender_trust_score < 0.3 {
            score -= 0.1;
            factors.push(PriorityFactor {
                factor: "low_trust".into(),
                weight: -0.1,
                description: "Sender has low trust score".into(),
            });
        }

        // In contacts boost
        if features.sender_in_contacts {
            score += 0.1;
            factors.push(PriorityFactor {
                factor: "in_contacts".into(),
                weight: 0.1,
                description: "Sender is in contacts".into(),
            });
        }

        // Reply history
        if features.sender_reply_history > 5 {
            score += 0.1;
            factors.push(PriorityFactor {
                factor: "reply_history".into(),
                weight: 0.1,
                description: format!("You've replied to {} emails from this sender", features.sender_reply_history),
            });
        }

        // Thread participation
        if features.user_participated {
            score += 0.15;
            factors.push(PriorityFactor {
                factor: "thread_participant".into(),
                weight: 0.15,
                description: "You're part of this conversation".into(),
            });
        }

        // Mentioned by name
        if features.mentioned_by_name {
            score += 0.2;
            factors.push(PriorityFactor {
                factor: "mentioned".into(),
                weight: 0.2,
                description: "You were mentioned by name".into(),
            });
        }

        // Urgent keywords
        let text = format!("{} {}", features.subject, features.body_preview).to_lowercase();
        let urgent_matches: Vec<_> = self.urgent_keywords
            .iter()
            .filter(|k| text.contains(*k))
            .collect();

        if !urgent_matches.is_empty() {
            let boost = (urgent_matches.len() as f32 * 0.1).min(0.3);
            score += boost;
            factors.push(PriorityFactor {
                factor: "urgent_keywords".into(),
                weight: boost,
                description: format!("Contains urgent keywords: {}", urgent_matches.iter().take(3).cloned().collect::<Vec<_>>().join(", ")),
            });
        }

        // Time sensitive patterns
        let time_matches: Vec<_> = self.time_sensitive_patterns
            .iter()
            .filter(|p| text.contains(*p))
            .collect();

        if !time_matches.is_empty() {
            score += 0.15;
            factors.push(PriorityFactor {
                factor: "time_sensitive".into(),
                weight: 0.15,
                description: "Email appears time-sensitive".into(),
            });
        }

        // Response expected
        if let Some(hours) = features.response_expected_hours {
            if hours < 4.0 {
                score += 0.2;
                factors.push(PriorityFactor {
                    factor: "quick_response_expected".into(),
                    weight: 0.2,
                    description: format!("Response expected within {:.0} hours", hours),
                });
            }
        }

        // Historical engagement
        if features.similar_emails_replied_rate > 0.7 {
            score += 0.1;
            factors.push(PriorityFactor {
                factor: "high_reply_rate".into(),
                weight: 0.1,
                description: "You typically reply to similar emails".into(),
            });
        }

        // Clamp score
        score = score.clamp(0.0, 1.0);

        // Determine level
        let level = match score {
            s if s >= 0.9 => PriorityLevel::Urgent,
            s if s >= 0.7 => PriorityLevel::High,
            s if s >= 0.4 => PriorityLevel::Normal,
            s if s >= 0.2 => PriorityLevel::Low,
            _ => PriorityLevel::Minimal,
        };

        // Suggest action
        let suggested_action = self.suggest_action(&level, features);

        PriorityScore {
            score,
            level,
            factors,
            suggested_action,
        }
    }

    fn suggest_action(&self, level: &PriorityLevel, features: &EmailFeatures) -> SuggestedAction {
        match level {
            PriorityLevel::Urgent => SuggestedAction::ReplyUrgently,
            PriorityLevel::High => SuggestedAction::ReplyToday,
            PriorityLevel::Normal => {
                if features.is_reply && features.user_participated {
                    SuggestedAction::ReplyToday
                } else {
                    SuggestedAction::ReviewLater
                }
            }
            PriorityLevel::Low => SuggestedAction::ReviewLater,
            PriorityLevel::Minimal => {
                if features.similar_emails_archived_rate > 0.8 {
                    SuggestedAction::Archive
                } else {
                    SuggestedAction::NoActionNeeded
                }
            }
        }
    }
}

// ============================================================================
// Combined Intelligence Service
// ============================================================================

pub struct EmailIntelligenceService {
    categorizer: CategorizationEngine,
    prioritizer: PriorityEngine,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailIntelligence {
    pub category: CategoryPrediction,
    pub priority: PriorityScore,
    pub smart_labels: Vec<String>,
    pub summary: Option<String>,
}

impl EmailIntelligenceService {
    pub fn new() -> Self {
        Self {
            categorizer: CategorizationEngine::new(),
            prioritizer: PriorityEngine::new(),
        }
    }

    pub fn analyze(&self, features: &EmailFeatures) -> EmailIntelligence {
        let category = self.categorizer.categorize(features);
        let priority = self.prioritizer.calculate_priority(features);

        // Generate smart labels
        let mut smart_labels = Vec::new();

        match &category.category {
            EmailCategory::Calendar => smart_labels.push("📅 Calendar".into()),
            EmailCategory::Finance => smart_labels.push("💰 Finance".into()),
            EmailCategory::Security => smart_labels.push("🔒 Security".into()),
            EmailCategory::Travel => smart_labels.push("✈️ Travel".into()),
            _ => {}
        }

        match &priority.level {
            PriorityLevel::Urgent => smart_labels.push("🔴 Urgent".into()),
            PriorityLevel::High => smart_labels.push("🟠 High Priority".into()),
            _ => {}
        }

        if features.has_attachments {
            smart_labels.push("📎 Attachments".into());
        }

        if features.mentioned_by_name {
            smart_labels.push("👤 Mentioned".into());
        }

        EmailIntelligence {
            category,
            priority,
            smart_labels,
            summary: None, // Would be filled by LLM summarization
        }
    }
}

impl Default for EmailIntelligenceService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_features() -> EmailFeatures {
        EmailFeatures {
            sender_domain: "company.com".into(),
            sender_in_contacts: true,
            sender_trust_score: 0.85,
            sender_reply_history: 10,
            sender_is_vip: false,
            subject: "Project update needed urgently".into(),
            body_preview: "Hi, I need the project status update by EOD...".into(),
            word_count: 150,
            has_attachments: false,
            attachment_types: vec![],
            has_links: false,
            link_count: 0,
            is_reply: false,
            thread_length: 1,
            user_participated: false,
            mentioned_by_name: true,
            sent_time_hour: 9,
            sent_day_of_week: 2,
            response_expected_hours: Some(4.0),
            similar_emails_opened_rate: 0.9,
            similar_emails_replied_rate: 0.8,
            similar_emails_archived_rate: 0.1,
        }
    }

    #[test]
    fn test_categorization() {
        let engine = CategorizationEngine::new();
        let features = sample_features();
        let result = engine.categorize(&features);
        assert_eq!(result.category, EmailCategory::Primary);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_priority_scoring() {
        let engine = PriorityEngine::new();
        let features = sample_features();
        let result = engine.calculate_priority(&features);
        assert!(result.score > 0.7);
        assert!(matches!(result.level, PriorityLevel::High | PriorityLevel::Urgent));
    }
}
