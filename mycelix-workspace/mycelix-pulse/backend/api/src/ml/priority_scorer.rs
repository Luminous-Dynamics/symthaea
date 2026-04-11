// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Priority Scoring and Smart Replies
//!
//! ML-based email importance ranking and suggested responses

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc, Timelike, Datelike};

/// Email priority result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityResult {
    /// Priority score (0.0 - 1.0)
    pub score: f32,
    /// Priority level
    pub level: PriorityLevel,
    /// Factors contributing to priority
    pub factors: Vec<PriorityFactor>,
    /// Suggested time to respond (in hours)
    pub suggested_response_time: Option<u32>,
    /// Should notify immediately
    pub notify_immediately: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PriorityLevel {
    Critical,    // Needs immediate attention
    High,        // Important, respond soon
    Normal,      // Regular priority
    Low,         // Can wait
    Bulk,        // Newsletters, notifications
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityFactor {
    pub name: String,
    pub weight: f32,
    pub description: String,
}

/// Smart reply suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartReply {
    /// Suggested reply text
    pub text: String,
    /// Confidence in this suggestion
    pub confidence: f32,
    /// Reply type
    pub reply_type: ReplyType,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ReplyType {
    Acknowledgment,
    Affirmative,
    Negative,
    Question,
    ThankYou,
    Scheduling,
    Custom,
}

/// Priority scorer for emails
pub struct PriorityScorer {
    /// Contact interaction history (email -> interaction count)
    contact_history: HashMap<String, ContactStats>,
    /// Keywords that indicate urgency
    urgency_keywords: Vec<(&'static str, f32)>,
    /// Keywords that indicate low priority
    low_priority_keywords: Vec<(&'static str, f32)>,
}

#[derive(Debug, Clone, Default)]
struct ContactStats {
    interaction_count: u32,
    avg_response_time_hours: f32,
    is_vip: bool,
    last_interaction: Option<DateTime<Utc>>,
}

impl PriorityScorer {
    pub fn new() -> Self {
        Self {
            contact_history: HashMap::new(),
            urgency_keywords: vec![
                ("urgent", 0.9),
                ("asap", 0.85),
                ("immediately", 0.9),
                ("critical", 0.95),
                ("emergency", 0.98),
                ("deadline", 0.7),
                ("today", 0.6),
                ("important", 0.65),
                ("priority", 0.7),
                ("action required", 0.8),
                ("response needed", 0.75),
                ("please respond", 0.65),
                ("time sensitive", 0.85),
                ("eod", 0.7),
                ("by end of day", 0.7),
            ],
            low_priority_keywords: vec![
                ("newsletter", -0.4),
                ("unsubscribe", -0.3),
                ("no reply", -0.2),
                ("automated", -0.3),
                ("notification", -0.25),
                ("digest", -0.35),
                ("weekly", -0.2),
                ("monthly", -0.25),
                ("fyi", -0.15),
                ("for your information", -0.15),
            ],
        }
    }

    /// Score email priority
    pub fn score(&self, email: &EmailContext) -> PriorityResult {
        let mut factors = Vec::new();
        let mut total_score = 0.5f32; // Start at neutral

        // Factor 1: Sender relationship
        if let Some(stats) = self.contact_history.get(&email.from) {
            if stats.is_vip {
                total_score += 0.3;
                factors.push(PriorityFactor {
                    name: "vip_sender".to_string(),
                    weight: 0.3,
                    description: "Sender is marked as VIP".to_string(),
                });
            } else if stats.interaction_count > 10 {
                let bonus = (stats.interaction_count as f32 / 100.0).min(0.2);
                total_score += bonus;
                factors.push(PriorityFactor {
                    name: "frequent_contact".to_string(),
                    weight: bonus,
                    description: format!("Frequent contact ({} interactions)", stats.interaction_count),
                });
            }
        }

        // Factor 2: Trust score
        if let Some(trust) = email.trust_score {
            if trust >= 0.8 {
                total_score += 0.15;
                factors.push(PriorityFactor {
                    name: "high_trust".to_string(),
                    weight: 0.15,
                    description: "Sender has high trust score".to_string(),
                });
            } else if trust < 0.3 {
                total_score -= 0.2;
                factors.push(PriorityFactor {
                    name: "low_trust".to_string(),
                    weight: -0.2,
                    description: "Sender has low trust score".to_string(),
                });
            }
        }

        // Factor 3: Direct recipient vs CC
        if email.is_direct_recipient {
            total_score += 0.15;
            factors.push(PriorityFactor {
                name: "direct_recipient".to_string(),
                weight: 0.15,
                description: "You are the direct recipient".to_string(),
            });
        } else if email.is_cc {
            total_score -= 0.1;
            factors.push(PriorityFactor {
                name: "cc_recipient".to_string(),
                weight: -0.1,
                description: "You are CC'd".to_string(),
            });
        }

        // Factor 4: Urgency keywords
        let subject_lower = email.subject.to_lowercase();
        let body_lower = email.body_preview.to_lowercase();
        let content = format!("{} {}", subject_lower, body_lower);

        for (keyword, weight) in &self.urgency_keywords {
            if content.contains(keyword) {
                total_score += weight;
                factors.push(PriorityFactor {
                    name: format!("urgency_{}", keyword.replace(' ', "_")),
                    weight: *weight,
                    description: format!("Contains urgency indicator: '{}'", keyword),
                });
                break; // Only count once
            }
        }

        for (keyword, weight) in &self.low_priority_keywords {
            if content.contains(keyword) {
                total_score += weight; // weight is negative
                factors.push(PriorityFactor {
                    name: format!("low_priority_{}", keyword.replace(' ', "_")),
                    weight: *weight,
                    description: format!("Low priority indicator: '{}'", keyword),
                });
                break;
            }
        }

        // Factor 5: Thread context
        if email.is_reply_expected {
            total_score += 0.2;
            factors.push(PriorityFactor {
                name: "reply_expected".to_string(),
                weight: 0.2,
                description: "This is part of an active thread".to_string(),
            });
        }

        // Factor 6: Time factors
        let hour = Utc::now().hour();
        let is_business_hours = hour >= 9 && hour < 18;
        let is_weekday = Utc::now().weekday().num_days_from_monday() < 5;

        if is_business_hours && is_weekday {
            // Slight boost for work hours
            total_score += 0.05;
        }

        // Factor 7: Has deadline mentioned
        if content.contains("by") && (content.contains("friday") || content.contains("monday")
            || content.contains("tomorrow") || content.contains("end of")) {
            total_score += 0.2;
            factors.push(PriorityFactor {
                name: "has_deadline".to_string(),
                weight: 0.2,
                description: "Contains specific deadline".to_string(),
            });
        }

        // Clamp score
        total_score = total_score.clamp(0.0, 1.0);

        // Determine level
        let level = if total_score >= 0.85 {
            PriorityLevel::Critical
        } else if total_score >= 0.7 {
            PriorityLevel::High
        } else if total_score >= 0.4 {
            PriorityLevel::Normal
        } else if total_score >= 0.2 {
            PriorityLevel::Low
        } else {
            PriorityLevel::Bulk
        };

        // Suggested response time
        let suggested_response_time = match level {
            PriorityLevel::Critical => Some(1),
            PriorityLevel::High => Some(4),
            PriorityLevel::Normal => Some(24),
            PriorityLevel::Low => Some(72),
            PriorityLevel::Bulk => None,
        };

        PriorityResult {
            score: total_score,
            level,
            factors,
            suggested_response_time,
            notify_immediately: level == PriorityLevel::Critical,
        }
    }

    /// Update contact history
    pub fn record_interaction(&mut self, email: &str, is_vip: bool) {
        let stats = self.contact_history.entry(email.to_string()).or_default();
        stats.interaction_count += 1;
        stats.last_interaction = Some(Utc::now());
        if is_vip {
            stats.is_vip = true;
        }
    }

    /// Mark contact as VIP
    pub fn set_vip(&mut self, email: &str, is_vip: bool) {
        let stats = self.contact_history.entry(email.to_string()).or_default();
        stats.is_vip = is_vip;
    }
}

impl Default for PriorityScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Smart reply generator
pub struct SmartReplyGenerator {
    /// Reply templates by category
    templates: HashMap<ReplyType, Vec<&'static str>>,
}

impl SmartReplyGenerator {
    pub fn new() -> Self {
        let mut templates = HashMap::new();

        templates.insert(ReplyType::Acknowledgment, vec![
            "Got it, thanks!",
            "Thanks for letting me know.",
            "Acknowledged, thank you.",
            "Noted, thanks!",
        ]);

        templates.insert(ReplyType::Affirmative, vec![
            "Yes, that works for me.",
            "Sounds good!",
            "I'm on board with that.",
            "Agreed, let's proceed.",
            "Yes, I can do that.",
        ]);

        templates.insert(ReplyType::Negative, vec![
            "Unfortunately, I won't be able to.",
            "I'm sorry, but that doesn't work for me.",
            "I'll have to pass on this one.",
            "Not this time, but thanks for asking.",
        ]);

        templates.insert(ReplyType::ThankYou, vec![
            "Thank you so much!",
            "Thanks, I really appreciate it!",
            "Thank you for your help!",
            "Thanks for getting back to me!",
        ]);

        templates.insert(ReplyType::Scheduling, vec![
            "How about Tuesday at 2pm?",
            "I'm free tomorrow afternoon.",
            "Let me check my calendar and get back to you.",
            "Would next week work for you?",
        ]);

        templates.insert(ReplyType::Question, vec![
            "Could you provide more details?",
            "When do you need this by?",
            "Who else should be involved?",
            "What's the priority on this?",
        ]);

        Self { templates }
    }

    /// Generate smart reply suggestions
    pub fn generate(&self, email: &EmailContext) -> Vec<SmartReply> {
        let mut suggestions = Vec::new();
        let content = format!("{} {}", email.subject, email.body_preview).to_lowercase();

        // Detect email type and generate appropriate replies
        if content.contains("meeting") || content.contains("schedule") || content.contains("calendar") {
            suggestions.extend(self.get_replies(ReplyType::Scheduling, 0.8));
            suggestions.extend(self.get_replies(ReplyType::Affirmative, 0.7));
        }

        if content.contains("thank") {
            suggestions.extend(self.get_replies(ReplyType::Acknowledgment, 0.85));
        }

        if content.contains("?") || content.contains("can you") || content.contains("would you") {
            suggestions.extend(self.get_replies(ReplyType::Affirmative, 0.75));
            suggestions.extend(self.get_replies(ReplyType::Negative, 0.6));
        }

        if content.contains("fyi") || content.contains("update") || content.contains("letting you know") {
            suggestions.extend(self.get_replies(ReplyType::Acknowledgment, 0.9));
            suggestions.extend(self.get_replies(ReplyType::ThankYou, 0.7));
        }

        if content.contains("help") || content.contains("assist") {
            suggestions.extend(self.get_replies(ReplyType::ThankYou, 0.85));
        }

        // Sort by confidence and take top 3
        suggestions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        suggestions.truncate(3);

        // If no suggestions, add generic ones
        if suggestions.is_empty() {
            suggestions.push(SmartReply {
                text: "Thanks for your email. I'll get back to you soon.".to_string(),
                confidence: 0.5,
                reply_type: ReplyType::Custom,
            });
        }

        suggestions
    }

    fn get_replies(&self, reply_type: ReplyType, base_confidence: f32) -> Vec<SmartReply> {
        self.templates
            .get(&reply_type)
            .map(|templates| {
                templates
                    .iter()
                    .take(2)
                    .enumerate()
                    .map(|(i, text)| SmartReply {
                        text: text.to_string(),
                        confidence: base_confidence - (i as f32 * 0.1),
                        reply_type,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Default for SmartReplyGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Email context for scoring
#[derive(Debug, Clone)]
pub struct EmailContext {
    pub from: String,
    pub subject: String,
    pub body_preview: String,
    pub trust_score: Option<f32>,
    pub is_direct_recipient: bool,
    pub is_cc: bool,
    pub is_reply_expected: bool,
    pub thread_length: u32,
    pub received_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_urgent() {
        let scorer = PriorityScorer::new();

        let email = EmailContext {
            from: "boss@company.com".to_string(),
            subject: "URGENT: Need response ASAP".to_string(),
            body_preview: "Please respond immediately regarding the critical issue.".to_string(),
            trust_score: Some(0.9),
            is_direct_recipient: true,
            is_cc: false,
            is_reply_expected: true,
            thread_length: 3,
            received_at: Utc::now(),
        };

        let result = scorer.score(&email);
        assert!(result.score >= 0.7);
        assert!(matches!(result.level, PriorityLevel::High | PriorityLevel::Critical));
    }

    #[test]
    fn test_priority_newsletter() {
        let scorer = PriorityScorer::new();

        let email = EmailContext {
            from: "newsletter@marketing.com".to_string(),
            subject: "Weekly Newsletter - December Edition".to_string(),
            body_preview: "Click to unsubscribe. This is an automated notification.".to_string(),
            trust_score: Some(0.4),
            is_direct_recipient: false,
            is_cc: false,
            is_reply_expected: false,
            thread_length: 1,
            received_at: Utc::now(),
        };

        let result = scorer.score(&email);
        assert!(result.score < 0.4);
        assert!(matches!(result.level, PriorityLevel::Low | PriorityLevel::Bulk));
    }

    #[test]
    fn test_smart_replies_meeting() {
        let generator = SmartReplyGenerator::new();

        let email = EmailContext {
            from: "colleague@company.com".to_string(),
            subject: "Meeting request".to_string(),
            body_preview: "Can we schedule a meeting for next week?".to_string(),
            trust_score: Some(0.8),
            is_direct_recipient: true,
            is_cc: false,
            is_reply_expected: true,
            thread_length: 1,
            received_at: Utc::now(),
        };

        let replies = generator.generate(&email);
        assert!(!replies.is_empty());
        assert!(replies.iter().any(|r| r.reply_type == ReplyType::Scheduling
            || r.reply_type == ReplyType::Affirmative));
    }
}
