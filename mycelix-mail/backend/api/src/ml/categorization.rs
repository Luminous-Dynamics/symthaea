// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Email Auto-Categorization
//!
//! Automatic label and folder assignment using ML

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Email category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailCategory {
    pub id: String,
    pub name: String,
    pub confidence: f64,
    pub source: CategorySource,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CategorySource {
    Keyword,
    Sender,
    Pattern,
    ML,
    User,
}

/// Built-in category types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BuiltInCategory {
    // Priority
    Primary,
    Social,
    Promotions,
    Updates,
    Forums,

    // Type
    Newsletter,
    Receipt,
    Shipping,
    Travel,
    Finance,
    Calendar,
    Document,

    // Action
    ActionRequired,
    FYI,
    Delegated,
    Waiting,
}

impl BuiltInCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Primary => "primary",
            Self::Social => "social",
            Self::Promotions => "promotions",
            Self::Updates => "updates",
            Self::Forums => "forums",
            Self::Newsletter => "newsletter",
            Self::Receipt => "receipt",
            Self::Shipping => "shipping",
            Self::Travel => "travel",
            Self::Finance => "finance",
            Self::Calendar => "calendar",
            Self::Document => "document",
            Self::ActionRequired => "action_required",
            Self::FYI => "fyi",
            Self::Delegated => "delegated",
            Self::Waiting => "waiting",
        }
    }
}

/// Email content for categorization
#[derive(Debug, Clone)]
pub struct EmailForCategorization {
    pub id: Uuid,
    pub from: String,
    pub to: Vec<String>,
    pub subject: String,
    pub body: String,
    pub headers: HashMap<String, String>,
}

/// Categorization rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategorizationRule {
    pub id: Uuid,
    pub name: String,
    pub conditions: Vec<RuleCondition>,
    pub category: String,
    pub priority: i32,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleCondition {
    pub field: RuleField,
    pub operator: RuleOperator,
    pub value: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RuleField {
    From,
    To,
    Subject,
    Body,
    Header,
    Domain,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RuleOperator {
    Contains,
    NotContains,
    Equals,
    StartsWith,
    EndsWith,
    Matches, // Regex
}

/// Email categorizer
pub struct EmailCategorizer {
    rules: Vec<CategorizationRule>,
    sender_patterns: HashMap<String, BuiltInCategory>,
    keyword_patterns: HashMap<String, Vec<BuiltInCategory>>,
}

impl EmailCategorizer {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            sender_patterns: Self::default_sender_patterns(),
            keyword_patterns: Self::default_keyword_patterns(),
        }
    }

    pub fn with_rules(rules: Vec<CategorizationRule>) -> Self {
        let mut categorizer = Self::new();
        categorizer.rules = rules;
        categorizer
    }

    fn default_sender_patterns() -> HashMap<String, BuiltInCategory> {
        let mut patterns = HashMap::new();

        // Social networks
        for domain in ["facebook.com", "twitter.com", "linkedin.com", "instagram.com"] {
            patterns.insert(domain.to_string(), BuiltInCategory::Social);
        }

        // E-commerce / Receipts
        for domain in ["amazon.com", "ebay.com", "stripe.com", "paypal.com"] {
            patterns.insert(domain.to_string(), BuiltInCategory::Receipt);
        }

        // Shipping
        for domain in ["ups.com", "fedex.com", "usps.com", "dhl.com"] {
            patterns.insert(domain.to_string(), BuiltInCategory::Shipping);
        }

        // Travel
        for domain in ["airbnb.com", "booking.com", "expedia.com", "delta.com", "united.com"] {
            patterns.insert(domain.to_string(), BuiltInCategory::Travel);
        }

        // Finance
        for domain in ["chase.com", "bankofamerica.com", "wellsfargo.com", "venmo.com"] {
            patterns.insert(domain.to_string(), BuiltInCategory::Finance);
        }

        patterns
    }

    fn default_keyword_patterns() -> HashMap<String, Vec<BuiltInCategory>> {
        let mut patterns = HashMap::new();

        // Newsletter indicators
        patterns.insert("unsubscribe".to_string(), vec![BuiltInCategory::Newsletter, BuiltInCategory::Promotions]);
        patterns.insert("newsletter".to_string(), vec![BuiltInCategory::Newsletter]);
        patterns.insert("weekly digest".to_string(), vec![BuiltInCategory::Newsletter]);

        // Receipt indicators
        patterns.insert("order confirmation".to_string(), vec![BuiltInCategory::Receipt]);
        patterns.insert("payment received".to_string(), vec![BuiltInCategory::Receipt]);
        patterns.insert("invoice".to_string(), vec![BuiltInCategory::Receipt, BuiltInCategory::Finance]);
        patterns.insert("receipt".to_string(), vec![BuiltInCategory::Receipt]);

        // Shipping indicators
        patterns.insert("shipped".to_string(), vec![BuiltInCategory::Shipping]);
        patterns.insert("tracking number".to_string(), vec![BuiltInCategory::Shipping]);
        patterns.insert("delivery".to_string(), vec![BuiltInCategory::Shipping]);
        patterns.insert("out for delivery".to_string(), vec![BuiltInCategory::Shipping]);

        // Travel indicators
        patterns.insert("flight confirmation".to_string(), vec![BuiltInCategory::Travel]);
        patterns.insert("booking confirmation".to_string(), vec![BuiltInCategory::Travel]);
        patterns.insert("itinerary".to_string(), vec![BuiltInCategory::Travel]);
        patterns.insert("boarding pass".to_string(), vec![BuiltInCategory::Travel]);

        // Calendar indicators
        patterns.insert("invitation".to_string(), vec![BuiltInCategory::Calendar]);
        patterns.insert("calendar event".to_string(), vec![BuiltInCategory::Calendar]);
        patterns.insert("meeting request".to_string(), vec![BuiltInCategory::Calendar]);

        // Action indicators
        patterns.insert("action required".to_string(), vec![BuiltInCategory::ActionRequired]);
        patterns.insert("please review".to_string(), vec![BuiltInCategory::ActionRequired]);
        patterns.insert("approval needed".to_string(), vec![BuiltInCategory::ActionRequired]);
        patterns.insert("sign required".to_string(), vec![BuiltInCategory::ActionRequired]);

        // Promotion indicators
        patterns.insert("sale".to_string(), vec![BuiltInCategory::Promotions]);
        patterns.insert("% off".to_string(), vec![BuiltInCategory::Promotions]);
        patterns.insert("limited time".to_string(), vec![BuiltInCategory::Promotions]);
        patterns.insert("deal".to_string(), vec![BuiltInCategory::Promotions]);

        patterns
    }

    /// Categorize an email
    pub fn categorize(&self, email: &EmailForCategorization) -> Vec<EmailCategory> {
        let mut categories = Vec::new();
        let mut category_scores: HashMap<String, f64> = HashMap::new();

        // Check custom rules first
        for rule in &self.rules {
            if rule.enabled && self.matches_rule(email, rule) {
                *category_scores.entry(rule.category.clone()).or_insert(0.0) += 0.8;
            }
        }

        // Check sender patterns
        let from_domain = email.from.split('@').nth(1).unwrap_or("");
        if let Some(category) = self.sender_patterns.get(from_domain) {
            *category_scores.entry(category.as_str().to_string()).or_insert(0.0) += 0.7;
        }

        // Check keyword patterns
        let text = format!("{} {}", email.subject, email.body).to_lowercase();
        for (keyword, cats) in &self.keyword_patterns {
            if text.contains(keyword) {
                for cat in cats {
                    *category_scores.entry(cat.as_str().to_string()).or_insert(0.0) += 0.3;
                }
            }
        }

        // Check for list-unsubscribe header (newsletter indicator)
        if email.headers.contains_key("list-unsubscribe") {
            *category_scores.entry("newsletter".to_string()).or_insert(0.0) += 0.5;
        }

        // Check for calendar content
        if email.headers.get("content-type").map(|ct| ct.contains("calendar")).unwrap_or(false) {
            *category_scores.entry("calendar".to_string()).or_insert(0.0) += 0.9;
        }

        // Convert scores to categories
        for (cat_name, score) in category_scores {
            if score >= 0.3 {
                categories.push(EmailCategory {
                    id: cat_name.clone(),
                    name: Self::category_display_name(&cat_name),
                    confidence: score.min(1.0),
                    source: CategorySource::Pattern,
                });
            }
        }

        // Sort by confidence
        categories.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        // If no categories, mark as primary
        if categories.is_empty() {
            categories.push(EmailCategory {
                id: "primary".to_string(),
                name: "Primary".to_string(),
                confidence: 0.5,
                source: CategorySource::Pattern,
            });
        }

        categories
    }

    fn matches_rule(&self, email: &EmailForCategorization, rule: &CategorizationRule) -> bool {
        rule.conditions.iter().all(|condition| {
            let value = match condition.field {
                RuleField::From => &email.from,
                RuleField::Subject => &email.subject,
                RuleField::Body => &email.body,
                RuleField::To => &email.to.join(", "),
                RuleField::Domain => email.from.split('@').nth(1).unwrap_or(""),
                RuleField::Header => {
                    // Header field requires special handling
                    return false;
                }
            };

            match condition.operator {
                RuleOperator::Contains => value.to_lowercase().contains(&condition.value.to_lowercase()),
                RuleOperator::NotContains => !value.to_lowercase().contains(&condition.value.to_lowercase()),
                RuleOperator::Equals => value.to_lowercase() == condition.value.to_lowercase(),
                RuleOperator::StartsWith => value.to_lowercase().starts_with(&condition.value.to_lowercase()),
                RuleOperator::EndsWith => value.to_lowercase().ends_with(&condition.value.to_lowercase()),
                RuleOperator::Matches => {
                    regex::Regex::new(&condition.value)
                        .map(|re| re.is_match(value))
                        .unwrap_or(false)
                }
            }
        })
    }

    fn category_display_name(id: &str) -> String {
        match id {
            "primary" => "Primary",
            "social" => "Social",
            "promotions" => "Promotions",
            "updates" => "Updates",
            "forums" => "Forums",
            "newsletter" => "Newsletter",
            "receipt" => "Receipt",
            "shipping" => "Shipping",
            "travel" => "Travel",
            "finance" => "Finance",
            "calendar" => "Calendar",
            "document" => "Document",
            "action_required" => "Action Required",
            "fyi" => "FYI",
            "delegated" => "Delegated",
            "waiting" => "Waiting",
            other => other,
        }.to_string()
    }

    /// Suggest folder based on categories
    pub fn suggest_folder(&self, categories: &[EmailCategory]) -> Option<String> {
        let primary_category = categories.first()?;

        match primary_category.id.as_str() {
            "social" => Some("Social".to_string()),
            "promotions" => Some("Promotions".to_string()),
            "updates" | "newsletter" => Some("Updates".to_string()),
            "receipt" | "finance" => Some("Finance".to_string()),
            "shipping" => Some("Shipping".to_string()),
            "travel" => Some("Travel".to_string()),
            _ => None,
        }
    }

    /// Learn from user corrections
    pub fn learn_from_correction(
        &mut self,
        email: &EmailForCategorization,
        correct_category: &str,
    ) {
        // Extract patterns from this email to improve future categorization
        let from_domain = email.from.split('@').nth(1).unwrap_or("").to_string();

        if !from_domain.is_empty() {
            // Add sender pattern (would persist to database in production)
            if let Ok(category) = correct_category.parse::<BuiltInCategory>() {
                self.sender_patterns.insert(from_domain, category);
            }
        }
    }
}

impl Default for EmailCategorizer {
    fn default() -> Self {
        Self::new()
    }
}

// Implement FromStr for BuiltInCategory
impl std::str::FromStr for BuiltInCategory {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "primary" => Ok(Self::Primary),
            "social" => Ok(Self::Social),
            "promotions" => Ok(Self::Promotions),
            "updates" => Ok(Self::Updates),
            "forums" => Ok(Self::Forums),
            "newsletter" => Ok(Self::Newsletter),
            "receipt" => Ok(Self::Receipt),
            "shipping" => Ok(Self::Shipping),
            "travel" => Ok(Self::Travel),
            "finance" => Ok(Self::Finance),
            "calendar" => Ok(Self::Calendar),
            "document" => Ok(Self::Document),
            "action_required" => Ok(Self::ActionRequired),
            "fyi" => Ok(Self::FYI),
            "delegated" => Ok(Self::Delegated),
            "waiting" => Ok(Self::Waiting),
            _ => Err(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_receipt_categorization() {
        let categorizer = EmailCategorizer::new();

        let email = EmailForCategorization {
            id: Uuid::new_v4(),
            from: "orders@amazon.com".to_string(),
            to: vec!["user@example.com".to_string()],
            subject: "Your order confirmation #123-456".to_string(),
            body: "Thank you for your order. Your receipt is attached.".to_string(),
            headers: HashMap::new(),
        };

        let categories = categorizer.categorize(&email);

        assert!(categories.iter().any(|c| c.id == "receipt"));
    }

    #[test]
    fn test_newsletter_categorization() {
        let categorizer = EmailCategorizer::new();

        let mut headers = HashMap::new();
        headers.insert("list-unsubscribe".to_string(), "<mailto:unsub@news.com>".to_string());

        let email = EmailForCategorization {
            id: Uuid::new_v4(),
            from: "newsletter@techblog.com".to_string(),
            to: vec!["user@example.com".to_string()],
            subject: "Weekly Digest: Top Stories".to_string(),
            body: "Here are this week's top stories. To unsubscribe, click here.".to_string(),
            headers,
        };

        let categories = categorizer.categorize(&email);

        assert!(categories.iter().any(|c| c.id == "newsletter"));
    }
}
