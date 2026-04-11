// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Advanced Email Rules Engine for Mycelix Mail
//!
//! A powerful, flexible rules engine that allows users to create complex
//! email filtering, forwarding, and automation rules.

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use regex::Regex;

/// A complete email rule with conditions and actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub user_id: Uuid,
    pub priority: i32,
    pub enabled: bool,
    pub stop_processing: bool, // If true, don't evaluate subsequent rules
    pub conditions: RuleConditionGroup,
    pub actions: Vec<RuleAction>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_matched: Option<DateTime<Utc>>,
    pub match_count: u64,
}

/// Logical grouping of conditions (AND/OR)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleConditionGroup {
    pub operator: LogicalOperator,
    pub conditions: Vec<RuleCondition>,
    pub nested_groups: Vec<RuleConditionGroup>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum LogicalOperator {
    And,
    Or,
}

/// Individual condition to match against an email
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleCondition {
    pub field: ConditionField,
    pub operator: ConditionOperator,
    pub value: ConditionValue,
    pub case_sensitive: bool,
}

/// Fields that can be matched against
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConditionField {
    From,
    To,
    Cc,
    Subject,
    Body,
    Headers(String), // Custom header name
    Attachments,
    AttachmentName,
    AttachmentSize,
    AttachmentType,
    Size,
    Date,
    TrustScore,
    IsRead,
    IsStarred,
    HasAttachments,
    RecipientCount,
    Folder,
    Label,
}

/// Comparison operators
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    Contains,
    NotContains,
    StartsWith,
    EndsWith,
    Matches, // Regex match
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
    In,      // Value is in list
    NotIn,   // Value is not in list
    Exists,
    NotExists,
}

/// Value types for conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConditionValue {
    String(String),
    Number(f64),
    Boolean(bool),
    List(Vec<String>),
    Regex(String),
}

/// Actions to perform when rule matches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleAction {
    pub action_type: ActionType,
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ActionType {
    // Organization
    MoveTo,
    CopyTo,
    AddLabel,
    RemoveLabel,
    Archive,
    Delete,
    MarkAsRead,
    MarkAsUnread,
    Star,
    Unstar,
    MarkAsSpam,
    NeverSpam,

    // Forwarding & Notifications
    Forward,
    ForwardAsAttachment,
    SendCopy,
    Notify,

    // Auto-responses
    Reply,
    ReplyWithTemplate,

    // Advanced
    SetPriority,
    SetCategory,
    AddNote,
    RunWebhook,
    ExecuteScript, // For advanced users
}

/// Email data for rule evaluation
#[derive(Debug, Clone)]
pub struct EmailContext {
    pub id: Uuid,
    pub from: String,
    pub to: Vec<String>,
    pub cc: Vec<String>,
    pub subject: String,
    pub body_text: String,
    pub body_html: Option<String>,
    pub headers: HashMap<String, String>,
    pub attachments: Vec<AttachmentInfo>,
    pub size_bytes: usize,
    pub received_at: DateTime<Utc>,
    pub trust_score: f64,
    pub is_read: bool,
    pub is_starred: bool,
    pub folder: String,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AttachmentInfo {
    pub name: String,
    pub mime_type: String,
    pub size_bytes: usize,
}

/// Result of rule evaluation
#[derive(Debug, Clone)]
pub struct RuleEvaluationResult {
    pub rule_id: Uuid,
    pub matched: bool,
    pub actions_to_execute: Vec<RuleAction>,
    pub stop_processing: bool,
}

/// The main rules engine
pub struct RulesEngine {
    compiled_regexes: HashMap<String, Regex>,
}

impl RulesEngine {
    pub fn new() -> Self {
        Self {
            compiled_regexes: HashMap::new(),
        }
    }

    /// Evaluate all rules against an email
    pub fn evaluate_rules(
        &mut self,
        email: &EmailContext,
        rules: &[Rule],
    ) -> Vec<RuleEvaluationResult> {
        let mut results = Vec::new();
        let mut sorted_rules: Vec<&Rule> = rules.iter().filter(|r| r.enabled).collect();
        sorted_rules.sort_by_key(|r| r.priority);

        for rule in sorted_rules {
            let matched = self.evaluate_condition_group(&rule.conditions, email);

            let result = RuleEvaluationResult {
                rule_id: rule.id,
                matched,
                actions_to_execute: if matched { rule.actions.clone() } else { Vec::new() },
                stop_processing: matched && rule.stop_processing,
            };

            results.push(result.clone());

            if result.stop_processing {
                break;
            }
        }

        results
    }

    /// Evaluate a group of conditions
    fn evaluate_condition_group(&mut self, group: &RuleConditionGroup, email: &EmailContext) -> bool {
        let condition_results: Vec<bool> = group
            .conditions
            .iter()
            .map(|c| self.evaluate_condition(c, email))
            .collect();

        let nested_results: Vec<bool> = group
            .nested_groups
            .iter()
            .map(|g| self.evaluate_condition_group(g, email))
            .collect();

        let all_results: Vec<bool> = condition_results
            .into_iter()
            .chain(nested_results)
            .collect();

        if all_results.is_empty() {
            return true; // No conditions means always match
        }

        match group.operator {
            LogicalOperator::And => all_results.iter().all(|&r| r),
            LogicalOperator::Or => all_results.iter().any(|&r| r),
        }
    }

    /// Evaluate a single condition
    fn evaluate_condition(&mut self, condition: &RuleCondition, email: &EmailContext) -> bool {
        let field_value = self.get_field_value(&condition.field, email);

        match field_value {
            FieldValue::String(s) => self.evaluate_string_condition(
                &s,
                &condition.operator,
                &condition.value,
                condition.case_sensitive,
            ),
            FieldValue::Number(n) => self.evaluate_number_condition(
                n,
                &condition.operator,
                &condition.value,
            ),
            FieldValue::Boolean(b) => self.evaluate_boolean_condition(
                b,
                &condition.operator,
                &condition.value,
            ),
            FieldValue::StringList(list) => self.evaluate_list_condition(
                &list,
                &condition.operator,
                &condition.value,
                condition.case_sensitive,
            ),
            FieldValue::None => matches!(
                condition.operator,
                ConditionOperator::NotExists | ConditionOperator::Equals
            ) && matches!(condition.value, ConditionValue::String(ref s) if s.is_empty()),
        }
    }

    /// Extract field value from email
    fn get_field_value(&self, field: &ConditionField, email: &EmailContext) -> FieldValue {
        match field {
            ConditionField::From => FieldValue::String(email.from.clone()),
            ConditionField::To => FieldValue::StringList(email.to.clone()),
            ConditionField::Cc => FieldValue::StringList(email.cc.clone()),
            ConditionField::Subject => FieldValue::String(email.subject.clone()),
            ConditionField::Body => FieldValue::String(email.body_text.clone()),
            ConditionField::Headers(name) => {
                email.headers.get(name)
                    .map(|v| FieldValue::String(v.clone()))
                    .unwrap_or(FieldValue::None)
            }
            ConditionField::Attachments => {
                FieldValue::StringList(email.attachments.iter().map(|a| a.name.clone()).collect())
            }
            ConditionField::AttachmentName => {
                FieldValue::StringList(email.attachments.iter().map(|a| a.name.clone()).collect())
            }
            ConditionField::AttachmentSize => {
                let total: usize = email.attachments.iter().map(|a| a.size_bytes).sum();
                FieldValue::Number(total as f64)
            }
            ConditionField::AttachmentType => {
                FieldValue::StringList(email.attachments.iter().map(|a| a.mime_type.clone()).collect())
            }
            ConditionField::Size => FieldValue::Number(email.size_bytes as f64),
            ConditionField::Date => FieldValue::Number(email.received_at.timestamp() as f64),
            ConditionField::TrustScore => FieldValue::Number(email.trust_score),
            ConditionField::IsRead => FieldValue::Boolean(email.is_read),
            ConditionField::IsStarred => FieldValue::Boolean(email.is_starred),
            ConditionField::HasAttachments => FieldValue::Boolean(!email.attachments.is_empty()),
            ConditionField::RecipientCount => {
                FieldValue::Number((email.to.len() + email.cc.len()) as f64)
            }
            ConditionField::Folder => FieldValue::String(email.folder.clone()),
            ConditionField::Label => FieldValue::StringList(email.labels.clone()),
        }
    }

    fn evaluate_string_condition(
        &mut self,
        value: &str,
        operator: &ConditionOperator,
        condition_value: &ConditionValue,
        case_sensitive: bool,
    ) -> bool {
        let (compare_value, compare_target) = if case_sensitive {
            (value.to_string(), self.extract_string(condition_value))
        } else {
            (value.to_lowercase(), self.extract_string(condition_value).to_lowercase())
        };

        match operator {
            ConditionOperator::Equals => compare_value == compare_target,
            ConditionOperator::NotEquals => compare_value != compare_target,
            ConditionOperator::Contains => compare_value.contains(&compare_target),
            ConditionOperator::NotContains => !compare_value.contains(&compare_target),
            ConditionOperator::StartsWith => compare_value.starts_with(&compare_target),
            ConditionOperator::EndsWith => compare_value.ends_with(&compare_target),
            ConditionOperator::Matches => {
                if let ConditionValue::Regex(pattern) | ConditionValue::String(pattern) = condition_value {
                    self.regex_matches(&compare_value, pattern)
                } else {
                    false
                }
            }
            ConditionOperator::In => {
                if let ConditionValue::List(list) = condition_value {
                    list.iter().any(|item| {
                        if case_sensitive {
                            item == value
                        } else {
                            item.to_lowercase() == compare_value
                        }
                    })
                } else {
                    false
                }
            }
            ConditionOperator::NotIn => {
                if let ConditionValue::List(list) = condition_value {
                    !list.iter().any(|item| {
                        if case_sensitive {
                            item == value
                        } else {
                            item.to_lowercase() == compare_value
                        }
                    })
                } else {
                    true
                }
            }
            ConditionOperator::Exists => !value.is_empty(),
            ConditionOperator::NotExists => value.is_empty(),
            _ => false,
        }
    }

    fn evaluate_number_condition(
        &self,
        value: f64,
        operator: &ConditionOperator,
        condition_value: &ConditionValue,
    ) -> bool {
        let target = self.extract_number(condition_value);

        match operator {
            ConditionOperator::Equals => (value - target).abs() < f64::EPSILON,
            ConditionOperator::NotEquals => (value - target).abs() >= f64::EPSILON,
            ConditionOperator::GreaterThan => value > target,
            ConditionOperator::LessThan => value < target,
            ConditionOperator::GreaterOrEqual => value >= target,
            ConditionOperator::LessOrEqual => value <= target,
            _ => false,
        }
    }

    fn evaluate_boolean_condition(
        &self,
        value: bool,
        operator: &ConditionOperator,
        condition_value: &ConditionValue,
    ) -> bool {
        let target = self.extract_boolean(condition_value);

        match operator {
            ConditionOperator::Equals => value == target,
            ConditionOperator::NotEquals => value != target,
            _ => false,
        }
    }

    fn evaluate_list_condition(
        &mut self,
        values: &[String],
        operator: &ConditionOperator,
        condition_value: &ConditionValue,
        case_sensitive: bool,
    ) -> bool {
        let target = self.extract_string(condition_value);
        let compare_target = if case_sensitive { target.clone() } else { target.to_lowercase() };

        match operator {
            ConditionOperator::Contains => {
                values.iter().any(|v| {
                    let compare = if case_sensitive { v.clone() } else { v.to_lowercase() };
                    compare.contains(&compare_target)
                })
            }
            ConditionOperator::NotContains => {
                !values.iter().any(|v| {
                    let compare = if case_sensitive { v.clone() } else { v.to_lowercase() };
                    compare.contains(&compare_target)
                })
            }
            ConditionOperator::Equals => {
                values.iter().any(|v| {
                    let compare = if case_sensitive { v.clone() } else { v.to_lowercase() };
                    compare == compare_target
                })
            }
            ConditionOperator::Exists => !values.is_empty(),
            ConditionOperator::NotExists => values.is_empty(),
            _ => false,
        }
    }

    fn regex_matches(&mut self, value: &str, pattern: &str) -> bool {
        if let Some(regex) = self.compiled_regexes.get(pattern) {
            return regex.is_match(value);
        }

        match Regex::new(pattern) {
            Ok(regex) => {
                let matches = regex.is_match(value);
                self.compiled_regexes.insert(pattern.to_string(), regex);
                matches
            }
            Err(_) => false,
        }
    }

    fn extract_string(&self, value: &ConditionValue) -> String {
        match value {
            ConditionValue::String(s) => s.clone(),
            ConditionValue::Regex(s) => s.clone(),
            ConditionValue::Number(n) => n.to_string(),
            ConditionValue::Boolean(b) => b.to_string(),
            ConditionValue::List(l) => l.join(", "),
        }
    }

    fn extract_number(&self, value: &ConditionValue) -> f64 {
        match value {
            ConditionValue::Number(n) => *n,
            ConditionValue::String(s) => s.parse().unwrap_or(0.0),
            _ => 0.0,
        }
    }

    fn extract_boolean(&self, value: &ConditionValue) -> bool {
        match value {
            ConditionValue::Boolean(b) => *b,
            ConditionValue::String(s) => s == "true" || s == "1",
            ConditionValue::Number(n) => *n != 0.0,
            _ => false,
        }
    }
}

impl Default for RulesEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
enum FieldValue {
    String(String),
    Number(f64),
    Boolean(bool),
    StringList(Vec<String>),
    None,
}

/// Rule builder for easier rule construction
pub struct RuleBuilder {
    rule: Rule,
}

impl RuleBuilder {
    pub fn new(name: impl Into<String>, user_id: Uuid) -> Self {
        Self {
            rule: Rule {
                id: Uuid::new_v4(),
                name: name.into(),
                description: None,
                user_id,
                priority: 0,
                enabled: true,
                stop_processing: false,
                conditions: RuleConditionGroup {
                    operator: LogicalOperator::And,
                    conditions: Vec::new(),
                    nested_groups: Vec::new(),
                },
                actions: Vec::new(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                last_matched: None,
                match_count: 0,
            },
        }
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.rule.description = Some(desc.into());
        self
    }

    pub fn priority(mut self, priority: i32) -> Self {
        self.rule.priority = priority;
        self
    }

    pub fn stop_processing(mut self) -> Self {
        self.rule.stop_processing = true;
        self
    }

    pub fn when_from_contains(mut self, pattern: impl Into<String>) -> Self {
        self.rule.conditions.conditions.push(RuleCondition {
            field: ConditionField::From,
            operator: ConditionOperator::Contains,
            value: ConditionValue::String(pattern.into()),
            case_sensitive: false,
        });
        self
    }

    pub fn when_subject_contains(mut self, pattern: impl Into<String>) -> Self {
        self.rule.conditions.conditions.push(RuleCondition {
            field: ConditionField::Subject,
            operator: ConditionOperator::Contains,
            value: ConditionValue::String(pattern.into()),
            case_sensitive: false,
        });
        self
    }

    pub fn when_has_attachments(mut self) -> Self {
        self.rule.conditions.conditions.push(RuleCondition {
            field: ConditionField::HasAttachments,
            operator: ConditionOperator::Equals,
            value: ConditionValue::Boolean(true),
            case_sensitive: false,
        });
        self
    }

    pub fn when_trust_score_below(mut self, score: f64) -> Self {
        self.rule.conditions.conditions.push(RuleCondition {
            field: ConditionField::TrustScore,
            operator: ConditionOperator::LessThan,
            value: ConditionValue::Number(score),
            case_sensitive: false,
        });
        self
    }

    pub fn then_move_to(mut self, folder: impl Into<String>) -> Self {
        self.rule.actions.push(RuleAction {
            action_type: ActionType::MoveTo,
            parameters: HashMap::from([("folder".to_string(), folder.into())]),
        });
        self
    }

    pub fn then_add_label(mut self, label: impl Into<String>) -> Self {
        self.rule.actions.push(RuleAction {
            action_type: ActionType::AddLabel,
            parameters: HashMap::from([("label".to_string(), label.into())]),
        });
        self
    }

    pub fn then_mark_as_read(mut self) -> Self {
        self.rule.actions.push(RuleAction {
            action_type: ActionType::MarkAsRead,
            parameters: HashMap::new(),
        });
        self
    }

    pub fn then_forward_to(mut self, address: impl Into<String>) -> Self {
        self.rule.actions.push(RuleAction {
            action_type: ActionType::Forward,
            parameters: HashMap::from([("to".to_string(), address.into())]),
        });
        self
    }

    pub fn then_delete(mut self) -> Self {
        self.rule.actions.push(RuleAction {
            action_type: ActionType::Delete,
            parameters: HashMap::new(),
        });
        self
    }

    pub fn build(self) -> Rule {
        self.rule
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_email() -> EmailContext {
        EmailContext {
            id: Uuid::new_v4(),
            from: "newsletter@company.com".to_string(),
            to: vec!["user@example.com".to_string()],
            cc: vec![],
            subject: "Weekly Newsletter - Special Offer Inside!".to_string(),
            body_text: "Check out our special offers...".to_string(),
            body_html: None,
            headers: HashMap::new(),
            attachments: vec![],
            size_bytes: 5000,
            received_at: Utc::now(),
            trust_score: 0.7,
            is_read: false,
            is_starred: false,
            folder: "inbox".to_string(),
            labels: vec![],
        }
    }

    #[test]
    fn test_simple_rule_match() {
        let user_id = Uuid::new_v4();
        let rule = RuleBuilder::new("Newsletter Filter", user_id)
            .when_from_contains("newsletter")
            .when_subject_contains("newsletter")
            .then_move_to("newsletters")
            .then_mark_as_read()
            .build();

        let mut engine = RulesEngine::new();
        let email = create_test_email();

        let results = engine.evaluate_rules(&email, &[rule]);

        assert_eq!(results.len(), 1);
        assert!(results[0].matched);
        assert_eq!(results[0].actions_to_execute.len(), 2);
    }

    #[test]
    fn test_rule_builder() {
        let user_id = Uuid::new_v4();
        let rule = RuleBuilder::new("Low Trust Filter", user_id)
            .description("Move low trust emails to spam")
            .priority(10)
            .when_trust_score_below(0.3)
            .then_move_to("spam")
            .stop_processing()
            .build();

        assert_eq!(rule.name, "Low Trust Filter");
        assert!(rule.stop_processing);
        assert_eq!(rule.priority, 10);
    }
}
