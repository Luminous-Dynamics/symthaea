// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Natural Language Search for Mycelix Mail
//!
//! Allows users to search emails using natural language queries like:
//! "emails from John last week about the project proposal"
//! "unread messages with attachments from marketing team"

use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration, NaiveDate, Datelike, Weekday};
use serde::{Deserialize, Serialize};
use regex::Regex;

/// Parsed search query with structured filters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ParsedQuery {
    /// Keywords to match in subject/body
    pub keywords: Vec<String>,
    /// Sender email or name patterns
    pub from_patterns: Vec<String>,
    /// Recipient patterns
    pub to_patterns: Vec<String>,
    /// Date range filter
    pub date_range: Option<DateRange>,
    /// Folder/label filters
    pub folders: Vec<String>,
    /// Has attachments filter
    pub has_attachments: Option<bool>,
    /// Unread filter
    pub is_unread: Option<bool>,
    /// Starred/flagged filter
    pub is_starred: Option<bool>,
    /// Size filter (larger/smaller than)
    pub size_filter: Option<SizeFilter>,
    /// Trust score filter
    pub min_trust_score: Option<f64>,
    /// Category filter
    pub category: Option<String>,
    /// Sort order
    pub sort: SortOrder,
    /// Original query for reference
    pub original_query: String,
    /// Confidence in parse accuracy (0.0 - 1.0)
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    pub start: Option<DateTime<Utc>>,
    pub end: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SizeFilter {
    LargerThan(usize),   // bytes
    SmallerThan(usize),  // bytes
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum SortOrder {
    #[default]
    DateDesc,
    DateAsc,
    Relevance,
    SenderAsc,
    SenderDesc,
}

/// Natural language query parser
pub struct NLQueryParser {
    date_patterns: Vec<DatePattern>,
    person_indicators: Vec<&'static str>,
    attachment_indicators: Vec<&'static str>,
    status_indicators: HashMap<&'static str, StatusType>,
    folder_mappings: HashMap<&'static str, &'static str>,
    size_pattern: Regex,
}

#[derive(Debug, Clone)]
struct DatePattern {
    pattern: Regex,
    parser: DateParserFn,
}

type DateParserFn = fn(&str, &regex::Captures) -> Option<DateRange>;

#[derive(Debug, Clone, Copy)]
enum StatusType {
    Unread,
    Read,
    Starred,
    NotStarred,
}

impl NLQueryParser {
    pub fn new() -> Self {
        Self {
            date_patterns: Self::build_date_patterns(),
            person_indicators: vec![
                "from", "by", "sent by", "written by",
                "to", "sent to", "addressed to",
            ],
            attachment_indicators: vec![
                "with attachment", "with attachments", "has attachment",
                "with files", "with documents", "containing files",
            ],
            status_indicators: HashMap::from([
                ("unread", StatusType::Unread),
                ("not read", StatusType::Unread),
                ("new", StatusType::Unread),
                ("read", StatusType::Read),
                ("starred", StatusType::Starred),
                ("flagged", StatusType::Starred),
                ("important", StatusType::Starred),
                ("not starred", StatusType::NotStarred),
            ]),
            folder_mappings: HashMap::from([
                ("inbox", "inbox"),
                ("sent", "sent"),
                ("sent mail", "sent"),
                ("drafts", "drafts"),
                ("draft", "drafts"),
                ("trash", "trash"),
                ("deleted", "trash"),
                ("spam", "spam"),
                ("junk", "spam"),
                ("archive", "archive"),
                ("archived", "archive"),
            ]),
            size_pattern: Regex::new(r"(?i)(larger|bigger|greater|smaller|less) than (\d+)\s*(kb|mb|gb|bytes?)?").unwrap(),
        }
    }

    fn build_date_patterns() -> Vec<DatePattern> {
        vec![
            // "last week", "this week"
            DatePattern {
                pattern: Regex::new(r"(?i)\b(this|last|past)\s+week\b").unwrap(),
                parser: |_, caps| {
                    let now = Utc::now();
                    let modifier = caps.get(1).map(|m| m.as_str().to_lowercase()).unwrap_or_default();

                    let (start, end) = if modifier == "this" {
                        let days_since_monday = now.weekday().num_days_from_monday();
                        let start = now - Duration::days(days_since_monday as i64);
                        (start, now)
                    } else {
                        // last/past week
                        let days_since_monday = now.weekday().num_days_from_monday();
                        let this_monday = now - Duration::days(days_since_monday as i64);
                        let last_monday = this_monday - Duration::days(7);
                        (last_monday, this_monday)
                    };

                    Some(DateRange {
                        start: Some(start),
                        end: Some(end),
                    })
                },
            },
            // "last month", "this month"
            DatePattern {
                pattern: Regex::new(r"(?i)\b(this|last|past)\s+month\b").unwrap(),
                parser: |_, caps| {
                    let now = Utc::now();
                    let modifier = caps.get(1).map(|m| m.as_str().to_lowercase()).unwrap_or_default();

                    if modifier == "this" {
                        let start = now.with_day(1).unwrap();
                        Some(DateRange {
                            start: Some(start),
                            end: Some(now),
                        })
                    } else {
                        let first_of_this_month = now.with_day(1).unwrap();
                        let last_month_end = first_of_this_month - Duration::days(1);
                        let last_month_start = last_month_end.with_day(1).unwrap();
                        Some(DateRange {
                            start: Some(last_month_start),
                            end: Some(first_of_this_month),
                        })
                    }
                },
            },
            // "yesterday", "today"
            DatePattern {
                pattern: Regex::new(r"(?i)\b(today|yesterday)\b").unwrap(),
                parser: |_, caps| {
                    let now = Utc::now();
                    let day = caps.get(1).map(|m| m.as_str().to_lowercase()).unwrap_or_default();

                    let target = if day == "today" {
                        now
                    } else {
                        now - Duration::days(1)
                    };

                    let start = target.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc();
                    let end = target.date_naive().and_hms_opt(23, 59, 59).unwrap().and_utc();

                    Some(DateRange {
                        start: Some(start),
                        end: Some(end),
                    })
                },
            },
            // "last N days/weeks/months"
            DatePattern {
                pattern: Regex::new(r"(?i)\blast\s+(\d+)\s+(day|week|month)s?\b").unwrap(),
                parser: |_, caps| {
                    let now = Utc::now();
                    let count: i64 = caps.get(1)?.as_str().parse().ok()?;
                    let unit = caps.get(2)?.as_str().to_lowercase();

                    let duration = match unit.as_str() {
                        "day" => Duration::days(count),
                        "week" => Duration::weeks(count),
                        "month" => Duration::days(count * 30),
                        _ => return None,
                    };

                    Some(DateRange {
                        start: Some(now - duration),
                        end: Some(now),
                    })
                },
            },
            // "in January", "in March 2024"
            DatePattern {
                pattern: Regex::new(r"(?i)\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?\b").unwrap(),
                parser: |_, caps| {
                    let month_name = caps.get(1)?.as_str().to_lowercase();
                    let year = caps.get(2)
                        .and_then(|m| m.as_str().parse().ok())
                        .unwrap_or_else(|| Utc::now().year());

                    let month = match month_name.as_str() {
                        "january" => 1, "february" => 2, "march" => 3, "april" => 4,
                        "may" => 5, "june" => 6, "july" => 7, "august" => 8,
                        "september" => 9, "october" => 10, "november" => 11, "december" => 12,
                        _ => return None,
                    };

                    let start = NaiveDate::from_ymd_opt(year, month, 1)?
                        .and_hms_opt(0, 0, 0)?
                        .and_utc();

                    let next_month = if month == 12 { 1 } else { month + 1 };
                    let next_year = if month == 12 { year + 1 } else { year };
                    let end = NaiveDate::from_ymd_opt(next_year, next_month, 1)?
                        .and_hms_opt(0, 0, 0)?
                        .and_utc();

                    Some(DateRange {
                        start: Some(start),
                        end: Some(end),
                    })
                },
            },
            // "before/after date"
            DatePattern {
                pattern: Regex::new(r"(?i)\b(before|after|since)\s+(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b").unwrap(),
                parser: |_, caps| {
                    let modifier = caps.get(1)?.as_str().to_lowercase();
                    let month: u32 = caps.get(2)?.as_str().parse().ok()?;
                    let day: u32 = caps.get(3)?.as_str().parse().ok()?;
                    let mut year: i32 = caps.get(4)?.as_str().parse().ok()?;

                    if year < 100 {
                        year += 2000;
                    }

                    let date = NaiveDate::from_ymd_opt(year, month, day)?
                        .and_hms_opt(0, 0, 0)?
                        .and_utc();

                    match modifier.as_str() {
                        "before" => Some(DateRange {
                            start: None,
                            end: Some(date),
                        }),
                        "after" | "since" => Some(DateRange {
                            start: Some(date),
                            end: None,
                        }),
                        _ => None,
                    }
                },
            },
        ]
    }

    /// Parse a natural language query into structured filters
    pub fn parse(&self, query: &str) -> ParsedQuery {
        let mut result = ParsedQuery {
            original_query: query.to_string(),
            confidence: 0.7, // Base confidence
            ..Default::default()
        };

        let mut remaining_query = query.to_string();

        // Extract date range
        for date_pattern in &self.date_patterns {
            if let Some(caps) = date_pattern.pattern.captures(&remaining_query) {
                if let Some(range) = (date_pattern.parser)(query, &caps) {
                    result.date_range = Some(range);
                    remaining_query = date_pattern.pattern.replace(&remaining_query, "").to_string();
                    result.confidence += 0.05;
                    break;
                }
            }
        }

        // Extract "from" patterns
        let from_regex = Regex::new(r"(?i)\b(?:from|by|sent by)\s+([a-zA-Z0-9._@-]+(?:\s+[a-zA-Z]+)?)\b").unwrap();
        for caps in from_regex.captures_iter(&remaining_query.clone()) {
            if let Some(person) = caps.get(1) {
                result.from_patterns.push(person.as_str().trim().to_string());
                result.confidence += 0.05;
            }
        }
        remaining_query = from_regex.replace_all(&remaining_query, "").to_string();

        // Extract "to" patterns
        let to_regex = Regex::new(r"(?i)\b(?:to|sent to)\s+([a-zA-Z0-9._@-]+(?:\s+[a-zA-Z]+)?)\b").unwrap();
        for caps in to_regex.captures_iter(&remaining_query.clone()) {
            if let Some(person) = caps.get(1) {
                result.to_patterns.push(person.as_str().trim().to_string());
                result.confidence += 0.05;
            }
        }
        remaining_query = to_regex.replace_all(&remaining_query, "").to_string();

        // Check for attachment indicators
        for indicator in &self.attachment_indicators {
            if remaining_query.to_lowercase().contains(indicator) {
                result.has_attachments = Some(true);
                remaining_query = remaining_query.replace(indicator, "");
                result.confidence += 0.05;
            }
        }

        // Check without attachments
        if remaining_query.to_lowercase().contains("without attachment") ||
           remaining_query.to_lowercase().contains("no attachment") {
            result.has_attachments = Some(false);
            remaining_query = Regex::new(r"(?i)(without|no)\s+attachments?")
                .unwrap()
                .replace_all(&remaining_query, "")
                .to_string();
        }

        // Check for status indicators
        for (indicator, status) in &self.status_indicators {
            if remaining_query.to_lowercase().contains(indicator) {
                match status {
                    StatusType::Unread => result.is_unread = Some(true),
                    StatusType::Read => result.is_unread = Some(false),
                    StatusType::Starred => result.is_starred = Some(true),
                    StatusType::NotStarred => result.is_starred = Some(false),
                }
                remaining_query = remaining_query.replace(indicator, "");
                result.confidence += 0.03;
            }
        }

        // Check for folder mentions
        for (mention, folder) in &self.folder_mappings {
            if remaining_query.to_lowercase().contains(&format!("in {}", mention)) ||
               remaining_query.to_lowercase().contains(&format!("from {}", mention)) {
                result.folders.push((*folder).to_string());
                let folder_regex = Regex::new(&format!(r"(?i)\b(in|from)\s+{}\b", mention)).unwrap();
                remaining_query = folder_regex.replace_all(&remaining_query, "").to_string();
                result.confidence += 0.03;
            }
        }

        // Check for size filters
        if let Some(caps) = self.size_pattern.captures(&remaining_query) {
            if let (Some(comparison), Some(size_str)) = (caps.get(1), caps.get(2)) {
                if let Ok(size) = size_str.as_str().parse::<usize>() {
                    let unit = caps.get(3).map(|m| m.as_str().to_lowercase()).unwrap_or_default();
                    let multiplier = match unit.as_str() {
                        "kb" => 1024,
                        "mb" => 1024 * 1024,
                        "gb" => 1024 * 1024 * 1024,
                        _ => 1,
                    };
                    let size_bytes = size * multiplier;

                    let comp = comparison.as_str().to_lowercase();
                    if comp == "larger" || comp == "bigger" || comp == "greater" {
                        result.size_filter = Some(SizeFilter::LargerThan(size_bytes));
                    } else {
                        result.size_filter = Some(SizeFilter::SmallerThan(size_bytes));
                    }
                }
            }
            remaining_query = self.size_pattern.replace_all(&remaining_query, "").to_string();
        }

        // Check for category mentions
        let categories = ["primary", "social", "promotions", "updates", "forums", "work", "personal"];
        for category in categories {
            if remaining_query.to_lowercase().contains(category) {
                result.category = Some(category.to_string());
                remaining_query = Regex::new(&format!(r"(?i)\b{}\b", category))
                    .unwrap()
                    .replace_all(&remaining_query, "")
                    .to_string();
            }
        }

        // Check for sort preferences
        if remaining_query.to_lowercase().contains("oldest first") ||
           remaining_query.to_lowercase().contains("oldest") {
            result.sort = SortOrder::DateAsc;
            remaining_query = Regex::new(r"(?i)oldest\s*(first)?")
                .unwrap()
                .replace_all(&remaining_query, "")
                .to_string();
        } else if remaining_query.to_lowercase().contains("most relevant") {
            result.sort = SortOrder::Relevance;
            remaining_query = remaining_query.replace("most relevant", "");
        }

        // Check for trust score mentions
        let trust_regex = Regex::new(r"(?i)\b(?:trusted|high trust|trust score)\s*(?:above|over|greater than)?\s*(\d+(?:\.\d+)?)?").unwrap();
        if let Some(caps) = trust_regex.captures(&remaining_query) {
            let score = caps.get(1)
                .and_then(|m| m.as_str().parse::<f64>().ok())
                .unwrap_or(0.7);
            result.min_trust_score = Some(score);
            remaining_query = trust_regex.replace_all(&remaining_query, "").to_string();
        }

        // Extract remaining keywords (clean up filler words)
        let filler_words = ["the", "a", "an", "about", "regarding", "concerning", "with", "and", "or", "emails", "email", "messages", "message"];
        let cleaned = remaining_query
            .split_whitespace()
            .filter(|word| {
                let lower = word.to_lowercase();
                !filler_words.contains(&lower.as_str()) && word.len() > 1
            })
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        result.keywords = cleaned;

        // Adjust confidence based on parse completeness
        if result.keywords.is_empty() && result.from_patterns.is_empty() && result.date_range.is_none() {
            result.confidence = 0.3;
        }

        result.confidence = result.confidence.min(1.0);

        result
    }

    /// Generate a human-readable explanation of the parsed query
    pub fn explain(&self, parsed: &ParsedQuery) -> String {
        let mut parts = Vec::new();

        if !parsed.keywords.is_empty() {
            parts.push(format!("containing \"{}\"", parsed.keywords.join(" ")));
        }

        if !parsed.from_patterns.is_empty() {
            parts.push(format!("from {}", parsed.from_patterns.join(" or ")));
        }

        if !parsed.to_patterns.is_empty() {
            parts.push(format!("to {}", parsed.to_patterns.join(" or ")));
        }

        if let Some(ref range) = parsed.date_range {
            match (&range.start, &range.end) {
                (Some(start), Some(end)) => {
                    parts.push(format!("between {} and {}",
                        start.format("%b %d, %Y"),
                        end.format("%b %d, %Y")));
                }
                (Some(start), None) => {
                    parts.push(format!("after {}", start.format("%b %d, %Y")));
                }
                (None, Some(end)) => {
                    parts.push(format!("before {}", end.format("%b %d, %Y")));
                }
                _ => {}
            }
        }

        if let Some(true) = parsed.has_attachments {
            parts.push("with attachments".to_string());
        } else if let Some(false) = parsed.has_attachments {
            parts.push("without attachments".to_string());
        }

        if let Some(true) = parsed.is_unread {
            parts.push("unread".to_string());
        }

        if let Some(true) = parsed.is_starred {
            parts.push("starred".to_string());
        }

        if !parsed.folders.is_empty() {
            parts.push(format!("in {}", parsed.folders.join(", ")));
        }

        if let Some(ref category) = parsed.category {
            parts.push(format!("categorized as {}", category));
        }

        if parts.is_empty() {
            "all emails".to_string()
        } else {
            format!("Searching for emails {}", parts.join(", "))
        }
    }
}

impl Default for NLQueryParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Search service that combines NL parsing with actual search execution
pub struct NLSearchService {
    parser: NLQueryParser,
}

impl NLSearchService {
    pub fn new() -> Self {
        Self {
            parser: NLQueryParser::new(),
        }
    }

    /// Parse and prepare a search from natural language
    pub fn prepare_search(&self, query: &str) -> SearchPrepResult {
        let parsed = self.parser.parse(query);
        let explanation = self.parser.explain(&parsed);

        SearchPrepResult {
            parsed_query: parsed,
            explanation,
        }
    }

    /// Convert parsed query to database query parameters
    pub fn to_query_params(&self, parsed: &ParsedQuery) -> SearchQueryParams {
        SearchQueryParams {
            full_text_terms: if parsed.keywords.is_empty() {
                None
            } else {
                Some(parsed.keywords.join(" "))
            },
            sender_patterns: if parsed.from_patterns.is_empty() {
                None
            } else {
                Some(parsed.from_patterns.clone())
            },
            recipient_patterns: if parsed.to_patterns.is_empty() {
                None
            } else {
                Some(parsed.to_patterns.clone())
            },
            date_from: parsed.date_range.as_ref().and_then(|r| r.start),
            date_to: parsed.date_range.as_ref().and_then(|r| r.end),
            folders: if parsed.folders.is_empty() {
                None
            } else {
                Some(parsed.folders.clone())
            },
            has_attachments: parsed.has_attachments,
            is_unread: parsed.is_unread,
            is_starred: parsed.is_starred,
            min_size: match &parsed.size_filter {
                Some(SizeFilter::LargerThan(s)) => Some(*s),
                _ => None,
            },
            max_size: match &parsed.size_filter {
                Some(SizeFilter::SmallerThan(s)) => Some(*s),
                _ => None,
            },
            category: parsed.category.clone(),
            min_trust_score: parsed.min_trust_score,
            sort_by: match parsed.sort {
                SortOrder::DateDesc => "date_desc",
                SortOrder::DateAsc => "date_asc",
                SortOrder::Relevance => "relevance",
                SortOrder::SenderAsc => "sender_asc",
                SortOrder::SenderDesc => "sender_desc",
            }.to_string(),
        }
    }
}

impl Default for NLSearchService {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of search preparation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPrepResult {
    pub parsed_query: ParsedQuery,
    pub explanation: String,
}

/// Database query parameters generated from parsed query
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchQueryParams {
    pub full_text_terms: Option<String>,
    pub sender_patterns: Option<Vec<String>>,
    pub recipient_patterns: Option<Vec<String>>,
    pub date_from: Option<DateTime<Utc>>,
    pub date_to: Option<DateTime<Utc>>,
    pub folders: Option<Vec<String>>,
    pub has_attachments: Option<bool>,
    pub is_unread: Option<bool>,
    pub is_starred: Option<bool>,
    pub min_size: Option<usize>,
    pub max_size: Option<usize>,
    pub category: Option<String>,
    pub min_trust_score: Option<f64>,
    pub sort_by: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_query() {
        let parser = NLQueryParser::new();
        let result = parser.parse("emails from John about project proposal");

        assert!(!result.from_patterns.is_empty());
        assert!(result.from_patterns[0].to_lowercase().contains("john"));
        assert!(result.keywords.iter().any(|k| k.to_lowercase() == "project" || k.to_lowercase() == "proposal"));
    }

    #[test]
    fn test_date_query() {
        let parser = NLQueryParser::new();
        let result = parser.parse("unread emails from last week");

        assert!(result.date_range.is_some());
        assert_eq!(result.is_unread, Some(true));
    }

    #[test]
    fn test_attachment_query() {
        let parser = NLQueryParser::new();
        let result = parser.parse("messages with attachments from marketing");

        assert_eq!(result.has_attachments, Some(true));
        assert!(!result.from_patterns.is_empty());
    }

    #[test]
    fn test_folder_query() {
        let parser = NLQueryParser::new();
        let result = parser.parse("emails in sent folder");

        assert!(result.folders.contains(&"sent".to_string()));
    }

    #[test]
    fn test_complex_query() {
        let parser = NLQueryParser::new();
        let result = parser.parse("starred unread emails from John last month with attachments in inbox");

        assert_eq!(result.is_starred, Some(true));
        assert_eq!(result.is_unread, Some(true));
        assert!(!result.from_patterns.is_empty());
        assert!(result.date_range.is_some());
        assert_eq!(result.has_attachments, Some(true));
        assert!(result.folders.contains(&"inbox".to_string()));
    }

    #[test]
    fn test_explain() {
        let parser = NLQueryParser::new();
        let parsed = parser.parse("emails from Alice last week");
        let explanation = parser.explain(&parsed);

        assert!(explanation.contains("Alice") || explanation.contains("alice"));
        assert!(explanation.contains("between") || explanation.contains("week"));
    }
}
