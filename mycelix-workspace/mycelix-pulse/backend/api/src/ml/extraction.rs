// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Meeting & Action Item Extraction
//!
//! Extract structured data from email content

use chrono::{DateTime, NaiveDate, NaiveTime, Utc, Weekday};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use uuid::Uuid;

/// Extracted meeting information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedMeeting {
    pub id: Uuid,
    pub title: Option<String>,
    pub date: Option<NaiveDate>,
    pub time: Option<NaiveTime>,
    pub duration_minutes: Option<i32>,
    pub location: Option<MeetingLocation>,
    pub attendees: Vec<String>,
    pub agenda: Vec<String>,
    pub confidence: f64,
    pub source_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeetingLocation {
    Physical(String),
    Virtual { platform: String, link: Option<String> },
    Phone(String),
    TBD,
}

/// Extracted action item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedAction {
    pub id: Uuid,
    pub description: String,
    pub assignee: Option<String>,
    pub due_date: Option<NaiveDate>,
    pub priority: ActionPriority,
    pub context: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ActionPriority {
    Low,
    Medium,
    High,
    Urgent,
}

/// Extracted contact information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedContact {
    pub name: Option<String>,
    pub email: Option<String>,
    pub phone: Option<String>,
    pub title: Option<String>,
    pub company: Option<String>,
}

/// Extracted deadline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedDeadline {
    pub description: String,
    pub date: NaiveDate,
    pub is_hard_deadline: bool,
    pub source_text: String,
}

/// Content extractor
pub struct ContentExtractor {
    // Compiled regex patterns
    email_regex: Regex,
    phone_regex: Regex,
    url_regex: Regex,
    date_regex: Regex,
    time_regex: Regex,
    meeting_link_regex: Regex,
}

impl ContentExtractor {
    pub fn new() -> Self {
        Self {
            email_regex: Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap(),
            phone_regex: Regex::new(r"(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}").unwrap(),
            url_regex: Regex::new(r"https?://[^\s<>\[\]]+").unwrap(),
            date_regex: Regex::new(r"(?i)(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|\d{1,2}/\d{1,2}(?:/\d{2,4})?").unwrap(),
            time_regex: Regex::new(r"(?i)\d{1,2}(?::\d{2})?\s*(?:am|pm)|(?:[01]?\d|2[0-3]):[0-5]\d").unwrap(),
            meeting_link_regex: Regex::new(r"(?i)(zoom\.us|meet\.google\.com|teams\.microsoft\.com|webex\.com)/[^\s<>]+").unwrap(),
        }
    }

    /// Extract all structured data from email
    pub fn extract_all(&self, subject: &str, body: &str) -> ExtractionResult {
        let text = format!("{}\n{}", subject, body);

        ExtractionResult {
            meetings: self.extract_meetings(&text),
            actions: self.extract_actions(&text),
            contacts: self.extract_contacts(&text),
            deadlines: self.extract_deadlines(&text),
            links: self.extract_links(&text),
            mentions: self.extract_mentions(&text),
        }
    }

    /// Extract meeting information
    pub fn extract_meetings(&self, text: &str) -> Vec<ExtractedMeeting> {
        let mut meetings = Vec::new();
        let text_lower = text.to_lowercase();

        // Meeting indicators
        let meeting_keywords = [
            "meeting", "call", "sync", "standup", "stand-up", "check-in",
            "discussion", "session", "workshop", "presentation", "demo",
            "interview", "conference", "webinar",
        ];

        // Check if this looks like a meeting email
        let has_meeting_keyword = meeting_keywords.iter().any(|kw| text_lower.contains(kw));
        if !has_meeting_keyword {
            return meetings;
        }

        // Extract date
        let date = self.extract_date(text);

        // Extract time
        let time = self.extract_time(text);

        // Extract location
        let location = self.extract_meeting_location(text);

        // Extract attendees
        let attendees = self.extract_attendees(text);

        // Extract title (usually from subject or first line)
        let title = self.extract_meeting_title(text, &meeting_keywords);

        // Calculate confidence
        let confidence = self.calculate_meeting_confidence(
            date.is_some(),
            time.is_some(),
            location.is_some(),
            !attendees.is_empty(),
        );

        if confidence > 0.3 {
            meetings.push(ExtractedMeeting {
                id: Uuid::new_v4(),
                title,
                date,
                time,
                duration_minutes: self.extract_duration(text),
                location,
                attendees,
                agenda: self.extract_agenda(text),
                confidence,
                source_text: text.chars().take(500).collect(),
            });
        }

        meetings
    }

    /// Extract action items
    pub fn extract_actions(&self, text: &str) -> Vec<ExtractedAction> {
        let mut actions = Vec::new();

        // Action indicators
        let action_patterns = [
            (r"(?i)please\s+(.+?)(?:\.|$)", ActionPriority::Medium),
            (r"(?i)can you\s+(.+?)(?:\.|$|\?)", ActionPriority::Medium),
            (r"(?i)could you\s+(.+?)(?:\.|$|\?)", ActionPriority::Low),
            (r"(?i)need(?:s)? to\s+(.+?)(?:\.|$)", ActionPriority::High),
            (r"(?i)must\s+(.+?)(?:\.|$)", ActionPriority::High),
            (r"(?i)action(?:\s+item)?:\s*(.+?)(?:\.|$)", ActionPriority::High),
            (r"(?i)todo:\s*(.+?)(?:\.|$)", ActionPriority::Medium),
            (r"(?i)(?:@\w+)\s+(.+?)(?:\.|$)", ActionPriority::Medium),
            (r"(?i)asap[:\s]+(.+?)(?:\.|$)", ActionPriority::Urgent),
            (r"(?i)urgent(?:ly)?[:\s]+(.+?)(?:\.|$)", ActionPriority::Urgent),
        ];

        for (pattern, priority) in action_patterns {
            if let Ok(re) = Regex::new(pattern) {
                for cap in re.captures_iter(text) {
                    if let Some(action_text) = cap.get(1) {
                        let description = action_text.as_str().trim().to_string();

                        // Skip very short or very long matches
                        if description.len() < 10 || description.len() > 200 {
                            continue;
                        }

                        // Extract assignee from @mention
                        let assignee = self.extract_assignee_from_text(&description);

                        // Extract due date
                        let due_date = self.extract_date(&description);

                        actions.push(ExtractedAction {
                            id: Uuid::new_v4(),
                            description: description.clone(),
                            assignee,
                            due_date,
                            priority,
                            context: text.chars().take(100).collect(),
                            confidence: 0.7,
                        });
                    }
                }
            }
        }

        // Deduplicate similar actions
        actions.dedup_by(|a, b| {
            similarity(&a.description, &b.description) > 0.8
        });

        actions
    }

    /// Extract contact information
    pub fn extract_contacts(&self, text: &str) -> Vec<ExtractedContact> {
        let mut contacts = Vec::new();

        // Extract emails
        let emails: Vec<String> = self.email_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect();

        // Extract phones
        let phones: Vec<String> = self.phone_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect();

        // Create contacts from emails
        for email in emails {
            let name = self.extract_name_near_email(&email, text);

            contacts.push(ExtractedContact {
                name,
                email: Some(email),
                phone: None,
                title: None,
                company: None,
            });
        }

        // Match phones to contacts or create new ones
        for phone in phones {
            let existing = contacts.iter_mut().find(|c| {
                c.name.as_ref().map(|n| text.contains(n)).unwrap_or(false)
            });

            if let Some(contact) = existing {
                contact.phone = Some(phone);
            } else {
                contacts.push(ExtractedContact {
                    name: None,
                    email: None,
                    phone: Some(phone),
                    title: None,
                    company: None,
                });
            }
        }

        contacts
    }

    /// Extract deadlines
    pub fn extract_deadlines(&self, text: &str) -> Vec<ExtractedDeadline> {
        let mut deadlines = Vec::new();
        let text_lower = text.to_lowercase();

        let deadline_patterns = [
            (r"(?i)deadline[:\s]+(.+?)(?:\.|$)", true),
            (r"(?i)due by[:\s]+(.+?)(?:\.|$)", true),
            (r"(?i)due[:\s]+(.+?)(?:\.|$)", true),
            (r"(?i)by\s+((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d+)", false),
            (r"(?i)before[:\s]+(.+?)(?:\.|$)", false),
            (r"(?i)no later than[:\s]+(.+?)(?:\.|$)", true),
        ];

        for (pattern, is_hard) in deadline_patterns {
            if let Ok(re) = Regex::new(pattern) {
                for cap in re.captures_iter(text) {
                    if let Some(deadline_text) = cap.get(1) {
                        if let Some(date) = self.extract_date(deadline_text.as_str()) {
                            deadlines.push(ExtractedDeadline {
                                description: deadline_text.as_str().trim().to_string(),
                                date,
                                is_hard_deadline: is_hard,
                                source_text: cap.get(0).unwrap().as_str().to_string(),
                            });
                        }
                    }
                }
            }
        }

        deadlines
    }

    /// Extract links
    pub fn extract_links(&self, text: &str) -> Vec<String> {
        self.url_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect()
    }

    /// Extract @mentions
    pub fn extract_mentions(&self, text: &str) -> Vec<String> {
        let mention_regex = Regex::new(r"@([a-zA-Z0-9_]+)").unwrap();

        mention_regex
            .captures_iter(text)
            .filter_map(|cap| cap.get(1))
            .map(|m| m.as_str().to_string())
            .collect()
    }

    // Helper methods

    fn extract_date(&self, text: &str) -> Option<NaiveDate> {
        let text_lower = text.to_lowercase();
        let today = Utc::now().date_naive();

        // Relative dates
        if text_lower.contains("today") {
            return Some(today);
        }
        if text_lower.contains("tomorrow") {
            return Some(today + chrono::Duration::days(1));
        }
        if text_lower.contains("next week") {
            return Some(today + chrono::Duration::weeks(1));
        }

        // Day names
        let days = [
            ("monday", Weekday::Mon),
            ("tuesday", Weekday::Tue),
            ("wednesday", Weekday::Wed),
            ("thursday", Weekday::Thu),
            ("friday", Weekday::Fri),
            ("saturday", Weekday::Sat),
            ("sunday", Weekday::Sun),
        ];

        for (name, weekday) in days {
            if text_lower.contains(name) {
                let days_ahead = (weekday.num_days_from_monday() as i64
                    - today.weekday().num_days_from_monday() as i64 + 7) % 7;
                let days_ahead = if days_ahead == 0 { 7 } else { days_ahead };
                return Some(today + chrono::Duration::days(days_ahead));
            }
        }

        // Parse explicit dates
        if let Some(date_match) = self.date_regex.find(text) {
            let date_str = date_match.as_str();
            // Try various formats
            if let Ok(date) = NaiveDate::parse_from_str(date_str, "%B %d, %Y") {
                return Some(date);
            }
            if let Ok(date) = NaiveDate::parse_from_str(date_str, "%B %d %Y") {
                return Some(date);
            }
            if let Ok(date) = NaiveDate::parse_from_str(date_str, "%m/%d/%Y") {
                return Some(date);
            }
            if let Ok(date) = NaiveDate::parse_from_str(date_str, "%m/%d/%y") {
                return Some(date);
            }
        }

        None
    }

    fn extract_time(&self, text: &str) -> Option<NaiveTime> {
        if let Some(time_match) = self.time_regex.find(text) {
            let time_str = time_match.as_str().to_uppercase();

            // Try parsing with AM/PM
            if time_str.contains("AM") || time_str.contains("PM") {
                if let Ok(time) = NaiveTime::parse_from_str(&time_str, "%I:%M %p") {
                    return Some(time);
                }
                if let Ok(time) = NaiveTime::parse_from_str(&time_str, "%I %p") {
                    return Some(time);
                }
            }

            // Try 24-hour format
            if let Ok(time) = NaiveTime::parse_from_str(time_match.as_str(), "%H:%M") {
                return Some(time);
            }
        }

        None
    }

    fn extract_meeting_location(&self, text: &str) -> Option<MeetingLocation> {
        let text_lower = text.to_lowercase();

        // Check for virtual meeting links
        if let Some(link_match) = self.meeting_link_regex.find(text) {
            let link = link_match.as_str();
            let platform = if link.contains("zoom") {
                "Zoom"
            } else if link.contains("meet.google") {
                "Google Meet"
            } else if link.contains("teams") {
                "Microsoft Teams"
            } else if link.contains("webex") {
                "Webex"
            } else {
                "Video Call"
            };

            return Some(MeetingLocation::Virtual {
                platform: platform.to_string(),
                link: Some(format!("https://{}", link)),
            });
        }

        // Check for phone
        if text_lower.contains("dial") || text_lower.contains("call in") {
            if let Some(phone) = self.phone_regex.find(text) {
                return Some(MeetingLocation::Phone(phone.as_str().to_string()));
            }
        }

        // Check for physical location patterns
        let location_patterns = [
            r"(?i)(?:room|conference room|meeting room)\s+([A-Z0-9-]+)",
            r"(?i)(?:at|location:)\s+([^.]+)",
        ];

        for pattern in location_patterns {
            if let Ok(re) = Regex::new(pattern) {
                if let Some(cap) = re.captures(text) {
                    if let Some(loc) = cap.get(1) {
                        return Some(MeetingLocation::Physical(loc.as_str().trim().to_string()));
                    }
                }
            }
        }

        None
    }

    fn extract_attendees(&self, text: &str) -> Vec<String> {
        let mut attendees: HashSet<String> = HashSet::new();

        // Extract emails as attendees
        for email_match in self.email_regex.find_iter(text) {
            attendees.insert(email_match.as_str().to_string());
        }

        // Extract @mentions as attendees
        let mention_regex = Regex::new(r"@([a-zA-Z0-9_]+)").unwrap();
        for cap in mention_regex.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                attendees.insert(m.as_str().to_string());
            }
        }

        attendees.into_iter().collect()
    }

    fn extract_meeting_title(&self, text: &str, keywords: &[&str]) -> Option<String> {
        let lines: Vec<&str> = text.lines().collect();

        // First line often contains the title
        if let Some(first_line) = lines.first() {
            if first_line.len() < 100 && keywords.iter().any(|kw| first_line.to_lowercase().contains(kw)) {
                return Some(first_line.trim().to_string());
            }
        }

        // Look for "Re:" stripped subject
        for line in &lines {
            let line = line.trim();
            if line.starts_with("Subject:") || line.starts_with("Re:") {
                let title = line
                    .trim_start_matches("Subject:")
                    .trim_start_matches("Re:")
                    .trim();
                if !title.is_empty() {
                    return Some(title.to_string());
                }
            }
        }

        None
    }

    fn extract_duration(&self, text: &str) -> Option<i32> {
        let duration_regex = Regex::new(r"(?i)(\d+)\s*(?:hour|hr|h)s?|(\d+)\s*(?:minute|min|m)s?").unwrap();

        if let Some(cap) = duration_regex.captures(text) {
            if let Some(hours) = cap.get(1) {
                return Some(hours.as_str().parse::<i32>().unwrap_or(1) * 60);
            }
            if let Some(mins) = cap.get(2) {
                return Some(mins.as_str().parse().unwrap_or(30));
            }
        }

        // Default meeting duration
        if text.to_lowercase().contains("quick") {
            return Some(15);
        }

        None
    }

    fn extract_agenda(&self, text: &str) -> Vec<String> {
        let mut agenda = Vec::new();

        // Look for numbered or bulleted items
        let list_regex = Regex::new(r"(?m)^[\s]*(?:\d+[.\):]|\*|-|•)\s*(.+)$").unwrap();

        for cap in list_regex.captures_iter(text) {
            if let Some(item) = cap.get(1) {
                let item_text = item.as_str().trim();
                if item_text.len() > 5 && item_text.len() < 200 {
                    agenda.push(item_text.to_string());
                }
            }
        }

        agenda
    }

    fn calculate_meeting_confidence(
        &self,
        has_date: bool,
        has_time: bool,
        has_location: bool,
        has_attendees: bool,
    ) -> f64 {
        let mut score = 0.2; // Base score for having meeting keywords

        if has_date { score += 0.25; }
        if has_time { score += 0.25; }
        if has_location { score += 0.15; }
        if has_attendees { score += 0.15; }

        score.min(1.0)
    }

    fn extract_assignee_from_text(&self, text: &str) -> Option<String> {
        let mention_regex = Regex::new(r"@([a-zA-Z0-9_]+)").unwrap();

        mention_regex.captures(text)
            .and_then(|cap| cap.get(1))
            .map(|m| m.as_str().to_string())
    }

    fn extract_name_near_email(&self, email: &str, text: &str) -> Option<String> {
        // Look for name patterns near the email
        let patterns = [
            format!(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*<{}>", regex::escape(email)),
            format!(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+{}", regex::escape(email)),
        ];

        for pattern in patterns {
            if let Ok(re) = Regex::new(&pattern) {
                if let Some(cap) = re.captures(text) {
                    if let Some(name) = cap.get(1) {
                        return Some(name.as_str().to_string());
                    }
                }
            }
        }

        None
    }
}

impl Default for ContentExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Extraction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub meetings: Vec<ExtractedMeeting>,
    pub actions: Vec<ExtractedAction>,
    pub contacts: Vec<ExtractedContact>,
    pub deadlines: Vec<ExtractedDeadline>,
    pub links: Vec<String>,
    pub mentions: Vec<String>,
}

/// Simple string similarity (Jaccard index on words)
fn similarity(a: &str, b: &str) -> f64 {
    let words_a: HashSet<&str> = a.split_whitespace().collect();
    let words_b: HashSet<&str> = b.split_whitespace().collect();

    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meeting_extraction() {
        let extractor = ContentExtractor::new();

        let text = "Let's schedule a meeting for tomorrow at 2pm. \
                    Join via https://zoom.us/j/123456789. \
                    Please confirm your attendance.";

        let meetings = extractor.extract_meetings(text);

        assert!(!meetings.is_empty());
        assert!(meetings[0].time.is_some());
        assert!(matches!(meetings[0].location, Some(MeetingLocation::Virtual { .. })));
    }

    #[test]
    fn test_action_extraction() {
        let extractor = ContentExtractor::new();

        let text = "Please review the document by Friday. \
                    @john can you update the spreadsheet? \
                    Action item: Send the report to the team.";

        let actions = extractor.extract_actions(text);

        assert!(actions.len() >= 2);
    }
}
