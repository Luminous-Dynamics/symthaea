// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Track CD: Calendar Integration
// Integration with calendar systems for scheduling and event detection

use chrono::{DateTime, Duration, NaiveDate, NaiveTime, Utc, Weekday};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use regex::Regex;

/// Supported calendar providers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CalendarProvider {
    GoogleCalendar,
    Outlook365,
    AppleCalendar,
    CalDAV(String), // Custom CalDAV server URL
    ICS, // iCalendar file import
}

/// Calendar connection for a user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarConnection {
    pub id: Uuid,
    pub user_id: Uuid,
    pub provider: CalendarProvider,
    pub name: String,
    pub email: String,
    pub access_token: Option<String>,
    pub refresh_token: Option<String>,
    pub token_expires_at: Option<DateTime<Utc>>,
    pub caldav_url: Option<String>,
    pub sync_enabled: bool,
    pub last_synced: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

/// Calendar event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarEvent {
    pub id: Uuid,
    pub external_id: Option<String>,
    pub calendar_id: Uuid,
    pub title: String,
    pub description: Option<String>,
    pub location: Option<String>,
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub all_day: bool,
    pub recurrence: Option<RecurrenceRule>,
    pub attendees: Vec<Attendee>,
    pub organizer: Option<String>,
    pub status: EventStatus,
    pub visibility: EventVisibility,
    pub reminders: Vec<Reminder>,
    pub conference: Option<ConferenceInfo>,
    pub source_email_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Event recurrence rule (simplified RRULE)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurrenceRule {
    pub frequency: RecurrenceFrequency,
    pub interval: u32,
    pub count: Option<u32>,
    pub until: Option<DateTime<Utc>>,
    pub by_day: Option<Vec<Weekday>>,
    pub by_month_day: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecurrenceFrequency {
    Daily,
    Weekly,
    Monthly,
    Yearly,
}

/// Event attendee
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attendee {
    pub email: String,
    pub name: Option<String>,
    pub response_status: ResponseStatus,
    pub optional: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResponseStatus {
    NeedsAction,
    Declined,
    Tentative,
    Accepted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EventStatus {
    Confirmed,
    Tentative,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EventVisibility {
    Public,
    Private,
    Confidential,
}

/// Event reminder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reminder {
    pub method: ReminderMethod,
    pub minutes_before: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReminderMethod {
    Email,
    Popup,
    Push,
}

/// Video conference info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConferenceInfo {
    pub provider: String,
    pub url: String,
    pub meeting_id: Option<String>,
    pub passcode: Option<String>,
}

/// Free/busy time slot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeBusySlot {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub status: BusyStatus,
    pub event_title: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BusyStatus {
    Free,
    Busy,
    Tentative,
    OutOfOffice,
}

/// Meeting suggestion based on attendee availability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingSuggestion {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub score: f64,
    pub conflicts: Vec<String>,
    pub within_working_hours: bool,
}

/// Email event detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedEvent {
    pub title: String,
    pub date: Option<NaiveDate>,
    pub time: Option<NaiveTime>,
    pub duration_minutes: Option<u32>,
    pub location: Option<String>,
    pub attendees: Vec<String>,
    pub confidence: f64,
    pub source_text: String,
}

/// Calendar service
pub struct CalendarService {
    connections: HashMap<Uuid, CalendarConnection>,
    events: HashMap<Uuid, CalendarEvent>,
    user_events: HashMap<Uuid, Vec<Uuid>>,
}

impl Default for CalendarService {
    fn default() -> Self {
        Self::new()
    }
}

impl CalendarService {
    pub fn new() -> Self {
        Self {
            connections: HashMap::new(),
            events: HashMap::new(),
            user_events: HashMap::new(),
        }
    }

    /// Connect a calendar provider
    pub async fn connect_calendar(
        &mut self,
        user_id: Uuid,
        provider: CalendarProvider,
        name: String,
        email: String,
        access_token: Option<String>,
        refresh_token: Option<String>,
        caldav_url: Option<String>,
    ) -> Result<CalendarConnection, CalendarError> {
        let connection = CalendarConnection {
            id: Uuid::new_v4(),
            user_id,
            provider,
            name,
            email,
            access_token,
            refresh_token,
            token_expires_at: None,
            caldav_url,
            sync_enabled: true,
            last_synced: None,
            created_at: Utc::now(),
        };

        self.connections.insert(connection.id, connection.clone());

        // Initial sync
        self.sync_calendar(connection.id).await?;

        Ok(connection)
    }

    /// Disconnect calendar
    pub async fn disconnect_calendar(&mut self, connection_id: Uuid) -> Result<(), CalendarError> {
        let connection = self.connections.remove(&connection_id)
            .ok_or(CalendarError::ConnectionNotFound)?;

        // Remove all events from this calendar
        let events_to_remove: Vec<Uuid> = self.events.iter()
            .filter(|(_, e)| e.calendar_id == connection_id)
            .map(|(id, _)| *id)
            .collect();

        for event_id in events_to_remove {
            self.events.remove(&event_id);
        }

        // Update user events index
        if let Some(user_events) = self.user_events.get_mut(&connection.user_id) {
            user_events.retain(|id| self.events.contains_key(id));
        }

        Ok(())
    }

    /// Sync calendar events from provider
    pub async fn sync_calendar(&mut self, connection_id: Uuid) -> Result<usize, CalendarError> {
        let connection = self.connections.get_mut(&connection_id)
            .ok_or(CalendarError::ConnectionNotFound)?;

        if !connection.sync_enabled {
            return Ok(0);
        }

        // In production, this would call the provider API
        // For now, we simulate sync
        match &connection.provider {
            CalendarProvider::GoogleCalendar => {
                // Call Google Calendar API
                // GET https://www.googleapis.com/calendar/v3/calendars/primary/events
            }
            CalendarProvider::Outlook365 => {
                // Call Microsoft Graph API
                // GET https://graph.microsoft.com/v1.0/me/calendar/events
            }
            CalendarProvider::CalDAV(url) => {
                // Perform REPORT request on CalDAV server
            }
            CalendarProvider::AppleCalendar => {
                // CalDAV to icloud.com
            }
            CalendarProvider::ICS => {
                // Parse ICS file - no ongoing sync needed
            }
        }

        connection.last_synced = Some(Utc::now());

        Ok(0)
    }

    /// Create an event
    pub async fn create_event(
        &mut self,
        calendar_id: Uuid,
        title: String,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        description: Option<String>,
        location: Option<String>,
        attendees: Vec<Attendee>,
    ) -> Result<CalendarEvent, CalendarError> {
        let connection = self.connections.get(&calendar_id)
            .ok_or(CalendarError::ConnectionNotFound)?;

        let event = CalendarEvent {
            id: Uuid::new_v4(),
            external_id: None,
            calendar_id,
            title,
            description,
            location,
            start,
            end,
            all_day: false,
            recurrence: None,
            attendees,
            organizer: Some(connection.email.clone()),
            status: EventStatus::Confirmed,
            visibility: EventVisibility::Private,
            reminders: vec![
                Reminder { method: ReminderMethod::Popup, minutes_before: 15 },
                Reminder { method: ReminderMethod::Email, minutes_before: 60 },
            ],
            conference: None,
            source_email_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        self.events.insert(event.id, event.clone());
        self.user_events
            .entry(connection.user_id)
            .or_default()
            .push(event.id);

        // In production: push to calendar provider

        Ok(event)
    }

    /// Create event from email
    pub async fn create_event_from_email(
        &mut self,
        calendar_id: Uuid,
        email_id: Uuid,
        detected: &DetectedEvent,
    ) -> Result<CalendarEvent, CalendarError> {
        // Build datetime from detected parts
        let start = if let (Some(date), Some(time)) = (detected.date, detected.time) {
            date.and_time(time).and_utc()
        } else {
            return Err(CalendarError::IncompleteEventInfo);
        };

        let duration = detected.duration_minutes.unwrap_or(60);
        let end = start + Duration::minutes(duration as i64);

        let attendees: Vec<Attendee> = detected.attendees.iter()
            .map(|email| Attendee {
                email: email.clone(),
                name: None,
                response_status: ResponseStatus::NeedsAction,
                optional: false,
            })
            .collect();

        let mut event = self.create_event(
            calendar_id,
            detected.title.clone(),
            start,
            end,
            None,
            detected.location.clone(),
            attendees,
        ).await?;

        event.source_email_id = Some(email_id);
        self.events.insert(event.id, event.clone());

        Ok(event)
    }

    /// Update an event
    pub async fn update_event(
        &mut self,
        event_id: Uuid,
        updates: EventUpdate,
    ) -> Result<CalendarEvent, CalendarError> {
        let event = self.events.get_mut(&event_id)
            .ok_or(CalendarError::EventNotFound)?;

        if let Some(title) = updates.title {
            event.title = title;
        }
        if let Some(start) = updates.start {
            event.start = start;
        }
        if let Some(end) = updates.end {
            event.end = end;
        }
        if let Some(description) = updates.description {
            event.description = description;
        }
        if let Some(location) = updates.location {
            event.location = location;
        }
        if let Some(status) = updates.status {
            event.status = status;
        }

        event.updated_at = Utc::now();

        // In production: push update to calendar provider

        Ok(event.clone())
    }

    /// Delete an event
    pub async fn delete_event(&mut self, event_id: Uuid) -> Result<(), CalendarError> {
        let event = self.events.remove(&event_id)
            .ok_or(CalendarError::EventNotFound)?;

        // Update user index
        if let Some(calendar) = self.connections.get(&event.calendar_id) {
            if let Some(user_events) = self.user_events.get_mut(&calendar.user_id) {
                user_events.retain(|id| *id != event_id);
            }
        }

        // In production: delete from calendar provider

        Ok(())
    }

    /// Respond to event invitation
    pub async fn respond_to_event(
        &mut self,
        event_id: Uuid,
        user_email: &str,
        response: ResponseStatus,
    ) -> Result<(), CalendarError> {
        let event = self.events.get_mut(&event_id)
            .ok_or(CalendarError::EventNotFound)?;

        if let Some(attendee) = event.attendees.iter_mut()
            .find(|a| a.email.eq_ignore_ascii_case(user_email))
        {
            attendee.response_status = response;
            event.updated_at = Utc::now();

            // In production: send response to organizer

            Ok(())
        } else {
            Err(CalendarError::NotAttendee)
        }
    }

    /// Get events for date range
    pub fn get_events(
        &self,
        user_id: Uuid,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<&CalendarEvent> {
        self.user_events.get(&user_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.events.get(id))
                    .filter(|e| e.start < end && e.end > start)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get free/busy slots for user
    pub fn get_free_busy(
        &self,
        user_id: Uuid,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<FreeBusySlot> {
        let events = self.get_events(user_id, start, end);

        events.iter().map(|e| {
            FreeBusySlot {
                start: e.start,
                end: e.end,
                status: match e.status {
                    EventStatus::Confirmed => BusyStatus::Busy,
                    EventStatus::Tentative => BusyStatus::Tentative,
                    EventStatus::Cancelled => BusyStatus::Free,
                },
                event_title: if e.visibility == EventVisibility::Public {
                    Some(e.title.clone())
                } else {
                    None
                },
            }
        }).collect()
    }

    /// Find available meeting times
    pub fn suggest_meeting_times(
        &self,
        attendee_ids: Vec<Uuid>,
        duration_minutes: i64,
        search_start: DateTime<Utc>,
        search_end: DateTime<Utc>,
        working_hours: Option<(u32, u32)>,
    ) -> Vec<MeetingSuggestion> {
        let working_hours = working_hours.unwrap_or((9, 17)); // 9 AM - 5 PM default
        let mut suggestions = Vec::new();

        // Collect all busy times for all attendees
        let mut all_busy: Vec<FreeBusySlot> = Vec::new();
        for attendee_id in &attendee_ids {
            all_busy.extend(self.get_free_busy(*attendee_id, search_start, search_end));
        }

        // Sort by start time
        all_busy.sort_by_key(|s| s.start);

        // Find gaps
        let duration = Duration::minutes(duration_minutes);
        let mut current = search_start;

        while current + duration <= search_end {
            let slot_end = current + duration;

            // Check if within working hours
            let hour = current.hour();
            let within_working = hour >= working_hours.0 && hour < working_hours.1;

            // Check conflicts
            let conflicts: Vec<String> = all_busy.iter()
                .filter(|b| b.start < slot_end && b.end > current && b.status == BusyStatus::Busy)
                .filter_map(|b| b.event_title.clone())
                .collect();

            if conflicts.is_empty() {
                let score = if within_working { 1.0 } else { 0.5 };

                suggestions.push(MeetingSuggestion {
                    start: current,
                    end: slot_end,
                    score,
                    conflicts: Vec::new(),
                    within_working_hours: within_working,
                });
            }

            // Advance by 30 minutes
            current = current + Duration::minutes(30);
        }

        // Sort by score (best first)
        suggestions.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        suggestions.truncate(10);

        suggestions
    }

    /// Detect events from email text using NLP patterns
    pub fn detect_events_from_email(&self, subject: &str, body: &str) -> Vec<DetectedEvent> {
        let mut events = Vec::new();
        let full_text = format!("{}\n{}", subject, body);

        // Date patterns
        let date_patterns = vec![
            // "January 15, 2024" or "Jan 15, 2024"
            r"(?i)(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})",
            // "15/01/2024" or "01/15/2024" (ambiguous, assume MM/DD/YYYY)
            r"(\d{1,2})/(\d{1,2})/(\d{4})",
            // "2024-01-15"
            r"(\d{4})-(\d{2})-(\d{2})",
            // "next Monday", "this Friday"
            r"(?i)(next|this)\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)",
            // "tomorrow", "today"
            r"(?i)(tomorrow|today)",
        ];

        // Time patterns
        let time_patterns = vec![
            // "3:00 PM", "3pm", "15:00"
            r"(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?",
            r"(\d{1,2})\s*(AM|PM|am|pm)",
            // "at 3 o'clock"
            r"(?i)at\s+(\d{1,2})\s+o'?clock",
        ];

        // Meeting indicator patterns
        let meeting_patterns = vec![
            r"(?i)meeting\s+(?:about|regarding|for|on)\s+(.+?)(?:\.|,|$)",
            r"(?i)let's\s+(?:meet|discuss|talk|chat)\s+(?:about|regarding)?\s*(.+?)(?:\.|,|$)",
            r"(?i)(?:schedule|scheduled|scheduling)\s+(?:a\s+)?(?:call|meeting|sync)\s*(?:about|for)?\s*(.+?)(?:\.|,|$)",
            r"(?i)invite(?:d)?\s+(?:you\s+)?to\s+(.+?)(?:\.|,|$)",
        ];

        // Location patterns
        let location_patterns = vec![
            r"(?i)(?:at|in|location:?)\s+([A-Z][^,.\n]+(?:Room|Office|Building|Conference|Floor)[^,.\n]*)",
            r"(?i)zoom\s*(?:link|meeting)?:?\s*(https?://\S+)",
            r"(?i)meet(?:\.google)?\.(?:google\.)?com/\S+",
            r"(?i)teams\.microsoft\.com/\S+",
        ];

        // Try to detect meeting keywords
        let has_meeting_keyword = full_text.to_lowercase().contains("meeting") ||
            full_text.to_lowercase().contains("schedule") ||
            full_text.to_lowercase().contains("appointment") ||
            full_text.to_lowercase().contains("call") ||
            full_text.to_lowercase().contains("invite");

        if !has_meeting_keyword {
            return events;
        }

        // Extract title from meeting patterns
        let mut title: Option<String> = None;
        for pattern in &meeting_patterns {
            if let Ok(re) = Regex::new(pattern) {
                if let Some(caps) = re.captures(&full_text) {
                    title = caps.get(1).map(|m| m.as_str().trim().to_string());
                    break;
                }
            }
        }

        // If no specific title found, use subject
        let title = title.unwrap_or_else(|| {
            subject.trim()
                .trim_start_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace())
                .to_string()
        });

        // Extract date (simplified for demo)
        let date: Option<NaiveDate> = None; // Would parse from date_patterns

        // Extract time (simplified for demo)
        let time: Option<NaiveTime> = None; // Would parse from time_patterns

        // Extract location
        let mut location: Option<String> = None;
        for pattern in &location_patterns {
            if let Ok(re) = Regex::new(pattern) {
                if let Some(caps) = re.captures(&full_text) {
                    location = caps.get(1).map(|m| m.as_str().trim().to_string());
                    break;
                }
            }
        }

        // Extract email addresses as potential attendees
        let email_re = Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap();
        let attendees: Vec<String> = email_re.find_iter(&full_text)
            .map(|m| m.as_str().to_lowercase())
            .collect();

        // Only add if we have enough information
        let confidence = if date.is_some() && time.is_some() {
            0.9
        } else if date.is_some() || time.is_some() {
            0.6
        } else {
            0.3
        };

        if has_meeting_keyword {
            events.push(DetectedEvent {
                title,
                date,
                time,
                duration_minutes: Some(60), // Default 1 hour
                location,
                attendees,
                confidence,
                source_text: full_text.chars().take(200).collect(),
            });
        }

        events
    }

    /// Get user's calendar connections
    pub fn get_connections(&self, user_id: Uuid) -> Vec<&CalendarConnection> {
        self.connections.values()
            .filter(|c| c.user_id == user_id)
            .collect()
    }

    /// Get upcoming events
    pub fn get_upcoming_events(
        &self,
        user_id: Uuid,
        limit: usize,
    ) -> Vec<&CalendarEvent> {
        let now = Utc::now();
        let future = now + Duration::days(30);

        let mut events = self.get_events(user_id, now, future);
        events.sort_by_key(|e| e.start);
        events.truncate(limit);
        events
    }

    /// Get today's agenda
    pub fn get_todays_agenda(&self, user_id: Uuid) -> Vec<&CalendarEvent> {
        let now = Utc::now();
        let today_start = now.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc();
        let today_end = now.date_naive().and_hms_opt(23, 59, 59).unwrap().and_utc();

        let mut events = self.get_events(user_id, today_start, today_end);
        events.sort_by_key(|e| e.start);
        events
    }
}

/// Event update fields
#[derive(Debug, Default)]
pub struct EventUpdate {
    pub title: Option<String>,
    pub start: Option<DateTime<Utc>>,
    pub end: Option<DateTime<Utc>>,
    pub description: Option<Option<String>>,
    pub location: Option<Option<String>>,
    pub status: Option<EventStatus>,
}

/// Calendar errors
#[derive(Debug, Clone)]
pub enum CalendarError {
    ConnectionNotFound,
    EventNotFound,
    IncompleteEventInfo,
    NotAttendee,
    SyncFailed(String),
    ProviderError(String),
    AuthenticationRequired,
}

impl std::fmt::Display for CalendarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionNotFound => write!(f, "Calendar connection not found"),
            Self::EventNotFound => write!(f, "Event not found"),
            Self::IncompleteEventInfo => write!(f, "Incomplete event information"),
            Self::NotAttendee => write!(f, "User is not an attendee of this event"),
            Self::SyncFailed(msg) => write!(f, "Calendar sync failed: {}", msg),
            Self::ProviderError(msg) => write!(f, "Calendar provider error: {}", msg),
            Self::AuthenticationRequired => write!(f, "Calendar authentication required"),
        }
    }
}

impl std::error::Error for CalendarError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_event() {
        let mut service = CalendarService::new();
        let user_id = Uuid::new_v4();

        // First connect a calendar
        let connection = service.connect_calendar(
            user_id,
            CalendarProvider::GoogleCalendar,
            "Personal".to_string(),
            "test@gmail.com".to_string(),
            Some("mock_token".to_string()),
            None,
            None,
        ).await.unwrap();

        // Create event
        let start = Utc::now() + Duration::hours(1);
        let end = start + Duration::hours(1);

        let event = service.create_event(
            connection.id,
            "Team Meeting".to_string(),
            start,
            end,
            Some("Weekly sync".to_string()),
            Some("Conference Room A".to_string()),
            vec![],
        ).await.unwrap();

        assert_eq!(event.title, "Team Meeting");

        // Get upcoming events
        let events = service.get_upcoming_events(user_id, 10);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_event_detection() {
        let service = CalendarService::new();

        let subject = "Meeting Request: Q4 Planning Session";
        let body = "Hi team,\n\nLet's schedule a meeting to discuss Q4 planning.\n\
                   When: January 15, 2024 at 3:00 PM\n\
                   Where: Conference Room B\n\
                   Attendees: alice@company.com, bob@company.com\n\
                   \nPlease confirm your attendance.";

        let events = service.detect_events_from_email(subject, body);

        assert!(!events.is_empty());
        assert!(events[0].attendees.contains(&"alice@company.com".to_string()));
    }

    #[test]
    fn test_meeting_suggestions() {
        let mut service = CalendarService::new();
        let user_id = Uuid::new_v4();

        let now = Utc::now();
        let search_start = now;
        let search_end = now + Duration::hours(8);

        let suggestions = service.suggest_meeting_times(
            vec![user_id],
            60,
            search_start,
            search_end,
            Some((9, 17)),
        );

        // Should find available slots
        assert!(!suggestions.is_empty());
    }
}
