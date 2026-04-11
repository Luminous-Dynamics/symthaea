// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Calendar Integration
//!
//! ICS parsing, event extraction, and calendar sync

use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use tracing::{info, warn};
use uuid::Uuid;

/// Calendar event extracted from email
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct CalendarEvent {
    pub id: Uuid,
    pub user_id: Uuid,
    pub tenant_id: Option<Uuid>,
    pub email_id: Option<Uuid>,
    pub uid: String,
    pub summary: String,
    pub description: Option<String>,
    pub location: Option<String>,
    pub organizer: Option<String>,
    pub attendees: Vec<String>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub all_day: bool,
    pub recurrence_rule: Option<String>,
    pub status: EventStatus,
    pub response_status: Option<ResponseStatus>,
    pub reminder_minutes: Option<i32>,
    pub source: EventSource,
    pub raw_ics: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "event_status", rename_all = "lowercase")]
pub enum EventStatus {
    Tentative,
    Confirmed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "response_status", rename_all = "lowercase")]
pub enum ResponseStatus {
    NeedsAction,
    Accepted,
    Declined,
    Tentative,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "event_source", rename_all = "lowercase")]
pub enum EventSource {
    Email,
    Manual,
    CalDav,
    Google,
    Outlook,
}

/// ICS Parser
pub struct IcsParser;

impl IcsParser {
    /// Parse ICS content into calendar events
    pub fn parse(ics_content: &str) -> Result<Vec<ParsedEvent>, IcsError> {
        let mut events = Vec::new();
        let mut current_event: Option<ParsedEventBuilder> = None;
        let mut in_vevent = false;

        for line in unfold_lines(ics_content) {
            let line = line.trim();

            if line.starts_with("BEGIN:VEVENT") {
                in_vevent = true;
                current_event = Some(ParsedEventBuilder::new());
            } else if line.starts_with("END:VEVENT") {
                if let Some(builder) = current_event.take() {
                    if let Ok(event) = builder.build() {
                        events.push(event);
                    }
                }
                in_vevent = false;
            } else if in_vevent {
                if let Some(ref mut builder) = current_event {
                    builder.parse_line(line);
                }
            }
        }

        if events.is_empty() && ics_content.contains("VEVENT") {
            return Err(IcsError::ParseError("No valid events found".to_string()));
        }

        Ok(events)
    }

    /// Extract calendar invites from email content
    pub fn extract_from_email(
        body: &str,
        attachments: &[(String, Vec<u8>)],
    ) -> Vec<ParsedEvent> {
        let mut events = Vec::new();

        // Check attachments for .ics files
        for (filename, content) in attachments {
            if filename.ends_with(".ics") || filename.ends_with(".ical") {
                if let Ok(content_str) = std::str::from_utf8(content) {
                    if let Ok(parsed) = Self::parse(content_str) {
                        events.extend(parsed);
                    }
                }
            }
        }

        // Check for inline ICS in body
        if body.contains("BEGIN:VCALENDAR") {
            if let Some(start) = body.find("BEGIN:VCALENDAR") {
                if let Some(end) = body.find("END:VCALENDAR") {
                    let ics_content = &body[start..end + "END:VCALENDAR".len()];
                    if let Ok(parsed) = Self::parse(ics_content) {
                        events.extend(parsed);
                    }
                }
            }
        }

        events
    }
}

/// Unfold ICS lines (lines can be continued with leading whitespace)
fn unfold_lines(content: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current_line = String::new();

    for line in content.lines() {
        if line.starts_with(' ') || line.starts_with('\t') {
            // Continuation line
            current_line.push_str(line.trim_start());
        } else {
            if !current_line.is_empty() {
                result.push(current_line);
            }
            current_line = line.to_string();
        }
    }

    if !current_line.is_empty() {
        result.push(current_line);
    }

    result
}

/// Parsed event from ICS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedEvent {
    pub uid: String,
    pub summary: String,
    pub description: Option<String>,
    pub location: Option<String>,
    pub organizer: Option<String>,
    pub attendees: Vec<Attendee>,
    pub start: EventDateTime,
    pub end: EventDateTime,
    pub recurrence_rule: Option<String>,
    pub status: Option<String>,
    pub sequence: i32,
    pub method: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attendee {
    pub email: String,
    pub name: Option<String>,
    pub role: Option<String>,
    pub status: Option<String>,
    pub rsvp: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventDateTime {
    DateTime(DateTime<Utc>),
    Date(NaiveDate),
}

impl EventDateTime {
    pub fn to_utc(&self) -> DateTime<Utc> {
        match self {
            EventDateTime::DateTime(dt) => *dt,
            EventDateTime::Date(d) => {
                Utc.from_utc_datetime(&d.and_hms_opt(0, 0, 0).unwrap())
            }
        }
    }

    pub fn is_all_day(&self) -> bool {
        matches!(self, EventDateTime::Date(_))
    }
}

struct ParsedEventBuilder {
    uid: Option<String>,
    summary: Option<String>,
    description: Option<String>,
    location: Option<String>,
    organizer: Option<String>,
    attendees: Vec<Attendee>,
    dtstart: Option<EventDateTime>,
    dtend: Option<EventDateTime>,
    rrule: Option<String>,
    status: Option<String>,
    sequence: i32,
    method: Option<String>,
}

impl ParsedEventBuilder {
    fn new() -> Self {
        Self {
            uid: None,
            summary: None,
            description: None,
            location: None,
            organizer: None,
            attendees: Vec::new(),
            dtstart: None,
            dtend: None,
            rrule: None,
            status: None,
            sequence: 0,
            method: None,
        }
    }

    fn parse_line(&mut self, line: &str) {
        if let Some((key, value)) = split_ics_line(line) {
            match key.split(';').next().unwrap_or(&key) {
                "UID" => self.uid = Some(value.to_string()),
                "SUMMARY" => self.summary = Some(unescape_ics_text(&value)),
                "DESCRIPTION" => self.description = Some(unescape_ics_text(&value)),
                "LOCATION" => self.location = Some(unescape_ics_text(&value)),
                "ORGANIZER" => {
                    self.organizer = Some(extract_email(&value));
                }
                "ATTENDEE" => {
                    self.attendees.push(parse_attendee(&key, &value));
                }
                "DTSTART" => {
                    self.dtstart = parse_datetime(&key, &value);
                }
                "DTEND" => {
                    self.dtend = parse_datetime(&key, &value);
                }
                "RRULE" => self.rrule = Some(value.to_string()),
                "STATUS" => self.status = Some(value.to_string()),
                "SEQUENCE" => {
                    self.sequence = value.parse().unwrap_or(0);
                }
                "METHOD" => self.method = Some(value.to_string()),
                _ => {}
            }
        }
    }

    fn build(self) -> Result<ParsedEvent, IcsError> {
        let uid = self.uid.ok_or_else(|| IcsError::MissingField("UID".to_string()))?;
        let summary = self.summary.unwrap_or_else(|| "(No title)".to_string());
        let start = self.dtstart.ok_or_else(|| IcsError::MissingField("DTSTART".to_string()))?;

        // If no end time, assume 1 hour duration for datetime, 1 day for date
        let end = self.dtend.unwrap_or_else(|| match &start {
            EventDateTime::DateTime(dt) => EventDateTime::DateTime(*dt + Duration::hours(1)),
            EventDateTime::Date(d) => EventDateTime::Date(*d + Duration::days(1)),
        });

        Ok(ParsedEvent {
            uid,
            summary,
            description: self.description,
            location: self.location,
            organizer: self.organizer,
            attendees: self.attendees,
            start,
            end,
            recurrence_rule: self.rrule,
            status: self.status,
            sequence: self.sequence,
            method: self.method,
        })
    }
}

fn split_ics_line(line: &str) -> Option<(String, String)> {
    let colon_pos = line.find(':')?;
    Some((
        line[..colon_pos].to_string(),
        line[colon_pos + 1..].to_string(),
    ))
}

fn unescape_ics_text(text: &str) -> String {
    text.replace("\\n", "\n")
        .replace("\\,", ",")
        .replace("\\;", ";")
        .replace("\\\\", "\\")
}

fn extract_email(value: &str) -> String {
    // Handle "mailto:email@example.com" format
    if value.to_lowercase().starts_with("mailto:") {
        value[7..].to_string()
    } else {
        value.to_string()
    }
}

fn parse_attendee(key: &str, value: &str) -> Attendee {
    let email = extract_email(value);
    let mut name = None;
    let mut role = None;
    let mut status = None;
    let mut rsvp = false;

    // Parse parameters from key (e.g., ATTENDEE;CN=John;ROLE=REQ-PARTICIPANT)
    for param in key.split(';').skip(1) {
        if let Some((param_key, param_value)) = param.split_once('=') {
            match param_key {
                "CN" => name = Some(param_value.trim_matches('"').to_string()),
                "ROLE" => role = Some(param_value.to_string()),
                "PARTSTAT" => status = Some(param_value.to_string()),
                "RSVP" => rsvp = param_value.eq_ignore_ascii_case("TRUE"),
                _ => {}
            }
        }
    }

    Attendee {
        email,
        name,
        role,
        status,
        rsvp,
    }
}

fn parse_datetime(key: &str, value: &str) -> Option<EventDateTime> {
    // Check for VALUE=DATE parameter
    let is_date_only = key.contains("VALUE=DATE") && !key.contains("VALUE=DATE-TIME");

    if is_date_only {
        // Parse as date only (YYYYMMDD)
        NaiveDate::parse_from_str(value, "%Y%m%d")
            .ok()
            .map(EventDateTime::Date)
    } else {
        // Parse as datetime
        // Try various formats
        let value = value.trim_end_matches('Z');

        if let Ok(dt) = NaiveDateTime::parse_from_str(value, "%Y%m%dT%H%M%S") {
            Some(EventDateTime::DateTime(Utc.from_utc_datetime(&dt)))
        } else if let Ok(dt) = NaiveDateTime::parse_from_str(value, "%Y%m%dT%H%M") {
            Some(EventDateTime::DateTime(Utc.from_utc_datetime(&dt)))
        } else if let Ok(d) = NaiveDate::parse_from_str(value, "%Y%m%d") {
            Some(EventDateTime::Date(d))
        } else {
            None
        }
    }
}

/// Calendar service
pub struct CalendarService {
    pool: PgPool,
}

impl CalendarService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create event from parsed ICS
    pub async fn create_from_ics(
        &self,
        user_id: Uuid,
        tenant_id: Option<Uuid>,
        email_id: Option<Uuid>,
        parsed: ParsedEvent,
        raw_ics: Option<String>,
    ) -> Result<CalendarEvent, CalendarError> {
        let id = Uuid::new_v4();
        let attendee_emails: Vec<String> = parsed.attendees.iter().map(|a| a.email.clone()).collect();

        let status = match parsed.status.as_deref() {
            Some("CANCELLED") => EventStatus::Cancelled,
            Some("TENTATIVE") => EventStatus::Tentative,
            _ => EventStatus::Confirmed,
        };

        let event = sqlx::query_as::<_, CalendarEvent>(
            r#"
            INSERT INTO calendar_events (
                id, user_id, tenant_id, email_id, uid, summary, description,
                location, organizer, attendees, start_time, end_time, all_day,
                recurrence_rule, status, source, raw_ics, created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, 'email', $16, NOW(), NOW())
            ON CONFLICT (user_id, uid) DO UPDATE SET
                summary = EXCLUDED.summary,
                description = EXCLUDED.description,
                location = EXCLUDED.location,
                start_time = EXCLUDED.start_time,
                end_time = EXCLUDED.end_time,
                status = EXCLUDED.status,
                updated_at = NOW()
            RETURNING *
            "#,
        )
        .bind(id)
        .bind(user_id)
        .bind(tenant_id)
        .bind(email_id)
        .bind(&parsed.uid)
        .bind(&parsed.summary)
        .bind(&parsed.description)
        .bind(&parsed.location)
        .bind(&parsed.organizer)
        .bind(&attendee_emails)
        .bind(parsed.start.to_utc())
        .bind(parsed.end.to_utc())
        .bind(parsed.start.is_all_day())
        .bind(&parsed.recurrence_rule)
        .bind(status)
        .bind(&raw_ics)
        .fetch_one(&self.pool)
        .await
        .map_err(CalendarError::Database)?;

        info!(
            event_id = %event.id,
            summary = %event.summary,
            start = %event.start_time,
            "Calendar event created"
        );

        Ok(event)
    }

    /// Respond to calendar invite
    pub async fn respond(
        &self,
        event_id: Uuid,
        user_id: Uuid,
        response: ResponseStatus,
    ) -> Result<CalendarEvent, CalendarError> {
        let event = sqlx::query_as::<_, CalendarEvent>(
            r#"
            UPDATE calendar_events
            SET response_status = $3, updated_at = NOW()
            WHERE id = $1 AND user_id = $2
            RETURNING *
            "#,
        )
        .bind(event_id)
        .bind(user_id)
        .bind(response)
        .fetch_optional(&self.pool)
        .await
        .map_err(CalendarError::Database)?
        .ok_or(CalendarError::NotFound)?;

        info!(
            event_id = %event_id,
            response = ?response,
            "Calendar event response recorded"
        );

        Ok(event)
    }

    /// Generate ICS response
    pub fn generate_response(
        &self,
        event: &CalendarEvent,
        response: ResponseStatus,
        user_email: &str,
    ) -> String {
        let status = match response {
            ResponseStatus::Accepted => "ACCEPTED",
            ResponseStatus::Declined => "DECLINED",
            ResponseStatus::Tentative => "TENTATIVE",
            ResponseStatus::NeedsAction => "NEEDS-ACTION",
        };

        format!(
            r#"BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Mycelix Mail//Calendar//EN
METHOD:REPLY
BEGIN:VEVENT
UID:{}
DTSTAMP:{}
DTSTART:{}
DTEND:{}
SUMMARY:{}
ATTENDEE;PARTSTAT={}:mailto:{}
END:VEVENT
END:VCALENDAR"#,
            event.uid,
            Utc::now().format("%Y%m%dT%H%M%SZ"),
            event.start_time.format("%Y%m%dT%H%M%SZ"),
            event.end_time.format("%Y%m%dT%H%M%SZ"),
            event.summary,
            status,
            user_email,
        )
    }

    /// Get upcoming events
    pub async fn get_upcoming(
        &self,
        user_id: Uuid,
        days: i32,
    ) -> Result<Vec<CalendarEvent>, sqlx::Error> {
        let until = Utc::now() + Duration::days(days as i64);

        sqlx::query_as::<_, CalendarEvent>(
            r#"
            SELECT * FROM calendar_events
            WHERE user_id = $1
              AND status != 'cancelled'
              AND start_time >= NOW()
              AND start_time <= $2
            ORDER BY start_time ASC
            "#,
        )
        .bind(user_id)
        .bind(until)
        .fetch_all(&self.pool)
        .await
    }

    /// Get events for date range
    pub async fn get_range(
        &self,
        user_id: Uuid,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<CalendarEvent>, sqlx::Error> {
        sqlx::query_as::<_, CalendarEvent>(
            r#"
            SELECT * FROM calendar_events
            WHERE user_id = $1
              AND status != 'cancelled'
              AND start_time < $3
              AND end_time > $2
            ORDER BY start_time ASC
            "#,
        )
        .bind(user_id)
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await
    }

    /// Get event by ID
    pub async fn get_event(
        &self,
        event_id: Uuid,
        user_id: Uuid,
    ) -> Result<Option<CalendarEvent>, sqlx::Error> {
        sqlx::query_as::<_, CalendarEvent>(
            "SELECT * FROM calendar_events WHERE id = $1 AND user_id = $2",
        )
        .bind(event_id)
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
    }

    /// Delete event
    pub async fn delete_event(
        &self,
        event_id: Uuid,
        user_id: Uuid,
    ) -> Result<bool, sqlx::Error> {
        let result = sqlx::query(
            "DELETE FROM calendar_events WHERE id = $1 AND user_id = $2",
        )
        .bind(event_id)
        .bind(user_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    /// Set reminder
    pub async fn set_reminder(
        &self,
        event_id: Uuid,
        user_id: Uuid,
        minutes: i32,
    ) -> Result<CalendarEvent, CalendarError> {
        sqlx::query_as::<_, CalendarEvent>(
            r#"
            UPDATE calendar_events
            SET reminder_minutes = $3, updated_at = NOW()
            WHERE id = $1 AND user_id = $2
            RETURNING *
            "#,
        )
        .bind(event_id)
        .bind(user_id)
        .bind(minutes)
        .fetch_optional(&self.pool)
        .await
        .map_err(CalendarError::Database)?
        .ok_or(CalendarError::NotFound)
    }

    /// Get events needing reminders
    pub async fn get_pending_reminders(&self) -> Result<Vec<CalendarEvent>, sqlx::Error> {
        sqlx::query_as::<_, CalendarEvent>(
            r#"
            SELECT * FROM calendar_events
            WHERE reminder_minutes IS NOT NULL
              AND status = 'confirmed'
              AND start_time > NOW()
              AND start_time <= NOW() + (reminder_minutes || ' minutes')::INTERVAL
              AND response_status IN ('accepted', 'tentative')
            "#,
        )
        .fetch_all(&self.pool)
        .await
    }
}

/// Calendar errors
#[derive(Debug)]
pub enum CalendarError {
    NotFound,
    Database(sqlx::Error),
    Ics(IcsError),
}

impl std::fmt::Display for CalendarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "Event not found"),
            Self::Database(e) => write!(f, "Database error: {}", e),
            Self::Ics(e) => write!(f, "ICS error: {}", e),
        }
    }
}

impl std::error::Error for CalendarError {}

/// ICS parsing errors
#[derive(Debug)]
pub enum IcsError {
    ParseError(String),
    MissingField(String),
    InvalidDate(String),
}

impl std::fmt::Display for IcsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::MissingField(field) => write!(f, "Missing required field: {}", field),
            Self::InvalidDate(msg) => write!(f, "Invalid date: {}", msg),
        }
    }
}

impl std::error::Error for IcsError {}

/// Migration for calendar
pub const CALENDAR_MIGRATION: &str = r#"
DO $$ BEGIN
    CREATE TYPE event_status AS ENUM ('tentative', 'confirmed', 'cancelled');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE response_status AS ENUM ('needs_action', 'accepted', 'declined', 'tentative');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE event_source AS ENUM ('email', 'manual', 'caldav', 'google', 'outlook');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

CREATE TABLE IF NOT EXISTS calendar_events (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    email_id UUID REFERENCES emails(id) ON DELETE SET NULL,
    uid VARCHAR(255) NOT NULL,
    summary VARCHAR(500) NOT NULL,
    description TEXT,
    location VARCHAR(500),
    organizer VARCHAR(255),
    attendees TEXT[] NOT NULL DEFAULT '{}',
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    all_day BOOLEAN NOT NULL DEFAULT false,
    recurrence_rule TEXT,
    status event_status NOT NULL DEFAULT 'confirmed',
    response_status response_status,
    reminder_minutes INTEGER,
    source event_source NOT NULL DEFAULT 'email',
    raw_ics TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, uid)
);

CREATE INDEX IF NOT EXISTS idx_calendar_events_user_time
    ON calendar_events(user_id, start_time);
CREATE INDEX IF NOT EXISTS idx_calendar_events_upcoming
    ON calendar_events(user_id, start_time)
    WHERE status = 'confirmed' AND start_time > NOW();
CREATE INDEX IF NOT EXISTS idx_calendar_events_email
    ON calendar_events(email_id)
    WHERE email_id IS NOT NULL;
"#;

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_ICS: &str = r#"BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Test//Test//EN
METHOD:REQUEST
BEGIN:VEVENT
UID:test-event-123@example.com
DTSTAMP:20240101T120000Z
DTSTART:20240115T140000Z
DTEND:20240115T150000Z
SUMMARY:Team Meeting
DESCRIPTION:Weekly sync meeting
LOCATION:Conference Room A
ORGANIZER;CN=John Doe:mailto:john@example.com
ATTENDEE;CN=Jane Smith;ROLE=REQ-PARTICIPANT;PARTSTAT=NEEDS-ACTION;RSVP=TRUE:mailto:jane@example.com
STATUS:CONFIRMED
END:VEVENT
END:VCALENDAR"#;

    #[test]
    fn test_parse_ics() {
        let events = IcsParser::parse(SAMPLE_ICS).unwrap();
        assert_eq!(events.len(), 1);

        let event = &events[0];
        assert_eq!(event.uid, "test-event-123@example.com");
        assert_eq!(event.summary, "Team Meeting");
        assert_eq!(event.location, Some("Conference Room A".to_string()));
        assert_eq!(event.attendees.len(), 1);
        assert_eq!(event.attendees[0].email, "jane@example.com");
    }

    #[test]
    fn test_parse_all_day_event() {
        let ics = r#"BEGIN:VCALENDAR
BEGIN:VEVENT
UID:allday@example.com
DTSTART;VALUE=DATE:20240115
DTEND;VALUE=DATE:20240116
SUMMARY:Holiday
END:VEVENT
END:VCALENDAR"#;

        let events = IcsParser::parse(ics).unwrap();
        assert_eq!(events.len(), 1);
        assert!(events[0].start.is_all_day());
    }
}
