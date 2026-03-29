// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Calendar Synchronization Module
//!
//! Bi-directional sync with Google Calendar, Microsoft Outlook, and CalDAV.

use async_trait::async_trait;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ============================================================================
// Core Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarEvent {
    pub id: String,
    pub external_id: Option<String>,
    pub title: String,
    pub description: Option<String>,
    pub location: Option<String>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub all_day: bool,
    pub timezone: String,
    pub attendees: Vec<Attendee>,
    pub organizer: Attendee,
    pub status: EventStatus,
    pub visibility: EventVisibility,
    pub recurrence: Option<RecurrenceRule>,
    pub reminders: Vec<Reminder>,
    pub conference: Option<ConferenceInfo>,
    pub source_email_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub sync_status: SyncStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attendee {
    pub email: String,
    pub name: Option<String>,
    pub response_status: ResponseStatus,
    pub optional: bool,
    pub organizer: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EventStatus {
    Confirmed,
    Tentative,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventVisibility {
    Public,
    Private,
    Confidential,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseStatus {
    NeedsAction,
    Accepted,
    Declined,
    Tentative,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurrenceRule {
    pub frequency: Frequency,
    pub interval: u32,
    pub count: Option<u32>,
    pub until: Option<DateTime<Utc>>,
    pub by_day: Vec<Weekday>,
    pub by_month_day: Vec<i32>,
    pub by_month: Vec<u32>,
    pub exceptions: Vec<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Frequency {
    Daily,
    Weekly,
    Monthly,
    Yearly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Weekday {
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reminder {
    pub method: ReminderMethod,
    pub minutes_before: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReminderMethod {
    Email,
    Popup,
    Sms,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConferenceInfo {
    pub provider: ConferenceProvider,
    pub url: String,
    pub meeting_id: Option<String>,
    pub passcode: Option<String>,
    pub phone_numbers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConferenceProvider {
    GoogleMeet,
    Zoom,
    MicrosoftTeams,
    WebEx,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStatus {
    pub synced: bool,
    pub last_synced: Option<DateTime<Utc>>,
    pub sync_error: Option<String>,
    pub local_changes: bool,
    pub remote_changes: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarAccount {
    pub id: String,
    pub user_id: String,
    pub provider: CalendarProvider,
    pub email: String,
    pub name: String,
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub token_expires_at: Option<DateTime<Utc>>,
    pub sync_enabled: bool,
    pub sync_direction: SyncDirection,
    pub last_sync: Option<DateTime<Utc>>,
    pub sync_token: Option<String>,
    pub calendars: Vec<CalendarInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CalendarProvider {
    Google,
    Microsoft,
    CalDAV,
    Apple,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncDirection {
    PullOnly,
    PushOnly,
    Bidirectional,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarInfo {
    pub id: String,
    pub name: String,
    pub color: String,
    pub primary: bool,
    pub selected: bool,
    pub access_role: AccessRole,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessRole {
    Owner,
    Writer,
    Reader,
    FreeBusyReader,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResult {
    pub success: bool,
    pub events_pulled: u32,
    pub events_pushed: u32,
    pub events_updated: u32,
    pub events_deleted: u32,
    pub conflicts: Vec<SyncConflict>,
    pub errors: Vec<String>,
    pub next_sync_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConflict {
    pub event_id: String,
    pub local_event: CalendarEvent,
    pub remote_event: CalendarEvent,
    pub conflict_type: ConflictType,
    pub resolution: Option<ConflictResolution>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictType {
    BothModified,
    DeletedLocally,
    DeletedRemotely,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
    UseLocal,
    UseRemote,
    Merge,
    Skip,
}

// ============================================================================
// Calendar Sync Trait
// ============================================================================

#[async_trait]
pub trait CalendarSync: Send + Sync {
    fn provider(&self) -> CalendarProvider;

    async fn authenticate(&self, auth_code: &str, redirect_uri: &str) -> Result<CalendarAccount, CalendarError>;
    async fn refresh_token(&self, account: &mut CalendarAccount) -> Result<(), CalendarError>;

    async fn list_calendars(&self, account: &CalendarAccount) -> Result<Vec<CalendarInfo>, CalendarError>;

    async fn sync_events(
        &self,
        account: &CalendarAccount,
        calendar_id: &str,
        sync_token: Option<&str>,
        time_min: DateTime<Utc>,
        time_max: DateTime<Utc>,
    ) -> Result<SyncResult, CalendarError>;

    async fn get_event(
        &self,
        account: &CalendarAccount,
        calendar_id: &str,
        event_id: &str,
    ) -> Result<CalendarEvent, CalendarError>;

    async fn create_event(
        &self,
        account: &CalendarAccount,
        calendar_id: &str,
        event: &CalendarEvent,
    ) -> Result<CalendarEvent, CalendarError>;

    async fn update_event(
        &self,
        account: &CalendarAccount,
        calendar_id: &str,
        event: &CalendarEvent,
    ) -> Result<CalendarEvent, CalendarError>;

    async fn delete_event(
        &self,
        account: &CalendarAccount,
        calendar_id: &str,
        event_id: &str,
    ) -> Result<(), CalendarError>;

    async fn respond_to_event(
        &self,
        account: &CalendarAccount,
        calendar_id: &str,
        event_id: &str,
        response: ResponseStatus,
    ) -> Result<(), CalendarError>;

    async fn get_free_busy(
        &self,
        account: &CalendarAccount,
        emails: Vec<String>,
        time_min: DateTime<Utc>,
        time_max: DateTime<Utc>,
    ) -> Result<HashMap<String, Vec<BusySlot>>, CalendarError>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusySlot {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

#[derive(Debug, thiserror::Error)]
pub enum CalendarError {
    #[error("Authentication failed: {0}")]
    AuthError(String),
    #[error("Token expired")]
    TokenExpired,
    #[error("Calendar not found: {0}")]
    CalendarNotFound(String),
    #[error("Event not found: {0}")]
    EventNotFound(String),
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    #[error("Rate limited: retry after {0} seconds")]
    RateLimited(u64),
    #[error("Sync conflict: {0}")]
    SyncConflict(String),
    #[error("API error: {0}")]
    ApiError(String),
}

// ============================================================================
// Google Calendar Implementation
// ============================================================================

pub struct GoogleCalendarSync {
    client: reqwest::Client,
    client_id: String,
    client_secret: String,
}

impl GoogleCalendarSync {
    pub fn new(client_id: String, client_secret: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            client_id,
            client_secret,
        }
    }
}

#[async_trait]
impl CalendarSync for GoogleCalendarSync {
    fn provider(&self) -> CalendarProvider {
        CalendarProvider::Google
    }

    async fn authenticate(&self, auth_code: &str, redirect_uri: &str) -> Result<CalendarAccount, CalendarError> {
        let token_response = self
            .client
            .post("https://oauth2.googleapis.com/token")
            .form(&[
                ("code", auth_code),
                ("client_id", &self.client_id),
                ("client_secret", &self.client_secret),
                ("redirect_uri", redirect_uri),
                ("grant_type", "authorization_code"),
            ])
            .send()
            .await
            .map_err(|e| CalendarError::ApiError(e.to_string()))?;

        if !token_response.status().is_success() {
            return Err(CalendarError::AuthError("Failed to exchange auth code".into()));
        }

        let token_data: serde_json::Value = token_response
            .json()
            .await
            .map_err(|e| CalendarError::ApiError(e.to_string()))?;

        Ok(CalendarAccount {
            id: Uuid::new_v4().to_string(),
            user_id: String::new(),
            provider: CalendarProvider::Google,
            email: String::new(),
            name: "Google Calendar".to_string(),
            access_token: token_data["access_token"].as_str().unwrap_or("").to_string(),
            refresh_token: token_data["refresh_token"].as_str().map(String::from),
            token_expires_at: Some(Utc::now() + Duration::seconds(token_data["expires_in"].as_i64().unwrap_or(3600))),
            sync_enabled: true,
            sync_direction: SyncDirection::Bidirectional,
            last_sync: None,
            sync_token: None,
            calendars: vec![],
        })
    }

    async fn refresh_token(&self, account: &mut CalendarAccount) -> Result<(), CalendarError> {
        let refresh_token = account
            .refresh_token
            .as_ref()
            .ok_or_else(|| CalendarError::AuthError("No refresh token".into()))?;

        let response = self
            .client
            .post("https://oauth2.googleapis.com/token")
            .form(&[
                ("client_id", self.client_id.as_str()),
                ("client_secret", self.client_secret.as_str()),
                ("refresh_token", refresh_token.as_str()),
                ("grant_type", "refresh_token"),
            ])
            .send()
            .await
            .map_err(|e| CalendarError::ApiError(e.to_string()))?;

        let token_data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| CalendarError::ApiError(e.to_string()))?;

        account.access_token = token_data["access_token"].as_str().unwrap_or("").to_string();
        account.token_expires_at = Some(Utc::now() + Duration::seconds(token_data["expires_in"].as_i64().unwrap_or(3600)));

        Ok(())
    }

    async fn list_calendars(&self, account: &CalendarAccount) -> Result<Vec<CalendarInfo>, CalendarError> {
        let response = self
            .client
            .get("https://www.googleapis.com/calendar/v3/users/me/calendarList")
            .bearer_auth(&account.access_token)
            .send()
            .await
            .map_err(|e| CalendarError::ApiError(e.to_string()))?;

        if response.status().as_u16() == 401 {
            return Err(CalendarError::TokenExpired);
        }

        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| CalendarError::ApiError(e.to_string()))?;

        let calendars = data["items"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|cal| CalendarInfo {
                id: cal["id"].as_str().unwrap_or("").to_string(),
                name: cal["summary"].as_str().unwrap_or("").to_string(),
                color: cal["backgroundColor"].as_str().unwrap_or("#4285F4").to_string(),
                primary: cal["primary"].as_bool().unwrap_or(false),
                selected: true,
                access_role: match cal["accessRole"].as_str() {
                    Some("owner") => AccessRole::Owner,
                    Some("writer") => AccessRole::Writer,
                    Some("reader") => AccessRole::Reader,
                    _ => AccessRole::FreeBusyReader,
                },
            })
            .collect();

        Ok(calendars)
    }

    async fn sync_events(
        &self,
        account: &CalendarAccount,
        calendar_id: &str,
        sync_token: Option<&str>,
        time_min: DateTime<Utc>,
        time_max: DateTime<Utc>,
    ) -> Result<SyncResult, CalendarError> {
        let mut url = format!(
            "https://www.googleapis.com/calendar/v3/calendars/{}/events",
            urlencoding::encode(calendar_id)
        );

        let mut params = vec![
            ("singleEvents", "true".to_string()),
            ("maxResults", "250".to_string()),
        ];

        if let Some(token) = sync_token {
            params.push(("syncToken", token.to_string()));
        } else {
            params.push(("timeMin", time_min.to_rfc3339()));
            params.push(("timeMax", time_max.to_rfc3339()));
        }

        let response = self
            .client
            .get(&url)
            .bearer_auth(&account.access_token)
            .query(&params)
            .send()
            .await
            .map_err(|e| CalendarError::ApiError(e.to_string()))?;

        if response.status().as_u16() == 410 {
            // Sync token expired, need full sync
            return Err(CalendarError::SyncConflict("Sync token expired, need full resync".into()));
        }

        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| CalendarError::ApiError(e.to_string()))?;

        let events_pulled = data["items"].as_array().map(|a| a.len() as u32).unwrap_or(0);

        Ok(SyncResult {
            success: true,
            events_pulled,
            events_pushed: 0,
            events_updated: 0,
            events_deleted: 0,
            conflicts: vec![],
            errors: vec![],
            next_sync_token: data["nextSyncToken"].as_str().map(String::from),
        })
    }

    async fn get_event(
        &self,
        account: &CalendarAccount,
        calendar_id: &str,
        event_id: &str,
    ) -> Result<CalendarEvent, CalendarError> {
        let url = format!(
            "https://www.googleapis.com/calendar/v3/calendars/{}/events/{}",
            urlencoding::encode(calendar_id),
            urlencoding::encode(event_id)
        );

        let response = self
            .client
            .get(&url)
            .bearer_auth(&account.access_token)
            .send()
            .await
            .map_err(|e| CalendarError::ApiError(e.to_string()))?;

        if response.status().as_u16() == 404 {
            return Err(CalendarError::EventNotFound(event_id.to_string()));
        }

        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| CalendarError::ApiError(e.to_string()))?;

        self.parse_google_event(&data)
    }

    async fn create_event(
        &self,
        account: &CalendarAccount,
        calendar_id: &str,
        event: &CalendarEvent,
    ) -> Result<CalendarEvent, CalendarError> {
        let url = format!(
            "https://www.googleapis.com/calendar/v3/calendars/{}/events",
            urlencoding::encode(calendar_id)
        );

        let google_event = self.to_google_event(event);

        let response = self
            .client
            .post(&url)
            .bearer_auth(&account.access_token)
            .json(&google_event)
            .send()
            .await
            .map_err(|e| CalendarError::ApiError(e.to_string()))?;

        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| CalendarError::ApiError(e.to_string()))?;

        self.parse_google_event(&data)
    }

    async fn update_event(
        &self,
        account: &CalendarAccount,
        calendar_id: &str,
        event: &CalendarEvent,
    ) -> Result<CalendarEvent, CalendarError> {
        let external_id = event.external_id.as_ref().ok_or_else(|| {
            CalendarError::EventNotFound("No external ID".into())
        })?;

        let url = format!(
            "https://www.googleapis.com/calendar/v3/calendars/{}/events/{}",
            urlencoding::encode(calendar_id),
            urlencoding::encode(external_id)
        );

        let google_event = self.to_google_event(event);

        let response = self
            .client
            .put(&url)
            .bearer_auth(&account.access_token)
            .json(&google_event)
            .send()
            .await
            .map_err(|e| CalendarError::ApiError(e.to_string()))?;

        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| CalendarError::ApiError(e.to_string()))?;

        self.parse_google_event(&data)
    }

    async fn delete_event(
        &self,
        account: &CalendarAccount,
        calendar_id: &str,
        event_id: &str,
    ) -> Result<(), CalendarError> {
        let url = format!(
            "https://www.googleapis.com/calendar/v3/calendars/{}/events/{}",
            urlencoding::encode(calendar_id),
            urlencoding::encode(event_id)
        );

        let response = self
            .client
            .delete(&url)
            .bearer_auth(&account.access_token)
            .send()
            .await
            .map_err(|e| CalendarError::ApiError(e.to_string()))?;

        if response.status().is_success() || response.status().as_u16() == 410 {
            Ok(())
        } else {
            Err(CalendarError::ApiError("Failed to delete event".into()))
        }
    }

    async fn respond_to_event(
        &self,
        account: &CalendarAccount,
        calendar_id: &str,
        event_id: &str,
        response: ResponseStatus,
    ) -> Result<(), CalendarError> {
        let url = format!(
            "https://www.googleapis.com/calendar/v3/calendars/{}/events/{}",
            urlencoding::encode(calendar_id),
            urlencoding::encode(event_id)
        );

        let response_str = match response {
            ResponseStatus::Accepted => "accepted",
            ResponseStatus::Declined => "declined",
            ResponseStatus::Tentative => "tentative",
            ResponseStatus::NeedsAction => "needsAction",
        };

        let patch_data = serde_json::json!({
            "attendees": [{
                "email": account.email,
                "responseStatus": response_str
            }]
        });

        let api_response = self
            .client
            .patch(&url)
            .bearer_auth(&account.access_token)
            .json(&patch_data)
            .send()
            .await
            .map_err(|e| CalendarError::ApiError(e.to_string()))?;

        if api_response.status().is_success() {
            Ok(())
        } else {
            Err(CalendarError::ApiError("Failed to update response".into()))
        }
    }

    async fn get_free_busy(
        &self,
        account: &CalendarAccount,
        emails: Vec<String>,
        time_min: DateTime<Utc>,
        time_max: DateTime<Utc>,
    ) -> Result<HashMap<String, Vec<BusySlot>>, CalendarError> {
        let request_body = serde_json::json!({
            "timeMin": time_min.to_rfc3339(),
            "timeMax": time_max.to_rfc3339(),
            "items": emails.iter().map(|e| serde_json::json!({"id": e})).collect::<Vec<_>>()
        });

        let response = self
            .client
            .post("https://www.googleapis.com/calendar/v3/freeBusy")
            .bearer_auth(&account.access_token)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| CalendarError::ApiError(e.to_string()))?;

        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| CalendarError::ApiError(e.to_string()))?;

        let mut result = HashMap::new();

        if let Some(calendars) = data["calendars"].as_object() {
            for (email, cal_data) in calendars {
                let busy_slots: Vec<BusySlot> = cal_data["busy"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|slot| {
                        Some(BusySlot {
                            start: DateTime::parse_from_rfc3339(slot["start"].as_str()?)
                                .ok()?
                                .with_timezone(&Utc),
                            end: DateTime::parse_from_rfc3339(slot["end"].as_str()?)
                                .ok()?
                                .with_timezone(&Utc),
                        })
                    })
                    .collect();

                result.insert(email.clone(), busy_slots);
            }
        }

        Ok(result)
    }
}

impl GoogleCalendarSync {
    fn parse_google_event(&self, data: &serde_json::Value) -> Result<CalendarEvent, CalendarError> {
        let start = data["start"]["dateTime"]
            .as_str()
            .or_else(|| data["start"]["date"].as_str())
            .ok_or_else(|| CalendarError::ApiError("Missing start time".into()))?;

        let end = data["end"]["dateTime"]
            .as_str()
            .or_else(|| data["end"]["date"].as_str())
            .ok_or_else(|| CalendarError::ApiError("Missing end time".into()))?;

        let all_day = data["start"]["date"].is_string();

        let start_time = if all_day {
            chrono::NaiveDate::parse_from_str(start, "%Y-%m-%d")
                .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc())
                .map_err(|e| CalendarError::ApiError(e.to_string()))?
        } else {
            DateTime::parse_from_rfc3339(start)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|e| CalendarError::ApiError(e.to_string()))?
        };

        let end_time = if all_day {
            chrono::NaiveDate::parse_from_str(end, "%Y-%m-%d")
                .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc())
                .map_err(|e| CalendarError::ApiError(e.to_string()))?
        } else {
            DateTime::parse_from_rfc3339(end)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|e| CalendarError::ApiError(e.to_string()))?
        };

        let attendees: Vec<Attendee> = data["attendees"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|a| Attendee {
                email: a["email"].as_str().unwrap_or("").to_string(),
                name: a["displayName"].as_str().map(String::from),
                response_status: match a["responseStatus"].as_str() {
                    Some("accepted") => ResponseStatus::Accepted,
                    Some("declined") => ResponseStatus::Declined,
                    Some("tentative") => ResponseStatus::Tentative,
                    _ => ResponseStatus::NeedsAction,
                },
                optional: a["optional"].as_bool().unwrap_or(false),
                organizer: a["organizer"].as_bool().unwrap_or(false),
            })
            .collect();

        let organizer = data["organizer"].as_object().map(|o| Attendee {
            email: o["email"].as_ref().and_then(|v| v.as_str()).unwrap_or("").to_string(),
            name: o["displayName"].as_ref().and_then(|v| v.as_str()).map(String::from),
            response_status: ResponseStatus::Accepted,
            optional: false,
            organizer: true,
        }).unwrap_or(Attendee {
            email: String::new(),
            name: None,
            response_status: ResponseStatus::Accepted,
            optional: false,
            organizer: true,
        });

        let conference = data["conferenceData"]["entryPoints"]
            .as_array()
            .and_then(|eps| eps.iter().find(|ep| ep["entryPointType"] == "video"))
            .map(|ep| ConferenceInfo {
                provider: ConferenceProvider::GoogleMeet,
                url: ep["uri"].as_str().unwrap_or("").to_string(),
                meeting_id: data["conferenceData"]["conferenceId"].as_str().map(String::from),
                passcode: None,
                phone_numbers: vec![],
            });

        Ok(CalendarEvent {
            id: Uuid::new_v4().to_string(),
            external_id: data["id"].as_str().map(String::from),
            title: data["summary"].as_str().unwrap_or("(No title)").to_string(),
            description: data["description"].as_str().map(String::from),
            location: data["location"].as_str().map(String::from),
            start_time,
            end_time,
            all_day,
            timezone: data["start"]["timeZone"].as_str().unwrap_or("UTC").to_string(),
            attendees,
            organizer,
            status: match data["status"].as_str() {
                Some("cancelled") => EventStatus::Cancelled,
                Some("tentative") => EventStatus::Tentative,
                _ => EventStatus::Confirmed,
            },
            visibility: match data["visibility"].as_str() {
                Some("private") => EventVisibility::Private,
                Some("confidential") => EventVisibility::Confidential,
                _ => EventVisibility::Public,
            },
            recurrence: None, // Would parse RRULE
            reminders: vec![],
            conference,
            source_email_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            sync_status: SyncStatus {
                synced: true,
                last_synced: Some(Utc::now()),
                sync_error: None,
                local_changes: false,
                remote_changes: false,
            },
        })
    }

    fn to_google_event(&self, event: &CalendarEvent) -> serde_json::Value {
        let mut google_event = serde_json::json!({
            "summary": event.title,
            "start": if event.all_day {
                serde_json::json!({ "date": event.start_time.format("%Y-%m-%d").to_string() })
            } else {
                serde_json::json!({
                    "dateTime": event.start_time.to_rfc3339(),
                    "timeZone": &event.timezone
                })
            },
            "end": if event.all_day {
                serde_json::json!({ "date": event.end_time.format("%Y-%m-%d").to_string() })
            } else {
                serde_json::json!({
                    "dateTime": event.end_time.to_rfc3339(),
                    "timeZone": &event.timezone
                })
            },
            "attendees": event.attendees.iter().map(|a| {
                serde_json::json!({
                    "email": a.email,
                    "displayName": a.name,
                    "optional": a.optional
                })
            }).collect::<Vec<_>>()
        });

        if let Some(desc) = &event.description {
            google_event["description"] = serde_json::json!(desc);
        }

        if let Some(loc) = &event.location {
            google_event["location"] = serde_json::json!(loc);
        }

        google_event
    }
}

// ============================================================================
// Calendar Sync Manager
// ============================================================================

pub struct CalendarSyncManager {
    google_sync: GoogleCalendarSync,
    accounts: HashMap<String, CalendarAccount>,
}

impl CalendarSyncManager {
    pub fn new(google_client_id: String, google_client_secret: String) -> Self {
        Self {
            google_sync: GoogleCalendarSync::new(google_client_id, google_client_secret),
            accounts: HashMap::new(),
        }
    }

    pub async fn sync_all(&mut self) -> Vec<SyncResult> {
        let mut results = vec![];

        for (_, account) in self.accounts.iter() {
            if !account.sync_enabled {
                continue;
            }

            let sync = match account.provider {
                CalendarProvider::Google => &self.google_sync as &dyn CalendarSync,
                _ => continue,
            };

            for calendar in &account.calendars {
                if !calendar.selected {
                    continue;
                }

                let time_min = Utc::now() - Duration::days(30);
                let time_max = Utc::now() + Duration::days(365);

                match sync.sync_events(
                    account,
                    &calendar.id,
                    account.sync_token.as_deref(),
                    time_min,
                    time_max,
                ).await {
                    Ok(result) => results.push(result),
                    Err(e) => results.push(SyncResult {
                        success: false,
                        events_pulled: 0,
                        events_pushed: 0,
                        events_updated: 0,
                        events_deleted: 0,
                        conflicts: vec![],
                        errors: vec![e.to_string()],
                        next_sync_token: None,
                    }),
                }
            }
        }

        results
    }

    pub fn add_account(&mut self, account: CalendarAccount) {
        self.accounts.insert(account.id.clone(), account);
    }

    pub fn remove_account(&mut self, account_id: &str) {
        self.accounts.remove(account_id);
    }

    pub fn get_account(&self, account_id: &str) -> Option<&CalendarAccount> {
        self.accounts.get(account_id)
    }

    pub fn list_accounts(&self) -> Vec<&CalendarAccount> {
        self.accounts.values().collect()
    }
}
