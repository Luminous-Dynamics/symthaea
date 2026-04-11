// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Meeting Scheduling & Availability
//!
//! Find available times, schedule meetings, and share availability

use chrono::{DateTime, Duration, NaiveTime, Utc, Weekday, Datelike, Timelike};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

// ============================================================================
// Scheduling Service
// ============================================================================

pub struct SchedulingService {
    pool: PgPool,
}

impl SchedulingService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get user's availability for a date range
    pub async fn get_availability(
        &self,
        user_id: Uuid,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Availability, SchedulingError> {
        // Get user's working hours preferences
        let prefs = self.get_scheduling_preferences(user_id).await?;

        // Get existing events
        let events: Vec<EventSlot> = sqlx::query_as(
            r#"
            SELECT start_time, end_time, summary, status::text as status
            FROM calendar_events
            WHERE user_id = $1
              AND start_time < $3
              AND end_time > $2
              AND status != 'cancelled'
            ORDER BY start_time
            "#,
        )
        .bind(user_id)
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| SchedulingError::Database(e.to_string()))?;

        // Build busy slots
        let busy_slots: Vec<TimeSlot> = events
            .iter()
            .map(|e| TimeSlot {
                start: e.start_time,
                end: e.end_time,
                status: if e.status == "tentative" {
                    SlotStatus::Tentative
                } else {
                    SlotStatus::Busy
                },
            })
            .collect();

        // Calculate free slots within working hours
        let free_slots = self.calculate_free_slots(start, end, &busy_slots, &prefs);

        Ok(Availability {
            user_id,
            range_start: start,
            range_end: end,
            busy_slots,
            free_slots,
            timezone: prefs.timezone.clone(),
        })
    }

    /// Find common availability among multiple participants
    pub async fn find_common_availability(
        &self,
        participants: Vec<Uuid>,
        duration_minutes: i64,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        options: FindOptions,
    ) -> Result<Vec<ProposedSlot>, SchedulingError> {
        if participants.is_empty() {
            return Err(SchedulingError::NoParticipants);
        }

        // Get availability for all participants
        let mut all_busy: Vec<TimeSlot> = Vec::new();
        for user_id in &participants {
            let availability = self.get_availability(*user_id, start, end).await?;
            all_busy.extend(availability.busy_slots);
        }

        // Sort and merge overlapping busy periods
        all_busy.sort_by_key(|s| s.start);
        let merged_busy = self.merge_overlapping_slots(&all_busy);

        // Get first participant's preferences for working hours
        let prefs = self.get_scheduling_preferences(participants[0]).await?;

        // Find gaps that fit the duration
        let duration = Duration::minutes(duration_minutes);
        let mut proposed_slots = Vec::new();

        let free_slots = self.calculate_free_slots(start, end, &merged_busy, &prefs);

        for slot in free_slots {
            if slot.end - slot.start >= duration {
                let score = self.score_slot(&slot, &prefs, &options);
                proposed_slots.push(ProposedSlot {
                    start: slot.start,
                    end: slot.start + duration,
                    score,
                    conflicts: 0,
                });
            }
        }

        // Sort by score (best first)
        proposed_slots.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Limit results
        proposed_slots.truncate(options.max_results.unwrap_or(10) as usize);

        Ok(proposed_slots)
    }

    /// Create a scheduling link
    pub async fn create_scheduling_link(
        &self,
        user_id: Uuid,
        config: SchedulingLinkConfig,
    ) -> Result<SchedulingLink, SchedulingError> {
        let link_id = Uuid::new_v4();
        let slug = generate_slug();

        sqlx::query(
            r#"
            INSERT INTO scheduling_links (
                id, user_id, slug, title, duration_minutes, buffer_minutes,
                availability_window_days, available_hours_start, available_hours_end,
                available_days, max_bookings_per_day, requires_confirmation,
                questions, created_at, expires_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, NOW(), $14)
            "#,
        )
        .bind(link_id)
        .bind(user_id)
        .bind(&slug)
        .bind(&config.title)
        .bind(config.duration_minutes)
        .bind(config.buffer_minutes)
        .bind(config.availability_window_days)
        .bind(config.available_hours_start)
        .bind(config.available_hours_end)
        .bind(&config.available_days)
        .bind(config.max_bookings_per_day)
        .bind(config.requires_confirmation)
        .bind(serde_json::to_value(&config.questions).unwrap_or_default())
        .bind(config.expires_at)
        .execute(&self.pool)
        .await
        .map_err(|e| SchedulingError::Database(e.to_string()))?;

        Ok(SchedulingLink {
            id: link_id,
            user_id,
            slug,
            title: config.title,
            url: format!("/schedule/{}", slug),
            duration_minutes: config.duration_minutes,
            created_at: Utc::now(),
            expires_at: config.expires_at,
        })
    }

    /// Get available slots for a scheduling link
    pub async fn get_link_availability(
        &self,
        slug: &str,
        date: DateTime<Utc>,
    ) -> Result<Vec<AvailableTime>, SchedulingError> {
        // Get link configuration
        let link: SchedulingLinkRow = sqlx::query_as(
            "SELECT * FROM scheduling_links WHERE slug = $1 AND (expires_at IS NULL OR expires_at > NOW())",
        )
        .bind(slug)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| SchedulingError::Database(e.to_string()))?
        .ok_or(SchedulingError::LinkNotFound)?;

        let start = date.date_naive().and_hms_opt(0, 0, 0).unwrap();
        let end = date.date_naive().and_hms_opt(23, 59, 59).unwrap();

        let start_utc = DateTime::from_naive_utc_and_offset(start, Utc);
        let end_utc = DateTime::from_naive_utc_and_offset(end, Utc);

        // Check if day is available
        let weekday = date.weekday().num_days_from_monday() as i32;
        if !link.available_days.contains(&weekday) {
            return Ok(vec![]);
        }

        // Get user's events for that day
        let events: Vec<EventSlot> = sqlx::query_as(
            r#"
            SELECT start_time, end_time, summary, status::text as status
            FROM calendar_events
            WHERE user_id = $1
              AND start_time < $3
              AND end_time > $2
              AND status != 'cancelled'
            "#,
        )
        .bind(link.user_id)
        .bind(start_utc)
        .bind(end_utc)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| SchedulingError::Database(e.to_string()))?;

        // Check existing bookings for max per day
        let existing_bookings: (i64,) = sqlx::query_as(
            r#"
            SELECT COUNT(*) FROM scheduled_meetings
            WHERE scheduling_link_id = $1
              AND start_time >= $2
              AND start_time < $3
              AND status != 'cancelled'
            "#,
        )
        .bind(link.id)
        .bind(start_utc)
        .bind(end_utc)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| SchedulingError::Database(e.to_string()))?;

        if let Some(max) = link.max_bookings_per_day {
            if existing_bookings.0 >= max as i64 {
                return Ok(vec![]);
            }
        }

        // Generate available time slots
        let mut slots = Vec::new();
        let duration = Duration::minutes(link.duration_minutes as i64);
        let buffer = Duration::minutes(link.buffer_minutes.unwrap_or(0) as i64);

        let day_start = start_utc + Duration::hours(link.available_hours_start as i64);
        let day_end = start_utc + Duration::hours(link.available_hours_end as i64);

        let mut current = day_start;
        while current + duration <= day_end {
            let slot_end = current + duration;

            // Check if slot conflicts with any event (including buffer)
            let slot_start_with_buffer = current - buffer;
            let slot_end_with_buffer = slot_end + buffer;

            let conflicts = events.iter().any(|e| {
                e.start_time < slot_end_with_buffer && e.end_time > slot_start_with_buffer
            });

            if !conflicts && current > Utc::now() {
                slots.push(AvailableTime {
                    start: current,
                    end: slot_end,
                });
            }

            // Move to next slot (with buffer)
            current = slot_end + buffer;
        }

        Ok(slots)
    }

    /// Book a meeting via scheduling link
    pub async fn book_meeting(
        &self,
        slug: &str,
        booking: BookingRequest,
    ) -> Result<ScheduledMeeting, SchedulingError> {
        // Get link
        let link: SchedulingLinkRow = sqlx::query_as(
            "SELECT * FROM scheduling_links WHERE slug = $1",
        )
        .bind(slug)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| SchedulingError::Database(e.to_string()))?
        .ok_or(SchedulingError::LinkNotFound)?;

        // Verify slot is still available
        let date = booking.start_time;
        let available = self.get_link_availability(slug, date).await?;

        if !available.iter().any(|s| s.start == booking.start_time) {
            return Err(SchedulingError::SlotNotAvailable);
        }

        // Create meeting
        let meeting_id = Uuid::new_v4();
        let end_time = booking.start_time + Duration::minutes(link.duration_minutes as i64);

        sqlx::query(
            r#"
            INSERT INTO scheduled_meetings (
                id, scheduling_link_id, host_user_id, guest_name, guest_email,
                start_time, end_time, status, answers, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
            "#,
        )
        .bind(meeting_id)
        .bind(link.id)
        .bind(link.user_id)
        .bind(&booking.guest_name)
        .bind(&booking.guest_email)
        .bind(booking.start_time)
        .bind(end_time)
        .bind(if link.requires_confirmation { "pending" } else { "confirmed" })
        .bind(serde_json::to_value(&booking.answers).unwrap_or_default())
        .execute(&self.pool)
        .await
        .map_err(|e| SchedulingError::Database(e.to_string()))?;

        // Create calendar event for host
        sqlx::query(
            r#"
            INSERT INTO calendar_events (
                id, user_id, uid, summary, description, start_time, end_time,
                status, source, created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, 'confirmed', 'manual', NOW(), NOW())
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(link.user_id)
        .bind(format!("meeting-{}", meeting_id))
        .bind(format!("Meeting with {}", booking.guest_name))
        .bind(format!("Booked via scheduling link: {}", link.title))
        .bind(booking.start_time)
        .bind(end_time)
        .execute(&self.pool)
        .await
        .map_err(|e| SchedulingError::Database(e.to_string()))?;

        // Send confirmation emails
        self.send_booking_confirmation(meeting_id, &link, &booking).await?;

        Ok(ScheduledMeeting {
            id: meeting_id,
            title: link.title,
            start_time: booking.start_time,
            end_time,
            guest_name: booking.guest_name,
            guest_email: booking.guest_email,
            status: if link.requires_confirmation {
                MeetingStatus::Pending
            } else {
                MeetingStatus::Confirmed
            },
        })
    }

    // ========================================================================
    // Private Helpers
    // ========================================================================

    async fn get_scheduling_preferences(
        &self,
        user_id: Uuid,
    ) -> Result<SchedulingPreferences, SchedulingError> {
        let prefs: Option<PreferencesRow> = sqlx::query_as(
            r#"
            SELECT working_hours_start, working_hours_end, working_days, timezone
            FROM scheduling_preferences
            WHERE user_id = $1
            "#,
        )
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| SchedulingError::Database(e.to_string()))?;

        Ok(prefs.map(|p| SchedulingPreferences {
            working_hours_start: p.working_hours_start as u8,
            working_hours_end: p.working_hours_end as u8,
            working_days: p.working_days.into_iter().map(|d| num_to_weekday(d)).collect(),
            timezone: p.timezone,
        }).unwrap_or_default())
    }

    fn calculate_free_slots(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        busy: &[TimeSlot],
        prefs: &SchedulingPreferences,
    ) -> Vec<TimeSlot> {
        let mut free = Vec::new();
        let mut current = start;

        for busy_slot in busy {
            if busy_slot.start > current {
                // There's a gap - check if within working hours
                let gap_start = current;
                let gap_end = busy_slot.start;

                if let Some(working_slot) = self.intersect_with_working_hours(gap_start, gap_end, prefs) {
                    free.push(working_slot);
                }
            }
            current = busy_slot.end.max(current);
        }

        // Check remaining time after last busy slot
        if current < end {
            if let Some(working_slot) = self.intersect_with_working_hours(current, end, prefs) {
                free.push(working_slot);
            }
        }

        free
    }

    fn intersect_with_working_hours(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        prefs: &SchedulingPreferences,
    ) -> Option<TimeSlot> {
        // Check if within working days
        if !prefs.working_days.contains(&start.weekday()) {
            return None;
        }

        // Clamp to working hours
        let day_start = start.date_naive().and_hms_opt(prefs.working_hours_start as u32, 0, 0)?;
        let day_end = start.date_naive().and_hms_opt(prefs.working_hours_end as u32, 0, 0)?;

        let work_start = DateTime::from_naive_utc_and_offset(day_start, Utc);
        let work_end = DateTime::from_naive_utc_and_offset(day_end, Utc);

        let clamped_start = start.max(work_start);
        let clamped_end = end.min(work_end);

        if clamped_start < clamped_end {
            Some(TimeSlot {
                start: clamped_start,
                end: clamped_end,
                status: SlotStatus::Free,
            })
        } else {
            None
        }
    }

    fn merge_overlapping_slots(&self, slots: &[TimeSlot]) -> Vec<TimeSlot> {
        if slots.is_empty() {
            return vec![];
        }

        let mut merged = vec![slots[0].clone()];

        for slot in &slots[1..] {
            let last = merged.last_mut().unwrap();
            if slot.start <= last.end {
                last.end = last.end.max(slot.end);
            } else {
                merged.push(slot.clone());
            }
        }

        merged
    }

    fn score_slot(&self, slot: &TimeSlot, prefs: &SchedulingPreferences, options: &FindOptions) -> f64 {
        let mut score = 1.0;

        let hour = slot.start.hour();

        // Prefer morning if requested
        if options.prefer_morning.unwrap_or(false) && hour >= 9 && hour <= 11 {
            score += 0.3;
        }

        // Avoid lunch time
        if hour == 12 || hour == 13 {
            score -= 0.2;
        }

        // Prefer early in the week
        let day = slot.start.weekday().num_days_from_monday();
        if day < 3 {
            score += 0.1;
        }

        score.max(0.0)
    }

    async fn send_booking_confirmation(
        &self,
        _meeting_id: Uuid,
        _link: &SchedulingLinkRow,
        _booking: &BookingRequest,
    ) -> Result<(), SchedulingError> {
        // Would send confirmation emails to both host and guest
        Ok(())
    }
}

fn generate_slug() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let chars: Vec<char> = "abcdefghijklmnopqrstuvwxyz0123456789".chars().collect();
    (0..8).map(|_| chars[rng.gen_range(0..chars.len())]).collect()
}

fn num_to_weekday(n: i32) -> Weekday {
    match n {
        0 => Weekday::Mon,
        1 => Weekday::Tue,
        2 => Weekday::Wed,
        3 => Weekday::Thu,
        4 => Weekday::Fri,
        5 => Weekday::Sat,
        _ => Weekday::Sun,
    }
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Availability {
    pub user_id: Uuid,
    pub range_start: DateTime<Utc>,
    pub range_end: DateTime<Utc>,
    pub busy_slots: Vec<TimeSlot>,
    pub free_slots: Vec<TimeSlot>,
    pub timezone: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSlot {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub status: SlotStatus,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SlotStatus {
    Free,
    Busy,
    Tentative,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedSlot {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub score: f64,
    pub conflicts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FindOptions {
    pub prefer_morning: Option<bool>,
    pub prefer_afternoon: Option<bool>,
    pub min_notice_hours: Option<i32>,
    pub max_results: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingPreferences {
    pub working_hours_start: u8,
    pub working_hours_end: u8,
    pub working_days: Vec<Weekday>,
    pub timezone: Option<String>,
}

impl Default for SchedulingPreferences {
    fn default() -> Self {
        Self {
            working_hours_start: 9,
            working_hours_end: 17,
            working_days: vec![
                Weekday::Mon, Weekday::Tue, Weekday::Wed,
                Weekday::Thu, Weekday::Fri,
            ],
            timezone: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingLinkConfig {
    pub title: String,
    pub duration_minutes: i32,
    pub buffer_minutes: Option<i32>,
    pub availability_window_days: i32,
    pub available_hours_start: i32,
    pub available_hours_end: i32,
    pub available_days: Vec<i32>,
    pub max_bookings_per_day: Option<i32>,
    pub requires_confirmation: bool,
    pub questions: Vec<SchedulingQuestion>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingQuestion {
    pub id: String,
    pub question: String,
    pub required: bool,
    pub question_type: QuestionType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuestionType {
    Text,
    TextArea,
    Select(Vec<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingLink {
    pub id: Uuid,
    pub user_id: Uuid,
    pub slug: String,
    pub title: String,
    pub url: String,
    pub duration_minutes: i32,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableTime {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookingRequest {
    pub guest_name: String,
    pub guest_email: String,
    pub start_time: DateTime<Utc>,
    pub answers: Vec<QuestionAnswer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionAnswer {
    pub question_id: String,
    pub answer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledMeeting {
    pub id: Uuid,
    pub title: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub guest_name: String,
    pub guest_email: String,
    pub status: MeetingStatus,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MeetingStatus {
    Pending,
    Confirmed,
    Cancelled,
    Completed,
}

#[derive(Debug, sqlx::FromRow)]
struct EventSlot {
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    summary: String,
    status: String,
}

#[derive(Debug, sqlx::FromRow)]
struct PreferencesRow {
    working_hours_start: i32,
    working_hours_end: i32,
    working_days: Vec<i32>,
    timezone: Option<String>,
}

#[derive(Debug, sqlx::FromRow)]
struct SchedulingLinkRow {
    id: Uuid,
    user_id: Uuid,
    slug: String,
    title: String,
    duration_minutes: i32,
    buffer_minutes: Option<i32>,
    availability_window_days: i32,
    available_hours_start: i32,
    available_hours_end: i32,
    available_days: Vec<i32>,
    max_bookings_per_day: Option<i32>,
    requires_confirmation: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum SchedulingError {
    #[error("Database error: {0}")]
    Database(String),

    #[error("No participants provided")]
    NoParticipants,

    #[error("Scheduling link not found")]
    LinkNotFound,

    #[error("Slot is no longer available")]
    SlotNotAvailable,
}
