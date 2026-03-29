// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Scheduler Integrity Zome
//!
//! Entry types for scheduled and recurring emails.

use hdi::prelude::*;

/// Scheduled email entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ScheduledEmail {
    pub id: String,
    pub draft_hash: ActionHash,
    pub scheduled_for: u64,
    pub timezone: String,
    pub recurrence: Option<RecurrenceRule>,
    pub status: ScheduleStatus,
    pub retries: u32,
    pub last_error: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
    pub metadata: ScheduleMetadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RecurrenceRule {
    pub frequency: RecurrenceFrequency,
    pub interval: u32,
    pub days_of_week: Option<Vec<u8>>,
    pub day_of_month: Option<u8>,
    pub month_of_year: Option<u8>,
    pub end_date: Option<u64>,
    pub max_occurrences: Option<u32>,
    pub occurrence_count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum RecurrenceFrequency {
    Daily,
    Weekly,
    Monthly,
    Yearly,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ScheduleStatus {
    Pending,
    Processing,
    Sent,
    Failed,
    Cancelled,
    Paused,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ScheduleMetadata {
    pub subject: Option<String>,
    pub recipient_count: Option<u32>,
    pub smart_send_optimized: Option<bool>,
    pub original_scheduled_for: Option<u64>,
}

/// Snooze reminder entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SnoozeReminder {
    pub id: String,
    pub email_hash: ActionHash,
    pub remind_at: u64,
    pub created_at: u64,
    pub dismissed: bool,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    ScheduledEmail(ScheduledEmail),
    SnoozeReminder(SnoozeReminder),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllSchedules,
    PendingSchedules,
    RecurringSchedules,
    DraftToSchedule,
    SnoozeReminders,
}

/// Validate scheduled email
fn validate_create_scheduled_email(
    _action: Create,
    scheduled: ScheduledEmail,
) -> ExternResult<ValidateCallbackResult> {
    // Validate ID
    if scheduled.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Schedule ID cannot be empty".to_string(),
        ));
    }

    // Note: Time-based validation moved to coordinator zome since sys_time
    // is not available in integrity zomes

    // Validate recurrence if present
    if let Some(ref recurrence) = scheduled.recurrence {
        if recurrence.interval == 0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Recurrence interval must be at least 1".to_string(),
            ));
        }

        if let Some(max) = recurrence.max_occurrences {
            if max == 0 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Max occurrences must be at least 1".to_string(),
                ));
            }
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate snooze reminder
fn validate_create_snooze_reminder(
    _action: Create,
    reminder: SnoozeReminder,
) -> ExternResult<ValidateCallbackResult> {
    if reminder.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Reminder ID cannot be empty".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Main validation dispatcher
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::ScheduledEmail(scheduled) => {
                    validate_create_scheduled_email(action, scheduled)
                }
                EntryTypes::SnoozeReminder(reminder) => {
                    validate_create_snooze_reminder(action, reminder)
                }
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::ScheduledEmail(scheduled) => {
                    // Validate status transitions
                    match scheduled.status {
                        ScheduleStatus::Sent => {
                            // Cannot modify sent schedules
                            if scheduled.recurrence.is_none() {
                                return Ok(ValidateCallbackResult::Invalid(
                                    "Cannot modify sent non-recurring schedule".to_string(),
                                ));
                            }
                        }
                        _ => {}
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}
