// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Scheduler Coordinator Zome
//!
//! Schedule emails for later delivery.

use hdk::prelude::*;
use scheduler_integrity::*;

const ALL_SCHEDULES_ANCHOR: &str = "all_schedules";
const PENDING_ANCHOR: &str = "pending_schedules";
const RECURRING_ANCHOR: &str = "recurring_schedules";
const SNOOZE_ANCHOR: &str = "snooze_reminders";

// ==================== SCHEDULE CRUD ====================

/// Create a scheduled email
#[hdk_extern]
pub fn schedule_email(scheduled: ScheduledEmail) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::ScheduledEmail(scheduled.clone()))?;

    // Link to all schedules
    let all_anchor = anchor_hash(ALL_SCHEDULES_ANCHOR)?;
    create_link(
        all_anchor,
        action_hash.clone(),
        LinkTypes::AllSchedules,
        (),
    )?;

    // Link to pending if status is pending
    if matches!(scheduled.status, ScheduleStatus::Pending) {
        let pending_anchor = anchor_hash(PENDING_ANCHOR)?;
        create_link(
            pending_anchor,
            action_hash.clone(),
            LinkTypes::PendingSchedules,
            scheduled.scheduled_for.to_be_bytes().to_vec(),
        )?;
    }

    // Link to recurring if has recurrence
    if scheduled.recurrence.is_some() {
        let recurring_anchor = anchor_hash(RECURRING_ANCHOR)?;
        create_link(
            recurring_anchor,
            action_hash.clone(),
            LinkTypes::RecurringSchedules,
            (),
        )?;
    }

    // Link from draft
    create_link(
        scheduled.draft_hash,
        action_hash.clone(),
        LinkTypes::DraftToSchedule,
        (),
    )?;

    Ok(action_hash)
}

/// Get scheduled email by hash
#[hdk_extern]
pub fn get_schedule(hash: ActionHash) -> ExternResult<Option<ScheduledEmail>> {
    match get(hash, GetOptions::default())? {
        Some(record) => {
            let scheduled: ScheduledEmail = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(e))?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Schedule not found".to_string()
                )))?;
            Ok(Some(scheduled))
        }
        None => Ok(None),
    }
}

/// Update a schedule
#[hdk_extern]
pub fn update_schedule(scheduled: ScheduledEmail) -> ExternResult<ActionHash> {
    // Find existing schedule by ID
    let all_anchor = anchor_hash(ALL_SCHEDULES_ANCHOR)?;
    let links = get_links(LinkQuery::try_new(all_anchor, LinkTypes::AllSchedules)?, GetStrategy::default())?;

    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(existing) = get_schedule(hash.clone())? {
                if existing.id == scheduled.id {
                    // Update the entry
                    let new_hash = update_entry(hash.clone(), EntryTypes::ScheduledEmail(scheduled.clone()))?;

                    // Update pending links if status changed
                    update_status_links(&scheduled, &hash, &new_hash)?;

                    return Ok(new_hash);
                }
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(
        "Schedule not found".to_string()
    )))
}

/// Cancel a schedule
#[hdk_extern]
pub fn cancel_schedule(hash: ActionHash) -> ExternResult<ActionHash> {
    if let Some(mut scheduled) = get_schedule(hash.clone())? {
        scheduled.status = ScheduleStatus::Cancelled;
        scheduled.updated_at = sys_time()?.as_micros() as u64;

        let new_hash = update_entry(hash.clone(), EntryTypes::ScheduledEmail(scheduled.clone()))?;

        // Remove from pending links
        remove_pending_link(&hash)?;

        Ok(new_hash)
    } else {
        Err(wasm_error!(WasmErrorInner::Guest(
            "Schedule not found".to_string()
        )))
    }
}

/// Get all schedules
#[hdk_extern]
pub fn get_all_schedules(_: ()) -> ExternResult<Vec<ScheduledEmail>> {
    let anchor = anchor_hash(ALL_SCHEDULES_ANCHOR)?;
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::AllSchedules)?, GetStrategy::default())?;

    let mut schedules = Vec::new();
    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(scheduled) = get_schedule(hash)? {
                schedules.push(scheduled);
            }
        }
    }

    // Sort by scheduled time
    schedules.sort_by(|a, b| a.scheduled_for.cmp(&b.scheduled_for));

    Ok(schedules)
}

/// Get pending schedules (ready to send)
#[hdk_extern]
pub fn get_pending_schedules(_: ()) -> ExternResult<Vec<ScheduledEmail>> {
    let anchor = anchor_hash(PENDING_ANCHOR)?;
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::PendingSchedules)?, GetStrategy::default())?;

    let now = sys_time()?.as_micros() as u64;
    let mut schedules = Vec::new();

    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(scheduled) = get_schedule(hash)? {
                // Only include if due
                if scheduled.scheduled_for <= now {
                    schedules.push(scheduled);
                }
            }
        }
    }

    // Sort by scheduled time (oldest first)
    schedules.sort_by(|a, b| a.scheduled_for.cmp(&b.scheduled_for));

    Ok(schedules)
}

/// Get recurring schedules
#[hdk_extern]
pub fn get_recurring_schedules(_: ()) -> ExternResult<Vec<ScheduledEmail>> {
    let anchor = anchor_hash(RECURRING_ANCHOR)?;
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::RecurringSchedules)?, GetStrategy::default())?;

    let mut schedules = Vec::new();
    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(scheduled) = get_schedule(hash)? {
                if scheduled.recurrence.is_some() {
                    schedules.push(scheduled);
                }
            }
        }
    }

    Ok(schedules)
}

/// Get schedule for draft
#[hdk_extern]
pub fn get_schedule_for_draft(draft_hash: ActionHash) -> ExternResult<Option<ScheduledEmail>> {
    let links = get_links(LinkQuery::try_new(draft_hash, LinkTypes::DraftToSchedule)?, GetStrategy::default())?;

    if let Some(link) = links.first() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            return get_schedule(hash);
        }
    }

    Ok(None)
}

// ==================== SNOOZE ====================

/// Create snooze reminder
#[hdk_extern]
pub fn snooze_email(input: SnoozeInput) -> ExternResult<ActionHash> {
    let reminder = SnoozeReminder {
        id: format!("snooze_{}", sys_time()?.as_micros()),
        email_hash: input.email_hash,
        remind_at: input.remind_at,
        created_at: sys_time()?.as_micros() as u64,
        dismissed: false,
    };

    let action_hash = create_entry(EntryTypes::SnoozeReminder(reminder))?;

    let anchor = anchor_hash(SNOOZE_ANCHOR)?;
    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::SnoozeReminders,
        input.remind_at.to_be_bytes().to_vec(),
    )?;

    Ok(action_hash)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnoozeInput {
    pub email_hash: ActionHash,
    pub remind_at: u64,
}

/// Get due reminders
#[hdk_extern]
pub fn get_due_reminders(_: ()) -> ExternResult<Vec<SnoozeReminder>> {
    let anchor = anchor_hash(SNOOZE_ANCHOR)?;
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::SnoozeReminders)?, GetStrategy::default())?;

    let now = sys_time()?.as_micros() as u64;
    let mut reminders = Vec::new();

    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(reminder) = record
                    .entry()
                    .to_app_option::<SnoozeReminder>()
                    .map_err(|e| wasm_error!(e))?
                {
                    if reminder.remind_at <= now && !reminder.dismissed {
                        reminders.push(reminder);
                    }
                }
            }
        }
    }

    Ok(reminders)
}

/// Dismiss reminder
#[hdk_extern]
pub fn dismiss_reminder(hash: ActionHash) -> ExternResult<ActionHash> {
    if let Some(record) = get(hash.clone(), GetOptions::default())? {
        if let Some(mut reminder) = record
            .entry()
            .to_app_option::<SnoozeReminder>()
            .map_err(|e| wasm_error!(e))?
        {
            reminder.dismissed = true;
            return update_entry(hash, EntryTypes::SnoozeReminder(reminder));
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(
        "Reminder not found".to_string()
    )))
}

// ==================== HELPERS ====================

fn anchor_hash(name: &str) -> ExternResult<EntryHash> {
    let path = Path::from(name);
    path.path_entry_hash()
}

fn update_status_links(
    scheduled: &ScheduledEmail,
    old_hash: &ActionHash,
    new_hash: &ActionHash,
) -> ExternResult<()> {
    let pending_anchor = anchor_hash(PENDING_ANCHOR)?;

    // Remove old pending link
    remove_pending_link(old_hash)?;

    // Add new pending link if still pending
    if matches!(scheduled.status, ScheduleStatus::Pending) {
        create_link(
            pending_anchor,
            new_hash.clone(),
            LinkTypes::PendingSchedules,
            scheduled.scheduled_for.to_be_bytes().to_vec(),
        )?;
    }

    Ok(())
}

fn remove_pending_link(hash: &ActionHash) -> ExternResult<()> {
    let pending_anchor = anchor_hash(PENDING_ANCHOR)?;
    let links = get_links(LinkQuery::try_new(pending_anchor, LinkTypes::PendingSchedules)?, GetStrategy::default())?;

    for link in links {
        if let Some(target_hash) = link.target.clone().into_action_hash() {
            if &target_hash == hash {
                delete_link(link.create_link_hash, GetOptions::default())?;
                break;
            }
        }
    }

    Ok(())
}
