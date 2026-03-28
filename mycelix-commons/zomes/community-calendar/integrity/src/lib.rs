// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Community Calendar Integrity Zome
//! Defines entry types and validation for calendar events, RSVPs, and anchors.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Recurrence pattern for calendar events
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Recurrence {
    None,
    Daily,
    Weekly,
    Monthly,
}

/// A community calendar event
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CalendarEvent {
    /// Event title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Physical location (optional)
    pub location: Option<commons_types::geo::GeoLocation>,
    /// Start time (UTC)
    pub start_time: Timestamp,
    /// End time (UTC)
    pub end_time: Timestamp,
    /// Recurrence pattern
    pub recurrence: Recurrence,
    /// DID of the organizer
    pub organizer_did: String,
    /// Event category (e.g., "meeting", "workshop", "celebration")
    pub category: String,
    /// Maximum number of attendees (0 = unlimited)
    pub max_attendees: u32,
    /// Whether RSVP is required to attend
    pub rsvp_required: bool,
}

/// RSVP status for an event
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RsvpStatus {
    Going,
    Maybe,
    NotGoing,
}

/// An RSVP response to a calendar event
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Rsvp {
    /// ActionHash of the CalendarEvent
    pub event_id: ActionHash,
    /// DID of the attendee
    pub attendee_did: String,
    /// RSVP status
    pub status: RsvpStatus,
    /// When the RSVP was submitted
    pub responded_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    CalendarEvent(CalendarEvent),
    Rsvp(Rsvp),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Anchor("all_events") → CalendarEvent
    AllEvents,
    /// Anchor("events_date:<YYYY-MM-DD>") → CalendarEvent
    EventsByDate,
    /// Anchor("events_cat:<category>") → CalendarEvent
    EventsByCategory,
    /// CalendarEvent → Rsvp
    EventToRsvp,
    /// Anchor("agent_rsvps:<agent>") → Rsvp
    AgentToRsvp,
    /// Anchor("agent_events:<agent>") → CalendarEvent (organized)
    AgentToEvent,
    /// Anchor("geo:<geohash>") → CalendarEvent
    GeoIndex,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// Validation
// ============================================================================

fn validate_create_event(
    _action: Create,
    event: CalendarEvent,
) -> ExternResult<ValidateCallbackResult> {
    // Title must be non-empty and bounded
    if event.title.trim().is_empty() || event.title.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Event title must be 1-512 non-whitespace characters".into(),
        ));
    }
    // Description bounded
    if event.description.len() > 8192 {
        return Ok(ValidateCallbackResult::Invalid(
            "Event description exceeds 8192 bytes".into(),
        ));
    }
    // Category bounded
    if event.category.trim().is_empty() || event.category.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Event category must be 1-128 non-whitespace characters".into(),
        ));
    }
    // Organizer DID bounded
    if event.organizer_did.trim().is_empty() || event.organizer_did.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "organizer_did must be 1-256 non-whitespace characters".into(),
        ));
    }
    // End time must be after start time
    if event.end_time <= event.start_time {
        return Ok(ValidateCallbackResult::Invalid(
            "end_time must be after start_time".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_rsvp(
    _action: Create,
    rsvp: Rsvp,
) -> ExternResult<ValidateCallbackResult> {
    if rsvp.attendee_did.trim().is_empty() || rsvp.attendee_did.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "attendee_did must be 1-256 non-whitespace characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CalendarEvent(event) => validate_create_event(action, event),
                EntryTypes::Rsvp(rsvp) => validate_create_rsvp(action, rsvp),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CalendarEvent(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Rsvp(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => {
            // Uniform tag length check for all link types
            let max_tag = match link_type {
                LinkTypes::EventsByCategory => 512,
                _ => 256,
            };
            if tag.0.len() > max_tag {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "Link tag too long (max {} bytes)",
                    max_tag
                )));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
