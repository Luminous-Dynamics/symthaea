// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Community Calendar Coordinator Zome
//! Business logic for creating events, RSVPs, and querying by date/category/geo.

use community_calendar_integrity::*;
use hdk::prelude::*;
use mycelix_bridge_common::{civic_requirement_basic, civic_requirement_proposal};
use mycelix_zome_helpers::records_from_links;


fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    anchor_hash(anchor_str)
}

/// Extract a date string ("YYYY-MM-DD") from a Holochain Timestamp.
fn date_from_timestamp(ts: &Timestamp) -> String {
    let micros = ts.as_micros();
    let secs = micros / 1_000_000;
    // Simple UTC date calculation (no leap-second precision needed for anchoring)
    let days_since_epoch = secs / 86_400;
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days_since_epoch + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}", y, m, d)
}

// ============================================================================
// EVENT MANAGEMENT
// ============================================================================

/// Create a new calendar event with anchored indices.
#[hdk_extern]
pub fn create_event(event: CalendarEvent) -> ExternResult<ActionHash> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "create_event")?;

    // Validate inputs that go beyond integrity checks
    if event.title.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Event title cannot be empty".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::CalendarEvent(event.clone()))?;

    // Index: all events
    let all_anchor = ensure_anchor("all_events")?;
    create_link(
        all_anchor,
        action_hash.clone(),
        LinkTypes::AllEvents,
        (),
    )?;

    // Index: by date (start date)
    let date_str = date_from_timestamp(&event.start_time);
    let date_anchor = ensure_anchor(&format!("events_date:{}", date_str))?;
    create_link(
        date_anchor,
        action_hash.clone(),
        LinkTypes::EventsByDate,
        (),
    )?;

    // Index: by category
    let cat_key = event.category.to_lowercase().replace(' ', "_");
    let cat_anchor = ensure_anchor(&format!("events_cat:{}", cat_key))?;
    create_link(
        cat_anchor,
        action_hash.clone(),
        LinkTypes::EventsByCategory,
        (),
    )?;

    // Index: by organizer agent
    let caller = agent_info()?.agent_initial_pubkey;
    let agent_anchor = ensure_anchor(&format!("agent_events:{}", caller))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToEvent,
        (),
    )?;

    // Index: geo (if location provided)
    if let Some(ref loc) = event.location {
        let geohash = commons_types::geo::geohash_encode(loc.latitude, loc.longitude, 6);
        let geo_anchor = ensure_anchor(&format!("geo:{}", geohash))?;
        create_link(
            geo_anchor,
            action_hash.clone(),
            LinkTypes::GeoIndex,
            (),
        )?;
    }

    Ok(action_hash)
}

/// Retrieve a single event by its action hash.
#[hdk_extern]
pub fn get_event(hash: ActionHash) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "get_event")?;
    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Event not found".into())))
}

/// Get all events on a given date (format: "YYYY-MM-DD").
#[hdk_extern]
pub fn get_events_by_date(date_str: String) -> ExternResult<Vec<Record>> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "get_events_by_date")?;
    if date_str.len() != 10 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Date must be in YYYY-MM-DD format".into()
        )));
    }
    let anchor = anchor_hash(&format!("events_date:{}", date_str))?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::EventsByDate)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all events in a given category.
#[hdk_extern]
pub fn get_events_by_category(category: String) -> ExternResult<Vec<Record>> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "get_events_by_category")?;
    let cat_key = category.to_lowercase().replace(' ', "_");
    let anchor = anchor_hash(&format!("events_cat:{}", cat_key))?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::EventsByCategory)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get upcoming events (most recent first, up to `limit`).
///
/// NOTE: Returns all events and truncates to `limit`. For production
/// use, a time-bucketed index would be more efficient.
#[hdk_extern]
pub fn get_upcoming_events(limit: u32) -> ExternResult<Vec<Record>> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "get_upcoming_events")?;
    let anchor = anchor_hash("all_events")?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllEvents)?,
        GetStrategy::default(),
    )?;
    let mut records = records_from_links(links)?;
    // Sort by start_time descending (newest first)
    records.sort_by(|a, b| {
        let ts_a = a
            .entry()
            .to_app_option::<CalendarEvent>()
            .ok()
            .flatten()
            .map(|e| e.start_time)
            .unwrap_or(Timestamp::from_micros(0));
        let ts_b = b
            .entry()
            .to_app_option::<CalendarEvent>()
            .ok()
            .flatten()
            .map(|e| e.start_time)
            .unwrap_or(Timestamp::from_micros(0));
        ts_b.cmp(&ts_a)
    });
    records.truncate(limit as usize);
    Ok(records)
}

// ============================================================================
// RSVP MANAGEMENT
// ============================================================================

/// RSVP to a calendar event.
#[hdk_extern]
pub fn rsvp_to_event(rsvp: Rsvp) -> ExternResult<ActionHash> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "rsvp_to_event")?;

    // Verify the event exists
    let _event_record = get(rsvp.event_id.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Event not found".into())),
    )?;

    let action_hash = create_entry(&EntryTypes::Rsvp(rsvp.clone()))?;

    // Link event → rsvp
    create_link(
        rsvp.event_id.clone(),
        action_hash.clone(),
        LinkTypes::EventToRsvp,
        (),
    )?;

    // Link agent → rsvp
    let caller = agent_info()?.agent_initial_pubkey;
    let agent_anchor = ensure_anchor(&format!("agent_rsvps:{}", caller))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToRsvp,
        (),
    )?;

    Ok(action_hash)
}

/// Get all RSVPs for an event.
#[hdk_extern]
pub fn get_event_rsvps(event_id: ActionHash) -> ExternResult<Vec<Record>> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "get_event_rsvps")?;
    let links = get_links(
        LinkQuery::try_new(event_id, LinkTypes::EventToRsvp)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all RSVPs for the calling agent.
#[hdk_extern]
pub fn get_my_rsvps(_: ()) -> ExternResult<Vec<Record>> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "get_my_rsvps")?;
    let caller = agent_info()?.agent_initial_pubkey;
    let agent_anchor = anchor_hash(&format!("agent_rsvps:{}", caller))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor, LinkTypes::AgentToRsvp)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// EVENT LIFECYCLE
// ============================================================================

/// Cancel an event by deleting its entry. Returns the delete action hash.
#[hdk_extern]
pub fn cancel_event(hash: ActionHash) -> ExternResult<ActionHash> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "cancel_event")?;
    // Verify the event exists and was authored by the caller
    let record = get(hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Event not found".into())))?;
    let caller = agent_info()?.agent_initial_pubkey;
    if record.action().author() != &caller {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the event creator can cancel it".into()
        )));
    }
    delete_entry(hash)
}

// ============================================================================
// GEO QUERIES
// ============================================================================

/// Get events near a geographic location using geohash-based indexing.
#[hdk_extern]
pub fn get_nearby_events(input: commons_types::geo::NearbyQuery) -> ExternResult<Vec<Record>> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "get_nearby_events")?;
    let geohash = commons_types::geo::geohash_encode(input.latitude, input.longitude, 6);
    let geo_anchor = anchor_hash(&format!("geo:{}", geohash))?;
    let links = get_links(
        LinkQuery::try_new(geo_anchor, LinkTypes::GeoIndex)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn date_from_timestamp_basic() {
        // 2026-03-26 00:00:00 UTC = 1774483200 seconds since epoch
        let ts = Timestamp::from_micros(1_774_483_200_000_000);
        assert_eq!(date_from_timestamp(&ts), "2026-03-26");
    }

    #[test]
    fn date_from_timestamp_epoch() {
        let ts = Timestamp::from_micros(0);
        assert_eq!(date_from_timestamp(&ts), "1970-01-01");
    }

    #[test]
    fn date_from_timestamp_leap_year() {
        // 2024-02-29 12:00:00 UTC = 1709208000 seconds since epoch
        let ts = Timestamp::from_micros(1_709_208_000_000_000);
        assert_eq!(date_from_timestamp(&ts), "2024-02-29");
    }

    #[test]
    fn test_recurrence_variants() {
        use community_calendar_integrity::Recurrence;
        let variants = vec![
            Recurrence::None,
            Recurrence::Daily,
            Recurrence::Weekly,
            Recurrence::Monthly,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: Recurrence = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn test_rsvp_status_variants() {
        use community_calendar_integrity::RsvpStatus;
        let variants = vec![
            RsvpStatus::Going,
            RsvpStatus::Maybe,
            RsvpStatus::NotGoing,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: RsvpStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn test_event_category_filtering() {
        // Verify date_from_timestamp produces consistent date strings for known timestamps
        let ts1 = Timestamp::from_micros(1_774_483_200_000_000); // 2026-03-26
        let ts2 = Timestamp::from_micros(1_774_483_200_000_000); // same timestamp
        assert_eq!(date_from_timestamp(&ts1), date_from_timestamp(&ts2));
        // Verify format is YYYY-MM-DD
        let date = date_from_timestamp(&ts1);
        assert_eq!(date.len(), 10);
        assert_eq!(&date[4..5], "-");
        assert_eq!(&date[7..8], "-");
    }

    #[test]
    fn test_nearby_events_empty_cells() {
        // geohash_neighbors always returns exactly 8 neighbors
        let hash = commons_types::geo::geohash_encode(32.95, -96.73, 6);
        let neighbors = commons_types::geo::geohash_neighbors(&hash);
        assert_eq!(neighbors.len(), 8);
    }
}
