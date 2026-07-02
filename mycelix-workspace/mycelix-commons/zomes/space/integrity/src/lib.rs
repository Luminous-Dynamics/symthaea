// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Space Integrity Zome
//!
//! Entry types and validation for private spaces (families, squads, cooperatives)
//! within the public Commons DHT. Implements the "membrane factory" concept from
//! the Fractal CivOS architecture.
//!
//! Spaces use capability grants for access control: only members with valid
//! grants can read/write space-scoped entries.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

/// Anchor for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A private space within the Commons
///
/// Spaces provide sub-group privacy within the public DHT using
/// capability-grant-based access control.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Space {
    /// Unique space identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Space type (family, squad, cooperative, custom)
    pub space_type: SpaceType,
    /// Description of the space's purpose
    pub description: String,
    /// Agent who created the space
    pub creator: AgentPubKey,
    /// Maximum number of members (0 = unlimited)
    pub max_members: u32,
    /// Whether new members need approval from existing members
    pub requires_approval: bool,
    /// Minimum number of approvals needed for new members
    pub approval_threshold: u32,
    /// Whether the space is currently accepting new members
    pub open: bool,
    /// Creation timestamp
    pub created_at: Timestamp,
}

/// Types of private spaces
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SpaceType {
    /// Family unit — small, high-trust
    Family,
    /// Squad — project team, working group
    Squad,
    /// Cooperative — economic collaboration
    Cooperative,
    /// Custom space type
    Custom(String),
}

/// Membership in a space
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Membership {
    /// Reference to parent space
    pub space_id: String,
    /// Member's agent public key
    pub member: AgentPubKey,
    /// Member's role within the space
    pub role: MemberRole,
    /// Whether membership is active
    pub active: bool,
    /// Who invited/approved this member
    pub invited_by: AgentPubKey,
    /// Timestamp of joining
    pub joined_at: Timestamp,
}

/// Roles within a space
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MemberRole {
    /// Full admin rights (can invite, remove, modify space)
    Admin,
    /// Standard member (can read/write within space)
    Member,
    /// Read-only observer
    Observer,
}

/// Capability token for space access
///
/// Wraps a Holochain capability grant with space-specific metadata.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SpaceCapability {
    /// Reference to parent space
    pub space_id: String,
    /// Agent this capability is granted to
    pub grantee: AgentPubKey,
    /// What functions this capability allows
    pub allowed_functions: Vec<String>,
    /// Expiry timestamp (None = no expiry)
    pub expires_at: Option<Timestamp>,
    /// Whether this capability has been revoked
    pub revoked: bool,
    /// Timestamp of grant
    pub granted_at: Timestamp,
}

/// Invitation to join a space (pending approval)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SpaceInvitation {
    /// Reference to parent space
    pub space_id: String,
    /// Agent being invited
    pub invitee: AgentPubKey,
    /// Agent who created the invitation
    pub inviter: AgentPubKey,
    /// Optional message
    pub message: String,
    /// Current approval count
    pub approvals: u32,
    /// Agents who have approved
    pub approved_by: Vec<AgentPubKey>,
    /// Status: pending, approved, rejected, expired
    pub status: InvitationStatus,
    /// Timestamp
    pub created_at: Timestamp,
}

/// Invitation status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum InvitationStatus {
    Pending,
    Approved,
    Rejected,
    Expired,
}

/// A booking for a shared resource within a space
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ResourceBooking {
    /// Unique booking identifier
    pub id: String,
    /// Reference to parent space
    pub space_id: String,
    /// Name of the resource being booked (room, tool, equipment)
    pub resource_name: String,
    /// Agent who made the booking
    pub booked_by: AgentPubKey,
    /// Start time (microseconds since epoch)
    pub start_time: u64,
    /// End time (microseconds since epoch)
    pub end_time: u64,
    /// Current booking status
    pub status: BookingStatus,
    /// Optional notes about the booking
    pub notes: String,
    /// Creation timestamp
    pub created_at: Timestamp,
}

/// Booking status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum BookingStatus {
    Pending,
    Confirmed,
    Cancelled,
}

/// A recurring schedule/event within a space
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SpaceSchedule {
    /// Unique schedule identifier
    pub id: String,
    /// Reference to parent space
    pub space_id: String,
    /// Event/meeting title
    pub title: String,
    /// Description of the event
    pub description: String,
    /// How often this recurs
    pub recurrence: Recurrence,
    /// Next occurrence (microseconds since epoch)
    pub next_occurrence: u64,
    /// Duration in minutes
    pub duration_minutes: u32,
    /// Agent who created the schedule
    pub creator: AgentPubKey,
    /// Creation timestamp
    pub created_at: Timestamp,
}

/// Recurrence pattern for schedules
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Recurrence {
    Once,
    Daily,
    Weekly,
    Monthly,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Space(Space),
    Membership(Membership),
    SpaceCapability(SpaceCapability),
    SpaceInvitation(SpaceInvitation),
    ResourceBooking(ResourceBooking),
    SpaceSchedule(SpaceSchedule),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// All spaces anchor → space
    AllSpaces,
    /// Space → its members
    SpaceToMembers,
    /// Space → capability grants
    SpaceToCapabilities,
    /// Space → invitations
    SpaceToInvitations,
    /// Agent → spaces they belong to
    AgentToSpaces,
    /// Space type anchor → spaces of that type
    TypeToSpaces,
    /// Space → resource bookings
    SpaceToBookings,
    /// Space → scheduled events
    SpaceToSchedules,
    /// Agent → their bookings
    AgentToBookings,
}

/// Validation callback
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Space(space) => validate_create_space(action, space),
                EntryTypes::Membership(membership) => {
                    validate_create_membership(action, membership)
                }
                EntryTypes::SpaceCapability(cap) => validate_create_capability(action, cap),
                EntryTypes::SpaceInvitation(inv) => validate_create_invitation(action, inv),
                EntryTypes::ResourceBooking(booking) => validate_create_booking(action, booking),
                EntryTypes::SpaceSchedule(schedule) => validate_create_schedule(action, schedule),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Space(space) => validate_update_space(action, space),
                EntryTypes::Membership(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::SpaceCapability(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::SpaceInvitation(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ResourceBooking(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::SpaceSchedule(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "delete",
            ))
        }
    }
}

fn validate_create_space(_action: Create, space: Space) -> ExternResult<ValidateCallbackResult> {
    if space.name.is_empty() || space.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Space name must be 1-256 characters".into(),
        ));
    }

    if space.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Description must be at most 4096 characters".into(),
        ));
    }

    if space.id.is_empty() || space.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Space ID must be 1-256 characters".into(),
        ));
    }

    if space.requires_approval && space.approval_threshold == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Approval threshold must be > 0 when approval is required".into(),
        ));
    }

    if let SpaceType::Custom(ref name) = space.space_type {
        if name.is_empty() || name.len() > 64 {
            return Ok(ValidateCallbackResult::Invalid(
                "Custom space type name must be 1-64 characters".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_space(_action: Update, space: Space) -> ExternResult<ValidateCallbackResult> {
    if space.name.is_empty() || space.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Space name must be 1-256 characters".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_membership(
    _action: Create,
    membership: Membership,
) -> ExternResult<ValidateCallbackResult> {
    if membership.space_id.is_empty() || membership.space_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Space ID must be 1-256 characters".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_capability(
    _action: Create,
    cap: SpaceCapability,
) -> ExternResult<ValidateCallbackResult> {
    if cap.space_id.is_empty() || cap.space_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Space ID must be 1-256 characters".into(),
        ));
    }

    if cap.allowed_functions.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Capability must allow at least one function".into(),
        ));
    }

    if cap.allowed_functions.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Capability cannot allow more than 50 functions".into(),
        ));
    }

    for func in &cap.allowed_functions {
        if func.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Allowed function name too long (max 256 chars per item)".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_invitation(
    _action: Create,
    inv: SpaceInvitation,
) -> ExternResult<ValidateCallbackResult> {
    if inv.space_id.is_empty() || inv.space_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Space ID must be 1-256 characters".into(),
        ));
    }

    if inv.message.len() > 1024 {
        return Ok(ValidateCallbackResult::Invalid(
            "Invitation message must be at most 1024 characters".into(),
        ));
    }

    if inv.status != InvitationStatus::Pending {
        return Ok(ValidateCallbackResult::Invalid(
            "New invitations must start with Pending status".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_booking(
    _action: Create,
    booking: ResourceBooking,
) -> ExternResult<ValidateCallbackResult> {
    validate_booking(booking)
}

fn validate_booking(booking: ResourceBooking) -> ExternResult<ValidateCallbackResult> {
    if booking.id.is_empty() || booking.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Booking ID must be 1-256 characters".into(),
        ));
    }

    if booking.space_id.is_empty() || booking.space_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Space ID must be 1-256 characters".into(),
        ));
    }

    if booking.resource_name.is_empty() || booking.resource_name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource name must be 1-256 characters".into(),
        ));
    }

    if booking.end_time <= booking.start_time {
        return Ok(ValidateCallbackResult::Invalid(
            "End time must be after start time".into(),
        ));
    }

    if booking.notes.len() > 1024 {
        return Ok(ValidateCallbackResult::Invalid(
            "Notes must be at most 1024 characters".into(),
        ));
    }

    if booking.status != BookingStatus::Pending {
        return Ok(ValidateCallbackResult::Invalid(
            "New bookings must start with Pending status".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_schedule(
    _action: Create,
    schedule: SpaceSchedule,
) -> ExternResult<ValidateCallbackResult> {
    validate_schedule(schedule)
}

fn validate_schedule(schedule: SpaceSchedule) -> ExternResult<ValidateCallbackResult> {
    if schedule.id.is_empty() || schedule.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Schedule ID must be 1-256 characters".into(),
        ));
    }

    if schedule.space_id.is_empty() || schedule.space_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Space ID must be 1-256 characters".into(),
        ));
    }

    if schedule.title.is_empty() || schedule.title.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Schedule title must be 1-256 characters".into(),
        ));
    }

    if schedule.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Description must be at most 4096 characters".into(),
        ));
    }

    if schedule.duration_minutes == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Duration must be greater than 0 minutes".into(),
        ));
    }

    if schedule.duration_minutes > 1440 {
        return Ok(ValidateCallbackResult::Invalid(
            "Duration cannot exceed 24 hours (1440 minutes)".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_type_serde_roundtrip() {
        let types = vec![
            SpaceType::Family,
            SpaceType::Squad,
            SpaceType::Cooperative,
            SpaceType::Custom("guild".into()),
        ];
        for t in types {
            let json = serde_json::to_string(&t).unwrap();
            let back: SpaceType = serde_json::from_str(&json).unwrap();
            assert_eq!(t, back);
        }
    }

    #[test]
    fn member_role_serde_roundtrip() {
        let roles = vec![MemberRole::Admin, MemberRole::Member, MemberRole::Observer];
        for r in roles {
            let json = serde_json::to_string(&r).unwrap();
            let back: MemberRole = serde_json::from_str(&json).unwrap();
            assert_eq!(r, back);
        }
    }

    #[test]
    fn invitation_status_serde_roundtrip() {
        let statuses = vec![
            InvitationStatus::Pending,
            InvitationStatus::Approved,
            InvitationStatus::Rejected,
            InvitationStatus::Expired,
        ];
        for s in statuses {
            let json = serde_json::to_string(&s).unwrap();
            let back: InvitationStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back);
        }
    }

    #[test]
    fn booking_status_serde_roundtrip() {
        let statuses = vec![
            BookingStatus::Pending,
            BookingStatus::Confirmed,
            BookingStatus::Cancelled,
        ];
        for s in statuses {
            let json = serde_json::to_string(&s).unwrap();
            let back: BookingStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back);
        }
    }

    #[test]
    fn recurrence_serde_roundtrip() {
        let variants = vec![
            Recurrence::Once,
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

    fn agent_1() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xaa; 36])
    }

    fn valid_booking() -> ResourceBooking {
        ResourceBooking {
            id: "booking_001".to_string(),
            space_id: "space:test:123".to_string(),
            resource_name: "Meeting Room A".to_string(),
            booked_by: agent_1(),
            start_time: 1000000,
            end_time: 2000000,
            status: BookingStatus::Pending,
            notes: "Team standup".to_string(),
            created_at: Timestamp::from_micros(1000000),
        }
    }

    fn valid_schedule() -> SpaceSchedule {
        SpaceSchedule {
            id: "sched_001".to_string(),
            space_id: "space:test:123".to_string(),
            title: "Weekly Sync".to_string(),
            description: "Weekly team synchronization".to_string(),
            recurrence: Recurrence::Weekly,
            next_occurrence: 1000000,
            duration_minutes: 60,
            creator: agent_1(),
            created_at: Timestamp::from_micros(1000000),
        }
    }

    #[test]
    fn booking_serde_roundtrip() {
        let b = valid_booking();
        let json = serde_json::to_string(&b).unwrap();
        let back: ResourceBooking = serde_json::from_str(&json).unwrap();
        assert_eq!(b.id, back.id);
        assert_eq!(b.resource_name, back.resource_name);
    }

    #[test]
    fn schedule_serde_roundtrip() {
        let s = valid_schedule();
        let json = serde_json::to_string(&s).unwrap();
        let back: SpaceSchedule = serde_json::from_str(&json).unwrap();
        assert_eq!(s.id, back.id);
        assert_eq!(s.recurrence, back.recurrence);
    }

    #[test]
    fn validate_booking_valid() {
        let result = validate_booking(valid_booking());
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn validate_booking_empty_id() {
        let mut b = valid_booking();
        b.id = "".to_string();
        let result = validate_booking(b);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn validate_booking_empty_resource_name() {
        let mut b = valid_booking();
        b.resource_name = "".to_string();
        let result = validate_booking(b);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn validate_booking_end_before_start() {
        let mut b = valid_booking();
        b.end_time = b.start_time;
        let result = validate_booking(b);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn validate_booking_notes_too_long() {
        let mut b = valid_booking();
        b.notes = "x".repeat(1025);
        let result = validate_booking(b);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn validate_booking_wrong_initial_status() {
        let mut b = valid_booking();
        b.status = BookingStatus::Confirmed;
        let result = validate_booking(b);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn validate_schedule_valid() {
        let result = validate_schedule(valid_schedule());
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn validate_schedule_empty_title() {
        let mut s = valid_schedule();
        s.title = "".to_string();
        let result = validate_schedule(s);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn validate_schedule_zero_duration() {
        let mut s = valid_schedule();
        s.duration_minutes = 0;
        let result = validate_schedule(s);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn validate_schedule_excessive_duration() {
        let mut s = valid_schedule();
        s.duration_minutes = 1441;
        let result = validate_schedule(s);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn validate_schedule_max_duration() {
        let mut s = valid_schedule();
        s.duration_minutes = 1440;
        let result = validate_schedule(s);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn validate_schedule_description_too_long() {
        let mut s = valid_schedule();
        s.description = "x".repeat(4097);
        let result = validate_schedule(s);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn validate_booking_notes_at_limit() {
        let mut b = valid_booking();
        b.notes = "x".repeat(1024);
        let result = validate_booking(b);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn validate_booking_id_at_limit() {
        let mut b = valid_booking();
        b.id = "x".repeat(256);
        let result = validate_booking(b);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn validate_booking_id_over_limit() {
        let mut b = valid_booking();
        b.id = "x".repeat(257);
        let result = validate_booking(b);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // ── String length boundary tests ──────────────────────────────────

    fn fake_create() -> Create {
        Create {
            author: agent_1(),
            timestamp: Timestamp::from_micros(0),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex(0),
                ZomeIndex(0),
                EntryVisibility::Public,
            )),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: EntryRateWeight::default(),
        }
    }

    fn valid_capability() -> SpaceCapability {
        SpaceCapability {
            space_id: "space:test:123".to_string(),
            grantee: agent_1(),
            allowed_functions: vec!["read_data".to_string(), "write_data".to_string()],
            expires_at: None,
            revoked: false,
            granted_at: Timestamp::from_micros(1000000),
        }
    }

    #[test]
    fn capability_allowed_function_at_limit_accepted() {
        let mut cap = valid_capability();
        cap.allowed_functions = vec!["x".repeat(256)];
        let result = validate_create_capability(fake_create(), cap);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn capability_allowed_function_over_limit_rejected() {
        let mut cap = valid_capability();
        cap.allowed_functions = vec!["x".repeat(257)];
        let result = validate_create_capability(fake_create(), cap);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn capability_second_function_over_limit_rejected() {
        let mut cap = valid_capability();
        cap.allowed_functions = vec!["valid_func".to_string(), "x".repeat(257)];
        let result = validate_create_capability(fake_create(), cap);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }
}
