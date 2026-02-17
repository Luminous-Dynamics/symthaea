//! Space Integrity Zome
//!
//! Entry types and validation for private spaces (families, squads, cooperatives)
//! within the public Commons DHT. Implements the "membrane factory" concept from
//! the Fractal CivOS architecture.
//!
//! Spaces use capability grants for access control: only members with valid
//! grants can read/write space-scoped entries.

use hdi::prelude::*;

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

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Space(Space),
    Membership(Membership),
    SpaceCapability(SpaceCapability),
    SpaceInvitation(SpaceInvitation),
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
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Space(space) => validate_update_space(action, space),
                EntryTypes::Membership(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::SpaceCapability(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::SpaceInvitation(_) => Ok(ValidateCallbackResult::Valid),
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
}
