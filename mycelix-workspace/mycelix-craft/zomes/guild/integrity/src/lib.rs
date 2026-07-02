#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//!
//! Guild integrity zome — entry types and validation for consciousness-gated
//! professional guilds within Mycelix Craft.
//!
//! Guilds are the highest fractal coordination layer (Guild/Bioregion in VISION.md),
//! implementing Elinor Ostrom's Nested Enterprises principle for professional
//! federations. Members progress through consciousness-gated roles:
//! Observer → Apprentice → Journeyman → Master → Elder.

use hdi::prelude::*;

// ---------------------------------------------------------------------------
// Entry types
// ---------------------------------------------------------------------------

/// A professional guild — consciousness-gated federation of practitioners.
///
/// Guilds coordinate skill certification, apprenticeship pathways, and
/// cross-bioregion standards. Governance model scales with membership size.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Guild {
    /// Human-readable guild name (e.g., "Rust Developers", "Regenerative Farmers")
    pub name: String,
    /// Brief description of the guild's purpose and domain
    pub description: String,
    /// Professional domain tag (e.g., "software_engineering", "agriculture")
    pub professional_domain: String,
    /// Minimum consciousness score (0-1000 permille) required to join
    pub consciousness_minimum_permille: u16,
    /// Parent guild (for nested/hierarchical guilds)
    pub parent_guild: Option<ActionHash>,
    /// Bioregion this guild is anchored to (optional — global guilds have None)
    pub bioregion: Option<String>,
    /// How collective decisions are made (scales with membership)
    pub governance_model: GuildGovernanceModel,
    /// When this guild was created
    pub created_at: Timestamp,
}

/// Guild governance model — scales with membership size.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GuildGovernanceModel {
    /// Full consensus required. Best for small guilds (<20 members).
    Consensus,
    /// Sociocratic circles with consent-based decision making (20-100 members).
    SociocraticCircle,
    /// Delegated liquid democracy for large guilds (100+ members).
    LiquidDemocracy,
}

/// A guild membership record — links an agent to a guild with a role.
///
/// Role transitions are consciousness-gated: each role has a minimum
/// consciousness threshold that must be met for promotion.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GuildMembership {
    /// The guild this membership belongs to
    pub guild_id: ActionHash,
    /// The member agent
    pub member: AgentPubKey,
    /// Current role within the guild
    pub role: GuildRole,
    /// When the member joined this guild
    pub joined_at: Timestamp,
    /// When the role was last changed
    pub last_role_change: Timestamp,
    /// Consciousness score at the time of joining (for audit)
    pub consciousness_at_join_permille: u16,
}

/// Guild roles — consciousness-gated progression through mastery levels.
///
/// Maps to historical craft guild traditions:
/// - Observer: Can view guild activity, not yet committed
/// - Apprentice: Learning under guidance, building initial credentials
/// - Journeyman: Independent practitioner, can take commissions
/// - Master: Can certify others, creates certification paths
/// - Elder: Guild governance authority, federation establishment
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GuildRole {
    /// Consciousness 0.1-0.3: Read-only access, can observe guild activity
    Observer,
    /// Consciousness 0.3-0.5: Learning under mentorship, building credentials
    Apprentice,
    /// Consciousness 0.5-0.75: Independent practitioner, can post/accept work
    Journeyman,
    /// Consciousness 0.75-0.9: Can certify apprentices, create certification paths
    Master,
    /// Consciousness 0.9+: Full governance authority, can establish federations
    Elder,
}

impl GuildRole {
    /// Minimum consciousness score (permille) required for this role.
    pub const fn minimum_consciousness_permille(&self) -> u16 {
        match self {
            Self::Observer => 100,
            Self::Apprentice => 300,
            Self::Journeyman => 500,
            Self::Master => 750,
            Self::Elder => 900,
        }
    }

    /// Whether this role can promote other members.
    pub const fn can_promote(&self) -> bool {
        matches!(self, Self::Master | Self::Elder)
    }

    /// Whether this role can create certification paths.
    pub const fn can_certify(&self) -> bool {
        matches!(self, Self::Master | Self::Elder)
    }
}

/// A certification path — defines requirements for guild-recognized mastery.
///
/// Masters and Elders create these to codify what it means to be certified
/// in a particular skill domain within the guild.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CertificationPath {
    /// The guild that owns this certification path
    pub guild_id: ActionHash,
    /// Human-readable name (e.g., "Holochain Developer Level 2")
    pub name: String,
    /// Detailed description of what this certification proves
    pub description: String,
    /// Ordered list of requirements to complete this certification
    pub requirements: Vec<CertificationRequirement>,
    /// Number of assessors required for certification (quorum)
    pub required_assessors: u8,
    /// Minimum guild role required to begin this certification path
    pub minimum_role: GuildRole,
    /// When this path was created
    pub created_at: Timestamp,
}

/// A single requirement within a certification path.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CertificationRequirement {
    /// What the candidate must demonstrate
    pub description: String,
    /// Optional credential type that satisfies this requirement (e.g., "rust_proficiency")
    pub credential_type: Option<String>,
    /// Minimum living credential vitality (0-1000 permille) — ensures ongoing competence
    pub minimum_vitality_permille: u16,
    /// Whether the candidate must provide evidence (portfolio, project, etc.)
    pub evidence_required: bool,
}

/// A federation link between two guilds (potentially in different DNAs/bioregions).
///
/// Federations allow guilds to share standards and recognize each other's
/// certifications. Only Elders can establish federation links.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GuildFederationLink {
    /// The local guild establishing this federation
    pub local_guild_id: ActionHash,
    /// The remote guild being federated with
    pub remote_guild_id: ActionHash,
    /// DNA hash of the remote guild (for cross-DNA verification)
    pub remote_dna_hash: Option<String>,
    /// Standards that both guilds agree to uphold
    pub shared_standards: Vec<String>,
    /// When this federation was established
    pub established_at: Timestamp,
}

/// Anchor for indexing (reuse pattern from craft-graph).
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GuildAnchor(pub String);

// ---------------------------------------------------------------------------
// Entry type registration
// ---------------------------------------------------------------------------

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Guild(Guild),
    #[entry_type(visibility = "public")]
    GuildMembership(GuildMembership),
    #[entry_type(visibility = "public")]
    CertificationPath(CertificationPath),
    #[entry_type(visibility = "public")]
    GuildFederationLink(GuildFederationLink),
    #[entry_type(visibility = "public")]
    GuildAnchor(GuildAnchor),
}

// ---------------------------------------------------------------------------
// Link types
// ---------------------------------------------------------------------------

#[hdk_link_types]
pub enum LinkTypes {
    /// Guild → its memberships
    GuildToMembership,
    /// Agent → their guild memberships
    AgentToGuildMembership,
    /// Guild → its certification paths
    GuildToCertificationPath,
    /// Guild → child guilds (nested hierarchy)
    GuildToChildGuild,
    /// Guild → federation links with other guilds
    GuildToFederationLink,
    /// Global anchor → all guilds (for discovery)
    AllGuilds,
    /// Domain anchor → guilds in that domain
    DomainToGuild,
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink { original_action, action, .. } => {
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this link".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
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
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can update their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can delete their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_guild(guild: &Guild) -> ExternResult<ValidateCallbackResult> {
    if guild.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Guild name cannot be empty".into()));
    }
    if guild.name.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid("Guild name exceeds 200 characters".into()));
    }
    if guild.description.len() > 2000 {
        return Ok(ValidateCallbackResult::Invalid("Guild description exceeds 2000 characters".into()));
    }
    if guild.professional_domain.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Professional domain cannot be empty".into()));
    }
    if guild.consciousness_minimum_permille > 1000 {
        return Ok(ValidateCallbackResult::Invalid("Consciousness minimum cannot exceed 1000".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_membership(membership: &GuildMembership) -> ExternResult<ValidateCallbackResult> {
    if membership.consciousness_at_join_permille > 1000 {
        return Ok(ValidateCallbackResult::Invalid("Consciousness score cannot exceed 1000".into()));
    }
    // Verify consciousness meets role minimum
    let role_min = membership.role.minimum_consciousness_permille();
    if membership.consciousness_at_join_permille < role_min {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Consciousness {} insufficient for {:?} role (minimum {})",
            membership.consciousness_at_join_permille, membership.role, role_min
        )));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_certification_path(path: &CertificationPath) -> ExternResult<ValidateCallbackResult> {
    if path.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Certification path name cannot be empty".into()));
    }
    if path.requirements.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Certification path must have at least one requirement".into()));
    }
    if path.required_assessors == 0 {
        return Ok(ValidateCallbackResult::Invalid("At least one assessor is required".into()));
    }
    for req in &path.requirements {
        if req.minimum_vitality_permille > 1000 {
            return Ok(ValidateCallbackResult::Invalid("Vitality requirement cannot exceed 1000".into()));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guild_role_consciousness_thresholds() {
        assert_eq!(GuildRole::Observer.minimum_consciousness_permille(), 100);
        assert_eq!(GuildRole::Apprentice.minimum_consciousness_permille(), 300);
        assert_eq!(GuildRole::Journeyman.minimum_consciousness_permille(), 500);
        assert_eq!(GuildRole::Master.minimum_consciousness_permille(), 750);
        assert_eq!(GuildRole::Elder.minimum_consciousness_permille(), 900);
    }

    #[test]
    fn only_masters_and_elders_can_promote() {
        assert!(!GuildRole::Observer.can_promote());
        assert!(!GuildRole::Apprentice.can_promote());
        assert!(!GuildRole::Journeyman.can_promote());
        assert!(GuildRole::Master.can_promote());
        assert!(GuildRole::Elder.can_promote());
    }

    #[test]
    fn only_masters_and_elders_can_certify() {
        assert!(!GuildRole::Observer.can_certify());
        assert!(!GuildRole::Apprentice.can_certify());
        assert!(!GuildRole::Journeyman.can_certify());
        assert!(GuildRole::Master.can_certify());
        assert!(GuildRole::Elder.can_certify());
    }

    #[test]
    fn consciousness_thresholds_increase_monotonically() {
        let roles = [
            GuildRole::Observer,
            GuildRole::Apprentice,
            GuildRole::Journeyman,
            GuildRole::Master,
            GuildRole::Elder,
        ];
        for window in roles.windows(2) {
            assert!(
                window[0].minimum_consciousness_permille()
                    < window[1].minimum_consciousness_permille(),
                "{:?} threshold should be less than {:?}",
                window[0],
                window[1]
            );
        }
    }

    // ---- Validation tests ----

    fn valid_guild() -> Guild {
        Guild {
            name: "Rust Developers".to_string(),
            description: "Guild for Rust practitioners".to_string(),
            professional_domain: "software_engineering".to_string(),
            consciousness_minimum_permille: 300,
            parent_guild: None,
            bioregion: Some("global".to_string()),
            governance_model: GuildGovernanceModel::Consensus,
            created_at: Timestamp::from_micros(0),
        }
    }

    #[test]
    fn valid_guild_passes() {
        assert!(matches!(
            validate_guild(&valid_guild()),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn empty_guild_name_rejected() {
        let mut g = valid_guild();
        g.name = String::new();
        assert!(matches!(
            validate_guild(&g),
            Ok(ValidateCallbackResult::Invalid(_))
        ));
    }

    #[test]
    fn guild_name_too_long_rejected() {
        let mut g = valid_guild();
        g.name = "x".repeat(201);
        assert!(matches!(
            validate_guild(&g),
            Ok(ValidateCallbackResult::Invalid(_))
        ));
    }

    #[test]
    fn guild_description_too_long_rejected() {
        let mut g = valid_guild();
        g.description = "x".repeat(2001);
        assert!(matches!(
            validate_guild(&g),
            Ok(ValidateCallbackResult::Invalid(_))
        ));
    }

    #[test]
    fn empty_domain_rejected() {
        let mut g = valid_guild();
        g.professional_domain = String::new();
        assert!(matches!(
            validate_guild(&g),
            Ok(ValidateCallbackResult::Invalid(_))
        ));
    }

    #[test]
    fn consciousness_over_1000_rejected() {
        let mut g = valid_guild();
        g.consciousness_minimum_permille = 1001;
        assert!(matches!(
            validate_guild(&g),
            Ok(ValidateCallbackResult::Invalid(_))
        ));
    }

    #[test]
    fn membership_role_consciousness_enforced() {
        // Observer needs 100, so 50 should fail
        let m = GuildMembership {
            guild_id: ActionHash::from_raw_36(vec![0; 36]),
            member: AgentPubKey::from_raw_36(vec![0; 36]),
            role: GuildRole::Observer,
            joined_at: Timestamp::from_micros(0),
            last_role_change: Timestamp::from_micros(0),
            consciousness_at_join_permille: 50,
        };
        assert!(matches!(
            validate_membership(&m),
            Ok(ValidateCallbackResult::Invalid(_))
        ));
    }

    #[test]
    fn membership_sufficient_consciousness_passes() {
        let m = GuildMembership {
            guild_id: ActionHash::from_raw_36(vec![0; 36]),
            member: AgentPubKey::from_raw_36(vec![0; 36]),
            role: GuildRole::Apprentice,
            joined_at: Timestamp::from_micros(0),
            last_role_change: Timestamp::from_micros(0),
            consciousness_at_join_permille: 400,
        };
        assert!(matches!(
            validate_membership(&m),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn master_needs_750_consciousness() {
        let m = GuildMembership {
            guild_id: ActionHash::from_raw_36(vec![0; 36]),
            member: AgentPubKey::from_raw_36(vec![0; 36]),
            role: GuildRole::Master,
            joined_at: Timestamp::from_micros(0),
            last_role_change: Timestamp::from_micros(0),
            consciousness_at_join_permille: 700,
        };
        assert!(matches!(
            validate_membership(&m),
            Ok(ValidateCallbackResult::Invalid(_))
        ));
    }

    #[test]
    fn certification_path_needs_name() {
        let path = CertificationPath {
            guild_id: ActionHash::from_raw_36(vec![0; 36]),
            name: String::new(),
            description: "test".to_string(),
            requirements: vec![CertificationRequirement {
                description: "Know Rust".to_string(),
                credential_type: Some("rust".to_string()),
                minimum_vitality_permille: 800,
                evidence_required: true,
            }],
            required_assessors: 2,
            minimum_role: GuildRole::Apprentice,
            created_at: Timestamp::from_micros(0),
        };
        assert!(matches!(
            validate_certification_path(&path),
            Ok(ValidateCallbackResult::Invalid(_))
        ));
    }

    #[test]
    fn certification_path_needs_requirements() {
        let path = CertificationPath {
            guild_id: ActionHash::from_raw_36(vec![0; 36]),
            name: "Rust Developer L1".to_string(),
            description: "test".to_string(),
            requirements: vec![],
            required_assessors: 2,
            minimum_role: GuildRole::Apprentice,
            created_at: Timestamp::from_micros(0),
        };
        assert!(matches!(
            validate_certification_path(&path),
            Ok(ValidateCallbackResult::Invalid(_))
        ));
    }

    #[test]
    fn certification_path_needs_assessors() {
        let path = CertificationPath {
            guild_id: ActionHash::from_raw_36(vec![0; 36]),
            name: "Rust Developer L1".to_string(),
            description: "test".to_string(),
            requirements: vec![CertificationRequirement {
                description: "Know Rust".to_string(),
                credential_type: None,
                minimum_vitality_permille: 500,
                evidence_required: false,
            }],
            required_assessors: 0,
            minimum_role: GuildRole::Apprentice,
            created_at: Timestamp::from_micros(0),
        };
        assert!(matches!(
            validate_certification_path(&path),
            Ok(ValidateCallbackResult::Invalid(_))
        ));
    }

    #[test]
    fn governance_model_serialization() {
        let json = serde_json::to_string(&GuildGovernanceModel::SociocraticCircle).unwrap();
        let deserialized: GuildGovernanceModel = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, GuildGovernanceModel::SociocraticCircle);
    }

    #[test]
    fn guild_role_serialization_roundtrip() {
        for role in [GuildRole::Observer, GuildRole::Apprentice, GuildRole::Journeyman, GuildRole::Master, GuildRole::Elder] {
            let json = serde_json::to_string(&role).unwrap();
            let deserialized: GuildRole = serde_json::from_str(&json).unwrap();
            assert_eq!(role, deserialized);
        }
    }
}
