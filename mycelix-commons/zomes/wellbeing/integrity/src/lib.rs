// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Wellbeing Integrity Zome
//!
//! Self-sovereign wellbeing telemetry. No medical gatekeeper. Private by default.
//! The on-chain equivalent of the simulation's `PsychologicalNeeds` struct.
//!
//! The simulation proved that care work reduces allostatic load by 96% and
//! boosts collective phi by 303% compared to the punitive model. This zome
//! provides the telemetry that drives the care economy.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

// =============================================================================
// Entry types
// =============================================================================

/// Anchor entry for deterministic link bases.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Self-reported wellbeing check-in. Private by default (source chain only).
/// Agents opt in separately to share data for aggregate computation.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WellbeingCheckIn {
    /// Allostatic load proxy: self-reported stress (0-10).
    pub stress_level: u8,
    /// Social connection score (0-10).
    pub connection_score: u8,
    /// Engagement/presence score (0-10).
    pub engagement_score: u8,
    /// Optional free-text mood note (max 2048 bytes).
    pub mood_note: Option<String>,
    /// Timestamp of check-in.
    pub checked_in_at: Timestamp,
    /// Author's agent key (for self-verification).
    pub author: AgentPubKey,
}

/// Opt-in consent to share check-in data for aggregate computation.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AggregateOptIn {
    /// The agent opting in.
    pub agent: AgentPubKey,
    /// Whether currently opted in.
    pub active: bool,
    /// When opt-in was created/updated.
    pub updated_at: Timestamp,
}

/// Computed world-level aggregate snapshot (anonymous).
/// Created periodically from opted-in check-ins. Does NOT reveal individual data.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WellbeingAggregate {
    /// Mean stress across opted-in agents (0.0-10.0).
    pub mean_stress: f32,
    /// Mean connection across opted-in agents (0.0-10.0).
    pub mean_connection: f32,
    /// Mean engagement across opted-in agents (0.0-10.0).
    pub mean_engagement: f32,
    /// Number of agents contributing.
    pub sample_size: u32,
    /// Computation timestamp.
    pub computed_at: Timestamp,
    /// Agent who computed this snapshot.
    pub computed_by: AgentPubKey,
}

/// Type of wellbeing nudge suggestion.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum NudgeType {
    /// Suggest connecting with a care circle.
    CareCircleConnection,
    /// Suggest initiating a TEND care exchange.
    TendCareExchange,
    /// Suggest a wellbeing resource.
    WellbeingResource,
}

/// A gentle nudge generated when sustained high stress is detected.
/// Private to the agent's source chain. Never a mandate.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WellbeingNudge {
    /// The agent receiving the nudge.
    pub agent: AgentPubKey,
    /// Type of nudge suggestion.
    pub nudge_type: NudgeType,
    /// Human-readable suggestion text.
    pub message: String,
    /// Check-in hashes that triggered this nudge.
    pub trigger_checkins: Vec<ActionHash>,
    /// When generated.
    pub created_at: Timestamp,
    /// Whether the agent has acknowledged/dismissed.
    pub acknowledged: bool,
}

/// Selective share: shares a specific check-in with a care circle.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CheckInShare {
    /// The check-in being shared.
    pub checkin_hash: ActionHash,
    /// The care circle to share with.
    pub circle_hash: ActionHash,
    /// When shared.
    pub shared_at: Timestamp,
}

// =============================================================================
// Entry and link type registration
// =============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    #[entry_type(visibility = "private")]
    WellbeingCheckIn(WellbeingCheckIn),
    AggregateOptIn(AggregateOptIn),
    WellbeingAggregate(WellbeingAggregate),
    #[entry_type(visibility = "private")]
    WellbeingNudge(WellbeingNudge),
    CheckInShare(CheckInShare),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Agent -> their private check-ins.
    AgentToCheckIns,
    /// Agent -> their opt-in record.
    AgentToOptIn,
    /// Opt-in anchor -> opted-in agents.
    OptInToAgent,
    /// Anchor -> aggregate snapshots.
    AllAggregates,
    /// Agent -> their nudges.
    AgentToNudges,
    /// Check-in -> share records.
    CheckInToShare,
    /// Circle -> shared check-ins from members.
    CircleToSharedCheckIns,
}

// =============================================================================
// Validation constants
// =============================================================================

/// Maximum score value for wellbeing dimensions.
const MAX_SCORE: u8 = 10;

/// Maximum mood note length in bytes.
const MAX_MOOD_NOTE_BYTES: usize = 2048;

/// Maximum mean value for aggregate fields.
const MAX_AGGREGATE_MEAN: f32 = 10.0;

/// Minimum number of trigger check-ins for a nudge.
const MIN_NUDGE_TRIGGERS: usize = 3;

/// Maximum nudge message length.
const MAX_NUDGE_MESSAGE_BYTES: usize = 1024;

/// Maximum link tag size.
const MAX_TAG_BYTES: usize = 256;

// =============================================================================
// Genesis
// =============================================================================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// Validation
// =============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => {
                validate_create_entry(app_entry, &action.author)
            }
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                ..
            } => {
                // Only original author can update
                let original_action = must_get_action(original_action_hash)?;
                let author_check = check_author_match(
                    original_action.action().author(),
                    &action.author,
                    "update",
                );
                if author_check != ValidateCallbackResult::Valid {
                    return Ok(author_check);
                }
                validate_create_entry(app_entry, &action.author)
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type, tag, ..
        } => validate_link_tag(&link_type, &tag),
        FlatOp::RegisterDeleteLink {
            link_type,
            tag,
            action,
            ..
        } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            let result = check_link_author_match(
                original_action.action().author(),
                &action.author,
            );
            if result != ValidateCallbackResult::Valid {
                return Ok(result);
            }
            validate_link_tag(&link_type, &tag)
        }
        FlatOp::StoreRecord(_)
        | FlatOp::RegisterAgentActivity(_)
        | FlatOp::RegisterUpdate(_)
        | FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_entry(
    app_entry: EntryTypes,
    author: &AgentPubKey,
) -> ExternResult<ValidateCallbackResult> {
    match app_entry {
        EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),

        EntryTypes::WellbeingCheckIn(checkin) => {
            if checkin.stress_level > MAX_SCORE {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "stress_level must be 0-{MAX_SCORE}, got {}",
                    checkin.stress_level
                )));
            }
            if checkin.connection_score > MAX_SCORE {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "connection_score must be 0-{MAX_SCORE}, got {}",
                    checkin.connection_score
                )));
            }
            if checkin.engagement_score > MAX_SCORE {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "engagement_score must be 0-{MAX_SCORE}, got {}",
                    checkin.engagement_score
                )));
            }
            if let Some(ref note) = checkin.mood_note {
                if note.len() > MAX_MOOD_NOTE_BYTES {
                    return Ok(ValidateCallbackResult::Invalid(format!(
                        "mood_note exceeds {MAX_MOOD_NOTE_BYTES} bytes: {}",
                        note.len()
                    )));
                }
            }
            if &checkin.author != author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Check-in author must match action author".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }

        EntryTypes::AggregateOptIn(opt_in) => {
            if &opt_in.agent != author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Opt-in agent must match action author".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }

        EntryTypes::WellbeingAggregate(agg) => {
            if !agg.mean_stress.is_finite() || agg.mean_stress < 0.0 || agg.mean_stress > MAX_AGGREGATE_MEAN {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "mean_stress must be finite and 0.0-{MAX_AGGREGATE_MEAN}"
                )));
            }
            if !agg.mean_connection.is_finite() || agg.mean_connection < 0.0 || agg.mean_connection > MAX_AGGREGATE_MEAN {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "mean_connection must be finite and 0.0-{MAX_AGGREGATE_MEAN}"
                )));
            }
            if !agg.mean_engagement.is_finite() || agg.mean_engagement < 0.0 || agg.mean_engagement > MAX_AGGREGATE_MEAN {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "mean_engagement must be finite and 0.0-{MAX_AGGREGATE_MEAN}"
                )));
            }
            if agg.sample_size == 0 {
                return Ok(ValidateCallbackResult::Invalid(
                    "sample_size must be > 0".into(),
                ));
            }
            if &agg.computed_by != author {
                return Ok(ValidateCallbackResult::Invalid(
                    "computed_by must match action author".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }

        EntryTypes::WellbeingNudge(nudge) => {
            if &nudge.agent != author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Nudge agent must match action author (self-generated)".into(),
                ));
            }
            if nudge.trigger_checkins.len() < MIN_NUDGE_TRIGGERS {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "Nudge requires >= {MIN_NUDGE_TRIGGERS} trigger check-ins, got {}",
                    nudge.trigger_checkins.len()
                )));
            }
            if nudge.message.len() > MAX_NUDGE_MESSAGE_BYTES {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "Nudge message exceeds {MAX_NUDGE_MESSAGE_BYTES} bytes"
                )));
            }
            Ok(ValidateCallbackResult::Valid)
        }

        EntryTypes::CheckInShare(_share) => {
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

fn validate_link_tag(
    _link_type: &LinkTypes,
    tag: &LinkTag,
) -> ExternResult<ValidateCallbackResult> {
    if tag.0.len() > MAX_TAG_BYTES {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Link tag exceeds {MAX_TAG_BYTES} bytes: {}",
            tag.0.len()
        )));
    }
    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkin_serde_roundtrip() {
        let checkin = WellbeingCheckIn {
            stress_level: 5,
            connection_score: 7,
            engagement_score: 8,
            mood_note: Some("Feeling grounded today".into()),
            checked_in_at: Timestamp::from_micros(1000000),
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
        };
        let bytes = holochain_serialized_bytes::encode(&checkin).unwrap();
        let decoded: WellbeingCheckIn = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(checkin, decoded);
    }

    #[test]
    fn test_aggregate_serde_roundtrip() {
        let agg = WellbeingAggregate {
            mean_stress: 4.2,
            mean_connection: 6.8,
            mean_engagement: 7.1,
            sample_size: 42,
            computed_at: Timestamp::from_micros(2000000),
            computed_by: AgentPubKey::from_raw_36(vec![0u8; 36]),
        };
        let bytes = holochain_serialized_bytes::encode(&agg).unwrap();
        let decoded: WellbeingAggregate = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(agg, decoded);
    }

    #[test]
    fn test_nudge_type_serde_all_variants() {
        for nudge_type in [
            NudgeType::CareCircleConnection,
            NudgeType::TendCareExchange,
            NudgeType::WellbeingResource,
        ] {
            let bytes = holochain_serialized_bytes::encode(&nudge_type).unwrap();
            let decoded: NudgeType = holochain_serialized_bytes::decode(&bytes).unwrap();
            assert_eq!(nudge_type, decoded);
        }
    }

    #[test]
    fn test_opt_in_serde_roundtrip() {
        let opt_in = AggregateOptIn {
            agent: AgentPubKey::from_raw_36(vec![1u8; 36]),
            active: true,
            updated_at: Timestamp::from_micros(3000000),
        };
        let bytes = holochain_serialized_bytes::encode(&opt_in).unwrap();
        let decoded: AggregateOptIn = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(opt_in, decoded);
    }

    #[test]
    fn test_check_in_share_serde_roundtrip() {
        let share = CheckInShare {
            checkin_hash: ActionHash::from_raw_36(vec![2u8; 36]),
            circle_hash: ActionHash::from_raw_36(vec![3u8; 36]),
            shared_at: Timestamp::from_micros(4000000),
        };
        let bytes = holochain_serialized_bytes::encode(&share).unwrap();
        let decoded: CheckInShare = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(share, decoded);
    }

    #[test]
    fn test_nudge_serde_roundtrip() {
        let nudge = WellbeingNudge {
            agent: AgentPubKey::from_raw_36(vec![4u8; 36]),
            nudge_type: NudgeType::CareCircleConnection,
            message: "Consider connecting with your care circle".into(),
            trigger_checkins: vec![
                ActionHash::from_raw_36(vec![5u8; 36]),
                ActionHash::from_raw_36(vec![6u8; 36]),
                ActionHash::from_raw_36(vec![7u8; 36]),
            ],
            created_at: Timestamp::from_micros(5000000),
            acknowledged: false,
        };
        let bytes = holochain_serialized_bytes::encode(&nudge).unwrap();
        let decoded: WellbeingNudge = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(nudge, decoded);
    }
}
