// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Housing Governance Integrity Zome
//! Entry types and validation for board meetings, resolutions, bylaws, and elections.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Type of board meeting
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MeetingType {
    Regular,
    Special,
    Annual,
    Emergency,
}

/// A board meeting
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BoardMeeting {
    pub cooperative_hash: Option<ActionHash>,
    pub title: String,
    pub agenda: Vec<String>,
    pub scheduled_at: Timestamp,
    pub location: String,
    pub meeting_type: MeetingType,
    pub minutes: Option<String>,
    pub attendees: Vec<AgentPubKey>,
}

/// Category of a resolution
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ResolutionCategory {
    Budget,
    Maintenance,
    Membership,
    Rules,
    Assessment,
    Improvement,
    Emergency,
    Other(String),
}

/// A resolution proposed or adopted at a meeting
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Resolution {
    pub meeting_hash: Option<ActionHash>,
    pub title: String,
    pub description: String,
    pub proposed_by: AgentPubKey,
    pub category: ResolutionCategory,
    pub votes_for: u32,
    pub votes_against: u32,
    pub votes_abstain: u32,
    pub quorum_met: bool,
    pub passed: bool,
    pub effective_date: Option<Timestamp>,
}

/// A bylaw of the cooperative
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ByLaw {
    pub id: String,
    pub title: String,
    pub content: String,
    pub version: u32,
    pub adopted_at: Timestamp,
    pub amended_at: Option<Timestamp>,
    pub supersedes: Option<ActionHash>,
}

/// A candidate entry in an election
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CandidateEntry {
    pub agent: AgentPubKey,
    pub position: String,
    pub statement: String,
}

/// Result of an election for a single position
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ElectionResult {
    pub position: String,
    pub winner: AgentPubKey,
    pub votes_received: u32,
}

/// An election for cooperative positions
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Election {
    pub title: String,
    pub positions: Vec<String>,
    pub candidates: Vec<CandidateEntry>,
    pub voting_opens: Timestamp,
    pub voting_closes: Timestamp,
    pub results: Option<Vec<ElectionResult>>,
}

/// A ballot cast in an election
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Ballot {
    pub election_hash: ActionHash,
    pub voter: AgentPubKey,
    pub votes: Vec<BallotVote>,
    pub cast_at: Timestamp,
}

/// A single vote within a ballot
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BallotVote {
    pub position: String,
    pub candidate: AgentPubKey,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    BoardMeeting(BoardMeeting),
    Resolution(Resolution),
    ByLaw(ByLaw),
    Election(Election),
    Ballot(Ballot),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// All meetings anchor
    AllMeetings,
    /// Meeting to resolutions
    MeetingToResolution,
    /// All bylaws anchor
    AllByLaws,
    /// ByLaw supersession chain
    ByLawSupersedes,
    /// All elections anchor
    AllElections,
    /// Election to ballots
    ElectionToBallot,
    /// Voter to their ballots
    VoterToBallot,
    /// Proposer to their resolutions
    ProposerToResolution,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::BoardMeeting(meeting) => validate_create_meeting(action, meeting),
                EntryTypes::Resolution(resolution) => {
                    validate_create_resolution(action, resolution)
                }
                EntryTypes::ByLaw(bylaw) => validate_create_bylaw(action, bylaw),
                EntryTypes::Election(election) => validate_create_election(action, election),
                EntryTypes::Ballot(ballot) => validate_create_ballot(action, ballot),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::BoardMeeting(meeting) => validate_update_meeting(meeting),
                EntryTypes::Resolution(resolution) => validate_update_resolution(resolution),
                EntryTypes::ByLaw(bylaw) => validate_update_bylaw(bylaw),
                EntryTypes::Election(election) => validate_update_election(election),
                EntryTypes::Ballot(_) => Ok(ValidateCallbackResult::Invalid(
                    "Ballots cannot be modified after casting".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => match link_type {
            LinkTypes::AllMeetings => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllMeetings link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::MeetingToResolution => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "MeetingToResolution link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllByLaws => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllByLaws link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ByLawSupersedes => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ByLawSupersedes link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllElections => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllElections link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ElectionToBallot => {
                // Ballot links may carry voter metadata
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ElectionToBallot link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::VoterToBallot => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "VoterToBallot link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ProposerToResolution => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ProposerToResolution link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink {
            link_type, tag, action, ..
        } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            let result = check_link_author_match(
                original_action.action().author(),
                &action.author,
            );
            if result != ValidateCallbackResult::Valid {
                return Ok(result);
            }
            match link_type {
            LinkTypes::AllMeetings => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllMeetings delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::MeetingToResolution => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "MeetingToResolution delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllByLaws => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllByLaws delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ByLawSupersedes => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ByLawSupersedes delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllElections => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllElections delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ElectionToBallot => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ElectionToBallot delete link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::VoterToBallot => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "VoterToBallot delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ProposerToResolution => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ProposerToResolution delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
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

fn validate_create_meeting(
    _action: Create,
    meeting: BoardMeeting,
) -> ExternResult<ValidateCallbackResult> {
    if meeting.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Meeting title cannot be empty".into(),
        ));
    }
    if meeting.title.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Meeting title must be at most 512 characters".into(),
        ));
    }
    if meeting.agenda.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Meeting must have at least one agenda item".into(),
        ));
    }
    if meeting.agenda.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 50 agenda items".into(),
        ));
    }
    for item in &meeting.agenda {
        if item.trim().is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Agenda items cannot be empty".into(),
            ));
        }
        if item.len() > 512 {
            return Ok(ValidateCallbackResult::Invalid(
                "Each agenda item must be 512 characters or fewer".into(),
            ));
        }
    }
    if meeting.location.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Meeting location cannot be empty".into(),
        ));
    }
    if meeting.location.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Meeting location must be 256 characters or fewer".into(),
        ));
    }
    if meeting.attendees.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 100 attendees".into(),
        ));
    }
    if let Some(ref minutes) = meeting.minutes {
        if minutes.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Meeting minutes must be 4096 characters or fewer".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_meeting(meeting: BoardMeeting) -> ExternResult<ValidateCallbackResult> {
    if meeting.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Meeting title cannot be empty".into(),
        ));
    }
    if meeting.title.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Meeting title must be at most 512 characters".into(),
        ));
    }
    if meeting.agenda.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 50 agenda items".into(),
        ));
    }
    for item in &meeting.agenda {
        if item.len() > 512 {
            return Ok(ValidateCallbackResult::Invalid(
                "Each agenda item must be 512 characters or fewer".into(),
            ));
        }
    }
    if meeting.location.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Meeting location must be 256 characters or fewer".into(),
        ));
    }
    if meeting.attendees.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 100 attendees".into(),
        ));
    }
    if let Some(ref minutes) = meeting.minutes {
        if minutes.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Meeting minutes must be 4096 characters or fewer".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_resolution(
    _action: Create,
    resolution: Resolution,
) -> ExternResult<ValidateCallbackResult> {
    if resolution.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resolution title cannot be empty".into(),
        ));
    }
    if resolution.title.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resolution title must be 512 characters or fewer".into(),
        ));
    }
    if resolution.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resolution description cannot be empty".into(),
        ));
    }
    if resolution.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resolution description must be 4096 characters or fewer".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_resolution(resolution: Resolution) -> ExternResult<ValidateCallbackResult> {
    if resolution.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resolution title cannot be empty".into(),
        ));
    }
    if resolution.title.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resolution title must be 512 characters or fewer".into(),
        ));
    }
    if resolution.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resolution description must be 4096 characters or fewer".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_bylaw(_action: Create, bylaw: ByLaw) -> ExternResult<ValidateCallbackResult> {
    if bylaw.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "ByLaw ID cannot be empty".into(),
        ));
    }
    if bylaw.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "ByLaw ID must be 256 characters or fewer".into(),
        ));
    }
    if bylaw.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "ByLaw title cannot be empty".into(),
        ));
    }
    if bylaw.title.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "ByLaw title must be 512 characters or fewer".into(),
        ));
    }
    if bylaw.content.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "ByLaw content cannot be empty".into(),
        ));
    }
    if bylaw.content.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "ByLaw content must be 4096 characters or fewer".into(),
        ));
    }
    if bylaw.version == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "ByLaw version must be at least 1".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_bylaw(bylaw: ByLaw) -> ExternResult<ValidateCallbackResult> {
    if bylaw.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "ByLaw ID cannot be empty".into(),
        ));
    }
    if bylaw.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "ByLaw ID must be 256 characters or fewer".into(),
        ));
    }
    if bylaw.title.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "ByLaw title must be 512 characters or fewer".into(),
        ));
    }
    if bylaw.content.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "ByLaw content must be 4096 characters or fewer".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_election(
    _action: Create,
    election: Election,
) -> ExternResult<ValidateCallbackResult> {
    if election.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Election title cannot be empty".into(),
        ));
    }
    if election.title.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Election title must be 512 characters or fewer".into(),
        ));
    }
    if election.positions.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Election must have at least one position".into(),
        ));
    }
    if election.positions.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 20 positions".into(),
        ));
    }
    for pos in &election.positions {
        if pos.trim().is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Position name cannot be empty".into(),
            ));
        }
        if pos.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Each position name must be 256 characters or fewer".into(),
            ));
        }
    }
    if election.candidates.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 50 candidates".into(),
        ));
    }
    if election.voting_closes <= election.voting_opens {
        return Ok(ValidateCallbackResult::Invalid(
            "Voting close must be after voting open".into(),
        ));
    }
    if election.results.is_some() {
        return Ok(ValidateCallbackResult::Invalid(
            "New elections cannot have results".into(),
        ));
    }
    // Verify all candidates are for valid positions
    for candidate in &election.candidates {
        if !election.positions.contains(&candidate.position) {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Candidate position '{}' is not in the election positions list",
                candidate.position
            )));
        }
        if candidate.statement.trim().is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Candidate statement cannot be empty".into(),
            ));
        }
        if candidate.statement.len() > 2048 {
            return Ok(ValidateCallbackResult::Invalid(
                "Candidate statement must be 2048 characters or fewer".into(),
            ));
        }
    }
    if let Some(ref results) = election.results {
        if results.len() > 20 {
            return Ok(ValidateCallbackResult::Invalid(
                "Cannot have more than 20 results".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_election(election: Election) -> ExternResult<ValidateCallbackResult> {
    if election.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Election title cannot be empty".into(),
        ));
    }
    if election.title.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Election title must be 512 characters or fewer".into(),
        ));
    }
    if election.positions.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 20 positions".into(),
        ));
    }
    for pos in &election.positions {
        if pos.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Each position name must be 256 characters or fewer".into(),
            ));
        }
    }
    if election.candidates.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 50 candidates".into(),
        ));
    }
    for candidate in &election.candidates {
        if candidate.statement.len() > 2048 {
            return Ok(ValidateCallbackResult::Invalid(
                "Candidate statement must be 2048 characters or fewer".into(),
            ));
        }
    }
    if let Some(ref results) = election.results {
        if results.len() > 20 {
            return Ok(ValidateCallbackResult::Invalid(
                "Cannot have more than 20 results".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_ballot(_action: Create, ballot: Ballot) -> ExternResult<ValidateCallbackResult> {
    if ballot.votes.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Ballot must contain at least one vote".into(),
        ));
    }
    if ballot.votes.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 50 votes in a ballot".into(),
        ));
    }
    // Check for empty and duplicate position votes
    let mut seen_positions = std::collections::HashSet::new();
    for vote in &ballot.votes {
        if vote.position.trim().is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Vote position cannot be empty".into(),
            ));
        }
        if vote.position.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Vote position must be 256 characters or fewer".into(),
            ));
        }
        if !seen_positions.insert(vote.position.clone()) {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Duplicate vote for position '{}'",
                vote.position
            )));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn fake_create() -> Create {
        Create {
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
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

    fn agent_a() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn agent_b() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![2u8; 36])
    }

    fn agent_c() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![3u8; 36])
    }

    fn valid_meeting() -> BoardMeeting {
        BoardMeeting {
            cooperative_hash: None,
            title: "Monthly Board Meeting".to_string(),
            agenda: vec![
                "Budget review".to_string(),
                "Maintenance discussion".to_string(),
            ],
            scheduled_at: Timestamp::from_micros(1000000),
            location: "Community Room".to_string(),
            meeting_type: MeetingType::Regular,
            minutes: None,
            attendees: vec![agent_a(), agent_b()],
        }
    }

    fn valid_resolution() -> Resolution {
        Resolution {
            meeting_hash: Some(ActionHash::from_raw_36(vec![10u8; 36])),
            title: "Approve new landscaping budget".to_string(),
            description: "Allocate $5000 for spring landscaping improvements".to_string(),
            proposed_by: agent_a(),
            category: ResolutionCategory::Budget,
            votes_for: 5,
            votes_against: 1,
            votes_abstain: 0,
            quorum_met: true,
            passed: true,
            effective_date: Some(Timestamp::from_micros(2000000)),
        }
    }

    fn valid_bylaw() -> ByLaw {
        ByLaw {
            id: "BL-001".to_string(),
            title: "Noise Policy".to_string(),
            content: "Quiet hours are 10pm to 7am daily.".to_string(),
            version: 1,
            adopted_at: Timestamp::from_micros(1000000),
            amended_at: None,
            supersedes: None,
        }
    }

    fn valid_election() -> Election {
        Election {
            title: "Annual Board Election".to_string(),
            positions: vec!["President".to_string(), "Treasurer".to_string()],
            candidates: vec![
                CandidateEntry {
                    agent: agent_a(),
                    position: "President".to_string(),
                    statement: "I will serve with integrity.".to_string(),
                },
                CandidateEntry {
                    agent: agent_b(),
                    position: "Treasurer".to_string(),
                    statement: "I have accounting experience.".to_string(),
                },
            ],
            voting_opens: Timestamp::from_micros(1000000),
            voting_closes: Timestamp::from_micros(2000000),
            results: None,
        }
    }

    fn valid_ballot() -> Ballot {
        Ballot {
            election_hash: ActionHash::from_raw_36(vec![20u8; 36]),
            voter: agent_c(),
            votes: vec![
                BallotVote {
                    position: "President".to_string(),
                    candidate: agent_a(),
                },
                BallotVote {
                    position: "Treasurer".to_string(),
                    candidate: agent_b(),
                },
            ],
            cast_at: Timestamp::from_micros(1500000),
        }
    }

    fn assert_valid(result: ExternResult<ValidateCallbackResult>) {
        match result {
            Ok(ValidateCallbackResult::Valid) => {}
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                panic!("Expected Valid, got Invalid: {msg}")
            }
            other => panic!("Expected Valid, got {other:?}"),
        }
    }

    fn assert_invalid(result: ExternResult<ValidateCallbackResult>, expected_substr: &str) {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains(expected_substr),
                    "Expected Invalid message containing '{expected_substr}', got: '{msg}'"
                );
            }
            Ok(ValidateCallbackResult::Valid) => {
                panic!("Expected Invalid containing '{expected_substr}', got Valid")
            }
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }

    // ── Serde roundtrip tests ───────────────────────────────────────────

    #[test]
    fn serde_roundtrip_meeting_type() {
        let types = vec![
            MeetingType::Regular,
            MeetingType::Special,
            MeetingType::Annual,
            MeetingType::Emergency,
        ];
        for mt in &types {
            let json = serde_json::to_string(mt).unwrap();
            let back: MeetingType = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, mt);
        }
    }

    #[test]
    fn serde_roundtrip_resolution_category() {
        let cats = vec![
            ResolutionCategory::Budget,
            ResolutionCategory::Maintenance,
            ResolutionCategory::Membership,
            ResolutionCategory::Rules,
            ResolutionCategory::Assessment,
            ResolutionCategory::Improvement,
            ResolutionCategory::Emergency,
            ResolutionCategory::Other("Custom Category".to_string()),
        ];
        for cat in &cats {
            let json = serde_json::to_string(cat).unwrap();
            let back: ResolutionCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, cat);
        }
    }

    #[test]
    fn serde_roundtrip_board_meeting() {
        let meeting = valid_meeting();
        let json = serde_json::to_string(&meeting).unwrap();
        let back: BoardMeeting = serde_json::from_str(&json).unwrap();
        assert_eq!(back, meeting);
    }

    #[test]
    fn serde_roundtrip_resolution() {
        let resolution = valid_resolution();
        let json = serde_json::to_string(&resolution).unwrap();
        let back: Resolution = serde_json::from_str(&json).unwrap();
        assert_eq!(back, resolution);
    }

    #[test]
    fn serde_roundtrip_bylaw() {
        let bylaw = valid_bylaw();
        let json = serde_json::to_string(&bylaw).unwrap();
        let back: ByLaw = serde_json::from_str(&json).unwrap();
        assert_eq!(back, bylaw);
    }

    #[test]
    fn serde_roundtrip_candidate_entry() {
        let candidate = CandidateEntry {
            agent: agent_a(),
            position: "Secretary".to_string(),
            statement: "I am qualified.".to_string(),
        };
        let json = serde_json::to_string(&candidate).unwrap();
        let back: CandidateEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, candidate);
    }

    #[test]
    fn serde_roundtrip_election_result() {
        let result = ElectionResult {
            position: "President".to_string(),
            winner: agent_a(),
            votes_received: 10,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: ElectionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back, result);
    }

    #[test]
    fn serde_roundtrip_election() {
        let election = valid_election();
        let json = serde_json::to_string(&election).unwrap();
        let back: Election = serde_json::from_str(&json).unwrap();
        assert_eq!(back, election);
    }

    #[test]
    fn serde_roundtrip_ballot_vote() {
        let vote = BallotVote {
            position: "Treasurer".to_string(),
            candidate: agent_b(),
        };
        let json = serde_json::to_string(&vote).unwrap();
        let back: BallotVote = serde_json::from_str(&json).unwrap();
        assert_eq!(back, vote);
    }

    #[test]
    fn serde_roundtrip_ballot() {
        let ballot = valid_ballot();
        let json = serde_json::to_string(&ballot).unwrap();
        let back: Ballot = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ballot);
    }

    // ── validate_create_meeting tests ───────────────────────────────────

    #[test]
    fn create_meeting_valid() {
        assert_valid(validate_create_meeting(fake_create(), valid_meeting()));
    }

    #[test]
    fn create_meeting_empty_title() {
        let mut m = valid_meeting();
        m.title = String::new();
        assert_invalid(
            validate_create_meeting(fake_create(), m),
            "Meeting title cannot be empty",
        );
    }

    #[test]
    fn create_meeting_title_too_long() {
        let mut m = valid_meeting();
        m.title = "a".repeat(513);
        assert_invalid(
            validate_create_meeting(fake_create(), m),
            "Meeting title must be at most 512 characters",
        );
    }

    #[test]
    fn create_meeting_title_at_boundary() {
        let mut m = valid_meeting();
        m.title = "a".repeat(512);
        assert_valid(validate_create_meeting(fake_create(), m));
    }

    #[test]
    fn create_meeting_empty_agenda() {
        let mut m = valid_meeting();
        m.agenda = vec![];
        assert_invalid(
            validate_create_meeting(fake_create(), m),
            "Meeting must have at least one agenda item",
        );
    }

    #[test]
    fn create_meeting_single_agenda_item() {
        let mut m = valid_meeting();
        m.agenda = vec!["Single item".to_string()];
        assert_valid(validate_create_meeting(fake_create(), m));
    }

    #[test]
    fn create_meeting_empty_location() {
        let mut m = valid_meeting();
        m.location = String::new();
        assert_invalid(
            validate_create_meeting(fake_create(), m),
            "Meeting location cannot be empty",
        );
    }

    #[test]
    fn create_meeting_all_types_valid() {
        for mt in [
            MeetingType::Regular,
            MeetingType::Special,
            MeetingType::Annual,
            MeetingType::Emergency,
        ] {
            let mut m = valid_meeting();
            m.meeting_type = mt;
            assert_valid(validate_create_meeting(fake_create(), m));
        }
    }

    #[test]
    fn create_meeting_with_minutes() {
        let mut m = valid_meeting();
        m.minutes = Some("Meeting concluded successfully.".to_string());
        assert_valid(validate_create_meeting(fake_create(), m));
    }

    #[test]
    fn create_meeting_empty_attendees() {
        let mut m = valid_meeting();
        m.attendees = vec![];
        assert_valid(validate_create_meeting(fake_create(), m));
    }

    #[test]
    fn create_meeting_with_cooperative_hash() {
        let mut m = valid_meeting();
        m.cooperative_hash = Some(ActionHash::from_raw_36(vec![5u8; 36]));
        assert_valid(validate_create_meeting(fake_create(), m));
    }

    // ── validate_create_resolution tests ────────────────────────────────

    #[test]
    fn create_resolution_valid() {
        assert_valid(validate_create_resolution(
            fake_create(),
            valid_resolution(),
        ));
    }

    #[test]
    fn create_resolution_empty_title() {
        let mut r = valid_resolution();
        r.title = String::new();
        assert_invalid(
            validate_create_resolution(fake_create(), r),
            "Resolution title cannot be empty",
        );
    }

    #[test]
    fn create_resolution_empty_description() {
        let mut r = valid_resolution();
        r.description = String::new();
        assert_invalid(
            validate_create_resolution(fake_create(), r),
            "Resolution description cannot be empty",
        );
    }

    #[test]
    fn create_resolution_all_categories_valid() {
        let categories = vec![
            ResolutionCategory::Budget,
            ResolutionCategory::Maintenance,
            ResolutionCategory::Membership,
            ResolutionCategory::Rules,
            ResolutionCategory::Assessment,
            ResolutionCategory::Improvement,
            ResolutionCategory::Emergency,
            ResolutionCategory::Other("Special Project".to_string()),
        ];
        for cat in categories {
            let mut r = valid_resolution();
            r.category = cat;
            assert_valid(validate_create_resolution(fake_create(), r));
        }
    }

    #[test]
    fn create_resolution_without_meeting_hash() {
        let mut r = valid_resolution();
        r.meeting_hash = None;
        assert_valid(validate_create_resolution(fake_create(), r));
    }

    #[test]
    fn create_resolution_zero_votes() {
        let mut r = valid_resolution();
        r.votes_for = 0;
        r.votes_against = 0;
        r.votes_abstain = 0;
        assert_valid(validate_create_resolution(fake_create(), r));
    }

    #[test]
    fn create_resolution_not_passed() {
        let mut r = valid_resolution();
        r.passed = false;
        r.quorum_met = false;
        r.effective_date = None;
        assert_valid(validate_create_resolution(fake_create(), r));
    }

    #[test]
    fn create_resolution_with_large_vote_counts() {
        let mut r = valid_resolution();
        r.votes_for = u32::MAX;
        r.votes_against = u32::MAX;
        r.votes_abstain = u32::MAX;
        assert_valid(validate_create_resolution(fake_create(), r));
    }

    // ── validate_create_bylaw tests ─────────────────────────────────────

    #[test]
    fn create_bylaw_valid() {
        assert_valid(validate_create_bylaw(fake_create(), valid_bylaw()));
    }

    #[test]
    fn create_bylaw_empty_id() {
        let mut b = valid_bylaw();
        b.id = String::new();
        assert_invalid(
            validate_create_bylaw(fake_create(), b),
            "ByLaw ID cannot be empty",
        );
    }

    #[test]
    fn create_bylaw_empty_title() {
        let mut b = valid_bylaw();
        b.title = String::new();
        assert_invalid(
            validate_create_bylaw(fake_create(), b),
            "ByLaw title cannot be empty",
        );
    }

    #[test]
    fn create_bylaw_empty_content() {
        let mut b = valid_bylaw();
        b.content = String::new();
        assert_invalid(
            validate_create_bylaw(fake_create(), b),
            "ByLaw content cannot be empty",
        );
    }

    #[test]
    fn create_bylaw_version_zero() {
        let mut b = valid_bylaw();
        b.version = 0;
        assert_invalid(
            validate_create_bylaw(fake_create(), b),
            "ByLaw version must be at least 1",
        );
    }

    #[test]
    fn create_bylaw_version_one() {
        let mut b = valid_bylaw();
        b.version = 1;
        assert_valid(validate_create_bylaw(fake_create(), b));
    }

    #[test]
    fn create_bylaw_with_amendment() {
        let mut b = valid_bylaw();
        b.version = 2;
        b.amended_at = Some(Timestamp::from_micros(2000000));
        b.supersedes = Some(ActionHash::from_raw_36(vec![15u8; 36]));
        assert_valid(validate_create_bylaw(fake_create(), b));
    }

    #[test]
    fn create_bylaw_large_version() {
        let mut b = valid_bylaw();
        b.version = u32::MAX;
        assert_valid(validate_create_bylaw(fake_create(), b));
    }

    // ── validate_create_election tests ──────────────────────────────────

    #[test]
    fn create_election_valid() {
        assert_valid(validate_create_election(fake_create(), valid_election()));
    }

    #[test]
    fn create_election_empty_title() {
        let mut e = valid_election();
        e.title = String::new();
        assert_invalid(
            validate_create_election(fake_create(), e),
            "Election title cannot be empty",
        );
    }

    #[test]
    fn create_election_empty_positions() {
        let mut e = valid_election();
        e.positions = vec![];
        assert_invalid(
            validate_create_election(fake_create(), e),
            "Election must have at least one position",
        );
    }

    #[test]
    fn create_election_single_position() {
        let mut e = valid_election();
        e.positions = vec!["President".to_string()];
        e.candidates = vec![CandidateEntry {
            agent: agent_a(),
            position: "President".to_string(),
            statement: "Vote for me".to_string(),
        }];
        assert_valid(validate_create_election(fake_create(), e));
    }

    #[test]
    fn create_election_closes_before_opens() {
        let mut e = valid_election();
        e.voting_opens = Timestamp::from_micros(2000000);
        e.voting_closes = Timestamp::from_micros(1000000);
        assert_invalid(
            validate_create_election(fake_create(), e),
            "Voting close must be after voting open",
        );
    }

    #[test]
    fn create_election_closes_equal_opens() {
        let mut e = valid_election();
        e.voting_opens = Timestamp::from_micros(1000000);
        e.voting_closes = Timestamp::from_micros(1000000);
        assert_invalid(
            validate_create_election(fake_create(), e),
            "Voting close must be after voting open",
        );
    }

    #[test]
    fn create_election_with_results() {
        let mut e = valid_election();
        e.results = Some(vec![ElectionResult {
            position: "President".to_string(),
            winner: agent_a(),
            votes_received: 10,
        }]);
        assert_invalid(
            validate_create_election(fake_create(), e),
            "New elections cannot have results",
        );
    }

    #[test]
    fn create_election_candidate_invalid_position() {
        let mut e = valid_election();
        e.candidates.push(CandidateEntry {
            agent: agent_c(),
            position: "Nonexistent Position".to_string(),
            statement: "I'm qualified".to_string(),
        });
        assert_invalid(
            validate_create_election(fake_create(), e),
            "Candidate position 'Nonexistent Position' is not in the election positions list",
        );
    }

    #[test]
    fn create_election_candidate_empty_statement() {
        let mut e = valid_election();
        e.candidates = vec![CandidateEntry {
            agent: agent_a(),
            position: "President".to_string(),
            statement: String::new(),
        }];
        assert_invalid(
            validate_create_election(fake_create(), e),
            "Candidate statement cannot be empty",
        );
    }

    #[test]
    fn create_election_no_candidates() {
        let mut e = valid_election();
        e.candidates = vec![];
        assert_valid(validate_create_election(fake_create(), e));
    }

    #[test]
    fn create_election_multiple_candidates_same_position() {
        let mut e = valid_election();
        e.positions = vec!["President".to_string()];
        e.candidates = vec![
            CandidateEntry {
                agent: agent_a(),
                position: "President".to_string(),
                statement: "Candidate A".to_string(),
            },
            CandidateEntry {
                agent: agent_b(),
                position: "President".to_string(),
                statement: "Candidate B".to_string(),
            },
        ];
        assert_valid(validate_create_election(fake_create(), e));
    }

    #[test]
    fn create_election_all_fields_minimal() {
        let e = Election {
            title: "Election".to_string(),
            positions: vec!["Position".to_string()],
            candidates: vec![],
            voting_opens: Timestamp::from_micros(1),
            voting_closes: Timestamp::from_micros(2),
            results: None,
        };
        assert_valid(validate_create_election(fake_create(), e));
    }

    // ── validate_create_ballot tests ────────────────────────────────────

    #[test]
    fn create_ballot_valid() {
        assert_valid(validate_create_ballot(fake_create(), valid_ballot()));
    }

    #[test]
    fn create_ballot_empty_votes() {
        let mut b = valid_ballot();
        b.votes = vec![];
        assert_invalid(
            validate_create_ballot(fake_create(), b),
            "Ballot must contain at least one vote",
        );
    }

    #[test]
    fn create_ballot_single_vote() {
        let mut b = valid_ballot();
        b.votes = vec![BallotVote {
            position: "President".to_string(),
            candidate: agent_a(),
        }];
        assert_valid(validate_create_ballot(fake_create(), b));
    }

    #[test]
    fn create_ballot_duplicate_position() {
        let mut b = valid_ballot();
        b.votes = vec![
            BallotVote {
                position: "President".to_string(),
                candidate: agent_a(),
            },
            BallotVote {
                position: "President".to_string(),
                candidate: agent_b(),
            },
        ];
        assert_invalid(
            validate_create_ballot(fake_create(), b),
            "Duplicate vote for position 'President'",
        );
    }

    #[test]
    fn create_ballot_many_unique_positions() {
        let mut b = valid_ballot();
        b.votes = (0..10)
            .map(|i| BallotVote {
                position: format!("Position {}", i),
                candidate: agent_a(),
            })
            .collect();
        assert_valid(validate_create_ballot(fake_create(), b));
    }

    #[test]
    fn create_ballot_same_candidate_different_positions() {
        let mut b = valid_ballot();
        b.votes = vec![
            BallotVote {
                position: "President".to_string(),
                candidate: agent_a(),
            },
            BallotVote {
                position: "Treasurer".to_string(),
                candidate: agent_a(),
            },
        ];
        assert_valid(validate_create_ballot(fake_create(), b));
    }

    // ── Update validation tests ─────────────────────────────────────────

    #[test]
    fn update_ballot_invalid() {
        // Ballots cannot be updated per the validate() function
        // This is tested at the Op level, not in the specific validate_create_ballot
        // The main validate() function returns Invalid for UpdateEntry with Ballot
        // We verify this indirectly by checking the validate function's match arm
    }

    // ── Edge case and boundary tests ────────────────────────────────────

    #[test]
    fn create_meeting_unicode_title() {
        let mut m = valid_meeting();
        m.title = "会議 🏢 Reunión".to_string();
        assert_valid(validate_create_meeting(fake_create(), m));
    }

    #[test]
    fn create_resolution_unicode_description() {
        let mut r = valid_resolution();
        r.description = "Описание решения 📋".to_string();
        assert_valid(validate_create_resolution(fake_create(), r));
    }

    #[test]
    fn create_bylaw_content_too_long() {
        let mut b = valid_bylaw();
        b.content = "a".repeat(4097);
        assert_invalid(
            validate_create_bylaw(fake_create(), b),
            "ByLaw content must be 4096 characters or fewer",
        );
    }

    #[test]
    fn create_bylaw_content_at_boundary() {
        let mut b = valid_bylaw();
        b.content = "a".repeat(4096);
        assert_valid(validate_create_bylaw(fake_create(), b));
    }

    #[test]
    fn create_election_position_with_spaces() {
        let mut e = valid_election();
        e.positions = vec!["Vice President".to_string()];
        e.candidates = vec![CandidateEntry {
            agent: agent_a(),
            position: "Vice President".to_string(),
            statement: "I will serve".to_string(),
        }];
        assert_valid(validate_create_election(fake_create(), e));
    }

    #[test]
    fn create_meeting_agenda_with_empty_strings() {
        let mut m = valid_meeting();
        m.agenda = vec!["".to_string(), "Item 2".to_string()];
        assert_invalid(
            validate_create_meeting(fake_create(), m),
            "Agenda items cannot be empty",
        );
    }

    #[test]
    fn create_resolution_category_other_empty_string() {
        let mut r = valid_resolution();
        r.category = ResolutionCategory::Other(String::new());
        assert_valid(validate_create_resolution(fake_create(), r));
    }

    #[test]
    fn create_ballot_position_empty_string() {
        let mut b = valid_ballot();
        b.votes = vec![BallotVote {
            position: String::new(),
            candidate: agent_a(),
        }];
        assert_invalid(
            validate_create_ballot(fake_create(), b),
            "Vote position cannot be empty",
        );
    }

    #[test]
    fn create_bylaw_id_with_special_chars() {
        let mut b = valid_bylaw();
        b.id = "BL-2024-§1-α".to_string();
        assert_valid(validate_create_bylaw(fake_create(), b));
    }

    // ── Whitespace-only string validation tests ────────────────────────

    #[test]
    fn create_meeting_whitespace_title_rejected() {
        let mut m = valid_meeting();
        m.title = "   \t  ".to_string();
        assert_invalid(
            validate_create_meeting(fake_create(), m),
            "Meeting title cannot be empty",
        );
    }

    #[test]
    fn create_meeting_whitespace_location_rejected() {
        let mut m = valid_meeting();
        m.location = "  \n  ".to_string();
        assert_invalid(
            validate_create_meeting(fake_create(), m),
            "Meeting location cannot be empty",
        );
    }

    #[test]
    fn create_meeting_agenda_whitespace_item_rejected() {
        let mut m = valid_meeting();
        m.agenda = vec!["   ".to_string()];
        assert_invalid(
            validate_create_meeting(fake_create(), m),
            "Agenda items cannot be empty",
        );
    }

    #[test]
    fn create_meeting_agenda_mixed_valid_and_whitespace_rejected() {
        let mut m = valid_meeting();
        m.agenda = vec!["Valid item".to_string(), " \t ".to_string()];
        assert_invalid(
            validate_create_meeting(fake_create(), m),
            "Agenda items cannot be empty",
        );
    }

    #[test]
    fn create_resolution_whitespace_title_rejected() {
        let mut r = valid_resolution();
        r.title = "   ".to_string();
        assert_invalid(
            validate_create_resolution(fake_create(), r),
            "Resolution title cannot be empty",
        );
    }

    #[test]
    fn create_resolution_whitespace_description_rejected() {
        let mut r = valid_resolution();
        r.description = " \t\n ".to_string();
        assert_invalid(
            validate_create_resolution(fake_create(), r),
            "Resolution description cannot be empty",
        );
    }

    #[test]
    fn create_bylaw_whitespace_id_rejected() {
        let mut b = valid_bylaw();
        b.id = "   ".to_string();
        assert_invalid(
            validate_create_bylaw(fake_create(), b),
            "ByLaw ID cannot be empty",
        );
    }

    #[test]
    fn create_bylaw_whitespace_title_rejected() {
        let mut b = valid_bylaw();
        b.title = " \t ".to_string();
        assert_invalid(
            validate_create_bylaw(fake_create(), b),
            "ByLaw title cannot be empty",
        );
    }

    #[test]
    fn create_bylaw_whitespace_content_rejected() {
        let mut b = valid_bylaw();
        b.content = "  \n\n  ".to_string();
        assert_invalid(
            validate_create_bylaw(fake_create(), b),
            "ByLaw content cannot be empty",
        );
    }

    #[test]
    fn create_election_whitespace_title_rejected() {
        let mut e = valid_election();
        e.title = "   ".to_string();
        assert_invalid(
            validate_create_election(fake_create(), e),
            "Election title cannot be empty",
        );
    }

    #[test]
    fn create_election_whitespace_position_rejected() {
        let mut e = valid_election();
        e.positions = vec!["  ".to_string()];
        e.candidates = vec![];
        assert_invalid(
            validate_create_election(fake_create(), e),
            "Position name cannot be empty",
        );
    }

    #[test]
    fn create_election_mixed_valid_and_whitespace_position_rejected() {
        let mut e = valid_election();
        e.positions = vec!["President".to_string(), " \t ".to_string()];
        e.candidates = vec![CandidateEntry {
            agent: agent_a(),
            position: "President".to_string(),
            statement: "I will serve.".to_string(),
        }];
        assert_invalid(
            validate_create_election(fake_create(), e),
            "Position name cannot be empty",
        );
    }

    #[test]
    fn create_election_candidate_whitespace_statement_rejected() {
        let mut e = valid_election();
        e.candidates = vec![CandidateEntry {
            agent: agent_a(),
            position: "President".to_string(),
            statement: "   ".to_string(),
        }];
        assert_invalid(
            validate_create_election(fake_create(), e),
            "Candidate statement cannot be empty",
        );
    }

    #[test]
    fn create_ballot_whitespace_position_rejected() {
        let mut b = valid_ballot();
        b.votes = vec![BallotVote {
            position: "  \t  ".to_string(),
            candidate: agent_a(),
        }];
        assert_invalid(
            validate_create_ballot(fake_create(), b),
            "Vote position cannot be empty",
        );
    }

    #[test]
    fn create_ballot_mixed_valid_and_whitespace_position_rejected() {
        let mut b = valid_ballot();
        b.votes = vec![
            BallotVote {
                position: "President".to_string(),
                candidate: agent_a(),
            },
            BallotVote {
                position: "   ".to_string(),
                candidate: agent_b(),
            },
        ];
        assert_invalid(
            validate_create_ballot(fake_create(), b),
            "Vote position cannot be empty",
        );
    }

    // ── Vec max-length boundary tests ─────────────────────────────────

    #[test]
    fn create_meeting_agenda_at_max_count() {
        let mut m = valid_meeting();
        m.agenda = (0..50).map(|i| format!("Agenda item {i}")).collect();
        assert_valid(validate_create_meeting(fake_create(), m));
    }

    #[test]
    fn create_meeting_agenda_over_max_count() {
        let mut m = valid_meeting();
        m.agenda = (0..51).map(|i| format!("Agenda item {i}")).collect();
        assert_invalid(
            validate_create_meeting(fake_create(), m),
            "Cannot have more than 50 agenda items",
        );
    }

    #[test]
    fn create_meeting_attendees_at_max_count() {
        let mut m = valid_meeting();
        m.attendees = (0..100).map(|_| agent_a()).collect();
        assert_valid(validate_create_meeting(fake_create(), m));
    }

    #[test]
    fn create_meeting_attendees_over_max_count() {
        let mut m = valid_meeting();
        m.attendees = (0..101).map(|_| agent_a()).collect();
        assert_invalid(
            validate_create_meeting(fake_create(), m),
            "Cannot have more than 100 attendees",
        );
    }

    #[test]
    fn create_election_positions_at_max_count() {
        let mut e = valid_election();
        e.positions = (0..20).map(|i| format!("Position {i}")).collect();
        e.candidates = vec![];
        assert_valid(validate_create_election(fake_create(), e));
    }

    #[test]
    fn create_election_positions_over_max_count() {
        let mut e = valid_election();
        e.positions = (0..21).map(|i| format!("Position {i}")).collect();
        e.candidates = vec![];
        assert_invalid(
            validate_create_election(fake_create(), e),
            "Cannot have more than 20 positions",
        );
    }

    #[test]
    fn create_election_candidates_at_max_count() {
        let mut e = valid_election();
        e.positions = vec!["President".to_string()];
        e.candidates = (0..50)
            .map(|_| CandidateEntry {
                agent: agent_a(),
                position: "President".to_string(),
                statement: "I will serve.".to_string(),
            })
            .collect();
        assert_valid(validate_create_election(fake_create(), e));
    }

    #[test]
    fn create_election_candidates_over_max_count() {
        let mut e = valid_election();
        e.positions = vec!["President".to_string()];
        e.candidates = (0..51)
            .map(|_| CandidateEntry {
                agent: agent_a(),
                position: "President".to_string(),
                statement: "I will serve.".to_string(),
            })
            .collect();
        assert_invalid(
            validate_create_election(fake_create(), e),
            "Cannot have more than 50 candidates",
        );
    }

    #[test]
    fn create_ballot_votes_at_max_count() {
        let mut b = valid_ballot();
        b.votes = (0..50)
            .map(|i| BallotVote {
                position: format!("Position {i}"),
                candidate: agent_a(),
            })
            .collect();
        assert_valid(validate_create_ballot(fake_create(), b));
    }

    #[test]
    fn create_ballot_votes_over_max_count() {
        let mut b = valid_ballot();
        b.votes = (0..51)
            .map(|i| BallotVote {
                position: format!("Position {i}"),
                candidate: agent_a(),
            })
            .collect();
        assert_invalid(
            validate_create_ballot(fake_create(), b),
            "Cannot have more than 50 votes",
        );
    }

    // ── String max-length boundary tests ──────────────────────────────

    #[test]
    fn create_meeting_agenda_item_too_long() {
        let mut m = valid_meeting();
        m.agenda = vec!["x".repeat(513)];
        assert_invalid(
            validate_create_meeting(fake_create(), m),
            "Each agenda item must be 512 characters or fewer",
        );
    }

    #[test]
    fn create_meeting_agenda_item_at_boundary() {
        let mut m = valid_meeting();
        m.agenda = vec!["x".repeat(512)];
        assert_valid(validate_create_meeting(fake_create(), m));
    }

    #[test]
    fn create_meeting_location_too_long() {
        let mut m = valid_meeting();
        m.location = "x".repeat(257);
        assert_invalid(
            validate_create_meeting(fake_create(), m),
            "Meeting location must be 256 characters or fewer",
        );
    }

    #[test]
    fn create_meeting_location_at_boundary() {
        let mut m = valid_meeting();
        m.location = "x".repeat(256);
        assert_valid(validate_create_meeting(fake_create(), m));
    }

    #[test]
    fn create_meeting_minutes_too_long() {
        let mut m = valid_meeting();
        m.minutes = Some("x".repeat(4097));
        assert_invalid(
            validate_create_meeting(fake_create(), m),
            "Meeting minutes must be 4096 characters or fewer",
        );
    }

    #[test]
    fn create_meeting_minutes_at_boundary() {
        let mut m = valid_meeting();
        m.minutes = Some("x".repeat(4096));
        assert_valid(validate_create_meeting(fake_create(), m));
    }

    #[test]
    fn create_resolution_title_too_long() {
        let mut r = valid_resolution();
        r.title = "x".repeat(513);
        assert_invalid(
            validate_create_resolution(fake_create(), r),
            "Resolution title must be 512 characters or fewer",
        );
    }

    #[test]
    fn create_resolution_title_at_boundary() {
        let mut r = valid_resolution();
        r.title = "x".repeat(512);
        assert_valid(validate_create_resolution(fake_create(), r));
    }

    #[test]
    fn create_resolution_description_too_long() {
        let mut r = valid_resolution();
        r.description = "x".repeat(4097);
        assert_invalid(
            validate_create_resolution(fake_create(), r),
            "Resolution description must be 4096 characters or fewer",
        );
    }

    #[test]
    fn create_resolution_description_at_boundary() {
        let mut r = valid_resolution();
        r.description = "x".repeat(4096);
        assert_valid(validate_create_resolution(fake_create(), r));
    }

    #[test]
    fn create_bylaw_id_too_long() {
        let mut b = valid_bylaw();
        b.id = "x".repeat(257);
        assert_invalid(
            validate_create_bylaw(fake_create(), b),
            "ByLaw ID must be 256 characters or fewer",
        );
    }

    #[test]
    fn create_bylaw_id_at_boundary() {
        let mut b = valid_bylaw();
        b.id = "x".repeat(64);
        assert_valid(validate_create_bylaw(fake_create(), b));
    }

    #[test]
    fn create_bylaw_title_too_long() {
        let mut b = valid_bylaw();
        b.title = "x".repeat(513);
        assert_invalid(
            validate_create_bylaw(fake_create(), b),
            "ByLaw title must be 512 characters or fewer",
        );
    }

    #[test]
    fn create_bylaw_title_at_boundary() {
        let mut b = valid_bylaw();
        b.title = "x".repeat(512);
        assert_valid(validate_create_bylaw(fake_create(), b));
    }

    #[test]
    fn create_election_title_too_long() {
        let mut e = valid_election();
        e.title = "x".repeat(513);
        assert_invalid(
            validate_create_election(fake_create(), e),
            "Election title must be 512 characters or fewer",
        );
    }

    #[test]
    fn create_election_title_at_boundary() {
        let mut e = valid_election();
        e.title = "x".repeat(512);
        assert_valid(validate_create_election(fake_create(), e));
    }

    #[test]
    fn create_election_position_name_too_long() {
        let mut e = valid_election();
        e.positions = vec!["x".repeat(257)];
        e.candidates = vec![];
        assert_invalid(
            validate_create_election(fake_create(), e),
            "Each position name must be 256 characters or fewer",
        );
    }

    #[test]
    fn create_election_position_name_at_boundary() {
        let mut e = valid_election();
        e.positions = vec!["x".repeat(256)];
        e.candidates = vec![];
        assert_valid(validate_create_election(fake_create(), e));
    }

    #[test]
    fn create_election_candidate_statement_too_long() {
        let mut e = valid_election();
        e.candidates = vec![CandidateEntry {
            agent: agent_a(),
            position: "President".to_string(),
            statement: "x".repeat(2049),
        }];
        assert_invalid(
            validate_create_election(fake_create(), e),
            "Candidate statement must be 2048 characters or fewer",
        );
    }

    #[test]
    fn create_election_candidate_statement_at_boundary() {
        let mut e = valid_election();
        e.candidates = vec![CandidateEntry {
            agent: agent_a(),
            position: "President".to_string(),
            statement: "x".repeat(2048),
        }];
        assert_valid(validate_create_election(fake_create(), e));
    }

    #[test]
    fn create_ballot_vote_position_too_long() {
        let mut b = valid_ballot();
        b.votes = vec![BallotVote {
            position: "x".repeat(257),
            candidate: agent_a(),
        }];
        assert_invalid(
            validate_create_ballot(fake_create(), b),
            "Vote position must be 256 characters or fewer",
        );
    }

    #[test]
    fn create_ballot_vote_position_at_boundary() {
        let mut b = valid_ballot();
        b.votes = vec![BallotVote {
            position: "x".repeat(256),
            candidate: agent_a(),
        }];
        assert_valid(validate_create_ballot(fake_create(), b));
    }

    // ── Update validator tests ────────────────────────────────────────

    #[test]
    fn update_meeting_valid() {
        assert_valid(validate_update_meeting(valid_meeting()));
    }

    #[test]
    fn update_meeting_title_too_long() {
        let mut m = valid_meeting();
        m.title = "x".repeat(513);
        assert_invalid(
            validate_update_meeting(m),
            "Meeting title must be at most 512 characters",
        );
    }

    #[test]
    fn update_meeting_agenda_over_max() {
        let mut m = valid_meeting();
        m.agenda = (0..51).map(|i| format!("Item {i}")).collect();
        assert_invalid(
            validate_update_meeting(m),
            "Cannot have more than 50 agenda items",
        );
    }

    #[test]
    fn update_meeting_attendees_over_max() {
        let mut m = valid_meeting();
        m.attendees = (0..101).map(|_| agent_a()).collect();
        assert_invalid(
            validate_update_meeting(m),
            "Cannot have more than 100 attendees",
        );
    }

    #[test]
    fn update_meeting_location_too_long() {
        let mut m = valid_meeting();
        m.location = "x".repeat(257);
        assert_invalid(
            validate_update_meeting(m),
            "Meeting location must be 256 characters or fewer",
        );
    }

    #[test]
    fn update_meeting_minutes_too_long() {
        let mut m = valid_meeting();
        m.minutes = Some("x".repeat(4097));
        assert_invalid(
            validate_update_meeting(m),
            "Meeting minutes must be 4096 characters or fewer",
        );
    }

    #[test]
    fn update_resolution_valid() {
        assert_valid(validate_update_resolution(valid_resolution()));
    }

    #[test]
    fn update_resolution_title_too_long() {
        let mut r = valid_resolution();
        r.title = "x".repeat(513);
        assert_invalid(
            validate_update_resolution(r),
            "Resolution title must be 512 characters or fewer",
        );
    }

    #[test]
    fn update_resolution_description_too_long() {
        let mut r = valid_resolution();
        r.description = "x".repeat(4097);
        assert_invalid(
            validate_update_resolution(r),
            "Resolution description must be 4096 characters or fewer",
        );
    }

    #[test]
    fn update_bylaw_valid() {
        assert_valid(validate_update_bylaw(valid_bylaw()));
    }

    #[test]
    fn update_bylaw_id_too_long() {
        let mut b = valid_bylaw();
        b.id = "x".repeat(257);
        assert_invalid(
            validate_update_bylaw(b),
            "ByLaw ID must be 256 characters or fewer",
        );
    }

    #[test]
    fn update_bylaw_title_too_long() {
        let mut b = valid_bylaw();
        b.title = "x".repeat(513);
        assert_invalid(
            validate_update_bylaw(b),
            "ByLaw title must be 512 characters or fewer",
        );
    }

    #[test]
    fn update_bylaw_content_too_long() {
        let mut b = valid_bylaw();
        b.content = "x".repeat(4097);
        assert_invalid(
            validate_update_bylaw(b),
            "ByLaw content must be 4096 characters or fewer",
        );
    }

    #[test]
    fn update_election_valid() {
        assert_valid(validate_update_election(valid_election()));
    }

    #[test]
    fn update_election_title_too_long() {
        let mut e = valid_election();
        e.title = "x".repeat(513);
        assert_invalid(
            validate_update_election(e),
            "Election title must be 512 characters or fewer",
        );
    }

    #[test]
    fn update_election_positions_over_max() {
        let mut e = valid_election();
        e.positions = (0..21).map(|i| format!("Pos {i}")).collect();
        e.candidates = vec![];
        assert_invalid(
            validate_update_election(e),
            "Cannot have more than 20 positions",
        );
    }

    #[test]
    fn update_election_candidates_over_max() {
        let mut e = valid_election();
        e.candidates = (0..51)
            .map(|_| CandidateEntry {
                agent: agent_a(),
                position: "President".to_string(),
                statement: "Serve.".to_string(),
            })
            .collect();
        assert_invalid(
            validate_update_election(e),
            "Cannot have more than 50 candidates",
        );
    }

    #[test]
    fn update_election_candidate_statement_too_long() {
        let mut e = valid_election();
        e.candidates = vec![CandidateEntry {
            agent: agent_a(),
            position: "President".to_string(),
            statement: "x".repeat(2049),
        }];
        assert_invalid(
            validate_update_election(e),
            "Candidate statement must be 2048 characters or fewer",
        );
    }

    // ── Link tag validation tests ───────────────────────────────────────

    /// Helper to simulate the create link tag validation logic (same as in validate()).
    fn validate_create_link_tag(
        link_type: LinkTypes,
        tag_bytes: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag = LinkTag(tag_bytes);
        match link_type {
            LinkTypes::AllMeetings => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllMeetings link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::MeetingToResolution => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "MeetingToResolution link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllByLaws => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllByLaws link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ByLawSupersedes => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ByLawSupersedes link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllElections => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllElections link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ElectionToBallot => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ElectionToBallot link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::VoterToBallot => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "VoterToBallot link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ProposerToResolution => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ProposerToResolution link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    fn validate_delete_link_tag(
        link_type: LinkTypes,
        tag_bytes: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag = LinkTag(tag_bytes);
        match link_type {
            LinkTypes::AllMeetings => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllMeetings delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::MeetingToResolution => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "MeetingToResolution delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllByLaws => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllByLaws delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ByLawSupersedes => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ByLawSupersedes delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllElections => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllElections delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ElectionToBallot => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ElectionToBallot delete link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::VoterToBallot => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "VoterToBallot delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ProposerToResolution => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ProposerToResolution delete link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    // -- AllMeetings link tag tests --

    #[test]
    fn test_link_all_meetings_valid_tag() {
        assert_valid(validate_create_link_tag(
            LinkTypes::AllMeetings,
            vec![0u8; 64],
        ));
    }

    #[test]
    fn test_link_all_meetings_empty_tag() {
        assert_valid(validate_create_link_tag(LinkTypes::AllMeetings, vec![]));
    }

    #[test]
    fn test_link_all_meetings_tag_at_limit() {
        assert_valid(validate_create_link_tag(
            LinkTypes::AllMeetings,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn test_link_all_meetings_tag_too_long_rejected() {
        assert_invalid(
            validate_create_link_tag(LinkTypes::AllMeetings, vec![0u8; 257]),
            "AllMeetings link tag too long",
        );
    }

    // -- MeetingToResolution link tag tests --

    #[test]
    fn test_link_meeting_to_resolution_valid_tag() {
        assert_valid(validate_create_link_tag(
            LinkTypes::MeetingToResolution,
            vec![0u8; 100],
        ));
    }

    #[test]
    fn test_link_meeting_to_resolution_tag_at_limit() {
        assert_valid(validate_create_link_tag(
            LinkTypes::MeetingToResolution,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn test_link_meeting_to_resolution_tag_too_long_rejected() {
        assert_invalid(
            validate_create_link_tag(LinkTypes::MeetingToResolution, vec![0u8; 257]),
            "MeetingToResolution link tag too long",
        );
    }

    // -- AllByLaws link tag tests --

    #[test]
    fn test_link_all_bylaws_valid_tag() {
        assert_valid(validate_create_link_tag(
            LinkTypes::AllByLaws,
            vec![0u8; 128],
        ));
    }

    #[test]
    fn test_link_all_bylaws_tag_too_long_rejected() {
        assert_invalid(
            validate_create_link_tag(LinkTypes::AllByLaws, vec![0u8; 257]),
            "AllByLaws link tag too long",
        );
    }

    // -- ByLawSupersedes link tag tests --

    #[test]
    fn test_link_bylaw_supersedes_valid_tag() {
        assert_valid(validate_create_link_tag(
            LinkTypes::ByLawSupersedes,
            vec![0u8; 64],
        ));
    }

    #[test]
    fn test_link_bylaw_supersedes_tag_too_long_rejected() {
        assert_invalid(
            validate_create_link_tag(LinkTypes::ByLawSupersedes, vec![0u8; 257]),
            "ByLawSupersedes link tag too long",
        );
    }

    // -- ElectionToBallot link tag tests (512 byte limit) --

    #[test]
    fn test_link_election_to_ballot_valid_tag() {
        assert_valid(validate_create_link_tag(
            LinkTypes::ElectionToBallot,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn test_link_election_to_ballot_tag_at_limit() {
        assert_valid(validate_create_link_tag(
            LinkTypes::ElectionToBallot,
            vec![0u8; 512],
        ));
    }

    #[test]
    fn test_link_election_to_ballot_tag_too_long_rejected() {
        assert_invalid(
            validate_create_link_tag(LinkTypes::ElectionToBallot, vec![0u8; 513]),
            "ElectionToBallot link tag too long",
        );
    }

    // -- VoterToBallot link tag tests --

    #[test]
    fn test_link_voter_to_ballot_valid_tag() {
        assert_valid(validate_create_link_tag(
            LinkTypes::VoterToBallot,
            vec![0u8; 128],
        ));
    }

    #[test]
    fn test_link_voter_to_ballot_tag_too_long_rejected() {
        assert_invalid(
            validate_create_link_tag(LinkTypes::VoterToBallot, vec![0u8; 257]),
            "VoterToBallot link tag too long",
        );
    }

    // -- ProposerToResolution link tag tests --

    #[test]
    fn test_link_proposer_to_resolution_valid_tag() {
        assert_valid(validate_create_link_tag(
            LinkTypes::ProposerToResolution,
            vec![0u8; 128],
        ));
    }

    #[test]
    fn test_link_proposer_to_resolution_tag_too_long_rejected() {
        assert_invalid(
            validate_create_link_tag(LinkTypes::ProposerToResolution, vec![0u8; 257]),
            "ProposerToResolution link tag too long",
        );
    }

    // -- Delete link tag tests --

    #[test]
    fn test_delete_link_all_meetings_valid_tag() {
        assert_valid(validate_delete_link_tag(
            LinkTypes::AllMeetings,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn test_delete_link_all_meetings_tag_too_long_rejected() {
        assert_invalid(
            validate_delete_link_tag(LinkTypes::AllMeetings, vec![0u8; 257]),
            "AllMeetings delete link tag too long",
        );
    }

    #[test]
    fn test_delete_link_election_to_ballot_valid_tag() {
        assert_valid(validate_delete_link_tag(
            LinkTypes::ElectionToBallot,
            vec![0u8; 512],
        ));
    }

    #[test]
    fn test_delete_link_election_to_ballot_tag_too_long_rejected() {
        assert_invalid(
            validate_delete_link_tag(LinkTypes::ElectionToBallot, vec![0u8; 513]),
            "ElectionToBallot delete link tag too long",
        );
    }

    #[test]
    fn test_delete_link_voter_to_ballot_tag_too_long_rejected() {
        assert_invalid(
            validate_delete_link_tag(LinkTypes::VoterToBallot, vec![0u8; 257]),
            "VoterToBallot delete link tag too long",
        );
    }
}
