// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Housing Governance Integrity Zome
//! Entry types and validation for board meetings, resolutions, bylaws, and elections.

use hdi::prelude::*;

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
                EntryTypes::BoardMeeting(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Resolution(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ByLaw(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Election(_) => Ok(ValidateCallbackResult::Valid),
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
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::AllMeetings => Ok(ValidateCallbackResult::Valid),
            LinkTypes::MeetingToResolution => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AllByLaws => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ByLawSupersedes => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AllElections => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ElectionToBallot => Ok(ValidateCallbackResult::Valid),
            LinkTypes::VoterToBallot => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProposerToResolution => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_meeting(
    _action: Create,
    meeting: BoardMeeting,
) -> ExternResult<ValidateCallbackResult> {
    if meeting.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Meeting title cannot be empty".into(),
        ));
    }
    if meeting.title.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Meeting title must be at most 256 characters".into(),
        ));
    }
    if meeting.agenda.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Meeting must have at least one agenda item".into(),
        ));
    }
    if meeting.location.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Meeting location cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_resolution(
    _action: Create,
    resolution: Resolution,
) -> ExternResult<ValidateCallbackResult> {
    if resolution.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resolution title cannot be empty".into(),
        ));
    }
    if resolution.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resolution description cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_bylaw(_action: Create, bylaw: ByLaw) -> ExternResult<ValidateCallbackResult> {
    if bylaw.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "ByLaw ID cannot be empty".into(),
        ));
    }
    if bylaw.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "ByLaw title cannot be empty".into(),
        ));
    }
    if bylaw.content.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "ByLaw content cannot be empty".into(),
        ));
    }
    if bylaw.version == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "ByLaw version must be at least 1".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_election(
    _action: Create,
    election: Election,
) -> ExternResult<ValidateCallbackResult> {
    if election.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Election title cannot be empty".into(),
        ));
    }
    if election.positions.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Election must have at least one position".into(),
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
        if candidate.statement.is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Candidate statement cannot be empty".into(),
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
    // Check for duplicate position votes
    let mut seen_positions = std::collections::HashSet::new();
    for vote in &ballot.votes {
        if !seen_positions.insert(vote.position.clone()) {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Duplicate vote for position '{}'",
                vote.position
            )));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}
