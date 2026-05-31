// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Housing Governance Coordinator Zome
//! Business logic for board meetings, resolutions, bylaws, and elections.

use hdk::prelude::*;
use housing_governance_integrity::*;
use std::collections::HashMap;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Schedule a board meeting
#[hdk_extern]
pub fn schedule_meeting(meeting: BoardMeeting) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::BoardMeeting(meeting))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_meetings".to_string())))?;
    create_link(
        anchor_hash("all_meetings")?,
        action_hash.clone(),
        LinkTypes::AllMeetings,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created meeting".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordMinutesInput {
    pub meeting_hash: ActionHash,
    pub minutes: String,
    pub attendees: Vec<AgentPubKey>,
}

/// Record minutes for a meeting
#[hdk_extern]
pub fn record_minutes(input: RecordMinutesInput) -> ExternResult<Record> {
    if input.minutes.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Minutes cannot be empty".into()
        )));
    }

    let record = get(input.meeting_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Meeting not found".into())
    ))?;

    let mut meeting: BoardMeeting = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid meeting entry".into()
        )))?;

    meeting.minutes = Some(input.minutes);
    meeting.attendees = input.attendees;

    let new_hash = update_entry(input.meeting_hash, &EntryTypes::BoardMeeting(meeting))?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated meeting".into()
    )))
}

/// Propose a resolution (optionally tied to a meeting)
#[hdk_extern]
pub fn propose_resolution(resolution: Resolution) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Resolution(resolution.clone()))?;

    // Link meeting to resolution if applicable
    if let Some(meeting_hash) = resolution.meeting_hash {
        create_link(
            meeting_hash,
            action_hash.clone(),
            LinkTypes::MeetingToResolution,
            (),
        )?;
    }

    // Link proposer to resolution
    create_link(
        resolution.proposed_by,
        action_hash.clone(),
        LinkTypes::ProposerToResolution,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created resolution".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VoteOnResolutionInput {
    pub resolution_hash: ActionHash,
    pub votes_for: u32,
    pub votes_against: u32,
    pub votes_abstain: u32,
    pub quorum_met: bool,
}

/// Record votes on a resolution and determine if it passed
#[hdk_extern]
pub fn vote_on_resolution(input: VoteOnResolutionInput) -> ExternResult<Record> {
    let record = get(input.resolution_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Resolution not found".into())
    ))?;

    let mut resolution: Resolution = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid resolution entry".into()
        )))?;

    resolution.votes_for = input.votes_for;
    resolution.votes_against = input.votes_against;
    resolution.votes_abstain = input.votes_abstain;
    resolution.quorum_met = input.quorum_met;

    // Resolution passes if quorum is met and more votes for than against
    resolution.passed = input.quorum_met && input.votes_for > input.votes_against;

    if resolution.passed {
        let now = sys_time()?;
        resolution.effective_date = Some(now);
    }

    let new_hash = update_entry(input.resolution_hash, &EntryTypes::Resolution(resolution))?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated resolution".into()
    )))
}

/// Adopt a new bylaw
#[hdk_extern]
pub fn adopt_bylaw(bylaw: ByLaw) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::ByLaw(bylaw.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_bylaws".to_string())))?;
    create_link(
        anchor_hash("all_bylaws")?,
        action_hash.clone(),
        LinkTypes::AllByLaws,
        (),
    )?;

    // If this supersedes another bylaw, create a supersession link
    if let Some(superseded_hash) = bylaw.supersedes {
        create_link(
            superseded_hash,
            action_hash.clone(),
            LinkTypes::ByLawSupersedes,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created bylaw".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AmendByLawInput {
    pub original_hash: ActionHash,
    pub new_content: String,
    pub new_title: Option<String>,
}

/// Amend an existing bylaw (creates a new version)
#[hdk_extern]
pub fn amend_bylaw(input: AmendByLawInput) -> ExternResult<Record> {
    if input.new_content.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Amended content cannot be empty".into()
        )));
    }

    let record = get(input.original_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("ByLaw not found".into())))?;

    let original: ByLaw = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid bylaw entry".into()
        )))?;

    let now = sys_time()?;

    let amended = ByLaw {
        id: original.id,
        title: input.new_title.unwrap_or(original.title),
        content: input.new_content,
        version: original.version + 1,
        adopted_at: original.adopted_at,
        amended_at: Some(now),
        supersedes: Some(input.original_hash.clone()),
    };

    let action_hash = create_entry(&EntryTypes::ByLaw(amended))?;

    // Link to all bylaws
    create_entry(&EntryTypes::Anchor(Anchor("all_bylaws".to_string())))?;
    create_link(
        anchor_hash("all_bylaws")?,
        action_hash.clone(),
        LinkTypes::AllByLaws,
        (),
    )?;

    // Supersession link
    create_link(
        input.original_hash,
        action_hash.clone(),
        LinkTypes::ByLawSupersedes,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created bylaw amendment".into()
    )))
}

/// Create an election
#[hdk_extern]
pub fn create_election(election: Election) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Election(election))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_elections".to_string())))?;
    create_link(
        anchor_hash("all_elections")?,
        action_hash.clone(),
        LinkTypes::AllElections,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created election".into()
    )))
}

/// Cast a ballot in an election
#[hdk_extern]
pub fn cast_ballot(ballot: Ballot) -> ExternResult<Record> {
    // Verify the election exists and is open
    let election_record = get(ballot.election_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Election not found".into())),
    )?;

    let election: Election = election_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid election entry".into()
        )))?;

    let now = sys_time()?;
    if now < election.voting_opens {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Voting has not opened yet".into()
        )));
    }
    if now > election.voting_closes {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Voting has closed".into()
        )));
    }

    // Check voter hasn't already voted
    let voter_links = get_links(
        LinkQuery::try_new(ballot.voter.clone(), LinkTypes::VoterToBallot)?,
        GetStrategy::default(),
    )?;
    for link in &voter_links {
        let link_hash = ActionHash::try_from(link.target.clone());
        if let Ok(existing_hash) = link_hash {
            if let Some(existing_record) = get(existing_hash, GetOptions::default())? {
                if let Some(existing_ballot) = existing_record
                    .entry()
                    .to_app_option::<Ballot>()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                {
                    if existing_ballot.election_hash == ballot.election_hash {
                        return Err(wasm_error!(WasmErrorInner::Guest(
                            "Voter has already cast a ballot in this election".into()
                        )));
                    }
                }
            }
        }
    }

    // Validate all votes are for valid candidates
    for vote in &ballot.votes {
        let valid_candidate = election
            .candidates
            .iter()
            .any(|c| c.agent == vote.candidate && c.position == vote.position);
        if !valid_candidate {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Invalid candidate for position '{}'",
                vote.position
            ))));
        }
    }

    let action_hash = create_entry(&EntryTypes::Ballot(ballot.clone()))?;

    // Link election to ballot
    create_link(
        ballot.election_hash,
        action_hash.clone(),
        LinkTypes::ElectionToBallot,
        (),
    )?;

    // Link voter to ballot
    create_link(
        ballot.voter,
        action_hash.clone(),
        LinkTypes::VoterToBallot,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created ballot".into()
    )))
}

/// Tally election results
#[hdk_extern]
pub fn tally_election(election_hash: ActionHash) -> ExternResult<Record> {
    let record = get(election_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Election not found".into())
    ))?;

    let mut election: Election = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid election entry".into()
        )))?;

    let now = sys_time()?;
    if now < election.voting_closes {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot tally before voting closes".into()
        )));
    }

    if election.results.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Election has already been tallied".into()
        )));
    }

    // Collect all ballots
    let ballot_links = get_links(
        LinkQuery::try_new(election_hash.clone(), LinkTypes::ElectionToBallot)?,
        GetStrategy::default(),
    )?;

    // Tally votes: position -> candidate -> count
    let mut vote_counts: HashMap<String, HashMap<AgentPubKey, u32>> = HashMap::new();

    for link in ballot_links {
        let ballot_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(ballot_record) = get(ballot_hash, GetOptions::default())? {
            if let Some(ballot) = ballot_record
                .entry()
                .to_app_option::<Ballot>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                for vote in &ballot.votes {
                    *vote_counts
                        .entry(vote.position.clone())
                        .or_default()
                        .entry(vote.candidate.clone())
                        .or_insert(0) += 1;
                }
            }
        }
    }

    // Determine winners for each position
    let mut results = Vec::new();
    for position in &election.positions {
        if let Some(candidates) = vote_counts.get(position) {
            if let Some((winner, &votes)) = candidates.iter().max_by_key(|(_, v)| *v) {
                results.push(ElectionResult {
                    position: position.clone(),
                    winner: winner.clone(),
                    votes_received: votes,
                });
            }
        }
    }

    election.results = Some(results);

    let new_hash = update_entry(election_hash, &EntryTypes::Election(election))?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated election".into()
    )))
}
