//! Housing Governance Coordinator Zome
//! Business logic for board meetings, resolutions, bylaws, and elections.

use hdk::prelude::*;
use housing_governance_integrity::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_constitutional, requirement_for_proposal,
    requirement_for_voting, GovernanceEligibility, GovernanceRequirement,
};
use std::collections::HashMap;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("commons_bridge", requirement, action_name)
}

/// Schedule a board meeting

fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                let latest_update = &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated meeting".into()
    )))
}

/// Propose a resolution (optionally tied to a meeting)
#[hdk_extern]
pub fn propose_resolution(resolution: Resolution) -> ExternResult<Record> {
    // Consciousness gate: Participant tier + identity >= 0.25
    let _eligibility = require_consciousness(&requirement_for_proposal(), "propose_resolution")?;

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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
    // Consciousness gate: Citizen tier + identity >= 0.25
    let _eligibility = require_consciousness(&requirement_for_voting(), "vote_on_resolution")?;

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

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated resolution".into()
    )))
}

/// Adopt a new bylaw
#[hdk_extern]
pub fn adopt_bylaw(bylaw: ByLaw) -> ExternResult<Record> {
    // Consciousness gate: Steward tier + identity >= 0.5 + community >= 0.3
    let _eligibility = require_consciousness(&requirement_for_constitutional(), "adopt_bylaw")?;

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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
    // Consciousness gate: Steward tier + identity >= 0.5 + community >= 0.3
    let _eligibility = require_consciousness(&requirement_for_constitutional(), "amend_bylaw")?;

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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created election".into()
    )))
}

/// Cast a ballot in an election
#[hdk_extern]
pub fn cast_ballot(ballot: Ballot) -> ExternResult<Record> {
    // Consciousness gate: Citizen tier + identity >= 0.25
    let _eligibility = require_consciousness(&requirement_for_voting(), "cast_ballot")?;

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
            if let Some(existing_record) = get_latest_record(existing_hash)? {
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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created ballot".into()
    )))
}

/// Tally election results
#[hdk_extern]
pub fn tally_election(election_hash: ActionHash) -> ExternResult<Record> {
    let record = get_latest_record(election_hash.clone())?.ok_or(wasm_error!(
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

        if let Some(ballot_record) = get_latest_record(ballot_hash)? {
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

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated election".into()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    // ── Coordinator struct serde roundtrips ────────────────────────────

    #[test]
    fn record_minutes_input_serde_roundtrip() {
        let agent_a = AgentPubKey::from_raw_36(vec![1u8; 36]);
        let agent_b = AgentPubKey::from_raw_36(vec![2u8; 36]);
        let input = RecordMinutesInput {
            meeting_hash: fake_action_hash(),
            minutes: "Discussed budget and maintenance schedule.".to_string(),
            attendees: vec![agent_a.clone(), agent_b.clone()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RecordMinutesInput = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.minutes,
            "Discussed budget and maintenance schedule."
        );
        assert_eq!(decoded.attendees.len(), 2);
        assert_eq!(decoded.attendees[0], agent_a);
    }

    #[test]
    fn record_minutes_input_empty_attendees_serde() {
        let input = RecordMinutesInput {
            meeting_hash: fake_action_hash(),
            minutes: "Brief meeting.".to_string(),
            attendees: vec![],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RecordMinutesInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.attendees.is_empty());
    }

    #[test]
    fn vote_on_resolution_input_serde_roundtrip() {
        let input = VoteOnResolutionInput {
            resolution_hash: fake_action_hash(),
            votes_for: 10,
            votes_against: 3,
            votes_abstain: 2,
            quorum_met: true,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: VoteOnResolutionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.votes_for, 10);
        assert_eq!(decoded.votes_against, 3);
        assert_eq!(decoded.votes_abstain, 2);
        assert!(decoded.quorum_met);
    }

    #[test]
    fn vote_on_resolution_input_no_quorum_serde() {
        let input = VoteOnResolutionInput {
            resolution_hash: fake_action_hash(),
            votes_for: 2,
            votes_against: 1,
            votes_abstain: 0,
            quorum_met: false,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: VoteOnResolutionInput = serde_json::from_str(&json).unwrap();
        assert!(!decoded.quorum_met);
    }

    #[test]
    fn vote_on_resolution_input_zero_votes_serde() {
        let input = VoteOnResolutionInput {
            resolution_hash: fake_action_hash(),
            votes_for: 0,
            votes_against: 0,
            votes_abstain: 0,
            quorum_met: false,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: VoteOnResolutionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.votes_for, 0);
        assert_eq!(decoded.votes_against, 0);
        assert_eq!(decoded.votes_abstain, 0);
    }

    #[test]
    fn amend_bylaw_input_serde_roundtrip() {
        let input = AmendByLawInput {
            original_hash: fake_action_hash(),
            new_content: "Updated quiet hours: 11pm to 6am.".to_string(),
            new_title: Some("Revised Noise Policy".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AmendByLawInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_content, "Updated quiet hours: 11pm to 6am.");
        assert_eq!(decoded.new_title, Some("Revised Noise Policy".to_string()));
    }

    #[test]
    fn amend_bylaw_input_no_new_title_serde() {
        let input = AmendByLawInput {
            original_hash: fake_action_hash(),
            new_content: "Minor wording change.".to_string(),
            new_title: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AmendByLawInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.new_title.is_none());
    }

    // ── Integrity enum serde roundtrips ────────────────────────────────

    #[test]
    fn meeting_type_all_variants_serde() {
        let variants = vec![
            MeetingType::Regular,
            MeetingType::Special,
            MeetingType::Annual,
            MeetingType::Emergency,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let decoded: MeetingType = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, v);
        }
    }

    #[test]
    fn resolution_category_all_variants_serde() {
        let variants = vec![
            ResolutionCategory::Budget,
            ResolutionCategory::Maintenance,
            ResolutionCategory::Membership,
            ResolutionCategory::Rules,
            ResolutionCategory::Assessment,
            ResolutionCategory::Improvement,
            ResolutionCategory::Emergency,
            ResolutionCategory::Other("Custom Category".to_string()),
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let decoded: ResolutionCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, v);
        }
    }

    // ── Integrity entry struct serde roundtrips ────────────────────────

    #[test]
    fn board_meeting_serde_roundtrip() {
        let meeting = BoardMeeting {
            cooperative_hash: None,
            title: "Monthly Board Meeting".to_string(),
            agenda: vec![
                "Budget review".to_string(),
                "New member applications".to_string(),
            ],
            scheduled_at: Timestamp::from_micros(1000000),
            location: "Community Room".to_string(),
            meeting_type: MeetingType::Regular,
            minutes: None,
            attendees: vec![fake_agent()],
        };
        let json = serde_json::to_string(&meeting).unwrap();
        let decoded: BoardMeeting = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, meeting);
    }

    #[test]
    fn board_meeting_with_minutes_serde_roundtrip() {
        let meeting = BoardMeeting {
            cooperative_hash: Some(fake_action_hash()),
            title: "Annual Meeting".to_string(),
            agenda: vec!["Election".to_string()],
            scheduled_at: Timestamp::from_micros(2000000),
            location: "Main Hall".to_string(),
            meeting_type: MeetingType::Annual,
            minutes: Some("Meeting concluded with elections.".to_string()),
            attendees: vec![fake_agent()],
        };
        let json = serde_json::to_string(&meeting).unwrap();
        let decoded: BoardMeeting = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, meeting);
    }

    #[test]
    fn resolution_serde_roundtrip() {
        let resolution = Resolution {
            meeting_hash: Some(fake_action_hash()),
            title: "Approve landscaping budget".to_string(),
            description: "Allocate $5000 for landscaping".to_string(),
            proposed_by: fake_agent(),
            category: ResolutionCategory::Budget,
            votes_for: 8,
            votes_against: 2,
            votes_abstain: 1,
            quorum_met: true,
            passed: true,
            effective_date: Some(Timestamp::from_micros(3000000)),
        };
        let json = serde_json::to_string(&resolution).unwrap();
        let decoded: Resolution = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, resolution);
    }

    #[test]
    fn bylaw_serde_roundtrip() {
        let bylaw = ByLaw {
            id: "BL-001".to_string(),
            title: "Noise Policy".to_string(),
            content: "Quiet hours 10pm to 7am.".to_string(),
            version: 1,
            adopted_at: Timestamp::from_micros(1000000),
            amended_at: None,
            supersedes: None,
        };
        let json = serde_json::to_string(&bylaw).unwrap();
        let decoded: ByLaw = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, bylaw);
    }

    #[test]
    fn bylaw_amended_serde_roundtrip() {
        let bylaw = ByLaw {
            id: "BL-001".to_string(),
            title: "Revised Noise Policy".to_string(),
            content: "Quiet hours 11pm to 6am.".to_string(),
            version: 2,
            adopted_at: Timestamp::from_micros(1000000),
            amended_at: Some(Timestamp::from_micros(2000000)),
            supersedes: Some(fake_action_hash()),
        };
        let json = serde_json::to_string(&bylaw).unwrap();
        let decoded: ByLaw = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, bylaw);
    }

    #[test]
    fn candidate_entry_serde_roundtrip() {
        let candidate = CandidateEntry {
            agent: fake_agent(),
            position: "President".to_string(),
            statement: "I will serve with integrity.".to_string(),
        };
        let json = serde_json::to_string(&candidate).unwrap();
        let decoded: CandidateEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, candidate);
    }

    #[test]
    fn election_result_serde_roundtrip() {
        let result = ElectionResult {
            position: "Treasurer".to_string(),
            winner: fake_agent(),
            votes_received: 15,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: ElectionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, result);
    }

    #[test]
    fn election_serde_roundtrip() {
        let election = Election {
            title: "Annual Board Election".to_string(),
            positions: vec!["President".to_string(), "Treasurer".to_string()],
            candidates: vec![CandidateEntry {
                agent: fake_agent(),
                position: "President".to_string(),
                statement: "Vote for me".to_string(),
            }],
            voting_opens: Timestamp::from_micros(1000000),
            voting_closes: Timestamp::from_micros(2000000),
            results: None,
        };
        let json = serde_json::to_string(&election).unwrap();
        let decoded: Election = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, election);
    }

    #[test]
    fn election_with_results_serde_roundtrip() {
        let election = Election {
            title: "Completed Election".to_string(),
            positions: vec!["President".to_string()],
            candidates: vec![CandidateEntry {
                agent: fake_agent(),
                position: "President".to_string(),
                statement: "I ran.".to_string(),
            }],
            voting_opens: Timestamp::from_micros(1000000),
            voting_closes: Timestamp::from_micros(2000000),
            results: Some(vec![ElectionResult {
                position: "President".to_string(),
                winner: fake_agent(),
                votes_received: 20,
            }]),
        };
        let json = serde_json::to_string(&election).unwrap();
        let decoded: Election = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, election);
    }

    #[test]
    fn ballot_vote_serde_roundtrip() {
        let vote = BallotVote {
            position: "President".to_string(),
            candidate: fake_agent(),
        };
        let json = serde_json::to_string(&vote).unwrap();
        let decoded: BallotVote = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, vote);
    }

    #[test]
    fn ballot_serde_roundtrip() {
        let ballot = Ballot {
            election_hash: fake_action_hash(),
            voter: fake_agent(),
            votes: vec![
                BallotVote {
                    position: "President".to_string(),
                    candidate: fake_agent(),
                },
                BallotVote {
                    position: "Treasurer".to_string(),
                    candidate: AgentPubKey::from_raw_36(vec![2u8; 36]),
                },
            ],
            cast_at: Timestamp::from_micros(1500000),
        };
        let json = serde_json::to_string(&ballot).unwrap();
        let decoded: Ballot = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, ballot);
    }
}
