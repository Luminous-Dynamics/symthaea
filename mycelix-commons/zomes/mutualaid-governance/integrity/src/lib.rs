//! Mutual Aid Governance Integrity Zome
//! Democratic decision-making for mutual aid circles.

use hdi::prelude::*;
use mutualaid_common::*;

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Proposal(Proposal),
    Vote(Vote),
    Rule(Rule),
    Member(Member),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllProposals,
    ProposalToVotes,
    AllRules,
    AllMembers,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(_op: Op) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}
