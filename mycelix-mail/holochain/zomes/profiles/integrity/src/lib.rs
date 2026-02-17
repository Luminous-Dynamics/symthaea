//! Profiles Integrity Zome for Mycelix Mail (Stub)
use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Profile {
    pub name: String,
    pub email: Option<String>,
    pub avatar_url: Option<String>,
    pub bio: Option<String>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Profile(Profile),
}

#[hdk_link_types]
pub enum LinkTypes {
    AgentToProfile,
    PathComponent,
}

#[hdk_extern]
pub fn validate(_op: Op) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}
