//! Profiles Coordinator Zome for Mycelix Mail (Stub)
use hdk::prelude::*;
use mail_profiles_integrity::*;

#[hdk_extern]
pub fn get_my_profile(_: ()) -> ExternResult<Option<Profile>> {
    Ok(None) // Stub implementation
}

#[hdk_extern]
pub fn set_profile(profile: Profile) -> ExternResult<ActionHash> {
    create_entry(EntryTypes::Profile(profile))
}
