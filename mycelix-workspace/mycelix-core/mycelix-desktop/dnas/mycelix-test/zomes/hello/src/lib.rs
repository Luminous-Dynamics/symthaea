// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdk::prelude::*;

/// Simple hello function that returns a greeting
#[hdk_extern]
pub fn hello(name: String) -> ExternResult<String> {
    Ok(format!("Hello, {}! Welcome to Mycelix! 🍄", name))
}

/// Function to get the current agent's public key
#[hdk_extern]
pub fn whoami(_: ()) -> ExternResult<AgentPubKey> {
    agent_info().map(|info| info.agent_initial_pubkey)
}

/// Simple echo function for testing
#[hdk_extern]
pub fn echo(message: String) -> ExternResult<String> {
    Ok(message)
}

/// Get agent info
#[hdk_extern]
pub fn get_agent_info(_: ()) -> ExternResult<AgentInfo> {
    agent_info()
}
