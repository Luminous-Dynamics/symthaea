// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DID-centric authorization helpers

use serde::{Deserialize, Serialize};

/// Party in a civic proceeding
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Party {
    pub did: String,
    pub role: PartyRole,
}

/// Role a party plays in a civic process
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PartyRole {
    // Justice roles
    Complainant,
    Respondent,
    Arbitrator,
    Mediator,
    Enforcer,
    // Emergency roles
    Coordinator,
    Responder,
    Evacuee,
    // Media roles
    Author,
    Contributor,
    FactChecker,
    Endorser,
    Curator,
}
