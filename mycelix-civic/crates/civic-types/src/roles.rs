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
