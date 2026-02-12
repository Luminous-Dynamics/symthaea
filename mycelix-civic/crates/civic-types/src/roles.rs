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

/// Bridge query for the civic cluster
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CivicQuery {
    pub domain: String,
    pub query_type: String,
    pub requester: String,
    pub params: String,
    pub result: Option<String>,
}

/// Bridge event for the civic cluster
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CivicEvent {
    pub domain: String,
    pub event_type: String,
    pub payload: String,
    pub related_hashes: Vec<String>,
}
