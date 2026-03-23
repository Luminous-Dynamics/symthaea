// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Space Module
//!
//! Space Situational Awareness client for mycelix-space cluster.
//!
//! Provides typed wrappers for calling space zome functions including
//! orbital tracking, conjunction screening, multi-party negotiation,
//! debris bounties, and trust-weighted observation fusion.
//!
//! ## Example
//!
//! ```rust
//! use mycelix_sdk::space::{SpaceClient, ScreenConjunctionRequest};
//!
//! let client = SpaceClient::new("did:mycelix:operator1");
//! assert_eq!(SpaceClient::role_name(), "space");
//!
//! let req = ScreenConjunctionRequest {
//!     primary_norad_id: 25544,
//!     secondary_norad_ids: vec![48274],
//!     hours_ahead: 72,
//!     pc_threshold: 1e-7,
//! };
//! let (fn_name, _payload) = SpaceClient::screen_conjunction_request(&req);
//! assert_eq!(fn_name, "screen_conjunction");
//! ```

use serde::{Deserialize, Serialize};

// =============================================================================
// SpaceClient
// =============================================================================

/// Client for mycelix-space cluster operations.
///
/// Lightweight struct that builds serialized request payloads for dispatching
/// via `CallTargetCell::OtherRole("space")` in Holochain, or for use in SDK
/// tests and simulations.
#[derive(Debug, Clone)]
pub struct SpaceClient {
    /// Agent decentralized identifier
    pub agent_did: String,
}

impl SpaceClient {
    /// Create a new space client for the given agent DID.
    pub fn new(agent_did: impl Into<String>) -> Self {
        Self {
            agent_did: agent_did.into(),
        }
    }

    /// Target role name for `CallTargetCell::OtherRole`.
    pub fn role_name() -> &'static str {
        "space"
    }
}

// =============================================================================
// Conjunction Screening
// =============================================================================

/// Request to screen for conjunctions between orbital objects.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScreenConjunctionRequest {
    /// NORAD catalog ID of the primary object
    pub primary_norad_id: u32,
    /// NORAD catalog IDs of secondary objects to screen against
    pub secondary_norad_ids: Vec<u32>,
    /// Screening window in hours from now
    pub hours_ahead: u64,
    /// Collision probability threshold for flagging
    pub pc_threshold: f64,
}

/// Result of a conjunction screening assessment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConjunctionAssessment {
    /// NORAD catalog ID of the primary object
    pub primary_norad_id: u32,
    /// NORAD catalog ID of the secondary object
    pub secondary_norad_id: u32,
    /// Predicted miss distance in kilometers
    pub miss_distance_km: f64,
    /// Estimated collision probability
    pub collision_probability: f64,
    /// Human-readable risk level (e.g., "Low", "Medium", "High", "Critical")
    pub risk_level: String,
    /// Relative velocity at TCA in km/s
    pub relative_velocity_kms: f64,
}

impl SpaceClient {
    // -- Screening --

    /// Build the request for conjunction screening.
    ///
    /// Zome function: `screen_conjunction`
    /// Expected response: `Vec<ConjunctionAssessment>`
    pub fn screen_conjunction_request(
        input: &ScreenConjunctionRequest,
    ) -> (&'static str, serde_json::Value) {
        (
            "screen_conjunction",
            serde_json::to_value(input).unwrap_or_default(),
        )
    }

    /// Build the request for rescreening a previously identified conjunction.
    ///
    /// Zome function: `rescreen_conjunction`
    pub fn rescreen_conjunction_request(event_hash: &str) -> (&'static str, serde_json::Value) {
        ("rescreen_conjunction", serde_json::json!(event_hash))
    }

    /// Build the request for retrieving all high-risk conjunctions.
    ///
    /// Zome function: `get_high_risk_conjunctions`
    /// Expected response: `Vec<ConjunctionAssessment>`
    pub fn get_high_risk_conjunctions_request() -> (&'static str, serde_json::Value) {
        ("get_high_risk_conjunctions", serde_json::json!(null))
    }
}

// =============================================================================
// Multi-Party Negotiation
// =============================================================================

/// Request to create a multi-party conjunction avoidance proposal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CreateProposalRequest {
    /// Unique proposal identifier
    pub proposal_id: String,
    /// ID of the conjunction event this proposal addresses
    pub conjunction_id: String,
    /// Agent public key strings of affected satellite operators
    pub affected_operators: Vec<String>,
    /// Time of closest approach (Unix timestamp)
    pub tca: i64,
    /// NORAD catalog ID of the primary object
    pub primary_norad_id: u32,
    /// NORAD catalog IDs of secondary objects
    pub secondary_norad_ids: Vec<u32>,
    /// Pre-computed maneuver options for voting
    pub maneuver_options: Vec<ManeuverOption>,
    /// Voting deadline (Unix timestamp)
    pub voting_deadline: i64,
    /// Fraction of total weight required for quorum (0.0 - 1.0)
    pub quorum_threshold: f64,
}

/// Pre-computed maneuver option included in a proposal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManeuverOption {
    /// Index of this option within the proposal
    pub option_index: u32,
    /// Human-readable description
    pub description: String,
    /// NORAD catalog ID of the object that would maneuver
    pub maneuvering_norad_id: u32,
    /// Required delta-v in m/s
    pub delta_v_ms: f64,
    /// Maneuver direction vector (unit or scaled)
    pub direction: [f64; 3],
    /// Predicted miss distance after maneuver in km
    pub post_maneuver_miss_km: f64,
    /// Predicted collision probability after maneuver
    pub post_maneuver_pc: f64,
}

/// Request to cast a vote on a conjunction avoidance proposal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VoteRequest {
    /// ActionHash of the proposal entry (as string)
    pub proposal_hash: String,
    /// Proposal identifier
    pub proposal_id: String,
    /// Index of the preferred maneuver option
    pub preferred_option_index: u32,
    /// Voter's weight (trust-weighted)
    pub vote_weight: f64,
    /// Optional justification text
    pub justification: Option<String>,
}

/// Tally result for a conjunction avoidance proposal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProposalTally {
    /// Proposal identifier
    pub proposal_id: String,
    /// Total number of votes cast
    pub total_votes: usize,
    /// Sum of all vote weights
    pub total_weight: f64,
    /// Whether quorum was reached
    pub quorum_met: bool,
    /// Per-option vote tallies
    pub option_tallies: Vec<OptionTally>,
    /// Winning option index (if quorum met)
    pub winner: Option<u32>,
    /// Proposal status (e.g., "Open", "Approved", "Rejected")
    pub status: String,
}

/// Per-option tally within a proposal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OptionTally {
    /// Maneuver option index
    pub option_index: u32,
    /// Number of votes for this option
    pub vote_count: usize,
    /// Sum of vote weights for this option
    pub total_weight: f64,
}

impl SpaceClient {
    // -- Multi-party negotiation --

    /// Build the request for creating a conjunction avoidance proposal.
    ///
    /// Zome function: `create_conjunction_proposal`
    pub fn create_conjunction_proposal_request(
        input: &CreateProposalRequest,
    ) -> (&'static str, serde_json::Value) {
        (
            "create_conjunction_proposal",
            serde_json::to_value(input).unwrap_or_default(),
        )
    }

    /// Build the request for voting on a proposal.
    ///
    /// Zome function: `vote_on_proposal`
    pub fn vote_on_proposal_request(input: &VoteRequest) -> (&'static str, serde_json::Value) {
        (
            "vote_on_proposal",
            serde_json::to_value(input).unwrap_or_default(),
        )
    }

    /// Build the request for tallying votes on a proposal.
    ///
    /// Zome function: `tally_proposal`
    /// Expected response: [`ProposalTally`]
    pub fn tally_proposal_request(proposal_hash: &str) -> (&'static str, serde_json::Value) {
        ("tally_proposal", serde_json::json!(proposal_hash))
    }

    /// Build the request for retrieving proposals for a conjunction.
    ///
    /// Zome function: `get_proposals_for_conjunction`
    pub fn get_proposals_for_conjunction_request(
        conjunction_id: &str,
    ) -> (&'static str, serde_json::Value) {
        (
            "get_proposals_for_conjunction",
            serde_json::json!(conjunction_id),
        )
    }

    /// Build the request for retrieving votes on a proposal.
    ///
    /// Zome function: `get_votes_for_proposal`
    pub fn get_votes_for_proposal_request(
        proposal_hash: &str,
    ) -> (&'static str, serde_json::Value) {
        (
            "get_votes_for_proposal",
            serde_json::json!(proposal_hash),
        )
    }
}

// =============================================================================
// Orbital Objects & Observations
// =============================================================================

impl SpaceClient {
    /// Build the request for submitting a Two-Line Element set.
    ///
    /// Zome function: `submit_tle`
    pub fn submit_tle_request(
        norad_id: u32,
        line1: &str,
        line2: &str,
        quality: u8,
    ) -> (&'static str, serde_json::Value) {
        (
            "submit_tle",
            serde_json::json!({
                "norad_id": norad_id,
                "line1": line1,
                "line2": line2,
                "quality": quality,
            }),
        )
    }

    /// Build the request for retrieving latest TLEs for given objects.
    ///
    /// Zome function: `get_latest_tles`
    pub fn get_latest_tles_request(norad_ids: &[u32]) -> (&'static str, serde_json::Value) {
        ("get_latest_tles", serde_json::json!(norad_ids))
    }

    /// Build the request for fusing observations for an orbital object.
    ///
    /// Zome function: `fuse_observations_for_object`
    pub fn fuse_observations_request(norad_id: u32) -> (&'static str, serde_json::Value) {
        (
            "fuse_observations_for_object",
            serde_json::json!(norad_id),
        )
    }
}

// =============================================================================
// Debris Bounties
// =============================================================================

/// Request to create a debris cleanup bounty.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CreateBountyRequest {
    /// Unique bounty identifier
    pub bounty_id: String,
    /// NORAD catalog ID of the debris object
    pub debris_norad_id: u32,
    /// Justification for the bounty (e.g., Kessler risk assessment)
    pub justification: String,
    /// Bounty amount in smallest currency unit
    pub amount: u64,
    /// Currency code (e.g., "MYCEL", "SAP")
    pub currency: String,
    /// Expiry time (Unix timestamp)
    pub expires_at: i64,
}

impl SpaceClient {
    /// Build the request for creating a debris bounty.
    ///
    /// Zome function: `create_bounty`
    pub fn create_bounty_request(input: &CreateBountyRequest) -> (&'static str, serde_json::Value) {
        (
            "create_bounty",
            serde_json::to_value(input).unwrap_or_default(),
        )
    }

    /// Build the request for retrieving all active bounties.
    ///
    /// Zome function: `get_active_bounties`
    pub fn get_active_bounties_request() -> (&'static str, serde_json::Value) {
        ("get_active_bounties", serde_json::json!(null))
    }

    /// Build the request for retrieving bounties for a specific debris object.
    ///
    /// Zome function: `get_bounties_for_debris`
    pub fn get_bounties_for_debris_request(norad_id: u32) -> (&'static str, serde_json::Value) {
        ("get_bounties_for_debris", serde_json::json!(norad_id))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_space_client_creation() {
        let client = SpaceClient::new("did:mycelix:test");
        assert_eq!(client.agent_did, "did:mycelix:test");
        assert_eq!(SpaceClient::role_name(), "space");
    }

    #[test]
    fn test_screen_conjunction_request() {
        let req = ScreenConjunctionRequest {
            primary_norad_id: 25544,
            secondary_norad_ids: vec![48274],
            hours_ahead: 72,
            pc_threshold: 1e-7,
        };
        let (fn_name, payload) = SpaceClient::screen_conjunction_request(&req);
        assert_eq!(fn_name, "screen_conjunction");
        assert!(payload.get("primary_norad_id").is_some());
    }

    #[test]
    fn test_rescreen_conjunction_request() {
        let (fn_name, _) = SpaceClient::rescreen_conjunction_request("hash456");
        assert_eq!(fn_name, "rescreen_conjunction");
    }

    #[test]
    fn test_get_high_risk_conjunctions_request() {
        let (fn_name, _) = SpaceClient::get_high_risk_conjunctions_request();
        assert_eq!(fn_name, "get_high_risk_conjunctions");
    }

    #[test]
    fn test_create_proposal_request() {
        let req = CreateProposalRequest {
            proposal_id: "prop-1".into(),
            conjunction_id: "conj-1".into(),
            affected_operators: vec!["op1".into(), "op2".into()],
            tca: 1_000_000,
            primary_norad_id: 25544,
            secondary_norad_ids: vec![48274],
            maneuver_options: vec![ManeuverOption {
                option_index: 0,
                description: "Raise orbit".into(),
                maneuvering_norad_id: 25544,
                delta_v_ms: 0.5,
                direction: [0.0, 0.0, 1.0],
                post_maneuver_miss_km: 5.0,
                post_maneuver_pc: 1e-8,
            }],
            voting_deadline: 2_000_000,
            quorum_threshold: 0.6,
        };
        let (fn_name, _) = SpaceClient::create_conjunction_proposal_request(&req);
        assert_eq!(fn_name, "create_conjunction_proposal");
    }

    #[test]
    fn test_vote_request() {
        let req = VoteRequest {
            proposal_hash: "hash123".into(),
            proposal_id: "prop-1".into(),
            preferred_option_index: 0,
            vote_weight: 0.8,
            justification: Some("Best option".into()),
        };
        let (fn_name, payload) = SpaceClient::vote_on_proposal_request(&req);
        assert_eq!(fn_name, "vote_on_proposal");
        assert_eq!(payload.get("vote_weight").unwrap().as_f64().unwrap(), 0.8);
    }

    #[test]
    fn test_tally_request() {
        let (fn_name, _) = SpaceClient::tally_proposal_request("hash123");
        assert_eq!(fn_name, "tally_proposal");
    }

    #[test]
    fn test_proposals_for_conjunction_request() {
        let (fn_name, _) = SpaceClient::get_proposals_for_conjunction_request("conj-1");
        assert_eq!(fn_name, "get_proposals_for_conjunction");
    }

    #[test]
    fn test_votes_for_proposal_request() {
        let (fn_name, _) = SpaceClient::get_votes_for_proposal_request("hash123");
        assert_eq!(fn_name, "get_votes_for_proposal");
    }

    #[test]
    fn test_submit_tle_request() {
        let (fn_name, payload) = SpaceClient::submit_tle_request(
            25544,
            "1 25544U 98067A   ...",
            "2 25544  51.6455 ...",
            80,
        );
        assert_eq!(fn_name, "submit_tle");
        assert_eq!(payload.get("norad_id").unwrap().as_u64().unwrap(), 25544);
        assert_eq!(payload.get("quality").unwrap().as_u64().unwrap(), 80);
    }

    #[test]
    fn test_get_latest_tles_request() {
        let (fn_name, payload) = SpaceClient::get_latest_tles_request(&[25544, 48274]);
        assert_eq!(fn_name, "get_latest_tles");
        assert!(payload.is_array());
        assert_eq!(payload.as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_fuse_observations_request() {
        let (fn_name, payload) = SpaceClient::fuse_observations_request(25544);
        assert_eq!(fn_name, "fuse_observations_for_object");
        assert_eq!(payload.as_u64().unwrap(), 25544);
    }

    #[test]
    fn test_bounty_request() {
        let req = CreateBountyRequest {
            bounty_id: "bounty-1".into(),
            debris_norad_id: 99999,
            justification: "Kessler risk".into(),
            amount: 1000,
            currency: "MYCEL".into(),
            expires_at: 3_000_000,
        };
        let (fn_name, _) = SpaceClient::create_bounty_request(&req);
        assert_eq!(fn_name, "create_bounty");
    }

    #[test]
    fn test_get_active_bounties_request() {
        let (fn_name, _) = SpaceClient::get_active_bounties_request();
        assert_eq!(fn_name, "get_active_bounties");
    }

    #[test]
    fn test_get_bounties_for_debris_request() {
        let (fn_name, payload) = SpaceClient::get_bounties_for_debris_request(99999);
        assert_eq!(fn_name, "get_bounties_for_debris");
        assert_eq!(payload.as_u64().unwrap(), 99999);
    }

    #[test]
    fn test_serde_roundtrip_conjunction_assessment() {
        let assessment = ConjunctionAssessment {
            primary_norad_id: 25544,
            secondary_norad_id: 48274,
            miss_distance_km: 0.5,
            collision_probability: 1e-5,
            risk_level: "Medium".into(),
            relative_velocity_kms: 7.8,
        };
        let json = serde_json::to_string(&assessment).unwrap();
        let back: ConjunctionAssessment = serde_json::from_str(&json).unwrap();
        assert_eq!(assessment, back);
    }

    #[test]
    fn test_serde_roundtrip_proposal_tally() {
        let tally = ProposalTally {
            proposal_id: "prop-1".into(),
            total_votes: 5,
            total_weight: 3.8,
            quorum_met: true,
            option_tallies: vec![
                OptionTally {
                    option_index: 0,
                    vote_count: 3,
                    total_weight: 2.4,
                },
                OptionTally {
                    option_index: 1,
                    vote_count: 2,
                    total_weight: 1.4,
                },
            ],
            winner: Some(0),
            status: "Approved".into(),
        };
        let json = serde_json::to_string(&tally).unwrap();
        let back: ProposalTally = serde_json::from_str(&json).unwrap();
        assert_eq!(tally, back);
    }

    #[test]
    fn test_serde_roundtrip_screen_conjunction_request() {
        let req = ScreenConjunctionRequest {
            primary_norad_id: 25544,
            secondary_norad_ids: vec![48274, 12345],
            hours_ahead: 72,
            pc_threshold: 1e-7,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: ScreenConjunctionRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, back);
    }

    #[test]
    fn test_serde_roundtrip_create_bounty_request() {
        let req = CreateBountyRequest {
            bounty_id: "bounty-1".into(),
            debris_norad_id: 99999,
            justification: "Kessler risk".into(),
            amount: 1000,
            currency: "MYCEL".into(),
            expires_at: 3_000_000,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: CreateBountyRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, back);
    }

    #[test]
    fn test_maneuver_option_direction_finite() {
        let opt = ManeuverOption {
            option_index: 0,
            description: "test".into(),
            maneuvering_norad_id: 25544,
            delta_v_ms: 0.5,
            direction: [0.0, 0.0, 1.0],
            post_maneuver_miss_km: 5.0,
            post_maneuver_pc: 1e-8,
        };
        for &d in &opt.direction {
            assert!(d.is_finite());
        }
    }
}
