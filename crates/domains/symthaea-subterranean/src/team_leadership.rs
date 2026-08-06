// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Byzantine-containment leadership-lease tallying.
//!
//! Peers report who they believe currently holds the team leadership lease
//! for a given term. If trusted, current peers disagree about the leader for
//! the same term -- a split-brain condition, e.g. after a partition heals
//! with two sides having elected different leaders -- team motion authority
//! must be held rather than silently following whichever vote happened to
//! arrive first. Only votes from peers [`crate::peer_trust::PeerTrustSupervisor`]
//! currently considers trusted are tallied; an untrusted or replayed vote
//! cannot manufacture a quorum or a conflict.

use crate::peer_trust::{PeerTrustPolicy, PeerTrustSupervisor};
use crate::team::AgentId;
use serde::{Deserialize, Serialize};

pub const TEAM_LEADERSHIP_SCHEMA_VERSION: u16 = 1;
pub const MAX_LEADERSHIP_VOTES: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeadershipLeaseVote {
    pub schema_version: u16,
    pub reporter: AgentId,
    pub leader: AgentId,
    pub term: u64,
    pub membership_digest: u64,
    pub epoch: u64,
    pub sequence: u64,
    pub issued_step: u64,
    pub expires_step: u64,
}

impl LeadershipLeaseVote {
    pub const fn validate(&self) -> bool {
        self.schema_version == TEAM_LEADERSHIP_SCHEMA_VERSION
            && self.issued_step <= self.expires_step
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoteRejection {
    Malformed,
    SchemaVersionMismatch,
    Superseded,
    LedgerFull,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ByzantineContainmentAuthority {
    Nominal,
    HoldForQuorum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByzantineContainmentAssessment {
    pub authority: ByzantineContainmentAuthority,
    pub current_term: u64,
    pub distinct_claimed_leaders: u8,
    pub trusted_votes: u8,
    pub quorum_leader: Option<AgentId>,
}

impl ByzantineContainmentAssessment {
    pub const fn nominal() -> Self {
        Self {
            authority: ByzantineContainmentAuthority::Nominal,
            current_term: 0,
            distinct_claimed_leaders: 0,
            trusted_votes: 0,
            quorum_leader: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TeamLeadershipPolicy {
    /// Fraction of currently-trusted peers that must have cast a fresh vote
    /// for the single agreed leader before authority is released to Nominal.
    pub quorum_fraction: f64,
}

impl Default for TeamLeadershipPolicy {
    fn default() -> Self {
        Self {
            quorum_fraction: 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TeamLeadershipSupervisor {
    schema_version: u16,
    votes: Vec<LeadershipLeaseVote>,
    last: ByzantineContainmentAssessment,
}

impl Default for TeamLeadershipSupervisor {
    fn default() -> Self {
        Self {
            schema_version: TEAM_LEADERSHIP_SCHEMA_VERSION,
            votes: Vec::new(),
            last: ByzantineContainmentAssessment::nominal(),
        }
    }
}

impl TeamLeadershipSupervisor {
    pub fn validate(&self) -> bool {
        self.schema_version == TEAM_LEADERSHIP_SCHEMA_VERSION
            && self.votes.len() <= MAX_LEADERSHIP_VOTES
            && self.votes.iter().all(LeadershipLeaseVote::validate)
    }

    /// Record a vote. Trust is not evaluated here (only the caller, at
    /// assessment time, knows the current step against which freshness and
    /// trust must be checked) -- only structural validity and per-reporter
    /// replay/supersession ordering are enforced.
    pub fn ingest(&mut self, vote: LeadershipLeaseVote) -> Result<(), VoteRejection> {
        if !vote.validate() {
            return Err(VoteRejection::Malformed);
        }
        if vote.schema_version != TEAM_LEADERSHIP_SCHEMA_VERSION {
            return Err(VoteRejection::SchemaVersionMismatch);
        }
        if let Some(existing) = self
            .votes
            .iter()
            .position(|existing| existing.reporter == vote.reporter)
        {
            let current = self.votes[existing];
            let superseded = vote.epoch < current.epoch
                || (vote.epoch == current.epoch && vote.sequence <= current.sequence);
            if superseded {
                return Err(VoteRejection::Superseded);
            }
            self.votes[existing] = vote;
            return Ok(());
        }
        if self.votes.len() >= MAX_LEADERSHIP_VOTES {
            return Err(VoteRejection::LedgerFull);
        }
        self.votes.push(vote);
        Ok(())
    }

    /// Tally fresh votes from currently-trusted reporters at the highest
    /// reported term. Multiple distinct claimed leaders (or membership
    /// views) at that term holds authority for quorum review; too few
    /// trusted, agreeing votes to positively confirm a single leader also
    /// withholds authority rather than assuming agreement from silence.
    pub fn assess(
        &mut self,
        current_step: u64,
        peer_trust: &PeerTrustSupervisor,
        trust_policy: PeerTrustPolicy,
        policy: TeamLeadershipPolicy,
    ) -> ByzantineContainmentAssessment {
        let fresh_trusted: Vec<LeadershipLeaseVote> = self
            .votes
            .iter()
            .copied()
            .filter(|vote| current_step <= vote.expires_step)
            .filter(|vote| peer_trust.is_trusted(vote.reporter, current_step, trust_policy))
            .collect();

        let current_term = fresh_trusted
            .iter()
            .map(|vote| vote.term)
            .max()
            .unwrap_or(0);
        let top_term: Vec<LeadershipLeaseVote> = fresh_trusted
            .into_iter()
            .filter(|vote| vote.term == current_term)
            .collect();

        let mut distinct_leaders: Vec<AgentId> = Vec::new();
        let mut distinct_digests: Vec<u64> = Vec::new();
        for vote in &top_term {
            if !distinct_leaders.contains(&vote.leader) {
                distinct_leaders.push(vote.leader);
            }
            if !distinct_digests.contains(&vote.membership_digest) {
                distinct_digests.push(vote.membership_digest);
            }
        }

        let trusted_peer_count = peer_trust.trusted_count(current_step, trust_policy).max(1);
        let quorum_met = distinct_leaders.len() == 1
            && distinct_digests.len() <= 1
            && (top_term.len() as f64 / trusted_peer_count as f64) >= policy.quorum_fraction;

        let authority = if top_term.is_empty() || quorum_met {
            ByzantineContainmentAuthority::Nominal
        } else {
            ByzantineContainmentAuthority::HoldForQuorum
        };

        self.last = ByzantineContainmentAssessment {
            authority,
            current_term,
            distinct_claimed_leaders: distinct_leaders.len().min(u8::MAX as usize) as u8,
            trusted_votes: top_term.len().min(u8::MAX as usize) as u8,
            quorum_leader: quorum_met.then(|| distinct_leaders[0]),
        };
        self.last
    }

    pub const fn last(&self) -> ByzantineContainmentAssessment {
        self.last
    }

    pub fn reset(&mut self) {
        self.votes.clear();
        self.last = ByzantineContainmentAssessment::nominal();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peer_trust::PeerAuthenticationAssertion;

    fn trusted_directory() -> PeerTrustSupervisor {
        let mut peer_trust = PeerTrustSupervisor::new(7);
        for agent in [2, 3] {
            peer_trust
                .ingest(PeerAuthenticationAssertion {
                    schema_version: crate::peer_trust::PEER_TRUST_SCHEMA_VERSION,
                    agent_id: AgentId::new(agent),
                    deployment_id: 7,
                    epoch: 1,
                    sequence: 1,
                    issued_step: 1,
                    expires_step: 100,
                    authentication_verified: true,
                    hardware_backed: true,
                })
                .unwrap();
        }
        peer_trust
    }

    fn vote(reporter: u64, leader: u64, term: u64) -> LeadershipLeaseVote {
        LeadershipLeaseVote {
            schema_version: TEAM_LEADERSHIP_SCHEMA_VERSION,
            reporter: AgentId::new(reporter),
            leader: AgentId::new(leader),
            term,
            membership_digest: 11,
            epoch: 1,
            sequence: 1,
            issued_step: 2,
            expires_step: 100,
        }
    }

    #[test]
    fn conflicting_leader_claims_hold_for_quorum() {
        let peer_trust = trusted_directory();
        let mut leadership = TeamLeadershipSupervisor::default();
        leadership.ingest(vote(2, 2, 4)).unwrap();
        leadership.ingest(vote(3, 3, 4)).unwrap();
        let assessment = leadership.assess(
            3,
            &peer_trust,
            PeerTrustPolicy {
                require_hardware_backed: true,
            },
            TeamLeadershipPolicy {
                quorum_fraction: 0.8,
            },
        );
        assert_eq!(
            assessment.authority,
            ByzantineContainmentAuthority::HoldForQuorum
        );
        assert_eq!(assessment.distinct_claimed_leaders, 2);
    }

    #[test]
    fn agreeing_trusted_votes_reach_quorum() {
        let peer_trust = trusted_directory();
        let mut leadership = TeamLeadershipSupervisor::default();
        leadership.ingest(vote(2, 2, 4)).unwrap();
        leadership.ingest(vote(3, 2, 4)).unwrap();
        let assessment = leadership.assess(
            3,
            &peer_trust,
            PeerTrustPolicy {
                require_hardware_backed: true,
            },
            TeamLeadershipPolicy {
                quorum_fraction: 0.8,
            },
        );
        assert_eq!(assessment.authority, ByzantineContainmentAuthority::Nominal);
        assert_eq!(assessment.quorum_leader, Some(AgentId::new(2)));
    }

    #[test]
    fn untrusted_reporter_votes_are_not_tallied() {
        let peer_trust = PeerTrustSupervisor::new(7);
        let mut leadership = TeamLeadershipSupervisor::default();
        leadership.ingest(vote(2, 2, 4)).unwrap();
        let assessment = leadership.assess(
            3,
            &peer_trust,
            PeerTrustPolicy::default(),
            TeamLeadershipPolicy::default(),
        );
        assert_eq!(assessment.authority, ByzantineContainmentAuthority::Nominal);
        assert_eq!(assessment.trusted_votes, 0);
    }
}
