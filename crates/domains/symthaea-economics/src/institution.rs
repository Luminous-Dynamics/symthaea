// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic, typed institutional-evolution state machine.
//!
//! The kernel models an institution as a versioned rule-set. Proposal, adoption,
//! implementation, and historical lineage remain distinct. Rule changes require
//! the higher-order rule currently governing the changed rule.

use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RuleLevel {
    Operational,
    CollectiveChoice,
    Constitutional,
    MetaConstitutional,
}

impl RuleLevel {
    pub fn required_authorizer(self) -> Option<Self> {
        match self {
            Self::Operational => Some(Self::CollectiveChoice),
            Self::CollectiveChoice => Some(Self::Constitutional),
            Self::Constitutional => Some(Self::MetaConstitutional),
            Self::MetaConstitutional => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticDelta {
    pub path: String,
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstitutionState {
    pub institution_id: String,
    pub semantic_version: String,
    pub parent_institution_id: Option<String>,
    pub profile_hash: String,
    pub rules: BTreeMap<RuleLevel, String>,
}

impl InstitutionState {
    pub fn rule_hash(&self, level: RuleLevel) -> Option<&str> {
        self.rules.get(&level).map(String::as_str)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutationCandidate {
    pub mutation_id: String,
    pub parent_institution_hash: String,
    pub candidate_institution_hash: String,
    pub rule_level: RuleLevel,
    pub semantic_delta: SemanticDelta,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Proposal {
    pub proposal_id: String,
    pub candidate: MutationCandidate,
    pub proposer: String,
    pub trigger: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdoptionDecision {
    pub proposal_id: String,
    pub adopted: bool,
    pub authority: Option<String>,
    pub authorizing_rule_hash: Option<String>,
    pub authorizing_rule_level: Option<RuleLevel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureDisposition {
    UnknownProposal,
    ParentMismatch,
    MissingAuthority,
    WrongAuthorityLevel,
    AuthorityRuleNotCurrent,
    SelfModificationUnauthorized,
    AlreadyImplemented,
    NotAdopted,
    MetaConstitutionalMutationDisabled,
    DuplicateProposal,
    AlreadyDecided,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstitutionLineageEvent {
    pub parent_institution_hash: String,
    pub candidate_institution_hash: String,
    pub mutation_id: String,
    pub rule_level: RuleLevel,
    pub transition: Transition,
    pub authorizing_rule_hash: Option<String>,
    pub proposer: Option<String>,
    pub authority: Option<String>,
    pub trigger: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    Proposed,
    Adopted,
    Implemented,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstitutionalEvolution {
    initial: InstitutionState,
    current: InstitutionState,
    allow_meta_constitutional_mutation: bool,
    proposals: BTreeMap<String, Proposal>,
    decisions: BTreeMap<String, AdoptionDecision>,
    lineage: Vec<InstitutionLineageEvent>,
}

impl InstitutionalEvolution {
    pub fn new(initial: InstitutionState) -> Self {
        Self {
            initial: initial.clone(),
            current: initial,
            allow_meta_constitutional_mutation: false,
            proposals: BTreeMap::new(),
            decisions: BTreeMap::new(),
            lineage: Vec::new(),
        }
    }

    pub fn with_meta_constitutional_mutation(mut self, enabled: bool) -> Self {
        self.allow_meta_constitutional_mutation = enabled;
        self
    }

    pub fn initial(&self) -> &InstitutionState {
        &self.initial
    }

    pub fn current(&self) -> &InstitutionState {
        &self.current
    }

    pub fn proposals(&self) -> &BTreeMap<String, Proposal> {
        &self.proposals
    }

    pub fn decisions(&self) -> &BTreeMap<String, AdoptionDecision> {
        &self.decisions
    }

    pub fn lineage(&self) -> &[InstitutionLineageEvent] {
        &self.lineage
    }

    pub fn propose(
        &mut self,
        proposal_id: impl Into<String>,
        candidate: MutationCandidate,
        proposer: impl Into<String>,
        trigger: impl Into<String>,
    ) -> Result<(), FailureDisposition> {
        if candidate.parent_institution_hash != self.current.profile_hash {
            return Err(FailureDisposition::ParentMismatch);
        }
        let proposal_id = proposal_id.into();
        if self.proposals.contains_key(&proposal_id) {
            return Err(FailureDisposition::DuplicateProposal);
        }
        if candidate.rule_level == RuleLevel::MetaConstitutional
            && !self.allow_meta_constitutional_mutation
        {
            return Err(FailureDisposition::MetaConstitutionalMutationDisabled);
        }

        let proposal = Proposal {
            proposal_id: proposal_id.clone(),
            candidate: candidate.clone(),
            proposer: proposer.into(),
            trigger: trigger.into(),
        };
        self.proposals.insert(proposal_id, proposal.clone());
        self.lineage.push(InstitutionLineageEvent {
            parent_institution_hash: candidate.parent_institution_hash,
            candidate_institution_hash: candidate.candidate_institution_hash,
            mutation_id: candidate.mutation_id,
            rule_level: candidate.rule_level,
            transition: Transition::Proposed,
            authorizing_rule_hash: None,
            proposer: Some(proposal.proposer),
            authority: None,
            trigger: Some(proposal.trigger),
        });
        Ok(())
    }

    pub fn decide(&mut self, decision: AdoptionDecision) -> Result<(), FailureDisposition> {
        let proposal = self
            .proposals
            .get(&decision.proposal_id)
            .ok_or(FailureDisposition::UnknownProposal)?;
        if self.decisions.contains_key(&decision.proposal_id) {
            return Err(FailureDisposition::AlreadyDecided);
        }
        if proposal.candidate.parent_institution_hash != self.current.profile_hash {
            return Err(FailureDisposition::ParentMismatch);
        }
        if decision.adopted {
            if proposal.candidate.rule_level == RuleLevel::MetaConstitutional
                && !self.allow_meta_constitutional_mutation
            {
                return Err(FailureDisposition::MetaConstitutionalMutationDisabled);
            }
            if decision.authority.is_none() {
                return Err(FailureDisposition::MissingAuthority);
            }

            if let Some(required_level) = proposal.candidate.rule_level.required_authorizer() {
                if decision.authorizing_rule_level != Some(required_level) {
                    return Err(FailureDisposition::WrongAuthorityLevel);
                }
                let Some(authorizing_hash) = decision.authorizing_rule_hash.as_deref() else {
                    return Err(FailureDisposition::MissingAuthority);
                };
                if self.current.rule_hash(required_level) != Some(authorizing_hash) {
                    return Err(FailureDisposition::AuthorityRuleNotCurrent);
                }
                if authorizing_hash == proposal.candidate.candidate_institution_hash {
                    return Err(FailureDisposition::SelfModificationUnauthorized);
                }
            }
        }

        self.decisions.insert(decision.proposal_id.clone(), decision.clone());
        self.lineage.push(InstitutionLineageEvent {
            parent_institution_hash: proposal.candidate.parent_institution_hash.clone(),
            candidate_institution_hash: proposal.candidate.candidate_institution_hash.clone(),
            mutation_id: proposal.candidate.mutation_id.clone(),
            rule_level: proposal.candidate.rule_level,
            transition: if decision.adopted { Transition::Adopted } else { Transition::Rejected },
            authorizing_rule_hash: decision.authorizing_rule_hash.clone(),
            proposer: Some(proposal.proposer.clone()),
            authority: decision.authority.clone(),
            trigger: Some(proposal.trigger.clone()),
        });
        Ok(())
    }

    pub fn implement(&mut self, proposal_id: &str) -> Result<(), FailureDisposition> {
        let proposal = self
            .proposals
            .get(proposal_id)
            .ok_or(FailureDisposition::UnknownProposal)?;
        let decision = self
            .decisions
            .get(proposal_id)
            .ok_or(FailureDisposition::NotAdopted)?;
        if !decision.adopted {
            return Err(FailureDisposition::NotAdopted);
        }
        if proposal.candidate.parent_institution_hash != self.current.profile_hash {
            return Err(FailureDisposition::ParentMismatch);
        }
        if proposal.candidate.candidate_institution_hash == self.current.profile_hash {
            return Err(FailureDisposition::AlreadyImplemented);
        }
        if self.current.rule_hash(proposal.candidate.rule_level)
            == Some(proposal.candidate.candidate_institution_hash.as_str())
        {
            return Err(FailureDisposition::AlreadyImplemented);
        }

        let mut next_rules = self.current.rules.clone();
        next_rules.insert(
            proposal.candidate.rule_level,
            proposal.candidate.candidate_institution_hash.clone(),
        );
        self.current = InstitutionState {
            institution_id: proposal.candidate.candidate_institution_hash.clone(),
            semantic_version: format!(
                "{}+{}",
                self.current.semantic_version,
                proposal.candidate.mutation_id
            ),
            parent_institution_id: Some(self.current.institution_id.clone()),
            profile_hash: proposal.candidate.candidate_institution_hash.clone(),
            rules: next_rules,
        };
        self.lineage.push(InstitutionLineageEvent {
            parent_institution_hash: proposal.candidate.parent_institution_hash.clone(),
            candidate_institution_hash: proposal.candidate.candidate_institution_hash.clone(),
            mutation_id: proposal.candidate.mutation_id.clone(),
            rule_level: proposal.candidate.rule_level,
            transition: Transition::Implemented,
            authorizing_rule_hash: decision.authorizing_rule_hash.clone(),
            proposer: Some(proposal.proposer.clone()),
            authority: decision.authority.clone(),
            trigger: Some(proposal.trigger.clone()),
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn initial() -> InstitutionState {
        InstitutionState {
            institution_id: "inst-v0".into(),
            semantic_version: "0".into(),
            parent_institution_id: None,
            profile_hash: "profile-v0".into(),
            rules: BTreeMap::from([
                (RuleLevel::Operational, "op-rule-v0".into()),
                (RuleLevel::CollectiveChoice, "cc-rule-v0".into()),
                (RuleLevel::Constitutional, "constitution-v0".into()),
                (RuleLevel::MetaConstitutional, "meta-v0".into()),
            ]),
        }
    }

    fn candidate(level: RuleLevel, hash: &str, mutation: &str) -> MutationCandidate {
        MutationCandidate {
            mutation_id: mutation.into(),
            parent_institution_hash: "profile-v0".into(),
            candidate_institution_hash: hash.into(),
            rule_level: level,
            semantic_delta: SemanticDelta {
                path: "allocation.rule".into(),
                from: "old".into(),
                to: "new".into(),
            },
        }
    }

    fn adopted_operational_decision() -> AdoptionDecision {
        AdoptionDecision {
            proposal_id: "p1".into(),
            adopted: true,
            authority: Some("authority".into()),
            authorizing_rule_hash: Some("cc-rule-v0".into()),
            authorizing_rule_level: Some(RuleLevel::CollectiveChoice),
        }
    }

    #[test]
    fn proposal_does_not_change_current_state() {
        let mut evolution = InstitutionalEvolution::new(initial());
        evolution.propose("p1", candidate(RuleLevel::Operational, "profile-v1", "m1"), "a", "failure").unwrap();
        assert_eq!(evolution.current().profile_hash, "profile-v0");
        assert_eq!(evolution.lineage()[0].transition, Transition::Proposed);
    }

    #[test]
    fn adopted_change_requires_current_higher_order_rule() {
        let mut evolution = InstitutionalEvolution::new(initial());
        evolution.propose("p1", candidate(RuleLevel::Operational, "profile-v1", "m1"), "a", "failure").unwrap();
        assert_eq!(
            evolution.decide(AdoptionDecision {
                proposal_id: "p1".into(),
                adopted: true,
                authority: Some("authority".into()),
                authorizing_rule_hash: Some("wrong-rule".into()),
                authorizing_rule_level: Some(RuleLevel::CollectiveChoice),
            }),
            Err(FailureDisposition::AuthorityRuleNotCurrent)
        );
    }

    #[test]
    fn adopted_and_implemented_are_distinct() {
        let mut evolution = InstitutionalEvolution::new(initial());
        evolution.propose("p1", candidate(RuleLevel::Operational, "profile-v1", "m1"), "a", "failure").unwrap();
        evolution.decide(adopted_operational_decision()).unwrap();
        assert_eq!(evolution.current().profile_hash, "profile-v0");
        evolution.implement("p1").unwrap();
        assert_eq!(evolution.current().profile_hash, "profile-v1");
        assert_eq!(evolution.current().rule_hash(RuleLevel::Operational), Some("profile-v1"));
        assert_eq!(evolution.current().rule_hash(RuleLevel::CollectiveChoice), Some("cc-rule-v0"));
        assert!(evolution.lineage().iter().any(|e| e.transition == Transition::Adopted));
        assert!(evolution.lineage().iter().any(|e| e.transition == Transition::Implemented));
    }

    #[test]
    fn wrong_parent_cannot_be_proposed() {
        let mut evolution = InstitutionalEvolution::new(initial());
        let mut wrong = candidate(RuleLevel::Operational, "profile-v1", "m1");
        wrong.parent_institution_hash = "other-profile".into();
        assert_eq!(
            evolution.propose("p1", wrong, "a", "failure"),
            Err(FailureDisposition::ParentMismatch)
        );
    }

    #[test]
    fn competing_proposal_becomes_stale_after_intervening_implementation() {
        let mut evolution = InstitutionalEvolution::new(initial());

        evolution.propose("p1", candidate(RuleLevel::Operational, "profile-v1", "m1"), "a", "failure").unwrap();
        evolution.propose("p2", candidate(RuleLevel::Operational, "profile-v2", "m2"), "b", "another-failure").unwrap();

        evolution.decide(adopted_operational_decision()).unwrap();
        evolution.implement("p1").unwrap();

        let decision = AdoptionDecision {
            proposal_id: "p2".into(),
            adopted: true,
            authority: Some("authority".into()),
            authorizing_rule_hash: Some("cc-rule-v0".into()),
            authorizing_rule_level: Some(RuleLevel::CollectiveChoice),
        };
        assert_eq!(evolution.decide(decision), Err(FailureDisposition::ParentMismatch));
    }

    #[test]
    fn rejected_proposal_never_becomes_current() {
        let mut evolution = InstitutionalEvolution::new(initial());
        evolution.propose("p1", candidate(RuleLevel::Operational, "profile-v1", "m1"), "a", "failure").unwrap();
        evolution.decide(AdoptionDecision {
            proposal_id: "p1".into(),
            adopted: false,
            authority: None,
            authorizing_rule_hash: None,
            authorizing_rule_level: None,
        }).unwrap();
        assert_eq!(evolution.current().profile_hash, "profile-v0");
        assert_eq!(evolution.implement("p1"), Err(FailureDisposition::NotAdopted));
    }

    #[test]
    fn duplicate_proposal_and_decision_fail_closed() {
        let mut evolution = InstitutionalEvolution::new(initial());
        let proposal = candidate(RuleLevel::Operational, "profile-v1", "m1");
        evolution.propose("p1", proposal.clone(), "a", "failure").unwrap();
        assert_eq!(
            evolution.propose("p1", proposal, "b", "other"),
            Err(FailureDisposition::DuplicateProposal)
        );
        let decision = AdoptionDecision {
            proposal_id: "p1".into(),
            adopted: false,
            authority: None,
            authorizing_rule_hash: None,
            authorizing_rule_level: None,
        };
        evolution.decide(decision.clone()).unwrap();
        assert_eq!(evolution.decide(decision), Err(FailureDisposition::AlreadyDecided));
    }

    #[test]
    fn constitutional_mutation_requires_meta_constitutional_rule() {
        let mut evolution = InstitutionalEvolution::new(initial());
        let mut candidate = candidate(RuleLevel::Constitutional, "profile-v1", "m-constitutional");
        candidate.semantic_delta.path = "amendment.threshold".into();
        evolution.propose("p1", candidate, "a", "capture").unwrap();
        assert_eq!(
            evolution.decide(AdoptionDecision {
                proposal_id: "p1".into(),
                adopted: true,
                authority: Some("authority".into()),
                authorizing_rule_hash: Some("meta-v0".into()),
                authorizing_rule_level: Some(RuleLevel::MetaConstitutional),
            }),
            Err(FailureDisposition::MetaConstitutionalMutationDisabled)
        );
    }

    #[test]
    fn self_authorizing_mutation_fails_closed() {
        let mut evolution = InstitutionalEvolution::new(initial());
        let mut candidate = candidate(RuleLevel::Operational, "cc-rule-v0", "m1");
        candidate.semantic_delta.path = "collective_choice.rule".into();
        evolution.propose("p1", candidate, "a", "capture").unwrap();
        assert_eq!(
            evolution.decide(adopted_operational_decision()),
            Err(FailureDisposition::SelfModificationUnauthorized)
        );
    }

    #[test]
    fn lineage_replay_is_deterministic_for_same_operations() {
        let mut first = InstitutionalEvolution::new(initial());
        let mut second = InstitutionalEvolution::new(initial());
        for evolution in [&mut first, &mut second] {
            evolution.propose("p1", candidate(RuleLevel::Operational, "profile-v1", "m1"), "a", "failure").unwrap();
            evolution.decide(adopted_operational_decision()).unwrap();
            evolution.implement("p1").unwrap();
        }
        assert_eq!(first.current(), second.current());
        assert_eq!(first.lineage(), second.lineage());
    }
}