// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Strategic-evidence primitives for institutional games.
//!
//! The model deliberately keeps content integrity, authorship, evidential
//! support and currentness as independent propositions. It contains no generic
//! `trusted` bit and grants no authorization. Verification/challenge are model
//! actions with explicit costs and scenario-declared observations.

use std::collections::BTreeMap;

/// Scenario-declared content-integrity state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceIntegrity {
    Unknown,
    Valid,
    Invalid,
}

/// Scenario-declared authorship/authentication state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceAuthorship {
    Unknown,
    Authenticated,
    Rejected,
}

/// Relationship between an evidence item and the claim under a declared evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceSupport {
    Supports,
    Contradicts,
    Neutral,
    Unknown,
}

/// Currentness/freshness under a declared evidence-time model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceCurrentness {
    Current,
    Stale,
    Unknown,
}

/// Scenario-declared result exposed after a challenge action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceChallengeOutcome {
    Sustained,
    Rejected,
    Inconclusive,
}

/// One evidence item in a formal institutional-game scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceItem {
    pub id: String,
    pub claim_id: String,
    pub source_id: String,
    pub integrity: EvidenceIntegrity,
    pub authorship: EvidenceAuthorship,
    pub support: EvidenceSupport,
    pub currentness: EvidenceCurrentness,
    pub challenge_outcome: EvidenceChallengeOutcome,
    pub disclosure_cost: f64,
    pub verification_cost: f64,
    pub challenge_cost: f64,
}

impl EvidenceItem {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: impl Into<String>,
        claim_id: impl Into<String>,
        source_id: impl Into<String>,
        integrity: EvidenceIntegrity,
        authorship: EvidenceAuthorship,
        support: EvidenceSupport,
        currentness: EvidenceCurrentness,
        challenge_outcome: EvidenceChallengeOutcome,
        disclosure_cost: f64,
        verification_cost: f64,
        challenge_cost: f64,
    ) -> Result<Self, String> {
        let id = id.into();
        let claim_id = claim_id.into();
        let source_id = source_id.into();
        if id.is_empty() || claim_id.is_empty() || source_id.is_empty() {
            return Err("evidence id, claim id and source id must be non-empty".to_string());
        }
        validate_cost(disclosure_cost, "disclosure cost")?;
        validate_cost(verification_cost, "verification cost")?;
        validate_cost(challenge_cost, "challenge cost")?;

        Ok(Self {
            id,
            claim_id,
            source_id,
            integrity,
            authorship,
            support,
            currentness,
            challenge_outcome,
            disclosure_cost,
            verification_cost,
            challenge_cost,
        })
    }
}

/// Strategic action affecting the decision maker's evidence view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceAction {
    Disclose(String),
    Withhold(String),
    Verify(String),
    Challenge(String),
}

/// Whether an evidence item has been intentionally disclosed or withheld.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisclosureState {
    Undisclosed,
    Withheld,
    Disclosed,
}

/// Observation state retained for one evidence item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceObservation {
    pub disclosure: DisclosureState,
    pub observed_integrity: Option<EvidenceIntegrity>,
    pub observed_challenge: Option<EvidenceChallengeOutcome>,
}

impl Default for EvidenceObservation {
    fn default() -> Self {
        Self {
            disclosure: DisclosureState::Undisclosed,
            observed_integrity: None,
            observed_challenge: None,
        }
    }
}

/// Cost ledger. Cost axes remain separate from substantive decision outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EvidenceCostLedger {
    pub disclosure_cost: f64,
    pub verification_cost: f64,
    pub challenge_cost: f64,
}

impl EvidenceCostLedger {
    pub fn total_cost(self) -> Result<f64, String> {
        let subtotal = checked_cost_add(self.disclosure_cost, self.verification_cost)?;
        checked_cost_add(subtotal, self.challenge_cost)
    }
}

/// Decision-maker view over one evidence scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceView {
    observations: BTreeMap<String, EvidenceObservation>,
    costs: EvidenceCostLedger,
}

impl EvidenceView {
    pub fn observation(&self, evidence_id: &str) -> Option<&EvidenceObservation> {
        self.observations.get(evidence_id)
    }

    pub fn costs(&self) -> EvidenceCostLedger {
        self.costs
    }
}

/// Explicit admissibility policy for evidence-sensitive decision summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceAdmissibilityPolicy {
    pub require_current: bool,
    pub require_verified_integrity: bool,
    pub require_authenticated_authorship: bool,
}

impl EvidenceAdmissibilityPolicy {
    pub const fn permissive() -> Self {
        Self {
            require_current: false,
            require_verified_integrity: false,
            require_authenticated_authorship: false,
        }
    }

    pub const fn strict_current_authenticated() -> Self {
        Self {
            require_current: true,
            require_verified_integrity: true,
            require_authenticated_authorship: true,
        }
    }
}

/// Transparent evidence-sensitive decision breakdown.
///
/// No single `trusted` or authorization verdict is produced. Consumers retain
/// the support counts, exclusions, missingness and failure states separately.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EvidenceDecisionBreakdown {
    pub supports: usize,
    pub contradicts: usize,
    pub neutral: usize,
    pub unknown_support: usize,
    pub undisclosed: usize,
    pub explicitly_withheld: usize,
    pub excluded_stale: usize,
    pub excluded_unknown_currentness: usize,
    pub excluded_integrity: usize,
    pub excluded_authorship: usize,
    pub verification_failures: usize,
    pub authorship_rejections: usize,
    pub challenges_sustained: usize,
    pub challenges_rejected: usize,
    pub challenges_inconclusive: usize,
}

/// Immutable evidence registry plus deterministic view-transition semantics.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceScenario {
    items: BTreeMap<String, EvidenceItem>,
}

impl EvidenceScenario {
    pub fn new(items: Vec<EvidenceItem>) -> Result<Self, String> {
        let mut indexed = BTreeMap::new();
        for item in items {
            let id = item.id.clone();
            if indexed.insert(id.clone(), item).is_some() {
                return Err(format!("duplicate evidence identifier: {id}"));
            }
        }
        Ok(Self { items: indexed })
    }

    pub fn initial_view(&self) -> EvidenceView {
        let observations = self
            .items
            .keys()
            .cloned()
            .map(|id| (id, EvidenceObservation::default()))
            .collect();
        EvidenceView {
            observations,
            costs: EvidenceCostLedger::default(),
        }
    }

    /// Apply one strategic evidence action to an existing view.
    ///
    /// Cost accumulation is checked before state mutation so a failed action is
    /// transactional: it leaves both observations and cost ledger unchanged.
    pub fn apply_action(
        &self,
        view: &mut EvidenceView,
        action: EvidenceAction,
    ) -> Result<(), String> {
        self.validate_view(view)?;
        let evidence_id = match &action {
            EvidenceAction::Disclose(id)
            | EvidenceAction::Withhold(id)
            | EvidenceAction::Verify(id)
            | EvidenceAction::Challenge(id) => id,
        };
        let item = self
            .items
            .get(evidence_id)
            .ok_or_else(|| format!("unknown evidence identifier: {evidence_id}"))?;

        let pending_cost = match &action {
            EvidenceAction::Disclose(_) => Some((
                CostAxis::Disclosure,
                checked_cost_add(view.costs.disclosure_cost, item.disclosure_cost)?,
            )),
            EvidenceAction::Verify(_) => Some((
                CostAxis::Verification,
                checked_cost_add(view.costs.verification_cost, item.verification_cost)?,
            )),
            EvidenceAction::Challenge(_) => Some((
                CostAxis::Challenge,
                checked_cost_add(view.costs.challenge_cost, item.challenge_cost)?,
            )),
            EvidenceAction::Withhold(_) => None,
        };

        let observation = view
            .observations
            .get_mut(evidence_id)
            .ok_or_else(|| "evidence view is missing a scenario item".to_string())?;

        match action {
            EvidenceAction::Disclose(_) => {
                if observation.disclosure == DisclosureState::Disclosed {
                    return Err("evidence is already disclosed".to_string());
                }
                observation.disclosure = DisclosureState::Disclosed;
            }
            EvidenceAction::Withhold(_) => {
                if observation.disclosure == DisclosureState::Disclosed {
                    return Err("already disclosed evidence cannot be made unseen".to_string());
                }
                observation.disclosure = DisclosureState::Withheld;
            }
            EvidenceAction::Verify(_) => {
                if observation.disclosure != DisclosureState::Disclosed {
                    return Err("evidence must be disclosed before verification".to_string());
                }
                if observation.observed_integrity.is_some() {
                    return Err("evidence integrity has already been observed".to_string());
                }
                observation.observed_integrity = Some(item.integrity);
            }
            EvidenceAction::Challenge(_) => {
                if observation.disclosure != DisclosureState::Disclosed {
                    return Err("evidence must be disclosed before challenge".to_string());
                }
                if observation.observed_challenge.is_some() {
                    return Err("evidence has already been challenged".to_string());
                }
                observation.observed_challenge = Some(item.challenge_outcome);
            }
        }

        if let Some((axis, updated_cost)) = pending_cost {
            match axis {
                CostAxis::Disclosure => view.costs.disclosure_cost = updated_cost,
                CostAxis::Verification => view.costs.verification_cost = updated_cost,
                CostAxis::Challenge => view.costs.challenge_cost = updated_cost,
            }
        }
        Ok(())
    }

    /// Summarize admissible disclosed evidence without granting authorization.
    pub fn decision_breakdown(
        &self,
        view: &EvidenceView,
        policy: EvidenceAdmissibilityPolicy,
    ) -> Result<EvidenceDecisionBreakdown, String> {
        self.validate_view(view)?;
        let mut result = EvidenceDecisionBreakdown::default();

        for (id, item) in &self.items {
            let observation = view
                .observations
                .get(id)
                .ok_or_else(|| "evidence view is missing a scenario item".to_string())?;

            match observation.disclosure {
                DisclosureState::Undisclosed => {
                    result.undisclosed += 1;
                    continue;
                }
                DisclosureState::Withheld => {
                    result.explicitly_withheld += 1;
                    continue;
                }
                DisclosureState::Disclosed => {}
            }

            if observation.observed_integrity == Some(EvidenceIntegrity::Invalid) {
                result.verification_failures += 1;
            }
            if item.authorship == EvidenceAuthorship::Rejected {
                result.authorship_rejections += 1;
            }

            match observation.observed_challenge {
                Some(EvidenceChallengeOutcome::Sustained) => result.challenges_sustained += 1,
                Some(EvidenceChallengeOutcome::Rejected) => result.challenges_rejected += 1,
                Some(EvidenceChallengeOutcome::Inconclusive) => result.challenges_inconclusive += 1,
                None => {}
            }

            if policy.require_current {
                match item.currentness {
                    EvidenceCurrentness::Current => {}
                    EvidenceCurrentness::Stale => {
                        result.excluded_stale += 1;
                        continue;
                    }
                    EvidenceCurrentness::Unknown => {
                        result.excluded_unknown_currentness += 1;
                        continue;
                    }
                }
            }

            if policy.require_verified_integrity
                && observation.observed_integrity != Some(EvidenceIntegrity::Valid)
            {
                result.excluded_integrity += 1;
                continue;
            }

            if policy.require_authenticated_authorship
                && item.authorship != EvidenceAuthorship::Authenticated
            {
                result.excluded_authorship += 1;
                continue;
            }

            match item.support {
                EvidenceSupport::Supports => result.supports += 1,
                EvidenceSupport::Contradicts => result.contradicts += 1,
                EvidenceSupport::Neutral => result.neutral += 1,
                EvidenceSupport::Unknown => result.unknown_support += 1,
            }
        }

        Ok(result)
    }

    fn validate_view(&self, view: &EvidenceView) -> Result<(), String> {
        if view.observations.len() != self.items.len()
            || !self
                .items
                .keys()
                .all(|id| view.observations.contains_key(id))
        {
            return Err("evidence view does not match scenario registry".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CostAxis {
    Disclosure,
    Verification,
    Challenge,
}

fn validate_cost(cost: f64, name: &str) -> Result<(), String> {
    if !cost.is_finite() || cost < 0.0 {
        return Err(format!("{name} must be finite and non-negative"));
    }
    Ok(())
}

fn checked_cost_add(total: f64, increment: f64) -> Result<f64, String> {
    let updated = total + increment;
    if !updated.is_finite() {
        return Err("evidence cost accumulation overflowed finite f64 range".to_string());
    }
    Ok(updated)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(
        id: &str,
        support: EvidenceSupport,
        integrity: EvidenceIntegrity,
        authorship: EvidenceAuthorship,
        currentness: EvidenceCurrentness,
    ) -> EvidenceItem {
        EvidenceItem::new(
            id,
            "claim",
            format!("source-{id}"),
            integrity,
            authorship,
            support,
            currentness,
            EvidenceChallengeOutcome::Inconclusive,
            1.0,
            2.0,
            3.0,
        )
        .unwrap()
    }

    #[test]
    fn integrity_verification_does_not_create_support() {
        let scenario = EvidenceScenario::new(vec![item(
            "e",
            EvidenceSupport::Unknown,
            EvidenceIntegrity::Valid,
            EvidenceAuthorship::Authenticated,
            EvidenceCurrentness::Current,
        )])
        .unwrap();
        let mut view = scenario.initial_view();
        scenario
            .apply_action(&mut view, EvidenceAction::Disclose("e".into()))
            .unwrap();
        scenario
            .apply_action(&mut view, EvidenceAction::Verify("e".into()))
            .unwrap();

        let result = scenario
            .decision_breakdown(&view, EvidenceAdmissibilityPolicy::strict_current_authenticated())
            .unwrap();
        assert_eq!(result.supports, 0);
        assert_eq!(result.unknown_support, 1);
    }

    #[test]
    fn authenticated_authorship_does_not_create_evidential_support() {
        let scenario = EvidenceScenario::new(vec![item(
            "e",
            EvidenceSupport::Unknown,
            EvidenceIntegrity::Unknown,
            EvidenceAuthorship::Authenticated,
            EvidenceCurrentness::Current,
        )])
        .unwrap();
        let mut view = scenario.initial_view();
        scenario
            .apply_action(&mut view, EvidenceAction::Disclose("e".into()))
            .unwrap();
        let result = scenario
            .decision_breakdown(&view, EvidenceAdmissibilityPolicy::permissive())
            .unwrap();
        assert_eq!(result.unknown_support, 1);
        assert_eq!(result.supports, 0);
    }

    #[test]
    fn stale_authenticated_integrity_valid_evidence_is_excluded_by_current_policy() {
        let scenario = EvidenceScenario::new(vec![item(
            "stale",
            EvidenceSupport::Supports,
            EvidenceIntegrity::Valid,
            EvidenceAuthorship::Authenticated,
            EvidenceCurrentness::Stale,
        )])
        .unwrap();
        let mut view = scenario.initial_view();
        scenario
            .apply_action(&mut view, EvidenceAction::Disclose("stale".into()))
            .unwrap();
        scenario
            .apply_action(&mut view, EvidenceAction::Verify("stale".into()))
            .unwrap();
        let result = scenario
            .decision_breakdown(&view, EvidenceAdmissibilityPolicy::strict_current_authenticated())
            .unwrap();
        assert_eq!(result.excluded_stale, 1);
        assert_eq!(result.supports, 0);
    }

    #[test]
    fn missing_contradiction_and_verification_failure_remain_distinct() {
        let scenario = EvidenceScenario::new(vec![
            item(
                "missing",
                EvidenceSupport::Supports,
                EvidenceIntegrity::Valid,
                EvidenceAuthorship::Authenticated,
                EvidenceCurrentness::Current,
            ),
            item(
                "contrary",
                EvidenceSupport::Contradicts,
                EvidenceIntegrity::Valid,
                EvidenceAuthorship::Authenticated,
                EvidenceCurrentness::Current,
            ),
            item(
                "bad-integrity",
                EvidenceSupport::Supports,
                EvidenceIntegrity::Invalid,
                EvidenceAuthorship::Authenticated,
                EvidenceCurrentness::Current,
            ),
        ])
        .unwrap();
        let mut view = scenario.initial_view();
        for id in ["contrary", "bad-integrity"] {
            scenario
                .apply_action(&mut view, EvidenceAction::Disclose(id.into()))
                .unwrap();
            scenario
                .apply_action(&mut view, EvidenceAction::Verify(id.into()))
                .unwrap();
        }

        let result = scenario
            .decision_breakdown(&view, EvidenceAdmissibilityPolicy::strict_current_authenticated())
            .unwrap();
        assert_eq!(result.undisclosed, 1);
        assert_eq!(result.contradicts, 1);
        assert_eq!(result.verification_failures, 1);
        assert_eq!(result.excluded_integrity, 1);
    }

    #[test]
    fn selective_disclosure_changes_view_without_changing_scenario_items() {
        let scenario = EvidenceScenario::new(vec![
            item(
                "for",
                EvidenceSupport::Supports,
                EvidenceIntegrity::Valid,
                EvidenceAuthorship::Authenticated,
                EvidenceCurrentness::Current,
            ),
            item(
                "against",
                EvidenceSupport::Contradicts,
                EvidenceIntegrity::Valid,
                EvidenceAuthorship::Authenticated,
                EvidenceCurrentness::Current,
            ),
        ])
        .unwrap();

        let mut selective = scenario.initial_view();
        scenario
            .apply_action(&mut selective, EvidenceAction::Disclose("for".into()))
            .unwrap();
        let selective_result = scenario
            .decision_breakdown(&selective, EvidenceAdmissibilityPolicy::permissive())
            .unwrap();
        assert_eq!(selective_result.supports, 1);
        assert_eq!(selective_result.contradicts, 0);
        assert_eq!(selective_result.undisclosed, 1);

        let mut full = scenario.initial_view();
        for id in ["for", "against"] {
            scenario
                .apply_action(&mut full, EvidenceAction::Disclose(id.into()))
                .unwrap();
        }
        let full_result = scenario
            .decision_breakdown(&full, EvidenceAdmissibilityPolicy::permissive())
            .unwrap();
        assert_eq!(full_result.supports, 1);
        assert_eq!(full_result.contradicts, 1);
        assert_eq!(full_result.undisclosed, 0);
    }

    #[test]
    fn evidence_cost_axes_remain_separate() {
        let scenario = EvidenceScenario::new(vec![item(
            "e",
            EvidenceSupport::Supports,
            EvidenceIntegrity::Valid,
            EvidenceAuthorship::Authenticated,
            EvidenceCurrentness::Current,
        )])
        .unwrap();
        let mut view = scenario.initial_view();
        scenario
            .apply_action(&mut view, EvidenceAction::Disclose("e".into()))
            .unwrap();
        scenario
            .apply_action(&mut view, EvidenceAction::Verify("e".into()))
            .unwrap();
        scenario
            .apply_action(&mut view, EvidenceAction::Challenge("e".into()))
            .unwrap();

        assert_eq!(
            view.costs(),
            EvidenceCostLedger {
                disclosure_cost: 1.0,
                verification_cost: 2.0,
                challenge_cost: 3.0,
            }
        );
        assert_eq!(view.costs().total_cost().unwrap(), 6.0);
    }

    #[test]
    fn failed_cost_accumulation_is_transactional() {
        let first = EvidenceItem::new(
            "first",
            "claim",
            "source-first",
            EvidenceIntegrity::Unknown,
            EvidenceAuthorship::Unknown,
            EvidenceSupport::Unknown,
            EvidenceCurrentness::Unknown,
            EvidenceChallengeOutcome::Inconclusive,
            f64::MAX,
            0.0,
            0.0,
        )
        .unwrap();
        let second = EvidenceItem::new(
            "second",
            "claim",
            "source-second",
            EvidenceIntegrity::Unknown,
            EvidenceAuthorship::Unknown,
            EvidenceSupport::Unknown,
            EvidenceCurrentness::Unknown,
            EvidenceChallengeOutcome::Inconclusive,
            f64::MAX,
            0.0,
            0.0,
        )
        .unwrap();
        let scenario = EvidenceScenario::new(vec![first, second]).unwrap();
        let mut view = scenario.initial_view();
        scenario
            .apply_action(&mut view, EvidenceAction::Disclose("first".into()))
            .unwrap();
        let before = view.clone();
        assert!(
            scenario
                .apply_action(&mut view, EvidenceAction::Disclose("second".into()))
                .is_err()
        );
        assert_eq!(view, before);
    }

    #[test]
    fn invalid_costs_duplicate_ids_and_unknown_actions_fail_closed() {
        assert!(
            EvidenceItem::new(
                "e",
                "claim",
                "source",
                EvidenceIntegrity::Unknown,
                EvidenceAuthorship::Unknown,
                EvidenceSupport::Unknown,
                EvidenceCurrentness::Unknown,
                EvidenceChallengeOutcome::Inconclusive,
                -1.0,
                0.0,
                0.0,
            )
            .is_err()
        );
        assert!(
            EvidenceItem::new(
                "e",
                "claim",
                "source",
                EvidenceIntegrity::Unknown,
                EvidenceAuthorship::Unknown,
                EvidenceSupport::Unknown,
                EvidenceCurrentness::Unknown,
                EvidenceChallengeOutcome::Inconclusive,
                0.0,
                f64::NAN,
                0.0,
            )
            .is_err()
        );

        let duplicate = item(
            "same",
            EvidenceSupport::Unknown,
            EvidenceIntegrity::Unknown,
            EvidenceAuthorship::Unknown,
            EvidenceCurrentness::Unknown,
        );
        assert!(EvidenceScenario::new(vec![duplicate.clone(), duplicate]).is_err());

        let scenario = EvidenceScenario::new(vec![]).unwrap();
        let mut view = scenario.initial_view();
        assert!(
            scenario
                .apply_action(&mut view, EvidenceAction::Disclose("missing".into()))
                .is_err()
        );
    }

    #[test]
    fn withholding_is_distinct_from_never_disclosed() {
        let scenario = EvidenceScenario::new(vec![
            item(
                "withheld",
                EvidenceSupport::Contradicts,
                EvidenceIntegrity::Valid,
                EvidenceAuthorship::Authenticated,
                EvidenceCurrentness::Current,
            ),
            item(
                "unseen",
                EvidenceSupport::Supports,
                EvidenceIntegrity::Valid,
                EvidenceAuthorship::Authenticated,
                EvidenceCurrentness::Current,
            ),
        ])
        .unwrap();
        let mut view = scenario.initial_view();
        scenario
            .apply_action(&mut view, EvidenceAction::Withhold("withheld".into()))
            .unwrap();
        let result = scenario
            .decision_breakdown(&view, EvidenceAdmissibilityPolicy::permissive())
            .unwrap();
        assert_eq!(result.explicitly_withheld, 1);
        assert_eq!(result.undisclosed, 1);
    }

    #[test]
    fn authorship_rejection_is_observable_separately() {
        let scenario = EvidenceScenario::new(vec![item(
            "e",
            EvidenceSupport::Supports,
            EvidenceIntegrity::Valid,
            EvidenceAuthorship::Rejected,
            EvidenceCurrentness::Current,
        )])
        .unwrap();
        let mut view = scenario.initial_view();
        scenario
            .apply_action(&mut view, EvidenceAction::Disclose("e".into()))
            .unwrap();
        scenario
            .apply_action(&mut view, EvidenceAction::Verify("e".into()))
            .unwrap();
        let result = scenario
            .decision_breakdown(&view, EvidenceAdmissibilityPolicy::strict_current_authenticated())
            .unwrap();
        assert_eq!(result.authorship_rejections, 1);
        assert_eq!(result.excluded_authorship, 1);
        assert_eq!(result.supports, 0);
    }
}
