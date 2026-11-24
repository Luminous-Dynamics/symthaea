// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic, fail-closed decision kernel for sequential retrieval planning.
//!
//! The caller supplies calibrated action-value estimates and an allowlisted set
//! of actions actually supported by available adapters. This module selects at
//! most one action under a cost/risk budget. It neither executes tools, learns
//! estimates, qualifies evidence, sees evaluator labels, nor authorizes actions.
//!
//! Estimates use fixed-point micro-units and integer arithmetic so identical
//! inputs produce identical decisions across runs and architectures.

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

const OFFER_DOMAIN: &[u8] = b"luminous:mace:retrieval-action-offer:v1\0";
const POLICY_DOMAIN: &[u8] = b"luminous:mace:retrieval-value-policy:v1\0";
const MAX_CONFIDENCE_Z_MILLI: u32 = 10_000;
const MAX_WEIGHT_PPM: u32 = 1_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum RetrievalActionKindV1 {
    Bm25Lexical = 1,
    DenseVector = 2,
    EntityLiteral = 3,
    TemporalState = 4,
    CausalLineage = 5,
    HdcAssociation = 6,
}

impl RetrievalActionKindV1 {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bm25Lexical => "bm25_lexical",
            Self::DenseVector => "dense_vector",
            Self::EntityLiteral => "entity_literal",
            Self::TemporalState => "temporal_state",
            Self::CausalLineage => "causal_lineage",
            Self::HdcAssociation => "hdc_association",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetrievalActionOfferV1 {
    /// One of the closed action kinds implemented by this protocol version.
    pub kind: RetrievalActionKindV1,
    /// Digest of the exact adapter implementation/configuration.
    pub tool_revision_sha256: String,
    /// Digest of the bounded, typed action template; never raw query text.
    pub action_template_sha256: String,
    /// Predicted reduction in downstream decision regret, in micro-units.
    /// Negative values are permitted and represent predicted harm.
    pub expected_regret_reduction_micros: i64,
    /// Standard-deviation estimate for predicted regret reduction.
    pub uncertainty_micros: u64,
    /// All-in incremental cost mapped to the policy's common micro-unit scale.
    pub estimated_cost_micros: u64,
    /// Risk estimate mapped to the policy's common micro-unit scale.
    pub estimated_risk_micros: u64,
}

impl RetrievalActionOfferV1 {
    /// Stable commitment to the exact tool, template and value estimates.
    pub fn offer_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(OFFER_DOMAIN);
        hasher.update([self.kind as u8]);
        hash_field(&mut hasher, &self.tool_revision_sha256);
        hash_field(&mut hasher, &self.action_template_sha256);
        hasher.update(self.expected_regret_reduction_micros.to_be_bytes());
        hasher.update(self.uncertainty_micros.to_be_bytes());
        hasher.update(self.estimated_cost_micros.to_be_bytes());
        hasher.update(self.estimated_risk_micros.to_be_bytes());
        hex_digest(hasher.finalize().as_slice())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetrievalValuePolicyV1 {
    /// One-sided confidence multiplier in thousandths (1645 is approximately 1.645).
    pub confidence_z_milli: u32,
    /// Cost penalty multiplier in parts per million, inclusive range 0..=1e6.
    pub cost_weight_ppm: u32,
    /// Risk penalty multiplier in parts per million, inclusive range 0..=1e6.
    pub risk_weight_ppm: u32,
    /// Hard upper bound; offers above it are never eligible.
    pub maximum_risk_micros: u64,
}

impl RetrievalValuePolicyV1 {
    pub fn validate(&self) -> Result<(), RetrievalControllerRejectV1> {
        if self.confidence_z_milli > MAX_CONFIDENCE_Z_MILLI
            || self.cost_weight_ppm > MAX_WEIGHT_PPM
            || self.risk_weight_ppm > MAX_WEIGHT_PPM
        {
            return Err(RetrievalControllerRejectV1::InvalidPolicy);
        }
        Ok(())
    }

    pub fn policy_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(POLICY_DOMAIN);
        hasher.update(self.confidence_z_milli.to_be_bytes());
        hasher.update(self.cost_weight_ppm.to_be_bytes());
        hasher.update(self.risk_weight_ppm.to_be_bytes());
        hasher.update(self.maximum_risk_micros.to_be_bytes());
        hex_digest(hasher.finalize().as_slice())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetrievalControllerStateV1 {
    pub remaining_tool_calls: u32,
    pub remaining_cost_micros: u64,
    /// This flag must be set only by the caller's independent sufficiency gate.
    pub qualified_evidence_sufficient: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetrievalControllerStopReasonV1 {
    EvidenceSufficient,
    ToolCallBudgetExhausted,
    CostBudgetExhausted,
    NoSafeActionAvailable,
    NonPositiveRiskAdjustedValue,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectedRetrievalActionV1 {
    pub kind: RetrievalActionKindV1,
    pub tool_revision_sha256: String,
    pub action_template_sha256: String,
    pub offer_sha256: String,
    pub policy_sha256: String,
    /// Lower confidence bound before cost and risk penalties.
    pub lower_bound_value_micros: i128,
    /// Lower confidence bound after cost and risk penalties.
    pub net_value_micros: i128,
    pub estimated_cost_micros: u64,
    pub estimated_risk_micros: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RetrievalControllerDecisionV1 {
    Execute(SelectedRetrievalActionV1),
    Stop(RetrievalControllerStopReasonV1),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetrievalControllerRejectV1 {
    InvalidPolicy,
    InvalidOfferDigest,
    DuplicateOffer,
}

/// Select the next action from offers supplied by adapters that are actually
/// available to the caller. This function does not invoke the selected adapter.
///
/// The score is:
///
///   LCB = predicted_regret_reduction - z * uncertainty
///   net = LCB - cost_weight * cost - risk_weight * risk
///
/// z is scaled by 1,000; weights are scaled by 1,000,000. All estimates must
/// already be calibrated onto the common micro-unit scale by a separate,
/// label-blind estimator. The sealed evaluator must never call this function
/// with hidden answer labels or realized benchmark correctness.
///
/// Offers are sorted by stable identity before tie-breaking. A non-positive
/// net value causes stopping rather than retrieving merely to increase recall.
pub fn select_next_retrieval_action_v1(
    offers: &[RetrievalActionOfferV1],
    policy: &RetrievalValuePolicyV1,
    state: RetrievalControllerStateV1,
) -> Result<RetrievalControllerDecisionV1, RetrievalControllerRejectV1> {
    policy.validate()?;

    if state.qualified_evidence_sufficient {
        return Ok(RetrievalControllerDecisionV1::Stop(
            RetrievalControllerStopReasonV1::EvidenceSufficient,
        ));
    }
    if state.remaining_tool_calls == 0 {
        return Ok(RetrievalControllerDecisionV1::Stop(
            RetrievalControllerStopReasonV1::ToolCallBudgetExhausted,
        ));
    }
    if offers.is_empty() {
        return Ok(RetrievalControllerDecisionV1::Stop(
            RetrievalControllerStopReasonV1::NoSafeActionAvailable,
        ));
    }

    let mut seen = BTreeSet::new();
    let mut ranked = Vec::with_capacity(offers.len());
    let mut cost_blocked = false;
    let mut risk_blocked = false;
    let mut value_evaluated = false;

    for offer in offers {
        if !valid_sha256(&offer.tool_revision_sha256)
            || !valid_sha256(&offer.action_template_sha256)
        {
            return Err(RetrievalControllerRejectV1::InvalidOfferDigest);
        }

        let offer_sha256 = offer.offer_sha256();
        let action_identity = (
            offer.kind as u8,
            offer.tool_revision_sha256.clone(),
            offer.action_template_sha256.clone(),
        );
        if !seen.insert(action_identity) {
            return Err(RetrievalControllerRejectV1::DuplicateOffer);
        }

        if offer.estimated_cost_micros > state.remaining_cost_micros {
            cost_blocked = true;
            continue;
        }
        if offer.estimated_risk_micros > policy.maximum_risk_micros {
            risk_blocked = true;
            continue;
        }

        value_evaluated = true;
        let uncertainty_penalty = i128::from(policy.confidence_z_milli)
            * i128::from(offer.uncertainty_micros)
            / 1_000;
        let cost_penalty = i128::from(policy.cost_weight_ppm)
            * i128::from(offer.estimated_cost_micros)
            / 1_000_000;
        let risk_penalty = i128::from(policy.risk_weight_ppm)
            * i128::from(offer.estimated_risk_micros)
            / 1_000_000;
        let lower_bound = i128::from(offer.expected_regret_reduction_micros)
            - uncertainty_penalty;
        let net_value = lower_bound - cost_penalty - risk_penalty;

        ranked.push((
            net_value,
            lower_bound,
            offer.kind,
            offer.action_template_sha256.as_str(),
            offer.tool_revision_sha256.as_str(),
            offer,
            offer_sha256,
        ));
    }

    ranked.sort_by(|left, right| {
        right.0.cmp(&left.0)
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| left.3.cmp(right.3))
            .then_with(|| left.4.cmp(right.4))
    });

    let Some((net_value, lower_bound, _, _, _, offer, offer_sha256)) = ranked.into_iter().next()
    else {
        let reason = if cost_blocked && !risk_blocked {
            RetrievalControllerStopReasonV1::CostBudgetExhausted
        } else {
            RetrievalControllerStopReasonV1::NoSafeActionAvailable
        };
        return Ok(RetrievalControllerDecisionV1::Stop(reason));
    };

    if net_value <= 0 {
        let reason = if value_evaluated {
            RetrievalControllerStopReasonV1::NonPositiveRiskAdjustedValue
        } else {
            RetrievalControllerStopReasonV1::NoSafeActionAvailable
        };
        return Ok(RetrievalControllerDecisionV1::Stop(reason));
    }

    Ok(RetrievalControllerDecisionV1::Execute(
        SelectedRetrievalActionV1 {
            kind: offer.kind,
            tool_revision_sha256: offer.tool_revision_sha256.clone(),
            action_template_sha256: offer.action_template_sha256.clone(),
            offer_sha256,
            policy_sha256: policy.policy_sha256(),
            lower_bound_value_micros: lower_bound,
            net_value_micros: net_value,
            estimated_cost_micros: offer.estimated_cost_micros,
            estimated_risk_micros: offer.estimated_risk_micros,
        },
    ))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOOL: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const TEMPLATE_A: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const TEMPLATE_B: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn offer(
        kind: RetrievalActionKindV1,
        template: &str,
        value: i64,
        uncertainty: u64,
        cost: u64,
        risk: u64,
    ) -> RetrievalActionOfferV1 {
        RetrievalActionOfferV1 {
            kind,
            tool_revision_sha256: TOOL.to_owned(),
            action_template_sha256: template.to_owned(),
            expected_regret_reduction_micros: value,
            uncertainty_micros: uncertainty,
            estimated_cost_micros: cost,
            estimated_risk_micros: risk,
        }
    }

    fn policy() -> RetrievalValuePolicyV1 {
        RetrievalValuePolicyV1 {
            confidence_z_milli: 1_645,
            cost_weight_ppm: 100_000,
            risk_weight_ppm: 500_000,
            maximum_risk_micros: 1_000,
        }
    }

    fn state() -> RetrievalControllerStateV1 {
        RetrievalControllerStateV1 {
            remaining_tool_calls: 3,
            remaining_cost_micros: 10_000,
            qualified_evidence_sufficient: false,
        }
    }

    fn selected(decision: RetrievalControllerDecisionV1) -> SelectedRetrievalActionV1 {
        match decision {
            RetrievalControllerDecisionV1::Execute(action) => action,
            other => panic!("expected action selection, got {other:?}"),
        }
    }

    #[test]
    fn selects_highest_risk_adjusted_lower_bound_not_highest_raw_mean() {
        let offers = [
            offer(RetrievalActionKindV1::Bm25Lexical, TEMPLATE_A, 100, 0, 0, 180),
            offer(RetrievalActionKindV1::EntityLiteral, TEMPLATE_B, 80, 0, 0, 0),
        ];
        let action = selected(select_next_retrieval_action_v1(&offers, &policy(), state())
            .expect("valid planning input"));
        assert_eq!(action.kind, RetrievalActionKindV1::EntityLiteral);
        assert_eq!(action.lower_bound_value_micros, 80);
        assert_eq!(action.net_value_micros, 80);
    }

    #[test]
    fn uncertainty_reduces_the_lower_confidence_bound() {
        let offers = [
            offer(RetrievalActionKindV1::Bm25Lexical, TEMPLATE_A, 100, 100, 0, 0),
            offer(RetrievalActionKindV1::DenseVector, TEMPLATE_B, 80, 0, 0, 0),
        ];
        let action = selected(select_next_retrieval_action_v1(&offers, &policy(), state())
            .expect("valid planning input"));
        assert_eq!(action.kind, RetrievalActionKindV1::DenseVector);
    }

    #[test]
    fn stops_when_the_best_risk_adjusted_value_is_non_positive() {
        let offers = [offer(RetrievalActionKindV1::Bm25Lexical, TEMPLATE_A, 10, 20, 0, 0)];
        let decision = select_next_retrieval_action_v1(&offers, &policy(), state())
            .expect("valid planning input");
        assert_eq!(
            decision,
            RetrievalControllerDecisionV1::Stop(
                RetrievalControllerStopReasonV1::NonPositiveRiskAdjustedValue
            )
        );
    }

    #[test]
    fn stops_when_qualified_evidence_is_already_sufficient() {
        let state = RetrievalControllerStateV1 {
            qualified_evidence_sufficient: true,
            ..state()
        };
        let decision = select_next_retrieval_action_v1(&[], &policy(), state)
            .expect("valid policy");
        assert_eq!(
            decision,
            RetrievalControllerDecisionV1::Stop(
                RetrievalControllerStopReasonV1::EvidenceSufficient
            )
        );
    }

    #[test]
    fn stops_when_tool_call_budget_is_exhausted() {
        let state = RetrievalControllerStateV1 {
            remaining_tool_calls: 0,
            ..state()
        };
        let decision = select_next_retrieval_action_v1(&[], &policy(), state)
            .expect("valid policy");
        assert_eq!(
            decision,
            RetrievalControllerDecisionV1::Stop(
                RetrievalControllerStopReasonV1::ToolCallBudgetExhausted
            )
        );
    }

    #[test]
    fn stops_when_all_offers_exceed_the_remaining_cost_budget() {
        let offers = [offer(RetrievalActionKindV1::Bm25Lexical, TEMPLATE_A, 100, 0, 20_000, 0)];
        let state = RetrievalControllerStateV1 {
            remaining_cost_micros: 10_000,
            ..state()
        };
        let decision = select_next_retrieval_action_v1(&offers, &policy(), state)
            .expect("valid planning input");
        assert_eq!(
            decision,
            RetrievalControllerDecisionV1::Stop(
                RetrievalControllerStopReasonV1::CostBudgetExhausted
            )
        );
    }

    #[test]
    fn rejects_offers_above_the_hard_risk_ceiling() {
        let offers = [offer(RetrievalActionKindV1::Bm25Lexical, TEMPLATE_A, 100, 0, 0, 1_001)];
        let decision = select_next_retrieval_action_v1(&offers, &policy(), state())
            .expect("valid planning input");
        assert_eq!(
            decision,
            RetrievalControllerDecisionV1::Stop(
                RetrievalControllerStopReasonV1::NoSafeActionAvailable
            )
        );
    }

    #[test]
    fn rejects_malformed_tool_or_template_digests() {
        let mut candidate = offer(RetrievalActionKindV1::Bm25Lexical, TEMPLATE_A, 100, 0, 0, 0);
        candidate.tool_revision_sha256 = "not-a-digest".to_owned();
        assert_eq!(
            select_next_retrieval_action_v1(&[candidate], &policy(), state()),
            Err(RetrievalControllerRejectV1::InvalidOfferDigest)
        );
    }

    #[test]
    fn rejects_duplicate_identical_offers() {
        let candidate = offer(RetrievalActionKindV1::Bm25Lexical, TEMPLATE_A, 100, 0, 0, 0);
        let same_action_with_conflicting_estimate =
            offer(RetrievalActionKindV1::Bm25Lexical, TEMPLATE_A, 120, 0, 0, 0);
        assert_eq!(
            select_next_retrieval_action_v1(
                &[candidate, same_action_with_conflicting_estimate],
                &policy(),
                state()
            ),
            Err(RetrievalControllerRejectV1::DuplicateOffer)
        );
    }

    #[test]
    fn stable_tie_break_does_not_depend_on_input_order() {
        let lexical = offer(RetrievalActionKindV1::Bm25Lexical, TEMPLATE_A, 100, 0, 0, 0);
        let entity = offer(RetrievalActionKindV1::EntityLiteral, TEMPLATE_B, 100, 0, 0, 0);
        let first = selected(select_next_retrieval_action_v1(
            &[entity.clone(), lexical.clone()], &policy(), state()
        ).expect("first order"));
        let second = selected(select_next_retrieval_action_v1(
            &[lexical, entity], &policy(), state()
        ).expect("second order"));
        assert_eq!(first.kind, RetrievalActionKindV1::Bm25Lexical);
        assert_eq!(first, second);
    }

    #[test]
    fn invalid_policy_weights_are_rejected() {
        let invalid = RetrievalValuePolicyV1 {
            cost_weight_ppm: 1_000_001,
            ..policy()
        };
        assert_eq!(
            select_next_retrieval_action_v1(&[], &invalid, state()),
            Err(RetrievalControllerRejectV1::InvalidPolicy)
        );
    }

    #[test]
    fn offer_digest_commits_to_value_and_resource_estimates() {
        let original = offer(RetrievalActionKindV1::Bm25Lexical, TEMPLATE_A, 100, 10, 5, 2);
        let mut changed = original.clone();
        changed.estimated_cost_micros += 1;
        assert_ne!(original.offer_sha256(), changed.offer_sha256());
        changed = original.clone();
        changed.expected_regret_reduction_micros -= 1;
        assert_ne!(original.offer_sha256(), changed.offer_sha256());
    }
}
