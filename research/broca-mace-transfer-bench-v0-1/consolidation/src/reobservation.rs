// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-qualified choice between retaining knowledge, retaining a locator,
//! caching a bounded snapshot, or not storing an item.
//!
//! Memory should preserve information that is costly or impossible to
//! reconstruct, while mutable state should be fetched again when a verified
//! source can provide it before the decision deadline. This module returns a
//! plan only; it does not execute re-observation tools or grant access.

use broca_mace_ingress::AsKnownQuery;
use broca_mace_policy::Scope;
use sha2::{Digest, Sha256};

use crate::salience::VerifiedFutureUsePredictionV1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryObjectKindV1 {
    CurrentState,
    StableFact,
    LearnedRule,
    Procedure,
    Preference,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReobservationAvailabilityV1 {
    Available {
        available_at_ms: i64,
        locator_sha256: String,
    },
    Unavailable,
}

/// Candidate profile submitted to the independent source/environment verifier.
/// Its claims are not trusted merely because its digest is well formed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReobservationProposalV1 {
    pub event_id: u32,
    pub evidence_sha256: String,
    pub scope: Scope,
    pub as_known: AsKnownQuery,
    pub object_kind: MemoryObjectKindV1,
    pub availability: ReobservationAvailabilityV1,
    /// Verifier-calibrated probability that a scheduled re-observation succeeds.
    pub availability_success_probability_bps: u16,
    /// Consequence cost if the locator cannot provide current state by the deadline.
    pub reobservation_failure_penalty_micros: u32,
    pub reobservation_latency_ms: u32,
    pub reobservation_cost_micros: u32,
    pub snapshot_context_tokens: u32,
    pub snapshot_valid_until_ms: Option<i64>,
    pub source_profile_sha256: String,
}

impl ReobservationProposalV1 {
    pub fn canonical_digest(&self) -> String {
        let (availability, available_at, locator) = match &self.availability {
            ReobservationAvailabilityV1::Available {
                available_at_ms,
                locator_sha256,
            } => ("available", available_at_ms.to_string(), locator_sha256.clone()),
            ReobservationAvailabilityV1::Unavailable => (
                "unavailable",
                String::new(),
                String::new(),
            ),
        };
        let fields = [
            self.event_id.to_string(),
            self.as_known.valid_at_ms.to_string(),
            self.as_known.known_at_ms.to_string(),
            self.evidence_sha256.clone(),
            scope_code(self.scope).to_owned(),
            self.as_known.valid_at_ms.to_string(),
            self.as_known.known_at_ms.to_string(),
            object_kind_code(self.object_kind).to_owned(),
            availability.to_owned(),
            available_at,
            locator,
            self.availability_success_probability_bps.to_string(),
            self.reobservation_failure_penalty_micros.to_string(),
            self.reobservation_latency_ms.to_string(),
            self.reobservation_cost_micros.to_string(),
            self.snapshot_context_tokens.to_string(),
            self.snapshot_valid_until_ms
                .map_or_else(String::new, |value| value.to_string()),
            self.source_profile_sha256.clone(),
        ];
        digest_fields(b"mycelix:mace:reobservation-proposal:v1\0", &fields)
    }
}

/// Receipt from a Mycelix source-profile / Symtropy environment observer.
/// The digest binds the exact proposal and the verifier validity interval.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReobservationReceiptV1 {
    pub proposal_sha256: String,
    pub verified_at_ms: i64,
    pub valid_until_ms: i64,
    pub verifier_id: String,
    pub receipt_sha256: String,
}

impl ReobservationReceiptV1 {
    pub fn canonical_digest(&self) -> String {
        let fields = [
            self.proposal_sha256.clone(),
            self.verified_at_ms.to_string(),
            self.valid_until_ms.to_string(),
            self.verifier_id.clone(),
        ];
        digest_fields(b"mycelix:mace:reobservation-receipt:v1\0", &fields)
    }
}

/// Production verifier must authenticate source availability, locator scope,
/// acquisition cost/latency, snapshot validity and receipt issuer.
pub trait ReobservationVerifierV1 {
    fn verify_profile(
        &self,
        proposal: &ReobservationProposalV1,
        target_scope: Scope,
        as_known: AsKnownQuery,
    ) -> Option<ReobservationReceiptV1>;
}

/// Opaque profile accepted only after the independent verifier and receipt
/// consistency checks have succeeded.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedReobservationProfileV1 {
    proposal: ReobservationProposalV1,
    receipt: ReobservationReceiptV1,
}

impl VerifiedReobservationProfileV1 {
    pub fn proposal(&self) -> &ReobservationProposalV1 {
        &self.proposal
    }

    pub fn receipt(&self) -> &ReobservationReceiptV1 {
        &self.receipt
    }

    pub fn profile_sha256(&self) -> String {
        digest_fields(
            b"mycelix:mace:verified-reobservation-profile:v1\0",
            &[
                self.proposal.canonical_digest(),
                self.receipt.canonical_digest(),
            ],
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReobservationReject {
    InvalidEventId,
    InvalidEvidenceDigest,
    ScopeMismatch,
    SnapshotMismatch,
    PredictionMismatch,
    InvalidSourceProfileDigest,
    InvalidLocatorDigest,
    InvalidAvailabilityTime,
    InvalidSnapshotValidity,
    InvalidSnapshotCost,
    InvalidVerifierReceipt,
    ReceiptNotValidAtSnapshot,
    PredictionCalibrationNotValidAtDeadline,
    DecisionBeforeKnowledgeTime,
    InvalidDecisionDeadline,
    PredictionHorizonTooShort,
    InvalidPolicyConfiguration,
}

pub fn verify_reobservation_profile(
    proposal: ReobservationProposalV1,
    target_scope: Scope,
    as_known: AsKnownQuery,
    verifier: &impl ReobservationVerifierV1,
) -> Result<VerifiedReobservationProfileV1, ReobservationReject> {
    if proposal.event_id == 0 {
        return Err(ReobservationReject::InvalidEventId);
    }
    if !is_lower_sha256(&proposal.evidence_sha256) {
        return Err(ReobservationReject::InvalidEvidenceDigest);
    }
    if proposal.scope != target_scope {
        return Err(ReobservationReject::ScopeMismatch);
    }
    if proposal.as_known != as_known {
        return Err(ReobservationReject::SnapshotMismatch);
    }
    if !is_lower_sha256(&proposal.source_profile_sha256) {
        return Err(ReobservationReject::InvalidSourceProfileDigest);
    }
    if proposal.snapshot_context_tokens == 0
        || proposal.snapshot_context_tokens > 16_384
    {
        return Err(ReobservationReject::InvalidSnapshotCost);
    }
    if proposal
        .snapshot_valid_until_ms
        .is_some_and(|expiry| expiry <= as_known.valid_at_ms)
    {
        return Err(ReobservationReject::InvalidSnapshotValidity);
    }

    match &proposal.availability {
        ReobservationAvailabilityV1::Available {
            available_at_ms,
            locator_sha256,
        } => {
            if *available_at_ms < as_known.known_at_ms
                || !is_lower_sha256(locator_sha256)
                || proposal.availability_success_probability_bps == 0
                || u32::from(proposal.availability_success_probability_bps) > 10_000
            {
                return Err(ReobservationReject::InvalidAvailabilityTime);
            }
        }
        ReobservationAvailabilityV1::Unavailable => {
            if proposal.reobservation_latency_ms != 0
                || proposal.reobservation_cost_micros != 0
                || proposal.availability_success_probability_bps != 0
                || proposal.reobservation_failure_penalty_micros != 0
            {
                return Err(ReobservationReject::InvalidAvailabilityTime);
            }
        }
    }

    let receipt = verifier
        .verify_profile(&proposal, target_scope, as_known)
        .ok_or(ReobservationReject::InvalidVerifierReceipt)?;
    if receipt.proposal_sha256 != proposal.canonical_digest()
        || receipt.verifier_id.trim().is_empty()
        || !is_lower_sha256(&receipt.receipt_sha256)
        || receipt.receipt_sha256 != receipt.canonical_digest()
    {
        return Err(ReobservationReject::InvalidVerifierReceipt);
    }
    if receipt.verified_at_ms > as_known.known_at_ms
        || receipt.valid_until_ms <= as_known.known_at_ms
    {
        return Err(ReobservationReject::ReceiptNotValidAtSnapshot);
    }

    Ok(VerifiedReobservationProfileV1 { proposal, receipt })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReobservationPolicyV1 {
    /// Exact bitemporal snapshot used by this decision.
    pub as_known: AsKnownQuery,
    pub decision_at_ms: i64,
    pub decision_deadline_ms: i64,
    pub context_budget_tokens: u32,
    pub context_cost_per_token_micros: u32,
    pub locator_context_tokens: u32,
    pub latency_cost_per_ms_micros: u32,
    pub maximum_reobservation_cost_micros: u32,
    pub minimum_net_value_micros: i64,
}

impl ReobservationPolicyV1 {
    pub fn is_valid(self) -> bool {
        self.decision_at_ms >= 0
            && self.decision_at_ms == self.as_known.known_at_ms
            && self.decision_deadline_ms >= self.decision_at_ms
            && self.context_budget_tokens > 0
            && self.context_budget_tokens <= 16_384
            && self.context_cost_per_token_micros > 0
            && self.locator_context_tokens > 0
            && self.locator_context_tokens <= self.context_budget_tokens
            && self.minimum_net_value_micros >= 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReobservationDispositionV1 {
    PersistKnowledge,
    StoreLocator,
    CacheSnapshot,
    Discard,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReobservationReasonV1 {
    KnowledgeIsNotReconstructible,
    StableKnowledgeCheaperToRetain,
    CheaperToReobserveBeforeDeadline,
    ReobservationTooSlowOrExpensive,
    SnapshotRemainsValidThroughDeadline,
    SnapshotExpiresBeforeDeadline,
    NoPositiveNetValue,
    ContextBudgetInsufficient,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReobservationPlanV1 {
    pub event_id: u32,
    pub as_known: AsKnownQuery,
    pub evidence_sha256: String,
    pub prediction_sha256: String,
    pub profile_sha256: String,
    pub disposition: ReobservationDispositionV1,
    pub reason: ReobservationReasonV1,
    pub context_tokens_reserved: u32,
    pub estimated_cost_micros: u64,
    pub expected_failure_loss_micros: i64,
    pub expected_value_micros: i64,
    pub net_value_micros: i64,
    pub locator_sha256: Option<String>,
    pub snapshot_valid_until_ms: Option<i64>,
    pub decision_deadline_ms: i64,
    pub plan_sha256: String,
}

impl ReobservationPlanV1 {
    pub fn canonical_digest(&self) -> String {
        let fields = [
            self.event_id.to_string(),
            self.as_known.valid_at_ms.to_string(),
            self.as_known.known_at_ms.to_string(),
            self.evidence_sha256.clone(),
            self.prediction_sha256.clone(),
            self.profile_sha256.clone(),
            disposition_code(self.disposition).to_owned(),
            reason_code(self.reason).to_owned(),
            self.context_tokens_reserved.to_string(),
            self.estimated_cost_micros.to_string(),
            self.expected_failure_loss_micros.to_string(),
            self.expected_value_micros.to_string(),
            self.net_value_micros.to_string(),
            self.locator_sha256.clone().unwrap_or_default(),
            self.snapshot_valid_until_ms
                .map_or_else(String::new, |value| value.to_string()),
            self.decision_deadline_ms.to_string(),
        ];
        digest_fields(b"mycelix:mace:reobservation-plan:v1\0", &fields)
    }
}

fn usable_locator(
    profile: &VerifiedReobservationProfileV1,
    policy: ReobservationPolicyV1,
) -> Option<(String, u64)> {
    if profile.receipt().valid_until_ms <= policy.decision_deadline_ms
        || policy.locator_context_tokens > policy.context_budget_tokens
    {
        return None;
    }
    let ReobservationAvailabilityV1::Available {
        available_at_ms,
        locator_sha256,
    } = &profile.proposal().availability
    else {
        return None;
    };
    let proposal = profile.proposal();
    let complete_at = available_at_ms
        .saturating_add(i64::from(proposal.reobservation_latency_ms));
    let latency_cost = u64::from(proposal.reobservation_latency_ms)
        .saturating_mul(u64::from(policy.latency_cost_per_ms_micros));
    let total_reobservation_cost = latency_cost
        .saturating_add(u64::from(proposal.reobservation_cost_micros));
    if complete_at > policy.decision_deadline_ms
        || total_reobservation_cost > u64::from(policy.maximum_reobservation_cost_micros)
    {
        return None;
    }
    let locator_storage_cost = u64::from(policy.locator_context_tokens)
        .saturating_mul(u64::from(policy.context_cost_per_token_micros));
    Some((
        locator_sha256.clone(),
        total_reobservation_cost.saturating_add(locator_storage_cost),
    ))
}

fn expected_success_value(value_micros: i64, success_probability_bps: u16) -> i64 {
    let product = i128::from(value_micros) * i128::from(success_probability_bps);
    // Floor signed expected value so a negative estimate is never made
    // artificially less negative by truncation toward zero.
    let expected = product
        .div_euclid(10_000)
        .clamp(i128::from(i64::MIN), i128::from(i64::MAX));
    i64::try_from(expected).unwrap_or(i64::MAX)
}

fn expected_failure_loss(
    failure_penalty_micros: u32,
    success_probability_bps: u16,
) -> i64 {
    let failure_probability_bps = 10_000_u32.saturating_sub(u32::from(success_probability_bps));
    // Expected failure cost is rounded up: fractional micro-loss must not
    // disappear and make re-observation look cheaper than its risk warrants.
    let numerator =
        u64::from(failure_probability_bps) * u64::from(failure_penalty_micros);
    let expected_loss = numerator.saturating_add(9_999) / 10_000;
    i64::try_from(expected_loss).unwrap_or(i64::MAX)
}

pub fn plan_memory_representation(
    prediction: &VerifiedFutureUsePredictionV1,
    profile: &VerifiedReobservationProfileV1,
    policy: ReobservationPolicyV1,
) -> Result<ReobservationPlanV1, ReobservationReject> {
    if !policy.is_valid() {
        return Err(ReobservationReject::InvalidPolicyConfiguration);
    }

    let proposal = profile.proposal();
    let raw_prediction = prediction.prediction();
    if raw_prediction.event_id != proposal.event_id
        || raw_prediction.evidence_sha256 != proposal.evidence_sha256
        || raw_prediction.target_scope != proposal.scope
    {
        return Err(ReobservationReject::PredictionMismatch);
    }
    let calibration = prediction.calibration_receipt();
    if proposal.as_known != policy.as_known
        || calibration.as_known != proposal.as_known
        || calibration.target_scope != proposal.scope
        || raw_prediction.predicted_at_ms > proposal.as_known.known_at_ms
        || profile.receipt().verified_at_ms > proposal.as_known.known_at_ms
    {
        return Err(ReobservationReject::SnapshotMismatch);
    }
    if calibration.valid_until_ms <= policy.decision_deadline_ms {
        return Err(ReobservationReject::PredictionCalibrationNotValidAtDeadline);
    }
    if policy.decision_at_ms < proposal.as_known.known_at_ms {
        return Err(ReobservationReject::DecisionBeforeKnowledgeTime);
    }
    if policy.decision_deadline_ms < policy.decision_at_ms {
        return Err(ReobservationReject::InvalidDecisionDeadline);
    }
    if raw_prediction.horizon_end_ms < policy.decision_deadline_ms {
        return Err(ReobservationReject::PredictionHorizonTooShort);
    }
    if profile.receipt().valid_until_ms <= proposal.as_known.known_at_ms {
        return Err(ReobservationReject::ReceiptNotValidAtSnapshot);
    }

    let expected = raw_prediction.expected_net_value_micros();
    let profile_sha = profile.profile_sha256();
    let prediction_sha = raw_prediction.prediction_sha256.clone();

    let mut disposition = ReobservationDispositionV1::Discard;
    let mut reason = ReobservationReasonV1::NoPositiveNetValue;
    let mut context_tokens = 0_u32;
    let mut estimated_cost = 0_u64;
    let mut expected_failure_loss = 0_i64;
    let mut locator = None;
    let mut snapshot_expiry = None;
    let mut net = i64::MIN;

    let snapshot_cost = u64::from(proposal.snapshot_context_tokens)
        .saturating_mul(u64::from(policy.context_cost_per_token_micros));
    let snapshot_fits = proposal.snapshot_context_tokens <= policy.context_budget_tokens;
    let snapshot_valid = proposal
        .snapshot_valid_until_ms
        .is_some_and(|expiry| expiry > policy.decision_deadline_ms);
    let snapshot_net = if snapshot_fits && snapshot_valid {
        expected.saturating_sub(i64::try_from(snapshot_cost).unwrap_or(i64::MAX))
    } else {
        i64::MIN
    };
    let stable_knowledge_net = if snapshot_fits {
        expected.saturating_sub(i64::try_from(snapshot_cost).unwrap_or(i64::MAX))
    } else {
        i64::MIN
    };

    match proposal.object_kind {
        MemoryObjectKindV1::LearnedRule
        | MemoryObjectKindV1::Procedure
        | MemoryObjectKindV1::Preference => {
            if !snapshot_fits {
                reason = ReobservationReasonV1::ContextBudgetInsufficient;
            } else if stable_knowledge_net > policy.minimum_net_value_micros {
                disposition = ReobservationDispositionV1::PersistKnowledge;
                reason = ReobservationReasonV1::KnowledgeIsNotReconstructible;
                context_tokens = proposal.snapshot_context_tokens;
                estimated_cost = snapshot_cost;
                net = stable_knowledge_net;
            } else {
                reason = ReobservationReasonV1::NoPositiveNetValue;
            }
        }
        MemoryObjectKindV1::StableFact => {
            let locator_option = usable_locator(profile, policy);
            let locator_failure_loss = expected_failure_loss(
                proposal.reobservation_failure_penalty_micros,
                proposal.availability_success_probability_bps,
            );
            let locator_success_value = expected_success_value(
                expected,
                proposal.availability_success_probability_bps,
            );
            let locator_net = locator_option.as_ref().map_or(i64::MIN, |(_, cost)| {
                locator_success_value
                    .saturating_sub(locator_failure_loss)
                    .saturating_sub(i64::try_from(*cost).unwrap_or(i64::MAX))
            });
            if locator_net > stable_knowledge_net
                && locator_net > policy.minimum_net_value_micros
            {
                disposition = ReobservationDispositionV1::StoreLocator;
                reason = ReobservationReasonV1::CheaperToReobserveBeforeDeadline;
                context_tokens = policy.locator_context_tokens;
                estimated_cost = locator_option.as_ref().map_or(0, |(_, cost)| *cost);
                locator = locator_option.map(|(digest, _)| digest);
                expected_failure_loss = locator_failure_loss;
                net = locator_net;
            } else if stable_knowledge_net > policy.minimum_net_value_micros {
                disposition = ReobservationDispositionV1::PersistKnowledge;
                reason = ReobservationReasonV1::StableKnowledgeCheaperToRetain;
                context_tokens = proposal.snapshot_context_tokens;
                estimated_cost = snapshot_cost;
                net = stable_knowledge_net;
            } else if !snapshot_fits {
                reason = ReobservationReasonV1::ContextBudgetInsufficient;
            }
        }
        MemoryObjectKindV1::CurrentState => {
            let locator_option = usable_locator(profile, policy);
            let locator_failure_loss = expected_failure_loss(
                proposal.reobservation_failure_penalty_micros,
                proposal.availability_success_probability_bps,
            );
            let locator_success_value = expected_success_value(
                expected,
                proposal.availability_success_probability_bps,
            );
            let locator_net = locator_option.as_ref().map_or(i64::MIN, |(_, cost)| {
                locator_success_value
                    .saturating_sub(locator_failure_loss)
                    .saturating_sub(i64::try_from(*cost).unwrap_or(i64::MAX))
            });

            if locator_net > policy.minimum_net_value_micros {
                disposition = ReobservationDispositionV1::StoreLocator;
                reason = ReobservationReasonV1::CheaperToReobserveBeforeDeadline;
                context_tokens = policy.locator_context_tokens;
                estimated_cost = locator_option.as_ref().map_or(0, |(_, cost)| *cost);
                locator = locator_option.map(|(digest, _)| digest);
                expected_failure_loss = locator_failure_loss;
                net = locator_net;
            } else if snapshot_net > policy.minimum_net_value_micros {
                disposition = ReobservationDispositionV1::CacheSnapshot;
                reason = match &proposal.availability {
                    ReobservationAvailabilityV1::Available { .. } => {
                        ReobservationReasonV1::ReobservationTooSlowOrExpensive
                    }
                    ReobservationAvailabilityV1::Unavailable => {
                        ReobservationReasonV1::SnapshotRemainsValidThroughDeadline
                    }
                };
                context_tokens = proposal.snapshot_context_tokens;
                estimated_cost = snapshot_cost;
                snapshot_expiry = proposal.snapshot_valid_until_ms;
                net = snapshot_net;
            } else if !snapshot_valid {
                reason = ReobservationReasonV1::SnapshotExpiresBeforeDeadline;
            } else if !snapshot_fits && locator_option.is_none() {
                reason = ReobservationReasonV1::ContextBudgetInsufficient;
            } else {
                reason = ReobservationReasonV1::NoPositiveNetValue;
            }
        }
    }

    let mut plan = ReobservationPlanV1 {
        event_id: proposal.event_id,
        as_known: proposal.as_known,
        evidence_sha256: proposal.evidence_sha256.clone(),
        prediction_sha256: prediction_sha,
        profile_sha256: profile_sha,
        disposition,
        reason,
        context_tokens_reserved: context_tokens,
        estimated_cost_micros: estimated_cost,
        expected_failure_loss_micros: expected_failure_loss,
        expected_value_micros: expected,
        net_value_micros: net,
        locator_sha256: locator,
        snapshot_valid_until_ms: snapshot_expiry,
        decision_deadline_ms: policy.decision_deadline_ms,
        plan_sha256: String::new(),
    };
    plan.plan_sha256 = plan.canonical_digest();
    Ok(plan)
}

fn object_kind_code(kind: MemoryObjectKindV1) -> &'static str {
    match kind {
        MemoryObjectKindV1::CurrentState => "current_state",
        MemoryObjectKindV1::StableFact => "stable_fact",
        MemoryObjectKindV1::LearnedRule => "learned_rule",
        MemoryObjectKindV1::Procedure => "procedure",
        MemoryObjectKindV1::Preference => "preference",
    }
}

fn disposition_code(value: ReobservationDispositionV1) -> &'static str {
    match value {
        ReobservationDispositionV1::PersistKnowledge => "persist_knowledge",
        ReobservationDispositionV1::StoreLocator => "store_locator",
        ReobservationDispositionV1::CacheSnapshot => "cache_snapshot",
        ReobservationDispositionV1::Discard => "discard",
    }
}

fn reason_code(value: ReobservationReasonV1) -> &'static str {
    match value {
        ReobservationReasonV1::KnowledgeIsNotReconstructible => {
            "knowledge_is_not_reconstructible"
        }
        ReobservationReasonV1::StableKnowledgeCheaperToRetain => {
            "stable_knowledge_cheaper_to_retain"
        }
        ReobservationReasonV1::CheaperToReobserveBeforeDeadline => {
            "cheaper_to_reobserve_before_deadline"
        }
        ReobservationReasonV1::ReobservationTooSlowOrExpensive => {
            "reobservation_too_slow_or_expensive"
        }
        ReobservationReasonV1::SnapshotRemainsValidThroughDeadline => {
            "snapshot_remains_valid_through_deadline"
        }
        ReobservationReasonV1::SnapshotExpiresBeforeDeadline => {
            "snapshot_expires_before_deadline"
        }
        ReobservationReasonV1::NoPositiveNetValue => "no_positive_net_value",
        ReobservationReasonV1::ContextBudgetInsufficient => {
            "context_budget_insufficient"
        }
    }
}

fn scope_code(scope: Scope) -> &'static str {
    use broca_mace_policy::{Objective, Regime};
    match (scope.regime, scope.objective) {
        (Regime::Baseline, Objective::Quality) => "baseline:quality",
        (Regime::Baseline, Objective::Cost) => "baseline:cost",
        (Regime::Baseline, Objective::Speed) => "baseline:speed",
        (Regime::Shifted, Objective::Quality) => "shifted:quality",
        (Regime::Shifted, Objective::Cost) => "shifted:cost",
        (Regime::Shifted, Objective::Speed) => "shifted:speed",
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest_fields(domain: &[u8], fields: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for field in fields {
        hasher.update((field.len() as u64).to_be_bytes());
        hasher.update(field.as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::salience::{
        verify_future_use_prediction, FutureUsePredictionV1,
        SalienceCalibrationReceiptV1, SaliencePredictionVerifierV1,
    };
    use broca_mace_policy::{Objective, Regime};

    fn scope() -> Scope {
        Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        }
    }

    fn digest(label: &str) -> String {
        sha256_hex(label.as_bytes())
    }

    struct FixtureVerifier {
        corrupt_receipt: bool,
        valid_for_ms: i64,
    }

    impl ReobservationVerifierV1 for FixtureVerifier {
        fn verify_profile(
            &self,
            proposal: &ReobservationProposalV1,
            _: Scope,
            _: AsKnownQuery,
        ) -> Option<ReobservationReceiptV1> {
            let mut receipt = ReobservationReceiptV1 {
                proposal_sha256: proposal.canonical_digest(),
                verified_at_ms: proposal.as_known.known_at_ms,
                valid_until_ms: proposal.as_known.known_at_ms + self.valid_for_ms,
                verifier_id: "fixture-observer-v1".to_owned(),
                receipt_sha256: String::new(),
            };
            receipt.receipt_sha256 = receipt.canonical_digest();
            if self.corrupt_receipt {
                receipt.proposal_sha256 = digest("substituted-proposal");
                receipt.receipt_sha256 = receipt.canonical_digest();
            }
            Some(receipt)
        }
    }

    struct FixtureSalienceVerifier {
        valid_for_ms: i64,
    }

    impl SaliencePredictionVerifierV1 for FixtureSalienceVerifier {
        fn verify_prediction(
            &self,
            prediction: &FutureUsePredictionV1,
            target_scope: Scope,
            as_known: AsKnownQuery,
        ) -> Option<SalienceCalibrationReceiptV1> {
            let mut receipt = SalienceCalibrationReceiptV1 {
                prediction_sha256: prediction.prediction_sha256.clone(),
                evidence_sha256: prediction.evidence_sha256.clone(),
                target_model_sha256: prediction.target_model_sha256.clone(),
                target_scope,
                as_known,
                model_id: prediction.model_id.clone(),
                model_version: prediction.model_version.clone(),
                model_config_sha256: prediction.model_config_sha256.clone(),
                training_dataset_sha256: digest("train"),
                calibration_dataset_sha256: digest("calibration"),
                heldout_evaluation_sha256: digest("heldout"),
                task_family_split_sha256: digest("task-split"),
                source_root_split_sha256: digest("root-split"),
                training_cluster_count: 32,
                calibration_cluster_count: 16,
                heldout_cluster_count: 16,
                brier_score_micros: 100_000,
                base_rate_brier_score_micros: 150_000,
                expected_calibration_error_micros: 30_000,
                worst_group_calibration_error_micros: 80_000,
                task_families_disjoint: true,
                source_roots_disjoint: true,
                verified_at_ms: as_known.known_at_ms,
                valid_until_ms: as_known.known_at_ms + self.valid_for_ms,
                verifier_id: "fixture-salience-v1".to_owned(),
                receipt_sha256: String::new(),
            };
            receipt.receipt_sha256 = receipt.canonical_digest();
            Some(receipt)
        }
    }

    fn as_known() -> AsKnownQuery {
        AsKnownQuery {
            valid_at_ms: 100,
            known_at_ms: 300,
        }
    }

    fn proposal(
        kind: MemoryObjectKindV1,
        availability: ReobservationAvailabilityV1,
        snapshot_valid_until_ms: Option<i64>,
    ) -> ReobservationProposalV1 {
        let (latency, cost) = match &availability {
            ReobservationAvailabilityV1::Available { .. } => (20, 5_000),
            ReobservationAvailabilityV1::Unavailable => (0, 0),
        };
        ReobservationProposalV1 {
            event_id: 7,
            evidence_sha256: digest("evidence"),
            scope: scope(),
            as_known: as_known(),
            object_kind: kind,
            availability,
            availability_success_probability_bps: if latency == 0 { 0 } else { 9_900 },
            reobservation_failure_penalty_micros: if latency == 0 { 0 } else { 20_000 },
            reobservation_latency_ms: latency,
            reobservation_cost_micros: cost,
            snapshot_context_tokens: 100,
            snapshot_valid_until_ms,
            source_profile_sha256: digest("source-profile"),
        }
    }

    fn prediction_for(
        event_id: u32,
        evidence_sha256: &str,
        expected_value_gain_micros: u32,
        uncertainty_penalty_micros: u32,
    ) -> VerifiedFutureUsePredictionV1 {
        let raw = FutureUsePredictionV1::new(
            event_id,
            digest("target-model"),
            scope(),
            150,
            10_000,
            9_000,
            expected_value_gain_micros,
            uncertainty_penalty_micros,
            "future-use-model",
            "v1",
            digest("model-config"),
            digest("rationale"),
            evidence_sha256,
        );
        verify_future_use_prediction(
            raw,
            scope(),
            as_known(),
            &FixtureSalienceVerifier {
                valid_for_ms: 10_000,
            },
        )
        .unwrap()
    }

    fn prediction_with_validity(
        expected_value_gain_micros: u32,
        uncertainty_penalty_micros: u32,
        valid_for_ms: i64,
    ) -> VerifiedFutureUsePredictionV1 {
        let raw = FutureUsePredictionV1::new(
            7,
            digest("target-model"),
            scope(),
            150,
            10_000,
            9_000,
            expected_value_gain_micros,
            uncertainty_penalty_micros,
            "future-use-model",
            "v1",
            digest("model-config"),
            digest("rationale"),
            digest("evidence"),
        );
        verify_future_use_prediction(
            raw,
            scope(),
            as_known(),
            &FixtureSalienceVerifier { valid_for_ms },
        )
        .unwrap()
    }

    fn prediction(
        expected_value_gain_micros: u32,
        uncertainty_penalty_micros: u32,
    ) -> VerifiedFutureUsePredictionV1 {
        prediction_for(
            7,
            &digest("evidence"),
            expected_value_gain_micros,
            uncertainty_penalty_micros,
        )
    }

    fn policy() -> ReobservationPolicyV1 {
        ReobservationPolicyV1 {
            as_known: as_known(),
            decision_at_ms: 300,
            decision_deadline_ms: 500,
            context_budget_tokens: 512,
            context_cost_per_token_micros: 100,
            locator_context_tokens: 8,
            latency_cost_per_ms_micros: 10,
            maximum_reobservation_cost_micros: 20_000,
            minimum_net_value_micros: 0,
        }
    }

    fn verified(
        proposal: ReobservationProposalV1,
    ) -> VerifiedReobservationProfileV1 {
        verify_reobservation_profile(
            proposal,
            scope(),
            as_known(),
            &FixtureVerifier {
                corrupt_receipt: false,
                valid_for_ms: 10_000,
            },
        )
        .unwrap()
    }

    #[test]
    fn decision_clock_must_match_the_bitemporal_knowledge_snapshot() {
        let profile = verified(proposal(
            MemoryObjectKindV1::CurrentState,
            ReobservationAvailabilityV1::Unavailable,
            Some(900),
        ));
        let mut stale_policy = policy();
        stale_policy.decision_at_ms += 1;
        assert_eq!(
            plan_memory_representation(
                &prediction(50_000, 1_000),
                &profile,
                stale_policy,
            ),
            Err(ReobservationReject::InvalidPolicyConfiguration)
        );
    }

    #[test]
    fn prediction_calibration_must_match_the_source_profile_snapshot() {
        let older_snapshot = AsKnownQuery {
            valid_at_ms: 100,
            known_at_ms: 200,
        };
        let raw = FutureUsePredictionV1::new(
            7,
            digest("target-model"),
            scope(),
            150,
            10_000,
            9_000,
            50_000,
            1_000,
            "future-use-model",
            "v1",
            digest("model-config"),
            digest("rationale"),
            digest("evidence"),
        );
        let old_prediction = verify_future_use_prediction(
            raw,
            scope(),
            older_snapshot,
            &FixtureSalienceVerifier {
                valid_for_ms: 10_000,
            },
        )
        .unwrap();
        let profile = verified(proposal(
            MemoryObjectKindV1::CurrentState,
            ReobservationAvailabilityV1::Unavailable,
            Some(900),
        ));
        assert_eq!(
            plan_memory_representation(&old_prediction, &profile, policy()),
            Err(ReobservationReject::SnapshotMismatch)
        );
    }

    #[test]
    fn prediction_calibration_expiring_at_the_deadline_is_not_usable() {
        let profile = verified(proposal(
            MemoryObjectKindV1::CurrentState,
            ReobservationAvailabilityV1::Unavailable,
            Some(900),
        ));
        let forecast = prediction_with_validity(50_000, 1_000, 200);
        assert_eq!(
            plan_memory_representation(&forecast, &profile, policy()),
            Err(ReobservationReject::PredictionCalibrationNotValidAtDeadline)
        );
    }

    #[test]
    fn a_reobservation_profile_cannot_be_reused_after_knowledge_time_advances() {
        let old_snapshot = AsKnownQuery {
            valid_at_ms: 100,
            known_at_ms: 200,
        };
        let old_proposal = ReobservationProposalV1 {
            as_known: old_snapshot,
            ..proposal(
                MemoryObjectKindV1::CurrentState,
                ReobservationAvailabilityV1::Available {
                    available_at_ms: 200,
                    locator_sha256: digest("mutable-source"),
                },
                Some(900),
            )
        };
        let old_profile = verify_reobservation_profile(
            old_proposal,
            scope(),
            old_snapshot,
            &FixtureVerifier {
                corrupt_receipt: false,
                valid_for_ms: 10_000,
            },
        )
        .unwrap();

        assert_eq!(
            plan_memory_representation(
                &prediction(50_000, 1_000),
                &old_profile,
                policy(),
            ),
            Err(ReobservationReject::SnapshotMismatch)
        );
    }

    #[test]
    fn cheap_live_state_becomes_a_locator_not_a_stale_snapshot() {
        let profile = verified(proposal(
            MemoryObjectKindV1::CurrentState,
            ReobservationAvailabilityV1::Available {
                available_at_ms: 300,
                locator_sha256: digest("live-source"),
            },
            Some(1_000),
        ));
        let plan = plan_memory_representation(&prediction(50_000, 1_000), &profile, policy())
            .unwrap();
        assert_eq!(plan.disposition, ReobservationDispositionV1::StoreLocator);
        assert_eq!(plan.context_tokens_reserved, 8);
        assert_eq!(plan.snapshot_valid_until_ms, None);
        assert!(plan.locator_sha256.is_some());
    }

    #[test]
    fn learned_strategy_is_persisted_even_when_current_state_is_not_queryable() {
        let profile = verified(proposal(
            MemoryObjectKindV1::LearnedRule,
            ReobservationAvailabilityV1::Unavailable,
            None,
        ));
        let plan = plan_memory_representation(&prediction(50_000, 1_000), &profile, policy())
            .unwrap();
        assert_eq!(plan.disposition, ReobservationDispositionV1::PersistKnowledge);
        assert_eq!(plan.context_tokens_reserved, 100);
        assert_eq!(plan.reason, ReobservationReasonV1::KnowledgeIsNotReconstructible);
    }

    #[test]
    fn unavailable_state_is_cached_only_until_its_verified_snapshot_expiry() {
        let profile = verified(proposal(
            MemoryObjectKindV1::CurrentState,
            ReobservationAvailabilityV1::Unavailable,
            Some(600),
        ));
        let plan = plan_memory_representation(&prediction(50_000, 1_000), &profile, policy())
            .unwrap();
        assert_eq!(plan.disposition, ReobservationDispositionV1::CacheSnapshot);
        assert_eq!(plan.snapshot_valid_until_ms, Some(600));
    }

    #[test]
    fn expired_snapshot_and_unavailable_source_are_not_retained() {
        let profile = verified(proposal(
            MemoryObjectKindV1::CurrentState,
            ReobservationAvailabilityV1::Unavailable,
            Some(450),
        ));
        let plan = plan_memory_representation(&prediction(50_000, 1_000), &profile, policy())
            .unwrap();
        assert_eq!(plan.disposition, ReobservationDispositionV1::Discard);
        assert_eq!(plan.reason, ReobservationReasonV1::SnapshotExpiresBeforeDeadline);
    }

    #[test]
    fn reobservation_that_finishes_after_the_decision_deadline_uses_a_fresh_snapshot() {
        let mut item = proposal(
            MemoryObjectKindV1::CurrentState,
            ReobservationAvailabilityV1::Available {
                available_at_ms: 450,
                locator_sha256: digest("slow-live-source"),
            },
            Some(800),
        );
        item.reobservation_latency_ms = 100;
        let profile = verified(item);
        let plan = plan_memory_representation(&prediction(50_000, 1_000), &profile, policy())
            .unwrap();
        assert_eq!(plan.disposition, ReobservationDispositionV1::CacheSnapshot);
        assert_eq!(
            plan.reason,
            ReobservationReasonV1::ReobservationTooSlowOrExpensive
        );
    }

    #[test]
    fn stale_or_wrong_evidence_profiles_cannot_be_used() {
        let profile = verified(proposal(
            MemoryObjectKindV1::CurrentState,
            ReobservationAvailabilityV1::Unavailable,
            Some(900),
        ));
        let wrong_prediction = prediction_for(
            8,
            &digest("evidence"),
            50_000,
            1_000,
        );
        let wrong_proposal = proposal(
            MemoryObjectKindV1::CurrentState,
            ReobservationAvailabilityV1::Unavailable,
            Some(900),
        );
        let mut wrong_proposal = ReobservationProposalV1 {
            event_id: 99,
            ..wrong_proposal
        };
        wrong_proposal.event_id = 0;
        assert_eq!(
            verify_reobservation_profile(
                wrong_proposal,
                scope(),
                as_known(),
                &FixtureVerifier {
                    corrupt_receipt: false,
                    valid_for_ms: 10_000,
                },
            ),
            Err(ReobservationReject::InvalidEventId)
        );
        assert_eq!(
            plan_memory_representation(&wrong_prediction, &profile, policy()),
            Err(ReobservationReject::PredictionMismatch)
        );
    }

    #[test]
    fn locator_receipt_must_remain_valid_through_the_decision_deadline() {
        let item = proposal(
            MemoryObjectKindV1::CurrentState,
            ReobservationAvailabilityV1::Available {
                available_at_ms: 300,
                locator_sha256: digest("live-source"),
            },
            Some(800),
        );
        let profile = verify_reobservation_profile(
            item,
            scope(),
            as_known(),
            &FixtureVerifier {
                corrupt_receipt: false,
                valid_for_ms: 200,
            },
        )
        .unwrap();
        let plan = plan_memory_representation(
            &prediction(50_000, 1_000),
            &profile,
            policy(),
        )
        .unwrap();
        assert_eq!(plan.disposition, ReobservationDispositionV1::CacheSnapshot);
        assert_eq!(plan.snapshot_valid_until_ms, Some(800));
    }

    #[test]
    fn stable_fact_uses_a_locator_only_when_its_risk_adjusted_value_wins() {
        let profile = verified(proposal(
            MemoryObjectKindV1::StableFact,
            ReobservationAvailabilityV1::Available {
                available_at_ms: 300,
                locator_sha256: digest("stable-fact-source"),
            },
            None,
        ));
        let plan = plan_memory_representation(
            &prediction(50_000, 1_000),
            &profile,
            policy(),
        )
        .unwrap();
        assert_eq!(plan.disposition, ReobservationDispositionV1::StoreLocator);
        assert!(plan.locator_sha256.is_some());
        assert_eq!(plan.as_known, as_known());
    }

    #[test]
    fn stable_fact_can_be_retained_without_an_expiring_snapshot() {
        let profile = verified(proposal(
            MemoryObjectKindV1::StableFact,
            ReobservationAvailabilityV1::Unavailable,
            None,
        ));
        let plan = plan_memory_representation(
            &prediction(50_000, 1_000),
            &profile,
            policy(),
        )
        .unwrap();
        assert_eq!(plan.disposition, ReobservationDispositionV1::PersistKnowledge);
        assert_eq!(
            plan.reason,
            ReobservationReasonV1::StableKnowledgeCheaperToRetain
        );
    }

    #[test]
    fn forged_receipt_cannot_qualify_a_reobservation_profile() {
        let item = proposal(
            MemoryObjectKindV1::CurrentState,
            ReobservationAvailabilityV1::Unavailable,
            Some(900),
        );
        assert_eq!(
            verify_reobservation_profile(
                item,
                scope(),
                as_known(),
                &FixtureVerifier {
                    corrupt_receipt: true,
                    valid_for_ms: 10_000,
                },
            ),
            Err(ReobservationReject::InvalidVerifierReceipt)
        );
    }

    #[test]
    fn no_positive_forecast_does_not_create_memory_or_a_locator() {
        let profile = verified(proposal(
            MemoryObjectKindV1::CurrentState,
            ReobservationAvailabilityV1::Available {
                available_at_ms: 300,
                locator_sha256: digest("live-source"),
            },
            Some(1_000),
        ));
        let plan = plan_memory_representation(&prediction(0, 1), &profile, policy()).unwrap();
        assert_eq!(plan.disposition, ReobservationDispositionV1::Discard);
    }

    #[test]
    fn fractional_expected_failure_loss_rounds_up_instead_of_disappearing() {
        assert_eq!(expected_failure_loss(1, 9_999), 1);
        assert_eq!(expected_failure_loss(1, 1), 1);
        assert_eq!(expected_failure_loss(100, 9_900), 1);
        assert_eq!(expected_failure_loss(1_000, 10_000), 0);
    }

    #[test]
    fn negative_expected_success_value_rounds_down_away_from_zero() {
        assert_eq!(expected_success_value(-1, 1), -1);
        assert_eq!(expected_success_value(-10_001, 5_000), -5_001);
        assert_eq!(expected_success_value(1, 1), 0);
    }

    #[test]
    fn low_reobservation_reliability_can_make_a_fresh_snapshot_better() {
        let mut item = proposal(
            MemoryObjectKindV1::CurrentState,
            ReobservationAvailabilityV1::Available {
                available_at_ms: 300,
                locator_sha256: digest("fragile-live-source"),
            },
            Some(800),
        );
        item.availability_success_probability_bps = 2_000;
        item.reobservation_failure_penalty_micros = 80_000;
        let profile = verified(item);
        let plan = plan_memory_representation(&prediction(50_000, 1_000), &profile, policy())
            .unwrap();
        assert_eq!(plan.disposition, ReobservationDispositionV1::CacheSnapshot);
        assert_eq!(plan.expected_failure_loss_micros, 0);
        assert!(plan.locator_sha256.is_none());
    }

    #[test]
    fn locator_and_snapshot_receipts_expiring_exactly_at_deadline_are_invalid() {
        let item = proposal(
            MemoryObjectKindV1::CurrentState,
            ReobservationAvailabilityV1::Available {
                available_at_ms: 300,
                locator_sha256: digest("deadline-source"),
            },
            Some(500),
        );
        let profile = verify_reobservation_profile(
            item,
            scope(),
            as_known(),
            &FixtureVerifier {
                corrupt_receipt: false,
                valid_for_ms: 200,
            },
        )
        .unwrap();
        let plan = plan_memory_representation(&prediction(50_000, 1_000), &profile, policy())
            .unwrap();
        assert_eq!(plan.disposition, ReobservationDispositionV1::Discard);
        assert_eq!(plan.expected_failure_loss_micros, 0);
    }

    #[test]
    fn malformed_availability_probability_is_rejected() {
        let mut item = proposal(
            MemoryObjectKindV1::CurrentState,
            ReobservationAvailabilityV1::Available {
                available_at_ms: 300,
                locator_sha256: digest("live-source"),
            },
            Some(1_000),
        );
        item.availability_success_probability_bps = 0;
        assert_eq!(
            verify_reobservation_profile(
                item,
                scope(),
                as_known(),
                &FixtureVerifier {
                    corrupt_receipt: false,
                    valid_for_ms: 10_000,
                },
            ),
            Err(ReobservationReject::InvalidAvailabilityTime)
        );
    }

    #[test]
    fn plan_digest_binds_representation_cost_expiry_and_deadline() {
        let profile = verified(proposal(
            MemoryObjectKindV1::CurrentState,
            ReobservationAvailabilityV1::Available {
                available_at_ms: 300,
                locator_sha256: digest("live-source"),
            },
            Some(1_000),
        ));
        let plan = plan_memory_representation(&prediction(50_000, 1_000), &profile, policy())
            .unwrap();
        assert_eq!(plan.plan_sha256, plan.canonical_digest());
        assert_eq!(plan.as_known, profile.proposal().as_known);
        assert_ne!(plan.profile_sha256, profile.proposal().canonical_digest());
    }
}
