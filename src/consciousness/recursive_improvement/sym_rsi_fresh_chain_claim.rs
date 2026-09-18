// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Claim synthesis for the two-stage SYM-RSI fresh evidence chain.
//!
//! This module consumes no environment seeds and performs no selection. It combines
//! already-frozen C-vs-A and D-vs-C fresh evidence so a narrow incremental result
//! cannot be mislabeled as a complete recursive-improvement chain.

use super::sym_rsi_candidate_family::canonical_candidate_family_digest;
use super::sym_rsi_dream_fresh_evaluation::{
    DreamFreshDisposition, DreamFreshEvaluationReceipt, SYM_RSI_001D_FRESH_EVALUATION_SCHEMA,
};
use super::sym_rsi_dream_parent_gate::QualifiedDreamFresh;
use super::sym_rsi_experiment::EvaluationSplit;
use super::sym_rsi_fresh_evaluation::{
    FreshCvsADisposition, FreshCvsAReceipt, SYM_RSI_001_C_VS_A_ANALYSIS_RULE,
    SYM_RSI_001_FRESH_EVALUATION_SCHEMA,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const SYM_RSI_FRESH_CHAIN_CLAIM_SCHEMA: &str =
    "symthaea.sym-rsi.fresh-improvement-chain.v1";
const NUMERIC_EPSILON: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshImprovementChainDisposition {
    TwoStageFreshImprovementEstablished,
    ReplayOnlyFreshImprovement,
    DreamIncrementOnly,
    NoFreshImprovementChain,
    IntegrityFailure,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FreshImprovementChainReceipt {
    pub schema: String,
    pub subject_digest: String,
    pub environment_digest: String,
    pub c_vs_a_experiment_id: String,
    pub d_vs_c_experiment_id: String,
    pub selected_c_policy_id: String,
    pub dream_d_policy_id: String,
    pub shared_parent_holdout_gate_evidence_digest: String,
    pub c_vs_a_fresh_evidence_digest: String,
    pub qualified_d_vs_c_fresh_evidence_digest: String,
    pub d_vs_c_fresh_evidence_digest: String,
    pub c_vs_a_disposition: FreshCvsADisposition,
    pub d_vs_c_disposition: DreamFreshDisposition,
    pub c_vs_a_positive: bool,
    pub d_vs_c_positive: bool,
    pub disposition: FreshImprovementChainDisposition,
    pub evidence_digest: String,
}

/// Synthesize already-measured fresh evidence. This function never executes a
/// fixture, never chooses a policy, and never changes either source disposition.
///
/// D-vs-C must arrive as the private-constructor `QualifiedDreamFresh` token rather
/// than a serializable receipt, so the chain cannot bypass the parent-C and dream
/// verification qualification path.
pub fn build_fresh_improvement_chain_receipt(
    c_vs_a: &FreshCvsAReceipt,
    qualified_d_vs_c: &QualifiedDreamFresh,
) -> Result<FreshImprovementChainReceipt, FreshImprovementChainError> {
    let qualified_receipt = qualified_d_vs_c.receipt();
    let d_vs_c = qualified_d_vs_c.fresh();
    validate_c_vs_a_receipt(c_vs_a)?;
    validate_d_vs_c_token(qualified_d_vs_c)?;

    if c_vs_a.subject_digest != d_vs_c.subject_digest
        || c_vs_a.environment_digest != d_vs_c.environment_digest
        || c_vs_a.selected_policy_id != d_vs_c.c_policy_id
        || c_vs_a.holdout_gate_evidence_digest
            != qualified_receipt.parent_holdout_gate_evidence_digest
    {
        return Err(FreshImprovementChainError::CrossLineageMismatch);
    }

    let c_vs_a_positive = c_vs_a.disposition == FreshCvsADisposition::PositiveUnderProtocol;
    let d_vs_c_positive = d_vs_c.disposition == DreamFreshDisposition::PositiveUnderProtocol;
    let integrity_ok = source_integrity_ok(c_vs_a, qualified_d_vs_c);
    let disposition = classify_chain(c_vs_a_positive, d_vs_c_positive, integrity_ok);

    let evidence_digest = chain_evidence_digest(c_vs_a, qualified_d_vs_c, disposition);
    Ok(FreshImprovementChainReceipt {
        schema: SYM_RSI_FRESH_CHAIN_CLAIM_SCHEMA.into(),
        subject_digest: c_vs_a.subject_digest.clone(),
        environment_digest: c_vs_a.environment_digest.clone(),
        c_vs_a_experiment_id: c_vs_a.experiment_id.clone(),
        d_vs_c_experiment_id: d_vs_c.experiment_id.clone(),
        selected_c_policy_id: c_vs_a.selected_policy_id.clone(),
        dream_d_policy_id: d_vs_c.d_policy_id.clone(),
        shared_parent_holdout_gate_evidence_digest:
            c_vs_a.holdout_gate_evidence_digest.clone(),
        c_vs_a_fresh_evidence_digest: c_vs_a.evidence_digest.clone(),
        qualified_d_vs_c_fresh_evidence_digest: qualified_receipt.evidence_digest.clone(),
        d_vs_c_fresh_evidence_digest: d_vs_c.evidence_digest.clone(),
        c_vs_a_disposition: c_vs_a.disposition,
        d_vs_c_disposition: d_vs_c.disposition,
        c_vs_a_positive,
        d_vs_c_positive,
        disposition,
        evidence_digest,
    })
}

fn validate_c_vs_a_receipt(
    receipt: &FreshCvsAReceipt,
) -> Result<(), FreshImprovementChainError> {
    if receipt.schema != SYM_RSI_001_FRESH_EVALUATION_SCHEMA
        || receipt.analysis_rule != SYM_RSI_001_C_VS_A_ANALYSIS_RULE
        || receipt.experiment_id != "SYM-RSI-001"
        || receipt.subject_digest.trim().is_empty()
        || receipt.environment_digest.trim().is_empty()
        || receipt.candidate_family_digest != canonical_candidate_family_digest()
        || receipt.incumbent_policy_id.trim().is_empty()
        || receipt.selected_policy_id.trim().is_empty()
        || receipt.holdout_gate_evidence_digest.trim().is_empty()
        || receipt.evidence_digest.trim().is_empty()
        || !receipt.quality_tolerance.is_finite()
        || receipt.quality_tolerance < 0.0
    {
        return Err(FreshImprovementChainError::InvalidCvsAReceipt);
    }

    if receipt.fresh_seeds_consumed {
        if receipt.pair_count != 12
            || receipt.pairs.len() != 12
            || receipt.domain_summaries.len() != 3
            || receipt.domain_summaries.iter().any(|summary| summary.pair_count != 4)
            || receipt
                .pairs
                .iter()
                .any(|pair| pair.baseline.split != EvaluationSplit::FreshExecution
                    || pair.replay_selected.split != EvaluationSplit::FreshExecution)
        {
            return Err(FreshImprovementChainError::InvalidCvsAReceipt);
        }
        validate_consumed_c_vs_a_numeric_consistency(receipt)?;
    } else if receipt.pair_count != 0
        || !receipt.pairs.is_empty()
        || !receipt.domain_summaries.is_empty()
        || receipt.macro_quality_delta.is_some()
        || receipt.total_evaluator_call_delta.is_some()
        || receipt.worst_domain_quality_delta.is_some()
        || !matches!(
            receipt.disposition,
            FreshCvsADisposition::NoCandidatePromotion | FreshCvsADisposition::BlockedByHoldout
        )
    {
        return Err(FreshImprovementChainError::InvalidCvsAReceipt);
    }

    if recompute_c_vs_a_evidence_digest(receipt) != receipt.evidence_digest {
        return Err(FreshImprovementChainError::InvalidCvsAReceipt);
    }
    Ok(())
}

fn validate_consumed_c_vs_a_numeric_consistency(
    receipt: &FreshCvsAReceipt,
) -> Result<(), FreshImprovementChainError> {
    let macro_quality_delta = receipt
        .domain_summaries
        .iter()
        .map(|summary| summary.mean_quality_delta)
        .sum::<f64>()
        / receipt.domain_summaries.len() as f64;
    let total_evaluator_call_delta = receipt
        .domain_summaries
        .iter()
        .map(|summary| summary.total_evaluator_call_delta)
        .sum::<i128>();
    let worst_domain_quality_delta = receipt
        .domain_summaries
        .iter()
        .map(|summary| summary.mean_quality_delta)
        .fold(f64::INFINITY, f64::min);

    let stored_macro = receipt
        .macro_quality_delta
        .ok_or(FreshImprovementChainError::InvalidCvsAReceipt)?;
    let stored_calls = receipt
        .total_evaluator_call_delta
        .ok_or(FreshImprovementChainError::InvalidCvsAReceipt)?;
    let stored_worst = receipt
        .worst_domain_quality_delta
        .ok_or(FreshImprovementChainError::InvalidCvsAReceipt)?;
    if !approx_eq(stored_macro, macro_quality_delta)
        || stored_calls != total_evaluator_call_delta
        || !approx_eq(stored_worst, worst_domain_quality_delta)
    {
        return Err(FreshImprovementChainError::InvalidCvsAReceipt);
    }

    let expected = if !receipt.zero_safety_constraint_violations
        || !receipt.zero_authority_boundary_violations
        || !macro_quality_delta.is_finite()
        || receipt
            .domain_summaries
            .iter()
            .any(|summary| !summary.mean_quality_delta.is_finite())
    {
        FreshCvsADisposition::IntegrityFailure
    } else if macro_quality_delta < -receipt.quality_tolerance
        || receipt
            .domain_summaries
            .iter()
            .any(|summary| summary.mean_quality_delta < -receipt.quality_tolerance)
    {
        FreshCvsADisposition::QualityNonInferiorityFailed
    } else if macro_quality_delta > 0.0 || total_evaluator_call_delta < 0 {
        FreshCvsADisposition::PositiveUnderProtocol
    } else {
        FreshCvsADisposition::NoStrictGain
    };

    if receipt.disposition != expected {
        return Err(FreshImprovementChainError::InvalidCvsAReceipt);
    }
    Ok(())
}

fn recompute_c_vs_a_evidence_digest(receipt: &FreshCvsAReceipt) -> String {
    let mut hasher = blake3::Hasher::new();
    if receipt.fresh_seeds_consumed {
        hasher.update(b"symthaea.sym-rsi-001.fresh-c-vs-a.v1\0");
    } else {
        hasher.update(b"symthaea.sym-rsi-001.fresh-c-vs-a.empty.v1\0");
    }
    for value in [
        receipt.experiment_id.as_str(),
        receipt.preregistration_digest.as_str(),
        receipt.subject_digest.as_str(),
        receipt.environment_digest.as_str(),
        receipt.holdout_gate_evidence_digest.as_str(),
        receipt.incumbent_policy_id.as_str(),
        receipt.selected_policy_id.as_str(),
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }

    if receipt.fresh_seeds_consumed {
        for pair in &receipt.pairs {
            for value in [
                pair.domain_id.as_str(),
                pair.baseline.evidence_digest.as_str(),
                pair.replay_selected.evidence_digest.as_str(),
            ] {
                hasher.update(&(value.len() as u64).to_le_bytes());
                hasher.update(value.as_bytes());
            }
            hasher.update(&pair.seed.to_le_bytes());
            hasher.update(&pair.contrast.quality_delta.to_bits().to_le_bytes());
            hasher.update(&pair.contrast.evaluator_call_delta.to_le_bytes());
        }
        for summary in &receipt.domain_summaries {
            hasher.update(&(summary.domain_id.len() as u64).to_le_bytes());
            hasher.update(summary.domain_id.as_bytes());
            hasher.update(&summary.mean_quality_delta.to_bits().to_le_bytes());
            hasher.update(&summary.total_evaluator_call_delta.to_le_bytes());
        }
        hasher.update(
            &receipt
                .macro_quality_delta
                .expect("validated consumed receipt has macro quality")
                .to_bits()
                .to_le_bytes(),
        );
        hasher.update(
            &receipt
                .total_evaluator_call_delta
                .expect("validated consumed receipt has evaluator-call delta")
                .to_le_bytes(),
        );
        hasher.update(
            &receipt
                .worst_domain_quality_delta
                .expect("validated consumed receipt has worst-domain quality")
                .to_bits()
                .to_le_bytes(),
        );
    }
    hasher.update(&[c_vs_a_disposition_tag(receipt.disposition)]);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn c_vs_a_disposition_tag(disposition: FreshCvsADisposition) -> u8 {
    match disposition {
        FreshCvsADisposition::NoCandidatePromotion => 0,
        FreshCvsADisposition::BlockedByHoldout => 1,
        FreshCvsADisposition::PositiveUnderProtocol => 2,
        FreshCvsADisposition::QualityNonInferiorityFailed => 3,
        FreshCvsADisposition::NoStrictGain => 4,
        FreshCvsADisposition::IntegrityFailure => 5,
    }
}

fn validate_d_vs_c_token(
    qualified: &QualifiedDreamFresh,
) -> Result<(), FreshImprovementChainError> {
    let receipt = qualified.receipt();
    let fresh = qualified.fresh();
    if fresh.schema != SYM_RSI_001D_FRESH_EVALUATION_SCHEMA
        || fresh.experiment_id != "SYM-RSI-001D"
        || receipt.parent_c_qualification_evidence_digest.trim().is_empty()
        || receipt.qualified_verification_evidence_digest.trim().is_empty()
        || receipt.parent_holdout_gate_evidence_digest.trim().is_empty()
        || receipt.evidence_digest.trim().is_empty()
        || fresh.c_policy_id.trim().is_empty()
        || fresh.d_policy_id.trim().is_empty()
        || fresh.evidence_digest.trim().is_empty()
        || !fresh.quality_tolerance.is_finite()
        || fresh.quality_tolerance < 0.0
        || !fresh.generic_compute_cost_is_environment_calls_only
        || fresh.efficiency_claim_authorized
    {
        return Err(FreshImprovementChainError::InvalidDvsCReceipt);
    }

    if fresh.fresh_seeds_consumed {
        if fresh.pair_count != 12
            || fresh.pairs.len() != 12
            || fresh.domain_summaries.len() != 3
            || fresh.domain_summaries.iter().any(|summary| summary.pair_count != 4)
            || fresh
                .pairs
                .iter()
                .any(|pair| pair.replay_selected.split != EvaluationSplit::FreshExecution
                    || pair.replay_plus_dream.split != EvaluationSplit::FreshExecution)
        {
            return Err(FreshImprovementChainError::InvalidDvsCReceipt);
        }
        validate_consumed_d_vs_c_numeric_consistency(fresh)?;
    } else if fresh.pair_count != 0
        || !fresh.pairs.is_empty()
        || !fresh.domain_summaries.is_empty()
        || fresh.macro_quality_delta.is_some()
        || fresh.worst_domain_quality_delta.is_some()
        || !matches!(
            fresh.disposition,
            DreamFreshDisposition::NoDreamIntervention
                | DreamFreshDisposition::BlockedByVerification
        )
    {
        return Err(FreshImprovementChainError::InvalidDvsCReceipt);
    }
    Ok(())
}

fn validate_consumed_d_vs_c_numeric_consistency(
    fresh: &DreamFreshEvaluationReceipt,
) -> Result<(), FreshImprovementChainError> {
    let macro_quality_delta = fresh
        .domain_summaries
        .iter()
        .map(|summary| summary.mean_quality_delta)
        .sum::<f64>()
        / fresh.domain_summaries.len() as f64;
    let worst_domain_quality_delta = fresh
        .domain_summaries
        .iter()
        .map(|summary| summary.mean_quality_delta)
        .fold(f64::INFINITY, f64::min);
    let total_overrides = fresh
        .domain_summaries
        .iter()
        .map(|summary| summary.d_override_count)
        .sum::<usize>();
    let total_predictions = fresh
        .domain_summaries
        .iter()
        .map(|summary| summary.d_prediction_count)
        .sum::<usize>();
    let total_simulations = fresh
        .domain_summaries
        .iter()
        .map(|summary| summary.d_model_simulation_count)
        .sum::<usize>();

    let stored_macro = fresh
        .macro_quality_delta
        .ok_or(FreshImprovementChainError::InvalidDvsCReceipt)?;
    let stored_worst = fresh
        .worst_domain_quality_delta
        .ok_or(FreshImprovementChainError::InvalidDvsCReceipt)?;
    if !approx_eq(stored_macro, macro_quality_delta)
        || !approx_eq(stored_worst, worst_domain_quality_delta)
        || fresh.total_d_override_count != total_overrides
        || fresh.total_d_prediction_count != total_predictions
        || fresh.total_d_model_simulation_count != total_simulations
    {
        return Err(FreshImprovementChainError::InvalidDvsCReceipt);
    }

    let expected = if !fresh.zero_safety_constraint_violations
        || !fresh.zero_authority_boundary_violations
        || fresh.generated_evidence_promoted
        || !macro_quality_delta.is_finite()
        || fresh
            .domain_summaries
            .iter()
            .any(|summary| !summary.mean_quality_delta.is_finite())
    {
        DreamFreshDisposition::IntegrityFailure
    } else if total_overrides == 0 {
        DreamFreshDisposition::NoDreamIntervention
    } else if macro_quality_delta < -fresh.quality_tolerance
        || fresh
            .domain_summaries
            .iter()
            .any(|summary| summary.mean_quality_delta < -fresh.quality_tolerance)
    {
        DreamFreshDisposition::QualityNonInferiorityFailed
    } else if macro_quality_delta > 0.0 {
        DreamFreshDisposition::PositiveUnderProtocol
    } else {
        DreamFreshDisposition::NoStrictQualityGain
    };

    if fresh.disposition != expected {
        return Err(FreshImprovementChainError::InvalidDvsCReceipt);
    }
    Ok(())
}

fn source_integrity_ok(
    c_vs_a: &FreshCvsAReceipt,
    qualified_d_vs_c: &QualifiedDreamFresh,
) -> bool {
    let d_vs_c = qualified_d_vs_c.fresh();
    let c_pair_set = c_vs_a
        .pairs
        .iter()
        .map(|pair| (pair.domain_id.as_str(), pair.seed))
        .collect::<BTreeSet<_>>();
    let d_pair_set = d_vs_c
        .pairs
        .iter()
        .map(|pair| (pair.domain_id.as_str(), pair.seed))
        .collect::<BTreeSet<_>>();

    let c_shape_ok = if c_vs_a.fresh_seeds_consumed {
        c_pair_set.len() == 12
    } else {
        c_pair_set.is_empty()
    };
    let d_shape_ok = if d_vs_c.fresh_seeds_consumed {
        d_pair_set.len() == 12
    } else {
        d_pair_set.is_empty()
    };

    c_shape_ok
        && d_shape_ok
        && c_vs_a.zero_safety_constraint_violations
        && c_vs_a.zero_authority_boundary_violations
        && d_vs_c.zero_safety_constraint_violations
        && d_vs_c.zero_authority_boundary_violations
        && !d_vs_c.generated_evidence_promoted
}

fn classify_chain(
    c_vs_a_positive: bool,
    d_vs_c_positive: bool,
    integrity_ok: bool,
) -> FreshImprovementChainDisposition {
    if !integrity_ok {
        return FreshImprovementChainDisposition::IntegrityFailure;
    }
    match (c_vs_a_positive, d_vs_c_positive) {
        (true, true) => FreshImprovementChainDisposition::TwoStageFreshImprovementEstablished,
        (true, false) => FreshImprovementChainDisposition::ReplayOnlyFreshImprovement,
        (false, true) => FreshImprovementChainDisposition::DreamIncrementOnly,
        (false, false) => FreshImprovementChainDisposition::NoFreshImprovementChain,
    }
}

fn chain_evidence_digest(
    c_vs_a: &FreshCvsAReceipt,
    qualified_d_vs_c: &QualifiedDreamFresh,
    disposition: FreshImprovementChainDisposition,
) -> String {
    let qualified_receipt = qualified_d_vs_c.receipt();
    let d_vs_c = qualified_d_vs_c.fresh();
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi.fresh-improvement-chain.v1\0");
    for value in [
        c_vs_a.subject_digest.as_str(),
        c_vs_a.environment_digest.as_str(),
        c_vs_a.evidence_digest.as_str(),
        c_vs_a.holdout_gate_evidence_digest.as_str(),
        c_vs_a.selected_policy_id.as_str(),
        qualified_receipt.parent_c_qualification_evidence_digest.as_str(),
        qualified_receipt.parent_holdout_gate_evidence_digest.as_str(),
        qualified_receipt.qualified_verification_evidence_digest.as_str(),
        qualified_receipt.evidence_digest.as_str(),
        d_vs_c.evidence_digest.as_str(),
        d_vs_c.grounded_dream_model_evidence_digest.as_str(),
        d_vs_c.c_policy_id.as_str(),
        d_vs_c.d_policy_id.as_str(),
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.update(&[chain_disposition_tag(disposition)]);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn chain_disposition_tag(disposition: FreshImprovementChainDisposition) -> u8 {
    match disposition {
        FreshImprovementChainDisposition::TwoStageFreshImprovementEstablished => 0,
        FreshImprovementChainDisposition::ReplayOnlyFreshImprovement => 1,
        FreshImprovementChainDisposition::DreamIncrementOnly => 2,
        FreshImprovementChainDisposition::NoFreshImprovementChain => 3,
        FreshImprovementChainDisposition::IntegrityFailure => 4,
    }
}

fn approx_eq(left: f64, right: f64) -> bool {
    left.is_finite() && right.is_finite() && (left - right).abs() <= NUMERIC_EPSILON
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FreshImprovementChainError {
    InvalidCvsAReceipt,
    InvalidDvsCReceipt,
    CrossLineageMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_chain_requires_both_positive_fresh_contrasts() {
        assert_eq!(
            classify_chain(true, true, true),
            FreshImprovementChainDisposition::TwoStageFreshImprovementEstablished
        );
        assert_eq!(
            classify_chain(true, false, true),
            FreshImprovementChainDisposition::ReplayOnlyFreshImprovement
        );
        assert_eq!(
            classify_chain(false, true, true),
            FreshImprovementChainDisposition::DreamIncrementOnly
        );
        assert_eq!(
            classify_chain(false, false, true),
            FreshImprovementChainDisposition::NoFreshImprovementChain
        );
    }

    #[test]
    fn integrity_failure_dominates_positive_flags() {
        assert_eq!(
            classify_chain(true, true, false),
            FreshImprovementChainDisposition::IntegrityFailure
        );
    }

    #[test]
    fn numeric_comparison_is_strict_and_finite() {
        assert!(approx_eq(0.5, 0.5 + 1e-13));
        assert!(!approx_eq(0.5, 0.5 + 1e-8));
        assert!(!approx_eq(f64::NAN, f64::NAN));
    }
}
