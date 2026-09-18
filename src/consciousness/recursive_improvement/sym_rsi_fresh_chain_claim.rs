// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Claim synthesis for the two-stage SYM-RSI fresh evidence chain.
//!
//! This module consumes no environment seeds and performs no selection. It combines
//! already-frozen C-vs-A and D-vs-C fresh receipts so a narrow incremental result
//! cannot be mislabeled as a complete recursive-improvement chain.

use super::sym_rsi_dream_fresh_evaluation::{
    DreamFreshDisposition, SYM_RSI_001D_FRESH_EVALUATION_SCHEMA,
};
use super::sym_rsi_dream_parent_gate::ParentQualifiedDreamFreshReceipt;
use super::sym_rsi_experiment::EvaluationSplit;
use super::sym_rsi_fresh_evaluation::{
    FreshCvsADisposition, FreshCvsAReceipt, SYM_RSI_001_C_VS_A_ANALYSIS_RULE,
    SYM_RSI_001_FRESH_EVALUATION_SCHEMA,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const SYM_RSI_FRESH_CHAIN_CLAIM_SCHEMA: &str =
    "symthaea.sym-rsi.fresh-improvement-chain.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshImprovementChainDisposition {
    /// Both preregistered fresh contrasts passed: C > A and D > C.
    TwoStageFreshImprovementEstablished,
    /// C > A passed, but D > C did not establish a strict fresh gain.
    ReplayOnlyFreshImprovement,
    /// D > C passed, but C > A did not establish the first fresh improvement stage.
    DreamIncrementOnly,
    /// Neither fresh contrast established its positive preregistered result.
    NoFreshImprovementChain,
    /// Receipts are inconsistent, incomplete, or violate integrity boundaries.
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
pub fn build_fresh_improvement_chain_receipt(
    c_vs_a: &FreshCvsAReceipt,
    qualified_d_vs_c: &ParentQualifiedDreamFreshReceipt,
) -> Result<FreshImprovementChainReceipt, FreshImprovementChainError> {
    let d_vs_c = &qualified_d_vs_c.fresh;
    validate_c_vs_a_receipt(c_vs_a)?;
    validate_d_vs_c_receipt(qualified_d_vs_c)?;

    if c_vs_a.subject_digest != d_vs_c.subject_digest
        || c_vs_a.environment_digest != d_vs_c.environment_digest
        || c_vs_a.selected_policy_id != d_vs_c.c_policy_id
        || c_vs_a.holdout_gate_evidence_digest
            != qualified_d_vs_c.parent_holdout_gate_evidence_digest
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
        qualified_d_vs_c_fresh_evidence_digest: qualified_d_vs_c.evidence_digest.clone(),
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
        || receipt.selected_policy_id.trim().is_empty()
        || receipt.holdout_gate_evidence_digest.trim().is_empty()
        || receipt.evidence_digest.trim().is_empty()
    {
        return Err(FreshImprovementChainError::InvalidCvsAReceipt);
    }

    if receipt.fresh_seeds_consumed {
        if receipt.pair_count != 12
            || receipt.pairs.len() != 12
            || receipt.domain_summaries.len() != 3
            || receipt
                .pairs
                .iter()
                .any(|pair| pair.baseline.split != EvaluationSplit::FreshExecution
                    || pair.replay_selected.split != EvaluationSplit::FreshExecution)
        {
            return Err(FreshImprovementChainError::InvalidCvsAReceipt);
        }
    } else if receipt.pair_count != 0
        || !receipt.pairs.is_empty()
        || !receipt.domain_summaries.is_empty()
        || receipt.macro_quality_delta.is_some()
        || receipt.total_evaluator_call_delta.is_some()
        || receipt.worst_domain_quality_delta.is_some()
    {
        return Err(FreshImprovementChainError::InvalidCvsAReceipt);
    }
    Ok(())
}

fn validate_d_vs_c_receipt(
    receipt: &ParentQualifiedDreamFreshReceipt,
) -> Result<(), FreshImprovementChainError> {
    let fresh = &receipt.fresh;
    if fresh.schema != SYM_RSI_001D_FRESH_EVALUATION_SCHEMA
        || fresh.experiment_id != "SYM-RSI-001D"
        || receipt.parent_c_qualification_evidence_digest.trim().is_empty()
        || receipt.qualified_verification_evidence_digest.trim().is_empty()
        || receipt.parent_holdout_gate_evidence_digest.trim().is_empty()
        || receipt.evidence_digest.trim().is_empty()
        || fresh.c_policy_id.trim().is_empty()
        || fresh.d_policy_id.trim().is_empty()
        || fresh.evidence_digest.trim().is_empty()
    {
        return Err(FreshImprovementChainError::InvalidDvsCReceipt);
    }

    if fresh.fresh_seeds_consumed {
        if fresh.pair_count != 12
            || fresh.pairs.len() != 12
            || fresh.domain_summaries.len() != 3
            || fresh
                .pairs
                .iter()
                .any(|pair| pair.replay_selected.split != EvaluationSplit::FreshExecution
                    || pair.replay_plus_dream.split != EvaluationSplit::FreshExecution)
        {
            return Err(FreshImprovementChainError::InvalidDvsCReceipt);
        }
    } else if fresh.pair_count != 0
        || !fresh.pairs.is_empty()
        || !fresh.domain_summaries.is_empty()
        || fresh.macro_quality_delta.is_some()
        || fresh.worst_domain_quality_delta.is_some()
    {
        return Err(FreshImprovementChainError::InvalidDvsCReceipt);
    }
    Ok(())
}

fn source_integrity_ok(
    c_vs_a: &FreshCvsAReceipt,
    qualified_d_vs_c: &ParentQualifiedDreamFreshReceipt,
) -> bool {
    let d_vs_c = &qualified_d_vs_c.fresh;
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
    qualified_d_vs_c: &ParentQualifiedDreamFreshReceipt,
    disposition: FreshImprovementChainDisposition,
) -> String {
    let d_vs_c = &qualified_d_vs_c.fresh;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi.fresh-improvement-chain.v1\0");
    for value in [
        c_vs_a.subject_digest.as_str(),
        c_vs_a.environment_digest.as_str(),
        c_vs_a.evidence_digest.as_str(),
        c_vs_a.holdout_gate_evidence_digest.as_str(),
        c_vs_a.selected_policy_id.as_str(),
        qualified_d_vs_c.parent_c_qualification_evidence_digest.as_str(),
        qualified_d_vs_c.parent_holdout_gate_evidence_digest.as_str(),
        qualified_d_vs_c.qualified_verification_evidence_digest.as_str(),
        qualified_d_vs_c.evidence_digest.as_str(),
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
}
