// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public token-gated synthesis for the two-stage SYM-RSI fresh evidence chain.
//!
//! The inner claim engine validates the frozen numeric evidence and source hashes.
//! This outer gate additionally requires private-constructor qualification tokens
//! for both fresh stages, so serialized receipts alone cannot manufacture a
//! two-stage recursive-improvement claim.

use super::sym_rsi_c_fresh_gate::QualifiedReplayFresh;
use super::sym_rsi_dream_parent_gate::QualifiedDreamFresh;
use super::sym_rsi_fresh_chain_claim::{
    build_fresh_improvement_chain_receipt, FreshImprovementChainError,
    FreshImprovementChainReceipt,
};
use serde::{Deserialize, Serialize};

pub const SYM_RSI_QUALIFIED_FRESH_CHAIN_SCHEMA: &str =
    "symthaea.sym-rsi.qualified-fresh-improvement-chain.v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualifiedFreshImprovementChainReceipt {
    pub schema: String,
    pub parent_c_qualification_evidence_digest: String,
    pub parent_holdout_gate_evidence_digest: String,
    pub qualified_c_vs_a_fresh_evidence_digest: String,
    pub qualified_d_vs_c_fresh_evidence_digest: String,
    pub chain: FreshImprovementChainReceipt,
    pub evidence_digest: String,
}

/// Build the only public two-stage fresh-improvement synthesis receipt.
///
/// This function consumes no seeds, executes no fixture, and performs no policy
/// selection. Both fresh results must already exist behind their respective
/// private-constructor qualification tokens.
pub fn build_qualified_fresh_improvement_chain(
    qualified_c_vs_a: &QualifiedReplayFresh,
    qualified_d_vs_c: &QualifiedDreamFresh,
) -> Result<QualifiedFreshImprovementChainReceipt, QualifiedFreshChainError> {
    validate_shared_parent_lineage(qualified_c_vs_a, qualified_d_vs_c)?;

    let chain = build_fresh_improvement_chain_receipt(
        qualified_c_vs_a.fresh(),
        qualified_d_vs_c,
    )
    .map_err(QualifiedFreshChainError::InnerClaim)?;

    let c_receipt = qualified_c_vs_a.receipt();
    let d_receipt = qualified_d_vs_c.receipt();
    let evidence_digest = qualified_chain_digest(c_receipt, d_receipt, &chain);

    Ok(QualifiedFreshImprovementChainReceipt {
        schema: SYM_RSI_QUALIFIED_FRESH_CHAIN_SCHEMA.into(),
        parent_c_qualification_evidence_digest:
            c_receipt.parent_c_qualification_evidence_digest.clone(),
        parent_holdout_gate_evidence_digest:
            c_receipt.parent_holdout_gate_evidence_digest.clone(),
        qualified_c_vs_a_fresh_evidence_digest: c_receipt.evidence_digest.clone(),
        qualified_d_vs_c_fresh_evidence_digest: d_receipt.evidence_digest.clone(),
        chain,
        evidence_digest,
    })
}

fn validate_shared_parent_lineage(
    qualified_c_vs_a: &QualifiedReplayFresh,
    qualified_d_vs_c: &QualifiedDreamFresh,
) -> Result<(), QualifiedFreshChainError> {
    let c = qualified_c_vs_a.receipt();
    let d = qualified_d_vs_c.receipt();
    let c_fresh = qualified_c_vs_a.fresh();
    let d_fresh = qualified_d_vs_c.fresh();

    if c.parent_c_qualification_evidence_digest.trim().is_empty()
        || c.parent_holdout_gate_evidence_digest.trim().is_empty()
        || c.training_selection_evidence_digest.trim().is_empty()
        || c.evidence_digest.trim().is_empty()
        || d.parent_c_qualification_evidence_digest.trim().is_empty()
        || d.parent_holdout_gate_evidence_digest.trim().is_empty()
        || d.qualified_verification_evidence_digest.trim().is_empty()
        || d.evidence_digest.trim().is_empty()
        || c.parent_c_qualification_evidence_digest
            != d.parent_c_qualification_evidence_digest
        || c.parent_holdout_gate_evidence_digest
            != d.parent_holdout_gate_evidence_digest
        || c.training_selection_evidence_digest != d_fresh.c_selection_evidence_digest
        || c_fresh.holdout_gate_evidence_digest != c.parent_holdout_gate_evidence_digest
        || c_fresh.selected_policy_id != d_fresh.c_policy_id
        || c_fresh.subject_digest != d_fresh.subject_digest
        || c_fresh.environment_digest != d_fresh.environment_digest
    {
        return Err(QualifiedFreshChainError::SharedParentLineageMismatch);
    }
    Ok(())
}

fn qualified_chain_digest(
    c: &super::sym_rsi_c_fresh_gate::ParentQualifiedReplayFreshReceipt,
    d: &super::sym_rsi_dream_parent_gate::ParentQualifiedDreamFreshReceipt,
    chain: &FreshImprovementChainReceipt,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi.qualified-fresh-improvement-chain.v1\0");
    for value in [
        c.parent_c_qualification_evidence_digest.as_str(),
        c.parent_holdout_gate_evidence_digest.as_str(),
        c.training_selection_evidence_digest.as_str(),
        c.evidence_digest.as_str(),
        d.parent_c_qualification_evidence_digest.as_str(),
        d.parent_holdout_gate_evidence_digest.as_str(),
        d.qualified_verification_evidence_digest.as_str(),
        d.evidence_digest.as_str(),
        chain.evidence_digest.as_str(),
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

#[derive(Debug)]
pub enum QualifiedFreshChainError {
    SharedParentLineageMismatch,
    InnerClaim(FreshImprovementChainError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outer_chain_digest_binds_both_qualified_stages() {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea.sym-rsi.qualified-fresh-improvement-chain.v1\0");
        for value in [
            "parent",
            "holdout",
            "selection",
            "qualified-c",
            "parent",
            "holdout",
            "verification",
            "qualified-d",
            "inner-chain",
        ] {
            hasher.update(&(value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        let digest = format!("blake3:{}", hasher.finalize().to_hex());
        assert!(digest.starts_with("blake3:"));
    }
}
