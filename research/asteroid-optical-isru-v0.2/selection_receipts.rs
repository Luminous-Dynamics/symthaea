// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Standalone adaptive-selection propensity receipt and audit sampler.
// Run tests with:
//   rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/selection_receipts.rs -o /tmp/isru-selection-tests
//   /tmp/isru-selection-tests

#![forbid(unsafe_code)]

//! Reproducible action-selection receipts for adaptive experiments.
//!
//! The audit sampler randomizes only over candidates already admitted by an
//! external gate. Receipts retain the full candidate set, gate references,
//! target probabilities, policy propensities, random stream identity and
//! evidence snapshot digest. This does not prove that the supplied gate,
//! digest, policy, or target distribution is truthful; it makes omissions
//! and internal inconsistencies testable.
//!
//! The propensity helpers enforce the positivity condition needed for simple
//! importance weighting. They deliberately refuse to manufacture support
//! where a policy had zero probability of selecting a target-relevant action.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

const PROBABILITY_TOLERANCE: f64 = 1.0e-9;

#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub id: String,
    pub admitted: bool,
    pub external_gate_receipt_id: String,
    /// Probability of this candidate in the frozen target design distribution.
    pub target_probability: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidatePropensity {
    pub candidate_id: String,
    pub admitted: bool,
    pub external_gate_receipt_id: String,
    pub target_probability: f64,
    /// p(selection = candidate | declared policy and candidate set).
    pub selection_probability: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectionReceipt {
    pub receipt_id: String,
    pub candidate_set_digest: String,
    pub evidence_snapshot_digest: String,
    pub policy_id: String,
    pub policy_version: String,
    pub policy_config_digest: String,
    pub random_stream_id: String,
    pub random_seed: u64,
    pub selected_candidate_id: String,
    pub candidates: Vec<CandidatePropensity>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReceiptError {
    EmptyReceiptField(&'static str),
    ZeroRandomSeed,
    EmptyCandidateSet,
    EmptyCandidateId,
    EmptyGateReceipt,
    DuplicateCandidate(String),
    InvalidProbability(String),
    IneligibleTargetMass(String),
    TargetDistributionNotNormalized(f64),
    IneligibleSelectionMass(String),
    SelectionDistributionNotNormalized(f64),
    SelectedCandidateMissing,
    SelectedCandidateIneligible,
    SelectedCandidateZeroPropensity,
    NoAdmissibleCandidates,
    NoPositivitySupport(String),
    InvalidWeight,
    EmptyWeights,
    NonFiniteWeight,
}

impl fmt::Display for ReceiptError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for ReceiptError {}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn validate_receipt(receipt: &SelectionReceipt) -> Result<(), ReceiptError> {
    for (name, value) in [
        ("receipt_id", receipt.receipt_id.as_str()),
        (
            "candidate_set_digest",
            receipt.candidate_set_digest.as_str(),
        ),
        (
            "evidence_snapshot_digest",
            receipt.evidence_snapshot_digest.as_str(),
        ),
        ("policy_id", receipt.policy_id.as_str()),
        ("policy_version", receipt.policy_version.as_str()),
        (
            "policy_config_digest",
            receipt.policy_config_digest.as_str(),
        ),
        ("random_stream_id", receipt.random_stream_id.as_str()),
        (
            "selected_candidate_id",
            receipt.selected_candidate_id.as_str(),
        ),
    ] {
        if !nonempty(value) {
            return Err(ReceiptError::EmptyReceiptField(name));
        }
    }

    if receipt.random_seed == 0 {
        return Err(ReceiptError::ZeroRandomSeed);
    }
    if receipt.candidates.is_empty() {
        return Err(ReceiptError::EmptyCandidateSet);
    }

    let mut ids = BTreeSet::new();
    let mut target_sum = 0.0;
    let mut selection_sum = 0.0;
    let mut admitted_count = 0_usize;
    let mut selected: Option<&CandidatePropensity> = None;

    for candidate in &receipt.candidates {
        if !nonempty(&candidate.candidate_id) {
            return Err(ReceiptError::EmptyCandidateId);
        }
        if !ids.insert(candidate.candidate_id.as_str()) {
            return Err(ReceiptError::DuplicateCandidate(
                candidate.candidate_id.clone(),
            ));
        }
        if !nonempty(&candidate.external_gate_receipt_id) {
            return Err(ReceiptError::EmptyGateReceipt);
        }
        if !candidate.target_probability.is_finite()
            || !(0.0..=1.0).contains(&candidate.target_probability)
            || !candidate.selection_probability.is_finite()
            || !(0.0..=1.0).contains(&candidate.selection_probability)
        {
            return Err(ReceiptError::InvalidProbability(
                candidate.candidate_id.clone(),
            ));
        }

        if candidate.admitted {
            admitted_count += 1;
            target_sum += candidate.target_probability;
            selection_sum += candidate.selection_probability;
        } else {
            if candidate.target_probability > PROBABILITY_TOLERANCE {
                return Err(ReceiptError::IneligibleTargetMass(
                    candidate.candidate_id.clone(),
                ));
            }
            if candidate.selection_probability > PROBABILITY_TOLERANCE {
                return Err(ReceiptError::IneligibleSelectionMass(
                    candidate.candidate_id.clone(),
                ));
            }
        }

        if candidate.candidate_id == receipt.selected_candidate_id {
            selected = Some(candidate);
        }
    }

    if admitted_count == 0 {
        return Err(ReceiptError::NoAdmissibleCandidates);
    }
    if (target_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(ReceiptError::TargetDistributionNotNormalized(target_sum));
    }
    if (selection_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(ReceiptError::SelectionDistributionNotNormalized(
            selection_sum,
        ));
    }

    let Some(selected) = selected else {
        return Err(ReceiptError::SelectedCandidateMissing);
    };
    if !selected.admitted {
        return Err(ReceiptError::SelectedCandidateIneligible);
    }
    if selected.selection_probability <= 0.0 {
        return Err(ReceiptError::SelectedCandidateZeroPropensity);
    }

    Ok(())
}

/// Select one action uniformly from the already-admitted candidate set.
///
/// The sampler uses rejection sampling for unbiased bounded integers. Candidate
/// order is retained in the receipt and is therefore part of the reproducible
/// input contract. The caller supplies immutable digests and gate receipt IDs.
pub fn select_uniform_audit_candidate(
    candidates: &[Candidate],
    receipt_id: &str,
    candidate_set_digest: &str,
    evidence_snapshot_digest: &str,
    policy_config_digest: &str,
    random_stream_id: &str,
    seed: u64,
) -> Result<SelectionReceipt, ReceiptError> {
    if seed == 0 {
        return Err(ReceiptError::ZeroRandomSeed);
    }
    if candidates.is_empty() {
        return Err(ReceiptError::EmptyCandidateSet);
    }

    let admitted: Vec<&Candidate> = candidates
        .iter()
        .filter(|candidate| candidate.admitted)
        .collect();
    if admitted.is_empty() {
        return Err(ReceiptError::NoAdmissibleCandidates);
    }

    let target_sum: f64 = admitted
        .iter()
        .map(|candidate| candidate.target_probability)
        .sum();
    if (target_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(ReceiptError::TargetDistributionNotNormalized(target_sum));
    }

    let mut state = seed;
    let selected_index = bounded_random(&mut state, admitted.len() as u64) as usize;
    let selected_candidate_id = admitted[selected_index].id.clone();
    let uniform_probability = 1.0 / admitted.len() as f64;

    let receipt = SelectionReceipt {
        receipt_id: receipt_id.to_string(),
        candidate_set_digest: candidate_set_digest.to_string(),
        evidence_snapshot_digest: evidence_snapshot_digest.to_string(),
        policy_id: "uniform_admitted_audit".to_string(),
        policy_version: "0.1".to_string(),
        policy_config_digest: policy_config_digest.to_string(),
        random_stream_id: random_stream_id.to_string(),
        random_seed: seed,
        selected_candidate_id,
        candidates: candidates
            .iter()
            .map(|candidate| CandidatePropensity {
                candidate_id: candidate.id.clone(),
                admitted: candidate.admitted,
                external_gate_receipt_id: candidate.external_gate_receipt_id.clone(),
                target_probability: candidate.target_probability,
                selection_probability: if candidate.admitted {
                    uniform_probability
                } else {
                    0.0
                },
            })
            .collect(),
    };

    validate_receipt(&receipt)?;
    Ok(receipt)
}

fn next_random(state: &mut u64) -> u64 {
    // xorshift64*: deterministic stream generator. Reproducibility, not
    // cryptographic unpredictability, is the requirement for this fixture.
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn bounded_random(state: &mut u64, upper_exclusive: u64) -> u64 {
    assert!(upper_exclusive > 0);
    let threshold = upper_exclusive.wrapping_neg() % upper_exclusive;
    loop {
        let value = next_random(state);
        if value >= threshold {
            return value % upper_exclusive;
        }
    }
}

/// Calculate one target-to-selection importance weight. Zero selection
/// probability for positive target mass is a positivity failure, not a
/// numerical inconvenience to be fixed by clipping.
pub fn importance_weight(
    target_probability: f64,
    selection_probability: f64,
) -> Result<f64, ReceiptError> {
    if !target_probability.is_finite()
        || !(0.0..=1.0).contains(&target_probability)
        || !selection_probability.is_finite()
        || !(0.0..=1.0).contains(&selection_probability)
    {
        return Err(ReceiptError::InvalidWeight);
    }
    if target_probability > 0.0 && selection_probability == 0.0 {
        return Err(ReceiptError::NoPositivitySupport(
            "positive target mass has zero selection probability".to_string(),
        ));
    }
    if target_probability == 0.0 {
        return Ok(0.0);
    }
    Ok(target_probability / selection_probability)
}

/// Kish effective sample size for a realized vector of non-negative weights.
///
/// This function does not claim independence. For ISRU campaigns, callers
/// should first aggregate repeated measurements by the appropriate batch,
/// specimen, acquisition, or common-cause cluster.
pub fn effective_sample_size(weights: &[f64]) -> Result<f64, ReceiptError> {
    if weights.is_empty() {
        return Err(ReceiptError::EmptyWeights);
    }
    let mut sum = 0.0;
    let mut sum_squares = 0.0;
    for weight in weights.iter().copied() {
        if !weight.is_finite() {
            return Err(ReceiptError::NonFiniteWeight);
        }
        if weight < 0.0 {
            return Err(ReceiptError::InvalidWeight);
        }
        sum += weight;
        sum_squares += weight * weight;
    }
    if sum_squares == 0.0 {
        return Ok(0.0);
    }
    Ok(sum * sum / sum_squares)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates() -> Vec<Candidate> {
        vec![
            Candidate {
                id: "a".to_string(),
                admitted: true,
                external_gate_receipt_id: "gate-a".to_string(),
                target_probability: 0.5,
            },
            Candidate {
                id: "b".to_string(),
                admitted: true,
                external_gate_receipt_id: "gate-b".to_string(),
                target_probability: 0.5,
            },
            Candidate {
                id: "blocked".to_string(),
                admitted: false,
                external_gate_receipt_id: "gate-blocked".to_string(),
                target_probability: 0.0,
            },
        ]
    }

    fn receipt(seed: u64) -> SelectionReceipt {
        select_uniform_audit_candidate(
            &candidates(),
            "receipt-1",
            "sha256:candidate-set",
            "sha256:evidence-snapshot",
            "sha256:policy-config",
            "campaign-rng-1",
            seed,
        )
        .unwrap()
    }

    #[test]
    fn uniform_audit_sampler_never_selects_a_gate_rejected_candidate() {
        for seed in 1..100 {
            let selected = receipt(seed);
            assert_ne!(selected.selected_candidate_id, "blocked");
            let blocked = selected
                .candidates
                .iter()
                .find(|candidate| candidate.candidate_id == "blocked")
                .unwrap();
            assert_eq!(blocked.selection_probability, 0.0);
        }
    }

    #[test]
    fn same_seed_and_candidate_order_replay_exactly() {
        assert_eq!(receipt(77), receipt(77));
    }

    #[test]
    fn different_seed_changes_at_least_one_selection_in_fixture_range() {
        let selected: BTreeSet<String> = (1..20)
            .map(|seed| receipt(seed).selected_candidate_id)
            .collect();
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn receipt_retains_all_candidates_gate_receipts_and_digests() {
        let value = receipt(44);
        assert_eq!(value.candidates.len(), 3);
        assert_eq!(value.candidate_set_digest, "sha256:candidate-set");
        assert_eq!(value.evidence_snapshot_digest, "sha256:evidence-snapshot");
        assert!(value
            .candidates
            .iter()
            .all(|candidate| !candidate.external_gate_receipt_id.is_empty()));
    }

    #[test]
    fn target_relevant_action_with_zero_propensity_is_rejected() {
        assert!(matches!(
            importance_weight(0.2, 0.0),
            Err(ReceiptError::NoPositivitySupport(_))
        ));
    }

    #[test]
    fn importance_weight_is_target_mass_over_selection_propensity() {
        assert!((importance_weight(0.25, 0.5).unwrap() - 0.5).abs() < 1.0e-12);
        assert_eq!(importance_weight(0.0, 0.0).unwrap(), 0.0);
    }

    #[test]
    fn effective_sample_size_matches_kish_hand_calculation() {
        let ess = effective_sample_size(&[1.0, 1.0, 2.0]).unwrap();
        assert!((ess - 16.0 / 6.0).abs() < 1.0e-12);
    }

    #[test]
    fn receipt_rejects_distribution_mass_on_a_blocked_action() {
        let mut value = receipt(44);
        let blocked = value
            .candidates
            .iter_mut()
            .find(|candidate| candidate.candidate_id == "blocked")
            .unwrap();
        blocked.selection_probability = 0.1;
        assert!(matches!(
            validate_receipt(&value),
            Err(ReceiptError::IneligibleSelectionMass(_))
        ));
    }

    #[test]
    fn receipt_rejects_missing_positive_probability_support() {
        let mut value = receipt(44);
        let selected = value.selected_candidate_id.clone();
        let chosen = value
            .candidates
            .iter_mut()
            .find(|candidate| candidate.candidate_id == selected)
            .unwrap();
        chosen.selection_probability = 0.0;
        let other = value
            .candidates
            .iter_mut()
            .find(|candidate| candidate.candidate_id != selected && candidate.admitted)
            .unwrap();
        other.selection_probability = 1.0;
        value.selected_candidate_id = selected;
        assert!(matches!(
            validate_receipt(&value),
            Err(ReceiptError::SelectedCandidateZeroPropensity)
        ));
    }

    #[test]
    fn effective_sample_size_rejects_invalid_weights() {
        assert_eq!(effective_sample_size(&[]), Err(ReceiptError::EmptyWeights));
        assert_eq!(
            effective_sample_size(&[1.0, f64::NAN]),
            Err(ReceiptError::NonFiniteWeight)
        );
        assert_eq!(
            effective_sample_size(&[1.0, -1.0]),
            Err(ReceiptError::InvalidWeight)
        );
    }

    #[test]
    fn audit_sampler_rejects_zero_seed_and_missing_gate_receipt() {
        assert_eq!(
            select_uniform_audit_candidate(&candidates(), "r", "c", "e", "p", "rng", 0),
            Err(ReceiptError::ZeroRandomSeed)
        );
        let mut invalid = candidates();
        invalid[0].external_gate_receipt_id.clear();
        assert!(matches!(
            select_uniform_audit_candidate(&invalid, "r", "c", "e", "p", "rng", 8),
            Err(ReceiptError::EmptyGateReceipt)
        ));
    }
}
