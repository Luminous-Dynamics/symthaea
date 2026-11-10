// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Anytime-valid, family-wise non-inferiority guardrails for WeightProgram candidates.
//!
//! An aggregate candidate improvement must not conceal a severe regression on a
//! preregistered task family, fault mechanism, machine class or protected capability.
//! This module accepts paired baseline/candidate losses assigned to frozen family
//! manifests and computes time-uniform Hoeffding intervals for every family.
//!
//! The simultaneous error allocation spends one campaign alpha budget across:
//! candidate ordinal j, registered family f, and interim look n:
//!
//! alpha[j,f,n] = alpha_campaign / (j(j+1) * F * n(n+1))
//!
//! The reciprocal-product sums over j and n are each one; Bonferroni allocation
//! across the fixed F-family roster therefore bounds the total family/look/candidate
//! error probability by campaign_alpha. Each paired improvement is in [-1, 1].
//!
//! Acceptance requires every family lower confidence bound to exceed its frozen
//! non-inferiority threshold (-margin). A candidate is rejected for a family only
//! when that family's upper bound is below the same threshold. Otherwise it remains
//! unresolved until evidence arrives or its preregistered maximum is exhausted.
//!
//! This is a statistical boundary, not an episode-provenance oracle. The campaign
//! must authenticate source record identity, group assignment, shard custody,
//! independence/conditional-mean assumptions and baseline/candidate losses.
//! Repeated windows from one machine/run are not automatically independent episodes.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

const DIGEST_LENGTH: usize = 64;
const MAX_FAMILIES: usize = 256;
const MAX_EPISODES_PER_FAMILY: usize = 100_000;
const LOSS_TOLERANCE: f64 = 1e-12;

/// Preregistered definition of one protected task/fault family.
///
/// Episode IDs must be stable source-record identities, sorted and unique after
/// plan construction. A single episode ID may belong to only one family in v1.
/// The margin is the maximum tolerated degradation in normalized loss units.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyGuardrailSpec {
    /// Stable family identity, e.g. "bearing.cross-load" or "protected.task.math".
    pub family_id: String,
    /// Maximum tolerated loss increase, expressed as baseline-minus-candidate >= -margin.
    pub non_inferiority_margin: f64,
    /// Minimum paired episodes required before this family can pass.
    pub minimum_observations: u64,
    /// Maximum paired episodes permitted for this family.
    pub maximum_observations: u64,
    /// Frozen eligible confirmatory episode identities for this family.
    pub episode_ids: Vec<String>,
}

/// Immutable, canonical family roster and episode assignment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyGuardrailPlan {
    families: Vec<FamilyGuardrailSpec>,
    plan_digest: String,
}

impl FamilyGuardrailPlan {
    /// Validate and canonically freeze the family roster and episode manifests.
    pub fn new(
        mut families: Vec<FamilyGuardrailSpec>,
    ) -> Result<Self, FamilyGuardrailError> {
        if families.is_empty() || families.len() > MAX_FAMILIES {
            return Err(FamilyGuardrailError::InvalidFamilyCount {
                actual: families.len(),
                maximum: MAX_FAMILIES,
            });
        }
        families.sort_by(|left, right| left.family_id.cmp(&right.family_id));
        let mut family_ids = BTreeSet::new();
        let mut episode_ids = BTreeSet::new();

        for family in &mut families {
            if family.family_id.trim().is_empty()
                || !family_ids.insert(family.family_id.clone())
            {
                return Err(FamilyGuardrailError::InvalidFamilyId {
                    family_id: family.family_id.clone(),
                });
            }
            if !family.non_inferiority_margin.is_finite()
                || !(0.0..=1.0).contains(&family.non_inferiority_margin)
            {
                return Err(FamilyGuardrailError::InvalidNonInferiorityMargin {
                    family_id: family.family_id.clone(),
                });
            }
            if family.minimum_observations == 0
                || family.minimum_observations > family.maximum_observations
                || family.maximum_observations > MAX_EPISODES_PER_FAMILY as u64
            {
                return Err(FamilyGuardrailError::InvalidObservationLimits {
                    family_id: family.family_id.clone(),
                });
            }
            if family.episode_ids.len() < family.maximum_observations as usize {
                return Err(FamilyGuardrailError::InsufficientManifestEpisodes {
                    family_id: family.family_id.clone(),
                    required: family.maximum_observations,
                    actual: family.episode_ids.len(),
                });
            }
            if family.episode_ids.iter().any(|id| id.trim().is_empty()) {
                return Err(FamilyGuardrailError::EmptyEpisodeId {
                    family_id: family.family_id.clone(),
                });
            }
            family.episode_ids.sort();
            if family.episode_ids.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err(FamilyGuardrailError::DuplicateEpisodeId {
                    family_id: family.family_id.clone(),
                    episode_id: family
                        .episode_ids
                        .windows(2)
                        .find(|pair| pair[0] == pair[1])
                        .map(|pair| pair[0].clone())
                        .unwrap_or_default(),
                });
            }
            for episode_id in &family.episode_ids {
                if !episode_ids.insert(episode_id.clone()) {
                    return Err(FamilyGuardrailError::EpisodeAssignedToMultipleFamilies {
                        episode_id: episode_id.clone(),
                    });
                }
            }
        }

        let encoded = serde_json::to_vec(&families)
            .map_err(|_| FamilyGuardrailError::PlanSerialization)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-family-noninferiority-plan-v1\0");
        hasher.update(&(encoded.len() as u64).to_le_bytes());
        hasher.update(&encoded);
        let plan_digest = hasher.finalize().to_hex().to_string();

        Ok(Self {
            families,
            plan_digest,
        })
    }

    /// Return the canonical, digest-bound family specification.
    pub fn families(&self) -> &[FamilyGuardrailSpec] {
        &self.families
    }

    /// Return the digest of family definitions, margins and episode assignments.
    pub fn plan_digest(&self) -> &str {
        &self.plan_digest
    }

    /// Revalidate deserialized or externally restored plan state.
    pub fn validate(&self) -> Result<(), FamilyGuardrailError> {
        let reconstructed = Self::new(self.families.clone())?;
        if reconstructed.plan_digest != self.plan_digest {
            return Err(FamilyGuardrailError::PlanDigestMismatch);
        }
        Ok(())
    }
}

/// One paired loss observation assigned to exactly one frozen family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyGuardrailObservation {
    /// Stable source-record/episode identity.
    pub episode_id: String,
    /// Frozen family identity from the plan.
    pub family_id: String,
    /// Baseline loss on this same episode, normalized to [0, 1].
    pub baseline_loss: f64,
    /// Candidate loss on this same episode, normalized to [0, 1].
    pub candidate_loss: f64,
}

/// Overall status of the family-level candidate gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FamilyGuardrailStatus {
    /// One or more family tests still require confirmatory observations.
    Collecting,
    /// Every family has simultaneously cleared its non-inferiority margin.
    AllFamiliesNonInferior,
    /// At least one family has evidence of a regression beyond its margin.
    RegressionDetected,
    /// At least one family reached its observation cap without resolving.
    InconclusiveAtBudget,
}

/// Status of one family-specific non-inferiority test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FamilyNonInferiorityStatus {
    /// The family has not reached its registered minimum sample count.
    InsufficientEvidence,
    /// Its lower confidence bound is above the non-inferiority threshold.
    NonInferior,
    /// Its upper confidence bound is below the non-inferiority threshold.
    RegressionEstablished,
    /// The evidence is not decisive yet and the family can receive more samples.
    Collecting,
    /// The maximum sample count was reached without a decisive result.
    UnresolvedAtBudget,
}

/// Sequential confidence-bound report for one preregistered family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyNonInferiorityResult {
    pub family_id: String,
    pub non_inferiority_margin: f64,
    pub observation_count: u64,
    pub minimum_observations: u64,
    pub maximum_observations: u64,
    pub mean_improvement: Option<f64>,
    pub lower_confidence_bound: Option<f64>,
    pub upper_confidence_bound: Option<f64>,
    pub alpha_spent_at_current_look: Option<f64>,
    pub status: FamilyNonInferiorityStatus,
}

/// Full report for one candidate's fixed family roster.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyGuardrailReport {
    pub format_version: u32,
    pub candidate_index: u64,
    pub campaign_alpha: f64,
    pub candidate_alpha_all_families: f64,
    pub family_count: usize,
    pub total_observation_count: u64,
    pub plan_digest: String,
    /// True: the alpha allocation covers candidates, families and interim looks.
    pub simultaneous_family_bounds: bool,
    pub status: FamilyGuardrailStatus,
    pub families: Vec<FamilyNonInferiorityResult>,
    pub report_digest: String,
}

/// Fail-closed validation and sequential-evaluation errors.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum FamilyGuardrailError {
    #[error("family count must be in 1..={maximum}, received {actual}")]
    InvalidFamilyCount { actual: usize, maximum: usize },
    #[error("family identifier is empty or duplicated: {family_id}")]
    InvalidFamilyId { family_id: String },
    #[error("non-inferiority margin is invalid for family {family_id}")]
    InvalidNonInferiorityMargin { family_id: String },
    #[error("observation limits are invalid for family {family_id}")]
    InvalidObservationLimits { family_id: String },
    #[error("family {family_id} requires {required} manifest episodes, received {actual}")]
    InsufficientManifestEpisodes {
        family_id: String,
        required: u64,
        actual: usize,
    },
    #[error("family {family_id} contains an empty episode identifier")]
    EmptyEpisodeId { family_id: String },
    #[error("family {family_id} contains duplicate episode identifier {episode_id}")]
    DuplicateEpisodeId { family_id: String, episode_id: String },
    #[error("episode {episode_id} is assigned to more than one family")]
    EpisodeAssignedToMultipleFamilies { episode_id: String },
    #[error("failed to serialize frozen family plan")]
    PlanSerialization,
    #[error("family-plan digest does not match its contents")]
    PlanDigestMismatch,
    #[error("campaign alpha must be finite and in (0, 1)")]
    InvalidCampaignAlpha,
    #[error("candidate ordinal must be positive")]
    InvalidCandidateIndex,
    #[error("unknown family identifier: {0}")]
    UnknownFamily(String),
    #[error("episode {episode_id} is not in the frozen manifest for family {family_id}")]
    EpisodeNotInFamilyManifest { family_id: String, episode_id: String },
    #[error("episode {0} was already consumed by this candidate")]
    DuplicateObservation(String),
    #[error("family {0} has exhausted its registered observation budget")]
    FamilyObservationBudgetExhausted(String),
    #[error("loss values must be finite and in [0, 1]")]
    InvalidLoss,
    #[error("a terminal family-guardrail decision has already been reached")]
    AlreadyTerminal,
    #[error("failed to serialize family-guardrail report")]
    ReportSerialization,
    #[error("family-guardrail report digest does not match its contents")]
    ReportDigestMismatch,
}

/// Mutable candidate-local evidence collector. It cannot mutate a model or sign
/// an acceptance receipt; it only reports family-specific statistical evidence.
#[derive(Debug, Clone)]
pub struct FamilyNonInferiorityGate {
    plan: FamilyGuardrailPlan,
    candidate_index: u64,
    campaign_alpha: f64,
    family_state: BTreeMap<String, FamilyAccumulator>,
    seen_episode_ids: BTreeSet<String>,
    total_observations: u64,
    terminal: bool,
}

#[derive(Debug, Clone, Default)]
struct FamilyAccumulator {
    count: u64,
    mean_improvement: f64,
}

impl FamilyNonInferiorityGate {
    /// Start a candidate-specific guardrail stream using the campaign's ordinal.
    ///
    /// The candidate index must be assigned by the existing sequential campaign
    /// ledger. This gate does not create a second candidate ledger or alpha budget.
    pub fn new(
        plan: FamilyGuardrailPlan,
        campaign_alpha: f64,
        candidate_index: u64,
    ) -> Result<Self, FamilyGuardrailError> {
        plan.validate()?;
        if !campaign_alpha.is_finite() || campaign_alpha <= 0.0 || campaign_alpha >= 1.0 {
            return Err(FamilyGuardrailError::InvalidCampaignAlpha);
        }
        if candidate_index == 0 {
            return Err(FamilyGuardrailError::InvalidCandidateIndex);
        }
        let family_state = plan
            .families
            .iter()
            .map(|family| (family.family_id.clone(), FamilyAccumulator::default()))
            .collect();
        Ok(Self {
            plan,
            candidate_index,
            campaign_alpha,
            family_state,
            seen_episode_ids: BTreeSet::new(),
            total_observations: 0,
            terminal: false,
        })
    }

    /// Consume one paired episode and return the updated simultaneous report.
    pub fn observe(
        &mut self,
        observation: FamilyGuardrailObservation,
    ) -> Result<FamilyGuardrailReport, FamilyGuardrailError> {
        if self.terminal {
            return Err(FamilyGuardrailError::AlreadyTerminal);
        }
        let family = self
            .plan
            .families
            .iter()
            .find(|family| family.family_id == observation.family_id)
            .ok_or_else(|| FamilyGuardrailError::UnknownFamily(observation.family_id.clone()))?;

        if family
            .episode_ids
            .binary_search(&observation.episode_id)
            .is_err()
        {
            return Err(FamilyGuardrailError::EpisodeNotInFamilyManifest {
                family_id: observation.family_id,
                episode_id: observation.episode_id,
            });
        }
        if self.seen_episode_ids.contains(&observation.episode_id) {
            return Err(FamilyGuardrailError::DuplicateObservation(
                observation.episode_id,
            ));
        }
        if !observation.baseline_loss.is_finite()
            || !observation.candidate_loss.is_finite()
            || !(0.0..=1.0).contains(&observation.baseline_loss)
            || !(0.0..=1.0).contains(&observation.candidate_loss)
        {
            return Err(FamilyGuardrailError::InvalidLoss);
        }

        let accumulator = self
            .family_state
            .get_mut(&observation.family_id)
            .ok_or_else(|| FamilyGuardrailError::UnknownFamily(observation.family_id.clone()))?;
        if accumulator.count >= family.maximum_observations {
            return Err(FamilyGuardrailError::FamilyObservationBudgetExhausted(
                observation.family_id,
            ));
        }

        let improvement = observation.baseline_loss - observation.candidate_loss;
        accumulator.count += 1;
        accumulator.mean_improvement +=
            (improvement - accumulator.mean_improvement) / accumulator.count as f64;
        self.seen_episode_ids.insert(observation.episode_id);
        self.total_observations += 1;

        let report = self.current_report()?;
        self.terminal = matches!(
            report.status,
            FamilyGuardrailStatus::AllFamiliesNonInferior
                | FamilyGuardrailStatus::RegressionDetected
                | FamilyGuardrailStatus::InconclusiveAtBudget
        );
        Ok(report)
    }

    /// Recompute the report without consuming another observation.
    pub fn report(&self) -> Result<FamilyGuardrailReport, FamilyGuardrailError> {
        self.current_report()
    }

    /// Whether the gate reached a terminal outcome.
    pub fn is_terminal(&self) -> bool {
        self.terminal
    }

    fn current_report(&self) -> Result<FamilyGuardrailReport, FamilyGuardrailError> {
        let candidate = self.candidate_index as f64;
        let candidate_alpha =
            self.campaign_alpha / (candidate * (candidate + 1.0));
        let family_alpha = candidate_alpha / self.plan.families.len() as f64;
        let mut family_results = Vec::with_capacity(self.plan.families.len());
        let mut regression_detected = false;
        let mut all_noninferior = true;
        let mut all_at_budget = true;

        for family in &self.plan.families {
            let accumulator = self
                .family_state
                .get(&family.family_id)
                .ok_or_else(|| FamilyGuardrailError::UnknownFamily(family.family_id.clone()))?;
            let count = accumulator.count;
            let (lower, upper, alpha_spent, status) = if count == 0 {
                all_noninferior = false;
                all_at_budget = false;
                (
                    None,
                    None,
                    None,
                    FamilyNonInferiorityStatus::InsufficientEvidence,
                )
            } else {
                let n = count as f64;
                let alpha_at_look = family_alpha / (n * (n + 1.0));
                let radius = (2.0 * (2.0 / alpha_at_look).ln() / n).sqrt();
                let lower = (accumulator.mean_improvement - radius).max(-1.0);
                let upper = (accumulator.mean_improvement + radius).min(1.0);
                if !radius.is_finite() || !lower.is_finite() || !upper.is_finite() {
                    return Err(FamilyGuardrailError::InvalidLoss);
                }

                let threshold = -family.non_inferiority_margin;
                let status = if count < family.minimum_observations {
                    all_noninferior = false;
                    all_at_budget = false;
                    FamilyNonInferiorityStatus::InsufficientEvidence
                } else if lower > threshold {
                    FamilyNonInferiorityStatus::NonInferior
                } else if upper < threshold {
                    regression_detected = true;
                    all_noninferior = false;
                    FamilyNonInferiorityStatus::RegressionEstablished
                } else if count >= family.maximum_observations {
                    all_noninferior = false;
                    FamilyNonInferiorityStatus::UnresolvedAtBudget
                } else {
                    all_noninferior = false;
                    all_at_budget = false;
                    FamilyNonInferiorityStatus::Collecting
                };
                if count < family.maximum_observations {
                    all_at_budget = false;
                }
                (Some(lower), Some(upper), Some(alpha_at_look), status)
            };

            family_results.push(FamilyNonInferiorityResult {
                family_id: family.family_id.clone(),
                non_inferiority_margin: family.non_inferiority_margin,
                observation_count: count,
                minimum_observations: family.minimum_observations,
                maximum_observations: family.maximum_observations,
                mean_improvement: if count == 0 {
                    None
                } else {
                    Some(accumulator.mean_improvement)
                },
                lower_confidence_bound: lower,
                upper_confidence_bound: upper,
                alpha_spent_at_current_look: alpha_spent,
                status,
            });
        }

        let status = if regression_detected {
            FamilyGuardrailStatus::RegressionDetected
        } else if all_noninferior {
            FamilyGuardrailStatus::AllFamiliesNonInferior
        } else if all_at_budget {
            FamilyGuardrailStatus::InconclusiveAtBudget
        } else {
            FamilyGuardrailStatus::Collecting
        };
        let mut report = FamilyGuardrailReport {
            format_version: 1,
            candidate_index: self.candidate_index,
            campaign_alpha: self.campaign_alpha,
            candidate_alpha_all_families: candidate_alpha,
            family_count: self.plan.families.len(),
            total_observation_count: self.total_observations,
            plan_digest: self.plan.plan_digest.clone(),
            simultaneous_family_bounds: true,
            status,
            families: family_results,
            report_digest: String::new(),
        };
        report.report_digest = family_guardrail_report_digest(&report)?;
        Ok(report)
    }
}

/// Recompute the digest of a family-guardrail report before it is bound elsewhere.
pub fn validate_family_guardrail_report(
    report: &FamilyGuardrailReport,
) -> Result<(), FamilyGuardrailError> {
    if report.format_version != 1
        || report.candidate_index == 0
        || report.family_count == 0
        || report.family_count != report.families.len()
        || !report.campaign_alpha.is_finite()
        || !(0.0..1.0).contains(&report.campaign_alpha)
        || !report.candidate_alpha_all_families.is_finite()
        || !report.simultaneous_family_bounds
        || !is_lowercase_hex_digest(&report.report_digest)
        || !is_lowercase_sha256_or_blake3(&report.plan_digest)
    {
        return Err(FamilyGuardrailError::ReportDigestMismatch);
    }
    let expected_alpha =
        report.campaign_alpha
            / (report.candidate_index as f64 * (report.candidate_index as f64 + 1.0));
    if (expected_alpha - report.candidate_alpha_all_families).abs() > LOSS_TOLERANCE {
        return Err(FamilyGuardrailError::ReportDigestMismatch);
    }
    let expected_digest = family_guardrail_report_digest(report)?;
    if expected_digest != report.report_digest {
        return Err(FamilyGuardrailError::ReportDigestMismatch);
    }
    Ok(())
}

fn family_guardrail_report_digest(
    report: &FamilyGuardrailReport,
) -> Result<String, FamilyGuardrailError> {
    #[derive(Serialize)]
    struct Claims<'a> {
        format_version: u32,
        candidate_index: u64,
        campaign_alpha: f64,
        candidate_alpha_all_families: f64,
        family_count: usize,
        total_observation_count: u64,
        plan_digest: &'a str,
        simultaneous_family_bounds: bool,
        status: FamilyGuardrailStatus,
        families: &'a [FamilyNonInferiorityResult],
    }
    let claims = Claims {
        format_version: report.format_version,
        candidate_index: report.candidate_index,
        campaign_alpha: report.campaign_alpha,
        candidate_alpha_all_families: report.candidate_alpha_all_families,
        family_count: report.family_count,
        total_observation_count: report.total_observation_count,
        plan_digest: &report.plan_digest,
        simultaneous_family_bounds: report.simultaneous_family_bounds,
        status: report.status,
        families: &report.families,
    };
    let encoded =
        serde_json::to_vec(&claims).map_err(|_| FamilyGuardrailError::ReportSerialization)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-family-noninferiority-report-v1\0");
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(hasher.finalize().to_hex().to_string())
}

fn is_lowercase_sha256_or_blake3(value: &str) -> bool {
    value.len() == DIGEST_LENGTH
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(family_id: &str, max: usize, margin: f64) -> FamilyGuardrailSpec {
        FamilyGuardrailSpec {
            family_id: family_id.to_owned(),
            non_inferiority_margin: margin,
            minimum_observations: (max as u64).div_ceil(2),
            maximum_observations: max as u64,
            episode_ids: (0..max)
                .map(|index| format!("{family_id}-episode-{index:04}"))
                .collect(),
        }
    }

    fn plan() -> FamilyGuardrailPlan {
        FamilyGuardrailPlan::new(vec![
            spec("family-b", 200, 0.05),
            spec("family-a", 200, 0.05),
        ])
        .unwrap()
    }

    fn observe_family(
        gate: &mut FamilyNonInferiorityGate,
        family_id: &str,
        prefix: &str,
        count: usize,
        baseline_loss: f64,
        candidate_loss: f64,
    ) -> FamilyGuardrailReport {
        let mut report = gate.report().unwrap();
        for index in 0..count {
            report = gate
                .observe(FamilyGuardrailObservation {
                    episode_id: format!("{prefix}-episode-{index:04}"),
                    family_id: family_id.to_owned(),
                    baseline_loss,
                    candidate_loss,
                })
                .unwrap();
        }
        report
    }

    #[test]
    fn all_families_must_clear_the_noninferiority_margin() {
        let mut gate = FamilyNonInferiorityGate::new(plan(), 0.05, 1).unwrap();
        observe_family(&mut gate, "family-a", "family-a", 180, 0.9, 0.1);
        let report = observe_family(&mut gate, "family-b", "family-b", 180, 0.9, 0.1);
        assert_eq!(report.status, FamilyGuardrailStatus::AllFamiliesNonInferior);
        assert!(report.simultaneous_family_bounds);
        assert_eq!(report.families.len(), 2);
        assert!(
            report
                .families
                .iter()
                .all(|family| family.status == FamilyNonInferiorityStatus::NonInferior)
        );
        assert!(gate.is_terminal());
        validate_family_guardrail_report(&report).unwrap();
    }

    #[test]
    fn aggregate_benefit_cannot_hide_a_critical_family_regression() {
        let mut gate = FamilyNonInferiorityGate::new(plan(), 0.05, 1).unwrap();
        observe_family(&mut gate, "family-a", "family-a", 180, 0.95, 0.05);
        let report = observe_family(&mut gate, "family-b", "family-b", 180, 0.0, 0.9);
        assert_eq!(report.status, FamilyGuardrailStatus::RegressionDetected);
        let regressed = report
            .families
            .iter()
            .find(|family| family.family_id == "family-b")
            .unwrap();
        assert_eq!(regressed.status, FamilyNonInferiorityStatus::RegressionEstablished);
        assert!(
            regressed.upper_confidence_bound.unwrap()
                < -regressed.non_inferiority_margin
        );
    }

    #[test]
    fn candidate_alpha_is_spent_over_ordinals_and_family_looks() {
        let mut sum = 0.0;
        for candidate in 1..=100_000_u64 {
            let alpha_candidate = 0.05 / (candidate as f64 * (candidate as f64 + 1.0));
            sum += alpha_candidate;
        }
        assert!(sum <= 0.05);
        assert!(sum > 0.0499);
        let gate = FamilyNonInferiorityGate::new(plan(), 0.05, 2).unwrap();
        let report = gate.report().unwrap();
        assert!((report.candidate_alpha_all_families - 0.05 / 6.0).abs() < 1e-15);
        assert!(report.families.iter().all(|family| family.alpha_spent_at_current_look.is_none()));
    }

    #[test]
    fn family_roster_and_episode_assignment_are_canonical_and_disjoint() {
        let first = plan();
        let second = FamilyGuardrailPlan::new(vec![
            spec("family-a", 200, 0.05),
            spec("family-b", 200, 0.05),
        ])
        .unwrap();
        assert_eq!(first.plan_digest(), second.plan_digest());
        let overlapping = FamilyGuardrailPlan::new(vec![
            FamilyGuardrailSpec {
                episode_ids: vec!["shared-episode".to_owned()],
                minimum_observations: 1,
                maximum_observations: 1,
                ..spec("family-a", 1, 0.05)
            },
            FamilyGuardrailSpec {
                episode_ids: vec!["shared-episode".to_owned()],
                minimum_observations: 1,
                maximum_observations: 1,
                ..spec("family-b", 1, 0.05)
            },
        ]);
        assert!(matches!(
            overlapping,
            Err(FamilyGuardrailError::EpisodeAssignedToMultipleFamilies { .. })
        ));
    }

    #[test]
    fn duplicate_unknown_or_misassigned_episodes_fail_closed() {
        let mut gate = FamilyNonInferiorityGate::new(plan(), 0.05, 1).unwrap();
        let observation = FamilyGuardrailObservation {
            episode_id: "family-a-episode-0000".to_owned(),
            family_id: "family-a".to_owned(),
            baseline_loss: 0.8,
            candidate_loss: 0.7,
        };
        gate.observe(observation.clone()).unwrap();
        assert!(matches!(
            gate.observe(observation),
            Err(FamilyGuardrailError::DuplicateObservation(_))
        ));
        assert!(matches!(
            gate.observe(FamilyGuardrailObservation {
                episode_id: "family-a-episode-0001".to_owned(),
                family_id: "missing-family".to_owned(),
                baseline_loss: 0.8,
                candidate_loss: 0.7,
            }),
            Err(FamilyGuardrailError::UnknownFamily(_))
        ));
        assert!(matches!(
            gate.observe(FamilyGuardrailObservation {
                episode_id: "family-b-episode-0000".to_owned(),
                family_id: "family-a".to_owned(),
                baseline_loss: 0.8,
                candidate_loss: 0.7,
            }),
            Err(FamilyGuardrailError::EpisodeNotInFamilyManifest { .. })
        ));
    }

    #[test]
    fn maximum_budget_without_decisive_evidence_is_inconclusive() {
        let mut gate = FamilyNonInferiorityGate::new(plan(), 0.05, 1).unwrap();
        observe_family(&mut gate, "family-a", "family-a", 200, 0.5, 0.5);
        let report = observe_family(&mut gate, "family-b", "family-b", 200, 0.5, 0.5);
        assert_eq!(report.status, FamilyGuardrailStatus::InconclusiveAtBudget);
        assert!(report.families.iter().all(|family| {
            family.status == FamilyNonInferiorityStatus::UnresolvedAtBudget
        }));
        assert!(gate.is_terminal());
    }

    #[test]
    fn loss_validation_and_post_terminal_observation_fail_closed() {
        let mut gate = FamilyNonInferiorityGate::new(plan(), 0.05, 1).unwrap();
        assert!(matches!(
            gate.observe(FamilyGuardrailObservation {
                episode_id: "family-a-episode-0000".to_owned(),
                family_id: "family-a".to_owned(),
                baseline_loss: f64::NAN,
                candidate_loss: 0.1,
            }),
            Err(FamilyGuardrailError::InvalidLoss)
        ));
        let mut gate = FamilyNonInferiorityGate::new(plan(), 0.05, 1).unwrap();
        observe_family(&mut gate, "family-a", "family-a", 180, 0.9, 0.1);
        let report = observe_family(&mut gate, "family-b", "family-b", 180, 0.9, 0.1);
        assert_eq!(report.status, FamilyGuardrailStatus::AllFamiliesNonInferior);
        assert!(matches!(
            gate.observe(FamilyGuardrailObservation {
                episode_id: "family-b-episode-0180".to_owned(),
                family_id: "family-b".to_owned(),
                baseline_loss: 0.9,
                candidate_loss: 0.1,
            }),
            Err(FamilyGuardrailError::AlreadyTerminal)
        ));
    }

    #[test]
    fn tampered_family_report_and_plan_are_rejected() {
        let mut gate = FamilyNonInferiorityGate::new(plan(), 0.05, 1).unwrap();
        let report = gate.report().unwrap();
        let mut tampered = report.clone();
        tampered.families[0].family_id = "tampered".to_owned();
        assert!(matches!(
            validate_family_guardrail_report(&tampered),
            Err(FamilyGuardrailError::ReportDigestMismatch)
        ));

        let mut tampered_plan = plan();
        tampered_plan.families[0].non_inferiority_margin = 0.9;
        assert!(matches!(
            tampered_plan.validate(),
            Err(FamilyGuardrailError::PlanDigestMismatch)
        ));
    }

    #[test]
    fn margins_and_observation_limits_fail_closed() {
        let invalid = FamilyGuardrailPlan::new(vec![FamilyGuardrailSpec {
            non_inferiority_margin: f64::NAN,
            ..spec("family", 2, 0.05)
        }]);
        assert!(matches!(
            invalid,
            Err(FamilyGuardrailError::InvalidNonInferiorityMargin { .. })
        ));
        let invalid = FamilyGuardrailPlan::new(vec![FamilyGuardrailSpec {
            minimum_observations: 3,
            ..spec("family", 2, 0.05)
        }]);
        assert!(matches!(
            invalid,
            Err(FamilyGuardrailError::InvalidObservationLimits { .. })
        ));
    }
}
