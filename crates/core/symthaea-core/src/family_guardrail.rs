// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Anytime-valid task-family non-inferiority guardrails for candidate verification.
//!
//! A candidate must preserve every preregistered critical family, not merely
//! improve an aggregate score. Each family has a bounded paired-loss stream and
//! a frozen non-inferiority margin. The candidate is eligible only when every
//! family has crossed its lower confidence bound above the negative margin.
//!
//! The family tests use an intersection-union decision: the null is that at
//! least one registered family is materially worse. Acceptance requires
//! rejecting every family-level null. A true null family therefore controls
//! the probability of a false all-family acceptance; no extra Bonferroni factor
//! over families is needed for this single all-must-pass decision. The
//! confidence bounds shown for individual families are NOT simultaneous
//! family-wise intervals and must not be presented as such.
//!
//! Candidate ordinals and interim looks each receive a reciprocal-product
//! alpha allocation. The caller must reserve a distinct portion of the overall
//! campaign alpha for this guardrail in addition to the aggregate-benefit
//! verifier (for example, alpha/2 for aggregate benefit and alpha/2 here).
//! The two campaign ledgers must use the same frozen candidate ordinal and
//! candidate/protocol identities. This module does not itself allocate alpha
//! jointly with SequentialWeightVerificationCampaign.
//!
//! Episode IDs must be stable source-record identities supplied by the
//! independent evaluator. Hashes and IDs do not authenticate their mapping to
//! physical source records; that remains an evaluator/provenance responsibility.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

const DIGEST_HEX_LENGTH: usize = 64;

/// Preregistered evaluation family and its candidate-specific confirmatory episodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyGuardrailSpec {
    pub family_id: String,
    /// Maximum tolerable loss increase for this family, in normalized loss units.
    pub noninferiority_margin: f64,
    pub minimum_observations: u64,
    pub maximum_observations: u64,
    /// Stable source-record episode IDs reserved for this family.
    pub episode_ids: Vec<String>,
}

/// Frozen identity and family membership for one candidate's guardrail evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyGuardrailManifest {
    pub format_version: u32,
    /// One-based ordinal assigned by the shared candidate campaign ledger.
    pub candidate_index: u64,
    pub baseline_model_digest: String,
    pub candidate_model_digest: String,
    pub evaluation_protocol_digest: String,
    /// Digest of the complete candidate-specific confirmatory episode manifest.
    pub confirmatory_manifest_digest: String,
    /// Alpha budget reserved for family guardrails, separate from aggregate benefit.
    pub campaign_alpha: f64,
    /// Sorted by family_id; each family's episode_ids are sorted.
    pub families: Vec<FamilyGuardrailSpec>,
    pub manifest_digest: String,
}

/// One paired baseline/candidate loss from one registered task family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyPairedLossObservation {
    pub family_id: String,
    pub episode_id: String,
    pub baseline_loss: f64,
    pub candidate_loss: f64,
}

/// Qualification state for one family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FamilyNonInferiorityStatus {
    InsufficientEvidence,
    NonInferior,
    MaterialRegression,
    Undetermined,
    UndeterminedAtBudget,
}

/// Overall family-guardrail decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FamilyGuardrailStatus {
    Collecting,
    AllFamiliesNonInferior,
    MaterialRegressionDetected,
    InconclusiveAtBudget,
}

/// Per-family anytime-valid report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyGuardrailResult {
    pub family_id: String,
    pub observation_count: u64,
    pub mean_improvement: Option<f64>,
    /// Lower/upper confidence-sequence bounds for baseline loss minus candidate loss.
    /// These are per-family bounds, not simultaneous intervals over all families.
    pub lower_improvement_bound: Option<f64>,
    pub upper_improvement_bound: Option<f64>,
    pub noninferiority_margin: f64,
    pub status: FamilyNonInferiorityStatus,
}

/// Complete all-families non-inferiority report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyGuardrailReport {
    pub format_version: u32,
    pub candidate_index: u64,
    pub baseline_model_digest: String,
    pub candidate_model_digest: String,
    pub evaluation_protocol_digest: String,
    pub manifest_digest: String,
    pub campaign_alpha: f64,
    pub status: FamilyGuardrailStatus,
    pub family_results: Vec<FamilyGuardrailResult>,
    pub report_digest: String,
}

/// Input, provenance, or numerical failure. All malformed inputs fail closed.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum FamilyGuardrailError {
    #[error("unsupported family guardrail manifest version {0}")]
    UnsupportedManifestVersion(u32),
    #[error("candidate index must be one-based")]
    InvalidCandidateIndex,
    #[error("model, protocol, and confirmatory manifest digests must be 64 lowercase hexadecimal characters")]
    InvalidDigest,
    #[error("guardrail campaign alpha must be finite and in (0, 1)")]
    InvalidCampaignAlpha,
    #[error("at least one evaluation family is required")]
    EmptyFamilySet,
    #[error("family identifier is empty or duplicated: {0}")]
    InvalidFamilyId(String),
    #[error("family {family_id} has an invalid non-inferiority margin")]
    InvalidNonInferiorityMargin { family_id: String },
    #[error("family {family_id} has an invalid observation budget")]
    InvalidObservationBudget { family_id: String },
    #[error("family {family_id} has {actual} reserved episode IDs but needs {required}")]
    FamilyManifestTooSmall { family_id: String, required: usize, actual: usize },
    #[error("episode ID is empty, duplicated, or belongs to multiple families: {0}")]
    InvalidEpisodeId(String),
    #[error("family guardrail manifest digest does not match its claims")]
    ManifestDigestMismatch,
    #[error("candidate guardrail evaluation is already terminal")]
    AlreadyTerminal,
    #[error("family is not registered: {0}")]
    UnknownFamily(String),
    #[error("episode ID is not reserved for family {family_id}: {episode_id}")]
    EpisodeNotInFamilyManifest { family_id: String, episode_id: String },
    #[error("episode ID has already been consumed: {0}")]
    DuplicateEpisodeId(String),
    #[error("baseline or candidate loss is non-finite or outside [0, 1]")]
    InvalidLoss,
    #[error("family {family_id} has exhausted its observation budget")]
    FamilyObservationBudgetExhausted { family_id: String },
    #[error("confidence-sequence calculation became non-finite")]
    NonFiniteConfidenceSequence,
    #[error("failed to compute deterministic report digest")]
    ReportDigestFailure,
}

#[derive(Debug, Clone, Default)]
struct FamilyAccumulator {
    observation_count: u64,
    mean_improvement: f64,
}

/// Stateful candidate-specific family guardrail verifier.
#[derive(Debug, Clone)]
pub struct FamilyGuardrailVerifier {
    manifest: FamilyGuardrailManifest,
    episode_to_family: BTreeMap<String, String>,
    accumulators: BTreeMap<String, FamilyAccumulator>,
    seen_episode_ids: BTreeSet<String>,
    terminal: bool,
}

impl FamilyGuardrailManifest {
    /// Construct a canonical manifest, sorting family and episode identities.
    pub fn new(
        candidate_index: u64,
        baseline_model_digest: impl Into<String>,
        candidate_model_digest: impl Into<String>,
        evaluation_protocol_digest: impl Into<String>,
        confirmatory_manifest_digest: impl Into<String>,
        campaign_alpha: f64,
        mut families: Vec<FamilyGuardrailSpec>,
    ) -> Result<Self, FamilyGuardrailError> {
        let mut manifest = Self {
            format_version: 1,
            candidate_index,
            baseline_model_digest: baseline_model_digest.into(),
            candidate_model_digest: candidate_model_digest.into(),
            evaluation_protocol_digest: evaluation_protocol_digest.into(),
            confirmatory_manifest_digest: confirmatory_manifest_digest.into(),
            campaign_alpha,
            families: {
                for family in &mut families {
                    family.episode_ids.sort();
                }
                families.sort_by(|left, right| left.family_id.cmp(&right.family_id));
                families
            },
            manifest_digest: String::new(),
        };
        validate_manifest_claims(&manifest)?;
        manifest.manifest_digest = compute_manifest_digest(&manifest)?;
        Ok(manifest)
    }

    /// Revalidate canonical shape and the complete manifest digest.
    pub fn validate(&self) -> Result<(), FamilyGuardrailError> {
        validate_manifest_claims(self)?;
        if compute_manifest_digest(self)? != self.manifest_digest {
            return Err(FamilyGuardrailError::ManifestDigestMismatch);
        }
        Ok(())
    }
}

impl FamilyGuardrailVerifier {
    /// Start a candidate-specific guardrail evaluation over a frozen manifest.
    pub fn new(manifest: FamilyGuardrailManifest) -> Result<Self, FamilyGuardrailError> {
        manifest.validate()?;
        let mut episode_to_family = BTreeMap::new();
        let mut accumulators = BTreeMap::new();
        for family in &manifest.families {
            accumulators.insert(family.family_id.clone(), FamilyAccumulator::default());
            for episode_id in &family.episode_ids {
                episode_to_family.insert(episode_id.clone(), family.family_id.clone());
            }
        }
        Ok(Self {
            manifest,
            episode_to_family,
            accumulators,
            seen_episode_ids: BTreeSet::new(),
            terminal: false,
        })
    }

    /// Consume one paired family episode and recompute the all-family decision.
    pub fn observe(
        &mut self,
        observation: FamilyPairedLossObservation,
    ) -> Result<FamilyGuardrailReport, FamilyGuardrailError> {
        if self.terminal {
            return Err(FamilyGuardrailError::AlreadyTerminal);
        }
        let family_id = observation.family_id.as_str();
        let registered_family = self
            .episode_to_family
            .get(&observation.episode_id)
            .ok_or_else(|| FamilyGuardrailError::EpisodeNotInFamilyManifest {
                family_id: observation.family_id.clone(),
                episode_id: observation.episode_id.clone(),
            })?;
        if registered_family != family_id {
            return Err(FamilyGuardrailError::EpisodeNotInFamilyManifest {
                family_id: observation.family_id,
                episode_id: observation.episode_id,
            });
        }
        if self.seen_episode_ids.contains(&observation.episode_id) {
            return Err(FamilyGuardrailError::DuplicateEpisodeId(observation.episode_id));
        }
        if !observation.baseline_loss.is_finite()
            || !(0.0..=1.0).contains(&observation.baseline_loss)
            || !observation.candidate_loss.is_finite()
            || !(0.0..=1.0).contains(&observation.candidate_loss)
        {
            return Err(FamilyGuardrailError::InvalidLoss);
        }

        let spec = self
            .manifest
            .families
            .iter()
            .find(|family| family.family_id == family_id)
            .expect("family membership is derived from validated manifest");
        let accumulator = self
            .accumulators
            .get_mut(family_id)
            .expect("accumulator is created for every validated family");
        if accumulator.observation_count >= spec.maximum_observations {
            return Err(FamilyGuardrailError::FamilyObservationBudgetExhausted {
                family_id: family_id.to_owned(),
            });
        }

        let improvement = observation.baseline_loss - observation.candidate_loss;
        accumulator.observation_count += 1;
        accumulator.mean_improvement +=
            (improvement - accumulator.mean_improvement) / accumulator.observation_count as f64;
        self.seen_episode_ids.insert(observation.episode_id);
        let report = self.report()?;
        self.terminal = matches!(
            report.status,
            FamilyGuardrailStatus::AllFamiliesNonInferior
                | FamilyGuardrailStatus::MaterialRegressionDetected
                | FamilyGuardrailStatus::InconclusiveAtBudget
        );
        Ok(report)
    }

    /// Recompute the current report without consuming another episode.
    pub fn report(&self) -> Result<FamilyGuardrailReport, FamilyGuardrailError> {
        let candidate = self.manifest.candidate_index as f64;
        let mut family_results = Vec::with_capacity(self.manifest.families.len());

        for family in &self.manifest.families {
            let accumulator = self
                .accumulators
                .get(&family.family_id)
                .expect("accumulator is created for every validated family");
            let count = accumulator.observation_count;
            let (lower, upper) = if count == 0 {
                (None, None)
            } else {
                let n = count as f64;
                // Reciprocal-product allocation: sum_j 1/[j(j+1)] = 1,
                // sum_n 1/[n(n+1)] = 1. The same candidate/look budget is
                // used for each family because the acceptance rule is an
                // intersection-union test (all families must pass).
                let alpha = self.manifest.campaign_alpha
                    / (candidate * (candidate + 1.0) * n * (n + 1.0));
                let radius = (2.0 * (2.0 / alpha).ln() / n).sqrt();
                let lower = (accumulator.mean_improvement - radius).max(-1.0);
                let upper = (accumulator.mean_improvement + radius).min(1.0);
                if !lower.is_finite() || !upper.is_finite() {
                    return Err(FamilyGuardrailError::NonFiniteConfidenceSequence);
                }
                (Some(lower), Some(upper))
            };

            let status = if count < family.minimum_observations {
                FamilyNonInferiorityStatus::InsufficientEvidence
            } else if lower.is_some_and(|bound| bound > -family.noninferiority_margin) {
                FamilyNonInferiorityStatus::NonInferior
            } else if upper.is_some_and(|bound| bound < -family.noninferiority_margin) {
                FamilyNonInferiorityStatus::MaterialRegression
            } else if count >= family.maximum_observations {
                FamilyNonInferiorityStatus::UndeterminedAtBudget
            } else {
                FamilyNonInferiorityStatus::Undetermined
            };

            family_results.push(FamilyGuardrailResult {
                family_id: family.family_id.clone(),
                observation_count: count,
                mean_improvement: (count > 0).then_some(accumulator.mean_improvement),
                lower_improvement_bound: lower,
                upper_improvement_bound: upper,
                noninferiority_margin: family.noninferiority_margin,
                status,
            });
        }

        let status = if family_results
            .iter()
            .any(|result| result.status == FamilyNonInferiorityStatus::MaterialRegression)
        {
            FamilyGuardrailStatus::MaterialRegressionDetected
        } else if family_results
            .iter()
            .all(|result| result.status == FamilyNonInferiorityStatus::NonInferior)
        {
            FamilyGuardrailStatus::AllFamiliesNonInferior
        } else if family_results.iter().any(|result| {
            result.status == FamilyNonInferiorityStatus::UndeterminedAtBudget
        }) {
            FamilyGuardrailStatus::InconclusiveAtBudget
        } else {
            FamilyGuardrailStatus::Collecting
        };

        let mut report = FamilyGuardrailReport {
            format_version: 1,
            candidate_index: self.manifest.candidate_index,
            baseline_model_digest: self.manifest.baseline_model_digest.clone(),
            candidate_model_digest: self.manifest.candidate_model_digest.clone(),
            evaluation_protocol_digest: self.manifest.evaluation_protocol_digest.clone(),
            manifest_digest: self.manifest.manifest_digest.clone(),
            campaign_alpha: self.manifest.campaign_alpha,
            status,
            family_results,
            report_digest: String::new(),
        };
        report.report_digest = compute_report_digest(&report)?;
        Ok(report)
    }

    /// The canonical frozen manifest used by this verifier.
    pub fn manifest(&self) -> &FamilyGuardrailManifest {
        &self.manifest
    }
}

fn validate_manifest_claims(
    manifest: &FamilyGuardrailManifest,
) -> Result<(), FamilyGuardrailError> {
    if manifest.format_version != 1 {
        return Err(FamilyGuardrailError::UnsupportedManifestVersion(
            manifest.format_version,
        ));
    }
    if manifest.candidate_index == 0 {
        return Err(FamilyGuardrailError::InvalidCandidateIndex);
    }
    for digest in [
        manifest.baseline_model_digest.as_str(),
        manifest.candidate_model_digest.as_str(),
        manifest.evaluation_protocol_digest.as_str(),
        manifest.confirmatory_manifest_digest.as_str(),
    ] {
        if !is_lowercase_sha256(digest) {
            return Err(FamilyGuardrailError::InvalidDigest);
        }
    }
    if !manifest.campaign_alpha.is_finite()
        || !(0.0..1.0).contains(&manifest.campaign_alpha)
    {
        return Err(FamilyGuardrailError::InvalidCampaignAlpha);
    }
    if manifest.families.is_empty() {
        return Err(FamilyGuardrailError::EmptyFamilySet);
    }
    if manifest
        .families
        .windows(2)
        .any(|pair| pair[0].family_id >= pair[1].family_id)
    {
        return Err(FamilyGuardrailError::InvalidFamilyId(
            "families must be sorted and unique".to_owned(),
        ));
    }

    let mut all_episode_ids = BTreeSet::new();
    for family in &manifest.families {
        if family.family_id.trim().is_empty() {
            return Err(FamilyGuardrailError::InvalidFamilyId(
                family.family_id.clone(),
            ));
        }
        if !family.noninferiority_margin.is_finite()
            || !(0.0..=1.0).contains(&family.noninferiority_margin)
        {
            return Err(FamilyGuardrailError::InvalidNonInferiorityMargin {
                family_id: family.family_id.clone(),
            });
        }
        if family.minimum_observations == 0
            || family.minimum_observations > family.maximum_observations
        {
            return Err(FamilyGuardrailError::InvalidObservationBudget {
                family_id: family.family_id.clone(),
            });
        }
        if family.episode_ids.len() < family.maximum_observations as usize {
            return Err(FamilyGuardrailError::FamilyManifestTooSmall {
                family_id: family.family_id.clone(),
                required: family.maximum_observations as usize,
                actual: family.episode_ids.len(),
            });
        }
        if family.episode_ids.iter().any(|id| id.trim().is_empty())
            || family.episode_ids.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(FamilyGuardrailError::InvalidEpisodeId(
                family.family_id.clone(),
            ));
        }
        for episode_id in &family.episode_ids {
            if !all_episode_ids.insert(episode_id.clone()) {
                return Err(FamilyGuardrailError::InvalidEpisodeId(
                    episode_id.clone(),
                ));
            }
        }
    }
    Ok(())
}

fn compute_manifest_digest(
    manifest: &FamilyGuardrailManifest,
) -> Result<String, FamilyGuardrailError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-family-guardrail-manifest-v1\0");
    hasher.update(&manifest.candidate_index.to_le_bytes());
    hasher.update(&manifest.campaign_alpha.to_bits().to_le_bytes());
    for value in [
        manifest.baseline_model_digest.as_str(),
        manifest.candidate_model_digest.as_str(),
        manifest.evaluation_protocol_digest.as_str(),
        manifest.confirmatory_manifest_digest.as_str(),
    ] {
        hash_string(&mut hasher, value);
    }
    hasher.update(&(manifest.families.len() as u64).to_le_bytes());
    for family in &manifest.families {
        hash_string(&mut hasher, &family.family_id);
        hasher.update(&family.noninferiority_margin.to_bits().to_le_bytes());
        hasher.update(&family.minimum_observations.to_le_bytes());
        hasher.update(&family.maximum_observations.to_le_bytes());
        hasher.update(&(family.episode_ids.len() as u64).to_le_bytes());
        for episode_id in &family.episode_ids {
            hash_string(&mut hasher, episode_id);
        }
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn compute_report_digest(
    report: &FamilyGuardrailReport,
) -> Result<String, FamilyGuardrailError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-family-guardrail-report-v1\0");
    hasher.update(&report.format_version.to_le_bytes());
    hasher.update(&report.candidate_index.to_le_bytes());
    hasher.update(&report.campaign_alpha.to_bits().to_le_bytes());
    hash_string(&mut hasher, &report.baseline_model_digest);
    hash_string(&mut hasher, &report.candidate_model_digest);
    hash_string(&mut hasher, &report.evaluation_protocol_digest);
    hash_string(&mut hasher, &report.manifest_digest);
    hasher.update(&[report.status as u8]);
    hasher.update(&(report.family_results.len() as u64).to_le_bytes());
    for result in &report.family_results {
        hash_string(&mut hasher, &result.family_id);
        hasher.update(&result.observation_count.to_le_bytes());
        hasher.update(&result.mean_improvement.unwrap_or(f64::NAN).to_bits().to_le_bytes());
        hasher.update(&result.lower_improvement_bound.unwrap_or(f64::NAN).to_bits().to_le_bytes());
        hasher.update(&result.upper_improvement_bound.unwrap_or(f64::NAN).to_bits().to_le_bytes());
        hasher.update(&result.noninferiority_margin.to_bits().to_le_bytes());
        hasher.update(&[result.status as u8]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == DIGEST_HEX_LENGTH
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn manifest(candidate_index: u64, margin: f64) -> FamilyGuardrailManifest {
        let family = |family_id: &str, prefix: &str| FamilyGuardrailSpec {
            family_id: family_id.to_owned(),
            noninferiority_margin: margin,
            minimum_observations: 256,
            maximum_observations: 512,
            episode_ids: (0..512)
                .map(|index| format!("{prefix}-{index:04}"))
                .collect(),
        };
        FamilyGuardrailManifest::new(
            candidate_index,
            digest('a'),
            digest('b'),
            digest('c'),
            digest('d'),
            0.025,
            vec![family("bearing", "bearing"), family("gearbox", "gearbox")],
        )
        .unwrap()
    }

    fn feed_family(
        verifier: &mut FamilyGuardrailVerifier,
        family_id: &str,
        prefix: &str,
        candidate_loss: f64,
    ) -> FamilyGuardrailReport {
        let mut report = verifier.report().unwrap();
        for index in 0..512 {
            report = verifier
                .observe(FamilyPairedLossObservation {
                    family_id: family_id.to_owned(),
                    episode_id: format!("{prefix}-{index:04}"),
                    baseline_loss: 0.5,
                    candidate_loss,
                })
                .unwrap();
        }
        report
    }

    #[test]
    fn all_registered_families_must_cross_their_noninferiority_boundary() {
        let mut verifier = FamilyGuardrailVerifier::new(manifest(1, 0.4)).unwrap();
        for family_id in ["bearing", "gearbox"] {
            let prefix = if family_id == "bearing" { "bearing" } else { "gearbox" };
            let report = feed_family(&mut verifier, family_id, prefix, 0.5);
            if family_id == "bearing" {
                assert_eq!(report.status, FamilyGuardrailStatus::Collecting);
            } else {
                assert_eq!(report.status, FamilyGuardrailStatus::AllFamiliesNonInferior);
            }
        }
        let report = verifier.report().unwrap();
        assert_eq!(report.status, FamilyGuardrailStatus::AllFamiliesNonInferior);
        assert_ne!(report.report_digest, digest('0'));
        assert!(report.family_results.iter().all(|result| {
            result.status == FamilyNonInferiorityStatus::NonInferior
                && result.lower_improvement_bound.unwrap() > -result.noninferiority_margin
        }));
    }

    #[test]
    fn strong_aggregate_gain_cannot_hide_a_critical_family_regression() {
        let mut verifier = FamilyGuardrailVerifier::new(manifest(1, 0.2)).unwrap();
        let good = feed_family(&mut verifier, "bearing", "bearing", 0.0);
        assert_eq!(good.status, FamilyGuardrailStatus::Collecting);
        let bad = feed_family(&mut verifier, "gearbox", "gearbox", 0.9);
        assert_eq!(bad.status, FamilyGuardrailStatus::MaterialRegressionDetected);
        assert!(
            bad.family_results
                .iter()
                .any(|result| result.status == FamilyNonInferiorityStatus::MaterialRegression)
        );
    }

    #[test]
    fn later_candidate_ordinals_receive_more_conservative_bounds() {
        let mut first = FamilyGuardrailVerifier::new(manifest(1, 0.4)).unwrap();
        let mut later = FamilyGuardrailVerifier::new(manifest(8, 0.4)).unwrap();
        let first_report = feed_family(&mut first, "bearing", "bearing", 0.5);
        let later_report = feed_family(&mut later, "bearing", "bearing", 0.5);
        assert!(
            later_report.family_results[0].lower_improvement_bound.unwrap()
                < first_report.family_results[0].lower_improvement_bound.unwrap()
        );
    }

    #[test]
    fn intersection_union_rule_does_not_claim_simultaneous_family_intervals() {
        let one_family = FamilyGuardrailManifest::new(
            1,
            digest('a'),
            digest('b'),
            digest('c'),
            digest('d'),
            0.025,
            vec![FamilyGuardrailSpec {
                family_id: "only".to_owned(),
                noninferiority_margin: 0.4,
                minimum_observations: 256,
                maximum_observations: 512,
                episode_ids: (0..512).map(|i| format!("only-{i:04}")).collect(),
            }],
        )
        .unwrap();
        let two_families = manifest(1, 0.4);
        let mut one = FamilyGuardrailVerifier::new(one_family).unwrap();
        let mut two = FamilyGuardrailVerifier::new(two_families).unwrap();
        let one_report = feed_family(&mut one, "only", "only", 0.5);
        let two_report = feed_family(&mut two, "bearing", "bearing", 0.5);
        assert_eq!(
            one_report.family_results[0].lower_improvement_bound,
            two_report.family_results[0].lower_improvement_bound
        );
    }

    #[test]
    fn duplicate_or_cross_family_episode_ids_are_rejected_in_the_manifest() {
        let duplicate = FamilyGuardrailManifest::new(
            1,
            digest('a'),
            digest('b'),
            digest('c'),
            digest('d'),
            0.025,
            vec![
                FamilyGuardrailSpec {
                    family_id: "a".to_owned(),
                    noninferiority_margin: 0.2,
                    minimum_observations: 1,
                    maximum_observations: 1,
                    episode_ids: vec!["same".to_owned()],
                },
                FamilyGuardrailSpec {
                    family_id: "b".to_owned(),
                    noninferiority_margin: 0.2,
                    minimum_observations: 1,
                    maximum_observations: 1,
                    episode_ids: vec!["same".to_owned()],
                },
            ],
        );
        assert!(matches!(duplicate, Err(FamilyGuardrailError::InvalidEpisodeId(_))));
    }

    #[test]
    fn unregistered_and_misassigned_episode_ids_fail_closed() {
        let mut verifier = FamilyGuardrailVerifier::new(manifest(1, 0.4)).unwrap();
        let unknown = verifier.observe(FamilyPairedLossObservation {
            family_id: "bearing".to_owned(),
            episode_id: "not-reserved".to_owned(),
            baseline_loss: 0.5,
            candidate_loss: 0.5,
        });
        assert!(matches!(
            unknown,
            Err(FamilyGuardrailError::EpisodeNotInFamilyManifest { .. })
        ));

        let misassigned = verifier.observe(FamilyPairedLossObservation {
            family_id: "gearbox".to_owned(),
            episode_id: "bearing-0000".to_owned(),
            baseline_loss: 0.5,
            candidate_loss: 0.5,
        });
        assert!(matches!(
            misassigned,
            Err(FamilyGuardrailError::EpisodeNotInFamilyManifest { .. })
        ));
    }

    #[test]
    fn duplicate_episode_ids_and_malformed_losses_fail_closed() {
        let mut verifier = FamilyGuardrailVerifier::new(manifest(1, 0.4)).unwrap();
        let observation = FamilyPairedLossObservation {
            family_id: "bearing".to_owned(),
            episode_id: "bearing-0000".to_owned(),
            baseline_loss: 0.5,
            candidate_loss: 0.5,
        };
        verifier.observe(observation.clone()).unwrap();
        assert!(matches!(
            verifier.observe(observation.clone()),
            Err(FamilyGuardrailError::DuplicateEpisodeId(_))
        ));
        let mut malformed = observation;
        malformed.episode_id = "bearing-0001".to_owned();
        malformed.candidate_loss = f64::NAN;
        assert!(matches!(
            verifier.observe(malformed),
            Err(FamilyGuardrailError::InvalidLoss)
        ));
    }

    #[test]
    fn maximum_observation_budget_exhaustion_is_inconclusive_not_a_pass() {
        let mut specs = manifest(1, 0.01).families;
        for spec in &mut specs {
            spec.minimum_observations = 1;
            spec.maximum_observations = 2;
            spec.episode_ids.truncate(2);
        }
        let small_manifest = FamilyGuardrailManifest::new(
            1,
            digest('a'),
            digest('b'),
            digest('c'),
            digest('d'),
            0.025,
            specs,
        )
        .unwrap();
        let mut verifier = FamilyGuardrailVerifier::new(small_manifest).unwrap();
        for index in 0..2 {
            let report = verifier.observe(FamilyPairedLossObservation {
                family_id: "bearing".to_owned(),
                episode_id: format!("bearing-{index:04}"),
                baseline_loss: 0.5,
                candidate_loss: 0.5,
            }).unwrap();
            if index == 1 {
                assert_eq!(report.status, FamilyGuardrailStatus::InconclusiveAtBudget);
            }
        }
        assert!(matches!(
            verifier.observe(FamilyPairedLossObservation {
                family_id: "gearbox".to_owned(),
                episode_id: "gearbox-0000".to_owned(),
                baseline_loss: 0.5,
                candidate_loss: 0.5,
            }),
            Err(FamilyGuardrailError::AlreadyTerminal)
        ));
    }

    #[test]
    fn manifest_tampering_and_bad_inputs_are_rejected() {
        let mut tampered = manifest(1, 0.4);
        tampered.families[0].noninferiority_margin = 0.9;
        assert!(matches!(
            FamilyGuardrailVerifier::new(tampered),
            Err(FamilyGuardrailError::ManifestDigestMismatch)
        ));

        assert!(matches!(
            FamilyGuardrailManifest::new(0, digest('a'), digest('b'), digest('c'), digest('d'), 0.025, vec![]),
            Err(FamilyGuardrailError::InvalidCandidateIndex)
        ));
        assert!(matches!(
            FamilyGuardrailManifest::new(1, digest('a'), digest('b'), digest('c'), digest('d'), 1.0, vec![]),
            Err(FamilyGuardrailError::InvalidCampaignAlpha)
        ));
    }
}
