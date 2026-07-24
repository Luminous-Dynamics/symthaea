// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Direct analysis of complete preference rankings.
//!
//! V8 converted ranks 1-4 into evenly spaced scores. V8.2 instead derives
//! pairwise Symthaea-versus-comparator wins directly from each complete ranking.
//! Confidence intervals use a two-way participant/family cluster bootstrap;
//! one-sided randomization is performed over family-level paired effects.

use crate::blinded_study::{BlindedSchedule, BlindingCodebook};
use crate::cognitive_experiment::CognitivePolicyArm;
use crate::confirmatory_analysis::ConfirmatoryComparator;
use crate::evidence_digest::canonical_json_sha256;
use crate::experiment_manifest::{ConfirmatoryEndpoint, FrozenStudyManifest};
use crate::methodology_plan::FrozenMethodologyPlan;
use crate::participant_evidence::{
    ParticipantEvidenceEnvelope, ParticipantEvidenceIssue, validate_participant_evidence,
};
use crate::participant_schedule::{ParticipantCohortSpec, ParticipantScheduleBook};
use crate::study_evidence::EvidenceBlockStatus;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const RANKED_PREFERENCE_ANALYSIS_VERSION: &str = "symthaea-muse-ranked-preference-analysis-v1";
pub const MIN_RANKED_PREFERENCE_FAMILIES: usize = 8;
pub const MIN_RANKED_PREFERENCE_PARTICIPANTS: usize = 12;
pub const MIN_RANKED_PREFERENCE_BOOTSTRAP_REPLICATES: usize = 2_000;
pub const MIN_RANKED_PREFERENCE_RANDOMIZATION_REPLICATES: usize = 10_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedPreferenceAnalysisPlan {
    pub analysis_version: String,
    pub manifest_sha256: String,
    pub methodology_sha256: String,
    pub participant_schedule_sha256: String,
    pub participant_evidence_sha256: String,
    pub alpha: f64,
    pub minimum_families: usize,
    pub minimum_participants: usize,
    pub bootstrap_replicates: usize,
    pub randomization_replicates: usize,
    pub rng_seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedPairObservation {
    pub participant_token: String,
    pub family_id: String,
    pub fixture_id: String,
    pub comparator: ConfirmatoryComparator,
    pub symthaea_rank: u8,
    pub comparator_rank: u8,
    pub symthaea_position: u8,
    pub comparator_position: u8,
    pub symthaea_won: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedPreferenceEffect {
    pub comparator: ConfirmatoryComparator,
    pub observation_count: usize,
    pub participant_count: usize,
    pub family_count: usize,
    pub symthaea_win_probability: f64,
    /// Win probability minus 0.5. Positive values favor Symthaea.
    pub effect_over_even_odds: f64,
    pub confidence_interval: [f64; 2],
    pub required_margin: f64,
    pub raw_one_sided_p: f64,
    pub holm_adjusted_p: f64,
    pub margin_gate_passed: bool,
    pub inferential_gate_passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedPreferenceConclusion {
    pub success: bool,
    pub comparisons: Vec<RankedPreferenceEffect>,
    pub observations: Vec<RankedPairObservation>,
    pub issues: Vec<RankedPreferenceIssue>,
    pub rationale: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RankedPreferenceIssue {
    WrongAnalysisVersion {
        found: String,
    },
    InvalidAlpha,
    AlphaMismatch,
    MinimumFamiliesTooSmall {
        found: usize,
        required: usize,
    },
    MinimumParticipantsTooSmall {
        found: usize,
        required: usize,
    },
    TooFewBootstrapReplicates {
        found: usize,
        required: usize,
    },
    TooFewRandomizationReplicates {
        found: usize,
        required: usize,
    },
    InvalidMethodology,
    PrimaryEndpointIsNotPreference,
    SerializationFailed {
        field: String,
    },
    DigestMismatch {
        field: String,
    },
    ParticipantEvidence {
        issue: ParticipantEvidenceIssue,
    },
    MissingArmRank {
        block_id: String,
        arm: CognitivePolicyArm,
    },
    DuplicateArmRank {
        block_id: String,
        arm: CognitivePolicyArm,
    },
    TooFewFamilies {
        found: usize,
        required: usize,
    },
    TooFewParticipants {
        found: usize,
        required: usize,
    },
}

#[allow(clippy::too_many_arguments)]
pub fn analyze_ranked_preference(
    manifest: &FrozenStudyManifest,
    methodology: &FrozenMethodologyPlan,
    base_schedule: &BlindedSchedule,
    codebook: &BlindingCodebook,
    cohort: &ParticipantCohortSpec,
    participant_schedule: &ParticipantScheduleBook,
    envelope: &ParticipantEvidenceEnvelope,
    plan: &RankedPreferenceAnalysisPlan,
) -> RankedPreferenceConclusion {
    let mut issues = validate_inputs(
        manifest,
        methodology,
        base_schedule,
        codebook,
        cohort,
        participant_schedule,
        envelope,
        plan,
    );
    let observations =
        match compile_ranked_observations(manifest, codebook, participant_schedule, envelope) {
            Ok(value) => value,
            Err(mut found) => {
                issues.append(&mut found);
                Vec::new()
            }
        };

    let observed_family_count = observations
        .iter()
        .map(|observation| observation.family_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let observed_participant_count = observations
        .iter()
        .map(|observation| observation.participant_token.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    if observed_family_count < plan.minimum_families {
        issues.push(RankedPreferenceIssue::TooFewFamilies {
            found: observed_family_count,
            required: plan.minimum_families,
        });
    }
    if observed_participant_count < plan.minimum_participants {
        issues.push(RankedPreferenceIssue::TooFewParticipants {
            found: observed_participant_count,
            required: plan.minimum_participants,
        });
    }

    let primary = methodology.primary_endpoint();
    let superiority_margin = primary
        .and_then(|endpoint| endpoint.superiority_margin)
        .unwrap_or(f64::INFINITY);
    let noninferiority_margin = primary
        .and_then(|endpoint| endpoint.heuristic_noninferiority_margin)
        .unwrap_or(f64::INFINITY);
    let mut comparisons = Vec::new();
    for comparator in ConfirmatoryComparator::ALL {
        let values: Vec<_> = observations
            .iter()
            .filter(|observation| observation.comparator == comparator)
            .collect();
        if values.is_empty() {
            continue;
        }
        let required_margin = match comparator {
            ConfirmatoryComparator::FixedSuperiority
            | ConfirmatoryComparator::RandomValidSuperiority => superiority_margin,
            ConfirmatoryComparator::HeuristicNonInferiority => noninferiority_margin,
        };
        let seed = plan.rng_seed ^ stable_hash(&format!("ranked|{comparator:?}"));
        let effect_values: Vec<_> = values
            .iter()
            .map(|observation| if observation.symthaea_won { 0.5 } else { -0.5 })
            .collect();
        let confidence_interval = two_way_cluster_bootstrap_interval(
            &values,
            plan.bootstrap_replicates,
            plan.alpha,
            seed ^ 0x2A11_C1A5,
        );
        let family_effects = family_centered_effects(&values, required_margin);
        let raw_one_sided_p = sign_randomization_p_value(
            &family_effects,
            plan.randomization_replicates,
            seed ^ 0x51A9_5EED,
        );
        let families: BTreeSet<_> = values
            .iter()
            .map(|observation| observation.family_id.as_str())
            .collect();
        let participants: BTreeSet<_> = values
            .iter()
            .map(|observation| observation.participant_token.as_str())
            .collect();
        let effect = mean(&effect_values);
        comparisons.push(RankedPreferenceEffect {
            comparator,
            observation_count: values.len(),
            participant_count: participants.len(),
            family_count: families.len(),
            symthaea_win_probability: effect + 0.5,
            effect_over_even_odds: effect,
            confidence_interval,
            required_margin,
            raw_one_sided_p,
            holm_adjusted_p: 1.0,
            margin_gate_passed: confidence_interval[0] > required_margin,
            inferential_gate_passed: false,
        });
    }
    apply_holm(&mut comparisons, plan.alpha);
    let success = issues.is_empty()
        && comparisons.len() == ConfirmatoryComparator::ALL.len()
        && comparisons
            .iter()
            .all(|comparison| comparison.inferential_gate_passed);
    RankedPreferenceConclusion {
        success,
        comparisons,
        observations,
        issues,
        rationale: vec![
            "complete ranks were analyzed as pairwise wins rather than evenly spaced scores".into(),
            "confidence intervals resampled participant and musical-family clusters independently".into(),
            "family-level sign randomization and Holm correction guarded the three primary comparator claims".into(),
            format!("the ranked preference gate {}", pass_fail(success)),
        ],
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_inputs(
    manifest: &FrozenStudyManifest,
    methodology: &FrozenMethodologyPlan,
    base_schedule: &BlindedSchedule,
    codebook: &BlindingCodebook,
    cohort: &ParticipantCohortSpec,
    participant_schedule: &ParticipantScheduleBook,
    envelope: &ParticipantEvidenceEnvelope,
    plan: &RankedPreferenceAnalysisPlan,
) -> Vec<RankedPreferenceIssue> {
    let mut issues = Vec::new();
    if plan.analysis_version != RANKED_PREFERENCE_ANALYSIS_VERSION {
        issues.push(RankedPreferenceIssue::WrongAnalysisVersion {
            found: plan.analysis_version.clone(),
        });
    }
    if !plan.alpha.is_finite() || plan.alpha <= 0.0 || plan.alpha > 0.10 {
        issues.push(RankedPreferenceIssue::InvalidAlpha);
    }
    if plan.alpha.to_bits() != manifest.alpha.to_bits() {
        issues.push(RankedPreferenceIssue::AlphaMismatch);
    }
    if plan.minimum_families < MIN_RANKED_PREFERENCE_FAMILIES {
        issues.push(RankedPreferenceIssue::MinimumFamiliesTooSmall {
            found: plan.minimum_families,
            required: MIN_RANKED_PREFERENCE_FAMILIES,
        });
    }
    if plan.minimum_participants < MIN_RANKED_PREFERENCE_PARTICIPANTS {
        issues.push(RankedPreferenceIssue::MinimumParticipantsTooSmall {
            found: plan.minimum_participants,
            required: MIN_RANKED_PREFERENCE_PARTICIPANTS,
        });
    }
    if plan.bootstrap_replicates < MIN_RANKED_PREFERENCE_BOOTSTRAP_REPLICATES {
        issues.push(RankedPreferenceIssue::TooFewBootstrapReplicates {
            found: plan.bootstrap_replicates,
            required: MIN_RANKED_PREFERENCE_BOOTSTRAP_REPLICATES,
        });
    }
    if plan.randomization_replicates < MIN_RANKED_PREFERENCE_RANDOMIZATION_REPLICATES {
        issues.push(RankedPreferenceIssue::TooFewRandomizationReplicates {
            found: plan.randomization_replicates,
            required: MIN_RANKED_PREFERENCE_RANDOMIZATION_REPLICATES,
        });
    }
    if !methodology.validate(manifest).is_empty() {
        issues.push(RankedPreferenceIssue::InvalidMethodology);
    }
    if methodology
        .primary_endpoint()
        .is_none_or(|endpoint| endpoint.endpoint != ConfirmatoryEndpoint::Preference)
    {
        issues.push(RankedPreferenceIssue::PrimaryEndpointIsNotPreference);
    }
    verify_digest(
        "manifest_sha256",
        canonical_json_sha256(manifest),
        &plan.manifest_sha256,
        &mut issues,
    );
    verify_digest(
        "methodology_sha256",
        canonical_json_sha256(methodology),
        &plan.methodology_sha256,
        &mut issues,
    );
    verify_digest(
        "participant_schedule_sha256",
        canonical_json_sha256(participant_schedule),
        &plan.participant_schedule_sha256,
        &mut issues,
    );
    verify_digest(
        "participant_evidence_sha256",
        canonical_json_sha256(envelope),
        &plan.participant_evidence_sha256,
        &mut issues,
    );
    issues.extend(
        validate_participant_evidence(
            manifest,
            base_schedule,
            codebook,
            cohort,
            participant_schedule,
            envelope,
        )
        .into_iter()
        .map(|issue| RankedPreferenceIssue::ParticipantEvidence { issue }),
    );
    issues
}

fn compile_ranked_observations(
    manifest: &FrozenStudyManifest,
    codebook: &BlindingCodebook,
    participant_schedule: &ParticipantScheduleBook,
    envelope: &ParticipantEvidenceEnvelope,
) -> Result<Vec<RankedPairObservation>, Vec<RankedPreferenceIssue>> {
    let assignments: BTreeMap<_, _> = participant_schedule
        .blocks
        .iter()
        .map(|block| (block.block_id.as_str(), block))
        .collect();
    let arms: BTreeMap<_, _> = codebook
        .entries
        .iter()
        .map(|entry| (entry.presentation_id.as_str(), entry.arm))
        .collect();
    let families: BTreeMap<_, _> = manifest
        .fixtures
        .iter()
        .map(|fixture| (fixture.key.clone(), fixture.family_id.as_str()))
        .collect();
    let mut issues = Vec::new();
    let mut observations = Vec::new();
    for block in envelope
        .evidence
        .listener_blocks
        .iter()
        .filter(|block| block.status == EvidenceBlockStatus::Included)
    {
        let Some(assignment) = assignments.get(block.block_id.as_str()) else {
            continue;
        };
        let mut ranks = BTreeMap::new();
        let mut positions = BTreeMap::new();
        for (position, response) in block.responses.iter().enumerate() {
            let Some(arm) = arms.get(response.presentation_id.as_str()) else {
                continue;
            };
            if ranks.insert(*arm, response.preference_rank).is_some() {
                issues.push(RankedPreferenceIssue::DuplicateArmRank {
                    block_id: block.block_id.clone(),
                    arm: *arm,
                });
            }
            positions.insert(*arm, position as u8);
        }
        for arm in CognitivePolicyArm::ALL {
            if !ranks.contains_key(&arm) {
                issues.push(RankedPreferenceIssue::MissingArmRank {
                    block_id: block.block_id.clone(),
                    arm,
                });
            }
        }
        let Some(symthaea_rank) = ranks.get(&CognitivePolicyArm::Symthaea).copied() else {
            continue;
        };
        let Some(symthaea_position) = positions.get(&CognitivePolicyArm::Symthaea).copied() else {
            continue;
        };
        for comparator in ConfirmatoryComparator::ALL {
            let arm = comparator_arm(comparator);
            let Some(comparator_rank) = ranks.get(&arm).copied() else {
                continue;
            };
            let Some(comparator_position) = positions.get(&arm).copied() else {
                continue;
            };
            observations.push(RankedPairObservation {
                participant_token: assignment.participant_token.clone(),
                family_id: families[&assignment.key].into(),
                fixture_id: assignment.key.fixture_id.clone(),
                comparator,
                symthaea_rank,
                comparator_rank,
                symthaea_position,
                comparator_position,
                symthaea_won: symthaea_rank < comparator_rank,
            });
        }
    }
    let family_count = observations
        .iter()
        .map(|observation| observation.family_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let participant_count = observations
        .iter()
        .map(|observation| observation.participant_token.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    if family_count < MIN_RANKED_PREFERENCE_FAMILIES {
        issues.push(RankedPreferenceIssue::TooFewFamilies {
            found: family_count,
            required: MIN_RANKED_PREFERENCE_FAMILIES,
        });
    }
    if participant_count < MIN_RANKED_PREFERENCE_PARTICIPANTS {
        issues.push(RankedPreferenceIssue::TooFewParticipants {
            found: participant_count,
            required: MIN_RANKED_PREFERENCE_PARTICIPANTS,
        });
    }
    if issues.is_empty() {
        Ok(observations)
    } else {
        Err(issues)
    }
}

fn two_way_cluster_bootstrap_interval(
    observations: &[&RankedPairObservation],
    replicates: usize,
    alpha: f64,
    seed: u64,
) -> [f64; 2] {
    let families: Vec<_> = observations
        .iter()
        .map(|observation| observation.family_id.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let participants: Vec<_> = observations
        .iter()
        .map(|observation| observation.participant_token.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut rng = SplitMix64::new(seed);
    let mut samples = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut family_weights: BTreeMap<&str, usize> = BTreeMap::new();
        let mut participant_weights: BTreeMap<&str, usize> = BTreeMap::new();
        for _ in 0..families.len() {
            *family_weights
                .entry(families[rng.index(families.len())])
                .or_default() += 1;
        }
        for _ in 0..participants.len() {
            *participant_weights
                .entry(participants[rng.index(participants.len())])
                .or_default() += 1;
        }
        let mut weighted_total = 0.0;
        let mut total_weight = 0usize;
        for observation in observations {
            let weight = family_weights
                .get(observation.family_id.as_str())
                .copied()
                .unwrap_or_default()
                * participant_weights
                    .get(observation.participant_token.as_str())
                    .copied()
                    .unwrap_or_default();
            if weight > 0 {
                let effect = if observation.symthaea_won { 0.5 } else { -0.5 };
                weighted_total += effect * weight as f64;
                total_weight += weight;
            }
        }
        if total_weight > 0 {
            samples.push(weighted_total / total_weight as f64);
        }
    }
    samples.sort_by(f64::total_cmp);
    let tail = alpha / 2.0;
    [
        samples[percentile_index(samples.len(), tail)],
        samples[percentile_index(samples.len(), 1.0 - tail)],
    ]
}

fn family_centered_effects(
    observations: &[&RankedPairObservation],
    required_margin: f64,
) -> Vec<f64> {
    let mut by_family: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for observation in observations {
        by_family
            .entry(observation.family_id.as_str())
            .or_default()
            .push(if observation.symthaea_won { 0.5 } else { -0.5 });
    }
    by_family
        .into_values()
        .map(|values| mean(&values) - required_margin)
        .collect()
}

fn sign_randomization_p_value(values: &[f64], replicates: usize, seed: u64) -> f64 {
    let observed = mean(values);
    if observed <= 0.0 {
        return 1.0;
    }
    if values.len() <= 20 {
        let permutations = 1usize << values.len();
        let extreme = (0..permutations)
            .filter(|mask| {
                let permuted = values
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        if mask & (1 << index) == 0 {
                            *value
                        } else {
                            -*value
                        }
                    })
                    .sum::<f64>()
                    / values.len() as f64;
                permuted >= observed - f64::EPSILON
            })
            .count();
        return extreme as f64 / permutations as f64;
    }
    let mut rng = SplitMix64::new(seed);
    let extreme = (0..replicates)
        .filter(|_| {
            let permuted = values
                .iter()
                .map(|value| {
                    if rng.next_u64() & 1 == 0 {
                        *value
                    } else {
                        -*value
                    }
                })
                .sum::<f64>()
                / values.len() as f64;
            permuted >= observed - f64::EPSILON
        })
        .count();
    (extreme + 1) as f64 / (replicates + 1) as f64
}

fn apply_holm(comparisons: &mut [RankedPreferenceEffect], alpha: f64) {
    let mut order: Vec<_> = (0..comparisons.len()).collect();
    order.sort_by(|left, right| {
        comparisons[*left]
            .raw_one_sided_p
            .total_cmp(&comparisons[*right].raw_one_sided_p)
    });
    let count = order.len();
    let mut running = 0.0f64;
    for (rank, index) in order.into_iter().enumerate() {
        let adjusted = ((count - rank) as f64 * comparisons[index].raw_one_sided_p).min(1.0);
        running = running.max(adjusted);
        comparisons[index].holm_adjusted_p = running;
        comparisons[index].inferential_gate_passed =
            comparisons[index].margin_gate_passed && running <= alpha;
    }
}

fn comparator_arm(comparator: ConfirmatoryComparator) -> CognitivePolicyArm {
    match comparator {
        ConfirmatoryComparator::FixedSuperiority => CognitivePolicyArm::Fixed,
        ConfirmatoryComparator::RandomValidSuperiority => CognitivePolicyArm::RandomValid,
        ConfirmatoryComparator::HeuristicNonInferiority => CognitivePolicyArm::Heuristic,
    }
}

fn verify_digest(
    field: &str,
    result: Result<String, serde_json::Error>,
    expected: &str,
    issues: &mut Vec<RankedPreferenceIssue>,
) {
    match result {
        Ok(value) if value == expected => {}
        Ok(_) => issues.push(RankedPreferenceIssue::DigestMismatch {
            field: field.into(),
        }),
        Err(_) => issues.push(RankedPreferenceIssue::SerializationFailed {
            field: field.into(),
        }),
    }
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn index(&mut self, len: usize) -> usize {
        (self.next_u64() % len as u64) as usize
    }
}

fn percentile_index(len: usize, percentile: f64) -> usize {
    (((len.saturating_sub(1)) as f64 * percentile).round() as usize).min(len.saturating_sub(1))
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn stable_hash(value: &str) -> u64 {
    value.bytes().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

fn pass_fail(value: bool) -> &'static str {
    if value { "passed" } else { "failed" }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(
        participant: usize,
        family: usize,
        comparator: ConfirmatoryComparator,
        won: bool,
    ) -> RankedPairObservation {
        RankedPairObservation {
            participant_token: format!("P{participant}"),
            family_id: format!("F{family}"),
            fixture_id: format!("X{family}"),
            comparator,
            symthaea_rank: if won { 1 } else { 2 },
            comparator_rank: if won { 2 } else { 1 },
            symthaea_position: 0,
            comparator_position: 1,
            symthaea_won: won,
        }
    }

    #[test]
    fn direct_rank_analysis_does_not_assume_equal_rank_spacing() {
        let owned: Vec<_> = (0..12)
            .flat_map(|participant| {
                (0..8).map(move |family| {
                    observation(
                        participant,
                        family,
                        ConfirmatoryComparator::FixedSuperiority,
                        (participant + family) % 5 != 0,
                    )
                })
            })
            .collect();
        let borrowed: Vec<_> = owned.iter().collect();
        let interval = two_way_cluster_bootstrap_interval(
            &borrowed,
            MIN_RANKED_PREFERENCE_BOOTSTRAP_REPLICATES,
            0.05,
            42,
        );
        assert!(interval[0].is_finite());
        assert!(interval[1].is_finite());
        assert!(interval[0] <= interval[1]);
    }
}
