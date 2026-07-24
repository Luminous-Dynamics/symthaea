// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Family-clustered confirmatory analysis for the single frozen primary claim.
//!
//! Related themes and orchestrations are not independent observations. This
//! analysis first averages paired fixture effects inside each musical family,
//! then performs bootstrap and sign-randomization inference over families.

use crate::cognitive_experiment::{CognitivePolicyArm, CognitiveTrialRecord, validate_experiment};
use crate::confirmatory_analysis::ConfirmatoryComparator;
use crate::evidence_digest::canonical_json_sha256;
use crate::experiment_manifest::{ConfirmatoryEndpoint, FrozenStudyManifest, StudySplit};
use crate::methodology_plan::{EndpointRole, FrozenMethodologyPlan};
use crate::study_evidence::CompiledStudyDataset;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const FAMILY_CLUSTERED_ANALYSIS_VERSION: &str = "symthaea-muse-family-clustered-analysis-v1";
pub const MIN_CONFIRMATORY_FAMILIES: usize = 8;
pub const MIN_CLUSTER_BOOTSTRAP_REPLICATES: usize = 2_000;
pub const MIN_CLUSTER_RANDOMIZATION_REPLICATES: usize = 10_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyClusteredAnalysisPlan {
    pub analysis_version: String,
    pub manifest_sha256: String,
    pub methodology_sha256: String,
    pub dataset_sha256: String,
    pub alpha: f64,
    pub minimum_confirmatory_families: usize,
    pub bootstrap_replicates: usize,
    pub randomization_replicates: usize,
    pub rng_seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyClusteredEffect {
    pub endpoint: ConfirmatoryEndpoint,
    pub comparator: ConfirmatoryComparator,
    pub family_count: usize,
    pub fixture_count: usize,
    pub mean_effect: f64,
    pub confidence_interval: [f64; 2],
    pub required_margin: f64,
    pub raw_one_sided_p: f64,
    pub holm_adjusted_p: f64,
    pub margin_gate_passed: bool,
    pub inferential_gate_passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyClusteredConclusion {
    pub success: bool,
    pub endpoint: Option<ConfirmatoryEndpoint>,
    pub comparisons: Vec<FamilyClusteredEffect>,
    pub issues: Vec<FamilyClusteredIssue>,
    pub rationale: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FamilyClusteredIssue {
    WrongAnalysisVersion {
        found: String,
    },
    InvalidAlpha,
    AlphaMismatch,
    TooFewBootstrapReplicates {
        found: usize,
        required: usize,
    },
    TooFewRandomizationReplicates {
        found: usize,
        required: usize,
    },
    InvalidMethodology,
    ManifestSerializationFailed,
    MethodologySerializationFailed,
    DatasetSerializationFailed,
    ManifestDigestMismatch,
    MethodologyDigestMismatch,
    DatasetDigestMismatch,
    MissingPrimaryEndpoint,
    MultiplePrimaryEndpoints,
    PreferenceRequiresDirectRankAnalysis,
    TooFewConfirmatoryFamilies {
        found: usize,
        required: usize,
    },
    InvalidExperimentRecords,
    MissingFixtureArm {
        fixture_id: String,
        arm: CognitivePolicyArm,
    },
    MissingEndpointValue {
        fixture_id: String,
        arm: CognitivePolicyArm,
        endpoint: ConfirmatoryEndpoint,
    },
    FamilyCrossesSplits {
        family_id: String,
    },
}

pub fn analyze_family_clustered(
    manifest: &FrozenStudyManifest,
    methodology: &FrozenMethodologyPlan,
    dataset: &CompiledStudyDataset,
    plan: &FamilyClusteredAnalysisPlan,
) -> FamilyClusteredConclusion {
    let mut issues = validate_inputs(manifest, methodology, dataset, plan);
    let Some(primary) = methodology.primary_endpoint() else {
        if !issues.contains(&FamilyClusteredIssue::MissingPrimaryEndpoint) {
            issues.push(FamilyClusteredIssue::MissingPrimaryEndpoint);
        }
        return FamilyClusteredConclusion {
            success: false,
            endpoint: None,
            comparisons: Vec::new(),
            issues,
            rationale: vec![
                "analysis refused because no unique primary endpoint was frozen".into(),
            ],
        };
    };

    let mut comparisons = Vec::new();
    for comparator in ConfirmatoryComparator::ALL {
        match family_effects(manifest, &dataset.records, primary.endpoint, comparator) {
            Ok((family_values, fixture_count)) => {
                let required_margin = match comparator {
                    ConfirmatoryComparator::FixedSuperiority
                    | ConfirmatoryComparator::RandomValidSuperiority => {
                        primary.superiority_margin.unwrap_or(f64::INFINITY)
                    }
                    ConfirmatoryComparator::HeuristicNonInferiority => primary
                        .heuristic_noninferiority_margin
                        .unwrap_or(f64::INFINITY),
                };
                let seed = plan.rng_seed
                    ^ stable_hash(&format!("{:?}|{:?}", primary.endpoint, comparator));
                let confidence_interval = cluster_bootstrap_interval(
                    &family_values,
                    plan.bootstrap_replicates,
                    plan.alpha,
                    seed ^ 0xF411_1A11,
                );
                let centered: Vec<_> = family_values
                    .iter()
                    .map(|value| value - required_margin)
                    .collect();
                let raw_one_sided_p = sign_randomization_p_value(
                    &centered,
                    plan.randomization_replicates,
                    seed ^ 0xC1A5_7E12,
                );
                comparisons.push(FamilyClusteredEffect {
                    endpoint: primary.endpoint,
                    comparator,
                    family_count: family_values.len(),
                    fixture_count,
                    mean_effect: mean(&family_values),
                    confidence_interval,
                    required_margin,
                    raw_one_sided_p,
                    holm_adjusted_p: 1.0,
                    margin_gate_passed: confidence_interval[0] > required_margin,
                    inferential_gate_passed: false,
                });
            }
            Err(mut found) => issues.append(&mut found),
        }
    }
    apply_holm(&mut comparisons, plan.alpha);
    let success = issues.is_empty()
        && comparisons.len() == ConfirmatoryComparator::ALL.len()
        && comparisons
            .iter()
            .all(|comparison| comparison.inferential_gate_passed);
    let rationale = vec![
        format!(
            "the frozen primary endpoint was {:?}",
            primary.endpoint
        ),
        "paired fixture effects were averaged within musical family before inference".into(),
        format!(
            "the family-clustered inferential gate {} across {} comparator tests",
            pass_fail(success),
            comparisons.len()
        ),
        "secondary and exploratory endpoints are reported separately and cannot rescue the primary claim".into(),
    ];
    FamilyClusteredConclusion {
        success,
        endpoint: Some(primary.endpoint),
        comparisons,
        issues,
        rationale,
    }
}

fn validate_inputs(
    manifest: &FrozenStudyManifest,
    methodology: &FrozenMethodologyPlan,
    dataset: &CompiledStudyDataset,
    plan: &FamilyClusteredAnalysisPlan,
) -> Vec<FamilyClusteredIssue> {
    let mut issues = Vec::new();
    if plan.analysis_version != FAMILY_CLUSTERED_ANALYSIS_VERSION {
        issues.push(FamilyClusteredIssue::WrongAnalysisVersion {
            found: plan.analysis_version.clone(),
        });
    }
    if !plan.alpha.is_finite() || plan.alpha <= 0.0 || plan.alpha > 0.10 {
        issues.push(FamilyClusteredIssue::InvalidAlpha);
    }
    if plan.alpha.to_bits() != manifest.alpha.to_bits() {
        issues.push(FamilyClusteredIssue::AlphaMismatch);
    }
    if plan.bootstrap_replicates < MIN_CLUSTER_BOOTSTRAP_REPLICATES {
        issues.push(FamilyClusteredIssue::TooFewBootstrapReplicates {
            found: plan.bootstrap_replicates,
            required: MIN_CLUSTER_BOOTSTRAP_REPLICATES,
        });
    }
    if plan.randomization_replicates < MIN_CLUSTER_RANDOMIZATION_REPLICATES {
        issues.push(FamilyClusteredIssue::TooFewRandomizationReplicates {
            found: plan.randomization_replicates,
            required: MIN_CLUSTER_RANDOMIZATION_REPLICATES,
        });
    }
    if !methodology.validate(manifest).is_empty() {
        issues.push(FamilyClusteredIssue::InvalidMethodology);
    }
    match canonical_json_sha256(manifest) {
        Ok(value) if value == plan.manifest_sha256 && value == dataset.manifest_sha256 => {}
        Ok(_) => issues.push(FamilyClusteredIssue::ManifestDigestMismatch),
        Err(_) => issues.push(FamilyClusteredIssue::ManifestSerializationFailed),
    }
    match canonical_json_sha256(methodology) {
        Ok(value) if value == plan.methodology_sha256 => {}
        Ok(_) => issues.push(FamilyClusteredIssue::MethodologyDigestMismatch),
        Err(_) => issues.push(FamilyClusteredIssue::MethodologySerializationFailed),
    }
    match canonical_json_sha256(dataset) {
        Ok(value) if value == plan.dataset_sha256 => {}
        Ok(_) => issues.push(FamilyClusteredIssue::DatasetDigestMismatch),
        Err(_) => issues.push(FamilyClusteredIssue::DatasetSerializationFailed),
    }
    if methodology
        .primary_endpoint()
        .is_some_and(|endpoint| endpoint.endpoint == ConfirmatoryEndpoint::Preference)
    {
        issues.push(FamilyClusteredIssue::PreferenceRequiresDirectRankAnalysis);
    }
    let primary_count = methodology
        .endpoints
        .iter()
        .filter(|endpoint| endpoint.role == EndpointRole::Primary)
        .count();
    if primary_count == 0 {
        issues.push(FamilyClusteredIssue::MissingPrimaryEndpoint);
    } else if primary_count > 1 {
        issues.push(FamilyClusteredIssue::MultiplePrimaryEndpoints);
    }
    if !validate_experiment(&dataset.records).is_empty() {
        issues.push(FamilyClusteredIssue::InvalidExperimentRecords);
    }
    let confirmatory_families: BTreeSet<_> = manifest
        .fixtures
        .iter()
        .filter(|fixture| fixture.split == StudySplit::Confirmatory)
        .map(|fixture| fixture.family_id.as_str())
        .collect();
    let required = plan
        .minimum_confirmatory_families
        .max(MIN_CONFIRMATORY_FAMILIES);
    if confirmatory_families.len() < required {
        issues.push(FamilyClusteredIssue::TooFewConfirmatoryFamilies {
            found: confirmatory_families.len(),
            required,
        });
    }
    let mut family_splits = BTreeMap::new();
    for fixture in &manifest.fixtures {
        if let Some(previous) = family_splits.insert(fixture.family_id.as_str(), fixture.split) {
            if previous != fixture.split {
                issues.push(FamilyClusteredIssue::FamilyCrossesSplits {
                    family_id: fixture.family_id.clone(),
                });
            }
        }
    }
    issues
}

fn family_effects(
    manifest: &FrozenStudyManifest,
    records: &[CognitiveTrialRecord],
    endpoint: ConfirmatoryEndpoint,
    comparator: ConfirmatoryComparator,
) -> Result<(Vec<f64>, usize), Vec<FamilyClusteredIssue>> {
    let mut issues = Vec::new();
    let by_fixture_arm: BTreeMap<_, _> = records
        .iter()
        .map(|record| ((record.key.clone(), record.arm), record))
        .collect();
    let comparator_arm = comparator_arm(comparator);
    let mut by_family: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut fixture_count = 0usize;
    for fixture in manifest
        .fixtures
        .iter()
        .filter(|fixture| fixture.split == StudySplit::Confirmatory)
    {
        let Some(symthaea) =
            by_fixture_arm.get(&(fixture.key.clone(), CognitivePolicyArm::Symthaea))
        else {
            issues.push(FamilyClusteredIssue::MissingFixtureArm {
                fixture_id: fixture.key.fixture_id.clone(),
                arm: CognitivePolicyArm::Symthaea,
            });
            continue;
        };
        let Some(baseline) = by_fixture_arm.get(&(fixture.key.clone(), comparator_arm)) else {
            issues.push(FamilyClusteredIssue::MissingFixtureArm {
                fixture_id: fixture.key.fixture_id.clone(),
                arm: comparator_arm,
            });
            continue;
        };
        let Some(symthaea_value) = endpoint_value(symthaea, endpoint) else {
            issues.push(FamilyClusteredIssue::MissingEndpointValue {
                fixture_id: fixture.key.fixture_id.clone(),
                arm: CognitivePolicyArm::Symthaea,
                endpoint,
            });
            continue;
        };
        let Some(baseline_value) = endpoint_value(baseline, endpoint) else {
            issues.push(FamilyClusteredIssue::MissingEndpointValue {
                fixture_id: fixture.key.fixture_id.clone(),
                arm: comparator_arm,
                endpoint,
            });
            continue;
        };
        let difference = if endpoint == ConfirmatoryEndpoint::LowerTimeToCommit {
            baseline_value - symthaea_value
        } else {
            symthaea_value - baseline_value
        };
        if difference.is_finite() {
            by_family
                .entry(fixture.family_id.as_str())
                .or_default()
                .push(difference);
            fixture_count += 1;
        }
    }
    if !issues.is_empty() {
        return Err(issues);
    }
    Ok((
        by_family
            .into_values()
            .map(|values| mean(&values))
            .collect(),
        fixture_count,
    ))
}

fn endpoint_value(record: &CognitiveTrialRecord, endpoint: ConfirmatoryEndpoint) -> Option<f64> {
    match endpoint {
        ConfirmatoryEndpoint::ReturnRecognition => record
            .perceptual
            .as_ref()?
            .return_recognition_rate
            .map(f64::from),
        ConfirmatoryEndpoint::EarnedRecapitulation => record
            .perceptual
            .as_ref()?
            .earned_recapitulation
            .map(f64::from),
        ConfirmatoryEndpoint::Preference => {
            record.perceptual.as_ref()?.preference_rate.map(f64::from)
        }
        ConfirmatoryEndpoint::KeepRate => Some(if record.workflow.as_ref()?.kept {
            1.0
        } else {
            0.0
        }),
        ConfirmatoryEndpoint::LowerTimeToCommit => record
            .workflow
            .as_ref()?
            .time_to_commit_seconds
            .map(|value| value as f64),
    }
}

fn comparator_arm(comparator: ConfirmatoryComparator) -> CognitivePolicyArm {
    match comparator {
        ConfirmatoryComparator::FixedSuperiority => CognitivePolicyArm::Fixed,
        ConfirmatoryComparator::RandomValidSuperiority => CognitivePolicyArm::RandomValid,
        ConfirmatoryComparator::HeuristicNonInferiority => CognitivePolicyArm::Heuristic,
    }
}

fn cluster_bootstrap_interval(
    family_values: &[f64],
    replicates: usize,
    alpha: f64,
    seed: u64,
) -> [f64; 2] {
    let mut rng = SplitMix64::new(seed);
    let mut samples = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let total = (0..family_values.len())
            .map(|_| family_values[rng.index(family_values.len())])
            .sum::<f64>();
        samples.push(total / family_values.len() as f64);
    }
    samples.sort_by(f64::total_cmp);
    let tail = alpha / 2.0;
    [
        samples[percentile_index(samples.len(), tail)],
        samples[percentile_index(samples.len(), 1.0 - tail)],
    ]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceFamilyStatistics {
    pub mean_effect: f64,
    pub confidence_interval: [f64; 2],
    pub raw_one_sided_p: f64,
}

pub fn reference_family_statistics(
    family_values: &[f64],
    required_margin: f64,
    alpha: f64,
    bootstrap_replicates: usize,
    randomization_replicates: usize,
    seed: u64,
) -> Option<ReferenceFamilyStatistics> {
    if family_values.is_empty()
        || family_values.iter().any(|value| !value.is_finite())
        || !required_margin.is_finite()
        || !alpha.is_finite()
        || alpha <= 0.0
        || alpha >= 1.0
        || bootstrap_replicates == 0
        || randomization_replicates == 0
    {
        return None;
    }
    let centered: Vec<_> = family_values
        .iter()
        .map(|value| value - required_margin)
        .collect();
    Some(ReferenceFamilyStatistics {
        mean_effect: mean(family_values),
        confidence_interval: cluster_bootstrap_interval(
            family_values,
            bootstrap_replicates,
            alpha,
            seed,
        ),
        raw_one_sided_p: sign_randomization_p_value(
            &centered,
            randomization_replicates,
            seed ^ 0xA11C_E5E5,
        ),
    })
}

fn apply_holm(comparisons: &mut [FamilyClusteredEffect], alpha: f64) {
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
    use crate::cognitive_experiment::{
        PerceptualTrialOutcome, StructuralTrialOutcome, WorkflowTrialOutcome,
    };
    use crate::experiment_manifest::{
        FrozenStudyFixture, MIN_CONFIRMATORY_FIXTURES, MIN_PILOT_FIXTURES, STUDY_MANIFEST_VERSION,
    };
    use crate::methodology_plan::{
        CandidateSetMode, EVIDENCE_ENCODING_PROFILE, EndpointDeclaration, EndpointRole,
        EqualPolicyBudget, ExternalPreregistration, FrozenModelCheckpoint, FrozenVerifierIdentity,
        METHODOLOGY_PLAN_VERSION,
    };

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn manifest() -> FrozenStudyManifest {
        let fixtures = (0..MIN_PILOT_FIXTURES + MIN_CONFIRMATORY_FIXTURES)
            .map(|index| FrozenStudyFixture {
                key: crate::cognitive_experiment::FrozenTrialKey {
                    fixture_id: format!("fixture-{index}"),
                    seed: index as u64 + 1,
                },
                family_id: if index < MIN_PILOT_FIXTURES {
                    format!("pilot-family-{index}")
                } else {
                    format!("family-{}", (index - MIN_PILOT_FIXTURES) / 3)
                },
                split: if index < MIN_PILOT_FIXTURES {
                    StudySplit::Pilot
                } else {
                    StudySplit::Confirmatory
                },
                frozen_input_sha256: format!("{:064x}", index + 1),
                subject_sha256: DIGEST.into(),
                renderer_sha256: DIGEST.into(),
                soundfont_sha256: DIGEST.into(),
                theory_constraints_sha256: DIGEST.into(),
                tonic: "C".into(),
                meter: "4/4".into(),
                orchestration: "piano".into(),
            })
            .collect();
        FrozenStudyManifest {
            manifest_version: STUDY_MANIFEST_VERSION.into(),
            preregistration_sha256: DIGEST.into(),
            analysis_plan_sha256: DIGEST.into(),
            randomization_commitment_sha256: DIGEST.into(),
            policy_versions: CognitivePolicyArm::ALL
                .into_iter()
                .map(|arm| (arm, "policy-v1".into()))
                .collect(),
            primary_endpoints: vec![ConfirmatoryEndpoint::EarnedRecapitulation],
            alpha: 0.05,
            fixtures,
        }
    }

    fn methodology(manifest: &FrozenStudyManifest) -> FrozenMethodologyPlan {
        FrozenMethodologyPlan {
            methodology_version: METHODOLOGY_PLAN_VERSION.into(),
            manifest_sha256: canonical_json_sha256(manifest).unwrap(),
            analysis_spec_sha256: DIGEST.into(),
            evidence_encoding_profile: EVIDENCE_ENCODING_PROFILE.into(),
            external_preregistration: ExternalPreregistration {
                registry: "OSF".into(),
                record_id: "example".into(),
                frozen_at_utc: "2026-07-14T00:00:00Z".into(),
                record_sha256: DIGEST.into(),
            },
            endpoints: vec![EndpointDeclaration {
                endpoint: ConfirmatoryEndpoint::EarnedRecapitulation,
                role: EndpointRole::Primary,
                superiority_margin: Some(0.05),
                heuristic_noninferiority_margin: Some(-0.02),
                rationale: "blinded preference".into(),
            }],
            model_checkpoint: FrozenModelCheckpoint {
                checkpoint_sha256: DIGEST.into(),
                training_data_sha256: DIGEST.into(),
                training_algorithm_version: "adaptive-outcome-v2".into(),
                hyperparameters_sha256: DIGEST.into(),
                completed_updates: 100,
                pilot_cutoff_utc: "2026-07-14T00:00:00Z".into(),
                rng_seed: 7,
            },
            verifier: FrozenVerifierIdentity {
                source_revision: "deadbeef".into(),
                binary_sha256: DIGEST.into(),
                rule_set_version: "theory-validation-v1".into(),
                environment_sha256: DIGEST.into(),
            },
            policy_budget: EqualPolicyBudget {
                candidate_set_mode: CandidateSetMode::SharedAcrossArms,
                candidates_per_fixture: 5,
                max_theory_validations_per_arm: 5,
                max_policy_evaluations_per_arm: 5,
                allowed_operators_sha256: DIGEST.into(),
                compute_environment_sha256: DIGEST.into(),
            },
        }
    }

    fn dataset(manifest: &FrozenStudyManifest) -> CompiledStudyDataset {
        let records = manifest
            .fixtures
            .iter()
            .flat_map(|fixture| {
                CognitivePolicyArm::ALL.map(|arm| CognitiveTrialRecord {
                    key: fixture.key.clone(),
                    arm,
                    frozen_input_sha256: fixture.frozen_input_sha256.clone(),
                    policy_version: "policy-v1".into(),
                    structural: StructuralTrialOutcome {
                        hard_constraints_valid: true,
                        obligations_total: 4,
                        obligations_fulfilled: 4,
                        voice_leading_violations: 0,
                        motif_return_similarity: Some(0.98),
                        tonic_returned: true,
                    },
                    perceptual: Some(PerceptualTrialOutcome {
                        listener_count: 12,
                        return_recognition_rate: Some(0.8),
                        development_instability: Some(0.7),
                        earned_recapitulation: Some(match arm {
                            CognitivePolicyArm::Fixed => 0.35,
                            CognitivePolicyArm::RandomValid => 0.30,
                            CognitivePolicyArm::Heuristic => 0.72,
                            CognitivePolicyArm::Symthaea => 0.82,
                        }),
                        preference_rate: Some(match arm {
                            CognitivePolicyArm::Fixed => 0.35,
                            CognitivePolicyArm::RandomValid => 0.30,
                            CognitivePolicyArm::Heuristic => 0.72,
                            CognitivePolicyArm::Symthaea => 0.82,
                        }),
                    }),
                    workflow: Some(WorkflowTrialOutcome {
                        kept: arm == CognitivePolicyArm::Symthaea,
                        edited: false,
                        rejected: arm != CognitivePolicyArm::Symthaea,
                        time_to_commit_seconds: Some(60),
                    }),
                })
            })
            .collect();
        CompiledStudyDataset {
            manifest_sha256: canonical_json_sha256(manifest).unwrap(),
            schedule_sha256: DIGEST.into(),
            codebook_sha256: DIGEST.into(),
            raw_evidence_sha256: DIGEST.into(),
            included_listener_blocks: 12,
            excluded_listener_blocks: 0,
            included_workflow_blocks: manifest.fixtures.len(),
            excluded_workflow_blocks: 0,
            records,
        }
    }

    #[test]
    fn language_neutral_reference_statistics_match_golden_fixture() {
        let summary = reference_family_statistics(
            &[0.12, 0.08, 0.11, 0.09, 0.14, 0.07, 0.10, 0.13],
            0.05,
            0.05,
            2_000,
            10_000,
            424_242,
        )
        .unwrap();
        assert!((summary.mean_effect - 0.10500000000000001).abs() < 1e-12);
        assert!((summary.confidence_interval[0] - 0.09).abs() < 1e-12);
        assert!((summary.confidence_interval[1] - 0.12000000000000001).abs() < 1e-12);
        assert!((summary.raw_one_sided_p - 0.00390625).abs() < 1e-12);
    }

    #[test]
    fn related_fixtures_are_clustered_before_inference() {
        let manifest = manifest();
        let methodology = methodology(&manifest);
        let dataset = dataset(&manifest);
        let plan = FamilyClusteredAnalysisPlan {
            analysis_version: FAMILY_CLUSTERED_ANALYSIS_VERSION.into(),
            manifest_sha256: canonical_json_sha256(&manifest).unwrap(),
            methodology_sha256: canonical_json_sha256(&methodology).unwrap(),
            dataset_sha256: canonical_json_sha256(&dataset).unwrap(),
            alpha: 0.05,
            minimum_confirmatory_families: MIN_CONFIRMATORY_FAMILIES,
            bootstrap_replicates: MIN_CLUSTER_BOOTSTRAP_REPLICATES,
            randomization_replicates: MIN_CLUSTER_RANDOMIZATION_REPLICATES,
            rng_seed: 42,
        };
        let conclusion = analyze_family_clustered(&manifest, &methodology, &dataset, &plan);
        assert!(conclusion.success, "{conclusion:#?}");
        assert!(
            conclusion
                .comparisons
                .iter()
                .all(|comparison| comparison.family_count == 8)
        );
    }
}
