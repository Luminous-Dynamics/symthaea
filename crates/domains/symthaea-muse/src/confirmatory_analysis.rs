// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Confirmatory paired analysis for the blinded four-arm cognition study.
//!
//! Effects are computed at the frozen-fixture level. Bootstrap resampling and
//! sign randomization therefore preserve the paired design. Holm correction is
//! applied across every preregistered primary comparison.

use crate::cognitive_experiment::{
    CognitivePolicyArm, CognitiveTrialRecord, ExperimentIssue, conclude_first_experiment,
    summarize_experiment, validate_experiment,
};
use crate::evidence_digest::canonical_json_sha256;
use crate::experiment_manifest::{
    ConfirmatoryEndpoint, FrozenStudyManifest, MIN_CONFIRMATORY_FIXTURES, StudySplit,
};
use crate::study_evidence::CompiledStudyDataset;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CONFIRMATORY_ANALYSIS_VERSION: &str = "symthaea-muse-confirmatory-analysis-v1";
pub const MIN_BOOTSTRAP_REPLICATES: usize = 1_000;
pub const MIN_RANDOMIZATION_REPLICATES: usize = 10_000;
pub const MIN_LISTENERS_PER_CONFIRMATORY_FIXTURE: usize = 12;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfirmatoryAnalysisPlan {
    pub analysis_version: String,
    pub manifest_sha256: String,
    pub schedule_sha256: String,
    pub codebook_sha256: String,
    pub analysis_spec_sha256: String,
    pub alpha: f64,
    pub min_confirmatory_pairs: usize,
    pub bootstrap_replicates: usize,
    pub randomization_replicates: usize,
    pub rng_seed: u64,
    pub minimum_listeners_per_fixture: usize,
    pub primary_endpoints: Vec<ConfirmatoryEndpoint>,
    /// Frozen before outcomes are observed. Multiplicity correction still
    /// applies across every primary comparison.
    pub minimum_primary_endpoints_passing: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ConfirmatoryComparator {
    FixedSuperiority,
    RandomValidSuperiority,
    HeuristicNonInferiority,
}

impl ConfirmatoryComparator {
    pub const ALL: [Self; 3] = [
        Self::FixedSuperiority,
        Self::RandomValidSuperiority,
        Self::HeuristicNonInferiority,
    ];

    fn arm(self) -> CognitivePolicyArm {
        match self {
            Self::FixedSuperiority => CognitivePolicyArm::Fixed,
            Self::RandomValidSuperiority => CognitivePolicyArm::RandomValid,
            Self::HeuristicNonInferiority => CognitivePolicyArm::Heuristic,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedEffectEstimate {
    pub endpoint: ConfirmatoryEndpoint,
    pub comparator: ConfirmatoryComparator,
    pub paired_fixtures: usize,
    /// Positive values always favor Symthaea. For time, this is comparator time
    /// minus Symthaea time.
    pub mean_effect: f64,
    pub confidence_interval_95: [f64; 2],
    pub required_margin: f64,
    pub raw_one_sided_p: f64,
    pub holm_adjusted_p: f64,
    pub margin_gate_passed: bool,
    pub inferential_gate_passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfirmatoryEndpointConclusion {
    pub endpoint: ConfirmatoryEndpoint,
    pub success: bool,
    pub comparisons: Vec<PairedEffectEstimate>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfirmatoryStudyConclusion {
    pub success: bool,
    pub structural_gate_passed: bool,
    pub analysis_gate_passed: bool,
    pub required_primary_endpoints: usize,
    pub passing_primary_endpoints: Vec<ConfirmatoryEndpoint>,
    pub endpoint_conclusions: Vec<ConfirmatoryEndpointConclusion>,
    pub issues: Vec<ConfirmatoryAnalysisIssue>,
    pub rationale: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryAnalysisIssue {
    InvalidManifest,
    WrongAnalysisVersion {
        found: String,
    },
    AlphaMismatch,
    TooFewRequiredPairs {
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
    ConfirmatoryFixtureCountMismatch {
        planned: usize,
        frozen: usize,
    },
    TooFewListenersPerFixture {
        planned: usize,
        required: usize,
    },
    EmptyPrimaryEndpoints,
    EndpointPlanMismatch,
    DuplicatePrimaryEndpoint {
        endpoint: ConfirmatoryEndpoint,
    },
    InvalidEndpointSuccessCount,
    DatasetManifestMismatch,
    DatasetScheduleMismatch,
    DatasetCodebookMismatch,
    AnalysisSpecMismatch,
    InvalidRawEvidenceDigest,
    InvalidExperimentRecord {
        issue: ExperimentIssue,
    },
    UnknownFixture {
        fixture_id: String,
        seed: u64,
    },
    FrozenInputMismatch {
        fixture_id: String,
        seed: u64,
    },
    PolicyVersionMismatch {
        fixture_id: String,
        seed: u64,
        arm: CognitivePolicyArm,
    },
    IncompleteEndpointPairing {
        endpoint: ConfirmatoryEndpoint,
        comparator: ConfirmatoryComparator,
        found: usize,
        required: usize,
    },
    ListenerStopRuleNotMet {
        endpoint: ConfirmatoryEndpoint,
        fixture_id: String,
        seed: u64,
        found: usize,
        required: usize,
    },
    NonFiniteEndpointValue {
        endpoint: ConfirmatoryEndpoint,
        fixture_id: String,
        seed: u64,
        arm: CognitivePolicyArm,
    },
}

pub fn validate_confirmatory_inputs(
    manifest: &FrozenStudyManifest,
    dataset: &CompiledStudyDataset,
    plan: &ConfirmatoryAnalysisPlan,
) -> Vec<ConfirmatoryAnalysisIssue> {
    let mut issues = Vec::new();
    if !manifest.validate().is_empty() {
        issues.push(ConfirmatoryAnalysisIssue::InvalidManifest);
    }
    if plan.analysis_version != CONFIRMATORY_ANALYSIS_VERSION {
        issues.push(ConfirmatoryAnalysisIssue::WrongAnalysisVersion {
            found: plan.analysis_version.clone(),
        });
    }
    if !plan.alpha.is_finite() || (plan.alpha - manifest.alpha).abs() > f64::EPSILON {
        issues.push(ConfirmatoryAnalysisIssue::AlphaMismatch);
    }
    if plan.min_confirmatory_pairs < MIN_CONFIRMATORY_FIXTURES {
        issues.push(ConfirmatoryAnalysisIssue::TooFewRequiredPairs {
            found: plan.min_confirmatory_pairs,
            required: MIN_CONFIRMATORY_FIXTURES,
        });
    }
    if plan.bootstrap_replicates < MIN_BOOTSTRAP_REPLICATES {
        issues.push(ConfirmatoryAnalysisIssue::TooFewBootstrapReplicates {
            found: plan.bootstrap_replicates,
            required: MIN_BOOTSTRAP_REPLICATES,
        });
    }
    if plan.randomization_replicates < MIN_RANDOMIZATION_REPLICATES {
        issues.push(ConfirmatoryAnalysisIssue::TooFewRandomizationReplicates {
            found: plan.randomization_replicates,
            required: MIN_RANDOMIZATION_REPLICATES,
        });
    }
    let frozen_confirmatory = manifest.confirmatory_fixture_count();
    if plan.min_confirmatory_pairs != frozen_confirmatory {
        issues.push(
            ConfirmatoryAnalysisIssue::ConfirmatoryFixtureCountMismatch {
                planned: plan.min_confirmatory_pairs,
                frozen: frozen_confirmatory,
            },
        );
    }
    if plan.minimum_listeners_per_fixture < MIN_LISTENERS_PER_CONFIRMATORY_FIXTURE {
        issues.push(ConfirmatoryAnalysisIssue::TooFewListenersPerFixture {
            planned: plan.minimum_listeners_per_fixture,
            required: MIN_LISTENERS_PER_CONFIRMATORY_FIXTURE,
        });
    }
    if plan.primary_endpoints.is_empty() {
        issues.push(ConfirmatoryAnalysisIssue::EmptyPrimaryEndpoints);
    }
    if as_set(&plan.primary_endpoints) != as_set(&manifest.primary_endpoints) {
        issues.push(ConfirmatoryAnalysisIssue::EndpointPlanMismatch);
    }
    let mut endpoints = BTreeSet::new();
    for endpoint in &plan.primary_endpoints {
        if !endpoints.insert(*endpoint) {
            issues.push(ConfirmatoryAnalysisIssue::DuplicatePrimaryEndpoint {
                endpoint: *endpoint,
            });
        }
    }
    if plan.minimum_primary_endpoints_passing == 0
        || plan.minimum_primary_endpoints_passing > plan.primary_endpoints.len()
    {
        issues.push(ConfirmatoryAnalysisIssue::InvalidEndpointSuccessCount);
    }
    if !is_sha256(&plan.manifest_sha256)
        || dataset.manifest_sha256 != plan.manifest_sha256
        || canonical_json_sha256(manifest).ok().as_deref() != Some(plan.manifest_sha256.as_str())
    {
        issues.push(ConfirmatoryAnalysisIssue::DatasetManifestMismatch);
    }
    if !is_sha256(&plan.schedule_sha256) || dataset.schedule_sha256 != plan.schedule_sha256 {
        issues.push(ConfirmatoryAnalysisIssue::DatasetScheduleMismatch);
    }
    if !is_sha256(&plan.codebook_sha256) || dataset.codebook_sha256 != plan.codebook_sha256 {
        issues.push(ConfirmatoryAnalysisIssue::DatasetCodebookMismatch);
    }
    if !is_sha256(&plan.analysis_spec_sha256)
        || plan.analysis_spec_sha256 != manifest.analysis_plan_sha256
    {
        issues.push(ConfirmatoryAnalysisIssue::AnalysisSpecMismatch);
    }
    if !is_sha256(&dataset.raw_evidence_sha256) {
        issues.push(ConfirmatoryAnalysisIssue::InvalidRawEvidenceDigest);
    }

    for issue in validate_experiment(&dataset.records) {
        issues.push(ConfirmatoryAnalysisIssue::InvalidExperimentRecord { issue });
    }
    for record in &dataset.records {
        match manifest.fixture(&record.key) {
            None => issues.push(ConfirmatoryAnalysisIssue::UnknownFixture {
                fixture_id: record.key.fixture_id.clone(),
                seed: record.key.seed,
            }),
            Some(fixture) => {
                if fixture.frozen_input_sha256 != record.frozen_input_sha256 {
                    issues.push(ConfirmatoryAnalysisIssue::FrozenInputMismatch {
                        fixture_id: record.key.fixture_id.clone(),
                        seed: record.key.seed,
                    });
                }
                if manifest
                    .policy_versions
                    .get(&record.arm)
                    .is_none_or(|version| version != &record.policy_version)
                {
                    issues.push(ConfirmatoryAnalysisIssue::PolicyVersionMismatch {
                        fixture_id: record.key.fixture_id.clone(),
                        seed: record.key.seed,
                        arm: record.arm,
                    });
                }
            }
        }
    }

    let required = manifest.confirmatory_fixture_count();
    for endpoint in &plan.primary_endpoints {
        if is_listener_endpoint(*endpoint) {
            for fixture in manifest
                .fixtures
                .iter()
                .filter(|fixture| fixture.split == StudySplit::Confirmatory)
            {
                let found = dataset
                    .records
                    .iter()
                    .find(|record| {
                        record.key == fixture.key && record.arm == CognitivePolicyArm::Symthaea
                    })
                    .and_then(|record| record.perceptual.as_ref())
                    .map_or(0, |perceptual| perceptual.listener_count);
                if found < plan.minimum_listeners_per_fixture {
                    issues.push(ConfirmatoryAnalysisIssue::ListenerStopRuleNotMet {
                        endpoint: *endpoint,
                        fixture_id: fixture.key.fixture_id.clone(),
                        seed: fixture.key.seed,
                        found,
                        required: plan.minimum_listeners_per_fixture,
                    });
                }
            }
        }
        for comparator in ConfirmatoryComparator::ALL {
            let pairs = paired_differences(manifest, &dataset.records, *endpoint, comparator);
            if pairs.len() != required {
                issues.push(ConfirmatoryAnalysisIssue::IncompleteEndpointPairing {
                    endpoint: *endpoint,
                    comparator,
                    found: pairs.len(),
                    required,
                });
            }
        }
    }
    issues
}

pub fn analyze_confirmatory_study(
    manifest: &FrozenStudyManifest,
    dataset: &CompiledStudyDataset,
    plan: &ConfirmatoryAnalysisPlan,
) -> ConfirmatoryStudyConclusion {
    let issues = validate_confirmatory_inputs(manifest, dataset, plan);
    let confirmatory_records: Vec<_> = dataset
        .records
        .iter()
        .filter(|record| {
            manifest
                .fixture(&record.key)
                .is_some_and(|fixture| fixture.split == StudySplit::Confirmatory)
        })
        .cloned()
        .collect();
    let descriptive = conclude_first_experiment(&summarize_experiment(&confirmatory_records));
    let structural_gate_passed = descriptive.structural_gate_passed;

    let mut comparisons = Vec::new();
    for endpoint in &plan.primary_endpoints {
        for comparator in ConfirmatoryComparator::ALL {
            let differences =
                paired_differences(manifest, &confirmatory_records, *endpoint, comparator);
            if differences.is_empty() {
                continue;
            }
            let required_margin = required_margin(*endpoint, comparator);
            let seed = plan.rng_seed ^ stable_hash(&format!("{:?}|{:?}", endpoint, comparator));
            let confidence_interval_95 =
                bootstrap_interval(&differences, plan.bootstrap_replicates, seed ^ 0xB007_57A9);
            let centered: Vec<f64> = differences
                .iter()
                .map(|value| value - required_margin)
                .collect();
            let raw_one_sided_p = sign_randomization_p_value(
                &centered,
                plan.randomization_replicates,
                seed ^ 0x51A9_F11F,
            );
            let margin_gate_passed = confidence_interval_95[0] > required_margin;
            comparisons.push(PairedEffectEstimate {
                endpoint: *endpoint,
                comparator,
                paired_fixtures: differences.len(),
                mean_effect: mean(&differences),
                confidence_interval_95,
                required_margin,
                raw_one_sided_p,
                holm_adjusted_p: 1.0,
                margin_gate_passed,
                inferential_gate_passed: false,
            });
        }
    }
    apply_holm(&mut comparisons, plan.alpha);

    let endpoint_conclusions: Vec<_> = plan
        .primary_endpoints
        .iter()
        .map(|endpoint| {
            let endpoint_comparisons: Vec<_> = comparisons
                .iter()
                .filter(|comparison| comparison.endpoint == *endpoint)
                .cloned()
                .collect();
            ConfirmatoryEndpointConclusion {
                endpoint: *endpoint,
                success: endpoint_comparisons.len() == 3
                    && endpoint_comparisons
                        .iter()
                        .all(|comparison| comparison.inferential_gate_passed),
                comparisons: endpoint_comparisons,
            }
        })
        .collect();
    let passing_primary_endpoints: Vec<_> = endpoint_conclusions
        .iter()
        .filter(|conclusion| conclusion.success)
        .map(|conclusion| conclusion.endpoint)
        .collect();
    let analysis_gate_passed = issues.is_empty()
        && passing_primary_endpoints.len() >= plan.minimum_primary_endpoints_passing;
    let success = structural_gate_passed && analysis_gate_passed;
    let rationale = vec![
        format!(
            "structural gate {} on all {} frozen confirmatory fixtures",
            pass_fail(structural_gate_passed),
            manifest.confirmatory_fixture_count()
        ),
        format!(
            "paired inferential gate {} with Holm correction across {} primary comparisons",
            pass_fail(analysis_gate_passed),
            comparisons.len()
        ),
        format!(
            "{} primary endpoints passed; {} were preregistered as required",
            passing_primary_endpoints.len(),
            plan.minimum_primary_endpoints_passing
        ),
        "a pass supports only the preregistered Sonata-study endpoints; it does not generalize to other forms, renderers, cultures, or claims of consciousness".into(),
    ];
    ConfirmatoryStudyConclusion {
        success,
        structural_gate_passed,
        analysis_gate_passed,
        required_primary_endpoints: plan.minimum_primary_endpoints_passing,
        passing_primary_endpoints,
        endpoint_conclusions,
        issues,
        rationale,
    }
}

fn is_listener_endpoint(endpoint: ConfirmatoryEndpoint) -> bool {
    matches!(
        endpoint,
        ConfirmatoryEndpoint::ReturnRecognition
            | ConfirmatoryEndpoint::EarnedRecapitulation
            | ConfirmatoryEndpoint::Preference
    )
}

fn paired_differences(
    manifest: &FrozenStudyManifest,
    records: &[CognitiveTrialRecord],
    endpoint: ConfirmatoryEndpoint,
    comparator: ConfirmatoryComparator,
) -> Vec<f64> {
    let mut grouped: BTreeMap<_, BTreeMap<CognitivePolicyArm, &CognitiveTrialRecord>> =
        BTreeMap::new();
    for record in records {
        if manifest
            .fixture(&record.key)
            .is_some_and(|fixture| fixture.split == StudySplit::Confirmatory)
        {
            grouped
                .entry(record.key.clone())
                .or_default()
                .insert(record.arm, record);
        }
    }
    grouped
        .into_values()
        .filter_map(|arms| {
            let symthaea = endpoint_value(arms.get(&CognitivePolicyArm::Symthaea)?, endpoint)?;
            let baseline = endpoint_value(arms.get(&comparator.arm())?, endpoint)?;
            let difference = if endpoint == ConfirmatoryEndpoint::LowerTimeToCommit {
                baseline - symthaea
            } else {
                symthaea - baseline
            };
            difference.is_finite().then_some(difference)
        })
        .collect()
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

fn required_margin(endpoint: ConfirmatoryEndpoint, comparator: ConfirmatoryComparator) -> f64 {
    match (endpoint, comparator) {
        (ConfirmatoryEndpoint::LowerTimeToCommit, ConfirmatoryComparator::FixedSuperiority)
        | (
            ConfirmatoryEndpoint::LowerTimeToCommit,
            ConfirmatoryComparator::RandomValidSuperiority,
        ) => 10.0,
        (
            ConfirmatoryEndpoint::LowerTimeToCommit,
            ConfirmatoryComparator::HeuristicNonInferiority,
        ) => -5.0,
        (_, ConfirmatoryComparator::FixedSuperiority)
        | (_, ConfirmatoryComparator::RandomValidSuperiority) => 0.05,
        (_, ConfirmatoryComparator::HeuristicNonInferiority) => -0.02,
    }
}

fn bootstrap_interval(values: &[f64], replicates: usize, seed: u64) -> [f64; 2] {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut bootstrap = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut total = 0.0;
        for _ in 0..values.len() {
            total += values[rng.gen_range(0..values.len())];
        }
        bootstrap.push(total / values.len() as f64);
    }
    bootstrap.sort_by(f64::total_cmp);
    let lower = percentile_index(bootstrap.len(), 0.025);
    let upper = percentile_index(bootstrap.len(), 0.975);
    [bootstrap[lower], bootstrap[upper]]
}

fn percentile_index(len: usize, percentile: f64) -> usize {
    (((len.saturating_sub(1)) as f64 * percentile).round() as usize).min(len.saturating_sub(1))
}

fn sign_randomization_p_value(values: &[f64], replicates: usize, seed: u64) -> f64 {
    let observed = mean(values);
    if observed <= 0.0 {
        return 1.0;
    }
    if values.len() <= 20 {
        let permutations = 1usize << values.len();
        let mut extreme = 0usize;
        for mask in 0..permutations {
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
            if permuted >= observed - f64::EPSILON {
                extreme += 1;
            }
        }
        return extreme as f64 / permutations as f64;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut extreme = 0usize;
    for _ in 0..replicates {
        let permuted = values
            .iter()
            .map(|value| {
                if rng.gen_range(0..2) == 0 {
                    *value
                } else {
                    -*value
                }
            })
            .sum::<f64>()
            / values.len() as f64;
        if permuted >= observed - f64::EPSILON {
            extreme += 1;
        }
    }
    (extreme + 1) as f64 / (replicates + 1) as f64
}

fn apply_holm(comparisons: &mut [PairedEffectEstimate], alpha: f64) {
    let mut order: Vec<usize> = (0..comparisons.len()).collect();
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

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn as_set(values: &[ConfirmatoryEndpoint]) -> BTreeSet<ConfirmatoryEndpoint> {
    values.iter().copied().collect()
}

fn stable_hash(value: &str) -> u64 {
    value.bytes().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
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
    use crate::evidence_digest::canonical_json_sha256;
    use crate::experiment_manifest::{
        FrozenStudyFixture, MIN_PILOT_FIXTURES, STUDY_MANIFEST_VERSION,
    };

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn manifest() -> FrozenStudyManifest {
        let mut fixtures = Vec::new();
        for index in 0..MIN_PILOT_FIXTURES + MIN_CONFIRMATORY_FIXTURES {
            fixtures.push(FrozenStudyFixture {
                key: crate::cognitive_experiment::FrozenTrialKey {
                    fixture_id: format!("fixture-{index}"),
                    seed: index as u64 + 1,
                },
                family_id: format!("family-{index}"),
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
            });
        }
        FrozenStudyManifest {
            manifest_version: STUDY_MANIFEST_VERSION.into(),
            preregistration_sha256: DIGEST.into(),
            analysis_plan_sha256: DIGEST.into(),
            randomization_commitment_sha256: DIGEST.into(),
            policy_versions: CognitivePolicyArm::ALL
                .into_iter()
                .map(|arm| (arm, "policy-v1".into()))
                .collect(),
            primary_endpoints: vec![ConfirmatoryEndpoint::Preference],
            alpha: 0.05,
            fixtures,
        }
    }

    fn dataset(manifest: &FrozenStudyManifest, symthaea_preference: f32) -> CompiledStudyDataset {
        let mut records = Vec::new();
        for fixture in &manifest.fixtures {
            for arm in CognitivePolicyArm::ALL {
                let preference = match arm {
                    CognitivePolicyArm::Fixed => 0.35,
                    CognitivePolicyArm::RandomValid => 0.30,
                    CognitivePolicyArm::Heuristic => 0.72,
                    CognitivePolicyArm::Symthaea => symthaea_preference,
                };
                records.push(CognitiveTrialRecord {
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
                        return_recognition_rate: Some(preference),
                        development_instability: Some(0.7),
                        earned_recapitulation: Some(preference),
                        preference_rate: Some(preference),
                    }),
                    workflow: Some(WorkflowTrialOutcome {
                        kept: arm == CognitivePolicyArm::Symthaea
                            || arm == CognitivePolicyArm::Heuristic,
                        edited: false,
                        rejected: arm == CognitivePolicyArm::Fixed
                            || arm == CognitivePolicyArm::RandomValid,
                        time_to_commit_seconds: Some(match arm {
                            CognitivePolicyArm::Fixed => 120,
                            CognitivePolicyArm::RandomValid => 130,
                            CognitivePolicyArm::Heuristic => 65,
                            CognitivePolicyArm::Symthaea => 60,
                        }),
                    }),
                });
            }
        }
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

    fn plan(manifest: &FrozenStudyManifest) -> ConfirmatoryAnalysisPlan {
        ConfirmatoryAnalysisPlan {
            analysis_version: CONFIRMATORY_ANALYSIS_VERSION.into(),
            manifest_sha256: canonical_json_sha256(manifest).unwrap(),
            schedule_sha256: DIGEST.into(),
            codebook_sha256: DIGEST.into(),
            analysis_spec_sha256: DIGEST.into(),
            alpha: 0.05,
            min_confirmatory_pairs: MIN_CONFIRMATORY_FIXTURES,
            bootstrap_replicates: MIN_BOOTSTRAP_REPLICATES,
            randomization_replicates: MIN_RANDOMIZATION_REPLICATES,
            rng_seed: 42,
            minimum_listeners_per_fixture: MIN_LISTENERS_PER_CONFIRMATORY_FIXTURE,
            primary_endpoints: vec![ConfirmatoryEndpoint::Preference],
            minimum_primary_endpoints_passing: 1,
        }
    }

    #[test]
    fn clear_paired_effect_passes_all_three_comparators() {
        let manifest = manifest();
        let conclusion =
            analyze_confirmatory_study(&manifest, &dataset(&manifest, 0.80), &plan(&manifest));
        assert!(conclusion.success, "{:#?}", conclusion);
        assert_eq!(
            conclusion.passing_primary_endpoints,
            vec![ConfirmatoryEndpoint::Preference]
        );
    }

    #[test]
    fn inferential_gate_rejects_descriptive_tie_with_heuristic() {
        let manifest = manifest();
        let conclusion =
            analyze_confirmatory_study(&manifest, &dataset(&manifest, 0.69), &plan(&manifest));
        assert!(!conclusion.success);
    }

    #[test]
    fn plan_cannot_change_primary_endpoints_after_manifest_freeze() {
        let manifest = manifest();
        let mut plan = plan(&manifest);
        plan.primary_endpoints = vec![ConfirmatoryEndpoint::KeepRate];
        assert!(
            validate_confirmatory_inputs(&manifest, &dataset(&manifest, 0.8), &plan)
                .iter()
                .any(|issue| matches!(issue, ConfirmatoryAnalysisIssue::EndpointPlanMismatch))
        );
    }
}
