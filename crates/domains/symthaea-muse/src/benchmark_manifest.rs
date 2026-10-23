// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Frozen benchmark authority, provenance, sampling, and claim boundaries.
//!
//! This module is deliberately benchmark-framework infrastructure rather than a
//! musical-quality oracle. It records exactly what was evaluated, how samples
//! were produced/selected, which evaluator versions were used, and what kind of
//! evidence each metric can provide. A valid benchmark manifest cannot grant
//! product authority or silently collapse automatic metrics into an artistic
//! verdict.

use crate::evidence_digest::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const BENCHMARK_MANIFEST_VERSION: &str = "melothaea-benchmark-manifest-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BenchmarkEvidenceAuthorityV1 {
    Invariant,
    Descriptive,
    Comparative,
    Causal,
    MachinePerceptual,
    HumanPerceptual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BenchmarkNonClaimV1 {
    AutomaticMetricDoesNotEstablishArtisticSuperiority,
    MachinePerceptualMetricDoesNotEstablishHumanPreference,
    ComparativeDifferenceDoesNotEstablishCausalMechanism,
    SymbolicDifferenceDoesNotEstablishHumanAudibility,
    RawObservationCountDoesNotEstablishStatisticalIndependence,
    BenchmarkResultDoesNotGrantProductAuthority,
}

pub const REQUIRED_BENCHMARK_NONCLAIMS_V1: [BenchmarkNonClaimV1; 6] = [
    BenchmarkNonClaimV1::AutomaticMetricDoesNotEstablishArtisticSuperiority,
    BenchmarkNonClaimV1::MachinePerceptualMetricDoesNotEstablishHumanPreference,
    BenchmarkNonClaimV1::ComparativeDifferenceDoesNotEstablishCausalMechanism,
    BenchmarkNonClaimV1::SymbolicDifferenceDoesNotEstablishHumanAudibility,
    BenchmarkNonClaimV1::RawObservationCountDoesNotEstablishStatisticalIndependence,
    BenchmarkNonClaimV1::BenchmarkResultDoesNotGrantProductAuthority,
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkProtocolIdentityV1 {
    pub protocol_id: String,
    pub protocol_version: String,
    pub protocol_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkImplementationIdentityV1 {
    /// Exact source-control revision of the benchmark subject/runner.
    pub source_revision: String,
    /// Canonical environment/reproducibility commitment.
    pub environment_sha256: String,
    /// Exact benchmark runner binary or executable artifact commitment.
    pub runner_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkDatasetUsageV1 {
    Synthetic,
    PublicResearch,
    ResearchOnly,
    UserSuppliedPrivate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkDatasetIdentityV1 {
    pub dataset_id: String,
    pub dataset_version: String,
    pub split_id: String,
    /// Commitment to the exact item/file manifest consumed by the benchmark.
    pub manifest_sha256: String,
    /// SPDX expression where available, otherwise a stable project license ID.
    pub license_id: String,
    pub usage: BenchmarkDatasetUsageV1,
    /// External/restricted corpora remain user-supplied or cached rather than
    /// being embedded into qualification CI.
    pub external_bytes_required: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkSubjectKindV1 {
    GeneratedContinuation,
    FullComposition,
    InterventionPair,
    ExternalCorpusItem,
    SyntheticControl,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkSubjectManifestV1 {
    pub subject_kind: BenchmarkSubjectKindV1,
    pub subject_count: usize,
    pub subject_manifest_sha256: String,
    pub dataset: Option<BenchmarkDatasetIdentityV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkSeedPolicyV1 {
    /// Canonical strictly increasing seed set frozen before execution.
    Frozen(Vec<u64>),
    /// The external benchmark protocol owns randomization/seed semantics.
    ExternalProtocol,
    /// Appropriate only when generation randomness is not part of the subject.
    NotApplicable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkSamplingPolicyV1 {
    pub samples_per_subject: u32,
    pub seed_policy: BenchmarkSeedPolicyV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkSelectorV1 {
    /// Manual cherry-pick by the submitting system/operator, as permitted by
    /// some external generation benchmarks.
    SystemSubmitterManual,
    /// Deterministic selection by one metric declared in this same manifest.
    DeterministicMetric { metric_id: String },
    /// Blinded human selection under a separately committed protocol.
    BlindedHumanProtocol { protocol_sha256: String },
    /// Selection semantics are owned by an external benchmark protocol.
    ExternalProtocol {
        protocol_id: String,
        protocol_sha256: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkSelectionPolicyV1 {
    /// Every generated sample is retained in the benchmark result population.
    AllGenerated,
    /// `n` candidates are produced for each subject and one is selected by the
    /// declared selector. The unselected candidates remain provenance-visible.
    BestOfN {
        n: u32,
        selector: BenchmarkSelectorV1,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkEvaluatorKindV1 {
    NativeAutomatic,
    ExternalAutomatic,
    HumanProtocol,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkMetricSpecV1 {
    pub metric_id: String,
    pub metric_version: String,
    pub authority: BenchmarkEvidenceAuthorityV1,
    pub evaluator_kind: BenchmarkEvaluatorKindV1,
    /// Exact evaluator implementation/model/protocol commitment.
    pub evaluator_sha256: String,
    /// Canonical metric parameter/configuration commitment.
    pub parameters_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkStatisticsPlanV1 {
    /// Human-readable but frozen unit, e.g. `piece`, `seed`, or `participant-item`.
    pub observation_unit: String,
    /// Exact preregistered/declared aggregation or statistical plan.
    pub plan_sha256: String,
    /// Must remain false in v1. Raw event/sample count is never permission to
    /// claim that those observations are statistically independent.
    pub raw_observation_count_is_independence_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenBenchmarkManifestV1 {
    pub manifest_version: String,
    pub benchmark_id: String,
    pub protocol: BenchmarkProtocolIdentityV1,
    pub implementation: BenchmarkImplementationIdentityV1,
    pub subjects: BenchmarkSubjectManifestV1,
    pub sampling: BenchmarkSamplingPolicyV1,
    pub selection: BenchmarkSelectionPolicyV1,
    /// Canonical ascending metric-id order is part of the v1 representation.
    pub metrics: Vec<BenchmarkMetricSpecV1>,
    pub statistics: BenchmarkStatisticsPlanV1,
    /// Exact required registry. Removing any entry invalidates the manifest.
    pub required_nonclaims: Vec<BenchmarkNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkManifestIssueV1 {
    WrongManifestVersion { found: String },
    EmptyField { field: String },
    InvalidDigest { field: String },
    InvalidSourceRevision,
    EmptySubjectSet,
    ExternalCorpusMissingDataset,
    RestrictedDatasetMustRemainExternal { dataset_id: String },
    EmptyFrozenSeedSet,
    NonCanonicalFrozenSeeds,
    ZeroSamplesPerSubject,
    InvalidBestOfN { n: u32, samples_per_subject: u32 },
    InvalidSelectorMetric { metric_id: String },
    EmptyMetricSet,
    MetricsNotCanonical,
    DuplicateMetricId { metric_id: String },
    HumanAuthorityRequiresHumanProtocol { metric_id: String },
    HumanProtocolCannotClaimNonHumanAuthority { metric_id: String },
    InvalidStatisticsIndependenceClaim,
    RequiredNonclaimsMismatch,
}

impl FrozenBenchmarkManifestV1 {
    pub fn canonical_sha256(&self) -> Result<String, serde_json::Error> {
        canonical_json_sha256(self)
    }

    pub fn validate(&self) -> Vec<BenchmarkManifestIssueV1> {
        let mut issues = Vec::new();

        if self.manifest_version != BENCHMARK_MANIFEST_VERSION {
            issues.push(BenchmarkManifestIssueV1::WrongManifestVersion {
                found: self.manifest_version.clone(),
            });
        }

        validate_nonempty("benchmark_id", &self.benchmark_id, &mut issues);
        validate_nonempty("protocol.protocol_id", &self.protocol.protocol_id, &mut issues);
        validate_nonempty(
            "protocol.protocol_version",
            &self.protocol.protocol_version,
            &mut issues,
        );
        validate_digest(
            "protocol.protocol_sha256",
            &self.protocol.protocol_sha256,
            &mut issues,
        );

        if !is_git_revision(&self.implementation.source_revision) {
            issues.push(BenchmarkManifestIssueV1::InvalidSourceRevision);
        }
        validate_digest(
            "implementation.environment_sha256",
            &self.implementation.environment_sha256,
            &mut issues,
        );
        validate_digest(
            "implementation.runner_sha256",
            &self.implementation.runner_sha256,
            &mut issues,
        );

        if self.subjects.subject_count == 0 {
            issues.push(BenchmarkManifestIssueV1::EmptySubjectSet);
        }
        validate_digest(
            "subjects.subject_manifest_sha256",
            &self.subjects.subject_manifest_sha256,
            &mut issues,
        );
        if self.subjects.subject_kind == BenchmarkSubjectKindV1::ExternalCorpusItem
            && self.subjects.dataset.is_none()
        {
            issues.push(BenchmarkManifestIssueV1::ExternalCorpusMissingDataset);
        }
        if let Some(dataset) = &self.subjects.dataset {
            validate_nonempty("subjects.dataset.dataset_id", &dataset.dataset_id, &mut issues);
            validate_nonempty(
                "subjects.dataset.dataset_version",
                &dataset.dataset_version,
                &mut issues,
            );
            validate_nonempty("subjects.dataset.split_id", &dataset.split_id, &mut issues);
            validate_nonempty("subjects.dataset.license_id", &dataset.license_id, &mut issues);
            validate_digest(
                "subjects.dataset.manifest_sha256",
                &dataset.manifest_sha256,
                &mut issues,
            );
            if matches!(
                dataset.usage,
                BenchmarkDatasetUsageV1::ResearchOnly
                    | BenchmarkDatasetUsageV1::UserSuppliedPrivate
            ) && !dataset.external_bytes_required
            {
                issues.push(BenchmarkManifestIssueV1::RestrictedDatasetMustRemainExternal {
                    dataset_id: dataset.dataset_id.clone(),
                });
            }
        }

        if self.sampling.samples_per_subject == 0 {
            issues.push(BenchmarkManifestIssueV1::ZeroSamplesPerSubject);
        }
        if let BenchmarkSeedPolicyV1::Frozen(seeds) = &self.sampling.seed_policy {
            if seeds.is_empty() {
                issues.push(BenchmarkManifestIssueV1::EmptyFrozenSeedSet);
            } else if seeds.windows(2).any(|pair| pair[0] >= pair[1]) {
                issues.push(BenchmarkManifestIssueV1::NonCanonicalFrozenSeeds);
            }
        }

        let metric_ids: BTreeSet<&str> = self
            .metrics
            .iter()
            .map(|metric| metric.metric_id.as_str())
            .collect();
        match &self.selection {
            BenchmarkSelectionPolicyV1::AllGenerated => {}
            BenchmarkSelectionPolicyV1::BestOfN { n, selector } => {
                if *n < 2 || *n != self.sampling.samples_per_subject {
                    issues.push(BenchmarkManifestIssueV1::InvalidBestOfN {
                        n: *n,
                        samples_per_subject: self.sampling.samples_per_subject,
                    });
                }
                match selector {
                    BenchmarkSelectorV1::SystemSubmitterManual => {}
                    BenchmarkSelectorV1::DeterministicMetric { metric_id } => {
                        if !metric_ids.contains(metric_id.as_str()) {
                            issues.push(BenchmarkManifestIssueV1::InvalidSelectorMetric {
                                metric_id: metric_id.clone(),
                            });
                        }
                    }
                    BenchmarkSelectorV1::BlindedHumanProtocol { protocol_sha256 } => {
                        validate_digest(
                            "selection.selector.protocol_sha256",
                            protocol_sha256,
                            &mut issues,
                        );
                    }
                    BenchmarkSelectorV1::ExternalProtocol {
                        protocol_id,
                        protocol_sha256,
                    } => {
                        validate_nonempty(
                            "selection.selector.protocol_id",
                            protocol_id,
                            &mut issues,
                        );
                        validate_digest(
                            "selection.selector.protocol_sha256",
                            protocol_sha256,
                            &mut issues,
                        );
                    }
                }
            }
        }

        if self.metrics.is_empty() {
            issues.push(BenchmarkManifestIssueV1::EmptyMetricSet);
        }
        let mut previous_metric_id: Option<&str> = None;
        let mut seen_metric_ids = BTreeSet::new();
        for metric in &self.metrics {
            validate_nonempty("metrics.metric_id", &metric.metric_id, &mut issues);
            validate_nonempty("metrics.metric_version", &metric.metric_version, &mut issues);
            validate_digest("metrics.evaluator_sha256", &metric.evaluator_sha256, &mut issues);
            validate_digest("metrics.parameters_sha256", &metric.parameters_sha256, &mut issues);
            if metric.authority == BenchmarkEvidenceAuthorityV1::HumanPerceptual
                && metric.evaluator_kind != BenchmarkEvaluatorKindV1::HumanProtocol
            {
                issues.push(BenchmarkManifestIssueV1::HumanAuthorityRequiresHumanProtocol {
                    metric_id: metric.metric_id.clone(),
                });
            }
            if metric.evaluator_kind == BenchmarkEvaluatorKindV1::HumanProtocol
                && metric.authority != BenchmarkEvidenceAuthorityV1::HumanPerceptual
            {
                issues.push(BenchmarkManifestIssueV1::HumanProtocolCannotClaimNonHumanAuthority {
                    metric_id: metric.metric_id.clone(),
                });
            }
            if let Some(previous) = previous_metric_id {
                if previous >= metric.metric_id.as_str() {
                    issues.push(BenchmarkManifestIssueV1::MetricsNotCanonical);
                }
            }
            if !seen_metric_ids.insert(metric.metric_id.as_str()) {
                issues.push(BenchmarkManifestIssueV1::DuplicateMetricId {
                    metric_id: metric.metric_id.clone(),
                });
            }
            previous_metric_id = Some(metric.metric_id.as_str());
        }

        validate_nonempty(
            "statistics.observation_unit",
            &self.statistics.observation_unit,
            &mut issues,
        );
        validate_digest("statistics.plan_sha256", &self.statistics.plan_sha256, &mut issues);
        if self.statistics.raw_observation_count_is_independence_claim {
            issues.push(BenchmarkManifestIssueV1::InvalidStatisticsIndependenceClaim);
        }

        if self.required_nonclaims.as_slice() != REQUIRED_BENCHMARK_NONCLAIMS_V1.as_slice() {
            issues.push(BenchmarkManifestIssueV1::RequiredNonclaimsMismatch);
        }

        issues
    }
}

fn validate_nonempty(field: &str, value: &str, issues: &mut Vec<BenchmarkManifestIssueV1>) {
    if value.trim().is_empty() || value.trim() != value {
        issues.push(BenchmarkManifestIssueV1::EmptyField {
            field: field.into(),
        });
    }
}

fn validate_digest(field: &str, value: &str, issues: &mut Vec<BenchmarkManifestIssueV1>) {
    if !is_lower_hex(value, 64) {
        issues.push(BenchmarkManifestIssueV1::InvalidDigest {
            field: field.into(),
        });
    }
}

fn is_git_revision(value: &str) -> bool {
    is_lower_hex(value, 40) || is_lower_hex(value, 64)
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

    fn metric(id: &str, authority: BenchmarkEvidenceAuthorityV1) -> BenchmarkMetricSpecV1 {
        BenchmarkMetricSpecV1 {
            metric_id: id.into(),
            metric_version: "v1".into(),
            authority,
            evaluator_kind: BenchmarkEvaluatorKindV1::NativeAutomatic,
            evaluator_sha256: DIGEST_A.into(),
            parameters_sha256: DIGEST_B.into(),
        }
    }

    fn manifest() -> FrozenBenchmarkManifestV1 {
        FrozenBenchmarkManifestV1 {
            manifest_version: BENCHMARK_MANIFEST_VERSION.into(),
            benchmark_id: "mel-bench-fixture".into(),
            protocol: BenchmarkProtocolIdentityV1 {
                protocol_id: "external-symbolic-continuation".into(),
                protocol_version: "2026".into(),
                protocol_sha256: DIGEST_A.into(),
            },
            implementation: BenchmarkImplementationIdentityV1 {
                source_revision: REVISION.into(),
                environment_sha256: DIGEST_A.into(),
                runner_sha256: DIGEST_B.into(),
            },
            subjects: BenchmarkSubjectManifestV1 {
                subject_kind: BenchmarkSubjectKindV1::GeneratedContinuation,
                subject_count: 8,
                subject_manifest_sha256: DIGEST_A.into(),
                dataset: None,
            },
            sampling: BenchmarkSamplingPolicyV1 {
                samples_per_subject: 1,
                seed_policy: BenchmarkSeedPolicyV1::Frozen(vec![3, 11, 23, 41]),
            },
            selection: BenchmarkSelectionPolicyV1::AllGenerated,
            metrics: vec![
                metric("a-structure", BenchmarkEvidenceAuthorityV1::Descriptive),
                metric("b-control", BenchmarkEvidenceAuthorityV1::Causal),
            ],
            statistics: BenchmarkStatisticsPlanV1 {
                observation_unit: "piece".into(),
                plan_sha256: DIGEST_A.into(),
                raw_observation_count_is_independence_claim: false,
            },
            required_nonclaims: REQUIRED_BENCHMARK_NONCLAIMS_V1.to_vec(),
        }
    }

    #[test]
    fn valid_manifest_passes_and_has_stable_commitment() {
        let manifest = manifest();
        assert!(manifest.validate().is_empty());
        let digest = manifest.canonical_sha256().unwrap();
        assert_eq!(digest.len(), 64);
        assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_eq!(digest, manifest.canonical_sha256().unwrap());
    }

    #[test]
    fn best_of_n_must_disclose_the_full_generated_population() {
        let mut manifest = manifest();
        manifest.sampling.samples_per_subject = 8;
        manifest.selection = BenchmarkSelectionPolicyV1::BestOfN {
            n: 8,
            selector: BenchmarkSelectorV1::SystemSubmitterManual,
        };
        assert!(manifest.validate().is_empty());

        manifest.sampling.samples_per_subject = 4;
        assert!(manifest.validate().iter().any(|issue| matches!(
            issue,
            BenchmarkManifestIssueV1::InvalidBestOfN { .. }
        )));
    }

    #[test]
    fn deterministic_selector_must_name_a_declared_metric() {
        let mut manifest = manifest();
        manifest.sampling.samples_per_subject = 4;
        manifest.selection = BenchmarkSelectionPolicyV1::BestOfN {
            n: 4,
            selector: BenchmarkSelectorV1::DeterministicMetric {
                metric_id: "missing".into(),
            },
        };
        assert!(manifest.validate().iter().any(|issue| matches!(
            issue,
            BenchmarkManifestIssueV1::InvalidSelectorMetric { metric_id } if metric_id == "missing"
        )));
    }

    #[test]
    fn frozen_seed_set_and_metric_order_are_canonical() {
        let mut manifest = manifest();
        manifest.sampling.seed_policy = BenchmarkSeedPolicyV1::Frozen(vec![11, 3, 23]);
        assert!(manifest.validate().iter().any(|issue| matches!(
            issue,
            BenchmarkManifestIssueV1::NonCanonicalFrozenSeeds
        )));

        let mut manifest = manifest();
        manifest.metrics.swap(0, 1);
        assert!(manifest.validate().iter().any(|issue| matches!(
            issue,
            BenchmarkManifestIssueV1::MetricsNotCanonical
        )));
    }

    #[test]
    fn external_corpus_requires_versioned_dataset_identity() {
        let mut manifest = manifest();
        manifest.subjects.subject_kind = BenchmarkSubjectKindV1::ExternalCorpusItem;
        assert!(manifest.validate().iter().any(|issue| matches!(
            issue,
            BenchmarkManifestIssueV1::ExternalCorpusMissingDataset
        )));

        manifest.subjects.dataset = Some(BenchmarkDatasetIdentityV1 {
            dataset_id: "pop909".into(),
            dataset_version: "frozen-test".into(),
            split_id: "held-out".into(),
            manifest_sha256: DIGEST_A.into(),
            license_id: "research-source-license".into(),
            usage: BenchmarkDatasetUsageV1::ResearchOnly,
            external_bytes_required: true,
        });
        assert!(manifest.validate().is_empty());
    }

    #[test]
    fn restricted_dataset_cannot_be_smuggled_into_ci() {
        let mut manifest = manifest();
        manifest.subjects.subject_kind = BenchmarkSubjectKindV1::ExternalCorpusItem;
        manifest.subjects.dataset = Some(BenchmarkDatasetIdentityV1 {
            dataset_id: "restricted-corpus".into(),
            dataset_version: "v1".into(),
            split_id: "test".into(),
            manifest_sha256: DIGEST_A.into(),
            license_id: "research-only".into(),
            usage: BenchmarkDatasetUsageV1::ResearchOnly,
            external_bytes_required: false,
        });
        assert!(manifest.validate().iter().any(|issue| matches!(
            issue,
            BenchmarkManifestIssueV1::RestrictedDatasetMustRemainExternal { dataset_id }
                if dataset_id == "restricted-corpus"
        )));
    }

    #[test]
    fn human_perceptual_authority_requires_a_human_protocol() {
        let mut manifest = manifest();
        manifest.metrics = vec![metric(
            "a-human-preference",
            BenchmarkEvidenceAuthorityV1::HumanPerceptual,
        )];
        assert!(manifest.validate().iter().any(|issue| matches!(
            issue,
            BenchmarkManifestIssueV1::HumanAuthorityRequiresHumanProtocol { metric_id }
                if metric_id == "a-human-preference"
        )));

        manifest.metrics[0].evaluator_kind = BenchmarkEvaluatorKindV1::HumanProtocol;
        assert!(manifest.validate().is_empty());
    }

    #[test]
    fn human_protocol_cannot_be_relabeled_as_machine_or_causal_evidence() {
        let mut manifest = manifest();
        manifest.metrics = vec![metric(
            "a-human-rating",
            BenchmarkEvidenceAuthorityV1::MachinePerceptual,
        )];
        manifest.metrics[0].evaluator_kind = BenchmarkEvaluatorKindV1::HumanProtocol;
        assert!(manifest.validate().iter().any(|issue| matches!(
            issue,
            BenchmarkManifestIssueV1::HumanProtocolCannotClaimNonHumanAuthority { metric_id }
                if metric_id == "a-human-rating"
        )));
    }

    #[test]
    fn required_nonclaims_are_load_bearing() {
        let mut manifest = manifest();
        manifest.required_nonclaims.pop();
        assert!(manifest.validate().iter().any(|issue| matches!(
            issue,
            BenchmarkManifestIssueV1::RequiredNonclaimsMismatch
        )));
    }

    #[test]
    fn raw_count_cannot_be_declared_independence() {
        let mut manifest = manifest();
        manifest.statistics.raw_observation_count_is_independence_claim = true;
        assert!(manifest.validate().iter().any(|issue| matches!(
            issue,
            BenchmarkManifestIssueV1::InvalidStatisticsIndependenceClaim
        )));
    }

    #[test]
    fn selection_policy_changes_the_manifest_identity() {
        let baseline = manifest();
        let mut selected = baseline.clone();
        selected.sampling.samples_per_subject = 8;
        selected.selection = BenchmarkSelectionPolicyV1::BestOfN {
            n: 8,
            selector: BenchmarkSelectorV1::SystemSubmitterManual,
        };
        assert_ne!(
            baseline.canonical_sha256().unwrap(),
            selected.canonical_sha256().unwrap()
        );
    }
}
