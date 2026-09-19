// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Temporal-leakage and interpretation governance for materials benchmarks.
//!
//! A family/structure holdout answers a different question from a retrospective
//! pre-publication-data experiment, and both differ from a genuinely prospective
//! challenge. This crate makes those claim regimes machine-visible so a current
//! database snapshot cannot silently be presented as evidence that a material
//! could have been discovered before its publication.
//!
//! A post-hoc algorithm run on pre-publication data remains *retrospective*: it can
//! test whether the source data existed before disclosure, but it cannot erase
//! method-design hindsight. Only a protocol committed before target disclosure can
//! enter the prospective-registration regime.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use symthaea_materials_benchmarks::{BenchmarkArtifactRef, BlindBenchmarkManifest};
use thiserror::Error;

/// Immutable external database/archive snapshot used to derive benchmark training data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusSnapshotRef {
    /// Stable snapshot identifier.
    pub snapshot_id: String,
    /// Provider, e.g. `OQMD`, `MaterialsProject`, `NOMAD`, or another named corpus.
    pub provider: String,
    /// Provider's immutable version/release label.
    pub version: String,
    /// Date the bound snapshot became available, ISO `YYYY-MM-DD`.
    pub release_date: String,
    /// SHA-256 of the exact archived snapshot, export, or immutable snapshot manifest.
    pub artifact_sha256: String,
    /// Source license identifier or explicit license label.
    pub license: String,
}

impl CorpusSnapshotRef {
    fn validate(&self) -> Result<NaiveDate, GovernanceError> {
        nonempty("snapshot_id", &self.snapshot_id)?;
        nonempty("snapshot provider", &self.provider)?;
        nonempty("snapshot version", &self.version)?;
        nonempty("snapshot license", &self.license)?;
        validate_sha256(&self.artifact_sha256)?;
        parse_date("snapshot release_date", &self.release_date)
    }
}

/// Lineage from one exact generator-visible training artifact to source snapshots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingArtifactLineage {
    /// Exact training artifact declared in the public benchmark manifest.
    pub training_artifact: BenchmarkArtifactRef,
    /// Source snapshot IDs from which the artifact was derived.
    pub source_snapshot_ids: Vec<String>,
    /// SHA-256 of the deterministic extraction/normalization implementation.
    pub derivation_artifact_sha256: String,
}

impl TrainingArtifactLineage {
    fn validate_shape(&self) -> Result<(), GovernanceError> {
        validate_artifact(&self.training_artifact)?;
        validate_sha256(&self.derivation_artifact_sha256)?;
        if self.source_snapshot_ids.is_empty() {
            return Err(GovernanceError::TrainingLineageHasNoSnapshot(
                self.training_artifact.artifact_id.clone(),
            ));
        }
        let mut seen = HashSet::new();
        for snapshot in &self.source_snapshot_ids {
            nonempty("source_snapshot_id", snapshot)?;
            if !seen.insert(snapshot.as_str()) {
                return Err(GovernanceError::DuplicateSnapshotReference(
                    snapshot.clone(),
                ));
            }
        }
        Ok(())
    }
}

/// Public target-disclosure event relevant to temporal leakage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetDisclosure {
    /// Stable target/source identifier.
    pub target_id: String,
    /// Date on which the target result became public, ISO `YYYY-MM-DD`.
    pub public_date: String,
    /// Publication/DOI/preprint or other source identifier.
    pub source_id: String,
}

impl TargetDisclosure {
    fn validate(&self) -> Result<NaiveDate, GovernanceError> {
        nonempty("target_id", &self.target_id)?;
        nonempty("target source_id", &self.source_id)?;
        parse_date("target public_date", &self.public_date)
    }
}

/// Scientific interpretation permitted by the benchmark construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "regime", rename_all = "snake_case")]
pub enum BenchmarkClaimRegime {
    /// Current/same-era corpus with family/structure holdout.
    ///
    /// This can test generalization but supports no historical-discovery statement.
    StructuralGeneralization,

    /// Post-hoc method evaluated from snapshots demonstrably available before target disclosure.
    ///
    /// This tests historical data sufficiency while explicitly retaining algorithmic hindsight.
    RetrospectivePrepublicationData {
        /// Latest date permitted for source-data snapshots.
        knowledge_cutoff_date: String,
        /// Immutable source snapshots used to derive allowed training artifacts.
        corpus_snapshots: Vec<CorpusSnapshotRef>,
        /// Exact source lineage for every allowed training artifact.
        training_lineage: Vec<TrainingArtifactLineage>,
        /// Target disclosure events that must postdate the knowledge cutoff.
        target_disclosures: Vec<TargetDisclosure>,
    },

    /// Generator/protocol/search view irreversibly committed before target disclosure.
    ///
    /// This is the only regime structurally eligible to support a prospectively
    /// registered benchmark statement. It still does not confer material authority.
    ProspectiveRegistered {
        /// Date the complete protocol was irreversibly registered.
        registration_date: String,
        /// SHA-256 of the exact registration/preregistration artifact.
        registration_artifact_sha256: String,
        /// SHA-256 of the exact generator/model/workflow artifact fixed at registration.
        generator_artifact_sha256: String,
        /// SHA-256 of the exact public search-phase view fixed at registration.
        search_phase_view_sha256: String,
        /// Public target commitment fixed at registration.
        sealed_targets_sha256: String,
        /// Target disclosure events, all of which must occur after registration.
        target_disclosures: Vec<TargetDisclosure>,
    },
}

/// Bound temporal interpretation record for one exact benchmark manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalBenchmarkGovernance {
    /// Governance schema version.
    pub schema_version: u32,
    /// Benchmark identity.
    pub benchmark_id: String,
    /// SHA-256 of the exact public benchmark manifest.
    pub manifest_sha256: String,
    /// Claim regime and temporal evidence.
    pub claim_regime: BenchmarkClaimRegime,
}

impl TemporalBenchmarkGovernance {
    /// Validate temporal leakage constraints against the exact public manifest.
    pub fn validate_for(
        &self,
        manifest: &BlindBenchmarkManifest,
    ) -> Result<(), GovernanceError> {
        if self.schema_version != 1 {
            return Err(GovernanceError::UnsupportedSchema(self.schema_version));
        }
        manifest.validate()?;
        nonempty("benchmark_id", &self.benchmark_id)?;
        validate_sha256(&self.manifest_sha256)?;
        if self.benchmark_id != manifest.benchmark_id {
            return Err(GovernanceError::BenchmarkIdentityMismatch);
        }
        if self.manifest_sha256 != manifest.manifest_sha256()? {
            return Err(GovernanceError::ManifestDigestMismatch);
        }

        match &self.claim_regime {
            BenchmarkClaimRegime::StructuralGeneralization => Ok(()),
            BenchmarkClaimRegime::RetrospectivePrepublicationData {
                knowledge_cutoff_date,
                corpus_snapshots,
                training_lineage,
                target_disclosures,
            } => validate_retrospective(
                manifest,
                knowledge_cutoff_date,
                corpus_snapshots,
                training_lineage,
                target_disclosures,
            ),
            BenchmarkClaimRegime::ProspectiveRegistered {
                registration_date,
                registration_artifact_sha256,
                generator_artifact_sha256,
                search_phase_view_sha256,
                sealed_targets_sha256,
                target_disclosures,
            } => validate_prospective(
                manifest,
                registration_date,
                registration_artifact_sha256,
                generator_artifact_sha256,
                search_phase_view_sha256,
                sealed_targets_sha256,
                target_disclosures,
            ),
        }
    }

    /// Whether this construction is prospectively registered before disclosure.
    ///
    /// This says nothing about benchmark score, novelty, synthesis, or MAT-001 authority.
    pub fn is_prospectively_registered(&self) -> bool {
        matches!(
            &self.claim_regime,
            BenchmarkClaimRegime::ProspectiveRegistered { .. }
        )
    }

    /// Whether this is explicitly a retrospective pre-publication-data reconstruction.
    ///
    /// Such a benchmark may test data sufficiency but does not erase method hindsight.
    pub fn is_retrospective_prepublication_data(&self) -> bool {
        matches!(
            &self.claim_regime,
            BenchmarkClaimRegime::RetrospectivePrepublicationData { .. }
        )
    }

    /// Deterministic identity of the complete interpretation/governance record.
    pub fn governance_sha256(&self) -> Result<String, GovernanceError> {
        self.validate_shape_only()?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }

    fn validate_shape_only(&self) -> Result<(), GovernanceError> {
        if self.schema_version != 1 {
            return Err(GovernanceError::UnsupportedSchema(self.schema_version));
        }
        nonempty("benchmark_id", &self.benchmark_id)?;
        validate_sha256(&self.manifest_sha256)
    }
}

fn validate_retrospective(
    manifest: &BlindBenchmarkManifest,
    cutoff_text: &str,
    corpus_snapshots: &[CorpusSnapshotRef],
    training_lineage: &[TrainingArtifactLineage],
    target_disclosures: &[TargetDisclosure],
) -> Result<(), GovernanceError> {
    let cutoff = parse_date("knowledge_cutoff_date", cutoff_text)?;
    if corpus_snapshots.is_empty() {
        return Err(GovernanceError::NoCorpusSnapshots);
    }
    if target_disclosures.is_empty() {
        return Err(GovernanceError::NoTargetDisclosures);
    }

    let mut snapshot_dates = HashMap::new();
    for snapshot in corpus_snapshots {
        let date = snapshot.validate()?;
        if date > cutoff {
            return Err(GovernanceError::SnapshotAfterKnowledgeCutoff {
                snapshot_id: snapshot.snapshot_id.clone(),
                release_date: snapshot.release_date.clone(),
                cutoff: cutoff_text.to_string(),
            });
        }
        if snapshot
            .artifact_sha256
            .eq_ignore_ascii_case(&manifest.sealed_targets_sha256)
        {
            return Err(GovernanceError::TargetArtifactUsedAsCorpusSnapshot);
        }
        if snapshot_dates
            .insert(snapshot.snapshot_id.as_str(), date)
            .is_some()
        {
            return Err(GovernanceError::DuplicateSnapshotId(
                snapshot.snapshot_id.clone(),
            ));
        }
    }

    validate_training_lineage(manifest, training_lineage, &snapshot_dates)?;

    let mut targets = HashSet::new();
    for target in target_disclosures {
        let date = target.validate()?;
        if date <= cutoff {
            return Err(GovernanceError::TargetNotAfterKnowledgeCutoff {
                target_id: target.target_id.clone(),
                public_date: target.public_date.clone(),
                cutoff: cutoff_text.to_string(),
            });
        }
        if !targets.insert(target.target_id.as_str()) {
            return Err(GovernanceError::DuplicateTargetDisclosure(
                target.target_id.clone(),
            ));
        }
    }
    Ok(())
}

fn validate_training_lineage(
    manifest: &BlindBenchmarkManifest,
    lineage: &[TrainingArtifactLineage],
    snapshots: &HashMap<&str, NaiveDate>,
) -> Result<(), GovernanceError> {
    let expected = manifest
        .allowed_training_artifacts
        .iter()
        .map(artifact_key)
        .collect::<Result<HashSet<_>, _>>()?;
    let mut actual = HashSet::new();
    for record in lineage {
        record.validate_shape()?;
        let key = artifact_key(&record.training_artifact)?;
        if !actual.insert(key.clone()) {
            return Err(GovernanceError::DuplicateTrainingLineage(key));
        }
        for snapshot_id in &record.source_snapshot_ids {
            if !snapshots.contains_key(snapshot_id.as_str()) {
                return Err(GovernanceError::UnknownSnapshotReference(
                    snapshot_id.clone(),
                ));
            }
        }
    }
    if actual != expected {
        return Err(GovernanceError::TrainingLineageSetMismatch {
            missing: expected.difference(&actual).cloned().collect(),
            unexpected: actual.difference(&expected).cloned().collect(),
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_prospective(
    manifest: &BlindBenchmarkManifest,
    registration_text: &str,
    registration_artifact_sha256: &str,
    generator_artifact_sha256: &str,
    search_phase_view_sha256: &str,
    sealed_targets_sha256: &str,
    target_disclosures: &[TargetDisclosure],
) -> Result<(), GovernanceError> {
    let registration = parse_date("registration_date", registration_text)?;
    validate_sha256(registration_artifact_sha256)?;
    validate_sha256(generator_artifact_sha256)?;
    validate_sha256(search_phase_view_sha256)?;
    validate_sha256(sealed_targets_sha256)?;
    if !sealed_targets_sha256.eq_ignore_ascii_case(&manifest.sealed_targets_sha256) {
        return Err(GovernanceError::TargetCommitmentMismatch);
    }
    let actual_search_view_sha = sha256_hex(&serde_json::to_vec(&manifest.search_phase_view()?)?);
    if !search_phase_view_sha256.eq_ignore_ascii_case(&actual_search_view_sha) {
        return Err(GovernanceError::SearchPhaseViewDigestMismatch);
    }
    if target_disclosures.is_empty() {
        return Err(GovernanceError::NoTargetDisclosures);
    }
    let mut targets = HashSet::new();
    for target in target_disclosures {
        let disclosure = target.validate()?;
        if disclosure <= registration {
            return Err(GovernanceError::TargetNotAfterRegistration {
                target_id: target.target_id.clone(),
                public_date: target.public_date.clone(),
                registration: registration_text.to_string(),
            });
        }
        if !targets.insert(target.target_id.as_str()) {
            return Err(GovernanceError::DuplicateTargetDisclosure(
                target.target_id.clone(),
            ));
        }
    }
    Ok(())
}

fn validate_artifact(artifact: &BenchmarkArtifactRef) -> Result<(), GovernanceError> {
    nonempty("artifact_id", &artifact.artifact_id)?;
    validate_sha256(&artifact.sha256)
}

fn artifact_key(artifact: &BenchmarkArtifactRef) -> Result<String, GovernanceError> {
    validate_artifact(artifact)?;
    Ok(format!(
        "{}:{}",
        artifact.artifact_id,
        artifact.sha256.to_ascii_lowercase()
    ))
}

fn parse_date(field: &'static str, value: &str) -> Result<NaiveDate, GovernanceError> {
    NaiveDate::parse_from_str(value, "%Y-%m-%d").map_err(|_| GovernanceError::InvalidDate {
        field,
        value: value.to_string(),
    })
}

fn nonempty(field: &'static str, value: &str) -> Result<(), GovernanceError> {
    if value.trim().is_empty() {
        Err(GovernanceError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_sha256(value: &str) -> Result<(), GovernanceError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(GovernanceError::InvalidSha256)
    } else {
        Ok(())
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Temporal benchmark-governance validation failure.
#[derive(Debug, Error)]
pub enum GovernanceError {
    /// Underlying public benchmark contract failed validation.
    #[error(transparent)]
    Benchmark(#[from] symthaea_materials_benchmarks::BenchmarkError),
    /// JSON serialization failed while binding governance state.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Governance schema unsupported.
    #[error("unsupported temporal-governance schema: {0}")]
    UnsupportedSchema(u32),
    /// Required string empty.
    #[error("required governance field is empty: {0}")]
    EmptyField(&'static str),
    /// SHA-256 malformed.
    #[error("invalid governance SHA-256")]
    InvalidSha256,
    /// ISO date malformed.
    #[error("invalid ISO date in {field}: {value}")]
    InvalidDate {
        /// Field name.
        field: &'static str,
        /// Invalid date text.
        value: String,
    },
    /// Governance record belongs to another benchmark.
    #[error("temporal-governance benchmark identity mismatch")]
    BenchmarkIdentityMismatch,
    /// Governance record belongs to another public manifest.
    #[error("temporal-governance manifest digest mismatch")]
    ManifestDigestMismatch,
    /// Historical profile supplied no immutable corpus snapshot.
    #[error("retrospective profile contains no corpus snapshots")]
    NoCorpusSnapshots,
    /// Historical/prospective profile supplied no target disclosure event.
    #[error("temporal profile contains no target disclosures")]
    NoTargetDisclosures,
    /// Corpus snapshot was not actually available by the declared knowledge cutoff.
    #[error("snapshot {snapshot_id} released {release_date} after cutoff {cutoff}")]
    SnapshotAfterKnowledgeCutoff {
        /// Snapshot ID.
        snapshot_id: String,
        /// Public snapshot release date.
        release_date: String,
        /// Historical knowledge cutoff.
        cutoff: String,
    },
    /// Exact target artifact was incorrectly used as a corpus snapshot.
    #[error("sealed target artifact cannot be a training corpus snapshot")]
    TargetArtifactUsedAsCorpusSnapshot,
    /// Snapshot IDs must be unique.
    #[error("duplicate corpus snapshot ID: {0}")]
    DuplicateSnapshotId(String),
    /// Training lineage must cite at least one source snapshot.
    #[error("training artifact has no source snapshot lineage: {0}")]
    TrainingLineageHasNoSnapshot(String),
    /// Same snapshot repeated within one training lineage.
    #[error("duplicate source snapshot reference: {0}")]
    DuplicateSnapshotReference(String),
    /// Training lineage references a snapshot absent from the profile.
    #[error("unknown source snapshot reference: {0}")]
    UnknownSnapshotReference(String),
    /// Exact training artifact lineage repeated.
    #[error("duplicate training artifact lineage: {0}")]
    DuplicateTrainingLineage(String),
    /// Training lineage did not cover exactly the public manifest training inputs.
    #[error("training-lineage set mismatch; missing={missing:?}, unexpected={unexpected:?}")]
    TrainingLineageSetMismatch {
        /// Manifest training artifacts lacking lineage.
        missing: Vec<String>,
        /// Lineage artifacts not declared by the manifest.
        unexpected: Vec<String>,
    },
    /// Target became public on/before the historical knowledge cutoff.
    #[error("target {target_id} disclosed {public_date} on/before cutoff {cutoff}")]
    TargetNotAfterKnowledgeCutoff {
        /// Target identifier.
        target_id: String,
        /// Public date.
        public_date: String,
        /// Cutoff date.
        cutoff: String,
    },
    /// Target disclosure repeated.
    #[error("duplicate target disclosure: {0}")]
    DuplicateTargetDisclosure(String),
    /// Prospective public target commitment disagreed with the benchmark manifest.
    #[error("prospective target commitment does not match manifest")]
    TargetCommitmentMismatch,
    /// Prospective public search view disagreed with the registered digest.
    #[error("registered search-phase view digest does not match manifest")]
    SearchPhaseViewDigestMismatch,
    /// Target was already public when the supposedly prospective protocol registered.
    #[error("target {target_id} disclosed {public_date} on/before registration {registration}")]
    TargetNotAfterRegistration {
        /// Target identifier.
        target_id: String,
        /// Target disclosure date.
        public_date: String,
        /// Registration date.
        registration: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_materials_benchmarks::{
        BenchmarkMetricKind, BenchmarkMetricSpec, BenchmarkPublicationSource,
        BenchmarkSplitPolicy,
    };

    const A64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B64: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C64: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D64: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const E64: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn artifact(id: &str, sha: &str) -> BenchmarkArtifactRef {
        BenchmarkArtifactRef {
            artifact_id: id.to_string(),
            sha256: sha.to_string(),
        }
    }

    fn manifest() -> BlindBenchmarkManifest {
        BlindBenchmarkManifest {
            schema_version: 1,
            benchmark_id: "MAG-temporal-fixture".to_string(),
            search_space_artifact: artifact("search-space", A64),
            allowed_training_artifacts: vec![artifact("training", B64)],
            split_policy: BenchmarkSplitPolicy::FamilyAndStructure {
                method_id: "family+structure-v1".to_string(),
                split_artifact: artifact("split", C64),
            },
            sealed_targets_sha256: D64.to_string(),
            sources: vec![BenchmarkPublicationSource {
                source_id: "target-paper".to_string(),
                title: "Target paper".to_string(),
                doi: "10.0000/target".to_string(),
                date: "2026-02-09".to_string(),
            }],
            metrics: vec![BenchmarkMetricSpec {
                metric_id: "k1".to_string(),
                property_id: "magnetocrystalline_anisotropy_k1".to_string(),
                kind: BenchmarkMetricKind::ScalarRegression,
                unit: "MJ/m3".to_string(),
                condition_signature: "T=0K".to_string(),
            }],
        }
    }

    fn prepublication_snapshot() -> CorpusSnapshotRef {
        CorpusSnapshotRef {
            snapshot_id: "fixture-prepublication-v1".to_string(),
            provider: "fixture-provider".to_string(),
            version: "v1".to_string(),
            release_date: "2025-05-01".to_string(),
            artifact_sha256: E64.to_string(),
            license: "fixture-license".to_string(),
        }
    }

    fn retrospective(snapshot: CorpusSnapshotRef) -> TemporalBenchmarkGovernance {
        let manifest = manifest();
        let snapshot_id = snapshot.snapshot_id.clone();
        TemporalBenchmarkGovernance {
            schema_version: 1,
            benchmark_id: manifest.benchmark_id.clone(),
            manifest_sha256: manifest.manifest_sha256().unwrap(),
            claim_regime: BenchmarkClaimRegime::RetrospectivePrepublicationData {
                knowledge_cutoff_date: "2026-01-31".to_string(),
                corpus_snapshots: vec![snapshot],
                training_lineage: vec![TrainingArtifactLineage {
                    training_artifact: manifest.allowed_training_artifacts[0].clone(),
                    source_snapshot_ids: vec![snapshot_id],
                    derivation_artifact_sha256: A64.to_string(),
                }],
                target_disclosures: vec![TargetDisclosure {
                    target_id: "target-paper".to_string(),
                    public_date: "2026-02-09".to_string(),
                    source_id: "10.0000/target".to_string(),
                }],
            },
        }
    }

    #[test]
    fn structural_holdout_does_not_become_historical_discovery_evidence() {
        let manifest = manifest();
        let governance = TemporalBenchmarkGovernance {
            schema_version: 1,
            benchmark_id: manifest.benchmark_id.clone(),
            manifest_sha256: manifest.manifest_sha256().unwrap(),
            claim_regime: BenchmarkClaimRegime::StructuralGeneralization,
        };
        governance.validate_for(&manifest).unwrap();
        assert!(!governance.is_prospectively_registered());
        assert!(!governance.is_retrospective_prepublication_data());
    }

    #[test]
    fn prepublication_snapshot_supports_only_retrospective_data_claim() {
        let manifest = manifest();
        let governance = retrospective(prepublication_snapshot());
        governance.validate_for(&manifest).unwrap();
        assert!(governance.is_retrospective_prepublication_data());
        assert!(!governance.is_prospectively_registered());
    }

    #[test]
    fn post_cutoff_snapshot_cannot_support_retrospective_data_claim() {
        let manifest = manifest();
        let mut snapshot = prepublication_snapshot();
        snapshot.snapshot_id = "fixture-post-cutoff".to_string();
        snapshot.release_date = "2026-02-15".to_string();
        let governance = retrospective(snapshot);
        assert!(matches!(
            governance.validate_for(&manifest),
            Err(GovernanceError::SnapshotAfterKnowledgeCutoff { .. })
        ));
    }

    #[test]
    fn target_must_be_disclosed_after_historical_cutoff() {
        let manifest = manifest();
        let mut governance = retrospective(prepublication_snapshot());
        if let BenchmarkClaimRegime::RetrospectivePrepublicationData {
            target_disclosures, ..
        } = &mut governance.claim_regime
        {
            target_disclosures[0].public_date = "2026-01-15".to_string();
        }
        assert!(matches!(
            governance.validate_for(&manifest),
            Err(GovernanceError::TargetNotAfterKnowledgeCutoff { .. })
        ));
    }

    #[test]
    fn prospective_registration_requires_target_to_still_be_undisclosed() {
        let manifest = manifest();
        let search_view_sha =
            sha256_hex(&serde_json::to_vec(&manifest.search_phase_view().unwrap()).unwrap());
        let governance = TemporalBenchmarkGovernance {
            schema_version: 1,
            benchmark_id: manifest.benchmark_id.clone(),
            manifest_sha256: manifest.manifest_sha256().unwrap(),
            claim_regime: BenchmarkClaimRegime::ProspectiveRegistered {
                registration_date: "2026-02-01".to_string(),
                registration_artifact_sha256: A64.to_string(),
                generator_artifact_sha256: B64.to_string(),
                search_phase_view_sha256: search_view_sha,
                sealed_targets_sha256: manifest.sealed_targets_sha256.clone(),
                target_disclosures: vec![TargetDisclosure {
                    target_id: "target".to_string(),
                    public_date: "2026-02-09".to_string(),
                    source_id: "paper".to_string(),
                }],
            },
        };
        governance.validate_for(&manifest).unwrap();
        assert!(governance.is_prospectively_registered());
        assert!(!governance.is_retrospective_prepublication_data());
    }

    #[test]
    fn post_disclosure_registration_is_not_prospective() {
        let manifest = manifest();
        let search_view_sha =
            sha256_hex(&serde_json::to_vec(&manifest.search_phase_view().unwrap()).unwrap());
        let governance = TemporalBenchmarkGovernance {
            schema_version: 1,
            benchmark_id: manifest.benchmark_id.clone(),
            manifest_sha256: manifest.manifest_sha256().unwrap(),
            claim_regime: BenchmarkClaimRegime::ProspectiveRegistered {
                registration_date: "2026-02-10".to_string(),
                registration_artifact_sha256: A64.to_string(),
                generator_artifact_sha256: B64.to_string(),
                search_phase_view_sha256: search_view_sha,
                sealed_targets_sha256: manifest.sealed_targets_sha256.clone(),
                target_disclosures: vec![TargetDisclosure {
                    target_id: "target".to_string(),
                    public_date: "2026-02-09".to_string(),
                    source_id: "paper".to_string(),
                }],
            },
        };
        assert!(matches!(
            governance.validate_for(&manifest),
            Err(GovernanceError::TargetNotAfterRegistration { .. })
        ));
    }
}
