// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic, fail-closed indexer for a pinned checkout of openai/math.
//!
//! The tool indexes catalog structure and provenance only. It does not ingest
//! proof text into a model, certify manuscript claims, or infer that a paper's
//! headline theorem is formally verified.
//!
//! Usage:
//!   cargo run -p symthaea-proof-insights -- /path/to/openai/math 1581d4cbdd69a0faf34b2089614fe2a814316a6a ./proof-insight-source-index.json
//!   cargo run -p symthaea-proof-insights -- --verify-splits ./proof-insight-source-index.json ./data/benchmarks/proof-insight-transfer-v0/splits.json
//!   cargo run -p symthaea-proof-insights -- --verify-temporal-manifest /path/to/pinned-repository ./temporal-manifest.json

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, File},
    io::Read,
    path::{Component, Path, PathBuf},
    process::{Command, ExitCode},
};

use anyhow::{Context, Result, bail};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const EXPECTED_UPSTREAM_REPOSITORY: &str = "https://github.com/openai/math";
const EXPECTED_MANIFEST_VERSION: &str = "v0.4";
const FAMILY_SPLIT_SEED: &str = "pitb-v0-openai-math-2026-10-19";
const FAMILY_SPLIT_ALGORITHM: &str = "sha256(seed || NUL || family_id), sort by digest then family ID, floor 70% train and floor 85% train+validation";

#[derive(Debug, Deserialize)]
struct UpstreamManifest {
    version: String,
    #[serde(default)]
    project: Option<ProjectMetadata>,
    #[serde(default)]
    sources: Vec<FormalizationSource>,
    #[serde(default)]
    status: ManifestStatus,
}

#[derive(Debug, Deserialize)]
struct ProjectMetadata {
    name: Option<String>,
    license: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FormalizationSource {
    title: String,
    id: String,
    #[serde(default)]
    authors: Vec<String>,
    #[serde(rename = "type", default)]
    source_type: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ManifestStatus {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    main_results: Vec<MainResult>,
}

#[derive(Debug, Deserialize, Serialize)]
struct MainResult {
    #[serde(default)]
    comparator_config: Option<String>,
    #[serde(default)]
    declaration: Option<String>,
    #[serde(default)]
    file: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct SourceFingerprint {
    path: String,
    sha256: String,
    bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
struct CatalogueLinkedArtifact {
    path: String,
    file_exists: bool,
    sha256: Option<String>,
    bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct IndexedFormalizationSource {
    title: String,
    source_id: String,
    repository_relative_path: String,
    authors: Vec<String>,
    source_type: Option<String>,
    mapped_family_id: Option<String>,
    source_file_exists: bool,
    source_file_sha256: Option<String>,
    source_file_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct FamilyRecord {
    family_id: String,
    title: String,
    linked_paths: Vec<String>,
    directly_mapped_formalization_sources: Vec<String>,
    split: String,
}

#[derive(Debug, Serialize)]
struct CatalogueClaims {
    readme_claimed_manuscript_count: usize,
    readme_claimed_family_count: usize,
    parsed_family_header_count: usize,
}

#[derive(Debug, Serialize)]
struct FamilySplitPlan {
    seed: &'static str,
    algorithm: &'static str,
    train_percent: u8,
    validation_percent: u8,
    test_percent: u8,
    train_family_count: usize,
    validation_family_count: usize,
    test_family_count: usize,
    partition_digest_sha256: String,
}

#[derive(Debug, Serialize)]
struct IndexSummary {
    family_count: usize,
    formalization_source_count: usize,
    formalization_sources_mapped_to_family: usize,
    formalization_sources_unmapped: usize,
    formalization_source_files_missing: usize,
    catalogue_linked_artifact_count: usize,
    catalogue_linked_artifacts_present: usize,
    catalogue_linked_artifacts_missing: usize,
    missing_catalogue_linked_artifact_paths: Vec<String>,
    families_with_directly_mapped_formalization_sources: usize,
    main_result_declaration_count: usize,
    main_result_artifacts_missing: Vec<String>,
    unmapped_formalization_source_ids: Vec<String>,
    missing_formalization_source_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TemporalReplayManifest {
    schema_version: String,
    repository_url: String,
    repository_commit: String,
    timeline_path: String,
    timeline_sha256: String,
    lean_toolchain_path: String,
    lean_toolchain: String,
    mathlib_manifest_path: String,
    mathlib_manifest_sha256: String,
    mathlib_revision: String,
    events: Vec<TemporalTimelineEvent>,
    artifacts: Vec<TemporalEvidenceArtifact>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TemporalTimelineEvent {
    position: usize,
    target_id: String,
    theorem_id: String,
    target_state_commit: String,
    proof_completion_commit: String,
    statement_sha256: String,
    statement_artifact_id: String,
    completion_statement_artifact_id: String,
    proof_artifact_id: String,
    visible_artifact_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TemporalEvidenceArtifact {
    artifact_id: String,
    path: String,
    artifact_kind: String,
    content_sha256: String,
    content_start_line: usize,
    content_end_line: usize,
    source_file_sha256: String,
    source_commit: String,
    available_at_position: Option<usize>,
    target_position: Option<usize>,
    theorem_id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MaterializedVisibleArtifact {
    artifact_id: String,
    path: String,
    artifact_kind: String,
    content_sha256: String,
    source_file_sha256: String,
    source_commit: String,
    available_at_position: Option<usize>,
    target_position: Option<usize>,
    theorem_id: Option<String>,
    content: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TemporalTargetPayload {
    schema_version: String,
    repository_url: String,
    repository_commit: String,
    target_id: String,
    theorem_id: String,
    position: usize,
    target_state_commit: String,
    statement_artifact_id: String,
    statement_sha256: String,
    visible_artifacts: Vec<MaterializedVisibleArtifact>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TemporalTargetBundle {
    payload_sha256: String,
    payload: TemporalTargetPayload,
}

#[derive(Debug, Serialize)]
struct TemporalTimelineAudit {
    schema_version: &'static str,
    repository_url: String,
    repository_commit: String,
    timeline_path: String,
    timeline_sha256: String,
    repository_readme_sha256: String,
    readme_claimed_timeline_row_count: Option<usize>,
    timeline_row_count_matches_readme_claim: Option<bool>,
    lean_toolchain: String,
    lean_toolchain_sha256: String,
    mathlib_manifest_path: String,
    mathlib_manifest_sha256: String,
    mathlib_revision: String,
    total_timeline_rows: usize,
    unique_completion_commit_count: usize,
    source_path_present_at_completion_count: usize,
    source_path_missing_at_completion_count: usize,
    source_path_not_recorded_count: usize,
    verification_status_counts: BTreeMap<String, usize>,
    events: Vec<TemporalTimelineAuditEvent>,
}

#[derive(Debug, Serialize)]
struct TemporalDatasetJoinAudit {
    schema_version: &'static str,
    repository_url: String,
    repository_commit: String,
    timeline_path: String,
    timeline_sha256: String,
    dataset_path: String,
    dataset_sha256: String,
    total_timeline_rows: usize,
    unique_timeline_path_count: usize,
    total_dataset_records: usize,
    unique_dataset_id_count: usize,
    blank_dataset_line_count: usize,
    unique_path_join_count: usize,
    ambiguous_path_join_count: usize,
    unmatched_path_join_count: usize,
    invalid_dataset_path_count: usize,
    commit_before_ancestor_of_head_count: usize,
    commit_before_not_ancestor_of_head_count: usize,
    commit_before_missing_commit_object_count: usize,
    commit_before_invalid_sha_count: usize,
    commit_before_ancestor_of_completion_count: usize,
    statement_match_relation_counts: BTreeMap<String, usize>,
    events: Vec<TemporalDatasetJoinAuditEvent>,
}

#[derive(Debug, Serialize)]
struct TemporalDatasetJoinAuditEvent {
    dataset_position: usize,
    dataset_id: String,
    theorem_name: String,
    file_path: String,
    commit_before: String,
    statement_sha256: String,
    commit_before_object_exists: Option<bool>,
    path_join_status: String,
    candidate_timeline_positions: Vec<usize>,
    matched_completion_commit: Option<String>,
    commit_before_is_ancestor_of_head: Option<bool>,
    commit_before_is_ancestor_of_completion: Option<bool>,
    statement_match_at_commit_before: Option<bool>,
    statement_match_at_completion: Option<bool>,
    statement_match_relation: String,
}

#[derive(Debug, Serialize)]
struct TemporalTimelineAuditEvent {
    position: usize,
    event_key: String,
    timeline_id: Option<String>,
    lean_name: Option<String>,
    spec_theorem_path: Option<String>,
    function: Option<String>,
    verified_status: String,
    date_proven: String,
    completion_commit: String,
    commit_message: String,
    completion_commit_ancestor_verified: bool,
    source_file_status: String,
    source_file_sha256: Option<String>,
    source_file_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct SourceIndex {
    schema_version: &'static str,
    upstream_repository: &'static str,
    upstream_commit: String,
    upstream_manifest_version: String,
    upstream_project_name: Option<String>,
    upstream_license_label: Option<String>,
    upstream_status_scope: Option<String>,
    catalogue_claims: CatalogueClaims,
    source_fingerprints: Vec<SourceFingerprint>,
    catalogue_linked_artifacts: Vec<CatalogueLinkedArtifact>,
    family_split_plan: FamilySplitPlan,
    summary: IndexSummary,
    families: Vec<FamilyRecord>,
    formalization_sources: Vec<IndexedFormalizationSource>,
    main_results: Vec<MainResult>,
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() == 4 && args[1] == "--verify-splits" {
        return match verify_split_files(Path::new(&args[2]), Path::new(&args[3])) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("proof-insights split verification: {error:#}");
                ExitCode::FAILURE
            }
        };
    }
    if args.len() == 4 && args[1] == "--verify-temporal-manifest" {
        return match verify_temporal_manifest_file(Path::new(&args[2]), Path::new(&args[3])) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("proof-insights temporal manifest verification: {error:#}");
                ExitCode::FAILURE
            }
        };
    }
    if args.len() == 6 && args[1] == "--materialize-temporal-target" {
        return match materialize_temporal_target(
            Path::new(&args[2]),
            Path::new(&args[3]),
            &args[4],
            Path::new(&args[5]),
        ) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("proof-insights temporal target materialization: {error:#}");
                ExitCode::FAILURE
            }
        };
    }
    if args.len() == 3 && args[1] == "--verify-temporal-bundle" {
        return match verify_temporal_target_bundle(Path::new(&args[2])) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("proof-insights temporal target bundle verification: {error:#}");
                ExitCode::FAILURE
            }
        };
    }
    if args.len() == 5 && args[1] == "--audit-temporal-timeline" {
        return match audit_temporal_timeline(
            Path::new(&args[2]),
            Path::new(&args[3]),
            Path::new(&args[4]),
        ) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("proof-insights temporal timeline audit: {error:#}");
                ExitCode::FAILURE
            }
        };
    }
    if args.len() == 6 && args[1] == "--audit-temporal-dataset" {
        return match audit_temporal_dataset(
            Path::new(&args[2]),
            Path::new(&args[3]),
            Path::new(&args[4]),
            Path::new(&args[5]),
        ) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("proof-insights temporal dataset join audit: {error:#}");
                ExitCode::FAILURE
            }
        };
    }
    if let Err(error) = execute_from_args() {
        eprintln!("proof-insights indexer: {error:#}");
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn execute_from_args() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        bail!(
            "usage: {} <openai-math-checkout> <expected-40-char-commit> <output-json>",
            args.first()
                .map(String::as_str)
                .unwrap_or("symthaea-proof-insights")
        );
    }

    let root = PathBuf::from(&args[1])
        .canonicalize()
        .with_context(|| format!("cannot resolve source root {}", args[1]))?;
    let expected_commit = args[2].trim().to_ascii_lowercase();
    if expected_commit.len() != 40
        || !expected_commit
            .chars()
            .all(|character| character.is_ascii_hexdigit())
    {
        bail!("expected commit must be a 40-character hexadecimal Git commit SHA");
    }

    let actual_commit = git_output(&root, &["rev-parse", "HEAD"])?;
    if actual_commit != expected_commit {
        bail!(
            "source commit mismatch: expected {}, found {}",
            expected_commit,
            actual_commit
        );
    }

    let status = git_output(&root, &["status", "--porcelain"])?;
    if !status.is_empty() {
        bail!(
            "source checkout is dirty; commit or discard changes before indexing (git status --porcelain was non-empty)"
        );
    }

    let contents_path = root.join("CONTENTS.md");
    let readme_path = root.join("README.md");
    let manifest_path = root.join("lean/formalization.yaml");

    let contents = fs::read_to_string(&contents_path)
        .with_context(|| format!("cannot read {}", contents_path.display()))?;
    let upstream_readme = fs::read_to_string(&readme_path)
        .with_context(|| format!("cannot read {}", readme_path.display()))?;
    let manifest_text = fs::read_to_string(&manifest_path)
        .with_context(|| format!("cannot read {}", manifest_path.display()))?;
    let manifest: UpstreamManifest = serde_yaml::from_str(&manifest_text)
        .context("cannot parse lean/formalization.yaml as the expected YAML data model")?;

    if manifest.version != EXPECTED_MANIFEST_VERSION {
        bail!(
            "unsupported upstream manifest version {}; expected {}",
            manifest.version,
            EXPECTED_MANIFEST_VERSION
        );
    }
    if manifest.sources.is_empty() {
        bail!("manifest parsed with zero source records; refusing an empty-looking index");
    }

    let mut families = extract_families(&contents)?;
    if families.is_empty() {
        bail!("no family headers found in CONTENTS.md; refusing an empty-looking index");
    }
    let catalogue_claims = parse_catalogue_claims(&upstream_readme, families.len())?;
    let family_split_plan = assign_family_splits(&mut families)?;

    let source_path_to_family = map_source_paths_to_families(&families)?;

    let mut indexed_sources = Vec::with_capacity(manifest.sources.len());
    let mut unmapped_source_ids = Vec::new();
    let mut missing_source_ids = Vec::new();
    let mut source_file_fingerprints: BTreeMap<String, SourceFingerprint> = BTreeMap::new();
    for source in &manifest.sources {
        let repository_relative_path = normalize_lean_relative_path(&source.id)
            .with_context(|| format!("invalid formalization source path: {}", source.id))?;
        let mapped_family_id = source_path_to_family
            .get(&repository_relative_path)
            .cloned();
        if mapped_family_id.is_none() {
            unmapped_source_ids.push(source.id.clone());
        }

        let source_path = root.join(&repository_relative_path);
        let source_fingerprint = if source_path.is_file() {
            let fingerprint = fingerprint(&root, &repository_relative_path)?;
            source_file_fingerprints
                .entry(repository_relative_path.clone())
                .or_insert_with(|| fingerprint.clone());
            Some(fingerprint)
        } else {
            missing_source_ids.push(source.id.clone());
            None
        };

        indexed_sources.push(IndexedFormalizationSource {
            title: source.title.clone(),
            source_id: source.id.clone(),
            repository_relative_path,
            authors: source.authors.clone(),
            source_type: source.source_type.clone(),
            mapped_family_id,
            source_file_exists: source_fingerprint.is_some(),
            source_file_sha256: source_fingerprint.as_ref().map(|fp| fp.sha256.clone()),
            source_file_bytes: source_fingerprint.as_ref().map(|fp| fp.bytes),
        });
    }

    // Inventory every local artifact linked from the catalogue, not only the subset
    // already listed in the formalization manifest. Missing links are explicit rows;
    // present files are content-hashed and added to the same source fingerprint set.
    let catalogue_linked_artifacts =
        inventory_catalogue_links(&root, &families, &mut source_file_fingerprints)?;

    let family_source_titles: BTreeMap<String, Vec<String>> = {
        let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for source in &indexed_sources {
            if let Some(family_id) = &source.mapped_family_id {
                map.entry(family_id.clone())
                    .or_default()
                    .push(source.title.clone());
            }
        }
        map
    };
    for family in &mut families {
        family.directly_mapped_formalization_sources = family_source_titles
            .get(&family.family_id)
            .cloned()
            .unwrap_or_default();
    }

    let mut source_fingerprints = vec![
        fingerprint(&root, "README.md")?,
        fingerprint(&root, "CONTENTS.md")?,
        fingerprint(&root, "lean/formalization.yaml")?,
        fingerprint(&root, "lean/ComparatorChallenges/README.md")?,
    ];
    let mut missing_main_result_artifacts = BTreeSet::new();
    for result in &manifest.status.main_results {
        for relative_path in [result.file.as_deref(), result.comparator_config.as_deref()]
            .into_iter()
            .flatten()
        {
            let normalized_path = normalize_lean_relative_path(relative_path)?;
            if !root.join(&normalized_path).is_file() {
                missing_main_result_artifacts.insert(normalized_path);
                continue;
            }
            let artifact_fingerprint = fingerprint(&root, &normalized_path)?;
            source_file_fingerprints
                .entry(normalized_path)
                .or_insert(artifact_fingerprint);
        }
    }
    source_fingerprints.extend(source_file_fingerprints.into_values());
    source_fingerprints.sort_by(|left, right| left.path.cmp(&right.path));
    let mut unique_fingerprints: Vec<SourceFingerprint> =
        Vec::with_capacity(source_fingerprints.len());
    for artifact in source_fingerprints {
        match unique_fingerprints.last() {
            Some(previous) if previous.path == artifact.path => {
                if previous.sha256 != artifact.sha256 || previous.bytes != artifact.bytes {
                    bail!(
                        "conflicting fingerprints for the same repository-relative path {}",
                        artifact.path
                    );
                }
                continue;
            }
            _ => {}
        }
        unique_fingerprints.push(artifact);
    }
    let source_fingerprints = unique_fingerprints;

    let mapped_count = indexed_sources
        .iter()
        .filter(|source| source.mapped_family_id.is_some())
        .count();
    let summary = IndexSummary {
        family_count: families.len(),
        formalization_source_count: indexed_sources.len(),
        formalization_sources_mapped_to_family: mapped_count,
        formalization_sources_unmapped: indexed_sources.len() - mapped_count,
        formalization_source_files_missing: missing_source_ids.len(),
        catalogue_linked_artifact_count: catalogue_linked_artifacts.len(),
        catalogue_linked_artifacts_present: catalogue_linked_artifacts
            .iter()
            .filter(|artifact| artifact.file_exists)
            .count(),
        catalogue_linked_artifacts_missing: catalogue_linked_artifacts
            .iter()
            .filter(|artifact| !artifact.file_exists)
            .count(),
        missing_catalogue_linked_artifact_paths: catalogue_linked_artifacts
            .iter()
            .filter(|artifact| !artifact.file_exists)
            .map(|artifact| artifact.path.clone())
            .collect(),
        families_with_directly_mapped_formalization_sources: families
            .iter()
            .filter(|family| !family.directly_mapped_formalization_sources.is_empty())
            .count(),
        main_result_declaration_count: manifest.status.main_results.len(),
        main_result_artifacts_missing: missing_main_result_artifacts.into_iter().collect(),
        unmapped_formalization_source_ids: unmapped_source_ids,
        missing_formalization_source_ids: missing_source_ids,
    };

    let index = SourceIndex {
        schema_version: "symthaea.openai-math-source-index.v0",
        upstream_repository: EXPECTED_UPSTREAM_REPOSITORY,
        upstream_commit: actual_commit,
        upstream_manifest_version: manifest.version,
        upstream_project_name: manifest.project.as_ref().and_then(|p| p.name.clone()),
        upstream_license_label: manifest.project.as_ref().and_then(|p| p.license.clone()),
        upstream_status_scope: manifest.status.scope,
        catalogue_claims,
        source_fingerprints,
        catalogue_linked_artifacts,
        family_split_plan,
        summary,
        families,
        formalization_sources: indexed_sources,
        main_results: manifest.status.main_results,
    };

    let output = PathBuf::from(&args[3]);
    let encoded = serde_json::to_vec_pretty(&index).context("cannot serialize source index")?;
    fs::write(&output, &encoded)
        .with_context(|| format!("cannot write output JSON {}", output.display()))?;

    eprintln!("source index written: {}", output.display());
    eprintln!("upstream commit: {}", index.upstream_commit);
    eprintln!("families: {}", index.summary.family_count);
    eprintln!(
        "README catalogue claims: {} manuscripts / {} families; parsed family headers: {}",
        index.catalogue_claims.readme_claimed_manuscript_count,
        index.catalogue_claims.readme_claimed_family_count,
        index.catalogue_claims.parsed_family_header_count
    );
    eprintln!(
        "pre-screen family split: train {} / validation {} / test {} (seed: {})",
        index.family_split_plan.train_family_count,
        index.family_split_plan.validation_family_count,
        index.family_split_plan.test_family_count,
        index.family_split_plan.seed
    );
    eprintln!(
        "formalization sources: {} (mapped {}, unmapped {})",
        index.summary.formalization_source_count,
        index.summary.formalization_sources_mapped_to_family,
        index.summary.formalization_sources_unmapped
    );
    eprintln!(
        "formalization source files missing: {}",
        index.summary.formalization_source_files_missing
    );
    eprintln!(
        "catalogue-linked artifacts: {} present / {} missing ({} total)",
        index.summary.catalogue_linked_artifacts_present,
        index.summary.catalogue_linked_artifacts_missing,
        index.summary.catalogue_linked_artifact_count
    );
    eprintln!(
        "declared main-result entries: {}",
        index.summary.main_result_declaration_count
    );
    if !index.summary.unmapped_formalization_source_ids.is_empty() {
        eprintln!(
            "NOTE: {} formalization source paths did not match a family link; see JSON output",
            index.summary.formalization_sources_unmapped
        );
    }
    Ok(())
}

fn audit_temporal_dataset(
    repository_root: &Path,
    timeline_path: &Path,
    dataset_path: &Path,
    output_path: &Path,
) -> Result<()> {
    let root = repository_root.canonicalize().with_context(|| {
        format!(
            "cannot resolve temporal dataset repository {}",
            repository_root.display()
        )
    })?;
    let repository_commit = git_output(&root, &["rev-parse", "HEAD"])?;
    validate_git_sha(&repository_commit).context("repository HEAD must be a full commit SHA")?;
    let repository_url = git_output(&root, &["remote", "get-url", "origin"])?;
    if !repository_url.starts_with("https://") {
        bail!("temporal dataset repository origin must be an HTTPS URL");
    }
    if !git_output(&root, &["status", "--porcelain"])?.is_empty() {
        bail!("temporal dataset repository checkout must be clean before audit");
    }

    let timeline_relative = timeline_path
        .to_str()
        .context("timeline path must be UTF-8")?;
    let timeline_relative = normalize_root_relative_path(timeline_relative)?;
    let dataset_relative = dataset_path
        .to_str()
        .context("dataset path must be UTF-8")?;
    let dataset_relative = normalize_root_relative_path(dataset_relative)?;
    let timeline_bytes = git_show_file(&root, &repository_commit, &timeline_relative)?;
    let dataset_bytes = git_show_file(&root, &repository_commit, &dataset_relative)?;
    let timeline_sha256 = format!("{:x}", Sha256::digest(&timeline_bytes));
    let dataset_sha256 = format!("{:x}", Sha256::digest(&dataset_bytes));

    let mut timeline_reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .from_reader(timeline_bytes.as_slice());
    let timeline_headers = timeline_reader
        .headers()
        .context("read temporal dataset audit timeline headers")?
        .clone();
    let path_col = timeline_headers
        .iter()
        .position(|header| header == "spec_theorem")
        .context("timeline CSV lacks spec_theorem column")?;
    let commit_col = timeline_headers
        .iter()
        .position(|header| header == "commit_hash")
        .context("timeline CSV lacks commit_hash column")?;

    #[derive(Clone)]
    struct TimelinePathCandidate {
        position: usize,
        completion_commit: String,
    }

    let mut candidates_by_path: BTreeMap<String, Vec<TimelinePathCandidate>> = BTreeMap::new();
    let mut timeline_row_count = 0_usize;
    for (position, row_result) in timeline_reader.records().enumerate() {
        let row =
            row_result.with_context(|| format!("parse timeline join row {}", position + 2))?;
        timeline_row_count += 1;
        let path = row.get(path_col).unwrap_or("").trim();
        if path.is_empty() {
            continue;
        }
        let normalized = normalize_root_relative_path(path)
            .with_context(|| format!("unsafe timeline spec path on row {}", position + 2))?;
        if normalized != path {
            bail!(
                "timeline spec path is not canonical on row {}",
                position + 2
            );
        }
        let completion_commit = row
            .get(commit_col)
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        validate_git_sha(&completion_commit).with_context(|| {
            format!("invalid timeline completion commit on row {}", position + 2)
        })?;
        candidates_by_path
            .entry(normalized)
            .or_default()
            .push(TimelinePathCandidate {
                position,
                completion_commit,
            });
    }

    let unique_timeline_path_count = candidates_by_path.len();
    let mut dataset_text =
        std::str::from_utf8(&dataset_bytes).context("dataset JSONL is not UTF-8")?;
    if dataset_text.starts_with('\u{feff}') {
        dataset_text = &dataset_text['\u{feff}'.len_utf8()..];
    }

    let mut events = Vec::new();
    let mut dataset_ids = BTreeSet::new();
    let mut blank_dataset_line_count = 0_usize;
    let mut unique_path_join_count = 0_usize;
    let mut ambiguous_path_join_count = 0_usize;
    let mut unmatched_path_join_count = 0_usize;
    let mut invalid_dataset_path_count = 0_usize;
    let mut commit_before_ancestor_of_head_count = 0_usize;
    let mut commit_before_not_ancestor_of_head_count = 0_usize;
    let mut commit_before_missing_commit_object_count = 0_usize;
    let mut commit_before_invalid_sha_count = 0_usize;
    let mut commit_before_ancestor_of_completion_count = 0_usize;
    let mut statement_match_relation_counts = BTreeMap::new();
    let mut head_ancestry_cache: BTreeMap<String, bool> = BTreeMap::new();
    let mut commit_existence_cache: BTreeMap<String, bool> = BTreeMap::new();

    for (dataset_position, line) in dataset_text.lines().enumerate() {
        if line.trim().is_empty() {
            blank_dataset_line_count += 1;
            continue;
        }
        let record: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("parse dataset JSONL line {}", dataset_position + 1))?;
        let required_text = |field: &str| -> Result<String> {
            record
                .get(field)
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
                .with_context(|| {
                    format!(
                        "dataset row {} lacks string field {field}",
                        dataset_position + 1
                    )
                })
        };
        let dataset_id = required_text("id")?;
        let theorem_name = required_text("theorem_name")?;
        let raw_file_path = required_text("file_path")?;
        let commit_before = required_text("commit_before")?.to_ascii_lowercase();
        let formal_statement = required_text("formal_statement")?;
        let formal_statement_is_empty = formal_statement.trim().is_empty();
        let statement_sha256 = format!("{:x}", Sha256::digest(formal_statement.as_bytes()));
        dataset_ids.insert(dataset_id.clone());

        let commit_before_object_exists = if validate_git_sha(&commit_before).is_ok() {
            let exists = match commit_existence_cache.get(&commit_before) {
                Some(exists) => *exists,
                None => {
                    let exists = git_commit_exists(&root, &commit_before)?;
                    commit_existence_cache.insert(commit_before.clone(), exists);
                    exists
                }
            };
            Some(exists)
        } else {
            None
        };
        let commit_before_is_ancestor_of_head = match commit_before_object_exists {
            Some(true) => {
                let is_ancestor = match head_ancestry_cache.get(&commit_before) {
                    Some(is_ancestor) => *is_ancestor,
                    None => {
                        let is_ancestor =
                            git_is_ancestor(&root, &commit_before, &repository_commit)?;
                        head_ancestry_cache.insert(commit_before.clone(), is_ancestor);
                        is_ancestor
                    }
                };
                if is_ancestor {
                    commit_before_ancestor_of_head_count += 1;
                } else {
                    commit_before_not_ancestor_of_head_count += 1;
                }
                Some(is_ancestor)
            }
            Some(false) => {
                commit_before_missing_commit_object_count += 1;
                Some(false)
            }
            None => {
                commit_before_invalid_sha_count += 1;
                None
            }
        };

        let mut path_join_status = String::new();
        let mut candidate_timeline_positions = Vec::new();
        let mut matched_completion_commit = None;
        let mut commit_before_is_ancestor_of_completion = None;
        let mut statement_match_at_commit_before = None;
        let mut statement_match_at_completion = None;
        let mut statement_match_relation = String::new();

        let normalized_dataset_path = match normalize_root_relative_path(&raw_file_path) {
            Ok(normalized) if normalized == raw_file_path => Some(normalized),
            _ => None,
        };
        let candidate_matches = normalized_dataset_path
            .as_ref()
            .and_then(|path| candidates_by_path.get(path));
        if normalized_dataset_path.is_none() {
            invalid_dataset_path_count += 1;
            path_join_status = "invalid_path".to_string();
        } else {
            match candidate_matches {
                None => {
                    unmatched_path_join_count += 1;
                    path_join_status = "no_path_match".to_string();
                }
                Some(candidates) if candidates.is_empty() => {
                    unmatched_path_join_count += 1;
                    path_join_status = "no_path_match".to_string();
                }
                Some(candidates) if candidates.len() == 1 => {
                    unique_path_join_count += 1;
                    path_join_status = "unique_path_match".to_string();
                    let matched = &candidates[0];
                    candidate_timeline_positions.push(matched.position);
                    matched_completion_commit = Some(matched.completion_commit.clone());

                    if let Some(true) = commit_before_object_exists {
                        let before_is_completion_ancestor =
                            if commit_before_is_ancestor_of_head == Some(true) {
                                git_is_ancestor(&root, &commit_before, &matched.completion_commit)?
                            } else {
                                false
                            };
                        commit_before_is_ancestor_of_completion =
                            Some(before_is_completion_ancestor);
                        if before_is_completion_ancestor {
                            commit_before_ancestor_of_completion_count += 1;
                            if !formal_statement_is_empty
                                && git_path_exists_at_commit(
                                    &root,
                                    &commit_before,
                                    &raw_file_path,
                                )?
                            {
                                let before_source =
                                    git_show_file(&root, &commit_before, &raw_file_path)?;
                                let normalized_statement = normalize_whitespace(&formal_statement);
                                let normalized_before_source = normalize_whitespace(
                                    std::str::from_utf8(&before_source)
                                        .context("dataset pre-state Lean source is not UTF-8")?,
                                );
                                statement_match_at_commit_before =
                                    Some(normalized_before_source.contains(&normalized_statement));
                            }
                        } else {
                            statement_match_at_commit_before = None;
                        }
                    }

                    if !formal_statement_is_empty
                        && git_path_exists_at_commit(
                            &root,
                            &matched.completion_commit,
                            &raw_file_path,
                        )?
                    {
                        let completion_source =
                            git_show_file(&root, &matched.completion_commit, &raw_file_path)?;
                        let normalized_statement = normalize_whitespace(&formal_statement);
                        let normalized_completion_source = normalize_whitespace(
                            std::str::from_utf8(&completion_source)
                                .context("dataset completion Lean source is not UTF-8")?,
                        );
                        statement_match_at_completion =
                            Some(normalized_completion_source.contains(&normalized_statement));
                    }

                    statement_match_relation = if formal_statement_is_empty {
                        "empty_formal_statement".to_string()
                    } else {
                        match (
                            statement_match_at_commit_before,
                            statement_match_at_completion,
                        ) {
                            (Some(true), Some(true)) => "matches_before_and_completion",
                            (Some(true), Some(false)) => "matches_pre_state_only",
                            (Some(false), Some(true)) => "matches_completion_only",
                            (Some(false), Some(false)) => "matches_neither_state",
                            (None, Some(true)) => "completion_match_pre_state_unavailable",
                            (Some(true), None) => "pre_state_match_completion_unavailable",
                            (Some(false), None) => "pre_state_nonmatch_completion_unavailable",
                            (None, Some(false)) => "completion_nonmatch_pre_state_unavailable",
                            (None, None) => "statement_check_unavailable",
                        }
                        .to_string()
                    };
                }
                Some(candidates) => {
                    ambiguous_path_join_count += 1;
                    path_join_status = "ambiguous_path_match".to_string();
                    candidate_timeline_positions
                        .extend(candidates.iter().map(|candidate| candidate.position));
                    statement_match_relation = "not_checked_due_to_ambiguous_path".to_string();
                }
            }
        }
        if statement_match_relation.is_empty() {
            statement_match_relation = match path_join_status.as_str() {
                "invalid_path" => "not_checked_due_to_invalid_path",
                "no_path_match" => "not_checked_due_to_unmatched_path",
                other => other,
            }
            .to_string();
        }
        *statement_match_relation_counts
            .entry(statement_match_relation.clone())
            .or_insert(0) += 1;

        events.push(TemporalDatasetJoinAuditEvent {
            dataset_position,
            dataset_id,
            theorem_name,
            file_path: raw_file_path,
            commit_before,
            statement_sha256,
            commit_before_object_exists,
            path_join_status,
            candidate_timeline_positions,
            matched_completion_commit,
            commit_before_is_ancestor_of_head,
            commit_before_is_ancestor_of_completion,
            statement_match_at_commit_before,
            statement_match_at_completion,
            statement_match_relation,
        });
    }

    if events.is_empty() {
        bail!("temporal dataset audit found zero dataset records");
    }
    let status_count_total = unique_path_join_count
        + ambiguous_path_join_count
        + unmatched_path_join_count
        + invalid_dataset_path_count;
    if status_count_total != events.len() {
        bail!(
            "temporal dataset path-join accounting mismatch: {} statuses for {} rows",
            status_count_total,
            events.len()
        );
    }

    let audit = TemporalDatasetJoinAudit {
        schema_version: "pitb.temporal-dataset-join-audit.v0",
        repository_url,
        repository_commit,
        timeline_path: timeline_relative,
        timeline_sha256,
        dataset_path: dataset_relative,
        dataset_sha256,
        total_timeline_rows: timeline_row_count,
        unique_timeline_path_count,
        total_dataset_records: events.len(),
        unique_dataset_id_count: dataset_ids.len(),
        blank_dataset_line_count,
        unique_path_join_count,
        ambiguous_path_join_count,
        unmatched_path_join_count,
        invalid_dataset_path_count,
        commit_before_ancestor_of_head_count,
        commit_before_not_ancestor_of_head_count,
        commit_before_missing_commit_object_count,
        commit_before_invalid_sha_count,
        commit_before_ancestor_of_completion_count,
        statement_match_relation_counts,
        events,
    };

    let output_parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !output_parent.is_dir() {
        bail!(
            "temporal dataset audit output directory must already exist: {}",
            output_parent.display()
        );
    }
    let canonical_parent = output_parent.canonicalize().with_context(|| {
        format!(
            "canonicalize temporal dataset audit output directory {}",
            output_parent.display()
        )
    })?;
    let resolved_output = canonical_parent.join(
        output_path
            .file_name()
            .context("temporal dataset audit output must have a file name")?,
    );
    if resolved_output.starts_with(&root) {
        bail!("temporal dataset audit output must be outside the repository checkout");
    }
    if fs::symlink_metadata(&resolved_output)
        .is_ok_and(|metadata| metadata.file_type().is_symlink())
    {
        bail!("refusing to overwrite a symlink as temporal dataset audit output");
    }
    let canonical_report =
        serde_json::to_vec(&audit).context("serialize canonical temporal dataset audit")?;
    let report_sha256 = format!("{:x}", Sha256::digest(&canonical_report));
    let mut encoded =
        serde_json::to_vec_pretty(&audit).context("serialize temporal dataset audit")?;
    encoded.push(b'\n');
    fs::write(&resolved_output, encoded)
        .with_context(|| format!("write temporal dataset audit {}", resolved_output.display()))?;
    println!(
        "PITB_TEMPORAL_DATASET_AUDIT_PASS: rows={} unique_path_join={} ambiguous_path_join={} unmatched_path_join={} invalid_path={} statements={} report_sha256={}",
        audit.total_dataset_records,
        audit.unique_path_join_count,
        audit.ambiguous_path_join_count,
        audit.unmatched_path_join_count,
        audit.invalid_dataset_path_count,
        audit
            .statement_match_relation_counts
            .values()
            .sum::<usize>(),
        report_sha256,
    );
    Ok(())
}

fn audit_temporal_timeline(
    repository_root: &Path,
    timeline_path: &Path,
    output_path: &Path,
) -> Result<()> {
    let root = repository_root.canonicalize().with_context(|| {
        format!(
            "cannot resolve timeline repository {}",
            repository_root.display()
        )
    })?;
    let repository_commit = git_output(&root, &["rev-parse", "HEAD"])?;
    validate_git_sha(&repository_commit)
        .context("repository HEAD must be a full Git commit SHA")?;
    let repository_url = git_output(&root, &["remote", "get-url", "origin"])?;
    if !repository_url.starts_with("https://") {
        bail!("timeline repository origin must be an explicit HTTPS URL");
    }
    if !git_output(&root, &["status", "--porcelain"])?.is_empty() {
        bail!("timeline repository checkout must be clean before audit");
    }

    let timeline_path = timeline_path
        .to_str()
        .context("timeline path must be UTF-8")?;
    let normalized_timeline_path = normalize_root_relative_path(timeline_path)?;
    if normalized_timeline_path != timeline_path {
        bail!("timeline path must be canonical and repository-relative");
    }
    let timeline_bytes = git_show_file(&root, &repository_commit, timeline_path)?;
    let timeline_sha256 = format!("{:x}", Sha256::digest(&timeline_bytes));
    let readme_bytes = git_show_file(&root, &repository_commit, "README.md")?;
    let readme_sha256 = format!("{:x}", Sha256::digest(&readme_bytes));
    let readme_text =
        std::str::from_utf8(&readme_bytes).context("pinned repository README must be UTF-8")?;
    let readme_count_pattern = Regex::new(r"(?i)proof_timeline\.csv[^\n]*?([0-9,]+)\s+theorems")
        .context("compile timeline README count pattern")?;
    let readme_claimed_timeline_row_count = readme_count_pattern
        .captures(readme_text)
        .map(|captures| {
            captures[1]
                .replace(',', "")
                .parse::<usize>()
                .context("README timeline theorem count is not an integer")
        })
        .transpose()?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .from_reader(timeline_bytes.as_slice());
    let headers = reader
        .headers()
        .context("read timeline CSV headers")?
        .clone();
    for required in [
        "id",
        "lean_name",
        "spec_theorem",
        "function",
        "verified",
        "date_proven",
        "commit_hash",
        "commit_message",
    ] {
        if headers.iter().filter(|header| *header == required).count() != 1 {
            bail!("timeline CSV must contain exactly one {required:?} column");
        }
    }
    let column = |name: &str| {
        headers
            .iter()
            .position(|header| header == name)
            .with_context(|| format!("timeline CSV lacks column {name}"))
    };
    let id_col = column("id")?;
    let lean_name_col = column("lean_name")?;
    let theorem_path_col = column("spec_theorem")?;
    let function_col = column("function")?;
    let verified_col = column("verified")?;
    let date_col = column("date_proven")?;
    let commit_col = column("commit_hash")?;
    let message_col = column("commit_message")?;

    let toolchain_bytes = git_show_file(&root, &repository_commit, "lean-toolchain")?;
    let lean_toolchain = std::str::from_utf8(&toolchain_bytes)
        .context("pinned Lean toolchain file is not UTF-8")?
        .trim()
        .to_owned();
    if lean_toolchain.is_empty() || lean_toolchain.eq_ignore_ascii_case("latest") {
        bail!("timeline repository must pin a concrete Lean toolchain");
    }
    let mathlib_manifest_path = "lake-manifest.json";
    let mathlib_manifest_bytes = git_show_file(&root, &repository_commit, mathlib_manifest_path)?;
    let mathlib_manifest: serde_json::Value =
        serde_json::from_slice(&mathlib_manifest_bytes).context("parse pinned Lake manifest")?;
    let mathlib_revision = find_mathlib_revision(&mathlib_manifest)
        .context("Lake manifest contains no identifiable Mathlib revision")?;
    validate_git_sha(&mathlib_revision)
        .context("Lake manifest Mathlib revision is not a full SHA")?;

    let mut events = Vec::new();
    let mut timeline_ids = BTreeSet::new();
    let mut verified_commit_cache: BTreeMap<String, bool> = BTreeMap::new();
    let mut unique_commits = BTreeSet::new();
    let mut source_path_present_at_completion_count = 0_usize;
    let mut source_path_missing_at_completion_count = 0_usize;
    let mut source_path_not_recorded_count = 0_usize;
    let mut verification_status_counts = BTreeMap::new();
    let mut previous_completion_commit: Option<String> = None;

    for (zero_based_position, row_result) in reader.records().enumerate() {
        let row = row_result
            .with_context(|| format!("parse timeline CSV row {}", zero_based_position + 2))?;
        let get = |index: usize| row.get(index).unwrap_or("").trim();
        let raw_id = get(id_col);
        let event_key = if raw_id.is_empty() {
            format!("row-{zero_based_position:04}")
        } else {
            if !timeline_ids.insert(raw_id.to_owned()) {
                bail!("duplicate non-empty timeline ID {raw_id}");
            }
            raw_id.to_owned()
        };
        let completion_commit = get(commit_col).to_ascii_lowercase();
        validate_git_sha(&completion_commit).with_context(|| {
            format!(
                "timeline row {} has an invalid completion commit",
                zero_based_position + 2
            )
        })?;
        let completion_commit_ancestor_verified =
            match verified_commit_cache.get(&completion_commit) {
                Some(verified) => *verified,
                None => {
                    let verified = git_is_ancestor(&root, &completion_commit, &repository_commit)
                        .with_context(|| {
                        format!("verify completion commit for timeline row {event_key}")
                    })?;
                    verified_commit_cache.insert(completion_commit.clone(), verified);
                    verified
                }
            };
        if !completion_commit_ancestor_verified {
            bail!(
                "completion commit {} for timeline row {} is not an ancestor of pinned HEAD",
                completion_commit,
                event_key
            );
        }
        if let Some(previous) = previous_completion_commit.as_deref()
            && !git_is_ancestor(&root, previous, &completion_commit)?
        {
            bail!(
                "timeline order violation: completion commit for {} does not descend from the previous timeline commit",
                event_key
            );
        }
        previous_completion_commit = Some(completion_commit.clone());
        unique_commits.insert(completion_commit.clone());

        let verified_status = get(verified_col).to_owned();
        if verified_status.is_empty() {
            bail!(
                "timeline row {} has an empty verified/status field",
                event_key
            );
        }
        *verification_status_counts
            .entry(verified_status.clone())
            .or_insert(0) += 1;

        let raw_path = get(theorem_path_col);
        let (spec_theorem_path, source_file_status, source_file_sha256, source_file_bytes) =
            if raw_path.is_empty() {
                source_path_not_recorded_count += 1;
                (None, "not_recorded".to_string(), None, None)
            } else {
                let normalized = normalize_root_relative_path(raw_path).with_context(|| {
                    format!("unsafe spec theorem path on timeline row {event_key}")
                })?;
                if normalized != raw_path {
                    bail!(
                        "spec theorem path on timeline row {} is not canonical: {}",
                        event_key,
                        raw_path
                    );
                }
                if git_path_exists_at_commit(&root, &completion_commit, &normalized)? {
                    let bytes = git_show_file(&root, &completion_commit, &normalized)
                        .with_context(|| {
                            format!(
                                "read source file {} at verified completion commit {}",
                                normalized, completion_commit
                            )
                        })?;
                    source_path_present_at_completion_count += 1;
                    (
                        Some(normalized),
                        "present_at_completion".to_string(),
                        Some(format!("{:x}", Sha256::digest(&bytes))),
                        Some(bytes.len() as u64),
                    )
                } else {
                    source_path_missing_at_completion_count += 1;
                    (
                        Some(normalized),
                        "missing_at_completion".to_string(),
                        None,
                        None,
                    )
                }
            };

        events.push(TemporalTimelineAuditEvent {
            position: zero_based_position,
            event_key,
            timeline_id: (!raw_id.is_empty()).then(|| raw_id.to_owned()),
            lean_name: (!get(lean_name_col).is_empty()).then(|| get(lean_name_col).to_owned()),
            spec_theorem_path,
            function: (!get(function_col).is_empty()).then(|| get(function_col).to_owned()),
            verified_status,
            date_proven: get(date_col).to_owned(),
            completion_commit,
            commit_message: get(message_col).to_owned(),
            completion_commit_ancestor_verified,
            source_file_status,
            source_file_sha256,
            source_file_bytes,
        });
    }

    if events.is_empty() {
        bail!("timeline audit found zero rows; refusing an empty inventory");
    }
    let audit = TemporalTimelineAudit {
        schema_version: "pitb.temporal-timeline-audit.v0",
        repository_url,
        repository_commit,
        timeline_path: timeline_path.to_owned(),
        timeline_sha256,
        repository_readme_sha256: readme_sha256,
        timeline_row_count_matches_readme_claim: readme_claimed_timeline_row_count
            .map(|claimed| claimed == events.len()),
        readme_claimed_timeline_row_count,
        lean_toolchain,
        lean_toolchain_sha256: format!("{:x}", Sha256::digest(&toolchain_bytes)),
        mathlib_manifest_path: mathlib_manifest_path.to_owned(),
        mathlib_manifest_sha256: format!("{:x}", Sha256::digest(&mathlib_manifest_bytes)),
        mathlib_revision,
        total_timeline_rows: events.len(),
        unique_completion_commit_count: unique_commits.len(),
        source_path_present_at_completion_count,
        source_path_missing_at_completion_count,
        source_path_not_recorded_count,
        verification_status_counts,
        events,
    };

    let output_parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !output_parent.is_dir() {
        bail!(
            "timeline audit output directory must already exist: {}",
            output_parent.display()
        );
    }
    let canonical_parent = output_parent
        .canonicalize()
        .with_context(|| format!("canonicalize output directory {}", output_parent.display()))?;
    let resolved_output = canonical_parent.join(
        output_path
            .file_name()
            .context("timeline audit output path must have a file name")?,
    );
    if resolved_output.starts_with(&root) {
        bail!("timeline audit output must be outside the pinned repository checkout");
    }
    if fs::symlink_metadata(&resolved_output)
        .is_ok_and(|metadata| metadata.file_type().is_symlink())
    {
        bail!("refusing to overwrite a symlink as temporal timeline audit output");
    }

    let mut serialized =
        serde_json::to_vec_pretty(&audit).context("serialize temporal timeline audit")?;
    serialized.push(b'\n');
    fs::write(&resolved_output, &serialized).with_context(|| {
        format!(
            "write temporal timeline audit {}",
            resolved_output.display()
        )
    })?;
    println!(
        "PITB_TEMPORAL_TIMELINE_AUDIT_PASS: rows={} unique_completion_commits={} source_present={} source_missing={} source_not_recorded={} timeline_sha256={} output={}",
        audit.total_timeline_rows,
        audit.unique_completion_commit_count,
        audit.source_path_present_at_completion_count,
        audit.source_path_missing_at_completion_count,
        audit.source_path_not_recorded_count,
        audit.timeline_sha256,
        resolved_output.display()
    );
    Ok(())
}

fn verify_temporal_manifest_file(repository_root: &Path, manifest_path: &Path) -> Result<()> {
    let root = repository_root.canonicalize().with_context(|| {
        format!(
            "cannot resolve temporal repository {}",
            repository_root.display()
        )
    })?;
    let bytes = fs::read(manifest_path)
        .with_context(|| format!("cannot read temporal manifest {}", manifest_path.display()))?;
    let manifest: TemporalReplayManifest =
        serde_json::from_slice(&bytes).context("parse temporal replay manifest JSON")?;
    validate_temporal_manifest(&manifest)?;

    let actual_head = git_output(&root, &["rev-parse", "HEAD"])?;
    if actual_head != manifest.repository_commit {
        bail!(
            "temporal repository HEAD mismatch: manifest pins {}, checkout is {}",
            manifest.repository_commit,
            actual_head
        );
    }
    let actual_origin = git_output(&root, &["remote", "get-url", "origin"])?;
    if normalize_repository_url(&actual_origin)
        != normalize_repository_url(&manifest.repository_url)
    {
        bail!(
            "temporal repository origin mismatch: manifest pins {}, checkout origin is {}",
            manifest.repository_url,
            actual_origin
        );
    }
    let status = git_output(&root, &["status", "--porcelain"])?;
    if !status.is_empty() {
        bail!("temporal repository checkout must be clean before validating replay evidence");
    }
    let timeline = git_show_file(&root, &manifest.repository_commit, &manifest.timeline_path)?;
    if format!("{:x}", Sha256::digest(&timeline)) != manifest.timeline_sha256 {
        bail!(
            "timeline digest does not match {} at the pinned repository commit",
            manifest.timeline_path
        );
    }

    let toolchain_bytes = git_show_file(
        &root,
        &manifest.repository_commit,
        &manifest.lean_toolchain_path,
    )?;
    let toolchain_at_commit = std::str::from_utf8(&toolchain_bytes)
        .context("pinned Lean toolchain file is not UTF-8")?
        .trim();
    if toolchain_at_commit != manifest.lean_toolchain {
        bail!(
            "Lean toolchain file {} declares {}, but manifest pins {}",
            manifest.lean_toolchain_path,
            toolchain_at_commit,
            manifest.lean_toolchain
        );
    }

    let mathlib_manifest = git_show_file(
        &root,
        &manifest.repository_commit,
        &manifest.mathlib_manifest_path,
    )?;
    if format!("{:x}", Sha256::digest(&mathlib_manifest)) != manifest.mathlib_manifest_sha256 {
        bail!(
            "Mathlib manifest digest does not match {} at the pinned repository commit",
            manifest.mathlib_manifest_path
        );
    }
    let parsed_mathlib_manifest: serde_json::Value = serde_json::from_slice(&mathlib_manifest)
        .context("parse pinned Mathlib dependency manifest")?;
    let recorded_mathlib_revision = find_mathlib_revision(&parsed_mathlib_manifest)
        .context("pinned dependency manifest contains no identifiable Mathlib package revision")?;
    if !recorded_mathlib_revision.eq_ignore_ascii_case(&manifest.mathlib_revision) {
        bail!(
            "Mathlib revision mismatch: manifest pins {}, dependency file declares {}",
            manifest.mathlib_revision,
            recorded_mathlib_revision
        );
    }

    let mut previous_completion_commit: Option<&str> = None;
    for event in &manifest.events {
        if !git_is_ancestor(
            &root,
            &event.target_state_commit,
            &manifest.repository_commit,
        )? {
            bail!(
                "target state commit {} for {} is not an ancestor of the pinned repository commit",
                event.target_state_commit,
                event.target_id
            );
        }
        if !git_is_ancestor(
            &root,
            &event.proof_completion_commit,
            &manifest.repository_commit,
        )? {
            bail!(
                "proof completion commit {} for {} is not an ancestor of the pinned repository commit",
                event.proof_completion_commit,
                event.target_id
            );
        }
        if !git_is_ancestor(
            &root,
            &event.target_state_commit,
            &event.proof_completion_commit,
        )? {
            bail!(
                "target state commit must precede or equal proof completion for {}",
                event.target_id
            );
        }
        if let Some(previous) = previous_completion_commit
            && !git_is_ancestor(&root, previous, &event.target_state_commit)?
        {
            bail!(
                "target {} is evaluated before the previous timeline proof was available",
                event.target_id
            );
        }

        for artifact_id in &event.visible_artifact_ids {
            let artifact = manifest
                .artifacts
                .iter()
                .find(|artifact| artifact.artifact_id == *artifact_id)
                .context("visible artifact disappeared after temporal manifest validation")?;
            if !git_is_ancestor(&root, &artifact.source_commit, &event.target_state_commit)? {
                bail!(
                    "future-commit leakage: artifact {} from {} was not in target {}'s historical state",
                    artifact.artifact_id,
                    artifact.source_commit,
                    event.target_id
                );
            }
        }
        previous_completion_commit = Some(event.proof_completion_commit.as_str());
    }

    let mut file_cache: BTreeMap<(String, String), Vec<u8>> = BTreeMap::new();
    for artifact in &manifest.artifacts {
        if !git_is_ancestor(&root, &artifact.source_commit, &manifest.repository_commit)? {
            bail!(
                "artifact {} references a source commit outside the pinned repository history",
                artifact.artifact_id
            );
        }
        let cache_key = (artifact.source_commit.clone(), artifact.path.clone());
        if !file_cache.contains_key(&cache_key) {
            let source = git_show_file(&root, &artifact.source_commit, &artifact.path)?;
            let file_digest = format!("{:x}", Sha256::digest(&source));
            if file_digest != artifact.source_file_sha256 {
                bail!(
                    "source-file digest mismatch for artifact {} at {}:{}",
                    artifact.artifact_id,
                    artifact.source_commit,
                    artifact.path
                );
            }
            file_cache.insert(cache_key.clone(), source);
        }
        let source = file_cache
            .get(&cache_key)
            .context("source-file cache entry missing after insertion")?;
        let span = exact_line_span(
            source,
            artifact.content_start_line,
            artifact.content_end_line,
        )
        .with_context(|| {
            format!(
                "invalid source locator for artifact {}",
                artifact.artifact_id
            )
        })?;
        let span_digest = format!("{:x}", Sha256::digest(span));
        if span_digest != artifact.content_sha256 {
            bail!(
                "artifact content digest mismatch for {} ({}:{}-{})",
                artifact.artifact_id,
                artifact.path,
                artifact.content_start_line,
                artifact.content_end_line
            );
        }
    }

    println!(
        "PITB_TEMPORAL_MANIFEST_PASS: events={} artifacts={} repository_commit={}",
        manifest.events.len(),
        manifest.artifacts.len(),
        manifest.repository_commit
    );
    Ok(())
}

fn materialize_temporal_target(
    repository_root: &Path,
    manifest_path: &Path,
    target_id: &str,
    output_path: &Path,
) -> Result<()> {
    // Validate every timeline edge and artifact first. No output is written if any
    // historical visibility, source digest, or hidden-proof control fails.
    verify_temporal_manifest_file(repository_root, manifest_path)?;

    let root = repository_root.canonicalize().with_context(|| {
        format!(
            "cannot resolve temporal repository {}",
            repository_root.display()
        )
    })?;
    let manifest_bytes = fs::read(manifest_path)
        .with_context(|| format!("cannot read temporal manifest {}", manifest_path.display()))?;
    let manifest: TemporalReplayManifest =
        serde_json::from_slice(&manifest_bytes).context("parse temporal replay manifest JSON")?;
    let event = manifest
        .events
        .iter()
        .find(|event| event.target_id == target_id)
        .with_context(|| format!("target {target_id} is not in the frozen temporal manifest"))?;

    let mut visible_artifacts = Vec::with_capacity(event.visible_artifact_ids.len());
    for artifact_id in &event.visible_artifact_ids {
        let artifact = manifest
            .artifacts
            .iter()
            .find(|artifact| artifact.artifact_id == *artifact_id)
            .with_context(|| {
                format!("visible artifact {artifact_id} disappeared after validation")
            })?;
        if artifact.artifact_kind == "completion_statement"
            || artifact.artifact_id == event.proof_artifact_id
            || artifact.target_position == Some(event.position)
                && artifact.artifact_kind == "proof_body"
        {
            bail!(
                "refusing to materialize evaluator-only artifact {} for target {}",
                artifact.artifact_id,
                target_id
            );
        }
        let source =
            git_show_file(&root, &artifact.source_commit, &artifact.path).with_context(|| {
                format!(
                    "read visible artifact {} from its historical commit",
                    artifact.artifact_id
                )
            })?;
        let source_digest = format!("{:x}", Sha256::digest(&source));
        if source_digest != artifact.source_file_sha256 {
            bail!(
                "visible source file digest changed for artifact {}",
                artifact.artifact_id
            );
        }
        let span = exact_line_span(
            &source,
            artifact.content_start_line,
            artifact.content_end_line,
        )?;
        let span_digest = format!("{:x}", Sha256::digest(span));
        if span_digest != artifact.content_sha256 {
            bail!(
                "visible artifact digest changed for {}",
                artifact.artifact_id
            );
        }
        let content = std::str::from_utf8(span)
            .with_context(|| format!("visible artifact {} is not UTF-8", artifact.artifact_id))?
            .to_owned();
        visible_artifacts.push(MaterializedVisibleArtifact {
            artifact_id: artifact.artifact_id.clone(),
            path: artifact.path.clone(),
            artifact_kind: artifact.artifact_kind.clone(),
            content_sha256: artifact.content_sha256.clone(),
            source_file_sha256: artifact.source_file_sha256.clone(),
            source_commit: artifact.source_commit.clone(),
            available_at_position: artifact.available_at_position,
            target_position: artifact.target_position,
            theorem_id: artifact.theorem_id.clone(),
            content,
        });
    }

    if !visible_artifacts
        .iter()
        .any(|artifact| artifact.artifact_id == event.statement_artifact_id)
    {
        bail!("materialized solver context lacks the exact target statement");
    }
    if visible_artifacts.iter().any(|artifact| {
        artifact.artifact_id == event.proof_artifact_id
            || artifact.artifact_kind == "completion_statement"
            || artifact.target_position == Some(event.position)
                && artifact.artifact_kind == "proof_body"
    }) {
        bail!("materialized solver context contains target-specific evaluator evidence");
    }

    let payload = TemporalTargetPayload {
        schema_version: "pitb.temporal-target-bundle.v0".to_string(),
        repository_url: manifest.repository_url,
        repository_commit: manifest.repository_commit,
        target_id: event.target_id.clone(),
        theorem_id: event.theorem_id.clone(),
        position: event.position,
        target_state_commit: event.target_state_commit.clone(),
        statement_artifact_id: event.statement_artifact_id.clone(),
        statement_sha256: event.statement_sha256.clone(),
        visible_artifacts,
    };
    let canonical_payload =
        serde_json::to_vec(&payload).context("serialize temporal target payload")?;
    let bundle = TemporalTargetBundle {
        payload_sha256: format!("{:x}", Sha256::digest(&canonical_payload)),
        payload,
    };
    let mut serialized =
        serde_json::to_vec_pretty(&bundle).context("serialize temporal target bundle")?;
    serialized.push(b'\n');

    let output_parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !output_parent.is_dir() {
        bail!(
            "target bundle output directory must already exist: {}",
            output_parent.display()
        );
    }
    let canonical_parent = output_parent.canonicalize().with_context(|| {
        format!(
            "canonicalize target bundle directory {}",
            output_parent.display()
        )
    })?;
    let resolved_output = canonical_parent.join(
        output_path
            .file_name()
            .context("target bundle output path must have a file name")?,
    );
    if resolved_output.starts_with(&root) {
        bail!("target bundle output must be outside the pinned repository checkout");
    }
    if fs::symlink_metadata(&resolved_output)
        .is_ok_and(|metadata| metadata.file_type().is_symlink())
    {
        bail!("refusing to overwrite a symlink as target bundle output");
    }
    fs::write(&resolved_output, &serialized).with_context(|| {
        format!(
            "write target-specific context bundle {}",
            resolved_output.display()
        )
    })?;
    println!(
        "PITB_TEMPORAL_TARGET_BUNDLE_PASS: target={} visible_artifacts={} payload_sha256={} output={}",
        target_id,
        bundle.payload.visible_artifacts.len(),
        bundle.payload_sha256,
        resolved_output.display()
    );
    Ok(())
}

fn verify_temporal_target_bundle(bundle_path: &Path) -> Result<()> {
    let bytes = fs::read(bundle_path).with_context(|| {
        format!(
            "cannot read temporal target bundle {}",
            bundle_path.display()
        )
    })?;
    let bundle: TemporalTargetBundle =
        serde_json::from_slice(&bytes).context("parse temporal target bundle JSON")?;
    let payload = &bundle.payload;
    if payload.schema_version != "pitb.temporal-target-bundle.v0" {
        bail!(
            "unsupported temporal target bundle schema {}",
            payload.schema_version
        );
    }
    if !payload.repository_url.starts_with("https://") {
        bail!("temporal target bundle must pin an HTTPS repository URL");
    }
    validate_git_sha(&payload.repository_commit).context("invalid bundle repository commit")?;
    validate_git_sha(&payload.target_state_commit).context("invalid target-state commit")?;
    validate_sha256_hex(&payload.statement_sha256).context("invalid target statement digest")?;
    if payload.target_id.trim().is_empty() || payload.theorem_id.trim().is_empty() {
        bail!("temporal target bundle contains invalid target metadata");
    }
    if payload.visible_artifacts.is_empty() {
        bail!("temporal target bundle cannot contain an empty visible context");
    }

    let canonical_payload =
        serde_json::to_vec(payload).context("serialize canonical temporal target payload")?;
    let expected_bundle_digest = format!("{:x}", Sha256::digest(&canonical_payload));
    if bundle.payload_sha256 != expected_bundle_digest {
        bail!(
            "target bundle payload digest mismatch: expected {}, found {}",
            expected_bundle_digest,
            bundle.payload_sha256
        );
    }

    let mut seen_ids = BTreeSet::new();
    let mut statement_found = false;
    for artifact in &payload.visible_artifacts {
        if artifact.artifact_id.trim().is_empty() || !seen_ids.insert(artifact.artifact_id.as_str())
        {
            bail!("bundle contains an empty or duplicate visible artifact ID");
        }
        let normalized_path = normalize_root_relative_path(&artifact.path)?;
        if normalized_path != artifact.path {
            bail!("bundle artifact path is not canonical: {}", artifact.path);
        }
        validate_sha256_hex(&artifact.content_sha256).with_context(|| {
            format!("invalid bundle content digest for {}", artifact.artifact_id)
        })?;
        validate_sha256_hex(&artifact.source_file_sha256).with_context(|| {
            format!(
                "invalid bundle source-file digest for {}",
                artifact.artifact_id
            )
        })?;
        validate_git_sha(&artifact.source_commit).with_context(|| {
            format!("invalid bundle source commit for {}", artifact.artifact_id)
        })?;
        let observed_content_digest = format!("{:x}", Sha256::digest(artifact.content.as_bytes()));
        if observed_content_digest != artifact.content_sha256 {
            bail!(
                "bundle artifact content does not match digest for {}",
                artifact.artifact_id
            );
        }

        if artifact.artifact_id == payload.statement_artifact_id {
            if artifact.artifact_kind != "theorem_statement"
                || artifact.theorem_id.as_deref() != Some(payload.theorem_id.as_str())
                || artifact.content_sha256 != payload.statement_sha256
                || artifact.target_position.is_some()
                || artifact
                    .available_at_position
                    .is_none_or(|position| position > payload.position)
            {
                bail!(
                    "bundle's exact target-statement artifact violates the statement/visibility contract"
                );
            }
            statement_found = true;
            continue;
        }

        match artifact.artifact_kind.as_str() {
            "proof_body" => {
                if artifact.available_at_position.is_some()
                    || artifact
                        .target_position
                        .is_none_or(|position| position >= payload.position)
                {
                    bail!(
                        "bundle proof {} is not an earlier completed proof",
                        artifact.artifact_id
                    );
                }
            }
            "definition" | "module_context" | "theorem_statement" => {
                if artifact.target_position.is_some()
                    || artifact
                        .available_at_position
                        .is_none_or(|position| position >= payload.position)
                {
                    bail!(
                        "bundle artifact {} was not available before the target",
                        artifact.artifact_id
                    );
                }
            }
            other => bail!("bundle contains forbidden artifact kind {other}"),
        }
        if artifact.theorem_id.as_deref() == Some(payload.theorem_id.as_str())
            && artifact.artifact_kind == "proof_body"
            && artifact.target_position == Some(payload.position)
        {
            bail!("bundle contains the target's own proof body");
        }
    }

    if !statement_found {
        bail!("bundle omits the exact target statement artifact");
    }
    println!(
        "PITB_TEMPORAL_BUNDLE_VERIFY_PASS: target={} visible_artifacts={} payload_sha256={}",
        payload.target_id,
        payload.visible_artifacts.len(),
        bundle.payload_sha256
    );
    Ok(())
}

fn validate_temporal_manifest(manifest: &TemporalReplayManifest) -> Result<()> {
    if manifest.schema_version != "pitb.temporal-replay-manifest.v0" {
        bail!(
            "unsupported temporal replay manifest schema {}",
            manifest.schema_version
        );
    }
    if !manifest.repository_url.starts_with("https://") {
        bail!("temporal repository URL must be an explicit HTTPS URL");
    }
    validate_git_sha(&manifest.repository_commit).context("invalid pinned repository commit")?;
    let normalized_timeline_path = normalize_root_relative_path(&manifest.timeline_path)?;
    if normalized_timeline_path != manifest.timeline_path || manifest.timeline_path.is_empty() {
        bail!("timeline path must be canonical and repository-relative");
    }
    validate_sha256_hex(&manifest.timeline_sha256).context("invalid timeline SHA-256")?;
    for (label, path) in [
        ("Lean toolchain", manifest.lean_toolchain_path.as_str()),
        ("Mathlib manifest", manifest.mathlib_manifest_path.as_str()),
    ] {
        let normalized = normalize_root_relative_path(path)?;
        if path.is_empty() || normalized != path {
            bail!("{label} path must be canonical and repository-relative");
        }
    }
    validate_sha256_hex(&manifest.mathlib_manifest_sha256)
        .context("invalid Mathlib manifest SHA-256")?;
    if manifest.lean_toolchain.trim().is_empty()
        || manifest.lean_toolchain.eq_ignore_ascii_case("latest")
        || manifest.mathlib_revision.trim().is_empty()
        || manifest.mathlib_revision.eq_ignore_ascii_case("latest")
    {
        bail!("temporal manifest must pin Lean and Mathlib versions");
    }
    validate_git_sha(&manifest.mathlib_revision)
        .context("Mathlib revision must be a full commit SHA")?;
    if manifest.events.is_empty() || manifest.artifacts.is_empty() {
        bail!("temporal replay manifest must contain both timeline events and evidence artifacts");
    }

    let mut event_positions = BTreeSet::new();
    let mut target_ids = BTreeSet::new();
    let mut event_theorems = BTreeMap::new();
    let mut previous_position = None;
    for event in &manifest.events {
        if event.target_id.trim().is_empty() || event.theorem_id.trim().is_empty() {
            bail!("temporal targets require non-empty target and theorem IDs");
        }
        if !target_ids.insert(event.target_id.as_str()) {
            bail!("duplicate temporal target ID {}", event.target_id);
        }
        if !event_positions.insert(event.position) {
            bail!("duplicate temporal position {}", event.position);
        }
        if previous_position.is_some_and(|previous| event.position <= previous) {
            bail!("temporal events must be in strictly increasing timeline order");
        }
        previous_position = Some(event.position);
        validate_git_sha(&event.target_state_commit)
            .with_context(|| format!("invalid target-state commit for {}", event.target_id))?;
        validate_git_sha(&event.proof_completion_commit)
            .with_context(|| format!("invalid proof-completion commit for {}", event.target_id))?;
        validate_sha256_hex(&event.statement_sha256)
            .with_context(|| format!("invalid statement digest for {}", event.target_id))?;
        if event.statement_artifact_id == event.proof_artifact_id {
            bail!(
                "target statement and hidden proof must be separate artifacts for {}",
                event.target_id
            );
        }
        if event_theorems
            .insert(event.proof_artifact_id.as_str(), event)
            .is_some()
        {
            bail!("duplicate hidden proof artifact ID in temporal timeline");
        }
    }

    let mut artifacts_by_id = BTreeMap::new();
    for artifact in &manifest.artifacts {
        if artifact.artifact_id.trim().is_empty() {
            bail!("temporal artifact IDs must not be empty");
        }
        if artifacts_by_id
            .insert(artifact.artifact_id.as_str(), artifact)
            .is_some()
        {
            bail!("duplicate temporal artifact ID {}", artifact.artifact_id);
        }
        let normalized_path = normalize_root_relative_path(&artifact.path)?;
        if normalized_path != artifact.path {
            bail!(
                "temporal artifact path must be canonical and repository-relative: {}",
                artifact.path
            );
        }
        if artifact.path.is_empty() {
            bail!("temporal artifact path must not be empty");
        }
        validate_sha256_hex(&artifact.content_sha256)
            .with_context(|| format!("invalid content digest for {}", artifact.artifact_id))?;
        validate_sha256_hex(&artifact.source_file_sha256)
            .with_context(|| format!("invalid source-file digest for {}", artifact.artifact_id))?;
        validate_git_sha(&artifact.source_commit)
            .with_context(|| format!("invalid source commit for {}", artifact.artifact_id))?;
        if artifact.content_start_line == 0
            || artifact.content_end_line < artifact.content_start_line
        {
            bail!(
                "artifact {} must have a valid one-based inclusive line range",
                artifact.artifact_id
            );
        }
        if ![
            "theorem_statement",
            "completion_statement",
            "proof_body",
            "definition",
            "module_context",
        ]
        .contains(&artifact.artifact_kind.as_str())
        {
            bail!(
                "unsupported temporal artifact kind {} for {}",
                artifact.artifact_kind,
                artifact.artifact_id
            );
        }
        match artifact.artifact_kind.as_str() {
            "proof_body" | "completion_statement" => {
                if artifact.available_at_position.is_some() || artifact.target_position.is_none() {
                    bail!(
                        "evaluator-only artifact {} must have target_position and no solver-visible available_at_position",
                        artifact.artifact_id
                    );
                }
                if artifact.target_position.is_some_and(|position| {
                    !manifest
                        .events
                        .iter()
                        .any(|event| event.position == position)
                }) {
                    bail!(
                        "proof artifact {} is associated with an unregistered target position",
                        artifact.artifact_id
                    );
                }
            }
            _ => {
                if artifact.available_at_position.is_none() || artifact.target_position.is_some() {
                    bail!(
                        "solver-visible artifact {} must have available_at_position and no target_position",
                        artifact.artifact_id
                    );
                }
                if artifact.available_at_position.is_some_and(|position| {
                    position
                        > manifest
                            .events
                            .last()
                            .map(|event| event.position)
                            .unwrap_or(0)
                }) {
                    bail!(
                        "artifact {} appears after the last registered timeline event",
                        artifact.artifact_id
                    );
                }
            }
        }
    }

    for event in &manifest.events {
        let statement = artifacts_by_id
            .get(event.statement_artifact_id.as_str())
            .with_context(|| {
                format!(
                    "target {} lacks its target-statement artifact",
                    event.target_id
                )
            })?;
        if statement.artifact_kind != "theorem_statement"
            || statement.theorem_id.as_deref() != Some(event.theorem_id.as_str())
            || statement
                .available_at_position
                .is_none_or(|position| position > event.position)
            || statement.target_position.is_some()
            || statement.content_sha256 != event.statement_sha256
        {
            bail!(
                "target-statement artifact does not match the statement/position contract for {}",
                event.target_id
            );
        }
        let proof = artifacts_by_id
            .get(event.proof_artifact_id.as_str())
            .with_context(|| {
                format!(
                    "target {} lacks its evaluator-only proof artifact",
                    event.target_id
                )
            })?;
        if proof.artifact_kind != "proof_body"
            || proof.theorem_id.as_deref() != Some(event.theorem_id.as_str())
            || proof.available_at_position.is_some()
            || proof.target_position != Some(event.position)
            || proof.source_commit != event.proof_completion_commit
        {
            bail!(
                "hidden proof artifact does not match its timeline event for {}",
                event.target_id
            );
        }
        if event.completion_statement_artifact_id == event.statement_artifact_id
            || event.completion_statement_artifact_id == event.proof_artifact_id
        {
            bail!("target statement, completion statement, and proof must be distinct artifacts");
        }
        let completion_statement = artifacts_by_id
            .get(event.completion_statement_artifact_id.as_str())
            .with_context(|| {
                format!(
                    "target {} lacks its evaluator-only completion statement artifact",
                    event.target_id
                )
            })?;
        if completion_statement.artifact_kind != "completion_statement"
            || completion_statement.theorem_id.as_deref() != Some(event.theorem_id.as_str())
            || completion_statement.available_at_position.is_some()
            || completion_statement.target_position != Some(event.position)
            || completion_statement.source_commit != event.proof_completion_commit
            || completion_statement.content_sha256 != event.statement_sha256
            || completion_statement.path != proof.path
            || completion_statement.source_file_sha256 != proof.source_file_sha256
            || completion_statement.content_end_line >= proof.content_start_line
        {
            bail!(
                "theorem statement changed between target state and proof completion for {}",
                event.target_id
            );
        }

        let mut visible_ids = BTreeSet::new();
        for artifact_id in &event.visible_artifact_ids {
            if !visible_ids.insert(artifact_id.as_str()) {
                bail!(
                    "target {} repeats visible artifact ID {}",
                    event.target_id,
                    artifact_id
                );
            }
            let artifact = artifacts_by_id.get(artifact_id.as_str()).with_context(|| {
                format!(
                    "target {} references unknown visible artifact {}",
                    event.target_id, artifact_id
                )
            })?;
            if artifact_id == &event.proof_artifact_id {
                bail!(
                    "target {} leaks its own proof artifact into model-visible context",
                    event.target_id
                );
            }
            if artifact_id != &event.statement_artifact_id {
                match artifact.artifact_kind.as_str() {
                    "completion_statement" => {
                        bail!(
                            "evaluator-only completion statement {} cannot be model-visible",
                            artifact.artifact_id
                        );
                    }
                    "proof_body" => {
                        if artifact
                            .target_position
                            .is_none_or(|position| position >= event.position)
                        {
                            bail!(
                                "future-proof leakage: proof {} is not from an earlier completed target than {}",
                                artifact.artifact_id,
                                event.target_id
                            );
                        }
                    }
                    _ => {
                        if artifact
                            .available_at_position
                            .is_none_or(|position| position >= event.position)
                        {
                            bail!(
                                "future-evidence leakage: artifact {} was not available before target {}",
                                artifact.artifact_id,
                                event.target_id
                            );
                        }
                    }
                }
            }
        }
        if !visible_ids.contains(event.statement_artifact_id.as_str()) {
            bail!(
                "target {} must expose its exact target statement to the solver",
                event.target_id
            );
        }
        if visible_ids.contains(event.proof_artifact_id.as_str()) {
            bail!(
                "target {} must hide the ground-truth proof from the solver",
                event.target_id
            );
        }

        for visible_id in &event.visible_artifact_ids {
            let visible = artifacts_by_id
                .get(visible_id.as_str())
                .context("visible artifact missing after prior lookup")?;
            for (future_proof_id, future_event) in &event_theorems {
                if future_event.position >= event.position {
                    let future_proof = artifacts_by_id
                        .get(*future_proof_id)
                        .context("future proof artifact missing after registration")?;
                    if visible.content_sha256 == future_proof.content_sha256 {
                        bail!(
                            "exact proof-content leakage: target {} sees artifact {} with the content digest of hidden/future proof {}",
                            event.target_id,
                            visible.artifact_id,
                            future_proof.artifact_id
                        );
                    }
                }
            }
        }
    }

    for artifact in &manifest.artifacts {
        if artifact.artifact_kind == "proof_body" {
            let event = event_theorems
                .get(artifact.artifact_id.as_str())
                .with_context(|| {
                    format!(
                        "proof artifact {} is not assigned to a registered timeline event",
                        artifact.artifact_id
                    )
                })?;
            if artifact.target_position != Some(event.position)
                || event.theorem_id.as_str() != artifact.theorem_id.as_deref().unwrap_or_default()
            {
                bail!(
                    "proof artifact {} is inconsistent with its timeline event",
                    artifact.artifact_id
                );
            }
        }
    }

    Ok(())
}

fn find_mathlib_revision(value: &serde_json::Value) -> Option<&str> {
    match value {
        serde_json::Value::Object(object) => {
            let is_mathlib_package = object
                .get("name")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|name| name.eq_ignore_ascii_case("mathlib"))
                || object
                    .get("url")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|url| url.to_ascii_lowercase().contains("mathlib4"));
            if is_mathlib_package
                && let Some(revision) = ["rev", "revision", "inputRev", "input_rev"]
                    .iter()
                    .find_map(|key| object.get(*key).and_then(serde_json::Value::as_str))
            {
                return Some(revision);
            }
            object.values().find_map(find_mathlib_revision)
        }
        serde_json::Value::Array(values) => values.iter().find_map(find_mathlib_revision),
        _ => None,
    }
}

fn normalize_repository_url(url: &str) -> String {
    let normalized = url.trim().trim_end_matches('/').to_ascii_lowercase();
    normalized
        .strip_suffix(".git")
        .unwrap_or(&normalized)
        .to_owned()
}

fn validate_git_sha(value: &str) -> Result<()> {
    if value.len() != 40 || !value.chars().all(|character| character.is_ascii_hexdigit()) {
        bail!("expected a full 40-character Git SHA");
    }
    Ok(())
}

fn validate_sha256_hex(value: &str) -> Result<()> {
    if value.len() != 64 || !value.chars().all(|character| character.is_ascii_hexdigit()) {
        bail!("expected a 64-character SHA-256 digest");
    }
    Ok(())
}

fn git_is_ancestor(root: &Path, ancestor: &str, descendant: &str) -> Result<bool> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["merge-base", "--is-ancestor", ancestor, descendant])
        .output()
        .context("failed to run git merge-base for temporal ancestry validation")?;
    if output.status.success() {
        Ok(true)
    } else if output.status.code() == Some(1) {
        Ok(false)
    } else {
        bail!(
            "git merge-base failed while checking temporal ancestry: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
    }
}

fn git_commit_exists(root: &Path, commit: &str) -> Result<bool> {
    let revision = format!("{commit}^{{commit}}");
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["cat-file", "-e", revision.as_str()])
        .output()
        .with_context(|| format!("failed to test historical commit object {commit}"))?;
    Ok(output.status.success())
}

fn git_path_exists_at_commit(root: &Path, commit: &str, path: &str) -> Result<bool> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "ls-tree",
            "--full-tree",
            "-r",
            "--name-only",
            commit,
            "--",
            path,
        ])
        .output()
        .with_context(|| format!("failed to list historical Git path {commit}:{path}"))?;
    if !output.status.success() {
        bail!(
            "git ls-tree for {commit}:{path} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let entries = std::str::from_utf8(&output.stdout)
        .context("git ls-tree returned non-UTF-8 historical paths")?;
    Ok(entries.lines().any(|entry| entry == path))
}

fn git_show_file(root: &Path, commit: &str, path: &str) -> Result<Vec<u8>> {
    let revision = format!("{commit}:{path}");
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["show", &revision])
        .output()
        .with_context(|| format!("failed to read pinned source {revision}"))?;
    if !output.status.success() {
        bail!(
            "git show {revision} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output.stdout)
}

fn exact_line_span(source: &[u8], start_line: usize, end_line: usize) -> Result<&[u8]> {
    let text = std::str::from_utf8(source).context("pinned source file is not UTF-8")?;
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    if start_line == 0 || end_line < start_line || end_line > lines.len() {
        bail!(
            "line range {start_line}-{end_line} is outside {} source lines",
            lines.len()
        );
    }
    let start = lines
        .iter()
        .take(start_line - 1)
        .map(|line| line.len())
        .sum::<usize>();
    let length = lines
        .iter()
        .skip(start_line - 1)
        .take(end_line - start_line + 1)
        .map(|line| line.len())
        .sum::<usize>();
    Ok(&source[start..start + length])
}

fn verify_split_files(generated_path: &Path, frozen_path: &Path) -> Result<()> {
    let generated_text = fs::read_to_string(generated_path).with_context(|| {
        format!(
            "cannot read generated source index {}",
            generated_path.display()
        )
    })?;
    let frozen_text = fs::read_to_string(frozen_path).with_context(|| {
        format!(
            "cannot read frozen split manifest {}",
            frozen_path.display()
        )
    })?;
    let generated: serde_json::Value =
        serde_json::from_str(&generated_text).context("generated index is not valid JSON")?;
    let frozen: serde_json::Value =
        serde_json::from_str(&frozen_text).context("frozen split manifest is not valid JSON")?;

    let generated_commit = generated
        .get("upstream_commit")
        .and_then(serde_json::Value::as_str)
        .context("generated index is missing upstream_commit")?;
    let frozen_commit = frozen
        .get("upstream")
        .and_then(|value| value.get("commit"))
        .and_then(serde_json::Value::as_str)
        .context("frozen split manifest is missing upstream.commit")?;
    if generated_commit != frozen_commit {
        bail!(
            "source revision mismatch: generated index {generated_commit}, frozen manifest {frozen_commit}"
        );
    }

    let generated_plan = generated
        .get("family_split_plan")
        .context("generated index has no family_split_plan")?;
    let frozen_plan = frozen
        .get("split_plan")
        .context("frozen manifest has no split_plan")?;
    for field in [
        "seed",
        "algorithm",
        "train_percent",
        "validation_percent",
        "test_percent",
    ] {
        if generated_plan.get(field) != frozen_plan.get(field) {
            bail!("split-plan metadata mismatch in field {field}");
        }
    }
    let frozen_counts = frozen_plan
        .get("counts")
        .context("frozen split plan has no counts object")?;
    for (generated_field, frozen_field) in [
        ("train_family_count", "train"),
        ("validation_family_count", "validation"),
        ("test_family_count", "test"),
    ] {
        if generated_plan.get(generated_field) != frozen_counts.get(frozen_field) {
            bail!(
                "split-plan count mismatch: generated {generated_field}, frozen counts.{frozen_field}"
            );
        }
    }
    let generated_digest = generated_plan
        .get("partition_digest_sha256")
        .and_then(serde_json::Value::as_str)
        .context("generated index has no partition digest")?;
    let frozen_digest = frozen_plan
        .get("partition_digest_sha256")
        .and_then(serde_json::Value::as_str)
        .context("frozen manifest has no partition digest")?;
    if generated_digest != frozen_digest {
        bail!("family split digest mismatch: generated {generated_digest}, frozen {frozen_digest}");
    }

    let fingerprint_map = |value: &serde_json::Value| -> Result<BTreeMap<String, (String, u64)>> {
        let fingerprints = value
            .get("source_fingerprints")
            .and_then(serde_json::Value::as_array)
            .context("source_fingerprints must be an array")?;
        let mut result = BTreeMap::new();
        for item in fingerprints {
            let path = item
                .get("path")
                .and_then(serde_json::Value::as_str)
                .context("fingerprint missing path")?;
            let digest = item
                .get("sha256")
                .and_then(serde_json::Value::as_str)
                .context("fingerprint missing sha256")?;
            let bytes = item
                .get("bytes")
                .and_then(serde_json::Value::as_u64)
                .context("fingerprint missing byte count")?;
            if result
                .insert(path.to_owned(), (digest.to_owned(), bytes))
                .is_some()
            {
                bail!("duplicate source fingerprint path: {path}");
            }
        }
        Ok(result)
    };
    let generated_fingerprints = fingerprint_map(&generated)?;
    let frozen_fingerprints = fingerprint_map(&frozen)?;
    for (path, expected) in &frozen_fingerprints {
        let actual = generated_fingerprints.get(path).with_context(|| {
            format!("generated index is missing pinned source fingerprint: {path}")
        })?;
        if actual != expected {
            bail!("source fingerprint mismatch for {path}");
        }
    }

    let rows = |value: &serde_json::Value, field: &str| -> Result<BTreeMap<String, String>> {
        let entries = value
            .get(field)
            .and_then(serde_json::Value::as_array)
            .with_context(|| format!("missing array field {field}"))?;
        let mut result = BTreeMap::new();
        for entry in entries {
            let id = entry
                .get("family_id")
                .and_then(serde_json::Value::as_str)
                .context("family record missing family_id")?;
            let split = entry
                .get("split")
                .and_then(serde_json::Value::as_str)
                .context("family record missing split")?;
            if result.insert(id.to_owned(), split.to_owned()).is_some() {
                bail!("duplicate family ID {id} in {field}");
            }
        }
        Ok(result)
    };
    let generated_rows = rows(&generated, "families")?;
    let frozen_rows = rows(&frozen, "families")?;
    if generated_rows != frozen_rows {
        let generated_only: Vec<_> = generated_rows
            .keys()
            .filter(|key| !frozen_rows.contains_key(*key))
            .cloned()
            .collect();
        let frozen_only: Vec<_> = frozen_rows
            .keys()
            .filter(|key| !generated_rows.contains_key(*key))
            .cloned()
            .collect();
        bail!(
            "family assignment mismatch (generated {}, frozen {}); generated-only IDs: {:?}; frozen-only IDs: {:?}",
            generated_rows.len(),
            frozen_rows.len(),
            generated_only,
            frozen_only
        );
    }

    eprintln!(
        "PASS: {} family assignments match frozen digest {} for source {}",
        generated_rows.len(),
        frozen_digest,
        generated_commit
    );
    Ok(())
}

fn git_output(root: &Path, arguments: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(arguments)
        .output()
        .with_context(|| format!("failed to launch git {}", arguments.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            arguments.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn fingerprint(root: &Path, relative_path: &str) -> Result<SourceFingerprint> {
    let canonical_root = root
        .canonicalize()
        .with_context(|| format!("cannot canonicalize source root {}", root.display()))?;
    let path = root.join(relative_path);
    let canonical_path = path
        .canonicalize()
        .with_context(|| format!("cannot resolve source artifact {}", path.display()))?;
    if !canonical_path.starts_with(&canonical_root) {
        bail!(
            "source artifact escapes the pinned repository checkout: {}",
            relative_path
        );
    }
    let mut file = File::open(&canonical_path)
        .with_context(|| format!("cannot open source artifact {}", canonical_path.display()))?;
    let bytes = file
        .metadata()
        .with_context(|| format!("cannot stat source artifact {}", canonical_path.display()))?
        .len();
    let mut content = Vec::new();
    file.read_to_end(&mut content)
        .with_context(|| format!("cannot read source artifact {}", path.display()))?;
    Ok(SourceFingerprint {
        path: relative_path.to_owned(),
        sha256: format!("{:x}", Sha256::digest(&content)),
        bytes,
    })
}

fn inventory_catalogue_links(
    root: &Path,
    families: &[FamilyRecord],
    source_file_fingerprints: &mut BTreeMap<String, SourceFingerprint>,
) -> Result<Vec<CatalogueLinkedArtifact>> {
    let linked_paths: BTreeSet<String> = families
        .iter()
        .flat_map(|family| family.linked_paths.iter().cloned())
        .collect();
    let mut inventory = Vec::with_capacity(linked_paths.len());

    for relative_path in linked_paths {
        let candidate_path = root.join(&relative_path);
        if candidate_path.is_file() {
            // fingerprint() canonicalizes and rejects symlinks that leave the pinned checkout.
            let artifact = fingerprint(root, &relative_path)?;
            source_file_fingerprints
                .entry(relative_path.clone())
                .or_insert_with(|| artifact.clone());
            inventory.push(CatalogueLinkedArtifact {
                path: relative_path,
                file_exists: true,
                sha256: Some(artifact.sha256),
                bytes: Some(artifact.bytes),
            });
        } else {
            inventory.push(CatalogueLinkedArtifact {
                path: relative_path,
                file_exists: false,
                sha256: None,
                bytes: None,
            });
        }
    }

    Ok(inventory)
}

fn extract_families(contents: &str) -> Result<Vec<FamilyRecord>> {
    // The pinned CONTENTS.md presents each result family as **NNN. Title**.
    // Anything outside this known catalogue shape is treated as a failure.
    let header_regex =
        Regex::new(r"\*\*(\d{3})\.\s*([^*]+?)\*\*").context("compile family-header expression")?;
    let link_regex =
        Regex::new(r"\[[^\]]*\]\(([^)]+)\)").context("compile Markdown-link expression")?;
    let headers: Vec<_> = header_regex.captures_iter(contents).collect();
    if headers.is_empty() {
        bail!("CONTENTS.md contains no supported **NNN. Title** family headers");
    }

    let mut seen_ids = BTreeSet::new();
    let mut families = Vec::with_capacity(headers.len());
    for (index, header) in headers.iter().enumerate() {
        let family_id = header[1].to_owned();
        if !seen_ids.insert(family_id.clone()) {
            bail!("duplicate family identifier {family_id} in CONTENTS.md");
        }
        let title = header[2].trim().to_owned();
        let start = header.get(0).context("family header match is empty")?.end();
        let end = headers
            .get(index + 1)
            .and_then(|next| next.get(0))
            .map(|matched| matched.start())
            .unwrap_or(contents.len());
        if start > end || end > contents.len() {
            bail!("invalid source range for family {family_id}");
        }
        let body = &contents[start..end];
        let linked_paths: BTreeSet<String> = link_regex
            .captures_iter(body)
            .filter_map(|capture| capture.get(1).map(|m| m.as_str().trim()))
            .filter(|target| !target.contains("://") && !target.starts_with('#'))
            .map(normalize_root_relative_path)
            .collect::<Result<BTreeSet<_>>>()?;

        families.push(FamilyRecord {
            family_id,
            title,
            linked_paths: linked_paths.into_iter().collect(),
            directly_mapped_formalization_sources: Vec::new(),
            split: String::new(),
        });
    }
    Ok(families)
}

fn map_source_paths_to_families(families: &[FamilyRecord]) -> Result<BTreeMap<String, String>> {
    let mut source_path_to_family: BTreeMap<String, String> = BTreeMap::new();
    for family in families {
        for linked_path in &family.linked_paths {
            match source_path_to_family.get(linked_path) {
                Some(previous_family_id) if previous_family_id != &family.family_id => {
                    bail!(
                        "catalogue path {} is linked from multiple result families: {} and {}",
                        linked_path,
                        previous_family_id,
                        family.family_id
                    );
                }
                Some(_) => {}
                None => {
                    source_path_to_family.insert(linked_path.clone(), family.family_id.clone());
                }
            }
        }
    }
    Ok(source_path_to_family)
}

fn parse_catalogue_claims(readme: &str, parsed_family_count: usize) -> Result<CatalogueClaims> {
    let expression = Regex::new(
        r"(?i)current catalogue contains ([0-9,]+) manuscripts organized into ([0-9,]+) families",
    )
    .context("compile OpenAI math catalogue claim expression")?;
    let captures = expression.captures(readme).context(
        "README.md does not contain the expected manuscript/family catalogue count; refusing unrecognized source metadata",
    )?;
    let parse_count = |value: &str| -> Result<usize> {
        value
            .replace(',', "")
            .parse::<usize>()
            .context("catalogue count is not a valid positive integer")
    };
    let manuscripts = parse_count(&captures[1])?;
    let claimed_families = parse_count(&captures[2])?;
    if manuscripts == 0 || claimed_families == 0 {
        bail!("README.md declares an empty manuscript or family catalogue");
    }
    if claimed_families != parsed_family_count {
        bail!(
            "catalogue family-count mismatch: README claims {}, CONTENTS.md has {} headers",
            claimed_families,
            parsed_family_count
        );
    }
    Ok(CatalogueClaims {
        readme_claimed_manuscript_count: manuscripts,
        readme_claimed_family_count: claimed_families,
        parsed_family_header_count: parsed_family_count,
    })
}

fn assign_family_splits(families: &mut [FamilyRecord]) -> Result<FamilySplitPlan> {
    if families.len() < 20 {
        bail!(
            "only {} families found; refusing to create a misleadingly small development/test split",
            families.len()
        );
    }
    let mut ranked: Vec<(Vec<u8>, String)> = families
        .iter()
        .map(|family| {
            let mut hasher = Sha256::new();
            hasher.update(FAMILY_SPLIT_SEED.as_bytes());
            hasher.update([0]);
            hasher.update(family.family_id.as_bytes());
            (hasher.finalize().to_vec(), family.family_id.clone())
        })
        .collect();
    ranked.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));

    let total = ranked.len();
    let train_count = total * 70 / 100;
    let validation_end = total * 85 / 100;
    let mut assignment = BTreeMap::new();
    for (rank, (_, family_id)) in ranked.into_iter().enumerate() {
        let split = if rank < train_count {
            "train"
        } else if rank < validation_end {
            "validation"
        } else {
            "test"
        };
        assignment.insert(family_id, split.to_owned());
    }
    for family in families.iter_mut() {
        family.split = assignment
            .get(&family.family_id)
            .cloned()
            .context("family missing from deterministic split assignment")?;
    }

    let mut partition_rows: Vec<(&str, &str)> = families
        .iter()
        .map(|family| (family.family_id.as_str(), family.split.as_str()))
        .collect();
    partition_rows.sort_unstable_by(|left, right| left.0.cmp(right.0));
    let mut partition_material = String::new();
    for (family_id, split) in partition_rows {
        partition_material.push_str(family_id);
        partition_material.push('\t');
        partition_material.push_str(split);
        partition_material.push('\n');
    }
    let partition_digest_sha256 = format!("{:x}", Sha256::digest(partition_material.as_bytes()));

    Ok(FamilySplitPlan {
        seed: FAMILY_SPLIT_SEED,
        algorithm: FAMILY_SPLIT_ALGORITHM,
        train_percent: 70,
        validation_percent: 15,
        test_percent: 15,
        train_family_count: train_count,
        validation_family_count: validation_end - train_count,
        test_family_count: total - validation_end,
        partition_digest_sha256,
    })
}

fn normalize_lean_relative_path(path: &str) -> Result<String> {
    normalize_with_base(["lean"], path)
}

fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalize_root_relative_path(path: &str) -> Result<String> {
    normalize_with_base(std::iter::empty::<&str>(), path)
}

fn normalize_with_base<'a>(base: impl IntoIterator<Item = &'a str>, path: &str) -> Result<String> {
    if path.contains("://") {
        bail!("URL is not a repository-relative artifact path: {path}");
    }
    let mut parts: Vec<String> = base.into_iter().map(str::to_owned).collect();
    for component in Path::new(path).components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if parts.pop().is_none() {
                    bail!("artifact path escapes repository root: {path}");
                }
            }
            Component::Normal(value) => parts.push(value.to_string_lossy().into_owned()),
            Component::RootDir | Component::Prefix(_) => {
                bail!("absolute path is not a repository-relative artifact path: {path}");
            }
        }
    }
    if parts.is_empty() {
        bail!("empty repository-relative artifact path: {path}");
    }
    Ok(parts.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporal_manifest_fixture() -> TemporalReplayManifest {
        let statement_one = "a".repeat(64);
        let proof_one = "b".repeat(64);
        let statement_two = "c".repeat(64);
        let proof_two = "d".repeat(64);
        let source_one = "e".repeat(64);
        let source_two = "f".repeat(64);
        let source_three = "1".repeat(64);
        let source_four = "2".repeat(64);
        let commit_one = "1".repeat(40);
        let commit_two = "2".repeat(40);
        let commit_three = "3".repeat(40);

        TemporalReplayManifest {
            schema_version: "pitb.temporal-replay-manifest.v0".to_string(),
            repository_url: "https://example.invalid/temporal-fixture".to_string(),
            repository_commit: "5".repeat(40),
            timeline_path: "proof_timeline.csv".to_string(),
            timeline_sha256: "9".repeat(64),
            lean_toolchain_path: "lean-toolchain".to_string(),
            lean_toolchain: "leanprover/lean4:v4.34.1".to_string(),
            mathlib_manifest_path: "lake-manifest.json".to_string(),
            mathlib_manifest_sha256: "8".repeat(64),
            mathlib_revision: "6".repeat(40),
            events: vec![
                TemporalTimelineEvent {
                    position: 0,
                    target_id: "target-000".to_string(),
                    theorem_id: "Example.first".to_string(),
                    target_state_commit: commit_one.clone(),
                    proof_completion_commit: commit_two.clone(),
                    statement_sha256: statement_one.clone(),
                    statement_artifact_id: "statement-000".to_string(),
                    completion_statement_artifact_id: "completion-statement-000".to_string(),
                    proof_artifact_id: "proof-000".to_string(),
                    visible_artifact_ids: vec!["statement-000".to_string()],
                },
                TemporalTimelineEvent {
                    position: 1,
                    target_id: "target-001".to_string(),
                    theorem_id: "Example.second".to_string(),
                    target_state_commit: commit_two.clone(),
                    proof_completion_commit: commit_three.clone(),
                    statement_sha256: statement_two.clone(),
                    statement_artifact_id: "statement-001".to_string(),
                    completion_statement_artifact_id: "completion-statement-001".to_string(),
                    proof_artifact_id: "proof-001".to_string(),
                    visible_artifact_ids: vec![
                        "statement-000".to_string(),
                        "proof-000".to_string(),
                        "statement-001".to_string(),
                    ],
                },
            ],
            artifacts: vec![
                TemporalEvidenceArtifact {
                    artifact_id: "statement-000".to_string(),
                    path: "lean/Example.lean".to_string(),
                    artifact_kind: "theorem_statement".to_string(),
                    content_sha256: statement_one.clone(),
                    content_start_line: 1,
                    content_end_line: 1,
                    source_file_sha256: source_one.clone(),
                    source_commit: commit_one.clone(),
                    available_at_position: Some(0),
                    target_position: None,
                    theorem_id: Some("Example.first".to_string()),
                },
                TemporalEvidenceArtifact {
                    artifact_id: "proof-000".to_string(),
                    path: "lean/Example.lean".to_string(),
                    artifact_kind: "proof_body".to_string(),
                    content_sha256: proof_one,
                    content_start_line: 2,
                    content_end_line: 2,
                    source_file_sha256: source_two.clone(),
                    source_commit: commit_two.clone(),
                    available_at_position: None,
                    target_position: Some(0),
                    theorem_id: Some("Example.first".to_string()),
                },
                TemporalEvidenceArtifact {
                    artifact_id: "completion-statement-000".to_string(),
                    path: "lean/Example.lean".to_string(),
                    artifact_kind: "completion_statement".to_string(),
                    content_sha256: statement_one.clone(),
                    content_start_line: 1,
                    content_end_line: 1,
                    source_file_sha256: source_two.clone(),
                    source_commit: commit_two.clone(),
                    available_at_position: None,
                    target_position: Some(0),
                    theorem_id: Some("Example.first".to_string()),
                },
                TemporalEvidenceArtifact {
                    artifact_id: "statement-001".to_string(),
                    path: "lean/Example.lean".to_string(),
                    artifact_kind: "theorem_statement".to_string(),
                    content_sha256: statement_two.clone(),
                    content_start_line: 3,
                    content_end_line: 3,
                    source_file_sha256: source_three.clone(),
                    source_commit: commit_two.clone(),
                    available_at_position: Some(1),
                    target_position: None,
                    theorem_id: Some("Example.second".to_string()),
                },
                TemporalEvidenceArtifact {
                    artifact_id: "completion-statement-001".to_string(),
                    path: "lean/Example.lean".to_string(),
                    artifact_kind: "completion_statement".to_string(),
                    content_sha256: statement_two.clone(),
                    content_start_line: 3,
                    content_end_line: 3,
                    source_file_sha256: source_four.clone(),
                    source_commit: commit_three.clone(),
                    available_at_position: None,
                    target_position: Some(1),
                    theorem_id: Some("Example.second".to_string()),
                },
                TemporalEvidenceArtifact {
                    artifact_id: "proof-001".to_string(),
                    path: "lean/Example.lean".to_string(),
                    artifact_kind: "proof_body".to_string(),
                    content_sha256: proof_two,
                    content_start_line: 4,
                    content_end_line: 4,
                    source_file_sha256: source_four.clone(),
                    source_commit: commit_three.clone(),
                    available_at_position: None,
                    target_position: Some(1),
                    theorem_id: Some("Example.second".to_string()),
                },
            ],
        }
    }

    #[test]
    fn temporal_manifest_replays_against_real_pinned_git_bytes() {
        use std::time::{SystemTime, UNIX_EPOCH};

        fn git(root: &Path, arguments: &[&str]) -> String {
            let output = Command::new("git")
                .arg("-C")
                .arg(root)
                .args(arguments)
                .output()
                .expect("git process launches");
            assert!(
                output.status.success(),
                "git {} failed: {}",
                arguments.join(" "),
                String::from_utf8_lossy(&output.stderr)
            );
            String::from_utf8_lossy(&output.stdout).trim().to_owned()
        }

        fn commit(root: &Path, message: &str) -> String {
            git(root, &["add", "-A"]);
            git(root, &["commit", "-m", message]);
            git(root, &["rev-parse", "HEAD"])
        }

        fn sha(bytes: &[u8]) -> String {
            format!("{:x}", Sha256::digest(bytes))
        }

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let base = env::temp_dir().join(format!(
            "pitb-temporal-git-replay-{}-{nonce}",
            std::process::id()
        ));
        let root = base.join("repo");
        let manifest_path = base.join("temporal-manifest.json");
        fs::create_dir_all(root.join("lean")).expect("create fixture repository");
        git(&root, &["init", "-q"]);
        git(&root, &["config", "user.name", "PITB Test"]);
        git(
            &root,
            &["config", "user.email", "pitb-test@example.invalid"],
        );
        git(
            &root,
            &[
                "remote",
                "add",
                "origin",
                "https://example.invalid/temporal-fixture",
            ],
        );

        let mut timeline = "position,target_id\n0,target-000\n1,target-001\n".to_string();
        fs::write(root.join("proof_timeline.csv"), &timeline).expect("write timeline");
        fs::write(root.join("lean-toolchain"), "leanprover/lean4:v4.34.1\n")
            .expect("write pinned Lean toolchain");
        let mathlib_revision = "6".repeat(40);
        let lake_manifest = format!(
            "{{\"packages\":[{{\"name\":\"mathlib\",\"url\":\"https://github.com/leanprover-community/mathlib4\",\"rev\":\"{mathlib_revision}\"}}]}}\n"
        );
        fs::write(root.join("lake-manifest.json"), &lake_manifest)
            .expect("write pinned Lake manifest");

        let initial_source =
            "theorem first : True := by\n  sorry\ntheorem second : True := by\n  sorry\n";
        fs::write(root.join("lean/Example.lean"), initial_source)
            .expect("write initial source state");
        // Deliberate source-document mismatch: the CSV will contain two rows,
        // while README claims three, and the audit must preserve that discrepancy.
        fs::write(
            root.join("README.md"),
            "The proof_timeline.csv currently records 4 theorems in this fixture.\n",
        )
        .expect("write deliberately inconsistent timeline README");
        let commit_one = commit(&root, "historical target statements");
        let source_one = git_show_file(&root, &commit_one, "lean/Example.lean")
            .expect("read first source revision");
        let source_one_sha = sha(&source_one);

        let second_source = initial_source.replacen("  sorry\n", "  exact True.intro\n", 1);
        fs::write(root.join("lean/Example.lean"), &second_source)
            .expect("complete first proof only");
        let commit_two = commit(&root, "complete first proof");
        let source_two = git_show_file(&root, &commit_two, "lean/Example.lean")
            .expect("read second source revision");
        let source_two_sha = sha(&source_two);

        let third_source = second_source.replacen("  sorry\n", "  trivial\n", 1);
        fs::write(root.join("lean/Example.lean"), &third_source).expect("complete second proof");
        let commit_three = commit(&root, "complete second proof");
        let source_three = git_show_file(&root, &commit_three, "lean/Example.lean")
            .expect("read third source revision");
        let source_three_sha = sha(&source_three);

        let statement_one_sha = sha(exact_line_span(&source_one, 1, 1).expect("line 1"));
        let proof_one_sha = sha(exact_line_span(&source_two, 2, 2).expect("line 2"));
        let statement_two_sha = sha(exact_line_span(&source_two, 3, 3).expect("line 3"));
        let proof_two_sha = sha(exact_line_span(&source_three, 4, 4).expect("line 4"));

        timeline = format!(
            "id,lean_name,spec_theorem,function,verified,date_proven,commit_hash,commit_message\n\
             tl_0000_first,,lean/Example.lean,Example.first,verified,2026-10-20,{commit_two},complete first proof\n\
             tl_0001_second,,lean/Example.lean,Example.second,verified,2026-10-21,{commit_three},complete second proof\n\
             tl_0002_missing,,lean/Missing.lean,Example.missing,externally verified,2026-10-22,{commit_three},source intentionally absent\n"
        );
        fs::write(root.join("proof_timeline.csv"), &timeline)
            .expect("write full timeline inventory");
        let repository_commit = commit(&root, "freeze historical timeline inventory");

        let manifest = serde_json::json!({
            "schema_version": "pitb.temporal-replay-manifest.v0",
            "repository_url": "https://example.invalid/temporal-fixture",
            "repository_commit": repository_commit,
            "timeline_path": "proof_timeline.csv",
            "timeline_sha256": sha(timeline.as_bytes()),
            "lean_toolchain_path": "lean-toolchain",
            "lean_toolchain": "leanprover/lean4:v4.34.1",
            "mathlib_manifest_path": "lake-manifest.json",
            "mathlib_manifest_sha256": sha(lake_manifest.as_bytes()),
            "mathlib_revision": mathlib_revision,
            "events": [
                {
                    "position": 0,
                    "target_id": "target-000",
                    "theorem_id": "Example.first",
                    "target_state_commit": commit_one,
                    "proof_completion_commit": commit_two,
                    "statement_sha256": statement_one_sha,
                    "statement_artifact_id": "statement-000",
                    "completion_statement_artifact_id": "completion-statement-000",
                    "proof_artifact_id": "proof-000",
                    "visible_artifact_ids": ["statement-000"]
                },
                {
                    "position": 1,
                    "target_id": "target-001",
                    "theorem_id": "Example.second",
                    "target_state_commit": commit_two,
                    "proof_completion_commit": commit_three,
                    "statement_sha256": statement_two_sha,
                    "statement_artifact_id": "statement-001",
                    "completion_statement_artifact_id": "completion-statement-001",
                    "proof_artifact_id": "proof-001",
                    "visible_artifact_ids": ["statement-000", "proof-000", "statement-001"]
                }
            ],
            "artifacts": [
                {
                    "artifact_id": "statement-000",
                    "path": "lean/Example.lean",
                    "artifact_kind": "theorem_statement",
                    "content_sha256": statement_one_sha,
                    "content_start_line": 1,
                    "content_end_line": 1,
                    "source_file_sha256": source_one_sha,
                    "source_commit": commit_one,
                    "available_at_position": 0,
                    "target_position": null,
                    "theorem_id": "Example.first"
                },
                {
                    "artifact_id": "proof-000",
                    "path": "lean/Example.lean",
                    "artifact_kind": "proof_body",
                    "content_sha256": proof_one_sha,
                    "content_start_line": 2,
                    "content_end_line": 2,
                    "source_file_sha256": source_two_sha,
                    "source_commit": commit_two,
                    "available_at_position": null,
                    "target_position": 0,
                    "theorem_id": "Example.first"
                },
                {
                    "artifact_id": "completion-statement-000",
                    "path": "lean/Example.lean",
                    "artifact_kind": "completion_statement",
                    "content_sha256": statement_one_sha,
                    "content_start_line": 1,
                    "content_end_line": 1,
                    "source_file_sha256": source_two_sha,
                    "source_commit": commit_two,
                    "available_at_position": null,
                    "target_position": 0,
                    "theorem_id": "Example.first"
                },
                {
                    "artifact_id": "statement-001",
                    "path": "lean/Example.lean",
                    "artifact_kind": "theorem_statement",
                    "content_sha256": statement_two_sha,
                    "content_start_line": 3,
                    "content_end_line": 3,
                    "source_file_sha256": source_two_sha,
                    "source_commit": commit_two,
                    "available_at_position": 1,
                    "target_position": null,
                    "theorem_id": "Example.second"
                },
                {
                    "artifact_id": "completion-statement-001",
                    "path": "lean/Example.lean",
                    "artifact_kind": "completion_statement",
                    "content_sha256": statement_two_sha,
                    "content_start_line": 3,
                    "content_end_line": 3,
                    "source_file_sha256": source_three_sha,
                    "source_commit": commit_three,
                    "available_at_position": null,
                    "target_position": 1,
                    "theorem_id": "Example.second"
                },
                {
                    "artifact_id": "proof-001",
                    "path": "lean/Example.lean",
                    "artifact_kind": "proof_body",
                    "content_sha256": proof_two_sha,
                    "content_start_line": 4,
                    "content_end_line": 4,
                    "source_file_sha256": source_three_sha,
                    "source_commit": commit_three,
                    "available_at_position": null,
                    "target_position": 1,
                    "theorem_id": "Example.second"
                }
            ]
        });
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize temporal manifest"),
        )
        .expect("write temporal manifest fixture");

        let tampered_source_manifest_path = base.join("tampered-source-manifest.json");
        let mut tampered_source_manifest = manifest.clone();
        tampered_source_manifest["artifacts"][0]["source_file_sha256"] =
            serde_json::Value::String("0".repeat(64));
        fs::write(
            &tampered_source_manifest_path,
            serde_json::to_vec_pretty(&tampered_source_manifest)
                .expect("serialize source-digest mutant"),
        )
        .expect("write source-digest mutant");
        let source_digest_error =
            verify_temporal_manifest_file(&root, &tampered_source_manifest_path)
                .expect_err("source-file mutation must be rejected against historical Git bytes");
        assert!(format!("{source_digest_error:#}").contains("source-file digest mismatch"));

        verify_temporal_manifest_file(&root, &manifest_path)
            .expect("pinned chronological history and exact artifact bytes should verify");
        let timeline_audit_path = base.join("timeline-audit.json");
        audit_temporal_timeline(&root, Path::new("proof_timeline.csv"), &timeline_audit_path)
            .expect("timeline auditor must preserve and validate historical rows");
        let timeline_audit: serde_json::Value = serde_json::from_slice(
            &fs::read(&timeline_audit_path).expect("read temporal timeline audit"),
        )
        .expect("parse temporal timeline audit JSON");
        assert_eq!(timeline_audit["total_timeline_rows"], 3);
        assert_eq!(timeline_audit["readme_claimed_timeline_row_count"], 4);
        assert_eq!(
            timeline_audit["timeline_row_count_matches_readme_claim"],
            false
        );
        assert_eq!(
            timeline_audit["repository_readme_sha256"],
            sha(git_show_file(&root, &repository_commit, "README.md")
                .expect("read pinned README bytes")
                .as_slice())
        );
        assert_eq!(timeline_audit["unique_completion_commit_count"], 2);
        assert_eq!(timeline_audit["source_path_present_at_completion_count"], 2);
        assert_eq!(timeline_audit["source_path_missing_at_completion_count"], 1);
        assert_eq!(timeline_audit["source_path_not_recorded_count"], 0);
        assert_eq!(
            timeline_audit["events"][2]["source_file_status"],
            "missing_at_completion"
        );
        assert_eq!(timeline_audit["events"][0]["event_key"], "tl_0000_first");
        assert_eq!(
            timeline_audit["events"][1]["source_file_sha256"],
            source_three_sha
        );
        let target_bundle_path = base.join("target-001-bundle.json");
        materialize_temporal_target(&root, &manifest_path, "target-001", &target_bundle_path)
            .expect("only historically available artifacts should be materialized");
        let bundle_bytes = fs::read(&target_bundle_path).expect("read materialized target context");
        let bundle: serde_json::Value =
            serde_json::from_slice(&bundle_bytes).expect("parse target context bundle");
        assert_eq!(
            bundle["payload"]["schema_version"],
            "pitb.temporal-target-bundle.v0"
        );
        let visible = bundle["payload"]["visible_artifacts"]
            .as_array()
            .expect("visible artifacts must be an array");
        let visible_ids: BTreeSet<&str> = visible
            .iter()
            .filter_map(|artifact| artifact["artifact_id"].as_str())
            .collect();
        assert!(visible_ids.contains("statement-001"));
        assert!(visible_ids.contains("proof-000"));
        assert!(!visible_ids.contains("proof-001"));
        assert!(!visible_ids.contains("completion-statement-001"));
        let bundle_text = String::from_utf8(bundle_bytes.clone()).expect("bundle JSON is UTF-8");
        assert!(bundle_text.contains("exact True.intro"));
        assert!(!bundle_text.contains("trivial"));
        verify_temporal_target_bundle(&target_bundle_path)
            .expect("materialized target bundle must verify independently");

        let tampered_bundle_path = base.join("tampered-target-001-bundle.json");
        let mut tampered_bundle: serde_json::Value =
            serde_json::from_slice(&bundle_bytes).expect("parse source bundle");
        tampered_bundle["payload"]["visible_artifacts"][1]["content"] =
            serde_json::Value::String("exact False.elim".to_string());
        fs::write(
            &tampered_bundle_path,
            serde_json::to_vec_pretty(&tampered_bundle).expect("serialize tampered bundle"),
        )
        .expect("write tampered bundle");
        let tamper_error = verify_temporal_target_bundle(&tampered_bundle_path)
            .expect_err("post-materialization context tampering must be detected");
        assert!(format!("{tamper_error:#}").contains("payload digest mismatch"));

        let first_target_bundle_path = base.join("target-000-bundle.json");
        materialize_temporal_target(
            &root,
            &manifest_path,
            "target-000",
            &first_target_bundle_path,
        )
        .expect("the first target must not see its own proof");
        verify_temporal_target_bundle(&first_target_bundle_path)
            .expect("first target bundle must pass structural verification");
        let first_bundle_text =
            fs::read_to_string(&first_target_bundle_path).expect("read first target bundle");
        assert!(!first_bundle_text.contains("exact True.intro"));
        assert!(!first_bundle_text.contains("proof-000"));

        let forbidden_output_path = root.join("must-not-be-created.json");
        let output_error = materialize_temporal_target(
            &root,
            &manifest_path,
            "target-001",
            &forbidden_output_path,
        )
        .expect_err("materialization output must remain outside the checked-out repository");
        assert!(format!("{output_error:#}").contains("outside the pinned repository checkout"));
        assert!(!forbidden_output_path.exists());

        assert!(
            git(&root, &["status", "--porcelain"]).is_empty(),
            "fixture checkout must remain clean after verification"
        );
        fs::remove_dir_all(base).expect("remove temporary replay fixture");
    }

    #[test]
    fn temporal_manifest_accepts_historical_context_and_hides_target_proofs() {
        let manifest = temporal_manifest_fixture();
        validate_temporal_manifest(&manifest)
            .expect("valid chronological evidence and hidden proof receipts should pass");
    }

    #[test]
    fn temporal_manifest_rejects_visible_future_proof() {
        let mut manifest = temporal_manifest_fixture();
        manifest.events[0]
            .visible_artifact_ids
            .push("proof-001".to_string());
        let error = validate_temporal_manifest(&manifest)
            .expect_err("a proof from the future timeline position must be rejected");
        assert!(format!("{error:#}").contains("future-proof leakage"));
    }

    #[test]
    fn temporal_manifest_rejects_its_own_hidden_proof() {
        let mut manifest = temporal_manifest_fixture();
        manifest.events[0]
            .visible_artifact_ids
            .push("proof-000".to_string());
        let error = validate_temporal_manifest(&manifest)
            .expect_err("the target proof must not be visible to the solver");
        assert!(format!("{error:#}").contains("own proof artifact"));
    }

    #[test]
    fn temporal_manifest_rejects_exact_future_proof_copies_under_new_ids() {
        let mut manifest = temporal_manifest_fixture();
        manifest.artifacts.push(TemporalEvidenceArtifact {
            artifact_id: "renamed-copy".to_string(),
            path: "lean/Example.lean".to_string(),
            artifact_kind: "definition".to_string(),
            content_sha256: "d".repeat(64),
            content_start_line: 8,
            content_end_line: 8,
            source_file_sha256: "8".repeat(64),
            source_commit: "1".repeat(40),
            available_at_position: Some(0),
            target_position: None,
            theorem_id: None,
        });
        manifest.events[1]
            .visible_artifact_ids
            .push("renamed-copy".to_string());
        let error = validate_temporal_manifest(&manifest)
            .expect_err("an exact copy of a future proof must be rejected even when renamed");
        assert!(format!("{error:#}").contains("exact proof-content leakage"));
    }

    #[test]
    fn temporal_manifest_rejects_a_proof_marked_available_before_completion() {
        let mut manifest = temporal_manifest_fixture();
        let proof = manifest
            .artifacts
            .iter_mut()
            .find(|artifact| artifact.artifact_id == "proof-001")
            .expect("later target proof exists");
        proof.available_at_position = Some(0);
        let error = validate_temporal_manifest(&manifest)
            .expect_err("a hidden proof cannot be marked as solver-visible evidence");
        assert!(format!("{error:#}").contains("evaluator-only artifact"));
    }

    #[test]
    fn temporal_manifest_rejects_an_unpinned_toolchain_label() {
        let mut manifest = temporal_manifest_fixture();
        manifest.lean_toolchain = "latest".to_string();
        let error = validate_temporal_manifest(&manifest)
            .expect_err("latest is not an immutable toolchain version");
        assert!(format!("{error:#}").contains("must pin Lean and Mathlib"));
    }

    #[test]
    fn temporal_manifest_rejects_statement_drift_at_proof_completion() {
        let mut manifest = temporal_manifest_fixture();
        let completion_statement = manifest
            .artifacts
            .iter_mut()
            .find(|artifact| artifact.artifact_id == "completion-statement-001")
            .expect("completion-time statement exists");
        completion_statement.content_sha256 = "9".repeat(64);
        let error = validate_temporal_manifest(&manifest).expect_err(
            "the proof completion statement must preserve the original target proposition",
        );
        assert!(
            format!("{error:#}")
                .contains("theorem statement changed between target state and proof completion")
        );
    }

    #[test]
    fn temporal_manifest_rejects_hidden_proof_associated_with_wrong_target() {
        let mut manifest = temporal_manifest_fixture();
        let proof = manifest
            .artifacts
            .iter_mut()
            .find(|artifact| artifact.artifact_id == "proof-001")
            .expect("later target proof exists");
        proof.target_position = Some(0);
        let error = validate_temporal_manifest(&manifest)
            .expect_err("a proof body's target association must match its event");
        assert!(format!("{error:#}").contains("hidden proof artifact does not match"));
    }

    #[test]
    fn temporal_manifest_rejects_noncanonical_paths_and_duplicate_ids() {
        let mut bad_path = temporal_manifest_fixture();
        bad_path.artifacts[0].path = "../outside.lean".to_string();
        assert!(validate_temporal_manifest(&bad_path).is_err());

        let mut duplicate = temporal_manifest_fixture();
        duplicate.artifacts[1].artifact_id = "statement-000".to_string();
        assert!(validate_temporal_manifest(&duplicate).is_err());
    }

    #[test]
    fn sha256_matches_standard_known_vector() {
        assert_eq!(
            format!("{:x}", Sha256::digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn source_paths_are_normalized_relative_to_lean() {
        assert_eq!(
            normalize_lean_relative_path("../preprints/a/paper.pdf").unwrap(),
            "preprints/a/paper.pdf"
        );
        assert_eq!(
            normalize_lean_relative_path("OAI/Algebra/Example.lean").unwrap(),
            "lean/OAI/Algebra/Example.lean"
        );
        assert!(normalize_lean_relative_path("../../outside.pdf").is_err());
    }

    #[test]
    fn root_relative_paths_remove_dot_segments() {
        assert_eq!(
            normalize_root_relative_path("./preprints/a/paper.pdf").unwrap(),
            "preprints/a/paper.pdf"
        );
        assert_eq!(
            normalize_root_relative_path("lean/docs/003.md").unwrap(),
            "lean/docs/003.md"
        );
        assert!(normalize_root_relative_path("../outside.pdf").is_err());
    }

    #[test]
    fn family_parser_extracts_titles_links_and_ids() {
        let contents = r#"
<td>**001. First family.**</td>
[manuscript](preprints/first/paper.pdf)
[Lean note](lean/docs/001.md)
**002. Second family.**
[manuscript](preprints/second/paper.pdf)
"#;
        let families = extract_families(contents).expect("parse synthetic catalog");
        assert_eq!(families.len(), 2);
        assert_eq!(families[0].family_id, "001");
        assert_eq!(families[0].title, "First family.");
        assert!(
            families[0]
                .linked_paths
                .contains(&"preprints/first/paper.pdf".to_owned())
        );
        assert_eq!(families[1].family_id, "002");
    }

    #[test]
    fn duplicate_family_identifiers_fail_closed() {
        let contents = "**001. First.**\n[a](one.pdf)\n**001. Duplicate.**\n[b](two.pdf)";
        assert!(extract_families(contents).is_err());
    }

    #[test]
    fn shared_source_path_across_families_is_rejected_to_prevent_split_leakage() {
        let contents = r#"
**001. First family.**
[shared manuscript](preprints/shared/paper.pdf)
**002. Second family.**
[shared manuscript](preprints/shared/paper.pdf)
"#;
        let families = extract_families(contents).expect("parse catalogue");
        let error = map_source_paths_to_families(&families)
            .expect_err("same source path across families must not be silently assigned");
        let message = format!("{error:#}");
        assert!(message.contains("preprints/shared/paper.pdf"));
        assert!(message.contains("001"));
        assert!(message.contains("002"));
    }

    #[test]
    fn source_paths_within_one_family_map_deterministically() {
        let contents = r#"
**001. First family.**
[manuscript](preprints/first/paper.pdf)
[Lean note](lean/docs/001.md)
**002. Second family.**
[manuscript](preprints/second/paper.pdf)
"#;
        let families = extract_families(contents).expect("parse catalogue");
        let map = map_source_paths_to_families(&families).expect("map distinct paths");
        assert_eq!(
            map.get("preprints/first/paper.pdf").map(String::as_str),
            Some("001")
        );
        assert_eq!(map.get("lean/docs/001.md").map(String::as_str), Some("001"));
        assert_eq!(
            map.get("preprints/second/paper.pdf").map(String::as_str),
            Some("002")
        );
    }

    #[test]
    fn catalogue_claims_must_match_parsed_family_headers() {
        let readme = "The current catalogue contains 719 manuscripts organized into 372 families.";
        let claims = parse_catalogue_claims(readme, 372).expect("parse catalogue counts");
        assert_eq!(claims.readme_claimed_manuscript_count, 719);
        assert_eq!(claims.readme_claimed_family_count, 372);
        assert_eq!(claims.parsed_family_header_count, 372);
        assert!(parse_catalogue_claims(readme, 371).is_err());
        assert!(parse_catalogue_claims("no count here", 372).is_err());
    }

    #[test]
    fn family_splits_are_deterministic_and_order_independent() {
        let make_families = || {
            (1..=100)
                .map(|id| FamilyRecord {
                    family_id: format!("{id:03}"),
                    title: format!("Family {id}"),
                    linked_paths: Vec::new(),
                    directly_mapped_formalization_sources: Vec::new(),
                    split: String::new(),
                })
                .collect::<Vec<_>>()
        };

        let mut first = make_families();
        let plan = assign_family_splits(&mut first).expect("assign split");
        let first_map: BTreeMap<_, _> = first
            .iter()
            .map(|family| (family.family_id.clone(), family.split.clone()))
            .collect();

        let mut second = make_families();
        second.reverse();
        let second_plan = assign_family_splits(&mut second).expect("assign split again");
        let second_map: BTreeMap<_, _> = second
            .iter()
            .map(|family| (family.family_id.clone(), family.split.clone()))
            .collect();

        assert_eq!(first_map, second_map);
        assert_eq!(plan.train_family_count, 70);
        assert_eq!(plan.validation_family_count, 15);
        assert_eq!(plan.test_family_count, 15);
        assert_eq!(plan.seed, second_plan.seed);
        assert_eq!(
            plan.partition_digest_sha256,
            second_plan.partition_digest_sha256
        );
        assert_eq!(first_map.len(), 100);
    }

    #[test]
    fn frozen_split_verifier_accepts_exact_rows_and_rejects_mutations() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let digest = format!(
            "{:x}",
            Sha256::digest([48_u8, 48, 49, 9, 116, 114, 97, 105, 110, 10])
        );
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is after Unix epoch")
            .as_nanos();
        let base = std::env::temp_dir();
        let generated_path = base.join(format!(
            "symthaea-proof-insights-generated-{}-{nonce}.json",
            std::process::id()
        ));
        let frozen_path = base.join(format!(
            "symthaea-proof-insights-frozen-{}-{nonce}.json",
            std::process::id()
        ));

        let generated = format!(
            r#"{{"upstream_commit":"abc","source_fingerprints":[{{"path":"README.md","sha256":"aaa","bytes":1}}],"family_split_plan":{{"seed":"seed","algorithm":"algo","train_percent":70,"validation_percent":15,"test_percent":15,"train_family_count":1,"validation_family_count":0,"test_family_count":0,"partition_digest_sha256":"{digest}"}},"families":[{{"family_id":"001","split":"train"}}]}}"#
        );
        let frozen = format!(
            r#"{{"upstream":{{"commit":"abc"}},"source_fingerprints":[{{"path":"README.md","sha256":"aaa","bytes":1}}],"split_plan":{{"seed":"seed","algorithm":"algo","train_percent":70,"validation_percent":15,"test_percent":15,"counts":{{"train":1,"validation":0,"test":0}},"partition_digest_sha256":"{digest}"}},"families":[{{"family_id":"001","split":"train"}}]}}"#
        );
        fs::write(&generated_path, &generated).expect("write generated fixture");
        fs::write(&frozen_path, &frozen).expect("write frozen fixture");
        assert!(verify_split_files(&generated_path, &frozen_path).is_ok());

        let mutated = generated.replace(r#""split":"train""#, r#""split":"test""#);
        fs::write(&generated_path, mutated).expect("write mutation fixture");
        assert!(verify_split_files(&generated_path, &frozen_path).is_err());

        fs::remove_file(&generated_path).expect("remove generated fixture");
        fs::remove_file(&frozen_path).expect("remove frozen fixture");
    }

    #[test]
    fn split_assignment_rejects_small_catalogues() {
        let mut families = vec![FamilyRecord {
            family_id: "001".to_owned(),
            title: "Only one".to_owned(),
            linked_paths: Vec::new(),
            directly_mapped_formalization_sources: Vec::new(),
            split: String::new(),
        }];
        assert!(assign_family_splits(&mut families).is_err());
    }

    #[test]
    fn catalogue_link_inventory_hashes_present_files_and_retains_missing_paths() {
        let root = std::env::temp_dir().join(format!(
            "symthaea-proof-insights-catalogue-links-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("preprints")).expect("create synthetic catalogue");
        let bytes = b"synthetic manuscript bytes";
        fs::write(root.join("preprints/present.pdf"), bytes).expect("write present artifact");

        let contents = r#"
**001. Synthetic family.**
[present manuscript](preprints/present.pdf)
[missing manuscript](preprints/missing.pdf)
"#;
        let families = extract_families(contents).expect("parse synthetic catalogue");
        let mut fingerprints = BTreeMap::new();
        let inventory = inventory_catalogue_links(&root, &families, &mut fingerprints)
            .expect("inventory linked artifacts");

        assert_eq!(inventory.len(), 2);
        let present = inventory
            .iter()
            .find(|artifact| artifact.path == "preprints/present.pdf")
            .expect("present artifact row");
        assert!(present.file_exists);
        assert_eq!(
            present.sha256.as_deref(),
            Some(format!("{:x}", Sha256::digest(bytes)).as_str())
        );
        assert_eq!(present.bytes, Some(bytes.len() as u64));
        assert!(fingerprints.contains_key("preprints/present.pdf"));

        let missing = inventory
            .iter()
            .find(|artifact| artifact.path == "preprints/missing.pdf")
            .expect("missing artifact row");
        assert!(!missing.file_exists);
        assert_eq!(missing.sha256, None);
        assert_eq!(missing.bytes, None);

        fs::remove_dir_all(&root).expect("remove synthetic catalogue");
    }

    #[test]
    fn catalog_paths_escaping_repository_root_are_rejected() {
        let contents = "**001. First.**\n[escape](../outside.pdf)";
        assert!(extract_families(contents).is_err());
    }

    #[test]
    fn formalization_yaml_model_parses_core_fields() {
        let yaml = r#"
version: "v0.4"
project:
  name: "OpenAI math repository"
  license: "Apache-2.0"
sources:
  - title: "Example theorem"
    authors: ["OpenAI"]
    id: ../preprints/example/paper.pdf
    type: article
status:
  scope: "Partial progress."
  main_results:
    - comparator_config: ComparatorChallenges/Example.json
      declaration: OAI.Example.main
      file: OAI/Example/Main.lean
"#;
        let manifest: UpstreamManifest =
            serde_yaml::from_str(yaml).expect("parse synthetic manifest");
        assert_eq!(manifest.version, "v0.4");
        assert_eq!(manifest.sources.len(), 1);
        assert_eq!(manifest.sources[0].title, "Example theorem");
        assert_eq!(manifest.status.main_results.len(), 1);
        assert_eq!(
            manifest.status.main_results[0].declaration.as_deref(),
            Some("OAI.Example.main")
        );
    }
}
