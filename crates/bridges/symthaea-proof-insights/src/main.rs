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

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, File},
    io::Read,
    path::{Component, Path, PathBuf},
    process::{Command, ExitCode},
};

use anyhow::{bail, Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const EXPECTED_UPSTREAM_REPOSITORY: &str = "https://github.com/openai/math";
const EXPECTED_MANIFEST_VERSION: &str = "v0.4";
const FAMILY_SPLIT_SEED: &str = "pitb-v0-openai-math-2026-10-19";
const FAMILY_SPLIT_ALGORITHM: &str = "sha256(seed || NUL || family_id), rank, 70/15/15";

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
    families_with_directly_mapped_formalization_sources: usize,
    main_result_declaration_count: usize,
    main_result_artifacts_missing: Vec<String>,
    unmapped_formalization_source_ids: Vec<String>,
    missing_formalization_source_ids: Vec<String>,
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
            args.first().map(String::as_str).unwrap_or("symthaea-proof-insights")
        );
    }

    let root = PathBuf::from(&args[1])
        .canonicalize()
        .with_context(|| format!("cannot resolve source root {}", args[1]))?;
    let expected_commit = args[2].trim().to_ascii_lowercase();
    if expected_commit.len() != 40
        || !expected_commit.chars().all(|character| character.is_ascii_hexdigit())
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
    let comparator_readme_path = root.join("lean/ComparatorChallenges/README.md");

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
        let mapped_family_id = source_path_to_family.get(&repository_relative_path).cloned();
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

fn verify_split_files(generated_path: &Path, frozen_path: &Path) -> Result<()> {
    let generated_text = fs::read_to_string(generated_path)
        .with_context(|| format!("cannot read generated source index {}", generated_path.display()))?;
    let frozen_text = fs::read_to_string(frozen_path)
        .with_context(|| format!("cannot read frozen split manifest {}", frozen_path.display()))?;
    let generated: serde_json::Value =
        serde_json::from_str(&generated_text).context("generated index is not valid JSON")?;
    let frozen: serde_json::Value =
        serde_json::from_str(&frozen_text).context("frozen split manifest is not valid JSON")?;

    let generated_commit = generated
        .get("upstream_commit").and_then(serde_json::Value::as_str)
        .context("generated index is missing upstream_commit")?;
    let frozen_commit = frozen
        .get("upstream").and_then(|value| value.get("commit")).and_then(serde_json::Value::as_str)
        .context("frozen split manifest is missing upstream.commit")?;
    if generated_commit != frozen_commit {
        bail!("source revision mismatch: generated index {generated_commit}, frozen manifest {frozen_commit}");
    }

    let generated_digest = generated
        .get("family_split_plan").and_then(|value| value.get("partition_digest_sha256"))
        .and_then(serde_json::Value::as_str)
        .context("generated index has no family_split_plan.partition_digest_sha256")?;
    let frozen_digest = frozen
        .get("split_plan").and_then(|value| value.get("partition_digest_sha256"))
        .and_then(serde_json::Value::as_str)
        .context("frozen manifest has no split_plan.partition_digest_sha256")?;
    if generated_digest != frozen_digest {
        bail!("family split digest mismatch: generated {generated_digest}, frozen {frozen_digest}");
    }

    let rows = |value: &serde_json::Value, field: &str| -> Result<BTreeMap<String, String>> {
        let entries = value.get(field).and_then(serde_json::Value::as_array)
            .with_context(|| format!("missing array field {field}"))?;
        let mut result = BTreeMap::new();
        for entry in entries {
            let id = entry.get("family_id").and_then(serde_json::Value::as_str)
                .context("family record missing family_id")?;
            let split = entry.get("split").and_then(serde_json::Value::as_str)
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
        let generated_only: Vec<_> = generated_rows.keys().filter(|key| !frozen_rows.contains_key(*key)).cloned().collect();
        let frozen_only: Vec<_> = frozen_rows.keys().filter(|key| !generated_rows.contains_key(*key)).cloned().collect();
        bail!(
            "family assignment mismatch (generated {}, frozen {}); generated-only IDs: {:?}; frozen-only IDs: {:?}",
            generated_rows.len(), frozen_rows.len(), generated_only, frozen_only
        );
    }

    eprintln!(
        "PASS: {} family assignments match frozen digest {} for source {}",
        generated_rows.len(), frozen_digest, generated_commit
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

fn extract_families(contents: &str) -> Result<Vec<FamilyRecord>> {
    // The pinned CONTENTS.md presents each result family as **NNN. Title**.
    // Anything outside this known catalogue shape is treated as a failure.
    let header_regex = Regex::new(r"\*\*(\d{3})\.\s*([^*]+?)\*\*")
        .context("compile family-header expression")?;
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
            .map(|path| normalize_root_relative_path(path))
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

fn map_source_paths_to_families(
    families: &[FamilyRecord],
) -> Result<BTreeMap<String, String>> {
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
                    source_path_to_family
                        .insert(linked_path.clone(), family.family_id.clone());
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
    ranked.sort_by(|left, right| {
        left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1))
    });

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
        partition_material.push('\\t');
        partition_material.push_str(split);
        partition_material.push('\\n');
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

fn normalize_root_relative_path(path: &str) -> Result<String> {
    normalize_with_base(std::iter::empty::<&str>(), path)
}

fn normalize_with_base<'a>(
    base: impl IntoIterator<Item = &'a str>,
    path: &str,
) -> Result<String> {
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
        assert!(families[0]
            .linked_paths
            .contains(&"preprints/first/paper.pdf".to_owned()));
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
        assert_eq!(map.get("preprints/first/paper.pdf").map(String::as_str), Some("001"));
        assert_eq!(map.get("lean/docs/001.md").map(String::as_str), Some("001"));
        assert_eq!(map.get("preprints/second/paper.pdf").map(String::as_str), Some("002"));
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
        assert_eq!(first_map.len(), 100);
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
