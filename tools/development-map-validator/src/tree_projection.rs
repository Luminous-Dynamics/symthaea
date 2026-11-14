//! Deterministic composition of exact Git tree inventories and source projections.
//!
//! Joins independently produced Cargo-manifest and Rust-syntax projections to
//! one caller-supplied recursive Git tree inventory. It never reads files,
//! follows symlinks/submodules, invokes Git/Cargo/rustc, or executes code.

use std::collections::{BTreeMap, BTreeSet};
use serde::{Deserialize, Serialize};
use crate::cargo_manifest::CargoManifestProjection;
use crate::rust_syntax::RustFileProjection;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct RepositoryTreeAnchor {
    pub repository_full_name: String,
    pub revision: String,
    pub tree_oid: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct RepositoryTreeEntry {
    pub path: String,
    pub mode: String,
    pub object_type: String,
    pub oid: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InventoryCompleteness {
    Complete,
    Partial,
}

impl InventoryCompleteness {
    fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Partial => "partial",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct TreeProjectionCoverage {
    pub cargo_manifests_in_inventory: usize,
    pub cargo_manifests_projected: usize,
    pub rust_sources_in_inventory: usize,
    pub rust_sources_projected: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct RepositoryTreeComposition {
    pub schema_version: String,
    pub authority: String,
    pub source: RepositoryTreeAnchor,
    pub inventory_completeness: String,
    pub tree_entries: Vec<RepositoryTreeEntry>,
    pub cargo_manifests: Vec<CargoManifestProjection>,
    pub rust_files: Vec<RustFileProjection>,
    pub unprojected_cargo_manifests: Vec<String>,
    pub unprojected_rust_sources: Vec<String>,
    pub coverage: TreeProjectionCoverage,
    pub limitations: Vec<String>,
}


#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct RepositoryTreeCompositionInput {
    pub source: RepositoryTreeAnchor,
    pub tree_entries: Vec<RepositoryTreeEntry>,
    pub cargo_manifests: Vec<CargoManifestProjection>,
    pub rust_files: Vec<RustFileProjection>,
    pub inventory_completeness: InventoryCompleteness,
}

/// Parse and compose a JSON request. This only deserializes data and performs
/// structural identity checks; it never fetches paths or executes repository code.
pub fn compose_repository_tree_json(raw_json: &str) -> Result<RepositoryTreeComposition, String> {
    let request: RepositoryTreeCompositionInput = serde_json::from_str(raw_json)
        .map_err(|error| format!("tree composition input JSON parse error: {error}"))?;
    compose_repository_tree(
        request.source,
        request.tree_entries,
        request.cargo_manifests,
        request.rust_files,
        request.inventory_completeness,
    )
}


/// The relevant subset of GitHub's recursive tree API response.
/// Unknown response fields are ignored; all fields used here are mandatory.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct GitHubTreeApiEntry {
    pub path: String,
    pub mode: String,
    #[serde(rename = "type")]
    pub object_type: String,
    pub sha: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct GitHubTreeApiResponse {
    pub sha: String,
    pub truncated: bool,
    pub tree: Vec<GitHubTreeApiEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct RepositoryTreeInventory {
    pub source: RepositoryTreeAnchor,
    /// Complete only when the API response explicitly reports truncated=false.
    /// This does not independently authenticate the API response or commit object.
    pub inventory_completeness: InventoryCompleteness,
    pub api_truncated: bool,
    pub tree_entries: Vec<RepositoryTreeEntry>,
}

/// Convert a saved GitHub recursive-tree JSON response into a pinned inventory.
///
/// source.tree_oid must be taken from the exact commit object for source.revision.
/// This function verifies that the recursive-tree response names that same root
/// tree OID, checks the API truncation flag, validates every path/mode/type/OID
/// tuple and rejects duplicate paths. It performs no network access and cannot
/// authenticate the downloaded response, commit signature, object contents or
/// the caller's commit-to-tree binding.
pub fn collect_github_tree_inventory_json(
    raw_json: &str,
    source: RepositoryTreeAnchor,
) -> Result<RepositoryTreeInventory, String> {
    validate_repository_anchor(&source)?;
    let response: GitHubTreeApiResponse = serde_json::from_str(raw_json)
        .map_err(|error| format!("GitHub tree API JSON parse error: {error}"))?;
    if response.sha != source.tree_oid {
        return Err(format!(
            "recursive tree root mismatch: expected {}, response returned {}",
            source.tree_oid, response.sha
        ));
    }

    let mut seen = BTreeSet::new();
    let mut tree_entries = Vec::with_capacity(response.tree.len());
    for item in response.tree {
        let entry = RepositoryTreeEntry {
            path: item.path,
            mode: item.mode,
            object_type: item.object_type,
            oid: item.sha,
        };
        validate_tree_entry(&entry)?;
        if !seen.insert(entry.path.clone()) {
            return Err(format!("duplicate GitHub tree path: {}", entry.path));
        }
        tree_entries.push(entry);
    }
    tree_entries.sort_by(|left, right| left.path.cmp(&right.path));

    Ok(RepositoryTreeInventory {
        source,
        inventory_completeness: if response.truncated {
            InventoryCompleteness::Partial
        } else {
            InventoryCompleteness::Complete
        },
        api_truncated: response.truncated,
        tree_entries,
    })
}

fn valid_git_oid(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64)
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_repository_anchor(anchor: &RepositoryTreeAnchor) -> Result<(), String> {
    if anchor.repository_full_name.trim().is_empty() {
        return Err("repository_full_name must not be empty".to_owned());
    }
    for (name, value) in [
        ("revision", anchor.revision.as_str()),
        ("tree_oid", anchor.tree_oid.as_str()),
    ] {
        if !valid_git_oid(value) {
            return Err(format!("{name} must be a 40- or 64-character hexadecimal Git object ID"));
        }
    }
    Ok(())
}

fn validate_path(path: &str) -> Result<(), String> {
    if path.trim().is_empty()
        || path.starts_with('/')
        || path.ends_with('/')
        || path.contains('\\')
        || path.contains('\0')
    {
        return Err(format!("non-canonical repository path: {path:?}"));
    }
    if path.split('/').any(|part| part.is_empty() || part == "." || part == "..") {
        return Err(format!("non-canonical repository path: {path:?}"));
    }
    Ok(())
}

fn validate_tree_entry(entry: &RepositoryTreeEntry) -> Result<(), String> {
    validate_path(&entry.path)?;
    if !valid_git_oid(&entry.oid) {
        return Err(format!("tree entry {} has an invalid Git object ID", entry.path));
    }
    match (entry.mode.as_str(), entry.object_type.as_str()) {
        ("100644" | "100755" | "120000", "blob")
        | ("040000", "tree")
        | ("160000", "commit") => Ok(()),
        _ => Err(format!(
            "tree entry {} has inconsistent or unsupported mode/type {}/{}",
            entry.path, entry.mode, entry.object_type
        )),
    }
}

fn is_regular_blob(entry: &RepositoryTreeEntry) -> bool {
    entry.object_type == "blob" && matches!(entry.mode.as_str(), "100644" | "100755")
}

fn validate_projection_anchor(
    repository: &str,
    revision: &str,
    tree_oid: &str,
    source_repository: &str,
    source_revision: &str,
    source_tree_oid: &str,
    source_path: &str,
    source_blob_oid: &str,
) -> Result<(), String> {
    if source_repository != repository
        || source_revision != revision
        || source_tree_oid != tree_oid
    {
        return Err(format!(
            "projection at {source_path} is not anchored to the exact repository revision and tree"
        ));
    }
    validate_path(source_path)?;
    if !valid_git_oid(source_blob_oid) {
        return Err(format!("projection at {source_path} has an invalid blob OID"));
    }
    Ok(())
}

/// Compose projections against one exact recursive Git tree inventory.
///
/// The caller supplies the tree entries including path, mode, type and OID.
/// This function checks equality and coverage, but does not authenticate the
/// Git objects or prove the recursive inventory is exhaustive. Complete mode
/// requires a projection for every regular Cargo.toml and Rust .rs file.
pub fn compose_repository_tree(
    source: RepositoryTreeAnchor,
    mut tree_entries: Vec<RepositoryTreeEntry>,
    mut cargo_manifests: Vec<CargoManifestProjection>,
    mut rust_files: Vec<RustFileProjection>,
    completeness: InventoryCompleteness,
) -> Result<RepositoryTreeComposition, String> {
    validate_repository_anchor(&source)?;

    let mut by_path = BTreeMap::new();
    for entry in &tree_entries {
        validate_tree_entry(entry)?;
        if by_path.insert(entry.path.clone(), entry).is_some() {
            return Err(format!("duplicate Git tree path: {}", entry.path));
        }
    }

    // Never omit path-shaped source files just because they are symlinks,
    // submodules or directories. This projector does not follow those objects.
    for entry in &tree_entries {
        let candidate = entry.path.rsplit('/').next().unwrap_or(&entry.path);
        if (candidate == "Cargo.toml" || candidate.ends_with(".rs"))
            && !is_regular_blob(entry)
        {
            return Err(format!(
                "source candidate {} is not a regular blob; symlinks, trees and gitlinks are not followed",
                entry.path
            ));
        }
    }

    let mut projected_manifests = BTreeSet::new();
    for manifest in &cargo_manifests {
        let anchor = &manifest.source;
        validate_projection_anchor(
            &source.repository_full_name, &source.revision, &source.tree_oid,
            &anchor.repository_full_name, &anchor.revision, &anchor.tree_oid,
            &anchor.manifest_path, &anchor.blob_oid,
        )?;
        if anchor.manifest_path.rsplit('/').next() != Some("Cargo.toml") {
            return Err(format!("Cargo manifest projection path must end in Cargo.toml: {}", anchor.manifest_path));
        }
        let entry = by_path.get(&anchor.manifest_path)
            .ok_or_else(|| format!("manifest path is absent from the supplied Git tree: {}", anchor.manifest_path))?;
        if !is_regular_blob(entry) {
            return Err(format!("manifest path is not a regular blob: {}", anchor.manifest_path));
        }
        if entry.oid != anchor.blob_oid {
            return Err(format!(
                "manifest blob mismatch at {}: tree has {}, projection has {}",
                anchor.manifest_path, entry.oid, anchor.blob_oid
            ));
        }
        if !projected_manifests.insert(anchor.manifest_path.clone()) {
            return Err(format!("duplicate Cargo manifest projection: {}", anchor.manifest_path));
        }
    }

    let mut projected_rust = BTreeSet::new();
    for file in &rust_files {
        let anchor = &file.source;
        validate_projection_anchor(
            &source.repository_full_name, &source.revision, &source.tree_oid,
            &anchor.repository_full_name, &anchor.revision, &anchor.tree_oid,
            &anchor.source_path, &anchor.blob_oid,
        )?;
        if !anchor.source_path.ends_with(".rs") {
            return Err(format!("Rust syntax projection path must end in .rs: {}", anchor.source_path));
        }
        let entry = by_path.get(&anchor.source_path)
            .ok_or_else(|| format!("Rust source path is absent from the supplied Git tree: {}", anchor.source_path))?;
        if !is_regular_blob(entry) {
            return Err(format!("Rust source path is not a regular blob: {}", anchor.source_path));
        }
        if entry.oid != anchor.blob_oid {
            return Err(format!(
                "Rust source blob mismatch at {}: tree has {}, projection has {}",
                anchor.source_path, entry.oid, anchor.blob_oid
            ));
        }
        if !projected_rust.insert(anchor.source_path.clone()) {
            return Err(format!("duplicate Rust syntax projection: {}", anchor.source_path));
        }
    }

    let mut manifest_paths = tree_entries.iter()
        .filter(|entry| is_regular_blob(entry) && entry.path.rsplit('/').next() == Some("Cargo.toml"))
        .map(|entry| entry.path.clone()).collect::<Vec<_>>();
    let mut rust_paths = tree_entries.iter()
        .filter(|entry| is_regular_blob(entry) && entry.path.ends_with(".rs"))
        .map(|entry| entry.path.clone()).collect::<Vec<_>>();
    manifest_paths.sort();
    rust_paths.sort();

    let unprojected_cargo_manifests = manifest_paths.iter()
        .filter(|path| !projected_manifests.contains(*path))
        .cloned().collect::<Vec<_>>();
    let unprojected_rust_sources = rust_paths.iter()
        .filter(|path| !projected_rust.contains(*path))
        .cloned().collect::<Vec<_>>();

    if completeness == InventoryCompleteness::Complete
        && (!unprojected_cargo_manifests.is_empty() || !unprojected_rust_sources.is_empty())
    {
        return Err(format!(
            "complete inventory has unprojected source candidates: {} Cargo.toml and {} Rust files",
            unprojected_cargo_manifests.len(), unprojected_rust_sources.len()
        ));
    }

    tree_entries.sort_by(|left, right| left.path.cmp(&right.path));
    cargo_manifests.sort_by(|left, right| left.source.manifest_path.cmp(&right.source.manifest_path));
    rust_files.sort_by(|left, right| left.source.source_path.cmp(&right.source.source_path));

    let mut limitations = vec![
        "Git object IDs are syntax-checked and compared across inputs; this function does not authenticate objects or compute Git hashes.".to_owned(),
        "Inventory completeness is caller-asserted; the function cannot prove that the supplied tree listing contains every repository path.".to_owned(),
        "Cargo workspace globs, inherited values, feature expressions, implicit targets and build scripts remain unresolved.".to_owned(),
        "Rust imports, cfg predicates, macros, out-of-line modules, compiler name resolution and field/variant inventories remain unresolved.".to_owned(),
        "Symlinks and submodule gitlinks are explicit boundaries and are never followed.".to_owned(),
        "A source projection is not evidence that a package builds, tests pass, a capability works or a cross-repository contract is compatible.".to_owned(),
    ];
    limitations.sort();

    Ok(RepositoryTreeComposition {
        schema_version: "0.1.0".to_owned(),
        authority: "composed_source_anchors".to_owned(),
        source,
        inventory_completeness: completeness.as_str().to_owned(),
        tree_entries,
        cargo_manifests,
        rust_files,
        unprojected_cargo_manifests,
        unprojected_rust_sources,
        coverage: TreeProjectionCoverage {
            cargo_manifests_in_inventory: manifest_paths.len(),
            cargo_manifests_projected: projected_manifests.len(),
            rust_sources_in_inventory: rust_paths.len(),
            rust_sources_projected: projected_rust.len(),
        },
        limitations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cargo_manifest::{project_cargo_manifest, ManifestAnchor};
    use crate::rust_syntax::{project_rust_file, RustFileAnchor};

    fn source_anchor() -> RepositoryTreeAnchor {
        RepositoryTreeAnchor {
            repository_full_name: "example/repo".to_owned(),
            revision: "a".repeat(40),
            tree_oid: "b".repeat(40),
        }
    }

    fn entry(path: &str, mode: &str, object_type: &str, oid: char) -> RepositoryTreeEntry {
        RepositoryTreeEntry {
            path: path.to_owned(), mode: mode.to_owned(), object_type: object_type.to_owned(),
            oid: oid.to_string().repeat(40),
        }
    }

    fn manifest(path: &str, blob: char) -> CargoManifestProjection {
        let source = source_anchor();
        project_cargo_manifest(
            "[workspace]\nmembers = [\"crates/*\"]\n",
            ManifestAnchor {
                repository_full_name: source.repository_full_name,
                revision: source.revision,
                tree_oid: source.tree_oid,
                manifest_path: path.to_owned(),
                blob_oid: blob.to_string().repeat(40),
            },
        ).expect("manifest projection")
    }

    fn rust_file(path: &str, blob: char) -> RustFileProjection {
        let source = source_anchor();
        project_rust_file(
            "pub struct Example;\n",
            RustFileAnchor {
                repository_full_name: source.repository_full_name,
                revision: source.revision,
                tree_oid: source.tree_oid,
                source_path: path.to_owned(),
                blob_oid: blob.to_string().repeat(40),
            },
        ).expect("Rust projection")
    }

    fn base_entries() -> Vec<RepositoryTreeEntry> {
        vec![
            entry("Cargo.toml", "100644", "blob", 'c'),
            entry("src/lib.rs", "100644", "blob", 'd'),
            entry("src", "040000", "tree", 'e'),
            entry("README.md", "100644", "blob", 'f'),
        ]
    }



    fn api_tree_json(root: &str, truncated: bool, entries: serde_json::Value) -> String {
        serde_json::json!({
            "sha": root,
            "truncated": truncated,
            "tree": entries
        }).to_string()
    }

    #[test]
    fn collects_exact_nontruncated_github_tree_as_complete() {
        let root = "b".repeat(40);
        let input = api_tree_json(&root, false, serde_json::json!([
            {"path":"src/lib.rs","mode":"100644","type":"blob","sha":"d".repeat(40)},
            {"path":"src","mode":"040000","type":"tree","sha":"e".repeat(40)}
        ]));
        let result = collect_github_tree_inventory_json(&input, source_anchor()).expect("inventory");
        assert_eq!(result.inventory_completeness, InventoryCompleteness::Complete);
        assert!(!result.api_truncated);
        assert_eq!(result.tree_entries[0].path, "src");
        assert_eq!(result.tree_entries[1].path, "src/lib.rs");
    }

    #[test]
    fn marks_truncated_github_tree_as_partial_not_complete() {
        let root = "b".repeat(40);
        let input = api_tree_json(&root, true, serde_json::json!([
            {"path":"Cargo.toml","mode":"100644","type":"blob","sha":"c".repeat(40)}
        ]));
        let result = collect_github_tree_inventory_json(&input, source_anchor()).expect("partial inventory");
        assert_eq!(result.inventory_completeness, InventoryCompleteness::Partial);
        assert!(result.api_truncated);
    }

    #[test]
    fn rejects_recursive_tree_root_mismatch() {
        let input = api_tree_json(&"9".repeat(40), false, serde_json::json!([]));
        let error = collect_github_tree_inventory_json(&input, source_anchor()).expect_err("root mismatch");
        assert!(error.contains("recursive tree root mismatch"));
    }

    #[test]
    fn rejects_duplicate_github_tree_paths() {
        let root = "b".repeat(40);
        let input = api_tree_json(&root, false, serde_json::json!([
            {"path":"src/lib.rs","mode":"100644","type":"blob","sha":"d".repeat(40)},
            {"path":"src/lib.rs","mode":"100644","type":"blob","sha":"d".repeat(40)}
        ]));
        let error = collect_github_tree_inventory_json(&input, source_anchor()).expect_err("duplicate path");
        assert!(error.contains("duplicate GitHub tree path"));
    }

    #[test]
    fn rejects_malformed_tree_entry_mode_type_and_object_id() {
        let root = "b".repeat(40);
        let cases = [
            serde_json::json!([{"path":"src/lib.rs","mode":"100644","type":"tree","sha":"d".repeat(40)}]),
            serde_json::json!([{"path":"src/lib.rs","mode":"100644","type":"blob","sha":"not-a-sha"}]),
            serde_json::json!([{"path":"../lib.rs","mode":"100644","type":"blob","sha":"d".repeat(40)}])
        ];
        for entries in cases {
            let input = api_tree_json(&root, false, entries);
            assert!(collect_github_tree_inventory_json(&input, source_anchor()).is_err());
        }
    }

    #[test]
    fn rejects_missing_or_nonboolean_truncation_flag() {
        let missing = serde_json::json!({"sha":"b".repeat(40),"tree":[]}).to_string();
        assert!(collect_github_tree_inventory_json(&missing, source_anchor()).is_err());
        let invalid = serde_json::json!({"sha":"b".repeat(40),"truncated":"false","tree":[]}).to_string();
        assert!(collect_github_tree_inventory_json(&invalid, source_anchor()).is_err());
    }

    #[test]
    fn accepts_a_serialized_json_composition_request() {
        let request = RepositoryTreeCompositionInput {
            source: source_anchor(),
            tree_entries: base_entries(),
            cargo_manifests: vec![manifest("Cargo.toml", 'c')],
            rust_files: vec![rust_file("src/lib.rs", 'd')],
            inventory_completeness: InventoryCompleteness::Complete,
        };
        let json = serde_json::to_string(&request).expect("serialize request");
        let result = compose_repository_tree_json(&json).expect("compose JSON request");
        assert_eq!(result.coverage.cargo_manifests_projected, 1);
        assert_eq!(result.coverage.rust_sources_projected, 1);
    }

    #[test]
    fn composes_exact_anchored_files_and_sorts_paths() {
        let result = compose_repository_tree(
            source_anchor(), base_entries(), vec![manifest("Cargo.toml", 'c')],
            vec![rust_file("src/lib.rs", 'd')], InventoryCompleteness::Complete,
        ).expect("composition");
        assert_eq!(result.authority, "composed_source_anchors");
        assert_eq!(result.coverage.cargo_manifests_in_inventory, 1);
        assert_eq!(result.coverage.cargo_manifests_projected, 1);
        assert_eq!(result.coverage.rust_sources_in_inventory, 1);
        assert_eq!(result.coverage.rust_sources_projected, 1);
        assert_eq!(result.tree_entries[0].path, "Cargo.toml");
        assert_eq!(result.cargo_manifests[0].source.manifest_path, "Cargo.toml");
        assert_eq!(result.rust_files[0].source.source_path, "src/lib.rs");
    }

    #[test]
    fn rejects_manifest_blob_mismatch() {
        let error = compose_repository_tree(
            source_anchor(), base_entries(), vec![manifest("Cargo.toml", '9')],
            vec![rust_file("src/lib.rs", 'd')], InventoryCompleteness::Partial,
        ).expect_err("blob mismatch");
        assert!(error.contains("manifest blob mismatch"));
    }

    #[test]
    fn rejects_projection_from_another_revision() {
        let mut foreign = manifest("Cargo.toml", 'c');
        foreign.source.revision = "9".repeat(40);
        let error = compose_repository_tree(
            source_anchor(), base_entries(), vec![foreign], vec![],
            InventoryCompleteness::Partial,
        ).expect_err("foreign revision");
        assert!(error.contains("exact repository revision and tree"));
    }

    #[test]
    fn rejects_duplicate_tree_paths() {
        let mut entries = base_entries();
        entries.push(entry("src/lib.rs", "100644", "blob", 'd'));
        let error = compose_repository_tree(
            source_anchor(), entries, vec![], vec![], InventoryCompleteness::Partial,
        ).expect_err("duplicate path");
        assert!(error.contains("duplicate Git tree path"));
    }

    #[test]
    fn rejects_parent_traversal_and_absolute_paths() {
        for invalid in ["/src/lib.rs", "../src/lib.rs", "src/../lib.rs", "src//lib.rs"] {
            let error = compose_repository_tree(
                source_anchor(), vec![entry(invalid, "100644", "blob", 'd')],
                vec![], vec![], InventoryCompleteness::Partial,
            ).expect_err("invalid path");
            assert!(error.contains("non-canonical repository path"));
        }
    }

    #[test]
    fn records_unprojected_files_without_claiming_complete_coverage() {
        let result = compose_repository_tree(
            source_anchor(), base_entries(), vec![manifest("Cargo.toml", 'c')], vec![],
            InventoryCompleteness::Partial,
        ).expect("partial composition");
        assert_eq!(result.unprojected_rust_sources, vec!["src/lib.rs".to_owned()]);
        assert_eq!(result.coverage.rust_sources_projected, 0);
        assert_eq!(result.inventory_completeness, "partial");
    }

    #[test]
    fn complete_inventory_requires_every_cargo_and_rust_projection() {
        let error = compose_repository_tree(
            source_anchor(), base_entries(), vec![], vec![], InventoryCompleteness::Complete,
        ).expect_err("incomplete projection coverage");
        assert!(error.contains("complete inventory has unprojected source candidates"));
    }

    #[test]
    fn rejects_projection_path_absent_from_tree() {
        let error = compose_repository_tree(
            source_anchor(), base_entries(), vec![], vec![rust_file("src/missing.rs", 'd')],
            InventoryCompleteness::Partial,
        ).expect_err("missing source");
        assert!(error.contains("absent from the supplied Git tree"));
    }

    #[test]
    fn refuses_symlinked_source_candidates() {
        let error = compose_repository_tree(
            source_anchor(), vec![entry("src/lib.rs", "120000", "blob", 'd')],
            vec![], vec![], InventoryCompleteness::Partial,
        ).expect_err("symlink boundary");
        assert!(error.contains("symlinks, trees and gitlinks are not followed"));
    }

    #[test]
    fn refuses_submodule_source_candidates() {
        let error = compose_repository_tree(
            source_anchor(), vec![entry("vendor/external.rs", "160000", "commit", 'd')],
            vec![], vec![], InventoryCompleteness::Partial,
        ).expect_err("gitlink boundary");
        assert!(error.contains("symlinks, trees and gitlinks are not followed"));
    }

    #[test]
    fn rejects_inconsistent_git_mode_and_object_type() {
        let error = compose_repository_tree(
            source_anchor(), vec![entry("src/lib.rs", "100644", "tree", 'd')],
            vec![], vec![], InventoryCompleteness::Partial,
        ).expect_err("invalid mode/type");
        assert!(error.contains("inconsistent or unsupported mode/type"));
    }

    #[test]
    fn rejects_duplicate_manifest_projections() {
        let error = compose_repository_tree(
            source_anchor(), base_entries(),
            vec![manifest("Cargo.toml", 'c'), manifest("Cargo.toml", 'c')],
            vec![], InventoryCompleteness::Partial,
        ).expect_err("duplicate manifest");
        assert!(error.contains("duplicate Cargo manifest projection"));
    }

    #[test]
    fn rejects_duplicate_rust_projections() {
        let error = compose_repository_tree(
            source_anchor(), base_entries(), vec![],
            vec![rust_file("src/lib.rs", 'd'), rust_file("src/lib.rs", 'd')],
            InventoryCompleteness::Partial,
        ).expect_err("duplicate Rust projection");
        assert!(error.contains("duplicate Rust syntax projection"));
    }

    #[test]
    fn rejects_manifest_projection_on_non_manifest_path() {
        let error = compose_repository_tree(
            source_anchor(), vec![entry("other.toml", "100644", "blob", 'c')],
            vec![manifest("other.toml", 'c')], vec![], InventoryCompleteness::Partial,
        ).expect_err("wrong manifest extension");
        assert!(error.contains("must end in Cargo.toml"));
    }

    #[test]
    fn rejects_rust_projection_on_non_rust_path() {
        let mut projection = rust_file("src/lib.rs", 'd');
        projection.source.source_path = "src/lib.txt".to_owned();
        let error = compose_repository_tree(
            source_anchor(), base_entries(), vec![], vec![projection],
            InventoryCompleteness::Partial,
        ).expect_err("wrong source extension");
        assert!(error.contains("must end in .rs"));
    }

    #[test]
    fn output_is_deterministic_under_input_entry_reordering() {
        let mut reversed = base_entries();
        reversed.reverse();
        let left = compose_repository_tree(
            source_anchor(), base_entries(), vec![manifest("Cargo.toml", 'c')],
            vec![rust_file("src/lib.rs", 'd')], InventoryCompleteness::Complete,
        ).expect("left composition");
        let right = compose_repository_tree(
            source_anchor(), reversed, vec![manifest("Cargo.toml", 'c')],
            vec![rust_file("src/lib.rs", 'd')], InventoryCompleteness::Complete,
        ).expect("right composition");
        assert_eq!(left, right);
    }
}
