//! Versioned inventory for known external-effect entry points.
//!
//! This command checks registry integrity and source anchors. It is not a
//! complete Rust/FFI/macro/indirect-effect scanner and does not qualify any
//! execution surface. An inventory-integrity PASS must never be reported as a
//! security or repository-wide no-bypass PASS.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path, PathBuf},
    process::Command,
};

const REGISTRY_ID: &str = "symthaea.effect_surface_registry.v1";
const REGISTRY_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Registry {
    registry_id: String,
    version: u32,
    source_baseline: SourceBaseline,
    entries: Vec<SurfaceEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceBaseline {
    repository: String,
    baseline_ref: String,
    baseline_commit: String,
    audited_at: String,
    completeness: String,
    scope_note: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SurfaceEntry {
    id: String,
    source_path: String,
    entrypoint_symbol: String,
    source_anchor: String,
    effect_classes: Vec<String>,
    execution_modes: Vec<String>,
    disposition: String,
    owner_issue: String,
    authority_boundary: String,
    test_seam: String,
    negative_control: String,
    positive_control: String,
    tests_status: String,
    known_limitations: Vec<String>,
    qualification_evidence: Option<QualificationEvidence>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QualificationEvidence {
    subject_commit: String,
    registry_digest_sha256: String,
    test_receipt_sha256: String,
    verification_profile: String,
}

struct Summary {
    entries: usize,
    source_anchors_checked: usize,
    qualified_entries: usize,
    unqualified_entries: usize,
}

pub fn check(root: &Path, registry_path: &Path, require_qualified: bool) -> Result<()> {
    let raw = fs::read_to_string(registry_path)
        .with_context(|| format!("read effect-surface registry: {}", registry_path.display()))?;
    let registry: Registry =
        serde_json::from_str(&raw).context("parse effect-surface registry JSON")?;
    let summary = validate(root, &registry)?;

    println!("registry_id={}", registry.registry_id);
    println!("registry_version={}", registry.version);
    println!(
        "registry_baseline={}@{}",
        registry.source_baseline.baseline_ref, registry.source_baseline.baseline_commit
    );
    println!("registry_integrity=PASS");
    println!(
        "source_anchors_checked={}/{}",
        summary.source_anchors_checked, summary.entries
    );
    println!("entry_count={}", summary.entries);
    println!("qualified_entries={}", summary.qualified_entries);
    println!("unqualified_entries={}", summary.unqualified_entries);
    println!(
        "inventory_completeness={}",
        registry.source_baseline.completeness
    );
    println!("security_qualification=NOT_EVALUATED_BY_THIS_COMMAND");
    println!("repository_wide_no_bypass=NOT_CLAIMED");

    if require_qualified && summary.unqualified_entries > 0 {
        bail!(
            "--require-qualified failed: {} of {} registry entries lack recorded qualification evidence",
            summary.unqualified_entries,
            summary.entries
        );
    }
    Ok(())
}

fn validate(root: &Path, registry: &Registry) -> Result<Summary> {
    if registry.registry_id != REGISTRY_ID {
        bail!(
            "unsupported registry_id {:?}; expected {:?}",
            registry.registry_id,
            REGISTRY_ID
        );
    }
    if registry.version != REGISTRY_VERSION {
        bail!(
            "unsupported registry version {}; expected {}",
            registry.version,
            REGISTRY_VERSION
        );
    }
    if registry.source_baseline.repository != "Luminous-Dynamics/symthaea" {
        bail!("registry repository identity does not match this project");
    }
    if registry.source_baseline.baseline_ref.trim().is_empty()
        || registry.source_baseline.audited_at.trim().is_empty()
        || registry.source_baseline.scope_note.trim().is_empty()
    {
        bail!("source baseline must identify its ref, audit date, and scope limits");
    }
    if registry.source_baseline.completeness != "initial_known_surfaces_only" {
        bail!("v1 registry must preserve its explicit incomplete-inventory claim boundary");
    }
    if !is_sha(&registry.source_baseline.baseline_commit, 40) {
        bail!("source_baseline.baseline_commit must be a 40-character hexadecimal commit SHA");
    }
    if registry.entries.is_empty() {
        bail!("effect-surface registry must not be empty");
    }

    let mut ids = BTreeSet::new();
    let mut source_anchors_checked = 0usize;
    let mut qualified_entries = 0usize;

    for entry in &registry.entries {
        if entry.id.trim().is_empty() || !ids.insert(entry.id.clone()) {
            bail!("empty or duplicate effect-surface id: {:?}", entry.id);
        }
        if entry.source_path.trim().is_empty()
            || entry.entrypoint_symbol.trim().is_empty()
            || entry.source_anchor.trim().is_empty()
            || entry.owner_issue.trim().is_empty()
            || entry.authority_boundary.trim().is_empty()
            || entry.test_seam.trim().is_empty()
            || entry.negative_control.trim().is_empty()
            || entry.positive_control.trim().is_empty()
            || entry.tests_status.trim().is_empty()
        {
            bail!("surface {} is missing required descriptive fields", entry.id);
        }
        if !is_safe_relative_path(&entry.source_path) {
            bail!(
                "surface {} has unsafe repository-relative source_path {:?}",
                entry.id,
                entry.source_path
            );
        }
        if entry.effect_classes.is_empty() || entry.execution_modes.is_empty() {
            bail!(
                "surface {} must declare at least one effect class and execution mode",
                entry.id
            );
        }
        if entry.known_limitations.is_empty()
            || entry.known_limitations.iter().any(|item| item.trim().is_empty())
        {
            bail!(
                "surface {} must state at least one known limitation",
                entry.id
            );
        }
        if !matches!(
            entry.disposition.as_str(),
            "migration_required" | "partial_guard" | "unqualified" | "qualified"
        ) {
            bail!(
                "surface {} has unsupported disposition {:?}",
                entry.id,
                entry.disposition
            );
        }

        let source_path = root.join(&entry.source_path);
        let source = fs::read_to_string(&source_path).with_context(|| {
            format!(
                "surface {} source does not exist or is unreadable: {}",
                entry.id,
                source_path.display()
            )
        })?;
        if !source.contains(&entry.source_anchor) {
            bail!(
                "surface {} source anchor {:?} not found in {} (entrypoint recorded as {})",
                entry.id,
                entry.source_anchor,
                entry.source_path,
                entry.entrypoint_symbol
            );
        }
        source_anchors_checked += 1;

        match (&entry.disposition[..], &entry.qualification_evidence) {
            ("qualified", Some(evidence)) => {
                if !is_sha(&evidence.subject_commit, 40)
                    || !is_sha(&evidence.registry_digest_sha256, 64)
                    || !is_sha(&evidence.test_receipt_sha256, 64)
                    || evidence.verification_profile.trim().is_empty()
                {
                    bail!(
                        "qualified surface {} requires exact subject SHA, registry SHA-256, test-receipt SHA-256, and verification profile",
                        entry.id
                    );
                }
                qualified_entries += 1;
            }
            ("qualified", None) => {
                bail!("surface {} is marked qualified without qualification evidence", entry.id)
            }
            (_, Some(_)) => {
                bail!(
                    "surface {} carries qualification evidence but is not marked qualified; resolve the disposition explicitly",
                    entry.id
                )
            }
            (_, None) => {}
        }
    }

    Ok(Summary {
        entries: registry.entries.len(),
        source_anchors_checked,
        qualified_entries,
        unqualified_entries: registry.entries.len() - qualified_entries,
    })
}

fn is_safe_relative_path(value: &str) -> bool {
    let path = Path::new(value);
    !value.trim().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn is_sha(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[allow(dead_code)]
fn working_tree_head(root: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|value| value.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::{is_safe_relative_path, is_sha};

    #[test]
    fn source_paths_must_stay_inside_repository() {
        assert!(is_safe_relative_path("src/action/mod.rs"));
        assert!(!is_safe_relative_path("../outside.rs"));
        assert!(!is_safe_relative_path("/tmp/source.rs"));
        assert!(!is_safe_relative_path(""));
        assert!(!is_safe_relative_path("src/../secrets.rs"));
    }

    #[test]
    fn digests_require_exact_hex_lengths() {
        assert!(is_sha("a".repeat(40).as_str(), 40));
        assert!(is_sha("a".repeat(64).as_str(), 64));
        assert!(!is_sha("g".repeat(40).as_str(), 40));
        assert!(!is_sha("a".repeat(39).as_str(), 40));
    }
}
