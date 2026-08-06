//! Crate truth registry — the machine-readable answer to "which of these 200+ crates is
//! actually trustworthy, and on what evidence?"
//!
//! ## Why this exists
//!
//! The workspace has grown past the point where its own contents can be held in one head. A
//! crate's *existence* says nothing about whether it is current, validated, safe to compose, or
//! safe to put on a production path — but a reader with no other signal will reasonably assume
//! it does. That assumption is the actual risk: not any single defect, but a body of work whose
//! trustworthy parts are indistinguishable from its exploratory parts.
//!
//! **The rule this file enforces: a crate's existence does not imply endorsement.**
//!
//! ## Design
//!
//! The inventory is *derived* (`cargo metadata`), never hand-maintained — a hand-written list of
//! 200+ packages is stale the day it is written. Only the human judgements that cannot be
//! derived live in `docs/crate-status.toml`. This module joins the two and enforces the
//! integrity rules below.
//!
//! ## Deliberately honest defaults
//!
//! A crate absent from the registry is **unclassified**, and that is reported as a number rather
//! than hidden. Seeding 200 entries with invented evidence levels would be precisely the
//! decorative machinery this registry exists to expose — an unclassified crate is an honest
//! statement that nobody has assessed it, which is different from and better than a fabricated
//! `E1`.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

/// How mature a crate is *as a thing to depend on*. Independent of evidence: a `foundation`
/// crate with weak evidence is a problem worth seeing, not a contradiction to hide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Lifecycle {
    /// Stable substrate other crates may build on.
    Foundation,
    /// Under active investigation; API and conclusions may move.
    ActiveResearch,
    /// Works, but not held to foundation standards.
    Prototype,
    /// Structure exists, behaviour largely absent.
    Skeleton,
    /// Retained for history; not a live dependency.
    Archived,
}

/// What has actually been *demonstrated* about a crate — deliberately separate from lifecycle,
/// because "we built it carefully" and "we checked it against something outside itself" are
/// different claims and the project has historically conflated them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub enum Evidence {
    /// Compiles / structurally exists.
    E0,
    /// Internal unit invariants.
    E1,
    /// Independent known-answer or reference comparison.
    E2,
    /// Controlled ablation with positive and negative controls.
    E3,
    /// External dataset or simulator validation.
    E4,
    /// Prospective real-world validation.
    E5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ApiStability {
    Stable,
    Unstable,
    Experimental,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SafetyCriticality {
    /// Cannot affect a physical actuator, a safety gate, or a security decision.
    None,
    /// Feeds something that can, indirectly.
    Low,
    /// Directly gates motor output, safety tiers, or authorization.
    High,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrateStatus {
    pub lifecycle: Lifecycle,
    pub evidence: Evidence,
    /// What the evidence level is actually based on. Required — an evidence level with no
    /// stated basis is an assertion, not a record.
    pub evidence_note: String,
    /// Commit at which the evidence was last confirmed. Empty means "never confirmed at a
    /// specific revision", which `--check` treats as disqualifying for production use.
    #[serde(default)]
    pub last_verified_commit: String,
    pub api_stability: ApiStability,
    pub safety_criticality: SafetyCriticality,
    /// Whether this crate may appear on a production path. Gated by the rules in
    /// [`check`] — it cannot simply be asserted.
    pub production_allowed: bool,
    #[serde(default)]
    pub ci_lane: String,
    #[serde(default)]
    pub external_validation: String,
    /// Crate this one should eventually fold into, if any.
    #[serde(default)]
    pub consolidation_target: String,
}

#[derive(Debug, Deserialize)]
struct Meta {
    schema_version: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Registry {
    meta: Meta,
    #[serde(default)]
    crates: BTreeMap<String, CrateStatus>,
    /// Owned by `duplicate_scan`, declared here only because this struct is
    /// `deny_unknown_fields` and both commands read the same registry file. Adding the section
    /// without this field made `crate-status` fail to parse — caught by running it after the
    /// change rather than assuming the two were independent.
    #[serde(default)]
    #[allow(dead_code)]
    known_name_collisions: toml::Table,
}

const SUPPORTED_SCHEMA: u32 = 1;

/// Workspace members, from `cargo metadata` — derived, never hand-listed.
fn workspace_members(workspace_root: &Path) -> Result<BTreeSet<String>> {
    let out = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(workspace_root)
        .output()
        .context("failed to run `cargo metadata`")?;
    if !out.status.success() {
        bail!(
            "`cargo metadata` failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let json: serde_json::Value = serde_json::from_slice(&out.stdout)?;
    Ok(json["packages"]
        .as_array()
        .context("cargo metadata had no `packages` array")?
        .iter()
        .filter_map(|p| p["name"].as_str().map(str::to_string))
        .collect())
}

fn load_registry(path: &Path) -> Result<Registry> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("could not read registry at {}", path.display()))?;
    let reg: Registry = toml::from_str(&text)
        .with_context(|| format!("could not parse registry at {}", path.display()))?;
    if reg.meta.schema_version != SUPPORTED_SCHEMA {
        bail!(
            "registry schema_version {} is not supported (this tool speaks {})",
            reg.meta.schema_version,
            SUPPORTED_SCHEMA
        );
    }
    Ok(reg)
}

/// Integrity rules. These are the teeth: without them the registry is just a file of opinions
/// that drifts away from the workspace unnoticed, which is the failure mode it was built to
/// prevent.
///
/// **Hard errors** — things a human can fix by editing this file, so failing on them is fair:
/// * A registry entry naming a crate that no longer exists (stale entry).
/// * An evidence level with an empty `evidence_note`.
/// * `production_allowed` at less than `E2`, or with no recorded verification commit —
///   internal unit tests alone (`E1`) do not license production use.
///
/// **Findings** — real gaps between what a crate *does* and what has been *demonstrated* about
/// it, which cannot be closed by editing this file (only by doing the work):
/// * `safety_criticality = high` below `E3`.
/// * `lifecycle = foundation` below `E2`.
///
/// Findings are printed always and fail only under `strict`. The distinction matters: the very
/// first honest entry for `symthaea-core` trips both findings, and a gate that fails on day one
/// gets disabled rather than satisfied — taking the enforceable rules down with it.
///
/// Unclassified crates are counted and reported, never treated as an error by default. An
/// honest, visible, countable gap is the intended intermediate state.
pub fn check(
    workspace_root: &Path,
    registry_path: &Path,
    require_classified: bool,
    strict: bool,
) -> Result<()> {
    let members = workspace_members(workspace_root)?;
    let reg = load_registry(registry_path)?;
    let mut errors: Vec<String> = Vec::new();
    let mut findings: Vec<String> = Vec::new();

    for (name, st) in &reg.crates {
        if !members.contains(name) {
            errors.push(format!(
                "`{name}` has a registry entry but is not a workspace member (stale entry — \
                 delete it or restore the crate)"
            ));
        }
        if st.evidence_note.trim().is_empty() {
            errors.push(format!(
                "`{name}` states evidence {:?} with an empty `evidence_note` — an evidence level \
                 with no stated basis is an assertion, not a record",
                st.evidence
            ));
        }
        if st.production_allowed {
            if st.evidence < Evidence::E2 {
                errors.push(format!(
                    "`{name}` sets production_allowed = true at evidence {:?}; production use \
                     requires at least E2 (independent known-answer or reference comparison)",
                    st.evidence
                ));
            }
            if st.last_verified_commit.trim().is_empty() {
                errors.push(format!(
                    "`{name}` sets production_allowed = true with no last_verified_commit"
                ));
            }
        }
        // Below: *findings*, not errors. These record real gaps between what a crate does and
        // what has been demonstrated about it — gaps that exist today and cannot be closed by
        // editing this file. Making them hard failures would mean the gate fails on its first
        // honest entry, and a gate that fails on day one gets disabled rather than satisfied,
        // taking the enforceable rules down with it. They fail only under `--strict`.
        if st.safety_criticality == SafetyCriticality::High && st.evidence < Evidence::E3 {
            findings.push(format!(
                "`{name}` is safety_criticality = high at evidence {:?}; a component that gates \
                 motor output or authorization warrants at least E3 (controlled ablation)",
                st.evidence
            ));
        }
        if st.lifecycle == Lifecycle::Foundation && st.evidence < Evidence::E2 {
            findings.push(format!(
                "`{name}` is lifecycle = foundation at evidence {:?}; foundation crates warrant \
                 at least E2",
                st.evidence
            ));
        }
    }

    let unclassified: Vec<&String> = members
        .iter()
        .filter(|m| !reg.crates.contains_key(*m))
        .collect();

    if require_classified && !unclassified.is_empty() {
        errors.push(format!(
            "{} crate(s) are unclassified and --require-classified was set",
            unclassified.len()
        ));
    }

    println!(
        "crate-status: {} workspace members, {} classified, {} unclassified",
        members.len(),
        reg.crates.len(),
        unclassified.len()
    );

    for f in &findings {
        println!("  finding: {f}");
    }
    if !errors.is_empty() {
        for e in &errors {
            eprintln!("  error: {e}");
        }
        bail!("{} registry integrity error(s)", errors.len());
    }
    if strict && !findings.is_empty() {
        bail!(
            "{} evidence-gap finding(s) and --strict was set",
            findings.len()
        );
    }
    println!(
        "crate-status: registry integrity OK ({} finding(s) reported)",
        findings.len()
    );
    Ok(())
}

/// Emits the derived inventory as markdown. Generated on demand and never checked in as a
/// hand-maintained list.
pub fn report(workspace_root: &Path, registry_path: &Path) -> Result<()> {
    let members = workspace_members(workspace_root)?;
    let reg = load_registry(registry_path)?;

    println!("# Crate status inventory\n");
    println!("Generated from `cargo metadata` joined with `docs/crate-status.toml`.");
    println!("**A crate's existence does not imply endorsement.**\n");
    println!(
        "{} workspace members; {} classified; {} unclassified.\n",
        members.len(),
        reg.crates.len(),
        members.len() - reg.crates.len().min(members.len())
    );
    println!(
        "| Crate | Lifecycle | Evidence | API | Safety | Prod | Verified at | CI lane | External validation | Consolidate into |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for name in &members {
        match reg.crates.get(name) {
            Some(s) => {
                let dash = |v: &str| {
                    if v.trim().is_empty() {
                        "—".to_string()
                    } else {
                        v.to_string()
                    }
                };
                println!(
                    "| `{}` | {:?} | {:?} | {:?} | {:?} | {} | {} | {} | {} | {} |",
                    name,
                    s.lifecycle,
                    s.evidence,
                    s.api_stability,
                    s.safety_criticality,
                    if s.production_allowed { "yes" } else { "no" },
                    dash(&s.last_verified_commit),
                    dash(&s.ci_lane),
                    dash(&s.external_validation),
                    dash(&s.consolidation_target),
                )
            }
            None => println!("| `{name}` | *unclassified* | — | — | — | no | — | — | — | — |"),
        }
    }
    Ok(())
}
