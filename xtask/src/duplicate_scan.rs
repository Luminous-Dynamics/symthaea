//! Discovery check: does a new standalone crate re-implement something `symthaea-core` already has?
//!
//! ## The failure this exists to stop
//!
//! Five crates were created inside a 48-hour window on 2026-07-09/10 — `statistics`,
//! `graph_theory`, `combinatorics`, `complex`, `game_theory` — each asserting in its commit
//! message or `lib.rs` docs that the workspace lacked the capability. **Every one of those claims
//! was false** against a `symthaea-core/src/hdc/` module predating it by 2–5 months.
//!
//! That is a *discovery* failure, not a migration failure, and in the 2026-07-29 consolidation
//! audit it produced more duplicate pairs (5) than migration failure did (2).
//!
//! `DOMAIN_CRATES_INDEX.md` exists to prevent exactly this and warns about it in its first three
//! lines — but it indexes only `crates/domains/`, so in-core modules are **invisible** to the very
//! check built to catch this. This closes that blind spot.
//!
//! ## Why an allowlist rather than a hard ban
//!
//! The same audit found the name-matching heuristic runs at roughly **22% precision**. Of nine
//! suspected pairs, seven were never migrations: `projection` is a pure homonym (trainable HDC
//! maps vs. a telemetry DTO, zero shared symbols), and `consciousness_topology` is a false pair —
//! the crate is a successful extraction of a *different* file.
//!
//! So a collision is a **question**, not a verdict. Known collisions are adjudicated once and
//! recorded; the check fails only on a **new, unadjudicated** one. That keeps the signal
//! actionable instead of training everyone to ignore a permanently-red check.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// One adjudicated collision, recorded so it is decided once rather than rediscovered.
#[derive(Debug, Deserialize)]
pub struct KnownCollision {
    /// What was determined: `duplicate`, `homonym`, `false-pair`, `independent-same-domain`,
    /// `retired`, or `undecided`.
    pub disposition: String,
    /// Evidence for that disposition. Required — an adjudication with no stated basis is an
    /// assertion, and this file exists precisely to stop those.
    pub note: String,
}

#[derive(Debug, Deserialize)]
struct Registry {
    #[serde(default)]
    known_name_collisions: BTreeMap<String, KnownCollision>,
}

/// Module names that are structural rather than capability-bearing, so a same-named crate is not
/// evidence of anything.
const IGNORED_MODULES: &[&str] = &["mod", "lib", "main", "types", "config", "error", "utils"];

fn core_module_names(workspace_root: &Path) -> Result<BTreeSet<String>> {
    let mut out = BTreeSet::new();
    let core = workspace_root.join("crates/core/symthaea-core/src");
    for sub in ["", "hdc"] {
        let dir = if sub.is_empty() {
            core.clone()
        } else {
            core.join(sub)
        };
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            let name = if p.is_dir() {
                // A directory module counts only if it has a mod.rs.
                if !p.join("mod.rs").exists() {
                    continue;
                }
                p.file_name().and_then(|s| s.to_str()).map(str::to_string)
            } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                p.file_stem().and_then(|s| s.to_str()).map(str::to_string)
            } else {
                None
            };
            if let Some(n) = name {
                if !IGNORED_MODULES.contains(&n.as_str()) {
                    out.insert(n);
                }
            }
        }
    }
    Ok(out)
}

fn crate_dir_names(workspace_root: &Path) -> Result<BTreeSet<String>> {
    let mut out = BTreeSet::new();
    for tier in ["crates/core", "crates/domains", "crates/bridges"] {
        let Ok(entries) = std::fs::read_dir(workspace_root.join(tier)) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() && p.join("Cargo.toml").exists() {
                if let Some(n) = p.file_name().and_then(|s| s.to_str()) {
                    out.insert(n.to_string());
                }
            }
        }
    }
    Ok(out)
}

/// `foo_bar` (module) -> `symthaea-foo-bar` (crate directory).
fn module_to_crate_name(module: &str) -> String {
    format!("symthaea-{}", module.replace('_', "-"))
}

pub fn scan(workspace_root: &Path, registry_path: &Path) -> Result<()> {
    let text = std::fs::read_to_string(registry_path)
        .with_context(|| format!("could not read {}", registry_path.display()))?;
    let reg: Registry = toml::from_str(&text)
        .with_context(|| format!("could not parse {}", registry_path.display()))?;

    let modules = core_module_names(workspace_root)?;
    let crates = crate_dir_names(workspace_root)?;

    let mut collisions: Vec<(String, String)> = Vec::new();
    for m in &modules {
        let c = module_to_crate_name(m);
        // Skip the self-collision: symthaea-core's own `core` module maps to the name of the
        // crate that contains it. That is an artifact of the naming scheme, not a duplicate.
        if c == "symthaea-core" {
            continue;
        }
        if crates.contains(&c) {
            collisions.push((m.clone(), c));
        }
    }

    let mut unadjudicated = Vec::new();
    println!(
        "duplicate-scan: {} symthaea-core modules, {} member crates, {} name collision(s)",
        modules.len(),
        crates.len(),
        collisions.len()
    );
    for (m, c) in &collisions {
        match reg.known_name_collisions.get(m) {
            Some(k) => println!("  known    {m:<28} <-> {c}   [{}]", k.disposition),
            None => {
                println!("  NEW      {m:<28} <-> {c}");
                unadjudicated.push((m.clone(), c.clone()));
            }
        }
    }

    // A recorded collision whose crate has since gone is stale bookkeeping, not a failure.
    for m in reg.known_name_collisions.keys() {
        if !collisions.iter().any(|(cm, _)| cm == m) {
            println!(
                "  stale    {m:<28} (recorded, but no collision found — entry can be removed)"
            );
        }
    }

    if !unadjudicated.is_empty() {
        eprintln!();
        for (m, c) in &unadjudicated {
            eprintln!("error: unadjudicated name collision: `{c}` vs `symthaea-core`'s `{m}`");
        }
        eprintln!();
        eprintln!(
            "A collision is a QUESTION, not a verdict — this heuristic ran at ~22% precision in\n\
             the 2026-07-29 audit, and most collisions turned out NOT to be duplicates. But five\n\
             crates were once created in 48 hours, each falsely claiming the workspace lacked a\n\
             capability it had held for months, because in-core modules were invisible to the\n\
             index meant to prevent it.\n\n\
             Read BOTH implementations, decide, and record it in `known_name_collisions` in\n\
             docs/crate-status.toml with a disposition (duplicate / homonym / false-pair /\n\
             independent-same-domain / retired / undecided) and the evidence."
        );
        bail!("{} unadjudicated name collision(s)", unadjudicated.len());
    }

    println!("duplicate-scan: OK — every name collision is adjudicated.");
    Ok(())
}
