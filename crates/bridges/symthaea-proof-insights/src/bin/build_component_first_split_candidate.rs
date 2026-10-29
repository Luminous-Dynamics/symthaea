// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Candidate-only component-first split generator for a future PITB protocol.
//!
//! This deliberately does not modify v0's frozen split or qualify a benchmark.
//! It asks what a successor split would look like if audited hard-connected
//! families were kept in one split from the beginning. The current v0 inventory
//! and receipt are bound to the old partition; a future confirmatory contract
//! must re-bind the complete relationship graph to the family universe itself,
//! verify that receipt independently, and reconstruct the graph before scoring.

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::Path,
};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const UPSTREAM: &str = "1581d4cbdd69a0faf34b2089614fe2a814316a6a";
const V0_PARTITION: &str = "e7ac417556cbd47cee1ddd51b98f9684fadfc8cd097587ae3beacc4dfb59f970";
const V0_SEED: &str = "pitb-v0-openai-math-2026-10-19";
const V0_ALGORITHM: &str = "sha256(seed || NUL || family_id), sort by digest then family ID, floor 70% train and floor 85% train+validation";
const CANDIDATE_SEED: &str = "pitb-component-first-candidate-2026-10-29";
const TARGET_SHARES: [f64; 3] = [0.70, 0.15, 0.15];
const SPLITS: [&str; 3] = ["train", "validation", "test"];

#[derive(Clone, Debug, PartialEq, Eq)]
struct ComponentUnit {
    component_id: String,
    family_ids: Vec<String>,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn required_str<'a>(value: &'a Value, path: &str) -> Result<&'a str> {
    path.split('.').try_fold(value, |current, key| {
        current.get(key).context(format!("missing JSON field {path}"))
    })?
    .as_str()
    .with_context(|| format!("field {path} must be a string"))
}

fn partition_digest(assignments: &BTreeMap<String, String>) -> String {
    let mut canonical = String::new();
    for (family_id, split) in assignments {
        canonical.push_str(family_id);
        canonical.push('\t');
        canonical.push_str(split);
        canonical.push('\n');
    }
    sha256(canonical.as_bytes())
}

fn parse_v0_splits(doc: &Value) -> Result<BTreeMap<String, String>> {
    if required_str(doc, "schema_version")? != "symthaea.proof-insight-transfer.splits.v0"
        || required_str(doc, "upstream.commit")? != UPSTREAM
        || required_str(doc, "split_plan.seed")? != V0_SEED
        || required_str(doc, "split_plan.algorithm")? != V0_ALGORITHM
        || required_str(doc, "split_plan.partition_digest_sha256")? != V0_PARTITION
    {
        bail!("input split manifest is not the pinned PITB v0 partition");
    }
    let rows = doc["families"]
        .as_array()
        .context("split manifest families must be an array")?;
    let mut assignments = BTreeMap::new();
    for row in rows {
        let id = row["family_id"]
            .as_str()
            .context("split family is missing family_id")?;
        let split = row["split"]
            .as_str()
            .context("split family is missing split")?;
        if id.trim().is_empty()
            || !SPLITS.contains(&split)
            || assignments.insert(id.to_owned(), split.to_owned()).is_some()
        {
            bail!("split manifest has blank/duplicate family or invalid split");
        }
    }
    if assignments.len() != 372 || partition_digest(&assignments) != V0_PARTITION {
        bail!("pinned v0 family count or canonical partition digest mismatch");
    }
    let counts = [
        assignments.values().filter(|split| *split == "train").count(),
        assignments
            .values()
            .filter(|split| *split == "validation")
            .count(),
        assignments.values().filter(|split| *split == "test").count(),
    ];
    if counts != [260, 56, 56] {
        bail!("pinned v0 split counts do not reconcile");
    }
    Ok(assignments)
}

fn parse_component_units(doc: &Value, frozen: &BTreeMap<String, String>) -> Result<Vec<ComponentUnit>> {
    if required_str(doc, "schema_version")? != "pitb.independent-component-manifest.v0"
        || required_str(doc, "upstream_commit")? != UPSTREAM
        || required_str(doc, "source_partition_digest_sha256")? != V0_PARTITION
    {
        bail!("component manifest is not bound to the pinned v0 source partition");
    }
    if doc["total_family_count"].as_u64() != Some(frozen.len() as u64) {
        bail!("component manifest family count does not match the frozen universe");
    }
    let inventory_digest = required_str(doc, "relationship_inventory_sha256")?;
    if !valid_sha256(inventory_digest) {
        bail!("component manifest relationship inventory digest is malformed");
    }
    let relationship_status = required_str(doc, "relationship_inventory_status")?;
    if !matches!(relationship_status, "incomplete" | "independently_reviewed_complete") {
        bail!("unsupported relationship inventory status");
    }

    let family_rows = doc["families"]
        .as_array()
        .context("component manifest families must be an array")?;
    let mut declared_family_map = BTreeMap::new();
    for row in family_rows {
        let id = row["family_id"]
            .as_str()
            .context("component family row missing family_id")?;
        let split = row["split"]
            .as_str()
            .context("component family row missing split")?;
        let component_id = row["component_id"]
            .as_str()
            .context("component family row missing component_id")?;
        if frozen.get(id).map(String::as_str) != Some(split)
            || component_id.trim().is_empty()
            || declared_family_map
                .insert(id.to_owned(), component_id.to_owned())
                .is_some()
        {
            bail!("component family rows disagree with v0 membership or contain duplicates");
        }
    }
    if declared_family_map.len() != frozen.len()
        || declared_family_map.keys().ne(frozen.keys())
    {
        bail!("component family rows do not cover the exact frozen family universe");
    }

    let component_rows = doc["components"]
        .as_array()
        .context("component manifest components must be an array")?;
    let mut units = Vec::with_capacity(component_rows.len());
    let mut component_ids = BTreeSet::new();
    let mut family_seen = BTreeSet::new();
    let mut observed_cross_split = 0usize;
    for row in component_rows {
        let component_id = row["component_id"]
            .as_str()
            .context("component row missing component_id")?
            .to_owned();
        if component_id.trim().is_empty() || !component_ids.insert(component_id.clone()) {
            bail!("component IDs must be unique and nonblank");
        }
        let members = row["family_ids"]
            .as_array()
            .context("component family_ids must be an array")?;
        if members.is_empty() {
            bail!("component must contain at least one family");
        }
        let mut family_ids = Vec::with_capacity(members.len());
        let mut old_splits = BTreeSet::new();
        for member in members {
            let id = member.as_str().context("component family ID must be a string")?;
            if !frozen.contains_key(id) || !family_seen.insert(id.to_owned()) {
                bail!("component references an unknown or multiply assigned family");
            }
            if declared_family_map.get(id) != Some(&component_id) {
                bail!("component rows disagree with the family-to-component table");
            }
            old_splits.insert(frozen[id].clone());
            family_ids.push(id.to_owned());
        }
        family_ids.sort();
        let declared_splits = row["split_names"]
            .as_array()
            .context("component split_names must be an array")?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .context("component split name must be a string")
            })
            .collect::<Result<BTreeSet<_>>>()?;
        if declared_splits != old_splits {
            bail!("component split_names do not match the frozen family assignments");
        }
        if old_splits.len() > 1 {
            observed_cross_split += 1;
        }
        units.push(ComponentUnit {
            component_id,
            family_ids,
        });
    }
    if family_seen.len() != frozen.len() {
        bail!("component rows do not cover the exact frozen family universe");
    }
    if doc["total_component_count"].as_u64() != Some(units.len() as u64)
        || doc["cross_split_component_count"].as_u64() != Some(observed_cross_split as u64)
    {
        bail!("component count or cross-split component count does not reconcile");
    }
    units.sort_by(|left, right| left.component_id.cmp(&right.component_id));
    Ok(units)
}

fn rank_digest(component_id: &str) -> String {
    sha256(format!("{CANDIDATE_SEED}\0{component_id}").as_bytes())
}

/// Assign each complete component to one split. Components are visited in a
/// deterministic pseudo-random order; each goes to the split with the largest
/// remaining *fractional* family-count deficit. The allocation is approximate
/// when components are indivisible and is never allowed to change PITB v0.
fn assign_components(units: &[ComponentUnit]) -> Result<BTreeMap<String, String>> {
    if units.is_empty() {
        bail!("cannot allocate an empty component set");
    }
    let mut component_ids = BTreeSet::new();
    let mut family_ids = BTreeSet::new();
    for unit in units {
        if unit.component_id.trim().is_empty()
            || unit.family_ids.is_empty()
            || !component_ids.insert(unit.component_id.clone())
        {
            bail!("component IDs must be unique/nonblank and components nonempty");
        }
        for id in &unit.family_ids {
            if id.trim().is_empty() || !family_ids.insert(id.clone()) {
                bail!("family IDs must be unique and assigned to exactly one component");
            }
        }
    }

    let family_count = family_ids.len();
    if family_count == 0 {
        bail!("component set contains no families");
    }
    let targets = TARGET_SHARES.map(|share| share * family_count as f64);
    let mut ranked = units.to_vec();
    ranked.sort_by(|left, right| {
        rank_digest(&left.component_id)
            .cmp(&rank_digest(&right.component_id))
            .then_with(|| left.component_id.cmp(&right.component_id))
    });

    let mut counts = [0usize; 3];
    let mut output = BTreeMap::new();
    for unit in ranked {
        let size = unit.family_ids.len();
        let best_split = (0..3)
            .max_by(|left, right| {
                let left_deficit = (targets[*left] - counts[*left] as f64) / targets[*left].max(1.0);
                let right_deficit =
                    (targets[*right] - counts[*right] as f64) / targets[*right].max(1.0);
                left_deficit
                    .total_cmp(&right_deficit)
                    // Stable tie-breaking: train, validation, then test.
                    .then_with(|| right.cmp(left))
            })
            .context("three split destinations must exist")?;
        let split = SPLITS[best_split].to_owned();
        counts[best_split] += size;
        for family_id in unit.family_ids {
            output.insert(family_id, split.clone());
        }
    }
    if output.len() != family_count {
        bail!("internal allocation did not preserve all families");
    }
    Ok(output)
}

fn execute() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 || args[1] == "--help" || args[1] == "-h" {
        println!(
            "usage: {} <pinned-v0-splits.json> <component-manifest.json> <candidate-output.json>",
            args.first().map(String::as_str).unwrap_or("build_component_first_split_candidate")
        );
        println!("Output is exploratory only and is never confirmatory qualification.");
        if args.len() == 4 {
            return Ok(());
        }
        if args.len() == 2 && (args[1] == "--help" || args[1] == "-h") {
            return Ok(());
        }
        bail!("expected three file arguments");
    }

    let split_path = Path::new(&args[1]);
    let component_path = Path::new(&args[2]);
    let output_path = Path::new(&args[3]);
    let split_bytes = fs::read(split_path)
        .with_context(|| format!("read split manifest {}", split_path.display()))?;
    let component_bytes = fs::read(component_path)
        .with_context(|| format!("read component manifest {}", component_path.display()))?;
    let split_doc: Value = serde_json::from_slice(&split_bytes).context("parse split manifest")?;
    let component_doc: Value =
        serde_json::from_slice(&component_bytes).context("parse component manifest")?;
    let old_assignments = parse_v0_splits(&split_doc)?;
    let units = parse_component_units(&component_doc, &old_assignments)?;
    let new_assignments = assign_components(&units)?;

    let mut component_split = BTreeMap::new();
    let mut component_family_rows = Vec::with_capacity(units.len());
    let mut family_component = BTreeMap::new();
    for unit in &units {
        let splits: BTreeSet<&str> = unit
            .family_ids
            .iter()
            .map(|id| new_assignments[id].as_str())
            .collect();
        if splits.len() != 1 {
            bail!("internal error: a hard-connected component was split across outputs");
        }
        let split = *splits.first().context("component has no assignment")?;
        component_split.insert(unit.component_id.clone(), split.to_owned());
        component_family_rows.push(json!({
            "component_id": unit.component_id,
            "split": split,
            "family_count": unit.family_ids.len(),
            "family_ids": unit.family_ids,
        }));
        for id in &unit.family_ids {
            family_component.insert(id.clone(), unit.component_id.clone());
        }
    }

    let family_rows: Vec<Value> = new_assignments
        .iter()
        .map(|(family_id, split)| {
            json!({
                "family_id": family_id,
                "split": split,
                "component_id": family_component[family_id],
            })
        })
        .collect();
    let counts = |split: &str| new_assignments.values().filter(|value| value.as_str() == split).count();
    let component_counts = |split: &str| component_split.values().filter(|value| value.as_str() == split).count();
    let result = json!({
        "schema_version": "pitb.component-first-split-candidate.v0",
        "status": "candidate_only",
        "upstream_commit": UPSTREAM,
        "candidate_seed": CANDIDATE_SEED,
        "allocation_algorithm": "order components by SHA-256(seed || NUL || component_id); assign each whole component to the split with the largest remaining fraction of target family capacity; tie order train, validation, test",
        "target_family_shares": {
            "train": TARGET_SHARES[0],
            "validation": TARGET_SHARES[1],
            "test": TARGET_SHARES[2]
        },
        "family_count": new_assignments.len(),
        "component_count": units.len(),
        "family_count_by_split": {
            "train": counts("train"),
            "validation": counts("validation"),
            "test": counts("test")
        },
        "component_count_by_split": {
            "train": component_counts("train"),
            "validation": component_counts("validation"),
            "test": component_counts("test")
        },
        "source_v0_partition_digest_sha256": V0_PARTITION,
        "source_split_manifest_sha256": sha256(&split_bytes),
        "source_component_manifest_sha256": sha256(&component_bytes),
        "relationship_inventory_sha256": required_str(&component_doc, "relationship_inventory_sha256")?,
        "candidate_partition_digest_sha256": partition_digest(&new_assignments),
        "source_inventory_status": required_str(&component_doc, "relationship_inventory_status")?,
        "source_receipt_verification_status": component_doc["coverage_receipt_verification_status"],
        "source_manifest_was_qualified": component_doc["qualified_for_confirmatory_resampling"],
        "qualified_for_confirmatory_use": false,
        "qualification_blockers": [
            "This is a candidate generator, not a source of qualification.",
            "The v0 relationship inventory and receipt are bound to the old split digest.",
            "A successor receipt contract must authenticate the complete sorted family universe and relationship inventory independently of any train/validation/test assignment.",
            "A confirmatory verifier must independently re-verify the new receipt and reconstruct the full hard-component graph from the signed inventory.",
            "Freeze the successor protocol, allocation, target eligibility audit, and power plan before any model outcomes are observed."
        ],
        "components": component_family_rows,
        "families": family_rows
    });
    fs::write(output_path, serde_json::to_vec_pretty(&result)?)
        .with_context(|| format!("write candidate output {}", output_path.display()))?;
    println!(
        "PITB_COMPONENT_FIRST_CANDIDATE_ONLY: families={} components={} counts=train:{},validation:{},test:{}",
        new_assignments.len(),
        units.len(),
        counts("train"),
        counts("validation"),
        counts("test")
    );
    Ok(())
}

fn main() -> std::process::ExitCode {
    match execute() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("component-first PITB split candidate rejected: {error:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn units(sizes: &[usize]) -> Vec<ComponentUnit> {
        let mut next = 0usize;
        sizes
            .iter()
            .enumerate()
            .map(|(component_index, size)| {
                let family_ids = (0..*size)
                    .map(|_| {
                        next += 1;
                        format!("family-{next:04}")
                    })
                    .collect();
                ComponentUnit {
                    component_id: format!("component-{component_index:04}"),
                    family_ids,
                }
            })
            .collect()
    }

    #[test]
    fn component_first_allocation_never_splits_a_component() {
        let source = units(&[4, 3, 2, 2, 1, 1, 1, 1, 1, 1]);
        let assignments = assign_components(&source).expect("valid components allocate");
        for unit in &source {
            let assigned: BTreeSet<&str> = unit
                .family_ids
                .iter()
                .map(|family| assignments[family].as_str())
                .collect();
            assert_eq!(assigned.len(), 1, "component {}", unit.component_id);
        }
        assert_eq!(assignments.len(), 17);
    }

    #[test]
    fn allocation_is_order_independent_and_deterministic() {
        let source = units(&[7, 4, 3, 2, 2, 1, 1, 1, 1, 1, 1]);
        let first = assign_components(&source).expect("valid components allocate");
        let mut reversed = source.clone();
        reversed.reverse();
        for unit in &mut reversed {
            unit.family_ids.reverse();
        }
        let second = assign_components(&reversed).expect("reordered input allocates");
        assert_eq!(first, second);
        assert_eq!(
            partition_digest(&first),
            partition_digest(&second),
            "canonical partition digest must be input-order invariant"
        );
    }

    #[test]
    fn singleton_components_approximate_the_registered_split_shares() {
        let source = units(&vec![1; 100]);
        let assignments = assign_components(&source).expect("singleton components allocate");
        let counts = [
            assignments.values().filter(|split| *split == "train").count(),
            assignments.values().filter(|split| *split == "validation").count(),
            assignments.values().filter(|split| *split == "test").count(),
        ];
        assert_eq!(counts, [70, 15, 15]);
    }

    #[test]
    fn duplicate_family_membership_is_rejected() {
        let source = vec![
            ComponentUnit {
                component_id: "component-a".to_owned(),
                family_ids: vec!["family-1".to_owned(), "family-2".to_owned()],
            },
            ComponentUnit {
                component_id: "component-b".to_owned(),
                family_ids: vec!["family-2".to_owned()],
            },
        ];
        assert!(assign_components(&source).is_err());
    }

    #[test]
    fn empty_components_are_rejected() {
        assert!(assign_components(&[]).is_err());
        assert!(assign_components(&[ComponentUnit {
            component_id: "component-empty".to_owned(),
            family_ids: vec![],
        }])
        .is_err());
    }
}
