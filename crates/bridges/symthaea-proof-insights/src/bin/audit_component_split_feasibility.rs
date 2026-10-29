// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Outcome-blind census/split reconciliation for PITB component-first candidates.
//!
//! This reports structural feasibility only. It does not authenticate external
//! reviewer receipts, verify statement semantics, or authorize confirmatory use.

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Component, Path},
    process::ExitCode,
};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const UPSTREAM: &str = "1581d4cbdd69a0faf34b2089614fe2a814316a6a";
const V0_PARTITION: &str = "e7ac417556cbd47cee1ddd51b98f9684fadfc8cd097587ae3beacc4dfb59f970";
const EXPECTED_FAMILY_UNIVERSE_SHA256: &str =
    "c8ee398840443fb0693e50baa1dfda9982cd8bfebe1f81d9de3050e3b15cb048";
const CENSUS_SCHEMA: &str = "pitb.pre-split-target-eligibility-census.v0";
const MIN_HELDOUT_COMPONENTS: usize = 20;
const SPLITS: [&str; 3] = ["train", "validation", "test"];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EligibilityCensus {
    schema_version: String,
    dataset_kind: String,
    upstream_commit: String,
    family_universe_sha256: String,
    eligibility_policy_sha256: String,
    census_receipt_sha256: String,
    sealed_before_evaluation: bool,
    outcomes_used: bool,
    families: Vec<CensusFamily>,
    targets: Vec<CensusTarget>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CensusFamily {
    family_id: String,
    eligibility_status: String,
    candidate_target_ids: Vec<String>,
    eligibility_receipt_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CensusTarget {
    target_id: String,
    family_id: String,
    status: String,
    artifact_path: String,
    artifact_sha256: String,
    statement_sha256: Value,
    eligibility_receipt_sha256: String,
}

#[derive(Debug)]
struct CandidateSplit {
    dataset_kind: String,
    bytes_sha256: String,
    partition_digest_sha256: String,
    family_to_split: BTreeMap<String, String>,
    family_to_component: BTreeMap<String, String>,
    components: BTreeMap<String, ComponentRow>,
}

#[derive(Debug)]
struct ComponentRow {
    split: String,
    family_ids: BTreeSet<String>,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_identifier(value: &str) -> bool {
    !value.trim().is_empty() && !value.chars().any(char::is_control)
}

fn validate_sha(value: &str, label: &str) -> Result<()> {
    if !valid_sha256(value) {
        bail!("{label} is not a 64-character SHA-256 digest");
    }
    Ok(())
}

fn family_universe_digest(families: &BTreeSet<String>) -> String {
    let mut material = String::new();
    for family_id in families {
        material.push_str(family_id);
        material.push('\n');
    }
    sha256(material.as_bytes())
}

fn partition_digest(assignments: &BTreeMap<String, String>) -> String {
    let mut material = String::new();
    for (family_id, split) in assignments {
        material.push_str(family_id);
        material.push('\t');
        material.push_str(split);
        material.push('\n');
    }
    sha256(material.as_bytes())
}

fn family_universe_digest(families: &BTreeSet<String>) -> String {
    let mut material = String::new();
    for family_id in families {
        material.push_str(family_id);
        material.push('\n');
    }
    sha256(material.as_bytes())
}

fn component_quotas(component_count: usize) -> [usize; 3] {
    let train_end = component_count * 70 / 100;
    let validation_end = component_count * 85 / 100;
    [
        train_end,
        validation_end - train_end,
        component_count - validation_end,
    ]
}

fn read_split_count(doc: &Value, field: &str, split: &str) -> Result<usize> {
    doc[field][split]
        .as_u64()
        .map(|count| count as usize)
        .with_context(|| format!("candidate {field}.{split} must be an integer"))
}

fn safe_repo_path(value: &str) -> bool {
    if value.trim().is_empty() {
        return false;
    }
    let path = Path::new(value);
    !path.is_absolute()
        && path.components().all(|component| {
            matches!(component, Component::Normal(_))
        })
}

fn parse_candidate_split(bytes: &[u8]) -> Result<CandidateSplit> {
    let doc: Value = serde_json::from_slice(bytes)
        .context("parse component-first candidate split")?;
    let dataset_kind = doc["dataset_kind"]
        .as_str()
        .context("candidate dataset_kind is missing")?;
    if !matches!(dataset_kind, "candidate_only" | "synthetic_fixture")
        || doc["schema_version"].as_str() != Some("pitb.component-first-split-candidate.v0")
        || doc["status"].as_str() != Some("candidate_only")
        || doc["upstream_commit"].as_str() != Some(UPSTREAM)
        || doc["source_v0_partition_digest_sha256"].as_str() != Some(V0_PARTITION)
        || doc["qualified_for_confirmatory_use"].as_bool() != Some(false)
    {
        bail!("split input must be the pinned, explicitly unqualified component-first candidate format");
    }
    for field in [
        "source_split_manifest_sha256",
        "source_component_manifest_sha256",
        "relationship_inventory_sha256",
    ] {
        let digest = doc[field]
            .as_str()
            .with_context(|| format!("candidate {field} is missing"))?;
        validate_sha(digest, field)?;
    }
    let expected_family_count = doc["family_count"]
        .as_u64()
        .context("candidate family_count must be an integer")? as usize;
    let expected_component_count = doc["component_count"]
        .as_u64()
        .context("candidate component_count must be an integer")? as usize;
    let expected_partition_digest = doc["candidate_partition_digest_sha256"]
        .as_str()
        .context("candidate partition digest is missing")?;
    validate_sha(expected_partition_digest, "candidate partition digest")?;
    let declared_universe_digest = doc["family_universe_sha256"]
        .as_str()
        .context("candidate family-universe digest is missing")?;
    validate_sha(declared_universe_digest, "candidate family-universe digest")?;

    let family_rows = doc["families"]
        .as_array()
        .context("candidate families must be an array")?;
    let component_rows = doc["components"]
        .as_array()
        .context("candidate components must be an array")?;
    if family_rows.len() != expected_family_count
        || component_rows.len() != expected_component_count
    {
        bail!("candidate declared counts do not match family/component row counts");
    }

    let mut family_to_split = BTreeMap::new();
    let mut family_to_component = BTreeMap::new();
    for row in family_rows {
        let family_id = row["family_id"]
            .as_str()
            .context("candidate family row is missing family_id")?;
        let split = row["split"]
            .as_str()
            .context("candidate family row is missing split")?;
        let component_id = row["component_id"]
            .as_str()
            .context("candidate family row is missing component_id")?;
        if !valid_identifier(family_id)
            || !SPLITS.contains(&split)
            || !valid_identifier(component_id)
        {
            bail!("candidate contains a blank ID or invalid split");
        }
        if family_to_split.insert(family_id.to_owned(), split.to_owned()).is_some()
            || family_to_component.insert(family_id.to_owned(), component_id.to_owned()).is_some()
        {
            bail!("candidate contains duplicate family assignment");
        }
    }

    let mut components = BTreeMap::new();
    let mut seen_component_families = BTreeSet::new();
    for row in component_rows {
        let component_id = row["component_id"]
            .as_str()
            .context("candidate component row is missing component_id")?;
        let split = row["split"]
            .as_str()
            .context("candidate component row is missing split")?;
        let members = row["family_ids"]
            .as_array()
            .context("candidate component family_ids must be an array")?;
        if !valid_identifier(component_id) || !SPLITS.contains(&split) || members.is_empty() {
            bail!("candidate contains a blank component, invalid split, or empty component");
        }
        let mut member_set = BTreeSet::new();
        for member in members {
            let family_id = member.as_str().context("component family ID must be a string")?;
            if !family_to_split.contains_key(family_id)
                || !seen_component_families.insert(family_id.to_owned())
                || !member_set.insert(family_id.to_owned())
            {
                bail!("component references an unknown, duplicated, or repeated family");
            }
            if family_to_split[family_id] != split
                || family_to_component[family_id] != component_id
            {
                bail!("family/component assignment disagrees between manifest tables");
            }
        }
        if row["family_count"].as_u64() != Some(member_set.len() as u64) {
            bail!("component family_count does not reconcile with its family_ids");
        }
        if components.insert(
            component_id.to_owned(),
            ComponentRow { split: split.to_owned(), family_ids: member_set },
        ).is_some() {
            bail!("candidate component IDs must be unique");
        }
    }

    let all_families: BTreeSet<String> = family_to_split.keys().cloned().collect();
    if seen_component_families != all_families || components.len() != expected_component_count {
        bail!("component table does not cover each candidate family exactly once");
    }
    if family_universe_digest(&all_families) != declared_universe_digest {
        bail!("candidate family-universe digest does not match its exact sorted family IDs");
    }
    if dataset_kind == "candidate_only"
        && (all_families.len() != 372
            || declared_universe_digest != EXPECTED_FAMILY_UNIVERSE_SHA256)
    {
        bail!("real candidate must use the exact pinned 372-family PITB universe");
    }
    if dataset_kind == "synthetic_fixture" && all_families.len() >= 372 {
        bail!("synthetic fixture candidate must not masquerade as the reviewed real universe");
    }
    if partition_digest(&family_to_split) != expected_partition_digest {
        bail!("candidate partition digest does not match its family assignments");
    }

    let mut family_counts = [0usize; 3];
    for split_name in family_to_split.values() {
        let index = SPLITS
            .iter()
            .position(|candidate| candidate == split_name)
            .context("candidate contains an unregistered split")?;
        family_counts[index] += 1;
    }
    let mut component_counts = [0usize; 3];
    for component in components.values() {
        let index = SPLITS
            .iter()
            .position(|candidate| *candidate == component.split)
            .context("candidate component contains an unregistered split")?;
        component_counts[index] += 1;
    }
    let expected_component_counts = component_quotas(expected_component_count);
    for (index, split_name) in SPLITS.iter().enumerate() {
        if read_split_count(&doc, "family_count_by_split", split_name)? != family_counts[index] {
            bail!("candidate family_count_by_split does not reconcile");
        }
        if read_split_count(&doc, "component_count_by_split", split_name)?
            != component_counts[index]
        {
            bail!("candidate component_count_by_split does not reconcile");
        }
        if read_split_count(&doc, "target_component_count_by_split", split_name)?
            != expected_component_counts[index]
        {
            bail!("candidate target component quotas do not match the frozen floor-based policy");
        }
        if component_counts[index] != expected_component_counts[index] {
            bail!("candidate actual component counts violate the frozen component quotas");
        }
    }

    Ok(CandidateSplit {
        dataset_kind: dataset_kind.to_owned(),
        bytes_sha256: sha256(bytes),
        partition_digest_sha256: expected_partition_digest.to_owned(),
        family_to_split,
        family_to_component,
        components,
    })
}

fn validate_census(census: &EligibilityCensus, split: &CandidateSplit) -> Result<()> {
    let dataset_kinds_match = matches!(
        (split.dataset_kind.as_str(), census.dataset_kind.as_str()),
        ("candidate_only", "reviewed_census") | ("synthetic_fixture", "synthetic_fixture")
    );
    if census.schema_version != CENSUS_SCHEMA
        || census.upstream_commit != UPSTREAM
        || !matches!(census.dataset_kind.as_str(), "synthetic_fixture" | "reviewed_census")
        || !dataset_kinds_match
        || !census.sealed_before_evaluation
        || census.outcomes_used
    {
        bail!("census schema, source, dataset kind, or outcome-blind freeze conditions are invalid");
    }
    validate_sha(&census.family_universe_sha256, "census family universe digest")?;
    validate_sha(&census.eligibility_policy_sha256, "eligibility policy digest")?;
    validate_sha(&census.census_receipt_sha256, "census receipt digest")?;

    let candidate_families: BTreeSet<String> = split.family_to_split.keys().cloned().collect();
    let census_families: BTreeSet<String> = census
        .families
        .iter()
        .map(|family| family.family_id.clone())
        .collect();
    if census_families.len() != census.families.len() || census_families != candidate_families {
        bail!("census family inventory must exactly cover the candidate family universe, without duplicates");
    }
    if family_universe_digest(&census_families) != census.family_universe_sha256 {
        bail!("census family-universe digest does not match its exact sorted family IDs");
    }
    if census.dataset_kind == "reviewed_census"
        && (census_families.len() != 372
            || census.family_universe_sha256 != EXPECTED_FAMILY_UNIVERSE_SHA256)
    {
        bail!("reviewed census must use the exact pinned 372-family PITB universe");
    }

    let mut declared_targets_by_family: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut family_status_by_id = BTreeMap::new();
    for family in &census.families {
        validate_sha(&family.eligibility_receipt_sha256, "family eligibility receipt digest")?;
        if !matches!(family.eligibility_status.as_str(), "eligible" | "excluded" | "unknown") {
            bail!("family has an unregistered eligibility status");
        }
        let mut ids = BTreeSet::new();
        for id in &family.candidate_target_ids {
            if !valid_identifier(id) || !ids.insert(id.clone()) {
                bail!("family target IDs must be unique and nonblank");
            }
        }
        declared_targets_by_family.insert(family.family_id.clone(), ids);
        family_status_by_id.insert(family.family_id.clone(), family.eligibility_status.as_str());
    }

    let mut observed_targets_by_family: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut observed_eligibility_counts: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    let mut target_ids = BTreeSet::new();
    for target in &census.targets {
        if !valid_identifier(&target.target_id) || !target_ids.insert(target.target_id.clone()) {
            bail!("target IDs must be unique and nonblank");
        }
        if !valid_identifier(&target.family_id) || !census_families.contains(&target.family_id) {
            bail!("target references a family outside the frozen universe");
        }
        if !matches!(target.status.as_str(), "eligible" | "excluded" | "unknown") {
            bail!("target has an unregistered eligibility status");
        }
        if !safe_repo_path(&target.artifact_path) {
            bail!("target artifact path must be a safe repository-relative path");
        }
        validate_sha(&target.artifact_sha256, "target artifact digest")?;
        validate_sha(&target.eligibility_receipt_sha256, "target eligibility receipt digest")?;
        if target.statement_sha256.is_null() {
            if target.status == "eligible" {
                bail!("eligible target requires an exact statement digest");
            }
        } else {
            let digest = target.statement_sha256
                .as_str()
                .context("statement_sha256 must be a digest string or explicit null")?;
            validate_sha(digest, "target statement digest")?;
        }
        observed_targets_by_family
            .entry(target.family_id.clone())
            .or_default()
            .insert(target.target_id.clone());
        let counts = observed_eligibility_counts.entry(target.family_id.clone()).or_default();
        match target.status.as_str() {
            "eligible" => counts.0 += 1,
            "excluded" => counts.1 += 1,
            "unknown" => counts.2 += 1,
            _ => unreachable!("validated target status"),
        }
    }
    for family_id in &census_families {
        let declared = declared_targets_by_family
            .get(family_id)
            .context("missing family target inventory")?;
        let observed = observed_targets_by_family.get(family_id).cloned().unwrap_or_default();
        if declared != &observed {
            bail!("family {family_id} candidate_target_ids do not exactly match target rows");
        }
        let counts = observed_eligibility_counts.get(family_id).copied().unwrap_or_default();
        let declared_status = family_status_by_id[family_id];
        let consistent = match declared_status {
            "eligible" => counts.0 > 0,
            "excluded" => counts.0 == 0 && counts.2 == 0,
            "unknown" => counts.0 == 0 && (counts.2 > 0 || declared.is_empty()),
            _ => false,
        };
        if !consistent {
            bail!("family {family_id} eligibility_status disagrees with its target rows");
        }
    }
    Ok(())
}

fn execute() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        println!("usage: {} <component-first-candidate.json> <pre-split-target-census.json> <report.json>", args.first().map(String::as_str).unwrap_or("audit_component_split_feasibility"));
        bail!("expected three file arguments");
    }

    let split_bytes = fs::read(&args[1])
        .with_context(|| format!("read candidate split {}", args[1]))?;
    let census_bytes = fs::read(&args[2])
        .with_context(|| format!("read target census {}", args[2]))?;
    let output_path = &args[3];
    let split = parse_candidate_split(&split_bytes)?;
    let census: EligibilityCensus = serde_json::from_slice(&census_bytes)
        .context("parse outcome-blind eligibility census")?;
    validate_census(&census, &split)?;

    let mut eligible_target_count = BTreeMap::<String, usize>::new();
    let mut excluded_target_count = BTreeMap::<String, usize>::new();
    let mut unknown_target_count = BTreeMap::<String, usize>::new();
    let mut family_status_by_id = BTreeMap::<String, (usize, usize, usize)>::new();
    for target in &census.targets {
        let counts = family_status_by_id.entry(target.family_id.clone()).or_default();
        match target.status.as_str() {
            "eligible" => counts.0 += 1,
            "excluded" => counts.1 += 1,
            "unknown" => counts.2 += 1,
            _ => unreachable!("validated target status"),
        }
    }
    let mut eligible_components_by_split = BTreeMap::<String, usize>::new();
    let mut all_components_by_split = BTreeMap::<String, usize>::new();
    for split_name in SPLITS {
        eligible_components_by_split.insert(split_name.to_owned(), 0);
        all_components_by_split.insert(split_name.to_owned(), 0);
        eligible_target_count.insert(split_name.to_owned(), 0);
        excluded_target_count.insert(split_name.to_owned(), 0);
        unknown_target_count.insert(split_name.to_owned(), 0);
    }

    let mut unknown_family_ids = BTreeSet::new();
    for (family_id, status_counts) in &family_status_by_id {
        let split_name = &split.family_to_split[family_id];
        *eligible_target_count.get_mut(split_name).expect("registered split") += status_counts.0;
        *excluded_target_count.get_mut(split_name).expect("registered split") += status_counts.1;
        *unknown_target_count.get_mut(split_name).expect("registered split") += status_counts.2;
        if status_counts.2 > 0 {
            unknown_family_ids.insert(family_id.clone());
        }
    }
    for family in &census.families {
        if family.eligibility_status == "unknown" {
            unknown_family_ids.insert(family.family_id.clone());
        }
    }
    let unknown_family_count = unknown_family_ids.len();
    for component in split.components.values() {
        *all_components_by_split.get_mut(&component.split).expect("registered split") += 1;
        let has_eligible_target = component.family_ids.iter().any(|family_id| {
            family_status_by_id.get(family_id).is_some_and(|counts| counts.0 > 0)
        });
        if has_eligible_target {
            *eligible_components_by_split.get_mut(&component.split).expect("registered split") += 1;
        }
    }

    let validation_eligible_components = eligible_components_by_split["validation"];
    let test_eligible_components = eligible_components_by_split["test"];
    let structural_floor_met = validation_eligible_components >= MIN_HELDOUT_COMPONENTS
        && test_eligible_components >= MIN_HELDOUT_COMPONENTS;
    let no_unknown_targets = unknown_target_count.values().all(|count| *count == 0)
        && unknown_family_count == 0;
    let structurally_feasible_for_power_planning = census.dataset_kind == "reviewed_census"
        && census.sealed_before_evaluation
        && !census.outcomes_used
        && structural_floor_met
        && no_unknown_targets;

    let mut blockers = vec![
        "This audit is structural and never authorizes confirmatory analysis.".to_owned(),
        "Receipt digests are validated as identifiers but this tool does not authenticate reviewer signatures.".to_owned(),
        "Simulation-based power, statement fidelity, graph coverage, and the primary analyzer gates remain separate requirements.".to_owned(),
    ];
    if !structural_floor_met {
        blockers.push(format!(
            "Held-out eligible-component floor failed: validation={validation_eligible_components}, test={test_eligible_components}, minimum={MIN_HELDOUT_COMPONENTS} each."
        ));
    }
    if !no_unknown_targets {
        blockers.push(format!("Eligibility census contains unknown target/family status; unknown families/target-bearing families counted={unknown_family_count}."));
    }
    if census.dataset_kind != "reviewed_census" {
        blockers.push("Synthetic fixture is not an eligible benchmark census.".to_owned());
    }

    let report = json!({
        "schema_version": "pitb.component-split-feasibility-report.v0",
        "status": "diagnostic_only",
        "candidate_dataset_kind": split.dataset_kind,
        "candidate_split_sha256": split.bytes_sha256,
        "candidate_partition_digest_sha256": split.partition_digest_sha256,
        "eligibility_census_sha256": sha256(&census_bytes),
        "family_universe_sha256": census.family_universe_sha256,
        "eligibility_policy_sha256": census.eligibility_policy_sha256,
        "dataset_kind": census.dataset_kind,
        "outcomes_used": census.outcomes_used,
        "family_count": split.family_to_split.len(),
        "component_count": split.components.len(),
        "component_count_by_split": all_components_by_split,
        "eligible_component_count_by_split": eligible_components_by_split,
        "eligible_target_count_by_split": eligible_target_count,
        "excluded_target_count_by_split": excluded_target_count,
        "unknown_target_count_by_split": unknown_target_count,
        "minimum_eligible_components_per_validation_and_test": MIN_HELDOUT_COMPONENTS,
        "heldout_structural_floor_met": structural_floor_met,
        "unknown_eligibility_present": !no_unknown_targets,
        "structurally_feasible_for_power_planning": structurally_feasible_for_power_planning,
        "qualified_for_confirmatory_use": false,
        "blockers": blockers
    });
    fs::write(output_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write report {output_path}"))?;
    println!(
        "PITB_COMPONENT_SPLIT_FEASIBILITY_DIAGNOSTIC: validation_eligible_components={} test_eligible_components={} heldout_floor_met={} confirmatory=false",
        validation_eligible_components,
        test_eligible_components,
        structural_floor_met
    );
    Ok(())
}

fn main() -> ExitCode {
    match execute() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("component split feasibility audit rejected input: {error:#}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SYNTHETIC_SPLIT: &[u8] = br#"{
      "schema_version":"pitb.component-first-split-candidate.v0",
      "dataset_kind":"synthetic_fixture",
      "status":"candidate_only",
      "upstream_commit":"1581d4cbdd69a0faf34b2089614fe2a814316a6a",
      "family_count":6,
      "component_count":6,
      "family_universe_sha256":"c6b39a37aa42bdd454f15806269ca1d0d417cd4823ec7a3db80908d6214f4dc",
      "source_v0_partition_digest_sha256":"e7ac417556cbd47cee1ddd51b98f9684fadfc8cd097587ae3beacc4dfb59f970",
      "source_split_manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "source_component_manifest_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "relationship_inventory_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      "family_count_by_split":{"train":4,"validation":1,"test":1},
      "component_count_by_split":{"train":4,"validation":1,"test":1},
      "target_component_count_by_split":{"train":4,"validation":1,"test":1},
      "candidate_partition_digest_sha256":"PLACEHOLDER",
      "qualified_for_confirmatory_use":false,
      "families":[
        {"family_id":"a","split":"train","component_id":"component-a"},
        {"family_id":"b","split":"train","component_id":"component-b"},
        {"family_id":"c","split":"train","component_id":"component-c"},
        {"family_id":"d","split":"train","component_id":"component-d"},
        {"family_id":"e","split":"validation","component_id":"component-e"},
        {"family_id":"f","split":"test","component_id":"component-f"}
      ],
      "components":[
        {"component_id":"component-a","split":"train","family_count":1,"family_ids":["a"]},
        {"component_id":"component-b","split":"train","family_count":1,"family_ids":["b"]},
        {"component_id":"component-c","split":"train","family_count":1,"family_ids":["c"]},
        {"component_id":"component-d","split":"train","family_count":1,"family_ids":["d"]},
        {"component_id":"component-e","split":"validation","family_count":1,"family_ids":["e"]},
        {"component_id":"component-f","split":"test","family_count":1,"family_ids":["f"]}
      ]
    }"#;

    const SYNTHETIC_CENSUS: &[u8] = br#"{
      "schema_version":"pitb.pre-split-target-eligibility-census.v0",
      "dataset_kind":"synthetic_fixture",
      "upstream_commit":"1581d4cbdd69a0faf34b2089614fe2a814316a6a",
      "family_universe_sha256":"c6b39a37aa42bdd454f15806269ca1d0d417cd4823ec7a3db809908d6214f4dc",
      "eligibility_policy_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "census_receipt_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "sealed_before_evaluation":true,
      "outcomes_used":false,
      "families":[
        {"family_id":"a","eligibility_status":"eligible","candidate_target_ids":["t-a"],"eligibility_receipt_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"},
        {"family_id":"b","eligibility_status":"eligible","candidate_target_ids":["t-b"],"eligibility_receipt_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"},
        {"family_id":"c","eligibility_status":"eligible","candidate_target_ids":["t-c"],"eligibility_receipt_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"},
        {"family_id":"d","eligibility_status":"excluded","candidate_target_ids":["t-d"],"eligibility_receipt_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"},
        {"family_id":"e","eligibility_status":"eligible","candidate_target_ids":["t-e"],"eligibility_receipt_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"},
        {"family_id":"f","eligibility_status":"unknown","candidate_target_ids":["t-f"],"eligibility_receipt_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"}
      ],
      "targets":[
        {"target_id":"t-a","family_id":"a","status":"eligible","artifact_path":"families/a.lean","artifact_sha256":"1111111111111111111111111111111111111111111111111111111111111111","statement_sha256":"2222222222222222222222222222222222222222222222222222222222222222","eligibility_receipt_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"},
        {"target_id":"t-b","family_id":"b","status":"eligible","artifact_path":"families/b.lean","artifact_sha256":"1111111111111111111111111111111111111111111111111111111111111111","statement_sha256":"2222222222222222222222222222222222222222222222222222222222222222","eligibility_receipt_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"},
        {"target_id":"t-c","family_id":"c","status":"eligible","artifact_path":"families/c.lean","artifact_sha256":"1111111111111111111111111111111111111111111111111111111111111111","statement_sha256":"2222222222222222222222222222222222222222222222222222222222222222","eligibility_receipt_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"},
        {"target_id":"t-d","family_id":"d","status":"excluded","artifact_path":"families/d.lean","artifact_sha256":"1111111111111111111111111111111111111111111111111111111111111111","statement_sha256":null,"eligibility_receipt_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"},
        {"target_id":"t-e","family_id":"e","status":"eligible","artifact_path":"families/e.lean","artifact_sha256":"1111111111111111111111111111111111111111111111111111111111111111","statement_sha256":"2222222222222222222222222222222222222222222222222222222222222222","eligibility_receipt_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"},
        {"target_id":"t-f","family_id":"f","status":"unknown","artifact_path":"families/f.lean","artifact_sha256":"1111111111111111111111111111111111111111111111111111111111111111","statement_sha256":null,"eligibility_receipt_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"}
      ]
    }"#;

    fn candidate_with_correct_digest() -> Vec<u8> {
        let value: Value = serde_json::from_slice(SYNTHETIC_SPLIT).expect("valid JSON fixture");
        let mut value = value;
        let rows = value["families"].as_array().expect("families");
        let mut assignments = BTreeMap::new();
        for row in rows {
            assignments.insert(
                row["family_id"].as_str().expect("family ID").to_owned(),
                row["split"].as_str().expect("split").to_owned(),
            );
        }
        value["candidate_partition_digest_sha256"] = json!(partition_digest(&assignments));
        serde_json::to_vec(&value).expect("serialize candidate fixture")
    }

    #[test]
    fn census_reconciliation_keeps_unknown_and_excluded_rows_visible() {
        let split_bytes = candidate_with_correct_digest();
        let split = parse_candidate_split(&split_bytes).expect("valid candidate split");
        let census: EligibilityCensus =
            serde_json::from_slice(SYNTHETIC_CENSUS).expect("valid census fixture");
        validate_census(&census, &split).expect("census reconciles");
        let eligible: BTreeSet<&str> = census.targets.iter()
            .filter(|target| target.status == "eligible")
            .map(|target| target.family_id.as_str())
            .collect();
        assert_eq!(eligible, BTreeSet::from(["a", "b", "c", "e"]));
        assert!(!census.outcomes_used);
    }

    #[test]
    fn census_with_outcome_derived_eligibility_is_rejected() {
        let split_bytes = candidate_with_correct_digest();
        let split = parse_candidate_split(&split_bytes).expect("valid candidate split");
        let mut census: EligibilityCensus =
            serde_json::from_slice(SYNTHETIC_CENSUS).expect("valid census fixture");
        census.outcomes_used = true;
        assert!(validate_census(&census, &split).is_err());
    }

    #[test]
    fn family_target_index_must_match_target_rows() {
        let split_bytes = candidate_with_correct_digest();
        let split = parse_candidate_split(&split_bytes).expect("valid candidate split");
        let mut census: EligibilityCensus =
            serde_json::from_slice(SYNTHETIC_CENSUS).expect("valid census fixture");
        census.families[0].candidate_target_ids = vec!["t-b".to_owned()];
        assert!(validate_census(&census, &split).is_err());
    }

    #[test]
    fn unqualified_or_wrongly_digested_split_is_rejected() {
        let mut doc: Value = serde_json::from_slice(SYNTHETIC_SPLIT).expect("valid JSON fixture");
        doc["qualified_for_confirmatory_use"] = json!(true);
        let bytes = serde_json::to_vec(&doc).expect("serialize fixture");
        assert!(parse_candidate_split(&bytes).is_err());

        let mut bad_digest: Value =
            serde_json::from_slice(SYNTHETIC_SPLIT).expect("valid JSON fixture");
        bad_digest["candidate_partition_digest_sha256"] = json!("0".repeat(64));
        let bytes = serde_json::to_vec(&bad_digest).expect("serialize fixture");
        assert!(parse_candidate_split(&bytes).is_err());

        let mut bad_quotas: Value = serde_json::from_slice(&candidate_with_correct_digest())
            .expect("valid candidate fixture");
        bad_quotas["component_count_by_split"]["train"] = json!(3);
        let bytes = serde_json::to_vec(&bad_quotas).expect("serialize fixture");
        assert!(parse_candidate_split(&bytes).is_err());

        let mut bad_universe: Value = serde_json::from_slice(SYNTHETIC_SPLIT)
            .expect("valid candidate fixture");
        bad_universe["family_universe_sha256"] = json!("0".repeat(64));
        let bytes = serde_json::to_vec(&bad_universe).expect("serialize fixture");
        assert!(parse_candidate_split(&bytes).is_err());

        let mut false_real_candidate: Value =
            serde_json::from_slice(&candidate_with_correct_digest())
                .expect("valid synthetic candidate fixture");
        false_real_candidate["dataset_kind"] = json!("candidate_only");
        false_real_candidate["family_universe_sha256"] =
            json!(EXPECTED_FAMILY_UNIVERSE_SHA256);
        let bytes = serde_json::to_vec(&false_real_candidate).expect("serialize fixture");
        assert!(
            parse_candidate_split(&bytes).is_err(),
            "a synthetic six-family universe must not pass as the real 372-family universe"
        );
    }
}
