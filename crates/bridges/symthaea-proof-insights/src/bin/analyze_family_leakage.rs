//! Deterministic, non-destructive cross-family leakage audit for PITB v0.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    env,
    fs,
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

const UPSTREAM_COMMIT: &str = "1581d4cbdd69a0faf34b2089614fe2a814316a6a";
const SPLIT_SEED: &str = "pitb-v0-openai-math-2026-10-19";
const SPLIT_ALGORITHM: &str = "sha256(seed || NUL || family_id), sort by digest then family ID, floor 70% train and floor 85% train+validation";
const EXPECTED_PARTITION_DIGEST: &str = "e7ac417556cbd47cee1ddd51b98f9684fadfc8cd097587ae3beacc4dfb59f970";
const EXPECTED_FAMILY_COUNTS: (usize, usize, usize) = (260, 56, 56);

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Inventory {
    schema_version: String,
    source_partition_digest_sha256: String,
    coverage_status: String,
    coverage_receipt_sha256: Value,
    edges: Vec<Edge>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Edge {
    edge_id: String,
    source_family_id: String,
    target_family_id: String,
    relation_kind: String,
    assessment_status: String,
    evidence_sha256: String,
    reviewer_ids: Vec<String>,
}

#[derive(Debug, Default, Serialize)]
struct SplitCounts {
    train: usize,
    validation: usize,
    test: usize,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: &'static str,
    source_partition_digest_sha256: String,
    relationship_inventory_status: String,
    coverage_receipt_sha256: Option<String>,
    total_family_count: usize,
    original_family_counts: SplitCounts,
    known_quarantined_family_count: usize,
    known_quarantined_family_counts: SplitCounts,
    remaining_family_counts_after_known_quarantine: SplitCounts,
    edge_counts: EdgeCounts,
    cross_split_components: Vec<Component>,
    quarantined_families: Vec<QuarantinedFamily>,
    certification_status: String,
}

#[derive(Debug, Serialize)]
struct EdgeCounts {
    total_edges: usize,
    confirmed_hard_edges: usize,
    unresolved_hard_edges: usize,
    dismissed_hard_edges: usize,
    non_hard_edges: usize,
    confirmed_cross_split_hard_edges: usize,
    unresolved_cross_split_hard_edges: usize,
}

#[derive(Debug, Serialize)]
struct Component {
    family_ids: Vec<String>,
    split_names: Vec<String>,
    edge_ids: Vec<String>,
    confirmed_edge_ids: Vec<String>,
    unresolved_edge_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
struct QuarantinedFamily {
    family_id: String,
    split: String,
    reason_edge_ids: Vec<String>,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn deterministic_splits(ids: &[String]) -> Result<BTreeMap<String, String>> {
    if ids.len() < 3 {
        bail!("at least three families are required");
    }
    let mut unique = BTreeSet::new();
    for id in ids {
        if id.trim().is_empty() || !unique.insert(id.clone()) {
            bail!("family IDs must be nonblank and unique");
        }
    }
    let mut ranked: Vec<(String, String)> = ids
        .iter()
        .map(|id| (sha256(format!("{SPLIT_SEED}\0{id}").as_bytes()), id.clone()))
        .collect();
    ranked.sort_unstable();

    let train_end = ranked.len() * 70 / 100;
    let validation_end = ranked.len() * 85 / 100;
    Ok(ranked
        .into_iter()
        .enumerate()
        .map(|(index, (_, id))| {
            let split = if index < train_end {
                "train"
            } else if index < validation_end {
                "validation"
            } else {
                "test"
            };
            (id, split.to_owned())
        })
        .collect())
}

fn partition_digest(splits: &BTreeMap<String, String>) -> String {
    let mut material = String::new();
    for (id, split) in splits {
        material.push_str(id);
        material.push('\t');
        material.push_str(split);
        material.push('\n');
    }
    sha256(material.as_bytes())
}

fn validate_split(doc: &Value) -> Result<(String, BTreeMap<String, String>)> {
    if doc["schema_version"].as_str()
        != Some("symthaea.proof-insight-transfer.splits.v0")
        || doc["upstream"]["commit"].as_str() != Some(UPSTREAM_COMMIT)
    {
        bail!("split manifest does not identify the pinned v0 catalogue");
    }
    let plan = &doc["split_plan"];
    if plan["seed"].as_str() != Some(SPLIT_SEED)
        || plan["algorithm"].as_str() != Some(SPLIT_ALGORITHM)
        || plan["train_percent"].as_u64() != Some(70)
        || plan["validation_percent"].as_u64() != Some(15)
        || plan["test_percent"].as_u64() != Some(15)
    {
        bail!("split plan differs from registered v0 settings");
    }

    let families = doc["families"].as_array().context("families must be an array")?;
    let mut actual = BTreeMap::new();
    for row in families {
        let id = row["family_id"].as_str().context("family_id is missing")?;
        let split = row["split"].as_str().context("family split is missing")?;
        if !matches!(split, "train" | "validation" | "test")
            || actual.insert(id.to_owned(), split.to_owned()).is_some()
            || id.trim().is_empty()
        {
            bail!("invalid or duplicate family record: {id}");
        }
    }
    let ids: Vec<String> = actual.keys().cloned().collect();
    if actual != deterministic_splits(&ids)? {
        bail!("family assignments do not match the registered deterministic algorithm");
    }
    let digest = partition_digest(&actual);
    if !valid_digest(plan["partition_digest_sha256"].as_str().unwrap_or(""))
        || plan["partition_digest_sha256"].as_str() != Some(digest.as_str())
        || digest != EXPECTED_PARTITION_DIGEST
        || actual.len() != 372
    {
        bail!("partition digest does not match computed assignments");
    }

    let counts = split_counts(&actual, None);
    for (name, count) in [
        ("train", counts.train),
        ("validation", counts.validation),
        ("test", counts.test),
    ] {
        if plan["counts"][name].as_u64() != Some(count as u64) {
            bail!("declared {name} count differs from family assignments");
        }
    }
    if (counts.train, counts.validation, counts.test) != EXPECTED_FAMILY_COUNTS {
        bail!("family counts differ from the frozen PITB v0 partition");
    }
    Ok((digest, actual))
}

fn hard_relation(kind: &str) -> bool {
    matches!(
        kind,
        "exact_duplicate"
            | "shared_source_artifact"
            | "formal_dependency"
            | "reused_lemma"
            | "equivalent_statement"
    )
}

fn validate_inventory(
    inv: &Inventory,
    digest: &str,
    splits: &BTreeMap<String, String>,
) -> Result<Option<String>> {
    if inv.schema_version != "pitb.family-leakage-edges.v0"
        || !valid_digest(&inv.source_partition_digest_sha256)
        || inv.source_partition_digest_sha256 != digest
    {
        bail!("relationship inventory schema or frozen-partition binding is invalid");
    }

    let coverage_receipt = if inv.coverage_receipt_sha256.is_null() {
        None
    } else {
        let value = inv.coverage_receipt_sha256.as_str().context(
            "coverage_receipt_sha256 must be a SHA-256 digest or null",
        )?;
        if !valid_digest(value) {
            bail!("coverage receipt digest is malformed");
        }
        Some(value.to_owned())
    };
    match inv.coverage_status.as_str() {
        "incomplete" if coverage_receipt.is_none() => {}
        "incomplete" => bail!("incomplete inventory must have a null coverage receipt"),
        "independently_reviewed_complete" if coverage_receipt.is_some() => {}
        "independently_reviewed_complete" => {
            bail!("complete inventory must reference a coverage receipt")
        }
        status => bail!("unknown inventory coverage status: {status}"),
    }

    let mut edge_ids = BTreeSet::new();
    let mut pair_kinds = BTreeSet::new();
    for edge in &inv.edges {
        if edge.edge_id.trim().is_empty() || !edge_ids.insert(edge.edge_id.clone()) {
            bail!("edge IDs must be nonblank and unique");
        }
        if edge.source_family_id == edge.target_family_id
            || !splits.contains_key(&edge.source_family_id)
            || !splits.contains_key(&edge.target_family_id)
        {
            bail!("edge {} has a self-link or unknown family", edge.edge_id);
        }
        if !valid_digest(&edge.evidence_sha256) {
            bail!("edge {} has a malformed evidence digest", edge.edge_id);
        }
        if !matches!(
            edge.relation_kind.as_str(),
            "exact_duplicate"
                | "shared_source_artifact"
                | "formal_dependency"
                | "reused_lemma"
                | "equivalent_statement"
                | "conceptual_similarity"
                | "citation_only"
        ) || !matches!(
            edge.assessment_status.as_str(),
            "confirmed" | "dismissed" | "unresolved"
        ) {
            bail!("edge {} has an unregistered relation or status", edge.edge_id);
        }

        let mut reviewers = BTreeSet::new();
        for id in &edge.reviewer_ids {
            if id.trim().is_empty() || !reviewers.insert(id) {
                bail!("edge {} contains a blank or duplicate reviewer ID", edge.edge_id);
            }
        }
        if edge.assessment_status != "unresolved" && reviewers.len() < 2 {
            bail!("resolved edge {} requires two distinct reviewer IDs", edge.edge_id);
        }

        let (left, right) = if edge.source_family_id < edge.target_family_id {
            (&edge.source_family_id, &edge.target_family_id)
        } else {
            (&edge.target_family_id, &edge.source_family_id)
        };
        if !pair_kinds.insert(format!("{left}\0{right}\0{}", edge.relation_kind)) {
            bail!("duplicate unordered family-pair/relation-kind edge");
        }
    }
    Ok(coverage_receipt)
}

fn split_counts(
    splits: &BTreeMap<String, String>,
    only: Option<&BTreeSet<String>>,
) -> SplitCounts {
    let mut counts = SplitCounts::default();
    for (id, split) in splits {
        if only.is_some_and(|selected| !selected.contains(id)) {
            continue;
        }
        match split.as_str() {
            "train" => counts.train += 1,
            "validation" => counts.validation += 1,
            "test" => counts.test += 1,
            _ => unreachable!("split was validated"),
        }
    }
    counts
}

fn analyze(doc: &Value, inv: &Inventory) -> Result<Report> {
    let (digest, splits) = validate_split(doc)?;
    let coverage_receipt = validate_inventory(inv, &digest, &splits)?;
    let mut counts = EdgeCounts {
        total_edges: inv.edges.len(),
        confirmed_hard_edges: 0,
        unresolved_hard_edges: 0,
        dismissed_hard_edges: 0,
        non_hard_edges: 0,
        confirmed_cross_split_hard_edges: 0,
        unresolved_cross_split_hard_edges: 0,
    };
    let mut adjacency: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for edge in &inv.edges {
        if !hard_relation(&edge.relation_kind) {
            counts.non_hard_edges += 1;
            continue;
        }
        match edge.assessment_status.as_str() {
            "confirmed" => counts.confirmed_hard_edges += 1,
            "unresolved" => counts.unresolved_hard_edges += 1,
            "dismissed" => counts.dismissed_hard_edges += 1,
            _ => unreachable!("status was validated"),
        }
        let cross = splits[&edge.source_family_id] != splits[&edge.target_family_id];
        match (cross, edge.assessment_status.as_str()) {
            (true, "confirmed") => counts.confirmed_cross_split_hard_edges += 1,
            (true, "unresolved") => counts.unresolved_cross_split_hard_edges += 1,
            _ => {}
        }
        if edge.assessment_status != "dismissed" {
            adjacency
                .entry(edge.source_family_id.clone())
                .or_default()
                .insert(edge.target_family_id.clone());
            adjacency
                .entry(edge.target_family_id.clone())
                .or_default()
                .insert(edge.source_family_id.clone());
        }
    }

    let mut visited = BTreeSet::new();
    let mut components = Vec::new();
    let mut quarantine: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for start in splits.keys() {
        if visited.contains(start) || !adjacency.contains_key(start) {
            continue;
        }
        let mut queue = VecDeque::from([start.clone()]);
        let mut ids = BTreeSet::new();
        while let Some(id) = queue.pop_front() {
            if !visited.insert(id.clone()) {
                continue;
            }
            ids.insert(id.clone());
            if let Some(neighbours) = adjacency.get(&id) {
                for neighbour in neighbours {
                    if !visited.contains(neighbour) {
                        queue.push_back(neighbour.clone());
                    }
                }
            }
        }

        let split_names: BTreeSet<String> = ids.iter().map(|id| splits[id].clone()).collect();
        if split_names.len() < 2 {
            continue;
        }
        let related: Vec<&Edge> = inv
            .edges
            .iter()
            .filter(|edge| {
                hard_relation(&edge.relation_kind)
                    && edge.assessment_status != "dismissed"
                    && ids.contains(&edge.source_family_id)
                    && ids.contains(&edge.target_family_id)
            })
            .collect();
        let edge_ids: BTreeSet<String> =
            related.iter().map(|edge| edge.edge_id.clone()).collect();
        let confirmed: BTreeSet<String> = related
            .iter()
            .filter(|edge| edge.assessment_status == "confirmed")
            .map(|edge| edge.edge_id.clone())
            .collect();
        let unresolved: BTreeSet<String> = related
            .iter()
            .filter(|edge| edge.assessment_status == "unresolved")
            .map(|edge| edge.edge_id.clone())
            .collect();
        for id in &ids {
            quarantine.entry(id.clone()).or_default().extend(edge_ids.iter().cloned());
        }
        components.push(Component {
            family_ids: ids.into_iter().collect(),
            split_names: split_names.into_iter().collect(),
            edge_ids: edge_ids.into_iter().collect(),
            confirmed_edge_ids: confirmed.into_iter().collect(),
            unresolved_edge_ids: unresolved.into_iter().collect(),
        });
    }
    components.sort_by(|left, right| left.family_ids.cmp(&right.family_ids));

    let quarantine_ids: BTreeSet<String> = quarantine.keys().cloned().collect();
    let original = split_counts(&splits, None);
    let removed = split_counts(&splits, Some(&quarantine_ids));
    let remaining = SplitCounts {
        train: original.train.saturating_sub(removed.train),
        validation: original.validation.saturating_sub(removed.validation),
        test: original.test.saturating_sub(removed.test),
    };
    let quarantined_families = quarantine_ids
        .iter()
        .map(|id| QuarantinedFamily {
            family_id: id.clone(),
            split: splits[id].clone(),
            reason_edge_ids: quarantine[id].iter().cloned().collect(),
        })
        .collect();

    let status = if inv.coverage_status == "incomplete" {
        "not_certified_incomplete_inventory"
    } else if !components.is_empty() {
        "cross_split_hard_relationships_require_quarantine"
    } else {
        "no_cross_split_hard_relationships_in_declared_inventory"
    };
    Ok(Report {
        schema_version: "pitb.family-leakage-report.v0",
        source_partition_digest_sha256: digest,
        relationship_inventory_status: inv.coverage_status.clone(),
        coverage_receipt_sha256: coverage_receipt,
        total_family_count: splits.len(),
        original_family_counts: original,
        known_quarantined_family_count: quarantine_ids.len(),
        known_quarantined_family_counts: removed,
        remaining_family_counts_after_known_quarantine: remaining,
        edge_counts: counts,
        cross_split_components: components,
        quarantined_families,
        certification_status: status.to_owned(),
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 3 {
        bail!("usage: analyze_family_leakage <splits.json> <edges.json> <report.json>");
    }
    let split: Value = serde_json::from_slice(
        &fs::read(&args[0]).with_context(|| format!("cannot read {}", args[0]))?,
    )
    .context("split manifest is not valid JSON")?;
    let inventory: Inventory = serde_json::from_slice(
        &fs::read(&args[1]).with_context(|| format!("cannot read {}", args[1]))?,
    )
    .context("relationship inventory is not valid JSON or violates the strict Rust shape")?;
    let report = analyze(&split, &inventory)?;
    fs::write(&args[2], serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("cannot write {}", args[2]))?;
    println!(
        "PITB_FAMILY_LEAKAGE_AUDIT: status={} quarantined={} components={} coverage={}",
        report.certification_status,
        report.known_quarantined_family_count,
        report.cross_split_components.len(),
        report.relationship_inventory_status
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_family_split_is_stable_and_balanced() {
        let ids = vec!["a".to_owned(), "b".to_owned(), "c".to_owned(), "d".to_owned()];
        let first = deterministic_splits(&ids).expect("first split");
        let second = deterministic_splits(&ids).expect("second split");
        assert_eq!(first, second);
        let counts = split_counts(&first, None);
        assert_eq!((counts.train, counts.validation, counts.test), (2, 1, 1));
        assert_eq!(partition_digest(&first), partition_digest(&second));
    }

    #[test]
    fn hard_leakage_requires_a_registered_structural_relation() {
        assert!(hard_relation("exact_duplicate"));
        assert!(hard_relation("formal_dependency"));
        assert!(hard_relation("reused_lemma"));
        assert!(!hard_relation("conceptual_similarity"));
        assert!(!hard_relation("citation_only"));
    }

    #[test]
    fn digest_shape_is_fail_closed() {
        assert!(valid_digest(&"a".repeat(64)));
        assert!(!valid_digest(&"a".repeat(63)));
        assert!(!valid_digest(&format!("{}z", "a".repeat(63))));
    }
}
