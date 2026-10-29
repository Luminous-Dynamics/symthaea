//! Complete, deterministic PITB family-to-component mapping for conservative resampling.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    env, fs,
};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[path = "../coverage_receipt.rs"]
mod coverage_receipt;
use coverage_receipt::verify_receipt;

const UPSTREAM: &str = "1581d4cbdd69a0faf34b2089614fe2a814316a6a";
const SEED: &str = "pitb-v0-openai-math-2026-10-19";
const ALGORITHM: &str = concat!(
    "sha256(seed || NUL || family_id), sort by digest then family ID, ",
    "floor 70% train and floor 85% train+validation"
);
const PARTITION: &str = "e7ac417556cbd47cee1ddd51b98f9684fadfc8cd097587ae3beacc4dfb59f970";

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

#[derive(Debug, Serialize)]
struct Counts {
    train: usize,
    validation: usize,
    test: usize,
}

#[derive(Debug, Serialize)]
struct EdgeCounts {
    total_edges: usize,
    hard_edges_included: usize,
    confirmed_hard_edges: usize,
    unresolved_hard_edges: usize,
    dismissed_hard_edges: usize,
    soft_edges: usize,
    cross_split_hard_edges: usize,
    unresolved_cross_split_hard_edges: usize,
}

#[derive(Debug, Serialize)]
struct Component {
    component_id: String,
    split_names: Vec<String>,
    family_ids: Vec<String>,
    hard_edge_ids: Vec<String>,
    confirmed_edge_ids: Vec<String>,
    unresolved_edge_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
struct FamilyRow {
    family_id: String,
    split: String,
    component_id: String,
}

#[derive(Debug, Serialize)]
struct Manifest {
    schema_version: &'static str,
    upstream_commit: String,
    source_partition_digest_sha256: String,
    relationship_inventory_sha256: String,
    relationship_inventory_status: String,
    coverage_receipt_sha256: Option<String>,
    total_family_count: usize,
    total_component_count: usize,
    independent_component_count_by_split: Counts,
    cross_split_component_count: usize,
    edge_counts: EdgeCounts,
    families: Vec<FamilyRow>,
    components: Vec<Component>,
    qualification_status: String,
    coverage_receipt_verification_status: &'static str,
    qualified_for_confirmatory_resampling: bool,
    qualification_limitations: Vec<String>,
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn digest_ok(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn hard(relation: &str) -> bool {
    matches!(
        relation,
        "exact_duplicate"
            | "shared_source_artifact"
            | "formal_dependency"
            | "reused_lemma"
            | "equivalent_statement"
    )
}

fn deterministic(ids: &[String]) -> Result<BTreeMap<String, String>> {
    if ids.len() < 3 {
        bail!("at least three family IDs required");
    }
    let mut unique = BTreeSet::new();
    for id in ids {
        if id.trim().is_empty() || !unique.insert(id.clone()) {
            bail!("family IDs must be unique and nonblank");
        }
    }

    let mut ranked: Vec<(String, String)> = ids
        .iter()
        .map(|id| (sha(format!("{SEED}\0{id}").as_bytes()), id.clone()))
        .collect();
    ranked.sort_unstable();
    let n = ranked.len();
    let train_end = n * 70 / 100;
    let validation_end = n * 85 / 100;

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
    for (family_id, split) in splits {
        material.push_str(family_id);
        material.push('\t');
        material.push_str(split);
        material.push('\n');
    }
    sha(material.as_bytes())
}

fn validate_splits(doc: &Value) -> Result<(String, BTreeMap<String, String>)> {
    if doc["schema_version"].as_str() != Some("symthaea.proof-insight-transfer.splits.v0")
        || doc["upstream"]["commit"].as_str() != Some(UPSTREAM)
    {
        bail!("wrong pinned split manifest");
    }
    let plan = &doc["split_plan"];
    if plan["seed"].as_str() != Some(SEED)
        || plan["algorithm"].as_str() != Some(ALGORITHM)
        || plan["train_percent"].as_u64() != Some(70)
        || plan["validation_percent"].as_u64() != Some(15)
        || plan["test_percent"].as_u64() != Some(15)
    {
        bail!("unregistered split plan");
    }
    let rows = doc["families"]
        .as_array()
        .context("families must be an array")?;
    let mut splits = BTreeMap::new();
    for row in rows {
        let family_id = row["family_id"].as_str().context("missing family_id")?;
        let split = row["split"].as_str().context("missing split")?;
        if family_id.trim().is_empty()
            || !matches!(split, "train" | "validation" | "test")
            || splits
                .insert(family_id.to_owned(), split.to_owned())
                .is_some()
        {
            bail!("invalid/duplicate family row");
        }
    }
    let ids: Vec<String> = splits.keys().cloned().collect();
    if splits != deterministic(&ids)? {
        bail!("split does not reproduce deterministic assignment");
    }

    let digest = partition_digest(&splits);
    if digest != PARTITION
        || plan["partition_digest_sha256"].as_str() != Some(digest.as_str())
        || splits.len() != 372
    {
        bail!("pinned partition digest/count mismatch");
    }
    let counts = (
        splits.values().filter(|split| *split == "train").count(),
        splits
            .values()
            .filter(|split| *split == "validation")
            .count(),
        splits.values().filter(|split| *split == "test").count(),
    );
    if counts != (260, 56, 56)
        || plan["counts"]["train"].as_u64() != Some(260)
        || plan["counts"]["validation"].as_u64() != Some(56)
        || plan["counts"]["test"].as_u64() != Some(56)
    {
        bail!("split counts do not reconcile");
    }
    Ok((digest, splits))
}

fn validate_edges(
    inventory: &Inventory,
    digest: &str,
    splits: &BTreeMap<String, String>,
) -> Result<Option<String>> {
    if inventory.schema_version != "pitb.family-leakage-edges.v0"
        || !digest_ok(&inventory.source_partition_digest_sha256)
        || inventory.source_partition_digest_sha256 != digest
    {
        bail!("edge inventory schema/partition mismatch");
    }

    let receipt = if inventory.coverage_receipt_sha256.is_null() {
        None
    } else {
        let value = inventory
            .coverage_receipt_sha256
            .as_str()
            .context("receipt must be a digest or null")?;
        if !digest_ok(value) {
            bail!("bad coverage receipt digest");
        }
        Some(value.to_owned())
    };
    match inventory.coverage_status.as_str() {
        "incomplete" if receipt.is_none() => {}
        "incomplete" => bail!("incomplete inventory cannot have receipt"),
        "independently_reviewed_complete" if receipt.is_some() => {}
        "independently_reviewed_complete" => bail!("complete inventory requires receipt"),
        other => bail!("unknown coverage status {other}"),
    }

    let mut edge_ids = BTreeSet::new();
    let mut pairs = BTreeSet::new();
    for edge in &inventory.edges {
        if edge.edge_id.trim().is_empty() || !edge_ids.insert(edge.edge_id.clone()) {
            bail!("duplicate/blank edge ID");
        }
        if edge.source_family_id == edge.target_family_id
            || !splits.contains_key(&edge.source_family_id)
            || !splits.contains_key(&edge.target_family_id)
        {
            bail!("self-edge or unknown family");
        }
        if !digest_ok(&edge.evidence_sha256) {
            bail!("bad evidence digest");
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
            bail!("unregistered edge relation/status");
        }
        let reviewers: BTreeSet<&str> =
            edge.reviewer_ids.iter().map(String::as_str).collect();
        if edge.reviewer_ids.iter().any(|reviewer| reviewer.trim().is_empty())
            || reviewers.len() != edge.reviewer_ids.len()
        {
            bail!("invalid reviewer IDs");
        }
        if edge.assessment_status != "unresolved" && reviewers.len() < 2 {
            bail!("resolved edge needs two reviewers");
        }
        let (left, right) = if edge.source_family_id < edge.target_family_id {
            (&edge.source_family_id, &edge.target_family_id)
        } else {
            (&edge.target_family_id, &edge.source_family_id)
        };
        if !pairs.insert(format!("{left}\0{right}\0{}", edge.relation_kind)) {
            bail!("duplicate unordered pair/relation edge");
        }
    }
    Ok(receipt)
}

fn component_id(ids: &[String]) -> String {
    let mut material = ids.join("\n");
    material.push('\n');
    format!("component-{}", sha(material.as_bytes()))
}

fn components(splits: &BTreeMap<String, String>, edges: &[Edge]) -> Vec<Component> {
    let mut adjacency: BTreeMap<String, BTreeSet<String>> = splits
        .keys()
        .map(|family_id| (family_id.clone(), BTreeSet::new()))
        .collect();
    for edge in edges {
        if hard(&edge.relation_kind) && edge.assessment_status != "dismissed" {
            adjacency
                .get_mut(&edge.source_family_id)
                .unwrap()
                .insert(edge.target_family_id.clone());
            adjacency
                .get_mut(&edge.target_family_id)
                .unwrap()
                .insert(edge.source_family_id.clone());
        }
    }

    let mut seen = BTreeSet::new();
    let mut output = Vec::new();
    for start in splits.keys() {
        if !seen.insert(start.clone()) {
            continue;
        }
        let mut queue = VecDeque::from([start.clone()]);
        let mut members = BTreeSet::from([start.clone()]);
        while let Some(family_id) = queue.pop_front() {
            for next in &adjacency[&family_id] {
                if seen.insert(next.clone()) {
                    members.insert(next.clone());
                    queue.push_back(next.clone());
                }
            }
        }

        let family_ids: Vec<String> = members.into_iter().collect();
        let split_names: Vec<String> = family_ids
            .iter()
            .map(|family_id| splits[family_id].clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let member_set: BTreeSet<&str> = family_ids.iter().map(String::as_str).collect();
        let related_hard_edges: Vec<&Edge> = edges
            .iter()
            .filter(|edge| {
                hard(&edge.relation_kind)
                    && edge.assessment_status != "dismissed"
                    && member_set.contains(edge.source_family_id.as_str())
                    && member_set.contains(edge.target_family_id.as_str())
            })
            .collect();
        let sorted = |mut ids: Vec<String>| {
            ids.sort();
            ids
        };
        let hard_edge_ids = sorted(
            related_hard_edges
                .iter()
                .map(|edge| edge.edge_id.clone())
                .collect(),
        );
        let confirmed_edge_ids = sorted(
            related_hard_edges
                .iter()
                .filter(|edge| edge.assessment_status == "confirmed")
                .map(|edge| edge.edge_id.clone())
                .collect(),
        );
        let unresolved_edge_ids = sorted(
            related_hard_edges
                .iter()
                .filter(|edge| edge.assessment_status == "unresolved")
                .map(|edge| edge.edge_id.clone())
                .collect(),
        );
        output.push(Component {
            component_id: component_id(&family_ids),
            split_names,
            family_ids,
            hard_edge_ids,
            confirmed_edge_ids,
            unresolved_edge_ids,
        });
    }
    output.sort_by(|left, right| left.family_ids.cmp(&right.family_ids));
    output
}

fn qualification_state(
    coverage_complete: bool,
    cross_split_components: usize,
    receipt_verified: bool,
) -> (&'static str, bool) {
    if !coverage_complete {
        ("not_qualified_incomplete_inventory", false)
    } else if cross_split_components > 0 {
        ("not_qualified_cross_split_hard_relationships", false)
    } else if !receipt_verified {
        ("structurally_eligible_receipt_unverified", false)
    } else {
        ("qualified_coverage_verified_no_cross_split_hard_relationships", true)
    }
}

fn build(
    doc: &Value,
    inventory: &Inventory,
    raw_inventory: &[u8],
    receipt_verified: bool,
) -> Result<Manifest> {
    let (digest, splits) = validate_splits(doc)?;
    let receipt = validate_edges(inventory, &digest, &splits)?;
    let components = components(&splits, &inventory.edges);
    let mut edge_counts = EdgeCounts {
        total_edges: inventory.edges.len(),
        hard_edges_included: 0,
        confirmed_hard_edges: 0,
        unresolved_hard_edges: 0,
        dismissed_hard_edges: 0,
        soft_edges: 0,
        cross_split_hard_edges: 0,
        unresolved_cross_split_hard_edges: 0,
    };
    for edge in &inventory.edges {
        if !hard(&edge.relation_kind) {
            edge_counts.soft_edges += 1;
            continue;
        }
        match edge.assessment_status.as_str() {
            "confirmed" => edge_counts.confirmed_hard_edges += 1,
            "unresolved" => edge_counts.unresolved_hard_edges += 1,
            "dismissed" => edge_counts.dismissed_hard_edges += 1,
            _ => unreachable!(),
        }
        if edge.assessment_status == "dismissed" {
            continue;
        }
        edge_counts.hard_edges_included += 1;
        if splits[&edge.source_family_id] != splits[&edge.target_family_id] {
            edge_counts.cross_split_hard_edges += 1;
            if edge.assessment_status == "unresolved" {
                edge_counts.unresolved_cross_split_hard_edges += 1;
            }
        }
    }

    let cross_split_component_count = components
        .iter()
        .filter(|component| component.split_names.len() > 1)
        .count();
    let (qualification_status, qualified_for_confirmatory_resampling) = qualification_state(
        inventory.coverage_status == "independently_reviewed_complete",
        cross_split_component_count,
        receipt_verified,
    );

    let mut counts = Counts {
        train: 0,
        validation: 0,
        test: 0,
    };
    let mut family_rows = Vec::new();
    for component in &components {
        if component.split_names.len() == 1 {
            match component.split_names[0].as_str() {
                "train" => counts.train += 1,
                "validation" => counts.validation += 1,
                "test" => counts.test += 1,
                _ => unreachable!(),
            }
        }
        for family_id in &component.family_ids {
            family_rows.push(FamilyRow {
                family_id: family_id.clone(),
                split: splits[family_id].clone(),
                component_id: component.component_id.clone(),
            });
        }
    }
    family_rows.sort_by(|left, right| left.family_id.cmp(&right.family_id));

    Ok(Manifest {
        schema_version: "pitb.independent-component-manifest.v0",
        upstream_commit: UPSTREAM.to_owned(),
        source_partition_digest_sha256: digest,
        relationship_inventory_sha256: sha(raw_inventory),
        relationship_inventory_status: inventory.coverage_status.clone(),
        coverage_receipt_sha256: receipt,
        total_family_count: splits.len(),
        total_component_count: components.len(),
        independent_component_count_by_split: counts,
        cross_split_component_count,
        edge_counts,
        families: family_rows,
        components,
        qualification_status: qualification_status.to_owned(),
        coverage_receipt_verification_status: if receipt_verified {
            "verified_signature_set"
        } else {
            "not_verified_receipt_bytes_not_supplied"
        },
        qualified_for_confirmatory_resampling,
        qualification_limitations: vec![
            if receipt_verified {
                concat!(
                    "Coverage receipt signatures were verified against caller-pinned methodology ",
                    "and trust-store digests; this attestation cannot mathematically ",
                    "prove reviewer competence or exhaustive discovery."
                )
                .into()
            } else {
                concat!(
                    "Receipt bytes, reviewer trust-root bytes, and review-methodology ",
                    "bytes were not supplied and verified; the manifest remains unqualified."
                )
                .into()
            },
            concat!(
                "Unresolved non-dismissed hard links are joined; any cross-split component ",
                "blocks qualification."
            )
            .into(),
            concat!(
                "Primary scorer and power planner must independently reverify the pinned receipt ",
                "inputs and bind this exact manifest digest before confirmatory qualification."
            )
            .into(),
        ],
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 8 {
        bail!(
            concat!(
                "usage: build_family_component_manifest <splits.json> <edges.json> ",
                "<receipt.json|none> <trust-store.json|none> <review-methodology.md|none> ",
                "<expected-trust-store-sha256|none> ",
                "<expected-methodology-sha256|none> <manifest.json>"
            )
        );
    }
    let split_bytes = fs::read(&args[0]).with_context(|| format!("cannot read {}", args[0]))?;
    let edge_bytes = fs::read(&args[1]).with_context(|| format!("cannot read {}", args[1]))?;
    let split_doc: Value = serde_json::from_slice(&split_bytes).context("invalid split JSON")?;
    let inventory: Inventory =
        serde_json::from_slice(&edge_bytes).context("invalid relationship JSON")?;
    let (partition_digest, families) = validate_splits(&split_doc)?;
    let independently_validated = coverage_receipt::validate_frozen_partition(&split_bytes)?;
    if independently_validated != families {
        bail!(
            concat!(
                "shared receipt verifier and component builder disagree on frozen family ",
                "assignments"
            )
        );
    }

    let all_none = args[2..7].iter().all(|value| value == "none");
    let all_supplied = args[2..7].iter().all(|value| value != "none");
    if !all_none && !all_supplied {
        bail!(
            concat!(
                "receipt, trust store, methodology, and both expected digests must be supplied ",
                "together or all be none"
            )
        );
    }
    let receipt_verified = if all_supplied {
        if inventory.coverage_status != "independently_reviewed_complete" {
            bail!("a receipt cannot qualify an inventory declared incomplete");
        }
        let receipt_bytes =
            fs::read(&args[2]).with_context(|| format!("cannot read {}", args[2]))?;
        let trust_store_bytes =
            fs::read(&args[3]).with_context(|| format!("cannot read {}", args[3]))?;
        let methodology_bytes =
            fs::read(&args[4]).with_context(|| format!("cannot read {}", args[4]))?;
        let report = verify_receipt(
            &edge_bytes,
            &receipt_bytes,
            &trust_store_bytes,
            &methodology_bytes,
            &args[5],
            &args[6],
            &partition_digest,
            &families,
        )?;
        if report.schema_version != "pitb.coverage-receipt-verification.v1"
            || report.verification_status != "verified_signature_set"
            || !report.cryptographically_verified
            || report.resampling_qualification != "requires_component_and_analyzer_validation"
            || report.upstream_commit != UPSTREAM
            || report.source_partition_digest_sha256 != partition_digest
            || !digest_ok(&report.relationship_inventory_edges_sha256)
            || report.relationship_inventory_sha256 != sha(&edge_bytes)
            || report.coverage_receipt_sha256 != sha(&receipt_bytes)
            || report.review_methodology_sha256 != sha(&methodology_bytes)
            || report.trust_store_sha256 != sha(&trust_store_bytes)
            || report.verified_reviewer_ids.len() < 2
            || report.distinct_reviewer_organization_count < 2
            || report.limitation.trim().is_empty()
        {
            bail!("coverage receipt verification report failed internal provenance reconciliation");
        }
        true
    } else {
        false
    };

    let manifest = build(&split_doc, &inventory, &edge_bytes, receipt_verified)?;
    fs::write(&args[7], serde_json::to_vec_pretty(&manifest)?)
        .with_context(|| format!("cannot write {}", args[7]))?;
    println!(
        concat!(
            "PITB_COMPONENT_MANIFEST: qualification={} families={} components={} ",
            "cross_split_components={} coverage={} receipt_verified={}"
        ),
        manifest.qualification_status,
        manifest.total_family_count,
        manifest.total_component_count,
        manifest.cross_split_component_count,
        manifest.relationship_inventory_status,
        receipt_verified
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualification_requires_complete_coverage_signed_receipt_and_no_cross_split_links() {
        assert_eq!(
            qualification_state(false, 0, false),
            ("not_qualified_incomplete_inventory", false)
        );
        assert_eq!(
            qualification_state(true, 1, true),
            ("not_qualified_cross_split_hard_relationships", false)
        );
        assert_eq!(
            qualification_state(true, 0, false),
            ("structurally_eligible_receipt_unverified", false)
        );
        assert_eq!(
            qualification_state(true, 0, true),
            (
                "qualified_coverage_verified_no_cross_split_hard_relationships",
                true
            )
        );
    }

    #[test]
    fn signed_synthetic_receipt_qualifies_only_the_clean_frozen_component_manifest() {
        let fixture = coverage_receipt::test_support::full_partition_fixture()
            .expect("full frozen partition fixture");
        let split_doc: Value =
            serde_json::from_slice(&fixture.split_bytes).expect("split JSON");
        let inventory: Inventory =
            serde_json::from_slice(&fixture.inventory_bytes).expect("inventory JSON");
        let (partition_digest, families) = validate_splits(&split_doc).expect("validated split");
        assert_eq!(partition_digest, fixture.partition_digest);
        let report = verify_receipt(
            &fixture.inventory_bytes,
            &fixture.receipt_bytes,
            &fixture.trust_store_bytes,
            &fixture.methodology_bytes,
            &fixture.trust_store_digest,
            &fixture.methodology_digest,
            &partition_digest,
            &families,
        )
        .expect("synthetic signatures verify");
        assert!(report.cryptographically_verified);

        let manifest = build(
            &split_doc,
            &inventory,
            &fixture.inventory_bytes,
            report.cryptographically_verified,
        )
        .expect("build qualified synthetic manifest");
        assert_eq!(
            manifest.qualification_status,
            "qualified_coverage_verified_no_cross_split_hard_relationships"
        );
        assert!(manifest.qualified_for_confirmatory_resampling);
        assert_eq!(
            manifest.coverage_receipt_verification_status,
            "verified_signature_set"
        );
        assert_eq!(
            manifest.coverage_receipt_sha256.as_deref(),
            Some(fixture.receipt_digest.as_str())
        );
        assert_eq!(manifest.relationship_inventory_sha256, fixture.inventory_digest);
        assert_eq!(manifest.total_family_count, 372);
        assert_eq!(manifest.total_component_count, 372);
        assert_eq!(manifest.cross_split_component_count, 0);
    }

    #[test]
    fn deterministic_split_assignment_is_stable_and_balanced() {
        let ids = vec![
            "a".to_owned(),
            "b".to_owned(),
            "c".to_owned(),
            "d".to_owned(),
        ];
        let first = deterministic(&ids).expect("first split");
        assert_eq!(first, deterministic(&ids).expect("repeat split"));
        let counts = (
            first.values().filter(|split| *split == "train").count(),
            first
                .values()
                .filter(|split| *split == "validation")
                .count(),
            first.values().filter(|split| *split == "test").count(),
        );
        assert_eq!(counts, (2, 1, 1));
    }

    fn edge(
        id: &str,
        source: &str,
        target: &str,
        relation: &str,
        status: &str,
    ) -> Edge {
        Edge {
            edge_id: id.into(),
            source_family_id: source.into(),
            target_family_id: target.into(),
            relation_kind: relation.into(),
            assessment_status: status.into(),
            evidence_sha256: "a".repeat(64),
            reviewer_ids: if status == "unresolved" {
                vec![]
            } else {
                vec!["r1".into(), "r2".into()]
            },
        }
    }

    fn split_map(rows: &[(&str, &str)]) -> BTreeMap<String, String> {
        rows.iter()
            .map(|(family_id, split)| ((*family_id).into(), (*split).into()))
            .collect()
    }

    #[test]
    fn includes_singletons() {
        let splits = split_map(&[
            ("001", "train"),
            ("002", "train"),
            ("003", "validation"),
            ("004", "test"),
        ]);
        let components = components(
            &splits,
            &[edge("e1", "001", "002", "formal_dependency", "confirmed")],
        );
        assert_eq!(components.len(), 3);
        assert_eq!(
            components
                .iter()
                .map(|component| component.family_ids.len())
                .sum::<usize>(),
            4
        );
        assert!(components
            .iter()
            .any(|component| component.family_ids == vec!["004".to_owned()]));
    }

    #[test]
    fn unresolved_hard_joins_soft_and_dismissed_do_not() {
        let splits = split_map(&[
            ("001", "train"),
            ("002", "train"),
            ("003", "train"),
            ("004", "train"),
        ]);
        let edges = vec![
            edge("e1", "001", "002", "reused_lemma", "unresolved"),
            edge("e2", "002", "003", "conceptual_similarity", "confirmed"),
            edge("e3", "003", "004", "formal_dependency", "dismissed"),
        ];
        let components = components(&splits, &edges);
        assert_eq!(components.len(), 3);
        assert_eq!(components[0].family_ids, vec!["001".to_owned(), "002".to_owned()]);
        assert_eq!(components[0].unresolved_edge_ids, vec!["e1".to_owned()]);
    }

    #[test]
    fn cross_split_link_is_one_component() {
        let splits = split_map(&[
            ("001", "train"),
            ("002", "test"),
            ("003", "train"),
        ]);
        let components = components(
            &splits,
            &[edge("e1", "001", "002", "formal_dependency", "confirmed")],
        );
        assert!(components.iter().any(|component| {
            component.family_ids == vec!["001".to_owned(), "002".to_owned()]
                && component.split_names.len() == 2
        }));
    }

    #[test]
    fn ids_are_content_addressed() {
        let first = vec!["001".to_owned(), "002".to_owned()];
        let same_members = first.clone();
        assert_eq!(component_id(&first), component_id(&same_members));
        assert_ne!(
            component_id(&first),
            component_id(&["001".to_owned(), "003".to_owned()])
        );
    }
}
