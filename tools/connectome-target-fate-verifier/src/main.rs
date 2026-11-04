#![forbid(unsafe_code)]

//! Fail-closed structural verifier for bidirectional target-inventory fate receipts.
//!
//! This tool recomputes exact input digests, target inventory closure, mapping
//! membership, ambiguity grouping and summary counters. It does not authenticate
//! referenced evidence bytes, edit-history authority, biological identity,
//! adjudicator identity, signer trust roots, or biological novelty.

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::process;

const VERSION: &str = "0.1.0";

#[derive(Debug, Clone)]
struct TargetObject {
    object_type: String,
}

#[derive(Debug, Clone)]
struct Mapping {
    relation_type: String,
    source_ids: Vec<String>,
    target_ids: BTreeSet<String>,
    support_status: String,
    alternative_set_id: Option<String>,
    evidence_ids: BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct CorrespondenceMappings {
    world_id: String,
    release_id: String,
    mappings: BTreeMap<String, Mapping>,
    alternative_groups: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Counts {
    mapped: u64,
    alternatives: u64,
    target_only: u64,
    out_of_scope: u64,
    rejected: u64,
    by_type: BTreeMap<String, u64>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn object<'a>(value: &'a Value, context: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{context}: expected object"))
}

fn array<'a>(value: &'a Value, context: &str) -> Result<&'a Vec<Value>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{context}: expected array"))
}

fn required_string(value: &Value, key: &str, context: &str) -> Result<String, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("{context}.{key}: expected non-empty string"))
}

fn optional_string(value: &Value, key: &str, context: &str) -> Result<Option<String>, String> {
    match value.get(key) {
        Some(Value::Null) => Ok(None),
        Some(Value::String(text)) if !text.trim().is_empty() => Ok(Some(text.clone())),
        _ => Err(format!(
            "{context}.{key}: expected non-empty string or null"
        )),
    }
}

fn required_digest(value: &Value, key: &str, context: &str) -> Result<String, String> {
    let digest = required_string(value, key, context)?;
    if !valid_sha256(&digest) {
        return Err(format!("{context}.{key}: expected lowercase SHA-256 hex"));
    }
    Ok(digest)
}

fn require_exact_keys(value: &Value, keys: &[&str], context: &str) -> Result<(), String> {
    let map = object(value, context)?;
    if map.len() != keys.len() || keys.iter().any(|key| !map.contains_key(*key)) {
        return Err(format!(
            "{context}: missing or unexpected keys; expected exactly {}",
            keys.join(", ")
        ));
    }
    Ok(())
}

fn unique_string_array(value: &Value, context: &str) -> Result<Vec<String>, String> {
    let values = array(value, context)?;
    let mut output = Vec::with_capacity(values.len());
    let mut seen = BTreeSet::new();
    for item in values {
        let text = item
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| format!("{context}: expected non-empty string items"))?
            .to_owned();
        if !seen.insert(text.clone()) {
            return Err(format!("{context}: duplicate item {text}"));
        }
        output.push(text);
    }
    Ok(output)
}

fn parse_target_inventory(
    bytes: &[u8],
) -> Result<(String, String, BTreeMap<String, TargetObject>), String> {
    let root: Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("target manifest JSON invalid: {error}"))?;
    if required_string(&root, "schema_version", "target_manifest")? != VERSION {
        return Err("unsupported target manifest schema version".to_owned());
    }
    let world = root
        .get("world")
        .ok_or_else(|| "target_manifest.world missing".to_owned())?;
    let world_id = required_string(world, "world_id", "target_manifest.world")?;
    let release_id = required_string(world, "release_id", "target_manifest.world")?;
    let object_values = array(
        world
            .get("objects")
            .ok_or_else(|| "target_manifest.world.objects missing".to_owned())?,
        "target_manifest.world.objects",
    )?;
    if object_values.is_empty() {
        return Err("target manifest inventory must not be empty".to_owned());
    }

    let mut inventory = BTreeMap::new();
    for (index, item) in object_values.iter().enumerate() {
        let context = format!("target_manifest.world.objects[{index}]");
        let id = required_string(item, "object_id", &context)?;
        let object_type = required_string(item, "object_type", &context)?;
        if !matches!(
            object_type.as_str(),
            "SEGMENT" | "NEURON" | "SYNAPSE" | "ARTIFACT"
        ) {
            return Err(format!("{context}.object_type invalid: {object_type}"));
        }
        if inventory
            .insert(id.clone(), TargetObject { object_type })
            .is_some()
        {
            return Err(format!("target manifest contains duplicate object_id {id}"));
        }
    }
    Ok((world_id, release_id, inventory))
}

fn parse_mappings(correspondence: &Value) -> Result<CorrespondenceMappings, String> {
    let world = correspondence
        .get("target_world")
        .ok_or_else(|| "correspondence.target_world missing".to_owned())?;
    let world_id = required_string(world, "world_id", "correspondence.target_world")?;
    let release_id = required_string(world, "release_id", "correspondence.target_world")?;
    let rows = array(
        correspondence
            .get("mappings")
            .ok_or_else(|| "correspondence.mappings missing".to_owned())?,
        "correspondence.mappings",
    )?;

    let mut mappings = BTreeMap::new();
    let mut target_owner = BTreeMap::<String, String>::new();
    let mut alternative_groups = BTreeMap::<String, Vec<String>>::new();

    for (index, row) in rows.iter().enumerate() {
        let context = format!("correspondence.mappings[{index}]");
        let id = required_string(row, "mapping_id", &context)?;
        let relation_type = required_string(row, "relation_type", &context)?;
        let source_ids = unique_string_array(
            row.get("source_object_ids")
                .ok_or_else(|| format!("{context}.source_object_ids missing"))?,
            &format!("{context}.source_object_ids"),
        )?;
        if source_ids.is_empty() {
            return Err(format!("{context}: source_object_ids must not be empty"));
        }
        let target_ids_vec = unique_string_array(
            row.get("target_object_ids")
                .ok_or_else(|| format!("{context}.target_object_ids missing"))?,
            &format!("{context}.target_object_ids"),
        )?;
        let target_ids: BTreeSet<String> = target_ids_vec.iter().cloned().collect();
        let support_status = required_string(row, "support_status", &context)?;
        if !matches!(
            support_status.as_str(),
            "CLAIMED_VERIFIED" | "AMBIGUOUS" | "NOT_ESTIMABLE" | "REJECTED"
        ) {
            return Err(format!("{context}.support_status invalid"));
        }
        let alternative_set_id = optional_string(row, "alternative_set_id", &context)?;

        let mut evidence_ids = BTreeSet::new();
        let evidence = array(
            row.get("evidence")
                .ok_or_else(|| format!("{context}.evidence missing"))?,
            &format!("{context}.evidence"),
        )?;
        for (evidence_index, evidence_item) in evidence.iter().enumerate() {
            let evidence_context = format!("{context}.evidence[{evidence_index}]");
            let evidence_id = required_string(evidence_item, "evidence_id", &evidence_context)?;
            if !evidence_ids.insert(evidence_id.clone()) {
                return Err(format!(
                    "{evidence_context}: duplicate evidence_id {evidence_id}"
                ));
            }
        }

        if support_status == "AMBIGUOUS" {
            let group = alternative_set_id.as_ref().ok_or_else(|| {
                format!("{context}: AMBIGUOUS mapping requires alternative_set_id")
            })?;
            alternative_groups
                .entry(group.clone())
                .or_default()
                .push(id.clone());
        } else if alternative_set_id.is_some() {
            return Err(format!(
                "{context}: alternative_set_id only valid for AMBIGUOUS mapping"
            ));
        }

        for target_id in &target_ids {
            if target_owner.insert(target_id.clone(), id.clone()).is_some() {
                return Err(format!(
                    "target object {target_id} is claimed by multiple correspondence rows"
                ));
            }
        }

        let mapping = Mapping {
            relation_type,
            source_ids,
            target_ids,
            support_status,
            alternative_set_id,
            evidence_ids,
        };
        if mappings.insert(id.clone(), mapping).is_some() {
            return Err(format!("duplicate mapping_id {id}"));
        }
    }
    Ok(CorrespondenceMappings {
        world_id,
        release_id,
        mappings,
        alternative_groups,
    })
}

fn verify(
    fate_bytes: &[u8],
    target_manifest_bytes: &[u8],
    correspondence_bytes: &[u8],
) -> Result<Value, String> {
    let fate: Value = serde_json::from_slice(fate_bytes)
        .map_err(|error| format!("fate receipt JSON invalid: {error}"))?;
    require_exact_keys(
        &fate,
        &[
            "schema_version",
            "fate_inventory_id",
            "target_world",
            "target_manifest_sha256",
            "correspondence_receipt_sha256",
            "fates",
            "summary",
            "provenance",
        ],
        "fate_receipt",
    )?;
    if required_string(&fate, "schema_version", "fate_receipt")? != VERSION {
        return Err("unsupported fate receipt schema version".to_owned());
    }
    let fate_inventory_id = required_string(&fate, "fate_inventory_id", "fate_receipt")?;

    let manifest_digest = sha256_hex(target_manifest_bytes);
    let correspondence_digest = sha256_hex(correspondence_bytes);
    if required_digest(&fate, "target_manifest_sha256", "fate_receipt")? != manifest_digest {
        return Err("target_manifest_sha256 does not match exact target manifest bytes".to_owned());
    }
    if required_digest(&fate, "correspondence_receipt_sha256", "fate_receipt")?
        != correspondence_digest
    {
        return Err(
            "correspondence_receipt_sha256 does not match exact correspondence receipt bytes"
                .to_owned(),
        );
    }

    let provenance = fate
        .get("provenance")
        .ok_or_else(|| "fate_receipt.provenance missing".to_owned())?;
    require_exact_keys(
        provenance,
        &["target_manifest_sha256", "correspondence_receipt_sha256"],
        "fate_receipt.provenance",
    )?;
    if required_digest(provenance, "target_manifest_sha256", "provenance")? != manifest_digest
        || required_digest(provenance, "correspondence_receipt_sha256", "provenance")?
            != correspondence_digest
    {
        return Err("provenance digests do not match exact input bytes".to_owned());
    }

    let (manifest_world_id, manifest_release_id, inventory) =
        parse_target_inventory(target_manifest_bytes)?;
    let fate_world = fate
        .get("target_world")
        .ok_or_else(|| "fate_receipt.target_world missing".to_owned())?;
    require_exact_keys(
        fate_world,
        &["world_id", "release_id"],
        "fate_receipt.target_world",
    )?;
    let fate_world_id = required_string(fate_world, "world_id", "fate_receipt.target_world")?;
    let fate_release_id = required_string(fate_world, "release_id", "fate_receipt.target_world")?;

    let correspondence: Value = serde_json::from_slice(correspondence_bytes)
        .map_err(|error| format!("correspondence receipt JSON invalid: {error}"))?;
    if required_string(&correspondence, "schema_version", "correspondence")? != VERSION {
        return Err("unsupported correspondence schema version".to_owned());
    }
    let correspondence_mappings = parse_mappings(&correspondence)?;

    // Cross-check every mapping target against the exact target inventory, including
    // mapping rows that have no fate record due to a malformed or incomplete receipt.
    for (mapping_id, mapping) in &correspondence_mappings.mappings {
        let expected_type = match mapping.relation_type.as_str() {
            "ARTIFACT_IDENTITY" => "ARTIFACT",
            "SEGMENT_LINEAGE" => "SEGMENT",
            "NEURON_CORRESPONDENCE" => "NEURON",
            "SYNAPSE_CORRESPONDENCE" => "SYNAPSE",
            other => {
                return Err(format!(
                    "mapping {mapping_id}: unsupported relation_type {other}"
                ));
            }
        };
        for target_id in &mapping.target_ids {
            let target = inventory.get(target_id).ok_or_else(|| {
                format!(
                    "mapping {mapping_id}: target object {target_id} is absent from target manifest"
                )
            })?;
            if target.object_type != expected_type {
                return Err(format!(
                    "mapping {mapping_id}: target object {target_id} is {}, expected {expected_type}",
                    target.object_type
                ));
            }
        }
    }

    for (name, value) in [
        ("manifest", &manifest_world_id),
        ("correspondence", &correspondence_mappings.world_id),
    ] {
        if value != &fate_world_id {
            return Err(format!("{name} world_id disagrees with fate receipt"));
        }
    }
    if manifest_release_id != fate_release_id || correspondence_mappings.release_id != fate_release_id {
        return Err(
            "target release_id disagrees across fate receipt, manifest and correspondence"
                .to_owned(),
        );
    }

    for (group_id, mapping_ids) in &correspondence_mappings.alternative_groups {
        if mapping_ids.len() < 2 {
            return Err(format!(
                "alternative set {group_id} must contain at least two mapping rows"
            ));
        }
        let first = correspondence_mappings.mappings
            .get(&mapping_ids[0])
            .ok_or_else(|| format!("alternative set {group_id}: missing mapping row"))?;
        let mut seen_targets = BTreeSet::new();
        for mapping_id in mapping_ids {
            let mapping = correspondence_mappings.mappings.get(mapping_id).ok_or_else(|| {
                format!("alternative set {group_id}: missing mapping {mapping_id}")
            })?;
            if mapping.relation_type != first.relation_type
                || mapping.source_ids != first.source_ids
            {
                return Err(format!(
                    "alternative set {group_id} mixes relation type or source ID sets"
                ));
            }
            for target_id in &mapping.target_ids {
                if !seen_targets.insert(target_id.clone()) {
                    return Err(format!(
                        "alternative set {group_id} overlaps target object {target_id}"
                    ));
                }
            }
        }
    }

    let fate_rows = array(
        fate.get("fates")
            .ok_or_else(|| "fate_receipt.fates missing".to_owned())?,
        "fate_receipt.fates",
    )?;
    if fate_rows.is_empty() {
        return Err("fate inventory must not be empty".to_owned());
    }

    let mut seen_fates = BTreeSet::new();
    let mut counts = Counts::default();

    for (index, row) in fate_rows.iter().enumerate() {
        let context = format!("fates[{index}]");
        require_exact_keys(
            row,
            &[
                "target_object_id",
                "object_type",
                "status",
                "mapping_id",
                "alternative_set_id",
                "scope_policy_sha256",
                "reason_code",
                "evidence_ids",
            ],
            &context,
        )?;
        let object_id = required_string(row, "target_object_id", &context)?;
        let object_type = required_string(row, "object_type", &context)?;
        let status = required_string(row, "status", &context)?;
        let mapping_id = optional_string(row, "mapping_id", &context)?;
        let alternative_set_id = optional_string(row, "alternative_set_id", &context)?;
        let scope_digest = optional_string(row, "scope_policy_sha256", &context)?;
        let reason_code = optional_string(row, "reason_code", &context)?;
        let evidence_ids_vec = unique_string_array(
            row.get("evidence_ids")
                .ok_or_else(|| format!("{context}.evidence_ids missing"))?,
            &format!("{context}.evidence_ids"),
        )?;
        if evidence_ids_vec.is_empty() {
            return Err(format!("{context}.evidence_ids must not be empty"));
        }

        if !seen_fates.insert(object_id.clone()) {
            return Err(format!("duplicate target fate for object {object_id}"));
        }
        let manifest_object = inventory.get(&object_id).ok_or_else(|| {
            format!("{context}: target object {object_id} absent from target manifest")
        })?;
        if manifest_object.object_type != object_type {
            return Err(format!(
                "{context}: object_type disagrees with target manifest"
            ));
        }
        if !matches!(
            object_type.as_str(),
            "SEGMENT" | "NEURON" | "SYNAPSE" | "ARTIFACT"
        ) {
            return Err(format!("{context}: unsupported object_type"));
        }
        *counts.by_type.entry(object_type.clone()).or_default() += 1;
        let evidence_ids: BTreeSet<String> = evidence_ids_vec.into_iter().collect();

        match status.as_str() {
            "MAPPED" => {
                if alternative_set_id.is_some() || scope_digest.is_some() || reason_code.is_some() {
                    return Err(format!(
                        "{context}: MAPPED has incompatible alternative/scope/reason fields"
                    ));
                }
                let mapping_id =
                    mapping_id.ok_or_else(|| format!("{context}: MAPPED requires mapping_id"))?;
                let mapping = mappings
                    .get(&mapping_id)
                    .ok_or_else(|| format!("{context}: unknown mapping_id {mapping_id}"))?;
                if mapping.support_status == "AMBIGUOUS" || mapping.support_status == "REJECTED" {
                    return Err(format!(
                        "{context}: MAPPED cannot point to AMBIGUOUS or REJECTED mapping"
                    ));
                }
                if !mapping.target_ids.contains(&object_id) {
                    return Err(format!(
                        "{context}: mapping {mapping_id} does not contain target object {object_id}"
                    ));
                }
                if !evidence_ids.is_subset(&mapping.evidence_ids) {
                    return Err(format!(
                        "{context}: fate evidence IDs are not present in referenced mapping"
                    ));
                }
                counts.mapped += 1;
            }
            "ALTERNATIVE_CANDIDATE" => {
                if scope_digest.is_some() || reason_code.is_some() {
                    return Err(format!(
                        "{context}: ALTERNATIVE_CANDIDATE has incompatible scope/reason fields"
                    ));
                }
                let mapping_id = mapping_id.ok_or_else(|| {
                    format!("{context}: ALTERNATIVE_CANDIDATE requires mapping_id")
                })?;
                let group_id = alternative_set_id.ok_or_else(|| {
                    format!("{context}: ALTERNATIVE_CANDIDATE requires alternative_set_id")
                })?;
                let mapping = mappings
                    .get(&mapping_id)
                    .ok_or_else(|| format!("{context}: unknown mapping_id {mapping_id}"))?;
                if mapping.support_status != "AMBIGUOUS"
                    || mapping.alternative_set_id.as_deref() != Some(group_id.as_str())
                {
                    return Err(format!(
                        "{context}: mapping/alternative-set claim disagrees with correspondence receipt"
                    ));
                }
                if !mapping.target_ids.contains(&object_id) {
                    return Err(format!(
                        "{context}: candidate mapping does not contain target object {object_id}"
                    ));
                }
                if !evidence_ids.is_subset(&mapping.evidence_ids) {
                    return Err(format!(
                        "{context}: fate evidence IDs are not present in candidate mapping"
                    ));
                }
                counts.alternatives += 1;
            }
            "TARGET_ONLY_UNRESOLVED" => {
                if mapping_id.is_some()
                    || alternative_set_id.is_some()
                    || scope_digest.is_some()
                    || reason_code.is_none()
                {
                    return Err(format!(
                        "{context}: TARGET_ONLY_UNRESOLVED requires only a reason and no mapping/scope fields"
                    ));
                }
                if mappings
                    .values()
                    .any(|mapping| mapping.target_ids.contains(&object_id))
                {
                    return Err(format!(
                        "{context}: TARGET_ONLY_UNRESOLVED object appears in a mapping row"
                    ));
                }
                counts.target_only += 1;
            }
            "OUT_OF_SCOPE" => {
                if mapping_id.is_some() || alternative_set_id.is_some() || reason_code.is_none() {
                    return Err(format!(
                        "{context}: OUT_OF_SCOPE requires policy digest and reason, with no mapping"
                    ));
                }
                let digest = scope_digest.ok_or_else(|| {
                    format!("{context}: OUT_OF_SCOPE requires scope_policy_sha256")
                })?;
                if !valid_sha256(&digest) {
                    return Err(format!("{context}: invalid scope policy digest"));
                }
                if mappings
                    .values()
                    .any(|mapping| mapping.target_ids.contains(&object_id))
                {
                    return Err(format!(
                        "{context}: OUT_OF_SCOPE object appears in a mapping row"
                    ));
                }
                counts.out_of_scope += 1;
            }
            "REJECTED" => {
                if alternative_set_id.is_some() || scope_digest.is_some() || reason_code.is_none() {
                    return Err(format!(
                        "{context}: REJECTED requires a reason and no alternative/scope fields"
                    ));
                }
                let mapping_id =
                    mapping_id.ok_or_else(|| format!("{context}: REJECTED requires mapping_id"))?;
                let mapping = mappings
                    .get(&mapping_id)
                    .ok_or_else(|| format!("{context}: unknown mapping_id {mapping_id}"))?;
                if mapping.support_status != "REJECTED" || !mapping.target_ids.contains(&object_id)
                {
                    return Err(format!(
                        "{context}: rejected mapping does not match target object or support status"
                    ));
                }
                if !evidence_ids.is_subset(&mapping.evidence_ids) {
                    return Err(format!(
                        "{context}: fate evidence IDs are not present in rejected mapping"
                    ));
                }
                counts.rejected += 1;
            }
            _ => return Err(format!("{context}: invalid target fate status {status}")),
        }
    }

    let inventory_ids: BTreeSet<String> = inventory.keys().cloned().collect();
    let fate_ids: BTreeSet<String> = seen_fates;
    if inventory_ids != fate_ids {
        let missing: Vec<String> = inventory_ids.difference(&fate_ids).cloned().collect();
        let extra: Vec<String> = fate_ids.difference(&inventory_ids).cloned().collect();
        return Err(format!(
            "target fate inventory mismatch; missing=[{}], extra=[{}]",
            missing.join(","),
            extra.join(",")
        ));
    }

    let summary = fate
        .get("summary")
        .ok_or_else(|| "fate_receipt.summary missing".to_owned())?;
    require_exact_keys(
        summary,
        &[
            "target_object_count",
            "mapped_object_count",
            "alternative_candidate_object_count",
            "target_only_unresolved_object_count",
            "out_of_scope_object_count",
            "rejected_object_count",
            "object_type_counts",
        ],
        "summary",
    )?;
    let expected_status = [
        ("target_object_count", fate_rows.len() as u64),
        ("mapped_object_count", counts.mapped),
        ("alternative_candidate_object_count", counts.alternatives),
        ("target_only_unresolved_object_count", counts.target_only),
        ("out_of_scope_object_count", counts.out_of_scope),
        ("rejected_object_count", counts.rejected),
    ];
    for (key, expected) in expected_status {
        let actual = summary
            .get(key)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("summary.{key} must be a non-negative integer"))?;
        if actual != expected {
            return Err(format!(
                "summary.{key} reports {actual}, recomputed {expected}"
            ));
        }
    }
    let type_counts = summary
        .get("object_type_counts")
        .ok_or_else(|| "summary.object_type_counts missing".to_owned())?;
    require_exact_keys(
        type_counts,
        &["SEGMENT", "NEURON", "SYNAPSE", "ARTIFACT"],
        "summary.object_type_counts",
    )?;
    for object_type in ["SEGMENT", "NEURON", "SYNAPSE", "ARTIFACT"] {
        let expected = counts.by_type.get(object_type).copied().unwrap_or(0);
        let actual = type_counts
            .get(object_type)
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                format!("summary.object_type_counts.{object_type} must be a non-negative integer")
            })?;
        if actual != expected {
            return Err(format!(
                "summary.object_type_counts.{object_type} reports {actual}, recomputed {expected}"
            ));
        }
    }

    Ok(json!({
        "schema_version": VERSION,
        "fate_inventory_id": fate_inventory_id,
        "status": "COUNTERS_RECOMPUTED",
        "trusted_promotion": false,
        "target_manifest_sha256": manifest_digest,
        "correspondence_receipt_sha256": correspondence_digest,
        "recomputed_summary": {
            "target_object_count": fate_rows.len(),
            "mapped_object_count": counts.mapped,
            "alternative_candidate_object_count": counts.alternatives,
            "target_only_unresolved_object_count": counts.target_only,
            "out_of_scope_object_count": counts.out_of_scope,
            "rejected_object_count": counts.rejected,
            "object_type_counts": {
                "SEGMENT": counts.by_type.get("SEGMENT").copied().unwrap_or(0),
                "NEURON": counts.by_type.get("NEURON").copied().unwrap_or(0),
                "SYNAPSE": counts.by_type.get("SYNAPSE").copied().unwrap_or(0),
                "ARTIFACT": counts.by_type.get("ARTIFACT").copied().unwrap_or(0)
            }
        },
        "trust_boundaries": {
            "evidence_artifact_authentication": "NOT_PERFORMED",
            "edit_history_authentication": "NOT_PERFORMED",
            "biological_identity": "NOT_ESTABLISHED",
            "biological_novelty": "NOT_ESTABLISHED",
            "signer_identity_and_revocation": "NOT_AUTHENTICATED"
        }
    }))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "usage: {} <fate-inventory.json> <target-world-manifest.json> <correspondence-receipt.json>",
            args.first()
                .map(String::as_str)
                .unwrap_or("connectome-target-fate-verifier")
        );
        process::exit(64);
    }
    let fate = fs::read(&args[1]).unwrap_or_else(|error| {
        eprintln!("cannot read fate inventory: {error}");
        process::exit(66);
    });
    let manifest = fs::read(&args[2]).unwrap_or_else(|error| {
        eprintln!("cannot read target manifest: {error}");
        process::exit(66);
    });
    let correspondence = fs::read(&args[3]).unwrap_or_else(|error| {
        eprintln!("cannot read correspondence receipt: {error}");
        process::exit(66);
    });

    match verify(&fate, &manifest, &correspondence) {
        Ok(report) => println!(
            "{}",
            serde_json::to_string(&report).expect("serialize report")
        ),
        Err(error) => {
            eprintln!("REJECTED: {error}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FATE: &str = include_str!(
        "../../../research/connectomics/fixtures/connectome-target-fate-inventory-known-answer-v0.1.json"
    );
    const TARGET: &str = include_str!(
        "../../../research/connectomics/fixtures/connectome-target-fate-target-world-v0.1.json"
    );
    const CORRESPONDENCE: &str = include_str!(
        "../../../research/connectomics/fixtures/connectome-target-fate-correspondence-v0.1.json"
    );

    fn value(text: &str) -> Value {
        serde_json::from_str(text).expect("fixture JSON")
    }

    fn bound_inputs(mut fate: Value, correspondence: Value) -> (String, String) {
        let corr_text =
            serde_json::to_string_pretty(&correspondence).expect("serialize correspondence") + "\n";
        let digest = sha256_hex(corr_text.as_bytes());
        fate["correspondence_receipt_sha256"] = Value::String(digest.clone());
        fate["provenance"]["correspondence_receipt_sha256"] = Value::String(digest);
        (
            serde_json::to_string_pretty(&fate).expect("serialize fate") + "\n",
            corr_text,
        )
    }

    #[test]
    fn accepts_known_answer_fixture_and_recomputes_fates() {
        let report = verify(
            FATE.as_bytes(),
            TARGET.as_bytes(),
            CORRESPONDENCE.as_bytes(),
        )
        .expect("valid fixture");
        let counts = &report["recomputed_summary"];
        assert_eq!(counts["target_object_count"], 5);
        assert_eq!(counts["mapped_object_count"], 1);
        assert_eq!(counts["alternative_candidate_object_count"], 2);
        assert_eq!(counts["target_only_unresolved_object_count"], 1);
        assert_eq!(counts["out_of_scope_object_count"], 1);
        assert_eq!(report["trusted_promotion"], false);
    }

    #[test]
    fn rejects_manifest_raw_byte_mutation() {
        let mut manifest = TARGET.to_owned();
        manifest.push(' ');
        assert!(
            verify(
                FATE.as_bytes(),
                manifest.as_bytes(),
                CORRESPONDENCE.as_bytes()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_correspondence_raw_byte_mutation() {
        let mut correspondence = CORRESPONDENCE.to_owned();
        correspondence.push(' ');
        assert!(
            verify(
                FATE.as_bytes(),
                TARGET.as_bytes(),
                correspondence.as_bytes()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_omitted_target_fate() {
        let mut fate = value(FATE);
        fate["fates"].as_array_mut().expect("fates").remove(0);
        assert!(
            verify(
                serde_json::to_string(&fate).unwrap().as_bytes(),
                TARGET.as_bytes(),
                CORRESPONDENCE.as_bytes()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_duplicate_target_fate() {
        let mut fate = value(FATE);
        let first = fate["fates"][0].clone();
        fate["fates"].as_array_mut().expect("fates").push(first);
        assert!(
            verify(
                serde_json::to_string(&fate).unwrap().as_bytes(),
                TARGET.as_bytes(),
                CORRESPONDENCE.as_bytes()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_unknown_target_object() {
        let mut fate = value(FATE);
        fate["fates"][0]["target_object_id"] = Value::String("unknown-target".to_owned());
        assert!(
            verify(
                serde_json::to_string(&fate).unwrap().as_bytes(),
                TARGET.as_bytes(),
                CORRESPONDENCE.as_bytes()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_object_type_mismatch() {
        let mut fate = value(FATE);
        fate["fates"][0]["object_type"] = Value::String("NEURON".to_owned());
        assert!(
            verify(
                serde_json::to_string(&fate).unwrap().as_bytes(),
                TARGET.as_bytes(),
                CORRESPONDENCE.as_bytes()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_unknown_mapping_id() {
        let mut fate = value(FATE);
        fate["fates"][0]["mapping_id"] = Value::String("missing-mapping".to_owned());
        assert!(
            verify(
                serde_json::to_string(&fate).unwrap().as_bytes(),
                TARGET.as_bytes(),
                CORRESPONDENCE.as_bytes()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_mapping_that_does_not_contain_target() {
        let fate = value(FATE);
        let mut correspondence = value(CORRESPONDENCE);
        correspondence["mappings"][0]["target_object_ids"] = json!(["segment-not-in-fate"]);
        let (fate_text, corr_text) = bound_inputs(fate, correspondence);
        assert!(
            verify(
                fate_text.as_bytes(),
                TARGET.as_bytes(),
                corr_text.as_bytes()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_alternative_set_mismatch() {
        let mut fate = value(FATE);
        fate["fates"][1]["alternative_set_id"] = Value::String("wrong-group".to_owned());
        assert!(
            verify(
                serde_json::to_string(&fate).unwrap().as_bytes(),
                TARGET.as_bytes(),
                CORRESPONDENCE.as_bytes()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_target_only_object_that_appears_in_mapping() {
        let fate = value(FATE);
        let mut correspondence = value(CORRESPONDENCE);
        correspondence["mappings"][0]["target_object_ids"] =
            json!(["segment-1", "synapse-target-only"]);
        let (fate_text, corr_text) = bound_inputs(fate, correspondence);
        assert!(
            verify(
                fate_text.as_bytes(),
                TARGET.as_bytes(),
                corr_text.as_bytes()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_mapping_marked_ambiguous_as_mapped() {
        let mut fate = value(FATE);
        fate["fates"][0]["mapping_id"] = Value::String("map-neuron-5-a".to_owned());
        fate["fates"][0]["target_object_id"] = Value::String("neuron-5-candidate-a".to_owned());
        fate["fates"][0]["object_type"] = Value::String("NEURON".to_owned());
        fate["fates"][0]["evidence_ids"] = json!(["ev-neuron-morphology-a"]);
        assert!(
            verify(
                serde_json::to_string(&fate).unwrap().as_bytes(),
                TARGET.as_bytes(),
                CORRESPONDENCE.as_bytes()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_single_row_alternative_group() {
        let fate = value(FATE);
        let mut correspondence = value(CORRESPONDENCE);
        correspondence["mappings"]
            .as_array_mut()
            .expect("mappings")
            .pop();
        let (fate_text, corr_text) = bound_inputs(fate, correspondence);
        assert!(
            verify(
                fate_text.as_bytes(),
                TARGET.as_bytes(),
                corr_text.as_bytes()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_overlapping_alternative_targets() {
        let fate = value(FATE);
        let mut correspondence = value(CORRESPONDENCE);
        correspondence["mappings"][2]["target_object_ids"] = json!(["neuron-5-candidate-a"]);
        let (fate_text, corr_text) = bound_inputs(fate, correspondence);
        assert!(
            verify(
                fate_text.as_bytes(),
                TARGET.as_bytes(),
                corr_text.as_bytes()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_out_of_scope_without_policy_digest() {
        let mut fate = value(FATE);
        fate["fates"][4]["scope_policy_sha256"] = Value::Null;
        assert!(
            verify(
                serde_json::to_string(&fate).unwrap().as_bytes(),
                TARGET.as_bytes(),
                CORRESPONDENCE.as_bytes()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_rejected_fate_pointing_to_non_rejected_mapping() {
        let mut fate = value(FATE);
        fate["fates"][0]["status"] = Value::String("REJECTED".to_owned());
        fate["fates"][0]["reason_code"] = Value::String("TEST_REJECTION".to_owned());
        assert!(
            verify(
                serde_json::to_string(&fate).unwrap().as_bytes(),
                TARGET.as_bytes(),
                CORRESPONDENCE.as_bytes()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_summary_counter_corruption() {
        let mut fate = value(FATE);
        fate["summary"]["mapped_object_count"] = Value::from(999);
        assert!(
            verify(
                serde_json::to_string(&fate).unwrap().as_bytes(),
                TARGET.as_bytes(),
                CORRESPONDENCE.as_bytes()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_world_or_release_mismatch() {
        let mut fate = value(FATE);
        fate["target_world"]["release_id"] = Value::String("another-release".to_owned());
        assert!(
            verify(
                serde_json::to_string(&fate).unwrap().as_bytes(),
                TARGET.as_bytes(),
                CORRESPONDENCE.as_bytes()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_duplicate_mapping_target_claim() {
        let fate = value(FATE);
        let mut correspondence = value(CORRESPONDENCE);
        correspondence["mappings"][2]["target_object_ids"] = json!(["segment-1"]);
        let (fate_text, corr_text) = bound_inputs(fate, correspondence);
        assert!(
            verify(
                fate_text.as_bytes(),
                TARGET.as_bytes(),
                corr_text.as_bytes()
            )
            .is_err()
        );
    }
}
