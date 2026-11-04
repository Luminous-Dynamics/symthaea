#![forbid(unsafe_code)]

//! Semantic verifier for typed multiplex connectome evidence claims.
//!
//! It verifies relation/evidence type consistency, release scoping, query path
//! grammar and non-laundering rules. It deliberately does not authenticate
//! external evidence bytes, signer identity, statistical model assumptions,
//! biological truth or causal identification.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::process;

const VERSION: &str = "0.1.0";

#[derive(Debug, Clone)]
struct ClaimInfo {
    relation_type: String,
    claim_status: String,
    source_id: String,
    target_id: String,
    source_world: String,
    target_world: String,
    source_kind: String,
    target_kind: String,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct ClaimCounts {
    anatomical: u64,
    correspondence: u64,
    statistical: u64,
    functional: u64,
    causal: u64,
}

#[derive(Debug)]
struct ClaimIndex {
    claims: BTreeMap<String, ClaimInfo>,
    counts: ClaimCounts,
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

fn array<'a>(value: &'a Value, context: &str) -> Result<&'a Vec<Value>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{context}: expected array"))
}

fn string_field(value: &Value, key: &str, context: &str) -> Result<String, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("{context}.{key}: expected non-empty string"))
}

fn digest_field(value: &Value, key: &str, context: &str) -> Result<String, String> {
    let digest = string_field(value, key, context)?;
    if !valid_sha256(&digest) {
        return Err(format!("{context}.{key}: expected lowercase SHA-256 hex"));
    }
    Ok(digest)
}

fn number_field(value: &Value, key: &str, context: &str) -> Result<f64, String> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
        .ok_or_else(|| format!("{context}.{key}: expected finite number"))
}

fn expected_evidence_type(relation_type: &str) -> Result<&'static str, String> {
    match relation_type {
        "ANATOMICAL_SYNAPSE_EDGE" => Ok("ANATOMICAL_SYNAPSE_EVIDENCE"),
        "RECONSTRUCTION_CORRESPONDENCE" => Ok("RECONSTRUCTION_CORRESPONDENCE_EVIDENCE"),
        "STATISTICAL_DEPENDENCE_EDGE" => Ok("STATISTICAL_DEPENDENCE_EVIDENCE"),
        "FUNCTIONAL_ASSOCIATION_EDGE" => Ok("FUNCTIONAL_ASSOCIATION_EVIDENCE"),
        "CAUSAL_EFFECT_CLAIM" => Ok("CAUSAL_EFFECT_EVIDENCE"),
        other => Err(format!("unsupported relation_type {other}")),
    }
}

fn expected_path_claim_type(relation_type: &str) -> Result<&'static str, String> {
    match relation_type {
        "ANATOMICAL_SYNAPSE_EDGE" => Ok("ANATOMICAL_PATH"),
        "RECONSTRUCTION_CORRESPONDENCE" => Ok("CORRESPONDENCE_PATH"),
        "STATISTICAL_DEPENDENCE_EDGE" => Ok("STATISTICAL_PATH"),
        "FUNCTIONAL_ASSOCIATION_EDGE" => Ok("FUNCTIONAL_PATH"),
        "CAUSAL_EFFECT_CLAIM" => Ok("CAUSAL_EFFECT"),
        other => Err(format!("unsupported relation_type {other}")),
    }
}

fn valid_claim_status(status: &str) -> bool {
    matches!(
        status,
        "CLAIMED_SUPPORTED"
            | "CLAIMED_CONDITIONAL"
            | "CLAIMED_AMBIGUOUS"
            | "CLAIMED_NOT_ESTIMABLE"
            | "CLAIMED_REJECTED"
    )
}

fn parse_claims(bundle: &Value) -> Result<ClaimIndex, String> {
    let claim_rows = array(
        bundle
            .get("claims")
            .ok_or_else(|| "claims missing".to_owned())?,
        "claims",
    )?;
    if claim_rows.is_empty() {
        return Err("claims must not be empty".to_owned());
    }

    let mut claims = BTreeMap::new();
    let mut counts = ClaimCounts::default();
    for (index, claim) in claim_rows.iter().enumerate() {
        let context = format!("claims[{index}]");
        let claim_id = string_field(claim, "claim_id", &context)?;
        let relation_type = string_field(claim, "relation_type", &context)?;
        let claim_status = string_field(claim, "claim_status", &context)?;
        if !valid_claim_status(&claim_status) {
            return Err(format!("{context}.claim_status invalid"));
        }
        let source = claim
            .get("source")
            .ok_or_else(|| format!("{context}.source missing"))?;
        let target = claim
            .get("target")
            .ok_or_else(|| format!("{context}.target missing"))?;
        let source_id = string_field(source, "object_id", &format!("{context}.source"))?;
        let target_id = string_field(target, "object_id", &format!("{context}.target"))?;
        let source_kind = string_field(source, "object_kind", &format!("{context}.source"))?;
        let target_kind = string_field(target, "object_kind", &format!("{context}.target"))?;
        let source_world = string_field(source, "world_id", &format!("{context}.source"))?;
        let target_world = string_field(target, "world_id", &format!("{context}.target"))?;
        let expected = expected_evidence_type(&relation_type)?;
        let evidence = array(
            claim
                .get("evidence")
                .ok_or_else(|| format!("{context}.evidence missing"))?,
            &format!("{context}.evidence"),
        )?;
        if evidence.is_empty() {
            return Err(format!("{context}: evidence must not be empty"));
        }

        for (evidence_index, item) in evidence.iter().enumerate() {
            let item_context = format!("{context}.evidence[{evidence_index}]");
            let actual = string_field(item, "evidence_type", &item_context)?;
            if actual != expected {
                return Err(format!(
                    "{item_context}: evidence_type {actual} cannot support {relation_type}"
                ));
            }
            match relation_type.as_str() {
                "ANATOMICAL_SYNAPSE_EDGE" => {
                    if !matches!(source_kind.as_str(), "NEURON" | "COMPARTMENT")
                        || !matches!(target_kind.as_str(), "NEURON" | "COMPARTMENT")
                    {
                        return Err(format!(
                            "{context}: anatomical edge endpoints must be neuron/compartment objects"
                        ));
                    }
                    if source_world != target_world {
                        return Err(format!(
                            "{context}: anatomical edge endpoints must be in the same world"
                        ));
                    }
                    string_field(item, "synapse_candidate_id", &item_context)?;
                    digest_field(item, "artifact_sha256", &item_context)?;
                    digest_field(item, "image_manifest_sha256", &item_context)?;
                    string_field(item, "coordinate_frame_id", &item_context)?;
                    let presence = string_field(item, "presence", &item_context)?;
                    let polarity = string_field(item, "polarity", &item_context)?;
                    let pre_partner = string_field(item, "pre_partner", &item_context)?;
                    let post_partner = string_field(item, "post_partner", &item_context)?;
                    let all_supported = presence == "CLAIMED_PRESENT"
                        && polarity == "CLAIMED_PRE_TO_POST"
                        && pre_partner == "CLAIMED_MATCH"
                        && post_partner == "CLAIMED_MATCH";
                    if claim_status == "CLAIMED_SUPPORTED" && !all_supported {
                        return Err(format!(
                            "{context}: supported anatomical edge lacks fully supported presence/polarity/partner claims"
                        ));
                    }
                }
                "RECONSTRUCTION_CORRESPONDENCE" => {
                    if source_world == target_world {
                        return Err(format!(
                            "{context}: reconstruction correspondence must cross distinct worlds/releases"
                        ));
                    }
                    let evidence_source = string_field(item, "source_world_id", &item_context)?;
                    let evidence_target = string_field(item, "target_world_id", &item_context)?;
                    if evidence_source != source_world || evidence_target != target_world {
                        return Err(format!(
                            "{context}: correspondence evidence world IDs disagree with endpoints"
                        ));
                    }
                    digest_field(item, "mapping_receipt_sha256", &item_context)?;
                    string_field(item, "mapping_id", &item_context)?;
                    let correspondence_relation =
                        string_field(item, "correspondence_relation", &item_context)?;
                    let valid_types = match correspondence_relation.as_str() {
                        "ARTIFACT_IDENTITY" => {
                            source_kind == "DATASET_RELEASE" && target_kind == "DATASET_RELEASE"
                        }
                        "SEGMENT_LINEAGE" => source_kind == "SEGMENT" && target_kind == "SEGMENT",
                        "NEURON_CORRESPONDENCE" => {
                            source_kind == "NEURON" && target_kind == "NEURON"
                        }
                        "SYNAPSE_CORRESPONDENCE" => {
                            source_kind == "SYNAPSE" && target_kind == "SYNAPSE"
                        }
                        _ => false,
                    };
                    if !valid_types {
                        return Err(format!(
                            "{context}: correspondence relation does not match endpoint types"
                        ));
                    }
                }
                "STATISTICAL_DEPENDENCE_EDGE" => {
                    digest_field(item, "model_sha256", &item_context)?;
                    digest_field(item, "sample_frame_sha256", &item_context)?;
                    digest_field(item, "conditioning_set_sha256", &item_context)?;
                    digest_field(item, "multiplicity_procedure_sha256", &item_context)?;
                    digest_field(item, "sampling_design_sha256", &item_context)?;
                    string_field(item, "hypothesis_id", &item_context)?;
                    string_field(item, "hypothesis_family_id", &item_context)?;
                    let p = number_field(item, "p_value", &item_context)?;
                    let q = number_field(item, "q_value", &item_context)?;
                    let se = number_field(item, "standard_error", &item_context)?;
                    let n = number_field(item, "effective_sample_size", &item_context)?;
                    if !(0.0..=1.0).contains(&p)
                        || !(0.0..=1.0).contains(&q)
                        || se < 0.0
                        || n <= 0.0
                    {
                        return Err(format!(
                            "{context}: statistical evidence values are outside admissible ranges"
                        ));
                    }
                }
                "FUNCTIONAL_ASSOCIATION_EDGE" => {
                    digest_field(item, "recording_manifest_sha256", &item_context)?;
                    digest_field(item, "acquisition_protocol_sha256", &item_context)?;
                    digest_field(item, "signal_pipeline_sha256", &item_context)?;
                    digest_field(item, "state_context_sha256", &item_context)?;
                    let start = number_field(item, "window_start_ms", &item_context)?;
                    let end = number_field(item, "window_end_ms", &item_context)?;
                    if end <= start {
                        return Err(format!(
                            "{context}: functional time window must be increasing"
                        ));
                    }
                }
                "CAUSAL_EFFECT_CLAIM" => {
                    for field in [
                        "intervention_protocol_sha256",
                        "assignment_mechanism_sha256",
                        "target_population_sha256",
                        "outcome_definition_sha256",
                        "identification_strategy_sha256",
                        "assumption_bundle_sha256",
                    ] {
                        digest_field(item, field, &item_context)?;
                    }
                    let positivity = string_field(item, "positivity_status", &item_context)?;
                    let selection = string_field(item, "selection_status", &item_context)?;
                    if claim_status == "CLAIMED_SUPPORTED"
                        && (positivity != "CLAIMED_ADEQUATE" || selection != "CLAIMED_HANDLED")
                    {
                        return Err(format!(
                            "{context}: supported causal claim has inadequate positivity or unresolved selection"
                        ));
                    }
                }
                _ => unreachable!(),
            }
        }

        match relation_type.as_str() {
            "ANATOMICAL_SYNAPSE_EDGE" => counts.anatomical += 1,
            "RECONSTRUCTION_CORRESPONDENCE" => counts.correspondence += 1,
            "STATISTICAL_DEPENDENCE_EDGE" => counts.statistical += 1,
            "FUNCTIONAL_ASSOCIATION_EDGE" => counts.functional += 1,
            "CAUSAL_EFFECT_CLAIM" => counts.causal += 1,
            _ => unreachable!(),
        }
        let info = ClaimInfo {
            relation_type,
            claim_status,
            source_id,
            target_id,
            source_world,
            target_world,
            source_kind,
            target_kind,
        };
        if claims.insert(claim_id.clone(), info).is_some() {
            return Err(format!("duplicate claim_id {claim_id}"));
        }
    }
    Ok(ClaimIndex { claims, counts })
}

fn validate_queries(bundle: &Value, index: &ClaimIndex) -> Result<Vec<Value>, String> {
    let query_rows = array(
        bundle
            .get("queries")
            .ok_or_else(|| "queries missing".to_owned())?,
        "queries",
    )?;
    if query_rows.is_empty() {
        return Err("queries must not be empty".to_owned());
    }
    let mut query_ids = BTreeSet::new();
    let mut outputs = Vec::new();

    for (query_index, query) in query_rows.iter().enumerate() {
        let context = format!("queries[{query_index}]");
        let query_id = string_field(query, "query_id", &context)?;
        if !query_ids.insert(query_id.clone()) {
            return Err(format!("duplicate query_id {query_id}"));
        }
        digest_field(query, "query_definition_sha256", &context)?;
        digest_field(query, "analysis_code_sha256", &context)?;
        let query_mode = string_field(query, "query_mode", &context)?;
        if !matches!(
            query_mode.as_str(),
            "SINGLE_RELATION_LAYER" | "DECLARED_MIXED_LAYERS"
        ) {
            return Err(format!("{context}.query_mode invalid"));
        }
        let requested_type = string_field(query, "requested_claim_type", &context)?;
        if !matches!(
            requested_type.as_str(),
            "ANATOMICAL_PATH"
                | "CORRESPONDENCE_PATH"
                | "STATISTICAL_PATH"
                | "FUNCTIONAL_PATH"
                | "CAUSAL_EFFECT"
                | "MIXED_EVIDENCE_PATH"
        ) {
            return Err(format!("{context}.requested_claim_type invalid"));
        }
        let result_status = string_field(query, "result_status", &context)?;
        if !matches!(
            result_status.as_str(),
            "CLAIMED_SUPPORTED"
                | "CLAIMED_CONDITIONAL"
                | "CLAIMED_NOT_ESTIMABLE"
                | "CLAIMED_REJECTED"
        ) {
            return Err(format!("{context}.result_status invalid"));
        }
        if result_status == "CLAIMED_NOT_ESTIMABLE" {
            if query.get("result_sha256") != Some(&Value::Null) {
                return Err(format!(
                    "{context}: CLAIMED_NOT_ESTIMABLE query must not bind a result digest"
                ));
            }
        } else {
            digest_field(query, "result_sha256", &context)?;
        }
        let allowed_rows = array(
            query
                .get("allowed_relation_type_paths")
                .ok_or_else(|| format!("{context}.allowed_relation_type_paths missing"))?,
            &format!("{context}.allowed_relation_type_paths"),
        )?;
        let mut allowed = BTreeMap::<String, Vec<String>>::new();
        for (path_index, path) in allowed_rows.iter().enumerate() {
            let path_context = format!("{context}.allowed_relation_type_paths[{path_index}]");
            let path_id = string_field(path, "path_id", &path_context)?;
            let relation_types = array(
                path.get("relation_types")
                    .ok_or_else(|| format!("{path_context}.relation_types missing"))?,
                &format!("{path_context}.relation_types"),
            )?;
            let mut types = Vec::new();
            for relation in relation_types {
                let relation = relation
                    .as_str()
                    .ok_or_else(|| format!("{path_context}: relation types must be strings"))?;
                expected_evidence_type(relation)?;
                types.push(relation.to_owned());
            }
            if types.is_empty() {
                return Err(format!("{path_context}: empty relation type sequence"));
            }
            if allowed.insert(path_id.clone(), types).is_some() {
                return Err(format!("{context}: duplicate path_id {path_id}"));
            }
        }

        let path_results = array(
            query
                .get("path_results")
                .ok_or_else(|| format!("{context}.path_results missing"))?,
            &format!("{context}.path_results"),
        )?;
        if path_results.is_empty() && result_status != "CLAIMED_NOT_ESTIMABLE" {
            return Err(format!(
                "{context}: a query without path results must be explicitly NOT_ESTIMABLE"
            ));
        }
        let mut seen_path_results = BTreeSet::new();
        for (result_index, result) in path_results.iter().enumerate() {
            let result_context = format!("{context}.path_results[{result_index}]");
            let path_id = string_field(result, "path_id", &result_context)?;
            if !seen_path_results.insert(path_id.clone()) {
                return Err(format!("{result_context}: duplicate path result {path_id}"));
            }
            let allowed_sequence = allowed.get(&path_id).ok_or_else(|| {
                format!("{result_context}: path_id is not declared by query grammar")
            })?;
            let claim_ids = array(
                result
                    .get("ordered_claim_ids")
                    .ok_or_else(|| format!("{result_context}.ordered_claim_ids missing"))?,
                &format!("{result_context}.ordered_claim_ids"),
            )?;
            if claim_ids.is_empty() {
                return Err(format!("{result_context}: ordered_claim_ids must not be empty"));
            }
            let mut used = Vec::new();
            let mut used_claim_ids = Vec::new();
            let mut seen_claims = BTreeSet::new();
            let mut relation_set = BTreeSet::new();
            let mut all_claims_supported = true;
            let mut previous_target: Option<(String, String, String)> = None;
            let mut path_is_contiguous = true;
            for claim_id_value in claim_ids {
                let claim_id = claim_id_value
                    .as_str()
                    .ok_or_else(|| format!("{result_context}: claim IDs must be strings"))?;
                if !seen_claims.insert(claim_id.to_owned()) {
                    return Err(format!("{result_context}: duplicate claim ID {claim_id}"));
                }
                let claim = index
                    .claims
                    .get(claim_id)
                    .ok_or_else(|| format!("{result_context}: unknown claim ID {claim_id}"))?;
                used.push(claim.relation_type.clone());
                used_claim_ids.push(claim_id.to_owned());
                relation_set.insert(claim.relation_type.clone());
                if claim.claim_status != "CLAIMED_SUPPORTED" {
                    all_claims_supported = false;
                }
                if let Some((previous_id, previous_world, previous_kind)) = previous_target.as_ref()
                    && (previous_id != &claim.source_id
                        || previous_world != &claim.source_world
                        || previous_kind != &claim.source_kind)
                {
                    path_is_contiguous = false;
                }
                previous_target = Some((
                    claim.target_id.clone(),
                    claim.target_world.clone(),
                    claim.target_kind.clone(),
                ));
            }
            let asserted = string_field(result, "asserted_claim_type", &result_context)?;
            if !matches!(
                asserted.as_str(),
                "ANATOMICAL_PATH"
                    | "CORRESPONDENCE_PATH"
                    | "STATISTICAL_PATH"
                    | "FUNCTIONAL_PATH"
                    | "CAUSAL_EFFECT"
                    | "MIXED_EVIDENCE_PATH"
            ) {
                return Err(format!("{result_context}.asserted_claim_type invalid"));
            }
            let claimed_result_status = string_field(result, "result_status", &result_context)?;
            if !matches!(
                claimed_result_status.as_str(),
                "CLAIMED_SUPPORTED"
                    | "CLAIMED_CONDITIONAL"
                    | "CLAIMED_NOT_ESTIMABLE"
                    | "CLAIMED_REJECTED"
            ) {
                return Err(format!("{result_context}.result_status invalid"));
            }
            let mut disposition = "ACCEPTED_WITHIN_TYPED_SCOPE";
            let mut reasons = Vec::<String>::new();

            if used != *allowed_sequence {
                disposition = "REJECTED_PATH_GRAMMAR";
                reasons.push(
                    "ordered claim relation types do not equal the frozen query path grammar"
                        .to_owned(),
                );
            }
            if query_mode == "SINGLE_RELATION_LAYER" && relation_set.len() != 1 {
                disposition = "REJECTED_PATH_GRAMMAR";
                reasons.push("single-layer query contains more than one relation type".to_owned());
            }
            if query_mode == "SINGLE_RELATION_LAYER" && requested_type == "MIXED_EVIDENCE_PATH" {
                disposition = "REJECTED_PATH_GRAMMAR";
                reasons.push("single-layer query cannot request MIXED_EVIDENCE_PATH".to_owned());
            }
            if relation_set.len() > 1 {
                if asserted != "MIXED_EVIDENCE_PATH" {
                    disposition = "REJECTED_TYPE_PROMOTION";
                    reasons.push("mixed relation types cannot be promoted to a single relation claim".to_owned());
                }
                if asserted == "MIXED_EVIDENCE_PATH"
                    && (query_mode != "DECLARED_MIXED_LAYERS"
                        || requested_type != "MIXED_EVIDENCE_PATH")
                {
                    disposition = "REJECTED_TYPE_PROMOTION";
                    reasons.push("MIXED_EVIDENCE_PATH requires declared mixed-layer mode and mixed requested type".to_owned());
                }
            }
            if asserted == "CAUSAL_EFFECT" && relation_set != BTreeSet::from(["CAUSAL_EFFECT_CLAIM".to_owned()]) {
                disposition = "REJECTED_TYPE_PROMOTION";
                reasons.push(
                    "causal conclusions require a standalone CAUSAL_EFFECT_CLAIM path; other evidence layers cannot promote causality".to_owned(),
                );
            }
            if relation_set.len() == 1 {
                let only_type = relation_set.iter().next().expect("one relation type");
                let expected_claim_type = expected_path_claim_type(only_type)?;
                if asserted != expected_claim_type {
                    disposition = "REJECTED_TYPE_PROMOTION";
                    reasons.push(format!(
                        "{only_type} cannot support asserted claim type {asserted}"
                    ));
                }
            }
            if requested_type != asserted {
                disposition = "REJECTED_TYPE_PROMOTION";
                reasons.push(format!("asserted path type {asserted} differs from requested query type {requested_type}"));
            }
            if disposition == "ACCEPTED_WITHIN_TYPED_SCOPE" {
                match claimed_result_status.as_str() {
                    "CLAIMED_REJECTED" => {
                        disposition = "REJECTED_EVIDENCE_CLAIM";
                        reasons.push("the producer marks the path result rejected".to_owned());
                    }
                    "CLAIMED_NOT_ESTIMABLE" => {
                        disposition = "NOT_ESTIMABLE";
                        reasons.push("the producer marks the path result not estimable".to_owned());
                    }
                    "CLAIMED_CONDITIONAL" => {
                        disposition = "CONDITIONAL_ONLY";
                        reasons.push("the producer marks the path result conditional".to_owned());
                    }
                    _ => {}
                }
            }
            if disposition == "ACCEPTED_WITHIN_TYPED_SCOPE" {
                if used_claim_ids.iter().any(|claim_id| {
                    index
                        .claims
                        .get(claim_id)
                        .is_some_and(|claim| claim.claim_status == "CLAIMED_REJECTED")
                }) {
                    disposition = "REJECTED_EVIDENCE_CLAIM";
                    reasons.push("path contains a producer-rejected claim".to_owned());
                } else if used_claim_ids.iter().any(|claim_id| {
                    index
                        .claims
                        .get(claim_id)
                        .is_some_and(|claim| claim.claim_status == "CLAIMED_NOT_ESTIMABLE")
                }) {
                    disposition = "NOT_ESTIMABLE";
                    reasons.push("path contains a NOT_ESTIMABLE claim".to_owned());
                } else if used_claim_ids.iter().any(|claim_id| {
                    index
                        .claims
                        .get(claim_id)
                        .is_some_and(|claim| claim.claim_status == "CLAIMED_AMBIGUOUS")
                }) {
                    disposition = "AMBIGUOUS_NOT_PROMOTABLE";
                    reasons.push("path contains an ambiguous claim".to_owned());
                } else if !all_claims_supported || claimed_result_status == "CLAIMED_CONDITIONAL" {
                    disposition = "CONDITIONAL_ONLY";
                    reasons.push("one or more producer claims are conditional".to_owned());
                }
            }
            if claimed_result_status == "CLAIMED_NOT_ESTIMABLE"
                && disposition == "ACCEPTED_WITHIN_TYPED_SCOPE"
            {
                disposition = "NOT_ESTIMABLE";
            }
            if !path_is_contiguous {
                disposition = "REJECTED_PATH_CONTINUITY";
                reasons.push(
                    "adjacent path claims do not share the same target/source object, kind and world"
                        .to_owned(),
                );
            }
            outputs.push(json!({
                "query_id": query_id,
                "path_id": path_id,
                "claimed_path_type": asserted,
                "computed_relation_types": used,
                "claimed_result_status": claimed_result_status,
                "recomputed_disposition": disposition,
                "reasons": reasons
            }));
        }
    }
    Ok(outputs)
}

fn verify_bundle(bytes: &[u8]) -> Result<Value, String> {
    let bundle: Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("claim bundle JSON invalid: {error}"))?;
    if string_field(&bundle, "schema_version", "bundle")? != VERSION {
        return Err("unsupported claim bundle schema version".to_owned());
    }
    let bundle_id = string_field(&bundle, "bundle_id", "bundle")?;
    let scope = bundle
        .get("scope")
        .ok_or_else(|| "scope missing".to_owned())?;
    let dataset_id = string_field(scope, "dataset_id", "scope")?;
    let source_volume_id = string_field(scope, "source_volume_id", "scope")?;
    let release_id = string_field(scope, "release_id", "scope")?;
    let source_manifest_digest = digest_field(scope, "source_manifest_sha256", "scope")?;

    let index = parse_claims(&bundle)?;
    let query_results = validate_queries(&bundle, &index)?;

    let summary = bundle
        .get("summary")
        .ok_or_else(|| "summary missing".to_owned())?;
    let expected_claims = index.claims.len() as u64;
    let expected_queries = array(
        bundle
            .get("queries")
            .ok_or_else(|| "queries missing".to_owned())?,
        "queries",
    )?
    .len() as u64;
    if summary.get("claim_count").and_then(Value::as_u64) != Some(expected_claims) {
        return Err("summary.claim_count does not match recomputed claim count".to_owned());
    }
    if summary.get("query_count").and_then(Value::as_u64) != Some(expected_queries) {
        return Err("summary.query_count does not match recomputed query count".to_owned());
    }
    let type_counts = summary
        .get("claim_type_counts")
        .ok_or_else(|| "summary.claim_type_counts missing".to_owned())?;
    for (field, expected) in [
        ("ANATOMICAL_SYNAPSE_EDGE", index.counts.anatomical),
        ("RECONSTRUCTION_CORRESPONDENCE", index.counts.correspondence),
        ("STATISTICAL_DEPENDENCE_EDGE", index.counts.statistical),
        ("FUNCTIONAL_ASSOCIATION_EDGE", index.counts.functional),
        ("CAUSAL_EFFECT_CLAIM", index.counts.causal),
    ] {
        if type_counts.get(field).and_then(Value::as_u64) != Some(expected) {
            return Err(format!("summary.claim_type_counts.{field} mismatch"));
        }
    }

    let provenance = bundle
        .get("provenance")
        .ok_or_else(|| "provenance missing".to_owned())?;
    for field in [
        "claim_policy_sha256",
        "query_grammar_sha256",
        "producer_sha256",
    ] {
        digest_field(provenance, field, "provenance")?;
    }
    let rejected = query_results
        .iter()
        .filter(|row| {
            row["recomputed_disposition"] == "REJECTED_TYPE_PROMOTION"
                || row["recomputed_disposition"] == "REJECTED_PATH_GRAMMAR"
                || row["recomputed_disposition"] == "REJECTED_EVIDENCE_CLAIM"
        })
        .count();
    Ok(json!({
        "schema_version": VERSION,
        "bundle_id": bundle_id,
        "status": "STRUCTURE_RECOMPUTED",
        "source_bundle_sha256": sha256_hex(bytes),
        "scope": {
            "dataset_id": dataset_id,
            "source_volume_id": source_volume_id,
            "release_id": release_id,
            "source_manifest_sha256": source_manifest_digest
        },
        "recomputed_counts": {
            "claim_count": expected_claims,
            "query_count": expected_queries,
            "anatomical_claim_count": index.counts.anatomical,
            "correspondence_claim_count": index.counts.correspondence,
            "statistical_claim_count": index.counts.statistical,
            "functional_claim_count": index.counts.functional,
            "causal_claim_count": index.counts.causal
        },
        "query_path_results": query_results,
        "rejected_path_count": rejected,
        "trusted_promotion": false,
        "trust_boundaries": {
            "external_evidence_bytes": "NOT_AUTHENTICATED",
            "signer_authority": "NOT_AUTHENTICATED",
            "biological_identity": "NOT_ESTABLISHED",
            "statistical_model_validity": "NOT_AUTHENTICATED",
            "causal_identification": "NOT_AUTHENTICATED",
            "result_output_bytes": "DIGEST_CLAIM_ONLY"
        }
    }))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!(
            "usage: {} <typed-claim-bundle.json>",
            args.first()
                .map(String::as_str)
                .unwrap_or("connectome-typed-claim-verifier")
        );
        process::exit(64);
    }
    let bytes = fs::read(&args[1]).unwrap_or_else(|error| {
        eprintln!("cannot read claim bundle: {error}");
        process::exit(66);
    });
    match verify_bundle(&bytes) {
        Ok(report) => println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("serialize report")
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

    const FIXTURE: &str = include_str!(
        "../../../research/connectomics/fixtures/connectome-typed-claim-bundle-known-answer-v0.1.json"
    );

    fn parse_fixture() -> Value {
        serde_json::from_str(FIXTURE).expect("fixture JSON")
    }

    fn verify(value: &Value) -> Result<Value, String> {
        verify_bundle(serde_json::to_string_pretty(value).unwrap().as_bytes())
    }

    #[test]
    fn recomputes_claim_and_query_counts() {
        let report = verify_bundle(FIXTURE.as_bytes()).expect("known-answer bundle");
        assert_eq!(report["recomputed_counts"]["claim_count"], 5);
        assert_eq!(report["recomputed_counts"]["query_count"], 3);
        assert_eq!(report["trusted_promotion"], false);
    }

    #[test]
    fn rejects_statistical_evidence_under_anatomical_relation() {
        let mut bundle = parse_fixture();
        bundle["claims"][0]["evidence"] = bundle["claims"][2]["evidence"].clone();
        assert!(verify(&bundle).is_err());
    }

    #[test]
    fn rejects_anatomical_evidence_under_causal_relation() {
        let mut bundle = parse_fixture();
        bundle["claims"][4]["evidence"] = bundle["claims"][0]["evidence"].clone();
        assert!(verify(&bundle).is_err());
    }

    #[test]
    fn rejects_anatomical_edge_across_worlds() {
        let mut bundle = parse_fixture();
        bundle["claims"][0]["target"]["world_id"] = Value::String("other-world".to_owned());
        assert!(verify(&bundle).is_err());
    }

    #[test]
    fn rejects_supported_anatomical_edge_with_ambiguous_polarity() {
        let mut bundle = parse_fixture();
        bundle["claims"][0]["evidence"][0]["polarity"] = Value::String("AMBIGUOUS".to_owned());
        assert!(verify(&bundle).is_err());
    }

    #[test]
    fn rejects_correspondence_with_same_source_and_target_world() {
        let mut bundle = parse_fixture();
        bundle["claims"][1]["target"]["world_id"] = Value::String("world-release-1".to_owned());
        bundle["claims"][1]["evidence"][0]["target_world_id"] =
            Value::String("world-release-1".to_owned());
        assert!(verify(&bundle).is_err());
    }

    #[test]
    fn rejects_correspondence_evidence_world_mismatch() {
        let mut bundle = parse_fixture();
        bundle["claims"][1]["evidence"][0]["source_world_id"] =
            Value::String("wrong-world".to_owned());
        assert!(verify(&bundle).is_err());
    }

    #[test]
    fn rejects_statistical_q_value_out_of_range() {
        let mut bundle = parse_fixture();
        bundle["claims"][2]["evidence"][0]["q_value"] = Value::from(1.5);
        assert!(verify(&bundle).is_err());
    }

    #[test]
    fn rejects_functional_time_window_with_nonpositive_duration() {
        let mut bundle = parse_fixture();
        bundle["claims"][3]["evidence"][0]["time_window_end_ms"] = Value::from(0);
        assert!(verify(&bundle).is_err());
    }

    #[test]
    fn rejects_supported_causal_claim_with_positivity_violation() {
        let mut bundle = parse_fixture();
        bundle["claims"][4]["claim_status"] = Value::String("CLAIMED_SUPPORTED".to_owned());
        assert!(verify(&bundle).is_err());
    }

    #[test]
    fn preserves_invalid_causal_promotion_as_rejected_path() {
        let report = verify_bundle(FIXTURE.as_bytes()).expect("bundle structure is valid");
        let attempted = report["query_path_results"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["query_id"] == "invalid-anatomy-to-causality-attempt")
            .unwrap();
        assert_eq!(
            attempted["recomputed_disposition"],
            "REJECTED_TYPE_PROMOTION"
        );
    }

    #[test]
    fn accepts_single_layer_anatomical_path_within_typed_scope() {
        let report = verify_bundle(FIXTURE.as_bytes()).expect("bundle");
        let path = &report["query_path_results"][0];
        assert_eq!(
            path["recomputed_disposition"],
            "ACCEPTED_WITHIN_TYPED_SCOPE"
        );
    }

    #[test]
    fn rejects_query_path_with_unknown_claim_id() {
        let mut bundle = parse_fixture();
        bundle["queries"][0]["path_results"][0]["ordered_claim_ids"] = json!(["missing-claim"]);
        assert!(verify(&bundle).is_err());
    }

    #[test]
    fn rejects_declared_relation_order_mismatch() {
        let mut bundle = parse_fixture();
        bundle["queries"][0]["allowed_relation_type_paths"][0]["relation_types"] =
            json!(["STATISTICAL_DEPENDENCE_EDGE"]);
        assert!(
            verify(&bundle).is_err() || {
                let report = verify(&bundle).unwrap();
                report["query_path_results"][0]["recomputed_disposition"] == "REJECTED_PATH_GRAMMAR"
            }
        );
    }

    #[test]
    fn rejects_single_layer_query_that_mixes_relation_types() {
        let mut bundle = parse_fixture();
        bundle["queries"][0]["allowed_relation_type_paths"][0]["relation_types"] =
            json!(["ANATOMICAL_SYNAPSE_EDGE", "STATISTICAL_DEPENDENCE_EDGE"]);
        bundle["queries"][0]["path_results"][0]["ordered_claim_ids"] =
            json!(["anat-syn-17", "stat-edge-activity"]);
        assert!(
            verify(&bundle).unwrap()["query_path_results"][0]["recomputed_disposition"]
                == "REJECTED_PATH_GRAMMAR"
        );
    }

    #[test]
    fn rejects_duplicate_claim_id() {
        let mut bundle = parse_fixture();
        bundle["claims"][1]["claim_id"] = Value::String("anat-syn-17".to_owned());
        assert!(verify(&bundle).is_err());
    }

    #[test]
    fn rejects_duplicate_query_id() {
        let mut bundle = parse_fixture();
        bundle["queries"][1]["query_id"] = Value::String("anatomical-path-query".to_owned());
        assert!(verify(&bundle).is_err());
    }

    #[test]
    fn rejects_summary_count_corruption() {
        let mut bundle = parse_fixture();
        bundle["summary"]["claim_count"] = Value::from(999);
        assert!(verify(&bundle).is_err());
    }

    #[test]
    fn rejects_supported_result_without_digest() {
        let mut bundle = parse_fixture();
        bundle["queries"][0]["result_sha256"] = Value::Null;
        assert!(verify(&bundle).is_err());
    }

    #[test]
    fn rejects_disconnected_multi_claim_path() {
        let mut bundle = parse_fixture();
        bundle["queries"][0]["allowed_relation_type_paths"][0]["relation_types"] =
            json!(["ANATOMICAL_SYNAPSE_EDGE", "STATISTICAL_DEPENDENCE_EDGE"]);
        bundle["queries"][0]["path_results"][0]["ordered_claim_ids"] =
            json!(["anat-syn-17", "stat-edge-activity"]);
        bundle["queries"][0]["path_results"][0]["asserted_claim_type"] =
            Value::String("MIXED_EVIDENCE_PATH".to_owned());
        bundle["queries"][0]["requested_claim_type"] =
            Value::String("MIXED_EVIDENCE_PATH".to_owned());
        bundle["queries"][0]["query_mode"] =
            Value::String("DECLARED_MIXED_LAYERS".to_owned());
        let report = verify(&bundle).expect("disconnected path is preserved for audit");
        assert_eq!(
            report["query_path_results"][0]["recomputed_disposition"],
            "REJECTED_PATH_CONTINUITY"
        );
    }

    #[test]
    fn accepts_standalone_supported_causal_claim_path() {
        let mut bundle = parse_fixture();
        bundle["claims"][4]["claim_status"] = Value::String("CLAIMED_SUPPORTED".to_owned());
        bundle["claims"][4]["evidence"][0]["positivity_status"] =
            Value::String("CLAIMED_ADEQUATE".to_owned());
        bundle["claims"][4]["evidence"][0]["selection_status"] =
            Value::String("CLAIMED_HANDLED".to_owned());
        bundle["queries"][2]["query_mode"] = Value::String("SINGLE_RELATION_LAYER".to_owned());
        bundle["queries"][2]["allowed_relation_type_paths"][0]["relation_types"] =
            json!(["CAUSAL_EFFECT_CLAIM"]);
        bundle["queries"][2]["path_results"][0]["ordered_claim_ids"] =
            json!(["causal-edge-unresolved"]);
        bundle["queries"][2]["path_results"][0]["asserted_claim_type"] =
            Value::String("CAUSAL_EFFECT".to_owned());
        let report = verify(&bundle).expect("standalone causal-effect claim");
        assert_eq!(
            report["query_path_results"][2]["recomputed_disposition"],
            "ACCEPTED_WITHIN_TYPED_SCOPE"
        );
    }
    #[test]
    fn rejects_mixed_path_requested_in_single_layer_mode() {
        let mut bundle = parse_fixture();
        bundle["queries"][0]["requested_claim_type"] =
            Value::String("MIXED_EVIDENCE_PATH".to_owned());
        bundle["queries"][0]["path_results"][0]["asserted_claim_type"] =
            Value::String("MIXED_EVIDENCE_PATH".to_owned());
        let report = verify(&bundle).expect("bundle structurally valid");
        assert_eq!(
            report["query_path_results"][0]["recomputed_disposition"],
            "REJECTED_PATH_GRAMMAR"
        );
    }

    #[test]
    fn rejects_invalid_path_result_status() {
        let mut bundle = parse_fixture();
        bundle["queries"][0]["path_results"][0]["result_status"] =
            Value::String("TRUSTED_VERIFIED".to_owned());
        assert!(verify(&bundle).is_err());
    }

    #[test]
    fn reports_conditional_claim_as_conditional_not_not_estimable() {
        let mut bundle = parse_fixture();
        bundle["claims"][2]["claim_status"] = Value::String("CLAIMED_CONDITIONAL".to_owned());
        bundle["queries"][1]["path_results"][0]["result_status"] =
            Value::String("CLAIMED_CONDITIONAL".to_owned());
        let report = verify(&bundle).expect("conditional statistical claim");
        assert_eq!(
            report["query_path_results"][1]["recomputed_disposition"],
            "CONDITIONAL_ONLY"
        );
    }

    #[test]
    fn recomputes_rejected_claim_as_rejected_evidence_not_conditional() {
        let mut bundle = parse_fixture();
        bundle["claims"][0]["claim_status"] = Value::String("CLAIMED_REJECTED".to_owned());
        let report = verify(&bundle).expect("well-shaped bundle");
        assert_eq!(
            report["query_path_results"][0]["recomputed_disposition"],
            "REJECTED_EVIDENCE_CLAIM"
        );
    }

    #[test]
    fn recomputes_ambiguous_claim_as_ambiguous_not_promotable() {
        let mut bundle = parse_fixture();
        bundle["claims"][0]["claim_status"] = Value::String("CLAIMED_AMBIGUOUS".to_owned());
        let report = verify(&bundle).expect("well-shaped bundle");
        assert_eq!(
            report["query_path_results"][0]["recomputed_disposition"],
            "AMBIGUOUS_NOT_PROMOTABLE"
        );
    }

    #[test]
    fn recomputes_not_estimable_claim_without_promoting_the_path() {
        let mut bundle = parse_fixture();
        bundle["claims"][0]["claim_status"] = Value::String("CLAIMED_NOT_ESTIMABLE".to_owned());
        let report = verify(&bundle).expect("well-shaped bundle");
        assert_eq!(
            report["query_path_results"][0]["recomputed_disposition"],
            "NOT_ESTIMABLE"
        );
    }

    #[test]
    fn rejects_empty_claim_id_path() {
        let mut bundle = parse_fixture();
        bundle["queries"][0]["path_results"][0]["ordered_claim_ids"] = json!([]);
        assert!(verify(&bundle).is_err());
    }

    #[test]
    fn rejects_unknown_relation_type() {
        let mut bundle = parse_fixture();
        bundle["claims"][0]["relation_type"] = Value::String("GENERIC_EDGE".to_owned());
        assert!(verify(&bundle).is_err());
    }
}
