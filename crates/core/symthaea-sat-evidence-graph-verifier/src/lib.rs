// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Independent, read-only semantic verification for SAT evidence-graph augmentations.
//!
//! This crate verifies graph structure, dependency semantics, evidence-reference
//! closure, conservative supported-claim admissibility and the domain-separated
//! graph-root commitment. It does not validate every JSON Schema field, verify
//! signatures, authenticate sources, prove witness independence, or qualify a
//! system. Those are separate layers and must remain separate.

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

const GRAPH_DOMAIN: &str = "SAT-EVIDENCE-GRAPH-v0.1\n";
const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const GRAPH_SCHEMA_VERSION: &str = "sat-benchmark-augmentation/v0.1";
const MAX_AUGMENTATION_BYTES: usize = 32 * 1024 * 1024;
const MAX_OBSERVATIONS: usize = 100_000;
const MAX_CLAIMS: usize = 100_000;
const MAX_CLAIM_EDGES: usize = 250_000;

#[derive(Debug, Clone, PartialEq)]
enum StrictJson {
    Null,
    Bool(bool),
    Integer(i64),
    String(String),
    Array(Vec<StrictJson>),
    Object(Vec<(String, StrictJson)>),
}

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = StrictJson;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SAT restricted JSON with unique object keys and safe integers")
    }

    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(StrictJson::Null)
    }

    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(StrictJson::Null)
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
        Ok(StrictJson::Bool(value))
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
        if value.unsigned_abs() > MAX_SAFE_INTEGER {
            return Err(E::custom("integer exceeds the I-JSON safe integer range"));
        }
        Ok(StrictJson::Integer(value))
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
        if value > MAX_SAFE_INTEGER {
            return Err(E::custom("integer exceeds the I-JSON safe integer range"));
        }
        Ok(StrictJson::Integer(value as i64))
    }

    fn visit_f64<E: de::Error>(self, _value: f64) -> Result<Self::Value, E> {
        Err(E::custom("non-integer JSON numbers are outside the SAT restricted profile"))
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
        Ok(StrictJson::String(value.to_owned()))
    }

    fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
        Ok(StrictJson::String(value))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<StrictJson>()? {
            values.push(value);
        }
        Ok(StrictJson::Array(values))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut entries = Vec::new();
        let mut seen = HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate JSON object key: {key}")));
            }
            entries.push((key, map.next_value::<StrictJson>()?));
        }
        Ok(StrictJson::Object(entries))
    }
}

impl<'de> Deserialize<'de> for StrictJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

impl StrictJson {
    fn get(&self, key: &str) -> Option<&StrictJson> {
        match self {
            Self::Object(entries) => entries
                .iter()
                .find(|(candidate, _)| candidate == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    fn string(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    fn array(&self) -> Option<&[StrictJson]> {
        match self {
            Self::Array(values) => Some(values),
            _ => None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Augmentation {
    schema_version: String,
    augmentation_id: String,
    created_at: String,
    native_benchmark: Value,
    subject: Value,
    run_context: Value,
    observations: Vec<Observation>,
    claims: Vec<Claim>,
    claim_edges: Vec<ClaimEdge>,
    evidence_graph_root_digest: String,
    native_metrics_ref: String,
    sat_metrics: Value,
    qualification_disposition: String,
    #[serde(default)]
    report_ref: Option<String>,
    #[serde(default)]
    report_digest: Option<String>,
    limitations: Vec<String>,
    #[serde(default)]
    supersedes_augmentation_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Observation {
    observation_id: String,
    kind: String,
    artifact_ref: String,
    artifact_digest: String,
    source_id: String,
    source_role: String,
    observed_at: String,
    scope_ref: String,
    #[serde(default)]
    attempt_id: Option<String>,
    authenticity_state: String,
    independence_state: String,
    #[serde(default)]
    limitations: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Claim {
    claim_id: String,
    claim_type: String,
    proposition: String,
    scope_ref: String,
    status: String,
    evidence_refs: Vec<String>,
    depends_on: Vec<String>,
    validity_state: String,
    #[serde(default)]
    valid_from: Option<String>,
    #[serde(default)]
    valid_until: Option<String>,
    consequence_if_false: String,
    #[serde(default)]
    verifier_ref: Option<String>,
    #[serde(default)]
    invalidation_triggers: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ClaimEdge {
    from_claim_id: String,
    to_claim_id: String,
    relation: String,
    #[serde(default)]
    rule_ref: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VerificationIssue {
    pub code: String,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct VerificationReport {
    pub schema_version: String,
    /// Full JSON Schema validation is an upstream, separately versioned step.
    pub json_schema_validation_performed: bool,
    pub augmentation_id: String,
    pub observation_count: usize,
    pub claim_count: usize,
    pub edge_count: usize,
    pub dependency_acyclic: bool,
    pub supported_claim_count: usize,
    pub supported_claims_passing_declared_state_checks: usize,
    pub computed_graph_root_digest: String,
    pub recorded_graph_root_digest: String,
    pub graph_root_matches: bool,
    pub semantic_valid: bool,
    pub graph_integrity_passed: bool,
    /// Always false: this crate does not verify source signatures or issuer key status.
    pub authenticity_verified: bool,
    /// Always false: declared independence labels are not a common-cause proof.
    pub source_independence_verified: bool,
    /// Always false: graph verification is not system qualification.
    pub qualification_assessed: bool,
    pub qualification_boundary: String,
    pub issues: Vec<VerificationIssue>,
}

fn issue(issues: &mut Vec<VerificationIssue>, code: &str, path: impl Into<String>, message: impl Into<String>) {
    issues.push(VerificationIssue {
        code: code.to_owned(),
        path: path.into(),
        message: message.into(),
    });
}

fn parse_strict_json(input: &str) -> Result<StrictJson, String> {
    if input.len() > MAX_AUGMENTATION_BYTES {
        return Err(format!("augmentation exceeds the {MAX_AUGMENTATION_BYTES}-byte input limit"));
    }
    let mut deserializer = serde_json::Deserializer::from_str(input);
    let value = StrictJson::deserialize(&mut deserializer).map_err(|error| error.to_string())?;
    deserializer.end().map_err(|error| error.to_string())?;
    Ok(value)
}

fn utf16_cmp(left: &str, right: &str) -> Ordering {
    left.encode_utf16().cmp(right.encode_utf16())
}

fn canonical_json(value: &StrictJson, output: &mut String) {
    match value {
        StrictJson::Null => output.push_str("null"),
        StrictJson::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        StrictJson::Integer(value) => output.push_str(&value.to_string()),
        StrictJson::String(value) => {
            output.push_str(&serde_json::to_string(value).expect("serializing a Rust string cannot fail"));
        }
        StrictJson::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                canonical_json(value, output);
            }
            output.push(']');
        }
        StrictJson::Object(entries) => {
            let mut ordered: Vec<_> = entries.iter().collect();
            ordered.sort_by(|(left, _), (right, _)| utf16_cmp(left, right));
            output.push('{');
            for (index, (key, value)) in ordered.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(&serde_json::to_string(key).expect("serializing a Rust string cannot fail"));
                output.push(':');
                canonical_json(value, output);
            }
            output.push('}');
        }
    }
}

fn sha256_domain(domain: &str, payload: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update(payload.as_bytes());
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest {
        use fmt::Write as _;
        write!(&mut hex, "{byte:02x}").expect("writing to String cannot fail");
    }
    format!("sha256:{hex}")
}

fn field_string<'a>(value: &'a StrictJson, key: &str) -> Option<&'a str> {
    value.get(key).and_then(StrictJson::string)
}

fn normalize_string_set(
    object: &mut StrictJson,
    field: &str,
    path: &str,
    issues: &mut Vec<VerificationIssue>,
) {
    let StrictJson::Object(fields) = object else {
        return;
    };
    let Some((_, value)) = fields.iter_mut().find(|(name, _)| name == field) else {
        if field == "invalidation_triggers" {
            fields.push((field.to_owned(), StrictJson::Array(Vec::new())));
        }
        return;
    };
    let StrictJson::Array(values) = value else {
        issue(
            issues,
            "invalid_set_field",
            format!("{path}.{field}"),
            "canonical set-valued claim field must be an array",
        );
        return;
    };
    let mut seen = HashSet::new();
    for (index, entry) in values.iter().enumerate() {
        match entry.string() {
            Some(text) => {
                if !seen.insert(text.to_owned()) {
                    issue(
                        issues,
                        "duplicate_set_member",
                        format!("{path}.{field}[{index}]"),
                        format!("duplicate member {text:?} in canonical set"),
                    );
                }
            }
            None => issue(
                issues,
                "invalid_set_member",
                format!("{path}.{field}[{index}]"),
                "canonical set members must be strings",
            ),
        }
    }
    values.sort_by(|left, right| {
        utf16_cmp(left.string().unwrap_or_default(), right.string().unwrap_or_default())
    });
}

fn sort_graph_array(
    values: &mut StrictJson,
    field: &str,
    path: &str,
    issues: &mut Vec<VerificationIssue>,
) {
    let StrictJson::Array(items) = values else {
        issue(issues, "invalid_graph_array", path, "graph member must be an array");
        return;
    };
    items.sort_by(|left, right| {
        utf16_cmp(
            field_string(left, field).unwrap_or_default(),
            field_string(right, field).unwrap_or_default(),
        )
    });
}

fn normalize_graph_payload(raw: &StrictJson, issues: &mut Vec<VerificationIssue>) -> Result<StrictJson, String> {
    for (field, limit) in [
        ("observations", MAX_OBSERVATIONS),
        ("claims", MAX_CLAIMS),
        ("claim_edges", MAX_CLAIM_EDGES),
    ] {
        let count = raw
            .get(field)
            .and_then(StrictJson::array)
            .ok_or_else(|| format!("graph member {field} must be an array"))?
            .len();
        if count > limit {
            return Err(format!("graph member {field} has {count} entries; limit is {limit}"));
        }
    }

    let required = [
        "schema_version",
        "augmentation_id",
        "native_benchmark",
        "subject",
        "run_context",
        "observations",
        "claims",
        "claim_edges",
    ];
    let mut members = Vec::with_capacity(required.len());
    for field in required {
        let value = raw
            .get(field)
            .cloned()
            .ok_or_else(|| format!("graph payload is missing required field {field}"))?;
        members.push((field.to_owned(), value));
    }

    let observations = members.iter_mut().find(|(key, _)| key == "observations").unwrap();
    sort_graph_array(&mut observations.1, "observation_id", "observations", issues);

    let claims = members.iter_mut().find(|(key, _)| key == "claims").unwrap();
    if let StrictJson::Array(items) = &mut claims.1 {
        for (index, claim) in items.iter_mut().enumerate() {
            let path = format!("claims[{index}]");
            normalize_string_set(claim, "evidence_refs", &path, issues);
            normalize_string_set(claim, "depends_on", &path, issues);
            normalize_string_set(claim, "invalidation_triggers", &path, issues);
        }
        sort_graph_array(&mut claims.1, "claim_id", "claims", issues);
    } else {
        issue(issues, "invalid_graph_array", "claims", "claims must be an array");
    }

    let edges = members.iter_mut().find(|(key, _)| key == "claim_edges").unwrap();
    if let StrictJson::Array(items) = &mut edges.1 {
        items.sort_by(|left, right| {
            let left_tuple = (
                field_string(left, "from_claim_id").unwrap_or_default(),
                field_string(left, "to_claim_id").unwrap_or_default(),
                field_string(left, "relation").unwrap_or_default(),
                field_string(left, "rule_ref").unwrap_or_default(),
            );
            let right_tuple = (
                field_string(right, "from_claim_id").unwrap_or_default(),
                field_string(right, "to_claim_id").unwrap_or_default(),
                field_string(right, "relation").unwrap_or_default(),
                field_string(right, "rule_ref").unwrap_or_default(),
            );
            utf16_cmp(left_tuple.0, right_tuple.0)
                .then_with(|| utf16_cmp(left_tuple.1, right_tuple.1))
                .then_with(|| utf16_cmp(left_tuple.2, right_tuple.2))
                .then_with(|| utf16_cmp(left_tuple.3, right_tuple.3))
        });
    } else {
        issue(issues, "invalid_graph_array", "claim_edges", "claim_edges must be an array");
    }

    Ok(StrictJson::Object(members))
}

pub fn compute_evidence_graph_root(input: &str) -> Result<String, String> {
    let raw = parse_strict_json(input)?;
    let mut normalization_issues = Vec::new();
    let graph_payload = normalize_graph_payload(&raw, &mut normalization_issues)?;
    if !normalization_issues.is_empty() {
        return Err(format!(
            "graph normalization failed: {}",
            normalization_issues
                .iter()
                .map(|entry| format!("{} at {}", entry.code, entry.path))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let mut canonical = String::new();
    canonical_json(&graph_payload, &mut canonical);
    Ok(sha256_domain(GRAPH_DOMAIN, &canonical))
}

fn is_one_of(value: &str, allowed: &[&str]) -> bool {
    allowed.contains(&value)
}

fn valid_sha256(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn detect_dependency_cycles(
    adjacency: &HashMap<String, Vec<String>>,
    claim_ids: &HashSet<String>,
    issues: &mut Vec<VerificationIssue>,
) -> bool {
    // Kahn's algorithm avoids recursive DFS stack exhaustion on attacker-sized graphs.
    let mut indegree: HashMap<String, usize> =
        claim_ids.iter().cloned().map(|claim_id| (claim_id, 0)).collect();

    for (from, targets) in adjacency {
        if !claim_ids.contains(from) {
            continue;
        }
        for target in targets {
            if let Some(degree) = indegree.get_mut(target) {
                *degree += 1;
            }
        }
    }

    let mut queue = VecDeque::new();
    for (claim_id, degree) in &indegree {
        if *degree == 0 {
            queue.push_back(claim_id.clone());
        }
    }

    let mut visited = 0usize;
    while let Some(node) = queue.pop_front() {
        visited += 1;
        if let Some(targets) = adjacency.get(&node) {
            for target in targets {
                if let Some(degree) = indegree.get_mut(target) {
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(target.clone());
                    }
                }
            }
        }
    }

    if visited == claim_ids.len() {
        true
    } else {
        let mut cyclic: Vec<_> = indegree
            .iter()
            .filter(|(_, degree)| **degree > 0)
            .map(|(claim_id, _)| claim_id.clone())
            .collect();
        cyclic.sort_by(|left, right| utf16_cmp(left, right));
        issue(
            issues,
            "dependency_cycle",
            "claim_edges",
            format!("dependency subgraph contains a cycle involving {}", cyclic.join(", ")),
        );
        false
    }
}

fn check_supported_claims(
    claims: &[Claim],
    observations: &HashMap<String, &Observation>,
    claims_by_id: &HashMap<String, &Claim>,
    issues: &mut Vec<VerificationIssue>,
) -> (usize, usize) {
    let mut supported_count = 0;
    let mut own_admissible: HashMap<String, bool> = HashMap::new();

    for claim in claims.iter().filter(|claim| claim.status == "supported") {
        supported_count += 1;
        let base_path = format!("claims.{}", claim.claim_id);
        let mut admissible = true;

        if claim.validity_state != "current" {
            issue(
                issues,
                "supported_claim_not_current",
                format!("{base_path}.validity_state"),
                "a supported claim must have current validity state",
            );
            admissible = false;
        }
        if claim.evidence_refs.is_empty() {
            issue(
                issues,
                "supported_claim_without_evidence",
                format!("{base_path}.evidence_refs"),
                "a supported claim must reference at least one observation",
            );
            admissible = false;
        }

        let mut observed_kinds = HashSet::new();
        for reference in &claim.evidence_refs {
            let Some(observation) = observations.get(reference) else {
                issue(
                    issues,
                    "dangling_evidence_reference",
                    format!("{base_path}.evidence_refs"),
                    format!("observation {reference} does not resolve"),
                );
                admissible = false;
                continue;
            };
            observed_kinds.insert(observation.kind.clone());

            let auth_ok = matches!(observation.authenticity_state.as_str(), "verified" | "not_applicable");
            let independence_ok = matches!(observation.independence_state.as_str(), "satisfied" | "not_applicable");
            if !auth_ok {
                issue(
                    issues,
                    "unsupported_evidence_authenticity",
                    format!("observations.{reference}.authenticity_state"),
                    "evidence with unknown or invalid authenticity cannot support a claim",
                );
                admissible = false;
            }
            if claim.consequence_if_false == "critical"
                && (observation.authenticity_state != "verified"
                    || observation.independence_state != "satisfied")
            {
                issue(
                    issues,
                    "critical_evidence_not_independent_and_verified",
                    format!("observations.{reference}"),
                    "critical claims require verified authenticity and satisfied independence for every supporting observation",
                );
                admissible = false;
            } else if !independence_ok {
                issue(
                    issues,
                    "unsupported_evidence_independence",
                    format!("observations.{reference}.independence_state"),
                    "evidence with unsatisfied or unverified independence cannot support a claim",
                );
                admissible = false;
            }
        }

        let has_required_kind = match claim.claim_type.as_str() {
            "authority" => observed_kinds.contains("authority_decision"),
            "disclosure" => observed_kinds.contains("disclosure_decision"),
            "effect_correspondence" => observed_kinds.contains("external_effect_readback"),
            "transparency_continuity" => {
                observed_kinds.contains("transparency_receipt") || observed_kinds.contains("checkpoint")
            }
            "witness_independence" => observed_kinds.contains("witness_attestation"),
            _ => true,
        };
        if !has_required_kind {
            issue(
                issues,
                "missing_claim_type_evidence",
                format!("{base_path}.evidence_refs"),
                format!("claim type {} lacks its required observation kind", claim.claim_type),
            );
            admissible = false;
        }
        if claim.claim_type == "effect_correspondence" {
            let has_independent_sink = claim.evidence_refs.iter().any(|reference| {
                observations.get(reference).is_some_and(|observation| {
                    observation.kind == "external_effect_readback"
                        && observation.source_role == "independent_sink"
                        && observation.authenticity_state == "verified"
                        && observation.independence_state == "satisfied"
                })
            });
            if !has_independent_sink {
                issue(
                    issues,
                    "effect_claim_without_independent_sink",
                    format!("{base_path}.evidence_refs"),
                    "effect-correspondence claim requires a verified independent_sink read-back",
                );
                admissible = false;
            }
        }

        own_admissible.insert(claim.claim_id.clone(), admissible);
    }

    // Propagate admissibility through the complete dependency DAG. A claim's
    // own evidence can be sound while one of its premises is not; status labels
    // alone must never let that dependent claim pass.
    let mut remaining_dependencies: HashMap<String, usize> =
        claims.iter().map(|claim| (claim.claim_id.clone(), 0)).collect();
    let mut dependents: HashMap<String, Vec<String>> = HashMap::new();

    for claim in claims {
        for dependency in &claim.depends_on {
            if claims_by_id.contains_key(dependency) {
                *remaining_dependencies.entry(claim.claim_id.clone()).or_insert(0) += 1;
                dependents
                    .entry(dependency.clone())
                    .or_default()
                    .push(claim.claim_id.clone());
            } else if claim.status == "supported" {
                issue(
                    issues,
                    "unknown_claim_dependency",
                    format!("claims.{}.depends_on", claim.claim_id),
                    format!("claim dependency {dependency} does not resolve"),
                );
                own_admissible.insert(claim.claim_id.clone(), false);
            }
        }
    }

    let mut ready = VecDeque::new();
    for claim in claims {
        if remaining_dependencies.get(&claim.claim_id).copied().unwrap_or(0) == 0 {
            ready.push_back(claim.claim_id.clone());
        }
    }

    let mut final_admissible: HashMap<String, bool> = HashMap::new();
    let mut dependency_failed: HashSet<String> = HashSet::new();

    while let Some(claim_id) = ready.pop_front() {
        let claim = match claims_by_id.get(&claim_id) {
            Some(claim) => *claim,
            None => continue,
        };
        let mut passes = own_admissible.get(&claim_id).copied().unwrap_or(false)
            && claim.status == "supported"
            && claim.validity_state == "current";

        if dependency_failed.contains(&claim_id) {
            passes = false;
            if claim.status == "supported" {
                let failed: Vec<_> = claim
                    .depends_on
                    .iter()
                    .filter(|dependency| {
                        final_admissible.get(*dependency).copied() == Some(false)
                    })
                    .cloned()
                    .collect();
                issue(
                    issues,
                    "supported_claim_has_inadmissible_dependency",
                    format!("claims.{claim_id}.depends_on"),
                    format!("one or more transitive premises failed declared-state admissibility: {}", failed.join(", ")),
                );
            }
        }

        final_admissible.insert(claim_id.clone(), passes);

        if let Some(children) = dependents.get(&claim_id) {
            for child in children {
                if !passes {
                    dependency_failed.insert(child.clone());
                }
                if let Some(remaining) = remaining_dependencies.get_mut(child) {
                    *remaining = remaining.saturating_sub(1);
                    if *remaining == 0 {
                        ready.push_back(child.clone());
                    }
                }
            }
        }
    }

    // Any node not reached by Kahn's traversal is in or downstream of a cycle.
    // The global cycle diagnostic is emitted by the graph-cycle check as well.
    for claim in claims {
        if !final_admissible.contains_key(&claim.claim_id) {
            if claim.status == "supported" {
                issue(
                    issues,
                    "supported_claim_dependency_unresolved",
                    format!("claims.{}.depends_on", claim.claim_id),
                    "dependency could not be resolved through an acyclic admissibility order",
                );
            }
            final_admissible.insert(claim.claim_id.clone(), false);
        }
    }

    let admissible_count = claims
        .iter()
        .filter(|claim| {
            claim.status == "supported"
                && final_admissible.get(&claim.claim_id).copied() == Some(true)
        })
        .count();

    (supported_count, admissible_count)
}

pub fn verify_augmentation(input: &str) -> Result<VerificationReport, String> {
    let raw = parse_strict_json(input)?;
    let augmentation: Augmentation =
        serde_json::from_str(input).map_err(|error| format!("augmentation shape: {error}"))?;

    if augmentation.observations.len() > MAX_OBSERVATIONS
        || augmentation.claims.len() > MAX_CLAIMS
        || augmentation.claim_edges.len() > MAX_CLAIM_EDGES
    {
        return Err(format!(
            "graph exceeds resource limits: observations={}/{MAX_OBSERVATIONS}, claims={}/{MAX_CLAIMS}, edges={}/{MAX_CLAIM_EDGES}",
            augmentation.observations.len(),
            augmentation.claims.len(),
            augmentation.claim_edges.len()
        ));
    }

    let mut issues = Vec::new();
    if augmentation.schema_version != GRAPH_SCHEMA_VERSION {
        issue(
            &mut issues,
            "unsupported_schema_version",
            "schema_version",
            format!("expected {GRAPH_SCHEMA_VERSION}"),
        );
    }

    let mut observation_map: HashMap<String, &Observation> = HashMap::new();
    for (index, observation) in augmentation.observations.iter().enumerate() {
        if observation_map
            .insert(observation.observation_id.clone(), observation)
            .is_some()
        {
            issue(
                &mut issues,
                "duplicate_observation_id",
                format!("observations[{index}].observation_id"),
                format!("duplicate observation ID {}", observation.observation_id),
            );
        }
        if !is_one_of(
            &observation.kind,
            &[
                "native_trace", "authority_decision", "disclosure_decision",
                "external_effect_readback", "source_attestation", "transparency_receipt",
                "checkpoint", "witness_attestation", "auditor_finding",
                "human_adjudication", "other",
            ],
        ) {
            issue(
                &mut issues,
                "unknown_observation_kind",
                format!("observations[{index}].kind"),
                "observation kind is outside the SAT v0.1 vocabulary",
            );
        }
        if !is_one_of(
            &observation.source_role,
            &[
                "subject", "benchmark_harness", "native_grader", "independent_sink",
                "independent_witness", "evaluator", "human_adjudicator", "other",
            ],
        ) {
            issue(
                &mut issues,
                "unknown_observation_source_role",
                format!("observations[{index}].source_role"),
                "source role is outside the SAT v0.1 vocabulary",
            );
        }
        if !is_one_of(
            &observation.authenticity_state,
            &["verified", "invalid", "unknown", "not_applicable"],
        ) {
            issue(
                &mut issues,
                "unknown_authenticity_state",
                format!("observations[{index}].authenticity_state"),
                "authenticity state is outside the SAT v0.1 vocabulary",
            );
        }
        if !is_one_of(
            &observation.independence_state,
            &["satisfied", "not_satisfied", "unverified", "not_applicable"],
        ) {
            issue(
                &mut issues,
                "unknown_independence_state",
                format!("observations[{index}].independence_state"),
                "independence state is outside the SAT v0.1 vocabulary",
            );
        }
        if !valid_sha256(&observation.artifact_digest) {
            issue(
                &mut issues,
                "invalid_artifact_digest",
                format!("observations[{index}].artifact_digest"),
                "artifact digest must be lowercase sha256:<64 hex characters>",
            );
        }
        let _metadata = (
            &observation.artifact_ref,
            &observation.source_id,
            &observation.source_role,
            &observation.observed_at,
            &observation.scope_ref,
            &observation.attempt_id,
            &observation.limitations,
        );
    }

    let mut claims_by_id: HashMap<String, &Claim> = HashMap::new();
    let mut claim_ids = HashSet::new();
    for (index, claim) in augmentation.claims.iter().enumerate() {
        if claims_by_id.insert(claim.claim_id.clone(), claim).is_some() {
            issue(
                &mut issues,
                "duplicate_claim_id",
                format!("claims[{index}].claim_id"),
                format!("duplicate claim ID {}", claim.claim_id),
            );
        }
        claim_ids.insert(claim.claim_id.clone());

        if !is_one_of(
            &claim.claim_type,
            &[
                "native_task_outcome", "authority", "disclosure", "effect_correspondence",
                "evidence_authenticity", "transparency_continuity", "witness_independence",
                "isolation", "auditor_finding", "change_continuity", "human_agency", "other",
            ],
        ) {
            issue(
                &mut issues,
                "unknown_claim_type",
                format!("claims[{index}].claim_type"),
                "claim type is outside the SAT v0.1 vocabulary",
            );
        }
        if !is_one_of(
            &claim.status,
            &["supported", "refuted", "inconclusive", "invalid_evidence", "withdrawn"],
        ) {
            issue(
                &mut issues,
                "unknown_claim_status",
                format!("claims[{index}].status"),
                "claim status is outside the SAT v0.1 vocabulary",
            );
        }
        if !is_one_of(&claim.validity_state, &["current", "stale", "expired", "unknown"]) {
            issue(
                &mut issues,
                "unknown_claim_validity_state",
                format!("claims[{index}].validity_state"),
                "claim validity state is outside the SAT v0.1 vocabulary",
            );
        }
        if !is_one_of(
            &claim.consequence_if_false,
            &["critical", "high", "medium", "low", "informational"],
        ) {
            issue(
                &mut issues,
                "unknown_claim_consequence",
                format!("claims[{index}].consequence_if_false"),
                "claim consequence is outside the SAT v0.1 vocabulary",
            );
        }

        for (field, values) in [
            ("evidence_refs", &claim.evidence_refs),
            ("depends_on", &claim.depends_on),
            ("invalidation_triggers", &claim.invalidation_triggers),
        ] {
            let unique: HashSet<_> = values.iter().collect();
            if unique.len() != values.len() {
                issue(
                    &mut issues,
                    "duplicate_set_member",
                    format!("claims[{index}].{field}"),
                    "set-valued claim field contains duplicate members",
                );
            }
        }
        for reference in &claim.evidence_refs {
            if !observation_map.contains_key(reference) {
                issue(
                    &mut issues,
                    "dangling_evidence_reference",
                    format!("claims[{index}].evidence_refs"),
                    format!("observation {reference} does not resolve exactly once"),
                );
            }
        }


        let _claim_metadata = (
            &claim.proposition,
            &claim.scope_ref,
            &claim.valid_from,
            &claim.valid_until,
            &claim.verifier_ref,
        );
    }

    for (index, claim) in augmentation.claims.iter().enumerate() {
        for dependency in &claim.depends_on {
            if !claim_ids.contains(dependency) {
                issue(
                    &mut issues,
                    "unknown_claim_dependency",
                    format!("claims[{index}].depends_on"),
                    format!("claim dependency {dependency} does not resolve"),
                );
            }
        }
    }

    let mut edge_keys = HashSet::new();
    let mut edge_dependency_counts: HashMap<(String, String), usize> = HashMap::new();
    let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
    for (index, edge) in augmentation.claim_edges.iter().enumerate() {
        if !claim_ids.contains(&edge.from_claim_id) {
            issue(
                &mut issues,
                "unknown_edge_source",
                format!("claim_edges[{index}].from_claim_id"),
                format!("claim {} does not exist", edge.from_claim_id),
            );
        }
        if !claim_ids.contains(&edge.to_claim_id) {
            issue(
                &mut issues,
                "unknown_edge_target",
                format!("claim_edges[{index}].to_claim_id"),
                format!("claim {} does not exist", edge.to_claim_id),
            );
        }
        if !is_one_of(
            &edge.relation,
            &["depends_on", "supports", "refutes", "supersedes", "invalidated_by"],
        ) {
            issue(
                &mut issues,
                "unknown_edge_relation",
                format!("claim_edges[{index}].relation"),
                "edge relation is outside the SAT v0.1 vocabulary",
            );
        }
        if edge.rule_ref.as_ref().is_some_and(String::is_empty) {
            issue(
                &mut issues,
                "empty_edge_rule_reference",
                format!("claim_edges[{index}].rule_ref"),
                "rule_ref must be non-empty when present",
            );
        }
        let key = (
            edge.from_claim_id.clone(),
            edge.to_claim_id.clone(),
            edge.relation.clone(),
            edge.rule_ref.clone().unwrap_or_default(),
        );
        if !edge_keys.insert(key) {
            issue(
                &mut issues,
                "duplicate_claim_edge",
                format!("claim_edges[{index}]"),
                "duplicate edge tuple",
            );
        }
        if edge.relation == "depends_on" {
            *edge_dependency_counts
                .entry((edge.from_claim_id.clone(), edge.to_claim_id.clone()))
                .or_insert(0) += 1;
            adjacency
                .entry(edge.from_claim_id.clone())
                .or_default()
                .push(edge.to_claim_id.clone());
        }
        let _edge_metadata = &edge.rule_ref;
    }

    let mut claim_dependency_pairs = HashSet::new();
    for claim in &augmentation.claims {
        for dependency in &claim.depends_on {
            claim_dependency_pairs.insert((claim.claim_id.clone(), dependency.clone()));
        }
    }
    for (from, to) in &claim_dependency_pairs {
        match edge_dependency_counts.get(&(from.clone(), to.clone())).copied().unwrap_or(0) {
            0 => issue(
                &mut issues,
                "missing_dependency_edge",
                format!("claims.{from}.depends_on"),
                format!("dependency {from} -> {to} lacks a corresponding claim_edges entry"),
            ),
            1 => {}
            _ => issue(
                &mut issues,
                "duplicate_dependency_edge",
                format!("claim_edges.{from}.{to}"),
                format!("dependency {from} -> {to} is represented by more than one typed edge"),
            ),
        }
    }
    for (from, to) in edge_dependency_counts.keys() {
        if !claim_dependency_pairs.contains(&(from.clone(), to.clone())) {
            issue(
                &mut issues,
                "orphan_dependency_edge",
                format!("claim_edges.{from}.{to}"),
                format!("depends_on edge {from} -> {to} is absent from the source claim's depends_on set"),
            );
        }
    }

    let dependency_acyclic = detect_dependency_cycles(&adjacency, &claim_ids, &mut issues);
    let (supported_claim_count, admissible_supported_claim_count) =
        check_supported_claims(&augmentation.claims, &observation_map, &claims_by_id, &mut issues);

    let computed_graph_root_digest = {
        let mut normalization_issues = Vec::new();
        let graph_payload = normalize_graph_payload(&raw, &mut normalization_issues)?;
        issues.extend(normalization_issues);
        let mut canonical = String::new();
        canonical_json(&graph_payload, &mut canonical);
        sha256_domain(GRAPH_DOMAIN, &canonical)
    };
    let graph_root_matches = computed_graph_root_digest == augmentation.evidence_graph_root_digest;
    if !valid_sha256(&augmentation.evidence_graph_root_digest) {
        issue(
            &mut issues,
            "invalid_graph_root_digest",
            "evidence_graph_root_digest",
            "graph root must be lowercase sha256:<64 hex characters>",
        );
    } else if !graph_root_matches {
        issue(
            &mut issues,
            "graph_root_mismatch",
            "evidence_graph_root_digest",
            "recorded root does not match the normalized graph payload",
        );
    }

    issues.sort_by(|left, right| {
        utf16_cmp(&left.code, &right.code)
            .then_with(|| utf16_cmp(&left.path, &right.path))
            .then_with(|| utf16_cmp(&left.message, &right.message))
    });

    let semantic_valid = issues
        .iter()
        .all(|entry| entry.code != "graph_root_mismatch");
    let graph_integrity_passed = semantic_valid && graph_root_matches && valid_sha256(&augmentation.evidence_graph_root_digest);
    let _top_level_metadata = (
        &augmentation.created_at,
        &augmentation.native_benchmark,
        &augmentation.subject,
        &augmentation.run_context,
        &augmentation.native_metrics_ref,
        &augmentation.sat_metrics,
        &augmentation.qualification_disposition,
        &augmentation.report_ref,
        &augmentation.report_digest,
        &augmentation.limitations,
        &augmentation.supersedes_augmentation_id,
    );

    Ok(VerificationReport {
        schema_version: GRAPH_SCHEMA_VERSION.to_owned(),
        json_schema_validation_performed: false,
        augmentation_id: augmentation.augmentation_id,
        observation_count: augmentation.observations.len(),
        claim_count: augmentation.claims.len(),
        edge_count: augmentation.claim_edges.len(),
        dependency_acyclic,
        supported_claim_count,
        supported_claims_passing_declared_state_checks: admissible_supported_claim_count,
        computed_graph_root_digest,
        recorded_graph_root_digest: augmentation.evidence_graph_root_digest,
        graph_root_matches,
        semantic_valid,
        graph_integrity_passed,
        authenticity_verified: false,
        source_independence_verified: false,
        qualification_assessed: false,
        qualification_boundary: "Graph structure and restricted root commitment only; not cryptographic authenticity, independence proof, effect truth or system qualification.".to_owned(),
        issues,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn base() -> Value {
        json!({
            "schema_version": "sat-benchmark-augmentation/v0.1",
            "augmentation_id": "SAT-TEST-001",
            "created_at": "2026-11-26T12:00:00Z",
            "native_benchmark": {},
            "subject": {},
            "run_context": {},
            "observations": [{
                "observation_id": "obs-1",
                "kind": "other",
                "artifact_ref": "artifact:test:1",
                "artifact_digest": format!("sha256:{}", "a".repeat(64)),
                "source_id": "source-1",
                "source_role": "other",
                "observed_at": "2026-11-26T12:00:00Z",
                "scope_ref": "scope:test",
                "authenticity_state": "verified",
                "independence_state": "satisfied",
                "limitations": []
            }],
            "claims": [{
                "claim_id": "claim-1",
                "claim_type": "other",
                "proposition": "The test evidence is admissible.",
                "scope_ref": "scope:test",
                "status": "supported",
                "evidence_refs": ["obs-1"],
                "depends_on": [],
                "validity_state": "current",
                "consequence_if_false": "low"
            }],
            "claim_edges": [],
            "evidence_graph_root_digest": format!("sha256:{}", "0".repeat(64)),
            "native_metrics_ref": "native:metrics:test",
            "sat_metrics": {},
            "qualification_disposition": "inconclusive",
            "limitations": []
        })
    }

    fn seal(value: &mut Value) {
        let input = serde_json::to_string(value).unwrap();
        let root = compute_evidence_graph_root(&input).unwrap();
        value["evidence_graph_root_digest"] = json!(root);
    }

    fn has_issue(report: &VerificationReport, code: &str) -> bool {
        report.issues.iter().any(|entry| entry.code == code)
    }

    #[test]
    fn valid_minimal_graph_recomputes_and_matches_root() {
        let mut value = base();
        seal(&mut value);
        let report = verify_augmentation(&serde_json::to_string(&value).unwrap()).unwrap();
        assert!(report.graph_integrity_passed, "{:?}", report.issues);
        assert_eq!(report.supported_claim_count, 1);
        assert_eq!(report.supported_claims_passing_declared_state_checks, 1);
        assert_eq!(
            report.computed_graph_root_digest,
            "sha256:e2b677dd4ad89972b5a306558a4f1fd32680749cfafd9aeb3a25102343851b82"
        );
        assert!(!report.qualification_assessed);
        assert!(!report.json_schema_validation_performed);
    }

    #[test]
    fn omitted_optional_invalidation_set_equals_explicit_empty_set() {
        let omitted = base();
        let mut explicit = base();
        explicit["claims"][0]["invalidation_triggers"] = json!([]);
        let a = compute_evidence_graph_root(&serde_json::to_string(&omitted).unwrap()).unwrap();
        let b = compute_evidence_graph_root(&serde_json::to_string(&explicit).unwrap()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn graph_root_is_invariant_to_permitted_graph_array_order() {
        let mut left = base();
        left["observations"].as_array_mut().unwrap().push(json!({
            "observation_id": "obs-2", "kind": "other", "artifact_ref": "artifact:test:2",
            "artifact_digest": format!("sha256:{}", "b".repeat(64)), "source_id": "source-2",
            "source_role": "other", "observed_at": "2026-11-26T12:00:00Z", "scope_ref": "scope:test",
            "authenticity_state": "verified", "independence_state": "satisfied", "limitations": []
        }));
        let mut right = left.clone();
        right["observations"].as_array_mut().unwrap().reverse();
        let a = compute_evidence_graph_root(&serde_json::to_string(&left).unwrap()).unwrap();
        let b = compute_evidence_graph_root(&serde_json::to_string(&right).unwrap()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn dangling_evidence_reference_is_rejected() {
        let mut value = base();
        value["claims"][0]["evidence_refs"] = json!(["obs-missing"]);
        seal(&mut value);
        let report = verify_augmentation(&serde_json::to_string(&value).unwrap()).unwrap();
        assert!(!report.graph_integrity_passed);
        assert!(has_issue(&report, "dangling_evidence_reference"));
    }

    #[test]
    fn duplicate_observation_identity_is_rejected() {
        let mut value = base();
        let duplicate = value["observations"][0].clone();
        value["observations"].as_array_mut().unwrap().push(duplicate);
        seal(&mut value);
        let report = verify_augmentation(&serde_json::to_string(&value).unwrap()).unwrap();
        assert!(has_issue(&report, "duplicate_observation_id"));
    }

    #[test]
    fn unknown_edge_endpoint_is_rejected() {
        let mut value = base();
        value["claim_edges"] = json!([{
            "from_claim_id": "claim-1", "to_claim_id": "claim-missing",
            "relation": "supports", "rule_ref": "rule:test"
        }]);
        seal(&mut value);
        let report = verify_augmentation(&serde_json::to_string(&value).unwrap()).unwrap();
        assert!(has_issue(&report, "unknown_edge_target"));
    }

    #[test]
    fn dependency_cycle_is_rejected() {
        let mut value = base();
        value["claims"].as_array_mut().unwrap().push(json!({
            "claim_id": "claim-2", "claim_type": "other", "proposition": "Second premise",
            "scope_ref": "scope:test", "status": "supported", "evidence_refs": ["obs-1"],
            "depends_on": ["claim-1"], "validity_state": "current", "consequence_if_false": "low"
        }));
        value["claims"][0]["depends_on"] = json!(["claim-2"]);
        value["claim_edges"] = json!([
            {"from_claim_id":"claim-1","to_claim_id":"claim-2","relation":"depends_on","rule_ref":"rule:a"},
            {"from_claim_id":"claim-2","to_claim_id":"claim-1","relation":"depends_on","rule_ref":"rule:b"}
        ]);
        seal(&mut value);
        let report = verify_augmentation(&serde_json::to_string(&value).unwrap()).unwrap();
        assert!(!report.dependency_acyclic);
        assert!(has_issue(&report, "dependency_cycle"));
    }

    #[test]
    fn claim_dependency_and_typed_edge_must_agree() {
        let mut value = base();
        value["claims"].as_array_mut().unwrap().push(json!({
            "claim_id": "claim-2", "claim_type": "other", "proposition": "Second premise",
            "scope_ref": "scope:test", "status": "supported", "evidence_refs": ["obs-1"],
            "depends_on": ["claim-1"], "validity_state": "current", "consequence_if_false": "low"
        }));
        seal(&mut value);
        let report = verify_augmentation(&serde_json::to_string(&value).unwrap()).unwrap();
        assert!(has_issue(&report, "missing_dependency_edge"));
    }

    #[test]
    fn critical_claim_rejects_unverified_or_non_independent_evidence() {
        let mut value = base();
        value["claims"][0]["consequence_if_false"] = json!("critical");
        value["observations"][0]["authenticity_state"] = json!("unknown");
        value["observations"][0]["independence_state"] = json!("unverified");
        seal(&mut value);
        let report = verify_augmentation(&serde_json::to_string(&value).unwrap()).unwrap();
        assert!(has_issue(&report, "critical_evidence_not_independent_and_verified"));
        assert_eq!(report.supported_claims_passing_declared_state_checks, 0);
    }

    #[test]
    fn supported_claim_cannot_depend_on_inconclusive_claim() {
        let mut value = base();
        value["claims"].as_array_mut().unwrap().push(json!({
            "claim_id":"claim-2", "claim_type":"other", "proposition":"Dependent claim",
            "scope_ref":"scope:test", "status":"supported", "evidence_refs":["obs-1"],
            "depends_on":["claim-1"], "validity_state":"current", "consequence_if_false":"low"
        }));
        value["claims"][0]["status"] = json!("inconclusive");
        value["claim_edges"] = json!([{
            "from_claim_id":"claim-2","to_claim_id":"claim-1","relation":"depends_on","rule_ref":"rule:premise"
        }]);
        seal(&mut value);
        let report = verify_augmentation(&serde_json::to_string(&value).unwrap()).unwrap();
        assert!(has_issue(&report, "supported_claim_has_unsupported_dependency"));
    }

    #[test]
    fn supported_dependency_chain_passes_when_every_premise_passes() {
        let mut value = base();
        value["claims"].as_array_mut().unwrap().push(json!({
            "claim_id": "claim-2", "claim_type": "other", "proposition": "Second claim",
            "scope_ref": "scope:test", "status": "supported", "evidence_refs": ["obs-1"],
            "depends_on": ["claim-1"], "validity_state": "current", "consequence_if_false": "low"
        }));
        value["claims"].as_array_mut().unwrap().push(json!({
            "claim_id": "claim-3", "claim_type": "other", "proposition": "Third claim",
            "scope_ref": "scope:test", "status": "supported", "evidence_refs": ["obs-1"],
            "depends_on": ["claim-2"], "validity_state": "current", "consequence_if_false": "low"
        }));
        value["claim_edges"] = json!([
            {"from_claim_id":"claim-2","to_claim_id":"claim-1","relation":"depends_on","rule_ref":"rule:premise-1"},
            {"from_claim_id":"claim-3","to_claim_id":"claim-2","relation":"depends_on","rule_ref":"rule:premise-2"}
        ]);
        seal(&mut value);
        let report = verify_augmentation(&serde_json::to_string(&value).unwrap()).unwrap();
        assert!(report.graph_integrity_passed, "{:?}", report.issues);
        assert_eq!(report.supported_claim_count, 3);
        assert_eq!(report.supported_claims_passing_declared_state_checks, 3);
    }

    #[test]
    fn unsupported_premise_propagates_through_multiple_dependency_levels() {
        let mut value = base();
        value["claims"][0]["evidence_refs"] = json!(["obs-missing"]);
        value["claims"].as_array_mut().unwrap().push(json!({
            "claim_id": "claim-2", "claim_type": "other", "proposition": "Second claim",
            "scope_ref": "scope:test", "status": "supported", "evidence_refs": ["obs-1"],
            "depends_on": ["claim-1"], "validity_state": "current", "consequence_if_false": "low"
        }));
        value["claims"].as_array_mut().unwrap().push(json!({
            "claim_id": "claim-3", "claim_type": "other", "proposition": "Third claim",
            "scope_ref": "scope:test", "status": "supported", "evidence_refs": ["obs-1"],
            "depends_on": ["claim-2"], "validity_state": "current", "consequence_if_false": "low"
        }));
        value["claim_edges"] = json!([
            {"from_claim_id":"claim-2","to_claim_id":"claim-1","relation":"depends_on","rule_ref":"rule:premise-1"},
            {"from_claim_id":"claim-3","to_claim_id":"claim-2","relation":"depends_on","rule_ref":"rule:premise-2"}
        ]);
        seal(&mut value);
        let report = verify_augmentation(&serde_json::to_string(&value).unwrap()).unwrap();
        assert!(has_issue(&report, "dangling_evidence_reference"));
        assert_eq!(report.supported_claim_count, 3);
        assert_eq!(report.supported_claims_passing_declared_state_checks, 0);
        assert!(report.issues.iter().any(|entry| {
            entry.code == "supported_claim_has_inadmissible_dependency"
                && entry.path == "claims.claim-2.depends_on"
        }));
        assert!(report.issues.iter().any(|entry| {
            entry.code == "supported_claim_has_inadmissible_dependency"
                && entry.path == "claims.claim-3.depends_on"
        }));
    }

    #[test]
    fn supported_claim_cannot_inherit_admissibility_from_invalid_supported_premise() {
        let mut value = base();
        value["claims"][0]["evidence_refs"] = json!(["obs-missing"]);
        value["claims"].as_array_mut().unwrap().push(json!({
            "claim_id": "claim-2", "claim_type": "other", "proposition": "Dependent claim",
            "scope_ref": "scope:test", "status": "supported", "evidence_refs": ["obs-1"],
            "depends_on": ["claim-1"], "validity_state": "current", "consequence_if_false": "low"
        }));
        value["claim_edges"] = json!([{
            "from_claim_id": "claim-2", "to_claim_id": "claim-1",
            "relation": "depends_on", "rule_ref": "rule:invalid-premise"
        }]);
        seal(&mut value);
        let report = verify_augmentation(&serde_json::to_string(&value).unwrap()).unwrap();
        assert!(has_issue(&report, "dangling_evidence_reference"));
        assert!(has_issue(&report, "supported_claim_has_inadmissible_dependency"));
        assert_eq!(report.supported_claim_count, 2);
        assert_eq!(report.supported_claims_passing_declared_state_checks, 0);
    }

    #[test]
    fn duplicate_json_object_keys_are_rejected_before_typed_parse() {
        let input = r#"{"schema_version":"sat-benchmark-augmentation/v0.1","schema_version":"attacker"}"#;
        let error = verify_augmentation(input).unwrap_err();
        assert!(error.contains("duplicate JSON object key"));
    }

    #[test]
    fn non_integer_json_numbers_are_rejected() {
        let input = r#"{"run_context":{"seed":1.5}}"#;
        let error = compute_evidence_graph_root(input).unwrap_err();
        assert!(error.contains("non-integer JSON numbers"));
    }

    #[test]
    fn utf16_code_unit_order_controls_observation_sorting() {
        let mut value = base();
        value["observations"].as_array_mut().unwrap().push(json!({
            "observation_id": "\u{e000}", "kind": "other", "artifact_ref": "artifact:test:e000",
            "artifact_digest": format!("sha256:{}", "b".repeat(64)), "source_id": "source-e000",
            "source_role": "other", "observed_at": "2026-11-26T12:00:00Z", "scope_ref": "scope:test",
            "authenticity_state": "verified", "independence_state": "satisfied", "limitations": []
        }));
        value["observations"].as_array_mut().unwrap().push(json!({
            "observation_id": "\u{10000}", "kind": "other", "artifact_ref": "artifact:test:10000",
            "artifact_digest": format!("sha256:{}", "c".repeat(64)), "source_id": "source-10000",
            "source_role": "other", "observed_at": "2026-11-26T12:00:00Z", "scope_ref": "scope:test",
            "authenticity_state": "verified", "independence_state": "satisfied", "limitations": []
        }));
        let input = serde_json::to_string(&value).unwrap();
        let raw = parse_strict_json(&input).unwrap();
        let mut normalization_issues = Vec::new();
        let graph = normalize_graph_payload(&raw, &mut normalization_issues).unwrap();
        assert!(normalization_issues.is_empty());
        let ids: Vec<_> = graph
            .get("observations")
            .unwrap()
            .array()
            .unwrap()
            .iter()
            .map(|item| field_string(item, "observation_id").unwrap().to_owned())
            .collect();
        assert_eq!(ids[0], "\u{10000}");
        assert_eq!(ids[1], "\u{e000}");
    }

    #[test]
    fn more_than_one_dependency_edge_for_same_pair_is_rejected() {
        let mut value = base();
        value["claims"].as_array_mut().unwrap().push(json!({
            "claim_id": "claim-2", "claim_type": "other", "proposition": "Dependent claim",
            "scope_ref": "scope:test", "status": "supported", "evidence_refs": ["obs-1"],
            "depends_on": ["claim-1"], "validity_state": "current", "consequence_if_false": "low"
        }));
        value["claim_edges"] = json!([
            {"from_claim_id":"claim-2","to_claim_id":"claim-1","relation":"depends_on","rule_ref":"rule:a"},
            {"from_claim_id":"claim-2","to_claim_id":"claim-1","relation":"depends_on","rule_ref":"rule:b"}
        ]);
        seal(&mut value);
        let report = verify_augmentation(&serde_json::to_string(&value).unwrap()).unwrap();
        assert!(has_issue(&report, "duplicate_dependency_edge"));
    }

    #[test]
    fn effect_claim_requires_independent_sink_readback() {
        let mut value = base();
        value["claims"][0]["claim_type"] = json!("effect_correspondence");
        seal(&mut value);
        let report = verify_augmentation(&serde_json::to_string(&value).unwrap()).unwrap();
        assert!(has_issue(&report, "missing_claim_type_evidence"));
        assert!(has_issue(&report, "effect_claim_without_independent_sink"));
    }

    #[test]
    fn valid_effect_claim_requires_and_accepts_independent_sink_readback() {
        let mut value = base();
        value["claims"][0]["claim_type"] = json!("effect_correspondence");
        value["observations"][0]["kind"] = json!("external_effect_readback");
        value["observations"][0]["source_role"] = json!("independent_sink");
        seal(&mut value);
        let report = verify_augmentation(&serde_json::to_string(&value).unwrap()).unwrap();
        assert!(report.graph_integrity_passed, "{:?}", report.issues);
        assert_eq!(report.supported_claims_passing_declared_state_checks, 1);
        assert!(!report.authenticity_verified);
        assert!(!report.source_independence_verified);
        assert!(!report.qualification_assessed);
    }

    #[test]
    fn unknown_edge_relation_is_rejected() {
        let mut value = base();
        value["claim_edges"] = json!([{
            "from_claim_id":"claim-1", "to_claim_id":"claim-1",
            "relation":"trust_me", "rule_ref":"rule:invalid"
        }]);
        seal(&mut value);
        let report = verify_augmentation(&serde_json::to_string(&value).unwrap()).unwrap();
        assert!(has_issue(&report, "unknown_edge_relation"));
        assert!(!report.graph_integrity_passed);
    }

    #[test]
    fn recorded_root_mutation_is_detected() {
        let mut value = base();
        seal(&mut value);
        value["evidence_graph_root_digest"] = json!(format!("sha256:{}", "f".repeat(64)));
        let report = verify_augmentation(&serde_json::to_string(&value).unwrap()).unwrap();
        assert!(!report.graph_root_matches);
        assert!(has_issue(&report, "graph_root_mismatch"));
    }
}
