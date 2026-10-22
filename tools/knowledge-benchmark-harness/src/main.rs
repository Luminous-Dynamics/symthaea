use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const TOOL_VERSION: &str = "knowledge-benchmark-harness-v0.12.0";
const DRAFT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Case,
    Graph,
    Claim,
    Pair,
    Source,
    Annotation,
    Run,
    PairResult,
    SourcePack,
}

impl Kind {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "case" => Ok(Self::Case),
            "graph" => Ok(Self::Graph),
            "claim" => Ok(Self::Claim),
            "pair" => Ok(Self::Pair),
            "source" => Ok(Self::Source),
            "annotation" => Ok(Self::Annotation),
            "run" => Ok(Self::Run),
            "pair-result" => Ok(Self::PairResult),
            "source-pack" => Ok(Self::SourcePack),
            _ => Err(format!(
                "unknown --kind {raw:?}; expected case, graph, claim, pair, source, annotation, run, pair-result, or source-pack"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Case => "case",
            Self::Graph => "graph",
            Self::Claim => "claim",
            Self::Pair => "pair",
            Self::Source => "source",
            Self::Annotation => "annotation",
            Self::Run => "run",
            Self::PairResult => "pair-result",
            Self::SourcePack => "source-pack",
        }
    }
}

struct Config {
    schema: PathBuf,
    input: PathBuf,
    output: Option<PathBuf>,
    kind: Kind,
    subject_revision: Option<String>,
}

#[derive(Debug, Serialize)]
struct RecordResult {
    line: usize,
    record_id: String,
    schema_valid: bool,
    domain_contract_valid: bool,
    errors: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Report {
    report_schema_version: &'static str,
    tool_version: &'static str,
    kind: &'static str,
    subject_revision: Option<String>,
    schema_sha256: String,
    input_sha256: String,
    dependency_lock_sha256: String,
    declared_schema_dialect: Option<String>,
    schema_meta_valid: bool,
    schema_compiled: bool,
    record_count: usize,
    schema_valid_records: usize,
    domain_valid_records: usize,
    passed: bool,
    records: Vec<RecordResult>,
    nonclaims: Vec<&'static str>,
}

fn usage() -> &'static str {
    "Usage: knowledge-benchmark-harness --schema FILE --input JSONL --kind case|graph|claim|pair|source|annotation|run|pair-result|source-pack [--out FILE] [--subject-revision SHA]\n\nValidates local data only. It never fetches source URLs or remote schemas."
}

fn parse_args() -> Result<Config, String> {
    let mut args = env::args().skip(1);
    let mut schema = None;
    let mut input = None;
    let mut output = None;
    let mut kind = None;
    let mut subject_revision = None;

    while let Some(arg) = args.next() {
        let value = |args: &mut std::iter::Skip<std::env::Args>, flag: &str| {
            args.next()
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match arg.as_str() {
            "--schema" => schema = Some(PathBuf::from(value(&mut args, "--schema")?)),
            "--input" => input = Some(PathBuf::from(value(&mut args, "--input")?)),
            "--out" => output = Some(PathBuf::from(value(&mut args, "--out")?)),
            "--kind" => kind = Some(Kind::parse(&value(&mut args, "--kind")?)?),
            "--subject-revision" => {
                subject_revision = Some(value(&mut args, "--subject-revision")?)
            }
            "--help" | "-h" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument {other:?}\n{}", usage())),
        }
    }

    Ok(Config {
        schema: schema.ok_or_else(|| format!("missing --schema\n{}", usage()))?,
        input: input.ok_or_else(|| format!("missing --input\n{}", usage()))?,
        output,
        kind: kind.ok_or_else(|| format!("missing --kind\n{}", usage()))?,
        subject_revision,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

/// Reject remote and filesystem reference targets before asking the validator
/// library to compile a schema. Local fragment references remain permitted.
fn find_external_schema_refs(value: &Value, path: &str, errors: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                if matches!(key.as_str(), "$ref" | "$dynamicRef" | "$recursiveRef") {
                    match child.as_str() {
                        Some(target) if target.starts_with('#') => {}
                        Some(target) => errors.push(format!(
                            "{path}/{key}: non-local schema reference {target:?} is refused"
                        )),
                        None => {
                            errors.push(format!("{path}/{key}: schema reference must be a string"))
                        }
                    }
                }
                let next = format!("{path}/{}", key.replace('~', "~0").replace('/', "~1"));
                find_external_schema_refs(child, &next, errors);
            }
        }
        Value::Array(values) => {
            for (index, child) in values.iter().enumerate() {
                find_external_schema_refs(child, &format!("{path}/{index}"), errors);
            }
        }
        _ => {}
    }
}

fn string_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn string_array<'a>(value: &'a Value, key: &str) -> Vec<&'a str> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default()
}

fn validate_graph_contract(graph: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    let sources = graph
        .get("source_records")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let nodes = graph
        .get("nodes")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let edges = graph
        .get("edges")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let synthetic_only = graph
        .get("synthetic_only")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let mut source_ids = HashSet::new();
    for source in &sources {
        if let Some(id) = string_field(source, "source_id") {
            if !source_ids.insert(id.to_string()) {
                errors.push(format!("duplicate source_id {id:?}"));
            }
        }
        if synthetic_only && string_field(source, "source_kind") != Some("synthetic_control") {
            errors.push("synthetic_only graph contains a non-synthetic source record".into());
        }
    }

    let mut node_ids = HashSet::new();
    for node in &nodes {
        if let Some(id) = string_field(node, "node_id") {
            if !node_ids.insert(id.to_string()) {
                errors.push(format!("duplicate node_id {id:?}"));
            }
        }
        for source_id in string_array(node, "source_ids") {
            if !source_ids.contains(source_id) {
                errors.push(format!(
                    "node {:?} references unknown source_id {:?}",
                    string_field(node, "node_id").unwrap_or("<missing>"),
                    source_id
                ));
            }
        }
    }

    let mut edge_ids = HashSet::new();
    for edge in &edges {
        let edge_id = string_field(edge, "edge_id").unwrap_or("<missing>");
        if !edge_ids.insert(edge_id.to_string()) {
            errors.push(format!("duplicate edge_id {edge_id:?}"));
        }
        let from = string_field(edge, "source_node").unwrap_or("<missing>");
        let to = string_field(edge, "target_node").unwrap_or("<missing>");
        if !node_ids.contains(from) {
            errors.push(format!("edge {edge_id:?} has unknown source node {from:?}"));
        }
        if !node_ids.contains(to) {
            errors.push(format!("edge {edge_id:?} has unknown target node {to:?}"));
        }
        for source_id in string_array(edge, "source_ids") {
            if !source_ids.contains(source_id) {
                errors.push(format!(
                    "edge {edge_id:?} references unknown source_id {source_id:?}"
                ));
            }
        }

        let relation = string_field(edge, "relation").unwrap_or("");
        let basis = string_field(edge, "evidence_basis").unwrap_or("");
        let epistemic_status = string_field(edge, "epistemic_status").unwrap_or("");
        if synthetic_only && relation == "historically_influences" {
            errors.push(format!(
                "synthetic graph edge {edge_id:?} cannot assert historical influence"
            ));
        }
        if relation == "historically_influences" {
            let has_non_synthetic_source = string_array(edge, "source_ids").iter().any(|id| {
                sources.iter().any(|source| {
                    string_field(source, "source_id") == Some(id)
                        && matches!(
                            string_field(source, "source_kind"),
                            Some(
                                "primary_source"
                                    | "scholarly_analysis"
                                    | "community_authorized_source"
                            )
                        )
                })
            });
            if !has_non_synthetic_source
                || !matches!(basis, "direct_source" | "scholarly_argument")
                || !matches!(
                    epistemic_status,
                    "source_attested" | "scholarly_interpretation"
                )
            {
                errors.push(format!(
                    "edge {edge_id:?}: historically_influences requires admissible non-synthetic evidence, a direct_source/scholarly_argument basis, and an attested or scholarly status"
                ));
            }
        }
        if relation == "causes" && !matches!(basis, "direct_source" | "scholarly_argument") {
            errors.push(format!(
                "edge {edge_id:?}: causes requires direct source evidence or an identified scholarly argument; temporal order alone is insufficient"
            ));
        }
        if synthetic_only && !matches!(basis, "synthetic_assertion" | "explicit_inference") {
            errors.push(format!(
                "synthetic graph edge {edge_id:?} has non-synthetic evidence_basis {basis:?}"
            ));
        }
    }
    errors
}

fn record_identity(kind: Kind, value: &Value, line: usize) -> String {
    if kind == Kind::SourcePack {
        let pair_id = string_field(value, "pair_id").unwrap_or("");
        let state = string_field(value, "state").unwrap_or("");
        if !pair_id.is_empty() && !state.is_empty() {
            return format!("{pair_id}:{state}");
        }
    }
    let key = match kind {
        Kind::Graph => "graph_id",
        Kind::Case | Kind::Claim => "id",
        Kind::Pair => "pair_id",
        Kind::Source => "source_id",
        Kind::Annotation => "annotation_id",
        Kind::Run => "plan_id",
        Kind::PairResult => "result_id",
        Kind::SourcePack => "pair_id",
    };
    string_field(value, key)
        .map(str::to_string)
        .unwrap_or_else(|| format!("line-{line}"))
}

fn collect_unique_object_ids(
    value: &Value,
    array_key: &str,
    id_key: &str,
    context: &str,
    errors: &mut Vec<String>,
) -> HashSet<String> {
    let mut ids = HashSet::new();
    let items = value
        .get(array_key)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for item in items {
        if let Some(id) = string_field(&item, id_key) {
            if !ids.insert(id.to_string()) {
                errors.push(format!("duplicate {context} identifier {id:?}"));
            }
        }
    }
    ids
}

fn validate_pair_contract(pair: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    let dimension = string_field(pair, "dimension").unwrap_or("");
    let expected = string_field(pair, "expected_relation").unwrap_or("");
    let split = string_field(pair, "split").unwrap_or("");
    let synthetic_only = pair
        .get("synthetic_only")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let declared_families = string_array(pair, "source_family_ids");
    if declared_families.is_empty() {
        errors.push("pair must declare at least one source_family_id cluster".into());
    }
    if declared_families.iter().collect::<HashSet<_>>().len() != declared_families.len() {
        errors.push("pair repeats source_family_id".into());
    }
    if synthetic_only != (split == "synthetic") {
        errors.push("pair split must be synthetic exactly when synthetic_only=true".into());
    }
    let baseline = pair.get("baseline").unwrap_or(&Value::Null);
    let mutated = pair.get("mutated").unwrap_or(&Value::Null);
    let intervention = pair.get("intervention").unwrap_or(&Value::Null);
    let field = string_field(intervention, "field").unwrap_or("");
    let before_value = string_field(intervention, "before_value").unwrap_or("");
    let after_value = string_field(intervention, "after_value").unwrap_or("");

    let baseline_evidence_count = baseline
        .get("evidence_pack")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    let mutated_evidence_count = mutated
        .get("evidence_pack")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);

    if before_value == after_value {
        errors.push("pair intervention before_value and after_value must differ".into());
    }

    let state_field = match field {
        "context_id" | "source_id" | "variant_id" | "tradition_scope" | "rights_status"
        | "access_decision" | "alias" | "source_text" => Some(field),
        "relation" => Some("proposed_relation"),
        _ => None,
    };
    if let Some(state_field) = state_field {
        if string_field(baseline, state_field) != Some(before_value) {
            errors.push(format!(
                "intervention before_value does not match baseline {state_field}"
            ));
        }
        if string_field(mutated, state_field) != Some(after_value) {
            errors.push(format!(
                "intervention after_value does not match mutated {state_field}"
            ));
        }
    }

    if matches!(field, "evidence_pack" | "counterevidence") {
        let parse_count = |raw: &str| raw.parse::<usize>().ok();
        if parse_count(before_value) != Some(baseline_evidence_count) {
            errors
                .push("intervention before_value must equal baseline evidence_pack length".into());
        }
        if parse_count(after_value) != Some(mutated_evidence_count) {
            errors.push("intervention after_value must equal mutated evidence_pack length".into());
        }
    }

    if matches!(
        expected,
        "local_change" | "abstain_or_revise" | "reject_unsupported_relation" | "deny_access"
    ) && pair
        .get("acceptance")
        .and_then(|a| a.get("must_change"))
        .and_then(Value::as_array)
        .is_none_or(Vec::is_empty)
    {
        errors.push("non-invariant pair requires at least one must_change criterion".into());
    }

    match dimension {
        "context_swap" => {
            if string_field(baseline, "context_id") == string_field(mutated, "context_id") {
                errors.push("context_swap must change context_id".into());
            }
            if field != "context_id" || expected != "local_change" {
                errors.push(
                    "context_swap requires context_id intervention and local_change expectation"
                        .into(),
                );
            }
        }
        "source_swap" => {
            if string_field(baseline, "source_id") == string_field(mutated, "source_id") {
                errors.push("source_swap must change source_id".into());
            }
            if field != "source_id" || !matches!(expected, "local_change" | "abstain_or_revise") {
                errors.push(
                    "source_swap requires source_id intervention and a change-or-abstain expectation".into(),
                );
            }
            let evidence_has_source = |state: &Value| {
                state
                    .get("evidence_pack")
                    .and_then(Value::as_array)
                    .is_some_and(|items| {
                        items.iter().any(|item| {
                            string_field(item, "source_id") == string_field(state, "source_id")
                        })
                    })
            };
            if !evidence_has_source(baseline) || !evidence_has_source(mutated) {
                errors.push(
                    "source_swap evidence must cite the current source identity in both states"
                        .into(),
                );
            }
        }
        "evidence_removal" => {
            if baseline_evidence_count <= mutated_evidence_count {
                errors.push("evidence_removal must reduce the evidence_pack size".into());
            }
            if field != "evidence_pack" || expected != "abstain_or_revise" {
                errors.push(
                    "evidence_removal requires evidence_pack intervention and abstain_or_revise expectation".into(),
                );
            }
        }
        "relation_mutation" => {
            if field != "relation"
                || expected != "reject_unsupported_relation"
                || before_value != "resembles"
                || after_value != "historically_influences"
                || string_field(baseline, "proposed_relation") != Some(before_value)
                || string_field(mutated, "proposed_relation") != Some(after_value)
            {
                errors.push(
                    "relation_mutation must bind the proposed relation and reject unsupported promotion from resemblance to historical influence"
                        .into(),
                );
            }
            if baseline.get("evidence_pack") != mutated.get("evidence_pack") {
                errors.push("relation_mutation must keep the evidence pack identical".into());
            }
        }
        "variant_isolation" => {
            if string_field(baseline, "variant_id") == string_field(mutated, "variant_id") {
                errors.push("variant_isolation must change variant_id".into());
            }
            if field != "variant_id" || expected != "local_change" {
                errors.push(
                    "variant_isolation requires variant_id intervention and local_change expectation".into(),
                );
            }
        }
        "tradition_transfer" => {
            if string_field(baseline, "tradition_scope") == string_field(mutated, "tradition_scope")
            {
                errors.push("tradition_transfer must change tradition_scope".into());
            }
            if field != "tradition_scope" || expected != "local_change" {
                errors.push(
                    "tradition_transfer requires tradition_scope intervention and local_change expectation".into(),
                );
            }
        }
        "counterevidence_addition" => {
            if mutated_evidence_count <= baseline_evidence_count {
                errors
                    .push("counterevidence_addition must add at least one evidence record".into());
            }
            if field != "counterevidence" || expected != "local_change" {
                errors.push(
                    "counterevidence_addition requires counterevidence intervention and local_change expectation".into(),
                );
            }
        }
        "rights_revocation" => {
            let denied = matches!(
                string_field(mutated, "access_decision"),
                Some("deny" | "require_permission" | "metadata_only")
            );
            if !denied
                || string_field(baseline, "access_decision")
                    == string_field(mutated, "access_decision")
            {
                errors.push(
                    "rights_revocation must change access_decision to deny or a more restrictive state"
                        .into(),
                );
            }
            if string_field(baseline, "rights_status") != string_field(mutated, "rights_status") {
                errors.push(
                    "rights_revocation must preserve the source's underlying rights_status".into(),
                );
            }
            if field != "access_decision" || expected != "deny_access" {
                errors.push(
                    "rights_revocation requires access_decision intervention and deny_access expectation"
                        .into(),
                );
            }
        }
        "alias_equivalence" => {
            if string_field(baseline, "source_id") != string_field(mutated, "source_id") {
                errors.push("alias_equivalence must preserve source identity".into());
            }
            if field != "alias" || expected != "invariant" {
                errors.push(
                    "alias_equivalence requires alias intervention and invariant expectation"
                        .into(),
                );
            }
            if baseline.get("evidence_pack") != mutated.get("evidence_pack") {
                errors.push("alias_equivalence must preserve the evidence pack".into());
            }
            for key in [
                "context_id",
                "tradition_scope",
                "source_id",
                "variant_id",
                "rights_status",
            ] {
                if baseline.get(key) != mutated.get(key) {
                    errors.push(format!("alias_equivalence unexpectedly changed {key}"));
                }
            }
        }
        "prompt_injection" => {
            if field != "source_text" || expected != "invariant" {
                errors.push(
                    "prompt_injection must preserve governing behavior under source-text injection"
                        .into(),
                );
            }
            if baseline.get("evidence_pack") != mutated.get("evidence_pack") {
                errors.push("prompt_injection must keep the evidence record unchanged".into());
            }
            for key in [
                "context_id",
                "tradition_scope",
                "source_id",
                "variant_id",
                "rights_status",
            ] {
                if baseline.get(key) != mutated.get(key) {
                    errors.push(format!("prompt_injection unexpectedly changed {key}"));
                }
            }
        }
        _ => errors.push(format!("unknown pair dimension {dimension:?}")),
    }

    for (state_name, state) in [("baseline", baseline), ("mutated", mutated)] {
        let mut evidence_ids = HashSet::new();
        let state_rights = string_field(state, "rights_status").unwrap_or("");
        // Top-level synthetic_only describes fixture provenance; state rights_status
        // describes the tested access state and can change in a rights_revocation pair.
        if !synthetic_only
            && (state_rights == "synthetic_only"
                || (matches!(split, "test" | "holdout")
                    && matches!(state_rights, "unknown" | "restricted")))
        {
            errors.push(format!(
                "{state_name} rights_status is not admissible for split {split:?}"
            ));
        }
        if let Some(items) = state.get("evidence_pack").and_then(Value::as_array) {
            for item in items {
                let kind = string_field(item, "kind").unwrap_or("");
                if synthetic_only && kind != "synthetic_control" {
                    errors.push(format!(
                        "{state_name} synthetic pair contains non-synthetic evidence kind {kind:?}"
                    ));
                }
                if !synthetic_only && matches!(kind, "synthetic_control" | "metadata_only") {
                    errors.push(format!(
                        "{state_name} non-synthetic pair contains inadmissible evidence kind {kind:?}"
                    ));
                }
                if let Some(id) = string_field(item, "evidence_id") {
                    if !evidence_ids.insert(id.to_string()) {
                        errors.push(format!(
                            "{state_name} contains duplicate evidence_id {id:?}"
                        ));
                    }
                }
            }
        }
    }

    errors
}

fn validate_source_contract(value: &Value, seen_ids: &mut HashSet<String>) -> Vec<String> {
    let mut errors = Vec::new();
    let source_id = string_field(value, "source_id").unwrap_or("");
    let source_family_id = string_field(value, "source_family_id").unwrap_or("");
    let split = string_field(value, "split").unwrap_or("");
    let rights_status = string_field(value, "rights_status").unwrap_or("");
    let content_access = string_field(value, "content_access").unwrap_or("");
    let rights_review = string_field(value, "rights_review").unwrap_or("");

    if source_id.is_empty() {
        errors.push("source_id must be a non-empty string".into());
    }
    if source_family_id.is_empty() {
        errors.push("source_family_id must be a non-empty string".into());
    }

    let splits = ["development", "validation", "test", "holdout"];
    if splits.iter().any(|candidate| {
        *candidate != split
            && seen_ids.contains(&format!(
                "source-family-split:{source_family_id}:{candidate}"
            ))
    }) {
        errors.push(format!(
            "source family {source_family_id:?} spans multiple evaluation splits"
        ));
    } else if !source_family_id.is_empty() && !split.is_empty() {
        // Multiple source IDs from one family may share a split. The marker is
        // intentionally idempotent; a family crossing splits is never allowed.
        seen_ids.insert(format!("source-family-split:{source_family_id}:{split}"));
    }

    if rights_status == "unknown" && content_access != "metadata_only" {
        errors.push("rights_status=unknown requires content_access=metadata_only".into());
    }
    if rights_review == "approved_limited_use" && matches!(rights_status, "unknown" | "restricted")
    {
        errors.push(
            "rights review cannot approve limited use while rights are unknown or restricted"
                .into(),
        );
    }
    if content_access == "verified_open_content"
        && !matches!(
            rights_status,
            "public_domain_verified" | "open_license_verified"
        )
    {
        errors.push(
            "verified_open_content requires public-domain or open-license verification".into(),
        );
    }
    if content_access == "verified_open_content" && rights_review != "approved_limited_use" {
        errors.push("verified_open_content requires rights_review=approved_limited_use".into());
    }
    if matches!(
        rights_status,
        "permission_required" | "restricted" | "community_governed"
    ) && content_access == "verified_open_content"
    {
        errors.push("restricted or permissioned source cannot be marked verified open".into());
    }

    errors
}

fn validate_annotation_contract(annotation: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    let answerability = string_field(annotation, "answerability").unwrap_or("");
    let review_status = string_field(annotation, "review_status").unwrap_or("");
    let annotation_role = string_field(annotation, "annotation_role").unwrap_or("");
    let mut claim_ids = HashSet::new();
    let claims = annotation
        .get("claim_annotations")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    for claim in &claims {
        if let Some(claim_id) = string_field(claim, "claim_id") {
            if !claim_ids.insert(claim_id.to_string()) {
                errors.push(format!("duplicate claim annotation ID {claim_id:?}"));
            }
        }
        let support_label = string_field(claim, "support_label").unwrap_or("");
        let epistemic_lane = string_field(claim, "epistemic_lane").unwrap_or("");
        let severity = string_field(claim, "severity").unwrap_or("");
        let failure_tags = string_array(claim, "failure_tags");
        let evidence_refs = claim
            .get("evidence_refs")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        if matches!(
            epistemic_lane,
            "source_attested" | "scholarly_interpretation"
        ) && matches!(support_label, "supported" | "partially_supported")
            && evidence_refs.is_empty()
        {
            errors.push(format!(
                "claim {:?}: source-grounded positive label requires at least one evidence reference",
                string_field(claim, "claim_id").unwrap_or("<missing>")
            ));
        }
        if severity == "release_blocker"
            && (failure_tags.is_empty() || failure_tags.iter().any(|tag| tag == "none"))
        {
            errors.push(format!(
                "claim {:?}: release_blocker requires explicit non-none failure tags",
                string_field(claim, "claim_id").unwrap_or("<missing>")
            ));
        }
        if failure_tags.iter().any(|tag| tag == "rights_violation") && severity != "release_blocker"
        {
            errors.push(format!(
                "claim {:?}: rights_violation must be a release_blocker",
                string_field(claim, "claim_id").unwrap_or("<missing>")
            ));
        }
        if failure_tags.iter().any(|tag| tag == "none") && failure_tags.len() > 1 {
            errors.push(format!(
                "claim {:?}: failure tag none cannot be combined with an error tag",
                string_field(claim, "claim_id").unwrap_or("<missing>")
            ));
        }
    }

    let mut alternative_ids = HashSet::new();
    let alternatives = annotation
        .get("valid_alternatives")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for alternative in &alternatives {
        if let Some(alternative_id) = string_field(alternative, "alternative_id") {
            if !alternative_ids.insert(alternative_id.to_string()) {
                errors.push(format!("duplicate alternative ID {alternative_id:?}"));
            }
        }
        if string_array(alternative, "source_ids").is_empty() {
            errors.push(format!(
                "alternative {:?}: every accepted interpretation must cite at least one source ID",
                string_field(alternative, "alternative_id").unwrap_or("<missing>")
            ));
        }
    }

    if matches!(
        annotation_role,
        "independent_reviewer" | "subject_expert" | "calibration_reviewer"
    ) && review_status == "independent"
        && !string_array(annotation, "prior_annotation_ids").is_empty()
    {
        errors.push(
            "independent annotations must not cite prior annotation IDs; preserve blind review"
                .into(),
        );
    }
    if review_status == "adjudicated" && string_array(annotation, "prior_annotation_ids").len() < 2
    {
        errors.push(
            "adjudicated annotations must retain at least two independent annotation IDs".into(),
        );
    }
    if answerability == "rights_prohibited"
        && string_field(annotation, "abstention_assessment") == Some("appropriate")
        && (!claims.is_empty() || !alternatives.is_empty())
    {
        errors.push(
            "an appropriate rights-prohibited abstention must not expose claim content or alternatives"
                .into(),
        );
    }

    let family_ids = string_array(annotation, "source_family_ids");
    if family_ids.iter().collect::<HashSet<_>>().len() != family_ids.len() {
        errors.push("duplicate source_family_id in annotation scope".into());
    }

    errors
}

fn validate_run_contract(plan: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    let phase = string_field(plan, "phase").unwrap_or("");
    let synthetic_only = plan
        .get("synthetic_only")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let benchmark = plan.get("benchmark").unwrap_or(&Value::Null);
    let split = string_field(benchmark, "split").unwrap_or("");
    let source_family_disjoint = benchmark
        .get("source_family_disjoint")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let rights_review = string_field(benchmark, "rights_review_status").unwrap_or("");
    let preregistration = plan.get("preregistration").unwrap_or(&Value::Null);
    let held_out_access = string_field(preregistration, "held_out_access").unwrap_or("");
    let systems = plan
        .get("systems")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut system_ids = HashSet::new();
    let mut families = HashSet::new();

    if systems.len() < 2 {
        errors.push("comparison plan must define at least two systems".into());
    }
    for system in &systems {
        let system_id = string_field(system, "system_id").unwrap_or("");
        if system_id.is_empty() {
            errors.push("system_id must be a non-empty string".into());
        } else if !system_ids.insert(system_id.to_string()) {
            errors.push(format!("duplicate comparison system_id {system_id:?}"));
        }
        let family = string_field(system, "baseline_family").unwrap_or("");
        if !family.is_empty() {
            families.insert(family.to_string());
        }
        for key in [
            "implementation",
            "revision",
            "prompt_sha256",
            "retriever_revision",
        ] {
            if string_field(system, key).is_none_or(str::is_empty) {
                errors.push(format!(
                    "comparison system {system_id:?} requires non-empty {key}"
                ));
            }
        }
    }
    if families.len() < 2 {
        errors.push("comparison must include at least two distinct baseline families".into());
    }

    let declared_system_ids: HashSet<&str> = plan
        .get("systems")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|system| string_field(system, "system_id"))
        .collect();
    let candidate_system_id = string_field(plan, "candidate_system_id").unwrap_or("");
    if candidate_system_id.is_empty() || !declared_system_ids.contains(candidate_system_id) {
        errors.push(format!(
            "candidate_system_id {candidate_system_id:?} must identify a declared comparison system"
        ));
    }

    match phase {
        "synthetic_control" => {
            if !synthetic_only || split != "synthetic" {
                errors.push(
                    "synthetic_control requires synthetic_only=true and benchmark split=synthetic"
                        .into(),
                );
            }
            if held_out_access != "not_applicable_synthetic" {
                errors.push(
                    "synthetic_control requires held_out_access=not_applicable_synthetic".into(),
                );
            }
        }
        "development_pilot" => {
            if synthetic_only || !matches!(split, "development" | "validation") {
                errors.push(
                    "development_pilot requires non-synthetic data in development/validation split"
                        .into(),
                );
            }
        }
        "preregistered_held_out" => {
            if synthetic_only {
                errors.push("preregistered_held_out cannot be synthetic".into());
            }
            if !matches!(split, "test" | "holdout") {
                errors
                    .push("preregistered_held_out requires benchmark split=test or holdout".into());
            }
            if !source_family_disjoint {
                errors.push("preregistered_held_out requires source-family-disjoint data".into());
            }
            if rights_review != "approved" {
                errors.push("preregistered_held_out requires rights_review_status=approved".into());
            }
            if !matches!(
                held_out_access,
                "sealed_until_freeze" | "independent_evaluator"
            ) {
                errors.push(
                    "preregistered_held_out requires sealed or independent-evaluator access".into(),
                );
            }
        }
        _ => errors.push(format!("unknown comparison phase {phase:?}")),
    }

    let benchmark_digests_ok = [
        "case_manifest_sha256",
        "source_catalog_sha256",
        "source_pack_sha256",
    ]
    .iter()
    .all(|key| {
        string_field(benchmark, key).is_some_and(|digest| {
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        })
    });
    if !benchmark_digests_ok {
        errors.push("comparison plan requires lowercase SHA-256 digests for case manifest, source catalog, and source pack".into());
    }

    errors
}

fn validate_annotation_catalog_refs(annotation: &Value) -> Vec<String> {
    if string_field(annotation, "annotation_role") == Some("synthetic_fixture") {
        return Vec::new();
    }

    let mut errors = Vec::new();
    let mut source_ids = HashSet::new();
    let mut source_family_ids = HashSet::new();
    let mut source_rights: std::collections::HashMap<String, (String, String, String)> =
        std::collections::HashMap::new();
    for (index, raw_line) in include_str!("../../../benchmarks/source-catalog/v0/sources.jsonl")
        .lines()
        .enumerate()
    {
        match serde_json::from_str::<Value>(raw_line) {
            Ok(source) => {
                if let Some(id) = string_field(&source, "source_id") {
                    source_ids.insert(id.to_string());
                }
                if let Some(id) = string_field(&source, "source_family_id") {
                    source_family_ids.insert(id.to_string());
                }
                if let Some(id) = string_field(&source, "source_id") {
                    source_rights.insert(
                        id.to_string(),
                        (
                            string_field(&source, "rights_status")
                                .unwrap_or("")
                                .to_string(),
                            string_field(&source, "content_access")
                                .unwrap_or("")
                                .to_string(),
                            string_field(&source, "rights_review")
                                .unwrap_or("")
                                .to_string(),
                        ),
                    );
                }
            }
            Err(error) => errors.push(format!(
                "bundled source catalog line {} is invalid JSON: {error}",
                index + 1
            )),
        }
    }

    let annotation_id = string_field(annotation, "annotation_id").unwrap_or("<unknown>");
    let claims = annotation
        .get("claim_annotations")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for claim in claims {
        let claim_id = string_field(&claim, "claim_id").unwrap_or("<unknown>");
        if let Some(refs) = claim.get("evidence_refs").and_then(Value::as_array) {
            for evidence_ref in refs {
                let source_id = string_field(evidence_ref, "source_id").unwrap_or("");
                if !source_ids.contains(source_id) {
                    errors.push(format!(
                        "annotation {annotation_id:?} claim {claim_id:?} references source ID {source_id:?} absent from catalog"
                    ));
                } else if let Some((rights_status, content_access, rights_review)) =
                    source_rights.get(source_id)
                {
                    if !matches!(
                        content_access.as_str(),
                        "verified_open_content" | "permissioned_full_text"
                    ) {
                        errors.push(format!(
                            "annotation {annotation_id:?} claim {claim_id:?} source {source_id:?} is metadata-only and cannot support content-grounded adjudication"
                        ));
                    }
                    if matches!(rights_status.as_str(), "unknown" | "restricted")
                        || rights_review != "approved_limited_use"
                    {
                        errors.push(format!(
                            "annotation {annotation_id:?} claim {claim_id:?} source {source_id:?} lacks approved rights review for content-grounded adjudication"
                        ));
                    }
                }
            }
        }
    }

    let alternatives = annotation
        .get("valid_alternatives")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for alternative in alternatives {
        let alternative_id = string_field(&alternative, "alternative_id").unwrap_or("<unknown>");
        for source_id in string_array(&alternative, "source_ids") {
            if !source_ids.contains(source_id) {
                errors.push(format!(
                    "annotation {annotation_id:?} alternative {alternative_id:?} references source ID {source_id:?} absent from catalog"
                ));
            } else if let Some((rights_status, content_access, rights_review)) =
                source_rights.get(source_id)
            {
                if !matches!(
                    content_access.as_str(),
                    "verified_open_content" | "permissioned_full_text"
                ) {
                    errors.push(format!(
                        "annotation {annotation_id:?} alternative {alternative_id:?} source {source_id:?} is metadata-only and cannot support content-grounded adjudication"
                    ));
                }
                if matches!(rights_status.as_str(), "unknown" | "restricted")
                    || rights_review != "approved_limited_use"
                {
                    errors.push(format!(
                        "annotation {annotation_id:?} alternative {alternative_id:?} source {source_id:?} lacks approved rights review for content-grounded adjudication"
                    ));
                }
            }
        }
    }

    for family_id in string_array(annotation, "source_family_ids") {
        if !source_family_ids.contains(family_id) {
            errors.push(format!(
                "annotation {annotation_id:?} references source_family_id {family_id:?} absent from catalog"
            ));
        }
    }

    errors
}

fn validate_source_pack_contract(record: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    let pair_id = string_field(record, "pair_id").unwrap_or("<missing>");
    let split = string_field(record, "split").unwrap_or("");
    let synthetic_only = record
        .get("synthetic_only")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let access_decision = string_field(record, "access_decision").unwrap_or("");
    let rights_status = string_field(record, "rights_status").unwrap_or("");
    let evidence = record
        .get("evidence_pack")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    if synthetic_only != (split == "synthetic") {
        errors.push(format!(
            "source-pack snapshot {pair_id:?} must be synthetic exactly when split=synthetic"
        ));
    }
    if !matches!(string_field(record, "state"), Some("baseline" | "mutated")) {
        errors.push(format!(
            "source-pack snapshot {pair_id:?} must identify baseline or mutated state"
        ));
    }
    if !matches!(
        access_decision,
        "synthetic_only" | "allow" | "deny" | "require_permission" | "metadata_only"
    ) {
        errors.push(format!(
            "source-pack snapshot {pair_id:?} has unsupported access_decision {access_decision:?}"
        ));
    }
    if access_decision != "allow"
        && !(synthetic_only && access_decision == "synthetic_only")
        && !evidence.is_empty()
    {
        errors.push(format!(
            "source-pack snapshot {pair_id:?} exposes evidence despite access_decision={access_decision:?}"
        ));
    }
    if synthetic_only && rights_status != "synthetic_only" {
        errors.push(format!(
            "source-pack snapshot {pair_id:?} must use synthetic_only rights_status"
        ));
    }
    if !synthetic_only && !evidence.is_empty() && matches!(rights_status, "unknown" | "restricted")
    {
        errors.push(format!(
            "source-pack snapshot {pair_id:?} uses content with inadmissible rights_status {rights_status:?}"
        ));
    }

    for item in &evidence {
        let kind = string_field(item, "kind").unwrap_or("");
        if synthetic_only && kind != "synthetic_control" {
            errors.push(format!(
                "source-pack snapshot {pair_id:?} synthetic evidence kind must be synthetic_control"
            ));
        }
        if !synthetic_only && matches!(kind, "synthetic_control" | "metadata_only") {
            errors.push(format!(
                "source-pack snapshot {pair_id:?} uses inadmissible real evidence kind {kind:?}"
            ));
        }
    }

    collect_unique_object_ids(
        record,
        "evidence_pack",
        "evidence_id",
        "source-pack evidence",
        &mut errors,
    );
    errors
}

fn validate_pair_catalog_refs(pair: &Value) -> Vec<String> {
    if pair.get("synthetic_only").and_then(Value::as_bool) == Some(true) {
        return Vec::new();
    }

    let mut errors = Vec::new();
    let pair_id = string_field(pair, "pair_id").unwrap_or("<unknown>");
    let split = string_field(pair, "split").unwrap_or("");
    let declared_families: HashSet<String> = string_array(pair, "source_family_ids")
        .into_iter()
        .map(str::to_string)
        .collect();
    let mut observed_families = HashSet::<String>::new();
    let mut catalog = std::collections::HashMap::<String, Value>::new();

    for (index, raw_line) in include_str!("../../../benchmarks/source-catalog/v0/sources.jsonl")
        .lines()
        .enumerate()
    {
        match serde_json::from_str::<Value>(raw_line) {
            Ok(source) => {
                if let Some(source_id) = string_field(&source, "source_id") {
                    catalog.insert(source_id.to_string(), source);
                }
            }
            Err(error) => errors.push(format!(
                "bundled source catalog line {} is invalid JSON: {error}",
                index + 1
            )),
        }
    }

    for (state_name, state) in [
        ("baseline", pair.get("baseline")),
        ("mutated", pair.get("mutated")),
    ] {
        let Some(state) = state else {
            errors.push(format!("pair {pair_id:?} has no {state_name} state"));
            continue;
        };
        let state_source_id = string_field(state, "source_id").unwrap_or("");
        let access_decision = string_field(state, "access_decision").unwrap_or("");
        if !matches!(
            access_decision,
            "allow" | "deny" | "require_permission" | "metadata_only"
        ) {
            errors.push(format!(
                "pair {pair_id:?} {state_name} has unsupported access_decision {access_decision:?}"
            ));
        }
        let evidence_items = state
            .get("evidence_pack")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if access_decision != "allow" && !evidence_items.is_empty() {
            errors.push(format!(
                "pair {pair_id:?} {state_name} has a non-allow access_decision but exposes an evidence pack"
            ));
        }
        let mut referenced_ids = HashSet::new();
        if !state_source_id.is_empty() {
            referenced_ids.insert(state_source_id.to_string());
        }
        for item in &evidence_items {
            if let Some(source_id) = string_field(item, "source_id") {
                referenced_ids.insert(source_id.to_string());
            } else {
                errors.push(format!(
                    "pair {pair_id:?} {state_name} evidence item has no source_id"
                ));
            }
            let kind = string_field(item, "kind").unwrap_or("");
            if matches!(kind, "synthetic_control" | "metadata_only") {
                errors.push(format!(
                    "pair {pair_id:?} {state_name} uses inadmissible real-evidence kind {kind:?}"
                ));
            }
        }

        for source_id in referenced_ids {
            let Some(source) = catalog.get(&source_id) else {
                errors.push(format!(
                    "pair {pair_id:?} {state_name} references source ID {source_id:?} absent from catalog"
                ));
                continue;
            };
            let catalog_split = string_field(source, "split").unwrap_or("");
            if let Some(family_id) = string_field(source, "source_family_id") {
                observed_families.insert(family_id.to_string());
            }
            let rights_status = string_field(source, "rights_status").unwrap_or("");
            let content_access = string_field(source, "content_access").unwrap_or("");
            let rights_review = string_field(source, "rights_review").unwrap_or("");
            if catalog_split != split {
                errors.push(format!(
                    "pair {pair_id:?} source {source_id:?} is in split {catalog_split:?}, not pair split {split:?}"
                ));
            }
            if state_source_id == source_id.as_str()
                && string_field(state, "rights_status") != Some(rights_status)
            {
                errors.push(format!(
                    "pair {pair_id:?} {state_name} rights_status differs from catalog for source {source_id:?}"
                ));
            }
            if !matches!(
                content_access,
                "verified_open_content" | "permissioned_full_text"
            ) {
                errors.push(format!(
                    "pair {pair_id:?} source {source_id:?} is metadata-only and cannot provide pair evidence"
                ));
            }
            if matches!(rights_status, "unknown" | "restricted")
                || rights_review != "approved_limited_use"
            {
                errors.push(format!(
                    "pair {pair_id:?} source {source_id:?} lacks approved rights for pair evidence"
                ));
            }
        }
    }

    if declared_families != observed_families {
        errors.push(format!(
            "pair {pair_id:?} declared source_family_ids do not match the source families referenced by its states/evidence"
        ));
    }

    errors
}

fn validate_pair_result_contract(result: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    let result_id = string_field(result, "result_id").unwrap_or("<missing>");
    if string_field(result, "plan_id").is_none_or(str::is_empty)
        || string_field(result, "system_id").is_none_or(str::is_empty)
        || string_field(result, "pair_id").is_none_or(str::is_empty)
    {
        errors.push(format!(
            "pair result {result_id:?} requires non-empty plan_id, system_id, and pair_id"
        ));
    }
    let allowed_relations = [
        "invariant",
        "local_change",
        "abstain_or_revise",
        "reject_unsupported_relation",
        "deny_access",
        "incorrect",
        "ambiguous",
    ];
    if !string_field(result, "observed_relation")
        .is_some_and(|relation| allowed_relations.contains(&relation))
    {
        errors.push(format!(
            "pair result {result_id:?} has unsupported observed_relation"
        ));
    }
    let forbidden = string_array(result, "forbidden_behaviors");
    if forbidden.iter().collect::<HashSet<_>>().len() != forbidden.len() {
        errors.push(format!(
            "pair result {result_id:?} repeats a forbidden_behavior tag"
        ));
    }
    if string_field(result, "notes").is_none_or(str::is_empty) {
        errors.push(format!(
            "pair result {result_id:?} requires non-empty notes"
        ));
    }
    if !matches!(
        string_field(result, "adjudication_status"),
        Some("synthetic_only" | "independent_review" | "blind_human_adjudicated")
    ) {
        errors.push(format!(
            "pair result {result_id:?} has unsupported adjudication_status"
        ));
    }
    errors
}

fn validate_domain_contract(
    kind: Kind,
    value: &Value,
    seen_ids: &mut HashSet<String>,
) -> Vec<String> {
    let mut errors = Vec::new();
    let id = record_identity(kind, value, 0);
    if !seen_ids.insert(id.clone()) {
        errors.push(format!("duplicate record ID {id:?}"));
    }

    match kind {
        Kind::Case => {
            collect_unique_object_ids(value, "source_refs", "source_id", "source", &mut errors);
            collect_unique_object_ids(
                value,
                "evidence_pack",
                "evidence_id",
                "evidence",
                &mut errors,
            );
        }
        Kind::Claim => {
            let source_ids = collect_unique_object_ids(
                value,
                "sources",
                "source_id",
                "claim source",
                &mut errors,
            );
            let counterclaims = value
                .get("counterclaims")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            for counterclaim in counterclaims {
                for source_id in string_array(&counterclaim, "source_ids") {
                    if !source_ids.contains(source_id) {
                        errors.push(format!(
                            "counterclaim references unknown claim source_id {source_id:?}"
                        ));
                    }
                }
            }
        }
        Kind::Graph => errors.extend(validate_graph_contract(value)),
        Kind::Pair => {
            errors.extend(validate_pair_contract(value));
            errors.extend(validate_pair_catalog_refs(value));
        }
        Kind::Source => errors.extend(validate_source_contract(value, seen_ids)),
        Kind::Annotation => {
            errors.extend(validate_annotation_contract(value));
            errors.extend(validate_annotation_catalog_refs(value));
        }
        Kind::Run => errors.extend(validate_run_contract(value)),
        Kind::PairResult => errors.extend(validate_pair_result_contract(value)),
        Kind::SourcePack => errors.extend(validate_source_pack_contract(value)),
    }
    errors
}

fn run(config: Config) -> Result<(Report, bool), String> {
    let schema_bytes = fs::read(&config.schema)
        .map_err(|e| format!("cannot read schema {}: {e}", config.schema.display()))?;
    let input_bytes = fs::read(&config.input)
        .map_err(|e| format!("cannot read input {}: {e}", config.input.display()))?;
    let schema_hash = sha256_hex(&schema_bytes);
    let input_hash = sha256_hex(&input_bytes);
    let lock_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock");
    let lock_bytes = fs::read(&lock_path).map_err(|e| {
        format!(
            "cannot read dependency lockfile {}: {e}; generate and review Cargo.lock before qualifying this harness",
            lock_path.display()
        )
    })?;
    let lock_hash = sha256_hex(&lock_bytes);

    let schema: Value = serde_json::from_slice(&schema_bytes)
        .map_err(|e| format!("schema is not valid JSON: {e}"))?;
    let dialect = string_field(&schema, "$schema").map(str::to_string);
    if dialect.as_deref() != Some(DRAFT_2020_12) {
        return Err(format!(
            "schema must declare JSON Schema Draft 2020-12 exactly; found {dialect:?}"
        ));
    }

    let mut ref_errors = Vec::new();
    find_external_schema_refs(&schema, "$", &mut ref_errors);
    if !ref_errors.is_empty() {
        return Err(format!(
            "schema reference policy failed: {}",
            ref_errors.join("; ")
        ));
    }

    let meta_validator = jsonschema::meta::validator_for(&schema)
        .map_err(|e| format!("cannot build Draft 2020-12 meta-validator: {e}"))?;
    let meta_valid = meta_validator.evaluate(&schema).flag().valid;
    if !meta_valid {
        return Err("schema does not validate against its detected JSON Schema meta-schema".into());
    }

    let validator = jsonschema::draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|e| format!("cannot compile case schema: {e}"))?;

    let input_text =
        std::str::from_utf8(&input_bytes).map_err(|e| format!("input is not valid UTF-8: {e}"))?;
    let mut seen_ids = HashSet::new();
    let mut results = Vec::new();

    for (zero_index, raw_line) in input_text.lines().enumerate() {
        let line = zero_index + 1;
        if raw_line.trim().is_empty() {
            results.push(RecordResult {
                line,
                record_id: format!("line-{line}"),
                schema_valid: false,
                domain_contract_valid: false,
                errors: vec!["blank JSONL line is not permitted".into()],
            });
            continue;
        }
        let value: Value = match serde_json::from_str(raw_line) {
            Ok(value) => value,
            Err(error) => {
                results.push(RecordResult {
                    line,
                    record_id: format!("line-{line}"),
                    schema_valid: false,
                    domain_contract_valid: false,
                    errors: vec![format!("invalid JSON: {error}")],
                });
                continue;
            }
        };

        let id = record_identity(config.kind, &value, line);
        let mut errors: Vec<String> = validator
            .iter_errors(&value)
            .map(|error| error.to_string())
            .collect();
        let schema_valid = errors.is_empty();
        let domain_errors = validate_domain_contract(config.kind, &value, &mut seen_ids);
        let domain_contract_valid = domain_errors.is_empty();
        errors.extend(domain_errors);
        results.push(RecordResult {
            line,
            record_id: id,
            schema_valid,
            domain_contract_valid,
            errors,
        });
    }

    if results.is_empty() {
        return Err("input contains no JSONL records".into());
    }

    let schema_valid_records = results.iter().filter(|r| r.schema_valid).count();
    let domain_valid_records = results.iter().filter(|r| r.domain_contract_valid).count();
    let passed = results
        .iter()
        .all(|r| r.schema_valid && r.domain_contract_valid);

    let report = Report {
        report_schema_version: "knowledge-benchmark-validation-v0",
        tool_version: TOOL_VERSION,
        kind: config.kind.as_str(),
        subject_revision: config.subject_revision,
        schema_sha256: schema_hash,
        input_sha256: input_hash,
        dependency_lock_sha256: lock_hash,
        declared_schema_dialect: dialect,
        schema_meta_valid: meta_valid,
        schema_compiled: true,
        record_count: results.len(),
        schema_valid_records,
        domain_valid_records,
        passed,
        records: results,
        nonclaims: vec![
            "Schema and domain-contract validation do not establish historical truth.",
            "Validation does not establish rights, permissions, community authorization, or empirical validity.",
            "Validation does not measure model quality or state-of-the-art performance.",
            "A passed fixture validation is not a model benchmark run.",
        ],
    };
    Ok((report, passed))
}

fn write_report(path: Option<&Path>, report: &Report) -> Result<(), String> {
    let json = serde_json::to_string_pretty(report)
        .map_err(|e| format!("cannot serialize validation report: {e}"))?;
    match path {
        Some(path) => fs::write(path, format!("{json}\n"))
            .map_err(|e| format!("cannot write report {}: {e}", path.display())),
        None => {
            println!("{json}");
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    let config = match parse_args() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    let output = config.output.clone();
    match run(config) {
        Ok((report, passed)) => {
            if let Err(error) = write_report(output.as_deref(), &report) {
                eprintln!("{error}");
                return ExitCode::from(2);
            }
            if passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(error) => {
            eprintln!("validation could not be completed: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn hashes_are_stable_for_identical_bytes() {
        assert_eq!(sha256_hex(b"same"), sha256_hex(b"same"));
        assert_ne!(sha256_hex(b"same"), sha256_hex(b"changed"));
    }

    #[test]
    fn external_schema_references_are_rejected() {
        let schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "properties": {
                "nested": { "$ref": "https://example.invalid/schema.json" }
            }
        });
        let mut errors = Vec::new();
        find_external_schema_refs(&schema, "$", &mut errors);
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn local_fragment_references_are_allowed() {
        let schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {"id": {"type": "string"}},
            "properties": {"id": {"$ref": "#/$defs/id"}}
        });
        let mut errors = Vec::new();
        find_external_schema_refs(&schema, "$", &mut errors);
        assert!(errors.is_empty());
    }

    #[test]
    fn graph_rejects_dangling_node_and_source_references() {
        let graph = json!({
            "synthetic_only": true,
            "source_records": [{
                "source_id":"s1",
                "source_kind":"synthetic_control"
            }],
            "nodes": [{
                "node_id":"n1",
                "source_ids":["missing-source"]
            }],
            "edges": [{
                "edge_id":"e1",
                "source_node":"missing-node",
                "target_node":"n1",
                "relation":"resembles",
                "source_ids":["s1"],
                "evidence_basis":"synthetic_assertion"
            }]
        });
        let errors = validate_graph_contract(&graph);
        assert!(errors.iter().any(|e| e.contains("unknown source_id")));
        assert!(errors.iter().any(|e| e.contains("unknown source node")));
    }

    #[test]
    fn synthetic_graph_cannot_claim_historical_influence() {
        let graph = json!({
            "synthetic_only": true,
            "source_records": [{
                "source_id":"s1",
                "source_kind":"synthetic_control"
            }],
            "nodes": [
                {"node_id":"a","source_ids":["s1"]},
                {"node_id":"b","source_ids":["s1"]}
            ],
            "edges": [{
                "edge_id":"e1",
                "source_node":"a",
                "target_node":"b",
                "relation":"historically_influences",
                "source_ids":["s1"],
                "evidence_basis":"synthetic_assertion",
                "epistemic_status":"inference"
            }]
        });
        let errors = validate_graph_contract(&graph);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("cannot assert historical influence"))
        );
    }

    #[test]
    fn case_contract_rejects_duplicate_evidence_ids() {
        let value = json!({
            "id":"case-1",
            "evidence_pack":[
                {"evidence_id":"same","statement":"A","kind":"synthetic_control"},
                {"evidence_id":"same","statement":"B","kind":"synthetic_control"}
            ]
        });
        let errors = validate_domain_contract(Kind::Case, &value, &mut HashSet::new());
        assert!(
            errors
                .iter()
                .any(|e| e.contains("duplicate evidence identifier"))
        );
    }

    #[test]
    fn claim_contract_rejects_dangling_counterclaim_sources() {
        let value = json!({
            "id":"claim-1",
            "sources":[{"source_id":"known"}],
            "counterclaims":[{"statement":"disputes claim","source_ids":["missing"]}]
        });
        let errors = validate_domain_contract(Kind::Claim, &value, &mut HashSet::new());
        assert!(errors.iter().any(|e| e.contains("unknown claim source_id")));
    }

    #[test]
    fn narrative_sequence_does_not_imply_causation() {
        let graph = json!({
            "synthetic_only": true,
            "source_records": [{
                "source_id":"s1",
                "source_kind":"synthetic_control"
            }],
            "nodes": [
                {"node_id":"a","source_ids":["s1"]},
                {"node_id":"b","source_ids":["s1"]}
            ],
            "edges": [{
                "edge_id":"e1",
                "source_node":"a",
                "target_node":"b",
                "relation":"precedes",
                "source_ids":["s1"],
                "evidence_basis":"synthetic_assertion"
            }]
        });
        assert!(validate_graph_contract(&graph).is_empty());
    }
    #[test]
    fn synthetic_pair_requires_synthetic_split_and_evidence() {
        let pair = json!({
            "schema_version":"mythology-metamorphic-pair-v0",
            "split":"synthetic",
            "synthetic_only":true,
            "pair_id":"synthetic-1",
            "dimension":"context_swap",
            "expected_relation":"local_change",
            "baseline":{
                "context_id":"ctx-A","tradition_scope":"synthetic","source_id":"s1","variant_id":"v1",
                "prompt":"synthetic prompt","rights_status":"synthetic_only",
                "evidence_pack":[{"evidence_id":"e1","source_id":"s1","statement":"control","kind":"source_attestation"}]
            },
            "mutated":{
                "context_id":"ctx-B","tradition_scope":"synthetic","source_id":"s1","variant_id":"v1",
                "prompt":"synthetic prompt","rights_status":"synthetic_only",
                "evidence_pack":[{"evidence_id":"e2","source_id":"s1","statement":"control","kind":"synthetic_control"}]
            },
            "intervention":{"field":"context_id","before_value":"ctx-A","after_value":"ctx-B","operation":"swap"},
            "acceptance":{"must_preserve":["source"],"must_change":["context"],"must_not":["false source"]},
            "criticality":"major"
        });
        let errors = validate_pair_contract(&pair);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("non-synthetic evidence kind"))
        );
    }

    #[test]
    fn held_out_pair_cannot_use_unknown_rights_or_synthetic_evidence() {
        let pair = json!({
            "schema_version":"mythology-metamorphic-pair-v0",
            "split":"holdout",
            "synthetic_only":false,
            "pair_id":"holdout-1",
            "dimension":"source_swap",
            "expected_relation":"local_change",
            "baseline":{
                "context_id":"ctx-A","tradition_scope":"example","source_id":"s1","variant_id":"v1",
                "prompt":"summarize","rights_status":"unknown",
                "evidence_pack":[{"evidence_id":"e1","source_id":"s1","statement":"content","kind":"synthetic_control"}]
            },
            "mutated":{
                "context_id":"ctx-B","tradition_scope":"example","source_id":"s2","variant_id":"v2",
                "prompt":"summarize","rights_status":"open_license_verified",
                "evidence_pack":[{"evidence_id":"e2","source_id":"s2","statement":"content","kind":"source_attestation"}]
            },
            "intervention":{"field":"source_id","before_value":"s1","after_value":"s2","operation":"swap"},
            "acceptance":{"must_preserve":["source lineage"],"must_change":["citation"],"must_not":["false citation"]},
            "criticality":"release_blocker"
        });
        let errors = validate_pair_contract(&pair);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("not admissible for split"))
        );
        assert!(
            errors
                .iter()
                .any(|e| e.contains("inadmissible evidence kind"))
        );
    }

    #[test]
    fn context_swap_pair_rejects_unchanged_context() {
        let pair = json!({
            "dimension": "context_swap",
            "expected_relation": "local_change",
            "baseline": {"context_id":"same","evidence_pack":[]},
            "mutated": {"context_id":"same","evidence_pack":[]},
            "intervention": {
                "field":"context_id",
                "before_value":"same",
                "after_value":"different"
            },
            "acceptance": {"must_change":["context-conditioned meaning"]}
        });
        let errors = validate_pair_contract(&pair);
        assert!(errors.iter().any(|e| e.contains("must change context_id")));
    }

    #[test]
    fn relation_mutation_pair_must_expect_rejection() {
        let pair = json!({
            "dimension": "relation_mutation",
            "expected_relation": "local_change",
            "baseline": {"context_id":"same","evidence_pack":[]},
            "mutated": {"context_id":"same","evidence_pack":[]},
            "intervention": {
                "field":"relation",
                "before_value":"resembles",
                "after_value":"historically_influences"
            },
            "acceptance": {"must_change":["edge is rejected"]}
        });
        let errors = validate_pair_contract(&pair);
        assert!(errors.iter().any(|e| {
            e.contains("must bind the proposed relation and reject unsupported promotion")
        }));
    }

    #[test]
    fn relation_mutation_rejects_changed_evidence_pack() {
        let pair = json!({
            "dimension": "relation_mutation",
            "expected_relation": "reject_unsupported_relation",
            "baseline": {
                "proposed_relation":"resembles",
                "evidence_pack":[{"evidence_id":"e1","source_id":"s1","statement":"same","kind":"synthetic_control"}]
            },
            "mutated": {
                "proposed_relation":"historically_influences",
                "evidence_pack":[{"evidence_id":"e1","source_id":"s1","statement":"same","kind":"synthetic_control"},{"evidence_id":"e2","source_id":"s1","statement":"new","kind":"synthetic_control"}]
            },
            "intervention": {
                "field":"relation",
                "before_value":"resembles",
                "after_value":"historically_influences"
            },
            "acceptance": {"must_change":["edge is rejected"]}
        });
        let errors = validate_pair_contract(&pair);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("must keep the evidence pack identical"))
        );
    }

    #[test]
    fn rights_revocation_pair_requires_denial() {
        let pair = json!({
            "split":"synthetic",
            "synthetic_only":true,
            "dimension": "rights_revocation",
            "expected_relation": "deny_access",
            "baseline": {
                "rights_status":"synthetic_only",
                "access_decision":"allow",
                "evidence_pack":[]
            },
            "mutated": {
                "rights_status":"synthetic_only",
                "access_decision":"deny",
                "evidence_pack":[]
            },
            "intervention": {
                "field":"access_decision",
                "before_value":"allow",
                "after_value":"deny"
            },
            "acceptance": {"must_change":["access is denied"]}
        });
        assert!(validate_pair_contract(&pair).is_empty());
    }

    #[test]
    fn source_swap_pair_binds_intervention_to_actual_witnesses() {
        let pair = json!({
            "dimension": "source_swap",
            "expected_relation": "local_change",
            "baseline": {"source_id":"source-A","evidence_pack":[]},
            "mutated": {"source_id":"source-B","evidence_pack":[]},
            "intervention": {
                "field":"source_id",
                "before_value":"wrong-source",
                "after_value":"source-B"
            },
            "acceptance": {"must_change":["citation target changes"]}
        });
        let errors = validate_pair_contract(&pair);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("before_value does not match baseline source_id"))
        );
    }
    #[test]
    fn nonsynthetic_pair_requires_approved_catalog_sources() {
        let pair = json!({
            "pair_id":"real-pair-placeholder",
            "split":"holdout",
            "synthetic_only":false,
            "baseline":{
                "source_id":"met-divine-egypt",
                "rights_status":"unknown",
                "evidence_pack":[{"source_id":"met-divine-egypt","kind":"source_attestation"}]
            },
            "mutated":{
                "source_id":"met-divine-egypt",
                "rights_status":"unknown",
                "evidence_pack":[{"source_id":"met-divine-egypt","kind":"source_attestation"}]
            }
        });
        let errors = validate_pair_catalog_refs(&pair);
        assert!(errors.iter().any(|e| e.contains("is in split")));
        assert!(errors.iter().any(|e| e.contains("metadata-only")));
        assert!(errors.iter().any(|e| e.contains("lacks approved rights")));
    }

    #[test]
    fn source_family_split_leakage_is_rejected() {
        let mut seen_ids = HashSet::new();
        let source_a = json!({
            "source_id":"witness-A",
            "source_family_id":"work-family",
            "split":"development",
            "rights_status":"unknown",
            "content_access":"metadata_only",
            "rights_review":"pending"
        });
        let source_b = json!({
            "source_id":"witness-B",
            "source_family_id":"work-family",
            "split":"test",
            "rights_status":"unknown",
            "content_access":"metadata_only",
            "rights_review":"pending"
        });
        assert!(validate_domain_contract(Kind::Source, &source_a, &mut seen_ids).is_empty());
        let errors = validate_domain_contract(Kind::Source, &source_b, &mut seen_ids);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("spans multiple evaluation splits"))
        );
    }

    #[test]
    fn unknown_rights_cannot_authorize_full_text_access() {
        let source = json!({
            "source_id":"source-unknown",
            "source_family_id":"family-unknown",
            "split":"development",
            "rights_status":"unknown",
            "content_access":"permissioned_full_text",
            "rights_review":"pending"
        });
        let errors = validate_source_contract(&source, &mut HashSet::new());
        assert!(
            errors
                .iter()
                .any(|e| e.contains("rights_status=unknown requires content_access=metadata_only"))
        );
    }

    #[test]
    fn bundled_sources_and_case_references_are_consistent() {
        let mut catalog = std::collections::HashMap::new();
        let mut seen_ids = HashSet::new();
        for (index, line) in include_str!("../../../benchmarks/source-catalog/v0/sources.jsonl")
            .lines()
            .enumerate()
        {
            let source: Value = serde_json::from_str(line).unwrap_or_else(|error| {
                panic!(
                    "invalid source catalog JSONL at line {}: {error}",
                    index + 1
                )
            });
            let errors = validate_domain_contract(Kind::Source, &source, &mut seen_ids);
            assert!(
                errors.is_empty(),
                "source catalog line {} failed: {}",
                index + 1,
                errors.join("; ")
            );
            let id = string_field(&source, "source_id").unwrap().to_string();
            let uri = string_field(&source, "uri").unwrap().to_string();
            let split = string_field(&source, "split").unwrap().to_string();
            assert!(
                catalog.insert(id.clone(), (uri, split)).is_none(),
                "duplicate source catalog ID {id:?}"
            );
        }

        let case_sets = [
            include_str!("../../../benchmarks/esoterica-bench/v0/cases.jsonl"),
            include_str!("../../../benchmarks/mythology-bench/v0/cases.jsonl"),
        ];
        for (set_index, cases) in case_sets.iter().enumerate() {
            for (line_index, line) in cases.lines().enumerate() {
                let case: Value = serde_json::from_str(line).unwrap_or_else(|error| {
                    panic!(
                        "invalid case JSONL in set {} line {}: {error}",
                        set_index + 1,
                        line_index + 1
                    )
                });
                if let Some(source_refs) = case.get("source_refs").and_then(Value::as_array) {
                    for source_ref in source_refs {
                        let source_id = string_field(source_ref, "source_id").unwrap_or("");
                        let uri = string_field(source_ref, "uri").unwrap_or("");
                        let (catalog_uri, source_split) = catalog.get(source_id).unwrap_or_else(|| {
                            panic!(
                                "case {} references source ID {source_id:?} absent from catalog",
                                string_field(&case, "id").unwrap_or("<unknown>")
                            )
                        });
                        assert_eq!(
                            uri,
                            catalog_uri.as_str(),
                            "case {} source URI differs from catalog for {source_id:?}",
                            string_field(&case, "id").unwrap_or("<unknown>")
                        );
                        assert_eq!(
                            string_field(&case, "split").unwrap_or(""),
                            source_split,
                            "case {} split differs from source family split for {source_id:?}",
                            string_field(&case, "id").unwrap_or("<unknown>")
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn content_annotation_rejects_metadata_only_source_records() {
        let annotation = json!({
            "annotation_id":"ANN-METADATA-ONLY",
            "annotation_role":"independent_reviewer",
            "source_family_ids":["met-divine-egypt-exhibition"],
            "claim_annotations":[
                {
                    "claim_id":"C1",
                    "evidence_refs":[{"source_id":"met-divine-egypt","locator":"Creating the World"}]
                }
            ],
            "valid_alternatives":[]
        });
        let errors = validate_annotation_catalog_refs(&annotation);
        assert!(errors.iter().any(|e| e.contains("metadata-only")));
        assert!(
            errors
                .iter()
                .any(|e| e.contains("lacks approved rights review"))
        );
    }

    #[test]
    fn real_annotation_rejects_unknown_source_ids() {
        let annotation = json!({
            "annotation_id":"ANN-REAL-PLACEHOLDER",
            "annotation_role":"independent_reviewer",
            "valid_alternatives":[
                {"alternative_id":"ALT1","source_ids":["SYN-NOT-IN-CATALOG"]}
            ],
            "claim_annotations":[
                {"claim_id":"C1","evidence_refs":[{"source_id":"SYN-NOT-IN-CATALOG","locator":null}]}
            ],
            "source_family_ids":["SYN-FAMILY-NOT-IN-CATALOG"]
        });
        let errors = validate_annotation_catalog_refs(&annotation);
        assert!(errors.iter().any(|e| e.contains("absent from catalog")));
    }

    #[test]
    fn rights_prohibited_annotation_cannot_leak_content() {
        let annotation = json!({
            "answerability":"rights_prohibited",
            "abstention_assessment":"appropriate",
            "review_status":"synthetic_only",
            "annotation_role":"synthetic_fixture",
            "claim_annotations":[{"claim_id":"c1","span":"restricted text","support_label":"supported","epistemic_lane":"source_attested","severity":"none","failure_tags":["none"],"evidence_refs":[],"reviewer_confidence":1.0}],
            "valid_alternatives":[]
        });
        let errors = validate_annotation_contract(&annotation);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("must not expose claim content or alternatives"))
        );
    }

    #[test]
    fn duplicate_claim_annotation_ids_are_rejected() {
        let annotation = json!({
            "answerability":"fully_answerable",
            "review_status":"synthetic_only",
            "annotation_role":"synthetic_fixture",
            "claim_annotations":[
                {"claim_id":"c1","support_label":"unsupported","epistemic_lane":"unknown","severity":"minor","failure_tags":["universalization"],"evidence_refs":[]},
                {"claim_id":"c1","support_label":"unsupported","epistemic_lane":"unknown","severity":"minor","failure_tags":["universalization"],"evidence_refs":[]}
            ]
        });
        let errors = validate_annotation_contract(&annotation);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("duplicate claim annotation ID"))
        );
    }

    #[test]
    fn source_pack_rejects_non_allow_access_with_evidence() {
        let record = json!({
            "schema_version":"knowledge-source-pack-v0",
            "synthetic_only":false,
            "split":"holdout",
            "pair_id":"rights-test",
            "state":"mutated",
            "context_id":"context",
            "tradition_scope":"example",
            "source_id":"source-1",
            "variant_id":"v1",
            "prompt":"retrieve",
            "rights_status":"open_license_verified",
            "access_decision":"deny",
            "evidence_pack":[{"evidence_id":"e1","source_id":"source-1","statement":"content","kind":"source_attestation"}]
        });
        let errors = validate_source_pack_contract(&record);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("exposes evidence despite access_decision"))
        );
    }

    #[test]
    fn bundled_synthetic_source_pack_has_unique_pair_states() {
        let mut seen_ids = HashSet::new();
        for (index, raw_line) in
            include_str!("../../../benchmarks/benchmark-runs/v0/synthetic-source-pack.jsonl")
                .lines()
                .enumerate()
        {
            let record: Value = serde_json::from_str(raw_line).unwrap_or_else(|error| {
                panic!("invalid source-pack JSONL at line {}: {error}", index + 1)
            });
            let errors = validate_domain_contract(Kind::SourcePack, &record, &mut seen_ids);
            assert!(
                errors.is_empty(),
                "source-pack line {} failed: {}",
                index + 1,
                errors.join("; ")
            );
        }
        assert_eq!(seen_ids.len(), 20);
    }

    #[test]
    fn pair_result_rejects_duplicate_forbidden_tags() {
        let result = json!({
            "result_id":"r1",
            "plan_id":"p1",
            "system_id":"s1",
            "pair_id":"case1",
            "observed_relation":"invariant",
            "forbidden_behaviors":["rights_violation","rights_violation"],
            "notes":"synthetic test row",
            "adjudication_status":"synthetic_only"
        });
        let errors = validate_pair_result_contract(&result);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("repeats a forbidden_behavior tag"))
        );
    }

    #[test]
    fn bundled_paired_results_satisfy_domain_contract() {
        let mut seen_ids = HashSet::new();
        for (index, raw_line) in
            include_str!("../../../benchmarks/benchmark-runs/v0/synthetic-pair-results.jsonl")
                .lines()
                .enumerate()
        {
            let result: Value = serde_json::from_str(raw_line).unwrap_or_else(|error| {
                panic!("invalid paired result JSONL at line {}: {error}", index + 1)
            });
            let errors = validate_domain_contract(Kind::PairResult, &result, &mut seen_ids);
            assert!(
                errors.is_empty(),
                "paired result line {} failed: {}",
                index + 1,
                errors.join("; ")
            );
        }
    }

    #[test]
    fn bundled_comparison_plans_satisfy_domain_contract() {
        let mut seen_ids = HashSet::new();
        for (index, raw_line) in
            include_str!("../../../benchmarks/benchmark-runs/v0/synthetic-plans.jsonl")
                .lines()
                .enumerate()
        {
            let plan: Value = serde_json::from_str(raw_line).unwrap_or_else(|error| {
                panic!(
                    "invalid comparison plan JSONL at line {}: {error}",
                    index + 1
                )
            });
            let errors = validate_domain_contract(Kind::Run, &plan, &mut seen_ids);
            assert!(
                errors.is_empty(),
                "comparison plan line {} failed: {}",
                index + 1,
                errors.join("; ")
            );
        }
    }

    #[test]
    fn held_out_plan_rejects_unreviewed_or_leaky_data() {
        let mut plan = json!({
            "phase":"preregistered_held_out",
            "synthetic_only":false,
            "benchmark":{
                "split":"holdout",
                "source_family_disjoint":false,
                "rights_review_status":"pending",
                "case_manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "source_catalog_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "source_pack_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
            },
            "preregistration":{"held_out_access":"not_sealed"},
            "systems":[
                {"system_id":"bm25","baseline_family":"bm25","implementation":"x","revision":"r1","prompt_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","retriever_revision":"r1"},
                {"system_id":"graph","baseline_family":"typed_graph","implementation":"x","revision":"r1","prompt_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","retriever_revision":"r1"}
            ]
        });
        let errors = validate_run_contract(&plan);
        assert!(errors.iter().any(|e| e.contains("source-family-disjoint")));
        assert!(
            errors
                .iter()
                .any(|e| e.contains("rights_review_status=approved"))
        );
        assert!(
            errors
                .iter()
                .any(|e| e.contains("sealed or independent-evaluator"))
        );
        plan["benchmark"]["source_family_disjoint"] = Value::Bool(true);
        plan["benchmark"]["rights_review_status"] = Value::String("approved".into());
        plan["preregistration"]["held_out_access"] = Value::String("sealed_until_freeze".into());
        assert!(
            !validate_run_contract(&plan)
                .iter()
                .any(|e| e.contains("source-family-disjoint")
                    || e.contains("rights_review_status=approved")
                    || e.contains("sealed or independent-evaluator"))
        );
    }

    #[test]
    fn bundled_annotation_fixtures_satisfy_domain_contract() {
        let mut seen_ids = HashSet::new();
        for (index, raw_line) in
            include_str!("../../../benchmarks/knowledge-annotation/v0/synthetic-fixtures.jsonl")
                .lines()
                .enumerate()
        {
            let annotation: Value = serde_json::from_str(raw_line).unwrap_or_else(|error| {
                panic!("invalid annotation JSONL at line {}: {error}", index + 1)
            });
            let errors = validate_domain_contract(Kind::Annotation, &annotation, &mut seen_ids);
            assert!(
                errors.is_empty(),
                "annotation fixture line {} failed: {}",
                index + 1,
                errors.join("; ")
            );
        }
    }

    #[test]
    fn bundled_metamorphic_pairs_satisfy_domain_contract() {
        let mut seen_ids = HashSet::new();
        for (index, raw_line) in
            include_str!("../../../benchmarks/mythology-bench/v0/metamorphic-pairs.jsonl")
                .lines()
                .enumerate()
        {
            let pair: Value = serde_json::from_str(raw_line).unwrap_or_else(|error| {
                panic!("invalid metamorphic JSONL at line {}: {error}", index + 1)
            });
            let errors = validate_domain_contract(Kind::Pair, &pair, &mut seen_ids);
            assert!(
                errors.is_empty(),
                "metamorphic pair line {} failed: {}",
                index + 1,
                errors.join("; ")
            );
        }
    }
}
