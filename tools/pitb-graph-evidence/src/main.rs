use serde::{Deserialize, Serialize};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs,
    io::{self, Read},
    path::{Component, Path, PathBuf},
    process::ExitCode,
};

const VERSION: &str = "pitb.proof-graph-evidence.v0";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Graph {
    schema_version: String,
    evidence_class: String,
    snapshot: Snapshot,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    artifacts: Vec<Artifact>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    snapshot_id: String,
    repository_commit: String,
    environment_sha256: String,
    extractor_id: String,
    extractor_version: String,
    extractor_config_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Node {
    node_id: String,
    kind: String,
    label: String,
    statement_sha256: String,
    statement_artifact_id: String,
    source_locator: String,
    formal_name: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Edge {
    edge_id: String,
    source_node_id: String,
    target_node_id: String,
    relation: String,
    evidence_class: String,
    status: String,
    evidence_artifact_id: String,
    confidence: Value,
    review_artifact_id: Value,
    reviewer_key_id: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    artifact_id: String,
    kind: String,
    path: String,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct Summary {
    snapshot_id: String,
    node_count: usize,
    edge_count: usize,
    artifact_count: usize,
    relation_counts: BTreeMap<String, usize>,
    role_counts: BTreeMap<String, usize>,
    formal_dependency_edge_count: usize,
    module_environment_edge_count: usize,
    retrieval_hint_edge_count: usize,
    rejected_edge_count: usize,
    snapshot_signature_verified: bool,
    scheduler_admission: bool,
    scheduler_admission_note: String,
}

fn fail(message: impl AsRef<str>) -> ExitCode {
    eprintln!("INVALID: {}", message.as_ref());
    ExitCode::FAILURE
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_sha256(value: &str) -> bool {
    valid_hex(value, 64)
}

fn valid_sha1(value: &str) -> bool {
    valid_hex(value, 40)
}

fn value_string<'a>(value: &'a Value, label: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .ok_or_else(|| format!("{label} must be a string"))
}

fn check_optional_pair(
    review_artifact_id: &Value,
    reviewer_key_id: &Value,
    required: bool,
    edge_id: &str,
) -> Result<(), String> {
    if required {
        let artifact = value_string(review_artifact_id, &format!("edge {edge_id} review_artifact_id"))?;
        let key = value_string(reviewer_key_id, &format!("edge {edge_id} reviewer_key_id"))?;
        if !non_empty(artifact) || !non_empty(key) {
            return Err(format!("edge {edge_id} requires non-empty review artifact and reviewer key IDs"));
        }
    } else if !review_artifact_id.is_null() || !reviewer_key_id.is_null() {
        return Err(format!(
            "edge {edge_id} must not claim reviewer evidence unless status is human_reviewed"
        ));
    }
    Ok(())
}

fn safe_relative_path(relative: &str) -> Result<(), String> {
    let path = Path::new(relative);
    if relative.trim().is_empty() || path.is_absolute() {
        return Err(format!("artifact path must be non-empty and relative: {relative}"));
    }
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            _ => return Err(format!("artifact path contains traversal or prefix: {relative}")),
        }
    }
    if relative.split('/').any(|part| part == "." || part == ".." || part.is_empty()) {
        return Err(format!("artifact path contains an unsafe segment: {relative}"));
    }
    Ok(())
}

fn expected_evidence_class(relation: &str) -> Option<&'static str> {
    match relation {
        "type_reference"
        | "signature_reference"
        | "proof_term_reference"
        | "definition_body_reference"
        | "structure_field_reference"
        | "inheritance_reference" => Some("lean_elaborator"),
        "instance_synthesis" => Some("lean_instance_trace"),
        "documentation_reference" => Some("lean_doc_parser"),
        "blueprint_dependency" => Some("blueprint_curated"),
        "informal_candidate_dependency" => Some("informal_extractor"),
        "cross_formality_match" => Some("llm_alignment"),
        "module_environment_dependency" => Some("lean_build_graph"),
        _ => None,
    }
}

fn is_formal_dependency(relation: &str) -> bool {
    matches!(
        relation,
        "type_reference"
            | "signature_reference"
            | "proof_term_reference"
            | "definition_body_reference"
            | "structure_field_reference"
            | "inheritance_reference"
            | "instance_synthesis"
            | "module_environment_dependency"
    )
}

fn is_observed_relation(relation: &str) -> bool {
    is_formal_dependency(relation) || relation == "documentation_reference"
}

fn expected_role(edge: &Edge) -> &'static str {
    if edge.status == "rejected" {
        return "excluded";
    }
    match edge.relation.as_str() {
        "type_reference" | "signature_reference" => "formal_signature_dependency_hash_bound",
        "proof_term_reference" => "proof_support_observation_hash_bound",
        "definition_body_reference" => "definition_body_reference_observation_hash_bound",
        "structure_field_reference" => "structure_field_type_reference_observation_hash_bound",
        "inheritance_reference" => "structure_or_class_inheritance_observation_hash_bound",
        "instance_synthesis" => "instance_environment_dependency_hash_bound",
        "module_environment_dependency" => "build_environment_dependency_hash_bound",
        "documentation_reference" => "documentation_retrieval_hint",
        "blueprint_dependency" if edge.status == "human_reviewed" => {
            "curated_search_prior_review_signature_unverified"
        }
        "blueprint_dependency" => "candidate_retrieval_hint",
        "informal_candidate_dependency" | "cross_formality_match" => "candidate_retrieval_hint",
        _ => "invalid",
    }
}

fn require_node_kind(
    node: &Node,
    allowed: &[&str],
    edge_id: &str,
    endpoint: &str,
) -> Result<(), String> {
    if allowed.contains(&node.kind.as_str()) {
        Ok(())
    } else {
        Err(format!(
            "edge {edge_id} {endpoint} node {} has incompatible kind {}",
            node.node_id, node.kind
        ))
    }
}

fn validate_edge_endpoints(edge: &Edge, source: &Node, target: &Node) -> Result<(), String> {
    match edge.relation.as_str() {
        "type_reference"
        | "signature_reference"
        | "proof_term_reference"
        | "definition_body_reference"
        | "structure_field_reference"
        | "inheritance_reference"
        | "instance_synthesis"
        | "documentation_reference" => {
            require_node_kind(source, &["formal_declaration"], &edge.edge_id, "source")?;
            require_node_kind(target, &["formal_declaration"], &edge.edge_id, "target")?;
        }
        "module_environment_dependency" => {
            require_node_kind(
                source,
                &["formal_declaration", "module"],
                &edge.edge_id,
                "source",
            )?;
            require_node_kind(
                target,
                &["module", "environment_artifact"],
                &edge.edge_id,
                "target",
            )?;
        }
        "blueprint_dependency" => {
            require_node_kind(
                source,
                &["blueprint_node", "formal_declaration", "informal_statement"],
                &edge.edge_id,
                "source",
            )?;
            require_node_kind(
                target,
                &["blueprint_node", "formal_declaration", "informal_statement"],
                &edge.edge_id,
                "target",
            )?;
        }
        "informal_candidate_dependency" => {
            require_node_kind(
                source,
                &["informal_statement"],
                &edge.edge_id,
                "source",
            )?;
            require_node_kind(
                target,
                &["informal_statement"],
                &edge.edge_id,
                "target",
            )?;
        }
        "cross_formality_match" => {
            let one_formal = source.kind == "formal_declaration"
                && matches!(target.kind.as_str(), "informal_statement" | "blueprint_node");
            let reverse_formal = target.kind == "formal_declaration"
                && matches!(source.kind.as_str(), "informal_statement" | "blueprint_node");
            if !one_formal && !reverse_formal {
                return Err(format!(
                    "edge {} must connect one formal declaration to one informal/blueprint node",
                    edge.edge_id
                ));
            }
        }
        other => return Err(format!("edge {} has unsupported relation {other}", edge.edge_id)),
    }
    Ok(())
}

fn validate_confidence(edge: &Edge) -> Result<(), String> {
    match &edge.confidence {
        Value::Null if edge.status == "observed" => Ok(()),
        Value::Null if edge.status == "rejected" => Ok(()),
        Value::Null if edge.status == "human_reviewed" && edge.relation == "blueprint_dependency" => {
            Ok(())
        }
        Value::Number(number) => {
            let value = number
                .as_f64()
                .ok_or_else(|| format!("edge {} confidence is not finite numeric", edge.edge_id))?;
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(format!("edge {} confidence must be in [0, 1]", edge.edge_id));
            }
            if edge.status == "observed" {
                return Err(format!("edge {} observed status must not carry model confidence", edge.edge_id));
            }
            Ok(())
        }
        _ => Err(format!(
            "edge {} confidence is required for candidate evidence and null for observed evidence",
            edge.edge_id
        )),
    }
}

fn visit_dependency(
    node: &str,
    graph: &BTreeMap<String, BTreeSet<String>>,
    states: &mut BTreeMap<String, u8>,
) -> Result<(), String> {
    match states.get(node).copied().unwrap_or(0) {
        1 => return Err(format!("formal dependency cycle includes node {node}")),
        2 => return Ok(()),
        _ => {}
    }
    states.insert(node.to_owned(), 1);
    if let Some(children) = graph.get(node) {
        for child in children {
            visit_dependency(child, graph, states)?;
        }
    }
    states.insert(node.to_owned(), 2);
    Ok(())
}

fn validate_value(value: &Value) -> Result<(Graph, Summary), String> {
    let graph: Graph = serde_json::from_value(value.clone())
        .map_err(|error| format!("graph JSON decoding failed: {error}"))?;

    if graph.schema_version != VERSION {
        return Err("unsupported graph schema_version".into());
    }
    if !matches!(graph.evidence_class.as_str(), "synthetic_contract_fixture" | "measured_snapshot") {
        return Err("unknown evidence_class".into());
    }
    let snapshot = &graph.snapshot;
    if !non_empty(&snapshot.snapshot_id)
        || !valid_sha1(&snapshot.repository_commit)
        || !valid_sha256(&snapshot.environment_sha256)
        || !non_empty(&snapshot.extractor_id)
        || !non_empty(&snapshot.extractor_version)
        || !valid_sha256(&snapshot.extractor_config_sha256)
    {
        return Err("snapshot identity is incomplete or malformed".into());
    }
    if graph.nodes.is_empty() {
        return Err("graph must contain at least one node".into());
    }

    let artifacts: BTreeMap<String, &Artifact> = {
        let mut map = BTreeMap::new();
        let mut paths = BTreeSet::new();
        for artifact in &graph.artifacts {
            if !non_empty(&artifact.artifact_id) || !valid_sha256(&artifact.sha256) {
                return Err(format!("artifact {} has incomplete identity", artifact.artifact_id));
            }
            if !matches!(
                artifact.kind.as_str(),
                "statement" | "edge_evidence" | "review_attestation"
            ) {
                return Err(format!("artifact {} has unknown kind", artifact.artifact_id));
            }
            safe_relative_path(&artifact.path)?;
            if !paths.insert(artifact.path.clone()) {
                return Err(format!("duplicate artifact path {}", artifact.path));
            }
            if map.insert(artifact.artifact_id.clone(), artifact).is_some() {
                return Err(format!("duplicate artifact ID {}", artifact.artifact_id));
            }
        }
        map
    };

    let mut used_artifacts = BTreeSet::new();
    let mut nodes = BTreeMap::<String, &Node>::new();
    for node in &graph.nodes {
        if !non_empty(&node.node_id)
            || !non_empty(&node.label)
            || !non_empty(&node.source_locator)
            || !valid_sha256(&node.statement_sha256)
        {
            return Err(format!("node {} has incomplete identity", node.node_id));
        }
        if !matches!(
            node.kind.as_str(),
            "formal_declaration" | "blueprint_node" | "informal_statement" | "module" | "environment_artifact"
        ) {
            return Err(format!("node {} has unknown kind", node.node_id));
        }
        match (&node.kind[..], &node.formal_name) {
            ("formal_declaration", Value::String(name)) if non_empty(name) => {}
            ("formal_declaration", _) => {
                return Err(format!("formal node {} requires a formal_name", node.node_id))
            }
            (_, Value::Null) => {}
            _ => return Err(format!("non-formal node {} must set formal_name to null", node.node_id)),
        }
        let statement_artifact = artifacts
            .get(&node.statement_artifact_id)
            .ok_or_else(|| format!("node {} references an unknown statement artifact", node.node_id))?;
        if statement_artifact.kind != "statement" || statement_artifact.sha256 != node.statement_sha256 {
            return Err(format!("node {} statement artifact kind/digest mismatch", node.node_id));
        }
        used_artifacts.insert(node.statement_artifact_id.clone());
        if nodes.insert(node.node_id.clone(), node).is_some() {
            return Err(format!("duplicate node ID {}", node.node_id));
        }
    }

    let mut edge_ids = BTreeSet::new();
    let mut edge_tuples = BTreeSet::new();
    let mut relation_counts = BTreeMap::<String, usize>::new();
    let mut role_counts = BTreeMap::<String, usize>::new();
    // Observed declaration references may contain valid mutual-recursion SCCs.
    // Only the module/environment import relation is required to be a DAG here.
    let mut module_dependency_graph = BTreeMap::<String, BTreeSet<String>>::new();
    let mut formal_dependency_edge_count = 0usize;
    let mut module_environment_edge_count = 0usize;
    let mut retrieval_hint_edge_count = 0usize;
    let mut rejected_edge_count = 0usize;

    for edge in &graph.edges {
        if !non_empty(&edge.edge_id)
            || !non_empty(&edge.source_node_id)
            || !non_empty(&edge.target_node_id)
        {
            return Err("edge identity fields must be non-empty".into());
        }
        if edge.source_node_id == edge.target_node_id {
            return Err(format!("edge {} cannot point to itself", edge.edge_id));
        }
        if !edge_ids.insert(edge.edge_id.clone()) {
            return Err(format!("duplicate edge ID {}", edge.edge_id));
        }
        let tuple = (
            edge.source_node_id.clone(),
            edge.target_node_id.clone(),
            edge.relation.clone(),
        );
        if !edge_tuples.insert(tuple) {
            return Err(format!("duplicate edge relation for {}", edge.edge_id));
        }

        let source = nodes
            .get(&edge.source_node_id)
            .ok_or_else(|| format!("edge {} references missing source node", edge.edge_id))?;
        let target = nodes
            .get(&edge.target_node_id)
            .ok_or_else(|| format!("edge {} references missing target node", edge.edge_id))?;
        validate_edge_endpoints(edge, source, target)?;

        let expected = expected_evidence_class(&edge.relation)
            .ok_or_else(|| format!("edge {} has unknown relation", edge.edge_id))?;
        if edge.evidence_class != expected {
            return Err(format!(
                "edge {} relation {} requires evidence_class {}, not {}",
                edge.edge_id, edge.relation, expected, edge.evidence_class
            ));
        }
        if !matches!(
            edge.status.as_str(),
            "observed" | "candidate" | "human_reviewed" | "rejected"
        ) {
            return Err(format!("edge {} has unknown status", edge.edge_id));
        }

        let formal = is_formal_dependency(&edge.relation);
        let observed = is_observed_relation(&edge.relation);
        match edge.status.as_str() {
            "observed" if !observed => {
                return Err(format!("edge {} cannot mark inferred relation as observed", edge.edge_id))
            }
            "candidate" if formal || observed => {
                return Err(format!("edge {} cannot mark formal dependency receipt as candidate", edge.edge_id))
            }
            "human_reviewed" if formal => {
                return Err(format!("edge {} formal dependency requires extractor receipt, not a review label", edge.edge_id))
            }
            _ => {}
        }
        validate_confidence(edge)?;

        let evidence = artifacts
            .get(&edge.evidence_artifact_id)
            .ok_or_else(|| format!("edge {} references unknown evidence artifact", edge.edge_id))?;
        if evidence.kind != "edge_evidence" {
            return Err(format!("edge {} evidence artifact must have kind edge_evidence", edge.edge_id));
        }
        used_artifacts.insert(edge.evidence_artifact_id.clone());

        if edge.status == "human_reviewed" {
            let review_id = value_string(&edge.review_artifact_id, "review_artifact_id")?;
            let reviewer = value_string(&edge.reviewer_key_id, "reviewer_key_id")?;
            if !non_empty(reviewer) {
                return Err(format!("edge {} reviewer_key_id cannot be empty", edge.edge_id));
            }
            let review = artifacts
                .get(review_id)
                .ok_or_else(|| format!("edge {} references unknown review artifact", edge.edge_id))?;
            if review.kind != "review_attestation" {
                return Err(format!("edge {} review artifact has the wrong kind", edge.edge_id));
            }
            used_artifacts.insert(review_id.to_owned());
        } else {
            check_optional_pair(
                &edge.review_artifact_id,
                &edge.reviewer_key_id,
                false,
                &edge.edge_id,
            )?;
        }

        let role = expected_role(edge);
        if role == "invalid" {
            return Err(format!("edge {} has no derived role", edge.edge_id));
        }
        *relation_counts.entry(edge.relation.clone()).or_default() += 1;
        *role_counts.entry(role.to_owned()).or_default() += 1;
        if edge.status == "rejected" {
            rejected_edge_count += 1;
        } else if is_formal_dependency(&edge.relation) {
            formal_dependency_edge_count += 1;
            if edge.relation == "module_environment_dependency" {
                module_environment_edge_count += 1;
                module_dependency_graph
                    .entry(edge.source_node_id.clone())
                    .or_default()
                    .insert(edge.target_node_id.clone());
            }
        } else {
            retrieval_hint_edge_count += 1;
        }
    }

    if used_artifacts.len() != artifacts.len() {
        let unused: Vec<_> = artifacts.keys().filter(|id| !used_artifacts.contains(*id)).cloned().collect();
        return Err(format!("unreferenced artifacts are forbidden: {}", unused.join(", ")));
    }

    let mut states = BTreeMap::new();
    for node_id in nodes.keys() {
        visit_dependency(node_id, &module_dependency_graph, &mut states)?;
    }

    let summary = Summary {
        snapshot_id: snapshot.snapshot_id.clone(),
        node_count: graph.nodes.len(),
        edge_count: graph.edges.len(),
        artifact_count: graph.artifacts.len(),
        relation_counts,
        role_counts,
        formal_dependency_edge_count,
        module_environment_edge_count,
        retrieval_hint_edge_count,
        rejected_edge_count,
        snapshot_signature_verified: false,
        scheduler_admission: false,
        scheduler_admission_note: "graph structure, artifact bytes and optional issuer signature do not prove extractor semantics; scheduler admission still requires independent Lean re-extraction, exact source/environment comparison and semantic review".into(),
    };
    Ok((graph, summary))
}

fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn resolve_artifact(root: &Path, relative: &str) -> Result<PathBuf, String> {
    safe_relative_path(relative)?;
    let mut current = root.to_path_buf();
    for component in Path::new(relative).components() {
        match component {
            Component::Normal(part) => current.push(part),
            _ => return Err(format!("artifact path contains traversal: {relative}")),
        }
        let metadata = fs::symlink_metadata(&current)
            .map_err(|error| format!("cannot inspect artifact {relative}: {error}"))?;
        if metadata.file_type().is_symlink() {
            return Err(format!("symlink artifact paths are forbidden: {relative}"));
        }
    }
    if !current.is_file() {
        return Err(format!("artifact is not a regular file: {relative}"));
    }
    Ok(current)
}



fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn decode_hex_array<const N: usize>(value: &str, label: &str) -> Result<[u8; N], String> {
    if !valid_hex(value, N * 2) {
        return Err(format!("{label} must be exactly {} lowercase hex characters", N * 2));
    }
    let mut decoded = [0_u8; N];
    for (index, output) in decoded.iter_mut().enumerate() {
        let start = index * 2;
        *output = u8::from_str_radix(&value[start..start + 2], 16)
            .map_err(|error| format!("decode {label}: {error}"))?;
    }
    Ok(decoded)
}

fn require_exact_fields(value: &Value, allowed: &[&str], label: &str) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be a JSON object"))?;
    let allowed_set: BTreeSet<&str> = allowed.iter().copied().collect();
    if object.len() != allowed_set.len()
        || object.keys().any(|key| !allowed_set.contains(key.as_str()))
    {
        return Err(format!("{label} contains missing or unknown fields"));
    }
    Ok(())
}

fn verify_snapshot_attestation(
    manifest_bytes: &[u8],
    attestation_bytes: &[u8],
    registry_bytes: &[u8],
    externally_pinned_registry_sha256: &str,
    graph: &Graph,
) -> Result<(), String> {
    if !valid_sha256(externally_pinned_registry_sha256) {
        return Err("externally pinned key-registry SHA-256 is malformed".into());
    }
    let actual_registry_sha256 = sha256_bytes(registry_bytes);
    if actual_registry_sha256 != externally_pinned_registry_sha256 {
        return Err("key-registry bytes differ from the out-of-band pinned SHA-256".into());
    }

    let registry: Value = serde_json::from_slice(registry_bytes)
        .map_err(|error| format!("parse graph signer key registry: {error}"))?;
    require_exact_fields(&registry, &["schema_version", "registry_id", "keys"], "key registry")?;
    if value_string(&registry["schema_version"], "registry.schema_version")?
        != "pitb.graph-signer-registry.v0"
        || !non_empty(value_string(&registry["registry_id"], "registry.registry_id")?)
    {
        return Err("unsupported or unnamed graph signer key registry".into());
    }
    let keys = registry["keys"]
        .as_array()
        .ok_or("registry.keys must be an array")?;
    if keys.is_empty() {
        return Err("graph signer key registry cannot be empty".into());
    }

    let mut previous_key_id: Option<&str> = None;
    let mut matching_key: Option<&Value> = None;
    for key in keys {
        require_exact_fields(
            key,
            &["key_id", "signer_id", "public_key_ed25519_hex", "status"],
            "graph signer key entry",
        )?;
        let key_id = value_string(&key["key_id"], "key.key_id")?;
        let signer_id = value_string(&key["signer_id"], "key.signer_id")?;
        let status = value_string(&key["status"], "key.status")?;
        if !non_empty(key_id) || !non_empty(signer_id) {
            return Err("key registry identifiers cannot be empty".into());
        }
        if previous_key_id.is_some_and(|previous| previous >= key_id) {
            return Err("graph signer keys must be sorted and unique by key_id".into());
        }
        previous_key_id = Some(key_id);
        if !matches!(status, "active" | "revoked") {
            return Err("graph signer registry contains an unknown key status".into());
        }
        let public_key_hex =
            value_string(&key["public_key_ed25519_hex"], "key.public_key_ed25519_hex")?;
        if !valid_hex(public_key_hex, 64) {
            return Err("graph signer public key must be 64 lowercase hex characters".into());
        }
    }

    let attestation: Value = serde_json::from_slice(attestation_bytes)
        .map_err(|error| format!("parse graph snapshot attestation: {error}"))?;
    require_exact_fields(
        &attestation,
        &[
            "schema_version",
            "canonicalization",
            "signer_id",
            "key_id",
            "graph_manifest_sha256",
            "snapshot_id",
            "repository_commit",
            "environment_sha256",
            "extractor_id",
            "extractor_version",
            "extractor_config_sha256",
            "signature_ed25519_hex",
        ],
        "graph snapshot attestation",
    )?;
    if value_string(&attestation["schema_version"], "attestation.schema_version")?
        != "pitb.graph-snapshot-attestation.v0"
        || value_string(&attestation["canonicalization"], "attestation.canonicalization")?
            != "serde-json-map-sorted-compact-v0"
    {
        return Err("unsupported graph snapshot attestation or canonicalization version".into());
    }

    let expected_bindings = [
        ("snapshot_id", graph.snapshot.snapshot_id.as_str()),
        ("repository_commit", graph.snapshot.repository_commit.as_str()),
        ("environment_sha256", graph.snapshot.environment_sha256.as_str()),
        ("extractor_id", graph.snapshot.extractor_id.as_str()),
        ("extractor_version", graph.snapshot.extractor_version.as_str()),
        (
            "extractor_config_sha256",
            graph.snapshot.extractor_config_sha256.as_str(),
        ),
        ("graph_manifest_sha256", sha256_bytes(manifest_bytes).as_str()),
    ];
    for (field, expected) in expected_bindings {
        let observed = value_string(&attestation[field], &format!("attestation.{field}"))?;
        if observed != expected {
            return Err(format!("graph snapshot attestation does not bind {field}"));
        }
    }

    let signer_id = value_string(&attestation["signer_id"], "attestation.signer_id")?;
    let key_id = value_string(&attestation["key_id"], "attestation.key_id")?;
    if !non_empty(signer_id) || !non_empty(key_id) {
        return Err("attestation signer_id and key_id cannot be empty".into());
    }
    for key in keys {
        if value_string(&key["key_id"], "key.key_id")? == key_id {
            matching_key = Some(key);
            break;
        }
    }
    let key = matching_key.ok_or("attestation key_id is not present in the pinned registry")?;
    if value_string(&key["status"], "key.status")? != "active" {
        return Err("graph snapshot signing key is revoked".into());
    }
    if value_string(&key["signer_id"], "key.signer_id")? != signer_id {
        return Err("attestation signer identity does not match the pinned registry".into());
    }

    let public_key_hex =
        value_string(&key["public_key_ed25519_hex"], "key.public_key_ed25519_hex")?;
    let public_key_bytes =
        decode_hex_array::<32>(public_key_hex, "graph snapshot Ed25519 public key")?;
    let verifying_key = VerifyingKey::from_bytes(&public_key_bytes)
        .map_err(|error| format!("parse graph snapshot Ed25519 key: {error}"))?;
    let signature_hex = value_string(
        &attestation["signature_ed25519_hex"],
        "attestation.signature_ed25519_hex",
    )?;
    let signature_bytes =
        decode_hex_array::<64>(signature_hex, "graph snapshot Ed25519 signature")?;
    let signature = Signature::from_bytes(&signature_bytes);

    let mut signed_payload = attestation.clone();
    signed_payload
        .as_object_mut()
        .ok_or("graph snapshot attestation must be an object")?
        .remove("signature_ed25519_hex");
    // serde_json::Map uses lexical key order in this crate configuration;
    // the compact encoding is part of the signed v0 protocol.
    let canonical_payload = serde_json::to_vec(&signed_payload)
        .map_err(|error| format!("canonicalize graph snapshot attestation: {error}"))?;
    verifying_key
        .verify(&canonical_payload, &signature)
        .map_err(|error| format!("graph snapshot Ed25519 signature verification failed: {error}"))?;
    Ok(())
}

fn signature_negative_control(
    label: &str,
    manifest_bytes: &[u8],
    attestation_bytes: &[u8],
    registry_bytes: &[u8],
    pinned_registry_sha256: &str,
    graph: &Graph,
) -> Result<(), String> {
    if verify_snapshot_attestation(
        manifest_bytes,
        attestation_bytes,
        registry_bytes,
        pinned_registry_sha256,
        graph,
    )
    .is_ok()
    {
        return Err(format!("snapshot attestation accepted negative control: {label}"));
    }
    println!("SIGNATURE_NEGATIVE_CONTROL_PASS {label}");
    Ok(())
}

fn synthetic_signature_controls(manifest_bytes: &[u8], graph: &Graph) -> Result<(), String> {
    // Deterministic test-only seed. It is synthetic and never used as a
    // production signer; the public fixture stores only the test public key.
    let secret_seed = [0x42_u8; 32];
    let signing_key = SigningKey::from_bytes(&secret_seed);
    let public_key_hex = encode_hex(&signing_key.verifying_key().to_bytes());
    let registry = serde_json::json!({
        "schema_version": "pitb.graph-signer-registry.v0",
        "registry_id": "synthetic-registry-only",
        "keys": [{
            "key_id": "synthetic-test-key-1",
            "signer_id": "synthetic-test-extractor",
            "public_key_ed25519_hex": public_key_hex,
            "status": "active"
        }]
    });
    let registry_bytes = serde_json::to_vec(&registry)
        .map_err(|error| format!("serialize synthetic registry: {error}"))?;
    let registry_sha256 = sha256_bytes(&registry_bytes);

    let mut attestation = serde_json::json!({
        "schema_version": "pitb.graph-snapshot-attestation.v0",
        "canonicalization": "serde-json-map-sorted-compact-v0",
        "signer_id": "synthetic-test-extractor",
        "key_id": "synthetic-test-key-1",
        "graph_manifest_sha256": sha256_bytes(manifest_bytes),
        "snapshot_id": graph.snapshot.snapshot_id,
        "repository_commit": graph.snapshot.repository_commit,
        "environment_sha256": graph.snapshot.environment_sha256,
        "extractor_id": graph.snapshot.extractor_id,
        "extractor_version": graph.snapshot.extractor_version,
        "extractor_config_sha256": graph.snapshot.extractor_config_sha256,
        "signature_ed25519_hex": ""
    });
    let mut payload = attestation.clone();
    payload
        .as_object_mut()
        .ok_or("synthetic attestation must be an object")?
        .remove("signature_ed25519_hex");
    let canonical_payload = serde_json::to_vec(&payload)
        .map_err(|error| format!("serialize synthetic signed payload: {error}"))?;
    let signature = signing_key.sign(&canonical_payload);
    attestation["signature_ed25519_hex"] = Value::String(encode_hex(&signature.to_bytes()));
    let attestation_bytes = serde_json::to_vec(&attestation)
        .map_err(|error| format!("serialize synthetic signed attestation: {error}"))?;

    verify_snapshot_attestation(
        manifest_bytes,
        &attestation_bytes,
        &registry_bytes,
        &registry_sha256,
        graph,
    )?;
    println!("SNAPSHOT_SIGNATURE_PASS: exact manifest bytes and extractor identity are signed by an externally pinned key");

    let mut wrong_registry_hash = "0".repeat(64);
    if wrong_registry_hash == registry_sha256 {
        wrong_registry_hash.replace_range(..1, "1");
    }
    signature_negative_control(
        "wrong out-of-band registry digest",
        manifest_bytes,
        &attestation_bytes,
        &registry_bytes,
        &wrong_registry_hash,
        graph,
    )?;

    let mut changed_manifest = manifest_bytes.to_vec();
    changed_manifest.push(b' ');
    signature_negative_control(
        "manifest bytes changed after signing",
        &changed_manifest,
        &attestation_bytes,
        &registry_bytes,
        &registry_sha256,
        graph,
    )?;

    let mut revoked_registry = registry.clone();
    revoked_registry["keys"][0]["status"] = Value::String("revoked".into());
    let revoked_bytes = serde_json::to_vec(&revoked_registry)
        .map_err(|error| format!("serialize revoked registry: {error}"))?;
    let revoked_sha256 = sha256_bytes(&revoked_bytes);
    signature_negative_control(
        "revoked signing key",
        manifest_bytes,
        &attestation_bytes,
        &revoked_bytes,
        &revoked_sha256,
        graph,
    )?;

    let mut tampered_attestation = attestation.clone();
    tampered_attestation["extractor_version"] = Value::String("9.9.9".into());
    let tampered_bytes = serde_json::to_vec(&tampered_attestation)
        .map_err(|error| format!("serialize tampered attestation: {error}"))?;
    signature_negative_control(
        "extractor identity changed without re-signing",
        manifest_bytes,
        &tampered_bytes,
        &registry_bytes,
        &registry_sha256,
        graph,
    )?;

    let mut unknown = attestation;
    unknown["unknown_field"] = Value::Bool(true);
    let unknown_bytes = serde_json::to_vec(&unknown)
        .map_err(|error| format!("serialize unknown-field attestation: {error}"))?;
    signature_negative_control(
        "unknown attestation field",
        manifest_bytes,
        &unknown_bytes,
        &registry_bytes,
        &registry_sha256,
        graph,
    )?;
    Ok(())
}

fn verify_artifact_bytes(graph: &Graph, root: &Path) -> Result<(), String> {
    let canonical_root = fs::canonicalize(root)
        .map_err(|error| format!("cannot resolve evidence root: {error}"))?;
    if !canonical_root.is_dir() {
        return Err("evidence root is not a directory".into());
    }
    for artifact in &graph.artifacts {
        let path = resolve_artifact(&canonical_root, &artifact.path)?;
        let actual = sha256_file(&path)
            .map_err(|error| format!("cannot hash {}: {error}", artifact.path))?;
        if actual != artifact.sha256 {
            return Err(format!(
                "artifact {} digest mismatch: expected {}, got {}",
                artifact.artifact_id, artifact.sha256, actual
            ));
        }
        println!(
            "VERIFIED_ARTIFACT {} kind={} sha256={}",
            artifact.artifact_id, artifact.kind, actual
        );
    }
    Ok(())
}

fn must_reject(label: &str, value: &Value) -> Result<(), String> {
    if validate_value(value).is_ok() {
        return Err(format!("negative control was accepted: {label}"));
    }
    println!("NEGATIVE_CONTROL_PASS {label}");
    Ok(())
}

fn self_test(manifest_path: &Path, evidence_root: &Path) -> Result<(), String> {
    let bytes = fs::read(manifest_path)
        .map_err(|error| format!("cannot read fixture {}: {error}", manifest_path.display()))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("fixture JSON is invalid: {error}"))?;
    let (graph, summary) = validate_value(&value)?;
    verify_artifact_bytes(&graph, evidence_root)?;
    if summary.node_count != 7
        || summary.edge_count != 12
        || summary.formal_dependency_edge_count != 8
        || summary.module_environment_edge_count != 1
        || summary.retrieval_hint_edge_count != 4
        || summary.rejected_edge_count != 0
        || summary.scheduler_admission
    {
        return Err("synthetic graph fixture does not demonstrate the registered trust separation".into());
    }
    println!("PITB_GRAPH_EVIDENCE_FIXTURE_PASS: formal dependencies, proof support, instance/build edges, Blueprint prior and informal hints remain distinct");

    let mut bad = value.clone();
    bad["edges"][0]["evidence_class"] = Value::String("llm_alignment".into());
    must_reject("formal type edge cannot use LLM alignment evidence", &bad)?;

    let mut bad = value.clone();
    bad["edges"][5]["confidence"] = Value::from(1.01);
    must_reject("candidate confidence outside [0,1]", &bad)?;

    let mut bad = value.clone();
    bad["edges"][4]["review_artifact_id"] = Value::Null;
    bad["edges"][4]["reviewer_key_id"] = Value::Null;
    must_reject("human-reviewed edge without review artifact", &bad)?;

    let mut bad = value.clone();
    bad["edges"][5]["status"] = Value::String("observed".into());
    bad["edges"][5]["confidence"] = Value::Null;
    must_reject("informal extractor edge cannot be marked observed", &bad)?;

    let mut bad = value.clone();
    bad["nodes"][0]["statement_sha256"] = Value::String("0".repeat(64));
    must_reject("statement node digest differs from referenced artifact", &bad)?;

    let mut bad = value.clone();
    bad["artifacts"][0]["path"] = Value::String("../outside.txt".into());
    must_reject("artifact path traversal", &bad)?;

    // Mutual declaration references are valid elaborator observations. They
    // must not be collapsed into the acyclic helper-plan graph.
    let mut mutual = value.clone();
    let mut reverse_type = mutual["edges"][0].clone();
    reverse_type["edge_id"] = Value::String("mutual-type-reference".into());
    reverse_type["source_node_id"] = Value::String("helper-normalize".into());
    reverse_type["target_node_id"] = Value::String("target-main".into());
    mutual["edges"].as_array_mut().ok_or("fixture edges must be an array")?.push(reverse_type);
    let (_, mutual_summary) = validate_value(&mutual)?;
    if mutual_summary.formal_dependency_edge_count != 5 {
        return Err("mutual formal declaration references were not preserved".into());
    }
    println!("MUTUAL_FORMAL_REFERENCE_PASS: observed declaration SCCs remain representable");

    // Module/import dependencies remain acyclic. Add a second module and two
    // opposing build edges to make the invalid cycle explicit.
    let mut bad = value.clone();
    bad["nodes"].as_array_mut().ok_or("fixture nodes must be an array")?.push(serde_json::json!({
        "node_id":"module-child",
        "kind":"module",
        "label":"Demo child module",
        "statement_sha256":"9".repeat(64),
        "statement_artifact_id":"statement-module-child",
        "source_locator":"Demo/Child.lean",
        "formal_name":null
    }));
    bad["artifacts"].as_array_mut().ok_or("fixture artifacts must be an array")?.push(serde_json::json!({
        "artifact_id":"statement-module-child",
        "kind":"statement",
        "path":"statements/module-child.txt",
        "sha256":"9".repeat(64)
    }));
    bad["artifacts"].as_array_mut().ok_or("fixture artifacts must be an array")?.push(serde_json::json!({
        "artifact_id":"edge-module-forward",
        "kind":"edge_evidence",
        "path":"edges/module-forward.txt",
        "sha256":"a".repeat(64)
    }));
    bad["artifacts"].as_array_mut().ok_or("fixture artifacts must be an array")?.push(serde_json::json!({
        "artifact_id":"edge-module-reverse",
        "kind":"edge_evidence",
        "path":"edges/module-reverse.txt",
        "sha256":"b".repeat(64)
    }));
    for (edge_id, source, target, artifact_id) in [
        ("module-forward", "module-root", "module-child", "edge-module-forward"),
        ("module-reverse", "module-child", "module-root", "edge-module-reverse"),
    ] {
        bad["edges"].as_array_mut().ok_or("fixture edges must be an array")?.push(serde_json::json!({
            "edge_id":edge_id,
            "source_node_id":source,
            "target_node_id":target,
            "relation":"module_environment_dependency",
            "evidence_class":"lean_build_graph",
            "status":"observed",
            "evidence_artifact_id":artifact_id,
            "confidence":null,
            "review_artifact_id":null,
            "reviewer_key_id":null
        }));
    }
    must_reject("module environment dependency cycle", &bad)?;

    let mut bad = value.clone();
    bad["unregistered_field"] = Value::Bool(true);
    must_reject("unknown top-level field", &bad)?;

    println!("SELF_TEST_OK");
    Ok(())
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 || !matches!(args[1].as_str(), "--verify" | "--self-test") {
        return Err(format!(
            "usage: {} --verify|--self-test <graph.json> <evidence-root>",
            args.first().map(String::as_str).unwrap_or("pitb-graph-evidence")
        ));
    }
    let manifest_path = Path::new(&args[2]);
    let evidence_root = Path::new(&args[3]);
    let bytes = fs::read(manifest_path)
        .map_err(|error| format!("cannot read graph manifest: {error}"))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("graph JSON is invalid: {error}"))?;
    let (graph, summary) = validate_value(&value)?;
    verify_artifact_bytes(&graph, evidence_root)?;
    if args[1] == "--self-test" {
        self_test(manifest_path, evidence_root)?;
    } else {
        println!(
            "{}",
            serde_json::to_string_pretty(&summary)
                .map_err(|error| format!("cannot serialize summary: {error}"))?
        );
        println!("PITB_GRAPH_EVIDENCE_INTEGRITY_OK");
    }
    Ok(())
}

fn main() -> ExitCode {
    match execute() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => fail(message),
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_paths_reject_escape_and_absolute_paths() {
        assert!(safe_relative_path("statements/theorem.lean").is_ok());
        assert!(safe_relative_path("../outside").is_err());
        assert!(safe_relative_path("/tmp/outside").is_err());
        assert!(safe_relative_path("a/./b").is_err());
        assert!(safe_relative_path("a//b").is_err());
        assert!(safe_relative_path("").is_err());
    }

    #[test]
    fn relation_to_evidence_class_mapping_is_explicit() {
        assert_eq!(expected_evidence_class("type_reference"), Some("lean_elaborator"));
        assert_eq!(expected_evidence_class("instance_synthesis"), Some("lean_instance_trace"));
        assert_eq!(
            expected_evidence_class("cross_formality_match"),
            Some("llm_alignment")
        );
        for relation in [
            "signature_reference",
            "proof_term_reference",
            "definition_body_reference",
            "structure_field_reference",
            "inheritance_reference",
        ] {
            assert_eq!(expected_evidence_class(relation), Some("lean_elaborator"));
            assert!(is_formal_dependency(relation));
        }
        assert_eq!(expected_evidence_class("documentation_reference"), Some("lean_doc_parser"));
        assert!(!is_formal_dependency("documentation_reference"));
        assert!(is_observed_relation("documentation_reference"));
        assert_eq!(expected_evidence_class("invented_relation"), None);
    }

    #[test]
    fn candidate_and_observed_confidence_policies_are_separate() {
        let make_edge = |status: &str, confidence: Value| Edge {
            edge_id: "edge-test".into(),
            source_node_id: "source".into(),
            target_node_id: "target".into(),
            relation: "informal_candidate_dependency".into(),
            evidence_class: "informal_extractor".into(),
            status: status.into(),
            evidence_artifact_id: "evidence".into(),
            confidence,
            review_artifact_id: Value::Null,
            reviewer_key_id: Value::Null,
        };
        assert!(validate_confidence(&make_edge("candidate", Value::from(0.7))).is_ok());
        assert!(validate_confidence(&make_edge("candidate", Value::from(1.01))).is_err());
        assert!(validate_confidence(&make_edge("candidate", Value::Null)).is_err());
        assert!(validate_confidence(&make_edge("observed", Value::Null)).is_ok());
        assert!(validate_confidence(&make_edge("observed", Value::from(0.9))).is_err());
    }

    #[test]
    fn only_registered_module_graph_must_be_acyclic() {
        let mut graph = BTreeMap::new();
        graph.insert("module-a".to_owned(), BTreeSet::from(["module-b".to_owned()]));
        let mut states = BTreeMap::new();
        assert!(visit_dependency("module-a", &graph, &mut states).is_ok());

        let mut cyclic = BTreeMap::new();
        cyclic.insert("module-a".to_owned(), BTreeSet::from(["module-b".to_owned()]));
        cyclic.insert("module-b".to_owned(), BTreeSet::from(["module-a".to_owned()]));
        assert!(visit_dependency("module-a", &cyclic, &mut BTreeMap::new()).is_err());
    }

    #[test]
    fn formal_and_candidate_edge_roles_never_collapse() {
        let observed = Edge {
            edge_id: "formal".into(),
            source_node_id: "a".into(),
            target_node_id: "b".into(),
            relation: "proof_term_reference".into(),
            evidence_class: "lean_elaborator".into(),
            status: "observed".into(),
            evidence_artifact_id: "proof".into(),
            confidence: Value::Null,
            review_artifact_id: Value::Null,
            reviewer_key_id: Value::Null,
        };
        let candidate = Edge {
            edge_id: "candidate".into(),
            source_node_id: "a".into(),
            target_node_id: "b".into(),
            relation: "cross_formality_match".into(),
            evidence_class: "llm_alignment".into(),
            status: "candidate".into(),
            evidence_artifact_id: "match".into(),
            confidence: Value::from(0.95),
            review_artifact_id: Value::Null,
            reviewer_key_id: Value::Null,
        };
        assert_eq!(expected_role(&observed), "proof_support_observation_hash_bound");
        assert_eq!(expected_role(&candidate), "candidate_retrieval_hint");
        assert_ne!(is_formal_dependency(&observed.relation), is_formal_dependency(&candidate.relation));
    }

    #[test]
    fn reviewer_artifact_and_key_are_required_as_a_pair() {
        assert!(check_optional_pair(
            &Value::String("review-1".into()),
            &Value::String("key-1".into()),
            true,
            "edge-1"
        )
        .is_ok());
        assert!(check_optional_pair(&Value::Null, &Value::Null, true, "edge-1").is_err());
        assert!(check_optional_pair(
            &Value::String("review-1".into()),
            &Value::Null,
            false,
            "edge-1"
        )
        .is_err());
    }
}
