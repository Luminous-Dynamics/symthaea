//! Independent structural/provenance verifier for enzyme evidence graphs.
//!
//! This is deliberately not a biochemical validator, license adjudicator, or model evaluator.
//! It verifies graph structure and the exact bytes of artifacts already present in the workspace;
//! it never downloads remote source URLs.
use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::{Component, Path, PathBuf};
use url::Url;

const NODE_TYPES: &[(&str, &str)] = &[
    ("source_record", "source_claim"),
    ("raw_signal_series", "observed"),
    ("control_series", "observed"),
    ("calibration", "derived_or_source_reported"),
    ("processed_measurement", "derived"),
    ("rate_estimate", "derived_or_fitted"),
    ("model_specification", "model_assumption"),
    ("fit_run", "computation"),
    ("reported_parameter_claim", "source_claim"),
    ("reproduced_parameter_estimate", "reproduced_computation"),
    ("prediction", "prediction"),
    ("evaluation", "evaluation"),
];

const ARTIFACT_ROLES: &[&str] = &[
    "source_record",
    "raw_signal",
    "control_signal",
    "calibration",
    "processed_measurement",
    "model_specification",
    "fit_output",
    "reported_parameter_table",
    "prediction_output",
    "evaluation_output",
    "supporting_metadata",
    "transformation_code",
    "transformation_configuration",
    "environment_lock",
];

const EDGE_TYPES: &[&str] = &[
    "describes_source",
    "measured_as",
    "calibrated_by",
    "corrected_with",
    "derived_from",
    "fitted_using",
    "estimates",
    "reported_by",
    "predicts",
    "evaluated_by",
];

const TERMINAL_STATUSES: &[&str] = &[
    "success",
    "null_result",
    "failed",
    "excluded",
    "out_of_domain",
    "blocked",
];

const REQUIRED_TRANSFORMED_NODE_TYPES: &[&str] = &[
    "processed_measurement",
    "rate_estimate",
    "fit_run",
    "reproduced_parameter_estimate",
    "prediction",
    "evaluation",
];

const UNCERTAINTY_STATUSES: &[&str] = &[
    "reported",
    "recomputed",
    "not_available",
    "not_estimable",
    "not_applicable",
];

const GRAPH_STATUSES: &[&str] = &["draft", "frozen"];
const RIGHTS_STATUSES: &[&str] = &[
    "approved_for_this_use",
    "review_pending",
    "restricted",
    "rejected",
    "unknown",
];
const ACQUISITION_STATUSES: &[&str] = &["available", "not_acquired"];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceGraph {
    schema_version: u32,
    graph_id: String,
    status: String,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    artifacts: Vec<ArtifactRecord>,
    rights_records: Vec<RightsRecord>,
    transformations: Vec<Transformation>,
    #[serde(default)]
    parameter_claims: Vec<ParameterClaim>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Node {
    node_id: String,
    #[serde(rename = "type")]
    node_type: String,
    evidence_class: String,
    #[serde(default)]
    artifact_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Edge {
    edge_id: String,
    #[serde(rename = "type")]
    edge_type: String,
    from_node_id: String,
    to_node_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactRecord {
    artifact_id: String,
    role: String,
    media_type: String,
    sha256: String,
    byte_length: u64,
    locator: String,
    rights_record_id: String,
    /// "available" means a workspace-local payload that this checker must verify.
    /// "not_acquired" means metadata only; remote content is never fetched.
    acquisition_status: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RightsRecord {
    rights_record_id: String,
    source_id: String,
    license_id: String,
    attribution: String,
    status: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Transformation {
    transformation_id: String,
    name: String,
    version: String,
    code_sha256: String,
    configuration_sha256: String,
    environment_lock_sha256: String,
    /// Optional for unsuccessful/blocked runs; required and byte-verified for success.
    #[serde(default)]
    code_artifact_id: Option<String>,
    /// Exact configuration bytes used by the transform, when available.
    #[serde(default)]
    configuration_artifact_id: Option<String>,
    /// Environment lockfile/container definition bytes, when available.
    #[serde(default)]
    environment_lock_artifact_id: Option<String>,
    input_node_ids: Vec<String>,
    output_node_ids: Vec<String>,
    input_artifact_ids: Vec<String>,
    output_artifact_ids: Vec<String>,
    deterministic: bool,
    terminal_status: String,
    #[serde(default)]
    terminal_reason: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct ParameterClaim {
    parameter_id: String,
    parameter_name: String,
    endpoint_definition: String,
    value: Value,
    unit: String,
    /// Optional normalized value; when present it must be numeric and paired with normalized_unit.
    #[serde(default)]
    normalized_value: Option<Value>,
    /// Optional normalized unit; when present it must be paired with normalized_value.
    #[serde(default)]
    normalized_unit: Option<String>,
    enzyme_form_basis: String,
    reaction_id: String,
    substrate_ids: Vec<String>,
    assay_context: Value,
    kinetic_equation_id: String,
    source_or_fit_node_id: String,
    uncertainty_status: String,
}

#[derive(Debug)]
struct Report {
    artifacts_verified: usize,
    artifacts_total: usize,
    transformations_successful: usize,
    transformations_terminal_non_success: usize,
    parameter_claims: usize,
    rights_review_unresolved: usize,
}

pub fn check(workspace_root: &Path, input: &Path, expected_sha256: Option<&str>) -> Result<()> {
    let root = fs::canonicalize(workspace_root)
        .with_context(|| format!("canonicalize workspace root {}", workspace_root.display()))?;
    let input_candidate = if input.is_absolute() {
        input.to_path_buf()
    } else {
        root.join(input)
    };
    let input_path = fs::canonicalize(&input_candidate)
        .with_context(|| format!("canonicalize evidence graph {}", input_candidate.display()))?;
    ensure!(
        input_path.starts_with(&root),
        "evidence graph input escapes workspace root: {}",
        input_path.display()
    );
    let bytes = fs::read(&input_path)
        .with_context(|| format!("read evidence graph {}", input_path.display()))?;
    let graph: EvidenceGraph = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse evidence graph JSON {}", input_path.display()))?;

    let actual_graph_sha256 = sha256(&bytes);
    if let Some(expected) = expected_sha256 {
        ensure!(is_sha256(expected), "--sha256 must be 64 hexadecimal characters");
        ensure!(
            actual_graph_sha256.eq_ignore_ascii_case(expected),
            "evidence graph SHA-256 mismatch: expected {}, got {}",
            expected,
            actual_graph_sha256
        );
    }
    ensure!(
        graph.status != "frozen" || expected_sha256.is_some(),
        "a frozen graph requires an external --sha256 pin for the exact serialized bytes"
    );

    let report = validate_graph(&root, &graph)?;
    println!(
        "Enzyme evidence graph verified: id={}, status={}, nodes={}, edges={}, artifacts_verified={}/{}, successful_transformations={}, terminal_non_success_transformations={}, parameter_claims={}, rights_reviews_unresolved={}",
        graph.graph_id,
        graph.status,
        graph.nodes.len(),
        graph.edges.len(),
        report.artifacts_verified,
        report.artifacts_total,
        report.transformations_successful,
        report.transformations_terminal_non_success,
        report.parameter_claims,
        report.rights_review_unresolved
    );
    if graph.status == "frozen" {
        println!("Serialized graph SHA-256: {actual_graph_sha256}");
    }
    println!(
        "Scope: structural/provenance integrity only; no claim of biochemical validity, license clearance, or predictive performance."
    );
    Ok(())
}

fn validate_graph(root: &Path, graph: &EvidenceGraph) -> Result<Report> {
    ensure!(graph.schema_version == 1, "unsupported evidence graph schema_version");
    ensure!(!graph.graph_id.trim().is_empty(), "graph_id must be nonempty");
    ensure!(
        GRAPH_STATUSES.contains(&graph.status.as_str()),
        "graph status must be one of {:?}",
        GRAPH_STATUSES
    );

    let rights_by_id = unique_index(
        graph.rights_records.iter().map(|item| (item.rights_record_id.as_str(), item)),
        "rights_record_id",
    )?;
    for rights in &graph.rights_records {
        ensure!(!rights.source_id.trim().is_empty(), "rights record {} has empty source_id", rights.rights_record_id);
        ensure!(!rights.license_id.trim().is_empty(), "rights record {} has empty license_id", rights.rights_record_id);
        ensure!(!rights.attribution.trim().is_empty(), "rights record {} has empty attribution", rights.rights_record_id);
        ensure!(
            RIGHTS_STATUSES.contains(&rights.status.as_str()),
            "rights record {} has unsupported status '{}'",
            rights.rights_record_id,
            rights.status
        );
    }

    let nodes_by_id = unique_index(
        graph.nodes.iter().map(|item| (item.node_id.as_str(), item)),
        "node_id",
    )?;
    for node in &graph.nodes {
        ensure!(!node.node_id.trim().is_empty(), "node_id must be nonempty");
        let expected_class = NODE_TYPES
            .iter()
            .find(|(node_type, _)| *node_type == node.node_type)
            .map(|(_, class)| *class)
            .with_context(|| format!("node {} has unknown type '{}'", node.node_id, node.node_type))?;
        ensure!(
            node.evidence_class == expected_class,
            "node {} type '{}' requires evidence_class '{}', got '{}'",
            node.node_id,
            node.node_type,
            expected_class,
            node.evidence_class
        );
        ensure_unique_values(&node.artifact_ids, &format!("node {} artifact_ids", node.node_id))?;
    }

    let artifacts_by_id = unique_index(
        graph.artifacts.iter().map(|item| (item.artifact_id.as_str(), item)),
        "artifact_id",
    )?;
    let mut verified_artifacts = BTreeMap::<String, bool>::new();
    let mut rights_review_unresolved = 0usize;
    for artifact in &graph.artifacts {
        ensure!(!artifact.artifact_id.trim().is_empty(), "artifact_id must be nonempty");
        ensure!(
            ARTIFACT_ROLES.contains(&artifact.role.as_str()),
            "artifact {} has unknown role '{}'",
            artifact.artifact_id,
            artifact.role
        );
        ensure!(!artifact.media_type.trim().is_empty(), "artifact {} media_type must be nonempty", artifact.artifact_id);
        ensure!(is_sha256(&artifact.sha256), "artifact {} sha256 must be 64 hexadecimal characters", artifact.artifact_id);
        ensure!(!artifact.locator.trim().is_empty(), "artifact {} locator must be nonempty", artifact.artifact_id);
        ensure!(
            rights_by_id.contains_key(artifact.rights_record_id.as_str()),
            "artifact {} references missing rights_record_id {}",
            artifact.artifact_id,
            artifact.rights_record_id
        );
        let rights = rights_by_id[artifact.rights_record_id.as_str()];
        if rights.status != "approved_for_this_use" {
            rights_review_unresolved += 1;
        }

        ensure!(
            ACQUISITION_STATUSES.contains(&artifact.acquisition_status.as_str()),
            "artifact {} has unsupported acquisition_status '{}'",
            artifact.artifact_id,
            artifact.acquisition_status
        );
        let verified = match artifact.acquisition_status.as_str() {
            "available" => {
                let relative_path = safe_relative_path(&artifact.locator)
                    .with_context(|| format!("invalid workspace locator for artifact {}", artifact.artifact_id))?;
                let candidate = root.join(relative_path);
                let canonical_path = fs::canonicalize(&candidate)
                    .with_context(|| format!("canonicalize artifact {} at {}", artifact.artifact_id, candidate.display()))?;
                ensure!(
                    canonical_path.starts_with(root),
                    "artifact {} escapes workspace root through a symlink",
                    artifact.artifact_id
                );
                let payload = fs::read(&canonical_path)
                    .with_context(|| format!("read artifact {}", artifact.artifact_id))?;
                ensure!(
                    payload.len() as u64 == artifact.byte_length,
                    "artifact {} byte_length mismatch: declared {}, actual {}",
                    artifact.artifact_id,
                    artifact.byte_length,
                    payload.len()
                );
                let actual = sha256(&payload);
                ensure!(
                    actual.eq_ignore_ascii_case(&artifact.sha256),
                    "artifact {} SHA-256 mismatch: declared {}, actual {}",
                    artifact.artifact_id,
                    artifact.sha256,
                    actual
                );
                true
            }
            "not_acquired" => {
                let url = Url::parse(&artifact.locator)
                    .with_context(|| format!("artifact {} locator must be an HTTPS URL when not acquired", artifact.artifact_id))?;
                ensure!(
                    url.scheme() == "https" && url.host_str().is_some(),
                    "artifact {} remote locator must use HTTPS",
                    artifact.artifact_id
                );
                false
            }
            other => bail!(
                "artifact {} has unsupported acquisition_status '{}'",
                artifact.artifact_id,
                other
            ),
        };
        verified_artifacts.insert(artifact.artifact_id.clone(), verified);
    }

    for node in &graph.nodes {
        for artifact_id in &node.artifact_ids {
            ensure!(
                artifacts_by_id.contains_key(artifact_id.as_str()),
                "node {} references missing artifact {}",
                node.node_id,
                artifact_id
            );
        }
    }

    let mut seen_edge_ids = BTreeSet::<String>::new();
    let mut seen_edge_pairs = BTreeSet::<(String, String, String)>::new();
    for edge in &graph.edges {
        ensure!(!edge.edge_id.trim().is_empty(), "edge_id must be nonempty");
        ensure!(seen_edge_ids.insert(edge.edge_id.clone()), "duplicate edge_id {}", edge.edge_id);
        ensure!(
            EDGE_TYPES.contains(&edge.edge_type.as_str()),
            "edge {} has unknown type '{}'",
            edge.edge_id,
            edge.edge_type
        );
        let from = nodes_by_id.get(edge.from_node_id.as_str()).with_context(|| {
            format!("edge {} has dangling from_node_id {}", edge.edge_id, edge.from_node_id)
        })?;
        let to = nodes_by_id.get(edge.to_node_id.as_str()).with_context(|| {
            format!("edge {} has dangling to_node_id {}", edge.edge_id, edge.to_node_id)
        })?;
        ensure!(
            edge.from_node_id != edge.to_node_id,
            "edge {} cannot point to itself",
            edge.edge_id
        );
        ensure!(
            edge_endpoints_valid(&edge.edge_type, &from.node_type, &to.node_type),
            "edge {} type '{}' does not permit {} -> {}",
            edge.edge_id,
            edge.edge_type,
            from.node_type,
            to.node_type
        );
        ensure!(
            seen_edge_pairs.insert((
                edge.edge_type.clone(),
                edge.from_node_id.clone(),
                edge.to_node_id.clone()
            )),
            "duplicate edge relation {}: {} -> {}",
            edge.edge_type,
            edge.from_node_id,
            edge.to_node_id
        );
    }
    ensure_acyclic(&graph.nodes, &graph.edges)?;

    let mut transformations_by_id = BTreeMap::<String, &Transformation>::new();
    for transformation in &graph.transformations {
        ensure!(!transformation.transformation_id.trim().is_empty(), "transformation_id must be nonempty");
        ensure!(
            transformations_by_id
                .insert(transformation.transformation_id.clone(), transformation)
                .is_none(),
            "duplicate transformation_id {}",
            transformation.transformation_id
        );
        ensure!(!transformation.name.trim().is_empty(), "transformation {} name must be nonempty", transformation.transformation_id);
        ensure!(!transformation.version.trim().is_empty(), "transformation {} version must be nonempty", transformation.transformation_id);
        ensure!(is_sha256(&transformation.code_sha256), "transformation {} code_sha256 is invalid", transformation.transformation_id);
        ensure!(is_sha256(&transformation.configuration_sha256), "transformation {} configuration_sha256 is invalid", transformation.transformation_id);
        ensure!(is_sha256(&transformation.environment_lock_sha256), "transformation {} environment_lock_sha256 is invalid", transformation.transformation_id);

        for (identity_kind, artifact_id, expected_sha256, expected_role) in [
            ("code", transformation.code_artifact_id.as_deref(), transformation.code_sha256.as_str(), "transformation_code"),
            ("configuration", transformation.configuration_artifact_id.as_deref(), transformation.configuration_sha256.as_str(), "transformation_configuration"),
            ("environment lock", transformation.environment_lock_artifact_id.as_deref(), transformation.environment_lock_sha256.as_str(), "environment_lock"),
        ] {
            if let Some(artifact_id) = artifact_id {
                let artifact = artifacts_by_id.get(artifact_id).with_context(|| {
                    format!(
                        "transformation {} references missing {} artifact {}",
                        transformation.transformation_id,
                        identity_kind,
                        artifact_id
                    )
                })?;
                ensure!(
                    artifact.role == expected_role,
                    "transformation {} {} artifact {} must have role '{}'",
                    transformation.transformation_id,
                    identity_kind,
                    artifact_id,
                    expected_role
                );
                ensure!(
                    artifact.sha256.eq_ignore_ascii_case(expected_sha256),
                    "transformation {} {} SHA-256 does not match referenced artifact {}",
                    transformation.transformation_id,
                    identity_kind,
                    artifact_id
                );
            }
        }

        ensure!(
            TERMINAL_STATUSES.contains(&transformation.terminal_status.as_str()),
            "transformation {} has unknown terminal_status '{}'",
            transformation.transformation_id,
            transformation.terminal_status
        );
        if transformation.terminal_status != "success" {
            ensure!(
                transformation.terminal_reason.as_deref().is_some_and(|reason| !reason.trim().is_empty()),
                "non-success transformation {} must preserve a terminal_reason",
                transformation.transformation_id
            );
        }
        ensure_unique_values(&transformation.input_node_ids, "transformation input_node_ids")?;
        ensure_unique_values(&transformation.output_node_ids, "transformation output_node_ids")?;
        ensure_unique_values(&transformation.input_artifact_ids, "transformation input_artifact_ids")?;
        ensure_unique_values(&transformation.output_artifact_ids, "transformation output_artifact_ids")?;

        for node_id in &transformation.input_node_ids {
            ensure!(
                nodes_by_id.contains_key(node_id.as_str()),
                "transformation {} references missing input node {}",
                transformation.transformation_id,
                node_id
            );
        }
        for node_id in &transformation.output_node_ids {
            ensure!(
                nodes_by_id.contains_key(node_id.as_str()),
                "transformation {} references missing output node {}",
                transformation.transformation_id,
                node_id
            );
        }
        for artifact_id in &transformation.input_artifact_ids {
            ensure!(
                artifacts_by_id.contains_key(artifact_id.as_str()),
                "transformation {} references missing input artifact {}",
                transformation.transformation_id,
                artifact_id
            );
        }
        for artifact_id in &transformation.output_artifact_ids {
            ensure!(
                artifacts_by_id.contains_key(artifact_id.as_str()),
                "transformation {} references missing output artifact {}",
                transformation.transformation_id,
                artifact_id
            );
        }

        let input_artifacts_linked = transformation.input_artifact_ids.iter().all(|artifact_id| {
            transformation.input_node_ids.iter().any(|node_id| {
                nodes_by_id[node_id.as_str()].artifact_ids.contains(artifact_id)
            })
        });
        let output_artifacts_linked = transformation.output_artifact_ids.iter().all(|artifact_id| {
            transformation.output_node_ids.iter().any(|node_id| {
                nodes_by_id[node_id.as_str()].artifact_ids.contains(artifact_id)
            })
        });
        ensure!(
            transformation.input_artifact_ids.is_empty() || input_artifacts_linked,
            "transformation {} input artifacts are not linked from its input nodes",
            transformation.transformation_id
        );
        ensure!(
            transformation.output_artifact_ids.is_empty() || output_artifacts_linked,
            "transformation {} output artifacts are not linked from its output nodes",
            transformation.transformation_id
        );

        // A null/failed/blocked computation can legitimately have no output node.
        // When outputs are declared, however, every input and output must be tied together
        // by a contract-recognized dependency edge.
        if !transformation.output_node_ids.is_empty() {
            for input_node_id in &transformation.input_node_ids {
                ensure!(
                    transformation.output_node_ids.iter().any(|output_node_id| {
                        graph.edges.iter().any(|edge| {
                            edge.from_node_id == *input_node_id
                                && edge.to_node_id == *output_node_id
                                && is_transformation_dependency_edge(&edge.edge_type)
                        })
                    }),
                    "transformation {} input node {} has no explicit edge to any declared output node",
                    transformation.transformation_id,
                    input_node_id
                );
            }
            for output_node_id in &transformation.output_node_ids {
                ensure!(
                    transformation.input_node_ids.iter().any(|input_node_id| {
                        graph.edges.iter().any(|edge| {
                            edge.from_node_id == *input_node_id
                                && edge.to_node_id == *output_node_id
                                && is_transformation_dependency_edge(&edge.edge_type)
                        })
                    }),
                    "transformation {} output node {} has no explicit incoming dependency from its declared inputs",
                    transformation.transformation_id,
                    output_node_id
                );
            }
        }

        if transformation.terminal_status == "success" {
            ensure!(
                !transformation.input_node_ids.is_empty(),
                "successful transformation {} has no input nodes",
                transformation.transformation_id
            );
            ensure!(
                !transformation.output_node_ids.is_empty(),
                "successful transformation {} has no output nodes",
                transformation.transformation_id
            );
            ensure!(
                !transformation.input_artifact_ids.is_empty(),
                "successful transformation {} has no input artifacts",
                transformation.transformation_id
            );
            ensure!(
                !transformation.output_artifact_ids.is_empty(),
                "successful transformation {} has no output artifacts",
                transformation.transformation_id
            );
            ensure!(
                transformation.output_artifact_ids.iter().all(|id| verified_artifacts.get(id) == Some(&true)),
                "successful transformation {} has one or more unavailable or unverified output artifacts",
                transformation.transformation_id
            );
            ensure!(
                transformation.input_artifact_ids.iter().all(|id| verified_artifacts.get(id) == Some(&true)),
                "successful transformation {} has one or more unavailable or unverified input artifacts",
                transformation.transformation_id
            );
            let code_id = transformation.code_artifact_id.as_ref().with_context(|| {
                format!("successful transformation {} is missing code_artifact_id", transformation.transformation_id)
            })?;
            let configuration_id = transformation.configuration_artifact_id.as_ref().with_context(|| {
                format!("successful transformation {} is missing configuration_artifact_id", transformation.transformation_id)
            })?;
            let environment_id = transformation.environment_lock_artifact_id.as_ref().with_context(|| {
                format!("successful transformation {} is missing environment_lock_artifact_id", transformation.transformation_id)
            })?;
            let identity_artifact_ids = [code_id, configuration_id, environment_id];
            for artifact_id in identity_artifact_ids {
                ensure!(
                    verified_artifacts.get(artifact_id) == Some(&true),
                    "successful transformation {} has unavailable or unverified execution-identity artifact {}",
                    transformation.transformation_id,
                    artifact_id
                );
            }
            for artifact_id in transformation
                .input_artifact_ids
                .iter()
                .chain(&transformation.output_artifact_ids)
                .chain(transformation.code_artifact_id.iter())
                .chain(transformation.configuration_artifact_id.iter())
                .chain(transformation.environment_lock_artifact_id.iter())
            {
                let artifact = artifacts_by_id[artifact_id.as_str()];
                let rights = rights_by_id[artifact.rights_record_id.as_str()];
                ensure!(
                    rights.status == "approved_for_this_use",
                    "successful transformation {} cannot consume or publish artifact {} with rights status '{}'; preserve the run as blocked until its source-specific use is reviewed",
                    transformation.transformation_id,
                    artifact_id,
                    rights.status
                );
            }
        }
        if transformation.terminal_status == "null_result" {
            ensure!(
                transformation.output_node_ids.is_empty() && transformation.output_artifact_ids.is_empty(),
                "null_result transformation {} must not claim output nodes or artifacts",
                transformation.transformation_id
            );
        }
    }

    for node in &graph.nodes {
        if REQUIRED_TRANSFORMED_NODE_TYPES.contains(&node.node_type.as_str()) {
            ensure!(
                graph.transformations.iter().any(|transformation| {
                    transformation.output_node_ids.contains(&node.node_id)
                        && !transformation.input_node_ids.is_empty()
                }),
                "derived/computed node {} ({}) has no matching transformation record with explicit inputs",
                node.node_id,
                node.node_type
            );
        }
    }

    // Every declared artifact must participate in evidence or execution provenance.
    // This prevents unrelated payloads from being silently included in an otherwise valid graph.
    let mut referenced_artifact_ids = BTreeSet::<String>::new();
    for node in &graph.nodes {
        referenced_artifact_ids.extend(node.artifact_ids.iter().cloned());
    }
    for transformation in &graph.transformations {
        referenced_artifact_ids.extend(transformation.input_artifact_ids.iter().cloned());
        referenced_artifact_ids.extend(transformation.output_artifact_ids.iter().cloned());
        referenced_artifact_ids.extend(transformation.code_artifact_id.iter().cloned());
        referenced_artifact_ids.extend(transformation.configuration_artifact_id.iter().cloned());
        referenced_artifact_ids.extend(transformation.environment_lock_artifact_id.iter().cloned());
    }
    for artifact in &graph.artifacts {
        ensure!(
            referenced_artifact_ids.contains(&artifact.artifact_id),
            "artifact {} is unreferenced by every evidence node and transformation",
            artifact.artifact_id
        );
    }

    validate_parameter_claims(&graph.parameter_claims, &nodes_by_id, &graph.edges)?;

    Ok(Report {
        artifacts_verified: verified_artifacts.values().filter(|verified| **verified).count(),
        artifacts_total: graph.artifacts.len(),
        transformations_successful: graph.transformations.iter().filter(|item| item.terminal_status == "success").count(),
        transformations_terminal_non_success: graph.transformations.iter().filter(|item| item.terminal_status != "success").count(),
        parameter_claims: graph.parameter_claims.len(),
        rights_review_unresolved,
    })
}

fn validate_parameter_claims(
    claims: &[ParameterClaim],
    nodes_by_id: &BTreeMap<String, &Node>,
    edges: &[Edge],
) -> Result<()> {
    let mut seen = BTreeSet::<String>::new();
    for claim in claims {
        ensure!(!claim.parameter_id.trim().is_empty(), "parameter_id must be nonempty");
        ensure!(seen.insert(claim.parameter_id.clone()), "duplicate parameter_id {}", claim.parameter_id);
        ensure!(!claim.parameter_name.trim().is_empty(), "parameter {} name must be nonempty", claim.parameter_id);
        ensure!(!claim.endpoint_definition.trim().is_empty(), "parameter {} endpoint_definition must be nonempty", claim.parameter_id);
        ensure!(!claim.unit.trim().is_empty(), "parameter {} unit must be nonempty", claim.parameter_id);
        ensure!(!claim.enzyme_form_basis.trim().is_empty(), "parameter {} enzyme_form_basis must be explicit", claim.parameter_id);
        ensure!(!claim.reaction_id.trim().is_empty(), "parameter {} reaction_id must be nonempty", claim.parameter_id);
        ensure!(!claim.kinetic_equation_id.trim().is_empty(), "parameter {} kinetic_equation_id must be nonempty", claim.parameter_id);
        ensure!(
            claim.value.is_number()
                || claim.value.as_str().is_some_and(|value| !value.trim().is_empty()),
            "parameter {} value must be a number or non-empty verbatim source string",
            claim.parameter_id
        );
        ensure!(
            match (&claim.normalized_value, &claim.normalized_unit) {
                (None, None) => true,
                (Some(value), Some(unit)) => value.is_number() && !unit.trim().is_empty(),
                _ => false,
            },
            "parameter {} normalized_value and normalized_unit must be supplied together, with a numeric value and non-empty unit",
            claim.parameter_id
        );
        ensure!(
            claim.assay_context.is_object() && !claim.assay_context.as_object().unwrap().is_empty(),
            "parameter {} assay_context must be a non-empty object; represent missing conditions explicitly as null/status fields",
            claim.parameter_id
        );
        ensure!(
            !claim.substrate_ids.is_empty(),
            "parameter {} must identify at least one substrate; unknown substrate identity must be explicit rather than an empty list",
            claim.parameter_id
        );
        ensure_unique_values(&claim.substrate_ids, &format!("parameter {} substrate_ids", claim.parameter_id));
        ensure!(
            UNCERTAINTY_STATUSES.contains(&claim.uncertainty_status.as_str()),
            "parameter {} has unknown uncertainty_status '{}'",
            claim.parameter_id,
            claim.uncertainty_status
        );
        let source_or_fit = nodes_by_id
            .get(claim.source_or_fit_node_id.as_str())
            .with_context(|| format!("parameter {} references missing source_or_fit_node_id {}", claim.parameter_id, claim.source_or_fit_node_id))?;
        match source_or_fit.node_type.as_str() {
            "source_record" => ensure!(
                edges.iter().any(|edge| {
                    edge.edge_type == "reported_by"
                        && edge.from_node_id == claim.source_or_fit_node_id
                        && nodes_by_id
                            .get(edge.to_node_id.as_str())
                            .is_some_and(|node| node.node_type == "reported_parameter_claim")
                }),
                "parameter {} references source record {} without a reported_by edge to a reported_parameter_claim node",
                claim.parameter_id,
                claim.source_or_fit_node_id
            ),
            "fit_run" => ensure!(
                edges.iter().any(|edge| {
                    edge.edge_type == "estimates"
                        && edge.from_node_id == claim.source_or_fit_node_id
                        && nodes_by_id
                            .get(edge.to_node_id.as_str())
                            .is_some_and(|node| node.node_type == "reproduced_parameter_estimate")
                }),
                "parameter {} references fit_run {} without an estimates edge to a reproduced_parameter_estimate node",
                claim.parameter_id,
                claim.source_or_fit_node_id
            ),
            "reported_parameter_claim" | "reproduced_parameter_estimate" => {}
            other => bail!(
                "parameter {} source_or_fit_node_id must identify a source/fit record with a claim edge or a parameter-claim node, got {}",
                claim.parameter_id,
                other
            ),
        }
    }
    Ok(())
}

fn edge_endpoints_valid(edge_type: &str, from: &str, to: &str) -> bool {
    match edge_type {
        "describes_source" => from == "source_record",
        "measured_as" => from == "raw_signal_series" && to == "control_series",
        "calibrated_by" => from == "raw_signal_series" && to == "calibration",
        "corrected_with" => from == "control_series" && to == "processed_measurement",
        "derived_from" => matches!(
            to,
            "calibration"
                | "processed_measurement"
                | "rate_estimate"
                | "fit_run"
                | "reproduced_parameter_estimate"
                | "prediction"
                | "evaluation"
        ),
        "fitted_using" => from == "model_specification" && to == "fit_run",
        "estimates" => from == "fit_run" && to == "reproduced_parameter_estimate",
        "reported_by" => from == "source_record" && to == "reported_parameter_claim",
        "predicts" => from == "model_specification" && to == "prediction",
        "evaluated_by" => from == "prediction" && to == "evaluation",
        _ => false,
    }
}

fn is_transformation_dependency_edge(edge_type: &str) -> bool {
    matches!(
        edge_type,
        "calibrated_by"
            | "corrected_with"
            | "derived_from"
            | "fitted_using"
            | "estimates"
            | "predicts"
            | "evaluated_by"
    )
}

fn ensure_acyclic(nodes: &[Node], edges: &[Edge]) -> Result<()> {
    let mut indegree = BTreeMap::<String, usize>::new();
    let mut adjacency = BTreeMap::<String, Vec<String>>::new();
    for node in nodes {
        indegree.insert(node.node_id.clone(), 0);
        adjacency.insert(node.node_id.clone(), Vec::new());
    }
    for edge in edges {
        adjacency.get_mut(&edge.from_node_id).unwrap().push(edge.to_node_id.clone());
        *indegree.get_mut(&edge.to_node_id).unwrap() += 1;
    }
    let mut ready = VecDeque::new();
    for (node_id, degree) in &indegree {
        if *degree == 0 {
            ready.push_back(node_id.clone());
        }
    }
    let mut visited = 0usize;
    while let Some(node_id) = ready.pop_front() {
        visited += 1;
        for next in &adjacency[&node_id] {
            let degree = indegree.get_mut(next).unwrap();
            *degree -= 1;
            if *degree == 0 {
                ready.push_back(next.clone());
            }
        }
    }
    ensure!(visited == nodes.len(), "evidence graph contains a directed cycle");
    Ok(())
}

fn safe_relative_path(value: &str) -> Result<PathBuf> {
    let path = Path::new(value);
    ensure!(!value.trim().is_empty(), "path is empty");
    ensure!(!path.is_absolute(), "absolute paths are prohibited");
    for component in path.components() {
        ensure!(
            matches!(component, Component::Normal(_)),
            "path must not contain '.', parent traversal, or root components"
        );
    }
    Ok(path.to_path_buf())
}

fn unique_index<'a, T>(
    iter: impl Iterator<Item = (&'a str, &'a T)>,
    field_name: &str,
) -> Result<BTreeMap<String, &'a T>> {
    let mut map = BTreeMap::new();
    for (id, item) in iter {
        ensure!(
            map.insert(id.to_owned(), item).is_none(),
            "duplicate {} '{}'",
            field_name,
            id
        );
    }
    Ok(map)
}

fn ensure_unique_values(values: &[String], context: &str) -> Result<()> {
    let mut seen = BTreeSet::new();
    for value in values {
        ensure!(!value.trim().is_empty(), "{} contains an empty value", context);
        ensure!(seen.insert(value), "{} contains duplicate value '{}'", context, value);
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempWorkspace(PathBuf);

    impl TempWorkspace {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!("symthaea-enzyme-evidence-{nonce}"));
            fs::create_dir_all(path.join("artifacts")).unwrap();
            Self(path)
        }

        fn root(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempWorkspace {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn payload(root: &Path, name: &str, bytes: &[u8]) -> Value {
        let path = root.join("artifacts").join(name);
        fs::write(&path, bytes).unwrap();
        json!({
            "artifact_id": format!("a_{name}"),
            "role": "raw_signal",
            "media_type": "application/octet-stream",
            "sha256": sha256(bytes),
            "byte_length": bytes.len(),
            "locator": format!("artifacts/{name}"),
            "rights_record_id": "rr_local",
            "acquisition_status": "available"
        })
    }

    fn fixture(root: &Path) -> Value {
        let input = b"synthetic-input\n";
        let output = b"synthetic-output\n";
        let code = b"fn synthetic_transform() {}\n";
        let configuration = b"{\"scale\":1}\n";
        let environment_lock = b"synthetic-lock-v1\n";
        let mut input_artifact = payload(root, "input.bin", input);
        let mut output_artifact = payload(root, "output.bin", output);
        let mut code_artifact = payload(root, "code.rs", code);
        let mut configuration_artifact = payload(root, "config.json", configuration);
        let mut environment_artifact = payload(root, "environment.lock", environment_lock);
        input_artifact["role"] = json!("raw_signal");
        output_artifact["role"] = json!("processed_measurement");
        code_artifact["role"] = json!("transformation_code");
        configuration_artifact["role"] = json!("transformation_configuration");
        environment_artifact["role"] = json!("environment_lock");
        let code_sha256 = code_artifact["sha256"].clone();
        let configuration_sha256 = configuration_artifact["sha256"].clone();
        let environment_lock_sha256 = environment_artifact["sha256"].clone();

        json!({
            "schema_version": 1,
            "graph_id": "synthetic-fixture",
            "status": "draft",
            "rights_records": [{
                "rights_record_id": "rr_local",
                "source_id": "synthetic-study",
                "license_id": "CC0-1.0",
                "attribution": "Synthetic test fixture; not experimental data",
                "status": "approved_for_this_use"
            }, {
                "rights_record_id": "rr_remote",
                "source_id": "remote-source",
                "license_id": "unknown",
                "attribution": "Metadata-only source reference",
                "status": "review_pending"
            }],
            "artifacts": [
                input_artifact,
                output_artifact,
                code_artifact,
                configuration_artifact,
                environment_artifact,
                {
                    "artifact_id": "a_remote",
                    "role": "source_record",
                    "media_type": "text/html",
                    "sha256": "a".repeat(64),
                    "byte_length": 123,
                    "locator": "https://example.org/source",
                    "rights_record_id": "rr_remote",
                    "acquisition_status": "not_acquired"
                }
            ],
            "nodes": [{
                "node_id": "n_source",
                "type": "source_record",
                "evidence_class": "source_claim",
                "artifact_ids": ["a_remote"]
            }, {
                "node_id": "n_raw",
                "type": "raw_signal_series",
                "evidence_class": "observed",
                "artifact_ids": ["a_input.bin"]
            }, {
                "node_id": "n_processed",
                "type": "processed_measurement",
                "evidence_class": "derived",
                "artifact_ids": ["a_output.bin"]
            }],
            "edges": [{
                "edge_id": "e_source_raw",
                "type": "describes_source",
                "from_node_id": "n_source",
                "to_node_id": "n_raw"
            }, {
                "edge_id": "e_source_processed",
                "type": "describes_source",
                "from_node_id": "n_source",
                "to_node_id": "n_processed"
            }, {
                "edge_id": "e_derive",
                "type": "derived_from",
                "from_node_id": "n_raw",
                "to_node_id": "n_processed"
            }],
            "transformations": [{
                "transformation_id": "t_process",
                "name": "synthetic identity process",
                "version": "1",
                "code_sha256": code_sha256,
                "configuration_sha256": configuration_sha256,
                "environment_lock_sha256": environment_lock_sha256,
                "code_artifact_id": "a_code.rs",
                "configuration_artifact_id": "a_config.json",
                "environment_lock_artifact_id": "a_environment.lock",
                "input_node_ids": ["n_raw"],
                "output_node_ids": ["n_processed"],
                "input_artifact_ids": ["a_input.bin"],
                "output_artifact_ids": ["a_output.bin"],
                "deterministic": true,
                "terminal_status": "success"
            }],
            "parameter_claims": []
        })
    }

    fn validate_fixture(root: &Path, value: &Value) -> Result<Report> {
        let graph: EvidenceGraph = serde_json::from_value(value.clone())?;
        validate_graph(root, &graph)
    }

    #[test]
    fn rejects_unreferenced_artifacts() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        let unreferenced = payload(workspace.root(), "unused.bin", b"unreferenced artifact\n");
        graph["artifacts"].as_array_mut().unwrap().push(unreferenced);

        let error = validate_fixture(workspace.root(), &graph).unwrap_err().to_string();
        assert!(error.contains("unreferenced by every evidence node and transformation"));
    }

    #[test]
    fn rejects_unknown_fields_at_every_typed_graph_level() {
        let workspace = TempWorkspace::new();

        let mut graph = fixture(workspace.root());
        graph["unrecognized_top_level"] = json!(true);
        assert!(
            serde_json::from_value::<EvidenceGraph>(graph).is_err(),
            "unknown top-level fields must not be silently ignored"
        );

        let mut graph = fixture(workspace.root());
        graph["nodes"][0]["evidnce_class"] = json!("observed");
        assert!(
            serde_json::from_value::<EvidenceGraph>(graph).is_err(),
            "unknown node fields must not be silently ignored"
        );

        let mut graph = fixture(workspace.root());
        graph["edges"][0]["from_nod_id"] = json!("n_source");
        assert!(
            serde_json::from_value::<EvidenceGraph>(graph).is_err(),
            "unknown edge fields must not be silently ignored"
        );

        let mut graph = fixture(workspace.root());
        graph["artifacts"][0]["byte_lenght"] = json!(17);
        assert!(
            serde_json::from_value::<EvidenceGraph>(graph).is_err(),
            "unknown artifact fields must not be silently ignored"
        );

        let mut graph = fixture(workspace.root());
        graph["rights_records"][0]["licence_id"] = json!("CC0-1.0");
        assert!(
            serde_json::from_value::<EvidenceGraph>(graph).is_err(),
            "unknown rights-record fields must not be silently ignored"
        );

        let mut graph = fixture(workspace.root());
        graph["transformations"][0]["terminal_reasn"] = json!("none");
        assert!(
            serde_json::from_value::<EvidenceGraph>(graph).is_err(),
            "unknown transformation fields must not be silently ignored"
        );

        let mut graph = fixture(workspace.root());
        graph["parameter_claims"] = json!([{
            "parameter_id": "p_kcat",
            "parameter_name": "kcat",
            "endpoint_definition": "turnover number",
            "value": 1.0,
            "unit": "s^-1",
            "enzyme_form_basis": "per catalytic site",
            "reaction_id": "reaction_1",
            "substrate_ids": ["substrate_1"],
            "assay_context": {"pH": 7.0},
            "kinetic_equation_id": "equation_1",
            "source_or_fit_node_id": "n_source",
            "uncertainty_status": "not_available",
            "uncertainty_stauts": "reported"
        }]);
        assert!(
            serde_json::from_value::<EvidenceGraph>(graph).is_err(),
            "unknown parameter-claim fields must not be silently ignored"
        );
    }

    #[test]
    fn rust_policy_vocabularies_match_the_machine_readable_contract() {
        let contract: Value = serde_json::from_str(include_str!(
            "../../docs/research/ENZYME_EVIDENCE_GRAPH_CONTRACT_V0.json"
        ))
        .unwrap();

        let contract_nodes: BTreeSet<(String, String)> = contract["node_types"]
            .as_array()
            .unwrap()
            .iter()
            .map(|node| (
                node["type"].as_str().unwrap().to_owned(),
                node["evidence_class"].as_str().unwrap().to_owned()
            ))
            .collect();
        let rust_nodes: BTreeSet<(String, String)> = NODE_TYPES
            .iter()
            .map(|(node_type, evidence_class)| ((*node_type).to_owned(), (*evidence_class).to_owned()))
            .collect();
        assert_eq!(rust_nodes, contract_nodes, "Rust node/evidence-class policy drifted from contract");

        let contract_edges: BTreeSet<String> = contract["edge_types"]
            .as_array()
            .unwrap()
            .iter()
            .map(|edge| edge["type"].as_str().unwrap().to_owned())
            .collect();
        let rust_edges: BTreeSet<String> = EDGE_TYPES.iter().map(|value| (*value).to_owned()).collect();
        assert_eq!(rust_edges, contract_edges, "Rust edge vocabulary drifted from contract");

        let contract_roles: BTreeSet<String> = contract["artifact_record"]["role_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect();
        let rust_roles: BTreeSet<String> = ARTIFACT_ROLES.iter().map(|value| (*value).to_owned()).collect();
        assert_eq!(rust_roles, contract_roles, "Rust artifact-role vocabulary drifted from contract");

        let contract_statuses: BTreeSet<String> = contract["graph_instance_v0"]["graph_status_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect();
        let rust_statuses: BTreeSet<String> = GRAPH_STATUSES.iter().map(|value| (*value).to_owned()).collect();
        assert_eq!(rust_statuses, contract_statuses, "Rust graph statuses drifted from contract");

        let contract_rights: BTreeSet<String> = contract["graph_instance_v0"]["rights_status_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect();
        let rust_rights: BTreeSet<String> = RIGHTS_STATUSES.iter().map(|value| (*value).to_owned()).collect();
        assert_eq!(rust_rights, contract_rights, "Rust rights statuses drifted from contract");

        let contract_acquisition: BTreeSet<String> = contract["graph_instance_v0"]["acquisition_status_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect();
        let rust_acquisition: BTreeSet<String> = ACQUISITION_STATUSES.iter().map(|value| (*value).to_owned()).collect();
        assert_eq!(rust_acquisition, contract_acquisition, "Rust artifact acquisition states drifted from contract");

        let contract_terminal: BTreeSet<String> = contract["transformation_record"]["terminal_status_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect();
        let rust_terminal: BTreeSet<String> = TERMINAL_STATUSES.iter().map(|value| (*value).to_owned()).collect();
        assert_eq!(rust_terminal, contract_terminal, "Rust transformation terminal statuses drifted from contract");

        let contract_uncertainty: BTreeSet<String> = contract["parameter_claim_record"]["uncertainty_status_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect();
        let rust_uncertainty: BTreeSet<String> = UNCERTAINTY_STATUSES.iter().map(|value| (*value).to_owned()).collect();
        assert_eq!(rust_uncertainty, contract_uncertainty, "Rust parameter uncertainty statuses drifted from contract");
    }

    #[test]
    fn accepts_a_hash_verified_local_graph_without_fetching_remote_metadata() {
        let workspace = TempWorkspace::new();
        let graph = fixture(workspace.root());
        let report = validate_fixture(workspace.root(), &graph).unwrap();
        assert_eq!(report.artifacts_verified, 5);
        assert_eq!(report.artifacts_total, 6);
        assert_eq!(report.transformations_successful, 1);
        assert_eq!(report.rights_review_unresolved, 1);
    }

    #[test]
    fn rejects_dangling_edge_endpoints() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["edges"][2]["to_node_id"] = json!("missing-node");
        let error = validate_fixture(workspace.root(), &graph).unwrap_err().to_string();
        assert!(error.contains("dangling to_node_id"));
    }

    #[test]
    fn rejects_directed_cycles_even_when_every_id_resolves() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["nodes"].as_array_mut().unwrap().push(json!({
            "node_id": "n_rate",
            "type": "rate_estimate",
            "evidence_class": "derived_or_fitted",
            "artifact_ids": []
        }));
        graph["edges"].as_array_mut().unwrap().push(json!({
            "edge_id": "e_cycle_a",
            "type": "derived_from",
            "from_node_id": "n_processed",
            "to_node_id": "n_rate"
        }));
        graph["edges"].as_array_mut().unwrap().push(json!({
            "edge_id": "e_cycle_b",
            "type": "derived_from",
            "from_node_id": "n_rate",
            "to_node_id": "n_processed"
        }));
        let error = validate_fixture(workspace.root(), &graph).unwrap_err().to_string();
        assert!(error.contains("directed cycle"));
    }

    #[test]
    fn rejects_tampered_local_artifact_bytes() {
        let workspace = TempWorkspace::new();
        let graph = fixture(workspace.root());
        fs::write(workspace.root().join("artifacts/input.bin"), b"tampered").unwrap();
        let error = validate_fixture(workspace.root(), &graph).unwrap_err().to_string();
        assert!(error.contains("byte_length mismatch") || error.contains("SHA-256 mismatch"));
    }

    #[test]
    fn refuses_unacquired_remote_artifact_as_successful_transform_input() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["nodes"][1]["artifact_ids"].as_array_mut().unwrap().push(json!("a_remote"));
        graph["transformations"][0]["input_artifact_ids"] = json!(["a_input.bin", "a_remote"]);
        let error = validate_fixture(workspace.root(), &graph).unwrap_err().to_string();
        assert!(error.contains("unavailable or unverified input artifacts") || error.contains("not linked from its input nodes"));
    }

    #[test]
    fn preserves_null_result_with_inputs_and_no_output_nodes() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["transformations"].as_array_mut().unwrap().push(json!({
            "transformation_id": "t_null",
            "name": "synthetic fit that returns no estimate",
            "version": "1",
            "code_sha256": "4".repeat(64),
            "configuration_sha256": "5".repeat(64),
            "environment_lock_sha256": "6".repeat(64),
            "input_node_ids": ["n_raw"],
            "output_node_ids": [],
            "input_artifact_ids": ["a_input.bin"],
            "output_artifact_ids": [],
            "deterministic": true,
            "terminal_status": "null_result",
            "terminal_reason": "Fit returned no finite parameter estimate"
        }));
        let report = validate_fixture(workspace.root(), &graph).unwrap();
        assert_eq!(report.transformations_terminal_non_success, 1);
        assert_eq!(report.transformations_successful, 1);
    }

    #[test]
    fn refuses_successful_transform_with_unreviewed_rights_even_when_bytes_match() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["rights_records"][0]["status"] = json!("review_pending");
        let error = validate_fixture(workspace.root(), &graph).unwrap_err().to_string();
        assert!(error.contains("rights status 'review_pending'"));
    }

    #[test]
    fn rejects_node_evidence_class_type_mismatch() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["nodes"][1]["evidence_class"] = json!("prediction");
        let error = validate_fixture(workspace.root(), &graph).unwrap_err().to_string();
        assert!(error.contains("requires evidence_class"));
    }

    #[test]
    fn rejects_successful_transform_without_verified_output_artifact() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["transformations"][0]["output_artifact_ids"] = json!(["a_remote"]);
        let error = validate_fixture(workspace.root(), &graph).unwrap_err().to_string();
        assert!(error.contains("output artifacts are not linked") || error.contains("unavailable or unverified output artifacts"));
    }

    #[test]
    fn requires_reason_for_failed_or_blocked_transformations() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["transformations"][0]["terminal_status"] = json!("blocked");
        graph["transformations"][0].as_object_mut().unwrap().remove("terminal_reason");
        let error = validate_fixture(workspace.root(), &graph).unwrap_err().to_string();
        assert!(error.contains("terminal_reason"));
    }

    #[test]
    fn fit_parameter_claim_requires_estimates_edge() {
        let fit = Node {
            node_id: "n_fit".to_owned(),
            node_type: "fit_run".to_owned(),
            evidence_class: "computation".to_owned(),
            artifact_ids: vec![],
        };
        let estimate = Node {
            node_id: "n_estimate".to_owned(),
            node_type: "reproduced_parameter_estimate".to_owned(),
            evidence_class: "reproduced_computation".to_owned(),
            artifact_ids: vec![],
        };
        let nodes = vec![fit, estimate];
        let nodes_by_id =
            unique_index(nodes.iter().map(|node| (node.node_id.as_str(), node)), "node_id")
                .unwrap();
        let claim = ParameterClaim {
            parameter_id: "p_fit".to_owned(),
            parameter_name: "kcat".to_owned(),
            endpoint_definition: "catalytic turnover number".to_owned(),
            value: json!(12.4),
            unit: "s^-1".to_owned(),
            normalized_value: None,
            normalized_unit: None,
            enzyme_form_basis: "per catalytic site".to_owned(),
            reaction_id: "reaction-1".to_owned(),
            substrate_ids: vec!["substrate-1".to_owned()],
            assay_context: json!({"pH": 7.0, "temperature_c": 25.0}),
            kinetic_equation_id: "michaelis-menten".to_owned(),
            source_or_fit_node_id: "n_fit".to_owned(),
            uncertainty_status: "not_available".to_owned(),
        };

        let error = validate_parameter_claims(&[claim.clone()], &nodes_by_id, &[])
            .unwrap_err()
            .to_string();
        assert!(error.contains("without an estimates edge"));

        let edge = Edge {
            edge_id: "e_estimates".to_owned(),
            edge_type: "estimates".to_owned(),
            from_node_id: "n_fit".to_owned(),
            to_node_id: "n_estimate".to_owned(),
        };
        assert!(validate_parameter_claims(&[claim], &nodes_by_id, &[edge]).is_ok());
    }

    #[test]
    fn rejects_parameter_claim_linked_to_source_without_reported_by_edge() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["parameter_claims"] = json!([{
            "parameter_id": "p_source",
            "parameter_name": "kcat",
            "endpoint_definition": "catalytic turnover number",
            "value": 12.4,
            "unit": "s^-1",
            "enzyme_form_basis": "not reported",
            "reaction_id": "reaction-1",
            "substrate_ids": ["substrate-1"],
            "assay_context": {"pH": null, "temperature_c": 25.0},
            "kinetic_equation_id": "michaelis-menten",
            "source_or_fit_node_id": "n_source",
            "uncertainty_status": "not_available"
        }]);

        let error = validate_fixture(workspace.root(), &graph).unwrap_err().to_string();
        assert!(error.contains("without a reported_by edge"));
    }

    #[test]
    fn accepts_parameter_claim_linked_to_source_by_reported_by_edge() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["nodes"].as_array_mut().unwrap().push(json!({
            "node_id": "n_reported",
            "type": "reported_parameter_claim",
            "evidence_class": "source_claim",
            "artifact_ids": []
        }));
        graph["edges"].as_array_mut().unwrap().push(json!({
            "edge_id": "e_reported_parameter",
            "type": "reported_by",
            "from_node_id": "n_source",
            "to_node_id": "n_reported"
        }));
        graph["parameter_claims"] = json!([{
            "parameter_id": "p_source",
            "parameter_name": "kcat",
            "endpoint_definition": "catalytic turnover number",
            "value": 12.4,
            "unit": "s^-1",
            "enzyme_form_basis": "not reported",
            "reaction_id": "reaction-1",
            "substrate_ids": ["substrate-1"],
            "assay_context": {"pH": null, "temperature_c": 25.0},
            "kinetic_equation_id": "michaelis-menten",
            "source_or_fit_node_id": "n_source",
            "uncertainty_status": "not_available"
        }]);

        let report = validate_fixture(workspace.root(), &graph).unwrap();
        assert_eq!(report.parameter_claims, 1);
    }

    #[test]
    fn rejects_unknown_parameter_claim_uncertainty_status() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["nodes"].as_array_mut().unwrap().push(json!({
            "node_id": "n_reported",
            "type": "reported_parameter_claim",
            "evidence_class": "source_claim",
            "artifact_ids": []
        }));
        graph["parameter_claims"] = json!([{
            "parameter_id": "p1",
            "parameter_name": "kcat",
            "endpoint_definition": "catalytic turnover number",
            "value": "12.4",
            "unit": "s^-1",
            "enzyme_form_basis": "not reported",
            "reaction_id": "reaction-1",
            "substrate_ids": ["substrate-1"],
            "assay_context": {"pH": null, "temperature_c": 25.0},
            "kinetic_equation_id": "michaelis-menten",
            "source_or_fit_node_id": "n_reported",
            "uncertainty_status": "zero"
        }]);
        let error = validate_fixture(workspace.root(), &graph).unwrap_err().to_string();
        assert!(error.contains("uncertainty_status"));
    }

    #[test]
    fn accepts_verbatim_and_normalized_parameter_value_with_unit_pair() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["nodes"].as_array_mut().unwrap().push(json!({
            "node_id": "n_reported",
            "type": "reported_parameter_claim",
            "evidence_class": "source_claim",
            "artifact_ids": []
        }));
        graph["parameter_claims"] = json!([{
            "parameter_id": "p1",
            "parameter_name": "kcat",
            "endpoint_definition": "catalytic turnover number",
            "value": "12.4 min^-1",
            "unit": "min^-1",
            "normalized_value": 0.2066667,
            "normalized_unit": "s^-1",
            "enzyme_form_basis": "not reported",
            "reaction_id": "reaction-1",
            "substrate_ids": ["substrate-1"],
            "assay_context": {"pH": null, "temperature_c": 25.0},
            "kinetic_equation_id": "michaelis-menten",
            "source_or_fit_node_id": "n_reported",
            "uncertainty_status": "reported"
        }]);
        let report = validate_fixture(workspace.root(), &graph).unwrap();
        assert_eq!(report.parameter_claims, 1);
    }

    #[test]
    fn rejects_object_or_array_as_parameter_value() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["nodes"].as_array_mut().unwrap().push(json!({
            "node_id": "n_reported",
            "type": "reported_parameter_claim",
            "evidence_class": "source_claim",
            "artifact_ids": []
        }));
        graph["parameter_claims"] = json!([{
            "parameter_id": "p1",
            "parameter_name": "kcat",
            "endpoint_definition": "catalytic turnover number",
            "value": {"lower": 12, "upper": 14},
            "unit": "s^-1",
            "enzyme_form_basis": "not reported",
            "reaction_id": "reaction-1",
            "substrate_ids": ["substrate-1"],
            "assay_context": {"pH": null, "temperature_c": 25.0},
            "kinetic_equation_id": "michaelis-menten",
            "source_or_fit_node_id": "n_reported",
            "uncertainty_status": "reported"
        }]);
        let error = validate_fixture(workspace.root(), &graph).unwrap_err().to_string();
        assert!(error.contains("value must be a number or non-empty verbatim source string"));
    }

    #[test]
    fn rejects_normalized_value_without_paired_unit() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["nodes"].as_array_mut().unwrap().push(json!({
            "node_id": "n_reported",
            "type": "reported_parameter_claim",
            "evidence_class": "source_claim",
            "artifact_ids": []
        }));
        graph["parameter_claims"] = json!([{
            "parameter_id": "p1",
            "parameter_name": "kcat",
            "endpoint_definition": "catalytic turnover number",
            "value": "reported as 12.4",
            "unit": "min^-1",
            "normalized_value": 0.2066667,
            "enzyme_form_basis": "not reported",
            "reaction_id": "reaction-1",
            "substrate_ids": ["substrate-1"],
            "assay_context": {"pH": null, "temperature_c": 25.0},
            "kinetic_equation_id": "michaelis-menten",
            "source_or_fit_node_id": "n_reported",
            "uncertainty_status": "reported"
        }]);
        let error = validate_fixture(workspace.root(), &graph).unwrap_err().to_string();
        assert!(error.contains("normalized_value and normalized_unit must be supplied together"));
    }

    #[test]
    fn rejects_empty_substrate_list_in_parameter_claim() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["nodes"].as_array_mut().unwrap().push(json!({
            "node_id": "n_reported",
            "type": "reported_parameter_claim",
            "evidence_class": "source_claim",
            "artifact_ids": []
        }));
        graph["parameter_claims"] = json!([{
            "parameter_id": "p1",
            "parameter_name": "kcat",
            "endpoint_definition": "catalytic turnover number",
            "value": 12.4,
            "unit": "s^-1",
            "enzyme_form_basis": "not reported",
            "reaction_id": "reaction-1",
            "substrate_ids": [],
            "assay_context": {"pH": null, "temperature_c": 25.0},
            "kinetic_equation_id": "michaelis-menten",
            "source_or_fit_node_id": "n_reported",
            "uncertainty_status": "reported"
        }]);
        let error = validate_fixture(workspace.root(), &graph).unwrap_err().to_string();
        assert!(error.contains("must identify at least one substrate"));
    }

    #[test]
    fn rejects_duplicate_substrate_ids_in_parameter_claim() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["nodes"].as_array_mut().unwrap().push(json!({
            "node_id": "n_reported",
            "type": "reported_parameter_claim",
            "evidence_class": "source_claim",
            "artifact_ids": []
        }));
        graph["parameter_claims"] = json!([{
            "parameter_id": "p1",
            "parameter_name": "kcat",
            "endpoint_definition": "catalytic turnover number",
            "value": 12.4,
            "unit": "s^-1",
            "normalized_value": 12.4,
            "normalized_unit": "s^-1",
            "enzyme_form_basis": "not reported",
            "reaction_id": "reaction-1",
            "substrate_ids": ["substrate-1", "substrate-1"],
            "assay_context": {"pH": null, "temperature_c": 25.0},
            "kinetic_equation_id": "michaelis-menten",
            "source_or_fit_node_id": "n_reported",
            "uncertainty_status": "reported"
        }]);
        let error = validate_fixture(workspace.root(), &graph).unwrap_err().to_string();
        assert!(error.contains("duplicate value 'substrate-1'"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape_even_when_external_bytes_match_the_declared_hash() {
        use std::os::unix::fs::symlink;
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let outside = workspace.root().parent().unwrap().join(format!("symthaea-enzyme-outside-{nonce}.bin"));
        let bytes = b"external bytes with matching metadata";
        fs::write(&outside, bytes).unwrap();
        symlink(&outside, workspace.root().join("artifacts/leak.bin")).unwrap();
        graph["artifacts"][0]["locator"] = json!("artifacts/leak.bin");
        graph["artifacts"][0]["sha256"] = json!(sha256(bytes));
        graph["artifacts"][0]["byte_length"] = json!(bytes.len());
        let error = validate_fixture(workspace.root(), &graph).unwrap_err().to_string();
        let _ = fs::remove_file(&outside);
        assert!(error.contains("escapes workspace root through a symlink"));
    }

    #[test]
    fn rejects_execution_fingerprint_that_disagrees_with_its_artifact_sha256() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["transformations"][0]["code_sha256"] = json!("f".repeat(64));
        let error = validate_fixture(workspace.root(), &graph).unwrap_err().to_string();
        assert!(error.contains("code SHA-256 does not match referenced artifact"));
    }

    #[test]
    fn successful_transform_requires_code_configuration_and_environment_artifacts() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["transformations"][0].as_object_mut().unwrap().remove("environment_lock_artifact_id");
        let error = validate_fixture(workspace.root(), &graph).unwrap_err().to_string();
        assert!(error.contains("missing environment_lock_artifact_id"));
    }

    #[test]
    fn rejects_malformed_transformation_fingerprint() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["transformations"][0]["code_sha256"] = json!("not-a-sha256");
        let error = validate_fixture(workspace.root(), &graph).unwrap_err().to_string();
        assert!(error.contains("code_sha256 is invalid"));
    }

    #[test]
    fn rejects_workspace_escape_via_parent_traversal() {
        assert!(safe_relative_path("../outside.json").is_err());
        assert!(safe_relative_path("/absolute/file.json").is_err());
    }

    #[test]
    fn frozen_graph_requires_external_exact_byte_pin() {
        let workspace = TempWorkspace::new();
        let mut graph = fixture(workspace.root());
        graph["status"] = json!("frozen");
        let path = workspace.root().join("graph.json");
        fs::write(&path, serde_json::to_vec_pretty(&graph).unwrap()).unwrap();
        let error = check(workspace.root(), Path::new("graph.json"), None).unwrap_err().to_string();
        assert!(error.contains("requires an external --sha256 pin"));
        let bytes = fs::read(&path).unwrap();
        let expected = sha256(&bytes);
        check(workspace.root(), Path::new("graph.json"), Some(&expected)).unwrap();
    }
}
