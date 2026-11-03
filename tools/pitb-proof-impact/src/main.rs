use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    env, fs,
    process::ExitCode,
};

const SNAPSHOT_VERSION: &str = "pitb.proof-impact-snapshot.v0";
const REPORT_VERSION: &str = "pitb.proof-impact-report.v0";

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    schema_version: String,
    snapshot_id: String,
    repository_commit: String,
    environment_sha256: String,
    dependency_inventory_complete: bool,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Node {
    node_id: String,
    kind: String,
    declaration_kind: String,
    reducibility: String,
    statement_sha256: Option<String>,
    body_sha256: Option<String>,
    proof_sha256: Option<String>,
    axiom_manifest_sha256: Option<String>,
    source_sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
struct Edge {
    edge_id: String,
    source_node_id: String,
    target_node_id: String,
    relation: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Action {
    Retain,
    RetrievalRefresh,
    SourceRevalidate,
    Revalidate,
    Reprove,
    Rebuild,
    UnknownConservative,
}

impl Action {
    fn as_str(self) -> &'static str {
        match self {
            Self::Retain => "retain",
            Self::RetrievalRefresh => "retrieval_refresh",
            Self::SourceRevalidate => "source_revalidate",
            Self::Revalidate => "revalidate",
            Self::Reprove => "reprove",
            Self::Rebuild => "rebuild",
            Self::UnknownConservative => "unknown_conservative",
        }
    }

    fn rank(self) -> u8 {
        match self {
            Self::Retain => 0,
            Self::RetrievalRefresh => 1,
            Self::SourceRevalidate => 2,
            Self::Revalidate => 3,
            Self::Reprove => 4,
            Self::Rebuild => 5,
            Self::UnknownConservative => 6,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct Impact {
    node_id: String,
    action: String,
    reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct Change {
    change_id: String,
    change_kind: String,
    old_sha256: Option<String>,
    new_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct SnapshotIdentity {
    snapshot_id: String,
    repository_commit: String,
    environment_sha256: String,
    raw_snapshot_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    schema_version: String,
    old_snapshot: SnapshotIdentity,
    new_snapshot: SnapshotIdentity,
    dependency_inventory_complete: bool,
    conservative_mode: bool,
    changes: Vec<Change>,
    impacts: Vec<Impact>,
    action_counts: BTreeMap<String, usize>,
    summary_note: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Propagation {
    Signature,
    TransparentBody,
    ProofEvidence,
    Environment,
    Retrieval,
}

#[derive(Debug, Clone, Copy)]
struct NodeChange {
    propagation: Propagation,
    action: Action,
    code: &'static str,
}

fn fail(message: impl AsRef<str>) -> ExitCode {
    eprintln!("INVALID: {}", message.as_ref());
    ExitCode::FAILURE
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

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn optional_digest(value: &Option<String>, label: &str) -> Result<(), String> {
    if let Some(value) = value {
        if !valid_sha256(value) {
            return Err(format!("{label} must be 64 lowercase hexadecimal characters"));
        }
    }
    Ok(())
}

fn is_formal(node: &Node) -> bool {
    node.kind == "formal_declaration"
}

fn validate_snapshot(snapshot: &Snapshot) -> Result<(), String> {
    if snapshot.schema_version != SNAPSHOT_VERSION {
        return Err("unsupported proof-impact snapshot schema".into());
    }
    if snapshot.snapshot_id.trim().is_empty()
        || !valid_sha1(&snapshot.repository_commit)
        || !valid_sha256(&snapshot.environment_sha256)
    {
        return Err("snapshot identity is incomplete or malformed".into());
    }

    let mut nodes = BTreeMap::new();
    for node in &snapshot.nodes {
        if node.node_id.trim().is_empty() {
            return Err("node_id cannot be empty".into());
        }
        if !matches!(
            node.kind.as_str(),
            "formal_declaration" | "module" | "environment_artifact" | "document"
                | "blueprint_node" | "informal_statement"
        ) {
            return Err(format!("node {} has unknown kind {}", node.node_id, node.kind));
        }
        if !matches!(
            node.declaration_kind.as_str(),
            "theorem" | "opaque" | "definition" | "instance" | "class" | "structure"
                | "inductive" | "module" | "environment" | "document" | "blueprint"
                | "informal" | "other"
        ) {
            return Err(format!("node {} has unknown declaration_kind", node.node_id));
        }
        if !matches!(node.reducibility.as_str(), "transparent" | "opaque" | "not_applicable") {
            return Err(format!("node {} has unknown reducibility", node.node_id));
        }
        if is_formal(node) && node.statement_sha256.is_none() {
            return Err(format!("formal declaration {} requires statement_sha256", node.node_id));
        }
        if !is_formal(node) && node.statement_sha256.is_some() {
            return Err(format!("non-formal node {} cannot carry a Lean statement digest", node.node_id));
        }
        if node.declaration_kind == "definition" && node.reducibility == "not_applicable" {
            return Err(format!("definition {} must declare transparent or opaque reducibility", node.node_id));
        }
        for (field, value) in [
            ("statement_sha256", &node.statement_sha256),
            ("body_sha256", &node.body_sha256),
            ("proof_sha256", &node.proof_sha256),
            ("axiom_manifest_sha256", &node.axiom_manifest_sha256),
            ("source_sha256", &node.source_sha256),
        ] {
            optional_digest(value, &format!("node {} {field}", node.node_id))?;
        }
        if nodes.insert(node.node_id.clone(), node).is_some() {
            return Err(format!("duplicate node ID {}", node.node_id));
        }
    }

    let mut edge_ids = BTreeSet::new();
    let mut edge_tuples = BTreeSet::new();
    for edge in &snapshot.edges {
        if edge.edge_id.trim().is_empty()
            || edge.source_node_id.trim().is_empty()
            || edge.target_node_id.trim().is_empty()
        {
            return Err("edge identity fields cannot be empty".into());
        }
        if edge.source_node_id == edge.target_node_id {
            return Err(format!("edge {} cannot point to itself", edge.edge_id));
        }
        if !edge_ids.insert(edge.edge_id.clone()) {
            return Err(format!("duplicate edge ID {}", edge.edge_id));
        }
        if !edge_tuples.insert((
            edge.source_node_id.clone(),
            edge.target_node_id.clone(),
            edge.relation.clone(),
        )) {
            return Err(format!("duplicate relation edge {}", edge.edge_id));
        }
        if !nodes.contains_key(&edge.source_node_id) || !nodes.contains_key(&edge.target_node_id) {
            return Err(format!("edge {} references a missing node", edge.edge_id));
        }
        if !matches!(
            edge.relation.as_str(),
            "type_reference" | "signature_reference" | "proof_term_reference"
                | "definition_body_reference" | "structure_field_reference"
                | "inheritance_reference" | "instance_synthesis"
                | "module_environment_dependency" | "documentation_reference"
                | "blueprint_dependency" | "informal_candidate_dependency"
                | "cross_formality_match"
        ) {
            return Err(format!("edge {} has unsupported relation {}", edge.edge_id, edge.relation));
        }
        if !relation_endpoint_kinds_valid(edge, nodes[&edge.source_node_id], nodes[&edge.target_node_id]) {
            return Err(format!("edge {} has incompatible endpoint node kinds", edge.edge_id));
        }
    }
    Ok(())
}

fn relation_endpoint_kinds_valid(edge: &Edge, source: &Node, target: &Node) -> bool {
    match edge.relation.as_str() {
        "type_reference" | "signature_reference" | "proof_term_reference"
        | "definition_body_reference" | "structure_field_reference"
        | "inheritance_reference" | "instance_synthesis" => {
            is_formal(source) && is_formal(target)
        }
        "module_environment_dependency" => {
            (is_formal(source) || source.kind == "module")
                && matches!(target.kind.as_str(), "module" | "environment_artifact")
        }
        "documentation_reference" => is_formal(source) && target.kind == "document",
        "blueprint_dependency" => {
            matches!(source.kind.as_str(), "formal_declaration" | "blueprint_node" | "informal_statement")
                && matches!(target.kind.as_str(), "formal_declaration" | "blueprint_node" | "informal_statement")
        }
        "informal_candidate_dependency" => {
            source.kind == "informal_statement" && target.kind == "informal_statement"
        }
        "cross_formality_match" => {
            (is_formal(source) && matches!(target.kind.as_str(), "informal_statement" | "blueprint_node"))
                || (is_formal(target) && matches!(source.kind.as_str(), "informal_statement" | "blueprint_node"))
        }
        _ => false,
    }
}

// Edges are oriented dependent -> dependency. These policies classify what a
// consumer must do when its target dependency changes.
fn direct_action_for_relation(relation: &str, changed_target: &Node) -> Option<Action> {
    match relation {
        "type_reference" | "signature_reference" | "structure_field_reference"
        | "inheritance_reference" | "instance_synthesis" => Some(Action::Reprove),
        "definition_body_reference" => {
            if changed_target.reducibility == "transparent" {
                Some(Action::Reprove)
            } else {
                Some(Action::Revalidate)
            }
        }
        "proof_term_reference" => Some(Action::Revalidate),
        "module_environment_dependency" => Some(Action::Rebuild),
        "documentation_reference" | "blueprint_dependency"
        | "informal_candidate_dependency" | "cross_formality_match" => {
            Some(Action::RetrievalRefresh)
        }
        _ => None,
    }
}

fn action_for_propagation(relation: &str, target: &Node, propagation: Propagation) -> Option<Action> {
    match propagation {
        Propagation::Signature => direct_action_for_relation(relation, target),
        Propagation::TransparentBody => match relation {
            "definition_body_reference" => Some(Action::Reprove),
            "proof_term_reference" => Some(Action::Revalidate),
            _ => None,
        },
        Propagation::ProofEvidence => match relation {
            "proof_term_reference" => Some(Action::Revalidate),
            _ => None,
        },
        Propagation::Environment => match relation {
            "module_environment_dependency" => Some(Action::Rebuild),
            "proof_term_reference" | "definition_body_reference"
            | "type_reference" | "signature_reference" | "structure_field_reference"
            | "inheritance_reference" | "instance_synthesis" => Some(Action::Rebuild),
            _ => None,
        },
        Propagation::Retrieval => match relation {
            "documentation_reference" | "blueprint_dependency"
            | "informal_candidate_dependency" | "cross_formality_match" => {
                Some(Action::RetrievalRefresh)
            }
            _ => None,
        },
    }
}

fn enqueue(
    queue: &mut VecDeque<(String, Propagation)>,
    queued: &mut BTreeSet<(String, Propagation)>,
    node_id: &str,
    propagation: Propagation,
) {
    let key = (node_id.to_owned(), propagation);
    if queued.insert(key.clone()) {
        queue.push_back(key);
    }
}

fn add_reason(
    actions: &mut BTreeMap<String, Action>,
    reasons: &mut BTreeMap<String, BTreeSet<String>>,
    node_id: &str,
    action: Action,
    reason: String,
) -> bool {
    let old = actions.get(node_id).copied().unwrap_or(Action::Retain);
    let changed = action.rank() > old.rank();
    if changed {
        actions.insert(node_id.to_owned(), action);
    } else {
        actions.entry(node_id.to_owned()).or_insert(action);
    }
    reasons.entry(node_id.to_owned()).or_default().insert(reason);
    changed
}

fn node_change(old: &Node, new: &Node) -> Vec<NodeChange> {
    let mut changes = Vec::new();
    if old.kind != new.kind
        || old.declaration_kind != new.declaration_kind
        || old.reducibility != new.reducibility
        || old.statement_sha256 != new.statement_sha256
    {
        changes.push(NodeChange {
            propagation: Propagation::Signature,
            action: Action::Reprove,
            code: "declaration_signature_or_kind_changed",
        });
    }
    if old.body_sha256 != new.body_sha256 {
        if new.reducibility == "transparent" {
            changes.push(NodeChange {
                propagation: Propagation::TransparentBody,
                action: Action::Reprove,
                code: "transparent_definition_body_changed",
            });
        } else {
            changes.push(NodeChange {
                propagation: Propagation::ProofEvidence,
                action: Action::Revalidate,
                code: "opaque_body_or_proof_term_changed",
            });
        }
    }
    if old.proof_sha256 != new.proof_sha256 || old.axiom_manifest_sha256 != new.axiom_manifest_sha256 {
        changes.push(NodeChange {
            propagation: Propagation::ProofEvidence,
            action: Action::Revalidate,
            code: "proof_or_axiom_evidence_changed",
        });
    }
    if old.source_sha256 != new.source_sha256
        && old.statement_sha256 == new.statement_sha256
        && old.body_sha256 == new.body_sha256
        && old.proof_sha256 == new.proof_sha256
        && old.axiom_manifest_sha256 == new.axiom_manifest_sha256
    {
        let action = if new.kind == "document" || new.kind == "blueprint_node" || new.kind == "informal_statement" {
            Action::RetrievalRefresh
        } else if new.kind == "module" || new.kind == "environment_artifact" {
            Action::Rebuild
        } else {
            Action::SourceRevalidate
        };
        changes.push(NodeChange {
            propagation: if action == Action::Rebuild {
                Propagation::Environment
            } else if action == Action::RetrievalRefresh {
                Propagation::Retrieval
            } else {
                Propagation::ProofEvidence
            },
            action,
            code: "source_bytes_changed_without_semantic_digest_change",
        });
    }
    changes
}

fn relation_event_action(edge: &Edge, target: Option<&Node>) -> Action {
    match edge.relation.as_str() {
        "documentation_reference" | "blueprint_dependency"
        | "informal_candidate_dependency" | "cross_formality_match" => Action::RetrievalRefresh,
        "module_environment_dependency" => Action::Rebuild,
        "proof_term_reference" => Action::Revalidate,
        "definition_body_reference" => {
            if target.is_some_and(|node| node.reducibility == "transparent") {
                Action::Reprove
            } else {
                Action::Revalidate
            }
        }
        _ => Action::Reprove,
    }
}

fn insert_change(changes: &mut Vec<Change>, change_id: String, kind: &str, old: Option<String>, new: Option<String>) {
    changes.push(Change {
        change_id,
        change_kind: kind.to_owned(),
        old_sha256: old,
        new_sha256: new,
    });
}

fn analyze(old_bytes: &[u8], new_bytes: &[u8]) -> Result<Report, String> {
    let old: Snapshot = serde_json::from_slice(old_bytes)
        .map_err(|error| format!("old snapshot JSON invalid: {error}"))?;
    let new: Snapshot = serde_json::from_slice(new_bytes)
        .map_err(|error| format!("new snapshot JSON invalid: {error}"))?;
    validate_snapshot(&old).map_err(|error| format!("old snapshot invalid: {error}"))?;
    validate_snapshot(&new).map_err(|error| format!("new snapshot invalid: {error}"))?;

    let old_nodes: BTreeMap<String, &Node> = old.nodes.iter().map(|node| (node.node_id.clone(), node)).collect();
    let new_nodes: BTreeMap<String, &Node> = new.nodes.iter().map(|node| (node.node_id.clone(), node)).collect();

    let mut actions: BTreeMap<String, Action> = new.nodes.iter()
        .filter(|node| is_formal(node))
        .map(|node| (node.node_id.clone(), Action::Retain))
        .collect();
    let mut reasons: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut changes = Vec::new();
    let mut queue = VecDeque::<(String, Propagation)>::new();
    let mut queued = BTreeSet::<(String, Propagation)>::new();

    for (id, old_node) in &old_nodes {
        match new_nodes.get(id) {
            None => {
                insert_change(
                    &mut changes,
                    id.clone(),
                    "node_removed",
                    old_node.statement_sha256.clone().or(old_node.body_sha256.clone()).or(old_node.source_sha256.clone()),
                    None,
                );
                if is_formal(old_node) {
                    // Removed declaration's dependents are discovered through old edges below.
                    enqueue(&mut queue, &mut queued, id, Propagation::Signature);
                }
            }
            Some(new_node) => {
                for change in node_change(old_node, new_node) {
                    insert_change(
                        &mut changes,
                        id.clone(),
                        change.code,
                        old_node.statement_sha256.clone().or(old_node.body_sha256.clone()).or(old_node.proof_sha256.clone()),
                        new_node.statement_sha256.clone().or(new_node.body_sha256.clone()).or(new_node.proof_sha256.clone()),
                    );
                    if is_formal(new_node) {
                        add_reason(
                            &mut actions,
                            &mut reasons,
                            id,
                            change.action,
                            change.code.to_owned(),
                        );
                    } else if change.action != Action::Retain {
                        add_reason(
                            &mut actions,
                            &mut reasons,
                            id,
                            change.action,
                            change.code.to_owned(),
                        );
                    }
                    enqueue(&mut queue, &mut queued, id, change.propagation);
                }
            }
        }
    }
    for (id, new_node) in &new_nodes {
        if !old_nodes.contains_key(id) {
            insert_change(&mut changes, id.clone(), "node_added", None, new_node.statement_sha256.clone().or(new_node.body_sha256.clone()).or(new_node.source_sha256.clone()));
            if is_formal(new_node) {
                add_reason(&mut actions, &mut reasons, id, Action::Reprove, "new_declaration_requires_initial_proof".into());
            }
        }
    }

    if old.environment_sha256 != new.environment_sha256 {
        insert_change(
            &mut changes,
            "snapshot:environment".into(),
            "global_environment_changed",
            Some(old.environment_sha256.clone()),
            Some(new.environment_sha256.clone()),
        );
        for node in &new.nodes {
            if is_formal(node) {
                add_reason(&mut actions, &mut reasons, &node.node_id, Action::Rebuild, "global_lean_environment_changed".into());
                enqueue(&mut queue, &mut queued, &node.node_id, Propagation::Environment);
            }
        }
    }

    let old_edges: BTreeMap<String, &Edge> = old.edges.iter().map(|edge| (edge.edge_id.clone(), edge)).collect();
    let new_edges: BTreeMap<String, &Edge> = new.edges.iter().map(|edge| (edge.edge_id.clone(), edge)).collect();
    for (id, old_edge) in &old_edges {
        let maybe_new = new_edges.get(id);
        if maybe_new.is_none() || maybe_new.is_some_and(|new_edge| *new_edge != *old_edge) {
            insert_change(&mut changes, id.clone(), "edge_removed_or_changed", None, None);
            let target = old_nodes.get(&old_edge.target_node_id).copied();
            let action = relation_event_action(old_edge, target);
            add_reason(
                &mut actions,
                &mut reasons,
                &old_edge.source_node_id,
                action,
                format!("dependency_edge_removed_or_changed:{}", old_edge.relation),
            );
            if action == Action::Reprove || action == Action::Revalidate {
                enqueue(&mut queue, &mut queued, &old_edge.source_node_id, Propagation::ProofEvidence);
            }
        }
    }
    for (id, new_edge) in &new_edges {
        if !old_edges.contains_key(id) {
            insert_change(&mut changes, id.clone(), "edge_added", None, None);
            let target = new_nodes.get(&new_edge.target_node_id).copied();
            let action = relation_event_action(new_edge, target);
            add_reason(
                &mut actions,
                &mut reasons,
                &new_edge.source_node_id,
                action,
                format!("dependency_edge_added:{}", new_edge.relation),
            );
            if action == Action::Reprove || action == Action::Revalidate {
                enqueue(&mut queue, &mut queued, &new_edge.source_node_id, Propagation::ProofEvidence);
            }
        }
    }

    if !new.dependency_inventory_complete {
        for node in &new.nodes {
            if is_formal(node) {
                add_reason(
                    &mut actions,
                    &mut reasons,
                    &node.node_id,
                    Action::UnknownConservative,
                    "dependency_inventory_incomplete_global_conservative_fallback".into(),
                );
            }
        }
    }

    // Reverse dependency traversal: each source node consumes its target node.
    // We propagate a verified-state change to its own consumers without treating
    // proof-term occurrence as logical necessity or forcing unrelated reproving.
    let mut visit_count = 0usize;
    while let Some((changed_dependency_id, propagation)) = queue.pop_front() {
        visit_count += 1;
        if visit_count > (old.edges.len() + new.edges.len() + old.nodes.len() + new.nodes.len() + 1) * 8 {
            return Err("impact propagation exceeded the finite graph bound".into());
        }
        let mut matching_edges: Vec<&Edge> = new.edges.iter()
            .filter(|edge| edge.target_node_id == changed_dependency_id)
            .collect();
        matching_edges.sort_by(|a, b| a.edge_id.cmp(&b.edge_id));

        for edge in matching_edges {
            let Some(target) = new_nodes.get(&changed_dependency_id).copied() else {
                continue;
            };
            let Some(action) = action_for_propagation(&edge.relation, target, propagation) else {
                continue;
            };
            let source = new_nodes.get(&edge.source_node_id).copied();
            if source.is_none() {
                continue;
            }
            let source = source.unwrap();
            add_reason(
                &mut actions,
                &mut reasons,
                &edge.source_node_id,
                action,
                format!("transitive_impact:{} via {}", changed_dependency_id, edge.relation),
            );
            if action == Action::Rebuild {
                enqueue(&mut queue, &mut queued, &edge.source_node_id, Propagation::Environment);
            } else if action == Action::RetrievalRefresh {
                enqueue(&mut queue, &mut queued, &edge.source_node_id, Propagation::Retrieval);
            } else if action == Action::Reprove || action == Action::Revalidate {
                enqueue(&mut queue, &mut queued, &edge.source_node_id, Propagation::ProofEvidence);
                if source.declaration_kind == "definition" && source.reducibility == "transparent" {
                    enqueue(&mut queue, &mut queued, &edge.source_node_id, Propagation::TransparentBody);
                }
            }
        }
    }

    if !old.dependency_inventory_complete || !new.dependency_inventory_complete {
        for node in &new.nodes {
            if is_formal(node) {
                add_reason(
                    &mut actions,
                    &mut reasons,
                    &node.node_id,
                    Action::UnknownConservative,
                    "old_or_new_dependency_inventory_incomplete".into(),
                );
            }
        }
    }

    let mut changes = changes;
    changes.sort_by(|a, b| a.change_id.cmp(&b.change_id).then(a.change_kind.cmp(&b.change_kind)));
    let impacts: Vec<Impact> = actions
        .iter()
        .map(|(node_id, action)| Impact {
            node_id: node_id.clone(),
            action: action.as_str().to_owned(),
            reasons: reasons.get(node_id).map(|set| set.iter().cloned().collect()).unwrap_or_default(),
        })
        .collect();
    let mut action_counts = BTreeMap::<String, usize>::new();
    for impact in &impacts {
        *action_counts.entry(impact.action.clone()).or_default() += 1;
    }
    let conservative_mode = !old.dependency_inventory_complete || !new.dependency_inventory_complete;
    Ok(Report {
        schema_version: REPORT_VERSION.into(),
        old_snapshot: SnapshotIdentity {
            snapshot_id: old.snapshot_id,
            repository_commit: old.repository_commit,
            environment_sha256: old.environment_sha256,
            raw_snapshot_sha256: digest(old_bytes),
        },
        new_snapshot: SnapshotIdentity {
            snapshot_id: new.snapshot_id,
            repository_commit: new.repository_commit,
            environment_sha256: new.environment_sha256,
            raw_snapshot_sha256: digest(new_bytes),
        },
        dependency_inventory_complete: old.dependency_inventory_complete && new.dependency_inventory_complete,
        conservative_mode,
        changes,
        impacts,
        action_counts,
        summary_note: "Impact output is a conservative planning result, not proof that a retained artifact is valid. Any action above retain requires the corresponding replay/repair before release; incomplete inventories force unknown_conservative for every formal declaration.".into(),
    })
}

fn run_self_tests() -> Result<(), String> {
    fn node(id: &str, kind: &str, declaration_kind: &str, reducibility: &str, statement: &str, body: &str, proof: &str, source: &str) -> Node {
        Node {
            node_id: id.into(),
            kind: kind.into(),
            declaration_kind: declaration_kind.into(),
            reducibility: reducibility.into(),
            statement_sha256: if kind == "formal_declaration" { Some(statement.into()) } else { None },
            body_sha256: Some(body.into()),
            proof_sha256: Some(proof.into()),
            axiom_manifest_sha256: Some("a".repeat(64)),
            source_sha256: Some(source.into()),
        }
    }
    fn edge(id: &str, from: &str, to: &str, relation: &str) -> Edge {
        Edge {
            edge_id: id.into(),
            source_node_id: from.into(),
            target_node_id: to.into(),
            relation: relation.into(),
        }
    }
    fn snap(id: &str, env: &str, nodes: Vec<Node>, edges: Vec<Edge>, complete: bool) -> Snapshot {
        Snapshot {
            schema_version: SNAPSHOT_VERSION.into(),
            snapshot_id: id.into(),
            repository_commit: "1".repeat(40),
            environment_sha256: env.into(),
            dependency_inventory_complete: complete,
            nodes,
            edges,
        }
    }
    let base_nodes = vec![
        node("helper", "formal_declaration", "theorem", "opaque", &"1".repeat(64), &"2".repeat(64), &"3".repeat(64), &"4".repeat(64)),
        node("root", "formal_declaration", "theorem", "opaque", &"5".repeat(64), &"6".repeat(64), &"7".repeat(64), &"8".repeat(64)),
        node("consumer", "formal_declaration", "theorem", "opaque", &"9".repeat(64), &"a".repeat(64), &"b".repeat(64), &"c".repeat(64)),
        node("transparent", "formal_declaration", "definition", "transparent", &"d".repeat(64), &"e".repeat(64), &"f".repeat(64), &"0".repeat(64)),
        node("doc", "document", "document", "not_applicable", &"0".repeat(64), &"1".repeat(64), &"2".repeat(64), &"3".repeat(64)),
    ];
    let base_edges = vec![
        edge("e-root-helper", "root", "helper", "signature_reference"),
        edge("e-consumer-root", "consumer", "root", "proof_term_reference"),
        edge("e-root-transparent", "root", "transparent", "definition_body_reference"),
        edge("e-root-doc", "root", "doc", "documentation_reference"),
    ];
    let old = snap("old", &"d".repeat(64), base_nodes.clone(), base_edges.clone(), true);

    let old_bytes = serde_json::to_vec(&old).map_err(|e| e.to_string())?;
    let same_bytes = serde_json::to_vec(&old).map_err(|e| e.to_string())?;
    let unchanged = analyze(&old_bytes, &same_bytes)?;
    if unchanged.impacts.iter().any(|x| x.action != "retain") {
        return Err("no-op snapshot diff failed to retain all declarations".into());
    }
    println!("SELF_TEST_PASS no-op preserves all declarations");

    let mut changed_nodes = base_nodes.clone();
    changed_nodes.iter_mut().find(|x| x.node_id == "helper").unwrap().statement_sha256 = Some("f".repeat(64));
    let signature = snap("signature-change", &"d".repeat(64), changed_nodes, base_edges.clone(), true);
    let report = analyze(&old_bytes, &serde_json::to_vec(&signature).map_err(|e| e.to_string())?)?;
    let actions: BTreeMap<_, _> = report.impacts.iter().map(|x| (x.node_id.as_str(), x.action.as_str())).collect();
    if actions.get("root") != Some(&"reprove") || actions.get("consumer") != Some(&"revalidate") {
        return Err(format!("signature change closure incorrect: {actions:?}"));
    }
    println!("SELF_TEST_PASS signature mutation re-proves direct consumer and revalidates downstream proof users");

    let mut proof_nodes = base_nodes.clone();
    proof_nodes.iter_mut().find(|x| x.node_id == "helper").unwrap().proof_sha256 = Some("f".repeat(64));
    let proof = snap("proof-change", &"d".repeat(64), proof_nodes, base_edges.clone(), true);
    let report = analyze(&old_bytes, &serde_json::to_vec(&proof).map_err(|e| e.to_string())?)?;
    let actions: BTreeMap<_, _> = report.impacts.iter().map(|x| (x.node_id.as_str(), x.action.as_str())).collect();
    if actions.get("root") != Some(&"retain") {
        // Root depends on helper's type in this synthetic graph; proof-only update
        // must not be promoted to signature invalidation. No proof-term edge exists.
        return Err(format!("proof-only change over-invalidated type-only consumer: {actions:?}"));
    }
    println!("SELF_TEST_PASS proof-body-only mutation preserves type-only dependents");

    let mut body_nodes = base_nodes.clone();
    body_nodes.iter_mut().find(|x| x.node_id == "transparent").unwrap().body_sha256 = Some("f".repeat(64));
    let body = snap("body-change", &"d".repeat(64), body_nodes, base_edges.clone(), true);
    let report = analyze(&old_bytes, &serde_json::to_vec(&body).map_err(|e| e.to_string())?)?;
    if !report.impacts.iter().any(|x| x.node_id == "root" && x.action == "reprove") {
        return Err("transparent definition change did not re-prove its consumer".into());
    }
    println!("SELF_TEST_PASS transparent definition-body mutation re-proves consumers");

    let module_nodes = base_nodes.clone();
    let environment = snap("environment-change", &"e".repeat(64), module_nodes, base_edges.clone(), true);
    let report = analyze(&old_bytes, &serde_json::to_vec(&environment).map_err(|e| e.to_string())?)?;
    if report.impacts.iter().filter(|x| x.action == "rebuild").count() != 4 {
        return Err(format!("environment change did not conservatively rebuild all formal declarations: {:?}", report.action_counts));
    }
    println!("SELF_TEST_PASS global environment change rebuilds every formal declaration");

    let mut doc_nodes = base_nodes.clone();
    doc_nodes.iter_mut().find(|x| x.node_id == "doc").unwrap().source_sha256 = Some("f".repeat(64));
    let doc = snap("doc-only-change", &"d".repeat(64), doc_nodes, base_edges.clone(), true);
    let report = analyze(&old_bytes, &serde_json::to_vec(&doc).map_err(|e| e.to_string())?)?;
    let actions: BTreeMap<_, _> = report.impacts.iter().map(|x| (x.node_id.as_str(), x.action.as_str())).collect();
    if actions.get("root") != Some(&"retrieval_refresh") {
        return Err(format!("documentation-only change should refresh retrieval only: {actions:?}"));
    }
    println!("SELF_TEST_PASS documentation-only change refreshes retrieval without invalidating proofs");

    let incomplete = snap("incomplete", &"d".repeat(64), base_nodes.clone(), base_edges.clone(), false);
    let report = analyze(&old_bytes, &serde_json::to_vec(&incomplete).map_err(|e| e.to_string())?)?;
    if report.impacts.iter().filter(|x| x.action == "unknown_conservative").count() != 4 {
        return Err("incomplete dependency inventory failed closed".into());
    }
    println!("SELF_TEST_PASS incomplete dependency inventory marks all declarations unknown_conservative");

    Ok(())
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        run_self_tests()?;
        println!("PITB_PROOF_IMPACT_SELF_TEST_PASS");
        return Ok(());
    }
    if args.len() != 4 {
        return Err(format!(
            "usage: {} <old-snapshot.json> <new-snapshot.json> <report.json> OR --self-test",
            args.first().map(String::as_str).unwrap_or("pitb-proof-impact")
        ));
    }
    let old_bytes = fs::read(&args[1]).map_err(|e| format!("read old snapshot: {e}"))?;
    let new_bytes = fs::read(&args[2]).map_err(|e| format!("read new snapshot: {e}"))?;
    let report = analyze(&old_bytes, &new_bytes)?;
    let output = serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize report: {e}"))?;
    fs::write(&args[3], output).map_err(|e| format!("write impact report: {e}"))?;
    println!("PITB_PROOF_IMPACT_ANALYSIS_OK");
    Ok(())
}

fn main() -> ExitCode {
    match execute() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => fail(message),
    }
}
