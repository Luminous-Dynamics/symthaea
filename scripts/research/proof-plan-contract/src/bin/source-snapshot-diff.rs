use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

#[derive(Debug, Clone, Deserialize)]
struct Snapshot {
    schema_version: String,
    source_repository: String,
    source_revision: String,
    lean_toolchain: String,
    lean_revision: String,
    declaration: String,
    declaration_kind: String,
    type_dependencies: Vec<String>,
    proof_value_dependencies: Vec<String>,
    constant_info_dependency_closure: Vec<String>,
    unresolved_dependencies: Vec<String>,
    edges: Vec<Edge>,
    #[serde(default)]
    helper_nodes: Vec<HelperNode>,
    trust_profile: TrustProfile,
}

#[derive(Debug, Clone, Deserialize)]
struct TrustProfile {
    closure_status: String,
    transitive_declaration_count: usize,
    axiom_declarations: Vec<String>,
    unsafe_declarations: Vec<String>,
    partial_declarations: Vec<String>,
    opaque_declarations: Vec<String>,
    sorry_ax_present: bool,
    unresolved_declarations: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct HelperNode {
    root_declaration: String,
    node_role: String,
    declaration: String,
    declaration_kind: String,
    proof_value_available: bool,
    type_dependencies: Vec<String>,
    proof_value_dependencies: Vec<String>,
    constant_info_dependency_closure: Vec<String>,
    unresolved_dependencies: Vec<String>,
    edges: Vec<Edge>,
    trust_profile: TrustProfile,
}

#[derive(Debug, Clone, Deserialize)]
struct Edge {
    relation: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct CohortIdentity {
    source_repository: String,
    source_revision: String,
    lean_toolchain: String,
    lean_revision: String,
}

#[derive(Debug, Clone, Serialize)]
struct SetDiff {
    baseline_count: usize,
    candidate_count: usize,
    intersection_count: usize,
    baseline_only: Vec<String>,
    candidate_only: Vec<String>,
    jaccard: f64,
}

#[derive(Debug, Clone, Serialize)]
struct DeclarationDiff {
    declaration: String,
    baseline_declaration_kind: String,
    candidate_declaration_kind: String,
    classification: String,
    type_dependencies: SetDiff,
    proof_value_dependencies: SetDiff,
    dependency_closure: SetDiff,
    baseline_closure_reconciles: bool,
    candidate_closure_reconciles: bool,
    baseline_unresolved_count: usize,
    candidate_unresolved_count: usize,
    baseline_edge_count: usize,
    candidate_edge_count: usize,
    edge_relation_counts_baseline: BTreeMap<String, usize>,
    edge_relation_counts_candidate: BTreeMap<String, usize>,
    helper_nodes_baseline: Vec<String>,
    helper_nodes_candidate: Vec<String>,
    helper_expanded_proof_dependencies: SetDiff,
    helper_expansion_explains_direct_drift: bool,
    axiom_declarations: SetDiff,
    unsafe_declarations: SetDiff,
    partial_declarations: SetDiff,
    opaque_declarations: SetDiff,
    baseline_sorry_ax_present: bool,
    candidate_sorry_ax_present: bool,
    baseline_transitive_declaration_count: usize,
    candidate_transitive_declaration_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct DifferentialReportV1 {
    schema_version: String,
    comparison_kind: String,
    baseline: CohortIdentity,
    candidate: CohortIdentity,
    source_revisions_differ: bool,
    lean_toolchains_differ: bool,
    declarations: Vec<DeclarationDiff>,
    claim_ceiling: String,
}

fn string_set(values: &[String]) -> BTreeSet<String> {
    values.iter().cloned().collect()
}

fn union_set(a: &BTreeSet<String>, b: &BTreeSet<String>) -> BTreeSet<String> {
    a.union(b).cloned().collect()
}

fn set_diff(baseline: &BTreeSet<String>, candidate: &BTreeSet<String>) -> SetDiff {
    let intersection = baseline.intersection(candidate).count();
    let union = baseline.union(candidate).count();
    SetDiff {
        baseline_count: baseline.len(),
        candidate_count: candidate.len(),
        intersection_count: intersection,
        baseline_only: baseline.difference(candidate).cloned().collect(),
        candidate_only: candidate.difference(baseline).cloned().collect(),
        jaccard: if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        },
    }
}

fn closure_reconciles(snapshot: &Snapshot) -> bool {
    let type_refs = string_set(&snapshot.type_dependencies);
    let proof_refs = string_set(&snapshot.proof_value_dependencies);
    let closure = string_set(&snapshot.constant_info_dependency_closure);
    union_set(&type_refs, &proof_refs) == closure
}

fn helper_expanded_proof_dependencies(snapshot: &Snapshot) -> BTreeSet<String> {
    let mut dependencies = string_set(&snapshot.proof_value_dependencies);
    for helper in &snapshot.helper_nodes {
        dependencies.extend(helper.proof_value_dependencies.iter().cloned());
    }
    dependencies
}

fn helper_names(snapshot: &Snapshot) -> Vec<String> {
    snapshot
        .helper_nodes
        .iter()
        .map(|helper| helper.declaration.clone())
        .collect()
}

fn helper_closure_reconciles(helper: &HelperNode) -> bool {
    let type_refs = string_set(&helper.type_dependencies);
    let proof_refs = string_set(&helper.proof_value_dependencies);
    let closure = string_set(&helper.constant_info_dependency_closure);
    union_set(&type_refs, &proof_refs) == closure
}

fn helper_expansion_explains_direct_drift(
    type_diff: &SetDiff,
    direct_proof_diff: &SetDiff,
    helper_expanded_diff: &SetDiff,
) -> bool {
    let type_interface_stable =
        type_diff.baseline_only.is_empty() && type_diff.candidate_only.is_empty();
    let direct_proof_set_shifted =
        !direct_proof_diff.baseline_only.is_empty() || !direct_proof_diff.candidate_only.is_empty();
    type_interface_stable
        && direct_proof_set_shifted
        && direct_proof_diff.jaccard < 0.5
        && helper_expanded_diff.jaccard >= 0.9
}

fn relation_counts(snapshot: &Snapshot) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for edge in &snapshot.edges {
        *counts.entry(edge.relation.clone()).or_insert(0) += 1;
    }
    counts
}

fn classify(type_diff: &SetDiff, proof_diff: &SetDiff, closure_diff: &SetDiff) -> &'static str {
    let type_stable = type_diff.baseline_only.is_empty() && type_diff.candidate_only.is_empty();
    let proof_stable = proof_diff.baseline_only.is_empty() && proof_diff.candidate_only.is_empty();
    let closure_stable =
        closure_diff.baseline_only.is_empty() && closure_diff.candidate_only.is_empty();

    match (type_stable, proof_stable, closure_stable) {
        (true, true, true) => "identical_type_proof_and_closure_sets",
        (true, false, false) => "stable_type_interface_proof_value_drift",
        (false, true, false) => "type_interface_drift_stable_proof_values",
        _ => "type_and_or_closure_dependency_drift",
    }
}

fn load_cohort(path: &Path) -> Result<(CohortIdentity, BTreeMap<String, Snapshot>), String> {
    let entries = fs::read_dir(path)
        .map_err(|error| format!("cannot read artifact directory {}: {error}", path.display()))?;
    let mut snapshots = BTreeMap::new();
    let mut identity: Option<CohortIdentity> = None;

    for entry in entries {
        let entry = entry.map_err(|error| format!("cannot read directory entry: {error}"))?;
        let file_path = entry.path();
        if file_path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }

        let input = fs::read_to_string(&file_path)
            .map_err(|error| format!("cannot read {}: {error}", file_path.display()))?;
        let snapshot: Snapshot = serde_json::from_str(&input)
            .map_err(|error| format!("invalid snapshot {}: {error}", file_path.display()))?;

        if snapshot.schema_version != "1.0.0" {
            return Err(format!(
                "{} has unsupported schema version {}",
                file_path.display(),
                snapshot.schema_version
            ));
        }

        let current_identity = CohortIdentity {
            source_repository: snapshot.source_repository.clone(),
            source_revision: snapshot.source_revision.clone(),
            lean_toolchain: snapshot.lean_toolchain.clone(),
            lean_revision: snapshot.lean_revision.clone(),
        };
        match &identity {
            Some(expected) if expected != &current_identity => {
                return Err(format!(
                    "mixed source/toolchain identities in {}",
                    path.display()
                ));
            }
            None => identity = Some(current_identity),
            _ => {}
        }

        if !closure_reconciles(&snapshot) {
            return Err(format!(
                "type/proof dependency union does not reconcile with ConstantInfo closure for {}",
                snapshot.declaration
            ));
        }
        if snapshot.trust_profile.closure_status != "complete"
            || !snapshot.trust_profile.unresolved_declarations.is_empty()
        {
            return Err(format!(
                "incomplete trust profile for {}: {} unresolved declarations",
                snapshot.declaration,
                snapshot.trust_profile.unresolved_declarations.len()
            ));
        }
        for helper in &snapshot.helper_nodes {
            if helper.trust_profile.closure_status != "complete"
                || !helper.trust_profile.unresolved_declarations.is_empty()
                || helper.root_declaration != snapshot.declaration
                || helper.node_role != "generated_or_private_helper"
                || !helper.proof_value_available
                || !helper.unresolved_dependencies.is_empty()
                || !helper_closure_reconciles(helper)
            {
                return Err(format!(
                    "invalid helper-node receipt {} attached to {}",
                    helper.declaration, snapshot.declaration
                ));
            }
            let _helper_kind = &helper.declaration_kind;
            let _helper_edges = helper.edges.len();
        }

        if snapshots
            .insert(snapshot.declaration.clone(), snapshot)
            .is_some()
        {
            return Err(format!(
                "duplicate declaration snapshot in {}",
                path.display()
            ));
        }
    }

    let identity =
        identity.ok_or_else(|| format!("no JSON snapshots found in {}", path.display()))?;
    Ok((identity, snapshots))
}

fn compare(
    baseline_identity: CohortIdentity,
    baseline: BTreeMap<String, Snapshot>,
    candidate_identity: CohortIdentity,
    candidate: BTreeMap<String, Snapshot>,
) -> Result<DifferentialReportV1, String> {
    let baseline_names = baseline.keys().cloned().collect::<BTreeSet<_>>();
    let candidate_names = candidate.keys().cloned().collect::<BTreeSet<_>>();
    if baseline_names != candidate_names {
        return Err(format!(
            "declaration cohorts differ; baseline_only={:?}; candidate_only={:?}",
            baseline_names
                .difference(&candidate_names)
                .collect::<Vec<_>>(),
            candidate_names
                .difference(&baseline_names)
                .collect::<Vec<_>>()
        ));
    }

    let mut declarations = Vec::new();
    for name in baseline_names {
        let left = baseline.get(&name).expect("baseline key exists");
        let right = candidate.get(&name).expect("candidate key exists");
        if left.source_repository != right.source_repository {
            return Err(format!("source repository mismatch for {name}"));
        }

        let left_type = string_set(&left.type_dependencies);
        let right_type = string_set(&right.type_dependencies);
        let left_proof = string_set(&left.proof_value_dependencies);
        let right_proof = string_set(&right.proof_value_dependencies);
        let left_closure = string_set(&left.constant_info_dependency_closure);
        let right_closure = string_set(&right.constant_info_dependency_closure);

        let type_dependencies = set_diff(&left_type, &right_type);
        let proof_value_dependencies = set_diff(&left_proof, &right_proof);
        let dependency_closure = set_diff(&left_closure, &right_closure);
        let classification = classify(
            &type_dependencies,
            &proof_value_dependencies,
            &dependency_closure,
        )
        .to_owned();
        let helper_expanded_diff = set_diff(
            &helper_expanded_proof_dependencies(left),
            &helper_expanded_proof_dependencies(right),
        );
        let helper_expansion_explains_direct_drift = helper_expansion_explains_direct_drift(
            &type_dependencies,
            &proof_value_dependencies,
            &helper_expanded_diff,
        );

        declarations.push(DeclarationDiff {
            declaration: name,
            baseline_declaration_kind: left.declaration_kind.clone(),
            candidate_declaration_kind: right.declaration_kind.clone(),
            classification,
            type_dependencies,
            proof_value_dependencies,
            dependency_closure,
            baseline_closure_reconciles: closure_reconciles(left),
            candidate_closure_reconciles: closure_reconciles(right),
            baseline_unresolved_count: left.unresolved_dependencies.len(),
            candidate_unresolved_count: right.unresolved_dependencies.len(),
            baseline_edge_count: left.edges.len(),
            candidate_edge_count: right.edges.len(),
            edge_relation_counts_baseline: relation_counts(left),
            edge_relation_counts_candidate: relation_counts(right),
            helper_nodes_baseline: helper_names(left),
            helper_nodes_candidate: helper_names(right),
            helper_expanded_proof_dependencies: helper_expanded_diff,
            helper_expansion_explains_direct_drift,
            axiom_declarations: set_diff(
                &string_set(&left.trust_profile.axiom_declarations),
                &string_set(&right.trust_profile.axiom_declarations),
            ),
            unsafe_declarations: set_diff(
                &string_set(&left.trust_profile.unsafe_declarations),
                &string_set(&right.trust_profile.unsafe_declarations),
            ),
            partial_declarations: set_diff(
                &string_set(&left.trust_profile.partial_declarations),
                &string_set(&right.trust_profile.partial_declarations),
            ),
            opaque_declarations: set_diff(
                &string_set(&left.trust_profile.opaque_declarations),
                &string_set(&right.trust_profile.opaque_declarations),
            ),
            baseline_sorry_ax_present: left.trust_profile.sorry_ax_present,
            candidate_sorry_ax_present: right.trust_profile.sorry_ax_present,
            baseline_transitive_declaration_count: left.trust_profile.transitive_declaration_count,
            candidate_transitive_declaration_count: right
                .trust_profile
                .transitive_declaration_count,
        });
    }

    Ok(DifferentialReportV1 {
        schema_version: "1.0.0".to_owned(),
        comparison_kind: "mathlib_source_snapshot_direct_dependency_differential".to_owned(),
        source_revisions_differ: baseline_identity.source_revision
            != candidate_identity.source_revision,
        lean_toolchains_differ: baseline_identity.lean_toolchain
            != candidate_identity.lean_toolchain,
        baseline: baseline_identity,
        candidate: candidate_identity,
        declarations,
        claim_ceiling: "This compares direct type/value dependency sets emitted by two pinned source/toolchain cohorts. It does not isolate Lean compiler effects from Mathlib source changes, establish mathematical-role correctness, or compare against the LeanGraph raw export itself.".to_owned(),
    })
}

fn main() -> ExitCode {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        eprintln!("usage: source-snapshot-diff <baseline-artifact-dir> <candidate-artifact-dir>");
        return ExitCode::from(2);
    }

    let baseline_path = Path::new(&args[0]);
    let candidate_path = Path::new(&args[1]);
    let result = (|| {
        let (baseline_identity, baseline) = load_cohort(baseline_path)?;
        let (candidate_identity, candidate) = load_cohort(candidate_path)?;
        let report = compare(baseline_identity, baseline, candidate_identity, candidate)?;
        serde_json::to_string_pretty(&report)
            .map_err(|error| format!("could not serialize differential report: {error}"))
    })();

    match result {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("SOURCE_SNAPSHOT_DIFFERENTIAL_REJECTED: {error}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(revision: &str, toolchain: &str) -> CohortIdentity {
        CohortIdentity {
            source_repository: "leanprover-community/mathlib4".to_owned(),
            source_revision: revision.to_owned(),
            lean_toolchain: toolchain.to_owned(),
            lean_revision: format!("lean-{toolchain}"),
        }
    }

    fn snapshot(
        declaration: &str,
        type_dependencies: &[&str],
        proof_dependencies: &[&str],
    ) -> Snapshot {
        let closure = type_dependencies
            .iter()
            .chain(proof_dependencies.iter())
            .map(|value| (*value).to_owned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        Snapshot {
            schema_version: "1.0.0".to_owned(),
            source_repository: "leanprover-community/mathlib4".to_owned(),
            source_revision: "fixture".to_owned(),
            lean_toolchain: "lean".to_owned(),
            lean_revision: "fixture".to_owned(),
            declaration: declaration.to_owned(),
            declaration_kind: "theorem".to_owned(),
            type_dependencies: type_dependencies.iter().map(|v| (*v).to_owned()).collect(),
            proof_value_dependencies: proof_dependencies.iter().map(|v| (*v).to_owned()).collect(),
            constant_info_dependency_closure: closure,
            unresolved_dependencies: Vec::new(),
            edges: Vec::new(),
            helper_nodes: Vec::new(),
            trust_profile: TrustProfile {
                closure_status: "complete".to_owned(),
                transitive_declaration_count: 3,
                axiom_declarations: Vec::new(),
                unsafe_declarations: Vec::new(),
                partial_declarations: Vec::new(),
                opaque_declarations: Vec::new(),
                sorry_ax_present: false,
                unresolved_declarations: Vec::new(),
            },
        }
    }

    #[test]
    fn identical_dependency_sets_are_classified_as_stable() {
        let a = snapshot("T", &["A", "B"], &["C"]);
        let b = snapshot("T", &["A", "B"], &["C"]);
        let report = compare(
            identity("v1", "lean-a"),
            BTreeMap::from([("T".to_owned(), a)]),
            identity("v2", "lean-b"),
            BTreeMap::from([("T".to_owned(), b)]),
        )
        .expect("identical declaration inventory");
        assert_eq!(
            report.declarations[0].classification,
            "identical_type_proof_and_closure_sets"
        );
        assert_eq!(report.declarations[0].proof_value_dependencies.jaccard, 1.0);
    }

    #[test]
    fn stable_type_with_changed_proof_value_is_classified_separately() {
        let a = snapshot("T", &["A"], &["B", "C"]);
        let b = snapshot("T", &["A"], &["B", "D"]);
        let report = compare(
            identity("v1", "lean-a"),
            BTreeMap::from([("T".to_owned(), a)]),
            identity("v2", "lean-b"),
            BTreeMap::from([("T".to_owned(), b)]),
        )
        .expect("valid declaration inventories");
        assert_eq!(
            report.declarations[0].classification,
            "stable_type_interface_proof_value_drift"
        );
        assert_eq!(
            report.declarations[0]
                .proof_value_dependencies
                .intersection_count,
            1
        );
    }

    #[test]
    fn mixed_dependency_changes_are_not_flattened_into_one_count() {
        let a = snapshot("T", &["A", "B"], &["C"]);
        let b = snapshot("T", &["A", "D"], &["C", "E"]);
        let report = compare(
            identity("v1", "lean-a"),
            BTreeMap::from([("T".to_owned(), a)]),
            identity("v2", "lean-b"),
            BTreeMap::from([("T".to_owned(), b)]),
        )
        .expect("valid declaration inventories");
        assert_eq!(
            report.declarations[0].classification,
            "type_and_or_closure_dependency_drift"
        );
        assert_eq!(
            report.declarations[0].type_dependencies.intersection_count,
            1
        );
        assert_eq!(
            report.declarations[0]
                .proof_value_dependencies
                .intersection_count,
            1
        );
    }

    #[test]
    fn helper_node_dependencies_are_included_in_expanded_proof_comparison() {
        let mut a = snapshot("Nat.choose_pos", &["Nat"], &["helper"]);
        a.helper_nodes.push(HelperNode {
            root_declaration: "Nat.choose_pos".to_owned(),
            node_role: "generated_or_private_helper".to_owned(),
            declaration: "Nat.choose_pos._f".to_owned(),
            declaration_kind: "definition".to_owned(),
            proof_value_available: true,
            type_dependencies: vec!["Nat".to_owned()],
            proof_value_dependencies: vec!["A".to_owned(), "B".to_owned()],
            constant_info_dependency_closure: vec![
                "A".to_owned(),
                "B".to_owned(),
                "Nat".to_owned(),
            ],
            unresolved_dependencies: Vec::new(),
            edges: Vec::new(),
            trust_profile: TrustProfile {
                closure_status: "complete".to_owned(),
                transitive_declaration_count: 3,
                axiom_declarations: Vec::new(),
                unsafe_declarations: Vec::new(),
                partial_declarations: Vec::new(),
                opaque_declarations: Vec::new(),
                sorry_ax_present: false,
                unresolved_declarations: Vec::new(),
            },
        });
        let mut b = snapshot("Nat.choose_pos", &["Nat"], &["helper"]);
        b.helper_nodes.push(HelperNode {
            root_declaration: "Nat.choose_pos".to_owned(),
            node_role: "generated_or_private_helper".to_owned(),
            declaration: "_private.Nat.choose_pos.match_1_1".to_owned(),
            declaration_kind: "definition".to_owned(),
            proof_value_available: true,
            type_dependencies: vec!["Nat".to_owned()],
            proof_value_dependencies: vec!["A".to_owned(), "B".to_owned()],
            constant_info_dependency_closure: vec![
                "A".to_owned(),
                "B".to_owned(),
                "Nat".to_owned(),
            ],
            unresolved_dependencies: Vec::new(),
            edges: Vec::new(),
            trust_profile: TrustProfile {
                closure_status: "complete".to_owned(),
                transitive_declaration_count: 3,
                axiom_declarations: Vec::new(),
                unsafe_declarations: Vec::new(),
                partial_declarations: Vec::new(),
                opaque_declarations: Vec::new(),
                sorry_ax_present: false,
                unresolved_declarations: Vec::new(),
            },
        });
        let report = compare(
            identity("v1", "lean-a"),
            BTreeMap::from([("Nat.choose_pos".to_owned(), a)]),
            identity("v2", "lean-b"),
            BTreeMap::from([("Nat.choose_pos".to_owned(), b)]),
        )
        .expect("same root declaration inventory");
        assert_eq!(report.declarations[0].helper_nodes_baseline.len(), 1);
        assert_eq!(report.declarations[0].helper_nodes_candidate.len(), 1);
        assert_eq!(
            report.declarations[0]
                .helper_expanded_proof_dependencies
                .jaccard,
            1.0
        );
    }

    #[test]
    fn helper_expansion_stability_signal_detects_generated_helper_repackaging() {
        let type_diff = SetDiff {
            baseline_count: 8,
            candidate_count: 8,
            intersection_count: 8,
            baseline_only: Vec::new(),
            candidate_only: Vec::new(),
            jaccard: 1.0,
        };
        let direct_proof_diff = SetDiff {
            baseline_count: 33,
            candidate_count: 10,
            intersection_count: 9,
            baseline_only: vec!["old-helper".to_owned()],
            candidate_only: vec!["new-helper".to_owned()],
            jaccard: 0.265,
        };
        let helper_expanded_diff = SetDiff {
            baseline_count: 35,
            candidate_count: 37,
            intersection_count: 35,
            baseline_only: Vec::new(),
            candidate_only: vec!["new-helper-expanded-edge".to_owned()],
            jaccard: 0.946,
        };
        assert!(helper_expansion_explains_direct_drift(
            &type_diff,
            &direct_proof_diff,
            &helper_expanded_diff
        ));
    }

    #[test]
    fn helper_expansion_signal_does_not_hide_type_interface_drift() {
        let type_diff = SetDiff {
            baseline_count: 6,
            candidate_count: 6,
            intersection_count: 4,
            baseline_only: vec!["Union.union".to_owned()],
            candidate_only: vec!["Max.max".to_owned()],
            jaccard: 0.5,
        };
        let direct_proof_diff = SetDiff {
            baseline_count: 22,
            candidate_count: 32,
            intersection_count: 20,
            baseline_only: vec!["Union.union".to_owned()],
            candidate_only: vec!["Max.max".to_owned()],
            jaccard: 0.588,
        };
        let helper_expanded_diff = SetDiff {
            baseline_count: 24,
            candidate_count: 34,
            intersection_count: 22,
            baseline_only: vec!["Union.union".to_owned()],
            candidate_only: vec!["Max.max".to_owned()],
            jaccard: 0.611,
        };
        assert!(!helper_expansion_explains_direct_drift(
            &type_diff,
            &direct_proof_diff,
            &helper_expanded_diff
        ));
    }

    #[test]
    fn closure_mismatch_is_rejected_before_comparison() {
        let mut bad = snapshot("T", &["A"], &["B"]);
        bad.constant_info_dependency_closure = vec!["A".to_owned()];
        assert!(!closure_reconciles(&bad));
    }
}
