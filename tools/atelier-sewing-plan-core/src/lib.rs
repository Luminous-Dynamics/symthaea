//! Deterministic validation for proposed sewing-assembly operation graphs.
//!
//! This crate validates references, topology, dependencies and declared machine
//! capabilities. It does not schedule a factory, infer construction intent,
//! validate physical sewability, verify machine safety or authorize actuation.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum OperationKind {
    Cut,
    Mark,
    Join,
    EdgeFinish,
    Dart,
    Gather,
    Attach,
    Insert,
    Press,
    Turn,
    Topstitch,
    Inspect,
    RepairRework,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PatternPart {
    pub part_id: String,
    pub pattern_piece_id: String,
    pub revision: u64,
    pub sha256: String,
    pub edge_ids: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct EdgeRef {
    pub part_id: String,
    pub edge_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SewingInterface {
    pub interface_id: String,
    pub edge_refs: Vec<EdgeRef>,
    pub seam_allowance_micrometres: u64,
    pub tolerance_micrometres: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MachineProfile {
    pub profile_id: String,
    pub revision: u64,
    pub sha256: String,
    pub supported_operations: BTreeSet<OperationKind>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SewingOperation {
    pub operation_id: String,
    pub kind: OperationKind,
    pub predecessor_ids: Vec<String>,
    pub part_refs: Vec<String>,
    pub interface_refs: Vec<String>,
    pub machine_profile_ref: Option<String>,
    pub manual: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SewingAssemblyPlan {
    pub plan_id: String,
    pub revision: u64,
    pub plan_sha256: String,
    pub source_brief_sha256: String,
    pub source_pattern_sha256: String,
    pub parts: Vec<PatternPart>,
    pub interfaces: Vec<SewingInterface>,
    pub machine_profiles: Vec<MachineProfile>,
    pub operations: Vec<SewingOperation>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationIssue {
    pub code: &'static str,
    pub path: String,
    pub detail: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationReport {
    pub issues: Vec<ValidationIssue>,
    pub topological_order: Option<Vec<String>>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }
}

pub fn validate_plan(
    plan: &SewingAssemblyPlan,
    expected_brief_sha256: &str,
    expected_pattern_sha256: &str,
) -> ValidationReport {
    let mut issues = Vec::new();
    if plan.plan_id.trim().is_empty() {
        issue(&mut issues, "empty_plan_id", "plan_id", "plan identifier must be non-empty");
    }
    if plan.revision == 0 {
        issue(&mut issues, "zero_plan_revision", "revision", "plan revision must be greater than zero");
    }
    if !is_sha256_hex(&plan.plan_sha256) {
        issue(&mut issues, "invalid_plan_digest", "plan_sha256", "expected 64 hexadecimal characters");
    }
    if !is_sha256_hex(&plan.source_brief_sha256) {
        issue(&mut issues, "invalid_brief_digest", "source_brief_sha256", "expected 64 hexadecimal characters");
    }
    if !is_sha256_hex(&plan.source_pattern_sha256) {
        issue(&mut issues, "invalid_pattern_digest", "source_pattern_sha256", "expected 64 hexadecimal characters");
    }
    if !plan.source_brief_sha256.eq_ignore_ascii_case(expected_brief_sha256) {
        issue(&mut issues, "stale_brief_revision", "source_brief_sha256", "plan is not bound to the expected brief digest");
    }
    if !plan.source_pattern_sha256.eq_ignore_ascii_case(expected_pattern_sha256) {
        issue(&mut issues, "stale_pattern_revision", "source_pattern_sha256", "plan is not bound to the expected pattern digest");
    }
    if plan.parts.is_empty() {
        issue(&mut issues, "empty_parts", "parts", "at least one pattern part is required");
    }
    if plan.operations.is_empty() {
        issue(&mut issues, "empty_operations", "operations", "at least one assembly operation is required");
    }

    let mut part_map = BTreeMap::new();
    for (index, part) in plan.parts.iter().enumerate() {
        let path = format!("parts[{index}]");
        if part.part_id.trim().is_empty() {
            issue(&mut issues, "empty_part_id", format!("{path}.part_id"), "part id must be non-empty");
        }
        if part.pattern_piece_id.trim().is_empty() {
            issue(&mut issues, "empty_pattern_piece_id", format!("{path}.pattern_piece_id"), "pattern piece id must be non-empty");
        }
        if part.revision == 0 {
            issue(&mut issues, "zero_part_revision", format!("{path}.revision"), "part revision must be greater than zero");
        }
        if !is_sha256_hex(&part.sha256) {
            issue(&mut issues, "invalid_part_digest", format!("{path}.sha256"), "expected 64 hexadecimal characters");
        }
        if part_map.insert(part.part_id.as_str(), part).is_some() {
            issue(&mut issues, "duplicate_part_id", format!("{path}.part_id"), "part id is duplicated");
        }
        let mut edge_ids = BTreeSet::new();
        for (edge_index, edge_id) in part.edge_ids.iter().enumerate() {
            if edge_id.trim().is_empty() || !edge_ids.insert(edge_id) {
                issue(&mut issues, "invalid_or_duplicate_edge_id", format!("{path}.edge_ids[{edge_index}]"), "edge id must be non-empty and unique within its part");
            }
        }
    }

    let mut interface_map = BTreeMap::new();
    for (index, interface) in plan.interfaces.iter().enumerate() {
        let path = format!("interfaces[{index}]");
        if interface.interface_id.trim().is_empty() {
            issue(&mut issues, "empty_interface_id", format!("{path}.interface_id"), "interface id must be non-empty");
        }
        if interface.edge_refs.len() < 2 {
            issue(&mut issues, "insufficient_interface_edges", format!("{path}.edge_refs"), "an assembly interface requires at least two edge references");
        }
        if interface.seam_allowance_micrometres > 100_000 {
            issue(&mut issues, "seam_allowance_out_of_range", format!("{path}.seam_allowance_micrometres"), "seam allowance exceeds the declared 100 mm guardrail");
        }
        if interface.tolerance_micrometres == 0 || interface.tolerance_micrometres > 20_000 {
            issue(&mut issues, "interface_tolerance_out_of_range", format!("{path}.tolerance_micrometres"), "interface tolerance must be in (0, 20 mm]");
        }
        let mut seen_edges = BTreeSet::new();
        for (edge_index, edge_ref) in interface.edge_refs.iter().enumerate() {
            if !seen_edges.insert(edge_ref) {
                issue(&mut issues, "duplicate_interface_edge", format!("{path}.edge_refs[{edge_index}]"), "same part edge appears more than once in the interface");
            }
            match part_map.get(edge_ref.part_id.as_str()) {
                None => issue(&mut issues, "unknown_interface_part", format!("{path}.edge_refs[{edge_index}].part_id"), "interface references an unknown part"),
                Some(part) if !part.edge_ids.contains(&edge_ref.edge_id) => {
                    issue(&mut issues, "unknown_pattern_edge", format!("{path}.edge_refs[{edge_index}].edge_id"), "interface references an edge absent from the referenced pattern part");
                }
                Some(_) => {}
            }
        }
        if interface_map.insert(interface.interface_id.as_str(), interface).is_some() {
            issue(&mut issues, "duplicate_interface_id", format!("{path}.interface_id"), "interface id is duplicated");
        }
    }

    let mut machine_map = BTreeMap::new();
    for (index, machine) in plan.machine_profiles.iter().enumerate() {
        let path = format!("machine_profiles[{index}]");
        if machine.profile_id.trim().is_empty() {
            issue(&mut issues, "empty_machine_profile_id", format!("{path}.profile_id"), "machine profile id must be non-empty");
        }
        if machine.revision == 0 {
            issue(&mut issues, "zero_machine_profile_revision", format!("{path}.revision"), "machine profile revision must be greater than zero");
        }
        if !is_sha256_hex(&machine.sha256) {
            issue(&mut issues, "invalid_machine_profile_digest", format!("{path}.sha256"), "expected 64 hexadecimal characters");
        }
        if machine_map.insert(machine.profile_id.as_str(), machine).is_some() {
            issue(&mut issues, "duplicate_machine_profile_id", format!("{path}.profile_id"), "machine profile id is duplicated");
        }
    }

    let mut operation_map = BTreeMap::new();
    for (index, operation) in plan.operations.iter().enumerate() {
        let path = format!("operations[{index}]");
        if operation.operation_id.trim().is_empty() {
            issue(&mut issues, "empty_operation_id", format!("{path}.operation_id"), "operation id must be non-empty");
        }
        if operation_map.insert(operation.operation_id.as_str(), operation).is_some() {
            issue(&mut issues, "duplicate_operation_id", format!("{path}.operation_id"), "operation id is duplicated");
        }
        if operation.manual && operation.machine_profile_ref.is_some() {
            issue(&mut issues, "manual_operation_has_machine_profile", format!("{path}.machine_profile_ref"), "manual operation must not silently declare machine execution");
        }
        if !operation.manual && operation.machine_profile_ref.is_none() {
            issue(&mut issues, "machine_profile_required", format!("{path}.machine_profile_ref"), "non-manual operation requires an exact machine profile reference");
        }
        if let Some(profile_id) = &operation.machine_profile_ref {
            match machine_map.get(profile_id.as_str()) {
                None => issue(&mut issues, "unknown_machine_profile", format!("{path}.machine_profile_ref"), "operation references an unknown machine profile"),
                Some(machine) if !machine.supported_operations.contains(&operation.kind) => {
                    issue(&mut issues, "unsupported_machine_operation", format!("{path}.kind"), "machine profile does not declare this operation kind");
                }
                Some(_) => {}
            }
        }
        let mut part_refs = BTreeSet::new();
        for (part_index, part_id) in operation.part_refs.iter().enumerate() {
            if !part_refs.insert(part_id) {
                issue(&mut issues, "duplicate_operation_part_ref", format!("{path}.part_refs[{part_index}]"), "part reference is duplicated");
            }
            if !part_map.contains_key(part_id.as_str()) {
                issue(&mut issues, "unknown_operation_part", format!("{path}.part_refs[{part_index}]"), "operation references an unknown part");
            }
        }
        let mut interface_refs = BTreeSet::new();
        for (interface_index, interface_id) in operation.interface_refs.iter().enumerate() {
            if !interface_refs.insert(interface_id) {
                issue(&mut issues, "duplicate_operation_interface_ref", format!("{path}.interface_refs[{interface_index}]"), "interface reference is duplicated");
            }
            if !interface_map.contains_key(interface_id.as_str()) {
                issue(&mut issues, "unknown_operation_interface", format!("{path}.interface_refs[{interface_index}]"), "operation references an unknown interface");
            }
        }
    }

    for (index, operation) in plan.operations.iter().enumerate() {
        let path = format!("operations[{index}]");
        let mut seen_predecessors = BTreeSet::new();
        for (pred_index, predecessor_id) in operation.predecessor_ids.iter().enumerate() {
            if predecessor_id == &operation.operation_id {
                issue(&mut issues, "self_predecessor", format!("{path}.predecessor_ids[{pred_index}]"), "operation cannot precede itself");
            }
            if !seen_predecessors.insert(predecessor_id) {
                issue(&mut issues, "duplicate_predecessor", format!("{path}.predecessor_ids[{pred_index}]"), "predecessor is duplicated");
            }
            if !operation_map.contains_key(predecessor_id.as_str()) {
                issue(&mut issues, "unknown_predecessor", format!("{path}.predecessor_ids[{pred_index}]"), "operation references an unknown predecessor");
            }
        }
    }

    let topological_order = topological_sort(&plan.operations);
    if topological_order.is_none() {
        issue(&mut issues, "operation_cycle", "operations", "precedence graph contains a cycle");
    }

    ValidationReport { issues, topological_order }
}

fn topological_sort(operations: &[SewingOperation]) -> Option<Vec<String>> {
    let mut indegree: BTreeMap<&str, usize> = operations
        .iter()
        .map(|operation| (operation.operation_id.as_str(), operation.predecessor_ids.len()))
        .collect();
    let mut successors: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for operation in operations {
        for predecessor in &operation.predecessor_ids {
            successors.entry(predecessor.as_str()).or_default().push(operation.operation_id.as_str());
        }
    }
    let mut ready: VecDeque<&str> = indegree
        .iter()
        .filter_map(|(id, degree)| if *degree == 0 { Some(*id) } else { None })
        .collect();
    let mut order = Vec::with_capacity(operations.len());
    while let Some(id) = ready.pop_front() {
        order.push(id.to_owned());
        if let Some(children) = successors.get(id) {
            for child in children {
                let degree = indegree.get_mut(child)?;
                *degree = degree.checked_sub(1)?;
                if *degree == 0 {
                    ready.push_back(child);
                }
            }
        }
    }
    if order.len() == operations.len() { Some(order) } else { None }
}

fn issue(issues: &mut Vec<ValidationIssue>, code: &'static str, path: impl Into<String>, detail: impl Into<String>) {
    issues.push(ValidationIssue { code, path: path.into(), detail: detail.into() });
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.chars().all(|character| character.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_plan() -> SewingAssemblyPlan {
        let front = PatternPart {
            part_id: "front".to_owned(),
            pattern_piece_id: "front-piece".to_owned(),
            revision: 1,
            sha256: "1".repeat(64),
            edge_ids: BTreeSet::from(["shoulder".to_owned(), "side".to_owned()]),
        };
        let back = PatternPart {
            part_id: "back".to_owned(),
            pattern_piece_id: "back-piece".to_owned(),
            revision: 1,
            sha256: "2".repeat(64),
            edge_ids: BTreeSet::from(["shoulder".to_owned(), "side".to_owned()]),
        };
        let profile = MachineProfile {
            profile_id: "lockstitch".to_owned(),
            revision: 1,
            sha256: "3".repeat(64),
            supported_operations: BTreeSet::from([OperationKind::Join, OperationKind::Topstitch]),
        };
        let shoulder = SewingInterface {
            interface_id: "shoulder".to_owned(),
            edge_refs: vec![
                EdgeRef { part_id: "front".to_owned(), edge_id: "shoulder".to_owned() },
                EdgeRef { part_id: "back".to_owned(), edge_id: "shoulder".to_owned() },
            ],
            seam_allowance_micrometres: 10_000,
            tolerance_micrometres: 1_500,
        };
        let side = SewingInterface {
            interface_id: "side".to_owned(),
            edge_refs: vec![
                EdgeRef { part_id: "front".to_owned(), edge_id: "side".to_owned() },
                EdgeRef { part_id: "back".to_owned(), edge_id: "side".to_owned() },
            ],
            seam_allowance_micrometres: 10_000,
            tolerance_micrometres: 1_500,
        };
        let join_shoulders = SewingOperation {
            operation_id: "join-shoulders".to_owned(),
            kind: OperationKind::Join,
            predecessor_ids: Vec::new(),
            part_refs: vec!["front".to_owned(), "back".to_owned()],
            interface_refs: vec!["shoulder".to_owned()],
            machine_profile_ref: Some("lockstitch".to_owned()),
            manual: false,
        };
        let join_sides = SewingOperation {
            operation_id: "join-sides".to_owned(),
            kind: OperationKind::Join,
            predecessor_ids: vec!["join-shoulders".to_owned()],
            part_refs: vec!["front".to_owned(), "back".to_owned()],
            interface_refs: vec!["side".to_owned()],
            machine_profile_ref: Some("lockstitch".to_owned()),
            manual: false,
        };
        SewingAssemblyPlan {
            plan_id: "synthetic-overshirt".to_owned(),
            revision: 1,
            plan_sha256: "a".repeat(64),
            source_brief_sha256: "b".repeat(64),
            source_pattern_sha256: "c".repeat(64),
            parts: vec![front, back],
            interfaces: vec![shoulder, side],
            machine_profiles: vec![profile],
            operations: vec![join_shoulders, join_sides],
        }
    }

    #[test]
    fn valid_plan_returns_deterministic_topological_order() {
        let plan = base_plan();
        let report = validate_plan(&plan, &"b".repeat(64), &"c".repeat(64));
        assert!(report.is_valid(), "{:?}", report.issues);
        assert_eq!(
            report.topological_order.unwrap(),
            vec!["join-shoulders".to_owned(), "join-sides".to_owned()]
        );
    }

    #[test]
    fn missing_pattern_edge_is_rejected() {
        let mut plan = base_plan();
        plan.interfaces[0].edge_refs[0].edge_id = "missing-edge".to_owned();
        let report = validate_plan(&plan, &"b".repeat(64), &"c".repeat(64));
        assert!(report.issues.iter().any(|v| v.code == "unknown_pattern_edge"));
    }

    #[test]
    fn unknown_predecessor_is_rejected() {
        let mut plan = base_plan();
        plan.operations[0].predecessor_ids.push("not-an-operation".to_owned());
        let report = validate_plan(&plan, &"b".repeat(64), &"c".repeat(64));
        assert!(report.issues.iter().any(|v| v.code == "unknown_predecessor"));
    }

    #[test]
    fn operation_cycle_is_rejected() {
        let mut plan = base_plan();
        plan.operations[0].predecessor_ids.push("join-sides".to_owned());
        let report = validate_plan(&plan, &"b".repeat(64), &"c".repeat(64));
        assert!(report.issues.iter().any(|v| v.code == "operation_cycle"));
    }

    #[test]
    fn unsupported_machine_operation_is_rejected() {
        let mut plan = base_plan();
        plan.operations[0].kind = OperationKind::Dart;
        let report = validate_plan(&plan, &"b".repeat(64), &"c".repeat(64));
        assert!(report.issues.iter().any(|v| v.code == "unsupported_machine_operation"));
    }

    #[test]
    fn unknown_machine_profile_is_rejected() {
        let mut plan = base_plan();
        plan.operations[0].machine_profile_ref = Some("missing-machine".to_owned());
        let report = validate_plan(&plan, &"b".repeat(64), &"c".repeat(64));
        assert!(report.issues.iter().any(|v| v.code == "unknown_machine_profile"));
    }

    #[test]
    fn stale_brief_or_pattern_digest_is_rejected() {
        let plan = base_plan();
        let report = validate_plan(&plan, &"d".repeat(64), &"e".repeat(64));
        assert!(report.issues.iter().any(|v| v.code == "stale_brief_revision"));
        assert!(report.issues.iter().any(|v| v.code == "stale_pattern_revision"));
    }

    #[test]
    fn non_manual_operation_requires_machine_profile() {
        let mut plan = base_plan();
        plan.operations[0].machine_profile_ref = None;
        let report = validate_plan(&plan, &"b".repeat(64), &"c".repeat(64));
        assert!(report.issues.iter().any(|v| v.code == "machine_profile_required"));
    }

    #[test]
    fn manual_operation_cannot_claim_machine_profile() {
        let mut plan = base_plan();
        plan.operations[0].manual = true;
        let report = validate_plan(&plan, &"b".repeat(64), &"c".repeat(64));
        assert!(report.issues.iter().any(|v| v.code == "manual_operation_has_machine_profile"));
    }

    #[test]
    fn duplicate_edge_reference_is_rejected() {
        let mut plan = base_plan();
        plan.interfaces[0].edge_refs.push(plan.interfaces[0].edge_refs[0].clone());
        let report = validate_plan(&plan, &"b".repeat(64), &"c".repeat(64));
        assert!(report.issues.iter().any(|v| v.code == "duplicate_interface_edge"));
    }

    #[test]
    fn interface_tolerance_must_be_positive_and_bounded() {
        let mut plan = base_plan();
        plan.interfaces[0].tolerance_micrometres = 0;
        let report = validate_plan(&plan, &"b".repeat(64), &"c".repeat(64));
        assert!(report.issues.iter().any(|v| v.code == "interface_tolerance_out_of_range"));
    }
}
