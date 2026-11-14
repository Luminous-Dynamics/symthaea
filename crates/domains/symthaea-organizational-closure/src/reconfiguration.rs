//! Deterministic structural-change admission for AOBench.
//!
//! This module compares a frozen organization with a proposed organization.
//! It computes exact component/process/edge changes and enforces explicit
//! mutation allowlists, protected roots, process protections, and an edit
//! budget. It does not establish that a candidate is viable, beneficial,
//! authorized in a deployed environment, or actually executed. Those claims
//! belong to the separate viability, receipt, runtime, and experiment layers.

use crate::{
    ComponentClass, ComponentId, DependencyId, OrganizationSpec, ProcessId,
};
use std::collections::BTreeSet;

/// Whether an input/output/dependency relationship is mandatory or optional.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EdgeRequirement {
    Required,
    Optional,
}

/// A process-to-component input edge in a candidate organization.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProcessInputEdge {
    pub process_id: ProcessId,
    pub component_id: ComponentId,
    pub requirement: EdgeRequirement,
}

/// A process-to-component output edge in a candidate organization.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProcessOutputEdge {
    pub process_id: ProcessId,
    pub component_id: ComponentId,
    pub requirement: EdgeRequirement,
}

/// A process-to-external-dependency declaration.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExternalDependencyEdge {
    pub process_id: ProcessId,
    pub dependency_id: DependencyId,
    pub requirement: EdgeRequirement,
}

/// Exact structural changes between the baseline and candidate organizations.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReconfigurationDiff {
    pub added_components: Vec<ComponentId>,
    pub removed_components: Vec<ComponentId>,
    pub changed_component_metadata: Vec<ComponentId>,
    pub added_processes: Vec<ProcessId>,
    pub removed_processes: Vec<ProcessId>,
    pub changed_process_contracts: Vec<ProcessId>,
    pub changed_process_versions: Vec<ProcessId>,
    pub changed_executable_identities: Vec<ProcessId>,
    pub added_inputs: Vec<ProcessInputEdge>,
    pub removed_inputs: Vec<ProcessInputEdge>,
    pub added_outputs: Vec<ProcessOutputEdge>,
    pub removed_outputs: Vec<ProcessOutputEdge>,
    pub added_external_dependencies: Vec<ExternalDependencyEdge>,
    pub removed_external_dependencies: Vec<ExternalDependencyEdge>,
}

impl ReconfigurationDiff {
    /// Counts primitive structural edits. Edge type changes count as one
    /// removal plus one addition; process additions also count their edges.
    pub fn structural_edit_count(&self) -> usize {
        self.added_components.len()
            + self.removed_components.len()
            + self.changed_component_metadata.len()
            + self.added_processes.len()
            + self.removed_processes.len()
            + self.changed_process_versions.len()
            + self.changed_executable_identities.len()
            + self.added_inputs.len()
            + self.removed_inputs.len()
            + self.added_outputs.len()
            + self.removed_outputs.len()
            + self.added_external_dependencies.len()
            + self.removed_external_dependencies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.structural_edit_count() == 0
    }
}

/// Explicitly approved changes for one reconfiguration proposal.
///
/// An empty/default policy allows no structural edits. Protected roots and
/// protected processes cannot be removed or modified even if accidentally
/// included in an allowlist. Component metadata changes are not supported in
/// v0.1; use a new component identity and an independently reviewed migration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReconfigurationPolicy {
    pub allowed_component_additions: BTreeSet<ComponentId>,
    pub allowed_component_removals: BTreeSet<ComponentId>,
    pub allowed_process_additions: BTreeSet<ProcessId>,
    pub allowed_process_removals: BTreeSet<ProcessId>,
    pub allowed_process_contract_changes: BTreeSet<ProcessId>,
    pub allowed_executable_identity_changes: BTreeSet<ProcessId>,
    pub protected_components: BTreeSet<ComponentId>,
    pub protected_processes: BTreeSet<ProcessId>,
    pub max_structural_edits: usize,
}

/// Deterministic outcome of structural admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconfigurationQualification {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconfigurationFindingCode {
    ComponentMapKeyMismatch,
    ProcessMapKeyMismatch,
    ComponentAdditionNotAllowed,
    ComponentRemovalNotAllowed,
    ProtectedComponentAdded,
    ProtectedComponentRemoved,
    ProtectedComponentMutation,
    ComponentMetadataMutationNotAllowed,
    ProcessAdditionNotAllowed,
    ProcessRemovalNotAllowed,
    ProtectedProcessRemoved,
    ProtectedProcessMutation,
    ProcessContractChangeNotAllowed,
    ExecutableIdentityChangeNotAllowed,
    UnknownProcessInput,
    UnknownProcessOutput,
    ProtectedInputEdgeRemoved,
    ProtectedComponentProduced,
    RequiredOptionalInputOverlap,
    RequiredOptionalOutputOverlap,
    RequiredOptionalDependencyOverlap,
    EditBudgetExceeded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconfigurationFinding {
    pub code: ReconfigurationFindingCode,
    pub subject: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconfigurationReport {
    pub qualification: ReconfigurationQualification,
    pub diff: ReconfigurationDiff,
    pub findings: Vec<ReconfigurationFinding>,
}

impl ReconfigurationReport {
    pub fn qualifies_candidate(&self) -> bool {
        self.qualification == ReconfigurationQualification::Pass
            && self.findings.is_empty()
    }
}

fn finding(
    findings: &mut Vec<ReconfigurationFinding>,
    code: ReconfigurationFindingCode,
    subject: Option<String>,
    detail: impl Into<String>,
) {
    findings.push(ReconfigurationFinding {
        code,
        subject,
        detail: detail.into(),
    });
}

fn is_protected_component(class: ComponentClass, required_for_viability: bool) -> bool {
    required_for_viability
        || matches!(
            class,
            ComponentClass::ProvisionedImmutable | ComponentClass::Constraint
        )
}

fn input_edges(organization: &OrganizationSpec) -> BTreeSet<ProcessInputEdge> {
    let mut edges = BTreeSet::new();
    for (process_id, process) in &organization.processes {
        for component_id in &process.required_inputs {
            edges.insert(ProcessInputEdge {
                process_id: process_id.clone(),
                component_id: component_id.clone(),
                requirement: EdgeRequirement::Required,
            });
        }
        for component_id in &process.optional_inputs {
            edges.insert(ProcessInputEdge {
                process_id: process_id.clone(),
                component_id: component_id.clone(),
                requirement: EdgeRequirement::Optional,
            });
        }
    }
    edges
}

fn output_edges(organization: &OrganizationSpec) -> BTreeSet<ProcessOutputEdge> {
    let mut edges = BTreeSet::new();
    for (process_id, process) in &organization.processes {
        for component_id in &process.required_outputs {
            edges.insert(ProcessOutputEdge {
                process_id: process_id.clone(),
                component_id: component_id.clone(),
                requirement: EdgeRequirement::Required,
            });
        }
        for component_id in &process.optional_outputs {
            edges.insert(ProcessOutputEdge {
                process_id: process_id.clone(),
                component_id: component_id.clone(),
                requirement: EdgeRequirement::Optional,
            });
        }
    }
    edges
}

fn external_dependency_edges(
    organization: &OrganizationSpec,
) -> BTreeSet<ExternalDependencyEdge> {
    let mut edges = BTreeSet::new();
    for (process_id, process) in &organization.processes {
        for dependency_id in &process.required_external_dependencies {
            edges.insert(ExternalDependencyEdge {
                process_id: process_id.clone(),
                dependency_id: dependency_id.clone(),
                requirement: EdgeRequirement::Required,
            });
        }
        for dependency_id in &process.optional_external_dependencies {
            edges.insert(ExternalDependencyEdge {
                process_id: process_id.clone(),
                dependency_id: dependency_id.clone(),
                requirement: EdgeRequirement::Optional,
            });
        }
    }
    edges
}

fn difference<T: Ord + Clone>(left: &BTreeSet<T>, right: &BTreeSet<T>) -> Vec<T> {
    left.difference(right).cloned().collect()
}

/// Compare a frozen baseline with a proposed organization and fail closed on
/// every unapproved or structurally invalid mutation.
pub fn evaluate_reconfiguration(
    baseline: &OrganizationSpec,
    candidate: &OrganizationSpec,
    policy: &ReconfigurationPolicy,
) -> ReconfigurationReport {
    let mut findings = Vec::new();
    let mut diff = ReconfigurationDiff::default();

    for (key, component) in &candidate.components {
        if key != &component.id {
            finding(
                &mut findings,
                ReconfigurationFindingCode::ComponentMapKeyMismatch,
                Some(key.0.clone()),
                "candidate component map key differs from embedded component ID",
            );
        }
    }

    for (key, process) in &candidate.processes {
        if key != &process.id {
            finding(
                &mut findings,
                ReconfigurationFindingCode::ProcessMapKeyMismatch,
                Some(key.0.clone()),
                "candidate process map key differs from embedded process ID",
            );
        }
    }

    let baseline_component_ids: BTreeSet<_> = baseline.components.keys().cloned().collect();
    let candidate_component_ids: BTreeSet<_> = candidate.components.keys().cloned().collect();
    diff.added_components = difference(&candidate_component_ids, &baseline_component_ids);
    diff.removed_components = difference(&baseline_component_ids, &candidate_component_ids);

    for component_id in &diff.added_components {
        let component = &candidate.components[component_id];
        if policy.protected_components.contains(component_id)
            || matches!(
                component.class,
                ComponentClass::ProvisionedImmutable | ComponentClass::Constraint
            )
        {
            finding(
                &mut findings,
                ReconfigurationFindingCode::ProtectedComponentAdded,
                Some(component_id.0.clone()),
                "candidate cannot create a new provisioned root or external constraint",
            );
        }
        if !policy.allowed_component_additions.contains(component_id) {
            finding(
                &mut findings,
                ReconfigurationFindingCode::ComponentAdditionNotAllowed,
                Some(component_id.0.clone()),
                "component addition is not present in the frozen allowlist",
            );
        }
    }

    for component_id in &diff.removed_components {
        let component = &baseline.components[component_id];
        let protected = is_protected_component(
            component.class,
            component.required_for_viability,
        ) || policy.protected_components.contains(component_id);
        if protected {
            finding(
                &mut findings,
                ReconfigurationFindingCode::ProtectedComponentRemoved,
                Some(component_id.0.clone()),
                "provisioned, constrained or viability-critical component cannot be removed",
            );
        } else if !policy.allowed_component_removals.contains(component_id) {
            finding(
                &mut findings,
                ReconfigurationFindingCode::ComponentRemovalNotAllowed,
                Some(component_id.0.clone()),
                "component removal is not present in the frozen allowlist",
            );
        }
    }

    for component_id in baseline_component_ids.intersection(&candidate_component_ids) {
        let before = &baseline.components[component_id];
        let after = &candidate.components[component_id];
        if before != after {
            diff.changed_component_metadata.push(component_id.clone());
            let protected = is_protected_component(
                before.class,
                before.required_for_viability,
            ) || is_protected_component(after.class, after.required_for_viability)
                || policy.protected_components.contains(component_id);
            if protected {
                finding(
                    &mut findings,
                    ReconfigurationFindingCode::ProtectedComponentMutation,
                    Some(component_id.0.clone()),
                    "protected component metadata or viability classification changed",
                );
            } else {
                finding(
                    &mut findings,
                    ReconfigurationFindingCode::ComponentMetadataMutationNotAllowed,
                    Some(component_id.0.clone()),
                    "component metadata mutation is not supported by this reconfiguration protocol",
                );
            }
        }
    }

    let baseline_process_ids: BTreeSet<_> = baseline.processes.keys().cloned().collect();
    let candidate_process_ids: BTreeSet<_> = candidate.processes.keys().cloned().collect();
    diff.added_processes = difference(&candidate_process_ids, &baseline_process_ids);
    diff.removed_processes = difference(&baseline_process_ids, &candidate_process_ids);

    for process_id in &diff.added_processes {
        if !policy.allowed_process_additions.contains(process_id) {
            finding(
                &mut findings,
                ReconfigurationFindingCode::ProcessAdditionNotAllowed,
                Some(process_id.0.clone()),
                "process addition is not present in the frozen allowlist",
            );
        }
    }

    for process_id in &diff.removed_processes {
        if policy.protected_processes.contains(process_id) {
            finding(
                &mut findings,
                ReconfigurationFindingCode::ProtectedProcessRemoved,
                Some(process_id.0.clone()),
                "protected process cannot be removed",
            );
        } else if !policy.allowed_process_removals.contains(process_id) {
            finding(
                &mut findings,
                ReconfigurationFindingCode::ProcessRemovalNotAllowed,
                Some(process_id.0.clone()),
                "process removal is not present in the frozen allowlist",
            );
        }
    }

    for process_id in baseline_process_ids.intersection(&candidate_process_ids) {
        let before = &baseline.processes[process_id];
        let after = &candidate.processes[process_id];
        if before != after {
            diff.changed_process_contracts.push(process_id.clone());
            if before.version != after.version {
                diff.changed_process_versions.push(process_id.clone());
            }
            if before.executable_digest != after.executable_digest {
                diff.changed_executable_identities.push(process_id.clone());
            }
            if policy.protected_processes.contains(process_id) {
                finding(
                    &mut findings,
                    ReconfigurationFindingCode::ProtectedProcessMutation,
                    Some(process_id.0.clone()),
                    "protected process contract cannot be changed",
                );
            } else if !policy.allowed_process_contract_changes.contains(process_id) {
                finding(
                    &mut findings,
                    ReconfigurationFindingCode::ProcessContractChangeNotAllowed,
                    Some(process_id.0.clone()),
                    "process contract change is not present in the frozen allowlist",
                );
            }
            if before.executable_digest != after.executable_digest
                && !policy
                    .allowed_executable_identity_changes
                    .contains(process_id)
            {
                finding(
                    &mut findings,
                    ReconfigurationFindingCode::ExecutableIdentityChangeNotAllowed,
                    Some(process_id.0.clone()),
                    "executable identity change requires its own explicit allowlist entry",
                );
            }
        }
    }

    let baseline_inputs = input_edges(baseline);
    let candidate_inputs = input_edges(candidate);
    diff.added_inputs = difference(&candidate_inputs, &baseline_inputs);
    diff.removed_inputs = difference(&baseline_inputs, &candidate_inputs);

    // A reconfiguration may add new consumers of provisioned roots and
    // constraints, but it cannot sever a pre-existing dependency on them.
    // Otherwise a candidate could retain a policy component in the inventory
    // while silently removing the process edge that enforces it.
    for edge in &diff.removed_inputs {
        let protected = baseline.components.get(&edge.component_id).is_some_and(|component| {
            matches!(
                component.class,
                ComponentClass::ProvisionedImmutable | ComponentClass::Constraint
            ) || policy.protected_components.contains(&edge.component_id)
        });
        if protected {
            finding(
                &mut findings,
                ReconfigurationFindingCode::ProtectedInputEdgeRemoved,
                Some(format!("{}:{}", edge.process_id.0, edge.component_id.0)),
                "candidate cannot remove or weaken a pre-existing input dependency on a provisioned root or constraint",
            );
        }
    }

    let baseline_outputs = output_edges(baseline);
    let candidate_outputs = output_edges(candidate);
    diff.added_outputs = difference(&candidate_outputs, &baseline_outputs);
    diff.removed_outputs = difference(&baseline_outputs, &candidate_outputs);

    let baseline_external = external_dependency_edges(baseline);
    let candidate_external = external_dependency_edges(candidate);
    diff.added_external_dependencies = difference(&candidate_external, &baseline_external);
    diff.removed_external_dependencies = difference(&baseline_external, &candidate_external);

    for (process_id, process) in &candidate.processes {
        for component_id in process.required_inputs.union(&process.optional_inputs) {
            if !candidate.components.contains_key(component_id) {
                finding(
                    &mut findings,
                    ReconfigurationFindingCode::UnknownProcessInput,
                    Some(process_id.0.clone()),
                    format!("process input {} is not declared in the candidate", component_id.0),
                );
            }
        }
        for component_id in process.required_outputs.union(&process.optional_outputs) {
            if !candidate.components.contains_key(component_id) {
                finding(
                    &mut findings,
                    ReconfigurationFindingCode::UnknownProcessOutput,
                    Some(process_id.0.clone()),
                    format!("process output {} is not declared in the candidate", component_id.0),
                );
            } else if let Some(component) = candidate.components.get(component_id)
                && (matches!(
                    component.class,
                    ComponentClass::ProvisionedImmutable | ComponentClass::Constraint
                ) || policy.protected_components.contains(component_id))
            {
                finding(
                    &mut findings,
                    ReconfigurationFindingCode::ProtectedComponentProduced,
                    Some(format!("{}:{}", process_id.0, component_id.0)),
                    "candidate process cannot manufacture a provisioned root or externally governed constraint",
                );
            }
        }
        if !process.required_inputs.is_disjoint(&process.optional_inputs) {
            finding(
                &mut findings,
                ReconfigurationFindingCode::RequiredOptionalInputOverlap,
                Some(process_id.0.clone()),
                "a component cannot be both a required and optional input",
            );
        }
        if !process.required_outputs.is_disjoint(&process.optional_outputs) {
            finding(
                &mut findings,
                ReconfigurationFindingCode::RequiredOptionalOutputOverlap,
                Some(process_id.0.clone()),
                "a component cannot be both a required and optional output",
            );
        }
        if !process
            .required_external_dependencies
            .is_disjoint(&process.optional_external_dependencies)
        {
            finding(
                &mut findings,
                ReconfigurationFindingCode::RequiredOptionalDependencyOverlap,
                Some(process_id.0.clone()),
                "an external dependency cannot be both required and optional",
            );
        }
    }

    let edit_count = diff.structural_edit_count();
    if edit_count > policy.max_structural_edits {
        finding(
            &mut findings,
            ReconfigurationFindingCode::EditBudgetExceeded,
            None,
            format!(
                "candidate requires {edit_count} structural edits; budget is {}",
                policy.max_structural_edits
            ),
        );
    }

    let qualification = if findings.is_empty() {
        ReconfigurationQualification::Pass
    } else {
        ReconfigurationQualification::Fail
    };

    ReconfigurationReport {
        qualification,
        diff,
        findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComponentSpec, ContentDigest, ProcessSpec};

    fn process(id: &'static str) -> ProcessSpec {
        ProcessSpec::new(id, "v1", ContentDigest::of(id.as_bytes()))
    }

    fn baseline() -> OrganizationSpec {
        let mut organization = OrganizationSpec::default();
        assert!(organization.insert_component(ComponentSpec::new(
            "memory",
            ComponentClass::RegenerativeInternal,
            true,
        )));
        assert!(organization.insert_component(ComponentSpec::new(
            "config-root",
            ComponentClass::ProvisionedImmutable,
            false,
        )));
        assert!(organization.insert_component(ComponentSpec::new(
            "policy",
            ComponentClass::Constraint,
            false,
        )));
        assert!(organization.insert_component(ComponentSpec::new(
            "cache-a",
            ComponentClass::RegenerativeInternal,
            false,
        )));
        assert!(organization.insert_component(ComponentSpec::new(
            "cache-b",
            ComponentClass::RegenerativeInternal,
            false,
        )));
        let mut producer = process("refresh");
        producer.required_inputs.insert("config-root".into());
        producer.optional_inputs.insert("cache-a".into());
        producer.required_outputs.insert("memory".into());
        assert!(organization.insert_process(producer));
        organization
    }

    #[test]
    fn unchanged_organization_passes_with_zero_edit_budget() {
        let organization = baseline();
        let report = evaluate_reconfiguration(
            &organization,
            &organization,
            &ReconfigurationPolicy::default(),
        );
        assert!(report.qualifies_candidate());
        assert!(report.diff.is_empty());
        assert_eq!(report.diff.structural_edit_count(), 0);
    }

    #[test]
    fn explicitly_allowed_alternate_producer_and_edges_pass() {
        let before = baseline();
        let mut after = before.clone();
        let mut repair = process("repair");
        repair.required_inputs.insert("config-root".into());
        repair.required_outputs.insert("memory".into());
        assert!(after.insert_process(repair));

        let mut policy = ReconfigurationPolicy {
            max_structural_edits: 3,
            ..ReconfigurationPolicy::default()
        };
        policy.allowed_process_additions.insert("repair".into());

        let report = evaluate_reconfiguration(&before, &after, &policy);
        assert!(report.qualifies_candidate(), "{:?}", report.findings);
        assert_eq!(report.diff.added_processes, vec![ProcessId::from("repair")]);
        assert_eq!(report.diff.added_inputs.len(), 1);
        assert_eq!(report.diff.added_outputs.len(), 1);
    }

    #[test]
    fn process_addition_without_allowlist_fails() {
        let before = baseline();
        let mut after = before.clone();
        assert!(after.insert_process(process("unapproved")));
        let report = evaluate_reconfiguration(&before, &after, &ReconfigurationPolicy {
            max_structural_edits: 10,
            ..ReconfigurationPolicy::default()
        });
        assert!(!report.qualifies_candidate());
        assert!(report.findings.iter().any(|item| {
            item.code == ReconfigurationFindingCode::ProcessAdditionNotAllowed
        }));
    }

    #[test]
    fn viability_critical_component_cannot_be_removed_even_if_allowlisted() {
        let before = baseline();
        let mut after = before.clone();
        after.components.remove(&ComponentId::from("memory"));
        let mut policy = ReconfigurationPolicy {
            max_structural_edits: 10,
            ..ReconfigurationPolicy::default()
        };
        policy.allowed_component_removals.insert("memory".into());
        let report = evaluate_reconfiguration(&before, &after, &policy);
        assert!(!report.qualifies_candidate());
        assert!(report.findings.iter().any(|item| {
            item.code == ReconfigurationFindingCode::ProtectedComponentRemoved
        }));
    }

    #[test]
    fn provisioned_roots_and_constraints_cannot_be_removed() {
        let before = baseline();
        let mut after = before.clone();
        after.components.remove(&ComponentId::from("config-root"));
        after.components.remove(&ComponentId::from("policy"));
        let mut policy = ReconfigurationPolicy {
            max_structural_edits: 10,
            ..ReconfigurationPolicy::default()
        };
        policy.allowed_component_removals.insert("config-root".into());
        policy.allowed_component_removals.insert("policy".into());
        let report = evaluate_reconfiguration(&before, &after, &policy);
        assert!(!report.qualifies_candidate());
        assert_eq!(
            report.findings.iter().filter(|item| {
                item.code == ReconfigurationFindingCode::ProtectedComponentRemoved
            }).count(),
            2
        );
    }

    #[test]
    fn candidate_cannot_create_a_new_trust_root_or_constraint() {
        let before = baseline();
        let mut after = before.clone();
        assert!(after.insert_component(ComponentSpec::new(
            "new-policy",
            ComponentClass::Constraint,
            false,
        )));
        let mut policy = ReconfigurationPolicy {
            max_structural_edits: 10,
            ..ReconfigurationPolicy::default()
        };
        policy.allowed_component_additions.insert("new-policy".into());
        let report = evaluate_reconfiguration(&before, &after, &policy);
        assert!(!report.qualifies_candidate());
        assert!(report.findings.iter().any(|item| {
            item.code == ReconfigurationFindingCode::ProtectedComponentAdded
        }));
    }

    #[test]
    fn component_reclassification_is_rejected() {
        let before = baseline();
        let mut after = before.clone();
        after.components.insert(
            "memory".into(),
            ComponentSpec::new("memory", ComponentClass::EvidenceArtifact, false),
        );
        let report = evaluate_reconfiguration(&before, &after, &ReconfigurationPolicy {
            max_structural_edits: 10,
            ..ReconfigurationPolicy::default()
        });
        assert!(!report.qualifies_candidate());
        assert!(report.findings.iter().any(|item| {
            item.code == ReconfigurationFindingCode::ProtectedComponentMutation
        }));
    }

    #[test]
    fn rewiring_requires_a_process_contract_allowlist() {
        let before = baseline();
        let mut after = before.clone();
        after.processes.get_mut(&ProcessId::from("refresh")).unwrap()
            .optional_inputs.insert("memory".into());
        let report = evaluate_reconfiguration(&before, &after, &ReconfigurationPolicy {
            max_structural_edits: 10,
            ..ReconfigurationPolicy::default()
        });
        assert!(!report.qualifies_candidate());
        assert!(report.findings.iter().any(|item| {
            item.code == ReconfigurationFindingCode::ProcessContractChangeNotAllowed
        }));
    }

    #[test]
    fn allowed_rewire_is_measured_as_edge_changes() {
        let before = baseline();
        let mut after = before.clone();
        let producer = after.processes.get_mut(&ProcessId::from("refresh")).unwrap();
        producer.optional_inputs.remove(&ComponentId::from("cache-a"));
        producer.optional_inputs.insert("cache-b".into());
        let mut policy = ReconfigurationPolicy {
            max_structural_edits: 3,
            ..ReconfigurationPolicy::default()
        };
        policy.allowed_process_contract_changes.insert("refresh".into());
        let report = evaluate_reconfiguration(&before, &after, &policy);
        assert!(report.qualifies_candidate(), "{:?}", report.findings);
        assert_eq!(report.diff.removed_inputs.len(), 1);
        assert_eq!(report.diff.added_inputs.len(), 1);
    }

    #[test]
    fn constraint_input_cannot_be_severed_even_with_contract_allowlist() {
        let before = baseline();
        let mut after = before.clone();
        after.processes.get_mut(&ProcessId::from("refresh")).unwrap()
            .required_inputs.insert("policy".into());
        // The baseline did not previously bind this process to policy; first
        // establish the binding in both snapshots, then try to remove it.
        let mut before = before;
        before.processes.get_mut(&ProcessId::from("refresh")).unwrap()
            .required_inputs.insert("policy".into());
        after.processes.get_mut(&ProcessId::from("refresh")).unwrap()
            .required_inputs.remove(&ComponentId::from("policy"));
        let mut policy = ReconfigurationPolicy {
            max_structural_edits: 10,
            ..ReconfigurationPolicy::default()
        };
        policy.allowed_process_contract_changes.insert("refresh".into());
        let report = evaluate_reconfiguration(&before, &after, &policy);
        assert!(!report.qualifies_candidate());
        assert!(report.findings.iter().any(|item| {
            item.code == ReconfigurationFindingCode::ProtectedInputEdgeRemoved
        }));
    }

    #[test]
    fn candidate_process_cannot_produce_a_provisioned_root_or_constraint() {
        let before = baseline();
        let mut after = before.clone();
        let producer = after.processes.get_mut(&ProcessId::from("refresh")).unwrap();
        producer.optional_outputs.insert("config-root".into());
        producer.optional_outputs.insert("policy".into());
        let mut policy = ReconfigurationPolicy {
            max_structural_edits: 10,
            ..ReconfigurationPolicy::default()
        };
        policy.allowed_process_contract_changes.insert("refresh".into());
        let report = evaluate_reconfiguration(&before, &after, &policy);
        assert!(!report.qualifies_candidate());
        assert_eq!(
            report.findings.iter().filter(|item| {
                item.code == ReconfigurationFindingCode::ProtectedComponentProduced
            }).count(),
            2
        );
    }

    #[test]
    fn executable_identity_change_needs_a_separate_allowlist() {
        let before = baseline();
        let mut after = before.clone();
        after.processes.get_mut(&ProcessId::from("refresh")).unwrap().executable_digest =
            ContentDigest::of(b"new-executable");
        let mut policy = ReconfigurationPolicy {
            max_structural_edits: 10,
            ..ReconfigurationPolicy::default()
        };
        policy.allowed_process_contract_changes.insert("refresh".into());
        let report = evaluate_reconfiguration(&before, &after, &policy);
        assert!(!report.qualifies_candidate());
        assert!(report.findings.iter().any(|item| {
            item.code == ReconfigurationFindingCode::ExecutableIdentityChangeNotAllowed
        }));
    }

    #[test]
    fn candidate_cannot_reference_undeclared_components() {
        let before = baseline();
        let mut after = before.clone();
        after.processes.get_mut(&ProcessId::from("refresh")).unwrap()
            .required_outputs.insert("ghost".into());
        let mut policy = ReconfigurationPolicy {
            max_structural_edits: 10,
            ..ReconfigurationPolicy::default()
        };
        policy.allowed_process_contract_changes.insert("refresh".into());
        let report = evaluate_reconfiguration(&before, &after, &policy);
        assert!(!report.qualifies_candidate());
        assert!(report.findings.iter().any(|item| {
            item.code == ReconfigurationFindingCode::UnknownProcessOutput
        }));
    }

    #[test]
    fn overlapping_required_and_optional_relationships_fail() {
        let before = baseline();
        let mut after = before.clone();
        after.processes.get_mut(&ProcessId::from("refresh")).unwrap()
            .required_inputs.insert("memory".into());
        after.processes.get_mut(&ProcessId::from("refresh")).unwrap()
            .optional_inputs.insert("memory".into());
        let mut policy = ReconfigurationPolicy {
            max_structural_edits: 10,
            ..ReconfigurationPolicy::default()
        };
        policy.allowed_process_contract_changes.insert("refresh".into());
        let report = evaluate_reconfiguration(&before, &after, &policy);
        assert!(!report.qualifies_candidate());
        assert!(report.findings.iter().any(|item| {
            item.code == ReconfigurationFindingCode::RequiredOptionalInputOverlap
        }));
    }

    #[test]
    fn edit_budget_exhaustion_rejects_candidate() {
        let before = baseline();
        let mut after = before.clone();
        assert!(after.insert_process(process("repair")));
        let mut policy = ReconfigurationPolicy {
            max_structural_edits: 0,
            ..ReconfigurationPolicy::default()
        };
        policy.allowed_process_additions.insert("repair".into());
        let report = evaluate_reconfiguration(&before, &after, &policy);
        assert!(!report.qualifies_candidate());
        assert!(report.findings.iter().any(|item| {
            item.code == ReconfigurationFindingCode::EditBudgetExceeded
        }));
    }

    #[test]
    fn protected_process_cannot_be_removed_or_modified() {
        let before = baseline();
        let mut after = before.clone();
        after.processes.remove(&ProcessId::from("refresh"));
        let mut policy = ReconfigurationPolicy {
            max_structural_edits: 10,
            ..ReconfigurationPolicy::default()
        };
        policy.protected_processes.insert("refresh".into());
        policy.allowed_process_removals.insert("refresh".into());
        let report = evaluate_reconfiguration(&before, &after, &policy);
        assert!(!report.qualifies_candidate());
        assert!(report.findings.iter().any(|item| {
            item.code == ReconfigurationFindingCode::ProtectedProcessRemoved
        }));
    }
}
