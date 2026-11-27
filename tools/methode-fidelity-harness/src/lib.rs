//! A small, dependency-free conformance auditor for reference-anchored scientific methods.
//!
//! This crate does not execute scientific workloads or decide whether a scientific
//! conclusion is true. It checks whether an execution receipt faithfully matches
//! an independently reviewed method contract, and keeps method fidelity separate
//! from scientific-result validation.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FitScope {
    TrainingOnly,
    TrainingAndValidation,
    AllRecords,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayOutcome {
    Passed,
    Failed,
    NotRun,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScientificResult {
    Supported,
    Unsupported,
    Inconclusive,
    NotEvaluated,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Disposition {
    NotExecuted,
    ExecutedWithDeclaredDeviation,
    ReferenceProtocolReproduced,
    Unresolved,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckStatus {
    Pass,
    DeclaredDeviation,
    Mismatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckResult {
    pub check_id: String,
    pub status: CheckStatus,
    pub explanation: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolReview {
    /// Set only by an independently maintained protocol-review process.
    pub approved: bool,
    pub reviewer_id: String,
    pub review_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetContract {
    pub dataset_id: String,
    pub version: String,
    pub artifact_sha256: String,
    pub split_manifest_sha256: String,
    pub train_records: u64,
    pub validation_records: u64,
    pub test_records: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreprocessingContract {
    pub pipeline_sha256: String,
    pub fit_scope: FitScope,
    pub fitted_on_split_manifest_sha256: String,
    pub fitted_record_count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentContract {
    pub role: String,
    pub implementation_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaselineContract {
    pub baseline_id: String,
    pub configuration_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricContract {
    pub metric_id: String,
    pub definition_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BudgetContract {
    pub policy_sha256: String,
    pub required_epochs: u64,
    pub max_training_records: u64,
    pub max_wall_time_ms: u64,
    pub max_peak_memory_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MethodContract {
    pub contract_id: String,
    pub contract_sha256: String,
    pub reference_claim_sha256: String,
    pub review: ProtocolReview,
    pub dataset: DatasetContract,
    pub preprocessing: PreprocessingContract,
    pub required_components: Vec<ComponentContract>,
    pub required_baselines: Vec<BaselineContract>,
    pub required_metrics: Vec<MetricContract>,
    pub budget: BudgetContract,
    pub required_outputs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentUse {
    pub role: String,
    pub implementation_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaselineUse {
    pub baseline_id: String,
    pub configuration_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricUse {
    pub metric_id: String,
    pub definition_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputArtifact {
    pub artifact_id: String,
    pub artifact_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayReceipt {
    pub outcome: ReplayOutcome,
    pub verifier_id: String,
    pub verifier_sha256: String,
    pub replayed_contract_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeclaredDeviation {
    /// Must match a check ID emitted by this auditor, e.g. "dataset.counts".
    pub check_id: String,
    pub rationale: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionReceipt {
    pub execution_id: String,
    pub method_contract_sha256: String,
    pub execution_started: bool,
    pub execution_finished: bool,
    /// Retained for audit only. This field never affects qualification.
    pub producer_asserted_success: bool,

    pub dataset_id: String,
    pub dataset_version: String,
    pub dataset_artifact_sha256: String,
    pub split_manifest_sha256: String,
    pub train_records: u64,
    pub validation_records: u64,
    pub test_records: u64,

    pub preprocessing_pipeline_sha256: String,
    pub preprocessing_fit_scope: FitScope,
    pub preprocessing_fit_split_manifest_sha256: String,
    pub preprocessing_fitted_record_count: u64,

    pub components: Vec<ComponentUse>,
    pub baselines: Vec<BaselineUse>,
    pub metrics: Vec<MetricUse>,

    pub budget_policy_sha256: String,
    pub epochs_executed: u64,
    pub training_records_used: u64,
    pub wall_time_ms: u64,
    pub peak_memory_bytes: u64,

    pub outputs: Vec<OutputArtifact>,
    pub replay: ReplayReceipt,
    pub scientific_result: ScientificResult,
    pub declared_deviations: Vec<DeclaredDeviation>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClaimPermissions {
    pub may_claim_reference_protocol_reproduced: bool,
    pub may_claim_scientific_result_supported: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditReport {
    pub disposition: Disposition,
    pub scientific_result: ScientificResult,
    pub permissions: ClaimPermissions,
    pub checks: Vec<CheckResult>,
}

#[derive(Default)]
struct AuditFlags {
    hard_failure: bool,
    undeclared_mismatch: bool,
    declared_mismatch: bool,
    invalid_deviation_declaration: bool,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn check(
    checks: &mut Vec<CheckResult>,
    flags: &mut AuditFlags,
    check_id: &str,
    passed: bool,
    hard_gate: bool,
    deviations: &BTreeMap<String, String>,
    explanation: &str,
) {
    if passed {
        checks.push(CheckResult {
            check_id: check_id.to_string(),
            status: CheckStatus::Pass,
            explanation: explanation.to_string(),
        });
        return;
    }

    let declared = deviations.get(check_id).is_some_and(|rationale| !rationale.trim().is_empty());
    if declared && !hard_gate {
        flags.declared_mismatch = true;
        checks.push(CheckResult {
            check_id: check_id.to_string(),
            status: CheckStatus::DeclaredDeviation,
            explanation: deviations[check_id].clone(),
        });
    } else {
        if hard_gate {
            flags.hard_failure = true;
        } else {
            flags.undeclared_mismatch = true;
        }
        checks.push(CheckResult {
            check_id: check_id.to_string(),
            status: CheckStatus::Mismatch,
            explanation: explanation.to_string(),
        });
    }
}

fn unique_deviations(receipt: &ExecutionReceipt) -> (BTreeMap<String, String>, bool) {
    let mut declarations = BTreeMap::new();
    let mut duplicate = false;
    for deviation in &receipt.declared_deviations {
        if declarations.insert(deviation.check_id.clone(), deviation.rationale.clone()).is_some() {
            duplicate = true;
        }
    }
    (declarations, duplicate)
}

fn valid_contract(contract: &MethodContract) -> bool {
    !contract.contract_id.trim().is_empty()
        && is_sha256(&contract.contract_sha256)
        && is_sha256(&contract.reference_claim_sha256)
        && contract.review.approved
        && !contract.review.reviewer_id.trim().is_empty()
        && is_sha256(&contract.review.review_sha256)
        && !contract.dataset.dataset_id.trim().is_empty()
        && !contract.dataset.version.trim().is_empty()
        && is_sha256(&contract.dataset.artifact_sha256)
        && is_sha256(&contract.dataset.split_manifest_sha256)
        && is_sha256(&contract.preprocessing.pipeline_sha256)
        && is_sha256(&contract.preprocessing.fitted_on_split_manifest_sha256)
        && !contract.required_components.is_empty()
        && contract.required_components.iter().all(|c| !c.role.trim().is_empty() && is_sha256(&c.implementation_sha256))
        && contract.required_baselines.iter().all(|b| !b.baseline_id.trim().is_empty() && is_sha256(&b.configuration_sha256))
        && contract.required_metrics.iter().all(|m| !m.metric_id.trim().is_empty() && is_sha256(&m.definition_sha256))
        && is_sha256(&contract.budget.policy_sha256)
        && !contract.required_outputs.is_empty()
        && contract.required_outputs.iter().all(|id| !id.trim().is_empty())
        && contract.preprocessing.fit_scope == FitScope::TrainingOnly
        && contract.preprocessing.fitted_on_split_manifest_sha256 == contract.dataset.split_manifest_sha256
        && contract.preprocessing.fitted_record_count == contract.dataset.train_records
        && contract.dataset.train_records > 0
        && contract.dataset.test_records > 0
        && contract.budget.required_epochs > 0
        && contract.budget.max_training_records >= contract.dataset.train_records
        && contract.budget.max_wall_time_ms > 0
        && contract.budget.max_peak_memory_bytes > 0
}

fn contains_required_components(contract: &MethodContract, receipt: &ExecutionReceipt) -> bool {
    contract.required_components.iter().all(|required| {
        receipt.components.iter().any(|actual| {
            actual.role == required.role && actual.implementation_sha256 == required.implementation_sha256
        })
    })
}

fn contains_required_baselines(contract: &MethodContract, receipt: &ExecutionReceipt) -> bool {
    contract.required_baselines.iter().all(|required| {
        receipt.baselines.iter().any(|actual| {
            actual.baseline_id == required.baseline_id
                && actual.configuration_sha256 == required.configuration_sha256
        })
    })
}

fn contains_required_metrics(contract: &MethodContract, receipt: &ExecutionReceipt) -> bool {
    contract.required_metrics.iter().all(|required| {
        receipt.metrics.iter().any(|actual| {
            actual.metric_id == required.metric_id
                && actual.definition_sha256 == required.definition_sha256
        })
    })
}

fn contains_required_outputs(contract: &MethodContract, receipt: &ExecutionReceipt) -> bool {
    contract.required_outputs.iter().all(|required_id| {
        receipt.outputs.iter().any(|actual| {
            actual.artifact_id == *required_id && is_sha256(&actual.artifact_sha256)
        })
    })
}

/// Audits a receipt against an independently reviewed method contract.
///
/// Declared non-critical deviations are reported as exploratory execution, never
/// as faithful reproduction. Missing artifacts, incomplete execution, resource
/// overruns and failed/unavailable independent replay remain unresolved.
pub fn audit_method(contract: &MethodContract, receipt: &ExecutionReceipt) -> AuditReport {
    let scientific_result = receipt.scientific_result.clone();

    if !receipt.execution_started {
        return AuditReport {
            disposition: Disposition::NotExecuted,
            scientific_result: ScientificResult::NotEvaluated,
            permissions: ClaimPermissions {
                may_claim_reference_protocol_reproduced: false,
                may_claim_scientific_result_supported: false,
            },
            checks: vec![CheckResult {
                check_id: "execution.started".to_string(),
                status: CheckStatus::Pass,
                explanation: "Execution did not start; no result is admitted.".to_string(),
            }],
        };
    }

    let (deviations, duplicate_declarations) = unique_deviations(receipt);
    let mut checks = Vec::new();
    let mut flags = AuditFlags::default();

    check(
        &mut checks,
        &mut flags,
        "contract.valid",
        valid_contract(contract),
        true,
        &deviations,
        "The reference contract is incomplete, malformed, or lacks independent protocol approval.",
    );
    check(
        &mut checks,
        &mut flags,
        "contract.binding",
        receipt.method_contract_sha256 == contract.contract_sha256 && is_sha256(&receipt.method_contract_sha256),
        true,
        &deviations,
        "Execution receipt is not bound to the exact reviewed method contract.",
    );
    check(
        &mut checks,
        &mut flags,
        "deviations.unique",
        !duplicate_declarations,
        true,
        &deviations,
        "Duplicate deviation declarations make the execution record ambiguous.",
    );

    check(
        &mut checks, &mut flags, "dataset.identity",
        receipt.dataset_id == contract.dataset.dataset_id
            && receipt.dataset_version == contract.dataset.version
            && receipt.dataset_artifact_sha256 == contract.dataset.artifact_sha256
            && is_sha256(&receipt.dataset_artifact_sha256),
        false, &deviations, "Dataset identity, version, or raw artifact digest differs from the reference.",
    );
    check(
        &mut checks, &mut flags, "dataset.split",
        receipt.split_manifest_sha256 == contract.dataset.split_manifest_sha256
            && is_sha256(&receipt.split_manifest_sha256),
        false, &deviations, "Dataset split manifest differs from the reference.",
    );
    check(
        &mut checks, &mut flags, "dataset.counts",
        receipt.train_records == contract.dataset.train_records
            && receipt.validation_records == contract.dataset.validation_records
            && receipt.test_records == contract.dataset.test_records,
        false, &deviations, "Observed dataset partition counts differ from the reference.",
    );
    check(
        &mut checks, &mut flags, "preprocessing.pipeline",
        receipt.preprocessing_pipeline_sha256 == contract.preprocessing.pipeline_sha256
            && is_sha256(&receipt.preprocessing_pipeline_sha256),
        false, &deviations, "Preprocessing graph/configuration digest differs from the reference.",
    );
    check(
        &mut checks, &mut flags, "preprocessing.fit",
        receipt.preprocessing_fit_scope == contract.preprocessing.fit_scope
            && receipt.preprocessing_fit_split_manifest_sha256 == contract.preprocessing.fitted_on_split_manifest_sha256
            && receipt.preprocessing_fitted_record_count == contract.preprocessing.fitted_record_count,
        false, &deviations, "Preprocessing fit scope or fitted-record lineage differs from the reference.",
    );
    check(
        &mut checks, &mut flags, "method.components",
        contains_required_components(contract, receipt),
        false, &deviations, "One or more mandatory method components are missing or substituted.",
    );
    check(
        &mut checks, &mut flags, "method.baselines",
        contains_required_baselines(contract, receipt),
        false, &deviations, "One or more mandatory baselines are missing or changed.",
    );
    check(
        &mut checks, &mut flags, "method.metrics",
        contains_required_metrics(contract, receipt),
        false, &deviations, "One or more required metric definitions are missing or changed.",
    );
    check(
        &mut checks, &mut flags, "budget.policy",
        receipt.budget_policy_sha256 == contract.budget.policy_sha256
            && is_sha256(&receipt.budget_policy_sha256),
        false, &deviations, "Execution used a different resource/budget policy.",
    );
    check(
        &mut checks, &mut flags, "budget.epochs",
        receipt.epochs_executed == contract.budget.required_epochs,
        false, &deviations, "Executed epoch/iteration count differs from the declared method.",
    );
    check(
        &mut checks, &mut flags, "budget.training_records",
        receipt.training_records_used == contract.dataset.train_records
            && receipt.training_records_used <= contract.budget.max_training_records,
        false, &deviations, "Training sample count differs from the reference or exceeds its cap.",
    );
    check(
        &mut checks, &mut flags, "budget.resources",
        receipt.wall_time_ms <= contract.budget.max_wall_time_ms
            && receipt.peak_memory_bytes <= contract.budget.max_peak_memory_bytes,
        true, &deviations, "Actual resource use exceeded the independently reviewed budget cap.",
    );
    check(
        &mut checks, &mut flags, "outputs.complete",
        contains_required_outputs(contract, receipt),
        true, &deviations, "One or more required output artifacts are missing or have invalid digests.",
    );
    check(
        &mut checks, &mut flags, "execution.complete",
        receipt.execution_finished,
        true, &deviations, "Execution did not complete.",
    );
    check(
        &mut checks, &mut flags, "replay.independent",
        receipt.replay.outcome == ReplayOutcome::Passed
            && !receipt.replay.verifier_id.trim().is_empty()
            && is_sha256(&receipt.replay.verifier_sha256)
            && receipt.replay.replayed_contract_sha256 == contract.contract_sha256,
        true, &deviations, "Independent replay is missing, failed, or not bound to the reviewed method contract.",
    );

    let used_deviation_ids: BTreeSet<String> = checks
        .iter()
        .filter(|result| result.status == CheckStatus::DeclaredDeviation)
        .map(|result| result.check_id.clone())
        .collect();
    let orphan_declaration = deviations.keys().any(|id| !used_deviation_ids.contains(id));
    if orphan_declaration {
        flags.invalid_deviation_declaration = true;
        checks.push(CheckResult {
            check_id: "deviations.no_orphans".to_string(),
            status: CheckStatus::Mismatch,
            explanation: "A declared deviation did not correspond to an observed contract mismatch.".to_string(),
        });
    }

    let disposition = if flags.hard_failure
        || flags.undeclared_mismatch
        || flags.invalid_deviation_declaration
    {
        Disposition::Unresolved
    } else if flags.declared_mismatch {
        Disposition::ExecutedWithDeclaredDeviation
    } else {
        Disposition::ReferenceProtocolReproduced
    };

    let reproduced = disposition == Disposition::ReferenceProtocolReproduced;
    AuditReport {
        disposition: disposition.clone(),
        scientific_result: scientific_result.clone(),
        permissions: ClaimPermissions {
            may_claim_reference_protocol_reproduced: reproduced,
            may_claim_scientific_result_supported: reproduced && scientific_result == ScientificResult::Supported,
        },
        checks,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(ch: char) -> String {
        std::iter::repeat(ch).take(64).collect()
    }

    fn contract() -> MethodContract {
        let split = h('b');
        MethodContract {
            contract_id: "study-2026-001".into(),
            contract_sha256: h('a'),
            reference_claim_sha256: h('c'),
            review: ProtocolReview {
                approved: true,
                reviewer_id: "independent-protocol-reviewer".into(),
                review_sha256: h('d'),
            },
            dataset: DatasetContract {
                dataset_id: "dataset-chemistry-x".into(),
                version: "v3.2".into(),
                artifact_sha256: h('e'),
                split_manifest_sha256: split.clone(),
                train_records: 1_000,
                validation_records: 200,
                test_records: 200,
            },
            preprocessing: PreprocessingContract {
                pipeline_sha256: h('f'),
                fit_scope: FitScope::TrainingOnly,
                fitted_on_split_manifest_sha256: split,
                fitted_record_count: 1_000,
            },
            required_components: vec![
                ComponentContract { role: "model".into(), implementation_sha256: h('1') },
                ComponentContract { role: "optimizer".into(), implementation_sha256: h('2') },
            ],
            required_baselines: vec![
                BaselineContract { baseline_id: "linear-reference".into(), configuration_sha256: h('3') },
            ],
            required_metrics: vec![
                MetricContract { metric_id: "heldout-mae".into(), definition_sha256: h('4') },
            ],
            budget: BudgetContract {
                policy_sha256: h('5'),
                required_epochs: 50,
                max_training_records: 1_000,
                max_wall_time_ms: 600_000,
                max_peak_memory_bytes: 8_000_000_000,
            },
            required_outputs: vec!["predictions.csv".into(), "metrics.json".into()],
        }
    }

    fn receipt(contract: &MethodContract) -> ExecutionReceipt {
        ExecutionReceipt {
            execution_id: "run-001".into(),
            method_contract_sha256: contract.contract_sha256.clone(),
            execution_started: true,
            execution_finished: true,
            producer_asserted_success: true,
            dataset_id: contract.dataset.dataset_id.clone(),
            dataset_version: contract.dataset.version.clone(),
            dataset_artifact_sha256: contract.dataset.artifact_sha256.clone(),
            split_manifest_sha256: contract.dataset.split_manifest_sha256.clone(),
            train_records: 1_000,
            validation_records: 200,
            test_records: 200,
            preprocessing_pipeline_sha256: contract.preprocessing.pipeline_sha256.clone(),
            preprocessing_fit_scope: FitScope::TrainingOnly,
            preprocessing_fit_split_manifest_sha256: contract.preprocessing.fitted_on_split_manifest_sha256.clone(),
            preprocessing_fitted_record_count: 1_000,
            components: vec![
                ComponentUse { role: "model".into(), implementation_sha256: h('1') },
                ComponentUse { role: "optimizer".into(), implementation_sha256: h('2') },
            ],
            baselines: vec![
                BaselineUse { baseline_id: "linear-reference".into(), configuration_sha256: h('3') },
            ],
            metrics: vec![
                MetricUse { metric_id: "heldout-mae".into(), definition_sha256: h('4') },
            ],
            budget_policy_sha256: contract.budget.policy_sha256.clone(),
            epochs_executed: 50,
            training_records_used: 1_000,
            wall_time_ms: 420_000,
            peak_memory_bytes: 4_000_000_000,
            outputs: vec![
                OutputArtifact { artifact_id: "predictions.csv".into(), artifact_sha256: h('6') },
                OutputArtifact { artifact_id: "metrics.json".into(), artifact_sha256: h('7') },
            ],
            replay: ReplayReceipt {
                outcome: ReplayOutcome::Passed,
                verifier_id: "independent-replay-v1".into(),
                verifier_sha256: h('8'),
                replayed_contract_sha256: contract.contract_sha256.clone(),
            },
            scientific_result: ScientificResult::Supported,
            declared_deviations: Vec::new(),
        }
    }

    fn disposition(contract: &MethodContract, receipt: &ExecutionReceipt) -> Disposition {
        audit_method(contract, receipt).disposition
    }

    #[test]
    fn exact_method_with_independent_replay_is_reproduced() {
        let c = contract();
        let r = receipt(&c);
        let report = audit_method(&c, &r);
        assert_eq!(report.disposition, Disposition::ReferenceProtocolReproduced);
        assert!(report.permissions.may_claim_reference_protocol_reproduced);
        assert!(report.permissions.may_claim_scientific_result_supported);
    }

    #[test]
    fn producer_success_flag_is_not_a_verification() {
        let c = contract();
        let mut r = receipt(&c);
        r.producer_asserted_success = true;
        r.replay.outcome = ReplayOutcome::NotRun;
        let report = audit_method(&c, &r);
        assert_eq!(report.disposition, Disposition::Unresolved);
        assert!(!report.permissions.may_claim_reference_protocol_reproduced);
    }

    #[test]
    fn declared_dataset_downscale_is_an_exploratory_deviation() {
        let c = contract();
        let mut r = receipt(&c);
        r.train_records = 500;
        r.training_records_used = 500;
        r.preprocessing_fitted_record_count = 500;
        r.declared_deviations.push(DeclaredDeviation {
            check_id: "dataset.counts".into(),
            rationale: "Exploratory half-size sensitivity run; not a reproduction.".into(),
        });
        r.declared_deviations.push(DeclaredDeviation {
            check_id: "preprocessing.fit".into(),
            rationale: "Preprocessing refit only on the declared half-size training subset.".into(),
        });
        r.declared_deviations.push(DeclaredDeviation {
            check_id: "budget.training_records".into(),
            rationale: "Training rows were deliberately reduced for this exploratory variant.".into(),
        });
        assert_eq!(disposition(&c, &r), Disposition::ExecutedWithDeclaredDeviation);
        assert!(!audit_method(&c, &r).permissions.may_claim_reference_protocol_reproduced);
    }

    #[test]
    fn silent_dataset_downscale_is_unresolved() {
        let c = contract();
        let mut r = receipt(&c);
        r.train_records = 500;
        r.training_records_used = 500;
        r.preprocessing_fitted_record_count = 500;
        assert_eq!(disposition(&c, &r), Disposition::Unresolved);
    }

    #[test]
    fn split_manifest_change_is_detected() {
        let c = contract();
        let mut r = receipt(&c);
        r.split_manifest_sha256 = h('9');
        assert_eq!(disposition(&c, &r), Disposition::Unresolved);
    }

    #[test]
    fn preprocessing_fit_on_all_records_is_not_faithful() {
        let c = contract();
        let mut r = receipt(&c);
        r.preprocessing_fit_scope = FitScope::AllRecords;
        r.preprocessing_fitted_record_count = 1_400;
        assert_eq!(disposition(&c, &r), Disposition::Unresolved);
    }

    #[test]
    fn component_substitution_requires_explicit_deviation_and_stays_non_reproduction() {
        let c = contract();
        let mut r = receipt(&c);
        r.components[0].implementation_sha256 = h('a');
        r.declared_deviations.push(DeclaredDeviation {
            check_id: "method.components".into(),
            rationale: "Alternative model implementation used for a separately labelled ablation.".into(),
        });
        assert_eq!(disposition(&c, &r), Disposition::ExecutedWithDeclaredDeviation);
    }

    #[test]
    fn missing_baseline_is_detected() {
        let c = contract();
        let mut r = receipt(&c);
        r.baselines.clear();
        assert_eq!(disposition(&c, &r), Disposition::Unresolved);
    }

    #[test]
    fn changed_metric_definition_is_detected() {
        let c = contract();
        let mut r = receipt(&c);
        r.metrics[0].definition_sha256 = h('a');
        assert_eq!(disposition(&c, &r), Disposition::Unresolved);
    }

    #[test]
    fn over_budget_execution_is_unresolved_even_when_declared() {
        let c = contract();
        let mut r = receipt(&c);
        r.wall_time_ms = c.budget.max_wall_time_ms + 1;
        r.declared_deviations.push(DeclaredDeviation {
            check_id: "budget.resources".into(),
            rationale: "The run exceeded the declared compute envelope.".into(),
        });
        assert_eq!(disposition(&c, &r), Disposition::Unresolved);
    }

    #[test]
    fn missing_required_output_is_unresolved() {
        let c = contract();
        let mut r = receipt(&c);
        r.outputs.pop();
        assert_eq!(disposition(&c, &r), Disposition::Unresolved);
    }

    #[test]
    fn failed_independent_replay_is_unresolved() {
        let c = contract();
        let mut r = receipt(&c);
        r.replay.outcome = ReplayOutcome::Failed;
        assert_eq!(disposition(&c, &r), Disposition::Unresolved);
    }

    #[test]
    fn execution_without_start_is_not_executed() {
        let c = contract();
        let mut r = receipt(&c);
        r.execution_started = false;
        r.scientific_result = ScientificResult::Supported;
        let report = audit_method(&c, &r);
        assert_eq!(report.disposition, Disposition::NotExecuted);
        assert_eq!(report.scientific_result, ScientificResult::NotEvaluated);
        assert!(!report.permissions.may_claim_scientific_result_supported);
    }

    #[test]
    fn faithful_method_does_not_guarantee_scientific_support() {
        let c = contract();
        let mut r = receipt(&c);
        r.scientific_result = ScientificResult::Unsupported;
        let report = audit_method(&c, &r);
        assert_eq!(report.disposition, Disposition::ReferenceProtocolReproduced);
        assert!(report.permissions.may_claim_reference_protocol_reproduced);
        assert!(!report.permissions.may_claim_scientific_result_supported);
    }

    #[test]
    fn declared_deviation_without_observed_mismatch_is_rejected() {
        let c = contract();
        let mut r = receipt(&c);
        r.declared_deviations.push(DeclaredDeviation {
            check_id: "dataset.counts".into(),
            rationale: "Claimed reduction, but observed counts match the reference.".into(),
        });
        assert_eq!(disposition(&c, &r), Disposition::Unresolved);
    }

    #[test]
    fn contract_without_independent_review_is_not_admissible() {
        let mut c = contract();
        c.review.approved = false;
        let r = receipt(&c);
        assert_eq!(disposition(&c, &r), Disposition::Unresolved);
    }

    #[test]
    fn invalid_digest_is_rejected() {
        let c = contract();
        let mut r = receipt(&c);
        r.outputs[0].artifact_sha256 = "not-a-digest".into();
        assert_eq!(disposition(&c, &r), Disposition::Unresolved);
    }
}
