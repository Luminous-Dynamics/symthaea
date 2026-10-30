//! Semantic verifier for query-scoped connectome reliability certificates.
//!
//! This validates certificate structure and cross-field invariants. It does not
//! fetch source artifacts, prove SHA-256 claims correspond to external bytes, or
//! establish biological truth. A passing report only means the declared contract
//! is internally consistent for the supplied certificate bytes.

use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-query-cert-audit",
    version,
    about = "Validate query-scoped connectome reliability certificate invariants"
)]
struct Args {
    /// Certificate JSON file.
    #[arg(long)]
    certificate: PathBuf,
    /// Optional JSON report path; stdout when omitted.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Certificate {
    schema_version: u32,
    certificate_id: String,
    state: String,
    query: QueryBinding,
    source_revisions: Vec<SourceRevision>,
    dependency_closure: Vec<Dependency>,
    applied_corrections: Vec<Correction>,
    evidence_bindings: Vec<EvidenceBinding>,
    quality_gates: Vec<QualityGate>,
    uncertainty_components: Vec<UncertaintyComponent>,
    scope_exclusions: Vec<String>,
    invalidation_policy_version: String,
    invalidation: Option<Invalidation>,
    verifier: Verifier,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QueryBinding {
    query_id: String,
    specification_sha256: String,
    implementation_sha256: String,
    parameters_sha256: String,
    scope: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceRevision {
    source_kind: String,
    source_id: String,
    revision_id: String,
    content_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Dependency {
    object_kind: String,
    object_id: String,
    source_revision_sha256: String,
    evidence_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Correction {
    correction_id: String,
    order: usize,
    correction_sha256: String,
    review_decision: String,
    application_state: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceBinding {
    evidence_type: String,
    artifact_id: String,
    artifact_sha256: String,
    role: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QualityGate {
    gate_id: String,
    status: String,
    critical: bool,
    scope_note: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UncertaintyComponent {
    component_type: String,
    status: String,
    #[serde(default)]
    estimate: Option<f64>,
    #[serde(default)]
    interval_95: Option<[f64; 2]>,
    assumptions: Vec<String>,
    #[serde(default)]
    scope_note: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Invalidation {
    event_sha256: String,
    changed_object_kind: String,
    changed_object_id: String,
    previous_revision_sha256: String,
    current_revision_sha256: String,
    reason: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Verifier {
    tool_id: String,
    tool_version: String,
    binary_sha256: String,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: u32,
    tool: &'static str,
    certificate_id: Option<String>,
    certificate_sha256: String,
    status: &'static str,
    certificate_consistency_check_passed: bool,
    checks_failed: usize,
    errors: Vec<String>,
    scope_note: &'static str,
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn allowed(value: &str, choices: &[&str]) -> bool {
    choices.contains(&value)
}

fn push_if_invalid(errors: &mut Vec<String>, condition: bool, message: impl Into<String>) {
    if condition {
        errors.push(message.into());
    }
}

fn validate(certificate: &Certificate) -> Vec<String> {
    let mut errors = Vec::new();

    push_if_invalid(&mut errors, certificate.schema_version != 1, "schema_version must equal 1");
    push_if_invalid(&mut errors, !nonempty(&certificate.certificate_id), "certificate_id must be non-empty");
    push_if_invalid(
        &mut errors,
        !allowed(
            &certificate.state,
            &[
                "CERTIFIED_WITHIN_DECLARED_SCOPE",
                "CONDITIONAL",
                "NOT_CERTIFIED",
                "INVALIDATED_BY_REVISION",
            ],
        ),
        "state is not a recognized certificate state",
    );
    push_if_invalid(&mut errors, !nonempty(&certificate.query.query_id), "query.query_id must be non-empty");
    push_if_invalid(&mut errors, !nonempty(&certificate.query.scope), "query.scope must be non-empty");
    for (name, digest) in [
        ("query.specification_sha256", certificate.query.specification_sha256.as_str()),
        ("query.implementation_sha256", certificate.query.implementation_sha256.as_str()),
        ("query.parameters_sha256", certificate.query.parameters_sha256.as_str()),
        ("verifier.binary_sha256", certificate.verifier.binary_sha256.as_str()),
    ] {
        push_if_invalid(&mut errors, !valid_sha256(digest), format!("{name} must be lowercase SHA-256"));
    }
    push_if_invalid(&mut errors, !nonempty(&certificate.verifier.tool_id), "verifier.tool_id must be non-empty");
    push_if_invalid(&mut errors, !nonempty(&certificate.verifier.tool_version), "verifier.tool_version must be non-empty");
    push_if_invalid(
        &mut errors,
        !nonempty(&certificate.invalidation_policy_version),
        "invalidation_policy_version must be non-empty",
    );

    push_if_invalid(&mut errors, certificate.source_revisions.is_empty(), "source_revisions must not be empty");
    push_if_invalid(&mut errors, certificate.dependency_closure.is_empty(), "dependency_closure must not be empty");
    push_if_invalid(&mut errors, certificate.evidence_bindings.is_empty(), "evidence_bindings must not be empty");
    push_if_invalid(&mut errors, certificate.quality_gates.is_empty(), "quality_gates must not be empty");
    push_if_invalid(
        &mut errors,
        certificate.uncertainty_components.is_empty(),
        "uncertainty_components must not be empty",
    );

    let mut source_keys = BTreeSet::new();
    let mut source_hashes = BTreeSet::new();
    for source in &certificate.source_revisions {
        push_if_invalid(
            &mut errors,
            !allowed(
                &source.source_kind,
                &["volume", "imaging", "segmentation", "annotation", "mesh", "skeleton", "synapse_table", "connectome_graph", "other"],
            ),
            format!("source {} has invalid source_kind", source.source_id),
        );
        push_if_invalid(
            &mut errors,
            !nonempty(&source.source_id) || !nonempty(&source.revision_id),
            "source_id and revision_id must be non-empty",
        );
        push_if_invalid(
            &mut errors,
            !valid_sha256(&source.content_sha256),
            format!("source {} has malformed content_sha256", source.source_id),
        );
        let key = (source.source_kind.clone(), source.source_id.clone(), source.revision_id.clone());
        push_if_invalid(&mut errors, !source_keys.insert(key), "duplicate source revision identity");
        source_hashes.insert(source.content_sha256.clone());
    }

    let mut dependency_keys = BTreeSet::new();
    for dependency in &certificate.dependency_closure {
        push_if_invalid(
            &mut errors,
            !allowed(
                &dependency.object_kind,
                &["volume_region", "neuron", "root", "segment", "synapse", "edge", "morphology_subgraph", "coordinate_transform", "query_derived_value", "other"],
            ),
            format!("dependency {} has invalid object_kind", dependency.object_id),
        );
        push_if_invalid(&mut errors, !nonempty(&dependency.object_id), "dependency object_id must be non-empty");
        push_if_invalid(
            &mut errors,
            !valid_sha256(&dependency.source_revision_sha256)
                || !valid_sha256(&dependency.evidence_sha256),
            format!("dependency {} has malformed digest", dependency.object_id),
        );
        push_if_invalid(
            &mut errors,
            !source_hashes.contains(&dependency.source_revision_sha256),
            format!("dependency {} refers to a source revision not bound by this certificate", dependency.object_id),
        );
        let key = (
            dependency.object_kind.clone(),
            dependency.object_id.clone(),
            dependency.source_revision_sha256.clone(),
        );
        push_if_invalid(&mut errors, !dependency_keys.insert(key), "duplicate dependency closure entry");
    }

    let mut correction_ids = BTreeSet::new();
    let mut correction_orders = BTreeSet::new();
    for correction in &certificate.applied_corrections {
        push_if_invalid(
            &mut errors,
            !nonempty(&correction.correction_id),
            "correction_id must be non-empty",
        );
        push_if_invalid(
            &mut errors,
            !valid_sha256(&correction.correction_sha256),
            format!("correction {} has malformed digest", correction.correction_id),
        );
        push_if_invalid(
            &mut errors,
            !allowed(&correction.review_decision, &["accept", "reject", "abstain", "not_reviewed"]),
            format!("correction {} has invalid review_decision", correction.correction_id),
        );
        push_if_invalid(
            &mut errors,
            !allowed(
                &correction.application_state,
                &["applied_to_reversible_branch", "not_applied", "stale_conflict", "reverted"],
            ),
            format!("correction {} has invalid application_state", correction.correction_id),
        );
        push_if_invalid(
            &mut errors,
            correction.application_state == "applied_to_reversible_branch"
                && correction.review_decision != "accept",
            format!("correction {} was applied without an accept decision", correction.correction_id),
        );
        push_if_invalid(
            &mut errors,
            !correction_ids.insert(correction.correction_id.clone()),
            "duplicate correction_id",
        );
        push_if_invalid(
            &mut errors,
            !correction_orders.insert(correction.order),
            "duplicate correction order",
        );
    }
    for (expected, actual) in correction_orders.iter().enumerate() {
        push_if_invalid(
            &mut errors,
            expected != *actual,
            "correction orders must be contiguous and start at zero",
        );
    }

    let mut evidence_ids = BTreeSet::new();
    for evidence in &certificate.evidence_bindings {
        push_if_invalid(
            &mut errors,
            !allowed(
                &evidence.evidence_type,
                &[
                    "dataset_intake", "candidate_generation_audit", "candidate_pool_audit",
                    "multi_rater_audit", "unseen_error_sensitivity", "query_impact",
                    "correction_portfolio", "spatial_null_model", "morphology_features",
                    "neurite_extension_audit", "prospective_trial", "other",
                ],
            ),
            format!("evidence {} has invalid evidence_type", evidence.artifact_id),
        );
        push_if_invalid(
            &mut errors,
            !nonempty(&evidence.artifact_id) || !nonempty(&evidence.role),
            "evidence artifact_id and role must be non-empty",
        );
        push_if_invalid(
            &mut errors,
            !valid_sha256(&evidence.artifact_sha256),
            format!("evidence {} has malformed artifact_sha256", evidence.artifact_id),
        );
        push_if_invalid(
            &mut errors,
            !evidence_ids.insert((evidence.evidence_type.clone(), evidence.artifact_id.clone())),
            "duplicate evidence binding identity",
        );
    }

    let mut gate_ids = BTreeSet::new();
    let mut critical_failure = false;
    let mut all_gates_pass = true;
    for gate in &certificate.quality_gates {
        push_if_invalid(&mut errors, !nonempty(&gate.gate_id), "gate_id must be non-empty");
        push_if_invalid(&mut errors, !nonempty(&gate.scope_note), format!("gate {} must have a scope_note", gate.gate_id));
        push_if_invalid(
            &mut errors,
            !allowed(&gate.status, &["PASS", "FAIL", "NOT_ESTIMABLE", "NOT_APPLICABLE"]),
            format!("gate {} has invalid status", gate.gate_id),
        );
        push_if_invalid(
            &mut errors,
            !gate_ids.insert(gate.gate_id.clone()),
            "duplicate quality gate identity",
        );
        if gate.critical && gate.status == "FAIL" {
            critical_failure = true;
        }
        if gate.status != "PASS" {
            all_gates_pass = false;
        }
    }

    let mut uncertainty_types = BTreeSet::new();
    let mut all_uncertainty_estimated = true;
    for component in &certificate.uncertainty_components {
        push_if_invalid(
            &mut errors,
            !allowed(
                &component.component_type,
                &[
                    "sampling_design", "candidate_pool_residual", "rater_reference", "unseen_error_mse",
                    "morphology_model", "extension_path", "correction_interaction", "query_measurement",
                    "null_model", "other",
                ],
            ),
            format!("uncertainty component {} has invalid component_type", component.component_type),
        );
        push_if_invalid(
            &mut errors,
            !allowed(&component.status, &["ESTIMATED", "SENSITIVITY_ONLY", "NOT_ESTIMABLE"]),
            format!("uncertainty component {} has invalid status", component.component_type),
        );
        push_if_invalid(
            &mut errors,
            !uncertainty_types.insert(component.component_type.clone()),
            "duplicate uncertainty component type",
        );
        if component.status != "ESTIMATED" {
            all_uncertainty_estimated = false;
        }
        if component.status == "ESTIMATED" && component.estimate.is_none() {
            errors.push(format!("estimated component {} must include a point estimate", component.component_type));
        }
        if let Some(note) = &component.scope_note {
            push_if_invalid(&mut errors, !nonempty(note), format!("uncertainty component {} has an empty scope_note", component.component_type));
        }
        if let Some(estimate) = component.estimate {
            push_if_invalid(
                &mut errors,
                !estimate.is_finite(),
                format!("uncertainty component {} has non-finite estimate", component.component_type),
            );
        }
        if let Some(interval) = component.interval_95 {
            push_if_invalid(
                &mut errors,
                !interval[0].is_finite() || !interval[1].is_finite() || interval[0] > interval[1],
                format!("uncertainty component {} has invalid 95% interval", component.component_type),
            );
            if let Some(estimate) = component.estimate {
                push_if_invalid(
                    &mut errors,
                    estimate.is_finite() && (estimate < interval[0] || estimate > interval[1]),
                    format!("uncertainty component {} point estimate lies outside interval", component.component_type),
                );
            }
        }
        if component.status == "NOT_ESTIMABLE" {
            push_if_invalid(
                &mut errors,
                component.estimate.is_some() || component.interval_95.is_some(),
                format!("not-estimable component {} must not claim an estimate or interval", component.component_type),
            );
        }
        push_if_invalid(
            &mut errors,
            component.assumptions.iter().any(|assumption| !nonempty(assumption)),
            format!("uncertainty component {} has an empty assumption", component.component_type),
        );
    }

    push_if_invalid(
        &mut errors,
        certificate.scope_exclusions.iter().any(|item| !nonempty(item)),
        "scope_exclusions must not contain empty strings",
    );

    if certificate.state == "CERTIFIED_WITHIN_DECLARED_SCOPE" {
        push_if_invalid(
            &mut errors,
            !all_gates_pass || !all_uncertainty_estimated,
            "certified state requires every quality gate PASS and every uncertainty component ESTIMATED",
        );
    }
    if certificate.state == "CONDITIONAL" {
        push_if_invalid(
            &mut errors,
            critical_failure,
            "conditional state is forbidden when a critical quality gate FAILS; use NOT_CERTIFIED",
        );
        push_if_invalid(
            &mut errors,
            all_gates_pass && all_uncertainty_estimated,
            "conditional state requires at least one non-PASS gate or non-estimated uncertainty component",
        );
    }
    if certificate.state == "INVALIDATED_BY_REVISION" {
        match &certificate.invalidation {
            None => errors.push("invalidated state requires an invalidation event".into()),
            Some(invalidation) => {
                push_if_invalid(
                    &mut errors,
                    !valid_sha256(&invalidation.event_sha256)
                        || !valid_sha256(&invalidation.previous_revision_sha256)
                        || !valid_sha256(&invalidation.current_revision_sha256),
                    "invalidation event and revision digests must be lowercase SHA-256",
                );
                push_if_invalid(
                    &mut errors,
                    invalidation.previous_revision_sha256 == invalidation.current_revision_sha256,
                    "invalidation must identify a changed revision digest",
                );
                push_if_invalid(
                    &mut errors,
                    !nonempty(&invalidation.changed_object_kind)
                        || !nonempty(&invalidation.changed_object_id)
                        || !nonempty(&invalidation.reason),
                    "invalidation object identity and reason must be non-empty",
                );
                push_if_invalid(
                    &mut errors,
                    !allowed(
                        &invalidation.changed_object_kind,
                        &["volume_region", "neuron", "root", "segment", "synapse", "edge", "morphology_subgraph", "coordinate_transform", "query_derived_value", "source_revision"],
                    ),
                    "invalidation changed_object_kind is not recognized",
                );
            }
        }
    } else {
        push_if_invalid(
            &mut errors,
            certificate.invalidation.is_some(),
            "invalidation details are only valid for INVALIDATED_BY_REVISION state",
        );
    }

    errors
}

fn run(args: Args) -> Result<i32, String> {
    let bytes = fs::read(&args.certificate)
        .map_err(|error| format!("could not read {}: {error}", args.certificate.display()))?;
    let certificate: Certificate = serde_json::from_slice(&bytes)
        .map_err(|error| format!("could not parse certificate JSON: {error}"))?;
    let errors = validate(&certificate);
    let checks_failed = errors.len();
    let status = if checks_failed == 0 { "PASS" } else { "FAIL" };
    let report = AuditReport {
        schema_version: 1,
        tool: "connectome-query-cert-audit",
        certificate_id: Some(certificate.certificate_id.clone()),
        certificate_sha256: sha256(&bytes),
        status,
        certificate_consistency_check_passed: checks_failed == 0,
        checks_failed,
        errors,
        scope_note: "Internal certificate consistency only; source bytes, external digests, scientific truth, and biological correctness are not independently verified.",
    };
    let rendered = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("could not serialize report: {error}"))?;
    if let Some(output) = args.output {
        fs::write(&output, format!("{rendered}\n"))
            .map_err(|error| format!("could not write {}: {error}", output.display()))?;
    } else {
        println!("{rendered}");
    }
    Ok(if status == "PASS" { 0 } else { 1 })
}

fn main() -> ExitCode {
    match Args::try_parse() {
        Ok(args) => match run(args) {
            Ok(0) => ExitCode::SUCCESS,
            Ok(_) => ExitCode::from(1),
            Err(message) => {
                eprintln!("connectome-query-cert-audit: {message}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            let code = error.exit_code();
            let _ = error.print();
            ExitCode::from(code as u8)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Certificate {
        let bytes = include_bytes!("../../../../docs/neuroscience/examples/CONNECTOME_QUERY_RELIABILITY_CERTIFICATE.example.json");
        serde_json::from_slice(bytes).expect("synthetic certificate should parse")
    }

    #[test]
    fn synthetic_conditional_certificate_is_internally_consistent() {
        let certificate = fixture();
        assert!(validate(&certificate).is_empty(), "{:?}", validate(&certificate));
    }

    #[test]
    fn certified_state_rejects_unestimable_critical_gate() {
        let mut certificate = fixture();
        certificate.state = "CERTIFIED_WITHIN_DECLARED_SCOPE".into();
        assert!(validate(&certificate).iter().any(|error| error.contains("certified state requires")));
    }

    #[test]
    fn conditional_state_rejects_failed_critical_gate() {
        let mut certificate = fixture();
        certificate.state = "CONDITIONAL".into();
        certificate.quality_gates[1].status = "FAIL".into();
        assert!(validate(&certificate).iter().any(|error| error.contains("critical quality gate FAILS")));
    }

    #[test]
    fn dependency_must_bind_a_declared_source_revision() {
        let mut certificate = fixture();
        certificate.dependency_closure[0].source_revision_sha256 = "9".repeat(64);
        assert!(validate(&certificate).iter().any(|error| error.contains("not bound by this certificate")));
    }

    #[test]
    fn duplicate_correction_orders_are_rejected() {
        let mut certificate = fixture();
        certificate.applied_corrections = vec![
            Correction {
                correction_id: "c1".into(),
                order: 0,
                correction_sha256: "a".repeat(64),
                review_decision: "accept".into(),
                application_state: "applied_to_reversible_branch".into(),
            },
            Correction {
                correction_id: "c2".into(),
                order: 0,
                correction_sha256: "b".repeat(64),
                review_decision: "accept".into(),
                application_state: "applied_to_reversible_branch".into(),
            },
        ];
        assert!(validate(&certificate).iter().any(|error| error.contains("duplicate correction order")));
    }

    #[test]
    fn applying_a_correction_requires_acceptance() {
        let mut certificate = fixture();
        certificate.applied_corrections = vec![Correction {
            correction_id: "c1".into(),
            order: 0,
            correction_sha256: "a".repeat(64),
            review_decision: "abstain".into(),
            application_state: "applied_to_reversible_branch".into(),
        }];
        assert!(validate(&certificate).iter().any(|error| error.contains("applied without an accept")));
    }

    #[test]
    fn malformed_digest_is_rejected() {
        let mut certificate = fixture();
        certificate.query.specification_sha256 = "NOT_A_SHA".into();
        assert!(validate(&certificate).iter().any(|error| error.contains("query.specification_sha256")));
    }

    #[test]
    fn not_estimable_component_cannot_claim_interval() {
        let mut certificate = fixture();
        certificate.uncertainty_components[0].interval_95 = Some([0.0, 0.0]);
        assert!(validate(&certificate).iter().any(|error| error.contains("not-estimable component")));
    }

    #[test]
    fn estimated_component_requires_a_point_estimate() {
        let mut certificate = fixture();
        certificate.uncertainty_components[0].status = "ESTIMATED".into();
        certificate.uncertainty_components[0].estimate = None;
        certificate.uncertainty_components[0].interval_95 = Some([0.1, 0.9]);
        assert!(validate(&certificate).iter().any(|error| error.contains("must include a point estimate")));
    }

    #[test]
    fn invalidated_state_requires_a_revision_event() {
        let mut certificate = fixture();
        certificate.state = "INVALIDATED_BY_REVISION".into();
        assert!(validate(&certificate).iter().any(|error| error.contains("requires an invalidation event")));
    }

    #[test]
    fn invalidated_certificate_records_a_changed_revision() {
        let mut certificate = fixture();
        certificate.state = "INVALIDATED_BY_REVISION".into();
        certificate.invalidation = Some(Invalidation {
            event_sha256: "6".repeat(64),
            changed_object_kind: "source_revision".into(),
            changed_object_id: "synthetic-volume".into(),
            previous_revision_sha256: "7".repeat(64),
            current_revision_sha256: "8".repeat(64),
            reason: "synthetic source revision changed".into(),
        });
        assert!(validate(&certificate).is_empty(), "{:?}", validate(&certificate));
    }

    #[test]
    fn estimate_must_lie_inside_its_interval() {
        let mut certificate = fixture();
        certificate.uncertainty_components[0].status = "ESTIMATED".into();
        certificate.uncertainty_components[0].estimate = Some(5.0);
        certificate.uncertainty_components[0].interval_95 = Some([0.0, 1.0]);
        assert!(validate(&certificate).iter().any(|error| error.contains("point estimate lies outside interval")));
    }
}
