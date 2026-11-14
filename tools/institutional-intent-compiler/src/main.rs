use serde::{Deserialize, Serialize};
use std::{collections::{BTreeMap, BTreeSet}, env, fs, process};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: String,
    brief_id: String,
    baseline_version: u64,
    sources: Vec<Source>,
    derivations: Vec<ApprovedDerivation>,
    requirements: Vec<Requirement>,
    decisions: Vec<Decision>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    id: String,
    sha256: String,
    byte_length: u64,
    classification: SourceClassification,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SourceClassification {
    Public,
    CustomerConfidential,
    Restricted,
    Personal,
    SecuritySensitive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceSpan {
    source_id: String,
    start_byte: u64,
    end_byte: u64,
    quote_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ApprovedDerivation {
    id: String,
    rationale: String,
    source_spans: Vec<SourceSpan>,
    approval_decision_ref: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Requirement {
    id: String,
    statement: String,
    class: RequirementClass,
    declared_modality: Modality,
    proposed_modality: Modality,
    lifecycle: Lifecycle,
    traceability: Traceability,
    verification: Verification,
    owner: Owner,
    links: Vec<RequirementLink>,
    review_decision_ref: Option<String>,
    modality_change_decision_ref: Option<String>,
    unverifiable_exception_decision_ref: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RequirementClass {
    Goal,
    Functional,
    Quality,
    Constraint,
    Interface,
    DataPrivacy,
    Security,
    Safety,
    LegalReviewNeeded,
    Operational,
    Continuity,
    NonGoal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Modality {
    Mandatory,
    Conditional,
    Preferred,
    Optional,
    Prohibited,
    Unresolved,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Lifecycle {
    Candidate,
    NeedsClarification,
    InConflict,
    ReviewReady,
    Accepted,
    Rejected,
    Deferred,
    Superseded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Traceability {
    source_spans: Vec<SourceSpan>,
    derivation_ref: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Verification {
    method: VerificationMethod,
    acceptance_condition: Option<String>,
    limitation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum VerificationMethod {
    Inspection,
    Analysis,
    Demonstration,
    Test,
    SourceReview,
    ExternalAuthorityConfirmation,
    NotYetVerifiable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Owner {
    system: OwnerSystem,
    reference: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum OwnerSystem {
    Symthaea,
    Mycelix,
    Symtropy,
    SolAtlas,
    CustomerSystem,
    ExternalAuthority,
    HumanReviewer,
    Unassigned,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequirementLink {
    kind: LinkKind,
    target_requirement_id: String,
    resolution_decision_ref: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum LinkKind {
    DependsOn,
    ConflictsWith,
    DuplicateOf,
    Supersedes,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Decision {
    id: String,
    subject_type: DecisionSubject,
    subject_id: String,
    action: DecisionAction,
    reviewer_ref: String,
    authority_profile_ref: String,
    baseline_version: u64,
    rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum DecisionSubject {
    Requirement,
    Derivation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum DecisionAction {
    Accept,
    Amend,
    Reject,
    Defer,
    RequestClarification,
    ApproveModalityChange,
    ApproveNotYetVerifiable,
    ApproveDerivation,
    ResolveConflict,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Diagnostic {
    code: String,
    path: String,
    message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ValidationReport {
    status: String,
    diagnostic_count: usize,
    diagnostics: Vec<Diagnostic>,
}

fn nonblank(value: &str) -> bool {
    !value.trim().is_empty()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn push(diagnostics: &mut Vec<Diagnostic>, code: &str, path: impl Into<String>, message: impl Into<String>) {
    diagnostics.push(Diagnostic {
        code: code.to_owned(),
        path: path.into(),
        message: message.into(),
    });
}

fn check_span(
    span: &SourceSpan,
    path: &str,
    sources: &BTreeMap<String, &Source>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    match sources.get(&span.source_id) {
        None => push(
            diagnostics,
            "unknown_source",
            path,
            format!("source '{}' is not declared in the manifest", span.source_id),
        ),
        Some(source) => {
            if span.start_byte >= span.end_byte || span.end_byte > source.byte_length {
                push(
                    diagnostics,
                    "invalid_source_span",
                    path,
                    format!(
                        "span [{}, {}) is outside source '{}' byte length {}",
                        span.start_byte, span.end_byte, span.source_id, source.byte_length
                    ),
                );
            }
        }
    }
    if !is_sha256(&span.quote_sha256) {
        push(
            diagnostics,
            "invalid_quote_digest",
            format!("{path}.quote_sha256"),
            "quote_sha256 must be exactly 64 hexadecimal characters",
        );
    }
}

fn decision_matches<'a>(
    decision_ref: &Option<String>,
    decisions: &'a BTreeMap<String, &'a Decision>,
    subject_type: DecisionSubject,
    subject_id: &str,
    allowed_actions: &[DecisionAction],
    baseline_version: u64,
    path: &str,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<&'a Decision> {
    let Some(id) = decision_ref.as_ref() else {
        push(diagnostics, "missing_decision", path, "required decision reference is missing");
        return None;
    };
    let Some(decision) = decisions.get(id) else {
        push(
            diagnostics,
            "unknown_decision",
            path,
            format!("decision '{id}' is not declared in the manifest"),
        );
        return None;
    };
    if std::mem::discriminant(&decision.subject_type) != std::mem::discriminant(&subject_type)
        || decision.subject_id != subject_id
    {
        push(
            diagnostics,
            "decision_subject_mismatch",
            path,
            format!("decision '{id}' does not apply to this subject"),
        );
    }
    if !allowed_actions.contains(&decision.action) {
        push(
            diagnostics,
            "decision_action_mismatch",
            path,
            format!("decision '{id}' has action {:?}, which is not allowed here", decision.action),
        );
    }
    if decision.baseline_version != baseline_version {
        push(
            diagnostics,
            "decision_baseline_mismatch",
            path,
            format!(
                "decision '{id}' is for baseline {}, expected {}",
                decision.baseline_version, baseline_version
            ),
        );
    }
    Some(*decision)
}

fn validate(manifest: &Manifest) -> ValidationReport {
    let mut diagnostics = Vec::new();

    if manifest.schema_version != "0.1" {
        push(&mut diagnostics, "unsupported_schema_version", "schema_version", "only schema_version '0.1' is supported");
    }
    if !nonblank(&manifest.brief_id) {
        push(&mut diagnostics, "empty_brief_id", "brief_id", "brief_id must not be blank");
    }
    if manifest.baseline_version == 0 {
        push(&mut diagnostics, "invalid_baseline_version", "baseline_version", "baseline_version must be greater than zero");
    }

    let mut sources = BTreeMap::new();
    for (index, source) in manifest.sources.iter().enumerate() {
        let path = format!("sources[{index}]");
        if !nonblank(&source.id) {
            push(&mut diagnostics, "empty_source_id", format!("{path}.id"), "source id must not be blank");
        }
        if sources.insert(source.id.clone(), source).is_some() {
            push(&mut diagnostics, "duplicate_source_id", format!("{path}.id"), format!("source id '{}' is duplicated", source.id));
        }
        if !is_sha256(&source.sha256) {
            push(&mut diagnostics, "invalid_source_digest", format!("{path}.sha256"), "sha256 must be exactly 64 hexadecimal characters");
        }
    }

    let mut decisions = BTreeMap::new();
    for (index, decision) in manifest.decisions.iter().enumerate() {
        let path = format!("decisions[{index}]");
        if !nonblank(&decision.id) {
            push(&mut diagnostics, "empty_decision_id", format!("{path}.id"), "decision id must not be blank");
        }
        if decisions.insert(decision.id.clone(), decision).is_some() {
            push(&mut diagnostics, "duplicate_decision_id", format!("{path}.id"), format!("decision id '{}' is duplicated", decision.id));
        }
        if !nonblank(&decision.subject_id)
            || !nonblank(&decision.reviewer_ref)
            || !nonblank(&decision.authority_profile_ref)
            || !nonblank(&decision.rationale)
        {
            push(&mut diagnostics, "incomplete_decision", path, "subject, reviewer, authority profile and rationale must be nonblank");
        }
    }

    let mut derivations = BTreeMap::new();
    for (index, derivation) in manifest.derivations.iter().enumerate() {
        let path = format!("derivations[{index}]");
        if derivations.insert(derivation.id.clone(), derivation).is_some() {
            push(&mut diagnostics, "duplicate_derivation_id", format!("{path}.id"), format!("derivation id '{}' is duplicated", derivation.id));
        }
        if !nonblank(&derivation.rationale) || derivation.source_spans.is_empty() {
            push(&mut diagnostics, "incomplete_derivation", path.clone(), "an approved derivation needs a rationale and at least one source span");
        }
        for (span_index, span) in derivation.source_spans.iter().enumerate() {
            check_span(span, &format!("{path}.source_spans[{span_index}]"), &sources, &mut diagnostics);
        }
        let approval_ref = Some(derivation.approval_decision_ref.clone());
        decision_matches(
            &approval_ref,
            &decisions,
            DecisionSubject::Derivation,
            &derivation.id,
            &[DecisionAction::ApproveDerivation],
            manifest.baseline_version,
            &format!("{path}.approval_decision_ref"),
            &mut diagnostics,
        );
    }

    let mut requirements = BTreeMap::new();
    for (index, requirement) in manifest.requirements.iter().enumerate() {
        if requirements.insert(requirement.id.clone(), requirement).is_some() {
            push(
                &mut diagnostics,
                "duplicate_requirement_id",
                format!("requirements[{index}].id"),
                format!("requirement id '{}' is duplicated", requirement.id),
            );
        }
    }

    for (index, requirement) in manifest.requirements.iter().enumerate() {
        let path = format!("requirements[{index}]");
        if !nonblank(&requirement.id) || !nonblank(&requirement.statement) {
            push(&mut diagnostics, "incomplete_requirement", path.clone(), "requirement id and statement must be nonblank");
        }

        for (span_index, span) in requirement.traceability.source_spans.iter().enumerate() {
            check_span(span, &format!("{path}.traceability.source_spans[{span_index}]"), &sources, &mut diagnostics);
        }

        if let Some(derivation_ref) = requirement.traceability.derivation_ref.as_ref() {
            if !derivations.contains_key(derivation_ref) {
                push(
                    &mut diagnostics,
                    "unknown_derivation",
                    format!("{path}.traceability.derivation_ref"),
                    format!("derivation '{derivation_ref}' is not declared"),
                );
            }
        }

        if requirement.lifecycle == Lifecycle::Accepted {
            if requirement.traceability.source_spans.is_empty() && requirement.traceability.derivation_ref.is_none() {
                push(
                    &mut diagnostics,
                    "accepted_without_trace",
                    format!("{path}.traceability"),
                    "accepted requirements need a source span or an approved derivation reference",
                );
            }
            if requirement.proposed_modality == Modality::Unresolved {
                push(
                    &mut diagnostics,
                    "accepted_unresolved_modality",
                    format!("{path}.proposed_modality"),
                    "an accepted requirement cannot retain unresolved modality",
                );
            }
            if matches!(requirement.owner.system, OwnerSystem::Unassigned) || requirement.owner.reference.as_deref().is_none_or(|v| !nonblank(v)) {
                push(
                    &mut diagnostics,
                    "accepted_without_owner",
                    format!("{path}.owner"),
                    "accepted requirements need an assigned owner system and owner reference",
                );
            }
            decision_matches(
                &requirement.review_decision_ref,
                &decisions,
                DecisionSubject::Requirement,
                &requirement.id,
                &[DecisionAction::Accept, DecisionAction::Amend],
                manifest.baseline_version,
                &format!("{path}.review_decision_ref"),
                &mut diagnostics,
            );
            match requirement.verification.method {
                VerificationMethod::NotYetVerifiable => {
                    if requirement.verification.limitation.as_deref().is_none_or(|v| !nonblank(v)) {
                        push(
                            &mut diagnostics,
                            "missing_verification_limitation",
                            format!("{path}.verification.limitation"),
                            "NotYetVerifiable requires an explicit limitation",
                        );
                    }
                    decision_matches(
                        &requirement.unverifiable_exception_decision_ref,
                        &decisions,
                        DecisionSubject::Requirement,
                        &requirement.id,
                        &[DecisionAction::ApproveNotYetVerifiable],
                        manifest.baseline_version,
                        &format!("{path}.unverifiable_exception_decision_ref"),
                        &mut diagnostics,
                    );
                }
                _ => {
                    if requirement.verification.acceptance_condition.as_deref().is_none_or(|v| !nonblank(v)) {
                        push(
                            &mut diagnostics,
                            "missing_acceptance_condition",
                            format!("{path}.verification.acceptance_condition"),
                            "accepted requirements need a nonblank acceptance condition",
                        );
                    }
                }
            }
        }

        if requirement.declared_modality != requirement.proposed_modality {
            decision_matches(
                &requirement.modality_change_decision_ref,
                &decisions,
                DecisionSubject::Requirement,
                &requirement.id,
                &[DecisionAction::ApproveModalityChange],
                manifest.baseline_version,
                &format!("{path}.modality_change_decision_ref"),
                &mut diagnostics,
            );
        }

        let mut seen_links = BTreeSet::new();
        for (link_index, link) in requirement.links.iter().enumerate() {
            let link_path = format!("{path}.links[{link_index}]");
            if !seen_links.insert((link.kind.clone(), link.target_requirement_id.clone())) {
                push(&mut diagnostics, "duplicate_requirement_link", link_path.clone(), "duplicate link of the same kind to the same requirement");
            }
            let Some(target) = requirements.get(&link.target_requirement_id) else {
                push(
                    &mut diagnostics,
                    "unknown_requirement_link_target",
                    link_path,
                    format!("target requirement '{}' is not declared", link.target_requirement_id),
                );
                continue;
            };
            if requirement.lifecycle == Lifecycle::Accepted && link.kind == LinkKind::DependsOn && target.lifecycle != Lifecycle::Accepted {
                push(
                    &mut diagnostics,
                    "accepted_dependency_not_accepted",
                    link_path.clone(),
                    format!("accepted requirement depends on '{}' which is not accepted", target.id),
                );
            }
            if requirement.lifecycle == Lifecycle::Accepted
                && target.lifecycle == Lifecycle::Accepted
                && link.kind == LinkKind::ConflictsWith
            {
                let resolution = link.resolution_decision_ref.as_ref().and_then(|id| decisions.get(id));
                match resolution {
                    Some(decision)
                        if decision.action == DecisionAction::ResolveConflict
                            && decision.subject_id == requirement.id
                            && decision.baseline_version == manifest.baseline_version => {}
                    _ => push(
                        &mut diagnostics,
                        "unresolved_accepted_conflict",
                        link_path,
                        format!("accepted requirement conflicts with accepted '{}' without a matching conflict-resolution decision", target.id),
                    ),
                }
            }
        }
    }

    diagnostics.sort_by(|a, b| (&a.path, &a.code, &a.message).cmp(&(&b.path, &b.code, &b.message)));
    let status = if diagnostics.is_empty() { "STRUCTURALLY_VALID" } else { "REJECTED" };
    ValidationReport {
        status: status.to_owned(),
        diagnostic_count: diagnostics.len(),
        diagnostics,
    }
}

fn run() -> Result<i32, String> {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        return Err("usage: institutional-intent-compiler <manifest.json>".to_owned());
    };
    if args.next().is_some() {
        return Err("expected exactly one manifest path".to_owned());
    }
    let input = fs::read_to_string(&path).map_err(|e| format!("cannot read '{path}': {e}"))?;
    let manifest: Manifest = serde_json::from_str(&input).map_err(|e| format!("invalid manifest JSON: {e}"))?;
    let report = validate(&manifest);
    println!("{}", serde_json::to_string_pretty(&report).map_err(|e| format!("cannot serialize report: {e}"))?);
    Ok(if report.diagnostic_count == 0 { 0 } else { 2 })
}

fn main() {
    let exit_code = match run() {
        Ok(code) => code,
        Err(message) => {
            eprintln!("{message}");
            2
        }
    };
    process::exit(exit_code);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_manifest() -> Manifest {
        Manifest {
            schema_version: "0.1".to_owned(),
            brief_id: "synthetic-brief-001".to_owned(),
            baseline_version: 1,
            sources: vec![Source {
                id: "brief".to_owned(),
                sha256: "a".repeat(64),
                byte_length: 100,
                classification: SourceClassification::Public,
            }],
            derivations: vec![],
            requirements: vec![Requirement {
                id: "REQ-001".to_owned(),
                statement: "The system shall not disclose secrets in a public view.".to_owned(),
                class: RequirementClass::DataPrivacy,
                declared_modality: Modality::Mandatory,
                proposed_modality: Modality::Mandatory,
                lifecycle: Lifecycle::Accepted,
                traceability: Traceability {
                    source_spans: vec![SourceSpan {
                        source_id: "brief".to_owned(),
                        start_byte: 0,
                        end_byte: 24,
                        quote_sha256: "b".repeat(64),
                    }],
                    derivation_ref: None,
                },
                verification: Verification {
                    method: VerificationMethod::Test,
                    acceptance_condition: Some("No synthetic secret marker appears in public output.".to_owned()),
                    limitation: None,
                },
                owner: Owner {
                    system: OwnerSystem::Mycelix,
                    reference: Some("privacy-view-contract".to_owned()),
                },
                links: vec![],
                review_decision_ref: Some("DEC-001".to_owned()),
                modality_change_decision_ref: None,
                unverifiable_exception_decision_ref: None,
            }],
            decisions: vec![Decision {
                id: "DEC-001".to_owned(),
                subject_type: DecisionSubject::Requirement,
                subject_id: "REQ-001".to_owned(),
                action: DecisionAction::Accept,
                reviewer_ref: "reviewer:synthetic-01".to_owned(),
                authority_profile_ref: "profile:synthetic-review-v1".to_owned(),
                baseline_version: 1,
                rationale: "Accepted in synthetic fixture.".to_owned(),
            }],
        }
    }

    #[test]
    fn accepts_well_formed_source_grounded_baseline() {
        let report = validate(&valid_manifest());
        assert_eq!(report.status, "STRUCTURALLY_VALID");
        assert_eq!(report.diagnostic_count, 0);
    }

    #[test]
    fn rejects_accepted_requirement_without_source_or_derivation() {
        let mut manifest = valid_manifest();
        manifest.requirements[0].traceability.source_spans.clear();
        let report = validate(&manifest);
        assert!(report.diagnostics.iter().any(|d| d.code == "accepted_without_trace"));
    }

    #[test]
    fn rejects_unknown_source_reference() {
        let mut manifest = valid_manifest();
        manifest.requirements[0].traceability.source_spans[0].source_id = "missing".to_owned();
        let report = validate(&manifest);
        assert!(report.diagnostics.iter().any(|d| d.code == "unknown_source"));
    }

    #[test]
    fn rejects_unapproved_modality_change() {
        let mut manifest = valid_manifest();
        manifest.requirements[0].declared_modality = Modality::Preferred;
        manifest.requirements[0].proposed_modality = Modality::Mandatory;
        let report = validate(&manifest);
        assert!(report.diagnostics.iter().any(|d| d.code == "missing_decision"));
    }

    #[test]
    fn rejects_accepted_requirement_without_human_review() {
        let mut manifest = valid_manifest();
        manifest.requirements[0].review_decision_ref = None;
        let report = validate(&manifest);
        assert!(report.diagnostics.iter().any(|d| d.code == "missing_decision"));
    }

    #[test]
    fn rejects_not_yet_verifiable_without_explicit_exception() {
        let mut manifest = valid_manifest();
        manifest.requirements[0].verification.method = VerificationMethod::NotYetVerifiable;
        manifest.requirements[0].verification.acceptance_condition = None;
        manifest.requirements[0].verification.limitation = Some("No qualified test method yet.".to_owned());
        let report = validate(&manifest);
        assert!(report.diagnostics.iter().any(|d| d.code == "missing_decision"));
    }

    #[test]
    fn rejects_dependency_on_unaccepted_requirement() {
        let mut manifest = valid_manifest();
        manifest.requirements[0].links.push(RequirementLink {
            kind: LinkKind::DependsOn,
            target_requirement_id: "REQ-002".to_owned(),
            resolution_decision_ref: None,
        });
        manifest.requirements.push(Requirement {
            id: "REQ-002".to_owned(),
            statement: "A prerequisite remains a candidate.".to_owned(),
            class: RequirementClass::Functional,
            declared_modality: Modality::Mandatory,
            proposed_modality: Modality::Mandatory,
            lifecycle: Lifecycle::Candidate,
            traceability: Traceability { source_spans: vec![], derivation_ref: None },
            verification: Verification { method: VerificationMethod::Test, acceptance_condition: None, limitation: None },
            owner: Owner { system: OwnerSystem::Unassigned, reference: None },
            links: vec![],
            review_decision_ref: None,
            modality_change_decision_ref: None,
            unverifiable_exception_decision_ref: None,
        });
        let report = validate(&manifest);
        assert!(report.diagnostics.iter().any(|d| d.code == "accepted_dependency_not_accepted"));
    }

    #[test]
    fn rejects_unknown_json_fields() {
        let json = r#"{
            "schema_version":"0.1",
            "brief_id":"x",
            "baseline_version":1,
            "sources":[],
            "derivations":[],
            "requirements":[],
            "decisions":[],
            "surprise":true
        }"#;
        let result = serde_json::from_str::<Manifest>(json);
        assert!(result.is_err());
    }

    #[test]
    fn diagnostic_order_is_stable() {
        let mut manifest = valid_manifest();
        manifest.requirements[0].traceability.source_spans[0].source_id = "missing".to_owned();
        manifest.requirements[0].review_decision_ref = None;
        let first = serde_json::to_string(&validate(&manifest)).unwrap();
        let second = serde_json::to_string(&validate(&manifest)).unwrap();
        assert_eq!(first, second);
    }
}
