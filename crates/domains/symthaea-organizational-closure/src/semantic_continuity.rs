//! Typed semantic continuity across organizational reconfiguration.
//!
//! Complements the finite LTS relation in identity.rs by evaluating typed
//! authority, consent, effects, provenance and substrate envelopes.
//! This module does not authenticate receipts or establish physical truth.
//! EvidenceRef inputs must come from an independently operated verifier.

use crate::ContentDigest;
use super::identity::{ContinuityQualification, RelationMode};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SemanticTransitionKey {
    pub action: String,
    pub target: String,
    pub outcome: String,
}

impl SemanticTransitionKey {
    pub fn new(action: impl Into<String>, target: impl Into<String>, outcome: impl Into<String>) -> Self {
        Self { action: action.into(), target: target.into(), outcome: outcome.into() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvidenceStatus { Verified, Rejected, Unavailable }

/// Disposition supplied by an independent receipt/oracle verifier.
/// A caller-controlled Verified value is not itself a security boundary.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EvidenceRef {
    pub reference: String,
    pub digest: ContentDigest,
    pub status: EvidenceStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrantStatus { Active, Revoked, Unknown }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorityBinding {
    pub principal_id: String,
    pub trust_root_digest: ContentDigest,
    pub grant_id: String,
    pub scopes: BTreeSet<String>,
    pub target_scope: BTreeSet<String>,
    pub status: GrantStatus,
    pub evidence: EvidenceRef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsentStatus { Active, Revoked, Unknown }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsentBinding {
    pub subject_id: String,
    pub consent_id: String,
    pub scope: BTreeSet<String>,
    pub status: ConsentStatus,
    pub evidence: EvidenceRef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EffectReversibility { Reversible, Irreversible, Unknown }

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EffectRecord {
    pub effect_id: String,
    pub class: String,
    pub target: String,
    pub reversibility: EffectReversibility,
    pub evidence: EvidenceRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineageStep {
    pub input_digest: ContentDigest,
    pub output_digest: ContentDigest,
    pub producer_id: String,
    pub consumer_id: String,
    pub producer_receipt: EvidenceRef,
    pub consumer_receipt: EvidenceRef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependencyStatus { Available, Unavailable, Denied, Unknown }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubstrateDependency {
    pub dependency_id: String,
    pub status: DependencyStatus,
    pub observation: EvidenceRef,
}

/// Semantic envelope for one externally observable transition.
/// effect_inventory_complete must be asserted by an independent observer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticTransition {
    pub key: SemanticTransitionKey,
    pub authority: AuthorityBinding,
    pub consent: Option<ConsentBinding>,
    pub effect_inventory_complete: bool,
    pub effect_inventory_receipt: EvidenceRef,
    pub effects: BTreeSet<EffectRecord>,
    pub lineage: Vec<LineageStep>,
    pub substrate: BTreeMap<String, SubstrateDependency>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticContinuityPolicy {
    pub mode: RelationMode,
    pub consent_required_for_actions: BTreeSet<String>,
    pub provenance_required_for_actions: BTreeSet<String>,
    /// Only additional reversible effects with these IDs may be admitted.
    /// Additional irreversible effects are rejected in v0.1.
    pub allowed_additional_reversible_effects: BTreeSet<String>,
    pub max_transition_checks: usize,
}

impl Default for SemanticContinuityPolicy {
    fn default() -> Self {
        Self {
            mode: RelationMode::Simulation,
            consent_required_for_actions: BTreeSet::new(),
            provenance_required_for_actions: BTreeSet::new(),
            allowed_additional_reversible_effects: BTreeSet::new(),
            max_transition_checks: 100_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SemanticPlane {
    TransitionCoverage, Authority, Consent, Effects, Provenance, Substrate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticFindingCode {
    EmptyBaselineTransitions,
    EmptyCandidateTransitions,
    DuplicateBaselineKey,
    DuplicateCandidateKey,
    MissingCandidateTransition,
    UnexpectedCandidateTransition,
    TransitionCheckBudgetExceeded,
    EmptySemanticLabel,
    InvalidEvidenceReference,
    EvidenceRejected,
    EvidenceUnavailable,
    AuthorityNotActive,
    AuthorityUnknown,
    AuthorityChanged,
    AuthorityScopeMissing,
    ConsentMissing,
    ConsentNotActive,
    ConsentUnknown,
    ConsentChanged,
    ConsentScopeMissing,
    EffectInventoryIncomplete,
    EffectInventoryDuplicate,
    UnknownEffectReversibility,
    BaselineEffectMissing,
    AdditionalIrreversibleEffect,
    AdditionalReversibleEffectNotAllowed,
    MissingRequiredLineage,
    InvalidLineageIdentity,
    BrokenLineageChain,
    MissingSubstrateObservation,
    SubstrateDependencyKeyMismatch,
    SubstrateDependencySetChanged,
    SubstrateStatusChanged,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticFinding {
    pub plane: SemanticPlane,
    pub code: SemanticFindingCode,
    pub transition: Option<SemanticTransitionKey>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticContinuityReport {
    pub qualification: ContinuityQualification,
    /// Per-plane results are retained independently; no plane compensates.
    pub plane_qualifications: BTreeMap<SemanticPlane, ContinuityQualification>,
    pub compared_transition_count: usize,
    pub candidate_only_transition_count: usize,
    pub comparison_checks: usize,
    pub findings: Vec<SemanticFinding>,
}

impl SemanticContinuityReport {
    pub fn preserves_semantic_continuity(&self) -> bool {
        self.qualification == ContinuityQualification::Preserved && self.findings.is_empty()
    }
}

fn push_finding(
    report: &mut SemanticContinuityReport,
    plane: SemanticPlane,
    code: SemanticFindingCode,
    transition: Option<SemanticTransitionKey>,
    detail: impl Into<String>,
    disposition: ContinuityQualification,
) {
    let current = report.plane_qualifications.get(&plane).copied()
        .unwrap_or(ContinuityQualification::Preserved);
    let updated = match (current, disposition) {
        (_, ContinuityQualification::NotPreserved)
        | (ContinuityQualification::NotPreserved, _) => ContinuityQualification::NotPreserved,
        (_, ContinuityQualification::Inconclusive)
        | (ContinuityQualification::Inconclusive, _) => ContinuityQualification::Inconclusive,
        _ => ContinuityQualification::Preserved,
    };
    report.plane_qualifications.insert(plane, updated);
    report.findings.push(SemanticFinding { plane, code, transition, detail: detail.into() });
}

fn finalize(report: &mut SemanticContinuityReport) {
    report.qualification = if report.plane_qualifications.values()
        .any(|v| *v == ContinuityQualification::NotPreserved) {
        ContinuityQualification::NotPreserved
    } else if report.plane_qualifications.values()
        .any(|v| *v == ContinuityQualification::Inconclusive) {
        ContinuityQualification::Inconclusive
    } else {
        ContinuityQualification::Preserved
    };
}

fn validate_label(value: &str, plane: SemanticPlane, key: &SemanticTransitionKey,
    report: &mut SemanticContinuityReport, label: &str) {
    if value.trim().is_empty() {
        push_finding(report, plane, SemanticFindingCode::EmptySemanticLabel, Some(key.clone()),
            format!("{label} must not be empty"), ContinuityQualification::NotPreserved);
    }
}

fn validate_evidence(evidence: &EvidenceRef, plane: SemanticPlane, key: &SemanticTransitionKey,
    report: &mut SemanticContinuityReport, label: &str) {
    if evidence.reference.trim().is_empty() {
        push_finding(report, plane, SemanticFindingCode::InvalidEvidenceReference, Some(key.clone()),
            format!("{label} has an empty evidence reference"), ContinuityQualification::NotPreserved);
        return;
    }
    match evidence.status {
        EvidenceStatus::Verified => {}
        EvidenceStatus::Rejected => push_finding(report, plane, SemanticFindingCode::EvidenceRejected,
            Some(key.clone()), format!("{label} was rejected by its evidence verifier"),
            ContinuityQualification::NotPreserved),
        EvidenceStatus::Unavailable => push_finding(report, plane, SemanticFindingCode::EvidenceUnavailable,
            Some(key.clone()), format!("{label} could not be independently qualified"),
            ContinuityQualification::Inconclusive),
    }
}

fn validate_record(record: &SemanticTransition, policy: &SemanticContinuityPolicy,
    report: &mut SemanticContinuityReport) {
    let key = &record.key;
    validate_label(&key.action, SemanticPlane::TransitionCoverage, key, report, "action");
    validate_label(&key.target, SemanticPlane::TransitionCoverage, key, report, "target");
    validate_label(&key.outcome, SemanticPlane::TransitionCoverage, key, report, "outcome");

    let authority = &record.authority;
    validate_label(&authority.principal_id, SemanticPlane::Authority, key, report, "principal ID");
    validate_label(&authority.grant_id, SemanticPlane::Authority, key, report, "grant ID");
    validate_evidence(&authority.evidence, SemanticPlane::Authority, key, report, "authority evidence");
    match authority.status {
        GrantStatus::Active => {}
        GrantStatus::Revoked => push_finding(report, SemanticPlane::Authority,
            SemanticFindingCode::AuthorityNotActive, Some(key.clone()), "authority grant is revoked",
            ContinuityQualification::NotPreserved),
        GrantStatus::Unknown => push_finding(report, SemanticPlane::Authority,
            SemanticFindingCode::AuthorityUnknown, Some(key.clone()), "authority grant status is unresolved",
            ContinuityQualification::Inconclusive),
    }
    if !authority.scopes.contains(&key.action) || !authority.target_scope.contains(&key.target)
        || authority.scopes.iter().any(|s| s.trim().is_empty())
        || authority.target_scope.iter().any(|s| s.trim().is_empty()) {
        push_finding(report, SemanticPlane::Authority, SemanticFindingCode::AuthorityScopeMissing,
            Some(key.clone()), "active grant does not cover the action/target or contains an empty scope",
            ContinuityQualification::NotPreserved);
    }

    let consent_required = policy.consent_required_for_actions.contains(&key.action);
    if consent_required && record.consent.is_none() {
        push_finding(report, SemanticPlane::Consent, SemanticFindingCode::ConsentMissing,
            Some(key.clone()), "this action requires consent but no consent binding was supplied",
            ContinuityQualification::NotPreserved);
    }
    if let Some(consent) = &record.consent {
        validate_label(&consent.subject_id, SemanticPlane::Consent, key, report, "consent subject");
        validate_label(&consent.consent_id, SemanticPlane::Consent, key, report, "consent ID");
        validate_evidence(&consent.evidence, SemanticPlane::Consent, key, report, "consent evidence");
        match consent.status {
            ConsentStatus::Active => {}
            ConsentStatus::Revoked => push_finding(report, SemanticPlane::Consent,
                SemanticFindingCode::ConsentNotActive, Some(key.clone()), "consent is revoked",
                ContinuityQualification::NotPreserved),
            ConsentStatus::Unknown => push_finding(report, SemanticPlane::Consent,
                SemanticFindingCode::ConsentUnknown, Some(key.clone()), "consent status is unresolved",
                ContinuityQualification::Inconclusive),
        }
        if !consent.scope.contains(&key.action) {
            push_finding(report, SemanticPlane::Consent, SemanticFindingCode::ConsentScopeMissing,
                Some(key.clone()), "consent scope does not cover the action",
                ContinuityQualification::NotPreserved);
        }
    }

    validate_evidence(&record.effect_inventory_receipt, SemanticPlane::Effects, key, report,
        "effect inventory receipt");
    if !record.effect_inventory_complete {
        push_finding(report, SemanticPlane::Effects, SemanticFindingCode::EffectInventoryIncomplete,
            Some(key.clone()), "effect inventory is not declared complete by its independent observer",
            ContinuityQualification::Inconclusive);
    }
    let mut effect_ids = BTreeSet::new();
    for effect in &record.effects {
        validate_label(&effect.effect_id, SemanticPlane::Effects, key, report, "effect ID");
        validate_label(&effect.class, SemanticPlane::Effects, key, report, "effect class");
        validate_label(&effect.target, SemanticPlane::Effects, key, report, "effect target");
        if !effect_ids.insert(effect.effect_id.clone()) {
            push_finding(report, SemanticPlane::Effects, SemanticFindingCode::EffectInventoryDuplicate,
                Some(key.clone()), "effect inventory contains duplicate effect IDs",
                ContinuityQualification::NotPreserved);
        }
        validate_evidence(&effect.evidence, SemanticPlane::Effects, key, report, "effect evidence");
        if effect.reversibility == EffectReversibility::Unknown {
            push_finding(report, SemanticPlane::Effects, SemanticFindingCode::UnknownEffectReversibility,
                Some(key.clone()), "effect reversibility is unknown", ContinuityQualification::Inconclusive);
        }
    }

    if policy.provenance_required_for_actions.contains(&key.action) && record.lineage.is_empty() {
        push_finding(report, SemanticPlane::Provenance, SemanticFindingCode::MissingRequiredLineage,
            Some(key.clone()), "this action requires a producer/consumer lineage chain",
            ContinuityQualification::NotPreserved);
    }
    for (index, step) in record.lineage.iter().enumerate() {
        if step.producer_id.trim().is_empty() || step.consumer_id.trim().is_empty() {
            push_finding(report, SemanticPlane::Provenance, SemanticFindingCode::InvalidLineageIdentity,
                Some(key.clone()), "lineage producer and consumer IDs must be non-empty",
                ContinuityQualification::NotPreserved);
        }
        validate_evidence(&step.producer_receipt, SemanticPlane::Provenance, key, report,
            "lineage producer receipt");
        validate_evidence(&step.consumer_receipt, SemanticPlane::Provenance, key, report,
            "lineage consumer receipt");
        if index > 0 && record.lineage[index - 1].output_digest != step.input_digest {
            push_finding(report, SemanticPlane::Provenance, SemanticFindingCode::BrokenLineageChain,
                Some(key.clone()), "adjacent lineage links do not share the same intermediate artifact digest",
                ContinuityQualification::NotPreserved);
        }
    }

    for (map_key, dependency) in &record.substrate {
        validate_label(map_key, SemanticPlane::Substrate, key, report, "substrate dependency key");
        if map_key != &dependency.dependency_id {
            push_finding(report, SemanticPlane::Substrate, SemanticFindingCode::SubstrateDependencyKeyMismatch,
                Some(key.clone()), "substrate map key differs from the dependency ID",
                ContinuityQualification::NotPreserved);
        }
        validate_evidence(&dependency.observation, SemanticPlane::Substrate, key, report,
            "substrate observation");
        if dependency.status == DependencyStatus::Unknown {
            push_finding(report, SemanticPlane::Substrate, SemanticFindingCode::MissingSubstrateObservation,
                Some(key.clone()), "substrate dependency status is unknown",
                ContinuityQualification::Inconclusive);
        }
    }
}

fn effect_signature(effect: &EffectRecord) -> (String, String, String, EffectReversibility) {
    (effect.effect_id.clone(), effect.class.clone(), effect.target.clone(), effect.reversibility)
}

fn compare_authority(left: &AuthorityBinding, right: &AuthorityBinding,
    key: &SemanticTransitionKey, report: &mut SemanticContinuityReport) {
    if left.principal_id != right.principal_id || left.trust_root_digest != right.trust_root_digest
        || left.scopes != right.scopes || left.target_scope != right.target_scope {
        push_finding(report, SemanticPlane::Authority, SemanticFindingCode::AuthorityChanged,
            Some(key.clone()), "principal, trust root, action scopes or target scopes changed without a separately qualified migration protocol",
            ContinuityQualification::NotPreserved);
    }
    // Grant IDs may change during renewal; both grants must remain active with equal semantic scope.
}

fn compare_consent(left: &Option<ConsentBinding>, right: &Option<ConsentBinding>,
    key: &SemanticTransitionKey, report: &mut SemanticContinuityReport) {
    match (left, right) {
        (None, None) => {}
        (Some(a), Some(b)) if a.subject_id == b.subject_id && a.scope == b.scope => {
            // Consent IDs may change during renewal; both need current evidence.
        }
        (Some(_), Some(_)) | (None, Some(_)) | (Some(_), None) => push_finding(report,
            SemanticPlane::Consent, SemanticFindingCode::ConsentChanged, Some(key.clone()),
            "consent subject or scope changed, or consent was added/removed",
            ContinuityQualification::NotPreserved),
    }
}

fn compare_effects(left: &BTreeSet<EffectRecord>, right: &BTreeSet<EffectRecord>,
    policy: &SemanticContinuityPolicy, key: &SemanticTransitionKey, report: &mut SemanticContinuityReport) {
    let left_signatures: BTreeSet<_> = left.iter().map(effect_signature).collect();
    let right_signatures: BTreeSet<_> = right.iter().map(effect_signature).collect();
    for signature in left_signatures.difference(&right_signatures) {
        push_finding(report, SemanticPlane::Effects, SemanticFindingCode::BaselineEffectMissing,
            Some(key.clone()), format!("candidate omitted baseline effect {}", signature.0),
            ContinuityQualification::NotPreserved);
    }
    for signature in right_signatures.difference(&left_signatures) {
        let id = &signature.0;
        match signature.3 {
            EffectReversibility::Irreversible => push_finding(report, SemanticPlane::Effects,
                SemanticFindingCode::AdditionalIrreversibleEffect, Some(key.clone()),
                format!("candidate added irreversible effect {id}"), ContinuityQualification::NotPreserved),
            EffectReversibility::Reversible if policy.allowed_additional_reversible_effects.contains(id) => {}
            EffectReversibility::Reversible => push_finding(report, SemanticPlane::Effects,
                SemanticFindingCode::AdditionalReversibleEffectNotAllowed, Some(key.clone()),
                format!("candidate added non-allowlisted reversible effect {id}"),
                ContinuityQualification::NotPreserved),
            EffectReversibility::Unknown => push_finding(report, SemanticPlane::Effects,
                SemanticFindingCode::UnknownEffectReversibility, Some(key.clone()),
                format!("candidate added effect {id} with unknown reversibility"),
                ContinuityQualification::Inconclusive),
        }
    }
}

fn compare_substrate(left: &BTreeMap<String, SubstrateDependency>,
    right: &BTreeMap<String, SubstrateDependency>, key: &SemanticTransitionKey,
    report: &mut SemanticContinuityReport) {
    let left_ids: BTreeSet<_> = left.keys().cloned().collect();
    let right_ids: BTreeSet<_> = right.keys().cloned().collect();
    if left_ids != right_ids {
        push_finding(report, SemanticPlane::Substrate, SemanticFindingCode::SubstrateDependencySetChanged,
            Some(key.clone()), "external dependency set changed; substrate continuity requires an explicit migration protocol",
            ContinuityQualification::NotPreserved);
    }
    for id in left_ids.intersection(&right_ids) {
        if left[id].status != right[id].status {
            push_finding(report, SemanticPlane::Substrate, SemanticFindingCode::SubstrateStatusChanged,
                Some(key.clone()), format!("external dependency {id} changed observed status"),
                ContinuityQualification::NotPreserved);
        }
    }
}

fn compare_pair(left: &SemanticTransition, right: &SemanticTransition,
    policy: &SemanticContinuityPolicy, report: &mut SemanticContinuityReport) {
    let key = &left.key;
    compare_authority(&left.authority, &right.authority, key, report);
    compare_consent(&left.consent, &right.consent, key, report);
    compare_effects(&left.effects, &right.effects, policy, key, report);
    compare_substrate(&left.substrate, &right.substrate, key, report);
    // Provenance may change under valid replacement. Each chain is independently
    // checked for complete, joined, receipted lineage rather than byte equality.
}

/// Evaluate typed semantic envelopes for two transition sets.
///
/// Simulation requires every baseline key to be represented by the candidate.
/// Candidate-only transitions are permitted, independently validated and
/// reported. Bisimulation also rejects candidate-only keys. The result is
/// relative to supplied evidence dispositions; it does not authenticate them.
pub fn evaluate_semantic_continuity(baseline: &[SemanticTransition],
    candidate: &[SemanticTransition], policy: &SemanticContinuityPolicy) -> SemanticContinuityReport {
    let mut report = SemanticContinuityReport {
        qualification: ContinuityQualification::Preserved,
        plane_qualifications: [
            SemanticPlane::TransitionCoverage, SemanticPlane::Authority, SemanticPlane::Consent,
            SemanticPlane::Effects, SemanticPlane::Provenance, SemanticPlane::Substrate,
        ].into_iter().map(|plane| (plane, ContinuityQualification::Preserved)).collect(),
        compared_transition_count: 0,
        candidate_only_transition_count: 0,
        comparison_checks: 0,
        findings: Vec::new(),
    };

    if baseline.is_empty() {
        push_finding(&mut report, SemanticPlane::TransitionCoverage,
            SemanticFindingCode::EmptyBaselineTransitions, None, "baseline semantic transition set is empty",
            ContinuityQualification::NotPreserved);
    }
    if candidate.is_empty() {
        push_finding(&mut report, SemanticPlane::TransitionCoverage,
            SemanticFindingCode::EmptyCandidateTransitions, None, "candidate semantic transition set is empty",
            ContinuityQualification::NotPreserved);
    }

    let mut left = BTreeMap::new();
    for item in baseline {
        if left.insert(item.key.clone(), item).is_some() {
            push_finding(&mut report, SemanticPlane::TransitionCoverage, SemanticFindingCode::DuplicateBaselineKey,
                Some(item.key.clone()), "baseline contains duplicate semantic transition keys",
                ContinuityQualification::NotPreserved);
        }
        validate_record(item, policy, &mut report);
    }
    let mut right = BTreeMap::new();
    for item in candidate {
        if right.insert(item.key.clone(), item).is_some() {
            push_finding(&mut report, SemanticPlane::TransitionCoverage, SemanticFindingCode::DuplicateCandidateKey,
                Some(item.key.clone()), "candidate contains duplicate semantic transition keys",
                ContinuityQualification::NotPreserved);
        }
        validate_record(item, policy, &mut report);
    }

    for (key, left_transition) in &left {
        report.comparison_checks = report.comparison_checks.saturating_add(1);
        if report.comparison_checks > policy.max_transition_checks {
            push_finding(&mut report, SemanticPlane::TransitionCoverage,
                SemanticFindingCode::TransitionCheckBudgetExceeded, Some(key.clone()),
                "semantic transition comparison budget was exhausted",
                ContinuityQualification::Inconclusive);
            finalize(&mut report);
            return report;
        }
        match right.get(key) {
            Some(right_transition) => {
                report.compared_transition_count += 1;
                compare_pair(left_transition, right_transition, policy, &mut report);
            }
            None => push_finding(&mut report, SemanticPlane::TransitionCoverage,
                SemanticFindingCode::MissingCandidateTransition, Some(key.clone()),
                "candidate omitted a registered baseline transition",
                ContinuityQualification::NotPreserved),
        }
    }

    for (key, _) in &right {
        if left.contains_key(key) { continue; }
        report.candidate_only_transition_count += 1;
        if policy.mode == RelationMode::Bisimulation {
            push_finding(&mut report, SemanticPlane::TransitionCoverage,
                SemanticFindingCode::UnexpectedCandidateTransition, Some(key.clone()),
                "bisimulation does not permit candidate-only semantic transitions",
                ContinuityQualification::NotPreserved);
        }
    }

    finalize(&mut report);
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(name: &str) -> EvidenceRef {
        EvidenceRef { reference: name.to_owned(), digest: ContentDigest::of(name.as_bytes()),
            status: EvidenceStatus::Verified }
    }
    fn authority(principal: &str) -> AuthorityBinding {
        AuthorityBinding {
            principal_id: principal.to_owned(), trust_root_digest: ContentDigest::of(b"authority-root-v1"),
            grant_id: "grant-v1".to_owned(), scopes: ["refresh".to_owned()].into(),
            target_scope: ["model".to_owned()].into(), status: GrantStatus::Active,
            evidence: evidence("authority-receipt"),
        }
    }
    fn consent() -> ConsentBinding {
        ConsentBinding { subject_id: "subject-1".to_owned(), consent_id: "consent-v1".to_owned(),
            scope: ["refresh".to_owned()].into(), status: ConsentStatus::Active,
            evidence: evidence("consent-receipt") }
    }
    fn effect(id: &str, reversibility: EffectReversibility) -> EffectRecord {
        EffectRecord { effect_id: id.to_owned(), class: "state-write".to_owned(),
            target: "model-store".to_owned(), reversibility, evidence: evidence(&format!("effect-{id}")) }
    }
    fn lineage(producer: &str, consumer: &str) -> Vec<LineageStep> {
        vec![LineageStep { input_digest: ContentDigest::of(b"input"),
            output_digest: ContentDigest::of(b"output"), producer_id: producer.to_owned(),
            consumer_id: consumer.to_owned(), producer_receipt: evidence("producer-receipt"),
            consumer_receipt: evidence("consumer-receipt") }]
    }
    fn transition() -> SemanticTransition {
        SemanticTransition {
            key: SemanticTransitionKey::new("refresh", "model", "updated"),
            authority: authority("service-principal"), consent: None,
            effect_inventory_complete: true, effect_inventory_receipt: evidence("effect-inventory"),
            effects: [effect("write-model", EffectReversibility::Reversible)].into(),
            lineage: lineage("producer-v1", "consumer-v1"), substrate: BTreeMap::new(),
        }
    }

    #[test]
    fn grant_and_consent_receipt_ids_may_rotate_without_semantic_change() {
        let mut baseline = transition();
        let mut candidate = transition();
        baseline.consent = Some(consent());
        candidate.consent = Some(consent());
        baseline.consent.as_mut().unwrap().consent_id = "consent-old".into();
        candidate.consent.as_mut().unwrap().consent_id = "consent-renewed".into();
        candidate.authority.grant_id = "grant-renewed".into();
        candidate.authority.evidence = evidence("renewed-authority");
        let report = evaluate_semantic_continuity(&[baseline], &[candidate], &SemanticContinuityPolicy::default());
        assert!(report.preserves_semantic_continuity(), "{:?}", report.findings);
    }

    #[test]
    fn same_output_with_different_principal_is_rejected() {
        let baseline = transition();
        let mut candidate = transition();
        candidate.authority.principal_id = "unapproved-principal".into();
        let report = evaluate_semantic_continuity(&[baseline], &[candidate], &SemanticContinuityPolicy::default());
        assert_eq!(report.qualification, ContinuityQualification::NotPreserved);
        assert!(report.findings.iter().any(|f| f.code == SemanticFindingCode::AuthorityChanged));
    }

    #[test]
    fn revoked_consent_is_rejected_even_when_outcome_matches() {
        let mut baseline = transition();
        baseline.consent = Some(consent());
        let mut candidate = baseline.clone();
        candidate.consent.as_mut().unwrap().status = ConsentStatus::Revoked;
        let policy = SemanticContinuityPolicy { consent_required_for_actions: ["refresh".into()].into(),
            ..SemanticContinuityPolicy::default() };
        let report = evaluate_semantic_continuity(&[baseline], &[candidate], &policy);
        assert_eq!(report.qualification, ContinuityQualification::NotPreserved);
        assert!(report.findings.iter().any(|f| f.code == SemanticFindingCode::ConsentNotActive));
    }

    #[test]
    fn missing_required_consent_is_rejected() {
        let baseline = transition();
        let candidate = transition();
        let policy = SemanticContinuityPolicy { consent_required_for_actions: ["refresh".into()].into(),
            ..SemanticContinuityPolicy::default() };
        let report = evaluate_semantic_continuity(&[baseline], &[candidate], &policy);
        assert_eq!(report.qualification, ContinuityQualification::NotPreserved);
        assert!(report.findings.iter().any(|f| f.code == SemanticFindingCode::ConsentMissing));
    }

    #[test]
    fn hidden_irreversible_effect_is_rejected() {
        let baseline = transition();
        let mut candidate = transition();
        candidate.effects.insert(effect("irreversible-delete", EffectReversibility::Irreversible));
        let report = evaluate_semantic_continuity(&[baseline], &[candidate], &SemanticContinuityPolicy::default());
        assert_eq!(report.qualification, ContinuityQualification::NotPreserved);
        assert!(report.findings.iter().any(|f| f.code == SemanticFindingCode::AdditionalIrreversibleEffect));
    }

    #[test]
    fn allowlisted_reversible_effect_addition_is_admitted() {
        let baseline = transition();
        let mut candidate = transition();
        candidate.effects.insert(effect("write-audit-index", EffectReversibility::Reversible));
        let policy = SemanticContinuityPolicy {
            allowed_additional_reversible_effects: ["write-audit-index".into()].into(),
            ..SemanticContinuityPolicy::default()
        };
        let report = evaluate_semantic_continuity(&[baseline], &[candidate], &policy);
        assert!(report.preserves_semantic_continuity(), "{:?}", report.findings);
    }

    #[test]
    fn different_complete_lineage_is_accepted() {
        let baseline = transition();
        let mut candidate = transition();
        candidate.lineage = lineage("replacement-producer", "replacement-consumer");
        let report = evaluate_semantic_continuity(&[baseline], &[candidate], &SemanticContinuityPolicy::default());
        assert!(report.preserves_semantic_continuity(), "{:?}", report.findings);
    }

    #[test]
    fn broken_lineage_chain_is_rejected() {
        let mut baseline = transition();
        baseline.lineage = vec![
            LineageStep { input_digest: ContentDigest::of(b"root"),
                output_digest: ContentDigest::of(b"middle-a"), producer_id: "p1".into(),
                consumer_id: "c1".into(), producer_receipt: evidence("p1"), consumer_receipt: evidence("c1") },
            LineageStep { input_digest: ContentDigest::of(b"middle-b"),
                output_digest: ContentDigest::of(b"terminal"), producer_id: "p2".into(),
                consumer_id: "c2".into(), producer_receipt: evidence("p2"), consumer_receipt: evidence("c2") },
        ];
        let candidate = baseline.clone();
        let report = evaluate_semantic_continuity(&[baseline], &[candidate], &SemanticContinuityPolicy::default());
        assert_eq!(report.qualification, ContinuityQualification::NotPreserved);
        assert!(report.findings.iter().any(|f| f.code == SemanticFindingCode::BrokenLineageChain));
    }

    #[test]
    fn unavailable_receipt_is_inconclusive_not_a_pass() {
        let baseline = transition();
        let mut candidate = transition();
        candidate.authority.evidence.status = EvidenceStatus::Unavailable;
        let report = evaluate_semantic_continuity(&[baseline], &[candidate], &SemanticContinuityPolicy::default());
        assert_eq!(report.qualification, ContinuityQualification::Inconclusive);
        assert!(report.findings.iter().any(|f| f.code == SemanticFindingCode::EvidenceUnavailable));
    }

    #[test]
    fn substrate_dependency_change_fails_only_the_substrate_plane() {
        let mut baseline = transition();
        let mut candidate = transition();
        baseline.substrate.insert("object-store".into(), SubstrateDependency {
            dependency_id: "object-store".into(), status: DependencyStatus::Available,
            observation: evidence("object-store-baseline") });
        candidate.substrate.insert("alternate-store".into(), SubstrateDependency {
            dependency_id: "alternate-store".into(), status: DependencyStatus::Available,
            observation: evidence("alternate-store-candidate") });
        let report = evaluate_semantic_continuity(&[baseline], &[candidate], &SemanticContinuityPolicy::default());
        assert_eq!(report.plane_qualifications[&SemanticPlane::Substrate], ContinuityQualification::NotPreserved);
        assert_eq!(report.qualification, ContinuityQualification::NotPreserved);
    }

    #[test]
    fn simulation_allows_candidate_only_transition_but_bisimulation_rejects_it() {
        let baseline = transition();
        let candidate = transition();
        let mut extra = transition();
        extra.key = SemanticTransitionKey::new("diagnose", "model", "diagnostic");
        extra.authority.scopes.insert("diagnose".into());
        extra.authority.grant_id = "grant-diagnose".into();
        let simulation = evaluate_semantic_continuity(&[baseline.clone()], &[candidate.clone(), extra.clone()],
            &SemanticContinuityPolicy::default());
        assert!(simulation.preserves_semantic_continuity(), "{:?}", simulation.findings);
        assert_eq!(simulation.candidate_only_transition_count, 1);
        let bisimulation = evaluate_semantic_continuity(&[baseline], &[candidate, extra],
            &SemanticContinuityPolicy { mode: RelationMode::Bisimulation, ..SemanticContinuityPolicy::default() });
        assert_eq!(bisimulation.qualification, ContinuityQualification::NotPreserved);
        assert!(bisimulation.findings.iter().any(|f| f.code == SemanticFindingCode::UnexpectedCandidateTransition));
    }

    #[test]
    fn comparison_budget_exhaustion_is_inconclusive() {
        let report = evaluate_semantic_continuity(&[transition()], &[transition()],
            &SemanticContinuityPolicy { max_transition_checks: 0, ..SemanticContinuityPolicy::default() });
        assert_eq!(report.qualification, ContinuityQualification::Inconclusive);
        assert!(report.findings.iter().any(|f| f.code == SemanticFindingCode::TransitionCheckBudgetExceeded));
    }

    #[test]
    fn missing_action_authority_scope_is_rejected() {
        let mut baseline = transition();
        baseline.authority.scopes.clear();
        let candidate = baseline.clone();
        let report = evaluate_semantic_continuity(&[baseline], &[candidate], &SemanticContinuityPolicy::default());
        assert_eq!(report.qualification, ContinuityQualification::NotPreserved);
        assert!(report.findings.iter().any(|f| f.code == SemanticFindingCode::AuthorityScopeMissing));
    }
}
