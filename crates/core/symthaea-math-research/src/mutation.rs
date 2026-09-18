// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Semantic mutation auditing for frozen mathematical specifications.
//!
//! This module never mutates a `FrozenChallenge`. It preregisters derived
//! statement mutations and records **structural diagnostic reports** about them.
//! A report may state that an evaluator *reported* proof acceptance or a
//! counterexample, but this module does not authenticate that execution. Formal
//! verifier authority must be composed in a later typed layer.
//!
//! Coverage is explicit: a complete report must observe every preregistered
//! mutation. A partial report must enumerate every omitted mutation and explain
//! why it is missing. Hard cases therefore cannot silently disappear from a
//! preregistered audit.

use crate::spec::{FrozenChallenge, Sha256Digest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const MUTATION_PLAN_VERSION: &str = "symthaea.math-spec-mutation-plan.v1";
pub const MUTATION_REPORT_VERSION: &str = "symthaea.math-spec-mutation-report.v2";
const PLAN_DOMAIN: &str = "symthaea.math-spec-mutation-plan-digest.v1";
const REPORT_DOMAIN: &str = "symthaea.math-spec-mutation-report-digest.v2";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum SpecificationMutationKind {
    HypothesisRemoval,
    HypothesisWeakening,
    HypothesisStrengthening,
    QuantifierSwap,
    DomainRestriction,
    DomainExpansion,
    ConstantPerturbation,
    BoundaryInstantiation,
    DefinitionSubstitution,
    ContradictionProbe,
    VacuityProbe,
}

impl SpecificationMutationKind {
    fn tag(self) -> &'static str {
        match self {
            Self::HypothesisRemoval => "hypothesis-removal",
            Self::HypothesisWeakening => "hypothesis-weakening",
            Self::HypothesisStrengthening => "hypothesis-strengthening",
            Self::QuantifierSwap => "quantifier-swap",
            Self::DomainRestriction => "domain-restriction",
            Self::DomainExpansion => "domain-expansion",
            Self::ConstantPerturbation => "constant-perturbation",
            Self::BoundaryInstantiation => "boundary-instantiation",
            Self::DefinitionSubstitution => "definition-substitution",
            Self::ContradictionProbe => "contradiction-probe",
            Self::VacuityProbe => "vacuity-probe",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecificationMutation {
    pub mutation_id: String,
    pub challenge_sha256: Sha256Digest,
    pub original_statement_sha256: Sha256Digest,
    pub mutated_statement_sha256: Sha256Digest,
    pub mutation_recipe_sha256: Sha256Digest,
    pub kind: SpecificationMutationKind,
    /// Human-readable motivation only. Authority comes from the hashes above.
    pub rationale: String,
}

impl SpecificationMutation {
    pub fn for_challenge(
        challenge: &FrozenChallenge,
        mutation_id: impl Into<String>,
        mutated_statement_sha256: Sha256Digest,
        mutation_recipe_sha256: Sha256Digest,
        kind: SpecificationMutationKind,
        rationale: impl Into<String>,
    ) -> Self {
        Self {
            mutation_id: mutation_id.into(),
            challenge_sha256: challenge.challenge_sha256().clone(),
            original_statement_sha256: challenge
                .specification()
                .lean_statement_sha256
                .clone(),
            mutated_statement_sha256,
            mutation_recipe_sha256,
            kind,
            rationale: rationale.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MutationPlanIssue {
    EmptyPlanId,
    EmptyMutationId { index: usize },
    EmptyRationale { mutation_id: String },
    ChallengeMismatch { mutation_id: String },
    OriginalStatementMismatch { mutation_id: String },
    MutationDidNotChangeStatement { mutation_id: String },
    DuplicateMutationId { mutation_id: String },
    DuplicateMutatedStatement { mutation_id: String },
    NoMutations,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenMutationPlan {
    schema_version: String,
    plan_id: String,
    challenge_sha256: Sha256Digest,
    mutations: Vec<SpecificationMutation>,
    plan_sha256: Sha256Digest,
}

impl FrozenMutationPlan {
    pub fn freeze(
        challenge: &FrozenChallenge,
        plan_id: impl Into<String>,
        mutations: Vec<SpecificationMutation>,
    ) -> Result<Self, Vec<MutationPlanIssue>> {
        let plan_id = plan_id.into();
        let mut issues = Vec::new();
        if plan_id.trim().is_empty() {
            issues.push(MutationPlanIssue::EmptyPlanId);
        }
        if mutations.is_empty() {
            issues.push(MutationPlanIssue::NoMutations);
        }

        let mut ids = BTreeSet::new();
        let mut statements = BTreeSet::new();
        for (index, mutation) in mutations.iter().enumerate() {
            if mutation.mutation_id.trim().is_empty() {
                issues.push(MutationPlanIssue::EmptyMutationId { index });
            }
            if mutation.rationale.trim().is_empty() {
                issues.push(MutationPlanIssue::EmptyRationale {
                    mutation_id: mutation.mutation_id.clone(),
                });
            }
            if mutation.challenge_sha256 != *challenge.challenge_sha256() {
                issues.push(MutationPlanIssue::ChallengeMismatch {
                    mutation_id: mutation.mutation_id.clone(),
                });
            }
            if mutation.original_statement_sha256
                != challenge.specification().lean_statement_sha256
            {
                issues.push(MutationPlanIssue::OriginalStatementMismatch {
                    mutation_id: mutation.mutation_id.clone(),
                });
            }
            if mutation.mutated_statement_sha256 == mutation.original_statement_sha256 {
                issues.push(MutationPlanIssue::MutationDidNotChangeStatement {
                    mutation_id: mutation.mutation_id.clone(),
                });
            }
            if !ids.insert(mutation.mutation_id.clone()) {
                issues.push(MutationPlanIssue::DuplicateMutationId {
                    mutation_id: mutation.mutation_id.clone(),
                });
            }
            if !statements.insert(mutation.mutated_statement_sha256.clone()) {
                issues.push(MutationPlanIssue::DuplicateMutatedStatement {
                    mutation_id: mutation.mutation_id.clone(),
                });
            }
        }
        if !issues.is_empty() {
            return Err(issues);
        }

        let mut mutations = mutations;
        mutations.sort_by(|left, right| left.mutation_id.cmp(&right.mutation_id));
        let plan_sha256 = compute_plan_digest(&plan_id, challenge.challenge_sha256(), &mutations);
        Ok(Self {
            schema_version: MUTATION_PLAN_VERSION.into(),
            plan_id,
            challenge_sha256: challenge.challenge_sha256().clone(),
            mutations,
            plan_sha256,
        })
    }

    pub fn plan_id(&self) -> &str {
        &self.plan_id
    }
    pub fn challenge_sha256(&self) -> &Sha256Digest {
        &self.challenge_sha256
    }
    pub fn mutations(&self) -> &[SpecificationMutation] {
        &self.mutations
    }
    pub fn plan_sha256(&self) -> &Sha256Digest {
        &self.plan_sha256
    }
    pub fn mutation(&self, mutation_id: &str) -> Option<&SpecificationMutation> {
        self.mutations
            .binary_search_by(|mutation| mutation.mutation_id.as_str().cmp(mutation_id))
            .ok()
            .map(|index| &self.mutations[index])
    }
}

/// Structurally reported evaluator outcome.
///
/// The `Reported*` prefix is intentional: possession of hashes identifying an
/// evaluator and an artifact does not prove that the evaluator really executed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum MutationObservationKind {
    ReportedFormalProofAccepted,
    ReportedCounterexampleAccepted,
    Inconclusive,
    Timeout,
    InvalidFormalization,
}

impl MutationObservationKind {
    fn tag(self) -> &'static str {
        match self {
            Self::ReportedFormalProofAccepted => "reported-formal-proof-accepted",
            Self::ReportedCounterexampleAccepted => "reported-counterexample-accepted",
            Self::Inconclusive => "inconclusive",
            Self::Timeout => "timeout",
            Self::InvalidFormalization => "invalid-formalization",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutationObservation {
    pub mutation_id: String,
    pub mutated_statement_sha256: Sha256Digest,
    pub observation: MutationObservationKind,
    /// Exact external proof/counterexample/log artifact claimed by the report.
    pub evidence_artifact_sha256: Sha256Digest,
    /// Declared evaluator/verifier process identity. Authentication is external.
    pub evaluator_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum MutationReportCoverage {
    Complete,
    Partial {
        omitted_mutation_ids: Vec<String>,
        rationale: String,
    },
}

impl MutationReportCoverage {
    fn tag(&self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Partial { .. } => "partial",
        }
    }

    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Complete)
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize,
)]
pub enum SemanticAuditSignalKind {
    PotentiallyRedundantHypothesis,
    PotentiallyInconsistentAssumptions,
    PotentialVacuity,
    BoundarySensitivity,
    DomainSensitivity,
    QuantifierSensitivity,
    ConstantSensitivity,
    DefinitionSensitivity,
}

impl SemanticAuditSignalKind {
    fn tag(self) -> &'static str {
        match self {
            Self::PotentiallyRedundantHypothesis => "potentially-redundant-hypothesis",
            Self::PotentiallyInconsistentAssumptions => "potentially-inconsistent-assumptions",
            Self::PotentialVacuity => "potential-vacuity",
            Self::BoundarySensitivity => "boundary-sensitivity",
            Self::DomainSensitivity => "domain-sensitivity",
            Self::QuantifierSensitivity => "quantifier-sensitivity",
            Self::ConstantSensitivity => "constant-sensitivity",
            Self::DefinitionSensitivity => "definition-sensitivity",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SemanticAuditSignal {
    pub mutation_id: String,
    pub kind: SemanticAuditSignalKind,
    /// Diagnostic source remains structural/reported, never proof authority.
    pub source_observation: MutationObservationKind,
    pub evidence_artifact_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MutationReportIssue {
    EmptyReportId,
    NoObservations,
    DuplicateObservation { mutation_id: String },
    UnknownMutation { mutation_id: String },
    MutatedStatementMismatch { mutation_id: String },
    MissingObservation { mutation_id: String },
    EmptyPartialRationale,
    DuplicateOmittedMutation { mutation_id: String },
    UnknownOmittedMutation { mutation_id: String },
    OmittedMutationWasObserved { mutation_id: String },
    MissingOmissionDeclaration { mutation_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SpecificationAuditReport {
    schema_version: String,
    report_id: String,
    challenge_sha256: Sha256Digest,
    plan_sha256: Sha256Digest,
    coverage: MutationReportCoverage,
    observations: Vec<MutationObservation>,
    signals: Vec<SemanticAuditSignal>,
    report_sha256: Sha256Digest,
}

impl SpecificationAuditReport {
    pub fn build_complete(
        plan: &FrozenMutationPlan,
        report_id: impl Into<String>,
        observations: Vec<MutationObservation>,
    ) -> Result<Self, Vec<MutationReportIssue>> {
        Self::build(plan, report_id, observations, MutationReportCoverage::Complete)
    }

    pub fn build_partial(
        plan: &FrozenMutationPlan,
        report_id: impl Into<String>,
        observations: Vec<MutationObservation>,
        omitted_mutation_ids: Vec<String>,
        rationale: impl Into<String>,
    ) -> Result<Self, Vec<MutationReportIssue>> {
        Self::build(
            plan,
            report_id,
            observations,
            MutationReportCoverage::Partial {
                omitted_mutation_ids,
                rationale: rationale.into(),
            },
        )
    }

    fn build(
        plan: &FrozenMutationPlan,
        report_id: impl Into<String>,
        observations: Vec<MutationObservation>,
        coverage: MutationReportCoverage,
    ) -> Result<Self, Vec<MutationReportIssue>> {
        let report_id = report_id.into();
        let mut issues = Vec::new();
        if report_id.trim().is_empty() {
            issues.push(MutationReportIssue::EmptyReportId);
        }
        if observations.is_empty() {
            issues.push(MutationReportIssue::NoObservations);
        }

        let plan_ids: BTreeSet<_> = plan
            .mutations()
            .iter()
            .map(|mutation| mutation.mutation_id.clone())
            .collect();
        let mut observed_ids = BTreeSet::new();
        for observation in &observations {
            if !observed_ids.insert(observation.mutation_id.clone()) {
                issues.push(MutationReportIssue::DuplicateObservation {
                    mutation_id: observation.mutation_id.clone(),
                });
                continue;
            }
            let Some(mutation) = plan.mutation(&observation.mutation_id) else {
                issues.push(MutationReportIssue::UnknownMutation {
                    mutation_id: observation.mutation_id.clone(),
                });
                continue;
            };
            if observation.mutated_statement_sha256 != mutation.mutated_statement_sha256 {
                issues.push(MutationReportIssue::MutatedStatementMismatch {
                    mutation_id: observation.mutation_id.clone(),
                });
            }
        }

        let missing: BTreeSet<_> = plan_ids.difference(&observed_ids).cloned().collect();
        let normalized_coverage = match coverage {
            MutationReportCoverage::Complete => {
                for mutation_id in &missing {
                    issues.push(MutationReportIssue::MissingObservation {
                        mutation_id: mutation_id.clone(),
                    });
                }
                MutationReportCoverage::Complete
            }
            MutationReportCoverage::Partial {
                omitted_mutation_ids,
                rationale,
            } => {
                if rationale.trim().is_empty() {
                    issues.push(MutationReportIssue::EmptyPartialRationale);
                }
                let mut omitted = BTreeSet::new();
                for mutation_id in omitted_mutation_ids {
                    if !omitted.insert(mutation_id.clone()) {
                        issues.push(MutationReportIssue::DuplicateOmittedMutation { mutation_id });
                        continue;
                    }
                    if !plan_ids.contains(&mutation_id) {
                        issues.push(MutationReportIssue::UnknownOmittedMutation { mutation_id });
                    } else if observed_ids.contains(&mutation_id) {
                        issues.push(MutationReportIssue::OmittedMutationWasObserved { mutation_id });
                    }
                }
                for mutation_id in missing.difference(&omitted) {
                    issues.push(MutationReportIssue::MissingOmissionDeclaration {
                        mutation_id: mutation_id.clone(),
                    });
                }
                for mutation_id in omitted.difference(&missing) {
                    if plan_ids.contains(mutation_id) && !observed_ids.contains(mutation_id) {
                        // This branch is normally unreachable because such IDs are in `missing`,
                        // but keep the normalization explicit for future schema changes.
                        issues.push(MutationReportIssue::MissingOmissionDeclaration {
                            mutation_id: mutation_id.clone(),
                        });
                    }
                }
                MutationReportCoverage::Partial {
                    omitted_mutation_ids: omitted.into_iter().collect(),
                    rationale: rationale.trim().to_string(),
                }
            }
        };

        if !issues.is_empty() {
            return Err(issues);
        }

        let mut observations = observations;
        observations.sort_by(|left, right| left.mutation_id.cmp(&right.mutation_id));
        let signals = derive_signals(plan, &observations);
        let report_sha256 = compute_report_digest(
            &report_id,
            plan.challenge_sha256(),
            plan.plan_sha256(),
            &normalized_coverage,
            &observations,
            &signals,
        );

        Ok(Self {
            schema_version: MUTATION_REPORT_VERSION.into(),
            report_id,
            challenge_sha256: plan.challenge_sha256().clone(),
            plan_sha256: plan.plan_sha256().clone(),
            coverage: normalized_coverage,
            observations,
            signals,
            report_sha256,
        })
    }

    pub fn signals(&self) -> &[SemanticAuditSignal] {
        &self.signals
    }
    pub fn observations(&self) -> &[MutationObservation] {
        &self.observations
    }
    pub fn coverage(&self) -> &MutationReportCoverage {
        &self.coverage
    }
    pub fn is_complete(&self) -> bool {
        self.coverage.is_complete()
    }
    pub fn report_sha256(&self) -> &Sha256Digest {
        &self.report_sha256
    }
    pub fn challenge_sha256(&self) -> &Sha256Digest {
        &self.challenge_sha256
    }
}

fn derive_signals(
    plan: &FrozenMutationPlan,
    observations: &[MutationObservation],
) -> Vec<SemanticAuditSignal> {
    let by_id: BTreeMap<_, _> = plan
        .mutations()
        .iter()
        .map(|mutation| (mutation.mutation_id.as_str(), mutation))
        .collect();
    let mut signals = Vec::new();

    for observation in observations {
        let mutation = by_id[observation.mutation_id.as_str()];
        let signal = match (mutation.kind, observation.observation) {
            (
                SpecificationMutationKind::HypothesisRemoval,
                MutationObservationKind::ReportedFormalProofAccepted,
            ) => Some(SemanticAuditSignalKind::PotentiallyRedundantHypothesis),
            (
                SpecificationMutationKind::ContradictionProbe,
                MutationObservationKind::ReportedFormalProofAccepted,
            ) => Some(SemanticAuditSignalKind::PotentiallyInconsistentAssumptions),
            (
                SpecificationMutationKind::VacuityProbe,
                MutationObservationKind::ReportedFormalProofAccepted,
            ) => Some(SemanticAuditSignalKind::PotentialVacuity),
            (
                SpecificationMutationKind::BoundaryInstantiation,
                MutationObservationKind::ReportedCounterexampleAccepted,
            ) => Some(SemanticAuditSignalKind::BoundarySensitivity),
            (
                SpecificationMutationKind::DomainRestriction
                | SpecificationMutationKind::DomainExpansion,
                MutationObservationKind::ReportedCounterexampleAccepted,
            ) => Some(SemanticAuditSignalKind::DomainSensitivity),
            (
                SpecificationMutationKind::QuantifierSwap,
                MutationObservationKind::ReportedCounterexampleAccepted,
            ) => Some(SemanticAuditSignalKind::QuantifierSensitivity),
            (
                SpecificationMutationKind::ConstantPerturbation,
                MutationObservationKind::ReportedCounterexampleAccepted,
            ) => Some(SemanticAuditSignalKind::ConstantSensitivity),
            (
                SpecificationMutationKind::DefinitionSubstitution,
                MutationObservationKind::ReportedCounterexampleAccepted,
            ) => Some(SemanticAuditSignalKind::DefinitionSensitivity),
            _ => None,
        };
        if let Some(kind) = signal {
            signals.push(SemanticAuditSignal {
                mutation_id: observation.mutation_id.clone(),
                kind,
                source_observation: observation.observation,
                evidence_artifact_sha256: observation.evidence_artifact_sha256.clone(),
            });
        }
    }

    signals.sort_by(|left, right| {
        left.mutation_id
            .cmp(&right.mutation_id)
            .then_with(|| left.kind.cmp(&right.kind))
    });
    signals
}

fn compute_plan_digest(
    plan_id: &str,
    challenge_sha256: &Sha256Digest,
    mutations: &[SpecificationMutation],
) -> Sha256Digest {
    let mut bytes = Vec::new();
    put_text(&mut bytes, PLAN_DOMAIN);
    put_text(&mut bytes, MUTATION_PLAN_VERSION);
    put_text(&mut bytes, plan_id);
    put_text(&mut bytes, challenge_sha256.as_str());
    for mutation in mutations {
        put_text(&mut bytes, &mutation.mutation_id);
        put_text(&mut bytes, mutation.challenge_sha256.as_str());
        put_text(&mut bytes, mutation.original_statement_sha256.as_str());
        put_text(&mut bytes, mutation.mutated_statement_sha256.as_str());
        put_text(&mut bytes, mutation.mutation_recipe_sha256.as_str());
        put_text(&mut bytes, mutation.kind.tag());
        put_text(&mut bytes, &mutation.rationale);
    }
    Sha256Digest::of_bytes(&bytes)
}

fn compute_report_digest(
    report_id: &str,
    challenge_sha256: &Sha256Digest,
    plan_sha256: &Sha256Digest,
    coverage: &MutationReportCoverage,
    observations: &[MutationObservation],
    signals: &[SemanticAuditSignal],
) -> Sha256Digest {
    let mut bytes = Vec::new();
    put_text(&mut bytes, REPORT_DOMAIN);
    put_text(&mut bytes, MUTATION_REPORT_VERSION);
    put_text(&mut bytes, report_id);
    put_text(&mut bytes, challenge_sha256.as_str());
    put_text(&mut bytes, plan_sha256.as_str());
    put_text(&mut bytes, coverage.tag());
    if let MutationReportCoverage::Partial {
        omitted_mutation_ids,
        rationale,
    } = coverage
    {
        for mutation_id in omitted_mutation_ids {
            put_text(&mut bytes, mutation_id);
        }
        put_text(&mut bytes, rationale);
    }
    for observation in observations {
        put_text(&mut bytes, &observation.mutation_id);
        put_text(&mut bytes, observation.mutated_statement_sha256.as_str());
        put_text(&mut bytes, observation.observation.tag());
        put_text(&mut bytes, observation.evidence_artifact_sha256.as_str());
        put_text(&mut bytes, observation.evaluator_sha256.as_str());
    }
    for signal in signals {
        put_text(&mut bytes, &signal.mutation_id);
        put_text(&mut bytes, signal.kind.tag());
        put_text(&mut bytes, signal.source_observation.tag());
        put_text(&mut bytes, signal.evidence_artifact_sha256.as_str());
    }
    Sha256Digest::of_bytes(&bytes)
}

fn put_text(output: &mut Vec<u8>, value: &str) {
    output.extend_from_slice(&(value.len() as u64).to_be_bytes());
    output.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{
        FormalizationReview, FormalizationReviewKind, MathematicalSpecification,
        SourceReference, SpecificationStatus,
    };

    fn digest(label: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(label.as_bytes())
    }

    fn challenge() -> FrozenChallenge {
        let mut spec = MathematicalSpecification::new(
            "AUDIT-TEST",
            SourceReference {
                locator: "https://example.invalid/problem".into(),
                revision: "v1".into(),
                retrieved_at_utc: "2026-09-18T00:00:00Z".into(),
            },
            digest("source"),
            digest("lean"),
            digest("defs"),
            "lean-test",
            "mathlib-test",
        );
        let lean = spec.lean_statement_sha256.clone();
        let defs = spec.definitions_sha256.clone();
        for (reviewer, domain, kind) in [
            ("a", "lineage-a", FormalizationReviewKind::DefinitionsAudit),
            ("a", "lineage-a", FormalizationReviewKind::SemanticReview),
            (
                "b",
                "lineage-b",
                FormalizationReviewKind::IndependentFormalization,
            ),
        ] {
            spec.add_review(FormalizationReview {
                reviewer_id: reviewer.into(),
                independence_domain_sha256: digest(domain),
                method_sha256: digest(&format!("method:{reviewer}:{domain}:{kind:?}")),
                kind,
                lean_statement_sha256: lean.clone(),
                definitions_sha256: defs.clone(),
                notes: "reviewed".into(),
            });
        }
        spec.advance_to(SpecificationStatus::FrozenChallenge).unwrap();
        spec.freeze().unwrap()
    }

    fn mutation(
        challenge: &FrozenChallenge,
        id: &str,
        kind: SpecificationMutationKind,
    ) -> SpecificationMutation {
        SpecificationMutation::for_challenge(
            challenge,
            id,
            digest(&format!("mutated:{id}")),
            digest(&format!("recipe:{id}")),
            kind,
            format!("diagnostic mutation {id}"),
        )
    }

    fn observation(
        mutation: &SpecificationMutation,
        kind: MutationObservationKind,
    ) -> MutationObservation {
        MutationObservation {
            mutation_id: mutation.mutation_id.clone(),
            mutated_statement_sha256: mutation.mutated_statement_sha256.clone(),
            observation: kind,
            evidence_artifact_sha256: digest(&format!("artifact:{}", mutation.mutation_id)),
            evaluator_sha256: digest("declared-evaluator"),
        }
    }

    #[test]
    fn plan_identity_is_order_independent() {
        let challenge = challenge();
        let a = mutation(&challenge, "a", SpecificationMutationKind::HypothesisRemoval);
        let b = mutation(&challenge, "b", SpecificationMutationKind::VacuityProbe);
        let left = FrozenMutationPlan::freeze(&challenge, "plan", vec![b.clone(), a.clone()]).unwrap();
        let right = FrozenMutationPlan::freeze(&challenge, "plan", vec![a, b]).unwrap();
        assert_eq!(left.plan_sha256(), right.plan_sha256());
    }

    #[test]
    fn complete_report_cannot_silently_omit_preregistered_mutation() {
        let challenge = challenge();
        let a = mutation(&challenge, "a", SpecificationMutationKind::HypothesisRemoval);
        let b = mutation(&challenge, "b", SpecificationMutationKind::VacuityProbe);
        let plan = FrozenMutationPlan::freeze(&challenge, "plan", vec![a.clone(), b]).unwrap();
        let issues = SpecificationAuditReport::build_complete(
            &plan,
            "report",
            vec![observation(&a, MutationObservationKind::Inconclusive)],
        )
        .expect_err("complete report must cover every preregistered mutation");
        assert!(issues.iter().any(|issue| matches!(
            issue,
            MutationReportIssue::MissingObservation { mutation_id } if mutation_id == "b"
        )));
    }

    #[test]
    fn partial_report_must_declare_every_omission_with_reason() {
        let challenge = challenge();
        let a = mutation(&challenge, "a", SpecificationMutationKind::HypothesisRemoval);
        let b = mutation(&challenge, "b", SpecificationMutationKind::VacuityProbe);
        let plan = FrozenMutationPlan::freeze(&challenge, "plan", vec![a.clone(), b]).unwrap();

        let issues = SpecificationAuditReport::build_partial(
            &plan,
            "report",
            vec![observation(&a, MutationObservationKind::Inconclusive)],
            vec![],
            "runner budget exhausted",
        )
        .expect_err("missing mutation must be explicitly declared omitted");
        assert!(issues.iter().any(|issue| matches!(
            issue,
            MutationReportIssue::MissingOmissionDeclaration { mutation_id } if mutation_id == "b"
        )));

        let report = SpecificationAuditReport::build_partial(
            &plan,
            "report",
            vec![observation(&a, MutationObservationKind::Inconclusive)],
            vec!["b".into()],
            "runner budget exhausted",
        )
        .unwrap();
        assert!(!report.is_complete());
    }

    #[test]
    fn observation_statement_must_match_preregistered_subject() {
        let challenge = challenge();
        let a = mutation(&challenge, "a", SpecificationMutationKind::HypothesisRemoval);
        let plan = FrozenMutationPlan::freeze(&challenge, "plan", vec![a.clone()]).unwrap();
        let mut bad = observation(&a, MutationObservationKind::Inconclusive);
        bad.mutated_statement_sha256 = digest("wrong");
        let issues = SpecificationAuditReport::build_complete(&plan, "report", vec![bad])
            .expect_err("wrong statement must fail");
        assert!(issues.iter().any(|issue| matches!(
            issue,
            MutationReportIssue::MutatedStatementMismatch { mutation_id } if mutation_id == "a"
        )));
    }

    #[test]
    fn reported_proof_acceptance_is_only_a_potential_signal() {
        let challenge = challenge();
        let original = challenge.challenge_sha256().clone();
        let a = mutation(&challenge, "remove-h0", SpecificationMutationKind::HypothesisRemoval);
        let plan = FrozenMutationPlan::freeze(&challenge, "plan", vec![a.clone()]).unwrap();
        let report = SpecificationAuditReport::build_complete(
            &plan,
            "report",
            vec![observation(
                &a,
                MutationObservationKind::ReportedFormalProofAccepted,
            )],
        )
        .unwrap();
        assert_eq!(report.signals().len(), 1);
        assert_eq!(
            report.signals()[0].kind,
            SemanticAuditSignalKind::PotentiallyRedundantHypothesis
        );
        assert_eq!(
            report.signals()[0].source_observation,
            MutationObservationKind::ReportedFormalProofAccepted
        );
        assert_eq!(challenge.challenge_sha256(), &original);
    }

    #[test]
    fn complete_and_partial_coverage_have_different_report_identity() {
        let challenge = challenge();
        let a = mutation(&challenge, "a", SpecificationMutationKind::HypothesisRemoval);
        let b = mutation(&challenge, "b", SpecificationMutationKind::VacuityProbe);
        let plan = FrozenMutationPlan::freeze(&challenge, "plan", vec![a.clone(), b.clone()]).unwrap();
        let complete = SpecificationAuditReport::build_complete(
            &plan,
            "report",
            vec![
                observation(&a, MutationObservationKind::Inconclusive),
                observation(&b, MutationObservationKind::Timeout),
            ],
        )
        .unwrap();
        let partial = SpecificationAuditReport::build_partial(
            &plan,
            "report",
            vec![observation(&a, MutationObservationKind::Inconclusive)],
            vec!["b".into()],
            "timeout result withheld for independent rerun",
        )
        .unwrap();
        assert_ne!(complete.report_sha256(), partial.report_sha256());
    }
}
