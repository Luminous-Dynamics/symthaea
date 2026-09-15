// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! ASSURE-003: deterministic non-scalar resolution over admitted campaign evidence.
//!
//! This reference resolver deliberately consumes ASSURE-002 only through its
//! public canonical interfaces. It does not parse arbitrary evidence payloads,
//! run evaluators, assign confidence scores, or create deployment authority.
//!
//! Governing theorem:
//!
//! ```text
//! admitted evidence
//!     != satisfied criterion
//!     != scalar confidence
//!     != deployment authority
//!
//! more evidence
//!     != stronger support
//! ```

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_assurance_campaign::{
    CampaignEvidenceAdmissionV1, CampaignPlanV1, TerminalRegistrationInViewV1,
};
use symthaea_assurance_core::{
    Claim, DigestSha256, EvidenceArtifact, QualificationOutcome, QualificationResult,
    ReproductionStatus, StableId, SupportTier,
};
use symthaea_assurance_semantics::SemanticCommitmentV1;
use symthaea_assurance_subject::AiSubjectManifest;

pub const ASSURE_RESOLVER_SCHEMA: &str = "symthaea.assurance.resolver.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PredicateRoleV1 {
    SupportCriterion,
    Control,
    FailureCondition,
    ContradictionCondition,
    InconclusiveCondition,
}

impl PredicateRoleV1 {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::SupportCriterion => "support-criterion",
            Self::Control => "control",
            Self::FailureCondition => "failure-condition",
            Self::ContradictionCondition => "contradiction-condition",
            Self::InconclusiveCondition => "inconclusive-condition",
        }
    }

    fn campaign_wire_prefix(self) -> &'static str {
        self.canonical_name()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PredicateDispositionV1 {
    Satisfied,
    NotSatisfied,
    Inconclusive,
}

impl PredicateDispositionV1 {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::Satisfied => "satisfied",
            Self::NotSatisfied => "not-satisfied",
            Self::Inconclusive => "inconclusive",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredicateObservationV1 {
    plan_digest: DigestSha256,
    evidence_digest: DigestSha256,
    role: PredicateRoleV1,
    semantic: SemanticCommitmentV1,
    disposition: PredicateDispositionV1,
}

impl PredicateObservationV1 {
    pub fn new(
        plan_digest: DigestSha256,
        evidence_digest: DigestSha256,
        role: PredicateRoleV1,
        semantic: SemanticCommitmentV1,
        disposition: PredicateDispositionV1,
    ) -> Self {
        Self {
            plan_digest,
            evidence_digest,
            role,
            semantic,
            disposition,
        }
    }

    pub fn digest(&self) -> DigestSha256 {
        digest_canonical(&self.canonical_bytes())
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from("symthaea-assurance-predicate-observation-v1\n");
        field(&mut out, "schema", ASSURE_RESOLVER_SCHEMA);
        field(&mut out, "plan", self.plan_digest.as_str());
        field(&mut out, "evidence", self.evidence_digest.as_str());
        field(&mut out, "role", self.role.canonical_name());
        field(&mut out, "semantic-id", self.semantic.semantic_id().as_str());
        field(
            &mut out,
            "semantic-commitment",
            self.semantic.digest().as_str(),
        );
        field(&mut out, "disposition", self.disposition.canonical_name());
        out.into_bytes()
    }
}

/// Explicit public projection of the identity-bearing fields in an opaque
/// ASSURE-002 admission. The resolver independently recomputes the admission
/// digest and evidence-root transition rather than gaining private field access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionBindingV1 {
    pub registration_digest: DigestSha256,
    pub plan_digest: DigestSha256,
    pub campaign_nonce: StableId,
    pub ordinal: u64,
    pub evidence_digest: DigestSha256,
    pub commitment_ordering_digest: DigestSha256,
    pub admission_ordering_digest: DigestSha256,
    pub previous_evidence_root: DigestSha256,
    pub evidence_root: DigestSha256,
}

impl AdmissionBindingV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        registration_digest: DigestSha256,
        plan_digest: DigestSha256,
        campaign_nonce: StableId,
        ordinal: u64,
        evidence_digest: DigestSha256,
        commitment_ordering_digest: DigestSha256,
        admission_ordering_digest: DigestSha256,
        previous_evidence_root: DigestSha256,
        evidence_root: DigestSha256,
    ) -> Self {
        Self {
            registration_digest,
            plan_digest,
            campaign_nonce,
            ordinal,
            evidence_digest,
            commitment_ordering_digest,
            admission_ordering_digest,
            previous_evidence_root,
            evidence_root,
        }
    }

    pub fn digest(&self) -> DigestSha256 {
        let mut out = String::from("symthaea-assurance-campaign-evidence-admission-v1\n");
        field(&mut out, "registration", self.registration_digest.as_str());
        field(&mut out, "plan", self.plan_digest.as_str());
        field(&mut out, "campaign-nonce", self.campaign_nonce.as_str());
        field(&mut out, "ordinal", &self.ordinal.to_string());
        field(&mut out, "evidence", self.evidence_digest.as_str());
        field(
            &mut out,
            "commitment-ordering",
            self.commitment_ordering_digest.as_str(),
        );
        field(
            &mut out,
            "admission-ordering",
            self.admission_ordering_digest.as_str(),
        );
        field(
            &mut out,
            "previous-evidence-root",
            self.previous_evidence_root.as_str(),
        );
        field(&mut out, "evidence-root", self.evidence_root.as_str());
        digest_canonical(out.as_bytes())
    }

    fn recomputed_evidence_root(&self) -> DigestSha256 {
        let mut out = String::from("symthaea-assurance-evidence-root-step-v1\n");
        field(
            &mut out,
            "previous",
            self.previous_evidence_root.as_str(),
        );
        field(&mut out, "ordinal", &self.ordinal.to_string());
        field(&mut out, "evidence", self.evidence_digest.as_str());
        field(
            &mut out,
            "commitment-ordering",
            self.commitment_ordering_digest.as_str(),
        );
        field(
            &mut out,
            "admission-ordering",
            self.admission_ordering_digest.as_str(),
        );
        digest_canonical(out.as_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolverEvidenceV1 {
    pub evidence: EvidenceArtifact,
    pub admission_digest: DigestSha256,
    pub admission_evidence_root: DigestSha256,
    pub admission_binding: AdmissionBindingV1,
    pub observation: PredicateObservationV1,
}

impl ResolverEvidenceV1 {
    pub fn new(
        evidence: EvidenceArtifact,
        admission: &CampaignEvidenceAdmissionV1,
        admission_binding: AdmissionBindingV1,
        observation: PredicateObservationV1,
    ) -> Self {
        Self {
            evidence,
            admission_digest: admission.digest(),
            admission_evidence_root: admission.evidence_root().clone(),
            admission_binding,
            observation,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionOutcomeV1 {
    Supported(SupportTier),
    Refuted,
    Conflicted,
    Insufficient,
    Malformed,
}

impl ResolutionOutcomeV1 {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::Supported(_) => "supported",
            Self::Refuted => "refuted",
            Self::Conflicted => "conflicted",
            Self::Insufficient => "insufficient",
            Self::Malformed => "malformed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolutionV1 {
    plan_digest: DigestSha256,
    registration_digest: DigestSha256,
    evidence_root: DigestSha256,
    observation_digests: Vec<DigestSha256>,
    outcome: ResolutionOutcomeV1,
}

impl ResolutionV1 {
    pub fn outcome(&self) -> ResolutionOutcomeV1 {
        self.outcome
    }

    pub fn digest(&self) -> DigestSha256 {
        digest_canonical(&self.canonical_bytes())
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from("symthaea-assurance-resolution-v1\n");
        field(&mut out, "schema", ASSURE_RESOLVER_SCHEMA);
        field(&mut out, "plan", self.plan_digest.as_str());
        field(&mut out, "registration", self.registration_digest.as_str());
        field(&mut out, "evidence-root", self.evidence_root.as_str());
        field(
            &mut out,
            "observation-count",
            &self.observation_digests.len().to_string(),
        );
        for digest in &self.observation_digests {
            field(&mut out, "observation", digest.as_str());
        }
        field(&mut out, "outcome", self.outcome.canonical_name());
        if let ResolutionOutcomeV1::Supported(tier) = self.outcome {
            field(&mut out, "support-tier", support_tier_name(tier));
        } else {
            field(&mut out, "support-tier", "");
        }
        out.into_bytes()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PredicateKey {
    role: PredicateRoleV1,
    commitment: DigestSha256,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PredicateState {
    Missing,
    Satisfied,
    NotSatisfied,
    Inconclusive,
    Conflicted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeclaredPredicate {
    role: PredicateRoleV1,
    semantic_id: String,
    commitment: DigestSha256,
    tier: Option<SupportTier>,
}

pub fn resolve_v1(
    plan: &CampaignPlanV1,
    terminal: &TerminalRegistrationInViewV1,
    subject: &AiSubjectManifest,
    claim: &Claim,
    final_evidence_root: &DigestSha256,
    evidence: &[ResolverEvidenceV1],
) -> ResolutionV1 {
    let mut observation_digests: Vec<_> = evidence
        .iter()
        .map(|item| item.observation.digest())
        .collect();
    observation_digests.sort();

    let make_result = |outcome| ResolutionV1 {
        plan_digest: plan.digest(),
        registration_digest: terminal.digest(),
        evidence_root: final_evidence_root.clone(),
        observation_digests: observation_digests.clone(),
        outcome,
    };

    if !context_matches(plan, terminal, subject, claim) {
        return make_result(ResolutionOutcomeV1::Malformed);
    }

    let Some(declared) = declared_predicates(plan) else {
        return make_result(ResolutionOutcomeV1::Malformed);
    };

    if !validate_evidence_chain(
        plan,
        terminal,
        claim,
        final_evidence_root,
        evidence,
        &declared,
    ) {
        return make_result(ResolutionOutcomeV1::Malformed);
    }

    let states = aggregate_states(evidence);
    if states
        .values()
        .any(|state| *state == PredicateState::Conflicted)
    {
        return make_result(ResolutionOutcomeV1::Conflicted);
    }

    let contradiction_state = |predicate: &DeclaredPredicate| {
        state_for(&states, predicate.role, &predicate.commitment)
    };

    if declared
        .iter()
        .filter(|predicate| predicate.role == PredicateRoleV1::ContradictionCondition)
        .any(|predicate| contradiction_state(predicate) == PredicateState::Satisfied)
    {
        return make_result(ResolutionOutcomeV1::Refuted);
    }

    for predicate in declared.iter().filter(|predicate| {
        matches!(
            predicate.role,
            PredicateRoleV1::Control
                | PredicateRoleV1::FailureCondition
                | PredicateRoleV1::ContradictionCondition
                | PredicateRoleV1::InconclusiveCondition
        )
    }) {
        let state = state_for(&states, predicate.role, &predicate.commitment);
        let cleared = match predicate.role {
            PredicateRoleV1::Control => state == PredicateState::Satisfied,
            PredicateRoleV1::FailureCondition
            | PredicateRoleV1::ContradictionCondition
            | PredicateRoleV1::InconclusiveCondition => state == PredicateState::NotSatisfied,
            PredicateRoleV1::SupportCriterion => unreachable!(),
        };
        if !cleared {
            return make_result(ResolutionOutcomeV1::Insufficient);
        }
    }

    let mut explicit_tiers: BTreeSet<SupportTier> = BTreeSet::new();
    for predicate in &declared {
        if predicate.role == PredicateRoleV1::SupportCriterion {
            if let Some(tier) = predicate.tier {
                explicit_tiers.insert(tier);
            }
        }
    }

    let core_subject = match subject.as_core_subject() {
        Ok(subject) => subject,
        Err(_) => return make_result(ResolutionOutcomeV1::Malformed),
    };
    let admitted_artifacts: Vec<_> = evidence.iter().map(|item| item.evidence.clone()).collect();

    for tier in [
        SupportTier::FunctionallySupported,
        SupportTier::CausallySupported,
        SupportTier::Observed,
        SupportTier::Structural,
    ] {
        if !explicit_tiers.contains(&tier)
            || support_tier_rank(tier) > support_tier_rank(plan.maximum_support())
        {
            continue;
        }
        let criteria_satisfied = declared
            .iter()
            .filter(|predicate| {
                predicate.role == PredicateRoleV1::SupportCriterion
                    && predicate.tier.is_some_and(|declared_tier| {
                        support_tier_rank(declared_tier) <= support_tier_rank(tier)
                    })
            })
            .all(|predicate| {
                state_for(&states, predicate.role, &predicate.commitment)
                    == PredicateState::Satisfied
            });
        if !criteria_satisfied {
            continue;
        }

        if QualificationResult::validate_and_bind(
            claim,
            &core_subject,
            &plan.core_plan(),
            &admitted_artifacts,
            QualificationOutcome::Supported(tier),
            ReproductionStatus::NotClaimed,
        )
        .is_ok()
        {
            return make_result(ResolutionOutcomeV1::Supported(tier));
        }
    }

    make_result(ResolutionOutcomeV1::Insufficient)
}

fn context_matches(
    plan: &CampaignPlanV1,
    terminal: &TerminalRegistrationInViewV1,
    subject: &AiSubjectManifest,
    claim: &Claim,
) -> bool {
    let Ok(subject_core_id) = subject.core_subject_id() else {
        return false;
    };
    terminal.receipt().plan_digest() == &plan.digest()
        && terminal.receipt().campaign_nonce() == plan.campaign_nonce()
        && plan.subject_manifest_id() == &subject.manifest_id()
        && plan.subject_core_id() == &subject_core_id
        && claim.subject_id() == &subject_core_id
        && plan.claim_digest() == &claim.digest()
}

fn validate_evidence_chain(
    plan: &CampaignPlanV1,
    terminal: &TerminalRegistrationInViewV1,
    claim: &Claim,
    final_evidence_root: &DigestSha256,
    evidence: &[ResolverEvidenceV1],
    declared: &[DeclaredPredicate],
) -> bool {
    if evidence.is_empty() {
        return final_evidence_root == terminal.receipt().empty_evidence_root();
    }

    let plan_digest = plan.digest();
    let registration_digest = terminal.digest();
    let mut seen_evidence_ids = BTreeSet::new();
    let mut seen_evidence_digests = BTreeSet::new();
    let mut seen_admission_digests = BTreeSet::new();

    for item in evidence {
        let evidence_digest = item.evidence.digest();
        let binding = &item.admission_binding;
        if !seen_evidence_ids.insert(item.evidence.evidence_id().clone())
            || !seen_evidence_digests.insert(evidence_digest.clone())
            || !seen_admission_digests.insert(item.admission_digest.clone())
        {
            return false;
        }
        if item.evidence.subject_id() != plan.subject_core_id()
            || item.evidence.claim_digest() != claim.digest()
            || binding.registration_digest != registration_digest
            || binding.plan_digest != plan_digest
            || &binding.campaign_nonce != plan.campaign_nonce()
            || binding.evidence_digest != evidence_digest
            || binding.digest() != item.admission_digest
            || binding.evidence_root != item.admission_evidence_root
            || binding.recomputed_evidence_root() != binding.evidence_root
            || item.observation.plan_digest != plan_digest
            || item.observation.evidence_digest != evidence_digest
            || !observation_is_declared(&item.observation, declared)
        {
            return false;
        }
    }

    let mut ordered: Vec<_> = evidence.iter().collect();
    ordered.sort_by_key(|item| item.admission_binding.ordinal);
    let mut prior_root = terminal.receipt().empty_evidence_root().clone();
    for (index, item) in ordered.iter().enumerate() {
        let Ok(expected_ordinal) = u64::try_from(index + 1) else {
            return false;
        };
        let binding = &item.admission_binding;
        if binding.ordinal != expected_ordinal || binding.previous_evidence_root != prior_root {
            return false;
        }
        prior_root = binding.evidence_root.clone();
    }
    &prior_root == final_evidence_root
}

fn observation_is_declared(
    observation: &PredicateObservationV1,
    declared: &[DeclaredPredicate],
) -> bool {
    let commitment = observation.semantic.digest();
    declared.iter().any(|predicate| {
        predicate.role == observation.role
            && predicate.semantic_id == observation.semantic.semantic_id().as_str()
            && predicate.commitment == commitment
    })
}

fn aggregate_states(evidence: &[ResolverEvidenceV1]) -> BTreeMap<PredicateKey, PredicateState> {
    let mut dispositions = BTreeMap::<PredicateKey, BTreeSet<PredicateDispositionV1>>::new();
    for item in evidence {
        let key = PredicateKey {
            role: item.observation.role,
            commitment: item.observation.semantic.digest(),
        };
        dispositions
            .entry(key)
            .or_default()
            .insert(item.observation.disposition);
    }

    dispositions
        .into_iter()
        .map(|(key, values)| {
            let state = if values.contains(&PredicateDispositionV1::Satisfied)
                && values.contains(&PredicateDispositionV1::NotSatisfied)
            {
                PredicateState::Conflicted
            } else if values.contains(&PredicateDispositionV1::Inconclusive) {
                PredicateState::Inconclusive
            } else if values.contains(&PredicateDispositionV1::Satisfied) {
                PredicateState::Satisfied
            } else if values.contains(&PredicateDispositionV1::NotSatisfied) {
                PredicateState::NotSatisfied
            } else {
                PredicateState::Missing
            };
            (key, state)
        })
        .collect()
}

fn state_for(
    states: &BTreeMap<PredicateKey, PredicateState>,
    role: PredicateRoleV1,
    commitment: &DigestSha256,
) -> PredicateState {
    states
        .get(&PredicateKey {
            role,
            commitment: commitment.clone(),
        })
        .copied()
        .unwrap_or(PredicateState::Missing)
}

fn declared_predicates(plan: &CampaignPlanV1) -> Option<Vec<DeclaredPredicate>> {
    let mut declared = Vec::new();
    for criterion in plan.support_criteria() {
        declared.push(DeclaredPredicate {
            role: PredicateRoleV1::SupportCriterion,
            semantic_id: criterion.criterion().semantic_id().as_str().to_owned(),
            commitment: criterion.criterion().digest(),
            tier: Some(criterion.tier()),
        });
    }

    let fields = parse_wire_fields(
        &plan.canonical_bytes(),
        "symthaea-assurance-campaign-plan-v1",
    )?;
    for role in [
        PredicateRoleV1::Control,
        PredicateRoleV1::FailureCondition,
        PredicateRoleV1::ContradictionCondition,
        PredicateRoleV1::InconclusiveCondition,
    ] {
        declared.extend(parse_semantic_set(&fields, role)?);
    }
    Some(declared)
}

fn parse_semantic_set(
    fields: &[(String, String)],
    role: PredicateRoleV1,
) -> Option<Vec<DeclaredPredicate>> {
    let prefix = role.campaign_wire_prefix();
    let count_label = format!("{prefix}-count");
    let id_label = format!("{prefix}-id");
    let commitment_label = format!("{prefix}-commitment");
    let count_index = fields
        .iter()
        .position(|(label, _)| label == &count_label)?;
    let count: usize = fields.get(count_index)?.1.parse().ok()?;
    let mut cursor = count_index + 1;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let (id_field, semantic_id) = fields.get(cursor)?;
        let (commitment_field, commitment) = fields.get(cursor + 1)?;
        if id_field != &id_label || commitment_field != &commitment_label {
            return None;
        }
        out.push(DeclaredPredicate {
            role,
            semantic_id: semantic_id.clone(),
            commitment: DigestSha256::new(commitment.clone()).ok()?,
            tier: None,
        });
        cursor += 2;
    }
    Some(out)
}

fn parse_wire_fields(bytes: &[u8], expected_header: &str) -> Option<Vec<(String, String)>> {
    let header_end = bytes.iter().position(|byte| *byte == b'\n')?;
    if std::str::from_utf8(&bytes[..header_end]).ok()? != expected_header {
        return None;
    }
    let mut cursor = header_end + 1;
    let mut fields = Vec::new();
    while cursor < bytes.len() {
        let label_end = bytes[cursor..]
            .iter()
            .position(|byte| *byte == b' ')
            .map(|offset| cursor + offset)?;
        let label = std::str::from_utf8(&bytes[cursor..label_end])
            .ok()?
            .to_owned();
        cursor = label_end + 1;

        let length_end = bytes[cursor..]
            .iter()
            .position(|byte| *byte == b':')
            .map(|offset| cursor + offset)?;
        let length: usize = std::str::from_utf8(&bytes[cursor..length_end])
            .ok()?
            .parse()
            .ok()?;
        cursor = length_end + 1;
        let value_end = cursor.checked_add(length)?;
        if value_end >= bytes.len() || bytes[value_end] != b'\n' {
            return None;
        }
        let value = std::str::from_utf8(&bytes[cursor..value_end])
            .ok()?
            .to_owned();
        fields.push((label, value));
        cursor = value_end + 1;
    }
    Some(fields)
}

fn support_tier_rank(tier: SupportTier) -> u8 {
    match tier {
        SupportTier::Structural => 0,
        SupportTier::Observed => 1,
        SupportTier::CausallySupported => 2,
        SupportTier::FunctionallySupported => 3,
    }
}

fn support_tier_name(tier: SupportTier) -> &'static str {
    match tier {
        SupportTier::Structural => "structural",
        SupportTier::Observed => "observed",
        SupportTier::CausallySupported => "causally-supported",
        SupportTier::FunctionallySupported => "functionally-supported",
    }
}

fn field(out: &mut String, label: &str, value: &str) {
    out.push_str(label);
    out.push(' ');
    out.push_str(&value.len().to_string());
    out.push(':');
    out.push_str(value);
    out.push('\n');
}

fn digest_canonical(bytes: &[u8]) -> DigestSha256 {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    DigestSha256::new(encoded).expect("SHA-256 encoding is always valid")
}

#[cfg(test)]
mod tests;
