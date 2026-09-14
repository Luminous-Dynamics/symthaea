// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! ASSURE-003: deterministic, non-scalar resolution of preregistered campaign evidence.
//!
//! This crate is deliberately a resolver, not an evaluator. Callers supply typed
//! predicate evaluations that are already bound to evidence from the exact campaign
//! context. The resolver preserves positive support, contradictions, inconclusive
//! findings and limitations without averaging heterogeneous evidence into a score.
//!
//! First-slice boundaries are explicit: registration currentness is only
//! `TerminalInSuppliedView`, action authority is not established here, production
//! witnessing is not inferred from later commitment ordering, and reproduction is
//! not evaluated.

#![deny(unsafe_code)]

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_assurance_campaign::{CampaignPlanV1, TerminalRegistrationInViewV1};
use symthaea_assurance_core::{DigestSha256, StableId, SupportTier};
use symthaea_assurance_semantics::SemanticCommitmentV1;
use thiserror::Error;

pub const ASSURE_RESOLUTION_SCHEMA: &str = "symthaea.assurance.resolution.v1";
const CAMPAIGN_WIRE_DOMAIN: &[u8] = b"symthaea-assurance-campaign-plan-v1\n";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ResolutionError {
    #[error("resolution context does not match the exact campaign plan")]
    ContextPlanMismatch,
    #[error("terminal registration does not match the exact campaign plan")]
    RegistrationPlanMismatch,
    #[error("zero-evidence context must bind the registration's empty evidence root")]
    EmptyLedgerRootMismatch,
    #[error("non-empty evidence context must not retain the empty evidence root")]
    NonEmptyLedgerHasEmptyRoot,
    #[error("predicate evaluation requires at least one admitted-evidence binding")]
    MissingEvaluationEvidence,
    #[error("predicate evaluation repeats one admitted-evidence binding")]
    DuplicateEvaluationEvidence,
    #[error("duplicate predicate evaluation for role {role} and semantic {semantic_id}")]
    DuplicateEvaluation {
        role: &'static str,
        semantic_id: String,
    },
    #[error("evaluation is not preregistered for role {role}: {semantic_id}")]
    UnregisteredPredicate {
        role: &'static str,
        semantic_id: String,
    },
    #[error("evaluation semantic commitment differs from the preregistered commitment")]
    SemanticCommitmentMismatch,
    #[error("campaign-v1 public wire is malformed or unsupported: {0}")]
    CampaignWire(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PredicateRoleV1 {
    SupportCriterion,
    Control,
    FailureCondition,
    ContradictionCondition,
    InconclusiveCondition,
}

impl PredicateRoleV1 {
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::SupportCriterion => "support-criterion",
            Self::Control => "control",
            Self::FailureCondition => "failure-condition",
            Self::ContradictionCondition => "contradiction-condition",
            Self::InconclusiveCondition => "inconclusive-condition",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PredicateDispositionV1 {
    Satisfied,
    NotDemonstrated,
    Contradicted,
    Inconclusive,
}

impl PredicateDispositionV1 {
    const fn canonical_name(self) -> &'static str {
        match self {
            Self::Satisfied => "satisfied",
            Self::NotDemonstrated => "not-demonstrated",
            Self::Contradicted => "contradicted",
            Self::Inconclusive => "inconclusive",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredicateEvaluationV1 {
    role: PredicateRoleV1,
    semantic: SemanticCommitmentV1,
    disposition: PredicateDispositionV1,
    evidence_admissions: Vec<DigestSha256>,
}

impl PredicateEvaluationV1 {
    pub fn new(
        role: PredicateRoleV1,
        semantic: SemanticCommitmentV1,
        disposition: PredicateDispositionV1,
        mut evidence_admissions: Vec<DigestSha256>,
    ) -> Result<Self, ResolutionError> {
        if evidence_admissions.is_empty() {
            return Err(ResolutionError::MissingEvaluationEvidence);
        }
        evidence_admissions.sort();
        if evidence_admissions.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ResolutionError::DuplicateEvaluationEvidence);
        }
        Ok(Self {
            role,
            semantic,
            disposition,
            evidence_admissions,
        })
    }

    pub fn role(&self) -> PredicateRoleV1 {
        self.role
    }

    pub fn semantic(&self) -> &SemanticCommitmentV1 {
        &self.semantic
    }

    pub fn disposition(&self) -> PredicateDispositionV1 {
        self.disposition
    }

    pub fn evidence_admissions(&self) -> &[DigestSha256] {
        &self.evidence_admissions
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from("symthaea-assurance-predicate-evaluation-v1\n");
        field(&mut out, "role", self.role.canonical_name());
        field(&mut out, "semantic-id", self.semantic.semantic_id().as_str());
        field(
            &mut out,
            "semantic-commitment",
            self.semantic.digest().as_str(),
        );
        field(&mut out, "disposition", self.disposition.canonical_name());
        field(
            &mut out,
            "evidence-admission-count",
            &self.evidence_admissions.len().to_string(),
        );
        for admission in &self.evidence_admissions {
            field(&mut out, "evidence-admission", admission.as_str());
        }
        out.into_bytes()
    }

    pub fn digest(&self) -> DigestSha256 {
        digest_canonical(&self.canonical_bytes())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistrationCurrentnessV1 {
    TerminalInSuppliedView,
}

impl RegistrationCurrentnessV1 {
    const fn canonical_name(self) -> &'static str {
        match self {
            Self::TerminalInSuppliedView => "terminal-in-supplied-view",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthorityResolutionV1 {
    NotEstablished,
}

impl AuthorityResolutionV1 {
    const fn canonical_name(self) -> &'static str {
        match self {
            Self::NotEstablished => "not-established",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReproductionResolutionV1 {
    NotEvaluated,
}

impl ReproductionResolutionV1 {
    const fn canonical_name(self) -> &'static str {
        match self {
            Self::NotEvaluated => "not-evaluated",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimFindingV1 {
    Supported,
    NotDemonstrated,
    Contradicted,
    Inconclusive,
}

impl ClaimFindingV1 {
    const fn canonical_name(self) -> &'static str {
        match self {
            Self::Supported => "supported",
            Self::NotDemonstrated => "not-demonstrated",
            Self::Contradicted => "contradicted",
            Self::Inconclusive => "inconclusive",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolutionContextV1 {
    claim_digest: DigestSha256,
    subject_manifest_id: DigestSha256,
    subject_core_id: DigestSha256,
    campaign_plan_digest: DigestSha256,
    registration_digest: DigestSha256,
    evidence_root: DigestSha256,
    admitted_count: u64,
    currentness: RegistrationCurrentnessV1,
}

impl ResolutionContextV1 {
    pub fn new(
        plan: &CampaignPlanV1,
        current: &TerminalRegistrationInViewV1,
        evidence_root: DigestSha256,
        admitted_count: u64,
    ) -> Result<Self, ResolutionError> {
        let receipt = current.receipt();
        if receipt.plan_digest() != &plan.digest()
            || receipt.campaign_nonce() != plan.campaign_nonce()
            || receipt.claim_digest() != plan.claim_digest()
            || receipt.subject_manifest_id() != plan.subject_manifest_id()
        {
            return Err(ResolutionError::RegistrationPlanMismatch);
        }
        if admitted_count == 0 && &evidence_root != receipt.empty_evidence_root() {
            return Err(ResolutionError::EmptyLedgerRootMismatch);
        }
        if admitted_count > 0 && &evidence_root == receipt.empty_evidence_root() {
            return Err(ResolutionError::NonEmptyLedgerHasEmptyRoot);
        }
        Ok(Self {
            claim_digest: plan.claim_digest().clone(),
            subject_manifest_id: plan.subject_manifest_id().clone(),
            subject_core_id: plan.subject_core_id().clone(),
            campaign_plan_digest: plan.digest(),
            registration_digest: current.digest(),
            evidence_root,
            admitted_count,
            currentness: RegistrationCurrentnessV1::TerminalInSuppliedView,
        })
    }

    pub fn claim_digest(&self) -> &DigestSha256 {
        &self.claim_digest
    }

    pub fn subject_manifest_id(&self) -> &DigestSha256 {
        &self.subject_manifest_id
    }

    pub fn subject_core_id(&self) -> &DigestSha256 {
        &self.subject_core_id
    }

    pub fn campaign_plan_digest(&self) -> &DigestSha256 {
        &self.campaign_plan_digest
    }

    pub fn registration_digest(&self) -> &DigestSha256 {
        &self.registration_digest
    }

    pub fn evidence_root(&self) -> &DigestSha256 {
        &self.evidence_root
    }

    pub fn admitted_count(&self) -> u64 {
        self.admitted_count
    }

    pub fn currentness(&self) -> RegistrationCurrentnessV1 {
        self.currentness
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from("symthaea-assurance-resolution-context-v1\n");
        field(&mut out, "claim", self.claim_digest.as_str());
        field(
            &mut out,
            "assure-001-subject",
            self.subject_manifest_id.as_str(),
        );
        field(&mut out, "core-subject", self.subject_core_id.as_str());
        field(&mut out, "campaign-plan", self.campaign_plan_digest.as_str());
        field(&mut out, "registration", self.registration_digest.as_str());
        field(&mut out, "evidence-root", self.evidence_root.as_str());
        field(&mut out, "admitted-count", &self.admitted_count.to_string());
        field(&mut out, "currentness", self.currentness.canonical_name());
        out.into_bytes()
    }

    pub fn digest(&self) -> DigestSha256 {
        digest_canonical(&self.canonical_bytes())
    }

    fn matches_plan(&self, plan: &CampaignPlanV1) -> bool {
        self.claim_digest == *plan.claim_digest()
            && self.subject_manifest_id == *plan.subject_manifest_id()
            && self.subject_core_id == *plan.subject_core_id()
            && self.campaign_plan_digest == plan.digest()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssuranceResolutionV1 {
    context: ResolutionContextV1,
    claim_finding: ClaimFindingV1,
    attained_support: Option<SupportTier>,
    evaluations: Vec<PredicateEvaluationV1>,
    authority: AuthorityResolutionV1,
    reproduction: ReproductionResolutionV1,
    limitations: Vec<StableId>,
}

impl AssuranceResolutionV1 {
    pub fn context(&self) -> &ResolutionContextV1 {
        &self.context
    }

    pub fn claim_finding(&self) -> ClaimFindingV1 {
        self.claim_finding
    }

    pub fn attained_support(&self) -> Option<SupportTier> {
        self.attained_support
    }

    pub fn evaluations(&self) -> &[PredicateEvaluationV1] {
        &self.evaluations
    }

    pub fn authority(&self) -> AuthorityResolutionV1 {
        self.authority
    }

    pub fn reproduction(&self) -> ReproductionResolutionV1 {
        self.reproduction
    }

    pub fn limitations(&self) -> &[StableId] {
        &self.limitations
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from("symthaea-assurance-resolution-v1\n");
        field(&mut out, "schema", ASSURE_RESOLUTION_SCHEMA);
        field(&mut out, "context", self.context.digest().as_str());
        field(&mut out, "claim-finding", self.claim_finding.canonical_name());
        field(
            &mut out,
            "attained-support",
            self.attained_support.map(support_tier_name).unwrap_or(""),
        );
        field(&mut out, "authority", self.authority.canonical_name());
        field(
            &mut out,
            "reproduction",
            self.reproduction.canonical_name(),
        );
        field(
            &mut out,
            "evaluation-count",
            &self.evaluations.len().to_string(),
        );
        for evaluation in &self.evaluations {
            field(&mut out, "evaluation", evaluation.digest().as_str());
        }
        field(
            &mut out,
            "limitation-count",
            &self.limitations.len().to_string(),
        );
        for limitation in &self.limitations {
            field(&mut out, "limitation", limitation.as_str());
        }
        out.into_bytes()
    }

    pub fn digest(&self) -> DigestSha256 {
        digest_canonical(&self.canonical_bytes())
    }
}

#[derive(Debug, Clone)]
struct ExpectedPredicate {
    commitment: DigestSha256,
    tier: Option<SupportTier>,
}

/// Resolve one exact preregistered campaign without assigning a scalar trust/safety score.
pub fn resolve_campaign(
    plan: &CampaignPlanV1,
    context: ResolutionContextV1,
    mut evaluations: Vec<PredicateEvaluationV1>,
) -> Result<AssuranceResolutionV1, ResolutionError> {
    if !context.matches_plan(plan) {
        return Err(ResolutionError::ContextPlanMismatch);
    }

    let expected = expected_predicates(plan)?;
    evaluations.sort_by(|left, right| {
        left.role
            .cmp(&right.role)
            .then_with(|| left.semantic.semantic_id().cmp(right.semantic.semantic_id()))
    });

    let mut supplied = BTreeMap::<(PredicateRoleV1, String), &PredicateEvaluationV1>::new();
    for evaluation in &evaluations {
        let key = (
            evaluation.role,
            evaluation.semantic.semantic_id().as_str().to_owned(),
        );
        let Some(expected_predicate) = expected.get(&key) else {
            return Err(ResolutionError::UnregisteredPredicate {
                role: evaluation.role.canonical_name(),
                semantic_id: key.1,
            });
        };
        if evaluation.semantic.digest() != expected_predicate.commitment {
            return Err(ResolutionError::SemanticCommitmentMismatch);
        }
        if supplied.insert(key.clone(), evaluation).is_some() {
            return Err(ResolutionError::DuplicateEvaluation {
                role: evaluation.role.canonical_name(),
                semantic_id: key.1,
            });
        }
    }

    let mut missing_required = false;
    let mut contradiction = false;
    let mut inconclusive = false;
    let mut failure = false;
    let mut controls_satisfied = true;

    for (key, predicate) in &expected {
        let Some(evaluation) = supplied.get(key).copied() else {
            missing_required = true;
            if key.0 == PredicateRoleV1::Control {
                controls_satisfied = false;
            }
            continue;
        };
        match key.0 {
            PredicateRoleV1::SupportCriterion => match evaluation.disposition {
                PredicateDispositionV1::Satisfied | PredicateDispositionV1::NotDemonstrated => {}
                PredicateDispositionV1::Contradicted => contradiction = true,
                PredicateDispositionV1::Inconclusive => inconclusive = true,
            },
            PredicateRoleV1::Control => match evaluation.disposition {
                PredicateDispositionV1::Satisfied => {}
                PredicateDispositionV1::NotDemonstrated => controls_satisfied = false,
                PredicateDispositionV1::Contradicted | PredicateDispositionV1::Inconclusive => {
                    controls_satisfied = false;
                    inconclusive = true;
                }
            },
            PredicateRoleV1::FailureCondition => match evaluation.disposition {
                PredicateDispositionV1::Satisfied => failure = true,
                PredicateDispositionV1::Inconclusive => inconclusive = true,
                PredicateDispositionV1::NotDemonstrated | PredicateDispositionV1::Contradicted => {}
            },
            PredicateRoleV1::ContradictionCondition => match evaluation.disposition {
                PredicateDispositionV1::Satisfied => contradiction = true,
                PredicateDispositionV1::Inconclusive => inconclusive = true,
                PredicateDispositionV1::NotDemonstrated | PredicateDispositionV1::Contradicted => {}
            },
            PredicateRoleV1::InconclusiveCondition => match evaluation.disposition {
                PredicateDispositionV1::Satisfied | PredicateDispositionV1::Inconclusive => {
                    inconclusive = true
                }
                PredicateDispositionV1::NotDemonstrated | PredicateDispositionV1::Contradicted => {}
            },
        }
        let _ = predicate;
    }

    let attained_support = if controls_satisfied {
        attained_support(plan, &supplied)
    } else {
        None
    };

    let claim_finding = if contradiction {
        ClaimFindingV1::Contradicted
    } else if inconclusive {
        ClaimFindingV1::Inconclusive
    } else if failure || missing_required || attained_support.is_none() {
        ClaimFindingV1::NotDemonstrated
    } else {
        ClaimFindingV1::Supported
    };

    let limitations = vec![
        stable("registration-currentness-terminal-in-supplied-view-only"),
        stable("registration-authority-not-established"),
        stable("production-witness-not-established"),
        stable("reproduction-not-evaluated"),
    ];

    Ok(AssuranceResolutionV1 {
        context,
        claim_finding,
        attained_support,
        evaluations,
        authority: AuthorityResolutionV1::NotEstablished,
        reproduction: ReproductionResolutionV1::NotEvaluated,
        limitations,
    })
}

fn attained_support(
    plan: &CampaignPlanV1,
    supplied: &BTreeMap<(PredicateRoleV1, String), &PredicateEvaluationV1>,
) -> Option<SupportTier> {
    let tiers = [
        SupportTier::Structural,
        SupportTier::Observed,
        SupportTier::CausallySupported,
        SupportTier::FunctionallySupported,
    ];
    let mut attained = None;
    for tier in tiers {
        if tier > plan.maximum_support() {
            break;
        }
        let relevant: Vec<_> = plan
            .support_criteria()
            .iter()
            .filter(|criterion| criterion.tier() <= tier)
            .collect();
        if relevant.is_empty() {
            continue;
        }
        let all_satisfied = relevant.iter().all(|criterion| {
            supplied
                .get(&(
                    PredicateRoleV1::SupportCriterion,
                    criterion.criterion().semantic_id().as_str().to_owned(),
                ))
                .is_some_and(|evaluation| {
                    evaluation.disposition == PredicateDispositionV1::Satisfied
                })
        });
        if all_satisfied {
            attained = Some(tier);
        } else {
            break;
        }
    }
    attained
}

fn expected_predicates(
    plan: &CampaignPlanV1,
) -> Result<BTreeMap<(PredicateRoleV1, String), ExpectedPredicate>, ResolutionError> {
    let mut expected = BTreeMap::new();
    for criterion in plan.support_criteria() {
        expected.insert(
            (
                PredicateRoleV1::SupportCriterion,
                criterion.criterion().semantic_id().as_str().to_owned(),
            ),
            ExpectedPredicate {
                commitment: criterion.criterion().digest(),
                tier: Some(criterion.tier()),
            },
        );
    }

    let fields = parse_campaign_fields(&plan.canonical_bytes())?;
    for (role, prefix) in [
        (PredicateRoleV1::Control, "control"),
        (PredicateRoleV1::FailureCondition, "failure-condition"),
        (
            PredicateRoleV1::ContradictionCondition,
            "contradiction-condition",
        ),
        (
            PredicateRoleV1::InconclusiveCondition,
            "inconclusive-condition",
        ),
    ] {
        for (semantic_id, commitment) in semantic_refs(&fields, prefix)? {
            expected.insert(
                (role, semantic_id),
                ExpectedPredicate {
                    commitment,
                    tier: None,
                },
            );
        }
    }
    Ok(expected)
}

fn parse_campaign_fields(bytes: &[u8]) -> Result<Vec<(String, String)>, ResolutionError> {
    if !bytes.starts_with(CAMPAIGN_WIRE_DOMAIN) {
        return Err(ResolutionError::CampaignWire(
            "unexpected campaign-v1 domain separator".into(),
        ));
    }
    let mut cursor = CAMPAIGN_WIRE_DOMAIN.len();
    let mut fields = Vec::new();
    while cursor < bytes.len() {
        let label_end = bytes[cursor..]
            .iter()
            .position(|byte| *byte == b' ')
            .map(|offset| cursor + offset)
            .ok_or_else(|| ResolutionError::CampaignWire("missing field separator".into()))?;
        let label = std::str::from_utf8(&bytes[cursor..label_end])
            .map_err(|_| ResolutionError::CampaignWire("non-UTF-8 field label".into()))?;
        cursor = label_end + 1;
        let length_end = bytes[cursor..]
            .iter()
            .position(|byte| *byte == b':')
            .map(|offset| cursor + offset)
            .ok_or_else(|| ResolutionError::CampaignWire("missing field length delimiter".into()))?;
        let length = std::str::from_utf8(&bytes[cursor..length_end])
            .map_err(|_| ResolutionError::CampaignWire("non-ASCII field length".into()))?
            .parse::<usize>()
            .map_err(|_| ResolutionError::CampaignWire("invalid field length".into()))?;
        let value_start = length_end + 1;
        let value_end = value_start
            .checked_add(length)
            .ok_or_else(|| ResolutionError::CampaignWire("field length overflow".into()))?;
        if value_end >= bytes.len() || bytes[value_end] != b'\n' {
            return Err(ResolutionError::CampaignWire(
                "field length does not match UTF-8 byte framing".into(),
            ));
        }
        let value = std::str::from_utf8(&bytes[value_start..value_end])
            .map_err(|_| ResolutionError::CampaignWire("non-UTF-8 field value".into()))?;
        fields.push((label.to_owned(), value.to_owned()));
        cursor = value_end + 1;
    }
    Ok(fields)
}

fn semantic_refs(
    fields: &[(String, String)],
    prefix: &str,
) -> Result<Vec<(String, DigestSha256)>, ResolutionError> {
    let count_label = format!("{prefix}-count");
    let id_label = format!("{prefix}-id");
    let commitment_label = format!("{prefix}-commitment");
    let counts: Vec<_> = fields
        .iter()
        .filter(|(label, _)| label == &count_label)
        .map(|(_, value)| value)
        .collect();
    if counts.len() != 1 {
        return Err(ResolutionError::CampaignWire(format!(
            "expected exactly one {count_label}"
        )));
    }
    let count = counts[0]
        .parse::<usize>()
        .map_err(|_| ResolutionError::CampaignWire(format!("invalid {count_label}")))?;
    let ids: Vec<_> = fields
        .iter()
        .filter(|(label, _)| label == &id_label)
        .map(|(_, value)| value.clone())
        .collect();
    let commitments: Vec<_> = fields
        .iter()
        .filter(|(label, _)| label == &commitment_label)
        .map(|(_, value)| value.clone())
        .collect();
    if ids.len() != count || commitments.len() != count {
        return Err(ResolutionError::CampaignWire(format!(
            "{prefix} semantic count mismatch"
        )));
    }
    let mut seen = BTreeSet::new();
    ids.into_iter()
        .zip(commitments)
        .map(|(semantic_id, encoded)| {
            if !seen.insert(semantic_id.clone()) {
                return Err(ResolutionError::CampaignWire(format!(
                    "duplicate {prefix} semantic id"
                )));
            }
            let commitment = DigestSha256::new(encoded).map_err(|_| {
                ResolutionError::CampaignWire(format!("invalid {prefix} semantic digest"))
            })?;
            Ok((semantic_id, commitment))
        })
        .collect()
}

fn support_tier_name(tier: SupportTier) -> &'static str {
    match tier {
        SupportTier::Structural => "structural",
        SupportTier::Observed => "observed",
        SupportTier::CausallySupported => "causally-supported",
        SupportTier::FunctionallySupported => "functionally-supported",
    }
}

fn stable(value: &str) -> StableId {
    StableId::new(value).expect("static ASSURE-003 limitation identifiers are valid")
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
