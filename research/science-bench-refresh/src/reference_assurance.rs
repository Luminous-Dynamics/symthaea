//! Fail-closed structural validation for evaluator-side reference packages.
//!
//! A valid package proves only that the declared records are structurally
//! consistent and content-committed. It does not prove truth, source authenticity,
//! byte-level source-span correspondence, human identity/independence, or runner
//! isolation. Those limitations are explicit in every verification report.

use crate::BUILT_GENERATOR_SOURCE_REVISION;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const REFERENCE_PACKAGE_SCHEMA: &str =
    "symthaea.science_bench_refresh_reference_assurance_package.v1";
pub const REFERENCE_PACKAGE_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_reference_assurance_report.v1";

const HASH_PREFIX: &[u8] = b"symthaea.science_bench_refresh.reference_assurance.v1\0";
const PACKAGE_DOMAIN: &[u8] = b"package.v1\0";
const MAX_CLAIMS: usize = 100_000;
const MAX_SOURCES: usize = 100_000;
const MAX_SPANS: usize = 500_000;
const MAX_ADJUDICATORS: usize = 500_000;
const MAX_ADJUDICATIONS: usize = 500_000;
const MAX_RESOLUTIONS: usize = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferencePackagePurposeV1 {
    ExampleOnly,
    AuditCandidate,
}

impl ReferencePackagePurposeV1 {
    fn is_audit_candidate(self) -> bool {
        self == Self::AuditCandidate
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceLifecycleV1 {
    Active,
    Superseded,
    Deleted,
    Retracted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceDecisionV1 {
    SupportedWithinScope,
    ContradictedWithinScope,
    MixedOrContested,
    InsufficientEvidence,
    ScopeAmbiguous,
    TemporallyStale,
    NotApplicable,
    NotEvaluable,
    Abstain,
}

impl ReferenceDecisionV1 {
    fn requires_evidence(self) -> bool {
        !matches!(
            self,
            Self::InsufficientEvidence
                | Self::ScopeAmbiguous
                | Self::NotApplicable
                | Self::NotEvaluable
                | Self::Abstain
        )
    }

    fn requires_current_source(self) -> bool {
        matches!(
            self,
            Self::SupportedWithinScope | Self::ContradictedWithinScope
        )
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ClaimScopeV1 {
    pub domain: String,
    pub jurisdiction: Option<String>,
    pub target_date: Option<String>,
    pub valid_from: Option<String>,
    pub valid_until: Option<String>,
    pub unit_context: Option<String>,
    pub time_independent: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReferenceClaimV1 {
    pub claim_id: String,
    pub text_sha256: String,
    pub material: bool,
    pub scope: ClaimScopeV1,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReferenceSourceV1 {
    pub source_id: String,
    pub snapshot_sha256: String,
    pub source_identity_sha256: String,
    pub ancestry_group_sha256: Option<String>,
    pub canonicalization_profile: String,
    pub publication_date: Option<String>,
    pub retrieved_date: String,
    pub lifecycle: SourceLifecycleV1,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceSpanV1 {
    pub span_id: String,
    pub source_id: String,
    pub start_offset: u64,
    pub end_offset: u64,
    pub offset_unit: String,
    pub span_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReferenceAdjudicatorV1 {
    pub adjudicator_id_sha256: String,
    pub independence_group_sha256: String,
    pub role: String,
    pub blind_to_system_identity: bool,
    pub blind_to_candidate_identity: bool,
    pub blind_to_judge_output: bool,
    pub conflict_of_interest_declared: bool,
    pub conflict_disclosure_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InitialAdjudicationV1 {
    pub record_id: String,
    pub claim_id: String,
    pub adjudicator_id_sha256: String,
    pub rubric_sha256: String,
    pub decision: ReferenceDecisionV1,
    pub confidence_millionths: Option<u32>,
    pub abstention_reason: Option<String>,
    pub evidence_span_ids: Vec<String>,
    pub rationale_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResolutionEventV1 {
    pub resolution_id: String,
    pub claim_id: String,
    pub adjudicator_id_sha256: String,
    pub decision: ReferenceDecisionV1,
    pub prior_record_ids: Vec<String>,
    pub evidence_span_ids: Vec<String>,
    pub rationale_sha256: String,
}

/// Commitment payload. Field order is part of the v1 serialization contract.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReferencePackageFieldsV1 {
    pub schema: String,
    pub purpose: ReferencePackagePurposeV1,
    pub package_id: String,
    pub validator_source_revision: String,
    pub rubric_sha256: String,
    pub scope_policy_sha256: String,
    pub claims: Vec<ReferenceClaimV1>,
    pub sources: Vec<ReferenceSourceV1>,
    pub spans: Vec<EvidenceSpanV1>,
    pub adjudicators: Vec<ReferenceAdjudicatorV1>,
    pub initial_adjudications: Vec<InitialAdjudicationV1>,
    pub resolutions: Vec<ResolutionEventV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReferencePackageV1 {
    pub schema: String,
    pub purpose: ReferencePackagePurposeV1,
    pub package_id: String,
    pub validator_source_revision: String,
    pub rubric_sha256: String,
    pub scope_policy_sha256: String,
    pub claims: Vec<ReferenceClaimV1>,
    pub sources: Vec<ReferenceSourceV1>,
    pub spans: Vec<EvidenceSpanV1>,
    pub adjudicators: Vec<ReferenceAdjudicatorV1>,
    pub initial_adjudications: Vec<InitialAdjudicationV1>,
    pub resolutions: Vec<ResolutionEventV1>,
    pub package_commitment_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReferenceAssuranceReportV1 {
    pub schema: String,
    pub package_sha256: String,
    pub package_commitment_sha256: String,
    pub package_commitment_matches: bool,
    pub valid: bool,
    pub package_id: String,
    pub claim_count: u64,
    pub material_claim_count: u64,
    pub source_count: u64,
    pub span_count: u64,
    pub adjudicator_count: u64,
    pub initial_adjudication_count: u64,
    pub resolution_count: u64,
    pub claims_missing_minimum_panel: u64,
    pub claims_with_disagreement: u64,
    pub abstention_count: u64,
    pub referenced_source_ancestry_groups: u64,
    pub referenced_sources_with_unknown_ancestry: u64,
    pub material_claims_without_cited_evidence: u64,
    pub fixed_reference_package_only: bool,
    pub adjudicator_independence_proven: bool,
    pub source_authenticity_proven: bool,
    pub source_span_bytes_verified: bool,
    pub runner_execution_proven: bool,
    pub external_anchor_required: bool,
    pub scoring_eligible_declared: bool,
    pub claim_boundary: String,
    pub issues: Vec<String>,
}

fn valid_digest(value: &str, allow_zero: bool) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && (allow_zero || value.bytes().any(|byte| byte != b'0'))
}

fn valid_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && value.bytes().any(|byte| byte != b'0')
}

fn valid_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 10
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || !bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| index == 4 || index == 7 || byte.is_ascii_digit())
    {
        return false;
    }
    let Ok(year) = value[0..4].parse::<u16>() else {
        return false;
    };
    let Ok(month) = value[5..7].parse::<u8>() else {
        return false;
    };
    let Ok(day) = value[8..10].parse::<u8>() else {
        return false;
    };
    if year == 0 || !(1..=12).contains(&month) {
        return false;
    }
    let leap_year = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let max_day = match month {
        2 if leap_year => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    (1..=max_day).contains(&day)
}

fn digest_framed(domain: &[u8], payload: &[u8]) -> Result<String, String> {
    let domain_len =
        u64::try_from(domain.len()).map_err(|_| "hash domain exceeds u64 length".to_owned())?;
    let payload_len =
        u64::try_from(payload.len()).map_err(|_| "hash payload exceeds u64 length".to_owned())?;
    let mut hasher = Sha256::new();
    hasher.update(HASH_PREFIX);
    hasher.update(domain_len.to_be_bytes());
    hasher.update(domain);
    hasher.update(payload_len.to_be_bytes());
    hasher.update(payload);
    Ok(hex_digest(&hasher.finalize()))
}

fn hex_digest(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn fields_from_package(package: &ReferencePackageV1) -> ReferencePackageFieldsV1 {
    ReferencePackageFieldsV1 {
        schema: package.schema.clone(),
        purpose: package.purpose,
        package_id: package.package_id.clone(),
        validator_source_revision: package.validator_source_revision.clone(),
        rubric_sha256: package.rubric_sha256.clone(),
        scope_policy_sha256: package.scope_policy_sha256.clone(),
        claims: package.claims.clone(),
        sources: package.sources.clone(),
        spans: package.spans.clone(),
        adjudicators: package.adjudicators.clone(),
        initial_adjudications: package.initial_adjudications.clone(),
        resolutions: package.resolutions.clone(),
    }
}

fn validate_fields(fields: &ReferencePackageFieldsV1) -> Vec<String> {
    let mut issues = Vec::new();
    let allow_zero = fields.purpose == ReferencePackagePurposeV1::ExampleOnly;
    if fields.schema != REFERENCE_PACKAGE_SCHEMA {
        issues.push("unsupported reference package schema".to_owned());
    }
    if fields.package_id.trim().is_empty() || fields.package_id.len() > 128 {
        issues.push("package_id must be non-empty and at most 128 bytes".to_owned());
    }
    for (name, value) in [
        ("rubric_sha256", fields.rubric_sha256.as_str()),
        ("scope_policy_sha256", fields.scope_policy_sha256.as_str()),
    ] {
        if !valid_digest(value, allow_zero) {
            issues.push(format!("{name} must be a valid lower-case SHA-256 digest"));
        }
    }
    if fields.purpose.is_audit_candidate() {
        if !valid_revision(&fields.validator_source_revision) {
            issues.push("audit candidate requires a non-zero validator source revision SHA".to_owned());
        } else if fields.validator_source_revision != BUILT_GENERATOR_SOURCE_REVISION {
            issues.push("validator source revision differs from inspected build revision".to_owned());
        }
    } else if fields.validator_source_revision.trim().is_empty() {
        issues.push("example-only validator source revision marker must be non-empty".to_owned());
    }

    if fields.claims.is_empty() || fields.claims.len() > MAX_CLAIMS {
        issues.push(format!("claims must contain 1..={MAX_CLAIMS} items"));
    }
    if fields.sources.len() > MAX_SOURCES {
        issues.push(format!("sources must contain at most {MAX_SOURCES} items"));
    }
    if fields.spans.len() > MAX_SPANS {
        issues.push(format!("spans must contain at most {MAX_SPANS} items"));
    }
    if fields.adjudicators.len() > MAX_ADJUDICATORS {
        issues.push(format!("adjudicators must contain at most {MAX_ADJUDICATORS} items"));
    }
    if fields.resolutions.len() > MAX_RESOLUTIONS {
        issues.push(format!("resolutions must contain at most {MAX_RESOLUTIONS} items"));
    }
    if fields.initial_adjudications.len() > MAX_ADJUDICATIONS {
        issues.push(format!("initial_adjudications must contain at most {MAX_ADJUDICATIONS} items"));
    }

    let mut claims = BTreeMap::new();
    for claim in &fields.claims {
        if claim.claim_id.trim().is_empty() || claim.claim_id.len() > 128 {
            issues.push("claim_id must be non-empty and at most 128 bytes".to_owned());
        }
        if claims.insert(claim.claim_id.as_str(), claim).is_some() {
            issues.push(format!("duplicate claim_id {}", claim.claim_id));
        }
        if !valid_digest(&claim.text_sha256, allow_zero) {
            issues.push(format!("claim {} has invalid text_sha256", claim.claim_id));
        }
        let scope = &claim.scope;
        if scope.domain.trim().is_empty() || scope.domain.len() > 256 {
            issues.push(format!("claim {} has empty/oversized domain scope", claim.claim_id));
        }
        for (name, value) in [
            ("target_date", scope.target_date.as_deref()),
            ("valid_from", scope.valid_from.as_deref()),
            ("valid_until", scope.valid_until.as_deref()),
        ] {
            if value.is_some_and(|date| !valid_date(date)) {
                issues.push(format!("claim {} has invalid {name}", claim.claim_id));
            }
        }
        if scope.time_independent {
            if scope.target_date.is_some() || scope.valid_from.is_some() || scope.valid_until.is_some() {
                issues.push(format!("time-independent claim {} must not declare temporal bounds", claim.claim_id));
            }
        } else if scope.target_date.is_none() {
            issues.push(format!("time-dependent claim {} requires target_date", claim.claim_id));
        }
        if let (Some(from), Some(until)) = (&scope.valid_from, &scope.valid_until) {
            if from > until {
                issues.push(format!("claim {} valid_from is after valid_until", claim.claim_id));
            }
        }
        if let Some(target) = &scope.target_date {
            if scope.valid_from.as_ref().is_some_and(|from| target < from)
                || scope.valid_until.as_ref().is_some_and(|until| target > until)
            {
                issues.push(format!("claim {} target_date is outside its declared validity interval", claim.claim_id));
            }
        }
    }

    let mut sources = BTreeMap::new();
    let mut source_snapshot_digests = BTreeSet::new();
    for source in &fields.sources {
        if source.source_id.trim().is_empty() || source.source_id.len() > 128 {
            issues.push("source_id must be non-empty and at most 128 bytes".to_owned());
        }
        if sources.insert(source.source_id.as_str(), source).is_some() {
            issues.push(format!("duplicate source_id {}", source.source_id));
        }
        if !source_snapshot_digests.insert(source.snapshot_sha256.as_str()) {
            issues.push(format!("source {} repeats an exact snapshot digest already assigned to another source", source.source_id));
        }
        for (name, value) in [
            ("snapshot_sha256", source.snapshot_sha256.as_str()),
            ("source_identity_sha256", source.source_identity_sha256.as_str()),
        ] {
            if !valid_digest(value, allow_zero) {
                issues.push(format!("source {} has invalid {name}", source.source_id));
            }
        }
        if source.ancestry_group_sha256.as_ref().is_some_and(|digest| !valid_digest(digest, allow_zero)) {
            issues.push(format!("source {} has invalid ancestry_group_sha256", source.source_id));
        }
        if source.canonicalization_profile.trim().is_empty() {
            issues.push(format!("source {} has no canonicalization profile", source.source_id));
        }
        if !valid_date(&source.retrieved_date)
            || source.publication_date.as_ref().is_some_and(|date| !valid_date(date))
        {
            issues.push(format!("source {} has invalid publication/retrieval date", source.source_id));
        }
    }

    let mut spans = BTreeMap::new();
    for span in &fields.spans {
        if span.span_id.trim().is_empty() || span.span_id.len() > 128 {
            issues.push("span_id must be non-empty and at most 128 bytes".to_owned());
        }
        if spans.insert(span.span_id.as_str(), span).is_some() {
            issues.push(format!("duplicate span_id {}", span.span_id));
        }
        if !sources.contains_key(span.source_id.as_str()) {
            issues.push(format!("span {} references unknown source {}", span.span_id, span.source_id));
        }
        if span.start_offset >= span.end_offset || span.offset_unit != "byte" {
            issues.push(format!("span {} must have a non-empty byte range", span.span_id));
        }
        if !valid_digest(&span.span_sha256, allow_zero) {
            issues.push(format!("span {} has invalid span_sha256", span.span_id));
        }
    }

    let mut adjudicators = BTreeMap::new();
    for adjudicator in &fields.adjudicators {
        if !valid_digest(&adjudicator.adjudicator_id_sha256, allow_zero)
            || !valid_digest(&adjudicator.independence_group_sha256, allow_zero)
        {
            issues.push("adjudicator identity and independence group must be valid digests".to_owned());
        }
        if adjudicators
            .insert(adjudicator.adjudicator_id_sha256.as_str(), adjudicator)
            .is_some()
        {
            issues.push("duplicate adjudicator_id_sha256".to_owned());
        }
        if adjudicator.role.trim().is_empty() || adjudicator.role.len() > 128 {
            issues.push(format!("adjudicator {} has invalid role", adjudicator.adjudicator_id_sha256));
        }
        if !valid_digest(&adjudicator.conflict_disclosure_sha256, allow_zero) {
            issues.push(format!("adjudicator {} has invalid COI disclosure digest", adjudicator.adjudicator_id_sha256));
        }
        if fields.purpose.is_audit_candidate()
            && (!adjudicator.blind_to_system_identity
                || !adjudicator.blind_to_candidate_identity
                || !adjudicator.blind_to_judge_output
                || !adjudicator.conflict_of_interest_declared)
        {
            issues.push(format!("adjudicator {} fails required blinding/COI declarations", adjudicator.adjudicator_id_sha256));
        }
    }

    let mut adjudications = BTreeMap::new();
    let mut per_claim_adjudicators: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut per_claim_independence_groups: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut seen_claim_adjudicator_pairs: BTreeSet<(&str, &str)> = BTreeSet::new();
    let mut seen_claim_independence_pairs: BTreeSet<(&str, &str)> = BTreeSet::new();
    let mut abstentions = 0_u64;
    for record in &fields.initial_adjudications {
        if record.record_id.trim().is_empty() || record.record_id.len() > 128 {
            issues.push("adjudication record_id must be non-empty and at most 128 bytes".to_owned());
        }
        if adjudications.insert(record.record_id.as_str(), record).is_some() {
            issues.push(format!("duplicate adjudication record_id {}", record.record_id));
        }
        if !claims.contains_key(record.claim_id.as_str()) {
            issues.push(format!("adjudication {} references unknown claim {}", record.record_id, record.claim_id));
            continue;
        }
        let Some(adjudicator) = adjudicators.get(record.adjudicator_id_sha256.as_str()) else {
            issues.push(format!("adjudication {} references unknown adjudicator", record.record_id));
            continue;
        };
        if !valid_digest(&record.rubric_sha256, allow_zero)
            || record.rubric_sha256 != fields.rubric_sha256
        {
            issues.push(format!("adjudication {} rubric digest differs from package rubric", record.record_id));
        }
        if !valid_digest(&record.rationale_sha256, allow_zero) {
            issues.push(format!("adjudication {} has invalid rationale digest", record.record_id));
        }
        if record.confidence_millionths.is_some_and(|value| value > 1_000_000) {
            issues.push(format!("adjudication {} confidence exceeds one million", record.record_id));
        }
        if record.decision == ReferenceDecisionV1::Abstain {
            abstentions += 1;
            if record.abstention_reason.as_deref().is_none_or(|reason| reason.trim().is_empty()) {
                issues.push(format!("abstaining adjudication {} requires abstention_reason", record.record_id));
            }
        } else if record.abstention_reason.is_some() {
            issues.push(format!("non-abstaining adjudication {} must not carry abstention_reason", record.record_id));
        }
        if record.decision.requires_evidence() && record.evidence_span_ids.is_empty() {
            issues.push(format!("adjudication {} requires at least one evidence span", record.record_id));
        }
        let mut seen_span_ids = BTreeSet::new();
        for span_id in &record.evidence_span_ids {
            if !seen_span_ids.insert(span_id.as_str()) {
                issues.push(format!("adjudication {} repeats span {}", record.record_id, span_id));
            }
            if !spans.contains_key(span_id.as_str()) {
                issues.push(format!("adjudication {} references unknown span {}", record.record_id, span_id));
            }
        }
        if !seen_claim_adjudicator_pairs.insert((record.claim_id.as_str(), record.adjudicator_id_sha256.as_str())) {
            issues.push(format!("claim {} has duplicate initial votes from one adjudicator", record.claim_id));
        }
        if !seen_claim_independence_pairs.insert((record.claim_id.as_str(), adjudicator.independence_group_sha256.as_str())) {
            issues.push(format!("claim {} has multiple initial votes from one declared independence group", record.claim_id));
        }
        per_claim_adjudicators.entry(record.claim_id.as_str()).or_default().insert(record.adjudicator_id_sha256.as_str());
        per_claim_independence_groups.entry(record.claim_id.as_str()).or_default().insert(adjudicator.independence_group_sha256.as_str());

        if fields.purpose.is_audit_candidate()
            && (!adjudicator.blind_to_system_identity
                || !adjudicator.blind_to_candidate_identity
                || !adjudicator.blind_to_judge_output)
        {
            issues.push(format!("adjudication {} is not blinded", record.record_id));
        }
    }

    for claim in &fields.claims {
        if !claim.material {
            continue;
        }
        let adjudicators_for_claim = per_claim_adjudicators.get(claim.claim_id.as_str());
        let independence_groups = per_claim_independence_groups.get(claim.claim_id.as_str());
        let count = adjudicators_for_claim.map_or(0, |set| set.len());
        let independent_group_count = independence_groups.map_or(0, |set| set.len());
        if fields.purpose.is_audit_candidate()
            && (count < 3 || independent_group_count < 3)
        {
            issues.push(format!("material claim {} requires at least three distinct initial adjudicators and declared independence groups", claim.claim_id));
        }
    }

    let mut resolution_ids = BTreeSet::new();
    for resolution in &fields.resolutions {
        if resolution.resolution_id.trim().is_empty() || resolution.resolution_id.len() > 128
            || !resolution_ids.insert(resolution.resolution_id.as_str())
        {
            issues.push(format!("invalid or duplicate resolution_id {}", resolution.resolution_id));
        }
        if !claims.contains_key(resolution.claim_id.as_str()) {
            issues.push(format!("resolution {} references unknown claim", resolution.resolution_id));
        }
        let Some(resolving_adjudicator) = adjudicators.get(resolution.adjudicator_id_sha256.as_str()) else {
            issues.push(format!("resolution {} references unknown resolving adjudicator", resolution.resolution_id));
            continue;
        };
        if !valid_digest(&resolution.adjudicator_id_sha256, allow_zero) {
            issues.push(format!("resolution {} has invalid resolving adjudicator digest", resolution.resolution_id));
        }
        if !valid_digest(&resolution.rationale_sha256, allow_zero) {
            issues.push(format!("resolution {} has invalid rationale digest", resolution.resolution_id));
        }
        if resolution.decision == ReferenceDecisionV1::Abstain {
            issues.push(format!("resolution {} cannot replace a resolution with an unqualified abstention", resolution.resolution_id));
        }
        if resolution.prior_record_ids.len() < 2 {
            issues.push(format!("resolution {} must reference at least two prior adjudications", resolution.resolution_id));
        }
        let mut seen = BTreeSet::new();
        let mut prior_adjudicator_ids = BTreeSet::new();
        let mut prior_independence_groups = BTreeSet::new();
        for record_id in &resolution.prior_record_ids {
            if !seen.insert(record_id.as_str()) {
                issues.push(format!("resolution {} repeats prior record {}", resolution.resolution_id, record_id));
            }
            match adjudications.get(record_id.as_str()) {
                Some(record) if record.claim_id == resolution.claim_id => {
                    prior_adjudicator_ids.insert(record.adjudicator_id_sha256.as_str());
                    if let Some(prior_adjudicator) = adjudicators.get(record.adjudicator_id_sha256.as_str()) {
                        prior_independence_groups.insert(prior_adjudicator.independence_group_sha256.as_str());
                    }
                }
                Some(_) => issues.push(format!("resolution {} references adjudication for a different claim", resolution.resolution_id)),
                None => issues.push(format!("resolution {} references unknown adjudication {}", resolution.resolution_id, record_id)),
            }
        }
        if prior_adjudicator_ids.len() < 2 {
            issues.push(format!("resolution {} must preserve disagreement from at least two distinct adjudicators", resolution.resolution_id));
        }
        if prior_adjudicator_ids.contains(resolution.adjudicator_id_sha256.as_str()) {
            issues.push(format!("resolution {} must be made by an adjudicator outside the initial voting set", resolution.resolution_id));
        }
        if fields.purpose.is_audit_candidate()
            && prior_independence_groups.contains(resolving_adjudicator.independence_group_sha256.as_str())
        {
            issues.push(format!("resolution {} resolving adjudicator shares an initial panel independence group", resolution.resolution_id));
        }
        if resolution.decision.requires_evidence() && resolution.evidence_span_ids.is_empty() {
            issues.push(format!("resolution {} requires at least one evidence span", resolution.resolution_id));
        }
        let mut seen_resolution_spans = BTreeSet::new();
        for span_id in &resolution.evidence_span_ids {
            if !seen_resolution_spans.insert(span_id.as_str()) {
                issues.push(format!("resolution {} repeats evidence span {}", resolution.resolution_id, span_id));
            }
            if !spans.contains_key(span_id.as_str()) {
                issues.push(format!("resolution {} references unknown span {}", resolution.resolution_id, span_id));
            }
        }
    }

    if fields.purpose.is_audit_candidate() {
        for record in &fields.initial_adjudications {
            if record.decision.requires_current_source() {
                let claim = claims.get(record.claim_id.as_str());
                for span_id in &record.evidence_span_ids {
                    if let Some(span) = spans.get(span_id.as_str()) {
                        if let Some(source) = sources.get(span.source_id.as_str()) {
                            if source.lifecycle != SourceLifecycleV1::Active {
                                issues.push(format!("adjudication {} uses a non-active source as current support without a source-validity interval", record.record_id));
                            }
                            if let (Some(target), Some(published)) = (
                                claim.and_then(|item| item.scope.target_date.as_deref()),
                                source.publication_date.as_deref(),
                            ) {
                                if published > target {
                                    issues.push(format!("adjudication {} uses a source published after the claim target date", record.record_id));
                                }
                            }
                        }
                    }
                }
            }
            if record.decision == ReferenceDecisionV1::TemporallyStale {
                let has_stale_source = record.evidence_span_ids.iter().any(|span_id| {
                    spans.get(span_id.as_str())
                        .and_then(|span| sources.get(span.source_id.as_str()))
                        .is_some_and(|source| source.lifecycle != SourceLifecycleV1::Active)
                });
                if !has_stale_source {
                    issues.push(format!("adjudication {} declares temporal staleness without a superseded/deleted/retracted source", record.record_id));
                }
            }
        }
    }
    issues
}

pub fn compute_reference_package_commitment(fields_bytes: &[u8]) -> Result<String, String> {
    let fields: ReferencePackageFieldsV1 = serde_json::from_slice(fields_bytes)
        .map_err(|error| format!("reference package fields malformed: {error}"))?;
    let issues = validate_fields(&fields);
    if !issues.is_empty() {
        return Err(format!("reference package fields invalid: {}", issues.join("; ")));
    }
    let payload = serde_json::to_vec(&fields)
        .map_err(|error| format!("cannot serialize reference package fields: {error}"))?;
    digest_framed(PACKAGE_DOMAIN, &payload)
}

pub fn verify_reference_assurance_package(
    package_bytes: &[u8],
) -> Result<ReferenceAssuranceReportV1, String> {
    let package: ReferencePackageV1 = serde_json::from_slice(package_bytes)
        .map_err(|error| format!("reference package JSON malformed: {error}"))?;
    let fields = fields_from_package(&package);
    let issues = validate_fields(&fields);
    let expected = if issues.is_empty() {
        compute_reference_package_commitment(
            &serde_json::to_vec(&fields)
                .map_err(|error| format!("cannot serialize reference package fields: {error}"))?,
        )?
    } else {
        String::new()
    };
    let commitment_matches = !expected.is_empty()
        && valid_digest(&package.package_commitment_sha256, false)
        && expected == package.package_commitment_sha256;
    let mut final_issues = issues;
    if !commitment_matches {
        final_issues.push("package commitment missing, invalid or mismatched".to_owned());
    }

    let mut decisions_by_claim: BTreeMap<&str, BTreeSet<ReferenceDecisionV1>> = BTreeMap::new();
    let mut missing_panels = 0_u64;
    let mut disagreement_claims = 0_u64;
    let mut material_claims_without_evidence = 0_u64;
    let mut referenced_ancestry_groups = BTreeSet::new();
    let mut referenced_unknown_ancestry_sources = BTreeSet::new();
    let mut claims_with_cited_spans = BTreeSet::new();
    let sources: BTreeMap<&str, &ReferenceSourceV1> = package.sources.iter()
        .map(|source| (source.source_id.as_str(), source))
        .collect();
    let spans: BTreeMap<&str, &EvidenceSpanV1> = package.spans.iter()
        .map(|span| (span.span_id.as_str(), span))
        .collect();
    let adjudicators: BTreeMap<&str, &ReferenceAdjudicatorV1> = package.adjudicators.iter()
        .map(|adjudicator| (adjudicator.adjudicator_id_sha256.as_str(), adjudicator))
        .collect();
    let mut panel_counts: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut abstention_count = 0_u64;
    for record in &package.initial_adjudications {
        if record.decision == ReferenceDecisionV1::Abstain {
            abstention_count += 1;
        } else {
            decisions_by_claim.entry(record.claim_id.as_str()).or_default().insert(record.decision);
        }
        panel_counts.entry(record.claim_id.as_str()).or_default().insert(record.adjudicator_id_sha256.as_str());
        if !record.evidence_span_ids.is_empty() {
            claims_with_cited_spans.insert(record.claim_id.as_str());
        }
        for span_id in &record.evidence_span_ids {
            if let Some(span) = spans.get(span_id.as_str()) {
                if let Some(source) = sources.get(span.source_id.as_str()) {
                    if let Some(group) = &source.ancestry_group_sha256 {
                        referenced_ancestry_groups.insert(group.as_str());
                    } else {
                        referenced_unknown_ancestry_sources.insert(source.source_id.as_str());
                    }
                }
            }
        }
    }
    for claim in package.claims.iter().filter(|claim| claim.material) {
        if !claims_with_cited_spans.contains(claim.claim_id.as_str()) {
            material_claims_without_evidence += 1;
        }
        if panel_counts.get(claim.claim_id.as_str()).map_or(0, |set| set.len()) < 3 {
            missing_panels += 1;
        }
        if decisions_by_claim.get(claim.claim_id.as_str()).is_some_and(|decisions| decisions.len() > 1) {
            disagreement_claims += 1;
        }
    }
    Ok(ReferenceAssuranceReportV1 {
        schema: REFERENCE_PACKAGE_REPORT_SCHEMA.to_owned(),
        package_sha256: hex_digest(&Sha256::digest(package_bytes)),
        package_commitment_sha256: package.package_commitment_sha256,
        package_commitment_matches: commitment_matches,
        valid: final_issues.is_empty(),
        package_id: package.package_id,
        claim_count: package.claims.len() as u64,
        material_claim_count: package.claims.iter().filter(|claim| claim.material).count() as u64,
        source_count: package.sources.len() as u64,
        span_count: package.spans.len() as u64,
        adjudicator_count: adjudicators.len() as u64,
        initial_adjudication_count: package.initial_adjudications.len() as u64,
        resolution_count: package.resolutions.len() as u64,
        claims_missing_minimum_panel: missing_panels,
        claims_with_disagreement: disagreement_claims,
        abstention_count,
        referenced_source_ancestry_groups: referenced_ancestry_groups.len() as u64,
        referenced_sources_with_unknown_ancestry: referenced_unknown_ancestry_sources.len() as u64,
        material_claims_without_cited_evidence: material_claims_without_evidence,
        fixed_reference_package_only: true,
        adjudicator_independence_proven: false,
        source_authenticity_proven: false,
        source_span_bytes_verified: false,
        runner_execution_proven: false,
        external_anchor_required: true,
        scoring_eligible_declared: false,
        claim_boundary: "Structural and commitment validity only; not truth, annotator independence, source authenticity, source-span byte correspondence, or runner isolation.".to_owned(),
        issues: final_issues,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    const FIXTURE: &str =
        include_str!("../fixtures/reference-assurance-package.example.json");

    fn fixture_value() -> Value {
        serde_json::from_str(FIXTURE).expect("fixture JSON is valid")
    }

    fn verify_value(value: &Value) -> ReferenceAssuranceReportV1 {
        verify_reference_assurance_package(
            &serde_json::to_vec(value).expect("test value serializes"),
        )
        .expect("test package parses")
    }

    fn reseal(value: &mut Value) {
        let mut fields = value.clone();
        fields.as_object_mut().unwrap().remove("package_commitment_sha256");
        let bytes = serde_json::to_vec(&fields).unwrap();
        let commitment = compute_reference_package_commitment(&bytes).unwrap();
        value["package_commitment_sha256"] = json!(commitment);
    }


    #[test]
    fn example_package_is_structurally_valid_but_never_scoring_eligible() {
        let report = verify_reference_assurance_package(FIXTURE.as_bytes()).unwrap();
        assert!(report.valid, "{:?}", report.issues);
        assert!(report.package_commitment_matches);
        assert_eq!(report.claim_count, 1);
        assert_eq!(report.material_claim_count, 1);
        assert_eq!(report.initial_adjudication_count, 3);
        assert_eq!(report.claims_with_disagreement, 1);
        assert_eq!(report.claims_missing_minimum_panel, 0);
        assert!(!report.adjudicator_independence_proven);
        assert!(!report.source_authenticity_proven);
        assert!(!report.source_span_bytes_verified);
        assert!(!report.runner_execution_proven);
        assert!(report.external_anchor_required);
        assert!(!report.scoring_eligible_declared);
    }

    #[test]
    fn audit_candidate_passes_structure_but_never_claims_scoring_eligibility() {
        let mut value = fixture_value();
        value["purpose"] = json!("audit_candidate");
        value["validator_source_revision"] = json!(BUILT_GENERATOR_SOURCE_REVISION);
        reseal(&mut value);
        let report = verify_value(&value);
        assert!(report.valid, "{:?}", report.issues);
        assert!(!report.adjudicator_independence_proven);
        assert!(!report.source_authenticity_proven);
        assert!(!report.source_span_bytes_verified);
        assert!(!report.runner_execution_proven);
        assert!(report.external_anchor_required);
        assert!(!report.scoring_eligible_declared);
    }

    #[test]
    fn invalid_calendar_day_is_rejected() {
        let mut value = fixture_value();
        value["claims"][0]["scope"]["target_date"] = json!("2026-02-30");
        let report = verify_value(&value);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("invalid target_date")));
    }

    #[test]
    fn repeated_verification_is_byte_deterministic() {
        let first = serde_json::to_vec(&verify_reference_assurance_package(FIXTURE.as_bytes()).unwrap()).unwrap();
        let second = serde_json::to_vec(&verify_reference_assurance_package(FIXTURE.as_bytes()).unwrap()).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn duplicate_claim_id_is_rejected() {
        let mut value = fixture_value();
        let duplicate = value["claims"][0].clone();
        value["claims"].as_array_mut().unwrap().push(duplicate);
        let report = verify_value(&value);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("duplicate claim_id")));
    }

    #[test]
    fn orphan_span_is_rejected() {
        let mut value = fixture_value();
        value["spans"][0]["source_id"] = json!("missing-source");
        let report = verify_value(&value);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("unknown source")));
    }

    #[test]
    fn duplicate_rater_vote_is_rejected() {
        let mut value = fixture_value();
        let duplicate = value["initial_adjudications"][0].clone();
        value["initial_adjudications"].as_array_mut().unwrap().push(duplicate);
        let report = verify_value(&value);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("duplicate adjudication record_id")));
    }

    #[test]
    fn invalid_span_range_is_rejected() {
        let mut value = fixture_value();
        value["spans"][0]["end_offset"] = json!(0);
        let report = verify_value(&value);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("byte range")));
    }

    #[test]
    fn unknown_span_reference_is_rejected() {
        let mut value = fixture_value();
        value["initial_adjudications"][0]["evidence_span_ids"] = json!(["does-not-exist"]);
        let report = verify_value(&value);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("unknown span")));
    }

    #[test]
    fn stale_source_cannot_be_used_as_current_support() {
        let mut value = fixture_value();
        value["sources"][0]["lifecycle"] = json!("retracted");
        let report = verify_value(&value);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("deleted/retracted source")));
    }

    #[test]
    fn superseded_source_cannot_be_used_as_current_support_without_validity_interval() {
        let mut value = fixture_value();
        value["sources"][0]["lifecycle"] = json!("superseded");
        let report = verify_value(&value);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("non-active source as current support")));
    }

    #[test]
    fn future_source_cannot_support_past_target_date() {
        let mut value = fixture_value();
        value["sources"][0]["publication_date"] = json!("2030-01-01");
        let report = verify_value(&value);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("published after")));
    }

    #[test]
    fn scope_interval_must_contain_target_date() {
        let mut value = fixture_value();
        value["claims"][0]["scope"]["valid_until"] = json!("2020-01-01");
        let report = verify_value(&value);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("outside its declared validity")));
    }

    #[test]
    fn abstention_requires_a_reason_and_is_preserved() {
        let mut value = fixture_value();
        value["initial_adjudications"][0]["decision"] = json!("abstain");
        value["initial_adjudications"][0]["abstention_reason"] = Value::Null;
        let report = verify_value(&value);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("requires abstention_reason")));
    }

    #[test]
    fn non_abstention_cannot_smuggle_abstention_reason() {
        let mut value = fixture_value();
        value["initial_adjudications"][0]["abstention_reason"] = json!("not enough evidence");
        let report = verify_value(&value);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("must not carry abstention_reason")));
    }

    #[test]
    fn disagreement_is_reported_not_collapsed() {
        let report = verify_reference_assurance_package(FIXTURE.as_bytes()).unwrap();
        assert_eq!(report.claims_with_disagreement, 1);
        assert_eq!(report.initial_adjudication_count, 3);
    }

    #[test]
    fn commitment_mutation_is_rejected() {
        let mut value = fixture_value();
        value["package_id"] = json!("changed-after-commitment");
        let report = verify_value(&value);
        assert!(!report.valid);
        assert!(!report.package_commitment_matches);
    }

    #[test]
    fn same_adjudicator_cannot_vote_twice_on_one_claim() {
        let mut value = fixture_value();
        value["initial_adjudications"][1]["adjudicator_id_sha256"] =
            value["initial_adjudications"][0]["adjudicator_id_sha256"].clone();
        let report = verify_value(&value);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("duplicate initial votes from one adjudicator")));
    }

    #[test]
    fn one_declared_independence_group_cannot_supply_multiple_votes() {
        let mut value = fixture_value();
        value["adjudicators"][1]["independence_group_sha256"] =
            value["adjudicators"][0]["independence_group_sha256"].clone();
        let report = verify_value(&value);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("one declared independence group")));
    }

    #[test]
    fn mirrors_in_one_ancestry_group_are_counted_once() {
        let mut value = fixture_value();
        value["sources"][1]["ancestry_group_sha256"] =
            value["sources"][0]["ancestry_group_sha256"].clone();
        reseal(&mut value);
        let report = verify_value(&value);
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.referenced_source_ancestry_groups, 1);
    }

    #[test]
    fn duplicate_exact_source_snapshot_is_rejected() {
        let mut value = fixture_value();
        value["sources"][1]["snapshot_sha256"] = value["sources"][0]["snapshot_sha256"].clone();
        let report = verify_value(&value);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("repeats an exact snapshot digest")));
    }

    #[test]
    fn unknown_source_ancestry_is_visible_not_assumed_independent() {
        let mut value = fixture_value();
        value["sources"][0]["ancestry_group_sha256"] = Value::Null;
        value["sources"][1]["ancestry_group_sha256"] = Value::Null;
        reseal(&mut value);
        let report = verify_value(&value);
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.referenced_source_ancestry_groups, 0);
        assert_eq!(report.referenced_sources_with_unknown_ancestry, 2);
        assert!(!report.adjudicator_independence_proven);
    }

    #[test]
    fn insufficient_evidence_is_not_forced_into_a_gold_label() {
        let mut value = fixture_value();
        for record in value["initial_adjudications"].as_array_mut().unwrap() {
            record["decision"] = json!("insufficient_evidence");
            record["evidence_span_ids"] = json!([]);
        }
        reseal(&mut value);
        let report = verify_value(&value);
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.material_claims_without_cited_evidence, 1);
        assert_eq!(report.claims_with_disagreement, 0);
        assert_eq!(report.initial_adjudication_count, 3);
    }

    #[test]
    fn target_date_must_be_inside_validity_interval() {
        let mut value = fixture_value();
        value["claims"][0]["scope"]["valid_from"] = json!("2027-01-01");
        let report = verify_value(&value);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("outside its declared validity")));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut value = fixture_value();
        value["unrecognized_trust_me"] = json!(true);
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(verify_reference_assurance_package(&bytes).is_err());
    }
}
