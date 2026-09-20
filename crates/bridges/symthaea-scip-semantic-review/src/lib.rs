// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authority-narrow semantic review contracts for accepted SCIP language surfaces.
//!
//! The central distinction is deliberate:
//!
//! ```text
//! surface accepted
//!     != semantic review structurally complete
//!     != semantic fidelity qualified
//! ```
//!
//! This crate binds a review to the exact grounded source identity and exact
//! accepted surface produced by the v15 receipt boundary. It can prove that a
//! review report is complete, canonical, and internally bound to that subject.
//! It cannot prove that the review findings are correct. Positive semantic
//! authority belongs to a later independently qualified verifier-execution
//! capability, not to this structural report type.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fmt;

use blake3::Hasher;
use symthaea_communication::GroundedConceptGraph;
use symthaea_interlingua::graph_semantic_hash;
use symthaea_scip_llm_adapter::digest_surface_text;
use symthaea_scip_realization_receipt::{
    RealizationClaimScope, ScipSurfaceRealizationReceiptV1,
};

pub const SCIP_SEMANTIC_REVIEW_PROFILE_V1: &str = "symthaea.scip-semantic-review/v1";

const REPORT_DOMAIN_V1: &[u8] = b"symthaea-scip-semantic-review-v1\0";
const MAX_VERIFIER_PROFILE_BYTES: usize = 4 * 1024;
const MAX_VERIFIER_RUN_ID_BYTES: usize = 4 * 1024;
const MAX_SOURCE_ANCHOR_BYTES: usize = 8 * 1024;
const MAX_SURFACE_SPANS_PER_FINDING: usize = 64;

/// Closed semantic dimensions for the first Broca realization review profile.
///
/// Adding a dimension changes the review contract and requires a new profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SemanticDimensionV1 {
    Polarity,
    EpistemicStrength,
    ModalForce,
    Attribution,
    QuantifierCardinality,
    TemporalScope,
    EntityReference,
    CausalForce,
    UncertaintyQualification,
    UnsupportedAddition,
    RequiredOmission,
    AmbiguityIntroduction,
}

impl SemanticDimensionV1 {
    pub const ALL: [Self; 12] = [
        Self::Polarity,
        Self::EpistemicStrength,
        Self::ModalForce,
        Self::Attribution,
        Self::QuantifierCardinality,
        Self::TemporalScope,
        Self::EntityReference,
        Self::CausalForce,
        Self::UncertaintyQualification,
        Self::UnsupportedAddition,
        Self::RequiredOmission,
        Self::AmbiguityIntroduction,
    ];

    fn code(self) -> u8 {
        match self {
            Self::Polarity => 0,
            Self::EpistemicStrength => 1,
            Self::ModalForce => 2,
            Self::Attribution => 3,
            Self::QuantifierCardinality => 4,
            Self::TemporalScope => 5,
            Self::EntityReference => 6,
            Self::CausalForce => 7,
            Self::UncertaintyQualification => 8,
            Self::UnsupportedAddition => 9,
            Self::RequiredOmission => 10,
            Self::AmbiguityIntroduction => 11,
        }
    }
}

/// A verifier's reported disposition for one semantic dimension.
///
/// These values are report contents, not trusted findings merely because they
/// appear in a structurally valid report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticVerdictV1 {
    Preserved,
    Violated,
    Inconclusive,
    NotApplicable,
}

impl SemanticVerdictV1 {
    fn code(self) -> u8 {
        match self {
            Self::Preserved => 0,
            Self::Violated => 1,
            Self::Inconclusive => 2,
            Self::NotApplicable => 3,
        }
    }
}

/// Summary of the *reported* findings. It is deliberately not named Faithful,
/// Verified, Passed, or Grounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReportDispositionV1 {
    NoReportedViolation,
    ContainsReportedInconclusive,
    ContainsReportedViolation,
}

/// The complete positive claim of the v1 output capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticReviewClaimScopeV1 {
    ReportStructureOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SurfaceSpanV1 {
    pub start_byte: u64,
    pub end_byte: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticFindingV1 {
    pub dimension: SemanticDimensionV1,
    pub verdict: SemanticVerdictV1,
    /// Descriptive verifier-owned pointer into the grounded source semantics.
    /// This string is bound into the report but is not itself proof that the
    /// verifier interpreted the source correctly.
    pub source_anchor: String,
    /// Zero or more exact UTF-8 byte spans in the accepted surface.
    /// Omission findings may legitimately have no surface span.
    pub surface_spans: Vec<SurfaceSpanV1>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticReviewCandidateV1 {
    pub realization_receipt_digest: String,
    pub verifier_profile: String,
    pub verifier_run_id: String,
    pub findings: Vec<SemanticFindingV1>,
}

/// Exact subject of a semantic review.
///
/// Construction proves only that the supplied grounded graph and accepted
/// surface match the identities already bound by the v15 realization receipt.
/// It does not establish that the generated surface faithfully expresses that
/// graph.
#[derive(Clone, PartialEq, Eq)]
pub struct SemanticReviewSubjectV1 {
    realization_receipt_digest: String,
    source_semantic_hash: String,
    surface_digest: String,
    surface_bytes: u64,
}

impl fmt::Debug for SemanticReviewSubjectV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SemanticReviewSubjectV1")
            .field("surface_bytes", &self.surface_bytes)
            .finish()
    }
}

impl SemanticReviewSubjectV1 {
    pub fn bind(
        source_graph: &GroundedConceptGraph,
        accepted_surface: &str,
        receipt: &ScipSurfaceRealizationReceiptV1,
    ) -> Result<Self, SemanticReviewError> {
        if receipt.claim_scope() != RealizationClaimScope::SurfaceAcceptedOnly {
            return Err(SemanticReviewError::UnsupportedReceiptClaimScope);
        }

        let source_semantic_hash =
            graph_semantic_hash(source_graph).map_err(|_| SemanticReviewError::InvalidSourceGraph)?;
        if source_semantic_hash != receipt.source_semantic_hash() {
            return Err(SemanticReviewError::SourceSemanticHashMismatch);
        }

        let surface_digest = digest_surface_text(accepted_surface);
        if surface_digest != receipt.surface_digest() {
            return Err(SemanticReviewError::SurfaceDigestMismatch);
        }
        let surface_bytes = accepted_surface.len() as u64;
        if surface_bytes != receipt.surface_bytes() {
            return Err(SemanticReviewError::SurfaceLengthMismatch);
        }

        Ok(Self {
            realization_receipt_digest: receipt.receipt_digest().to_owned(),
            source_semantic_hash,
            surface_digest,
            surface_bytes,
        })
    }

    pub fn realization_receipt_digest(&self) -> &str {
        &self.realization_receipt_digest
    }

    pub fn source_semantic_hash(&self) -> &str {
        &self.source_semantic_hash
    }

    pub fn surface_digest(&self) -> &str {
        &self.surface_digest
    }

    pub fn surface_bytes(&self) -> u64 {
        self.surface_bytes
    }
}

/// Opaque structural validation witness for one complete semantic review.
///
/// Private fields prevent callers from constructing a positive-looking report
/// without traversing the completeness/span/subject checks below. The witness
/// still carries no semantic-fidelity authority: correctness of the findings is
/// intentionally outside this type's claim scope.
#[derive(Clone, PartialEq, Eq)]
pub struct StructurallyValidatedSemanticReviewV1 {
    report_digest: String,
    realization_receipt_digest: String,
    verifier_profile: String,
    verifier_run_id: String,
    disposition: ReportDispositionV1,
    findings: Vec<SemanticFindingV1>,
}

impl fmt::Debug for StructurallyValidatedSemanticReviewV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StructurallyValidatedSemanticReviewV1")
            .field("profile", &SCIP_SEMANTIC_REVIEW_PROFILE_V1)
            .field(
                "claim_scope",
                &SemanticReviewClaimScopeV1::ReportStructureOnly,
            )
            .field("disposition", &self.disposition)
            .field("finding_count", &self.findings.len())
            .finish()
    }
}

impl StructurallyValidatedSemanticReviewV1 {
    pub fn validate(
        subject: &SemanticReviewSubjectV1,
        accepted_surface: &str,
        candidate: SemanticReviewCandidateV1,
    ) -> Result<Self, SemanticReviewError> {
        if candidate.realization_receipt_digest != subject.realization_receipt_digest {
            return Err(SemanticReviewError::ReceiptBindingMismatch);
        }
        if digest_surface_text(accepted_surface) != subject.surface_digest
            || accepted_surface.len() as u64 != subject.surface_bytes
        {
            return Err(SemanticReviewError::SurfaceBindingMismatch);
        }
        validate_bounded_nonempty(
            &candidate.verifier_profile,
            MAX_VERIFIER_PROFILE_BYTES,
            SemanticReviewError::InvalidVerifierProfile,
        )?;
        validate_bounded_nonempty(
            &candidate.verifier_run_id,
            MAX_VERIFIER_RUN_ID_BYTES,
            SemanticReviewError::InvalidVerifierRunId,
        )?;

        if candidate.findings.len() != SemanticDimensionV1::ALL.len() {
            return Err(SemanticReviewError::IncompleteDimensionCensus);
        }

        let mut seen = BTreeSet::new();
        let mut findings = candidate.findings;
        for finding in &mut findings {
            if !seen.insert(finding.dimension) {
                return Err(SemanticReviewError::DuplicateDimension(finding.dimension));
            }
            validate_and_canonicalize_finding(finding, accepted_surface)?;
        }
        if SemanticDimensionV1::ALL
            .into_iter()
            .any(|dimension| !seen.contains(&dimension))
        {
            return Err(SemanticReviewError::IncompleteDimensionCensus);
        }

        findings.sort_by_key(|finding| finding.dimension);
        let disposition = derive_disposition(&findings);
        let report_digest = report_digest(
            subject,
            &candidate.verifier_profile,
            &candidate.verifier_run_id,
            &findings,
        );

        Ok(Self {
            report_digest,
            realization_receipt_digest: subject.realization_receipt_digest.clone(),
            verifier_profile: candidate.verifier_profile,
            verifier_run_id: candidate.verifier_run_id,
            disposition,
            findings,
        })
    }

    pub fn profile(&self) -> &'static str {
        SCIP_SEMANTIC_REVIEW_PROFILE_V1
    }

    pub fn claim_scope(&self) -> SemanticReviewClaimScopeV1 {
        SemanticReviewClaimScopeV1::ReportStructureOnly
    }

    pub fn report_digest(&self) -> &str {
        &self.report_digest
    }

    pub fn realization_receipt_digest(&self) -> &str {
        &self.realization_receipt_digest
    }

    pub fn verifier_profile(&self) -> &str {
        &self.verifier_profile
    }

    pub fn verifier_run_id(&self) -> &str {
        &self.verifier_run_id
    }

    pub fn disposition(&self) -> ReportDispositionV1 {
        self.disposition
    }

    pub fn findings(&self) -> &[SemanticFindingV1] {
        &self.findings
    }

    /// Structural validation alone never establishes semantic fidelity.
    pub fn semantic_fidelity_established(&self) -> bool {
        false
    }

    /// A bound profile/run identifier is not authentication of a verifier.
    pub fn verifier_authenticated(&self) -> bool {
        false
    }

    /// Structural separation of code or labels is not evidence of independent
    /// implementation, organization, model, data, or failure modes.
    pub fn verifier_independence_established(&self) -> bool {
        false
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutationDirectionV1 {
    Inversion,
    Strengthening,
    Substitution,
    Addition,
    Omission,
    Ambiguation,
}

/// Deterministic adversarial linguistic mutation used to qualify later semantic
/// verifier implementations. The corpus is a falsification target, not a model
/// benchmark score and not proof that passing it establishes general fidelity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdversarialSemanticCaseV1 {
    pub id: &'static str,
    pub dimension: SemanticDimensionV1,
    pub direction: MutationDirectionV1,
    pub source: &'static str,
    pub mutated: &'static str,
    pub expected: SemanticVerdictV1,
}

pub fn adversarial_semantic_corpus_v1() -> Vec<AdversarialSemanticCaseV1> {
    use MutationDirectionV1::{
        Addition, Ambiguation, Inversion, Omission, Strengthening, Substitution,
    };
    use SemanticDimensionV1::{
        AmbiguityIntroduction, Attribution, CausalForce, EntityReference, EpistemicStrength,
        ModalForce, Polarity, QuantifierCardinality, RequiredOmission, TemporalScope,
        UncertaintyQualification, UnsupportedAddition,
    };

    vec![
        adversarial(
            "polarity-01",
            Polarity,
            Inversion,
            "No fault was observed.",
            "A fault was observed.",
        ),
        adversarial(
            "polarity-scope-02",
            Polarity,
            Inversion,
            "The test did not establish that node A failed.",
            "The test established that node A did not fail.",
        ),
        adversarial(
            "epistemic-01",
            EpistemicStrength,
            Strengthening,
            "The evidence suggests the service recovered.",
            "The evidence proves the service recovered.",
        ),
        adversarial(
            "modal-01",
            ModalForce,
            Strengthening,
            "The service may restart.",
            "The service will restart.",
        ),
        adversarial(
            "attribution-01",
            Attribution,
            Omission,
            "Alice reported that node 7 failed.",
            "Node 7 failed.",
        ),
        adversarial(
            "attribution-02",
            Attribution,
            Omission,
            "According to the simulation, the design passed.",
            "The design passed.",
        ),
        adversarial(
            "quantifier-01",
            QuantifierCardinality,
            Strengthening,
            "Some nodes responded.",
            "All nodes responded.",
        ),
        adversarial(
            "quantifier-02",
            QuantifierCardinality,
            Substitution,
            "3 of 10 nodes responded.",
            "10 of 10 nodes responded.",
        ),
        adversarial(
            "temporal-01",
            TemporalScope,
            Strengthening,
            "The service was available during the test window.",
            "The service is available.",
        ),
        adversarial(
            "entity-01",
            EntityReference,
            Substitution,
            "Sensor A exceeded the threshold.",
            "Sensor B exceeded the threshold.",
        ),
        adversarial(
            "causal-01",
            CausalForce,
            Strengthening,
            "A was associated with B.",
            "A caused B.",
        ),
        adversarial(
            "causal-02",
            CausalForce,
            Strengthening,
            "Recovery followed the configuration change.",
            "The configuration change caused recovery.",
        ),
        adversarial(
            "uncertainty-01",
            UncertaintyQualification,
            Omission,
            "The estimate is 10 ± 2.",
            "The value is 10.",
        ),
        adversarial(
            "uncertainty-02",
            UncertaintyQualification,
            Omission,
            "The estimate is 10 with 95% CI [8, 12].",
            "The value is 10.",
        ),
        adversarial(
            "unsupported-01",
            UnsupportedAddition,
            Addition,
            "The sample passed the checksum.",
            "The sample passed the checksum and is safe to deploy.",
        ),
        adversarial(
            "omission-01",
            RequiredOmission,
            Omission,
            "The sample passed checksum and failed latency.",
            "The sample passed checksum.",
        ),
        adversarial(
            "ambiguity-01",
            AmbiguityIntroduction,
            Ambiguation,
            "Node A sent the packet to Node B.",
            "It sent it there.",
        ),
    ]
}

fn adversarial(
    id: &'static str,
    dimension: SemanticDimensionV1,
    direction: MutationDirectionV1,
    source: &'static str,
    mutated: &'static str,
) -> AdversarialSemanticCaseV1 {
    AdversarialSemanticCaseV1 {
        id,
        dimension,
        direction,
        source,
        mutated,
        expected: SemanticVerdictV1::Violated,
    }
}

fn validate_and_canonicalize_finding(
    finding: &mut SemanticFindingV1,
    accepted_surface: &str,
) -> Result<(), SemanticReviewError> {
    validate_bounded_nonempty(
        &finding.source_anchor,
        MAX_SOURCE_ANCHOR_BYTES,
        SemanticReviewError::InvalidSourceAnchor,
    )?;
    if finding.surface_spans.len() > MAX_SURFACE_SPANS_PER_FINDING {
        return Err(SemanticReviewError::TooManySurfaceSpans);
    }
    for span in &finding.surface_spans {
        let start = usize::try_from(span.start_byte)
            .map_err(|_| SemanticReviewError::InvalidSurfaceSpan)?;
        let end =
            usize::try_from(span.end_byte).map_err(|_| SemanticReviewError::InvalidSurfaceSpan)?;
        if start >= end
            || end > accepted_surface.len()
            || !accepted_surface.is_char_boundary(start)
            || !accepted_surface.is_char_boundary(end)
        {
            return Err(SemanticReviewError::InvalidSurfaceSpan);
        }
    }
    finding.surface_spans.sort_unstable();
    if finding
        .surface_spans
        .windows(2)
        .any(|pair| pair[0] == pair[1])
    {
        return Err(SemanticReviewError::DuplicateSurfaceSpan);
    }
    Ok(())
}

fn validate_bounded_nonempty(
    value: &str,
    max_bytes: usize,
    error: SemanticReviewError,
) -> Result<(), SemanticReviewError> {
    if value.is_empty() || value.len() > max_bytes {
        return Err(error);
    }
    Ok(())
}

fn derive_disposition(findings: &[SemanticFindingV1]) -> ReportDispositionV1 {
    if findings
        .iter()
        .any(|finding| finding.verdict == SemanticVerdictV1::Violated)
    {
        ReportDispositionV1::ContainsReportedViolation
    } else if findings
        .iter()
        .any(|finding| finding.verdict == SemanticVerdictV1::Inconclusive)
    {
        ReportDispositionV1::ContainsReportedInconclusive
    } else {
        ReportDispositionV1::NoReportedViolation
    }
}

fn report_digest(
    subject: &SemanticReviewSubjectV1,
    verifier_profile: &str,
    verifier_run_id: &str,
    findings: &[SemanticFindingV1],
) -> String {
    let mut hasher = Hasher::new();
    hasher.update(REPORT_DOMAIN_V1);
    update_str(&mut hasher, SCIP_SEMANTIC_REVIEW_PROFILE_V1);
    hasher.update(&[0]); // ReportStructureOnly
    update_str(&mut hasher, &subject.realization_receipt_digest);
    update_str(&mut hasher, &subject.source_semantic_hash);
    update_str(&mut hasher, &subject.surface_digest);
    update_u64(&mut hasher, subject.surface_bytes);
    update_str(&mut hasher, verifier_profile);
    update_str(&mut hasher, verifier_run_id);
    update_u64(&mut hasher, findings.len() as u64);
    for finding in findings {
        hasher.update(&[finding.dimension.code()]);
        hasher.update(&[finding.verdict.code()]);
        update_str(&mut hasher, &finding.source_anchor);
        update_u64(&mut hasher, finding.surface_spans.len() as u64);
        for span in &finding.surface_spans {
            update_u64(&mut hasher, span.start_byte);
            update_u64(&mut hasher, span.end_byte);
        }
    }
    hasher.finalize().to_hex().to_string()
}

fn update_str(hasher: &mut Hasher, value: &str) {
    update_u64(hasher, value.len() as u64);
    hasher.update(value.as_bytes());
}

fn update_u64(hasher: &mut Hasher, value: u64) {
    hasher.update(&value.to_be_bytes());
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticReviewError {
    UnsupportedReceiptClaimScope,
    InvalidSourceGraph,
    SourceSemanticHashMismatch,
    SurfaceDigestMismatch,
    SurfaceLengthMismatch,
    ReceiptBindingMismatch,
    SurfaceBindingMismatch,
    InvalidVerifierProfile,
    InvalidVerifierRunId,
    IncompleteDimensionCensus,
    DuplicateDimension(SemanticDimensionV1),
    InvalidSourceAnchor,
    TooManySurfaceSpans,
    InvalidSurfaceSpan,
    DuplicateSurfaceSpan,
}

impl fmt::Display for SemanticReviewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedReceiptClaimScope => {
                f.write_str("unsupported realization receipt claim scope")
            }
            Self::InvalidSourceGraph => {
                f.write_str("source graph could not be canonically identified")
            }
            Self::SourceSemanticHashMismatch => {
                f.write_str("source graph does not match realization receipt")
            }
            Self::SurfaceDigestMismatch => {
                f.write_str("surface digest does not match realization receipt")
            }
            Self::SurfaceLengthMismatch => {
                f.write_str("surface length does not match realization receipt")
            }
            Self::ReceiptBindingMismatch => {
                f.write_str("review candidate is bound to a different realization receipt")
            }
            Self::SurfaceBindingMismatch => {
                f.write_str("review surface does not match the bound subject")
            }
            Self::InvalidVerifierProfile => f.write_str("invalid verifier profile identifier"),
            Self::InvalidVerifierRunId => f.write_str("invalid verifier run identifier"),
            Self::IncompleteDimensionCensus => {
                f.write_str("semantic review does not cover exactly the required dimensions")
            }
            Self::DuplicateDimension(dimension) => {
                write!(f, "duplicate semantic dimension: {dimension:?}")
            }
            Self::InvalidSourceAnchor => f.write_str("invalid source anchor"),
            Self::TooManySurfaceSpans => {
                f.write_str("too many surface spans for one semantic finding")
            }
            Self::InvalidSurfaceSpan => f.write_str("invalid UTF-8 surface byte span"),
            Self::DuplicateSurfaceSpan => f.write_str("duplicate surface byte span"),
        }
    }
}

impl std::error::Error for SemanticReviewError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn subject(surface: &str) -> SemanticReviewSubjectV1 {
        SemanticReviewSubjectV1 {
            realization_receipt_digest: "receipt-identity".into(),
            source_semantic_hash: "source-semantic-identity".into(),
            surface_digest: digest_surface_text(surface),
            surface_bytes: surface.len() as u64,
        }
    }

    fn findings(verdict: SemanticVerdictV1) -> Vec<SemanticFindingV1> {
        SemanticDimensionV1::ALL
            .into_iter()
            .map(|dimension| SemanticFindingV1 {
                dimension,
                verdict,
                source_anchor: format!("source:{dimension:?}"),
                surface_spans: vec![],
            })
            .collect()
    }

    fn candidate(findings: Vec<SemanticFindingV1>) -> SemanticReviewCandidateV1 {
        SemanticReviewCandidateV1 {
            realization_receipt_digest: "receipt-identity".into(),
            verifier_profile: "test-verifier/v1".into(),
            verifier_run_id: "run-001".into(),
            findings,
        }
    }

    #[test]
    fn adversarial_corpus_covers_every_required_dimension() {
        let corpus = adversarial_semantic_corpus_v1();
        let dimensions: BTreeSet<_> = corpus.iter().map(|case| case.dimension).collect();
        let ids: BTreeSet<_> = corpus.iter().map(|case| case.id).collect();

        assert_eq!(dimensions.len(), SemanticDimensionV1::ALL.len());
        assert_eq!(ids.len(), corpus.len());
        for dimension in SemanticDimensionV1::ALL {
            assert!(dimensions.contains(&dimension));
        }
        for case in corpus {
            assert_ne!(case.source, case.mutated);
            assert_eq!(case.expected, SemanticVerdictV1::Violated);
        }
    }

    #[test]
    fn missing_dimension_fails_closed() {
        let surface = "accepted surface";
        let mut all = findings(SemanticVerdictV1::Preserved);
        all.pop();

        assert_eq!(
            StructurallyValidatedSemanticReviewV1::validate(
                &subject(surface),
                surface,
                candidate(all),
            ),
            Err(SemanticReviewError::IncompleteDimensionCensus)
        );
    }

    #[test]
    fn duplicate_dimension_fails_closed() {
        let surface = "accepted surface";
        let mut all = findings(SemanticVerdictV1::Preserved);
        all[11].dimension = all[0].dimension;

        assert!(matches!(
            StructurallyValidatedSemanticReviewV1::validate(
                &subject(surface),
                surface,
                candidate(all),
            ),
            Err(SemanticReviewError::DuplicateDimension(_))
        ));
    }

    #[test]
    fn invalid_utf8_and_out_of_bounds_spans_fail_closed() {
        let surface = "AéB";
        let mut all = findings(SemanticVerdictV1::Preserved);
        all[0].surface_spans = vec![SurfaceSpanV1 {
            start_byte: 2,
            end_byte: 3,
        }];
        assert_eq!(
            StructurallyValidatedSemanticReviewV1::validate(
                &subject(surface),
                surface,
                candidate(all),
            ),
            Err(SemanticReviewError::InvalidSurfaceSpan)
        );

        let mut all = findings(SemanticVerdictV1::Preserved);
        all[0].surface_spans = vec![SurfaceSpanV1 {
            start_byte: 0,
            end_byte: 99,
        }];
        assert_eq!(
            StructurallyValidatedSemanticReviewV1::validate(
                &subject(surface),
                surface,
                candidate(all),
            ),
            Err(SemanticReviewError::InvalidSurfaceSpan)
        );
    }

    #[test]
    fn canonical_report_identity_is_finding_and_span_order_independent() {
        let surface = "accepted surface";
        let mut ordered = findings(SemanticVerdictV1::Preserved);
        ordered[0].surface_spans = vec![
            SurfaceSpanV1 {
                start_byte: 9,
                end_byte: 16,
            },
            SurfaceSpanV1 {
                start_byte: 0,
                end_byte: 8,
            },
        ];
        let mut reversed = ordered.clone();
        reversed.reverse();
        reversed
            .iter_mut()
            .find(|finding| finding.dimension == SemanticDimensionV1::Polarity)
            .unwrap()
            .surface_spans
            .reverse();

        let a = StructurallyValidatedSemanticReviewV1::validate(
            &subject(surface),
            surface,
            candidate(ordered),
        )
        .unwrap();
        let b = StructurallyValidatedSemanticReviewV1::validate(
            &subject(surface),
            surface,
            candidate(reversed),
        )
        .unwrap();

        assert_eq!(a.report_digest(), b.report_digest());
        assert_eq!(a.findings(), b.findings());
    }

    #[test]
    fn duplicate_surface_span_fails_closed() {
        let surface = "accepted surface";
        let mut all = findings(SemanticVerdictV1::Preserved);
        let span = SurfaceSpanV1 {
            start_byte: 0,
            end_byte: 8,
        };
        all[0].surface_spans = vec![span, span];

        assert_eq!(
            StructurallyValidatedSemanticReviewV1::validate(
                &subject(surface),
                surface,
                candidate(all),
            ),
            Err(SemanticReviewError::DuplicateSurfaceSpan)
        );
    }

    #[test]
    fn no_reported_violation_still_mints_no_semantic_authority() {
        let surface = "accepted surface";
        let review = StructurallyValidatedSemanticReviewV1::validate(
            &subject(surface),
            surface,
            candidate(findings(SemanticVerdictV1::Preserved)),
        )
        .unwrap();

        assert_eq!(
            review.disposition(),
            ReportDispositionV1::NoReportedViolation
        );
        assert_eq!(
            review.claim_scope(),
            SemanticReviewClaimScopeV1::ReportStructureOnly
        );
        assert!(!review.semantic_fidelity_established());
        assert!(!review.verifier_authenticated());
        assert!(!review.verifier_independence_established());
    }

    #[test]
    fn violation_and_inconclusive_dispositions_remain_report_descriptions() {
        let surface = "accepted surface";
        let mut violated = findings(SemanticVerdictV1::Preserved);
        violated[0].verdict = SemanticVerdictV1::Violated;
        let violated_review = StructurallyValidatedSemanticReviewV1::validate(
            &subject(surface),
            surface,
            candidate(violated),
        )
        .unwrap();
        assert_eq!(
            violated_review.disposition(),
            ReportDispositionV1::ContainsReportedViolation
        );

        let mut inconclusive = findings(SemanticVerdictV1::Preserved);
        inconclusive[0].verdict = SemanticVerdictV1::Inconclusive;
        let inconclusive_review = StructurallyValidatedSemanticReviewV1::validate(
            &subject(surface),
            surface,
            candidate(inconclusive),
        )
        .unwrap();
        assert_eq!(
            inconclusive_review.disposition(),
            ReportDispositionV1::ContainsReportedInconclusive
        );
        assert!(!inconclusive_review.semantic_fidelity_established());
    }

    #[test]
    fn exact_surface_binding_is_rechecked_at_review_validation() {
        let surface = "accepted surface";
        assert_eq!(
            StructurallyValidatedSemanticReviewV1::validate(
                &subject(surface),
                "different surface",
                candidate(findings(SemanticVerdictV1::Preserved)),
            ),
            Err(SemanticReviewError::SurfaceBindingMismatch)
        );
    }
}
