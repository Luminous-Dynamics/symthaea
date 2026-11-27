//! Fail-closed reference contract for dependence-regime qualification.
//!
//! The contract binds a frozen dependence envelope to the exact score, model,
//! scope, block length and alpha. It qualifies only when the complete observed
//! dependence interval lies inside that envelope. Partial overlap is
//! inconclusive; a disjoint interval or mismatched metadata is not qualified.
//!
//! This module validates receipt shape and decision logic only. It does not
//! verify signatures, prove that the envelope was independently validated,
//! establish that an AR(1) model is adequate, or detect arbitrary dependence.
//!
//! Run:
//!   rustc --edition=2024 --test scripts/research/sdcb_dependence_envelope_contract.rs -o /tmp/sdcb-envelope-tests
//!   /tmp/sdcb-envelope-tests

const CONTRACT_VERSION_V1: u16 = 1;
const MAX_IDENTIFIER_BYTES: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarginalFamilyV1 {
    Gaussian,
    StandardizedStudentT5CommonScale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DependenceModelV1 {
    StationaryGaussianAr1,
    Ar1WithStudentT5CommonScale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreDefinitionV1 {
    BlockMaximumAbsoluteResidual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelAssessmentV1 {
    AdequateWithinDeclaredModel,
    Rejected,
    Inconclusive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualificationStatusV1 {
    QualifiedWithinEnvelope,
    NotQualified,
    Inconclusive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualificationReasonV1 {
    EntireIntervalInsideFrozenEnvelope,
    InvalidEnvelopeContract,
    ScopeMismatch,
    MarginalFamilyMismatch,
    DependenceModelMismatch,
    ScoreDefinitionMismatch,
    BlockLengthMismatch,
    AlphaMismatch,
    DependenceModelRejected,
    DiagnosticAssessmentInconclusive,
    InsufficientDiagnosticSamples,
    InvalidObservationDigest,
    InvalidDependenceInterval,
    IntervalEntirelyOutsideEnvelope,
    IntervalOverlapsEnvelopeBoundary,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedDependenceEnvelopeV1 {
    pub schema_version: u16,
    pub envelope_id: String,
    pub scope_id: String,
    pub marginal_family: MarginalFamilyV1,
    pub dependence_model: DependenceModelV1,
    pub score_definition: ScoreDefinitionV1,
    pub block_length: u32,
    /// Alpha encoded in basis points: 1,000 = 10%; 500 = 5%.
    pub alpha_basis_points: u16,
    pub rho_lower_bound: f64,
    pub rho_upper_bound: f64,
    pub minimum_diagnostic_samples: u64,
    /// Digest of the independent validation receipt. Digest shape is checked;
    /// authentication and reviewer authorization are external trust boundaries.
    pub validation_receipt_sha256: String,
    /// Digest binding the held-out regime set used to validate this envelope.
    pub heldout_regime_set_sha256: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DependenceObservationV1 {
    pub scope_id: String,
    pub marginal_family: MarginalFamilyV1,
    pub dependence_model: DependenceModelV1,
    pub score_definition: ScoreDefinitionV1,
    pub block_length: u32,
    pub alpha_basis_points: u16,
    /// Conservative interval emitted by the declared dependence diagnostic.
    pub rho_lower_bound: f64,
    pub rho_upper_bound: f64,
    pub diagnostic_sample_count: u64,
    pub model_assessment: ModelAssessmentV1,
    /// Digest of the exact observation window/diagnostic evidence.
    pub evidence_sha256: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QualificationDecisionV1 {
    pub status: QualificationStatusV1,
    pub reason: QualificationReasonV1,
}

fn is_nonempty_bounded_identifier(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= MAX_IDENTIFIER_BYTES
}

fn is_canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn family_matches_dependence_model(
    family: MarginalFamilyV1,
    model: DependenceModelV1,
) -> bool {
    matches!(
        (family, model),
        (
            MarginalFamilyV1::Gaussian,
            DependenceModelV1::StationaryGaussianAr1
        ) | (
            MarginalFamilyV1::StandardizedStudentT5CommonScale,
            DependenceModelV1::Ar1WithStudentT5CommonScale
        )
    )
}

fn envelope_is_well_formed(envelope: &ValidatedDependenceEnvelopeV1) -> bool {
    envelope.schema_version == CONTRACT_VERSION_V1
        && family_matches_dependence_model(envelope.marginal_family, envelope.dependence_model)
        && is_nonempty_bounded_identifier(&envelope.envelope_id)
        && is_nonempty_bounded_identifier(&envelope.scope_id)
        && envelope.block_length > 0
        && (1..=5_000).contains(&envelope.alpha_basis_points)
        && envelope.minimum_diagnostic_samples > 0
        && envelope.rho_lower_bound.is_finite()
        && envelope.rho_upper_bound.is_finite()
        && (-1.0..=1.0).contains(&envelope.rho_lower_bound)
        && (-1.0..=1.0).contains(&envelope.rho_upper_bound)
        && envelope.rho_lower_bound <= envelope.rho_upper_bound
        && is_canonical_sha256(&envelope.validation_receipt_sha256)
        && is_canonical_sha256(&envelope.heldout_regime_set_sha256)
}

fn decision(
    status: QualificationStatusV1,
    reason: QualificationReasonV1,
) -> QualificationDecisionV1 {
    QualificationDecisionV1 { status, reason }
}

/// Classify an observation against a frozen, externally validated envelope.
///
/// Positive qualification requires exact scope/model/score/block/alpha matches,
/// a passing model-adequacy assessment, enough diagnostic samples, a canonical
/// evidence digest, and an observed confidence interval wholly inside the
/// envelope. No partial-overlap or malformed-evidence path can qualify.
pub fn classify_dependence_regime_v1(
    envelope: &ValidatedDependenceEnvelopeV1,
    observation: &DependenceObservationV1,
) -> QualificationDecisionV1 {
    if !envelope_is_well_formed(envelope) {
        return decision(
            QualificationStatusV1::NotQualified,
            QualificationReasonV1::InvalidEnvelopeContract,
        );
    }

    if observation.scope_id != envelope.scope_id {
        return decision(
            QualificationStatusV1::NotQualified,
            QualificationReasonV1::ScopeMismatch,
        );
    }
    if observation.marginal_family != envelope.marginal_family {
        return decision(
            QualificationStatusV1::NotQualified,
            QualificationReasonV1::MarginalFamilyMismatch,
        );
    }
    if observation.dependence_model != envelope.dependence_model {
        return decision(
            QualificationStatusV1::NotQualified,
            QualificationReasonV1::DependenceModelMismatch,
        );
    }
    if observation.score_definition != envelope.score_definition {
        return decision(
            QualificationStatusV1::NotQualified,
            QualificationReasonV1::ScoreDefinitionMismatch,
        );
    }
    if observation.block_length != envelope.block_length {
        return decision(
            QualificationStatusV1::NotQualified,
            QualificationReasonV1::BlockLengthMismatch,
        );
    }
    if observation.alpha_basis_points != envelope.alpha_basis_points {
        return decision(
            QualificationStatusV1::NotQualified,
            QualificationReasonV1::AlphaMismatch,
        );
    }

    match observation.model_assessment {
        ModelAssessmentV1::AdequateWithinDeclaredModel => {}
        ModelAssessmentV1::Rejected => {
            return decision(
                QualificationStatusV1::NotQualified,
                QualificationReasonV1::DependenceModelRejected,
            );
        }
        ModelAssessmentV1::Inconclusive => {
            return decision(
                QualificationStatusV1::Inconclusive,
                QualificationReasonV1::DiagnosticAssessmentInconclusive,
            );
        }
    }

    if observation.diagnostic_sample_count < envelope.minimum_diagnostic_samples {
        return decision(
            QualificationStatusV1::Inconclusive,
            QualificationReasonV1::InsufficientDiagnosticSamples,
        );
    }
    if !is_canonical_sha256(&observation.evidence_sha256) {
        return decision(
            QualificationStatusV1::Inconclusive,
            QualificationReasonV1::InvalidObservationDigest,
        );
    }
    if !observation.rho_lower_bound.is_finite()
        || !observation.rho_upper_bound.is_finite()
        || !(-1.0..=1.0).contains(&observation.rho_lower_bound)
        || !(-1.0..=1.0).contains(&observation.rho_upper_bound)
        || observation.rho_lower_bound > observation.rho_upper_bound
    {
        return decision(
            QualificationStatusV1::Inconclusive,
            QualificationReasonV1::InvalidDependenceInterval,
        );
    }

    if observation.rho_lower_bound >= envelope.rho_lower_bound
        && observation.rho_upper_bound <= envelope.rho_upper_bound
    {
        return decision(
            QualificationStatusV1::QualifiedWithinEnvelope,
            QualificationReasonV1::EntireIntervalInsideFrozenEnvelope,
        );
    }

    if observation.rho_upper_bound < envelope.rho_lower_bound
        || observation.rho_lower_bound > envelope.rho_upper_bound
    {
        return decision(
            QualificationStatusV1::NotQualified,
            QualificationReasonV1::IntervalEntirelyOutsideEnvelope,
        );
    }

    decision(
        QualificationStatusV1::Inconclusive,
        QualificationReasonV1::IntervalOverlapsEnvelopeBoundary,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope() -> ValidatedDependenceEnvelopeV1 {
        ValidatedDependenceEnvelopeV1 {
            schema_version: CONTRACT_VERSION_V1,
            envelope_id: "sdcb-ar1-v1".into(),
            scope_id: "sensor-network-A".into(),
            marginal_family: MarginalFamilyV1::Gaussian,
            dependence_model: DependenceModelV1::StationaryGaussianAr1,
            score_definition: ScoreDefinitionV1::BlockMaximumAbsoluteResidual,
            block_length: 8,
            alpha_basis_points: 1_000,
            rho_lower_bound: 0.10,
            rho_upper_bound: 0.70,
            minimum_diagnostic_samples: 500,
            validation_receipt_sha256: "a".repeat(64),
            heldout_regime_set_sha256: "b".repeat(64),
        }
    }

    fn observation() -> DependenceObservationV1 {
        DependenceObservationV1 {
            scope_id: "sensor-network-A".into(),
            marginal_family: MarginalFamilyV1::Gaussian,
            dependence_model: DependenceModelV1::StationaryGaussianAr1,
            score_definition: ScoreDefinitionV1::BlockMaximumAbsoluteResidual,
            block_length: 8,
            alpha_basis_points: 1_000,
            rho_lower_bound: 0.25,
            rho_upper_bound: 0.55,
            diagnostic_sample_count: 2_000,
            model_assessment: ModelAssessmentV1::AdequateWithinDeclaredModel,
            evidence_sha256: "c".repeat(64),
        }
    }

    fn classify(
        env: &ValidatedDependenceEnvelopeV1,
        obs: &DependenceObservationV1,
    ) -> QualificationDecisionV1 {
        classify_dependence_regime_v1(env, obs)
    }

    #[test]
    fn wholly_contained_interval_qualifies_with_exact_metadata() {
        assert_eq!(
            classify(&envelope(), &observation()),
            QualificationDecisionV1 {
                status: QualificationStatusV1::QualifiedWithinEnvelope,
                reason: QualificationReasonV1::EntireIntervalInsideFrozenEnvelope,
            }
        );
    }

    #[test]
    fn exact_envelope_boundaries_are_inclusive() {
        let mut obs = observation();
        obs.rho_lower_bound = 0.10;
        obs.rho_upper_bound = 0.70;
        assert_eq!(
            classify(&envelope(), &obs).status,
            QualificationStatusV1::QualifiedWithinEnvelope
        );
    }

    #[test]
    fn entirely_outside_interval_is_not_qualified() {
        let mut obs = observation();
        obs.rho_lower_bound = 0.75;
        obs.rho_upper_bound = 0.90;
        assert_eq!(
            classify(&envelope(), &obs),
            QualificationDecisionV1 {
                status: QualificationStatusV1::NotQualified,
                reason: QualificationReasonV1::IntervalEntirelyOutsideEnvelope,
            }
        );
    }

    #[test]
    fn interval_partially_overlapping_envelope_is_inconclusive() {
        let mut obs = observation();
        obs.rho_lower_bound = 0.65;
        obs.rho_upper_bound = 0.78;
        assert_eq!(
            classify(&envelope(), &obs),
            QualificationDecisionV1 {
                status: QualificationStatusV1::Inconclusive,
                reason: QualificationReasonV1::IntervalOverlapsEnvelopeBoundary,
            }
        );
    }

    #[test]
    fn model_or_score_mismatch_cannot_inherit_calibration() {
        let env = envelope();
        let mut obs = observation();
        obs.block_length = 32;
        assert_eq!(
            classify(&env, &obs).reason,
            QualificationReasonV1::BlockLengthMismatch
        );

        let mut obs = observation();
        obs.score_definition = ScoreDefinitionV1::BlockMaximumAbsoluteResidual;
        obs.marginal_family = MarginalFamilyV1::StandardizedStudentT5CommonScale;
        assert_eq!(
            classify(&env, &obs).reason,
            QualificationReasonV1::MarginalFamilyMismatch
        );
    }

    #[test]
    fn inadequate_or_inconclusive_model_diagnostics_never_qualify() {
        let mut obs = observation();
        obs.model_assessment = ModelAssessmentV1::Rejected;
        assert_eq!(
            classify(&envelope(), &obs).status,
            QualificationStatusV1::NotQualified
        );

        let mut obs = observation();
        obs.model_assessment = ModelAssessmentV1::Inconclusive;
        assert_eq!(
            classify(&envelope(), &obs).status,
            QualificationStatusV1::Inconclusive
        );
    }

    #[test]
    fn insufficient_samples_or_missing_digest_is_inconclusive() {
        let mut obs = observation();
        obs.diagnostic_sample_count = 499;
        assert_eq!(
            classify(&envelope(), &obs).reason,
            QualificationReasonV1::InsufficientDiagnosticSamples
        );

        let mut obs = observation();
        obs.evidence_sha256 = "not-a-digest".into();
        assert_eq!(
            classify(&envelope(), &obs).reason,
            QualificationReasonV1::InvalidObservationDigest
        );
    }

    #[test]
    fn marginal_family_and_dependence_model_must_form_a_supported_pair() {
        let mut env = envelope();
        env.marginal_family = MarginalFamilyV1::StandardizedStudentT5CommonScale;
        assert_eq!(
            classify(&env, &observation()).reason,
            QualificationReasonV1::InvalidEnvelopeContract
        );
    }

    #[test]
    fn invalid_envelope_and_invalid_observation_interval_fail_closed() {
        let mut env = envelope();
        env.rho_lower_bound = 0.8;
        env.rho_upper_bound = 0.2;
        assert_eq!(
            classify(&env, &observation()).reason,
            QualificationReasonV1::InvalidEnvelopeContract
        );

        let mut obs = observation();
        obs.rho_lower_bound = f64::NAN;
        assert_eq!(
            classify(&envelope(), &obs).reason,
            QualificationReasonV1::InvalidDependenceInterval
        );
    }

    #[test]
    fn scope_model_and_alpha_mismatches_are_rejected() {
        let env = envelope();
        let mut obs = observation();
        obs.scope_id = "other-scope".into();
        assert_eq!(
            classify(&env, &obs).reason,
            QualificationReasonV1::ScopeMismatch
        );

        let mut obs = observation();
        obs.alpha_basis_points = 500;
        assert_eq!(
            classify(&env, &obs).reason,
            QualificationReasonV1::AlphaMismatch
        );

        let mut obs = observation();
        obs.dependence_model = DependenceModelV1::Ar1WithStudentT5CommonScale;
        assert_eq!(
            classify(&env, &obs).reason,
            QualificationReasonV1::DependenceModelMismatch
        );
    }
}
