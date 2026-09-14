// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! PARADOX-001: production-neutral structured-inconsistency observatory.
//!
//! This crate deliberately contains no cognitive-loop integration and no
//! behavior authority. It provides a typed, deterministic measurement surface
//! for later matched experiments under PARADOX-000.
//!
//! ## Claim boundary
//!
//! The observatory measures relationships among supplied normalized signals.
//! Its outputs are not measurements or classifications of consciousness,
//! sentience, phenomenal experience, IIT Phi, intelligence, or moral status.

/// Closed vocabulary for the kind of inconsistency being studied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InconsistencyKind {
    PredictionError,
    Ambiguity,
    ActionConflict,
    EvidenceContradiction,
    Underdetermination,
    SelfReferentialConflict,
    OntologyFailure,
    FormalParadox,
}

/// Four-valued evidence-polarity state.
///
/// `SupportsBoth` means only that the supplied evidence supports incompatible
/// propositions. It is not a claim that reality is logically inconsistent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidencePolarity {
    Neither,
    SupportsPropositionOnly,
    SupportsNegationOnly,
    SupportsBoth,
}

/// Explicit resolution state supplied by the experimental harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionState {
    Stable,
    TransientConflict,
    PersistentUnresolved,
    ResolvedWithoutRevision,
    ResolvedByRepresentationRevision,
    IrreducibleUnderCurrentModel,
}

/// Validation failure for normalized observatory inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationError {
    NonFinite(&'static str),
    OutOfRange(&'static str),
}

impl std::fmt::Display for ObservationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite(name) => write!(f, "{name} must be finite"),
            Self::OutOfRange(name) => write!(f, "{name} must be within [0, 1]"),
        }
    }
}

impl std::error::Error for ObservationError {}

/// Normalized evidence support retained independently for a proposition and its
/// explicit negation/incompatible counterpart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvidenceSupport {
    pub proposition: f64,
    pub negation: f64,
}

impl EvidenceSupport {
    pub fn new(proposition: f64, negation: f64) -> Result<Self, ObservationError> {
        validate_unit("evidence_support.proposition", proposition)?;
        validate_unit("evidence_support.negation", negation)?;
        Ok(Self {
            proposition,
            negation,
        })
    }

    /// Classifies only exact absence versus presence of support.
    ///
    /// There is intentionally no tunable or hidden threshold in PARADOX-001.
    pub fn polarity(self) -> EvidencePolarity {
        match (self.proposition > 0.0, self.negation > 0.0) {
            (false, false) => EvidencePolarity::Neither,
            (true, false) => EvidencePolarity::SupportsPropositionOnly,
            (false, true) => EvidencePolarity::SupportsNegationOnly,
            (true, true) => EvidencePolarity::SupportsBoth,
        }
    }
}

/// Explicit normalized inputs to the observatory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuredInconsistencyInput {
    pub kind: InconsistencyKind,
    /// Model/world mismatch. Kept independent from internal disagreement.
    pub external_surprise: f64,
    /// Disagreement among simultaneously admitted internal models/evidence.
    pub internal_disagreement: f64,
    /// Uncertainty under the experiment's separately declared semantics.
    pub uncertainty: f64,
    /// Persistence of the disagreement over the experiment's declared window.
    pub persistence: f64,
    /// Relevance of the inconsistency to the system's own state/model.
    pub self_referential_relevance: f64,
    pub evidence_support: EvidenceSupport,
    pub resolution: ResolutionState,
}

impl StructuredInconsistencyInput {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: InconsistencyKind,
        external_surprise: f64,
        internal_disagreement: f64,
        uncertainty: f64,
        persistence: f64,
        self_referential_relevance: f64,
        evidence_support: EvidenceSupport,
        resolution: ResolutionState,
    ) -> Result<Self, ObservationError> {
        validate_unit("external_surprise", external_surprise)?;
        validate_unit("internal_disagreement", internal_disagreement)?;
        validate_unit("uncertainty", uncertainty)?;
        validate_unit("persistence", persistence)?;
        validate_unit("self_referential_relevance", self_referential_relevance)?;

        Ok(Self {
            kind,
            external_surprise,
            internal_disagreement,
            uncertainty,
            persistence,
            self_referential_relevance,
            evidence_support,
            resolution,
        })
    }
}

/// Deterministic measurement report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuredInconsistencyReport {
    pub kind: InconsistencyKind,
    pub external_surprise: f64,
    pub internal_disagreement: f64,
    pub uncertainty: f64,
    pub persistence: f64,
    pub self_referential_relevance: f64,
    pub evidence_support: EvidenceSupport,
    pub evidence_polarity: EvidencePolarity,
    /// Descriptive complement of internal disagreement only.
    pub integration_coherence: f64,
    /// Conflict magnitude weighted by its persistence, independent of surprise.
    pub conflict_load: f64,
    /// Frozen descriptive index for experiment stratification only.
    ///
    /// This is not observed metacognition and must not be used as behavior
    /// authority. Later tranches compare this predictor against independently
    /// measured metacognitive activity.
    pub candidate_recruitment_index: f64,
    pub resolution: ResolutionState,
}

/// Stateless production-neutral observatory.
#[derive(Debug, Default, Clone, Copy)]
pub struct StructuredInconsistencyObservatory;

impl StructuredInconsistencyObservatory {
    pub const fn new() -> Self {
        Self
    }

    /// Produce a deterministic measurement report from validated explicit input.
    ///
    /// No external state is read or mutated.
    pub fn observe(&self, input: StructuredInconsistencyInput) -> StructuredInconsistencyReport {
        let integration_coherence = 1.0 - input.internal_disagreement;

        // Persistence increases the descriptive load of the same disagreement
        // without making external surprise part of internal conflict.
        let conflict_load =
            input.internal_disagreement * (0.5 + 0.5 * input.persistence);

        // Equal weights are frozen in PARADOX-001. This is intentionally a
        // transparent stratification statistic rather than a fitted model.
        let candidate_recruitment_index = (input.internal_disagreement
            + input.uncertainty
            + input.persistence
            + input.self_referential_relevance)
            / 4.0;

        StructuredInconsistencyReport {
            kind: input.kind,
            external_surprise: input.external_surprise,
            internal_disagreement: input.internal_disagreement,
            uncertainty: input.uncertainty,
            persistence: input.persistence,
            self_referential_relevance: input.self_referential_relevance,
            evidence_support: input.evidence_support,
            evidence_polarity: input.evidence_support.polarity(),
            integration_coherence,
            conflict_load,
            candidate_recruitment_index,
            resolution: input.resolution,
        }
    }
}

fn validate_unit(name: &'static str, value: f64) -> Result<(), ObservationError> {
    if !value.is_finite() {
        return Err(ObservationError::NonFinite(name));
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(ObservationError::OutOfRange(name));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn support(p: f64, n: f64) -> EvidenceSupport {
        EvidenceSupport::new(p, n).expect("valid support")
    }

    fn input(
        surprise: f64,
        disagreement: f64,
        uncertainty: f64,
        persistence: f64,
        self_reference: f64,
        evidence: EvidenceSupport,
    ) -> StructuredInconsistencyInput {
        StructuredInconsistencyInput::new(
            InconsistencyKind::EvidenceContradiction,
            surprise,
            disagreement,
            uncertainty,
            persistence,
            self_reference,
            evidence,
            ResolutionState::PersistentUnresolved,
        )
        .expect("valid input")
    }

    #[test]
    fn neutral_input_is_coherent_and_non_conflicting() {
        let report = StructuredInconsistencyObservatory::new().observe(input(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            support(0.0, 0.0),
        ));

        assert_eq!(report.evidence_polarity, EvidencePolarity::Neither);
        assert_eq!(report.integration_coherence, 1.0);
        assert_eq!(report.conflict_load, 0.0);
        assert_eq!(report.candidate_recruitment_index, 0.0);
    }

    #[test]
    fn surprise_and_internal_disagreement_are_independent() {
        let high_surprise = StructuredInconsistencyObservatory::new().observe(input(
            1.0,
            0.0,
            0.2,
            0.2,
            0.0,
            support(1.0, 0.0),
        ));
        let high_disagreement = StructuredInconsistencyObservatory::new().observe(input(
            0.0,
            1.0,
            0.2,
            0.2,
            0.0,
            support(1.0, 1.0),
        ));

        assert_eq!(high_surprise.conflict_load, 0.0);
        assert_eq!(high_surprise.integration_coherence, 1.0);
        assert_eq!(high_disagreement.external_surprise, 0.0);
        assert!(high_disagreement.conflict_load > high_surprise.conflict_load);
        assert!(
            high_disagreement.integration_coherence < high_surprise.integration_coherence
        );
    }

    #[test]
    fn persistence_increases_conflict_load_for_same_disagreement() {
        let transient = StructuredInconsistencyObservatory::new().observe(input(
            0.0,
            0.8,
            0.5,
            0.0,
            0.0,
            support(1.0, 1.0),
        ));
        let persistent = StructuredInconsistencyObservatory::new().observe(input(
            0.0,
            0.8,
            0.5,
            1.0,
            0.0,
            support(1.0, 1.0),
        ));

        assert!(persistent.conflict_load > transient.conflict_load);
        assert!(
            persistent.candidate_recruitment_index > transient.candidate_recruitment_index
        );
        assert_eq!(persistent.integration_coherence, transient.integration_coherence);
    }

    #[test]
    fn self_reference_changes_only_self_reference_sensitive_index() {
        let ordinary = StructuredInconsistencyObservatory::new().observe(input(
            0.1,
            0.7,
            0.6,
            0.8,
            0.0,
            support(1.0, 1.0),
        ));
        let reflexive = StructuredInconsistencyObservatory::new().observe(input(
            0.1,
            0.7,
            0.6,
            0.8,
            1.0,
            support(1.0, 1.0),
        ));

        assert_eq!(ordinary.external_surprise, reflexive.external_surprise);
        assert_eq!(ordinary.internal_disagreement, reflexive.internal_disagreement);
        assert_eq!(ordinary.integration_coherence, reflexive.integration_coherence);
        assert_eq!(ordinary.conflict_load, reflexive.conflict_load);
        assert!(
            reflexive.candidate_recruitment_index > ordinary.candidate_recruitment_index
        );
    }

    #[test]
    fn evidence_polarity_preserves_both_sides() {
        assert_eq!(support(0.0, 0.0).polarity(), EvidencePolarity::Neither);
        assert_eq!(
            support(0.3, 0.0).polarity(),
            EvidencePolarity::SupportsPropositionOnly
        );
        assert_eq!(
            support(0.0, 0.4).polarity(),
            EvidencePolarity::SupportsNegationOnly
        );
        assert_eq!(
            support(0.3, 0.4).polarity(),
            EvidencePolarity::SupportsBoth
        );
    }

    #[test]
    fn non_finite_and_out_of_range_inputs_fail_closed() {
        assert_eq!(
            EvidenceSupport::new(f64::NAN, 0.0),
            Err(ObservationError::NonFinite("evidence_support.proposition"))
        );
        assert_eq!(
            EvidenceSupport::new(0.0, 1.1),
            Err(ObservationError::OutOfRange("evidence_support.negation"))
        );
        assert_eq!(
            StructuredInconsistencyInput::new(
                InconsistencyKind::Ambiguity,
                f64::INFINITY,
                0.0,
                0.0,
                0.0,
                0.0,
                support(0.0, 0.0),
                ResolutionState::Stable,
            ),
            Err(ObservationError::NonFinite("external_surprise"))
        );
        assert_eq!(
            StructuredInconsistencyInput::new(
                InconsistencyKind::Ambiguity,
                0.0,
                -0.01,
                0.0,
                0.0,
                0.0,
                support(0.0, 0.0),
                ResolutionState::Stable,
            ),
            Err(ObservationError::OutOfRange("internal_disagreement"))
        );
    }

    #[test]
    fn repeated_observation_is_bit_deterministic() {
        let observatory = StructuredInconsistencyObservatory::new();
        let subject = input(0.2, 0.9, 0.7, 0.8, 0.4, support(0.8, 0.6));

        let first = observatory.observe(subject);
        let second = observatory.observe(subject);

        assert_eq!(first, second);
        assert_eq!(
            first.candidate_recruitment_index.to_bits(),
            second.candidate_recruitment_index.to_bits()
        );
        assert_eq!(first.conflict_load.to_bits(), second.conflict_load.to_bits());
    }

    #[test]
    fn higher_disagreement_can_lower_coherence_while_raising_recruitment_index() {
        let low = StructuredInconsistencyObservatory::new().observe(input(
            0.2,
            0.2,
            0.6,
            0.8,
            0.3,
            support(0.8, 0.2),
        ));
        let high = StructuredInconsistencyObservatory::new().observe(input(
            0.2,
            0.9,
            0.6,
            0.8,
            0.3,
            support(0.8, 0.8),
        ));

        assert!(high.integration_coherence < low.integration_coherence);
        assert!(high.candidate_recruitment_index > low.candidate_recruitment_index);
    }
}
