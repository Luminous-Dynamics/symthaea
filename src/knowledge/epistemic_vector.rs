// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multidimensional claim uncertainty.
//!
//! A single confidence scalar cannot distinguish reducible ignorance from
//! irreducible variability, a potentially wrong ontology, or distribution
//! shift. This module keeps those dimensions separate and represents an
//! unassessed dimension as `None` rather than silently treating it as zero.
//!
//! It is intentionally descriptive. No function in this module collapses the
//! vector into a truth probability or automatically changes knowledge state.

use super::claim_evidence::{ClaimId, EpistemicLedger, EvidenceId};
use std::error::Error;
use std::fmt;

/// The four uncertainty dimensions tracked for a claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UncertaintyDimension {
    /// Reducible uncertainty caused by insufficient evidence or knowledge.
    Epistemic,
    /// Irreducible outcome variability/noise under the represented conditions.
    Aleatoric,
    /// Uncertainty that the concepts/model structure themselves are appropriate.
    Ontological,
    /// Uncertainty introduced because current conditions differ from evidence conditions.
    DistributionShift,
}

/// A finite uncertainty value in the closed interval [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct UncertaintyValue(f64);

impl UncertaintyValue {
    pub fn new(value: f64) -> Result<Self, UncertaintyError> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(UncertaintyError::OutOfRange(value));
        }
        Ok(Self(value))
    }

    pub fn get(self) -> f64 {
        self.0
    }
}

/// Claim uncertainty kept as independent optional dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EpistemicVector {
    pub epistemic: Option<UncertaintyValue>,
    pub aleatoric: Option<UncertaintyValue>,
    pub ontological: Option<UncertaintyValue>,
    pub distribution_shift: Option<UncertaintyValue>,
}

impl EpistemicVector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(
        mut self,
        dimension: UncertaintyDimension,
        value: f64,
    ) -> Result<Self, UncertaintyError> {
        self.set(dimension, value)?;
        Ok(self)
    }

    pub fn set(
        &mut self,
        dimension: UncertaintyDimension,
        value: f64,
    ) -> Result<(), UncertaintyError> {
        let value = UncertaintyValue::new(value)?;
        match dimension {
            UncertaintyDimension::Epistemic => self.epistemic = Some(value),
            UncertaintyDimension::Aleatoric => self.aleatoric = Some(value),
            UncertaintyDimension::Ontological => self.ontological = Some(value),
            UncertaintyDimension::DistributionShift => self.distribution_shift = Some(value),
        }
        Ok(())
    }

    pub fn clear(&mut self, dimension: UncertaintyDimension) {
        match dimension {
            UncertaintyDimension::Epistemic => self.epistemic = None,
            UncertaintyDimension::Aleatoric => self.aleatoric = None,
            UncertaintyDimension::Ontological => self.ontological = None,
            UncertaintyDimension::DistributionShift => self.distribution_shift = None,
        }
    }

    pub fn get(&self, dimension: UncertaintyDimension) -> Option<UncertaintyValue> {
        match dimension {
            UncertaintyDimension::Epistemic => self.epistemic,
            UncertaintyDimension::Aleatoric => self.aleatoric,
            UncertaintyDimension::Ontological => self.ontological,
            UncertaintyDimension::DistributionShift => self.distribution_shift,
        }
    }

    /// Number of uncertainty dimensions that have actually been assessed.
    pub fn assessed_dimension_count(&self) -> usize {
        [
            self.epistemic,
            self.aleatoric,
            self.ontological,
            self.distribution_shift,
        ]
        .into_iter()
        .flatten()
        .count()
    }

    /// Highest *assessed* uncertainty dimension.
    ///
    /// Returns `None` when no dimensions have been assessed. This is not an
    /// aggregate confidence or truth estimate; it is only a diagnostic for which
    /// known uncertainty component is currently largest.
    pub fn dominant_assessed_dimension(&self) -> Option<(UncertaintyDimension, UncertaintyValue)> {
        let candidates = [
            (UncertaintyDimension::Epistemic, self.epistemic),
            (UncertaintyDimension::Aleatoric, self.aleatoric),
            (UncertaintyDimension::Ontological, self.ontological),
            (
                UncertaintyDimension::DistributionShift,
                self.distribution_shift,
            ),
        ];

        candidates
            .into_iter()
            .filter_map(|(dimension, value)| value.map(|value| (dimension, value)))
            .max_by(|(_, left), (_, right)| {
                left.get()
                    .partial_cmp(&right.get())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

/// A claim-scoped uncertainty assessment with explicit evidence basis.
#[derive(Debug, Clone, PartialEq)]
pub struct ClaimUncertaintyAssessment {
    pub claim_id: ClaimId,
    pub vector: EpistemicVector,
    /// Evidence records actually used to produce this assessment.
    pub basis_evidence_ids: Vec<EvidenceId>,
    pub assessed_at_cycle: u64,
}

impl ClaimUncertaintyAssessment {
    /// Construct an assessment only when the claim exists and every evidence
    /// basis record exists and belongs to that same claim.
    pub fn new(
        ledger: &EpistemicLedger,
        claim_id: ClaimId,
        vector: EpistemicVector,
        basis_evidence_ids: Vec<EvidenceId>,
        assessed_at_cycle: u64,
    ) -> Result<Self, UncertaintyError> {
        if ledger.claim(claim_id).is_none() {
            return Err(UncertaintyError::UnknownClaim(claim_id));
        }

        for evidence_id in &basis_evidence_ids {
            let evidence = ledger
                .evidence(*evidence_id)
                .ok_or(UncertaintyError::UnknownEvidence(*evidence_id))?;
            if evidence.claim_id != claim_id {
                return Err(UncertaintyError::EvidenceForDifferentClaim {
                    evidence_id: *evidence_id,
                    expected_claim: claim_id,
                    actual_claim: evidence.claim_id,
                });
            }
        }

        Ok(Self {
            claim_id,
            vector,
            basis_evidence_ids,
            assessed_at_cycle,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UncertaintyError {
    OutOfRange(f64),
    UnknownClaim(ClaimId),
    UnknownEvidence(EvidenceId),
    EvidenceForDifferentClaim {
        evidence_id: EvidenceId,
        expected_claim: ClaimId,
        actual_claim: ClaimId,
    },
}

impl fmt::Display for UncertaintyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange(value) => {
                write!(f, "uncertainty value must be finite and within [0, 1], got {value}")
            }
            Self::UnknownClaim(id) => write!(f, "unknown claim id {}", id.0),
            Self::UnknownEvidence(id) => write!(f, "unknown evidence id {}", id.0),
            Self::EvidenceForDifferentClaim {
                evidence_id,
                expected_claim,
                actual_claim,
            } => write!(
                f,
                "evidence {} belongs to claim {}, not claim {}",
                evidence_id.0, actual_claim.0, expected_claim.0
            ),
        }
    }
}

impl Error for UncertaintyError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::claim_evidence::{
        ClaimKind, EvidenceKind, EvidencePolarity,
    };

    #[test]
    fn unassessed_is_not_zero_uncertainty() {
        let mut vector = EpistemicVector::new();
        assert_eq!(vector.epistemic, None);
        assert_eq!(vector.assessed_dimension_count(), 0);

        vector
            .set(UncertaintyDimension::Epistemic, 0.0)
            .unwrap();
        assert_eq!(vector.epistemic.unwrap().get(), 0.0);
        assert_eq!(vector.assessed_dimension_count(), 1);
    }

    #[test]
    fn dimensions_remain_independent() {
        let vector = EpistemicVector::new()
            .with(UncertaintyDimension::Epistemic, 0.8)
            .unwrap()
            .with(UncertaintyDimension::Aleatoric, 0.2)
            .unwrap()
            .with(UncertaintyDimension::Ontological, 0.6)
            .unwrap();

        assert_eq!(vector.epistemic.unwrap().get(), 0.8);
        assert_eq!(vector.aleatoric.unwrap().get(), 0.2);
        assert_eq!(vector.ontological.unwrap().get(), 0.6);
        assert_eq!(vector.distribution_shift, None);
        assert_eq!(vector.assessed_dimension_count(), 3);
        assert_eq!(
            vector.dominant_assessed_dimension().unwrap().0,
            UncertaintyDimension::Epistemic
        );
    }

    #[test]
    fn invalid_uncertainty_values_fail_closed() {
        for value in [-0.01, 1.01, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert!(UncertaintyValue::new(value).is_err());
        }
        assert!(UncertaintyValue::new(0.0).is_ok());
        assert!(UncertaintyValue::new(1.0).is_ok());
    }

    #[test]
    fn assessment_basis_must_belong_to_claim() {
        let mut ledger = EpistemicLedger::new();
        let source = ledger
            .add_provenance("source", None, None, 1, vec![])
            .unwrap();
        let claim_a = ledger.add_claim("A", ClaimKind::Descriptive, None, None, 1);
        let claim_b = ledger.add_claim("B", ClaimKind::Descriptive, None, None, 1);
        let evidence_b = ledger
            .add_evidence(
                claim_b,
                EvidenceKind::Observation,
                EvidencePolarity::Supports,
                source,
                2,
                None,
                None,
            )
            .unwrap();

        let vector = EpistemicVector::new()
            .with(UncertaintyDimension::Epistemic, 0.5)
            .unwrap();
        let error = ClaimUncertaintyAssessment::new(
            &ledger,
            claim_a,
            vector,
            vec![evidence_b],
            3,
        )
        .unwrap_err();

        assert_eq!(
            error,
            UncertaintyError::EvidenceForDifferentClaim {
                evidence_id: evidence_b,
                expected_claim: claim_a,
                actual_claim: claim_b,
            }
        );
    }

    #[test]
    fn assessment_keeps_explicit_evidence_basis() {
        let mut ledger = EpistemicLedger::new();
        let source = ledger
            .add_provenance("experiment", None, None, 1, vec![])
            .unwrap();
        let claim = ledger.add_claim("A causes B", ClaimKind::Causal, None, None, 1);
        let evidence = ledger
            .add_evidence(
                claim,
                EvidenceKind::Intervention,
                EvidencePolarity::Supports,
                source,
                2,
                Some("lab".into()),
                Some("controlled intervention".into()),
            )
            .unwrap();

        let vector = EpistemicVector::new()
            .with(UncertaintyDimension::Epistemic, 0.25)
            .unwrap()
            .with(UncertaintyDimension::DistributionShift, 0.7)
            .unwrap();
        let assessment = ClaimUncertaintyAssessment::new(
            &ledger,
            claim,
            vector,
            vec![evidence],
            3,
        )
        .unwrap();

        assert_eq!(assessment.claim_id, claim);
        assert_eq!(assessment.basis_evidence_ids, vec![evidence]);
        assert_eq!(
            assessment
                .vector
                .get(UncertaintyDimension::DistributionShift)
                .unwrap()
                .get(),
            0.7
        );
    }
}
