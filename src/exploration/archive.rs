// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Confidence-qualified Pareto archive for generativity evidence.
//!
//! The archive preserves trade-offs instead of collapsing the generativity vector into
//! a scalar objective. Callers provide niche identities; Pareto comparison happens only
//! within the same niche so distinct classes of viable solutions remain represented.
//!
//! This module is descriptive/selection-support only. It does not execute candidates or
//! grant authority to any archived entry.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};

use super::generativity::{
    GenerativityEstimate, GenerativityValidationError, GenerativityVector,
};

/// Result of comparing two qualified generativity vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParetoRelation {
    /// Left vector is no worse on every dimension and better on at least one.
    Dominates,
    /// Right vector is no worse on every dimension and better on at least one.
    DominatedBy,
    /// All compared values are equivalent within epsilon.
    Equivalent,
    /// Each vector has at least one meaningful advantage.
    Incomparable,
    /// At least one required estimate is below the configured confidence floor.
    InsufficientEvidence,
}

/// Explicit comparison policy. This is not an objective weighting.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ParetoPolicy {
    /// Minimum confidence required for every compared dimension.
    pub min_confidence: f64,
    /// Differences at or below epsilon are treated as equivalent.
    pub epsilon: f64,
}

impl Default for ParetoPolicy {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            epsilon: 1e-9,
        }
    }
}

impl ParetoPolicy {
    pub fn new(min_confidence: f64, epsilon: f64) -> Result<Self, ArchiveValidationError> {
        let policy = Self {
            min_confidence,
            epsilon,
        };
        policy.validate()?;
        Ok(policy)
    }

    pub fn validate(&self) -> Result<(), ArchiveValidationError> {
        if !self.min_confidence.is_finite() || !(0.0..=1.0).contains(&self.min_confidence) {
            return Err(ArchiveValidationError::InvalidConfidenceFloor(
                self.min_confidence,
            ));
        }
        if !self.epsilon.is_finite() || self.epsilon < 0.0 {
            return Err(ArchiveValidationError::InvalidEpsilon(self.epsilon));
        }
        Ok(())
    }
}

/// Candidate retained in a niche-local Pareto front.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchiveEntry {
    /// Stable identity for this candidate/evidence record.
    pub entry_id: String,
    /// Caller-defined niche identity, e.g. `low-water`, `repairable`, `portable`.
    pub niche: String,
    /// Evidence vector used for Pareto comparison.
    pub vector: GenerativityVector,
}

impl ArchiveEntry {
    pub fn new(
        entry_id: impl Into<String>,
        niche: impl Into<String>,
        vector: GenerativityVector,
    ) -> Self {
        Self {
            entry_id: entry_id.into(),
            niche: niche.into(),
            vector,
        }
    }

    pub fn validate(&self) -> Result<(), ArchiveValidationError> {
        if self.entry_id.trim().is_empty() {
            return Err(ArchiveValidationError::EmptyField("entry_id"));
        }
        if self.niche.trim().is_empty() {
            return Err(ArchiveValidationError::EmptyField("niche"));
        }
        self.vector.validate()?;
        Ok(())
    }
}

/// Outcome of attempting to add an entry to the archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchiveInsertOutcome {
    /// Candidate was retained. Any listed entries were Pareto-dominated by it.
    Inserted { removed_entry_ids: Vec<String> },
    /// Candidate was dominated by one or more qualified entries in the same niche.
    RejectedDominated { by_entry_ids: Vec<String> },
    /// Candidate cannot enter the qualified front because its confidence is insufficient.
    RejectedInsufficientEvidence,
    /// Entry identity already exists in the archive.
    RejectedDuplicateId,
}

/// A niche-preserving collection of Pareto fronts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerativityParetoArchive {
    policy: ParetoPolicy,
    entries: Vec<ArchiveEntry>,
}

impl GenerativityParetoArchive {
    pub fn new(policy: ParetoPolicy) -> Result<Self, ArchiveValidationError> {
        policy.validate()?;
        Ok(Self {
            policy,
            entries: Vec::new(),
        })
    }

    pub fn policy(&self) -> ParetoPolicy {
        self.policy
    }

    pub fn entries(&self) -> &[ArchiveEntry] {
        &self.entries
    }

    /// Entries retained for a caller-defined niche.
    pub fn entries_in_niche<'a>(&'a self, niche: &'a str) -> impl Iterator<Item = &'a ArchiveEntry> {
        self.entries.iter().filter(move |entry| entry.niche == niche)
    }

    /// Insert into the niche-local Pareto front.
    ///
    /// Low-confidence entries are not admitted to the qualified front. Entries in other
    /// niches do not dominate one another; this is the diversity-preserving boundary.
    pub fn insert(
        &mut self,
        candidate: ArchiveEntry,
    ) -> Result<ArchiveInsertOutcome, ArchiveValidationError> {
        candidate.validate()?;
        self.policy.validate()?;

        if self
            .entries
            .iter()
            .any(|entry| entry.entry_id == candidate.entry_id)
        {
            return Ok(ArchiveInsertOutcome::RejectedDuplicateId);
        }

        if !is_confidence_qualified(&candidate.vector, self.policy.min_confidence) {
            return Ok(ArchiveInsertOutcome::RejectedInsufficientEvidence);
        }

        let mut dominators = Vec::new();
        let mut dominated_indices = Vec::new();

        for (index, existing) in self.entries.iter().enumerate() {
            if existing.niche != candidate.niche {
                continue;
            }

            match compare_vectors(&candidate.vector, &existing.vector, self.policy)? {
                ParetoRelation::DominatedBy => dominators.push(existing.entry_id.clone()),
                ParetoRelation::Dominates => dominated_indices.push(index),
                ParetoRelation::Equivalent
                | ParetoRelation::Incomparable
                | ParetoRelation::InsufficientEvidence => {}
            }
        }

        if !dominators.is_empty() {
            return Ok(ArchiveInsertOutcome::RejectedDominated {
                by_entry_ids: dominators,
            });
        }

        let removed_entry_ids = dominated_indices
            .iter()
            .map(|&index| self.entries[index].entry_id.clone())
            .collect::<Vec<_>>();

        for index in dominated_indices.into_iter().rev() {
            self.entries.remove(index);
        }

        self.entries.push(candidate);
        Ok(ArchiveInsertOutcome::Inserted { removed_entry_ids })
    }
}

impl Default for GenerativityParetoArchive {
    fn default() -> Self {
        Self::new(ParetoPolicy::default()).expect("default Pareto policy is valid")
    }
}

/// Compare two generativity vectors under an explicit confidence/epsilon policy.
///
/// Positive-capacity dimensions are maximized; risk dimensions are minimized.
/// No dimension is assigned a cross-dimension weight.
pub fn compare_vectors(
    left: &GenerativityVector,
    right: &GenerativityVector,
    policy: ParetoPolicy,
) -> Result<ParetoRelation, ArchiveValidationError> {
    left.validate()?;
    right.validate()?;
    policy.validate()?;

    if !is_confidence_qualified(left, policy.min_confidence)
        || !is_confidence_qualified(right, policy.min_confidence)
    {
        return Ok(ParetoRelation::InsufficientEvidence);
    }

    let left_objectives = objective_values(left);
    let right_objectives = objective_values(right);

    let mut left_better = false;
    let mut right_better = false;

    for (left_value, right_value) in left_objectives.iter().zip(right_objectives.iter()) {
        let delta = left_value - right_value;
        if delta > policy.epsilon {
            left_better = true;
        } else if delta < -policy.epsilon {
            right_better = true;
        }

        if left_better && right_better {
            return Ok(ParetoRelation::Incomparable);
        }
    }

    Ok(match (left_better, right_better) {
        (true, false) => ParetoRelation::Dominates,
        (false, true) => ParetoRelation::DominatedBy,
        (false, false) => ParetoRelation::Equivalent,
        (true, true) => ParetoRelation::Incomparable,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub enum ArchiveValidationError {
    EmptyField(&'static str),
    InvalidConfidenceFloor(f64),
    InvalidEpsilon(f64),
    Generativity(GenerativityValidationError),
}

impl From<GenerativityValidationError> for ArchiveValidationError {
    fn from(value: GenerativityValidationError) -> Self {
        Self::Generativity(value)
    }
}

impl std::fmt::Display for ArchiveValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "required field is empty: {field}"),
            Self::InvalidConfidenceFloor(value) => {
                write!(f, "minimum confidence must be finite and within [0, 1], got {value}")
            }
            Self::InvalidEpsilon(value) => {
                write!(f, "epsilon must be finite and non-negative, got {value}")
            }
            Self::Generativity(err) => write!(f, "invalid generativity vector: {err}"),
        }
    }
}

impl std::error::Error for ArchiveValidationError {}

fn is_confidence_qualified(vector: &GenerativityVector, min_confidence: f64) -> bool {
    estimates(vector)
        .iter()
        .all(|estimate| estimate.confidence >= min_confidence)
}

fn estimates(vector: &GenerativityVector) -> [GenerativityEstimate; 11] {
    [
        vector.immediate_utility,
        vector.epistemic_gain,
        vector.option_value,
        vector.diversity,
        vector.capability_gain,
        vector.diffusion,
        vector.commons_gain,
        vector.regeneration,
        vector.dependency_risk,
        vector.concentration_risk,
        vector.irreversibility_risk,
    ]
}

/// Convert all dimensions to a "higher is better" orientation for Pareto comparison.
/// Risk dimensions are negated; they are not weighted against positive dimensions.
fn objective_values(vector: &GenerativityVector) -> [f64; 11] {
    [
        vector.immediate_utility.value,
        vector.epistemic_gain.value,
        vector.option_value.value,
        vector.diversity.value,
        vector.capability_gain.value,
        vector.diffusion.value,
        vector.commons_gain.value,
        vector.regeneration.value,
        -vector.dependency_risk.value,
        -vector.concentration_risk.value,
        -vector.irreversibility_risk.value,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(value: f64) -> GenerativityEstimate {
        GenerativityEstimate::new(value, 0.9).unwrap()
    }

    fn vector(positive: f64, risk: f64) -> GenerativityVector {
        GenerativityVector {
            immediate_utility: e(positive),
            epistemic_gain: e(positive),
            option_value: e(positive),
            diversity: e(positive),
            capability_gain: e(positive),
            diffusion: e(positive),
            commons_gain: e(positive),
            regeneration: e(positive),
            dependency_risk: e(risk),
            concentration_risk: e(risk),
            irreversibility_risk: e(risk),
        }
    }

    #[test]
    fn stronger_capacity_and_lower_risk_dominates() {
        let better = vector(0.8, 0.2);
        let worse = vector(0.6, 0.4);
        assert_eq!(
            compare_vectors(&better, &worse, ParetoPolicy::default()).unwrap(),
            ParetoRelation::Dominates
        );
    }

    #[test]
    fn tradeoffs_remain_incomparable() {
        let mut a = vector(0.7, 0.3);
        let mut b = vector(0.7, 0.3);
        a.option_value = e(0.9);
        a.regeneration = e(0.4);
        b.option_value = e(0.5);
        b.regeneration = e(0.9);

        assert_eq!(
            compare_vectors(&a, &b, ParetoPolicy::default()).unwrap(),
            ParetoRelation::Incomparable
        );
    }

    #[test]
    fn low_confidence_fails_qualification() {
        let mut a = vector(0.9, 0.1);
        a.option_value = GenerativityEstimate::new(0.9, 0.2).unwrap();
        let b = vector(0.5, 0.5);

        assert_eq!(
            compare_vectors(&a, &b, ParetoPolicy::default()).unwrap(),
            ParetoRelation::InsufficientEvidence
        );
    }

    #[test]
    fn archive_rejects_dominated_candidate_in_same_niche() {
        let mut archive = GenerativityParetoArchive::default();
        archive
            .insert(ArchiveEntry::new("better", "low-water", vector(0.8, 0.2)))
            .unwrap();

        let outcome = archive
            .insert(ArchiveEntry::new("worse", "low-water", vector(0.6, 0.4)))
            .unwrap();
        assert_eq!(
            outcome,
            ArchiveInsertOutcome::RejectedDominated {
                by_entry_ids: vec!["better".into()]
            }
        );
    }

    #[test]
    fn stronger_candidate_prunes_dominated_entry_in_same_niche() {
        let mut archive = GenerativityParetoArchive::default();
        archive
            .insert(ArchiveEntry::new("old", "repairable", vector(0.5, 0.5)))
            .unwrap();

        let outcome = archive
            .insert(ArchiveEntry::new("new", "repairable", vector(0.8, 0.2)))
            .unwrap();
        assert_eq!(
            outcome,
            ArchiveInsertOutcome::Inserted {
                removed_entry_ids: vec!["old".into()]
            }
        );
        assert_eq!(archive.entries_in_niche("repairable").count(), 1);
    }

    #[test]
    fn niches_preserve_diverse_solution_classes() {
        let mut archive = GenerativityParetoArchive::default();
        archive
            .insert(ArchiveEntry::new("high", "grid-scale", vector(0.9, 0.1)))
            .unwrap();
        let outcome = archive
            .insert(ArchiveEntry::new("portable", "portable", vector(0.4, 0.5)))
            .unwrap();

        assert!(matches!(outcome, ArchiveInsertOutcome::Inserted { .. }));
        assert_eq!(archive.entries().len(), 2);
    }

    #[test]
    fn low_confidence_candidate_is_not_admitted() {
        let mut archive = GenerativityParetoArchive::default();
        let mut uncertain = vector(0.9, 0.1);
        uncertain.epistemic_gain = GenerativityEstimate::new(0.9, 0.1).unwrap();

        assert_eq!(
            archive
                .insert(ArchiveEntry::new("uncertain", "research", uncertain))
                .unwrap(),
            ArchiveInsertOutcome::RejectedInsufficientEvidence
        );
        assert!(archive.entries().is_empty());
    }

    #[test]
    fn duplicate_identity_is_rejected_without_replacement() {
        let mut archive = GenerativityParetoArchive::default();
        archive
            .insert(ArchiveEntry::new("same", "niche-a", vector(0.5, 0.5)))
            .unwrap();
        assert_eq!(
            archive
                .insert(ArchiveEntry::new("same", "niche-b", vector(0.9, 0.1)))
                .unwrap(),
            ArchiveInsertOutcome::RejectedDuplicateId
        );
        assert_eq!(archive.entries().len(), 1);
    }

    #[test]
    fn policy_rejects_invalid_thresholds() {
        assert!(ParetoPolicy::new(1.1, 0.0).is_err());
        assert!(ParetoPolicy::new(0.5, -0.1).is_err());
        assert!(ParetoPolicy::new(f64::NAN, 0.0).is_err());
    }
}
