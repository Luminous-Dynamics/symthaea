// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Non-serializable deliberation receipts for physical-agency portfolios.
//!
//! `CandidatePortfolio` and its report-oriented `PortfolioOutcome` are ordinary
//! planning data. This module adds a stricter runtime boundary: a
//! [`SelectedCandidate`] can only be minted by evaluating a portfolio and then
//! selecting an id that actually survived onto its Pareto frontier.
//!
//! PA-08 additionally binds deliberation to an immutable world-state snapshot
//! reference. The target frame in the desired transition must match the frame
//! identified by that snapshot before a frontier receipt can be minted.
//!
//! PA-12 makes the snapshot digest scheme part of that immutable lineage. The
//! legacy two-argument constructor remains available for historical opaque
//! identifiers, but strict solver evidence can only consume snapshots explicitly
//! created with a supported cryptographic digest scheme.
//!
//! The receipt grants no execution authority. It exists solely to keep
//! deliberation lineage from being replaced by caller-assembled proposal or
//! world-state data at the later simulation-qualification boundary.

use crate::portfolio::{
    CandidateAssessment, CandidatePortfolio, PortfolioError, PortfolioOutcome, PortfolioPolicy,
};
use serde::{Deserialize, Serialize};
use symthaea_physical_effects::{AbstentionReason, DesiredTransition};
use thiserror::Error;

/// Digest semantics attached to an immutable world snapshot.
///
/// `LegacyOpaque` preserves the PA-08 compatibility surface for historical
/// identifiers that were stable names rather than cryptographic content
/// digests. It is deliberately ineligible for PA-12 strict solver evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotDigestAlgorithm {
    #[default]
    LegacyOpaque,
    Blake3,
    Sha256,
}

/// Immutable reference to the world/digital-twin state used for deliberation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorldSnapshotRef {
    frame_id: String,
    /// Missing algorithm metadata in historical PA-08 records is conservatively
    /// interpreted as `LegacyOpaque`, never as a cryptographic assertion.
    #[serde(default)]
    digest_algorithm: SnapshotDigestAlgorithm,
    snapshot_digest: String,
}

impl WorldSnapshotRef {
    /// Backward-compatible constructor for pre-PA-12 stable snapshot identities.
    ///
    /// These values remain valid deliberation lineage but are not eligible for
    /// the strict solver-evidence path because no cryptographic digest scheme is
    /// asserted.
    pub fn new(frame_id: impl Into<String>, snapshot_digest: impl Into<String>) -> Self {
        Self {
            frame_id: frame_id.into(),
            digest_algorithm: SnapshotDigestAlgorithm::LegacyOpaque,
            snapshot_digest: snapshot_digest.into(),
        }
    }

    /// Construct a snapshot with an explicit cryptographic content-digest
    /// scheme. Validation requires a 32-byte hexadecimal digest.
    pub fn cryptographic(
        frame_id: impl Into<String>,
        digest_algorithm: SnapshotDigestAlgorithm,
        snapshot_digest: impl Into<String>,
    ) -> Self {
        Self {
            frame_id: frame_id.into(),
            digest_algorithm,
            snapshot_digest: snapshot_digest.into(),
        }
    }

    pub fn frame_id(&self) -> &str {
        &self.frame_id
    }

    pub fn digest_algorithm(&self) -> SnapshotDigestAlgorithm {
        self.digest_algorithm
    }

    pub fn snapshot_digest(&self) -> &str {
        &self.snapshot_digest
    }

    pub fn validate(&self) -> Result<(), DeliberationError> {
        if self.frame_id.trim().is_empty() {
            return Err(DeliberationError::EmptySnapshotField(
                "world_snapshot.frame_id",
            ));
        }
        if self.snapshot_digest.trim().is_empty() {
            return Err(DeliberationError::EmptySnapshotField(
                "world_snapshot.snapshot_digest",
            ));
        }
        if self.digest_algorithm != SnapshotDigestAlgorithm::LegacyOpaque
            && (self.snapshot_digest.len() != 64
                || !self
                    .snapshot_digest
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit()))
        {
            return Err(DeliberationError::InvalidCryptographicSnapshotDigest);
        }
        Ok(())
    }
}

/// Pareto frontier produced by an actual portfolio evaluation.
///
/// Intentionally implements neither `Serialize` nor `Deserialize`.
#[derive(Debug, Clone, PartialEq)]
pub struct DeliberatedFrontier {
    transition: DesiredTransition,
    world_snapshot: WorldSnapshotRef,
    policy: PortfolioPolicy,
    candidates: Vec<CandidateAssessment>,
}

impl DeliberatedFrontier {
    pub fn transition(&self) -> &DesiredTransition {
        &self.transition
    }

    pub fn world_snapshot(&self) -> &WorldSnapshotRef {
        &self.world_snapshot
    }

    pub fn policy(&self) -> PortfolioPolicy {
        self.policy
    }

    pub fn candidates(&self) -> &[CandidateAssessment] {
        &self.candidates
    }

    /// Select one candidate that actually survived Pareto filtering.
    ///
    /// The resulting receipt remains deliberative only; it carries no solver,
    /// HAL, actuator, or physical-execution capability.
    pub fn select(&self, proposal_id: &str) -> Option<SelectedCandidate> {
        self.candidates
            .iter()
            .find(|candidate| candidate.proposal.id == proposal_id)
            .cloned()
            .map(|assessment| SelectedCandidate {
                transition: self.transition.clone(),
                world_snapshot: self.world_snapshot.clone(),
                policy: self.policy,
                assessment,
            })
    }
}

/// Non-serializable receipt that a candidate came from a specific evaluated
/// transition/policy/world-snapshot frontier.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedCandidate {
    transition: DesiredTransition,
    world_snapshot: WorldSnapshotRef,
    policy: PortfolioPolicy,
    assessment: CandidateAssessment,
}

impl SelectedCandidate {
    pub fn transition(&self) -> &DesiredTransition {
        &self.transition
    }

    pub fn world_snapshot(&self) -> &WorldSnapshotRef {
        &self.world_snapshot
    }

    pub fn policy(&self) -> PortfolioPolicy {
        self.policy
    }

    pub fn assessment(&self) -> &CandidateAssessment {
        &self.assessment
    }
}

/// Runtime deliberation result. A frontier is still not an execution decision.
#[derive(Debug, Clone, PartialEq)]
pub enum DeliberationOutcome {
    ParetoFrontier(DeliberatedFrontier),
    Abstain(AbstentionReason),
}

/// Evaluate an ordinary portfolio against one immutable world snapshot and
/// convert a surviving frontier into a non-serializable runtime receipt.
pub fn deliberate(
    portfolio: &CandidatePortfolio,
    world_snapshot: &WorldSnapshotRef,
    policy: PortfolioPolicy,
) -> Result<DeliberationOutcome, DeliberationError> {
    world_snapshot.validate()?;
    if portfolio.transition.target.frame_id != world_snapshot.frame_id {
        return Err(DeliberationError::SnapshotFrameMismatch {
            target_frame: portfolio.transition.target.frame_id.clone(),
            snapshot_frame: world_snapshot.frame_id.clone(),
        });
    }

    match portfolio
        .evaluate(policy)
        .map_err(DeliberationError::Portfolio)?
    {
        PortfolioOutcome::ParetoFrontier(candidates) => {
            Ok(DeliberationOutcome::ParetoFrontier(DeliberatedFrontier {
                transition: portfolio.transition.clone(),
                world_snapshot: world_snapshot.clone(),
                policy,
                candidates,
            }))
        }
        PortfolioOutcome::Abstain(reason) => Ok(DeliberationOutcome::Abstain(reason)),
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum DeliberationError {
    #[error("invalid portfolio: {0}")]
    Portfolio(PortfolioError),
    #[error("required world-snapshot field is empty: {0}")]
    EmptySnapshotField(&'static str),
    #[error("cryptographic world-snapshot digest must be exactly 32 hexadecimal bytes")]
    InvalidCryptographicSnapshotDigest,
    #[error(
        "target frame {target_frame:?} does not match world snapshot frame {snapshot_frame:?}"
    )]
    SnapshotFrameMismatch {
        target_frame: String,
        snapshot_frame: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portfolio::ModelPrediction;
    use symthaea_physical_effects::{
        AuthorityClass, EffectKind, MechanismRef, PhysicalModality, PredictedOutcome,
        ProposedIntervention, TargetRegion,
    };

    fn portfolio() -> CandidatePortfolio {
        let transition = DesiredTransition::simulation_only(
            "t-1",
            "test deliberation receipt",
            TargetRegion::new("world", "fixture"),
            EffectKind::Characterize,
            vec![PhysicalModality::Acoustic, PhysicalModality::Photonic],
        );
        let candidate = CandidateAssessment {
            proposal: ProposedIntervention {
                id: "acoustic".into(),
                transition_id: "t-1".into(),
                mechanism: MechanismRef {
                    backend: "fixture".into(),
                    mechanism: "diagnostic".into(),
                    modality: PhysicalModality::Acoustic,
                },
                required_authority: AuthorityClass::SimulationOnly,
                predicted_outcome: PredictedOutcome {
                    success_probability: 0.9,
                    epistemic_uncertainty: 0.1,
                    aleatoric_uncertainty: 0.05,
                },
            },
            model_predictions: vec![
                ModelPrediction {
                    model_id: "a".into(),
                    success_probability: 0.9,
                },
                ModelPrediction {
                    model_id: "b".into(),
                    success_probability: 0.88,
                },
            ],
            expected_energy_j: 1.0,
            expected_power_w: None,
            expected_duration_ms: 100,
            information_gain: 0.8,
            reversibility_score: 1.0,
            safety_margin: 0.9,
        };
        CandidatePortfolio {
            transition,
            candidates: vec![candidate],
        }
    }

    fn snapshot() -> WorldSnapshotRef {
        WorldSnapshotRef::new("world", "snapshot-digest-001")
    }

    #[test]
    fn only_frontier_member_can_mint_selection_receipt() {
        let frontier = match deliberate(&portfolio(), &snapshot(), PortfolioPolicy::default()).unwrap()
        {
            DeliberationOutcome::ParetoFrontier(frontier) => frontier,
            other => panic!("expected frontier, got {other:?}"),
        };

        assert_eq!(frontier.world_snapshot(), &snapshot());
        assert!(frontier.select("acoustic").is_some());
        assert!(frontier.select("not-on-frontier").is_none());
    }

    #[test]
    fn mismatched_world_snapshot_frame_fails_closed() {
        let wrong = WorldSnapshotRef::new("another-world", "snapshot-digest-002");
        assert!(matches!(
            deliberate(&portfolio(), &wrong, PortfolioPolicy::default()),
            Err(DeliberationError::SnapshotFrameMismatch { .. })
        ));
    }

    #[test]
    fn empty_snapshot_digest_is_rejected() {
        let empty = WorldSnapshotRef::new("world", "");
        assert!(matches!(
            deliberate(&portfolio(), &empty, PortfolioPolicy::default()),
            Err(DeliberationError::EmptySnapshotField(_))
        ));
    }

    #[test]
    fn cryptographic_snapshot_scheme_is_part_of_lineage() {
        let blake = WorldSnapshotRef::cryptographic(
            "world",
            SnapshotDigestAlgorithm::Blake3,
            "a".repeat(64),
        );
        let sha = WorldSnapshotRef::cryptographic(
            "world",
            SnapshotDigestAlgorithm::Sha256,
            "a".repeat(64),
        );
        assert_ne!(blake, sha);
        assert!(blake.validate().is_ok());
        assert!(sha.validate().is_ok());
    }

    #[test]
    fn malformed_declared_cryptographic_digest_fails_closed() {
        let malformed = WorldSnapshotRef::cryptographic(
            "world",
            SnapshotDigestAlgorithm::Blake3,
            "not-a-content-digest",
        );
        assert_eq!(
            malformed.validate(),
            Err(DeliberationError::InvalidCryptographicSnapshotDigest)
        );
    }
}
