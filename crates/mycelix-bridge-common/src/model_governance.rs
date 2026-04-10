// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Model Governance — Community-Driven Scoring Model Selection
//!
//! The lifecycle for proposing, testing, and adopting scoring models:
//!
//! 1. Researcher registers a new model via [`ModelProposal`]
//! 2. Model runs in shadow mode for a minimum [`MIN_SHADOW_DAYS`]
//! 3. Community reviews divergence data from shadow evaluation
//! 4. Steward-tier proposal to switch active model
//! 5. Constitutional envelope validates the proposed model
//! 6. Transition blends old→new over [`MIN_RAMP_DAYS`]
//!
//! ## Constitutional Guards
//!
//! No model can be activated that violates the [`ConstitutionalEnvelope`].
//! The shadow period cannot be shortened below [`MIN_SHADOW_DAYS`].
//! The transition ramp cannot be shortened below [`MIN_RAMP_DAYS`].

use serde::{Deserialize, Serialize};

use crate::constitutional_envelope::{self, sanitize_score, ConstitutionalViolation};
use crate::scoring_model::ModelDescriptor;

// ============================================================================
// Governance Constants
// ============================================================================

/// Minimum shadow evaluation period before a model can be activated (days).
/// Ensures sufficient divergence data for informed community decisions.
pub const MIN_SHADOW_DAYS: u32 = 30;

/// Minimum transition ramp period when switching models (days).
/// During the ramp, scores blend linearly from old model to new.
/// This prevents governance disruption from sudden model changes.
pub const MIN_RAMP_DAYS: u32 = 7;

/// Maximum ramp period (days).  Prevents indefinite transition states.
pub const MAX_RAMP_DAYS: u32 = 90;

/// Microseconds per day (for timestamp arithmetic).
const US_PER_DAY: u64 = 86_400_000_000;

// ============================================================================
// Model Proposal
// ============================================================================

/// A proposal to register a new scoring model for shadow evaluation.
///
/// Proposals do NOT immediately activate the model — they start shadow mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProposal {
    /// The proposed model descriptor.
    pub model: ModelDescriptor,
    /// Why this model is being proposed (human-readable rationale).
    pub rationale: String,
    /// Proposed shadow evaluation period (days, >= MIN_SHADOW_DAYS).
    pub shadow_days: u32,
    /// When the proposal was submitted (microseconds).
    pub proposed_at: u64,
    /// Who proposed it (agent DID).
    pub proposed_by: String,
    /// Status of the proposal.
    pub status: ProposalStatus,
}

/// Lifecycle status of a model proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalStatus {
    /// Awaiting community review.
    Pending,
    /// Model is running in shadow mode.
    Shadow,
    /// Shadow period complete, awaiting activation vote.
    ReadyForVote,
    /// Community voted to activate.
    Approved,
    /// Community rejected or proposal expired.
    Rejected,
    /// Model is now the active model (transition started).
    Activated,
}

// ============================================================================
// Model Transition
// ============================================================================

/// An active transition between two scoring models.
///
/// During the ramp period, governance scores are blended:
///   blended = old_score × (1 - progress) + new_score × progress
/// where progress = days_elapsed / ramp_days.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTransition {
    /// The model being replaced.
    pub old_model_id: String,
    /// The model being adopted.
    pub new_model_id: String,
    /// Ramp duration (days, >= MIN_RAMP_DAYS).
    pub ramp_days: u32,
    /// When the transition started (microseconds).
    pub started_at: u64,
    /// When the transition completes (microseconds).
    pub completes_at: u64,
}

impl ModelTransition {
    /// Create a new transition with constitutional bounds enforcement.
    pub fn new(
        old_model_id: String,
        new_model_id: String,
        ramp_days: u32,
        started_at: u64,
    ) -> Result<Self, String> {
        let effective_ramp = ramp_days.clamp(MIN_RAMP_DAYS, MAX_RAMP_DAYS);
        let completes_at = started_at + (effective_ramp as u64 * US_PER_DAY);

        Ok(Self {
            old_model_id,
            new_model_id,
            ramp_days: effective_ramp,
            started_at,
            completes_at,
        })
    }

    /// Compute the blending progress at a given timestamp.
    ///
    /// Returns 0.0 at `started_at`, 1.0 at `completes_at`.
    pub fn progress(&self, now_us: u64) -> f64 {
        if now_us <= self.started_at {
            return 0.0;
        }
        if now_us >= self.completes_at {
            return 1.0;
        }
        let elapsed = (now_us - self.started_at) as f64;
        let total = (self.completes_at - self.started_at) as f64;
        if total <= 0.0 {
            return 1.0;
        }
        (elapsed / total).clamp(0.0, 1.0)
    }

    /// Whether the transition is complete.
    pub fn is_complete(&self, now_us: u64) -> bool {
        now_us >= self.completes_at
    }
}

/// Compute a blended score during a model transition.
///
/// Linearly interpolates between the old model's score and the new model's
/// score based on transition progress.
///
/// `progress` is 0.0 (start) to 1.0 (complete).
pub fn blended_score(old_score: f64, new_score: f64, progress: f64) -> f64 {
    let p = progress.clamp(0.0, 1.0);
    sanitize_score(old_score * (1.0 - p) + new_score * p)
}

// ============================================================================
// Validation
// ============================================================================

/// Validate a model proposal before accepting it for shadow evaluation.
pub fn validate_proposal(proposal: &ModelProposal) -> Result<(), String> {
    // Constitutional check on the model itself
    if let Err(violation) = proposal.model.validate() {
        return Err(format!("Constitutional violation: {}", violation));
    }

    // Shadow period minimum
    if proposal.shadow_days < MIN_SHADOW_DAYS {
        return Err(format!(
            "Shadow period {} days < minimum {} days",
            proposal.shadow_days, MIN_SHADOW_DAYS
        ));
    }

    // Rationale must be non-empty
    if proposal.rationale.trim().is_empty() {
        return Err("Proposal rationale cannot be empty".into());
    }

    // Model ID must be non-empty
    if proposal.model.model_id.trim().is_empty() {
        return Err("Model ID cannot be empty".into());
    }

    Ok(())
}

/// Check if a shadow period has elapsed and the model is ready for vote.
pub fn is_shadow_complete(proposal: &ModelProposal, now_us: u64) -> bool {
    if proposal.status != ProposalStatus::Shadow {
        return false;
    }
    let shadow_duration_us = proposal.shadow_days as u64 * US_PER_DAY;
    now_us >= proposal.proposed_at + shadow_duration_us
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring_model::{Canonical4D, ScoringModel};

    fn sample_proposal() -> ModelProposal {
        let model = Canonical4D::default();
        ModelProposal {
            model: ModelDescriptor::from_model(&model, 1_000_000, "did:mycelix:researcher".into()),
            rationale: "Testing alternative weight distribution".into(),
            shadow_days: 30,
            proposed_at: 0,
            proposed_by: "did:mycelix:researcher".into(),
            status: ProposalStatus::Pending,
        }
    }

    // ---- Proposal validation ----

    #[test]
    fn valid_proposal_passes() {
        assert!(validate_proposal(&sample_proposal()).is_ok());
    }

    #[test]
    fn proposal_with_constitutional_violation_rejected() {
        let mut proposal = sample_proposal();
        proposal.model.weights = vec![0.60, 0.40, 0.0, 0.0]; // 0.60 > 0.50
        assert!(validate_proposal(&proposal).is_err());
    }

    #[test]
    fn proposal_with_short_shadow_rejected() {
        let mut proposal = sample_proposal();
        proposal.shadow_days = 7; // < MIN_SHADOW_DAYS (30)
        assert!(validate_proposal(&proposal).is_err());
    }

    #[test]
    fn proposal_with_empty_rationale_rejected() {
        let mut proposal = sample_proposal();
        proposal.rationale = "".into();
        assert!(validate_proposal(&proposal).is_err());
    }

    #[test]
    fn proposal_with_empty_model_id_rejected() {
        let mut proposal = sample_proposal();
        proposal.model.model_id = "".into();
        assert!(validate_proposal(&proposal).is_err());
    }

    // ---- Shadow period ----

    #[test]
    fn shadow_not_complete_immediately() {
        let mut proposal = sample_proposal();
        proposal.status = ProposalStatus::Shadow;
        proposal.proposed_at = 0;
        assert!(!is_shadow_complete(&proposal, 1_000_000));
    }

    #[test]
    fn shadow_complete_after_30_days() {
        let mut proposal = sample_proposal();
        proposal.status = ProposalStatus::Shadow;
        proposal.proposed_at = 0;
        let thirty_days_us = 30 * US_PER_DAY;
        assert!(is_shadow_complete(&proposal, thirty_days_us));
    }

    #[test]
    fn shadow_not_complete_if_not_in_shadow_status() {
        let mut proposal = sample_proposal();
        proposal.status = ProposalStatus::Pending;
        let thirty_days_us = 30 * US_PER_DAY;
        assert!(!is_shadow_complete(&proposal, thirty_days_us));
    }

    // ---- Model transition ----

    #[test]
    fn transition_creation_enforces_min_ramp() {
        let t = ModelTransition::new("old".into(), "new".into(), 1, 0).unwrap();
        assert_eq!(t.ramp_days, MIN_RAMP_DAYS); // clamped up from 1
    }

    #[test]
    fn transition_creation_enforces_max_ramp() {
        let t = ModelTransition::new("old".into(), "new".into(), 200, 0).unwrap();
        assert_eq!(t.ramp_days, MAX_RAMP_DAYS); // clamped down from 200
    }

    #[test]
    fn transition_progress_zero_at_start() {
        let t = ModelTransition::new("old".into(), "new".into(), 30, 1000).unwrap();
        assert_eq!(t.progress(1000), 0.0);
    }

    #[test]
    fn transition_progress_one_at_end() {
        let t = ModelTransition::new("old".into(), "new".into(), 30, 0).unwrap();
        assert_eq!(t.progress(t.completes_at), 1.0);
    }

    #[test]
    fn transition_progress_half_at_midpoint() {
        let t = ModelTransition::new("old".into(), "new".into(), 30, 0).unwrap();
        let midpoint = t.completes_at / 2;
        let p = t.progress(midpoint);
        assert!((p - 0.5).abs() < 0.01, "Midpoint progress should be ~0.5, got {}", p);
    }

    #[test]
    fn transition_complete_after_ramp() {
        let t = ModelTransition::new("old".into(), "new".into(), 7, 0).unwrap();
        assert!(!t.is_complete(0));
        assert!(!t.is_complete(US_PER_DAY * 3));
        assert!(t.is_complete(t.completes_at));
        assert!(t.is_complete(t.completes_at + 1));
    }

    // ---- Blended score ----

    #[test]
    fn blended_score_at_zero_progress() {
        assert!((blended_score(0.8, 0.4, 0.0) - 0.8).abs() < 1e-10);
    }

    #[test]
    fn blended_score_at_full_progress() {
        assert!((blended_score(0.8, 0.4, 1.0) - 0.4).abs() < 1e-10);
    }

    #[test]
    fn blended_score_at_half_progress() {
        let b = blended_score(0.8, 0.4, 0.5);
        assert!((b - 0.6).abs() < 1e-10, "Half blend of 0.8 and 0.4 should be 0.6, got {}", b);
    }

    #[test]
    fn blended_score_clamps_progress() {
        assert!((blended_score(0.8, 0.4, -1.0) - 0.8).abs() < 1e-10);
        assert!((blended_score(0.8, 0.4, 2.0) - 0.4).abs() < 1e-10);
    }

    #[test]
    fn blended_score_sanitizes_nan() {
        let b = blended_score(f64::NAN, 0.5, 0.5);
        assert!(b.is_finite());
        assert!(b >= 0.0 && b <= 1.0);
    }

    // ---- Serde ----

    #[test]
    fn proposal_serde_roundtrip() {
        let p = sample_proposal();
        let json = serde_json::to_string(&p).unwrap();
        let back: ModelProposal = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model.model_id, p.model.model_id);
        assert_eq!(back.shadow_days, p.shadow_days);
        assert_eq!(back.status, ProposalStatus::Pending);
    }

    #[test]
    fn transition_serde_roundtrip() {
        let t = ModelTransition::new("old-v1".into(), "new-v2".into(), 30, 1_000_000).unwrap();
        let json = serde_json::to_string(&t).unwrap();
        let back: ModelTransition = serde_json::from_str(&json).unwrap();
        assert_eq!(back.old_model_id, "old-v1");
        assert_eq!(back.new_model_id, "new-v2");
        assert_eq!(back.ramp_days, 30);
    }

    #[test]
    fn all_proposal_statuses_serde_roundtrip() {
        for status in [
            ProposalStatus::Pending,
            ProposalStatus::Shadow,
            ProposalStatus::ReadyForVote,
            ProposalStatus::Approved,
            ProposalStatus::Rejected,
            ProposalStatus::Activated,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: ProposalStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, back);
        }
    }
}
