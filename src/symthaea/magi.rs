// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MAGI loop calibration methods.
//!
//! All items in this module are gated behind `#[cfg(feature = "magi_loop")]`.

use std::path::PathBuf;

use crate::consciousness::recursive_improvement::{
    BrierScoreTracker, CalibrationConfig, CalibrationSummary, PersistenceConfig,
    PersistenceManager, PredictionDomain, StartupMode,
};
use crate::mind::SemanticIntent;
use crate::mind::structured_thought::EpistemicStatus;

use super::Symthaea;

/// Default on-disk location for the facade's Brier calibration state,
/// resolved relative to the home directory by `PersistenceConfig::full_path`
/// (same convention as the RSI subsystem's `.symthaea/magi_state.json`).
const FACADE_CALIBRATION_STATE_PATH: &str = ".symthaea/facade_calibration.json";

impl Symthaea {
    /// Initialize the facade Brier calibration tracker, restoring persisted
    /// state from disk when available (Tier 0.3, 2026-07-06).
    ///
    /// Without this, `calibration` was rebuilt `with_defaults()` every
    /// session, so the Phase 4.5 confidence adjustment could never actually
    /// calibrate. Reuses the RSI persistence types (`MagiStateSnapshot` /
    /// `PersistedDomainCalibration`) — no new schema.
    ///
    /// Set `SYMTHAEA_FACADE_CALIBRATION_PATH` to override the state path, or
    /// to the empty string to disable persistence entirely (tests, ephemeral
    /// runs).
    pub(super) fn init_facade_calibration() -> (BrierScoreTracker, Option<PersistenceManager>) {
        let state_path = match std::env::var("SYMTHAEA_FACADE_CALIBRATION_PATH") {
            Ok(p) if p.trim().is_empty() => {
                tracing::info!(
                    target: "symthaea::broca::calibration",
                    "Facade calibration persistence disabled (SYMTHAEA_FACADE_CALIBRATION_PATH is empty)"
                );
                return (BrierScoreTracker::with_defaults(), None);
            }
            Ok(p) => PathBuf::from(p),
            Err(_) => PathBuf::from(FACADE_CALIBRATION_STATE_PATH),
        };

        let config = PersistenceConfig {
            state_path,
            autosave_interval: 0, // facade drives the save cadence explicitly
            create_backups: true,
            max_backups: 3,
            enabled: true,
        };
        let mut manager = PersistenceManager::new(config);
        match manager.initialize() {
            Ok(StartupMode::WarmStart { session, .. }) => {
                let snapshot = manager.current();
                let tracker = BrierScoreTracker::from_persisted(
                    CalibrationConfig::default(),
                    &snapshot.calibration,
                    &snapshot.global_stats,
                );
                tracing::info!(
                    target: "symthaea::broca::calibration",
                    session,
                    total_predictions = tracker.total_predictions(),
                    path = %manager.config().state_path.display(),
                    "Facade calibration restored from persisted state"
                );
                (tracker, Some(manager))
            }
            Ok(_) => (BrierScoreTracker::with_defaults(), Some(manager)),
            Err(e) => {
                tracing::warn!(
                    target: "symthaea::broca::calibration",
                    error = %e,
                    "Facade calibration persistence unavailable; calibration will be in-memory only"
                );
                (BrierScoreTracker::with_defaults(), None)
            }
        }
    }

    /// Persist the facade calibration tracker to disk (atomic temp+rename
    /// inside `PersistenceManager::save`). Warns and continues on IO errors.
    pub(super) fn persist_facade_calibration(&mut self) {
        let Some(ref mut pm) = self.calibration_persistence else {
            return;
        };
        pm.update_from_tracker(&self.calibration);
        if let Err(e) = pm.save() {
            tracing::warn!(
                target: "symthaea::broca::calibration",
                error = %e,
                "Failed to persist facade calibration; continuing"
            );
        } else {
            tracing::debug!(
                target: "symthaea::broca::calibration",
                total_predictions = self.calibration.total_predictions(),
                path = %pm.config().state_path.display(),
                "Facade calibration persisted"
            );
        }
    }

    /// Get the current calibration summary.
    ///
    /// Returns global and per-domain Brier scores, ECE, accuracy, and
    /// whether the system is currently well-calibrated.
    pub fn calibration_summary(&self) -> CalibrationSummary {
        self.calibration.calibration_summary()
    }

    /// Map EpistemicStatus to a confidence float for calibration tracking.
    ///
    /// These values represent the system's belief about being correct:
    /// - Certain: 0.95 (very high confidence)
    /// - Probable: 0.75 (moderate-high confidence)
    /// - Uncertain: 0.45 (moderate-low confidence)
    /// - Unknown: 0.15 (very low confidence)
    /// - OutOfDomain: 0.10 (minimal confidence)
    pub(super) fn epistemic_to_confidence(status: &EpistemicStatus) -> f64 {
        match status {
            EpistemicStatus::Certain => 0.95,
            EpistemicStatus::Probable => 0.75,
            EpistemicStatus::Uncertain => 0.45,
            EpistemicStatus::Unknown => 0.15,
            EpistemicStatus::OutOfDomain => 0.10,
        }
    }

    /// Map a confidence float back to EpistemicStatus.
    ///
    /// Inverse of `epistemic_to_confidence`, using midpoint thresholds.
    pub(super) fn confidence_to_epistemic(confidence: f64) -> EpistemicStatus {
        if confidence >= 0.85 {
            EpistemicStatus::Certain
        } else if confidence >= 0.60 {
            EpistemicStatus::Probable
        } else if confidence >= 0.30 {
            EpistemicStatus::Uncertain
        } else if confidence >= 0.12 {
            EpistemicStatus::Unknown
        } else {
            EpistemicStatus::OutOfDomain
        }
    }

    /// Map SemanticIntent to a PredictionDomain for calibration tracking.
    ///
    /// Groups different intents into calibration domains:
    /// - Answer, Clarify -> Factual (knowledge-based predictions)
    /// - ProposeAction -> ToolUse (action outcome predictions)
    /// - Acknowledge, Continue -> UserBehavior (social interaction predictions)
    /// - Reflect -> SystemState (introspective predictions)
    /// - ExpressUncertainty, Unknown -> Factual (default calibration domain)
    pub(super) fn map_intent_to_domain(intent: &SemanticIntent) -> PredictionDomain {
        match intent {
            SemanticIntent::Answer | SemanticIntent::Clarify => PredictionDomain::Factual,
            SemanticIntent::ProposeAction => PredictionDomain::ToolUse,
            SemanticIntent::Acknowledge | SemanticIntent::Continue => {
                PredictionDomain::UserBehavior
            }
            SemanticIntent::Reflect => PredictionDomain::SystemState,
            SemanticIntent::ExpressUncertainty | SemanticIntent::Unknown => {
                PredictionDomain::Factual
            }
        }
    }
}
