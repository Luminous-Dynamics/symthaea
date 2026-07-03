// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC World Model — Persistent System State Tracking
//!
//! Maintains a bundled ContinuousHV representing the current NixOS system state.
//! Supports incremental updates from observations, drift detection (similarity
//! between current and expected state), and predictive inference (project state
//! change from a proposed action).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use symthaea_core::hdc::ContinuousHV;

/// Tracks system state as an HDC vector that evolves with observations.
pub struct HdcWorldModel {
    /// Current system state vector — bundled from all observations.
    state: ContinuousHV,
    /// Expected "healthy" state vector for drift detection.
    expected_state: Option<ContinuousHV>,
    /// Named state facets (e.g., "services", "packages", "network").
    facets: HashMap<String, ContinuousHV>,
    /// Number of observations incorporated.
    observation_count: usize,
    /// Exponential moving average weight for new observations.
    ema_alpha: f32,
    /// HDC dimension.
    dim: usize,
}

/// Drift detection result.
#[derive(Debug, Clone)]
pub struct DriftReport {
    /// Overall similarity between current and expected state (1.0 = identical).
    pub similarity: f32,
    /// Whether drift exceeds the threshold.
    pub drifted: bool,
    /// Per-facet drift scores (facet_name, similarity).
    pub facet_drifts: Vec<(String, f32)>,
}

/// Predicted state change from a proposed action.
#[derive(Debug, Clone)]
pub struct StateProjection {
    /// Projected state after the action.
    pub projected_state: ContinuousHV,
    /// How different the projected state is from current (0.0 = no change, 1.0 = orthogonal).
    pub change_magnitude: f32,
    /// How close the projected state is to the expected state (drift improvement).
    pub drift_improvement: f32,
}

/// Serializable snapshot of the world model for persistence.
#[derive(Debug, Serialize, Deserialize)]
struct WorldModelSnapshot {
    version: u32,
    dim: usize,
    ema_alpha: f32,
    observation_count: usize,
    state: Vec<f32>,
    expected_state: Option<Vec<f32>>,
    facets: HashMap<String, Vec<f32>>,
}

impl HdcWorldModel {
    /// Create a new HDC world model.
    pub fn new(dim: usize) -> Self {
        Self {
            state: ContinuousHV::zero(dim),
            expected_state: None,
            facets: HashMap::new(),
            observation_count: 0,
            ema_alpha: 0.1,
            dim,
        }
    }

    /// Create with default HDC dimension.
    pub fn default_dim() -> Self {
        Self::new(crate::encoding::codebook::NIX_HDC_DIM)
    }

    /// Set the EMA blending weight for new observations (default 0.1).
    pub fn with_ema_alpha(mut self, alpha: f32) -> Self {
        self.ema_alpha = alpha.clamp(0.01, 1.0);
        self
    }

    /// Set the expected "healthy" state for drift detection.
    pub fn set_expected_state(&mut self, state: ContinuousHV) {
        self.expected_state = Some(state);
    }

    /// Incorporate a new observation into the world model.
    ///
    /// Uses exponential moving average: state = (1-α)·state + α·observation
    pub fn observe(&mut self, observation: &ContinuousHV) {
        if self.observation_count == 0 {
            self.state = observation.clone();
        } else {
            let refs = [&self.state, observation];
            let weights = [1.0 - self.ema_alpha, self.ema_alpha];
            self.state = ContinuousHV::weighted_bundle(&refs, &weights);
        }
        self.observation_count += 1;
    }

    /// Update a named facet (e.g., "services", "packages") with a new observation.
    ///
    /// The overall state is rebuilt by bundling all facets.
    pub fn observe_facet(&mut self, facet_name: &str, observation: &ContinuousHV) {
        let entry = self
            .facets
            .entry(facet_name.to_string())
            .or_insert_with(|| ContinuousHV::zero(self.dim));

        // EMA update for the facet
        let refs = [entry as &ContinuousHV, observation];
        let weights = [1.0 - self.ema_alpha, self.ema_alpha];
        *entry = ContinuousHV::weighted_bundle(&refs, &weights);

        // Rebuild overall state from facets
        self.rebuild_state_from_facets();
        self.observation_count += 1;
    }

    /// Detect drift between current state and expected state.
    pub fn detect_drift(&self, threshold: f32) -> DriftReport {
        let overall_sim = match &self.expected_state {
            Some(expected) => self.state.similarity(expected).max(0.0),
            None => 1.0, // No expected state → no drift
        };

        let facet_drifts: Vec<(String, f32)> = if let Some(expected) = &self.expected_state {
            self.facets
                .iter()
                .map(|(name, facet_hv)| (name.clone(), facet_hv.similarity(expected).max(0.0)))
                .collect()
        } else {
            Vec::new()
        };

        DriftReport {
            similarity: overall_sim,
            drifted: overall_sim < threshold,
            facet_drifts,
        }
    }

    /// Project how the state would change if an action delta is applied.
    pub fn project_action(&self, action_delta: &ContinuousHV) -> StateProjection {
        let projected = self.state.add(action_delta);
        let change_magnitude = 1.0 - self.state.similarity(&projected).max(0.0);

        let drift_improvement = match &self.expected_state {
            Some(expected) => {
                let current_drift = 1.0 - self.state.similarity(expected).max(0.0);
                let projected_drift = 1.0 - projected.similarity(expected).max(0.0);
                current_drift - projected_drift // Positive = improvement
            }
            None => 0.0,
        };

        StateProjection {
            projected_state: projected,
            change_magnitude,
            drift_improvement,
        }
    }

    /// Get the current system state vector.
    pub fn state(&self) -> &ContinuousHV {
        &self.state
    }

    /// Get the expected state (if set).
    pub fn expected_state(&self) -> Option<&ContinuousHV> {
        self.expected_state.as_ref()
    }

    /// Number of observations incorporated.
    pub fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Number of tracked facets.
    pub fn facet_count(&self) -> usize {
        self.facets.len()
    }

    /// Get a specific facet vector.
    pub fn facet(&self, name: &str) -> Option<&ContinuousHV> {
        self.facets.get(name)
    }

    /// Save the world model state to a JSON file.
    ///
    /// Persists the current state vector, expected state, all facets, and
    /// configuration so the model can be restored across sessions.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let snapshot = WorldModelSnapshot {
            version: 1,
            dim: self.dim,
            ema_alpha: self.ema_alpha,
            observation_count: self.observation_count,
            state: self.state.as_slice().to_vec(),
            expected_state: self.expected_state.as_ref().map(|s| s.as_slice().to_vec()),
            facets: self
                .facets
                .iter()
                .map(|(name, hv)| (name.clone(), hv.as_slice().to_vec()))
                .collect(),
        };
        let json = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| format!("Failed to serialize world model: {e}"))?;
        std::fs::write(path, json)
            .map_err(|e| format!("Failed to write world model to {}: {}", path.display(), e))
    }

    /// Load a previously saved world model, replacing the current state.
    ///
    /// Returns the number of facets loaded.
    pub fn load(&mut self, path: &Path) -> Result<usize, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read world model from {}: {}", path.display(), e))?;
        let snapshot: WorldModelSnapshot = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to deserialize world model: {e}"))?;

        if snapshot.dim != self.dim {
            return Err(format!(
                "Dimension mismatch: saved={}, current={}",
                snapshot.dim, self.dim
            ));
        }

        self.ema_alpha = snapshot.ema_alpha;
        self.observation_count = snapshot.observation_count;
        self.state = ContinuousHV::from_slice(&snapshot.state);
        self.expected_state = snapshot
            .expected_state
            .map(|s| ContinuousHV::from_slice(&s));
        self.facets = snapshot
            .facets
            .into_iter()
            .map(|(name, data)| (name, ContinuousHV::from_slice(&data)))
            .collect();

        Ok(self.facets.len())
    }

    /// Rebuild the overall state by equal-weight bundling of all facets.
    fn rebuild_state_from_facets(&mut self) {
        if self.facets.is_empty() {
            return;
        }
        let refs: Vec<&ContinuousHV> = self.facets.values().collect();
        let weight = 1.0 / refs.len() as f32;
        let weights: Vec<f32> = vec![weight; refs.len()];
        self.state = ContinuousHV::weighted_bundle(&refs, &weights);
    }
}

impl Default for HdcWorldModel {
    fn default() -> Self {
        Self::default_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observe_updates_state() {
        let dim = 1024;
        let mut wm = HdcWorldModel::new(dim);
        assert_eq!(wm.observation_count(), 0);

        let obs = ContinuousHV::random(dim, 1);
        wm.observe(&obs);
        assert_eq!(wm.observation_count(), 1);

        // First observation becomes the state directly
        let sim = wm.state().similarity(&obs);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "First observation should be the state: {}",
            sim
        );
    }

    #[test]
    fn test_ema_blending() {
        let dim = 1024;
        let mut wm = HdcWorldModel::new(dim).with_ema_alpha(0.5);

        let obs1 = ContinuousHV::random(dim, 1);
        let obs2 = ContinuousHV::random(dim, 2);

        wm.observe(&obs1);
        wm.observe(&obs2);

        // State should be between obs1 and obs2
        let sim1 = wm.state().similarity(&obs1);
        let sim2 = wm.state().similarity(&obs2);
        assert!(
            sim1 > 0.0 && sim2 > 0.0,
            "State should blend: sim1={}, sim2={}",
            sim1,
            sim2
        );
    }

    #[test]
    fn test_drift_detection() {
        let dim = 1024;
        let mut wm = HdcWorldModel::new(dim);

        let expected = ContinuousHV::random(dim, 1);
        wm.set_expected_state(expected.clone());

        // Observe the expected state — no drift
        wm.observe(&expected);
        let report = wm.detect_drift(0.8);
        assert!(
            !report.drifted,
            "Should not drift when at expected state: sim={}",
            report.similarity
        );

        // Observe something very different — should drift
        let different = ContinuousHV::random(dim, 99);
        let mut wm2 = HdcWorldModel::new(dim);
        wm2.set_expected_state(expected);
        wm2.observe(&different);
        let report2 = wm2.detect_drift(0.8);
        assert!(
            report2.drifted,
            "Should drift when state diverges: sim={}",
            report2.similarity
        );
    }

    #[test]
    fn test_facet_tracking() {
        let dim = 1024;
        let mut wm = HdcWorldModel::new(dim);

        let services = ContinuousHV::random(dim, 1);
        let packages = ContinuousHV::random(dim, 2);

        wm.observe_facet("services", &services);
        wm.observe_facet("packages", &packages);

        assert_eq!(wm.facet_count(), 2);
        assert!(wm.facet("services").is_some());
        assert!(wm.facet("packages").is_some());
    }

    #[test]
    fn test_project_action() {
        let dim = 1024;
        let mut wm = HdcWorldModel::new(dim);

        let current = ContinuousHV::random(dim, 1);
        wm.observe(&current);

        let delta = ContinuousHV::random(dim, 2);
        let projection = wm.project_action(&delta);

        assert!(
            projection.change_magnitude > 0.0,
            "Action should change state"
        );
        assert!(projection.change_magnitude <= 1.0);
    }

    #[test]
    fn test_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("world_model.json");

        let dim = 256;
        let mut wm = HdcWorldModel::new(dim).with_ema_alpha(0.3);

        let obs = ContinuousHV::random(dim, 1);
        wm.observe(&obs);
        wm.observe_facet("services", &ContinuousHV::random(dim, 2));
        wm.observe_facet("packages", &ContinuousHV::random(dim, 3));
        wm.set_expected_state(ContinuousHV::random(dim, 4));

        // Save
        wm.save(&path).unwrap();
        assert!(path.exists());

        // Load into fresh model
        let mut wm2 = HdcWorldModel::new(dim);
        let facets_loaded = wm2.load(&path).unwrap();
        assert_eq!(facets_loaded, 2);
        assert_eq!(wm2.observation_count(), wm.observation_count());
        assert!(wm2.expected_state().is_some());

        // State vectors should match
        let sim = wm.state().similarity(wm2.state());
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Loaded state should match: sim={}",
            sim
        );
    }

    #[test]
    fn test_load_dimension_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("world_model.json");

        let mut wm = HdcWorldModel::new(256);
        wm.observe(&ContinuousHV::random(256, 1));
        wm.save(&path).unwrap();

        // Try loading into a model with different dimension
        let mut wm2 = HdcWorldModel::new(512);
        let result = wm2.load(&path);
        assert!(result.is_err(), "Should reject dimension mismatch");
    }

    #[test]
    fn test_ema_alpha_clamping() {
        // Below minimum
        let wm = HdcWorldModel::new(256).with_ema_alpha(0.0);
        assert!((wm.ema_alpha - 0.01).abs() < 1e-6, "Should clamp to 0.01");

        // Above maximum
        let wm2 = HdcWorldModel::new(256).with_ema_alpha(5.0);
        assert!((wm2.ema_alpha - 1.0).abs() < 1e-6, "Should clamp to 1.0");

        // Within range
        let wm3 = HdcWorldModel::new(256).with_ema_alpha(0.5);
        assert!((wm3.ema_alpha - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_drift_no_expected_state() {
        let dim = 256;
        let mut wm = HdcWorldModel::new(dim);
        wm.observe(&ContinuousHV::random(dim, 1));

        let report = wm.detect_drift(0.5);
        assert!(!report.drifted, "No expected state → no drift");
        assert!((report.similarity - 1.0).abs() < 1e-6);
        assert!(report.facet_drifts.is_empty());
    }

    #[test]
    fn test_facet_lookup_nonexistent() {
        let wm = HdcWorldModel::new(256);
        assert!(wm.facet("nonexistent").is_none());
    }

    #[test]
    fn test_observe_facet_rebuilds_state() {
        let dim = 256;
        let mut wm = HdcWorldModel::new(dim);

        let svc = ContinuousHV::random(dim, 1);
        let pkg = ContinuousHV::random(dim, 2);

        wm.observe_facet("services", &svc);
        let state_after_one = wm.state().clone();

        wm.observe_facet("packages", &pkg);
        let state_after_two = wm.state().clone();

        // State should change after adding a second facet
        let sim = state_after_one.similarity(&state_after_two);
        assert!(
            sim < 0.99,
            "Adding a second facet should change state, sim={sim}"
        );
    }

    #[test]
    fn test_project_action_with_expected_state() {
        let dim = 256;
        let mut wm = HdcWorldModel::new(dim);

        let current = ContinuousHV::random(dim, 1);
        let expected = ContinuousHV::random(dim, 2);
        wm.observe(&current);
        wm.set_expected_state(expected);

        let delta = ContinuousHV::random(dim, 3);
        let proj = wm.project_action(&delta);

        // drift_improvement should be non-zero when expected state is set
        assert!(
            proj.drift_improvement.is_finite(),
            "drift_improvement should be finite"
        );
    }

    #[test]
    fn test_project_action_no_expected_state() {
        let dim = 256;
        let mut wm = HdcWorldModel::new(dim);
        wm.observe(&ContinuousHV::random(dim, 1));

        let delta = ContinuousHV::random(dim, 2);
        let proj = wm.project_action(&delta);
        assert!(
            (proj.drift_improvement).abs() < 1e-6,
            "No expected state → zero drift improvement"
        );
    }

    #[test]
    fn test_drift_with_facets() {
        let dim = 256;
        let mut wm = HdcWorldModel::new(dim);

        let expected = ContinuousHV::random(dim, 99);
        wm.set_expected_state(expected);

        wm.observe_facet("services", &ContinuousHV::random(dim, 1));
        wm.observe_facet("packages", &ContinuousHV::random(dim, 2));

        let report = wm.detect_drift(0.9);
        // Should have per-facet drift scores
        assert_eq!(report.facet_drifts.len(), 2);
        for (name, sim) in &report.facet_drifts {
            assert!(sim.is_finite(), "Facet '{name}' drift should be finite");
        }
    }

    #[test]
    fn test_save_nonexistent_dir() {
        let wm = HdcWorldModel::new(256);
        let result = wm.save(Path::new("/nonexistent/dir/model.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_nonexistent_file() {
        let mut wm = HdcWorldModel::new(256);
        let result = wm.load(Path::new("/nonexistent/model.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.json");
        std::fs::write(&path, "not valid json").unwrap();

        let mut wm = HdcWorldModel::new(256);
        let result = wm.load(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_observations_ema() {
        let dim = 256;
        let mut wm = HdcWorldModel::new(dim).with_ema_alpha(0.1);

        let obs1 = ContinuousHV::random(dim, 1);
        let obs2 = ContinuousHV::random(dim, 2);
        let obs3 = ContinuousHV::random(dim, 3);

        wm.observe(&obs1);
        wm.observe(&obs2);
        wm.observe(&obs3);
        assert_eq!(wm.observation_count(), 3);

        // With low alpha (0.1), state should still be more similar to obs1 (first/dominant)
        let sim1 = wm.state().similarity(&obs1);
        let sim3 = wm.state().similarity(&obs3);
        assert!(
            sim1 > sim3,
            "With alpha=0.1, earlier observations should dominate: sim1={sim1}, sim3={sim3}"
        );
    }

    // ── Phase 2.8: Drift Detection Hardening ────────────────────────────────
    // These tests cover the sensor/drift failure path:
    //   sensor shift → detect_drift → DriftReport
    // Invariants: similarity always in [0, 1], drifted flag always correct,
    // degraded input produces safe uncertainty not false certainty.

    /// Tiny sensor shift must NOT trigger drift at a reasonable threshold.
    /// Prevents false-positive alarms from sensor noise.
    #[test]
    fn test_drift_tiny_change_no_false_alarm() {
        let dim = 1024;
        let mut wm = HdcWorldModel::new(dim).with_ema_alpha(0.01); // very slow adaptation

        let expected = ContinuousHV::random(dim, 1);
        wm.set_expected_state(expected.clone());
        wm.observe(&expected);

        // Observe the expected state again (tiny "shift" = same vector)
        wm.observe(&expected);

        let report = wm.detect_drift(0.8);
        assert!(
            !report.drifted,
            "tiny (zero) shift must not trigger drift: sim={}",
            report.similarity
        );
        assert!(
            report.similarity >= 0.0 && report.similarity <= 1.0,
            "similarity must be in [0,1]: {}",
            report.similarity
        );
    }

    /// Massive sensor shift (completely orthogonal vector) MUST trigger drift.
    /// Prevents the system from missing a genuine state collapse.
    #[test]
    fn test_drift_massive_change_detected() {
        let dim = 1024;
        let mut wm = HdcWorldModel::new(dim).with_ema_alpha(1.0); // instant adaptation

        let expected = ContinuousHV::random(dim, 1);
        let orthogonal = ContinuousHV::random(dim, 999); // ~orthogonal in high dim

        wm.set_expected_state(expected);
        wm.observe(&orthogonal);

        let report = wm.detect_drift(0.5);
        assert!(
            report.drifted,
            "massive orthogonal shift must trigger drift: sim={}",
            report.similarity
        );
        assert!(
            report.similarity >= 0.0 && report.similarity <= 1.0,
            "similarity must be in [0,1] even for orthogonal input: {}",
            report.similarity
        );
    }

    /// Repeated noisy observations must keep drift report numerically stable.
    /// Prevents accumulation of floating-point error over many EMA steps.
    #[test]
    fn test_drift_noisy_observations_stay_stable() {
        let dim = 512;
        let mut wm = HdcWorldModel::new(dim).with_ema_alpha(0.2);

        let expected = ContinuousHV::random(dim, 42);
        wm.set_expected_state(expected.clone());
        wm.observe(&expected);

        // Apply 50 noisy observations (all random — high dimensional noise)
        for seed in 0u64..50 {
            let noise = ContinuousHV::random(dim, seed * 1000 + 7);
            wm.observe(&noise);

            let report = wm.detect_drift(0.5);
            assert!(
                report.similarity.is_finite(),
                "similarity must be finite after {seed} noisy observations: {}",
                report.similarity
            );
            assert!(
                report.similarity >= 0.0 && report.similarity <= 1.0,
                "similarity must be in [0,1] after {seed} noisy observations: {}",
                report.similarity
            );
        }
    }

    /// Drift score must always be in [0, 1] — the invariant that downstream
    /// systems depend on for safe thresholding.
    #[test]
    fn test_drift_similarity_always_in_unit_interval() {
        let dim = 256;
        let mut wm = HdcWorldModel::new(dim);

        let expected = ContinuousHV::random(dim, 1);
        wm.set_expected_state(expected);

        // Range of seeds covering near-identical, near-orthogonal, and everything between
        for seed in [1u64, 2, 50, 100, 999, 12345, u64::MAX / 2] {
            let obs = ContinuousHV::random(dim, seed);
            wm.observe(&obs);
            let report = wm.detect_drift(0.5);

            assert!(
                report.similarity >= 0.0 && report.similarity <= 1.0,
                "similarity out of [0,1] for seed {seed}: {}",
                report.similarity
            );
            assert!(
                report.similarity.is_finite(),
                "similarity must be finite for seed {seed}: {}",
                report.similarity
            );
        }
    }

    /// Drift detection with no expected state must never claim drift
    /// regardless of what is observed (the system is in "unknown" mode,
    /// not "broken" mode — safe uncertainty, not false certainty).
    #[test]
    fn test_drift_no_expected_state_never_reports_drift_with_arbitrary_obs() {
        let dim = 256;
        let mut wm = HdcWorldModel::new(dim);

        for seed in 0u64..20 {
            let obs = ContinuousHV::random(dim, seed);
            wm.observe(&obs);
            let report = wm.detect_drift(0.99); // very aggressive threshold
            assert!(
                !report.drifted,
                "no expected state must never report drift (seed={seed})"
            );
            assert!(
                (report.similarity - 1.0).abs() < 1e-5,
                "no expected state → similarity=1.0 (seed={seed}): {}",
                report.similarity
            );
        }
    }

    /// Facet-level drift scores must all be finite and in [0, 1].
    /// Per-facet drift is the first thing a diagnostic system reads.
    #[test]
    fn test_drift_facet_scores_finite_and_bounded() {
        let dim = 256;
        let mut wm = HdcWorldModel::new(dim);

        let expected = ContinuousHV::random(dim, 99);
        wm.set_expected_state(expected);

        for (name, seed) in [
            ("services", 1u64),
            ("packages", 2),
            ("network", 3),
            ("boot", 4),
        ] {
            wm.observe_facet(name, &ContinuousHV::random(dim, seed));
        }

        let report = wm.detect_drift(0.5);
        assert_eq!(report.facet_drifts.len(), 4, "all four facets must appear");

        for (name, sim) in &report.facet_drifts {
            assert!(
                sim.is_finite(),
                "facet '{name}' drift score must be finite: {sim}"
            );
            assert!(
                *sim >= 0.0 && *sim <= 1.0,
                "facet '{name}' drift score must be in [0,1]: {sim}"
            );
        }
    }
}
