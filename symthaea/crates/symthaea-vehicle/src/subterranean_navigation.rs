// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Subterranean navigation estimator — position tracking in GPS-denied environments.
//!
//! Minimal stubs for training integration. Full SLAM + acoustic ranging is future work.

use serde::{Deserialize, Serialize};

use crate::subterranean::{SubterraneanBridge, SurveyAnchor, TunnelRelayNode, RelayPriority};
use crate::types::VehicleState;

/// Navigation estimate output.
#[derive(Debug, Clone, Copy, Default)]
pub struct SubterraneanEstimate {
    pub position_m: [f64; 3],
    pub position_sigma_m: f64,
    pub update_count: u64,
}

/// Context for subterranean training: anchors, relay nodes, radii.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubterraneanTrainingContext {
    pub anchors: Vec<SurveyAnchor>,
    pub relay_node: TunnelRelayNode,
    pub priority: RelayPriority,
    pub survey_fix_radius_m: f64,
    pub range_radius_m: f64,
    pub acoustic_ranging: bool,
    pub blackout_mesh_confidence_threshold: f64,
}

impl Default for SubterraneanTrainingContext {
    fn default() -> Self {
        Self {
            anchors: Vec::new(),
            relay_node: TunnelRelayNode::default(),
            priority: RelayPriority::Normal,
            survey_fix_radius_m: 5.0,
            range_radius_m: 50.0,
            acoustic_ranging: false,
            blackout_mesh_confidence_threshold: 0.3,
        }
    }
}

/// Position estimator with dead-reckoning + anchor corrections.
#[derive(Debug, Clone)]
pub struct SubterraneanNavigator {
    position: [f64; 3],
    sigma: f64,
    update_count: u64,
}

impl SubterraneanNavigator {
    /// Create a new navigator at a starting position.
    pub fn new(initial_position: [f64; 3], initial_sigma_m: f64) -> Self {
        Self {
            position: initial_position,
            sigma: initial_sigma_m,
            update_count: 0,
        }
    }

    /// Ingest vehicle odometry (dead reckoning).
    pub fn ingest_vehicle_odometry(&mut self, state: &VehicleState) {
        self.position[0] = state.position_x;
        self.position[1] = state.position_y;
        // Sigma grows slightly from dead reckoning error
        self.sigma += 0.001;
    }

    /// Ingest a survey fix from an admitted anchor (high precision).
    pub fn ingest_survey_fix(
        &mut self,
        _bridge: &SubterraneanBridge,
        _timestamp: f64,
        anchor: &SurveyAnchor,
    ) {
        if anchor.admitted {
            // Survey fix is precise — teleport estimate to anchor position
            self.position = anchor.position_m;
            self.sigma = (self.sigma * 0.3).max(0.5);
            self.update_count += 1;
        }
    }

    /// Ingest an anchor range measurement (lower precision).
    pub fn ingest_anchor_range(
        &mut self,
        _bridge: &SubterraneanBridge,
        _timestamp: f64,
        anchor: &SurveyAnchor,
        _distance_m: f64,
        sigma_m: f64,
        _acoustic: bool,
    ) {
        // Partial correction toward anchor, weighted by inverse sigma
        let weight = 1.0 / (sigma_m.max(0.1));
        let normalization = 1.0 + weight;
        for i in 0..3 {
            self.position[i] =
                (self.position[i] + weight * anchor.position_m[i]) / normalization;
        }
        self.sigma = (self.sigma * 0.9 + sigma_m * 0.1).max(0.5);
        self.update_count += 1;
    }

    /// Current estimate.
    pub fn estimate(&self) -> SubterraneanEstimate {
        SubterraneanEstimate {
            position_m: self.position,
            position_sigma_m: self.sigma,
            update_count: self.update_count,
        }
    }
}

/// Find the nearest anchor to a given ego position.
pub fn nearest_subterranean_anchor(
    anchors: &[SurveyAnchor],
    ego_position: [f64; 3],
) -> Option<(&SurveyAnchor, f64)> {
    anchors.iter()
        .filter(|a| a.admitted)
        .map(|a| {
            let dx = a.position_m[0] - ego_position[0];
            let dy = a.position_m[1] - ego_position[1];
            let dz = a.position_m[2] - ego_position[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            (a, dist)
        })
        .min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_navigator_basic() {
        let mut nav = SubterraneanNavigator::new([0.0, 0.0, 0.0], 10.0);
        let e = nav.estimate();
        assert_eq!(e.position_m, [0.0, 0.0, 0.0]);
        assert_eq!(e.position_sigma_m, 10.0);
        assert_eq!(e.update_count, 0);
    }

    #[test]
    fn test_survey_fix_reduces_sigma() {
        let mut nav = SubterraneanNavigator::new([5.0, 5.0, 0.0], 10.0);
        let bridge = SubterraneanBridge::new();
        let anchor = SurveyAnchor {
            anchor_id: "a1".to_string(),
            position_m: [0.0, 0.0, 0.0],
            admitted: true,
        };
        nav.ingest_survey_fix(&bridge, 1.0, &anchor);
        let e = nav.estimate();
        assert_eq!(e.position_m, [0.0, 0.0, 0.0]);
        assert!(e.position_sigma_m < 10.0);
        assert_eq!(e.update_count, 1);
    }

    #[test]
    fn test_nearest_anchor() {
        let anchors = vec![
            SurveyAnchor { anchor_id: "a".to_string(), position_m: [10.0, 0.0, 0.0], admitted: true },
            SurveyAnchor { anchor_id: "b".to_string(), position_m: [0.0, 5.0, 0.0], admitted: true },
        ];
        let (nearest, dist) = nearest_subterranean_anchor(&anchors, [0.0, 0.0, 0.0]).unwrap();
        assert_eq!(nearest.anchor_id, "b");
        assert!((dist - 5.0).abs() < 0.01);
    }
}
