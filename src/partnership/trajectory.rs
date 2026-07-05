// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::relational_consciousness::RelationshipStage;

/// A single observation along the relationship trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    pub time: f64,
    pub stage: RelationshipStage,
    pub phi_dyad: f64,
}

/// Summary of how a relationship is evolving.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendSummary {
    pub phi_delta: f64,
    pub stages_visited: Vec<RelationshipStage>,
}

/// Simple in-memory relationship trajectory.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct RelationshipTrajectory {
    points: Vec<TrajectoryPoint>,
}

impl RelationshipTrajectory {
    /// Record a new trajectory point.
    pub fn record(&mut self, time: f64, stage: RelationshipStage, phi_dyad: f64) {
        self.points.push(TrajectoryPoint {
            time,
            stage,
            phi_dyad,
        });
    }

    /// Get the latest point, if any.
    pub fn latest(&self) -> Option<&TrajectoryPoint> {
        self.points.last()
    }

    /// Compute a very simple trend summary (first vs last Φ and stages visited).
    pub fn trend(&self) -> Option<TrendSummary> {
        if self.points.len() < 2 {
            return None;
        }

        let first = &self.points[0];
        let last = self.points.last()?;

        let phi_delta = last.phi_dyad - first.phi_dyad;
        let stages_visited = self.points.iter().map(|p| p.stage).collect();

        Some(TrendSummary {
            phi_delta,
            stages_visited,
        })
    }

    /// Get all recorded points.
    pub fn points(&self) -> &[TrajectoryPoint] {
        &self.points
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::relational_consciousness::RelationshipStage;

    #[test]
    fn test_empty_trajectory() {
        let traj = RelationshipTrajectory::default();
        assert!(traj.latest().is_none());
        assert!(traj.trend().is_none());
        assert!(traj.points().is_empty());
    }

    #[test]
    fn test_single_point_no_trend() {
        let mut traj = RelationshipTrajectory::default();
        traj.record(0.0, RelationshipStage::NoRelation, 0.1);
        assert!(traj.latest().is_some());
        assert!(traj.trend().is_none(), "Trend requires at least 2 points");
        assert_eq!(traj.points().len(), 1);
    }

    #[test]
    fn test_trend_with_two_points() {
        let mut traj = RelationshipTrajectory::default();
        traj.record(0.0, RelationshipStage::NoRelation, 0.1);
        traj.record(1.0, RelationshipStage::Awareness, 0.4);

        let trend = traj.trend().expect("Should have trend with 2 points");
        assert!(
            (trend.phi_delta - 0.3).abs() < 1e-10,
            "phi_delta should be 0.3"
        );
        assert_eq!(trend.stages_visited.len(), 2);
    }

    #[test]
    fn test_latest_returns_last_recorded() {
        let mut traj = RelationshipTrajectory::default();
        traj.record(0.0, RelationshipStage::NoRelation, 0.1);
        traj.record(1.0, RelationshipStage::Awareness, 0.2);
        traj.record(2.0, RelationshipStage::Contact, 0.5);

        let latest = traj.latest().unwrap();
        assert_eq!(latest.time, 2.0);
        assert_eq!(latest.phi_dyad, 0.5);
        assert_eq!(latest.stage, RelationshipStage::Contact);
    }

    #[test]
    fn test_trend_declining_phi() {
        let mut traj = RelationshipTrajectory::default();
        traj.record(0.0, RelationshipStage::Attunement, 0.8);
        traj.record(1.0, RelationshipStage::Contact, 0.3);

        let trend = traj.trend().unwrap();
        assert!(
            trend.phi_delta < 0.0,
            "Declining phi should give negative delta"
        );
    }

    #[test]
    fn test_trend_stages_visited_includes_all() {
        let mut traj = RelationshipTrajectory::default();
        traj.record(0.0, RelationshipStage::NoRelation, 0.0);
        traj.record(1.0, RelationshipStage::Awareness, 0.1);
        traj.record(2.0, RelationshipStage::Contact, 0.2);
        traj.record(3.0, RelationshipStage::Attunement, 0.3);

        let trend = traj.trend().unwrap();
        assert_eq!(trend.stages_visited.len(), 4);
    }

    #[test]
    fn test_record_preserves_order() {
        let mut traj = RelationshipTrajectory::default();
        for i in 0..10 {
            traj.record(i as f64, RelationshipStage::NoRelation, i as f64 * 0.1);
        }
        let points = traj.points();
        for i in 1..points.len() {
            assert!(points[i].time > points[i - 1].time);
        }
    }
}
