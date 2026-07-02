// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conjunction Analysis and Collision Probability
//!
//! Implements algorithms for:
//! - Close approach detection (screening)
//! - Probability of collision (Pc) calculation
//! - Conjunction Data Message (CDM) generation
//! - Risk assessment and prioritization
//!
//! # Key Concepts
//!
//! **Miss Distance**: The minimum distance between two objects at closest approach.
//! Alone, this is insufficient - a 1km miss with 10km uncertainty is dangerous,
//! but a 1km miss with 10m uncertainty is safe.
//!
//! **Probability of Collision (Pc)**: The integral of the combined position PDF
//! over the hard-body radius. This properly accounts for uncertainty.
//!
//! **Hard-Body Radius (HBR)**: Combined size of both objects plus safety margin.
//! Typically 5-50 meters depending on object sizes.

use chrono::{DateTime, Utc};
use nalgebra::{Matrix2, Vector3};
use serde::{Deserialize, Serialize};

use crate::covariance::CovarianceMatrix;
use crate::state::{OrbitalState, StateVector};

/// Result of conjunction screening
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConjunctionAssessment {
    /// Time of closest approach (TCA)
    pub tca: DateTime<Utc>,

    /// Primary object (the one we're protecting)
    pub primary_norad_id: u32,

    /// Secondary object (potential threat)
    pub secondary_norad_id: u32,

    /// Miss distance at TCA (km)
    pub miss_distance_km: f64,

    /// Relative velocity at TCA (km/s)
    pub relative_velocity_kms: f64,

    /// Probability of collision
    pub collision_probability: CollisionProbability,

    /// Combined hard-body radius used (meters)
    pub hard_body_radius_m: f64,

    /// Risk assessment
    pub risk_level: RiskLevel,

    /// Primary state at TCA
    pub primary_state: StateVector,

    /// Secondary state at TCA
    pub secondary_state: StateVector,

    /// Combined covariance at TCA (if available)
    pub combined_covariance: Option<CovarianceMatrix>,
}

/// Probability of collision with confidence
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CollisionProbability {
    /// Point estimate of Pc
    pub pc: f64,

    /// Lower bound (conservative)
    pub pc_lower: f64,

    /// Upper bound (worst case)
    pub pc_upper: f64,

    /// Method used for calculation
    pub method: PcMethod,

    /// Whether covariance data was available
    pub has_covariance: bool,
}

/// Method used for Pc calculation
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum PcMethod {
    /// Alfano's method (2D projection)
    Alfano2D,
    /// Foster's method (1D simplification)
    Foster1D,
    /// Monte Carlo simulation
    MonteCarlo { samples: u32 },
    /// Estimated from miss distance only (low confidence)
    MissDistanceOnly,
}

/// Risk level for prioritization
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    /// No action needed (Pc < 1e-7 or miss > 5km)
    Negligible,
    /// Monitor but no immediate action (Pc < 1e-5)
    Low,
    /// Increased monitoring, prepare for possible action (Pc < 1e-4)
    Medium,
    /// Maneuver planning should begin (Pc < 1e-3)
    High,
    /// Emergency - maneuver required (Pc >= 1e-3)
    Emergency,
}

impl RiskLevel {
    pub fn from_pc(pc: f64) -> Self {
        if pc >= 1e-3 {
            RiskLevel::Emergency
        } else if pc >= 1e-4 {
            RiskLevel::High
        } else if pc >= 1e-5 {
            RiskLevel::Medium
        } else if pc >= 1e-7 {
            RiskLevel::Low
        } else {
            RiskLevel::Negligible
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            RiskLevel::Negligible => "No action required",
            RiskLevel::Low => "Standard monitoring",
            RiskLevel::Medium => "Enhanced monitoring, evaluate options",
            RiskLevel::High => "Begin maneuver planning",
            RiskLevel::Emergency => "Execute collision avoidance maneuver",
        }
    }
}

/// Conjunction analyzer for computing collision probabilities
pub struct ConjunctionAnalyzer {
    /// Default hard-body radius (meters)
    default_hbr_m: f64,

    /// Screening threshold (km)
    screening_threshold_km: f64,

    /// Time step for TCA refinement (seconds)
    _tca_refinement_step_s: f64,
}

impl Default for ConjunctionAnalyzer {
    fn default() -> Self {
        Self {
            default_hbr_m: 20.0,         // 20 meters combined radius
            screening_threshold_km: 5.0, // Screen for approaches within 5 km
            _tca_refinement_step_s: 1.0, // 1-second refinement
        }
    }
}

impl ConjunctionAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the hard-body radius
    pub fn with_hbr(mut self, hbr_m: f64) -> Self {
        self.default_hbr_m = hbr_m;
        self
    }

    /// Set screening threshold
    pub fn with_screening_threshold(mut self, threshold_km: f64) -> Self {
        self.screening_threshold_km = threshold_km;
        self
    }

    /// Assess conjunction between two states at same epoch
    pub fn assess(
        &self,
        primary: &OrbitalState,
        secondary: &OrbitalState,
    ) -> ConjunctionAssessment {
        let miss_distance = primary.state.distance_to(&secondary.state);
        let relative_velocity = primary.state.relative_velocity(&secondary.state);

        // Calculate Pc
        let collision_probability =
            self.calculate_pc(primary, secondary, miss_distance, self.default_hbr_m);

        let risk_level = RiskLevel::from_pc(collision_probability.pc);

        // Combine covariances if both available
        let combined_covariance = match (&primary.covariance, &secondary.covariance) {
            (Some(c1), Some(c2)) => c1.fuse(c2),
            (Some(c), None) | (None, Some(c)) => Some(c.clone()),
            (None, None) => None,
        };

        ConjunctionAssessment {
            tca: primary.epoch,
            primary_norad_id: primary.norad_id,
            secondary_norad_id: secondary.norad_id,
            miss_distance_km: miss_distance,
            relative_velocity_kms: relative_velocity,
            collision_probability,
            hard_body_radius_m: self.default_hbr_m,
            risk_level,
            primary_state: primary.state.clone(),
            secondary_state: secondary.state.clone(),
            combined_covariance,
        }
    }

    /// Calculate probability of collision
    fn calculate_pc(
        &self,
        primary: &OrbitalState,
        secondary: &OrbitalState,
        miss_distance_km: f64,
        hbr_m: f64,
    ) -> CollisionProbability {
        let hbr_km = hbr_m / 1000.0;

        // Get covariances
        let has_covariance = primary.has_covariance() || secondary.has_covariance();

        if !has_covariance {
            // Fallback: estimate Pc from miss distance alone
            // This is very approximate and should be flagged
            return self.pc_from_miss_distance(miss_distance_km, hbr_km);
        }

        // Get combined covariance in encounter frame
        let cov_primary = primary.covariance.as_ref().cloned().unwrap_or_default();
        let cov_secondary = secondary.covariance.as_ref().cloned().unwrap_or_default();

        // Combined position covariance
        let combined_pos_cov =
            cov_primary.position_covariance() + cov_secondary.position_covariance();

        // Relative position (miss vector)
        let miss_vector = secondary.state.position() - primary.state.position();

        // Relative velocity for encounter frame
        let rel_vel = secondary.state.velocity() - primary.state.velocity();

        // Project to 2D encounter plane (perpendicular to relative velocity)
        let (projected_miss, cov_2d) =
            self.project_to_encounter_plane(miss_vector, rel_vel, combined_pos_cov);

        // Alfano's 2D Pc calculation
        let pc = self.alfano_2d_pc(projected_miss, cov_2d, hbr_km);

        // Estimate bounds (±1 order of magnitude for TLE-based)
        let pc_lower = pc / 10.0;
        let pc_upper = (pc * 10.0).min(1.0);

        CollisionProbability {
            pc,
            pc_lower,
            pc_upper,
            method: PcMethod::Alfano2D,
            has_covariance: true,
        }
    }

    /// Fallback Pc estimation from miss distance only
    fn pc_from_miss_distance(&self, miss_km: f64, hbr_km: f64) -> CollisionProbability {
        // Very rough approximation assuming typical LEO uncertainties
        // This should be used only as a screening metric
        let assumed_sigma_km = 1.0; // Assume 1 km 1-sigma position uncertainty

        // Simple Gaussian approximation
        let x = miss_km / assumed_sigma_km;
        let pc = (-(x * x) / 2.0).exp() * (hbr_km / assumed_sigma_km).powi(2);

        CollisionProbability {
            pc: pc.min(1.0),
            pc_lower: pc / 100.0, // Very uncertain
            pc_upper: (pc * 100.0).min(1.0),
            method: PcMethod::MissDistanceOnly,
            has_covariance: false,
        }
    }

    /// Project to 2D encounter plane
    fn project_to_encounter_plane(
        &self,
        miss_vector: Vector3<f64>,
        rel_velocity: Vector3<f64>,
        cov_3d: nalgebra::Matrix3<f64>,
    ) -> (nalgebra::Vector2<f64>, Matrix2<f64>) {
        // Build encounter frame basis
        // Guard against zero relative velocity (co-moving objects)
        let v_hat = if rel_velocity.norm() < 1e-10 {
            // Fall back to miss vector direction, or arbitrary axis
            if miss_vector.norm() < 1e-10 {
                Vector3::new(0.0, 0.0, 1.0)
            } else {
                miss_vector.normalize()
            }
        } else {
            rel_velocity.normalize()
        };

        // Choose arbitrary perpendicular vector
        let temp = if v_hat.x.abs() < 0.9 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };

        let u1 = v_hat.cross(&temp).normalize();
        let u2 = v_hat.cross(&u1);

        // Project miss vector to encounter plane
        let miss_2d = nalgebra::Vector2::new(miss_vector.dot(&u1), miss_vector.dot(&u2));

        // Project covariance to encounter plane
        // C_2d = P * C_3d * P^T where P is 2x3 projection matrix
        let proj = nalgebra::Matrix2x3::from_rows(&[u1.transpose(), u2.transpose()]);

        let cov_2d = proj * cov_3d * proj.transpose();

        (miss_2d, cov_2d)
    }

    /// Alfano's 2D Pc calculation
    fn alfano_2d_pc(
        &self,
        miss_2d: nalgebra::Vector2<f64>,
        cov_2d: Matrix2<f64>,
        hbr_km: f64,
    ) -> f64 {
        // Eigendecomposition of covariance
        let det = cov_2d.determinant();
        if det <= 0.0 {
            return 0.0; // Invalid covariance
        }

        let trace = cov_2d.trace();
        let discriminant = (trace * trace - 4.0 * det).max(0.0);
        let sqrt_disc = discriminant.sqrt();

        let lambda1 = (trace + sqrt_disc) / 2.0;
        let lambda2 = (trace - sqrt_disc) / 2.0;

        let sigma1 = lambda1.sqrt();
        let sigma2 = lambda2.sqrt();

        // Normalized miss distance
        let inv_cov = cov_2d.try_inverse().unwrap_or(Matrix2::identity());
        let d_squared = (miss_2d.transpose() * inv_cov * miss_2d)[(0, 0)];

        // Hard-body radius normalized
        let r_norm = hbr_km / (sigma1 * sigma2).sqrt();

        // Pc approximation (Foster-style simplification of full integral)
        // Pc ≈ (π * r²) / (2π * σ1 * σ2) * exp(-d²/2)
        let pc = (r_norm * r_norm) * (-d_squared / 2.0).exp() / 2.0;

        pc.clamp(0.0, 1.0)
    }
}

/// Screen multiple objects for conjunctions
pub fn screen_catalog(
    protected: &[OrbitalState],
    catalog: &[OrbitalState],
    threshold_km: f64,
) -> Vec<(u32, u32, f64)> {
    let mut conjunctions = Vec::new();

    for p in protected {
        for s in catalog {
            if p.norad_id == s.norad_id {
                continue;
            }

            let miss = p.state.distance_to(&s.state);
            if miss < threshold_km {
                conjunctions.push((p.norad_id, s.norad_id, miss));
            }
        }
    }

    // Sort by miss distance
    conjunctions.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    conjunctions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::covariance::CovarianceMatrix;

    #[test]
    fn test_risk_level_from_pc() {
        assert_eq!(RiskLevel::from_pc(1e-8), RiskLevel::Negligible);
        assert_eq!(RiskLevel::from_pc(1e-6), RiskLevel::Low);
        assert_eq!(RiskLevel::from_pc(5e-5), RiskLevel::Medium);
        assert_eq!(RiskLevel::from_pc(5e-4), RiskLevel::High);
        assert_eq!(RiskLevel::from_pc(5e-3), RiskLevel::Emergency);
    }

    #[test]
    fn test_conjunction_assessment() {
        use crate::state::DataSource;
        use chrono::Utc;

        let now = Utc::now();

        // Two objects 1 km apart
        let primary = OrbitalState::new(
            25544,
            now,
            StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0),
            DataSource::SpaceTrack,
        )
        .with_covariance(CovarianceMatrix::diagonal([
            0.5, 0.5, 0.5, 0.001, 0.001, 0.001,
        ]));

        let secondary = OrbitalState::new(
            12345,
            now,
            StateVector::new(7001.0, 0.0, 0.0, 0.0, 7.5, 0.0),
            DataSource::SpaceTrack,
        )
        .with_covariance(CovarianceMatrix::diagonal([
            0.5, 0.5, 0.5, 0.001, 0.001, 0.001,
        ]));

        let analyzer = ConjunctionAnalyzer::new();
        let assessment = analyzer.assess(&primary, &secondary);

        assert!((assessment.miss_distance_km - 1.0).abs() < 0.01);
        assert!(assessment.collision_probability.has_covariance);
    }

    /// End-to-end pipeline test: TLE parse -> SGP4 propagate -> conjunction assess
    #[test]
    fn test_tle_to_conjunction_pipeline() {
        use crate::propagator::Propagator;
        use crate::tle::TwoLineElement;
        use chrono::Duration;

        // ISS TLE
        let iss_tle = TwoLineElement::parse(
            "ISS (ZARYA)\n\
             1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997\n\
             2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391424577",
        )
        .expect("Failed to parse ISS TLE");

        // Fictional debris TLE (same epoch, slightly different orbit)
        let debris_tle = TwoLineElement::parse(
            "DEBRIS\n\
             1 49863U 21999A   24001.50000000  .00001000  00000-0  50000-4 0  9992\n\
             2 49863  51.6500 247.5000 0007000 131.0000 325.0000 15.72000000 10005",
        )
        .expect("Failed to parse debris TLE");

        // Create propagators
        let iss_prop = Propagator::from_tle(&iss_tle).expect("ISS propagator failed");
        let debris_prop = Propagator::from_tle(&debris_tle).expect("Debris propagator failed");

        // Propagate both to 1 hour after epoch
        let screen_time = iss_tle.epoch + Duration::hours(1);
        let iss_state = iss_prop
            .propagate_to(screen_time)
            .expect("ISS propagation failed");
        let debris_state = debris_prop
            .propagate_to(screen_time)
            .expect("Debris propagation failed");

        // Assess conjunction
        let analyzer = ConjunctionAnalyzer::new();
        let assessment = analyzer.assess(&iss_state, &debris_state);

        // Verify the pipeline produced meaningful results
        assert_eq!(assessment.primary_norad_id, 25544);
        assert_eq!(assessment.secondary_norad_id, 49863);
        assert!(
            assessment.miss_distance_km >= 0.0,
            "Miss distance must be non-negative"
        );
        assert!(
            assessment.relative_velocity_kms >= 0.0,
            "Relative velocity must be non-negative"
        );
        assert!(
            assessment.collision_probability.pc >= 0.0,
            "Pc must be non-negative"
        );
        assert!(
            assessment.collision_probability.pc <= 1.0,
            "Pc must be <= 1.0"
        );
        assert!(assessment.hard_body_radius_m > 0.0, "HBR must be positive");

        // The assessment should have covariance (propagator estimates it from TLE age)
        assert!(assessment.collision_probability.has_covariance);

        // Risk level should be consistent with Pc
        let expected_risk = RiskLevel::from_pc(assessment.collision_probability.pc);
        assert_eq!(assessment.risk_level, expected_risk);
    }

    #[test]
    fn test_screening() {
        use crate::state::DataSource;
        use chrono::Utc;

        let now = Utc::now();

        let objects: Vec<OrbitalState> = (0..5)
            .map(|i| {
                OrbitalState::new(
                    25544 + i,
                    now,
                    StateVector::new(7000.0 + i as f64, 0.0, 0.0, 0.0, 7.5, 0.0),
                    DataSource::SpaceTrack,
                )
            })
            .collect();

        let conjunctions = screen_catalog(&objects[0..1], &objects, 2.0);

        // Should find object 1 (1 km away) but not object 4 (4 km away)
        assert!(conjunctions.iter().any(|(_, s, _)| *s == 25545));
        assert!(!conjunctions.iter().any(|(_, s, _)| *s == 25548));
    }
}
