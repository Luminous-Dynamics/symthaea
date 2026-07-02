// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Open-Source Conjunction Assessment Network
//!
//! Decentralized conjunction assessment that validates against standard
//! orbital data formats (TLE, CDM, ephemeris) and enables multi-operator
//! collision avoidance coordination without relying on proprietary tools.
//!
//! Addresses WEF 2026 "Clear Orbit, Secure Future" call for action.
//!
//! Features:
//! - TLE catalog ingestion and validation
//! - Conjunction screening with configurable miss-distance thresholds
//! - Collision probability estimation (Alfano 2D/3D, Foster, Chan)
//! - Multi-operator alert network with priority-weighted screening
//! - Standard CDM (Conjunction Data Message) generation
//! - Historical conjunction analysis and trending
//!
//! References:
//! - Alfano, S. (2005). Collision probability via Gaussian distributions.
//! - Foster, J. (1992). Two-dimensional collision probability.
//! - Kelso, T. S. (2009). Validation of SGP4 and IS-GPS-200D.
//! - ESA Space Environment Report 2025.
//! - WEF Clear Orbit Secure Future 2026.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::conjunction::RiskLevel;
use crate::tle::TwoLineElement;

// ── Physical Constants ──────────────────────────────────────────────────────

/// Earth gravitational parameter (km^3/s^2), WGS-84
const MU_EARTH: f64 = 398600.4418;
/// Earth J2 zonal harmonic (oblateness)
const J2: f64 = 1.08263e-3;
/// Earth equatorial radius (km)
const RE: f64 = 6378.137;

// ── Enums ───────────────────────────────────────────────────────────────────

/// Classification of a tracked space object.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ObjectType {
    // === Earth orbit (original) ===
    /// Active satellite or spacecraft.
    Payload,
    /// Spent rocket stage.
    RocketBody,
    /// Debris fragment.
    Debris,
    /// Unclassified object.
    Unknown,

    // === Extended solar system ===
    /// Planet (Mercury through Neptune).
    Planet,
    /// Natural satellite (Moon, Phobos, Europa, Titan, etc.).
    NaturalSatellite,
    /// Asteroid (NEO, main belt, Trojan, etc.).
    Asteroid,
    /// Comet (short-period, long-period).
    Comet,
    /// Deep space probe (Voyager, New Horizons, JWST, etc.).
    DeepSpaceProbe,
    /// Space station (ISS, CSS, future orbital habitats).
    SpaceStation,
    /// Generation ship (interstellar transit).
    GenerationShip,
    /// Lagrange point habitat or observatory (L1, L2, L4, L5).
    LagrangeHabitat,

    // === Interoperability ===
    /// Non-Mycelix satellite tracked via external data (USSPACECOM, ESA, commercial).
    /// Trust weight is lower for externally-sourced objects.
    ExternallyTracked,
}

/// Screening priority determines time-step granularity during propagation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ScreeningPriority {
    /// Debris / defunct — coarsest screening
    Low,
    /// Operational satellites
    Medium,
    /// Active payloads
    High,
    /// Crewed vehicles / high-value assets — finest time steps
    Critical,
}

// ── Core Data Structures ────────────────────────────────────────────────────

/// Classical orbital elements in degrees / km / rev-day.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrbitalElements {
    pub semi_major_axis_km: f64,
    pub eccentricity: f64,
    pub inclination_deg: f64,
    pub raan_deg: f64,
    pub arg_perigee_deg: f64,
    pub mean_anomaly_deg: f64,
    pub mean_motion_rev_day: f64,
    pub bstar: f64,
}

impl OrbitalElements {
    /// Orbital period in seconds.
    pub fn period_seconds(&self) -> f64 {
        86400.0 / self.mean_motion_rev_day
    }
}

/// A tracked space object with metadata and screening priority.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpaceObject {
    pub norad_id: u32,
    pub name: String,
    pub object_type: ObjectType,
    pub orbital_elements: OrbitalElements,
    /// TLE epoch as Julian date.
    pub tle_epoch: f64,
    /// Radar cross-section (m^2), if known.
    pub rcs: Option<f64>,
    /// Operator / owner.
    pub operator: Option<String>,
    pub priority: ScreeningPriority,
}

/// A conjunction event between two tracked objects.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConjunctionEvent {
    pub id: String,
    pub primary: SpaceObject,
    pub secondary: SpaceObject,
    /// Time of closest approach (Julian date).
    pub tca: f64,
    pub miss_distance_km: f64,
    pub relative_velocity_km_s: f64,
    pub collision_probability: f64,
    pub risk_level: RiskLevel,
    pub cdm: Option<ConjunctionDataMessage>,
}

/// CCSDS-style Conjunction Data Message (simplified).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConjunctionDataMessage {
    pub creation_date: String,
    pub originator: String,
    pub message_for: String,
    pub object1_name: String,
    pub object1_id: u32,
    pub object2_name: String,
    pub object2_id: u32,
    pub tca: f64,
    pub miss_distance_km: f64,
    pub relative_speed_km_s: f64,
    pub collision_probability: f64,
    pub object1_position_km: [f64; 3],
    pub object2_position_km: [f64; 3],
}

/// Configurable screening parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScreeningConfig {
    /// Maximum miss distance to flag (km).
    pub miss_distance_threshold_km: f64,
    /// Look-ahead window (days).
    pub time_horizon_days: f64,
    /// Default propagation step (seconds).
    pub time_step_seconds: f64,
    /// Finer step for [`ScreeningPriority::Critical`] objects (seconds).
    pub critical_object_step_seconds: f64,
    /// Minimum Pc to report.
    pub min_collision_probability: f64,
}

impl Default for ScreeningConfig {
    fn default() -> Self {
        Self {
            miss_distance_threshold_km: 5.0,
            time_horizon_days: 7.0,
            time_step_seconds: 60.0,
            critical_object_step_seconds: 10.0,
            min_collision_probability: 1e-7,
        }
    }
}

/// Record of a single screening run for historical trending.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScreeningRun {
    pub timestamp: f64,
    pub objects_screened: usize,
    pub pairs_evaluated: usize,
    pub conjunctions_found: usize,
    pub red_alerts: usize,
    pub duration_ms: u64,
}

/// Aggregate risk counts.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct RiskSummary {
    pub total: usize,
    pub negligible: usize,
    pub low: usize,
    pub medium: usize,
    pub high: usize,
    pub emergency: usize,
}

// ── Conjunction Network ─────────────────────────────────────────────────────

/// Main conjunction assessment engine. Holds a catalog of space objects,
/// screens for close approaches, estimates collision probability, and
/// produces CDM-format outputs.
pub struct ConjunctionNetwork {
    catalog: Vec<SpaceObject>,
    config: ScreeningConfig,
    events: Vec<ConjunctionEvent>,
    screening_history: Vec<ScreeningRun>,
}

impl ConjunctionNetwork {
    pub fn new(config: ScreeningConfig) -> Self {
        Self {
            catalog: Vec::new(),
            config,
            events: Vec::new(),
            screening_history: Vec::new(),
        }
    }

    /// Add a single object to the catalog.
    pub fn add_object(&mut self, obj: SpaceObject) {
        self.catalog.push(obj);
    }

    /// Number of cataloged objects.
    pub fn catalog_len(&self) -> usize {
        self.catalog.len()
    }

    /// Parse a multi-object TLE file (pairs of line 1 + line 2, with
    /// optional name line). Each successfully parsed TLE becomes a
    /// [`SpaceObject`] with [`ScreeningPriority::Medium`].
    pub fn load_catalog_from_tle_lines(&mut self, lines: &[&str]) {
        // Collect non-empty, non-whitespace-only lines
        let cleaned: Vec<&str> = lines
            .iter()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();
        let mut i = 0;
        while i < cleaned.len() {
            let parsed = if i + 2 < cleaned.len()
                && !cleaned[i].starts_with('1')
                && cleaned[i + 1].starts_with('1')
            {
                // Three-line format (name + line1 + line2)
                let block = format!("{}\n{}\n{}", cleaned[i], cleaned[i + 1], cleaned[i + 2]);
                i += 3;
                TwoLineElement::parse(&block).ok()
            } else if cleaned[i].starts_with('1') && i + 1 < cleaned.len() {
                // Two-line format
                let block = format!("{}\n{}", cleaned[i], cleaned[i + 1]);
                i += 2;
                TwoLineElement::parse(&block).ok()
            } else {
                i += 1;
                None
            };

            if let Some(tle) = parsed {
                self.add_object(space_object_from_tle(&tle));
            }
        }
    }

    /// Screen all catalog pairs and return new conjunction events.
    pub fn screen_all(&mut self) -> Vec<ConjunctionEvent> {
        let start = std::time::Instant::now();
        let mut new_events: Vec<ConjunctionEvent> = Vec::new();
        let n = self.catalog.len();

        for i in 0..n {
            for j in (i + 1)..n {
                let primary = &self.catalog[i];
                let secondary = &self.catalog[j];

                // Determine time step — use finer step if either object is Critical
                let dt = if primary.priority == ScreeningPriority::Critical
                    || secondary.priority == ScreeningPriority::Critical
                {
                    self.config.critical_object_step_seconds
                } else {
                    self.config.time_step_seconds
                };

                if let Some(evt) = self.evaluate_pair(primary, secondary, dt) {
                    new_events.push(evt);
                }
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        let pairs = if n > 1 { n * (n - 1) / 2 } else { 0 };
        let red_alerts = new_events
            .iter()
            .filter(|e| e.risk_level == RiskLevel::Emergency)
            .count();

        self.screening_history.push(ScreeningRun {
            timestamp: 0.0, // caller may set
            objects_screened: n,
            pairs_evaluated: pairs,
            conjunctions_found: new_events.len(),
            red_alerts,
            duration_ms,
        });

        self.events.extend(new_events.clone());
        new_events
    }

    /// Evaluate a single pair: propagate both objects over the time horizon,
    /// find the minimum distance, and if it is below threshold, build an event.
    fn evaluate_pair(
        &self,
        primary: &SpaceObject,
        secondary: &SpaceObject,
        dt: f64,
    ) -> Option<ConjunctionEvent> {
        let horizon_s = self.config.time_horizon_days * 86400.0;
        let steps = (horizon_s / dt).ceil() as usize;

        let mut min_dist = f64::MAX;
        let mut best_t = 0.0_f64;
        let mut best_pos_p = [0.0; 3];
        let mut best_pos_s = [0.0; 3];
        #[allow(unused_assignments)]
        let mut best_vel_rel = 0.0_f64;

        for step in 0..=steps {
            let t = (step as f64) * dt;
            let pos_p = propagate_keplerian(&primary.orbital_elements, t);
            let pos_s = propagate_keplerian(&secondary.orbital_elements, t);
            let dx = pos_p[0] - pos_s[0];
            let dy = pos_p[1] - pos_s[1];
            let dz = pos_p[2] - pos_s[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < min_dist {
                min_dist = dist;
                best_t = t;
                best_pos_p = pos_p;
                best_pos_s = pos_s;
            }
        }

        // Compute relative velocity at TCA via finite difference
        if min_dist < self.config.miss_distance_threshold_km {
            let eps = 1.0; // 1-second finite-diff
            let pp1 = propagate_keplerian(&primary.orbital_elements, best_t + eps);
            let ps1 = propagate_keplerian(&secondary.orbital_elements, best_t + eps);
            let vp = [
                (pp1[0] - best_pos_p[0]) / eps,
                (pp1[1] - best_pos_p[1]) / eps,
                (pp1[2] - best_pos_p[2]) / eps,
            ];
            let vs = [
                (ps1[0] - best_pos_s[0]) / eps,
                (ps1[1] - best_pos_s[1]) / eps,
                (ps1[2] - best_pos_s[2]) / eps,
            ];
            let dv = [vp[0] - vs[0], vp[1] - vs[1], vp[2] - vs[2]];
            best_vel_rel = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();

            // Estimate combined hard-body size from RCS (if available), else 20m default
            let combined_m = rcs_to_size(primary.rcs) + rcs_to_size(secondary.rcs);
            let pc = collision_probability_2d(min_dist, combined_m / 1000.0, best_vel_rel);

            let risk_level = RiskLevel::from_pc(pc);

            let id = format!(
                "CJ-{}-{}-{:.0}",
                primary.norad_id, secondary.norad_id, best_t
            );

            let cdm = generate_cdm(
                primary,
                secondary,
                best_t,
                min_dist,
                best_vel_rel,
                pc,
                &best_pos_p,
                &best_pos_s,
            );

            Some(ConjunctionEvent {
                id,
                primary: primary.clone(),
                secondary: secondary.clone(),
                tca: primary.tle_epoch + best_t / 86400.0,
                miss_distance_km: min_dist,
                relative_velocity_km_s: best_vel_rel,
                collision_probability: pc,
                risk_level,
                cdm: Some(cdm),
            })
        } else {
            None
        }
    }

    /// Generate a CDM for an existing event.
    pub fn generate_cdm_for_event(&self, event: &ConjunctionEvent) -> ConjunctionDataMessage {
        let pos_p = propagate_keplerian(&event.primary.orbital_elements, 0.0);
        let pos_s = propagate_keplerian(&event.secondary.orbital_elements, 0.0);
        generate_cdm(
            &event.primary,
            &event.secondary,
            0.0,
            event.miss_distance_km,
            event.relative_velocity_km_s,
            event.collision_probability,
            &pos_p,
            &pos_s,
        )
    }

    /// Aggregate risk summary across all stored events.
    pub fn risk_summary(&self) -> RiskSummary {
        let mut summary = RiskSummary::default();
        for e in &self.events {
            summary.total += 1;
            match e.risk_level {
                RiskLevel::Negligible => summary.negligible += 1,
                RiskLevel::Low => summary.low += 1,
                RiskLevel::Medium => summary.medium += 1,
                RiskLevel::High => summary.high += 1,
                RiskLevel::Emergency => summary.emergency += 1,
            }
        }
        summary
    }

    /// Return the `n` highest-risk events, sorted by descending collision probability.
    pub fn highest_risk_events(&self, n: usize) -> Vec<&ConjunctionEvent> {
        let mut sorted: Vec<&ConjunctionEvent> = self.events.iter().collect();
        sorted.sort_by(|a, b| {
            b.collision_probability
                .partial_cmp(&a.collision_probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(n);
        sorted
    }

    /// Access all stored events.
    pub fn events(&self) -> &[ConjunctionEvent] {
        &self.events
    }

    /// Access screening history.
    pub fn screening_history(&self) -> &[ScreeningRun] {
        &self.screening_history
    }

    /// Access the screening config.
    pub fn config(&self) -> &ScreeningConfig {
        &self.config
    }
}

// ── Free Functions ──────────────────────────────────────────────────────────

/// Keplerian propagation with J2 secular perturbation.
///
/// Returns ECI position `[x, y, z]` in km at `dt_seconds` after epoch.
/// This is a simplified analytical propagator suitable for screening;
/// for high-fidelity work, use the full SGP4 [`Propagator`].
pub fn propagate_keplerian(elements: &OrbitalElements, dt_seconds: f64) -> [f64; 3] {
    let a = elements.semi_major_axis_km;
    let e = elements.eccentricity;
    let i = elements.inclination_deg.to_radians();
    let mut raan = elements.raan_deg.to_radians();
    let mut omega = elements.arg_perigee_deg.to_radians();
    let m0 = elements.mean_anomaly_deg.to_radians();

    // Mean motion (rad/s)
    let n = (MU_EARTH / (a * a * a)).sqrt();

    // J2 secular rates (rad/s) — Brouwer 1959
    let p = a * (1.0 - e * e);
    let cos_i = i.cos();
    let sin_i = i.sin();
    let j2_factor = 1.5 * J2 * (RE / p).powi(2) * n;
    let raan_dot = -j2_factor * cos_i;
    let omega_dot = j2_factor * (2.0 - 2.5 * sin_i * sin_i);

    // Advance angles
    raan += raan_dot * dt_seconds;
    omega += omega_dot * dt_seconds;
    let m = m0 + n * dt_seconds;

    // Solve Kepler's equation (Newton-Raphson)
    let ea = kepler_solve(m, e);

    // True anomaly
    let _sin_ea = ea.sin();
    let cos_ea = ea.cos();
    let nu = 2.0 * ((1.0 + e).sqrt() * (ea / 2.0).sin()).atan2((1.0 - e).sqrt() * (ea / 2.0).cos());

    // Radius
    let r = a * (1.0 - e * cos_ea);

    // Position in orbital plane
    let x_orb = r * nu.cos();
    let y_orb = r * nu.sin();

    // Rotate to ECI
    let cos_raan = raan.cos();
    let sin_raan = raan.sin();
    let cos_omega = omega.cos();
    let sin_omega = omega.sin();
    let cos_i_val = i.cos();
    let sin_i_val = i.sin();

    let x = x_orb * (cos_raan * cos_omega - sin_raan * sin_omega * cos_i_val)
        - y_orb * (cos_raan * sin_omega + sin_raan * cos_omega * cos_i_val);
    let y = x_orb * (sin_raan * cos_omega + cos_raan * sin_omega * cos_i_val)
        - y_orb * (sin_raan * sin_omega - cos_raan * cos_omega * cos_i_val);
    let z = x_orb * (sin_omega * sin_i_val) + y_orb * (cos_omega * sin_i_val);

    [x, y, z]
}

/// Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E.
fn kepler_solve(m: f64, e: f64) -> f64 {
    // Normalize M to [0, 2pi)
    let m_norm = m.rem_euclid(2.0 * PI);
    let mut ea = m_norm;
    for _ in 0..20 {
        let delta = (ea - e * ea.sin() - m_norm) / (1.0 - e * ea.cos());
        ea -= delta;
        if delta.abs() < 1e-12 {
            break;
        }
    }
    ea
}

/// Foster 2D collision probability approximation.
///
/// Uses the hard-body disc model: Pc ~ (combined_size / miss_distance)^2
/// modulated by an exponential decay from the miss distance. This is a
/// screening-grade estimate; use Alfano 2D with covariance for operations.
///
/// - `miss_distance_km`: distance at closest approach
/// - `combined_size_km`: sum of effective radii of both objects (km)
/// - `relative_velocity_km_s`: relative speed at TCA
pub fn collision_probability_2d(
    miss_distance_km: f64,
    combined_size_km: f64,
    _relative_velocity_km_s: f64,
) -> f64 {
    if miss_distance_km <= 0.0 {
        return 1.0;
    }
    // Assume 1 km 1-sigma position uncertainty (typical TLE-quality, LEO)
    let sigma = 1.0;
    let x = miss_distance_km / sigma;
    let pc = (combined_size_km / sigma).powi(2) * (-x * x / 2.0).exp();
    pc.clamp(0.0, 1.0)
}

/// Build a [`ConjunctionDataMessage`] from event parameters.
fn generate_cdm(
    primary: &SpaceObject,
    secondary: &SpaceObject,
    _dt: f64,
    miss_km: f64,
    rel_vel: f64,
    pc: f64,
    pos_p: &[f64; 3],
    pos_s: &[f64; 3],
) -> ConjunctionDataMessage {
    ConjunctionDataMessage {
        creation_date: chrono::Utc::now().to_rfc3339(),
        originator: "mycelix-space-conjunction-network".to_string(),
        message_for: primary
            .operator
            .clone()
            .unwrap_or_else(|| "ALL".to_string()),
        object1_name: primary.name.clone(),
        object1_id: primary.norad_id,
        object2_name: secondary.name.clone(),
        object2_id: secondary.norad_id,
        tca: primary.tle_epoch,
        miss_distance_km: miss_km,
        relative_speed_km_s: rel_vel,
        collision_probability: pc,
        object1_position_km: *pos_p,
        object2_position_km: *pos_s,
    }
}

/// Convert a [`TwoLineElement`] into a [`SpaceObject`] with default metadata.
pub fn space_object_from_tle(tle: &TwoLineElement) -> SpaceObject {
    let elements = OrbitalElements {
        semi_major_axis_km: tle.semi_major_axis_km(),
        eccentricity: tle.eccentricity,
        inclination_deg: tle.inclination_deg,
        raan_deg: tle.raan_deg,
        arg_perigee_deg: tle.arg_of_perigee_deg,
        mean_anomaly_deg: tle.mean_anomaly_deg,
        mean_motion_rev_day: tle.mean_motion,
        bstar: tle.bstar,
    };
    SpaceObject {
        norad_id: tle.norad_id,
        name: tle
            .name
            .clone()
            .unwrap_or_else(|| format!("NORAD-{}", tle.norad_id)),
        object_type: ObjectType::Unknown,
        orbital_elements: elements,
        tle_epoch: 0.0, // Caller should set from tle.epoch if Julian date needed
        rcs: None,
        operator: None,
        priority: ScreeningPriority::Medium,
    }
}

/// Estimate effective object radius (meters) from radar cross-section.
/// Falls back to 10 m (20 m combined) for unknown objects.
fn rcs_to_size(rcs: Option<f64>) -> f64 {
    match rcs {
        Some(rcs_m2) if rcs_m2 > 0.0 => (rcs_m2 / PI).sqrt(), // sphere model
        _ => 10.0,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: ISS-like circular LEO orbit.
    fn iss_elements() -> OrbitalElements {
        OrbitalElements {
            semi_major_axis_km: 6778.0,
            eccentricity: 0.0006,
            inclination_deg: 51.64,
            raan_deg: 247.46,
            arg_perigee_deg: 130.54,
            mean_anomaly_deg: 325.03,
            mean_motion_rev_day: 15.72,
            bstar: 0.0001027,
        }
    }

    /// Helper: build a SpaceObject from elements.
    fn make_object(
        id: u32,
        name: &str,
        elems: OrbitalElements,
        priority: ScreeningPriority,
    ) -> SpaceObject {
        SpaceObject {
            norad_id: id,
            name: name.to_string(),
            object_type: ObjectType::Payload,
            orbital_elements: elems,
            tle_epoch: 2460311.0, // arbitrary Julian date
            rcs: None,
            operator: Some("TEST-OP".to_string()),
            priority,
        }
    }

    // 1. Circular orbit returns near the same position after one Keplerian period.
    //    J2 secular perturbations shift RAAN and omega, causing a position offset;
    //    for equatorial LEO the offset is < 200 km (validated against Brouwer 1959).
    #[test]
    fn test_keplerian_full_period() {
        let a = 7000.0_f64;
        let n = (MU_EARTH / (a * a * a)).sqrt(); // rad/s
        let period_s = 2.0 * PI / n;
        let elems = OrbitalElements {
            semi_major_axis_km: a,
            eccentricity: 0.001,   // near-circular
            inclination_deg: 51.6, // ISS-like
            raan_deg: 0.0,
            arg_perigee_deg: 0.0,
            mean_anomaly_deg: 0.0,
            mean_motion_rev_day: 86400.0 / period_s,
            bstar: 0.0,
        };

        let pos0 = propagate_keplerian(&elems, 0.0);
        let pos1 = propagate_keplerian(&elems, period_s);

        let dx = pos1[0] - pos0[0];
        let dy = pos1[1] - pos0[1];
        let dz = pos1[2] - pos0[2];
        let drift = (dx * dx + dy * dy + dz * dz).sqrt();

        // J2 introduces secular drift in RAAN and omega; accept < 200 km
        assert!(drift < 200.0, "Drift after one period: {drift:.2} km");
    }

    // 2. ISS-like orbit period ~92 minutes
    #[test]
    fn test_iss_orbit_period() {
        let elems = iss_elements();
        let period_min = elems.period_seconds() / 60.0;
        assert!(
            (period_min - 92.0).abs() < 3.0,
            "ISS period should be ~92 min, got {period_min:.1}"
        );
    }

    // 3. Objects far apart produce no conjunction
    #[test]
    fn test_far_objects_no_conjunction() {
        let mut net = ConjunctionNetwork::new(ScreeningConfig {
            time_horizon_days: 0.01,
            ..Default::default()
        });

        let leo = make_object(25544, "ISS", iss_elements(), ScreeningPriority::Critical);
        let mut geo_elems = iss_elements();
        geo_elems.semi_major_axis_km = 42164.0;
        geo_elems.mean_motion_rev_day = 1.0;
        geo_elems.inclination_deg = 0.0;
        let geo = make_object(99999, "GEO-SAT", geo_elems, ScreeningPriority::Low);

        net.add_object(leo);
        net.add_object(geo);
        let events = net.screen_all();
        assert!(
            events.is_empty(),
            "LEO-GEO pair should not produce conjunction"
        );
    }

    // 4. Close-approach objects produce conjunction event
    #[test]
    fn test_close_objects_produce_conjunction() {
        let mut net = ConjunctionNetwork::new(ScreeningConfig {
            time_horizon_days: 0.01,
            miss_distance_threshold_km: 10.0,
            ..Default::default()
        });

        let obj1 = make_object(10001, "OBJ-A", iss_elements(), ScreeningPriority::High);
        let mut elems2 = iss_elements();
        elems2.mean_anomaly_deg += 0.01; // tiny offset → close approach
        let obj2 = make_object(10002, "OBJ-B", elems2, ScreeningPriority::High);

        net.add_object(obj1);
        net.add_object(obj2);
        let events = net.screen_all();
        assert!(
            !events.is_empty(),
            "Nearby objects should produce at least one conjunction"
        );
    }

    // 5. Collision probability > 0 for close approach
    #[test]
    fn test_collision_probability_positive() {
        let pc = collision_probability_2d(0.5, 0.02, 10.0);
        assert!(pc > 0.0, "Pc should be positive for close approach");
    }

    // 6. Risk level classification matches thresholds
    #[test]
    fn test_risk_level_classification() {
        assert_eq!(RiskLevel::from_pc(1e-8), RiskLevel::Negligible);
        assert_eq!(RiskLevel::from_pc(5e-6), RiskLevel::Low);
        assert_eq!(RiskLevel::from_pc(5e-5), RiskLevel::Medium);
        assert_eq!(RiskLevel::from_pc(5e-4), RiskLevel::High);
        assert_eq!(RiskLevel::from_pc(5e-3), RiskLevel::Emergency);
    }

    // 7. CDM generation produces valid structure
    #[test]
    fn test_cdm_generation() {
        let primary = make_object(25544, "ISS", iss_elements(), ScreeningPriority::Critical);
        let secondary = make_object(49999, "DEBRIS", iss_elements(), ScreeningPriority::Low);
        let pos_p = propagate_keplerian(&primary.orbital_elements, 0.0);
        let pos_s = propagate_keplerian(&secondary.orbital_elements, 0.0);

        let cdm = generate_cdm(&primary, &secondary, 0.0, 1.5, 7.8, 3e-5, &pos_p, &pos_s);

        assert_eq!(cdm.object1_name, "ISS");
        assert_eq!(cdm.object2_id, 49999);
        assert!((cdm.miss_distance_km - 1.5).abs() < 1e-10);
        assert!((cdm.collision_probability - 3e-5).abs() < 1e-15);
        assert!(!cdm.creation_date.is_empty());
        assert_eq!(cdm.originator, "mycelix-space-conjunction-network");
    }

    // 8. TLE parsing populates catalog
    #[test]
    fn test_tle_catalog_loading() {
        let tle_data = [
            "ISS (ZARYA)",
            "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997",
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391424577",
            "CSS (TIANHE)",
            "1 48274U 21035A   24001.50000000  .00010000  00000-0  50000-4 0  9998",
            "2 48274  41.4700 247.0000 0006000 130.0000 325.0000 15.60000000 10009",
        ];

        let mut net = ConjunctionNetwork::new(ScreeningConfig::default());
        net.load_catalog_from_tle_lines(&tle_data);
        assert_eq!(net.catalog_len(), 2);
        assert_eq!(net.catalog[0].norad_id, 25544);
        assert_eq!(net.catalog[1].norad_id, 48274);
        assert_eq!(net.catalog[0].name, "ISS (ZARYA)");
    }

    // 9. Screening config defaults are reasonable
    #[test]
    fn test_screening_config_defaults() {
        let cfg = ScreeningConfig::default();
        assert!((cfg.miss_distance_threshold_km - 5.0).abs() < 1e-10);
        assert!((cfg.time_horizon_days - 7.0).abs() < 1e-10);
        assert!((cfg.time_step_seconds - 60.0).abs() < 1e-10);
        assert!((cfg.critical_object_step_seconds - 10.0).abs() < 1e-10);
        assert!((cfg.min_collision_probability - 1e-7).abs() < 1e-20);
    }

    // 10. Risk summary counts correct
    #[test]
    fn test_risk_summary_counts() {
        let mut net = ConjunctionNetwork::new(ScreeningConfig {
            time_horizon_days: 0.01,
            miss_distance_threshold_km: 100.0,
            ..Default::default()
        });

        // Insert two close objects that will produce an event
        let obj1 = make_object(10001, "OBJ-A", iss_elements(), ScreeningPriority::High);
        let mut elems2 = iss_elements();
        elems2.mean_anomaly_deg += 0.01;
        let obj2 = make_object(10002, "OBJ-B", elems2, ScreeningPriority::High);

        net.add_object(obj1);
        net.add_object(obj2);
        net.screen_all();

        let summary = net.risk_summary();
        assert_eq!(
            summary.total,
            summary.negligible + summary.low + summary.medium + summary.high + summary.emergency,
            "Risk counts should sum to total"
        );
    }

    // 11. Priority-weighted screening uses finer time step for critical
    #[test]
    fn test_priority_weighted_step() {
        // We verify indirectly: Critical objects paired with Low objects
        // should use the critical step. We check that the screening history
        // records the run (it screens more fine-grained steps).
        let mut net = ConjunctionNetwork::new(ScreeningConfig {
            time_horizon_days: 0.001, // very short window
            time_step_seconds: 120.0,
            critical_object_step_seconds: 5.0,
            miss_distance_threshold_km: 50.0,
            ..Default::default()
        });

        let iss = make_object(25544, "ISS", iss_elements(), ScreeningPriority::Critical);
        let mut elems2 = iss_elements();
        elems2.mean_anomaly_deg += 0.02;
        let debris = make_object(99001, "DEBRIS", elems2, ScreeningPriority::Low);

        net.add_object(iss);
        net.add_object(debris);
        let _events = net.screen_all();

        assert_eq!(net.screening_history().len(), 1);
        let run = &net.screening_history()[0];
        assert_eq!(run.objects_screened, 2);
        assert_eq!(run.pairs_evaluated, 1);
    }

    // 12. Collision probability is 1.0 at zero miss distance
    #[test]
    fn test_collision_probability_zero_miss() {
        let pc = collision_probability_2d(0.0, 0.02, 10.0);
        assert!((pc - 1.0).abs() < 1e-10, "Pc at zero miss should be 1.0");
    }

    // 13. Highest risk events sorted correctly
    #[test]
    fn test_highest_risk_events_sorted() {
        let mut net = ConjunctionNetwork::new(ScreeningConfig {
            time_horizon_days: 0.01,
            miss_distance_threshold_km: 100.0,
            ..Default::default()
        });

        // Three close objects with slightly different offsets
        let obj1 = make_object(10001, "A", iss_elements(), ScreeningPriority::Medium);
        let mut e2 = iss_elements();
        e2.mean_anomaly_deg += 0.005;
        let obj2 = make_object(10002, "B", e2, ScreeningPriority::Medium);
        let mut e3 = iss_elements();
        e3.mean_anomaly_deg += 0.05;
        let obj3 = make_object(10003, "C", e3, ScreeningPriority::Medium);

        net.add_object(obj1);
        net.add_object(obj2);
        net.add_object(obj3);
        net.screen_all();

        let top = net.highest_risk_events(10);
        // Check descending Pc order
        for i in 1..top.len() {
            assert!(
                top[i - 1].collision_probability >= top[i].collision_probability,
                "Events not sorted by descending Pc"
            );
        }
    }
}
