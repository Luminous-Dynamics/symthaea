// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reproducible external-oracle vocabulary for cislunar validation.
//!
//! This module deliberately does not fetch ephemerides. It records the source,
//! epoch, timescale, frame, units, and state vectors needed to compare an
//! implementation against checked-in or externally generated reference data.

use serde::{Deserialize, Serialize};

use crate::cislunar::{EARTH_MU_KM3_S2, MOON_MU_KM3_S2};

/// DE440-derived gravitational parameters published by JPL, km^3/s^2.
pub const DE440_EARTH_MU_KM3_S2: f64 = 398_600.435_507;
pub const DE440_MOON_MU_KM3_S2: f64 = 4_902.800_118;
pub const DE440_SUN_MU_KM3_S2: f64 = 132_712_440_041.279_42;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeScale {
    Tdb,
    Tt,
    Tai,
    Utc,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameContract {
    /// Stable frame name, e.g. `ICRF`, `earth-moon-barycentric-rotating`.
    pub name: String,
    /// Declared origin/center, e.g. `SSB`, `Earth`, `Earth-Moon barycenter`.
    pub origin: String,
    /// Human-readable axes definition or external contract reference.
    pub axes: String,
    pub time_scale: TimeScale,
    /// Explicit position unit. v0.1 reference states use `km`.
    pub position_unit: String,
    /// Explicit velocity unit. v0.1 reference states use `km/s`.
    pub velocity_unit: String,
    /// States whether translation/rotation derivative terms are included.
    pub derivative_convention: String,
    pub contract_ref: String,
}

impl FrameContract {
    pub fn is_well_formed(&self) -> bool {
        !self.name.trim().is_empty()
            && !self.origin.trim().is_empty()
            && !self.axes.trim().is_empty()
            && !self.position_unit.trim().is_empty()
            && !self.velocity_unit.trim().is_empty()
            && !self.derivative_convention.trim().is_empty()
            && !self.contract_ref.trim().is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EphemerisSource {
    /// Producer/provider, e.g. `JPL SSD`.
    pub provider: String,
    /// Product/query family, e.g. `DE440` or `Horizons`.
    pub product: String,
    /// Product/API/kernel version visible when the sample was generated.
    pub version: String,
    /// Exact query, kernel set, script, or generation configuration reference.
    pub configuration_ref: String,
}

impl EphemerisSource {
    pub fn is_well_formed(&self) -> bool {
        !self.provider.trim().is_empty()
            && !self.product.trim().is_empty()
            && !self.version.trim().is_empty()
            && !self.configuration_ref.trim().is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StateVectorKm {
    pub position_km: [f64; 3],
    pub velocity_km_s: [f64; 3],
}

impl StateVectorKm {
    pub fn is_well_formed(&self) -> bool {
        all_finite(self.position_km) && all_finite(self.velocity_km_s)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BodyStateSample {
    /// Stable target/body identifier used by the reference source.
    pub body: String,
    pub state: StateVectorKm,
}

impl BodyStateSample {
    pub fn is_well_formed(&self) -> bool {
        !self.body.trim().is_empty() && self.state.is_well_formed()
    }
}

/// One reproducible multi-body oracle snapshot.
///
/// `epoch_jd` is interpreted only through `frame.time_scale`; callers must not
/// silently reinterpret a UTC epoch as TDB or vice versa.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CislunarOracleSnapshot {
    pub snapshot_id: String,
    pub epoch_jd: f64,
    pub frame: FrameContract,
    pub source: EphemerisSource,
    pub states: Vec<BodyStateSample>,
    pub evidence_refs: Vec<String>,
}

impl CislunarOracleSnapshot {
    pub fn is_well_formed(&self) -> bool {
        if self.snapshot_id.trim().is_empty()
            || !self.epoch_jd.is_finite()
            || self.epoch_jd <= 0.0
            || !self.frame.is_well_formed()
            || !self.source.is_well_formed()
            || self.states.is_empty()
            || self.states.iter().any(|sample| !sample.is_well_formed())
            || self
                .evidence_refs
                .iter()
                .any(|reference| reference.trim().is_empty())
        {
            return false;
        }

        for (index, sample) in self.states.iter().enumerate() {
            if self.states[index + 1..]
                .iter()
                .any(|other| other.body == sample.body)
            {
                return false;
            }
        }
        true
    }

    pub fn state(&self, body: &str) -> Option<StateVectorKm> {
        self.states
            .iter()
            .find(|sample| sample.body == body)
            .map(|sample| sample.state)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StateTolerance {
    pub position_norm_km: f64,
    pub velocity_norm_km_s: f64,
}

impl StateTolerance {
    pub fn is_well_formed(&self) -> bool {
        self.position_norm_km.is_finite()
            && self.position_norm_km >= 0.0
            && self.velocity_norm_km_s.is_finite()
            && self.velocity_norm_km_s >= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StateResidual {
    pub position_norm_km: f64,
    pub velocity_norm_km_s: f64,
    pub within_tolerance: bool,
}

pub fn compare_state(
    candidate: StateVectorKm,
    reference: StateVectorKm,
    tolerance: StateTolerance,
) -> Option<StateResidual> {
    if !candidate.is_well_formed()
        || !reference.is_well_formed()
        || !tolerance.is_well_formed()
    {
        return None;
    }

    let position_norm_km = norm(sub(candidate.position_km, reference.position_km));
    let velocity_norm_km_s = norm(sub(candidate.velocity_km_s, reference.velocity_km_s));
    Some(StateResidual {
        position_norm_km,
        velocity_norm_km_s,
        within_tolerance: position_norm_km <= tolerance.position_norm_km
            && velocity_norm_km_s <= tolerance.velocity_norm_km_s,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GravitationalParameterProfile {
    pub earth_mu_km3_s2: f64,
    pub moon_mu_km3_s2: f64,
    pub sun_mu_km3_s2: Option<f64>,
}

impl GravitationalParameterProfile {
    /// Constants retained by the mean-CR3BP v0.1 regression lineage.
    pub const fn legacy_cr3bp_regression() -> Self {
        Self {
            earth_mu_km3_s2: EARTH_MU_KM3_S2,
            moon_mu_km3_s2: MOON_MU_KM3_S2,
            sun_mu_km3_s2: None,
        }
    }

    /// Current DE440-derived values published by JPL SSD.
    pub const fn jpl_de440() -> Self {
        Self {
            earth_mu_km3_s2: DE440_EARTH_MU_KM3_S2,
            moon_mu_km3_s2: DE440_MOON_MU_KM3_S2,
            sun_mu_km3_s2: Some(DE440_SUN_MU_KM3_S2),
        }
    }

    pub fn is_well_formed(&self) -> bool {
        positive_finite(self.earth_mu_km3_s2)
            && positive_finite(self.moon_mu_km3_s2)
            && self.sun_mu_km3_s2.is_none_or(positive_finite)
    }
}

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn all_finite(value: [f64; 3]) -> bool {
    value.into_iter().all(f64::is_finite)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm(value: [f64; 3]) -> f64 {
    value.into_iter().map(|component| component * component).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(time_scale: TimeScale) -> FrameContract {
        FrameContract {
            name: "ICRF".into(),
            origin: "SSB".into(),
            axes: "ICRF3/J2000-compatible inertial axes".into(),
            time_scale,
            position_unit: "km".into(),
            velocity_unit: "km/s".into(),
            derivative_convention: "inertial derivative in declared frame".into(),
            contract_ref: "jpl-de440-reference".into(),
        }
    }

    fn source() -> EphemerisSource {
        EphemerisSource {
            provider: "JPL SSD".into(),
            product: "DE440/Horizons".into(),
            version: "reference-fixture-v1".into(),
            configuration_ref: "fixtures/de440-query-001".into(),
        }
    }

    #[test]
    fn legacy_and_de440_constant_profiles_are_distinct_and_reproducible() {
        let legacy = GravitationalParameterProfile::legacy_cr3bp_regression();
        let de440 = GravitationalParameterProfile::jpl_de440();
        assert!(legacy.is_well_formed());
        assert!(de440.is_well_formed());
        assert_ne!(legacy.earth_mu_km3_s2, de440.earth_mu_km3_s2);
        assert_ne!(legacy.moon_mu_km3_s2, de440.moon_mu_km3_s2);
        assert_eq!(de440.sun_mu_km3_s2, Some(DE440_SUN_MU_KM3_S2));
    }

    #[test]
    fn snapshot_rejects_duplicate_body_ids() {
        let state = StateVectorKm {
            position_km: [1.0, 2.0, 3.0],
            velocity_km_s: [0.1, 0.2, 0.3],
        };
        let snapshot = CislunarOracleSnapshot {
            snapshot_id: "oracle-1".into(),
            epoch_jd: 2_460_000.5,
            frame: frame(TimeScale::Tdb),
            source: source(),
            states: vec![
                BodyStateSample {
                    body: "Earth".into(),
                    state,
                },
                BodyStateSample {
                    body: "Earth".into(),
                    state,
                },
            ],
            evidence_refs: vec!["jpl-query-output".into()],
        };
        assert!(!snapshot.is_well_formed());
    }

    #[test]
    fn comparison_is_norm_based_and_fail_closed() {
        let reference = StateVectorKm {
            position_km: [1.0, 2.0, 3.0],
            velocity_km_s: [0.1, 0.2, 0.3],
        };
        let candidate = StateVectorKm {
            position_km: [1.003, 2.004, 3.0],
            velocity_km_s: [0.100_3, 0.200_4, 0.3],
        };
        let residual = compare_state(
            candidate,
            reference,
            StateTolerance {
                position_norm_km: 0.006,
                velocity_norm_km_s: 0.000_6,
            },
        )
        .unwrap();
        assert!(residual.within_tolerance);

        let invalid = StateVectorKm {
            position_km: [f64::NAN, 0.0, 0.0],
            velocity_km_s: [0.0; 3],
        };
        assert!(compare_state(invalid, reference, StateTolerance {
            position_norm_km: 1.0,
            velocity_norm_km_s: 1.0,
        })
        .is_none());
    }

    #[test]
    fn timescale_is_explicit_not_inferred_from_epoch_number() {
        let tdb = frame(TimeScale::Tdb);
        let utc = frame(TimeScale::Utc);
        assert_ne!(tdb.time_scale, utc.time_scale);
    }
}
