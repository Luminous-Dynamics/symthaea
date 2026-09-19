// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence semantics for material/compliance behavior of human-facing somatic layers.
//!
//! This module deliberately records evidence rather than production limits. It
//! distinguishes bench measurement from simulation and supplier declaration,
//! binds claims to exact layer/material/specimen identity, and keeps fatigue,
//! hysteresis, creep, and environmental context explicit.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SomaticMechanicalEvidenceOrigin {
    BenchMeasured,
    Simulation,
    SupplierDeclared,
}

impl SomaticMechanicalEvidenceOrigin {
    pub const fn is_physical_measurement(self) -> bool {
        matches!(self, Self::BenchMeasured)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StressStrainPoint {
    /// Engineering strain. Unitless and non-negative for the represented loading branch.
    pub strain: f64,
    /// Engineering stress in pascals.
    pub stress_pa: f64,
}

impl StressStrainPoint {
    fn is_valid(self) -> bool {
        self.strain.is_finite()
            && self.stress_pa.is_finite()
            && self.strain >= 0.0
            && self.stress_pa >= 0.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SomaticMechanicalResponseV1 {
    /// Optional measured/modelled compression response samples.
    pub compression: Vec<StressStrainPoint>,
    /// Optional measured/modelled shear response samples.
    pub shear: Vec<StressStrainPoint>,
    /// Energy-loss / hysteresis fraction where established, in [0, 1].
    pub hysteresis_fraction: Option<f64>,
    /// Remaining deformation fraction after a stated recovery interval, in [0, 1].
    pub permanent_set_fraction: Option<f64>,
    /// Relative creep deformation after the stated test interval, non-negative.
    pub creep_fraction: Option<f64>,
}

impl SomaticMechanicalResponseV1 {
    fn has_any_claim(&self) -> bool {
        !self.compression.is_empty()
            || !self.shear.is_empty()
            || self.hysteresis_fraction.is_some()
            || self.permanent_set_fraction.is_some()
            || self.creep_fraction.is_some()
    }

    fn is_valid(&self) -> bool {
        self.has_any_claim()
            && self.compression.iter().copied().all(StressStrainPoint::is_valid)
            && self.shear.iter().copied().all(StressStrainPoint::is_valid)
            && valid_fraction(self.hysteresis_fraction)
            && valid_fraction(self.permanent_set_fraction)
            && valid_non_negative_optional(self.creep_fraction)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SomaticMechanicalTestEnvironmentV1 {
    pub temperature_c: f64,
    /// Relative humidity as a fraction in [0, 1].
    pub relative_humidity: Option<f64>,
    /// Whether the specimen/interface was tested in a wet-contact condition.
    pub wet_condition: bool,
}

impl SomaticMechanicalTestEnvironmentV1 {
    fn is_valid(self) -> bool {
        self.temperature_c.is_finite()
            && self
                .relative_humidity
                .is_none_or(|value| value.is_finite() && (0.0..=1.0).contains(&value))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SomaticFatigueStateV1 {
    /// Number of representative loading cycles accumulated before this evidence snapshot.
    pub prior_cycles: u64,
    /// Number of cycles applied during the evidence campaign represented here.
    pub test_cycles: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SomaticMechanicalEvidenceError {
    InvalidEvidenceId,
    InvalidProfileId,
    InvalidLayerId,
    InvalidMaterialLotId,
    InvalidSpecimenId,
    InvalidMethodId,
    InvalidTimeWindow,
    InvalidEnvironment,
    MissingMechanicalClaim,
    InvalidMechanicalClaim,
    UnagedEvidenceCannotClaimPriorCycles,
}

/// One evidence snapshot for one exact somatic layer material/specimen state.
///
/// This value is evidence metadata and measured/modelled response only. It does
/// not by itself establish that the layer is qualified for human contact.
#[derive(Debug, Clone, PartialEq)]
pub struct SomaticMechanicalEvidenceV1 {
    evidence_id: String,
    profile_id: String,
    layer_id: String,
    material_lot_id: String,
    specimen_id: String,
    method_id: String,
    origin: SomaticMechanicalEvidenceOrigin,
    captured_at_ns: u64,
    valid_until_ns: u64,
    environment: SomaticMechanicalTestEnvironmentV1,
    fatigue: SomaticFatigueStateV1,
    response: SomaticMechanicalResponseV1,
}

impl SomaticMechanicalEvidenceV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        evidence_id: impl Into<String>,
        profile_id: impl Into<String>,
        layer_id: impl Into<String>,
        material_lot_id: impl Into<String>,
        specimen_id: impl Into<String>,
        method_id: impl Into<String>,
        origin: SomaticMechanicalEvidenceOrigin,
        captured_at_ns: u64,
        valid_until_ns: u64,
        environment: SomaticMechanicalTestEnvironmentV1,
        fatigue: SomaticFatigueStateV1,
        response: SomaticMechanicalResponseV1,
    ) -> Result<Self, SomaticMechanicalEvidenceError> {
        let evidence_id = normalized_id(evidence_id, SomaticMechanicalEvidenceError::InvalidEvidenceId)?;
        let profile_id = normalized_id(profile_id, SomaticMechanicalEvidenceError::InvalidProfileId)?;
        let layer_id = normalized_id(layer_id, SomaticMechanicalEvidenceError::InvalidLayerId)?;
        let material_lot_id = normalized_id(
            material_lot_id,
            SomaticMechanicalEvidenceError::InvalidMaterialLotId,
        )?;
        let specimen_id = normalized_id(specimen_id, SomaticMechanicalEvidenceError::InvalidSpecimenId)?;
        let method_id = normalized_id(method_id, SomaticMechanicalEvidenceError::InvalidMethodId)?;

        if valid_until_ns <= captured_at_ns {
            return Err(SomaticMechanicalEvidenceError::InvalidTimeWindow);
        }
        if !environment.is_valid() {
            return Err(SomaticMechanicalEvidenceError::InvalidEnvironment);
        }
        if !response.has_any_claim() {
            return Err(SomaticMechanicalEvidenceError::MissingMechanicalClaim);
        }
        if !response.is_valid() {
            return Err(SomaticMechanicalEvidenceError::InvalidMechanicalClaim);
        }
        if fatigue.prior_cycles > 0 && fatigue.test_cycles == 0 {
            return Err(SomaticMechanicalEvidenceError::UnagedEvidenceCannotClaimPriorCycles);
        }

        Ok(Self {
            evidence_id,
            profile_id,
            layer_id,
            material_lot_id,
            specimen_id,
            method_id,
            origin,
            captured_at_ns,
            valid_until_ns,
            environment,
            fatigue,
            response,
        })
    }

    pub fn evidence_id(&self) -> &str {
        &self.evidence_id
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn layer_id(&self) -> &str {
        &self.layer_id
    }

    pub fn material_lot_id(&self) -> &str {
        &self.material_lot_id
    }

    pub fn specimen_id(&self) -> &str {
        &self.specimen_id
    }

    pub fn method_id(&self) -> &str {
        &self.method_id
    }

    pub const fn origin(&self) -> SomaticMechanicalEvidenceOrigin {
        self.origin
    }

    pub const fn environment(&self) -> SomaticMechanicalTestEnvironmentV1 {
        self.environment
    }

    pub const fn fatigue(&self) -> SomaticFatigueStateV1 {
        self.fatigue
    }

    pub fn response(&self) -> &SomaticMechanicalResponseV1 {
        &self.response
    }

    pub fn is_fresh_at(&self, now_ns: u64) -> bool {
        now_ns >= self.captured_at_ns && now_ns < self.valid_until_ns
    }

    /// Convenience predicate for downstream evidence joins. A physical bench
    /// measurement still requires separate qualification; this method only keeps
    /// simulations/declarations from masquerading as measurements.
    pub fn is_fresh_physical_measurement_at(&self, now_ns: u64) -> bool {
        self.origin.is_physical_measurement() && self.is_fresh_at(now_ns)
    }
}

fn normalized_id(
    value: impl Into<String>,
    error: SomaticMechanicalEvidenceError,
) -> Result<String, SomaticMechanicalEvidenceError> {
    let value = value.into().trim().to_owned();
    if value.is_empty() || value.len() > 160 {
        Err(error)
    } else {
        Ok(value)
    }
}

fn valid_fraction(value: Option<f64>) -> bool {
    value.is_none_or(|value| value.is_finite() && (0.0..=1.0).contains(&value))
}

fn valid_non_negative_optional(value: Option<f64>) -> bool {
    value.is_none_or(|value| value.is_finite() && value >= 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response() -> SomaticMechanicalResponseV1 {
        SomaticMechanicalResponseV1 {
            compression: vec![
                StressStrainPoint {
                    strain: 0.05,
                    stress_pa: 1_000.0,
                },
                StressStrainPoint {
                    strain: 0.10,
                    stress_pa: 2_500.0,
                },
            ],
            shear: vec![],
            hysteresis_fraction: Some(0.12),
            permanent_set_fraction: Some(0.02),
            creep_fraction: Some(0.03),
        }
    }

    fn evidence(origin: SomaticMechanicalEvidenceOrigin) -> SomaticMechanicalEvidenceV1 {
        SomaticMechanicalEvidenceV1::new(
            "ev-1",
            "somatic.surface.v1",
            "compliant-layer",
            "lot-a",
            "specimen-7",
            "bench-method-a",
            origin,
            100,
            200,
            SomaticMechanicalTestEnvironmentV1 {
                temperature_c: 25.0,
                relative_humidity: Some(0.50),
                wet_condition: false,
            },
            SomaticFatigueStateV1 {
                prior_cycles: 0,
                test_cycles: 1_000,
            },
            response(),
        )
        .unwrap()
    }

    #[test]
    fn measured_and_modelled_evidence_remain_distinct() {
        let measured = evidence(SomaticMechanicalEvidenceOrigin::BenchMeasured);
        let simulated = evidence(SomaticMechanicalEvidenceOrigin::Simulation);
        assert!(measured.is_fresh_physical_measurement_at(150));
        assert!(!simulated.is_fresh_physical_measurement_at(150));
    }

    #[test]
    fn stale_measurement_is_not_current_physical_evidence() {
        let measured = evidence(SomaticMechanicalEvidenceOrigin::BenchMeasured);
        assert!(!measured.is_fresh_physical_measurement_at(200));
    }

    #[test]
    fn missing_mechanical_claim_fails_closed() {
        let result = SomaticMechanicalEvidenceV1::new(
            "ev-1",
            "profile",
            "layer",
            "lot",
            "specimen",
            "method",
            SomaticMechanicalEvidenceOrigin::BenchMeasured,
            100,
            200,
            SomaticMechanicalTestEnvironmentV1 {
                temperature_c: 25.0,
                relative_humidity: None,
                wet_condition: false,
            },
            SomaticFatigueStateV1 {
                prior_cycles: 0,
                test_cycles: 0,
            },
            SomaticMechanicalResponseV1 {
                compression: vec![],
                shear: vec![],
                hysteresis_fraction: None,
                permanent_set_fraction: None,
                creep_fraction: None,
            },
        );
        assert_eq!(
            result,
            Err(SomaticMechanicalEvidenceError::MissingMechanicalClaim)
        );
    }

    #[test]
    fn malformed_fraction_fails_closed() {
        let mut r = response();
        r.hysteresis_fraction = Some(1.1);
        let result = SomaticMechanicalEvidenceV1::new(
            "ev-1",
            "profile",
            "layer",
            "lot",
            "specimen",
            "method",
            SomaticMechanicalEvidenceOrigin::BenchMeasured,
            100,
            200,
            SomaticMechanicalTestEnvironmentV1 {
                temperature_c: 25.0,
                relative_humidity: Some(0.5),
                wet_condition: false,
            },
            SomaticFatigueStateV1 {
                prior_cycles: 0,
                test_cycles: 100,
            },
            r,
        );
        assert_eq!(
            result,
            Err(SomaticMechanicalEvidenceError::InvalidMechanicalClaim)
        );
    }

    #[test]
    fn fatigue_state_cannot_claim_prior_use_without_campaign_cycles() {
        let result = SomaticMechanicalEvidenceV1::new(
            "ev-1",
            "profile",
            "layer",
            "lot",
            "specimen",
            "method",
            SomaticMechanicalEvidenceOrigin::BenchMeasured,
            100,
            200,
            SomaticMechanicalTestEnvironmentV1 {
                temperature_c: 25.0,
                relative_humidity: None,
                wet_condition: false,
            },
            SomaticFatigueStateV1 {
                prior_cycles: 10_000,
                test_cycles: 0,
            },
            response(),
        );
        assert_eq!(
            result,
            Err(SomaticMechanicalEvidenceError::UnagedEvidenceCannotClaimPriorCycles)
        );
    }

    #[test]
    fn environment_must_be_finite_and_bounded() {
        let result = SomaticMechanicalEvidenceV1::new(
            "ev-1",
            "profile",
            "layer",
            "lot",
            "specimen",
            "method",
            SomaticMechanicalEvidenceOrigin::BenchMeasured,
            100,
            200,
            SomaticMechanicalTestEnvironmentV1 {
                temperature_c: f64::NAN,
                relative_humidity: Some(0.5),
                wet_condition: false,
            },
            SomaticFatigueStateV1 {
                prior_cycles: 0,
                test_cycles: 100,
            },
            response(),
        );
        assert_eq!(result, Err(SomaticMechanicalEvidenceError::InvalidEnvironment));
    }
}
